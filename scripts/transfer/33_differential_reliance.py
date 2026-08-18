#!/usr/bin/env python3
"""Does ablating one crosscoder latent move one checkpoint's behaviour more than the other's?

**What this stage is for.** EXP-R2-210. ``32_crosscoder.py`` reports the relative
decoder norm, which is a property of the dictionary's parameters; R2.4's
admission list actually requires a causal definition -- *a latent is
model-specific if ablating it changes behaviour in one checkpoint and not the
other* -- which is measured on the models, is strictly stronger, and does not
need the decoder-norm L1 at all. It is therefore computable on the lambda = 0
dictionary that already fits, and this stage computes it.

The intervention, its control, its blind spot and its packing rule are declared
once in :mod:`src.transfer.differential_reliance`; this stage owns the
checkpoints, the cohort, the admissibility input and the artefact. It never fits
a dictionary: the dictionary is an input, and a run whose dictionary does not name
this checkpoint pair, this mode and this tensor is refused before a forward pass.

**The dictionary has to be an input, and today no stage writes one.**
``32_crosscoder.py`` writes a JSON readout and lets the fitted object fall out of
scope, so no crosscoder weights exist on disk in this repository. A causal
readout cannot be recovered from a readout -- it needs ``W_dec`` and the frozen
normalisation scales -- so ``--dictionary`` names a file written by
``differential_reliance.save_crosscoder`` and the round is blocked until one
exists. That is a two-line addition to the trainer, owned by the trainer, and it
is named here rather than worked around because inferring a decoder from a
histogram is exactly the kind of reconstruction that would produce finite numbers
about nothing.

**Two passes over the cohort, and why the first one is not optional.** The census
pass records which latents fire and in which sequences, because both the live
basis a diff is read over and the row schedule the packer needs are properties of
this cohort rather than of the checkpoint the dictionary was fitted with. The
measurement pass then re-streams the same cohort and runs the interventions. The
census costs two forward passes per batch against the measurement's thousands.

**The cost, stated before it is spent.** The unit of work is one
``(latent, sequence-row)`` cell and every cell is run four times -- two
checkpoints times measurement and matched control. At 11,691 live latents and the
128-sequence cohort that is 1.5M cells, and no packing rule reduces it, because
packing fills passes rather than removing work. ``--rows-per-latent`` is the knob
that does reduce it, by measuring each latent on a slice of the cohort rather
than on all of it, and its default is the whole cohort: the pre-registered
behaviour at the pre-registered cost. The artefact carries the arithmetic from the
run's own numbers.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.transfer import crosscoder as cc  # noqa: E402
from src.transfer import differential_reliance as dr  # noqa: E402
from src.transfer import joint_modes  # noqa: E402
from src.transfer.arms import (  # noqa: E402
    DEFAULT_CORPUS_DRAW_SEED,
    REPO,
    corpus_location,
    iter_corpus_records,
)
from src.transfer.io import sha256_file, write_json  # noqa: E402
from src.transfer.replaceable import JOINT_MODES, joint_mode_corpus  # noqa: E402


def _load_stage(filename: str) -> Any:
    """Import a stage whose module name starts with a digit.

    The same loader ``32_crosscoder.py`` declares and for the same reason: this
    stage's numbers are only readable if the cohort, the loader and the
    identical-input refusals are the *same* computation those stages perform.
    """

    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(f"_transfer_stage_{filename[:2]}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE17 = _load_stage("17_train_transcoder.py")
STAGE21 = _load_stage("21_joint_mode_qualification.py")
STAGE25 = _load_stage("25_model_diffing_baselines.py")
STAGE32 = _load_stage("32_crosscoder.py")

SCHEMA_VERSION = "r2_transfer_differential_reliance_v1"
DEFAULT_OUT = REPO / "results/transfer/differential_reliance"
INFERENCE_DTYPE = "bfloat16"

#: Arguments that name a real campaign, absent under ``--synthetic-check`` and
#: therefore excluded from the echoed settings. Two of them end in ``_per_site``
#: and would otherwise be walked by :func:`crosscoder.assert_per_layer_fields` as
#: collapsed per-site fields -- the defect that stopped ``32_crosscoder.py``
#: writing its own instrument certificate after all three fits had finished.
CAMPAIGN_ONLY_FLAGS = (
    "base", "adapted", "dictionary", "rendering", "mode", "admissible_layers",
)

PROVENANCE_MODULES = (
    "src/transfer/differential_reliance.py",
    "src/transfer/crosscoder.py",
    "src/transfer/transcoders.py",
    "src/transfer/replaceable.py",
    "src/transfer/joint_modes.py",
    "src/transfer/arms.py",
    "src/transfer/io.py",
    "scripts/transfer/17_train_transcoder.py",
    "scripts/transfer/21_joint_mode_qualification.py",
    "scripts/transfer/25_model_diffing_baselines.py",
    "scripts/transfer/32_crosscoder.py",
)


# ------------------------------------------------------------------- helpers


def chunked(records: Sequence[Any], size: int) -> Iterator[tuple[int, list[Any]]]:
    """The cohort in batches, with the global index of each batch's first row."""

    for start in range(0, len(records), size):
        chunk = list(records[start : start + size])
        if chunk:
            yield start, chunk


def render_batch(model: Any, chunk: Sequence[tuple[str, str | None]]) -> dict[str, torch.Tensor]:
    """One batch, rendered and batched by the checkpoint that will run it."""

    batch = model.batch(
        model.render([record for record, _ in chunk], ec_labels=[label for _, label in chunk])
    )
    model.forget_rendered()
    return batch


@torch.no_grad()
def paired_clean(
    base: Any,
    adapted: Any,
    chunk: Sequence[tuple[str, str | None]],
    *,
    sites: Sequence[int],
) -> tuple[dict[str, torch.Tensor], dr.CleanPass, dr.CleanPass]:
    """The identical-input guarantee, then both checkpoints' unperturbed passes.

    ``25_model_diffing_baselines.assert_identical_batches`` renders and batches
    each side independently and compares the renderings, every field of the batch
    tensor and the content masks, raising rather than proceeding on any mismatch.
    Having proved the two batches identical, **one** of them is then handed to
    both forward passes: a shared tensor cannot drift where two rebuilt ones
    could, and the check above is what licenses sharing it.
    """

    STAGE25.assert_identical_batches(base, adapted, chunk)
    if base.device != adapted.device:
        raise RuntimeError(
            f"the two checkpoints are on {base.device} and {adapted.device}; this "
            "readout feeds one batch tensor to both and cannot straddle devices"
        )
    batch = render_batch(base, chunk)
    render_batch(adapted, chunk)
    return batch, dr.clean_pass(base, batch, sites=sites), dr.clean_pass(adapted, batch, sites=sites)


@torch.no_grad()
def site_latents(
    dictionary: cc.Crosscoder,
    clean_base: dr.CleanPass,
    clean_adapted: dr.CleanPass,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(latents, rows, positions)`` at the positions this readout is defined on.

    The crosscoder's latent is a function of **both** checkpoints' activations at
    the same position -- that is its defining property -- so it is read from the
    clean paired capture once and then held fixed while each checkpoint is
    perturbed. It cannot be recomputed inside an ablated pass: the dictionary is
    not part of either model, and a latent recomputed from a perturbed activation
    would be a different feature.
    """

    if not torch.equal(clean_base.scored, clean_adapted.scored):
        raise RuntimeError(
            "the two checkpoints scored different positions of one batch, so no "
            "position of one corresponds to a position of the other"
        )
    rows, positions = clean_base.scored.nonzero(as_tuple=True)
    sites = dictionary.config.sites
    stacked_base = torch.stack(
        [clean_base.block_outputs[int(site)][rows, positions] for site in sites]
    ).float()
    stacked_adapted = torch.stack(
        [clean_adapted.block_outputs[int(site)][rows, positions] for site in sites]
    ).float()
    latents, _, _ = dictionary.encode(stacked_base, stacked_adapted)
    return latents, rows, positions


def assign_rows(hits: torch.Tensor, *, rows_per_latent: int) -> torch.Tensor:
    """Which cohort rows each latent is measured on, spread across the cohort.

    ``hits`` is ``(cohort_rows, d_hidden)`` from the census. Rows are taken at an
    even stride through the ones a latent actually fires in, rather than the
    first ones: an even stride puts a latent's assigned rows in **different
    batches**, which is what lets a pass carry one row from each of many latents
    and is therefore what makes the packing dense. Taking a prefix instead would
    put them all in the first batch, where they would occupy the whole pass.

    ``rows_per_latent`` of zero keeps every row a latent fires in, which is the
    pre-registered behaviour and the default.
    """

    if rows_per_latent <= 0:
        return hits.clone()
    assigned = torch.zeros_like(hits)
    for latent in range(hits.shape[1]):
        available = hits[:, latent].nonzero(as_tuple=True)[0]
        if available.numel() == 0:
            continue
        if available.numel() <= rows_per_latent:
            assigned[available, latent] = True
            continue
        # floor(i * n / r) is strictly increasing for r <= n, so this takes
        # exactly `rows_per_latent` DISTINCT rows. A rounded linspace does not:
        # it collapses to duplicates as r approaches n and would silently
        # measure a latent on fewer rows than the run declared.
        stride = (
            torch.arange(rows_per_latent, dtype=torch.long) * available.numel()
        ) // rows_per_latent
        assigned[available[stride], latent] = True
    return assigned


def restrict(
    supports: dict[int, dr.LatentSupport], assigned: torch.Tensor, *, row_offset: int
) -> list[dr.LatentSupport]:
    """Keep only the firings that fall in a latent's assigned cohort rows."""

    kept: list[dr.LatentSupport] = []
    for latent in sorted(supports):
        support = supports[latent]
        mask = assigned[support.rows + int(row_offset), latent]
        if not bool(mask.any()):
            continue
        kept.append(
            dr.LatentSupport(
                latent=latent,
                rows=support.rows[mask],
                positions=support.positions[mask],
                coefficients=support.coefficients[mask],
            )
        )
    return kept


# ------------------------------------------------------------- synthetic mode


def run_synthetic_check(args: argparse.Namespace) -> dict[str, Any]:
    """The whole instrument on a paired backbone whose differential reliance is known.

    Two backbones sharing every weight and differing only in the gain of a channel
    that reads one direction of the residual stream at the intercepted site into
    one token's logit. Ablating that direction lowers the adapted model's readout
    logit by exactly ``gain * coefficient`` and the base model's not at all, so the
    differential reliance of the latent whose decoder is that direction has a
    closed form: ``delta + log(p * exp(-delta) + 1 - p)`` with ``p`` the readout
    token's clean probability. A latent along any direction orthogonal to it is the
    equal-reliance case and must read as no difference at the control's scale.

    Reported beside the packing check, which is the correctness crux: a packed
    pass must be **bitwise** identical to ablating each of its latents in its own
    forward pass, or it is a different experiment rather than an optimisation.
    """

    site = int(args.synthetic_site)
    fire = int(args.synthetic_fire_position)
    n_latents = int(args.synthetic_latents)
    generator = torch.Generator().manual_seed(int(args.seed))
    injected = torch.randn(args.synthetic_d_model, generator=generator)
    injected = injected / injected.norm()
    base = dr.SyntheticPairedBackbone(
        vocab=args.synthetic_vocab,
        d_model=args.synthetic_d_model,
        n_layers=args.synthetic_layers,
        seed=int(args.seed),
        reliance_site=site,
        reliance_gain=0.0,
        reliance_direction=injected,
        readout_token=int(args.synthetic_readout_token),
    )
    batch = dr.synthetic_batch(
        rows=n_latents,
        width=int(args.synthetic_width),
        vocab=int(args.synthetic_vocab),
        seed=int(args.seed) + 1,
    )
    # The token the injected channel predicts is the true next token wherever the
    # latents fire, so the channel is behaviourally load-bearing rather than an
    # unused direction whose ablation nothing could notice.
    batch["input_ids"][:, fire + 1] = int(args.synthetic_readout_token)
    # The channel is centred on its own clean value at the firing position, which
    # leaves the ablation's effect exactly gain * coefficient and keeps the clean
    # readout probability off saturation. Uncentred, the injected logit swamps the
    # vocabulary and the ground truth degenerates to a null.
    reference = float(base.reliance_channel(batch["input_ids"])[:, fire].mean())
    adapted = base.paired_with(
        reliance_gain=float(args.synthetic_gain), reliance_reference=reference
    )

    directions = [injected]
    for _ in range(n_latents - 1):
        draw = torch.randn(args.synthetic_d_model, generator=generator)
        orthogonal = draw - (draw @ injected) * injected
        directions.append(orthogonal / orthogonal.norm())
    supports = dr.synthetic_supports(
        rows_per_latent=[[index] for index in range(n_latents)],
        positions_per_latent=[[fire] for _ in range(n_latents)],
        coefficient=float(args.synthetic_coefficient),
    )
    clean_base = dr.clean_pass(base, batch, sites=[site])
    clean_adapted = dr.clean_pass(adapted, batch, sites=[site])

    by_index = {index: directions[index] for index in range(n_latents)}
    effect_base = dr.measure_pack(
        base, batch, clean_base, site=site, pack=supports, directions=by_index
    )
    effect_adapted = dr.measure_pack(
        adapted, batch, clean_adapted, site=site, pack=supports, directions=by_index
    )
    stacked = torch.stack([torch.stack(directions), torch.stack(directions)])
    control = dr.matched_random_directions(
        stacked, seed=int(args.control_seed), site=site, latents=list(range(n_latents))
    )
    control_base = dr.measure_pack(
        base, batch, clean_base, site=site, pack=supports,
        directions={index: control[0, index] for index in range(n_latents)},
    )
    control_adapted = dr.measure_pack(
        adapted, batch, clean_adapted, site=site, pack=supports,
        directions={index: control[1, index] for index in range(n_latents)},
    )

    measured = [
        float((effect_adapted[index] - effect_base[index]).mean()) for index in range(n_latents)
    ]
    controlled = [
        float((control_adapted[index] - control_base[index]).mean()) for index in range(n_latents)
    ]
    delta = float(args.synthetic_gain) * float(args.synthetic_coefficient)
    probability = float(torch.exp(-clean_adapted.nll[0, fire]))
    predicted = delta + math.log(probability * math.exp(-delta) + 1.0 - probability)

    equal = torch.tensor(measured[1:])
    random_arm = torch.tensor(controlled)

    packed = dr.measure_pack(
        adapted, batch, clean_adapted, site=site, pack=supports, directions=by_index
    )
    individually = {}
    for support in supports:
        individually.update(
            dr.measure_pack(
                adapted, batch, clean_adapted, site=site, pack=[support], directions=by_index
            )
        )
    bitwise = all(
        torch.equal(packed[index], individually[index]) for index in range(n_latents)
    )
    repeated = dr.measure_pack(
        adapted, batch, clean_adapted, site=site, pack=supports, directions=by_index
    )
    deterministic = all(
        torch.equal(packed[index], repeated[index]) for index in range(n_latents)
    )
    twin = base.paired_with(reliance_gain=0.0, reliance_reference=reference)
    clean_twin = dr.clean_pass(twin, batch, sites=[site])
    twin_effect = dr.measure_pack(
        twin, batch, clean_twin, site=site, pack=supports, directions=by_index
    )
    identical_zero = all(
        torch.equal(effect_base[index], twin_effect[index]) for index in range(n_latents)
    )

    return {
        "kind": "synthetic_instrument_check",
        "estimand": dr.ESTIMAND,
        "ground_truth": {
            "construction": run_synthetic_check.__doc__,
            "injected_gain": float(args.synthetic_gain),
            "ablation_coefficient": float(args.synthetic_coefficient),
            "logit_drop": delta,
            "channel_reference": reference,
            "readout_token_clean_probability": probability,
            "predicted_differential_reliance": predicted,
            "measured_differential_reliance": measured[0],
            "absolute_error_nats": abs(measured[0] - predicted),
        },
        "equal_reliance": {
            "n_latents": int(equal.numel()),
            "construction": (
                "directions orthogonal to the injected channel, so ablating them "
                "moves the channel by exactly zero and the two checkpoints -- which "
                "share every weight -- are relied upon equally by construction"
            ),
            "max_absolute_differential_reliance": float(equal.abs().max()),
            "mean_differential_reliance": float(equal.mean()),
            "standard_deviation": float(equal.std()),
        },
        "matched_random_control": {
            "rule": dr.CONTROL_RULE,
            "mean": float(random_arm.mean()),
            "standard_deviation": float(random_arm.std()),
            "max_absolute": float(random_arm.abs().max()),
            "injected_effect_in_control_standard_deviations": (
                measured[0] / float(random_arm.std()) if float(random_arm.std()) else None
            ),
        },
        "packing": {
            "rule": dr.PACKING_RULE,
            "n_latents": n_latents,
            "n_passes": len(dr.pack_disjoint_supports(supports)),
            "bitwise_identical_to_single_latent_passes": bool(bitwise),
        },
        "determinism": {
            "repeated_pass_bitwise_identical": bool(deterministic),
            "identical_checkpoints_read_exactly_zero": bool(identical_zero),
        },
        # Single-site by construction, so the per-site discipline is weak here and
        # is applied anyway: a guard that runs on one artefact kind and not the
        # other is a guard a reader cannot rely on.
        "site_per_site": [
            {
                "layer": site,
                "admissible": True,
                "n_latents": n_latents,
                "differential_reliance_per_latent": measured,
                "control_differential_reliance_per_latent": controlled,
            }
        ],
        "live_latents_per_site": [n_latents],
        "measured_latents_per_site": [n_latents],
        "passes_per_site": [len(dr.pack_disjoint_supports(supports))],
        # One row each by construction, which is the fully packable case and the
        # opposite of what the real cohort gives; carried so the synthetic
        # artefact is held to the same sizing discipline as a campaign one.
        "mean_cohort_rows_per_live_latent_per_site": [1.0],
        "limitations": {
            "differential_reliance_not_possession": dr.RELIANCE_BLIND_SPOT,
            "synthetic_scope": (
                "this check certifies the intervention, the packing rule, the "
                "matched control and the statistic. It does NOT exercise the "
                "Crosscoder encoder: the latent coefficients here are declared "
                "rather than read from a fitted dictionary, and the path from "
                "activations through encode() to a support is covered by "
                "tests/test_differential_reliance.py and by the end-to-end run"
            ),
            "control_spread_is_inflated_at_this_width": (
                "the control's spread here exceeds the equal-reliance latents', "
                "and that is a property of the toy width rather than of the "
                "control. A random direction overlaps the injected one by about "
                "1/sqrt(d_model), so at d_model 32 it recovers roughly a fifth of "
                "the injected channel while the orthogonal latents recover exactly "
                "none. At the campaign's d_model 4096 the same overlap is 1/64 and "
                "the control sits with the equal-reliance latents, which is where "
                "a floor belongs"
            ),
        },
    }


# ------------------------------------------------------------------- the CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="pre-adaptation checkpoint")
    parser.add_argument("--adapted", help="adapted checkpoint")
    parser.add_argument(
        "--dictionary",
        type=Path,
        help="a fitted Crosscoder written by differential_reliance.save_crosscoder",
    )
    parser.add_argument("--rendering", choices=joint_modes.RENDERING_NAMES)
    parser.add_argument("--mode", choices=JOINT_MODES)
    parser.add_argument(
        "--admissible-layers",
        help="where a diff may be REPORTED; a subset of the dictionary's fitted "
        "sites, required, and never inferred here",
    )
    parser.add_argument("--tensor", default="block_output", choices=STAGE25.TENSORS)
    parser.add_argument(
        "--rows-per-latent",
        type=int,
        default=0,
        help="cohort sequences each latent is measured on; 0 keeps every sequence "
        "it fires in, which is the pre-registered behaviour and its full cost",
    )
    parser.add_argument("--live-threshold", type=int, default=1)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="sequence rows per forward pass. Free to choose: it sizes the "
        "measurement, not the cohort",
    )
    parser.add_argument(
        "--fit-batch-size",
        type=int,
        default=None,
        help="the batch size the DICTIONARY was fitted at. Required, and separate "
        "from --batch-size, because the held-out offset is steps x fit batch size: "
        "re-deriving the cohort at this stage's own batch size would draw a "
        "different population under the same name",
    )
    parser.add_argument("--eval-sequences", type=int, default=128)
    parser.add_argument(
        "--steps",
        type=int,
        default=20000,
        help="the dictionary's training step budget, so the held-out cohort is "
        "drawn past everything that budget reached -- it must match the fit",
    )
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--control-seed", type=int, default=20260818)
    parser.add_argument("--corpus-seed", type=int, default=DEFAULT_CORPUS_DRAW_SEED)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--protein-context", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--allow-self-pair", action="store_true")
    parser.add_argument("--synthetic-check", action="store_true")
    parser.add_argument("--synthetic-d-model", type=int, default=32)
    parser.add_argument("--synthetic-vocab", type=int, default=64)
    parser.add_argument("--synthetic-layers", type=int, default=3)
    parser.add_argument("--synthetic-site", type=int, default=1)
    parser.add_argument("--synthetic-width", type=int, default=24)
    parser.add_argument("--synthetic-latents", type=int, default=16)
    parser.add_argument("--synthetic-fire-position", type=int, default=8)
    parser.add_argument("--synthetic-readout-token", type=int, default=5)
    parser.add_argument("--synthetic-gain", type=float, default=2.0)
    parser.add_argument("--synthetic-coefficient", type=float, default=1.0)
    return parser


def resolve(args: argparse.Namespace) -> None:
    """Refuse a half-declared run before anything loads."""

    if args.synthetic_check:
        named = [flag for flag in CAMPAIGN_ONLY_FLAGS if getattr(args, flag) is not None]
        if named:
            raise ValueError(
                f"--synthetic-check names no checkpoint pair, so {named} describe "
                "nothing about this run and must be absent"
            )
        return
    missing = [flag for flag in CAMPAIGN_ONLY_FLAGS if getattr(args, flag) is None]
    if missing:
        raise ValueError(
            f"a checkpoint-pair run requires {missing}. --admissible-layers is "
            "required rather than defaulted because it depends on per-layer "
            "measurements this stage does not make"
        )
    if args.fit_batch_size is None:
        raise ValueError(
            "--fit-batch-size is required: the held-out offset is steps x the "
            "batch size the DICTIONARY was fitted at, so defaulting it to this "
            "stage's own --batch-size would silently draw a different cohort"
        )
    args.admissible_layers = STAGE32.parse_layers(args.admissible_layers)


def main() -> None:
    args = build_parser().parse_args()
    resolve(args)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.synthetic_check:
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "settings": {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in vars(args).items()
                if key not in CAMPAIGN_ONLY_FLAGS
            },
            "provenance": {
                "runner": {
                    "path": "scripts/transfer/33_differential_reliance.py",
                    "sha256": sha256_file(Path(__file__).resolve()),
                },
                "modules": {
                    name: sha256_file(REPO_ROOT / name) for name in PROVENANCE_MODULES
                },
            },
            **run_synthetic_check(args),
        }
        cc.assert_per_layer_fields(payload, n_sites=1)
        dr.assert_required_per_site_fields(payload)
        destination = args.out / "differential_reliance__synthetic_check.json"
        write_json(destination, payload)
        print(f"wrote {destination}")
        return

    declaration = joint_modes.rendering(args.rendering)
    source = joint_mode_corpus(args.mode)
    corpus = corpus_location(source)
    print(f"[paths] base    {Path(args.base).resolve()}")
    print(f"[paths] adapted {Path(args.adapted).resolve()}")
    print(f"[paths] dict    {args.dictionary.resolve()}")
    print(f"[paths] corpus  {corpus}  ({source}, mode {args.mode})")

    base_path, base_tokenizer = STAGE21.load_tokenizer(Path(args.base))
    adapted_path, adapted_tokenizer = STAGE21.load_tokenizer(Path(args.adapted))
    vocabulary = STAGE25.assert_identical_tokenizers(base_tokenizer, adapted_tokenizer)
    _, base_facts, base_model = STAGE25.load_side(
        base_path, base_tokenizer, declaration=declaration, args=args
    )
    tokenisation, adapted_facts, adapted_model = STAGE25.load_side(
        adapted_path, adapted_tokenizer, declaration=declaration, args=args
    )
    shape = STAGE25.assert_comparable_shape(
        base_model, adapted_model, reference_facts=base_facts, target_facts=adapted_facts
    )
    declared_tensor = STAGE25.tensor_declaration(base_model, args.tensor)

    base_digest = base_model.weights_digest()
    adapted_digest = adapted_model.weights_digest()
    self_pair = base_digest == adapted_digest
    if self_pair and not args.allow_self_pair:
        raise ValueError(
            "both roles resolve to the same weights, so every differential "
            "reliance is zero by construction. That is a legitimate end-to-end "
            "check of this path and is not a diff; pass --allow-self-pair"
        )
    pair_digest = (
        base_digest if self_pair else cc.pair_backbone_digest(base_digest, adapted_digest)
    )

    dictionary, manifest = dr.load_crosscoder(args.dictionary, device=args.device)
    fit_cohort = {
        "steps": int(args.steps),
        "fit_batch_size": int(args.fit_batch_size),
        "eval_sequences": int(args.eval_sequences),
        "corpus_seed": int(args.corpus_seed),
        "max_tokens": int(args.max_tokens),
    }
    dr.assert_dictionary_matches(
        manifest,
        backbone_pair_sha256=pair_digest,
        mode=args.mode,
        tensor=args.tensor,
        d_model=int(shape["d_model"]),
        n_layers=int(shape["n_layers"]),
        cohort=fit_cohort,
    )
    sites = list(dictionary.config.sites)
    admissible = cc.assert_admissible_subset(args.admissible_layers, sites)
    print(
        f"[dict] {manifest['sha256'][:12]}.. sites {sites}, d_hidden "
        f"{dictionary.config.d_hidden}, k {dictionary.config.k}, lambda "
        f"{dictionary.config.decoder_norm_penalty}, admissible {list(admissible)}"
    )

    low, high = STAGE17.CORPUS_BAND[source]
    if tokenisation is not None:
        low, high = STAGE17.joint_protein_band(
            tokenisation, max_tokens=args.max_tokens, protein_context=args.protein_context
        )

    def records() -> Iterator[tuple[str, str | None]]:
        return iter_corpus_records(source, min_symbols=low, max_symbols=high)

    symbol_unit = "characters" if source == "openwebtext" else "residues"
    # The DICTIONARY's batch size, not this stage's: the offset is
    # steps x batch_size, so the cohort is a property of the fit and re-deriving
    # it at the measurement's own batch size would be a different population
    # under the same name.
    held_out_records, screen, held_out_offset = STAGE17.held_out_cohort(
        records,
        corpus_seed=args.corpus_seed,
        steps=args.steps,
        batch_size=args.fit_batch_size,
        eval_sequences=args.eval_sequences,
        symbol_unit=symbol_unit,
    )
    cohort = list(held_out_records)
    print(f"[cohort] {len(cohort)} held-out records past a skip of {held_out_offset}")

    d_hidden = dictionary.config.d_hidden
    counts = torch.zeros(len(sites), d_hidden, dtype=torch.long)
    hits = torch.zeros(len(sites), len(cohort), d_hidden, dtype=torch.bool)

    print("[census] one clean paired pass over the cohort")
    for offset, chunk in chunked(cohort, args.batch_size):
        _, clean_base, clean_adapted = paired_clean(
            base_model, adapted_model, chunk, sites=sites
        )
        latents, rows, _ = site_latents(dictionary, clean_base, clean_adapted)
        active = latents > 0
        counts += active.sum(dim=1).cpu()
        for index in range(len(sites)):
            per_row = torch.zeros(
                len(chunk), d_hidden, dtype=torch.uint8, device=active.device
            )
            per_row.scatter_reduce_(
                0,
                rows.unsqueeze(1).expand(-1, d_hidden),
                active[index].to(torch.uint8),
                reduce="amax",
            )
            hits[index, offset : offset + len(chunk)] = per_row.bool().cpu()
        del latents, active, clean_base, clean_adapted

    live = counts >= int(args.live_threshold)
    live_per_site = [int(value) for value in live.sum(dim=1)]
    assigned = torch.stack(
        [assign_rows(hits[index], rows_per_latent=args.rows_per_latent) & live[index][None, :]
         for index in range(len(sites))]
    )
    print(f"[census] live per site {live_per_site}")
    projected = dr.packed_cost(
        live_latents=max(
            1,
            sum(
                int(live[index].sum())
                for index in range(len(sites))
                if int(sites[index]) in admissible
            ),
        ),
        cohort_rows=len(cohort),
        rows_per_latent=(args.rows_per_latent if args.rows_per_latent > 0 else len(cohort)),
        batch_rows=int(args.batch_size),
    )
    # Printed before the measurement rather than only written after it: the
    # difference between --rows-per-latent 8 and the full cohort is a factor of
    # 32 here, and an operator should see which one this run committed to while
    # there is still time to stop it.
    print(
        f"[cost] {projected['packed_forward_passes']:,} forward passes projected "
        f"({projected['naive_forward_passes']:,} at the full cohort), up to "
        f"{projected['latents_per_pass_ceiling']} latents per pass"
    )

    arms = {"ablation": {}, "control": {}}
    accumulators: dict[tuple[str, str, int], dr.RelianceAccumulator] = {}
    directions: dict[int, dict[str, torch.Tensor]] = {}
    measured_latents: list[list[int]] = []
    for index, site in enumerate(sites):
        latent_index = [int(value) for value in live[index].nonzero(as_tuple=True)[0]]
        measured_latents.append(latent_index)
        decoder = dr.ablation_directions(dictionary, site=int(site))[:, latent_index]
        directions[int(site)] = {
            "ablation": decoder,
            "control": dr.matched_random_directions(
                decoder, seed=int(args.control_seed), site=int(site), latents=latent_index
            ),
        }
        for arm in arms:
            for role in cc.ROLES:
                accumulators[(arm, role, int(site))] = dr.RelianceAccumulator(latent_index)

    passes_per_site = [0 for _ in sites]
    members_per_site = [0 for _ in sites]
    print("[measure] streaming the cohort once more, with interventions")
    for offset, chunk in chunked(cohort, args.batch_size):
        batch, clean_base, clean_adapted = paired_clean(
            base_model, adapted_model, chunk, sites=sites
        )
        latents, rows, positions = site_latents(dictionary, clean_base, clean_adapted)
        for index, site in enumerate(sites):
            if int(site) not in admissible:
                continue
            supports = dr.latent_supports(
                latents[index], rows=rows, positions=positions, keep=measured_latents[index]
            )
            here = restrict(supports, assigned[index], row_offset=offset)
            if not here:
                continue
            packs = dr.pack_disjoint_supports(here)
            passes_per_site[index] += len(packs)
            members_per_site[index] += len(here)
            column = {
                latent: position
                for position, latent in enumerate(measured_latents[index])
            }
            for arm in arms:
                per_role = {
                    "base": (base_model, clean_base, directions[int(site)][arm][0]),
                    "adapted": (adapted_model, clean_adapted, directions[int(site)][arm][1]),
                }
                for role, (model, clean, matrix) in per_role.items():
                    for pack in packs:
                        members = [here[member] for member in pack]
                        effects = dr.measure_pack(
                            model,
                            batch,
                            clean,
                            site=int(site),
                            pack=members,
                            directions={
                                support.latent: matrix[column[support.latent]]
                                for support in members
                            },
                        )
                        accumulators[(arm, role, int(site))].update(effects)
        del latents, clean_base, clean_adapted
        print(
            f"[measure] rows {offset}..{offset + len(chunk) - 1}  passes so far "
            f"{sum(passes_per_site)}",
            flush=True,
        )

    site_records: list[dict[str, Any]] = []
    for index, site in enumerate(sites):
        if int(site) not in admissible:
            site_records.append(
                {"layer": int(site), "admissible": False, "verdict": "ADMISSIBILITY_REFUSED"}
            )
            continue
        latent_index = measured_latents[index]
        ablation_base = accumulators[("ablation", "base", int(site))]
        ablation_adapted = accumulators[("ablation", "adapted", int(site))]
        control_base = accumulators[("control", "base", int(site))]
        control_adapted = accumulators[("control", "adapted", int(site))]
        reliance = dr.differential_reliance(ablation_base.mean(), ablation_adapted.mean())
        control = dr.differential_reliance(control_base.mean(), control_adapted.mean())
        finite = torch.isfinite(reliance)
        site_records.append(
            {
                "layer": int(site),
                "admissible": True,
                "n_live": int(live_per_site[index]),
                "n_measured": int(finite.sum()),
                "latent_index": latent_index,
                "n_scored_positions_per_latent": [
                    int(value) for value in ablation_base.count
                ],
                "delta_nll_base_per_latent": _finite_list(ablation_base.mean()),
                "delta_nll_adapted_per_latent": _finite_list(ablation_adapted.mean()),
                "differential_reliance_per_latent": _finite_list(reliance),
                "control_delta_nll_base_per_latent": _finite_list(control_base.mean()),
                "control_delta_nll_adapted_per_latent": _finite_list(control_adapted.mean()),
                "control_differential_reliance_per_latent": _finite_list(control),
                "standard_error_base_per_latent": _finite_list(ablation_base.standard_error()),
                "standard_error_adapted_per_latent": _finite_list(
                    ablation_adapted.standard_error()
                ),
                # A summary over LATENTS at one site, which is what a site record
                # is for. Nothing here is a summary over sites: the per-site
                # vectors beside it are the artefact and are never reduced.
                "summary": _summary(reliance[finite], control[finite]),
            }
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "checkpoint_pair",
        "settings": {
            key: (str(value) if isinstance(value, Path) else list(value)
                  if isinstance(value, tuple) else value)
            for key, value in vars(args).items()
        },
        "provenance": {
            "runner": {
                "path": "scripts/transfer/33_differential_reliance.py",
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "modules": {name: sha256_file(REPO_ROOT / name) for name in PROVENANCE_MODULES},
        },
        "estimand": dr.ESTIMAND,
        "base": STAGE25.checkpoint_record(
            base_path, Path(args.base), base_facts, base_model, role="base"
        ),
        "adapted": STAGE25.checkpoint_record(
            adapted_path, Path(args.adapted), adapted_facts, adapted_model, role="adapted"
        ),
        "tokenizer_vocabulary": vocabulary,
        "comparability": shape,
        "tensor": declared_tensor,
        "self_pair": self_pair,
        "backbone_pair_sha256": pair_digest,
        "dictionary": manifest,
        "identical_input_guarantee": (
            "every batch is rendered and batched independently by each checkpoint "
            "and the two results are compared -- the rendered strings, every field "
            "of the batch tensor and the content mask -- by "
            "25_model_diffing_baselines.assert_identical_batches, which raises "
            "rather than proceeding on any mismatch. Having proved them identical, "
            "ONE batch tensor is handed to both forward passes"
        ),
        "intervention": {
            "form": (
                "ADDITIVE and single-latent: f_i * scale_m * W_dec[m, l, i] is "
                "SUBTRACTED from layer l's own output at the positions latent i "
                "fires on, and the model is otherwise intact. The crosscoder's "
                "reconstruction is never spliced in as a replacement model"
            ),
            "why_not_replacement": (
                "R2.3 measured behavioural recovery on this joint checkpoint as "
                "NEGATIVE in all four cells, and negative in text on the base "
                "checkpoint too under the same recipe, corpus and seed. Recovery "
                "is (ablated - replacement)/(ablated - clean), so negative means "
                "the spliced model is worse than mean-ablating the entire block. "
                "There is no dynamic range inside that in which to read one latent "
                "of thousands, and the additive form needs only the decoder "
                "direction to be meaningful"
            ),
            "latents_are_read_from_the_clean_paired_pass": (
                "a crosscoder latent is a function of BOTH checkpoints' "
                "activations at the same position, so it is computed once from the "
                "clean paired capture and held fixed while each checkpoint is "
                "perturbed. It is not recomputed inside an ablated pass, where it "
                "would be a different feature"
            ),
            "scored_positions": (
                "the effect is read at the latent's own firing positions, on the "
                "next-token prediction each of those positions makes. The "
                "intervention is applied at exactly the positions the effect is "
                "read at, so no unmeasured perturbation enters the pass"
            ),
        },
        "control": {"rule": dr.CONTROL_RULE, "seed": int(args.control_seed)},
        # Everything a successor campaign needs to size an ablation round on this
        # lineage, measured here rather than assumed there. This block is a SIZING
        # INPUT, not a diagnostic: `occupancy` is the quantity that decides whether
        # packing packs at all, and reading it as a curiosity is how the 16-47
        # minute estimate happened.
        "sizing": {
            "how_to_size_a_successor_campaign": (
                "the unit of work is one (latent, sequence-row) CELL and there are "
                "live x rows_per_latent x 2 checkpoints x 2 arms of them; forward "
                "passes are ceil(cells / batch_rows). PACKING DOES NOT REDUCE "
                "CELLS -- it only stops a pass leaving rows idle -- so the ONLY "
                "lever that removes work is rows_per_latent, and the saving "
                "against the full cohort is exactly cohort_rows / rows_per_latent. "
                "The number of different latents a pass can carry is "
                "min(batch_rows, cohort_rows / rows_per_latent), because latents "
                "sharing a sequence cannot share a pass. Check that against "
                "mean_cohort_rows_per_live_latent_per_site below: a latent that "
                "fires somewhere in a large fraction of the cohort shares a row "
                "with almost every other latent, so at rows_per_latent = "
                "cohort_rows the colouring degenerates to ONE latent per pass and "
                "there is no packing saving at any batch size"
            ),
            "why_more_positions_per_latent_buy_little": (
                "the per-latent mean has a standard error going as "
                "1/sqrt(positions), but the DETECTION THRESHOLD is the "
                "across-latent spread of the matched control, and that has two "
                "parts. Sampling noise shrinks with positions. A SYSTEMATIC "
                "per-direction component does not: a given direction of a given "
                "norm genuinely moves the two checkpoints by different amounts, "
                "and averaging more positions of the SAME direction cannot reduce "
                "a property of the direction. Measured on a paired backbone whose "
                "difference is real (d_model 512, 256 latents): the null "
                "population's spread is FLAT at 0.0278 nats across a 15x range of "
                "positions, 6 to 92; fitting sd^2 = a/n + b to the control gives a "
                "systematic floor of 0.046 nats against a sampling term of 0.010 "
                "at 46 positions; and the threshold moves 0.0476 -> 0.0458 nats "
                "from 46 to 92 positions, a factor of 1.04 where 1/sqrt(n) alone "
                "predicts 1.41. A graded sweep of true effects from 0.02 to 0.50 "
                "nats returns IDENTICAL detection verdicts at 46 and at 92. "
                "Replicated on a real 3B llama over openwebtext: the threshold "
                "falls 0.0234 -> 0.0197 between 27 and 52 positions, then "
                "0.0197 -> 0.0195 to 97 positions, then does not move at all to "
                "183 -- flat to four decimal places across a 3.5x range where "
                "sampling alone predicts 1.87x. Both curves fall as 1/sqrt(n) while "
                "sampling noise dominates and flatten once it no longer does, "
                "which is what identifies the two components rather than assuming "
                "them; the flattening sets in around fifty positions per latent. "
                "So rows_per_latent buys resolution only up to that point, and "
                "spending past it buys almost nothing"
            ),
            "cohort_rows": len(cohort),
            "live_latents_per_site": live_per_site,
            # Mean number of cohort sequences a live latent fires somewhere in.
            # Divide cohort_rows by this to get the largest number of latents that
            # could ever share a pass at the full cohort.
            "mean_cohort_rows_per_live_latent_per_site": [
                (
                    float(hits[index][:, live[index]].double().sum(dim=0).mean())
                    if int(live[index].sum())
                    else 0.0
                )
                for index in range(len(sites))
            ],
            "cohort_row_occupancy_fraction_per_site": [
                (
                    float(hits[index][:, live[index]].double().sum(dim=0).mean())
                    / max(1, len(cohort))
                    if int(live[index].sum())
                    else 0.0
                )
                for index in range(len(sites))
            ],
            # What the colouring actually achieved here, against the ceiling the
            # cost model predicts. The two differ whenever a latent takes more
            # than one row of a batch.
            "realised_latents_per_pass_per_site": [
                (members_per_site[index] / passes_per_site[index])
                if passes_per_site[index]
                else 0.0
                for index in range(len(sites))
            ],
        },
        "packing": {
            "rule": dr.PACKING_RULE,
            "passes_per_site": passes_per_site,
            "cohort_rows": len(cohort),
            # Sized on the latents this run actually measures -- the live basis at
            # the ADMISSIBLE sites -- rather than on the whole fitted dictionary,
            # and reported as an ideal beside the passes the run really scheduled:
            # a pack is short whenever a latent occupies more than one row of a
            # batch, which the ideal does not model.
            "cost": dr.packed_cost(
                live_latents=max(
                    1,
                    sum(
                        len(entry)
                        for index, entry in enumerate(measured_latents)
                        if int(sites[index]) in admissible
                    ),
                ),
                cohort_rows=len(cohort),
                rows_per_latent=(
                    args.rows_per_latent if args.rows_per_latent > 0 else len(cohort)
                ),
                batch_rows=int(args.batch_size),
            ),
            "realised_forward_passes": sum(passes_per_site) * 4,
            "realised_note": (
                "four forward passes per pack -- two checkpoints times ablation and "
                "matched control -- so this is the number the round actually cost"
            ),
        },
        "admissibility": {
            "fitted_layers": sites,
            "admissible_layers": list(admissible),
            "inadmissible_layers": [int(s) for s in sites if int(s) not in admissible],
            "rule": (
                "a diff may be reported at layer l only where both cells' "
                "dictionaries carry at least that layer's own r99 in live latents "
                "and that r99 is itself non-degenerate. The set is an INPUT to this "
                "stage and is never inferred here; at an inadmissible fitted site "
                "the record carries ADMISSIBILITY_REFUSED in place of numbers"
            ),
            "site_independence": cc.SITE_INDEPENDENCE_NOTE,
        },
        "cohort": {
            "corpus": str(corpus),
            "corpus_source": source,
            "symbol_band": [low, high],
            "symbol_unit": symbol_unit,
            "input_rendering": base_model.rendering_note,
            "scored_positions": base_model.scoring_note,
            "held_out_offset": held_out_offset,
            "near_duplicate_screen": screen,
            "n_rows": len(cohort),
            "measurement_batch_size": int(args.batch_size),
            "rows_per_latent": int(args.rows_per_latent),
            "fit_parameters": fit_cohort,
            "draw": (
                "17_train_transcoder.held_out_cohort at the DICTIONARY's step "
                "budget and batch size, carried in fit_parameters. The offset is "
                "steps x batch size, so those two are properties of the fit and "
                "not of this run; measurement_batch_size sizes the forward passes "
                "and is deliberately free of them. Where the dictionary records "
                "its own cohort parameters they are checked against these and a "
                "disagreement is a refusal"
            ),
        },
        "site_per_site": site_records,
        "live_latents_per_site": live_per_site,
        "measured_latents_per_site": [len(entry) for entry in measured_latents],
        "passes_per_site": passes_per_site,
        "limitations": {
            "differential_reliance_not_possession": dr.RELIANCE_BLIND_SPOT,
            "shuffled_null_is_retired_here": (
                "the shuffled-pairing null that qualifies 32_crosscoder.py's "
                "representational readout is NOT used and is not applicable: it "
                "controls pairing correspondence, not the size of an intervention, "
                "and it is a ceiling rather than the floor an effect size must be "
                "read against"
            ),
            "text_only_on_this_lineage": (
                "a causal DIFFERENCE needs a behavioural quantity on both sides. "
                "The pre-adaptation checkpoint's protein mode is behaviourally "
                "unmeasurable on this lineage -- context information +0.0843 "
                "nats/token, reversal cost -0.0013 nats/residue (EXP-R2-152) -- so "
                "in protein this statistic is UNDEFINED rather than expensive. "
                "Nothing in this stage prevents a protein run; the artefact would "
                "be a one-sided measurement wearing a difference's clothes"
            ),
            "one_lineage_one_draw": (
                "one checkpoint pair, one mode, one corpus draw at one seed, one "
                "fitted dictionary"
            ),
            "precision": (
                f"both checkpoints run at {INFERENCE_DTYPE} and the dictionary is "
                "float32; the ablation vector is cast to the activation dtype at "
                "the point it is subtracted, identically in both roles"
            ),
            "positions_are_not_independent": (
                "the standard error carried per latent is a dispersion over that "
                "latent's own firing positions, which sit inside a small number of "
                "sequences and are not independent. It separates a latent measured "
                "on nine positions from one measured on nine hundred; it is not a "
                "confidence interval"
            ),
        },
    }
    if self_pair:
        payload["limitations"]["self_pair"] = (
            "both roles are the same checkpoint, so every differential reliance is "
            "zero up to the two decoders' own difference; this run is an end-to-end "
            "check of the path and carries no diff"
        )

    cc.assert_per_layer_fields(payload, n_sites=len(sites))
    dr.assert_required_per_site_fields(payload)
    destination = args.out / (
        "differential_reliance__"
        + re.sub(r"[^A-Za-z0-9._-]+", "-", base_path.name)
        + "__to__"
        + re.sub(r"[^A-Za-z0-9._-]+", "-", adapted_path.name)
        + f"__{args.mode}__{args.tensor}.json"
    )
    write_json(destination, payload)
    print()
    for record in site_records:
        if not record["admissible"]:
            print(f"[L{record['layer']:2d}] ADMISSIBILITY_REFUSED")
            continue
        summary = record["summary"]
        print(
            f"[L{record['layer']:2d}] measured {record['n_measured']:6d}  "
            f"median D {summary['median']:+.4f}  control median "
            f"{summary['control_median']:+.4f}  |D| above control p95 "
            f"{summary['fraction_above_control_p95']:.3f}"
        )
    print(f"wrote {destination}")


def _finite_list(values: torch.Tensor) -> list[float | None]:
    """A per-latent vector as JSON, with ``nan`` written as ``null``.

    ``src.transfer.io.write_json`` refuses non-finite numbers, and rightly: a bare
    ``NaN`` is not JSON and a reader that accepts one plots it. A latent that was
    never measured is ``null`` rather than zero, because "no difference" and "no
    measurement" are the two answers this readout must never confuse.
    """

    return [None if not math.isfinite(float(value)) else float(value) for value in values]


def _summary(reliance: torch.Tensor, control: torch.Tensor) -> dict[str, Any]:
    """One site's distribution of differential reliance, read against its own control."""

    if reliance.numel() == 0:
        return {"n": 0}
    threshold = float(torch.quantile(control.abs(), 0.95)) if control.numel() else float("nan")
    return {
        "n": int(reliance.numel()),
        "median": float(reliance.median()),
        "mean": float(reliance.mean()),
        "standard_deviation": float(reliance.std()) if reliance.numel() > 1 else 0.0,
        "control_median": float(control.median()) if control.numel() else None,
        "control_standard_deviation": (
            float(control.std()) if control.numel() > 1 else 0.0
        ),
        "control_absolute_p95": threshold,
        "fraction_above_control_p95": (
            float((reliance.abs() > threshold).double().mean())
            if math.isfinite(threshold)
            else None
        ),
        "reading": (
            "a raw differential effect is unreadable without the matched control's "
            "scale (L17). fraction_above_control_p95 is the share of latents whose "
            "differential reliance exceeds the 95th percentile of the control's "
            "own absolute differential effect; under no signal it is 0.05"
        ),
    }


if __name__ == "__main__":
    main()
