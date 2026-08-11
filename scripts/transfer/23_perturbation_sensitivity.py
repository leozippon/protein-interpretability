#!/usr/bin/env python3
"""Is a joint model's PROTEIN mode more fragile than its TEXT mode, at matched relative damage?

**The question, and why one checkpoint answers it and two cannot.** On the
architecture-matched standalone pair -- ``gpt2-large`` and ``protgpt2``, identical
depth, width, heads, vocabulary size and parameter count -- two dictionaries
reconstruct to nearly the same relative error (mean per-layer NMSE 0.2768 and
0.2555) and yet substituting a random perturbation of each dictionary's own norm
leaves the text arm at +0.9283 recovery and the protein arm at +0.3286. At matched
relative reconstruction error the same-sized MLP perturbation costs the protein
model roughly six times more of its behaviour. That pair controls architecture but
not weights and not training. **One joint checkpoint controls all three at once**:
the same weights process both modalities, so a difference between its two modes
cannot be a difference of architecture, scale, or parameters.

**The estimand is 15_replacement_faithfulness.py's, deliberately.** The fraction
of the clean-to-mean-ablated cross-entropy gap that survives a perturbation --
``(mean_ablated - perturbed) / (mean_ablated - clean)`` -- measured on the same
kind of splice, over the same kind of content positions, and with the same
fully-ablated endpoint. That is what makes a number here commensurable with every
replacement and neuron-basis number this programme has already measured, and it is
why the two floors and the denominator are reported in absolute nats beside every
ratio (standing rule 27, and §4b's requirement to report the numerator as well as
the ratio).

**The perturbation, and where epsilon is anchored.** At every position of every
replaced block, a uniformly random direction ``r`` is drawn and scaled so that
``||r|| = epsilon * ||y||``, where ``y`` is *that block's output at that position
in the perturbed forward pass*. The block writes ``y + r``. Three consequences,
each load-bearing:

* the manipulation is **sequential** in the sense the replacement is -- layer
  ``t`` is perturbed on top of what layers ``< t`` already did -- because ``y`` is
  read from the pass being perturbed rather than from a stored clean pass;
* epsilon is **relative to the model's own block output norm**, measured per arm,
  per mode, per layer and per position, so a fixed epsilon is a genuinely matched
  manipulation across two modes whose activation scales differ. The measured mean
  block-output norm per layer is recorded so the reader can convert epsilon into
  absolute units for either mode;
* the ratio is **not bounded below by zero**. Mean ablation removes the block's
  information; a perturbation removes it *and* injects norm the residual stream
  never carried, which also damages the attention and embedding pathways. At large
  epsilon the recovery therefore falls through the mean-ablated floor and goes
  negative. This is a property of the manipulation, not a defect, and the artefact
  says so rather than leaving a reader to assume a ``[0, 1]`` range.

Degenerate positions raise. A block output of zero norm has no scale for epsilon
to be relative to, and returning a zero perturbation there would report an
unperturbed position as a perturbed one.

**Arms, and why both kinds are required.** ``--checkpoint`` plus ``--rendering``
plus ``--modes`` measures a joint checkpoint (the ProLLaMA lineage: ``Llama-2-7b-hf``,
``ProLLaMA_Stage_1``, ``ProLLaMA``, rendering family ``prollama``), loading and
scoring it through ``21_joint_mode_qualification.py``'s own machinery so that the
scored span is the one that stage qualified. ``--arm`` measures a panel arm --
``gpt2-large`` and ``protgpt2`` above all -- through the identical code path.
Without the panel arms a joint difference could not be attributed between joint
training, modality, tokenizer and lineage; without the joint checkpoint the
standalone pair cannot separate modality from weights.

**The protein scale ladder, and what the perturbation target is on it.** ``--arm``
also reaches all four rungs of ``src.transfer.arms.PROTEIN_SCALE_LADDER`` --
``progen2-small`` (151M), ``progen2-medium`` (764M), ``progen2-large`` (2.78B) and
``progen2-xlarge`` (6.44B): one lineage, one 31-token residue tokenizer, one
UniRef90+BFD30 mixture, so a tolerance curve across them isolates **scale** with
everything else this stage depends on held fixed. Two of the four are staged
non-members of the panel (``src.transfer.arms.STAGED_ARMS``) and are reached here
rather than admitted to it, because a panel arm carries campaign obligations --
above all the ``budget`` family, whose ``arm_power`` reads ``config.vocab_size``,
a key ``progen2-large`` declares as 51200 against a 31-token tokenizer and
``progen2-xlarge`` does not declare at all -- that a tolerance measurement does
not need and those two checkpoints cannot meet. Nothing on this stage's path
reads that key: the alphabet never enters, and the expansion beside every
magnitude is measured from the tokenizer and the rendering.

ProGen2's block is GPT-J-style **parallel**: attention and feed-forward read the
same ``ln_1`` and both sum into the residual, so there is no sequential "block
output" of the kind the GPT-2 arms' interception rests on. **The declared
perturbation target is the feed-forward output** -- the analogue of the
MLP-output tensor perturbed on every other arm, the same object before anything
is added to it -- and the identity that certifies it,
``block output == (attention output + intercepted feed-forward output) + ln_1
input``, is verified **exactly** against the live forward pass by
``src.transfer.replaceable.DenseReplaceable.estimand_identity`` before the arm is
measured, the way ``residual + mlp_out == block_out`` is verified on the serial
arms. An architecture with no such declaration is refused rather than
duck-typed, and the per-arm declaration reaches the artefact as
``perturbation_target``, so a reader of two arms' curves can see which tensor
each one perturbed. The attention contribution is not perturbed, on either
layout.

**Two comparability requirements, both enforced in the artefact.**

*Raw nats are not comparable across modes.* ProLLaMA writes protein at about 1.54
residues per token over the unmodified LLaMA-2 vocabulary and text at its own rate,
so a nats-per-token magnitude from one mode is not the other's unit (Appendix B
rule 26, limitation L23). The cross-mode comparison is made on the **dimensionless
recovery fraction**, each mode normalised by its own clean-to-mean-ablated gap;
each mode's gap in nats and its measured symbols per token are recorded beside it,
and ``cross_mode.magnitude_comparison`` is marked NOT_LICENSED in its own field,
the way ``21_joint_mode_qualification.py`` marks its protein magnitudes.

*Epsilon is matched by construction.* Because it is relative to the measured block
output norm, the same epsilon is the same fractional perturbation in both modes
even though their absolute activation scales differ.

**Reuse.** There is one splice in this repository --
:meth:`src.transfer.replaceable.ReplaceableModel.block_intercept` -- and this stage
adds no second forward path. The fully-ablated endpoint comes from stage 15's
``mean_ablation_context``, the per-sequence sweep and the paired recovery record
from stage 22, and the joint checkpoint's loader and rendering resolution from
stage 21, all imported rather than restated. The only new computation is the
perturbation itself and the block-output norm epsilon is anchored on.

An external baseline, not a registered panel stage: a joint checkpoint is reached
by path, so it cannot be scheduled through ``panel_contract.STAGE_CONTRACTS``.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# The stage directory itself, so `panel_contract` imports under every invocation
# rather than only when the caller happens to run from scripts/transfer.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from panel_contract import CAMPAIGN_PANEL  # noqa: E402
from src.transfer import joint_modes  # noqa: E402
from src.transfer.arms import (  # noqa: E402
    DEFAULT_CORPUS_DRAW_SEED,
    REPO,
    Cohort,
    protein_cohort,
    symbols_per_token as arm_symbols_per_token,
    text_cohort,
)
from src.transfer.io import sha256_file, write_json  # noqa: E402
from src.transfer.replaceable import (  # noqa: E402
    JOINT_MODES,
    DenseReplaceable,
    JointReplaceable,
    ReplaceableModel,
    arm_evaluation_cohort_source,
    checkpoint_weights_digest,
    joint_mode_corpus,
    load_replaceable,
    perturbable_arms,
)
from src.transfer.statistics import bootstrap_unit_floor, mean_interval  # noqa: E402


def _load_stage(filename: str) -> Any:
    """Import a stage whose module name starts with a digit.

    Three of them are imported here rather than copied, because this stage's
    numbers are only worth anything if they are the *same* computation those
    stages perform: the same fully-ablated endpoint, the same per-sequence sweep
    and paired recovery record, the same joint checkpoint loader and scored span.
    Appendix B rule 12 -- a single declaration, imported, never reimplemented --
    does not stop being true because the declaration lives in a file whose name
    starts with a digit.
    """

    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(f"_transfer_stage_{filename[:2]}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE15 = _load_stage("15_replacement_faithfulness.py")
STAGE21 = _load_stage("21_joint_mode_qualification.py")
STAGE22 = _load_stage("22_neuron_basis_circuit.py")

SCHEMA_VERSION = "r2_transfer_perturbation_sensitivity_v2"
DEFAULT_OUT = REPO / "results/transfer/perturbation_sensitivity"

#: The arms ``--arm`` offers, composed once so that the names argparse advertises
#: and the names :func:`~src.transfer.replaceable.load_replaceable` accepts are
#: one list rather than two that can drift.
ADMISSIBLE_ARMS = perturbable_arms(CAMPAIGN_PANEL)

#: Modules and stages whose content decides these numbers, hashed into the
#: artefact. The splice module is first because it is the one that decides what a
#: block output IS on each arm, and the rendering module second because it has
#: been worth 2.9 nats/token when wrong.
PROVENANCE_MODULES = (
    "src/transfer/replaceable.py",
    "src/transfer/joint_modes.py",
    "src/transfer/arms.py",
    "src/transfer/statistics.py",
    "scripts/transfer/15_replacement_faithfulness.py",
    "scripts/transfer/21_joint_mode_qualification.py",
    "scripts/transfer/22_neuron_basis_circuit.py",
)

#: Relative perturbation magnitudes swept by default. Doubling from 0.05 to 0.8
#: spans the range in which the standalone pair's matched-perturbation control
#: separated them: the dictionary errors that produced the +0.9283/+0.3286 split
#: sit at a per-layer NMSE around 0.26, which is a relative error of about 0.5 in
#: norm, so the grid brackets it on both sides rather than sampling one point.
DEFAULT_EPSILONS = (0.05, 0.1, 0.2, 0.4, 0.8)

#: Independent random directions per epsilon. Three, and reported individually:
#: one draw of a random control is a point estimate of a distribution, and this
#: repository admits no single-draw point estimate.
DEFAULT_DRAWS_PER_EPSILON = 3

#: How far the epsilon = 0 point may sit from the clean cross-entropy, in nats per
#: token. A zero-magnitude perturbation adds exactly the zero vector, so the
#: intervention is an identity and the difference must be zero; anything
#: measurable means the splice path is not a no-op and the whole sweep is shifted.
IDENTITY_TOLERANCE_NATS = 1e-6

#: What a reader may and may not do with two modes of one checkpoint.
CROSS_MODE_NOTE = (
    "the two modes tokenise differently -- ProLLaMA writes protein at about 1.54 "
    "residues per token over the unmodified LLaMA-2 vocabulary and text at its own "
    "rate -- so a magnitude in nats per token from one mode is not the other's "
    "unit (Appendix B rule 26, limitation L23). The licensed cross-mode comparison "
    "is the dimensionless recovery fraction, each mode normalised by its OWN "
    "clean-to-mean-ablated gap; that gap is recorded in nats for each mode below, "
    "beside each mode's measured symbols per scored token, so a reader can see "
    "exactly what is and is not comparable. Epsilon itself is comparable by "
    "construction: it is relative to each mode's own measured block output norm"
)


# --------------------------------------------------------------- perturbation


def relative_perturbation(
    block_output: torch.Tensor, epsilon: float, generator: torch.Generator
) -> torch.Tensor:
    """A uniformly random direction of norm ``epsilon * ||y||``, per position.

    ``y`` is the block's output at that position **in the pass being perturbed**,
    which is what makes epsilon a relative magnitude rather than an absolute one:
    activation scale varies by an order of magnitude across depth and differs
    between a text and a protein mode, so a fixed absolute perturbation would be a
    different manipulation at every layer and in every mode. Anchoring on ``||y||``
    per position removes that freedom entirely.

    The direction is drawn in the orthant-free sense -- an isotropic Gaussian,
    normalised -- so nothing about it is aligned with the block output, the
    residual stream, or any learned subspace. That is the point: the control asks
    what a perturbation of this *size* costs, with its direction carrying no
    information.

    Degenerate positions raise rather than returning a zero perturbation. With
    ``||y|| = 0`` there is no scale for epsilon to be relative to, and a "matched"
    perturbation of zero would report an unperturbed position as a perturbed one.
    """

    if epsilon < 0.0:
        raise ValueError("a relative perturbation magnitude cannot be negative")
    reference = block_output.float()
    norm = reference.norm(dim=-1, keepdim=True)
    degenerate = int((norm == 0).sum())
    if degenerate:
        raise ValueError(
            f"the relative perturbation is undefined at {degenerate} positions whose "
            "block output has zero norm: epsilon is a fraction of that norm, so there "
            "is nothing there for it to be a fraction of, and returning a zero "
            "perturbation would report an unperturbed position as a perturbed one"
        )
    noise = torch.randn(
        reference.shape,
        generator=generator,
        device=reference.device,
        dtype=reference.dtype,
    )
    noise_norm = noise.norm(dim=-1, keepdim=True)
    if int((noise_norm == 0).sum()):
        raise RuntimeError(
            "a Gaussian draw collapsed to the zero vector, so no random direction is "
            "available at that position"
        )
    return (float(epsilon) * norm * (noise / noise_norm)).to(block_output.dtype)


def perturbation_context(
    model: ReplaceableModel, *, epsilon: float, seed: int
) -> Callable[[], Any]:
    """The replacement's splice, with a relative random perturbation in its place.

    It runs through :meth:`src.transfer.replaceable.ReplaceableModel.block_intercept`
    -- the same primitive stage 15's ``replacement_context``,
    ``mean_ablation_context`` and ``matched_perturbation_context`` use -- so there
    is one splice implementation in this repository and not a second forward path
    that could drift from it.

    Sequential in exactly the sense the replacement is: the perturbation is formed
    against the block output of *this* pass, so layer ``t`` is perturbed on top of
    the perturbation layers ``< t`` already introduced.

    The generator is created once per draw rather than per batch, so the draws are
    independent of each other and reproducible from ``seed`` and the batch order.
    """

    generator = torch.Generator(device=model.device)
    generator.manual_seed(int(seed))

    def factory() -> Any:
        def perturb(layer: int, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return y + relative_perturbation(y, epsilon, generator)

        return model.block_intercept(perturb)

    return factory


@torch.no_grad()
def block_output_reference(
    model: ReplaceableModel, inputs: list[str], *, batch_size: int
) -> dict[str, Any]:
    """One clean sweep: the per-layer mean block output, and its mean norm.

    The mean is the fully-ablated endpoint every recovery ratio is divided by, and
    it is taken over the cohort's **content** positions -- padding, delimiters and
    any conditioning prompt excluded -- which is the convention stage 15's endpoint
    is defined under, so the two stages' denominators are the same object. This is
    ``15_replacement_faithfulness.py``'s ``clean_pass`` without the reconstruction
    half, which needs a trained dictionary this stage does not have.

    The mean norm is what epsilon is anchored on, recorded per layer so that a
    relative magnitude in the sweep below can be read in absolute units. It is a
    description of the manipulation, never an input to it: the perturbation is
    scaled by each position's own norm in the perturbed pass, not by this average.
    """

    n_layers, width = model.n_layers, model.width
    total = torch.zeros(n_layers, width, dtype=torch.float64)
    norm_total = torch.zeros(n_layers, dtype=torch.float64)
    counted = torch.zeros(n_layers, dtype=torch.float64)
    scored: dict[str, torch.Tensor] = {}

    def tap(layer: int, x: torch.Tensor, y: torch.Tensor) -> None:
        keep = scored["mask"]
        flat = y.reshape(-1, y.shape[-1])[keep].float()
        total[layer] += flat.sum(0).double().cpu()
        norm_total[layer] += float(flat.norm(dim=-1).sum())
        counted[layer] += float(keep.sum())
        return None

    for start in range(0, len(inputs), batch_size):
        batch = model.batch(inputs[start : start + batch_size])
        scored["mask"] = model.content_mask(batch).reshape(-1)
        with model.block_intercept(tap):
            model.run(batch)

    if not counted.gt(0).all():
        raise RuntimeError(
            "the cohort supplied no content positions to average over, so the "
            "fully-ablated endpoint every ratio divides by is undefined"
        )
    return {
        "block_output_mean": (total / counted[:, None]).float(),
        "mean_block_output_norm_per_layer": (norm_total / counted).tolist(),
        "n_content_positions_per_layer": counted.tolist(),
    }


# ------------------------------------------------------------------ reporting


def across_draws(draws: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    """Every draw's value for one quantity, then its spread.

    The values come first and the summary second because that is the reading order
    this repository admits: three draws of a random perturbation are three numbers,
    and a mean of three that hides a factor-of-two spread between them is a point
    estimate with the evidence for its own instability removed.
    """

    values = [record.get(key) for record in draws]
    if any(value is None for value in values):
        return None
    numbers = [float(value) for value in values]
    return {
        "values": numbers,
        "mean": float(np.mean(numbers)),
        "min": float(np.min(numbers)),
        "max": float(np.max(numbers)),
    }


def endpoints_record(
    clean: np.ndarray, ablated: np.ndarray, identity: np.ndarray
) -> dict[str, Any]:
    """Both ends of the ratio in absolute nats, and the one check on them.

    Standing rule 27: a recovery ratio whose denominator is not published is not a
    measurement, because the same 0.9 can come from a 0.02-nat gap or a 2-nat one.
    Both endpoints are therefore reported as nats per scored token with their
    intervals, and every ratio in the sweep is derived from them.

    ``identity`` is the epsilon = 0 sweep point. A zero-magnitude perturbation adds
    the zero vector at every position, so it must reproduce the clean
    cross-entropy exactly; anything measurable means the splice path is not a no-op
    when it should be, and every point of the sweep is shifted with it.
    """

    denominator = float(ablated.mean() - clean.mean())
    identity_gap = float(abs(identity.mean() - clean.mean()))
    return {
        "clean_nats_per_token": float(clean.mean()),
        "clean_interval": mean_interval(clean.tolist()),
        "mean_ablated_nats_per_token": float(ablated.mean()),
        "mean_ablated_interval": mean_interval(ablated.tolist()),
        "denominator_nats_per_token": denominator,
        "denominator_definition": "mean_ablated - clean, over the same cohort sequences",
        "fully_ablated_endpoint": (
            "every replaced block's output substituted by its per-layer mean over "
            "this cohort's content positions, which is "
            "15_replacement_faithfulness.py's endpoint"
        ),
        "zero_epsilon_minus_clean_nats": identity_gap,
        "identity_tolerance_nats": IDENTITY_TOLERANCE_NATS,
        "verdict": (
            "PASS"
            if denominator > 0 and identity_gap <= IDENTITY_TOLERANCE_NATS
            else "FAIL"
        ),
    }


def symbols_per_token(model: ReplaceableModel, inputs: list[str]) -> dict[str, Any]:
    """The arm's measured expansion, so every nats-per-token figure can be re-read.

    Dispatched on the implementation rather than duck-typed, because the two
    conventions genuinely differ and neither can be inferred from the other: a
    panel arm resolves through :func:`src.transfer.arms.symbols_per_token`, the
    repository's one declaration of the quantity, and a joint checkpoint through
    the rendering that located its scored span.
    """

    if isinstance(model, DenseReplaceable):
        value = arm_symbols_per_token(model.arm, list(inputs), model.max_tokens)
        unit = "residues" if model.cohort_kind == "protein" else "characters"
        basis = (
            "src.transfer.arms.symbols_per_token over the rendered inputs truncated "
            "to --max-tokens"
        )
    elif isinstance(model, JointReplaceable):
        value = model.symbols_per_token(list(inputs))
        unit = "residues" if model.mode == "protein" else "characters"
        basis = (
            "residues per scored token read off the declared rendering's own records"
            if model.mode == "protein"
            else "characters per token over the window truncated to --max-tokens"
        )
    else:
        raise TypeError(
            f"{type(model).__name__} declares no symbols-per-token convention, and "
            "this stage reports the measured expansion beside every magnitude "
            "(standing rule 27, limitation L23). The arms it is designed for are the "
            "architecture-matched standalone pair and a joint checkpoint reached by "
            "--checkpoint"
        )
    return {
        "value": float(value),
        "unit": f"{unit} per scored token",
        "basis": basis,
        "note": (
            "divide any nats-per-token figure in this mode by this to read it per "
            "content symbol; the two units are not interchangeable across modes or "
            "arms (Appendix B rule 26, L23)"
        ),
    }


def cross_mode_record(modes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """What may be compared between two modes of one checkpoint, and what may not."""

    per_mode = {
        name: {
            "denominator_nats_per_token": record["gates"]["endpoints"][
                "denominator_nats_per_token"
            ],
            "clean_nats_per_token": record["gates"]["endpoints"]["clean_nats_per_token"],
            "mean_ablated_nats_per_token": record["gates"]["endpoints"][
                "mean_ablated_nats_per_token"
            ],
            "symbols_per_token": record["symbols_per_token"]["value"],
            "symbols_per_token_unit": record["symbols_per_token"]["unit"],
            "recovery_by_epsilon": {
                str(point["epsilon"]): point["across_draws"]["recovery"]
                for point in record["sweep"]
                if point["across_draws"]["recovery"] is not None
            },
        }
        for name, record in modes.items()
    }
    return {
        "modes_measured": sorted(modes),
        "licensed_comparison": (
            "the dimensionless recovery fraction at a common epsilon, each mode "
            "normalised by its own clean-to-mean-ablated gap"
        ),
        "magnitude_comparison": {
            "verdict": "NOT_LICENSED",
            "reason": CROSS_MODE_NOTE,
        },
        "epsilon_comparability": (
            "epsilon IS comparable across modes: it is defined relative to the block "
            "output norm, which is measured per arm, per mode, per layer and per "
            "position, so a fixed epsilon is the same fractional manipulation in both"
        ),
        "per_mode": per_mode,
        "readable_only_with_two_modes": len(modes) > 1,
    }


# ---------------------------------------------------------------------- cohort


def build_cohort(args: argparse.Namespace, *, source: str, label: str) -> Cohort:
    """The cohort one mode is scored on, drawn from the corpus that mode declares.

    One dispatch on the declared corpus, so that the population is the one
    ``15_replacement_faithfulness.py`` and ``22_neuron_basis_circuit.py`` score the
    same arm on and the three stages' ratios describe the same sequences.
    """

    if source == "openwebtext":
        return text_cohort(
            args.sequences,
            args.text_min_chars,
            skip=args.cohort_skip,
            name=label,
            seed=args.cohort_draw_seed or None,
        )
    if source in ("swissprot", "zymctrl_ec"):
        return protein_cohort(
            args.sequences,
            args.protein_min_len,
            args.protein_max_len,
            skip=args.cohort_skip,
            name=label,
            with_ec=source == "zymctrl_ec",
            seed=args.cohort_draw_seed or None,
        )
    raise ValueError(f"this stage cannot build a cohort from {source!r}")


def cohort_record(cohort: Cohort, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "name": cohort.name,
        "kind": cohort.kind,
        "digest": cohort.digest,
        "provenance_digest": cohort.provenance_digest,
        "sampling": cohort.sampling,
        "n_sequences": len(cohort),
        "residue_band": [args.protein_min_len, args.protein_max_len],
        "text_min_chars": args.text_min_chars,
        "band_note": (
            "the residue band applies to a protein cohort and the character floor to "
            "a text one; which is in force is decided by kind above"
        ),
    }


# ----------------------------------------------------------------- the sweep


def measure_mode(
    args: argparse.Namespace,
    model: ReplaceableModel,
    *,
    source: str,
    label: str,
) -> dict[str, Any]:
    """One arm in one mode: both endpoints, then the epsilon sweep."""

    print(f"[{label}] loader self-check")
    loader = model.self_check()

    print(f"[{label}] drawing the cohort")
    cohort = build_cohort(args, source=source, label=f"{label}_perturbation")
    inputs = model.render(cohort.records, ec_labels=cohort.metadata.get("ec_labels"))

    print(f"[{label}] clean sweep: block-output mean, its norm, and the clean endpoint")
    reference = block_output_reference(model, inputs, batch_size=args.batch_size)
    clean = STAGE22.scored_cross_entropy(model, inputs, batch_size=args.batch_size)
    ablated = STAGE22.scored_cross_entropy(
        model,
        inputs,
        batch_size=args.batch_size,
        factory=STAGE15.mean_ablation_context(model, reference["block_output_mean"]),
    )
    picks = STAGE22.bootstrap_indices(
        len(clean), replicates=args.bootstrap, seed=args.seed
    )

    grid = [0.0, *(float(value) for value in args.epsilons)]
    print(f"[{label}] sweeping {len(grid)} magnitudes x {args.draws} draws")
    sweep: list[dict[str, Any]] = []
    identity: np.ndarray | None = None
    counter = 0
    for epsilon in grid:
        draws: list[dict[str, Any]] = []
        # A zero-magnitude perturbation is the zero vector whatever direction was
        # drawn, so several draws of it would be one number reported three times
        # -- and would read as a measured spread across directions that was never
        # measured. It is the identity anchor, and one draw is all of it there is.
        requested = 1 if epsilon == 0.0 else args.draws
        for _ in range(requested):
            counter += 1
            seed = args.seed + counter
            perturbed = STAGE22.scored_cross_entropy(
                model,
                inputs,
                batch_size=args.batch_size,
                factory=perturbation_context(model, epsilon=epsilon, seed=seed),
            )
            if epsilon == 0.0 and identity is None:
                identity = perturbed
            draws.append(
                {
                    "seed": int(seed),
                    **STAGE22.recovery_record(clean, perturbed, ablated, picks=picks),
                }
            )
        sweep.append(
            {
                "epsilon": float(epsilon),
                "n_draws": len(draws),
                "draws": draws,
                "across_draws": {
                    key: across_draws(draws, key)
                    for key in (
                        "recovery",
                        "damage_nats_per_token",
                        "cross_entropy_nats_per_token",
                    )
                },
                **(
                    {
                        "draws_note": (
                            "one draw, because a zero-magnitude perturbation is the "
                            "zero vector in every direction: this point is the "
                            "identity anchor, not a sample of a distribution"
                        )
                    }
                    if epsilon == 0.0
                    else {}
                ),
            }
        )
        summary = sweep[-1]["across_draws"]["recovery"]
        print(
            f"  epsilon {epsilon:<5} damage "
            f"{sweep[-1]['across_draws']['damage_nats_per_token']['mean']:+.4f} nats  "
            f"recovery {None if summary is None else round(summary['mean'], 4)}"
        )

    assert identity is not None  # epsilon = 0 is always the first point of the grid
    endpoints = endpoints_record(clean, ablated, identity)
    print(
        f"  clean {endpoints['clean_nats_per_token']:.4f} -> mean-ablated "
        f"{endpoints['mean_ablated_nats_per_token']:.4f} (denominator "
        f"{endpoints['denominator_nats_per_token']:.4f}); identity gap "
        f"{endpoints['zero_epsilon_minus_clean_nats']:.3e}  {endpoints['verdict']}"
    )

    record: dict[str, Any] = {
        "cohort": cohort_record(cohort, args),
        "scoring_note": model.scoring_note,
        # Which tensor was perturbed, in the model's own declaration rather than
        # this stage's prose. "The block output" names one tensor on a serial
        # residual block and a different one on a parallel block, and a reader of
        # two arms' curves has no other way to tell which each one perturbed.
        "perturbation_target": model.perturbation_target,
        "symbols_per_token": symbols_per_token(model, inputs),
        "gates": {"loader": loader, "endpoints": endpoints},
        "block_output_norm": {
            "mean_per_layer": reference["mean_block_output_norm_per_layer"],
            "n_content_positions_per_layer": reference["n_content_positions_per_layer"],
            "note": (
                "measured over this mode's content positions, and recorded so a "
                "relative epsilon can be read in absolute units. The perturbation is "
                "scaled by each position's own norm in the perturbed pass, never by "
                "this average"
            ),
        },
        "resampling": {
            "unit": "cohort sequence, one index set shared by every epsilon and draw",
            "bootstrap_replicates": int(args.bootstrap),
            **bootstrap_unit_floor(len(clean)),
        },
        "sweep": sweep,
    }
    if isinstance(model, JointReplaceable) and model.tokenisation is not None:
        record["rendering"] = model.tokenisation.facts()
    elif isinstance(model, JointReplaceable):
        record["rendering"] = {
            "verdict": "NOT_RESOLVED",
            "declared_family": model.declaration.name,
            "reason": (
                "the text mode's scored positions are the tokenizer's own next-token "
                "targets and do not depend on the protein rendering, so the declared "
                "family is recorded but not resolved against this tokenizer. A "
                "protein-mode run resolves it and is refused when it does not hold"
            ),
        }
    return record


# --------------------------------------------------------------------- driver


def resolve_protein_rendering(
    tokenizer: Any, name: str
) -> joint_modes.JointTokenisation:
    """The declared rendering resolved against this checkpoint's tokenizer.

    The refusal point for a checkpoint/family pair, and it happens before the
    weights are read: a tokenizer that does not carry the declared residue alphabet,
    or that merges residues where the family declares one token per residue, is
    refused here rather than producing a complete artefact for a different object.
    An undeclared family name is refused by ``joint_modes.rendering`` itself, which
    is the single place either mode's format is decided (Appendix B rule 12).
    """

    return joint_modes.resolve(tokenizer, joint_modes.rendering(name))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--arm",
        default=None,
        choices=ADMISSIBLE_ARMS,
        help="a declared arm to measure. The set is composed by "
        "src.transfer.replaceable.perturbable_arms: stages 15, 17 and 22's "
        "eligible arms, plus the parallel-residual ProGen2 rungs whose block this "
        "stage's estimand -- which reads a block's output and not its input -- is "
        "defined on. The matched standalone pair gpt2-large and protgpt2 are what "
        "a joint result is read against, and the four progen2 rungs are the "
        "protein scale ladder",
    )
    target.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="directory of a joint language-protein checkpoint. A path and not an "
        "arm name: a checkpoint that has not passed 21_joint_mode_qualification.py "
        "must not be in the panel, so there is nothing for a default to point at",
    )
    parser.add_argument(
        "--rendering",
        default=None,
        choices=joint_modes.RENDERING_NAMES,
        help="which declared family's input format --checkpoint takes. Required "
        "with --checkpoint and refused with --arm. The set is composed by "
        "src.transfer.joint_modes, the single place either mode's format is decided",
    )
    parser.add_argument(
        "--modes",
        default=None,
        choices=("text", "protein", "both"),
        help="which modes of a joint checkpoint to measure; defaults to 'both', "
        "because the pivotal comparison is between one checkpoint's two modes and "
        "a single mode cannot make it. Refused with --arm, whose mode is the arm's "
        "declared modality",
    )
    parser.add_argument(
        "--protein-context",
        default=None,
        help="optional document context a joint checkpoint's protein block is "
        "embedded in, filled into the family's declared template. Omitted means the "
        "bare block, and whichever was used reaches the artefact",
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=list(DEFAULT_EPSILONS),
        help="relative perturbation magnitudes, as a fraction of the block output "
        "norm at each position. 0 is always prepended: it is the identity that "
        "checks the splice path is a no-op when it should be",
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=DEFAULT_DRAWS_PER_EPSILON,
        help="independent random directions per epsilon, reported individually. "
        "One draw of a random perturbation is a point estimate of a distribution",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=tuple(STAGE21._DTYPES),
        help="inference dtype, read back from the loaded parameters",
    )
    parser.add_argument("--sequences", type=int, default=128)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="token cap the inputs are truncated to. A rendered protein is never "
        "truncated on a joint checkpoint: the stage raises instead, because dropping "
        "the closing delimiter would silently change the scored span",
    )
    parser.add_argument(
        "--text-min-chars",
        type=int,
        default=800,
        help="floor of the text cohort, in characters; src.transfer.arms.text_cohort's "
        "own default, so the population is the one every other text measurement in "
        "this repository uses",
    )
    parser.add_argument(
        "--protein-min-len",
        type=int,
        default=64,
        help="lower edge of the residue band. The default is the shared 64-246 band "
        "the replacement comparison is anchored on (F11)",
    )
    parser.add_argument("--protein-max-len", type=int, default=246)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument(
        "--cohort-draw-seed",
        type=int,
        default=DEFAULT_CORPUS_DRAW_SEED,
        help="seed for the permutation this stage's cohorts are drawn under; "
        "0 selects the historical file-order prefix, which is a declared choice and "
        "not a default (transfer audit, Appendix B rule 1)",
    )
    parser.add_argument("--cohort-skip", type=int, default=0)
    return parser


def resolve_target(args: argparse.Namespace) -> tuple[str, ...]:
    """Refuse an incoherent target before anything is loaded, and name the modes."""

    if args.draws < 1:
        raise ValueError("--draws must be at least 1; a random control needs a draw")
    for value in args.epsilons:
        if value < 0:
            raise ValueError("a relative perturbation magnitude cannot be negative")
    if args.checkpoint is not None:
        if args.rendering is None:
            raise ValueError(
                "--checkpoint needs --rendering: the input format a joint checkpoint "
                "was trained on is a declaration, and a run that guessed it would "
                "produce a complete artefact for a different object (Appendix B rule 4)"
            )
        modes = ("protein", "text") if (args.modes or "both") == "both" else (args.modes,)
        unknown = sorted(set(modes) - set(JOINT_MODES))
        if unknown:
            raise ValueError(f"unknown joint mode(s) {unknown}; declared: {JOINT_MODES}")
        return modes
    if args.rendering is not None:
        raise ValueError(
            "--rendering declares a joint checkpoint's input format; a panel arm's "
            "rendering is declared by src.transfer.arms.PANEL and is not chosen here"
        )
    if args.modes is not None:
        raise ValueError(
            "--modes selects which modes of a JOINT checkpoint to measure; a panel "
            "arm has one mode, its declared modality"
        )
    return ()


def main() -> None:
    args = build_parser().parse_args()
    joint_mode_names = resolve_target(args)
    args.out.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "provenance": {
            "runner": {
                "path": "scripts/transfer/23_perturbation_sensitivity.py",
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "modules": {
                name: sha256_file(REPO_ROOT / name) for name in PROVENANCE_MODULES
            },
        },
        "estimand": (
            "fraction of the clean-to-mean-ablated cross-entropy gap that survives a "
            "random perturbation of declared relative magnitude epsilon applied to "
            "every replaced block's output: (mean_ablated - perturbed) / "
            "(mean_ablated - clean), over the cohort's scored targets. The same "
            "estimand 15_replacement_faithfulness.py and 22_neuron_basis_circuit.py "
            "measure, under a different intervention, so the three are directly "
            "comparable"
        ),
        "perturbation": {
            "construction": (
                "at every position of every replaced block, r is a uniformly random "
                "direction scaled so that ||r|| = epsilon * ||y||, where y is that "
                "block's declared perturbation target at that position in the "
                "perturbed forward pass; the block writes y + r"
            ),
            "target_declaration": (
                "y is the per-layer FEED-FORWARD output, before anything is added to "
                "it. On a serial residual block (gpt2, llama) that tensor is the whole "
                "of the block's residual write; on a parallel GPT-J-style block "
                "(progen2) it is one of two terms in a single three-way sum and the "
                "attention term is left untouched, exactly as it is on a serial arm. "
                "The identity certifying the interception is verified exactly against "
                "the live forward pass before the arm is measured, and each mode's "
                "modes.<mode>.perturbation_target records which layout it ran on"
            ),
            "epsilon_anchored_at": (
                "the block output norm, measured per arm, per mode, per layer and per "
                "position, so a fixed epsilon is a genuinely matched manipulation "
                "across modes whose activation scales differ"
            ),
            "applied_at": "every position of every replaced block, sequentially over "
            "every layer, spliced through the same "
            "ReplaceableModel.block_intercept the replacement and the mean ablation "
            "use",
            "epsilon_grid": [0.0, *(float(value) for value in args.epsilons)],
            "draws_per_epsilon": int(args.draws),
            "degenerate_positions": (
                "a block output of zero norm is refused rather than perturbed by zero: "
                "epsilon is a fraction of that norm, and a zero perturbation would "
                "report an unperturbed position as a perturbed one"
            ),
            "range_note": (
                "the recovery fraction is NOT bounded below by zero. Mean ablation "
                "removes the block's information; a perturbation removes it and also "
                "injects norm the residual stream never carried, which damages the "
                "attention and embedding pathways as well, so at large epsilon the "
                "ratio falls through the mean-ablated floor and goes negative. That is "
                "a property of the manipulation and not a defect"
            ),
            "limitation": (
                "the perturbation is formed in float32 and cast back to the block's own "
                "dtype, so what the model sees carries that dtype's rounding (a "
                "relative error of order 4e-3 at bfloat16); and an isotropic direction "
                "bounds what a perturbation of this SIZE costs, not what the worst "
                "perturbation of that size costs"
            ),
        },
    }

    modes: dict[str, dict[str, Any]] = {}
    if args.checkpoint is not None:
        declaration = joint_modes.rendering(args.rendering)
        print(f"[load] {args.checkpoint} as {declaration.name} on {args.device}")
        # The tokenizer alone first, so a wrong checkpoint/family pair fails before
        # a multi-gigabyte load. Stage 21's loaders, imported, so that this stage
        # reads the checkpoint the qualification stage qualified.
        resolved, tokenizer = STAGE21.load_tokenizer(args.checkpoint)
        tokenisation = (
            resolve_protein_rendering(tokenizer, args.rendering)
            if "protein" in joint_mode_names
            else None
        )
        backbone, checkpoint_facts = STAGE21.load_model(
            resolved, tokenizer, device=args.device, dtype=args.dtype
        )
        checkpoint_facts["requested_path"] = str(args.checkpoint)
        print(
            f"  {checkpoint_facts['n_layers']}L x {checkpoint_facts['d_model']}d x "
            f"{checkpoint_facts['n_heads']}h, vocab {checkpoint_facts['vocab_size']}"
        )
        target: dict[str, Any] = {
            "kind": "joint_checkpoint",
            "rendering_family": declaration.name,
            "checkpoint": checkpoint_facts,
            # Digested before the sweeps rather than after them: a checkpoint whose
            # weight files cannot be identified should stop the run in a second, not
            # after every mode has been measured.
            "weights_sha256": checkpoint_weights_digest(resolved),
            "n_layers": int(checkpoint_facts["n_layers"]),
            "n_heads": int(checkpoint_facts["n_heads"]),
            "d_model": int(checkpoint_facts["d_model"]),
            "loading_note": JointReplaceable.loading_note,
            "modes_measured": list(joint_mode_names),
        }
        for mode in joint_mode_names:
            modes[mode] = measure_mode(
                args,
                JointReplaceable(
                    model=backbone,
                    tokenizer=tokenizer,
                    checkpoint=resolved,
                    declaration=declaration,
                    mode=mode,
                    tokenisation=tokenisation if mode == "protein" else None,
                    max_tokens=args.max_tokens,
                    protein_context=args.protein_context,
                ),
                # The corpus a joint mode is read from is declared once in
                # src.transfer.replaceable, because 15_replacement_faithfulness.py
                # and 17_train_transcoder.py read the same declaration and a
                # dictionary trained on one population and scored on another is
                # the train/eval gap EXP-R2-135 priced at 4.1x.
                source=joint_mode_corpus(mode),
                label=f"{declaration.name}:{mode}",
            )
    else:
        print(f"[load] panel arm {args.arm} on {args.device}")
        model = load_replaceable(
            args.arm,
            campaign_panel=CAMPAIGN_PANEL,
            admissible=ADMISSIBLE_ARMS,
            device=args.device,
            dtype=args.dtype,
            max_tokens=args.max_tokens,
        )
        mode = model.cohort_kind
        target = {
            "kind": "panel_arm",
            "arm": args.arm,
            "checkpoint": str(model.checkpoint),
            "weights_sha256": model.weights_digest(),
            "n_layers": model.n_layers,
            "n_heads": model.n_heads,
            "d_model": model.width,
            "loading_note": model.loading_note,
            "modes_measured": [mode],
        }
        modes[mode] = measure_mode(
            args,
            model,
            source=arm_evaluation_cohort_source(args.arm),
            label=f"{args.arm}:{mode}",
        )

    target["dtype"] = args.dtype
    target["device"] = args.device
    target["block_kind"] = "every replaced block, in every layer"
    payload.update(
        {
            "target": target,
            "modes": modes,
            "cross_mode": cross_mode_record(modes),
            "verdict": (
                "PASS"
                if all(
                    record["gates"]["endpoints"]["verdict"] == "PASS"
                    for record in modes.values()
                )
                else "FAIL"
            ),
        }
    )
    destination = args.out / "perturbation_sensitivity.json"
    write_json(destination, payload)
    print()
    for name, record in modes.items():
        print(
            f"[{name}] denominator "
            f"{record['gates']['endpoints']['denominator_nats_per_token']:.4f} nats/token at "
            f"{record['symbols_per_token']['value']:.3f} "
            f"{record['symbols_per_token']['unit']}"
        )
        for point in record["sweep"]:
            summary = point["across_draws"]["recovery"]
            damage = point["across_draws"]["damage_nats_per_token"]
            print(
                f"  epsilon {point['epsilon']:<5} damage {damage['mean']:+.4f} nats "
                f"[{damage['min']:+.4f}, {damage['max']:+.4f}]  recovery "
                + (
                    "None"
                    if summary is None
                    else f"{summary['mean']:.4f} [{summary['min']:.4f}, {summary['max']:.4f}]"
                )
            )
    print(f"wrote {destination}  verdict {payload['verdict']}")


if __name__ == "__main__":
    main()
