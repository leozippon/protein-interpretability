#!/usr/bin/env python3
"""Which directions does adaptation change, and are they shared with the pre-adaptation checkpoint?

**What this stage is for.** R2.4's same-input Model Diffing unit compares
``Llama-2-7b-hf`` with ``ProLLaMA_Stage_1`` on identical rendered inputs.
``25_model_diffing_baselines.py`` has already answered the question that does not
need a dictionary -- whether an affine or orthogonal map removes the difference --
and it does not, in either mode: the held-out ridge residual under true pairing is
0.400 on protein and 0.684 on text, far below their shuffled-pairing values of
1.002 and 1.001, so both modes measure correspondence rather than capacity and no
linear map explains what is left. What no stage can yet say is *which* directions
the remaining difference lives in and whether they are shared between the two
checkpoints or specific to one. A Crosscoder is the object that answers that: one
dictionary trained jointly over both checkpoints' activations at the same position
of the same record, with a shared latent space and per-model decoders, read out
through the relative norms of those decoders.

The formulation, its deviations and the readout are declared once in
:mod:`src.transfer.crosscoder`; this stage owns the checkpoints, the corpus, the
admissibility rule and the artefact.

**The identical-input guarantee is the premise and is not this stage's own work.**
``25_model_diffing_baselines.py`` established it and this stage imports it
unchanged: ``assert_identical_tokenizers`` refuses two checkpoints that are not
one vocabulary, ``assert_comparable_shape`` refuses two that are not one
architecture at one width, and ``paired_capture`` renders and batches each side
independently on every batch, compares the two renderings, compares the two batch
tensors field by field, and compares the two content masks -- so a position of one
checkpoint is the same token in the same context as the position of the other, or
the run stops. Nothing here weakens any of that, and a Crosscoder trained on
mispaired positions is exactly the artefact that would look finite and plausible
while comparing unrelated things.

**A per-site reconstruction number is not readable without the effective
dimension of what it reconstructed, so the stage refuses to run without it.**
``--r99-per-site`` is required on a checkpoint pair and is carried inside every
per-site record rather than only in a limitations block. The reason is an
interpretive trap sitting in the numbers this unit already has: at layers 27-28
R2.3's per-layer transcoders read 0.0670 and 0.0585 held-out NMSE on protein
against 0.5739 and 0.5369 on text, roughly ninefold better, and the naive reading
is that the protein dictionaries are the better ones. The likelier reading is the
opposite in spirit -- protein activations at those depths occupy far fewer
directions, so there is much less structure to reconstruct, and a low NMSE there
says how little the data does rather than how well the dictionary did it.

**The dependence is not confined to modes, and the rule is therefore stated
positively.** Effective dimension varies across modes, across the two checkpoints
and across layers, and the reconstruction follows it on all three axes: within
text the two roles read median ``r99`` 3,670 against 2,954 with NMSE sums 16.08
against 5.71, the larger cloud reconstructing worse, and the protein pair moves
the same way; across layers inside one cell the rank correlation between ``r99``
and NMSE is +0.98, +0.82 and +0.73 in three of the four cells. So **a per-site
NMSE may be compared only with one fitted to the same cloud -- same mode, same
role, same site, which means R2.3's per-layer transcoder at that site and
width -- and never across modes, across the two halves of the ``[base, adapted]``
pair, or between entries of a per-site vector.** It is a prohibition rather than
a correction because the relationship is directional and not proportional, with
one cell running the other way in its interior: there is no factor to divide out.

**Per-layer admissibility is a required input and is never inferred.** R2.4's
operative rule, after both of the unit's original basis criteria went void on
their own text control, is admissibility *at a layer*: a diff may be reported at
layer ``l`` only where both cells' dictionaries carry at least that layer's own
effective dimension ``r99`` in live latents, and -- under the amendment that
followed -- only where that layer's ``r99`` is itself non-degenerate. ``--layers``
says where a dictionary is fitted; ``--admissible-layers`` says where a diff may
be reported, is **required**, must be a subset of ``--layers``, and reaches the
artefact. At an inadmissible site the artefact carries an explicit
``ADMISSIBILITY_REFUSED`` in place of the numbers. Reconstruction quality is
reported at every fitted site, because it is a statement about the dictionary and
not a diff.

**The shuffled-pairing null is fitted in the same pass, not in a second run.** Two
Crosscoders are trained over one capture of the activations -- one on the true
pairing, one with the adapted checkpoint's positions permuted within each batch --
so they see literally the same positions in the same order under the same frozen
normalisation constants and differ in exactly the pairing. That is what stops the
null being dispatched separately, dispatched later, or not dispatched at all, and
it removes the draw-to-draw variation two invocations would have carried. The
permutation's known bound travels with it: it is within a batch and not global,
which makes it conservative (see ``src.transfer.crosscoder.SHUFFLE_NOTE``).

**Why a narrow site set is the default rather than the whole stack.** The sites of
this Crosscoder are parameter-disjoint and the objective is a sum over them, with
the two mechanisms that could have coupled them -- a shared initialisation draw
and a global gradient clip -- removed rather than disclosed. So training the whole
stack and reporting at two layers yields the *same fitted dictionary at those two
layers* as training those two layers alone; the choice is economics, and the
narrow run is 16x cheaper in dictionary state. ``tests/test_crosscoder.py`` checks
the claim by fitting one site alone and inside a wider run.

**The instrument check, on data whose answer is known.** ``--synthetic-check``
fits the same object on paired activations carrying a declared number of shared,
base-only and adapted-only features and reports injected against recovered counts
per category, beside the same shuffled-pairing null. A Crosscoder's readout is an
unsupervised claim that nothing in a real run can falsify, so this is the only
place it is falsifiable and it is run before a real campaign rather than after.

**Memory, stated because it decides the site set.** Per site this object carries
``4 * d_model * d_hidden`` parameters -- two encoders and two decoders -- against a
per-layer transcoder's ``2 * d_model * d_hidden``. At ``d_model`` 4096 and
``d_hidden`` 8192 that is 134.2M parameters per site, so a 32-site Crosscoder has
the same 4.295B parameters as a single-model transcoder at ``d_hidden`` 16384: at
fp32 with AdamW's two moments and gradients, 65.5 GB of optimiser state, plus two
resident bfloat16 backbones at ~27 GB, plus paired activations. A two-site
Crosscoder at the same width is 4.1 GB of optimiser state instead. The full
arithmetic is in the artefact's ``memory`` block, recomputed from the run's own
configuration rather than quoted.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.transfer import crosscoder as cc  # noqa: E402
from src.transfer import joint_modes  # noqa: E402
from src.transfer.arms import (  # noqa: E402
    DEFAULT_CORPUS_DRAW_SEED,
    REPO,
    corpus_location,
    iter_corpus_records,
)
from src.transfer.io import sha256_file, write_json  # noqa: E402
from src.transfer.replaceable import (  # noqa: E402
    JOINT_MODES,
    JointReplaceable,
    joint_mode_corpus,
)
from src.transfer.transcoders import (  # noqa: E402
    DEAD_STEPS_SEQUENCES,
    MATCHED_TRAINING_KEY,
    MatchedTraining,
)


def _load_stage(filename: str) -> Any:
    """Import a stage whose module name starts with a digit.

    Three of them, and each because this stage's numbers are only readable if
    they are the *same* computation that stage performs: stage 21 owns the
    checkpoint loader, stage 17 owns the captured tensors, the seeded stream, the
    residue band and the screened held-out cohort, and stage 25 owns every
    identical-input refusal this stage rests on. Appendix B rule 12 does not stop
    applying because the declaration lives in a file whose name starts with a
    digit.
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

SCHEMA_VERSION = "r2_transfer_crosscoder_v1"
DEFAULT_OUT = REPO / "results/transfer/crosscoder"

#: Both checkpoints at one precision, for the reason stage 25 hard-codes it: a
#: comparison between two checkpoints held at two precisions is partly a
#: comparison between two quantisations. The dictionary itself trains in float32.
INFERENCE_DTYPE = "bfloat16"

PROVENANCE_MODULES = (
    "src/transfer/crosscoder.py",
    "src/transfer/transcoders.py",
    "src/transfer/replaceable.py",
    "src/transfer/joint_modes.py",
    "src/transfer/arms.py",
    "src/transfer/near_duplicates.py",
    "src/transfer/io.py",
    "scripts/transfer/17_train_transcoder.py",
    "scripts/transfer/21_joint_mode_qualification.py",
    "scripts/transfer/25_model_diffing_baselines.py",
)


# ------------------------------------------------------------------- helpers


def parse_layers(argument: str) -> tuple[int, ...]:
    """``"0,27-29,31"`` into ``(0, 27, 28, 29, 31)``.

    Ranges are inclusive at both ends, because a layer set is written by a person
    reading a per-layer table and a half-open range would silently drop the layer
    they meant to include.
    """

    layers: list[int] = []
    for piece in argument.replace(" ", "").split(","):
        if not piece:
            continue
        if "-" in piece.lstrip("-"):
            low, _, high = piece.partition("-")
            start, stop = int(low), int(high)
            if stop < start:
                raise argparse.ArgumentTypeError(
                    f"{piece!r} is a descending range; write it low-high"
                )
            layers.extend(range(start, stop + 1))
        else:
            layers.append(int(piece))
    if not layers:
        raise argparse.ArgumentTypeError("a layer set cannot be empty")
    if len(set(layers)) != len(layers):
        raise argparse.ArgumentTypeError(f"{argument!r} names a layer twice")
    return tuple(sorted(layers))


def parse_int_vector(argument: str) -> tuple[int, ...]:
    """``"3690,2709,2588"`` into ``(3690, 2709, 2588)``, order preserved.

    Deliberately **not** :func:`parse_layers`, which sorts and rejects repeats
    because a layer set is a set. This is a vector of measurements indexed by
    position: two sites may legitimately share an effective dimension, and sorting
    it would silently re-associate every value with the wrong layer.
    """

    values: list[int] = []
    for piece in argument.replace(" ", "").split(","):
        if piece:
            values.append(int(piece))
    if not values:
        raise argparse.ArgumentTypeError("an empty vector carries no measurement")
    return tuple(values)


def memory_arithmetic(
    config: cc.CrosscoderConfig,
    *,
    n_dictionaries: int,
    tokens_per_batch: int,
    n_backbones: int,
    backbone_parameters: float,
) -> dict[str, Any]:
    """Every term of the live memory requirement, recomputed from this run.

    Recomputed rather than quoted, because a reservation reported by
    ``nvidia-smi`` measures how much cache the allocator was permitted to keep and
    not what a run needs: two cells at 2x different parameter counts have been
    observed at within 1.2% of each other's reservation purely because one had
    been squeezed and released cache and the other had not. Every figure below is
    a live requirement.
    """

    mib = 1024.0 * 1024.0
    parameters = config.n_parameters()
    parameter_bytes = parameters * 4.0
    # AdamW keeps two moments beside the parameter and the gradient: four copies.
    optimiser_bytes = 4.0 * parameter_bytes * n_dictionaries
    backbone_bytes = n_backbones * backbone_parameters * 2.0
    # Live working set per token, measured in the shape this objective allocates:
    # the latent path is (n_sites, tokens, d_hidden) and the encode/decode path
    # holds about six such copies across forward and backward; the activation
    # path is (n_sites, tokens, d_model) for two roles, their scaled copies,
    # their reconstructions and the residual, about twelve copies.
    latent_bytes_per_token = 6.0 * config.n_sites * config.d_hidden * 4.0
    activation_bytes_per_token = 12.0 * config.n_sites * config.d_model * 4.0
    working_bytes = (
        (latent_bytes_per_token + activation_bytes_per_token)
        * tokens_per_batch
        * n_dictionaries
    )
    return {
        "assumptions": (
            "fp32 dictionary with AdamW (parameter + gradient + two moments = 4 "
            "copies), bfloat16 backbones, and a live activation working set of "
            "about six (n_sites, tokens, d_hidden) copies and twelve "
            "(n_sites, tokens, d_model) copies across the forward and backward "
            "pass. Excludes the CUDA context and the transient capture buffer, "
            "which are reported separately by the caller. These are LIVE "
            "requirements: an allocator reservation is not one"
        ),
        "n_dictionaries": int(n_dictionaries),
        "tokens_per_batch": int(tokens_per_batch),
        "dictionary_parameters_each": int(parameters),
        "dictionary_parameters_total": int(parameters * n_dictionaries),
        "dictionary_fp32_mib": parameter_bytes * n_dictionaries / mib,
        "optimiser_state_mib": optimiser_bytes / mib,
        "backbones_bf16_mib": backbone_bytes / mib,
        "working_set_mib": working_bytes / mib,
        "working_set_mib_per_token": (
            (latent_bytes_per_token + activation_bytes_per_token) * n_dictionaries / mib
        ),
        "live_total_mib": (optimiser_bytes + backbone_bytes + working_bytes) / mib,
    }


def paired_batches(
    base: JointReplaceable,
    adapted: JointReplaceable,
    records: Callable[[], Iterator[tuple[str, str | None]]],
    *,
    tensor: str,
    batch_size: int,
    site_index: torch.Tensor,
) -> cc.PairedBatches:
    """A factory of ``(base, adapted)`` activations at the declared sites.

    ``25_model_diffing_baselines.paired_capture`` does the work unchanged, so
    every batch is rendered and batched independently by each checkpoint, the two
    renderings are compared, the two batch tensors are compared field by field and
    the two content masks are compared -- and the run stops rather than proceeding
    on a mismatch. This function only selects the fitted sites from the full-depth
    capture, which is free: the forward pass costs what it costs whatever subset
    of its layers is kept.
    """

    def factory() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        chunk: list[tuple[str, str | None]] = []
        for record in records():
            chunk.append(record)
            if len(chunk) < batch_size:
                continue
            left, right = STAGE25.paired_capture(base, adapted, chunk, tensor=tensor)
            chunk = []
            if left.shape[1] == 0:
                continue
            yield left[site_index], right[site_index]
        if chunk:
            left, right = STAGE25.paired_capture(base, adapted, chunk, tensor=tensor)
            if left.shape[1] > 0:
                yield left[site_index], right[site_index]

    return factory


def readout_for(
    model: cc.Crosscoder,
    evaluation: dict[str, Any],
    *,
    admissible: Sequence[int],
    exclusive_cut: float,
    shared_halfwidth: float,
    live_threshold: int,
    effective_dimension: Sequence[int] | None,
) -> dict[str, Any]:
    """One fitted Crosscoder's per-site reconstruction and per-site diff readout."""

    counts = evaluation.pop("_counts")
    live = cc.live_mask(counts, minimum=live_threshold)
    live_per_site = [int(value) for value in live.sum(dim=1)]
    readout = cc.specificity_readout(
        model,
        live=live,
        admissible=admissible,
        exclusive_cut=exclusive_cut,
        shared_halfwidth=shared_halfwidth,
    )
    return {
        "pairing": model.config.pairing,
        "held_out": evaluation,
        # Two definitions of a live basis, side by side and never blended: the
        # census on the held-out cohort, which is what the readout masks with and
        # what a diff is actually read over, and the checkpoint's own dead-latent
        # counter. Both per site.
        "live_latents_per_site": live_per_site,
        "live_from_silent_steps_per_site": model.live_latents_per_site(),
        "live_threshold": int(live_threshold),
        "nmse_per_site": evaluation["nmse_per_site"],
        # The reconstruction numbers with the quantity that qualifies them in the
        # same record. `nmse_per_site` above is the bare vector and is kept
        # because the per-site guard is keyed on it; this is the form a reader
        # should take the number from, and it cannot be taken without its limit.
        "reconstruction_per_site": cc.reconstruction_per_site(
            sites=model.config.sites,
            nmse_by_role=evaluation["nmse_by_role_per_site"],
            nmse_total=evaluation["nmse_per_site"],
            live=live_per_site,
            effective_dimension=effective_dimension,
        ),
        "readout": readout,
    }


def null_comparison(
    true_readout: dict[str, Any],
    null_readout: dict[str, Any],
    *,
    sites: Sequence[int],
    effective_dimension: Sequence[int] | None,
) -> dict[str, Any]:
    """The measurement beside its null, per site, at the admissible sites only.

    What the null controls, stated because the direction is not the obvious one.
    Mispairing destroys token-level correspondence, so a genuinely shared latent
    -- one that explains the same position in both checkpoints -- cannot exist
    under it and the optimum is a union of two independent dictionaries. The
    null's exclusive counts are therefore the **free-capacity floor**: the number
    of model-specific latents a dictionary of this width produces when there is by
    construction no model-specific structure to find. A specificity signal is a
    shared fraction *above* the null and an exclusive fraction *below* it, and a
    true-pairing exclusive count at or above the null's is not a finding.
    """

    per_site: list[dict[str, Any]] = []
    for index, layer in enumerate(sites):
        left = true_readout["readout"]["site_per_site"][index]
        right = null_readout["readout"]["site_per_site"][index]
        if not left["admissible"]:
            per_site.append(
                {
                    "layer": int(layer),
                    "admissible": False,
                    "verdict": "ADMISSIBILITY_REFUSED",
                }
            )
            continue

        def exclusive(entry: dict[str, Any]) -> float:
            return (
                entry["fractions"]["base_specific"]
                + entry["fractions"]["adapted_specific"]
            )

        gap = left["fractions"]["shared"] - right["fractions"]["shared"]
        per_site.append(
            {
                "layer": int(layer),
                "admissible": True,
                "shared_fraction": [left["fractions"]["shared"], right["fractions"]["shared"]],
                "exclusive_fraction": [exclusive(left), exclusive(right)],
                "shared_fraction_above_null": gap,
                "exclusive_fraction_above_null": exclusive(left) - exclusive(right),
                "n_live": [left["n_live"], right["n_live"]],
                "held_out_nmse": [
                    true_readout["nmse_per_site"][index],
                    null_readout["nmse_per_site"][index],
                ],
                # C3's gap is read here, and the quantity that decides whether it
                # can separate at all is this layer's effective dimension: the gap
                # measures 0.90 at rank ratios of 0.63 and above and 0.19 at
                # 0.125 (EXP-R2-205). It travels in the same record for the same
                # reason the NMSE carries it -- so the limit is where the number
                # is.
                "r99_effective_dimension": (
                    None if effective_dimension is None else list(effective_dimension[index])
                ),
                "held_out_nmse_comparability": cc.NMSE_COMPARABILITY_NOTE,
            }
        )
    return {
        "order": ["true", "shuffled"],
        "site_per_site": per_site,
        "what_the_null_controls": null_comparison.__doc__,
        "shuffle_note": cc.SHUFFLE_NOTE,
    }


# ------------------------------------------------------------- synthetic mode


def run_synthetic_check(args: argparse.Namespace) -> dict[str, Any]:
    """Fit the same instrument on paired activations whose answer is known.

    Three constructions, and the third is the one that bears on whether this
    instrument is usable on protein activations at all:

    ``recovery``
        known counts of shared, base-only and adapted-only features at full rank.
    ``rank_deficient``
        the same, with every direction confined to a subspace narrower than
        ``d_model``. This is the operating regime and not a corner case: the
        measured effective dimension at this programme's dictionary site is 2,588
        to 3,670 against ``d_model`` 4,096 on the four R2.4 cells, and it collapses
        to double digits at the top of the stack.
    ``all_shared``
        a negative control with **zero** injected model-specific features, so any
        exclusive latent the readout reports is spurious by construction.
    """

    torch.manual_seed(args.seed)
    truths = {
        "recovery": cc.SyntheticGroundTruth(
            d_model=args.synthetic_d_model,
            n_sites=1,
            n_shared=args.synthetic_shared,
            n_base_specific=args.synthetic_exclusive,
            n_adapted_specific=args.synthetic_exclusive,
            active_per_token=args.k,
            seed=args.seed,
        ),
        "rank_deficient": cc.SyntheticGroundTruth(
            d_model=args.synthetic_d_model,
            n_sites=1,
            n_shared=args.synthetic_shared,
            n_base_specific=args.synthetic_exclusive,
            n_adapted_specific=args.synthetic_exclusive,
            active_per_token=args.k,
            seed=args.seed + 1,
            rank=args.synthetic_rank,
        ),
        "all_shared": cc.SyntheticGroundTruth(
            d_model=args.synthetic_d_model,
            n_sites=1,
            n_shared=args.synthetic_shared + 2 * args.synthetic_exclusive,
            n_base_specific=0,
            n_adapted_specific=0,
            active_per_token=args.k,
            seed=args.seed + 2,
            rank=args.synthetic_rank,
        ),
    }

    cells: dict[str, Any] = {}
    for name, truth in truths.items():
        configs = [
            cc.CrosscoderConfig(
                sites=(0,),
                d_model=truth.d_model,
                d_hidden=args.d_hidden,
                k=args.k,
                auxk=args.auxk,
                dead_steps=args.dead_steps,
                decoder_norm_penalty=args.decoder_norm_penalty,
                pairing=pairing,
            )
            for pairing in args.pairings
        ]
        training = truth.batches(
            tokens_per_batch=args.synthetic_tokens,
            n_batches=args.steps + args.warm_up_batches + 8,
            seed=args.seed + 10,
        )
        held_out = truth.batches(
            tokens_per_batch=args.synthetic_tokens, n_batches=8, seed=args.seed + 20
        )
        models, _, extra = cc.train_crosscoders(
            configs,
            training,
            steps=args.steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            seed=args.seed,
            warm_up_batches=args.warm_up_batches,
            held_out=held_out,
            eval_every=max(1, args.steps // 3),
            log=lambda line: print(f"[{name}]{line}", flush=True),
        )
        rng = np.random.default_rng(args.seed + 30)
        probe_base, probe_adapted, coefficients = truth.draw(
            args.synthetic_tokens * 4, rng=rng
        )
        by_pairing: dict[str, Any] = {}
        for index, model in enumerate(models):
            evaluation = extra["held_out"][index]
            live = cc.live_mask(evaluation.pop("_counts"), minimum=args.live_threshold)
            by_pairing[model.config.pairing] = {
                # The same name the real artefact uses for the same quantity: the
                # per-site guard is keyed on names, so two spellings would let a
                # reader -- and the guard -- treat them as different fields.
                "nmse_per_site": evaluation["nmse_per_site"],
                "live_latents_per_site": [int(value) for value in live.sum(dim=1)],
                "recovery": cc.recovery_report(
                    truth,
                    model,
                    site=0,
                    base=probe_base,
                    adapted=probe_adapted,
                    coefficients=coefficients,
                    live=live,
                    exclusive_cut=args.exclusive_cut,
                    shared_halfwidth=args.shared_halfwidth,
                    correlation_floor=args.correlation_floor,
                ),
                "readout": cc.specificity_readout(
                    model,
                    live=live,
                    admissible=(0,),
                    exclusive_cut=args.exclusive_cut,
                    shared_halfwidth=args.shared_halfwidth,
                ),
            }
        cells[name] = {"ground_truth": truth.record(), "by_pairing": by_pairing}

    return {
        "cells": cells,
        "reading": {
            "recovery": (
                "injected against recovered counts per category. A true feature is "
                "RECOVERED when some latent's activation correlates with its "
                "coefficient above the declared floor on held-out positions, and "
                "CATEGORISED when that latent's relative decoder norm puts it in "
                "the injected category. The two are separate because a Crosscoder "
                "that finds every feature and mislabels half is a different failure "
                "from one that finds none"
            ),
            "null": (
                "under the shuffled pairing a shared feature is still recovered -- "
                "it is present in the base checkpoint's activations either way -- "
                "but it must be categorised as EXCLUSIVE, because the position it "
                "occupies in the other checkpoint is now someone else's. "
                "'categorised' collapsing to zero for the shared category while "
                "'recovered' stays high is the null behaving correctly, and is a "
                "sharper statement than a count moving"
            ),
            "all_shared": (
                "zero exclusive latents are injected, so every exclusive latent "
                "the readout reports at true pairing is spurious. Its count is the "
                "false-positive rate of the instrument in this rank regime, and it "
                "is the number that decides whether the readout may be used on "
                "activations of this effective dimension at all"
            ),
        },
    }


# ------------------------------------------------------------------- the CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        default=None,
        help="directory of the PRE-adaptation checkpoint, role index 0. A path and "
        "not an arm name, for the reason 21_joint_mode_qualification.py gives",
    )
    parser.add_argument(
        "--adapted",
        type=Path,
        default=None,
        help="directory of the adapted checkpoint, role index 1. The relative "
        "decoder norm runs 0 (base-specific) to 1 (adapted-specific), so which "
        "checkpoint is which decides the sign of every reading",
    )
    parser.add_argument(
        "--rendering", default=None, choices=joint_modes.RENDERING_NAMES,
        help="which declared family's input format BOTH checkpoints take",
    )
    parser.add_argument(
        "--mode", default=None, choices=JOINT_MODES,
        help="which mode to fit. One mode per run: the two modes have different "
        "corpora, different scored spans and different position counts",
    )
    parser.add_argument(
        "--tensor", default="block_output", choices=STAGE25.TENSORS,
        help="which per-layer tensor the dictionary is fitted to. It reaches the "
        "artefact by name, because 'the block output' names a different tensor on "
        "different block layouts",
    )
    parser.add_argument(
        "--layers", type=parse_layers, default=None,
        help="backbone layers a dictionary is FITTED at, as '0,27-29'. The sites "
        "are parameter-disjoint, so a narrow set yields the same dictionary at "
        "those layers as the whole stack would and costs proportionally less",
    )
    parser.add_argument(
        "--admissible-layers", type=parse_layers, default=None,
        help="backbone layers a DIFF may be reported at; must be a subset of "
        "--layers. REQUIRED and never defaulted: R2.4's admission rule is a "
        "positive statement about where a diff is defined, and a stage that "
        "defaulted it to 'everywhere fitted' would report the thing the rule "
        "forbids while looking exactly like the thing it permits",
    )
    for role in cc.ROLES:
        parser.add_argument(
            f"--r99-{role}-per-site", type=parse_int_vector, default=None,
            help=f"the measured effective dimension r99 of the {role} checkpoint's "
            "activations at each fitted layer, in --layers order, for THIS mode at "
            "THIS tensor -- from 30_activation_spectrum.py. REQUIRED on a "
            "checkpoint pair and never inferred here. Two vectors and not one: the "
            "two checkpoints do not share an effective dimension at the same layer "
            "(protein layer 28 reads 2,232 against 1,563), so one vector would "
            "qualify a role's reconstruction with the other role's geometry. It is "
            "the quantity that makes a per-site NMSE readable, and it moves "
            "opposite to the NMSE across modes",
        )
    parser.add_argument(
        "--pairings", nargs="+", default=list(cc.PAIRINGS), choices=list(cc.PAIRINGS),
        help="which pairings to fit, in one pass over the activations. Both by "
        "default: the null is not optional and running it as a second invocation "
        "would let it be skipped and would put the two fits on two draws",
    )
    parser.add_argument("--d-hidden", type=int, default=8192)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--auxk", type=int, default=192)
    parser.add_argument("--decoder-norm-penalty", type=float, default=3e-3,
        help="coefficient of the published L1-of-per-model-decoder-norms term, "
        "which is the entire mechanism behind the shared/exclusive separation. "
        "Zero is the pure-TopK control and is a legitimate setting")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--train-tokens", type=int, default=0,
        help="training budget in SCORED TOKENS. 0 runs --steps steps and makes "
        "two mode cells certify UNMATCHED_BUDGET rather than MATCHED, because a "
        "text record and a protein record carry different numbers of scored "
        "positions and equal steps are then a matched schedule over unequal data. "
        "A positive value stops at the first step to reach it; --steps then bounds "
        "the run and sets the held-out offset")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-sequences", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--warm-up-batches", type=int, default=8,
        help="batches the frozen per-(role, site) normalisation constants are "
        "estimated on before the first optimiser step")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--corpus-seed", type=int, default=DEFAULT_CORPUS_DRAW_SEED)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--protein-context", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--exclusive-cut", type=float, default=0.95,
        help="relative decoder norm at or above which a latent is adapted-specific, "
        "and at or below whose reflection it is base-specific. 0.95 is the value "
        "the 2025 crosscoder-diffing note reads its exclusive features at")
    parser.add_argument("--shared-halfwidth", type=float, default=0.10,
        help="half-width of the band around 0.5 read as shared")
    parser.add_argument("--live-threshold", type=int, default=1,
        help="times a latent must fire on the held-out cohort to count as live. "
        "The readout is taken over live latents only: a latent that never fires "
        "carries whatever decoder norms the initialiser left it")
    parser.add_argument("--allow-self-pair", action="store_true",
        help="permit both roles to name the same checkpoint. Every relative "
        "decoder norm is then exactly 0.5 and every decoder cosine exactly 1, "
        "which is a usable end-to-end check of the whole path and is NOT a model "
        "diff; the artefact records that it was one checkpoint against itself")
    parser.add_argument("--synthetic-check", action="store_true",
        help="fit the instrument on paired activations with known ground truth "
        "instead of on checkpoints, and write the recovery certificate")
    parser.add_argument("--synthetic-d-model", type=int, default=64)
    parser.add_argument("--synthetic-shared", type=int, default=24)
    parser.add_argument("--synthetic-exclusive", type=int, default=12)
    parser.add_argument("--synthetic-rank", type=int, default=24)
    parser.add_argument("--synthetic-tokens", type=int, default=256)
    parser.add_argument("--correlation-floor", type=float, default=0.5)
    parser.add_argument("--dead-steps", type=int, default=0,
        help="steps a latent may stay silent before the revival term reaches it. "
        "0 derives it from the batch size the way 17_train_transcoder.py does, so "
        "--batch-size cannot silently change how long a latent may stay silent")
    return parser


def resolve(args: argparse.Namespace) -> None:
    """Refuse an incoherent request before a corpus is opened or a model is loaded."""

    if args.dead_steps == 0:
        args.dead_steps = max(1, DEAD_STEPS_SEQUENCES // max(1, args.batch_size))
    args.pairings = tuple(dict.fromkeys(args.pairings))
    campaign = (
        "base", "adapted", "rendering", "mode", "layers", "admissible_layers",
        "r99_base_per_site", "r99_adapted_per_site",
    )
    if args.synthetic_check:
        for flag in campaign:
            if getattr(args, flag) is not None:
                raise ValueError(
                    f"--{flag.replace('_', '-')} names a real campaign and is "
                    "meaningless beside --synthetic-check, which fits the same "
                    "instrument on data whose answer is known"
                )
        return
    missing = [flag for flag in campaign if getattr(args, flag) is None]
    if missing:
        raise ValueError(
            "this stage needs "
            + ", ".join(f"--{flag.replace('_', '-')}" for flag in missing)
            + ". Three of these are never defaulted and all are inputs from other "
            "measurements: --admissible-layers is the pre-registered statement of "
            "where a diff may be reported, and the two --r99-*-per-site vectors "
            "are what make each site's reconstruction number readable at all"
        )


def main() -> None:
    args = build_parser().parse_args()
    resolve(args)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.synthetic_check:
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "kind": "synthetic_instrument_check",
            "settings": {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in vars(args).items()
            },
            "provenance": {
                "runner": {
                    "path": "scripts/transfer/32_crosscoder.py",
                    "sha256": sha256_file(Path(__file__).resolve()),
                },
                "modules": {
                    name: sha256_file(REPO_ROOT / name) for name in PROVENANCE_MODULES
                },
            },
            **run_synthetic_check(args),
        }
        # The synthetic artefact is held to the same per-site discipline as a real
        # one. Its cells are single-site, so the check is weak here -- and it is
        # applied anyway, because a guard that runs on one artefact kind and not
        # the other is a guard a reader cannot rely on.
        cc.assert_per_layer_fields(payload, n_sites=1)
        cc.assert_required_per_site_fields(payload)
        destination = args.out / "crosscoder__synthetic_check.json"
        write_json(destination, payload)
        print(f"wrote {destination}")
        return

    declaration = joint_modes.rendering(args.rendering)
    source = joint_mode_corpus(args.mode)
    corpus = corpus_location(source)
    print(f"[paths] base    {Path(args.base).resolve()}")
    print(f"[paths] adapted {Path(args.adapted).resolve()}")
    print(f"[paths] corpus  {corpus}  ({source}, mode {args.mode})")
    print(f"[paths] out     {args.out.resolve()}")

    base_path, base_tokenizer = STAGE21.load_tokenizer(Path(args.base))
    adapted_path, adapted_tokenizer = STAGE21.load_tokenizer(Path(args.adapted))
    vocabulary = STAGE25.assert_identical_tokenizers(base_tokenizer, adapted_tokenizer)

    tokenisation, base_facts, base_model = STAGE25.load_side(
        base_path, base_tokenizer, declaration=declaration, args=args
    )
    _, adapted_facts, adapted_model = STAGE25.load_side(
        adapted_path, adapted_tokenizer, declaration=declaration, args=args
    )
    shape = STAGE25.assert_comparable_shape(
        base_model, adapted_model, reference_facts=base_facts, target_facts=adapted_facts
    )
    declared_tensor = STAGE25.tensor_declaration(base_model, args.tensor)
    print(
        f"[shape] {shape['n_layers']}L x {shape['d_model']}d, vocabulary "
        f"{vocabulary['vocabulary_sha256'][:12]}.., tensor {args.tensor}"
    )

    outside = sorted(layer for layer in args.layers if not 0 <= layer < shape["n_layers"])
    if outside:
        raise ValueError(
            f"--layers names {outside}, which are outside this backbone's "
            f"0..{shape['n_layers'] - 1}"
        )
    admissible = cc.assert_admissible_subset(args.admissible_layers, args.layers)
    # One [base, adapted] pair per site: each role's reconstruction is qualified
    # by its own checkpoint's cloud, and the two differ materially at the same
    # layer -- 2,232 against 1,563 at layer 28 in protein mode.
    effective_dimension = [
        [int(base_r99), int(adapted_r99)]
        for base_r99, adapted_r99 in zip(
            cc.assert_effective_dimension(
                args.r99_base_per_site, args.layers, d_model=int(shape["d_model"])
            ),
            cc.assert_effective_dimension(
                args.r99_adapted_per_site, args.layers, d_model=int(shape["d_model"])
            ),
        )
    ]

    base_digest = base_model.weights_digest()
    adapted_digest = adapted_model.weights_digest()
    self_pair = base_digest == adapted_digest
    if self_pair and not args.allow_self_pair:
        raise ValueError(
            "both roles resolve to the same weights, so this is one checkpoint "
            "against itself and every relative decoder norm would be exactly 0.5 "
            "by construction. That is a legitimate end-to-end check of this path "
            "and is not a model diff; pass --allow-self-pair, which records it"
        )
    pair_digest = (
        base_digest if self_pair else cc.pair_backbone_digest(base_digest, adapted_digest)
    )

    low, high = STAGE17.CORPUS_BAND[source]
    if tokenisation is not None:
        low, high = STAGE17.joint_protein_band(
            tokenisation, max_tokens=args.max_tokens, protein_context=args.protein_context
        )

    def records() -> Iterator[tuple[str, str | None]]:
        return iter_corpus_records(source, min_symbols=low, max_symbols=high)

    symbol_unit = "characters" if source == "openwebtext" else "residues"
    held_out_records, screen, held_out_offset = STAGE17.held_out_cohort(
        records,
        corpus_seed=args.corpus_seed,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_sequences=args.eval_sequences,
        symbol_unit=symbol_unit,
    )
    print(
        f"[cohort] {len(held_out_records)} held-out records past a skip of "
        f"{held_out_offset}, band {[low, high]}"
    )

    site_index = torch.tensor([int(layer) for layer in args.layers], dtype=torch.long)

    def training_records() -> Iterator[tuple[str, str | None]]:
        return STAGE17.stream_records(records, seed=args.corpus_seed, skip=0, limit=None)

    def held_out_stream() -> Iterator[tuple[str, str | None]]:
        return iter(held_out_records)

    training = paired_batches(
        base_model,
        adapted_model,
        training_records,
        tensor=args.tensor,
        batch_size=args.batch_size,
        site_index=site_index,
    )
    held_out = paired_batches(
        base_model,
        adapted_model,
        held_out_stream,
        tensor=args.tensor,
        batch_size=args.batch_size,
        site_index=site_index,
    )

    configs = [
        cc.CrosscoderConfig(
            sites=tuple(int(layer) for layer in args.layers),
            d_model=int(shape["d_model"]),
            d_hidden=args.d_hidden,
            k=args.k,
            auxk=args.auxk,
            dead_steps=args.dead_steps,
            decoder_norm_penalty=args.decoder_norm_penalty,
            pairing=pairing,
        )
        for pairing in args.pairings
    ]
    print(
        f"[model] {len(configs)} Crosscoder(s), {len(args.layers)} site(s) x "
        f"{configs[0].n_parameters() / 1e6:.1f}M parameters each, admissible "
        f"{list(admissible)}"
    )

    models, training_records_out, extra = cc.train_crosscoders(
        configs,
        training,
        steps=args.steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        seed=args.seed,
        device=args.device,
        warm_up_batches=args.warm_up_batches,
        token_budget=args.train_tokens or None,
        held_out=held_out,
        eval_every=args.eval_every,
        log=lambda line: print(line, flush=True),
    )

    fitted: dict[str, Any] = {}
    for index, model in enumerate(models):
        fitted[model.config.pairing] = {
            **readout_for(
                model,
                extra["held_out"][index],
                admissible=admissible,
                exclusive_cut=args.exclusive_cut,
                shared_halfwidth=args.shared_halfwidth,
                live_threshold=args.live_threshold,
                effective_dimension=effective_dimension,
            ),
            "training": training_records_out[index].record(),
        }

    matched = MatchedTraining(
        target=f"{declaration.name}:{args.mode}",
        backbone_sha256=pair_digest,
        architecture="CROSSCODER",
        num_layers=len(args.layers),
        d_model=int(shape["d_model"]),
        d_hidden=args.d_hidden,
        k=args.k,
        auxk=args.auxk,
        training_token_budget=args.train_tokens or None,
        training_tokens=training_records_out[0].tokens,
        evaluation_sequences=int(args.eval_sequences),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        corpus_seed=int(args.corpus_seed),
        max_tokens=int(args.max_tokens),
    )

    limitations: dict[str, Any] = {
        "representational_only": (
            "a dictionary fitted to two activation tensors is not a behavioural "
            "quantity. Nothing here says either checkpoint does anything "
            "differently, only that a sparse basis over their joint activations "
            "does or does not allocate latents to one of them"
        ),
        "shuffled_null_scope": cc.SHUFFLE_NOTE,
        "exclusive_latents_are_dense_and_polysemantic": (
            "the 2025 crosscoder-diffing note measures model-exclusive features at "
            "about an order of magnitude higher activation density than shared "
            "ones, and finds many of them polysemantic, because shared features "
            "buy twice the reconstruction for twice the penalty and therefore win "
            "the competition for a limited feature budget. Its mitigation -- a "
            "designated subset of weight-shared latents at a reduced penalty -- is "
            "NOT implemented here, because this stage counts and categorises "
            "latents and reads none of them. A count is not a claim that the "
            "counted latents are interpretable, and this stage makes none"
        ),
        "scale_is_normalised_away": (
            "the per-(role, site) normalisation constants remove the two "
            "checkpoints' activation-scale difference from the readout, because a "
            "ratio of decoder norms would otherwise measure it. The ratio the "
            "normalisation removed is reported in normalisation.adapted_over_base_"
            "scale_ratio_per_site: a site where the two checkpoints differ only in "
            "scale is a real difference this readout is not capable of expressing"
        ),
        "per_site_nmse_is_comparable_only_within_one_cloud": cc.NMSE_COMPARABILITY_NOTE,
        "admissibility_is_not_informativeness": (
            "an admissible layer is one where both cells' dictionaries carry "
            "enough live latents for a diff to be defined. It is not a claim that "
            "a diff there is large, meaningful, or about anything in particular"
        ),
        "one_lineage_one_draw": (
            "one checkpoint pair, one mode, one corpus draw at one seed. The "
            "skip-offset sensitivity Appendix B rule 1 asks for is a second run"
        ),
        "precision": (
            f"both checkpoints are loaded at {INFERENCE_DTYPE} and the dictionary "
            "trains in float32. The two are quantised identically, so the "
            "quantisation cannot favour either role"
        ),
    }
    if self_pair:
        limitations["self_pair"] = (
            "both roles are the same checkpoint. Every relative decoder norm is "
            "0.5 and every decoder cosine is 1 by construction; this run is an "
            "end-to-end check of the path and carries no diff"
        )
    if args.mode == "protein":
        limitations["protein_mode_base_checkpoint"] = (
            "the pre-adaptation checkpoint's protein mode is behaviourally "
            "unmeasurable on this lineage (context information +0.084 nats/token, "
            "reversal cost -0.001 nats/residue, EXP-R2-152). A representational "
            "comparison does not require a measurable behavioural estimand -- the "
            "activations exist and are comparable position by position -- but no "
            "reading of this artefact is a behavioural claim about either side"
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
                "path": "scripts/transfer/32_crosscoder.py",
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "modules": {name: sha256_file(REPO_ROOT / name) for name in PROVENANCE_MODULES},
        },
        "estimand": (
            "the relative decoder norm ||W_dec^adapted_i|| / (||W_dec^base_i|| + "
            "||W_dec^adapted_i||) of every live latent of a Crosscoder trained "
            "jointly on both checkpoints' declared per-layer tensor at the same "
            "position of the same record, per fitted site, reported only at the "
            "declared admissible layers and always beside its shuffled-pairing null"
        ),
        "base": STAGE25.checkpoint_record(
            base_path, Path(args.base), base_facts, base_model, role="base"
        ),
        "adapted": STAGE25.checkpoint_record(
            adapted_path, Path(args.adapted), adapted_facts, adapted_model, role="adapted"
        ),
        "tokenizer_vocabulary": vocabulary,
        "comparability": shape,
        "tensor": declared_tensor,
        "identical_input_guarantee": (
            "every batch is rendered and batched independently by each checkpoint "
            "and the two results are compared -- the rendered strings, every field "
            "of the batch tensor, and the content mask -- by "
            "25_model_diffing_baselines.paired_capture, which raises rather than "
            "proceeding on any mismatch. The tokenizer vocabulary digest and the "
            "architecture comparison above are checked before the weights load"
        ),
        "config": configs[0].record(),
        "self_pair": self_pair,
        "backbone_pair_sha256": pair_digest,
        "admissibility": {
            "fitted_layers": list(args.layers),
            "admissible_layers": list(admissible),
            "inadmissible_layers": [
                int(layer) for layer in args.layers if layer not in admissible
            ],
            "rule": (
                "a diff may be reported at layer l only where BOTH cells' "
                "dictionaries carry at least that layer's own r99 live latents, and "
                "only where that layer's r99 is itself non-degenerate. The set is "
                "an input to this stage and is never inferred here: it depends on "
                "per-layer live-latent curves and per-layer r99 vectors that other "
                "stages measure, and it will move if the running width campaign "
                "shows the protein shortfall is capacity-limited"
            ),
            "site_independence": cc.SITE_INDEPENDENCE_NOTE,
            "r99_per_site": effective_dimension,
            "r99_roles": list(cc.ROLES),
            "r99_source": (
                "declared by --r99-base-per-site and --r99-adapted-per-site from "
                "scripts/transfer/30_activation_spectrum.py's measurement at this "
                "mode, this tensor and these layers. An input, never inferred "
                "here. It is repeated inside every per-site reconstruction record "
                "and every per-site null comparison rather than left only here, "
                "because it is the quantity that qualifies the numbers in those "
                "records and this unit's recurring failure has been limits written "
                "down somewhere other than where the number appears"
            ),
        },
        "normalisation": extra["scales"],
        "cohort": {
            "corpus": str(corpus),
            "corpus_source": source,
            "symbol_band": [low, high],
            "symbol_unit": symbol_unit,
            "input_rendering": base_model.rendering_note,
            "scored_positions": base_model.scoring_note,
            "held_out_offset": held_out_offset,
            "near_duplicate_screen": screen,
            "batch_size": int(args.batch_size),
            "draw": (
                "17_train_transcoder.py's own seeded block-shuffled stream for "
                "training and its own screened held-out cohort, drawn past "
                "everything the step budget reaches. That is the cohort every "
                "dictionary this programme has fitted on this lineage was held out "
                "on, which is what makes a Crosscoder's per-site reconstruction "
                "readable against the per-layer transcoders' -- and it is why "
                "25_model_diffing_baselines.draw_splits is NOT used here: that "
                "stage fits a map on one half of one pool, this one trains a "
                "dictionary against a step budget, and two definitions of the "
                "held-out set would be two populations under one name"
            ),
        },
        MATCHED_TRAINING_KEY: matched.record(),
        "crosscoder_matched_fields": {
            "sites": list(args.layers),
            "decoder_norm_penalty": float(args.decoder_norm_penalty),
            "pairing": list(args.pairings),
            "backbone_pair_sha256": pair_digest,
            "note": (
                "the four fields a Crosscoder adds to "
                "transcoders.MATCHED_TRAINING_FIELDS, which is frozen and is not "
                "extended here. src.transfer.crosscoder.crosscoder_certificate "
                "composes the two halves; any disagreement on either is MISMATCH"
            ),
        },
        "fitted": fitted,
        # Sized against the widest batch this run actually drew and against the
        # cap side by side. The cap is not the requirement: on this programme's
        # protein cohort the realised mean is 919 tokens against a 4,096-token
        # cap, so a campaign sized on the cap is sized on a batch the corpus never
        # produces -- and one sized on the mean would be under-sized. The realised
        # maximum is the number a successor run should budget against.
        "memory": {
            "at_realised_widest_batch": memory_arithmetic(
                configs[0],
                n_dictionaries=len(configs),
                tokens_per_batch=extra["widest_batch_positions"],
                n_backbones=2,
                backbone_parameters=6.74e9,
            ),
            "at_the_token_cap": memory_arithmetic(
                configs[0],
                n_dictionaries=len(configs),
                tokens_per_batch=args.batch_size * args.max_tokens,
                n_backbones=2,
                backbone_parameters=6.74e9,
            ),
            "realised_widest_batch_positions": extra["widest_batch_positions"],
            "realised_mean_batch_positions": extra["mean_batch_positions"],
            "token_cap_batch_positions": args.batch_size * args.max_tokens,
        },
        "limitations": limitations,
    }
    if len(models) == 2 and set(args.pairings) == set(cc.PAIRINGS):
        payload["null_comparison"] = null_comparison(
            fitted["true"],
            fitted["shuffled"],
            sites=args.layers,
            effective_dimension=effective_dimension,
        )

    cc.assert_per_layer_fields(payload, n_sites=len(args.layers))
    cc.assert_required_per_site_fields(payload)

    destination = args.out / (
        "crosscoder__"
        + re.sub(r"[^A-Za-z0-9._-]+", "-", base_path.name)
        + "__to__"
        + re.sub(r"[^A-Za-z0-9._-]+", "-", adapted_path.name)
        + f"__{args.mode}__{args.tensor}.json"
    )
    write_json(destination, payload)
    print()
    for pairing, block in fitted.items():
        for entry in block["readout"]["site_per_site"]:
            if not entry["admissible"]:
                continue
            counts = entry["counts"]
            print(
                f"[{pairing:8s} L{entry['layer']:2d}] live {entry['n_live']:6d}  "
                f"base {counts['base_specific']:6d}  shared {counts['shared']:6d}  "
                f"adapted {counts['adapted_specific']:6d}  "
                f"interm {counts['intermediate']:6d}"
            )
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
