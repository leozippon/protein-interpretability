#!/usr/bin/env python3
"""Threshold robustness and scale/modality separation for the induction census.

Reads the census artefacts already on disk -- no model is loaded and no GPU is
touched -- and answers the two objections to the programme's one surviving
finding:

1. the 0.10 cut-off is arbitrary, so the ordering is recomputed at every
   threshold the census emits and, more importantly, with rank statistics that
   use no cut-off at all;
2. scale and modality are confounded, so a modality indicator is fitted against
   each scale covariate in turn with the collinearity between them reported
   beside every coefficient, and the two model pairs that settle the question
   without any fit are computed directly.

The per-probe cluster bootstrap is NOT run here: it needs per-probe scores,
which the stored artefacts do not carry because the census averages over probes
before writing.  ``13_induction_probe_bootstrap.py`` recomputes those on a GPU.

Parameter counts are read from the checkpoint headers rather than declared, so
that a mis-copied figure cannot enter the scale axis.  Buffers -- GPT-2's causal
mask, rotary inverse frequencies -- and a tied ``lm_head`` duplicate are excluded,
which is what makes the counts agree with the published figures (GPT-2-large
773,891,840; DialoGPT-small 124,412,160).
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer.io import write_json  # noqa: E402
from src.transfer.arms import PANEL  # noqa: E402
from src.transfer.induction_robustness import (  # noqa: E402
    PRIMARY_PROBE,
    PROBES,
    SCALE_COVARIATES,
    SCHEMA_VERSION,
    ArmCensus,
    collinearity_verdict,
    corpus_contrast,
    lineage_scale_ladder,
    load_census,
    model_level_exact_test,
    pairwise_auc,
    pairwise_ks,
    quantile_dominance,
    scale_inversion_checks,
    scale_modality_fit,
    score_quantiles,
    survival_dominance,
    threshold_sweep,
    variance_decomposition,
    verify_against_stored_counts,
)

TRANSFER_RESULTS = REPO_ROOT / "results/transfer_20260728"

#: Which artefact each arm's census is read from.  The four text arms were
#: measured in the ``circuit_primitives_text_control`` run and the three protein
#: arms in ``circuit_primitives_approximate``; GPT-2-large appears in both and
#: the two agree exactly on the synthetic probe (108/70/35/21 heads above the
#: four fixed cuts), which is checked rather than assumed by
#: ``--cross-check-arm``.
DEFAULT_SOURCES: dict[str, Path] = {
    "gpt2-large": TRANSFER_RESULTS / "circuit_primitives_text_control/gpt2-large.json",
    "qwen2.5-0.5b": TRANSFER_RESULTS / "circuit_primitives_text_control/qwen2.5-0.5b.json",
    "dialogpt-small": TRANSFER_RESULTS / "circuit_primitives_text_control/dialogpt-small.json",
    "llama-3.2-3b": TRANSFER_RESULTS / "circuit_primitives_text_control/llama-3.2-3b.json",
    "protgpt2": TRANSFER_RESULTS / "circuit_primitives_approximate/protgpt2.json",
    "progen2-medium": TRANSFER_RESULTS / "circuit_primitives_approximate/progen2-medium.json",
    "zymctrl": TRANSFER_RESULTS / "circuit_primitives_approximate/zymctrl.json",
}

#: Arms that identify the scale slope and the corpus effect, measured at exactly
#: the settings ``DEFAULT_SOURCES`` were measured at -- same seed, same cohort
#: digests, same probe count, same dtype -- because a rung measured differently
#: cannot enter the same table as the rungs it is there to calibrate.  Held apart
#: from the seven-arm panel so that the headline table stays what it was and the
#: ladder is visibly an addition rather than a redefinition.
LADDER_SOURCES: dict[str, Path] = {
    "gpt2": TRANSFER_RESULTS / "circuit_primitives_text_ladder/gpt2.json",
    "gpt2-medium": TRANSFER_RESULTS / "circuit_primitives_text_ladder/gpt2-medium.json",
    "gpt2-xl": TRANSFER_RESULTS / "circuit_primitives_text_ladder/gpt2-xl.json",
    "progen2-base": TRANSFER_RESULTS / "circuit_primitives_text_ladder/progen2-base.json",
}

#: Pairs identical in architecture, parameter count and tokeniser, differing only
#: in pretraining corpus.  ``arms.py`` names them ``TEXT_DATA_CONTRAST`` and
#: ``MATCHED_DATA_CONTRAST``.
CORPUS_PAIRS: dict[str, tuple[str, str]] = {
    "text": ("gpt2", "dialogpt-small"),
    "protein": ("progen2-base", "progen2-medium"),
}

#: The same arm measured in a second run, used as a reproducibility check on the
#: numbers every conclusion below rests on.
CROSS_CHECK: dict[str, Path] = {
    "gpt2-large": TRANSFER_RESULTS / "circuit_primitives/gpt2-large.json",
    "protgpt2": TRANSFER_RESULTS / "circuit_primitives/protgpt2.json",
    "progen2-medium": TRANSFER_RESULTS / "circuit_primitives/progen2-medium.json",
    "zymctrl": TRANSFER_RESULTS / "circuit_primitives/zymctrl.json",
}

DEFAULT_OUTPUT = TRANSFER_RESULTS / "induction_robustness"

#: Tensors that are buffers rather than parameters.  Counting GPT-2's causal mask
#: would add 37.9 million to a 774 million figure and put the arm in the wrong
#: place on a log-parameter axis.
_BUFFER = re.compile(r"(attn\.bias|masked_bias|inv_freq|causal_mask|rotary_emb)")
_EMBEDDING = re.compile(r"(wte\.weight|wpe\.weight|embed_tokens\.weight|lm_head\.weight|embed_out\.weight)")


def tensor_shapes(directory: Path) -> dict[str, tuple[int, ...]]:
    """Every tensor's shape, from safetensors headers or a torch state dict."""

    shapes: dict[str, tuple[int, ...]] = {}
    safetensors = sorted(glob.glob(str(directory / "*.safetensors")))
    if safetensors:
        for path in safetensors:
            with open(path, "rb") as handle:
                length = struct.unpack("<Q", handle.read(8))[0]
                header = json.loads(handle.read(length))
            for key, value in header.items():
                if key == "__metadata__":
                    continue
                shapes[key] = tuple(value["shape"])
        return shapes
    import torch

    binaries = sorted(glob.glob(str(directory / "*.bin")))
    if not binaries:
        raise FileNotFoundError(f"{directory}: no safetensors or .bin weights")
    for path in binaries:
        state = torch.load(path, map_location="cpu", weights_only=True)
        for key, value in state.items():
            shapes[key] = tuple(value.shape)
    return shapes


def count_parameters(directory: Path) -> dict[str, int]:
    """Total and non-embedding parameter counts, buffers and tied copies removed."""

    shapes = tensor_shapes(directory)
    total = 0
    embedding = 0
    dropped = 0
    tied_shapes = {
        shape for key, shape in shapes.items() if key.endswith(("wte.weight", "embed_tokens.weight"))
    }
    for key, shape in sorted(shapes.items()):
        count = 1
        for dimension in shape:
            count *= int(dimension)
        if _BUFFER.search(key):
            dropped += count
            continue
        if key.endswith("lm_head.weight") and shape in tied_shapes:
            dropped += count
            continue
        total += count
        if _EMBEDDING.search(key):
            embedding += count
    return {
        "total": total,
        "non_embedding": total - embedding,
        "embedding": embedding,
        "excluded_buffers_and_tied_copies": dropped,
    }


def cross_check(name: str, primary: ArmCensus, other: Path, probe: str) -> dict[str, Any]:
    """Two independent runs of the same arm must give the same head matrix."""

    replicate = load_census(other, parameters=primary.parameters, probe=probe)
    delta = np.abs(primary.scores - replicate.scores)
    return {
        "arm": name,
        "primary_source": str(primary.source),
        "replicate_source": str(other),
        "max_absolute_head_score_difference": float(delta.max()),
        "fraction_primary": primary.fraction_above(0.10),
        "fraction_replicate": replicate.fraction_above(0.10),
        "counts_identical": primary.stored_counts == replicate.stored_counts,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    parameters: dict[str, dict[str, int]] = {}
    arms: list[ArmCensus] = []
    verification: list[dict[str, Any]] = []
    for name, path in DEFAULT_SOURCES.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: census artefact missing at {path}")
        parameters[name] = count_parameters(Path(PANEL[name].path))
        arm = load_census(path, parameters=parameters[name]["total"], probe=args.probe)
        verification.append(verify_against_stored_counts(arm))
        arms.append(arm)

    # The v1 artefacts carry only the synthetic probe, so a natural-probe run has
    # no replicate to check against. That is recorded rather than silently
    # skipped: "no cross-check was possible" and "the cross-check passed" are
    # different statements about the same field.
    replicates: list[dict[str, Any]] = []
    for name, path in CROSS_CHECK.items():
        if not any(a.name == name for a in arms):
            # Not in this invocation's arm list; there is nothing to cross-check
            # against, and no claim is made about the arm either way.
            continue
        if not path.exists():
            # The branch below records "the replicate exists but carries no
            # census for this probe". This one records "there is no replicate",
            # which the comment above says is a different statement and which
            # used to be a bare `continue` -- so the field was simply shorter and
            # a reader could not tell an absent replicate from an unattempted one.
            replicates.append(
                {
                    "arm": name,
                    "replicate_source": str(path),
                    "available": False,
                    "reason": "the replicate artefact does not exist",
                }
            )
            continue
        replicate_probes = json.loads(path.read_text(encoding="utf-8"))["induction"]
        if args.probe not in replicate_probes:
            replicates.append(
                {
                    "arm": name,
                    "replicate_source": str(path),
                    "available": False,
                    "reason": f"the replicate artefact carries no {args.probe} census",
                }
            )
            continue
        entry = cross_check(name, next(a for a in arms if a.name == name), path, args.probe)
        entry["available"] = True
        replicates.append(entry)

    ladder_arms: list[ArmCensus] = []
    for name, path in LADDER_SOURCES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{name}: ladder census missing at {path}; run 04_circuit_primitives.py "
                "on the lineage rungs at the settings the panel was measured at"
            )
        parameters[name] = count_parameters(Path(PANEL[name].path))
        arm = load_census(path, parameters=parameters[name]["total"], probe=args.probe)
        verification.append(verify_against_stored_counts(arm))
        ladder_arms.append(arm)
    extended = arms + ladder_arms

    sweep = threshold_sweep(arms)
    fits = {
        covariate: scale_modality_fit(arms, threshold=args.threshold, covariate=covariate)
        for covariate in SCALE_COVARIATES
    }

    fractions = {arm.name: arm.fraction_above(args.threshold) for arm in arms}
    median_scores = {arm.name: float(np.median(arm.flat_over_uniform)) for arm in arms}
    q99_scores = {arm.name: float(np.quantile(arm.flat_over_uniform, 0.99)) for arm in arms}
    mean_scores = {arm.name: float(arm.flat_over_uniform.mean()) for arm in arms}

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "probe": args.probe,
        "headline_threshold": args.threshold,
        "sources": {name: str(path) for name, path in DEFAULT_SOURCES.items()},
        "parameters": parameters,
        "arms": [
            {
                "name": arm.name,
                "modality": arm.modality,
                "architecture": arm.architecture,
                "n_layer": arm.n_layer,
                "n_head_per_layer": arm.n_head_per_layer,
                "n_heads": arm.n_heads,
                "d_model": arm.d_model,
                "parameters": arm.parameters,
                "n_probes": arm.n_probes,
                "uniform_baseline": arm.uniform_baseline,
                "stored_counts": dict(arm.stored_counts),
                "data_driven_threshold": arm.data_driven_threshold,
                "quantiles_raw": score_quantiles(arm),
                "quantiles_over_uniform": score_quantiles(arm, over_uniform=True),
            }
            for arm in arms
        ],
        "per_head_matrix_agrees_with_stored_census": verification,
        "reproducibility_cross_checks": replicates,
        "threshold_sweep": sweep,
        "rank_statistics": {
            "raw": pairwise_auc(arms, over_uniform=False),
            "over_uniform": pairwise_auc(arms, over_uniform=True),
        },
        "tail_statistics": {
            "survival_dominance": survival_dominance(arms),
            "quantile_dominance": quantile_dominance(arms),
            "pairwise_one_sided_ks": pairwise_ks(arms),
        },
        "model_level_tests": {
            "fraction_above_threshold": model_level_exact_test(
                arms, statistic=f"fraction_above_{args.threshold:.2f}", values=fractions
            ),
            "mean_head_score_over_uniform": model_level_exact_test(
                arms, statistic="mean_head_score_over_uniform", values=mean_scores
            ),
            "median_head_score_over_uniform": model_level_exact_test(
                arms, statistic="median_head_score_over_uniform", values=median_scores
            ),
            "q99_head_score_over_uniform": model_level_exact_test(
                arms, statistic="q99_head_score_over_uniform", values=q99_scores
            ),
        },
        "scale_modality_fits": fits,
        "scale_modality_verdict": collinearity_verdict(fits),
        "scale_inversion": scale_inversion_checks(arms, threshold=args.threshold),
        "within_lineage_ladder": lineage_scale_ladder(extended, threshold=args.threshold),
        "variance_decomposition": variance_decomposition(extended, threshold=args.threshold),
        "corpus_contrast": corpus_contrast(
            extended, threshold=args.threshold, pairs=CORPUS_PAIRS
        ),
        "extended_panel_fits": {
            covariate: scale_modality_fit(
                extended, threshold=args.threshold, covariate=covariate
            )
            for covariate in SCALE_COVARIATES
        },
        "extended_panel_arms": [
            {
                "name": arm.name,
                "modality": arm.modality,
                "parameters": arm.parameters,
                "n_layer": arm.n_layer,
                "n_heads": arm.n_heads,
                "d_model": arm.d_model,
                "count_above_threshold": int((arm.flat >= args.threshold).sum()),
                "fraction": arm.fraction_above(args.threshold),
            }
            for arm in extended
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", default=PRIMARY_PROBE, choices=list(PROBES))
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"induction_robustness_{args.probe}.json"
    write_json(path, payload)
    print(f"wrote {path}", flush=True)

    sweep = payload["threshold_sweep"]
    print("\nthreshold sweep (fraction of heads above cut)")
    header = ["threshold"] + [a["name"] for a in payload["arms"]] + ["worst_text/best_protein"]
    print(" | ".join(header))
    for row in sweep["rows"]:
        cells = [row["threshold"]]
        for a in payload["arms"]:
            cells.append(f"{row['fractions'][a['name']]:.4f}")
        ratio = row["worst_text_over_best_protein"]
        cells.append("inf" if ratio is None else f"{ratio:.3f}")
        print(" | ".join(cells))
    print(
        "\nseparation holds at every threshold: "
        f"{sweep['separation_holds_at_every_threshold']}; breaks at "
        f"{sweep['thresholds_where_separation_breaks']}"
    )
    for scale, block in payload["rank_statistics"].items():
        print(
            f"\nAUC ({scale}): pooled {block['pooled_auc']:.4f}; "
            f"pairwise min {block['min_pairwise_auc']:.4f} max {block['max_pairwise_auc']:.4f}; "
            f"{block['pairs_above_half']}/{block['n_pairs']} pairs above 0.5"
        )
    tail = payload["tail_statistics"]
    print("\nquantile dominance (x uniform baseline)")
    print("quantile | worst_text | best_protein | ratio | separates")
    for row in tail["quantile_dominance"]["rows"]:
        ratio = row["ratio"]
        print(
            f"{row['quantile']:.3f} | {row['worst_text']:.3f} | {row['best_protein']:.3f} | "
            f"{'n/a' if ratio is None else f'{ratio:.3f}'} | {row['separates']}"
        )
    survival = tail["survival_dominance"]
    print(
        "\nsurvival dominance: separation holds on "
        f"{survival['fraction_of_informative_grid_where_separation_holds']:.4f} of the "
        f"informative cut grid; widest separating interval "
        f"{survival['widest_separating_interval']}; largest ratio "
        f"{survival['largest_ratio']}"
    )
    ks = tail["pairwise_one_sided_ks"]
    print(
        f"\none-sided KS: D+ in [{ks['min_d_plus']:.4f}, {ks['max_d_plus']:.4f}]; "
        f"{ks['pairs_where_text_stochastically_dominates']}/{ks['n_pairs']} pairs "
        "with text stochastically dominating throughout"
    )
    ladder = payload["within_lineage_ladder"]
    print("\nwithin-lineage GPT-2 ladder (architecture, tokeniser and corpus held fixed)")
    print("arm | parameters | n_layer | n_heads | count | fraction")
    for rung in ladder["rungs"]:
        print(
            f"{rung['arm']} | {rung['parameters']:,} | {rung['n_layer']} | "
            f"{rung['n_heads']} | {rung['count_above_threshold']} | {rung['fraction']:.4f}"
        )
    print(
        f"log10 slope per decade of parameters: {ladder['log10_slope_per_decade']:+.4f} "
        f"[{ladder['slope_interval'][0]:+.4f}, {ladder['slope_interval'][1]:+.4f}] "
        f"-> {ladder['direction']}"
    )
    print(f"verdict: {ladder['verdict']}")
    print("\nscale-matched prediction against observation")
    print("arm | modality | predicted | prediction interval | observed | shortfall | below PI")
    for name, row in sorted(ladder["predictions"].items()):
        shortfall = row["shortfall_ratio"]
        print(
            f"{name} | {row['modality']} | {row['predicted_fraction']:.4f} | "
            f"[{row['prediction_interval'][0]:.4f}, {row['prediction_interval'][1]:.4f}] | "
            f"{row['observed_fraction']:.4f} | "
            f"{'inf' if shortfall is None else f'{shortfall:.2f}x'} | "
            f"{row['observed_below_prediction_interval']}"
        )
    print("\ncorpus-only contrasts (architecture, size and tokeniser held fixed)")
    for label, row in payload["corpus_contrast"]["pairs"].items():
        if not row.get("available"):
            print(f"{label}: unavailable {row.get('missing')}")
            continue
        ratio = row["ratio"]
        print(
            f"{label}: {row['higher']['arm']} {row['higher']['fraction']:.4f} vs "
            f"{row['lower']['arm']} {row['lower']['fraction']:.4f} -> "
            f"{'inf' if ratio is None else f'{ratio:.2f}x'} "
            f"(shape match {row['shape_match']}, parameter ratio {row['parameter_ratio']:.3f})"
        )
    decomposition = payload["variance_decomposition"]
    print(
        f"\nvariance decomposition on log10(fraction), n={decomposition['n_arms']} arms "
        f"(dropped at zero: {decomposition['dropped_at_zero']})"
    )
    for label, model in decomposition["models"].items():
        print(f"  R^2 {label:26s} {model['r_squared']:.4f}  (dof {model['residual_dof']})")
    print(
        f"  increment modality | scale, lineage : "
        f"{decomposition['increment_modality_given_scale_and_lineage']:+.4f}"
    )
    print(
        f"  increment lineage  | scale, modality: "
        f"{decomposition['increment_lineage_given_scale_and_modality']:+.4f}"
    )
    print(
        f"  increment scale    | modality, lineage: "
        f"{decomposition['increment_scale_given_modality_and_lineage']:+.4f}"
    )
    print(
        f"  modality variance surviving scale+lineage: "
        f"{decomposition['modality_variance_surviving_scale_and_lineage']:.4f}"
    )
    print(f"  identification: {json.dumps(decomposition['identification'], indent=1)}")
    print("\nscale/modality:", json.dumps(payload["scale_modality_verdict"], indent=1))
    print("\nscale inversion:", json.dumps(payload["scale_inversion"]["verdict"], indent=1))


if __name__ == "__main__":
    main()
