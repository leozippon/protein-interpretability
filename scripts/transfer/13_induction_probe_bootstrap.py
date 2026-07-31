#!/usr/bin/env python3
"""Probe-cluster bootstrap of the induction head fraction.

The census averages each head's prefix-matching score over probes before it
writes, so the stored artefacts cannot support a bootstrap over probes: the
per-probe contributions are gone.  This script recomputes them -- the same
forward passes, the same probes, the same seed -- while retaining the per-probe
axis, and then resamples PROBES as clusters.

Three sampling units are conflated in casual talk about "bootstrapping the
census", and only one of them is a sampling unit:

- **Probes are.**  A different draw of 16 synthetic probes would give slightly
  different head scores, and that is real, quantifiable noise.  Probes are
  therefore the resampling cluster: a resample redraws whole probes with
  replacement and recomputes each head's mean over the redrawn set, which
  preserves the within-probe correlation across all heads of the model.
- **Layers are not.**  A layer is a coordinate of the model, not a draw from a
  population of layers.  Resampling layers would answer a question about a
  hypothetical model with a randomly chosen subset of its own depth.
- **Heads are not, either.**  Within one model the heads are the entire
  population.  There is no superpopulation of GPT-2-large heads that these 720
  were sampled from, so an interval around a within-model head fraction would be
  a confidence statement with no estimand behind it.  The interval reported here
  is exclusively probe-sampling noise, and the field is named to say so.

The cross-modality contrast is therefore NOT bootstrapped.  Its unit is the
model: four text arms against three protein arms, handled by the exact
permutation test in ``12_induction_robustness.py``, where the smallest attainable
p-value is 1/C(7,3) = 0.0286 and is reported alongside the result.

Cheap: 16 probes of 129 tokens per arm, one forward pass each with attentions
retained.  Runs on one L20.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer.io import write_json  # noqa: E402
from src.transfer.arms import (  # noqa: E402
    DEFAULT_CORPUS_DRAW_SEED,
    PANEL,
    Arm,
    load_arm,
    protein_cohort,
    text_cohort,
)
from src.transfer.circuits import (  # noqa: E402
    INDUCTION_THRESHOLDS,
    RepeatProbe,
    _pad_probe_batch,
    fit_unigram,
    n_head,
    synthetic_repeat_probes,
)
from src.transfer.induction_robustness import (  # noqa: E402
    SCHEMA_VERSION,
    cluster_bootstrap_fraction,
    contrast_ratio_bootstrap,
    threshold_label,
)

DEFAULT_OUTPUT = REPO_ROOT / "results/transfer/induction_robustness"

#: The eleven arms with a census.  Ordered text first, and within text by
#: lineage, so the printed table matches the manuscript's.
DEFAULT_ARMS = (
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "dialogpt-small",
    "qwen2.5-0.5b",
    "llama-3.2-3b",
    "protgpt2",
    "progen2-base",
    "progen2-medium",
    "zymctrl",
)

#: Contrasts reported with an interval.  Each names a pair and what is held fixed
#: between them, so that a ratio is never quoted without the design that produced
#: it.  The corpus pairs are the important ones: they bound how much of a
#: cross-modality difference is the training data rather than the modality, and
#: on the induction axis specifically that bound is the whole question, because
#: an induction head is only useful on a corpus that repeats.
CONTRASTS: dict[str, dict[str, str]] = {
    "corpus_text": {
        "high": "gpt2",
        "low": "dialogpt-small",
        "held_fixed": "architecture, 124M parameters, 12x768, GPT-2 BPE",
        "varies": "WebText against conversational Reddit",
    },
    "corpus_protein": {
        "high": "progen2-medium",
        "low": "progen2-base",
        "held_fixed": "architecture, 765M parameters, 27x1536, residue tokenisation",
        "varies": "UniRef90+BFD against the ProGen2-base mixture",
    },
    "lineage_qwen_at_scale": {
        "high": "gpt2-medium",
        "low": "qwen2.5-0.5b",
        "held_fixed": "24 layers, 355M against 494M parameters",
        "varies": "GPT-2 lineage and WebText against Qwen2 lineage and its own mixture",
    },
    "lineage_llama_at_scale": {
        "high": "gpt2-xl",
        "low": "llama-3.2-3b",
        "held_fixed": "nearest available scale, 1.56B against 3.21B parameters",
        "varies": "GPT-2 lineage and WebText against Llama lineage and its own corpus",
    },
    "modality_matched_pair": {
        "high": "gpt2-large",
        "low": "protgpt2",
        "held_fixed": "architecture, 773,891,840 parameters, 36x1280, 50257 vocabulary",
        "varies": "modality AND corpus AND corpus repeat prevalence, jointly",
    },
    "modality_scale_inverted": {
        "high": "dialogpt-small",
        "low": "protgpt2",
        "held_fixed": "nothing; the text arm is 6.2x SMALLER",
        "varies": "modality, with the scale gradient pointing the other way",
    },
}


@torch.no_grad()
def per_probe_prefix_matching(
    arm: Arm,
    probes: list[RepeatProbe],
    *,
    batch_size: int,
) -> np.ndarray:
    """Per-probe, per-head prefix-matching scores, shape (probe, layer, head).

    Identical arithmetic to ``circuits.attention_alignment_scores`` except that
    the probe axis is kept instead of being summed away.  Each probe's entry is
    the mean over that probe's own scored query positions, so the census
    statistic is recovered as a weighted mean over probes -- and because the
    synthetic probes all carry the same number of scored positions, the plain
    mean over probes reproduces the census exactly.  That equality is asserted by
    the caller rather than assumed.
    """

    arm.require("circuits")
    heads = n_head(arm)
    layers = arm.spec.n_layer
    out = np.zeros((len(probes), layers, heads), dtype=np.float64)
    for begin in range(0, len(probes), batch_size):
        chunk = probes[begin : begin + batch_size]
        ids, mask = _pad_probe_batch(arm, chunk)
        output = arm.model(
            input_ids=ids.to(arm.device),
            attention_mask=mask.to(arm.device),
            output_attentions=True,
            use_cache=False,
        )
        attentions = output.attentions
        if len(attentions) != layers or any(item is None for item in attentions):
            raise RuntimeError(
                f"{arm.name}: attention weights unavailable; load with attn_implementation='eager'"
            )
        for layer, pattern in enumerate(attentions):
            if pattern.shape[1] != heads:
                raise RuntimeError(f"{arm.name}: layer {layer} returned {pattern.shape[1]} heads")
            for row, probe in enumerate(chunk):
                query = torch.tensor(probe.query_positions, device=arm.device)
                key = torch.tensor(probe.key_positions, device=arm.device)
                block = pattern[row].float()
                out[begin + row, layer] = (
                    block[:, query, key].mean(dim=1).cpu().numpy()
                )
        del output, attentions
    return out


def build_cohort(modality: str, args: argparse.Namespace):
    """The analysis cohort ``04_circuit_primitives.py`` builds, with its own defaults.

    Reproduced call-for-call rather than imported, because that script builds it
    inside a function that also builds two repeat cohorts by scanning the corpus
    for internal repeats -- minutes of CPU this analysis has no use for.  The
    arguments here match its defaults exactly, which is what makes the
    recomputed census reproduce the stored one; ``compare_with_stored`` checks
    that it does rather than trusting this comment.

    ``--cohort-draw-seed`` is one of those defaults and it is the one that moved.
    ``04`` now draws its analysis cohort under a seeded permutation of the whole
    corpus (EXP-R2-068), so this stage's default follows it. A census stored
    *before* that change was measured on a file-order prefix, and re-analysing it
    here requires ``--cohort-draw-seed 0``. The mismatch is not silent: the two
    cohorts have different digests, the recomputed census does not reproduce the
    stored one, and ``compare_with_stored`` raises.
    """

    draw_seed = args.cohort_draw_seed or None
    if modality == "text":
        return text_cohort(args.cohort_size, min_chars=args.text_min_chars, seed=draw_seed)
    if modality == "protein":
        return protein_cohort(
            args.cohort_size,
            args.protein_min_len,
            args.protein_max_len,
            with_ec=True,
            name="swissprot_ec_long",
            seed=draw_seed,
        )
    raise ValueError(f"unsupported modality {modality!r}")


#: Per-probe score matrices, keyed by arm, retained across ``run_arm`` calls so
#: the contrasts can be bootstrapped without a second pass over the GPU.
per_probe_store: dict[str, np.ndarray] = {}


def run_arm(name: str, args: argparse.Namespace, cohorts: dict[str, Any]) -> dict[str, Any]:
    spec = PANEL[name]
    started = time.time()
    arm = load_arm(
        name,
        device=args.device,
        dtype=args.dtype,
        attn_implementation="eager",
    )
    if arm.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(arm.device)
    try:
        cohort = cohorts[spec.modality]
        strings = cohort.input_strings(arm)
        unigram = fit_unigram(arm, strings, max_tokens=args.unigram_max_tokens)
        ec_label = None
        if spec.input_format == "ec_conditioned":
            labels = cohort.metadata.get("ec_labels")
            if not labels:
                raise ValueError(f"{name}: analysis cohort carries no EC labels")
            ec_label = labels[0]
        probes = synthetic_repeat_probes(
            arm,
            unigram,
            n_probes=args.synthetic_probes,
            copy_len=args.synthetic_copy_len,
            seed=args.seed,
            ec_label=ec_label,
        )
        scores = per_probe_prefix_matching(arm, probes, batch_size=args.probe_batch_size)
        peak = int(torch.cuda.max_memory_allocated(arm.device)) if arm.device.startswith("cuda") else 0
    finally:
        del arm
        torch.cuda.empty_cache()

    per_probe_store[name] = scores
    census_matrix = scores.mean(axis=0)
    # The contrast is published at ``--contrast-threshold`` and the per-arm
    # intervals were computed at the census ladder, so a run at any other cut
    # reported a ratio at a threshold no arm carried an interval for. Unioned in,
    # the same remedy ``circuits._threshold_sweep`` and
    # ``induction_robustness.threshold_sweep`` carry: Appendix B rule 17.
    thresholds = sorted(set(INDUCTION_THRESHOLDS) | {float(args.contrast_threshold)})
    intervals = {
        threshold_label(threshold): cluster_bootstrap_fraction(
            scores,
            threshold=threshold,
            resamples=args.resamples,
            seed=args.seed + 1,
        )
        for threshold in thresholds
    }
    return {
        "arm": name,
        "modality": spec.modality,
        "n_layer": spec.n_layer,
        "n_heads": int(census_matrix.size),
        "n_probes": len(probes),
        "probe": "synthetic_repeat",
        "thresholds": thresholds,
        "census_ladder_thresholds": list(INDUCTION_THRESHOLDS),
        "census_fraction_recomputed": {
            threshold_label(threshold): float(
                (census_matrix.reshape(-1) >= threshold).sum()
            )
            / census_matrix.size
            for threshold in thresholds
        },
        "probe_cluster_bootstrap": intervals,
        "runtime_seconds": round(time.time() - started, 2),
        "peak_gpu_bytes": peak,
    }


#: How far a recomputed census fraction may sit from the stored one before the
#: recomputation is not a reproduction.
#:
#: The recomputation is the *same* forward passes, the same probes and the same
#: seed, so the only sources of difference are non-deterministic kernel
#: reduction order and dtype rounding. A head fraction is a count over hundreds
#: of heads, so a threshold-straddling head moves it by ~1/n_heads: 0.0014 for
#: gpt2-large's 720. This floor admits a couple of such heads and refuses
#: anything that could be a different cohort, a different probe set or a
#: different checkpoint.
MAX_CENSUS_REPRODUCTION_DIFFERENCE = 0.01


def compare_with_stored(rows: dict[str, dict[str, Any]], stored_path: Path) -> dict[str, Any]:
    """The recomputed census must reproduce the stored one, or nothing below holds.

    That sentence was the docstring before EXP-R2-067 and it was not true of the
    code: the function computed ``max_absolute_fraction_difference``, returned
    it, and nothing anywhere compared it to anything. An arbitrarily large
    disagreement produced a number in the artefact and an exit status of zero,
    which is a control that is measured, reported and never gated -- the shape
    this programme has retracted a result over. It now raises.
    """

    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    by_arm = {entry["name"]: entry for entry in stored["arms"]}
    checks: list[dict[str, Any]] = []
    for name, row in rows.items():
        if name not in by_arm:
            checks.append({"arm": name, "checked": False, "reason": "absent from the stored run"})
            continue
        stored_counts = by_arm[name]["stored_counts"]
        n_heads = by_arm[name]["n_heads"]
        deltas = {
            label: row["census_fraction_recomputed"][label] - stored_counts[label] / n_heads
            for label in stored_counts
        }
        checks.append(
            {
                "arm": name,
                "checked": True,
                "max_absolute_fraction_difference": max(abs(v) for v in deltas.values()),
                "per_threshold_difference": deltas,
            }
        )
    diverged = [
        check
        for check in checks
        if check["checked"]
        and check["max_absolute_fraction_difference"] > MAX_CENSUS_REPRODUCTION_DIFFERENCE
    ]
    if diverged:
        raise RuntimeError(
            "the recomputed census does not reproduce "
            f"{stored_path}: "
            + "; ".join(
                f"{check['arm']} differs by {check['max_absolute_fraction_difference']:.4f}"
                for check in diverged
            )
            + f" (floor {MAX_CENSUS_REPRODUCTION_DIFFERENCE}). The probe-cluster "
            "intervals below resample these recomputed scores, so they would be "
            "intervals around a different census than the one they are reported "
            "against."
        )
    return {
        "stored_source": str(stored_path),
        "checks": checks,
        "max_reproduction_difference_allowed": MAX_CENSUS_REPRODUCTION_DIFFERENCE,
        "reproduces_within_floor": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS), choices=sorted(PANEL))
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--cohort-size", type=int, default=24)
    parser.add_argument(
        "--cohort-draw-seed",
        type=int,
        default=DEFAULT_CORPUS_DRAW_SEED,
        help="must match the draw the stored census was measured on; 0 selects "
        "the file-order prefix that every census stored before EXP-R2-068 used. "
        "A mismatch is caught by compare_with_stored, not tolerated",
    )
    parser.add_argument("--protein-min-len", type=int, default=600)
    parser.add_argument("--protein-max-len", type=int, default=1000)
    parser.add_argument("--text-min-chars", type=int, default=3000)
    parser.add_argument("--unigram-max-tokens", type=int, default=256)
    parser.add_argument("--synthetic-probes", type=int, default=16)
    parser.add_argument("--synthetic-copy-len", type=int, default=64)
    parser.add_argument("--probe-batch-size", type=int, default=4)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--contrast-threshold", type=float, default=0.10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--compare-with",
        type=Path,
        default=DEFAULT_OUTPUT / "induction_robustness_synthetic_repeat.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"{args.device} was requested but no CUDA device is present")
    torch.manual_seed(args.seed)
    modalities = {PANEL[name].modality for name in args.arms}
    cohorts = {modality: build_cohort(modality, args) for modality in sorted(modalities)}
    rows: dict[str, dict[str, Any]] = {}
    headline = threshold_label(args.contrast_threshold)
    for name in args.arms:
        rows[name] = run_arm(name, args, cohorts)
        entry = rows[name]
        interval = entry["probe_cluster_bootstrap"][headline]
        print(
            f"[{name}] fraction at {headline} {interval['point_estimate']:.4f} "
            f"[{interval['interval'][0]:.4f}, {interval['interval'][1]:.4f}] "
            f"over {interval['n_probe_clusters']} probe clusters and "
            f"{interval['n_distinct_resampled_fractions']} distinct resample values, "
            f"{entry['runtime_seconds']}s",
            flush=True,
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "resampling_unit": "probe",
        "not_resampled": ["layer", "head", "model"],
        "why": (
            "a layer is a coordinate of the model and a model's heads are its "
            "entire population, so neither carries sampling uncertainty; the "
            "cross-modality contrast takes the model as its unit and is tested "
            "by exact permutation over 4 text against 3 protein arms elsewhere"
        ),
        "configuration": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "arms": rows,
        "contrasts": {
            label: (
                {
                    **spec,
                    **contrast_ratio_bootstrap(
                        per_probe_store[spec["high"]],
                        per_probe_store[spec["low"]],
                        threshold=args.contrast_threshold,
                        resamples=args.resamples,
                        seed=args.seed + 2,
                    ),
                }
                if spec["high"] in per_probe_store and spec["low"] in per_probe_store
                else {**spec, "available": False}
            )
            for label, spec in CONTRASTS.items()
        },
        "contrast_note": (
            "the corpus rows bound how much of a cross-modality difference is the "
            "training data rather than the modality; the modality rows vary corpus "
            "and modality together and cannot separate them, because no protein "
            "decoder trained on a repeat-rich corpus exists"
        ),
        # A missing comparison file is recorded as "not attempted", never as a
        # bare null that reads the same as "attempted and found nothing to say".
        "reproduces_stored_census": (
            compare_with_stored(rows, args.compare_with)
            if args.compare_with.exists()
            else {
                "stored_source": str(args.compare_with),
                "checked": False,
                "reason": (
                    "the stored census named by --compare-with does not exist, so this "
                    "run's recomputation was not verified against any earlier one"
                ),
            }
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "induction_probe_bootstrap.json"
    write_json(path, payload)
    print(f"wrote {path}", flush=True)
    print(
        "\ncontrast | fractions | ratio | 95% probe-cluster interval | atoms | "
        "status | held fixed"
    )
    for label, row in payload["contrasts"].items():
        if row.get("available") is False:
            print(f"{label}: unavailable")
            continue
        interval = row["interval"]
        ratio = row["ratio"]
        print(
            f"{label} ({row['high']}/{row['low']}) | "
            f"{row['fraction_high']:.4f}/{row['fraction_low']:.4f} | "
            f"{'undefined' if ratio is None else f'{ratio:.2f}x'} | "
            f"{'n/a' if interval is None else f'[{interval[0]:.4f}, {interval[1]:.4f}]'} | "
            f"{row['n_distinct_resampled_ratios']} | "
            f"{row['interval_status']} | "
            f"{row['held_fixed']}"
        )


if __name__ == "__main__":
    main()
