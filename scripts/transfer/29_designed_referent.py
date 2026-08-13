#!/usr/bin/env python3
"""Does a protein decoder predict phenotype on proteins its corpus does not contain?

**What is new here, and it is the only thing that is.** F10 could never exclude
retrieval, only estimate it: over 187 ProteinGym wild types the least homologous
is still 55.5% identical to a UniRef50 cluster the model trained on. EXP-R2-190
staged a referent where retrieval is excluded *by construction* -- 132 of 148 de
novo designs from Tsuboyama et al. 2023 return no DIAMOND hit against the
60.3 M-cluster staged UniRef50 at F10's own gate, against a within-cohort natural
control from the same file, the same assay and matched length that hits at
328 of 330. Designs and naturals therefore differ in whether the corpus contains
their homologues and in nothing else that this file could confound.

**The estimand** is F10's, unchanged: per wild type, the Spearman correlation
between an arm's summed log-likelihood of a variant and that variant's measured
folding stability. ``ddG_ML`` is the phenotype, which is the column ProteinGym
itself publishes as ``DMS_score`` for its 64 Tsuboyama assays. It is a
sequence-likelihood estimand, so L31's tokenisation constraint does not bind it.

**Pre-registered before any score existed** (EXP-R2-191, and
``designed_referent.arm_verdict`` is the executable form):

``positive``      the arm's series-clustered interval on MODEL - BASELINE lies
                  wholly above zero for EVERY free and fragment-level baseline on
                  the corpus-disjoint designs, AND the identical conjunction
                  holds on the natural control.
``negative``      the design-side conjunction fails while the control passes.
                  This is a clean negative: the instrument clears its bar where
                  retrieval is available and the model does not clear it where
                  retrieval is excluded.
``uninterpretable_instrument_bound``
                  the control fails. Nothing may then be read from the design
                  side, because the gate was never shown attainable.

The baselines are the free family -- BLOSUM62 and ``profiles.free_baselines``'
position index, hydropathy pair and corpus composition -- plus two fragment-level
retrieval baselines built as proper conditional sequence models over the
**corrected** record-local corpus k-mer background. The fragment baselines are
what replaces F10's profile LOOKUP, which is empty by construction on a referent
with no homologues: a homologue-free referent is not an information-free one.

**Unit of analysis.** The design series -- topology x round, run-family x run, or
hallucination round -- as EXP-R2-190 enumerated them, resampled under
``profiles.cluster_bootstrap`` at the package's declared 8-unit floor. The
natural control's unit is the dataset's own ``WT_cluster``. Two further readings
are pre-registered rather than chosen afterwards: a strict one over the design
series that are entirely zero-hit, and a topology-level one that is expected to
fall below the floor and is reported with the refusal rather than as a number.

Stages are independent and resumable. ``cohort``, ``baseline`` and ``analyse``
are CPU-only; ``score`` is the only stage that needs a GPU, and it reads the
cohort artefact rather than the staged parquet so that the pod needs no dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer import designed_referent as D  # noqa: E402
from src.transfer.arms import (  # noqa: E402
    PANEL,
    REPO,
    Cohort,
    load_arm,
    tokenize_batch,
)
from src.transfer.io import sha256_file, write_json  # noqa: E402
from src.transfer.kmer_background import load as load_kmer_background  # noqa: E402

SCHEMA_VERSION = D.SCHEMA_VERSION
STAGES = ("cohort", "baseline", "score", "analyse")
DEFAULT_OUT = REPO / "results/transfer/designed_referent"
DEFAULT_ARMS = ("protgpt2", "progen2-small", "progen2-base", "progen2-medium")


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist; run the stage that writes it")
    return json.loads(path.read_text(encoding="utf-8"))


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------ stage: cohort


def stage_cohort(args: argparse.Namespace) -> dict[str, Any]:
    referent = D.build_referent(
        megascale_dir=args.megascale_dir,
        certificate_dir=args.certificate_dir,
        min_variants=min(D.VARIANT_FLOOR_SWEEP + (args.min_variants,)),
    )
    sweep = _floor_sweep(referent)
    referent = D.Referent(
        wildtypes=tuple(
            wt for wt in referent.wildtypes if len(wt.mutants) >= args.min_variants
        ),
        provenance={
            **referent.provenance,
            "min_variants": int(args.min_variants),
            "counts": D.cohort_counts(referent.wildtypes, min_variants=args.min_variants),
        },
    )
    path = args.out / "cohort.json"
    D.save_referent(referent, path)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": "cohort",
        "created_utc": _timestamp(),
        "cohort_sha256": sha256_file(path),
        "provenance": referent.provenance,
        "variant_floor_sweep": sweep,
    }
    write_json(args.out / "cohort_summary.json", summary)
    counts = referent.provenance["counts"]
    print(json.dumps(counts, indent=2, sort_keys=True))
    return summary


def _floor_sweep(referent: D.Referent) -> list[dict[str, Any]]:
    """The cohort at every floor in the declared sweep.

    A constant that changes nothing is worth more than an argued one: the
    MegaScale variant count per wild type is bimodal, so this sweep is expected to
    return the identical cohort at every value, and if it ever does not the
    headline floor is doing work that has to be declared. Counted by filtering the
    one built cohort rather than by rebuilding it, so the sweep cannot disagree
    with the headline about anything except the floor.
    """

    return [
        {"min_variants": floor, **D.cohort_counts(referent.wildtypes, min_variants=floor)}
        for floor in D.VARIANT_FLOOR_SWEEP
    ]


# ---------------------------------------------------------------- stage: baseline


def stage_baseline(args: argparse.Namespace) -> dict[str, Any]:
    referent = D.load_referent(args.cohort or args.out / "cohort.json")
    background = load_kmer_background(args.kmer_background_dir)
    background_record = D.require_corrected_background(background)
    print(f"[baseline] k-mer background {background_record['source']}")
    print(f"[baseline] totals {background_record['totals']}")
    log_conditional = {
        k: D.conditional_log_probabilities(background, k) for k in (3, 4)
    }
    residue_background = D.corpus_residue_background(args.retrieval_bound_dir)

    rows: dict[str, dict[str, Any]] = {}
    for index, wildtype in enumerate(referent.wildtypes):
        scores = D.baseline_scores(
            wildtype,
            residue_background=residue_background,
            log_conditional=log_conditional,
        )
        rows[wildtype.name] = {
            "kind": wildtype.kind,
            "unit": wildtype.unit,
            "n_variants": len(wildtype.mutants),
            "spearman": {
                name: D.spearman(value, wildtype.phenotype)
                for name, value in scores.items()
            },
        }
        if index % 50 == 0:
            print(f"[baseline] {index + 1}/{len(referent.wildtypes)} {wildtype.name}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": "baseline",
        "created_utc": _timestamp(),
        "cohort_sha256": sha256_file(args.cohort or args.out / "cohort.json"),
        "kmer_background": background_record,
        "residue_background_source": str(args.retrieval_bound_dir),
        "baselines": list(D.BASELINES),
        "wildtypes": rows,
    }
    write_json(args.out / "baselines.json", payload)
    return payload


# ------------------------------------------------------------------- stage: score


class _ArmLikelihood:
    """Summed log-likelihood of a sequence under one arm, in its own rendering.

    The convention is stage 20's: render through ``Cohort.input_strings`` so the
    panel decides the input format, then sum the token log-probabilities over
    every position whose target and context are both real. ProtGPT2's FASTA
    wrapping is worth 1.42 nats/token and is not this stage's decision to make.
    """

    def __init__(self, name: str, *, device: str, dtype: str, batch_size: int) -> None:
        import torch

        self.torch = torch
        self.name = name
        self.batch_size = batch_size
        self.arm = load_arm(name, device=device, dtype=dtype)
        config = self.arm.model.config
        self.context = int(
            getattr(config, "n_positions", None)
            or getattr(config, "max_position_embeddings")
        )

    def render(self, sequences: list[str]) -> list[str]:
        cohort = Cohort(
            name="designed_referent",
            kind="protein",
            records=list(sequences),
            min_symbols=min(len(s) for s in sequences),
            max_symbols=max(len(s) for s in sequences),
            metadata={},
        )
        return cohort.input_strings(self.arm)

    def token_lengths(self, texts: list[str]) -> list[int]:
        encoded = self.arm.tokenizer(list(texts), return_tensors=None)["input_ids"]
        return [len(row) for row in encoded]

    def log_likelihood(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        totals = np.empty(len(texts), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                chunk = texts[start : start + self.batch_size]
                ids, mask = tokenize_batch(self.arm, chunk, self.context)
                ids = ids.to(self.arm.device)
                mask = mask.to(self.arm.device)
                logits = self.arm.model(input_ids=ids, attention_mask=mask).logits
                logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
                targets = ids[:, 1:]
                token = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                keep = (mask[:, 1:] * mask[:, :-1]).bool()
                totals[start : start + len(chunk)] = (
                    (token * keep).sum(1).double().cpu().numpy()
                )
        return totals

    def release(self) -> None:
        del self.arm
        self.torch.cuda.empty_cache()


def stage_score(args: argparse.Namespace) -> dict[str, Any]:
    cohort_path = args.cohort or args.out / "cohort.json"
    referent = D.load_referent(cohort_path)
    digest = sha256_file(cohort_path)
    results: dict[str, Any] = {}
    for name in args.arms:
        if name in D.EXCLUDED_ARMS:
            raise KeyError(f"{name} is excluded: {D.EXCLUDED_ARMS[name]}")
        if name not in PANEL or PANEL[name].modality != "protein":
            raise KeyError(f"{name} is not a protein panel arm")
        print(f"[score] {name} over {len(referent.wildtypes)} wild types", flush=True)
        scorer = _ArmLikelihood(
            name, device=args.device, dtype=args.dtype, batch_size=args.batch_size
        )
        rows: dict[str, Any] = {}
        skipped: list[dict[str, Any]] = []
        blocks: list[np.ndarray] = []
        order: list[str] = []
        for index, wildtype in enumerate(referent.wildtypes):
            texts = scorer.render(wildtype.sequences())
            lengths = scorer.token_lengths(texts)
            if max(lengths) > scorer.context:
                skipped.append(
                    {
                        "wildtype": wildtype.name,
                        "max_tokens": int(max(lengths)),
                        "context": scorer.context,
                        "reason": "the rendered variant exceeds this arm's context",
                    }
                )
                continue
            scores = scorer.log_likelihood(texts)
            rows[wildtype.name] = {
                "kind": wildtype.kind,
                "unit": wildtype.unit,
                "n_variants": len(wildtype.mutants),
                "max_tokens": int(max(lengths)),
                "spearman": D.spearman(scores, wildtype.phenotype),
            }
            blocks.append(scores)
            order.append(wildtype.name)
            if index % 25 == 0:
                print(
                    f"[score] {name} {index + 1}/{len(referent.wildtypes)} "
                    f"{wildtype.name} {rows[wildtype.name]['spearman']}",
                    flush=True,
                )
        if not blocks:
            raise RuntimeError(f"{name}: no wild type was scored")
        np.savez_compressed(
            args.out / f"model_{name}.npz",
            scores=np.concatenate(blocks),
            offsets=np.cumsum([0] + [block.size for block in blocks]),
            wildtypes=np.array(order),
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "stage": "score",
            "arm": name,
            "created_utc": _timestamp(),
            "cohort_sha256": digest,
            "identification": dict(D.ARM_IDENTIFICATION[name]),
            "settings": {
                "checkpoint": str(PANEL[name].path),
                "input_format": PANEL[name].input_format,
                "context": scorer.context,
                "device": args.device,
                "dtype": args.dtype,
                "batch_size": args.batch_size,
                "score": "summed log-likelihood of the rendered variant",
            },
            "wildtypes": rows,
            "skipped": skipped,
        }
        # Written last: the driver treats a JSON in the output directory as
        # completion, so the array must already be on disk when it appears.
        write_json(args.out / f"model_{name}.json", payload)
        results[name] = payload
        scorer.release()
    return results


# ----------------------------------------------------------------- stage: analyse


def _side_report(
    model: dict[str, float],
    baselines: dict[str, dict[str, float]],
    units: dict[str, str],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    names = sorted(model)
    report: dict[str, Any] = {
        "n_wildtypes": len(names),
        "n_units": len({units[name] for name in names}),
        "model": D.unit_bootstrap(
            [model[name] for name in names],
            [units[name] for name in names],
            resamples=resamples,
            seed=seed,
        ),
        "baselines": {},
        "contrasts": {},
    }
    gates: dict[str, bool] = {}
    for name in D.BASELINES:
        values = baselines[name]
        shared = sorted(set(values) & set(model))
        report["baselines"][name] = D.unit_bootstrap(
            [values[key] for key in shared],
            [units[key] for key in shared],
            resamples=resamples,
            seed=seed,
        )
        contrast = D.channel_comparison(
            model, values, units, resamples=resamples, seed=seed
        )
        report["contrasts"][name] = contrast
        gates[name] = bool(contrast["beats_baseline"])
    report["gates"] = gates
    return report


def stage_analyse(args: argparse.Namespace) -> dict[str, Any]:
    cohort_path = args.cohort or args.out / "cohort.json"
    referent = D.load_referent(cohort_path)
    digest = sha256_file(cohort_path)
    baseline_payload = _read(args.out / "baselines.json")
    if baseline_payload["cohort_sha256"] != digest:
        raise RuntimeError("the baselines were computed on a different cohort")

    by_name = {wt.name: wt for wt in referent.wildtypes}
    zero_hit = {wt.name for wt in referent.side("design")}
    clean_series = {wt.name for wt in referent.side("design") if wt.series_zero_hit}
    naturals = {wt.name for wt in referent.side("natural")}

    def baseline_channel(names: set[str], key: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for name in names:
            entry = baseline_payload["wildtypes"].get(name)
            if entry is None:
                continue
            value = entry["spearman"][key]
            if value is not None:
                out[name] = float(value)
        return out

    arms: dict[str, Any] = {}
    for arm in args.arms:
        path = args.out / f"model_{arm}.json"
        if not path.is_file():
            print(f"[analyse] no model scores for {arm}; skipping")
            continue
        model_payload = _read(path)
        if model_payload["cohort_sha256"] != digest:
            raise RuntimeError(f"{arm} was scored on a different cohort")
        model_all = {
            name: float(entry["spearman"])
            for name, entry in model_payload["wildtypes"].items()
            if entry["spearman"] is not None
        }

        def side(names: set[str], units: dict[str, str], label: str) -> dict[str, Any]:
            model = {name: value for name, value in model_all.items() if name in names}
            baselines = {key: baseline_channel(names, key) for key in D.BASELINES}
            report = _side_report(
                model, baselines, units, resamples=args.bootstrap, seed=args.seed
            )
            report["label"] = label
            return report

        series_units = {name: by_name[name].unit for name in by_name}
        topology_units = {
            name: (
                f"topology:{by_name[name].cluster}"
                if by_name[name].kind == "design"
                else by_name[name].unit
            )
            for name in by_name
        }
        designs = side(zero_hit, series_units, "a: certified zero-hit designs, series units")
        control = side(naturals, series_units, "b: natural-domain control, WT_cluster units")
        arms[arm] = {
            "identification": model_payload["identification"],
            "designs": designs,
            "control": control,
            "verdict": D.arm_verdict(designs["gates"], control["gates"]),
            "preregistered_companions": {
                "designs_in_entirely_zero_hit_series": side(
                    clean_series, series_units, "a-strict: entirely zero-hit series only"
                ),
                "designs_at_topology_units": side(
                    zero_hit, topology_units, "a-topology: WT_cluster topology units"
                ),
            },
        }
        verdict = arms[arm]["verdict"]
        print(
            f"[analyse] {arm}: {verdict['verdict']} "
            f"(designs {designs['model']['point']:+.4f}, "
            f"control {control['model']['point']:+.4f})"
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": "analyse",
        "created_utc": _timestamp(),
        "cohort_sha256": digest,
        "cohort": referent.provenance,
        "kmer_background": baseline_payload["kmer_background"],
        "settings": {
            "bootstrap": args.bootstrap,
            "seed": args.seed,
            "baselines": list(D.BASELINES),
        },
        "excluded_arms": dict(D.EXCLUDED_ARMS),
        "arms": arms,
    }
    write_json(args.out / "designed_referent.json", payload)
    return payload


# ------------------------------------------------------------------------ main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--cohort",
        type=Path,
        default=None,
        help="the frozen cohort artefact; defaults to cohort.json under --out",
    )
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    parser.add_argument("--megascale-dir", type=Path, default=None)
    parser.add_argument("--certificate-dir", type=Path, default=None)
    parser.add_argument(
        "--kmer-background-dir", type=Path, default=D.KMER_BACKGROUND_DIR
    )
    parser.add_argument("--retrieval-bound-dir", type=Path, default=None)
    parser.add_argument("--min-variants", type=int, default=D.MIN_VARIANTS)
    parser.add_argument("--bootstrap", type=int, default=D.BOOTSTRAP_RESAMPLES)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[paths] out={args.out}")
    print(f"[paths] megascale={args.megascale_dir or D.MEGASCALE_DIR}")
    print(f"[paths] certificate={args.certificate_dir or D.CERTIFICATE_DIR}")
    print(f"[paths] kmer_background={args.kmer_background_dir}")
    print(f"[paths] cohort={args.cohort or args.out / 'cohort.json'}")
    print(f"[paths] arms={args.arms}")

    runners = {
        "cohort": stage_cohort,
        "baseline": stage_baseline,
        "score": stage_score,
        "analyse": stage_analyse,
    }
    for stage in args.stages:
        print(f"=== {stage}")
        runners[stage](args)


if __name__ == "__main__":
    main()
