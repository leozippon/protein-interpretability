#!/usr/bin/env python
"""Score resource-ready external R1 baselines on the matched local cohorts.

This script intentionally runs only assets that are already staged:

* AlphaMissense scores from the previously matched TSV.
* gMVP from the standalone GRCh38 table, matched by gene and protein change.
* ESM-1v checkpoints, scored as masked-marginal LLR.

PrimateAI-3D remains excluded until the gated dataset is available.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import roc_auc_score


REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "r1_encoder_interpretability_benchmark" / "results" / "variant_effect"
DEFAULT_INPUT = OUT_DIR / "alphamissense_matched_scores_20260504.tsv"
DEFAULT_GMVP = Path(os.environ.get(
    "BIOCC_GMVP_HG38",
    "/oss-pvc/zhk_zip/biocc/external_resources/baselines/gmvp/gMVP.2021-02-28.csv.gz",
))
DEFAULT_ESM1V_DIR = Path(os.environ.get(
    "BIOCC_ESM1V_CHECKPOINT_DIR",
    "/oss-pvc/zhk_zip/biocc/external_resources/baselines/esm1v/checkpoints",
))
DEFAULT_SWISSPROT = Path(os.environ.get(
    "BIOCC_SWISSPROT_CACHE",
    "/gpfs/jiaotongdamoxing/zhk_zip/data/processed/swissprot_all_max1022.pkl",
))

AA_CHANGE_RE = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")


def parse_variant(v: str) -> tuple[str, int, str] | None:
    m = AA_CHANGE_RE.match(str(v).strip())
    if not m:
        return None
    return m.group(1), int(m.group(2)), m.group(3)


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            parsed = parse_variant(row.get("protein_variant") or row.get("variant") or "")
            if parsed is None:
                continue
            ref, pos, alt = parsed
            row = dict(row)
            row["row_id"] = str(i)
            row["gene"] = row["gene"].upper()
            row["label"] = int(row["label"])
            row["ref_aa"] = ref
            row["protein_position"] = pos
            row["alt_aa"] = alt
            for key in ["am_pathogenicity"]:
                if key in row and row[key] not in {"", "."}:
                    row[key] = float(row[key])
            rows.append(row)
    return rows


def auc_ci(y: Iterable[int], score: Iterable[float], n_boot: int, seed: int = 0) -> dict:
    y = np.asarray(list(y), dtype=np.int32)
    score = np.asarray(list(score), dtype=np.float64)
    mask = np.isfinite(score)
    y = y[mask]
    score = score[mask]
    out = {"n": int(len(y)), "n_pathogenic": int(y.sum()), "n_benign": int((1 - y).sum())}
    if len(y) == 0 or len(np.unique(y)) < 2:
        out.update({"auc": math.nan, "ci95": [math.nan, math.nan]})
        return out
    out["auc"] = float(roc_auc_score(y, score))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(float(roc_auc_score(y[idx], score[idx])))
    out["ci95"] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [math.nan, math.nan]
    return out


def cohort_metrics(rows: list[dict], score_key: str, n_boot: int) -> dict:
    out = {}
    for cohort in sorted({r["cohort"] for r in rows}):
        sub = [r for r in rows if r["cohort"] == cohort and score_key in r]
        out[cohort] = auc_ci([r["label"] for r in sub], [r[score_key] for r in sub], n_boot=n_boot)
        out[cohort]["n_total"] = sum(1 for r in rows if r["cohort"] == cohort)
        out[cohort]["match_rate"] = out[cohort]["n"] / max(out[cohort]["n_total"], 1)
    return out


def score_gmvp(rows: list[dict], gmvp_path: Path) -> dict:
    targets = {
        (r["gene"], str(r["protein_position"]), r["ref_aa"], r["alt_aa"])
        for r in rows
    }
    found: dict[tuple[str, str, str, str], tuple[float, float]] = {}
    t0 = time.time()
    with gzip.open(gmvp_path, "rt") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            key = (
                row["gene_symbol"].upper(),
                row["protein_position"],
                row["ref_aa"],
                row["alt_aa"],
            )
            if key not in targets:
                continue
            try:
                score = float(row["gMVP"])
                rank = float(row["gMVP_rankscore"])
            except ValueError:
                continue
            old = found.get(key)
            if old is None or rank > old[1]:
                found[key] = (score, rank)
            if len(found) == len(targets):
                break
    for r in rows:
        key = (r["gene"], str(r["protein_position"]), r["ref_aa"], r["alt_aa"])
        if key in found:
            r["gmvp_score"], r["gmvp_rankscore"] = found[key]
    return {"n_targets": len(targets), "n_found": len(found), "elapsed_s": round(time.time() - t0, 1)}


def load_sequence_map(cache_path: Path) -> dict[str, str]:
    sys.path.insert(0, str(REPO / "r1_encoder_interpretability_benchmark"))
    with cache_path.open("rb") as f:
        annotations = pickle.load(f)
    return {ann.accession: ann.sequence for ann in annotations}


def esm1v_checkpoints(checkpoint_dir: Path, limit: int | None) -> list[Path]:
    paths = sorted(checkpoint_dir.glob("esm1v_t33_650M_UR90S_*.pt"))
    return paths[:limit] if limit else paths


def score_esm1v(
    rows: list[dict],
    sequence_map: dict[str, str],
    checkpoint_dir: Path,
    checkpoint_limit: int | None,
    device: str,
    partial_tsv: Path,
) -> dict:
    import torch
    import torch.nn.functional as F
    import esm

    original_torch_load = torch.load

    def torch_load_compat(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_torch_load(*load_args, **load_kwargs)

    torch.load = torch_load_compat
    ckpts = esm1v_checkpoints(checkpoint_dir, checkpoint_limit)
    if not ckpts:
        return {"status": "missing_checkpoints", "checkpoint_dir": str(checkpoint_dir)}

    scorable = []
    for r in rows:
        seq = sequence_map.get(r["uniprot_id"])
        pos = int(r["protein_position"])
        if not seq or pos < 1 or pos > len(seq) or len(seq) > 1022:
            continue
        if seq[pos - 1] != r["ref_aa"]:
            continue
        scorable.append((r, seq))

    per_model_keys = []
    t0 = time.time()
    for mi, ckpt in enumerate(ckpts, start=1):
        model_t0 = time.time()
        print(f"[ESM-1v] loading {ckpt}", flush=True)
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(ckpt))
        model.eval()
        model = model.to(device)
        if device.startswith("cuda"):
            model = model.half()
        batch_converter = alphabet.get_batch_converter()
        mask_idx = alphabet.mask_idx
        key = f"esm1v_model{mi}_pathogenicity"
        per_model_keys.append(key)
        with torch.no_grad():
            for ri, (r, seq) in enumerate(scorable, start=1):
                _, _, tokens = batch_converter([(r["row_id"], seq)])
                tokens = tokens.to(device)
                pos = int(r["protein_position"])
                tokens[0, pos] = mask_idx
                with torch.amp.autocast("cuda", enabled=device.startswith("cuda"), dtype=torch.float16):
                    out = model(tokens)
                    logits = out["logits"][0, pos].float()
                log_probs = F.log_softmax(logits, dim=-1)
                ref_idx = alphabet.get_idx(r["ref_aa"])
                alt_idx = alphabet.get_idx(r["alt_aa"])
                llr = float(log_probs[alt_idx] - log_probs[ref_idx])
                r[key] = -llr
                if ri % 100 == 0:
                    print(f"[ESM-1v] model {mi}/{len(ckpts)} scored {ri}/{len(scorable)}", flush=True)
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        write_tsv(rows, partial_tsv)
        print(f"[ESM-1v] finished model {mi}/{len(ckpts)} in {time.time() - model_t0:.1f}s", flush=True)

    for r in rows:
        vals = [r[k] for k in per_model_keys if k in r and math.isfinite(float(r[k]))]
        if vals:
            r["esm1v_ensemble_pathogenicity"] = float(np.mean(vals))
    return {
        "status": "completed",
        "n_checkpoints": len(ckpts),
        "n_scorable": len(scorable),
        "elapsed_s": round(time.time() - t0, 1),
    }


def write_tsv(rows: list[dict], path: Path) -> None:
    base = [
        "row_id", "cohort", "gene", "variant", "uniprot_id", "protein_variant",
        "label", "am_pathogenicity", "gmvp_score", "gmvp_rankscore",
        "esm1v_ensemble_pathogenicity",
    ]
    model_keys = sorted({k for r in rows for k in r if k.startswith("esm1v_model")})
    fields = base + model_keys
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def markdown(summary: dict) -> str:
    lines = ["# Resource-Ready External Baselines (2026-05-07)\n"]
    lines.append("PrimateAI-3D is excluded because the gated dataset is still pending.\n")
    lines.append("| Method | Cohort | Matched | AUC | 95% CI |")
    lines.append("|---|---|---:|---:|---|")
    for method, by_cohort in summary["metrics"].items():
        for cohort, vals in by_cohort.items():
            ci = vals["ci95"]
            auc = vals["auc"]
            auc_s = "nan" if math.isnan(auc) else f"{auc:.4f}"
            ci_s = "[nan, nan]" if any(math.isnan(x) for x in ci) else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            lines.append(f"| {method} | {cohort} | {vals['n']}/{vals['n_total']} | {auc_s} | {ci_s} |")
    lines.append("\n## Run Status\n")
    for key, vals in summary["run_status"].items():
        lines.append(f"- {key}: `{vals}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-tsv", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--gmvp", type=Path, default=DEFAULT_GMVP)
    ap.add_argument("--esm1v-dir", type=Path, default=DEFAULT_ESM1V_DIR)
    ap.add_argument("--swissprot-cache", type=Path, default=DEFAULT_SWISSPROT)
    ap.add_argument("--skip-gmvp", action="store_true")
    ap.add_argument("--skip-esm1v", action="store_true")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--esm1v-checkpoint-limit", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--out-json", type=Path, default=OUT_DIR / "external_baselines_available_20260507.json")
    ap.add_argument("--out-md", type=Path, default=OUT_DIR / "external_baselines_available_20260507.md")
    ap.add_argument("--out-tsv", type=Path, default=OUT_DIR / "external_baselines_available_scores_20260507.tsv")
    args = ap.parse_args()

    rows = load_rows(args.input_tsv)
    if args.max_rows is not None:
        rows = rows[:args.max_rows]
    run_status = {"input_rows": len(rows)}
    if not args.skip_gmvp:
        run_status["gMVP"] = score_gmvp(rows, args.gmvp)
    if not args.skip_esm1v:
        sequence_map = load_sequence_map(args.swissprot_cache)
        run_status["ESM-1v"] = score_esm1v(
            rows,
            sequence_map,
            args.esm1v_dir,
            args.esm1v_checkpoint_limit,
            args.device,
            args.out_tsv,
        )

    metrics = {"AlphaMissense": cohort_metrics(rows, "am_pathogenicity", args.n_bootstrap)}
    if any("gmvp_rankscore" in r for r in rows):
        metrics["gMVP_rankscore"] = cohort_metrics(rows, "gmvp_rankscore", args.n_bootstrap)
    if any("esm1v_ensemble_pathogenicity" in r for r in rows):
        metrics["ESM-1v_ensemble"] = cohort_metrics(rows, "esm1v_ensemble_pathogenicity", args.n_bootstrap)

    summary = {
        "task": "R1 T1-A resource-ready external baselines",
        "status": "completed_available_resources",
        "input_tsv": str(args.input_tsv),
        "run_status": run_status,
        "metrics": metrics,
        "excluded": {"PrimateAI-3D": "gated dataset still pending"},
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    write_tsv(rows, args.out_tsv)
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n")
    args.out_md.write_text(markdown(summary))
    print(f"Saved {args.out_json}")
    print(f"Saved {args.out_md}")
    print(f"Saved {args.out_tsv}")


if __name__ == "__main__":
    main()
