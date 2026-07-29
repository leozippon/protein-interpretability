#!/usr/bin/env python3
"""Protein-sequence baselines for IndelMissense v1.

This is R1-Add-1 from OPUS_NEXT_20260516.md. It compares the current SAE
damage score with baselines that do not require genomic coordinates:

- length / position / variant-class heuristics,
- gnomAD gene constraint proxies,
- a grouped-CV logistic model over those cheap features,
- optionally, ESM-2 masked-region pseudo-likelihood scores.

The ESM-2 mode is resume-friendly and intended for the H200 pod.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


REPO = Path(__file__).resolve().parents[2]
R1_OUT = REPO / "r1_encoder_interpretability_benchmark/results/variant_effect"
DEFAULT_RECORDS = REPO / "data/indelmissense/v1/records.jsonl"
DEFAULT_DAMAGE = REPO / "data/indelmissense/v1/baseline_scores.tsv"
DEFAULT_CONSTRAINT = REPO / "data/gnomad/gnomad.v4.1.constraint_metrics.tsv"
DEFAULT_ESM = R1_OUT / "indel_esm_region_scores_20260516.tsv"
DEFAULT_OUT = R1_OUT / "indel_protein_baselines_20260516"
AA = set("ACDEFGHIKLMNPQRSTVWY")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fnum(x: Any, default: float = math.nan) -> float:
    try:
        if x in {"", "NA", ".", None}:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def load_damage(path: Path) -> dict[str, float]:
    return {r["indel_id"]: fnum(r.get("damage_score")) for r in read_tsv(path)}


def load_constraint(path: Path) -> dict[str, dict[str, float]]:
    by_gene: dict[str, dict[str, float]] = {}
    if not path.exists():
        return by_gene
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gene = (row.get("gene") or "").upper()
            if not gene:
                continue
            canonical = row.get("canonical") == "true"
            mane = row.get("mane_select") == "true"
            level = fnum(row.get("level"), 0.0)
            rec = {
                "lof_pLI": fnum(row.get("lof.pLI")),
                "lof_oe": fnum(row.get("lof.oe")),
                "lof_oe_ci_upper": fnum(row.get("lof.oe_ci.upper")),
                "lof_z_score": fnum(row.get("lof.z_score")),
                "mis_z_score": fnum(row.get("mis.z_score")),
                "_priority": (2 if mane else 0) + (1 if canonical else 0) + level / 100.0,
            }
            prev = by_gene.get(gene)
            if prev is None or rec["_priority"] > prev.get("_priority", -1):
                by_gene[gene] = rec
    for rec in by_gene.values():
        rec.pop("_priority", None)
    return by_gene


def load_esm_scores(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    out = {}
    for row in read_tsv(path):
        if row.get("status") != "ok":
            continue
        out[row["indel_id"]] = {
            "esm_region_delta_mean_nll": fnum(row.get("delta_mean_nll")),
            "esm_region_delta_sum_nll": fnum(row.get("delta_sum_nll")),
            "esm_wt_mean_nll": fnum(row.get("wt_mean_nll")),
            "esm_mut_mean_nll": fnum(row.get("mut_mean_nll")),
        }
    return out


def auc_ci(y: np.ndarray, score: np.ndarray, n_boot: int, seed: int) -> tuple[float, list[float]]:
    mask = np.isfinite(score)
    y = y[mask]
    score = score[mask]
    auc = float(roc_auc_score(y, score))
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(float(roc_auc_score(y[idx], score[idx])))
    ci = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [math.nan, math.nan]
    return auc, ci


def grouped_lr_score(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    X = X.astype(np.float64)
    for j in range(X.shape[1]):
        col = X[:, j]
        med = np.nanmedian(col)
        col[~np.isfinite(col)] = med if np.isfinite(med) else 0.0
        X[:, j] = col
    pred = np.full(len(y), np.nan)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, y, groups):
        scaler = StandardScaler()
        xtr = scaler.fit_transform(X[tr])
        xte = scaler.transform(X[te])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(xtr, y[tr])
        pred[te] = clf.predict_proba(xte)[:, 1]
    return pred


def make_feature_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    records = read_jsonl(args.records)
    damage = load_damage(args.damage_scores)
    constraints = load_constraint(args.constraint)
    esm = load_esm_scores(args.esm_scores)
    rows = []
    for rec in records:
        indel_id = rec["indel_id"]
        gene = rec["gene"].upper()
        wt_len = len(rec.get("wt_seq") or "") or int(rec.get("wt_len") or 0)
        mut_len = len(rec.get("mut_seq") or "") or int(rec.get("mut_len") or 0)
        start = int(rec.get("start") or 1)
        end = int(rec.get("end") or start)
        length_delta = int(rec.get("length_delta", mut_len - wt_len))
        abs_delta = abs(length_delta)
        affected_span = max(1, end - start + 1)
        pos_norm = (start - 1) / max(wt_len - 1, 1)
        con = constraints.get(gene, {})
        row = {
            "indel_id": indel_id,
            "gene": gene,
            "uniprot_id": rec.get("uniprot_id", ""),
            "variant_class": rec.get("variant_class", ""),
            "label": rec.get("label", ""),
            "y": 1 if rec.get("label") == "pathogenic" else 0,
            "split": rec.get("split", ""),
            "wt_len": wt_len,
            "mut_len": mut_len,
            "start": start,
            "end": end,
            "position_norm": pos_norm,
            "length_delta": length_delta,
            "abs_length_delta": abs_delta,
            "rel_abs_length_delta": abs_delta / max(wt_len, 1),
            "affected_span_rel": affected_span / max(wt_len, 1),
            "truncating_score": 1.0 if rec.get("truncating") else 0.0,
            "early_position_score": 1.0 - pos_norm,
            "early_truncation_score": (1.0 - pos_norm) if rec.get("truncating") else 0.0,
            "is_deletion": 1.0 if rec.get("variant_class") == "deletion" else 0.0,
            "is_insertion": 1.0 if rec.get("variant_class") == "insertion" else 0.0,
            "is_delins": 1.0 if rec.get("variant_class") == "delins" else 0.0,
            "is_duplication": 1.0 if rec.get("variant_class") == "duplication" else 0.0,
            "sae_damage_score": damage.get(indel_id, math.nan),
            "gnomad_pLI": con.get("lof_pLI", math.nan),
            "gnomad_loeuf_score": -con.get("lof_oe_ci_upper", math.nan),
            "gnomad_lof_z": con.get("lof_z_score", math.nan),
            "gnomad_mis_z": con.get("mis_z_score", math.nan),
        }
        row.update(esm.get(indel_id, {}))
        rows.append(row)
    fields = list(rows[0].keys()) if rows else []
    return rows, fields


def summarize(args: argparse.Namespace) -> None:
    rows, fields = make_feature_rows(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.out_dir / "feature_table.tsv", rows, fields)

    y = np.array([r["y"] for r in rows], dtype=np.int32)
    groups = np.array([r["uniprot_id"] for r in rows])
    metrics: dict[str, dict[str, Any]] = {}
    score_cols = [
        "sae_damage_score",
        "truncating_score",
        "abs_length_delta",
        "rel_abs_length_delta",
        "early_position_score",
        "early_truncation_score",
        "gnomad_pLI",
        "gnomad_loeuf_score",
        "gnomad_lof_z",
        "gnomad_mis_z",
        "esm_region_delta_mean_nll",
        "esm_region_delta_sum_nll",
    ]
    for col in score_cols:
        if col not in fields:
            continue
        score = np.array([fnum(r.get(col)) for r in rows], dtype=np.float64)
        if np.isfinite(score).sum() < 50:
            continue
        auc, ci = auc_ci(y, score, args.n_bootstrap, args.seed + len(metrics))
        metrics[col] = {"auc": auc, "ci95": ci, "n": int(np.isfinite(score).sum())}

    cheap_cols = [
        "truncating_score",
        "abs_length_delta",
        "rel_abs_length_delta",
        "affected_span_rel",
        "early_position_score",
        "early_truncation_score",
        "is_deletion",
        "is_insertion",
        "is_delins",
        "is_duplication",
        "gnomad_pLI",
        "gnomad_loeuf_score",
        "gnomad_lof_z",
        "gnomad_mis_z",
    ]
    X = np.array([[fnum(r.get(c)) for c in cheap_cols] for r in rows], dtype=np.float64)
    pred = grouped_lr_score(X, y, groups, args.seed)
    auc, ci = auc_ci(y, pred, args.n_bootstrap, args.seed + 100)
    metrics["cheap_feature_grouped_lr"] = {"auc": auc, "ci95": ci, "n": int(np.isfinite(pred).sum()), "features": cheap_cols}

    if "esm_region_delta_mean_nll" in fields and np.isfinite([fnum(r.get("esm_region_delta_mean_nll")) for r in rows]).sum() >= 50:
        combo_cols = cheap_cols + ["esm_region_delta_mean_nll", "esm_region_delta_sum_nll", "sae_damage_score"]
        X2 = np.array([[fnum(r.get(c)) for c in combo_cols] for r in rows], dtype=np.float64)
        pred2 = grouped_lr_score(X2, y, groups, args.seed + 1)
        auc2, ci2 = auc_ci(y, pred2, args.n_bootstrap, args.seed + 101)
        metrics["sae_esm_cheap_grouped_lr"] = {"auc": auc2, "ci95": ci2, "n": int(np.isfinite(pred2).sum()), "features": combo_cols}

    sae_auc = metrics.get("sae_damage_score", {}).get("auc", math.nan)
    beatable = {
        k: v for k, v in metrics.items()
        if k not in {"sae_damage_score", "sae_esm_cheap_grouped_lr"} and math.isfinite(v.get("auc", math.nan))
    }
    best_baseline_name, best_baseline = max(beatable.items(), key=lambda kv: kv[1]["auc"], default=("", {"auc": math.nan}))
    beat_any_by_003 = any(sae_auc - v["auc"] >= 0.03 for v in beatable.values()) if math.isfinite(sae_auc) else False
    acceptance = bool(math.isfinite(sae_auc) and sae_auc >= 0.78 and beat_any_by_003)
    summary = {
        "task": "R1-Add-1 IndelMissense protein-sequence baselines",
        "status": "completed",
        "n_records": len(rows),
        "label_counts": dict(Counter(r["label"] for r in rows)),
        "metrics": metrics,
        "best_non_sae_baseline": {"name": best_baseline_name, **best_baseline},
        "acceptance_gate": "SAE damage AUC >= 0.78 and beats at least one non-SAE baseline by >=0.03 AUC.",
        "acceptance_pass": acceptance,
        "esm_scores_present": bool("esm_region_delta_mean_nll" in fields),
        "outputs": {
            "feature_table": str(args.out_dir / "feature_table.tsv"),
            "summary_md": str(args.out_dir / "summary.md"),
        },
    }
    with (args.out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    ranked = sorted(metrics.items(), key=lambda kv: kv[1]["auc"], reverse=True)
    md = [
        "# IndelMissense Protein-Sequence Baselines",
        "",
        "This analysis compares the current SAE damage score against baselines that do not require genomic coordinates.",
        "",
        f"- Records: {len(rows)}",
        f"- Label counts: {dict(Counter(r['label'] for r in rows))}",
        f"- ESM region scores present: {summary['esm_scores_present']}",
        f"- Acceptance gate: {'PASS' if acceptance else 'FAIL'}",
        "",
        "## AUC Table",
        "",
        "| Method | AUC | 95% CI | n |",
        "|---|---:|---:|---:|",
    ]
    for name, metric in ranked:
        ci = metric.get("ci95", [math.nan, math.nan])
        md.append(f"| {name} | {metric['auc']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {metric['n']} |")
    md += [
        "",
        "## Interpretation",
        "",
        "- If the gate passes, IndelMissense v1 can be framed as an interpretable protein-indel benchmark with a competitive SAE damage baseline.",
        "- If the gate fails, IndelMissense v1 remains useful as a bounded benchmark/resource, but not as evidence for a stronger indel scorer.",
        "- Genomic-coordinate baselines such as CADD/REVEL remain excluded because the packaged records do not preserve chrom/pos/ref/alt.",
        "",
    ]
    (args.out_dir / "summary.md").write_text("\n".join(md))
    print(f"Wrote {args.out_dir / 'feature_table.tsv'}")
    print(f"Wrote {args.out_dir / 'summary.md'}")
    print(f"Acceptance: {'PASS' if acceptance else 'FAIL'}")


def region_positions(length: int, start: int, end: int, inserted_len: int, window: int, kind: str) -> list[int]:
    if kind == "wt":
        a = max(1, start - window)
        b = min(length, end + window)
    else:
        affected = max(1, end - start + 1 + inserted_len)
        a = max(1, start - window)
        b = min(length, start + affected + window)
    if a > b:
        return [min(max(1, start), length)] if length else []
    return list(range(a, b + 1))


def esm_region(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, EsmForMaskedLM

    records = read_jsonl(args.records)
    if args.max_records:
        records = records[: args.max_records]
    done = set()
    if args.esm_scores.exists() and not args.force:
        for row in read_tsv(args.esm_scores):
            if row.get("status") == "ok":
                done.add(row["indel_id"])
    todo = [r for r in records if r["indel_id"] not in done]
    args.esm_scores.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.esm_scores.exists() or args.force
    mode = "w" if args.force else "a"
    fields = [
        "indel_id",
        "gene",
        "uniprot_id",
        "variant_class",
        "label",
        "wt_region_n",
        "mut_region_n",
        "wt_mean_nll",
        "mut_mean_nll",
        "delta_mean_nll",
        "wt_sum_nll",
        "mut_sum_nll",
        "delta_sum_nll",
        "status",
        "error",
    ]

    print(f"records={len(records)} done={len(done)} todo={len(todo)}")
    tokenizer = AutoTokenizer.from_pretrained(args.esm_model)
    model = EsmForMaskedLM.from_pretrained(args.esm_model, torch_dtype=torch.float16).to(args.device).eval()

    def score_jobs(jobs: list[dict[str, Any]]) -> list[dict[str, float]]:
        seqs = [" ".join(j["seq"]) for j in jobs]
        enc = tokenizer(seqs, padding=True, truncation=True, max_length=args.max_len + 2, return_tensors="pt").to(args.device)
        input_ids = enc["input_ids"].clone()
        for j, job in enumerate(jobs):
            for pos in job["positions"]:
                if 1 <= pos < input_ids.shape[1] - 1:
                    input_ids[j, pos] = tokenizer.mask_token_id
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(input_ids=input_ids, attention_mask=enc["attention_mask"])
        logp = F.log_softmax(out.logits.float(), dim=-1)
        scored = []
        for j, job in enumerate(jobs):
            vals = []
            for pos in job["positions"]:
                if not (1 <= pos <= len(job["seq"])):
                    continue
                aa = job["seq"][pos - 1]
                if aa not in AA:
                    continue
                tok = tokenizer.convert_tokens_to_ids(aa)
                vals.append(float(-logp[j, pos, tok].detach().cpu()))
            scored.append({"n": len(vals), "mean": float(np.mean(vals)) if vals else math.nan, "sum": float(np.sum(vals)) if vals else math.nan})
        return scored

    with args.esm_scores.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        if write_header:
            writer.writeheader()
        batch_out = []
        for i, rec in enumerate(todo, start=1):
            try:
                wt_seq = rec["wt_seq"][: args.max_len]
                mut_seq = rec["mut_seq"][: args.max_len]
                start = int(rec.get("start") or 1)
                end = int(rec.get("end") or start)
                inserted_len = len(rec.get("inserted_sequence") or "")
                jobs = [
                    {"seq": wt_seq, "positions": region_positions(len(wt_seq), start, end, inserted_len, args.local_window, "wt")},
                    {"seq": mut_seq, "positions": region_positions(len(mut_seq), start, end, inserted_len, args.local_window, "mut")},
                ]
                wt_s, mut_s = score_jobs(jobs)
                row = {
                    "indel_id": rec["indel_id"],
                    "gene": rec["gene"],
                    "uniprot_id": rec["uniprot_id"],
                    "variant_class": rec["variant_class"],
                    "label": rec["label"],
                    "wt_region_n": wt_s["n"],
                    "mut_region_n": mut_s["n"],
                    "wt_mean_nll": wt_s["mean"],
                    "mut_mean_nll": mut_s["mean"],
                    "delta_mean_nll": mut_s["mean"] - wt_s["mean"],
                    "wt_sum_nll": wt_s["sum"],
                    "mut_sum_nll": mut_s["sum"],
                    "delta_sum_nll": mut_s["sum"] - wt_s["sum"],
                    "status": "ok",
                    "error": "",
                }
            except Exception as exc:
                row = {
                    "indel_id": rec.get("indel_id", ""),
                    "gene": rec.get("gene", ""),
                    "uniprot_id": rec.get("uniprot_id", ""),
                    "variant_class": rec.get("variant_class", ""),
                    "label": rec.get("label", ""),
                    "status": "error",
                    "error": repr(exc),
                }
            batch_out.append(row)
            if len(batch_out) >= args.write_every:
                writer.writerows(batch_out)
                f.flush()
                os.fsync(f.fileno())
                print(f"wrote {i}/{len(todo)}", flush=True)
                batch_out = []
        if batch_out:
            writer.writerows(batch_out)
            f.flush()
    print(f"Saved {args.esm_scores}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["summarize", "esm-region"], default="summarize")
    ap.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    ap.add_argument("--damage-scores", type=Path, default=DEFAULT_DAMAGE)
    ap.add_argument("--constraint", type=Path, default=DEFAULT_CONSTRAINT)
    ap.add_argument("--esm-scores", type=Path, default=DEFAULT_ESM)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--esm-model", type=Path, default=Path("/oss-pvc/zhk_zip/models/esm2_t36_3B_UR50D"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-len", type=int, default=1022)
    ap.add_argument("--local-window", type=int, default=8)
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--write-every", type=int, default=25)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.mode == "esm-region":
        esm_region(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
