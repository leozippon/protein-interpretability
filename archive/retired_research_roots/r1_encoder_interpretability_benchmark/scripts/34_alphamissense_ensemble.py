#!/usr/bin/env python
"""F-T1-3: SAE x AlphaMissense ensemble under protein/gene-level CV.

This is a no-wet-lab final-plan gate. It tests whether annotation-selected
SAE features add pathogenicity signal on top of AlphaMissense without using a
variant-level split that leaks variants from the same gene across train/test.

The grouping unit is the gene symbol because the staged ClinVar/SAE metadata is
gene-keyed. This is a conservative approximation of protein-level holdout for
the current 2,000-variant ClinVar cohort.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
OUT_DIR = REPO / "r1_encoder_interpretability_benchmark" / "results" / "variant_effect"


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_alphamissense(path: Path) -> dict[tuple[str, str], float]:
    scores = {}
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("cohort") != "ClinVar2000":
                continue
            val = row.get("am_pathogenicity", "")
            if val in {"", ".", "nan", None}:
                continue
            scores[(row["gene"].upper(), row["variant"])] = float(val)
    return scores


def z_with_train(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = float(np.mean(x_train))
    sd = float(np.std(x_train)) + 1e-8
    return (x_train - mu) / sd, (x_test - mu) / sd


def auc_ci(y: np.ndarray, score: np.ndarray, n_boot: int, seed: int) -> tuple[float, list[float]]:
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


def group_bootstrap_delta(
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    groups: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[float, list[float]]:
    delta = float(roc_auc_score(y, a) - roc_auc_score(y, b))
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(set(groups.tolist())))
    group_to_idx = {g: np.where(groups == g)[0] for g in unique}
    vals = []
    for _ in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([group_to_idx[g] for g in sampled])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(float(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx])))
    ci = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [math.nan, math.nan]
    return delta, ci


def build_data(am_scores: dict[tuple[str, str], float]):
    os.chdir(REPO)
    ens = load_script(ROOT / "scripts" / "14_ensemble_annotated_llr.py", "ensemble_annotated_llr_for_am")
    meta_by_layer = ens.load_annotation_metadata()
    X, y, meta = ens.build_feature_vectors(meta_by_layer)
    good = X.std(0) > 1e-8
    X = X[:, good]

    rows = []
    keep = []
    for i, m in enumerate(meta):
        key = (m["gene"].upper(), m["variant"])
        if key not in am_scores:
            continue
        keep.append(i)
        rows.append({
            "idx": len(rows),
            "gene": key[0],
            "variant": key[1],
            "label": int(y[i]),
            "am_pathogenicity": float(am_scores[key]),
            "clinical_significance": m.get("clinical_significance", ""),
        })
    keep_arr = np.array(keep, dtype=np.int64)
    return X[keep_arr].astype(np.float32), y[keep_arr].astype(np.int32), rows


def grouped_predictions(X: np.ndarray, y: np.ndarray, rows: list[dict], n_splits: int, seed: int) -> dict:
    groups = np.array([r["gene"] for r in rows])
    am = np.array([r["am_pathogenicity"] for r in rows], dtype=np.float64)
    sae = np.full(len(y), np.nan, dtype=np.float64)
    stack = np.full(len(y), np.nan, dtype=np.float64)
    zsum = np.full(len(y), np.nan, dtype=np.float64)
    fold_ids = np.full(len(y), -1, dtype=np.int32)

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        sae_clf = LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs")
        sae_clf.fit(Xtr, y[tr])
        sae_tr = sae_clf.predict_proba(Xtr)[:, 1]
        sae_te = sae_clf.predict_proba(Xte)[:, 1]
        sae[te] = sae_te

        ens = LogisticRegression(max_iter=2000, solver="lbfgs")
        ens.fit(np.column_stack([sae_tr, am[tr]]), y[tr])
        stack[te] = ens.predict_proba(np.column_stack([sae_te, am[te]]))[:, 1]

        sae_tr_z, sae_te_z = z_with_train(sae_tr, sae_te)
        am_tr_z, am_te_z = z_with_train(am[tr], am[te])
        _ = am_tr_z
        zsum[te] = sae_te_z + am_te_z
        fold_ids[te] = fold

    if not np.isfinite(sae).all() or not np.isfinite(stack).all() or not np.isfinite(zsum).all():
        raise RuntimeError("some rows did not receive grouped-CV predictions")
    return {"groups": groups, "am": am, "sae": sae, "stack": stack, "zsum": zsum, "fold_ids": fold_ids}


def write_predictions(path: Path, rows: list[dict], pred: dict) -> None:
    fields = [
        "idx", "fold", "gene", "variant", "label", "clinical_significance",
        "am_pathogenicity", "sae_lr_groupcv", "am_sae_stack_groupcv", "am_sae_zsum_groupcv",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for i, row in enumerate(rows):
            out = dict(row)
            out["fold"] = int(pred["fold_ids"][i])
            out["sae_lr_groupcv"] = float(pred["sae"][i])
            out["am_sae_stack_groupcv"] = float(pred["stack"][i])
            out["am_sae_zsum_groupcv"] = float(pred["zsum"][i])
            writer.writerow(out)


def make_summary(y: np.ndarray, rows: list[dict], pred: dict, args) -> dict:
    groups = pred["groups"]
    methods = {
        "AlphaMissense": pred["am"],
        "SAE-LR_groupCV": pred["sae"],
        "AM_plus_SAE_stack": pred["stack"],
        "AM_plus_SAE_zsum": pred["zsum"],
    }
    metrics = {}
    for name, score in methods.items():
        auc, ci = auc_ci(y, score, args.n_bootstrap, args.seed + len(metrics))
        metrics[name] = {"auc": auc, "ci95_variant_bootstrap": ci, "n": int(len(y))}

    deltas = {}
    for name in ["AM_plus_SAE_stack", "AM_plus_SAE_zsum", "SAE-LR_groupCV"]:
        delta, ci = group_bootstrap_delta(y, methods[name], methods["AlphaMissense"], groups, args.n_bootstrap, args.seed + 100 + len(deltas))
        deltas[f"{name}_minus_AlphaMissense"] = {"delta_auc": delta, "ci95_group_bootstrap": ci}

    gate = deltas["AM_plus_SAE_stack_minus_AlphaMissense"]
    gate_pass = bool(gate["ci95_group_bootstrap"][0] > 0.005)
    return {
        "task": "F-T1-3 SAE x AlphaMissense ensemble under gene-level CV",
        "status": "completed",
        "n_variants": int(len(y)),
        "n_genes": int(len(set(groups.tolist()))),
        "label_counts": dict(Counter(map(int, y))),
        "group_unit": "gene_symbol",
        "cv": {"type": "StratifiedGroupKFold", "n_splits": args.n_splits, "seed": args.seed},
        "metrics": metrics,
        "deltas_vs_alphamissense": deltas,
        "acceptance": {
            "criterion": "Keep complementarity claim only if stacked ensemble minus AlphaMissense has group-bootstrap CI lower bound > 0.005 AUC.",
            "pass": gate_pass,
            "decision": (
                "keep SAE+AlphaMissense ensemble as a major complementarity claim"
                if gate_pass
                else "do not claim ensemble improvement over AlphaMissense; use SAE as interpretation/residual diagnostic only"
            ),
        },
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }


def markdown(summary: dict) -> str:
    lines = ["# F-T1-3 SAE x AlphaMissense Ensemble\n"]
    lines.append(f"- Variants: {summary['n_variants']}")
    lines.append(f"- Genes/groups: {summary['n_genes']}")
    lines.append(f"- Group unit: {summary['group_unit']}\n")
    lines.append("## AUC\n")
    lines.append("| Method | AUC | Variant bootstrap 95% CI |")
    lines.append("|---|---:|---|")
    for name, vals in summary["metrics"].items():
        ci = vals["ci95_variant_bootstrap"]
        lines.append(f"| {name} | {vals['auc']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |")
    lines.append("\n## Delta vs AlphaMissense\n")
    lines.append("| Method | Delta AUC | Group bootstrap 95% CI |")
    lines.append("|---|---:|---|")
    for name, vals in summary["deltas_vs_alphamissense"].items():
        ci = vals["ci95_group_bootstrap"]
        lines.append(f"| {name} | {vals['delta_auc']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] |")
    lines.append("\n## Gate Decision\n")
    lines.append(summary["acceptance"]["decision"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--external-scores", type=Path, default=OUT_DIR / "external_baselines_available_scores_20260507.tsv")
    ap.add_argument("--out-json", type=Path, default=OUT_DIR / "alphamissense_sae_ensemble_20260511.json")
    ap.add_argument("--out-md", type=Path, default=OUT_DIR / "alphamissense_sae_ensemble_20260511.md")
    ap.add_argument("--out-tsv", type=Path, default=OUT_DIR / "alphamissense_sae_ensemble_predictions_20260511.tsv")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    am = load_alphamissense(args.external_scores)
    X, y, rows = build_data(am)
    pred = grouped_predictions(X, y, rows, args.n_splits, args.seed)
    summary = make_summary(y, rows, pred, args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as f:
        json.dump(summary, f, indent=2)
    with args.out_md.open("w") as f:
        f.write(markdown(summary))
    write_predictions(args.out_tsv, rows, pred)
    print(json.dumps({
        "json": str(args.out_json),
        "md": str(args.out_md),
        "tsv": str(args.out_tsv),
        "decision": summary["acceptance"]["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
