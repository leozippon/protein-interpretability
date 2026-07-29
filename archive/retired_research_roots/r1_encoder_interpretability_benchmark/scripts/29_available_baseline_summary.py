#!/usr/bin/env python
"""Build the R1 available-baseline summary table and residual diagnostics."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
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


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - float(np.mean(x))) / (float(np.std(x)) + 1e-8)


def fmt(x: float) -> str:
    return "nan" if x is None or not math.isfinite(float(x)) else f"{float(x):.4f}"


def recompute_clinvar_scores() -> dict[tuple[str, str], dict[str, float]]:
    os.chdir(REPO)
    ens = load_script(ROOT / "scripts" / "14_ensemble_annotated_llr.py", "ensemble_annotated_llr_for_summary")
    meta_by_layer = ens.load_annotation_metadata()
    X, y, meta = ens.build_feature_vectors(meta_by_layer)
    good = X.std(0) > 1e-8
    X = X[:, good]
    X_scaled = StandardScaler().fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sae_scores = cross_val_predict(
        LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs"),
        X_scaled,
        y,
        cv=cv,
        method="predict_proba",
    )[:, 1]
    with open(OUT_DIR / "esm2_per_variant_llr.json") as f:
        llr_rows = json.load(f)
    llr_lookup = {(r["gene"], r["variant"]): -float(r["llr"]) for r in llr_rows}
    llr = np.zeros(len(meta), dtype=np.float32)
    for i, m in enumerate(meta):
        llr[i] = llr_lookup.get((m["gene"], m["variant"]), 0.0)
    ensemble = zscore(sae_scores) + zscore(llr)
    out = {}
    for i, m in enumerate(meta):
        out[(m["gene"], m["variant"])] = {
            "SAE-LR": float(sae_scores[i]),
            "ESM-2 LLR": float(llr[i]),
            "SAE+LLR": float(ensemble[i]),
        }
    return out


def load_cancer_scores() -> dict[tuple[str, str], dict[str, float]]:
    with open(OUT_DIR / "cancer_holdout.json") as f:
        data = json.load(f)
    out = {}
    for r in data["predictions"]:
        out[(r["gene"], r["variant"])] = {
            "SAE-LR": float(r["score_sae"]),
            "ESM-2 LLR": float(r["score_llr"]),
            "SAE+LLR": float(r["score_sae_plus_llr"]),
        }
    return out


def load_external_scores(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            row = dict(row)
            row["label"] = int(row["label"])
            row["gene"] = row["gene"].upper()
            for key in [
                "am_pathogenicity",
                "gmvp_rankscore",
                "esm1v_ensemble_pathogenicity",
                "esm1v_model1_pathogenicity",
                "esm1v_model2_pathogenicity",
                "esm1v_model3_pathogenicity",
                "esm1v_model4_pathogenicity",
                "esm1v_model5_pathogenicity",
            ]:
                if row.get(key) not in {None, "", "."}:
                    row[key] = float(row[key])
            rows.append(row)
    return rows


def merge_local_scores(rows: list[dict]) -> None:
    clinvar = recompute_clinvar_scores()
    cancer = load_cancer_scores()
    for row in rows:
        key = (row["gene"], row["variant"])
        local = clinvar.get(key) if row["cohort"] == "ClinVar2000" else cancer.get(key)
        if local:
            row.update(local)


def load_mechanism_labels() -> dict[tuple[str, str], str]:
    labels = {}
    with (OUT_DIR / "variant_mechanisms.tsv").open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            labels[(row["gene"].upper(), row["variant"])] = row["mechanism"]
    return labels


def scalar_mechanism_auc(rows: list[dict], score_key: str) -> dict:
    labels = load_mechanism_labels()
    mech_rows = []
    for row in rows:
        if row["cohort"] != "ClinVar2000" or row.get(score_key) in {None, "", "."}:
            continue
        mech = labels.get((row["gene"], row["variant"]), "UNLABELED")
        if mech != "UNLABELED":
            mech_rows.append((mech, float(row[score_key])))
    classes = sorted({m for m, _ in mech_rows})
    out = {}
    for cls in classes:
        y = np.array([1 if m == cls else 0 for m, _ in mech_rows], dtype=np.int32)
        s = np.array([v for _, v in mech_rows], dtype=np.float64)
        out[cls] = float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else math.nan
    out["macro_auc"] = float(np.nanmean([out[c] for c in classes])) if classes else math.nan
    out["n"] = len(mech_rows)
    return out


def pairwise_spearman(rows: list[dict], cohort: str, methods: dict[str, str]) -> list[dict]:
    sub = [r for r in rows if r["cohort"] == cohort]
    out = []
    for a, b in combinations(methods, 2):
        ka, kb = methods[a], methods[b]
        pairs = [
            (float(r[ka]), float(r[kb]))
            for r in sub
            if r.get(ka) not in {None, "", "."} and r.get(kb) not in {None, "", "."}
        ]
        if len(pairs) < 10:
            continue
        x, y = np.array(pairs).T
        rho, p = spearmanr(x, y)
        out.append({"cohort": cohort, "method_a": a, "method_b": b, "n": len(pairs), "spearman_rho": float(rho), "p": float(p)})
    return out


def residual_cases(rows: list[dict], methods: dict[str, str], out_tsv: Path) -> list[dict]:
    sub = [r for r in rows if r["cohort"] == "ClinVar2000" and "SAE-LR" in r and "am_pathogenicity" in r and "esm1v_ensemble_pathogenicity" in r]
    X = np.array([[float(r["am_pathogenicity"]), float(r["esm1v_ensemble_pathogenicity"])] for r in sub], dtype=np.float64)
    y = np.array([float(r["SAE-LR"]) for r in sub], dtype=np.float64)
    pred = LinearRegression().fit(X, y).predict(X)
    resid = y - pred
    for r, pred_i, resid_i in zip(sub, pred, resid):
        r["sae_lr_pred_from_external"] = float(pred_i)
        r["sae_lr_residual_vs_external"] = float(resid_i)
        r["abs_sae_lr_residual_vs_external"] = abs(float(resid_i))
    top = sorted(sub, key=lambda r: r["abs_sae_lr_residual_vs_external"], reverse=True)[:50]
    labels = load_mechanism_labels()
    fields = [
        "cohort", "gene", "variant", "label", "mechanism", "SAE-LR",
        "sae_lr_pred_from_external", "sae_lr_residual_vs_external",
        "am_pathogenicity", "gmvp_rankscore", "esm1v_ensemble_pathogenicity",
    ]
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for r in top:
            rr = dict(r)
            rr["mechanism"] = labels.get((r["gene"], r["variant"]), "UNLABELED")
            writer.writerow(rr)
    return [
        {
            "gene": r["gene"],
            "variant": r["variant"],
            "label": r["label"],
            "mechanism": labels.get((r["gene"], r["variant"]), "UNLABELED"),
            "sae_lr": float(r["SAE-LR"]),
            "sae_lr_residual_vs_external": float(r["sae_lr_residual_vs_external"]),
            "alphamissense": float(r["am_pathogenicity"]),
            "esm1v": float(r["esm1v_ensemble_pathogenicity"]),
        }
        for r in top[:20]
    ]


def markdown(summary: dict) -> str:
    lines = ["# R1 Available Baseline Summary (2026-05-07)\n"]
    lines.append("PrimateAI-3D is still excluded because access is pending.\n")
    lines.append("## Pathogenicity AUC\n")
    lines.append("| Cohort | Method | Matched n | AUC | 95% CI |")
    lines.append("|---|---|---:|---:|---|")
    for cohort in ["ClinVar2000", "CancerHoldout101"]:
        for row in summary["pathogenicity_table"]:
            if row["cohort"] != cohort:
                continue
            ci = row.get("ci95", [math.nan, math.nan])
            lines.append(f"| {cohort} | {row['method']} | {row['n']} | {fmt(row['auc'])} | [{fmt(ci[0])}, {fmt(ci[1])}] |")
    lines.append("\n## Mechanism One-Vs-Rest AUC From Scalar External Baselines\n")
    lines.append("| Method | n labeled variants | DN | GOF | LOF | Macro |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for method, vals in summary["mechanism_scalar_auc"].items():
        lines.append(f"| {method} | {vals['n']} | {fmt(vals.get('DN', math.nan))} | {fmt(vals.get('GOF', math.nan))} | {fmt(vals.get('LOF', math.nan))} | {fmt(vals.get('macro_auc', math.nan))} |")
    lines.append("\n## External Correlations\n")
    lines.append("| Cohort | Method A | Method B | n | Spearman rho |")
    lines.append("|---|---|---|---:|---:|")
    for row in summary["spearman_correlations"]:
        lines.append(f"| {row['cohort']} | {row['method_a']} | {row['method_b']} | {row['n']} | {row['spearman_rho']:.3f} |")
    lines.append("\n## Interpretation\n")
    lines.append(summary["interpretation"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readiness", type=Path, default=OUT_DIR / "baseline_headtohead_readiness_20260504.json")
    ap.add_argument("--external", type=Path, default=OUT_DIR / "external_baselines_available_20260507.json")
    ap.add_argument("--scores", type=Path, default=OUT_DIR / "external_baselines_available_scores_20260507.tsv")
    ap.add_argument("--out-json", type=Path, default=OUT_DIR / "available_baseline_summary_20260507.json")
    ap.add_argument("--out-md", type=Path, default=OUT_DIR / "available_baseline_summary_20260507.md")
    ap.add_argument("--residual-tsv", type=Path, default=OUT_DIR / "available_baseline_sae_residual_cases_20260507.tsv")
    args = ap.parse_args()

    readiness = json.loads(args.readiness.read_text())
    external = json.loads(args.external.read_text())
    rows = load_external_scores(args.scores)
    merge_local_scores(rows)

    method_map = {
        "SAE-LR": "SAE-LR",
        "ESM-2 LLR": "ESM-2 LLR",
        "SAE+LLR": "SAE+LLR",
        "AlphaMissense": "am_pathogenicity",
        "gMVP": "gmvp_rankscore",
        "ESM-1v": "esm1v_ensemble_pathogenicity",
    }
    path_table = []
    for cohort, methods in readiness["pathogenicity"].items():
        for method in ["SAE-LR", "ESM-2 LLR", "SAE+LLR"]:
            vals = methods[method]
            path_table.append({"cohort": cohort, "method": method, **vals})
    for method, by_cohort in external["metrics"].items():
        name = {"gMVP_rankscore": "gMVP", "ESM-1v_ensemble": "ESM-1v"}.get(method, method)
        for cohort, vals in by_cohort.items():
            path_table.append({"cohort": cohort, "method": name, **vals})

    mechanism = {
        "AlphaMissense": scalar_mechanism_auc(rows, "am_pathogenicity"),
        "gMVP": scalar_mechanism_auc(rows, "gmvp_rankscore"),
        "ESM-1v": scalar_mechanism_auc(rows, "esm1v_ensemble_pathogenicity"),
    }
    for method, vals in readiness["mechanism"].items():
        mechanism[method] = {"n": 497, "macro_auc": vals["protein_macro_auc"], **vals["protein_level"]}

    spearman_rows = []
    for cohort in ["ClinVar2000", "CancerHoldout101"]:
        spearman_rows.extend(pairwise_spearman(rows, cohort, method_map))
    residual_top = residual_cases(rows, method_map, args.residual_tsv)

    summary = {
        "task": "R1 T1-A available baseline summary",
        "status": "completed_available_methods_primateai_pending",
        "pathogenicity_table": path_table,
        "mechanism_scalar_auc": mechanism,
        "spearman_correlations": spearman_rows,
        "top_sae_residual_cases": residual_top,
        "residual_tsv": str(args.residual_tsv),
        "excluded": {"PrimateAI-3D": "gated dataset pending"},
        "interpretation": (
            "AlphaMissense and gMVP are stronger than the SAE/LLR ensemble on scalar pathogenicity. "
            "The useful R1 claim should therefore be framed around complementary, interpretable SAE residual signal and mechanism diagnostics, not beating every scalar predictor."
        ),
    }
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n")
    args.out_md.write_text(markdown(summary))
    print(f"Saved {args.out_json}")
    print(f"Saved {args.out_md}")
    print(f"Saved {args.residual_tsv}")


if __name__ == "__main__":
    main()
