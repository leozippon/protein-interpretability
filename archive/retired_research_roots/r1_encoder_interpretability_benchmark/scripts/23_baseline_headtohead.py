#!/usr/bin/env python
"""T1-A baseline head-to-head readiness and available-score table.

The TODO_NEXT T1-A acceptance criterion requires AlphaMissense, PrimateAI-3D,
gMVP, and ESM-1v assets. Those assets are not staged in the current repo. This
script therefore does two things without silent fallback:

1. Recomputes the available local R1 pathogenicity table for SAE-LR, ESM-2 LLR,
   and SAE+LLR, including bootstrap CIs.
2. Emits an explicit missing-asset table for the unstaged competitors, so T1-A
   is marked blocked rather than accidentally treated as complete.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np


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


def auc_ci(y: np.ndarray, score: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    score = np.asarray(score)
    mask = np.isfinite(score)
    y = y[mask]
    score = score[mask]
    auc = float(roc_auc_score(y, score)) if len(set(y.tolist())) == 2 else float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx].tolist())) < 2:
            continue
        vals.append(float(roc_auc_score(y[idx], score[idx])))
    ci = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [float("nan"), float("nan")]
    return {"auc": auc, "ci95": ci, "n": int(len(y))}


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - float(np.mean(x))) / (float(np.std(x)) + 1e-8)


def compute_clinvar_available(n_boot: int) -> dict:
    os.chdir(REPO)
    ens = load_script(ROOT / "scripts" / "14_ensemble_annotated_llr.py", "ensemble_annotated_llr_14")

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler

    meta_by_layer = ens.load_annotation_metadata()
    X, y, meta = ens.build_feature_vectors(meta_by_layer)
    good = X.std(0) > 1e-8
    X = X[:, good]
    X_scaled = StandardScaler().fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    sae_clf = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs")
    sae_scores = cross_val_predict(sae_clf, X_scaled, y, cv=cv, method="predict_proba")[:, 1]

    with open(OUT_DIR / "esm2_per_variant_llr.json") as f:
        pv = json.load(f)
    llr_lookup = {(r["gene"], r["variant"]): float(r["llr"]) for r in pv}
    llr = np.zeros(len(meta), dtype=np.float32)
    matched = 0
    for i, m in enumerate(meta):
        key = (m["gene"], m["variant"])
        if key in llr_lookup:
            llr[i] = -llr_lookup[key]
            matched += 1

    ensemble = zscore(sae_scores) + zscore(llr)
    return {
        "SAE-LR": auc_ci(y, sae_scores, n_boot=n_boot),
        "ESM-2 LLR": auc_ci(y, llr, n_boot=n_boot),
        "SAE+LLR": auc_ci(y, ensemble, n_boot=n_boot),
        "n_llr_matched": int(matched),
    }


def compute_cancer_available(n_boot: int) -> dict:
    with open(OUT_DIR / "cancer_holdout.json") as f:
        data = json.load(f)
    rows = data["predictions"]
    y = np.array([r["label"] for r in rows], dtype=np.int32)
    return {
        "SAE-LR": auc_ci(y, np.array([r["score_sae"] for r in rows]), n_boot=n_boot),
        "ESM-2 LLR": auc_ci(y, np.array([r["score_llr"] for r in rows]), n_boot=n_boot),
        "SAE+LLR": auc_ci(y, np.array([r["score_sae_plus_llr"] for r in rows]), n_boot=n_boot),
    }


def load_mechanism_available() -> dict:
    path = OUT_DIR / "mechanism_classifier_results_t0_protein_holdout_20260429.json"
    with open(path) as f:
        data = json.load(f)
    return {
        "SAE-LR": {
            "variant_level": data["variant_level_cv"]["sae_only"]["per_class_auc"],
            "protein_level": data["protein_level_cv"]["sae_only"]["per_class_auc"],
            "protein_macro_auc": data["protein_level_cv"]["sae_only"]["macro_auc"],
        },
        "ESM-2 LLR": {
            "variant_level": data["variant_level_cv"]["llr_only"]["per_class_auc"],
            "protein_level": data["protein_level_cv"]["llr_only"]["per_class_auc"],
            "protein_macro_auc": data["protein_level_cv"]["llr_only"]["macro_auc"],
        },
        "SAE+LLR": {
            "variant_level": data["variant_level_cv"]["sae_plus_llr"]["per_class_auc"],
            "protein_level": data["protein_level_cv"]["sae_plus_llr"]["per_class_auc"],
            "protein_macro_auc": data["protein_level_cv"]["sae_plus_llr"]["macro_auc"],
        },
    }


def find_assets(patterns: list[str]) -> list[str]:
    roots = [
        REPO / "data",
        REPO / "external_resources",
        REPO / "r1_encoder_interpretability_benchmark" / "results",
        REPO / "r1_encoder_interpretability_benchmark" / "models",
    ]
    found = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            found.extend(str(p.relative_to(REPO)) for p in root.rglob(pattern))
    return sorted(set(found))


def competitor_status() -> dict:
    spec = {
        "AlphaMissense": ["*AlphaMissense*", "*alphamissense*"],
        "PrimateAI-3D": ["*PrimateAI*", "*primateai*"],
        "gMVP": ["*gMVP*", "*gmvp*"],
        "ESM-1v": ["*ESM-1v*", "*esm1v*", "*esm-1v*"],
    }
    out = {}
    for name, patterns in spec.items():
        assets = find_assets(patterns)
        out[name] = {
            "status": "available_unscored" if assets else "missing_assets",
            "assets": assets,
        }
    return out


def markdown(result: dict) -> str:
    lines = ["# T1-A Baseline Head-to-Head Readiness\n"]
    lines.append("This table uses only staged local artifacts. Missing competitors are not imputed.\n")
    lines.append("## Pathogenicity AUC\n")
    lines.append("| Cohort | Method | AUC | 95% CI | n |")
    lines.append("|---|---|---:|---|---:|")
    for cohort in ["ClinVar2000", "CancerHoldout101"]:
        for method, vals in result["pathogenicity"][cohort].items():
            if not isinstance(vals, dict) or "auc" not in vals:
                continue
            ci = vals["ci95"]
            lines.append(f"| {cohort} | {method} | {vals['auc']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {vals['n']} |")
    lines.append("\n## Mechanism AUC\n")
    lines.append("| Method | Protein-Level Macro-AUC | Protein-Level Per-Class AUC |")
    lines.append("|---|---:|---|")
    for method, vals in result["mechanism"].items():
        per = ", ".join(f"{k}={v:.4f}" for k, v in vals["protein_level"].items())
        lines.append(f"| {method} | {vals['protein_macro_auc']:.4f} | {per} |")
    lines.append("\n## Competitor Asset Status\n")
    lines.append("| Competitor | Status | Assets |")
    lines.append("|---|---|---|")
    for name, vals in result["competitors"].items():
        assets = "<br>".join(vals["assets"]) if vals["assets"] else ""
        lines.append(f"| {name} | {vals['status']} | {assets} |")
    lines.append("\n## Decision\n")
    lines.append(result["decision"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default="r1_encoder_interpretability_benchmark/results/variant_effect/baseline_headtohead_readiness_20260504.json")
    ap.add_argument("--out-md", default="r1_encoder_interpretability_benchmark/results/variant_effect/baseline_headtohead_readiness_20260504.md")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    args = ap.parse_args()

    result = {
        "task": "T1-A baseline head-to-head",
        "status": "blocked_missing_competitor_assets",
        "pathogenicity": {
            "ClinVar2000": compute_clinvar_available(args.n_bootstrap),
            "CancerHoldout101": compute_cancer_available(args.n_bootstrap),
        },
        "mechanism": load_mechanism_available(),
        "competitors": competitor_status(),
    }
    missing = [k for k, v in result["competitors"].items() if v["status"] == "missing_assets"]
    result["acceptance_met"] = len(missing) == 0
    result["decision"] = (
        "T1-A is blocked: missing competitor assets for "
        + ", ".join(missing)
        + ". Per TODO_NEXT, these methods must be staged and calibrated before "
        "they are used for reviewer-facing claims."
    )

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    with open(args.out_md, "w") as f:
        f.write(markdown(result))
    print(f"Saved JSON: {args.out_json}")
    print(f"Saved markdown: {args.out_md}")
    print(result["decision"])


if __name__ == "__main__":
    main()
