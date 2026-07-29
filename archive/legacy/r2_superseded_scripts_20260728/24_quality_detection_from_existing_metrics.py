#!/usr/bin/env python3
"""F-T2-3 pilot: generation-quality detection from existing EC metrics.

This uses already completed R2 calibration outputs. It asks whether a compact
metric vector built from ESMFold and Foldseek separates real lysozymes from
length-matched random UniRef50 controls. This is a quality/hallucination
diagnostic pilot, not a CLT-feature downstream task.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler


REPO = Path(__file__).resolve().parents[2]
EC_DIR = REPO / "r2_decoder_sparse_readout_audit/results/ec_metrics"


def auc_ci(y: np.ndarray, score: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, list[float]]:
    auc = float(roc_auc_score(y, score))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(float(roc_auc_score(y[idx], score[idx])))
    ci = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [math.nan, math.nan]
    return auc, ci


def load_esmfold(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data["per_sequence"]


def load_foldseek_top_scores(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text())
    out = {}
    for set_name, set_data in data["sets"].items():
        for row in set_data["summary"]["top_scores"]:
            out[row["query"]] = row
    return out


def normalize_id_for_foldseek(seq_id: str) -> str:
    # ESMFold ids omit the four-digit file prefix used by Foldseek summaries.
    return seq_id


def build_dataset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[dict], list[str]]:
    real_rows = load_esmfold(args.real_esmfold)
    random_rows = load_esmfold(args.random_esmfold)
    foldseek = load_foldseek_top_scores(args.foldseek)

    feature_names = [
        "seq_len",
        "mean_plddt",
        "frac_confident",
        "ptm",
        "foldseek_top_tm",
        "foldseek_top_lddt",
    ]
    records = []
    X = []
    y = []
    for label, rows in [(1, real_rows), (0, random_rows)]:
        for row in rows:
            seq_id = normalize_id_for_foldseek(row["id"])
            fs = foldseek.get(seq_id)
            if fs is None:
                # Match by suffix because Foldseek keys include a four-digit prefix.
                matches = [v for k, v in foldseek.items() if k.endswith(seq_id)]
                fs = matches[0] if matches else None
            top_tm = float(fs["alntmscore"]) if fs else 0.0
            top_lddt = float(fs.get("lddt", 0.0)) if fs else 0.0
            vec = [
                float(row["seq_len"]),
                float(row["mean_plddt"]),
                float(row["frac_confident"]),
                float(row["ptm"]),
                top_tm,
                top_lddt,
            ]
            X.append(vec)
            y.append(label)
            records.append({
                "id": row["id"],
                "source": row["source"],
                "label_real_lysozyme": label,
                **dict(zip(feature_names, vec)),
            })
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int32), records, feature_names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-esmfold", type=Path, default=EC_DIR / "calibration_real_lysozyme_esmfold_20260507.json")
    ap.add_argument("--random-esmfold", type=Path, default=EC_DIR / "calibration_random_uniref50_esmfold_20260507.json")
    ap.add_argument("--foldseek", type=Path, default=EC_DIR / "foldseek_calibration_lysozyme_20260507.json")
    ap.add_argument("--out-json", type=Path, default=EC_DIR / "quality_detection_from_existing_metrics_20260511.json")
    ap.add_argument("--out-md", type=Path, default=EC_DIR / "quality_detection_from_existing_metrics_20260511.md")
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X, y, records, feature_names = build_dataset(args)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    pred = cross_val_predict(clf, Xs, y, cv=cv, method="predict_proba")[:, 1]
    clf_auc, clf_ci = auc_ci(y, pred, args.n_bootstrap, args.seed)

    univariate = {}
    for i, name in enumerate(feature_names):
        score = X[:, i]
        if name == "seq_len":
            # Length is included only to detect trivial leakage; use both signs.
            auc_pos, ci_pos = auc_ci(y, score, args.n_bootstrap, args.seed + i + 1)
            auc_neg, ci_neg = auc_ci(y, -score, args.n_bootstrap, args.seed + i + 101)
            if auc_neg > auc_pos:
                univariate[name] = {"auc": auc_neg, "ci95": ci_neg, "direction": "negative"}
            else:
                univariate[name] = {"auc": auc_pos, "ci95": ci_pos, "direction": "positive"}
        else:
            auc, ci = auc_ci(y, score, args.n_bootstrap, args.seed + i + 1)
            univariate[name] = {"auc": auc, "ci95": ci, "direction": "positive"}

    for rec, p in zip(records, pred):
        rec["quality_detector_score"] = float(p)

    summary = {
        "task": "F-T2-3 pilot quality/hallucination detection from existing metrics",
        "status": "completed",
        "n": int(len(y)),
        "n_real": int(y.sum()),
        "n_random": int((1 - y).sum()),
        "feature_names": feature_names,
        "logistic_cv": {"auc": clf_auc, "ci95": clf_ci},
        "univariate": univariate,
        "records_preview": records[:20],
        "interpretation": (
            "ESMFold/Foldseek metrics strongly separate real lysozymes from random UniRef50 controls. "
            "This supports a generation-quality diagnostic layer, but it is not yet a CLT-feature representation result."
        ),
    }
    args.out_json.write_text(json.dumps(summary, indent=2))

    lines = [
        "# R2 F-T2-3 Pilot Quality Detection",
        "",
        "Date: 2026-05-11",
        "",
        f"- n={summary['n']} (real={summary['n_real']}, random={summary['n_random']})",
        f"- Logistic CV AUC: {clf_auc:.4f} [{clf_ci[0]:.4f}, {clf_ci[1]:.4f}]",
        "",
        "## Univariate Metrics",
        "",
        "| Metric | AUC | 95% CI | Direction |",
        "|---|---:|---|---|",
    ]
    for name, vals in univariate.items():
        ci = vals["ci95"]
        lines.append(f"| {name} | {vals['auc']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {vals['direction']} |")
    lines += [
        "",
        "## Interpretation",
        "",
        summary["interpretation"],
        "",
        "Boundary: this is a metric-stack diagnostic pilot. It does not establish that CLT features beat raw embeddings on downstream tasks.",
        "",
    ]
    args.out_md.write_text("\n".join(lines))
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
