#!/usr/bin/env python3
"""Resource-ready CLT representation pilot on lysozyme calibration sequences.

This is a small downstream-representation probe that does not require CB513,
DeepLoc or stability datasets. It compares pooled CLT features against pooled
raw hidden features for real EC 3.2.1.17 lysozymes versus length-matched
random UniRef50 proteins.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler


REPO = Path(__file__).resolve().parents[2]
R2 = REPO / "r2_decoder_sparse_readout_audit"
sys.path.insert(0, str(R2))

os.environ.setdefault("R2_MODEL_BASE_DIR", "/gpfs/jiaotongdamoxing/zhk_zip/models")

from src.models.model_loader import load_model  # noqa: E402
from src.training.clt_trainer import CLTForTraining  # noqa: E402


def load_clt(ckpt_dir: Path, device: str) -> CLTForTraining:
    with (ckpt_dir / "config.yaml").open() as f:
        config = yaml.safe_load(f)
    state = torch.load(ckpt_dir / "clt.pt", map_location=device)
    clt_cfg = config["clt"]
    clt = CLTForTraining(
        n_layers=int(state["W_enc"].shape[0]),
        d_model=int(state["W_enc"].shape[2]),
        d_clt=int(clt_cfg["d_clt"]),
        k=int(clt_cfg["k"]),
        window=int(clt_cfg.get("window", 8)),
    )
    clt.load_state_dict(state)
    clt.to(device).eval()
    return clt


def load_records(path: Path, limit_per_class: int | None) -> list[dict]:
    data = json.loads(path.read_text())
    by_source: dict[str, list[dict]] = {}
    for row in data["records"]:
        by_source.setdefault(row["source"], []).append(row)
    records = []
    for source in ["real_lysozyme", "random_uniref50"]:
        rows = by_source[source]
        if limit_per_class:
            rows = rows[:limit_per_class]
        records.extend(rows)
    return records


def auc_ci(y: np.ndarray, score: np.ndarray, n_boot: int, seed: int) -> tuple[float, list[float]]:
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


def linear_probe(X: np.ndarray, y: np.ndarray, seed: int, n_boot: int) -> dict:
    good = X.std(axis=0) > 1e-8
    X = X[:, good]
    X = StandardScaler().fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    clf = LogisticRegression(C=0.1, max_iter=2000, solver="liblinear")
    pred = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    auc, ci = auc_ci(y, pred, n_boot, seed)
    return {
        "auc": auc,
        "ci95": ci,
        "n_features_after_constant_filter": int(X.shape[1]),
        "predictions": pred.tolist(),
    }


@torch.no_grad()
def extract_features(records: list[dict], model, clt: CLTForTraining, layers: list[int], device: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    raw_rows = []
    clt_rows = []
    meta = []
    for i, row in enumerate(records, start=1):
        seq = row["sequence"]
        toks = model.tokenizer(seq, return_tensors="pt", truncation=True, max_length=256)
        input_ids = toks["input_ids"].to(device)
        cache = model.get_activations(input_ids)
        resid_pre = [x.float() for x in cache.resid_pre]
        mlp_out = [x.float() for x in cache.mlp_out]
        features = clt.encode(resid_pre)

        raw_vec = []
        clt_vec = []
        for layer in layers:
            raw_vec.append(resid_pre[layer][0].mean(dim=0).detach().cpu().numpy())
            clt_vec.append(features[layer][0].mean(dim=0).detach().cpu().numpy())
        raw_rows.append(np.concatenate(raw_vec).astype(np.float32))
        clt_rows.append(np.concatenate(clt_vec).astype(np.float32))
        meta.append({
            "id": row["id"],
            "source": row["source"],
            "label": 1 if row["source"] == "real_lysozyme" else 0,
            "length": len(seq),
        })
        if i % 20 == 0:
            print(f"extracted {i}/{len(records)}", flush=True)

    return np.stack(raw_rows), np.stack(clt_rows), meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=REPO / "r2_decoder_sparse_readout_audit/results/ec_metrics/calibration_lysozyme_20260507/calibration_sequences.json")
    ap.add_argument("--ckpt", type=Path, default=Path("/oss-pvc/zhk_zip/outputs/research2/clt_weights/zymctrl_v2/step_200000"))
    ap.add_argument("--model", default="zymctrl")
    ap.add_argument("--layers", default="3,12,30")
    ap.add_argument("--limit-per-class", type=int, default=100)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", type=Path, default=REPO / "r2_decoder_sparse_readout_audit/results/ec_metrics/clt_representation_lysozyme_probe_20260511.json")
    ap.add_argument("--out-md", type=Path, default=REPO / "r2_decoder_sparse_readout_audit/results/ec_metrics/clt_representation_lysozyme_probe_20260511.md")
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    records = load_records(args.records, args.limit_per_class)
    y = np.asarray([1 if r["source"] == "real_lysozyme" else 0 for r in records], dtype=np.int32)

    print(f"device={device} model={args.model} n={len(records)} layers={layers}", flush=True)
    protein_model = load_model(args.model, device=device, dtype=torch.float16 if device.startswith("cuda") else torch.float32)
    clt = load_clt(args.ckpt, device)
    raw_X, clt_X, meta = extract_features(records, protein_model, clt, layers, device)

    raw_probe = linear_probe(raw_X, y, args.seed, args.n_bootstrap)
    clt_probe = linear_probe(clt_X, y, args.seed + 1, args.n_bootstrap)

    summary = {
        "task": f"F-T2-1 resource-ready pilot: {args.model} CLT representation on lysozyme-vs-random calibration",
        "status": "completed",
        "model": args.model,
        "checkpoint": str(args.ckpt),
        "layers": layers,
        "n": int(len(y)),
        "n_real": int(y.sum()),
        "n_random": int((1 - y).sum()),
        "raw_hidden_probe": {k: v for k, v in raw_probe.items() if k != "predictions"},
        "clt_probe": {k: v for k, v in clt_probe.items() if k != "predictions"},
        "delta_clt_minus_raw": float(clt_probe["auc"] - raw_probe["auc"]),
        "meta_preview": meta[:20],
        "interpretation": (
            "This is a narrow resource-ready pilot. It tests whether pooled CLT features are usable for a lysozyme-vs-random discrimination task, "
            "but it does not replace the planned five-task downstream benchmark."
        ),
    }
    args.out_json.write_text(json.dumps(summary, indent=2))

    lines = [
        "# R2 CLT Representation Lysozyme Probe",
        "",
        "Date: 2026-05-11",
        "",
        f"- Model: {args.model}",
        f"- Layers: {layers}",
        f"- n={summary['n']} (real={summary['n_real']}, random={summary['n_random']})",
        "",
        "| Representation | AUC | 95% CI | Features |",
        "|---|---:|---|---:|",
    ]
    for name, vals in [("raw hidden", summary["raw_hidden_probe"]), ("CLT pooled", summary["clt_probe"])]:
        ci = vals["ci95"]
        lines.append(f"| {name} | {vals['auc']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {vals['n_features_after_constant_filter']} |")
    lines += [
        "",
        f"Delta CLT minus raw: {summary['delta_clt_minus_raw']:.4f}",
        "",
        "Interpretation: " + summary["interpretation"],
        "",
    ]
    args.out_md.write_text("\n".join(lines))
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
