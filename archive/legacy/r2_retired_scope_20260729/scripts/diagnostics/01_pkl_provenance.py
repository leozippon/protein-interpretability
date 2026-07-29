#!/usr/bin/env python
"""Audit R2 ec_features.pkl provenance and CLT dimensionality."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

import torch


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_shape(checkpoint_dir: str) -> dict:
    state = torch.load(os.path.join(checkpoint_dir, "clt.pt"), map_location="cpu")
    return {
        "n_layers": int(state["W_enc"].shape[0]),
        "d_clt": int(state["W_enc"].shape[1]),
        "d_model": int(state["W_enc"].shape[2]),
        "sha256_clt_pt": sha256(os.path.join(checkpoint_dir, "clt.pt")),
    }


def ec_features_shape(path: str) -> dict:
    with open(path, "rb") as f:
        data = pickle.load(f)
    classes = list(data.keys())
    layer_shapes = {}
    for cls in classes:
        rec = data[cls]
        means = rec.get("mean") if isinstance(rec, dict) else None
        if means is None:
            continue
        if isinstance(means, dict):
            iterator = means.items()
        else:
            iterator = enumerate(means)
        for layer, arr in iterator:
            shape = list(getattr(arr, "shape", []))
            layer_shapes.setdefault(str(layer), set()).add(tuple(shape))
    normalized = {
        layer: [list(shape) for shape in sorted(shapes)]
        for layer, shapes in sorted(layer_shapes.items(), key=lambda kv: int(kv[0]))
    }
    d_clt_values = sorted({shape[-1] for shapes in normalized.values() for shape in shapes if shape})
    return {
        "path": path,
        "exists": os.path.exists(path),
        "sha256": sha256(path),
        "n_classes": len(classes),
        "classes": classes,
        "layer_shapes": normalized,
        "d_clt_values": d_clt_values,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ec-features", required=True)
    ap.add_argument("--clt", required=True)
    ap.add_argument("--expected-d-clt", type=int, default=8192)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ec = ec_features_shape(args.ec_features)
    ckpt = checkpoint_shape(args.clt)
    passed = (
        ec["d_clt_values"] == [args.expected_d_clt]
        and ckpt["d_clt"] == args.expected_d_clt
        and ckpt["n_layers"] == 36
    )
    out = {
        "diagnostic": "01_pkl_provenance",
        "ec_features": ec,
        "checkpoint": ckpt,
        "expected_d_clt": args.expected_d_clt,
        "passed": passed,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
