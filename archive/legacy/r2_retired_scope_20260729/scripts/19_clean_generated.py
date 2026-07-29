#!/usr/bin/env python
"""Run CLEAN EC prediction on generated R2 sequences with staged weights."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO / "r2_interpretability_transfer" / "results" / "drug_design" / "ec_lysozyme_leads_v2.json"
DEFAULT_CLEAN_ROOT = Path(os.environ.get(
    "BIOCC_CLEAN_ROOT",
    "/oss-pvc/zhk_zip/biocc/external_resources/ec_metrics/clean/CLEAN",
))
DEFAULT_ESM1B = Path(os.environ.get(
    "BIOCC_CLEAN_ESM1B",
    "/oss-pvc/zhk_zip/biocc/external_resources/ec_metrics/clean/esm1b_checkpoints/esm1b_t33_650M_UR50S.pt",
))
OUT_DIR = REPO / "r2_interpretability_transfer" / "results" / "ec_metrics"
AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_seq(seq: str) -> str:
    return "".join(c for c in str(seq).upper() if c in AA)


def load_records(path: Path, max_records: int | None) -> list[dict]:
    data = json.loads(path.read_text())
    records = []

    def add_many(items, source):
        if not isinstance(items, list):
            return
        for r in items:
            if isinstance(r, str):
                seq = r
                meta = {}
            elif isinstance(r, dict):
                seq = r.get("sequence") or r.get("steered") or r.get("raw_output") or r.get("unsteered")
                meta = r
            else:
                continue
            seq = clean_seq(seq)
            if len(seq) < 30:
                continue
            src = str(meta.get("source", source)) if isinstance(meta, dict) else source
            rid = str(meta.get("id", f"{src}_{len(records):04d}")) if isinstance(meta, dict) else f"{src}_{len(records):04d}"
            records.append({
                "id": rid,
                "source": src,
                "sequence": seq[:1022],
                "meta": meta,
            })

    add_many(data.get("leads"), "steered_leads")
    add_many(data.get("all_records"), "steered_all")
    add_many(data.get("unsteered_baseline"), "unsteered")
    add_many(data.get("records"), "records")
    if max_records is not None:
        records = records[:max_records]
    return records


def maximum_separation(distances: list[float]) -> int:
    if len(distances) <= 1:
        return 0
    arr = np.asarray(distances, dtype=np.float64)
    gamma = np.append(arr[1:], np.repeat(arr[-1], 10))
    sep = np.abs(arr - np.mean(gamma))
    grad = np.abs(sep[:-1] - sep[1:])
    large = np.where(grad > np.mean(grad))[0]
    if len(large) == 0:
        return 0
    idx = int(large[0])
    return 0 if idx >= 5 else idx


def load_clean(clean_root: Path, esm1b_path: Path, device: str):
    import torch
    import esm

    original_torch_load = torch.load

    def torch_load_compat(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_torch_load(*load_args, **load_kwargs)

    torch.load = torch_load_compat
    app = clean_root / "app"
    sys.path.insert(0, str(app / "src"))
    from CLEAN.distance_map import get_dist_map_test
    from CLEAN.model import LayerNormNet
    from CLEAN.utils import get_ec_id_dict

    print("[CLEAN] loading ESM-1b", flush=True)
    esm_model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(esm1b_path))
    esm_model.eval().to(device)
    if device.startswith("cuda"):
        esm_model = esm_model.half()

    print("[CLEAN] loading CLEAN model and EC centers", flush=True)
    clean_model = LayerNormNet(512, 128, device, torch.float32)
    checkpoint = torch.load(app / "data" / "pretrained" / "split100.pth", map_location=device)
    clean_model.load_state_dict(checkpoint)
    clean_model.eval()
    _, ec_id_dict_train = get_ec_id_dict(str(app / "data" / "split100.csv"))
    emb_train = torch.load(app / "data" / "pretrained" / "100.pt", map_location=device)
    return {
        "torch": torch,
        "alphabet": alphabet,
        "esm_model": esm_model,
        "clean_model": clean_model,
        "emb_train": emb_train,
        "ec_id_dict_train": ec_id_dict_train,
        "get_dist_map_test": get_dist_map_test,
        "device": device,
    }


def embed_records(ctx: dict, records: list[dict], toks_per_batch: int) -> dict[str, object]:
    torch = ctx["torch"]
    alphabet = ctx["alphabet"]
    model = ctx["esm_model"]
    device = ctx["device"]
    batch_converter = alphabet.get_batch_converter()
    labels = [r["id"] for r in records]
    seqs = [r["sequence"] for r in records]
    order = sorted(range(len(records)), key=lambda i: len(seqs[i]))
    embeddings = {}
    batch = []
    batch_tokens = 0

    def flush():
        nonlocal batch, batch_tokens
        if not batch:
            return
        pairs = [(labels[i], seqs[i]) for i in batch]
        _, strs, toks = batch_converter(pairs)
        toks = toks.to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.startswith("cuda"), dtype=torch.float16):
            out = model(toks, repr_layers=[33], return_contacts=False)
        reps = out["representations"][33].float().cpu()
        for bi, idx in enumerate(batch):
            seq_len = min(len(strs[bi]), 1022)
            embeddings[labels[idx]] = reps[bi, 1:seq_len + 1].mean(0)
        batch = []
        batch_tokens = 0

    for idx in order:
        cost = len(seqs[idx]) + 2
        if batch and max(batch_tokens, cost) * (len(batch) + 1) > toks_per_batch:
            flush()
        batch.append(idx)
        batch_tokens = max(batch_tokens, cost)
    flush()
    return embeddings


def predict_clean(ctx: dict, embeddings: dict[str, object]) -> dict[str, list[dict]]:
    torch = ctx["torch"]
    device = ctx["device"]
    ids = list(embeddings)
    esm_emb = torch.stack([embeddings[i] for i in ids]).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        model_emb = ctx["clean_model"](esm_emb)
    id_ec_dummy = {i: [] for i in ids}
    dist = ctx["get_dist_map_test"](
        ctx["emb_train"], model_emb, ctx["ec_id_dict_train"], id_ec_dummy, device, torch.float32
    )
    preds = {}
    for seq_id, ec_dist in dist.items():
        ranked = sorted(ec_dist.items(), key=lambda kv: kv[1])[:10]
        cutoff = maximum_separation([float(x[1]) for x in ranked])
        preds[seq_id] = [
            {"ec": ec, "distance": float(d), "rank": i + 1}
            for i, (ec, d) in enumerate(ranked[:cutoff + 1])
        ]
    return preds


def summarize(records: list[dict], preds: dict[str, list[dict]], target_ec: str) -> dict:
    by_source = defaultdict(lambda: {"n": 0, "n_pred": 0, "n_exact": 0, "n_prefix_3": 0, "top_ec": Counter()})
    examples = []
    for r in records:
        src = r["source"]
        ps = preds.get(r["id"], [])
        by_source[src]["n"] += 1
        if ps:
            by_source[src]["n_pred"] += 1
            top = ps[0]["ec"]
            by_source[src]["top_ec"][top] += 1
            if top == target_ec:
                by_source[src]["n_exact"] += 1
            if top.startswith("3.2.1"):
                by_source[src]["n_prefix_3"] += 1
        if len(examples) < 20:
            examples.append({"id": r["id"], "source": src, "length": len(r["sequence"]), "predictions": ps[:5]})
    out = {}
    for src, vals in by_source.items():
        n = vals["n"]
        out[src] = {
            "n": n,
            "n_pred": vals["n_pred"],
            "prediction_rate": vals["n_pred"] / max(n, 1),
            "exact_target_rate": vals["n_exact"] / max(n, 1),
            "glycosidase_prefix_rate": vals["n_prefix_3"] / max(n, 1),
            "top_ec": vals["top_ec"].most_common(10),
        }
    return {"by_source": out, "examples": examples}


def markdown(result: dict) -> str:
    lines = ["# CLEAN EC Prediction On Generated Lysozyme Sequences\n"]
    lines.append("| Source | n | Any prediction | Exact 3.2.1.17 | 3.2.1.x prefix | Top EC calls |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for src, vals in result["by_source"].items():
        top = ", ".join(f"{ec} ({n})" for ec, n in vals["top_ec"][:5])
        lines.append(
            f"| {src} | {vals['n']} | {vals['prediction_rate']:.3f} | "
            f"{vals['exact_target_rate']:.3f} | {vals['glycosidase_prefix_rate']:.3f} | {top} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--clean-root", type=Path, default=DEFAULT_CLEAN_ROOT)
    ap.add_argument("--esm1b", type=Path, default=DEFAULT_ESM1B)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--target-ec", default="3.2.1.17")
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--toks-per-batch", type=int, default=4096)
    ap.add_argument("--out-json", type=Path, default=OUT_DIR / "clean_generated_lysozyme_20260507.json")
    args = ap.parse_args()

    t0 = time.time()
    records = load_records(args.input, args.max_records)
    ctx = load_clean(args.clean_root, args.esm1b, args.device)
    embeddings = embed_records(ctx, records, args.toks_per_batch)
    preds = predict_clean(ctx, embeddings)
    summary = summarize(records, preds, args.target_ec)
    result = {
        "task": "R2 T2-C CLEAN generated-sequence EC prediction",
        "status": "completed",
        "input": str(args.input),
        "clean_root": str(args.clean_root),
        "esm1b": str(args.esm1b),
        "target_ec": args.target_ec,
        "n_records": len(records),
        "elapsed_s": round(time.time() - t0, 1),
        **summary,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n")
    args.out_json.with_suffix(".md").write_text(markdown(result))
    print(f"Saved {args.out_json}")
    print(f"Saved {args.out_json.with_suffix('.md')}")


if __name__ == "__main__":
    main()
