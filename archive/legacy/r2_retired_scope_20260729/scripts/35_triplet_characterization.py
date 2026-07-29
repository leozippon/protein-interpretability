#!/usr/bin/env python3
"""M-1 characterization of conserved universal triplets.

After the biological-label and downstream-probe gates failed, this script asks
what the 38 conserved triplets do encode: k-mer context, normalized position,
ProtGPT2 token-boundary artifacts, attention-sink behavior, or high hidden-state
norm.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
R2 = REPO / "r2_interpretability_transfer"
AA = set("ACDEFGHIKLMNPQRSTVWY")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


s33 = load_module(SCRIPT_DIR / "33_swissprot_triplet_annotation.py", "swissprot_triplet_annotation_33")
u29 = s33.u29


def clean_sequence(seq: str) -> str:
    return "".join(c for c in (seq or "").upper() if c in AA)


def load_json_records(path: Path, field: str = "records", max_sequences: int | None = None, max_length: int = 400) -> list[dict]:
    if max_sequences is not None and max_sequences <= 0:
        return []
    data = json.loads(path.read_text())
    cur = data
    for part in field.split("."):
        if part:
            cur = cur[part]
    rows = []
    for i, row in enumerate(cur):
        if isinstance(row, str):
            rec = {"id": f"{path.stem}_{i:05d}", "source": path.stem, "sequence": row, "meta": {}}
        else:
            rec = {
                "id": str(row.get("id") or row.get("accession") or f"{path.stem}_{i:05d}"),
                "source": str(row.get("source") or path.stem),
                "sequence": row.get("sequence") or row.get("seq") or row.get("protein_sequence") or "",
                "meta": row.get("meta", {}),
            }
        seq = clean_sequence(rec["sequence"])[:max_length]
        if seq:
            rec["sequence"] = seq
            rows.append(rec)
        if max_sequences is not None and len(rows) >= max_sequences:
            break
    return rows


def load_or_build_cohort(args) -> tuple[list[dict], dict]:
    records = []
    seen = set()
    meta = {"sources": []}

    if args.n1_cohort.exists():
        data = json.loads(args.n1_cohort.read_text())
        for row in data["records"][: args.max_swissprot]:
            seq = clean_sequence(row["sequence"])[: args.max_length]
            if not seq:
                continue
            rec = dict(row)
            rec["id"] = str(rec.get("id") or rec.get("accession"))
            rec["sequence"] = seq
            rec["source"] = "swissprot_n1"
            if rec["id"] not in seen:
                records.append(rec)
                seen.add(rec["id"])
        meta["sources"].append({"name": "n1_cohort", "path": str(args.n1_cohort), "n": len(records)})
    else:
        seqs, _ann_by_acc, _pfam, cohort_meta = s33.choose_cohort(
            args.swissprot_cache, args.pfam_residue, args.max_swissprot, args.min_len, args.max_length, args.seed
        )
        for row in seqs:
            row = dict(row)
            row["source"] = "swissprot_rebuilt"
            if row["id"] not in seen:
                records.append(row)
                seen.add(row["id"])
        meta["sources"].append({"name": "rebuilt_swissprot", "meta": cohort_meta, "n": len(seqs)})

    balanced = load_json_records(args.balanced_json, args.balanced_json_field, args.max_balanced, args.max_length)
    n_balanced = 0
    for row in balanced:
        key = row["id"]
        if key in seen:
            key = f"{row['source']}:{key}"
            row["id"] = key
        if key not in seen:
            records.append(row)
            seen.add(key)
            n_balanced += 1
    meta["sources"].append({"name": "balanced_calibration", "path": str(args.balanced_json), "n": n_balanced})
    meta["n_records"] = len(records)
    return records, meta


def kmer(seq: str, pos0: int, k: int) -> str:
    half = k // 2
    start = pos0 - half
    end = pos0 + half + 1
    if start < 0 or end > len(seq):
        return "edge"
    return seq[start:end]


def labels_to_codes(labels: list[str]) -> tuple[np.ndarray, int]:
    mp = {x: i for i, x in enumerate(sorted(set(labels)))}
    return np.asarray([mp[x] for x in labels], dtype=np.int32), len(mp)


def mi_encoded(codes: np.ndarray, binary: np.ndarray, n_labels: int) -> float:
    if codes.size == 0 or n_labels == 0:
        return 0.0
    joint = np.bincount(codes * 2 + binary.astype(np.int32), minlength=n_labels * 2).reshape(n_labels, 2)
    total = float(joint.sum())
    if total <= 0:
        return 0.0
    pxy = joint / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    denom = px @ py
    mask = (pxy > 0) & (denom > 0)
    return float(np.sum(pxy[mask] * np.log(pxy[mask] / denom[mask])))


def top_binary(values: np.ndarray, top_n: int) -> tuple[np.ndarray, np.ndarray]:
    n = int(values.size)
    top_n = min(top_n, n)
    order = np.argsort(-values)
    binary = np.zeros(n, dtype=np.int8)
    binary[order[:top_n]] = 1
    return binary, order[:top_n]


def subset_for_permutation(binary: np.ndarray, max_positions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    top = np.flatnonzero(binary == 1)
    bg = np.flatnonzero(binary == 0)
    room = max(0, max_positions - len(top))
    if len(bg) > room:
        bg = rng.choice(bg, size=room, replace=False)
    idx = np.concatenate([top, bg])
    rng.shuffle(idx)
    return idx


def kmer_test(labels: list[str], binary: np.ndarray, seed: int, n_perm: int, max_positions: int) -> dict:
    idx = subset_for_permutation(binary, max_positions, seed)
    b = binary[idx]
    codes, n_labels = labels_to_codes([labels[i] for i in idx])
    obs = mi_encoded(codes, b, n_labels)
    rng = np.random.default_rng(seed + 17)
    top_n = int(b.sum())
    vals = []
    for _ in range(n_perm):
        perm = np.zeros_like(b)
        perm[rng.choice(len(b), size=top_n, replace=False)] = 1
        vals.append(mi_encoded(codes, perm, n_labels))
    p = (1 + sum(v >= obs for v in vals)) / (n_perm + 1)
    return {"mi_nats": obs, "p": float(p), "n_labels": int(n_labels), "n_positions": int(len(idx))}


def ks_uniform_p(values: np.ndarray, seed: int, n_perm: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return {"stat": 0.0, "p": 1.0}
    vals = np.sort(np.clip(values, 0.0, 1.0))
    n = len(vals)
    ecdf_hi = np.arange(1, n + 1) / n
    ecdf_lo = np.arange(0, n) / n
    obs = float(max(np.max(np.abs(ecdf_hi - vals)), np.max(np.abs(vals - ecdf_lo))))
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        sim = np.sort(rng.random(n))
        stat = max(np.max(np.abs(ecdf_hi - sim)), np.max(np.abs(sim - ecdf_lo)))
        ge += stat >= obs
    return {"stat": obs, "p": float((ge + 1) / (n_perm + 1))}


def binary_enrichment_p(label: np.ndarray, binary: np.ndarray, seed: int, n_perm: int) -> dict:
    label = np.asarray(label, dtype=np.int8)
    if label.size == 0 or len(np.unique(label)) < 2:
        return {"top_rate": math.nan, "background_rate": math.nan, "delta": 0.0, "p": 1.0}
    top = binary == 1
    if top.sum() == 0 or (~top).sum() == 0:
        return {"top_rate": math.nan, "background_rate": math.nan, "delta": 0.0, "p": 1.0}
    top_rate = float(label[top].mean())
    bg_rate = float(label[~top].mean())
    obs = top_rate - bg_rate
    if obs <= 0:
        return {"top_rate": top_rate, "background_rate": bg_rate, "delta": float(obs), "p": 1.0}
    rng = np.random.default_rng(seed)
    top_n = int(top.sum())
    vals = []
    for _ in range(n_perm):
        idx = rng.choice(label.size, size=top_n, replace=False)
        mask = np.zeros(label.size, dtype=bool)
        mask[idx] = True
        vals.append(float(label[mask].mean()) - float(label[~mask].mean()))
    p = (1 + sum(v >= obs for v in vals)) / (n_perm + 1)
    return {"top_rate": top_rate, "background_rate": bg_rate, "delta": float(top_rate - bg_rate), "p": float(p)}


def pearson_p(x: np.ndarray, y: np.ndarray, seed: int, n_perm: int, max_positions: int) -> dict:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size > max_positions:
        rng = np.random.default_rng(seed)
        idx = rng.choice(x.size, size=max_positions, replace=False)
        x = x[idx]
        y = y[idx]
    if x.size < 10 or x.std() <= 1e-8 or y.std() <= 1e-8:
        return {"r": 0.0, "p": 1.0, "n": int(x.size)}
    xz = (x - x.mean()) / (x.std() + 1e-8)
    yz = (y - y.mean()) / (y.std() + 1e-8)
    obs = float((xz * yz).mean())
    if obs <= 0:
        return {"r": obs, "p": 1.0, "n": int(x.size)}
    rng = np.random.default_rng(seed + 31)
    ge = 0
    for _ in range(n_perm):
        ge += float((xz * rng.permutation(yz)).mean()) >= obs
    return {"r": obs, "p": float((ge + 1) / (n_perm + 1)), "n": int(x.size)}


def bh_adjust(pvals: list[float]) -> list[float]:
    n = len(pvals)
    order = np.argsort(np.asarray(pvals, dtype=np.float64))
    q = np.ones(n, dtype=np.float64)
    prev = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        true_rank = n - rank + 1
        val = min(prev, pvals[idx] * n / max(true_rank, 1))
        q[idx] = val
        prev = val
    return q.tolist()


def token_boundary_values(spans: list[list[int]], length: int) -> np.ndarray:
    arr = np.zeros(length, dtype=np.int8)
    for span in spans:
        if not span:
            continue
        arr[span[0]] = 1
        arr[span[-1]] = 1
    return arr


@torch.no_grad()
def collect_model_signals(model_name: str, ckpt: str, records: list[dict], triplets: list[dict], device: str) -> dict:
    needed = {t["triplet_id"]: t["features"][model_name] for t in triplets if model_name in t["features"]}
    if not needed:
        return {}

    print(f"\n[{model_name}] loading model and CLT", flush=True)
    pm = u29.load_model(model_name, device=device)
    clt = u29.load_trained_clt(ckpt, device=device)
    out = {
        tid: {"activation": [], "attention": [], "hidden_norm": [], "bpe_boundary": []}
        for tid in needed
    }
    t0 = time.time()

    for i, rec in enumerate(records):
        seq = rec["sequence"]
        ids = pm.tokenize(seq)
        spans = u29.token_residue_spans(pm.tokenizer, ids, seq)
        boundary = token_boundary_values(spans, len(seq))
        cache = pm.get_activations(ids)
        features = clt.encode([x.float() for x in cache.resid_pre])

        attn_by_layer = {}
        try:
            attn_out = pm.model(ids.to(device), output_attentions=True, use_cache=False)
            if getattr(attn_out, "attentions", None) is not None:
                for layer_idx, attn in enumerate(attn_out.attentions):
                    if attn is None:
                        continue
                    # heads x query x key -> mean attention received by key token
                    received = attn[0].detach().float().mean(dim=0).mean(dim=0).cpu().numpy()
                    attn_by_layer[layer_idx] = received
        except Exception as exc:
            if i == 0:
                print(f"  warning: attention extraction unavailable for {model_name}: {exc}", flush=True)

        for tid, spec in needed.items():
            layer = int(spec["layer"])
            feat_idx = int(spec["feature"])
            act_tok = features[layer][0, :, feat_idx].detach().float().cpu().numpy()
            norm_tok = cache.resid_pre[layer][0].detach().float().norm(dim=-1).cpu().numpy()
            attn_tok = attn_by_layer.get(layer, np.full_like(act_tok, np.nan, dtype=np.float32))
            out[tid]["activation"].append(u29.add_token_values_to_residues(act_tok, spans, len(seq)))
            out[tid]["hidden_norm"].append(u29.add_token_values_to_residues(norm_tok, spans, len(seq)))
            out[tid]["attention"].append(u29.add_token_values_to_residues(attn_tok, spans, len(seq)))
            if model_name == "protgpt2":
                out[tid]["bpe_boundary"].append(boundary.astype(np.float32))

        if (i + 1) % 10 == 0 or i == len(records) - 1:
            print(f"  {model_name}: {i+1}/{len(records)} sequences ({time.time()-t0:.1f}s)", flush=True)

    del pm
    del clt
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def z_arrays(arrays: list[np.ndarray]) -> list[np.ndarray]:
    finite_parts = [a[np.isfinite(a)] for a in arrays if np.isfinite(a).any()]
    if not finite_parts:
        return [np.full_like(a, np.nan, dtype=np.float32) for a in arrays]
    flat = np.concatenate(finite_parts)
    mu = float(flat.mean())
    sd = float(flat.std() + 1e-6)
    return [(a - mu) / sd for a in arrays]


def consensus_signal(model_signals: dict[str, dict], tid: str, key: str, n_records: int) -> list[np.ndarray]:
    per_model = []
    for signals in model_signals.values():
        if tid not in signals:
            continue
        arrays = signals[tid].get(key, [])
        if not arrays:
            continue
        per_model.append(z_arrays(arrays))

    out = []
    for i in range(n_records):
        arrs = [m[i] for m in per_model if i < len(m)]
        if not arrs:
            out.append(np.zeros(0, dtype=np.float32))
            continue
        max_len = max(len(a) for a in arrs)
        vals = np.full((len(arrs), max_len), np.nan, dtype=np.float32)
        for j, arr in enumerate(arrs):
            vals[j, : len(arr)] = arr
        out.append(np.nanmean(vals, axis=0))
    return out


def flatten_for_triplet(records: list[dict], activation: list[np.ndarray], attention: list[np.ndarray], hidden_norm: list[np.ndarray]) -> dict:
    vals = []
    attn = []
    norms = []
    pos_norm = []
    k3 = []
    k5 = []
    row_meta = []
    for seq_idx, rec in enumerate(records):
        seq = rec["sequence"]
        n = min(len(seq), len(activation[seq_idx]))
        for pos0 in range(n):
            v = float(activation[seq_idx][pos0])
            if not np.isfinite(v):
                continue
            vals.append(v)
            attn.append(float(attention[seq_idx][pos0]) if pos0 < len(attention[seq_idx]) else math.nan)
            norms.append(float(hidden_norm[seq_idx][pos0]) if pos0 < len(hidden_norm[seq_idx]) else math.nan)
            pos_norm.append((pos0 + 0.5) / max(len(seq), 1))
            k3.append(kmer(seq, pos0, 3))
            k5.append(kmer(seq, pos0, 5))
            row_meta.append((seq_idx, pos0, seq[max(0, pos0 - 2) : min(len(seq), pos0 + 3)]))
    return {
        "activation": np.asarray(vals, dtype=np.float32),
        "attention": np.asarray(attn, dtype=np.float32),
        "hidden_norm": np.asarray(norms, dtype=np.float32),
        "pos_norm": np.asarray(pos_norm, dtype=np.float32),
        "k3": k3,
        "k5": k5,
        "row_meta": row_meta,
    }


def analyze(records: list[dict], triplets: list[dict], model_signals: dict, args) -> dict:
    per_triplet = []
    top_rows = []
    p_entries = []
    category_order = ["k-mer", "positional", "bpe-boundary", "attention-sink", "high-norm"]

    for t in triplets:
        tid = t["triplet_id"]
        activation = consensus_signal(model_signals, tid, "activation", len(records))
        attention = consensus_signal(model_signals, tid, "attention", len(records))
        hidden_norm = consensus_signal(model_signals, tid, "hidden_norm", len(records))
        flat = flatten_for_triplet(records, activation, attention, hidden_norm)
        values = flat["activation"]
        binary, top_idx = top_binary(values, args.top_positions)
        seed = args.seed + int(t["rank"]) * 101

        k3_res = kmer_test(flat["k3"], binary, seed, args.n_perm, args.max_perm_positions)
        k5_res = kmer_test(flat["k5"], binary, seed + 1, args.n_perm, args.max_perm_positions)
        if k5_res["p"] < k3_res["p"]:
            kmer_res = {"best_k": 5, **k5_res}
        else:
            kmer_res = {"best_k": 3, **k3_res}

        pos_res = ks_uniform_p(flat["pos_norm"][top_idx], seed + 2, args.n_perm)
        attn_res = pearson_p(values, flat["attention"], seed + 3, args.n_perm, args.max_corr_positions)
        norm_res = pearson_p(values, flat["hidden_norm"], seed + 4, args.n_perm, args.max_corr_positions)

        bpe_res = {"top_rate": math.nan, "background_rate": math.nan, "delta": 0.0, "p": 1.0}
        prot = model_signals.get("protgpt2", {})
        if tid in prot and prot[tid].get("activation") and prot[tid].get("bpe_boundary"):
            prot_act = z_arrays(prot[tid]["activation"])
            prot_flat = []
            prot_boundary = []
            for seq_idx, rec in enumerate(records):
                n = min(len(prot_act[seq_idx]), len(prot[tid]["bpe_boundary"][seq_idx]))
                for pos0 in range(n):
                    if np.isfinite(prot_act[seq_idx][pos0]):
                        prot_flat.append(float(prot_act[seq_idx][pos0]))
                        prot_boundary.append(int(prot[tid]["bpe_boundary"][seq_idx][pos0] > 0))
            if prot_flat:
                prot_binary, _ = top_binary(np.asarray(prot_flat, dtype=np.float32), args.top_positions)
                bpe_res = binary_enrichment_p(np.asarray(prot_boundary, dtype=np.int8), prot_binary, seed + 5, args.n_perm)

        test_map = {
            "k-mer": kmer_res["p"],
            "positional": pos_res["p"],
            "bpe-boundary": bpe_res["p"],
            "attention-sink": attn_res["p"],
            "high-norm": norm_res["p"],
        }
        for cat, p in test_map.items():
            p_entries.append((tid, cat, p))

        top_k3 = Counter(flat["k3"][i] for i in top_idx).most_common(5)
        top_k5 = Counter(flat["k5"][i] for i in top_idx).most_common(5)
        row = {
            "triplet_id": tid,
            "rank": int(t["rank"]),
            "min_abs_corr": float(t["min_abs_corr"]),
            "n_positions": int(values.size),
            "kmer_best_k": kmer_res["best_k"],
            "kmer_mi_nats": kmer_res["mi_nats"],
            "kmer_p": kmer_res["p"],
            "positional_ks": pos_res["stat"],
            "positional_p": pos_res["p"],
            "bpe_top_rate": bpe_res["top_rate"],
            "bpe_bg_rate": bpe_res["background_rate"],
            "bpe_p": bpe_res["p"],
            "attention_r": attn_res["r"],
            "attention_p": attn_res["p"],
            "attention_n": attn_res["n"],
            "hidden_norm_r": norm_res["r"],
            "hidden_norm_p": norm_res["p"],
            "hidden_norm_n": norm_res["n"],
            "top_k3": json.dumps(top_k3),
            "top_k5": json.dumps(top_k5),
        }
        per_triplet.append(row)

        for rank, idx in enumerate(top_idx[: args.top_position_rows], start=1):
            seq_idx, pos0, context = flat["row_meta"][int(idx)]
            top_rows.append(
                {
                    "triplet_id": tid,
                    "top_rank": rank,
                    "sequence_id": records[seq_idx]["id"],
                    "source": records[seq_idx]["source"],
                    "position_1based": pos0 + 1,
                    "sequence_length": len(records[seq_idx]["sequence"]),
                    "position_norm": f"{flat['pos_norm'][idx]:.6g}",
                    "activation_z": f"{values[idx]:.6g}",
                    "k3": flat["k3"][idx],
                    "k5": flat["k5"][idx],
                    "context": context,
                }
            )
        print(f"  analyzed {tid}: kmer_p={kmer_res['p']:.4g} pos_p={pos_res['p']:.4g} attn_r={attn_res['r']:.3g}", flush=True)

    qvals = bh_adjust([x[2] for x in p_entries])
    q_lookup = {(tid, cat): q for (tid, cat, _p), q in zip(p_entries, qvals)}
    for row in per_triplet:
        q_by_cat = {cat: q_lookup[(row["triplet_id"], cat)] for cat in category_order}
        p_by_cat = {
            "k-mer": row["kmer_p"],
            "positional": row["positional_p"],
            "bpe-boundary": row["bpe_p"],
            "attention-sink": row["attention_p"],
            "high-norm": row["hidden_norm_p"],
        }
        significant = [(cat, q_by_cat[cat], p_by_cat[cat]) for cat in category_order if q_by_cat[cat] < args.q_threshold]
        if significant:
            significant.sort(key=lambda x: (x[1], x[2], category_order.index(x[0])))
            category = significant[0][0]
        else:
            category = "unknown"
        row["assigned_category"] = category
        for cat in category_order:
            row[f"{cat.replace('-', '_')}_q"] = q_by_cat[cat]

    counts = Counter(row["assigned_category"] for row in per_triplet)
    categorized = sum(v for k, v in counts.items() if k != "unknown")
    if categorized >= 25:
        outcome = "PASS"
    elif categorized >= 10:
        outcome = "PARTIAL"
    else:
        outcome = "FAIL"
    return {
        "per_triplet": per_triplet,
        "top_rows": top_rows,
        "category_counts": dict(counts),
        "n_categorized": int(categorized),
        "outcome": outcome,
    }


def write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--triplets", type=Path, default=R2 / "results/circuit_analysis/universal_atlas_balanced200_wide_triplets_20260512.tsv")
    ap.add_argument("--n1-cohort", type=Path, default=R2 / "results/circuit_analysis/swissprot_triplet_annotation_20260513/cohort.json")
    ap.add_argument("--balanced-json", type=Path, default=R2 / "results/ec_metrics/calibration_lysozyme_20260507/calibration_sequences.json")
    ap.add_argument("--balanced-json-field", default="records")
    ap.add_argument("--swissprot-cache", type=Path, default=REPO / "data/processed/swissprot_all_max1022.pkl")
    ap.add_argument("--pfam-residue", type=Path, default=REPO / "data/interpro/pfam_residue.tsv")
    ap.add_argument("--max-swissprot", type=int, default=500)
    ap.add_argument("--max-balanced", type=int, default=200)
    ap.add_argument("--min-len", type=int, default=100)
    ap.add_argument("--max-length", type=int, default=400)
    ap.add_argument("--max-triplets", type=int, default=38)
    ap.add_argument("--top-positions", type=int, default=100)
    ap.add_argument("--top-position-rows", type=int, default=20)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--max-perm-positions", type=int, default=50000)
    ap.add_argument("--max-corr-positions", type=int, default=50000)
    ap.add_argument("--q-threshold", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20260514)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", type=Path, default=R2 / "results/circuit_analysis/triplet_characterization_20260514")
    ap.add_argument("--model-spec", action="append", default=[
        "protgpt2=/oss-pvc/zhk_zip/outputs/research2/clt_weights/protgpt2_v2/step_200000",
        "zymctrl=/oss-pvc/zhk_zip/outputs/research2/clt_weights/zymctrl_v2/step_200000",
        "progen2-medium=/gpfs/jiaotongdamoxing/zhk_zip/biocc/paper_r2_nature_mi/results/final_checkpoints/r2_clt_progen2_medium_rerun_20260403/clt_weights/progen2-medium/step_100000",
    ])
    args = ap.parse_args()

    os.environ.setdefault("R2_MODEL_BASE_DIR", "/gpfs/jiaotongdamoxing/zhk_zip/models")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("M-1 triplet characterization", flush=True)
    records, cohort_meta = load_or_build_cohort(args)
    triplets = u29.read_triplets(args.triplets, args.max_triplets)
    (args.out_dir / "cohort.json").write_text(json.dumps({"meta": cohort_meta, "records": records}, indent=2) + "\n")
    print(json.dumps({"cohort": cohort_meta, "n_triplets": len(triplets)}, indent=2), flush=True)

    model_signals = {}
    for model_name, ckpt in [u29.parse_model_spec(x) for x in args.model_spec]:
        model_signals[model_name] = collect_model_signals(model_name, ckpt, records, triplets, args.device)

    result = analyze(records, triplets, model_signals, args)
    write_tsv(args.out_dir / "triplet_characterization.tsv", result["per_triplet"])
    write_tsv(args.out_dir / "top_firing_positions.tsv", result["top_rows"])

    summary = {
        "task": "M-1 triplet characterization: k-mer / positional / BPE-boundary / attention-sink / high-norm",
        "status": "completed",
        "runtime_seconds": time.time() - t0,
        "cohort_meta": cohort_meta,
        "n_triplets": len(triplets),
        "top_positions": args.top_positions,
        "n_perm": args.n_perm,
        "q_threshold": args.q_threshold,
        "category_counts": result["category_counts"],
        "n_categorized": result["n_categorized"],
        "outcome": result["outcome"],
        "outputs": {
            "triplet_characterization": str(args.out_dir / "triplet_characterization.tsv"),
            "top_firing_positions": str(args.out_dir / "top_firing_positions.tsv"),
            "summary_md": str(args.out_dir / "summary.md"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# M-1 Triplet Characterization",
        "",
        f"- Cohort size: {len(records)}",
        f"- Triplets: {len(triplets)}",
        f"- Categorized triplets: {result['n_categorized']} / {len(triplets)}",
        f"- Outcome: {result['outcome']}",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]
    for cat, count in sorted(result["category_counts"].items()):
        lines.append(f"| {cat} | {count} |")
    lines += [
        "",
        "| Triplet | Assigned category | k-mer q | positional q | BPE q | attention q | high-norm q |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["per_triplet"]:
        lines.append(
            f"| {row['triplet_id']} | {row['assigned_category']} | "
            f"{row['k_mer_q']:.4g} | {row['positional_q']:.4g} | {row['bpe_boundary_q']:.4g} | "
            f"{row['attention_sink_q']:.4g} | {row['high_norm_q']:.4g} |"
        )
    lines += [
        "",
        "## Evidence Boundary",
        "",
        "- k-mer significance uses permutation MI against random top-position assignments.",
        "- Positional significance uses a one-sample KS statistic against Uniform(0,1).",
        "- Attention-sink and high-norm tests are one-sided positive Pearson-correlation permutation tests.",
        "- BPE-boundary is ProtGPT2-only and tests top-firing enrichment at token start/end residues.",
        "",
    ]
    (args.out_dir / "summary.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
