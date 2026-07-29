#!/usr/bin/env python3
"""Direct attention-head ablation for R2 N-terminal sink-associated triplets.

The previous causal test ablated CLT sparse features through the MLP-output
patching path. This script tests the readout hypothesis instead: T011/T018/T023
may identify attention heads that dump probability mass onto the N-terminal
edge. We first pair triplets with candidate sink heads, then ablate the paired
heads directly at the attention output projection input.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

R2 = Path(__file__).resolve().parents[1]
if str(R2) not in sys.path:
    sys.path.insert(0, str(R2))

from src.analysis.circuit_discovery import load_trained_clt
from src.models.model_loader import load_model


AA = set("ACDEFGHIKLMNPQRSTVWY")
TARGET_TRIPLETS = ["T011", "T018", "T023"]
SPECIFICITY_TRIPLETS = ["T025"]


def clean_sequence(seq: str) -> str:
    return "".join(c for c in (seq or "").upper() if c in AA)


def parse_model_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"bad --model-spec {spec!r}; expected model=checkpoint_dir")
    name, ckpt = spec.split("=", 1)
    return name.strip(), ckpt.strip()


def read_triplets(path: Path) -> list[dict]:
    with path.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    out = []
    for rank, row in enumerate(rows, start=1):
        rec = {
            "triplet_id": f"T{rank:03d}",
            "rank": rank,
            "anchor_layer": int(row["anchor_layer"]),
            "min_abs_corr": float(row["min_abs_corr"]),
            "mean_abs_corr": float(row["mean_abs_corr"]),
            "features": {},
        }
        for key, value in row.items():
            if key.endswith("_feature"):
                model = key[: -len("_feature")]
                rec["features"][model] = {
                    "layer": int(row[f"{model}_layer"]),
                    "feature": int(value),
                }
        out.append(rec)
    return out


def read_cohort(path: Path, source: str, max_sequences: int, max_length: int) -> list[dict]:
    data = json.loads(path.read_text())
    rows = data.get("records", data if isinstance(data, list) else [])
    out = []
    for i, item in enumerate(rows):
        if not isinstance(item, dict):
            continue
        src = str(item.get("source", "unknown"))
        if source != "all" and src != source:
            continue
        seq = clean_sequence(item.get("sequence", ""))[:max_length]
        if not seq:
            continue
        out.append({
            "id": str(item.get("id") or f"seq_{i:06d}"),
            "source": src,
            "sequence": seq,
            "starts_m": int(seq.startswith("M")),
        })
        if len(out) >= max_sequences:
            break
    return out


def token_residue_spans(tokenizer, input_ids: torch.Tensor, sequence: str) -> list[list[int]]:
    ids = input_ids.detach().cpu().view(-1).tolist()
    spans: list[list[int]] = []
    cursor = 0
    for tok_id in ids:
        piece = tokenizer.decode([tok_id], skip_special_tokens=True)
        letters = clean_sequence(piece)
        if not letters:
            spans.append([])
            continue
        found = sequence.find(letters, cursor)
        if found < 0 and len(letters) == 1:
            for j in range(cursor, min(len(sequence), cursor + 8)):
                if sequence[j] == letters:
                    found = j
                    break
        if found < 0:
            spans.append([])
            continue
        span = list(range(found, min(len(sequence), found + len(letters))))
        spans.append(span)
        cursor = max(cursor, found + len(letters))
    return spans


def first2_token_mask(spans: list[list[int]]) -> np.ndarray:
    return np.asarray([bool(span and any(pos <= 1 for pos in span)) for span in spans], dtype=bool)


def shifted_nll(logits: torch.Tensor, input_ids: torch.Tensor) -> np.ndarray:
    ids = input_ids[0] if input_ids.dim() == 2 else input_ids
    logits = logits[0]
    out = torch.full((ids.shape[0],), float("nan"), device=logits.device, dtype=torch.float32)
    if ids.shape[0] >= 2:
        out[1:] = F.cross_entropy(logits[:-1].float(), ids[1:].long(), reduction="none")
    return out.detach().cpu().numpy()


def mean_or_nan(vals: list[float]) -> float:
    vals = [float(x) for x in vals if math.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


def summarize_delta_nll(delta: np.ndarray, spans: list[list[int]]) -> dict:
    bins = defaultdict(list)
    for tok_idx, span in enumerate(spans):
        if tok_idx >= len(delta) or not span or not math.isfinite(float(delta[tok_idx])):
            continue
        pos0 = min(span)
        val = float(delta[tok_idx])
        bins["all"].append(val)
        if pos0 <= 1:
            bins["first2"].append(val)
        if 1 <= pos0 <= 9:
            bins["pos2_10"].append(val)
        if 2 <= pos0 <= 9:
            bins["pos3_10"].append(val)
        if pos0 >= 10:
            bins["pos11_plus"].append(val)
    out = {}
    for key in ["all", "first2", "pos2_10", "pos3_10", "pos11_plus"]:
        out[f"delta_nll_{key}"] = mean_or_nan(bins[key])
        out[f"n_tokens_{key}"] = len(bins[key])
    out["specificity_pos2_10_minus_11plus"] = (
        out["delta_nll_pos2_10"] - out["delta_nll_pos11_plus"]
        if math.isfinite(out["delta_nll_pos2_10"]) and math.isfinite(out["delta_nll_pos11_plus"])
        else float("nan")
    )
    return out


def head_config(pm) -> tuple[int, int]:
    n_heads = getattr(pm.config, "n_head", None)
    if n_heads is None:
        n_heads = getattr(pm.config, "num_attention_heads", None)
    if n_heads is None:
        raise ValueError(f"cannot infer n_heads for {pm.model_name}")
    head_dim = int(pm.d_model) // int(n_heads)
    return int(n_heads), head_dim


def attn_projection_module(pm, layer: int):
    attn = getattr(pm._get_block(layer), "attn", None)
    if attn is None:
        raise ValueError(f"no .attn module for {pm.model_name} layer {layer}")
    for name in ("c_proj", "out_proj", "dense"):
        if hasattr(attn, name):
            return getattr(attn, name)
    raise ValueError(f"cannot find attention output projection for {pm.model_name} layer {layer}")


def make_head_ablation_hook(head: int, n_heads: int, multiplier: float):
    def hook(_module, inputs):
        x = inputs[0]
        if not torch.is_tensor(x) or x.shape[-1] % n_heads != 0:
            return inputs
        y = x.clone()
        head_dim = x.shape[-1] // n_heads
        start = int(head) * head_dim
        end = start + head_dim
        y[..., start:end] = y[..., start:end] * multiplier
        return (y,) + tuple(inputs[1:])
    return hook


def forward_with_head(pm, input_ids: torch.Tensor, condition: dict | None, multiplier: float):
    hooks = []
    if condition and condition.get("condition_kind") != "sham":
        module = attn_projection_module(pm, int(condition["head_layer"]))
        hooks.append(module.register_forward_pre_hook(make_head_ablation_hook(
            int(condition["head"]), int(condition["n_heads"]), multiplier
        )))
    try:
        with torch.no_grad():
            return pm.model(input_ids, output_attentions=True, use_cache=False)
    finally:
        for h in hooks:
            h.remove()


def attention_first2_by_head(attentions, spans: list[list[int]]) -> dict[tuple[int, int], float]:
    first2 = first2_token_mask(spans)
    out = {}
    if attentions is None or not first2.any():
        return out
    for layer_idx, attn in enumerate(attentions):
        if attn is None:
            continue
        # batch x heads x query x key
        x = attn[0].detach().float().cpu().numpy()
        n_keys = min(x.shape[-1], len(first2))
        mask = first2[:n_keys]
        if not mask.any():
            continue
        # Mean over query positions. This is the fraction of each head's
        # attention directed to N-terminal key tokens.
        mass = x[:, :, :n_keys][:, :, mask].sum(axis=-1).mean(axis=-1)
        for head_idx, val in enumerate(mass):
            out[(layer_idx, head_idx)] = float(val)
    return out


def attention_distribution(attentions, spans: list[list[int]], start_layer: int) -> np.ndarray | None:
    first = []
    if attentions is None:
        return None
    for layer_idx in range(max(0, start_layer), len(attentions)):
        attn = attentions[layer_idx]
        if attn is None:
            continue
        recv = attn[0].detach().float().mean(dim=0).mean(dim=0).cpu().numpy()
        n = min(len(recv), len(spans))
        mapped = np.asarray([bool(s) for s in spans[:n]], dtype=bool)
        if mapped.any():
            vals = recv[:n].astype(np.float64)
            vals = vals * mapped
            vals = vals / max(float(vals.sum()), 1e-12)
            first.append(vals)
    if not first:
        return None
    max_len = max(len(x) for x in first)
    arr = np.zeros((len(first), max_len), dtype=np.float64)
    for i, x in enumerate(first):
        arr[i, : len(x)] = x
    return arr.mean(axis=0)


def attention_delta_metrics(intact_attn, ablated_attn, spans: list[list[int]], start_layer: int) -> dict:
    a = attention_distribution(intact_attn, spans, start_layer)
    b = attention_distribution(ablated_attn, spans, start_layer)
    if a is None or b is None:
        return {
            "attention_first2_delta": float("nan"),
            "attention_cosine_distance": float("nan"),
        }
    n = min(len(a), len(b), len(spans))
    a = a[:n]
    b = b[:n]
    first2 = first2_token_mask(spans[:n])
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12)
    return {
        "attention_first2_delta": float(b[first2].sum() - a[first2].sum()) if first2.any() else float("nan"),
        "attention_cosine_distance": 1.0 - float(np.dot(a, b) / denom),
    }


def feature_first2_activation(pm, clt, input_ids: torch.Tensor, sequence: str, layer: int, feature: int,
                              condition: dict | None, multiplier: float) -> float:
    hooks = []
    if condition and condition.get("condition_kind") != "sham":
        module = attn_projection_module(pm, int(condition["head_layer"]))
        hooks.append(module.register_forward_pre_hook(make_head_ablation_hook(
            int(condition["head"]), int(condition["n_heads"]), multiplier
        )))
    try:
        spans = token_residue_spans(pm.tokenizer, input_ids, sequence)
        cache = pm.get_activations(input_ids)
        feats = clt.encode([x.float() for x in cache.resid_pre])
        vals = feats[int(layer)][0, :, int(feature)].detach().float().cpu().numpy()
        mask = first2_token_mask(spans)
        n = min(len(vals), len(mask))
        if n == 0 or not mask[:n].any():
            return float("nan")
        return float(np.nanmax(vals[:n][mask[:n]]))
    finally:
        for h in hooks:
            h.remove()


def corr(x: list[float], y: list[float]) -> float:
    pairs = [(a, b) for a, b in zip(x, y) if math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 3:
        return float("nan")
    a = np.asarray([p[0] for p in pairs], dtype=np.float64)
    b = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if a.std() <= 1e-12 or b.std() <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def identify_heads(pm, clt, records: list[dict], triplets: list[dict], model_name: str,
                   args) -> tuple[list[dict], dict[str, dict]]:
    n_heads, _head_dim = head_config(pm)
    wanted = TARGET_TRIPLETS + SPECIFICITY_TRIPLETS
    specs = {
        t["triplet_id"]: t["features"][model_name]
        for t in triplets
        if t["triplet_id"] in wanted and model_name in t["features"]
    }
    head_mass_by_seq: list[dict[tuple[int, int], float]] = []
    feat_by_tid: dict[str, list[float]] = {tid: [] for tid in specs}
    ids = []
    t0 = time.time()
    for i, rec in enumerate(records[: args.head_discovery_sequences]):
        seq = rec["sequence"]
        input_ids = pm.tokenize(seq)
        spans = token_residue_spans(pm.tokenizer, input_ids, seq)
        out = forward_with_head(pm, input_ids, None, 1.0)
        head_mass_by_seq.append(attention_first2_by_head(out.attentions, spans))
        ids.append(rec["id"])
        cache = pm.get_activations(input_ids)
        feats = clt.encode([x.float() for x in cache.resid_pre])
        mask = first2_token_mask(spans)
        for tid, spec in specs.items():
            vals = feats[int(spec["layer"])][0, :, int(spec["feature"])].detach().float().cpu().numpy()
            n = min(len(vals), len(mask))
            feat_by_tid[tid].append(float(np.nanmax(vals[:n][mask[:n]])) if n and mask[:n].any() else float("nan"))
        if (i + 1) % args.report_every == 0 or i == min(len(records), args.head_discovery_sequences) - 1:
            print(f"  head discovery {model_name}: {i+1}/{min(len(records), args.head_discovery_sequences)} ({time.time()-t0:.1f}s)", flush=True)

    all_heads = sorted({k for d in head_mass_by_seq for k in d})
    rows = []
    best_by_tid = {}
    for tid in specs:
        best = None
        for layer, head in all_heads:
            masses = [d.get((layer, head), float("nan")) for d in head_mass_by_seq]
            mean_mass = mean_or_nan(masses)
            r = corr(masses, feat_by_tid[tid])
            rec = {
                "model": model_name,
                "triplet_id": tid,
                "head_layer": layer,
                "head": head,
                "n_heads": n_heads,
                "mean_first2_mass": mean_mass,
                "corr_with_feature_first2": r,
                "selection_score": (0.0 if not math.isfinite(r) else max(r, 0.0)) + mean_mass,
            }
            rows.append(rec)
            eligible = mean_mass >= args.min_sink_mass
            key = (eligible, rec["selection_score"], mean_mass)
            if best is None or key > best[0]:
                best = (key, rec)
        if best is not None:
            best_by_tid[tid] = best[1]
    return rows, best_by_tid


def make_conditions(best_by_tid: dict[str, dict], n_heads: int, args) -> list[dict]:
    rng = random.Random(args.seed)
    conditions = []
    for tid in TARGET_TRIPLETS:
        if tid not in best_by_tid:
            continue
        rec = best_by_tid[tid]
        conditions.append({
            "condition": tid,
            "condition_kind": "target",
            "triplet_id": tid,
            **rec,
        })
        layer = int(rec["head_layer"])
        used = {int(x["head"]) for x in best_by_tid.values() if int(x["head_layer"]) == layer}
        for j in range(args.n_random_heads):
            head = rng.randrange(n_heads)
            tries = 0
            while head in used and tries < 1000:
                head = rng.randrange(n_heads)
                tries += 1
            used.add(head)
            conditions.append({
                "condition": f"RND_{tid}_{j}",
                "condition_kind": "random_head",
                "triplet_id": "",
                "model": rec["model"],
                "head_layer": layer,
                "head": head,
                "n_heads": n_heads,
                "mean_first2_mass": float("nan"),
                "corr_with_feature_first2": float("nan"),
                "selection_score": float("nan"),
                "control_for": tid,
            })
    for tid in SPECIFICITY_TRIPLETS:
        if tid in best_by_tid:
            conditions.append({
                "condition": tid,
                "condition_kind": "specificity",
                "triplet_id": tid,
                **best_by_tid[tid],
            })
    conditions.append({
        "condition": "SHAM",
        "condition_kind": "sham",
        "triplet_id": "",
        "model": next(iter(best_by_tid.values()))["model"] if best_by_tid else "",
        "head_layer": 0,
        "head": 0,
        "n_heads": n_heads,
        "mean_first2_mass": float("nan"),
        "corr_with_feature_first2": float("nan"),
        "selection_score": float("nan"),
        "control_for": "",
    })
    return conditions


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_row(path: Path, row: dict, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        clean = {}
        for k in fields:
            v = row.get(k, "")
            if isinstance(v, float):
                clean[k] = f"{v:.8g}" if math.isfinite(v) else "nan"
            else:
                clean[k] = v
        writer.writerow(clean)


def existing_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    keys = set()
    with path.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            keys.add((row["model"], row["condition"], row["sequence_id"]))
    return keys


def run(args) -> None:
    os.environ.setdefault("R2_MODEL_BASE_DIR", "/gpfs/jiaotongdamoxing/zhk_zip/models")
    records = read_cohort(args.cohort, args.source, args.max_sequences, args.max_length)
    triplets = read_triplets(args.triplets)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metric_fields = [
        "model", "condition", "condition_kind", "control_for", "triplet_id",
        "head_layer", "head", "n_heads", "paired_mean_first2_mass",
        "paired_corr_with_feature", "sequence_id", "source", "starts_m",
        "seq_len", "n_tokens", "delta_nll_all", "delta_nll_first2",
        "delta_nll_pos2_10", "delta_nll_pos3_10", "delta_nll_pos11_plus",
        "specificity_pos2_10_minus_11plus", "n_tokens_all", "n_tokens_first2",
        "n_tokens_pos2_10", "n_tokens_pos3_10", "n_tokens_pos11_plus",
        "attention_first2_delta", "attention_cosine_distance",
        "feature_first2_intact", "feature_first2_ablated", "feature_first2_drop_fraction",
    ]
    metrics_path = args.out_dir / "per_sequence_metrics.tsv"
    done = set() if args.force else existing_keys(metrics_path)

    for model_name, ckpt in [parse_model_spec(s) for s in args.model_spec]:
        print(f"\n=== {model_name} ===", flush=True)
        pm = load_model(model_name, device=args.device)
        clt = load_trained_clt(ckpt, device=args.device)
        n_heads, _ = head_config(pm)
        head_rows, best_by_tid = identify_heads(pm, clt, records, triplets, model_name, args)
        write_tsv(
            args.out_dir / f"{model_name}_head_discovery.tsv",
            head_rows,
            ["model", "triplet_id", "head_layer", "head", "n_heads", "mean_first2_mass",
             "corr_with_feature_first2", "selection_score"],
        )
        conditions = make_conditions(best_by_tid, n_heads, args)
        (args.out_dir / f"{model_name}_conditions.json").write_text(json.dumps(conditions, indent=2) + "\n")
        specs = {
            t["triplet_id"]: t["features"][model_name]
            for t in triplets
            if model_name in t["features"]
        }
        t0 = time.time()
        for i, rec in enumerate(records):
            pending = [c for c in conditions if (model_name, c["condition"], rec["id"]) not in done]
            if not pending:
                continue
            seq = rec["sequence"]
            input_ids = pm.tokenize(seq)
            spans = token_residue_spans(pm.tokenizer, input_ids, seq)
            intact = forward_with_head(pm, input_ids, None, 1.0)
            intact_nll = shifted_nll(intact.logits, input_ids)
            for cond in pending:
                ablated = forward_with_head(pm, input_ids, cond, args.multiplier)
                ablated_nll = shifted_nll(ablated.logits, input_ids)
                delta = ablated_nll - intact_nll
                row = {
                    "model": model_name,
                    "condition": cond["condition"],
                    "condition_kind": cond["condition_kind"],
                    "control_for": cond.get("control_for", ""),
                    "triplet_id": cond.get("triplet_id", ""),
                    "head_layer": cond["head_layer"],
                    "head": cond["head"],
                    "n_heads": cond["n_heads"],
                    "paired_mean_first2_mass": cond.get("mean_first2_mass", float("nan")),
                    "paired_corr_with_feature": cond.get("corr_with_feature_first2", float("nan")),
                    "sequence_id": rec["id"],
                    "source": rec["source"],
                    "starts_m": rec["starts_m"],
                    "seq_len": len(seq),
                    "n_tokens": int(input_ids.shape[-1]),
                }
                row.update(summarize_delta_nll(delta, spans))
                row.update(attention_delta_metrics(intact.attentions, ablated.attentions, spans, int(cond["head_layer"])))
                if cond.get("triplet_id") in specs:
                    spec = specs[cond["triplet_id"]]
                    v0 = feature_first2_activation(pm, clt, input_ids, seq, int(spec["layer"]), int(spec["feature"]), None, 1.0)
                    v1 = feature_first2_activation(pm, clt, input_ids, seq, int(spec["layer"]), int(spec["feature"]), cond, args.multiplier)
                    row["feature_first2_intact"] = v0
                    row["feature_first2_ablated"] = v1
                    row["feature_first2_drop_fraction"] = (v0 - v1) / max(abs(v0), 1e-8) if math.isfinite(v0) and math.isfinite(v1) else float("nan")
                else:
                    row["feature_first2_intact"] = float("nan")
                    row["feature_first2_ablated"] = float("nan")
                    row["feature_first2_drop_fraction"] = float("nan")
                append_row(metrics_path, row, metric_fields)
                done.add((model_name, cond["condition"], rec["id"]))
            if (i + 1) % args.report_every == 0 or i == len(records) - 1:
                print(f"  ablation {model_name}: {i+1}/{len(records)} ({time.time()-t0:.1f}s)", flush=True)
        del pm
        del clt
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    summarize(args)


def numeric(row: dict, key: str) -> float:
    try:
        x = float(row.get(key, "nan"))
    except Exception:
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def bootstrap_ci(vals: list[float], seed: int, n_boot: int = 1000) -> tuple[float, float]:
    vals = np.asarray([v for v in vals if math.isfinite(v)], dtype=np.float64)
    if vals.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(vals, size=vals.size, replace=True).mean()) for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(args) -> None:
    path = args.out_dir / "per_sequence_metrics.tsv"
    with path.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["condition"], r["condition_kind"], r.get("control_for", ""))].append(r)
    summary_rows = []
    for (model, condition, kind, control_for), items in sorted(groups.items()):
        rec = {
            "model": model,
            "condition": condition,
            "condition_kind": kind,
            "control_for": control_for,
            "n_sequences": len(items),
            "head_layer": items[0].get("head_layer", ""),
            "head": items[0].get("head", ""),
            "paired_mean_first2_mass": numeric(items[0], "paired_mean_first2_mass"),
            "paired_corr_with_feature": numeric(items[0], "paired_corr_with_feature"),
        }
        for key in [
            "delta_nll_pos2_10", "delta_nll_pos11_plus",
            "specificity_pos2_10_minus_11plus", "attention_first2_delta",
            "attention_cosine_distance", "feature_first2_drop_fraction",
        ]:
            vals = [numeric(r, key) for r in items]
            vals = [v for v in vals if math.isfinite(v)]
            rec[f"mean_{key}"] = float(np.mean(vals)) if vals else float("nan")
            if key in {"delta_nll_pos2_10", "specificity_pos2_10_minus_11plus", "feature_first2_drop_fraction"}:
                lo, hi = bootstrap_ci(vals, args.seed + len(summary_rows))
                rec[f"{key}_ci_low"] = lo
                rec[f"{key}_ci_high"] = hi
        summary_rows.append(rec)
    fields = list(summary_rows[0].keys()) if summary_rows else ["model", "condition"]
    write_tsv(args.out_dir / "condition_summary.tsv", summary_rows, fields)

    random_by_model = defaultdict(list)
    for r in summary_rows:
        if r["condition_kind"] == "random_head":
            random_by_model[r["model"]].append(r)

    gate_rows = []
    for r in summary_rows:
        if r["condition_kind"] != "target":
            continue
        rand_vals = [
            x["mean_delta_nll_pos2_10"]
            for x in random_by_model.get(r["model"], [])
            if math.isfinite(x["mean_delta_nll_pos2_10"])
        ]
        rand_mean = float(np.mean(rand_vals)) if rand_vals else 0.0
        nll_hit = (
            math.isfinite(r["mean_delta_nll_pos2_10"])
            and r["mean_delta_nll_pos2_10"] >= args.gate_nll
            and r["mean_delta_nll_pos2_10"] > rand_mean + args.gate_random_margin
        )
        feature_hit = (
            math.isfinite(r["mean_feature_first2_drop_fraction"])
            and r["mean_feature_first2_drop_fraction"] >= args.gate_feature_drop
        )
        hit = bool(nll_hit and feature_hit)
        gate_rows.append({**r, "random_head_delta_nll_mean": rand_mean, "nll_hit": nll_hit, "feature_hit": feature_hit, "gate_hit": hit})

    hit_models = {r["model"] for r in gate_rows if r["gate_hit"]}
    feature_hit_models = {r["model"] for r in gate_rows if r["feature_hit"]}
    if len(hit_models) >= 2:
        outcome = "PASS"
    elif len(hit_models) == 1 or len(feature_hit_models) >= 1:
        outcome = "PARTIAL"
    else:
        outcome = "FAIL"

    payload = {
        "task": "R2 E-1 direct attention-head sink ablation",
        "outcome": outcome,
        "n_rows": len(rows),
        "gate": {
            "pass_rule": "target heads pass NLL and feature-drop gates in at least two models",
            "gate_nll": args.gate_nll,
            "gate_feature_drop": args.gate_feature_drop,
            "gate_random_margin": args.gate_random_margin,
        },
        "target_gate_rows": gate_rows,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Direct Attention-Head Sink Ablation",
        "",
        f"- Per-sequence rows: {len(rows)}",
        f"- Outcome: {outcome}",
        "",
        "| Model | Triplet | Head | mean first2 mass | corr(feature) | dNLL pos2-10 | random dNLL | feature drop | attention dFirst2 | gate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in gate_rows:
        lines.append(
            f"| {r['model']} | {r['condition']} | L{r['head_layer']}H{r['head']} | "
            f"{r['paired_mean_first2_mass']:.4f} | {r['paired_corr_with_feature']:.4f} | "
            f"{r['mean_delta_nll_pos2_10']:.4f} | {r['random_head_delta_nll_mean']:.4f} | "
            f"{r['mean_feature_first2_drop_fraction']:.4f} | {r['mean_attention_first2_delta']:.4f} | {r['gate_hit']} |"
        )
    lines += [
        "",
        "Notes:",
        "- Head ablation zeros a contiguous attention-head slice before the attention output projection.",
        "- The selected head is the highest-scoring N-terminal sink candidate paired with each triplet's first-two-residue activation.",
        "- PASS requires both an NLL increase and a post-ablation feature-activation drop, beyond random-head controls.",
        "",
    ]
    (args.out_dir / "summary.md").write_text("\n".join(lines))
    print(f"Wrote {args.out_dir / 'summary.md'}")
    print(f"Outcome: {outcome}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["run", "summarize"], default="run")
    ap.add_argument("--triplets", type=Path, default=R2 / "results/circuit_analysis/universal_atlas_balanced200_wide_triplets_20260512.tsv")
    ap.add_argument("--cohort", type=Path, default=R2 / "results/circuit_analysis/triplet_characterization_20260515_nperm2000/cohort.json")
    ap.add_argument("--source", default="swissprot_n1")
    ap.add_argument("--max-sequences", type=int, default=200)
    ap.add_argument("--head-discovery-sequences", type=int, default=100)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--min-sink-mass", type=float, default=0.20)
    ap.add_argument("--n-random-heads", type=int, default=1)
    ap.add_argument("--multiplier", type=float, default=0.0)
    ap.add_argument("--gate-nll", type=float, default=0.5)
    ap.add_argument("--gate-feature-drop", type=float, default=0.5)
    ap.add_argument("--gate-random-margin", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20260518)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--report-every", type=int, default=10)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=R2 / "results/circuit_analysis/attention_head_sink_ablation_20260518")
    ap.add_argument("--model-spec", action="append", default=[
        "protgpt2=/oss-pvc/zhk_zip/outputs/research2/clt_weights/protgpt2_v2/step_200000",
        "zymctrl=/oss-pvc/zhk_zip/outputs/research2/clt_weights/zymctrl_v2/step_200000",
        "progen2-medium=/gpfs/jiaotongdamoxing/zhk_zip/biocc/paper_r2_nature_mi/results/final_checkpoints/r2_clt_progen2_medium_rerun_20260403/clt_weights/progen2-medium/step_100000",
    ])
    args = ap.parse_args()
    if args.mode == "run":
        run(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
