#!/usr/bin/env python3
"""Pilot sparse model for attention-output features in ZymCTRL.

This is a deliberately small rescue/redesign probe for R2.  The previous CLT
experiments decomposed MLP outputs; the final failed mechanism was an
attention-sink phenomenon.  This script instead decomposes the tensor that
enters each selected attention output projection (`attn.c_proj` input in
GPT-2-like models), i.e. concatenated per-head attention outputs.

The pilot answers three questions:
  1. Can a small sparse dictionary reconstruct attention-output vectors?
  2. Do any learned features read out N-terminal attention-sink behavior?
  3. Does ablating those feature contributions before `c_proj` measurably
     change N-terminal teacher-forced NLL beyond random-feature controls?

It is not intended as a final architecture; it is a cheap decision experiment.
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
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

R2 = Path(__file__).resolve().parents[1]
REPO = R2.parent
if str(R2) not in sys.path:
    sys.path.insert(0, str(R2))

from src.models.model_loader import load_model


AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_sequence(seq: str) -> str:
    return "".join(c for c in (seq or "").upper() if c in AA)


def extract_protein_sequence(text: str) -> str:
    """Extract the amino-acid sequence from ZymCTRL FASTA text.

    ZymCTRL training records often look like:
      EC<sep><start>SEQUENCE<end>

    A plain AA filter would incorrectly keep letters from "sep" and "start".
    """
    s = (text or "").strip()
    if "<start>" in s:
        s = s.split("<start>", 1)[1]
    if "<end>" in s:
        s = s.split("<end>", 1)[0]
    return clean_sequence(s)


def read_fasta_records(path: Path, max_sequences: int, max_length: int, seed: int) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    cur_id = None
    cur: list[str] = []
    with path.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if cur_id is not None:
                    seq = extract_protein_sequence("".join(cur))[:max_length]
                    if seq:
                        records.append({"id": cur_id, "sequence": seq})
                cur_id = line[1:].split()[0]
                cur = []
            else:
                cur.append(line)
        if cur_id is not None:
            seq = extract_protein_sequence("".join(cur))[:max_length]
            if seq:
                records.append({"id": cur_id, "sequence": seq})
    rng = random.Random(seed)
    rng.shuffle(records)
    if max_sequences > 0:
        records = records[:max_sequences]
    return records


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean = {}
            for key in fields:
                val = row.get(key, "")
                if isinstance(val, float):
                    clean[key] = f"{val:.8g}" if math.isfinite(val) else "nan"
                elif isinstance(val, (list, dict)):
                    clean[key] = json.dumps(val, sort_keys=True)
                else:
                    clean[key] = val
            writer.writerow(clean)


def append_tsv(path: Path, row: dict[str, Any], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        if not exists:
            writer.writeheader()
        clean = {}
        for key in fields:
            val = row.get(key, "")
            if isinstance(val, float):
                clean[key] = f"{val:.8g}" if math.isfinite(val) else "nan"
            elif isinstance(val, (list, dict)):
                clean[key] = json.dumps(val, sort_keys=True)
            else:
                clean[key] = val
        writer.writerow(clean)


def attn_projection_module(pm, layer: int):
    attn = getattr(pm._get_block(layer), "attn", None)
    if attn is None:
        raise ValueError(f"no .attn module for {pm.model_name} layer {layer}")
    for name in ("c_proj", "out_proj", "dense"):
        if hasattr(attn, name):
            return getattr(attn, name)
    raise ValueError(f"cannot find attention output projection for {pm.model_name} layer {layer}")


def capture_attention_outputs(pm, input_ids: torch.Tensor, layers: list[int], output_attentions: bool = False):
    captured: dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(layer: int):
        def hook(_module, inputs):
            captured[layer] = inputs[0].detach()
            return inputs
        return hook

    for layer in layers:
        hooks.append(attn_projection_module(pm, layer).register_forward_pre_hook(make_hook(layer)))
    try:
        with torch.no_grad():
            out = pm.model(input_ids, output_attentions=output_attentions, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    return captured, out


class AttentionOutputSAE(nn.Module):
    def __init__(self, layers: list[int], d_model: int, d_sae: int, k: int):
        super().__init__()
        self.layers = list(layers)
        self.layer_to_idx = {layer: i for i, layer in enumerate(self.layers)}
        self.d_model = int(d_model)
        self.d_sae = int(d_sae)
        self.k = int(k)
        n = len(self.layers)
        self.W_enc = nn.Parameter(torch.empty(n, d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(n, d_sae))
        self.W_dec = nn.Parameter(torch.empty(n, d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(n, d_model))
        self.register_buffer("scale", torch.ones(n))
        self.register_buffer("last_fired", torch.zeros(n, d_sae, dtype=torch.long))
        self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for i in range(len(self.layers)):
            nn.init.kaiming_uniform_(self.W_enc[i])
            self.W_enc.data[i] *= 0.05
            nn.init.kaiming_uniform_(self.W_dec[i])
            self.W_dec.data[i] *= 0.05

    def encode_idx(self, idx: int, x_norm: torch.Tensor) -> torch.Tensor:
        pre = torch.einsum("bsd,fd->bsf", x_norm, self.W_enc[idx]) + self.b_enc[idx]
        pre = F.relu(pre)
        k = min(self.k, pre.shape[-1])
        vals, inds = pre.topk(k, dim=-1)
        sparse = torch.zeros_like(pre)
        sparse.scatter_(-1, inds, vals)
        return sparse

    def decode_idx(self, idx: int, feats: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bsf,fd->bsd", feats, self.W_dec[idx]) + self.b_dec[idx]

    def forward_layer(self, layer: int, x: torch.Tensor) -> dict[str, torch.Tensor]:
        idx = self.layer_to_idx[layer]
        x_norm = x.float() / self.scale[idx].clamp_min(1e-6)
        feats = self.encode_idx(idx, x_norm)
        recon = self.decode_idx(idx, feats)
        fired = (feats > 0).any(dim=0).any(dim=0)
        self.last_fired[idx, fired] = self.global_step
        return {"x_norm": x_norm, "features": feats, "recon": recon}

    def delta_for_ablation(self, layer: int, x: torch.Tensor, feature_ids: list[int], multiplier: float = 0.0) -> torch.Tensor:
        idx = self.layer_to_idx[layer]
        x_norm = x.float() / self.scale[idx].clamp_min(1e-6)
        feats = self.encode_idx(idx, x_norm)
        recon_full = self.decode_idx(idx, feats)
        feats_mod = feats.clone()
        for fid in feature_ids:
            if 0 <= int(fid) < self.d_sae:
                feats_mod[..., int(fid)] *= multiplier
        recon_mod = self.decode_idx(idx, feats_mod)
        return (recon_mod - recon_full) * self.scale[idx].clamp_min(1e-6)

    def reinit_dead(self, threshold: int) -> int:
        step = int(self.global_step.item())
        n_reset = 0
        with torch.no_grad():
            dead = step - self.last_fired > threshold
            for i in range(len(self.layers)):
                ids = torch.nonzero(dead[i], as_tuple=False).flatten()
                if ids.numel() == 0:
                    continue
                n_reset += int(ids.numel())
                nn.init.kaiming_uniform_(self.W_enc[i, ids])
                self.W_enc[i, ids] *= 0.05
                nn.init.kaiming_uniform_(self.W_dec[i, ids])
                self.W_dec[i, ids] *= 0.05
                self.b_enc[i, ids] = 0
                self.last_fired[i, ids] = step
        return n_reset


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


def shifted_nll(logits: torch.Tensor, input_ids: torch.Tensor) -> np.ndarray:
    ids = input_ids[0] if input_ids.dim() == 2 else input_ids
    logits = logits[0]
    out = torch.full((ids.shape[0],), float("nan"), device=logits.device, dtype=torch.float32)
    if ids.shape[0] >= 2:
        out[1:] = F.cross_entropy(logits[:-1].float(), ids[1:].long(), reduction="none")
    return out.detach().cpu().numpy()


def summarize_delta_nll(delta: np.ndarray, spans: list[list[int]]) -> dict[str, float]:
    bins: dict[str, list[float]] = defaultdict(list)
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
        if pos0 >= 10:
            bins["pos11_plus"].append(val)
    out = {}
    for key in ["all", "first2", "pos2_10", "pos11_plus"]:
        vals = bins[key]
        out[f"delta_nll_{key}"] = float(np.mean(vals)) if vals else float("nan")
    return out


def attention_received_first2(attentions, spans: list[list[int]], layer: int) -> np.ndarray:
    if attentions is None or layer >= len(attentions) or attentions[layer] is None:
        return np.full(len(spans), np.nan, dtype=np.float64)
    attn = attentions[layer][0].detach().float().cpu().numpy()  # heads x query x key
    recv = attn.mean(axis=0).mean(axis=0)
    n = min(len(recv), len(spans))
    out = np.full(len(spans), np.nan, dtype=np.float64)
    out[:n] = recv[:n]
    return out


def calibrate_scales(pm, sae: AttentionOutputSAE, records: list[dict[str, str]], n_sequences: int, device: str) -> list[float]:
    sums = {layer: 0.0 for layer in sae.layers}
    counts = {layer: 0 for layer in sae.layers}
    for rec in records[:n_sequences]:
        ids = pm.tokenize(rec["sequence"]).to(device)
        caps, _ = capture_attention_outputs(pm, ids, sae.layers, output_attentions=False)
        for layer, x in caps.items():
            sums[layer] += float((x.float() ** 2).sum().item())
            counts[layer] += int(x.numel())
    scales = []
    with torch.no_grad():
        for i, layer in enumerate(sae.layers):
            rms = math.sqrt(sums[layer] / max(counts[layer], 1))
            if not math.isfinite(rms) or rms <= 0:
                rms = 1.0
            sae.scale[i] = float(rms)
            scales.append(float(rms))
    return scales


def train(args, pm, records: list[dict[str, str]], sae: AttentionOutputSAE) -> list[dict[str, Any]]:
    opt = torch.optim.AdamW(sae.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    rng = random.Random(args.seed + 101)
    metrics: list[dict[str, Any]] = []
    fields = ["step", "loss", "fvu_mean", "alive_mean", "n_reset", "seconds"]
    t0 = time.time()
    metric_path = args.out_dir / "training_metrics.tsv"
    if metric_path.exists() and args.force:
        metric_path.unlink()
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        losses = []
        fvus = []
        for _ in range(args.batch_size):
            rec = rng.choice(records)
            ids = pm.tokenize(rec["sequence"]).to(args.device)
            caps, _ = capture_attention_outputs(pm, ids, sae.layers, output_attentions=False)
            for layer in sae.layers:
                out = sae.forward_layer(layer, caps[layer])
                diff = out["recon"] - out["x_norm"]
                mse = (diff ** 2).mean()
                var = out["x_norm"].var().clamp_min(1e-8)
                losses.append(mse)
                fvus.append(float((mse / var).detach().cpu().item()))
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), args.grad_clip)
        opt.step()
        sae.global_step += 1

        n_reset = 0
        if args.resample_every > 0 and step >= args.dead_threshold and step % args.resample_every == 0:
            n_reset = sae.reinit_dead(args.dead_threshold)

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            with torch.no_grad():
                alive = ((int(sae.global_step.item()) - sae.last_fired) <= args.dead_threshold).float().mean().item()
            row = {
                "step": step,
                "loss": float(loss.detach().cpu().item()),
                "fvu_mean": float(np.mean(fvus)) if fvus else float("nan"),
                "alive_mean": float(alive),
                "n_reset": n_reset,
                "seconds": time.time() - t0,
            }
            metrics.append(row)
            append_tsv(metric_path, row, fields)
            print(
                f"step={step} loss={row['loss']:.5f} fvu={row['fvu_mean']:.4f} "
                f"alive={row['alive_mean']:.3f} reset={n_reset} elapsed={row['seconds']:.1f}s",
                flush=True,
            )
    return metrics


def evaluate(args, pm, records: list[dict[str, str]], sae: AttentionOutputSAE) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    per_layer = {layer: {
        "sum_x": torch.zeros(sae.d_sae, dtype=torch.float64),
        "sum_x2": torch.zeros(sae.d_sae, dtype=torch.float64),
        "sum_y": 0.0,
        "sum_y2": 0.0,
        "sum_xy": torch.zeros(sae.d_sae, dtype=torch.float64),
        "n": 0,
        "first2_sum": torch.zeros(sae.d_sae, dtype=torch.float64),
        "first2_n": 0,
        "other_sum": torch.zeros(sae.d_sae, dtype=torch.float64),
        "other_n": 0,
        "fired": torch.zeros(sae.d_sae, dtype=torch.bool),
        "fvu": [],
    } for layer in sae.layers}

    for rec in records[: args.eval_sequences]:
        ids = pm.tokenize(rec["sequence"]).to(args.device)
        spans = token_residue_spans(pm.tokenizer, ids, rec["sequence"])
        first2 = np.asarray([bool(span and any(pos <= 1 for pos in span)) for span in spans], dtype=bool)
        caps, out = capture_attention_outputs(pm, ids, sae.layers, output_attentions=True)
        for layer in sae.layers:
            encoded = sae.forward_layer(layer, caps[layer])
            feats = encoded["features"][0].detach().float().cpu()
            n = min(feats.shape[0], len(spans))
            feats = feats[:n]
            y = attention_received_first2(out.attentions, spans, layer)[:n]
            valid = np.isfinite(y)
            if valid.any():
                x = feats[valid].double()
                yy = torch.from_numpy(y[valid].astype(np.float64))
                st = per_layer[layer]
                st["sum_x"] += x.sum(dim=0)
                st["sum_x2"] += (x ** 2).sum(dim=0)
                st["sum_y"] += float(yy.sum().item())
                st["sum_y2"] += float((yy ** 2).sum().item())
                st["sum_xy"] += (x * yy[:, None]).sum(dim=0)
                st["n"] += int(x.shape[0])
            fmask = first2[:n]
            st = per_layer[layer]
            if fmask.any():
                st["first2_sum"] += feats[fmask].double().sum(dim=0)
                st["first2_n"] += int(fmask.sum())
            if (~fmask).any():
                st["other_sum"] += feats[~fmask].double().sum(dim=0)
                st["other_n"] += int((~fmask).sum())
            st["fired"] |= (feats > 0).any(dim=0)
            mse = ((encoded["recon"] - encoded["x_norm"]) ** 2).mean()
            var = encoded["x_norm"].var().clamp_min(1e-8)
            st["fvu"].append(float((mse / var).detach().cpu().item()))

    for layer in sae.layers:
        st = per_layer[layer]
        n = max(int(st["n"]), 1)
        mean_x = st["sum_x"] / n
        mean_y = st["sum_y"] / n
        cov = st["sum_xy"] / n - mean_x * mean_y
        var_x = st["sum_x2"] / n - mean_x ** 2
        var_y = st["sum_y2"] / n - mean_y ** 2
        corr = cov / torch.sqrt(var_x.clamp_min(1e-12) * max(var_y, 1e-12))
        first2_mean = st["first2_sum"] / max(int(st["first2_n"]), 1)
        other_mean = st["other_sum"] / max(int(st["other_n"]), 1)
        delta = first2_mean - other_mean
        score = torch.nan_to_num(corr, nan=0.0).clamp_min(0) + delta.clamp_min(0)
        top = torch.topk(score, k=min(args.top_features, sae.d_sae)).indices.tolist()
        for fid in top:
            rows.append({
                "layer": layer,
                "feature": int(fid),
                "score": float(score[fid].item()),
                "corr_attention_received": float(corr[fid].item()),
                "first2_mean": float(first2_mean[fid].item()),
                "other_mean": float(other_mean[fid].item()),
                "first2_delta": float(delta[fid].item()),
                "fired": int(bool(st["fired"][fid].item())),
                "layer_alive_fraction": float(st["fired"].float().mean().item()),
                "layer_eval_fvu": float(np.mean(st["fvu"])) if st["fvu"] else float("nan"),
            })

    rows.sort(key=lambda r: (r["score"], r["corr_attention_received"], r["first2_delta"]), reverse=True)
    write_tsv(
        args.out_dir / "feature_diagnostics.tsv",
        rows,
        [
            "layer", "feature", "score", "corr_attention_received", "first2_mean",
            "other_mean", "first2_delta", "fired", "layer_alive_fraction", "layer_eval_fvu",
        ],
    )
    summary = {
        "eval_sequences": min(args.eval_sequences, len(records)),
        "layers": sae.layers,
        "top_features": rows[:20],
        "mean_eval_fvu": float(np.mean([r["layer_eval_fvu"] for r in rows if math.isfinite(float(r["layer_eval_fvu"]))])) if rows else float("nan"),
        "max_attention_corr": max((float(r["corr_attention_received"]) for r in rows), default=float("nan")),
        "max_first2_delta": max((float(r["first2_delta"]) for r in rows), default=float("nan")),
    }
    return rows, summary


def make_ablation_hook(sae: AttentionOutputSAE, layer: int, features: list[int], multiplier: float):
    def hook(_module, inputs):
        x = inputs[0]
        delta = sae.delta_for_ablation(layer, x, features, multiplier=multiplier)
        return (x + delta.to(x.dtype),) + tuple(inputs[1:])
    return hook


def forward_with_feature_patch(pm, ids: torch.Tensor, sae: AttentionOutputSAE, condition: dict[str, Any] | None):
    hooks = []
    if condition is not None and condition.get("features"):
        layer = int(condition["layer"])
        module = attn_projection_module(pm, layer)
        hooks.append(module.register_forward_pre_hook(make_ablation_hook(
            sae, layer, [int(x) for x in condition["features"]], float(condition.get("multiplier", 0.0))
        )))
    try:
        with torch.no_grad():
            return pm.model(ids, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()


def ablation(args, pm, records: list[dict[str, str]], sae: AttentionOutputSAE, feature_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not feature_rows:
        return {"status": "skipped", "reason": "no feature diagnostics"}
    by_layer: dict[int, list[int]] = defaultdict(list)
    for row in feature_rows:
        layer = int(row["layer"])
        if len(by_layer[layer]) < args.ablate_features_per_layer:
            by_layer[layer].append(int(row["feature"]))
    rng = random.Random(args.seed + 333)
    conditions = []
    for layer, feats in sorted(by_layer.items()):
        conditions.append({"condition": f"target_L{layer}", "kind": "target", "layer": layer, "features": feats, "multiplier": 0.0})
        for j in range(args.n_random_controls):
            rand_feats = rng.sample(range(sae.d_sae), k=len(feats))
            conditions.append({"condition": f"random{j+1}_L{layer}", "kind": "random", "layer": layer, "features": rand_feats, "multiplier": 0.0})

    rows = []
    for rec in records[: args.ablate_sequences]:
        ids = pm.tokenize(rec["sequence"]).to(args.device)
        spans = token_residue_spans(pm.tokenizer, ids, rec["sequence"])
        base = forward_with_feature_patch(pm, ids, sae, None)
        base_nll = shifted_nll(base.logits, ids)
        for cond in conditions:
            patched = forward_with_feature_patch(pm, ids, sae, cond)
            patched_nll = shifted_nll(patched.logits, ids)
            delta = patched_nll - base_nll
            rec_delta = summarize_delta_nll(delta, spans)
            rows.append({
                "seq_id": rec["id"],
                "condition": cond["condition"],
                "kind": cond["kind"],
                "layer": cond["layer"],
                "features": cond["features"],
                **rec_delta,
            })

    write_tsv(
        args.out_dir / "ablation_per_sequence.tsv",
        rows,
        ["seq_id", "condition", "kind", "layer", "features", "delta_nll_all", "delta_nll_first2", "delta_nll_pos2_10", "delta_nll_pos11_plus"],
    )
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["kind"])].append(row)
    summary_rows = []
    for (condition, kind), items in sorted(groups.items()):
        out = {"condition": condition, "kind": kind, "n_sequences": len(items)}
        for key in ["delta_nll_all", "delta_nll_first2", "delta_nll_pos2_10", "delta_nll_pos11_plus"]:
            vals = [float(x[key]) for x in items if math.isfinite(float(x[key]))]
            out[key] = float(np.mean(vals)) if vals else float("nan")
        summary_rows.append(out)
    write_tsv(
        args.out_dir / "ablation_summary.tsv",
        summary_rows,
        ["condition", "kind", "n_sequences", "delta_nll_all", "delta_nll_first2", "delta_nll_pos2_10", "delta_nll_pos11_plus"],
    )
    return {"conditions": conditions, "summary_rows": summary_rows}


def save_checkpoint(args, sae: AttentionOutputSAE, summary: dict[str, Any]) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": sae.state_dict(),
            "layers": sae.layers,
            "d_model": sae.d_model,
            "d_sae": sae.d_sae,
            "k": sae.k,
            "summary": summary,
        },
        args.out_dir / "attention_output_sae.pt",
    )
    (args.out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))


def load_checkpoint(args, device: str) -> AttentionOutputSAE:
    payload = torch.load(args.checkpoint, map_location=device)
    sae = AttentionOutputSAE(payload["layers"], payload["d_model"], payload["d_sae"], payload["k"])
    sae.load_state_dict(payload["state_dict"])
    sae.to(device).eval()
    return sae


def write_summary(args, summary: dict[str, Any]) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    lines = [
        "# Attention-Output Sparse Pilot",
        "",
        f"- Model: `{args.model}`",
        f"- Layers: {summary.get('layers')}",
        f"- d_sae: {args.d_sae}",
        f"- k: {args.k}",
        f"- Train records: {summary.get('n_train_records')}",
        f"- Train steps: {summary.get('steps')}",
        f"- Mean eval FVU: {summary.get('eval', {}).get('mean_eval_fvu')}",
        f"- Max attention-received correlation: {summary.get('eval', {}).get('max_attention_corr')}",
        f"- Max first2 delta: {summary.get('eval', {}).get('max_first2_delta')}",
        "",
        "## Interpretation",
        "",
    ]
    eval_summary = summary.get("eval", {})
    fvu = float(eval_summary.get("mean_eval_fvu", float("nan")))
    corr = float(eval_summary.get("max_attention_corr", float("nan")))
    delta = float(eval_summary.get("max_first2_delta", float("nan")))
    if math.isfinite(fvu) and fvu < args.pass_fvu and ((math.isfinite(corr) and corr >= args.pass_corr) or (math.isfinite(delta) and delta > args.pass_delta)):
        lines.append("- Pilot readout gate: PASS. The attention-output sparse model is worth scaling.")
    else:
        lines.append("- Pilot readout gate: FAIL/WEAK. Do not scale unless manual inspection finds a compelling feature.")
    lines += [
        "- This is a single-model pilot and is not a cross-model conservation result.",
        "- A positive ablation trend should be treated as hypothesis-generating until repeated with random controls and all three models.",
        "",
        "## Files",
        "",
        "- `training_metrics.tsv`",
        "- `feature_diagnostics.tsv`",
        "- `ablation_summary.tsv`",
        "- `ablation_per_sequence.tsv`",
        "- `attention_output_sae.pt`",
        "- `summary.json`",
        "",
    ]
    (args.out_dir / "summary.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train_eval_ablate", "eval_ablate"], default="train_eval_ablate")
    ap.add_argument("--model", default="zymctrl")
    ap.add_argument("--data", type=Path, default=REPO / "data/zymctrl/ec_labeled_swissprot.fasta")
    ap.add_argument("--layers", type=int, nargs="+", default=[23])
    ap.add_argument("--d-sae", type=int, default=2048)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-sequences", type=int, default=5000)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--calibration-sequences", type=int, default=64)
    ap.add_argument("--eval-sequences", type=int, default=120)
    ap.add_argument("--ablate-sequences", type=int, default=80)
    ap.add_argument("--top-features", type=int, default=100)
    ap.add_argument("--ablate-features-per-layer", type=int, default=3)
    ap.add_argument("--n-random-controls", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--resample-every", type=int, default=500)
    ap.add_argument("--dead-threshold", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260514)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--pass-fvu", type=float, default=0.50)
    ap.add_argument("--pass-corr", type=float, default=0.30)
    ap.add_argument("--pass-delta", type=float, default=0.05)
    ap.add_argument("--out-dir", type=Path, default=R2 / "results/circuit_analysis/attention_output_transcoder_pilot_20260514")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading records from {args.data}", flush=True)
    records = read_fasta_records(args.data, args.num_sequences, args.max_length, args.seed)
    if len(records) < max(args.batch_size, 2):
        raise RuntimeError(f"not enough records loaded from {args.data}")
    print(f"Loaded {len(records)} records", flush=True)

    print(f"Loading model {args.model}", flush=True)
    pm = load_model(args.model, device=args.device, dtype=torch.float16)
    pm.model.eval()
    for p in pm.model.parameters():
        p.requires_grad_(False)

    if args.mode == "eval_ablate":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for eval_ablate")
        sae = load_checkpoint(args, args.device)
    else:
        sae = AttentionOutputSAE(args.layers, pm.d_model, args.d_sae, args.k).to(args.device)
        print("Calibrating per-layer attention-output RMS scales", flush=True)
        scales = calibrate_scales(pm, sae, records, args.calibration_sequences, args.device)
        print(f"Scales: {dict(zip(args.layers, scales))}", flush=True)
        train_metrics = train(args, pm, records, sae)
        summary_seed = {"train_metrics_tail": train_metrics[-5:] if train_metrics else []}
        save_checkpoint(args, sae, summary_seed)

    sae.eval()
    feature_rows, eval_summary = evaluate(args, pm, records, sae)
    ablation_summary = ablation(args, pm, records, sae, feature_rows)

    summary = {
        "task": "R2 attention-output sparse pilot",
        "model": args.model,
        "layers": args.layers,
        "d_sae": args.d_sae,
        "k": args.k,
        "steps": args.steps if args.mode == "train_eval_ablate" else 0,
        "n_train_records": len(records),
        "data": str(args.data),
        "checkpoint": str(args.out_dir / "attention_output_sae.pt"),
        "eval": eval_summary,
        "ablation": ablation_summary,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_checkpoint(args, sae, summary)
    write_summary(args, summary)
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
