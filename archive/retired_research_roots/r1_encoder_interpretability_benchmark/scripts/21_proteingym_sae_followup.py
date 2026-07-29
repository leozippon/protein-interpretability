#!/usr/bin/env python
"""ProteinGym SAE follow-up scoring.

This fills the R1-D gap left by ``17_proteingym_benchmark.py``: the original
run computed ESM-2 masked-marginal LLR for ProteinGym, but left
``spearman_sae`` and ``spearman_ensemble`` as null. This script computes a
resume-friendly SAE perturbation score for each assay and updates a separate
JSON after every assay.

The score is deliberately simple and aligned with the R1 variant-effect
pipeline: for each mutant, compare WT and mutant SAE activations at mutated
positions, weight absolute feature deltas by annotation F1, then correlate the
per-mutant scores with DMS fitness. Because this SAE score is a damage
magnitude, the sign-corrected ensemble is an assay-local z-sum of LLR minus
the SAE score. The older plus-sign ensemble is still reported for backwards
compatibility and diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, EsmForMaskedLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.feature_annotation import load_our_sae


AA = set("ACDEFGHIKLMNPQRSTVWY")


def find_assays(proteingym_dir: str) -> list[str]:
    return [str(p) for p in sorted(Path(proteingym_dir).rglob("*.csv")) if p.is_file()]


def reconstruct_wt(mut_seq: str, mut_str: str) -> str | None:
    try:
        first = mut_str.split(":")[0]
        wt_aa = first[0]
        pos = int(first[1:-1])
        if not (0 < pos <= len(mut_seq)):
            return None
        return mut_seq[:pos - 1] + wt_aa + mut_seq[pos:]
    except (ValueError, IndexError):
        return None


def load_assay(csv_path: str, max_mutants: int = 0) -> dict | None:
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return None
        target_col = None
        for cand in ("DMS_score", "DMS_score_bin", "experimental_score"):
            if cand in reader.fieldnames:
                target_col = cand
                break
        if target_col is None or "mutant" not in reader.fieldnames:
            return None
        for row in reader:
            rows.append(row)
            if max_mutants and len(rows) >= max_mutants:
                break
    if not rows:
        return None
    try:
        mutants = [r["mutant"] for r in rows]
        sequences = [r.get("mutated_sequence", "") for r in rows]
        scores = np.array([float(r[target_col]) for r in rows], dtype=np.float32)
    except (KeyError, ValueError):
        return None
    wt_seq = reconstruct_wt(sequences[0], mutants[0]) if sequences else None
    if not wt_seq:
        return None
    return {
        "name": Path(csv_path).stem,
        "path": csv_path,
        "wt_seq": clean_sequence(wt_seq),
        "mutants": mutants,
        "sequences": [clean_sequence(s) for s in sequences],
        "scores": scores,
    }


def clean_sequence(seq: str) -> str:
    return "".join(c for c in seq.upper() if c in AA)


def parse_mut_positions(mutant: str, max_len: int) -> list[int]:
    out = []
    for part in mutant.split(":"):
        try:
            pos = int(part[1:-1])
        except (ValueError, IndexError):
            continue
        idx = pos - 1
        if 0 <= idx < max_len:
            out.append(idx)
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 5:
        return float("nan")
    return float(spearmanr(a[mask], b[mask]).correlation)


def zscore(x: np.ndarray) -> np.ndarray:
    y = x.astype(np.float32).copy()
    mask = np.isfinite(y)
    if mask.sum() < 2:
        return np.full_like(y, np.nan, dtype=np.float32)
    mu = float(y[mask].mean())
    sd = float(y[mask].std())
    if sd < 1e-8:
        return np.zeros_like(y, dtype=np.float32)
    y[mask] = (y[mask] - mu) / sd
    y[~mask] = np.nan
    return y


def load_f1_weights(annotation_dir: str, layers: list[int]) -> dict[int, np.ndarray]:
    weights = {}
    for layer in layers:
        path = Path(annotation_dir) / f"ours_3B_l{layer}_step500000.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        arr = np.zeros(len(data["results"]), dtype=np.float32)
        for r in data["results"]:
            if getattr(r, "alive", False):
                arr[r.feature_idx] = float(getattr(r, "best_f1", 0.0) or 0.0)
        weights[layer] = arr
    return weights


@torch.no_grad()
def masked_llr(model, tokenizer, wt_seq: str, mutants: list[str],
               device: str, max_len: int, batch_size: int) -> np.ndarray:
    seq = wt_seq[:max_len]
    toks = tokenizer(" ".join(seq), return_tensors="pt").to(device)
    n = toks["input_ids"].shape[1]
    pos_positions = list(range(1, n - 1))
    mask_id = tokenizer.mask_token_id
    input_ids = toks["input_ids"].repeat(len(pos_positions), 1)
    for bi, p in enumerate(pos_positions):
        input_ids[bi, p] = mask_id

    logps = []
    for start in range(0, len(pos_positions), batch_size):
        chunk = input_ids[start:start + batch_size].to(device)
        out = model(chunk)
        lp = torch.log_softmax(out.logits.float(), dim=-1)
        for bi, p in enumerate(pos_positions[start:start + batch_size]):
            logps.append(lp[bi, p].detach().cpu().numpy())
    logp = np.stack(logps)
    vocab = tokenizer.get_vocab()

    scores = np.full(len(mutants), np.nan, dtype=np.float32)
    for i, mutant in enumerate(mutants):
        deltas = []
        for part in mutant.split(":"):
            try:
                wt_aa = part[0]
                pos = int(part[1:-1])
                mu_aa = part[-1]
            except (ValueError, IndexError):
                continue
            idx = pos - 1
            if not (0 <= idx < logp.shape[0]):
                continue
            if wt_aa not in vocab or mu_aa not in vocab:
                continue
            row = logp[idx]
            deltas.append(float(row[vocab[mu_aa]] - row[vocab[wt_aa]]))
        if deltas:
            scores[i] = float(sum(deltas))
    return scores


@torch.no_grad()
def layer_sae_scores(model, tokenizer, sae, layer: int, f1_weights: np.ndarray,
                     wt_seq: str, sequences: list[str], mutants: list[str],
                     device: str, max_len: int, batch_size: int) -> np.ndarray:
    wt = wt_seq[:max_len]
    wt_encoded = tokenizer(" ".join(wt), return_tensors="pt",
                           truncation=True, max_length=max_len + 2).to(device)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        wt_out = model(
            input_ids=wt_encoded["input_ids"],
            attention_mask=wt_encoded["attention_mask"],
            output_hidden_states=True,
        )
    wt_acts = wt_out.hidden_states[layer + 1][0, 1:len(wt) + 1].float()
    wt_feats = sae(wt_acts).f.detach().cpu().numpy()

    scores = np.full(len(sequences), np.nan, dtype=np.float32)
    for start in range(0, len(sequences), batch_size):
        batch_seqs = [s[:max_len] for s in sequences[start:start + batch_size]]
        spaced = [" ".join(s) for s in batch_seqs]
        enc = tokenizer(
            spaced, padding=True, truncation=True, max_length=max_len + 2,
            return_tensors="pt",
        ).to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
            )
        layer_acts = out.hidden_states[layer + 1].float()
        for j, seq in enumerate(batch_seqs):
            global_idx = start + j
            pos = parse_mut_positions(mutants[global_idx], min(len(seq), len(wt)))
            if not pos:
                continue
            aa_acts = layer_acts[j, 1:len(seq) + 1]
            feats = sae(aa_acts).f.detach().cpu().numpy()
            vals = []
            for pi in pos:
                if pi >= feats.shape[0] or pi >= wt_feats.shape[0]:
                    continue
                delta = np.abs(feats[pi] - wt_feats[pi])
                vals.append(float((delta * f1_weights).sum()))
            if vals:
                scores[global_idx] = float(sum(vals))
    return scores


def load_existing(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"per_assay": [], "summary": {}}


def write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proteingym-dir", default="data/proteingym/DMS_ProteinGym_substitutions")
    ap.add_argument("--esm-path", default="/Data/public/esm2_t36_3B_UR50D")
    ap.add_argument("--checkpoint-root", default="results/final_checkpoints/r1_h200_2gpu_20260401/sae_weights")
    ap.add_argument("--annotation-dir", default="results/annotation_alignment")
    ap.add_argument("--layers", default="19,23,27,31,35")
    ap.add_argument("--existing", default="results/variant_effect/proteingym_benchmark.json")
    ap.add_argument("--out", default="results/variant_effect/proteingym_benchmark_sae.json")
    ap.add_argument("--max-assays", type=int, default=0)
    ap.add_argument("--max-mutants-per-assay", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--llr-batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=1022)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    assays = find_assays(args.proteingym_dir)
    if args.max_assays:
        assays = assays[:args.max_assays]

    result = load_existing(args.out)
    done = {r["name"] for r in result.get("per_assay", [])
            if r.get("spearman_sae") is not None
            and r.get("spearman_ensemble_signed") is not None
            and not args.force}

    print("=" * 70)
    print("  R1-D ProteinGym SAE follow-up")
    print("=" * 70)
    print(f"  assays={len(assays)} layers={layers} done={len(done)}")

    tokenizer = AutoTokenizer.from_pretrained(args.esm_path)
    model = EsmForMaskedLM.from_pretrained(
        args.esm_path, torch_dtype=torch.float16
    ).to(args.device).eval()
    f1 = load_f1_weights(args.annotation_dir, layers)

    existing_llr = {}
    if os.path.exists(args.existing):
        with open(args.existing) as f:
            for rec in json.load(f).get("per_assay", []):
                existing_llr[rec["name"]] = rec.get("spearman_llr")

    per_assay = [r for r in result.get("per_assay", []) if r["name"] in done]
    by_name = {r["name"]: r for r in per_assay}

    for idx, csv_path in enumerate(assays, 1):
        name = Path(csv_path).stem
        if name in done:
            print(f"[{idx}/{len(assays)}] {name}: skip")
            continue
        assay = load_assay(csv_path, args.max_mutants_per_assay)
        if assay is None or len(assay["mutants"]) < 20:
            print(f"[{idx}/{len(assays)}] {name}: skip invalid")
            continue

        t0 = time.time()
        print(f"[{idx}/{len(assays)}] {name}: n={len(assay['mutants'])}", flush=True)
        llr = masked_llr(
            model, tokenizer, assay["wt_seq"], assay["mutants"],
            args.device, args.max_len, args.llr_batch_size,
        )

        sae_total = np.zeros(len(assay["mutants"]), dtype=np.float32)
        valid_any = np.zeros(len(assay["mutants"]), dtype=bool)
        for layer in layers:
            ckpt = Path(args.checkpoint_root) / f"layer_{layer}" / "step_500000"
            print(f"  layer {layer}: {ckpt}", flush=True)
            sae = load_our_sae(str(ckpt), device=args.device)
            layer_scores = layer_sae_scores(
                model, tokenizer, sae, layer, f1[layer], assay["wt_seq"],
                assay["sequences"], assay["mutants"], args.device,
                args.max_len, args.batch_size,
            )
            mask = np.isfinite(layer_scores)
            sae_total[mask] += layer_scores[mask]
            valid_any |= mask
            del sae
            torch.cuda.empty_cache()
        sae_total[~valid_any] = np.nan

        dms = assay["scores"]
        rho_llr = spearman(llr, dms)
        rho_sae = spearman(sae_total, dms)
        ensemble_legacy = zscore(llr) + zscore(sae_total)
        ensemble_signed = zscore(llr) - zscore(sae_total)
        rho_ensemble_legacy = spearman(ensemble_legacy, dms)
        rho_ensemble_signed = spearman(ensemble_signed, dms)
        rec = {
            "name": name,
            "n_mutants": len(assay["mutants"]),
            "wt_len": len(assay["wt_seq"]),
            "spearman_llr": rho_llr,
            "spearman_llr_existing": existing_llr.get(name),
            "spearman_sae": rho_sae,
            "spearman_ensemble": rho_ensemble_legacy,
            "spearman_ensemble_legacy_plus": rho_ensemble_legacy,
            "spearman_ensemble_signed": rho_ensemble_signed,
            "n_sae_scored": int(valid_any.sum()),
            "elapsed_s": round(time.time() - t0, 1),
        }
        by_name[name] = rec
        per_assay = [by_name[k] for k in sorted(by_name)]
        vals_llr = [r["spearman_llr"] for r in per_assay if math.isfinite(r["spearman_llr"])]
        vals_sae = [r["spearman_sae"] for r in per_assay if math.isfinite(r["spearman_sae"])]
        vals_ens = [r["spearman_ensemble"] for r in per_assay if math.isfinite(r["spearman_ensemble"])]
        vals_ens_signed = [
            r["spearman_ensemble_signed"] for r in per_assay
            if math.isfinite(r.get("spearman_ensemble_signed", float("nan")))
        ]
        result = {
            "per_assay": per_assay,
            "summary": {
                "n_assays_scored": len(per_assay),
                "n_assays_with_sae": len(vals_sae),
                "mean_rho_llr": float(np.mean(vals_llr)) if vals_llr else float("nan"),
                "mean_rho_sae": float(np.mean(vals_sae)) if vals_sae else float("nan"),
                "mean_rho_ensemble": float(np.mean(vals_ens)) if vals_ens else float("nan"),
                "mean_rho_ensemble_legacy_plus": float(np.mean(vals_ens)) if vals_ens else float("nan"),
                "mean_rho_ensemble_signed": float(np.mean(vals_ens_signed)) if vals_ens_signed else float("nan"),
                "layers": layers,
                "max_mutants_per_assay": args.max_mutants_per_assay,
            },
        }
        write_json(args.out, result)
        print(
            f"  rho LLR={rho_llr:+.4f} SAE={rho_sae:+.4f} "
            f"ENS+={rho_ensemble_legacy:+.4f} ENSsigned={rho_ensemble_signed:+.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
