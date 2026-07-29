#!/usr/bin/env python
"""ProteinGym substitution benchmark for SAE+LLR ensemble (R1-D).

ProteinGym (Notin et al. 2023) ships 217 deep mutational scanning (DMS)
assays with standardized scoring. ESM-2 zero-shot LLR is the standard
baseline (published Spearman ~0.40 averaged across assays). This benchmark
places our SAE+LLR ensemble on the same ruler.

Pipeline (per assay):
  1. Load WT sequence + mutant table from ProteinGym CSV
  2. Run ESM-2 LLR on all mutants → `llr` column
  3. Run SAE perturbation signature on all mutants → compute
     annotation-weighted score using the SAE already trained for R1
  4. Spearman correlation of (LLR, SAE, LLR+SAE) vs DMS target column
  5. Aggregate across all assays; report mean Spearman + per-assay deltas

Skipped by default if ProteinGym not present. Once the dataset is staged
under `data/proteingym/DMS_ProteinGym_substitutions/`, rerun the script.

Usage:
    python r1_encoder_interpretability_benchmark/scripts/17_proteingym_benchmark.py \
        --proteingym-dir data/proteingym/DMS_ProteinGym_substitutions \
        --checkpoints r1_encoder_interpretability_benchmark/results/final_checkpoints \
        --out r1_encoder_interpretability_benchmark/results/variant_effect/proteingym_benchmark.json

Notes:
  ProteinGym download instructions:
    https://github.com/OATML-Markslab/ProteinGym
    `https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip`
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_assays(proteingym_dir: str) -> list[str]:
    """Locate per-assay CSVs under ProteinGym substitutions folder."""
    out = []
    p = Path(proteingym_dir)
    if not p.exists():
        return out
    for csv_path in sorted(p.rglob("*.csv")):
        # ProteinGym v1 layout: one CSV per assay inside the folder
        if csv_path.is_file():
            out.append(str(csv_path))
    return out


def load_assay(csv_path: str) -> dict | None:
    """Read a ProteinGym assay CSV. Returns dict with wt_seq, mutants, scores."""
    rows = []
    target_col = None
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return None
        # Find the DMS score column — ProteinGym typically uses "DMS_score"
        for cand in ("DMS_score", "DMS_score_bin", "experimental_score"):
            if cand in reader.fieldnames:
                target_col = cand
                break
        if target_col is None:
            return None
        for r in reader:
            rows.append(r)
    if not rows:
        return None
    # Extract WT and per-mutant info. ProteinGym columns typically:
    # mutant, mutated_sequence, DMS_score
    try:
        mutants = [r["mutant"] for r in rows]
        sequences = [r.get("mutated_sequence", "") for r in rows]
        scores = np.array([float(r[target_col]) for r in rows], dtype=np.float32)
    except (KeyError, ValueError):
        return None

    # The WT sequence is the first mutant's sequence with the mutation reverted,
    # OR deduced from the mutant string. Here we assume a WT reference file or
    # reconstruct from the mutations.
    wt_seq = None
    if sequences and mutants:
        wt_seq = reconstruct_wt(sequences[0], mutants[0])
    return {
        "name": Path(csv_path).stem,
        "wt_seq": wt_seq,
        "mutants": mutants,
        "sequences": sequences,
        "scores": scores,
        "n": len(mutants),
    }


def reconstruct_wt(mut_seq: str, mut_str: str) -> str | None:
    """Given a mutated sequence and mutant string like 'A123B', recover WT."""
    try:
        wt_aa = mut_str[0]
        pos = int(mut_str[1:-1])
        if not (0 < pos <= len(mut_seq)):
            return None
        return mut_seq[:pos - 1] + wt_aa + mut_seq[pos:]
    except (ValueError, IndexError):
        return None


@torch.no_grad()
def esm_llr_batch(model, tokenizer, wt_seq: str, mutants: list[str],
                  device: str = "cuda", max_len: int = 1022) -> np.ndarray:
    """Compute masked-marginal LLR for each mutant vs WT."""
    # Truncate very long sequences
    if len(wt_seq) > max_len:
        wt_seq = wt_seq[:max_len]

    tokens = tokenizer(wt_seq, return_tensors="pt").to(device)
    n = tokens["input_ids"].shape[1]

    # Predict log probabilities for every position with position masked.
    # For efficiency we batch mask all positions in one pass using the
    # diagonal-mask trick.
    # logp[i,a] = log P(residue a | seq with position i masked)
    mask_id = tokenizer.mask_token_id
    pos_positions = list(range(1, n - 1))  # skip BOS/EOS if present
    # Batched masked predictions
    input_ids = tokens["input_ids"].repeat(len(pos_positions), 1)
    for bi, p in enumerate(pos_positions):
        input_ids[bi, p] = mask_id
    # Break into mini-batches to fit memory
    batch_size = 32
    logps = []
    for s in range(0, len(pos_positions), batch_size):
        chunk = input_ids[s:s + batch_size].to(device)
        out = model(chunk)
        logits = out.logits.float()
        lp = torch.log_softmax(logits, dim=-1)
        for bi, p in enumerate(pos_positions[s:s + batch_size]):
            logps.append(lp[bi, p].cpu().numpy())
    logp = np.stack(logps)  # (seq_len-2, vocab)

    # Score each mutant
    scores = np.zeros(len(mutants), dtype=np.float32)
    for i, m in enumerate(mutants):
        # Multi-substitution mutants use ":" separator in ProteinGym
        deltas = []
        for part in m.split(":"):
            try:
                wt_aa = part[0]
                pos = int(part[1:-1])
                mu_aa = part[-1]
            except (ValueError, IndexError):
                continue
            if not (0 < pos - 1 < len(pos_positions)):
                continue
            row = logp[pos - 1]  # 0-indexed into pos_positions
            try:
                wt_t = tokenizer.get_vocab()[wt_aa]
                mu_t = tokenizer.get_vocab()[mu_aa]
            except KeyError:
                continue
            deltas.append(row[mu_t] - row[wt_t])
        scores[i] = sum(deltas) if deltas else np.nan
    return scores


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 5:
        return float("nan")
    return float(spearmanr(a[mask], b[mask]).correlation)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proteingym-dir",
                    default="data/proteingym/DMS_ProteinGym_substitutions")
    ap.add_argument("--esm-path",
                    default="/Data/public/esm2_t36_3B_UR50D")
    ap.add_argument("--out",
                    default="r1_encoder_interpretability_benchmark/results/variant_effect/proteingym_benchmark.json")
    ap.add_argument("--max-assays", type=int, default=0,
                    help="0 = all assays")
    ap.add_argument("--llr-only", action="store_true",
                    help="Run only the LLR part (SAE scoring requires extra infra)")
    args = ap.parse_args()

    print("=" * 70)
    print("  R1-D ProteinGym benchmark")
    print("=" * 70)

    assays = find_assays(args.proteingym_dir)
    if not assays:
        print(f"\n  ProteinGym not found at {args.proteingym_dir}")
        print("  Download:")
        print("    mkdir -p data/proteingym")
        print("    wget https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip -P data/proteingym/")
        print("    unzip data/proteingym/DMS_ProteinGym_substitutions.zip -d data/proteingym/")
        print("  Then rerun this script.")
        return
    if args.max_assays:
        assays = assays[:args.max_assays]
    print(f"  Found {len(assays)} assays")

    # Load ESM-2
    print(f"\n  Loading ESM-2-3B from {args.esm_path}...")
    from transformers import AutoTokenizer, EsmForMaskedLM
    tokenizer = AutoTokenizer.from_pretrained(args.esm_path)
    model = EsmForMaskedLM.from_pretrained(args.esm_path, dtype=torch.float16).cuda().eval()

    results = {"per_assay": [], "summary": {}}

    for ai, path in enumerate(assays):
        name = Path(path).stem
        print(f"\n  [{ai+1}/{len(assays)}] {name}", flush=True)
        assay = load_assay(path)
        if assay is None or not assay["wt_seq"]:
            print(f"    skip (no WT or score column)")
            continue
        if assay["n"] < 20:
            print(f"    skip (n={assay['n']} < 20)")
            continue

        t0 = time.time()
        llr = esm_llr_batch(model, tokenizer, assay["wt_seq"], assay["mutants"])
        dms = assay["scores"]
        # ProteinGym convention: higher DMS = fitter. LLR(mut|WT) higher = fitter.
        rho_llr = spearman(llr, dms)

        rec = {
            "name": name,
            "n_mutants": assay["n"],
            "spearman_llr": rho_llr,
            "wt_len": len(assay["wt_seq"]),
            "elapsed_s": round(time.time() - t0, 1),
        }
        print(f"    n={assay['n']}  rho_LLR={rho_llr:+.4f}  ({rec['elapsed_s']}s)")

        if not args.llr_only:
            # SAE scoring requires a per-protein SAE-run; to avoid O(n_proteins)
            # ESM forward passes here, we leave a hook for the full pipeline:
            # populate these fields via a follow-up script that re-runs the
            # perturbation signature on ProteinGym mutants.
            rec["spearman_sae"] = None
            rec["spearman_ensemble"] = None

        results["per_assay"].append(rec)

    rhos_llr = [r["spearman_llr"] for r in results["per_assay"]
                if np.isfinite(r["spearman_llr"])]
    results["summary"] = {
        "n_assays_scored": len(rhos_llr),
        "mean_rho_llr": float(np.mean(rhos_llr)) if rhos_llr else float("nan"),
        "median_rho_llr": float(np.median(rhos_llr)) if rhos_llr else float("nan"),
        "sae_pending": not args.llr_only,
        "notes": (
            "LLR is the standard ESM-2 zero-shot baseline. The SAE Spearman "
            "will be added by a follow-up run that computes perturbation "
            "signatures on ProteinGym mutants using the R1 trained SAEs."
        ),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\n" + "=" * 70)
    print(f"  Saved: {args.out}")
    print(f"  Mean Spearman (LLR): {results['summary']['mean_rho_llr']:.4f}")
    print(f"  Median Spearman (LLR): {results['summary']['median_rho_llr']:.4f}")


if __name__ == "__main__":
    main()
