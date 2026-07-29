#!/usr/bin/env python
"""Structural validation of generated sequences via ESMFold (R2-F).

For a JSON produced by 10_steered_generation.py or 12_drug_design_case_study.py,
we fold each generation with ESMFold, record per-residue pLDDT, and compute
aggregate metrics to ask: does steering preserve foldability?

Metrics per sequence:
  - mean pLDDT
  - fraction of residues with pLDDT > 70 (confident region)
  - pTM (predicted TM-score, if available in ESMFold output)

Aggregate metrics across the set:
  - mean/median pLDDT distribution
  - fraction of sequences with globally confident fold (mean pLDDT > 70)
  - for drug-design leads: per-lead structural report

Outputs:
  - `<input_stem>_esmfold_metrics.json` next to the input file
  - `<input_stem>_esmfold_pdbs/` directory with up to K predicted structures

Usage:
    python r2_interpretability_transfer/scripts/13_structural_qc.py \
        --input r2_interpretability_transfer/results/drug_design/ec_lysozyme_leads.json \
        --field leads --max-structures 20 \
        --out r2_interpretability_transfer/results/drug_design/ec_lysozyme_esmfold_metrics.json

Notes:
  Requires esm installed (`pip install fair-esm` or esm==2.0.1). On H200,
  load ESMFold in float16 with `--dtype fp16`. For very long sequences
  (>500aa) this script falls back to chunked folding; shorter sequences
  (e.g., 30-60aa drug binders) fold in a few seconds each.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from urllib import request
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_esmfold_local(dtype_str: str = "fp16"):
    import torch
    try:
        from transformers import AutoTokenizer, EsmForProteinFolding
    except ImportError as e:
        raise RuntimeError(
            "transformers ESMFold is required. Install via `pip install "
            "transformers accelerate` and ensure you have the facebook/esmfold_v1 "
            "weights available."
        ) from e
    model_path = os.environ.get("ESMFOLD_PATH", "/Data/public/esmfold_v1")
    if not os.path.exists(model_path):
        model_path = "facebook/esmfold_v1"
    print(f"  Loading ESMFold from {model_path}...")
    dtype = torch.float16 if dtype_str == "fp16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = EsmForProteinFolding.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model = model.eval().cuda()
    return model, tokenizer


def fold_sequence_api(seq: str, timeout_s: int = 180):
    """Fold a sequence via the public ESM Atlas API.

    This is slower than local weights but avoids the need to stage the
    2.6 GB checkpoint in environments that cannot reach Hugging Face.
    """
    seq = "".join(c for c in seq.upper() if c.isalpha())
    if not seq or len(seq) < 5:
        return None
    req = request.Request(
        "https://api.esmatlas.com/foldSequence/v1/pdb/",
        data=seq.encode("utf-8"),
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_s) as resp:
        pdb_bytes = resp.read()
    pdb_str = pdb_bytes.decode("utf-8")

    # ESM Atlas writes pLDDT into the B-factor field. Use CA atoms to get
    # one score per residue.
    plddt = []
    for line in pdb_str.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom = line[12:16].strip()
        if atom != "CA":
            continue
        try:
            plddt.append(float(line[60:66].strip()))
        except ValueError:
            continue
    if not plddt:
        return None
    plddt = np.array(plddt, dtype=np.float32)
    # The public API writes normalized confidence into the B-factor field.
    if float(plddt.max()) <= 1.5:
        plddt *= 100.0
    return {
        "plddt": plddt,
        "mean_plddt": float(plddt.mean()),
        "frac_confident": float((plddt > 70).mean()),
        "ptm": None,
        "pdb": pdb_str,
    }


def _to_pdb(output, seq_len):
    """Convert ESMFold output to PDB string."""
    from transformers.models.esm.openfold_utils.protein import to_pdb, Protein
    from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
    final_atom_positions = atom14_to_atom37(output["positions"][-1], output)
    final_atom_mask = output["atom37_atom_exists"]
    plddt = output["plddt"][0].detach().cpu().numpy()
    if plddt.ndim > 1:
        plddt = plddt.mean(-1)
    if float(np.nanmax(plddt)) <= 1.5:
        plddt = plddt * 100.0
    pred = Protein(
        aatype=output["aatype"][0].cpu().numpy(),
        atom_positions=final_atom_positions[0].cpu().numpy(),
        atom_mask=final_atom_mask[0].cpu().numpy(),
        residue_index=output["residue_index"][0].cpu().numpy() + 1,
        b_factors=plddt[:, None].repeat(37, axis=-1),
        chain_index=np.zeros(seq_len, dtype=np.int32),
    )
    return to_pdb(pred)


def fold_sequence_local(model, tokenizer, seq: str, chunk_size: int = 256):
    """Run ESMFold on a single sequence. Returns dict with plddt and pdb."""
    import torch
    if not seq or len(seq) < 5:
        return None
    seq = "".join(c for c in seq.upper() if c.isalpha())
    input_ids = tokenizer([seq], return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
    with torch.no_grad():
        model.trunk.set_chunk_size(chunk_size)
        output = model(input_ids)
    plddt = output["plddt"][0].cpu().numpy()         # (seq_len, 37) or (seq_len,)
    if plddt.ndim > 1:
        plddt = plddt.mean(-1)
    # Some transformers ESMFold versions return confidence in 0..1 while
    # others return 0..100. Normalize to the conventional pLDDT scale.
    if float(np.nanmax(plddt)) <= 1.5:
        plddt = plddt * 100.0
    ptm = output.get("ptm", None)
    try:
        pdb_str = _to_pdb(output, len(seq))
    except Exception as e:
        print(f"  pdb conversion failed: {e}")
        pdb_str = None
    return {
        "plddt": plddt,
        "mean_plddt": float(plddt.mean()),
        "frac_confident": float((plddt > 70).mean()),
        "ptm": float(ptm.item()) if ptm is not None else None,
        "pdb": pdb_str,
    }


def load_records(path: str, field: str):
    with open(path) as f:
        data = json.load(f)
    if field in data and isinstance(data[field], list):
        recs = data[field]
    elif "records" in data and isinstance(data["records"], list):
        recs = data["records"]
    elif isinstance(data, list):
        recs = data
    else:
        raise ValueError(f"Could not find records in {path} under field '{field}'")
    normalized = []
    for r in recs:
        if isinstance(r, str):
            normalized.append({"sequence": r})
        elif isinstance(r, dict):
            # Accept sequence | steered | raw_output
            seq = r.get("sequence") or r.get("steered") or r.get("raw_output")
            if seq:
                normalized.append({**r, "sequence": seq})
    return normalized, data


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="Input JSON from 10_steered_generation or 12_drug_design")
    ap.add_argument("--field", default="leads",
                    help="Which field of the input JSON to fold "
                         "(leads | records | all_records)")
    ap.add_argument("--max-structures", type=int, default=20,
                    help="Save PDBs for top-N sequences only")
    ap.add_argument("--max-fold", type=int, default=100,
                    help="Run ESMFold on at most this many sequences")
    ap.add_argument("--selection", choices=["first", "random"], default="first",
                    help="Select the first eligible records or a seeded random sample")
    ap.add_argument("--sample-seed", type=int, default=20260716)
    ap.add_argument("--min-length", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--require-complete", action="store_true",
                    help="Fail if any selected record cannot be folded")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pdb-dir", default=None,
                    help="Where to save PDBs; defaults to <out_stem>_pdbs/")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--backend", default="auto", choices=["auto", "local", "api"],
                    help="Use local ESMFold weights, the public ESM Atlas API, "
                         "or auto-fallback from local to API.")
    args = ap.parse_args()

    print("=" * 70)
    print(f"  Structural QC (ESMFold) — {args.input}")
    print("=" * 70)

    recs, orig = load_records(args.input, args.field)
    recs = [
        record for record in recs
        if len(record["sequence"]) >= args.min_length
        and (args.max_length is None or len(record["sequence"]) <= args.max_length)
    ]
    n_eligible = len(recs)
    if args.selection == "random":
        random.Random(args.sample_seed).shuffle(recs)
    recs = recs[:args.max_fold]
    if len(recs) < args.max_fold:
        raise ValueError(
            f"requested {args.max_fold} structures but only {len(recs)} records "
            f"passed length filters"
        )
    print(f"  Folding {len(recs)} sequences")

    model = tokenizer = None
    fold_fn = None
    backend_used = args.backend
    if args.backend in {"auto", "local"}:
        try:
            model, tokenizer = load_esmfold_local(args.dtype)
            fold_fn = lambda seq: fold_sequence_local(model, tokenizer, seq)
            backend_used = "local"
        except Exception as e:
            if args.backend == "local":
                raise
            print(f"  Local ESMFold unavailable ({e}); falling back to API.")
    if fold_fn is None:
        fold_fn = fold_sequence_api
        backend_used = "api"

    software = {"backend": backend_used}
    if backend_used == "local":
        import torch
        import transformers
        model_path = os.environ.get("ESMFOLD_PATH", "/Data/public/esmfold_v1")
        config_path = os.path.join(model_path, "config.json")
        software.update({
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "esmfold_path": model_path,
            "esmfold_config_sha256": sha256_file(config_path) if os.path.isfile(config_path) else None,
            "dtype": args.dtype,
        })

    pdb_dir = args.pdb_dir or args.out.replace(".json", "_pdbs")
    os.makedirs(pdb_dir, exist_ok=True)

    results = []
    t0 = time.time()
    for i, r in enumerate(recs):
        seq = r["sequence"]
        if not seq:
            continue
        try:
            fold = fold_fn(seq)
        except Exception as e:
            print(f"    [{i}] ERROR: {e}")
            fold = None
        if fold is None:
            continue
        rec = {
            "idx": r.get("idx", i),
            "id": r.get("id", f"record_{i:04d}"),
            "source": r.get("source", args.field),
            "seq_len": len(seq),
            "mean_plddt": fold["mean_plddt"],
            "frac_confident": fold["frac_confident"],
            "ptm": fold["ptm"],
            "rank_score_input": r.get("rank_score"),
            "sequence_sha256": hashlib.sha256(seq.encode()).hexdigest(),
        }
        if i < args.max_structures and fold["pdb"]:
            safe_id = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(rec["id"]))[:80]
            pdb_path = os.path.join(pdb_dir, f"{i:04d}_{safe_id}.pdb")
            with open(pdb_path, "w") as pf:
                pf.write(fold["pdb"])
            rec["pdb_path"] = pdb_path
        results.append(rec)
        if (i + 1) % 10 == 0 or i == len(recs) - 1:
            print(f"    {i+1}/{len(recs)}  mean_plddt={fold['mean_plddt']:.2f}  "
                  f"({time.time()-t0:.1f}s)")

    if args.require_complete and len(results) != len(recs):
        raise RuntimeError(f"folded {len(results)}/{len(recs)} selected records")
    arr_plddt = np.array([r["mean_plddt"] for r in results], dtype=np.float32)
    aggregate = {
        "n_folded": len(results),
        "mean_plddt_dist": {
            "mean": float(arr_plddt.mean()) if arr_plddt.size else float("nan"),
            "median": float(np.median(arr_plddt)) if arr_plddt.size else float("nan"),
            "q25": float(np.percentile(arr_plddt, 25)) if arr_plddt.size else float("nan"),
            "q75": float(np.percentile(arr_plddt, 75)) if arr_plddt.size else float("nan"),
        },
        "frac_globally_confident": float((arr_plddt > 70).mean()) if arr_plddt.size else float("nan"),
    }

    out = {
        "input": args.input,
        "input_sha256": sha256_file(args.input),
        "field": args.field,
        "backend": backend_used,
        "software": software,
        "selection": {
            "method": args.selection,
            "seed": args.sample_seed if args.selection == "random" else None,
            "min_length": args.min_length,
            "max_length": args.max_length,
            "n_eligible": n_eligible,
            "n_selected": len(recs),
        },
        "aggregate": aggregate,
        "per_sequence": results,
        "pdb_dir": pdb_dir,
        "elapsed_s": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, allow_nan=False)
    print("\n" + "=" * 70)
    print(f"  Saved: {args.out}")
    print(f"  Mean pLDDT: {aggregate['mean_plddt_dist']['mean']:.2f}")
    print(f"  Fraction globally confident (mean pLDDT > 70): "
          f"{aggregate['frac_globally_confident']*100:.1f}%")


if __name__ == "__main__":
    main()
