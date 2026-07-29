#!/usr/bin/env python3
"""Experiment 44 (PROTOCOL §8): cache the representations for the recoverability audit.

For each protein generator x layer x cohort, caches the mean-pooled
R_raw (residual-stream ceiling), R_code (CLT sparse-code floor) and R_recon (CLT
reconstruction), plus the ESM-2 reference and the composition baseline. Optionally
caches per-residue R_raw/R_code for the residue-SS task (T4) and the ZymCTRL
decoder-native EC cohort (T5).

Outputs land in --out-dir (default results/representation_audit_20260604/cache/).
Run from the repo root with the `ct` env. Generator base models are resolved via
R2_MODEL_BASE_DIR (default /Data/public/models_R2 on this machine).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def parse_args():
    here = Path(__file__).resolve().parent
    pkg = here.parent
    repo = pkg.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-spec", action="append", default=None,
                    help="model=clt_checkpoint_dir; repeatable. Defaults to the local rerun CLTs.")
    ap.add_argument("--model-base-dir", default=os.environ.get("R2_MODEL_BASE_DIR", "/Data/public/models_R2"))
    ap.add_argument("--esm2-model", default="/Data/public/esm2_t36_3B_UR50D")
    ap.add_argument("--swissprot-cache", type=Path, default=repo / "data/processed/swissprot_all_max1022.pkl")
    ap.add_argument("--pfam-residue", type=Path, default=repo / "data/interpro/pfam_residue.tsv")
    ap.add_argument("--ec-fasta", type=Path, default=repo / "data/zymctrl/ec_labeled_swissprot.fasta")
    ap.add_argument("--goa-gaf", type=Path, default=repo / "data/go/goa_uniprot_all.gaf.gz")
    ap.add_argument("--go-obo", type=Path, default=repo / "data/go/go-basic.obo")
    ap.add_argument("--decoder-ec-json", type=Path,
                    default=pkg / "results/steering_benchmark/zymctrl_v2_onmanifold_direct_20260503.json")
    ap.add_argument("--min-len", type=int, default=100)
    ap.add_argument("--max-len", type=int, default=400)
    ap.add_argument("--pfam-classes", type=int, default=20)
    ap.add_argument("--pfam-per-class", type=int, default=12)
    ap.add_argument("--ec-per-class", type=int, default=40)
    ap.add_argument("--ss-n", type=int, default=300)
    ap.add_argument("--decoder-per-class", type=int, default=60)
    ap.add_argument("--layers", default="all", help="'all' or comma-separated layer indices")
    ap.add_argument("--residue-layers", default="even6", help="'all','even6','none' or comma list")
    ap.add_argument("--esm-batch-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260604)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-sequences", type=int, default=0, help="smoke-test: cap protein cohort size")
    ap.add_argument("--out-dir", type=Path, default=pkg / "results/representation_audit_20260604/cache")
    return ap.parse_args()


def pick_layers(spec: str, n_layers: int):
    if spec == "all":
        return list(range(n_layers))
    if spec == "none":
        return []
    if spec == "even6":
        return sorted(set(int(round(x)) for x in np.linspace(0, n_layers - 1, 6)))
    return [int(x) for x in spec.split(",") if x.strip() != ""]


def main():
    args = parse_args()
    os.environ["R2_MODEL_BASE_DIR"] = args.model_base_dir  # before importing the model loader
    import recoverability_audit as ra
    from src.models.model_loader import load_model
    from src.analysis.circuit_discovery import load_trained_clt
    import torch

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    specs = [ra.U29.parse_model_spec(s) for s in (args.model_spec or ra.DEFAULT_MODEL_SPECS)]

    # ---- cohort + labels (T1-T4) ----
    print("[44] building Swiss-Prot cohort", flush=True)
    records, meta = ra.build_cohort(args)
    if args.limit_sequences:
        records = records[: args.limit_sequences]
        meta["limited_to"] = len(records)
    (args.out_dir / "cohort.json").write_text(json.dumps({"meta": meta, "records": records}, indent=2) + "\n")
    print(f"  cohort: {len(records)} proteins", flush=True)

    ss_by_acc = ra.residue_ss_labels(records, args.swissprot_cache)
    print(f"  residue-SS proteins: {len(ss_by_acc)}", flush=True)

    # ---- composition baseline + ESM-2 reference (protein-level) ----
    np.save(args.out_dir / "ngram.npy", ra.ngram_baseline(records))
    print("[44] ESM-2 reference embeddings", flush=True)
    esm2 = ra.S34.esm2_matrix(records, Path(args.esm2_model), args.device, args.max_len, args.esm_batch_size)
    np.save(args.out_dir / "esm2.npy", esm2)

    # ---- decoder-native EC cohort (T5) ----
    decoder_records = ra.load_decoder_ec_cohort(args.decoder_ec_json, args.decoder_per_class, args.min_len, args.max_len)
    if decoder_records:
        (args.out_dir / "decoder_cohort.json").write_text(json.dumps(decoder_records, indent=2) + "\n")
    print(f"[44] decoder-native EC cohort: {len(decoder_records)} sequences", flush=True)

    manifest = {"seed": args.seed, "n_proteins": len(records), "models": {},
                "esm2_model": args.esm2_model, "decoder_ec_n": len(decoder_records),
                "residue_ss_proteins": len(ss_by_acc)}

    for model_name, ckpt in specs:
        print(f"\n[44] === {model_name} ({ckpt}) ===", flush=True)
        pm = load_model(model_name, device=args.device)
        clt = load_trained_clt(ckpt, device=args.device)
        layers = pick_layers(args.layers, pm.n_layers)
        print(f"  layers: {layers}", flush=True)

        reps = ra.extract_protein_matrices(pm, clt, records, layers, args.device, recon=True)
        np.savez(args.out_dir / f"reps_{model_name}.npz",
                 **{f"raw_L{l}": reps["R_raw"][l] for l in layers},
                 **{f"code_L{l}": reps["R_code"][l] for l in layers},
                 **{f"recon_L{l}": reps["R_recon"][l] for l in layers})

        rlayers = pick_layers(args.residue_layers, pm.n_layers)
        res_info = {"layers": []}
        if rlayers and ss_by_acc:
            rres, ry, rg, rb = ra.extract_residue_matrices(pm, clt, records, ss_by_acc, rlayers, args.device)
            np.savez(args.out_dir / f"residue_{model_name}.npz",
                     y=ry.astype(str), groups=rg, baseline=rb,
                     **{f"raw_L{l}": rres["R_raw"][l] for l in rlayers},
                     **{f"code_L{l}": rres["R_code"][l] for l in rlayers})
            res_info = {"layers": rlayers, "n_residues": int(len(ry))}

        dec_info = None
        if model_name == "zymctrl" and decoder_records:
            dreps = ra.extract_protein_matrices(pm, clt, decoder_records, layers, args.device, recon=False)
            np.savez(args.out_dir / f"decoder_{model_name}.npz",
                     labels=np.array([r["decoder_ec"] for r in decoder_records]),
                     baseline=ra.ngram_baseline(decoder_records),
                     **{f"raw_L{l}": dreps["R_raw"][l] for l in layers},
                     **{f"code_L{l}": dreps["R_code"][l] for l in layers})
            dec_info = {"n": len(decoder_records), "layers": layers}

        manifest["models"][model_name] = {
            "checkpoint": ckpt, "n_layers": int(pm.n_layers), "d_model": int(pm.d_model),
            "d_clt": int(clt.d_clt), "k": int(clt.k), "layers": layers,
            "residue": res_info, "decoder_ec": dec_info,
        }
        del pm, clt
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    manifest["runtime_seconds"] = time.time() - t0
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n[44] done in {time.time()-t0:.1f}s -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
