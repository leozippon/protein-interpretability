#!/usr/bin/env python3
"""Experiment 48 (PROTOCOL §6.3/§6.4): gated high-cost dictionary retrain.

Runs ONLY if script 47 returned GO. Trains one stronger CLT on the GO target
(>=4x width, dead-feature resampling, longer schedule) via the existing
01_train_clt.py entrypoint, then leaves instructions to re-run 44/45 on the new
checkpoint and apply the §6.4 success criterion (dead<0.30, FVU<0.15, rho up
>=+0.20 on >=2 tasks). Defaults to --dry-run: it prepares the command and config
but does not launch the multi-hour job unless --execute is passed.

This is the single expensive step; per the protocol stopping rule, run it once on
the largest-gap model only. A second retrain needs a dated amendment.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
BASE_CONFIG = {
    "zymctrl": "configs/clt_training_zymctrl_v2.yaml",
    "protgpt2": "configs/clt_training_protgpt2_v2.yaml",
    "progen2-medium": "configs/clt_training.yaml",
}


def parse_args():
    base = PKG / "results/representation_audit_20260604"
    ap = argparse.ArgumentParser()
    ap.add_argument("--decision", type=Path, default=base / "decision/decision.json")
    ap.add_argument("--target", default=None, help="override the GO target model")
    ap.add_argument("--config", default=None, help="override base training config (repo-relative)")
    ap.add_argument("--width-factor", type=int, default=4, help=">=4 per PROTOCOL §6.4")
    ap.add_argument("--total-steps", type=int, default=300000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-sequences", type=int, default=300000)
    ap.add_argument("--resample-every", type=int, default=2000)
    ap.add_argument("--dead-threshold", type=int, default=2000)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--fasta-path", default=os.environ.get(
        "R2_FASTA_PATH", "/gpfs/jiaotongdamoxing/zhk_zip/data/uniref50/uniref50.fasta"))
    ap.add_argument("--save-root", default="/oss-pvc/zhk_zip/outputs")
    ap.add_argument("--out-name", default="r2_clt_recoverability_retrain_20260604")
    ap.add_argument("--execute", action="store_true", help="actually launch (default: dry-run / print command)")
    return ap.parse_args()


def base_width(config_path: Path) -> int:
    import yaml
    cfg = yaml.safe_load(config_path.read_text())
    return int(cfg.get("clt", {}).get("d_clt", 8192))


def main():
    args = parse_args()
    if not args.decision.exists():
        sys.exit(f"[48] no decision file at {args.decision}; run 47 first.")
    decision = json.loads(args.decision.read_text())["retrain_decision"]
    if decision["decision"] != "GO":
        print(f"[48] decision is {decision['decision']} ({decision['reason']}). "
              "Retrain is NOT justified; per PROTOCOL §6.3 the audit stops here.")
        return
    target = args.target or decision["retrain_target"]
    if target not in BASE_CONFIG and not args.config:
        sys.exit(f"[48] no base config mapped for target {target!r}; pass --config.")
    config_rel = args.config or BASE_CONFIG[target]
    config_path = PKG / config_rel
    new_width = base_width(config_path) * args.width_factor
    save_dir = f"{args.save_root.rstrip('/')}/{args.out_name}/clt_weights/{target}"

    overrides = [
        f"clt.d_clt={new_width}",
        f"clt.resample_every={args.resample_every}",
        f"clt.dead_feature_threshold={args.dead_threshold}",
        f"data.fasta_path={args.fasta_path}",
        f"data.num_sequences={args.num_sequences}",
        f"training.batch_size={args.batch_size}",
        f"training.total_steps={args.total_steps}",
        f"checkpoint.save_dir={save_dir}",
        f"model.name={target}",
        "logging.wandb_project=null",
    ]
    cmd = ["python3", "scripts/01_train_clt.py", "--config", config_rel,
           "--gpu", str(args.gpu), "--override", *overrides]
    final_ckpt = f"{save_dir}/step_{args.total_steps}"
    launch = {
        "target": target,
        "config": config_rel,
        "save_dir": save_dir,
        "final_checkpoint": final_ckpt,
        "width": new_width,
        "total_steps": args.total_steps,
        "batch_size": args.batch_size,
        "num_sequences": args.num_sequences,
        "fasta_path": args.fasta_path,
        "cmd": cmd,
    }
    launch_path = args.decision.parent / "retrain_launch.json"
    launch_path.write_text(json.dumps(launch, indent=2) + "\n")

    print(f"[48] GO target: {target}  (base d_clt {base_width(config_path)} -> {new_width})")
    print("[48] retrain command (run from", PKG, "):")
    print("    " + " ".join(cmd))
    print("\n[48] After training completes, re-run on the new checkpoint:")
    print(f"    python scripts/44_cache_representations.py --model-spec {target}={final_ckpt} "
          f"--out-dir results/representation_audit_20260604/cache_retrain")
    print("    python scripts/45_probe_ceiling_floor.py --cache-dir results/representation_audit_20260604/cache_retrain "
          "--out-dir results/representation_audit_20260604/probes_retrain")
    print("[48] Success (confirms H1) per §6.4: dead<0.30 AND FVU<0.15 AND rho up >=+0.20 on >=2 tasks.")
    print(f"[48] launch manifest: {launch_path}")

    if args.execute:
        print("\n[48] launching (this is the multi-hour gated job)...", flush=True)
        subprocess.run(cmd, cwd=str(PKG), check=True)
    else:
        print("\n[48] dry-run: pass --execute to launch.")


if __name__ == "__main__":
    main()
