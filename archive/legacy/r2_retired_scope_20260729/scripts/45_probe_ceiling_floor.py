#!/usr/bin/env python3
"""Experiment 45 (PROTOCOL §5/§8): ceiling / floor / gap probes.

Loads the caches from script 44 and, for each model x task x representation,
runs linear probes with grouped (family-disjoint) CV, computes chance-corrected
*skill*, the per-layer profile, and the headline ceiling C (R_raw), floor F
(R_code), gap = C-F, recovery rho = F/C, baseline (B_ngram / chance), matched-dim
control (R_rand) and reference fraction phi = C / skill(ESM2). CPU-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import recoverability_audit as ra

CLS_TASKS = {"ec_topclass", "pfam_family", "residue_ss", "decoder_ec"}
REG_TASKS = {"secondary_fraction"}


def parse_args():
    pkg = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=pkg / "results/representation_audit_20260604/cache")
    ap.add_argument("--out-dir", type=Path, default=pkg / "results/representation_audit_20260604/probes")
    ap.add_argument("--seed", type=int, default=20260604)
    ap.add_argument("--n-boot", type=int, default=1000, help="bootstrap reps for headline CIs")
    ap.add_argument("--tasks", default=",".join(ra.PRIMARY_TASKS))
    ap.add_argument("--pca-dim", type=int, default=256,
                    help="PCA cap (in-pipeline) for every probe input; 0 disables. "
                         "Matches ceiling/floor dimensionality and conditions the regression.")
    ap.add_argument("--allow-cv-fallback", action="store_true",
                    help="Allow non-grouped CV if grouped CV is impossible. Do not use for headline runs.")
    return ap.parse_args()


def score_rep(X, kind, target, groups, seed, n_boot, C=1.0, allow_fallback=False, pca_dim=None):
    if X is None or X.shape[0] == 0 or X.shape[1] == 0:
        return None
    if kind == "classification":
        yi, pred, _ = ra.cv_predict_classification(X, target, groups, seed, C=C,
                                                   allow_fallback=allow_fallback, pca_dim=pca_dim)
        return ra.score_classification(yi, pred, seed, n_boot=n_boot, groups=groups)
    pred = ra.cv_predict_regression(X, target, groups, seed, alpha=1.0 / max(C, 1e-8),
                                    allow_fallback=allow_fallback, pca_dim=pca_dim)
    return ra.score_regression(target, pred, seed, n_boot=max(n_boot, 1), groups=groups)


def protein_task_inputs(task, records, cache, model):
    """Return (idx, kind, target, groups, layers, get_raw, get_code, get_recon, ngram, esm2)."""
    # `task` maps to a label column; the CV variant is set by the grouping below.
    label_task = "ec_topclass" if task == "ec_topclass_stratified" else task
    idx = [i for i, r in enumerate(records) if r["tasks"].get(label_task)]
    if label_task == "ec_topclass":
        target = np.array([records[i]["ec_topclass"] for i in idx], dtype=object)
        kind = "classification"
    elif task == "pfam_family":
        target = np.array([records[i]["dominant_pfam"] for i in idx], dtype=object)
        kind = "classification"
    elif task == "secondary_fraction":
        # Drop the near-constant turn fraction (mean ~0.03); regress (helix, strand).
        target = np.array([records[i]["secondary_fraction"][:2] for i in idx], dtype=np.float32)
        kind = "regression"
    else:
        raise ValueError(task)
    if task in ("pfam_family", "ec_topclass_stratified"):
        # Stratified CV. For pfam_family the label IS the family, so family-disjoint
        # grouping would hold out every test class (macro-F1 -> 0); for the
        # stratified EC variant we deliberately let family leak to quantify the
        # confound. Family-disjoint grouping is the homology control only where
        # the family is a *confound* (ec_topclass, secondary_fraction).
        groups = np.array([f"__solo_{i}" for i in idx], dtype=object)
    else:
        groups = ra._make_groups(records, idx)
    reps = np.load(cache / f"reps_{model}.npz")
    layers = sorted(int(k[5:]) for k in reps.files if k.startswith("raw_L"))
    ngram = np.load(cache / "ngram.npy")[idx]
    esm2 = np.load(cache / "esm2.npy")[idx]
    return {"idx": idx, "kind": kind, "target": target, "groups": groups, "layers": layers,
            "reps": reps, "ngram": ngram, "esm2": esm2}


def run_model_task(task, records, cache, model, seed, n_boot, allow_fallback=False, pca_dim=None):
    # ---- assemble representations + targets per task type ----
    if task in {"ec_topclass", "ec_topclass_stratified", "pfam_family", "secondary_fraction"}:
        P = protein_task_inputs(task, records, cache, model)
        if len(P["idx"]) < 10:
            return {"task": task, "status": "skipped", "reason": f"n={len(P['idx'])}"}
        kind, target, groups, layers, reps = P["kind"], P["target"], P["groups"], P["layers"], P["reps"]
        get = lambda kind_, l: reps[f"{kind_}_L{l}"][P["idx"]]
        ngram, esm2 = P["ngram"], P["esm2"]
        has_recon = True
    elif task == "residue_ss":
        f = cache / f"residue_{model}.npz"
        if not f.exists():
            return {"task": task, "status": "skipped", "reason": "no residue cache"}
        d = np.load(f, allow_pickle=True)
        target = d["y"].astype(object)
        groups = d["groups"]
        layers = sorted(int(k[5:]) for k in d.files if k.startswith("raw_L"))
        if len(target) < 50:
            return {"task": task, "status": "skipped", "reason": f"n_res={len(target)}"}
        kind = "classification"
        get = lambda kind_, l: d[f"{'raw' if kind_=='raw' else 'code'}_L{l}"]
        ngram = d["baseline"] if "baseline" in d.files else None
        esm2 = None
        has_recon = False
    elif task == "decoder_ec":
        f = cache / f"decoder_{model}.npz"
        if not f.exists():
            return {"task": task, "status": "skipped", "reason": "decoder cohort only for zymctrl"}
        d = np.load(f, allow_pickle=True)
        target = d["labels"].astype(object)
        groups = np.array([f"__solo_{i}" for i in range(len(target))], dtype=object)
        layers = sorted(int(k[5:]) for k in d.files if k.startswith("raw_L"))
        kind = "classification"
        get = lambda kind_, l: d[f"{'raw' if kind_=='raw' else 'code'}_L{l}"]
        ngram = d["baseline"] if "baseline" in d.files else None
        esm2 = None
        has_recon = False
    else:
        raise ValueError(task)

    sr = lambda X, C, nb: score_rep(X, kind, target, groups, seed, nb, C=C,
                                    allow_fallback=allow_fallback, pca_dim=pca_dim)

    # ---- per-layer skill profile (fast: no bootstrap) ----
    prof = {"ceiling": {}, "floor": {}, "recon": {}}
    for l in layers:
        c = sr(get("raw", l), 1.0, 0)
        prof["ceiling"][l] = c["skill"] if c else float("nan")
        Xc = get("code", l)
        alive = ra.alive_columns(Xc)
        fsc = sr(Xc[:, alive] if alive.size else Xc, 0.5, 0)
        prof["floor"][l] = fsc["skill"] if fsc else float("nan")
        if has_recon:
            r = sr(get("recon", l), 1.0, 0)
            prof["recon"][l] = r["skill"] if r else float("nan")

    best_c = max(layers, key=lambda l: (prof["ceiling"][l] if not np.isnan(prof["ceiling"][l]) else -9))
    best_f = max(layers, key=lambda l: (prof["floor"][l] if not np.isnan(prof["floor"][l]) else -9))

    # ---- headline (best ceiling layer) with bootstrap CIs ----
    ceiling = sr(get("raw", best_c), 1.0, n_boot)
    Xc = get("code", best_c)
    alive = ra.alive_columns(Xc)
    floor_same = sr(Xc[:, alive] if alive.size else Xc, 0.5, n_boot)
    # R_rand: report-only diagnostic at a small fixed dim (the gate was dropped).
    Xraw = get("raw", best_c)
    rand = sr(ra.random_projection(Xraw, min(256, Xraw.shape[1]), seed), 1.0, 0)
    base = (sr(ngram, 1.0, n_boot) if ngram is not None
            else {"skill": 0.0, "ci95": [0.0, 0.0], "metric": 0.0, "chance": 0.0})  # chance baseline for residue/decoder
    esm = sr(esm2, 1.0, 0) if esm2 is not None else None

    C = ceiling["skill"]
    F = floor_same["skill"] if floor_same else float("nan")
    rec = ra.recoverability(C, F)
    phi = (C / esm["skill"]) if (esm and esm["skill"] > 1e-6) else float("nan")

    return {
        "task": task, "status": "ok", "n": ceiling["n"], "metric_name": ceiling["metric_name"],
        "best_ceiling_layer": best_c, "best_floor_layer": best_f, "n_alive_features": int(alive.size),
        "ceiling": ceiling, "floor_same_layer": floor_same,
        "floor_best_layer_skill": prof["floor"][best_f],
        "recon_skill": prof["recon"].get(best_c) if has_recon else None,
        "baseline": base, "rand": rand, "esm2": esm,
        "gap": rec["gap"], "recovery_ratio": rec["recovery_ratio"], "phi_ref_fraction": phi,
        "layer_profile": {k: {int(l): v for l, v in prof[k].items()} for k in prof},
    }


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cohort = json.loads((args.cache_dir / "cohort.json").read_text())
    records = cohort["records"]
    manifest = json.loads((args.cache_dir / "manifest.json").read_text())
    tasks = [t for t in args.tasks.split(",") if t.strip()]
    pca_dim = args.pca_dim if args.pca_dim and args.pca_dim > 0 else None

    results = {"models": {}, "thresholds": ra.DECISION_THRESHOLDS, "pca_dim": pca_dim}
    for model in manifest["models"]:
        print(f"[45] === {model} ===", flush=True)
        results["models"][model] = {}
        for task in tasks:
            print(f"  - {task}", flush=True)
            try:
                res = run_model_task(task, records, args.cache_dir, model, args.seed,
                                     args.n_boot, args.allow_cv_fallback, pca_dim)
            except Exception as exc:
                res = {"task": task, "status": "failed", "reason": str(exc)}
            results["models"][model][task] = res
            if res.get("status") == "ok":
                print(f"      C={res['ceiling']['skill']:.3f} F={res['floor_same_layer']['skill']:.3f}"
                      f" gap={res['gap']:.3f} rho={res['recovery_ratio']:.3f}"
                      f" base={res['baseline']['skill']:.3f} phi={res['phi_ref_fraction']}", flush=True)

    (args.out_dir / "probe_results.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
    _write_markdown(results, args.out_dir / "probe_results.md")
    print(f"[45] done -> {args.out_dir}", flush=True)


def _write_markdown(results, path):
    lines = ["# Recoverability probes (Experiment 45)", "",
             "Skill = metric - chance. C = ceiling (R_raw), F = floor (R_code, same layer),",
             "gap = C-F, rho = F/C, base = composition/chance baseline, phi = C/ESM2.", "",
             "| Model | Task | metric | C | F | gap | rho | base | phi | n |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model, tasks in results["models"].items():
        for task, r in tasks.items():
            if r.get("status") != "ok":
                lines.append(f"| {model} | {task} | _{r.get('status')}: {r.get('reason','')}_ | | | | | | | |")
                continue
            phi = r["phi_ref_fraction"]
            lines.append(f"| {model} | {task} | {r['metric_name']} | {r['ceiling']['skill']:.3f} | "
                         f"{r['floor_same_layer']['skill']:.3f} | {r['gap']:.3f} | {r['recovery_ratio']:.3f} | "
                         f"{r['baseline']['skill']:.3f} | {phi if isinstance(phi,str) else f'{phi:.2f}'} | {r['n']} |")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
