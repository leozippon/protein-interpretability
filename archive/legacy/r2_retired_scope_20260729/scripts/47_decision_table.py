#!/usr/bin/env python3
"""Experiment 47 (PROTOCOL §6/§8): apply the frozen decision thresholds.

Reads the probe results from script 45 (and, if present, the oracle-steering
result from script 46) and emits the per-model substrate verdict (H1 vs H2) and
the single GO/NO-GO for the high-cost dictionary retrain (Experiment 48).
Thresholds are frozen in recoverability_audit.DECISION_THRESHOLDS / PROTOCOL §6.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import recoverability_audit as ra


def parse_args():
    base = Path(__file__).resolve().parent.parent / "results/representation_audit_20260604"
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", type=Path, default=base / "probes/probe_results.json")
    ap.add_argument("--steering", type=Path, default=base / "steering/oracle_steering.json")
    ap.add_argument("--out-dir", type=Path, default=base / "decision")
    return ap.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    probes = json.loads(args.probes.read_text())
    steering = json.loads(args.steering.read_text()) if args.steering.exists() else None

    model_verdicts, model_gaps, model_task_views = {}, {}, {}
    for model, tasks in probes["models"].items():
        view = {}
        for task, r in tasks.items():
            if r.get("status") != "ok":
                continue
            view[task] = {
                "ceiling": r["ceiling"]["skill"],
                "floor": r["floor_same_layer"]["skill"],
                "baseline": r["baseline"]["skill"],
                "rho": r["recovery_ratio"],
                "gap": r["gap"],
                "rand": r["rand"]["skill"] if r.get("rand") else -1e9,
                "ceiling_ci": r["ceiling"].get("ci95", [0, 0]),
                "baseline_ci": r["baseline"].get("ci95", [0, 0]),
            }
        if not view:
            continue
        verdict = ra.per_model_verdict(view)
        model_verdicts[model] = verdict
        # mean gap over the bottleneck tasks (drivers of GO), else over rich tasks
        drivers = verdict["bottleneck_tasks"] or verdict["rich_tasks"]
        model_gaps[model] = float(np.mean([view[t]["gap"] for t in drivers])) if drivers else 0.0
        model_task_views[model] = view

    go = ra.retrain_go_nogo(model_verdicts, model_gaps)

    # controllability verdict from script 46 (if available)
    controllability = None
    if steering:
        controllability = {m: v.get("verdict") for m, v in steering.get("models", {}).items()}

    out = {
        "thresholds": ra.DECISION_THRESHOLDS,
        "per_model_verdict": model_verdicts,
        "model_mean_gap": model_gaps,
        "retrain_decision": go,
        "controllability": controllability,
    }
    (args.out_dir / "decision.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    _write_markdown(out, model_task_views, args.out_dir / "decision.md")
    print(json.dumps({"retrain_decision": go, "verdicts": {m: {k: v[k] for k in
          ("substrate_rich", "substrate_thin", "dictionary_bottleneck", "dictionary_near_faithful")}
          for m, v in model_verdicts.items()}}, indent=2))


def _write_markdown(out, views, path):
    thr = out["thresholds"]
    lines = ["# Decision table (Experiment 47)", "",
             f"Thresholds (frozen): margin={thr['margin_macro_f1']}, rho_lo={thr['rho_lo']}, "
             f"rho_hi={thr['rho_hi']}, phi_rich={thr['phi_rich']}, min_tasks={thr['min_tasks']}.", "",
             "## Per-model verdict", "",
             "| Model | rich tasks | bottleneck tasks | substrate | dictionary | mean gap |",
             "|---|---|---|---|---|---:|"]
    for m, v in out["per_model_verdict"].items():
        sub = "RICH" if v["substrate_rich"] else ("THIN" if v["substrate_thin"] else "mixed")
        dic = ("bottleneck" if v["dictionary_bottleneck"] else
               "near-faithful" if v["dictionary_near_faithful"] else "-")
        lines.append(f"| {m} | {', '.join(v['rich_tasks']) or '-'} | {', '.join(v['bottleneck_tasks']) or '-'} | "
                     f"{sub} | {dic} | {out['model_mean_gap'].get(m, 0):.3f} |")
    go = out["retrain_decision"]
    lines += ["", "## Retrain GO/NO-GO (PROTOCOL §6.3)", "",
              f"**Decision: {go['decision']}** — {go['reason']}.",
              f"Retrain target: `{go['retrain_target']}`." if go["retrain_target"] else "No retrain.", ""]
    if out.get("controllability"):
        lines += ["## Controllability (oracle steering, §6.2)", ""]
        for m, verd in out["controllability"].items():
            lines.append(f"- {m}: {verd}")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
