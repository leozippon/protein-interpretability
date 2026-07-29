#!/usr/bin/env python3
"""Low-homology stratification rescue test for R1.

This bounded R1-Save-1 test uses the staged UniRef50 FASTA and extracts the
UniRef50 representative cluster size (`n=` in the header) for ClinVar proteins.
That cluster size is a homologous-coverage proxy, not a full MSA Meff value.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


REPO = Path(__file__).resolve().parents[2]
R1 = REPO / "r1_encoder_interpretability_benchmark"
DEFAULT_PRED = R1 / "results" / "variant_effect" / "alphamissense_sae_ensemble_predictions_20260511.tsv"
DEFAULT_EXTERNAL = R1 / "results" / "variant_effect" / "external_baselines_available_scores_20260507.tsv"
DEFAULT_LLR = R1 / "results" / "variant_effect" / "esm2_per_variant_llr.json"
DEFAULT_UNIREF50 = REPO / "data" / "uniref50" / "uniref50.fasta"
DEFAULT_OUT = R1 / "results" / "variant_effect" / "low_homology_stratification_20260518"


def parse_float(x: Any) -> float:
    try:
        if x in {"", "NA", None}:
            return float("nan")
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_llr(path: Path) -> dict[tuple[str, str], float]:
    rows = json.loads(path.read_text())
    return {(r["gene"].upper(), r["variant"]): -float(r["llr"]) for r in rows}


def zscore(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(arr)
    out = np.full_like(arr, np.nan, dtype=np.float64)
    if mask.sum() < 2:
        return out.tolist()
    sd = float(arr[mask].std())
    if sd <= 1e-12:
        return out.tolist()
    out[mask] = (arr[mask] - float(arr[mask].mean())) / sd
    return out.tolist()


def extract_uniref50_cluster_sizes(uniprot_ids: set[str], fasta: Path, rg_bin: str = "rg") -> dict[str, dict[str, Any]]:
    patterns = []
    for uid in sorted(uniprot_ids):
        patterns.extend([f"UniRef50_{uid}", f"UniRef50_{uid}-", f"RepID={uid}", f"RepID={uid}-"])
    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        tmp.write("\n".join(patterns) + "\n")
        pattern_path = Path(tmp.name)
    try:
        proc = subprocess.run(
            [rg_bin, "-F", "-f", str(pattern_path), str(fasta)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    finally:
        pattern_path.unlink(missing_ok=True)
    if proc.returncode not in {0, 1}:
        raise RuntimeError(f"rg failed with exit {proc.returncode}: {proc.stderr[:1000]}")

    cluster_re = re.compile(r"^>UniRef50_(\S+)")
    n_re = re.compile(r"\bn=(\d+)\b")
    rep_re = re.compile(r"\bRepID=([^\s]+)")
    by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in proc.stdout.splitlines():
        if not line.startswith(">"):
            continue
        cluster = cluster_re.search(line)
        n_match = n_re.search(line)
        rep = rep_re.search(line)
        if not cluster or not n_match:
            continue
        cluster_id = cluster.group(1)
        rep_id = rep.group(1) if rep else ""
        cluster_base = cluster_id.split("-", 1)[0]
        rep_base = rep_id.split("-", 1)[0]
        n = int(n_match.group(1))
        for uid in uniprot_ids:
            exact = cluster_id == uid or rep_id == uid
            iso = cluster_base == uid or rep_base == uid
            if exact or iso:
                by_uid[uid].append(
                    {
                        "uniprot_id": uid,
                        "uniref50_cluster": cluster_id,
                        "uniref50_rep": rep_id,
                        "cluster_n": n,
                        "match_type": "exact" if exact else "isoform_or_base",
                        "header": line,
                    }
                )
    out = {}
    for uid, hits in by_uid.items():
        hits = sorted(hits, key=lambda h: (h["match_type"] == "exact", h["cluster_n"]), reverse=True)
        out[uid] = hits[0]
    return out


def auc(y: list[int], s: list[float]) -> float:
    pairs = [(int(a), float(b)) for a, b in zip(y, s) if math.isfinite(float(b))]
    if len(pairs) < 20 or len({p[0] for p in pairs}) != 2:
        return float("nan")
    return float(roc_auc_score([p[0] for p in pairs], [p[1] for p in pairs]))


def grouped_bootstrap_auc(rows: list[dict[str, Any]], method: str, group_key: str, seed: int, n_boot: int) -> tuple[float, float]:
    groups = defaultdict(list)
    for r in rows:
        if math.isfinite(parse_float(r.get(method))):
            groups[str(r[group_key])].append(r)
    keys = sorted(groups)
    if len(keys) < 5:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        sample = [r for k in rng.choice(keys, size=len(keys), replace=True) for r in groups[k]]
        val = auc([int(r["label"]) for r in sample], [parse_float(r.get(method)) for r in sample])
        if math.isfinite(val):
            vals.append(val)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def grouped_bootstrap_delta(rows: list[dict[str, Any]], method_a: str, method_b: str, group_key: str, seed: int, n_boot: int) -> tuple[float, float]:
    groups = defaultdict(list)
    for r in rows:
        if math.isfinite(parse_float(r.get(method_a))) and math.isfinite(parse_float(r.get(method_b))):
            groups[str(r[group_key])].append(r)
    keys = sorted(groups)
    if len(keys) < 5:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        sample = [r for k in rng.choice(keys, size=len(keys), replace=True) for r in groups[k]]
        y = [int(r["label"]) for r in sample]
        va = auc(y, [parse_float(r.get(method_a)) for r in sample])
        vb = auc(y, [parse_float(r.get(method_b)) for r in sample])
        if math.isfinite(va) and math.isfinite(vb):
            vals.append(va - vb)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def quantile_cut(values: list[float], q: float) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    return float(np.quantile(vals, q)) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, default=DEFAULT_PRED)
    ap.add_argument("--external", type=Path, default=DEFAULT_EXTERNAL)
    ap.add_argument("--llr", type=Path, default=DEFAULT_LLR)
    ap.add_argument("--uniref50", type=Path, default=DEFAULT_UNIREF50)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    pred = load_tsv(args.predictions)
    external = {(r["gene"].upper(), r["variant"]): r for r in load_tsv(args.external)}
    llr_damage = load_llr(args.llr)
    uniprot_ids = {r.get("uniprot_id", "") for r in external.values() if r.get("uniprot_id")}
    cluster_hits = extract_uniref50_cluster_sizes(uniprot_ids, args.uniref50)

    rows = []
    for r in pred:
        key = (r["gene"].upper(), r["variant"])
        ext = external.get(key, {})
        uid = ext.get("uniprot_id", "")
        hit = cluster_hits.get(uid, {})
        rows.append(
            {
                "gene": r["gene"].upper(),
                "variant": r["variant"],
                "uniprot_id": uid,
                "label": int(r["label"]),
                "AlphaMissense": parse_float(r.get("am_pathogenicity")),
                "SAE_LR": parse_float(r.get("sae_lr_groupcv")),
                "ESM2_LLR_damage": llr_damage.get(key, float("nan")),
                "gMVP": parse_float(ext.get("gmvp_score")),
                "ESM1v": parse_float(ext.get("esm1v_ensemble_pathogenicity")),
                "cluster_n": parse_float(hit.get("cluster_n")),
                "cluster_match_type": hit.get("match_type", ""),
                "uniref50_cluster": hit.get("uniref50_cluster", ""),
            }
        )

    sae_z = zscore([r["SAE_LR"] for r in rows])
    llr_z = zscore([r["ESM2_LLR_damage"] for r in rows])
    for r, a, b in zip(rows, sae_z, llr_z):
        r["SAE_LLR_z"] = a + b if math.isfinite(a) and math.isfinite(b) else float("nan")

    protein_cluster = {}
    for r in rows:
        if r["uniprot_id"] and math.isfinite(r["cluster_n"]):
            protein_cluster[r["uniprot_id"]] = r["cluster_n"]
    q1 = quantile_cut(list(protein_cluster.values()), 0.25)
    q2 = quantile_cut(list(protein_cluster.values()), 0.50)
    q3 = quantile_cut(list(protein_cluster.values()), 0.75)
    for r in rows:
        n = r["cluster_n"]
        if not math.isfinite(n):
            r["homology_stratum"] = "missing"
        elif n <= q1:
            r["homology_stratum"] = "low_q1"
        elif n <= q2:
            r["homology_stratum"] = "mid_q2"
        elif n <= q3:
            r["homology_stratum"] = "mid_q3"
        else:
            r["homology_stratum"] = "high_q4"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    row_fields = [
        "gene", "variant", "uniprot_id", "label", "cluster_n", "cluster_match_type",
        "uniref50_cluster", "homology_stratum", "AlphaMissense", "SAE_LR",
        "ESM2_LLR_damage", "SAE_LLR_z", "gMVP", "ESM1v",
    ]
    with (args.out_dir / "variant_homology_scores.tsv").open("w", newline="") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=row_fields)
        w.writeheader()
        w.writerows(rows)

    methods = ["AlphaMissense", "SAE_LR", "ESM2_LLR_damage", "SAE_LLR_z", "gMVP", "ESM1v"]
    strata = ["low_q1", "mid_q2", "mid_q3", "high_q4", "missing", "all_matched"]
    summary_rows = []
    for stratum in strata:
        sub = [r for r in rows if r["homology_stratum"] == stratum] if stratum != "all_matched" else [r for r in rows if r["homology_stratum"] != "missing"]
        for m in methods:
            usable = [r for r in sub if math.isfinite(parse_float(r.get(m)))]
            val = auc([int(r["label"]) for r in usable], [parse_float(r.get(m)) for r in usable])
            lo, hi = grouped_bootstrap_auc(usable, m, "gene", args.seed + len(summary_rows), args.n_boot)
            summary_rows.append(
                {
                    "stratum": stratum,
                    "method": m,
                    "n_variants": len(usable),
                    "n_genes": len({r["gene"] for r in usable}),
                    "n_pathogenic": sum(int(r["label"]) for r in usable),
                    "auc": val,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    with (args.out_dir / "stratified_auc.tsv").open("w", newline="") as f:
        fields = list(summary_rows[0].keys())
        w = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)

    low = [r for r in rows if r["homology_stratum"] == "low_q1"]
    high = [r for r in rows if r["homology_stratum"] == "high_q4"]
    low_delta = auc([r["label"] for r in low], [r["SAE_LLR_z"] for r in low]) - auc([r["label"] for r in low], [r["AlphaMissense"] for r in low])
    high_delta = auc([r["label"] for r in high], [r["SAE_LLR_z"] for r in high]) - auc([r["label"] for r in high], [r["AlphaMissense"] for r in high])
    low_ci = grouped_bootstrap_delta(low, "SAE_LLR_z", "AlphaMissense", "gene", args.seed + 100, args.n_boot)
    high_ci = grouped_bootstrap_delta(high, "SAE_LLR_z", "AlphaMissense", "gene", args.seed + 200, args.n_boot)
    pass_gate = math.isfinite(low_delta) and low_delta >= -0.02 and math.isfinite(low_ci[0]) and low_ci[0] > -0.04
    partial_gate = math.isfinite(low_delta) and low_delta >= -0.04

    summary = {
        "task": "R1 low-homology stratification using UniRef50 cluster-size proxy",
        "status": "completed",
        "proxy_warning": "UniRef50 representative cluster size, not full MSA Meff.",
        "n_variants": len(rows),
        "n_variants_with_cluster": sum(r["homology_stratum"] != "missing" for r in rows),
        "n_proteins_with_cluster": len(protein_cluster),
        "cluster_n_quartiles": {"q1": q1, "q2": q2, "q3": q3},
        "low_q1_delta_sae_llr_minus_am": low_delta,
        "low_q1_delta_ci": low_ci,
        "high_q4_delta_sae_llr_minus_am": high_delta,
        "high_q4_delta_ci": high_ci,
        "acceptance_gate": "low_q1 SAE_LLR_z within 0.02 AUC of AM and 95% group-bootstrap CI lower bound > -0.04",
        "acceptance_pass": bool(pass_gate),
        "partial_pass": bool(partial_gate),
        "outputs": {
            "variant_homology_scores": str(args.out_dir / "variant_homology_scores.tsv"),
            "stratified_auc": str(args.out_dir / "stratified_auc.tsv"),
            "summary_md": str(args.out_dir / "summary.md"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    by = {(r["stratum"], r["method"]): r for r in summary_rows}
    md = [
        "# R1 Low-Homology Stratification",
        "",
        "This is a final rescue diagnostic for the hypothesis that SAE+LLR is more competitive with AlphaMissense in low-homology proteins.",
        "",
        "- Proxy: UniRef50 representative cluster size (`n=`), not full MSA Meff.",
        f"- Variants: {len(rows)}",
        f"- Variants with UniRef50 cluster-size proxy: {summary['n_variants_with_cluster']}",
        f"- Proteins with cluster-size proxy: {len(protein_cluster)}",
        f"- Cluster-size quartiles: q1={q1:.3g}, q2={q2:.3g}, q3={q3:.3g}",
        f"- Low-q1 SAE+LLR minus AM delta: {low_delta:.4f} [{low_ci[0]:.4f}, {low_ci[1]:.4f}]",
        f"- High-q4 SAE+LLR minus AM delta: {high_delta:.4f} [{high_ci[0]:.4f}, {high_ci[1]:.4f}]",
        f"- Acceptance gate: {'PASS' if pass_gate else ('PARTIAL' if partial_gate else 'FAIL')}",
        "",
        "## AUC by Stratum",
        "",
        "| Stratum | Method | n | genes | AUC | 95% CI |",
        "|---|---|---:|---:|---:|---|",
    ]
    for stratum in ["low_q1", "mid_q2", "mid_q3", "high_q4", "all_matched"]:
        for m in methods:
            r = by.get((stratum, m))
            if not r:
                continue

            def fmt(x):
                x = parse_float(x)
                return "nan" if not math.isfinite(x) else f"{x:.4f}"

            md.append(
                f"| {stratum} | {m} | {r['n_variants']} | {r['n_genes']} | "
                f"{fmt(r['auc'])} | [{fmt(r['ci_low'])}, {fmt(r['ci_high'])}] |"
            )
    md += [
        "",
        "## Interpretation",
        "",
        "- PASS would support a defensible low-homology applicability scope for SAE+LLR.",
        "- FAIL means the low-homology rescue does not work under this staged UniRef50 proxy.",
        "- A full DIAMOND/Meff calculation could refine the proxy, but should not be expected to reverse a large negative gap.",
        "",
    ]
    (args.out_dir / "summary.md").write_text("\n".join(md))
    print(args.out_dir / "summary.md")
    print("Gate:", "PASS" if pass_gate else ("PARTIAL" if partial_gate else "FAIL"))


if __name__ == "__main__":
    main()
