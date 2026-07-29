#!/usr/bin/env python3
"""ProteinGym/VAMP-style abundance proxy analysis for R1-Save-2.

This is a bounded CPU-only pass: it reuses the already completed R1
ProteinGym SAE benchmark and adds AlphaMissense correlations for staged
abundance / VAMP-like assays. It does not rescore SAE features; it tests
whether existing results already support the abundance-rescue hypothesis.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from scipy.stats import spearmanr


REPO = Path(__file__).resolve().parents[2]
R1 = REPO / "r1_encoder_interpretability_benchmark"
DEFAULT_DMS = REPO / "data" / "proteingym" / "DMS_ProteinGym_substitutions"
DEFAULT_FASTA = REPO / "data" / "swissprot" / "uniprot_sprot.fasta.gz"
DEFAULT_AM = REPO / "external_resources" / "baselines" / "alphamissense" / "AlphaMissense_aa_substitutions.tsv.gz"
DEFAULT_BENCH = R1 / "results" / "variant_effect" / "proteingym_benchmark_sae_latest.json"
DEFAULT_OUT_DIR = R1 / "results" / "variant_effect" / "vampseq_abundance_proxy_20260518"


DEFAULT_ASSAYS = [
    "PTEN_HUMAN_Matreyek_2021",
    "TPMT_HUMAN_Matreyek_2018",
    "MSH2_HUMAN_Jia_2020",
    "NUD15_HUMAN_Suiter_2020",
    "VKOR1_HUMAN_Chiasson_2020_abundance",
    "CP2C9_HUMAN_Amorosi_2021_abundance",
    "HXK4_HUMAN_Gersing_2023_abundance",
    "RASK_HUMAN_Weng_2022_abundance",
    "S22A1_HUMAN_Yee_2023_abundance",
]


def parse_float(x: Any) -> float:
    try:
        if x in {"", "NA", None}:
            return float("nan")
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_entry_to_accession(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with gzip.open(path, "rt", errors="replace") as f:
        for line in f:
            if not line.startswith(">"):
                continue
            # >sp|P60484|PTEN_HUMAN ...
            parts = line[1:].split("|")
            if len(parts) >= 3:
                out[parts[2].split()[0]] = parts[1]
    return out


def assay_entry_name(assay: str) -> str:
    parts = assay.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return assay


def load_benchmark(path: Path) -> dict[str, dict[str, Any]]:
    obj = json.loads(path.read_text())
    return {r["name"]: r for r in obj.get("per_assay", [])}


def load_dms_targets(assays: list[str], dms_dir: Path, entry_to_acc: dict[str, str]) -> tuple[dict[tuple[str, str], list[tuple[str, float]]], dict[str, dict[str, Any]]]:
    wanted: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    assay_meta: dict[str, dict[str, Any]] = {}
    single_re = re.compile(r"^[A-Z](\d+)[A-Z*]$")
    for assay in assays:
        path = dms_dir / f"{assay}.csv"
        entry = assay_entry_name(assay)
        acc = entry_to_acc.get(entry, "")
        meta = {
            "assay": assay,
            "file": str(path),
            "entry_name": entry,
            "uniprot_id": acc,
            "n_single_mutants": 0,
            "n_with_numeric_score": 0,
        }
        if not path.exists() or not acc:
            assay_meta[assay] = meta
            continue
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mutant = row.get("mutant", "")
                if not single_re.fullmatch(mutant):
                    continue
                meta["n_single_mutants"] += 1
                score = parse_float(row.get("DMS_score"))
                if not math.isfinite(score):
                    continue
                meta["n_with_numeric_score"] += 1
                wanted[(acc, mutant)].append((assay, score))
        assay_meta[assay] = meta
    return wanted, assay_meta


def stream_alphamissense(path: Path, wanted: dict[tuple[str, str], list[tuple[str, float]]]) -> dict[str, list[tuple[float, float]]]:
    hits: dict[str, list[tuple[float, float]]] = defaultdict(list)
    if not wanted:
        return hits
    remaining = set(wanted)
    with gzip.open(path, "rt", errors="replace") as f:
        fields = None
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            break
        if fields is None:
            return hits
        for line in f:
            values = line.rstrip("\n").split("\t")
            if len(values) != len(fields):
                continue
            row = dict(zip(fields, values))
            key = (row.get("uniprot_id", ""), row.get("protein_variant", ""))
            if key not in remaining:
                continue
            am = parse_float(row.get("am_pathogenicity"))
            if not math.isfinite(am):
                continue
            for assay, dms in wanted[key]:
                hits[assay].append((dms, am))
            remaining.discard(key)
            if not remaining:
                break
    return hits


def corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    r = spearmanr(xs, ys, nan_policy="omit").statistic
    return float(r) if r is not None else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dms-dir", type=Path, default=DEFAULT_DMS)
    ap.add_argument("--swissprot-fasta", type=Path, default=DEFAULT_FASTA)
    ap.add_argument("--alphamissense", type=Path, default=DEFAULT_AM)
    ap.add_argument("--benchmark", type=Path, default=DEFAULT_BENCH)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--assay", action="append", default=None, help="Assay basename without .csv; repeatable.")
    args = ap.parse_args()

    assays = args.assay or DEFAULT_ASSAYS
    entry_to_acc = load_entry_to_accession(args.swissprot_fasta)
    bench = load_benchmark(args.benchmark)
    wanted, assay_meta = load_dms_targets(assays, args.dms_dir, entry_to_acc)
    am_hits = stream_alphamissense(args.alphamissense, wanted)

    rows = []
    for assay in assays:
        meta = assay_meta[assay]
        pairs = am_hits.get(assay, [])
        dms = [p[0] for p in pairs]
        am = [p[1] for p in pairs]
        am_raw = corr(dms, am)
        am_abundance = corr(dms, [-x for x in am])
        b = bench.get(assay, {})
        sae_damage = parse_float(b.get("spearman_sae"))
        sae_abundance = -sae_damage if math.isfinite(sae_damage) else float("nan")
        llr = parse_float(b.get("spearman_llr"))
        ensemble = parse_float(b.get("spearman_ensemble"))
        best_sae_family = max([x for x in [sae_abundance, ensemble] if math.isfinite(x)] or [float("nan")])
        am_comparable = math.isfinite(am_abundance)
        rows.append(
            {
                **meta,
                "n_alphamissense_matched": len(pairs),
                "spearman_am_damage_raw": am_raw,
                "spearman_am_abundance_oriented": am_abundance,
                "spearman_llr": llr,
                "spearman_sae_damage_raw": sae_damage,
                "spearman_sae_abundance_oriented": sae_abundance,
                "spearman_sae_llr_ensemble": ensemble,
                "best_sae_or_ensemble_abundance": best_sae_family,
                "sae_family_beats_am": bool(am_comparable and math.isfinite(best_sae_family) and best_sae_family > am_abundance),
                "best_method": max(
                    [
                        ("AM_abundance", am_abundance),
                        ("LLR", llr),
                        ("SAE_abundance", sae_abundance),
                        ("SAE_LLR_ensemble", ensemble),
                    ],
                    key=lambda item: item[1] if math.isfinite(item[1]) else -999,
                )[0],
            }
        )

    usable = [r for r in rows if int(r["n_alphamissense_matched"]) >= 30]
    wins = [r for r in usable if r["sae_family_beats_am"]]
    pass_gate = len(wins) >= 3

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with (args.out_dir / "abundance_proxy_assay_summary.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "task": "R1-Save-2 bounded ProteinGym/VAMP abundance proxy",
        "status": "completed",
        "note": "CPU-only proxy using existing ProteinGym SAE aggregate results plus newly joined AlphaMissense scores; no new SAE rescoring was performed.",
        "n_assays_requested": len(assays),
        "n_assays_with_am_n_ge_30": len(usable),
        "n_sae_family_beats_am": len(wins),
        "acceptance_gate_proxy": "SAE damage (abundance-oriented) or SAE+LLR ensemble beats AM on >=3 usable abundance/VAMP-like assays.",
        "acceptance_pass_proxy": pass_gate,
        "outputs": {
            "assay_summary": str(args.out_dir / "abundance_proxy_assay_summary.tsv"),
            "summary_md": str(args.out_dir / "summary.md"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    md = [
        "# ProteinGym/VAMP Abundance Proxy",
        "",
        "This bounded analysis tests whether already-computed SAE ProteinGym results support the R1-Save-2 abundance rescue. It adds AlphaMissense correlations by joining staged AlphaMissense amino-acid substitutions to selected ProteinGym assays.",
        "",
        f"- Assays requested: {len(assays)}",
        f"- Assays with >=30 AlphaMissense matches: {len(usable)}",
        f"- SAE-family beats AM on usable assays: {len(wins)}",
        f"- Proxy gate: {'PASS' if pass_gate else 'FAIL'}",
        "",
        "| Assay | n AM | AM abundance rho | LLR rho | SAE abundance rho | SAE+LLR rho | best | SAE-family > AM |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        def fmt(v: Any) -> str:
            v = parse_float(v)
            return "nan" if not math.isfinite(v) else f"{v:.3f}"

        md.append(
            f"| {r['assay']} | {r['n_alphamissense_matched']} | {fmt(r['spearman_am_abundance_oriented'])} | "
            f"{fmt(r['spearman_llr'])} | {fmt(r['spearman_sae_abundance_oriented'])} | "
            f"{fmt(r['spearman_sae_llr_ensemble'])} | {r['best_method']} | {r['sae_family_beats_am']} |"
        )
    md += [
        "",
        "## Interpretation",
        "",
        "- This is not the full R1-Save-2 run because it does not rescore new VAMP datasets with SAE; it reuses the staged ProteinGym benchmark.",
        "- A FAIL here is strong evidence against spending GPU time on the same abundance hypothesis unless Opus wants a stricter complete run for completeness.",
        "",
    ]
    (args.out_dir / "summary.md").write_text("\n".join(md))
    print(f"Wrote {args.out_dir / 'summary.md'}")
    print(f"Proxy gate: {'PASS' if pass_gate else 'FAIL'}")


if __name__ == "__main__":
    main()
