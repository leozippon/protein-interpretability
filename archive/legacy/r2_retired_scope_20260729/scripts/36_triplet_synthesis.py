#!/usr/bin/env python3
"""Synthesize M-1 triplet characterization into multi-test signatures.

This is M-2 from OPUS_NEXT_20260515.md. It does not run models; it only
post-processes the M-1 TSV outputs into manuscript-facing tables.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TESTS = [
    ("k-mer", "k_mer_q"),
    ("positional", "positional_q"),
    ("bpe-boundary", "bpe_boundary_q"),
    ("attention-sink", "attention_sink_q"),
    ("high-norm", "high_norm_q"),
]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_motif_json(value: str) -> list[list[Any]]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    out: list[list[Any]] = []
    for item in parsed:
        if isinstance(item, list) and len(item) >= 2:
            out.append(item)
    return out


def significant_tests(row: dict[str, str], q_threshold: float) -> list[str]:
    sig = []
    for name, col in TESTS:
        if fnum(row.get(col), 1.0) < q_threshold:
            sig.append(name)
    return sig


def cluster_signature(sig: list[str]) -> str:
    s = set(sig)
    if not s:
        return "unknown"
    if "attention-sink" in s:
        return "attention-sink subset"
    if {"k-mer", "positional", "high-norm"}.issubset(s):
        return "multi-entangled kmer+positional+high-norm"
    if s == {"k-mer"}:
        return "kmer-only"
    if "k-mer" in s:
        return "kmer-dominant mixed"
    if {"positional", "high-norm"}.issubset(s):
        return "positional+high-norm"
    if s == {"positional"}:
        return "positional-only"
    if s == {"high-norm"}:
        return "high-norm-only"
    if "bpe-boundary" in s:
        return "bpe-boundary subset"
    return "other-mixed"


def position_bucket(x: float) -> str:
    if x < 0.2:
        return "0.0-0.2"
    if x < 0.4:
        return "0.2-0.4"
    if x < 0.6:
        return "0.4-0.6"
    if x < 0.8:
        return "0.6-0.8"
    return "0.8-1.0"


def classify_position_profile(counts: Counter[str], total: int) -> str:
    if total == 0:
        return "no-top-position-rows"
    n_term = counts["0.0-0.2"] / total
    center = counts["0.4-0.6"] / total
    c_term = counts["0.8-1.0"] / total
    if n_term >= 0.5:
        return "N-term-enriched"
    if c_term >= 0.5:
        return "C-term-enriched"
    if center >= 0.5:
        return "centre"
    if n_term >= 0.25 and c_term >= 0.25:
        return "bimodal"
    return "non-trivial"


def make_overlap(rows: list[dict[str, str]], q_threshold: float) -> list[dict[str, Any]]:
    sig_sets = {
        name: {row["triplet_id"] for row in rows if fnum(row.get(col), 1.0) < q_threshold}
        for name, col in TESTS
    }
    out = []
    for a, _ in TESTS:
        for b, _ in TESTS:
            sa = sig_sets[a]
            sb = sig_sets[b]
            both = len(sa & sb)
            union = len(sa | sb)
            out.append(
                {
                    "test_a": a,
                    "test_b": b,
                    "n_a": len(sa),
                    "n_b": len(sb),
                    "n_both": both,
                    "n_union": union,
                    "jaccard": both / union if union else 0.0,
                }
            )
    return out


def make_triplet_signatures(rows: list[dict[str, str]], q_threshold: float) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        sig = significant_tests(row, q_threshold)
        out.append(
            {
                "triplet_id": row["triplet_id"],
                "rank": row.get("rank", ""),
                "assigned_category": row.get("assigned_category", ""),
                "n_significant": len(sig),
                "significant_tests": ";".join(sig) if sig else "none",
                "signature": "+".join(sig) if sig else "none",
                "cluster": cluster_signature(sig),
                "k_mer_q": row.get("k_mer_q", ""),
                "positional_q": row.get("positional_q", ""),
                "bpe_boundary_q": row.get("bpe_boundary_q", ""),
                "attention_sink_q": row.get("attention_sink_q", ""),
                "high_norm_q": row.get("high_norm_q", ""),
                "attention_r": row.get("attention_r", ""),
                "hidden_norm_r": row.get("hidden_norm_r", ""),
                "kmer_best_k": row.get("kmer_best_k", ""),
                "kmer_mi_nats": row.get("kmer_mi_nats", ""),
                "positional_ks": row.get("positional_ks", ""),
            }
        )
    return out


def make_kmer_motifs(rows: list[dict[str, str]], q_threshold: float) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if fnum(row.get("k_mer_q"), 1.0) >= q_threshold:
            continue
        sig = significant_tests(row, q_threshold)
        for motif_type, col in [("3mer", "top_k3"), ("5mer", "top_k5")]:
            for idx, item in enumerate(parse_motif_json(row.get(col, ""))[:3], start=1):
                out.append(
                    {
                        "triplet_id": row["triplet_id"],
                        "motif_type": motif_type,
                        "motif_rank": idx,
                        "motif": item[0],
                        "count": item[1],
                        "k_mer_q": row.get("k_mer_q", ""),
                        "signature": "+".join(sig) if sig else "none",
                    }
                )
    return out


def make_position_profiles(
    rows: list[dict[str, str]],
    top_rows: list[dict[str, str]],
    q_threshold: float,
) -> list[dict[str, Any]]:
    by_triplet: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in top_rows:
        by_triplet[row["triplet_id"]].append(row)

    out = []
    for row in rows:
        if fnum(row.get("positional_q"), 1.0) >= q_threshold:
            continue
        counts: Counter[str] = Counter()
        vals = []
        for pos in by_triplet.get(row["triplet_id"], []):
            x = fnum(pos.get("position_norm"))
            if x == x:
                vals.append(x)
                counts[position_bucket(x)] += 1
        total = len(vals)
        out.append(
            {
                "triplet_id": row["triplet_id"],
                "n_top_position_rows": total,
                "profile_class": classify_position_profile(counts, total),
                "mean_position_norm": sum(vals) / total if total else "",
                "bucket_0_0_0_2": counts["0.0-0.2"],
                "bucket_0_2_0_4": counts["0.2-0.4"],
                "bucket_0_4_0_6": counts["0.4-0.6"],
                "bucket_0_6_0_8": counts["0.6-0.8"],
                "bucket_0_8_1_0": counts["0.8-1.0"],
                "positional_q": row.get("positional_q", ""),
                "signature": "+".join(significant_tests(row, q_threshold)) or "none",
            }
        )
    return out


def fmt_float(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def write_summary(
    path: Path,
    args: argparse.Namespace,
    rows: list[dict[str, str]],
    signatures: list[dict[str, Any]],
    overlap: list[dict[str, Any]],
    motifs: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
) -> dict[str, Any]:
    n_triplets = len(rows)
    sig_counts = {name: sum(fnum(row.get(col), 1.0) < args.q_threshold for row in rows) for name, col in TESTS}
    n_multi3 = sum(int(s["n_significant"]) >= 3 for s in signatures)
    n_any = sum(int(s["n_significant"]) >= 1 for s in signatures)
    signature_counts = Counter(s["signature"] for s in signatures)
    cluster_counts = Counter(s["cluster"] for s in signatures)
    attention = [s for s in signatures if "attention-sink" in str(s["significant_tests"]).split(";")]

    overlap_lookup = {(r["test_a"], r["test_b"]): r for r in overlap}
    key_pairs = [("k-mer", "positional"), ("k-mer", "high-norm"), ("positional", "high-norm")]

    motif_examples = []
    motif_by_triplet: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for m in motifs:
        motif_by_triplet[m["triplet_id"]].append(m)
    for tid in sorted(motif_by_triplet)[:10]:
        ms = motif_by_triplet[tid]
        motif_examples.append(
            {
                "triplet_id": tid,
                "motifs": ", ".join(f"{m['motif_type']}:{m['motif']}({m['count']})" for m in ms[:4]),
            }
        )

    profile_counts = Counter(p["profile_class"] for p in profiles)
    max_top_rows = max((int(p["n_top_position_rows"]) for p in profiles), default=0)
    min_top_rows = min((int(p["n_top_position_rows"]) for p in profiles), default=0)

    md = []
    md.append("# M-2 Triplet Characterization Synthesis")
    md.append("")
    md.append(f"- Characterization input: `{args.characterization}`")
    md.append(f"- Top firing positions input: `{args.top_positions}`")
    md.append(f"- Triplets: {n_triplets}")
    md.append(f"- q threshold: {args.q_threshold}")
    md.append(f"- Triplets significant on >=1 test: {n_any} / {n_triplets}")
    md.append(f"- Triplets significant on >=3 tests: {n_multi3} / {n_triplets}")
    md.append(f"- Top-position rows available per positional triplet: {min_top_rows}-{max_top_rows}")
    md.append("")
    md.append("## Per-Test Significant Counts")
    md.append("")
    md.append("| Test | Significant triplets |")
    md.append("|---|---:|")
    for name, _ in TESTS:
        md.append(f"| {name} | {sig_counts[name]} |")
    md.append("")
    md.append("## Key Cross-Test Overlaps")
    md.append("")
    md.append("| Test pair | Both significant | Jaccard |")
    md.append("|---|---:|---:|")
    for a, b in key_pairs:
        rec = overlap_lookup[(a, b)]
        md.append(f"| {a} + {b} | {rec['n_both']} | {rec['jaccard']:.3f} |")
    md.append("")
    md.append("## Signature Clusters")
    md.append("")
    md.append("| Cluster | Count |")
    md.append("|---|---:|")
    for cluster, count in cluster_counts.most_common():
        md.append(f"| {cluster} | {count} |")
    md.append("")
    md.append("## Full Signatures")
    md.append("")
    md.append("| Signature | Count |")
    md.append("|---|---:|")
    for sig, count in signature_counts.most_common():
        md.append(f"| {sig} | {count} |")
    md.append("")
    md.append("## Attention-Sink Subset")
    md.append("")
    if attention:
        md.append("| Triplet | Attention r | Attention q | Full signature |")
        md.append("|---|---:|---:|---|")
        for s in attention:
            md.append(
                f"| {s['triplet_id']} | {fmt_float(s['attention_r'])} | "
                f"{fmt_float(s['attention_sink_q'])} | {s['signature']} |"
            )
    else:
        md.append("No triplets pass the attention-sink test at the selected q threshold.")
    md.append("")
    md.append("## Motif Examples")
    md.append("")
    md.append("| Triplet | Top motif entries |")
    md.append("|---|---|")
    for ex in motif_examples:
        md.append(f"| {ex['triplet_id']} | {ex['motifs']} |")
    md.append("")
    md.append("## Positional Profiles")
    md.append("")
    md.append("| Profile class | Count |")
    md.append("|---|---:|")
    for profile, count in profile_counts.most_common():
        md.append(f"| {profile} | {count} |")
    md.append("")
    md.append("## Manuscript Interpretation")
    md.append("")
    md.append(
        "The conserved triplets should not be described as mutually exclusive "
        "k-mer, positional, attention, BPE, or high-norm categories. Most "
        "significant triplets are significant on multiple low-level tests, so "
        "the previous single assigned category is an order-dependent summary."
    )
    md.append("")
    md.append(
        "The more defensible interpretation is that conserved cross-model CLT "
        "triplets capture entangled local sequence context, normalized "
        "position-in-protein, and residual-stream magnitude. A smaller subset "
        "also shows strong attention-sink behavior. This supports the R2 "
        "framing of statistical conservation without biological convergence."
    )
    md.append("")
    md.append("## Evidence Boundary")
    md.append("")
    md.append(
        "- This synthesis is a post-processing analysis of M-1 outputs; it does "
        "not re-extract activations or attentions."
    )
    md.append(
        "- Positional profile classes use the saved top-firing rows, not the "
        "full activation arrays. If M-1 is rerun with more saved rows, rerun "
        "this script for the final figure tables."
    )
    md.append(
        "- The high-norm test is not independent of CLT activation because both "
        "are derived from the same residual vector; it should be reported as a "
        "diagnostic covariate, not as a separate biological mechanism."
    )
    path.write_text("\n".join(md) + "\n")

    return {
        "n_triplets": n_triplets,
        "q_threshold": args.q_threshold,
        "n_any_significant": n_any,
        "n_significant_ge_3_tests": n_multi3,
        "per_test_significant_counts": sig_counts,
        "signature_counts": dict(signature_counts),
        "cluster_counts": dict(cluster_counts),
        "attention_sink_triplets": [s["triplet_id"] for s in attention],
        "position_profile_counts": dict(profile_counts),
        "top_position_rows_range": [min_top_rows, max_top_rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--characterization",
        type=Path,
        default=Path("r2_interpretability_transfer/results/circuit_analysis/triplet_characterization_20260514/triplet_characterization.tsv"),
    )
    parser.add_argument(
        "--top-positions",
        type=Path,
        default=Path("r2_interpretability_transfer/results/circuit_analysis/triplet_characterization_20260514/top_firing_positions.tsv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("r2_interpretability_transfer/results/circuit_analysis/triplet_synthesis_20260515"),
    )
    parser.add_argument("--q-threshold", type=float, default=0.05)
    args = parser.parse_args()

    rows = read_tsv(args.characterization)
    top_rows = read_tsv(args.top_positions)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    overlap = make_overlap(rows, args.q_threshold)
    signatures = make_triplet_signatures(rows, args.q_threshold)
    motifs = make_kmer_motifs(rows, args.q_threshold)
    profiles = make_position_profiles(rows, top_rows, args.q_threshold)

    write_tsv(
        args.out_dir / "cross_test_overlap.tsv",
        overlap,
        ["test_a", "test_b", "n_a", "n_b", "n_both", "n_union", "jaccard"],
    )
    write_tsv(
        args.out_dir / "triplet_signatures.tsv",
        signatures,
        [
            "triplet_id",
            "rank",
            "assigned_category",
            "n_significant",
            "significant_tests",
            "signature",
            "cluster",
            "k_mer_q",
            "positional_q",
            "bpe_boundary_q",
            "attention_sink_q",
            "high_norm_q",
            "attention_r",
            "hidden_norm_r",
            "kmer_best_k",
            "kmer_mi_nats",
            "positional_ks",
        ],
    )
    write_tsv(
        args.out_dir / "kmer_motifs.tsv",
        motifs,
        ["triplet_id", "motif_type", "motif_rank", "motif", "count", "k_mer_q", "signature"],
    )
    write_tsv(
        args.out_dir / "positional_profiles.tsv",
        profiles,
        [
            "triplet_id",
            "n_top_position_rows",
            "profile_class",
            "mean_position_norm",
            "bucket_0_0_0_2",
            "bucket_0_2_0_4",
            "bucket_0_4_0_6",
            "bucket_0_6_0_8",
            "bucket_0_8_1_0",
            "positional_q",
            "signature",
        ],
    )
    summary = write_summary(args.out_dir / "summary.md", args, rows, signatures, overlap, motifs, profiles)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
