#!/usr/bin/env python3
"""Summarize the R2 attention-sink triplet subset for manuscript use.

This is R2-Add-1 from OPUS_NEXT_20260516.md. It does not re-run models; it
promotes the four M-2 attention-sink triplets into a compact evidence table.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_CHAR = REPO / "r2_interpretability_transfer/results/circuit_analysis/triplet_characterization_20260515_nperm2000/triplet_characterization.tsv"
DEFAULT_TOP = REPO / "r2_interpretability_transfer/results/circuit_analysis/triplet_characterization_20260515_nperm2000/top_firing_positions.tsv"
DEFAULT_SIG = REPO / "r2_interpretability_transfer/results/circuit_analysis/triplet_synthesis_20260515_nperm2000/triplet_signatures.tsv"
DEFAULT_MOTIFS = REPO / "r2_interpretability_transfer/results/circuit_analysis/triplet_synthesis_20260515_nperm2000/kmer_motifs.tsv"
DEFAULT_PROFILES = REPO / "r2_interpretability_transfer/results/circuit_analysis/triplet_synthesis_20260515_nperm2000/positional_profiles.tsv"
DEFAULT_OUT = REPO / "r2_interpretability_transfer/results/circuit_analysis/attention_sink_subset_20260516"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def top_motifs(rows: list[dict[str, str]], triplet_id: str, motif_type: str) -> str:
    vals = [
        f"{r['motif']}({r['count']})"
        for r in rows
        if r["triplet_id"] == triplet_id and r["motif_type"] == motif_type
    ]
    return ", ".join(vals[:3])


def source_summary(rows: list[dict[str, str]]) -> str:
    counts = Counter(r.get("source", "") for r in rows)
    return "; ".join(f"{k}:{v}" for k, v in counts.most_common())


def representative_contexts(rows: list[dict[str, str]], n: int = 8) -> str:
    seen = []
    for r in rows:
        ctx = r.get("context") or r.get("k5") or r.get("k3") or ""
        if ctx and ctx not in seen:
            seen.append(ctx)
        if len(seen) >= n:
            break
    return ", ".join(seen)


def summarize_triplet(
    tid: str,
    char: dict[str, str],
    sig: dict[str, str],
    profile: dict[str, str],
    motif_rows: list[dict[str, str]],
    top_rows: list[dict[str, str]],
) -> dict[str, Any]:
    vals = [fnum(r.get("position_norm")) for r in top_rows]
    pos = [int(fnum(r.get("position_1based"), -1)) for r in top_rows]
    n = len(top_rows)
    edge = sum(1 for r in top_rows if (r.get("k5") or "").lower() == "edge")
    nterm20 = sum(1 for x in vals if x == x and x < 0.2)
    first2 = sum(1 for x in pos if 1 <= x <= 2)
    first5 = sum(1 for x in pos if 1 <= x <= 5)
    first10 = sum(1 for x in pos if 1 <= x <= 10)
    context_starts_m = sum(1 for r in top_rows if (r.get("context") or r.get("k3") or "").startswith("M"))
    return {
        "triplet_id": tid,
        "rank": sig.get("rank", char.get("rank", "")),
        "anchor_layer": "",  # triplet ids are rank-based; layer evidence is in the atlas TSV.
        "attention_r": char.get("attention_r", sig.get("attention_r", "")),
        "attention_q": sig.get("attention_sink_q", char.get("attention_sink_q", "")),
        "positional_q": sig.get("positional_q", char.get("positional_q", "")),
        "k_mer_q": sig.get("k_mer_q", char.get("k_mer_q", "")),
        "n_top_rows": n,
        "n_unique_sequences": len({r.get("sequence_id", "") for r in top_rows}),
        "source_counts": source_summary(top_rows),
        "mean_position_norm": sum(vals) / n if n else "",
        "profile_class": profile.get("profile_class", ""),
        "first2_fraction": first2 / n if n else "",
        "first5_fraction": first5 / n if n else "",
        "first10_fraction": first10 / n if n else "",
        "nterm20_fraction": nterm20 / n if n else "",
        "edge_k5_fraction": edge / n if n else "",
        "context_starts_m_fraction": context_starts_m / n if n else "",
        "top_3mers": top_motifs(motif_rows, tid, "3mer"),
        "top_5mers": top_motifs(motif_rows, tid, "5mer"),
        "representative_contexts": representative_contexts(top_rows),
        "signature": sig.get("signature", ""),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--characterization", type=Path, default=DEFAULT_CHAR)
    ap.add_argument("--top-firing", type=Path, default=DEFAULT_TOP)
    ap.add_argument("--signatures", type=Path, default=DEFAULT_SIG)
    ap.add_argument("--motifs", type=Path, default=DEFAULT_MOTIFS)
    ap.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    char_by_id = {r["triplet_id"]: r for r in read_tsv(args.characterization)}
    sig_rows = read_tsv(args.signatures)
    sig_by_id = {r["triplet_id"]: r for r in sig_rows}
    profile_by_id = {r["triplet_id"]: r for r in read_tsv(args.profiles)}
    motif_rows = read_tsv(args.motifs)
    top_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_tsv(args.top_firing):
        top_by_id[row["triplet_id"]].append(row)

    attention_ids = [
        r["triplet_id"]
        for r in sig_rows
        if "attention-sink" in (r.get("significant_tests") or "").split(";")
    ]
    summary_rows = [
        summarize_triplet(
            tid,
            char_by_id.get(tid, {}),
            sig_by_id.get(tid, {}),
            profile_by_id.get(tid, {}),
            motif_rows,
            top_by_id.get(tid, []),
        )
        for tid in attention_ids
    ]

    fields = [
        "triplet_id",
        "rank",
        "attention_r",
        "attention_q",
        "positional_q",
        "k_mer_q",
        "n_top_rows",
        "n_unique_sequences",
        "source_counts",
        "mean_position_norm",
        "profile_class",
        "first2_fraction",
        "first5_fraction",
        "first10_fraction",
        "nterm20_fraction",
        "edge_k5_fraction",
        "context_starts_m_fraction",
        "top_3mers",
        "top_5mers",
        "representative_contexts",
        "signature",
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.out_dir / "attention_sink_subset.tsv", summary_rows, fields)
    with (args.out_dir / "summary.json").open("w") as f:
        json.dump(
            {
                "task": "R2-Add-1 attention-sink subset summary",
                "attention_sink_triplets": attention_ids,
                "n_attention_sink_triplets": len(attention_ids),
                "outputs": {
                    "table": str(args.out_dir / "attention_sink_subset.tsv"),
                    "summary_md": str(args.out_dir / "summary.md"),
                },
            },
            f,
            indent=2,
        )

    md = [
        "# Attention-Sink Triplet Subset",
        "",
        "This analysis promotes the M-2 attention-sink subset into a manuscript-facing table.",
        "It uses the final `n_perm=2000` M-1/M-2 outputs and does not re-extract model activations.",
        "",
        "## Summary",
        "",
        "| Triplet | attention r | attention q | profile | first 2 residues | first 5 residues | N-term <0.2 | edge 5-mer | top 3-mers | representative contexts |",
        "|---|---:|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for r in summary_rows:
        md.append(
            f"| {r['triplet_id']} | {fnum(r['attention_r']):.3g} | {fnum(r['attention_q']):.3g} | "
            f"{r['profile_class']} | {fnum(r['first2_fraction']):.2f} | {fnum(r['first5_fraction']):.2f} | "
            f"{fnum(r['nterm20_fraction']):.2f} | {fnum(r['edge_k5_fraction']):.2f} | "
            f"{r['top_3mers']} | {r['representative_contexts']} |"
        )
    md += [
        "",
        "## Interpretation",
        "",
        "- T011, T018 and T023 form a clear N-terminal edge subset: their saved top-firing rows are concentrated in the first residues of proteins and have edge-truncated 5-mer contexts.",
        "- T025 passes the attention-sink statistic but has a different profile: its saved top-firing rows are not N-terminal and instead concentrate on an HNC/IHNCY-like local motif.",
        "- The most defensible manuscript claim is therefore not a single universal biological primitive, but a protein-LM attention-sink family with at least one strong N-terminal-edge subtype.",
        "",
    ]
    (args.out_dir / "summary.md").write_text("\n".join(md))
    print(f"Wrote {args.out_dir / 'attention_sink_subset.tsv'}")
    print(f"Wrote {args.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
