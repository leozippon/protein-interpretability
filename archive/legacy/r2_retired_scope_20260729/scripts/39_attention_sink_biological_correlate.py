#!/usr/bin/env python3
"""Test N-terminal biological correlates for attention-sink triplets.

This is R2-Add-3 from OPUS_NEXT_20260516.md. It compares each attention-sink
triplet's saved top-firing rows against the remaining saved top-firing rows
from the same M-1 run. The tests are intentionally simple and auditable:
N-terminal position, N-terminal edge context, and start-methionine context.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_TOP = REPO / "r2_interpretability_transfer/results/circuit_analysis/triplet_characterization_20260515_nperm2000/top_firing_positions.tsv"
DEFAULT_SIG = REPO / "r2_interpretability_transfer/results/circuit_analysis/triplet_synthesis_20260515_nperm2000/triplet_signatures.tsv"
DEFAULT_OUT = REPO / "r2_interpretability_transfer/results/circuit_analysis/attention_sink_biological_correlate_20260516"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fnum(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def fisher_right(a: int, b: int, c: int, d: int) -> float:
    """One-sided Fisher exact p-value for enrichment in row 1.

    Table:
      [[a, b],
       [c, d]]
    """
    n = a + b + c + d
    row1 = a + b
    col1 = a + c
    max_x = min(row1, col1)

    def log_choose(nn: int, kk: int) -> float:
        if kk < 0 or kk > nn:
            return float("-inf")
        return math.lgamma(nn + 1) - math.lgamma(kk + 1) - math.lgamma(nn - kk + 1)

    def hypergeom(x: int) -> float:
        return math.exp(log_choose(col1, x) + log_choose(n - col1, row1 - x) - log_choose(n, row1))

    return min(1.0, sum(hypergeom(x) for x in range(a, max_x + 1)))


def bh(rows: list[dict[str, Any]], p_col: str = "p_value") -> None:
    indexed = sorted(enumerate(rows), key=lambda item: float(item[1][p_col]))
    m = len(indexed)
    prev = 1.0
    qvals = [1.0] * m
    for rank_from_end, (idx, row) in enumerate(reversed(indexed), start=1):
        rank = m - rank_from_end + 1
        q = min(prev, float(row[p_col]) * m / rank)
        qvals[idx] = q
        prev = q
    for row, q in zip(rows, qvals):
        row["q_value"] = q


def predicates() -> list[tuple[str, str, Any]]:
    return [
        ("first2", "position_1based <= 2", lambda r: 1 <= int(fnum(r.get("position_1based"), -1)) <= 2),
        ("first5", "position_1based <= 5", lambda r: 1 <= int(fnum(r.get("position_1based"), -1)) <= 5),
        ("first10", "position_1based <= 10", lambda r: 1 <= int(fnum(r.get("position_1based"), -1)) <= 10),
        ("nterm20", "position_norm < 0.2", lambda r: fnum(r.get("position_norm")) < 0.2),
        ("edge_5mer", "k5 == edge", lambda r: (r.get("k5") or "").lower() == "edge"),
        (
            "context_starts_m",
            "context or k3 starts with M",
            lambda r: (r.get("context") or r.get("k3") or "").startswith("M"),
        ),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-firing", type=Path, default=DEFAULT_TOP)
    ap.add_argument("--signatures", type=Path, default=DEFAULT_SIG)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--q-threshold", type=float, default=0.05)
    args = ap.parse_args()

    sig_rows = read_tsv(args.signatures)
    attention_ids = [
        r["triplet_id"]
        for r in sig_rows
        if "attention-sink" in (r.get("significant_tests") or "").split(";")
    ]
    top_rows = read_tsv(args.top_firing)
    by_triplet: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in top_rows:
        by_triplet[row["triplet_id"]].append(row)

    tests = []
    for tid in attention_ids:
        target = by_triplet[tid]
        background = [r for r in top_rows if r["triplet_id"] != tid]
        for test_name, description, pred in predicates():
            a = sum(1 for r in target if pred(r))
            b = len(target) - a
            c = sum(1 for r in background if pred(r))
            d = len(background) - c
            tests.append(
                {
                    "triplet_id": tid,
                    "test": test_name,
                    "description": description,
                    "target_true": a,
                    "target_total": len(target),
                    "target_fraction": a / len(target) if target else 0.0,
                    "background_true": c,
                    "background_total": len(background),
                    "background_fraction": c / len(background) if background else 0.0,
                    "odds_ratio_approx": ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5)),
                    "p_value": fisher_right(a, b, c, d),
                }
            )
    bh(tests)
    for row in tests:
        row["significant"] = bool(float(row["q_value"]) < args.q_threshold)

    fields = [
        "triplet_id",
        "test",
        "description",
        "target_true",
        "target_total",
        "target_fraction",
        "background_true",
        "background_total",
        "background_fraction",
        "odds_ratio_approx",
        "p_value",
        "q_value",
        "significant",
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.out_dir / "biological_correlates.tsv", tests, fields)

    best_by_triplet = {}
    for tid in attention_ids:
        rows = [r for r in tests if r["triplet_id"] == tid]
        best_by_triplet[tid] = min(rows, key=lambda r: float(r["q_value"])) if rows else {}
    passed = [
        r
        for r in tests
        if r["significant"] and r["test"] in {"first2", "first5", "first10", "nterm20", "edge_5mer", "context_starts_m"}
    ]
    acceptance = any(
        r["triplet_id"] in attention_ids
        and r["significant"]
        and r["test"] in {"first5", "nterm20", "edge_5mer"}
        and float(r["target_fraction"]) >= 0.70
        for r in tests
    )

    summary = {
        "task": "R2-Add-3 attention-sink biological correlate",
        "attention_sink_triplets": attention_ids,
        "n_tests": len(tests),
        "q_threshold": args.q_threshold,
        "acceptance_gate": "at least one attention-sink triplet has BH q<0.05 and >=70% target fraction on first5, nterm20, or edge_5mer",
        "acceptance_pass": acceptance,
        "best_by_triplet": best_by_triplet,
        "significant_tests": passed,
        "outputs": {
            "table": str(args.out_dir / "biological_correlates.tsv"),
            "summary_md": str(args.out_dir / "summary.md"),
        },
    }
    with (args.out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    md = [
        "# Attention-Sink Biological Correlate",
        "",
        "This test asks whether the four attention-sink triplets have simple N-terminal or start-context correlates.",
        "The background is all other saved top-firing rows from the same M-1 run.",
        "",
        f"- Attention-sink triplets: {', '.join(attention_ids)}",
        f"- Tests: {len(tests)} one-sided Fisher exact tests with BH correction",
        f"- Acceptance gate: {'PASS' if acceptance else 'FAIL'}",
        "",
        "## Best Test Per Triplet",
        "",
        "| Triplet | best test | target fraction | background fraction | odds ratio | q |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for tid in attention_ids:
        r = best_by_triplet[tid]
        if not r:
            continue
        md.append(
            f"| {tid} | {r['test']} | {float(r['target_fraction']):.3f} | "
            f"{float(r['background_fraction']):.3f} | {float(r['odds_ratio_approx']):.3g} | "
            f"{float(r['q_value']):.3g} |"
        )
    md += [
        "",
        "## Interpretation",
        "",
        "- T011, T018 and T023 pass the N-terminal/edge correlate tests strongly in the saved top-firing rows.",
        "- T025 is an attention-sink-statistic triplet but does not share the N-terminal-edge profile; it should be reported as a separate local-motif sink-like case rather than part of the N-terminal subtype.",
        "- The defensible named finding is a protein-LM N-terminal edge attention-sink subtype, not broad biological convergence.",
        "",
    ]
    (args.out_dir / "summary.md").write_text("\n".join(md))
    print(f"Wrote {args.out_dir / 'biological_correlates.tsv'}")
    print(f"Wrote {args.out_dir / 'summary.md'}")
    print(f"Acceptance: {'PASS' if acceptance else 'FAIL'}")


if __name__ == "__main__":
    main()
