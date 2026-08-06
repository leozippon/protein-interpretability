#!/usr/bin/env python3
"""Read a PAA-census panel from its artefacts, without moving the matrices.

**Why this exists.** The controller pulls nothing back, and the obvious way to
read a panel is to pull each arm-draw and open it locally. That is ~20 MB per
arm-draw over a tunnel, and on 2026-08-06 it was the reason fourteen finished
measurements had never been read at all -- four of them on the arm carrying a
live claim. The statistic does not need the matrices: since EXP-R2-124 every run
writes ``census_causal_agreement`` into its own ``paa_gate_report.json``, roughly
a megabyte, so a whole panel can be read from the reports alone.

**The one thing this script will not do is compute the statistic itself.**
Artefacts written before that function existed carry no such key, and for those
it calls :func:`prediction_addressed.census_causal_agreement` on the retained
matrices -- the module, never a local reimplementation. That is not fussiness: on
ProtGPT2 three defensible reductions of the per-sequence census matrix disagreed
by up to 0.05, which is half the modality gap D2.c was arguing about, and the
disagreement is what forced the function to exist. ``--reports-only`` refuses the
recomputation instead of guessing, so the mode that runs anywhere cannot quietly
answer from a different definition than the mode that runs beside the matrices.

**Condition filtering is on by default and is not cosmetic.** A panel pools
arm-draws only if they share a condition, and this tree holds runs at n=200 and
n=600, at two ban depths, and from before and after the decoy repair that moved
ProtGPT2 +0.1987 -> +0.0743 on an identical draw (EXP-R2-125). Pooling across any
of those is pooling two instruments. The filter states what it dropped.

Retrieval is reported against **each arm's own chance level**, ``k^2/n_heads``,
which falls with grid size -- so the x-chance column is a per-arm classification
and not a cross-arm ranking. Only arms sharing a grid size are comparable on it
(EXP-R2-101), and the script prints the grid so that is visible rather than
assumed.

Usage::

    python scripts/transfer/read_paa_panel.py results/transfer_20260801
    python scripts/transfer/read_paa_panel.py <root> --reports-only --json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: The condition every published D2.c figure was measured under. A draw that
#: differs on any of these is a different measurement, not a further draw.
DECLARED_CONDITION = {"census_sequences": 600, "census_ban_depth": 3, "width": 192}


def _agreement(report: dict[str, Any], directory: Path, reports_only: bool) -> dict[str, Any] | None:
    """The versioned statistic, stored or recomputed through the module."""

    stored = report.get("causal", {}).get("census_causal_agreement")
    if stored and "withheld_reason" not in stored:
        return stored
    if reports_only:
        return None
    matrices, causal = directory / "census_matrices.npz", directory / "causal.json"
    if not (matrices.is_file() and causal.is_file()):
        return None
    import numpy as np

    from src.transfer.prediction_addressed import census_causal_agreement

    try:
        return census_causal_agreement(
            np.load(matrices)["paa_specific_matched_per_sequence"],
            json.loads(causal.read_text(encoding="utf-8"))["heads"],
        )
    except (KeyError, ValueError):
        # A selective census raises here by design (standing rule 24) and a
        # pre-schema artefact lacks the field. Both are "not readable", and
        # neither is a reason to invent a number.
        return None


def collect(root: Path, *, reports_only: bool, any_condition: bool) -> tuple[dict, list[str]]:
    """Every readable arm-draw under ``root``, grouped by arm."""

    per_arm: dict[str, list[dict[str, Any]]] = {}
    dropped: list[str] = []
    for report_path in sorted(root.rglob("paa_gate_report.json")):
        directory = report_path.parent
        report = json.loads(report_path.read_text(encoding="utf-8"))
        census = report.get("census")
        if not census:
            dropped.append(f"{directory.name}: no census section")
            continue
        settings = report.get("settings", {})
        if not any_condition:
            differs = [
                f"{key}={settings.get(key)!r}"
                for key, value in DECLARED_CONDITION.items()
                if settings.get(key) != value
            ]
            if differs:
                dropped.append(f"{directory.name}: off-condition ({', '.join(differs)})")
                continue
            pool = census.get("a1_candidate_pool", {})
            if pool.get("layout_tokens_excluded_from_decoys") is None:
                dropped.append(f"{directory.name}: predates the decoy layout guard")
                continue
        agreement = _agreement(report, directory, reports_only)
        if agreement is None:
            dropped.append(f"{directory.name}: no readable agreement statistic")
            continue
        retrieval = agreement["retrieval"]
        per_arm.setdefault(census["arm"], []).append(
            {
                "source": str(directory.relative_to(root)),
                "n_heads": agreement["n_heads"],
                "spearman": agreement["spearman_census_vs_causal_magnitude"],
                "within_layer": agreement["depth_controlled"]["within_layer"],
                "hit_at_k": retrieval["hit_at_k"],
                "chance": retrieval["chance"],
            }
        )
    return per_arm, dropped


def summarise(per_arm: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for arm, draws in per_arm.items():
        hits = sorted(draw["hit_at_k"] for draw in draws)
        rho = [draw["spearman"] for draw in draws]
        # The declared condition constrains sequences, ban depth and width, but
        # NOT the head-grid size, which is set per arm by --causal-heads. Two
        # on-condition draws of one arm run at different grid sizes have
        # different chance levels, and pooling them would divide one draw's hits
        # by another draw's chance while printing a single grid -- the exact
        # substitution this reader exists to prevent. Refuse instead.
        grids = {draw["n_heads"] for draw in draws}
        chances = {draw["chance"] for draw in draws}
        if len(grids) != 1 or len(chances) != 1:
            raise RuntimeError(
                f"{arm}: draws span head grids {sorted(grids)} and chance levels "
                f"{sorted(chances)}. hit@20 has a grid-dependent ceiling, so these "
                "draws are not one population and must not be pooled"
            )
        chance = draws[0]["chance"]
        rows.append(
            {
                "arm": arm,
                "k": len(draws),
                "grid": draws[0]["n_heads"],
                "hit_at_k": hits,
                "median_over_own_chance": statistics.median(hits) / chance,
                "spearman_min": min(rho),
                "spearman_max": max(rho),
            }
        )
    return sorted(rows, key=lambda row: (-row["grid"], row["arm"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="a results root holding paa_gate_report.json files")
    parser.add_argument(
        "--reports-only",
        action="store_true",
        help="refuse to recompute from matrices; read only artefacts that already carry the statistic",
    )
    parser.add_argument(
        "--any-condition",
        action="store_true",
        help="do not filter on the declared condition; the output then pools measurements",
    )
    parser.add_argument("--json", action="store_true", help="emit the summary as JSON")
    args = parser.parse_args()

    if not args.root.is_dir():
        parser.error(f"not a directory: {args.root}")
    per_arm, dropped = collect(
        args.root, reports_only=args.reports_only, any_condition=args.any_condition
    )
    rows = summarise(per_arm)
    if args.json:
        print(json.dumps({"arms": rows, "dropped": dropped}, indent=2))
        return 0

    if args.any_condition:
        print("POOLING ACROSS CONDITIONS -- these rows may mix two instruments\n")
    print(f"{'arm':18}{'K':>3}{'grid':>6}  {'hit@k':<24}{'x own chance':>13}  all-grid rho")
    for row in rows:
        hits = ",".join(str(hit) for hit in row["hit_at_k"])
        print(
            f"{row['arm']:18}{row['k']:>3}{row['grid']:>6}  {hits:<24}"
            f"{row['median_over_own_chance']:>12.1f}x  "
            f"{row['spearman_min']:+.4f}..{row['spearman_max']:+.4f}"
        )
    print(
        "\nx own chance is k^2/n_heads and falls with grid size, so it classifies an arm "
        "against itself.\nOnly arms sharing a grid size are comparable on it."
    )
    if dropped:
        print(f"\nnot pooled ({len(dropped)}):")
        for entry in dropped:
            print(f"  {entry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
