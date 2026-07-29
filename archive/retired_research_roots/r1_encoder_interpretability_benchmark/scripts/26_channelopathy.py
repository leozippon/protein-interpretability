#!/usr/bin/env python
"""T1-E channelopathy cohort readiness audit.

The T1-E acceptance criterion requires mechanism-curated channelopathy variants
from ClinGen/literature and optional retrospective drug-response labels. Those
curated files are not currently staged. This script audits what is available
locally and writes a reproducible blocker report.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "r1_encoder_interpretability_benchmark" / "results" / "variant_effect"
CHANNEL_GENES = ["KCNQ1", "SCN5A", "KCNH2", "CACNA1C"]


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_mechanism_labels(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    rows = {}
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            rows[(parts[idx["gene"]].upper(), parts[idx["variant"]])] = {
                "clinical_significance": parts[idx["clinical_significance"]],
                "is_pathogenic": int(parts[idx["is_pathogenic"]]),
                "mechanism": parts[idx["mechanism"]],
                "source": parts[idx["source"]],
            }
    return rows


def find_assets() -> dict[str, list[str]]:
    patterns = {
        "clingen_or_literature": [
            "*clingen*", "*ClinGen*", "*lqts*", "*LQTS*", "*brugada*",
            "*Brugada*", "*channelopathy*", "*Channelopathy*",
        ],
        "drug_response": [
            "*mexiletine*", "*beta*blocker*", "*propranolol*", "*nadolol*",
            "*drug*response*", "*therapy*",
        ],
        "dms_channel": ["*KCNH2*", "*SCN5A*", "*KCNQ1*", "*CACNA1C*"],
    }
    roots = [REPO / "data", REPO / "r1_encoder_interpretability_benchmark" / "data", REPO / "r1_encoder_interpretability_benchmark" / "results"]
    out = {}
    for group, pats in patterns.items():
        found = []
        for root in roots:
            if not root.exists():
                continue
            for pat in pats:
                found.extend(str(p.relative_to(REPO)) for p in root.rglob(pat) if p.is_file())
        out[group] = sorted(set(found))
    return out


def build_report() -> dict:
    llr_rows = load_json(OUT_DIR / "esm2_per_variant_llr.json") or []
    pred_rows = load_json(OUT_DIR / "variant_predictions.json") or []
    mechanisms = load_mechanism_labels(OUT_DIR / "variant_mechanisms.tsv")
    curated_path = REPO / "r1_encoder_interpretability_benchmark" / "data" / "channelopathy" / "channelopathy_mechanism_positive_labels.tsv"
    drug_path = REPO / "r1_encoder_interpretability_benchmark" / "data" / "channelopathy" / "channelopathy_drug_response_labels.tsv"
    curated_rows = []
    drug_rows = []
    if curated_path.exists():
        with curated_path.open() as f:
            header = f.readline().rstrip("\n").split("\t")
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == len(header):
                    curated_rows.append(dict(zip(header, parts)))
    if drug_path.exists():
        with drug_path.open() as f:
            header = f.readline().rstrip("\n").split("\t")
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == len(header):
                    drug_rows.append(dict(zip(header, parts)))

    llr_channel = [r for r in llr_rows if r.get("gene", "").upper() in CHANNEL_GENES]
    pred_channel = [r for r in pred_rows if r.get("gene", "").upper() in CHANNEL_GENES]
    mechanism_channel = [
        {"gene": g, "variant": v, **m}
        for (g, v), m in mechanisms.items()
        if g in CHANNEL_GENES and m["mechanism"] != "UNLABELED"
    ]
    assets = find_assets()
    curated_assets = assets["clingen_or_literature"]
    drug_assets = assets["drug_response"]

    status = "blocked_missing_curated_mechanism_labels"
    acceptance_met = False
    if curated_assets and curated_rows:
        status = "curated_labels_staged_needs_scoring"

    return {
        "task": "T1-E channelopathy clinical cohort",
        "status": status,
        "acceptance_met": acceptance_met,
        "genes": CHANNEL_GENES,
        "available_current_variant_subset": {
            "llr_rows": len(llr_channel),
            "llr_by_gene": dict(Counter(r["gene"].upper() for r in llr_channel)),
            "prediction_rows": len(pred_channel),
            "prediction_by_gene": dict(Counter(r["gene"].upper() for r in pred_channel)),
            "mechanism_labeled_rows": len(mechanism_channel),
            "mechanism_by_gene": dict(Counter(r["gene"].upper() for r in mechanism_channel)),
            "curated_channelopathy_rows": len(curated_rows),
            "curated_channelopathy_by_gene": dict(Counter(r["gene"].upper() for r in curated_rows)),
            "drug_response_rows": len(drug_rows),
            "drug_response_by_gene": dict(Counter(r["gene"].upper() for r in drug_rows)),
        },
        "asset_status": assets,
        "decision": (
            "T1-E curated labels are now staged. The next blocker is scoring "
            "these variants with the R1 pipeline and checking mechanism-label "
            "concordance; drug-response labels are sparse and should be analyzed "
            "as secondary research labels."
            if curated_rows else
            "T1-E is blocked: the repo has channel DMS files for SCN5A/KCNH2 "
            "and 16 KCNQ1 ClinVar LLR rows, but no staged ClinGen/literature "
            "mechanism-curated channelopathy cohort or retrospective drug-response "
            "labels. Acceptance requires those curated labels before mechanism "
            "concordance can be measured."
        ),
        "example_llr_rows": llr_channel[:10],
        "example_prediction_rows": pred_channel[:10],
        "example_mechanism_rows": mechanism_channel[:10],
    }


def markdown(report: dict) -> str:
    lines = ["# T1-E Channelopathy Cohort Readiness\n"]
    lines.append(report["decision"] + "\n")
    lines.append("## Local Variant Coverage\n")
    cov = report["available_current_variant_subset"]
    lines.append("| Source | Rows | By Gene |")
    lines.append("|---|---:|---|")
    lines.append(f"| ESM-2 LLR ClinVar subset | {cov['llr_rows']} | {cov['llr_by_gene']} |")
    lines.append(f"| Existing R1 predictions | {cov['prediction_rows']} | {cov['prediction_by_gene']} |")
    lines.append(f"| Existing mechanism labels | {cov['mechanism_labeled_rows']} | {cov['mechanism_by_gene']} |")
    lines.append(f"| Curated channelopathy mechanism labels | {cov['curated_channelopathy_rows']} | {cov['curated_channelopathy_by_gene']} |")
    lines.append(f"| Curated drug-response labels | {cov['drug_response_rows']} | {cov['drug_response_by_gene']} |")
    lines.append("\n## Asset Status\n")
    lines.append("| Asset Group | Files |")
    lines.append("|---|---|")
    for group, files in report["asset_status"].items():
        text = "<br>".join(files) if files else ""
        lines.append(f"| {group} | {text} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default="r1_encoder_interpretability_benchmark/results/variant_effect/channelopathy_readiness_20260504.json")
    ap.add_argument("--out-md", default="r1_encoder_interpretability_benchmark/results/variant_effect/channelopathy_readiness_20260504.md")
    args = ap.parse_args()

    report = build_report()
    out_json = REPO / args.out_json
    out_md = REPO / args.out_md
    os.makedirs(out_json.parent, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    with open(out_md, "w") as f:
        f.write(markdown(report))
    print(f"Saved JSON: {out_json}")
    print(f"Saved markdown: {out_md}")
    print(report["decision"])


if __name__ == "__main__":
    main()
