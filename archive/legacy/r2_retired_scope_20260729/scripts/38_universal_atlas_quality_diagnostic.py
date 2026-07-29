#!/usr/bin/env python3
"""Use universal-triplet recovery as a CLT checkpoint quality diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations, product
from pathlib import Path


def parse_named_path(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"bad --atlas {spec!r}; expected name=path")
    name, path = spec.split("=", 1)
    return name.strip(), Path(path.strip())


def node(model: str, layer: int, feature: int) -> tuple[str, int, int]:
    return model, int(layer), int(feature)


def triplets_from_conservation(data: dict, threshold: float) -> list[dict]:
    models = [m["model"] for m in data.get("models", [])]
    if len(models) != 3:
        return []

    edges_by_layer: dict[int, dict[frozenset, float]] = {}
    nodes_by_layer_model: dict[tuple[int, str], set[tuple[str, int, int]]] = {}
    for pair in data.get("pairwise", []):
        ma, mb = pair["model_a"], pair["model_b"]
        for layer in pair.get("layers", []):
            anchor = int(layer["anchor_layer"])
            la, lb = int(layer["layer_a"]), int(layer["layer_b"])
            edges = edges_by_layer.setdefault(anchor, {})
            for match in layer.get("top_feature_matches", []):
                corr = float(match.get("abs_corr", 0.0))
                if corr < threshold:
                    continue
                na = node(ma, la, int(match["feature_a"]))
                nb = node(mb, lb, int(match["feature_b"]))
                edges[frozenset((na, nb))] = corr
                nodes_by_layer_model.setdefault((anchor, ma), set()).add(na)
                nodes_by_layer_model.setdefault((anchor, mb), set()).add(nb)

    out = []
    for anchor, edges in edges_by_layer.items():
        model_nodes = [sorted(nodes_by_layer_model.get((anchor, m), set())) for m in models]
        if any(not xs for xs in model_nodes):
            continue
        for combo in product(*model_nodes):
            pair_keys = [frozenset((a, b)) for a, b in combinations(combo, 2)]
            if not all(k in edges for k in pair_keys):
                continue
            vals = [edges[k] for k in pair_keys]
            row = {
                "anchor_layer": anchor,
                "min_abs_corr": min(vals),
                "mean_abs_corr": sum(vals) / len(vals),
            }
            for model, item in zip(models, combo):
                row[f"{model}_layer"] = item[1]
                row[f"{model}_feature"] = item[2]
            out.append(row)
    out.sort(key=lambda r: (r["min_abs_corr"], r["mean_abs_corr"]), reverse=True)
    return out


def finite_mean(xs: list[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else float("nan")


def summarize_atlas(name: str, path: Path, threshold: float) -> dict:
    data = json.loads(path.read_text())
    triplets = triplets_from_conservation(data, threshold)
    layer_rows = [layer for pair in data.get("pairwise", []) for layer in pair.get("layers", [])]
    return {
        "name": name,
        "path": str(path),
        "threshold": threshold,
        "n_models": len(data.get("models", [])),
        "models": [m.get("model") for m in data.get("models", [])],
        "n_sequences": data.get("input", {}).get("n_sequences"),
        "top_feature_pairs": data.get("config", {}).get("top_feature_pairs"),
        "feature_pool_size": data.get("config", {}).get("feature_pool_size"),
        "n_universal_triplets": len(triplets),
        "mean_layer_cka": finite_mean([r.get("cka", float("nan")) for r in layer_rows]),
        "mean_abs_match_corr": finite_mean([r.get("mean_abs_match_corr", float("nan")) for r in layer_rows]),
        "top_triplets": triplets[:10],
    }


def write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "name",
        "n_universal_triplets",
        "threshold",
        "n_sequences",
        "top_feature_pairs",
        "feature_pool_size",
        "mean_layer_cka",
        "mean_abs_match_corr",
        "path",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", action="append", required=True, help="Repeated: name=conservation.json")
    ap.add_argument("--reference-name", default="v2_reference")
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--reference-pass-min", type=int, default=30)
    ap.add_argument("--diagnostic-fail-max", type=int, default=20)
    ap.add_argument("--out-dir", type=Path, default=Path("r2_interpretability_transfer/results/circuit_analysis/universal_atlas_quality_diagnostic_20260516"))
    args = ap.parse_args()

    rows = [summarize_atlas(name, path, args.threshold) for name, path in map(parse_named_path, args.atlas)]
    reference = next((r for r in rows if r["name"] == args.reference_name), None)
    if reference is None:
        reference = max(rows, key=lambda r: r["n_universal_triplets"])

    weak = [
        r for r in rows
        if r["name"] != reference["name"] and r["n_universal_triplets"] < args.diagnostic_fail_max
    ]
    outcome = "PASS" if reference["n_universal_triplets"] >= args.reference_pass_min and weak else "FAIL"

    payload = {
        "task": "R2-Add-2 universal atlas checkpoint-quality diagnostic",
        "threshold": args.threshold,
        "reference_name": reference["name"],
        "reference_pass_min": args.reference_pass_min,
        "diagnostic_fail_max": args.diagnostic_fail_max,
        "outcome": outcome,
        "atlas_rows": rows,
        "interpretation": (
            "PASS means the mature reference CLTs recover many universal triplets while at least one "
            "candidate checkpoint loses the structure, supporting universal-triplet count as a practical "
            "CLT quality diagnostic. FAIL means the diagnostic should not be emphasized."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_tsv(args.out_dir / "atlas_quality.tsv", rows)

    lines = [
        "# Universal Atlas Quality Diagnostic",
        "",
        f"- Threshold: abs(r) >= {args.threshold:.2f}",
        f"- Reference atlas: {reference['name']} ({reference['n_universal_triplets']} triplets)",
        f"- Outcome: {outcome}",
        "",
        "| Atlas | Universal triplets | Mean layer CKA | Mean match abs(r) | Sequences |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda r: r["n_universal_triplets"], reverse=True):
        lines.append(
            f"| {row['name']} | {row['n_universal_triplets']} | "
            f"{row['mean_layer_cka']:.4f} | {row['mean_abs_match_corr']:.4f} | {row['n_sequences']} |"
        )
    lines += [
        "",
        "Interpretation: " + payload["interpretation"],
        "",
    ]
    (args.out_dir / "summary.md").write_text("\n".join(lines))
    print(f"Wrote {args.out_dir / 'summary.md'}")
    print(f"Outcome: {outcome}")


if __name__ == "__main__":
    main()
