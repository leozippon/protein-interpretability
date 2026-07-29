#!/usr/bin/env python
"""Audit mechanism-classifier features against annotation metadata.

For each mechanism class and each production SAE layer, this script retrains
the multinomial LR used by `16_mechanism_classifier.py`, extracts the top
positive coefficient features, and joins them to the current annotation pkl.

The output is intentionally a markdown table for paper/figure triage. If the
annotation pkl lacks `firing_positions`, the script records that limitation
instead of inventing max-activating sequence examples.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
LAYERS = [19, 23, 27, 31, 35]


def load_mechanism_module():
    path = ROOT / "scripts" / "16_mechanism_classifier.py"
    spec = importlib.util.spec_from_file_location("mechanism_classifier_16", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mechanism_classifier_16"] = module
    spec.loader.exec_module(module)
    return module


def load_annotation_results(annotation_dir: Path, layer: int) -> dict[int, object]:
    expanded = annotation_dir / f"ours_3B_l{layer}_step500000_expanded.pkl"
    base = annotation_dir / f"ours_3B_l{layer}_step500000.pkl"
    path = expanded if expanded.exists() else base
    if not path.exists():
        raise FileNotFoundError(f"missing annotation pkl for L{layer}: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    results = {int(r.feature_idx): r for r in data["results"]}
    expanded_data = data.get("expanded", {}) if isinstance(data, dict) else {}
    labels = expanded_data.get("new_best_label")
    combined = expanded_data.get("combined_f1")
    if labels is not None:
        for feat_idx, label in enumerate(labels):
            if feat_idx in results and label:
                setattr(results[feat_idx], "expanded_best_label", str(label))
                if combined is not None and feat_idx < len(combined):
                    setattr(results[feat_idx], "expanded_best_f1", float(combined[feat_idx]))
    return results


def best_matching(scores: dict, prefixes: tuple[str, ...], contains: tuple[str, ...] = ()) -> str:
    rows = []
    for ann, vals in (scores or {}).items():
        low = ann.lower()
        if ann.startswith(prefixes) or any(x in low for x in contains):
            f1 = vals[0] if isinstance(vals, (list, tuple)) else vals
            rows.append((float(f1), ann))
    if not rows:
        return ""
    return sorted(rows, reverse=True)[0][1]


def interpret(best_ann: str, top_pfam: str, top_go: str, top_binding: str) -> str:
    text = " ".join([best_ann, top_pfam, top_go, top_binding]).lower()
    if any(x in text for x in ["binding", "interface", "dimer", "oligomer", "complex"]):
        return "binding/interface-like signal; check firing positions before claiming DN biology"
    if any(x in text for x in ["kinase", "regulat", "alloster", "signal", "receptor"]):
        return "regulatory/signaling-like signal; plausible GOF axis, needs manual residue audit"
    if any(x in text for x in ["active", "catalytic", "enzyme", "domain", "fold", "core"]):
        return "domain/function-like signal; plausible LOF/fold-disruption axis"
    if any(x in text for x in ["transmembrane", "topological", "signal peptide"]):
        return "topology/localization-like signal; interpret as structural context"
    return "annotation weak or family-level; needs firing-position rerun for manual interpretation"


def format_examples(examples: list | None, n: int = 5) -> str:
    if not examples:
        return ""
    rows = []
    for item in examples[:n]:
        if not isinstance(item, dict):
            continue
        acc = item.get("accession", "")
        pos = item.get("position", "")
        aa = item.get("aa", "")
        act = item.get("activation", 0.0)
        rows.append(f"{acc}:{pos}{aa}({float(act):.2f})")
    return "; ".join(rows)


def fit_lr_and_rank(mech_module, mechanisms_path: str, top_k: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    meta_by_layer = mech_module.load_annotation_metadata()
    labels = mech_module.load_mechanism_labels(mechanisms_path)
    keep_classes = sorted(c for c in set(labels.values()) if c != "UNLABELED")
    X, _llr, y, _groups, col_meta = mech_module.build_features(meta_by_layer, labels, keep_classes)
    good = X.std(0) > 1e-8
    X = X[:, good]
    col_meta = [col_meta[i] for i, keep in enumerate(good) if keep]

    scaler = StandardScaler()
    Xn = scaler.fit_transform(X)
    clf = LogisticRegression(C=0.1, solver="lbfgs", max_iter=3000, class_weight="balanced")
    clf.fit(Xn, y)

    ranked = {}
    for class_idx, cls in enumerate(clf.classes_):
        coef = clf.coef_[class_idx]
        by_layer_feature = defaultdict(list)
        for col_idx, value in enumerate(coef):
            layer, feat_idx, kind = col_meta[col_idx]
            by_layer_feature[(layer, feat_idx)].append((float(value), kind))

        ranked[cls] = {}
        for layer in LAYERS:
            rows = []
            for (l, feat_idx), vals in by_layer_feature.items():
                if l != layer:
                    continue
                best_coef, best_kind = max(vals, key=lambda x: x[0])
                rows.append((best_coef, feat_idx, best_kind))
            rows.sort(reverse=True, key=lambda x: x[0])
            ranked[cls][layer] = rows[:top_k]
    return ranked, list(clf.classes_)


def markdown_table(rows: list[dict]) -> str:
    headers = [
        "class", "layer", "feature", "coef", "kind", "best_annotation",
        "best_f1", "top_pfam_or_domain", "top_go_or_functional",
        "top_binding", "expanded_best_label", "expanded_best_f1",
        "top_firing_examples", "manual_interpretation",
    ]
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        vals = []
        for h in headers:
            v = r.get(h, "")
            if isinstance(v, float):
                v = f"{v:.4g}"
            vals.append(str(v).replace("|", "/"))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mechanisms", default="r1_encoder_interpretability_benchmark/results/variant_effect/variant_mechanisms.tsv")
    ap.add_argument("--annotation-dir", default="r1_encoder_interpretability_benchmark/results/annotation_alignment")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out-md", default="r1_encoder_interpretability_benchmark/results/variant_effect/mechanism_feature_audit_20260503.md")
    ap.add_argument("--out-json", default="r1_encoder_interpretability_benchmark/results/variant_effect/mechanism_feature_audit_20260503.json")
    args = ap.parse_args()

    os.chdir(REPO)
    sys.path.insert(0, str(ROOT))
    mech_module = load_mechanism_module()
    ranked, classes = fit_lr_and_rank(mech_module, args.mechanisms, args.top_k)

    annotation_dir = Path(args.annotation_dir)
    ann_by_layer = {layer: load_annotation_results(annotation_dir, layer) for layer in LAYERS}

    rows = []
    for cls in classes:
        for layer in LAYERS:
            for coef, feat_idx, kind in ranked[cls][layer]:
                r = ann_by_layer[layer].get(feat_idx)
                scores = getattr(r, "all_scores", {}) if r is not None else {}
                best_ann = getattr(r, "best_annotation", "") if r is not None else ""
                top_pfam = best_matching(scores, ("pfam/", "domain/"))
                top_go = best_matching(scores, ("go/", "functional/"))
                top_binding = best_matching(scores, ("biolip/",), ("binding", "active site"))
                has_firing = bool(getattr(r, "firing_positions", None)) if r is not None else False
                expanded_label = getattr(r, "expanded_best_label", "") if r is not None else ""
                expanded_f1 = getattr(r, "expanded_best_f1", 0.0) if r is not None else 0.0
                rows.append({
                    "class": cls,
                    "layer": layer,
                    "feature": feat_idx,
                    "coef": coef,
                    "kind": kind,
                    "best_annotation": best_ann,
                    "best_f1": float(getattr(r, "best_f1", 0.0) or 0.0) if r is not None else 0.0,
                    "classification": getattr(r, "classification", "") if r is not None else "",
                    "mean_activation": float(getattr(r, "mean_activation", 0.0) or 0.0) if r is not None else 0.0,
                    "num_tokens_active": int(getattr(r, "num_tokens_active", 0) or 0) if r is not None else 0,
                    "top_pfam_or_domain": top_pfam,
                    "top_go_or_functional": top_go,
                    "top_binding": top_binding,
                    "expanded_best_label": expanded_label,
                    "expanded_best_f1": float(expanded_f1 or 0.0),
                    "has_firing_positions": has_firing,
                    "top_firing_examples": format_examples(
                        getattr(r, "top_firing_examples", None) if r is not None else None
                    ),
                    "manual_interpretation": interpret(
                        " ".join([best_ann, expanded_label]), top_pfam, top_go, top_binding
                    ),
                })

    out = {
        "task": "T1-B mechanism feature audit",
        "classes": classes,
        "layers": LAYERS,
        "top_k_per_class_layer": args.top_k,
        "n_rows": len(rows),
        "note": (
            "This audit uses firing-position-enabled annotation pkls from T1-D. "
            "Top firing examples are residue-level examples, not full manual "
            "biological validation."
        ),
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    with open(args.out_md, "w") as f:
        f.write("# T1-B Mechanism Feature Audit\n\n")
        f.write(out["note"] + "\n\n")
        f.write(markdown_table(rows))

    print(f"Saved JSON: {args.out_json}")
    print(f"Saved markdown: {args.out_md}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
