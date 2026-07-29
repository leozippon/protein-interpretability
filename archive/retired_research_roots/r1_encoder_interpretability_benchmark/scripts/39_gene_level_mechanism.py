#!/usr/bin/env python3
"""Gene-level R1 mechanism classifier with Pfam-family holdout groups.

Opus asked for a gene-level / Pfam-clan mechanism test.  The current staged
resources include UniProt-to-Pfam family intervals, but not a Pfam clan map.
This script therefore uses the dominant Pfam family per UniProt protein as the
holdout group and records that limitation explicitly.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
OUT_DIR = REPO / "r1_encoder_interpretability_benchmark" / "results" / "variant_effect"


def load_script_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_gene_uniprot(path: Path) -> dict[tuple[str, str], str]:
    out = {}
    if not path.exists():
        return out
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gene = (row.get("gene") or "").upper()
            variant = row.get("variant") or row.get("protein_variant") or ""
            uniprot = row.get("uniprot_id") or ""
            if gene and variant and uniprot:
                out[(gene, variant)] = uniprot
    return out


def load_uniprot_pfam(path: Path) -> dict[str, str]:
    pfams = defaultdict(Counter)
    if not path.exists():
        return {}
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            uniprot = row.get("uniprot")
            pfam = row.get("pfam_id")
            if not uniprot or not pfam:
                continue
            try:
                span = max(1, int(row.get("end", 0)) - int(row.get("start", 0)) + 1)
            except ValueError:
                span = 1
            pfams[uniprot][pfam] += span
    return {u: counts.most_common(1)[0][0] for u, counts in pfams.items() if counts}


def build_variant_identity(mech_labels, keep_classes):
    with open(OUT_DIR / "scaled_perturbation_signatures.pkl", "rb") as f:
        import pickle

        sigs = pickle.load(f)
    identities = []
    for key, var_sigs in sigs.items():
        s0 = var_sigs[0]
        gene = s0["gene"].upper()
        variant = s0["variant_str"]
        mech = mech_labels.get((gene, variant))
        if mech is None or mech not in keep_classes:
            continue
        identities.append((gene, variant))
    return identities


def build_features_from_cache(cache_path: Path, mechanisms_path: Path, keep_classes: list[str]):
    with cache_path.open("rb") as f:
        cache = pickle.load(f)
    with open(OUT_DIR / "scaled_perturbation_signatures.pkl", "rb") as f:
        sigs = pickle.load(f)
    with open(OUT_DIR / "esm2_per_variant_llr.json") as f:
        per_variant_llr = json.load(f)
    llr_lookup = {
        (r["gene"].upper(), r["variant"]): float(r["llr"])
        for r in per_variant_llr
    }
    mech_labels = {}
    with mechanisms_path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            mech_labels[(row["gene"].upper(), row["variant"])] = row["mechanism"]

    selected_by_layer = {int(k): [int(x) for x in v] for k, v in cache["selected_by_layer"].items()}
    weights_by_layer = {
        int(k): {int(feat): float(w) for feat, w in zip(selected_by_layer[int(k)], v)}
        for k, v in cache["weights_by_layer"].items()
    }

    rows = []
    llrs = []
    labels = []
    identities = []
    for _key, var_sigs in sigs.items():
        s0 = var_sigs[0]
        gene = s0["gene"].upper()
        variant = s0["variant_str"]
        mech = mech_labels.get((gene, variant))
        if mech is None or mech not in keep_classes:
            continue
        sig_by_layer = {int(s["layer"]): s for s in var_sigs}
        cols = []
        for layer, feat_idx, kind in cache["col_meta"]:
            layer = int(layer)
            feat_idx = int(feat_idx)
            s = sig_by_layer.get(layer)
            if s is None:
                cols.append(0.0)
                continue
            if kind == "abs_local":
                cols.append(abs(float(s["delta_local"][feat_idx])))
            elif kind == "abs_global":
                cols.append(abs(float(s["delta_global"][feat_idx])))
            elif kind == "weighted_local":
                cols.append(float(s["delta_local"][feat_idx]) * weights_by_layer[layer].get(feat_idx, 0.0))
            else:
                raise ValueError(f"unknown feature kind {kind!r}")
        rows.append(cols)
        llrs.append(-llr_lookup.get((gene, variant), 0.0))
        labels.append(mech)
        identities.append((gene, variant))
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(llrs, dtype=np.float32),
        np.asarray(labels),
        identities,
    )


def aggregate_by_gene(X, llr, y, identities, gene_uniprot, uniprot_pfam):
    rows = []
    by_gene = defaultdict(list)
    for i, (gene, variant) in enumerate(identities):
        by_gene[gene].append(i)

    for gene, idxs in sorted(by_gene.items()):
        labels = Counter(y[idxs])
        label, label_n = labels.most_common(1)[0]
        if len(labels) > 1 and label_n / len(idxs) < 0.6:
            # Ambiguous genes would create noisy gene-level supervision.
            continue
        Xi = X[idxs]
        llri = llr[idxs]
        feat = np.concatenate(
            [
                Xi.mean(axis=0),
                np.percentile(Xi, 75, axis=0),
                Xi.max(axis=0),
                np.array([llri.mean(), np.percentile(llri, 75), llri.max()], dtype=np.float32),
            ]
        ).astype(np.float32)
        uniprots = [gene_uniprot.get((gene, identities[i][1]), "") for i in idxs]
        uniprot = Counter(u for u in uniprots if u).most_common(1)
        uniprot_id = uniprot[0][0] if uniprot else ""
        pfam_group = uniprot_pfam.get(uniprot_id) or f"NO_PFAM::{gene}"
        rows.append(
            {
                "gene": gene,
                "label": label,
                "n_variants": len(idxs),
                "label_counts": dict(labels),
                "uniprot_id": uniprot_id,
                "holdout_group": pfam_group,
                "features": feat,
            }
        )
    return rows


def make_cv(y, groups, n_splits):
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    min_class = min(Counter(y).values())
    n = min(n_splits, min_class)
    if n < 2:
        raise ValueError(f"not enough genes per class for CV: {Counter(y)}")
    if len(set(groups)) >= n and max(Counter(groups).values()) < len(groups):
        return StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=42), n, "pfam_family_group"
    return StratifiedKFold(n_splits=n, shuffle=True, random_state=42), n, "stratified_gene"


def evaluate(X, y, groups, classes, n_splits):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    cv, real_splits, cv_unit = make_cv(y, groups, n_splits)
    probs_all = np.zeros((len(y), len(classes)), dtype=np.float32)
    pred_all = np.empty(len(y), dtype=object)
    split_iter = cv.split(X, y, groups) if cv_unit == "pfam_family_group" else cv.split(X, y)
    fold_groups = []
    for fold, (tr, te) in enumerate(split_iter):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        clf = LogisticRegression(C=0.1, solver="lbfgs", max_iter=5000, class_weight="balanced")
        clf.fit(Xtr, y[tr])
        probs = clf.predict_proba(Xte)
        for ci, c in enumerate(classes):
            if c in clf.classes_:
                probs_all[te, ci] = probs[:, list(clf.classes_).index(c)]
        pred_all[te] = clf.predict(Xte)
        fold_groups.append(
            {
                "fold": fold,
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "test_groups": sorted(set(groups[te].tolist()))[:20],
            }
        )

    per_class_auc = {}
    for ci, c in enumerate(classes):
        y_bin = (y == c).astype(int)
        if 0 < y_bin.sum() < len(y_bin):
            per_class_auc[c] = float(roc_auc_score(y_bin, probs_all[:, ci]))
        else:
            per_class_auc[c] = float("nan")
    return {
        "cv_unit": cv_unit,
        "n_splits": int(real_splits),
        "per_class_auc": per_class_auc,
        "macro_auc": float(np.nanmean(list(per_class_auc.values()))),
        "macro_f1": float(f1_score(y, pred_all, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, pred_all)),
        "confusion_matrix": confusion_matrix(y, pred_all, labels=classes).tolist(),
        "classes": classes,
        "folds": fold_groups,
    }


def write_markdown(path: Path, payload: dict) -> None:
    r = payload["result"]
    gate = payload["acceptance_gate"]
    lines = [
        "# R1 Gene-Level Mechanism Classifier",
        "",
        f"- Genes evaluated: {payload['n_genes']}",
        f"- CV unit: {r['cv_unit']}",
        f"- Holdout groups: {payload['n_holdout_groups']}",
        f"- Macro-AUC: {r['macro_auc']:.4f}",
        f"- Macro-F1: {r['macro_f1']:.4f}",
        f"- Accuracy: {r['accuracy']:.4f}",
        f"- Gate: {gate}",
        "",
        "## Per-Class AUC",
        "",
        "| Class | AUC |",
        "|---|---:|",
    ]
    for c, auc in r["per_class_auc"].items():
        lines.append(f"| {c} | {auc:.4f} |")
    lines += [
        "",
        "## Limitation",
        "",
        "The current repository has Pfam family intervals but no Pfam clan map. "
        "This run therefore uses the dominant Pfam family per UniProt protein "
        "as a clan proxy. Treat this as a bounded mechanism diagnostic, not a "
        "full Pfam-clan holdout.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mechanisms", type=Path, default=OUT_DIR / "variant_mechanisms.tsv")
    ap.add_argument("--baseline-scores", type=Path, default=OUT_DIR / "external_baselines_available_scores_20260507.tsv")
    ap.add_argument("--classifier-cache", type=Path, default=OUT_DIR / "indel_mechanism_classifier_20260504.pkl")
    ap.add_argument("--pfam-residue", type=Path, default=REPO / "data" / "interpro" / "pfam_residue.tsv")
    ap.add_argument("--out-json", type=Path, default=OUT_DIR / "gene_level_mechanism_20260512.json")
    ap.add_argument("--out-md", type=Path, default=OUT_DIR / "gene_level_mechanism_20260512.md")
    ap.add_argument("--min-per-class", type=int, default=15)
    ap.add_argument("--n-splits", type=int, default=10)
    args = ap.parse_args()

    os.chdir(REPO)
    mech_labels = {}
    with args.mechanisms.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            mech_labels[(row["gene"].upper(), row["variant"])] = row["mechanism"]
    class_counts = Counter(mech_labels.values())
    keep_classes = sorted(c for c, n in class_counts.items() if n >= args.min_per_class and c != "UNLABELED")
    X, llr, y, identities = build_features_from_cache(args.classifier_cache, args.mechanisms, keep_classes)

    gene_uniprot = load_gene_uniprot(args.baseline_scores)
    uniprot_pfam = load_uniprot_pfam(args.pfam_residue)
    rows = aggregate_by_gene(X, llr, y, identities, gene_uniprot, uniprot_pfam)
    y_gene = np.array([r["label"] for r in rows])
    groups = np.array([r["holdout_group"] for r in rows])
    X_gene = np.stack([r["features"] for r in rows]).astype(np.float32)
    classes = sorted(set(y_gene.tolist()))
    result = evaluate(X_gene, y_gene, groups, classes, args.n_splits)

    macro = result["macro_auc"]
    if macro >= 0.70:
        gate = "main_text_candidate"
    elif macro >= 0.60:
        gate = "supplement_only"
    else:
        gate = "drop_mechanism_headline"
    payload = {
        "task": "R1 B-1 gene-level mechanism with Pfam-family holdout proxy",
        "status": "completed",
        "classes": classes,
        "n_variants_input": int(len(y)),
        "n_genes": int(len(rows)),
        "n_holdout_groups": int(len(set(groups.tolist()))),
        "class_counts_gene": dict(Counter(y_gene.tolist())),
        "holdout_resource": "dominant Pfam family per UniProt; Pfam clan map unavailable",
        "result": result,
        "acceptance_gate": gate,
        "gene_rows": [
            {k: v for k, v in r.items() if k != "features"}
            for r in rows
        ],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n")
    write_markdown(args.out_md, payload)
    print(json.dumps({k: payload[k] for k in ["status", "n_genes", "class_counts_gene", "acceptance_gate"]}, indent=2))
    print(f"macro_auc={result['macro_auc']:.4f} macro_f1={result['macro_f1']:.4f}")
    print(f"saved {args.out_json}")


if __name__ == "__main__":
    main()
