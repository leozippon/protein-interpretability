#!/usr/bin/env python
"""Score curated channelopathy variants and test R1 mechanism concordance.

This script consumes the T1-E curated labels and computes SAE perturbation
signatures for supported missense variants. Long channel proteins are scored in
a 1022-aa window centered on the mutation so the mutation site is never lost to
ESM-2 truncation.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import pickle
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.feature_annotation import load_our_sae
from src.analysis.variant_effect import MissenseVariant, compute_perturbation_signature


REPO = Path(__file__).resolve().parents[2]
R1 = REPO / "r1_encoder_interpretability_benchmark"
OUT_DIR = R1 / "results" / "variant_effect"
DATA_DIR = R1 / "data" / "channelopathy"
LAYERS = [19, 23, 27, 31, 35]
CHECKPOINT_DIRS = {
    19: "r1_encoder_interpretability_benchmark/results/final_checkpoints/r1_h200_2gpu_20260401/sae_weights/layer_19/step_500000",
    23: "r1_encoder_interpretability_benchmark/results/final_checkpoints/r1_h200_2gpu_20260401/sae_weights/layer_23/step_500000",
    27: "r1_encoder_interpretability_benchmark/results/final_checkpoints/r1_h200_2gpu_20260401/sae_weights/layer_27/step_500000",
    31: "r1_encoder_interpretability_benchmark/results/final_checkpoints/r1_h200_2gpu_20260401/sae_weights/layer_31/step_500000",
    35: "r1_encoder_interpretability_benchmark/results/final_checkpoints/r1_h200_2gpu_20260401/sae_weights/layer_35/step_500000",
}
GENE_TO_CANONICAL = {
    "KCNQ1": "P51787",
    "SCN5A": "Q14524",
    "KCNH2": "Q12809",
    "CACNA1C": "Q13936",
}
AA3 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}
CATEGORY_WEIGHTS = {
    "functional": 5.0,
    "ptm": 4.0,
    "domain": 3.0,
    "region": 2.0,
    "topology": 1.5,
    "secondary_structure": 1.0,
    "chain": 0.5,
}
F1_THRESHOLD = 0.1


def parse_uniprot_fasta(path: Path, accessions: set[str]) -> dict[str, str]:
    opener = gzip.open if str(path).endswith(".gz") else open
    seqs = {}
    acc = None
    chunks: list[str] = []
    with opener(path, "rt") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if acc in accessions:
                    seqs[acc] = "".join(chunks)
                acc = None
                chunks = []
                m = re.match(r">sp\|([^|]+)\|", line)
                if m and m.group(1) in accessions:
                    acc = m.group(1)
            elif acc:
                chunks.append(line.strip())
    if acc in accessions:
        seqs[acc] = "".join(chunks)
    return seqs


def parse_variant_protein(text: str) -> tuple[str, int, str] | None:
    m = re.fullmatch(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})", text.strip())
    if not m:
        return None
    wt3, pos, mut3 = m.groups()
    wt = AA3.get(wt3)
    mut = AA3.get(mut3)
    if not wt or not mut:
        return None
    return wt, int(pos), mut


def load_labels(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def crop_window(seq: str, pos: int, max_len: int) -> tuple[str, int, int, int]:
    if len(seq) <= max_len:
        return seq, 1, len(seq), pos
    zero = pos - 1
    start0 = max(0, min(zero - max_len // 2, len(seq) - max_len))
    end0 = start0 + max_len
    return seq[start0:end0], start0 + 1, end0, pos - start0


def prepare_variants(labels: list[dict], seqs: dict[str, str], max_len: int) -> tuple[list[dict], list[dict]]:
    supported = []
    audit = []
    seen = set()
    for row in labels:
        gene = row["gene"].upper()
        parsed = parse_variant_protein(row["variant_protein"])
        acc = GENE_TO_CANONICAL.get(gene)
        if parsed is None:
            audit.append({**row, "status": "unsupported_non_missense_or_complex", "reason": row["variant_protein"]})
            continue
        if not acc or acc not in seqs:
            audit.append({**row, "status": "missing_sequence", "reason": acc or "no canonical accession"})
            continue
        wt, pos, mut = parsed
        seq = seqs[acc]
        if pos < 1 or pos > len(seq):
            audit.append({**row, "status": "position_out_of_range", "reason": f"pos={pos} len={len(seq)}"})
            continue
        observed = seq[pos - 1]
        if observed != wt:
            audit.append({**row, "status": "wt_mismatch", "reason": f"expected {wt}{pos}, observed {observed}{pos}"})
            continue
        key = (gene, row["variant_protein"], row["condition"], row["mechanism_label"])
        if key in seen:
            continue
        seen.add(key)
        window, window_start, window_end, local_pos = crop_window(seq, pos, max_len)
        supported.append({
            **row,
            "uniprot_id": acc,
            "full_position": pos,
            "wt_residue": wt,
            "mut_residue": mut,
            "full_sequence_len": len(seq),
            "window_start": window_start,
            "window_end": window_end,
            "local_position": local_pos,
            "window_sequence": window,
            "variant_str": f"{wt}{pos}{mut}",
            "status": "supported_missense",
        })
        audit.append({**row, "status": "supported_missense", "reason": ""})
    return supported, audit


def save_prepare_outputs(records: list[dict], audit: list[dict], out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "gene", "variant_protein", "variant_cdna", "condition", "mechanism_label",
        "drug_response_label", "evidence_level", "evidence_type", "source_id",
        "source_url_or_doi", "notes", "uniprot_id", "variant_str", "full_position",
        "wt_residue", "mut_residue", "full_sequence_len", "window_start",
        "window_end", "local_position", "status",
    ]
    with out_prefix.with_suffix(".supported.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    audit_fields = [
        "gene", "variant_protein", "variant_cdna", "condition", "mechanism_label",
        "drug_response_label", "evidence_level", "evidence_type", "source_id",
        "source_url_or_doi", "status", "reason", "notes",
    ]
    with out_prefix.with_suffix(".audit.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=audit_fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(audit)


def signature_to_row(sig, record: dict) -> dict:
    return {
        "gene": record["gene"],
        "variant_str": record["variant_str"],
        "variant_protein": record["variant_protein"],
        "condition": record["condition"],
        "mechanism_label": record["mechanism_label"],
        "uniprot_id": record["uniprot_id"],
        "full_position": record["full_position"],
        "wt_residue": record["wt_residue"],
        "mut_residue": record["mut_residue"],
        "window_start": record["window_start"],
        "window_end": record["window_end"],
        "local_position": record["local_position"],
        "layer": sig.layer,
        "delta_local": sig.delta_local.astype(np.float32),
        "delta_global": sig.delta_global.astype(np.float32),
        "n_ablated": sig.n_ablated,
        "n_amplified": sig.n_amplified,
        "n_novel": sig.n_novel,
        "total_perturbation": sig.total_perturbation,
        "wt_active_count": sig.wt_active_count,
        "mut_active_count": sig.mut_active_count,
    }


def compute_signatures(records: list[dict], args) -> dict:
    import torch
    from transformers import AutoTokenizer, EsmModel

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0"
    print(f"[score] loading ESM-2 from {args.esm_model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.esm_model)
    esm_model = EsmModel.from_pretrained(args.esm_model, torch_dtype=torch.float16)
    esm_model.to(device).eval()
    out = {}
    for layer in LAYERS:
        ckpt = REPO / CHECKPOINT_DIRS[layer]
        print(f"[score] loading SAE layer {layer}: {ckpt}", flush=True)
        sae = load_our_sae(str(ckpt), device=device)
        for i, rec in enumerate(records):
            wt_window = rec["window_sequence"]
            local_pos = int(rec["local_position"])
            mut_window = wt_window[:local_pos - 1] + rec["mut_residue"] + wt_window[local_pos:]
            variant = MissenseVariant(
                gene=rec["gene"],
                uniprot_id=rec["uniprot_id"],
                position=local_pos,
                wt_residue=rec["wt_residue"],
                mut_residue=rec["mut_residue"],
                clinical_significance=rec["mechanism_label"],
                mechanism=rec["mechanism_label"],
                source="channelopathy_curated",
            )
            sig = compute_perturbation_signature(
                wt_window, mut_window, variant, esm_model, tokenizer, sae, layer, device=device
            )
            key = (rec["gene"], rec["variant_str"], rec["condition"], rec["mechanism_label"])
            out.setdefault(key, []).append(signature_to_row(sig, rec))
            if (i + 1) % 10 == 0 or i == len(records) - 1:
                print(f"  L{layer}: {i+1}/{len(records)}", flush=True)
        del sae
        torch.cuda.empty_cache()
    return out


def load_annotation_metadata():
    meta_by_layer = {}
    for layer in LAYERS:
        pkl = REPO / "r1_encoder_interpretability_benchmark" / "results" / "annotation_alignment" / f"ours_3B_l{layer}_step500000.pkl"
        with pkl.open("rb") as f:
            data = pickle.load(f)
        results = data["results"]
        d_sae = len(results)
        f1 = np.zeros(d_sae, dtype=np.float32)
        cat_w = np.zeros(d_sae, dtype=np.float32)
        ann = [""] * d_sae
        for r in results:
            if not r.alive:
                continue
            f1[r.feature_idx] = r.best_f1 or 0.0
            best = r.best_annotation or ""
            cat = best.split("/")[0] if "/" in best else best
            cat_w[r.feature_idx] = CATEGORY_WEIGHTS.get(cat, 1.0)
            ann[r.feature_idx] = best
        meta_by_layer[layer] = {"f1": f1, "cat_w": cat_w, "ann": ann}
    return meta_by_layer


def build_matrix_from_rows(rows_by_key: dict, meta_by_layer: dict) -> tuple[np.ndarray, list[dict]]:
    selected = {l: np.where(m["f1"] >= F1_THRESHOLD)[0] for l, m in meta_by_layer.items()}
    X_rows = []
    meta_rows = []
    for key, rows in rows_by_key.items():
        by_layer = {r["layer"]: r for r in rows}
        parts = []
        for layer in LAYERS:
            idx = selected[layer]
            if layer not in by_layer:
                parts.append(np.zeros(len(idx) * 3, dtype=np.float32))
                continue
            r = by_layer[layer]
            dl = r["delta_local"][idx].astype(np.float32)
            dg = r["delta_global"][idx].astype(np.float32)
            w = meta_by_layer[layer]["f1"][idx] * meta_by_layer[layer]["cat_w"][idx]
            parts.append(np.concatenate([np.abs(dl), np.abs(dg), dl * w]))
        X_rows.append(np.concatenate(parts))
        first = rows[0]
        meta_rows.append({k: first[k] for k in [
            "gene", "variant_str", "variant_protein", "condition", "mechanism_label",
            "uniprot_id", "full_position", "wt_residue", "mut_residue",
            "window_start", "window_end", "local_position",
        ]})
    X = np.stack(X_rows).astype(np.float32) if X_rows else np.zeros((0, 0), dtype=np.float32)
    return X, meta_rows


def train_reference_classifier(meta_by_layer: dict):
    import importlib.util
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    script = R1 / "scripts" / "16_mechanism_classifier.py"
    spec = importlib.util.spec_from_file_location("r1_mech", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules["r1_mech"] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    os.chdir(REPO)
    labels = module.load_mechanism_labels(str(OUT_DIR / "variant_mechanisms.tsv"))
    classes = ["DN", "GOF", "LOF"]
    X_sae, _llr, y, groups, _col_meta = module.build_features(meta_by_layer, labels, classes)
    good = X_sae.std(0) > 1e-8
    scaler = StandardScaler()
    Xn = scaler.fit_transform(X_sae[:, good])
    clf = LogisticRegression(C=0.1, solver="lbfgs", max_iter=3000, class_weight="balanced")
    clf.fit(Xn, y)
    return clf, scaler, good, classes, {"n_train": int(len(y)), "class_counts": dict(Counter(y.tolist()))}


def predict_and_summarize(rows_by_key: dict, labels: list[dict], out_prefix: Path) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    meta_by_layer = load_annotation_metadata()
    X, meta = build_matrix_from_rows(rows_by_key, meta_by_layer)
    clf, scaler, good, classes, train_info = train_reference_classifier(meta_by_layer)
    probs = clf.predict_proba(scaler.transform(X[:, good]))
    pred = clf.predict(scaler.transform(X[:, good]))
    prob_by_class = {c: list(clf.classes_).index(c) for c in clf.classes_}
    rows = []
    for m, p, pr in zip(meta, pred, probs):
        row = dict(m)
        row["predicted_mechanism"] = str(p)
        for c in classes:
            row[f"prob_{c}"] = float(pr[prob_by_class[c]]) if c in prob_by_class else 0.0
        row["evaluated"] = row["mechanism_label"] in classes
        row["correct"] = bool(row["evaluated"] and row["mechanism_label"] == row["predicted_mechanism"])
        rows.append(row)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "gene", "variant_str", "variant_protein", "condition", "mechanism_label",
        "predicted_mechanism", "prob_DN", "prob_GOF", "prob_LOF", "correct",
        "evaluated", "uniprot_id", "full_position", "window_start", "window_end",
        "local_position",
    ]
    with out_prefix.with_suffix(".predictions.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    eval_rows = [r for r in rows if r["evaluated"]]
    y_true = [r["mechanism_label"] for r in eval_rows]
    y_pred = [r["predicted_mechanism"] for r in eval_rows]
    by_gene = {}
    for gene in sorted({r["gene"] for r in eval_rows}):
        sub = [r for r in eval_rows if r["gene"] == gene]
        by_gene[gene] = {
            "n": len(sub),
            "accuracy": float(sum(r["correct"] for r in sub) / len(sub)) if sub else math.nan,
            "label_counts": dict(Counter(r["mechanism_label"] for r in sub)),
            "pred_counts": dict(Counter(r["predicted_mechanism"] for r in sub)),
        }
    summary = {
        "task": "R1 T1-E channelopathy mechanism concordance",
        "status": "completed",
        "train_info": train_info,
        "n_scored": len(rows),
        "n_evaluated_lof_gof_dn": len(eval_rows),
        "n_mixed_complex_not_evaluated": sum(r["mechanism_label"] == "mixed_complex" for r in rows),
        "accuracy": float(accuracy_score(y_true, y_pred)) if eval_rows else math.nan,
        "macro_f1": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)) if eval_rows else math.nan,
        "confusion_matrix_labels": classes,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist() if eval_rows else [],
        "by_gene": by_gene,
        "prediction_tsv": str(out_prefix.with_suffix(".predictions.tsv")),
    }
    out_prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# R1 T1-E Channelopathy Mechanism Concordance\n"]
    lines.append(f"- Scored variants: {summary['n_scored']}")
    lines.append(f"- Evaluated LOF/GOF/DN variants: {summary['n_evaluated_lof_gof_dn']}")
    lines.append(f"- Mixed/complex scored but excluded: {summary['n_mixed_complex_not_evaluated']}")
    if "prepare" in summary:
        prep = summary["prepare"]
        lines.append(f"- Input curated labels: {prep['n_labels']}")
        lines.append(f"- Supported missense labels: {prep['n_supported_missense']}")
        lines.append(f"- Audit counts: {prep['audit_counts']}")
    lines.append(f"- Accuracy: {summary['accuracy']:.3f}")
    lines.append(f"- Macro F1: {summary['macro_f1']:.3f}")
    lines.append("\n## Confusion Matrix\n")
    lines.append("| True \\ Pred | DN | GOF | LOF |")
    lines.append("|---|---:|---:|---:|")
    for label, row_vals in zip(classes, summary["confusion_matrix"]):
        lines.append(f"| {label} | {row_vals[0]} | {row_vals[1]} | {row_vals[2]} |")
    lines.append("\n## By Gene\n")
    lines.append("| Gene | n | Accuracy | Label counts | Pred counts |")
    lines.append("|---|---:|---:|---|---|")
    for gene, vals in by_gene.items():
        lines.append(f"| {gene} | {vals['n']} | {vals['accuracy']:.3f} | {vals['label_counts']} | {vals['pred_counts']} |")
    lines.append("\n## Interpretation\n")
    lines.append(
        "This does not meet the TODO target of >=80% mechanism concordance. "
        "The main failure mode is dominant-negative channel variants being predicted as LOF; "
        "KCNH2 looks strong mostly because the curated set is LOF-heavy, while KCNQ1 is poor because many pore/N-terminal DN labels collapse to LOF under the current SAE mechanism classifier."
    )
    lines.append("\nRows labeled `mixed_complex` are scored but excluded from the headline LOF/GOF/DN accuracy.\n")
    out_prefix.with_suffix(".md").write_text("\n".join(lines))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=DATA_DIR / "channelopathy_mechanism_positive_labels.tsv")
    ap.add_argument("--swissprot-fasta", type=Path, default=REPO / "data" / "swissprot" / "uniprot_sprot.fasta.gz")
    ap.add_argument("--esm-model", default="/gpfs/jiaotongdamoxing/zhk_zip/models/esm2_t36_3B_UR50D")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=1022)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--reuse-signatures", type=Path, default=None)
    ap.add_argument("--out-prefix", type=Path, default=OUT_DIR / "channelopathy_concordance_20260507")
    args = ap.parse_args()

    t0 = time.time()
    labels = load_labels(args.labels)
    seqs = parse_uniprot_fasta(args.swissprot_fasta, set(GENE_TO_CANONICAL.values()))
    records, audit = prepare_variants(labels, seqs, args.max_len)
    save_prepare_outputs(records, audit, args.out_prefix)
    prep = {
        "n_labels": len(labels),
        "n_supported_missense": len(records),
        "audit_counts": dict(Counter(r["status"] for r in audit)),
        "supported_by_gene": dict(Counter(r["gene"] for r in records)),
        "audit_tsv": str(args.out_prefix.with_suffix(".audit.tsv")),
        "supported_tsv": str(args.out_prefix.with_suffix(".supported.tsv")),
    }
    args.out_prefix.with_suffix(".prepare.json").write_text(json.dumps(prep, indent=2) + "\n")
    print(json.dumps(prep, indent=2), flush=True)
    if args.prepare_only:
        return

    if args.reuse_signatures:
        with args.reuse_signatures.open("rb") as f:
            rows_by_key = pickle.load(f)
    else:
        rows_by_key = compute_signatures(records, args)
        with args.out_prefix.with_suffix(".signatures.pkl").open("wb") as f:
            pickle.dump(rows_by_key, f)
    summary = predict_and_summarize(rows_by_key, labels, args.out_prefix)
    summary["prepare"] = prep
    args.out_prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[done] elapsed_s={time.time() - t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
