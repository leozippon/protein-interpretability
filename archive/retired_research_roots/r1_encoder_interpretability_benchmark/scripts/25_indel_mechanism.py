#!/usr/bin/env python
"""T1-C indel / frameshift mechanism transfer experiment.

This script transfers the R1 missense LOF/GOF/DN classifier to reconstructed
ClinVar protein indels. It is intentionally resume-friendly:

1. ``prepare`` reconstructs WT/mutant protein sequences from the staged
   ClinVar indel TSV and writes compact JSONL records.
2. ``fit`` trains the missense mechanism classifier once and saves a small
   sklearn cache, avoiding the need to copy the 1.3GB missense signature pkl to
   H200.
3. ``predict`` computes SAE perturbation features for indel records on GPU and
   appends predictions to JSONL after every batch.

The indel local feature is a heuristic affected-region average, not a direct
missense position delta. Treat this as a transfer/diagnostic experiment.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
LAYERS = [19, 23, 27, 31, 35]
AA = set("ACDEFGHIKLMNPQRSTVWY")


def load_script_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def clean_sequence(seq: str) -> str:
    return "".join(c for c in (seq or "").upper() if c in AA)


def parse_layers(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def iter_jsonl(path: str):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


def write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def prepare_records(args) -> None:
    sys.path.insert(0, str(ROOT))
    indel_mod = load_script_module(ROOT / "scripts" / "20_indel_extension.py", "indel_extension_20")
    seq_map = indel_mod.build_uniprot_sequence_map(args.swissprot_cache)

    n_seen = n_written = 0
    counts = Counter()
    labels = Counter()
    label_filter = set(args.labels.split(",")) if args.labels else set()
    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    with open(args.indels) as f, open(args.out_jsonl, "w") as out:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            n_seen += 1
            if row.get("supported") != "True":
                continue
            if label_filter and row.get("label") not in label_filter:
                continue
            wt_seq = seq_map.get(row.get("uniprot_id", ""))
            if not wt_seq:
                continue
            recon = indel_mod.reconstruct_variant(wt_seq, row["protein_hgvs"])
            if not recon.get("supported"):
                continue
            mut_seq = clean_sequence(recon["mut_sequence"])
            wt_seq = clean_sequence(wt_seq)
            if len(wt_seq) < args.min_len or len(mut_seq) < args.min_len:
                continue
            if len(wt_seq) > args.max_len or len(mut_seq) > args.max_len:
                continue
            rec = {
                "idx": n_written,
                "source_row": n_seen - 1,
                "gene": row["gene"],
                "uniprot_id": row["uniprot_id"],
                "protein_hgvs": row["protein_hgvs"],
                "variant_class": row["variant_class"],
                "label": row.get("label", ""),
                "clinical_significance": row.get("clinical_significance", ""),
                "truncating": row.get("truncating") == "True",
                "wt_seq": wt_seq,
                "mut_seq": mut_seq,
                "start": int(recon.get("start", 1) or 1),
                "end": int(recon.get("end", recon.get("start", 1)) or 1),
                "inserted_sequence": recon.get("inserted_sequence", ""),
                "length_delta": len(mut_seq) - len(wt_seq),
            }
            out.write(json.dumps(rec, separators=(",", ":")) + "\n")
            n_written += 1
            counts[rec["variant_class"]] += 1
            labels[rec["label"]] += 1
            if args.max_records and n_written >= args.max_records:
                break

    summary = {
        "task": "T1-C prepare indel records",
        "input_rows_seen": n_seen,
        "n_records": n_written,
        "variant_class_counts": dict(counts),
        "label_counts": dict(labels),
        "filters": {
            "min_len": args.min_len,
            "max_len": args.max_len,
            "labels": args.labels,
            "max_records": args.max_records,
        },
        "out_jsonl": args.out_jsonl,
    }
    write_json(args.summary_out, summary)
    print(json.dumps(summary, indent=2))


def fit_classifier(args) -> None:
    os.chdir(REPO)
    sys.path.insert(0, str(ROOT))
    mech = load_script_module(ROOT / "scripts" / "16_mechanism_classifier.py", "mechanism_classifier_16")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    labels = mech.load_mechanism_labels(args.mechanisms)
    class_counts = Counter(labels.values())
    keep_classes = sorted(c for c, n in class_counts.items() if n >= args.min_per_class and c != "UNLABELED")
    meta_by_layer = mech.load_annotation_metadata()
    X, _llr, y, _groups, col_meta = mech.build_features(meta_by_layer, labels, keep_classes)
    good = X.std(0) > 1e-8
    X = X[:, good]
    col_meta = [col_meta[i] for i, keep in enumerate(good) if keep]

    scaler = StandardScaler()
    Xn = scaler.fit_transform(X)
    clf = LogisticRegression(C=args.C, solver="lbfgs", max_iter=3000, class_weight="balanced")
    clf.fit(Xn, y)

    selected_by_layer: dict[int, list[int]] = {}
    weights_by_layer: dict[int, list[float]] = {}
    for layer in LAYERS:
        feats = sorted({int(feat_idx) for l, feat_idx, _kind in col_meta if l == layer})
        selected_by_layer[layer] = feats
        f1 = meta_by_layer[layer]["f1"]
        cat_w = meta_by_layer[layer]["cat_w"]
        weights_by_layer[layer] = [float(f1[i] * cat_w[i]) for i in feats]

    cache = {
        "task": "T1-C indel mechanism classifier cache",
        "classes": list(clf.classes_),
        "layers": LAYERS,
        "col_meta": col_meta,
        "selected_by_layer": selected_by_layer,
        "weights_by_layer": weights_by_layer,
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
        "clf_coef": clf.coef_.astype(np.float32),
        "clf_intercept": clf.intercept_.astype(np.float32),
        "n_train": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "class_counts_raw": dict(class_counts),
        "keep_classes": keep_classes,
        "mechanisms": args.mechanisms,
    }
    os.makedirs(os.path.dirname(args.out_cache) or ".", exist_ok=True)
    with open(args.out_cache, "wb") as f:
        pickle.dump(cache, f)
    print(f"Saved classifier cache: {args.out_cache}")
    print(f"n_train={X.shape[0]} n_features={X.shape[1]} classes={list(clf.classes_)}")


def affected_region_delta(wt_feats: np.ndarray, mut_feats: np.ndarray, rec: dict, window: int) -> np.ndarray:
    start = max(int(rec.get("start", 1)) - 1, 0)
    end = max(int(rec.get("end", start + 1)), start + 1)
    ins_len = len(rec.get("inserted_sequence") or "")
    wt_a = max(0, start - window)
    wt_b = min(wt_feats.shape[0], end + window)
    mut_bare_len = max(1, end - start + ins_len)
    mut_a = max(0, start - window)
    mut_b = min(mut_feats.shape[0], start + mut_bare_len + window)
    if wt_a >= wt_b or mut_a >= mut_b:
        n = min(wt_feats.shape[0], mut_feats.shape[0])
        return mut_feats[:n].mean(0) - wt_feats[:n].mean(0)
    return mut_feats[mut_a:mut_b].mean(0) - wt_feats[wt_a:wt_b].mean(0)


def build_prediction_row(layer_outputs: dict[int, tuple[np.ndarray, np.ndarray]], rec: dict, cache: dict, window: int) -> tuple[np.ndarray, dict]:
    per_layer = {}
    diagnostics = {"layer_damage": {}}
    selected_by_layer = {int(k): v for k, v in cache["selected_by_layer"].items()}
    weights_by_layer = {int(k): np.array(v, dtype=np.float32) for k, v in cache["weights_by_layer"].items()}
    for layer in cache["layers"]:
        wt_feats, mut_feats = layer_outputs[layer]
        idx = np.array(selected_by_layer[layer], dtype=np.int64)
        weights = weights_by_layer[layer]
        local = affected_region_delta(wt_feats, mut_feats, rec, window)
        global_delta = np.abs(mut_feats.mean(0) - wt_feats.mean(0))
        weight_map = {int(feat): float(w) for feat, w in zip(selected_by_layer[layer], weights)}
        per_layer[layer] = {
            "local": local,
            "global": global_delta,
            "weight_map": weight_map,
        }
        diagnostics["layer_damage"][str(layer)] = {
            "local_l1_selected": float(np.abs(local[idx]).sum()),
            "global_l1_selected": float(np.abs(global_delta[idx]).sum()),
        }

    cols = []
    for layer, feat_idx, kind in cache["col_meta"]:
        layer = int(layer)
        feat_idx = int(feat_idx)
        vals = per_layer[layer]
        if kind == "abs_local":
            value = abs(float(vals["local"][feat_idx]))
        elif kind == "abs_global":
            value = abs(float(vals["global"][feat_idx]))
        elif kind == "weighted_local":
            value = float(vals["local"][feat_idx]) * vals["weight_map"].get(feat_idx, 0.0)
        else:
            raise ValueError(f"unknown feature kind in cache col_meta: {kind}")
        cols.append(value)
    return np.array(cols, dtype=np.float32), diagnostics


def predict_proba_from_cache(cache: dict, x: np.ndarray) -> np.ndarray:
    mean = np.asarray(cache["scaler_mean"], dtype=np.float32)
    scale = np.asarray(cache["scaler_scale"], dtype=np.float32)
    coef = np.asarray(cache["clf_coef"], dtype=np.float32)
    intercept = np.asarray(cache["clf_intercept"], dtype=np.float32)
    x = x.astype(np.float32, copy=False)
    z = (x - mean) / np.where(scale == 0, 1.0, scale)
    logits = coef @ z + intercept
    logits = logits - float(np.max(logits))
    exp = np.exp(logits)
    return exp / float(exp.sum())


def summarize_predictions(records: list[dict], path: str, args) -> None:
    from sklearn.metrics import roc_auc_score

    labels = [r.get("label", "") for r in records]
    is_binary = np.array([x in {"pathogenic", "benign"} for x in labels], dtype=bool)
    pathogenic = np.array([x == "pathogenic" for x in labels], dtype=np.int32)
    damage = np.array([r.get("damage_score", np.nan) for r in records], dtype=np.float32)
    auc = None
    if is_binary.sum() >= 20 and len(set(pathogenic[is_binary].tolist())) == 2:
        auc = float(roc_auc_score(pathogenic[is_binary], damage[is_binary]))
    summary = {
        "task": "T1-C indel mechanism transfer",
        "status": "completed",
        "n_records": len(records),
        "label_counts": dict(Counter(labels)),
        "variant_class_counts": dict(Counter(r.get("variant_class", "") for r in records)),
        "predicted_mechanism_counts": dict(Counter(r.get("predicted_mechanism", "") for r in records)),
        "pathogenicity_auc_damage": auc,
        "config": {k: v for k, v in vars(args).items() if k != "func"},
    }
    write_json(path, summary)


def predict(args) -> None:
    import torch
    from transformers import AutoTokenizer, EsmForMaskedLM

    sys.path.insert(0, str(ROOT))
    from src.analysis.feature_annotation import load_our_sae

    with open(args.classifier_cache, "rb") as f:
        cache = pickle.load(f)
    layers = parse_layers(args.layers)
    if layers != list(cache["layers"]):
        raise ValueError(
            "This classifier cache was trained with all production layers; "
            f"got --layers={layers}, expected {cache['layers']}"
        )

    records = list(iter_jsonl(args.records))
    if args.max_records:
        records = records[:args.max_records]
    done = set()
    existing = []
    if os.path.exists(args.out_jsonl) and not args.force:
        existing = list(iter_jsonl(args.out_jsonl))
        done = {int(r["idx"]) for r in existing}
    todo = [r for r in records if int(r["idx"]) not in done]

    print("=" * 70)
    print("  T1-C indel mechanism transfer")
    print("=" * 70)
    print(f"records={len(records)} done={len(done)} todo={len(todo)} layers={layers}")

    if not todo:
        summarize_predictions(existing, args.summary_out, args)
        return

    tokenizer = AutoTokenizer.from_pretrained(args.esm_path)
    model = EsmForMaskedLM.from_pretrained(args.esm_path, torch_dtype=torch.float16)
    model = model.to(args.device).eval()

    classes = list(cache["classes"])
    all_new = []
    t0 = time.time()

    for start in range(0, len(todo), args.batch_size):
        batch = todo[start:start + args.batch_size]
        layer_outputs = {int(r["idx"]): {} for r in batch}
        seqs = []
        owners = []
        for r in batch:
            seqs.append(r["wt_seq"][:args.max_len])
            owners.append((int(r["idx"]), "wt"))
            seqs.append(r["mut_seq"][:args.max_len])
            owners.append((int(r["idx"]), "mut"))

        for layer in layers:
            ckpt = Path(args.checkpoint_root) / f"layer_{layer}" / "step_500000"
            print(f"batch {start // args.batch_size + 1}: layer {layer} {ckpt}", flush=True)
            sae = load_our_sae(str(ckpt), device=args.device)
            feats_by_owner = {}
            for s0 in range(0, len(seqs), args.forward_batch_size):
                sub = seqs[s0:s0 + args.forward_batch_size]
                spaced = [" ".join(s) for s in sub]
                enc = tokenizer(
                    spaced, padding=True, truncation=True,
                    max_length=args.max_len + 2, return_tensors="pt",
                ).to(args.device)
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                    out = model(
                        input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"],
                        output_hidden_states=True,
                    )
                acts = out.hidden_states[layer + 1].float()
                for j, seq in enumerate(sub):
                    owner = owners[s0 + j]
                    aa_acts = acts[j, 1:len(seq) + 1]
                    feats_by_owner[owner] = sae(aa_acts).f.detach().cpu().numpy()
            for r in batch:
                rid = int(r["idx"])
                layer_outputs[rid][layer] = (
                    feats_by_owner[(rid, "wt")],
                    feats_by_owner[(rid, "mut")],
                )
            del sae
            torch.cuda.empty_cache()

        out_rows = []
        for r in batch:
            rid = int(r["idx"])
            x, diag = build_prediction_row(layer_outputs[rid], r, cache, args.local_window)
            prob = predict_proba_from_cache(cache, x)
            pred = classes[int(np.argmax(prob))]
            damage = float(sum(v["local_l1_selected"] + v["global_l1_selected"] for v in diag["layer_damage"].values()))
            row = {
                "idx": rid,
                "gene": r["gene"],
                "uniprot_id": r["uniprot_id"],
                "protein_hgvs": r["protein_hgvs"],
                "variant_class": r["variant_class"],
                "label": r.get("label", ""),
                "truncating": bool(r.get("truncating", False)),
                "wt_len": len(r["wt_seq"]),
                "mut_len": len(r["mut_seq"]),
                "length_delta": int(r.get("length_delta", len(r["mut_seq"]) - len(r["wt_seq"]))),
                "predicted_mechanism": pred,
                "mechanism_probs": {c: float(p) for c, p in zip(classes, prob)},
                "damage_score": damage,
                **diag,
            }
            out_rows.append(row)
        append_jsonl(args.out_jsonl, out_rows)
        all_new.extend(out_rows)
        print(f"wrote {len(all_new)}/{len(todo)} new records elapsed={time.time() - t0:.1f}s", flush=True)

    all_rows = existing + all_new
    summarize_predictions(all_rows, args.summary_out, args)
    print(f"Saved predictions: {args.out_jsonl}")
    print(f"Saved summary: {args.summary_out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--indels", default="r1_encoder_interpretability_benchmark/results/variant_effect/clinvar_indels.tsv")
    p.add_argument("--swissprot-cache", default="data/processed/swissprot_all_max1022.pkl")
    p.add_argument("--out-jsonl", default="r1_encoder_interpretability_benchmark/results/variant_effect/indel_records_supported_20260504.jsonl")
    p.add_argument("--summary-out", default="r1_encoder_interpretability_benchmark/results/variant_effect/indel_records_supported_20260504_summary.json")
    p.add_argument("--labels", default="pathogenic,benign")
    p.add_argument("--min-len", type=int, default=20)
    p.add_argument("--max-len", type=int, default=1022)
    p.add_argument("--max-records", type=int, default=0)
    p.set_defaults(func=prepare_records)

    p = sub.add_parser("fit")
    p.add_argument("--mechanisms", default="r1_encoder_interpretability_benchmark/results/variant_effect/variant_mechanisms.tsv")
    p.add_argument("--out-cache", default="r1_encoder_interpretability_benchmark/results/variant_effect/indel_mechanism_classifier_20260504.pkl")
    p.add_argument("--min-per-class", type=int, default=15)
    p.add_argument("--C", type=float, default=0.1)
    p.set_defaults(func=fit_classifier)

    p = sub.add_parser("predict")
    p.add_argument("--records", default="r1_encoder_interpretability_benchmark/results/variant_effect/indel_records_supported_20260504.jsonl")
    p.add_argument("--classifier-cache", default="r1_encoder_interpretability_benchmark/results/variant_effect/indel_mechanism_classifier_20260504.pkl")
    p.add_argument("--esm-path", default="/oss-pvc/zhk_zip/models/esm2_t36_3B_UR50D")
    p.add_argument("--checkpoint-root", default="/gpfs/jiaotongdamoxing/zhk_zip/biocc/Research1/results/final_checkpoints/r1_h200_2gpu_20260401/sae_weights")
    p.add_argument("--layers", default="19,23,27,31,35")
    p.add_argument("--out-jsonl", default="r1_encoder_interpretability_benchmark/results/variant_effect/indel_mechanism_predictions_20260504.jsonl")
    p.add_argument("--summary-out", default="r1_encoder_interpretability_benchmark/results/variant_effect/indel_mechanism_predictions_20260504_summary.json")
    p.add_argument("--max-records", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--forward-batch-size", type=int, default=2)
    p.add_argument("--max-len", type=int, default=1022)
    p.add_argument("--local-window", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=predict)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
