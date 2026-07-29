#!/usr/bin/env python3
"""N-2: downstream probes for the 38-dimensional universal-triplet basis.

This is the low-risk fallback from OPUS_NEXT_20260513.md.  It asks whether the
conserved triplet activations form a useful protein-level representation even
if individual rich-label annotation is incomplete.

Available-resource version:
- Pfam clan is approximated by dominant Pfam family because no clan map is
  staged in the repo.
- EC top-class is recovered from GO molecular-function annotations whose GO
  terms carry EC cross-references in go-basic.obo.
- Secondary-structure fractions come from Swiss-Prot residue features.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
R2 = REPO / "r2_interpretability_transfer"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


s33 = load_module(SCRIPT_DIR / "33_swissprot_triplet_annotation.py", "swissprot_triplet_annotation_33")
u29 = s33.u29


AA = set("ACDEFGHIKLMNPQRSTVWY")
EC_CLASS_NAMES = {
    "1": "oxidoreductases",
    "2": "transferases",
    "3": "hydrolases",
    "4": "lyases",
    "5": "isomerases",
    "6": "ligases",
    "7": "translocases",
}


def clean_sequence(seq: str) -> str:
    return "".join(c for c in (seq or "").upper() if c in AA)


def dominant_pfam(intervals: list[tuple[int, int, str]]) -> str | None:
    if not intervals:
        return None
    fam, _ = Counter(pfam for _, _, pfam in intervals).most_common(1)[0]
    return fam


def secondary_fractions(features: list[tuple[int, int, str, str, str]], length: int) -> tuple[float, float, float] | None:
    if length <= 0:
        return None
    counts = Counter()
    for start, end, feat_type, _desc, category in features:
        if category != "secondary_structure":
            continue
        span = max(0, min(length, int(end)) - max(1, int(start)) + 1)
        if span <= 0:
            continue
        if feat_type in {"helix", "strand", "turn"}:
            counts[feat_type] += span
    total = counts["helix"] + counts["strand"] + counts["turn"]
    if total < max(20, int(0.2 * length)):
        return None
    return (
        counts["helix"] / length,
        counts["strand"] / length,
        counts["turn"] / length,
    )


def parse_go_ec_topclasses(obo: Path) -> dict[str, set[str]]:
    go_to_ec: dict[str, set[str]] = {}
    cur_id = None
    cur_ec: set[str] = set()

    def flush() -> None:
        if cur_id and cur_ec:
            go_to_ec[cur_id] = set(cur_ec)

    with obo.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "[Term]":
                flush()
                cur_id = None
                cur_ec = set()
                continue
            if line.startswith("id: GO:"):
                cur_id = line.split("id: ", 1)[1]
            elif line.startswith("xref: EC:"):
                ec = line.split("xref: EC:", 1)[1].split()[0]
                top = ec.split(".", 1)[0]
                if top in EC_CLASS_NAMES:
                    cur_ec.add(top)
        flush()
    return go_to_ec


def load_ec_topclass_labels(gaf: Path, go_to_ec: dict[str, set[str]]) -> dict[str, str]:
    labels: dict[str, Counter] = defaultdict(Counter)
    with gzip.open(gaf, "rt", errors="replace") as f:
        for line in f:
            if not line or line.startswith("!"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[0] != "UniProtKB" or parts[8] != "F":
                continue
            acc = parts[1]
            go_id = parts[4]
            for top in go_to_ec.get(go_id, ()):
                labels[acc][top] += 1
    out = {}
    for acc, counts in labels.items():
        if not counts:
            continue
        top_two = counts.most_common(2)
        if len(top_two) > 1 and top_two[0][1] == top_two[1][1]:
            continue
        out[acc] = top_two[0][0]
    return out


def parse_ec_fasta(path: Path, min_len: int, max_len: int) -> dict[str, dict]:
    if not path.exists():
        alt = path.with_name("ec_labeled_swissprot.fasta")
        if alt.exists():
            path = alt
    if not path.exists():
        return {}

    records = {}
    cur_id = None
    cur_ec = None
    cur_lines = []

    def flush() -> None:
        if not cur_id or not cur_ec:
            return
        text = "".join(cur_lines)
        if "<start>" in text and "<end>" in text:
            seq = text.split("<start>", 1)[1].split("<end>", 1)[0]
        elif "<sep>" in text:
            seq = text.split("<sep>", 1)[1]
        else:
            seq = text
        seq = clean_sequence(seq)
        top = cur_ec.split(".", 1)[0]
        if top in EC_CLASS_NAMES and min_len <= len(seq) <= max_len:
            records[cur_id] = {
                "id": cur_id,
                "accession": cur_id,
                "sequence": seq,
                "source": "swissprot_ec_fasta",
                "dominant_pfam": None,
                "ec_topclass": top,
                "secondary_fraction": None,
            }

    with path.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                flush()
                header = line[1:].split()[0]
                if "|" in header:
                    cur_id, cur_ec = header.split("|", 1)
                else:
                    cur_id, cur_ec = header, None
                cur_lines = []
            else:
                cur_lines.append(line)
        flush()
    return records


def build_records(
    swissprot_cache: Path,
    pfam_residue: Path,
    ec_fasta: Path,
    goa_gaf: Path,
    go_obo: Path,
    min_len: int,
    max_len: int,
    pfam_classes: int,
    pfam_per_class: int,
    ec_per_class: int,
    ss_n: int,
    seed: int,
) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    anns = s33.load_swissprot(swissprot_cache)
    pfam = s33.load_pfam_intervals(pfam_residue)
    ec_records = parse_ec_fasta(ec_fasta, min_len, max_len)
    go_to_ec = {}
    ec_labels = {}
    ec_label_source = "data/zymctrl/ec_labeled.fasta"
    if not ec_records:
        go_to_ec = parse_go_ec_topclasses(go_obo)
        ec_labels = load_ec_topclass_labels(goa_gaf, go_to_ec)
        ec_label_source = "GO molecular-function EC xrefs"

    by_acc = {}
    by_pfam: dict[str, list] = defaultdict(list)
    by_ec: dict[str, list] = defaultdict(list)
    ss_candidates = []
    for ann in anns:
        seq = clean_sequence(ann.sequence)
        if not (min_len <= len(seq) <= max_len):
            continue
        acc = ann.accession
        intervals = pfam.get(acc, [])
        dom = dominant_pfam(intervals)
        ss_frac = secondary_fractions(ann.features, len(seq))
        ec = ec_records.get(acc, {}).get("ec_topclass") or ec_labels.get(acc)
        rec = {
            "id": acc,
            "accession": acc,
            "sequence": seq,
            "source": "swissprot_probe",
            "dominant_pfam": dom,
            "ec_topclass": ec,
            "secondary_fraction": ss_frac,
        }
        by_acc[acc] = rec
        if dom:
            by_pfam[dom].append(rec)
        if ec:
            by_ec[ec].append(rec)
        if ss_frac:
            ss_candidates.append(rec)

    for acc, rec in ec_records.items():
        if acc in by_acc:
            continue
        by_acc[acc] = rec
        by_ec[rec["ec_topclass"]].append(rec)

    selected: dict[str, dict] = {}
    pfam_top = [fam for fam, rows in Counter({fam: len(rows) for fam, rows in by_pfam.items()}).most_common(pfam_classes)]
    pfam_task = []
    for fam in pfam_top:
        rows = list(by_pfam[fam])
        rng.shuffle(rows)
        for rec in rows[:pfam_per_class]:
            selected[rec["accession"]] = rec
            pfam_task.append(rec["accession"])

    ec_task = []
    for ec in sorted(EC_CLASS_NAMES):
        rows = list(by_ec.get(ec, []))
        rng.shuffle(rows)
        for rec in rows[:ec_per_class]:
            selected[rec["accession"]] = rec
            ec_task.append(rec["accession"])

    rng.shuffle(ss_candidates)
    ss_task = []
    for rec in ss_candidates[:ss_n]:
        selected[rec["accession"]] = rec
        ss_task.append(rec["accession"])

    records = list(selected.values())
    if ec_label_source == "data/zymctrl/ec_labeled.fasta":
        ec_note = "EC top-class labels are read from the staged EC-labeled Swiss-Prot FASTA."
    else:
        ec_note = "EC top-class labels are inferred from GO molecular-function terms with EC xrefs in go-basic.obo."

    meta = {
        "n_swissprot_total": len(anns),
        "n_length_filtered_unique": len(by_acc),
        "ec_label_source": ec_label_source,
        "n_go_terms_with_ec_xref": len(go_to_ec),
        "n_ec_labeled_length_filtered": sum(1 for r in by_acc.values() if r["ec_topclass"]),
        "n_pfam_families_length_filtered": len(by_pfam),
        "n_records_union": len(records),
        "pfam_task_n": len(pfam_task),
        "pfam_task_classes": len(set(by_acc[a]["dominant_pfam"] for a in pfam_task)),
        "ec_task_n": len(ec_task),
        "ec_task_class_counts": dict(Counter(by_acc[a]["ec_topclass"] for a in ec_task)),
        "ss_task_n": len(ss_task),
        "resource_deviations": [
            "Pfam clan is approximated by dominant Pfam family; no clan map was staged.",
            ec_note,
        ],
    }
    for rec in records:
        rec["tasks"] = {
            "pfam_family": rec["accession"] in pfam_task,
            "ec_topclass": rec["accession"] in ec_task,
            "secondary_fraction": rec["accession"] in ss_task,
        }
    return records, meta


def triplet_matrix(records: list[dict], triplets: list[dict], specs: list[tuple[str, str]], device: str) -> tuple[np.ndarray, list[str]]:
    model_values = {}
    for model_name, ckpt in specs:
        vals = u29.collect_model_values(model_name, ckpt, records, triplets, device)
        if vals:
            model_values[model_name] = vals

    X = np.full((len(records), len(triplets)), np.nan, dtype=np.float32)
    for j, triplet in enumerate(triplets):
        tid = triplet["triplet_id"]
        per_model = []
        for model_name, values_by_triplet in model_values.items():
            vals = values_by_triplet.get(tid)
            if not vals:
                continue
            finite = np.concatenate([x[np.isfinite(x)] for x in vals if np.isfinite(x).any()])
            if finite.size == 0:
                continue
            mu = float(finite.mean())
            sd = float(finite.std() + 1e-6)
            per_model.append([(x - mu) / sd for x in vals])
        if not per_model:
            continue
        for i in range(len(records)):
            arrays = [m[i] for m in per_model if i < len(m)]
            if not arrays:
                continue
            consensus = np.nanmean(np.stack(arrays, axis=0), axis=0)
            finite = consensus[np.isfinite(consensus)]
            if finite.size:
                X[i, j] = float(np.percentile(finite, 95))
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, [t["triplet_id"] for t in triplets]


@torch.no_grad()
def esm2_matrix(records: list[dict], model_path: Path, device: str, max_length: int, batch_size: int) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_path), local_files_only=True)
    model.to(device).eval()
    if device.startswith("cuda"):
        model.half()

    rows = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        seqs = [" ".join(list(rec["sequence"][:max_length])) for rec in batch]
        toks = tokenizer(seqs, return_tensors="pt", padding=True, truncation=True, max_length=max_length + 2).to(device)
        with torch.amp.autocast("cuda", enabled=device.startswith("cuda"), dtype=torch.float16):
            out = model(**toks)
        hidden = out.last_hidden_state.float()
        attn = toks["attention_mask"]
        for bi, rec in enumerate(batch):
            n = min(len(rec["sequence"]), max_length)
            vec = hidden[bi, 1 : n + 1].mean(dim=0).detach().cpu().numpy()
            rows.append(vec.astype(np.float32))
        print(f"  ESM2 embeddings: {min(start + len(batch), len(records))}/{len(records)}", flush=True)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return np.stack(rows)


def classification_probe(X: np.ndarray, y: list[str], seed: int) -> dict:
    labels = sorted(set(y))
    label_to_i = {label: i for i, label in enumerate(labels)}
    yi = np.asarray([label_to_i[x] for x in y], dtype=np.int32)
    min_class = min(Counter(yi).values())
    n_splits = max(2, min(5, min_class))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs", class_weight="balanced"),
    )
    pred = cross_val_predict(clf, X, yi, cv=cv, method="predict")
    return {
        "n": int(len(yi)),
        "n_classes": len(labels),
        "n_splits": int(n_splits),
        "labels": labels,
        "accuracy": float(accuracy_score(yi, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(yi, pred)),
        "macro_f1": float(f1_score(yi, pred, average="macro")),
    }


def regression_probe(X: np.ndarray, Y: np.ndarray, seed: int) -> dict:
    n_splits = max(2, min(5, len(Y)))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    clf = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    pred = cross_val_predict(clf, X, Y, cv=cv)
    return {
        "n": int(len(Y)),
        "n_targets": int(Y.shape[1]),
        "n_splits": int(n_splits),
        "r2_uniform_average": float(r2_score(Y, pred, multioutput="uniform_average")),
        "mae_uniform_average": float(mean_absolute_error(Y, pred, multioutput="uniform_average")),
        "target_names": ["helix_fraction", "strand_fraction", "turn_fraction"],
    }


def run_task(records: list[dict], X_triplet: np.ndarray, X_esm: np.ndarray, task: str, seed: int) -> dict:
    idx = [i for i, rec in enumerate(records) if rec["tasks"].get(task)]
    if task == "pfam_family":
        y = [records[i]["dominant_pfam"] for i in idx]
        return {
            "task": task,
            "label_source": "dominant Pfam family proxy",
            "triplet_basis": classification_probe(X_triplet[idx], y, seed),
            "esm2_mean_pooled": classification_probe(X_esm[idx], y, seed + 11),
        }
    if task == "ec_topclass":
        y = [records[i]["ec_topclass"] for i in idx]
        return {
            "task": task,
            "label_source": "staged EC-labeled Swiss-Prot FASTA (data/zymctrl/ec_labeled.fasta)",
            "class_names": EC_CLASS_NAMES,
            "triplet_basis": classification_probe(X_triplet[idx], y, seed),
            "esm2_mean_pooled": classification_probe(X_esm[idx], y, seed + 11),
        }
    if task == "secondary_fraction":
        Y = np.asarray([records[i]["secondary_fraction"] for i in idx], dtype=np.float32)
        return {
            "task": task,
            "label_source": "Swiss-Prot secondary_structure residue features",
            "triplet_basis": regression_probe(X_triplet[idx], Y, seed),
            "esm2_mean_pooled": regression_probe(X_esm[idx], Y, seed + 11),
        }
    raise ValueError(task)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--triplets", type=Path, default=R2 / "results/circuit_analysis/universal_atlas_balanced200_wide_triplets_20260512.tsv")
    ap.add_argument("--swissprot-cache", type=Path, default=REPO / "data/processed/swissprot_all_max1022.pkl")
    ap.add_argument("--pfam-residue", type=Path, default=REPO / "data/interpro/pfam_residue.tsv")
    ap.add_argument("--ec-fasta", type=Path, default=REPO / "data/zymctrl/ec_labeled.fasta")
    ap.add_argument("--goa-gaf", type=Path, default=REPO / "data/go/goa_uniprot_all.gaf.gz")
    ap.add_argument("--go-obo", type=Path, default=REPO / "data/go/go-basic.obo")
    ap.add_argument("--esm2-model", type=Path, default=Path("/gpfs/jiaotongdamoxing/zhk_zip/models/esm2_t36_3B_UR50D"))
    ap.add_argument("--max-triplets", type=int, default=38)
    ap.add_argument("--min-len", type=int, default=100)
    ap.add_argument("--max-len", type=int, default=400)
    ap.add_argument("--pfam-classes", type=int, default=20)
    ap.add_argument("--pfam-per-class", type=int, default=12)
    ap.add_argument("--ec-per-class", type=int, default=40)
    ap.add_argument("--ss-n", type=int, default=300)
    ap.add_argument("--esm-batch-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260513)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", type=Path, default=R2 / "results/circuit_analysis/triplet_basis_probes_20260513")
    ap.add_argument("--model-spec", action="append", default=[
        "protgpt2=/oss-pvc/zhk_zip/outputs/research2/clt_weights/protgpt2_v2/step_200000",
        "zymctrl=/oss-pvc/zhk_zip/outputs/research2/clt_weights/zymctrl_v2/step_200000",
        "progen2-medium=/gpfs/jiaotongdamoxing/zhk_zip/biocc/paper_r2_nature_mi/results/final_checkpoints/r2_clt_progen2_medium_rerun_20260403/clt_weights/progen2-medium/step_100000",
    ])
    args = ap.parse_args()

    os.environ.setdefault("R2_MODEL_BASE_DIR", "/gpfs/jiaotongdamoxing/zhk_zip/models")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("N-2 triplet-basis probes", flush=True)
    records, meta = build_records(
        args.swissprot_cache,
        args.pfam_residue,
        args.ec_fasta,
        args.goa_gaf,
        args.go_obo,
        args.min_len,
        args.max_len,
        args.pfam_classes,
        args.pfam_per_class,
        args.ec_per_class,
        args.ss_n,
        args.seed,
    )
    (args.out_dir / "cohort.json").write_text(json.dumps({"meta": meta, "records": records}, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)

    triplets = u29.read_triplets(args.triplets, args.max_triplets)
    specs = [u29.parse_model_spec(s) for s in args.model_spec]
    X_triplet, triplet_ids = triplet_matrix(records, triplets, specs, args.device)
    np.save(args.out_dir / "triplet_basis.npy", X_triplet)
    (args.out_dir / "triplet_basis_columns.json").write_text(json.dumps(triplet_ids, indent=2) + "\n")

    X_esm = esm2_matrix(records, args.esm2_model, args.device, args.max_len, args.esm_batch_size)
    np.save(args.out_dir / "esm2_mean_pooled.npy", X_esm)

    tasks = []
    for task in ["pfam_family", "ec_topclass", "secondary_fraction"]:
        tasks.append(run_task(records, X_triplet, X_esm, task, args.seed))

    summary = {
        "task": "N-2 38-dimensional universal-triplet-basis probes",
        "status": "completed",
        "runtime_seconds": time.time() - t0,
        "cohort_meta": meta,
        "triplet_basis_shape": list(X_triplet.shape),
        "esm2_mean_pooled_shape": list(X_esm.shape),
        "tasks": tasks,
        "acceptance_gate": "triplet representation matches ESM2 on at least one of three tasks; CI not computed in this quick probe.",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# N-2 Triplet Basis Probes",
        "",
        f"- Records in union cohort: {meta['n_records_union']}",
        f"- Triplet basis shape: {list(X_triplet.shape)}",
        f"- ESM2 mean-pooled shape: {list(X_esm.shape)}",
        "",
        "| Task | Label source | Triplet metric | ESM2 metric |",
        "|---|---|---:|---:|",
    ]
    for task in tasks:
        if task["task"] == "secondary_fraction":
            t_metric = task["triplet_basis"]["r2_uniform_average"]
            e_metric = task["esm2_mean_pooled"]["r2_uniform_average"]
            metric_name = "R2"
        else:
            t_metric = task["triplet_basis"]["macro_f1"]
            e_metric = task["esm2_mean_pooled"]["macro_f1"]
            metric_name = "macro-F1"
        lines.append(f"| {task['task']} ({metric_name}) | {task['label_source']} | {t_metric:.4f} | {e_metric:.4f} |")
    lines += [
        "",
        "## Evidence Boundary",
        "",
        "- Pfam clan is approximated by dominant Pfam family because no Pfam clan map is staged.",
        "- EC labels use the staged EC-labeled Swiss-Prot FASTA when available; GO EC-xref inference is only a fallback.",
        "- This is a low-dimensional representation fallback, not a biological naming result for individual triplets.",
        "",
    ]
    (args.out_dir / "summary.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
