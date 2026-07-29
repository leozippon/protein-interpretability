#!/usr/bin/env python
"""Run Foldseek/PDB100 for real-vs-random calibration ESMFold structures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import tarfile
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_FOLDSEEK = Path(os.environ.get(
    "BIOCC_FOLDSEEK",
    "/oss-pvc/zhk_zip/biocc/external_resources/tools/foldseek/bin/foldseek",
))
DEFAULT_ARCHIVE = Path(os.environ.get(
    "BIOCC_FOLDSEEK_PDB100_ARCHIVE",
    "/oss-pvc/zhk_zip/biocc/external_resources/ec_metrics/foldseek/pdb100_20240101.tar.gz",
))
DEFAULT_DB_DIR = Path(os.environ.get(
    "BIOCC_FOLDSEEK_PDB100_DIR",
    "/gpfs/jiaotongdamoxing/zhk_zip/biocc/external_resources/foldseek/pdb100_20240101",
))
CAL_DIR = REPO / "r2_interpretability_transfer" / "results" / "ec_metrics" / "calibration_lysozyme_20260507"
OUT_DIR = REPO / "r2_interpretability_transfer" / "results" / "ec_metrics"

FIELDS = [
    "query", "target", "evalue", "bits", "alntmscore", "qtmscore",
    "ttmscore", "lddt", "rmsd", "prob",
]


def ensure_db(archive: Path, db_dir: Path) -> Path:
    db_path = db_dir / "pdb"
    if (db_dir / "pdb.dbtype").exists() and db_path.exists():
        return db_path
    db_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(db_dir)
    if not (db_dir / "pdb.dbtype").exists():
        raise RuntimeError(f"extracted archive but did not find {db_dir / 'pdb.dbtype'}")
    return db_path


def pdb_files(path: Path, limit: int | None) -> list[Path]:
    files = sorted(path.glob("*.pdb"))
    return files[:limit] if limit else files


def run_foldseek(
    foldseek: Path,
    query_pdbs: list[Path],
    target_db: Path,
    out_tsv: Path,
    tmp_dir: Path,
    threads: int,
    max_seqs: int,
) -> dict:
    if not query_pdbs:
        return {"status": "no_query_pdbs", "returncode": None}
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(foldseek), "easy-search",
        *[str(p) for p in query_pdbs],
        str(target_db), str(out_tsv), str(tmp_dir),
        "--threads", str(threads),
        "--max-seqs", str(max_seqs),
        "--alignment-type", "1",
        "--format-output", ",".join(FIELDS),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {
        "status": "completed" if proc.returncode == 0 else "foldseek_failed",
        "returncode": proc.returncode,
        "elapsed_s": round(time.time() - t0, 1),
        "cmd": cmd,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "out_tsv": str(out_tsv),
    }


def to_float(v: str) -> float:
    try:
        return float(v)
    except ValueError:
        return math.nan


def parse_hits(path: Path) -> tuple[list[dict], dict]:
    hits = []
    if not path.exists():
        return hits, {"n_queries_with_hits": 0, "top_scores": []}
    with path.open() as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < len(FIELDS):
                continue
            rec = dict(zip(FIELDS, row))
            for key in FIELDS[2:]:
                rec[key] = to_float(rec[key])
            hits.append(rec)
    by_query = {}
    for h in hits:
        q = h["query"]
        old = by_query.get(q)
        if old is None or h.get("alntmscore", math.nan) > old.get("alntmscore", math.nan):
            by_query[q] = h
    top = list(by_query.values())
    tm = [h["alntmscore"] for h in top if math.isfinite(h["alntmscore"])]
    lddt = [h["lddt"] for h in top if math.isfinite(h["lddt"])]
    summary = {
        "n_hits": len(hits),
        "n_queries_with_hits": len(top),
        "mean_top_alntmscore": float(sum(tm) / len(tm)) if tm else math.nan,
        "median_top_alntmscore": float(sorted(tm)[len(tm) // 2]) if tm else math.nan,
        "frac_top_tm_ge_0_5": float(sum(x >= 0.5 for x in tm) / len(tm)) if tm else math.nan,
        "frac_top_tm_ge_0_7": float(sum(x >= 0.7 for x in tm) / len(tm)) if tm else math.nan,
        "mean_top_lddt": float(sum(lddt) / len(lddt)) if lddt else math.nan,
        "top_scores": [{"query": h["query"], "alntmscore": h["alntmscore"], "target": h["target"]} for h in top],
        "top_examples": sorted(top, key=lambda h: h.get("alntmscore", math.nan), reverse=True)[:10],
    }
    return hits, summary


def markdown(result: dict) -> str:
    lines = ["# Foldseek/PDB100 Real-vs-Random Calibration\n"]
    lines.append("| Set | Query PDBs | Queries with hits | Mean top TM | TM >= 0.5 | TM >= 0.7 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, vals in result["sets"].items():
        s = vals["summary"]
        lines.append(
            f"| {name} | {vals['n_query_pdbs']} | {s.get('n_queries_with_hits', 0)} | "
            f"{s.get('mean_top_alntmscore', math.nan):.3f} | "
            f"{s.get('frac_top_tm_ge_0_5', math.nan):.3f} | "
            f"{s.get('frac_top_tm_ge_0_7', math.nan):.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--foldseek", type=Path, default=DEFAULT_FOLDSEEK)
    ap.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    ap.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    ap.add_argument("--real-pdb-dir", type=Path, default=Path(os.environ.get("BIOCC_REAL_CAL_PDB_DIR", CAL_DIR / "real_esmfold_pdbs")))
    ap.add_argument("--random-pdb-dir", type=Path, default=Path(os.environ.get("BIOCC_RANDOM_CAL_PDB_DIR", CAL_DIR / "random_esmfold_pdbs")))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--max-seqs", type=int, default=20)
    ap.add_argument("--out-json", type=Path, default=OUT_DIR / "foldseek_calibration_lysozyme_20260507.json")
    args = ap.parse_args()

    target_db = ensure_db(args.archive, args.db_dir)
    work_dir = args.out_json.parent / "foldseek_calibration_lysozyme_20260507"
    sets = {}
    for name, path in [("real_lysozyme", args.real_pdb_dir), ("random_uniref50", args.random_pdb_dir)]:
        queries = pdb_files(path, args.limit)
        out_tsv = work_dir / f"{name}.m8"
        status = run_foldseek(args.foldseek, queries, target_db, out_tsv, work_dir / f"{name}_tmp", args.threads, args.max_seqs)
        _, summary = parse_hits(out_tsv)
        sets[name] = {"query_dir": str(path), "n_query_pdbs": len(queries), "run": status, "summary": summary}
        if status["returncode"] not in {0, None}:
            raise SystemExit(status["returncode"])

    result = {
        "task": "R2 T2-C Foldseek/PDB100 real-vs-random calibration",
        "status": "completed",
        "target_db": str(target_db),
        "sets": sets,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n")
    args.out_json.with_suffix(".md").write_text(markdown(result))
    print(f"Saved {args.out_json}")
    print(f"Saved {args.out_json.with_suffix('.md')}")


if __name__ == "__main__":
    main()
