#!/usr/bin/env python
"""Run HMMER/Pfam scans for generated R2 sequences.

This is the first real external-metric check for generated sequences after
staging HMMER and Pfam-A. It deliberately reports the observed domain hit rate
instead of replacing missing CLEAN/Foldseek metrics with a heuristic.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

AA = set("ACDEFGHIKLMNPQRSTVWY")
LYSOZYME_PATTERNS = [
    re.compile(p, re.I)
    for p in ["lyso", "muramid", "glyco", "transglycos", "amidase", "peptidoglycan", "hydrolase"]
]


def clean_seq(seq: str) -> str:
    return "".join(c for c in str(seq).upper() if c in AA)


def load_records(path: Path, max_records: int | None = None) -> list[dict]:
    data = json.loads(path.read_text())
    records = []

    def add_many(items, source):
        if not isinstance(items, list):
            return
        for i, r in enumerate(items):
            if isinstance(r, str):
                seq = r
                meta = {}
            elif isinstance(r, dict):
                seq = r.get("sequence") or r.get("steered") or r.get("raw_output") or r.get("unsteered")
                meta = r
            else:
                continue
            seq = clean_seq(seq)
            if len(seq) < 30:
                continue
            src = str(meta.get("source", source)) if isinstance(meta, dict) else source
            rid = str(meta.get("id", f"{src}_{len(records):04d}")) if isinstance(meta, dict) else f"{src}_{len(records):04d}"
            records.append({"id": rid, "source": src, "sequence": seq, "meta": meta})

    add_many(data.get("leads"), "steered_leads")
    add_many(data.get("all_records"), "steered_all")
    add_many(data.get("unsteered_baseline"), "unsteered")
    add_many(data.get("records"), "records")
    if max_records is not None:
        records = records[:max_records]
    return records


def write_fasta(records: list[dict], path: Path) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(f">{r['id']} source={r['source']} len={len(r['sequence'])}\n")
            seq = r["sequence"]
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")


def parse_domtblout(path: Path, evalue_cutoff: float) -> dict[str, list[dict]]:
    hits = defaultdict(list)
    for line in path.read_text(errors="replace").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=22)
        if len(parts) < 22:
            continue
        target_name = parts[0]
        target_acc = parts[1]
        query_name = parts[3]
        full_e = float(parts[6])
        full_score = float(parts[7])
        i_e = float(parts[12])
        dom_score = float(parts[13])
        ali_from = int(parts[17])
        ali_to = int(parts[18])
        desc = parts[22] if len(parts) > 22 else ""
        if i_e > evalue_cutoff and full_e > evalue_cutoff:
            continue
        hits[query_name].append({
            "pfam_name": target_name,
            "pfam_accession": target_acc,
            "full_evalue": full_e,
            "full_score": full_score,
            "domain_ievalue": i_e,
            "domain_score": dom_score,
            "ali_from": ali_from,
            "ali_to": ali_to,
            "description": desc,
            "lysozyme_like": any(p.search(" ".join([target_name, target_acc, desc])) for p in LYSOZYME_PATTERNS),
        })
    for q in hits:
        hits[q].sort(key=lambda h: (h["domain_ievalue"], -h["domain_score"]))
    return dict(hits)


def summarize(records: list[dict], hits: dict[str, list[dict]]) -> dict:
    by_source = defaultdict(lambda: {"n": 0, "n_with_hit": 0, "n_with_lysozyme_like_hit": 0, "top_domains": Counter()})
    examples = []
    for r in records:
        src = r["source"]
        hs = hits.get(r["id"], [])
        by_source[src]["n"] += 1
        if hs:
            by_source[src]["n_with_hit"] += 1
            by_source[src]["top_domains"][hs[0]["pfam_name"]] += 1
        if any(h["lysozyme_like"] for h in hs):
            by_source[src]["n_with_lysozyme_like_hit"] += 1
        if hs and len(examples) < 20:
            examples.append({"id": r["id"], "source": src, "length": len(r["sequence"]), "top_hits": hs[:3]})
    out = {}
    for src, vals in by_source.items():
        n = vals["n"]
        out[src] = {
            "n": n,
            "n_with_hit": vals["n_with_hit"],
            "hit_rate": vals["n_with_hit"] / max(n, 1),
            "n_with_lysozyme_like_hit": vals["n_with_lysozyme_like_hit"],
            "lysozyme_like_hit_rate": vals["n_with_lysozyme_like_hit"] / max(n, 1),
            "top_domains": vals["top_domains"].most_common(10),
        }
    return {"by_source": out, "examples": examples}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--pfam", type=Path, default=Path(os.environ.get("BIOCC_PFAM_A_HMM", "/oss-pvc/zhk_zip/biocc/external_resources/ec_metrics/pfam/Pfam-A.hmm")))
    ap.add_argument("--hmmscan", default=os.environ.get("HMM_SCAN", "hmmscan"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, default=Path("r2_interpretability_transfer/results/ec_metrics/pfam_scan_20260504"))
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--cpu", type=int, default=8)
    ap.add_argument("--evalue", type=float, default=1e-5)
    args = ap.parse_args()

    records = load_records(args.input, args.max_records)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    fasta = args.work_dir / "generated_sequences.fasta"
    domtbl = args.work_dir / "generated_sequences.domtblout"
    tblout = args.work_dir / "generated_sequences.tblout"
    write_fasta(records, fasta)

    cmd = [args.hmmscan, "--cpu", str(args.cpu), "--noali", "--domtblout", str(domtbl), "--tblout", str(tblout), str(args.pfam), str(fasta)]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    hits = parse_domtblout(domtbl, args.evalue) if domtbl.exists() else {}
    summary = summarize(records, hits)
    result = {
        "task": "R2 T2-C Pfam/HMMER generated-sequence scan",
        "status": "completed" if proc.returncode == 0 else "hmmscan_failed",
        "input": str(args.input),
        "pfam": str(args.pfam),
        "n_records": len(records),
        "evalue_cutoff": args.evalue,
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "fasta": str(fasta),
        "domtblout": str(domtbl),
        "tblout": str(tblout),
        **summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    md = args.out.with_suffix(".md")
    lines = ["# Pfam/HMMER Scan Of Generated Lysozyme Sequences\n", "| Source | n | Pfam hit rate | Lysozyme-like hit rate | Top domains |", "|---|---:|---:|---:|---|"]
    for src, vals in result["by_source"].items():
        top = ", ".join(f"{name} ({count})" for name, count in vals["top_domains"][:5])
        lines.append(f"| {src} | {vals['n']} | {vals['hit_rate']:.3f} | {vals['lysozyme_like_hit_rate']:.3f} | {top} |")
    lines.append("\nThis is a Pfam-domain sanity check only; CLEAN and Foldseek remain missing from the full T2-C metric triad.\n")
    md.write_text("\n".join(lines))
    print(f"Saved {args.out}")
    print(f"Saved {md}")
    print(json.dumps(result["by_source"], indent=2))
    if proc.returncode:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
