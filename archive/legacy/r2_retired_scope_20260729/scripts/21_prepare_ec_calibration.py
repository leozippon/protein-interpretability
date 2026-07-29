#!/usr/bin/env python
"""Prepare real lysozyme and random UniRef50 calibration records for R2 T2-C."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_seq(seq: str) -> str:
    return "".join(c for c in str(seq).upper() if c in AA)


def clean_zymctrl_seq(seq: str) -> str:
    text = str(seq)
    if "<start>" in text:
        text = text.split("<start>", 1)[1]
    if "<end>" in text:
        text = text.split("<end>", 1)[0]
    return clean_seq(text)


def fasta_iter(path: Path):
    header = None
    chunks = []
    with path.open(errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)
    if header is not None:
        yield header, "".join(chunks)


def safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")[:96] or "record"


def zymctrl_ec(header: str) -> str:
    parts = header.split("|")
    return parts[1].strip() if len(parts) >= 2 else ""


def select_real_lysozyme(path: Path, target_ec: str, n: int, min_len: int, max_len: int, seed: int) -> list[dict]:
    eligible = []
    for header, seq in fasta_iter(path):
        ec = zymctrl_ec(header)
        clean = clean_zymctrl_seq(seq)
        if ec == target_ec and min_len <= len(clean) <= max_len:
            eligible.append({
                "id": f"real_{safe_id(header)}",
                "source": "real_lysozyme",
                "sequence": clean,
                "meta": {"header": header, "ec": ec, "length": len(clean)},
            })
    rng = random.Random(seed)
    rng.shuffle(eligible)
    return eligible[:min(n, len(eligible))]


def reservoir_random_uniref(
    path: Path,
    n: int,
    min_len: int,
    max_len: int,
    seed: int,
    max_candidates: int | None,
) -> tuple[list[dict], int]:
    rng = random.Random(seed)
    sample: list[dict] = []
    seen = 0
    for header, seq in fasta_iter(path):
        clean = clean_seq(seq)
        if not (min_len <= len(clean) <= max_len):
            continue
        seen += 1
        rec = {
            "id": f"random_{safe_id(header.split()[0])}",
            "source": "random_uniref50",
            "sequence": clean,
            "meta": {"header": header, "length": len(clean)},
        }
        if len(sample) < n:
            sample.append(rec)
        else:
            j = rng.randrange(seen)
            if j < n:
                sample[j] = rec
        if max_candidates is not None and seen >= max_candidates:
            break
    sample.sort(key=lambda r: r["id"])
    return sample, seen


def write_json(path: Path, records: list[dict], args: argparse.Namespace, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": "R2 T2-C EC metric calibration sequence set",
        "status": "prepared",
        "target_ec": args.target_ec,
        "seed": args.seed,
        "length_filter": {"min_len": args.min_len, "max_len": args.max_len},
        "n_records": len(records),
        "records": records,
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_fasta(path: Path, records: list[dict]) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(f">{r['id']} source={r['source']} len={len(r['sequence'])}\n")
            seq = r["sequence"]
            for i in range(0, len(seq), 80):
                f.write(seq[i:i + 80] + "\n")


def markdown(manifest: dict) -> str:
    lines = ["# EC Metric Calibration Set (2026-05-07)\n"]
    lines.append("| Set | n | Notes |")
    lines.append("|---|---:|---|")
    lines.append(f"| real_lysozyme | {manifest['counts']['real_lysozyme']} | SwissProt/ZymCTRL EC {manifest['target_ec']} records after length filtering |")
    lines.append(f"| random_uniref50 | {manifest['counts']['random_uniref50']} | Reservoir sample from UniRef50 with the same length filter |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zymctrl-fasta", type=Path, default=Path(os.environ.get("BIOCC_ZYMCTRL_FASTA", REPO / "data" / "zymctrl" / "ec_labeled_swissprot.fasta")))
    ap.add_argument("--uniref50-fasta", type=Path, default=Path(os.environ.get("BIOCC_UNIREF50_FASTA", REPO / "data" / "uniref50" / "uniref50.fasta")))
    ap.add_argument("--target-ec", default="3.2.1.17")
    ap.add_argument("--n-per-set", type=int, default=100)
    ap.add_argument("--min-len", type=int, default=80)
    ap.add_argument("--max-len", type=int, default=250)
    ap.add_argument(
        "--max-random-candidates",
        type=int,
        default=50000,
        help="Stop after this many length-matched UniRef50 candidates; set 0 to scan the full FASTA.",
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", type=Path, default=REPO / "r2_interpretability_transfer" / "results" / "ec_metrics" / "calibration_lysozyme_20260507")
    args = ap.parse_args()

    real = select_real_lysozyme(args.zymctrl_fasta, args.target_ec, args.n_per_set, args.min_len, args.max_len, args.seed)
    max_candidates = args.max_random_candidates if args.max_random_candidates > 0 else None
    random_records, random_pool = reservoir_random_uniref(
        args.uniref50_fasta, args.n_per_set, args.min_len, args.max_len, args.seed, max_candidates
    )
    all_records = real + random_records

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "calibration_real_lysozyme.json", real, args, {"source": "real_lysozyme"})
    write_json(args.out_dir / "calibration_random_uniref50.json", random_records, args, {"source": "random_uniref50", "eligible_random_pool": random_pool})
    write_json(args.out_dir / "calibration_sequences.json", all_records, args, {"sources": ["real_lysozyme", "random_uniref50"], "eligible_random_pool": random_pool})
    write_fasta(args.out_dir / "calibration_sequences.fasta", all_records)
    manifest = {
        "task": "R2 T2-C EC metric calibration set",
        "status": "prepared",
        "target_ec": args.target_ec,
        "seed": args.seed,
        "zymctrl_fasta": str(args.zymctrl_fasta),
        "uniref50_fasta": str(args.uniref50_fasta),
        "length_filter": {"min_len": args.min_len, "max_len": args.max_len},
        "max_random_candidates": max_candidates,
        "counts": {"real_lysozyme": len(real), "random_uniref50": len(random_records)},
        "eligible_random_pool": random_pool,
        "outputs": {
            "combined_json": str(args.out_dir / "calibration_sequences.json"),
            "real_json": str(args.out_dir / "calibration_real_lysozyme.json"),
            "random_json": str(args.out_dir / "calibration_random_uniref50.json"),
            "fasta": str(args.out_dir / "calibration_sequences.fasta"),
        },
    }
    (args.out_dir / "calibration_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out_dir / "calibration_manifest.md").write_text(markdown(manifest))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
