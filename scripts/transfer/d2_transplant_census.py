#!/usr/bin/env python3
"""Exact CPU tensor census of the parent/Stage 1 transplant pair, by declared group."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.transfer.capability_transplant import census
from src.transfer.io import sha256_file, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination', required=True, type=Path, help='Checkpoint transplanted into')
    p.add_argument('--source', required=True, type=Path, help='Checkpoint transplanted from')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--device', default='cpu', choices=['cpu'])
    args = p.parse_args()
    import torch
    torch.set_num_threads(4)
    record = census(args.destination, args.source,
                    progress=lambda row: print(
                        f"{row['name']}: differing FP32 elements="
                        f"{row['differing_elements_fp32']}/{row['n_elements']}", flush=True))
    record['script_sha256'] = sha256_file(Path(__file__))
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out/'d2_transplant_census.json', record)


if __name__ == '__main__':
    main()
