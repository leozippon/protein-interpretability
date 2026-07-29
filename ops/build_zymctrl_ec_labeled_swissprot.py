#!/usr/bin/env python3
"""Build a reviewed-enzyme EC-labeled FASTA for ZymCTRL-style CLT training.

This is a fallback replacement for the missing original `ec_labeled.fasta`.
It parses Swiss-Prot XML, extracts full EC numbers, and writes records in
the sequence format ZymCTRL was trained on:

    <EC><sep><start><AA_SEQUENCE><end>

If a protein entry has multiple complete EC numbers, one FASTA record is
emitted per EC label.
"""

from __future__ import annotations

import argparse
import gzip
import re
from pathlib import Path
from typing import Iterable
from xml.etree.ElementTree import iterparse


FULL_EC_RE = re.compile(r"^\d+\.\d+\.\d+\.\d+$")


def iter_entries(xml_path: str) -> Iterable[tuple[str, str, list[str]]]:
    """Yield ``(accession, sequence, ec_labels)`` for every enzyme entry.

    The namespace is read from the document's own root element rather than
    hard-coded. It used to be the literal ``{https://uniprot.org/uniprot}``,
    while UniProt's canonical spelling is ``http://uniprot.org/uniprot``; on a
    release using the canonical form every ``elem.tag`` comparison failed, every
    entry was skipped, and ``main`` wrote an empty FASTA and exited 0. The
    corpus that replaces ZymCTRL's unavailable training set is not a place for a
    parser that reports success on having parsed nothing.
    """

    opener = gzip.open if xml_path.endswith(".gz") else open
    with opener(xml_path, "rb") as f:
        events = iterparse(f, events=("start-ns", "end"))
        namespace: str | None = None
        for event, payload in events:
            if event == "start-ns":
                prefix, uri = payload
                if prefix == "":
                    namespace = f"{{{uri}}}"
                continue
            elem = payload
            if namespace is None:
                raise RuntimeError(
                    f"{xml_path} declares no default XML namespace; this parser "
                    "resolves UniProt element names through it"
                )
            NS = namespace
            if elem.tag != f"{NS}entry":
                continue

            acc_elem = elem.find(f"{NS}accession")
            seq_elem = elem.find(f"{NS}sequence")
            if acc_elem is None or seq_elem is None or seq_elem.text is None:
                elem.clear()
                continue

            accession = (acc_elem.text or "").strip()
            sequence = seq_elem.text.replace("\n", "").strip()
            ecs = set()

            for ec_elem in elem.findall(f".//{NS}ecNumber"):
                if ec_elem.text and FULL_EC_RE.fullmatch(ec_elem.text.strip()):
                    ecs.add(ec_elem.text.strip())
            for db_ref in elem.findall(f".//{NS}dbReference[@type='EC']"):
                ec_id = db_ref.attrib.get("id", "").strip()
                if FULL_EC_RE.fullmatch(ec_id):
                    ecs.add(ec_id)

            if accession and sequence and ecs:
                yield accession, sequence, sorted(ecs)
            elem.clear()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--xml",
        default="/Data/lzp/BioInterpretebility-CC/data/swissprot/uniprot_sprot.xml.gz",
    )
    ap.add_argument(
        "--out",
        default="/Data/lzp/BioInterpretebility-CC/data/zymctrl/ec_labeled_swissprot.fasta",
    )
    ap.add_argument("--max-seq-len", type=int, default=1022)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Written beside the destination and renamed only once the record count has
    # been checked. Streaming straight into `out_path` left an interrupted run
    # holding a partial FASTA that is indistinguishable from a complete one, and
    # ops/sync_required_assets_to_h200.sh would then stage it.
    temporary = out_path.with_name(f".{out_path.name}.partial")
    n_entries = 0
    n_records = 0
    try:
        with temporary.open("w") as out_f:
            for accession, sequence, ecs in iter_entries(args.xml):
                if len(sequence) > args.max_seq_len:
                    continue
                n_entries += 1
                for ec in ecs:
                    record = f"{ec}<sep><start>{sequence}<end>"
                    out_f.write(f">{accession}|{ec}\n{record}\n")
                    n_records += 1
                    if n_records % 50000 == 0:
                        print(f"Wrote {n_records:,} EC-labeled sequences")
        if n_records == 0:
            raise RuntimeError(
                f"{args.xml} yielded no EC-labelled records. An empty corpus is not a "
                "result: either the XML holds no reviewed enzymes with a complete "
                "four-field EC number, or its element names did not resolve. Nothing "
                "has been written to "
                f"{out_path}."
            )
        temporary.replace(out_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(f"Done. Entries with EC: {n_entries:,}")
    print(f"Done. FASTA records: {n_records:,}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
