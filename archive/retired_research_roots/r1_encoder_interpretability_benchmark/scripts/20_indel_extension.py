#!/usr/bin/env python
"""Build an indel / frameshift extension dataset from ClinVar.

This script fills the R1-G gap by staging non-SNV protein variants that ESM-2
LLR cannot score cleanly. The default mode is data-prep only:

  1. Parse protein-level HGVS for ClinVar non-SNV variants.
  2. Map genes to UniProt sequences.
  3. Reconstruct mutant protein sequences for small in-frame events when
     possible (del / dup / ins / delins, including truncating delins).
  4. Emit a TSV plus a JSON summary that makes the supported subset explicit.

Frameshifts and complex extensions are retained in the table but marked as
unsupported for direct sequence reconstruction; they require transcript-aware
translation, which is outside the scope of this lightweight staging script.

Usage:
    python r1_encoder_interpretability_benchmark/scripts/20_indel_extension.py \
        --out-tsv r1_encoder_interpretability_benchmark/results/variant_effect/clinvar_indels.tsv \
        --summary-out r1_encoder_interpretability_benchmark/results/variant_effect/indel_extension_summary.json
"""

import argparse
import gzip
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.variant_effect import build_uniprot_sequence_map


AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V", "Ter": "*",
}

PROTEIN_HGVS_RE = re.compile(r"p\.([^\)\s]+)")
SINGLE_OR_RANGE_RE = re.compile(
    r"^(?P<aa1>[A-Z][a-z]{2})(?P<pos1>\d+)"
    r"(?:_(?P<aa2>[A-Z][a-z]{2})(?P<pos2>\d+))?"
    r"(?P<suffix>del|dup)$"
)
INS_RE = re.compile(
    r"^(?P<aa1>[A-Z][a-z]{2})(?P<pos1>\d+)"
    r"_(?P<aa2>[A-Z][a-z]{2})(?P<pos2>\d+)"
    r"ins(?P<inserted>[A-Za-z\*\?]+)$"
)
DELINS_RE = re.compile(
    r"^(?P<aa1>[A-Z][a-z]{2})(?P<pos1>\d+)"
    r"(?:_(?P<aa2>[A-Z][a-z]{2})(?P<pos2>\d+))?"
    r"delins(?P<inserted>[A-Za-z\*\?]+)$"
)


def load_gene_to_uniprot_map(idmapping_path: str) -> dict[str, str]:
    gene_map = {}
    opener = gzip.open if idmapping_path.endswith(".gz") else open
    with opener(idmapping_path, "rt") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[1] == "Gene_Name":
                gene = parts[2].strip().upper()
                gene_map.setdefault(gene, parts[0])
    return gene_map


def clinvar_binary_label(clin_sig: str) -> str:
    sig = (clin_sig or "").lower()
    if "pathogenic" in sig and "benign" not in sig and "conflicting" not in sig:
        return "pathogenic"
    if "benign" in sig and "pathogenic" not in sig:
        return "benign"
    return "other"


def parse_inserted_sequence(raw: str) -> dict:
    tokens = re.findall(r"(?:[A-Z][a-z]{2}|Ter|\*|\?)", raw)
    if not tokens:
        return {
            "sequence": "",
            "contains_stop": False,
            "contains_unknown": "?" in raw,
            "valid": False,
        }
    seq = []
    contains_stop = False
    contains_unknown = False
    for tok in tokens:
        if tok in {"*", "Ter"}:
            contains_stop = True
            break
        if tok == "?":
            contains_unknown = True
            continue
        aa = AA3_TO_1.get(tok)
        if aa is None:
            return {
                "sequence": "",
                "contains_stop": contains_stop,
                "contains_unknown": contains_unknown,
                "valid": False,
            }
        seq.append(aa)
    return {
        "sequence": "".join(seq),
        "contains_stop": contains_stop,
        "contains_unknown": contains_unknown,
        "valid": True,
    }


def safe_residue(aa3: str) -> str | None:
    return AA3_TO_1.get(aa3)


def classify_hgvs(protein_hgvs: str) -> str:
    if "fs" in protein_hgvs:
        return "frameshift"
    if "delins" in protein_hgvs:
        return "delins"
    if protein_hgvs.endswith("del"):
        return "deletion"
    if protein_hgvs.endswith("dup"):
        return "duplication"
    if "ins" in protein_hgvs:
        return "insertion"
    if "ext" in protein_hgvs:
        return "extension"
    return "other"


def reconstruct_variant(wt_seq: str, protein_hgvs: str) -> dict:
    variant_class = classify_hgvs(protein_hgvs)
    if variant_class == "frameshift":
        return {"supported": False, "reason": "frameshift_requires_transcript"}
    if variant_class == "extension":
        return {"supported": False, "reason": "extension_requires_stop_context"}
    if protein_hgvs == "?":
        return {"supported": False, "reason": "unknown_protein_effect"}

    m = SINGLE_OR_RANGE_RE.match(protein_hgvs)
    if m:
        aa1 = safe_residue(m.group("aa1"))
        aa2 = safe_residue(m.group("aa2")) if m.group("aa2") else aa1
        pos1 = int(m.group("pos1"))
        pos2 = int(m.group("pos2")) if m.group("pos2") else pos1
        suffix = m.group("suffix")
        if not aa1 or not aa2:
            return {"supported": False, "reason": "unsupported_residue_code"}
        if pos1 < 1 or pos2 > len(wt_seq):
            return {"supported": False, "reason": "position_out_of_range"}
        if wt_seq[pos1 - 1] != aa1 or wt_seq[pos2 - 1] != aa2:
            return {"supported": False, "reason": "wildtype_mismatch"}
        removed = wt_seq[pos1 - 1:pos2]
        prefix = wt_seq[:pos1 - 1]
        suffix_seq = wt_seq[pos2:]
        if suffix == "del":
            mut = prefix + suffix_seq
        else:
            mut = prefix + removed + removed + suffix_seq
        return {
            "supported": True,
            "mut_sequence": mut,
            "start": pos1,
            "end": pos2,
            "inserted_sequence": removed if suffix == "dup" else "",
            "truncating": False,
        }

    m = INS_RE.match(protein_hgvs)
    if m:
        aa1 = safe_residue(m.group("aa1"))
        aa2 = safe_residue(m.group("aa2"))
        pos1 = int(m.group("pos1"))
        pos2 = int(m.group("pos2"))
        ins = parse_inserted_sequence(m.group("inserted"))
        if not aa1 or not aa2:
            return {"supported": False, "reason": "unsupported_residue_code"}
        if pos1 < 1 or pos2 > len(wt_seq):
            return {"supported": False, "reason": "position_out_of_range"}
        if wt_seq[pos1 - 1] != aa1 or wt_seq[pos2 - 1] != aa2:
            return {"supported": False, "reason": "wildtype_mismatch"}
        if not ins["valid"]:
            return {"supported": False, "reason": "unsupported_inserted_sequence"}
        mut = wt_seq[:pos1] + ins["sequence"] + wt_seq[pos1:]
        if ins["contains_stop"]:
            mut = wt_seq[:pos1] + ins["sequence"]
        return {
            "supported": True,
            "mut_sequence": mut,
            "start": pos1,
            "end": pos2,
            "inserted_sequence": ins["sequence"],
            "truncating": bool(ins["contains_stop"]),
        }

    m = DELINS_RE.match(protein_hgvs)
    if m:
        aa1 = safe_residue(m.group("aa1"))
        aa2 = safe_residue(m.group("aa2")) if m.group("aa2") else aa1
        pos1 = int(m.group("pos1"))
        pos2 = int(m.group("pos2")) if m.group("pos2") else pos1
        ins = parse_inserted_sequence(m.group("inserted"))
        if not aa1 or not aa2:
            return {"supported": False, "reason": "unsupported_residue_code"}
        if pos1 < 1 or pos2 > len(wt_seq):
            return {"supported": False, "reason": "position_out_of_range"}
        if wt_seq[pos1 - 1] != aa1 or wt_seq[pos2 - 1] != aa2:
            return {"supported": False, "reason": "wildtype_mismatch"}
        if not ins["valid"]:
            return {"supported": False, "reason": "unsupported_inserted_sequence"}
        mut = wt_seq[:pos1 - 1] + ins["sequence"] + wt_seq[pos2:]
        if ins["contains_stop"]:
            mut = wt_seq[:pos1 - 1] + ins["sequence"]
        return {
            "supported": True,
            "mut_sequence": mut,
            "start": pos1,
            "end": pos2,
            "inserted_sequence": ins["sequence"],
            "truncating": bool(ins["contains_stop"]),
        }

    return {"supported": False, "reason": "unparsed_hgvs"}


def iter_clinvar_rows(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        header = f.readline().rstrip("\n").split("\t")
        cols = {h: i for i, h in enumerate(header)}
        need = {
            "Type": cols["Type"],
            "Name": cols["Name"],
            "GeneSymbol": cols.get("GeneSymbol", cols.get("#GeneSymbol")),
            "ClinicalSignificance": cols["ClinicalSignificance"],
            "Assembly": cols["Assembly"],
        }
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(need.values()):
                continue
            yield {
                "type": parts[need["Type"]],
                "name": parts[need["Name"]],
                "gene": parts[need["GeneSymbol"]].upper(),
                "clinical_significance": parts[need["ClinicalSignificance"]],
                "assembly": parts[need["Assembly"]],
            }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--clinvar",
        default="data/clinvar/variant_summary.txt.gz",
    )
    ap.add_argument(
        "--idmapping",
        default="data/swissprot/HUMAN_9606_idmapping.dat.gz",
    )
    ap.add_argument(
        "--swissprot-cache",
        default="data/processed/swissprot_all_max1022.pkl",
    )
    ap.add_argument(
        "--out-tsv",
        default="r1_encoder_interpretability_benchmark/results/variant_effect/clinvar_indels.tsv",
    )
    ap.add_argument(
        "--summary-out",
        default="r1_encoder_interpretability_benchmark/results/variant_effect/indel_extension_summary.json",
    )
    ap.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="0 = parse all rows",
    )
    args = ap.parse_args()

    print("=" * 70)
    print("  R1-G ClinVar indel / frameshift staging")
    print("=" * 70)

    gene_to_uniprot = load_gene_to_uniprot_map(args.idmapping)
    seq_map = build_uniprot_sequence_map(args.swissprot_cache)
    print(f"  Gene map: {len(gene_to_uniprot)}")
    print(f"  Sequence map: {len(seq_map)}")

    rows = []
    seen = set()
    counts = Counter()
    supported = Counter()
    unsupported = Counter()

    for row in iter_clinvar_rows(args.clinvar):
        if row["type"] == "single nucleotide variant":
            continue
        m = PROTEIN_HGVS_RE.search(row["name"])
        if not m:
            counts["no_protein_hgvs"] += 1
            continue
        protein_hgvs = m.group(1)
        key = (row["gene"], protein_hgvs, row["clinical_significance"])
        if key in seen:
            continue
        seen.add(key)

        variant_class = classify_hgvs(protein_hgvs)
        counts[variant_class] += 1

        uniprot_id = gene_to_uniprot.get(row["gene"], "")
        wt_seq = seq_map.get(uniprot_id)
        if not uniprot_id or wt_seq is None:
            unsupported["missing_uniprot_sequence"] += 1
            rec = {
                "gene": row["gene"],
                "uniprot_id": uniprot_id,
                "protein_hgvs": protein_hgvs,
                "variant_class": variant_class,
                "label": clinvar_binary_label(row["clinical_significance"]),
                "clinical_significance": row["clinical_significance"],
                "supported": False,
                "support_reason": "missing_uniprot_sequence",
                "truncating": False,
                "wt_length": "",
                "mut_length": "",
                "length_delta": "",
            }
            rows.append(rec)
            continue

        recon = reconstruct_variant(wt_seq, protein_hgvs)
        if recon.get("supported"):
            mut_seq = recon["mut_sequence"]
            supported[variant_class] += 1
            rec = {
                "gene": row["gene"],
                "uniprot_id": uniprot_id,
                "protein_hgvs": protein_hgvs,
                "variant_class": variant_class,
                "label": clinvar_binary_label(row["clinical_significance"]),
                "clinical_significance": row["clinical_significance"],
                "supported": True,
                "support_reason": "reconstructed",
                "truncating": recon.get("truncating", False),
                "wt_length": len(wt_seq),
                "mut_length": len(mut_seq),
                "length_delta": len(mut_seq) - len(wt_seq),
            }
        else:
            reason = recon.get("reason", "unsupported")
            unsupported[reason] += 1
            rec = {
                "gene": row["gene"],
                "uniprot_id": uniprot_id,
                "protein_hgvs": protein_hgvs,
                "variant_class": variant_class,
                "label": clinvar_binary_label(row["clinical_significance"]),
                "clinical_significance": row["clinical_significance"],
                "supported": False,
                "support_reason": reason,
                "truncating": False,
                "wt_length": len(wt_seq),
                "mut_length": "",
                "length_delta": "",
            }
        rows.append(rec)

        if args.max_records and len(rows) >= args.max_records:
            break

    os.makedirs(os.path.dirname(args.out_tsv) or ".", exist_ok=True)
    with open(args.out_tsv, "w") as f:
        cols = [
            "gene", "uniprot_id", "protein_hgvs", "variant_class", "label",
            "clinical_significance", "supported", "support_reason",
            "truncating", "wt_length", "mut_length", "length_delta",
        ]
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")

    summary = {
        "n_rows": len(rows),
        "variant_class_counts": dict(counts),
        "supported_by_class": dict(supported),
        "unsupported_reasons": dict(unsupported.most_common()),
        "label_counts": dict(Counter(r["label"] for r in rows)),
        "config": {
            "clinvar": args.clinvar,
            "idmapping": args.idmapping,
            "swissprot_cache": args.swissprot_cache,
            "max_records": args.max_records,
        },
    }
    os.makedirs(os.path.dirname(args.summary_out) or ".", exist_ok=True)
    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Parsed rows: {len(rows)}")
    print(f"  Variant classes: {dict(counts)}")
    print(f"  Supported subset: {dict(supported)}")
    print(f"  Top unsupported reasons: {dict(unsupported.most_common(6))}")
    print(f"\n  Saved: {args.out_tsv}")
    print(f"  Saved: {args.summary_out}")


if __name__ == "__main__":
    main()
