#!/usr/bin/env python3
"""Measure the capacity of the imported explanation channel for protein decoders.

Four measurements, none of which depends on any interpretability method:

A  the analytic event-selection ceiling of a top-k design;
B  the marginal entropy of curated residue-level Pfam labels;
C  the marginal entropy of structural-oracle attributes from AlphaFold;
D  within-sequence label entropy over one fixed window, for text token identity,
   residue identity, Pfam domain label and structural attributes.

Part D is the decisive one. R2's matched null permutes labels inside a sequence,
and a label that is constant inside a sequence is invariant under that null, so
for such labels the test has no power by construction rather than by result.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoTokenizer  # noqa: E402

from src.transfer.io import write_json  # noqa: E402
from src.transfer.arms import (  # noqa: E402
    PANEL,
    REPO,
    SWISSPROT_FASTA,
    Cohort,
    iter_fasta,
    text_cohort,
)
from src.transfer.channels import (  # noqa: E402
    ALPHAFOLD_ROOT,
    DEFAULT_WINDOW,
    PFAM_RESIDUE_TSV,
    SECONDARY_STRUCTURE_CAVEAT,
    UNANNOTATED,
    alphafold_model_sample,
    event_selection_ceiling,
    label_distribution,
    load_pfam_spans,
    pfam_residue_labels,
    read_alphafold_model,
    structural_attribute_labels,
    within_unit_label_entropy,
)

SCHEMA_VERSION = "r2_transfer_explanation_channel_v1"
DEFAULT_OUT = REPO / "results/transfer_20260728/explanation_channel"


def swissprot_accession(header: str) -> str:
    return header.split("|")[1] if "|" in header else header.split()[0]


def swissprot_records(limit: int) -> dict[str, str]:
    """The first ``limit`` Swiss-Prot entries in deterministic file order."""

    records: dict[str, str] = {}
    for header, sequence in iter_fasta(SWISSPROT_FASTA):
        records[swissprot_accession(header)] = sequence
        if len(records) >= limit:
            break
    if len(records) < limit:
        raise RuntimeError(f"only {len(records)}/{limit} Swiss-Prot entries available")
    return records


def curated_channel(
    records: dict[str, str], *, window: int, max_units: int
) -> tuple[dict[str, Any], list[list[str]], list[list[str]], Cohort]:
    """Marginal Pfam statistics plus matched Pfam and residue-identity units."""

    spans = load_pfam_spans(PFAM_RESIDUE_TSV, accessions=set(records))
    covered: Counter[str] = Counter()
    for accession, entries in spans.items():
        length = len(records[accession])
        for start, end, pfam in entries:
            covered[pfam] += min(end, length) - start + 1
    residues_total = sum(len(sequence) for sequence in records.values())
    annotated = int(sum(covered.values()))
    if annotated == 0:
        raise RuntimeError("no Pfam residue coverage for the sampled accessions")

    including_unannotated = Counter(covered)
    including_unannotated[UNANNOTATED] = max(residues_total - annotated, 0)

    pfam_units: list[list[str]] = []
    residue_units: list[list[str]] = []
    unit_sequences: list[str] = []
    overlap_residues = 0
    for accession in sorted(spans):
        sequence = records[accession]
        if len(sequence) < window:
            continue
        labels, overlaps = pfam_residue_labels(len(sequence), spans[accession])
        overlap_residues += overlaps
        pfam_units.append(labels)
        residue_units.append(list(sequence))
        unit_sequences.append(sequence)
        if len(pfam_units) >= max_units:
            break

    summary = {
        "n_proteins_scanned": len(records),
        "n_proteins_with_pfam": len(spans),
        "residues_scanned": residues_total,
        "residues_with_pfam": annotated,
        "pfam_residue_coverage": annotated / residues_total,
        "n_distinct_pfam_domains": len(covered),
        "overlapping_pfam_residues_in_units": overlap_residues,
        "marginal_including_unannotated": label_distribution(including_unannotated),
        "marginal_annotated_only": label_distribution(covered),
    }
    cohort = Cohort("pfam_units", "protein", unit_sequences, window, 0, {})
    return summary, pfam_units, residue_units, cohort


def structural_channel(
    *, n_structures: int, window: int, max_units: int, min_residues: int, seed: int
) -> tuple[dict[str, Any], list[list[tuple[int, int, int]]], Cohort]:
    """Marginal structural-attribute statistics plus within-protein units.

    The structures are drawn under a seeded permutation of the whole AlphaFold
    catalogue. They used to be ``alphafold_models(root, limit=n)``, which is a
    filename-order prefix -- and filename order is UniProt-accession order, which
    front-loads whole-proteome dumps of closely related entries. The L9 figures
    this stage produces (structural-oracle attributes at 3.61 bits/symbol, Pfam
    domain label at 0.74) are per-symbol entropies of a label channel *measured
    over whichever structures were read*, so a taxonomically clustered prefix
    understates them by however much that neighbourhood is more uniform than the
    catalogue. The audit records those figures as carrying a declared source-order
    bias; this is the draw that discharges it.
    """

    secondary: Counter[int] = Counter()
    contacts: Counter[int] = Counter()
    confidence: Counter[int] = Counter()
    joint: Counter[tuple[int, int, int]] = Counter()
    residues: Counter[str] = Counter()
    units: list[list[tuple[int, int, int]]] = []
    unit_sequences: list[str] = []
    used = 0
    excluded_short = 0
    excluded_non_canonical = 0

    selection, sampling = alphafold_model_sample(
        ALPHAFOLD_ROOT, limit=n_structures, seed=seed
    )
    for path in selection:
        structure = read_alphafold_model(path)
        if structure.n_non_canonical_residues > 0:
            excluded_non_canonical += 1
            continue
        if len(structure) < min_residues:
            excluded_short += 1
            continue
        attributes = structural_attribute_labels(structure)
        used += 1
        residues.update(structure.sequence)
        for state, contact, plddt in attributes:
            secondary[state] += 1
            contacts[contact] += 1
            confidence[plddt] += 1
            joint[(state, contact, plddt)] += 1
        if len(structure) >= window and len(units) < max_units:
            units.append(attributes)
            unit_sequences.append(structure.sequence)
    if used == 0:
        raise RuntimeError("no AlphaFold model satisfied the residue-count filter")

    total = int(sum(secondary.values()))
    summary = {
        "sampling": sampling,
        "n_models_examined": int(n_structures),
        "n_models_used": used,
        "n_models_excluded_short": excluded_short,
        "n_models_excluded_non_canonical_residues": excluded_non_canonical,
        "residues": total,
        "secondary_structure_method": SECONDARY_STRUCTURE_CAVEAT,
        "secondary_structure_fractions": {
            str(state): count / total for state, count in sorted(secondary.items())
        },
        "marginal_secondary_structure3": label_distribution(secondary),
        "marginal_contact_number_bin": label_distribution(contacts),
        "marginal_plddt_bin": label_distribution(confidence),
        "marginal_joint_structural_attributes": label_distribution(joint),
        "marginal_residue_identity": label_distribution(residues),
    }
    cohort = Cohort("alphafold_units", "protein", unit_sequences, window, 0, {})
    return summary, units, cohort


def text_channel(
    *, n_documents: int, min_chars: int, window: int, max_units: int
) -> tuple[list[list[int]], Cohort]:
    """GPT-2 token-identity units, the closed-vocabulary reference case."""

    cohort = text_cohort(n_documents, min_chars=min_chars, name="openwebtext_units")
    tokenizer = AutoTokenizer.from_pretrained(str(PANEL["gpt2-large"].path))
    units: list[list[int]] = []
    for document in cohort.records:
        ids = tokenizer(document, return_tensors=None)["input_ids"][: 2 * window]
        if len(ids) >= window:
            units.append(ids)
        if len(units) >= max_units:
            break
    return units, cohort


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--min-units", type=int, default=30)
    parser.add_argument("--max-units", type=int, default=2000)
    parser.add_argument("--n-pfam-proteins", type=int, default=20000)
    parser.add_argument("--n-structures", type=int, default=1500)
    parser.add_argument(
        "--structure-seed",
        type=int,
        default=20260728,
        help="seed for the permutation the AlphaFold structures are drawn under; "
        "filename order is accession order and a prefix is a taxonomic "
        "neighbourhood rather than a sample (transfer audit, Appendix B rule 1)",
    )
    parser.add_argument("--structure-min-residues", type=int, default=30)
    parser.add_argument("--n-text-documents", type=int, default=3000)
    parser.add_argument("--text-min-chars", type=int, default=2000)
    parser.add_argument("--cohort-positions", type=int, default=122_671)
    parser.add_argument("--realised-event-count", type=int, default=100)
    parser.add_argument("--realised-gate-nats", type=float, default=0.1)
    parser.add_argument(
        "--event-counts", type=int, nargs="*", default=[10, 100, 1000, 5000, 20000]
    )
    parser.add_argument(
        "--gate-nats", type=float, nargs="*", default=[0.01, 0.05, 0.1, 0.2]
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    ceiling = event_selection_ceiling(
        args.cohort_positions,
        event_counts=args.event_counts,
        gate_nats=args.gate_nats,
        realised_event_count=args.realised_event_count,
        realised_gate_nats=args.realised_gate_nats,
    )

    records = swissprot_records(args.n_pfam_proteins)
    curated, pfam_units, residue_units, pfam_cohort = curated_channel(
        records, window=args.window, max_units=args.max_units
    )
    structural, structural_units, structural_cohort = structural_channel(
        n_structures=args.n_structures,
        window=args.window,
        max_units=args.max_units,
        min_residues=args.structure_min_residues,
        seed=args.structure_seed,
    )
    text_units, text_cohort_records = text_channel(
        n_documents=args.n_text_documents,
        min_chars=args.text_min_chars,
        window=args.window,
        max_units=args.max_units,
    )

    def measure(units) -> dict[str, Any]:
        return within_unit_label_entropy(
            units, window=args.window, min_units=args.min_units
        )

    within = {
        "text_token_identity_within_document": measure(text_units),
        "protein_residue_identity_within_protein": measure(residue_units),
        "protein_pfam_label_within_protein": measure(pfam_units),
        "protein_structural_attributes_within_protein": measure(structural_units),
    }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "explanation_channel_report",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_digest": {
            "pfam_units": pfam_cohort.digest,
            "alphafold_units": structural_cohort.digest,
            "text_units": text_cohort_records.digest,
        },
        "arm_spec": {
            "text_tokenizer": {
                "arm": "gpt2-large",
                "path": str(PANEL["gpt2-large"].path),
                "tokenisation": PANEL["gpt2-large"].tokenisation,
            },
            "protein_label_sources": {
                "pfam_residue_tsv": str(PFAM_RESIDUE_TSV),
                "alphafold_root": str(ALPHAFOLD_ROOT),
                "swissprot_fasta": str(SWISSPROT_FASTA),
            },
        },
        # Two channels are still drawn in corpus file order and say so: the Pfam
        # channel takes the first --n-pfam-proteins Swiss-Prot entries and the
        # text channel takes arms.text_cohort's file-order draw. Only the
        # structural channel is permuted, because it is the one whose selection
        # was measured to matter (see structural_channel). Recorded per channel
        # rather than as one blanket statement, which is what previously made the
        # difference invisible.
        "seeds": {
            "structural_channel_permutation": int(args.structure_seed),
            "curated_pfam_channel": "swissprot_file_order_prefix",
            "text_channel": "arms.text_cohort file order",
            "hazard": (
                "a file-order prefix of a biological corpus is a set of near-clonal "
                "homologues rather than a sample; the two channels above are not yet "
                "permuted and their bits/symbol carry that bias"
            ),
        },
        "thresholds": {
            "window_symbols": int(args.window),
            "min_units": int(args.min_units),
            "max_units": int(args.max_units),
            "structure_min_residues": int(args.structure_min_residues),
        },
        "part_a_event_selection_ceiling": ceiling,
        "part_b_curated_residue_annotation": curated,
        "part_c_structural_oracle_attributes": structural,
        "part_d_within_sequence_label_entropy": within,
        "comparison_within_sequence_bits": {
            name: block["entropy_bits"]["mean"] for name, block in within.items()
        },
    }
    destination = args.out / "explanation_channel.json"
    write_json(destination, payload)
    print(f"wrote {destination}")
    print(
        "event-selection ceiling: "
        f"{ceiling['realised_event_count']} of {ceiling['cohort_positions']} positions "
        f"-> {ceiling['realised_max_possible_mi_nats']:.4f} nats max, "
        f"gate {ceiling['realised_gate_nats']} nats is "
        f"{'attainable' if ceiling['gate_is_attainable'] else 'unattainable'} "
        f"({ceiling['realised_gate_over_ceiling']:.1f}x the ceiling)"
    )
    print(f"pfam residue coverage: {curated['pfam_residue_coverage']:.4f}")
    for name, block in within.items():
        print(
            f"  {name:52s} {block['entropy_bits']['mean']:6.3f} bits  "
            f"(units {block['n_units']}, majority share "
            f"{block['mean_majority_label_share']:.3f}, constant-window fraction "
            f"{block['permutation_null_degenerate_fraction']:.3f})"
        )


if __name__ == "__main__":
    main()
