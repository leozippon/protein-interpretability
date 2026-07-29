"""TG-04: how many bits of explanation are available per symbol?

Anthropic's loop labels a feature in the model's own output vocabulary and scores
the label against the same tokens the model predicts. Protein decoders have no
such closure: the explanation vocabulary has to be imported from annotation
databases. This script measures the capacity of that imported channel and
compares three candidate vocabularies.

Part A  the information ceiling of a top-k event design (analytic)
Part B  residue-level Pfam/InterPro domain labels (sparse, curated)
Part C  residue-level structural-oracle attributes from AlphaFold (dense)

**Correction against the 2026-07-24 run.** Every one of Parts B, C and D drew its
units in source order: the first N records of Swiss-Prot, the first N AlphaFold
models by filename, the first N OpenWebText documents. The bits/symbol contrast
this produced -- text token identity 7.32, structural attributes 3.61, Pfam
domain label 0.74 -- is quoted in the audit with the standing caveat that it is
"single-run with declared source-order bias and must be re-derived under a seeded
permutation before it is quoted". Every unit set is now a seeded permutation of
the eligible population, which discharges that caveat. Nothing else changed, so a
difference against the recorded figures is the size of the bias.

Swiss-Prot is sorted by accession, which groups by source organism and by
curation date; the AlphaFold directory is sorted by accession too, so Parts B, C
and D shared the bias rather than averaging over it.
"""

from __future__ import annotations

import argparse
import gzip
import math
from collections import Counter
from pathlib import Path

import numpy as np

from tg_common import DEFAULT_COHORT_SEED, REPO, iter_fasta, write_json

PFAM_RESIDUE = REPO / "data/interpro/pfam_residue.tsv"
ALPHAFOLD = REPO / "data/alphafold"
SWISSPROT = REPO / "data/swissprot/uniprot_sprot.fasta.gz"


def entropy_bits(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        raise ValueError("empty distribution")
    return float(
        -sum((c / total) * math.log2(c / total) for c in counter.values() if c > 0)
    )


def binary_entropy_nats(p: float) -> float:
    if not 0 < p < 1:
        return 0.0
    return -p * math.log(p) - (1 - p) * math.log(1 - p)


def required_event_count(target_nats: float, n_positions: int) -> int:
    """Smallest event count whose entropy reaches `target_nats`."""
    lo, hi = 1, n_positions // 2
    while lo < hi:
        mid = (lo + hi) // 2
        if binary_entropy_nats(mid / n_positions) < target_nats:
            lo = mid + 1
        else:
            hi = mid
    return lo


# ------------------------------------------------------------------ structures

def read_alphafold(path: Path):
    """CA coordinates, pLDDT and one-letter sequence from an AlphaFold model."""
    three_to_one = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
        "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
        "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
        "TRP": "W", "TYR": "Y",
    }
    coords, plddt, seq = [], [], []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                resname = line[17:20].strip()
                if resname not in three_to_one:
                    continue
                coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
                plddt.append(float(line[60:66]))
                seq.append(three_to_one[resname])
    return np.asarray(coords), np.asarray(plddt), "".join(seq)


def ca_secondary_structure(ca: np.ndarray) -> np.ndarray:
    """Three-state assignment from the CA trace (P-SEA distance criterion).

    A coordinate-only approximation, used because it is uniform across all
    AlphaFold models without an external DSSP dependency.
    """
    n = len(ca)
    ss = np.full(n, 2, dtype=np.int8)  # 0 helix, 1 strand, 2 coil
    if n < 5:
        return ss
    def ca_distance(a: int, b: int) -> float:
        return float(np.linalg.norm(ca[a] - ca[b]))

    for i in range(n - 4):
        d2 = ca_distance(i, i + 2)
        d3 = ca_distance(i, i + 3)
        d4 = ca_distance(i, i + 4)
        if abs(d2 - 5.5) < 0.5 and abs(d3 - 5.3) < 0.5 and abs(d4 - 6.4) < 0.6:
            ss[i : i + 5] = 0
        elif abs(d2 - 6.7) < 0.6 and abs(d3 - 9.9) < 0.9 and abs(d4 - 12.4) < 1.1:
            ss[i : i + 5] = 1
    return ss


def seeded_sample(population: list, size: int, seed: int) -> list:
    """A seeded permutation of ``population``, truncated to ``size``.

    Truncating a permutation rather than taking the head of the source is the
    whole correction: it makes the sample a sample of the corpus instead of a
    sample of whichever block of it the file happens to begin with.
    """

    order = np.random.default_rng(seed).permutation(len(population))[:size]
    return [population[i] for i in order]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-structures", type=int, default=1500)
    ap.add_argument("--n-proteins-pfam", type=int, default=20000)
    ap.add_argument("--text-token-entropy-bits", type=float, default=10.854,
                    help="GPT-2-large token entropy from TG-01, for comparison")
    ap.add_argument("--seed", type=int, default=DEFAULT_COHORT_SEED)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # ---------------------------------------------------------------- Part A
    # A feature that fires on m of N positions carries at most h(m/N) nats, so
    # any mutual-information gate applied to a top-k event list is capped
    # regardless of how biologically meaningful the feature is.
    n_positions = 122_671  # the R2 Swiss-Prot re-audit cohort
    part_a = dict(
        cohort_positions=n_positions,
        r2_event_count=100,
        r2_event_prevalence=100 / n_positions,
        r2_max_possible_mi_nats=binary_entropy_nats(100 / n_positions),
        r2_gate_nats=0.1,
        r2_gate_over_ceiling=0.1 / binary_entropy_nats(100 / n_positions),
        events_required_for_gate={
            f"{t}": required_event_count(t, n_positions)
            for t in (0.01, 0.05, 0.1, 0.2)
        },
        ceiling_by_event_count_nats={
            str(m): binary_entropy_nats(m / n_positions)
            for m in (10, 100, 1000, 5000, 20000)
        },
    )

    # ---------------------------------------------------------------- Part B
    # The whole Swiss-Prot index is read, then sampled. Reading until the target
    # count is reached would reintroduce exactly the source-order bias this run
    # exists to remove.
    all_records = [
        (header.split("|")[1] if "|" in header else header.split()[0], seq)
        for header, seq in iter_fasta(SWISSPROT)
    ]
    lengths = {
        acc: len(seq)
        for acc, seq in seeded_sample(all_records, args.n_proteins_pfam, args.seed)
    }

    covered = Counter()
    residues_total = sum(lengths.values())
    with open(PFAM_RESIDUE) as handle:
        next(handle)
        for line in handle:
            acc, start, end, pfam = line.rstrip("\n").split("\t")
            if acc not in lengths:
                continue
            covered[pfam] += min(int(end), lengths[acc]) - int(start) + 1
    annotated = sum(covered.values())
    if annotated == 0:
        raise RuntimeError("no Pfam residue coverage for the sampled accessions")
    label_dist = Counter(covered)
    label_dist["__unannotated__"] = max(residues_total - annotated, 0)

    part_b = dict(
        proteins=len(lengths),
        residues=residues_total,
        residues_with_pfam=annotated,
        pfam_residue_coverage=annotated / residues_total,
        distinct_pfam_domains=len(covered),
        entropy_bits_per_residue_including_unannotated=entropy_bits(label_dist),
        entropy_bits_per_residue_annotated_only=entropy_bits(Counter(covered)),
    )

    # ---------------------------------------------------------------- Part C
    available = sorted(ALPHAFOLD.glob("AF-*-model_v*.pdb.gz"))
    if len(available) < 100:
        raise RuntimeError(f"only {len(available)} AlphaFold models under {ALPHAFOLD}")
    files = seeded_sample(available, args.n_structures, args.seed)
    ss_c, cn_c, pl_c, joint_c, aa_c = Counter(), Counter(), Counter(), Counter(), Counter()
    n_res = 0
    n_scored_structures = 0
    for path in files:
        ca, plddt, seq = read_alphafold(path)
        if len(ca) < 30 or len(ca) != len(seq):
            continue
        n_scored_structures += 1
        ss = ca_secondary_structure(ca)
        dist = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
        np.fill_diagonal(dist, 1e9)
        contacts = (dist < 10.0).sum(1)
        cn_bin = np.clip(contacts // 4, 0, 7)
        pl_bin = np.clip((plddt // 20).astype(int), 0, 4)
        for a, s, c, p in zip(seq, ss, cn_bin, pl_bin):
            aa_c[a] += 1
            ss_c[int(s)] += 1
            cn_c[int(c)] += 1
            pl_c[int(p)] += 1
            joint_c[(int(s), int(c), int(p))] += 1
            n_res += 1
    part_c = dict(
        # `structures` used to be `len(files)`, the sampled list, while the loop
        # above skips every structure with fewer than 30 residues or a
        # coordinate/sequence length mismatch -- so the count included
        # structures that contributed nothing. `coverage` was the literal 1.0,
        # asserted rather than measured, and it is published beside Part B's
        # genuinely measured `pfam_residue_coverage` in the same artefact and
        # quoted against it in the audit's L9 row.
        structures=n_scored_structures,
        structures_sampled=len(files),
        residues=n_res,
        coverage=(n_scored_structures / len(files)) if files else None,
        coverage_definition=(
            "fraction of sampled AlphaFold models that yielded a scorable chain "
            "(>=30 residues, coordinates and sequence of equal length). This is a "
            "structure-level coverage; every residue of a scored chain carries a "
            "structural attribute by construction, which is what makes this "
            "channel an oracle."
        ),
        entropy_bits_secondary_structure3=entropy_bits(ss_c),
        entropy_bits_contact_number_bin=entropy_bits(cn_c),
        entropy_bits_plddt_bin=entropy_bits(pl_c),
        entropy_bits_joint_structural_attributes=entropy_bits(joint_c),
        entropy_bits_residue_identity=entropy_bits(aa_c),
        secondary_structure_fractions={
            k: v / n_res for k, v in sorted(ss_c.items())
        },
    )

    # ---------------------------------------------------------------- Part D
    # The decisive quantity is not the marginal entropy of a label but its
    # entropy *within one sequence*. R2's matched null permutes inside each
    # protein, so any label that is constant across a protein contributes
    # exactly zero by construction. Curated family/domain labels are constant
    # over hundreds of residues; text tokens are not; structural attributes are
    # not. All three are measured over the same 300-symbol window so that the
    # log2(300) sampling ceiling is identical.
    window = 300

    def conditional_entropy_bits(units: list[list]) -> float:
        """Mean within-unit label entropy, Miller-Madow corrected."""
        totals = []
        for labels in units:
            if len(labels) < window:
                continue
            block = labels[:window]
            counts = Counter(block)
            h = entropy_bits(counts)
            h += (len(counts) - 1) / (2 * window * math.log(2))  # Miller-Madow
            totals.append(h)
        if len(totals) < 30:
            raise RuntimeError(f"only {len(totals)} usable units for conditioning")
        return float(np.mean(totals)), len(totals)

    pfam_units, aa_units = [], []
    spans: dict[str, list[tuple[int, int, str]]] = {}
    with open(PFAM_RESIDUE) as handle:
        next(handle)
        for line in handle:
            acc, start, end, pfam = line.rstrip("\n").split("\t")
            if acc in lengths:
                spans.setdefault(acc, []).append((int(start), int(end), pfam))
    annotated = [
        (acc, seq)
        for acc, seq in all_records
        if acc in spans and len(seq) >= window
    ]
    for acc, seq in seeded_sample(annotated, 3000, args.seed):
        labels = ["__none__"] * len(seq)
        for start, end, pfam in spans[acc]:
            for i in range(start - 1, min(end, len(seq))):
                labels[i] = pfam
        pfam_units.append(labels)
        aa_units.append(list(seq))

    struct_units = []
    for path in files:
        ca, plddt, seq = read_alphafold(path)
        if len(ca) < window or len(ca) != len(seq):
            continue
        ss = ca_secondary_structure(ca)
        dist = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
        np.fill_diagonal(dist, 1e9)
        cn_bin = np.clip((dist < 10.0).sum(1) // 4, 0, 7)
        pl_bin = np.clip((plddt // 20).astype(int), 0, 4)
        struct_units.append(list(zip(ss.tolist(), cn_bin.tolist(), pl_bin.tolist())))

    from tg_common import load_text
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("/Data/public/gpt2-large")
    text_units = []
    for doc in load_text(3000):
        ids = tokenizer(doc, return_tensors=None)["input_ids"][: window * 2]
        if len(ids) >= window:
            text_units.append(ids)

    pfam_h, pfam_n = conditional_entropy_bits(pfam_units)
    aa_h, aa_n = conditional_entropy_bits(aa_units)
    struct_h, struct_n = conditional_entropy_bits(struct_units)
    text_h, text_n = conditional_entropy_bits(text_units)
    part_d = dict(
        window_symbols=window,
        sampling_ceiling_bits=math.log2(window),
        text_token_identity_within_document=dict(bits=text_h, units=text_n),
        protein_residue_identity_within_protein=dict(bits=aa_h, units=aa_n),
        protein_pfam_label_within_protein=dict(bits=pfam_h, units=pfam_n),
        protein_structural_attributes_within_protein=dict(bits=struct_h, units=struct_n),
    )

    payload = dict(
        seed=args.seed,
        unit_selection="seeded_permutation_of_all_eligible_units",
        part_a_event_design_ceiling=part_a,
        part_b_curated_residue_annotation=part_b,
        part_c_structural_oracle_attributes=part_c,
        part_d_within_sequence_label_entropy=part_d,
        comparison_bits_per_symbol=dict(
            text_token_identity=args.text_token_entropy_bits,
            protein_residue_identity=part_c["entropy_bits_residue_identity"],
            protein_pfam_domain_label=part_b[
                "entropy_bits_per_residue_including_unannotated"
            ],
            protein_structural_oracle_joint=part_c[
                "entropy_bits_joint_structural_attributes"
            ],
        ),
    )
    out = Path(args.out) if args.out else (
        REPO / "results/transfer_gap_20260729_corrected/tg04"
    )
    write_json(out / "explanation_channel.json", payload)
    print(payload["comparison_bits_per_symbol"])
    print("within-sequence:", {k: (v["bits"] if isinstance(v, dict) and "bits" in v else v) for k, v in part_d.items()})
    print("event ceiling (nats):", part_a["ceiling_by_event_count_nats"])
    print("events needed:", part_a["events_required_for_gate"])
    print("pfam coverage:", part_b["pfam_residue_coverage"])


if __name__ == "__main__":
    main()
