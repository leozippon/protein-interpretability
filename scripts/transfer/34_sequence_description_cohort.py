#!/usr/bin/env python3
"""Build the leakage-controlled sequence-description cohort D3.g is gated on.

**What this stage is for.** Audit §8 item 4 admits D3.g only if its alignment
"beat[s] every applicable baseline, survive[s] description and homology leakage
checks, preserve[s] unrelated concepts, transfer[s] at least one graded
protein-model intervention in the predicted direction, and reproduce[s] on an
unseen protein family". Three of those clauses are properties of the data, not of
the method, and none of them can be checked after the fact on a cohort that was
not built to carry them. This stage builds the cohort and *measures* all three:
genuine Swiss-Prot sequence-description pairs, a fit/eval split taken over
near-duplicate groups rather than over records, and a curated-family-disjoint
holdout for the unseen-family clause.

**The description leak is the one that would be silent.** R3.1 closed because
ZymCTRL's single positive cell dissolved into its own EC conditioning tag (L15).
The analogue here is a curated description that literally names the concept being
predicted: with ``kinase`` in the text, an alignment between a protein activation
and a text embedding can score arbitrarily well while the protein side
contributes nothing, and every artefact would look finite and plausible. So every
surface form of the concept's own identity -- and of the record's own curated
identity, which is the same fact written out -- is replaced by one fixed
placeholder, the removal is recorded per record, and the residual is measured
per concept as a raw-versus-masked leak rate. The declarations live in
:mod:`src.transfer.sequence_description`; this stage owns the corpus, the splits,
the leakage measurements and the artefact.

**The homology leak is the one this programme has already paid for.** L30: a
record-level held-out split is not held out on Swiss-Prot -- 42.5% of held-out
records keep a >=95%-identity relative on the training side and an exact-string
guard reaches at most 18.1 of those points. The remedy is
:mod:`src.transfer.near_duplicates`' group split, and this stage does not assert
that it worked: it runs one DIAMOND all-against-all over the pool and reads the
identity-leakage curve for the record-level split *and* the group split off that
same alignment, so the split is the only thing that differs between the two rows.
``--masking 0`` is not a tuning choice (audit §0.05); it is inherited from
``homology.run_diamond_blastp``, where default repeat masking truncated HSPs and
caused a retraction.

**Nothing here is a knob that can be turned until a criterion is met.** The
thresholds are EXP-R2-213's, frozen before this stage produced a number; each is
carried by a required, never-defaulted flag so that the request states it, and
:func:`resolve` refuses any value that *weakens* one, naming the entry. A
refusal -- a family that cannot be split at the requested fraction, a masked
description that still names a concept, a family-label coverage below the
declared floor -- raises with the measured number in the message. The two
outcomes that are measurements rather than defects are the pre-registered STOP-34
conditions: too thin a concept panel on either reporting split, or straddling
pairs left by the group split. Those write the artefact, record which condition
fired, and end the campaign at this stage.

The stage is CPU-bound and takes ``--device`` for the uniform external-baseline
contract only; the value is recorded and no accelerator is touched.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer import homology, sequence_description as sd  # noqa: E402
from src.transfer.families import (  # noqa: E402
    boundary_leakage,
    family_assignment,
    family_disjoint_split,
    load_cath_superfamilies,
    load_pfam_families,
)
from src.transfer.io import sha256_file, write_json  # noqa: E402
from src.transfer.near_duplicates import (  # noqa: E402
    NEAR_DUPLICATE_CONTAINMENT,
    boundary_containment,
    group_disjoint_split,
    near_duplicate_groups,
)
from src.transfer.relational import homology_disjoint_split  # noqa: E402
from src.transfer.statistics import MINIMUM_BOOTSTRAP_UNITS  # noqa: E402

SCHEMA_VERSION = "r2_transfer_sequence_description_cohort_v1"
DEFAULT_OUT = REPO_ROOT / "results/transfer/sequence_description"

PROVENANCE_MODULES = (
    "src/transfer/sequence_description.py",
    "src/transfer/near_duplicates.py",
    "src/transfer/families.py",
    "src/transfer/homology.py",
    "src/transfer/relational.py",
    "src/transfer/channels.py",
    "src/transfer/io.py",
)

#: Flags that carry a pre-registered decision. None has a default, and
#: :func:`resolve` names every one that is missing rather than the first, because
#: a stage that reports the first sends an operator round the loop once per flag.
REQUIRED_FLAGS = (
    "family_source",
    "go_evidence",
    "length_band",
    "pool_size",
    "holdout_fraction",
    "fit_fraction",
    "seed",
    "work",
    "min_family_groups_per_side",
    "min_concept_groups_per_cell",
    "stop_min_concepts_eval",
    "stop_min_concepts_family_holdout",
    "straddling_refusal_boundary",
)

#: EXP-R2-213's frozen stage-34 criteria, by the flag that carries each. The
#: flags are required so that the request states them, and :func:`resolve`
#: refuses any value that *weakens* one: a pre-registration whose numbers a later
#: invocation can relax is not a pre-registration. Tightening is allowed and is
#: recorded, because tightening cannot manufacture a pass.
PREREGISTERED_EXP_R2_213: dict[str, Any] = {
    "min_family_groups_per_side": 8,
    "min_concept_groups_per_cell": 8,
    "stop_min_concepts_eval": 8,
    "stop_min_concepts_family_holdout": 4,
    "straddling_refusal_boundary": 90.0,
}

#: Shortest function comment that counts as a description. A one-clause stub
#: ("May be a transporter.") carries the curator's uncertainty rather than the
#: protein's function, and a cohort of them would measure how a model handles
#: hedging. Frozen here rather than exposed, because it describes what a
#: description *is* and not what a run is about.
MIN_FUNCTION_CHARS = 40

#: Below this, requiring a curated family label stops filtering the population
#: and starts selecting one -- structurally characterised, well-studied proteins
#: -- and the cohort is no longer a cohort of Swiss-Prot. Measured on the entries
#: that pass every other eligibility rule, reported for both sources, and
#: enforced on the source the run declared.
MIN_FAMILY_COVERAGE = 0.5

#: The five identity boundaries EXP-R2-213 C34-3 fixes. 95% is L30's quoted
#: boundary and 90% is where the group split's pass condition is stated; 80, 70
#: and 50 make the reading a curve, and L30's own residual is quoted at 70.
#: Byte-identical relatives are reported beside the curve rather than as a sixth
#: boundary, so the pre-registered curve is exactly these five.
IDENTITY_BOUNDARIES = (95.0, 90.0, 80.0, 70.0, 50.0)

DIAMOND_EVALUE = 1e-3
DIAMOND_SENSITIVITY = "very-sensitive"

#: Irreducible bounds on what this cohort can support, carried inside every
#: artefact rather than only in a document. Each one is a statement this stage
#: cannot make true by construction, so a reader who quotes a number from here
#: has the caveat in the same file.
LIMITATIONS: dict[str, str] = {
    "masking_is_lexical_and_paraphrase_survives": (
        "removing 'kinase activity' does not remove 'catalyses the "
        "transfer of a phosphate group from ATP', which is the same fact "
        "in other words. No lexical procedure removes it. The per-concept "
        "raw-versus-masked rates bound the literal leak; the residual "
        "paraphrase leak is bounded only by an alignment's own baselines, "
        "and a masked-description result is therefore a floor on the "
        "leak-free result rather than a leak-free result"
    ),
    "annotation_is_not_independent_evidence": (
        "a GO term inferred electronically from a family signature is "
        "downstream of the same family label this cohort splits on. "
        "--go-evidence records which population was built; under 'all' a "
        "concept label and a family label are not independent, and a "
        "biological-knowledge claim needs evidence from outside UniProt "
        "whichever policy was used"
    ),
    "no_control_over_model_pretraining": (
        "Swiss-Prot lies inside UniRef50 by construction, so every record "
        "here is a candidate member of a protein decoder's pretraining "
        "corpus. This cohort's splits control leakage between ITS OWN "
        "fitting and evaluation sides and nothing else; the retrieval "
        "bound and the homology stratification are separate instruments "
        "and neither is replaced by a family-disjoint split"
    ),
    "curated_description_is_not_a_neutral_text": (
        "a Swiss-Prot function comment is written in a house style over a "
        "small vocabulary and cites the same evidence the annotations come "
        "from. It is a genuine sequence-description pair, which is what the "
        "gate asks for, and it is not a sample of natural language about "
        "proteins"
    ),
    "sequence_level_only": (
        "every record is one sequence and one description. No residue-level "
        "correspondence is declared or measured here, so a downstream "
        "estimand that wants one must declare its own and cannot read it "
        "off this cohort"
    ),
}


# ------------------------------------------------------------------- the pool


def reservoir_sample(
    stream: Iterator[Any], *, size: int, seed: int
) -> tuple[list[Any], int]:
    """A uniform draw of ``size`` items from a stream of unknown length.

    Appendix B rule 1 -- never take the first N records of a biological corpus --
    with the corpus read once. Swiss-Prot XML is ordered by entry name, so a
    head-of-file prefix is a set of near-clonal homologues of whatever sorts
    first, and the rule's usual remedy (permute, then take) needs the population
    in memory. Algorithm R gives the same uniform draw in one pass, and because
    the generator is seeded the draw is reproducible from the seed and the corpus
    alone.

    Returns the reservoir and the number of items seen, so a caller can report
    the sampling fraction rather than only the sample.
    """

    if size < 1:
        raise ValueError("a pool needs at least one record")
    generator = np.random.default_rng(seed)
    reservoir: list[Any] = []
    seen = 0
    for item in stream:
        if seen < size:
            reservoir.append(item)
        else:
            position = int(generator.integers(0, seen + 1))
            if position < size:
                reservoir[position] = item
        seen += 1
    return reservoir, seen


def eligible_entries(
    *,
    args: argparse.Namespace,
    ontology: sd.GoOntology,
    pfam: dict[str, frozenset[str]],
    cath: dict[str, frozenset[str]],
) -> Iterator[dict[str, Any]]:
    """Swiss-Prot entries that can carry a sequence-description pair, in order.

    Each rejection is counted under the first rule it fails, which makes the
    counts a partition of the corpus rather than an overlapping tally. The
    counter is written into ``args`` so ``main`` can report it beside the pool;
    an eligibility rule whose cost is not reported is a population change nobody
    can see.
    """

    low, high = args.length_band
    counts = args.rejections
    for entry in sd.iter_swissprot_entries(args.swissprot_xml):
        if args.max_entries_scanned and args.scanned >= args.max_entries_scanned:
            return
        args.scanned += 1
        if not low <= len(entry.sequence) <= high:
            counts["length_out_of_band"] += 1
            continue
        if not set(entry.sequence) <= sd.STANDARD_RESIDUES:
            counts["non_standard_residue"] += 1
            continue
        if not entry.protein_name:
            counts["no_recommended_name"] += 1
            continue
        function_text = " ".join(entry.function_texts)
        if len(function_text) < MIN_FUNCTION_CHARS:
            counts["function_text_too_short"] += 1
            continue

        annotations = [
            annotation
            for annotation in entry.go
            if args.go_evidence == "all" or annotation.evidence != sd.IEA_EVIDENCE
        ]
        args.go_annotations_seen += len(entry.go)
        args.go_annotations_kept += len(annotations)
        canonical: list[str] = []
        terms: list[str] = []
        for annotation in annotations:
            primary = ontology.canonical(annotation.go_id)
            if primary is None:
                args.go_unresolved.add(annotation.go_id)
                continue
            if primary != annotation.go_id:
                args.go_remapped += 1
            canonical.append(primary)
            terms.append(annotation.term)
        if not canonical and not entry.ec:
            counts["no_go_or_ec_annotation"] += 1
            continue

        args.family_labelled["pfam"] += int(bool(pfam.get(entry.accession)))
        args.family_labelled["cath_superfamily"] += int(bool(cath.get(entry.accession)))
        args.family_eligible += 1
        families = (pfam if args.family_source == "pfam" else cath).get(entry.accession)
        if not families:
            counts["no_family_label"] += 1
            continue

        description = sd.canonical_description(entry.protein_name, entry.function_texts)
        verbatim = sd.sequence_in_description(entry.sequence, description)
        if verbatim["contains_full_sequence"] or verbatim["shared_runs"]:
            counts["description_quotes_own_sequence"] += 1
            args.verbatim_examples.append(entry.accession)
            continue

        args.eligible += 1
        yield {
            "accession": entry.accession,
            "sequence": entry.sequence,
            "name": entry.protein_name,
            "function_text": function_text,
            "description_raw": description,
            "ec": entry.ec,
            "go": tuple(dict.fromkeys(canonical)),
            "go_terms": tuple(terms),
            "pfam_entries": entry.pfam,
            "interpro_entries": entry.interpro,
            "families": frozenset(families),
            "pfam": tuple(sorted(pfam.get(entry.accession, frozenset()))),
            "cath": tuple(sorted(cath.get(entry.accession, frozenset()))),
        }


# --------------------------------------------------------------- the alignment


def write_pool_fasta(path: Path, sequences: list[str]) -> None:
    """One record per pool index. Positional ids, because every join is by index."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, sequence in enumerate(sequences):
            handle.write(f">r{index}\n{sequence}\n")


def align_pool(
    tool: homology.DiamondTool, fasta: Path, work: Path, *, threads: int, n_records: int
) -> tuple[list[tuple[int, int, int, int, int]], dict[str, Any]]:
    """All-against-all over the pool, through the module that owns the settings.

    ``homology.run_diamond_blastp`` is used rather than a local subprocess call
    because ``--masking 0`` lives inside it: default repeat masking truncated the
    HSPs of a repeat-selected cohort and put a verbatim corpus member in the
    "diverged relative" bin, which is the defect §0.05 retracts. A second call
    site would be a second place for that flag to go missing.

    ``--max-target-seqs`` is the pool size and not a small constant: the reading
    is the maximum identity to *any* record on the other side of a split, so a
    truncated hit list would silently under-report leakage.
    """

    database = homology.build_database(
        tool,
        fasta,
        work / "pool.dmnd",
        threads=threads,
        tmpdir=work / "diamond_tmp",
    )
    hits_tsv = work / "hits.tsv"
    command, log = homology.run_diamond_blastp(
        tool,
        database,
        fasta,
        hits_tsv,
        threads=threads,
        sensitivity=DIAMOND_SENSITIVITY,
        evalue=DIAMOND_EVALUE,
        max_target_seqs=n_records,
    )
    hits = homology.parse_hits(hits_tsv)
    pairs = [
        (int(hit.query[1:]), int(hit.subject[1:]), hit.nident, hit.qlen, hit.slen)
        for hit in hits
    ]
    return pairs, {
        "tool": tool.record(),
        "database": database.record(),
        "command": command,
        "log_tail": log,
        "n_hits": len(pairs),
        "identity_definition": (
            "n_identical over the SHORTER of the two sequences, not over the "
            "alignment: pident is identity within the aligned region and calls a "
            "60%-length fragment a near-duplicate"
        ),
    }


def leakage_reading(
    pairs: list[tuple[int, int, int, int, int]],
    *,
    train: np.ndarray,
    held_out: np.ndarray,
    label: str,
    note: str,
) -> dict[str, Any]:
    """The identity-leakage curve one split leaves, off the whole-pool alignment.

    ``train`` and ``held_out`` are boolean masks over the *pool*, so two readings
    can differ in the split alone -- which is the entire point of reporting the
    record-level row beside the group row.

    Two statistics, because C34-3 states the pass condition on one and L30 quotes
    the other. ``straddling_pairs`` counts unordered cross-boundary pairs at or
    above each identity, which is what "zero straddling pairs at >=90%" refers to;
    ``at_or_above`` counts held-out *records* keeping a relative, which is L30's
    42.5% and is the one a reader compares against it. They cannot disagree about
    zero, and away from zero they measure different things.
    """

    held_index = np.flatnonzero(held_out)
    if held_index.size == 0:
        raise ValueError(f"{label}: no held-out records to read leakage on")
    if np.any(train & held_out):
        raise ValueError(f"{label}: a record is on both sides of the split")
    best = {int(index): 0.0 for index in held_index}
    straddling: dict[float, set[tuple[int, int]]] = {
        boundary: set() for boundary in IDENTITY_BOUNDARIES
    }
    identical_relative: set[int] = set()
    for query, subject, nident, qlen, slen in pairs:
        if query == subject or query not in best or not train[subject]:
            continue
        identity = 100.0 * nident / min(qlen, slen)
        if identity > best[query]:
            best[query] = identity
        if identity >= 100.0:
            identical_relative.add(query)
        for boundary in IDENTITY_BOUNDARIES:
            if identity >= boundary:
                straddling[boundary].add((min(query, subject), max(query, subject)))
    values = np.array([best[int(index)] for index in held_index], dtype=np.float64)
    return {
        "split": label,
        "note": note,
        "n_train": int(train.sum()),
        "n_held_out": int(values.size),
        "at_or_above": {
            f"{boundary:g}": {
                "n": int((values >= boundary).sum()),
                "fraction": float((values >= boundary).mean()),
            }
            for boundary in IDENTITY_BOUNDARIES
        },
        "straddling_pairs": {
            f"{boundary:g}": len(entries) for boundary, entries in straddling.items()
        },
        "n_held_out_with_a_byte_identical_relative": len(identical_relative),
        "median_max_identity": float(np.median(values)),
        "mean_max_identity": float(values.mean()),
        "n_with_no_detectable_homologue": int((values == 0.0).sum()),
        "fraction_with_no_detectable_homologue": float((values == 0.0).mean()),
        "statistics": (
            "at_or_above counts HELD-OUT RECORDS whose maximum identity to any "
            "training-side record reaches the boundary (L30's statistic); "
            "straddling_pairs counts unordered cross-boundary PAIRS at or above it "
            "(C34-3's pass condition). Identity is n_identical over the shorter of "
            "the two sequences"
        ),
    }


# ------------------------------------------------------------------- the stage


def pre_registration_record(args: argparse.Namespace) -> dict[str, Any]:
    """What EXP-R2-213 asked of this stage, as the run resolved it.

    Written into the artefact from the resolved arguments rather than from the
    frozen table, so a reader sees the criteria the run was actually held to
    beside the values they were frozen at. :func:`resolve` has already refused
    anything weaker, so the two can differ only by a tightening.
    """

    return {
        "entry": "EXP-R2-213",
        "criteria_implemented_here": {
            "C34-1": (
                "concept-name masking is a refusal: the per-record term set is "
                "re-applied to description_masked and the run raises if any "
                "span survives. Records quoting their own sequence are "
                "excluded and counted under corpus_scan"
            ),
            "C34-2": (
                "the raw description is retained beside the masked one and the "
                "per-concept raw-versus-masked leak rates are reported under "
                "description_leakage; the paraphrase residual is declared "
                "under limitations"
            ),
            "C34-3": (
                "one DIAMOND all-against-all, read for the record-level and "
                "the group-level split off that same alignment at "
                f"{list(IDENTITY_BOUNDARIES)}; zero straddling pairs at or "
                f"above {args.straddling_refusal_boundary} under the group "
                "split is a STOP condition"
            ),
            "C34-4": (
                "family-disjoint holdout with at least "
                f"{args.min_family_groups_per_side} family groups per side, "
                "unlabelled units refused (every pool record carries a curated "
                "family by eligibility, so the refusal cannot be reached "
                "silently)"
            ),
            "C34-5": (
                f"at least {args.min_concept_groups_per_cell} bearing and "
                f"{args.min_concept_groups_per_cell} non-bearing near-duplicate "
                f"groups in each of {list(sd.DECIDING_SPLITS)}"
            ),
            "C34-6": (
                "the surviving-concept count as a function of the per-cell "
                f"floor at {list(sd.ADMISSION_FLOOR_CURVE)}, computed before "
                "the declared floor is applied"
            ),
            "STOP-34": (
                f"fewer than {args.stop_min_concepts_eval} concepts admissible "
                f"in eval, or fewer than "
                f"{args.stop_min_concepts_family_holdout} in family_holdout, or "
                "straddling pairs left by the group split"
            ),
        },
        "criteria_that_belong_to_stages_35_and_36": (
            "per-layer evaluation, float32 arithmetic and the group bootstrap "
            "over near-duplicate groups are conditions on readouts this stage "
            "does not take: it has no layer, computes no interval and does no "
            "floating-point estimation. What this stage owes them is the unit "
            "they resample -- dup_group -- and the C34-5 floor, which is "
            "MINIMUM_BOOTSTRAP_UNITS so that no admitted concept can reach "
            "either reporting split with fewer than eight groups on a side"
        ),
        "minimum_bootstrap_units": MINIMUM_BOOTSTRAP_UNITS,
        "frozen_values": PREREGISTERED_EXP_R2_213,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family-source", default=None, choices=("pfam", "cath_superfamily"),
        help="REQUIRED. Which curated label defines a family for the holdout. "
        "cath_superfamily is the stricter grouping -- Gene3D superfamilies join "
        "remote homologues Pfam separates -- and is preferred where its coverage "
        "permits; both coverages are measured on this run's eligible entries and "
        "reported, and the declared source is refused below the coverage floor")
    parser.add_argument(
        "--go-evidence", default=None, choices=sd.GO_EVIDENCE_POLICIES,
        help="REQUIRED. 'all' keeps every GO cross-reference; 'non_iea' drops the "
        "electronically inferred ones. An IEA term assigned by a UniRule that "
        "fires on a family signature is not independent of the family label, so "
        "the two policies define different populations and neither may be a "
        "default")
    parser.add_argument(
        "--length-band", type=int, nargs=2, default=None, metavar=("LOW", "HIGH"),
        help="REQUIRED. Residue band, inclusive at both ends. Appendix B rule 13: "
        "a per-stage band is a legitimate compute choice and an undeclared one "
        "lets a verdict be read as covering a population it never measured")
    parser.add_argument(
        "--pool-size", type=int, default=None,
        help="REQUIRED. Records drawn, by seeded reservoir over every eligible "
        "entry of the corpus")
    parser.add_argument(
        "--holdout-fraction", type=float, default=None,
        help="REQUIRED. Share of the pool carved off as the family-disjoint "
        "holdout, before the fit/eval split is taken over what remains")
    parser.add_argument(
        "--fit-fraction", type=float, default=None,
        help="REQUIRED. Share of the non-holdout records on the fit side of the "
        "near-duplicate-group split")
    parser.add_argument("--seed", type=int, default=None, help="REQUIRED.")
    parser.add_argument(
        "--work", type=Path, default=None,
        help="REQUIRED. Scratch directory for the DIAMOND binary, database and "
        "hit table. Kept outside the repository: a 25 GB index and an extracted "
        "binary must never enter version control")
    parser.add_argument(
        "--min-family-groups-per-side", type=int, default=None,
        help="REQUIRED (EXP-R2-213 C34-4). Family groups each side of the "
        "family-disjoint holdout must carry. A side holding one family cannot "
        "support a claim about unseen families, and the floor is the same eight "
        "units every interval downstream resamples over")
    parser.add_argument(
        "--min-concept-groups-per-cell", type=int, default=None,
        help="REQUIRED (EXP-R2-213 C34-5). Distinct near-duplicate groups that "
        "must bear a concept, AND that must not bear it, in each deciding split. "
        "Counted in groups and not records: forty near-clonal records are one "
        "unit for every bootstrap this campaign will report")
    parser.add_argument(
        "--stop-min-concepts-eval", type=int, default=None,
        help="REQUIRED (EXP-R2-213 STOP-34). Below this many concepts admissible "
        "in eval the campaign stops at this stage and the counts are the result")
    parser.add_argument(
        "--stop-min-concepts-family-holdout", type=int, default=None,
        help="REQUIRED (EXP-R2-213 STOP-34). The same for the family holdout, "
        "which is allowed to be thinner because it is one split of the cohort "
        "rather than the panel a map is chosen on")
    parser.add_argument(
        "--straddling-refusal-boundary", type=float, default=None,
        help="REQUIRED (EXP-R2-213 C34-3). Percent identity at and above which "
        "the group split must leave zero straddling pairs. Must be one of the "
        "declared boundaries and may only be TIGHTENED below the pre-registered "
        "90: a criterion an invocation can loosen is not a criterion")
    parser.add_argument(
        "--max-entries-scanned", type=int, default=0,
        help="stop the corpus scan after this many entries. 0 scans the whole "
        "release, which is the only setting whose draw is a sample: a capped "
        "scan is a head-of-file PREFIX of a name-ordered corpus, the artefact "
        "records it as such, and its numbers are a smoke test and not a cohort")
    parser.add_argument(
        "--swissprot-xml", type=Path, default=sd.SWISSPROT_XML)
    parser.add_argument("--go-obo", type=Path, default=sd.GO_OBO)
    parser.add_argument("--enzyme-dat", type=Path, default=sd.ENZYME_DAT)
    parser.add_argument("--interpro-entry-list", type=Path, default=sd.INTERPRO_ENTRY_LIST)
    parser.add_argument(
        "--diamond-tarball", type=Path,
        default=REPO_ROOT / "external_resources/tools/diamond-linux64-v2.1.24.tar.gz",
        help="the staged tarball; its .sha256 sidecar is verified before "
        "extraction, because two tarballs are staged and only one is published")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument(
        "--device", default="cuda:0",
        help="accepted for the uniform external-baseline contract and recorded in "
        "the artefact. This stage is CPU-bound and touches no accelerator")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser


def resolve(args: argparse.Namespace) -> None:
    """Refuse an incoherent request before the corpus is opened."""

    missing = [flag for flag in REQUIRED_FLAGS if getattr(args, flag) is None]
    if missing:
        raise ValueError(
            "this stage needs "
            + ", ".join(f"--{flag.replace('_', '-')}" for flag in missing)
            + ". Every one of them is a pre-registered decision about which "
            "population is built and how it is divided; a default would let the "
            "population change without the request changing"
        )
    low, high = args.length_band
    if not 1 <= low <= high:
        raise ValueError(f"--length-band {low} {high} is not an ascending residue band")
    if not 0.0 < args.holdout_fraction < 1.0:
        raise ValueError("--holdout-fraction must lie strictly between 0 and 1")
    if not 0.0 < args.fit_fraction < 1.0:
        raise ValueError("--fit-fraction must lie strictly between 0 and 1")
    if args.pool_size < 1:
        raise ValueError("--pool-size must be positive")
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if args.max_entries_scanned < 0:
        raise ValueError("--max-entries-scanned must not be negative")

    # The pre-registration is frozen, so a value that weakens one of its criteria
    # is refused by name rather than recorded as a setting. Every count may be
    # raised and the identity boundary may be lowered; both directions make the
    # criterion harder, and neither can turn a fail into a pass.
    weakened = [
        f"--{flag.replace('_', '-')} {getattr(args, flag)} weakens EXP-R2-213's "
        f"{frozen}"
        for flag, frozen in PREREGISTERED_EXP_R2_213.items()
        if flag != "straddling_refusal_boundary" and getattr(args, flag) < frozen
    ]
    if args.straddling_refusal_boundary > PREREGISTERED_EXP_R2_213[
        "straddling_refusal_boundary"
    ]:
        weakened.append(
            f"--straddling-refusal-boundary {args.straddling_refusal_boundary} is "
            "above EXP-R2-213's 90, which would let straddling pairs through"
        )
    if args.straddling_refusal_boundary not in IDENTITY_BOUNDARIES:
        weakened.append(
            f"--straddling-refusal-boundary {args.straddling_refusal_boundary} is "
            f"not one of the declared boundaries {list(IDENTITY_BOUNDARIES)}, so "
            "the pass condition would be read at an identity the curve does not "
            "report"
        )
    if weakened:
        raise ValueError(
            "this stage implements EXP-R2-213's frozen stage-34 criteria and "
            "cannot be run below them: " + "; ".join(weakened)
        )

    # Accumulators the eligibility scan writes into. Declared here so that a
    # rejection reason cannot be invented at the point it fires: the keys are the
    # eligibility rules, and every one of them is reported whether or not it
    # rejected anything.
    args.rejections = {
        "length_out_of_band": 0,
        "non_standard_residue": 0,
        "no_recommended_name": 0,
        "function_text_too_short": 0,
        "no_go_or_ec_annotation": 0,
        "no_family_label": 0,
        "description_quotes_own_sequence": 0,
    }
    args.family_labelled = {"pfam": 0, "cath_superfamily": 0}
    args.family_eligible = 0
    args.go_annotations_seen = 0
    args.go_annotations_kept = 0
    args.go_remapped = 0
    args.go_unresolved = set()
    args.verbatim_examples = []
    args.scanned = 0
    args.eligible = 0


#: Namespace attributes the eligibility scan accumulates into. They are counters
#: and not settings, so they are kept out of the artefact's ``settings`` echo and
#: reported under ``corpus_scan`` where a reader looks for them.
SCAN_COUNTERS = (
    "rejections",
    "family_labelled",
    "family_eligible",
    "go_annotations_seen",
    "go_annotations_kept",
    "go_remapped",
    "go_unresolved",
    "verbatim_examples",
    "scanned",
    "eligible",
)


def main() -> None:
    args = build_parser().parse_args()
    resolve(args)
    args.out.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)

    print(f"[paths] swissprot {Path(args.swissprot_xml).resolve()}")
    print(f"[paths] go        {Path(args.go_obo).resolve()}")
    print(f"[paths] enzyme    {Path(args.enzyme_dat).resolve()}")
    print(f"[paths] interpro  {Path(args.interpro_entry_list).resolve()}")
    print(f"[paths] work      {args.work.resolve()}")
    print(f"[paths] out       {args.out.resolve()}")
    print(f"[device] {args.device} recorded; this stage runs on host CPU")

    ontology = sd.load_go_ontology(args.go_obo)
    enzyme = sd.load_enzyme_descriptions(args.enzyme_dat)
    interpro_names = sd.load_interpro_entry_names(args.interpro_entry_list)
    pfam = load_pfam_families()
    cath = load_cath_superfamilies()
    print(
        f"[labels] GO {len(ontology.names)} terms ({ontology.version}), "
        f"ENZYME {len(enzyme)}, InterPro {len(interpro_names)}, "
        f"Pfam {len(pfam)} accessions, CATH {len(cath)} accessions"
    )

    pool, scanned = reservoir_sample(
        eligible_entries(args=args, ontology=ontology, pfam=pfam, cath=cath),
        size=args.pool_size,
        seed=args.seed,
    )
    if len(pool) < args.pool_size:
        raise RuntimeError(
            f"only {len(pool)} of {args.pool_size} requested records are eligible "
            f"in the {args.scanned} entries scanned; the pool this stage is about "
            "cannot be built. Widen --length-band, relax --go-evidence, or scan "
            "more of the corpus -- do not report a smaller pool as the requested one"
        )
    # Canonical record order, so the artefact is a function of the seed and the
    # corpus rather than of reservoir replacement order.
    pool.sort(key=lambda item: item["accession"])
    accessions = [item["accession"] for item in pool]
    sequences = [item["sequence"] for item in pool]
    if len(set(accessions)) != len(accessions):
        raise RuntimeError("the pool holds a repeated accession; the draw is not a set")
    print(
        f"[pool] {len(pool)} records drawn from {args.scanned} entries scanned, "
        f"{len(set(sequences))} distinct sequences"
    )

    coverage = {
        source: (
            args.family_labelled[source] / args.family_eligible
            if args.family_eligible
            else 0.0
        )
        for source in ("pfam", "cath_superfamily")
    }
    print(
        f"[families] coverage over {args.family_eligible} otherwise-eligible "
        f"entries: pfam {coverage['pfam']:.3f}, cath {coverage['cath_superfamily']:.3f}"
    )
    if coverage[args.family_source] < MIN_FAMILY_COVERAGE:
        raise RuntimeError(
            f"--family-source {args.family_source} labels "
            f"{coverage[args.family_source]:.3f} of the {args.family_eligible} "
            f"otherwise-eligible entries, below the declared floor "
            f"{MIN_FAMILY_COVERAGE}. Below it the label requirement selects a "
            "sub-population -- structurally characterised, well-studied proteins "
            "-- rather than filtering one, and the cohort would no longer be about "
            f"Swiss-Prot. The other source covers "
            f"{coverage['pfam' if args.family_source != 'pfam' else 'cath_superfamily']:.3f}"
        )

    # ---------------------------------------------------------------- grouping
    groups, grouping = near_duplicate_groups(sequences, unit="residues")
    print(
        f"[near-duplicates] {grouping['n_groups']} groups, largest "
        f"{grouping['largest_group_size']}"
    )

    # The family holdout must also be near-duplicate clean, or a record in the
    # "unseen family" would have a near-copy on the fitting side and the clause
    # would be answered by memorisation. Rather than splitting and then hoping,
    # each record's near-duplicate group is carried into the family grouping as a
    # label of its own, so the family split cannot divide one by construction --
    # and the property is still re-verified on the returned masks below.
    labels = {
        accession: frozenset(item["families"] | {f"__near_duplicate_group_{group}"})
        for accession, item, group in zip(accessions, pool, groups, strict=True)
    }
    assignment = family_assignment(
        accessions, labels, source=args.family_source, multi_label="merge"
    )
    if assignment.unit_ids != tuple(accessions):
        raise RuntimeError(
            "the family assignment dropped or reordered units; every pool record "
            "carries a curated family by construction, so this cannot happen "
            "without the eligibility rule having changed underneath it"
        )
    holdout_split = family_disjoint_split(
        assignment,
        seed=args.seed + 1,
        train_fraction=1.0 - args.holdout_fraction,
        min_groups_per_side=args.min_family_groups_per_side,
    )
    holdout = np.asarray(holdout_split.test, dtype=bool)
    kept = ~holdout
    print(
        f"[family holdout] {int(holdout.sum())} records in "
        f"{int(np.unique(assignment.group_ids[holdout]).size)} family groups"
    )

    # ------------------------------------------------------------------ splits
    kept_index = np.flatnonzero(kept)
    n_fit = int(round(args.fit_fraction * kept_index.size))
    subset_groups = groups[kept_index]
    fit_mask_subset, group_split = group_disjoint_split(
        subset_groups, n_train=n_fit, seed=args.seed + 2
    )
    # The pre-repair procedure, restated exactly: every record its own group,
    # which is the singleton case of the same mask, at the same seed and the same
    # fraction. Written this way so that the two leakage rows below differ in the
    # GROUPING and in nothing else -- not the seed, not the fraction, not the code
    # path (ops/measure_pool_homology_leakage.py takes the same care).
    record_mask_subset = homology_disjoint_split(
        np.arange(kept_index.size),
        train_fraction=n_fit / kept_index.size,
        seed=args.seed + 2,
        min_side=1,
    )

    splits = np.array(["family_holdout"] * len(pool), dtype=object)
    splits[kept_index[fit_mask_subset]] = "fit"
    splits[kept_index[~fit_mask_subset]] = "eval"
    print(
        f"[splits] fit {int((splits == 'fit').sum())}, "
        f"eval {int((splits == 'eval').sum())}, "
        f"family_holdout {int((splits == 'family_holdout').sum())}"
    )

    certificates = verify_splits(
        splits=splits,
        groups=groups,
        pool=pool,
        assignment_groups=assignment.group_ids,
    )

    # ----------------------------------------------------------- concept labels
    # One closure per record, computed once: it is the object every concept's
    # positive rule reads, and recomputing it per concept would make the ancestor
    # walk the cost of the stage.
    propagated = [ontology.close(item["go"]) for item in pool]
    # A concept the ontology release no longer carries is undefined for every
    # record rather than an exception: the declaration has gone stale, which is a
    # different fact from a thin cohort, and admit_concepts reports it as such.
    declaration = {
        spec.concept_id: sd.declaration_reason(spec, ontology) for spec in sd.CONCEPTS
    }
    label_columns: dict[str, list[int | None]] = {
        spec.concept_id: (
            [None] * len(pool)
            if declaration[spec.concept_id] is not None
            else [
                sd.concept_label(
                    spec, go_propagated=closure, ec=item["ec"], ontology=ontology
                )
                for item, closure in zip(pool, propagated, strict=True)
            ]
        )
        for spec in sd.CONCEPTS
    }
    admission = sd.admit_concepts(
        label_columns,
        list(splits),
        [int(group) for group in groups],
        ontology=ontology,
        min_groups_per_cell=args.min_concept_groups_per_cell,
    )
    print(f"[concepts] admitted {len(admission.admitted)} of {len(sd.CONCEPTS)}: "
          f"{', '.join(admission.admitted) or '(none)'}")
    for split in sd.DECIDING_SPLITS:
        print(
            f"[concepts] admissible in {split}: "
            f"{len(admission.admissible_per_split[split])}"
        )
    print(
        "[concepts] floor curve (groups per cell -> surviving): "
        + ", ".join(
            f"{floor}:{cell['both']}" for floor, cell in admission.floor_curve.items()
        )
    )

    # ---------------------------------------------------------------- masking
    # Every DECLARED concept's forms, not only the admitted ones. C34-1 is stated
    # over declared surface forms, and making the masked text depend on the
    # admission outcome would mean two runs of the same corpus carried different
    # descriptions because their pools admitted different concepts.
    concept_forms = {
        spec.concept_id: sd.concept_surface_forms(spec, ontology=ontology)
        for spec in sd.CONCEPTS
    }
    shared_forms = sorted({form for forms in concept_forms.values() for form in forms})

    records: list[sd.SequenceDescriptionRecord] = []
    total_spans = 0
    surviving: list[dict[str, Any]] = []
    for index, (item, group, split, closure) in enumerate(
        zip(pool, groups, splits, propagated, strict=True)
    ):
        identity = sd.record_identity_forms(
            go_ids=item["go"],
            go_terms=item["go_terms"],
            ec=item["ec"],
            pfam_entries=item["pfam_entries"],
            interpro_entries=item["interpro_entries"],
            ontology=ontology,
            enzyme=enzyme,
            interpro_names=interpro_names,
        )
        terms = (*identity, *shared_forms)
        masked, matched, spans = sd.mask_description(item["description_raw"], terms)
        total_spans += spans
        # C34-1 is a refusal and is checked here, on the object that was built:
        # re-applying the same term set to the masked text must find nothing. This
        # covers the record's own identity forms as well as the concept forms, so
        # it is the whole of "any declared surface form survives" rather than the
        # per-concept half of it.
        _, residual_terms, residual_spans = sd.mask_description(masked, terms)
        if residual_spans:
            surviving.append(
                {
                    "accession": item["accession"],
                    "terms": list(residual_terms),
                    "spans": residual_spans,
                }
            )
        records.append(
            sd.SequenceDescriptionRecord(
                accession=item["accession"],
                sequence=item["sequence"],
                length=len(item["sequence"]),
                name=item["name"],
                function_text=item["function_text"],
                description_raw=item["description_raw"],
                description_masked=masked,
                masked_terms=matched,
                ec=item["ec"],
                go=item["go"],
                go_propagated=closure,
                pfam=item["pfam"],
                cath=item["cath"],
                dup_group=int(group),
                family_group=f"{args.family_source}:g{int(assignment.group_ids[index])}",
                split=str(split),
            )
        )
    print(f"[masking] {total_spans} spans replaced across {len(records)} descriptions")
    if surviving:
        raise RuntimeError(
            f"C34-1 (EXP-R2-213): {len(surviving)} masked descriptions still "
            f"contain a declared surface form (first: {surviving[:3]}). The cohort "
            "is refused. This is a defect in the surface-form table or in the "
            "matcher, not a property of the corpus, and it is not a threshold to "
            "be relaxed"
        )

    description_leakage = measure_description_leakage(
        records, label_columns, concept_forms, tuple(spec.concept_id for spec in sd.CONCEPTS)
    )
    print(
        "[description leak] "
        + ", ".join(
            f"{concept} {cell['positive_raw_rate']:.3f}->{cell['positive_masked_rate']:.3f}"
            for concept, cell in description_leakage["per_concept"].items()
        )
    )

    # -------------------------------------------------------------- homology
    tool = homology.prepare_diamond(
        args.diamond_tarball,
        Path(str(args.diamond_tarball) + ".sha256"),
        args.work / "diamond",
    )
    fasta = args.work / "pool.fasta"
    write_pool_fasta(fasta, sequences)
    print(f"[diamond] {tool.version}: all-against-all over {len(sequences)} records")
    pairs, aligner = align_pool(
        tool, fasta, args.work, threads=args.threads, n_records=len(sequences)
    )
    print(f"[diamond] {len(pairs)} hits")

    fit = splits == "fit"
    evaluation = splits == "eval"
    record_fit = np.zeros(len(pool), dtype=bool)
    record_fit[kept_index[record_mask_subset]] = True
    record_eval = np.zeros(len(pool), dtype=bool)
    record_eval[kept_index[~record_mask_subset]] = True
    readings = [
        leakage_reading(
            pairs,
            train=record_fit,
            held_out=record_eval,
            label="record_level",
            note="the procedure L30 replaced: every record its own group. Reported "
            "as the contrast, not as an option",
        ),
        leakage_reading(
            pairs,
            train=fit,
            held_out=evaluation,
            label="near_duplicate_group",
            note="the committed procedure: whole near-duplicate groups on one side",
        ),
        leakage_reading(
            pairs,
            train=fit | evaluation,
            held_out=splits == "family_holdout",
            label="family_holdout_vs_fit_and_eval",
            note="the split the unseen-family clause of the §8 item 4 gate is read "
            "on, measured against everything the method may see",
        ),
    ]
    for reading in readings:
        cell = reading["at_or_above"]["95"]
        print(
            f"[leakage:{reading['split']}] {cell['n']}/{reading['n_held_out']} "
            f"({cell['fraction']:.1%}) at >=95%, straddling pairs "
            f"{reading['straddling_pairs']}, median max identity "
            f"{reading['median_max_identity']:.1f}%"
        )

    stop = stop_34(args, admission=admission, readings=readings)
    if stop:
        for entry in stop:
            print(f"[STOP-34] {entry['condition']}: {entry['reading']}")

    raw = {}
    for name, source in (("pool.fasta", fasta), ("hits.tsv", args.work / "hits.tsv")):
        destination = args.out / f"{name}.gz"
        # mtime=0, so the archive is a function of its contents alone. gzip writes
        # the wall clock into its header by default, and the digest recorded beside
        # it then changes between two runs that produced the identical alignment --
        # which defeats the only reason to record a digest. Verified: two runs of
        # this stage at one seed produce byte-identical pool.fasta and hits.tsv.
        with source.open("rb") as reader, destination.open("wb") as handle:
            with gzip.GzipFile(fileobj=handle, mode="wb", mtime=0) as writer:
                shutil.copyfileobj(reader, writer)
        raw[name] = {"path": destination.name, "sha256": sha256_file(destination)}

    # ---------------------------------------------------------------- artefact
    # The records are written on a stop as well as on an admission. A stop is a
    # measurement about the cohort and the cohort is what it is about, so
    # discarding it would leave the counts unauditable.
    sd.write_records(args.out / "records.jsonl", records)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "STOP-34" if stop else "COHORT_ADMITTED",
        "stop": stop,
        "pre_registration": pre_registration_record(args),
        "settings": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
            if key not in SCAN_COUNTERS
        },
        "declared_constants": {
            "min_function_chars": MIN_FUNCTION_CHARS,
            "min_family_coverage": MIN_FAMILY_COVERAGE,
            "identity_boundaries": list(IDENTITY_BOUNDARIES),
            "near_duplicate_containment": NEAR_DUPLICATE_CONTAINMENT,
            "verbatim_shingle": sd.VERBATIM_SHINGLE,
            "mask_placeholder": sd.MASK_PLACEHOLDER,
            "concept_max_root_depth": sd.CONCEPT_MAX_ROOT_DEPTH,
            "diamond_evalue": DIAMOND_EVALUE,
            "diamond_sensitivity": DIAMOND_SENSITIVITY,
        },
        "provenance": {
            "runner": {
                "path": "scripts/transfer/34_sequence_description_cohort.py",
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "modules": {
                name: sha256_file(REPO_ROOT / name) for name in PROVENANCE_MODULES
            },
            "inputs": {
                "swissprot_xml": {
                    "path": str(args.swissprot_xml),
                    "sha256": sha256_file(args.swissprot_xml),
                },
                "go_obo": {
                    "path": str(args.go_obo),
                    "sha256": sha256_file(args.go_obo),
                    "data_version": ontology.version,
                },
                "enzyme_dat": {
                    "path": str(args.enzyme_dat),
                    "sha256": sha256_file(args.enzyme_dat),
                },
                "interpro_entry_list": {
                    "path": str(args.interpro_entry_list),
                    "sha256": sha256_file(args.interpro_entry_list),
                },
            },
        },
        "corpus_scan": {
            "n_entries_scanned": args.scanned,
            "n_eligible": args.eligible,
            "sampling_fraction": args.pool_size / args.eligible if args.eligible else 0.0,
            "rejections": args.rejections,
            "draw": (
                "seeded reservoir (Algorithm R) over every eligible entry, one "
                "pass; Appendix B rule 1"
            ),
            "scan_is_corpus_prefix": bool(args.max_entries_scanned),
            "go_annotations_seen": args.go_annotations_seen,
            "go_annotations_kept_under_policy": args.go_annotations_kept,
            "go_ids_remapped_through_alt_id": args.go_remapped,
            "n_go_ids_unknown_to_this_ontology": len(args.go_unresolved),
            "go_ids_unknown_to_this_ontology": sorted(args.go_unresolved)[:20],
            "n_records_excluded_for_quoting_own_sequence": len(args.verbatim_examples),
            "accessions_excluded_for_quoting_own_sequence": args.verbatim_examples[:20],
        },
        "pool": {
            "n_records": len(pool),
            "n_distinct_sequences": len(set(sequences)),
            "length_min": int(min(len(sequence) for sequence in sequences)),
            "length_median": float(np.median([len(s) for s in sequences])),
            "length_max": int(max(len(sequence) for sequence in sequences)),
            "family_label_coverage_over_eligible": coverage,
            "family_source_used": args.family_source,
            "family_source_rationale": (
                "cath_superfamily groups remote homologues Pfam separates and is "
                "the stricter split; both coverages are reported above and the "
                "declared source is the one enforced against the coverage floor"
            ),
            "near_duplicate_grouping": grouping,
        },
        "splits": {
            "counts": {
                name: int((splits == name).sum()) for name in sd.SPLIT_NAMES
            },
            "grouping_relations": {
                "family_holdout": (
                    "connected components of 'shares a curated "
                    f"{args.family_source} family OR is a near-duplicate'. The "
                    "near-duplicate relation is carried into the family grouping "
                    "as a label of its own, so the holdout boundary is both "
                    "family-disjoint and near-duplicate-clean by construction "
                    "rather than by inspection. family_group names this component"
                ),
                "fit_eval": (
                    "connected components of the near-duplicate relation alone "
                    "(src.transfer.near_duplicates), which is L30's remedy. "
                    "Curated families are free to cross this boundary; the "
                    "unseen-family property is the holdout's and not this split's"
                ),
            },
            "family_holdout": holdout_split.summary,
            "fit_eval_group_split": group_split,
            "certificates": certificates,
            "family_boundary_leakage": boundary_leakage(
                holdout_split,
                {item["accession"]: item["sequence"] for item in pool},
                seed=args.seed + 3,
            ),
            "fit_eval_boundary_containment": boundary_containment(
                [sequences[int(index)] for index in kept_index],
                fit_mask_subset,
                unit="residues",
            ),
        },
        "homology_leakage": {
            "aligner": aligner,
            "readings": readings,
            "raw_output": {
                **raw,
                "note": (
                    "the alignment itself, gzipped, keyed by the record index in "
                    "pool.fasta, so every row of the readings above can be "
                    "recomputed without re-running the aligner"
                ),
            },
        },
        "description_leakage": description_leakage,
        "concepts": admission.record(),
        "limitations": dict(LIMITATIONS),
    }
    if args.max_entries_scanned:
        payload["limitations"]["draw_is_a_corpus_prefix"] = (
            f"--max-entries-scanned {args.max_entries_scanned} stopped the scan "
            "after a head-of-file prefix of a name-ordered corpus. Appendix B rule "
            "1: this draw is not a sample and no number in this artefact may be "
            "cited as a property of Swiss-Prot"
        )
    destination = args.out / "cohort.json"
    write_json(destination, payload)
    print(f"[verdict] {payload['verdict']}")
    print(f"[done] wrote {destination} and {args.out / 'records.jsonl'}")


def stop_34(
    args: argparse.Namespace,
    *,
    admission: sd.ConceptAdmission,
    readings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """EXP-R2-213's STOP-34 conditions, evaluated on what was measured.

    A stop is a measurement outcome and not a failure: the artefact is written,
    it says which condition fired, and the campaign ends here. The three
    conditions are separate because they are statements about different things --
    two about the cohort's concept panel and one, per the pre-registration's own
    branch table, about the evaluation interface.

    Nothing here may be relaxed by an invocation: :func:`resolve` has already
    refused a request that weakens any of the three, so the values read off
    ``args`` are the pre-registered ones or tighter.
    """

    stops: list[dict[str, Any]] = []
    for split, floor in (
        ("eval", args.stop_min_concepts_eval),
        ("family_holdout", args.stop_min_concepts_family_holdout),
    ):
        admissible = admission.admissible_per_split[split]
        if len(admissible) < floor:
            stops.append(
                {
                    "condition": f"too_few_admissible_concepts_in_{split}",
                    "reading": (
                        f"{len(admissible)} admissible against a floor of {floor}"
                    ),
                    "statement_about": "the cohort, not the model",
                    "admissible": list(admissible),
                }
            )
    group = next(
        reading for reading in readings if reading["split"] == "near_duplicate_group"
    )
    straddling = {
        boundary: count
        for boundary, count in group["straddling_pairs"].items()
        if float(boundary) >= args.straddling_refusal_boundary and count
    }
    if straddling:
        stops.append(
            {
                "condition": "group_split_leaves_straddling_pairs",
                "reading": (
                    f"{straddling} straddling pairs at or above "
                    f"{args.straddling_refusal_boundary}% identity under the "
                    "near-duplicate-group split"
                ),
                "statement_about": "the evaluation interface (L30)",
            }
        )
    return stops


def verify_splits(
    *,
    splits: np.ndarray,
    groups: np.ndarray,
    pool: list[dict[str, Any]],
    assignment_groups: np.ndarray,
) -> dict[str, Any]:
    """Re-derive the three disjointness properties from the returned assignment.

    Appendix B rule 24 and rule 32: a property the construction is *supposed* to
    have is checked on the object that was built, and a fixture or a stage that
    states it in prose has established nothing. All three are cheap and all three
    have failed somewhere in this programme.
    """

    holdout = splits == "family_holdout"
    sides = {
        int(group): set(splits[groups == group]) for group in np.unique(groups)
    }
    dup_across_fit_eval = [
        group for group, seen in sides.items() if {"fit", "eval"} <= seen
    ]
    dup_across_holdout = [
        group
        for group, seen in sides.items()
        if "family_holdout" in seen and len(seen) > 1
    ]
    holdout_families = {
        family
        for item, keep in zip(pool, holdout, strict=True)
        if keep
        for family in item["families"]
    }
    kept_families = {
        family
        for item, keep in zip(pool, ~holdout, strict=True)
        if keep
        for family in item["families"]
    }
    shared_families = sorted(holdout_families & kept_families)
    shared_family_groups = np.intersect1d(
        np.unique(assignment_groups[holdout]), np.unique(assignment_groups[~holdout])
    )
    for name, offenders in (
        ("near-duplicate groups spanning fit and eval", dup_across_fit_eval),
        ("near-duplicate groups spanning the family holdout", dup_across_holdout),
        ("curated families on both sides of the family holdout", shared_families),
        ("family groups on both sides of the holdout", list(shared_family_groups)),
    ):
        if offenders:
            raise RuntimeError(
                f"{len(offenders)} {name}: {offenders[:5]}. The split is not the "
                "one this stage claims to have taken"
            )
    return {
        "verdict": "SPLITS_VERIFIED",
        "checked": (
            "recomputed from the returned split labels rather than trusted from "
            "the construction: no near-duplicate group spans two splits, no "
            "curated family appears on both sides of the family holdout, and no "
            "family group does either"
        ),
        "n_near_duplicate_groups": int(np.unique(groups).size),
        "n_family_groups": int(np.unique(assignment_groups).size),
        "n_distinct_curated_families": len(holdout_families | kept_families),
        "n_curated_families_in_holdout": len(holdout_families),
        # No count of straddling groups is published here: the raises above cover
        # every way a group can span two splits, so such a count could only ever
        # read zero, and a number that cannot vary is not evidence.
    }


def measure_description_leakage(
    records: list[sd.SequenceDescriptionRecord],
    labels: dict[str, list[int | None]],
    concept_forms: dict[str, tuple[str, ...]],
    concepts: tuple[str, ...],
) -> dict[str, Any]:
    """How often a description named the concept, before masking and after.

    C34-2's leak measurement: the *difference* between a result on the raw and on
    the masked description is what says whether an alignment was reading the
    concept name, and the raw rate per concept is what sizes it. The masked
    column is the same statistic on the deciding variant and reads zero because
    C34-1's refusal in the caller has already established that -- it is reported
    per concept so a reader can see which concepts the mask had to work on rather
    than taking one aggregate on trust.

    Computed over every DECLARED concept, admitted or not. A rejected concept's
    leak rate costs nothing and is the number that says whether it was rejected
    for thinness or would also have been unusable.
    """

    per_concept: dict[str, Any] = {}
    for concept in concepts:
        forms = concept_forms[concept]
        cell = {
            "n_positive": 0,
            "n_negative": 0,
            "positive_raw_hits": 0,
            "positive_masked_hits": 0,
            "negative_raw_hits": 0,
        }
        for record, label in zip(records, labels[concept], strict=True):
            if label is None:
                continue
            _, _, raw_spans = sd.mask_description(record.description_raw, forms)
            if label:
                cell["n_positive"] += 1
                cell["positive_raw_hits"] += int(raw_spans > 0)
                _, _, masked_spans = sd.mask_description(
                    record.description_masked, forms
                )
                cell["positive_masked_hits"] += int(masked_spans > 0)
            else:
                cell["n_negative"] += 1
                cell["negative_raw_hits"] += int(raw_spans > 0)
        cell["positive_raw_rate"] = (
            cell["positive_raw_hits"] / cell["n_positive"] if cell["n_positive"] else 0.0
        )
        cell["positive_masked_rate"] = (
            cell["positive_masked_hits"] / cell["n_positive"]
            if cell["n_positive"]
            else 0.0
        )
        cell["negative_raw_rate"] = (
            cell["negative_raw_hits"] / cell["n_negative"] if cell["n_negative"] else 0.0
        )
        cell["n_surface_forms"] = len(forms)
        per_concept[concept] = cell
    # The refusal for a surviving surface form lives in the caller, on the whole
    # per-record term set. A second raise here would be a guard that cannot fire,
    # which this programme has already removed once as unfalsifiable.

    lengths = [len(record.description_raw) for record in records]
    return {
        "per_concept": per_concept,
        "cohort": {
            "n_records": len(records),
            "n_records_with_a_masked_term": sum(
                1 for record in records if record.masked_terms
            ),
            "n_masked_terms_total": sum(len(record.masked_terms) for record in records),
            "median_masked_terms_per_record": float(
                np.median([len(record.masked_terms) for record in records])
            ),
            "median_description_chars": float(np.median(lengths)),
            "mean_chars_removed_fraction": float(
                np.mean(
                    [
                        1.0
                        - len(record.description_masked.replace(sd.MASK_PLACEHOLDER, ""))
                        / max(1, len(record.description_raw))
                        for record in records
                    ]
                )
            ),
        },
        "statistic": (
            "a description 'names' a concept when any surface form of the concept "
            "-- its GO id, name or EXACT/NARROW synonym, or its EC class name -- "
            "occurs in it under case-insensitive, inflection-aware, "
            "non-alphanumeric-bounded matching. The masked column must be zero by "
            "construction and is measured rather than assumed"
        ),
    }


if __name__ == "__main__":
    main()
