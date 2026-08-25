#!/usr/bin/env python3
"""Does a protein decoder know more about function than a lookup of its corpus?

**The gap this stage closes.** EXP-R2-134 measured MODEL - FREE over all 217
ProteinGym substitution assays: ProGen3-112M beats a BLOSUM62 lookup by +0.0647
Spearman [+0.0386, +0.0909]. That is the entirety of this programme's evidence
that any protein decoder knows anything about function, and it cannot separate
knowledge that was *acquired* from knowledge that was *retrieved*, because
BLOSUM62 is free of the corpus and a model that had merely stored its training
data would beat it just as convincingly. This stage measures the second
subtraction, MODEL - LOOKUP, where LOOKUP is a site-independent position-specific
profile built by aligning each assay's wild type against the arm's own declared
pretraining corpus. It is measured against wet-lab phenotype rather than a
predicted structure or a sequence-inferred label, which is the property the D1.c
survey records the objective's second half as lacking an instrument for.

**Nothing in the LOOKUP channel is fitted and no DMS label touches it.** It does
not route through a trained probe -- audit section 7 rejects probe-derived
directions on arrival -- and its one constant, the pseudocount weight, is
declared in ``profiles.PSEUDOCOUNT_ALPHA`` and swept afterwards.

**Pre-registered, before any number exists** (``profiles.equivalence_verdict``):

``acquired``            the cluster-bootstrap interval on MODEL - LOOKUP lies
                        wholly above zero.
``retrieval_bounded``   the interval lies wholly inside half of the arm's own
                        measured MODEL - BLOSUM62 advantage. This is the null
                        that establishes something: at least half of what the
                        model has over a free baseline is also in the corpus
                        lookup.
``indeterminate``       neither; reported with the cluster count that could not
                        resolve it.

Four gates can stop a reading before it is made:

``anchor``              every wild type that is byte-identical to a corpus record
                        must read as a near-duplicate of itself. This is
                        EXP-R2-061 made executable -- there, DIAMOND's default
                        repeat masking truncated exactly those alignments and
                        every error ran in the direction that defeated the
                        hypothesis under test.
``positive_control``    LOOKUP must clear the free baseline it is meant to
                        supersede. Below it, ``Delta_lookup`` is large because the
                        instrument is weak and says nothing about the model
                        (Appendix B rule 2).
``mismatched_profile``  a profile taken from a *different* wild type, matched on
                        length and support, must lose to BLOSUM62. If it does
                        not, the channel is reading generic protein composition
                        and the run reports a defect.
``label_shuffle``       every channel against permuted labels, standardised by
                        each assay's own null scale ``1/sqrt(n-1)`` and read
                        against the Bonferroni normal quantile for the number of
                        values taken. Standardised because assay sizes span 63 to
                        1000 variants, so a raw maximum measures the smallest
                        assay's null width rather than any channel's calibration.

**Unit of analysis.** 217 assays carry 187 distinct wild types, which are
clustered at 50% identity; differences are averaged within cluster and every
interval resamples clusters under ``statistics.bootstrap_unit_floor``. "Millions
of mutant rows over one wild type are not independent units" is the error L1's
cohort made one level down.

**Stratifier, declared in residues.** Maximum percent identity to the corpus and
log10 Neff of homologues at >=30% identity over >=80% query coverage, each
reported over a sweep of bin edges with the ordering required invariant
(Appendix B rule 17) and beside a threshold-free Kendall tau. A verbatim-presence
split is *not* the axis: 78 of 187 wild types are present byte-identically, so
that cut is 39/61 and answers a cruder question.

Stages are independent and resumable; each writes one artefact the next reads.
``wildtypes``, ``search``, ``profile`` and ``lookup`` are CPU-only; ``score`` is
the only stage that needs a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer import profiles as P  # noqa: E402
from src.transfer.arms import (  # noqa: E402
    PANEL,
    REPO,
    STAGED_SCALE_ARMS,
    UNIREF90_BFD30_INCOMPLETE_SEARCH,
    Cohort,
    arm_spec,
    env_path,
    load_arm,
    load_arm_spec,
    require_input_path,
    tokenize_batch,
)
from src.transfer.fitness import (  # noqa: E402
    available_assays,
    load_assay,
    wildtype_of,
)
from src.transfer.homology import (  # noqa: E402
    ALIGNMENT_FIELDS,
    TRUNCATION_RULES,
    assign_homology,
    build_database,
    parse_hits,
    potential_identity_over_query,
    prepare_diamond,
    run_diamond_blastp,
    truncated_alignment,
    write_query_fasta,
)
from src.transfer.io import sha256_file, write_json  # noqa: E402
from src.transfer.statistics import mean_interval  # noqa: E402

SCHEMA_VERSION = P.SCHEMA_VERSION
STAGES = ("wildtypes", "search", "profile", "lookup", "score", "analyse")
DEFAULT_OUT = REPO / "results/transfer/retrieval_bound"

#: Which corpus each arm's LOOKUP channel is entitled to, and what is actually
#: searchable on this host. Only one arm is exactly identified; the other two are
#: bounded, and both bounds run the same way -- an under-supported profile is a
#: weaker LOOKUP, which *inflates* MODEL - LOOKUP and biases the run toward
#: "acquired". Recorded here so that a PASS on those arms is read as an upper
#: bound rather than as a measurement.
#:
#: The declared corpus comes from ``arms.arm_spec`` so a panel member and a
#: staged scale rung cannot drift from the shared declaration (rule 12).
#: Default ``--arms`` stays the three keys below; large/xlarge are resolved by
#: :func:`corpus_record` when explicitly requested.
ARM_CORPUS: dict[str, dict[str, str]] = {
    "protgpt2": {
        "declared": arm_spec("protgpt2").pretraining_corpus,
        "identification": "exact",
        "note": "ProtGPT2's declared pretraining corpus IS the staged UniRef50, so "
        "MODEL - LOOKUP is identified on this arm. The local snapshot is newer "
        "than the 2021_04 release the model saw, which can only add homologues "
        "the model never had -- a stronger LOOKUP, so the bias runs against "
        "'acquired' here rather than for it",
    },
    "progen2-medium": {
        "declared": arm_spec("progen2-medium").pretraining_corpus,
        "identification": "lower bound on support",
        "note": "UniRef90 and BFD30 are not staged. UniRef50 representatives are "
        "members of both, so the searched corpus is a subset of the declared one: "
        "LOOKUP under-counts this arm's retrievable support, which inflates "
        "MODEL - LOOKUP and biases toward passing",
    },
    "progen3-112m": {
        "declared": "undeclared",
        "identification": "lower bound on support",
        "note": "the released ProGen3-112M card states no training corpus, so no "
        "corpus can be attributed to it. UniRef50 is searched as a proxy because "
        "it is what ProGenMech trains its transcoders on and what this repository "
        "stages; the same direction of bias applies and is stronger, since the "
        "true corpus is certainly larger",
    },
}


def corpus_record(arm: str) -> dict[str, str]:
    """Corpus identification for a default arm or an explicit staged scale rung."""

    if arm in ARM_CORPUS:
        return ARM_CORPUS[arm]
    if arm in STAGED_SCALE_ARMS:
        spec = arm_spec(arm)
        return {
            "declared": spec.pretraining_corpus,
            "identification": "lower bound on support",
            "note": UNIREF90_BFD30_INCOMPLETE_SEARCH,
        }
    raise KeyError(
        f"unknown arm {arm!r}; arms are {sorted(set(ARM_CORPUS) | set(STAGED_SCALE_ARMS))}"
    )


SCOREABLE_ARMS = tuple(sorted(set(ARM_CORPUS) | set(STAGED_SCALE_ARMS)))

#: Per-assay covariates the difficulty control is fitted on. Every one is a
#: property of the assay, computable before either channel is scored, so the
#: control cannot absorb the effect it is meant to adjust for.
DIFFICULTY_COVARIATES = (
    "log10_wildtype_length",
    "log10_variants",
    "multi_substitution_fraction",
    "mean_substitutions",
    "dms_score_sd",
)


def _spearman(prediction: np.ndarray, measured: np.ndarray) -> float:
    value = float(stats.spearmanr(prediction, measured).statistic)
    if not math.isfinite(value):
        raise RuntimeError("a channel produced a non-finite Spearman correlation")
    return value


def _digest(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


# --------------------------------------------------------------- stage: wildtypes


def stage_wildtypes(args: argparse.Namespace) -> dict[str, Any]:
    """Catalogue the benchmark's distinct wild types, and read the corpus once."""

    names = list(args.assays)
    print(f"[wildtypes] cataloguing {len(names)} assays")
    wildtype_of_assay = {name: wildtype_of(name, args.proteingym_dir) for name in names}
    distinct = sorted(set(wildtype_of_assay.values()))
    identifier_of = {sequence: f"q{index:05d}" for index, sequence in enumerate(distinct)}
    lengths = [len(sequence) for sequence in distinct]
    print(
        f"[wildtypes] {len(distinct)} distinct wild types; length min {min(lengths)} "
        f"median {int(np.median(lengths))} max {max(lengths)}"
    )

    print(f"[wildtypes] scanning {args.corpus_fasta} for background and verbatim members")
    scan = P.scan_corpus(require_input_path(args.corpus_fasta, "TRANSFER_UNIREF50_FASTA"), distinct)
    print(
        f"[wildtypes] {scan.records} corpus records; {len(scan.verbatim)} of "
        f"{len(distinct)} wild types are byte-identical corpus records"
    )

    cohort = _wildtype_cohort(distinct)
    fasta = args.out / "wildtypes.faa"
    identifiers = write_query_fasta(cohort, fasta)
    if identifiers != [identifier_of[sequence] for sequence in distinct]:
        raise RuntimeError("the query FASTA identifiers do not follow the catalogue order")

    clusters = _cluster_wildtypes(args, distinct, identifier_of, fasta)
    n_clusters = len(set(clusters.values()))
    print(f"[wildtypes] {n_clusters} families at {P.FAMILY_IDENTITY:.0f}% identity")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": "wildtypes",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "assays": len(names),
        "assay_to_wildtype": {
            name: identifier_of[sequence] for name, sequence in wildtype_of_assay.items()
        },
        "wildtypes": {
            identifier_of[sequence]: {
                "length": len(sequence),
                "cluster": clusters[identifier_of[sequence]],
                "verbatim_in_corpus": sequence in scan.verbatim,
                "assays": sorted(
                    name for name, other in wildtype_of_assay.items() if other == sequence
                ),
            }
            for sequence in distinct
        },
        "length_summary": {
            "min": int(min(lengths)),
            "median": float(np.median(lengths)),
            "max": int(max(lengths)),
            "over_1022_residues": int(sum(length > 1022 for length in lengths)),
            "total_residues": int(sum(lengths)),
        },
        "clusters": {
            "n": n_clusters,
            "identity": P.FAMILY_IDENTITY,
            "coverage": P.FAMILY_COVERAGE,
            "sizes": sorted(
                (int(sum(1 for value in clusters.values() if value == cluster)))
                for cluster in sorted(set(clusters.values()))
            ),
        },
        "corpus": scan.record(),
        "verbatim": sorted(identifier_of[sequence] for sequence in scan.verbatim),
        "query_fasta": str(fasta),
    }
    write_json(args.out / "wildtypes.json", payload)
    return payload


def _wildtype_cohort(sequences: list[str]) -> Cohort:
    """The wild types as a ``Cohort``, so the shared DIAMOND helpers apply.

    ``repeats`` is required by ``homology.assign_homology`` and is one
    ``(first, second, span)`` triple per record. For a whole-sequence query the
    span *is* the whole record, so the triple below makes
    ``best_hit_spans_repeat`` read as "the best hit covers the entire wild type",
    which is exactly the flag this stage wants beside an identity.
    """

    lengths = [len(sequence) for sequence in sequences]
    return Cohort(
        name="proteingym_wildtypes",
        kind="protein",
        records=list(sequences),
        min_symbols=min(lengths),
        max_symbols=max(lengths),
        metadata={"repeats": [(0, 0, length) for length in lengths]},
    )


def _cluster_wildtypes(
    args: argparse.Namespace,
    sequences: list[str],
    identifier_of: dict[str, str],
    fasta: Path,
) -> dict[str, int]:
    """All-against-all over the wild types, then single linkage at 50%/80%."""

    tool = prepare_diamond(args.diamond_tarball, args.diamond_checksum, args.diamond_dir)
    database = build_database(
        tool,
        fasta,
        args.out / "wildtypes.dmnd",
        threads=args.threads,
        tmpdir=args.diamond_tmpdir,
        rebuild=True,
    )
    output = args.out / "wildtype_self_hits.tsv"
    run_diamond_blastp(
        tool,
        database,
        fasta,
        output,
        threads=args.threads,
        sensitivity=args.sensitivity,
        evalue=args.evalue,
        max_target_seqs=len(sequences),
    )
    hits = parse_hits(output)
    return P.cluster_by_identity(
        [identifier_of[sequence] for sequence in sequences],
        {identifier_of[sequence]: len(sequence) for sequence in sequences},
        hits,
    )


# ------------------------------------------------------------------ stage: search


def stage_search(args: argparse.Namespace) -> dict[str, Any]:
    """Align every wild type against the corpus, and check the aligner's answer."""

    catalogue = _read(args.out / "wildtypes.json")
    sequences, identifiers = _catalogue_sequences(args, catalogue)
    cohort = _wildtype_cohort(sequences)
    fasta = args.out / "wildtypes.faa"

    tool = prepare_diamond(args.diamond_tarball, args.diamond_checksum, args.diamond_dir)
    database = build_database(
        tool,
        require_input_path(args.corpus_fasta, "TRANSFER_UNIREF50_FASTA"),
        args.diamond_db,
        threads=args.threads,
        tmpdir=args.diamond_tmpdir,
        rebuild=args.rebuild_db,
    )
    output = args.out / "corpus_hits.tsv"
    print(
        f"[search] {len(sequences)} wild types against {database.sequences} corpus "
        f"records at --max-target-seqs {args.max_target_seqs}"
    )
    command, log_tail = run_diamond_blastp(
        tool,
        database,
        fasta,
        output,
        threads=args.threads,
        sensitivity=args.sensitivity,
        evalue=args.evalue,
        max_target_seqs=args.max_target_seqs,
        fields=ALIGNMENT_FIELDS,
    )
    hits = parse_hits(output, fields=ALIGNMENT_FIELDS)
    print(f"[search] {len(hits)} HSPs")

    # `assign_homology` refuses a truncated alignment. Kept, not re-implemented:
    # it is the guard EXP-R2-061 was repaired with. Under
    # `truncation_rule="stratum_changing"` it stops on a flagged alignment whose
    # repair could move its record up a stratum, which is the harm the guard
    # names -- measured here at 5000 targets per query, where the strict rule
    # flags 11 of 22399 alignments, all of them ordinary calmodulin biology
    # against a query whose own verbatim corpus record was found at 100%
    # identity in the same search. This stage additionally carries the direct
    # anchor the guard is a proxy for: byte-identical corpus membership, read
    # from the corpus rather than inferred from an alignment.
    assignments = assign_homology(
        cohort,
        identifiers,
        hits,
        max_target_seqs=args.max_target_seqs,
        truncation_rule=args.truncation_rule,
    )
    flagged = [
        {
            "query_id": hit.query,
            "subject": hit.subject,
            "identity_over_query": hit.identity_over_query,
            "potential_identity_over_query": potential_identity_over_query(hit),
        }
        for hit in hits
        if truncated_alignment(hit)
    ]
    identity = {a.query_id: a.max_identity_over_query for a in assignments}
    by_identifier = dict(zip(identifiers, sequences))
    anchor = P.verbatim_anchor_check(
        [by_identifier[name] for name in catalogue["verbatim"]],
        by_identifier,
        identity,
    )
    print(f"[anchor] {anchor['anchors']} byte-identical corpus members: {anchor['message']}")
    if not anchor["passes"]:
        raise RuntimeError(anchor["message"])

    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": "search",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "diamond": tool.record(),
        "database": database.record(),
        "command": command,
        "log_tail": log_tail,
        "hits_tsv": str(output),
        "hits_sha256": sha256_file(output),
        "n_hsps": len(hits),
        "max_target_seqs": args.max_target_seqs,
        "assignments": [a.record() for a in assignments],
        "truncation": {
            "rule": args.truncation_rule,
            "flagged_alignments": len(flagged),
            "flagged_queries": sorted({row["query_id"] for row in flagged}),
            "examples": flagged[:5],
            "note": "every alignment `truncated_alignment` flags is published here "
            "whether or not the rule stopped the run, so the strict rule's "
            "false-positive rate on this cohort is readable from the artefact",
        },
        "gates": {"anchor": anchor},
    }
    write_json(args.out / "search.json", payload)
    return payload


def _catalogue_sequences(
    args: argparse.Namespace, catalogue: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """Rebuild the exact wild-type list the catalogue stage wrote, in its order."""

    identifiers = sorted(catalogue["wildtypes"])
    by_identifier: dict[str, str] = {}
    for name, identifier in catalogue["assay_to_wildtype"].items():
        by_identifier.setdefault(identifier, wildtype_of(name, args.proteingym_dir))
    sequences = [by_identifier[identifier] for identifier in identifiers]
    for identifier, sequence in zip(identifiers, sequences):
        if len(sequence) != catalogue["wildtypes"][identifier]["length"]:
            raise RuntimeError(
                f"{identifier}: the assay directory no longer produces the wild type "
                "this catalogue was built from"
            )
    return sequences, identifiers


# ----------------------------------------------------------------- stage: profile


def stage_profile(args: argparse.Namespace) -> dict[str, Any]:
    """Build one weighted column-frequency profile per wild type."""

    catalogue = _read(args.out / "wildtypes.json")
    search = _read(args.out / "search.json")
    sequences, identifiers = _catalogue_sequences(args, catalogue)
    hits = parse_hits(Path(search["hits_tsv"]), fields=ALIGNMENT_FIELDS)
    grouped: dict[str, list] = {identifier: [] for identifier in identifiers}
    for hit in hits:
        if hit.query not in grouped:
            raise ValueError(f"hit for unknown query {hit.query!r}")
        grouped[hit.query].append(hit)

    # Two different caps can censor Neff and the artefact has to distinguish
    # them: DIAMOND's own --max-target-seqs, recorded per query by
    # `assign_homology`, and this stage's profile cap. Reporting only the second
    # would let a query whose hit list DIAMOND truncated read as fully supported.
    diamond_saturated = {
        row["query_id"]: bool(row["hit_list_saturated"]) for row in search["assignments"]
    }

    records: dict[str, Any] = {}
    frequencies: dict[str, np.ndarray] = {}
    for index, (identifier, sequence) in enumerate(zip(identifiers, sequences)):
        profile = P.build_profile(
            sequence,
            identifier,
            grouped[identifier],
            max_sequences=args.profile_max_sequences,
        )
        records[identifier] = {
            **profile.record(),
            "diamond_hit_list_saturated": diamond_saturated[identifier],
        }
        frequencies[identifier] = profile.frequencies.astype(np.float32)
        if (index + 1) % 25 == 0 or index + 1 == len(identifiers):
            print(f"[profile] {index + 1}/{len(identifiers)} wild types")

    np.savez_compressed(args.out / "profiles.npz", **frequencies)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": "profile",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "profile_max_sequences": args.profile_max_sequences,
            "coverage_floor": P.PROFILE_COVERAGE_FLOOR,
            "reweight_identity": P.REWEIGHT_IDENTITY_FLOOR,
            "neff_identity_floor": P.NEFF_IDENTITY_FLOOR,
        },
        "profiles": records,
        "saturated": sorted(k for k, v in records.items() if v["hit_list_saturated"]),
        "diamond_saturated": sorted(
            k for k, v in records.items() if v["diamond_hit_list_saturated"]
        ),
        "no_support": sorted(k for k, v in records.items() if v["n_profile_sequences"] == 0),
    }
    write_json(args.out / "profiles.json", payload)
    print(
        f"[profile] {len(payload['saturated'])} of {len(records)} hit the profile cap, "
        f"{len(payload['diamond_saturated'])} hit DIAMOND's --max-target-seqs "
        f"(Neff is right-censored on those); "
        f"{len(payload['no_support'])} have no qualifying corpus support"
    )
    return payload


# ------------------------------------------------------------------ stage: lookup


def stage_lookup(args: argparse.Namespace) -> dict[str, Any]:
    """Score every assay's variants with LOOKUP, the free baselines and the controls."""

    catalogue = _read(args.out / "wildtypes.json")
    profiles = _read(args.out / "profiles.json")
    corpus = catalogue["corpus"]
    background = np.array([corpus["background"][residue] for residue in P.AA20])
    stored = np.load(args.out / "profiles.npz")
    sequences, identifiers = _catalogue_sequences(args, catalogue)
    wildtype_by_id = dict(zip(identifiers, sequences))

    lengths = {i: catalogue["wildtypes"][i]["length"] for i in identifiers}
    clusters = {i: catalogue["wildtypes"][i]["cluster"] for i in identifiers}
    log10_neff = {i: profiles["profiles"][i]["log10_neff"] for i in identifiers}
    donors = P.mismatched_donors(identifiers, lengths, log10_neff, clusters)

    rows: list[dict[str, Any]] = []
    for index, name in enumerate(args.assays):
        assay = load_assay(
            name,
            n=args.variants,
            seed=args.seed + index,
            directory=args.proteingym_dir,
        )
        identifier = catalogue["assay_to_wildtype"][name]
        if assay.wildtype != wildtype_by_id[identifier]:
            raise RuntimeError(f"{name}: drawn wild type differs from the catalogue's")
        variants = assay.substitutions
        profile = _profile_from_store(stored, identifier, wildtype_by_id[identifier], profiles)

        channels: dict[str, np.ndarray] = {
            "lookup": P.profile_scores(profile, background, variants, alpha=args.alpha),
            "blosum62": assay.blosum,
        }
        channels.update(
            P.free_baselines(variants, background, wildtype_length=len(assay.wildtype))
        )
        donor = donors[identifier]["donor"]
        if donor is not None:
            donor_profile = _profile_from_store(
                stored, donor, wildtype_by_id[donor], profiles
            )
            channels["mismatched_profile"] = P.profile_scores(
                donor_profile, background, variants, alpha=args.alpha, check_wildtype=False
            )
        shuffled = np.random.default_rng(args.seed + 100000 + index).permutation(assay.scores)

        row: dict[str, Any] = {
            "assay": name,
            "wildtype_id": identifier,
            "cluster": clusters[identifier],
            "n_variants": len(assay.sequences),
            "mutant_digest": _digest(assay.mutants),
            "spearman": {
                key: _spearman(value, assay.scores) for key, value in channels.items()
            },
            "shuffled_label_spearman": {
                key: _spearman(value, shuffled) for key, value in channels.items()
            },
            "alpha_sweep": {
                f"{alpha:g}": _spearman(
                    P.profile_scores(profile, background, variants, alpha=alpha), assay.scores
                )
                for alpha in P.ALPHA_SWEEP
            },
            "difficulty": _difficulty_covariates(assay),
        }
        rows.append(row)
        print(
            f"  {name:44s} lookup {row['spearman']['lookup']:+.4f}  "
            f"blosum {row['spearman']['blosum62']:+.4f}  "
            f"neff10 {log10_neff[identifier]:.2f}"
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": "lookup",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "alpha": args.alpha,
            "alpha_sweep": list(P.ALPHA_SWEEP),
            "variants": args.variants,
            "seed": args.seed,
            "variant_draw": "seeded permutation of the eligible rows, never a prefix "
            "(Appendix B rule 1)",
        },
        "donors": donors,
        "assays": rows,
    }
    write_json(args.out / "lookup.json", payload)
    return payload


def _profile_from_store(
    stored: Any, identifier: str, wildtype: str, profiles: dict[str, Any]
) -> P.Profile:
    frequencies = np.asarray(stored[identifier], dtype=np.float64)
    record = profiles["profiles"][identifier]
    return P.Profile(
        query_id=identifier,
        wildtype=wildtype,
        n_hits=record["n_hits"],
        n_sequences=record["n_profile_sequences"],
        saturated=record["hit_list_saturated"],
        frequencies=frequencies,
        column_weight=frequencies.sum(axis=1),
        neff=record["neff"],
        max_identity_over_query=record["max_identity_over_query"],
    )


def _difficulty_covariates(assay: Any) -> dict[str, float]:
    counts = [len(subs) for subs in assay.substitutions]
    return {
        "log10_wildtype_length": math.log10(len(assay.wildtype)),
        "log10_variants": math.log10(len(assay.sequences)),
        "multi_substitution_fraction": float(np.mean([c > 1 for c in counts])),
        "mean_substitutions": float(np.mean(counts)),
        "dms_score_sd": float(np.std(assay.scores, ddof=1)),
    }


# ------------------------------------------------------------------- stage: score


def stage_score(args: argparse.Namespace) -> dict[str, Any]:
    """The one stage that needs a GPU: each arm's own zero-shot fitness score."""

    catalogue = _read(args.out / "wildtypes.json")
    results: dict[str, Any] = {}
    for arm in args.arms:
        if arm == "progen3-112m" or arm in ARM_CORPUS or arm in STAGED_SCALE_ARMS:
            pass
        else:
            raise KeyError(
                f"unknown arm {arm!r}; arms are {sorted(SCOREABLE_ARMS)}"
            )
        print(f"[score] {arm}")
        scorer, context, loader_record = _load_scorer(arm, args)
        rows: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for index, name in enumerate(args.assays):
            assay = load_assay(
                name,
                n=args.variants,
                seed=args.seed + index,
                directory=args.proteingym_dir,
            )
            tokens = scorer.token_lengths(assay.sequences)
            if context is not None and max(tokens) > context:
                skipped.append(
                    {
                        "assay": name,
                        "max_tokens": int(max(tokens)),
                        "context": int(context),
                        "reason": "the rendered variant exceeds this arm's context; "
                        "truncating would score a sequence that may not contain the "
                        "mutated position",
                    }
                )
                print(f"  {name:44s} SKIPPED ({max(tokens)} > {context} tokens)")
                continue
            prediction = scorer.log_likelihood(assay.sequences)
            rows.append(
                {
                    "assay": name,
                    "wildtype_id": catalogue["assay_to_wildtype"][name],
                    "mutant_digest": _digest(assay.mutants),
                    "spearman": _spearman(prediction, assay.scores),
                    "max_tokens": int(max(tokens)),
                }
            )
            print(f"  {name:44s} model {rows[-1]['spearman']:+.4f}")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "stage": "score",
            "arm": arm,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "corpus": corpus_record(arm),
            "loader": loader_record,
            "settings": {
                "variants": args.variants,
                "seed": args.seed,
                "batch_size": args.batch_size,
                "dtype": args.dtype,
                "device": args.device,
                "score": "summed log-likelihood of the rendered variant",
            },
            "assays": rows,
            "skipped": skipped,
        }
        write_json(args.out / f"model_{arm}.json", payload)
        results[arm] = payload
        scorer.release()
    return results


class _ArmScorer:
    """Summed log-likelihood of a variant under one arm, in its own rendering."""

    def __init__(self, arm_name: str, args: argparse.Namespace) -> None:
        import torch

        self.torch = torch
        self.name = arm_name
        self.batch_size = args.batch_size
        spec = arm_spec(arm_name)
        self.arm = (
            load_arm(arm_name, device=args.device, dtype=args.dtype)
            if arm_name in PANEL
            else load_arm_spec(spec, device=args.device, dtype=args.dtype)
        )
        config = self.arm.model.config
        self.context = int(
            getattr(config, "n_positions", None)
            or getattr(config, "max_position_embeddings")
        )

    def _render(self, sequences: list[str]) -> list[str]:
        # Rendering is the panel's decision, not this stage's: Appendix B rule 12
        # and the 1.42 nat/token cost of getting ProtGPT2's FASTA wrapping wrong.
        cohort = Cohort(
            name="variants",
            kind="protein",
            records=list(sequences),
            min_symbols=min(len(s) for s in sequences),
            max_symbols=max(len(s) for s in sequences),
            metadata={},
        )
        return cohort.input_strings(self.arm)

    def token_lengths(self, sequences: list[str]) -> list[int]:
        return [
            len(self.arm.tokenizer(text, return_tensors=None)["input_ids"])
            for text in self._render(sequences)
        ]

    def log_likelihood(self, sequences: list[str]) -> np.ndarray:
        torch = self.torch
        texts = self._render(sequences)
        totals = np.empty(len(texts), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                chunk = texts[start : start + self.batch_size]
                ids, mask = tokenize_batch(self.arm, chunk, self.context)
                ids = ids.to(self.arm.device)
                mask = mask.to(self.arm.device)
                logits = self.arm.model(input_ids=ids, attention_mask=mask).logits
                logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
                targets = ids[:, 1:]
                token = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                keep = (mask[:, 1:] * mask[:, :-1]).bool()
                totals[start : start + len(chunk)] = (
                    (token * keep).sum(1).double().cpu().numpy()
                )
        return totals

    def release(self) -> None:
        del self.arm
        self.torch.cuda.empty_cache()


class _ProGen3Scorer:
    """ProGen3-112M, scored bidirectionally as its published perplexities are."""

    def __init__(self, args: argparse.Namespace) -> None:
        import torch

        from src.transfer.progen3 import load_progen3, self_check

        self.torch = torch
        self.batch_size = args.batch_size
        kwargs: dict[str, Any] = {"device": args.device, "dtype": getattr(torch, args.dtype)}
        if args.progen3_checkpoint is not None:
            kwargs["checkpoint"] = args.progen3_checkpoint
        self.pg = load_progen3(**kwargs)
        self.check = self_check(self.pg)
        self.context = None

    def token_lengths(self, sequences: list[str]) -> list[int]:
        # ProGen3 is residue-tokenised with two terminal tokens and has no
        # position limit this stage can exceed at ProteinGym's lengths; the
        # count is still reported so the artefact carries one.
        return [len(sequence) + 2 for sequence in sequences]

    def log_likelihood(self, sequences: list[str]) -> np.ndarray:
        from src.transfer.progen3 import scored_logits, token_nll

        torch = self.torch
        totals = np.zeros((len(sequences), 2), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(sequences), self.batch_size):
                chunk = sequences[start : start + self.batch_size]
                for index, reverse in enumerate((False, True)):
                    batch = self.pg.batch(chunk, reverse=reverse)
                    logits, targets, mask = scored_logits(self.pg, batch)
                    nll = token_nll(logits, targets)
                    totals[start : start + len(chunk), index] = (
                        -(nll * mask).sum(1).double().cpu().numpy()
                    )
        return totals.mean(axis=1)

    def release(self) -> None:
        del self.pg
        self.torch.cuda.empty_cache()


def _load_scorer(arm: str, args: argparse.Namespace) -> tuple[Any, int | None, dict[str, Any]]:
    if arm == "progen3-112m":
        scorer = _ProGen3Scorer(args)
        return scorer, None, {"self_check": scorer.check}
    scorer = _ArmScorer(arm, args)
    return (
        scorer,
        scorer.context,
        {
            "checkpoint": str(arm_spec(arm).path),
            "context": scorer.context,
            "input_format": arm_spec(arm).input_format,
        },
    )


# ----------------------------------------------------------------- stage: analyse


def stage_analyse(args: argparse.Namespace) -> dict[str, Any]:
    """Every gate, control and interval, over the artefacts the earlier stages wrote."""

    catalogue = _read(args.out / "wildtypes.json")
    profiles = _read(args.out / "profiles.json")
    lookup = _read(args.out / "lookup.json")
    by_assay = {row["assay"]: row for row in lookup["assays"]}

    controls = _channel_controls(lookup, args)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": "analyse",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
            if key != "assays"
        },
        "n_assays": len(lookup["assays"]),
        "n_wildtypes": len(catalogue["wildtypes"]),
        "n_clusters": catalogue["clusters"]["n"],
        "corpus": catalogue["corpus"],
        "profile_summary": {
            "saturated": len(profiles["saturated"]),
            "no_support": len(profiles["no_support"]),
        },
        "gates": controls,
        "arms": {},
    }

    for arm in args.arms:
        path = args.out / f"model_{arm}.json"
        if not path.is_file():
            print(f"[analyse] no model scores for {arm}; skipping")
            continue
        model = _read(path)
        payload["arms"][arm] = _arm_analysis(arm, model, by_assay, catalogue, profiles, args)

    payload["verdicts"] = {
        arm: block["verdict"]["verdict"] for arm, block in payload["arms"].items()
    }
    write_json(args.out / "retrieval_bound.json", payload)
    _report(payload)
    return payload


def _channel_controls(lookup: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """The three gates that do not need a model, plus the anchor already checked."""

    rows = lookup["assays"]
    clusters = [row["cluster"] for row in rows]
    # A channel is scored on the assays that have it, not on the assays every
    # channel has. The longest wild type has no mismatched-profile donor -- there
    # is nothing longer to borrow columns from -- and an intersection over all
    # channels would have let that one assay silently delete the whole control.
    every = sorted(set().union(*(set(row["spearman"]) for row in rows)))
    spearman = {
        key: [row["spearman"][key] for row in rows if key in row["spearman"]]
        for key in every
    }
    shuffled = {
        key: [
            row["shuffled_label_spearman"][key]
            for row in rows
            if key in row["shuffled_label_spearman"]
        ]
        for key in every
    }
    #: The variant count each shuffled correlation was computed on, in the same
    #: order. A Spearman correlation under permuted labels has null scale
    #: 1/sqrt(n-1), so a correlation cannot be read without the n that produced it.
    shuffled_n = {
        key: [
            row["n_variants"]
            for row in rows
            if key in row["shuffled_label_spearman"]
        ]
        for key in every
    }
    coverage = {key: len(values) for key, values in spearman.items()}

    means = {key: float(np.mean(values)) for key, values in spearman.items()}
    free = {
        key: float(np.mean(np.abs(spearman[key])))
        for key in ("position_index", "wt_hydropathy", "hydropathy_change", "background_composition")
    }
    if coverage["lookup"] != len(rows) or coverage["blosum62"] != len(rows):
        raise RuntimeError("the lookup and BLOSUM62 channels must cover every assay")
    lookup_minus_blosum = [
        a - b for a, b in zip(spearman["lookup"], spearman["blosum62"])
    ]
    lookup_minus_composition = [
        a - b for a, b in zip(spearman["lookup"], spearman["background_composition"])
    ]
    positive = {
        "lookup_mean_spearman": means["lookup"],
        "declared_floor": P.FROZEN_BLOSUM62_MEAN_SPEARMAN,
        "floor_source": "EXP-R2-134's frozen BLOSUM62 mean over the same 217 assays. "
        "The floor is this repository's own measurement on this cohort rather than "
        "a published ProteinGym baseline: no reference table is staged on this host "
        "and the workstation has no route to the model hub, so an external number "
        "could not be verified before the run and must not gate one",
        "blosum62_mean_spearman_this_run": means["blosum62"],
        # Only the whole benchmark reproduces a whole-benchmark number. On any
        # subset the comparison is a coincidence of which assays were drawn, so
        # it is withheld rather than reported as agreement.
        "blosum62_reproduces_frozen": (
            bool(abs(means["blosum62"] - P.FROZEN_BLOSUM62_MEAN_SPEARMAN) < 0.02)
            if len(rows) == 217
            else None
        ),
        "assays_scored": len(rows),
        "lookup_minus_blosum62": P.cluster_bootstrap(
            lookup_minus_blosum, clusters, resamples=args.bootstrap, seed=args.seed + 11
        ),
        "lookup_minus_background_composition": P.cluster_bootstrap(
            lookup_minus_composition, clusters, resamples=args.bootstrap, seed=args.seed + 12
        ),
    }
    positive["passes"] = bool(
        positive["lookup_mean_spearman"] >= positive["declared_floor"]
        and (positive["lookup_minus_blosum62"]["interval"] or [-1.0])[0] > 0.0
        and (positive["lookup_minus_background_composition"]["interval"] or [-1.0])[0] > 0.0
    )
    positive["note"] = (
        "PASS means the LOOKUP channel is a working retrieval instrument: it clears "
        "the position-independent substitution matrix it is meant to supersede and "
        "the background-composition limit it degenerates to when a profile supports "
        "no column. FAIL means Delta_lookup is large because the instrument is weak, "
        "and says nothing about the model (Appendix B rule 2)"
    )

    mismatch = {
        "measured": coverage.get("mismatched_profile", 0) > 0,
        "assays_with_a_donor": coverage.get("mismatched_profile", 0),
        "assays_total": len(rows),
        "mean_spearman": means.get("mismatched_profile"),
        "blosum62_mean_spearman": float(
            np.mean(
                [
                    row["spearman"]["blosum62"]
                    for row in rows
                    if "mismatched_profile" in row["spearman"]
                ]
            )
        )
        if coverage.get("mismatched_profile", 0)
        else means["blosum62"],
        "wild_types_without_a_donor": sorted(
            k for k, v in lookup["donors"].items() if v["donor"] is None
        ),
        "length_matched": int(
            sum(1 for v in lookup["donors"].values() if v.get("length_matched"))
        ),
        "neff_matched": int(
            sum(1 for v in lookup["donors"].values() if v.get("neff_matched"))
        ),
        "donors": len(lookup["donors"]),
    }
    mismatch["passes"] = bool(
        mismatch["measured"] and mismatch["mean_spearman"] < mismatch["blosum62_mean_spearman"]
    )
    mismatch["note"] = (
        "A profile from a different wild type, matched on length and support, must "
        "lose to BLOSUM62. If it does not, the channel is reading generic protein "
        "composition rather than this protein's columns and the run reports a defect"
    )

    # A maximum of raw correlations is dominated by the smallest assay, because
    # the null scale of a Spearman correlation is 1/sqrt(n-1) and the assays here
    # span n = 63 to 1000. Read against a constant, that maximum fails a
    # correctly calibrated channel whenever one small assay is present: this
    # cohort's smallest (n=63, null scale 0.127) reaches |rho| 0.247 under
    # permuted labels, which is 1.9 of its own standard deviations and nothing at
    # all. Each value is therefore standardised by the scale of the assay that
    # produced it, and the maximum is read against the two-sided Bonferroni
    # normal quantile for the number of values actually taken -- a reference the
    # data's own sizes determine rather than a constant (Appendix B rule 17).
    z_by_channel = {
        key: [
            abs(value) * math.sqrt(max(n - 1, 1))
            for value, n in zip(shuffled[key], shuffled_n[key])
        ]
        for key in every
    }
    z_values = [value for values in z_by_channel.values() for value in values]
    critical_z = float(stats.norm.ppf(1.0 - 0.05 / (2.0 * len(z_values))))
    shuffle = {
        "max_abs_spearman": float(
            max(abs(value) for values in shuffled.values() for value in values)
        ),
        "max_abs_z": float(max(z_values)),
        "critical_z": critical_z,
        "n_values": len(z_values),
        "familywise_alpha": 0.05,
        "max_abs_z_by_channel": {key: float(max(values)) for key, values in z_by_channel.items()},
        "mean_by_channel": {key: float(np.mean(values)) for key, values in shuffled.items()},
        "note": (
            "standardised because the raw maximum is a statement about the "
            "smallest assay's null width, not about whether any channel reads "
            "signal from permuted labels"
        ),
    }
    shuffle["passes"] = bool(shuffle["max_abs_z"] < critical_z)

    return {
        "anchor": "checked and enforced in the search stage; see search.json",
        "positive_control": positive,
        "mismatched_profile": mismatch,
        "label_shuffle": shuffle,
        "free_baselines_mean_abs_spearman": free,
        "free_baseline_envelope_mean_abs_spearman": float(
            np.mean(
                [
                    max(abs(row["spearman"][key]) for key in free)
                    for row in rows
                ]
            )
        ),
        "free_baseline_envelope_note": (
            "the per-assay maximum over four free baselines. A maximum of noisy "
            "quantities is biased upward, so this is a deliberately conservative "
            "comparator for any channel measured against it, not an estimate of "
            "what any one baseline achieves"
        ),
        "channel_mean_spearman": means,
        "channel_assay_coverage": coverage,
        "alpha_sweep_mean_spearman": {
            key: float(np.mean([row["alpha_sweep"][key] for row in rows]))
            for key in rows[0]["alpha_sweep"]
        },
    }


def _arm_analysis(
    arm: str,
    model: dict[str, Any],
    by_assay: dict[str, Any],
    catalogue: dict[str, Any],
    profiles: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """MODEL - LOOKUP for one arm, with its strata, controls and verdict."""

    rows = []
    for entry in model["assays"]:
        lookup_row = by_assay.get(entry["assay"])
        if lookup_row is None:
            raise RuntimeError(f"{entry['assay']}: scored by {arm} but not by LOOKUP")
        if lookup_row["mutant_digest"] != entry["mutant_digest"]:
            raise RuntimeError(
                f"{entry['assay']}: {arm} scored a different variant draw from the "
                "LOOKUP channel; the two stages must share the seed and the count"
            )
        rows.append((entry, lookup_row))
    if not rows:
        raise RuntimeError(f"{arm}: no assay was scored by both channels")

    clusters = [row["cluster"] for _, row in rows]
    delta_lookup = [entry["spearman"] - row["spearman"]["lookup"] for entry, row in rows]
    delta_blosum = [entry["spearman"] - row["spearman"]["blosum62"] for entry, row in rows]
    lookup_minus_blosum = [
        row["spearman"]["lookup"] - row["spearman"]["blosum62"] for _, row in rows
    ]

    boot_lookup = P.cluster_bootstrap(delta_lookup, clusters, resamples=args.bootstrap, seed=args.seed)
    boot_blosum = P.cluster_bootstrap(
        delta_blosum, clusters, resamples=args.bootstrap, seed=args.seed + 1
    )
    boot_retrieved = P.cluster_bootstrap(
        lookup_minus_blosum, clusters, resamples=args.bootstrap, seed=args.seed + 2
    )

    identity = {
        identifier: profiles["profiles"][identifier]["max_identity_over_query"]
        for identifier in profiles["profiles"]
    }
    log10_neff = {
        identifier: profiles["profiles"][identifier]["log10_neff"]
        for identifier in profiles["profiles"]
    }
    unit_delta, unit_labels = P.cluster_means(delta_lookup, clusters)
    unit_identity, _ = P.cluster_means(
        [identity[row["wildtype_id"]] for _, row in rows], clusters
    )
    unit_neff, _ = P.cluster_means(
        [log10_neff[row["wildtype_id"]] for _, row in rows], clusters
    )

    covariates = np.array(
        [[row["difficulty"][key] for key in DIFFICULTY_COVARIATES] for _, row in rows],
        dtype=np.float64,
    )
    if int(unit_labels.size) < args.folds:
        # Withheld rather than forced: cluster-disjoint folds are the whole point
        # of this control, and there are not enough families to build them. The
        # count is the finding, in the shape `bootstrap_unit_floor` uses.
        difficulty = {
            "withheld_reason": (
                f"{unit_labels.size} families is below the {args.folds} "
                "cluster-disjoint folds this control is defined over"
            ),
            "out_of_fold_r2": None,
        }
        residual = None
        unit_residual = None
    else:
        difficulty = P.out_of_fold_difficulty_residual(
            delta_lookup, covariates, clusters, n_splits=args.folds, seed=args.seed + 3
        )
        residual = difficulty.pop("residual")
        unit_residual, _ = P.cluster_means(residual.tolist(), clusters)

    # Appendix B rule 17 for the one constant the channel carries. The model
    # score is fixed, so sweeping alpha sweeps MODEL - LOOKUP directly; the
    # headline is an ordering and it has to survive the sweep or it is a
    # statement about the pseudocount.
    alpha_sweep = {
        key: P.cluster_bootstrap(
            [entry["spearman"] - row["alpha_sweep"][key] for entry, row in rows],
            clusters,
            resamples=args.bootstrap,
            seed=args.seed + 6,
        )
        for key in sorted(rows[0][1]["alpha_sweep"])
    }
    signs = {int(np.sign(block["point"])) for block in alpha_sweep.values()}

    verdict = P.equivalence_verdict(boot_lookup, boot_blosum)
    return {
        "alpha_sweep_delta_lookup": alpha_sweep,
        "alpha_sweep_sign_invariant": bool(len(signs - {0}) <= 1),
        "corpus": corpus_record(arm),
        "assays_scored": len(rows),
        "assays_skipped": model["skipped"],
        "clusters": int(unit_labels.size),
        "mean_spearman": {
            "model": float(np.mean([entry["spearman"] for entry, _ in rows])),
            "lookup": float(np.mean([row["spearman"]["lookup"] for _, row in rows])),
            "blosum62": float(np.mean([row["spearman"]["blosum62"] for _, row in rows])),
        },
        "delta_lookup": boot_lookup,
        "delta_lookup_t_interval": mean_interval(unit_delta.tolist()),
        "delta_blosum62": boot_blosum,
        "lookup_minus_blosum62": boot_retrieved,
        "retrieval_share": P.share_bootstrap(
            lookup_minus_blosum,
            delta_blosum,
            clusters,
            resamples=args.bootstrap,
            seed=args.seed + 5,
        ),
        "strata": {
            "max_identity": P.bin_sweep(
                unit_identity.tolist(), unit_delta.tolist(), P.IDENTITY_EDGE_SWEEP
            ),
            "log10_neff": P.bin_sweep(
                unit_neff.tolist(), unit_delta.tolist(), P.NEFF_EDGE_SWEEP
            ),
        },
        "kendall": {
            "max_identity": P.kendall_tau(unit_identity.tolist(), unit_delta.tolist()),
            "log10_neff": P.kendall_tau(unit_neff.tolist(), unit_delta.tolist()),
        },
        "difficulty_control": {
            **difficulty,
            "covariates": list(DIFFICULTY_COVARIATES),
            "residual_delta_lookup": (
                None
                if residual is None
                else P.cluster_bootstrap(
                    residual.tolist(), clusters, resamples=args.bootstrap, seed=args.seed + 4
                )
            ),
            "residual_kendall_identity": (
                None
                if unit_residual is None
                else P.kendall_tau(unit_identity.tolist(), unit_residual.tolist())
            ),
            "note": "the difficulty model is fitted on cluster-disjoint training folds "
            "and subtracted out of fold. It is NOT a partial correlation, which is the "
            "shape this repository has retracted twice",
        },
        "verdict": verdict,
    }


def _report(payload: dict[str, Any]) -> None:
    gates = payload["gates"]
    print()
    print(f"[clusters] {payload['n_clusters']} families over {payload['n_assays']} assays")
    positive = gates["positive_control"]
    print(
        f"[gate positive_control] lookup {positive['lookup_mean_spearman']:+.4f} against "
        f"floor {positive['declared_floor']:+.4f}: "
        f"{'PASS' if positive['passes'] else 'FAIL'}"
    )
    mismatch = gates["mismatched_profile"]
    if mismatch["measured"]:
        print(
            f"[gate mismatched_profile] {mismatch['mean_spearman']:+.4f} against BLOSUM62 "
            f"{mismatch['blosum62_mean_spearman']:+.4f}: "
            f"{'PASS' if mismatch['passes'] else 'DEFECT'}"
        )
    print(
        f"[gate label_shuffle] max |z| {gates['label_shuffle']['max_abs_z']:.2f} against "
        f"{gates['label_shuffle']['critical_z']:.2f} over {gates['label_shuffle']['n_values']} "
        f"values (max |rho| {gates['label_shuffle']['max_abs_spearman']:.4f}): "
        f"{'PASS' if gates['label_shuffle']['passes'] else 'FAIL'}"
    )
    for arm, block in payload["arms"].items():
        interval = block["delta_lookup"]["interval"]
        text = "degenerate" if interval is None else f"[{interval[0]:+.4f}, {interval[1]:+.4f}]"
        print(
            f"[{arm}] model {block['mean_spearman']['model']:+.4f} lookup "
            f"{block['mean_spearman']['lookup']:+.4f}  "
            f"MODEL-LOOKUP {block['delta_lookup']['point']:+.4f} {text}  "
            f"-> {block['verdict']['verdict']}"
        )


# ------------------------------------------------------------------------- main


def _read(path: Path) -> dict[str, Any]:
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"{path} does not exist; run the stage that writes it first"
        )
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--assays",
        nargs="+",
        default=None,
        help="ProteinGym substitution assays; the default is every assay staged, "
        "which is the population EXP-R2-134's measurement is over",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=sorted(ARM_CORPUS),
        choices=sorted(SCOREABLE_ARMS),
    )
    parser.add_argument("--variants", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--alpha",
        type=float,
        default=P.PSEUDOCOUNT_ALPHA,
        help="pseudocount weight on the corpus background, declared in "
        "src.transfer.profiles before any run",
    )
    parser.add_argument("--profile-max-sequences", type=int, default=5000)
    parser.add_argument("--proteingym-dir", type=Path, default=None)
    parser.add_argument(
        "--corpus-fasta",
        type=Path,
        default=env_path("TRANSFER_UNIREF50_FASTA", REPO / "data/uniref50/uniref50.fasta"),
    )
    parser.add_argument(
        "--diamond-tarball",
        type=Path,
        default=env_path(
            "TRANSFER_DIAMOND_TARBALL",
            REPO / "external_resources/tools/diamond-linux64-v2.1.24.tar.gz",
        ),
    )
    parser.add_argument(
        "--diamond-checksum",
        type=Path,
        default=env_path(
            "TRANSFER_DIAMOND_CHECKSUM",
            REPO / "external_resources/tools/diamond-linux64-v2.1.24.tar.gz.sha256",
        ),
    )
    parser.add_argument(
        "--diamond-dir",
        type=Path,
        default=env_path("TRANSFER_DIAMOND_DIR", Path("/Data/lzp/tools/diamond-2.1.24")),
    )
    parser.add_argument(
        "--diamond-db",
        type=Path,
        default=env_path(
            "TRANSFER_DIAMOND_DB", Path("/Data/lzp/homology_db/uniref50_full.dmnd")
        ),
    )
    parser.add_argument(
        "--diamond-tmpdir",
        type=Path,
        default=env_path("TRANSFER_DIAMOND_TMPDIR", Path("/Data/lzp/homology_db/tmp")),
    )
    parser.add_argument("--rebuild-db", action="store_true")
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--sensitivity", default="very-sensitive")
    parser.add_argument("--evalue", type=float, default=1e-3)
    parser.add_argument("--max-target-seqs", type=int, default=5000)
    parser.add_argument(
        "--truncation-rule",
        default="stratum_changing",
        choices=list(TRUNCATION_RULES),
        help="which truncated-looking alignments stop the run. 'any' is "
        "homology.assign_homology's own default and is unrunnable at thousands of "
        "targets per query; 'stratum_changing' stops on the ones whose repair "
        "could move a record up a stratum, which is the harm the guard names",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--progen3-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    if args.assays is None:
        args.assays = list(available_assays(args.proteingym_dir))
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[paths] out={args.out}")
    print(f"[paths] corpus={args.corpus_fasta}")
    print(f"[paths] assays={len(args.assays)}")

    for stage in STAGES:
        if stage not in args.stages:
            continue
        {
            "wildtypes": stage_wildtypes,
            "search": stage_search,
            "profile": stage_profile,
            "lookup": stage_lookup,
            "score": stage_score,
            "analyse": stage_analyse,
        }[stage](args)


if __name__ == "__main__":
    main()
