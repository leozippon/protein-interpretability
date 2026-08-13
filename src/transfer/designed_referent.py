"""Phenotype prediction on proteins the corpus does not contain.

Why this module exists
======================

F10 bounds what a protein decoder knows about function *relative to a lookup of
its own corpus*, and it cannot do better than bound it: every natural protein has
homologues in UniRef50, so over 187 ProteinGym wild types the least homologous is
still 55.5% identical to a cluster the model was trained on (EXP-R2-189).
Retrieval could be estimated and subtracted; it could never be excluded.

EXP-R2-190 staged a referent where it is excluded by construction. 132 of the 148
de novo designs in Tsuboyama et al. 2023 return **no** DIAMOND hit against the
60.3 M-cluster staged UniRef50 at the gate F10's homology control runs, against a
within-cohort natural control from the same file, the same assay and matched
length that hits at 328 of 330. This module is the measurement that referent was
staged for: the same zero-shot phenotype estimand F10 uses, run on designs where
retrieval is excluded and on the natural domains beside them where it is not.

What the estimand is, and what it is not
========================================

Per wild type, the Spearman correlation between an arm's summed log-likelihood of
a variant and that variant's measured stability (``ddG_ML``, the column
ProteinGym itself publishes as ``DMS_score`` for its 64 Tsuboyama assays -- the
two agree to 9e-16 on every shared variant). Per-wild-type correlations are
averaged inside a unit and the unit is resampled.

This is a **sequence-likelihood** estimand, so L31 does not bind it: a likelihood
needs no token alignment between the variant and the wild type, and the
multi-residue-BPE selection effect that makes position-level interventions
undefined on part of a ProtGPT2 cohort has no purchase here.

**A homologue-free referent is not an information-free one.** DIAMOND finding no
alignment says nothing about whether a design is built from *fragments* the
corpus is full of, and a decoder does not need to have seen a sequence to score
it above chance. So the baseline that replaces F10's profile LOOKUP -- empty by
construction on a disjoint referent -- has to be fragment-level, and it is built
here from the corrected corpus k-mer background as a proper conditional sequence
model rather than as a heuristic.

Identification is not uniform across the arms
=============================================

ProtGPT2's declared pretraining corpus **is** the staged UniRef50, so on that arm
alone the exclusion is *identified*: absence from the searched snapshot implies
absence from the 17%-smaller 2021_04 release it was trained on. Every ProGen2
rung saw UniRef90 and BFD30, and **BFD30 is not staged and was not searched**, so
a design absent from UniRef50 may still be in the corpus those arms read. That
asymmetry runs in the direction that flatters the model and is recorded in
:data:`ARM_IDENTIFICATION` beside every arm it applies to, so a ProGen2 number
cannot borrow ProtGPT2's certificate.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

from . import profiles as P
from .arms import REPO, env_path
from .fitness import BLOSUM62, parse_mutant
from .io import write_json
from .kmer_background import ALPHABET, KmerBackground

SCHEMA_VERSION = "r2_designed_referent_v1"

# ------------------------------------------------------------------- locations

MEGASCALE_DIR = env_path(
    "TRANSFER_MEGASCALE_DIR", REPO / "data/megascale_tsuboyama2023"
)

#: EXP-R2-190's disjointness certificate. The design/natural label, the design
#: series and the list of designs that hit the corpus all come from here rather
#: than being re-derived, so this stage cannot disagree with the certificate
#: about which sequences the referent contains.
CERTIFICATE_DIR = env_path(
    "TRANSFER_MEGASCALE_CERTIFICATE_DIR", REPO / "results/transfer/megascale_disjointness"
)

#: The corrected, record-local k-mer background (EXP-R2-190 section 4).
KMER_BACKGROUND_DIR = env_path(
    "TRANSFER_KMER_BACKGROUND_DIR", REPO / "data/kmer_background/uniref50"
)

#: F10's own artefact, read for one thing: the corpus residue background that the
#: ``background_composition`` free baseline is defined against. Taking it from
#: there rather than recomputing it makes this stage's composition channel the
#: same channel F10 reports (Appendix B rule 12).
RETRIEVAL_BOUND_DIR = env_path(
    "TRANSFER_RETRIEVAL_BOUND_DIR", REPO / "results/transfer/retrieval_bound"
)

# ------------------------------------------------------- the background gate

#: Totals of the **corrected** record-local pass. The first pass over this corpus
#: treated the newline as an invalid symbol, which silently discarded every
#: window spanning the 60-column FASTA line wrapping -- 3.09% of 3-mers and 4.71%
#: of 4-mers. As a *distribution* the error was small (total-variation 0.00028 at
#: k=3), but one 4-mer's frequency moves by 42%, and a baseline that scores
#: fragments individually reads exactly that. This module refuses to run against
#: the superseded counts.
CORRECTED_KMER_TOTALS: Mapping[int, int] = {3: 17_154_643_378, 4: 17_093_475_529}

#: The superseded totals, named so the refusal can say which artefact it found.
SUPERSEDED_KMER_TOTALS: Mapping[int, int] = {3: 16_640_807_917, 4: 16_324_342_706}

# ------------------------------------------------------------------ the cohort

#: The phenotype column. ProteinGym publishes exactly this column as
#: ``DMS_score`` for its 64 Tsuboyama assays; higher is more stable.
PHENOTYPE_COLUMN = "ddG_ML"

#: Wild types carrying fewer eligible variants than this are not scored. The
#: distribution is bimodal -- a MegaScale wild type carries either none or more
#: than a hundred -- so the constant is reported over
#: :data:`VARIANT_FLOOR_SWEEP` and the cohort is identical at every value in it.
MIN_VARIANTS = 30
VARIANT_FLOOR_SWEEP: tuple[int, ...] = (10, 30, 50, 100)

#: Free baselines: computable from the mutation string alone, before any model
#: and before any corpus. The four after BLOSUM62 are
#: :func:`~.profiles.free_baselines`' own family, imported rather than respelled.
FREE_BASELINES: tuple[str, ...] = (
    "blosum62",
    "position_index",
    "wt_hydropathy",
    "hydropathy_change",
    "background_composition",
)

#: Fragment-level retrieval baselines: what the corpus's own k-mer statistics
#: predict, with no model. These are what stands in for F10's profile LOOKUP,
#: which is empty by construction on a referent with no homologues.
FRAGMENT_BASELINES: tuple[str, ...] = ("fragment_markov_k3", "fragment_markov_k4")

BASELINES: tuple[str, ...] = FREE_BASELINES + FRAGMENT_BASELINES

#: Resamples and floor, both the values every other clustered interval in this
#: line of work is taken at (``profiles.cluster_bootstrap``,
#: ``statistics.MINIMUM_BOOTSTRAP_UNITS``).
BOOTSTRAP_RESAMPLES = 2000

#: What each arm's corpus lets the certificate say. Not a caveat block: the
#: certificate was run against UniRef50 alone, so it *identifies* the exclusion
#: for ProtGPT2 and does not for any other arm, and the direction of the gap is
#: recorded because it is not symmetric.
ARM_IDENTIFICATION: Mapping[str, Mapping[str, str]] = {
    "protgpt2": {
        "identification": "exact",
        "note": (
            "ProtGPT2's declared pretraining corpus is UniRef50, which is the "
            "database the certificate searched. UniRef50 2021_04, the release it "
            "saw, holds 49.9 M clusters against the searched snapshot's 60.3 M, "
            "so absence from the searched snapshot implies absence from the "
            "trained-on one and the bias runs safe. This is the only arm on "
            "which the exclusion is identified rather than bounded."
        ),
    },
    "progen2-small": {
        "identification": "unbounded_in_the_model_favouring_direction",
        "note": (
            "UniRef90 + BFD30. BFD30 is not staged and was not searched, so a "
            "design absent from UniRef50 may still be present in this arm's "
            "corpus. The gap runs in the direction that flatters the model."
        ),
    },
    "progen2-base": {
        "identification": "unbounded_in_the_model_favouring_direction",
        "note": (
            "ProGen2-base's mixture is broader still than the UniRef90 + BFD30 "
            "rungs and none of its non-UniRef50 part was searched. Same "
            "direction, at least as wide."
        ),
    },
    "progen2-medium": {
        "identification": "unbounded_in_the_model_favouring_direction",
        "note": (
            "UniRef90 + BFD30, as progen2-small; BFD30 is not staged and was not "
            "searched."
        ),
    },
}

#: Arms excluded from this estimand, with the reason, so that a narrower panel is
#: a declared decision rather than a default nobody checked (L18).
EXCLUDED_ARMS: Mapping[str, str] = {
    "zymctrl": (
        "its rendering is EC-conditioned and a de novo design carries no EC "
        "number, so the estimand is not defined on this referent for this arm"
    ),
    "progen3-112m": (
        "not a panel member, and its published scoring convention is "
        "bidirectional, which is a different estimand from the summed "
        "left-to-right log-likelihood every arm here is read under"
    ),
}

_SUBSTITUTION_TOKEN = re.compile(r"^[A-Z]\d+[A-Z]$")

_ENCODE = np.full(256, -1, dtype=np.int64)
for _index, _symbol in enumerate(ALPHABET):
    _ENCODE[ord(_symbol)] = _index


# ------------------------------------------------------------ variant grammar


def eligible_substitutions(mut_type: str) -> tuple[tuple[str, int, str], ...] | None:
    """The substitutions of one MegaScale ``mut_type``, or ``None`` if ineligible.

    A filter rather than a parser: ``dataset2`` carries wild-type rows, insertions
    and deletions alongside substitutions, and a token whose two residues are
    equal (``T38T``) is a no-op the file spells as a mutation. The grammar itself
    is :func:`~.fitness.parse_mutant`'s, imported rather than respelled, so a
    variant cannot mean one thing here and another in F10's cohort.

    Verified against the benchmark: on the 64 ProteinGym Tsuboyama assays this
    rule selects a **strict subset** of ProteinGym's own variant set -- zero
    variants selected here are absent from ProteinGym, on all 64 -- and
    reproduces it exactly on 27 of them. The 536 variants ProteinGym has and this
    rule does not carry ``ddG_ML == "-"`` in ``dataset2``.
    """

    tokens = mut_type.split(":")
    if not all(_SUBSTITUTION_TOKEN.match(token) for token in tokens):
        return None
    parsed = parse_mutant(mut_type)
    if any(wild == mutated for wild, _, mutated in parsed):
        return None
    return parsed


def apply_substitutions(
    sequence: str, substitutions: Sequence[tuple[str, int, str]]
) -> str:
    """The variant a mutation string names, checked against the wild type.

    Raises when the wild type does not carry the residue the mutation string
    claims, which is the check that lets a cohort be transmitted as wild types
    plus mutation strings rather than as half a million sequences.
    """

    out = list(sequence)
    for wild, position, mutated in substitutions:
        if position < 1 or position > len(out):
            raise ValueError(
                f"position {position} is outside a wild type of {len(out)} residues"
            )
        if out[position - 1] != wild:
            raise ValueError(
                f"the wild type carries {out[position - 1]!r} at position "
                f"{position}, but the mutation string says {wild!r}"
            )
        out[position - 1] = mutated
    return "".join(out)


# ------------------------------------------------------------------ the cohort


@dataclass(frozen=True)
class WildType:
    """One MegaScale wild type and every eligible variant measured on it."""

    name: str
    kind: str
    series: str
    cluster: str
    zero_hit: bool
    series_zero_hit: bool
    sequence: str
    mutants: tuple[str, ...]
    phenotype: np.ndarray
    replicates: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.kind not in ("design", "natural"):
            raise ValueError(f"{self.name}: unknown kind {self.kind!r}")
        if len(self.mutants) != self.phenotype.size:
            raise ValueError(f"{self.name}: mutants and phenotype disagree in length")
        if len(self.mutants) != len(self.replicates):
            raise ValueError(f"{self.name}: mutants and replicates disagree in length")
        if not np.isfinite(self.phenotype).all():
            raise ValueError(f"{self.name}: a non-finite phenotype reached the cohort")

    @property
    def unit(self) -> str:
        """The resampling unit this wild type belongs to.

        A design's unit is its **design series** -- topology x design round,
        run-family x run, or hallucination round -- which is the unit EXP-R2-190
        enumerated and the one this experiment is pre-registered on. A natural
        domain's unit is the dataset's own ``WT_cluster``, which is the analogue
        of F10's 50%-identity family and is not a threshold invented here.
        """

        return f"design:{self.series}" if self.kind == "design" else f"natural:{self.cluster}"

    @property
    def substitutions(self) -> list[tuple[tuple[str, int, str], ...]]:
        return [parse_mutant(mutant) for mutant in self.mutants]

    def sequences(self) -> list[str]:
        return [
            apply_substitutions(self.sequence, entry) for entry in self.substitutions
        ]

    def record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "series": self.series,
            "cluster": self.cluster,
            "zero_hit": self.zero_hit,
            "series_zero_hit": self.series_zero_hit,
            "sequence": self.sequence,
            "mutants": list(self.mutants),
            "phenotype": [float(value) for value in self.phenotype],
            "replicates": list(self.replicates),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "WildType":
        return cls(
            name=str(record["name"]),
            kind=str(record["kind"]),
            series=str(record["series"]),
            cluster=str(record["cluster"]),
            zero_hit=bool(record["zero_hit"]),
            series_zero_hit=bool(record["series_zero_hit"]),
            sequence=str(record["sequence"]),
            mutants=tuple(str(value) for value in record["mutants"]),
            phenotype=np.asarray(record["phenotype"], dtype=np.float64),
            replicates=tuple(int(value) for value in record["replicates"]),
        )


@dataclass(frozen=True)
class Referent:
    """The frozen cohort: every scored wild type, and how it was assembled."""

    wildtypes: tuple[WildType, ...]
    provenance: dict[str, Any]

    def side(self, kind: str, *, zero_hit_only: bool = True) -> tuple[WildType, ...]:
        """The design side or the natural control.

        ``zero_hit_only`` is what excludes the 16 designs that hit the corpus.
        It has no meaning on the natural side, where hitting the corpus is the
        expected state and the control's whole point.
        """

        if kind == "design":
            return tuple(
                wt
                for wt in self.wildtypes
                if wt.kind == "design" and (wt.zero_hit or not zero_hit_only)
            )
        return tuple(wt for wt in self.wildtypes if wt.kind == kind)

    def record(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "provenance": dict(self.provenance),
            "wildtypes": [wt.record() for wt in self.wildtypes],
        }


def build_referent(
    *,
    megascale_dir: Path | None = None,
    certificate_dir: Path | None = None,
    min_variants: int = MIN_VARIANTS,
) -> Referent:
    """Assemble the referent from the staged parquet and EXP-R2-190's certificate.

    A **census**, not a draw: every wild type the certificate searched and every
    eligible variant of it. There is nothing to sample and therefore no seeded
    permutation and no skip-offset sensitivity to report (Appendix B rule 1's
    hazard is absent rather than answered).

    Duplicate measurements of one variant -- 4,071 of 531,275 rows, spread over
    53 wild types -- are averaged rather than dropped or double-weighted, and the
    number averaged into each point is carried into the artefact.
    """

    import pyarrow.parquet as pq

    megascale = Path(megascale_dir) if megascale_dir is not None else MEGASCALE_DIR
    certificates = Path(certificate_dir) if certificate_dir is not None else CERTIFICATE_DIR
    shards = sorted((megascale / "dataset2").rglob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no dataset2 parquet shards under {megascale}")

    certificate = json.loads((certificates / "certificate.json").read_text(encoding="utf-8"))
    index = json.loads((certificates / "query_index.json").read_text(encoding="utf-8"))
    if certificate.get("schema_version") != "megascale_disjointness_certificate_v1":
        raise ValueError(
            f"{certificates} carries schema {certificate.get('schema_version')!r}, "
            "not megascale_disjointness_certificate_v1"
        )
    hit_designs = {entry["WT_name"] for entry in certificate["design_hits_at_gate"]}
    series_zero_hit = {
        name: bool(entry["zero_hit"]) for name, entry in certificate["design_series"].items()
    }
    catalogue = {entry["WT_name"]: entry for entry in index}

    measurements: dict[str, dict[str, list[float]]] = {}
    clusters: dict[str, str] = {}
    rows_read = 0
    for shard in shards:
        table = pq.read_table(
            shard, columns=["mut_type", "WT_name", "WT_cluster", "ddG_ML", "aa_seq"]
        ).to_pydict()
        rows_read += len(table["WT_name"])
        for mut_type, name, cluster, value, aa_seq in zip(
            table["mut_type"],
            table["WT_name"],
            table["WT_cluster"],
            table["ddG_ML"],
            table["aa_seq"],
        ):
            entry = catalogue.get(name)
            if entry is None:
                continue
            clusters.setdefault(name, str(cluster))
            if value is None or value == "-":
                continue
            substitutions = eligible_substitutions(mut_type)
            if substitutions is None:
                continue
            # The file's own variant sequence is the check that a cohort
            # transmitted as wild type + mutation string reconstructs to the
            # sequence that was actually measured.
            if apply_substitutions(entry["sequence"], substitutions) != aa_seq:
                raise ValueError(
                    f"{name}/{mut_type}: the reconstructed variant differs from "
                    "the sequence the file records"
                )
            measurements.setdefault(name, {}).setdefault(mut_type, []).append(float(value))

    wildtypes: list[WildType] = []
    below_floor: list[dict[str, Any]] = []
    for name in sorted(catalogue):
        entry = catalogue[name]
        variants = measurements.get(name, {})
        if len(variants) < min_variants:
            below_floor.append({"name": name, "kind": entry["kind"], "n": len(variants)})
            continue
        mutants = tuple(sorted(variants))
        design = entry["kind"] == "design"
        series = f"{entry['group']}/{entry['series']}" if design else "-"
        wildtypes.append(
            WildType(
                name=name,
                kind=entry["kind"],
                series=series,
                cluster=clusters[name],
                zero_hit=(name not in hit_designs) if design else False,
                series_zero_hit=series_zero_hit[series] if design else False,
                sequence=entry["sequence"],
                mutants=mutants,
                phenotype=np.array(
                    [float(np.mean(variants[mutant])) for mutant in mutants],
                    dtype=np.float64,
                ),
                replicates=tuple(len(variants[mutant]) for mutant in mutants),
            )
        )

    provenance = {
        "megascale_dir": str(megascale),
        "certificate_dir": str(certificates),
        "certificate_gate_evalue": certificate["gate"]["evalue"],
        "phenotype": PHENOTYPE_COLUMN,
        "phenotype_orientation": "higher ddG_ML is more stable",
        "sampling": {
            "mode": "census",
            "note": (
                "every wild type the certificate searched and every eligible "
                "variant of it; nothing is sampled, so there is no draw to seed"
            ),
        },
        "min_variants": int(min_variants),
        "rows_read": rows_read,
        "duplicate_measurements_averaged": sum(
            sum(count - 1 for count in wt.replicates) for wt in wildtypes
        ),
        "counts": cohort_counts(wildtypes),
        "below_build_floor": below_floor,
    }
    return Referent(wildtypes=tuple(wildtypes), provenance=provenance)


def cohort_counts(
    wildtypes: Sequence[WildType], *, min_variants: int = 0
) -> dict[str, int]:
    """What the cohort contains, at one variant floor.

    One declaration, so the headline cohort and the floor sweep cannot count
    themselves differently.
    """

    kept = [wt for wt in wildtypes if len(wt.mutants) >= min_variants]
    designs = [wt for wt in kept if wt.kind == "design"]
    zero_hit = [wt for wt in designs if wt.zero_hit]
    naturals = [wt for wt in kept if wt.kind == "natural"]
    return {
        "designs_scored": len(designs),
        "designs_zero_hit_scored": len(zero_hit),
        "designs_hit_excluded_from_a": len(designs) - len(zero_hit),
        "design_series_scored": len({wt.series for wt in zero_hit}),
        "design_series_entirely_zero_hit_scored": len(
            {wt.series for wt in zero_hit if wt.series_zero_hit}
        ),
        "naturals_scored": len(naturals),
        "natural_clusters_scored": len({wt.cluster for wt in naturals}),
        "design_variants": sum(len(wt.mutants) for wt in zero_hit),
        "natural_variants": sum(len(wt.mutants) for wt in naturals),
    }


def save_referent(referent: Referent, path: Path) -> None:
    write_json(Path(path), referent.record())


def load_referent(path: Path) -> Referent:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path} carries schema {payload.get('schema_version')!r}, not {SCHEMA_VERSION}"
        )
    return Referent(
        wildtypes=tuple(WildType.from_record(record) for record in payload["wildtypes"]),
        provenance=dict(payload["provenance"]),
    )


# ---------------------------------------------------------------- baselines


def require_corrected_background(background: KmerBackground) -> dict[str, Any]:
    """Refuse anything but the corrected, record-local k-mer count.

    Checked by total rather than by directory name, because the superseded counts
    were retained on disk beside the corrected ones and a path is not evidence of
    what a file holds.
    """

    totals = {k: int(background.counts[k].sum()) for k in background.ks}
    for k, expected in CORRECTED_KMER_TOTALS.items():
        if k not in totals:
            raise ValueError(f"the background carries no k = {k} counts")
        if totals[k] == SUPERSEDED_KMER_TOTALS[k]:
            raise ValueError(
                f"k = {k} totals {totals[k]}, which is the superseded line-local "
                "count that dropped every window spanning the corpus's 60-column "
                "wrapping; use data/kmer_background/uniref50"
            )
        if totals[k] != expected:
            raise ValueError(
                f"k = {k} totals {totals[k]}, not the corrected {expected}"
            )
        observed, possible = background.coverage(k)
        if observed != possible:
            raise ValueError(
                f"k = {k} observes {observed} of {possible} k-mers; a conditional "
                "model over this background would divide by zero"
            )
    return {
        "source": str(background.source),
        "residues": int(background.residues),
        "records": int(background.records),
        "totals": totals,
        "verified": "corrected record-local counts (EXP-R2-190 section 4)",
    }


def conditional_log_probabilities(background: KmerBackground, k: int) -> np.ndarray:
    """``log P(x_t | x_{t-k+1..t-1})`` for every k-mer, from the corpus counts.

    The (k-1)-mer denominator is the k-mer table's own row sum, so the
    conditional is normalised over exactly the windows the background counted and
    no second pass over the corpus is needed. Every k-mer is observed, so no
    smoothing rule is required and none is invented.
    """

    width = len(ALPHABET)
    table = background.counts[k].astype(np.float64).reshape(width ** (k - 1), width)
    prefix = table.sum(axis=1)
    if (table <= 0).any() or (prefix <= 0).any():
        raise ValueError(f"k = {k} carries an unobserved k-mer; the conditional is undefined")
    return (np.log(table) - np.log(prefix)[:, None]).reshape(-1)


def encode_sequences(sequences: Sequence[str]) -> np.ndarray:
    """Equal-length sequences as an ``(n, length)`` array of alphabet codes."""

    if not sequences:
        raise ValueError("no sequences to encode")
    length = len(sequences[0])
    if any(len(sequence) != length for sequence in sequences):
        raise ValueError("every sequence must have the same length")
    raw = np.frombuffer("".join(sequences).encode("ascii"), dtype=np.uint8)
    encoded = _ENCODE[raw]
    if (encoded < 0).any():
        raise ValueError("a sequence carries a residue outside the canonical alphabet")
    return encoded.reshape(len(sequences), length)


def fragment_log_likelihood(
    sequences: Sequence[str], log_conditional: np.ndarray, k: int
) -> np.ndarray:
    """The corpus k-mer model's log-likelihood of each sequence.

    The first ``k - 1`` residues carry no emission term. Every variant of one wild
    type has the same length and the same convention, so the omission is a
    constant offset within the unit the correlation is taken over.
    """

    encoded = encode_sequences(sequences)
    width = len(ALPHABET)
    n, length = encoded.shape
    if length < k:
        raise ValueError(f"a sequence of {length} residues carries no {k}-mer")
    index = np.zeros((n, length - k + 1), dtype=np.int64)
    for offset in range(k):
        index *= width
        index += encoded[:, offset : length - k + 1 + offset]
    return log_conditional[index].sum(axis=1)


def corpus_residue_background(path: Path | None = None) -> np.ndarray:
    """The corpus residue frequencies F10's composition baseline is defined on."""

    root = Path(path) if path is not None else RETRIEVAL_BOUND_DIR
    payload = json.loads((root / "wildtypes.json").read_text(encoding="utf-8"))
    background = payload["corpus"]["background"]
    vector = np.array([float(background[residue]) for residue in P.AA20], dtype=np.float64)
    if not np.isfinite(vector).all() or (vector <= 0).any():
        raise ValueError(f"{root}/wildtypes.json carries an unusable residue background")
    return vector / vector.sum()


def baseline_scores(
    wildtype: WildType,
    *,
    residue_background: np.ndarray,
    log_conditional: Mapping[int, np.ndarray],
) -> dict[str, np.ndarray]:
    """Every baseline's prediction for every variant of one wild type."""

    substitutions = wildtype.substitutions
    scores: dict[str, np.ndarray] = {
        "blosum62": np.array(
            [
                sum(BLOSUM62[(wild, mutated)] for wild, _, mutated in entry)
                for entry in substitutions
            ],
            dtype=np.float64,
        )
    }
    scores.update(
        P.free_baselines(
            substitutions, residue_background, wildtype_length=len(wildtype.sequence)
        )
    )
    sequences = wildtype.sequences()
    for k, table in sorted(log_conditional.items()):
        scores[f"fragment_markov_k{k}"] = fragment_log_likelihood(sequences, table, k)
    missing = set(BASELINES) - set(scores)
    if missing:
        raise ValueError(f"{wildtype.name}: baselines {sorted(missing)} were not computed")
    return {name: scores[name] for name in BASELINES}


# --------------------------------------------------------------- statistics


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Rank correlation, or ``None`` when one side is constant.

    Returned rather than raised: a channel that cannot be ranked on one wild type
    is a fact about that wild type, and dropping it silently or writing a NaN
    into the artefact are the two ways this goes wrong later.
    """

    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if left.size < 3 or left.size != right.size:
        return None
    if not (np.isfinite(left).all() and np.isfinite(right).all()):
        return None
    # Checked before the call rather than after: scipy warns and returns NaN on a
    # constant input, and a warning is not a value a stage can branch on.
    if left.min() == left.max() or right.min() == right.max():
        return None
    value = float(stats.spearmanr(left, right).statistic)
    return None if not np.isfinite(value) else value


def unit_bootstrap(
    values: Sequence[float], units: Sequence[str], *, resamples: int, seed: int
) -> dict[str, Any]:
    """Percentile interval on the unit-mean average, resampling units.

    ``profiles.cluster_bootstrap`` performs the resample and carries
    ``statistics.bootstrap_unit_floor``'s refusal record; this wrapper only turns
    the unit names into the integer labels it expects, so the floor, the
    replicate count and the degeneracy record all have one declaration.
    """

    order = {name: position for position, name in enumerate(sorted(set(units)))}
    record = P.cluster_bootstrap(
        values,
        [order[name] for name in units],
        resamples=resamples,
        seed=seed,
    )
    record["unit"] = "design series / natural WT_cluster"
    record["unit_names"] = sorted(order)
    return record


def channel_comparison(
    model: Mapping[str, float],
    baseline: Mapping[str, float],
    units: Mapping[str, str],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int,
) -> dict[str, Any]:
    """MODEL - BASELINE over the wild types both channels could be read on."""

    shared = sorted(set(model) & set(baseline))
    differences = [model[name] - baseline[name] for name in shared]
    record = unit_bootstrap(differences, [units[name] for name in shared], resamples=resamples, seed=seed)
    record["n_wildtypes"] = len(shared)
    record["n_wildtypes_dropped"] = len(set(model) | set(baseline)) - len(shared)
    record["beats_baseline"] = bool(
        record["interval"] is not None and record["interval"][0] > 0.0
    )
    return record


def arm_verdict(design: Mapping[str, bool], control: Mapping[str, bool]) -> dict[str, Any]:
    """The pre-registered rule, applied.

    A positive needs the arm to beat **every** free and fragment-level baseline on
    the corpus-disjoint designs *and* the same conjunction to be attainable on the
    natural control run under the identical procedure. A null on the designs with
    the control passing is a clean negative. A null on the designs with the
    control **failing** says nothing about the model: the instrument could not
    clear its own bar where retrieval is available, so the design-side null is an
    instrument bound and is reported as one.
    """

    missing = (set(BASELINES) - set(design)) | (set(BASELINES) - set(control))
    if missing:
        raise ValueError(f"the verdict needs every baseline; missing {sorted(missing)}")
    attainable = all(control[name] for name in BASELINES)
    beats_all = all(design[name] for name in BASELINES)
    if not attainable:
        verdict = "uninterpretable_instrument_bound"
    elif beats_all:
        verdict = "positive"
    else:
        verdict = "negative"
    return {
        "verdict": verdict,
        "attainable_on_control": attainable,
        "beats_every_baseline_on_designs": beats_all,
        "control_failures": sorted(name for name in BASELINES if not control[name]),
        "design_failures": sorted(name for name in BASELINES if not design[name]),
    }
