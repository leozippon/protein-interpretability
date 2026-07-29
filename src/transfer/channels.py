"""Stage-3 semantics: how many bits of explanation are available at all.

A text decoder labels a feature in its own output vocabulary: the explanation
and the thing explained live in the same space, so the loop closes. Protein
decoders have no such closure. Their explanation vocabulary is imported from
annotation databases, and the capacity of that imported channel is a hard cap on
what any labelling method can deliver. Two caps matter and neither is a property
of the interpretability method.

The first is the event-selection ceiling. A feature that fires on ``m`` of ``N``
cohort positions carries at most ``h(m/N)`` nats about any label whatsoever,
because the firing event is a binary variable with that entropy. A historical
gate demanded 0.1 nats from a top-100-of-122,671 design whose ceiling is
0.0066 nats: the design could not have passed under any circumstances, and the
resulting "no semantic content" verdict was about arithmetic, not biology.

The second is within-sequence label entropy. R2's matched null permutes labels
inside one sequence, so a label that is constant across a sequence contributes
exactly zero by construction. Curated family and domain labels are near-constant
over hundreds of residues; text tokens are not. This module measures that
directly and reports the degeneracy explicitly rather than letting it surface as
an unexplained null result.

Structural attributes are derived from AlphaFold CA traces. The secondary
structure assignment is a coordinate-only P-SEA distance approximation, not
DSSP: it needs no hydrogen bonding and therefore no external dependency, and it
is used uniformly across every model so the comparison stays internally
consistent. It is not a substitute for DSSP and must not be reported as one.
"""

from __future__ import annotations

import gzip
import math
from collections import Counter
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .arms import REPO, env_path, require_input_path
from .statistics import mean_interval

PFAM_RESIDUE_TSV = env_path(
    "TRANSFER_PFAM_RESIDUE_TSV", REPO / "data/interpro/pfam_residue.tsv"
)
ALPHAFOLD_ROOT = env_path("TRANSFER_ALPHAFOLD_DIR", REPO / "data/alphafold")
PFAM_TSV_HEADER = ("uniprot", "start", "end", "pfam_id")

#: Label assigned to residues with no Pfam coverage. Excluding them instead
#: would inflate the measured entropy by conditioning on annotation existing.
UNANNOTATED = "__unannotated__"

#: Fixed comparison window. Every within-sequence entropy is measured over the
#: same number of symbols so that the log2(window) sampling ceiling is identical
#: across text tokens, residues, domain labels and structural attributes.
DEFAULT_WINDOW = 300

THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
    "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
    "TRP": "W", "TYR": "Y",
}

LN2 = math.log(2.0)

#: Emitted alongside every secondary-structure figure. The distance-only subset
#: of the P-SEA rule assigns roughly half of all residues to the strand state,
#: well above the ~20-25% DSSP would give, because the strand distance windows
#: also admit extended coil. The three-state label is therefore usable as a
#: coordinate-derived attribute channel - which is all the entropy measurement
#: needs - but its state fractions are not secondary-structure content and must
#: not be quoted as such.
SECONDARY_STRUCTURE_CAVEAT = (
    "ca_trace_psea_distance_criterion_only; no hydrogen bonding and no angle "
    "criterion, so the strand state is over-assigned relative to DSSP; state "
    "fractions are not DSSP secondary-structure content"
)


# --------------------------------------------------------- event-selection cap


def binary_entropy_nats(p: float) -> float:
    """Entropy of a Bernoulli(``p``) variable in nats.

    This is the exact ceiling on ``I(event; label)`` for any label, because the
    firing indicator cannot carry more information than it has entropy.
    """

    if not math.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError("p must be a probability in [0, 1]")
    if p in (0.0, 1.0):
        return 0.0
    return -p * math.log(p) - (1.0 - p) * math.log(1.0 - p)


def required_event_count(target_nats: float, n_positions: int) -> int:
    """Smallest event count whose selection entropy reaches ``target_nats``.

    Binary search is valid because ``h(m/N)`` increases monotonically in ``m``
    over ``m <= N/2``; counts above ``N/2`` are the mirror image and are never
    the smallest solution.
    """

    if not math.isfinite(target_nats) or target_nats <= 0.0:
        raise ValueError("target_nats must be finite and positive")
    if n_positions < 2:
        raise ValueError("n_positions must be at least two")
    ceiling = binary_entropy_nats(0.5)
    if target_nats > ceiling:
        raise ValueError(
            f"target of {target_nats} nats exceeds the maximum selection entropy "
            f"of {ceiling} nats attainable at any event count"
        )
    low, high = 1, n_positions // 2
    while low < high:
        middle = (low + high) // 2
        if binary_entropy_nats(middle / n_positions) < target_nats:
            low = middle + 1
        else:
            high = middle
    return low


def event_selection_ceiling(
    n_positions: int,
    *,
    event_counts: Sequence[int],
    gate_nats: Sequence[float],
    realised_event_count: int,
    realised_gate_nats: float,
) -> dict[str, Any]:
    """Analytic ceiling table for a top-k event design on a fixed cohort."""

    if n_positions < 2:
        raise ValueError("n_positions must be at least two")
    if not event_counts or not gate_nats:
        raise ValueError("event_counts and gate_nats must both be non-empty")
    if not 0 < realised_event_count <= n_positions:
        raise ValueError("realised_event_count must lie inside the cohort")
    if any(not 0 < count <= n_positions for count in event_counts):
        raise ValueError("every event count must lie inside the cohort")
    realised_ceiling = binary_entropy_nats(realised_event_count / n_positions)
    if realised_ceiling <= 0.0:
        # h(1) = 0: an "event" that fires at every position is not an event, and
        # dividing the gate by its ceiling would be a division by zero rather
        # than a ratio of 15.1 that the caller could act on.
        raise ValueError(
            f"the realised event fires at all {n_positions} positions, so its "
            "selection entropy is zero and no gate is attainable against it"
        )
    return {
        "cohort_positions": int(n_positions),
        "realised_event_count": int(realised_event_count),
        "realised_event_prevalence": realised_event_count / n_positions,
        "realised_max_possible_mi_nats": realised_ceiling,
        "realised_gate_nats": float(realised_gate_nats),
        "realised_gate_over_ceiling": float(realised_gate_nats) / realised_ceiling,
        "gate_is_attainable": float(realised_gate_nats) <= realised_ceiling,
        "ceiling_by_event_count_nats": {
            str(count): binary_entropy_nats(count / n_positions) for count in event_counts
        },
        "events_required_for_gate": {
            str(target): required_event_count(target, n_positions) for target in gate_nats
        },
    }


# ------------------------------------------------------------------- entropies


def entropy_bits(counts: Mapping[Hashable, int]) -> float:
    """Plug-in Shannon entropy of a count distribution, in bits."""

    if not counts:
        raise ValueError("cannot take the entropy of an empty distribution")
    values = np.asarray(list(counts.values()), dtype=np.float64)
    if np.any(values < 0):
        raise ValueError("counts must be non-negative")
    total = values.sum()
    if total <= 0:
        raise ValueError("count distribution has zero mass")
    probabilities = values[values > 0] / total
    return float(-(probabilities * np.log2(probabilities)).sum())


def miller_madow_entropy_bits(counts: Mapping[Hashable, int]) -> float:
    """Miller-Madow corrected entropy, in bits.

    Over a 300-symbol window the plug-in estimator is biased low by a
    non-negligible amount for high-cardinality alphabets such as text tokens,
    which is precisely the comparison this module has to get right.
    """

    plugin = entropy_bits(counts)
    total = sum(counts.values())
    observed = sum(1 for value in counts.values() if value > 0)
    return plugin + (observed - 1) / (2.0 * total * LN2)


def label_distribution(counts: Mapping[Hashable, int]) -> dict[str, Any]:
    """Marginal entropy summary for one label vocabulary."""

    return {
        "n_labelled_symbols": int(sum(counts.values())),
        "n_distinct_labels": int(sum(1 for value in counts.values() if value > 0)),
        "entropy_bits": entropy_bits(counts),
        "entropy_miller_madow_bits": miller_madow_entropy_bits(counts),
    }


def within_unit_label_entropy(
    units: Iterable[Sequence[Hashable]],
    *,
    window: int = DEFAULT_WINDOW,
    min_units: int = 30,
) -> dict[str, Any]:
    """Mean Miller-Madow label entropy inside a fixed-length window per unit.

    ``permutation_null_degenerate_fraction`` is the decisive diagnostic. A
    within-unit permutation null resamples label positions inside one unit, so a
    unit whose window carries a single distinct label is invariant under every
    permutation: the null has identically zero power there, and a test built on
    it cannot detect structure however strong the structure is.
    """

    if window < 2:
        raise ValueError("window must span at least two symbols")
    if min_units < 2:
        raise ValueError("min_units must be at least two so an interval exists")

    entropies: list[float] = []
    distinct: list[int] = []
    majority: list[float] = []
    considered = 0
    for labels in units:
        considered += 1
        if len(labels) < window:
            continue
        counts = Counter(labels[:window])
        entropies.append(miller_madow_entropy_bits(counts))
        distinct.append(len(counts))
        majority.append(max(counts.values()) / window)
    if len(entropies) < min_units:
        raise RuntimeError(
            f"only {len(entropies)} of {considered} units reach the {window}-symbol "
            f"window; need at least {min_units}"
        )
    return {
        "window_symbols": int(window),
        "sampling_ceiling_bits": math.log2(window),
        "n_units": len(entropies),
        "n_units_considered": considered,
        "entropy_bits": mean_interval(entropies),
        "mean_distinct_labels_per_window": float(np.mean(distinct)),
        "median_distinct_labels_per_window": float(np.median(distinct)),
        "mean_majority_label_share": float(np.mean(majority)),
        "permutation_null_degenerate_fraction": float(np.mean([c < 2 for c in distinct])),
        "near_degenerate_fraction_majority_over_0p9": float(
            np.mean([share > 0.9 for share in majority])
        ),
    }


# ------------------------------------------------------------ curated labels


def load_pfam_spans(
    path: Path = PFAM_RESIDUE_TSV, accessions: set[str] | None = None
) -> dict[str, list[tuple[int, int, str]]]:
    """Residue-level Pfam spans, optionally restricted to given accessions."""

    spans: dict[str, list[tuple[int, int, str]]] = {}
    require_input_path(Path(path), "TRANSFER_PFAM_RESIDUE_TSV")
    with Path(path).open(encoding="utf-8") as handle:
        header = tuple(next(handle).rstrip("\n").split("\t"))
        if header != PFAM_TSV_HEADER:
            raise ValueError(f"{path}: expected columns {PFAM_TSV_HEADER}, found {header}")
        for number, line in enumerate(handle, 2):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 4:
                raise ValueError(f"{path}:{number}: expected 4 columns, found {len(fields)}")
            accession, start, end, pfam = fields
            if accessions is not None and accession not in accessions:
                continue
            begin, finish = int(start), int(end)
            if begin < 1 or finish < begin:
                raise ValueError(f"{path}:{number}: invalid span {begin}-{finish}")
            spans.setdefault(accession, []).append((begin, finish, pfam))
    if not spans:
        raise RuntimeError(f"{path}: no Pfam spans matched the requested accessions")
    return spans


def pfam_residue_labels(
    length: int, spans: Sequence[tuple[int, int, str]]
) -> tuple[list[str], int]:
    """Per-residue Pfam labels and the number of residues claimed twice.

    Spans are applied in ascending ``(start, end)`` order and a later span wins,
    so the assignment is deterministic. Pfam domains rarely overlap; the overlap
    count is returned so that the rare cases are auditable instead of silent.
    """

    if length < 1:
        raise ValueError("length must be positive")
    labels = [UNANNOTATED] * length
    overlaps = 0
    for start, end, pfam in sorted(spans):
        for position in range(start - 1, min(end, length)):
            if labels[position] != UNANNOTATED:
                overlaps += 1
            labels[position] = pfam
    return labels, overlaps


# ------------------------------------------------------- structural attributes


@dataclass(frozen=True)
class Structure:
    """CA trace and per-residue confidence from one AlphaFold model."""

    accession: str
    sequence: str
    ca: np.ndarray
    plddt: np.ndarray
    n_non_canonical_residues: int

    def __post_init__(self) -> None:
        if self.ca.ndim != 2 or self.ca.shape[1] != 3:
            raise ValueError(f"{self.accession}: CA coordinates must have shape (n, 3)")
        if self.plddt.ndim != 1 or self.plddt.shape[0] != self.ca.shape[0]:
            raise ValueError(f"{self.accession}: pLDDT does not align with the CA trace")
        if len(self.sequence) != self.ca.shape[0]:
            raise ValueError(f"{self.accession}: sequence does not align with the CA trace")
        if self.ca.shape[0] == 0:
            raise ValueError(f"{self.accession}: no CA atoms")
        if not np.isfinite(self.ca).all() or not np.isfinite(self.plddt).all():
            raise ValueError(f"{self.accession}: non-finite coordinates or pLDDT")

    def __len__(self) -> int:
        return len(self.sequence)


def accession_from_alphafold_path(path: Path) -> str:
    parts = Path(path).name.split("-")
    if len(parts) < 3 or parts[0] != "AF":
        raise ValueError(f"{path}: not an AlphaFold model filename")
    return parts[1]


def read_alphafold_model(path: Path) -> Structure:
    """Parse CA coordinates, pLDDT (B-factor column) and the one-letter sequence.

    Residues outside the canonical twenty are counted rather than tolerated
    silently, so callers can exclude such models by an explicit predicate.
    """

    coordinates: list[tuple[float, float, float]] = []
    confidence: list[float] = []
    residues: list[str] = []
    skipped = 0
    previous = None
    with gzip.open(Path(path), "rt") as handle:
        for line in handle:
            if not line.startswith("ATOM") or line[12:16].strip() != "CA":
                continue
            number = int(line[22:26])
            if previous is not None and number <= previous:
                raise ValueError(f"{path}: CA residue numbers are not strictly increasing")
            previous = number
            name = line[17:20].strip()
            if name not in THREE_TO_ONE:
                skipped += 1
                continue
            coordinates.append(
                (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            )
            confidence.append(float(line[60:66]))
            residues.append(THREE_TO_ONE[name])
    if not coordinates:
        raise ValueError(f"{path}: no canonical CA atoms")
    return Structure(
        accession=accession_from_alphafold_path(path),
        sequence="".join(residues),
        ca=np.asarray(coordinates, dtype=np.float64),
        plddt=np.asarray(confidence, dtype=np.float64),
        n_non_canonical_residues=skipped,
    )


def ca_secondary_structure(ca: np.ndarray) -> np.ndarray:
    """Three-state assignment from the CA trace alone (0 helix, 1 strand, 2 coil).

    P-SEA distance criterion. This is a coordinate-only approximation to DSSP
    that ignores hydrogen bonding; it is used because it applies uniformly to
    every AlphaFold model without an external dependency, and its absolute
    fractions must not be quoted as DSSP secondary structure content.
    """

    if ca.ndim != 2 or ca.shape[1] != 3:
        raise ValueError("CA coordinates must have shape (n, 3)")
    n = ca.shape[0]
    assignment = np.full(n, 2, dtype=np.int8)
    if n < 5:
        return assignment
    d2 = np.linalg.norm(ca[2:] - ca[:-2], axis=1)
    d3 = np.linalg.norm(ca[3:] - ca[:-3], axis=1)
    d4 = np.linalg.norm(ca[4:] - ca[:-4], axis=1)
    for start in range(n - 4):
        a, b, c = d2[start], d3[start], d4[start]
        if abs(a - 5.5) < 0.5 and abs(b - 5.3) < 0.5 and abs(c - 6.4) < 0.6:
            assignment[start : start + 5] = 0
        elif abs(a - 6.7) < 0.6 and abs(b - 9.9) < 0.9 and abs(c - 12.4) < 1.1:
            assignment[start : start + 5] = 1
    return assignment


def contact_number_bins(
    ca: np.ndarray, *, radius: float = 10.0, bin_width: int = 4, n_bins: int = 8
) -> np.ndarray:
    """Binned count of other CA atoms within ``radius`` angstrom."""

    if radius <= 0 or bin_width < 1 or n_bins < 2:
        raise ValueError("invalid contact-number binning parameters")
    distance = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
    np.fill_diagonal(distance, np.inf)
    contacts = (distance < radius).sum(axis=1)
    return np.clip(contacts // bin_width, 0, n_bins - 1).astype(np.int8)


def plddt_bins(plddt: np.ndarray, *, bin_width: float = 20.0, n_bins: int = 5) -> np.ndarray:
    """Binned per-residue AlphaFold confidence."""

    if bin_width <= 0 or n_bins < 2:
        raise ValueError("invalid pLDDT binning parameters")
    return np.clip((plddt // bin_width).astype(np.int64), 0, n_bins - 1).astype(np.int8)


def structural_attribute_labels(
    structure: Structure,
    *,
    contact_radius: float = 10.0,
    contact_bin_width: int = 4,
    contact_bins: int = 8,
    plddt_bin_width: float = 20.0,
    plddt_bin_count: int = 5,
) -> list[tuple[int, int, int]]:
    """Joint (secondary structure, contact-number bin, pLDDT bin) per residue."""

    secondary = ca_secondary_structure(structure.ca)
    contacts = contact_number_bins(
        structure.ca,
        radius=contact_radius,
        bin_width=contact_bin_width,
        n_bins=contact_bins,
    )
    confidence = plddt_bins(
        structure.plddt, bin_width=plddt_bin_width, n_bins=plddt_bin_count
    )
    return [
        (int(s), int(c), int(p)) for s, c, p in zip(secondary, contacts, confidence)
    ]


def alphafold_models(root: Path = ALPHAFOLD_ROOT, *, limit: int | None = None) -> list[Path]:
    """AlphaFold PDB models in deterministic filename order.

    Filename order is UniProt-accession order, which front-loads whole-proteome
    dumps of closely related entries, so ``limit`` returns a taxonomically
    clustered prefix rather than a sample. It is kept because the full catalogue
    is what most callers want and because a frozen artefact was produced with it;
    :func:`alphafold_model_sample` is what a *limited* selection should use.
    """

    require_input_path(Path(root), "TRANSFER_ALPHAFOLD_DIR")
    paths = sorted(Path(root).glob("AF-*-model_v*.pdb.gz"))
    if not paths:
        raise RuntimeError(f"no AlphaFold PDB models under {root}")
    return paths if limit is None else paths[:limit]


def alphafold_model_sample(
    root: Path = ALPHAFOLD_ROOT, *, limit: int, seed: int
) -> tuple[list[Path], dict[str, Any]]:
    """``limit`` models drawn under a seeded permutation, with the draw recorded.

    The explanation-channel figures -- text token identity 7.32 bits/symbol
    against a Pfam domain label at 0.74 -- are per-symbol entropies of label
    channels measured over whichever structures were read. Reading the first
    ``limit`` filenames makes that a measurement of one taxonomic neighbourhood,
    which is why the audit records those figures as needing re-derivation under a
    seeded permutation before they are quoted. This is the draw that does it, and
    it returns its own provenance so the artefact can say which one produced the
    number instead of leaving a reader to infer it.
    """

    if limit < 1:
        raise ValueError("limit must be positive")
    catalogue = alphafold_models(root)
    if len(catalogue) < limit:
        raise RuntimeError(
            f"only {len(catalogue)} AlphaFold models under {root} for a draw of {limit}"
        )
    order = np.random.default_rng(seed).permutation(len(catalogue))[:limit]
    selection = [catalogue[int(index)] for index in sorted(int(i) for i in order)]
    return selection, {
        "mode": "seeded_permutation",
        "seed": int(seed),
        "requested": int(limit),
        "catalogue_size": len(catalogue),
        "root": str(root),
    }
