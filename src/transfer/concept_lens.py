"""The concept-aligned lens: a coarsened lens readout with a null (D3.b / R3.1).

What this adds to the lens family, and why the addition is the point
===================================================================

``lenses.residue_class_trajectory`` already reads a protein decoder's lens
distribution through a coarsening -- one hand-declared four-way chemical
partition of the alphabet -- and reports whether the coarse term resolves at
shallower depth than the fine one. It has no null. Without one, "the protein
decoder resolves chemical class before residue identity" is uninterpretable,
because **every** coarsening of a distribution resolves earlier than the
distribution: collapsing twenty residues onto four classes takes the 64-246 band
marginal from 4.1566 bits to about 1.58, so a lens has less than 40% as much to
be right about and will look like it got there sooner.

The contribution here is therefore not a new readout. It is the control
structure that makes a coarsened readout mean something: a pre-declared property
basis, a shuffled-property null, a rank-matched partition null that holds the
coarsened entropy fixed, the arm's own priors, and a family-disjoint report side.
A concept result is quoted as its **excess over the null**, never as its value.

Why this is admissible where the property-conditioned Jacobian is not
====================================================================

§7 of the audit rejects property-conditioned Jacobian subspaces, three times,
on one ground: secondary structure, Pfam, EC and fitness are not functionals of
the next-token distribution, so their directions can only come from a trained
probe, which makes the object the Jacobian of the probe.

``E[phi(next token)]`` **is** a functional of the next-token distribution.
``phi`` is a fixed table indexed by token, so the concept direction

    d/dh E[phi] = J_LN^T W_U^T (diag(p) - p p^T) phi

is closed form from the model's own final norm and unembedding and a constant.
Nothing is fitted, and no variant in this module may be defined by a direction
that is. If one ever needs a probe to say where the concept lives, it is out of
scope rather than in need of a caveat.

The two conventions that are not free choices
=============================================

*Which residue a token is.* Taken from ``lenses.residue_vocabulary``, which keys
a multi-residue BPE token on the residue it **starts** with -- the only
residue-level question a next-token distribution answers on such an arm. That
declaration is imported, not restated (Appendix B rule 12).

*What the lens does with mass outside the alphabet.* It is renormalised away and
the discarded mass is reported per layer as ``abstain_mass``. On ProtGPT2 the
FASTA newline is a large attractor and was the origin of a retracted spectrum
result (§0.1), so a design that let it vanish into a denominator would be
repeating a known failure.

What a text arm is doing here
=============================

Closing the method reading, and nothing else. The same procedure runs on a text
decoder with a different pre-declared token-to-property table -- surface
properties, which are deterministic functions of the token id in exactly the way
hydropathy is deterministic in the residue. If the aperture gain is as large
there, it is a property of coarsening and therefore of the **method** (§5's
organising rule), whatever the protein arms show.

The asymmetry runs one way and is not repairable. A text arm needs no
renormalisation, because every token has a defined length and frequency, while a
protein arm discards non-residue mass. So the text control can *close* the method
reading; it cannot *open* a modality reading, and no cross-modality coefficient
is computed here. D3.b admits cross-modality comparison only on modality-neutral
abstractions, and a biochemical property is not one.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy import stats

from .arms import AA20, Arm
from .lenses import LensHead, ResidualCache, ResidueVocabulary, resolution_depth
from .profiles import KYTE_DOOLITTLE

#: Formal side-chain charge at pH 7. Histidine carries the fractional value its
#: pKa near 6 implies rather than being rounded to zero or one, so the vector is
#: the standard convention and not a choice made here.
CHARGE_PH7: dict[str, float] = {residue: 0.0 for residue in AA20} | {
    "D": -1.0,
    "E": -1.0,
    "K": 1.0,
    "R": 1.0,
    "H": 0.1,
}

#: Side-chain van der Waals volume in cubic angstroms (Creighton/Richards).
SIDE_CHAIN_VOLUME: dict[str, float] = {
    "A": 67.0, "R": 148.0, "N": 96.0, "D": 91.0, "C": 86.0,
    "Q": 114.0, "E": 109.0, "G": 48.0, "H": 118.0, "I": 124.0,
    "L": 124.0, "K": 135.0, "M": 124.0, "F": 135.0, "P": 90.0,
    "S": 73.0, "T": 93.0, "W": 163.0, "Y": 141.0, "V": 105.0,
}

#: The declared basis, frozen before any run. Three properties rather than a
#: menu, because these are the axes the classical amino-acid factor analyses
#: recover -- hydrophobicity, charge and size -- so declaring them is declaring a
#: basis, while adding a fourth on inspection of a result would be fishing.
PROPERTY_BASIS: dict[str, dict[str, float]] = {
    "hydropathy": dict(KYTE_DOOLITTLE),
    "charge": dict(CHARGE_PH7),
    "volume": dict(SIDE_CHAIN_VOLUME),
}

#: Surface properties of a text token, the analogue table. Each is a
#: deterministic function of the token id -- computed from the tokenizer alone,
#: with no corpus and no estimator -- which is the property that makes the text
#: arm the same procedure rather than a different one. ``uppercase_fraction`` is
#: the graded analogue of hydropathy, ``character_length`` of volume, and
#: ``word_initial`` (a leading space in the GPT-2 byte-BPE) of charge, being the
#: near-binary one.
#:
#: A token-frequency property was considered and rejected before the run: it
#: would need a held-out unigram over 50,257 ids, most of which any affordable
#: draw never sees, so the vector would be a floor with a spike and the null
#: would be permuting a constant. That is a property of the estimator, not of
#: text, and it would have made the control weaker exactly where it has to be
#: strong.
TEXT_PROPERTY_NAMES: tuple[str, ...] = (
    "uppercase_fraction",
    "character_length",
    "word_initial",
)


def text_property_table(arm: Arm) -> dict[str, np.ndarray]:
    """Surface properties of every output token, from the tokenizer alone.

    Tokens that decode to the empty string are given the alphabet's mean on each
    property rather than a sentinel, because a sentinel would make the null's
    permutation move a value no real token carries.
    """

    if arm.modality != "text":
        raise ValueError(f"{arm.name}: the surface-property table is the text-arm table")
    tokenizer = arm.tokenizer
    size = int(arm.model.config.vocab_size)
    limit = min(size, len(tokenizer))
    length = np.zeros(size, dtype=np.float64)
    initial = np.zeros(size, dtype=np.float64)
    upper = np.zeros(size, dtype=np.float64)
    seen = np.zeros(size, dtype=bool)
    for token_id in range(limit):
        piece = tokenizer.convert_ids_to_tokens(token_id)
        if piece is None:
            continue
        text = tokenizer.convert_tokens_to_string([piece])
        stripped = text.strip()
        if not stripped:
            continue
        seen[token_id] = True
        length[token_id] = float(len(stripped))
        initial[token_id] = 1.0 if text[:1].isspace() or piece.startswith("Ġ") else 0.0
        letters = [character for character in stripped if character.isalpha()]
        upper[token_id] = (
            sum(1 for character in letters if character.isupper()) / len(letters)
            if letters
            else 0.0
        )
    if not seen.any():
        raise ValueError(f"{arm.name}: no output token decodes to text")
    table = {
        "character_length": length,
        "word_initial": initial,
        "uppercase_fraction": upper,
    }
    for values in table.values():
        values[~seen] = float(values[seen].mean())
    return table

#: Class counts the coarsened cross-entropy is swept over. A partition is a
#: threshold and Appendix B rule 17 requires the ordering to survive its sweep.
CLASS_COUNT_SWEEP: tuple[int, ...] = (2, 3, 4, 5)

#: Fractions of a trajectory's total reduction at which resolution depth is
#: read, through ``lenses.resolution_depth``. 0.5 is what
#: ``lenses.half_resolution_depth`` reports; the other two are the sweep.
RESOLUTION_TAUS: tuple[float, ...] = (0.25, 0.5, 0.75)

#: Null draws. Both nulls are re-scorings of a cached lens distribution, so this
#: costs one matrix multiply rather than one forward pass per draw.
NULL_DRAWS = 1000

#: The final layer is the model's own distribution, so a concept that does not
#: clear this quantile of its null there is not measurable on that arm at all.
#: Declared before any run, with the arm reported unmeasurable rather than null.
POSITIVE_CONTROL_QUANTILE = 0.999

#: The quantile an intermediate-layer result must clear to be read as anything.
DECISION_QUANTILE = 0.99


# --------------------------------------------------------- declared basis


def basis_matrix(names: Sequence[str] = tuple(PROPERTY_BASIS)) -> np.ndarray:
    """The declared properties as rows over :data:`arms.AA20` order."""

    missing = [name for name in names if name not in PROPERTY_BASIS]
    if missing:
        raise KeyError(f"undeclared properties {missing}; basis is {list(PROPERTY_BASIS)}")
    return np.asarray(
        [[PROPERTY_BASIS[name][residue] for residue in AA20] for name in names],
        dtype=np.float64,
    )


def basis_correlations(weights: Sequence[float] | None = None) -> dict[str, Any]:
    """Pairwise correlations of the declared basis, for the pre-declaration.

    Computed and recorded **before** any model is consulted, because the two
    numbers that matter are predictions against the method: how far the three
    properties are from independent, and how far each is from the alphabet's own
    frequency. A property strongly correlated with log frequency is the one most
    likely to be reading composition rather than computation, and saying so in
    advance is what makes a later positive result on it interpretable.
    """

    names = tuple(PROPERTY_BASIS)
    matrix = basis_matrix(names)
    report: dict[str, Any] = {"properties": list(names), "unweighted": {}, "weighted": {}}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            j = names.index(right)
            report["unweighted"][f"{left}__{right}"] = {
                "pearson": float(stats.pearsonr(matrix[i], matrix[j]).statistic),
                "spearman": float(stats.spearmanr(matrix[i], matrix[j]).statistic),
            }
    if weights is None:
        return report
    w = _checked_weights(weights)
    log_frequency = np.log(w)
    for i, name in enumerate(names):
        report["unweighted"][f"{name}__log_frequency"] = {
            "pearson": float(stats.pearsonr(matrix[i], log_frequency).statistic),
            "spearman": float(stats.spearmanr(matrix[i], log_frequency).statistic),
        }
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            j = names.index(right)
            report["weighted"][f"{left}__{right}"] = _weighted_pearson(
                matrix[i], matrix[j], w
            )
    report["marginal_entropy_bits"] = float(-(w * np.log2(w)).sum())
    return report


def _checked_weights(weights: Sequence[float]) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (len(AA20),):
        raise ValueError(f"weights must be one per residue, got {w.shape}")
    if not np.all(w > 0.0) or not math.isclose(float(w.sum()), 1.0, rel_tol=1e-6):
        raise ValueError("weights must be a strictly positive distribution summing to one")
    return w


def _weighted_pearson(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    mx, my = float(np.average(x, weights=w)), float(np.average(y, weights=w))
    cov = float(np.average((x - mx) * (y - my), weights=w))
    vx = float(np.average((x - mx) ** 2, weights=w))
    vy = float(np.average((y - my) ** 2, weights=w))
    if vx <= 0.0 or vy <= 0.0:
        raise ValueError("a declared property is constant under these weights")
    return cov / math.sqrt(vx * vy)


def equal_mass_classes(values: Sequence[float], weights: Sequence[float], k: int) -> np.ndarray:
    """Partition symbols into ``k`` classes of near-equal probability mass.

    Equal *mass* rather than equal count, so that the coarsened marginal entropy
    is close to ``log2(k)`` whatever the property's shape. That is what makes the
    rank-matched null a null: a random partition drawn to the same mass profile
    poses a question of the same difficulty, and any remaining advantage of the
    declared property is not the coarsening.
    """

    if k < 2:
        raise ValueError("k must be at least two")
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if v.shape != w.shape or v.ndim != 1:
        raise ValueError("values and weights must be aligned one-dimensional arrays")
    order = np.argsort(v, kind="stable")
    cumulative = np.cumsum(w[order]) / w.sum()
    edges = np.searchsorted(cumulative, np.arange(1, k) / k)
    classes = np.empty(v.size, dtype=np.int64)
    classes[order] = np.searchsorted(edges, np.arange(v.size), side="left")
    _, classes = np.unique(classes, return_inverse=True)
    return classes.astype(np.int64)


def class_mass_profile(classes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    counts = np.bincount(classes, weights=weights, minlength=int(classes.max()) + 1)
    return counts / counts.sum()


# ------------------------------------------------------------------- nulls


def shuffled_property_null(
    values: Sequence[float], *, draws: int, seed: int
) -> np.ndarray:
    """Permutations of a property over the symbols, as columns.

    Column zero is the declared property itself, so every downstream statistic
    reads the truth and its null out of one array and cannot accidentally
    compare them under different code paths.
    """

    v = np.asarray(values, dtype=np.float64)
    if draws < 1:
        raise ValueError("draws must be positive")
    generator = np.random.default_rng(seed)
    columns = np.empty((v.size, draws + 1), dtype=np.float64)
    columns[:, 0] = v
    for draw in range(draws):
        columns[:, draw + 1] = generator.permutation(v)
    return columns


def rank_matched_partitions(
    classes: np.ndarray, weights: Sequence[float], *, draws: int, seed: int
) -> np.ndarray:
    """Random partitions holding the declared partition's mass profile fixed.

    The stricter of the two nulls, and the one that decides whether a coarsened
    result is about the concept. A permutation of the *symbols* between classes
    changes which residues are grouped while leaving the number of classes and
    their probability masses close to unchanged, so the coarsened entropy -- the
    quantity that makes any coarsening look good -- is held.

    Matching is by greedy assignment against the declared mass profile rather
    than by exact combinatorial enumeration, and the achieved profiles are
    returned by :func:`partition_null_quality` so the match is measured.
    """

    w = np.asarray(weights, dtype=np.float64)
    target = class_mass_profile(np.asarray(classes), w)
    n_classes = target.size
    generator = np.random.default_rng(seed)
    out = np.empty((w.size, draws + 1), dtype=np.int64)
    out[:, 0] = classes
    for draw in range(draws):
        order = generator.permutation(w.size)
        assigned = np.empty(w.size, dtype=np.int64)
        filled = np.zeros(n_classes, dtype=np.float64)
        for symbol in order:
            deficit = target - filled
            # Never leave a class empty: a partition with fewer classes than the
            # declared one is an easier question, not a matched null.
            empty = np.flatnonzero(filled == 0.0)
            choice = int(empty[np.argmax(deficit[empty])]) if empty.size else int(np.argmax(deficit))
            assigned[symbol] = choice
            filled[choice] += w[symbol]
        out[:, draw + 1] = assigned
    return out


def partition_null_quality(
    partitions: np.ndarray, weights: Sequence[float]
) -> dict[str, Any]:
    """How well the rank-matched null actually matched, as a measurement."""

    w = np.asarray(weights, dtype=np.float64)
    entropies = []
    for column in range(partitions.shape[1]):
        profile = class_mass_profile(partitions[:, column], w)
        profile = profile[profile > 0.0]
        entropies.append(float(-(profile * np.log2(profile)).sum()))
    declared = entropies[0]
    null = np.asarray(entropies[1:], dtype=np.float64)
    return {
        "declared_class_entropy_bits": declared,
        "null_class_entropy_bits_mean": float(null.mean()),
        "null_class_entropy_bits_min": float(null.min()),
        "null_class_entropy_bits_max": float(null.max()),
        "max_absolute_entropy_mismatch_bits": float(np.abs(null - declared).max()),
        "n_classes_declared": int(partitions[:, 0].max()) + 1,
        "n_null_partitions_with_fewer_classes": int(
            sum(1 for c in range(1, partitions.shape[1]) if partitions[:, c].max() + 1 < int(partitions[:, 0].max()) + 1)
        ),
    }


# ------------------------------------------------------------- statistics


def within_unit_centred_spearman(
    readout: np.ndarray, realised: np.ndarray, unit: np.ndarray
) -> np.ndarray:
    """Spearman of readout against realised value, centred within each protein.

    Centring within the sequence is what makes this statistic prior-free by
    construction rather than by control. Any predictor that is constant along a
    protein -- the arm's unigram distribution, the protein's own composition, a
    retrieval profile's marginal -- has zero within-protein variance, so its
    correlation here is exactly zero and cannot be mistaken for computation.
    The uncentred statistic is reported beside it, where those priors are not
    zero and have to be subtracted rather than assumed away.

    ``readout`` may carry a null in its trailing axis; the return has that shape.
    """

    values = np.atleast_2d(readout.T).T if readout.ndim == 1 else readout
    if values.shape[0] != realised.size or unit.size != realised.size:
        raise ValueError("readout, realised values and unit ids must be aligned")
    order = np.argsort(unit, kind="stable")
    boundaries = np.flatnonzero(np.diff(unit[order])) + 1
    centred = values.astype(np.float64, copy=True)
    centred_y = realised.astype(np.float64).copy()
    # The scale each column is judged constant against, taken before centring.
    # Subtracting a block's own mean from a block that is genuinely constant
    # leaves rounding noise of order 1e-17, and a *rank* transform turns that
    # noise into a full-range signal that correlates with the target by chance.
    # A per-protein-constant predictor -- the arm's unigram, the protein's own
    # composition, a retrieval profile's marginal -- would then score non-zero,
    # which is precisely the reading this statistic exists to make impossible.
    scale = np.abs(centred).max(axis=0, keepdims=True)
    tolerance = 1e-12 * np.where(scale > 0.0, scale, 1.0)
    for block in np.split(order, boundaries):
        if block.size < 2:
            centred[block] = np.nan
            centred_y[block] = np.nan
            continue
        centred[block] -= centred[block].mean(axis=0, keepdims=True)
        centred_y[block] -= centred_y[block].mean()
    finite = np.isfinite(centred)
    centred[finite & (np.abs(centred) < tolerance)] = 0.0
    keep = np.isfinite(centred_y) & np.isfinite(centred).all(axis=1)
    if keep.sum() < 2:
        raise ValueError("fewer than two positions survive within-unit centring")
    kept = centred[keep]
    y = centred_y[keep]
    ranks_y = stats.rankdata(y)
    out = np.empty(kept.shape[1], dtype=np.float64)
    for column in range(kept.shape[1]):
        x = kept[:, column]
        if np.ptp(x) == 0.0:
            out[column] = 0.0
            continue
        out[column] = float(np.corrcoef(stats.rankdata(x), ranks_y)[0, 1])
    return out if readout.ndim > 1 else out[:1]


def coarsened_cross_entropy(
    log_posterior: np.ndarray, target_symbol: np.ndarray, partitions: np.ndarray
) -> np.ndarray:
    """Mean nats of the true symbol's class under each partition column.

    A proper scoring rule on the coarsened outcome, and directly comparable with
    the arm's own symbol-level lens cross-entropy because both are negative log
    probabilities of an event the model assigns mass to.
    """

    if log_posterior.ndim != 2 or log_posterior.shape[0] != target_symbol.size:
        raise ValueError("log posterior must be positions by symbols, aligned to targets")
    if partitions.shape[0] != log_posterior.shape[1]:
        raise ValueError("partitions must assign every symbol")
    posterior = np.exp(log_posterior)
    out = np.empty(partitions.shape[1], dtype=np.float64)
    for column in range(partitions.shape[1]):
        classes = partitions[:, column]
        n_classes = int(classes.max()) + 1
        mass = np.zeros((posterior.shape[0], n_classes), dtype=np.float64)
        np.add.at(mass.T, classes, posterior.T)
        true_mass = mass[np.arange(target_symbol.size), classes[target_symbol]]
        out[column] = float(-np.log(np.clip(true_mass, 1e-300, None)).mean())
    return out


def null_excess(observed: float, null: np.ndarray, *, quantile: float) -> dict[str, Any]:
    """An observation against its null, stated as an excess and a rank.

    The empirical p-value uses the ``(1 + count) / (1 + draws)`` form, which
    cannot return zero: a statistic that beat every one of a thousand draws is
    evidence bounded by the number of draws taken, not evidence without bound.
    """

    values = np.asarray(null, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("a null needs at least two draws")
    cut = float(np.quantile(values, quantile))
    return {
        "observed": float(observed),
        "null_mean": float(values.mean()),
        "null_sd": float(values.std(ddof=1)),
        "null_quantile": float(quantile),
        "null_cut": cut,
        "excess_over_null_mean": float(observed - values.mean()),
        "clears_null": bool(observed > cut),
        "empirical_p": float((1 + int((values >= observed).sum())) / (1 + values.size)),
        "n_draws": int(values.size),
    }


def aperture_gain(
    depths: Sequence[float],
    symbol_values: Sequence[float],
    concept_values: Sequence[float],
) -> dict[str, Any]:
    """Depth ordering of the concept readout against symbol identity, swept.

    Positive gain means the concept resolves at shallower relative depth than the
    identity of the symbol -- the claim D3.b's semantic-aperture motivation makes
    and the one a token-level lens on a 32-symbol vocabulary cannot see. The
    sweep over ``tau`` is Appendix B rule 17: the sign has to survive it, and a
    gain that appears at one fraction only is reported as not surviving.
    """

    per_tau: dict[str, Any] = {}
    signs: list[int] = []
    for tau in RESOLUTION_TAUS:
        symbol_depth = resolution_depth(depths, symbol_values, tau)
        concept_depth = resolution_depth(depths, concept_values, tau)
        gain = (
            None
            if symbol_depth is None or concept_depth is None
            else float(symbol_depth - concept_depth)
        )
        per_tau[f"tau_{tau:.2f}"] = {
            "symbol_resolution_depth": symbol_depth,
            "concept_resolution_depth": concept_depth,
            "aperture_gain": gain,
        }
        if gain is not None:
            signs.append(int(np.sign(gain)))
    return {
        "per_tau": per_tau,
        "sign_invariant_across_sweep": bool(len(signs) == len(RESOLUTION_TAUS) and len(set(signs)) == 1),
        "n_tau_defined": len(signs),
    }


# ---------------------------------------------------- per-layer posterior


@dataclass(frozen=True)
class SymbolAxis:
    """The symbol alphabet a concept lens reads a distribution over.

    One object for both modalities so that the statistics have a single code
    path. On a protein arm the symbols are the twenty residues and
    ``token_groups`` collects the token ids that emit each one; on a text arm the
    symbols are the vocabulary itself and no grouping happens.
    """

    name: str
    symbols: tuple[str, ...]
    token_groups: tuple[torch.Tensor, ...] | None
    renormalised: bool

    @property
    def n_symbols(self) -> int:
        return len(self.symbols)


def residue_axis(vocabulary: ResidueVocabulary, *, device: str) -> SymbolAxis:
    """The twenty-residue axis, taking its token mapping from ``lenses``."""

    return SymbolAxis(
        name="residue",
        symbols=tuple(AA20),
        token_groups=tuple(group.to(device) for group in vocabulary.group_index),
        renormalised=True,
    )


def token_axis(arm: Arm) -> SymbolAxis:
    """The full-vocabulary axis a text arm is read on, with no renormalisation."""

    if arm.modality != "text":
        raise ValueError(f"{arm.name}: the token axis is the text-arm axis")
    size = int(arm.model.config.vocab_size)
    return SymbolAxis(
        name="token",
        symbols=tuple(str(index) for index in range(size)),
        token_groups=None,
        renormalised=False,
    )


@torch.no_grad()
def symbol_log_posterior(
    head: LensHead,
    residual: torch.Tensor,
    axis: SymbolAxis,
    *,
    device: str,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Lens log-posterior over the symbol axis, and the mass it discarded.

    Returns ``(log_posterior, abstain_mass)``. On a renormalised axis the
    abstained mass is the lens probability outside the alphabet, reported per
    position rather than divided away, because on ProtGPT2 that mass is the FASTA
    newline and hiding it has cost this programme a result before.
    """

    if chunk < 1:
        raise ValueError("chunk must be positive")
    rows: list[np.ndarray] = []
    abstained: list[np.ndarray] = []
    for start in range(0, residual.shape[0], chunk):
        block = residual[start : start + chunk].to(device=device, dtype=torch.float32)
        log_probs = head.log_probs(block)
        if axis.token_groups is None:
            rows.append(log_probs.cpu().numpy())
            abstained.append(np.zeros(block.shape[0], dtype=np.float64))
            continue
        columns = [
            torch.logsumexp(log_probs.index_select(-1, group), dim=-1)
            for group in axis.token_groups
        ]
        stacked = torch.stack(columns, dim=-1)
        total = torch.logsumexp(stacked, dim=-1)
        rows.append((stacked - total.unsqueeze(-1)).cpu().numpy())
        abstained.append((-torch.expm1(total)).clamp_min(0.0).cpu().numpy())
    return (
        np.concatenate(rows, axis=0).astype(np.float64),
        np.concatenate(abstained, axis=0).astype(np.float64),
    )


@torch.no_grad()
def layer_concept_statistics(
    head: LensHead,
    residual: torch.Tensor,
    axis: SymbolAxis,
    *,
    targets: np.ndarray,
    properties: Mapping[str, np.ndarray],
    partitions: Mapping[str, Mapping[int, np.ndarray]],
    device: str,
    chunk: int,
) -> dict[str, Any]:
    """Every layer statistic in one streaming pass over the scored positions.

    The posterior is never materialised. A text arm has 50,257 symbols and tens
    of thousands of scored positions, so the full float array would be tens of
    gigabytes; more importantly the coarsened cross-entropy under a thousand null
    partitions is a scatter over that array, which is the expensive operation
    here and the reason both statistics are accumulated chunk by chunk on the
    accelerator rather than assembled and then reduced.

    What survives the pass is what the statistics need: the concept readout per
    position (positions by null columns, which is small because the null axis is
    small), the mean symbol cross-entropy, and per-partition cross-entropy sums.
    """

    if chunk < 1:
        raise ValueError("chunk must be positive")
    n = int(residual.shape[0])
    if targets.size != n:
        raise ValueError("targets must align with the cached positions")
    target_tensor = torch.as_tensor(targets, dtype=torch.long, device=device)
    property_columns = {
        name: torch.as_tensor(values, dtype=torch.float32, device=device)
        for name, values in properties.items()
    }
    membership: dict[tuple[str, int], torch.Tensor] = {}
    true_class: dict[tuple[str, int], torch.Tensor] = {}
    for name, by_k in partitions.items():
        for k, columns in by_k.items():
            assignment = torch.as_tensor(columns, dtype=torch.long, device=device)
            n_classes = int(assignment.max().item()) + 1
            # Flattened to (symbols x columns*classes) so the whole null is one
            # matrix multiply per chunk. Looping the columns instead issues a
            # thousand tiny memory-bound matmuls and, on a 50,257-symbol text
            # arm, turns a seven-minute layer into an hour.
            one_hot = torch.nn.functional.one_hot(assignment, num_classes=n_classes)
            membership[(name, k)] = (
                one_hot.to(torch.float32).reshape(assignment.shape[0], -1)
            )
            offsets = torch.arange(assignment.shape[1], device=device) * n_classes
            true_class[(name, k)] = (
                assignment.index_select(0, target_tensor) + offsets.unsqueeze(0)
            )

    readouts: dict[str, list[torch.Tensor]] = {name: [] for name in property_columns}
    class_ce_sum = {
        key: torch.zeros(value.shape[1], device=device, dtype=torch.float64)
        for key, value in true_class.items()
    }
    symbol_ce_sum = torch.zeros((), device=device, dtype=torch.float64)
    abstain_sum = torch.zeros((), device=device, dtype=torch.float64)
    abstain_max = torch.zeros((), device=device, dtype=torch.float64)

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = residual[start:stop].to(device=device, dtype=torch.float32)
        log_probs = head.log_probs(block)
        if axis.token_groups is None:
            log_posterior = log_probs
            abstain = torch.zeros(stop - start, device=device)
        else:
            stacked = torch.stack(
                [
                    torch.logsumexp(log_probs.index_select(-1, group), dim=-1)
                    for group in axis.token_groups
                ],
                dim=-1,
            )
            total = torch.logsumexp(stacked, dim=-1)
            log_posterior = stacked - total.unsqueeze(-1)
            abstain = (-torch.expm1(total)).clamp_min(0.0)
        posterior = log_posterior.exp()
        rows = target_tensor[start:stop]
        symbol_ce_sum -= log_posterior.gather(1, rows.unsqueeze(1)).sum().double()
        abstain_sum += abstain.sum().double()
        abstain_max = torch.maximum(abstain_max, abstain.max().double())
        for name, columns in property_columns.items():
            readouts[name].append((posterior @ columns).cpu())
        for key, flat_membership in membership.items():
            mass = posterior @ flat_membership
            picked = mass.gather(1, true_class[key][start:stop]).clamp_min(1e-30)
            class_ce_sum[key] -= picked.log().double().sum(dim=0)

    return {
        "symbol_cross_entropy_nats": float(symbol_ce_sum.item() / n),
        "abstain_mass_mean": float(abstain_sum.item() / n),
        "abstain_mass_max": float(abstain_max.item()),
        "readout": {
            name: torch.cat(blocks, dim=0).numpy().astype(np.float64)
            for name, blocks in readouts.items()
        },
        "class_cross_entropy_nats": {
            key: (value / n).cpu().numpy().astype(np.float64)
            for key, value in class_ce_sum.items()
        },
        "n_positions": n,
    }


def target_symbols(cache: ResidualCache, axis: SymbolAxis, *, vocab_size: int) -> np.ndarray:
    """The realised symbol at every scored position.

    On the residue axis a target token that emits no canonical residue has no
    symbol; such positions are returned as ``-1`` and every statistic drops them,
    with the count reported. They are format tokens, and scoring them would be
    the all-positions spectrum error of Appendix B rule 11 in another guise.
    """

    targets = cache.target_ids.cpu().numpy().astype(np.int64)
    if axis.token_groups is None:
        if targets.max(initial=0) >= vocab_size:
            raise ValueError("a target token id lies outside the vocabulary")
        return targets
    lookup = np.full(vocab_size, -1, dtype=np.int64)
    for index, group in enumerate(axis.token_groups):
        lookup[group.cpu().numpy()] = index
    return lookup[targets]
