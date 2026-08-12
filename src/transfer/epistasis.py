"""Second-order retrieval bound: measured pairwise epistasis, and the channels it is read against.

**The subtraction this module supplies, and why F10 does not already contain it.**
F10 (EXP-R2-143) measured MODEL - LOOKUP over 217 ProteinGym substitution assays
and found no protein decoder exceeds a position-independent profile lookup over
its own pretraining corpus. Both channels in that comparison are **additive over
substitutions**: :func:`.profiles.lookup_score` is an exact sum of per-column
log-odds and :attr:`.fitness.Assay.blosum` an exact sum over the mutation string.
So both predict **identically zero epistasis for every multi-substitution
variant**. That is algebra about those two functions, not a conjecture that they
missed something, and it is the whole reason a second-order estimand is not
already bounded by the first-order result.

What replaces them at second order is a *pairwise* corpus channel -- the coupling
between two alignment columns -- and that channel is what :func:`coupling_apc`
supplies. The question D3.d asks is whether a decoder's predicted epistasis
tracks wet-lab epistasis beyond what its own corpus's column couplings already
carry.

**Three hazards this module is built around.**

*Global epistasis.* Most of the apparent pair-level epistasis in a DMS assay is
the assay's own measurement nonlinearity, not specific coupling between two
positions. Measured on ten assays before any model was loaded (EXP-R2-176):
removing a monotone fit of observed on additive costs 55-77% of the pair-level
spread on six of them, GRB2 falling from 0.237 to 0.058. A design that reads raw
double-mutant deviation as "epistasis" would therefore return a confident
correlation for any predictor that merely ranks single mutants. The correction is
:func:`specific_epistasis`, and it is **cross-fitted** rather than in-sample.

*Plug-in mutual information.* A plug-in MI over 20x20 alignment columns is biased
upward by roughly ``19**2 / (2 * Neff)`` nats and the bias grows exactly as
alignment depth falls, so shallow proteins would read as strongly coupled --
Appendix B rule 3 and L12 in second-order form. :func:`coupling_apc` cross-fits
the estimate: the joint and marginals are estimated on one fold and the log-ratio
is *evaluated* on the other, which cannot manufacture association from depth
alone, and its null is a column permutation preserving marginals and depth.

*A second spelling of an existing decision.* :func:`build_profile` already
decides which subject residue lands in which query column, and this module needs
the same alignment as a row matrix rather than as column frequencies. Rather than
restate that mapping and hope the two agree, :func:`alignment_rows` rebuilds the
rows and :func:`verify_rows_against_profile` reduces them by the same weighting
and **requires the result to reproduce** :func:`build_profile`'s frequencies.
Appendix B rule 12 asks for one declaration; where a second reader is unavoidable
this is what makes it a checked agreement instead of a divergence waiting to
happen.

Nothing here is fitted on a DMS label except the global-epistasis transform,
which is fitted on the *measured* phenotype alone and never sees a model score;
the model and corpus channels are computed without reference to either.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .fitness import parse_mutant
from .homology import Hit
from .probes import PROTEINGYM_ROOT
from .profiles import (
    AA20,
    GAP_CODE,
    NEFF_IDENTITY_FLOOR,
    PROFILE_COVERAGE_FLOOR,
    REWEIGHT_IDENTITY_FLOOR,
    Profile,
    _encode,
    sequence_weights,
)

SCHEMA_VERSION = "r2_transfer_epistasis_v1"

#: A pair is scored only when this many doubles were measured at it. Ten is what
#: makes the pair mean an estimate rather than a reading; the reliability of the
#: resulting statistic is measured per protein rather than assumed (EXP-R2-176).
MIN_DOUBLES_PER_PAIR = 10

#: The global-epistasis corrections swept. ``none`` is retained so the size of
#: the correction is visible in the artefact rather than being asserted here.
GLOBAL_EPISTASIS_MODES = ("none", "isotonic", "spline")

#: Cross-fitting folds for both the global transform and the coupling estimate.
#: Two, because both are evaluated out of fold in each direction and averaged, so
#: every unit is used exactly once as an evaluation unit.
CROSS_FIT_FOLDS = 2

#: Pseudocount on the coupling channel's fitted fold, in effective sequences.
COUPLING_PSEUDOCOUNT = 0.5

#: Symbols in the repeated span an attainability probe plants. A one-symbol
#: repeat is not a dependency any arm here is claimed to have learned: the
#: mechanism this programme has established on both modalities keys on a repeated
#: *context*, and one symbol does not supply one. Measured on the first probe
#: geometry, which planted exactly that: both arms sat at chance. A positive
#: control that plants a relation nobody claims the model has is a specification
#: defect, which is Appendix B rule 2's own category.
PLANTED_MOTIF_SYMBOLS = 8


# --------------------------------------------------------------- measured side


@dataclass(frozen=True)
class PairTable:
    """Measured specific epistasis at the position pairs of one assay."""

    assay: str
    wildtype: str
    positions: np.ndarray
    """``(n_pairs, 2)`` of one-based positions, ``i < j``."""
    epistasis: np.ndarray
    """Pair-mean out-of-fold residual about the global-epistasis transform."""
    n_doubles: np.ndarray
    additive_magnitude: np.ndarray
    """Mean ``|additive prediction|`` at the pair -- a covariate of the null, and
    computable without any model, which is standing rule 28's requirement."""
    separation: np.ndarray
    variants: tuple[tuple[tuple[str, ...], ...], ...]
    """Per pair, the measured double-mutant strings, so the model channel is
    scored on exactly the variants the phenotype channel was measured on."""
    reliability: float
    mode: str
    seed: int
    n_doubles_total: int

    def __post_init__(self) -> None:
        n = len(self.positions)
        if not n:
            raise ValueError(f"{self.assay}: no pair survived the coverage floor")
        for name in ("epistasis", "n_doubles", "additive_magnitude", "separation"):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"{self.assay}: {name} is not one value per pair")
        if len(self.variants) != n:
            raise ValueError(f"{self.assay}: variant lists are not one per pair")
        if np.any(self.positions[:, 0] >= self.positions[:, 1]):
            raise ValueError(f"{self.assay}: a pair is not ordered i < j")

    @property
    def attenuation_ceiling(self) -> float:
        """``sqrt(reliability)`` -- the largest correlation this assay can show.

        Quoted beside every statistic taken against :attr:`epistasis` rather than
        used to inflate one. A measured ceiling is what stops a small correlation
        being read as a weak model when it is a noisy referent.
        """

        return float(np.sqrt(max(self.reliability, 0.0)))

    def record(self) -> dict[str, Any]:
        return {
            "assay": self.assay,
            "wildtype_length": len(self.wildtype),
            "n_pairs": int(len(self.positions)),
            "n_doubles_scored": int(self.n_doubles.sum()),
            "n_doubles_eligible": int(self.n_doubles_total),
            "median_doubles_per_pair": float(np.median(self.n_doubles)),
            "separation_median": float(np.median(self.separation)),
            "separation_max": int(self.separation.max()),
            "reliability_split_half": float(self.reliability),
            "attenuation_ceiling": self.attenuation_ceiling,
            "global_epistasis_mode": self.mode,
            "seed": int(self.seed),
        }


def _monotone_fit(
    additive: np.ndarray, observed: np.ndarray, *, mode: str, increasing: bool
):
    """The global-epistasis transform, as a callable from additive to observed."""

    if mode == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        model = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
        model.fit(additive, observed)
        return model.predict
    if mode == "spline":
        # A monotone alternative with a different bias profile, so the sweep is
        # over a genuinely different smoother rather than over one's parameter.
        edges = np.quantile(additive, np.linspace(0.0, 1.0, 21))
        edges = np.unique(edges)
        if len(edges) < 3:
            return lambda x: np.full(np.shape(x), float(np.mean(observed)))
        index = np.clip(np.searchsorted(edges, additive, side="right") - 1, 0, len(edges) - 2)
        centres = 0.5 * (edges[:-1] + edges[1:])
        means = np.array(
            [
                observed[index == b].mean() if np.any(index == b) else np.nan
                for b in range(len(centres))
            ]
        )
        good = np.isfinite(means)
        if good.sum() < 2:
            return lambda x: np.full(np.shape(x), float(np.mean(observed)))
        from sklearn.isotonic import IsotonicRegression

        smoother = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
        smoother.fit(centres[good], means[good])
        return smoother.predict
    raise ValueError(f"unknown global-epistasis mode {mode!r}")


def specific_epistasis(
    additive: np.ndarray, observed: np.ndarray, *, mode: str, seed: int
) -> np.ndarray:
    """Observed minus the global-epistasis transform of additive, **out of fold**.

    The transform is fitted on one half of the doubles and evaluated on the
    other, in both directions, so every returned residual is out of fold and none
    of them is a residual about a curve fitted to itself. In-sample residuals
    would shrink toward zero exactly where the curve has the most data, which is
    a depth-dependent distortion of the referent the whole design is read
    against.

    Orientation is taken from the fitting fold's own rank correlation rather than
    declared: ProteinGym scores are oriented so that higher is fitter, but that
    is a property of the benchmark and not of this arithmetic, and a reversed
    assay would otherwise be fitted with a monotone curve running the wrong way.
    """

    if mode not in GLOBAL_EPISTASIS_MODES:
        raise ValueError(f"unknown global-epistasis mode {mode!r}")
    additive = np.asarray(additive, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    if additive.shape != observed.shape or additive.ndim != 1:
        raise ValueError("additive and observed must be one-dimensional and aligned")
    if mode == "none":
        return observed - additive
    from scipy import stats

    residual = np.empty_like(observed)
    fold = np.random.default_rng(seed).integers(0, CROSS_FIT_FOLDS, size=len(observed))
    for held in range(CROSS_FIT_FOLDS):
        fit = fold != held
        evaluate = ~fit
        if fit.sum() < 8 or evaluate.sum() < 1:
            raise RuntimeError(
                "a cross-fitting fold is too small for a global-epistasis transform; "
                f"{int(fit.sum())} fitting and {int(evaluate.sum())} evaluation doubles"
            )
        rank = stats.spearmanr(additive[fit], observed[fit]).statistic
        transform = _monotone_fit(
            additive[fit], observed[fit], mode=mode, increasing=not (rank < 0)
        )
        residual[evaluate] = observed[evaluate] - transform(additive[evaluate])
    return residual


def _split_half_reliability(
    values: np.ndarray, pair_index: np.ndarray, *, n_pairs: int, seed: int
) -> float:
    """Spearman-Brown reliability of the pair means, over the doubles inside each pair.

    This is the ceiling on any correlation taken against the pair means, and it
    is a property of the referent rather than of any predictor. Reported, never
    used to correct a gated statistic (Appendix B rule 25's reasoning: read the
    reliability before reading a low correlation as absence of signal, but do not
    publish the disattenuated number as the result).
    """

    rng = np.random.default_rng(seed)
    left = np.full(n_pairs, np.nan)
    right = np.full(n_pairs, np.nan)
    for pair in range(n_pairs):
        member = values[pair_index == pair]
        if len(member) < 4:
            continue
        order = rng.permutation(len(member))
        half = len(member) // 2
        left[pair] = member[order[:half]].mean()
        right[pair] = member[order[half : 2 * half]].mean()
    good = np.isfinite(left) & np.isfinite(right)
    if good.sum() < 8:
        return float("nan")
    correlation = float(np.corrcoef(left[good], right[good])[0, 1])
    if not np.isfinite(correlation) or correlation <= -1.0:
        return float("nan")
    return float(2.0 * correlation / (1.0 + correlation))


def assay_pairs(
    name: str,
    *,
    seed: int,
    min_doubles: int = MIN_DOUBLES_PER_PAIR,
    mode: str = "isotonic",
    max_doubles_per_pair: int | None = None,
    directory: Path | None = None,
) -> PairTable:
    """The measured specific-epistasis table of one ProteinGym substitution assay.

    A double mutant enters only when **both** of its constituent singles were
    measured in the same assay, because the additive prediction is what the
    global transform is fitted against and a missing single would make it an
    imputation. Where a pair carries more doubles than ``max_doubles_per_pair``
    the retained ones are drawn under a seeded permutation, never as a prefix
    (Appendix B rule 1: a ProteinGym CSV is ordered by position).
    """

    root = Path(directory) if directory is not None else PROTEINGYM_ROOT
    path = root / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"no ProteinGym assay at {path}")

    singles: dict[str, float] = {}
    doubles: list[tuple[str, str, str, float]] = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            tokens = row["mutant"].split(":")
            if len(tokens) == 1:
                singles[tokens[0]] = float(row["DMS_score"])
    with path.open() as handle:
        for row in csv.DictReader(handle):
            tokens = row["mutant"].split(":")
            if len(tokens) != 2:
                continue
            if tokens[0] in singles and tokens[1] in singles:
                doubles.append((row["mutant"], tokens[0], tokens[1], float(row["DMS_score"])))
    if not doubles:
        raise RuntimeError(f"{name}: no double mutant carries both of its singles")

    rng = np.random.default_rng(seed)
    grouped: dict[tuple[int, int], list[int]] = {}
    for index, (_, first, second, _score) in enumerate(doubles):
        (_, position_a, _), = parse_mutant(first)
        (_, position_b, _), = parse_mutant(second)
        key = (min(position_a, position_b), max(position_a, position_b))
        grouped.setdefault(key, []).append(index)

    kept_keys = sorted(key for key, members in grouped.items() if len(members) >= min_doubles)
    if not kept_keys:
        raise RuntimeError(
            f"{name}: no position pair carries {min_doubles} doubles with both singles"
        )

    selected: list[int] = []
    pair_index: list[int] = []
    for pair, key in enumerate(kept_keys):
        members = np.asarray(grouped[key])
        order = rng.permutation(len(members))
        if max_doubles_per_pair is not None:
            order = order[:max_doubles_per_pair]
        chosen = members[np.sort(order)]
        selected.extend(chosen.tolist())
        pair_index.extend([pair] * len(chosen))

    pair_index_array = np.asarray(pair_index)
    additive = np.array(
        [singles[doubles[i][1]] + singles[doubles[i][2]] for i in selected], dtype=np.float64
    )
    observed = np.array([doubles[i][3] for i in selected], dtype=np.float64)
    residual = specific_epistasis(additive, observed, mode=mode, seed=seed + 1)

    n_pairs = len(kept_keys)
    epistasis = np.array(
        [residual[pair_index_array == pair].mean() for pair in range(n_pairs)]
    )
    n_doubles = np.array(
        [int((pair_index_array == pair).sum()) for pair in range(n_pairs)]
    )
    magnitude = np.array(
        [np.abs(additive[pair_index_array == pair]).mean() for pair in range(n_pairs)]
    )
    variants = tuple(
        tuple(doubles[i][0] for i in np.asarray(selected)[pair_index_array == pair])
        for pair in range(n_pairs)
    )
    positions = np.asarray(kept_keys, dtype=np.int64)

    wildtype = _wildtype_from(doubles[selected[0]][0], root / f"{name}.csv")
    return PairTable(
        assay=name,
        wildtype=wildtype,
        positions=positions,
        epistasis=epistasis,
        n_doubles=n_doubles,
        additive_magnitude=magnitude,
        separation=(positions[:, 1] - positions[:, 0]).astype(np.float64),
        variants=variants,
        reliability=_split_half_reliability(
            residual, pair_index_array, n_pairs=n_pairs, seed=seed + 2
        ),
        mode=mode,
        seed=seed,
        n_doubles_total=len(doubles),
    )


def _wildtype_from(mutant: str, path: Path) -> str:
    """The assay's wild type, reverting the first row that carries ``mutant``."""

    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row["mutant"] != mutant:
                continue
            out = list(row["mutated_sequence"])
            for wild, position, mutated in parse_mutant(mutant):
                if out[position - 1] != mutated:
                    raise ValueError(
                        f"{mutant}: variant carries {out[position - 1]!r} at position "
                        f"{position} where the mutation string says {mutated!r}"
                    )
                out[position - 1] = wild
            return "".join(out)
    raise RuntimeError(f"{path.name}: {mutant} is not in the file it was read from")


# ------------------------------------------------------------- corpus channel


def alignment_rows(
    wildtype: str,
    query_id: str,
    hits: Sequence[Hit],
    *,
    max_sequences: int,
    coverage_floor: float = PROFILE_COVERAGE_FLOOR,
    reweight_identity: float = REWEIGHT_IDENTITY_FLOOR,
) -> tuple[np.ndarray, np.ndarray, float]:
    """The corpus alignment of one wild type as ``(sequences, columns)`` residue codes.

    The column mapping is :func:`build_profile`'s, restated here because that
    function returns frequencies and a coupling estimate needs the rows. The
    restatement is checked rather than trusted -- see
    :func:`verify_rows_against_profile`, which every caller is expected to run.
    """

    length = len(wildtype)
    best: dict[str, Hit] = {}
    for hit in hits:
        if hit.qlen != length:
            raise ValueError(
                f"{query_id}: DIAMOND reports qlen {hit.qlen} for a "
                f"{length}-residue wild type"
            )
        if hit.qseq_gapped is None or hit.sseq_gapped is None:
            raise ValueError(
                f"{query_id}: hit against {hit.subject} carries no aligned sequences; "
                "the search must request homology.ALIGNMENT_FIELDS"
            )
        if 100.0 * (hit.qend - hit.qstart + 1) / length < coverage_floor:
            continue
        previous = best.get(hit.subject)
        if previous is None or hit.bitscore > previous.bitscore:
            best[hit.subject] = hit
    ordered = sorted(best.values(), key=lambda hit: -hit.bitscore)[:max_sequences]

    rows = np.full((len(ordered), length), GAP_CODE, dtype=np.int8)
    for index, hit in enumerate(ordered):
        if len(hit.qseq_gapped) != len(hit.sseq_gapped):
            raise ValueError(
                f"{query_id}: gapped query and subject differ in length against "
                f"{hit.subject}; the search must request qseq_gapped/sseq_gapped"
            )
        position = hit.qstart - 1
        codes = _encode(hit.sseq_gapped)
        for column, residue in enumerate(hit.qseq_gapped):
            if residue == "-":
                continue
            if position >= length:
                raise ValueError(
                    f"{query_id}: alignment against {hit.subject} runs past the "
                    "wild type's last residue"
                )
            rows[index, position] = codes[column]
            position += 1
        if position != hit.qend:
            raise ValueError(
                f"{query_id}: alignment against {hit.subject} covers query residues "
                f"{hit.qstart}-{position} where DIAMOND reports {hit.qstart}-{hit.qend}"
            )
    weights = (
        sequence_weights(rows, identity=reweight_identity)
        if len(ordered)
        else np.zeros(0, dtype=np.float64)
    )
    neff = float(
        sum(
            weight
            for weight, hit in zip(weights, ordered)
            if hit.identity_over_query >= NEFF_IDENTITY_FLOOR
        )
    )
    return rows, weights, neff


def verify_rows_against_profile(
    rows: np.ndarray, weights: np.ndarray, profile: Profile, *, tolerance: float = 1e-9
) -> None:
    """Require the rebuilt rows to reduce to the profile they were rebuilt beside.

    Appendix B rule 12 asks for one declaration of a shared decision. Two readers
    of one alignment are unavoidable here, because a coupling needs rows and the
    existing declaration returns columns; what is avoidable is their disagreeing
    silently. This raises instead.
    """

    frequencies = np.zeros_like(profile.frequencies, dtype=np.float64)
    for code in range(len(AA20)):
        frequencies[:, code] = (weights[:, None] * (rows == code)).sum(axis=0)
    column_weight = frequencies.sum(axis=1)
    positive = column_weight > 0
    frequencies[positive] /= column_weight[positive, None]
    difference = float(np.abs(frequencies - profile.frequencies).max())
    if difference > tolerance:
        raise RuntimeError(
            f"{profile.query_id}: the rebuilt alignment rows do not reproduce "
            f"build_profile's frequencies (max |difference| {difference:.3e} > "
            f"{tolerance:.0e}); the two readers of this alignment have diverged"
        )


def _pair_counts(
    rows: np.ndarray,
    weights: np.ndarray,
    columns: np.ndarray,
    subset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted joint counts over ``subset`` rows, plus the per-column present mass."""

    codes = rows[np.ix_(subset, columns)]
    present = codes != GAP_CODE
    weight = weights[subset]
    n = len(columns)
    joint = np.zeros((n, n, len(AA20), len(AA20)), dtype=np.float64)
    for a in range(n):
        for b in range(a + 1, n):
            both = present[:, a] & present[:, b]
            if not both.any():
                continue
            np.add.at(
                joint[a, b],
                (codes[both, a], codes[both, b]),
                weight[both],
            )
    return joint, present.astype(np.float64).T @ weight


def coupling_apc(
    rows: np.ndarray,
    weights: np.ndarray,
    columns: Sequence[int],
    *,
    seed: int,
    pseudocount: float = COUPLING_PSEUDOCOUNT,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-fitted, APC-corrected column coupling over a set of query columns.

    The estimate is **held out, never plug-in** (Appendix B rule 3). The joint
    and the two marginals are estimated on one fold with a pseudocount, and the
    log-ratio is *evaluated* on the other fold's pairs; the two directions are
    averaged. Under independence the expected value of an out-of-fold log-ratio
    is at most zero, so alignment depth cannot manufacture a coupling the way a
    plug-in mutual information does -- its bias is ``19**2 / (2 * Neff)`` and
    grows exactly where the corpus support is weakest, which is where a spurious
    coupling would most flatter the retrieval channel.

    The average-product correction is taken over the **submatrix of scored
    columns**, and that is a declared choice rather than the usual whole-protein
    background: the columns scored here are the ones a DMS assay mutated, so a
    whole-protein background would be a different population from the one the
    statistic is read over.

    Returns ``(coupling, effective_depth)``, both ``(n_columns, n_columns)``.
    """

    columns = np.asarray(list(columns), dtype=np.int64)
    n = len(columns)
    if n < 2:
        raise ValueError("a coupling needs at least two columns")
    if rows.shape[0] != len(weights):
        raise ValueError("one weight per alignment row")
    coupling = np.zeros((n, n), dtype=np.float64)
    depth = np.zeros((n, n), dtype=np.float64)
    if rows.shape[0] < 2 * CROSS_FIT_FOLDS:
        return coupling, depth

    fold = np.random.default_rng(seed).integers(0, CROSS_FIT_FOLDS, size=rows.shape[0])
    accumulated = np.zeros((n, n), dtype=np.float64)
    contributions = np.zeros((n, n), dtype=np.float64)
    for held in range(CROSS_FIT_FOLDS):
        fit = np.flatnonzero(fold != held)
        evaluate = np.flatnonzero(fold == held)
        if len(fit) < 2 or len(evaluate) < 1:
            continue
        joint_fit, _ = _pair_counts(rows, weights, columns, fit)
        joint_eval, _ = _pair_counts(rows, weights, columns, evaluate)
        for a in range(n):
            for b in range(a + 1, n):
                fitted = joint_fit[a, b] + pseudocount / (len(AA20) ** 2)
                total = fitted.sum()
                if total <= 0:
                    continue
                fitted /= total
                marginal_a = fitted.sum(axis=1)
                marginal_b = fitted.sum(axis=0)
                ratio = np.log(fitted) - np.log(marginal_a)[:, None] - np.log(marginal_b)[None, :]
                mass = joint_eval[a, b]
                observed = mass.sum()
                if observed <= 0:
                    continue
                accumulated[a, b] += float((mass * ratio).sum())
                contributions[a, b] += observed
                depth[a, b] += observed
    scored = contributions > 0
    coupling[scored] = accumulated[scored] / contributions[scored]
    coupling = coupling + coupling.T
    depth = depth + depth.T

    # Average-product correction over the scored submatrix. Row means exclude the
    # diagonal, which carries no estimate here.
    off = ~np.eye(n, dtype=bool)
    row_mean = np.where(off, coupling, 0.0).sum(axis=1) / max(n - 1, 1)
    grand = float(np.where(off, coupling, 0.0).sum() / max(n * (n - 1), 1))
    if abs(grand) > 0:
        coupling = coupling - np.outer(row_mean, row_mean) / grand
    coupling[~off] = 0.0
    return coupling, depth


def coupling_column_null(
    rows: np.ndarray,
    weights: np.ndarray,
    columns: Sequence[int],
    *,
    draws: int,
    seed: int,
) -> np.ndarray:
    """``draws`` couplings with each column independently permuted across rows.

    A column permutation preserves that column's residue marginal and its gap
    pattern's depth exactly and destroys only which row pairs which residue with
    which. It is therefore the construction-matched null for this channel, in the
    same sense EXP-R2-171's collision null is for the induction census: the thing
    that is held is everything the estimator reads except the association.
    """

    columns = np.asarray(list(columns), dtype=np.int64)
    rng = np.random.default_rng(seed)
    out = np.empty((draws, len(columns), len(columns)), dtype=np.float64)
    for draw in range(draws):
        shuffled = rows.copy()
        for column in columns:
            shuffled[:, column] = rows[rng.permutation(rows.shape[0]), column]
        out[draw] = coupling_apc(
            shuffled, weights, columns, seed=int(rng.integers(1 << 30))
        )[0]
    return out


# ------------------------------------------------------ construction-matched null


def stratum_labels(covariates: Mapping[str, np.ndarray], *, quantiles: int) -> np.ndarray:
    """A stratum index per unit, from the product of per-covariate quantile bins.

    Every covariate handed here must be computable without the model; that is
    standing rule 28's requirement and it is the property that makes a
    permutation inside these strata a *construction-matched* null rather than a
    label shuffle.
    """

    if not covariates:
        raise ValueError("a stratification needs at least one covariate")
    sizes = {len(value) for value in covariates.values()}
    if len(sizes) != 1:
        raise ValueError("all covariates must carry one value per unit")
    n = sizes.pop()
    label = np.zeros(n, dtype=np.int64)
    for values in covariates.values():
        values = np.asarray(values, dtype=np.float64)
        edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, quantiles + 1)))
        binned = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, max(len(edges) - 2, 0))
        label = label * max(len(edges) - 1, 1) + binned
    _, compact = np.unique(label, return_inverse=True)
    return compact


def stratified_permutation_null(
    statistic: np.ndarray,
    referent: np.ndarray,
    strata: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Spearman of ``statistic`` against ``referent``, and its within-stratum null.

    The permutation moves ``statistic`` among units **inside** a stratum, so
    every covariate that defines the stratum is preserved exactly and only the
    unit's identity is destroyed. A stratum of size one contributes no
    permutation and is reported rather than dropped, because a null built mostly
    from singletons is not a null and the reader has to be able to see it.
    """

    from scipy import stats

    statistic = np.asarray(statistic, dtype=np.float64)
    referent = np.asarray(referent, dtype=np.float64)
    strata = np.asarray(strata)
    if not statistic.shape == referent.shape == strata.shape:
        raise ValueError("statistic, referent and strata must be aligned")
    observed = float(stats.spearmanr(statistic, referent).statistic)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == value) for value in np.unique(strata)]
    permutable = sum(len(group) for group in groups if len(group) > 1)
    null = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        shuffled = statistic.copy()
        for group in groups:
            if len(group) > 1:
                shuffled[group] = statistic[rng.permutation(group)]
        null[draw] = stats.spearmanr(shuffled, referent).statistic
    finite = null[np.isfinite(null)]
    return {
        "observed": observed,
        "null_mean": float(np.mean(finite)) if len(finite) else float("nan"),
        "null_sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan"),
        "null_q95": float(np.quantile(finite, 0.95)) if len(finite) else float("nan"),
        "null_q99": float(np.quantile(finite, 0.99)) if len(finite) else float("nan"),
        "excess_over_null_mean": (
            observed - float(np.mean(finite)) if len(finite) else float("nan")
        ),
        "n_units": int(len(statistic)),
        "n_strata": int(len(groups)),
        "n_units_in_permutable_strata": int(permutable),
        "draws": int(draws),
    }


# ----------------------------------------------------------- A1 planted probes


@dataclass(frozen=True)
class CouplingProbe:
    """One sequence carrying exactly one planted copy dependency.

    The planted pair satisfies ``symbols[j] == symbols[i]``; the control pairs
    are the sequence as it was drawn. All pairs share the substituting symbol and
    the separation, so a statistic that separates them is separating the planted
    relation and not the geometry.

    The unit is a *symbol* rather than a character, and ``separator`` is what
    joins them, because the natural symbol differs by modality: a residue on a
    protein arm, joined by nothing, and a word on the text control, joined by a
    space. Substituting one character of English is not the text analogue of
    substituting one residue -- it is a typo, and it would retokenise the
    sequence around it. This keeps one implementation and one arithmetic while
    letting each modality carry its own unit, which is the same discipline
    ``Cohort.input_strings`` applies to rendering.
    """

    index: int
    symbols: tuple[str, ...]
    separator: str
    pairs: tuple[tuple[int, int], ...]
    planted: int
    substitute: str
    second_substitute: str
    motif_span: tuple[int, int, int]
    """``(first_site, second_site, length)`` of the planted repeat, in symbols.

    Carried so a caller can ask whether the arm's tokenizer preserved the repeat
    it was handed. On a residue-level arm it always does; on a multi-residue BPE
    arm two identical residue spans at different phase offsets need not tokenise
    alike, and then the probe presents no token-level repeat for an induction
    mechanism to key on. That is an interface fact about the arm and not a
    statement about its computation, and the two are only separable if the
    geometry is recorded."""

    @property
    def sequence(self) -> str:
        return self.separator.join(self.symbols)

    def variants(self) -> list[tuple[str, str]]:
        """``(role, sequence)`` for the wild type and every pair's mutants.

        **Two substitution schemes, because the obvious one contaminates its own
        control, and that is provable without looking at a number.** Write ``B``
        for the log-probability a model gains when a symbol repeats an earlier
        one. Under the *shared*-substitute scheme -- position ``i`` and position
        ``j`` both set to the same symbol ``a`` -- the planted pair reads
        ``m = B - 0 - 0 + B = 2B`` because the wild type and the double mutant
        both carry a repeat, but a **control** pair reads ``m = B - 0 - 0 + 0 = B``
        because its double mutant *plants a new repeat that was not there*. The
        contrast is then 2B against B rather than B against zero, and a control
        that acquires half the effect it exists to exclude is not a control.

        Under the *distinct*-substitute scheme -- ``i`` to ``a`` and ``j`` to a
        different ``b`` -- no variant of a control pair ever repeats, so the
        control reads zero and the planted pair reads ``m = 0 - 0 - 0 + B = B``,
        carried entirely by the wild type's own relation. That is the scheme the
        gate is read on.

        Both are scored, at a cost of two extra sequences per pair, so the
        contamination is reported as a measurement beside the argument for it
        rather than asserted. Roles: ``wt``; ``a{k}``, ``b{k}``, ``ab{k}`` for
        the shared scheme; ``b2{k}``, ``ab2{k}`` for the distinct one, which
        reuses ``a{k}`` because position ``i`` is treated identically in both.
        """

        out = [("wt", self.sequence)]
        for k, (i, j) in enumerate(self.pairs):
            first = list(self.symbols)
            first[i] = self.substitute
            shared_j = list(self.symbols)
            shared_j[j] = self.substitute
            shared_both = list(first)
            shared_both[j] = self.substitute
            distinct_j = list(self.symbols)
            distinct_j[j] = self.second_substitute
            distinct_both = list(first)
            distinct_both[j] = self.second_substitute
            out.append((f"a{k}", self.separator.join(first)))
            out.append((f"b{k}", self.separator.join(shared_j)))
            out.append((f"ab{k}", self.separator.join(shared_both)))
            out.append((f"b2{k}", self.separator.join(distinct_j)))
            out.append((f"ab2{k}", self.separator.join(distinct_both)))
        return out


def planted_coupling_probes(
    records: Sequence[Sequence[str]],
    *,
    n_pairs: int,
    separation: int,
    alphabet: Sequence[str],
    separator: str,
    seed: int,
    motif: int = PLANTED_MOTIF_SYMBOLS,
) -> list[CouplingProbe]:
    """Sequences carrying one planted copy dependency among ``n_pairs`` candidates.

    At the planted pair the wild type and the double mutant both satisfy
    ``s[j] == s[i]`` while each single mutant breaks the relation, so a model
    that has learned to copy must return a large positive predicted epistasis
    there and no such requirement holds at the controls. This is the estimand's
    own arithmetic applied to a referent that is true by construction, which is
    the only form standing rule 2 can take when the biological referent has no
    text analogue.

    Every pair in a probe carries the same separation, so the within-probe
    permutation null is separation-matched without a stratifier.
    """

    if n_pairs < 2:
        raise ValueError("a probe needs a planted pair and at least one control")
    if separation < 1:
        raise ValueError("separation must be positive")
    rng = np.random.default_rng(seed)
    probes: list[CouplingProbe] = []
    for index, record in enumerate(records):
        symbols = list(record)
        span = separation + motif + 2
        if len(symbols) < span + n_pairs * 2 + 4:
            continue
        # The repeated span, and the aligned pair inside it. A one-symbol repeat
        # is not a dependency either modality is claimed to have learned -- an
        # induction mechanism keys on a repeated *context*, which one symbol does
        # not supply -- so the planted relation is a copied motif and the planted
        # pair is a position inside it against its own image one separation later.
        # This is `04_circuit_primitives.py`'s [prefix][S][S] geometry, with the
        # scored pair named rather than left implicit.
        first_site = int(rng.integers(1, max(2, len(symbols) - separation - motif - 1)))
        second_site = first_site + separation
        if second_site + motif >= len(symbols):
            continue
        symbols[second_site : second_site + motif] = symbols[
            first_site : first_site + motif
        ]
        # Late in the motif, so the model has read enough of the repeat to
        # recognise it before the scored position arrives.
        offset = int(rng.integers(motif // 2, motif))
        planted_pair = (first_site + offset, first_site + offset + separation)

        forbidden = set(range(first_site, first_site + motif)) | set(
            range(second_site, second_site + motif)
        )
        controls: list[tuple[int, int]] = []
        attempts = 0
        while len(controls) < n_pairs - 1 and attempts < 200:
            attempts += 1
            start = int(rng.integers(0, len(symbols) - separation))
            stop = start + separation
            if start in forbidden or stop in forbidden:
                continue
            if any(abs(start - a) < 2 or abs(stop - b) < 2 for a, b in controls):
                continue
            if symbols[start] == symbols[stop]:
                # An accidental match is a second planted pair the referent
                # scores as zero, so it is excluded rather than left to add noise
                # in the direction that defeats the gate.
                continue
            controls.append((start, stop))
        if len(controls) < n_pairs - 1:
            continue

        pairs = [planted_pair] + controls
        order = rng.permutation(len(pairs))
        shuffled = [pairs[position] for position in order]
        planted = int(np.flatnonzero(order == 0)[0])

        anchor = symbols[planted_pair[0]]
        candidates = [symbol for symbol in alphabet if symbol != anchor]
        if len(candidates) < 2:
            continue
        first, second = rng.choice(len(candidates), size=2, replace=False)
        probes.append(
            CouplingProbe(
                index=index,
                symbols=tuple(symbols),
                separator=separator,
                pairs=tuple(shuffled),
                planted=planted,
                substitute=candidates[int(first)],
                second_substitute=candidates[int(second)],
                motif_span=(first_site, second_site, motif),
            )
        )
    if not probes:
        raise RuntimeError(
            "no drawn record is long enough for the requested pair geometry"
        )
    return probes


def probe_epistasis(
    scores: Mapping[str, float], probe: CouplingProbe, *, scheme: str = "distinct"
) -> np.ndarray:
    """``logP(ab) - logP(a) - logP(b) + logP(wt)`` for each pair of one probe.

    The same difference-in-differences the measured estimand uses, so the
    attainability gate exercises the arithmetic that will carry the result rather
    than a proxy for it. ``scheme`` selects which substitution scheme's variants
    are read; see :meth:`CouplingProbe.variants` for why ``shared`` contaminates
    its own control and ``distinct`` is what the gate is read on.
    """

    if scheme not in ("shared", "distinct"):
        raise ValueError(f"unknown substitution scheme {scheme!r}")
    suffix = "" if scheme == "shared" else "2"
    wt = scores["wt"]
    return np.array(
        [
            scores[f"ab{suffix}{k}"] - scores[f"a{k}"] - scores[f"b{suffix}{k}"] + wt
            for k in range(len(probe.pairs))
        ],
        dtype=np.float64,
    )
