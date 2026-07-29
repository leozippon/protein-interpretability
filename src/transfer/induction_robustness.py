"""Is the induction-head deficit a threshold artefact, or a scale artefact?

The programme's one surviving cross-modality finding is that protein decoders
carry a lower *prevalence* of induction heads than text decoders.  As stated it
rests on a single number per arm -- the fraction of heads whose prefix-matching
score clears 0.10 -- and that number is exposed to two objections which are
really the same objection twice.

*The threshold is arbitrary.*  A fraction above a cut-off is a thresholded
statistic, and this programme has already been bitten by one: in the
path-patching work a ratio-floor choice moved GPT-2-large's mediated fraction
from 0.234 to 0.574 while ProtGPT2's did not move at all.  A cut-off that
reorders the arms is not a measurement of the arms.  So every fixed threshold
the census already emits is read here, together with the per-arm data-driven cut
(mean + 3 sd of that arm's own head distribution), and -- the part that actually
settles it -- statistics that use no cut-off at all.  The rank statistics are
primary.  :func:`pairwise_auc` is the probability that a head drawn at random
from a text arm out-scores a head drawn at random from a protein arm, which is
defined without reference to any threshold and is invariant to any monotone
rescaling of the score.

*Scale and modality are confounded.*  The fraction divides by head count and by
nothing else, so it is not normalised for depth, width or parameter count.  In
the seven-arm panel the lowest text arm, ``llama-3.2-3b``, is also the widest and
deepest, which is exactly the pattern a pure scale effect would produce.
:func:`scale_modality_fit` enters a modality indicator alongside one scale
covariate at a time and reports, beside every coefficient, how collinear the two
regressors are.  **n = 7.  This is descriptive, not inferential**, and the module
says so in its own output: :func:`scale_modality_fit` records
``inferential=False`` on every fit and :func:`collinearity_verdict` returns
``cannot_separate`` when the modality indicator and the scale covariate are close
enough to identical that no fit could tell them apart.  A coefficient that looks
clean under a design that cannot identify it is worse than no coefficient.

The fit is not the strongest evidence available and is not presented as such.
Two model pairs settle more than the regression does:

- ``gpt2-large`` against ``ProtGPT2`` is matched on depth, width, head count,
  vocabulary size and parameter count to four significant figures, so any
  difference between them is not a scale difference.
- ``dialogpt-small`` against ``ProtGPT2`` *inverts* the scale direction: the text
  arm is six times smaller and still carries the higher fraction.  A pure scale
  account predicts the opposite sign, so this pair falsifies it directly.

:func:`scale_inversion_checks` computes both.

*Sampling units.*  Three things are conflated in casual talk about
"bootstrapping the census" and they are kept apart here.  Probes are the sampling
unit, so :mod:`scripts.transfer.12_induction_robustness` resamples probes as
clusters when per-probe scores are available.  A layer is not a sampling unit at
all -- it is a coordinate of the model.  And the heads of one model are a
*population*, not a draw from a superpopulation: an interval around a within-model
head fraction would answer a question nobody asked.  The unit for the
cross-modality claim is the model, n = 4 text against n = 3 protein, and
:func:`model_level_exact_test` reports it as such, with the floor on the
attainable p-value stated rather than buried.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

SCHEMA_VERSION = "r2_transfer_induction_robustness_v1"

#: Probe whose census the headline table is read from.  The synthetic probe is
#: the one the seven-arm comparison was measured on and is the only probe for
#: which all seven arms share a cohort by construction: it is built in token
#: space from each arm's own unigram, so its uniform-attention baseline is the
#: same 0.0107 for every arm but ZymCTRL, whose EC prefix shifts the scored
#: positions.  The natural probes draw on different corpora per modality and are
#: reported alongside as a sensitivity, never as the headline.
PRIMARY_PROBE = "synthetic_repeat"

PROBES: tuple[str, ...] = (
    "synthetic_repeat",
    "natural_repeat_exact",
    "natural_repeat_approximate",
)

#: Key into ``induction.per_head`` for each probe's per-head score matrix.
PER_HEAD_KEY = {
    "synthetic_repeat": "prefix_matching_synthetic",
    "natural_repeat_exact": "prefix_matching_natural_exact",
    "natural_repeat_approximate": "prefix_matching_natural_approximate",
}

MODALITIES = ("text", "protein")

#: Scale covariates offered to the fit, one at a time.  ``log10_parameters`` is
#: the conventional scale axis; the other three are the specific structural
#: quantities the head-count fraction fails to normalise for.
SCALE_COVARIATES: tuple[str, ...] = (
    "log10_parameters",
    "n_layer",
    "n_head",
    "d_model",
)

#: |correlation| between the modality indicator and a scale covariate above which
#: the two are declared inseparable in this panel.  Set at the level where the
#: variance inflation factor passes 10, the conventional line: r = 0.9487 gives
#: VIF = 1 / (1 - r^2) = 10.  Declared here so the verdict cannot be chosen after
#: the correlations are seen.
COLLINEARITY_LIMIT = 0.9487


def _finite(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{label} is not finite")
    return float(value)


@dataclass(frozen=True)
class ArmCensus:
    """One arm's induction census, as read off a ``circuit_primitives`` artefact.

    ``scores`` is the per-head prefix-matching matrix of shape (layer, head).  It
    is the full distribution, so every threshold and every rank statistic in this
    module is recomputed from it rather than being taken on trust from the
    counts the census happened to emit -- which also means the recomputed counts
    can be checked against the stored ones, and are.
    """

    name: str
    modality: str
    architecture: str
    n_layer: int
    n_head_per_layer: int
    d_model: int
    parameters: int
    probe: str
    scores: np.ndarray
    uniform_baseline: float
    n_probes: int
    stored_counts: Mapping[str, int]
    data_driven_threshold: float
    stored_data_driven_count: int
    source: Path

    def __post_init__(self) -> None:
        if self.modality not in MODALITIES:
            raise ValueError(f"{self.name}: unknown modality {self.modality!r}")
        if self.scores.shape != (self.n_layer, self.n_head_per_layer):
            raise ValueError(
                f"{self.name}: score matrix {self.scores.shape} does not match the "
                f"declared shape ({self.n_layer}, {self.n_head_per_layer})"
            )
        if not np.isfinite(self.scores).all():
            raise ValueError(f"{self.name}: score matrix carries a non-finite entry")
        if self.uniform_baseline <= 0.0:
            raise ValueError(f"{self.name}: uniform baseline must be positive")

    @property
    def n_heads(self) -> int:
        return int(self.scores.size)

    @property
    def flat(self) -> np.ndarray:
        return self.scores.reshape(-1)

    @property
    def flat_over_uniform(self) -> np.ndarray:
        """Head scores in multiples of the probe's own uniform-attention baseline.

        The baseline is ``mean(1 / (position + 1))`` over the scored queries, so
        it differs between arms whose repeats sit at different token depths and a
        raw score compares two heads at two different chance levels.  On the
        synthetic probe the baselines are near-identical by construction, so this
        rescaling changes almost nothing there and is reported anyway: a rank
        statistic that moved under it would be a baseline artefact, and knowing
        that it does not move is the point.
        """

        return self.flat / self.uniform_baseline

    def fraction_above(self, threshold: float) -> float:
        return float((self.flat >= threshold).sum()) / self.n_heads


def load_census(
    path: Path,
    *,
    parameters: int,
    probe: str = PRIMARY_PROBE,
) -> ArmCensus:
    """Read one arm's census out of a ``circuit_primitives`` artefact.

    ``parameters`` is supplied by the caller rather than read from the artefact
    because the artefact does not carry it; :func:`count_parameters` derives it
    from the checkpoint on disk.
    """

    import json

    if probe not in PROBES:
        raise ValueError(f"unknown probe {probe!r}")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    arm = payload["arm"]
    induction = payload["induction"]
    if probe not in induction:
        raise KeyError(f"{path}: artefact carries no {probe} census")
    alignment = induction[probe]
    census = alignment["census"]
    scores = np.asarray(induction["per_head"][PER_HEAD_KEY[probe]], dtype=np.float64)
    return ArmCensus(
        name=arm["name"],
        modality=arm["modality"],
        # ``architecture`` is absent from the v1 schema; the four v1 artefacts
        # are all GPT-2-family or ProGen and the field is descriptive only.
        architecture=arm.get("architecture") or "unrecorded",
        n_layer=int(arm["n_layer"]),
        n_head_per_layer=int(scores.shape[1]),
        d_model=int(arm["d_model"]),
        parameters=int(parameters),
        probe=probe,
        scores=scores,
        uniform_baseline=_finite(float(alignment["uniform_baseline"]), "uniform baseline"),
        n_probes=int(alignment["n_probes"]),
        stored_counts={str(k): int(v) for k, v in census["count_above_threshold"].items()},
        data_driven_threshold=_finite(
            float(census["data_driven_threshold"]), "data-driven threshold"
        ),
        stored_data_driven_count=int(census["count_above_data_driven"]),
        source=Path(path),
    )


def verify_against_stored_counts(arm: ArmCensus) -> dict[str, Any]:
    """Recompute every stored count from the score matrix and demand agreement.

    The whole analysis re-derives the census from ``per_head``, so if that matrix
    were not the matrix the stored counts were computed from, every number here
    would be wrong in a way no downstream check would catch.  Disagreement is an
    error, not a warning.
    """

    mismatches: list[str] = []
    for label, stored in sorted(arm.stored_counts.items()):
        recomputed = int((arm.flat >= float(label)).sum())
        if recomputed != stored:
            mismatches.append(f"{label}: stored {stored}, recomputed {recomputed}")
    recomputed_dd = int((arm.flat >= arm.data_driven_threshold).sum())
    if recomputed_dd != arm.stored_data_driven_count:
        mismatches.append(
            f"data-driven: stored {arm.stored_data_driven_count}, recomputed {recomputed_dd}"
        )
    if mismatches:
        raise ValueError(f"{arm.name}: per-head matrix disagrees with stored census; " + "; ".join(mismatches))
    return {
        "arm": arm.name,
        "thresholds_checked": sorted(arm.stored_counts),
        "data_driven_threshold": arm.data_driven_threshold,
        "agreed": True,
    }


# ------------------------------------------------------------ threshold sweep


def threshold_sweep(arms: Sequence[ArmCensus]) -> dict[str, Any]:
    """Per-arm fraction at every threshold the census carries, plus the ordering.

    Reports, at each threshold: every arm's fraction; the lowest text fraction
    and the highest protein fraction; their ratio; and whether the two modality
    ranges are disjoint.  The ratio is the statistic the finding is quoted at, so
    it is the one whose threshold dependence matters.  A ratio at or below one is
    a broken separation and is labelled as such rather than being reported as a
    small effect.
    """

    fixed = sorted({float(label) for arm in arms for label in arm.stored_counts})
    rows: list[dict[str, Any]] = []
    for threshold in fixed:
        rows.append(_sweep_row(arms, f"{threshold:.2f}", {arm.name: threshold for arm in arms}))
    rows.append(
        _sweep_row(
            arms,
            "data_driven_mean_plus_3sd",
            {arm.name: arm.data_driven_threshold for arm in arms},
        )
    )
    orderings_preserved = all(row["separation_holds"] for row in rows)
    breaks = [row["threshold"] for row in rows if not row["separation_holds"]]
    return {
        "rows": rows,
        "fixed_thresholds": fixed,
        "separation_holds_at_every_threshold": orderings_preserved,
        "thresholds_where_separation_breaks": breaks,
        "note": (
            "the data-driven cut is per-arm (mean + 3 sd of that arm's own head "
            "distribution), so it is not a common cut-off and its row compares "
            "arms standardised by their own dispersion rather than on one scale"
        ),
    }


def _sweep_row(
    arms: Sequence[ArmCensus],
    label: str,
    cuts: Mapping[str, float],
) -> dict[str, Any]:
    fractions = {arm.name: arm.fraction_above(cuts[arm.name]) for arm in arms}
    counts = {arm.name: int((arm.flat >= cuts[arm.name]).sum()) for arm in arms}
    text = {a.name: fractions[a.name] for a in arms if a.modality == "text"}
    protein = {a.name: fractions[a.name] for a in arms if a.modality == "protein"}
    if not text or not protein:
        raise ValueError("the sweep needs at least one arm of each modality")
    worst_text_name = min(text, key=lambda k: text[k])
    best_protein_name = max(protein, key=lambda k: protein[k])
    worst_text = text[worst_text_name]
    best_protein = protein[best_protein_name]
    ratio = worst_text / best_protein if best_protein > 0.0 else None
    return {
        "threshold": label,
        "cuts": {name: float(value) for name, value in sorted(cuts.items())},
        "counts": counts,
        "fractions": fractions,
        "text_range": [min(text.values()), max(text.values())],
        "protein_range": [min(protein.values()), max(protein.values())],
        "worst_text": {"arm": worst_text_name, "fraction": worst_text},
        "best_protein": {"arm": best_protein_name, "fraction": best_protein},
        "worst_text_over_best_protein": ratio,
        "ranges_disjoint": bool(worst_text > best_protein),
        # ``ratio is None`` means the best protein arm sits at zero, which is a
        # disjoint separation with an undefined ratio, not a failed one.
        "separation_holds": bool(worst_text > best_protein),
        "mean_text_over_mean_protein": (
            float(np.mean(list(text.values())) / np.mean(list(protein.values())))
            if np.mean(list(protein.values())) > 0.0
            else None
        ),
    }


# --------------------------------------------------------------- rank / AUC


def score_quantiles(arm: ArmCensus, *, over_uniform: bool = False) -> dict[str, float]:
    """The head-score distribution without a cut-off anywhere in it."""

    values = arm.flat_over_uniform if over_uniform else arm.flat
    probabilities = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    quantiles = np.quantile(values, probabilities)
    out = {f"q{int(p * 100):03d}": float(q) for p, q in zip(probabilities, quantiles)}
    out["mean"] = float(values.mean())
    out["sd"] = float(values.std(ddof=1))
    out["n_heads"] = int(values.size)
    return out


def auc(higher: np.ndarray, lower: np.ndarray) -> float:
    """P(X > Y) + 0.5 P(X = Y) for X drawn from ``higher``, Y from ``lower``.

    This is the Mann-Whitney U statistic normalised by the product of the sample
    sizes, i.e. the common-language effect size.  It needs no threshold and is
    unchanged by any strictly increasing transform of the scores, so it cannot be
    an artefact of either the 0.10 cut-off or the choice to divide by the uniform
    baseline.  0.5 is no separation; 1.0 is complete separation.
    """

    if higher.size < 1 or lower.size < 1:
        raise ValueError("AUC needs a non-empty sample on each side")
    statistic = stats.mannwhitneyu(higher, lower, alternative="two-sided")
    return float(statistic.statistic) / float(higher.size * lower.size)


def pairwise_auc(arms: Sequence[ArmCensus], *, over_uniform: bool = False) -> dict[str, Any]:
    """Every text-arm-against-protein-arm AUC, plus the pooled comparison.

    Head-level p-values are deliberately NOT reported.  Heads within one model
    are not independent draws and are not a sample from anything; a Mann-Whitney
    p-value over 720 against 720 heads would be a statement about an effective
    sample size the design does not have.  The AUC is reported as a descriptive
    effect size and the inference is done at the model level in
    :func:`model_level_exact_test`.
    """

    text = [a for a in arms if a.modality == "text"]
    protein = [a for a in arms if a.modality == "protein"]
    if not text or not protein:
        raise ValueError("the AUC comparison needs at least one arm of each modality")

    def values(arm: ArmCensus) -> np.ndarray:
        return arm.flat_over_uniform if over_uniform else arm.flat

    pairs: list[dict[str, Any]] = []
    for t, p in itertools.product(text, protein):
        pairs.append(
            {
                "text_arm": t.name,
                "protein_arm": p.name,
                "auc": auc(values(t), values(p)),
                "n_text_heads": t.n_heads,
                "n_protein_heads": p.n_heads,
            }
        )
    pooled_text = np.concatenate([values(a) for a in text])
    pooled_protein = np.concatenate([values(a) for a in protein])
    aucs = [row["auc"] for row in pairs]
    return {
        "scale": "score_over_uniform_baseline" if over_uniform else "raw_attention_score",
        "pairs": pairs,
        "n_pairs": len(pairs),
        "pairs_above_half": int(sum(1 for value in aucs if value > 0.5)),
        "min_pairwise_auc": float(min(aucs)),
        "max_pairwise_auc": float(max(aucs)),
        "median_pairwise_auc": float(np.median(aucs)),
        "pooled_auc": auc(pooled_text, pooled_protein),
        "pooled_note": (
            "pooling heads across arms weights each arm by its head count, so the "
            "pooled figure is reported for completeness and the pairwise minimum "
            "is the statistic to read: it is the worst case over all arm pairs"
        ),
    }


def survival_dominance(
    arms: Sequence[ArmCensus],
    *,
    n_grid: int = 2001,
    over_uniform: bool = True,
) -> dict[str, Any]:
    """Where on the threshold axis do the two modalities' fractions separate?

    The fraction above a cut-off is the survival function of the head-score
    distribution evaluated at that cut-off, so sweeping the cut-off over a fine
    grid and asking where ``min over text arms`` exceeds ``max over protein
    arms`` turns "does the finding survive the threshold choice" from an
    assertion at four points into a measured interval.  This is the honest form
    of threshold invariance: not "the number does not move" -- it moves a great
    deal -- but "the ORDERING holds for every cut-off in this range and fails
    outside it".

    The grid runs over the pooled score range, so it needs no cut-off of its own.
    """

    text = [a for a in arms if a.modality == "text"]
    protein = [a for a in arms if a.modality == "protein"]
    if not text or not protein:
        raise ValueError("survival dominance needs at least one arm of each modality")

    def values(arm: ArmCensus) -> np.ndarray:
        return arm.flat_over_uniform if over_uniform else arm.flat

    pooled = np.concatenate([values(a) for a in arms])
    grid = np.linspace(float(pooled.min()), float(pooled.max()), n_grid)
    min_text = np.empty(n_grid)
    max_protein = np.empty(n_grid)
    for index, cut in enumerate(grid):
        min_text[index] = min(float((values(a) >= cut).mean()) for a in text)
        max_protein[index] = max(float((values(a) >= cut).mean()) for a in protein)
    holds = min_text > max_protein
    # Only the region where at least one text arm still has a head above the cut
    # is informative; past the largest text score both sides are zero and the
    # comparison is vacuous rather than failed.
    informative = min_text > 0.0
    segments = _true_segments(holds & informative, grid)
    return {
        "scale": "score_over_uniform_baseline" if over_uniform else "raw_attention_score",
        "grid_min": float(grid[0]),
        "grid_max": float(grid[-1]),
        "n_grid": int(n_grid),
        "fraction_of_informative_grid_where_separation_holds": (
            float((holds & informative).sum()) / float(max(int(informative.sum()), 1))
        ),
        "n_informative_grid_points": int(informative.sum()),
        "separating_intervals": segments,
        "widest_separating_interval": (
            max(segments, key=lambda s: s["high"] - s["low"]) if segments else None
        ),
        "largest_ratio": _largest_ratio(grid, min_text, max_protein, informative),
        "note": (
            "cuts are in multiples of the probe's uniform-attention baseline; the "
            "headline 0.10 raw cut is 9.35x uniform on the synthetic probe for "
            "every arm but ZymCTRL (10.18x)"
        ),
    }


def _true_segments(mask: np.ndarray, grid: np.ndarray) -> list[dict[str, float]]:
    segments: list[dict[str, float]] = []
    start: int | None = None
    for index, flag in enumerate(mask):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            segments.append({"low": float(grid[start]), "high": float(grid[index - 1])})
            start = None
    if start is not None:
        segments.append({"low": float(grid[start]), "high": float(grid[-1])})
    return segments


def _largest_ratio(
    grid: np.ndarray,
    min_text: np.ndarray,
    max_protein: np.ndarray,
    informative: np.ndarray,
) -> dict[str, Any]:
    ratio = np.full(grid.shape, np.nan)
    usable = informative & (max_protein > 0.0)
    ratio[usable] = min_text[usable] / max_protein[usable]
    if not np.isfinite(ratio).any():
        return {"defined": False, "reason": "every protein arm is at zero wherever text is not"}
    index = int(np.nanargmax(ratio))
    return {
        "defined": True,
        "cut_over_uniform": float(grid[index]),
        "worst_text_fraction": float(min_text[index]),
        "best_protein_fraction": float(max_protein[index]),
        "ratio": float(ratio[index]),
    }


def quantile_dominance(
    arms: Sequence[ArmCensus],
    *,
    probabilities: Sequence[float] = (0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.995, 1.0),
    over_uniform: bool = True,
) -> dict[str, Any]:
    """The dual view: at which quantile of its own head distribution does an arm's
    modality show?

    Reading the distributions by quantile rather than by cut-off answers the
    question the fraction is a proxy for -- how far into the tail does one have to
    go before text and protein look different -- and it does so without ever
    choosing a score.  A row where the lowest text value exceeds the highest
    protein value is a complete separation of the two modalities at that
    quantile.
    """

    text = [a for a in arms if a.modality == "text"]
    protein = [a for a in arms if a.modality == "protein"]

    def values(arm: ArmCensus) -> np.ndarray:
        return arm.flat_over_uniform if over_uniform else arm.flat

    rows: list[dict[str, Any]] = []
    for probability in probabilities:
        text_values = {a.name: float(np.quantile(values(a), probability)) for a in text}
        protein_values = {a.name: float(np.quantile(values(a), probability)) for a in protein}
        worst_text = min(text_values.values())
        best_protein = max(protein_values.values())
        rows.append(
            {
                "quantile": float(probability),
                "text": text_values,
                "protein": protein_values,
                "worst_text": worst_text,
                "best_protein": best_protein,
                "ratio": worst_text / best_protein if best_protein > 0.0 else None,
                "separates": bool(worst_text > best_protein),
            }
        )
    separating = [row["quantile"] for row in rows if row["separates"]]
    return {
        "scale": "score_over_uniform_baseline" if over_uniform else "raw_attention_score",
        "rows": rows,
        "quantiles_that_separate": separating,
        "quantiles_that_do_not": [row["quantile"] for row in rows if not row["separates"]],
    }


def one_sided_ks(higher: np.ndarray, lower: np.ndarray) -> dict[str, Any]:
    """``sup_t [S_higher(t) - S_lower(t)]``: the largest prevalence gap, over all t.

    Threshold-free because it maximises over the threshold rather than fixing
    one, and it reports the cut at which the maximum is attained -- which is the
    quantity a fixed cut-off silently assumes it knows.  This is the tail-aware
    counterpart to :func:`auc`: the AUC integrates the whole distribution and is
    therefore dominated by the bulk of near-chance heads, while this statistic
    reads the single point of greatest separation.
    """

    if higher.size < 1 or lower.size < 1:
        raise ValueError("the KS statistic needs a non-empty sample on each side")
    cuts = np.unique(np.concatenate([higher, lower]))
    survival_high = np.array([(higher >= cut).mean() for cut in cuts])
    survival_low = np.array([(lower >= cut).mean() for cut in cuts])
    difference = survival_high - survival_low
    index = int(np.argmax(difference))
    reverse = int(np.argmin(difference))
    return {
        "d_plus": float(difference[index]),
        "cut_at_d_plus": float(cuts[index]),
        "survival_higher_at_cut": float(survival_high[index]),
        "survival_lower_at_cut": float(survival_low[index]),
        "d_minus": float(-difference[reverse]),
        "cut_at_d_minus": float(cuts[reverse]),
        "signed_dominance": (
            "higher_dominates"
            if difference.min() >= -1e-12
            else "curves_cross"
        ),
    }


def pairwise_ks(arms: Sequence[ArmCensus], *, over_uniform: bool = True) -> dict[str, Any]:
    """Every text-against-protein one-sided KS statistic."""

    text = [a for a in arms if a.modality == "text"]
    protein = [a for a in arms if a.modality == "protein"]

    def values(arm: ArmCensus) -> np.ndarray:
        return arm.flat_over_uniform if over_uniform else arm.flat

    pairs: list[dict[str, Any]] = []
    for t, p in itertools.product(text, protein):
        entry = one_sided_ks(values(t), values(p))
        entry["text_arm"] = t.name
        entry["protein_arm"] = p.name
        pairs.append(entry)
    return {
        "scale": "score_over_uniform_baseline" if over_uniform else "raw_attention_score",
        "pairs": pairs,
        "min_d_plus": float(min(row["d_plus"] for row in pairs)),
        "max_d_plus": float(max(row["d_plus"] for row in pairs)),
        "pairs_with_positive_d_plus": int(sum(1 for row in pairs if row["d_plus"] > 0.0)),
        "pairs_where_text_stochastically_dominates": int(
            sum(1 for row in pairs if row["signed_dominance"] == "higher_dominates")
        ),
        "n_pairs": len(pairs),
    }


def model_level_exact_test(
    arms: Sequence[ArmCensus],
    *,
    statistic: str,
    values: Mapping[str, float],
) -> dict[str, Any]:
    """Exact permutation test of the modality contrast with the MODEL as the unit.

    Four text arms against three protein arms gives C(7, 3) = 35 assignments, so
    the smallest attainable one-sided p-value is 1/35 = 0.0286 and there is no
    design under which this panel can produce a smaller one.  That floor is
    reported with the result, because a p-value at its own floor carries no more
    evidence than "the separation is complete" and should not be read as if it
    did.
    """

    text = [a.name for a in arms if a.modality == "text"]
    protein = [a.name for a in arms if a.modality == "protein"]
    missing = [name for name in text + protein if name not in values]
    if missing:
        raise KeyError(f"no {statistic} for {sorted(missing)}")
    observed = float(np.mean([values[n] for n in text]) - np.mean([values[n] for n in protein]))
    names = text + protein
    pool = np.array([values[n] for n in names], dtype=np.float64)
    n_text = len(text)
    at_least_as_extreme = 0
    total = 0
    for combination in itertools.combinations(range(len(names)), n_text):
        mask = np.zeros(len(names), dtype=bool)
        mask[list(combination)] = True
        difference = pool[mask].mean() - pool[~mask].mean()
        total += 1
        if difference >= observed - 1e-12:
            at_least_as_extreme += 1
    return {
        "statistic": statistic,
        "unit": "model",
        "n_text_models": len(text),
        "n_protein_models": len(protein),
        "text_values": {n: values[n] for n in text},
        "protein_values": {n: values[n] for n in protein},
        "observed_difference_in_means": observed,
        "n_assignments": total,
        "p_one_sided": at_least_as_extreme / total,
        "smallest_attainable_p": 1.0 / total,
        "at_floor": at_least_as_extreme == 1,
        "complete_separation": bool(
            min(values[n] for n in text) > max(values[n] for n in protein)
        ),
        "note": (
            "the model is the sampling unit for a cross-modality claim; heads "
            "within a model are a population and carry no sampling uncertainty "
            "of their own, so no head-level interval is reported"
        ),
    }


# ------------------------------------------------------------- scale vs modality


def _ols(design: np.ndarray, response: np.ndarray, confidence: float = 0.95) -> dict[str, Any]:
    n, k = design.shape
    dof = n - k
    if dof < 1:
        raise ValueError(f"a {k}-parameter fit on {n} points leaves no residual degrees of freedom")
    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    residual = response - design @ coefficients
    sigma_squared = float(residual @ residual) / dof
    gram_inverse = np.linalg.pinv(design.T @ design)
    variance = sigma_squared * np.diag(gram_inverse)
    standard_error = np.sqrt(np.maximum(variance, 0.0))
    critical = float(stats.t.ppf(0.5 + confidence / 2.0, dof))
    return {
        "coefficients": coefficients,
        "standard_error": standard_error,
        "critical": critical,
        "dof": dof,
        "condition_number": float(np.linalg.cond(design)),
        "residual_sd": math.sqrt(sigma_squared),
    }


def scale_modality_fit(
    arms: Sequence[ArmCensus],
    *,
    threshold: float,
    covariate: str,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """``fraction ~ intercept + modality + scale``, one scale covariate at a time.

    Every fit carries ``inferential = False``.  With seven points, one indicator
    and one covariate there are four residual degrees of freedom, the panel is
    not a random sample of decoders, and the arms were chosen partly for the
    contrast being tested.  The interval is a description of how much the seven
    points constrain a line, not a confidence statement about decoders in
    general.

    ``collinearity`` decides whether the coefficient may be read at all.  If the
    modality indicator and the scale covariate are nearly the same vector in this
    panel, then no amount of arithmetic separates their contributions and the
    honest output is that the fit cannot answer the question -- which is reported
    as the verdict, in place of a coefficient that would look clean only because
    the design is degenerate.
    """

    if covariate not in SCALE_COVARIATES:
        raise ValueError(f"unknown scale covariate {covariate!r}")
    names = [arm.name for arm in arms]
    response = np.array([arm.fraction_above(threshold) for arm in arms], dtype=np.float64)
    modality = np.array([1.0 if arm.modality == "protein" else 0.0 for arm in arms])
    scale = np.array([covariate_value(arm, covariate) for arm in arms], dtype=np.float64)
    design = np.column_stack([np.ones_like(scale), modality, scale])
    fit = _ols(design, response, confidence=confidence)
    coefficients = fit["coefficients"]
    standard_error = fit["standard_error"]
    critical = fit["critical"]
    labels = ("intercept", "protein_offset", "scale_slope")
    collinearity = collinearity_report(modality, scale)
    out: dict[str, Any] = {
        "covariate": covariate,
        "threshold": float(threshold),
        "arms": names,
        "n": len(names),
        "residual_dof": fit["dof"],
        "design_condition_number": fit["condition_number"],
        "residual_sd": fit["residual_sd"],
        "inferential": False,
        "inferential_note": (
            "n=7, arms selected for the contrast under test, not a random sample "
            "of decoders; the interval describes how tightly these seven points "
            "pin a line and is not a confidence statement about decoders at large"
        ),
        "collinearity": collinearity,
        "readable": collinearity["separable"],
        "coefficients": {},
    }
    for index, label in enumerate(labels):
        estimate = float(coefficients[index])
        error = float(standard_error[index])
        out["coefficients"][label] = {
            "estimate": estimate,
            "standard_error": error,
            "interval": [estimate - critical * error, estimate + critical * error],
        }
    if not collinearity["separable"]:
        out["verdict"] = "cannot_separate"
        out["verdict_reason"] = (
            f"the modality indicator and {covariate} correlate at "
            f"{collinearity['correlation']:+.4f} (VIF {collinearity['vif']:.1f}); "
            "this panel cannot attribute the response to one rather than the "
            "other, and the coefficient above must not be read as a modality effect"
        )
        return out
    interval = out["coefficients"]["protein_offset"]["interval"]
    out["verdict"] = (
        "modality_offset_excludes_zero" if interval[1] < 0.0 or interval[0] > 0.0 else "inconclusive"
    )
    out["verdict_reason"] = (
        f"the modality coefficient's descriptive interval is [{interval[0]:.5f}, "
        f"{interval[1]:.5f}] with {covariate} in the design"
    )
    return out


def covariate_value(arm: ArmCensus, covariate: str) -> float:
    if covariate == "log10_parameters":
        return math.log10(float(arm.parameters))
    if covariate == "n_layer":
        return float(arm.n_layer)
    if covariate == "n_head":
        return float(arm.n_heads)
    if covariate == "d_model":
        return float(arm.d_model)
    raise ValueError(f"unknown scale covariate {covariate!r}")


def collinearity_report(modality: np.ndarray, scale: np.ndarray) -> dict[str, Any]:
    """How much of the modality indicator is already in the scale covariate.

    A point-biserial correlation, which for a 0/1 indicator against a continuous
    covariate is just Pearson's r, and the variance inflation factor it implies.
    Both are reported because the VIF is the quantity with a conventional
    threshold and the correlation is the quantity with a sign.
    """

    if modality.std() == 0.0 or scale.std() == 0.0:
        raise ValueError("a constant regressor cannot enter a collinearity check")
    correlation = float(np.corrcoef(modality, scale)[0, 1])
    vif = 1.0 / max(1.0 - correlation**2, 1e-12)
    return {
        "correlation": correlation,
        "abs_correlation": abs(correlation),
        "vif": vif,
        "limit": COLLINEARITY_LIMIT,
        "separable": abs(correlation) < COLLINEARITY_LIMIT,
    }


def collinearity_verdict(fits: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Roll the per-covariate fits up into one statement about the panel."""

    unreadable = sorted(name for name, fit in fits.items() if not fit["readable"])
    readable = sorted(name for name, fit in fits.items() if fit["readable"])
    return {
        "covariates_where_modality_is_separable": readable,
        "covariates_where_modality_is_not_separable": unreadable,
        "any_separable": bool(readable),
        "verdict": "cannot_separate" if not readable else "partially_separable",
    }


def scale_inversion_checks(arms: Sequence[ArmCensus], *, threshold: float) -> dict[str, Any]:
    """The two model pairs that answer the scale objection without a fit.

    A regression on seven points is a weak instrument.  Two pairs are stronger:

    *Matched pair.*  ``gpt2-large`` and ``ProtGPT2`` share architecture, depth,
    width, head count, vocabulary size and parameter count.  Any difference is
    therefore not scale.

    *Scale inversion.*  ``dialogpt-small`` is roughly a sixth of ``ProtGPT2``'s
    parameter count, a third of its depth and a fifth of its head count.  A pure
    scale account -- "bigger decoders have a lower fraction" -- predicts the
    smaller model has the HIGHER fraction only if the account is about scale and
    not modality; the two accounts make opposite predictions here, which is what
    makes the pair decisive rather than merely consistent.
    """

    index = {arm.name: arm for arm in arms}
    out: dict[str, Any] = {"threshold": float(threshold), "pairs": {}}
    for label, text_name, protein_name in (
        ("matched_scale", "gpt2-large", "protgpt2"),
        ("inverted_scale", "dialogpt-small", "protgpt2"),
        ("inverted_scale_progen", "dialogpt-small", "progen2-medium"),
    ):
        if text_name not in index or protein_name not in index:
            out["pairs"][label] = {"available": False}
            continue
        text = index[text_name]
        protein = index[protein_name]
        text_fraction = text.fraction_above(threshold)
        protein_fraction = protein.fraction_above(threshold)
        out["pairs"][label] = {
            "available": True,
            "text_arm": text_name,
            "protein_arm": protein_name,
            "text_fraction": text_fraction,
            "protein_fraction": protein_fraction,
            "ratio": text_fraction / protein_fraction if protein_fraction > 0.0 else None,
            "modality_ordering_holds": bool(text_fraction > protein_fraction),
            "text_parameters": text.parameters,
            "protein_parameters": protein.parameters,
            "parameter_ratio_text_over_protein": text.parameters / protein.parameters,
            "text_n_layer": text.n_layer,
            "protein_n_layer": protein.n_layer,
            "text_n_heads": text.n_heads,
            "protein_n_heads": protein.n_heads,
            "text_d_model": text.d_model,
            "protein_d_model": protein.d_model,
            "text_auc_over_protein": auc(text.flat, protein.flat),
            "scale_direction": (
                "text_smaller" if text.parameters < protein.parameters else "text_larger_or_equal"
            ),
        }
    matched = out["pairs"].get("matched_scale", {})
    inverted = out["pairs"].get("inverted_scale", {})
    out["verdict"] = _inversion_verdict(matched, inverted)
    return out


def _inversion_verdict(matched: Mapping[str, Any], inverted: Mapping[str, Any]) -> dict[str, Any]:
    if not matched.get("available") or not inverted.get("available"):
        return {"decidable": False, "reason": "one of the two pairs is absent from this panel"}
    matched_holds = bool(matched["modality_ordering_holds"])
    inverted_holds = bool(inverted["modality_ordering_holds"])
    inverted_is_smaller = inverted["parameter_ratio_text_over_protein"] < 1.0
    if matched_holds and inverted_holds and inverted_is_smaller:
        return {
            "decidable": True,
            "verdict": "scale_alone_does_not_explain_the_ordering",
            "reason": (
                "the ordering holds in a pair matched on every scale variable, and "
                "also holds when the text arm is the SMALLER model, which is the "
                "sign a pure scale account forbids"
            ),
        }
    return {
        "decidable": True,
        "verdict": "scale_account_not_excluded",
        "reason": (
            f"matched-pair ordering holds: {matched_holds}; inverted-pair ordering "
            f"holds: {inverted_holds}; inverted pair really is smaller on the text "
            f"side: {inverted_is_smaller}"
        ),
    }


# --------------------------------------------------- within-lineage scale ladder

#: The GPT-2 lineage in ascending size.  Architecture, the 50257-piece BPE and
#: the WebText corpus are held fixed across all four, so whatever slope the
#: induction fraction has against scale here is measured with lineage controlled
#: and *inside* the text side -- it does not borrow identification from the
#: modality contrast it is meant to adjudicate.
GPT2_LINEAGE: tuple[str, ...] = ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")


def lineage_scale_ladder(
    arms: Sequence[ArmCensus],
    *,
    threshold: float,
    lineage: Sequence[str] = GPT2_LINEAGE,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Fraction against scale within one lineage, and what it predicts elsewhere.

    Fitted in logs on both axes because a fraction is positive and varies
    multiplicatively; the slope is then an elasticity, "per decade of
    parameters", which is the form in which it can be carried to another model.

    **Every scale variable is collinear inside a lineage.** GPT-2's four rungs
    move together on depth (12/24/36/48), width (768/1024/1280/1600), head count
    (144/384/720/1200) and parameters, so this ladder identifies *scale as a
    bundle* and cannot say which component drives the slope.  That is a real
    limitation and is recorded in the output rather than left implicit -- but it
    does not weaken the use made of the ladder here, which is only to produce a
    scale-matched expectation for a model of a stated size and shape.

    The prediction interval, not the confidence interval, is the one reported:
    the question is what a *new* decoder of that size would score, not where the
    fitted line's mean lies.  With two residual degrees of freedom it is wide,
    and it is meant to be.
    """

    index = {arm.name: arm for arm in arms}
    present = [name for name in lineage if name in index]
    if len(present) < 3:
        raise ValueError(
            f"a scale ladder needs at least three rungs; found {present} of {list(lineage)}"
        )
    rungs = [index[name] for name in present]
    fractions = np.array([arm.fraction_above(threshold) for arm in rungs])
    if (fractions <= 0.0).any():
        raise ValueError("a log-log ladder fit needs every rung above the threshold")
    scale = np.array([math.log10(float(arm.parameters)) for arm in rungs])
    response = np.log10(fractions)
    design = np.column_stack([np.ones_like(scale), scale])
    fit = _ols(design, response, confidence=confidence)
    intercept, slope = (float(v) for v in fit["coefficients"])
    critical = fit["critical"]
    sigma = fit["residual_sd"]
    gram_inverse = np.linalg.pinv(design.T @ design)

    def predict(parameters: int) -> dict[str, float]:
        x = math.log10(float(parameters))
        row = np.array([1.0, x])
        mean = intercept + slope * x
        leverage = float(row @ gram_inverse @ row)
        se_mean = sigma * math.sqrt(leverage)
        se_new = sigma * math.sqrt(1.0 + leverage)
        return {
            "log10_parameters": x,
            "predicted_fraction": 10.0**mean,
            "confidence_interval": [
                10.0 ** (mean - critical * se_mean),
                10.0 ** (mean + critical * se_mean),
            ],
            "prediction_interval": [
                10.0 ** (mean - critical * se_new),
                10.0 ** (mean + critical * se_new),
            ],
        }

    predictions: dict[str, Any] = {}
    for arm in arms:
        if arm.name in present:
            continue
        prediction = predict(arm.parameters)
        observed = arm.fraction_above(threshold)
        prediction.update(
            {
                "modality": arm.modality,
                "observed_fraction": observed,
                "observed_count": int((arm.flat >= threshold).sum()),
                "n_heads": arm.n_heads,
                "predicted_count": prediction["predicted_fraction"] * arm.n_heads,
                "shortfall_ratio": (
                    prediction["predicted_fraction"] / observed if observed > 0.0 else None
                ),
                "observed_below_prediction_interval": bool(
                    observed < prediction["prediction_interval"][0]
                ),
                "observed_inside_prediction_interval": bool(
                    prediction["prediction_interval"][0]
                    <= observed
                    <= prediction["prediction_interval"][1]
                ),
            }
        )
        predictions[arm.name] = prediction

    return {
        "lineage": present,
        "threshold": float(threshold),
        "rungs": [
            {
                "arm": arm.name,
                "parameters": arm.parameters,
                "n_layer": arm.n_layer,
                "d_model": arm.d_model,
                "n_heads": arm.n_heads,
                "count_above_threshold": int((arm.flat >= threshold).sum()),
                "fraction": float(value),
            }
            for arm, value in zip(rungs, fractions)
        ],
        "log10_slope_per_decade": slope,
        "slope_interval": [
            slope - critical * float(fit["standard_error"][1]),
            slope + critical * float(fit["standard_error"][1]),
        ],
        "intercept": intercept,
        "residual_dof": fit["dof"],
        "residual_sd_log10": sigma,
        "fraction_ratio_smallest_over_largest": float(fractions[0] / fractions[-1]),
        "count_ratio_largest_over_smallest": float(
            (fractions[-1] * rungs[-1].n_heads) / (fractions[0] * rungs[0].n_heads)
        ),
        "direction": (
            "falls_with_scale" if slope < 0.0 else "flat_or_rises_with_scale"
        ),
        "verdict": (
            "a scale account is NOT refuted on the text side: the fraction falls "
            "with scale within one lineage, so the protein deficit must be read "
            "against a scale-matched expectation rather than against a text mean"
            if slope < 0.0
            else "a scale account is refuted on the text side alone: the fraction "
            "does not fall with scale within one lineage"
        ),
        "collinearity_warning": (
            "depth, width, head count and parameter count are collinear across a "
            "single lineage, so this slope identifies scale as a bundle and "
            "attributes it to no single component"
        ),
        "predictions": predictions,
    }


def corpus_contrast(
    arms: Sequence[ArmCensus],
    *,
    threshold: float,
    pairs: Mapping[str, tuple[str, str]],
) -> dict[str, Any]:
    """How much does the fraction move when ONLY the pretraining corpus changes?

    Two pairs hold architecture, parameter count and tokeniser fixed and vary the
    corpus: ``gpt2``/``dialogpt-small`` on the text side and
    ``progen2-base``/``progen2-medium`` on the protein side.  They bound how much
    of any cross-modality difference is corpus rather than modality, which is a
    quantity the seven-arm table cannot supply and which turns out to be large
    enough that it must be quoted beside any modality ratio.
    """

    index = {arm.name: arm for arm in arms}
    out: dict[str, Any] = {"threshold": float(threshold), "pairs": {}}
    for label, (first, second) in pairs.items():
        if first not in index or second not in index:
            out["pairs"][label] = {"available": False, "missing": [
                n for n in (first, second) if n not in index
            ]}
            continue
        a, b = index[first], index[second]
        high, low = (a, b) if a.fraction_above(threshold) >= b.fraction_above(threshold) else (b, a)
        low_fraction = low.fraction_above(threshold)
        out["pairs"][label] = {
            "available": True,
            "modality": a.modality,
            "higher": {"arm": high.name, "fraction": high.fraction_above(threshold)},
            "lower": {"arm": low.name, "fraction": low_fraction},
            "ratio": (
                high.fraction_above(threshold) / low_fraction if low_fraction > 0.0 else None
            ),
            "parameters_match": a.parameters == b.parameters,
            "parameter_ratio": a.parameters / b.parameters,
            "shape_match": (a.n_layer, a.d_model, a.n_heads) == (b.n_layer, b.d_model, b.n_heads),
        }
    return out


# ------------------------------------------- corpus, lineage, variance shares

#: Architecture lineage of each panel arm.  This is deliberately NOT the modality
#: label: the GPT-2 lineage spans both modalities (five text rungs, plus ProtGPT2
#: and ZymCTRL), which is the only reason a modality term can be separated from a
#: lineage term at all in this panel -- and, as :func:`variance_decomposition`
#: records, it is separated by ProtGPT2 alone once ZymCTRL's zero is dropped.
ARM_LINEAGE = {
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2",
    "gpt2-large": "gpt2",
    "gpt2-xl": "gpt2",
    "dialogpt-small": "gpt2",
    "protgpt2": "gpt2",
    "zymctrl": "gpt2",
    "qwen2.5-0.5b": "qwen2",
    "llama-3.2-3b": "llama",
    "progen2-base": "progen",
    "progen2-medium": "progen",
}


def variance_decomposition(
    arms: Sequence[ArmCensus],
    *,
    threshold: float,
    lineage: Mapping[str, str] = ARM_LINEAGE,
) -> dict[str, Any]:
    """What share of the between-arm spread is modality, and what is everything else?

    Fitted on ``log10(fraction)``, because the arms span an order of magnitude and
    an additive decomposition of a quantity that varies multiplicatively would
    attribute most of the variance to whichever arm happens to be largest.  Arms
    at exactly zero have no logarithm and are dropped, which is recorded: ZymCTRL
    is the arm with the largest deficit, so dropping it makes every modality share
    below an UNDERSTATEMENT, not an overstatement.

    Reported as a commonality analysis rather than a single sequential table,
    because sequential sums of squares depend on the order the terms are entered
    and the order is exactly what is in dispute.  The two increments that matter
    are modality given scale and lineage, and lineage given scale and modality.

    **Identification warning, and it is the main output.** Modality is separated
    from lineage only by arms whose lineage spans both modalities.  In this panel
    that is the GPT-2 lineage alone, and once the zero-fraction arm is dropped it
    is a single protein arm against five text ones.  The reported modality share
    therefore rests on ProtGPT2, and the function computes and returns how many
    arms of each modality actually carry the contrast rather than leaving the
    reader to work it out.
    """

    usable = [arm for arm in arms if arm.fraction_above(threshold) > 0.0]
    dropped = [arm.name for arm in arms if arm.fraction_above(threshold) <= 0.0]
    if len(usable) < 6:
        raise ValueError("a variance decomposition needs at least six arms above the threshold")
    names = [arm.name for arm in usable]
    missing = [name for name in names if name not in lineage]
    if missing:
        raise KeyError(f"no lineage declared for {sorted(missing)}")

    response = np.log10(np.array([arm.fraction_above(threshold) for arm in usable]))
    scale = np.array([math.log10(float(arm.parameters)) for arm in usable])
    modality = np.array([1.0 if arm.modality == "protein" else 0.0 for arm in usable])
    families = sorted({lineage[name] for name in names})
    reference = families[0]
    lineage_columns = np.column_stack(
        [
            np.array([1.0 if lineage[name] == family else 0.0 for name in names])
            for family in families[1:]
        ]
    ) if len(families) > 1 else np.zeros((len(names), 0))

    total_variance = float(response.var(ddof=1))
    intercept = np.ones((len(names), 1))

    def r_squared(*blocks: np.ndarray) -> dict[str, Any]:
        design = np.column_stack([intercept, *[b for b in blocks if b.size]])
        coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
        residual = response - design @ coefficients
        rss = float(residual @ residual)
        tss = float(((response - response.mean()) ** 2).sum())
        return {
            "r_squared": 1.0 - rss / tss if tss > 0 else float("nan"),
            "n_parameters": int(design.shape[1]),
            "residual_dof": int(design.shape[0] - np.linalg.matrix_rank(design)),
        }

    scale_block = scale.reshape(-1, 1)
    modality_block = modality.reshape(-1, 1)
    models = {
        "scale": r_squared(scale_block),
        "scale_modality": r_squared(scale_block, modality_block),
        "scale_lineage": r_squared(scale_block, lineage_columns),
        "scale_modality_lineage": r_squared(scale_block, modality_block, lineage_columns),
        "modality_only": r_squared(modality_block),
        "lineage_only": r_squared(lineage_columns),
    }

    # How much of the modality indicator survives being projected off the scale
    # and lineage columns: if almost none does, the increment below is a number
    # about rounding error rather than about modality.
    nuisance = np.column_stack([intercept, scale_block, lineage_columns])
    projection, *_ = np.linalg.lstsq(nuisance, modality, rcond=None)
    modality_residual = modality - nuisance @ projection
    residual_share = float(
        (modality_residual @ modality_residual) / max(float(modality @ modality), 1e-12)
    )

    spanning = {
        family: {
            "text": sum(
                1 for a in usable if lineage[a.name] == family and a.modality == "text"
            ),
            "protein": sum(
                1 for a in usable if lineage[a.name] == family and a.modality == "protein"
            ),
        }
        for family in families
    }
    spanning_families = [
        family for family, counts in spanning.items() if counts["text"] and counts["protein"]
    ]
    carriers = [
        a.name
        for a in usable
        if lineage[a.name] in spanning_families and a.modality == "protein"
    ]

    return {
        "threshold": float(threshold),
        "response": "log10_fraction_above_threshold",
        "n_arms": len(usable),
        "arms": names,
        "dropped_at_zero": dropped,
        "dropped_note": (
            "an arm at exactly zero has no logarithm; the dropped arms are the "
            "largest deficits in the panel, so every modality share reported here "
            "is an understatement rather than an overstatement"
        ),
        "lineage_families": families,
        "lineage_reference_level": reference,
        "total_variance_log10": total_variance,
        "models": models,
        "increment_modality_given_scale_and_lineage": (
            models["scale_modality_lineage"]["r_squared"] - models["scale_lineage"]["r_squared"]
        ),
        "increment_lineage_given_scale_and_modality": (
            models["scale_modality_lineage"]["r_squared"] - models["scale_modality"]["r_squared"]
        ),
        "increment_scale_given_modality_and_lineage": (
            models["scale_modality_lineage"]["r_squared"]
            - r_squared(modality_block, lineage_columns)["r_squared"]
        ),
        "modality_variance_surviving_scale_and_lineage": residual_share,
        "identification": {
            "families_spanning_both_modalities": spanning_families,
            "arms_per_family": spanning,
            "protein_arms_carrying_the_within_lineage_contrast": carriers,
            "note": (
                "modality is separated from lineage only inside a family that "
                "contains both modalities; every other family's modality label is "
                "its lineage label under another name"
            ),
        },
        "inferential": False,
        "corpus_repeat_caveat": (
            "none of this addresses the corpus-repeat confound. Approximate "
            "repeats occur in 32.3 per cent of text documents against 0.402 per "
            "cent of protein entries, and an induction head is only useful on a "
            "corpus that repeats, so a surviving modality term is equally "
            "consistent with efficient allocation against the data. No protein "
            "decoder trained on a repeat-rich corpus exists, so this cannot be "
            "separated observationally by any arrangement of these arms"
        ),
    }


def contrast_ratio_bootstrap(
    per_probe_high: np.ndarray,
    per_probe_low: np.ndarray,
    *,
    threshold: float,
    resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Ratio of two arms' head fractions, with probe clusters resampled in each.

    The two arms' probes are resampled independently rather than in pairs,
    because a synthetic probe is built from its own arm's unigram: probe *i* of
    one arm and probe *i* of the other are different token sequences that share
    only a seed, so pairing them would assert a correspondence that does not
    exist.

    A resample in which the denominator arm has no head above the cut leaves the
    ratio undefined; those draws are counted and reported rather than dropped
    silently, because a contrast that is undefined a third of the time is not a
    contrast with a wide interval, it is a contrast that has not been measured.
    """

    if per_probe_high.ndim != 3 or per_probe_low.ndim != 3:
        raise ValueError("per-probe scores must have shape (probe, layer, head)")
    if resamples < 1:
        raise ValueError("resamples must be positive")

    def fraction(block: np.ndarray, picks: np.ndarray) -> float:
        mean = block[picks].mean(axis=0).reshape(-1)
        return float((mean >= threshold).sum()) / mean.size

    high_n = per_probe_high.shape[0]
    low_n = per_probe_low.shape[0]
    point_high = fraction(per_probe_high, np.arange(high_n))
    point_low = fraction(per_probe_low, np.arange(low_n))
    rng = np.random.default_rng(seed)
    ratios: list[float] = []
    undefined = 0
    for _ in range(resamples):
        numerator = fraction(per_probe_high, rng.integers(0, high_n, size=high_n))
        denominator = fraction(per_probe_low, rng.integers(0, low_n, size=low_n))
        if denominator <= 0.0:
            undefined += 1
            continue
        ratios.append(numerator / denominator)
    tail = (1.0 - confidence) / 2.0
    interval = (
        [float(v) for v in np.quantile(ratios, [tail, 1.0 - tail])] if ratios else None
    )
    return {
        "threshold": float(threshold),
        "fraction_high": point_high,
        "fraction_low": point_low,
        "ratio": point_high / point_low if point_low > 0.0 else None,
        "interval": interval,
        "confidence": confidence,
        "resamples": int(resamples),
        "undefined_resamples": undefined,
        "undefined_share": undefined / resamples,
        "resampling_unit": "probe",
    }


# ------------------------------------------------------------ probe bootstrap


def cluster_bootstrap_fraction(
    per_probe: np.ndarray,
    *,
    threshold: float,
    resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Resample PROBES as clusters and recompute the head fraction each time.

    ``per_probe`` has shape (probe, layer, head) and holds each probe's own
    contribution to the head score, so that the census statistic -- the mean over
    probes -- can be recomputed on a resampled probe set.  This is the only
    resampling this analysis does, and it answers exactly one question: how much
    of the arm's fraction is probe-sampling noise.

    It deliberately does NOT resample heads or layers.  A layer is a coordinate
    of the model, not a draw.  A model's heads are its entire population: there
    is no superpopulation of GPT-2-large heads to have sampled from, so an
    interval around a within-model head fraction would be answering a question
    about a hypothetical model that does not exist.  The cross-modality contrast
    takes the model as its unit and is handled by
    :func:`model_level_exact_test`.
    """

    if per_probe.ndim != 3:
        raise ValueError("per-probe scores must have shape (probe, layer, head)")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    n_probes = per_probe.shape[0]
    if n_probes < 2:
        raise ValueError("a cluster bootstrap needs at least two probes")
    n_heads = per_probe.shape[1] * per_probe.shape[2]
    point = float((per_probe.mean(axis=0).reshape(-1) >= threshold).sum()) / n_heads
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        pick = rng.integers(0, n_probes, size=n_probes)
        mean = per_probe[pick].mean(axis=0).reshape(-1)
        draws[index] = float((mean >= threshold).sum()) / n_heads
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [tail, 1.0 - tail])
    return {
        "threshold": float(threshold),
        "point_estimate": point,
        "interval": [float(low), float(high)],
        "confidence": confidence,
        "resamples": int(resamples),
        "n_probe_clusters": int(n_probes),
        "n_heads": int(n_heads),
        "bootstrap_sd": float(draws.std(ddof=1)),
        "resampling_unit": "probe",
        "note": (
            "probes are the cluster; heads and layers are not resampled because "
            "within one model they are a population and a coordinate respectively"
        ),
    }
