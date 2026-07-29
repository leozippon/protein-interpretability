"""Interval estimation and group-disjoint resampling for transfer measurements.

Three primitives, each answering a question this programme has got wrong at least
once.

:func:`mean_interval`
    A Student-t interval over a **finite** sample. The finiteness check is not
    defensive noise: a single non-finite value silently turns a mean and its
    interval into ``nan``, and ``nan`` compares false against every gate, so a
    broken arm reads as a failing arm.

:func:`paired_group_bootstrap`
    Resamples **groups**, not rows, and scores both prediction vectors on the
    same resampled rows. Research plan rule 4 -- "seeds and intervals or nothing"
    -- is only worth anything if the resampling unit is the unit of independence.
    The audit's §0.05 records the cost of getting this wrong in the other
    direction: the path-patching matched-pair interval resampled an arm's
    induction *heads*, its entire population, while the probe records that were
    the real sampling unit contributed nothing. The point estimate survived; the
    interval did not mean what it said.

:func:`make_group_splits`
    Identity-group-disjoint cross-validation folds, validated after
    construction rather than trusted. A probe trained on one member of a protein
    family and tested on another measures family membership, not the concept.

Vendored into ``src/transfer`` by EXP-R2-066 from ``src/revision/statistics.py``
and ``src/revision/nested_recoverability.py``, whose remaining contents served
the retired CLT / dictionary-qualification scope.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy import stats
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


#: How much of a requested bootstrap may be discarded for non-finiteness before
#: the percentile interval stops describing the distribution it claims to.
#:
#: A resample whose metric is not finite -- a Spearman correlation over a
#: resample with no variance, say -- is a fact about the data, not a glitch to
#: skip. Dropping it silently changes the estimand from "the bootstrap
#: distribution of this metric" to "that distribution conditioned on being
#: finite", which is narrower and biased towards well-conditioned resamples.
#: Below this fraction the two are too different to publish under one name, so
#: :func:`paired_group_bootstrap` refuses instead of returning the conditioned
#: interval under the requested draw count.
MINIMUM_FINITE_DRAW_FRACTION = 0.95


def _finite_vector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2:
        raise ValueError(
            f"{name} must be a one-dimensional vector with at least two values"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def mean_interval(values: Sequence[float], confidence: float = 0.95) -> dict:
    """Student-t interval for a finite sample mean."""

    sample = _finite_vector(values, "values")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    mean = float(sample.mean())
    standard_error = float(stats.sem(sample))
    if standard_error == 0.0:
        interval = [mean, mean]
    else:
        radius = (
            float(stats.t.ppf((1.0 + confidence) / 2.0, sample.size - 1))
            * standard_error
        )
        interval = [mean - radius, mean + radius]
    return {
        "mean": mean,
        "standard_error": standard_error,
        "confidence": float(confidence),
        "interval": interval,
        "n": int(sample.size),
    }


def paired_group_bootstrap(
    y: Sequence,
    left_predictions: Sequence,
    right_predictions: Sequence,
    groups: Sequence,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    seed: int,
    n_bootstrap: int = 1000,
) -> dict:
    """Paired group bootstrap for a metric difference and ratio.

    Groups are sampled with replacement and all observations in a sampled group
    are retained. The same sampled rows score both prediction vectors, so the
    pairing that makes the difference meaningful is preserved draw by draw.
    """

    truth = np.asarray(y)
    left = np.asarray(left_predictions)
    right = np.asarray(right_predictions)
    group_ids = np.asarray(groups)
    if truth.ndim != 1 or left.shape != truth.shape or right.shape != truth.shape:
        raise ValueError(
            "truth and prediction vectors must have identical one-dimensional shape"
        )
    if group_ids.shape != truth.shape:
        raise ValueError("groups must align with truth")
    unique_groups = np.unique(group_ids)
    if unique_groups.size < 2:
        raise ValueError("at least two groups are required")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")

    left_score = float(metric(truth, left))
    right_score = float(metric(truth, right))
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    ratios: list[float] = []
    for _ in range(n_bootstrap):
        sampled_groups = rng.choice(
            unique_groups, size=unique_groups.size, replace=True
        )
        indices = np.concatenate(
            [np.flatnonzero(group_ids == group) for group in sampled_groups]
        )
        left_boot = float(metric(truth[indices], left[indices]))
        right_boot = float(metric(truth[indices], right[indices]))
        if not np.isfinite(left_boot) or not np.isfinite(right_boot):
            continue
        differences.append(left_boot - right_boot)
        if abs(right_boot) > 1e-12:
            ratios.append(left_boot / right_boot)
    if not differences:
        raise RuntimeError("no finite paired bootstrap draws were produced")
    if len(differences) < MINIMUM_FINITE_DRAW_FRACTION * n_bootstrap:
        raise RuntimeError(
            f"only {len(differences)} of {n_bootstrap} paired bootstrap draws were "
            f"finite, below the {MINIMUM_FINITE_DRAW_FRACTION:.0%} floor; the "
            "percentile interval over what survives is conditioned on finiteness "
            "and is not the requested bootstrap distribution"
        )
    return {
        "left_score": left_score,
        "right_score": right_score,
        "difference": left_score - right_score,
        "difference_ci95": [
            float(np.percentile(differences, 2.5)),
            float(np.percentile(differences, 97.5)),
        ],
        "ratio": left_score / right_score if abs(right_score) > 1e-12 else None,
        "ratio_ci95": (
            [float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))]
            if ratios
            else None
        ),
        "n_bootstrap": int(n_bootstrap),
        # The requested count is not the count the intervals were taken over.
        # `difference_ci95` spans `n_finite_draws` values and `ratio_ci95` spans
        # `n_ratio_draws`, which is smaller again whenever a resampled right-hand
        # score lands on zero. Publishing only `n_bootstrap` let a percentile over
        # a handful of surviving draws be read as a 1000-draw interval.
        "n_bootstrap_requested": int(n_bootstrap),
        "n_finite_draws": len(differences),
        "n_non_finite_draws": int(n_bootstrap) - len(differences),
        "n_ratio_draws": len(ratios),
        "n_groups": int(unique_groups.size),
    }


def make_group_splits(
    y: Sequence,
    groups: Sequence,
    *,
    n_splits: int,
    seed: int,
    task_type: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create and validate identity-group-disjoint folds.

    The post-construction checks are the point. scikit-learn's grouped splitters
    are correct, but a caller that hands them a grouping vector derived from the
    wrong column gets folds that look fine and leak. Every fold is therefore
    re-checked for group intersection and for class coverage, and every sample is
    required to be tested exactly once.
    """

    truth = np.asarray(y)
    group_ids = np.asarray(groups)
    if n_splits < 2 or np.unique(group_ids).size < n_splits:
        raise ValueError("n_splits requires at least that many unique groups")
    if task_type == "classification":
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        iterator = splitter.split(np.zeros((truth.size, 1)), truth, group_ids)
    elif task_type == "regression":
        splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        iterator = splitter.split(np.zeros((truth.size, 1)), truth, group_ids)
    else:
        raise ValueError("task_type must be classification or regression")
    splits = [(np.asarray(train), np.asarray(test)) for train, test in iterator]
    test_counts = np.zeros(truth.size, dtype=int)
    for train, test in splits:
        if np.intersect1d(group_ids[train], group_ids[test]).size:
            raise RuntimeError("group leakage detected in a generated fold")
        if task_type == "classification" and np.unique(truth[train]).size < 2:
            raise ValueError("a training fold contains fewer than two classes")
        test_counts[test] += 1
    if not np.all(test_counts == 1):
        raise RuntimeError(
            "group fold construction did not test every sample exactly once"
        )
    return splits
