"""Small statistical primitives shared by confirmatory revision analyses."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy import stats


def _finite_vector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2:
        raise ValueError(f"{name} must be a one-dimensional vector with at least two values")
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
        radius = float(stats.t.ppf((1.0 + confidence) / 2.0, sample.size - 1)) * standard_error
        interval = [mean - radius, mean + radius]
    return {
        "mean": mean,
        "standard_error": standard_error,
        "confidence": float(confidence),
        "interval": interval,
        "n": int(sample.size),
    }


def tost_paired(differences: Sequence[float], margin: float, alpha: float = 0.05) -> dict:
    """Two one-sided equivalence test for paired differences.

    The equivalence region is frozen as ``[-margin, +margin]``.  The returned
    interval is the ``1 - 2*alpha`` interval corresponding to TOST; a separate
    95% interval is included for effect reporting.
    """

    sample = _finite_vector(differences, "differences")
    if not np.isfinite(margin) or margin <= 0.0:
        raise ValueError("margin must be finite and positive")
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must lie strictly between zero and one half")
    mean = float(sample.mean())
    standard_error = float(stats.sem(sample))
    if standard_error == 0.0:
        p_lower = 0.0 if mean > -margin else 1.0
        p_upper = 0.0 if mean < margin else 1.0
    else:
        degrees = sample.size - 1
        p_lower = float(stats.t.sf((mean + margin) / standard_error, degrees))
        p_upper = float(stats.t.cdf((mean - margin) / standard_error, degrees))
    tost_interval = mean_interval(sample, confidence=1.0 - 2.0 * alpha)["interval"]
    report_interval = mean_interval(sample, confidence=0.95)["interval"]
    return {
        "mean_difference": mean,
        "equivalence_band": [-float(margin), float(margin)],
        "alpha": float(alpha),
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_tost": max(p_lower, p_upper),
        "equivalent": bool(max(p_lower, p_upper) < alpha),
        "tost_interval": tost_interval,
        "ci95": report_interval,
        "n_pairs": int(sample.size),
    }


def benjamini_hochberg(pvalues: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values in original order."""

    p = np.asarray(pvalues, dtype=np.float64)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("pvalues must be a non-empty one-dimensional vector")
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("pvalues must be finite and lie in [0, 1]")
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    adjusted = ranked * p.size / np.arange(1, p.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


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

    Groups are sampled with replacement and all observations in a sampled
    group are retained.  The same sampled rows score both prediction vectors.
    """

    truth = np.asarray(y)
    left = np.asarray(left_predictions)
    right = np.asarray(right_predictions)
    group_ids = np.asarray(groups)
    if truth.ndim != 1 or left.shape != truth.shape or right.shape != truth.shape:
        raise ValueError("truth and prediction vectors must have identical one-dimensional shape")
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
        sampled_groups = rng.choice(unique_groups, size=unique_groups.size, replace=True)
        indices = np.concatenate([np.flatnonzero(group_ids == group) for group in sampled_groups])
        left_boot = float(metric(truth[indices], left[indices]))
        right_boot = float(metric(truth[indices], right[indices]))
        if not np.isfinite(left_boot) or not np.isfinite(right_boot):
            continue
        differences.append(left_boot - right_boot)
        if abs(right_boot) > 1e-12:
            ratios.append(left_boot / right_boot)
    if not differences:
        raise RuntimeError("no finite paired bootstrap draws were produced")
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
        "n_groups": int(unique_groups.size),
    }
