"""Confirmatory cross-model atlas construction from cached activation matrices.

Inputs are nested mappings ``model -> layer -> [sample, feature]``.  Discovery
selects feature pools and matches on cohort A, returning an immutable
``FrozenAtlas``.  ``score_atlas`` evaluates those exact identities on cohort B;
it never reselects a feature or changes a layer.

The implementation deliberately contains no model or dataset code.  Cached
NumPy matrices can therefore be reused across matcher, threshold, pool and null
sensitivity analyses without another forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from typing import Hashable, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


ModelMatrices = Mapping[str, Mapping[Hashable, np.ndarray]]
LayerGroups = Mapping[Hashable, Mapping[str, Hashable]]


@dataclass(frozen=True)
class EdgeAssignment:
    """A one-to-one assignment in a pairwise score matrix."""

    left: int
    right: int
    score: float
    ambiguity: int
    confidence: float
    transport_weight: float | None = None


@dataclass(frozen=True)
class TriangleAssignment:
    """A disjoint three-way assignment assembled from three pair scores."""

    first: int
    second: int
    third: int
    pair_scores: tuple[float, float, float]
    score: float
    ambiguity: int
    confidence: float


@dataclass(frozen=True)
class FeatureRef:
    model: str
    layer: Hashable
    feature: int


@dataclass(frozen=True)
class FrozenLayerGroup:
    """Layer choices and cohort-A feature pools, aligned to atlas model order."""

    name: Hashable
    layers: tuple[Hashable, ...]
    feature_pools: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class FrozenMatch:
    """A correspondence whose identities were selected only on cohort A."""

    group: Hashable
    features: tuple[FeatureRef, ...]
    discovery_signed_correlations: tuple[float, ...]
    discovery_matching_scores: tuple[float, ...]
    ambiguity: int
    confidence: float

    @property
    def identity(self) -> tuple[tuple[str, Hashable, int], ...]:
        return tuple((x.model, x.layer, x.feature) for x in self.features)

    @property
    def minimum_matching_score(self) -> float:
        return min(self.discovery_matching_scores)


@dataclass(frozen=True)
class FrozenAtlas:
    """Immutable cohort-A candidate pools, layer choices and matches."""

    models: tuple[str, ...]
    groups: tuple[FrozenLayerGroup, ...]
    matches: tuple[FrozenMatch, ...]
    correlation_mode: str
    matcher: str
    threshold: float
    max_matches: int | None
    ambiguity_tolerance: float
    ot_regularization: float
    joint_candidate_width: int


@dataclass(frozen=True)
class HeldoutMatch:
    """Signed cohort-B scores for one frozen cohort-A match."""

    frozen: FrozenMatch
    signed_correlations: tuple[float, ...]
    matching_scores: tuple[float, ...]
    ambiguity: int
    confidence: float
    passes_threshold: bool

    @property
    def identity(self) -> tuple[tuple[str, Hashable, int], ...]:
        return self.frozen.identity

    @property
    def mean_signed_correlation(self) -> float:
        return float(np.mean(self.signed_correlations))

    @property
    def minimum_matching_score(self) -> float:
        return min(self.matching_scores)


@dataclass(frozen=True)
class AtlasEvaluation:
    """Held-out scores and retention of exact cohort-A identities."""

    matches: tuple[HeldoutMatch, ...]
    n_passing: int
    retained_identity_jaccard: float


@dataclass(frozen=True)
class IdentityOverlap:
    n_left: int
    n_right: int
    n_intersection: int
    n_union: int
    jaccard: float


@dataclass(frozen=True)
class PermutationResult:
    """Coherent model-wise rematching null with plus-one upper-tail p-values."""

    observed_count: int
    null_counts: tuple[int, ...]
    count_pvalue: float
    observed_mean_score: float
    null_mean_scores: tuple[float, ...]
    mean_score_pvalue: float
    permutations: tuple[tuple[tuple[str, tuple[int, ...]], ...], ...] | None


def _as_matrix(value: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(value, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    if x.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two samples")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains non-finite values")
    return x


def correlation_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return finite, signed Pearson correlations between feature columns."""

    x = _as_matrix(left, "left")
    y = _as_matrix(right, "right")
    if x.shape[0] != y.shape[0]:
        raise ValueError("left and right must contain the same samples")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    denom = np.sqrt(np.sum(x * x, axis=0)[:, None] * np.sum(y * y, axis=0)[None, :])
    out = np.zeros((x.shape[1], y.shape[1]), dtype=np.float64)
    np.divide(x.T @ y, denom, out=out, where=denom > 0)
    return np.clip(out, -1.0, 1.0)


def _matching_scores(correlations: np.ndarray, mode: str) -> np.ndarray:
    if mode == "positive":
        return correlations
    if mode == "absolute":
        return np.abs(correlations)
    raise ValueError("correlation_mode must be 'positive' or 'absolute'")


def _edge_diagnostics(
    scores: np.ndarray, left: int, right: int, tolerance: float
) -> tuple[int, float]:
    selected = float(scores[left, right])
    alternatives = np.concatenate((np.delete(scores[left], right), np.delete(scores[:, right], left)))
    if alternatives.size == 0:
        return 0, 1.0
    ambiguity = int(np.count_nonzero(alternatives >= selected - tolerance))
    return ambiguity, selected - float(np.max(alternatives))


def _validate_match_args(
    scores: np.ndarray,
    min_score: float | None,
    max_matches: int | None,
    ambiguity_tolerance: float,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError("scores must be a non-empty finite matrix")
    if min_score is not None and not np.isfinite(min_score):
        raise ValueError("min_score must be finite or None")
    if max_matches is not None and max_matches < 1:
        raise ValueError("max_matches must be positive or None")
    if ambiguity_tolerance < 0 or not np.isfinite(ambiguity_tolerance):
        raise ValueError("ambiguity_tolerance must be finite and nonnegative")
    return values


def greedy_match(
    scores: np.ndarray,
    *,
    min_score: float | None = None,
    max_matches: int | None = None,
    ambiguity_tolerance: float = 0.02,
) -> tuple[EdgeAssignment, ...]:
    """Deterministic descending-score greedy one-to-one matching."""

    values = _validate_match_args(scores, min_score, max_matches, ambiguity_tolerance)
    rows, cols = np.indices(values.shape)
    order = np.lexsort((cols.ravel(), rows.ravel(), -values.ravel()))
    used_left: set[int] = set()
    used_right: set[int] = set()
    result: list[EdgeAssignment] = []
    for flat in order:
        i, j = int(rows.ravel()[flat]), int(cols.ravel()[flat])
        score = float(values[i, j])
        if min_score is not None and score < min_score:
            break
        if i in used_left or j in used_right:
            continue
        ambiguity, confidence = _edge_diagnostics(values, i, j, ambiguity_tolerance)
        result.append(EdgeAssignment(i, j, score, ambiguity, confidence))
        used_left.add(i)
        used_right.add(j)
        if len(result) == min(values.shape) or (
            max_matches is not None and len(result) == max_matches
        ):
            break
    return tuple(result)


def _partial_linear_assignment(
    objective: np.ndarray, eligible: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Maximize eligible cardinality, then the supplied objective."""

    n_left, n_right = objective.shape
    span = float(np.ptp(objective))
    bonus = (span + 1.0) * (min(n_left, n_right) + 1)
    utility = np.zeros((n_left + n_right, n_right + n_left))
    utility[:n_left, :n_right] = np.where(
        eligible, bonus + objective - float(np.min(objective)), -bonus
    )
    return linear_sum_assignment(utility, maximize=True)


def hungarian_match(
    scores: np.ndarray,
    *,
    min_score: float | None = None,
    max_matches: int | None = None,
    ambiguity_tolerance: float = 0.02,
) -> tuple[EdgeAssignment, ...]:
    """Hungarian assignment, with maximum eligible cardinality at a threshold."""

    values = _validate_match_args(scores, min_score, max_matches, ambiguity_tolerance)
    if min_score is None:
        rows, cols = linear_sum_assignment(values, maximize=True)
    else:
        # Dummy rows/columns make unmatched endpoints explicit, so an edge
        # below the reporting threshold cannot displace an eligible edge.
        rows, cols = _partial_linear_assignment(values, values >= min_score)
    pairs = sorted(
        (
            (int(i), int(j))
            for i, j in zip(rows, cols)
            if i < values.shape[0] and j < values.shape[1]
        ),
        key=lambda ij: (-values[ij], ij[0], ij[1]),
    )
    result = []
    for i, j in pairs:
        score = float(values[i, j])
        if min_score is not None and score < min_score:
            continue
        ambiguity, confidence = _edge_diagnostics(values, i, j, ambiguity_tolerance)
        result.append(EdgeAssignment(i, j, score, ambiguity, confidence))
        if max_matches is not None and len(result) == max_matches:
            break
    return tuple(result)


def entropic_ot_match(
    scores: np.ndarray,
    *,
    regularization: float = 0.05,
    max_iterations: int = 1_000,
    tolerance: float = 1e-9,
    min_score: float | None = None,
    max_matches: int | None = None,
    ambiguity_tolerance: float = 0.02,
) -> tuple[EdgeAssignment, ...]:
    """Sinkhorn transport followed by a discrete maximum-mass assignment.

    Uniform feature marginals make the transport plans directly comparable
    across matchers.  ``transport_weight`` retains the soft coupling mass.
    """

    values = _validate_match_args(scores, min_score, max_matches, ambiguity_tolerance)
    if regularization <= 0 or not np.isfinite(regularization):
        raise ValueError("regularization must be finite and positive")
    if max_iterations < 1 or tolerance <= 0:
        raise ValueError("max_iterations and tolerance must be positive")

    kernel = np.exp(np.clip((values - np.max(values)) / regularization, -700.0, 0.0))
    a = np.full(values.shape[0], 1.0 / values.shape[0])
    b = np.full(values.shape[1], 1.0 / values.shape[1])
    u = np.ones_like(a)
    v = np.ones_like(b)
    for iteration in range(max_iterations):
        u = a / (kernel @ v)
        v = b / (kernel.T @ u)
        if (iteration + 1) % 10 and iteration + 1 != max_iterations:
            continue
        candidate = (u[:, None] * kernel) * v[None, :]
        marginal_error = max(
            float(np.max(np.abs(candidate.sum(axis=1) - a))),
            float(np.max(np.abs(candidate.sum(axis=0) - b))),
        )
        if marginal_error <= tolerance:
            break
    plan = (u[:, None] * kernel) * v[None, :]
    # Deterministic transport rounding gives exact uniform marginals when a
    # sharp coupling converges slowly.  The correction mass is bounded by the
    # final marginal residual and does not change the score matrix.
    row_scale = np.minimum(1.0, a / plan.sum(axis=1))
    plan *= row_scale[:, None]
    column_scale = np.minimum(1.0, b / plan.sum(axis=0))
    plan *= column_scale[None, :]
    row_residual = np.maximum(a - plan.sum(axis=1), 0.0)
    column_residual = np.maximum(b - plan.sum(axis=0), 0.0)
    residual_mass = float(row_residual.sum())
    if residual_mass > 0:
        plan += np.outer(row_residual, column_residual) / residual_mass
    marginal_error = max(
        float(np.max(np.abs(plan.sum(axis=1) - a))),
        float(np.max(np.abs(plan.sum(axis=0) - b))),
    )
    if not np.isfinite(plan).all() or marginal_error > max(1e-7, 10 * tolerance):
        raise RuntimeError("entropic optimal transport produced invalid marginals")
    if min_score is None:
        rows, cols = linear_sum_assignment(plan, maximize=True)
    else:
        rows, cols = _partial_linear_assignment(plan, values >= min_score)
    pairs = sorted(
        (
            (int(i), int(j))
            for i, j in zip(rows, cols)
            if i < values.shape[0] and j < values.shape[1]
        ),
        key=lambda ij: (-plan[ij], -values[ij], ij[0], ij[1]),
    )
    result = []
    for i, j in pairs:
        score = float(values[i, j])
        if min_score is not None and score < min_score:
            continue
        ambiguity, confidence = _edge_diagnostics(values, i, j, ambiguity_tolerance)
        result.append(
            EdgeAssignment(i, j, score, ambiguity, confidence, float(plan[i, j]))
        )
        if max_matches is not None and len(result) == max_matches:
            break
    return tuple(result)


def _top_neighbors(values: np.ndarray, width: int, min_score: float | None) -> np.ndarray:
    order = np.lexsort((np.arange(values.size), -values))
    if min_score is not None:
        order = order[values[order] >= min_score]
    return order[:width]


def joint_triangle_match(
    scores_ab: np.ndarray,
    scores_ac: np.ndarray,
    scores_bc: np.ndarray,
    *,
    min_score: float | None = None,
    max_matches: int | None = None,
    candidate_width: int = 8,
    ambiguity_tolerance: float = 0.02,
) -> tuple[TriangleAssignment, ...]:
    """Greedily optimize disjoint triangles using all three pair-score terms.

    Candidate triangles are the union obtained by anchoring at each of the
    three models and crossing its best ``candidate_width`` neighbors in the
    other models.  This avoids cubic enumeration while allowing the closing
    edge to alter a pairwise choice.
    """

    ab = _validate_match_args(scores_ab, min_score, max_matches, ambiguity_tolerance)
    ac = _validate_match_args(scores_ac, min_score, max_matches, ambiguity_tolerance)
    bc = _validate_match_args(scores_bc, min_score, max_matches, ambiguity_tolerance)
    if ab.shape[0] != ac.shape[0] or ab.shape[1] != bc.shape[0] or ac.shape[1] != bc.shape[1]:
        raise ValueError("pair-score matrices have incompatible model dimensions")
    if candidate_width < 1:
        raise ValueError("candidate_width must be positive")

    candidates: set[tuple[int, int, int]] = set()
    for a in range(ab.shape[0]):
        for b in _top_neighbors(ab[a], candidate_width, min_score):
            for c in _top_neighbors(ac[a], candidate_width, min_score):
                candidates.add((a, int(b), int(c)))
    for b in range(ab.shape[1]):
        for a in _top_neighbors(ab[:, b], candidate_width, min_score):
            for c in _top_neighbors(bc[b], candidate_width, min_score):
                candidates.add((int(a), b, int(c)))
    for c in range(ac.shape[1]):
        for a in _top_neighbors(ac[:, c], candidate_width, min_score):
            for b in _top_neighbors(bc[:, c], candidate_width, min_score):
                candidates.add((int(a), int(b), c))

    scored: list[tuple[float, float, tuple[int, int, int], tuple[float, float, float]]] = []
    for index in candidates:
        a, b, c = index
        edge_scores = (float(ab[a, b]), float(ac[a, c]), float(bc[b, c]))
        minimum = min(edge_scores)
        if min_score is not None and minimum < min_score:
            continue
        scored.append((float(np.mean(edge_scores)), minimum, index, edge_scores))
    scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
    del candidates

    # Index competitors by each endpoint.  This keeps ambiguity diagnostics
    # linear in the sparse candidate graph rather than rescanning every
    # triangle for every selected match.
    by_a: list[list[int]] = [[] for _ in range(ab.shape[0])]
    by_b: list[list[int]] = [[] for _ in range(ab.shape[1])]
    by_c: list[list[int]] = [[] for _ in range(ac.shape[1])]
    for position, candidate in enumerate(scored):
        a, b, c = candidate[2]
        by_a[a].append(position)
        by_b[b].append(position)
        by_c[c].append(position)

    selected = []
    used_a: set[int] = set()
    used_b: set[int] = set()
    used_c: set[int] = set()
    for position, (mean_score, _, (a, b, c), edge_scores) in enumerate(scored):
        if a in used_a or b in used_b or c in used_c:
            continue
        competitor_positions = set(by_a[a]) | set(by_b[b]) | set(by_c[c])
        competitor_positions.discard(position)
        competitor_scores = [scored[position][0] for position in competitor_positions]
        if competitor_scores:
            ambiguity = int(
                np.count_nonzero(np.asarray(competitor_scores) >= mean_score - ambiguity_tolerance)
            )
            confidence = mean_score - max(competitor_scores)
        else:
            ambiguity, confidence = 0, 1.0
        selected.append(
            TriangleAssignment(a, b, c, edge_scores, mean_score, ambiguity, confidence)
        )
        used_a.add(a)
        used_b.add(b)
        used_c.add(c)
        if len(selected) == min(ab.shape[0], ab.shape[1], ac.shape[1]) or (
            max_matches is not None and len(selected) == max_matches
        ):
            break
    return tuple(selected)


def _normalise_matcher(name: str) -> str:
    aliases = {
        "greedy": "greedy",
        "hungarian": "hungarian",
        "ot": "optimal_transport",
        "optimal_transport": "optimal_transport",
        "joint": "joint_triangle",
        "joint_triangle": "joint_triangle",
    }
    try:
        return aliases[name]
    except KeyError as exc:
        raise ValueError(f"unknown matcher: {name}") from exc


def _normalise_groups(
    matrices: ModelMatrices,
    models: tuple[str, ...],
    layer_groups: LayerGroups | None,
) -> tuple[tuple[Hashable, tuple[Hashable, ...]], ...]:
    if layer_groups is None:
        common = set(matrices[models[0]])
        for model in models[1:]:
            common.intersection_update(matrices[model])
        if not common:
            raise ValueError("models have no common layer key; provide layer_groups")
        return tuple((layer, (layer,) * len(models)) for layer in sorted(common, key=repr))
    result = []
    for name, mapping in layer_groups.items():
        if set(mapping) != set(models):
            raise ValueError(f"layer group {name!r} must specify every model exactly once")
        result.append((name, tuple(mapping[model] for model in models)))
    if not result:
        raise ValueError("layer_groups must not be empty")
    return tuple(result)


def _select_pool(matrix: np.ndarray, pool_size: int | None) -> tuple[int, ...]:
    n_features = matrix.shape[1]
    if pool_size is not None and pool_size < 1:
        raise ValueError("feature_pool_size must be positive or None")
    keep = n_features if pool_size is None else min(pool_size, n_features)
    variance = np.var(matrix, axis=0)
    order = np.lexsort((np.arange(n_features), -variance))
    return tuple(int(x) for x in order[:keep])


def _pair_assignments(
    scores: np.ndarray,
    atlas: FrozenAtlas,
) -> tuple[EdgeAssignment, ...]:
    kwargs = {
        "min_score": atlas.threshold,
        "max_matches": atlas.max_matches,
        "ambiguity_tolerance": atlas.ambiguity_tolerance,
    }
    if atlas.matcher == "greedy":
        return greedy_match(scores, **kwargs)
    if atlas.matcher == "hungarian":
        return hungarian_match(scores, **kwargs)
    if atlas.matcher == "optimal_transport":
        return entropic_ot_match(scores, regularization=atlas.ot_regularization, **kwargs)
    raise ValueError("joint_triangle does not make pairwise assignments")


def _matrices_for_group(
    matrices: ModelMatrices,
    models: tuple[str, ...],
    group: FrozenLayerGroup,
) -> tuple[np.ndarray, ...]:
    selected = []
    n_samples = None
    for model, layer, pool in zip(models, group.layers, group.feature_pools):
        if model not in matrices or layer not in matrices[model]:
            raise KeyError(f"missing matrix for model={model!r}, layer={layer!r}")
        full = _as_matrix(matrices[model][layer], f"{model}/{layer}")
        if pool and max(pool) >= full.shape[1]:
            raise ValueError(f"{model}/{layer} lacks a frozen feature column")
        if n_samples is None:
            n_samples = full.shape[0]
        elif full.shape[0] != n_samples:
            raise ValueError("all models in a layer group must contain the same samples")
        selected.append(full[:, pool])
    return tuple(selected)


def _build_group_matches(
    matrices: ModelMatrices,
    template: FrozenAtlas,
    group: FrozenLayerGroup,
) -> list[FrozenMatch]:
    arrays = _matrices_for_group(matrices, template.models, group)
    signed = {
        pair: correlation_matrix(arrays[pair[0]], arrays[pair[1]])
        for pair in combinations(range(len(template.models)), 2)
    }
    scores = {pair: _matching_scores(value, template.correlation_mode) for pair, value in signed.items()}
    result = []

    if len(template.models) == 2:
        for edge in _pair_assignments(scores[(0, 1)], template):
            refs = tuple(
                FeatureRef(model, layer, group.feature_pools[i][index])
                for i, (model, layer, index) in enumerate(
                    zip(template.models, group.layers, (edge.left, edge.right))
                )
            )
            result.append(
                FrozenMatch(
                    group.name,
                    refs,
                    (float(signed[(0, 1)][edge.left, edge.right]),),
                    (edge.score,),
                    edge.ambiguity,
                    edge.confidence,
                )
            )
        return result

    if template.matcher == "joint_triangle":
        triangles = joint_triangle_match(
            scores[(0, 1)],
            scores[(0, 2)],
            scores[(1, 2)],
            min_score=template.threshold,
            max_matches=template.max_matches,
            candidate_width=template.joint_candidate_width,
            ambiguity_tolerance=template.ambiguity_tolerance,
        )
        for triangle in triangles:
            indices = (triangle.first, triangle.second, triangle.third)
            refs = tuple(
                FeatureRef(model, layer, group.feature_pools[i][indices[i]])
                for i, (model, layer) in enumerate(zip(template.models, group.layers))
            )
            signed_edges = tuple(
                float(signed[pair][indices[pair[0]], indices[pair[1]]])
                for pair in combinations(range(3), 2)
            )
            result.append(
                FrozenMatch(
                    group.name,
                    refs,
                    signed_edges,
                    triangle.pair_scores,
                    triangle.ambiguity,
                    triangle.confidence,
                )
            )
        return result

    pair_matches = {pair: _pair_assignments(value, template) for pair, value in scores.items()}
    ac_by_a = {edge.left: edge for edge in pair_matches[(0, 2)]}
    bc_by_b = {edge.left: edge for edge in pair_matches[(1, 2)]}
    for ab_edge in pair_matches[(0, 1)]:
        ac_edge = ac_by_a.get(ab_edge.left)
        bc_edge = bc_by_b.get(ab_edge.right)
        if ac_edge is None or bc_edge is None or ac_edge.right != bc_edge.right:
            continue
        indices = (ab_edge.left, ab_edge.right, ac_edge.right)
        edges = (ab_edge, ac_edge, bc_edge)
        refs = tuple(
            FeatureRef(model, layer, group.feature_pools[i][indices[i]])
            for i, (model, layer) in enumerate(zip(template.models, group.layers))
        )
        signed_edges = tuple(
            float(signed[pair][indices[pair[0]], indices[pair[1]]])
            for pair in combinations(range(3), 2)
        )
        result.append(
            FrozenMatch(
                group.name,
                refs,
                signed_edges,
                tuple(edge.score for edge in edges),
                sum(edge.ambiguity for edge in edges),
                min(edge.confidence for edge in edges),
            )
        )
    return result


def _rematch(matrices: ModelMatrices, template: FrozenAtlas) -> FrozenAtlas:
    matches = []
    for group in template.groups:
        matches.extend(_build_group_matches(matrices, template, group))
    matches.sort(key=lambda x: (-x.minimum_matching_score, repr(x.identity)))
    return replace(template, matches=tuple(matches))


def discover_atlas(
    cohort_a: ModelMatrices,
    *,
    layer_groups: LayerGroups | None = None,
    models: Sequence[str] | None = None,
    feature_pool_size: int | None = None,
    matcher: str = "greedy",
    correlation_mode: str = "positive",
    threshold: float = 0.0,
    max_matches: int | None = None,
    ambiguity_tolerance: float = 0.02,
    ot_regularization: float = 0.05,
    joint_candidate_width: int = 8,
) -> FrozenAtlas:
    """Discover and freeze candidate identities using cohort A only."""

    model_order = tuple(cohort_a) if models is None else tuple(models)
    if len(model_order) not in (2, 3) or len(set(model_order)) != len(model_order):
        raise ValueError("atlas discovery requires two or three distinct models")
    if any(model not in cohort_a for model in model_order):
        raise KeyError("models contains an unknown model")
    matcher = _normalise_matcher(matcher)
    if matcher == "joint_triangle" and len(model_order) != 3:
        raise ValueError("joint_triangle requires exactly three models")
    _matching_scores(np.zeros((1, 1)), correlation_mode)
    if not np.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and within [-1, 1]")
    if max_matches is not None and max_matches < 1:
        raise ValueError("max_matches must be positive or None")
    if ambiguity_tolerance < 0 or not np.isfinite(ambiguity_tolerance):
        raise ValueError("ambiguity_tolerance must be finite and nonnegative")
    if ot_regularization <= 0 or joint_candidate_width < 1:
        raise ValueError("matcher tuning parameters must be positive")

    groups = []
    for name, layers in _normalise_groups(cohort_a, model_order, layer_groups):
        matrices = []
        n_samples = None
        for model, layer in zip(model_order, layers):
            if layer not in cohort_a[model]:
                raise KeyError(f"missing matrix for model={model!r}, layer={layer!r}")
            matrix = _as_matrix(cohort_a[model][layer], f"{model}/{layer}")
            if n_samples is None:
                n_samples = matrix.shape[0]
            elif matrix.shape[0] != n_samples:
                raise ValueError("all models in a layer group must contain the same samples")
            matrices.append(matrix)
        pools = tuple(_select_pool(matrix, feature_pool_size) for matrix in matrices)
        groups.append(FrozenLayerGroup(name, layers, pools))

    empty = FrozenAtlas(
        model_order,
        tuple(groups),
        (),
        correlation_mode,
        matcher,
        float(threshold),
        max_matches,
        float(ambiguity_tolerance),
        float(ot_regularization),
        int(joint_candidate_width),
    )
    return _rematch(cohort_a, empty)


def score_atlas(atlas: FrozenAtlas, cohort_b: ModelMatrices) -> AtlasEvaluation:
    """Evaluate frozen cohort-A identities with signed correlations on cohort B."""

    group_cache: dict[Hashable, tuple[dict, dict, list[dict[int, int]]]] = {}
    for group in atlas.groups:
        arrays = _matrices_for_group(cohort_b, atlas.models, group)
        signed = {
            pair: correlation_matrix(arrays[pair[0]], arrays[pair[1]])
            for pair in combinations(range(len(atlas.models)), 2)
        }
        scores = {pair: _matching_scores(value, atlas.correlation_mode) for pair, value in signed.items()}
        pool_indices = [
            {feature: i for i, feature in enumerate(pool)} for pool in group.feature_pools
        ]
        group_cache[group.name] = (signed, scores, pool_indices)

    heldout = []
    for match in atlas.matches:
        signed, scores, pool_indices = group_cache[match.group]
        positions = tuple(
            pool_indices[i][feature.feature] for i, feature in enumerate(match.features)
        )
        signed_edges = []
        matching_edges = []
        ambiguities = []
        confidences = []
        for pair in combinations(range(len(atlas.models)), 2):
            i, j = positions[pair[0]], positions[pair[1]]
            signed_edges.append(float(signed[pair][i, j]))
            matching_edges.append(float(scores[pair][i, j]))
            ambiguity, confidence = _edge_diagnostics(
                scores[pair], i, j, atlas.ambiguity_tolerance
            )
            ambiguities.append(ambiguity)
            confidences.append(confidence)
        passes = all(value >= atlas.threshold for value in matching_edges)
        heldout.append(
            HeldoutMatch(
                match,
                tuple(signed_edges),
                tuple(matching_edges),
                sum(ambiguities),
                min(confidences),
                passes,
            )
        )
    n_passing = sum(match.passes_threshold for match in heldout)
    frozen_ids = {match.identity for match in atlas.matches}
    passing_ids = {match.identity for match in heldout if match.passes_threshold}
    union = frozen_ids | passing_ids
    retained = len(frozen_ids & passing_ids) / len(union) if union else 1.0
    return AtlasEvaluation(tuple(heldout), n_passing, float(retained))


def _identity_set(value: FrozenAtlas | AtlasEvaluation | Iterable) -> set:
    if isinstance(value, FrozenAtlas):
        return {match.identity for match in value.matches}
    if isinstance(value, AtlasEvaluation):
        return {match.identity for match in value.matches if match.passes_threshold}
    return {match.identity for match in value}


def identity_overlap(
    left: FrozenAtlas | AtlasEvaluation | Iterable,
    right: FrozenAtlas | AtlasEvaluation | Iterable,
) -> IdentityOverlap:
    """Report exact model/layer/feature identity overlap between atlas outputs."""

    a, b = _identity_set(left), _identity_set(right)
    intersection = len(a & b)
    union = len(a | b)
    return IdentityOverlap(len(a), len(b), intersection, union, intersection / union if union else 1.0)


def identity_jaccard(
    left: FrozenAtlas | AtlasEvaluation | Iterable,
    right: FrozenAtlas | AtlasEvaluation | Iterable,
) -> float:
    return identity_overlap(left, right).jaccard


def _sample_count(matrices: ModelMatrices, models: Sequence[str]) -> int:
    counts = {
        _as_matrix(matrix, f"{model}/{layer}").shape[0]
        for model in models
        for layer, matrix in matrices[model].items()
    }
    if len(counts) != 1:
        raise ValueError("coherent permutations require the same sample count in every matrix")
    return counts.pop()


def draw_model_permutations(
    matrices: ModelMatrices,
    rng: np.random.Generator,
    *,
    models: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Draw one row permutation per model, independent of layer."""

    model_order = tuple(matrices) if models is None else tuple(models)
    if any(model not in matrices for model in model_order):
        raise KeyError("models contains an unknown model")
    n_samples = _sample_count(matrices, model_order)
    return {model: rng.permutation(n_samples) for model in model_order}


def apply_model_permutations(
    matrices: ModelMatrices,
    permutations: Mapping[str, np.ndarray],
) -> dict[str, dict[Hashable, np.ndarray]]:
    """Apply each model's single permutation to every one of its layers."""

    if set(permutations) != set(matrices):
        raise ValueError("permutations must specify every model exactly once")
    result = {}
    for model, layers in matrices.items():
        permutation = np.asarray(permutations[model])
        if permutation.ndim != 1 or not np.issubdtype(permutation.dtype, np.integer):
            raise ValueError("permutation indices must be a one-dimensional integer array")
        if not np.array_equal(np.sort(permutation), np.arange(permutation.size)):
            raise ValueError("invalid row permutation")
        result[model] = {}
        for layer, matrix in layers.items():
            values = _as_matrix(matrix, f"{model}/{layer}")
            if values.shape[0] != permutation.size:
                raise ValueError("permutation length does not match a layer matrix")
            result[model][layer] = values[permutation]
    return result


def empirical_pvalue(
    observed: float,
    null_values: Sequence[float],
    *,
    alternative: str = "greater",
) -> float:
    """Plus-one empirical p-value; equality is included in the null tail."""

    null = np.asarray(null_values, dtype=np.float64)
    if null.ndim != 1 or null.size < 1 or not np.isfinite(null).all() or not np.isfinite(observed):
        raise ValueError("observed and null values must be finite and the null non-empty")
    if alternative == "greater":
        extreme = np.count_nonzero(null >= observed)
    elif alternative == "less":
        extreme = np.count_nonzero(null <= observed)
    elif alternative == "two-sided":
        extreme = np.count_nonzero(np.abs(null) >= abs(observed))
    else:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'")
    return float((1 + extreme) / (null.size + 1))


def _mean_atlas_score(atlas: FrozenAtlas) -> float:
    if not atlas.matches:
        return 0.0
    return float(np.mean([match.minimum_matching_score for match in atlas.matches]))


def coherent_permutation_test(
    atlas: FrozenAtlas,
    matrices: ModelMatrices,
    *,
    n_permutations: int = 1_000,
    seed: int = 0,
    return_permutations: bool = False,
) -> PermutationResult:
    """Rematch frozen pools under coherent, model-wise sequence reassignments.

    A replicate draws exactly one permutation for each model and applies it to
    all of that model's layers and all downstream model pairs.  Feature pools
    and layer mappings remain frozen.  Both reported p-values use the plus-one
    correction.
    """

    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    # Validate all matrices up front, including layers not visited first.
    _sample_count(matrices, atlas.models)
    selected_matrices = {model: matrices[model] for model in atlas.models}
    observed_atlas = _rematch(selected_matrices, atlas)
    observed_count = len(observed_atlas.matches)
    observed_mean = _mean_atlas_score(observed_atlas)
    rng = np.random.default_rng(seed)
    null_counts = []
    null_means = []
    saved = [] if return_permutations else None
    for _ in range(n_permutations):
        permutations = draw_model_permutations(selected_matrices, rng)
        permuted = apply_model_permutations(selected_matrices, permutations)
        null_atlas = _rematch(permuted, atlas)
        null_counts.append(len(null_atlas.matches))
        null_means.append(_mean_atlas_score(null_atlas))
        if saved is not None:
            saved.append(
                tuple((model, tuple(int(x) for x in permutations[model])) for model in atlas.models)
            )
    return PermutationResult(
        observed_count,
        tuple(null_counts),
        empirical_pvalue(observed_count, null_counts),
        observed_mean,
        tuple(null_means),
        empirical_pvalue(observed_mean, null_means),
        tuple(saved) if saved is not None else None,
    )
