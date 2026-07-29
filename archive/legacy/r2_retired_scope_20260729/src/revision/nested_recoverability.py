"""Nested, repeated, group-aware representation recoverability evaluation.

Layer selection occurs only inside each outer training fold.  Every
representation and control uses the same outer/inner group folds, and all
ceiling/floor comparisons are paired on the untouched outer predictions.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Mapping, Sequence

import numpy as np
from scipy import stats
from sklearn.decomposition import FastICA, NMF, PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .statistics import paired_group_bootstrap


RepresentationSet = Mapping[str, Mapping[object, np.ndarray]]
QualityByFloorLayer = Mapping[str, Mapping[object, Sequence[float]]]
PRODUCTION_MINIMUM_SAMPLES = 480


def _derived_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _validate_inputs(
    representations: RepresentationSet,
    y: Sequence,
    groups: Sequence,
    *,
    task_type: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[object, np.ndarray]]]:
    truth = np.asarray(y)
    group_ids = np.asarray(groups)
    if truth.ndim != 1 or truth.size < 8:
        raise ValueError(
            "y must be a one-dimensional vector with at least eight samples"
        )
    if group_ids.shape != truth.shape:
        raise ValueError("groups must align with y")
    if task_type not in {"classification", "regression"}:
        raise ValueError("task_type must be classification or regression")
    if task_type == "classification" and np.unique(truth).size < 2:
        raise ValueError("classification requires at least two classes")
    if task_type == "regression":
        truth = truth.astype(np.float64)
        if not np.isfinite(truth).all():
            raise ValueError("regression targets contain non-finite values")
        if np.ptp(truth) == 0.0:
            raise ValueError("regression targets must vary")
    checked: dict[str, dict[object, np.ndarray]] = {}
    if not representations:
        raise ValueError("at least one representation is required")
    for name, layers in representations.items():
        if not layers:
            raise ValueError(f"representation {name!r} has no layers")
        checked[name] = {}
        for layer, value in layers.items():
            matrix = np.asarray(value, dtype=np.float64)
            if matrix.ndim != 2 or matrix.shape[0] != truth.size or matrix.shape[1] < 1:
                raise ValueError(
                    f"representation {name!r} layer {layer!r} must have shape [n_samples, n_features]"
                )
            if not np.isfinite(matrix).all():
                raise ValueError(
                    f"representation {name!r} layer {layer!r} contains non-finite values"
                )
            checked[name][layer] = matrix
    return truth, group_ids, checked


def make_group_splits(
    y: Sequence,
    groups: Sequence,
    *,
    n_splits: int,
    seed: int,
    task_type: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create and validate identity-group-disjoint folds."""

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


def _metric(task_type: str, labels: np.ndarray | None = None):
    if task_type == "classification":
        frozen_labels = np.asarray(labels)

        def classification_metric(y_true: np.ndarray, prediction: np.ndarray) -> float:
            return float(
                f1_score(
                    y_true,
                    prediction,
                    labels=frozen_labels,
                    average="macro",
                    zero_division=0,
                )
            )

        return classification_metric

    def regression_metric(y_true: np.ndarray, prediction: np.ndarray) -> float:
        return float(r2_score(y_true, prediction))

    return regression_metric


def _fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    task_type: str,
    seed: int,
    comparison_dimension: int | None = None,
) -> np.ndarray:
    if task_type == "classification":
        estimator = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=3000,
            random_state=seed,
        )
    else:
        estimator = Ridge(alpha=1.0)
    steps: list = [StandardScaler()]
    if comparison_dimension is not None:
        dimension = int(comparison_dimension)
        maximum = min(x_train.shape[0] - 1, x_train.shape[1])
        if dimension < 2 or dimension > maximum:
            raise ValueError(
                f"comparison dimension {dimension} is infeasible for a training "
                f"matrix with shape {x_train.shape}; maximum centered rank is {maximum}"
            )
        steps.append(
            PCA(
                n_components=dimension,
                svd_solver="randomized",
                random_state=seed,
            )
        )
    steps.append(estimator)
    pipeline = make_pipeline(*steps)
    pipeline.fit(x_train, y_train)
    return np.asarray(pipeline.predict(x_test))


def _select_layer(
    layers: Mapping[object, np.ndarray],
    truth: np.ndarray,
    train_indices: np.ndarray,
    *,
    task_type: str,
    inner_folds: Sequence[tuple[np.ndarray, np.ndarray]],
    seed: int,
    metric,
    comparison_dimension: int,
) -> tuple[object, dict[str, float]]:
    scores: dict[object, float] = {}
    for layer in sorted(layers, key=str):
        predictions = np.empty(
            train_indices.size,
            dtype=truth.dtype if task_type == "classification" else float,
        )
        for fold_index, (inner_train, inner_test) in enumerate(inner_folds):
            global_train = train_indices[inner_train]
            global_test = train_indices[inner_test]
            predictions[inner_test] = _fit_predict(
                layers[layer][global_train],
                truth[global_train],
                layers[layer][global_test],
                task_type=task_type,
                seed=_derived_seed(seed, layer, fold_index),
                comparison_dimension=comparison_dimension,
            )
        scores[layer] = metric(truth[train_indices], predictions)
    selected = max(sorted(scores, key=str), key=lambda layer: scores[layer])
    return selected, {
        str(layer): float(scores[layer]) for layer in sorted(scores, key=str)
    }


def _control_transform(
    method: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    *,
    dimension: int,
    seed: int,
    metadata: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    dimension = int(dimension)
    maximum = min(x_train.shape[0] - 1, x_train.shape[1])
    if dimension < 2:
        raise ValueError("control dimension is too small for the training fold")
    if dimension > maximum:
        raise ValueError(
            f"requested control dimension {dimension} exceeds the outer-training "
            f"rank limit {maximum}; refusing a silently unmatched control"
        )
    if method == "pca":
        transform = make_pipeline(
            StandardScaler(),
            PCA(n_components=dimension, svd_solver="randomized", random_state=seed),
        )
        if metadata is not None:
            metadata.update({"method": method, "dimension": dimension})
        return transform.fit_transform(x_train), transform.transform(x_test)
    if method == "random_projection":
        scaler = StandardScaler().fit(x_train)
        train_scaled, test_scaled = scaler.transform(x_train), scaler.transform(x_test)
        rng = np.random.default_rng(seed)
        projection = rng.normal(size=(x_train.shape[1], dimension)) / np.sqrt(
            x_train.shape[1]
        )
        if metadata is not None:
            metadata.update({"method": method, "dimension": dimension})
        return train_scaled @ projection, test_scaled @ projection
    if method == "random_dictionary":
        scaler = StandardScaler().fit(x_train)
        train_scaled, test_scaled = scaler.transform(x_train), scaler.transform(x_test)
        rng = np.random.default_rng(seed)
        dictionary = rng.normal(size=(x_train.shape[1], dimension)) / np.sqrt(
            x_train.shape[1]
        )
        if metadata is not None:
            metadata.update({"method": method, "dimension": dimension})
        return np.maximum(train_scaled @ dictionary, 0.0), np.maximum(
            test_scaled @ dictionary, 0.0
        )
    if method == "nmf":
        shift = np.minimum(x_train.min(axis=0), 0.0)
        train_nonnegative = x_train - shift
        test_nonnegative = np.maximum(x_test - shift, 0.0)
        transform = NMF(
            n_components=dimension,
            init="nndsvda",
            random_state=seed,
            max_iter=1000,
            tol=1e-4,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                train_transformed = transform.fit_transform(train_nonnegative)
                test_transformed = transform.transform(test_nonnegative)
            except ConvergenceWarning as error:
                raise RuntimeError(
                    "NMF failed to converge in the outer training fold"
                ) from error
        if metadata is not None:
            metadata.update(
                {
                    "method": method,
                    "dimension": dimension,
                    "iterations": int(transform.n_iter_),
                }
            )
        return train_transformed, test_transformed
    if method == "ica":
        scaler = StandardScaler().fit(x_train)
        train_scaled, test_scaled = scaler.transform(x_train), scaler.transform(x_test)
        # FastICA can depend materially on initialization. Use a fixed,
        # prespecified 16-start estimator and record the accepted start;
        # never accept a warning or exhausted iteration budget.
        for attempt in range(16):
            transform = FastICA(
                n_components=dimension,
                whiten="unit-variance",
                random_state=_derived_seed(seed, "ica", attempt),
                algorithm="parallel",
                max_iter=5000,
                tol=1e-3,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                try:
                    train_transformed = transform.fit_transform(train_scaled)
                except ConvergenceWarning:
                    continue
            if transform.n_iter_ >= transform.max_iter:
                continue
            if metadata is not None:
                metadata.update(
                    {
                        "method": method,
                        "dimension": dimension,
                        "accepted_start": attempt,
                        "n_starts": 16,
                        "iterations": int(transform.n_iter_),
                    }
                )
            return train_transformed, transform.transform(test_scaled)
        raise RuntimeError(
            "ICA failed all 16 prespecified starts in the outer training fold"
        )
    raise ValueError(f"unknown control method: {method}")


def _spearman_group_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    n_bootstrap: int,
) -> dict:
    if left.shape != right.shape or left.shape != groups.shape:
        raise ValueError("correlation vectors and groups must align")
    if np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return {
            "spearman_rho": None,
            "ci95": None,
            "n_groups": int(np.unique(groups).size),
        }
    point = float(stats.spearmanr(left, right).statistic)
    if not np.isfinite(point):
        return {
            "spearman_rho": None,
            "ci95": None,
            "n_groups": int(np.unique(groups).size),
        }
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    values = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_groups, size=unique_groups.size, replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        if np.ptp(left[indices]) == 0.0 or np.ptp(right[indices]) == 0.0:
            continue
        value = float(stats.spearmanr(left[indices], right[indices]).statistic)
        if np.isfinite(value):
            values.append(value)
    return {
        "spearman_rho": point,
        "ci95": (
            [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
            if values
            else None
        ),
        "n_groups": int(unique_groups.size),
    }


def run_nested_recoverability(
    representations: RepresentationSet,
    y: Sequence,
    groups: Sequence,
    *,
    ceiling_name: str,
    floor_names: Sequence[str],
    task_type: str,
    analysis_seeds: Sequence[int],
    outer_splits: int = 5,
    inner_splits: int = 4,
    control_methods: Sequence[str] = (
        "pca",
        "random_projection",
        "nmf",
        "ica",
        "random_dictionary",
    ),
    comparison_dimension: int,
    active_width_dimension: int | None = None,
    n_bootstrap: int = 1000,
    reconstruction_error_by_floor_layer: QualityByFloorLayer | None = None,
    intervention_effect_by_floor_layer: QualityByFloorLayer | None = None,
    confirmatory_real: bool = False,
) -> dict:
    """Run matched-dimensional nested probes with paired inference."""

    truth, group_ids, matrices = _validate_inputs(
        representations, y, groups, task_type=task_type
    )
    if ceiling_name not in matrices:
        raise ValueError(f"missing ceiling representation {ceiling_name!r}")
    if not floor_names or any(name not in matrices for name in floor_names):
        raise ValueError("every floor representation must exist")
    if len(set(floor_names)) != len(floor_names):
        raise ValueError("floor representation names must be unique")
    if ceiling_name in floor_names:
        raise ValueError("ceiling representation cannot also be a floor")
    if not analysis_seeds or len(set(analysis_seeds)) != len(analysis_seeds):
        raise ValueError("analysis_seeds must be unique and non-empty")
    allowed_controls = {"pca", "random_projection", "nmf", "ica", "random_dictionary"}
    if (
        len(set(control_methods)) != len(control_methods)
        or not set(control_methods) <= allowed_controls
    ):
        raise ValueError("control methods must be unique supported names")
    if set(control_methods) & set(matrices):
        raise ValueError(
            "control method names cannot collide with representation names"
        )
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    comparison_dimension = int(comparison_dimension)
    if comparison_dimension < 2:
        raise ValueError("comparison_dimension must be at least two")
    if active_width_dimension is not None:
        active_width_dimension = int(active_width_dimension)
        if active_width_dimension < 2:
            raise ValueError("active_width_dimension must be at least two")
        if active_width_dimension >= comparison_dimension:
            raise ValueError(
                "active_width_dimension must be smaller than comparison_dimension"
            )
    maximum_dimension = max(
        comparison_dimension,
        active_width_dimension if active_width_dimension is not None else 0,
    )
    insufficient = [
        f"{name}:{layer}={matrix.shape[1]}"
        for name, layers in matrices.items()
        for layer, matrix in layers.items()
        if matrix.shape[1] < maximum_dimension
    ]
    if insufficient:
        raise ValueError(
            "every representation layer must support the largest declared common "
            f"dimension {maximum_dimension}; insufficient widths: {', '.join(insufficient)}"
        )

    if confirmatory_real and (
        reconstruction_error_by_floor_layer is None
        or intervention_effect_by_floor_layer is None
    ):
        raise ValueError(
            "confirmatory real mode requires seed/layer-specific reconstruction "
            "error and intervention effect"
        )
    if (
        intervention_effect_by_floor_layer is not None
        and reconstruction_error_by_floor_layer is None
    ):
        raise ValueError("intervention effect requires reconstruction error")

    def validate_quality(
        values: QualityByFloorLayer | None,
        *,
        label: str,
        nonnegative: bool,
    ) -> dict[str, dict[object, np.ndarray]] | None:
        if values is None:
            return None
        if not isinstance(values, Mapping) or set(values) != set(floor_names):
            raise ValueError(f"{label} must contain exactly every floor seed")
        normalized: dict[str, dict[object, np.ndarray]] = {}
        for floor_name in floor_names:
            layers = values[floor_name]
            expected_layers = set(matrices[floor_name])
            if not isinstance(layers, Mapping) or set(layers) != expected_layers:
                raise ValueError(
                    f"{label} for {floor_name} must contain exactly every "
                    "representation layer"
                )
            normalized[floor_name] = {}
            for layer, source in layers.items():
                vector = np.asarray(source, dtype=np.float64)
                if vector.shape != truth.shape or not np.isfinite(vector).all():
                    raise ValueError(f"{label} must be finite and align with y")
                if nonnegative and np.any(vector < 0.0):
                    raise ValueError(f"{label} must be non-negative")
                normalized[floor_name][layer] = vector
        return normalized

    reconstruction = validate_quality(
        reconstruction_error_by_floor_layer,
        label="reconstruction error",
        nonnegative=True,
    )
    intervention = validate_quality(
        intervention_effect_by_floor_layer,
        label="intervention effect",
        nonnegative=False,
    )
    if confirmatory_real and truth.size < PRODUCTION_MINIMUM_SAMPLES:
        raise ValueError(
            "confirmatory real P0-8 requires the frozen enlarged-cohort minimum "
            f"of {PRODUCTION_MINIMUM_SAMPLES} samples"
        )

    dimension_tracks = {
        "primary_common_dimension": {
            "dimension": comparison_dimension,
            "role": "confirmatory_primary",
            "interpretation": (
                "Every sparse, dense, random and control arm enters its probe at "
                "this exact fold-fitted dimension."
            ),
        }
    }
    if active_width_dimension is not None:
        dimension_tracks["active_width_rank_sensitivity"] = {
            "dimension": active_width_dimension,
            "role": "sensitivity_only",
            "interpretation": (
                "A separately labelled active-width/rank sensitivity analysis; "
                "it is not a raw-coordinate-width comparison."
            ),
        }
    labels = np.unique(truth) if task_type == "classification" else None
    metric = _metric(task_type, labels)
    base_names = list(matrices)
    prediction_runs: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    fold_rows: list[dict] = []
    fold_manifest: list[dict] = []

    for analysis_seed in analysis_seeds:
        outer = make_group_splits(
            truth,
            group_ids,
            n_splits=outer_splits,
            seed=int(analysis_seed),
            task_type=task_type,
        )
        run_predictions = {
            track_name: {
                name: np.empty(
                    truth.size,
                    dtype=truth.dtype if task_type == "classification" else float,
                )
                for name in [*base_names, *control_methods]
            }
            for track_name in dimension_tracks
        }
        for outer_fold, (train, test) in enumerate(outer):
            shared_inner_folds = make_group_splits(
                truth[train],
                group_ids[train],
                n_splits=inner_splits,
                seed=_derived_seed("inner", analysis_seed, outer_fold),
                task_type=task_type,
            )
            fold_manifest.append(
                {
                    "analysis_seed": int(analysis_seed),
                    "outer_fold": int(outer_fold),
                    "train_indices": train.tolist(),
                    "test_indices": test.tolist(),
                    "train_groups": [
                        str(value) for value in np.unique(group_ids[train])
                    ],
                    "test_groups": [str(value) for value in np.unique(group_ids[test])],
                    "inner_folds": [
                        {
                            "inner_fold": int(inner_fold),
                            "train_indices": train[inner_train].tolist(),
                            "test_indices": train[inner_test].tolist(),
                            "train_groups": [
                                str(value)
                                for value in np.unique(group_ids[train[inner_train]])
                            ],
                            "test_groups": [
                                str(value)
                                for value in np.unique(group_ids[train[inner_test]])
                            ],
                        }
                        for inner_fold, (inner_train, inner_test) in enumerate(
                            shared_inner_folds
                        )
                    ],
                }
            )
            for track_name, track in dimension_tracks.items():
                dimension = int(track["dimension"])
                selections: dict[str, object] = {}
                for name in base_names:
                    layer, scores = _select_layer(
                        matrices[name],
                        truth,
                        train,
                        task_type=task_type,
                        # The same materialized folds and common dimension are
                        # passed to every sparse and dense representation.
                        inner_folds=shared_inner_folds,
                        seed=_derived_seed(
                            "inner-probe", analysis_seed, outer_fold, track_name
                        ),
                        metric=metric,
                        comparison_dimension=dimension,
                    )
                    selections[name] = layer
                    raw_matrix = matrices[name][layer]
                    predictions = _fit_predict(
                        raw_matrix[train],
                        truth[train],
                        raw_matrix[test],
                        task_type=task_type,
                        seed=_derived_seed(
                            "outer", analysis_seed, outer_fold, track_name, name
                        ),
                        comparison_dimension=dimension,
                    )
                    run_predictions[track_name][name][test] = predictions
                    fold_rows.append(
                        {
                            "analysis_seed": int(analysis_seed),
                            "outer_fold": int(outer_fold),
                            "dimension_track": track_name,
                            "track_role": track["role"],
                            "representation": name,
                            "arm_role": (
                                "sparse_code_pca"
                                if name in floor_names
                                else (
                                    "dense_clt_input_pca"
                                    if name == ceiling_name
                                    else "dense_representation_pca"
                                )
                            ),
                            "selected_layer": str(layer),
                            "inner_layer_scores": scores,
                            "raw_coordinate_width": int(raw_matrix.shape[1]),
                            "common_transform": "train_fold_pca",
                            "probe_input_dimension": dimension,
                            "outer_metric": metric(truth[test], predictions),
                            "n_test": int(test.size),
                        }
                    )

                ceiling_layer = selections[ceiling_name]
                for method in control_methods:
                    transform_metadata: dict = {}
                    x_train, x_test = _control_transform(
                        method,
                        matrices[ceiling_name][ceiling_layer][train],
                        matrices[ceiling_name][ceiling_layer][test],
                        dimension=dimension,
                        seed=_derived_seed(
                            "control", analysis_seed, outer_fold, track_name, method
                        ),
                        metadata=transform_metadata,
                    )
                    if x_train.shape[1] != dimension or x_test.shape[1] != dimension:
                        raise RuntimeError(
                            "control transform violated the common dimension"
                        )
                    predictions = _fit_predict(
                        x_train,
                        truth[train],
                        x_test,
                        task_type=task_type,
                        seed=_derived_seed(
                            "control-probe",
                            analysis_seed,
                            outer_fold,
                            track_name,
                            method,
                        ),
                    )
                    run_predictions[track_name][method][test] = predictions
                    fold_rows.append(
                        {
                            "analysis_seed": int(analysis_seed),
                            "outer_fold": int(outer_fold),
                            "dimension_track": track_name,
                            "track_role": track["role"],
                            "representation": method,
                            "arm_role": "matched_dimension_control",
                            "selected_layer": str(ceiling_layer),
                            "derived_from": ceiling_name,
                            "raw_coordinate_width": int(
                                matrices[ceiling_name][ceiling_layer].shape[1]
                            ),
                            "common_transform": method,
                            "probe_input_dimension": dimension,
                            "transform": transform_metadata,
                            "outer_metric": metric(truth[test], predictions),
                            "n_test": int(test.size),
                        }
                    )
        prediction_runs[int(analysis_seed)] = run_predictions

    per_seed_metrics = {
        str(seed): {
            track_name: {
                name: metric(truth, predictions)
                for name, predictions in track_predictions.items()
            }
            for track_name, track_predictions in prediction_runs[seed].items()
        }
        for seed in prediction_runs
    }
    representation_summary = {
        track_name: {
            name: {
                "mean_metric": float(
                    np.mean(
                        [
                            per_seed_metrics[str(seed)][track_name][name]
                            for seed in prediction_runs
                        ]
                    )
                ),
                "seed_metrics": {
                    str(seed): float(per_seed_metrics[str(seed)][track_name][name])
                    for seed in prediction_runs
                },
            }
            for name in prediction_runs[next(iter(prediction_runs))][track_name]
        }
        for track_name in dimension_tracks
    }

    comparisons = []
    for track_name, track in dimension_tracks.items():
        for floor_name in floor_names:
            for reference_name in (ceiling_name, *control_methods):
                seed_results = []
                for analysis_seed in prediction_runs:
                    track_predictions = prediction_runs[analysis_seed][track_name]
                    seed_results.append(
                        {
                            "analysis_seed": int(analysis_seed),
                            **paired_group_bootstrap(
                                truth,
                                track_predictions[floor_name],
                                track_predictions[reference_name],
                                group_ids,
                                metric,
                                seed=_derived_seed(
                                    "bootstrap",
                                    analysis_seed,
                                    track_name,
                                    floor_name,
                                    reference_name,
                                ),
                                n_bootstrap=n_bootstrap,
                            ),
                        }
                    )
                comparisons.append(
                    {
                        "dimension_track": track_name,
                        "track_role": track["role"],
                        "probe_input_dimension": int(track["dimension"]),
                        "floor": floor_name,
                        "reference": reference_name,
                        "reference_role": (
                            "dense_pca_ceiling"
                            if reference_name == ceiling_name
                            else "matched_dimension_control"
                        ),
                        "estimand": "matched_dimension_floor_minus_reference",
                        "seed_results": seed_results,
                        "mean_difference": float(
                            np.mean([row["difference"] for row in seed_results])
                        ),
                        "mean_ratio": (
                            float(
                                np.mean(
                                    [
                                        row["ratio"]
                                        for row in seed_results
                                        if row["ratio"] is not None
                                    ]
                                )
                            )
                            if any(row["ratio"] is not None for row in seed_results)
                            else None
                        ),
                    }
                )

    relationships: list[dict] = []
    if reconstruction is not None:
        for track_name, track in dimension_tracks.items():
            for floor_name in floor_names:
                for quality_layer, reconstruction_vector in reconstruction[
                    floor_name
                ].items():
                    for analysis_seed in prediction_runs:
                        predictions = prediction_runs[analysis_seed][track_name][
                            floor_name
                        ]
                        probe_error = (
                            (predictions != truth).astype(float)
                            if task_type == "classification"
                            else np.abs(
                                predictions.astype(float) - truth.astype(float)
                            )
                        )
                        common = {
                            "analysis_seed": int(analysis_seed),
                            "dimension_track": track_name,
                            "track_role": track["role"],
                            "representation": floor_name,
                            "quality_layer": str(quality_layer),
                        }
                        relationships.append(
                            {
                                **common,
                                "relationship": "reconstruction_error_vs_probe_error",
                                **_spearman_group_bootstrap(
                                    reconstruction_vector,
                                    probe_error,
                                    group_ids,
                                    seed=_derived_seed(
                                        "relationship",
                                        "reconstruction",
                                        analysis_seed,
                                        track_name,
                                        floor_name,
                                        quality_layer,
                                    ),
                                    n_bootstrap=n_bootstrap,
                                ),
                            }
                        )
                        if intervention is not None:
                            relationships.append(
                                {
                                    **common,
                                    "relationship": "intervention_effect_vs_probe_error",
                                    **_spearman_group_bootstrap(
                                        intervention[floor_name][quality_layer],
                                        probe_error,
                                        group_ids,
                                        seed=_derived_seed(
                                            "relationship",
                                            "intervention",
                                            analysis_seed,
                                            track_name,
                                            floor_name,
                                            quality_layer,
                                        ),
                                        n_bootstrap=n_bootstrap,
                                    ),
                                }
                            )
        if intervention is not None:
            for floor_name in floor_names:
                for quality_layer, reconstruction_vector in reconstruction[
                    floor_name
                ].items():
                    relationships.append(
                        {
                            "representation": floor_name,
                            "quality_layer": str(quality_layer),
                            "relationship": (
                                "reconstruction_error_vs_intervention_effect"
                            ),
                            **_spearman_group_bootstrap(
                                reconstruction_vector,
                                intervention[floor_name][quality_layer],
                                group_ids,
                                seed=_derived_seed(
                                    "relationship",
                                    "reconstruction_intervention",
                                    floor_name,
                                    quality_layer,
                                ),
                                n_bootstrap=n_bootstrap,
                            ),
                        }
                    )

    return {
        "schema_version": "r2-nested-recoverability-v3",
        "scope": (
            "Nested recoverability estimates for the supplied representations; "
            "no biological or causal gate is implied by infrastructure validation."
        ),
        "task_type": task_type,
        "n_samples": int(truth.size),
        "n_identity_groups": int(np.unique(group_ids).size),
        "analysis_seeds": [int(seed) for seed in analysis_seeds],
        "dictionary_representations": list(floor_names),
        "outer_splits": int(outer_splits),
        "inner_splits": int(inner_splits),
        "control_methods": list(control_methods),
        "confirmatory_real": bool(confirmatory_real),
        "dimension_tracks": dimension_tracks,
        "dimensional_matching": {
            "basis": "prespecified_fold_fitted_common_dimension",
            "comparison_dimension": comparison_dimension,
            "active_width_dimension": active_width_dimension,
            "raw_coordinate_widths": {
                name: sorted({int(matrix.shape[1]) for matrix in layers.values()})
                for name, layers in matrices.items()
            },
            "all_arm_probe_inputs_exact_declared_dimension": True,
            "interpretation": (
                "Each raw sparse or dense representation is reduced by PCA fitted "
                "inside its training fold; every derived control emits the same exact "
                "dimension. The sparse-code arm therefore tests matched-rank retained "
                "information, not preservation of coordinate sparsity."
            ),
        },
        "fold_manifest": fold_manifest,
        "fold_results": fold_rows,
        "representation_summary": representation_summary,
        "paired_comparisons": comparisons,
        "quality_relationships": relationships,
        "outer_predictions": {
            str(seed): {
                track_name: {
                    name: predictions.tolist()
                    for name, predictions in track_predictions.items()
                }
                for track_name, track_predictions in prediction_runs[seed].items()
            }
            for seed in prediction_runs
        },
    }


def synthetic_recoverability_fixture(
    *,
    seed: int,
    n_groups: int = 36,
    samples_per_group: int = 4,
    dictionary_seeds: Sequence[int] = (0, 1),
) -> tuple[
    dict[str, dict[int, np.ndarray]],
    np.ndarray,
    np.ndarray,
    dict[str, dict[int, np.ndarray]],
    dict[str, dict[int, np.ndarray]],
]:
    """Create a lightweight identity-grouped fixture for protocol validation."""

    if n_groups < 12 or samples_per_group < 2:
        raise ValueError(
            "synthetic fixture requires at least 12 groups and two samples per group"
        )
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), samples_per_group)
    group_labels = np.arange(n_groups) % 2
    y = np.repeat(group_labels, samples_per_group)
    # Independent non-Gaussian sources make ICA a meaningful fixture rather
    # than asking it to identify an arbitrary rotation of Gaussian noise.
    group_nuisance = np.repeat(
        rng.laplace(scale=0.15, size=(n_groups, 8)), samples_per_group, axis=0
    )
    latent = rng.laplace(size=(groups.size, 8)) + group_nuisance
    latent[:, 0] += (2.0 * y - 1.0) * 1.8
    raw_projection = rng.normal(size=(8, 12))
    raw_good = latent @ raw_projection + rng.normal(scale=0.15, size=(groups.size, 12))
    raw_noise = rng.normal(size=(groups.size, 12))
    representations: dict[str, dict[int, np.ndarray]] = {
        "ceiling": {
            0: raw_noise,
            1: raw_good,
            2: 0.45 * raw_good + rng.normal(size=raw_good.shape),
        },
        "reconstruction": {
            0: rng.normal(size=raw_good.shape),
            1: raw_good + rng.normal(scale=0.25, size=raw_good.shape),
            2: 0.40 * raw_good + rng.normal(size=raw_good.shape),
        },
    }
    for dictionary_seed in dictionary_seeds:
        local = np.random.default_rng(
            _derived_seed("dictionary", seed, dictionary_seed)
        )
        projection = local.normal(size=(12, 5))
        code_good = np.maximum(raw_good @ projection, 0.0)
        representations[f"code_seed_{dictionary_seed}"] = {
            0: np.maximum(raw_noise @ projection, 0.0),
            1: code_good + np.abs(local.normal(scale=0.08, size=code_good.shape)),
            2: np.maximum(
                (0.4 * raw_good + local.normal(size=raw_good.shape)) @ projection, 0.0
            ),
        }
    reconstruction_error: dict[str, dict[int, np.ndarray]] = {}
    intervention_effect: dict[str, dict[int, np.ndarray]] = {}
    for dictionary_seed in dictionary_seeds:
        floor_name = f"code_seed_{dictionary_seed}"
        local = np.random.default_rng(
            _derived_seed("fixture-quality", seed, dictionary_seed)
        )
        reconstruction_error[floor_name] = {}
        intervention_effect[floor_name] = {}
        for layer in representations[floor_name]:
            residual = np.linalg.norm(
                representations["reconstruction"][layer]
                - representations["ceiling"][layer],
                axis=1,
            )
            reconstruction_error[floor_name][layer] = residual + np.abs(
                local.normal(scale=0.01, size=groups.size)
            )
            intervention_effect[floor_name][layer] = (
                0.6
                - 0.1 * reconstruction_error[floor_name][layer]
                + local.normal(scale=0.03, size=groups.size)
            )
    return representations, y, groups, reconstruction_error, intervention_effect
