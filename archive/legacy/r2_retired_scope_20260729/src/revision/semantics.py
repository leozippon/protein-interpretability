"""Blocked conditional tests for continuous feature activations.

The estimand is the out-of-fold reduction in squared error obtained by adding
one prespecified biological label to low-level sequence covariates.  Folds are
blocked by protein or family, label randomization is performed within protein,
and uncertainty is bootstrapped over proteins.  This module contains no model,
dataset, or file-format code so the same folds can be reused for sparse, dense,
and randomized representations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class ConditionalEffect:
    """One representation/feature/label/blocking result."""

    representation: str
    feature: str
    label: str
    blocking: str
    n_observations: int
    n_proteins: int
    n_blocks: int
    baseline_mse: float
    full_mse: float
    delta_mse: float
    delta_r2: float
    permutation_pvalue: float
    qvalue: float
    bootstrap_delta_mse_ci95: tuple[float, float]
    bootstrap_delta_r2_ci95: tuple[float, float]
    bootstrap_standard_error_delta_mse: float
    retrospective_bootstrap_detectable_delta_mse: float
    prospective_minimum_detectable_delta_mse: float | None
    permutable_row_fraction: float
    permutation_degenerate: bool
    fold_hash: str


def _vector(values: Sequence, name: str, n: int | None = None) -> np.ndarray:
    out = np.asarray(values)
    if out.ndim != 1 or out.size < 1:
        raise ValueError(f"{name} must be a non-empty vector")
    if n is not None and out.size != n:
        raise ValueError(f"{name} has {out.size} rows; expected {n}")
    return out


def _finite_matrix(values: np.ndarray, name: str, n: int | None = None) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64)
    if out.ndim != 2 or min(out.shape) < 1:
        raise ValueError(f"{name} must be a non-empty matrix")
    if n is not None and out.shape[0] != n:
        raise ValueError(f"{name} has {out.shape[0]} rows; expected {n}")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains non-finite values")
    return out


def _categorical(values: Sequence, name: str, n: int | None = None) -> np.ndarray:
    out = _vector(values, name, n)
    if np.issubdtype(out.dtype, np.number) and not np.isfinite(out.astype(np.float64)).all():
        raise ValueError(f"{name} contains non-finite values")
    strings = out.astype(str)
    if np.any(strings == ""):
        raise ValueError(f"{name} contains empty values")
    return strings


def blocked_fold_ids(
    block_ids: Sequence,
    *,
    n_folds: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Assign whole blocks to deterministically row-balanced folds."""

    blocks = _categorical(block_ids, "block_ids")
    unique, inverse, counts = np.unique(blocks, return_inverse=True, return_counts=True)
    if n_folds < 2 or unique.size < n_folds:
        raise ValueError("n_folds must be at least two and no larger than the block count")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique.size)
    order = shuffled[np.argsort(-counts[shuffled], kind="stable")]
    fold_sizes = np.zeros(n_folds, dtype=np.int64)
    block_fold = np.empty(unique.size, dtype=np.int64)
    for block in order:
        fold = int(np.argmin(fold_sizes))
        block_fold[block] = fold
        fold_sizes[fold] += counts[block]
    return block_fold[inverse]


def fold_assignment_hash(block_ids: Sequence, fold_ids: Sequence[int]) -> str:
    """Hash the explicit row-level blocking plan for provenance."""

    blocks = _categorical(block_ids, "block_ids")
    folds = _vector(fold_ids, "fold_ids", blocks.size).astype(np.int64)
    digest = sha256()
    for block, fold in zip(blocks, folds):
        digest.update(block.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(fold)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _stable_bucket(value: str, n_buckets: int) -> int:
    digest = sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % n_buckets


def low_level_design(
    *,
    position: Sequence[float],
    kmer: Sequence,
    input_norm: Sequence[float],
    protein_length: Sequence[float],
    sequence_source: Sequence,
    position_degree: int = 3,
    kmer_hash_buckets: int = 64,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build a prespecified finite design for the required low-level covariates.

    ``position`` must be normalized to [0, 1].  K-mers use stable SHA-256
    hashing, avoiding a data-dependent vocabulary.  Sequence source is encoded
    with a deterministic full one-hot basis; ridge regularization handles the
    redundant intercept direction.
    """

    pos = _vector(position, "position").astype(np.float64)
    n = pos.size
    norm_values = _vector(input_norm, "input_norm", n).astype(np.float64)
    lengths = _vector(protein_length, "protein_length", n).astype(np.float64)
    kmers = _categorical(kmer, "kmer", n)
    sources = _categorical(sequence_source, "sequence_source", n)
    if not np.isfinite(pos).all() or np.any((pos < 0) | (pos > 1)):
        raise ValueError("position must be finite and normalized to [0, 1]")
    if not np.isfinite(norm_values).all() or not np.isfinite(lengths).all():
        raise ValueError("input_norm and protein_length must be finite")
    if np.any(lengths <= 0):
        raise ValueError("protein_length must be positive")
    if position_degree < 1 or kmer_hash_buckets < 1:
        raise ValueError("position_degree and kmer_hash_buckets must be positive")

    columns = [pos**power for power in range(1, position_degree + 1)]
    names = [f"position_power_{power}" for power in range(1, position_degree + 1)]
    columns.extend((norm_values, np.log(lengths)))
    names.extend(("input_norm", "log_protein_length"))

    kmer_design = np.zeros((n, kmer_hash_buckets), dtype=np.float64)
    kmer_design[np.arange(n), [_stable_bucket(x, kmer_hash_buckets) for x in kmers]] = 1.0
    columns.extend(kmer_design[:, index] for index in range(kmer_hash_buckets))
    names.extend(f"kmer_hash_{index}" for index in range(kmer_hash_buckets))

    for source in sorted(set(sources)):
        columns.append((sources == source).astype(np.float64))
        names.append(f"source={source}")
    return np.column_stack(columns), tuple(names)


def categorical_design(labels: Sequence) -> tuple[np.ndarray, tuple[str, ...]]:
    """Encode a categorical label using a deterministic reference category."""

    values = _categorical(labels, "labels")
    categories = tuple(sorted(set(values)))
    if len(categories) < 2:
        raise ValueError("a semantic label must contain at least two categories")
    # The lexicographically first category is the prespecified reference.
    columns = [(values == category).astype(np.float64) for category in categories[1:]]
    return np.column_stack(columns), categories


def _ridge_oof(
    target: np.ndarray,
    design: np.ndarray,
    folds: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Return leakage-free ridge predictions for one fixed fold plan."""

    y = _finite_matrix(target, "target")
    x = _finite_matrix(design, "design", y.shape[0])
    fold_ids = _vector(folds, "folds", y.shape[0]).astype(np.int64)
    if alpha <= 0 or not np.isfinite(alpha):
        raise ValueError("ridge alpha must be finite and positive")
    unique_folds = np.unique(fold_ids)
    if unique_folds.size < 2:
        raise ValueError("at least two folds are required")
    predictions = np.empty_like(y)
    for fold in unique_folds:
        test = fold_ids == fold
        train = ~test
        if not np.any(test) or np.count_nonzero(train) < 2:
            raise ValueError("every fold requires held-out rows and at least two training rows")
        mean_x = x[train].mean(axis=0)
        scale_x = x[train].std(axis=0)
        scale_x[scale_x == 0] = 1.0
        train_x = (x[train] - mean_x) / scale_x
        test_x = (x[test] - mean_x) / scale_x
        mean_y = y[train].mean(axis=0)
        centered_y = y[train] - mean_y
        gram = train_x.T @ train_x
        coefficients = np.linalg.solve(
            gram + alpha * np.eye(gram.shape[0]), train_x.T @ centered_y
        )
        predictions[test] = test_x @ coefficients + mean_y
    return predictions


def _effect_metrics(
    target: np.ndarray,
    baseline_prediction: np.ndarray,
    full_prediction: np.ndarray,
    indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if indices is None:
        y = target
        base = baseline_prediction
        full = full_prediction
    else:
        y = target[indices]
        base = baseline_prediction[indices]
        full = full_prediction[indices]
    baseline_mse = np.mean((y - base) ** 2, axis=0)
    full_mse = np.mean((y - full) ** 2, axis=0)
    delta_mse = baseline_mse - full_mse
    denominator = np.mean((y - y.mean(axis=0)) ** 2, axis=0)
    delta_r2 = np.divide(
        delta_mse,
        denominator,
        out=np.zeros_like(delta_mse),
        where=denominator > 0,
    )
    return baseline_mse, full_mse, delta_mse, delta_r2


def within_protein_permutation(
    labels: Sequence,
    protein_ids: Sequence,
    rng: np.random.Generator,
) -> np.ndarray:
    """Shuffle labels independently within each protein, preserving prevalence."""

    values = _vector(labels, "labels")
    _categorical(values, "labels")
    proteins = _categorical(protein_ids, "protein_ids", values.size)
    result = values.copy()
    for protein in np.unique(proteins):
        indices = np.flatnonzero(proteins == protein)
        result[indices] = values[indices][rng.permutation(indices.size)]
    return result


def permutable_row_fraction(labels: Sequence, protein_ids: Sequence) -> float:
    values = _categorical(labels, "labels")
    proteins = _categorical(protein_ids, "protein_ids", values.size)
    count = 0
    for protein in np.unique(proteins):
        indices = np.flatnonzero(proteins == protein)
        if len(set(values[indices])) > 1:
            count += len(indices)
    return float(count / values.size)


def benjamini_hochberg(pvalues: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg q-values with monotonicity enforcement."""

    p = _vector(pvalues, "pvalues").astype(np.float64)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("pvalues must be finite and within [0, 1]")
    order = np.argsort(p, kind="stable")
    ranked = p[order] * p.size / np.arange(1, p.size + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty_like(ranked)
    q[order] = np.clip(ranked, 0.0, 1.0)
    return q


def _cluster_bootstrap(
    target: np.ndarray,
    baseline_prediction: np.ndarray,
    full_prediction: np.ndarray,
    protein_ids: np.ndarray,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    proteins = np.unique(protein_ids)
    indices_by_protein = [np.flatnonzero(protein_ids == protein) for protein in proteins]
    delta_mse = []
    delta_r2 = []
    for _ in range(n_bootstrap):
        sampled = rng.integers(0, len(proteins), len(proteins))
        indices = np.concatenate([indices_by_protein[index] for index in sampled])
        metrics = _effect_metrics(target, baseline_prediction, full_prediction, indices)
        delta_mse.append(metrics[2])
        delta_r2.append(metrics[3])
    boot_mse = np.asarray(delta_mse)
    boot_r2 = np.asarray(delta_r2)
    mse_ci = np.percentile(boot_mse, [2.5, 97.5], axis=0)
    r2_ci = np.percentile(boot_r2, [2.5, 97.5], axis=0)
    return mse_ci, r2_ci, np.std(boot_mse, axis=0, ddof=1)


def run_conditional_semantics(
    representations: Mapping[str, np.ndarray],
    labels: Mapping[str, Sequence],
    covariates: np.ndarray,
    protein_ids: Sequence,
    family_ids: Sequence,
    *,
    feature_names: Mapping[str, Sequence[str]] | None = None,
    n_folds: int = 5,
    n_permutations: int = 1_000,
    n_bootstrap: int = 1_000,
    ridge_alpha: float = 1.0,
    seed: int = 0,
    fdr_alpha: float = 0.05,
    power: float = 0.80,
    require_equal_dimensions: bool = True,
    prospective_standard_errors_delta_mse: Mapping[
        tuple[str, str, str, str], float
    ]
    | None = None,
) -> tuple[ConditionalEffect, ...]:
    """Run protein- and family-blocked conditional tests.

    All representations use the same row-level folds and covariate design.
    When ``require_equal_dimensions`` is true (the confirmatory default),
    sparse, dense, and randomized controls must expose the same number of
    prespecified features. Prospective MDEs are reported only when a complete
    mapping of standard errors from an independent pilot source is supplied;
    standard errors estimated from these analyzed observations are explicitly
    labeled retrospective.
    """

    if not representations or not labels:
        raise ValueError("representations and labels must be non-empty")
    proteins = _categorical(protein_ids, "protein_ids")
    n = proteins.size
    families = _categorical(family_ids, "family_ids", n)
    x = _finite_matrix(covariates, "covariates", n)
    matrices = {
        name: _finite_matrix(values, f"representation {name!r}", n)
        for name, values in representations.items()
    }
    widths = {values.shape[1] for values in matrices.values()}
    if require_equal_dimensions and len(widths) != 1:
        raise ValueError("confirmatory representations must have matched dimensions")
    label_values = {}
    for name, values in labels.items():
        current = _vector(values, f"label {name!r}", n)
        _categorical(current, f"label {name!r}", n)
        label_values[name] = current
    if n_permutations < 1 or n_bootstrap < 2:
        raise ValueError("n_permutations must be positive and n_bootstrap at least two")
    if not 0 < fdr_alpha < 1 or not 0 < power < 1:
        raise ValueError("fdr_alpha and power must be within (0, 1)")

    names = {}
    for representation, values in matrices.items():
        supplied = None if feature_names is None else feature_names.get(representation)
        current = (
            tuple(str(index) for index in range(values.shape[1]))
            if supplied is None
            else tuple(str(value) for value in supplied)
        )
        if len(current) != values.shape[1] or len(set(current)) != len(current):
            raise ValueError(f"feature names for {representation!r} must be unique and complete")
        names[representation] = current

    blockings = {"protein": proteins, "family": families}
    fold_plans = {
        blocking: blocked_fold_ids(blocks, n_folds=n_folds, seed=seed + offset)
        for offset, (blocking, blocks) in enumerate(blockings.items())
    }
    representation_slices = {}
    offset = 0
    for representation, values in matrices.items():
        representation_slices[representation] = slice(offset, offset + values.shape[1])
        offset += values.shape[1]
    combined_target = np.column_stack(tuple(matrices.values()))
    expected_power_keys = {
        (representation, feature, label, blocking)
        for representation, features in names.items()
        for feature in features
        for label in label_values
        for blocking in blockings
    }
    prospective_errors: dict[tuple[str, str, str, str], float] | None = None
    if prospective_standard_errors_delta_mse is not None:
        supplied_keys = set(prospective_standard_errors_delta_mse)
        if supplied_keys != expected_power_keys:
            missing = sorted(expected_power_keys - supplied_keys)
            extra = sorted(supplied_keys - expected_power_keys)
            raise ValueError(
                "prospective standard errors must exactly cover all hypotheses; "
                f"missing={missing[:3]}, extra={extra[:3]}"
            )
        prospective_errors = {}
        for key, raw_value in prospective_standard_errors_delta_mse.items():
            value = float(raw_value)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f"prospective standard error for {key!r} must be finite and positive"
                )
            prospective_errors[key] = value

    # The one-sided effect test asks whether adding the label reduces MSE.
    # Bonferroni planning is deliberately conservative; BH q-values remain the
    # inferential correction for the observed tests.
    per_test_alpha = fdr_alpha / len(expected_power_keys)
    mde_multiplier = norm.ppf(1 - per_test_alpha) + norm.ppf(power)
    baseline_combined = {
        blocking: _ridge_oof(combined_target, x, fold_plans[blocking], ridge_alpha)
        for blocking in blockings
    }

    records: list[ConditionalEffect] = []
    # Separate random streams make each label/block combination reproducible
    # even if another representation is added later.
    seed_sequence = np.random.SeedSequence(seed)
    streams = iter(seed_sequence.spawn(len(blockings) * len(label_values)))
    for blocking, blocks in blockings.items():
        folds = fold_plans[blocking]
        fold_hash = fold_assignment_hash(blocks, folds)
        for label_name, raw_label in label_values.items():
            rng = np.random.default_rng(next(streams))
            label_matrix, _ = categorical_design(raw_label)
            full_design = np.column_stack((x, label_matrix))
            fraction = permutable_row_fraction(raw_label, proteins)
            permutation_statistics = {name: [] for name in matrices}
            full_combined = _ridge_oof(combined_target, full_design, folds, ridge_alpha)
            full_predictions = {}
            observed_metrics = {}
            for representation, target in matrices.items():
                columns = representation_slices[representation]
                baseline = baseline_combined[blocking][:, columns]
                full = full_combined[:, columns]
                full_predictions[representation] = full
                observed_metrics[representation] = _effect_metrics(target, baseline, full)

            if fraction > 0:
                for _ in range(n_permutations):
                    permuted = within_protein_permutation(raw_label, proteins, rng)
                    permuted_matrix, _ = categorical_design(permuted)
                    permuted_design = np.column_stack((x, permuted_matrix))
                    permuted_prediction = _ridge_oof(
                        combined_target, permuted_design, folds, ridge_alpha
                    )
                    for representation, target in matrices.items():
                        columns = representation_slices[representation]
                        statistic = _effect_metrics(
                            target,
                            baseline_combined[blocking][:, columns],
                            permuted_prediction[:, columns],
                        )[2]
                        permutation_statistics[representation].append(statistic)
            else:
                for representation, target in matrices.items():
                    statistic = observed_metrics[representation][2]
                    permutation_statistics[representation] = [statistic] * n_permutations

            for representation, target in matrices.items():
                baseline_mse, full_mse, delta_mse, delta_r2 = observed_metrics[representation]
                null = np.asarray(permutation_statistics[representation])
                pvalues = (1 + np.sum(null >= delta_mse[None, :], axis=0)) / (n_permutations + 1)
                bootstrap_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
                mse_ci, r2_ci, bootstrap_se = _cluster_bootstrap(
                    target,
                    baseline_combined[blocking][:, representation_slices[representation]],
                    full_predictions[representation],
                    proteins,
                    n_bootstrap=n_bootstrap,
                    rng=bootstrap_rng,
                )
                for feature_index, feature in enumerate(names[representation]):
                    power_key = (representation, feature, label_name, blocking)
                    prospective_mde = (
                        None
                        if prospective_errors is None
                        else float(prospective_errors[power_key] * mde_multiplier)
                    )
                    records.append(
                        ConditionalEffect(
                            representation,
                            feature,
                            label_name,
                            blocking,
                            n,
                            len(set(proteins)),
                            len(set(blocks)),
                            float(baseline_mse[feature_index]),
                            float(full_mse[feature_index]),
                            float(delta_mse[feature_index]),
                            float(delta_r2[feature_index]),
                            float(pvalues[feature_index]),
                            1.0,
                            (float(mse_ci[0, feature_index]), float(mse_ci[1, feature_index])),
                            (float(r2_ci[0, feature_index]), float(r2_ci[1, feature_index])),
                            float(bootstrap_se[feature_index]),
                            float(bootstrap_se[feature_index] * mde_multiplier),
                            prospective_mde,
                            fraction,
                            fraction == 0.0,
                            fold_hash,
                        )
                    )

    qvalues = benjamini_hochberg([record.permutation_pvalue for record in records])
    return tuple(
        replace(
            record,
            qvalue=float(qvalue),
        )
        for record, qvalue in zip(records, qvalues)
    )
