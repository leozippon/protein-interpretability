"""Generation contrasts preserve the shared seed-batch randomization unit."""
from __future__ import annotations

import numpy as np

from .statistics import MINIMUM_BOOTSTRAP_UNITS


def paired_generation_contrast(cells, weights, *, seed=20260923, resamples=2000,
                               batch_size=8):
    """Bootstrap common seed batches jointly, never arms or attempts separately."""
    metrics = ('any_profile_hit', 'native_complete',
               'native_complete_with_profile', 'native_complete_without_profile')
    if not weights:
        return {metric: {'difference_fraction': 0.,
                         'paired_seed_batch_percentile_95': [0., 0.],
                         'interval_status': 'exact_parameter_identity_not_sampling_inference'}
                for metric in metrics}
    arrays = {}
    expected = None
    for key in weights:
        rows = cells[key]
        indices = [row['attempt_index'] for row in rows]
        if sorted(indices) != list(range(len(rows))) or len(rows) % batch_size:
            raise ValueError('generation attempt indices must form complete seed batches')
        if expected is not None and len(rows) != expected:
            raise ValueError('generation seed-batch support mismatch')
        expected = len(rows)
        rows = sorted(rows, key=lambda row: row['attempt_index'])
        if any(type(row.get(m)) is not bool for row in rows
               for m in ('native_complete', 'any_profile_hit')):
            raise ValueError('missing or nonboolean generation outcome')
        complete = np.array([row['native_complete'] for row in rows], dtype=float)
        profile = np.array([row['any_profile_hit'] for row in rows], dtype=float)
        arrays[key] = dict(zip(metrics, (profile, complete, complete * profile,
                                        complete * (1 - profile))))
    n_batches = expected // batch_size
    if n_batches < 1:
        raise ValueError('no generation seed batches')
    draw = np.random.default_rng(seed).integers(0, n_batches, (resamples, n_batches))
    result = {}
    for metric in metrics:
        paired = sum(weight * arrays[key][metric].reshape(n_batches, batch_size).mean(axis=1)
                     for key, weight in weights.items())
        # The seed batch is the resampling unit, so the shared floor governs this
        # interval; imported rather than restated, so that a change to the floor
        # reaches it.
        interval = (np.quantile(paired[draw].mean(axis=1), [.025, .975]).tolist()
                    if n_batches >= MINIMUM_BOOTSTRAP_UNITS else None)
        result[metric] = {'difference_fraction': float(paired.mean()),
                          'paired_seed_batch_percentile_95': interval,
                          'n_seed_batches': n_batches, 'attempts_per_batch': batch_size,
                          'resamples': resamples,
                          'interval_status': ('exploratory' if interval else
                                              f'fewer_than_{MINIMUM_BOOTSTRAP_UNITS}_seed_batches')}
    return result
