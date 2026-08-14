"""How many dimensions an activation cloud actually occupies, where a dictionary is fitted.

Why this module exists
======================

R2.4's basis-adequacy gate B2 asks for more live dictionary latents per layer
than the backbone is wide -- 4,096 on the ``Llama-2-7b-hf`` / ``ProLLaMA_Stage_1``
lineage -- on the premise that a sparse over-complete dictionary decomposes a
space into more features than that space has dimensions. The premise names
``d_model`` as the dimension of the space. ``d_model`` is the dimension of the
*ambient* space the activations are written in, which is an upper bound on the
dimension of the space they occupy and not a measurement of it.

That distinction decides how EXP-R2-194's B2 reading should be read. Its protein
cells carry 1,634 and 2,188 live latents per layer against 4,096 and were
recorded as under-complete bases, with the shortfall attributed to a training
recipe -- revival budget, sparsity, width or token budget. The same two cells are
also the ones whose cross-checkpoint activations a ridge map aligns *best*
(residual 0.400 in protein mode against 0.684 in text). More alignable and less
able to support live latents is the signature a low-dimensional activation cloud
would leave, and under that reading B2 is unattainable in principle in protein
mode and is measuring data geometry rather than dictionary adequacy. Nothing on
disk separated the two, because no stage had ever measured the rank of the cloud
itself.

This module is that measurement, and only that: the centred covariance of an
activation cloud, accumulated in float64 over sampled token positions, and the
handful of scale-free summaries of its eigenvalue spectrum that a rank claim can
be read from. It fits no map, trains nothing and takes no decision; the stage
that calls it (``scripts/transfer/30_activation_spectrum.py``) owns the corpus
draw and the pre-registered rule.

What the summaries mean, and what they do not
=============================================

``r95`` / ``r99`` / ``r999``
    The smallest number of principal directions carrying 95%, 99% and 99.9% of
    the total variance. These are **variance-based effective dimensions and not
    algebraic ranks**: a direction beyond ``r99`` carries under 1% of the
    variance between it and every other direction beyond ``r99``, which is a
    statement about what a fit at a finite token budget can resolve, not a proof
    that the direction is empty. ``r999`` travels beside ``r99`` so a reader can
    see how much the choice of cut moves the answer.
``participation_ratio``
    ``(sum lambda)^2 / sum lambda^2``. Dominated by the largest eigenvalues, so
    it falls fast when a few directions carry most of the variance and is the
    statistic Appendix B rule 11 is written about.
``effective_rank``
    ``exp`` of the Shannon entropy of the normalised eigenvalues. Sensitive to
    the whole spectrum rather than to its head, so it and the participation ratio
    disagreeing is informative about spectrum shape rather than a contradiction.

Every one of these is invariant to the scale of the covariance, which is what
makes them comparable across layers whose activation norm differs by orders of
magnitude, and across two modes whose corpora are not the same corpus.

The ceiling, which is not optional
==================================

A covariance estimated from ``N`` samples has rank at most ``min(N, d)``, so a
spectrum measured at ``N`` below ``d`` reports the sampling budget and not the
data. :func:`spectrum_statistics` therefore reports ``sample_rank_ceiling``
beside every rank it returns, and the calling stage refuses a budget below a
declared multiple of ``d``. Centring costs one further degree of freedom, which
is why the ceiling is reported rather than compared against.

Two controls, and why the obvious form of the second one is not a control
=========================================================================

:func:`isotropic_control_spectrum` draws ``N`` isotropic Gaussian samples whose
trace matches the measured covariance's and passes them through the same
accumulator and the same summary. It is an instrument check: an estimator that
cannot return ``r99`` near ``d`` on data that is exactly full-rank is broken, and
no reading of a real spectrum may be taken until it does. It also prices the
finite-sample deflation at this ``N`` and ``d`` directly -- at ``N/d = 16`` the
Marchenko-Pastur edge is wide enough that ``r99`` sits a percent or two below
``d`` on genuinely isotropic data, and a reader needs that number to calibrate
every other one.

:func:`coordinate_independent_spectrum` is the second control, and it is
deliberately **not** the permutation the phrase "token-shuffled" first suggests.
Permuting whole activation vectors across token positions leaves the covariance
of the sampled set exactly unchanged -- a covariance does not see the order of
its samples -- so that permutation is a no-op and cannot be a control. The
permutation that is one is per-coordinate: independently permute each of the
``d`` coordinates across the sampled positions. That preserves every
coordinate's marginal variance exactly and destroys every correlation between
coordinates, and the covariance it induces is exactly ``diag(C)``. Its spectrum
is therefore the sorted per-coordinate variances, computed here in closed form.
A realised permutation would add off-diagonal terms of order ``N^{-1/2}``; those
are not reproduced, so this control carries no finite-sample noise while the real
spectrum does, and the isotropic control above is what prices that gap. What it
answers is the question that matters: whether a low ``r99`` is anisotropy of the
individual coordinates, which a dictionary basis is free to represent, or
correlation between them, which is what makes the cloud low-dimensional.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

#: Cumulative-mass curve reported per layer, at ranks spaced by powers of two.
#: Thirteen numbers per layer, which is what lets a reader re-read any threshold
#: they like instead of only the three this module names.
CUMULATIVE_MASS_RANKS: tuple[int, ...] = (
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
)


class CovarianceAccumulator:
    """Streaming centred covariance of a stack of per-layer activation clouds.

    ``update`` takes ``(n_layers, n_tokens, d_model)`` and accumulates in
    float64; ``covariance_at`` returns one layer's ``(d_model, d_model)``
    unbiased centred covariance. The layer axis is kept resident because the
    activations arrive one batch at a time for every layer at once, and a
    per-layer pass would mean one forward pass per layer.

    Second moments are accumulated about a **fixed shift** taken from the first
    update rather than about the origin. ``Cov(x) = Cov(x - c)`` exactly for any
    constant ``c``, and a feed-forward output on this family carries coordinates
    whose mean is large against their spread, where ``sum(x x^T) - n mu mu^T``
    cancels most of its significant digits before the subtraction is done. The
    shift costs one subtraction per update and removes the cancellation; the
    residual mean it leaves is reported as ``mean_shift_residual`` so the claim
    is checkable rather than asserted.
    """

    def __init__(
        self, *, n_layers: int, d_model: int, device: torch.device | str
    ) -> None:
        if n_layers < 1 or d_model < 1:
            raise ValueError("a covariance needs at least one layer and one dimension")
        self.n_layers = int(n_layers)
        self.d_model = int(d_model)
        self._device = torch.device(device)
        self._count = 0
        self._shift: torch.Tensor | None = None
        self._sum = torch.zeros(
            (self.n_layers, self.d_model), dtype=torch.float64, device=self._device
        )
        self._second = torch.zeros(
            (self.n_layers, self.d_model, self.d_model),
            dtype=torch.float64,
            device=self._device,
        )

    @property
    def n_samples(self) -> int:
        return self._count

    def update(self, activations: torch.Tensor) -> None:
        """Add ``(n_layers, n_tokens, d_model)`` token positions to the estimate."""

        if activations.ndim != 3:
            raise ValueError(
                "expected (n_layers, n_tokens, d_model) activations, got shape "
                f"{tuple(activations.shape)}"
            )
        n_layers, n_tokens, d_model = activations.shape
        if n_layers != self.n_layers or d_model != self.d_model:
            raise ValueError(
                f"this accumulator was built for {self.n_layers} layers of "
                f"{self.d_model} dimensions and was handed {n_layers} of {d_model}"
            )
        if n_tokens == 0:
            return
        block = activations.to(device=self._device, dtype=torch.float64)
        if self._shift is None:
            self._shift = block.mean(dim=1)
        block = block - self._shift[:, None, :]
        self._sum += block.sum(dim=1)
        self._second.baddbmm_(block.transpose(1, 2), block)
        self._count += int(n_tokens)

    def covariance_at(self, layer: int) -> torch.Tensor:
        """One layer's unbiased centred covariance, symmetrised, in float64."""

        if self._count < 2:
            raise RuntimeError(
                f"a covariance needs at least two samples and this accumulator has "
                f"{self._count}"
            )
        if not 0 <= layer < self.n_layers:
            raise IndexError(f"layer {layer} is outside 0..{self.n_layers - 1}")
        mean = self._sum[layer] / self._count
        covariance = (
            self._second[layer] - self._count * torch.outer(mean, mean)
        ) / (self._count - 1)
        return 0.5 * (covariance + covariance.T)

    def mean_shift_residual(self) -> float:
        """Largest coordinate of the mean that survived the shift, over all layers.

        Small against the activation scale means the shift did its job. Reported
        rather than checked, because there is no threshold at which a large value
        would be wrong -- it would only mean the float64 subtraction had more
        cancellation to absorb than intended.
        """

        if self._count < 1:
            return 0.0
        return float((self._sum / self._count).abs().max())


def spectrum_statistics(
    eigenvalues: torch.Tensor | np.ndarray, *, n_samples: int, d_model: int
) -> dict[str, Any]:
    """Scale-free summaries of one covariance spectrum.

    ``eigenvalues`` may arrive in any order and may carry the small negative
    values a float64 eigensolver returns for a numerically rank-deficient
    matrix. Those are clipped to zero and their total magnitude is reported as
    ``negative_eigenvalue_mass``: it is the estimator's own error bar, and a
    value that is not negligible against ``total_variance`` means the spectrum
    should not be read.
    """

    values = np.asarray(
        eigenvalues.detach().cpu().numpy()
        if isinstance(eigenvalues, torch.Tensor)
        else eigenvalues,
        dtype=np.float64,
    ).reshape(-1)
    if values.size != d_model:
        raise ValueError(
            f"expected {d_model} eigenvalues and got {values.size}; a spectrum "
            "shorter than the ambient dimension is a different measurement"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("the eigensolver returned a non-finite eigenvalue")
    negative_mass = float(np.abs(values[values < 0.0].sum()))
    ordered = np.sort(np.clip(values, 0.0, None))[::-1]
    total = float(ordered.sum())
    if total <= 0.0:
        raise ValueError("a covariance with no variance has no spectrum to summarise")

    cumulative = np.cumsum(ordered)

    def rank_at(quantile: float) -> int:
        reached = int(np.searchsorted(cumulative, quantile * total, side="left")) + 1
        return min(reached, d_model)

    positive = ordered[ordered > 0.0]
    weights = positive / total
    entropy = float(-(weights * np.log(weights)).sum())
    return {
        "r95": rank_at(0.95),
        "r99": rank_at(0.99),
        "r999": rank_at(0.999),
        "participation_ratio": float(total**2 / float((ordered**2).sum())),
        "effective_rank": float(np.exp(entropy)),
        "total_variance": total,
        "top_eigenvalue_share": float(ordered[0] / total),
        "n_positive_eigenvalues": int(positive.size),
        "negative_eigenvalue_mass": negative_mass,
        "n_samples": int(n_samples),
        "sample_rank_ceiling": int(min(n_samples, d_model)),
        "cumulative_mass": {
            str(rank): float(cumulative[rank - 1] / total)
            for rank in CUMULATIVE_MASS_RANKS
            if rank <= d_model
        },
    }


def isotropic_control_spectrum(
    *,
    total_variance: float,
    d_model: int,
    n_samples: int,
    chunk: int,
    device: torch.device | str,
    generator: torch.Generator,
) -> dict[str, Any]:
    """The same estimator, on data whose true covariance is exactly full rank.

    ``n_samples`` isotropic Gaussian vectors with per-coordinate variance
    ``total_variance / d_model`` -- so the trace matches the covariance this
    control is reported beside -- streamed through :class:`CovarianceAccumulator`
    in blocks of ``chunk`` and summarised by :func:`spectrum_statistics`.

    An estimator that does not return ``r99`` close to ``d_model`` here is
    broken, and the stage that calls this refuses to read any real spectrum until
    it does. The three rank statistics are invariant to the scale of the
    covariance, so trace-matching changes the eigenvalues' units and none of the
    reported ranks; it is done anyway, per layer, so that the control is run at
    the same numerical scale as the measurement it certifies rather than at a
    convenient one.
    """

    if total_variance <= 0.0:
        raise ValueError("an isotropic control needs a positive trace to match")
    if chunk < 1 or n_samples < 2:
        raise ValueError("an isotropic control needs at least two samples")
    scale = (float(total_variance) / float(d_model)) ** 0.5
    accumulator = CovarianceAccumulator(n_layers=1, d_model=d_model, device=device)
    drawn = 0
    while drawn < n_samples:
        rows = min(chunk, n_samples - drawn)
        noise = torch.randn(
            (1, rows, d_model),
            generator=generator,
            device=generator.device,
            dtype=torch.float64,
        )
        accumulator.update(noise * scale)
        drawn += rows
    eigenvalues = torch.linalg.eigvalsh(accumulator.covariance_at(0))
    return spectrum_statistics(eigenvalues, n_samples=n_samples, d_model=d_model)


def coordinate_independent_spectrum(
    covariance: torch.Tensor, *, n_samples: int, d_model: int
) -> dict[str, Any]:
    """The spectrum an independent per-coordinate permutation of the sample induces.

    Exactly ``diag(covariance)``, sorted -- see this module's docstring for why
    that, and not a permutation of whole activation vectors, is the control.
    """

    if covariance.shape != (d_model, d_model):
        raise ValueError(
            f"expected a {d_model}x{d_model} covariance, got "
            f"{tuple(covariance.shape)}"
        )
    return spectrum_statistics(
        torch.diagonal(covariance), n_samples=n_samples, d_model=d_model
    )


def sample_positions(
    content_mask: torch.Tensor, *, per_record: int, generator: torch.Generator
) -> tuple[torch.Tensor, list[int], list[int]]:
    """Which flattened token positions to take from one batch, capped per record.

    ``content_mask`` is the ``(batch, length)`` mask
    :meth:`src.transfer.replaceable.ReplaceableModel.content_mask` returns, and
    the indices this function produces index the token axis of
    ``17_train_transcoder.flatten``'s output -- the scored positions of the batch
    in row-major order.

    A covariance over ``N`` positions taken from a handful of records estimates
    the geometry of those records, because positions within one protein or one
    document are strongly correlated. So the budget is spent as a hard cap of
    ``per_record`` positions per record, drawn uniformly without replacement
    from within that record, and a record carrying fewer than ``per_record``
    scored positions contributes **nothing** rather than contributing what it
    has: an equal number per record is what keeps the sample a sample of records
    and not a sample of the longest records. The refused rows are returned so the
    caller can report how much of the draw that cost.

    Returns ``(indices, used_rows, short_rows)``.
    """

    if per_record < 1:
        raise ValueError("per_record must be positive")
    if content_mask.ndim != 2:
        raise ValueError(
            f"expected a (batch, length) content mask, got shape "
            f"{tuple(content_mask.shape)}"
        )
    counts = content_mask.detach().cpu().to(torch.long).sum(dim=1)
    offsets = torch.cat(
        [torch.zeros(1, dtype=torch.long), counts.cumsum(0)[:-1]]
    )
    chosen: list[torch.Tensor] = []
    used: list[int] = []
    short: list[int] = []
    for row in range(counts.numel()):
        available = int(counts[row])
        if available < per_record:
            short.append(row)
            continue
        picked = torch.randperm(available, generator=generator)[:per_record]
        chosen.append(int(offsets[row]) + torch.sort(picked).values)
        used.append(row)
    indices = (
        torch.cat(chosen) if chosen else torch.zeros(0, dtype=torch.long)
    )
    return indices, used, short
