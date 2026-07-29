"""Causal footprint of a sublayer pathway, measured without any dictionary.

Transcoder and cross-layer-transcoder circuit tracing decomposes the MLP
sublayer, so everything such a method can explain about next-token prediction
has to travel through MLP outputs. The ceiling on the method is therefore a
property of the model, not of the dictionary, and it is directly measurable:
replace the output of a declared set of sublayers with a constant and record
what the model loses. That loss is the denominator every downstream recovery
fraction is divided by, which is why it is measured here on its own.

Three design choices are deliberate.

*Scopes are explicit.* One MLP at mid-depth, an eight-layer window and the whole
MLP pathway are different estimands whose footprints differ by two orders of
magnitude. The scope is therefore named, resolved into concrete submodule
targets, and recorded alongside the number it produced.

*Baselines are first-class.* Ablating to the evaluation-cohort mean and ablating
to zero remove different things -- the second also removes the sublayer's
constant bias direction -- and the measured denominator moves substantially
between them. The baseline is recorded with its provenance rather than assumed.

*Intervals are sequence-clustered.* Tokens inside one sequence are not
independent, so resampling tokens would understate the interval. The resampling
unit is the sequence.

Scoring positions are exactly those where the attention mask is valid and the
next token is valid. No modality-specific position filtering is applied, so for
ZymCTRL the EC-conditioning prefix is scored like any other context; that keeps
one scoring rule across the panel and is declared in the output.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
import torch

from .arms import Arm, Cohort, tokenize_batch
from .scoring import (
    aggregate_variant,
    per_sequence_scores,
    source_layers_for_target,
)

#: Submodule outputs this module knows how to intercept. ``block`` is the whole
#: transformer block output and is the reference scope against which a single
#: pathway's footprint is read.
SUBMODULE_KINDS = ("attn", "block", "mlp")

#: Scope families. ``single`` and ``window`` are anchored at a layer and can be
#: swept over depth; ``all`` covers every layer.
SCOPE_FAMILIES = ("all", "single", "window")

#: Ablation-baseline identifiers. Both are constants broadcast over positions.
BASELINE_KINDS = ("cohort_mean", "zero")

BASELINE_DEFINITIONS = {
    "cohort_mean": (
        "mean of the submodule output over every attention-mask-valid position "
        "of the evaluation cohort, accumulated in float64"
    ),
    "zero": "exact zero vector; removes the submodule's constant bias direction as well",
}

#: Production P0-2b denominator guards, in nats per token. An estimand whose
#: mean-ablation footprint falls below these cannot support a recovery-fraction
#: gate at all, whatever the dictionary does.
P0_2B_MINIMUM_CE_DELTA_NATS = 0.05
P0_2B_MINIMUM_KL_NATS = 0.01

#: Estimators for the context-free baseline that normalises every share.
#: ``disjoint`` fits a unigram model on a held-out corpus and evaluates it on
#: the scored targets. ``plugin`` fits it on the scored targets themselves; that
#: is biased low, severely so for the 50k-token arms, and inflates every share
#: computed against it, so it exists only as an explicit opt-in diagnostic.
UNIGRAM_ESTIMATORS = ("disjoint", "plugin")

#: Additive smoothing for the held-out unigram model, over the declared
#: vocabulary. Laplace rather than a tuned constant: it is the standard choice,
#: it needs no justification from the data, and it errs upwards, which makes the
#: resulting share conservative.
LAPLACE_SMOOTHING = 1.0

SCHEMA_VERSION_MEASUREMENT = "r2_transfer_pathway_measurement_v1"
SCHEMA_VERSION_BOOTSTRAP = "r2_transfer_pathway_cluster_bootstrap_v1"


class Target(NamedTuple):
    """One intercepted submodule output."""

    kind: str
    layer: int


# --------------------------------------------------------------------- scopes


@dataclass(frozen=True)
class AblationScope:
    """A named set of submodule outputs to replace with a baseline constant.

    The scope is stored as a specification rather than a layer list so that one
    scope object can be resolved against models of different depth; ``resolve``
    is the only place a layer index is checked against a model.
    """

    name: str
    family: str
    submodules: tuple[str, ...]
    anchor_layer: int | None
    width: int | None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ablation scope must be named")
        if self.family not in SCOPE_FAMILIES:
            raise ValueError(f"unknown scope family {self.family!r}; known {SCOPE_FAMILIES}")
        if not self.submodules or any(kind not in SUBMODULE_KINDS for kind in self.submodules):
            raise ValueError(f"scope {self.name!r} names unknown submodules {self.submodules}")
        if len(set(self.submodules)) != len(self.submodules):
            raise ValueError(f"scope {self.name!r} repeats a submodule kind")
        if self.anchor_layer is not None and self.anchor_layer < 0:
            raise ValueError(f"scope {self.name!r} has a negative anchor layer")
        if (self.anchor_layer is None) != (self.family == "all"):
            raise ValueError(f"scope {self.name!r} anchor does not match family {self.family!r}")
        if (self.width is not None) != (self.family == "window"):
            raise ValueError(f"scope {self.name!r} width does not match family {self.family!r}")
        if self.width is not None and self.width < 1:
            raise ValueError(f"scope {self.name!r} has a non-positive window width")

    def resolve(self, n_layer: int) -> tuple[Target, ...]:
        """Concrete ``(submodule kind, layer)`` targets for a model of this depth."""

        if n_layer < 1:
            raise ValueError(f"scope {self.name!r}: invalid layer count {n_layer}")
        if self.anchor_layer is None:
            layers: tuple[int, ...] = tuple(range(n_layer))
        elif self.anchor_layer >= n_layer:
            raise ValueError(
                f"scope {self.name!r} anchors layer {self.anchor_layer} "
                f"in a {n_layer}-layer model"
            )
        elif self.width is None:
            layers = (self.anchor_layer,)
        else:
            # Same convention as a windowed transcoder's source window: the
            # window ends at the anchor layer and is clipped at layer 0. The
            # clip is visible in the scope name, which carries both endpoints.
            layers = source_layers_for_target(
                self.anchor_layer, n_layers=n_layer, window=self.width
            )
        return tuple(
            sorted(Target(kind, layer) for kind in self.submodules for layer in layers)
        )


def mlp_single(layer: int) -> AblationScope:
    """The P0-2b estimand: one MLP output at one layer."""

    return AblationScope(f"mlp_single_l{layer}", "single", ("mlp",), layer, None)


def mlp_window(layer: int, width: int) -> AblationScope:
    """The MLP outputs a windowed transcoder of this width spans, ending at ``layer``."""

    if width < 1:
        raise ValueError("window width must be positive")
    low = max(0, layer - width + 1)
    return AblationScope(f"mlp_window{width}_l{low}_{layer}", "window", ("mlp",), layer, width)


def mlp_all() -> AblationScope:
    """Every MLP output: the total budget available to MLP-decomposing methods."""

    return AblationScope("mlp_all", "all", ("mlp",), None, None)


def attn_single(layer: int) -> AblationScope:
    return AblationScope(f"attn_single_l{layer}", "single", ("attn",), layer, None)


def attn_window(layer: int, width: int) -> AblationScope:
    if width < 1:
        raise ValueError("window width must be positive")
    low = max(0, layer - width + 1)
    return AblationScope(f"attn_window{width}_l{low}_{layer}", "window", ("attn",), layer, width)


def attn_all() -> AblationScope:
    """Every attention output: the pathway MLP-decomposing methods do not model."""

    return AblationScope("attn_all", "all", ("attn",), None, None)


def mlp_and_attn_all() -> AblationScope:
    """Both pathways at once, bounding what any sublayer decomposition can reach."""

    return AblationScope("mlp_and_attn_all", "all", ("attn", "mlp"), None, None)


def resid_block(layer: int) -> AblationScope:
    """The whole block output at one layer: the reference a pathway is read against."""

    return AblationScope(f"resid_block_l{layer}", "single", ("block",), layer, None)


def submodule_for(arm: Arm, target: Target) -> torch.nn.Module:
    """The live module whose output ``target`` names."""

    if target.layer < 0 or target.layer >= arm.n_layer:
        raise ValueError(f"{arm.name}: layer {target.layer} outside 0..{arm.n_layer - 1}")
    if target.kind == "mlp":
        return arm.mlp(target.layer)
    if target.kind == "attn":
        return arm.attention(target.layer)
    if target.kind == "block":
        return arm.blocks()[target.layer]
    raise ValueError(f"unknown submodule kind {target.kind!r}; known {SUBMODULE_KINDS}")


def _primary_tensor(output: Any, label: str, width: int) -> torch.Tensor:
    """The activation carried by a submodule output, tensor or tuple alike.

    ProGen2 blocks and GPT-2 attention return tuples; GPT-2 and ProGen2 MLPs
    return bare tensors. Both forms are legitimate and both must be handled, but
    anything else is a panel change and must fail rather than be guessed at.
    """

    tensor = output[0] if isinstance(output, tuple) else output
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{label}: submodule output is not a tensor or tuple of tensors")
    if tensor.ndim != 3 or tensor.shape[-1] != width:
        raise ValueError(
            f"{label}: expected a [batch, token, {width}] output, got {tuple(tensor.shape)}"
        )
    return tensor


# ------------------------------------------------------------------- batching


@dataclass(frozen=True)
class ScoredBatch:
    """One tokenised batch with the positions that will be scored."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    target_mask: torch.Tensor
    sequence_indices: tuple[int, ...]


def prepare_batches(
    arm: Arm, cohort: Cohort, *, max_len: int, batch_size: int
) -> list[ScoredBatch]:
    """Tokenise a cohort in the arm's native input format, once per measurement.

    The batches are reused for every scope so that the clean reference, the
    ablation baseline and every ablated variant are scored on identical
    positions; recomputing them per scope would allow them to drift apart.
    """

    if max_len < 2 or batch_size < 1:
        raise ValueError("max_len must be at least two tokens and batch_size positive")
    texts = cohort.input_strings(arm)
    if not texts:
        raise ValueError(f"{arm.name}: cohort {cohort.name!r} is empty")
    batches: list[ScoredBatch] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        ids, mask = tokenize_batch(arm, chunk, max_len)
        if ids.shape[1] < 2:
            raise ValueError(
                f"{arm.name}: batch at sequence {start} tokenises to fewer than two tokens"
            )
        ids = ids.to(arm.device)
        mask = mask.to(arm.device)
        target_mask = mask[:, 1:].bool() & mask[:, :-1].bool()
        empty = torch.nonzero(target_mask.sum(dim=1) < 1, as_tuple=False).flatten()
        if empty.numel() > 0:
            raise ValueError(
                f"{arm.name}: sequences {[start + int(i) for i in empty]} have no scored targets"
            )
        batches.append(
            ScoredBatch(
                input_ids=ids,
                attention_mask=mask,
                target_mask=target_mask,
                sequence_indices=tuple(range(start, start + len(chunk))),
            )
        )
    return batches


#: Corpora a protein cohort can be drawn from. The production P0-2b
#: qualification evaluated on the EC-labelled corpus, so anything claiming to
#: explain that qualification must use ``ec_labelled_swissprot`` for every
#: protein arm; the corpora are not interchangeable and mixing them silently
#: moves clean cross-entropy by more than a nat per token.
PROTEIN_COHORT_SOURCES = ("ec_labelled_swissprot", "plain_swissprot")
TEXT_COHORT_SOURCE = "openwebtext_screen"
COHORT_SOURCES = (*PROTEIN_COHORT_SOURCES, TEXT_COHORT_SOURCE)


def cohort_composition(cohort: Cohort, *, source: str) -> dict[str, Any]:
    """Composition diagnostics the cohort's digest cannot express.

    The corpus a cohort came from is recorded explicitly because ``Cohort``
    hashes only its records: two cohorts from different corpora do get different
    digests, but nothing in the digest says which corpus, and comparing an
    EC-labelled run against a plain Swiss-Prot run is a category error.

    ``arms.protein_cohort`` also takes records in file order, and the
    EC-labelled source is grouped by enzyme family, so a small pool drawn from
    the head of that file is a set of near-clonal homologues rather than a
    representative sample. A near-clonal cohort is unusually predictable, which
    shrinks the context information every share is divided by. The realised
    label diversity is therefore recorded too, and drawing a pool much larger
    than the per-seed subsample is how it is kept high.
    """

    if source not in COHORT_SOURCES:
        raise ValueError(f"unknown cohort source {source!r}; known {COHORT_SOURCES}")
    if not cohort.records:
        raise ValueError(f"cohort {cohort.name!r} is empty")
    if (cohort.kind == "text") != (source == TEXT_COHORT_SOURCE):
        raise ValueError(f"cohort kind {cohort.kind!r} does not match source {source!r}")
    labels = cohort.metadata.get("ec_labels")
    if source == "ec_labelled_swissprot" and labels is None:
        raise ValueError("an EC-labelled cohort must carry its EC labels")
    lengths = sorted(len(record) for record in cohort.records)
    return {
        "source": source,
        "name": cohort.name,
        "kind": cohort.kind,
        "digest": cohort.digest,
        "sequences": len(cohort.records),
        "distinct_conditioning_labels": None if labels is None else len(set(labels)),
        "symbols_min": lengths[0],
        "symbols_median": lengths[len(lengths) // 2],
        "symbols_max": lengths[-1],
    }


def subsample_cohort(cohort: Cohort, size: int, seed: int) -> Cohort:
    """A seeded sub-cohort, carrying its conditioning metadata with it.

    A seed is meaningful here because the ``cohort_mean`` baseline is estimated
    on the evaluation cohort itself, so resampling the cohort resamples both the
    intervention and the thing it is scored on. The returned cohort has its own
    content digest, which is what makes two seeds distinguishable in the record.
    """

    if size < 1 or size > len(cohort):
        raise ValueError(f"cannot draw {size} of {len(cohort)} sequences from {cohort.name!r}")
    generator = np.random.default_rng(seed)
    indices = sorted(int(i) for i in generator.choice(len(cohort), size=size, replace=False))
    metadata: dict[str, Any] = {}
    labels = cohort.metadata.get("ec_labels")
    if labels is not None:
        if len(labels) != len(cohort.records):
            raise ValueError(f"cohort {cohort.name!r}: EC labels do not align with records")
        metadata["ec_labels"] = [labels[i] for i in indices]
    return Cohort(
        name=f"{cohort.name}_n{size}_seed{seed}",
        kind=cohort.kind,
        records=[cohort.records[i] for i in indices],
        min_symbols=cohort.min_symbols,
        max_symbols=cohort.max_symbols,
        metadata=metadata,
    )


# ------------------------------------------------------------------ baselines


@dataclass(frozen=True)
class BaselineBank:
    """Replacement constants for a set of targets, with their provenance."""

    kind: str
    vectors: Mapping[Target, torch.Tensor]
    provenance: dict[str, Any]

    def vector(self, target: Target) -> torch.Tensor:
        if target not in self.vectors:
            raise KeyError(
                f"baseline {self.kind!r} holds no vector for {target.kind} layer {target.layer}"
            )
        return self.vectors[target]

    def covers(self, targets: Sequence[Target]) -> None:
        missing = [target for target in targets if target not in self.vectors]
        if missing:
            raise KeyError(f"baseline {self.kind!r} is missing targets {missing}")


class _MeanAccumulator:
    """Masked float64 accumulation of submodule outputs over a cohort."""

    def __init__(self, arm: Arm, targets: Sequence[Target]) -> None:
        self.arm = arm
        self.targets = tuple(dict.fromkeys(targets))
        self.sums = {
            target: torch.zeros(arm.d_model, dtype=torch.float64, device=arm.device)
            for target in self.targets
        }
        self.tokens = 0
        self.keep: torch.Tensor | None = None
        self.fired: dict[Target, int] = {target: 0 for target in self.targets}
        self._handles: list[Any] = []

    def __enter__(self) -> _MeanAccumulator:
        for target in self.targets:
            self._handles.append(
                submodule_for(self.arm, target).register_forward_hook(self._hook(target))
            )
        return self

    def __exit__(self, *_exception: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _hook(self, target: Target):
        def hook(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
            keep = self.keep
            if keep is None:
                raise RuntimeError("baseline hook fired outside a scored batch")
            tensor = _primary_tensor(
                output, f"{self.arm.name} {target.kind} layer {target.layer}", self.arm.d_model
            )
            self.sums[target] += tensor[keep].double().sum(dim=0)
            self.fired[target] += 1

        return hook


@torch.inference_mode()
def build_baseline(
    arm: Arm,
    batches: Sequence[ScoredBatch],
    targets: Sequence[Target],
    *,
    kind: str,
    cohort_digest: str,
) -> BaselineBank:
    """Replacement constants for ``targets`` under the named baseline definition.

    ``cohort_mean`` uses the whole cohort. Capping the number of batches would
    make the denominator depend on an undeclared subsample of the very cohort it
    is compared against, so no cap is offered.
    """

    if kind not in BASELINE_KINDS:
        raise ValueError(f"unknown ablation baseline {kind!r}; known {BASELINE_KINDS}")
    unique = tuple(dict.fromkeys(targets))
    if not unique:
        raise ValueError("an ablation baseline needs at least one target")
    for target in unique:
        submodule_for(arm, target)
    if kind == "zero":
        return BaselineBank(
            kind=kind,
            vectors={
                target: torch.zeros(arm.d_model, dtype=torch.float32, device=arm.device)
                for target in unique
            },
            provenance={
                "definition": BASELINE_DEFINITIONS["zero"],
                "estimated_on_cohort": False,
                "cohort_digest": cohort_digest,
            },
        )
    if not batches:
        raise ValueError("cohort-mean baseline needs at least one scored batch")
    with _MeanAccumulator(arm, unique) as accumulator:
        for batch in batches:
            accumulator.keep = batch.attention_mask.bool()
            arm.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
            accumulator.tokens += int(accumulator.keep.sum())
            accumulator.keep = None
    if accumulator.tokens < 1:
        raise RuntimeError(f"{arm.name}: cohort-mean baseline saw no valid positions")
    unexpected = {
        target: count for target, count in accumulator.fired.items() if count != len(batches)
    }
    if unexpected:
        raise RuntimeError(f"{arm.name}: baseline hooks fired unexpectedly {unexpected}")
    vectors: dict[Target, torch.Tensor] = {}
    for target, total in accumulator.sums.items():
        mean = (total / accumulator.tokens).to(torch.float32)
        if not torch.isfinite(mean).all():
            raise FloatingPointError(
                f"{arm.name}: non-finite mean for {target.kind} layer {target.layer}"
            )
        vectors[target] = mean
    return BaselineBank(
        kind=kind,
        vectors=vectors,
        provenance={
            "definition": BASELINE_DEFINITIONS["cohort_mean"],
            "estimated_on_cohort": True,
            "cohort_digest": cohort_digest,
            "positions": "attention_mask_valid",
            "tokens": accumulator.tokens,
            "sequences": sum(len(batch.sequence_indices) for batch in batches),
            "accumulation_dtype": "float64",
        },
    )


class _PathwayAblation:
    """Replace every target's output with its baseline constant, for one scope."""

    def __init__(self, arm: Arm, targets: Sequence[Target], bank: BaselineBank) -> None:
        self.arm = arm
        self.targets = tuple(dict.fromkeys(targets))
        if not self.targets:
            raise ValueError("an ablation needs at least one target")
        bank.covers(self.targets)
        self.bank = bank
        self.fired: dict[Target, int] = {target: 0 for target in self.targets}
        self._handles: list[Any] = []

    def __enter__(self) -> _PathwayAblation:
        for target in self.targets:
            self._handles.append(
                submodule_for(self.arm, target).register_forward_hook(self._hook(target))
            )
        return self

    def __exit__(self, *_exception: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _hook(self, target: Target):
        value = self.bank.vector(target)

        def hook(_module: torch.nn.Module, _inputs: Any, output: Any) -> Any:
            tensor = _primary_tensor(
                output, f"{self.arm.name} {target.kind} layer {target.layer}", self.arm.d_model
            )
            self.fired[target] += 1
            # Materialised rather than left as a stride-zero broadcast view: a
            # downstream in-place write into a broadcast tensor would corrupt
            # every position at once and would not be visible in the metrics.
            replacement = (
                value.to(device=tensor.device, dtype=tensor.dtype).expand_as(tensor).contiguous()
            )
            if isinstance(output, tuple):
                return (replacement, *output[1:])
            return replacement

        return hook


# ----------------------------------------------------------------- measurement


@dataclass
class PathwayRun:
    """Per-sequence rows for every scope, scored against one clean reference."""

    rows_by_scope: dict[str, list[dict[str, float | int]]]
    targets_by_scope: dict[str, tuple[Target, ...]]
    scored_tokens: int
    scored_sequences: int
    target_token_counts: np.ndarray = field(repr=False)


@torch.inference_mode()
def measure_pathways(
    arm: Arm,
    batches: Sequence[ScoredBatch],
    scopes: Sequence[AblationScope],
    bank: BaselineBank,
) -> PathwayRun:
    """Score every scope against one clean forward pass per batch.

    The clean reference is computed once per batch and shared by all scopes so
    that ``ce_clean`` cannot drift between them; that invariant is checked
    rather than assumed, because a drifting clean reference would silently
    corrupt every delta.
    """

    if not batches:
        raise ValueError("measurement needs at least one scored batch")
    if not scopes:
        raise ValueError("measurement needs at least one ablation scope")
    names = [scope.name for scope in scopes]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate scope names in {names}")
    targets_by_scope = {scope.name: scope.resolve(arm.n_layer) for scope in scopes}
    for targets in targets_by_scope.values():
        bank.covers(targets)
    vocab = int(arm.model.config.vocab_size)
    counts = torch.zeros(vocab, dtype=torch.int64, device=arm.device)
    rows_by_scope: dict[str, list[dict[str, float | int]]] = {name: [] for name in names}
    scored_tokens = 0
    scored_sequences = 0

    for batch in batches:
        clean_logits = arm.model(
            input_ids=batch.input_ids, attention_mask=batch.attention_mask
        ).logits
        if not torch.isfinite(clean_logits).all():
            raise FloatingPointError(f"{arm.name}: non-finite clean logits")
        targets_in_batch = batch.input_ids[:, 1:][batch.target_mask]
        if int(targets_in_batch.max()) >= vocab:
            raise ValueError(f"{arm.name}: target token id outside the declared vocabulary")
        counts += torch.bincount(targets_in_batch, minlength=vocab)
        scored_tokens += int(targets_in_batch.numel())
        scored_sequences += len(batch.sequence_indices)
        reference: list[float] | None = None
        for scope in scopes:
            with _PathwayAblation(arm, targets_by_scope[scope.name], bank) as ablation:
                variant_logits = arm.model(
                    input_ids=batch.input_ids, attention_mask=batch.attention_mask
                ).logits
            unexpected = {
                target: count for target, count in ablation.fired.items() if count != 1
            }
            if unexpected:
                raise RuntimeError(
                    f"{arm.name}: scope {scope.name!r} hooks fired {unexpected} times"
                )
            if not torch.isfinite(variant_logits).all():
                raise FloatingPointError(
                    f"{arm.name}: non-finite logits under scope {scope.name!r}"
                )
            rows = per_sequence_scores(
                clean_logits, variant_logits, batch.input_ids, batch.target_mask
            )
            del variant_logits
            clean_sums = [float(row["clean_nll_sum"]) for row in rows]
            if reference is None:
                reference = clean_sums
            elif clean_sums != reference:
                raise RuntimeError(
                    f"{arm.name}: clean reference changed between scopes at scope {scope.name!r}"
                )
            for index, row in zip(batch.sequence_indices, rows):
                rows_by_scope[scope.name].append({"sequence_index": index, **row})
        del clean_logits

    if scored_tokens < 1:
        raise RuntimeError(f"{arm.name}: no scored next-token targets")
    return PathwayRun(
        rows_by_scope=rows_by_scope,
        targets_by_scope=targets_by_scope,
        scored_tokens=scored_tokens,
        scored_sequences=scored_sequences,
        target_token_counts=counts.cpu().numpy(),
    )


def scored_target_entropy_nats(counts: np.ndarray) -> float:
    """Plug-in entropy of the empirical next-token marginal over scored targets.

    This is an in-sample estimate on exactly the positions the measurement
    scores. For large vocabularies it underestimates the true marginal entropy,
    which shrinks the context-information denominator and therefore inflates
    ``share_of_context_information``. It is retained as a diagnostic and as an
    explicit opt-in, never as the default; ``disjoint_unigram_cross_entropy_nats``
    is the estimator a headline number should use.
    """

    array = np.asarray(counts, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or np.any(array < 0):
        raise ValueError("token counts must be a non-negative vector over the vocabulary")
    total = array.sum()
    if total <= 0:
        raise ValueError("token counts are empty")
    probabilities = array[array > 0] / total
    return float(-(probabilities * np.log(probabilities)).sum())


def cohort_target_token_counts(arm: Arm, cohort: Cohort, *, max_len: int) -> np.ndarray:
    """Next-token-target counts for a cohort, without running the model.

    A unigram model needs only token counts, so the held-out corpus that
    supplies the context-free baseline costs one tokenisation pass and no
    forward pass. The counted multiset matches ``prepare_batches``: every token
    of a sequence except its first, truncated at ``max_len``.
    """

    if max_len < 2:
        raise ValueError("max_len must admit at least one next-token target")
    vocab = int(arm.model.config.vocab_size)
    counts = np.zeros(vocab, dtype=np.int64)
    for text in cohort.input_strings(arm):
        ids = arm.tokenizer(text, return_tensors=None)["input_ids"][:max_len]
        if len(ids) < 2:
            continue
        targets = np.asarray(ids[1:], dtype=np.int64)
        if targets.min() < 0 or targets.max() >= vocab:
            raise ValueError(f"{arm.name}: token id outside the declared vocabulary")
        counts += np.bincount(targets, minlength=vocab)
    if counts.sum() < 1:
        raise RuntimeError(f"{arm.name}: cohort {cohort.name!r} yields no next-token targets")
    return counts


def disjoint_unigram_cross_entropy_nats(
    reference_counts: np.ndarray,
    target_counts: np.ndarray,
    *,
    smoothing: float = LAPLACE_SMOOTHING,
) -> float:
    """Cross-entropy of a held-out unigram model on the scored targets.

    This is the context-free baseline the model has to beat, estimated the same
    way the model itself is scored: a predictor fitted on data it will not be
    evaluated on, then evaluated on the scored targets. Because the reference
    corpus is disjoint, the estimate carries none of the downward bias the
    in-cohort plug-in has, and the additive smoothing biases it upwards, so any
    share computed against it is conservative rather than inflated.
    """

    reference = np.asarray(reference_counts, dtype=np.float64)
    targets = np.asarray(target_counts, dtype=np.float64)
    if reference.ndim != 1 or reference.shape != targets.shape or reference.size < 2:
        raise ValueError("reference and target counts must be vectors over one vocabulary")
    if np.any(reference < 0) or np.any(targets < 0):
        raise ValueError("token counts must be non-negative")
    if not smoothing > 0:
        raise ValueError("additive smoothing must be positive")
    reference_total = reference.sum()
    target_total = targets.sum()
    if reference_total < 1 or target_total < 1:
        raise ValueError("reference and target count vectors must both be non-empty")
    probabilities = (reference + smoothing) / (reference_total + smoothing * reference.size)
    return float(-(targets * np.log(probabilities)).sum() / target_total)


def held_out_cohort(candidate: Cohort, scored: Cohort) -> tuple[Cohort, dict[str, int]]:
    """``candidate`` with every record whose content also occurs in ``scored`` removed.

    Swiss-Prot and the EC-labelled corpus both carry the same sequence under
    several accessions, so taking a later block of records in file order does
    not by itself produce a held-out corpus. Fitting the context-free baseline
    on content it will then be evaluated on is precisely the leak this estimator
    exists to avoid, so the duplicates are removed by content and the number
    removed is returned for the record rather than absorbed silently.
    """

    if candidate.kind != scored.kind:
        raise ValueError("a held-out corpus must have the same kind as the cohort it serves")
    excluded = set(scored.records)
    keep = [index for index, record in enumerate(candidate.records) if record not in excluded]
    if not keep:
        raise ValueError(
            f"reference cohort {candidate.name!r} is entirely contained in {scored.name!r}"
        )
    metadata: dict[str, Any] = {}
    labels = candidate.metadata.get("ec_labels")
    if labels is not None:
        if len(labels) != len(candidate.records):
            raise ValueError(f"cohort {candidate.name!r}: EC labels do not align with records")
        metadata["ec_labels"] = [labels[index] for index in keep]
    cohort = Cohort(
        name=candidate.name,
        kind=candidate.kind,
        records=[candidate.records[index] for index in keep],
        min_symbols=candidate.min_symbols,
        max_symbols=candidate.max_symbols,
        metadata=metadata,
    )
    return cohort, {
        "requested_sequences": len(candidate.records),
        "retained_sequences": len(keep),
        "dropped_sequences_shared_with_cohort": len(candidate.records) - len(keep),
    }


def assert_disjoint(scored: Cohort, reference: Cohort) -> None:
    """Refuse a reference corpus that overlaps the cohort it will normalise."""

    overlap = set(scored.records) & set(reference.records)
    if overlap:
        raise ValueError(
            f"reference cohort {reference.name!r} shares {len(overlap)} sequences with "
            f"{scored.name!r}; a held-out baseline must be disjoint"
        )
    if scored.digest == reference.digest:
        raise ValueError(f"reference cohort {reference.name!r} is the scored cohort")


def unigram_baseline(
    arm: Arm,
    *,
    estimator: str,
    target_counts: np.ndarray,
    reference_counts: np.ndarray | None = None,
    reference: Mapping[str, Any] | None = None,
    override_nats: float | None = None,
    smoothing: float = LAPLACE_SMOOTHING,
) -> dict[str, Any]:
    """The context-free baseline for one arm, with the estimator declared.

    There is no fallback path. Asking for the held-out estimator without a
    held-out corpus is a configuration error and raises, because a silent
    downgrade to the plug-in would move the headline share by tens of percent on
    the large-vocabulary arms without changing anything a reader can see.
    """

    if estimator not in UNIGRAM_ESTIMATORS:
        raise ValueError(f"unknown unigram estimator {estimator!r}; known {UNIGRAM_ESTIMATORS}")
    plug_in = scored_target_entropy_nats(target_counts)
    record: dict[str, Any] = {
        "estimator": estimator,
        "cohort_plug_in_entropy_nats": plug_in,
        "smoothing": None,
        "reference": None,
    }
    if override_nats is not None:
        if not math.isfinite(override_nats) or override_nats <= 0:
            raise ValueError(f"{arm.name}: supplied unigram entropy must be finite and positive")
        return {**record, "nats": float(override_nats), "source": "external_override"}
    if estimator == "plugin":
        return {**record, "nats": plug_in, "source": "cohort_scored_target_plug_in"}
    if reference_counts is None or reference is None:
        raise RuntimeError(
            f"{arm.name}: the disjoint unigram estimator needs a held-out reference corpus; "
            "supply one or opt in to --unigram-estimator plugin explicitly"
        )
    nats = disjoint_unigram_cross_entropy_nats(
        reference_counts, target_counts, smoothing=smoothing
    )
    return {
        **record,
        "nats": nats,
        "source": "disjoint_reference_cross_entropy",
        "smoothing": float(smoothing),
        "reference": {
            **dict(reference),
            "tokens": int(np.asarray(reference_counts).sum()),
            "distinct_tokens": int((np.asarray(reference_counts) > 0).sum()),
        },
    }


def pathway_metrics(
    rows: Sequence[Mapping[str, float | int]],
    *,
    unigram_entropy_nats: float,
    minimum_ce_delta_nats: float = P0_2B_MINIMUM_CE_DELTA_NATS,
    minimum_kl_nats: float = P0_2B_MINIMUM_KL_NATS,
) -> dict[str, Any]:
    """Footprint of one scope in nats/token, normalised and guard-checked.

    ``unigram_entropy_nats`` is the context-free cross-entropy of the cohort and
    is supplied by the caller rather than recomputed here, so that one cohort
    budget can normalise every scope and every arm identically.

    An arm whose clean cross-entropy is not below its own context-free baseline
    is off-distribution on this cohort: it extracts no context information, so
    there is no denominator to normalise against and its guard verdict carries
    no meaning. That case is reported as ``context_information_valid`` false
    with a null share and ``measurable`` false, following the same convention as
    ``pathway_metrics``' own recovery reporting. It is not an error --
    a matched panel will contain arms that are off-distribution on some cohort,
    and that fact is part of the result.
    """

    if not math.isfinite(unigram_entropy_nats):
        raise ValueError("unigram entropy must be finite")
    if minimum_ce_delta_nats < 0 or minimum_kl_nats < 0:
        raise ValueError("denominator guards must be non-negative")
    aggregate = aggregate_variant(rows)
    ce_clean = aggregate["clean_ce_nats"]
    ce_ablated = aggregate["variant_ce_nats"]
    ce_delta = ce_ablated - ce_clean
    kl = aggregate["clean_to_variant_kl_nats"]
    context_information = unigram_entropy_nats - ce_clean
    context_valid = context_information > 0
    passes_ce = ce_delta >= minimum_ce_delta_nats
    passes_kl = kl >= minimum_kl_nats
    return {
        "schema_version": SCHEMA_VERSION_MEASUREMENT,
        "ce_clean_nats": ce_clean,
        "ce_ablated_nats": ce_ablated,
        "ce_delta_nats": ce_delta,
        "kl_clean_to_ablated_nats": kl,
        "argmax_agreement": aggregate["argmax_agreement"],
        "unigram_entropy_nats": float(unigram_entropy_nats),
        "context_information_nats": context_information,
        "context_information_valid": bool(context_valid),
        "share_of_context_information": (
            ce_delta / context_information if context_valid else None
        ),
        "minimum_ce_delta_nats": float(minimum_ce_delta_nats),
        "minimum_kl_nats": float(minimum_kl_nats),
        "passes_ce_guard": bool(passes_ce),
        "passes_kl_guard": bool(passes_kl),
        "measurable": bool(passes_ce and passes_kl and context_valid),
        "scored_tokens": aggregate["scored_tokens"],
        "sequences": aggregate["sequences"],
    }


def _interval(values: Sequence[float]) -> dict[str, float] | None:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 1:
        return None
    return {
        "q025": float(np.quantile(array, 0.025)),
        "median": float(np.quantile(array, 0.5)),
        "q975": float(np.quantile(array, 0.975)),
    }


def pathway_cluster_bootstrap(
    rows: Sequence[Mapping[str, float | int]],
    *,
    samples: int,
    seed: int,
    unigram_entropy_nats: float,
    minimum_ce_delta_nats: float = P0_2B_MINIMUM_CE_DELTA_NATS,
    minimum_kl_nats: float = P0_2B_MINIMUM_KL_NATS,
) -> dict[str, Any]:
    """Sequence-cluster bootstrap of one scope's footprint.

    A dictionary-fidelity bootstrap resamples a paired dictionary/mean-ablation
    design and returns recovery fractions; there is no dictionary here, so the
    resampled quantities differ. The cohort's context-free baseline is held at its full-cohort
    value across draws, so the interval reflects uncertainty in the ablation
    effect only.
    """

    if not rows:
        raise ValueError("bootstrap needs at least one sequence row")
    if samples < 1:
        raise ValueError("bootstrap sample count must be positive")
    generator = np.random.default_rng(seed)
    n = len(rows)
    ce_delta: list[float] = []
    kl: list[float] = []
    share: list[float] = []
    ce_guard_passes = 0
    kl_guard_passes = 0
    context_invalid = 0
    for _ in range(samples):
        indices = generator.integers(0, n, size=n)
        metrics = pathway_metrics(
            [rows[int(index)] for index in indices],
            unigram_entropy_nats=unigram_entropy_nats,
            minimum_ce_delta_nats=minimum_ce_delta_nats,
            minimum_kl_nats=minimum_kl_nats,
        )
        ce_delta.append(metrics["ce_delta_nats"])
        kl.append(metrics["kl_clean_to_ablated_nats"])
        if metrics["context_information_valid"]:
            share.append(metrics["share_of_context_information"])
        else:
            context_invalid += 1
        ce_guard_passes += int(metrics["passes_ce_guard"])
        kl_guard_passes += int(metrics["passes_kl_guard"])
    return {
        "schema_version": SCHEMA_VERSION_BOOTSTRAP,
        "cluster_unit": "sequence",
        "samples": int(samples),
        "seed": int(seed),
        "clusters": n,
        "ce_delta_nats": _interval(ce_delta),
        "kl_clean_to_ablated_nats": _interval(kl),
        "share_of_context_information": _interval(share),
        "context_invalid_samples": context_invalid,
        "ce_guard_pass_fraction": ce_guard_passes / samples,
        "kl_guard_pass_fraction": kl_guard_passes / samples,
    }


def scope_record(scope: AblationScope, targets: Sequence[Target]) -> dict[str, Any]:
    """JSON-serialisable identity of a scope as it was resolved for one arm."""

    return {
        "name": scope.name,
        "family": scope.family,
        "submodules": list(scope.submodules),
        "anchor_layer": scope.anchor_layer,
        "width": scope.width,
        "targets": [[target.kind, target.layer] for target in targets],
        "n_targets": len(targets),
        "layers": sorted({target.layer for target in targets}),
    }
