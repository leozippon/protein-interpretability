#!/usr/bin/env python3
"""Is the representational difference between two checkpoints of one lineage already explained by a simple map?

**The question this stage exists to answer, and what it gates.** R2.4 asks whether
a Crosscoder trained on ``Llama-2-7b-hf`` -> ``ProLLaMA_Stage_1`` would buy
anything. A Crosscoder is the most expensive object this programme could train and
the one whose result is hardest to falsify afterwards, so ``docs/RESEARCH_PLAN.md``
§4c fixes the readings before the evidence exists. One of them is decided here:
*linear or orthogonal alignment already accounts for the discrepancy on held-out
positions, against its own shuffled-pairing null*. If a 4096-vector or an
orthogonal matrix already carries the target's representation from the
reference's, there is no difference for a dictionary basis to represent and the
compute goes elsewhere.

**Why this lineage and only this lineage.** ``Llama-2-7b-hf`` and
``ProLLaMA_Stage_1`` are both ``LlamaForCausalLM`` 32 x 4096 over one unmodified
32000-piece SentencePiece vocabulary with no added tokens (EXP-R2-152,
EXP-R2-163). That is what makes a position-by-position comparison *defined*: the
same rendered record tokenises to the same ids in both, so activation *i* of A and
activation *i* of B are two models' readings of one token in one context. Nothing
here establishes that premise -- it is checked, by digest over the full
id-to-token map, and refused when it does not hold. Two checkpoints with different
tokenizers never saw the same input, and every number below would compare
unrelated positions while still coming out finite and plausible.

**What is measured.** Both checkpoints are fed the same records in the same order,
and at a declared per-layer tensor, over content-masked scored positions only,
four maps are fitted on a training split and reported on a disjoint held-out
split:

``identity``
    no map at all -- the raw activation difference.
``mean_shift``
    the target predicted by the reference plus a constant vector, i.e. both sides
    centred on their training means.
``procrustes``
    centred, one orthogonal matrix and one global scale: the rigid alignment.
``ridge``
    centred, a full ``d x d`` linear map at a declared ridge coefficient: the most
    any linear method can do.

The headline per layer and per method is the **normalised residual**
``||B - f(A)||^2 / ||B - mean_train(B)||^2`` on held-out positions. It is
scale-free (Appendix B rule 21), which matters because a LLaMA's residual scale
grows by orders of magnitude with depth, and its denominator travels with it in
the artefact (rule 27) so a reader can convert back. The mean cosine between B and
``f(A)`` and each side's mean norm are reported beside it: a rotation, a rescaling
and new content are three different findings and the residual alone does not
separate them.

**Two controls, neither optional.**

*Shuffled pairing.* Every fit and every evaluation is repeated with the target's
positions permuted relative to the reference's, from a declared seed. This is the
null. ``ridge`` has ``d^2 = 16.8M`` free parameters and is fitted on ``n`` tokens;
without the null there is no way to tell a real correspondence from that capacity,
and a good ``ridge`` number would be unreadable. It is reported *beside* every
true-pairing number, never instead of it.

*Adjacent layer, within the reference.* The same four methods applied to layer
``l`` -> layer ``l+1`` of the reference alone. This is the unit: it says how large
the cross-checkpoint shift is **in units of one layer of ordinary computation**.
Without it "0.31 normalised residual" has no scale.

**What this stage does NOT do.** It does not invent a threshold for "a Crosscoder
is warranted". It reports the three quantities that decide it, states in words
what each of the three possible readings would be, and leaves the decision to a
person reading the artefact. And it makes no behavioural claim at all: a residual
between two activation tensors is a representational statement, and the two are
not interchangeable (Evidence Discipline, §8 of the audit).

**The one limitation that travels with the protein mode.** The base checkpoint's
protein mode is unmeasurable -- context information +0.084 nats per token,
reversal cost -0.001 nats per residue (EXP-R2-152). This stage may still be
pointed at it, because a representational comparison does not require a measurable
behavioural estimand: the activations exist and are comparable position by
position whether or not the model does anything useful with them. But the artefact
carries that limitation in its own field on every protein-mode run, and the
verdict says in its own words that it is not a behavioural claim.

**Memory, stated because it decides the defaults.** The activations of one split
cannot be held: 32 layers x 4096 wide x ~660k scored positions is ~170 GB per
model in float32, and there are two. Sufficient statistics are accumulated instead
-- per layer, counts, sums and cross-moment matrices, in float64 on the compute
device -- and the evaluation is a second streaming pass. At 32 x 4096 one float64
``d x d`` matrix is ``4096^2 * 8 B = 134.2 MB``, and the arithmetic is:

* training statistics -- ``A'A`` (32), ``A'B`` (32), ``A'PB`` (32), ``A_l'A_l+1``
  (31) and ``A_l'PA_l+1`` (31), so ``5L - 2 = 158`` matrices = **21.2 GB**;
* fitted maps -- 2 comparisons x 2 pairings x 2 matrix-valued methods x 63
  layer-pairs = 252 matrices = **33.8 GB**, built layer by layer while that
  layer's statistics are released, so the solve ends at 33.8 GB and never holds
  both in full;
* two 6.74B checkpoints at bfloat16 = **27.0 GB**;
* per batch, ``17_train_transcoder.py``'s ``capture`` stacks block input and block
  output for both models: ``4 * 32 * 8 * 512 * 4096 * 2 B`` = **4.3 GB**, plus the
  flattened copies and at most three per-layer float64 working copies -- ~7 GB in
  total, transient.

Peak is therefore about **68 GB**, inside one H200's 143771 MiB, and it is one
forward pass per model per split.

An external baseline, not a registered panel stage: a checkpoint is reached by
path, so it cannot be scheduled through ``panel_contract.STAGE_CONTRACTS``.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# The stage directory itself, so the stages imported below resolve their own
# `panel_contract` import under every invocation rather than only when the caller
# happens to run from scripts/transfer.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.transfer import joint_modes  # noqa: E402
from src.transfer.arms import (  # noqa: E402
    DEFAULT_CORPUS_DRAW_SEED,
    REPO,
    corpus_location,
    iter_corpus_records,
)
from src.transfer.io import sha256_file, write_json  # noqa: E402
from src.transfer.replaceable import (  # noqa: E402
    JOINT_MODES,
    JointReplaceable,
    joint_mode_corpus,
    joint_tokenisation,
)


def _load_stage(filename: str) -> Any:
    """Import a stage whose module name starts with a digit.

    Three of them, and each because this stage's numbers are only readable if they
    are the *same* computation that stage performs: stage 21 owns the checkpoint
    loader -- the tokenizer read before the weights, the shape read back off the
    built model rather than echoed from the request -- stage 17 owns the captured
    tensors, the seeded shuffled-reservoir draw and the residue band, and stage 24
    owns the digest that decides whether two checkpoints' positions are comparable
    at all. Appendix B rule 12, a single declaration imported and never
    reimplemented, does not stop applying because the declaration lives in a file
    whose name starts with a digit.
    """

    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(f"_transfer_stage_{filename[:2]}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE17 = _load_stage("17_train_transcoder.py")
STAGE21 = _load_stage("21_joint_mode_qualification.py")
STAGE24 = _load_stage("24_component_swap.py")

SCHEMA_VERSION = "r2_transfer_model_diffing_baselines_v1"
DEFAULT_OUT = REPO / "results/transfer/model_diffing_baselines"

#: Both checkpoints are loaded at one precision, hard-coded rather than exposed
#: for the reason ``17_train_transcoder.py`` hard-codes it on the joint path: a
#: residual between two checkpoints held at two precisions is partly a residual
#: between two quantisations. Every statistic, fit and evaluation below is
#: float64 regardless of this.
INFERENCE_DTYPE = "bfloat16"

#: Modules and stages whose content decides these numbers, hashed into the
#: artefact. The splice module is first because it decides what the declared
#: tensor IS on this block layout, and the rendering module second because it has
#: been worth 2.9 nats/token when wrong.
PROVENANCE_MODULES = (
    "src/transfer/replaceable.py",
    "src/transfer/joint_modes.py",
    "src/transfer/arms.py",
    "src/transfer/io.py",
    "scripts/transfer/17_train_transcoder.py",
    "scripts/transfer/21_joint_mode_qualification.py",
    "scripts/transfer/24_component_swap.py",
)

#: The four maps, in increasing order of what each is allowed to do. The order is
#: load-bearing: the reading of the artefact is *which* of them first drives the
#: residual down, and a method that was not a superset of the one before it would
#: make that sequence meaningless.
METHODS: tuple[str, ...] = ("identity", "mean_shift", "procrustes", "ridge")

METHOD_DEFINITIONS: dict[str, str] = {
    "identity": (
        "f(a) = a. No map and no fit: the raw activation difference between the "
        "two checkpoints at the same position. Every other method's residual is "
        "also reported as the fraction of this one it removes"
    ),
    "mean_shift": (
        "f(a) = a - mean_train(A) + mean_train(B). One constant vector, fitted on "
        "the training split; equivalently, centre both sides on their own training "
        "means and compare"
    ),
    "procrustes": (
        "f(a) = s * (a - mean_train(A)) @ Q + mean_train(B), with Q orthogonal and "
        "s one global scale, both fitted on the training split. The rigid "
        "alignment: it may rotate and rescale the representation but not reshape "
        "it, so a low residual here means the two checkpoints hold the same "
        "geometry in a different frame"
    ),
    "ridge": (
        "f(a) = (a - mean_train(A)) @ W + mean_train(B), with W the ridge solution "
        "of the centred least-squares problem at the declared coefficient. The most "
        "any linear method can do, and therefore the one whose success says least "
        "without the shuffled null beside it: W carries d^2 free parameters"
    ),
}

#: The two pairings every fit and every evaluation is run under.
PAIRINGS: tuple[str, ...] = ("true", "shuffled")

#: The two comparisons. ``cross`` is the question; ``adjacent`` is its unit.
COMPARISONS: tuple[str, ...] = ("cross", "adjacent")

COMPARISON_DEFINITIONS: dict[str, str] = {
    "cross": (
        "reference layer l predicts target layer l, at the same position of the "
        "same record. This is the cross-checkpoint difference the stage exists to "
        "measure"
    ),
    "adjacent": (
        "reference layer l predicts reference layer l+1, at the same position of "
        "the same record, inside the reference checkpoint alone. This is the UNIT: "
        "it says how large the cross-checkpoint shift is in units of one layer of "
        "ordinary computation. Defined for l = 0 .. n_layers - 2"
    ),
}

TENSORS: tuple[str, ...] = ("block_input", "block_output")

#: What a permutation within one batch is and is not, recorded rather than left to
#: a reader who sees the word "shuffled".
SHUFFLE_NOTE = (
    "the null pairs reference position i with target position pi(i) for a "
    "permutation drawn per batch from the declared seed, so token-level "
    "correspondence is destroyed while both marginal distributions are exactly "
    "preserved -- a permutation does not change the multiset of target rows, so "
    "the denominator ||B - mean_train(B)||^2 and the target's mean norm are "
    "identical under both pairings and only the pairing moves. The permutation is "
    "WITHIN a batch and not across the whole split, because a global permutation "
    "would need every position of the split resident at once, which is the memory "
    "this stage exists to avoid. A batch holds --batch-size unrelated records "
    "drawn from a seeded shuffled stream, so what the null retains is "
    "between-batch covariance of the two models' means and nothing finer. That "
    "makes it CONSERVATIVE: it can only make the null look better than a global "
    "permutation would, and therefore only understate the gap between true and "
    "shuffled pairing"
)


# ------------------------------------------------------------------- refusals


def assert_identical_tokenizers(reference: Any, target: Any) -> dict[str, Any]:
    """Refuse two checkpoints whose tokenizers are not the same vocabulary.

    The digest is ``24_component_swap.py``'s, imported rather than restated,
    because it is the thing that decides the answer and one definition of "the
    same vocabulary" is what stops a swap and a diff meaning different things by
    it. The comparison and its message are stated here because that stage's roles
    are *host* and *donor*, which name a swap; these two are a reference and a
    target and nothing moves between them.

    This is the premise the whole stage rests on. Two tokenizers that differ mean
    the two models never saw the same input, so "activation i of A" and
    "activation i of B" are readings of different tokens in different contexts and
    every residual below compares unrelated things -- while every number still
    comes out finite and plausible.
    """

    sizes = (int(len(reference)), int(len(target)))
    digests = (STAGE24.vocabulary_digest(reference), STAGE24.vocabulary_digest(target))
    if sizes[0] != sizes[1] or digests[0] != digests[1]:
        raise ValueError(
            "the reference and target tokenizers are not the same vocabulary "
            f"(sizes {sizes}, digests {digests[0][:12]}.. / {digests[1][:12]}..). "
            "A position-by-position comparison is defined only when one record "
            "tokenises to the same ids in both checkpoints; otherwise the two "
            "models never saw the same input and every residual this stage reports "
            "compares unrelated positions while still coming out finite. The "
            "ProLLaMA lineage carries one unmodified vocabulary at all three points "
            "with no added tokens, which is the premise this stage rests on -- it "
            "is checked, not assumed"
        )
    return {
        "verdict": "IDENTICAL",
        "vocabulary_size": sizes[0],
        "vocabulary_sha256": digests[0],
        "reference_tokenizer_class": type(reference).__name__,
        "target_tokenizer_class": type(target).__name__,
        "digest_source": (
            "scripts/transfer/24_component_swap.py::vocabulary_digest -- SHA-256 "
            "over the full id-to-token map, one declaration shared with the "
            "component swap (Appendix B rule 12)"
        ),
    }


#: Config facts that must agree before a single position of one is compared with a
#: position of the other. Read back from each loaded model by
#: ``21_joint_mode_qualification.load_model``, never echoed from a request.
COMPARABLE_FACTS = ("model_type", "architectures", "n_layers", "d_model", "vocab_size")


def assert_comparable_shape(
    reference: JointReplaceable,
    target: JointReplaceable,
    *,
    reference_facts: dict[str, Any],
    target_facts: dict[str, Any],
) -> dict[str, Any]:
    """Refuse two checkpoints whose per-layer activations are not the same object.

    Depth and width are checked twice: on the two configs, and on the two loaded
    handles, which are what the capture actually indexes. The declared tensor is
    compared too -- ``perturbation_target`` is what says which tensor "the block
    output" names on a given block layout, and two layouts that named different
    tensors by it would put a residual between two different quantities.
    """

    differing = {
        name: [reference_facts.get(name), target_facts.get(name)]
        for name in COMPARABLE_FACTS
        if reference_facts.get(name) != target_facts.get(name)
    }
    if differing:
        raise ValueError(
            f"the reference and target disagree on {sorted(differing)}: {differing}. "
            "A per-layer, per-position representational comparison is defined only "
            "between checkpoints of one architecture and one shape"
        )
    if (reference.n_layers, reference.width) != (target.n_layers, target.width):
        raise ValueError(
            "the loaded handles disagree on shape: reference "
            f"{reference.n_layers}L x {reference.width}d against target "
            f"{target.n_layers}L x {target.width}d. The two configs agreed, so one "
            "of them did not build what it declared"
        )
    if reference.perturbation_target != target.perturbation_target:
        raise ValueError(
            "the two checkpoints declare different per-layer tensors "
            f"({reference.perturbation_target['tensor']!r} against "
            f"{target.perturbation_target['tensor']!r}), so one --tensor name would "
            "select a different object in each and the residual would be between "
            "two different quantities"
        )
    return {
        "verdict": "COMPARABLE",
        "n_layers": int(reference.n_layers),
        "d_model": int(reference.width),
        "shared_facts": {name: reference_facts.get(name) for name in COMPARABLE_FACTS},
        "note": (
            "depth, width and the declared per-layer tensor agree, read back off "
            "the two loaded handles rather than echoed from the two configs"
        ),
    }


def tensor_declaration(model: JointReplaceable, tensor: str) -> dict[str, Any]:
    """Which tensor ``--tensor`` selected, by name, on this block layout.

    "The block output" names a different tensor on different block layouts -- on a
    parallel-residual block it is not the block's output at all -- so the name has
    to reach the artefact together with the layout's own declaration of what it
    means, rather than being left for a reader to infer from the stage name.
    """

    if tensor not in TENSORS:
        raise ValueError(f"unknown tensor {tensor!r}; declared: {list(TENSORS)}")
    feed_forward = model.layout.feed_forward
    return {
        "selected": tensor,
        "is": (
            f"the input to each layer's {feed_forward!r}, which on this layout is "
            f"the output of {model.layout.pre_feed_forward_norm!r}"
            if tensor == "block_input"
            else f"the output of each layer's {feed_forward!r}, before the residual add"
        ),
        "block_layout": model.perturbation_target,
        "captured_by": (
            "scripts/transfer/17_train_transcoder.py::capture, the same pair a "
            "transcoder is fitted to, over the same content mask"
        ),
    }


def draw_splits(
    records: Callable[[], Iterator[tuple[str, str | None]]],
    *,
    n_train: int,
    n_eval: int,
    seed: int,
    skip: int,
) -> tuple[list[tuple[str, str | None]], list[tuple[str, str | None]], dict[str, Any]]:
    """One seeded pool, split into a training and a held-out half.

    **One pool and not two draws**, which is the point. A biological corpus is
    ordered by cluster and a web corpus by shard, so two draws taken at different
    offsets are two *regions* rather than two samples: on this very stream,
    ``17_train_transcoder.py`` records the first shuffle block at a mean 394
    residues against 878 for blocks 2-7. Fitting on one region and evaluating on
    another would put a population gap between the map and the number that judges
    it, and the gap would read as a failure of the map. Drawing one pool and
    splitting it under a seeded permutation makes the two halves samples of one
    population by construction, and disjoint by construction.

    ``skip`` moves the pool through the corpus in file order. A second run at a
    different value IS the skip-offset sensitivity Appendix B rule 1 requires; the
    seed cannot supply it, because the shuffled reservoir permutes *within* blocks
    that are read in file order and so leaves the region unchanged.
    """

    total = int(n_train) + int(n_eval)
    if n_train < 1 or n_eval < 1:
        raise ValueError("both splits need at least one record")
    pool = list(STAGE17.stream_records(records, seed=seed, skip=int(skip), limit=total))
    if len(pool) < total:
        raise RuntimeError(
            f"the corpus ran out: {len(pool)} of {total} eligible records past a "
            f"skip of {skip}. Lower --train-records/--eval-records or --skip rather "
            "than fitting and evaluating on fewer positions than declared"
        )
    order = np.random.default_rng(seed + 1).permutation(len(pool))
    train = [pool[int(index)] for index in order[:n_train]]
    evaluation = [pool[int(index)] for index in order[n_train:]]

    shared = {record for record, _ in train} & {record for record, _ in evaluation}
    if shared:
        raise RuntimeError(
            f"{len(shared)} record(s) appear in both splits, so the held-out split "
            "is not held out and every map fitted on the training split would be "
            "reported partly on itself. The two halves are index-disjoint by "
            "construction, so this means the corpus carries duplicate records in "
            "this band; raise --skip or narrow the band"
        )
    return (
        train,
        evaluation,
        {
            "verdict": "DISJOINT",
            "pool_records": len(pool),
            "train_records": len(train),
            "eval_records": len(evaluation),
            "skip_records": int(skip),
            "shuffle_block": int(STAGE17.SHUFFLE_BLOCK),
            "blocks_spanned": -(-total // int(STAGE17.SHUFFLE_BLOCK)),
            "draw": (
                "one seeded shuffled-reservoir draw of train+eval records past "
                "--skip, then a seeded permutation splits it. Both halves are "
                "therefore samples of ONE population and disjoint by index; content "
                "overlap is checked separately and refuses"
            ),
            "skip_offset_sensitivity": (
                "not measured by one run. Re-run at a different --skip: the seed "
                "permutes within blocks read in file order and cannot move the "
                "region (Appendix B rule 1)"
            ),
        },
    )


# ---------------------------------------------------------------- the capture


def assert_identical_batches(
    reference: JointReplaceable,
    target: JointReplaceable,
    records: Sequence[tuple[str, str | None]],
) -> None:
    """Both checkpoints receive the same rendered strings and the same ids.

    Checked on every batch rather than inferred from the tokenizer digest, because
    the digest covers the id-to-token map and a batch depends on more than that:
    the rendering, the token cap, the padding id and -- in text mode -- the
    tokenizer's declared special ids, which decide the content mask and are not
    part of the vocabulary digest. Each side renders and batches independently and
    the two results are compared, so this is agreement between two objects rather
    than one object read twice.
    """

    sequences = [record for record, _ in records]
    labels = [label for _, label in records]
    rendered = (
        reference.render(sequences, ec_labels=labels),
        target.render(sequences, ec_labels=labels),
    )
    if rendered[0] != rendered[1]:
        raise RuntimeError(
            "the two checkpoints rendered the same records into different strings, "
            "so they are not being fed the same input and no position of one "
            "corresponds to a position of the other"
        )
    batches = (reference.batch(rendered[0]), target.batch(rendered[1]))
    reference.forget_rendered()
    target.forget_rendered()
    if sorted(batches[0]) != sorted(batches[1]):
        raise RuntimeError(
            f"the two batches carry different fields ({sorted(batches[0])} against "
            f"{sorted(batches[1])}), so they cannot be compared position by position"
        )
    for field, left in batches[0].items():
        right = batches[1][field]
        if left.shape != right.shape or not bool(torch.equal(left, right)):
            raise RuntimeError(
                f"the two checkpoints' {field!r} differ on the same records, so "
                "position i of the reference and position i of the target are not "
                "the same token in the same context, and every residual below would "
                "compare unrelated positions"
            )


@torch.no_grad()
def _one_side(
    model: JointReplaceable, records: Sequence[tuple[str, str | None]], *, tensor: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """One model's declared tensor and content mask on one batch."""

    block_input, block_output, mask = STAGE17.capture(model, list(records))
    kept = STAGE17.flatten(block_input, block_output, mask)
    del block_input, block_output
    return kept[0 if tensor == "block_input" else 1], mask


@torch.no_grad()
def paired_capture(
    reference: JointReplaceable,
    target: JointReplaceable,
    records: Sequence[tuple[str, str | None]],
    *,
    tensor: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The declared tensor of both checkpoints on one batch, ``(layers, tokens, d)``.

    ``17_train_transcoder.py``'s ``capture`` and ``flatten`` do the work unchanged,
    so the tensors compared here are the tensors a transcoder would be fitted to
    and the kept positions are the mode's own content positions -- in protein mode
    the rendering's scored span, in text mode every non-padding, non-special
    position.
    """

    assert_identical_batches(reference, target, records)
    a, a_mask = _one_side(reference, records, tensor=tensor)
    b, b_mask = _one_side(target, records, tensor=tensor)
    if a_mask.shape != b_mask.shape or not bool(torch.equal(a_mask, b_mask)):
        raise RuntimeError(
            "the two checkpoints' content masks disagree on one batch, so they kept "
            "different positions and no position of one corresponds to a position "
            "of the other"
        )
    if a.shape != b.shape:
        raise RuntimeError(
            f"the two checkpoints produced different activation shapes on one batch "
            f"({tuple(a.shape)} against {tuple(b.shape)})"
        )
    return a, b


# ------------------------------------------------------ sufficient statistics


class PairedMoments:
    """Per-layer counts, sums and cross-moments for one split, in float64.

    The whole reason this class exists: the activations of one split do not fit. At
    32 x 4096 and ~660k scored positions one model's are ~170 GB in float32, and
    there are two. Everything the four fits need is a second moment, so the pass
    accumulates ``5L - 2`` matrices of ``d x d`` instead -- 21.2 GB at this shape --
    and the evaluation is a separate streaming pass.

    Float64 throughout, and not as a precaution. These are sums of ``n`` outer
    products of activations whose scale grows by orders of magnitude with depth; in
    float32 the accumulation error at ``n`` of order 1e6 is comparable to the
    residual the stage is trying to resolve.

    Per-layer *lists* rather than one stacked tensor, so that :meth:`release` can
    actually free a layer. A slice of a stacked allocation cannot be freed, and the
    solve would then hold the statistics and the maps at once -- 55 GB where either
    alone is at most 34.
    """

    def __init__(self, *, n_layers: int, d_model: int, device: torch.device) -> None:
        if n_layers < 2:
            raise ValueError(
                f"{n_layers} layer(s) leaves no adjacent pair, so the unit the "
                "cross-checkpoint residual has to be read in would not exist"
            )
        self.n_layers = int(n_layers)
        self.d_model = int(d_model)
        self.n = 0
        options: dict[str, Any] = {"dtype": torch.float64, "device": device}

        def square() -> torch.Tensor:
            return torch.zeros(d_model, d_model, **options)

        self.sum_a = [torch.zeros(d_model, **options) for _ in range(n_layers)]
        self.sum_b = [torch.zeros(d_model, **options) for _ in range(n_layers)]
        self.xx: list[torch.Tensor | None] = [square() for _ in range(n_layers)]
        self.xy: dict[str, list[torch.Tensor | None]] = {
            pairing: [square() for _ in range(n_layers)] for pairing in PAIRINGS
        }
        self.adjacent: dict[str, list[torch.Tensor | None]] = {
            pairing: [square() for _ in range(n_layers - 1)] for pairing in PAIRINGS
        }

    @torch.no_grad()
    def update(self, a: torch.Tensor, b: torch.Tensor, permutation: torch.Tensor) -> None:
        """One batch of paired positions, cast to float64 one layer at a time."""

        if a.shape != b.shape or a.shape[0] != self.n_layers or a.shape[2] != self.d_model:
            raise ValueError(
                f"expected paired ({self.n_layers}, tokens, {self.d_model}) tensors, "
                f"got {tuple(a.shape)} and {tuple(b.shape)}"
            )
        self.n += int(a.shape[1])
        previous: torch.Tensor | None = None
        for layer in range(self.n_layers):
            x = a[layer].to(torch.float64)
            y = b[layer].to(torch.float64)
            self.sum_a[layer] += x.sum(0)
            self.sum_b[layer] += y.sum(0)
            transposed = x.T
            self.xx[layer] += transposed @ x
            self.xy["true"][layer] += transposed @ y
            self.xy["shuffled"][layer] += transposed @ y[permutation]
            if previous is not None:
                self.adjacent["true"][layer - 1] += previous @ x
                self.adjacent["shuffled"][layer - 1] += previous @ x[permutation]
            previous = transposed

    def target_sum(self, comparison: str, layer: int) -> torch.Tensor:
        """The summed target of one comparison: the target model, or the next layer."""

        return self.sum_b[layer] if comparison == "cross" else self.sum_a[layer + 1]

    def cross_moment(self, comparison: str, pairing: str, layer: int) -> torch.Tensor:
        matrices = self.xy if comparison == "cross" else self.adjacent
        matrix = matrices[pairing][layer]
        if matrix is None:
            raise RuntimeError(f"layer {layer}'s {comparison} statistics were already released")
        return matrix

    def predictor_moment(self, layer: int) -> torch.Tensor:
        matrix = self.xx[layer]
        if matrix is None:
            raise RuntimeError(f"layer {layer}'s predictor statistics were already released")
        return matrix

    def release(self, layer: int) -> None:
        """Drop one layer's matrices once its maps are built."""

        self.xx[layer] = None
        for pairing in PAIRINGS:
            self.xy[pairing][layer] = None
            if layer < self.n_layers - 1:
                self.adjacent[pairing][layer] = None


# ------------------------------------------------------------------- the maps


@dataclass(frozen=True)
class LinearMap:
    """``f(a) = (a - centre) @ matrix + offset``, with ``None`` meaning "skip".

    One object for all four methods rather than four. ``identity`` carries no
    centre, no matrix and no offset; ``mean_shift`` carries the two means and no
    matrix; ``procrustes`` and ``ridge`` carry all three and differ only in what
    the matrix is. That keeps the evaluation loop free of a per-method branch, so
    the four numbers come out of one piece of arithmetic and cannot diverge in how
    they were computed.
    """

    centre: torch.Tensor | None
    matrix: torch.Tensor | None
    offset: torch.Tensor | None
    facts: dict[str, Any]

    def apply(self, a: torch.Tensor) -> torch.Tensor:
        value = a if self.centre is None else a - self.centre
        if self.matrix is not None:
            value = value @ self.matrix
        return value if self.offset is None else value + self.offset


def fit_maps(
    *,
    n: int,
    predictor_sum: torch.Tensor,
    target_sum: torch.Tensor,
    predictor_moment: torch.Tensor,
    cross_moment: torch.Tensor,
    ridge: float,
) -> dict[str, LinearMap]:
    """The four maps for one (predictor, target) pair, from training statistics alone.

    ``ridge`` is **relative**, not absolute: the coefficient applied is
    ``ridge * trace(centred predictor moment) / d``, a fraction of the mean
    eigenvalue. A LLaMA's activation scale spans orders of magnitude across depth,
    so one absolute coefficient would be a different regularisation at every layer
    and the per-layer curve would partly be a curve of how hard each layer was
    regularised (Appendix B rule 26's shape, applied to a hyper-parameter). The
    realised absolute value is recorded per layer beside it.
    """

    centre = predictor_sum / n
    offset = target_sum / n
    centred_predictor = predictor_moment - n * torch.outer(centre, centre)
    centred_cross = cross_moment - n * torch.outer(centre, offset)
    width = centred_predictor.shape[0]
    total = float(torch.trace(centred_predictor))
    if not total > 0.0:
        raise RuntimeError(
            f"the predictor's centred second moment has trace {total}, so the "
            "predictor is constant over the training split: there is nothing for an "
            "orthogonal or a linear map to be fitted on and both would be arbitrary"
        )

    # Orthogonal Procrustes with one global scale: with C = Xc'Yc = U S V', the
    # orthogonal Q maximising tr(Q'C) is UV', and the scale that then minimises
    # ||Yc - s Xc Q||^2 is sum(S) / tr(Xc'Xc).
    left, singular, right = torch.linalg.svd(centred_cross)
    rotation = left @ right
    scale = float(singular.sum()) / total

    lam = float(ridge) * total / width
    identity = torch.eye(width, dtype=centred_predictor.dtype, device=centred_predictor.device)
    weights = torch.cholesky_solve(
        centred_cross, torch.linalg.cholesky(centred_predictor + lam * identity)
    )

    return {
        "identity": LinearMap(None, None, None, {"free_parameters": 0}),
        "mean_shift": LinearMap(centre, None, offset, {"free_parameters": int(width)}),
        "procrustes": LinearMap(
            centre,
            scale * rotation,
            offset,
            {
                "free_parameters": int(width + width * (width - 1) // 2 + 1),
                "procrustes_scale": scale,
            },
        ),
        "ridge": LinearMap(
            centre,
            weights,
            offset,
            {
                "free_parameters": int(width + width * width),
                "ridge_relative": float(ridge),
                "ridge_absolute": lam,
            },
        ),
    }


def fit_all(
    moments: PairedMoments, *, ridge: float, progress: Callable[[str], None]
) -> dict[tuple[str, str, int], dict[str, LinearMap]]:
    """Every map the evaluation pass needs, built layer by layer.

    Layer by layer, releasing each layer's statistics as its maps are built,
    because the two together are 55 GB at 32 x 4096 while neither alone exceeds 34.
    """

    if moments.n <= moments.d_model:
        raise RuntimeError(
            f"the training split carries {moments.n} scored positions against "
            f"d_model {moments.d_model}. A full linear map has d^2 free parameters, "
            "and n <= d makes the fit underdetermined: ridge would interpolate the "
            "training split and the shuffled control would read a near-zero "
            "residual for that trivial reason rather than because the pairing "
            "carries information. Raise --train-records"
        )
    maps: dict[tuple[str, str, int], dict[str, LinearMap]] = {}
    for layer in range(moments.n_layers):
        for comparison in COMPARISONS:
            if comparison == "adjacent" and layer == moments.n_layers - 1:
                continue
            for pairing in PAIRINGS:
                maps[(comparison, pairing, layer)] = fit_maps(
                    n=moments.n,
                    predictor_sum=moments.sum_a[layer],
                    target_sum=moments.target_sum(comparison, layer),
                    predictor_moment=moments.predictor_moment(layer),
                    cross_moment=moments.cross_moment(comparison, pairing, layer),
                    ridge=ridge,
                )
        moments.release(layer)
        progress(f"  [solve] layer {layer}")
    return maps


# -------------------------------------------------------------- the evaluation


def _row_norms(value: torch.Tensor, *, role: str) -> torch.Tensor:
    """Per-position norms, refusing a degenerate row rather than reporting one.

    A zero-norm row has no direction, so a cosine involving it is undefined and a
    mean cosine taken over it would silently report a direction it does not have --
    ``23_perturbation_sensitivity.py`` refuses the same degeneracy for the same
    reason.
    """

    norms = value.norm(dim=-1)
    degenerate = int((norms == 0).sum())
    if degenerate:
        raise RuntimeError(
            f"{degenerate} held-out {role} row(s) have zero norm, so the cosine "
            "between them and anything else is undefined and a mean cosine over "
            "them would report a direction they do not have"
        )
    return norms


class HeldOut:
    """Streaming held-out residuals, cosines and norms for every cell.

    Every quantity here is a sum over positions, so the pass is a single sweep and
    the artefact is assembled from the totals afterwards. Cosine and mean norm are
    accumulated rather than derived from second moments, because neither is a
    function of them: the mean of ``||b||`` is not recoverable from the mean of
    ``||b||^2``, and a rotation and a rescaling are exactly what those two
    separate.
    """

    def __init__(
        self, *, n_layers: int, maps: dict[tuple[str, str, int], dict[str, LinearMap]]
    ) -> None:
        self.n_layers = int(n_layers)
        self.maps = maps
        self.n = 0
        self.residual: dict[tuple[str, str, str, int], float] = {}
        self.cosine: dict[tuple[str, str, str, int], float] = {}
        self.prediction_norm: dict[tuple[str, str, str, int], float] = {}
        self.denominator: dict[tuple[str, int], float] = {}
        self.target_norm: dict[tuple[str, int], float] = {}
        self.predictor_norm: dict[int, float] = {}

    def _add(self, store: dict[Any, float], key: Any, value: float) -> None:
        store[key] = store.get(key, 0.0) + value

    def _comparison(
        self,
        comparison: str,
        layer: int,
        predictor: torch.Tensor,
        true_target: torch.Tensor,
        permutation: torch.Tensor,
    ) -> None:
        training_mean = self.maps[(comparison, "true", layer)]["mean_shift"].offset
        assert training_mean is not None  # mean_shift always carries one
        key = (comparison, layer)
        self._add(self.denominator, key, float((true_target - training_mean).pow(2).sum()))
        target_norms = _row_norms(true_target, role="target")
        self._add(self.target_norm, key, float(target_norms.sum()))
        for pairing in PAIRINGS:
            target = true_target if pairing == "true" else true_target[permutation]
            norms = target_norms if pairing == "true" else target_norms[permutation]
            for method, mapping in self.maps[(comparison, pairing, layer)].items():
                prediction = mapping.apply(predictor)
                prediction_norms = _row_norms(prediction, role=f"{method} prediction")
                cell = (comparison, pairing, method, layer)
                self._add(self.residual, cell, float((target - prediction).pow(2).sum()))
                self._add(self.prediction_norm, cell, float(prediction_norms.sum()))
                self._add(
                    self.cosine,
                    cell,
                    float(((target * prediction).sum(-1) / (norms * prediction_norms)).sum()),
                )

    @torch.no_grad()
    def update(self, a: torch.Tensor, b: torch.Tensor, permutation: torch.Tensor) -> None:
        """One batch, holding at most three per-layer float64 copies at a time."""

        self.n += int(a.shape[1])
        current = a[0].to(torch.float64)
        for layer in range(self.n_layers):
            following = a[layer + 1].to(torch.float64) if layer + 1 < self.n_layers else None
            self._add(self.predictor_norm, layer, float(_row_norms(current, role="reference").sum()))
            self._comparison(
                "cross", layer, current, b[layer].to(torch.float64), permutation
            )
            if following is not None:
                self._comparison("adjacent", layer, current, following, permutation)
            current = following if following is not None else current

    def cell(self, comparison: str, pairing: str, method: str, layer: int) -> dict[str, Any]:
        key = (comparison, pairing, method, layer)
        denominator = self.denominator[(comparison, layer)]
        identity = self.residual[(comparison, pairing, "identity", layer)]
        residual = self.residual[key]
        return {
            "normalised_residual": residual / denominator,
            "identity_residual_removed": None if identity == 0.0 else 1.0 - residual / identity,
            "mean_cosine": self.cosine[key] / self.n,
            "mean_prediction_norm": self.prediction_norm[key] / self.n,
            **self.maps[(comparison, pairing, layer)][method].facts,
        }

    def layer_record(self, layer: int) -> dict[str, Any]:
        record: dict[str, Any] = {"layer": layer, "n_positions": self.n}
        for comparison in COMPARISONS:
            if comparison == "adjacent" and layer == self.n_layers - 1:
                continue
            key = (comparison, layer)
            if not self.denominator[key] > 0.0:
                raise RuntimeError(
                    f"layer {layer}'s {comparison} target is constant on the "
                    "held-out split, so the normalised residual has no denominator "
                    "and every ratio at this layer would be infinite"
                )
            record[comparison] = {
                "denominator_per_position": self.denominator[key] / self.n,
                "mean_predictor_norm": self.predictor_norm[layer] / self.n,
                "mean_target_norm": self.target_norm[key] / self.n,
                **{
                    pairing: {
                        method: self.cell(comparison, pairing, method, layer)
                        for method in METHODS
                    }
                    for pairing in PAIRINGS
                },
            }
        return record


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def summarise(layers: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-comparison, per-pairing, per-method aggregates over layers.

    The mean **over layers** of a per-layer scale-free residual, and not a residual
    pooled over all positions of all layers: the latter is dominated by the deepest
    layers, whose activation norms are an order of magnitude larger, and would
    report a depth-weighted number as a model-level one (Appendix B rule 21).
    """

    summary: dict[str, Any] = {}
    for comparison in COMPARISONS:
        present = [record for record in layers if comparison in record]
        if not present:
            raise RuntimeError(f"no layer carries the {comparison!r} comparison")
        summary[comparison] = {"n_layers": len(present)}
        for pairing in PAIRINGS:
            summary[comparison][pairing] = {}
            for method in METHODS:
                values = [
                    record[comparison][pairing][method]["normalised_residual"]
                    for record in present
                ]
                summary[comparison][pairing][method] = {
                    "mean_normalised_residual": _mean(values),
                    "min_normalised_residual": min(values),
                    "max_normalised_residual": max(values),
                }
    return summary


def verdict_record(summary: dict[str, Any]) -> dict[str, Any]:
    """The three quantities that decide R2.4, and the three readings of them.

    **No threshold is invented here.** This programme's standing rule is to prefer
    threshold-free statistics and to sweep a threshold where one is unavoidable
    (Appendix B rule 17); a cut on "how much residual removed counts as explained"
    would be neither, and it would settle the most expensive decision available
    from a number nobody had calibrated. The stage reports the three quantities and
    says what each reading would be; a person reading the artefact decides.
    """

    return {
        "statement": (
            "REPRESENTATIONAL ONLY. Every number here is a residual between two "
            "activation tensors on held-out positions. It says how much of one "
            "checkpoint's representation another's predicts under a map of a "
            "declared class; it says nothing about behaviour, and a mode whose "
            "behavioural estimand is unmeasurable still produces every number "
            "below. A difference found here stays representational until an "
            "intervention shows a corresponding behavioural change"
        ),
        "quantities": {
            "true_pairing": summary["cross"]["true"],
            "shuffled_null": summary["cross"]["shuffled"],
            "adjacent_layer_unit": summary["adjacent"]["true"],
        },
        "how_to_read_them": (
            "compare each method's cross-checkpoint TRUE residual against two "
            "things and never against one: its own SHUFFLED value at the same "
            "method, layer and ridge coefficient, which is what the estimator "
            "reports when there is no correspondence at all; and the ADJACENT-layer "
            "true residual at the same layer, which is what one layer of the "
            "reference model's own ordinary computation costs in the same units"
        ),
        "readings": {
            "a_simple_map_already_explains_the_difference": (
                "mean_shift or procrustes drives the cross-checkpoint normalised "
                "residual close to zero under true pairing, far below its own "
                "shuffled value. Then the target's representation is the "
                "reference's up to a shift, or up to a rotation and a global scale, "
                "and a Crosscoder would be learning a map that a d-vector or an "
                "orthogonal matrix already gives. This is RESEARCH_PLAN.md §4c's "
                "third reading: no Crosscoder is trained and the difference is "
                "reported as an alignable representational change"
            ),
            "no_linear_map_explains_it_and_the_gap_exceeds_one_layer": (
                "ridge leaves a substantial cross-checkpoint residual under true "
                "pairing, clearly BELOW its own shuffled value -- so the fit is real "
                "rather than capacity -- and that residual is large beside the "
                "adjacent-layer unit. Then the difference between the checkpoints is "
                "not affine and is large in the natural unit, and there is something "
                "a dictionary basis could represent that no map of this class does"
            ),
            "the_fit_is_capacity_rather_than_correspondence": (
                "ridge under true pairing sits close to ridge under shuffled "
                "pairing. Then d^2 free parameters fitted to n tokens account for "
                "most of what the map achieved, no reading of the other two "
                "quantities is interpretable at this n, and the answer is more "
                "training positions rather than a conclusion. The stage refuses "
                "n_train <= d_model outright, which is the extreme of this regime; "
                "the regime itself is visible only in this comparison"
            ),
        },
        "decision": (
            "NOT MADE HERE. No threshold is declared for 'a Crosscoder is "
            "warranted': the three quantities above are reported, the three "
            "readings are stated, and the decision belongs to a person reading them "
            "beside R2.3's per-mode dictionary result (RESEARCH_PLAN.md §4c)"
        ),
    }


# ---------------------------------------------------------------- the artefact


def checkpoint_record(
    resolved: Path,
    requested: Path,
    facts: dict[str, Any],
    model: JointReplaceable,
    *,
    role: str,
) -> dict[str, Any]:
    """One side of the comparison, identified by path and by the bytes on disk."""

    record = dict(facts)
    record.update(
        {
            "role": role,
            "requested_path": str(requested),
            "name": resolved.name,
            "weights_sha256": model.weights_digest(),
            "loader_gate": model.self_check(),
        }
    )
    return record


def artefact_name(reference: Path, target: Path, mode: str, tensor: str) -> str:
    """A file name that says which comparison this is.

    Derived rather than fixed, because a campaign runs this stage over two modes
    and two tensors into one ``--out`` directory and a constant name would let the
    second run silently overwrite the first.
    """

    def safe(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "unnamed"

    return (
        f"model_diffing__{safe(reference.name)}__to__{safe(target.name)}"
        f"__{safe(mode)}__{safe(tensor)}.json"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="directory of the checkpoint the map is fitted FROM. A path and not an "
        "arm name, for the reason 21_joint_mode_qualification.py gives: a checkpoint "
        "that has not passed that stage must not be in the panel",
    )
    parser.add_argument(
        "--target",
        type=Path,
        required=True,
        help="directory of the checkpoint the map is fitted TO. Naming the reference "
        "again makes every residual exactly zero and is a usable identity check of "
        "the whole path",
    )
    parser.add_argument(
        "--rendering",
        required=True,
        choices=joint_modes.RENDERING_NAMES,
        help="which declared family's input format BOTH checkpoints take. The set is "
        "composed by src.transfer.joint_modes, the single place either mode's format "
        "is decided (Appendix B rule 12)",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=JOINT_MODES,
        help="which mode of the two checkpoints to compare. One mode per run: the "
        "two modes have different corpora, different scored spans and different "
        "position counts, so a run that mixed them would fit one map to two "
        "populations",
    )
    parser.add_argument(
        "--tensor",
        default="block_output",
        choices=TENSORS,
        help="which per-layer tensor is compared. It reaches the artefact by name, "
        "because 'the block output' names a different tensor on different block "
        "layouts -- see "
        "src.transfer.replaceable.ReplaceableModel.perturbation_target",
    )
    parser.add_argument(
        "--train-records",
        type=int,
        default=8192,
        help="records the four maps are fitted on. The refusal that matters is on "
        "POSITIONS rather than records: n_train must exceed d_model or the ridge fit "
        "is underdetermined and the shuffled control reads near zero for a trivial "
        "reason. The default spans a whole shuffle block",
    )
    parser.add_argument(
        "--eval-records",
        type=int,
        default=2048,
        help="records every number is reported on. Drawn as part of ONE pool with "
        "the training records and split from it under a seeded permutation, so the "
        "two halves are samples of one population rather than two regions of a "
        "clustered corpus",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="eligible records skipped in file order before the seeded draw begins. "
        "A second run at a different value IS the skip-offset sensitivity (Appendix B "
        "rule 1); the seed cannot supply it, because the shuffled reservoir permutes "
        "within blocks that are read in file order",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CORPUS_DRAW_SEED,
        help="one seed with four uses, derived deterministically: the corpus stream, "
        "the train/eval split (+1), and the shuffled-pairing permutations of the "
        "training (+2) and held-out (+3) passes",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-4,
        help="ridge coefficient RELATIVE to the mean eigenvalue of the centred "
        "predictor moment; the realised absolute value is recorded per layer. "
        "Relative because a LLaMA's activation scale spans orders of magnitude "
        "across depth, so one absolute coefficient would regularise every layer "
        "differently and the depth curve would partly be a curve of that",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="device both checkpoints run on and the float64 statistics live on. One "
        "device, because the two models are compared on one batch of inputs and a "
        "second device would need a copy of every tensor to compare them at all",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="token cap. A protein rendering is never truncated: the residue ceiling "
        "is derived from this cap and the measured rendering wrapper, so a record "
        "that would not fit cannot be drawn",
    )
    parser.add_argument(
        "--protein-context",
        default=None,
        help="optional document context the protein block is embedded in, filled "
        "into the family's declared template. Omitted means the bare block -- the "
        "format stage 1 was trained on -- and whichever was used reaches the artefact "
        "and sets the residue band",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="records per forward pass. Also the scope of the shuffled-pairing "
        "permutation, which is why it is recorded beside the null",
    )
    return parser


def load_side(
    resolved: Path,
    tokenizer: Any,
    *,
    declaration: joint_modes.JointRendering,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any], JointReplaceable]:
    """One checkpoint's weights and the handle that reads them."""

    tokenisation = joint_tokenisation(tokenizer, declaration, args.mode)
    model, facts = STAGE21.load_model(
        resolved, tokenizer, device=args.device, dtype=INFERENCE_DTYPE
    )
    handle = JointReplaceable(
        model=model,
        tokenizer=tokenizer,
        checkpoint=resolved,
        declaration=declaration,
        mode=args.mode,
        tokenisation=tokenisation,
        max_tokens=args.max_tokens,
        protein_context=args.protein_context,
    )
    return tokenisation, facts, handle


def stream_split(
    reference: JointReplaceable,
    target: JointReplaceable,
    records: Sequence[tuple[str, str | None]],
    *,
    tensor: str,
    batch_size: int,
    generator: torch.Generator,
    consume: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None],
    label: str,
) -> None:
    """One forward pass per model over one split, consumed batch by batch."""

    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        if not chunk:
            continue
        a, b = paired_capture(reference, target, chunk, tensor=tensor)
        if a.shape[1] == 0:
            continue
        permutation = torch.randperm(a.shape[1], generator=generator).to(a.device)
        consume(a, b, permutation)
        del a, b
        if (start // batch_size) % 32 == 0:
            print(f"  [{label}] {start + len(chunk)}/{len(records)} records", flush=True)


def main() -> None:
    args = build_parser().parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    declaration = joint_modes.rendering(args.rendering)
    source = joint_mode_corpus(args.mode)
    corpus = corpus_location(source)
    # Every resolved path printed before anything loads: one environment variable
    # has silently narrowed a campaign's panel before (Appendix B rule 7).
    print(f"[paths] reference {Path(args.reference).resolve()}")
    print(f"[paths] target    {Path(args.target).resolve()}")
    print(f"[paths] corpus    {corpus}  ({source}, mode {args.mode})")
    print(f"[paths] out       {args.out.resolve()}")

    # The tokenizers and their comparison first, so a wrong checkpoint/family pair
    # or two different vocabularies fail in a second rather than after two
    # multi-gigabyte loads.
    reference_path, reference_tokenizer = STAGE21.load_tokenizer(Path(args.reference))
    target_path, target_tokenizer = STAGE21.load_tokenizer(Path(args.target))
    vocabulary = assert_identical_tokenizers(reference_tokenizer, target_tokenizer)

    tokenisation, reference_facts, reference = load_side(
        reference_path, reference_tokenizer, declaration=declaration, args=args
    )
    _, target_facts, target = load_side(
        target_path, target_tokenizer, declaration=declaration, args=args
    )
    shape = assert_comparable_shape(
        reference, target, reference_facts=reference_facts, target_facts=target_facts
    )
    declared_tensor = tensor_declaration(reference, args.tensor)
    print(
        f"[shape] {shape['n_layers']}L x {shape['d_model']}d, vocabulary "
        f"{vocabulary['vocabulary_sha256'][:12]}.., tensor {args.tensor}"
    )

    low, high = STAGE17.CORPUS_BAND[source]
    if tokenisation is not None:
        low, high = STAGE17.joint_protein_band(
            tokenisation,
            max_tokens=args.max_tokens,
            protein_context=args.protein_context,
        )

    def records() -> Iterator[tuple[str, str | None]]:
        return iter_corpus_records(source, min_symbols=low, max_symbols=high)

    train, evaluation, splits = draw_splits(
        records,
        n_train=args.train_records,
        n_eval=args.eval_records,
        seed=args.seed,
        skip=args.skip,
    )
    print(
        f"[cohort] {splits['train_records']} train / {splits['eval_records']} "
        f"held-out records, band {[low, high]}, {splits['blocks_spanned']} "
        "shuffle block(s)"
    )

    moments = PairedMoments(
        n_layers=reference.n_layers,
        d_model=reference.width,
        device=torch.device(args.device),
    )
    stream_split(
        reference,
        target,
        train,
        tensor=args.tensor,
        batch_size=args.batch_size,
        generator=torch.Generator().manual_seed(args.seed + 2),
        consume=moments.update,
        label="train",
    )
    n_train = moments.n
    print(f"[train] {n_train} scored positions, {n_train / moments.d_model:.1f} per d_model")

    maps = fit_all(moments, ridge=args.ridge, progress=lambda line: print(line, flush=True))
    del moments

    held_out = HeldOut(n_layers=reference.n_layers, maps=maps)
    stream_split(
        reference,
        target,
        evaluation,
        tensor=args.tensor,
        batch_size=args.batch_size,
        generator=torch.Generator().manual_seed(args.seed + 3),
        consume=held_out.update,
        label="eval",
    )
    if held_out.n == 0:
        raise RuntimeError(
            "the held-out split produced no scored positions, so nothing was "
            "evaluated; refusing to write an artefact with no number in it"
        )
    layers = [held_out.layer_record(layer) for layer in range(reference.n_layers)]
    summary = summarise(layers)

    limitations: dict[str, Any] = {
        "representational_only": (
            "a residual between two activation tensors is not a behavioural "
            "quantity. Nothing here says either checkpoint does anything "
            "differently, only that its representation is or is not predictable "
            "from the other's under a map of a declared class"
        ),
        "shuffled_null_scope": SHUFFLE_NOTE,
        "one_lineage": (
            "the comparison is defined only between checkpoints of one lineage with "
            "one tokenizer and one shape, which is why the tokenizer digest and the "
            "shape are refusals rather than recorded facts. It therefore generalises "
            "to no other pair, and in particular says nothing about two models of "
            "different modalities"
        ),
        "linear_class": (
            "the four maps are nested and all affine. A low ridge residual bounds "
            "what a non-linear method could add; a high one does not establish that "
            "the difference is meaningful, only that it is not affine"
        ),
        "one_draw": (
            "one pool at one --skip. The skip-offset sensitivity Appendix B rule 1 "
            "requires is a second run, and this artefact is one point of it"
        ),
        "precision": (
            f"both checkpoints are loaded at {INFERENCE_DTYPE} and every statistic, "
            "fit and evaluation is float64. The two are quantised identically, so "
            "the quantisation cannot favour either side, but it does put a floor "
            "under the identity residual of order the squared relative quantisation "
            "step -- far below any residual this stage would report as a finding, "
            "and not separately measured"
        ),
    }
    if args.mode == "protein":
        limitations["protein_mode_reference"] = (
            STAGE24.PROTEIN_REFERENCE_LIMITATION
            + ". This stage may still be pointed at that mode, because a "
            "representational comparison does not require a measurable behavioural "
            "estimand -- the activations exist and are comparable position by "
            "position whether or not the model does anything useful with them. What "
            "the limitation forbids is reading the verdict as a behavioural claim "
            "about either side, and the verdict says so in its own words"
        )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "provenance": {
            "runner": {
                "path": "scripts/transfer/25_model_diffing_baselines.py",
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "modules": {name: sha256_file(REPO_ROOT / name) for name in PROVENANCE_MODULES},
        },
        "estimand": (
            "the normalised residual ||B - f(A)||^2 / ||B - mean_train(B)||^2 on "
            "held-out content positions, per layer and per method, where A is the "
            "reference checkpoint's declared per-layer tensor, B the target's at the "
            "same position of the same record, and f is fitted on a disjoint "
            "training split. Scale-free, so it is comparable across layers whose "
            "activation norms differ by orders of magnitude (Appendix B rule 21); "
            "its denominator per position is recorded beside it (rule 27)"
        ),
        "reference": checkpoint_record(
            reference_path, Path(args.reference), reference_facts, reference, role="reference"
        ),
        "target": checkpoint_record(
            target_path, Path(args.target), target_facts, target, role="target"
        ),
        "tokenizer_vocabulary": vocabulary,
        "comparability": shape,
        "tensor": declared_tensor,
        "rendering": (
            tokenisation.facts()
            if tokenisation is not None
            else {
                "verdict": "NOT_RESOLVED",
                "declared_family": declaration.name,
                "reason": (
                    "the text mode's content positions are the tokenizer's own "
                    "non-special positions and do not depend on the protein "
                    "rendering, so the declared family is recorded but not resolved "
                    "against this tokenizer"
                ),
            }
        ),
        "methods": {
            "order": list(METHODS),
            "definitions": METHOD_DEFINITIONS,
            "pairings": {
                "true": "reference position i against target position i",
                "shuffled": SHUFFLE_NOTE,
            },
            "comparisons": COMPARISON_DEFINITIONS,
            "fitted_on": "the training split only; every number is on the held-out split",
        },
        "cohort": {
            "corpus": str(corpus),
            "corpus_source": source,
            "symbol_band": [low, high],
            "symbol_unit": "characters" if source == "openwebtext" else "residues",
            "input_rendering": reference.rendering_note,
            "scored_positions": reference.scoring_note,
            "splits": splits,
            "n_train_positions": n_train,
            "n_eval_positions": held_out.n,
            "n_train_over_d_model": n_train / float(shape["d_model"]),
            "batch_size": int(args.batch_size),
        },
        "seeds": {
            "corpus_stream": int(args.seed),
            "split_permutation": int(args.seed) + 1,
            "train_pairing_permutation": int(args.seed) + 2,
            "held_out_pairing_permutation": int(args.seed) + 3,
        },
        "layers": layers,
        "summary": summary,
        "verdict": verdict_record(summary),
        "limitations": limitations,
    }

    destination = args.out / artefact_name(reference_path, target_path, args.mode, args.tensor)
    write_json(destination, payload)
    print()
    for comparison in COMPARISONS:
        for pairing in PAIRINGS:
            line = "  ".join(
                f"{method} {summary[comparison][pairing][method]['mean_normalised_residual']:.4f}"
                for method in METHODS
            )
            print(f"[{comparison}/{pairing}] {line}")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
