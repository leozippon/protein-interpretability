"""When each mode of ONE joint decoder arrives at its own answer, read by depth.

Every modality coefficient this panel fits rests on a single arm. `docs/INTERPRETABILITY_TRANSFER_AUDIT.md`
§2 records the reason: the only checkpoint family spanning both modalities is
GPT-2, with five text arms against one protein arm, so ProtGPT2 alone carries the
contrast and "protein" is not separable from "this particular checkpoint".
Galactica spans both modalities at four rungs of one training run
(125m/1.3b/6.7b/30b), and `e500d14` taught the lens the architecture behind it.
This module is the measurement that spends that: a logit-lens depth trajectory
read in **both modes of the same weights**.

What the design holds fixed, and why that is the point
------------------------------------------------------
Between the two modes of one Galactica rung, the following are not merely
matched but *identical objects*: the parameters, the depth, the width, the final
layer norm, the unembedding, the tokenizer, the training run and the corpus
mixture. Only the content of the scored positions differs. No previous modality
contrast in this programme has held any of those fixed -- the matched pair
`gpt2-large`/`protgpt2` shares a shape and a vocabulary *size*, not a checkpoint.

What it therefore CANNOT do, stated before anything else it can
---------------------------------------------------------------
It is **not** a test of the limited-output-interface hypothesis
(:mod:`src.transfer.lenses`' opening paragraph). Galactica's residues are
ordinary single-letter pieces of a 50,000-token text vocabulary --
``JOINT_RENDERINGS["galactica"].residue_subspace_disjoint_from_text`` is
``False`` -- and both modes emit through the same head. So this design holds the
output interface fixed and varies the content, which is the *separation* of a
confound every earlier comparison carried, not a test of it. A reading here says
nothing about what a twenty-output protein unembedding can express.

The estimand, and why it is a depth rather than a magnitude
-----------------------------------------------------------
For one mode, at each point of the relative-depth grid, the logit lens gives a
distribution; two of its properties are read against the model's **own final**
distribution at the same position. The top-1 agreement -- the fraction of scored
positions where the lens' argmax is the model's own final argmax -- rises to
exactly 1.0 at the deepest grid point, and the KL falls to exactly 0.0 there,
both by construction, because the lens head applied to the final residual *is*
the model's head.

The primary statistic is the relative depth at which top-1 agreement first
reaches an **absolute level** (:data:`AGREEMENT_LEVELS`). It divides by nothing:
a level is a point on a scale that means the same thing in both modes, so no
per-mode normaliser enters, and it is unit-free, which is what lets it cross the
mode boundary where a nats-per-token magnitude cannot. Limitation L23 and
Appendix B rules 26/27 bar a per-token magnitude differenced across readings
whose symbols differ, and one scored protein token here is one residue while one
scored text token is a BPE piece. **One tokenizer serves both modes, which
removes the tokenizer half of that hazard and not the other half**: the symbol a
token carries still differs, so every per-token magnitude in the artefact is
reported per mode and is never differenced across modes.

The secondary statistic is the span-normalised depth of a falling quantity
(:func:`src.transfer.lenses.resolution_depth`), kept as a second functional whose
defect points the other way; :data:`AGREEMENT_QUANTITY` records both defects and
why the compound asks for both.

The primary contrast is within one checkpoint,

    delta(rung) = depth(protein_declared) - depth(text_declared),

read on one grid, so the ``1 / n_layer`` floor of any depth -- which differs
across rungs (0.083, 0.042, 0.031, 0.021) -- cancels exactly inside a rung and
does not across one. A per-mode depth is reported per rung and is a descriptive
curve; the contrast is the quantity a verdict is taken on.

The largest limitation, named rather than left to be found
-----------------------------------------------------------
No tuned lens is fitted here. The untuned logit lens carries a basis error that
grows with distance from the final layer (:mod:`src.transfer.lenses`, point 2),
and this measurement does **not** separate "the computation resolves later in
this mode" from "this mode's intermediate states sit further from the final
basis". A tuned-lens replication is the declared next measurement and no reading
here is citable as a claim about *when a computation resolves* until one exists.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace as dataclass_replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import joint_modes
from .arms import (
    INPUT_FORMAT_UNDECLARED,
    Arm,
    ArmSpec,
    Cohort,
    config_shape,
)
from .lenses import (
    LayerPoint,
    LensHead,
    ScoredWindow,
    cache_residuals,
    lens_metrics,
    lens_trajectory,
    resolution_depth,
)
from .replaceable import joint_mode_corpus
from .statistics import bootstrap_unit_floor

SCHEMA_VERSION = "r2_transfer_joint_mode_lens_v1"

# --------------------------------------------------------------- frozen draw

#: The cohort draw. These are `21_joint_mode_qualification.py`'s own defaults and
#: the values every Galactica rung was qualified under, so this stage scores the
#: identical 128 records: Swiss-Prot content digest
#: ``0dd37e88b0db6947f4f6949f516ee4af95a3d27625ef9754f3d3629cd392c70e`` and
#: OpenWebText ``2c9f8a8ea44850c0abacc2efcc66bd353840faa3369c8878f3ab3828f54c06c3``.
#: Appendix B rule 13 asks a stage to declare its band against the band its arms
#: were qualified on; here the two are the same draw and the digests are checked
#: rather than declared.
#: The seed is :data:`src.transfer.arms.DEFAULT_CORPUS_DRAW_SEED` and is not
#: restated here: it is the panel's one declaration of which records exist, the
#: value 21_joint_mode_qualification.py itself draws under, and a second copy
#: would stop tracking it the day it moves (Appendix B rule 12).
SEQUENCES = 128
PROTEIN_BAND = (64, 246)
TEXT_MIN_CHARS = 800

#: Scored window for the text mode, in tokens. Chosen **before any reading** so
#: that the two modes are matched on scored positions per record, which is also
#: what matches their position-in-context distributions: the 128-record protein
#: cohort carries 20,866 residues (mean 163.0 per record), so a 164-token text
#: window yields 163 scored targets per document and 20,864 in total, a 0.01%
#: difference. Position in context is a strong driver of every lens quantity, so
#: leaving the text window at the qualification stage's 512 would have put a
#: position confound directly on the contrast.
TEXT_WINDOW_TOKENS = 164

#: The token index above which a protein scored position has no text counterpart
#: at :data:`TEXT_WINDOW_TOKENS`. 13.9% of protein scored positions lie above it
#: (2,907 of 20,866, in the 62 records longer than 163 residues), which is what
#: :data:`MODE_POSITION_CAPPED` exists to price.
POSITION_CAP = TEXT_WINDOW_TOKENS - 1

#: The four cells one rung is measured in, over one loaded set of weights. The
#: first two are the contrast; the third prices the position tail above; the
#: fourth prices the rendering (Appendix B rule 4) in the estimand's own units.
MODE_TEXT = "text_declared"
MODE_PROTEIN = "protein_declared"
MODE_POSITION_CAPPED = "protein_declared_capped"
MODE_PROTEIN_NAIVE = "protein_naive"
MODES: tuple[str, ...] = (MODE_TEXT, MODE_PROTEIN, MODE_POSITION_CAPPED, MODE_PROTEIN_NAIVE)

#: ``text_declared`` and ``protein_declared`` are spelled exactly as
#: `41_context_information_bootstrap.py` spells the arms of a joint qualification
#: report, so a row here joins its own rung's identification verdict by name
#: rather than by a reader's inference.
GATED_MODES = (MODE_TEXT, MODE_PROTEIN)

#: The falling quantities a resolution depth is read from. KL to the model's own
#: final distribution is the primary: it is target-free, so it stays defined on a
#: mode that reads nothing from context, which the cross-entropy does not.
#: Top-1 disagreement is the robustness reading -- a different functional of the
#: same trajectory, so a depth ordering carried by the particular shape of a KL
#: curve does not survive it by construction.
#: The two families of depth statistic, and why there are two.
#:
#: ``agreement`` is the **primary**. It is the fraction of scored positions at
#: which the lens' own top-1 prediction equals the model's own final top-1, and
#: it is read at an **absolute level** -- the relative depth at which that
#: fraction first reaches 0.25, 0.50, 0.75. Both modes' curves end at exactly 1.0
#: because the deepest grid point IS the model, so the level is a point on one
#: scale that means the same thing in both modes and needs no per-mode
#: normaliser. It is also always defined for a level below one, so it produces no
#: undefined bootstrap draw.
#:
#: Its one asymmetry is a floor and it is bounded and signed. A lens predicting
#: at random agrees with the final top-1 about ``1/20`` of the time in protein
#: mode, where the final distribution lives on twenty residue letters, and about
#: ``1/50000`` in text mode. That lifts the protein curve by at most 0.05 at
#: every depth and therefore makes the protein mode reach any level EARLIER than
#: it otherwise would -- so it biases **against** a finding that the protein mode
#: resolves later and cannot manufacture one. Every level is above that floor.
#:
#: ``span`` is the **secondary**, kept as a second functional of the same
#: trajectory: the relative depth at which a falling quantity has fallen by
#: fraction ``tau`` of its own total fall
#: (:func:`src.transfer.lenses.resolution_depth`). Its defect is the mirror image
#: and is the reason it is not primary: the normaliser is the quantity's value at
#: the shallowest grid point, where the logit lens is reading a residual that is
#: still close to the embedding and tends to reproduce the current token. That
#: value is mode-dependent for reasons that have nothing to do with resolution --
#: measured on the instrument rung, the protein mode's shallowest-layer lens has
#: an entropy of 0.44 nats against the text mode's 2.46 -- so a depth normalised
#: by it inherits the asymmetry. Requiring the two families to agree is stronger
#: than either, because their defects do not point the same way.
AGREEMENT_QUANTITY = "top1_agreement_with_final"
KL_QUANTITY = "kl_to_final_nats"
DISAGREEMENT_QUANTITY = "top1_disagreement_with_final"

#: Absolute agreement levels the primary depth is read at, and the ``tau`` sweep
#: the secondary is read at. Both are swept rather than singular, because a depth
#: ordering read at one cut is a threshold result (Appendix B rule 17); the
#: ``tau`` values are :data:`src.transfer.concept_lens.RESOLUTION_TAUS`, restated
#: here only as the default a caller may override, and the levels are the same
#: three numbers on the agreement scale so the two families are read at matching
#: coarseness.
AGREEMENT_LEVELS: tuple[float, ...] = (0.25, 0.50, 0.75)
SPAN_TAUS: tuple[float, ...] = (0.25, 0.50, 0.75)

#: Sequence-cluster bootstrap. Tokens inside one record share a context, so the
#: resampling unit is the record, as it is everywhere else in the lens family.
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260826

#: A bootstrap draw in which a falling quantity does not fall across the grid has
#: no crossing point, and :func:`src.transfer.lenses.resolution_depth` returns
#: ``None`` rather than inventing one. Such draws are counted and excluded; above
#: this fraction the interval is refused outright, because a percentile taken
#: over a silently thinned draw set is not the interval it claims to be. It can
#: only bind on the ``span`` family: an agreement level is always reached.
MAX_UNDEFINED_DRAW_FRACTION = 0.05

#: The ceiling :func:`src.transfer.lenses.verify_lens_head` is held to on every
#: cell of every rung, in nats. It is a ceiling on a **known rounding term** and
#: not a threshold anything is decided at: the lens head is materialised in
#: float32 and this is how far the float32 head is allowed to sit from the
#: model's own final distribution before nothing downstream means what it claims
#: to. Stage 08 uses 1e-3, which is a float32 number; 1e-2 sits an order of
#: magnitude above the float32-forward floor and two orders below the smallest
#: number this estimand's trajectory is read at.
#:
#: **It was not chosen after seeing which rung it refused and it is not moved.**
#: EXP-R2-229 froze it before any rung was scored and it stopped
#: ``galactica-30b`` at bfloat16; EXP-R2-230 carries the same number to a
#: float32 ladder rather than relaxing it, because a lens head that does not
#: reconstruct the model's own logits is measuring nothing.
LENS_HEAD_TOLERANCE_NATS = 1e-2

#: The ceiling the instrument probe (``47_joint_mode_lens.py --stage verify``)
#: passes instead, so that a reconstruction error ABOVE
#: :data:`LENS_HEAD_TOLERANCE_NATS` is reported as a number rather than raised as
#: a stop. It exists only to keep ``verify_lens_head``'s non-finite guard live
#: while the probe measures the floor; no measurement is ever taken under it, and
#: the probe publishes its verdict against :data:`LENS_HEAD_TOLERANCE_NATS`.
DIAGNOSTIC_LENS_HEAD_CEILING_NATS = 1e3

#: What the float32 lens head's reconstruction of a **bfloat16** forward pass
#: measured, per rung, at EXP-R2-229's dispatch and re-measured at EXP-R2-230's:
#: the floor is a property of the width, and it crosses
#: :data:`LENS_HEAD_TOLERANCE_NATS` between 4096 and 7168. Recorded here as the
#: reason a precision is declared rather than inherited.
BFLOAT16_LENS_HEAD_FLOOR_NATS: dict[int, float] = {
    768: 6.05e-04,
    2048: 9.64e-04,
    4096: 1.88e-03,
    7168: 8.57e-02,
}


def agreement_key(level: float) -> str:
    return f"agreement_reaches_{level:.2f}"


def span_key(quantity: str, tau: float) -> str:
    return f"{quantity}_span_tau_{tau:.2f}"


#: The statistic the gate's first two clauses are read on, and the one its third
#: clause is read on. Named here so the gate cannot be pointed at a different
#: statistic than the one this module publishes.
PRIMARY_KEYS: tuple[str, ...] = tuple(agreement_key(level) for level in AGREEMENT_LEVELS)
SECOND_FUNCTIONAL_KEY = span_key(KL_QUANTITY, 0.50)


# ------------------------------------------------------------- declarations


def joint_arm_spec(
    checkpoint: Path,
    *,
    name: str,
    mode: str,
    config: Any,
    architecture: str,
) -> ArmSpec:
    """A per-run declaration for a joint checkpoint reached by path.

    A joint checkpoint is deliberately **not** in :data:`src.transfer.arms.PANEL`
    and must not be: `21_joint_mode_qualification.py` states that a checkpoint
    which has not passed it "must not be in ``arms.py`` at all", and `e500d14`
    reverted a ``STAGED_ARMS`` declaration on exactly that ground. Nothing here
    enters ``arms.py``. The declaration is constructed for the run and handed to
    :func:`src.transfer.arms.load_arm_spec`, which is the door
    :func:`src.transfer.scaling.register_arm_spec` names for a checkpoint that
    must be loaded without being admitted, and which carries the shape check, the
    dtype read-back, the strict-loading check and Galactica's config-declared pad
    token in one implementation rather than a second copy of each.

    One consequence is recorded rather than hidden, in the same terms
    ``register_arm_spec`` records it: the depth and width in this declaration are
    read from the checkpoint's own config, so ``load_arm_spec``'s comparison
    against that config is a tautology here and asserts nothing, where for a
    panel member it is a real check against a hand-written shape.

    ``input_format`` is :data:`src.transfer.arms.INPUT_FORMAT_UNDECLARED` for the
    protein mode, following the convention `e500d14` applied to ProteinGLM: the
    sentinel marks a rendering **no branch of** ``Cohort.input_strings`` **can
    emit**, which is true here even though the rendering is fully evidenced --
    it is declared in :mod:`src.transfer.joint_modes` and reached through
    :func:`protein_windows`, never through the panel renderer. Every renderer in
    ``arms`` raises on the sentinel, so a caller that reached for the wrong door
    is stopped rather than served a format this checkpoint was not trained on.
    """

    if mode not in ("text", "protein"):
        raise ValueError(f"unknown joint mode {mode!r}; declared: ('text', 'protein')")
    n_layer, d_model = config_shape(config)
    return ArmSpec(
        name=f"{name}:{mode}",
        path=Path(checkpoint),
        # Declared where the path is made (Appendix B rules 12 and 16). This
        # checkpoint is named by a command-line path and by no environment base,
        # so recording one would be a claim about provenance that is not true.
        path_variable="--checkpoint",
        modality=mode,
        n_layer=int(n_layer),
        d_model=int(d_model),
        tokenisation="bpe",
        input_format="raw" if mode == "text" else INPUT_FORMAT_UNDECLARED,
        evaluation_cohort_source=joint_mode_corpus(mode),
        architecture=architecture,
        pretraining_corpus="galactica_scientific_corpus",
        # The lens is the only measurement family this declaration may enter, and
        # the only one `e500d14` verified for this architecture. `opt` is
        # deliberately absent from `arms._DECOMPOSABLE`, `arms._ATTENTION_PATH`,
        # `arms._MLP_NEURON_TENSOR` and `circuits._CIRCUIT_ARCHITECTURES`, so a
        # pathway, circuit or relational request must be refused by capability
        # here rather than fail inside a hook that emits the wrong rank.
        capabilities=frozenset({"lens"}),
    )


def mode_arm(arm: Arm, spec: ArmSpec) -> Arm:
    """The same loaded weights under a second mode's declaration.

    One load serves both modes -- they are one checkpoint -- so the per-mode view
    shares the model and tokenizer objects and differs only in the declaration a
    measurement reads off it.
    """

    return dataclass_replace(arm, spec=spec)


# ----------------------------------------------------------------- windows


@dataclass(frozen=True)
class ProteinWindows:
    """Rendered protein windows and the census of what was rendered."""

    windows: list[ScoredWindow]
    census: dict[str, Any]


def protein_windows(
    arm: Arm,
    tokenisation: joint_modes.JointTokenisation,
    cohort: Cohort,
    *,
    protein_context: str | None,
    variant: str,
    batch_size: int,
    position_cap: int | None = None,
) -> ProteinWindows:
    """Scored windows for a joint checkpoint's protein mode.

    The counterpart of :func:`src.transfer.lenses.prepare_windows`, which cannot
    serve this mode: it renders through ``Cohort.input_strings`` and masks
    through ``scoring.target_rule``, and neither knows this rendering. The
    rendering, its scored span and the refusal of a tokenizer that merged
    residues all come from :mod:`src.transfer.joint_modes` -- the one place
    either is decided (Appendix B rule 12) -- so the scored symbols here are
    exactly the symbols `21_joint_mode_qualification.py` scored on the same
    records.

    ``position_cap`` restricts the mask to scored targets at or below one token
    index. It exists because the text mode's window ends at
    :data:`TEXT_WINDOW_TOKENS` while a long protein runs past it, and pricing
    that asymmetry needs the same forward pass read under both masks rather than
    an argument about whether it matters.

    A rendered protein is never truncated. Truncation would cut the closing
    delimiter and leave a scored span that was located on the untruncated string,
    so the mask would index positions the row no longer has; the refusal is
    explicit rather than a silent shortening.
    """

    if cohort.kind != "protein":
        raise ValueError(f"{arm.name}: protein windows need a protein cohort")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    pad = arm.tokenizer.pad_token_id
    if pad is None:
        raise ValueError(f"{arm.name}: tokenizer declares no pad token id")

    rendered = [
        tokenisation.render(record, context=protein_context, variant=variant)
        for record in cohort.records
    ]
    windows: list[ScoredWindow] = []
    kept: list[int] = []
    for offset in range(0, len(rendered), batch_size):
        chunk = rendered[offset : offset + batch_size]
        rows = [list(record.token_ids) for record in chunk]
        width = max(len(row) for row in rows)
        if width < 2:
            raise ValueError(f"{arm.name}: a rendered protein tokenises to fewer than two tokens")
        ids = torch.full((len(rows), width), int(pad), dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        target_mask = torch.zeros((len(rows), width - 1), dtype=torch.bool)
        for row_index, (row, record) in enumerate(zip(rows, chunk)):
            ids[row_index, : len(row)] = torch.tensor(row, dtype=torch.long)
            mask[row_index, : len(row)] = 1
            selected = [
                position
                for position in record.scored_positions
                if position_cap is None or position <= position_cap
            ]
            if not selected:
                raise ValueError(
                    f"{arm.name}: record {offset + row_index} has no scored target at or "
                    f"below token index {position_cap}"
                )
            kept.append(len(selected))
            for position in selected:
                target_mask[row_index, position - 1] = True
        windows.append(
            ScoredWindow(
                input_ids=ids.to(arm.device),
                attention_mask=mask.to(arm.device),
                target_mask=target_mask.to(arm.device),
                sequence_indices=tuple(range(offset, offset + len(rows))),
            )
        )

    residues = sum(record.n_residues for record in rendered)
    scored = sum(kept)
    lengths = [len(record.token_ids) for record in rendered]
    census = {
        "variant": variant,
        "protein_context": protein_context,
        "position_cap": position_cap,
        "n_records": len(rendered),
        "n_residues": residues,
        "n_scored_positions": scored,
        "scored_positions_per_record_mean": scored / len(rendered),
        "residues_per_scored_token": residues / scored,
        "rendered_tokens_min": min(lengths),
        "rendered_tokens_max": max(lengths),
        "scored_span_rule": tokenisation.declaration.scored_target_rule,
        "one_token_per_residue": scored == residues and position_cap is None,
    }
    return ProteinWindows(windows=windows, census=census)


def text_window_census(windows: Sequence[ScoredWindow]) -> dict[str, Any]:
    """The same census fields a protein cell publishes, for the text cell."""

    per_record = [int(count) for window in windows for count in window.target_mask.sum(dim=1)]
    scored = sum(per_record)
    return {
        "variant": joint_modes.DECLARED,
        "protein_context": None,
        "position_cap": None,
        "n_records": len(per_record),
        "n_residues": None,
        "n_scored_positions": scored,
        "scored_positions_per_record_mean": scored / len(per_record),
        "residues_per_scored_token": None,
        "rendered_tokens_min": min(int(window.attention_mask.sum(dim=1).min()) for window in windows),
        "rendered_tokens_max": max(int(window.attention_mask.sum(dim=1).max()) for window in windows),
        "scored_span_rule": "all_valid, every non-padding target after the first",
        "one_token_per_residue": None,
    }


# -------------------------------------------------------------- trajectory


def blocked_trajectory(
    arm: Arm,
    head: LensHead,
    windows: Sequence[ScoredWindow],
    layers: Sequence[int],
    *,
    block_windows: int,
    metric_chunk: int,
    max_bytes: int,
) -> dict[int, list[dict[str, float | int]]]:
    """Per-record lens sums at every grid layer, cached a block at a time.

    :func:`src.transfer.lenses.cache_residuals` holds every grid layer's residual
    for the whole cohort at once, which is 7.2 GiB of host memory at 48 layers of
    width 7168. The rows :func:`src.transfer.lenses.lens_trajectory` returns are
    **per record**, and a record lives inside exactly one block, so a block's
    rows are that block's records and concatenating the blocks in cohort order
    reproduces the whole-cohort row set. Nothing is aggregated across a block
    boundary, which is why this is a memory schedule and not a second estimator.
    """

    if block_windows < 1:
        raise ValueError("block_windows must be positive")
    grid = tuple(dict.fromkeys(int(layer) for layer in layers))
    rows_by_layer: dict[int, list[dict[str, float | int]]] = {layer: [] for layer in grid}
    for start in range(0, len(windows), block_windows):
        block = list(windows[start : start + block_windows])
        cache = cache_residuals(arm, block, grid, max_bytes=max_bytes)
        block_rows = lens_trajectory(head, cache, device=arm.device, chunk=metric_chunk)
        for layer in grid:
            rows_by_layer[layer].extend(block_rows[layer])
        del cache, block_rows
    counts = {layer: len(rows) for layer, rows in rows_by_layer.items()}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"{arm.name}: grid layers disagree on record count {counts}")
    return rows_by_layer


def level_depth(
    depths: Sequence[float], values: Sequence[float], level: float
) -> float:
    """Relative depth at which a rising quantity first reaches an absolute ``level``.

    The primary statistic. Unlike :func:`src.transfer.lenses.resolution_depth` it
    divides by nothing: the level is a point on the quantity's own scale, so no
    per-mode normaliser enters and two modes read at one level are read at the
    same place.

    It is always defined for ``0 < level < 1`` on a top-1 agreement trajectory,
    because the deepest grid point is the model's own distribution and agrees
    with itself exactly. A quantity already at or above the level at the
    shallowest grid point returns that depth rather than extrapolating below the
    grid. "First reaches" is meant literally and is what a non-monotone
    trajectory is read under: the first crossing, not the last.
    """

    if not 0.0 < level < 1.0:
        raise ValueError("an absolute level must lie strictly between zero and one")
    if len(depths) != len(values) or len(depths) < 2:
        raise ValueError("depths and values must be aligned vectors of length at least two")
    if float(values[0]) >= level:
        return float(depths[0])
    for index in range(1, len(values)):
        previous, current = float(values[index - 1]), float(values[index])
        if previous < level <= current:
            if math.isclose(previous, current):
                return float(depths[index])
            weight = (level - previous) / (current - previous)
            return float(depths[index - 1]) + weight * (
                float(depths[index]) - float(depths[index - 1])
            )
    raise ValueError(
        f"a top-1 agreement trajectory ending at {float(values[-1]):.6f} never reached "
        f"{level}; the deepest grid point must be the model's own distribution, where "
        "agreement is exactly one"
    )


def _row_matrix(rows: Sequence[Mapping[str, float | int]]) -> np.ndarray:
    """One record per row: token count, cross-entropy, KL, agreement, entropy sums."""

    return np.asarray(
        [
            [
                float(row["token_count"]),
                float(row["ce_sum"]),
                float(row["kl_sum"]),
                float(row["agreement_count"]),
                float(row["entropy_sum"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )


def _weighted_means(matrix: np.ndarray, index: np.ndarray) -> dict[str, float]:
    """Token-weighted lens rates over the records ``index`` selects.

    The vectorised twin of :func:`src.transfer.lenses.lens_metrics`, used only
    inside the resampling loop; the published point estimate comes from
    ``lens_metrics`` itself, and ``test_joint_mode_lens`` holds the two equal on
    the full sample so this cannot drift into being a second estimator.
    """

    selected = matrix[index]
    tokens = selected[:, 0].sum()
    if tokens < 1:
        raise ValueError("a resample selected no scored tokens")
    agreement = selected[:, 3].sum() / tokens
    return {
        "ce_nats": selected[:, 1].sum() / tokens,
        KL_QUANTITY: selected[:, 2].sum() / tokens,
        AGREEMENT_QUANTITY: agreement,
        DISAGREEMENT_QUANTITY: 1.0 - agreement,
        "entropy_nats": selected[:, 4].sum() / tokens,
    }


def layer_metrics(
    rows_by_layer: Mapping[int, Sequence[Mapping[str, float | int]]], layers: Sequence[int]
) -> dict[int, dict[str, Any]]:
    """The published per-layer point estimates, from the lens module's own function."""

    metrics: dict[int, dict[str, Any]] = {}
    for layer in layers:
        published = dict(lens_metrics(rows_by_layer[layer]))
        published[DISAGREEMENT_QUANTITY] = 1.0 - float(published[AGREEMENT_QUANTITY])
        metrics[int(layer)] = published
    return metrics


def depth_statistics(
    depths: Sequence[float],
    per_layer: Mapping[int, Mapping[str, Any]],
    layers: Sequence[int],
    *,
    levels: Sequence[float],
    taus: Sequence[float],
) -> dict[str, float | None]:
    """Every published depth statistic of one trajectory, in one flat mapping.

    Flat because the contrast is a difference of two modes taken statistic by
    statistic and the gate names the statistics it reads; a nested shape would
    make both of those walk a tree to find one number.
    """

    def series(quantity: str) -> list[float]:
        return [float(per_layer[int(layer)][quantity]) for layer in layers]

    out: dict[str, float | None] = {}
    agreement = series(AGREEMENT_QUANTITY)
    for level in levels:
        out[agreement_key(float(level))] = level_depth(depths, agreement, float(level))
    for quantity in (KL_QUANTITY, DISAGREEMENT_QUANTITY):
        values = series(quantity)
        for tau in taus:
            out[span_key(quantity, float(tau))] = resolution_depth(
                depths, values, float(tau)
            )
    return out


def depth_bootstrap(
    rows_by_layer: Mapping[int, Sequence[Mapping[str, float | int]]],
    depths: Sequence[float],
    layers: Sequence[int],
    *,
    levels: Sequence[float],
    taus: Sequence[float],
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Record-cluster bootstrap of every depth statistic, one draw per trajectory.

    The resampled record set is drawn **once per draw and reused at every grid
    layer**. Resampling per layer would give each layer an independent cohort and
    the resulting sequence of numbers would not be a trajectory, so a crossing
    point taken from it would not be a depth this cohort has.

    The per-draw values are returned as well as their percentiles, because the
    contrast is a difference of two independently resampled modes and has to be
    formed draw by draw rather than from two published intervals.
    """

    if resamples < 1:
        raise ValueError("resample count must be positive")
    grid = [int(layer) for layer in layers]
    matrices = {layer: _row_matrix(rows_by_layer[layer]) for layer in grid}
    n_records = matrices[grid[0]].shape[0]
    floor = bootstrap_unit_floor(n_records)
    draws: dict[str, list[float | None]] = {}
    if not floor["degenerate"]:
        generator = np.random.default_rng(seed)
        for _ in range(resamples):
            index = generator.integers(0, n_records, size=n_records)
            per_layer = {layer: _weighted_means(matrices[layer], index) for layer in grid}
            for key, value in depth_statistics(
                depths, per_layer, grid, levels=levels, taus=taus
            ).items():
                draws.setdefault(key, []).append(value)
    else:
        for level in levels:
            draws[agreement_key(float(level))] = []
        for quantity in (KL_QUANTITY, DISAGREEMENT_QUANTITY):
            for tau in taus:
                draws[span_key(quantity, float(tau))] = []
    return {
        "schema_version": SCHEMA_VERSION,
        "cluster_unit": "record",
        "resamples": int(resamples),
        "seed": int(seed),
        # The floor record is merged at the top level rather than nested, which is
        # the convention src.transfer.statistics.bootstrap_unit_floor states: a
        # caller merges it in and nulls its interval fields, so that a unit count
        # and an interval can never sit side by side without the verdict between
        # them. Below the floor no draw is taken and every interval is refused.
        **floor,
        "draws": draws,
        "intervals": {key: _draw_interval(values) for key, values in draws.items()},
    }


def _draw_interval(values: Sequence[float | None]) -> dict[str, Any]:
    """Percentiles of a draw set, refusing one thinned past the declared cap."""

    defined = [float(value) for value in values if value is not None]
    undefined = len(values) - len(defined)
    fraction = 1.0 if not values else undefined / len(values)
    if not values or fraction > MAX_UNDEFINED_DRAW_FRACTION or len(defined) < 2:
        return {
            "refused": True,
            "reason": (
                f"{undefined} of {len(values)} draws left the quantity without a "
                f"crossing point ({fraction:.3f} of the draw set, above the "
                f"{MAX_UNDEFINED_DRAW_FRACTION:.2f} cap)"
                if values
                else "no draws were taken; the record count is below the bootstrap unit floor"
            ),
            "n_defined_draws": len(defined),
            "n_undefined_draws": undefined,
            "q025": None,
            "median": None,
            "q975": None,
        }
    array = np.asarray(defined, dtype=np.float64)
    return {
        "refused": False,
        "reason": None,
        "n_defined_draws": len(defined),
        "n_undefined_draws": undefined,
        "q025": float(np.quantile(array, 0.025)),
        "median": float(np.quantile(array, 0.5)),
        "q975": float(np.quantile(array, 0.975)),
    }


def depth_contrast(
    protein: Mapping[str, Any],
    text: Mapping[str, Any],
    *,
    point_protein: Mapping[str, float | None],
    point_text: Mapping[str, float | None],
) -> dict[str, Any]:
    """``protein depth - text depth``, statistic by statistic and draw by draw.

    The two modes are scored on two corpora, so their record sets are independent
    and the draws are not paired in the statistical sense; pairing them by index
    is simply how a bootstrap of a difference of two independent statistics is
    assembled. A draw in which either side has no crossing point contributes
    nothing and is counted, and the cap that refuses a thinned one-sided interval
    refuses a thinned difference.
    """

    if protein["resamples"] != text["resamples"]:
        raise ValueError("the two modes were resampled a different number of times")
    if set(protein["draws"]) != set(text["draws"]):
        raise ValueError("the two modes published different depth statistics")
    contrast: dict[str, Any] = {}
    for key in sorted(protein["draws"]):
        differences: list[float | None] = [
            None if a is None or b is None else float(a) - float(b)
            for a, b in zip(protein["draws"][key], text["draws"][key])
        ]
        interval = _draw_interval(differences)
        left, right = point_protein.get(key), point_text.get(key)
        interval["point"] = (
            None if left is None or right is None else float(left) - float(right)
        )
        interval["excludes_zero"] = bool(
            not interval["refused"]
            and interval["q025"] is not None
            and (interval["q025"] > 0.0 or interval["q975"] < 0.0)
        )
        interval["sign"] = None if interval["point"] is None else int(np.sign(interval["point"]))
        contrast[key] = interval
    return contrast


# ------------------------------------------------------------------- gate


#: The verdict vocabulary, fixed before any trajectory exists.
VERDICTS = (
    "modality_depth_separation",
    "no_modality_depth_separation",
    "refused",
)


def mode_gate(contrast: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The one compound this measurement is decided on, per rung.

    Three clauses, each answering a way the reading could be an artefact rather
    than a difference between the two modes of one checkpoint:

    1. the protein-minus-text depth at which top-1 agreement with the model's own
       final prediction reaches an absolute level excludes zero at **every** level
       of :data:`AGREEMENT_LEVELS`. One level would be a threshold result, which
       Appendix B rule 17 does not admit.
    2. its sign is the same at every level. A sweep whose intervals all exclude
       zero with the sign changing across it is not one ordering.
    3. the span-normalised KL depth at ``tau = 0.50`` carries that same sign with
       its own interval excluding zero. A reading that lives in one functional of
       the trajectory does not survive a second one whose defect points the other
       way.

    Every clause is attainable in both directions by construction: a depth
    difference ranges over ``+/- (1 - 1/n_layer)`` and the primary statistic is
    defined on every trajectory that ends at the model itself. Appendix B rule 2
    still requires the control to be checked before a protein reading is taken,
    and the stage does that on the text mode of the same weights.
    """

    missing = [key for key in (*PRIMARY_KEYS, SECOND_FUNCTIONAL_KEY) if key not in contrast]
    if missing:
        raise KeyError(f"the contrast does not carry {missing}")
    refused = [key for key in PRIMARY_KEYS if contrast[key]["refused"]]
    if refused:
        return {
            "verdict": "refused",
            "clauses": None,
            "direction": None,
            "reason": f"the primary contrast interval was refused at {refused}",
        }
    excludes = {key: bool(contrast[key]["excludes_zero"]) for key in PRIMARY_KEYS}
    signs = {key: contrast[key]["sign"] for key in PRIMARY_KEYS}
    second = contrast[SECOND_FUNCTIONAL_KEY]
    middle = agreement_key(0.50)
    clause_one = all(excludes.values())
    clause_two = None not in signs.values() and len(set(signs.values())) == 1
    clause_three = bool(
        not second["refused"]
        and second["excludes_zero"]
        and second["sign"] == signs[middle]
    )
    passed = clause_one and clause_two and clause_three
    return {
        "verdict": "modality_depth_separation" if passed else "no_modality_depth_separation",
        "clauses": {
            "agreement_interval_excludes_zero_at_every_level": clause_one,
            "agreement_sign_invariant_across_the_level_sweep": clause_two,
            "kl_span_depth_agrees_at_tau_0.50": clause_three,
            "per_level_excludes_zero": excludes,
            "per_level_sign": signs,
            "second_functional": {
                "key": SECOND_FUNCTIONAL_KEY,
                "excludes_zero": bool(second["excludes_zero"]),
                "sign": second["sign"],
                "refused": bool(second["refused"]),
            },
        },
        "reason": None,
        "direction": (
            None
            if not passed
            else ("protein_resolves_deeper" if signs[middle] > 0 else "protein_resolves_shallower")
        ),
    }


#: The binding ceiling, carried into every artefact this measurement writes so
#: that a reader of one file does not have to find the freeze to know what the
#: numbers in it may not be used for. Frozen at registration (EXP-R2-229) and
#: not softened by a result.
CEILING: tuple[str, ...] = (
    "NOT a test of the limited-output-interface hypothesis. Galactica's residues "
    "are ordinary single-letter pieces of a 50000-token text vocabulary and both "
    "modes emit through one head, so this design separates output-interface size "
    "from content modality and cannot test the former",
    "NOT a causal claim about scale. Depth, width and parameter count co-vary "
    "across 12x768 / 24x2048 / 32x4096 / 48x7168, and all four rungs come from "
    "one training run on one corpus mixture",
    "NOT a claim about when a computation resolves. No tuned lens is fitted, so "
    "the untuned logit lens' basis error is present in both modes and is not "
    "separated; this cannot distinguish 'the computation resolves later in this "
    "mode' from 'this mode's intermediate states sit further from the final "
    "basis'. A tuned-lens replication is the declared next measurement",
    "NOT causal at all. A lens is a correlational read of what a residual stream "
    "projects to through the model's own head; nothing here intervenes",
    "One checkpoint family, one rendering, one seed, ONE draw. No second-draw "
    "sensitivity is run and every interval is within-draw",
    "NO per-token magnitude crosses the mode boundary. Only the dimensionless "
    "depths do; L23 and Appendix B rules 26/27 bar the rest, and one shared "
    "tokenizer removes only the tokenizer half of that hazard",
    "Nothing here reaches a pure-protein decoder, and the panel's modality "
    "coefficient is not refitted",
    "Galactica's pretraining corpus is not identified, so nothing here is "
    "contamination-controlled and F15 stands over any reading",
    "galactica-125m is an instrument and not a rung of the result: its protein "
    "mode is not identified on the qualification cohort",
)


def grid_record(grid: Sequence[LayerPoint], n_layer: int) -> list[dict[str, Any]]:
    return [
        {
            "layer": point.layer,
            "relative_depth": point.relative_depth,
            "depth_fractions": list(point.depth_fractions),
            "n_layer": int(n_layer),
        }
        for point in grid
    ]


def monotone_non_increasing(values: Sequence[float]) -> bool:
    return all(b <= a + 1e-12 for a, b in zip(values, values[1:]))


def trajectory_record(
    grid: Sequence[LayerPoint], metrics: Mapping[int, Mapping[str, Any]]
) -> dict[str, Any]:
    """The trajectory as it reaches the artefact, with its own sanity anchors."""

    layers = [point.layer for point in grid]
    series = {
        quantity: [float(metrics[layer][quantity]) for layer in layers]
        for quantity in (
            AGREEMENT_QUANTITY,
            KL_QUANTITY,
            DISAGREEMENT_QUANTITY,
            "ce_nats",
            "entropy_nats",
        )
    }
    kl = series[KL_QUANTITY]
    agreement = series[AGREEMENT_QUANTITY]
    # The deepest grid point is the model's own final residual read through the
    # model's own head, so its lens distribution IS the model's distribution:
    # zero KL to itself and exact agreement with itself. Both are checked rather
    # than assumed, because everything downstream is measured against that point
    # and an architecture whose block list or final norm were resolved to the
    # wrong object would fail here rather than publish a trajectory.
    if not math.isclose(kl[-1], 0.0, abs_tol=1e-9):
        raise FloatingPointError(
            "the deepest grid point is the model's own distribution, so its KL to "
            f"itself must be zero; got {kl[-1]:.3e}"
        )
    if not math.isclose(agreement[-1], 1.0, abs_tol=1e-12):
        raise FloatingPointError(
            "the deepest grid point must agree with itself on every scored position; "
            f"got {agreement[-1]:.12f}"
        )
    return {
        "layers": layers,
        "relative_depth": [point.relative_depth for point in grid],
        **series,
        "scored_tokens": int(metrics[layers[0]]["scored_tokens"]),
        "records": int(metrics[layers[0]]["sequences"]),
        "kl_monotone_non_increasing_with_depth": monotone_non_increasing(kl),
        "agreement_monotone_non_decreasing_with_depth": monotone_non_increasing(
            [-value for value in agreement]
        ),
        "ce_monotone_non_increasing_with_depth": monotone_non_increasing(series["ce_nats"]),
        "falls_across_the_grid": {
            quantity: bool(series[quantity][0] > series[quantity][-1])
            for quantity in (KL_QUANTITY, DISAGREEMENT_QUANTITY)
        },
        "agreement_at_shallowest_grid_point": agreement[0],
        "agreement_reaches_every_level": {
            f"{level:.2f}": bool(max(agreement) >= level) for level in AGREEMENT_LEVELS
        },
    }
