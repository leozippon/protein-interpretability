"""The block a replacement model replaces, on a MoE decoder and on a dense one.

**Why this exists.** Every replacement-faithfulness number this programme owns
was measured on ProGen3-112M: a sparse-MoE *protein* decoder read through a
loader that only that checkpoint needs. So "a transcoder replacement recovers
11-16% of the ablation gap and fails its causal gate" (L25) cannot be attributed
-- to protein, to mixture-of-experts, or to transcoder replacement as such --
because the design has no text control and no dense control. Standing rule 2
requires a gate be shown attainable on the text control before it is applied to a
protein arm, and §5's organising rule says a limitation shown on the text control
is a property of the METHOD while one appearing only on protein arms is a
property of the TRANSFER. Neither reading is available from one arm.

**The estimand transfers exactly, and that is the fact this module rests on.**
ProGen3's transcoder reads a MoE block's input -- the output of
``post_attention_layernorm`` -- and predicts the block's output *before* the
residual add. ``GPT2Block.forward`` does::

    residual = hidden_states
    hidden_states = self.ln_2(hidden_states)
    feed_forward_hidden_states = self.mlp(hidden_states)
    hidden_states = residual + feed_forward_hidden_states

so a forward hook on ``block.mlp`` sees exactly those two tensors: its input is
the post-attention residual normalised, and its output is what the residual
stream is about to have added to it. :meth:`DenseReplaceable.self_check`
verifies that identity on the loaded model rather than trusting this paragraph.

**What is shared and what is not.** The two implementations differ only where
the models differ. ProGen3's operations are *delegated to*
:mod:`src.transfer.progen3`, never restated here, so the arm that produced every
published number keeps its own loader, its own scoring convention and its own
residue mask (Appendix B rule 12). The dense arm reaches its checkpoint,
its rendering and its attention output projection through the declarations that
already own them -- :data:`src.transfer.arms.PANEL`,
:meth:`src.transfer.arms.Cohort.input_strings` and
:func:`src.transfer.path_patching.attention_output_projection` -- for the same
reason. What this module adds is one interface over both, so that a stage
carries no per-architecture branch of its own.

**What a dense arm brings that ProGen3 does not.** ProGen3 needs a loader gate
because ``from_pretrained`` returns a random-expert model without raising (L24).
A panel arm loads through :func:`src.transfer.arms.load_arm`, which checks depth,
width and dtype -- but "did it load" is exactly the question L24 says is not
enough, and this modality pair has a second silent failure of its own: feeding
ProtGPT2 an unwrapped sequence costs **1.42 nats/token** and makes a correctly
loaded model look untrained (L11). :func:`dense_self_check` therefore scores
frozen inputs *through the arm's declared rendering* and refuses outside a
measured band, and it checks the estimand identity above at the same time.

**Two estimands, two admissible sets, one identity check.** A *replacement*
needs the block's input as well as its output, so it needs the serial layout
above: :data:`DENSE_ARCHITECTURES` names it and :func:`eligible_arms` composes
the arms stages 15, 17 and 22 may run. A *perturbation* -- the tolerance curve
``23_perturbation_sensitivity.py`` measures -- reads only the output, and
therefore needs the weaker claim that the intercepted tensor is a term the
residual stream receives. :data:`RESIDUAL_WRITE` declares that per architecture
and :func:`perturbable_arms` composes the wider set, which adds the ProGen2
lineage. ProGen2's block is GPT-J-style parallel -- ``hidden = attn_out + ff_out
+ residual``, both terms read from one pre-attention ``ln_1`` -- so there is no
sequential block output of the kind GPT-2's interception rests on, and the
declared reconstruction is ``(attn_out + ff_out) + residual`` instead.
:meth:`DenseReplaceable.estimand_identity` verifies whichever the declaration
names, exactly, on the live forward pass; the serial reconstruction applied to a
ProGen2 block misses by 14.25 at bfloat16, so the check is load-bearing rather
than decorative.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import torch

from . import joint_modes
from .arms import (
    PANEL,
    STAGED_ARMS,
    Arm,
    Cohort,
    arm_spec,
    conditioning_boundary_ids,
    load_arm_spec,
    rendering_marker_ids,
    tokenize_batch,
)
from .io import sha256_file
from .path_patching import attention_output_projection
from .scoring import sequence_target_mask, target_rule
from .progen3 import (
    PROGEN3_CHECKPOINT,
    SELF_CHECK_SEQUENCES,
    Component,
    ProGen3,
    ablated as progen3_ablated,
    component_grid,
    content_mask as progen3_content_mask,
    forward as progen3_forward,
    load_progen3,
    moe_intercept,
    scored_logits as progen3_scored_logits,
    self_check as progen3_self_check,
    token_nll,
)

#: The name ``--arm`` takes for the external ProGen3 baseline. It is not a member
#: of :data:`src.transfer.arms.PANEL` -- it is a third-party checkpoint this
#: programme gates rather than a panel member it measures -- so it cannot be
#: named through the panel and is named here, once.
PROGEN3_ARM = "progen3"

#: Architectures whose blocks carry the ProGen3 **replacement** estimand
#: unchanged: a block whose *input* is the post-attention normalisation and whose
#: output is added to the residual stream.
#:
#: ``gpt2`` only, and deliberately. ``progen`` (ProGen2) is a *parallel* residual
#: block -- its feed-forward reads the same normalisation as its attention, not a
#: post-attention one -- so a transcoder trained at that tap predicts a different
#: object and the two arms' recovery ratios would not be the same measurement.
#: The rotary lineages (``llama``, ``qwen2``) do have the serial layout, and are
#: still excluded: nothing here has been verified against them, and the identity
#: check in :meth:`DenseReplaceable.self_check` is a verification of the arm that
#: runs, not a licence for the ones that have not.
#:
#: This set gates the *transcoder* stages (15, 17, 22) through
#: :func:`eligible_arms`, and it is deliberately narrower than
#: :data:`RESIDUAL_WRITE`: a measurement that reads only the block's **output**
#: needs the weaker of the two claims, and :func:`perturbable_arms` composes
#: that one.
DENSE_ARCHITECTURES = frozenset({"gpt2"})


@dataclass(frozen=True)
class ResidualWrite:
    """How one architecture's block writes its feed-forward term into the residual.

    The declaration :meth:`DenseReplaceable.block_intercept` is verified against.
    It answers the two questions that decide whether an intercepted feed-forward
    output *is* the term the residual stream receives, and it answers them per
    architecture rather than by trying attribute names:

    ``residual_norm``
        the normalisation whose **input** is the residual the block writes into.
        On a serial GPT-2 block that is ``ln_2``, applied after the attention has
        already been added; on a parallel GPT-J-style block it is ``ln_1``, whose
        input is the block's own input because nothing has been added yet.
    ``parallel_attention``
        whether the attention sublayer writes into the **same** sum. False on a
        serial block, where the attention term is already inside the residual by
        the time the feed-forward runs; True on a parallel one, where
        ``hidden = attn_out + ff_out + residual`` is a single three-term add.

    The attention and feed-forward *modules* are not named here. They are
    resolved through :meth:`src.transfer.arms.Arm.attention` and
    :meth:`src.transfer.arms.Arm.mlp`, which are the panel's own declarations of
    where each architecture keeps them (Appendix B rule 12); a second copy of
    ``"attn"`` in this file is exactly the drift that declaration exists to stop.
    """

    residual_norm: str
    parallel_attention: bool

    @property
    def identity(self) -> str:
        """The equality a live forward pass must satisfy exactly, as a sentence.

        Read by both :meth:`DenseReplaceable.estimand_identity`, which checks it,
        and :attr:`DenseReplaceable.perturbation_target`, which publishes it, so
        the artefact cannot claim an identity other than the one that was tested.
        """

        if self.parallel_attention:
            return (
                "block output == (attention output + intercepted feed-forward "
                f"output) + {self.residual_norm} input"
            )
        return (
            f"block output == {self.residual_norm} input + intercepted "
            "feed-forward output"
        )


#: Where each architecture's per-layer feed-forward output lands in the residual
#: stream, keyed by :attr:`src.transfer.arms.ArmSpec.architecture`.
#:
#: **The perturbation target is the feed-forward output on both layouts**, and
#: that is the point of declaring them together. Stage 23 perturbs the tensor a
#: block adds to the residual stream through its feed-forward; on ``gpt2`` that
#: tensor is the whole of the block's residual write, and on ``progen`` it is one
#: of two terms in it. The *tensor* is the same object in both cases -- the MLP's
#: own output, before anything is added to it -- so a tolerance curve measured
#: across the two is a curve over one manipulation. What differs is the identity
#: that has to hold for the interception to be that tensor, which is why the
#: declaration carries the identity rather than a prose note.
#:
#: An undeclared architecture raises in :func:`residual_write`. Duck-typing an
#: attribute called ``mlp`` would perturb *a* tensor on a LLaMA or an OPT block
#: and report it as this measurement without saying so.
RESIDUAL_WRITE: dict[str, ResidualWrite] = {
    "gpt2": ResidualWrite(residual_norm="ln_2", parallel_attention=False),
    "progen": ResidualWrite(residual_norm="ln_1", parallel_attention=True),
}

#: The parallel-residual architectures, derived from the table above rather than
#: listed again: ``parallel_attention`` is the property that decides it.
PARALLEL_ARCHITECTURES = frozenset(
    name for name, write in RESIDUAL_WRITE.items() if write.parallel_attention
)


def residual_write(architecture: str) -> ResidualWrite:
    """The declared residual write for an architecture, refusing an undeclared one.

    Answerable from an architecture name alone, so a stage refuses an arm whose
    block layout nobody verified before a checkpoint reaches the GPU.
    """

    declared = RESIDUAL_WRITE.get(architecture)
    if declared is None:
        raise TypeError(
            f"no residual write is declared for {architecture!r}, so the tensor a "
            "perturbation would be applied to has not been identified on it "
            f"(declared: {sorted(RESIDUAL_WRITE)}). Duck-typing an attribute called "
            "'mlp' would perturb a tensor whose relationship to the residual stream "
            "is unverified and report it as this measurement"
        )
    return declared


if not DENSE_ARCHITECTURES <= set(RESIDUAL_WRITE):
    raise AssertionError(
        f"{sorted(DENSE_ARCHITECTURES - set(RESIDUAL_WRITE))} carry the replacement "
        "estimand but declare no residual write, so the identity that estimand "
        "rests on could not be checked"
    )

#: The ``Component.kind`` of the block each model family replaces. ProGen3 keeps
#: ``moe_block``, which is what every frozen artefact under
#: ``results/transfer/external_baseline/`` names its per-block family; a dense arm
#: has no experts and says so.
PROGEN3_BLOCK_KIND = "moe_block"
DENSE_BLOCK_KIND = "mlp_block"

#: ProGen3's transcoders are trained at UniRef50 scale and scored on Swiss-Prot.
#: The two corpora are declared separately below because for this arm they
#: differ, and a single "corpus" field would have to be wrong about one of them.
PROGEN3_TRAINING_CORPUS = "uniref50"
PROGEN3_EVALUATION_COHORT_SOURCE = "swissprot"


def arm_cohort_kind(arm: str) -> str:
    """The :class:`src.transfer.arms.Cohort` kind an arm can be fed."""

    if arm == PROGEN3_ARM:
        return "protein"
    return "text" if arm_spec(arm).modality == "text" else "protein"


def arm_evaluation_cohort_source(arm: str) -> str:
    """The corpus an arm's evaluation cohort is drawn from.

    The panel already declares this per arm; ProGen3 is not a panel member and
    declares it here. Answerable before a checkpoint is loaded, because a stage
    draws its cohort first and a missing corpus should fail in a second rather
    than after a model is on the GPU.
    """

    if arm == PROGEN3_ARM:
        return PROGEN3_EVALUATION_COHORT_SOURCE
    return arm_spec(arm).evaluation_cohort_source


def arm_training_corpus(arm: str) -> str:
    """The corpus a replacement for this arm is trained on.

    Not always :func:`arm_evaluation_cohort_source`. ProGen3's transcoders are
    trained on UniRef50 and scored on Swiss-Prot, which is what its published
    runs did. A dense arm trains on the corpus its own cohorts are drawn from, so
    that training and evaluation see one population: the alternative -- each
    arm's own pretraining corpus -- is not available symmetrically (WebText is
    not public, OpenWebText is its replication and is what the panel evaluates
    on) and would reintroduce the train/eval population gap EXP-R2-135 priced at
    4.1x in NLL recovery.
    """

    if arm == PROGEN3_ARM:
        return PROGEN3_TRAINING_CORPUS
    return arm_spec(arm).evaluation_cohort_source


def checkpoint_weights_files(checkpoint: Path) -> list[Path]:
    """A checkpoint directory's weight files, preferring safetensors.

    Declared rather than globbed for anything readable: a directory that carries
    both formats would otherwise digest whichever the glob happened to order
    first, and two runs of one checkpoint would disagree about its identity.
    """

    for pattern in ("*.safetensors", "pytorch_model*.bin"):
        found = sorted(Path(checkpoint).glob(pattern))
        if found:
            return found
    raise FileNotFoundError(
        f"{checkpoint} carries no *.safetensors and no pytorch_model*.bin, "
        "so the loaded weights cannot be identified in the artefact"
    )


def checkpoint_weights_digest(checkpoint: Path) -> str:
    """SHA-256 of the weight file, or of the shard digests when there are several."""

    files = checkpoint_weights_files(checkpoint)
    if len(files) == 1:
        return sha256_file(files[0])
    combined = hashlib.sha256()
    for path in files:
        combined.update(f"{path.name} {sha256_file(path)}\n".encode())
    return combined.hexdigest()


# --------------------------------------------------------------- the interface


class ReplaceableModel(ABC):
    """A decoder whose per-layer blocks a replacement model can be spliced into.

    The surface is exactly what ``15_replacement_faithfulness.py`` and
    ``17_train_transcoder.py`` consume, and no more. Two members are worth
    naming because they are where a second copy of a decision would otherwise
    appear:

    :meth:`render` is the only place a record becomes a model input. Rendering
    is worth 1.42 nats/token on ProtGPT2 and a duplicated copy of that decision
    already caused one retraction (audit §0.1), so a stage renders through this
    and never by hand. ``ec_labels`` carries the conditioning prompt an
    ``ec_conditioned`` arm cannot be rendered without; every implementation
    refuses labels it would silently drop and refuses to render without labels it
    needs, because both failures are invisible in the numbers that follow.

    :meth:`block_intercept` is the only place a block's input and output are
    read or substituted. ``fn(layer, block_input, block_output)`` returns a
    tensor to substitute for the output, or ``None`` to leave it alone, which is
    the one primitive capture, replacement and mean ablation all need.
    """

    #: The ``--arm`` value that selects this model.
    name: str
    #: ``Component.kind`` of the blocks this model's replacement replaces.
    block_kind: str
    #: How the checkpoint was loaded, for the artefact's condition block.
    loading_note: str
    #: Which direction(s) the scoring convention covers, for the same block.
    scoring_note: str
    #: Which declaration turned a record into the string this model was fed.
    #: Declared per implementation rather than written into a stage's condition
    #: block, because the three do genuinely differ -- a panel arm renders
    #: through the panel, ProGen3's own batch preparer renders itself, and a
    #: joint checkpoint renders through the family declaration in
    #: :mod:`src.transfer.joint_modes` -- and a stage that stated one sentence
    #: for all three would put a false statement about the fed string into the
    #: artefact of the arm whose fed string is worth 1.42 nats (L11).
    rendering_note: str

    @property
    def cohort_kind(self) -> str:
        """Which :class:`src.transfer.arms.Cohort` kind this model can be fed.

        The corpus *names* are deliberately not properties here: a stage resolves
        them with :func:`arm_evaluation_cohort_source` and
        :func:`arm_training_corpus` **before** loading a checkpoint, so that a
        host which has not staged the corpus fails in a second rather than after
        a model is on the GPU.
        """

        return arm_cohort_kind(self.name)

    # -- shape -------------------------------------------------------------

    @property
    @abstractmethod
    def n_layers(self) -> int: ...

    @property
    @abstractmethod
    def n_heads(self) -> int: ...

    @property
    @abstractmethod
    def width(self) -> int:
        """Residual-stream width: the ``d_model`` a replacement is fitted at."""

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @property
    @abstractmethod
    def checkpoint(self) -> Path: ...

    @abstractmethod
    def weights_digest(self) -> str:
        """SHA-256 identifying the loaded weights, for the artefact."""

    # -- inputs ------------------------------------------------------------

    @abstractmethod
    def render(
        self, records: Sequence[str], *, ec_labels: Sequence[str | None] | None = None
    ) -> list[str]:
        """Cohort or corpus records as this model's own input strings."""

    @abstractmethod
    def batch(self, inputs: Sequence[str]) -> dict[str, torch.Tensor]:
        """Model kwargs for one batch of rendered inputs, on this model's device."""

    @abstractmethod
    def content_mask(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Positions carrying content: everything but padding and markers."""

    def forget_rendered(self) -> None:
        """Drop whatever per-record state :meth:`render` kept, once it is batched.

        A no-op for the two implementations that keep none: a panel arm and
        ProGen3 re-derive everything a batch needs from the input string.
        :class:`JointReplaceable` cannot -- a protein record's scored span is
        located by the rendering rule and is not recoverable from the rendered
        text -- so it keeps one record per rendered string, and a trainer that
        renders a fresh batch every step for tens of thousands of steps would
        otherwise accumulate them all. Declared here so a stage calls it once
        and carries no per-implementation branch.
        """

    # -- running -----------------------------------------------------------

    @abstractmethod
    def run(self, batch: dict[str, torch.Tensor]) -> Any:
        """One forward pass. The single spelling of this model's forward call."""

    @abstractmethod
    def scored_logits(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(logits, targets, mask)`` aligned for next-token scoring, fp32 logits."""

    @abstractmethod
    def block_intercept(
        self, fn: Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor | None]
    ) -> Any:
        """Context manager reading or replacing every replaceable block's output."""

    @property
    @abstractmethod
    def perturbation_target(self) -> dict[str, Any]:
        """What tensor :meth:`block_intercept` hands a stage, in the artefact's words.

        Declared rather than described in a stage's prose, and abstract rather
        than defaulted, because "the block output" names a different tensor on a
        serial and on a parallel residual block and a reader of two arms' numbers
        has no other way to tell which was perturbed. Every implementation states
        the tensor, the layout it sits in, the identity that was verified on the
        live forward pass, and -- where the block writes the residual through more
        than one path -- what was *not* perturbed.
        """

    # -- components --------------------------------------------------------

    def component_families(self) -> tuple[str, ...]:
        """The families :meth:`ablated` will zero on this model, in grid order.

        Attention heads and the replaceable block, which is what every model that
        declares an attention output projection carries. An implementation that
        cannot ablate one of them overrides this and says why, and that override is
        the *only* place the difference is spelled: :meth:`components` builds the
        grid from it and :meth:`ablated` refuses from it, so a stage cannot be
        handed a component the model will not accept.
        """

        return ("attention_head", self.block_kind)

    def components(self) -> list[Component]:
        """Every component this model can be ablated on, in layer order.

        The grid is :func:`src.transfer.progen3.component_grid` restricted to
        :meth:`component_families`, so the ordering the saved effect matrices are
        indexed by stays that one declaration's.

        **It is restricted rather than assumed**, and that is a repair. This
        method used to emit every attention head unconditionally while
        :meth:`JointReplaceable.ablated` refused every one of them, so a joint
        checkpoint reached the causal sweep -- after the whole behavioural sweep
        had been computed and before anything was written -- and raised there.
        Four scoring runs of a live campaign were discarded at their last step
        that way, and re-dispatching reproduced it exactly.
        """

        families = self.component_families()
        return [
            component
            for component in component_grid(
                self.n_layers, self.n_heads, block_kind=self.block_kind
            )
            if component.kind in families
        ]

    @abstractmethod
    def ablated(self, component: Component) -> Any:
        """Context manager zeroing one component's contribution to the residual."""

    # -- gates -------------------------------------------------------------

    @abstractmethod
    def self_check(self) -> dict[str, Any]:
        """Refuse to be measured unless a scored quantity is in its declared band."""


# ------------------------------------------------------------------- ProGen3


class ProGen3Replaceable(ReplaceableModel):
    """ProGen3-112M behind the shared interface.

    Every operation delegates to :mod:`src.transfer.progen3`. Nothing is
    reimplemented, so the arm that produced every published external-baseline
    number is the same computation it was before this module existed.
    """

    name = PROGEN3_ARM
    block_kind = PROGEN3_BLOCK_KIND
    loading_note = (
        "eager MoE, converted from the released megablocks packing by "
        "src.transfer.progen3; from_pretrained's own eager path returns random "
        "experts without raising"
    )
    scoring_note = (
        "N->C only (the reverse direction doubles every sweep and this stage runs "
        "one per component per model)"
    )
    rendering_note = (
        "src.transfer.progen3's own batch preparer, which adds the terminus and "
        "direction tokens itself; a corpus record IS the input string here"
    )

    def __init__(self, pg: ProGen3) -> None:
        self.pg = pg

    @property
    def n_layers(self) -> int:
        return self.pg.n_layers

    @property
    def n_heads(self) -> int:
        return self.pg.n_heads

    @property
    def width(self) -> int:
        return int(self.pg.config.hidden_size)

    @property
    def device(self) -> torch.device:
        return self.pg.device

    @property
    def checkpoint(self) -> Path:
        return self.pg.checkpoint

    def weights_digest(self) -> str:
        return sha256_file(self.pg.checkpoint / "model.safetensors")

    def render(
        self, records: Sequence[str], *, ec_labels: Sequence[str | None] | None = None
    ) -> list[str]:
        # ProGen3's batch preparer adds the terminus and direction tokens itself,
        # so a record IS the input string and wrapping it here would render twice.
        if ec_labels is not None and any(label is not None for label in ec_labels):
            raise ValueError(
                "ProGen3 has no conditioning prompt; EC labels handed to it would "
                "be dropped and the run would look like a conditioned one"
            )
        return list(records)

    def batch(self, inputs: Sequence[str]) -> dict[str, torch.Tensor]:
        return self.pg.batch(list(inputs))

    def content_mask(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return progen3_content_mask(self.pg, batch["input_ids"])

    def run(self, batch: dict[str, torch.Tensor]) -> Any:
        return progen3_forward(self.pg, batch)

    def scored_logits(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return progen3_scored_logits(self.pg, batch)

    def block_intercept(
        self, fn: Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor | None]
    ) -> Any:
        return moe_intercept(self.pg, fn)

    #: The normalisation whose input is the residual each block's output is added
    #: to. One declaration, read by :meth:`estimand_identity` and by
    #: :attr:`perturbation_target`, so the equality that is checked and the
    #: sentence that reports it cannot come apart.
    residual_norm = "post_attention_layernorm"

    @property
    def identity(self) -> str:
        return (
            f"block output == {self.residual_norm} input + intercepted MoE "
            "block output"
        )

    @property
    def perturbation_target(self) -> dict[str, Any]:
        return {
            "tensor": (
                "the MoE block's output, before the residual add: what "
                "src.transfer.progen3.moe_intercept hands its callback"
            ),
            "block_layout": (
                f"serial: the block reads {self.residual_norm} and its output is "
                "the whole of the term added to the residual stream"
            ),
            "identity_verified": (
                f"{self.identity}, verified exactly on the live forward pass by "
                "this arm's own self_check, which runs it before its likelihood "
                "band -- the same certification the dense arms carry"
            ),
            "not_perturbed": (
                "the attention contribution, which this block layout has already "
                "added to the residual before the block runs"
            ),
        }

    def ablated(self, component: Component) -> Any:
        return progen3_ablated(self.pg, component)

    @torch.no_grad()
    def estimand_identity(self) -> dict[str, Any]:
        """Verify that the intercepted MoE output IS this block's residual write.

        :meth:`DenseReplaceable.estimand_identity`'s check on ProGen3's own
        modules, and it is here because this arm is the **reference** every dense
        arm is compared against. It was certified by declaration while they were
        certified by measurement; EXP-R2-181 closed that gap by measuring it and
        found exactly zero on all ten layers. That measurement establishes one
        checkpoint and one conversion, and this gate is what keeps it true.

        **What it would catch, and nothing else in this stack would.** The eager
        MoE conversion is this repository's own code, and
        :func:`src.transfer.progen3.moe_intercept` unwraps a
        ``(hidden_states, router_probabilities)`` tuple before handing the hidden
        half on. If either ever dropped the router term or addressed a different
        submodule, the tap would stop being the residual write while every number
        derived from it still looked well formed, and every dense arm would be
        compared against a reference that had moved. The likelihood band does not
        see this: it scores the model's own forward pass, which the interception
        does not disturb.

        The tolerance is **exact**, because the model performs that addition on
        those tensors in that dtype. The decoder layers are walked directly rather
        than through :attr:`src.transfer.progen3.ProGen3.moe_blocks`, because this
        check needs each block's *parent* -- the layer that owns the residual and
        the normalisation -- and that property deliberately exposes only the block.
        """

        layers = list(self.pg.model.model.layers)
        residual: dict[int, torch.Tensor] = {}
        contribution: dict[int, torch.Tensor] = {}
        produced: dict[int, torch.Tensor] = {}
        handles = []

        def before_norm(layer: int) -> Callable[..., None]:
            def hook(module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
                residual[layer] = inputs[0].detach()

            return hook

        def after_layer(layer: int) -> Callable[..., None]:
            def hook(module: torch.nn.Module, inputs: Any, output: Any) -> None:
                produced[layer] = (
                    output[0] if isinstance(output, tuple) else output
                ).detach()

            return hook

        for layer, block in enumerate(layers):
            norm = getattr(block, self.residual_norm, None)
            if norm is None:
                raise TypeError(
                    f"{self.name}: decoder layer {layer} has no {self.residual_norm}, "
                    "so the residual the MoE block writes into cannot be read"
                )
            handles.append(norm.register_forward_pre_hook(before_norm(layer)))
            handles.append(block.register_forward_hook(after_layer(layer)))

        def tap(layer: int, x: torch.Tensor, y: torch.Tensor) -> None:
            contribution[layer] = y.detach()
            return None

        try:
            with self.block_intercept(tap):
                self.run(self.batch(self.render(SELF_CHECK_SEQUENCES[:2])))
        finally:
            for handle in handles:
                handle.remove()

        worst = 0.0
        for layer in range(len(layers)):
            rebuilt = residual[layer] + contribution[layer]
            worst = max(worst, float((rebuilt - produced[layer]).abs().max()))
        record = {
            "max_absolute_difference": worst,
            "n_layers": len(layers),
            "identity": self.identity,
            "block_layout": "serial",
            "verdict": "PASS" if worst == 0.0 else "FAIL",
        }
        if worst != 0.0:
            raise RuntimeError(
                f"{self.name}: the intercepted MoE output plus its input residual "
                f"differs from the block's own output by {worst:.3e}. The tensor "
                "this interceptor reads is not the residual write it is declared to "
                "be, so every dense arm measured against this reference would be "
                "compared with a different object."
            )
        return record

    def self_check(self) -> dict[str, Any]:
        """The estimand identity, then the scored band. Both refuse rather than report.

        The same order and the same contract as
        :meth:`DenseReplaceable.self_check`, which is the point of adding it: the
        reference arm and the arms compared against it are now gated the same way
        rather than by a band here and a band plus an identity there.
        """

        estimand = self.estimand_identity()
        record = dict(progen3_self_check(self.pg))
        record["estimand"] = estimand
        return record


# --------------------------------------------------------------- dense arms


#: What :func:`dense_self_check` scores on a **text** arm. Eight paragraphs of
#: ordinary English, frozen as literals for the reason ProGen3's own eight
#: sequences are: the check has to work on a host that carries no corpus, and a
#: check whose input can move is not a check. A protein arm is scored on
#: :data:`src.transfer.progen3.SELF_CHECK_SEQUENCES` -- literally the same eight
#: Swiss-Prot records ProGen3 is checked on -- so the modality pair is gated on
#: one text set and one protein set rather than on four.
SELF_CHECK_DOCUMENTS: tuple[str, ...] = (
    "The harbour opened onto a shallow bay, and by mid-morning the fishing boats "
    "had already returned. Their crews unloaded crates onto the quay, counted them "
    "twice, and left the tally with the harbour master. Nobody hurried. The same "
    "work had been done in the same order for as long as anyone could remember.",
    "A compiler translates a program written in one language into another, usually "
    "a form a processor can execute directly. The translation happens in stages: "
    "the text is broken into tokens, the tokens are arranged into a tree, the tree "
    "is checked for consistency, and only then is code emitted.",
    "Rainfall in the region is concentrated in two short seasons, so the reservoirs "
    "fill quickly and are drawn down slowly. Farmers plant according to the second "
    "of these seasons rather than the first, because the soil holds water better "
    "once the ground has already been wetted.",
    "The committee met on Thursday and agreed on three points. Funding would "
    "continue at its current level for another year; the report would be published "
    "in full rather than in summary; and a second review would be scheduled once "
    "the new figures were available.",
    "Iron rusts when it is exposed to both oxygen and water, and neither alone is "
    "enough. The reaction produces a flaky oxide that does not protect the metal "
    "underneath, which is why an untreated iron structure eventually corrodes "
    "through rather than forming a stable surface layer.",
    "She read the letter twice before answering it. The handwriting was familiar, "
    "the address was not, and the date at the top had been written and then "
    "corrected. Whatever had happened in the intervening months, the writer had not "
    "wanted to explain it on paper.",
    "Most of the cost of a long railway journey is not the fuel but the track: it "
    "has to be inspected, drained, cleared of vegetation and occasionally replaced, "
    "whether or not a train runs on it that day. Timetables are therefore built "
    "around keeping the line busy.",
    "The experiment was repeated with the temperature held constant, and the effect "
    "disappeared. That result was more informative than the original one, because "
    "it identified the variable the first design had left free and showed that the "
    "apparent difference had been a consequence of it.",
)

#: What :func:`dense_self_check` scores on an **EC-conditioned** arm: eight
#: ``(ec_number, sequence)`` pairs, frozen as literals for the same reason the
#: other two sets are.
#:
#: A third set, and it is not a preference. ZymCTRL's rendering carries an EC tag
#: and :data:`src.transfer.progen3.SELF_CHECK_SEQUENCES` carries none -- only two
#: of those eight records appear in the EC-labelled corpus at all, so six of them
#: could only be conditioned by inventing an enzyme class for a protein that has
#: none. That is a false fact in the gate that exists to catch false facts, and
#: it would put the model off its own training distribution while claiming to
#: check that it is on it (Appendix B rule 4).
#:
#: Drawn under a seeded permutation of the EC-labelled corpus rather than from
#: its head (Appendix B rule 1): seed 20260809 over the 6740 records of
#: ``data/zymctrl/ec_labeled_swissprot.fasta`` whose sequence length is 85-138
#: residues, which is the length range of the unconditioned protein set so that
#: the two gates score comparable amounts of evidence. Seven distinct EC classes,
#: and disjoint from both cohorts ``15_replacement_faithfulness.py`` draws at
#: :data:`src.transfer.arms.DEFAULT_CORPUS_DRAW_SEED`.
SELF_CHECK_EC_RECORDS: tuple[tuple[str, str], ...] = (
    (
        "1.3.7.7",
        "KRLLQNLGIEINQVIPEGGFIEDLQNLPKAWFNFVPYREIGLMTAVYLEKEFGMPYVSITPMGIVDTAE"
        "CIRQIQKHINELAVVSLEETVDYEPYIYQQTKFV",
    ),
    (
        "5.4.99.62",
        "MKKTGILNSHLAKLADDLGHTDRVCIGDLGLPVPNGIPKIDLSLTSGIPSFQEVLDIYLENILVEKVIL"
        "AEEIKEANPDQLSRLLAKLDNSVSIEYVSHNHLKQMTQDVKAVIRTGENTPYSNIILQSGVII",
    ),
    (
        "3.6.1.31",
        "MSDTLTRLAEVLEARKGAAPDSSYVASLYHKGLNKILEKVGEESVETILAAKDAAVSGDSSDLIYETAD"
        "LWFHSLVMLAALGQHPQAVLDELDRRFGLSGHAEKAARPQT",
    ),
    (
        "3.5.1.135",
        "MQPNDITFFQRFQNDILAGRKTITIRDASESHFKAGDVLRVGRFEDDGYFCTIEVTGTSTVTLDTLNEK"
        "HAQQENMSLDELKRVIAEIYPNQTQFYVIDFKCL",
    ),
    (
        "4.2.1.96",
        "MARNRLTESEMNEALRALDGWQKVDGREAITRSFKFKDFSTAFGFMAQAALYAEKLDHHPEWFNAYNRV"
        "DVTLATHSENGVTELDIKMARKMNAIAG",
    ),
    (
        "1.5.3.24",
        "MMLIECPNCGPRNENEFKYGGEAHVAYPEDPNALSDKEWSRYLFYRGNKKGIFAERWVHSGGCRKWFNA"
        "LRDTVSYEFKAVYRAGEARPQLDSTEGGTR",
    ),
    (
        "3.6.1.31",
        "MARFTLHDLAATVDARAASGGESSYTKKLLDKGPEHCAKKFGEEAVEMVIAAVENDRGHLISETADVLF"
        "HMLVLLKSRGVKLEEVEAALAQRTSMSGLEEKASRKRD",
    ),
    (
        "3.1.26.5",
        "MNTYAFNRELRLLTPEHYQNVFQQAHRAGSPHFTIIARNNKLSHPRLGLAVPKKQIKTAVGRNRFKRLA"
        "RESFRNNQHQLPNKDFVVIAKKSAQDLSNEELFKLFDKLWHRLSRPSRG",
    ),
)

#: The token cap the self-check tokenises under, declared here rather than taken
#: from the run's own ``--max-tokens``. All three frozen sets fit inside it, so the
#: check scores the same tokens whatever a campaign is configured to do; a check
#: that moved with the configuration would be measuring the configuration.
SELF_CHECK_MAX_TOKENS = 256

#: Mean per-token NLL of the frozen inputs above, under this module's own scoring
#: convention (bfloat16, batch 4, one L20). Measured, not quoted from a model
#: card: these are this repository's rendering, this repository's mask and these
#: eight records.
#:
#: **Per token, so these are not comparable across arms** and no reading should
#: try: ProtGPT2's multi-residue BPE puts several residues in a token, which is
#: exactly the unit mismatch L23 is about. Each value is a fixed point for its own
#: arm and nothing else.
#:
#: An arm absent from this table cannot be measured, because its loader gate would
#: have nothing to check -- see :data:`DENSE_ARMS_WITHOUT_A_BAND`.
#:
#: **Two measurement conditions, recorded rather than blended.** The four GPT-2
#: lineage values were measured at bfloat16, batch 4, on one L20. The four ProGen2
#: rungs were measured at bfloat16, batch 4, on **CPU**, because ``progen2-xlarge``
#: is 6.44B parameters and the local L20s are shared. The two conditions are
#: comparable to within the half-width by a long way: ``progen2-small`` reads
#: 2.3112 on CPU and 2.3284 on an L20, a device spread of 0.0172 against a
#: half-width of 0.30.
MEASURED_DENSE_SELF_CHECK_NLL: dict[str, float] = {
    "gpt2": 3.7651,
    "gpt2-large": 3.1706,
    "protgpt2": 5.2006,
    "zymctrl": 0.7292,
    # The protein-side scale ladder (src.transfer.arms.PROTEIN_SCALE_LADDER). The
    # upper two are STAGED_ARMS rather than panel members; a band is a property of
    # a checkpoint and its rendering, so it is measured the same way for both.
    "progen2-small": 2.3112,
    "progen2-medium": 1.8452,
    "progen2-large": 1.7849,
    "progen2-xlarge": 1.2678,
}

#: What the same measurement gives when an arm is broken in a way that does not
#: raise, **keyed by the arm it was measured on**. Recorded so that the band below
#: is sized against a measured distance rather than against taste, exactly as
#: ProGen3's is.
#:
#: Keyed by arm rather than flat, because a corruption of one arm says nothing
#: about another and reading the table as a cross-product is actively misleading
#: here: ZymCTRL rendered without its EC tag scores 3.1779 and gpt2-large scores
#: **3.1706** when it is perfectly healthy. A flat table invited the comparison,
#: and the two numbers are 0.0073 apart.
#:
#: * ``rendered_raw`` on ProtGPT2 -- the FASTA wrapping removed, which is L11's
#:   defect, the one that cost a retraction. Worth 1.42 nats/token on the 600-2000
#:   residue cohort it was priced on and **4.38** on these eight short records,
#:   where the missing end-of-text prefix is a larger share of the sequence.
#: * ``randomly_initialised`` -- the same architecture and tokenizer built from
#:   the arm's config with no weights read, which is L24's shape on a dense arm.
#:   ZymCTRL's 6.2108 is ln(458) to two decimals, its vocabulary being 458.
#: * ``rendered_without_its_ec_tag`` on ZymCTRL -- ``<start>{seq}<end>`` with the
#:   EC number and its separator dropped, scored on exactly the same 869 residue
#:   targets. This is L11's shape on a conditioned arm and it is the nearest
#:   silent failure this arm has, at **+2.4487 nats/token**: that figure is L15's
#:   conditioning leak measured on this gate's own inputs, against EXP-R2-034's
#:   1.73 on the cohort it was priced on.
#: * ``randomly_initialised`` on ``progen2-small`` -- the same architecture and
#:   tokenizer built from the arm's config with no weights read, measured on CPU
#:   beside that arm's own band. 3.6384 against 2.3112 is 1.33 nats, so L24's
#:   shape is caught on this lineage as well.
MEASURED_DENSE_SELF_CHECK_CORRUPTIONS: dict[str, dict[str, float]] = {
    "gpt2": {"randomly_initialised": 11.0789},
    "protgpt2": {"rendered_raw": 9.5802},
    "zymctrl": {
        "rendered_without_its_ec_tag": 3.1779,
        "randomly_initialised": 6.2108,
    },
    "progen2-small": {"randomly_initialised": 3.6384},
}

#: Silent failures this band **does not** separate, keyed by the arm they were
#: measured on. The other half of the table above, and it is data rather than a
#: caveat because a limitation nobody wrote down is one a later reader assumes
#: away.
#:
#: ``rendered_raw`` on a ProGen2 arm is the N-to-C control token ``"1"`` dropped
#: from the rendering -- L11's shape on this lineage, and the failure the band was
#: expected to catch. It costs **0.140** nats/token on ``progen2-small`` (2.4516
#: against 2.3112), 0.142 on ``progen2-medium``, 0.204 on ``progen2-large`` and
#: 0.112 on ``progen2-xlarge``: every one of them inside a half-width of 0.30, so
#: the band admits a ProGen2 arm rendered without its control token.
#:
#: That is a property of the lineage rather than of the band. ProtGPT2's FASTA
#: wrapping is worth 4.38 nats on these inputs because its BPE merges were learnt
#: over the wrapped byte stream; ProGen2's control token is one token of context
#: in front of an otherwise in-distribution N-to-C residue run, and the model is
#: barely disturbed by losing it. Narrowing the half-width would not repair it
#: either -- 0.112 on ``progen2-xlarge`` is inside any band wide enough to hold
#: the 0.0172 device spread with margin -- so the honest response is to record the
#: number, publish it beside the gate's verdict, and rely on the rendering being
#: reached through one implementation
#: (:meth:`src.transfer.arms.Cohort.input_strings`) rather than on the likelihood.
#:
#: The second entry is the same kind of fact about a *checkpoint* rather than a
#: rendering: ``progen2-medium`` and ``progen2-large`` are 0.060 nats apart on
#: these inputs, so the band cannot tell those two rungs of the ladder apart. The
#: declared depth and width check in :func:`src.transfer.arms.load_arm_spec`
#: (27L/1536d against 32L/2560d) and the weight digest in the artefact do.
UNSEPARATED_DENSE_SELF_CHECK_CORRUPTIONS: dict[str, dict[str, float]] = {
    "progen2-small": {"rendered_raw": 2.4516},
    "progen2-medium": {"rendered_raw": 1.9874, "another_rung_progen2_large": 1.7849},
    "progen2-large": {"rendered_raw": 1.9884, "another_rung_progen2_medium": 1.8452},
    "progen2-xlarge": {"rendered_raw": 1.3800},
}

#: Half-width of the band :func:`check_dense_nll` accepts, in nats/token.
#:
#: Sized from two measurements, like ProGen3's. The **spread of a correct arm**
#: across what an environment can change is at most 0.015 nats: over batch sizes
#: 1, 4 and 8 at bfloat16, gpt2 moves 3.7537/3.7651/3.7685, gpt2-large 0.0008,
#: ProtGPT2 0.0049 and ZymCTRL 0.0007, and float16 moves each by less than 0.011.
#: The **distance from each arm to its own nearest corruption that raises
#: nothing** is 4.38 nats on ProtGPT2 (rendered raw), 7.31 on gpt2 and 2.45 on
#: ZymCTRL (its EC tag dropped). A half-width of 0.30 is therefore at least 20x
#: the observed spread on every arm and still leaves two nats of clearance below
#: the nearest silent failure on the tightest of them.
#:
#: The lower end is not a numerical tolerance either. A value materially below the
#: measured one means the scored-target convention moved -- a mask that stopped
#: scoring the hard positions, say -- which corrupts everything downstream while
#: looking like an improvement. On ZymCTRL that end is load-bearing rather than
#: theoretical: a mask that let the conditioning prompt's near-deterministic
#: ``<sep>``/``<start>`` positions into the likelihood would pull the average
#: *down*, and 0.7292 is already low enough that there is little room beneath it.
DENSE_SELF_CHECK_HALF_WIDTH = 0.30

#: Panel arms this module could otherwise admit and does not, with the reason.
#: An unexplained absence and a decision must not be spelled the same way, which
#: is the discipline ``panel_contract.PANEL_MEMBERS_NOT_STAGED`` applies to the
#: campaign panel; :func:`_check_dense_arms` makes it an import-time failure here.
DENSE_ARMS_WITHOUT_A_BAND: dict[str, str] = {
    "gpt2-medium": (
        "no band measured. The ladder rungs are admissible in principle and each "
        "costs one short scoring run; only the arms a comparison needs -- the "
        "matched pair (gpt2-large, protgpt2), the tokenisation control (zymctrl) "
        "and the cheap smoke arm (gpt2) -- were measured"
    ),
    "gpt2-xl": "no band measured; as gpt2-medium",
    "dialogpt-small": "no band measured; as gpt2-medium",
    "progen2-base": (
        "no band measured. It is the corpus twin of progen2-medium rather than a "
        "rung of src.transfer.arms.PROTEIN_SCALE_LADDER, so no measurement here "
        "needed it; one short scoring run on the frozen inputs would admit it"
    ),
}


def _check_dense_arms() -> None:
    """A band names a real declared arm, and no arm is both measured and refused.

    The other half of this invariant -- that every *staged* arm carrying a
    residual-write declaration is in one table or the other -- is checked in
    ``tests/test_replaceable_arms.py``, because it needs
    ``panel_contract.CAMPAIGN_PANEL`` and a library module must not import a
    stage script to validate itself.

    The architecture condition is :data:`RESIDUAL_WRITE` and not
    :data:`DENSE_ARCHITECTURES`, and the two say different things. A band is a
    property of a *checkpoint and its rendering* -- it catches weights that did
    not load and a rendering the model was not trained behind -- and neither of
    those depends on whether a transcoder tap is defined on the block. The
    narrower claim is made where it belongs, in :func:`eligible_arms`.
    """

    for name, value in MEASURED_DENSE_SELF_CHECK_NLL.items():
        if name not in PANEL and name not in STAGED_ARMS:
            raise AssertionError(
                f"{name} has a self-check band but is neither a panel arm nor a "
                "declared staged checkpoint"
            )
        architecture = arm_spec(name).architecture
        if architecture not in RESIDUAL_WRITE:
            raise AssertionError(
                f"{name} has a self-check band but its {architecture!r} architecture "
                "declares no residual write, so the gate could not check the identity "
                "the band's own scoring convention rests on"
            )
        if not value > 0.0:
            raise AssertionError(f"{name} declares a non-positive self-check NLL")
    both = set(MEASURED_DENSE_SELF_CHECK_NLL) & set(DENSE_ARMS_WITHOUT_A_BAND)
    if both:
        raise AssertionError(f"{sorted(both)} are both measured and refused")
    for name, reason in DENSE_ARMS_WITHOUT_A_BAND.items():
        if name not in PANEL:
            raise AssertionError(f"{name} is refused but is not a panel arm")
        if not reason:
            raise AssertionError(f"{name} is refused without a reason")
    # A corruption filed under a name with no band is data nobody reads:
    # check_dense_nll resolves the arm's corruptions by key and would report an
    # empty set rather than raise, which is the silent shape this table exists
    # to make impossible. The same holds for the corruptions the band is recorded
    # as NOT separating, which reach the artefact through the same record.
    for table, label in (
        (MEASURED_DENSE_SELF_CHECK_CORRUPTIONS, "corruptions"),
        (UNSEPARATED_DENSE_SELF_CHECK_CORRUPTIONS, "unseparated corruptions"),
    ):
        orphans = sorted(set(table) - set(MEASURED_DENSE_SELF_CHECK_NLL))
        if orphans:
            raise AssertionError(
                f"{orphans} declare {label} but have no self-check band"
            )


_check_dense_arms()


def check_dense_nll(arm: str, value: float) -> dict[str, Any]:
    """Refuse a dense arm whose frozen-input NLL is outside its declared band.

    Separated from :meth:`DenseReplaceable.self_check` so that the band can be
    tested against the recorded corruption values without a checkpoint or a GPU.
    """

    if arm not in MEASURED_DENSE_SELF_CHECK_NLL:
        raise KeyError(
            f"no self-check band has been measured for {arm!r}, so its loader gate "
            "would degenerate to 'did it load' -- the question L24 exists to say is "
            f"not enough. Measured: {sorted(MEASURED_DENSE_SELF_CHECK_NLL)}"
        )
    reference = MEASURED_DENSE_SELF_CHECK_NLL[arm]
    low = reference - DENSE_SELF_CHECK_HALF_WIDTH
    high = reference + DENSE_SELF_CHECK_HALF_WIDTH
    inside = low <= value <= high
    # This arm's own corruptions, never the whole table: ZymCTRL rendered without
    # its EC tag reads 3.1779 and a healthy gpt2-large reads 3.1706, so a reader
    # handed both arms' corruptions can draw a comparison that means nothing.
    corruptions = dict(MEASURED_DENSE_SELF_CHECK_CORRUPTIONS.get(arm, {}))
    unseparated = dict(UNSEPARATED_DENSE_SELF_CHECK_CORRUPTIONS.get(arm, {}))
    record = {
        "nll": float(value),
        "band": [float(low), float(high)],
        "reference": float(reference),
        "corruptions": corruptions,
        # Published beside the verdict rather than only in this module's source,
        # so that an artefact carrying a PASS also carries what the PASS does not
        # rule out on this arm (see UNSEPARATED_DENSE_SELF_CHECK_CORRUPTIONS).
        "unseparated_corruptions": unseparated,
        "verdict": "PASS" if inside else "FAIL",
    }
    if not inside:
        raise RuntimeError(
            f"{arm} self-check NLL {value:.4f} nats/token is outside the declared "
            f"band [{low:.4f}, {high:.4f}]. Above the band the most likely cause is "
            "a wrong rendering or a checkpoint that did not load its weights, "
            f"neither of which raises; corruptions measured on this arm: "
            f"{corruptions}. Below it, the scored-target "
            "convention moved. Either way the model must not be measured."
        )
    return record


class DenseReplaceable(ReplaceableModel):
    """A dense panel arm whose per-layer MLP a replacement replaces.

    The dense counterpart of :class:`ProGen3Replaceable`, and the reason both
    exist: with one of these beside it, a replacement result on ProGen3 can be
    read against a text control (``gpt2-large``) and against a dense protein
    model of *identical* architecture, depth, width, vocabulary and parameter
    count (``protgpt2``) -- the matched modality pair of audit §2.

    :attr:`architectures` is this class's own admissible set, declared as a class
    attribute rather than read from a module constant inside ``__init__`` so that
    :class:`ParallelResidualReplaceable` can declare a different one without a
    second copy of anything else. This class is the *serial* layout, the one the
    replacement estimand is defined on.
    """

    #: The architectures this implementation's block layout covers.
    architectures = DENSE_ARCHITECTURES
    block_kind = DENSE_BLOCK_KIND
    loading_note = (
        "src.transfer.arms.load_arm_spec: AutoModelForCausalLM at a declared "
        "dtype, with the checkpoint's depth and width checked against the "
        "declaration and the loaded dtype read back from the parameters"
    )
    rendering_note = (
        "src.transfer.arms.Cohort.input_strings, the panel's one declaration of "
        "what string each arm is fed"
    )

    @property
    def scoring_note(self) -> str:
        """What this arm actually scored, for the artefact's condition block.

        A property rather than a constant because the two conditioned and
        unconditioned rules genuinely differ, and a stage that recorded one
        sentence for both would put a false statement about the scored span into
        the artefact of the arm whose scored span is the thing at issue.
        """

        if self.arm.spec.input_format == "ec_conditioned":
            return (
                "left to right, residue targets only: the EC conditioning prompt "
                "and the terminator are excluded "
                "(src.transfer.scoring.target_rule -> 'between_boundaries'), so "
                "the 1.73-nat conditioning leak (L15) is not scored as content"
            )
        return "left to right, every non-padding target after the first"

    def __init__(self, arm: Arm, *, max_tokens: int) -> None:
        if arm.spec.architecture not in self.architectures:
            raise TypeError(
                f"{arm.name}: {arm.spec.architecture!r} is not covered by "
                f"{type(self).__name__} ({sorted(self.architectures)}); its block "
                "does not carry the estimand this implementation is defined on"
            )
        if arm.name not in MEASURED_DENSE_SELF_CHECK_NLL:
            reason = DENSE_ARMS_WITHOUT_A_BAND.get(arm.name, "no band measured")
            raise ValueError(f"{arm.name} cannot be measured here: {reason}")
        if max_tokens < 1:
            raise ValueError("--max-tokens must be positive")
        self.arm = arm
        # Resolved in the constructor rather than at each use, so an arm whose
        # residual write nobody declared is refused before a cohort is drawn.
        self.residual_write = residual_write(arm.spec.architecture)
        self.max_tokens = int(max_tokens)
        self.name = arm.name

    # -- shape -------------------------------------------------------------

    @property
    def n_layers(self) -> int:
        return self.arm.n_layer

    @property
    def n_heads(self) -> int:
        heads = getattr(self.arm.model.config, "n_head", None)
        if heads is None:
            raise TypeError(
                f"{self.arm.name}: config declares no n_head, so the attention grid "
                "cannot be built"
            )
        return int(heads)

    @property
    def width(self) -> int:
        return int(self.arm.d_model)

    @property
    def device(self) -> torch.device:
        return torch.device(self.arm.device)

    @property
    def checkpoint(self) -> Path:
        return Path(self.arm.spec.path)

    def weights_files(self) -> list[Path]:
        """The checkpoint's weight files, preferring safetensors."""

        return checkpoint_weights_files(self.checkpoint)

    def weights_digest(self) -> str:
        """SHA-256 of the weight file, or of the shard digests when there are several."""

        return checkpoint_weights_digest(self.checkpoint)

    # -- inputs ------------------------------------------------------------

    def render(
        self, records: Sequence[str], *, ec_labels: Sequence[str | None] | None = None
    ) -> list[str]:
        """Through :meth:`src.transfer.arms.Cohort.input_strings`, never by hand.

        The two ways a conditioning prompt can be got wrong are both silent, so
        both raise. Rendering an ``ec_conditioned`` arm without labels would drop
        the prompt the model was trained behind -- the L11 failure, on the arm
        whose prompt is additionally worth 1.73 nats (L15). Handing labels to an
        arm that has no prompt would discard them while the caller believed a
        conditioned rendering had been produced.
        """

        labels = None if ec_labels is None else list(ec_labels)
        conditioned = self.arm.spec.input_format == "ec_conditioned"
        if conditioned:
            if labels is None or len(labels) != len(records) or any(
                label is None for label in labels
            ):
                raise ValueError(
                    f"{self.arm.name} renders an EC-conditioned prompt, so every "
                    "record must arrive with its EC label; rendering without one "
                    "would feed the model a format it was not trained on"
                )
        elif labels is not None and any(label is not None for label in labels):
            raise ValueError(
                f"{self.arm.name} has no conditioning prompt, so the EC labels "
                "handed to it would be silently dropped"
            )
        return Cohort(
            name=f"{self.arm.name}_render",
            kind=self.cohort_kind,
            records=list(records),
            min_symbols=0,
            max_symbols=0,
            metadata={"ec_labels": labels} if conditioned else {},
        ).input_strings(self.arm)

    def batch(
        self, inputs: Sequence[str], *, max_tokens: int | None = None
    ) -> dict[str, torch.Tensor]:
        ids, mask = tokenize_batch(
            self.arm, list(inputs), self.max_tokens if max_tokens is None else max_tokens
        )
        device = self.device
        return {"input_ids": ids.to(device), "attention_mask": mask.to(device)}

    def _target_mask(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Which next-token targets belong to this arm's content, over ``T - 1``.

        Resolved from :func:`src.transfer.scoring.target_rule` and
        :func:`src.transfer.scoring.sequence_target_mask` -- the repository's one
        declaration of what a conditioned rendering scores -- rather than by an
        arm name or a second copy of the rule (Appendix B rule 12). For every
        unconditioned arm the rule is ``all_valid`` and the result is
        ``attention_mask[:, 1:] & attention_mask[:, :-1]``, which under right
        padding is bit-identical to the ``attention_mask[..., 1:]`` this stage
        scored before, so no frozen number moves.
        """

        start, end = conditioning_boundary_ids(self.arm)
        return sequence_target_mask(
            batch["input_ids"],
            batch["attention_mask"],
            rule=target_rule(self.arm.spec.input_format),
            start_token_id=start,
            end_token_id=end,
        )

    def content_mask(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Positions carrying content: for a conditioned arm, the residues only.

        **Unconditioned arms: non-padding positions that are neither a tokenizer
        special token nor a marker the rendering added.** The pad token *is* the
        end-of-text token on every GPT-2-lineage arm, so the mask cannot be built
        from the ids alone -- padding and ProtGPT2's end-of-text prefix carry the
        same id, and one of them is a position the model was trained to predict
        from. The validity mask separates them, and the two id sets then remove the
        markers, which is what makes this the counterpart of ProGen3 excluding its
        terminus tokens.

        **Both id sets, because neither covers the other.** ProGen3's
        :data:`src.transfer.progen3.NON_RESIDUE_TOKENS` names ``"1"`` and ``"2"``
        alongside ``<pad>``/``<bos>``/``<eos>``; ProGen2 declares only the latter
        kind special and renders the same direction marker as an *ordinary*
        vocabulary entry, so ``all_special_ids`` alone kept position 0 of every
        ProGen2 record while the arm this measurement is compared against dropped
        it. :func:`src.transfer.arms.rendering_marker_ids` resolves the markers
        from the same declaration :meth:`src.transfer.arms.Cohort.input_strings`
        renders them from, so the mask and the rendering cannot come to disagree
        about which positions the arm's own format added.

        **A conditioned arm cannot use that rule**, and this is the reason ZymCTRL
        was refused here until now. Its ``<sep>``, ``<start>`` and ``<end>``
        markers are ordinary vocabulary entries rather than tokenizer special
        tokens, so the rule above would keep every one of them *and* the seven
        tokens of the EC number -- ten positions of a ~197-token cohort record.
        Those positions would then enter the per-layer mean that is the ablation
        endpoint, the reconstruction NMSE, and the transcoder's own objective,
        which would make the arm's replacement estimand a different object from
        the other arms'. The conditioned branch keeps exactly the positions
        strictly inside ``<start>`` and ``<end>``: the residues, which is what the
        other protein arms' content mask keeps.
        """

        ids = batch["input_ids"]
        if self.arm.spec.input_format == "ec_conditioned":
            # Target column q governs input position q + 1, so the residue
            # *positions* are the residue *targets* shifted by one. Derived from
            # the one target rule rather than restated, so the objective and the
            # likelihood cannot come to disagree about what content is.
            inside = torch.zeros_like(batch["attention_mask"], dtype=torch.bool)
            inside[:, 1:] = self._target_mask(batch)
            return inside
        mask = batch["attention_mask"].bool()
        excluded = sorted(
            {token for token in self.arm.tokenizer.all_special_ids if token is not None}
            | set(rendering_marker_ids(self.arm))
        )
        if excluded:
            markers = torch.tensor(excluded, device=ids.device)
            mask = mask & ~torch.isin(ids, markers)
        return mask

    # -- running -----------------------------------------------------------

    def run(self, batch: dict[str, torch.Tensor]) -> Any:
        return self.arm.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            return_dict=True,
        )

    def scored_logits(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Every non-padding target after the first, which is ProGen3's convention.

        ``targets`` is the shifted ids and ``mask`` the shifted validity mask, so
        a padded row contributes only the positions it actually holds. The
        special-token *markers* are scored, exactly as ProGen3 scores its
        terminus tokens: they are positions the model predicts, and excluding
        them from the likelihood while including them in the context would be a
        third scoring convention.

        **An ``ec_conditioned`` arm is the one exception, and it is not a third
        convention but the one the repository already declares.** Its EC number,
        ``<sep>`` and ``<start>`` are a *conditioning prompt* rather than content:
        the tag supplies 1.73 nats of label information (L15), so scoring the
        prompt would put that leak into the clean cross-entropy, into the
        fully-ablated endpoint and therefore into both ends of the recovery
        ratio. :func:`src.transfer.scoring.target_rule` selects
        ``between_boundaries`` for exactly that reason, and this method reads it
        rather than deciding again.
        """

        output = self.run(batch)
        ids = batch["input_ids"]
        return (
            output.logits[..., :-1, :].float(),
            ids[..., 1:],
            self._target_mask(batch),
        )

    def _blocks(self) -> list[torch.nn.Module]:
        return [self.arm.mlp(layer) for layer in range(self.n_layers)]

    @contextmanager
    def block_intercept(
        self, fn: Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor | None]
    ) -> Iterator[None]:
        """Read or replace every MLP's output while the model runs.

        The dense counterpart of :func:`src.transfer.progen3.moe_intercept`, and
        deliberately the same contract: interceptors compose in entry order, so
        the innermost ``with`` sees the output the outer one produced, which is
        what lets an ablation be applied on top of a replacement. A ``GPT2MLP``
        returns a bare tensor rather than ProGen3's ``(hidden, router)`` tuple,
        which is the whole of the difference.
        """

        handles = []
        for layer, block in enumerate(self._blocks()):

            def hook(
                module: torch.nn.Module,
                inputs: tuple[Any, ...],
                output: Any,
                layer: int = layer,
            ) -> Any:
                if isinstance(output, tuple):
                    raise TypeError(
                        f"{self.arm.name}: layer {layer}'s feed-forward returned a "
                        "tuple; this interceptor is written for a module whose "
                        "output IS the residual contribution"
                    )
                return fn(layer, inputs[0], output)

            handles.append(block.register_forward_hook(hook))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    @contextmanager
    def ablated(self, component: Component) -> Iterator[None]:
        """Zero one component's contribution to the residual stream.

        An attention head is zeroed at the input of the output projection, which
        is the only place its own slice still exists: ``GPT2Attention`` reshapes
        ``(bsz, len, heads, head_dim)`` to ``(bsz, len, heads * head_dim)`` with
        heads contiguous, so head ``h`` owns columns ``h * head_dim`` to
        ``(h + 1) * head_dim``. The projection itself is resolved through
        :func:`src.transfer.path_patching.attention_output_projection`, which
        reads the panel's one declaration of where each architecture keeps it.
        """

        if component.kind == self.block_kind:
            target = component.layer

            def zero(layer: int, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor | None:
                return torch.zeros_like(y) if layer == target else None

            with self.block_intercept(zero):
                yield
            return

        if component.kind != "attention_head":
            raise ValueError(
                f"no ablation is implemented for component kind {component.kind!r}"
            )

        head_dim = self.width // self.n_heads
        low = component.index * head_dim
        high = low + head_dim
        projection = attention_output_projection(self.arm, component.layer)

        def pre(module: torch.nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
            masked = inputs[0].clone()
            masked[..., low:high] = 0
            return (masked,) + tuple(inputs[1:])

        handle = projection.register_forward_pre_hook(pre)
        try:
            yield
        finally:
            handle.remove()

    # -- gates -------------------------------------------------------------

    @property
    def perturbation_target(self) -> dict[str, Any]:
        """The tensor a stage perturbs on this arm, and what it is not.

        Built from :attr:`residual_write` so that the sentence in the artefact and
        the equality :meth:`estimand_identity` tested are the same declaration.
        """

        write = self.residual_write
        return {
            "tensor": (
                "the per-layer feed-forward output -- the input of "
                "src.transfer.arms.Arm.mlp's own forward hook -- before anything "
                "is added to it"
            ),
            "block_layout": (
                "parallel (GPT-J style): the attention and the feed-forward read "
                f"the same {write.residual_norm} and both sum into the residual, "
                "so there is no sequential block output of the kind a serial "
                "layout has"
                if write.parallel_attention
                else (
                    "serial: the attention has already been added to the residual "
                    f"by the time {write.residual_norm} runs, so the feed-forward "
                    "output IS the whole of this block's residual write"
                )
            ),
            "identity_verified": write.identity,
            "not_perturbed": (
                "the attention contribution, which on this layout is a SECOND "
                "residual write made from the same normalisation. It is left "
                "untouched, exactly as the attention term is left untouched on a "
                "serial arm, so the manipulation is the same object on both"
                if write.parallel_attention
                else (
                    "the attention contribution, which this layout has already "
                    "added to the residual before the feed-forward runs"
                )
            ),
        }

    @torch.no_grad()
    def estimand_identity(self) -> dict[str, Any]:
        """Verify that the intercepted tensor IS this block's feed-forward write.

        What rests on it differs by stage and both readings need the same check.
        A transcoder trained here reads what :meth:`block_intercept` calls the
        block input and predicts what it calls the block output, and the
        cross-model comparison rests on that being the object ProGen3's transcoder
        is trained on; a perturbation reads only the output half, and rests on it
        being the term the residual stream receives. Either way the block's own
        output must be reconstructible from the intercepted tensor and the other
        declared terms.

        **The reconstruction is the declared one**, resolved from
        :attr:`residual_write`. On a serial block that is ``residual + ff``. On a
        parallel GPT-J-style block the model computes
        ``attn_out + ff_out + residual`` as a single left-associated sum, so the
        check rebuilds ``(attn_out + ff_out) + residual`` in that order and reads
        the attention term from :meth:`src.transfer.arms.Arm.attention`, the
        panel's own declaration of where the attention lives.

        The tolerance is **exact**, because the model performs those additions on
        those tensors in that dtype: anything but equality means the block does
        something this interceptor does not see. The serial reconstruction applied
        to a ProGen2 block misses by 14.25 at bfloat16, which is what makes this
        an identity check rather than a formality.
        """

        write = self.residual_write
        blocks = list(self.arm.blocks())
        residual: dict[int, torch.Tensor] = {}
        attention: dict[int, torch.Tensor] = {}
        contribution: dict[int, torch.Tensor] = {}
        produced: dict[int, torch.Tensor] = {}
        handles = []

        def before_norm(layer: int) -> Callable[..., None]:
            def hook(module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
                residual[layer] = inputs[0].detach()

            return hook

        def after_module(store: dict[int, torch.Tensor], layer: int) -> Callable[..., None]:
            def hook(module: torch.nn.Module, inputs: Any, output: Any) -> None:
                store[layer] = (output[0] if isinstance(output, tuple) else output).detach()

            return hook

        for layer, block in enumerate(blocks):
            norm = getattr(block, write.residual_norm, None)
            if norm is None:
                raise TypeError(
                    f"{self.arm.name}: block {layer} has no {write.residual_norm}, so "
                    "the residual the feed-forward writes into cannot be read"
                )
            handles.append(norm.register_forward_pre_hook(before_norm(layer)))
            handles.append(block.register_forward_hook(after_module(produced, layer)))
            if write.parallel_attention:
                handles.append(
                    self.arm.attention(layer).register_forward_hook(
                        after_module(attention, layer)
                    )
                )

        def tap(layer: int, x: torch.Tensor, y: torch.Tensor) -> None:
            contribution[layer] = y.detach()
            return None

        records, labels = self._self_check_records()
        try:
            with self.block_intercept(tap):
                self.run(
                    self.batch(
                        self.render(
                            records[:2],
                            ec_labels=None if labels is None else labels[:2],
                        )
                    )
                )
        finally:
            for handle in handles:
                handle.remove()

        worst = 0.0
        for layer in range(self.n_layers):
            if write.parallel_attention:
                rebuilt = (attention[layer] + contribution[layer]) + residual[layer]
            else:
                rebuilt = residual[layer] + contribution[layer]
            worst = max(worst, float((rebuilt - produced[layer]).abs().max()))
        record = {
            "max_absolute_difference": worst,
            "n_layers": self.n_layers,
            "identity": write.identity,
            "block_layout": "parallel" if write.parallel_attention else "serial",
            "verdict": "PASS" if worst == 0.0 else "FAIL",
        }
        if worst != 0.0:
            raise RuntimeError(
                f"{self.arm.name}: the declared reconstruction -- {write.identity} -- "
                f"differs from the block's own output by {worst:.3e}. The tensor this "
                "interceptor reads is not the residual write it is declared to be, so "
                "a replacement or a perturbation measured here would not be the "
                "measurement the other arms' is."
            )
        return record

    def _self_check_records(self) -> tuple[tuple[str, ...], tuple[str, ...] | None]:
        """The frozen inputs this arm's gate scores, and their labels if it has any."""

        if self.arm.spec.input_format == "ec_conditioned":
            return (
                tuple(sequence for _, sequence in SELF_CHECK_EC_RECORDS),
                tuple(ec for ec, _ in SELF_CHECK_EC_RECORDS),
            )
        return (
            SELF_CHECK_DOCUMENTS if self.cohort_kind == "text" else SELF_CHECK_SEQUENCES,
            None,
        )

    @torch.no_grad()
    def self_check(self) -> dict[str, Any]:
        """The estimand identity, then the scored band. Both refuse rather than report."""

        estimand = self.estimand_identity()
        record = check_dense_nll(self.arm.name, dense_self_check_nll(self))
        record["estimand"] = estimand
        record["n_records"] = len(self._self_check_records()[0])
        record["scored_under"] = (
            f"{self.arm.spec.input_format} rendering, {SELF_CHECK_MAX_TOKENS}-token cap"
        )
        return record


@torch.no_grad()
def dense_self_check_nll(model: DenseReplaceable, *, batch_size: int = 4) -> float:
    """Mean per-token NLL of the frozen inputs, in this module's own convention.

    Exposed so that a band can be *measured* by the same code path that later
    checks it. A band measured through a second implementation would be a band on
    that implementation.
    """

    records, labels = model._self_check_records()
    inputs = model.render(records, ec_labels=labels)
    total = 0.0
    count = 0
    for start in range(0, len(inputs), batch_size):
        chunk = inputs[start : start + batch_size]
        batch = model.batch(chunk, max_tokens=SELF_CHECK_MAX_TOKENS)
        logits, targets, mask = model.scored_logits(batch)
        nll = token_nll(logits, targets)
        total += float((nll * mask).sum())
        count += int(mask.sum())
    if count == 0:
        raise RuntimeError(f"{model.arm.name}: the frozen self-check inputs scored no tokens")
    return total / count


class ParallelResidualReplaceable(DenseReplaceable):
    """An arm whose attention and feed-forward write the residual in one sum.

    ProGen2, and the reason it needs its own admissible set rather than an entry
    in :data:`DENSE_ARCHITECTURES`. Its block is GPT-J-style::

        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states, ...)[0]
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_output + feed_forward_hidden_states + residual

    so the feed-forward reads a *pre*-attention normalisation. A transcoder
    trained at that tap predicts a different object from the one ProGen3's
    predicts, which is why :data:`DENSE_ARCHITECTURES` -- the replacement
    estimand's admissible set -- excludes it and still does.

    **What this class admits is strictly weaker and is enough for a
    perturbation.** ``23_perturbation_sensitivity.py`` never reads the block
    *input*: it perturbs the feed-forward *output*, anchored on that tensor's own
    norm. That tensor is the same object on both layouts -- the MLP's output
    before anything is added to it -- and the only thing that changes is the
    identity which certifies the interception, which
    :meth:`DenseReplaceable.estimand_identity` resolves from
    :data:`RESIDUAL_WRITE` and verifies exactly on the live forward pass.

    Everything else is inherited unchanged, deliberately: the rendering, the
    scored-target rule, the content mask, the splice and the loader band are
    properties of a panel-style checkpoint and not of its residual layout, so a
    second copy of any of them would be a second measurement.
    """

    architectures = PARALLEL_ARCHITECTURES


# ------------------------------------------------------- joint checkpoints


#: The two modes one joint language-protein checkpoint is measured in. A mode is
#: not an arm: it selects the rendering, the corpus and the scored span, and the
#: weights are the same object in both -- which is the whole reason a joint
#: checkpoint controls architecture, scale AND weights where two standalone
#: models control only the first.
JOINT_MODES = ("text", "protein")

#: The corpus each joint mode is read from, in the vocabulary
#: :data:`src.transfer.arms.CORPUS_SOURCES` uses.
#:
#: One declaration because three stages need it and they must agree. A
#: transcoder trained on one population and scored on another is the train/eval
#: gap EXP-R2-135 priced at 4.1x in NLL recovery, and it is exactly what
#: :func:`arm_training_corpus` exists to prevent on a panel arm; a joint
#: checkpoint has no panel declaration to read it from, so it is declared here
#: and read by ``15_replacement_faithfulness.py``, ``17_train_transcoder.py``
#: and ``23_perturbation_sensitivity.py`` alike.
#:
#: Swiss-Prot rather than UniRef50 for the protein mode, and that is a choice
#: rather than an inheritance: ProGen3's transcoders train on UniRef50 because
#: its published runs did, while every joint measurement this programme owns --
#: ``21_joint_mode_qualification.py``'s context information above all -- is
#: taken on Swiss-Prot, so training elsewhere would put a population gap between
#: the dictionary and every number it is read beside.
JOINT_MODE_CORPUS: dict[str, str] = {"text": "openwebtext", "protein": "swissprot"}


def joint_mode_corpus(mode: str) -> str:
    """The corpus one joint mode is both trained on and scored on."""

    if mode not in JOINT_MODE_CORPUS:
        raise ValueError(
            f"unknown joint mode {mode!r}; declared: {sorted(JOINT_MODE_CORPUS)}"
        )
    return JOINT_MODE_CORPUS[mode]


def joint_tokenisation(
    tokenizer: Any, declaration: joint_modes.JointRendering, mode: str
) -> joint_modes.JointTokenisation | None:
    """The declared rendering resolved against this tokenizer, when the mode needs it.

    The refusal point for a checkpoint/family/mode triple, and it runs on the
    tokenizer alone so that a wrong pairing fails before a multi-gigabyte load.
    Protein mode resolves -- which is what refuses a tokenizer that cannot carry
    the declared residue alphabet, or that merges residues where the family
    declares one token per residue. Text mode does not: its scored positions are
    the tokenizer's own next-token targets and do not depend on the protein
    format, so resolving would refuse a checkpoint on a property the run never
    reads.

    Declared once because the two stages that build a :class:`JointReplaceable`
    per mode would otherwise each decide it, and the failure of deciding it
    wrongly is silent in one direction: a text run that resolved would refuse a
    measurable mode, and a protein run that did not would be refused by
    :class:`JointReplaceable` itself -- but only after the weights were read.
    """

    if mode not in JOINT_MODES:
        raise ValueError(f"unknown joint mode {mode!r}; declared: {JOINT_MODES}")
    return joint_modes.resolve(tokenizer, declaration) if mode == "protein" else None


@dataclass(frozen=True)
class JointBlockLayout:
    """Where one joint architecture keeps the block a perturbation is spliced into.

    ``layer_list`` is the attribute path from the causal-LM wrapper to the list of
    decoder layers, ``feed_forward`` the attribute on one layer whose output IS
    the term added to the residual stream, and ``pre_feed_forward_norm`` the
    normalisation whose *input* is the residual it is added to. The last is what
    :meth:`JointReplaceable.estimand_identity` reconstructs the block output
    from, so the layout is verified against the live forward pass rather than
    believed.
    """

    layer_list: tuple[str, ...]
    feed_forward: str
    pre_feed_forward_norm: str


#: Joint architectures whose block carries the replacement estimand unchanged,
#: keyed by ``config.model_type``.
#:
#: ``llama`` only, and for the reason :data:`DENSE_ARCHITECTURES` is a frozenset
#: of one. The ProLLaMA lineage (``Llama-2-7b-hf``, ``ProLLaMA_Stage_1``,
#: ``ProLLaMA``) is LLaMA-2, whose decoder layer is the serial
#: ``residual + mlp(post_attention_layernorm(residual))`` this estimand is defined
#: on. Galactica is ``opt``, which splits its feed-forward into ``fc1``/``fc2``
#: with no single module whose output is the residual contribution, so it has no
#: block for this declaration to name; adding it would need a different
#: interception point and a separate verification.
JOINT_ARCHITECTURES: dict[str, JointBlockLayout] = {
    "llama": JointBlockLayout(
        layer_list=("model", "layers"),
        feed_forward="mlp",
        pre_feed_forward_norm="post_attention_layernorm",
    ),
}


def joint_block_layout(model_type: str) -> JointBlockLayout:
    """The declared layout for one joint architecture, refusing an undeclared one."""

    if model_type not in JOINT_ARCHITECTURES:
        raise TypeError(
            f"{model_type!r} is not a declared joint architecture "
            f"({sorted(JOINT_ARCHITECTURES)}); its block has not been verified to be "
            "the serial post-attention feed-forward this estimand is defined on, and "
            "duck-typing an attribute called 'mlp' would measure a different object "
            "without saying so"
        )
    return JOINT_ARCHITECTURES[model_type]


class JointReplaceable(ReplaceableModel):
    """One mode of one joint language-protein checkpoint, behind the shared interface.

    The third implementation, and the one the joint axis needs: a
    :class:`DenseReplaceable` is a *panel* arm with a measured likelihood band,
    and a joint checkpoint is deliberately not in the panel -- a checkpoint that
    has not passed ``21_joint_mode_qualification.py`` must not be in ``arms.py``
    at all. So the joint checkpoint is reached by path, its rendering is named,
    and one instance is built per mode over **the same loaded weights**.

    Everything about a mode that could be decided twice is read from
    :mod:`src.transfer.joint_modes` instead: the protein rendering, its scored
    span, and the refusal of a tokenizer that cannot carry the declared alphabet.
    Nothing here re-derives a format.

    **What one instance's scored positions are.** In protein mode they are the
    rendering's own scored span -- the token run whose targets carry residues --
    which is exactly what ``21_joint_mode_qualification.py`` scores, so the two
    stages describe the same symbols. In text mode they are every non-padding
    target after the first, which is what every dense arm scores.

    **What its content positions are**, the positions a per-layer mean is taken
    over: in protein mode the scored span, so delimiters, the instruction prefix
    and the beginning-of-sequence token stay out of the fully-ablated endpoint;
    in text mode every non-padding, non-special position, which is
    :meth:`DenseReplaceable.content_mask`'s unconditioned rule on a rendering that
    prefixes no marker of its own -- this mode's text records are the corpus
    strings, so a panel arm's :func:`src.transfer.arms.rendering_marker_ids` term
    would be empty here and there is nothing for it to resolve against.
    """

    block_kind = DENSE_BLOCK_KIND
    loading_note = (
        "AutoModelForCausalLM at a declared dtype, read back from the loaded "
        "parameters; the block layout is resolved from a declaration keyed by "
        "config.model_type and verified against the live forward pass"
    )

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        checkpoint: Path,
        declaration: joint_modes.JointRendering,
        mode: str,
        tokenisation: joint_modes.JointTokenisation | None = None,
        max_tokens: int = 512,
        protein_context: str | None = None,
    ) -> None:
        if mode not in JOINT_MODES:
            raise ValueError(f"unknown joint mode {mode!r}; declared: {JOINT_MODES}")
        if mode == "protein" and tokenisation is None:
            raise ValueError(
                f"{declaration.name}: protein mode needs the rendering resolved against "
                "this checkpoint's tokenizer (src.transfer.joint_modes.resolve), which "
                "is what refuses a tokenizer that does not carry the declared residue "
                "alphabet. Scoring protein without it would score whatever the "
                "tokenizer happened to produce"
            )
        if max_tokens < 2:
            raise ValueError("--max-tokens must leave at least one scored target")
        self.layout = joint_block_layout(str(getattr(model.config, "model_type", "")))
        self.model = model
        self.tokenizer = tokenizer
        self._checkpoint = Path(checkpoint)
        self.declaration = declaration
        self.mode = mode
        self.tokenisation = tokenisation
        self.max_tokens = int(max_tokens)
        self.protein_context = protein_context
        self.name = f"{declaration.name}:{mode}"
        self._rendered: dict[str, joint_modes.RenderedProtein] = {}
        self._pad_id, self._pad_source = self._padding_id()

    # -- declarations ------------------------------------------------------

    @property
    def cohort_kind(self) -> str:
        """The mode decides the corpus, not a panel declaration."""

        return self.mode

    @property
    def scoring_note(self) -> str:
        if self.mode == "protein":
            return (
                "left to right, the declared rendering's scored span only "
                f"({self.declaration.scored_target_rule}): the delimiters, the "
                "optional document context and the beginning-of-sequence token are "
                "excluded, so the likelihood is over the same symbols "
                "21_joint_mode_qualification.py scores"
            )
        return "left to right, every non-padding target after the first"

    @property
    def rendering_note(self) -> str:
        if self.mode == "protein":
            return (
                "src.transfer.joint_modes.JointTokenisation.render, the declared "
                f"{self.declaration.name} protein format, which locates and "
                "verifies the scored span in the same call that produces the "
                "string; NOT src.transfer.arms.Cohort.input_strings, which is "
                "the panel's declaration and does not describe this checkpoint"
            )
        return (
            "the corpus record itself, tokenised as this checkpoint's own text; "
            "the declared protein format does not apply to a text record and no "
            "wrapper is added to one"
        )

    def _padding_id(self) -> tuple[int, str]:
        """A right-padding id, named rather than assumed.

        The value cannot change a number -- padding is right of the content, the
        attention is causal and no scored target lands on it -- but a tokenizer
        that declares none at all would pad with ``None``, so the fallback order
        is declared and the source is recorded rather than guessed at silently.
        """

        for attribute in ("pad_token_id", "eos_token_id", "unk_token_id"):
            value = getattr(self.tokenizer, attribute, None)
            if value is not None:
                return int(value), attribute
        raise ValueError(
            "this tokenizer declares no pad, end-of-sequence or unknown token id, so "
            "a batch of unequal lengths cannot be padded"
        )

    # -- shape -------------------------------------------------------------

    def _layers(self) -> list[torch.nn.Module]:
        node: Any = self.model
        for attribute in self.layout.layer_list:
            node = getattr(node, attribute, None)
            if node is None:
                raise TypeError(
                    f"{self.name}: this checkpoint has no "
                    f"{'.'.join(self.layout.layer_list)}, so the declared block layout "
                    "does not describe it"
                )
        return list(node)

    @property
    def n_layers(self) -> int:
        return len(self._layers())

    @property
    def n_heads(self) -> int:
        heads = getattr(self.model.config, "num_attention_heads", None)
        if heads is None:
            raise TypeError(
                f"{self.name}: config declares no num_attention_heads, so the "
                "attention grid cannot be built"
            )
        return int(heads)

    @property
    def width(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def checkpoint(self) -> Path:
        return self._checkpoint

    def weights_digest(self) -> str:
        return checkpoint_weights_digest(self._checkpoint)

    # -- inputs ------------------------------------------------------------

    def render(
        self, records: Sequence[str], *, ec_labels: Sequence[str | None] | None = None
    ) -> list[str]:
        """Records as this mode's own input strings, through the declared rendering.

        A protein record goes through :meth:`joint_modes.JointTokenisation.render`,
        which raises rather than warns when the declared rendering did not reach
        the symbol unit its family declares. The resulting record -- its ids and
        its scored span -- is kept, because re-deriving the span from the rendered
        string at batch time would be a second copy of the rule that located it.
        """

        if ec_labels is not None and any(label is not None for label in ec_labels):
            raise ValueError(
                f"{self.name} has no EC conditioning prompt; labels handed to it would "
                "be silently dropped and the run would look like a conditioned one"
            )
        if self.mode == "text":
            return list(records)
        assert self.tokenisation is not None  # enforced in __init__
        rendered: list[str] = []
        for sequence in records:
            record = self.tokenisation.render(sequence, context=self.protein_context)
            if len(record.token_ids) > self.max_tokens:
                raise ValueError(
                    f"a rendered protein needs {len(record.token_ids)} tokens and "
                    f"max_tokens is {self.max_tokens}; truncating it would drop the "
                    "closing delimiter and silently change the scored span. Raise "
                    "--max-tokens or lower --protein-max-len"
                )
            self._rendered[record.text] = record
            rendered.append(record.text)
        return rendered

    def _rows(self, inputs: Sequence[str], cap: int) -> list[tuple[list[int], set[int]]]:
        """Each input's ids and the positions whose *target* this mode scores."""

        rows: list[tuple[list[int], set[int]]] = []
        for text in inputs:
            if self.mode == "protein":
                record = self._rendered.get(text)
                if record is None:
                    raise KeyError(
                        f"{self.name}: this string did not come from render(), so its "
                        "scored span is unknown. A protein input must be rendered "
                        "through the declared rendering, which is where the span is "
                        "located and verified"
                    )
                rows.append((list(record.token_ids), set(record.scored_positions)))
                continue
            ids = joint_modes.encode(self.tokenizer, text)[:cap]
            if len(ids) < 2:
                raise ValueError(
                    f"{self.name}: a text record tokenised to {len(ids)} tokens; "
                    "nothing predicts the first token of a sequence, so there is no "
                    "scored target"
                )
            rows.append((ids, set(range(1, len(ids)))))
        return rows

    def batch(
        self, inputs: Sequence[str], *, max_tokens: int | None = None
    ) -> dict[str, torch.Tensor]:
        """Right-padded ids with this mode's content and target masks alongside.

        Both masks travel with the batch rather than being recomputed from the
        ids, because in protein mode neither is derivable from the ids alone: the
        scored span is located by the rendering's own rule, and a residue token is
        an ordinary piece of the text vocabulary on this family.
        """

        rows = self._rows(list(inputs), self.max_tokens if max_tokens is None else max_tokens)
        if not rows:
            raise ValueError("an empty batch has nothing to score")
        width = max(len(ids) for ids, _ in rows)
        input_ids = torch.full((len(rows), width), self._pad_id, dtype=torch.long)
        attention = torch.zeros((len(rows), width), dtype=torch.long)
        content = torch.zeros((len(rows), width), dtype=torch.bool)
        target = torch.zeros((len(rows), width - 1), dtype=torch.bool)
        for row, (ids, scored) in enumerate(rows):
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention[row, : len(ids)] = 1
            for position in sorted(scored):
                target[row, position - 1] = True
            if self.mode == "protein":
                content[row, sorted(scored)] = True
            else:
                content[row, : len(ids)] = True
                special = [
                    token
                    for token in getattr(self.tokenizer, "all_special_ids", []) or []
                    if token is not None
                ]
                if special:
                    marker = torch.tensor(special, dtype=torch.long)
                    content[row] &= ~torch.isin(input_ids[row], marker)
        device = self.device
        return {
            "input_ids": input_ids.to(device),
            "attention_mask": attention.to(device),
            "content_mask": content.to(device),
            "target_mask": target.to(device),
        }

    def content_mask(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return batch["content_mask"]

    def forget_rendered(self) -> None:
        """Drop the rendering records, once the batch that needed them exists.

        Unbounded otherwise, and the bound matters: this is the one
        implementation that keeps per-record state, and a transcoder trainer
        renders a fresh batch every step. At the joint campaign's own budget --
        of order 3e5 protein records, each carrying its rendered text, its token
        ids and its scored positions -- the dictionary reaches gigabytes over a
        run, for records no later step reads. A caller that renders once and
        batches many times (every scoring stage) simply never calls this.
        """

        self._rendered.clear()

    # -- running -----------------------------------------------------------

    def run(self, batch: dict[str, torch.Tensor]) -> Any:
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            return_dict=True,
        )

    def scored_logits(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.run(batch)
        ids = batch["input_ids"]
        return output.logits[..., :-1, :].float(), ids[..., 1:], batch["target_mask"]

    @contextmanager
    def block_intercept(
        self, fn: Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor | None]
    ) -> Iterator[None]:
        """Read or replace every feed-forward's output while the model runs.

        The same contract :meth:`DenseReplaceable.block_intercept` declares --
        ``fn(layer, block_input, block_output)`` returns a substitute or ``None``
        -- so a stage carries one splice and not a per-architecture branch.
        """

        handles = []
        for layer, block in enumerate(self._layers()):
            module = getattr(block, self.layout.feed_forward, None)
            if module is None:
                raise TypeError(
                    f"{self.name}: layer {layer} has no {self.layout.feed_forward!r}, "
                    "so the declared block layout does not describe it"
                )

            def hook(
                module: torch.nn.Module,
                inputs: tuple[Any, ...],
                output: Any,
                layer: int = layer,
            ) -> Any:
                if isinstance(output, tuple):
                    raise TypeError(
                        f"{self.name}: layer {layer}'s feed-forward returned a tuple; "
                        "this interceptor is written for a module whose output IS the "
                        "residual contribution"
                    )
                return fn(layer, inputs[0], output)

            handles.append(module.register_forward_hook(hook))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    @property
    def perturbation_target(self) -> dict[str, Any]:
        return {
            "tensor": (
                f"the per-layer {self.layout.feed_forward} output, before the "
                "residual add"
            ),
            "block_layout": (
                "serial: the attention has already been added to the residual by "
                f"the time {self.layout.pre_feed_forward_norm} runs, so the "
                "feed-forward output IS this block's residual write"
            ),
            "identity_verified": (
                f"block output == {self.layout.pre_feed_forward_norm} input + "
                f"intercepted {self.layout.feed_forward} output"
            ),
            "not_perturbed": (
                "the attention contribution, which this layout has already added "
                "to the residual before the feed-forward runs"
            ),
        }

    def component_families(self) -> tuple[str, ...]:
        """The block family alone: this checkpoint has no ablatable attention head.

        An attention head is zeroed at the input of the output projection, which
        :func:`src.transfer.path_patching.attention_output_projection` resolves
        from the *panel's* per-architecture declaration -- and a joint checkpoint is
        reached by path precisely because it is not a panel arm (a checkpoint that
        has not passed ``21_joint_mode_qualification.py`` must not be in
        ``arms.py`` at all), so there is no declaration to resolve it against. The
        family is therefore absent from the grid rather than located by searching
        for a plausible module.

        **What this costs a reader is a scoping rule, not a caveat**, and it is
        declared where a reader meets it: ``15_replacement_faithfulness.py``
        writes it into every artefact's ``component_family_comparability`` from
        the families the grid actually carries. In short, a comparison against a
        dense arm is refused and the within-checkpoint two-mode comparison is not,
        because both modes carry this identical block-only grid. It is stated
        there rather than here so that the artefact a result is read from carries
        it.
        """

        return (self.block_kind,)

    @contextmanager
    def ablated(self, component: Component) -> Iterator[None]:
        """Zero one block's contribution to the residual stream.

        Refuses anything outside :meth:`component_families`, which is the same
        declaration :meth:`components` builds the grid from, so this branch guards
        a hand-built component rather than the grid a stage sweeps.
        """

        if component.kind not in self.component_families():
            raise ValueError(
                f"{self.name}: no ablation is implemented for component kind "
                f"{component.kind!r}; this checkpoint declares "
                f"{list(self.component_families())}. The attention output projection "
                "is declared per panel architecture and this checkpoint is not a "
                "panel arm"
            )
        target = component.layer

        def zero(layer: int, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor | None:
            return torch.zeros_like(y) if layer == target else None

        with self.block_intercept(zero):
            yield

    # -- measurement -------------------------------------------------------

    def symbols_per_token(self, inputs: Sequence[str]) -> float:
        """Measured expansion over exactly the window this mode scores.

        Protein counts residues per scored token, read off the rendering records
        rather than recounted, which is the quantity
        ``21_joint_mode_qualification.py`` reports beside every protein magnitude
        of a token-unit family. Text counts characters per token over the
        truncated window, which is
        :func:`src.transfer.arms.symbols_per_token`'s text convention.
        """

        tokens = 0
        symbols = 0
        for text in inputs:
            if self.mode == "protein":
                record = self._rendered.get(text)
                if record is None:
                    raise KeyError(
                        f"{self.name}: this string did not come from render(), so its "
                        "scored span is unknown"
                    )
                tokens += record.n_scored_tokens
                symbols += record.n_residues
                continue
            ids = joint_modes.encode(self.tokenizer, text)[: self.max_tokens]
            tokens += len(ids)
            symbols += len(self.tokenizer.decode(ids))
        if tokens == 0:
            raise RuntimeError(f"{self.name}: the scored inputs carry no tokens")
        return symbols / tokens

    # -- gates -------------------------------------------------------------

    def _self_check_records(self) -> tuple[str, ...]:
        """The frozen inputs the structural gate runs on.

        Literally the sets the dense arms are checked on -- eight paragraphs of
        English and eight Swiss-Prot records -- so that the joint checkpoint's
        gate reads the same inputs the standalone pair's does and no third frozen
        set enters the repository.
        """

        return SELF_CHECK_DOCUMENTS if self.mode == "text" else SELF_CHECK_SEQUENCES

    @torch.no_grad()
    def estimand_identity(self) -> dict[str, Any]:
        """Verify that the intercepted **output** IS this block's residual write.

        :meth:`DenseReplaceable.estimand_identity`'s check, on the declared layout:
        the block's own output must equal the residual its pre-feed-forward
        normalisation read plus the intercepted feed-forward output. The tolerance
        is **exact**, because the model performs that addition on those two
        tensors in that dtype, so anything but equality means the block does
        something this interceptor does not see.

        The output half is what this identity pins, and the wording matters
        because :meth:`block_intercept` hands a stage a *pair*. The input half is
        not left unchecked, it is checked differently: it is
        ``self.layout.feed_forward``'s own first positional argument on every
        layer of this pass, so a layout that named the wrong module would raise
        here rather than pass, and an architecture nobody declared is refused by
        :func:`joint_block_layout` before a forward runs.
        """

        blocks = self._layers()
        residual: dict[int, torch.Tensor] = {}
        contribution: dict[int, torch.Tensor] = {}
        produced: dict[int, torch.Tensor] = {}
        handles = []

        def before_norm(layer: int) -> Callable[..., None]:
            def hook(module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
                residual[layer] = inputs[0].detach()

            return hook

        def after_block(layer: int) -> Callable[..., None]:
            def hook(module: torch.nn.Module, inputs: Any, output: Any) -> None:
                produced[layer] = (
                    output[0] if isinstance(output, tuple) else output
                ).detach()

            return hook

        for layer, block in enumerate(blocks):
            norm = getattr(block, self.layout.pre_feed_forward_norm, None)
            if norm is None:
                raise TypeError(
                    f"{self.name}: block {layer} has no "
                    f"{self.layout.pre_feed_forward_norm!r}, so the residual the "
                    "feed-forward writes into cannot be read"
                )
            handles.append(norm.register_forward_pre_hook(before_norm(layer)))
            handles.append(block.register_forward_hook(after_block(layer)))

        def tap(layer: int, x: torch.Tensor, y: torch.Tensor) -> None:
            contribution[layer] = y.detach()
            return None

        try:
            with self.block_intercept(tap):
                self.run(self.batch(self.render(self._self_check_records()[:2])))
        finally:
            for handle in handles:
                handle.remove()

        worst = 0.0
        for layer in range(len(blocks)):
            rebuilt = residual[layer] + contribution[layer]
            worst = max(worst, float((rebuilt - produced[layer]).abs().max()))
        record = {
            "max_absolute_difference": worst,
            "n_layers": len(blocks),
            "identity": (
                f"block output == {self.layout.pre_feed_forward_norm} input + "
                f"intercepted {self.layout.feed_forward} output"
            ),
            "verdict": "PASS" if worst == 0.0 else "FAIL",
        }
        if worst != 0.0:
            raise RuntimeError(
                f"{self.name}: the intercepted feed-forward output plus its input "
                f"residual differs from the block's own output by {worst:.3e}. The "
                "perturbation estimand is not what this interceptor reads, so a "
                "number measured here would not be the measurement the dense arms' is."
            )
        return record

    def self_check(self) -> dict[str, Any]:
        """The structural gate, and an honest statement of the one it does not have.

        A dense panel arm is gated on a **measured** frozen-input likelihood band,
        which is what catches a checkpoint that loaded without its weights or was
        fed the wrong rendering (L11, L24). No such band has been measured for a
        joint checkpoint -- it is reached by path precisely because it is not a
        panel arm -- so the band is recorded as withheld with its reason instead of
        being invented, and the two gates this arm does have are what run:
        resolution of the declared rendering against this tokenizer, which
        happened before the weights were read, and the estimand identity above.
        """

        return {
            "estimand": self.estimand_identity(),
            "rendering_resolved": self.tokenisation is not None,
            "likelihood_band": {
                "verdict": "WITHHELD",
                "reason": (
                    "no frozen-input likelihood band has been measured for this "
                    "checkpoint, because it is reached by path rather than declared "
                    "in src.transfer.arms.PANEL (a checkpoint that has not passed "
                    "21_joint_mode_qualification.py must not be in the panel). This "
                    "arm's loader gate is therefore structural only: a checkpoint "
                    "that loaded without its weights would be caught by neither the "
                    "rendering resolution nor the estimand identity. Read this "
                    "artefact beside that stage's context-information verdict for "
                    "the same checkpoint and mode"
                ),
            },
            "n_records": len(self._self_check_records()),
            "verdict": "PASS",
        }


# ------------------------------------------------------------------ dispatch


def eligible_arms(campaign_panel: Sequence[str]) -> list[str]:
    """The ``--arm`` values these stages accept, and where each comes from.

    Composed rather than written down. ``campaign_panel`` is
    ``panel_contract.CAMPAIGN_PANEL``, which decides which checkpoints a campaign
    may schedule at all; :data:`DENSE_ARCHITECTURES` decides which module layouts
    carry this estimand; :data:`MEASURED_DENSE_SELF_CHECK_NLL` decides which arms
    have a loader gate that can fail. Three declarations, composed here and
    restated nowhere (Appendix B rule 12). ``progen3`` is prepended because it is
    a third-party checkpoint rather than a panel member, so no panel declaration
    can admit it.

    The panel is passed in rather than imported because it is declared in
    ``scripts/transfer/panel_contract.py``: the stages already import it, and a
    library module that reached into the stage directory to validate itself would
    invert the dependency the contract file exists to keep one-way.
    """

    unknown = sorted(set(campaign_panel) - set(PANEL))
    if unknown:
        raise KeyError(f"campaign panel names arms outside src.transfer.arms.PANEL: {unknown}")
    return [PROGEN3_ARM] + [
        name
        for name in campaign_panel
        if PANEL[name].architecture in DENSE_ARCHITECTURES
        and name in MEASURED_DENSE_SELF_CHECK_NLL
    ]


def perturbable_arms(campaign_panel: Sequence[str]) -> list[str]:
    """The ``--arm`` values ``23_perturbation_sensitivity.py`` accepts.

    :func:`eligible_arms` widened by exactly the difference between the two
    estimands, and composed rather than written down for the same reason.

    A *replacement* reads a block's input and predicts its output, so it needs
    the serial post-attention layout :data:`DENSE_ARCHITECTURES` declares. A
    *perturbation* reads only the output, so it needs the weaker claim
    :data:`RESIDUAL_WRITE` declares -- that the intercepted tensor is a term the
    residual stream receives, under an identity verified exactly on the live
    forward pass. Two additions follow:

    * the **parallel-residual panel arms** (``progen2-small``,
      ``progen2-medium``), which a campaign already schedules and which the
      replacement stages still refuse;
    * the **staged non-members** (:data:`src.transfer.arms.STAGED_ARMS`), which
      complete ``src.transfer.arms.PROTEIN_SCALE_LADDER``. They are reachable
      here and not in :data:`~src.transfer.arms.PANEL` because a panel arm
      carries campaign obligations -- above all the ``budget`` family, whose
      ``arm_power`` reads ``config.vocab_size`` -- that a tolerance measurement
      does not need and that these two checkpoints cannot meet.

    Both additions still require a measured loader band, so an arm nobody has
    scored on the frozen inputs is not reachable from here either.
    """

    admitted = eligible_arms(campaign_panel)
    admitted += [
        name
        for name in campaign_panel
        if PANEL[name].architecture in PARALLEL_ARCHITECTURES
        and name in MEASURED_DENSE_SELF_CHECK_NLL
    ]
    admitted += [
        name
        for name in sorted(STAGED_ARMS)
        if STAGED_ARMS[name].architecture in RESIDUAL_WRITE
        and name in MEASURED_DENSE_SELF_CHECK_NLL
    ]
    return admitted


def replaceable_implementation(architecture: str) -> type[DenseReplaceable]:
    """Which implementation covers one architecture's block layout.

    Resolved from the classes' own :attr:`DenseReplaceable.architectures`
    declarations rather than from a third table, so an architecture cannot be
    admitted by one and dispatched by the other.
    """

    for implementation in (DenseReplaceable, ParallelResidualReplaceable):
        if architecture in implementation.architectures:
            return implementation
    raise TypeError(
        f"no replaceable implementation covers {architecture!r}; declared: "
        + ", ".join(
            f"{cls.__name__}{sorted(cls.architectures)}"
            for cls in (DenseReplaceable, ParallelResidualReplaceable)
        )
    )


def load_replaceable(
    arm: str,
    *,
    campaign_panel: Sequence[str],
    admissible: Sequence[str] | None = None,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    max_tokens: int = 512,
    checkpoint: Path | None = None,
) -> ReplaceableModel:
    """Load one arm behind the shared interface, refusing an ineligible name.

    ``admissible`` is the set the *calling stage* declared, so that the names its
    ``--arm`` offers and the names this function accepts are one list. ``None``
    means :func:`eligible_arms`, which is the replacement stages' set and their
    long-standing behaviour; ``23_perturbation_sensitivity.py`` passes
    :func:`perturbable_arms`.

    ``checkpoint`` relocates ProGen3's weights only; every other checkpoint's
    location is its declaration's and is relocated by its own environment
    variable (:attr:`src.transfer.arms.ArmSpec.path_variable`), never by a stage
    flag.
    """

    if arm == PROGEN3_ARM:
        return ProGen3Replaceable(
            load_progen3(
                checkpoint=PROGEN3_CHECKPOINT if checkpoint is None else checkpoint,
                device=device,
                dtype=getattr(torch, dtype),
            )
        )
    if checkpoint is not None:
        raise ValueError(
            f"--checkpoint relocates ProGen3's weights; {arm} is a declared "
            "checkpoint whose location is declared by "
            f"{arm_spec(arm).path_variable if arm in PANEL or arm in STAGED_ARMS else 'the panel'}"
        )
    admitted = list(eligible_arms(campaign_panel) if admissible is None else admissible)
    if arm not in admitted:
        raise ValueError(f"arm {arm!r} cannot be measured here; eligible: {admitted}")
    spec = arm_spec(arm)
    implementation = replaceable_implementation(spec.architecture)
    return implementation(
        load_arm_spec(spec, device=device, dtype=dtype), max_tokens=max_tokens
    )
