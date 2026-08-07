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
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import torch

from .arms import PANEL, Arm, Cohort, load_arm, tokenize_batch
from .io import sha256_file
from .path_patching import attention_output_projection
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

#: Architectures whose blocks carry the ProGen3 estimand unchanged: a block whose
#: input is the post-attention normalisation and whose output is added to the
#: residual stream.
#:
#: ``gpt2`` only, and deliberately. ``progen`` (ProGen2) is a *parallel* residual
#: block -- its feed-forward reads the same normalisation as its attention, not a
#: post-attention one -- so a transcoder trained at that tap predicts a different
#: object and the two arms' recovery ratios would not be the same measurement.
#: The rotary lineages (``llama``, ``qwen2``) do have the serial layout, and are
#: still excluded: nothing here has been verified against them, and the identity
#: check in :meth:`DenseReplaceable.self_check` is a verification of the arm that
#: runs, not a licence for the ones that have not.
DENSE_ARCHITECTURES = frozenset({"gpt2"})

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
    return "text" if PANEL[arm].modality == "text" else "protein"


def arm_evaluation_cohort_source(arm: str) -> str:
    """The corpus an arm's evaluation cohort is drawn from.

    The panel already declares this per arm; ProGen3 is not a panel member and
    declares it here. Answerable before a checkpoint is loaded, because a stage
    draws its cohort first and a missing corpus should fail in a second rather
    than after a model is on the GPU.
    """

    if arm == PROGEN3_ARM:
        return PROGEN3_EVALUATION_COHORT_SOURCE
    return PANEL[arm].evaluation_cohort_source


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
    return PANEL[arm].evaluation_cohort_source


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
    and never by hand.

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
    def render(self, records: Sequence[str]) -> list[str]:
        """Cohort or corpus records as this model's own input strings."""

    @abstractmethod
    def batch(self, inputs: Sequence[str]) -> dict[str, torch.Tensor]:
        """Model kwargs for one batch of rendered inputs, on this model's device."""

    @abstractmethod
    def content_mask(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Positions carrying content: everything but padding and markers."""

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

    # -- components --------------------------------------------------------

    def components(self) -> list[Component]:
        """Every attention head, then every replaceable block, in layer order."""

        return component_grid(self.n_layers, self.n_heads, block_kind=self.block_kind)

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

    def render(self, records: Sequence[str]) -> list[str]:
        # ProGen3's batch preparer adds the terminus and direction tokens itself,
        # so a record IS the input string and wrapping it here would render twice.
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

    def ablated(self, component: Component) -> Any:
        return progen3_ablated(self.pg, component)

    def self_check(self) -> dict[str, Any]:
        return progen3_self_check(self.pg)


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

#: The token cap the self-check tokenises under, declared here rather than taken
#: from the run's own ``--max-tokens``. Both frozen sets fit inside it, so the
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
MEASURED_DENSE_SELF_CHECK_NLL: dict[str, float] = {
    "gpt2": 3.7651,
    "gpt2-large": 3.1706,
    "protgpt2": 5.2006,
}

#: What the same measurement gives when the arm is broken in the two ways that do
#: not raise. Recorded so that the band below is sized against a measured
#: distance rather than against taste, exactly as ProGen3's is:
#:
#: * ``protgpt2_rendered_raw`` -- the FASTA wrapping removed, which is L11's
#:   defect, the one that cost a retraction. Worth 1.42 nats/token on the 600-2000
#:   residue cohort it was priced on and **4.38** on these eight short records,
#:   where the missing end-of-text prefix is a larger share of the sequence.
#: * ``gpt2_randomly_initialised`` -- the same architecture and tokenizer built
#:   from its config with no weights read, which is L24's shape on a dense arm.
MEASURED_DENSE_SELF_CHECK_CORRUPTIONS: dict[str, float] = {
    "protgpt2_rendered_raw": 9.5802,
    "gpt2_randomly_initialised": 11.0789,
}

#: Half-width of the band :func:`check_dense_nll` accepts, in nats/token.
#:
#: Sized from two measurements, like ProGen3's. The **spread of a correct arm**
#: across what an environment can change is at most 0.015 nats: over batch sizes
#: 1, 4 and 8 at bfloat16, gpt2 moves 3.7537/3.7651/3.7685, gpt2-large 0.0008 and
#: ProtGPT2 0.0049, and float16 moves each by less than 0.011. The **distance to
#: the nearest corruption that raises nothing** is 4.38 nats -- ProtGPT2 rendered
#: raw. A half-width of 0.30 is therefore 20x the observed spread and still leaves
#: four nats of clearance below the nearest silent failure.
#:
#: The lower end is not a numerical tolerance either. A value materially below the
#: measured one means the scored-target convention moved -- a mask that stopped
#: scoring the hard positions, say -- which corrupts everything downstream while
#: looking like an improvement.
DENSE_SELF_CHECK_HALF_WIDTH = 0.30

#: Panel arms this module could otherwise admit and does not, with the reason.
#: An unexplained absence and a decision must not be spelled the same way, which
#: is the discipline ``panel_contract.PANEL_MEMBERS_NOT_STAGED`` applies to the
#: campaign panel; :func:`_check_dense_arms` makes it an import-time failure here.
DENSE_ARMS_WITHOUT_A_BAND: dict[str, str] = {
    "zymctrl": (
        "its rendering carries an EC conditioning tag worth 1.73 nats of leak "
        "(L15), and every scored position in this stage's convention is inside "
        "the conditioned span. Admitting it needs "
        "arms.conditioning_boundary_ids applied to the scored-target mask, which "
        "neither stage carries; a band measured without that would be a band on "
        "the leak"
    ),
    "gpt2-medium": (
        "no band measured. The ladder rungs are admissible in principle and each "
        "costs one short scoring run; only the three arms the matched-pair "
        "comparison needs (gpt2-large, protgpt2) and its cheap smoke arm (gpt2) "
        "were measured"
    ),
    "gpt2-xl": "no band measured; as gpt2-medium",
    "dialogpt-small": "no band measured; as gpt2-medium",
}


def _check_dense_arms() -> None:
    """A band names a real dense arm, and no arm is both measured and refused.

    The other half of this invariant -- that every *staged* dense arm is in one
    table or the other -- is checked in ``tests/test_replaceable_arms.py``,
    because it needs ``panel_contract.CAMPAIGN_PANEL`` and a library module must
    not import a stage script to validate itself.
    """

    for name, value in MEASURED_DENSE_SELF_CHECK_NLL.items():
        if name not in PANEL:
            raise AssertionError(f"{name} has a self-check band but is not a panel arm")
        if PANEL[name].architecture not in DENSE_ARCHITECTURES:
            raise AssertionError(
                f"{name} has a self-check band but its {PANEL[name].architecture!r} "
                "architecture is not declared dense-replaceable"
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
    record = {
        "nll": float(value),
        "band": [float(low), float(high)],
        "reference": float(reference),
        "corruptions": dict(MEASURED_DENSE_SELF_CHECK_CORRUPTIONS),
        "verdict": "PASS" if inside else "FAIL",
    }
    if not inside:
        raise RuntimeError(
            f"{arm} self-check NLL {value:.4f} nats/token is outside the declared "
            f"band [{low:.4f}, {high:.4f}]. Above the band the most likely cause is "
            "a wrong rendering or a checkpoint that did not load its weights, "
            f"neither of which raises; reference corruptions: "
            f"{MEASURED_DENSE_SELF_CHECK_CORRUPTIONS}. Below it, the scored-target "
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
    """

    block_kind = DENSE_BLOCK_KIND
    loading_note = (
        "src.transfer.arms.load_arm: AutoModelForCausalLM at a declared dtype, "
        "with the checkpoint's depth and width checked against the panel "
        "declaration and the loaded dtype read back from the parameters"
    )
    scoring_note = "left to right, every non-padding target after the first"

    def __init__(self, arm: Arm, *, max_tokens: int) -> None:
        if arm.spec.architecture not in DENSE_ARCHITECTURES:
            raise TypeError(
                f"{arm.name}: {arm.spec.architecture!r} is not declared "
                f"dense-replaceable ({sorted(DENSE_ARCHITECTURES)}); its block does "
                "not carry the replacement estimand this measurement is defined on"
            )
        if arm.name not in MEASURED_DENSE_SELF_CHECK_NLL:
            reason = DENSE_ARMS_WITHOUT_A_BAND.get(arm.name, "no band measured")
            raise ValueError(f"{arm.name} cannot be measured here: {reason}")
        if max_tokens < 1:
            raise ValueError("--max-tokens must be positive")
        self.arm = arm
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
        """The checkpoint's weight files, preferring safetensors.

        Declared rather than globbed for anything readable: a checkpoint
        directory that carries both formats would otherwise digest whichever the
        glob happened to order first, and two runs of one arm would disagree
        about its identity.
        """

        for pattern in ("*.safetensors", "pytorch_model*.bin"):
            found = sorted(self.checkpoint.glob(pattern))
            if found:
                return found
        raise FileNotFoundError(
            f"{self.checkpoint} carries no *.safetensors and no pytorch_model*.bin, "
            "so the loaded weights cannot be identified in the artefact"
        )

    def weights_digest(self) -> str:
        """SHA-256 of the weight file, or of the shard digests when there are several."""

        files = self.weights_files()
        if len(files) == 1:
            return sha256_file(files[0])
        combined = hashlib.sha256()
        for path in files:
            combined.update(f"{path.name} {sha256_file(path)}\n".encode())
        return combined.hexdigest()

    # -- inputs ------------------------------------------------------------

    def render(self, records: Sequence[str]) -> list[str]:
        """Through :meth:`src.transfer.arms.Cohort.input_strings`, never by hand."""

        return Cohort(
            name=f"{self.arm.name}_render",
            kind=self.cohort_kind,
            records=list(records),
            min_symbols=0,
            max_symbols=0,
        ).input_strings(self.arm)

    def batch(
        self, inputs: Sequence[str], *, max_tokens: int | None = None
    ) -> dict[str, torch.Tensor]:
        ids, mask = tokenize_batch(
            self.arm, list(inputs), self.max_tokens if max_tokens is None else max_tokens
        )
        device = self.device
        return {"input_ids": ids.to(device), "attention_mask": mask.to(device)}

    def content_mask(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Non-padding, non-special positions.

        The pad token *is* the end-of-text token on every GPT-2-lineage arm, so
        the mask cannot be built from the ids alone -- padding and ProtGPT2's
        end-of-text prefix carry the same id, and one of them is a position the
        model was trained to predict from. The validity mask separates them; the
        special-token ids then remove the marker itself, which is the counterpart
        of ProGen3 excluding its terminus tokens.
        """

        ids = batch["input_ids"]
        mask = batch["attention_mask"].bool()
        special = [
            token for token in self.arm.tokenizer.all_special_ids if token is not None
        ]
        if special:
            marker = torch.tensor(special, device=ids.device)
            mask = mask & ~torch.isin(ids, marker)
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
        """

        output = self.run(batch)
        ids = batch["input_ids"]
        return (
            output.logits[..., :-1, :].float(),
            ids[..., 1:],
            batch["attention_mask"][..., 1:].bool(),
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

    @torch.no_grad()
    def estimand_identity(self) -> dict[str, Any]:
        """Verify that the intercepted pair IS the replacement estimand.

        A transcoder trained here reads what :meth:`block_intercept` calls the
        block input and predicts what it calls the block output, and the whole
        cross-model comparison rests on that being the same object ProGen3's
        transcoder is trained on: the post-attention normalisation's output, and
        the term added to the residual stream. Both halves are checked against
        the live forward pass -- the block's own output must equal its input
        residual plus the intercepted feed-forward output -- and the tolerance is
        **exact**, because the model performs that addition on the same two
        tensors in the same dtype, so anything but equality means the block does
        something this interceptor does not see.
        """

        blocks = list(self.arm.blocks())
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
                produced[layer] = (output[0] if isinstance(output, tuple) else output).detach()

            return hook

        for layer, block in enumerate(blocks):
            if not hasattr(block, "ln_2"):
                raise TypeError(
                    f"{self.arm.name}: block {layer} has no ln_2, so the residual the "
                    "feed-forward writes into cannot be read"
                )
            handles.append(block.ln_2.register_forward_pre_hook(before_norm(layer)))
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
        for layer in range(self.n_layers):
            rebuilt = residual[layer] + contribution[layer]
            worst = max(worst, float((rebuilt - produced[layer]).abs().max()))
        record = {
            "max_absolute_difference": worst,
            "n_layers": self.n_layers,
            "identity": "block output == ln_2 input + intercepted feed-forward output",
            "verdict": "PASS" if worst == 0.0 else "FAIL",
        }
        if worst != 0.0:
            raise RuntimeError(
                f"{self.arm.name}: the intercepted feed-forward output plus its "
                f"input residual differs from the block's own output by {worst:.3e}. "
                "The transcoder estimand is not what this interceptor reads, so a "
                "replacement measured here would not be the measurement ProGen3's is."
            )
        return record

    def _self_check_records(self) -> tuple[str, ...]:
        return (
            SELF_CHECK_DOCUMENTS
            if self.cohort_kind == "text"
            else SELF_CHECK_SEQUENCES
        )

    @torch.no_grad()
    def self_check(self) -> dict[str, Any]:
        """The estimand identity, then the scored band. Both refuse rather than report."""

        estimand = self.estimand_identity()
        record = check_dense_nll(self.arm.name, dense_self_check_nll(self))
        record["estimand"] = estimand
        record["n_records"] = len(self._self_check_records())
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

    records = model._self_check_records()
    inputs = model.render(records)
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


def load_replaceable(
    arm: str,
    *,
    campaign_panel: Sequence[str],
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    max_tokens: int = 512,
    checkpoint: Path | None = None,
) -> ReplaceableModel:
    """Load one arm behind the shared interface, refusing an ineligible name.

    ``checkpoint`` relocates ProGen3's weights only; a panel arm's location is
    the panel's declaration and is relocated by its own environment variable
    (:attr:`src.transfer.arms.ArmSpec.path_variable`), never by a stage flag.
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
            f"--checkpoint relocates ProGen3's weights; {arm} is a panel arm whose "
            "location is declared by "
            f"{PANEL[arm].path_variable if arm in PANEL else 'the panel'}"
        )
    admissible = eligible_arms(campaign_panel)
    if arm not in admissible:
        raise ValueError(f"arm {arm!r} cannot be measured here; eligible: {admissible}")
    return DenseReplaceable(
        load_arm(arm, device=device, dtype=dtype), max_tokens=max_tokens
    )
