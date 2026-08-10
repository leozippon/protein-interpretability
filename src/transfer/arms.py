"""The matched model panel and its frozen evaluation cohorts.

Every transfer measurement compares a text decoder against protein decoders, so
the panel must be matched where it can be and its mismatches must be recorded
rather than hidden. GPT-2-large and ProtGPT2 share depth, width and vocabulary
size exactly, which makes that pair the controlled comparison; ZymCTRL and
ProGen2-medium differ and their differences are declared in ``ArmSpec``.

Cohorts are content-addressed. A cohort is a frozen, hashed list of sequences
plus the per-arm input strings derived from it, so that two runs either use the
same cohort or fail loudly.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import os


def env_path(variable: str, default: Path) -> Path:
    """Read one input location from the environment, defaulting to the L20 host.

    The same code runs on the local L20 host and inside H200 pods, which mount
    their checkpoints and corpora elsewhere. Every location is therefore a named
    variable whose default is the local value, so an unset environment
    reproduces local behaviour exactly and relocating a host is a matter of
    exports rather than of edits.

    Existence is deliberately not checked here. A module-level check would make
    importing the package depend on data that a given measurement never touches,
    and would fail while a corpus is still being staged; it is checked at first
    use instead, by :func:`require_input_path`.
    """

    return Path(os.environ.get(variable, str(default)))


def require_input_path(path: Path, variable: str) -> Path:
    """Fail on a missing input, naming the variable that relocates it.

    A missing corpus or checkpoint has to stop the run, because both ways of
    tolerating it are invisible in the numbers that follow. ``transformers``
    treats an absent local directory as a Hub repository id and goes to the
    network; a glob over a missing directory returns an empty list, which reads
    downstream as "no eligible records" rather than as "wrong host".
    """

    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist; set {variable} to its location on this host"
        )
    return path


#: Repository root. It is only a default for the corpora below, each of which
#: carries its own variable, so a host that mounts data outside its checkout
#: does not have to pretend that the two live together.
REPO = env_path("TRANSFER_REPO_ROOT", Path(__file__).resolve().parents[2])
MODEL_ROOT = env_path("TRANSFER_MODEL_BASE_DIR", REPO.parent / "models")
#: Parent of the text checkpoints that are addressed by name rather than
#: declared one by one: the ByGPT5 rungs below and the GPT-2 ladder in
#: :mod:`src.transfer.scaling`.
TEXT_MODEL_BASE = env_path("TRANSFER_TEXT_MODEL_BASE_DIR", REPO.parent / "text_models")
TEXT_MODEL_ROOT = env_path("TRANSFER_TEXT_MODEL_DIR", TEXT_MODEL_BASE / "gpt2-large")
OPENWEBTEXT = env_path(
    "TRANSFER_OPENWEBTEXT_DIR", Path("/Data/public/datasets/openwebtext-screen/plain_text")
)
SWISSPROT_FASTA = env_path(
    "TRANSFER_SWISSPROT_FASTA", REPO / "data/swissprot/uniprot_sprot.fasta.gz"
)
ZYMCTRL_FASTA = env_path(
    "TRANSFER_ZYMCTRL_FASTA", REPO / "data/zymctrl/ec_labeled_swissprot.fasta"
)
#: ProGen3's own pretraining-scale corpus, and the one ProGenMech trains its
#: transcoders on. ``TRANSFER_UNIREF50_FASTA`` was already exported to the pod
#: and named in the resource manifest while no module declared it, so every
#: caller resolved it by hand or not at all.
UNIREF50_FASTA = env_path(
    "TRANSFER_UNIREF50_FASTA", REPO / "data/uniref50/uniref50.fasta"
)

#: Named in the failure message when an arm's checkpoint is absent. Which of the
#: three applies depends on the arm, and an operator needs the candidate list
#: rather than the one this module happened to resolve.
_MODEL_PATH_VARIABLES = "TRANSFER_MODEL_BASE_DIR, TRANSFER_TEXT_MODEL_DIR or TRANSFER_TEXT_MODEL_BASE_DIR"

AA20 = "ACDEFGHIKLMNPQRSTVWY"

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

#: Value of :attr:`ArmSpec.pretraining_corpus` for a checkpoint whose model card
#: does not state its training corpus. It is a sentinel rather than a guess:
#: inventing a corpus would put a false fact into every artefact that records the
#: panel, and the corpus contrasts below are only valid between arms that declare
#: one.
PRETRAINING_UNDECLARED = "undeclared"

#: How a cohort's records were drawn from their corpus. ``seeded_permutation``
#: is the only mode Appendix B rule 1 of the transfer audit permits for a
#: reported number; ``file_order`` is retained because several frozen artefacts
#: were produced with it and must remain reproducible, and because a census over
#: a whole corpus is order-independent. The mode is written into every cohort's
#: metadata so that no artefact can be read without knowing which one produced
#: it -- that invisibility, not the file order itself, is what manufactured an
#: effect three times.
SAMPLING_MODES = ("seeded_permutation", "file_order")

#: The seed every campaign stage draws its corpus under, declared once.
#:
#: Two facts make a shared constant the right object rather than a per-stage
#: default. A stage's cohort must be comparable with the cohort
#: ``01_cohort_power.py`` *qualified* its arms on, and cohorts drawn under
#: different seeds are different populations -- EXP-R2-060 priced protein
#: cohort-block sensitivity at 0.16-0.60 nats, which is the size of several
#: effects this programme has reported. And a per-stage default is a second
#: declaration of the same decision, which is the hazard Appendix B rule 12 was
#: written about.
#:
#: A stage may still be pointed at a different draw from the command line, and
#: ``0`` selects the historical file-order draw. Both are declared choices that
#: reach the artefact through :attr:`Cohort.sampling`; neither is a default that
#: nobody notices.
DEFAULT_CORPUS_DRAW_SEED = 20260728

#: Named in the sampling record of a file-order cohort so the hazard travels
#: with the number rather than living only in a document.
FILE_ORDER_HAZARD = (
    "records are the first eligible entries of the corpus in file order; "
    "biological corpora are grouped by family, so a head-of-file draw is a set "
    "of near-clonal homologues rather than a sample. Pass seed= to draw under a "
    "seeded permutation (transfer audit, Appendix B rule 1)"
)


#: Measurement families an arm can legitimately enter. These are capabilities,
#: not preferences: an arm without a capability must raise rather than return a
#: number that is not commensurate with the rest of the panel.
CAPABILITIES = frozenset({"budget", "lens", "pathway", "circuits", "relational"})

#: Architectures that follow the Llama module convention: a block list at
#: ``model.layers``, attention at ``block.self_attn``, RMSNorm in place of
#: LayerNorm, a gated feed-forward, grouped-query attention and rotary position
#: embeddings in place of a learned position table. They are grouped here because
#: the panel resolves all of them identically, and are still declared one by one
#: on each :class:`ArmSpec` because they are different pretraining runs whose
#: differences a downstream fit has to be able to separate.
_ROTARY_DECODERS = frozenset({"llama", "qwen2"})

#: Where each architecture keeps a block's attention submodule, as the path from
#: the block down to it. Declared per architecture rather than found by trying
#: attribute names in turn: a panel member whose attention cannot be named is a
#: panel change that must be declared, and a search would silently resolve a new
#: architecture to whichever candidate attribute happened to exist on it.
#:
#: A path rather than a single attribute because ByGPT5's attention is not an
#: attribute of its block at all: a ``ByGPT5Block`` holds a ``ModuleList`` whose
#: first entry is the ``T5LayerSelfAttention`` wrapper and whose second is the
#: ``nn.Identity`` that replaces cross-attention, and the module that computes
#: the pattern is the ``T5Attention`` inside that wrapper. Integers index, strings
#: are attributes.
_ATTENTION_PATH: dict[str, tuple[str | int, ...]] = {
    "gpt2": ("attn",),
    "progen": ("attn",),
    **{architecture: ("self_attn",) for architecture in _ROTARY_DECODERS},
    "t5_decoder": ("layer", 0, "SelfAttention"),
}

#: Architectures whose per-sublayer decomposition is commensurate with a
#: standard causal decoder.
#:
#: Declared explicitly rather than derived from :data:`_ATTENTION_PATH`, because
#: the two questions have come apart and it is now their *difference* that has to
#: be kept honest. Naming an architecture's attention submodule is what a pattern
#: read needs; claiming its sublayers are the same objects the rest of the panel
#: decomposes is a stronger statement, and ByGPT5's T5 decoder satisfies the first
#: and not the second -- its relative position bias and gated feed-forward make
#: its sublayers different objects, so it is reachable for a pattern and refused
#: for a decomposition. Reformer is absent from both: LSH attention and reversible
#: layers mean its "attention share" and "residual stream" are not the same
#: quantities the rest of the panel measures. Grouped-query attention is not such
#: a reason: sharing one value projection across a group of query heads changes
#: how a per-head decomposition must be built but not what the attention pathway
#: is, and the pathway measurements ablate a whole sublayer output rather than a
#: head.
_DECOMPOSABLE = frozenset({"gpt2", "progen", *_ROTARY_DECODERS})


@dataclass(frozen=True)
class MlpNeuronTensor:
    """How one architecture's MLP **hidden** activation is reached and checked.

    **The tensor is the MLP's hidden layer, never the MLP's output**, and the
    distinction is the whole content of this declaration. A GPT-2 feed-forward is
    ``c_proj(act(c_fc(x)))``: ``c_fc`` lifts the residual stream from ``d_model``
    to ``d_mlp``, the nonlinearity is applied there, and ``c_proj`` projects back
    down. A "neuron" is a coordinate of that ``d_mlp``-wide post-nonlinearity
    tensor. The ``d_model``-wide output is a different object -- it is a dense
    mixture of every neuron and is far less sparse -- so a neuron-basis circuit
    measured there would understate what a sparse basis can recover on *any*
    arm, and on this panel that would manufacture the conclusion that protein
    models specifically need a learned dictionary.

    Declared per architecture and reached by walking a path, exactly as
    :data:`_ATTENTION_PATH` is and for the same reason: an undeclared
    architecture must raise rather than resolve to whichever attribute happens to
    exist on it. The gated rotary lineages are absent on purpose -- their hidden
    tensor is the product of two projections, so its coordinates are not the same
    object a non-gated GELU neuron is -- and so is ``progen``, whose parallel
    residual block this programme's replacement estimand already excludes.
    """

    #: Path from the MLP module down to the module whose **input** is the hidden
    #: tensor. Hooking the down-projection's input rather than the nonlinearity's
    #: output is what makes the tensor definitionally the one the projection
    #: consumes, with no second opinion about where the MLP's stages begin.
    down_projection: tuple[str, ...]
    #: Config attributes that declare ``d_mlp``, in the order they are consulted.
    width_attributes: tuple[str, ...]
    #: What ``d_mlp`` is when every one of those is absent or ``None``. Every
    #: GPT-2 checkpoint on this panel leaves ``n_inner`` unset and means
    #: ``4 * n_embd`` by it. ``None`` here means an implicit width is not defined
    #: for the architecture and an absent attribute must raise.
    implicit_width_multiple: int | None
    #: The config attribute naming the nonlinearity.
    activation_attribute: str
    #: Greatest lower bound of each **declared** nonlinearity's output. The
    #: hidden tensor is post-nonlinearity by construction, so a measured minimum
    #: below this bound means something else was read: a pre-activation tensor is
    #: unbounded below and gives itself away immediately. An activation that is
    #: not declared raises rather than being checked against a bound that may not
    #: hold for it, which is also what keeps the non-gated claim honest -- only
    #: non-gated variants appear here.
    activation_lower_bound: dict[str, float]


#: Where each architecture keeps the activation its down-projection consumes.
#: ``gpt2`` only, and see :class:`MlpNeuronTensor` for why the others are absent.
#:
#: ``gelu_new`` is the tanh approximation GPT-2, ProtGPT2 and ZymCTRL all declare;
#: its minimum is -0.169 at x = -0.752, so -0.17 is a true lower bound and any
#: value materially below it is not a GELU output. ``gelu`` and
#: ``gelu_pytorch_tanh`` are the same function to within 1e-3 and share the bound;
#: ``relu`` is floored at zero.
_MLP_NEURON_TENSOR: dict[str, MlpNeuronTensor] = {
    "gpt2": MlpNeuronTensor(
        down_projection=("c_proj",),
        width_attributes=("n_inner", "intermediate_size"),
        implicit_width_multiple=4,
        activation_attribute="activation_function",
        activation_lower_bound={
            "gelu_new": -0.17,
            "gelu": -0.17,
            "gelu_pytorch_tanh": -0.17,
            "relu": 0.0,
        },
    ),
}


def mlp_neuron_declaration(architecture: str) -> MlpNeuronTensor:
    """The MLP hidden-tensor declaration for an architecture, or a refusal.

    Answerable from an architecture name alone, so a stage refuses an arm it
    cannot measure before a checkpoint reaches the GPU rather than after.
    """

    declared = _MLP_NEURON_TENSOR.get(architecture)
    if declared is None:
        raise TypeError(
            f"no MLP hidden-activation tensor is declared for {architecture!r}, so "
            "a neuron basis cannot be resolved for it; declared: "
            f"{sorted(_MLP_NEURON_TENSOR)}"
        )
    return declared


@dataclass(frozen=True)
class ArmSpec:
    """Declared properties of one panel member.

    ``capabilities`` encodes which measurement families this arm may enter. It
    exists because the panel deliberately spans architectures in order to break
    the modality/tokenisation collinearity, and that breadth is only safe if a
    non-commensurate arm cannot silently reach a metric that assumes a standard
    decoder. Prose caveats get lost; a raised exception does not.

    **Two corpora, two fields.** This class used to carry one field named
    ``source``, holding the corpus the arm's *evaluation cohort* is drawn from.
    Every text arm carried ``"openwebtext"``, which is true of the cohort and
    false of the pretraining data for six of the seven text arms -- and the bare
    name ``source`` reads as provenance, so the field invited exactly the
    misreading it could not support. The two facts are now separate fields:

    ``evaluation_cohort_source``
        The corpus this arm is *scored* on. It selects a rendering and a length
        band and it is what makes two arms' cross-entropies comparable; it says
        nothing about how the arm was trained.
    ``pretraining_corpus``
        The corpus the checkpoint was *trained* on, as stated by its own model
        card, or :data:`PRETRAINING_UNDECLARED` where no card states one. This is
        the field the corpus contrasts (:data:`MATCHED_DATA_CONTRAST`,
        :data:`TEXT_DATA_CONTRAST`) are defined against, and the one an
        interpretation like "the deficit is a property of the training data"
        depends on.

    ``source`` remains as a read-only alias of ``evaluation_cohort_source`` so
    that frozen artefact schemas and existing runners keep working; it should not
    be used in new code.
    """

    name: str
    path: Path
    #: The environment variable ``path`` is built from, declared rather than
    #: inferred. It used to be recovered by comparing the resolved ``path``
    #: against the three constants, which is correct only while the three resolve
    #: to *different* directories. The H200 pod sets
    #: ``TRANSFER_TEXT_MODEL_BASE_DIR="${TRANSFER_MODEL_BASE_DIR}"`` because every
    #: checkpoint sits in one GPFS directory, so on that host the comparison
    #: aliased and six text arms classified as protein-root arms. The worker
    #: re-derives the generated contract inside the pod and refused the campaign
    #: -- correctly, and only because that check exists. Appendix B rule 12: the
    #: declaration is made where the path is made.
    path_variable: str
    modality: str
    n_layer: int
    d_model: int
    tokenisation: str
    input_format: str
    evaluation_cohort_source: str
    architecture: str
    pretraining_corpus: str = PRETRAINING_UNDECLARED
    capabilities: frozenset[str] = CAPABILITIES

    @property
    def source(self) -> str:
        """Deprecated alias of :attr:`evaluation_cohort_source`.

        Kept because it is the spelling several frozen artefacts and runners
        use. It is *not* the pretraining corpus and never was.
        """

        return self.evaluation_cohort_source


PANEL: dict[str, ArmSpec] = {
    "gpt2-large": ArmSpec(
        name="gpt2-large",
        path=TEXT_MODEL_ROOT,
        path_variable="TRANSFER_TEXT_MODEL_DIR",
        modality="text",
        n_layer=36,
        d_model=1280,
        tokenisation="bpe",
        input_format="raw",
        evaluation_cohort_source="openwebtext",
        architecture="gpt2",
        pretraining_corpus="webtext",
    ),
    "protgpt2": ArmSpec(
        name="protgpt2",
        path=MODEL_ROOT / "ProtGPT2",
        path_variable="TRANSFER_MODEL_BASE_DIR",
        modality="protein",
        n_layer=36,
        d_model=1280,
        tokenisation="multi_residue_bpe",
        input_format="fasta_wrapped",
        evaluation_cohort_source="swissprot",
        architecture="gpt2",
        pretraining_corpus="uniref50",
    ),
    "zymctrl": ArmSpec(
        name="zymctrl",
        path=MODEL_ROOT / "ZymCTRL",
        path_variable="TRANSFER_MODEL_BASE_DIR",
        modality="protein",
        n_layer=36,
        d_model=1280,
        tokenisation="residue",
        input_format="ec_conditioned",
        evaluation_cohort_source="zymctrl_ec",
        architecture="gpt2",
        pretraining_corpus="uniprot_ec_annotated",
    ),
    "progen2-medium": ArmSpec(
        name="progen2-medium",
        path=MODEL_ROOT / "progen2-medium",
        path_variable="TRANSFER_MODEL_BASE_DIR",
        modality="protein",
        n_layer=27,
        d_model=1536,
        tokenisation="residue",
        input_format="n_to_c_control",
        evaluation_cohort_source="swissprot",
        architecture="progen",
        pretraining_corpus="uniref90_bfd30",
    ),
    # The protein-side scale rung the panel lacked. 151M against ProGen2-medium's
    # 765M, same architecture, same residue tokeniser, same UniRef90+BFD30
    # pretraining mixture -- so ProGen2-small / ProGen2-medium is a within-lineage
    # scale contrast on the PROTEIN side, holding corpus and tokenisation fixed.
    #
    # Why that matters more than one more arm. Until it was admitted, scale was
    # measurable only on the text side: the GPT-2 ladder falls monotonically
    # (0.1597 -> 0.0850 on the induction fraction) and the audit's scale-adjusted
    # restatement of the head-count shortfall rests entirely on that text-side
    # slope being transportable to protein, which no measurement had tested. This
    # rung tests it inside the protein lineage.
    #
    # Load-checked on the pod before admission (EXP-R2-068): 151.1M parameters,
    # 12 blocks of width 1024, 16 heads, vocab_size 32, n_positions 1024,
    # ProGenAttention, and a forward pass returning logits of width 32 against a
    # 31-token tokenizer -- the same shape as the two ProGen2 arms already here.
    "progen2-small": ArmSpec(
        name="progen2-small",
        path=MODEL_ROOT / "progen2-small",
        path_variable="TRANSFER_MODEL_BASE_DIR",
        modality="protein",
        n_layer=12,
        d_model=1024,
        tokenisation="residue",
        input_format="n_to_c_control",
        evaluation_cohort_source="swissprot",
        architecture="progen",
        pretraining_corpus="uniref90_bfd30",
    ),
    # Architecturally identical to progen2-medium down to the parameter count
    # (764,803,616), differing only in pretraining corpus. That makes the pair a
    # controlled contrast on training data with architecture, scale and
    # tokenisation all held fixed, which is the cleanest available test of
    # whether an interpretability metric tracks data rather than modality.
    "progen2-base": ArmSpec(
        name="progen2-base",
        path=MODEL_ROOT / "progen2-base",
        path_variable="TRANSFER_MODEL_BASE_DIR",
        modality="protein",
        n_layer=27,
        d_model=1536,
        tokenisation="residue",
        input_format="n_to_c_control",
        evaluation_cohort_source="swissprot",
        architecture="progen",
        pretraining_corpus="progen2_base_mixture",
    ),
}

# DialoGPT-small is GPT-2's architecture, GPT-2's tokenizer and GPT-2's size,
# pretrained on conversational Reddit threads instead of WebText. It is therefore
# the text-side analogue of the progen2-base/progen2-medium pair: corpus varies,
# everything else is held. It exists here because the panel had three protein
# arms spanning corpora, tokenisers and architectures against a single GPT-2
# lineage on the text side, which left every cross-modality difference open to
# the reading that it is a GPT-2-large idiosyncrasy rather than a modality one.
PANEL["dialogpt-small"] = ArmSpec(
    name="dialogpt-small",
    # Addressed through TEXT_MODEL_BASE like every other checkpoint named rather
    # than declared, so that a host which mounts its text models elsewhere moves
    # this arm with the rest instead of failing on it alone.
    path=TEXT_MODEL_BASE / "DialoGPT-small",
    path_variable="TRANSFER_TEXT_MODEL_BASE_DIR",
    modality="text",
    n_layer=12,
    d_model=768,
    tokenisation="bpe",
    input_format="raw",
    evaluation_cohort_source="openwebtext",
    architecture="gpt2",
    pretraining_corpus="reddit_dialogue",
)

# The within-lineage scale ladder, as measurable arms rather than only as
# convergence-control rungs.
#
# These three checkpoints already existed in :data:`src.transfer.scaling.
# DEFAULT_LADDER`, but a ``LadderMember`` is not an ``ArmSpec``: the ladder
# feeds the convergence fit, and nothing in it can be handed to the circuit
# census. That gap is why the induction-prevalence result is still confounded.
# ``llama-3.2-3b`` is simultaneously the lowest-scoring text arm and the widest
# and deepest, so "protein decoders have fewer induction heads" and "larger
# decoders have a lower fraction" currently predict the same table.
#
# gpt2 / gpt2-medium / gpt2-large / gpt2-xl separate them, because they vary
# scale over an order of magnitude -- 124M, 355M, 774M, 1558M; 12, 24, 36 and
# 48 layers; 12, 16, 20 and 25 heads -- while holding architecture, the
# 50257-piece BPE and the WebText corpus fixed. Whatever slope the induction
# fraction has against scale is therefore measured with lineage controlled, and
# it is measured *within* the text side, so it does not borrow identification
# from the modality contrast it is meant to adjudicate. If the fraction is flat
# or rising across this ladder, a scale explanation for the protein deficit is
# refuted on the text side alone; if it falls steeply, the deficit must be
# restated against a scale-matched expectation rather than against a mean.
#
# ProtGPT2 sits inside this ladder's range at 774,030,080 parameters and
# 36x1280 -- not merely the same depth and width as gpt2-large but the same
# parameter count to the unit, verified from both checkpoints -- so the ladder
# also yields a point prediction for what ProtGPT2's fraction *should* be under
# a pure scale account.
for _name, _dir, _n_layer, _d_model in (
    ("gpt2", "gpt2", 12, 768),
    ("gpt2-medium", "gpt2-medium", 24, 1024),
    ("gpt2-xl", "gpt2-xl", 48, 1600),
):
    PANEL[_name] = ArmSpec(
        name=_name,
        path=TEXT_MODEL_BASE / _dir,
        path_variable="TRANSFER_TEXT_MODEL_BASE_DIR",
        modality="text",
        n_layer=_n_layer,
        d_model=_d_model,
        tokenisation="bpe",
        input_format="raw",
        evaluation_cohort_source="openwebtext",
        architecture="gpt2",
        pretraining_corpus="webtext",
    )
del _name, _dir, _n_layer, _d_model

# Architecturally diverse text decoders. Every text arm above is one lineage:
# GPT-2's architecture, GPT-2's 50257-piece BPE and either WebText or a Reddit
# corpus scraped the same way, at five sizes. The protein side meanwhile spans
# three architectures, two tokenisation families and four corpora, so a
# text-versus-protein difference measured against that text side is not
# separable from a GPT-2-versus-everything-else difference. These two arms exist
# to make it separable, with models rather than with caveats.
#
# Both are rotary, RMSNorm, gated-feed-forward, grouped-query decoders -- a
# family that postdates every other member of the panel -- from two different
# laboratories, with byte-level BPE vocabularies two-and-a-half to three times
# GPT-2's learned over corpora GPT-2 never saw. Their tokenisation *family* is
# still ``bpe``, because a byte-level BPE over 128k or 152k pieces is the same
# kind of object on the subword/symbol axis the design identifies against; what
# they add is architecture, corpus and vocabulary scale, and the measured
# characters-per-token is what says how much.
#
# Rotary position embeddings were audited rather than assumed harmless. Nothing
# the granted capabilities touch reads a position table: `budget` and `pathways`
# see only forward passes and module outputs, and the two places in
# `src.transfer.circuits` that would care -- `direct_logit_attribution`, whose
# embedding term is `transformer.drop`'s output and therefore the token plus
# position sum, and `ov_copying_scores`, which reads `transformer.wte` and states
# that it drops positional embeddings -- both raise on the missing `transformer`
# attribute before reaching a position table. Position ids are the default
# `arange` rather than a cumulative sum of the attention mask, so right padding
# behaves exactly as it does for a learned table: pad positions sit after the
# content and are masked out of every causal query.
#
# Three further candidates were rejected on the evidence of their own model
# cards rather than on preference. Qwen3-0.6B, Qwen3-1.7B and
# Llama-3.2-1B-Instruct are all post-trained (SFT, preference optimisation and,
# for Qwen3, a thinking mode), while every other panel member is a pure
# next-token pretrained decoder. Admitting one would confound the architecture
# and corpus contrast these arms exist to draw with a training-objective
# contrast that no ``ArmSpec`` field records, and it would do so on the very
# axis the budget stage measures: post-training moves a model's cross-entropy on
# raw web text, which is the denominator of the convergence axis. No Qwen3 base
# checkpoint is staged on this host, so the Qwen3 generation cannot currently be
# admitted without that confound.

#: What a rotary decoder may enter. ``budget`` needs only a forward pass and a
#: tokenizer. ``lens`` is granted on the same footing as the ByGPT5 rungs: the
#: output aperture of a final normalisation followed by a linear unembedding is
#: the same quantity here as in GPT-2, but ``src.transfer.lenses.lens_head``
#: resolves that normalisation as ``transformer.ln_f`` and requires an
#: ``nn.LayerNorm`` with a learned bias, so until it grows an RMSNorm branch the
#: capability is an intent that ``src.transfer.scaling.lens_supported`` records
#: as a reasoned skip. ``pathway`` is granted outright: it ablates a whole
#: sublayer output and needs nothing but the block, MLP and attention modules
#: this file resolves.
#:
#: ``circuits`` is withheld. Grouped-query attention is *not* the obstacle -- a
#: correct per-head decomposition replicates each key/value head's ``W_V`` across
#: its group of query heads, and rebuilding an attention layer from those weights
#: reproduces the live forward pass to a relative maximum error of 2.6e-03 in
#: bfloat16 and 5e-07 in float32, far inside the 5e-02 tolerance
#: ``circuits.verify_head_decomposition`` enforces. The obstacle is that
#: ``src.transfer.circuits`` cannot express that decomposition or reach these
#: models at all: ``head_ov_weights`` knows only GPT-2's fused ``c_attn`` and
#: ProGen2's ``qkv_proj`` and infers ``d_head`` as ``d_model / n_head``;
#: ``verify_head_decomposition`` hooks ``block.ln_1``;
#: ``ov_copying_scores`` reads ``model.transformer.wte``; and
#: ``direct_logit_attribution`` requires ``transformer.drop``,
#: ``transformer.ln_f`` and a LayerNorm bias, none of which exist on an RMSNorm
#: decoder. Declaring the capability would not produce a wrong number, it would
#: produce an exception at an arbitrary depth of the run; withholding it produces
#: the panel's own refusal, with the reason attached.
# Updated 2026-07-28: ``circuits`` granted. The module now resolves the
# q_proj/v_proj/o_proj layout with the grouped-query index mapping, the
# per-architecture pre-attention norm, the tied embedding, and an explicit
# RMSNorm linearisation. Verified per architecture at bfloat16 against a 5e-2
# tolerance: OV rebuild error 4.11e-3 (Qwen2) and 5.69e-3 (Llama), about 3e-7 in
# float32, so those are rounding rather than structure.
#
# Two failure modes were checked rather than assumed, and they differ in a way
# that matters. Omitting Qwen2's ``v_proj`` bias gives a rebuild error of 0.518,
# ten times over tolerance, so it fails loudly. Applying LayerNorm's algebra to
# an RMSNorm decoder *passes* the reconstruction gate at 0.49% logit error while
# producing a systematically wrong attribution -- silent, and the reason the
# explicit normalisation-form resolution is load-bearing rather than cosmetic.
_ROTARY_CAPABILITIES = frozenset({"budget", "lens", "pathway", "circuits"})

# Qwen2.5-0.5B: the base checkpoint, pretrained only. 24 layers of width 896 is
# a depth-to-width ratio no other panel member has, and its 151936-piece
# vocabulary is the largest in the panel by a factor of three. Corpus: the
# Qwen2.5 pretraining mixture, multilingual web text with a heavy code and
# mathematics component, which shares neither language distribution nor
# provenance with WebText.
PANEL["qwen2.5-0.5b"] = ArmSpec(
    name="qwen2.5-0.5b",
    path=TEXT_MODEL_BASE / "Qwen2.5-0.5B",
    path_variable="TRANSFER_TEXT_MODEL_BASE_DIR",
    modality="text",
    n_layer=24,
    d_model=896,
    tokenisation="bpe",
    input_format="raw",
    evaluation_cohort_source="openwebtext",
    pretraining_corpus="qwen2.5_pretraining_mixture",
    architecture="qwen2",
    capabilities=_ROTARY_CAPABILITIES,
)

# Llama-3.2-3B: the base checkpoint, not the instruction-tuned sibling that is
# staged beside it. 28 layers matches ProGen2-medium's 27 almost exactly while
# differing in everything else, and at 3.2B parameters it extends the panel's
# scale range above GPT-2-xl. Corpus: up to 9T tokens of public web data with
# Llama-3.1 logits used as token-level targets during pretraining, per its model
# card -- a distillation signal no other panel member carries, recorded here
# because it is the one respect in which this arm is not a plain next-token run.
PANEL["llama-3.2-3b"] = ArmSpec(
    name="llama-3.2-3b",
    path=TEXT_MODEL_BASE / "Llama-3.2-3B",
    path_variable="TRANSFER_TEXT_MODEL_BASE_DIR",
    modality="text",
    n_layer=28,
    d_model=3072,
    tokenisation="bpe",
    input_format="raw",
    evaluation_cohort_source="openwebtext",
    pretraining_corpus="llama3_web_corpus_with_llama3.1_logit_distillation",
    architecture="llama",
    capabilities=_ROTARY_CAPABILITIES,
)

#: Same architecture and parameter count, different pretraining corpus.
MATCHED_DATA_CONTRAST = ("progen2-base", "progen2-medium")

#: The protein-side scale ladder: same architecture, same residue tokeniser, same
#: UniRef90+BFD30 mixture, 151M against 765M. The text side has had a four-rung
#: ladder since EXP-R2-057 and the protein side had none, which is why every
#: scale-adjusted statement about the head-count shortfall has had to assume the
#: text-side slope transports. This pair is the first protein-internal test of
#: that assumption. Two rungs is a slope estimate with no curvature, and it is
#: declared as a contrast rather than a ladder for that reason.
PROTEIN_SCALE_CONTRAST = ("progen2-small", "progen2-medium")

#: The text-side equivalent: gpt2 and DialoGPT-small share architecture,
#: tokeniser and size (12 layers, width 768, 50257 vocabulary) and differ only in
#: pretraining corpus. Comparing the two contrasts bounds how much of any
#: cross-modality difference is corpus rather than modality.
TEXT_DATA_CONTRAST = ("gpt2", "dialogpt-small")

#: The text arms that are outside the GPT-2 lineage in architecture, tokeniser
#: and corpus at once. A cross-modality difference has to survive replacing
#: GPT-2-large with these before it can be read as a text/protein difference
#: rather than a property of GPT-2, which is the objection they exist to answer.
#: Named here so that the arms carrying that argument are a value the analysis
#: can select on, not a fact a reader has to reconstruct from the panel.
TEXT_ARCHITECTURE_CONTRAST = ("qwen2.5-0.5b", "llama-3.2-3b")

# Byte-level text decoders. These exist to populate the text x symbol-level cell
# of the modality x tokenisation design, which was empty and left the two
# indicators nearly collinear. ProtGPT2 is the only public protein model with
# genuine subword tokenisation, so the protein x subword cell cannot be grown
# past n=1 and this is the only side of the design that can be repaired.
#
# The cost is an architecture difference, declared here rather than in prose:
# ByGPT5 is T5-derived (relative position bias, T5 layer norm, gated GELU) and
# admits no GPT-2-style sublayer decomposition, so it carries no `pathway`
# capability and no arm of it may enter the per-head circuit decomposition of
# `src.transfer.circuits`. Reformer uses LSH attention with reversible layers and
# is restricted to budget alone -- its attention share and residual stream are not
# the quantities the rest of the panel measures.
#
# Updated 2026-08-05: `circuits` granted, `pathway` still withheld, and the gap
# between the two is the point. The prediction-addressed-attention census reads
# and overrides *attention patterns*; it never splits a block into an attention
# and an MLP term, never rebuilds an OV circuit and never touches a position
# table, so the decomposition objection above does not reach it. Withholding
# `circuits` on that objection left the D2.c census with no byte-level TEXT arm at
# all, which is the one control that separates "symbol-level tokenisation" from
# "protein model" in its head-retrieval result: every symbol-level arm in the
# panel is otherwise a protein decoder (transfer audit, EXP-R2-114).
#
# What the grant admits and what it does not was checked rather than assumed.
# `circuit_primitives` and `induction_path_patching` both gate on their own
# module's architecture declaration -- `circuits._CIRCUIT_ARCHITECTURES` and
# `path_patching.SUPPORTED_ARCHITECTURES` -- and neither contains `t5_decoder`,
# so those stages still refuse these arms with their own reason attached.
for _name, _layers, _width in (
    ("bygpt5-small-en", 4, 1472),
    ("bygpt5-base-en", 6, 1536),
    ("bygpt5-medium-en", 12, 1536),
):
    PANEL[_name] = ArmSpec(
        name=_name,
        path=TEXT_MODEL_BASE / _name,
        path_variable="TRANSFER_TEXT_MODEL_BASE_DIR",
        modality="text",
        n_layer=_layers,
        d_model=_width,
        tokenisation="byte",
        input_format="raw",
        evaluation_cohort_source="openwebtext",
        # No ByGPT5 model card on this host states a pretraining corpus, so no
        # corpus contrast is defined against these rungs. A guess here would be a
        # false fact in every artefact that records the panel.
        pretraining_corpus=PRETRAINING_UNDECLARED,
        architecture="t5_decoder",
        capabilities=frozenset({"budget", "lens", "circuits"}),
    )

# google/reformer-enwik8 was staged as an architecturally independent byte-level
# text model and is deliberately NOT in the panel. It ships no tokenizer: the
# checkpoint expects text encoded manually as byte+2, and AutoTokenizer resolves
# a ReformerTokenizer that then fails for want of a sentencepiece vocab file.
# Admitting it would mean a bespoke tokenizer shim in this shared module for a
# model that, being LSH-attention and reversible-layered, can only ever
# contribute to the `budget` family anyway. Three ByGPT5 rungs already populate
# the text x byte-level cell. The checkpoint remains on disk if an independent
# architecture check is later judged worth that cost.

#: The exactly matched pair. Depth, width and vocabulary size are identical, so
#: any difference between these two is a modality difference, not an
#: architecture difference.
MATCHED_PAIR = ("gpt2-large", "protgpt2")


def _check_corpus_contrast(pair: tuple[str, str]) -> None:
    """A corpus contrast must hold everything but the pretraining corpus fixed.

    The two pairs below are the only evidence in the panel for how much of a
    cross-modality difference is training data rather than modality, and the
    claim rests entirely on what is held fixed. That used to be a comment; it is
    now checked at import, because the field it depends on
    (:attr:`ArmSpec.pretraining_corpus`) is new and a future edit that leaves it
    at :data:`PRETRAINING_UNDECLARED` would turn the contrast into a comparison
    of two arms with no declared difference at all.
    """

    left, right = (PANEL[name] for name in pair)
    held = ("modality", "n_layer", "d_model", "tokenisation", "architecture")
    differing = [key for key in held if getattr(left, key) != getattr(right, key)]
    if differing:
        raise AssertionError(f"corpus contrast {pair} does not hold {differing} fixed")
    if PRETRAINING_UNDECLARED in (left.pretraining_corpus, right.pretraining_corpus):
        raise AssertionError(
            f"corpus contrast {pair} has an undeclared pretraining corpus, so the "
            "quantity it varies is not recorded"
        )
    if left.pretraining_corpus == right.pretraining_corpus:
        raise AssertionError(
            f"corpus contrast {pair} declares one pretraining corpus "
            f"{left.pretraining_corpus!r} for both arms, so it varies nothing"
        )


for _pair in (MATCHED_DATA_CONTRAST, TEXT_DATA_CONTRAST):
    _check_corpus_contrast(_pair)
del _pair


def _check_architecture_contrast() -> None:
    """The arms that answer "is this just a GPT-2 property?" must not be GPT-2.

    Checked rather than commented for the same reason as the corpus contrasts,
    and with more at stake: this is the declaration behind the audit's §5.05(d)
    argument that the pathway-budget separation "survives replacing GPT-2-large
    with a Qwen2 and a Llama decoder", and it is what killed the QK/OV finding
    when it turned out to be a GPT-2-lineage property. An edit that left a
    GPT-2-architecture arm in this tuple would leave that argument stated and
    unsupported, with nothing raising.
    """

    for name in TEXT_ARCHITECTURE_CONTRAST:
        spec = PANEL[name]
        if spec.modality != "text":
            raise AssertionError(
                f"{name} is in TEXT_ARCHITECTURE_CONTRAST but is not a text arm"
            )
        if spec.architecture == "gpt2":
            raise AssertionError(
                f"{name} declares the gpt2 architecture, so it cannot witness that a "
                "finding survives leaving the GPT-2 lineage"
            )


_check_architecture_contrast()


@dataclass
class Arm:
    """A loaded panel member, with the contracts a measurement can rely on.

    ``attn_implementation`` is the attention kernel the checkpoint was actually
    loaded with, read back from the built model rather than from the request, so
    that a build which ignored or overrode the request is visible.
    """

    spec: ArmSpec
    model: torch.nn.Module
    tokenizer: object
    device: str
    dtype: str
    attn_implementation: str | None = None

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def modality(self) -> str:
        return self.spec.modality

    @property
    def n_layer(self) -> int:
        return self.spec.n_layer

    @property
    def d_model(self) -> int:
        return self.spec.d_model

    def supports(self, capability: str) -> bool:
        if capability not in CAPABILITIES:
            raise ValueError(f"unknown capability {capability!r}; known: {sorted(CAPABILITIES)}")
        return capability in self.spec.capabilities

    def require(self, capability: str) -> None:
        """Refuse a measurement this arm cannot enter commensurably."""
        if not self.supports(capability):
            raise ValueError(
                f"{self.name} ({self.spec.architecture}) does not support the "
                f"{capability!r} measurement family; its declared capabilities are "
                f"{sorted(self.spec.capabilities)}. Producing a number here would "
                "not be commensurate with the rest of the panel."
            )

    def require_eager_attention(self, measurement: str) -> None:
        """Refuse a measurement that reads or overrides attention patterns on a
        non-eager kernel.

        The fused kernels never materialise the pattern, so they return ``None``
        for the weights and ignore a per-head additive mask. Every caller that
        needs a pattern already raises when it gets ``None`` back, but that check
        fires deep inside a run and only for the read path; an *override* -- a
        per-head knockout mask, a frozen pattern -- would be accepted by the
        fused kernel's signature and quietly not applied. The contract is
        therefore stated at the top of the measurement, against the
        implementation the model was actually built with.
        """

        if self.attn_implementation is None:
            raise ValueError(
                f"{self.name}: {measurement} needs a declared attention "
                "implementation; load the arm with attn_implementation='eager'"
            )
        if self.attn_implementation != "eager":
            raise ValueError(
                f"{self.name}: {measurement} reads or overrides attention patterns, "
                f"which the {self.attn_implementation!r} kernel does not materialise; "
                "load the arm with attn_implementation='eager'"
            )

    def blocks(self) -> torch.nn.ModuleList:
        """The transformer block list, resolved per architecture.

        Resolution is explicit rather than duck-typed: a panel member whose
        block list cannot be named is a panel change that must be declared, not
        guessed at.
        """
        architecture = self.spec.architecture
        if architecture in ("gpt2", "progen"):
            if not hasattr(self.model, "transformer") or not hasattr(self.model.transformer, "h"):
                raise TypeError(f"{self.name}: declared {architecture} but no transformer.h")
            return self.model.transformer.h
        if architecture == "t5_decoder":
            if not hasattr(self.model, "decoder") or not hasattr(self.model.decoder, "block"):
                raise TypeError(f"{self.name}: declared t5_decoder but no decoder.block")
            return self.model.decoder.block
        if architecture in _ROTARY_DECODERS:
            # The causal-LM wrapper holds the bare decoder at ``.model``, one
            # level below where GPT-2 keeps ``.transformer``, and the block list
            # at ``.layers`` rather than ``.h``.
            inner = getattr(self.model, "model", None)
            if inner is None or not hasattr(inner, "layers"):
                raise TypeError(f"{self.name}: declared {architecture} but no model.layers")
            return inner.layers
        if architecture == "reformer":
            return self.model.reformer.encoder.layers
        raise TypeError(f"{self.name}: unsupported architecture {architecture!r}")

    def mlp(self, layer: int) -> torch.nn.Module:
        self.require("pathway")
        if self.spec.architecture not in _DECOMPOSABLE:
            raise TypeError(
                f"{self.name}: sublayer decomposition is not defined for "
                f"{self.spec.architecture!r}"
            )
        return self.blocks()[layer].mlp

    def _resolve_d_mlp(self) -> tuple[int, str]:
        """``d_mlp`` and the declaration it came from."""

        declared = mlp_neuron_declaration(self.spec.architecture)
        for attribute in declared.width_attributes:
            value = getattr(self.model.config, attribute, None)
            if value is not None:
                return int(value), f"config.{attribute}"
        if declared.implicit_width_multiple is None:
            raise TypeError(
                f"{self.name}: none of {list(declared.width_attributes)} is set on the "
                f"config and {self.spec.architecture!r} declares no implicit width, so "
                "the MLP hidden width is unknown"
            )
        multiple = int(declared.implicit_width_multiple)
        return multiple * int(self.d_model), (
            f"{multiple} x d_model ({'/'.join(declared.width_attributes)} unset)"
        )

    @property
    def d_mlp(self) -> int:
        """Width of the MLP hidden layer: the number of neurons in one block."""

        return self._resolve_d_mlp()[0]

    def mlp_down_projection(self, layer: int) -> torch.nn.Module:
        """The module whose **input** is this layer's MLP hidden activation.

        The neuron tensor is that input, and hooking the projection rather than
        the nonlinearity is what makes it definitionally so. Resolved by walking
        :attr:`MlpNeuronTensor.down_projection` from :meth:`mlp`, so an
        architecture nobody declared raises instead of being searched.
        """

        self.require("pathway")
        declared = mlp_neuron_declaration(self.spec.architecture)
        module: object = self.mlp(layer)
        for step in declared.down_projection:
            if not hasattr(module, step):
                raise TypeError(
                    f"{self.name}: declared {self.spec.architecture} but its layer-"
                    f"{layer} MLP has no {step} on the path {declared.down_projection}"
                )
            module = getattr(module, step)
        return module  # type: ignore[return-value]

    def mlp_neuron_facts(self) -> dict[str, object]:
        """What the artefact must carry about the tensor a neuron basis reads.

        Resolved once, from the declaration and the loaded config, so that the
        stage records the path it actually hooked and the width it expects rather
        than a sentence about them. An activation the declaration does not name
        raises here: its output bound is unknown, so the check that the hooked
        tensor is post-nonlinearity could not be applied to it.
        """

        declared = mlp_neuron_declaration(self.spec.architecture)
        width, source = self._resolve_d_mlp()
        activation = getattr(self.model.config, declared.activation_attribute, None)
        if activation not in declared.activation_lower_bound:
            raise TypeError(
                f"{self.name}: {declared.activation_attribute}={activation!r} is not a "
                f"declared nonlinearity for {self.spec.architecture!r} "
                f"({sorted(declared.activation_lower_bound)}); its output bound is "
                "unknown, so the hooked tensor cannot be verified as post-nonlinearity"
            )
        return {
            "architecture": self.spec.architecture,
            "tensor": "input of " + ".".join(("mlp", *declared.down_projection)),
            "declared_width": int(width),
            "width_source": source,
            "d_model": int(self.d_model),
            "activation": str(activation),
            "activation_lower_bound": float(declared.activation_lower_bound[activation]),
        }

    def _resolve_attention(self, layer: int) -> torch.nn.Module:
        """Walk :data:`_ATTENTION_PATH` from the block to the attention submodule.

        Searching a block for the first plausible attribute would resolve a newly
        admitted architecture silently, which is the one failure this panel
        cannot afford: an arm that reaches a measurement through an attribute
        nobody declared produces a number that looks like every other number in
        the table. One walker for both accessors, so the two cannot disagree
        about which module an arm's attention *is* while disagreeing, as they
        must, about what may be measured on it.
        """

        architecture = self.spec.architecture
        path = _ATTENTION_PATH.get(architecture)
        if path is None:
            raise TypeError(
                f"{self.name}: no attention submodule is declared for {architecture!r}; "
                f"declared: {sorted(_ATTENTION_PATH)}"
            )
        module: object = self.blocks()[layer]
        for step in path:
            if isinstance(step, int):
                try:
                    module = module[step]  # type: ignore[index]
                except (IndexError, TypeError) as error:
                    raise TypeError(
                        f"{self.name}: declared {architecture} but block {layer} has no "
                        f"entry {step} on the path {path}"
                    ) from error
                continue
            if not hasattr(module, step):
                raise TypeError(
                    f"{self.name}: declared {architecture} but block {layer} has no "
                    f"{step} on the path {path}"
                )
            module = getattr(module, step)
        return module  # type: ignore[return-value]

    def attention_pattern_module(self, layer: int) -> torch.nn.Module:
        """The attention submodule whose forward computes the pattern.

        Reading or overriding an attention pattern needs the module and nothing
        else. It does *not* need the block's sublayers to be commensurate with a
        standard causal decoder, which is what :meth:`attention` additionally
        asserts, and conflating the two kept ByGPT5 out of the prediction-addressed
        census on a decomposition ground the census never relies on.

        Gated on ``circuits`` rather than ``pathway``: everything that reads a
        pattern through this accessor computes a *per-head* statistic, which is
        the family ``circuits`` declares, while ``pathway`` declares that whole
        sublayer outputs are commensurate -- a claim this accessor's callers
        neither make nor need.
        """

        self.require("circuits")
        return self._resolve_attention(layer)

    def attention(self, layer: int) -> torch.nn.Module:
        """The attention submodule, as one term of a commensurate sublayer split.

        Same module as :meth:`attention_pattern_module`, stronger claim: callers
        of this accessor read or ablate it *as the attention sublayer* of a
        decomposition whose other term is the MLP, so an architecture whose
        sublayers are not those objects is refused even though its attention
        module can be named.
        """
        self.require("pathway")
        architecture = self.spec.architecture
        if architecture not in _DECOMPOSABLE:
            raise TypeError(
                f"{self.name}: sublayer decomposition is not defined for {architecture!r}"
            )
        return self._resolve_attention(layer)


def load_arm(
    name: str,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    attn_implementation: str | None = None,
) -> Arm:
    """Load a panel member and verify its declared shape and inference dtype."""
    if name not in PANEL:
        raise KeyError(f"unknown arm {name!r}; panel is {sorted(PANEL)}")
    if dtype not in _DTYPES:
        raise ValueError(f"unsupported inference dtype {dtype!r}")
    spec = PANEL[name]
    path = str(require_input_path(spec.path, _MODEL_PATH_VARIABLES))

    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    n_layer = getattr(config, "n_layer", None) or getattr(config, "num_hidden_layers")
    d_model = (
        getattr(config, "n_embd", None)
        or getattr(config, "hidden_size", None)
        or getattr(config, "embed_dim")
    )
    if (n_layer, d_model) != (spec.n_layer, spec.d_model):
        raise ValueError(
            f"{name}: declared {spec.n_layer}L/{spec.d_model}d, loaded {n_layer}L/{d_model}d"
        )

    extra: dict[str, object] = {}
    if attn_implementation is not None:
        # sdpa returns None for attention weights and cannot be intercepted;
        # anything that reads or overrides patterns must ask for eager.
        extra["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(
        path,
        # ``torch_dtype`` rather than ``dtype``: the H200 pod runs transformers
        # 4.52.4, where ``dtype`` is not a recognised loading argument and would
        # be swallowed as a config keyword, leaving a float32 model. ``dtype``
        # is the newer spelling and 4.57.3 warns that ``torch_dtype`` is
        # deprecated, but it is the only spelling both versions honour, and the
        # observed-dtype check below is what actually enforces the outcome.
        torch_dtype=_DTYPES[dtype],
        trust_remote_code=True,
        device_map={"": device},
        **extra,
    )
    model.eval()

    observed = sorted(
        {str(p.dtype).removeprefix("torch.") for p in model.parameters() if p.is_floating_point()}
    )
    if observed != [dtype]:
        raise ValueError(f"{name}: declared dtype {dtype}, observed {observed}")

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Read back rather than echo the request. A remote-code architecture that
    # never consults ``attn_implementation`` would otherwise be recorded as
    # eager on the strength of having been asked, and ``require_eager_attention``
    # would then vouch for a contract nothing enforced.
    resolved = getattr(model.config, "_attn_implementation", None)
    return Arm(
        spec=spec,
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        attn_implementation=None if resolved is None else str(resolved),
    )


# ------------------------------------------------------------------- cohorts


#: The FASTA corpora this package reads, and the variable that relocates each.
#: :func:`iter_fasta` is the single door to both, and several callers reach it
#: without going through :func:`protein_cohort`, so the existence check belongs
#: here rather than at each call site.
_FASTA_VARIABLE: dict[Path, str] = {
    SWISSPROT_FASTA: "TRANSFER_SWISSPROT_FASTA",
    ZYMCTRL_FASTA: "TRANSFER_ZYMCTRL_FASTA",
}


def iter_fasta(path: Path):
    path = Path(path)
    require_input_path(path, _FASTA_VARIABLE.get(path, "the TRANSFER_* variable naming it"))
    opener = gzip.open if str(path).endswith(".gz") else open
    header, chunks = None, []
    with opener(path, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header, chunks = line[1:].strip(), []
            else:
                chunks.append(line.strip())
    if header is not None:
        yield header, "".join(chunks)


def sampling_record(
    *,
    seed: int | None,
    skip: int,
    requested: int,
    eligible: int | None,
    corpus: str,
) -> dict:
    """How a cohort was drawn, as a value that travels with the cohort.

    Recorded on every cohort this module builds. Which records were drawn is the
    single most expensive thing this programme has got wrong -- three separate
    incidents, one worth 1.01 nats -- and every one of them was invisible
    afterwards because the artefact said what the numbers were and not where the
    records came from. ``mode`` is therefore mandatory in the record, and the
    file-order mode carries its own hazard text.
    """

    if requested < 1:
        raise ValueError("a cohort must request at least one record")
    if skip < 0:
        raise ValueError("skip must be non-negative")
    mode = "file_order" if seed is None else "seeded_permutation"
    if mode not in SAMPLING_MODES:  # pragma: no cover - guards a future third mode
        raise AssertionError(
            f"sampling mode {mode!r} is not in SAMPLING_MODES {SAMPLING_MODES}; a "
            "mode that is not declared has no recorded hazard text and no reader "
            "knows how to interpret it"
        )
    record: dict = {
        "mode": mode,
        "seed": None if seed is None else int(seed),
        "skip": int(skip),
        "requested": int(requested),
        "corpus": corpus,
        "eligible_records": None if eligible is None else int(eligible),
    }
    if mode == "file_order":
        record["hazard"] = FILE_ORDER_HAZARD
    return record


@dataclass
class Cohort:
    """A frozen evaluation cohort, identified by the hash of its contents."""

    name: str
    kind: str
    records: list[str]
    min_symbols: int
    max_symbols: int
    metadata: dict = field(default_factory=dict)

    @property
    def digest(self) -> str:
        """Content hash: the records themselves, and nothing else.

        Deliberately unchanged, so that a digest quoted in a frozen artefact
        still identifies the same content. It does *not* separate two cohorts
        that hold the same records under different metadata -- an exact-repeat
        and an approximate-repeat cohort can coincide on records while being
        different measurements -- which is what :attr:`provenance_digest` is for.
        """

        payload = json.dumps(
            {
                "kind": self.kind,
                "min": self.min_symbols,
                "max": self.max_symbols,
                "records": self.records,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    @property
    def provenance_digest(self) -> str:
        """Content hash extended with everything that decided the content.

        Two cohorts agreeing on :attr:`digest` can still be different objects:
        the nested exact and approximate repeat cohorts can hold identical
        records under different criteria, an EC-labelled draw carries labels a
        plain draw does not, and a seeded draw and a file-order draw that happen
        to coincide are not the same evidence. This digest separates them.
        Metadata that is not JSON-serialisable is represented by its repr rather
        than dropped, so nothing silently falls out of the hash.
        """

        def canonical(value: object) -> object:
            try:
                json.dumps(value)
            except TypeError:
                return repr(value)
            return value

        payload = json.dumps(
            {
                "content": self.digest,
                "metadata": {key: canonical(value) for key, value in sorted(self.metadata.items())},
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    @property
    def sampling(self) -> dict:
        """The draw that produced this cohort, or an explicit "not recorded"."""

        recorded = self.metadata.get("sampling")
        if isinstance(recorded, dict):
            return dict(recorded)
        return {
            "mode": "unrecorded",
            "hazard": (
                "this cohort was constructed without a sampling record, so whether "
                "its records are a seeded sample or the head of a file is not "
                "knowable from the artefact"
            ),
        }

    def __len__(self) -> int:
        return len(self.records)

    def input_strings(self, arm: Arm) -> list[str]:
        """Render the cohort in the arm's native input format."""
        if self.kind == "text":
            if arm.modality != "text":
                raise ValueError(f"{arm.name}: text cohort given to a protein arm")
            return list(self.records)
        if arm.modality != "protein":
            raise ValueError(f"{arm.name}: protein cohort given to a text arm")
        fmt = arm.spec.input_format
        if fmt == "raw":
            return list(self.records)
        if fmt == "fasta_wrapped":
            # ProtGPT2 was pretrained on FASTA-formatted UniRef50: sequences are
            # hard-wrapped at 60 residues and separated by the end-of-text token,
            # and its BPE merges were learned over exactly that byte stream.
            # Feeding one unwrapped line is off-distribution and costs 1.42
            # nats/token, measured on 80 Swiss-Prot sequences of 600-2000
            # residues: 8.046 raw versus 6.652 wrapped versus 6.623 with the
            # end-of-text prefix. Getting this wrong makes the model look
            # untrained and drove a spurious modality effect.
            eot = arm.tokenizer.eos_token
            if eot is None:
                raise ValueError(f"{arm.name}: tokenizer has no end-of-text token")
            return [
                eot + "\n" + "\n".join(s[i : i + 60] for i in range(0, len(s), 60))
                for s in self.records
            ]
        if fmt == "n_to_c_control":
            return ["1" + s for s in self.records]
        if fmt == "ec_conditioned":
            labels = self.metadata.get("ec_labels")
            if labels is None or len(labels) != len(self.records):
                raise ValueError(
                    f"{arm.name} requires EC labels; cohort {self.name!r} has none"
                )
            return [
                f"{ec}<sep>{CONDITIONING_START}{seq}{CONDITIONING_END}"
                for ec, seq in zip(labels, self.records)
            ]
        raise ValueError(f"unsupported input format {fmt!r}")


#: The markers ``Cohort.input_strings`` wraps a conditioned arm's content in.
#: Declared beside the rendering that emits them: a measurement module that needs
#: to *find* them must resolve these, not a second pair spelled by hand. Three
#: modules used to spell them independently, and the third did not check the
#: tokenizer's unknown-token id, so a tokenizer without them would have returned
#: a valid-looking id for a token it does not have.
CONDITIONING_START = "<start>"
CONDITIONING_END = "<end>"


def conditioning_boundary_ids(
    arm: Arm, *, ec_conditioning: str = "native"
) -> tuple[int | None, int | None]:
    """Token ids delimiting the scored content of a conditioned rendering.

    ``(None, None)`` when the rendering carries no conditioning prompt -- either
    the arm's input format has none, or ``ec_conditioning="unconditioned"``
    deliberately removed it -- so the result can be passed straight to
    :func:`src.transfer.scoring.sequence_target_mask` beside the rule that
    :func:`src.transfer.scoring.target_rule` selects for the same two inputs.

    Raises rather than returning a plausible id when the tokenizer does not carry
    the markers. The conditioning prompt is the span a measurement must *not*
    score -- EXP-R2-034 prices ZymCTRL's EC tag at 1.73 nats of leak -- so a
    silently wrong boundary id lands directly on the quantity being measured.
    """

    if arm.spec.input_format != "ec_conditioned" or ec_conditioning == "unconditioned":
        return None, None
    unknown = arm.tokenizer.unk_token_id
    ids: list[int] = []
    for token in (CONDITIONING_START, CONDITIONING_END):
        resolved = arm.tokenizer.convert_tokens_to_ids(token)
        if resolved is None or resolved == unknown:
            raise ValueError(
                f"{arm.name}: tokenizer has no {token!r} id, but its input format is "
                "ec_conditioned, so the span that must not be scored cannot be located"
            )
        ids.append(int(resolved))
    return ids[0], ids[1]


def selected_positions(
    eligible: int, *, n: int, skip: int, seed: int | None, label: str
) -> list[int]:
    """Which eligible records a draw selects, given a mode.

    ``seed is None`` reproduces the historical file-order draw exactly:
    positions ``skip .. skip + n``. With a seed the corpus is permuted first and
    the same half-open window is taken from the permutation, which makes two
    draws at the same seed and different ``skip`` genuinely disjoint -- the
    skip-offset sensitivity Appendix B rule 1 asks for is only a sensitivity if
    the offsets do not overlap.

    Returned in ascending corpus order so that a second pass over the corpus can
    collect them in one sweep; the *identity* of the selected set is what the
    seed decides, not the order they end up in.
    """

    if n < 1:
        raise ValueError("a cohort must request at least one record")
    if skip < 0:
        raise ValueError("skip must be non-negative")
    if eligible < skip + n:
        raise RuntimeError(
            f"cohort {label!r}: {eligible} eligible records cannot supply {n} "
            f"after a skip of {skip}"
        )
    if seed is None:
        return list(range(skip, skip + n))
    import numpy as np

    order = np.random.default_rng(seed).permutation(eligible)
    return sorted(int(index) for index in order[skip : skip + n])


def _eligible_protein_records(min_len: int, max_len: int, *, with_ec: bool):
    """Every corpus entry passing the length and alphabet filter, in file order.

    One generator serves both the counting pass and the collecting pass, so the
    two cannot disagree about what "eligible" means -- which is the way a
    two-pass sampler silently selects the wrong records.
    """

    allowed = set(AA20)
    if with_ec:
        for _, body in iter_fasta(ZYMCTRL_FASTA):
            if "<start>" not in body or "<end>" not in body:
                continue
            sequence = body.split("<start>")[1].split("<end>")[0]
            if not (min_len <= len(sequence) <= max_len) or not set(sequence) <= allowed:
                continue
            yield sequence, body.split("<sep>")[0]
    else:
        for _, sequence in iter_fasta(SWISSPROT_FASTA):
            if not (min_len <= len(sequence) <= max_len) or not set(sequence) <= allowed:
                continue
            yield sequence, None


def _eligible_text_documents(min_chars: int):
    """Every screening-subset document at or above ``min_chars``, in shard order."""

    import pyarrow.parquet as pq

    require_input_path(OPENWEBTEXT, "TRANSFER_OPENWEBTEXT_DIR")
    shards = sorted(OPENWEBTEXT.glob("*.parquet"))
    if not shards:
        raise RuntimeError(f"no parquet shards under {OPENWEBTEXT}")
    for shard in shards:
        for value in pq.read_table(shard, columns=["text"]).column("text"):
            document = value.as_py()
            if document is None or len(document) < min_chars:
                continue
            yield document


#: The corpus each declared source name streams from, and the variable that
#: relocates it. Read by :func:`iter_corpus_records`, which serves the stages
#: that need more records than a frozen cohort holds -- a transcoder trainer
#: consumes millions, which the cohort constructors cannot supply because they
#: count the whole corpus before selecting.
#:
#: The names are the ones :attr:`ArmSpec.evaluation_cohort_source` already uses,
#: plus ``uniref50``, which no arm evaluates on and ProGen3's transcoders are
#: trained on.
CORPUS_SOURCES: dict[str, tuple[Path, str]] = {
    "uniref50": (UNIREF50_FASTA, "TRANSFER_UNIREF50_FASTA"),
    "swissprot": (SWISSPROT_FASTA, "TRANSFER_SWISSPROT_FASTA"),
    "zymctrl_ec": (ZYMCTRL_FASTA, "TRANSFER_ZYMCTRL_FASTA"),
    "openwebtext": (OPENWEBTEXT, "TRANSFER_OPENWEBTEXT_DIR"),
}


def corpus_location(source: str, *, path: Path | None = None) -> Path:
    """Where one corpus source lives, checked to exist before anything loads a model.

    ``path`` overrides the declared location for ``uniref50`` only, which is the
    one source a stage flag has ever relocated. The others refuse an override
    rather than accept and ignore one: ``swissprot`` is read through the same
    eligibility generator :func:`protein_cohort` uses and ``openwebtext`` through
    the parquet reader, and neither consults a caller's path -- so an accepted
    override would silently stream the declared corpus under the name of another.
    """

    if source not in CORPUS_SOURCES:
        raise KeyError(f"unknown corpus source {source!r}; declared: {sorted(CORPUS_SOURCES)}")
    declared, variable = CORPUS_SOURCES[source]
    if path is None:
        return require_input_path(declared, variable)
    if source != "uniref50":
        raise ValueError(
            f"the {source!r} stream is read through this module's own reader and "
            f"cannot be relocated by a caller's path; set {variable} instead"
        )
    return require_input_path(Path(path), variable)


def iter_corpus_records(
    source: str,
    *,
    min_symbols: int,
    max_symbols: int | None = None,
    path: Path | None = None,
) -> Iterator[tuple[str, str | None]]:
    """Every eligible record of one corpus, in file order, in its own symbol unit.

    Symbols are residues for a protein corpus and characters for a text one, so
    one band argument means the thing the corpus is made of rather than a token
    count that would differ between arms (Appendix B rule 21's shape, one level
    down: a band declared in tokens is not the same band on two tokenisers).

    Each record is yielded as ``(record, conditioning_label)``. The label is
    ``None`` for every corpus whose arms take an unconditional rendering and is
    the EC number for ``zymctrl_ec``, because :meth:`Cohort.input_strings` cannot
    render an ``ec_conditioned`` arm without it. Carrying the pair rather than
    the bare record is what keeps the conditioned arm on the panel's one
    rendering declaration instead of a second copy inside a stage (Appendix B
    rule 12).

    **Eligibility is not uniform across the four, and the difference is
    deliberate.** ``swissprot`` and ``zymctrl_ec`` apply the canonical-alphabet
    filter :func:`protein_cohort` applies -- and ``zymctrl_ec`` additionally
    requires the ``<start>``/``<end>`` markers its records carry -- so a stage
    streaming either and a stage drawing a cohort from it see one population.
    ``uniref50`` applies the length band only, because that is what the ProGen3
    transcoder campaign streamed and its published runs must stay reproducible.
    ``openwebtext`` takes a floor and no ceiling, because the text path truncates
    at tokenisation rather than discarding long documents -- which is what
    :func:`text_cohort` does, and a ceiling here would select a different
    population from the cohort a replacement is later scored on.

    File order is the *stream* order, not a sampling decision: the callers shuffle
    it. The hazard Appendix B rule 1 names is theirs to answer, and
    ``17_train_transcoder.py`` answers it with a seeded block shuffle whose block
    size it records.
    """

    location = corpus_location(source, path=path)
    if source == "openwebtext":
        if max_symbols is not None:
            raise ValueError(
                "the openwebtext stream takes no upper bound: a text record is "
                "truncated at tokenisation rather than filtered out, and a "
                "character ceiling here would select a different population from "
                "the one text_cohort draws"
            )
        return ((document, None) for document in _eligible_text_documents(min_symbols))
    if max_symbols is None:
        raise ValueError(f"the {source!r} stream needs an upper residue bound")
    if source in ("swissprot", "zymctrl_ec"):
        return _eligible_protein_records(
            min_symbols, max_symbols, with_ec=source == "zymctrl_ec"
        )
    return (
        (sequence, None)
        for _, sequence in iter_fasta(location)
        if min_symbols <= len(sequence) <= max_symbols
    )


def protein_cohort(
    n: int,
    min_len: int,
    max_len: int,
    *,
    skip: int = 0,
    name: str = "swissprot",
    with_ec: bool = False,
    seed: int | None = None,
) -> Cohort:
    """Canonical-alphabet Swiss-Prot sequences, drawn under a declared mode.

    ``with_ec`` draws from the EC-labelled source so that one cohort can serve
    both the unconditional arms and ZymCTRL, which needs its conditioning tag.

    ``seed`` selects the draw. **Pass one.** Swiss-Prot and the EC-labelled
    corpus are both grouped by family, so the first ``n`` eligible entries are a
    set of near-clonal homologues rather than a sample: they are unusually
    predictable, which shrinks the context information every share is divided by,
    and reading past them has moved a headline figure by 1.01 nats. ``seed=None``
    keeps the historical file-order draw, because several frozen artefacts were
    produced with it and must stay reproducible; it is recorded as
    ``sampling.mode == "file_order"`` with its hazard attached, so no artefact
    built on it can be read without seeing which draw produced it.

    Under a seed the whole corpus is counted first and the draw is a window of a
    seeded permutation, so ``skip`` produces a genuinely disjoint second sample
    of the same corpus rather than a different prefix of the same file.
    """

    corpus = "ec_labelled_swissprot" if with_ec else "plain_swissprot"
    if seed is None:
        records: list[str] = []
        labels: list[str] = []
        eligible: int | None = None
        seen = 0
        for sequence, label in _eligible_protein_records(min_len, max_len, with_ec=with_ec):
            seen += 1
            if seen <= skip:
                continue
            records.append(sequence)
            if label is not None:
                labels.append(label)
            if len(records) >= n:
                break
        if len(records) < n:
            raise RuntimeError(f"cohort {name!r}: only {len(records)}/{n} eligible sequences")
    else:
        eligible = sum(
            1 for _ in _eligible_protein_records(min_len, max_len, with_ec=with_ec)
        )
        wanted = set(selected_positions(eligible, n=n, skip=skip, seed=seed, label=name))
        records = []
        labels = []
        for position, (sequence, label) in enumerate(
            _eligible_protein_records(min_len, max_len, with_ec=with_ec)
        ):
            if position not in wanted:
                continue
            records.append(sequence)
            if label is not None:
                labels.append(label)
        if len(records) != n:
            raise RuntimeError(
                f"cohort {name!r}: the corpus changed between the counting and the "
                f"collecting pass ({len(records)} of {n} selected records found)"
            )
    metadata: dict = {
        "sampling": sampling_record(
            seed=seed, skip=skip, requested=n, eligible=eligible, corpus=corpus
        )
    }
    if with_ec:
        metadata["ec_labels"] = labels
    return Cohort(name, "protein", records, min_len, max_len, metadata)


def text_cohort(
    n: int,
    min_chars: int = 800,
    *,
    skip: int = 0,
    name: str = "openwebtext",
    seed: int | None = None,
) -> Cohort:
    """Documents from the frozen OpenWebText screening subset.

    ``seed`` has the same meaning as in :func:`protein_cohort`. The text control
    is drawn the same way as the protein cohorts on purpose: a control drawn
    under a different sampling rule from the arm it controls is not a control.
    """

    if seed is None:
        records: list[str] = []
        eligible: int | None = None
        seen = 0
        for document in _eligible_text_documents(min_chars):
            seen += 1
            if seen <= skip:
                continue
            records.append(document)
            if len(records) >= n:
                break
        if len(records) < n:
            raise RuntimeError(f"cohort {name!r}: only {len(records)}/{n} documents")
    else:
        eligible = sum(1 for _ in _eligible_text_documents(min_chars))
        wanted = set(selected_positions(eligible, n=n, skip=skip, seed=seed, label=name))
        records = [
            document
            for position, document in enumerate(_eligible_text_documents(min_chars))
            if position in wanted
        ]
        if len(records) != n:
            raise RuntimeError(
                f"cohort {name!r}: the corpus changed between the counting and the "
                f"collecting pass ({len(records)} of {n} selected documents found)"
            )
    metadata = {
        "sampling": sampling_record(
            seed=seed,
            skip=skip,
            requested=n,
            eligible=eligible,
            corpus="openwebtext_screen",
        )
    }
    return Cohort(name, "text", records, min_chars, 0, metadata)


def tokenize_batch(
    arm: Arm, texts: list[str], max_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-padded ids and a validity mask, truncated to ``max_len``."""
    if not texts:
        raise ValueError(f"{arm.name}: cannot tokenise an empty batch")
    if max_len < 1:
        raise ValueError("max_len must be positive")
    rows = [arm.tokenizer(t, return_tensors=None)["input_ids"][:max_len] for t in texts]
    empty = [index for index, row in enumerate(rows) if not row]
    if empty:
        # A zero-token row would contribute a fully masked line to the batch and
        # therefore to every mean computed over it, without appearing anywhere
        # as a dropped record.
        raise ValueError(f"{arm.name}: rows {empty} of the batch tokenise to no tokens")
    width = max(len(r) for r in rows)
    pad = arm.tokenizer.pad_token_id
    if pad is None:
        raise ValueError(f"{arm.name}: tokenizer has no pad token")
    ids = torch.full((len(rows), width), pad, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, row in enumerate(rows):
        ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
        mask[i, : len(row)] = 1
    return ids, mask


def symbols_per_token(arm: Arm, texts: list[str], max_len: int) -> float:
    """Measured tokenizer expansion over exactly the scored window.

    Protein arms count residues; the text arm counts characters. Counting before
    truncation inflates the ratio for long sequences, so both are counted after.
    """
    tokens = 0
    symbols = 0
    for text in texts:
        ids = arm.tokenizer(text, return_tensors=None)["input_ids"][:max_len]
        decoded = arm.tokenizer.decode(ids)
        tokens += len(ids)
        symbols += (
            sum(1 for c in decoded if c in AA20) if arm.modality == "protein" else len(decoded)
        )
    if tokens == 0:
        raise RuntimeError(f"{arm.name}: empty cohort")
    return symbols / tokens
