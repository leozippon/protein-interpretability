"""Convergence control: is the protein interpretability deficit modality or maturity?

Every transfer result in this programme so far compares GPT-2-large against
ProtGPT2, ZymCTRL and ProGen2-medium. That panel is matched on architecture and
on nothing else. The protein decoders differ from the text decoder in
pretraining-token budget, corpus quality, tokenizer implementation and degree of
convergence, and they are scored on a cohort none of them was validated against.
So a protein arm that carries less measurable structure may be a *protein* arm,
or an *undertrained* arm, or an arm evaluated *off its own distribution*. Nothing
in the programme separates the three, which means the central claim is not yet
falsifiable.

This module builds the separation. Each model is placed on a convergence axis
measured on its own in-distribution cohort, the same interpretability metrics are
placed on the y-axis, and the question becomes quantitative: do the two
modalities lie on one curve, and is there a residual offset once convergence is
held fixed?

Four design commitments make this a control rather than a demonstration.

*The estimand and the verdict rule are declared before the data.*
:data:`PRIMARY_METRIC`, :data:`PRIMARY_AXIS`, :data:`DEFAULT_EQUIVALENCE_MARGIN`
and :data:`DEFAULT_MIN_RESIDUAL_DOF` fix what will be read and what will count as
an answer. :func:`decide_verdict` can return ``residual_modality_gap`` - which
supports the programme's hypothesis - only when the modality coefficient's
interval excludes zero, and can return ``gap_explained_by_convergence`` only when
that interval is tight enough to be informative. Everything else is
``underpowered``, which is the default.

*A coefficient is only read under the label it was fitted with.* Modality and
tokenisation are confounded across this panel and are separated by one rung,
ProtGPT2, which is protein and subword at once. :func:`identification_check` is a
precondition of the verdict, not a caveat on it: with that rung gone the fitted
indicator is a tokenisation indicator wearing a modality label, and neither
reading may be taken from it.

*The ladder is configuration, not code.* :data:`DEFAULT_LADDER` names the models
this control expects; :func:`parse_ladder_table` reads an operator-supplied table
instead. :func:`inspect_member` decides availability from what is on disk, and a
member that is not there is recorded as absent with a reason rather than dropped.

*Every arm is scored on its own native cohort.* Perplexity on a shared corpus is
not a shared quantity: it is each model's distance from a corpus none of them was
trained on, and the arm whose training distribution happens to sit furthest away
is penalised for that rather than for being less converged. So the cohort corpus
and the sequence-length band travel with the ladder member, and the
in-distribution rule is applied only after an arm has been given the distribution
it was trained for. This is not cosmetic: ProtGPT2 was pretrained on FASTA-style
UniRef50 and is off-distribution on a 64-246-residue cohort while being
in-distribution on a 300-1000-residue one, and since ProtGPT2 is the only rung
that is protein *and* subword, that single choice decides whether the modality
coefficient is identified at all. :func:`cohort_sensitivity_rows` therefore
measures every protein rung on every protein cohort in the run, so the length
dependence is evidence in the record rather than an assumption behind it.

*The convergence axis is tokenizer-independent where it has to be.* Realized
information fraction is a ratio of two quantities measured in the same token
alphabet, so it survives the comparison of a multi-residue-BPE arm with a
residue-level one; clean cross-entropy is reported per *symbol* for the same
reason. Per-token cross-entropy is not comparable across the ladder and is not
offered as an axis.

*An off-distribution model is excluded, loudly.* If a model's clean
cross-entropy on its own evaluation cohort does not beat that cohort's unigram
entropy, every normalised interpretability figure computed on it is a ratio with
a non-positive denominator. :func:`convergence_row` flags it and
:func:`analysis_frame` keeps it out of every fit while leaving it visible in the
table.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from tokenizers import Tokenizer
from transformers import AutoConfig, AutoTokenizer

from .arms import (
    CAPABILITIES,
    MODEL_ROOT,
    PANEL,
    PRETRAINING_UNDECLARED,
    STAGED_ARMS,
    TEXT_MODEL_BASE,
    ArmSpec,
)
from .budget import ratio_denominator_admissibility
from .circuits import _CIRCUIT_ARCHITECTURES
from .lenses import FINAL_LAYER_NORM_PATH
from .statistics import mean_interval

#: v3 (EXP-R2-218): the convergence row carries the denominator's own bootstrap
#: standard error and :func:`analysis_frame` decides admissibility from it, so a
#: v2 record -- which carries a ``denominator_floor_nats`` in its rows and no
#: standard error anywhere -- is not readable under this schema.
SCHEMA_VERSION = "r2_transfer_convergence_control_v3"

#: Natural log of two, for the per-symbol conversions.
LN2 = math.log(2.0)

#: The interpretability metric and convergence axis whose fit decides the
#: verdict. Declared here so that the reading cannot be chosen after the numbers
#: are in; every other pair in :data:`INTERPRETABILITY_METRICS` x
#: :data:`CONVERGENCE_AXES` is reported as a secondary, non-deciding fit.
PRIMARY_METRIC = "mlp_share_of_context_information"
PRIMARY_AXIS = "realized_information_fraction"

#: Convergence / quality axes offered to the fits. All three are comparable
#: across tokenizers: a ratio of same-alphabet quantities, a per-symbol figure,
#: and a pure architecture count.
CONVERGENCE_AXES: tuple[str, ...] = (
    "realized_information_fraction",
    "clean_ce_bits_per_symbol",
    "log10_parameters",
)

#: Interpretability metrics offered to the fits. Each is normalised so that it
#: does not move mechanically with depth or head count, which vary across the
#: ladder: pathway shares divide by the cohort's own context information,
#: the induction figure divides by the probe's uniform-attention baseline, and
#: the attribution figures divide by the component count.
INTERPRETABILITY_METRICS: tuple[str, ...] = (
    "mlp_share_of_context_information",
    "attn_share_of_context_information",
    "induction_natural_max_prefix_matching_over_uniform",
    "induction_natural_fraction_of_heads_above_threshold",
    "induction_synthetic_max_prefix_matching_over_uniform",
    "induction_synthetic_fraction_of_heads_above_threshold",
    "dla_participation_fraction",
    "dla_mlp_magnitude_fraction",
    "aperture_gain_alignment_ratio",
)

#: Measurement family each interpretability metric belongs to. The panel now
#: spans architectures deliberately, and an arm that cannot enter a family must
#: not contribute a number to a fit over that family: ByGPT5 is T5-derived and
#: admits no GPT-2-style sublayer decomposition, so it has no MLP pathway share
#: and no induction census, and inventing one would be a category error rather
#: than a missing value. Each fit therefore runs on the rungs that declare the
#: capability the metric needs.
METRIC_CAPABILITY = {
    "mlp_share_of_context_information": "pathway",
    "attn_share_of_context_information": "pathway",
    "induction_natural_max_prefix_matching_over_uniform": "circuits",
    "induction_natural_fraction_of_heads_above_threshold": "circuits",
    "induction_synthetic_max_prefix_matching_over_uniform": "circuits",
    "induction_synthetic_fraction_of_heads_above_threshold": "circuits",
    "dla_participation_fraction": "circuits",
    "dla_mlp_magnitude_fraction": "circuits",
    "aperture_gain_alignment_ratio": "lens",
}

#: The induction probe whose census is the reported one. Natural repeats are real
#: repeated spans in real records, so they are in-distribution for every rung. The
#: synthetic probe is a random token block repeated in token space: it is
#: off-distribution for any protein decoder, and specifically so for ProtGPT2,
#: whose native rendering breaks a line every 60 residues while a 128-token
#: synthetic probe contains no break at all. Both are measured and fitted, because
#: whether the probe choice changes the answer is itself worth knowing, but the
#: natural variant is the one quoted.
PRIMARY_INDUCTION_PROBE = "natural"

#: Half-width the modality coefficient's interval must fall inside before a null
#: result may be read as "the gap is convergence". Expressed on the scale of
#: :data:`PRIMARY_METRIC`, a share of the cohort's context information: a
#: modality offset of ten points of context information would be material, so an
#: interval wider than that cannot distinguish "no offset" from "a large one".
#:
#: **Its attainability on this design was never checked, and it is not
#: attained.** This is the only route to ``gap_explained_by_convergence`` and it
#: requires *both* fits inside the margin; the widest half-width the shipped runs
#: produce is 0.26-0.34 against a margin of 0.10, so on the evidence to date the
#: verdict is unreachable and every ``underpowered`` reading here is a statement
#: about the ladder's width rather than about convergence. Evidence-discipline
#: rule 1 and Appendix B rule 2 say a gate whose positive case cannot be produced
#: is a specification defect, and the same shape has already cost this programme
#: L1 and L2. The margin is left where it is rather than widened to whatever the
#: data happen to reach -- moving a criterion to admit a result is the failure
#: this module warns about elsewhere -- and :func:`decide_verdict` now returns
#: the attainability record beside the verdict so the defect is visible in the
#: artefact instead of inferable from two numbers in it.
DEFAULT_EQUIVALENCE_MARGIN = 0.10

#: Residual degrees of freedom below which no fit is read at all. Two is the
#: smallest value at which the residual variance is estimated from more than a
#: single number; with a three-parameter model that requires five usable ladder
#: points.
DEFAULT_MIN_RESIDUAL_DOF = 2

#: Prefix-matching cut-off used for the induction census headline. One of
#: :data:`src.transfer.circuits.INDUCTION_THRESHOLDS`, fixed here so the ladder
#: is counted identically at every rung.
DEFAULT_INDUCTION_THRESHOLD = 0.10

VERDICTS = (
    "gap_explained_by_convergence",
    "residual_modality_gap",
    "underpowered",
)

#: Columns a ladder table must supply. Extra columns are ignored; a missing one
#: is a hard error, because guessing a model's modality or input format would
#: silently mis-render its evaluation cohort.
#:
#: ``source`` keeps its legacy spelling because operator-supplied tables already
#: use it; it is the *evaluation cohort* source and is read into
#: :attr:`LadderMember.evaluation_cohort_source`. The pretraining corpus is a
#: different fact and has its own optional column, :data:`PRETRAINING_COLUMN`.
LADDER_TABLE_COLUMNS: tuple[str, ...] = (
    "name",
    "path",
    "modality",
    "tokenisation",
    "input_format",
    "source",
    "cohort_corpus",
    "cohort_min_symbols",
    "cohort_max_symbols",
)

#: Optional ladder-table column carrying the corpus a checkpoint was *trained*
#: on. Optional rather than required so that an existing table still parses; a
#: table that omits it yields ``arms.PRETRAINING_UNDECLARED``, which is visible
#: in every row of the analysis frame. That is the whole point of the split: the
#: absence of the fact is now recorded as an absence instead of being read off a
#: field that says ``openwebtext`` and means something else.
PRETRAINING_COLUMN = "pretraining_corpus"

#: Corpora a cohort may be drawn from, named as
#: ``src.transfer.pathways.COHORT_SOURCES`` names them so that one vocabulary
#: describes a cohort everywhere it is recorded.
TEXT_CORPUS = "openwebtext_screen"
PROTEIN_CORPORA = ("ec_labelled_swissprot", "plain_swissprot")
COHORT_CORPORA = (*PROTEIN_CORPORA, TEXT_CORPUS)

MODALITIES = ("protein", "text")

#: Tokenisation collapsed to the distinction that matters for identification: a
#: learned multi-symbol vocabulary against one symbol per token. GPT-2's BPE and
#: ProtGPT2's multi-residue BPE are the same kind of object on this axis, and
#: ZymCTRL and the ProGen2 rungs are the other kind.
TOKENISATION_FAMILIES = {
    "bpe": "subword",
    "multi_residue_bpe": "subword",
    "residue": "symbol",
    "byte": "symbol",
}

#: Weight-file suffixes accepted as evidence that a model directory is complete.
#: A directory that holds a config but no weights is a download in flight, which
#: is exactly the state a size ladder is in while it is being assembled.
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt")

#: Head padding a checkpoint may legitimately carry above its tokenizer's
#: vocabulary. Rounding an output projection up to a multiple of 64 is ordinary
#: kernel alignment and the surplus columns are trained alongside the rest; a
#: surplus far larger than that is a different vocabulary bolted onto the same
#: tokenizer, and its columns were never meaningfully trained.
LOGIT_WIDTH_PADDING = 64

#: Input formats ``arms.Cohort.input_strings`` can render.
INPUT_FORMATS = ("raw", "fasta_wrapped", "n_to_c_control", "ec_conditioned")

#: Input formats ``src.transfer.circuits`` can score, mirroring the branches in
#: its ``content_bounds`` and ``prefix_ids``. Those two decide which token
#: positions are modality content and how a synthetic probe reproduces an arm's
#: prompt, and both raise on a format they do not know.
#:
#: This list is declared rather than derived, so it can fall behind ``circuits``.
#: The asymmetry is deliberate and safe in the direction it can fail: too narrow
#: costs a recorded, reasoned skip of one metric, while too wide surfaces as the
#: ValueError ``circuits`` itself raises. It is never silently wrong.
CIRCUITS_INPUT_FORMATS = ("raw", "fasta_wrapped", "n_to_c_control", "ec_conditioned")

#: Architectures ``src.transfer.lenses.lens_head`` can build a lens head for.
#:
#: **Read from that module rather than restated here.** This was a literal pair
#: for as long as the lens resolved one hard-coded ``transformer.ln_f``, and the
#: comment defended the duplication on the ground that it can only fail safely:
#: too narrow costs a recorded skip, too wide surfaces as the module's own
#: TypeError. That defence is no longer needed. The lens now declares where each
#: architecture keeps its final normalisation, in
#: :data:`src.transfer.lenses.FINAL_LAYER_NORM_PATH`, so the set it can serve is
#: a fact that module holds and this one reads. Neither direction of drift is
#: possible any more, and a new architecture is admitted by extending one table.
#:
#: A T5-derived decoder is still outside it even though the panel grants ByGPT5
#: the ``lens`` capability: the capability is an intent, this is what the module
#: delivers, and :func:`lens_supported` is where the two are compared.
LENS_ARCHITECTURES = tuple(sorted(FINAL_LAYER_NORM_PATH))

#: Capabilities the rotary text rungs carry.  Written out rather than imported
#: from :mod:`src.transfer.arms`, where it is private, because
#: :func:`resolve_member` refuses any ladder declaration that disagrees with the
#: frozen panel declaration: if the two ever drift apart the run stops with the
#: two dictionaries printed side by side, which is a better failure than an
#: import that silently follows a change nobody meant to make here.  The
#: difference from :data:`src.transfer.arms.CAPABILITIES` is ``relational``,
#: which the rotary arms do not carry.
ROTARY_TEXT_CAPABILITIES = frozenset({"budget", "lens", "pathway", "circuits"})

#: What a ByGPT5 rung may enter, and it must be the same frozenset
#: ``arms.PANEL`` declares -- :func:`register_arm_spec` refuses the run outright
#: when the ladder and the panel disagree about a member, which is the check that
#: caught this file when the panel granted ``circuits`` and this one had not.
#:
#: ``circuits`` here means what it means in ``arms.py``: per-head attention
#: statistics are readable, which is what the prediction-addressed census needs.
#: It does NOT mean ``src.transfer.circuits`` can resolve the rung --
#: :func:`circuits_supported` is the declaration that answers that, and it
#: refuses ``t5_decoder`` on the module's own architecture set, so this control's
#: induction and attribution axes stay recorded-as-skipped for these three rungs.
BYTE_TEXT_CAPABILITIES = frozenset({"budget", "lens", "circuits"})


@dataclass(frozen=True)
class LadderMember:
    """One configured rung of the size ladder.

    Carries only what cannot be read off a config file. Depth, width and
    vocabulary are discovered by :func:`inspect_member` so that the ladder can be
    extended without anyone hand-copying a shape and getting it wrong; modality,
    tokenisation and input format cannot be discovered and must be declared,
    because rendering a cohort in the wrong input format would put a model off
    its own distribution and the resulting deficit would be blamed on modality.

    The cohort corpus and length band are declared per member for the same
    reason. A model's pretraining distribution is a fact about the model, so it
    belongs beside the model, and a length band chosen for one arm is not a
    neutral choice for another: ProtGPT2 is off-distribution on short Swiss-Prot
    fragments and in-distribution on full-length proteins.
    """

    name: str
    path: Path
    modality: str
    tokenisation: str
    input_format: str
    evaluation_cohort_source: str
    cohort_corpus: str
    cohort_min_symbols: int
    cohort_max_symbols: int
    pretraining_corpus: str = PRETRAINING_UNDECLARED
    architecture: str = "gpt2"
    capabilities: frozenset[str] = CAPABILITIES

    @property
    def source(self) -> str:
        """Deprecated alias of :attr:`evaluation_cohort_source`.

        Retained for the ladder-table column of the same name and for artefact
        readers that already use this spelling. It is the corpus the rung is
        *scored* on, never the corpus it was trained on -- see
        :class:`src.transfer.arms.ArmSpec` for why the two were separated.
        """

        return self.evaluation_cohort_source

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a ladder member must be named")
        if self.modality not in MODALITIES:
            raise ValueError(f"{self.name}: unknown modality {self.modality!r}")
        if self.tokenisation not in TOKENISATION_FAMILIES:
            raise ValueError(f"{self.name}: unknown tokenisation {self.tokenisation!r}")
        if self.input_format not in INPUT_FORMATS:
            raise ValueError(f"{self.name}: unknown input format {self.input_format!r}")
        if not self.evaluation_cohort_source:
            raise ValueError(f"{self.name}: an evaluation-cohort source must be declared")
        if not self.pretraining_corpus:
            raise ValueError(
                f"{self.name}: pretraining_corpus must be a declared corpus or "
                f"{PRETRAINING_UNDECLARED!r}, never empty"
            )
        if self.cohort_corpus not in COHORT_CORPORA:
            raise ValueError(f"{self.name}: unknown cohort corpus {self.cohort_corpus!r}")
        if (self.modality == "text") != (self.cohort_corpus == TEXT_CORPUS):
            raise ValueError(
                f"{self.name}: modality {self.modality!r} does not match cohort corpus "
                f"{self.cohort_corpus!r}"
            )
        if self.input_format == "ec_conditioned" and self.cohort_corpus != "ec_labelled_swissprot":
            raise ValueError(
                f"{self.name}: an EC-conditioned arm needs the EC-labelled corpus, which "
                "is the only one carrying the conditioning labels its prompt requires"
            )
        if self.cohort_min_symbols < 1:
            raise ValueError(f"{self.name}: the cohort length band must start above zero")
        # ``arms.text_cohort`` selects on a minimum character count and imposes no
        # upper bound, so a text member declares zero and the zero is meaningful.
        if self.modality == "text":
            if self.cohort_max_symbols != 0:
                raise ValueError(
                    f"{self.name}: a text cohort has no upper length bound; declare zero"
                )
        elif self.cohort_max_symbols < self.cohort_min_symbols:
            raise ValueError(f"{self.name}: the cohort length band is empty")

    @property
    def cohort_key(self) -> tuple[str, int, int]:
        """Identity of the cohort this member needs, so members can share one pool.

        Two members with the same corpus and length band must be scored on the
        same sequences or their convergence axes are not comparable; keying the
        pool by that triple is what makes the sharing explicit instead of
        accidental.
        """

        return (self.cohort_corpus, self.cohort_min_symbols, self.cohort_max_symbols)


#: Residue band for the unconditional protein rungs. ProtGPT2 was pretrained on
#: FASTA-formatted UniRef50 and the ProGen2 family on full-length UniRef90 and
#: BFD, so full-length proteins are the native distribution for both and a short
#: band is not. The dependence is steep and was measured rather than assumed:
#: ProtGPT2's context information on plain Swiss-Prot runs -1.21 nats at 100-246
#: residues, +0.18 to +0.44 at 300-1000 and +0.98 at 600-2000, so only the last
#: of those clears ``src.transfer.budget``'s 0.30-nat measurability floor with any
#: margin. That matters beyond ProtGPT2 itself: it is the only rung that is
#: protein and subword at once, so if its denominator is unusable the modality
#: coefficient is not separable from a tokenisation coefficient at all. The band
#: is therefore chosen on measurability, before any interpretability metric is
#: read, and both unconditional protein rungs share it so that they share one
#: pool and one digest.
UNCONDITIONAL_PROTEIN_BAND = (600, 2000)

#: Residue band for the EC-conditioned rung. ZymCTRL's prompt carries an EC label
#: and its scored window must contain the closing ``<end>`` marker, so its band is
#: capped by the token budget rather than chosen; the difference from
#: :data:`UNCONDITIONAL_PROTEIN_BAND` is forced, not preferred, and is recorded
#: with every ZymCTRL row.
CONDITIONED_PROTEIN_BAND = (64, 246)

#: Minimum document length for the text rungs, in characters.
TEXT_MIN_CHARS = 800

#: The ladder this control expects. The four members that already exist in
#: :data:`src.transfer.arms.PANEL` are named exactly as the panel names them, so
#: that their declared shapes remain the panel's assertions rather than being
#: re-derived here.
#:
#: Checkpoint locations come from :data:`src.transfer.arms.MODEL_ROOT` and
#: :data:`src.transfer.arms.TEXT_MODEL_BASE` rather than being written out, so
#: that one host's mount points move the ladder and the panel together. Their
#: defaults are the local L20 paths, so an unset environment reproduces the
#: literals this table used to carry. A rung that is absent on a given host is
#: not an error here: :func:`inspect_member` records it as unavailable with the
#: reason, which is how a partially staged ladder stays auditable.
DEFAULT_LADDER: tuple[LadderMember, ...] = (
    LadderMember(
        name="gpt2",
        path=TEXT_MODEL_BASE / "gpt2",
        modality="text",
        tokenisation="bpe",
        input_format="raw",
        evaluation_cohort_source="openwebtext",
        pretraining_corpus="webtext",
        cohort_corpus=TEXT_CORPUS,
        cohort_min_symbols=TEXT_MIN_CHARS,
        cohort_max_symbols=0,
    ),
    LadderMember(
        name="gpt2-medium",
        path=TEXT_MODEL_BASE / "gpt2-medium",
        modality="text",
        tokenisation="bpe",
        input_format="raw",
        evaluation_cohort_source="openwebtext",
        pretraining_corpus="webtext",
        cohort_corpus=TEXT_CORPUS,
        cohort_min_symbols=TEXT_MIN_CHARS,
        cohort_max_symbols=0,
    ),
    LadderMember(
        name="gpt2-large",
        path=TEXT_MODEL_BASE / "gpt2-large",
        modality="text",
        tokenisation="bpe",
        input_format="raw",
        evaluation_cohort_source="openwebtext",
        pretraining_corpus="webtext",
        cohort_corpus=TEXT_CORPUS,
        cohort_min_symbols=TEXT_MIN_CHARS,
        cohort_max_symbols=0,
    ),
    LadderMember(
        name="gpt2-xl",
        path=TEXT_MODEL_BASE / "gpt2-xl",
        modality="text",
        tokenisation="bpe",
        input_format="raw",
        evaluation_cohort_source="openwebtext",
        pretraining_corpus="webtext",
        cohort_corpus=TEXT_CORPUS,
        cohort_min_symbols=TEXT_MIN_CHARS,
        cohort_max_symbols=0,
    ),
    # The three arms below break the text side's single lineage. Until they were
    # added, every text rung that could carry the ``circuits`` capability was a
    # GPT-2 checkpoint trained on WebText, so any fitted modality coefficient was
    # equally consistent with "GPT-2 has unusually many induction heads" -- the
    # mirror image of the n=1 protein-and-subword cell, and the objection this
    # control exists to answer. They are ordinary next-token pretrained
    # decoders, not post-trained ones: an SFT/DPO checkpoint would confound the
    # architecture and corpus contrast with a training-objective contrast that no
    # LadderMember field records, and would do so on the convergence axis' own
    # denominator, since post-training moves cross-entropy on raw web text.
    #
    # ``source`` and ``cohort_corpus`` stay ``openwebtext``: it is the evaluation
    # cohort, not a claim about pretraining data. None of these three was trained
    # on WebText, which is a limitation of the convergence axis for them and is
    # the same limitation gpt2-large's own rungs do not have. It is recorded here
    # rather than left to be inferred, and it cuts against the finding: an arm
    # scored off its own pretraining distribution is being measured at a
    # disadvantage, so any deficit these text arms show relative to the GPT-2
    # rungs is at least partly this.
    LadderMember(
        name="dialogpt-small",
        path=TEXT_MODEL_BASE / "DialoGPT-small",
        modality="text",
        tokenisation="bpe",
        input_format="raw",
        evaluation_cohort_source="openwebtext",
        pretraining_corpus="reddit_dialogue",
        cohort_corpus=TEXT_CORPUS,
        cohort_min_symbols=TEXT_MIN_CHARS,
        cohort_max_symbols=0,
    ),
    LadderMember(
        name="qwen2.5-0.5b",
        path=TEXT_MODEL_BASE / "Qwen2.5-0.5B",
        modality="text",
        tokenisation="bpe",
        input_format="raw",
        evaluation_cohort_source="openwebtext",
        pretraining_corpus="qwen2.5_pretraining_mixture",
        cohort_corpus=TEXT_CORPUS,
        cohort_min_symbols=TEXT_MIN_CHARS,
        cohort_max_symbols=0,
        architecture="qwen2",
        capabilities=ROTARY_TEXT_CAPABILITIES,
    ),
    LadderMember(
        name="llama-3.2-3b",
        path=TEXT_MODEL_BASE / "Llama-3.2-3B",
        modality="text",
        tokenisation="bpe",
        input_format="raw",
        evaluation_cohort_source="openwebtext",
        pretraining_corpus="llama3_web_corpus_with_llama3.1_logit_distillation",
        cohort_corpus=TEXT_CORPUS,
        cohort_min_symbols=TEXT_MIN_CHARS,
        cohort_max_symbols=0,
        architecture="llama",
        capabilities=ROTARY_TEXT_CAPABILITIES,
    ),
    LadderMember(
        name="bygpt5-small-en",
        path=TEXT_MODEL_BASE / "bygpt5-small-en",
        modality="text",
        tokenisation="byte",
        input_format="raw",
        evaluation_cohort_source="openwebtext",
        pretraining_corpus=PRETRAINING_UNDECLARED,
        cohort_corpus=TEXT_CORPUS,
        cohort_min_symbols=TEXT_MIN_CHARS,
        cohort_max_symbols=0,
        architecture="t5_decoder",
        capabilities=BYTE_TEXT_CAPABILITIES,
    ),
    LadderMember(
        name="bygpt5-base-en",
        path=TEXT_MODEL_BASE / "bygpt5-base-en",
        modality="text",
        tokenisation="byte",
        input_format="raw",
        evaluation_cohort_source="openwebtext",
        pretraining_corpus=PRETRAINING_UNDECLARED,
        cohort_corpus=TEXT_CORPUS,
        cohort_min_symbols=TEXT_MIN_CHARS,
        cohort_max_symbols=0,
        architecture="t5_decoder",
        capabilities=BYTE_TEXT_CAPABILITIES,
    ),
    LadderMember(
        name="bygpt5-medium-en",
        path=TEXT_MODEL_BASE / "bygpt5-medium-en",
        modality="text",
        tokenisation="byte",
        input_format="raw",
        evaluation_cohort_source="openwebtext",
        pretraining_corpus=PRETRAINING_UNDECLARED,
        cohort_corpus=TEXT_CORPUS,
        cohort_min_symbols=TEXT_MIN_CHARS,
        cohort_max_symbols=0,
        architecture="t5_decoder",
        capabilities=BYTE_TEXT_CAPABILITIES,
    ),
    LadderMember(
        name="protgpt2",
        path=MODEL_ROOT / "ProtGPT2",
        modality="protein",
        tokenisation="multi_residue_bpe",
        input_format="fasta_wrapped",
        evaluation_cohort_source="swissprot",
        pretraining_corpus="uniref50",
        cohort_corpus="plain_swissprot",
        cohort_min_symbols=UNCONDITIONAL_PROTEIN_BAND[0],
        cohort_max_symbols=UNCONDITIONAL_PROTEIN_BAND[1],
    ),
    LadderMember(
        name="zymctrl",
        path=MODEL_ROOT / "ZymCTRL",
        modality="protein",
        tokenisation="residue",
        input_format="ec_conditioned",
        evaluation_cohort_source="zymctrl_ec",
        pretraining_corpus="uniprot_ec_annotated",
        cohort_corpus="ec_labelled_swissprot",
        cohort_min_symbols=CONDITIONED_PROTEIN_BAND[0],
        cohort_max_symbols=CONDITIONED_PROTEIN_BAND[1],
    ),
    # ProGen2 is a GPT-J-style parallel-residual decoder, which ``arms.PANEL``,
    # ``arms.STAGED_ARMS`` and ``circuits._GPT_STYLE`` all name ``progen``.
    # ``architecture`` defaults to ``gpt2`` because most of this ladder is GPT-2,
    # so every ProGen2 rung has to say so explicitly: the field is read by
    # :func:`circuits_supported` and :func:`lens_supported`, it travels into the
    # analysis frame, and :func:`register_arm_spec` refuses a rung whose
    # declaration disagrees with the panel's.
    LadderMember(
        name="progen2-small",
        path=MODEL_ROOT / "progen2-small",
        modality="protein",
        tokenisation="residue",
        input_format="n_to_c_control",
        evaluation_cohort_source="swissprot",
        pretraining_corpus="uniref90_bfd30",
        cohort_corpus="plain_swissprot",
        cohort_min_symbols=UNCONDITIONAL_PROTEIN_BAND[0],
        cohort_max_symbols=UNCONDITIONAL_PROTEIN_BAND[1],
        architecture="progen",
    ),
    LadderMember(
        name="progen2-base",
        path=MODEL_ROOT / "progen2-base",
        modality="protein",
        tokenisation="residue",
        input_format="n_to_c_control",
        evaluation_cohort_source="swissprot",
        pretraining_corpus="progen2_base_mixture",
        cohort_corpus="plain_swissprot",
        cohort_min_symbols=UNCONDITIONAL_PROTEIN_BAND[0],
        cohort_max_symbols=UNCONDITIONAL_PROTEIN_BAND[1],
        architecture="progen",
    ),
    LadderMember(
        name="progen2-medium",
        path=MODEL_ROOT / "progen2-medium",
        modality="protein",
        tokenisation="residue",
        input_format="n_to_c_control",
        evaluation_cohort_source="swissprot",
        pretraining_corpus="uniref90_bfd30",
        cohort_corpus="plain_swissprot",
        cohort_min_symbols=UNCONDITIONAL_PROTEIN_BAND[0],
        cohort_max_symbols=UNCONDITIONAL_PROTEIN_BAND[1],
        architecture="progen",
    ),
    LadderMember(
        name="progen2-large",
        path=MODEL_ROOT / "progen2-large",
        modality="protein",
        tokenisation="residue",
        input_format="n_to_c_control",
        evaluation_cohort_source="swissprot",
        pretraining_corpus="uniref90_bfd30",
        cohort_corpus="plain_swissprot",
        cohort_min_symbols=UNCONDITIONAL_PROTEIN_BAND[0],
        cohort_max_symbols=UNCONDITIONAL_PROTEIN_BAND[1],
        architecture="progen",
    ),
    LadderMember(
        name="progen2-xlarge",
        path=MODEL_ROOT / "progen2-xlarge",
        modality="protein",
        tokenisation="residue",
        input_format="n_to_c_control",
        evaluation_cohort_source="swissprot",
        pretraining_corpus="uniref90_bfd30",
        cohort_corpus="plain_swissprot",
        cohort_min_symbols=UNCONDITIONAL_PROTEIN_BAND[0],
        cohort_max_symbols=UNCONDITIONAL_PROTEIN_BAND[1],
        architecture="progen",
    ),
)


# --------------------------------------------------------------------- ladder


def parse_ladder_table(path: Path) -> tuple[LadderMember, ...] | None:
    """Read a ladder from a markdown table, replacing :data:`DEFAULT_LADDER`.

    The ladder is assembled by a separate process, so a machine-readable table is
    authoritative when one exists. Two outcomes are distinguished rather than
    conflated. A document carrying no table with every column of
    :data:`LADDER_TABLE_COLUMNS` is a human staging report, not a ladder
    declaration, and returns ``None`` so the caller can record that and fall back;
    a document that *does* declare such a table but whose rows are malformed is an
    error, because a member with a guessed modality or input format would be
    measured off its own distribution and would then look like a modality effect.
    """

    text = Path(path).read_text(encoding="utf-8")
    header: list[str] | None = None
    members: list[LadderMember] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            header = None
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if header is None:
            lowered = [cell.lower() for cell in cells]
            if all(column in lowered for column in LADDER_TABLE_COLUMNS):
                header = lowered
            continue
        if all(set(cell) <= set("-: ") for cell in cells):
            continue
        if len(cells) != len(header):
            raise ValueError(
                f"{path}: row {cells} has {len(cells)} cells against a "
                f"{len(header)}-column header"
            )
        row = dict(zip(header, cells))
        members.append(
            LadderMember(
                name=row["name"],
                path=Path(row["path"]),
                modality=row["modality"],
                tokenisation=row["tokenisation"],
                input_format=row["input_format"],
                # The legacy ``source`` column is the evaluation-cohort corpus.
                evaluation_cohort_source=row["source"],
                # Optional: a table that does not declare a pretraining corpus
                # gets the sentinel rather than the cohort corpus under another
                # name, so "we do not know" stays distinguishable from
                # "openwebtext".
                pretraining_corpus=row.get(PRETRAINING_COLUMN) or PRETRAINING_UNDECLARED,
                cohort_corpus=row["cohort_corpus"],
                cohort_min_symbols=int(row["cohort_min_symbols"]),
                cohort_max_symbols=int(row["cohort_max_symbols"]),
            )
        )
    if header is None:
        return None
    if not members:
        raise ValueError(f"{path}: the ladder table has a header but no rows")
    names = [member.name for member in members]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: the ladder table repeats a member name")
    return tuple(members)


def inspect_member(member: LadderMember) -> dict[str, Any]:
    """Decide whether a ladder member can be measured, and record why not.

    Availability is a property of the filesystem at run time: the ladder is being
    downloaded while this control is being written, so a member may hold a config
    and no weights. Every negative answer carries the reason it was reached, so
    that "which rungs existed" is part of the record instead of being inferred
    from which rows are missing.

    The vocabulary check is not incidental. ``src.transfer.budget`` and
    ``src.transfer.pathways`` both read ``config.vocab_size`` to build the
    context-free baseline; a config that publishes only a head-specific vocabulary
    would make those two modules read different alphabets, so such a member is
    declared unavailable rather than measured on an unstated baseline.

    The ``auto_map`` check keeps the probe hermetic. A remote-code config whose
    module file has not landed yet sends ``AutoConfig`` to the network, where it
    either stalls or fetches code that is not the code on disk; a directory in
    that state is an incomplete download and is reported as one.
    """

    record: dict[str, Any] = {
        "name": member.name,
        "path": str(member.path),
        "modality": member.modality,
        "tokenisation": member.tokenisation,
        "input_format": member.input_format,
        # ``source`` is kept for artefact-schema stability and is the *evaluation
        # cohort* corpus; the two explicit keys beside it are what new readers
        # should use, because a bare "source" inside an arm block reads as
        # pretraining provenance and never was.
        "source": member.evaluation_cohort_source,
        "evaluation_cohort_source": member.evaluation_cohort_source,
        "pretraining_corpus": member.pretraining_corpus,
        "available": False,
        "unavailable_reason": None,
        "n_layer": None,
        "d_model": None,
        "vocab_size": None,
    }
    if not member.path.is_dir():
        record["unavailable_reason"] = "model directory does not exist"
        return record
    config_path = member.path / "config.json"
    if not config_path.is_file():
        record["unavailable_reason"] = "model directory holds no config.json"
        return record
    published = json.loads(config_path.read_text(encoding="utf-8"))
    tokenizer_path = member.path / "tokenizer.json"

    # The scoring-width hazard is decided from files where that is possible, so it
    # is decided for every rung including ones a later check will reject for a
    # mechanical reason: a rung whose logits are the wrong width is wrong in a way
    # that makes the model look worse, and that has to reach the record even when
    # the rung was going to be dropped anyway. A model shipping a custom tokenizer
    # class instead of a serialised fast tokenizer - ByGPT5 does - has no such file,
    # and for those the vocabulary is read through the tokenizer class after the
    # remote-code checks below have confirmed the directory is self-contained.
    tokenizer_vocab: int | None = None
    tokenizer_source = "deferred_to_tokenizer_class"
    if tokenizer_path.is_file():
        tokenizer_vocab = int(Tokenizer.from_file(str(tokenizer_path)).get_vocab_size())
        tokenizer_source = "tokenizer_json"
    declared_vocab = published.get("vocab_size")
    allowance = (
        None
        if tokenizer_vocab is None
        else -(-tokenizer_vocab // LOGIT_WIDTH_PADDING) * LOGIT_WIDTH_PADDING
    )
    record["tokenizer_vocab_size"] = tokenizer_vocab
    record["tokenizer_vocab_source"] = tokenizer_source
    record["logit_columns_used"] = None if declared_vocab is None else int(declared_vocab)
    if tokenizer_vocab is not None and declared_vocab is not None and int(declared_vocab) > allowance:
        record["unavailable_reason"] = (
            f"the checkpoint emits {int(declared_vocab)} logit columns against a "
            f"{tokenizer_vocab}-token tokenizer, so roughly "
            f"{int(declared_vocab) - tokenizer_vocab} columns correspond to no token "
            "and were never meaningfully trained. src.transfer.budget and "
            "src.transfer.pathways take log_softmax over the full width, which would "
            "put those columns in the normaliser and inflate both the clean "
            "cross-entropy and the unigram entropy for this rung; neither module is "
            "owned by this control, so the rung is excluded rather than reported "
            "unsliced"
        )
        return record

    auto_map = published.get("auto_map", {})
    remote = sorted(str(target) for target in auto_map.values() if "--" in str(target))
    if remote:
        record["unavailable_reason"] = (
            f"config auto_map is repo-qualified {remote}, so transformers resolves the "
            "modelling code from the Hub rather than from this directory; the run would "
            "depend on the network and on code that is not the code on disk"
        )
        return record
    missing_modules = sorted(
        {
            f"{str(target).split('.')[0]}.py"
            for target in auto_map.values()
            if not (member.path / f"{str(target).split('.')[0]}.py").is_file()
        }
    )
    if missing_modules:
        record["unavailable_reason"] = (
            f"config declares remote code {missing_modules} that is not on disk; "
            "the model directory is incomplete"
        )
        return record
    weights = [
        entry.name
        for entry in member.path.iterdir()
        if entry.is_file() and entry.suffix in WEIGHT_SUFFIXES
    ]
    if not weights:
        record["unavailable_reason"] = (
            f"model directory holds no weight file with suffix {list(WEIGHT_SUFFIXES)}"
        )
        return record
    config = AutoConfig.from_pretrained(str(member.path), trust_remote_code=True)
    n_layer = getattr(config, "n_layer", None) or getattr(config, "num_hidden_layers", None)
    d_model = (
        getattr(config, "n_embd", None)
        or getattr(config, "hidden_size", None)
        or getattr(config, "embed_dim", None)
    )
    vocab_size = getattr(config, "vocab_size", None)
    if n_layer is None or d_model is None:
        record["unavailable_reason"] = "config publishes no layer count or hidden width"
        return record
    if vocab_size is None:
        record["unavailable_reason"] = (
            "config publishes no vocab_size; src.transfer.budget and "
            "src.transfer.pathways read config.vocab_size for the context-free baseline"
        )
        return record
    if tokenizer_vocab is None:
        tokenizer = AutoTokenizer.from_pretrained(str(member.path), trust_remote_code=True)
        tokenizer_vocab = max(int(len(tokenizer)), int(tokenizer.vocab_size))
        record["tokenizer_vocab_size"] = tokenizer_vocab
        record["tokenizer_vocab_source"] = "tokenizer_class"
        allowance = -(-tokenizer_vocab // LOGIT_WIDTH_PADDING) * LOGIT_WIDTH_PADDING
        if int(vocab_size) > allowance:
            record["unavailable_reason"] = (
                f"the checkpoint emits {int(vocab_size)} logit columns against a "
                f"{tokenizer_vocab}-token tokenizer; the scoring helpers normalise "
                "over the full width"
            )
            return record
    record.update(
        {
            "available": True,
            "n_layer": int(n_layer),
            "d_model": int(d_model),
            "vocab_size": int(vocab_size),
            "weight_files": sorted(weights),
        }
    )
    return record


def arm_declaration(source: LadderMember | ArmSpec) -> dict[str, Any]:
    """The fields a ladder rung and its frozen panel entry must agree on.

    One list, read both by :func:`register_arm_spec`, which refuses a run on any
    disagreement, and by the invariant test that asserts there is none. Written
    once because the two used to check different fields: the test compared
    ``capabilities`` alone while the refusal compared all of these, so the
    ProGen2 rungs could declare -- and did declare -- ``gpt2`` here against the
    panel's ``progen`` with nothing failing until a campaign was launched.
    """

    return {
        "path": str(source.path),
        "modality": source.modality,
        "tokenisation": source.tokenisation,
        "input_format": source.input_format,
        "evaluation_cohort_source": source.evaluation_cohort_source,
        "pretraining_corpus": source.pretraining_corpus,
        "architecture": source.architecture,
        "capabilities": sorted(source.capabilities),
    }


def register_arm_spec(member: LadderMember, probe: Mapping[str, Any]) -> ArmSpec:
    """Make a ladder member loadable by ``arms.load_arm`` without editing ``arms``.

    ``arms.PANEL`` is the registry ``load_arm`` validates a request against, and
    ``arms.py`` is owned by the panel, not by this control. Registering the rung
    into that registry at run time reuses ``load_arm``'s dtype contract, shape
    check and tokenizer setup instead of duplicating them here, and keeps the
    panel file untouched.

    One honest consequence is recorded rather than hidden: for the four original
    panel members the shape in ``PANEL`` was written by hand and ``load_arm``'s
    comparison against the config is a real assertion, whereas for a rung
    registered here the shape came from that same config, so the comparison is a
    tautology. A conflicting re-declaration of an existing panel member is
    therefore refused outright.

    A checkpoint that ``src.transfer.arms.STAGED_ARMS`` declares a **non**-member
    is refused too, and for a stronger reason than consistency. That table is the
    declaration that a checkpoint loads and runs and must still not enter a
    panel-wide statistic. ``progen2-large`` and ``progen2-xlarge`` remain outside
    :data:`PANEL` even though they now declare a scoring-target alphabet of 32
    for opt-in budget measurements: registering either here would give the name
    two live declarations with different capability sets and would treat a
    staged scale rung as a campaign arm.
    """

    if member.name in STAGED_ARMS:
        raise ValueError(
            f"{member.name} is declared in src.transfer.arms.STAGED_ARMS as a staged "
            "NON-member of the panel, so it must not be registered into PANEL. An "
            "opt-in measurement that needs the declared scoring-target alphabet "
            "reaches it through arms.arm_spec and arms.load_arm_spec instead. Name "
            "it out of this run with --members, or declare a ladder table without it"
        )
    existing = PANEL.get(member.name)
    if existing is not None:
        declared = arm_declaration(existing)
        requested = arm_declaration(member)
        if declared != requested:
            raise ValueError(
                f"{member.name}: ladder declaration {requested} conflicts with the "
                f"frozen panel declaration {declared}"
            )
        if (existing.n_layer, existing.d_model) != (probe["n_layer"], probe["d_model"]):
            raise ValueError(
                f"{member.name}: panel declares {existing.n_layer}L/{existing.d_model}d, "
                f"config publishes {probe['n_layer']}L/{probe['d_model']}d"
            )
        return existing
    spec = ArmSpec(
        name=member.name,
        path=member.path,
        modality=member.modality,
        n_layer=int(probe["n_layer"]),
        d_model=int(probe["d_model"]),
        tokenisation=member.tokenisation,
        input_format=member.input_format,
        evaluation_cohort_source=member.evaluation_cohort_source,
        pretraining_corpus=member.pretraining_corpus,
        architecture=member.architecture,
        capabilities=member.capabilities,
    )
    PANEL[member.name] = spec
    return spec


# ------------------------------------------------------------------ x-axis


def convergence_row(
    baseline: Mapping[str, Any],
    *,
    clean_ce_nats: float,
    context_information_se_nats: float | None,
    symbols_per_token: float,
    n_scored_tokens: int,
    vocab_size: int,
    n_parameters: int,
    n_layer: int,
    d_model: int,
) -> dict[str, Any]:
    """Convergence / quality axes for one model on its own cohort.

    ``baseline`` is a ``src.transfer.pathways.unigram_baseline`` record and
    ``clean_ce_nats`` is measured on the same scored-token multiset the baseline
    was evaluated on. Realized information fraction is the share of that
    context-free baseline the model actually removes, so it is a ratio of two
    same-alphabet quantities and is comparable between a fifty-thousand-piece BPE
    arm and a twenty-residue one; the per-token quantities it is built from are
    not.

    The estimator behind that baseline is not a detail. A plug-in entropy
    computed on the very tokens it scores is biased downwards, and the bias
    scales with vocabulary against sample size: on a 32-sequence cohort it is
    about +0.003 nats for a 32-token protein alphabet and about +1.65 nats for a
    50257-piece BPE one. Since the denominator here is ``H - CE``, understating
    ``H`` understates the denominator and inflates every share built on it, by
    far more for the large-vocabulary arms than the small-vocabulary ones. That
    is a differential bias aligned exactly with the tokenisation contrast this
    control is trying to measure, so the plug-in figure is carried alongside as
    ``unigram_entropy_plug_in_nats`` and the bias is reported per rung rather
    than being invisible inside the axis.

    ``in_distribution`` is the exclusion criterion, not a diagnostic. A model
    whose clean cross-entropy does not beat the cohort's own unigram entropy has
    learned nothing usable about that cohort, so every normalised interpretability
    figure measured on it divides by a non-positive number.

    ``context_information_se_nats`` is the bootstrap standard error of that
    denominator and is required rather than defaulted, because
    :func:`analysis_frame` decides admissibility from it and
    :func:`src.transfer.budget.ratio_denominator_admissibility` has no fallback
    when it is absent. ``None`` is the one honest answer for a rung whose
    measurement produced no such bootstrap, and it travels as ``None`` into the
    row rather than being replaced by a constant.
    """

    if n_parameters < 1 or n_layer < 1 or d_model < 1:
        raise ValueError("parameter count, depth and width must be positive")
    if symbols_per_token <= 0.0:
        raise ValueError("symbols per token must be positive")
    if context_information_se_nats is not None and (
        not math.isfinite(context_information_se_nats)
        or context_information_se_nats <= 0.0
    ):
        raise ValueError(
            "a recorded context-information standard error must be finite and "
            f"strictly positive; got {context_information_se_nats!r}. Pass None "
            "where no bootstrap produced one"
        )
    entropy = float(baseline["nats"])
    plug_in = float(baseline["cohort_plug_in_entropy_nats"])
    clean_ce = float(clean_ce_nats)
    if not math.isfinite(entropy) or entropy <= 0.0:
        raise ValueError("the cohort's unigram baseline must be finite and positive")
    if not math.isfinite(clean_ce):
        raise ValueError("clean cross-entropy must be finite")
    context_information = entropy - clean_ce
    return {
        "unigram_entropy_nats": entropy,
        "unigram_estimator": str(baseline["estimator"]),
        "unigram_source": str(baseline["source"]),
        "unigram_entropy_plug_in_nats": plug_in,
        "unigram_plug_in_bias_nats": entropy - plug_in,
        "unigram_reference": baseline["reference"],
        "clean_ce_nats": clean_ce,
        "context_information_nats": context_information,
        "context_information_se_nats": (
            None
            if context_information_se_nats is None
            else float(context_information_se_nats)
        ),
        "realized_information_fraction": context_information / entropy,
        "clean_ce_bits_per_symbol": clean_ce / LN2 / float(symbols_per_token),
        "unigram_entropy_bits_per_symbol": entropy / LN2 / float(symbols_per_token),
        "symbols_per_token": float(symbols_per_token),
        "n_parameters": int(n_parameters),
        "log10_parameters": math.log10(float(n_parameters)),
        "n_layer": int(n_layer),
        "d_model": int(d_model),
        "vocab_size": int(vocab_size),
        "n_scored_tokens": int(n_scored_tokens),
        "in_distribution": bool(context_information > 0.0),
        "in_distribution_rule": "clean_ce_nats < unigram_entropy_nats on the arm's own cohort",
    }


def renderable(member: LadderMember, corpus: str) -> tuple[bool, str | None]:
    """Can this member's input format be produced from that corpus?

    ``Cohort.input_strings`` refuses to render an EC-conditioned arm without
    conditioning labels, and refuses a cross-modality pairing outright. Deciding
    that here rather than discovering it in an exception keeps the
    cohort-sensitivity sweep an enumerated design instead of a set of attempts
    with the failures thrown away.
    """

    if corpus not in COHORT_CORPORA:
        raise ValueError(f"unknown cohort corpus {corpus!r}")
    if (member.modality == "text") != (corpus == TEXT_CORPUS):
        return False, "the corpus belongs to the other modality"
    if member.input_format == "ec_conditioned" and corpus != "ec_labelled_swissprot":
        return False, "an EC-conditioned prompt needs the EC-labelled corpus"
    return True, None


def circuits_supported(member: LadderMember) -> tuple[bool, str | None]:
    """Can ``src.transfer.circuits`` score this member at all?

    Two independent conditions, because two independent things can be missing.

    *The rendering.* The induction census and the attribution decomposition both
    route through ``circuits.content_bounds``, which decides which token positions
    are modality content rather than prompt syntax, and the census also routes
    through ``circuits.prefix_ids``. Neither knows every format ``arms`` can
    render, and a format they do not know is a hard error rather than a degraded
    measurement.

    *The module layout.* ``circuits._CIRCUIT_ARCHITECTURES`` is that module's own
    declaration of the architectures it can resolve, and the attribution
    decomposition raises on anything outside it. That check used to be carried
    here only by ``ArmSpec.capabilities``, which is an *intent* and not the same
    statement -- ByGPT5 declares ``circuits`` because its attention patterns are
    readable per head, which is what the prediction-addressed census needs, while
    ``circuits.final_norm`` and ``circuits.head_ov_weights`` still have no
    ``t5_decoder`` branch. Reading the capability as though it guaranteed the
    module would have crashed this control inside ``measure_attribution`` on three
    rungs.

    Checking here keeps both an enumerated property of the design instead of an
    exception discovered halfway through a rung: the budget and pathway axes,
    which do not touch ``circuits``, still complete for every member, and the
    metrics that cannot be produced are recorded as absent with the reason naming
    what is missing.
    """

    if member.input_format not in INPUT_FORMATS:
        raise ValueError(f"{member.name}: unknown input format {member.input_format!r}")
    if member.input_format not in CIRCUITS_INPUT_FORMATS:
        return False, (
            f"src.transfer.circuits.content_bounds and .prefix_ids have no "
            f"{member.input_format!r} branch, so the induction census and the direct "
            "logit attribution cannot be scored on this arm's native rendering"
        )
    if member.architecture not in _CIRCUIT_ARCHITECTURES:
        return False, (
            f"architecture {member.architecture!r} is not in "
            f"src.transfer.circuits._CIRCUIT_ARCHITECTURES "
            f"{sorted(_CIRCUIT_ARCHITECTURES)}, so that module cannot resolve this "
            "rung's normalisation, embedding or per-head projections"
        )
    return True, None


def lens_supported(member: LadderMember) -> tuple[bool, str | None]:
    """Can ``src.transfer.lenses`` build a lens head for this member?

    The capability flag on ``ArmSpec`` says the panel intends an arm for the lens
    family; this says whether the module can currently serve it. The two can
    disagree while a module is catching up with the panel, and when they do the
    honest output is a recorded skip rather than a number produced by a path that
    was never written for this architecture.
    """

    if member.architecture in LENS_ARCHITECTURES:
        return True, None
    return False, (
        f"architecture {member.architecture!r} is not in "
        f"src.transfer.lenses.FINAL_LAYER_NORM_PATH {sorted(LENS_ARCHITECTURES)}, so "
        "src.transfer.lenses.lens_head cannot resolve the final normalisation this "
        "rung's own output head applies and the output aperture cannot be measured "
        "on this arm"
    )


def cohort_sensitivity_rows(
    member: LadderMember,
    power_by_corpus: Mapping[tuple[str, int, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """One row per cohort this member was scored on, native one flagged.

    The in-distribution rule is an exclusion rule, so the claim that a rung fails
    it has to be checked against the alternative rather than asserted. Scoring
    every protein rung on every protein cohort in the run turns "ProtGPT2 is
    off-distribution" into "ProtGPT2 is off-distribution on this band and
    in-distribution on that one", which is a different and far more useful
    statement, and it is what makes the length dependence auditable instead of
    assumed.

    One window mismatch is declared rather than hidden: the cross-entropy comes
    from ``budget.arm_power``, which drops an EC-conditioned arm's prompt tokens,
    while the baseline is counted over the ``prepare_batches`` multiset, which
    does not. The two coincide for every arm whose input format carries no prompt
    scaffolding, and the one arm where they differ - ZymCTRL - has exactly one
    renderable cohort, so no ZymCTRL row in this sweep is compared against
    another.
    """

    rows: list[dict[str, Any]] = []
    for key, entry in sorted(power_by_corpus.items()):
        corpus, low, high = key
        power = entry["power"]
        entropy = float(entry["unigram_entropy_nats"])
        plug_in = float(entry["unigram_entropy_plug_in_nats"])
        clean_ce = float(power["clean_ce_nats"])
        rows.append(
            {
                "cohort_corpus": corpus,
                "cohort_min_symbols": low,
                "cohort_max_symbols": high,
                "native": key == member.cohort_key,
                "unigram_entropy_nats": entropy,
                "unigram_estimator": str(entry["unigram_estimator"]),
                "unigram_entropy_plug_in_nats": plug_in,
                "unigram_plug_in_bias_nats": entropy - plug_in,
                "clean_ce_nats": clean_ce,
                "context_information_nats": entropy - clean_ce,
                "realized_information_fraction": (entropy - clean_ce) / entropy,
                "clean_ce_bits_per_symbol": float(power["clean_ce_bits_per_symbol"]),
                "symbols_per_token": float(power["symbols_per_token"]),
                "n_scored_tokens": int(power["n_scored_tokens"]),
                "in_distribution": bool(entropy - clean_ce > 0.0),
            }
        )
    return rows


# ------------------------------------------------------------------ y-axis


def pathway_summary(
    mlp: Mapping[str, Any],
    attention: Mapping[str, Any],
) -> dict[str, Any]:
    """Collapse the two whole-pathway ablations into the shares used as y-values.

    Both shares are read against one cohort budget, so their difference is a
    within-model contrast and does not inherit the cohort's overall difficulty.
    That is what makes an MLP share comparable between a text decoder scored on
    OpenWebText and a protein decoder scored on Swiss-Prot.
    """

    raw_mlp = mlp["share_of_context_information"]
    raw_attn = attention["share_of_context_information"]
    mlp_share = None if raw_mlp is None else float(raw_mlp)
    attn_share = None if raw_attn is None else float(raw_attn)
    return {
        "mlp_share_of_context_information": mlp_share,
        "attn_share_of_context_information": attn_share,
        "mlp_minus_attn_share": (
            None if mlp_share is None or attn_share is None else mlp_share - attn_share
        ),
        "mlp_ce_delta_nats": float(mlp["ce_delta_nats"]),
        "attn_ce_delta_nats": float(attention["ce_delta_nats"]),
        "mlp_kl_nats": float(mlp["kl_clean_to_ablated_nats"]),
        "attn_kl_nats": float(attention["kl_clean_to_ablated_nats"]),
        "pathway_context_information_nats": float(mlp["context_information_nats"]),
        "pathway_unigram_entropy_nats": float(mlp["unigram_entropy_nats"]),
        "mlp_measurable": bool(mlp["measurable"]),
        "attn_measurable": bool(attention["measurable"]),
    }


def induction_summary(
    alignment: Mapping[str, Any],
    census: Mapping[str, Any],
    *,
    threshold: float = DEFAULT_INDUCTION_THRESHOLD,
    prefix: str = "induction_",
) -> dict[str, Any]:
    """Census headline, normalised so that a deeper model is not counted higher.

    ``alignment`` is :func:`src.transfer.circuits.attention_alignment_scores` and
    ``census`` is :func:`src.transfer.circuits.head_census` applied to its
    prefix-matching matrix. The raw count of induction heads scales with depth
    times head count, which varies over a size ladder, so the fraction of heads is
    what enters the fits. The maximum prefix-matching score is divided by the
    probe's uniform-attention baseline for the same reason: a longer probe lowers
    uniform attention and would inflate an unnormalised maximum.
    """

    key = f"{threshold:.2f}"
    counts = census["count_above_threshold"]
    if key not in counts:
        raise KeyError(
            f"induction threshold {key} is not in the census; available {sorted(counts)}"
        )
    distribution = census["distribution"]
    n_heads = int(distribution["n_heads"])
    if n_heads < 1:
        raise ValueError("the census covered no heads")
    uniform = float(alignment["uniform_baseline"])
    if not math.isfinite(uniform) or uniform <= 0.0:
        raise ValueError("the uniform-attention baseline must be finite and positive")
    maximum = float(distribution["max"])
    return {
        f"{prefix}probe_kind": str(alignment["kind"]),
        f"{prefix}n_probes": int(alignment["n_probes"]),
        f"{prefix}scored_query_positions": int(alignment["scored_query_positions"]),
        f"{prefix}uniform_baseline": uniform,
        f"{prefix}n_heads": n_heads,
        f"{prefix}threshold": float(threshold),
        f"{prefix}max_prefix_matching": maximum,
        f"{prefix}max_prefix_matching_over_uniform": maximum / uniform,
        f"{prefix}mean_prefix_matching": float(distribution["mean"]),
        f"{prefix}count_above_threshold": int(counts[key]),
        f"{prefix}fraction_of_heads_above_threshold": int(counts[key]) / n_heads,
        f"{prefix}count_above_data_driven": int(census["count_above_data_driven"]),
        f"{prefix}data_driven_threshold": float(census["data_driven_threshold"]),
    }


def attribution_summary(attribution: Mapping[str, Any]) -> dict[str, Any]:
    """Direct-logit-attribution concentration, on depth-normalised scales.

    The component count is ``2 * n_layer + 1``, so a raw top-1 share falls as the
    ladder deepens whatever the model is doing. The participation ratio divided by
    the component count and the entropy divided by ``log(components)`` do not, so
    those are what enter the fits; the raw top-1 share is kept in the record as a
    diagnostic and is explicitly marked as depth-confounded.
    """

    concentration = attribution["concentration"]
    magnitude = attribution["pathway_magnitude_fraction"]
    return {
        "n_components": int(attribution["n_components"]),
        "scored_positions": int(attribution["scored_positions"]),
        "dla_participation_fraction": float(concentration["participation_fraction"]),
        "dla_normalised_entropy": float(concentration["normalised_entropy"]),
        "dla_top1_share_depth_confounded": float(concentration["top1_share"]),
        "dla_mlp_magnitude_fraction": float(magnitude["mlp"]),
        "dla_attention_magnitude_fraction": float(magnitude["attention"]),
        "dla_embed_magnitude_fraction": float(magnitude["embed"]),
        "dla_logit_max_absolute_error": float(attribution["logit_max_absolute_error"]),
        "dla_logit_mean_absolute_error": float(attribution["logit_mean_absolute_error"]),
        "dla_residual_relative_l2_error": float(attribution["residual_relative_l2_error"]),
        "dla_residual_max_absolute_ratio": float(attribution["residual_max_absolute_ratio"]),
    }


# ----------------------------------------------------------------- analysis


def aperture_summary(
    layer_record: Mapping[str, Any],
    *,
    d_model: int,
    vocab_size: int,
) -> dict[str, Any]:
    """Output-aperture figures for one layer, with the forced ones marked.

    The Jacobian ``d logits / d h_l`` has rank at most ``min(d_model, V - 1)``.
    For ZymCTRL, every ProGen2 rung and every byte-level rung that bound is
    ``V - 1`` and it is far below the width, so those arms are blind to most of
    their own residual variance *as a matter of algebra*: the blind-variance
    fraction is not a discovery about the model and a modality coefficient fitted
    on it would be a coefficient on the vocabulary size.

    ``rank_is_forced`` records which side of that line each arm falls, and only
    ``gain_alignment_ratio`` is offered to the fits. That ratio compares how much
    activation variance the Jacobian actually weights against what an isotropic
    sensitivity of the same trace would weight, so it is a statement about how
    variance distributes over the surviving directions rather than about how many
    there are - which is the part the algebra does not fix.
    """

    if d_model < 1 or vocab_size < 2:
        raise ValueError("width must be positive and the vocabulary at least two")
    probe_mean = layer_record["probe_mean"]
    bound = min(int(d_model), int(vocab_size) - 1)
    return {
        "aperture_layer": int(layer_record["layer"]),
        "aperture_relative_depth": float(layer_record["relative_depth"]),
        "aperture_probes": int(layer_record["probes"]),
        "aperture_algebraic_rank_bound": bound,
        "aperture_numerical_rank": float(probe_mean["numerical_rank"]),
        "aperture_rank_over_bound": float(probe_mean["numerical_rank"]) / bound,
        "aperture_rank_is_forced": bool(int(vocab_size) - 1 < int(d_model)),
        "aperture_blind_variance_fraction_not_fitted": float(
            probe_mean["blind_variance_fraction"]
        ),
        "aperture_expressed_energy_fraction": float(probe_mean["expressed_energy_fraction"]),
        "aperture_chance_expressed_fraction": float(
            layer_record["chance_expressed_fraction"]
        ),
        "aperture_mean_squared_principal_cosine": float(
            probe_mean["mean_squared_principal_cosine"]
        ),
        "aperture_gain_alignment_ratio": float(probe_mean["gain_alignment_ratio"]),
        "aperture_stable_rank": float(probe_mean["stable_rank"]),
    }


def _corpus_field(record: Mapping[str, Any], key: str, fallback: str) -> str:
    """Resolve one corpus field of a per-model record, panel first.

    A record written before the ``source`` split carries neither
    ``evaluation_cohort_source`` nor ``pretraining_corpus``. Reading the frozen
    panel declaration for that arm is better than defaulting, because the
    default for a pretraining corpus would otherwise be an evaluation-cohort
    name -- which is precisely the confusion the split exists to end.
    """

    value = record.get(key)
    if value:
        return str(value)
    spec = PANEL.get(str(record.get("name", "")))
    if spec is not None:
        return str(getattr(spec, key))
    return fallback


#: Why a rung carries no denominator verdict, rather than carrying a false one.
#:
#: The Fieller precondition is evaluated against the denominator's *own*
#: bootstrap standard error, and a rung whose measurement produced no such
#: bootstrap has nothing to evaluate it against. Substituting a constant is
#: exactly the defect EXP-R2-218 measured -- the retired 0.30-nat floor is up to
#: 3.2x too lax for this job -- so the verdict is withheld and named instead.
NO_DENOMINATOR_STANDARD_ERROR = (
    "no bootstrap standard error for this rung's context information was "
    "recorded, so budget.ratio_denominator_admissibility cannot be evaluated. "
    "There is no fallback: a magnitude constant substituted for an unavailable "
    "SE is the defect EXP-R2-218 measured. The rung is neither admitted nor "
    "refused as a denominator here"
)


def analysis_frame(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Flatten per-model records into the rows the fits and contrasts consume.

    A metric a model could not produce - because it was excluded as
    off-distribution, or because its pathway denominator was non-positive -
    becomes ``None`` rather than being imputed or dropped, so a row is always
    present for every measured member and the reason it carries no y-value stays
    attached to it.

    ``measurable_denominator`` is a second, stricter flag alongside
    ``in_distribution``. The sign rule asks whether the model beat the
    context-free baseline at all; this asks whether it beat it by enough that
    dividing by the margin is arithmetic rather than noise. A rung can pass the
    first and fail the second - ProtGPT2 on full-length Swiss-Prot clears the
    baseline by 0.18 nats and its whole-MLP share is consequently above twelve -
    so the two are recorded separately and only the sign rule gates the deciding
    fit.

    **The second question is a ratio's, so it is asked of precision and not of a
    magnitude.** It used to be a comparison against ``budget``'s 0.30-nat floor,
    which EXP-R2-218 measured to be up to 3.2x too lax for this exact job: on 12
    of 15 panel arms a denominator at 0.30 nats still leaves a share's confidence
    set unbounded. It is now
    :func:`src.transfer.budget.ratio_denominator_admissibility` -- Fieller's
    precondition ``I_hat > 8.765 * SE(I_hat)`` against this rung's own bootstrap
    standard error, which is a different number on every arm.

    A rung that recorded no such standard error gets ``None`` and
    :data:`NO_DENOMINATOR_STANDARD_ERROR`, never a fallback to a constant. It is
    then absent from both sides of any admissibility split rather than silently
    counted on one of them.
    """

    frame: list[dict[str, Any]] = []
    for record in records:
        convergence = record["convergence"]
        cohort = record["cohort"]
        standard_error = convergence.get("context_information_se_nats")
        admissibility = (
            None
            if standard_error is None
            else ratio_denominator_admissibility(
                float(convergence["context_information_nats"]),
                float(standard_error),
                baseline_entropy_nats=float(convergence["unigram_entropy_nats"]),
                symbols_per_token=float(convergence["symbols_per_token"]),
            )
        )
        row: dict[str, Any] = {
            "name": str(record["name"]),
            "modality": str(record["modality"]),
            "tokenisation": str(record["tokenisation"]),
            "input_format": str(record["input_format"]),
            "conditioning": (
                "ec_conditioned"
                if record["input_format"] == "ec_conditioned"
                else "unconditional"
            ),
            # Three corpus-flavoured fields sit together here and they are three
            # different facts, so all three are named. ``source`` is the legacy
            # spelling of ``evaluation_cohort_source`` and is kept only so a
            # frozen artefact reader does not lose a column. A runner that
            # predates the split supplies neither new key, so both fall back to
            # the frozen panel declaration for that arm, which
            # :func:`register_arm_spec` has already reconciled against the
            # ladder.
            "source": str(record["source"]),
            "evaluation_cohort_source": _corpus_field(
                record, "evaluation_cohort_source", str(record["source"])
            ),
            "pretraining_corpus": _corpus_field(
                record, "pretraining_corpus", PRETRAINING_UNDECLARED
            ),
            "cohort_corpus": str(cohort["source"]),
            "cohort_digest": str(cohort["digest"]),
            "cohort_min_symbols": int(record["cohort_min_symbols"]),
            "cohort_max_symbols": int(record["cohort_max_symbols"]),
            "in_distribution": bool(convergence["in_distribution"]),
            # ZymCTRL's EC tag very nearly determines the Pfam family, so any
            # family-flavoured quantity read from it under native conditioning may
            # be reading the prompt rather than the model. Its realized information
            # fraction is the highest in the panel and is partly its own prompt
            # supplying the answer, so the flag travels with the row and every fit
            # is also reported without the rungs that carry it.
            "conditioning_leak": bool(record["input_format"] == "ec_conditioned"),
            "architecture": str(record["architecture"]),
            "capabilities": sorted(record["capabilities"]),
            "measurable_denominator": (
                None if admissibility is None else bool(admissibility["admissible"])
            ),
            "denominator_admissibility": admissibility,
            "denominator_admissibility_unavailable_reason": (
                None if admissibility is not None else NO_DENOMINATOR_STANDARD_ERROR
            ),
            "unigram_estimator": str(convergence["unigram_estimator"]),
            "unigram_plug_in_bias_nats": float(convergence["unigram_plug_in_bias_nats"]),
        }
        for axis in CONVERGENCE_AXES:
            row[axis] = float(convergence[axis])
        row["n_parameters"] = int(convergence["n_parameters"])
        row["n_layer"] = int(convergence["n_layer"])
        row["d_model"] = int(convergence["d_model"])
        row["context_information_nats"] = float(convergence["context_information_nats"])
        measured: dict[str, Any] = {}
        for block_name in ("pathways", "induction", "attribution", "aperture"):
            block = record.get(block_name)
            if block is not None:
                measured.update(block)
        for metric in INTERPRETABILITY_METRICS:
            value = measured.get(metric)
            row[metric] = None if value is None else float(value)
        frame.append(row)
    return frame


def fit_modality_offset(
    frame: Sequence[Mapping[str, Any]],
    *,
    metric_key: str,
    axis_key: str,
    confidence: float = 0.95,
    include_tokenisation: bool = False,
) -> dict[str, Any]:
    """Ordinary least squares of ``metric ~ 1 + convergence + protein``.

    With ``include_tokenisation`` a symbol-level indicator joins the design, and
    the modality coefficient becomes the offset at matched convergence *and*
    matched tokenisation. The unadjusted fit is retained beside it because the two
    disagreeing is itself the finding: if a modality offset evaporates once
    tokenisation is held fixed, it was a tokenisation offset.

    What this design holds fixed is exactly that: one convergence axis, and
    optionally tokenisation. Scale is never a covariate here -- ``log10_parameters``
    is one of :data:`CONVERGENCE_AXES`, so it enters as an *alternative* axis in a
    separate fit rather than beside another one -- and architecture and lineage
    are not in the design at all. Distribution is handled by exclusion rather than
    by adjustment: ``analysis_frame`` flags a rung scored off its own distribution
    and this fit drops it. The confounds outside that list are what
    :func:`paired_architecture_contrast`, :func:`tokenisation_contrast` and
    :func:`conditioning_control` report separately, and they are not absorbed into
    this coefficient.

    The question this control exists to answer is whether a protein decoder at the
    *same* convergence as a text decoder shows the same interpretability metric.
    Under a common slope that is exactly the coefficient on the protein indicator,
    which is why the indicator model is fitted rather than two separate curves:
    two separate curves at this sample size have no overlapping support and their
    difference is an extrapolation.

    The common-slope restriction is real and cannot be tested with a handful of
    rungs. So is the collinearity risk: if every text model sits at one end of the
    convergence axis and every protein model at the other, the indicator is a
    relabelling of the slope and the offset is not identified. Both are reported -
    rank deficiency refuses the fit outright, and the residual degrees of freedom
    travel with every coefficient - instead of being absorbed into a number that
    looks like an answer.
    """

    if metric_key not in INTERPRETABILITY_METRICS:
        raise KeyError(f"unknown interpretability metric {metric_key!r}")
    if axis_key not in CONVERGENCE_AXES:
        raise KeyError(f"unknown convergence axis {axis_key!r}")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")

    capability = METRIC_CAPABILITY[metric_key]
    usable = [
        row
        for row in frame
        if row["in_distribution"]
        and capability in row["capabilities"]
        and row[metric_key] is not None
        and row[axis_key] is not None
    ]
    kept = {row["name"] for row in usable}
    excluded = [row["name"] for row in frame if row["name"] not in kept]
    counts = {
        modality: sum(1 for row in usable if row["modality"] == modality)
        for modality in MODALITIES
    }
    names = ("intercept", "convergence_slope", "protein_offset")
    if include_tokenisation:
        names = (*names, "symbol_tokenisation_offset")
    result: dict[str, Any] = {
        "metric": metric_key,
        "axis": axis_key,
        "required_capability": capability,
        "adjusts_for_tokenisation": bool(include_tokenisation),
        "terms": list(names),
        "n_usable": len(usable),
        "usable_members": [row["name"] for row in usable],
        "excluded_members": excluded,
        "modality_counts": counts,
        "confidence": float(confidence),
        "fitted": False,
        "unfitted_reason": None,
        "residual_dof": None,
        "coefficients": None,
        "r_squared": None,
        "residual_standard_deviation": None,
        "saturated": None,
        "identification": identification_check(usable),
        "per_modality": _per_modality_slopes(usable, metric_key, axis_key),
    }
    if min(counts.values()) < 1:
        result["unfitted_reason"] = "a modality contributes no usable member"
        return result
    if len(usable) <= len(names):
        result["unfitted_reason"] = (
            f"{len(usable)} usable members against {len(names)} fitted parameters; "
            "at least one more is required for any residual degrees of freedom"
        )
        return result

    x = np.asarray([float(row[axis_key]) for row in usable], dtype=np.float64)
    y = np.asarray([float(row[metric_key]) for row in usable], dtype=np.float64)
    protein = np.asarray(
        [1.0 if row["modality"] == "protein" else 0.0 for row in usable], dtype=np.float64
    )
    columns = [np.ones_like(x), x, protein]
    if include_tokenisation:
        columns.append(
            np.asarray(
                [
                    1.0 if TOKENISATION_FAMILIES[row["tokenisation"]] == "symbol" else 0.0
                    for row in usable
                ],
                dtype=np.float64,
            )
        )
    design = np.column_stack(columns)
    rank = int(np.linalg.matrix_rank(design))
    result["design_rank"] = rank
    # A rank-deficient design has an infinite condition number, which is true but
    # not representable in strict JSON; the rank below already says it exactly.
    condition = float(np.linalg.cond(design))
    result["design_condition_number"] = condition if math.isfinite(condition) else None
    if rank < design.shape[1]:
        result["unfitted_reason"] = (
            "design matrix is rank deficient: two of the convergence axis, the "
            "modality indicator and the tokenisation indicator carry the same "
            "information, so no offset is separately identified"
        )
        return result
    dof = design.shape[0] - design.shape[1]
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    residual_sum_of_squares = float(residual @ residual)
    variance = residual_sum_of_squares / dof
    covariance = np.linalg.inv(design.T @ design) * variance
    standard_errors = np.sqrt(np.diag(covariance))
    radius = float(stats.t.ppf((1.0 + confidence) / 2.0, dof))
    total_sum_of_squares = float(((y - y.mean()) ** 2).sum())
    result.update(
        {
            "fitted": True,
            "residual_dof": int(dof),
            "residual_standard_deviation": math.sqrt(variance),
            "saturated": bool(variance <= 0.0),
            "r_squared": (
                1.0 - residual_sum_of_squares / total_sum_of_squares
                if total_sum_of_squares > 0.0
                else None
            ),
            "coefficients": {
                name: {
                    "estimate": float(estimate),
                    "standard_error": float(error),
                    "interval": [
                        float(estimate - radius * error),
                        float(estimate + radius * error),
                    ],
                }
                for name, estimate, error in zip(names, coefficients, standard_errors)
            },
        }
    )
    return result


def identification_check(usable: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Is the fitted indicator a modality indicator, or a relabelled tokenisation one?

    The panel confounds the two: the text arm and ProtGPT2 use a learned subword
    vocabulary, ZymCTRL and the ProGen2 rungs use one token per residue. ProtGPT2
    is the single rung that breaks the confound, because it is protein *and*
    subword. If it drops out - and it drops out whenever it is off-distribution on
    its own cohort - then every usable protein rung is symbol-level and every
    usable text rung is subword, the two indicators span the same column, and the
    coefficient this control fits is a tokenisation coefficient wearing a modality
    label.

    Reporting that coefficient as a modality effect would be exactly the
    confounding this control was built to remove, so the check is a precondition
    of the verdict rather than a caveat attached to it. It blocks both readings
    equally: an unidentified coefficient cannot establish a residual gap and
    cannot rule one out.
    """

    families: dict[str, set[str]] = {}
    for row in usable:
        tokenisation = str(row["tokenisation"])
        if tokenisation not in TOKENISATION_FAMILIES:
            raise KeyError(f"unknown tokenisation {tokenisation!r}")
        families.setdefault(str(row["modality"]), set()).add(
            TOKENISATION_FAMILIES[tokenisation]
        )
    text = families.get("text", set())
    protein = families.get("protein", set())
    shared = sorted(text & protein)
    identified = bool(text and protein and shared)
    return {
        "tokenisation_families_by_modality": {
            modality: sorted(values) for modality, values in sorted(families.items())
        },
        "shared_tokenisation_families": shared,
        "modality_identified": identified,
        "reason": (
            None
            if identified
            else (
                "no tokenisation family is represented in both modalities among the "
                "usable members, so the modality indicator and a tokenisation "
                "indicator span the same column and are not separately identified"
            )
        ),
    }


def _per_modality_slopes(
    usable: Sequence[Mapping[str, Any]], metric_key: str, axis_key: str
) -> dict[str, Any]:
    """Within-modality trend, reported without inference.

    With two or three rungs per modality a slope is a description of those points
    and nothing more, so no interval is attached to it. It is here because the
    common-slope assumption of :func:`fit_modality_offset` is otherwise invisible.
    """

    output: dict[str, Any] = {}
    for modality in MODALITIES:
        rows = [row for row in usable if row["modality"] == modality]
        entry: dict[str, Any] = {
            "n": len(rows),
            "members": [row["name"] for row in rows],
            "slope": None,
            "intercept": None,
        }
        if len(rows) >= 2:
            x = np.asarray([float(row[axis_key]) for row in rows], dtype=np.float64)
            y = np.asarray([float(row[metric_key]) for row in rows], dtype=np.float64)
            if float(x.std()) > 0.0:
                slope, intercept = np.polyfit(x, y, 1)
                entry["slope"] = float(slope)
                entry["intercept"] = float(intercept)
        output[modality] = entry
    return output


def nearest_neighbour_contrasts(
    frame: Sequence[Mapping[str, Any]],
    *,
    metric_key: str,
    axis_key: str,
) -> list[dict[str, Any]]:
    """Pair each protein rung with the text rung closest on the convergence axis.

    This is the reading that survives when there are too few points to fit
    anything: if a protein model at the same realized information fraction as a
    text model shows the same metric, the apparent gap is convergence. The axis
    gap is reported with every pair, because a pair matched only loosely on
    convergence says correspondingly less.

    Eligibility is the same rule :func:`fit_modality_offset` applies, including
    the capability gate. It used to be weaker here -- a row was admitted on
    having a value, not on being allowed to have one -- so a runner that
    populated a metric for an arm the panel refuses that measurement family for
    would have been silently excluded from the fit and silently *included* in
    this contrast, which is the same number under two different eligibility
    rules in one artefact.
    """

    if metric_key not in INTERPRETABILITY_METRICS:
        raise KeyError(f"unknown interpretability metric {metric_key!r}")
    if axis_key not in CONVERGENCE_AXES:
        raise KeyError(f"unknown convergence axis {axis_key!r}")
    capability = METRIC_CAPABILITY[metric_key]
    usable = [
        row
        for row in frame
        if row["in_distribution"]
        and capability in row["capabilities"]
        and row[metric_key] is not None
        and row[axis_key] is not None
    ]
    text_rows = [row for row in usable if row["modality"] == "text"]
    contrasts: list[dict[str, Any]] = []
    for row in usable:
        if row["modality"] != "protein":
            continue
        if not text_rows:
            contrasts.append(
                {
                    "protein_member": row["name"],
                    "text_member": None,
                    "reason": "no usable text member to match against",
                }
            )
            continue
        partner = min(text_rows, key=lambda other: abs(other[axis_key] - row[axis_key]))
        contrasts.append(
            {
                "protein_member": row["name"],
                "text_member": partner["name"],
                "required_capability": capability,
                "axis": axis_key,
                "protein_axis_value": float(row[axis_key]),
                "text_axis_value": float(partner[axis_key]),
                "axis_gap": float(row[axis_key] - partner[axis_key]),
                "metric": metric_key,
                "protein_metric": float(row[metric_key]),
                "text_metric": float(partner[metric_key]),
                "metric_gap_protein_minus_text": float(row[metric_key] - partner[metric_key]),
            }
        )
    return contrasts


def tokenisation_contrast(
    frame: Sequence[Mapping[str, Any]],
    *,
    keys: Sequence[str],
) -> dict[str, Any]:
    """Multi-residue BPE against residue-level, inside the protein modality.

    Tokenisation is confounded with modality across the panel - the text arm and
    ProtGPT2 are both BPE, ZymCTRL and the ProGen2 rungs are residue-level - so
    the only place it can be isolated is within protein. This contrast is reported
    separately from the fits and never feeds the verdict, because with one BPE
    protein rung it is a description of two groups rather than a test.
    """

    groups: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for row in frame:
        if row["modality"] != "protein":
            continue
        bucket = groups.setdefault(row["tokenisation"], {"used": [], "off_distribution": []})
        bucket["used" if row["in_distribution"] else "off_distribution"].append(row)
    summary: dict[str, Any] = {
        "groups": {},
        "difference_bpe_minus_residue": {},
        "runnable": False,
        "unrunnable_reason": None,
        "note": (
            "within-protein contrast only; tokenisation is fully confounded with "
            "modality across the panel"
        ),
    }
    for tokenisation, bucket in sorted(groups.items()):
        rows = bucket["used"]
        entry: dict[str, Any] = {
            "n": len(rows),
            "members": [row["name"] for row in rows],
            "members_excluded_off_distribution": [
                row["name"] for row in bucket["off_distribution"]
            ],
        }
        for key in keys:
            values = [float(row[key]) for row in rows if row[key] is not None]
            entry[key] = {
                "n": len(values),
                "mean": float(np.mean(values)) if values else None,
                "interval": mean_interval(values) if len(values) >= 2 else None,
            }
        summary["groups"][tokenisation] = entry
    bpe = summary["groups"].get("multi_residue_bpe")
    residue = summary["groups"].get("residue")
    # A tokenisation group whose only members were excluded as off-distribution
    # must be reported as an unrunnable control rather than as a missing row: the
    # contrast the brief asks for is precisely what such an exclusion destroys.
    empty = [
        tokenisation
        for tokenisation, entry in summary["groups"].items()
        if entry["n"] == 0
    ]
    if bpe is None or residue is None:
        summary["unrunnable_reason"] = (
            "the protein rungs cover only "
            f"{sorted(summary['groups'])}; both a multi-residue-BPE and a "
            "residue-level rung are required"
        )
    elif empty:
        summary["unrunnable_reason"] = (
            f"tokenisation groups {sorted(empty)} retained no in-distribution member, "
            "so the within-protein tokenisation contrast cannot be formed"
        )
    else:
        summary["runnable"] = True
    for key in keys:
        left = None if bpe is None else bpe[key]["mean"]
        right = None if residue is None else residue[key]["mean"]
        summary["difference_bpe_minus_residue"][key] = (
            None if left is None or right is None else left - right
        )
    return summary


def conditioning_control(
    frame: Sequence[Mapping[str, Any]],
    *,
    keys: Sequence[str],
) -> dict[str, Any]:
    """Conditioned against unconditional protein rungs, reported apart.

    ZymCTRL's prompt supplies an EC label; ProtGPT2 and the ProGen2 rungs are
    given nothing. Information delivered by the prompt is information the model
    does not have to recover from context, and realized information fraction is
    measured on exactly that recovery, so a conditioned rung's position on the
    convergence axis is not on the same footing as an unconditional one's. Any
    reading that puts them on one curve - in either direction - is reading across
    that difference, which is why the split is reported rather than folded in.
    """

    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in frame:
        if row["modality"] != "protein":
            continue
        groups.setdefault(str(row["conditioning"]), []).append(row)
    summary: dict[str, Any] = {
        "groups": {},
        "caveat": (
            "an EC-conditioned prompt supplies information the unconditional arms are "
            "not given, so realized information fraction is not comparable across this "
            "split without accounting for it"
        ),
    }
    for conditioning, rows in sorted(groups.items()):
        entry: dict[str, Any] = {
            "n": len(rows),
            "members": [row["name"] for row in rows],
            "members_in_distribution": [
                row["name"] for row in rows if row["in_distribution"]
            ],
        }
        usable = [row for row in rows if row["in_distribution"]]
        for key in keys:
            values = [float(row[key]) for row in usable if row[key] is not None]
            entry[key] = {
                "n": len(values),
                "mean": float(np.mean(values)) if values else None,
                "interval": mean_interval(values) if len(values) >= 2 else None,
            }
        summary["groups"][conditioning] = entry
    return summary


def paired_architecture_contrast(
    frame: Sequence[Mapping[str, Any]],
    *,
    keys: Sequence[str],
) -> dict[str, Any]:
    """Rungs identical in architecture and parameter count, contrasted pairwise.

    ProGen2-base and ProGen2-medium are 27 layers, 1536 wide and 764,803,616
    parameters each; in the published family they differ in pretraining corpus and
    in nothing else. That is a natural experiment on the exact confound this
    control exists to address, and it needs no modality contrast to be
    informative: if an interpretability metric moves appreciably between two
    architecturally identical models that differ only in training data, the metric
    tracks data rather than modality; if it barely moves while a text-protein
    offset persists, it does not.

    The grouping is on measured architecture rather than on names, so any future
    same-shape pair is picked up without being anticipated here. Pairs are
    reported raw: with two members there is no interval to attach and none is
    invented.
    """

    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in frame:
        signature = (
            str(row["modality"]),
            int(row["n_layer"]),
            int(row["d_model"]),
            int(row["n_parameters"]),
        )
        groups.setdefault(signature, []).append(row)

    pairs: list[dict[str, Any]] = []
    for signature, rows in sorted(groups.items()):
        if len(rows) < 2:
            continue
        ordered = sorted(rows, key=lambda item: str(item["name"]))
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                differences: dict[str, Any] = {}
                for key in keys:
                    if left[key] is None or right[key] is None:
                        differences[key] = None
                        continue
                    differences[key] = {
                        left["name"]: float(left[key]),
                        right["name"]: float(right[key]),
                        "difference": float(right[key]) - float(left[key]),
                    }
                pairs.append(
                    {
                        "modality": signature[0],
                        "n_layer": signature[1],
                        "d_model": signature[2],
                        "n_parameters": signature[3],
                        "members": [left["name"], right["name"]],
                        "difference_direction": f"{right['name']} minus {left['name']}",
                        "same_tokenisation": left["tokenisation"] == right["tokenisation"],
                        "same_cohort_digest": left["cohort_digest"] == right["cohort_digest"],
                        "differences": differences,
                    }
                )
    return {
        "interpretation": (
            "architecture and parameter count are held exactly fixed, so a difference "
            "here is attributable to pretraining data rather than to scale, modality "
            "or tokenisation"
        ),
        "n_pairs": len(pairs),
        "pairs": pairs,
        "note": (
            None
            if pairs
            else "no two measured rungs share a modality, depth, width and parameter count"
        ),
    }


def distribution_control(frame: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Which models were on their own distribution, and what that invalidates.

    An off-distribution model is not a weak result; it is an unusable one. Any
    earlier comparison that used such a model is invalidated by this table, so it
    is reported as a first-class output rather than as a footnote to the fits.
    """

    rows = [
        {
            "name": row["name"],
            "modality": row["modality"],
            "cohort_corpus": row["cohort_corpus"],
            "cohort_band_symbols": [row["cohort_min_symbols"], row["cohort_max_symbols"]],
            "cohort_digest": row["cohort_digest"],
            "in_distribution": bool(row["in_distribution"]),
            "context_information_nats": float(row["context_information_nats"]),
            "realized_information_fraction": float(row["realized_information_fraction"]),
        }
        for row in frame
    ]
    excluded = [row["name"] for row in rows if not row["in_distribution"]]
    return {
        "rule": (
            "clean_ce_nats < unigram_entropy_on_cohort_nats on the model's own native "
            "cohort; the corpus and length band the rule was applied on travel with "
            "each row, because the verdict is band-dependent"
        ),
        "rows": rows,
        "excluded_from_fits": excluded,
        "n_excluded": len(excluded),
        "consequence": (
            "an excluded model divides every normalised interpretability figure by a "
            "non-positive denominator; any comparison using it is invalid, not merely weak"
        ),
    }


def decide_verdict(
    fits: Mapping[str, Mapping[str, Any]],
    *,
    tokenisation_adjusted_fits: Mapping[str, Mapping[str, Any]],
    primary_metric: str = PRIMARY_METRIC,
    primary_axis: str = PRIMARY_AXIS,
    equivalence_margin: float = DEFAULT_EQUIVALENCE_MARGIN,
    min_residual_dof: int = DEFAULT_MIN_RESIDUAL_DOF,
) -> dict[str, Any]:
    """Map the primary fits onto one of :data:`VERDICTS`, by a rule fixed in advance.

    The rule is deliberately hard to satisfy in both directions.
    ``residual_modality_gap`` - the reading that supports the programme's
    hypothesis - requires the modality coefficient's interval to exclude zero.
    ``gap_explained_by_convergence`` - the reading that overturns it - requires
    that interval to both contain zero and be narrower than a margin declared
    before the data. An interval that contains zero because it is enormous
    supports neither, and returns ``underpowered``, which is also what an absent
    or unreadable fit returns.

    Both readings must hold in the fit that adjusts for tokenisation as well as in
    the one that does not. A coefficient that only survives while tokenisation is
    omitted from the design is a tokenisation coefficient: across this panel the
    text rungs are subword, ProtGPT2 is protein and subword, and the remaining
    protein rungs are symbol-level, so a modality indicator alone will absorb a
    tokenisation effect wherever one exists. Requiring both is the same principle
    :func:`identification_check` applies structurally, applied quantitatively, and
    it blocks the two non-default verdicts equally.

    "Holding in both" is holding in the same direction. ``residual_modality_gap``
    requires the two intervals to exclude zero on the same side of it, because a
    coefficient that is positive in one design and negative in the other has
    reversed sign rather than been confirmed twice; that returns ``underpowered``
    naming the reversal.
    """

    if equivalence_margin <= 0.0:
        raise ValueError("the equivalence margin must be positive")
    if min_residual_dof < 1:
        raise ValueError("at least one residual degree of freedom must be required")
    key = f"{primary_metric}~{primary_axis}"
    decision: dict[str, Any] = {
        "verdict": "underpowered",
        "primary_fit": key,
        "equivalence_margin": float(equivalence_margin),
        "minimum_residual_dof": int(min_residual_dof),
        "reason": None,
    }
    readings: dict[str, tuple[float, float]] = {}
    for label, family in (
        ("unadjusted", fits),
        ("tokenisation_adjusted", tokenisation_adjusted_fits),
    ):
        fit = family.get(key)
        if fit is None:
            decision["reason"] = f"the {label} primary fit {key!r} was not produced"
            return decision
        if not fit["fitted"]:
            decision["reason"] = (
                f"the {label} primary fit was refused: {fit['unfitted_reason']}"
            )
            return decision
        if int(fit["residual_dof"]) < min_residual_dof:
            decision["reason"] = (
                f"the {label} primary fit has {fit['residual_dof']} residual degrees of "
                f"freedom, below the {min_residual_dof} declared in advance"
            )
            return decision
        if fit["saturated"]:
            decision["reason"] = (
                f"the {label} primary fit has zero residual variance, so its standard "
                "errors are not estimable"
            )
            return decision
        identification = fit["identification"]
        decision.setdefault("identification", identification)
        if not identification["modality_identified"]:
            decision["reason"] = (
                f"the modality coefficient is not identified: {identification['reason']}"
            )
            return decision
        offset = fit["coefficients"]["protein_offset"]
        low, high = (float(value) for value in offset["interval"])
        readings[label] = (low, high)
        # Published so that a reader can re-apply any margin other than the one
        # declared here without refitting; the rule is fixed, the evidence is not
        # hidden behind it.
        decision[f"protein_offset_{label}"] = offset
        decision[f"interval_half_width_{label}"] = (high - low) / 2.0

    # Which side of zero each interval sits on, not merely whether it misses it.
    # Two intervals on opposite sides both "exclude zero", so a rule reading only
    # that would return ``residual_modality_gap`` on a coefficient that reversed
    # direction when tokenisation entered the design -- one fit saying protein
    # decoders are above text at matched convergence and the other saying they
    # are below. That is not one effect confirmed twice, and the blind spot opens
    # only toward this programme's own hypothesis.
    sides = {
        label: 1 if low > 0.0 else -1 if high < 0.0 else 0
        for label, (low, high) in readings.items()
    }
    excludes_zero = {label: side != 0 for label, side in sides.items()}
    if all(excludes_zero.values()):
        if len(set(sides.values())) > 1:
            decision["reason"] = (
                "the modality coefficient excludes zero in both fits but on opposite "
                f"sides of it - unadjusted {list(np.round(readings['unadjusted'], 4))} "
                "against tokenisation-adjusted "
                f"{list(np.round(readings['tokenisation_adjusted'], 4))}; a coefficient "
                "that reverses sign when tokenisation enters the design cannot be "
                "attributed to modality in either direction"
            )
            return decision
        decision["verdict"] = "residual_modality_gap"
        decision["reason"] = (
            "the modality coefficient's interval excludes zero at matched convergence "
            f"both unadjusted {list(np.round(readings['unadjusted'], 4))} and with "
            f"tokenisation held fixed {list(np.round(readings['tokenisation_adjusted'], 4))}"
        )
        return decision
    if any(excludes_zero.values()):
        survivor = "unadjusted" if excludes_zero["unadjusted"] else "tokenisation_adjusted"
        casualty = "tokenisation_adjusted" if survivor == "unadjusted" else "unadjusted"
        decision["reason"] = (
            f"the modality coefficient excludes zero in the {survivor} fit "
            f"{list(np.round(readings[survivor], 4))} but not in the {casualty} one "
            f"{list(np.round(readings[casualty], 4))}; a coefficient that depends on "
            "whether tokenisation is in the design cannot be attributed to modality"
        )
        return decision
    half_widths = {
        label: (high - low) / 2.0 for label, (low, high) in readings.items()
    }
    # Attainability, reported rather than left to be inferred. The margin is the
    # only route to ``gap_explained_by_convergence``, so whether this design can
    # produce a half-width that small is a property of the design and belongs in
    # the artefact beside the verdict it gates. The rung multiple assumes the
    # half-width falls as one over the square root of the usable rungs; it
    # ignores the t-quantile's own fall with residual degrees of freedom and is
    # therefore an over-estimate of what would be needed.
    widest = max(half_widths.values())
    decision["equivalence_margin_attainability"] = {
        "margin": float(equivalence_margin),
        "half_widths": {label: float(value) for label, value in half_widths.items()},
        "widest_half_width": float(widest),
        "attained": bool(widest <= equivalence_margin),
        "usable_rung_multiple_required": float((widest / equivalence_margin) ** 2),
        "note": (
            "gap_explained_by_convergence is reachable only when both half-widths "
            "sit inside the margin; where they do not, 'underpowered' is a "
            "statement about this ladder's width and must not be read as evidence "
            "for either the convergence or the residual-gap reading"
        ),
    }
    if all(width <= equivalence_margin for width in half_widths.values()):
        decision["verdict"] = "gap_explained_by_convergence"
        decision["reason"] = (
            "the modality coefficient's interval contains zero in both the unadjusted "
            f"{list(np.round(readings['unadjusted'], 4))} and the tokenisation-adjusted "
            f"{list(np.round(readings['tokenisation_adjusted'], 4))} fit, and both "
            f"half-widths are inside the declared margin {equivalence_margin:.4f}"
        )
        return decision
    decision["reason"] = (
        "the modality coefficient's interval contains zero in both fits - unadjusted "
        f"{list(np.round(readings['unadjusted'], 4))}, tokenisation-adjusted "
        f"{list(np.round(readings['tokenisation_adjusted'], 4))} - but the widest "
        f"half-width {widest:.4f} exceeds the declared margin "
        f"{equivalence_margin:.4f}, so no offset is excluded. The margin is "
        f"{widest / equivalence_margin:.1f}x away, which is roughly "
        f"{(widest / equivalence_margin) ** 2:.0f}x this ladder's usable rungs: the "
        "equivalence verdict is not attainable on this design and its absence is "
        "not evidence against convergence"
    )
    return decision
