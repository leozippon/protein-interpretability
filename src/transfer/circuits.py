"""Pre-dictionary circuit primitives, measured identically on text and protein decoders.

The circuit toolkit that predates dictionary learning -- the induction-head
census, direct logit attribution and activation patching -- is what a
mechanistic-interpretability practitioner reaches for first.  Each of the three
carries an assumption that was calibrated on natural language and is never
restated when the toolkit is pointed at a protein decoder:

``induction``   that sequences contain literal repeats, so that a head can be
                identified by prefix matching and its OV circuit read as copying;
``attribution`` that a component's contribution has a readable image under the
                unembedding, which is a much stronger claim when the output
                alphabet is twenty residues than when it is fifty thousand BPE
                pieces;
``patching``    that replacing one token produces a downstream logit change that
                is large enough to measure and localised enough to attribute.

This module measures each assumption on the matched panel from :mod:`.arms`, so
that a transfer failure can be attributed to a specific broken assumption rather
than to "protein models are different".  Every approximation is stated in the
docstring of the function that makes it, and nothing degrades silently: a probe
that cannot be constructed raises instead of returning a weaker probe.

The ``induction`` assumption is carried by two natural-repeat probes rather than
one, and the difference between them is itself a measurement.  Pomerants,
Nikankin, Reusch, Tsaban, Schueler-Furman and Belinkov, "Induction Meets Biology:
Mechanisms of Repeat Detection in Protein Language Models", arXiv:2602.23179 v5,
show on masked protein encoders that approximate-repeat detection functionally
subsumes exact-repeat detection -- the approximate circuit generalises to
identical repeats with cross-task faithfulness above 1.0, while circuits fitted
on exact repeats do not recover approximate-repeat performance -- and that the
substitution tolerance is carried in part by neurons encoding BLOSUM62
substitution groups.  An exact-repeat probe therefore measures a special case of
the mechanism, and it is a special case that text under BPE supplies far more
readily than protein sequence does.  :data:`PROTEIN_APPROXIMATE_CRITERION` and
:data:`TEXT_APPROXIMATE_CRITERION` add the general case; the exact criteria are
kept unchanged so that the comparison between the two probes is the evidence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from typing import Any

import numpy as np
import torch

from .statistics import MINIMUM_BOOTSTRAP_UNITS
from .arms import (
    AA20,
    Arm,
    Cohort,
    ZYMCTRL_FASTA,
    iter_fasta,
    sampling_record,
    selected_positions,
    text_cohort,
)

SCHEMA_VERSION = "r2_transfer_circuit_primitives_v2"

#: Component kinds patched in :func:`activation_patching`.  ``resid_post`` is the
#: residual stream leaving a block, so patching it carries everything up to and
#: including that layer; the two sublayer kinds isolate one pathway.
COMPONENT_KINDS = ("attn_out", "mlp_out", "resid_post")

#: Inclusive ``q - p`` bands for the patching distance sweep.
DISTANCE_BANDS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (2, 4),
    (5, 8),
    (9, 16),
    (17, 32),
    (33, 64),
)

#: Smallest ``|signed total| / total magnitude`` at which a *signed* pathway
#: fraction is reported. Below it the pathways have cancelled and the fraction is
#: a large numerator over a residual denominator, which is the arm-dependent
#: denominator failure the programme has already paid for once. One per mille is
#: three orders of magnitude below the smallest share worth reading and is
#: declared here rather than chosen per arm.
SIGNED_FRACTION_MINIMUM_RATIO = 1e-3

#: Prefix-matching scores above these values are counted as induction heads.
#: Several fixed cut-offs are reported because no single value is principled;
#: a data-driven cut-off relative to each arm's own head distribution is
#: reported alongside them.
INDUCTION_THRESHOLDS: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30)


# ------------------------------------------------------------------- BLOSUM62

#: Published row and column order of the BLOSUM62 half-bit substitution matrix.
BLOSUM62_ORDER = "ARNDCQEGHILKMFPSTWYV"

#: BLOSUM62 as distributed by NCBI, restricted to the twenty standard residues.
#: Transcribed rather than loaded from a library because the substitution
#: tolerance of the approximate-repeat probe is defined by these numbers and must
#: not depend on which optional package happens to be installed on a host.
#: :func:`_build_blosum62` re-derives symmetry and a set of published entries at
#: import, so a transcription slip fails immediately instead of quietly changing
#: which substitutions the probe accepts.
BLOSUM62_ROWS: tuple[str, ...] = (
    "  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0",
    " -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3",
    " -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3",
    " -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3",
    "  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1",
    " -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2",
    " -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2",
    "  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3",
    " -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3",
    " -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3",
    " -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1",
    " -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2",
    " -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1",
    " -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1",
    " -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2",
    "  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2",
    "  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0",
    " -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3",
    " -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1",
    "  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4",
)

#: A handful of published BLOSUM62 entries, checked at import.  These are the
#: values that carry the biology of the criterion: the two extreme diagonals, the
#: aliphatic and aromatic conservative pairs, and two substitutions the matrix
#: penalises heavily.
_BLOSUM62_PUBLISHED: tuple[tuple[str, str, int], ...] = (
    ("A", "A", 4),
    ("W", "W", 11),
    ("C", "C", 9),
    ("I", "V", 3),
    ("L", "M", 2),
    ("F", "Y", 3),
    ("K", "R", 2),
    ("D", "E", 2),
    ("G", "W", -2),
    ("P", "W", -4),
)


def _build_blosum62() -> np.ndarray:
    """BLOSUM62 reindexed onto :data:`~.arms.AA20`, validated against publication."""

    raw = np.array(
        [[int(value) for value in row.split()] for row in BLOSUM62_ROWS], dtype=np.int64
    )
    if raw.shape != (len(BLOSUM62_ORDER), len(BLOSUM62_ORDER)):
        raise ValueError("BLOSUM62 transcription is not a square 20x20 table")
    if not (raw == raw.T).all():
        raise ValueError("BLOSUM62 transcription is not symmetric")
    if sorted(BLOSUM62_ORDER) != sorted(AA20):
        raise ValueError("BLOSUM62 row order does not cover the canonical alphabet")
    order = [BLOSUM62_ORDER.index(residue) for residue in AA20]
    matrix = raw[np.ix_(order, order)]
    for left, right, expected in _BLOSUM62_PUBLISHED:
        observed = int(matrix[AA20.index(left), AA20.index(right)])
        if observed != expected:
            raise ValueError(
                f"BLOSUM62[{left},{right}] transcribed as {observed}, published value {expected}"
            )
    return matrix


#: BLOSUM62 indexed by position in :data:`~.arms.AA20`.
BLOSUM62 = _build_blosum62()

#: Byte value to :data:`~.arms.AA20` index, with -1 for everything else.
_RESIDUE_INDEX = np.full(256, -1, dtype=np.int64)
for _position, _residue in enumerate(AA20):
    _RESIDUE_INDEX[ord(_residue)] = _position


# --------------------------------------------------------------------- basics


def _finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def n_head(arm: Arm) -> int:
    """Number of *query* heads.

    Under grouped-query attention this exceeds the number of key/value heads, and
    it is the query count that indexes an attention pattern and therefore every
    per-head score in this module.
    """

    config = arm.model.config
    heads = getattr(config, "n_head", None) or getattr(config, "num_attention_heads", None)
    if heads is None:
        raise TypeError(f"{arm.name}: config declares no attention-head count")
    heads = int(heads)
    if heads < 1:
        raise ValueError(f"{arm.name}: non-positive attention-head count {heads}")
    return heads


def head_dim(arm: Arm) -> int:
    """Per-head width, taken from the config when it declares one.

    ``d_model / n_head`` is an inference, not a definition.  It is right for
    every arm currently in the panel, but Qwen3 declares a ``head_dim`` that
    differs from it, so preferring the declaration turns a future silent
    mis-slicing of the value projection into an immediate shape check.
    """

    declared = getattr(arm.model.config, "head_dim", None)
    if declared:
        return int(declared)
    heads = n_head(arm)
    if arm.d_model % heads != 0:
        raise ValueError(
            f"{arm.name}: config declares no head_dim and {arm.d_model}d is not "
            f"divisible by {heads} query heads"
        )
    return arm.d_model // heads


def n_key_value_head(arm: Arm) -> int:
    """Number of key/value heads; equal to the query count outside GQA."""

    declared = getattr(arm.model.config, "num_key_value_heads", None)
    heads = n_head(arm)
    groups = int(declared) if declared else heads
    if groups < 1 or heads % groups != 0:
        raise ValueError(
            f"{arm.name}: {heads} query heads do not divide into {groups} key/value heads"
        )
    return groups


def _query_to_key_value(arm: Arm, device: torch.device) -> torch.Tensor:
    """Index mapping each query head to the key/value head it reads."""

    heads = n_head(arm)
    groups = n_key_value_head(arm)
    return torch.arange(heads, device=device) // (heads // groups)


#: Architectures whose per-head circuit decomposition this module implements, and
#: the module-naming convention each follows.  Resolution is by declaration
#: rather than by attribute search: an architecture this module has not been
#: taught must fail here, not resolve to whichever attribute happened to exist on
#: it and yield a number that looks like every other number in the table.
_GPT_STYLE = frozenset({"gpt2", "progen"})
_ROTARY_STYLE = frozenset({"llama", "qwen2"})
_CIRCUIT_ARCHITECTURES = _GPT_STYLE | _ROTARY_STYLE


def circuit_architecture(arm: Arm) -> str:
    """The arm's architecture, refused unless a per-head decomposition exists."""

    architecture = arm.spec.architecture
    if architecture not in _CIRCUIT_ARCHITECTURES:
        raise TypeError(
            f"{arm.name}: per-head circuit decomposition is not defined for "
            f"{architecture!r}; implemented: {sorted(_CIRCUIT_ARCHITECTURES)}"
        )
    return architecture


def _inner_decoder(arm: Arm) -> torch.nn.Module:
    """The module holding the embedding table, block list and final norm."""

    architecture = circuit_architecture(arm)
    holder = "transformer" if architecture in _GPT_STYLE else "model"
    inner = getattr(arm.model, holder, None)
    if inner is None:
        raise TypeError(f"{arm.name}: declared {architecture} but no model.{holder}")
    return inner


def pre_attention_norm(arm: Arm, layer: int) -> torch.nn.Module:
    """The normalisation an attention sublayer reads, per architecture."""

    architecture = circuit_architecture(arm)
    attribute = "ln_1" if architecture in _GPT_STYLE else "input_layernorm"
    block = arm.blocks()[layer]
    if not hasattr(block, attribute):
        raise TypeError(
            f"{arm.name}: declared {architecture} but block {layer} has no {attribute}"
        )
    return getattr(block, attribute)


def final_norm(arm: Arm) -> torch.nn.Module:
    """The normalisation applied to the residual stream before the unembedding."""

    architecture = circuit_architecture(arm)
    attribute = "ln_f" if architecture in _GPT_STYLE else "norm"
    inner = _inner_decoder(arm)
    if not hasattr(inner, attribute):
        raise TypeError(f"{arm.name}: declared {architecture} but no final {attribute}")
    return getattr(inner, attribute)


def embedding_module(arm: Arm) -> torch.nn.Module:
    """Module whose output is the residual stream entering block zero.

    GPT-2 adds a learned position table to the token embedding and passes the sum
    through ``transformer.drop``, so that dropout module's output is the whole
    initial residual.  A rotary decoder puts no position information in the
    residual stream at all -- position enters inside attention, on the queries and
    keys -- so its token embedding *is* the initial residual.
    """

    architecture = circuit_architecture(arm)
    attribute = "drop" if architecture in _GPT_STYLE else "embed_tokens"
    inner = _inner_decoder(arm)
    if not hasattr(inner, attribute):
        raise TypeError(f"{arm.name}: declared {architecture} but no {attribute}")
    return getattr(inner, attribute)


def embedding_weight(arm: Arm) -> torch.Tensor:
    """The token embedding matrix, shape ``(vocab, d_model)``."""

    architecture = circuit_architecture(arm)
    attribute = "wte" if architecture in _GPT_STYLE else "embed_tokens"
    inner = _inner_decoder(arm)
    if not hasattr(inner, attribute):
        raise TypeError(f"{arm.name}: declared {architecture} but no {attribute}")
    return getattr(inner, attribute).weight


@dataclass(frozen=True)
class NormalisationForm:
    """How a final normalisation must be linearised for attribution.

    LayerNorm centres its input and adds a learned bias; RMSNorm does neither.
    Applying LayerNorm's form to an RMSNorm decoder would subtract a mean the
    model never subtracts and add a bias it does not have, which makes the
    attribution wrong rather than merely approximate -- and wrong in a way that
    still sums to something, so it would not announce itself.
    """

    gain: torch.Tensor
    bias: torch.Tensor | None
    epsilon: float
    centred: bool


def normalisation_form(module: torch.nn.Module, label: str) -> NormalisationForm:
    """Classify a normalisation module into the form attribution must assume."""

    if isinstance(module, torch.nn.LayerNorm):
        if module.bias is None:
            raise TypeError(f"{label}: LayerNorm without a bias is not handled")
        return NormalisationForm(
            gain=module.weight.detach().float(),
            bias=module.bias.detach().float(),
            epsilon=float(module.eps),
            centred=True,
        )
    epsilon = getattr(module, "variance_epsilon", None)
    if epsilon is not None and hasattr(module, "weight"):
        if getattr(module, "bias", None) is not None:
            raise TypeError(f"{label}: RMSNorm with a bias is not handled")
        return NormalisationForm(
            gain=module.weight.detach().float(),
            bias=None,
            epsilon=float(epsilon),
            centred=False,
        )
    raise TypeError(f"{label}: unsupported normalisation {type(module).__name__}")


def sequence_start_id(arm: Arm) -> int:
    """The token a document begins after.

    Qwen2's tokenizer declares no begin-of-sequence token, but its pretraining
    stream separates documents with the end-of-text token, so that token is the
    document-start anchor for it in exactly the sense ``bos_token_id`` is for
    GPT-2 and Llama.  This is one rule with two spellings, not a fallback: the
    probe needs the token the model saw at a document boundary, and an arm that
    declares neither has no such token and raises.
    """

    for candidate in (arm.tokenizer.bos_token_id, arm.tokenizer.eos_token_id):
        if candidate is not None:
            return int(candidate)
    raise ValueError(f"{arm.name}: tokenizer declares neither a BOS nor an end-of-text token")


def content_bounds(arm: Arm, ids: Sequence[int], n_valid: int) -> tuple[int, int]:
    """Half-open token span holding modality content, excluding format scaffolding.

    Positions outside this span are EC tags, control tokens and boundary markers.
    They are excluded everywhere so that a protein arm is never scored on its
    prompt syntax while the text arm is scored on words.
    """

    if n_valid < 1 or n_valid > len(ids):
        raise ValueError(f"{arm.name}: invalid token count {n_valid} for row of {len(ids)}")
    fmt = arm.spec.input_format
    if fmt == "raw":
        # Llama's tokenizer prepends a begin-of-text token; GPT-2's and Qwen2's
        # do not. That is a property of the tokenizer rather than of the format,
        # so it is read off the row that was actually produced instead of being
        # declared per arm -- scoring the marker as content would charge one arm
        # for predicting the first word of a document from nothing.
        start = arm.tokenizer.bos_token_id
        if start is not None and n_valid > 1 and int(ids[0]) == int(start):
            return 1, n_valid
        return 0, n_valid
    if fmt == "n_to_c_control":
        # ProGen2's N-to-C control marker is one token.
        return 1, n_valid
    if fmt == "fasta_wrapped":
        eos = arm.tokenizer.eos_token_id
        if eos is None or int(ids[0]) != int(eos):
            raise ValueError(f"{arm.name}: FASTA rendering does not start with end-of-text")
        low = 1
        saw_line_break = False
        while low < n_valid:
            decoded = arm.tokenizer.decode([int(ids[low])])
            if decoded and all(character in "\r\n" for character in decoded):
                saw_line_break = True
                low += 1
                continue
            break
        if not saw_line_break:
            raise ValueError(f"{arm.name}: FASTA rendering has no line break after end-of-text")
        if low >= n_valid:
            raise ValueError(f"{arm.name}: FASTA rendering contains no sequence content")
        return low, n_valid
    if fmt == "ec_conditioned":
        vocabulary = arm.tokenizer.get_vocab()
        for marker in ("<start>", "<end>"):
            if marker not in vocabulary:
                raise ValueError(f"{arm.name}: tokenizer lacks the {marker!r} marker")
        start_id = vocabulary["<start>"]
        end_id = vocabulary["<end>"]
        row = list(ids[:n_valid])
        if row.count(start_id) != 1:
            raise ValueError(f"{arm.name}: row does not contain exactly one <start>")
        # Both boundaries are required, and required to be unique. This used to
        # read `high = row.index(end_id) if end_id in row else n_valid`, so a row
        # truncated before its <end> had everything to the end of the valid span
        # scored as cohort content. `scoring.sequence_target_mask` raises on that
        # row and `budget.scored_tokens` raises on it with the truncation named,
        # which left the panel with two answers to "what is scored" and the
        # silent one on the arm whose conditioning tag is priced at 1.73 nats.
        if row.count(end_id) != 1:
            raise ValueError(
                f"{arm.name}: row does not contain exactly one <end>; a conditioned "
                "row truncated before its boundary has no defined content span, and "
                "scoring to the end of the valid tokens would count the prompt as "
                "cohort content"
            )
        low = row.index(start_id) + 1
        high = row.index(end_id)
        if high <= low:
            raise ValueError(f"{arm.name}: empty content span between <start> and <end>")
        return low, high
    raise ValueError(f"{arm.name}: unsupported input format {fmt!r}")


def prefix_ids(arm: Arm, *, ec_label: str | None = None) -> list[int]:
    """Arm-native prompt prefix that a synthetic probe must start from.

    A synthetic probe is built directly in token space, so it has to reproduce
    by hand the conditioning that :meth:`Cohort.input_strings` produces for real
    records; otherwise the protein arms are probed off their own input format.
    """

    fmt = arm.spec.input_format
    if fmt == "raw":
        return [sequence_start_id(arm)]
    if fmt == "fasta_wrapped":
        eos = arm.tokenizer.eos_token_id
        if eos is None:
            raise ValueError(f"{arm.name}: tokenizer defines no end-of-text token")
        return [int(eos)]
    if fmt == "n_to_c_control":
        return [int(i) for i in arm.tokenizer("1", return_tensors=None)["input_ids"]]
    if fmt == "ec_conditioned":
        if ec_label is None:
            raise ValueError(f"{arm.name}: an EC label is required to build a prompt prefix")
        encoded = arm.tokenizer(f"{ec_label}<sep><start>", return_tensors=None)["input_ids"]
        return [int(i) for i in encoded]
    raise ValueError(f"{arm.name}: unsupported input format {fmt!r}")


# ------------------------------------------------------------------- unigrams


@dataclass(frozen=True)
class Unigram:
    """Empirical token distribution over an arm's own content tokens.

    Synthetic probes and patching corruptions both need "a different token that
    this model could plausibly have seen here".  Sampling uniformly from the
    vocabulary would put a protein arm far off distribution and a text arm on
    junk BPE pieces, so both draw from this instead.
    """

    token_ids: np.ndarray
    counts: np.ndarray
    total_tokens: int
    scored_sequences: int
    layout_tokens_excluded: int = 0
    layout_mass_excluded: float = 0.0

    def __post_init__(self) -> None:
        if self.token_ids.ndim != 1 or self.token_ids.shape != self.counts.shape:
            raise ValueError("unigram support and counts must be aligned vectors")
        if self.token_ids.size < 2:
            raise ValueError("unigram needs at least two distinct tokens")
        if self.counts.min() < 1:
            raise ValueError("unigram counts must be positive")

    @property
    def probabilities(self) -> np.ndarray:
        return self.counts.astype(np.float64) / float(self.counts.sum())

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        if size < 1:
            raise ValueError("sample size must be positive")
        return rng.choice(self.token_ids, size=size, replace=True, p=self.probabilities)

    def sample_other(self, rng: np.random.Generator, exclude: int, attempts: int = 64) -> int:
        """Draw one token that differs from ``exclude``."""

        for _ in range(attempts):
            candidate = int(self.sample(rng, 1)[0])
            if candidate != exclude:
                return candidate
        raise RuntimeError(f"could not draw a token different from {exclude} in {attempts} tries")

    def summary(self) -> dict[str, Any]:
        probabilities = self.probabilities
        entropy = float(-(probabilities * np.log(probabilities)).sum())
        return {
            "support_size": int(self.token_ids.size),
            "total_tokens": int(self.total_tokens),
            "scored_sequences": int(self.scored_sequences),
            "entropy_nats": _finite(entropy, "unigram entropy"),
            "top_token_probability": _finite(float(probabilities.max()), "unigram top mass"),
            "layout_tokens_excluded": int(self.layout_tokens_excluded),
            "layout_mass_excluded": _finite(
                float(self.layout_mass_excluded), "unigram layout mass"
            ),
        }


#: Slack over the residue count for a conditioned rendering's own tokens: an EC
#: tag, ``<sep>``, ``<start>`` and ``<end>``. Measured at nine tokens for
#: ZymCTRL's longest tag; thirty-two is a bound, not a fit.
CONDITIONING_TOKEN_SLACK = 32


def conditioned_token_budget(arm: Arm, requested: int, max_symbols: int) -> int:
    """A unigram window wide enough to keep a conditioned rendering's ``<end>``.

    An ``ec_conditioned`` arm wraps its residues in ``<start>`` and ``<end>``, and
    :func:`content_bounds` refuses a row whose ``<end>`` was truncated away --
    correctly, because scoring to the end of the valid tokens would count the
    conditioning prompt as cohort content. But the stages' defaults contradict
    each other: a 256-token unigram window against a 1000-residue protein band
    puts ZymCTRL's rows at 621-816 tokens, so they lose their ``<end>`` and the run
    dies inside :func:`fit_unigram`.

    Declared here rather than in a stage. It was first written inside
    ``04_circuit_primitives.py``, and ``11_induction_path_patching.py`` then failed
    the same way on the same arm for the same reason -- a second copy of a decision
    that had been made properly one import away, which is Appendix B rule 12.

    Resolved **per arm** rather than refused for the group: a campaign dispatches
    these stages as one process over every eligible arm, so an argument-time
    refusal would lose the arms that were fine along with the one that was not.
    Widening one arm's window changes no other arm's unigram.
    """

    if arm.spec.input_format != "ec_conditioned":
        return int(requested)
    return max(int(requested), int(max_symbols) + CONDITIONING_TOKEN_SLACK)


def fit_unigram(arm: Arm, strings: Sequence[str], *, max_tokens: int) -> Unigram:
    """Count content tokens over arm-native input strings.

    Tokens that carry a line break are dropped from the support.  They are real
    content tokens -- ProtGPT2's FASTA rendering emits one every sixty residues
    and they are 5% of its stream -- but this distribution exists only to draw
    replacement tokens for synthetic probes and patching corruptions, and
    dropping a line break at an arbitrary position perturbs a record's *layout*
    rather than its sequence.  Leaving them in would give the one arm whose
    rendering has layout tokens a systematically different perturbation from
    every other arm.  The excluded count and mass are reported, and nothing that
    scores real inputs uses this filter.
    """

    if not strings:
        raise ValueError(f"{arm.name}: cannot fit a unigram on an empty cohort")
    if max_tokens < 2:
        raise ValueError("max_tokens must be at least two")
    counter: dict[int, int] = {}
    total = 0
    for text in strings:
        row = arm.tokenizer(text, return_tensors=None)["input_ids"][:max_tokens]
        low, high = content_bounds(arm, row, len(row))
        for token in row[low:high]:
            counter[int(token)] = counter.get(int(token), 0) + 1
            total += 1
    if total < 1:
        raise RuntimeError(f"{arm.name}: cohort produced no content tokens")
    layout = {
        token
        for token in counter
        if any(character in arm.tokenizer.decode([token]) for character in ("\n", "\r"))
    }
    layout_mass = sum(counter[token] for token in layout)
    for token in layout:
        del counter[token]
    if not counter:
        raise RuntimeError(f"{arm.name}: cohort produced no non-layout content tokens")
    order = sorted(counter, key=lambda token: (-counter[token], token))
    return Unigram(
        token_ids=np.asarray(order, dtype=np.int64),
        counts=np.asarray([counter[token] for token in order], dtype=np.int64),
        total_tokens=total - layout_mass,
        scored_sequences=len(strings),
        layout_tokens_excluded=len(layout),
        layout_mass_excluded=layout_mass / total,
    )


# --------------------------------------------------------------- repeat probes


def find_internal_repeat(
    symbols: str,
    *,
    min_unit: int,
    max_gap_ratio: float,
    min_distinct: int,
) -> tuple[int, int, int] | None:
    """Longest exact internal repeat ``(first_start, second_start, length)``.

    Two occurrences are required to be non-overlapping and close enough that the
    pair is a tandem repeat rather than an incidental coincidence:
    ``second_start - first_start <= max_gap_ratio * length``.  ``min_distinct``
    rejects low-complexity runs, where matching "the token after the earlier
    occurrence" is satisfied by attending almost anywhere and the prefix-matching
    score would be inflated without any induction circuit being present.
    """

    if min_unit < 2 or max_gap_ratio < 1.0 or min_distinct < 1:
        raise ValueError("invalid internal-repeat search parameters")
    length = len(symbols)
    if length < 2 * min_unit:
        return None
    first_seen: dict[str, int] = {}
    best: tuple[int, int, int] | None = None
    for start in range(length - min_unit + 1):
        key = symbols[start : start + min_unit]
        earlier = first_seen.get(key)
        if earlier is None:
            first_seen[key] = start
            continue
        if start - earlier < min_unit:
            continue
        span = min_unit
        while start + span < length and earlier + span < start and symbols[earlier + span] == symbols[start + span]:
            span += 1
        if start - earlier > max_gap_ratio * span:
            continue
        if len(set(symbols[earlier : earlier + span])) < min_distinct:
            continue
        if best is None or span > best[2]:
            best = (earlier, start, span)
    return best


# -------------------------------------------------- approximate repeat criteria


@dataclass(frozen=True)
class RepeatCriterion:
    """What a record must contain to enter a natural-repeat cohort.

    The criterion is a value rather than a set of loose keyword arguments because
    every artefact has to record which one produced it: an induction census run
    on exact repeats and one run on approximate repeats are different
    measurements, and a reader who cannot tell them apart cannot use either.

    ``kind`` selects the search.  ``"exact"`` runs :func:`find_internal_repeat`
    unchanged, so the exact arm of the comparison is bit-identical to the census
    that predates this criterion; ``"approximate"`` runs
    :func:`find_approximate_internal_repeat`.  ``similarity`` names the rule that
    decides whether a substituted position is admissible:

    ``identity``
        No rule.  Any substitution is admitted, subject only to
        ``max_substitution_rate``.
    ``blosum62_nonadverse``
        The mean BLOSUM62 score over the substituted positions of the window must
        be at least zero.  BLOSUM62 entries are log-odds of a substitution in
        aligned blocks of homologous proteins against the background, so zero is
        the point at which the observed substitutions are exactly as likely under
        the homology model as under chance, and the rule reads as "these two
        segments are diverged copies of one another, not two unrelated segments
        that happen to share half their residues".  The threshold is the
        log-odds neutral point, not a fitted number.
    """

    kind: str
    min_unit: int
    max_gap_ratio: float
    min_distinct: int
    max_substitution_rate: float
    similarity: str

    _KINDS = frozenset({"exact", "approximate"})
    _SIMILARITIES = frozenset({"identity", "blosum62_nonadverse"})

    def __post_init__(self) -> None:
        if self.kind not in self._KINDS:
            raise ValueError(f"unknown repeat criterion kind {self.kind!r}")
        if self.similarity not in self._SIMILARITIES:
            raise ValueError(f"unknown similarity rule {self.similarity!r}")
        if self.min_unit < 2 or self.max_gap_ratio < 1.0 or self.min_distinct < 1:
            raise ValueError("invalid internal-repeat search parameters")
        if not 0.0 <= self.max_substitution_rate < 1.0:
            raise ValueError("max_substitution_rate must lie in [0, 1)")
        if self.kind == "exact":
            if self.max_substitution_rate != 0.0 or self.similarity != "identity":
                raise ValueError("an exact criterion tolerates no substitution")
        elif self.max_substitution_rate <= 0.0:
            raise ValueError("an approximate criterion must tolerate substitution")
        if self.similarity == "blosum62_nonadverse" and self.kind != "approximate":
            raise ValueError("a BLOSUM62 rule only applies to an approximate criterion")
        if self.tolerance.denominator > 1000:
            raise ValueError("max_substitution_rate must be a simple fraction")

    @property
    def tolerance(self) -> Fraction:
        """``max_substitution_rate`` as an exact fraction.

        The window test is ``substitutions <= rate * length``, which is evaluated
        on integer prefix sums.  Carrying the rate as a fraction keeps that test
        exact at the boundary, where a float rate would decide records by
        rounding.
        """

        return Fraction(str(self.max_substitution_rate))

    @property
    def uses_amino_acid_alphabet(self) -> bool:
        return self.similarity == "blosum62_nonadverse"

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "min_unit": int(self.min_unit),
            "max_gap_ratio": float(self.max_gap_ratio),
            "min_distinct": int(self.min_distinct),
            "max_substitution_rate": float(self.max_substitution_rate),
            "similarity": self.similarity,
            "occurrences": 2,
            "indels": False,
        }


#: The exact protein criterion the induction census was originally run on.
PROTEIN_EXACT_CRITERION = RepeatCriterion(
    kind="exact",
    min_unit=16,
    max_gap_ratio=2.0,
    min_distinct=8,
    max_substitution_rate=0.0,
    similarity="identity",
)

#: The approximate protein criterion.  Every geometric parameter is held at the
#: exact criterion's value so that exactly one thing changes between the two
#: probes; ``max_substitution_rate`` and the two-occurrence, no-indel scope come
#: from arXiv:2602.23179 v5, and ``similarity`` is the BLOSUM62 rule that paper's
#: amino-acid-similarity neurons motivate.  Any exact repeat satisfies this
#: criterion, so the approximate cohort is a superset of the exact one, which is
#: the cohort-level image of that paper's finding that approximate-repeat
#: detection subsumes exact-repeat detection.
PROTEIN_APPROXIMATE_CRITERION = RepeatCriterion(
    kind="approximate",
    min_unit=16,
    max_gap_ratio=2.0,
    min_distinct=8,
    max_substitution_rate=0.5,
    similarity="blosum62_nonadverse",
)

#: The exact text criterion the induction census was originally run on.
TEXT_EXACT_CRITERION = RepeatCriterion(
    kind="exact",
    min_unit=40,
    max_gap_ratio=2.0,
    min_distinct=15,
    max_substitution_rate=0.0,
    similarity="identity",
)

#: The approximate text control.
#:
#: Matched to :data:`PROTEIN_APPROXIMATE_CRITERION` on everything the two
#: modalities share -- ungapped alignment, exactly two occurrences, the same
#: substitution cap, and each modality's own exact-probe geometry -- and
#: deliberately *not* matched on the similarity rule, because there is no honest
#: analogue of BLOSUM62 for text.  BLOSUM62 is an empirical log-odds table
#: estimated from aligned blocks of homologous protein families; no comparable
#: table exists over characters or BPE pieces, and inventing one (case folding,
#: keyboard adjacency, embedding cosine) would either be near-vacuous or would
#: smuggle a model's own representation into the definition of its probe.  The
#: honest analogue is therefore the same criterion with the rule dropped, which
#: makes the text probe strictly *more* permissive than the protein probe: it
#: admits substitutions the protein probe rejects.  That asymmetry is the one
#: direction worth accepting, because it cannot be read as the text arm having
#: been handed a stingier probe.  It does bias the other way -- the text arm's
#: accepted repeats include substitutions no similarity rule vouched for, which
#: are harder to detect -- and that is recorded rather than corrected.
TEXT_APPROXIMATE_CRITERION = RepeatCriterion(
    kind="approximate",
    min_unit=40,
    max_gap_ratio=2.0,
    min_distinct=15,
    max_substitution_rate=0.5,
    similarity="identity",
)


@dataclass(frozen=True)
class RepeatHit:
    """One internal repeat, with the statistics that show what was tolerated."""

    first_start: int
    second_start: int
    length: int
    substituted: int
    mean_blosum62_substituted: float | None

    def __post_init__(self) -> None:
        if self.length < 1 or self.first_start < 0:
            raise ValueError("a repeat needs a non-negative start and a positive length")
        if self.second_start - self.first_start < self.length:
            raise ValueError("the two occurrences of a repeat must not overlap")
        if not 0 <= self.substituted <= self.length:
            raise ValueError("substituted positions must lie within the repeat")
        if self.substituted == 0 and self.mean_blosum62_substituted is not None:
            raise ValueError("an exact repeat has no substituted positions to score")

    @property
    def coordinates(self) -> tuple[int, int, int]:
        return self.first_start, self.second_start, self.length

    @property
    def identity_fraction(self) -> float:
        return (self.length - self.substituted) / self.length

    def as_dict(self) -> dict[str, Any]:
        return {
            "first_start": int(self.first_start),
            "second_start": int(self.second_start),
            "length": int(self.length),
            "substituted": int(self.substituted),
            "identity_fraction": _finite(self.identity_fraction, "repeat identity"),
            "mean_blosum62_substituted": (
                None
                if self.mean_blosum62_substituted is None
                else _finite(self.mean_blosum62_substituted, "repeat BLOSUM62 mean")
            ),
        }


def _encode_symbols(symbols: str, *, amino_acid: bool) -> np.ndarray:
    """Symbols as integer codes; residues as :data:`~.arms.AA20` indices."""

    if not symbols:
        raise ValueError("cannot encode an empty symbol string")
    if amino_acid:
        codes = _RESIDUE_INDEX[np.frombuffer(symbols.encode("ascii"), dtype=np.uint8)]
        if (codes < 0).any():
            raise ValueError("sequence contains a symbol outside the canonical alphabet")
        return codes
    return np.frombuffer(symbols.encode("utf-32-le"), dtype=np.uint32).astype(np.int64)


def _widest_tolerant_span(cumulative: np.ndarray) -> int:
    """Widest ``i - j`` with ``j < i`` and ``cumulative[i] <= cumulative[j]``, else -1.

    ``cumulative`` is the prefix sum of a per-position budget that is positive at
    a substituted position and negative at a matched one, scaled so that a window
    is within the substitution cap exactly when its sum is at most zero.  Only
    the strict prefix maxima of ``cumulative`` can be the left end of a widest
    pair -- any other left end is dominated by an earlier one with a value at
    least as large -- and their values increase with index, so one binary search
    per right end gives the earliest admissible left end.  Used as a rejection
    test: no window survives the length floor if even the widest does not reach
    it, and that is the common case at almost every period.
    """

    size = int(cumulative.size)
    if size < 2:
        return -1
    running = np.maximum.accumulate(cumulative)
    is_peak = np.empty(size, dtype=bool)
    is_peak[0] = True
    is_peak[1:] = cumulative[1:] > running[:-1]
    peak_at = np.flatnonzero(is_peak)
    rank = np.searchsorted(cumulative[peak_at], cumulative, side="left")
    widths = np.where(
        rank < peak_at.size,
        np.arange(size) - peak_at[np.minimum(rank, peak_at.size - 1)],
        -1,
    )
    return int(widths.max())


def _admissible_window(
    codes: np.ndarray,
    cumulative: np.ndarray,
    *,
    period: int,
    low: int,
    high: int,
    criterion: RepeatCriterion,
) -> RepeatHit | None:
    """Longest window at one period that passes every gate, or ``None``.

    The composition and similarity gates are applied inside the length sweep
    rather than to a single pre-selected window.  That is not an optimisation, it
    is the difference between a correct search and an incorrect one: under
    substitution tolerance the longest window at a given period is frequently a
    chance alignment somewhere else in the record, and gating it after the fact
    discards the real repeat sitting at the same period.  Checked directly by
    ``test_approximate_subsumes_exact``.
    """

    widest = _widest_tolerant_span(cumulative)
    if widest < low:
        return None
    for span in range(min(high, widest), low - 1, -1):
        for start in np.flatnonzero(cumulative[span:] <= cumulative[:-span]):
            start = int(start)
            first = codes[start : start + span]
            second = codes[start + period : start + period + span]
            if np.unique(first).size < criterion.min_distinct:
                continue
            differing = first != second
            substituted = int(differing.sum())
            mean_blosum: float | None = None
            if criterion.similarity == "blosum62_nonadverse" and substituted:
                mean_blosum = float(BLOSUM62[first[differing], second[differing]].mean())
                if mean_blosum < 0.0:
                    continue
            return RepeatHit(
                first_start=start,
                second_start=start + period,
                length=span,
                substituted=substituted,
                mean_blosum62_substituted=mean_blosum,
            )
    return None


def find_approximate_internal_repeat(
    symbols: str, criterion: RepeatCriterion
) -> RepeatHit | None:
    """Longest substitution-tolerant internal repeat, or ``None``.

    The two occurrences are aligned without gaps and at a constant offset, which
    is the scope arXiv:2602.23179 v5 works in: exactly two occurrences, at most
    ``max_substitution_rate`` of the aligned positions substituted, no insertions
    or deletions.  Ungapped alignment is what makes the probe usable at all --
    the prefix-matching score is read off a single period, so an indel would
    desynchronise every position after it -- and it is also the regime the prior
    work reports as the one its models handle stably.

    The geometric constraints are the exact probe's, unchanged:
    ``length <= second_start - first_start <= max_gap_ratio * length`` keeps the
    pair non-overlapping and tandem, and ``min_distinct`` rejects low-complexity
    runs.  Low-complexity rejection matters more here than in the exact probe,
    because substitution tolerance admits compositionally biased regions that
    exact matching does not.

    Because every exact repeat satisfies every clause of an approximate
    criterion with the same geometry, this search returns a hit whenever
    :func:`find_internal_repeat` does.  The cohorts are nested by construction,
    which is what makes the two columns of the census a paired comparison rather
    than two samples.
    """

    if criterion.kind != "approximate":
        raise ValueError(f"{criterion.kind!r} criterion routed to the approximate search")
    codes = _encode_symbols(symbols, amino_acid=criterion.uses_amino_acid_alphabet)
    size = int(codes.size)
    if size < 2 * criterion.min_unit:
        return None
    tolerance = criterion.tolerance
    numerator = int(tolerance.numerator)
    denominator = int(tolerance.denominator)
    best: RepeatHit | None = None
    for period in range(criterion.min_unit, size - criterion.min_unit + 1):
        low = max(criterion.min_unit, math.ceil(period / criterion.max_gap_ratio))
        high = min(period, size - period)
        if best is not None:
            low = max(low, best.length + 1)
        if low > high:
            continue
        substituted = codes[:-period] != codes[period:]
        budget = denominator * substituted.astype(np.int64) - numerator
        cumulative = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(budget)))
        found = _admissible_window(
            codes, cumulative, period=period, low=low, high=high, criterion=criterion
        )
        if found is not None:
            best = found
    return best


def find_repeat(symbols: str, criterion: RepeatCriterion) -> RepeatHit | None:
    """Dispatch a record to the search its criterion names.

    The exact branch calls :func:`find_internal_repeat` rather than the
    approximate search at zero tolerance.  The two searches agree on the
    criterion but not on the seeding, so routing exact records through the newer
    code would silently move the exact cohort and destroy the only baseline the
    approximate probe can be read against.
    """

    if criterion.kind == "exact":
        found = find_internal_repeat(
            symbols,
            min_unit=criterion.min_unit,
            max_gap_ratio=criterion.max_gap_ratio,
            min_distinct=criterion.min_distinct,
        )
        if found is None:
            return None
        first, second, span = found
        return RepeatHit(
            first_start=first,
            second_start=second,
            length=span,
            substituted=0,
            mean_blosum62_substituted=None,
        )
    return find_approximate_internal_repeat(symbols, criterion)


def scan_for_repeats(
    records: Sequence[str], criterion: RepeatCriterion, *, workers: int
) -> list[RepeatHit | None]:
    """Search every record, in input order, optionally across processes.

    The whole eligible corpus is searched rather than only its prefix, because
    the number of records that satisfy a criterion is the quantity that caps a
    cohort's size and it has to be measured rather than assumed.  Output order is
    the input order regardless of ``workers``, so the cohort a scan produces does
    not depend on how many cores ran it.
    """

    if workers < 1:
        raise ValueError("workers must be positive")
    if not records:
        raise ValueError("no records were supplied to the repeat scan")
    if workers == 1:
        return [find_repeat(record, criterion) for record in records]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(
            pool.map(partial(find_repeat, criterion=criterion), records, chunksize=64)
        )


def _repeat_cohort_metadata(
    criterion: RepeatCriterion, hits: Sequence[RepeatHit], scanned: int, matching: int
) -> dict[str, Any]:
    """Cohort provenance: the criterion, the census it implies, and per-record stats."""

    if scanned < 1:
        raise ValueError("a repeat census must have scanned at least one record")
    identity = [hit.identity_fraction for hit in hits]
    scored = [
        hit.mean_blosum62_substituted
        for hit in hits
        if hit.mean_blosum62_substituted is not None
    ]
    return {
        "repeats": [list(hit.coordinates) for hit in hits],
        "repeat_stats": [hit.as_dict() for hit in hits],
        "criterion": criterion.as_dict(),
        "census": {
            "scanned_eligible": int(scanned),
            "n_matching": int(matching),
            "match_rate": _finite(matching / scanned, "repeat match rate"),
        },
        "cohort_identity_fraction_mean": _finite(
            float(np.mean(identity)), "cohort identity"
        ),
        "cohort_identity_fraction_min": _finite(float(np.min(identity)), "cohort identity"),
        "cohort_repeat_length_mean": _finite(
            float(np.mean([hit.length for hit in hits])), "cohort repeat length"
        ),
        "cohort_mean_blosum62_substituted": (
            _finite(float(np.mean(scored)), "cohort BLOSUM62 mean") if scored else None
        ),
    }


def _select_matching(
    found: Sequence[RepeatHit | None], *, n: int, skip: int, seed: int | None, name: str
) -> list[int]:
    """Indices of the records to keep, out of those that matched the criterion.

    The census is over the whole corpus either way; this only decides which of
    the matching records enter the cohort. In file order that is the head of a
    family-grouped file, so the induction probe would be built on near-clonal
    homologues -- the identical hazard the audit's Appendix B rule 1 names, one
    level further in, because the filter runs first and hides it.
    """

    matching = [index for index, hit in enumerate(found) if hit is not None]
    if len(matching) < skip + n:
        raise RuntimeError(
            f"cohort {name!r}: only {len(matching)} matching records for {n} after a "
            f"skip of {skip}"
        )
    positions = selected_positions(
        len(matching), n=n, skip=skip, seed=seed, label=name
    )
    return [matching[position] for position in positions]


def protein_repeat_cohort(
    n: int,
    *,
    min_len: int,
    max_len: int,
    criterion: RepeatCriterion = PROTEIN_EXACT_CRITERION,
    workers: int = 1,
    name: str = "swissprot_tandem_repeat",
    skip: int = 0,
    seed: int | None = None,
) -> Cohort:
    """Swiss-Prot proteins that contain an internal repeat under ``criterion``.

    Drawn from the EC-labelled source so that one cohort serves ZymCTRL and the
    unconditional protein arms alike, which is what keeps the digest identical
    across the protein arms.

    The whole eligible corpus is searched even after ``n`` records have been
    found.  Under the exact criterion that census is the finding: tandem repeats
    of sixteen or more identical residues occur roughly once in four thousand
    entries, so the cohort has a hard ceiling in the low tens and the per-arm
    precision of the induction census is capped by it.  Under the approximate
    criterion the same census is what shows whether that ceiling was a fact about
    proteins or an artefact of demanding literal identity.
    """

    if n < 1 or min_len < 2 * criterion.min_unit or max_len < min_len:
        raise ValueError("invalid repeat-cohort length parameters")
    allowed = set(AA20)
    sequences: list[str] = []
    labels: list[str] = []
    for header, body in iter_fasta(ZYMCTRL_FASTA):
        if "<start>" not in body or "<end>" not in body:
            continue
        sequence = body.split("<start>")[1].split("<end>")[0]
        if not (min_len <= len(sequence) <= max_len) or not set(sequence) <= allowed:
            continue
        label = header.split()[0].split("|")[-1]
        if not label:
            raise ValueError(f"cannot parse an EC label from {header!r}")
        sequences.append(sequence)
        labels.append(label)
    if not sequences:
        raise RuntimeError(f"cohort {name!r}: no eligible entries in the EC-labelled source")
    found = scan_for_repeats(sequences, criterion, workers=workers)
    matching = sum(1 for hit in found if hit is not None)
    chosen = _select_matching(found, n=n, skip=skip, seed=seed, name=name)
    records = [sequences[index] for index in chosen]
    selected_labels = [labels[index] for index in chosen]
    hits = [found[index] for index in chosen]
    metadata = _repeat_cohort_metadata(criterion, hits, len(sequences), matching)
    metadata["ec_labels"] = selected_labels
    metadata["source"] = "zymctrl_ec_labelled_swissprot"
    metadata["sampling"] = sampling_record(
        seed=seed,
        skip=skip,
        requested=n,
        eligible=matching,
        corpus="zymctrl_ec_labelled_swissprot_matching_the_repeat_criterion",
    )
    return Cohort(name, "protein", records, min_len, max_len, metadata)


def text_repeat_cohort(
    n: int,
    *,
    max_chars: int = 2000,
    criterion: RepeatCriterion = TEXT_EXACT_CRITERION,
    scan_documents: int = 3000,
    workers: int = 1,
    name: str = "openwebtext_repeat",
    skip: int = 0,
    seed: int | None = None,
) -> Cohort:
    """OpenWebText documents that contain a repeated span under ``criterion``.

    The text control for :func:`protein_repeat_cohort`.  Documents are truncated
    before the search so that the detected repeat is guaranteed to lie inside the
    scored window, and the census is over the truncated documents for the same
    reason.
    """

    if n < 1 or max_chars < 4 * criterion.min_unit or scan_documents < n:
        raise ValueError("invalid text repeat-cohort parameters")
    pool = text_cohort(scan_documents, min_chars=max_chars, name=name, seed=seed)
    documents = [document[:max_chars] for document in pool.records]
    found = scan_for_repeats(documents, criterion, workers=workers)
    matching = sum(1 for hit in found if hit is not None)
    chosen = _select_matching(found, n=n, skip=skip, seed=seed, name=name)
    records = [documents[index] for index in chosen]
    hits = [found[index] for index in chosen]
    metadata = _repeat_cohort_metadata(criterion, hits, len(documents), matching)
    metadata["source"] = "openwebtext_screening_subset"
    # Two draws happen here and both are recorded: the scanning pool comes from
    # ``text_cohort`` under the same seed, and the cohort is a draw from the
    # documents in that pool which matched the criterion.
    metadata["sampling"] = sampling_record(
        seed=seed,
        skip=skip,
        requested=n,
        eligible=matching,
        corpus="openwebtext_screening_subset_matching_the_repeat_criterion",
    )
    metadata["scan_pool_sampling"] = pool.sampling
    return Cohort(name, "text", records, max_chars, max_chars, metadata)


@dataclass(frozen=True)
class RepeatProbe:
    """One sequence together with the (query, key) pairs a prefix-matching head must hit.

    ``key_positions[i]`` is the position of the token that *followed* the earlier
    occurrence of the token at ``query_positions[i]``.  ``coverage`` is the
    fraction of candidate second-copy tokens that could be aligned; it is below
    one only for multi-residue BPE, where the two copies of an identical span can
    be segmented differently.
    """

    kind: str
    input_ids: tuple[int, ...]
    query_positions: tuple[int, ...]
    key_positions: tuple[int, ...]
    coverage: float
    repeat_symbols: int

    def __post_init__(self) -> None:
        if self.kind not in {
            "synthetic_repeat",
            "natural_repeat_exact",
            "natural_repeat_approximate",
        }:
            raise ValueError(f"unknown probe kind {self.kind!r}")
        if not self.query_positions or len(self.query_positions) != len(self.key_positions):
            raise ValueError("probe query and key positions must be non-empty and aligned")
        length = len(self.input_ids)
        for query, key in zip(self.query_positions, self.key_positions):
            if not 0 < key < query < length:
                raise ValueError(f"probe position pair ({query}, {key}) is not causal")


def synthetic_repeat_probes(
    arm: Arm,
    unigram: Unigram,
    *,
    n_probes: int,
    copy_len: int,
    seed: int,
    ec_label: str | None = None,
) -> list[RepeatProbe]:
    """``[prefix][random S][same S]`` probes built directly in token space.

    Building the probe in token space rather than in symbol space guarantees that
    the second copy is token-identical to the first for every arm, including
    multi-residue BPE and including wrapped renderings: no tokenisation happens
    here at all, so layout characters inserted by a rendering cannot
    desynchronise the two copies.  What a wrapped rendering does change is the
    sampling distribution, and layout tokens are excluded from it in
    :func:`fit_unigram`, so the probe carries no line breaks.

    The price is that these sequences are off-distribution for a protein model in
    a way that a repeated English paragraph is not for a text model -- and, for a
    wrapped arm, additionally off-distribution in carrying no line break across
    its whole length.  :func:`natural_repeat_probes` is the in-distribution
    counterpart and both are reported; for a wrapped arm the natural-repeat score
    is the one to trust.
    """

    if n_probes < 1 or copy_len < 4:
        raise ValueError("invalid synthetic-probe parameters")
    prompt = prefix_ids(arm, ec_label=ec_label)
    offset = len(prompt)
    rng = np.random.default_rng(seed)
    probes: list[RepeatProbe] = []
    for _ in range(n_probes):
        body = [int(token) for token in unigram.sample(rng, copy_len)]
        ids = prompt + body + body
        queries = tuple(offset + copy_len + index for index in range(copy_len - 1))
        keys = tuple(offset + index + 1 for index in range(copy_len - 1))
        probes.append(
            RepeatProbe(
                kind="synthetic_repeat",
                input_ids=tuple(ids),
                query_positions=queries,
                key_positions=keys,
                coverage=1.0,
                repeat_symbols=copy_len,
            )
        )
    return probes


def record_symbol_offsets(arm: Arm, text: str, record: str) -> list[int]:
    """Character offset in ``text`` of every symbol of ``record``.

    A rendering is not obliged to leave its record contiguous.  ProtGPT2's FASTA
    rendering hard-wraps the sequence at sixty residues, so the record is not a
    substring of its own input string and two corresponding positions in a repeat
    are not a constant character distance apart.  Callers therefore align in
    symbol space and use this map to reach characters, rather than locating the
    record by substring search and adding a fixed shift.

    The map is computed per format and then verified symbol by symbol, so a
    rendering change in :mod:`.arms` that this function has not been taught about
    fails here instead of silently mis-aligning a probe.
    """

    fmt = arm.spec.input_format
    if fmt in {"raw", "n_to_c_control", "ec_conditioned"}:
        if text.count(record) != 1:
            raise ValueError(f"{arm.name}: record is not uniquely locatable in its input string")
        offsets = [text.index(record) + index for index in range(len(record))]
    elif fmt == "fasta_wrapped":
        marker = arm.tokenizer.eos_token
        if marker is None or not text.startswith(marker):
            raise ValueError(f"{arm.name}: FASTA rendering does not open with the end-of-text marker")
        offsets = []
        cursor = len(marker)
        for symbol in record:
            while cursor < len(text) and text[cursor] == "\n":
                cursor += 1
            if cursor >= len(text) or text[cursor] != symbol:
                raise ValueError(f"{arm.name}: FASTA rendering does not contain its own record")
            offsets.append(cursor)
            cursor += 1
    else:
        raise ValueError(f"{arm.name}: unsupported input format {fmt!r}")
    if len(offsets) != len(record) or any(text[at] != symbol for at, symbol in zip(offsets, record)):
        raise ValueError(f"{arm.name}: symbol offset map does not reproduce the record")
    return offsets


def natural_repeat_probes(
    arm: Arm,
    cohort: Cohort,
    *,
    max_tokens: int,
) -> list[RepeatProbe]:
    """Probes over real sequences that contain a repeated span.

    Alignment is done in symbol space and by *position*, never by token identity.
    Each token is mapped to the contiguous run of record symbols it covers, and a
    second-copy token is scored against the token that follows its aligned
    counterpart one period earlier.  Under an exact cohort that counterpart holds
    the same symbols; under an approximate cohort it need not, and the score is
    then exactly what the approximate mechanism requires -- attention to the
    aligned position rather than to a literal earlier copy of the current token.
    Nothing in this function changes between the two cohorts, because the repeat
    is ungapped and therefore has one constant period either way.

    A second-copy token is scored only when the first copy is segmented at
    exactly the same symbol boundaries, so that "the token that followed the
    aligned earlier position" is unambiguous.  For a residue-level tokenizer that
    is automatic.  For multi-residue BPE it is not, and it is stricter under an
    approximate cohort than under an exact one, because a substitution can move a
    merge boundary and desynchronise the two segmentations.  The cost is paid in
    ``coverage``, which is reported per probe rather than repaired: redirecting a
    query to a differently segmented neighbour would score something other than
    induction.

    The key must be the token *literally* following the aligned earlier position.
    Under a wrapped rendering that token is sometimes a line break, which carries
    no record symbol; such pairs are dropped rather than redirected to the next
    residue token, because attending across a layout token is a different
    computation from induction and scoring it as induction would flatter the one
    arm whose rendering has layout tokens.  Everything dropped is reported as
    ``coverage`` rather than approximated.
    """

    if max_tokens < 8:
        raise ValueError("max_tokens must leave room for a repeat")
    repeats = cohort.metadata.get("repeats")
    if repeats is None or len(repeats) != len(cohort.records):
        raise ValueError(f"cohort {cohort.name!r} carries no per-record repeat coordinates")
    criterion = cohort.metadata.get("criterion")
    if criterion is None or criterion.get("kind") not in {"exact", "approximate"}:
        raise ValueError(f"cohort {cohort.name!r} does not declare a repeat criterion")
    probe_kind = f"natural_repeat_{criterion['kind']}"
    if not arm.tokenizer.is_fast:
        raise TypeError(f"{arm.name}: natural-repeat alignment needs a fast tokenizer")
    strings = cohort.input_strings(arm)
    probes: list[RepeatProbe] = []
    for text, record, coordinates in zip(strings, cohort.records, repeats):
        first, second, span = (int(value) for value in coordinates)
        symbol_at = record_symbol_offsets(arm, text, record)
        symbol_of = {offset: index for index, offset in enumerate(symbol_at)}
        encoded = arm.tokenizer(text, return_tensors=None, return_offsets_mapping=True)
        ids = encoded["input_ids"][:max_tokens]
        offsets = encoded["offset_mapping"][: len(ids)]

        # A token carries the contiguous run of record symbols its characters
        # cover. Layout characters contribute nothing, so a line-break token has
        # no run and can never be a query or a key.
        covered: dict[int, tuple[int, int]] = {}
        for index, (begin, end) in enumerate(offsets):
            symbols = [symbol_of[at] for at in range(begin, end) if at in symbol_of]
            if not symbols or symbols != list(range(symbols[0], symbols[-1] + 1)):
                continue
            covered[index] = (symbols[0], symbols[-1])
        starts_at = {low: index for index, (low, _) in covered.items()}

        period = second - first
        queries: list[int] = []
        keys: list[int] = []
        candidates = 0
        for index, (low, high) in sorted(covered.items()):
            if low < second or high >= second + span:
                continue
            candidates += 1
            mirror = starts_at.get(low - period)
            if mirror is None or covered[mirror] != (low - period, high - period):
                continue
            key = mirror + 1
            key_span = covered.get(key)
            if key_span is None or key_span[0] != high - period + 1:
                continue
            if key_span[1] >= first + span or key >= index:
                continue
            queries.append(index)
            keys.append(key)
        if not queries:
            continue
        probes.append(
            RepeatProbe(
                kind=probe_kind,
                input_ids=tuple(int(token) for token in ids),
                query_positions=tuple(queries),
                key_positions=tuple(keys),
                coverage=len(queries) / candidates,
                repeat_symbols=span,
            )
        )
    if not probes:
        raise RuntimeError(
            f"{arm.name}: no {probe_kind} probe survived token alignment on cohort "
            f"{cohort.name!r}"
        )
    return probes


# --------------------------------------------------------- attention alignment


def _pad_probe_batch(
    arm: Arm, probes: Sequence[RepeatProbe]
) -> tuple[torch.Tensor, torch.Tensor]:
    pad = arm.tokenizer.pad_token_id
    if pad is None:
        raise ValueError(f"{arm.name}: tokenizer has no pad token")
    width = max(len(probe.input_ids) for probe in probes)
    ids = torch.full((len(probes), width), int(pad), dtype=torch.long)
    mask = torch.zeros((len(probes), width), dtype=torch.long)
    for row, probe in enumerate(probes):
        length = len(probe.input_ids)
        ids[row, :length] = torch.tensor(probe.input_ids, dtype=torch.long)
        mask[row, :length] = 1
    return ids, mask


@torch.no_grad()
def attention_alignment_scores(
    arm: Arm,
    probes: Sequence[RepeatProbe],
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Per-head prefix-matching score plus two specificity controls.

    ``prefix_matching`` is the mean attention from a repeated token to the token
    that followed its earlier occurrence.  ``same_token`` is the mean attention
    to the earlier occurrence itself, which separates a duplicate-token head from
    an induction head, and ``offset_two`` is the mean attention one position
    further along, which separates a genuine offset-one match from a head that
    simply smears over the earlier region.
    """

    arm.require("circuits")
    arm.require_eager_attention("the prefix-matching census")
    if not probes:
        raise ValueError("no probes were supplied")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    heads = n_head(arm)
    layers = arm.n_layer
    totals = {
        key: np.zeros((layers, heads), dtype=np.float64)
        for key in ("prefix_matching", "same_token", "offset_two")
    }
    scored = 0
    uniform = 0.0
    for begin in range(0, len(probes), batch_size):
        chunk = probes[begin : begin + batch_size]
        ids, mask = _pad_probe_batch(arm, chunk)
        ids = ids.to(arm.device)
        mask = mask.to(arm.device)
        output = arm.model(
            input_ids=ids,
            attention_mask=mask,
            output_attentions=True,
            use_cache=False,
        )
        attentions = output.attentions
        if len(attentions) != layers or any(item is None for item in attentions):
            raise RuntimeError(
                f"{arm.name}: attention weights unavailable; load the arm with "
                "attn_implementation='eager'"
            )
        for layer, pattern in enumerate(attentions):
            if pattern.shape[1] != heads:
                raise RuntimeError(f"{arm.name}: layer {layer} returned {pattern.shape[1]} heads")
            for row, probe in enumerate(chunk):
                query = torch.tensor(probe.query_positions, device=arm.device)
                key = torch.tensor(probe.key_positions, device=arm.device)
                block = pattern[row].float()
                totals["prefix_matching"][layer] += block[:, query, key].sum(dim=1).cpu().numpy()
                totals["same_token"][layer] += block[:, query, key - 1].sum(dim=1).cpu().numpy()
                totals["offset_two"][layer] += block[:, query, key + 1].sum(dim=1).cpu().numpy()
        for probe in chunk:
            scored += len(probe.query_positions)
            uniform += sum(1.0 / (position + 1) for position in probe.query_positions)
        del output, attentions
    if scored < 1:
        raise RuntimeError(f"{arm.name}: probes contributed no scored query positions")
    means = {key: value / scored for key, value in totals.items()}
    return {
        "kind": probes[0].kind,
        "n_probes": len(probes),
        "scored_query_positions": scored,
        "mean_coverage": _finite(
            float(np.mean([probe.coverage for probe in probes])), "probe coverage"
        ),
        "mean_repeat_symbols": _finite(
            float(np.mean([probe.repeat_symbols for probe in probes])), "repeat length"
        ),
        "uniform_baseline": _finite(uniform / scored, "uniform baseline"),
        "scores": means,
    }


def summarise_head_matrix(values: np.ndarray, label: str) -> dict[str, Any]:
    """Distribution summary of a per-head score matrix of shape (layer, head)."""

    if values.ndim != 2:
        raise ValueError(f"{label} must be a (layer, head) matrix")
    flat = values.reshape(-1)
    if not np.isfinite(flat).all():
        raise ValueError(f"{label} contains non-finite values")
    quantiles = np.quantile(flat, [0.5, 0.9, 0.99, 1.0])
    return {
        "mean": _finite(float(flat.mean()), f"{label} mean"),
        "sd": _finite(float(flat.std(ddof=1)), f"{label} sd"),
        "median": _finite(float(quantiles[0]), f"{label} median"),
        "q90": _finite(float(quantiles[1]), f"{label} q90"),
        "q99": _finite(float(quantiles[2]), f"{label} q99"),
        "max": _finite(float(quantiles[3]), f"{label} max"),
        "n_heads": int(flat.size),
    }


def head_census(
    prefix_matching: np.ndarray,
    *,
    thresholds: Sequence[float] = INDUCTION_THRESHOLDS,
    data_driven_sigma: float = 3.0,
) -> dict[str, Any]:
    """Count and locate heads whose prefix-matching score clears each threshold."""

    if prefix_matching.ndim != 2:
        raise ValueError("prefix-matching scores must be a (layer, head) matrix")
    if data_driven_sigma <= 0:
        raise ValueError("data_driven_sigma must be positive")
    summary = summarise_head_matrix(prefix_matching, "prefix_matching")
    cut = summary["mean"] + data_driven_sigma * summary["sd"]
    layers = prefix_matching.shape[0]
    counts = {f"{value:.2f}": int((prefix_matching >= value).sum()) for value in thresholds}
    above = np.argwhere(prefix_matching >= cut)
    return {
        "distribution": summary,
        "count_above_threshold": counts,
        "data_driven_threshold": _finite(cut, "data-driven threshold"),
        "data_driven_sigma": float(data_driven_sigma),
        "count_above_data_driven": int(above.shape[0]),
        "data_driven_layer_fractions": [
            _finite(float(layer) / max(layers - 1, 1), "layer fraction")
            for layer in sorted(int(item[0]) for item in above)
        ],
    }


def induction_headline(
    alignment: Mapping[str, Any],
    census: Mapping[str, Any],
    *,
    threshold: float = 0.10,
) -> dict[str, Any]:
    """The four numbers an induction census is actually read by.

    Peak prefix matching is reported as a multiple of the probe's own uniform
    baseline because the baseline is ``mean(1 / (position + 1))`` over the scored
    queries and therefore differs between arms whose repeats sit at different
    depths; a raw peak compares two heads at two different chance levels.  The
    count above a threshold is reported as a fraction of the arm's heads for the
    same reason: ProGen2-medium has 432 heads against GPT-2-large's 720.

    Derived here rather than at each call site so that the exact and approximate
    probes, and the panel summary that sets them side by side, cannot drift into
    computing the headline three slightly different ways.
    """

    label = f"{threshold:.2f}"
    counts = census["count_above_threshold"]
    if label not in counts:
        raise KeyError(f"threshold {label} is not among the census thresholds {sorted(counts)}")
    baseline = _finite(float(alignment["uniform_baseline"]), "uniform baseline")
    if baseline <= 0.0:
        raise ValueError("uniform baseline must be positive")
    peak = _finite(float(census["distribution"]["max"]), "peak prefix matching")
    n_heads = int(census["distribution"]["n_heads"])
    if n_heads < 1:
        raise ValueError("head census reports no heads")
    above = int(counts[label])
    return {
        "n_probes": int(alignment["n_probes"]),
        "scored_query_positions": int(alignment["scored_query_positions"]),
        "mean_coverage": _finite(float(alignment["mean_coverage"]), "probe coverage"),
        "uniform_baseline": baseline,
        "peak_prefix_matching": peak,
        "peak_over_uniform": _finite(peak / baseline, "peak over uniform"),
        "threshold": float(threshold),
        "n_heads": n_heads,
        "n_above_threshold": above,
        "fraction_above_threshold": _finite(above / n_heads, "fraction above threshold"),
        "n_above_data_driven": int(census["count_above_data_driven"]),
    }


def top_heads(
    scores: Mapping[str, np.ndarray],
    *,
    key: str,
    count: int,
) -> list[dict[str, Any]]:
    """The ``count`` highest-scoring heads under ``key``, with all scores attached."""

    if key not in scores:
        raise KeyError(f"unknown ranking key {key!r}")
    if count < 1:
        raise ValueError("count must be positive")
    primary = scores[key]
    order = np.argsort(primary, axis=None)[::-1][:count]
    rows: list[dict[str, Any]] = []
    for flat in order:
        layer, head = np.unravel_index(int(flat), primary.shape)
        entry: dict[str, Any] = {"layer": int(layer), "head": int(head)}
        for name, matrix in scores.items():
            entry[name] = _finite(float(matrix[layer, head]), f"{name}[{layer},{head}]")
        rows.append(entry)
    return rows


# ------------------------------------------------------------------ OV circuit


def head_ov_weights(arm: Arm, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head ``(W_V, W_O)`` with shapes ``(head, d, d_head)`` and ``(head, d_head, d)``.

    Both are expressed as right-multiplied maps, so ``x @ W_V[h] @ W_O[h]`` is the
    residual-stream contribution head ``h`` makes for a value read from ``x``.
    Three layouts are supported: GPT-2's fused ``Conv1D``; ProGen2's GPT-J-style
    ``qkv_proj``, whose model-parallel interleaving puts the value block in the
    middle of each of eight partitions; and the ``q_proj``/``v_proj``/``o_proj``
    triple of a rotary decoder.

    Under grouped-query attention the value projection emits only
    ``n_kv * d_head`` channels, and each key/value head is read by a whole group
    of query heads.  Attention patterns are indexed by query head, so the
    decomposition returns one ``W_V`` per *query* head by replicating each
    key/value head's slice across its group.  That replication is what makes the
    per-head OV map commensurate with the per-head attention score; it is not an
    approximation, and :func:`verify_head_decomposition` checks the whole slicing
    against the live forward pass because every way of getting it wrong is
    silent.
    """

    arm.require("circuits")
    if not 0 <= layer < arm.n_layer:
        raise ValueError(f"{arm.name}: layer {layer} out of range")
    attention = arm.attention(layer)
    width = arm.d_model
    heads = n_head(arm)
    dim = head_dim(arm)
    if hasattr(attention, "c_attn") and hasattr(attention, "c_proj"):
        fused = attention.c_attn.weight
        output = attention.c_proj.weight
        if tuple(fused.shape) != (width, 3 * width) or tuple(output.shape) != (width, width):
            raise TypeError(f"{arm.name}: unexpected fused attention shapes at layer {layer}")
        value = fused[:, 2 * width :]
    elif hasattr(attention, "qkv_proj") and hasattr(attention, "out_proj"):
        fused = attention.qkv_proj.weight
        if tuple(fused.shape) != (3 * width, width):
            raise TypeError(f"{arm.name}: unexpected qkv shape at layer {layer}")
        partitions = 8
        if width % partitions != 0:
            raise TypeError(f"{arm.name}: width {width} is not divisible by {partitions}")
        block = 3 * width // partitions
        chunk = width // partitions
        index = torch.cat(
            [
                torch.arange(part * block + chunk, part * block + 2 * chunk)
                for part in range(partitions)
            ]
        ).to(fused.device)
        value = fused.index_select(0, index).t()
        output = attention.out_proj.weight.t()
    elif all(hasattr(attention, name) for name in ("q_proj", "v_proj", "o_proj")):
        groups = n_key_value_head(arm)
        value_weight = attention.v_proj.weight
        output_weight = attention.o_proj.weight
        if tuple(value_weight.shape) != (groups * dim, width):
            raise TypeError(
                f"{arm.name}: layer {layer} v_proj is {tuple(value_weight.shape)}, "
                f"expected {(groups * dim, width)} for {groups} key/value heads of width {dim}"
            )
        if tuple(output_weight.shape) != (width, heads * dim):
            raise TypeError(
                f"{arm.name}: layer {layer} o_proj is {tuple(output_weight.shape)}, "
                f"expected {(width, heads * dim)} for {heads} query heads of width {dim}"
            )
        grouped = value_weight.t().reshape(width, groups, dim).permute(1, 0, 2)
        value_heads = grouped.index_select(
            0, _query_to_key_value(arm, value_weight.device)
        ).contiguous()
        output_heads = output_weight.t().reshape(heads, dim, width).contiguous()
        return value_heads.float(), output_heads.float()
    else:
        raise TypeError(f"{arm.name}: unsupported attention layout at layer {layer}")
    value_heads = value.reshape(width, heads, dim).permute(1, 0, 2).contiguous()
    output_heads = output.reshape(heads, dim, width).contiguous()
    return value_heads.float(), output_heads.float()


def head_ov_biases(arm: Arm, layer: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Per-head value bias ``(head, d_head)`` and the output bias ``(d,)``.

    Qwen2 puts a bias on ``v_proj`` where Llama, GPT-2's ``c_proj`` and ProGen2
    put none, and omitting it inflates the rebuild error by nearly an order of
    magnitude.  It is resolved per layout rather than assumed absent, and a
    layout carrying a bias this function cannot place raises instead of dropping
    it.
    """

    attention = arm.attention(layer)
    heads = n_head(arm)
    dim = head_dim(arm)
    if hasattr(attention, "c_attn"):
        fused = attention.c_attn.bias
        value = None if fused is None else fused[2 * arm.d_model :].reshape(heads, dim)
        output = attention.c_proj.bias
    elif hasattr(attention, "qkv_proj"):
        if attention.qkv_proj.bias is not None:
            raise TypeError(f"{arm.name}: a biased fused qkv projection is not handled")
        value, output = None, attention.out_proj.bias
    elif hasattr(attention, "v_proj"):
        raw = attention.v_proj.bias
        if raw is None:
            value = None
        else:
            grouped = raw.reshape(n_key_value_head(arm), dim)
            value = grouped.index_select(0, _query_to_key_value(arm, raw.device))
        output = attention.o_proj.bias
    else:
        raise TypeError(f"{arm.name}: unsupported attention layout at layer {layer}")
    return (
        None if value is None else value.detach().float(),
        None if output is None else output.detach().float(),
    )


@torch.no_grad()
def verify_head_decomposition(
    arm: Arm,
    layer: int,
    input_ids: torch.Tensor,
    *,
    relative_tolerance: float = 0.05,
) -> float:
    """Rebuild one attention layer's output from per-head weights and compare.

    Any error in the per-head value/output slicing would corrupt the copying
    score without changing its shape, so the slicing is checked against the live
    module rather than trusted.  Returns the observed relative error.
    """

    if relative_tolerance <= 0:
        raise ValueError("relative_tolerance must be positive")
    attention = arm.attention(layer)
    captured: dict[str, torch.Tensor] = {}

    def capture_norm(_module, _args, output: torch.Tensor) -> None:
        captured["normed"] = output.detach()

    def capture_attention(_module, _args, output: Any) -> None:
        captured["attn_out"] = _leading_tensor(output).detach()

    handles = [
        pre_attention_norm(arm, layer).register_forward_hook(capture_norm),
        attention.register_forward_hook(capture_attention),
    ]
    try:
        result = arm.model(
            input_ids=input_ids.to(arm.device), output_attentions=True, use_cache=False
        )
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != {"normed", "attn_out"}:
        raise RuntimeError(f"{arm.name}: attention capture failed at layer {layer}")
    pattern = result.attentions[layer]
    if pattern is None:
        raise RuntimeError(f"{arm.name}: no attention weights; load with eager attention")
    value_heads, output_heads = head_ov_weights(arm, layer)
    value_bias, output_bias = head_ov_biases(arm, layer)
    normed = captured["normed"].float()
    values = torch.einsum("btd,hde->bhte", normed, value_heads)
    if value_bias is not None:
        values = values + value_bias.reshape(1, value_heads.shape[0], 1, -1)
    mixed = torch.einsum("bhts,bhse->bhte", pattern.float(), values)
    rebuilt = torch.einsum("bhte,hed->btd", mixed, output_heads)
    if output_bias is not None:
        rebuilt = rebuilt + output_bias
    reference = captured["attn_out"].float()
    scale = float(reference.abs().max())
    if scale <= 0:
        raise RuntimeError(f"{arm.name}: layer {layer} produced an all-zero attention output")
    error = float((rebuilt - reference).abs().max()) / scale
    if error > relative_tolerance:
        raise RuntimeError(
            f"{arm.name}: per-head OV decomposition disagrees with the forward pass at "
            f"layer {layer} (relative error {error:.3f} > {relative_tolerance})"
        )
    return error


@torch.no_grad()
def ov_copying_scores(
    arm: Arm,
    token_ids: Sequence[int],
    *,
    layers: Sequence[int] | None = None,
) -> dict[str, np.ndarray]:
    """Diagonal dominance of the effective OV map ``W_U W_O^h W_V^h W_E``.

    The approximation is the standard full-OV-circuit linearisation: both
    normalisations are dropped, positional embeddings are dropped, attention
    biases are dropped, and the map is restricted to a sampled token subset ``T``
    so that the ``|T| x |T|`` matrix is tractable.  A head that copies promotes
    the token it attends to, so the diagonal of that matrix should dominate its
    row.  Where the embedding and unembedding are tied, ``W_E`` and ``W_U`` are
    one matrix and the map is exactly symmetric in that pair.

    Two statistics are returned per head.  ``diagonal_fraction`` is the share of
    rows whose maximum sits on the diagonal, which is the usual reading but
    depends strongly on ``|T|``.  ``mean_normalised_rank`` is the mean share of
    off-diagonal columns the diagonal beats, which is comparable across ``|T|``
    and is the statistic to use when a fifty-thousand-piece text vocabulary is
    set beside a twenty-residue protein alphabet.

    Both matrices are always ``(n_layer, n_head)``. Rows for layers outside
    ``layers`` hold NaN rather than zero, so a partial scan cannot be summarised
    as though it were a full one: zero is a legal score and would have read as a
    measured head that never copies.
    """

    tokens = np.asarray(token_ids, dtype=np.int64)
    if tokens.ndim != 1 or tokens.size < 2:
        raise ValueError("at least two sampled tokens are required")
    if np.unique(tokens).size != tokens.size:
        raise ValueError("sampled tokens must be distinct")
    embedding = embedding_weight(arm)
    unembedding = arm.model.lm_head.weight
    if int(tokens.max()) >= min(embedding.shape[0], unembedding.shape[0]):
        raise ValueError(f"{arm.name}: sampled token id exceeds the embedding or unembedding rows")
    index = torch.tensor(tokens, device=arm.device)
    source = embedding.index_select(0, index).float()
    target = unembedding.index_select(0, index).float()
    selected = list(range(arm.n_layer)) if layers is None else [int(layer) for layer in layers]
    if not selected:
        raise ValueError(f"{arm.name}: at least one layer must be scored")
    if any(not 0 <= layer < arm.n_layer for layer in selected):
        raise ValueError(f"{arm.name}: copying-score layer outside 0..{arm.n_layer - 1}")
    heads = n_head(arm)
    # NaN, not zero, for a layer that was not scored. Zero is a legal copying
    # score meaning "this head never promotes the token it reads", so a
    # zero-filled row for an unscored layer is a plausible measurement that no
    # measurement produced -- and ``summarise_head_matrix`` averages the whole
    # matrix. NaN makes the same mistake raise there instead.
    diagonal = np.full((arm.n_layer, heads), np.nan, dtype=np.float64)
    ranks = np.full((arm.n_layer, heads), np.nan, dtype=np.float64)
    size = tokens.size
    identity = torch.arange(size, device=arm.device)
    for layer in selected:
        value_heads, output_heads = head_ov_weights(arm, layer)
        projected = torch.einsum("nd,hde->hne", source, value_heads)
        written = torch.einsum("hne,hed->hnd", projected, output_heads)
        scores = torch.einsum("hnd,md->hnm", written, target)
        own = scores[:, identity, identity]
        diagonal[layer] = (scores.argmax(dim=2) == identity).float().mean(dim=1).cpu().numpy()
        beaten = (scores < own.unsqueeze(2)).sum(dim=2).float()
        ranks[layer] = (beaten / (size - 1)).mean(dim=1).cpu().numpy()
        del projected, written, scores
    return {"diagonal_fraction": diagonal, "mean_normalised_rank": ranks}


def matched_copying_scores(
    arm: Arm,
    token_ids: Sequence[int],
    *,
    matched_n: int,
    repeats: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Copying scores averaged over subsamples of a fixed size.

    Diagonal dominance is a within-row argmax over the sampled columns, so it is
    mechanically easier for a small sample.  Text and protein arms have very
    different effective vocabularies, so the comparison is only meaningful at a
    matched sample size; this repeats the measurement on ``repeats`` independent
    subsamples of ``matched_n`` tokens.
    """

    tokens = np.asarray(token_ids, dtype=np.int64)
    if matched_n < 2 or repeats < 1:
        raise ValueError("invalid matched-subsample parameters")
    if tokens.size < matched_n:
        raise ValueError(f"{arm.name}: only {tokens.size} tokens available for a {matched_n} sample")
    rng = np.random.default_rng(seed)
    accumulated: dict[str, np.ndarray] | None = None
    for _ in range(repeats):
        subset = rng.choice(tokens, size=matched_n, replace=False)
        scores = ov_copying_scores(arm, subset)
        if accumulated is None:
            accumulated = {key: value.copy() for key, value in scores.items()}
        else:
            for key, value in scores.items():
                accumulated[key] += value
    if accumulated is None:
        raise RuntimeError("no matched subsample was scored")
    return {key: value / repeats for key, value in accumulated.items()}


# ------------------------------------------------------ direct logit attribution


def _leading_tensor(output: Any) -> torch.Tensor:
    tensor = output[0] if isinstance(output, tuple) else output
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
        raise TypeError("expected a [batch, token, d_model] module output")
    return tensor


@torch.no_grad()
def direct_logit_attribution(
    arm: Arm,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    reconstruction_tolerance: float = 0.02,
) -> dict[str, Any]:
    """Decompose the correct-next-token logit into per-component contributions.

    The residual stream entering the final normalisation is exactly the sum of
    the embedding output and every attention and MLP sublayer output, so the only
    approximation is the treatment of that normalisation.  The standard
    linearisation is used: the scale is computed once on the true final residual
    and then held fixed while each component is pushed through the normalisation,
    the learnt gain and the unembedding.  Given that fixed scale the decomposition
    is exact and is checked against the model logits, but it is not a causal
    decomposition -- deleting a component would change the scale, so the reported
    contributions are first-order.

    The normalisation's *form* is resolved rather than assumed.  LayerNorm
    centres its input, scales by ``1/sqrt(var + eps)`` and adds a learned bias;
    RMSNorm does none of the first and last and scales by
    ``1/sqrt(mean(x^2) + eps)``.  Applying LayerNorm's form to an RMSNorm decoder
    would subtract a mean the model never subtracts and add a bias it does not
    have, which is wrong rather than approximate -- and wrong in a way that still
    produces a plausible-looking sum.

    Two reconstruction gates guard the decomposition, both relative to
    ``reconstruction_tolerance``: the components must sum to the captured final
    residual in relative L2, and the attributions plus the normalisation constant
    must sum to the model's own logit.  Norm-relative rather than max-element
    error is gated because the arms run in bfloat16, whose per-element rounding
    on a single outlier channel is not evidence of a structural error; the
    max-element ratio is reported alongside so that the reader can see it.

    Positions are scored only inside the content span, and the target is the
    token that actually follows.
    """

    arm.require("circuits")
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input ids and attention mask must share a [batch, token] shape")
    if reconstruction_tolerance <= 0:
        raise ValueError("reconstruction_tolerance must be positive")
    norm = final_norm(arm)
    form = normalisation_form(norm, f"{arm.name} final normalisation")
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def store(name: str):
        def hook(_module, _args, output: Any) -> None:
            captured[name] = _leading_tensor(output).detach().float()

        return hook

    def store_input(name: str):
        def hook(_module, args: tuple) -> None:
            captured[name] = args[0].detach().float()

        return hook

    handles.append(embedding_module(arm).register_forward_hook(store("embed")))
    handles.append(norm.register_forward_pre_hook(store_input("final_residual")))
    for layer in range(arm.n_layer):
        handles.append(arm.attention(layer).register_forward_hook(store(f"attn.{layer}")))
        handles.append(arm.mlp(layer).register_forward_hook(store(f"mlp.{layer}")))
    try:
        output = arm.model(
            input_ids=input_ids.to(arm.device),
            attention_mask=attention_mask.to(arm.device),
            use_cache=False,
        )
    finally:
        for handle in handles:
            handle.remove()

    names = ["embed"] + [
        f"{kind}.{layer}" for layer in range(arm.n_layer) for kind in ("attn", "mlp")
    ]
    missing = [name for name in [*names, "final_residual"] if name not in captured]
    if missing:
        raise RuntimeError(f"{arm.name}: component capture missed {missing}")
    final = captured["final_residual"]
    stacked = torch.stack([captured[name] for name in names], dim=0)
    deviation = stacked.sum(dim=0) - final
    residual_l2 = float(deviation.norm() / final.norm())
    residual_max = float(deviation.abs().max()) / max(float(final.abs().max()), 1e-6)
    if residual_l2 > reconstruction_tolerance:
        raise RuntimeError(
            f"{arm.name}: component sum does not reproduce the final residual "
            f"(relative L2 error {residual_l2:.4f} > {reconstruction_tolerance})"
        )

    if form.centred:
        scale = (final.var(dim=-1, keepdim=True, unbiased=False) + form.epsilon).rsqrt()
    else:
        scale = (final.pow(2).mean(dim=-1, keepdim=True) + form.epsilon).rsqrt()
    head_bias = arm.model.lm_head.bias
    ids = input_ids.to(arm.device)
    mask = attention_mask.to(arm.device)
    targets = ids[:, 1:]
    # Gather the target rows before widening to float32. Casting the whole
    # unembedding first costs 1.5 GB on a 128k-piece vocabulary and is pure
    # waste: only one row per scored position is ever read.
    direction = (
        arm.model.lm_head.weight.index_select(0, targets.reshape(-1))
        .float()
        .reshape(targets.shape[0], targets.shape[1], -1)
    )
    components = stacked[:, :, :-1]
    if form.centred:
        components = components - components.mean(dim=-1, keepdim=True)
    contributions = torch.einsum(
        "cbtd,btd->cbt", components * scale[:, :-1] * form.gain, direction
    )
    if form.bias is None:
        constant = torch.zeros(targets.shape, dtype=torch.float32, device=direction.device)
    else:
        constant = torch.einsum("d,btd->bt", form.bias, direction)
    if head_bias is not None:
        constant = constant + head_bias.float().index_select(0, targets.reshape(-1)).reshape(
            targets.shape
        )
    actual = (
        output.logits[:, :-1].gather(-1, targets.unsqueeze(-1)).squeeze(-1).float()
    )
    rebuilt = contributions.sum(dim=0) + constant

    keep = torch.zeros_like(targets, dtype=torch.bool)
    for row in range(ids.shape[0]):
        valid = int(mask[row].sum())
        low, high = content_bounds(arm, [int(t) for t in ids[row].tolist()], valid)
        if high - low < 2:
            raise ValueError(f"{arm.name}: row {row} has fewer than two content tokens")
        keep[row, low : high - 1] = True
    scored = int(keep.sum())
    if scored < 1:
        raise RuntimeError(f"{arm.name}: no scored positions after content masking")
    logit_deviation = (rebuilt[keep] - actual[keep]).abs()
    logit_scale = float(actual[keep].abs().mean())
    logit_mean_error = float(logit_deviation.mean())
    if logit_mean_error > reconstruction_tolerance * max(logit_scale, 1e-6):
        raise RuntimeError(
            f"{arm.name}: attributions do not reproduce the model logit "
            f"(mean absolute error {logit_mean_error:.4f} on a mean magnitude of {logit_scale:.4f})"
        )

    selected = contributions[:, keep]
    magnitude = selected.abs()
    total_magnitude = magnitude.sum(dim=0)
    shares = magnitude / total_magnitude.clamp_min(1e-9)
    sorted_shares = shares.sort(dim=0, descending=True).values
    entropy = -(shares.clamp_min(1e-12) * shares.clamp_min(1e-12).log()).sum(dim=0)
    participation = 1.0 / (shares.pow(2).sum(dim=0)).clamp_min(1e-12)

    per_component = selected.mean(dim=1).cpu().numpy()
    attention_rows = [index for index, name in enumerate(names) if name.startswith("attn.")]
    mlp_rows = [index for index, name in enumerate(names) if name.startswith("mlp.")]
    pathway_signed = {
        "embed": float(per_component[0]),
        "attention": float(per_component[attention_rows].sum()),
        "mlp": float(per_component[mlp_rows].sum()),
    }
    constant_mean = float(constant[keep].mean())
    total_signed = sum(pathway_signed.values()) + constant_mean
    pathway_magnitude = {
        "embed": float(magnitude[0].mean()),
        "attention": float(magnitude[attention_rows].sum(dim=0).mean()),
        "mlp": float(magnitude[mlp_rows].sum(dim=0).mean()),
    }
    magnitude_total = sum(pathway_magnitude.values())
    # Signed pathway contributions cancel: a strongly positive attention term and
    # a strongly negative MLP term can leave a target logit near zero. The signed
    # *fractions* then divide large numerators by a small denominator and read as
    # enormous pathway shares, and the denominator's size is arm-dependent
    # because the mean target logit is. The existing 1e-9 guard only catches
    # exact cancellation, so the cancellation ratio is reported and the fraction
    # is withheld once the signed total is negligible against the magnitude the
    # components actually carry.
    signed_over_magnitude = (
        abs(total_signed) / magnitude_total if magnitude_total > 0.0 else 0.0
    )
    signed_fraction_valid = signed_over_magnitude >= SIGNED_FRACTION_MINIMUM_RATIO

    return {
        "n_components": len(names),
        "component_names": names,
        "scored_positions": scored,
        "residual_relative_l2_error": _finite(residual_l2, "residual L2 reconstruction"),
        "residual_max_absolute_ratio": _finite(residual_max, "residual max reconstruction"),
        "logit_mean_absolute_error": _finite(logit_mean_error, "logit mean reconstruction"),
        "logit_max_absolute_error": _finite(
            float(logit_deviation.max()), "logit max reconstruction"
        ),
        "mean_target_logit": _finite(float(actual[keep].mean()), "target logit"),
        "normalisation_form": "layernorm" if form.centred else "rmsnorm",
        "normalisation_constant": _finite(constant_mean, "normalisation constant"),
        "mean_contribution": {
            name: _finite(float(value), f"contribution {name}")
            for name, value in zip(names, per_component)
        },
        "pathway_signed_mean": {
            key: _finite(value, f"signed {key}") for key, value in pathway_signed.items()
        },
        "pathway_signed_total_mean": _finite(total_signed, "signed total"),
        "pathway_signed_over_magnitude": _finite(
            signed_over_magnitude, "signed-to-magnitude ratio"
        ),
        "pathway_signed_fraction_valid": bool(signed_fraction_valid),
        "pathway_signed_fraction_minimum_ratio": SIGNED_FRACTION_MINIMUM_RATIO,
        "pathway_signed_fraction": {
            key: _finite(value / total_signed, f"signed fraction {key}")
            if signed_fraction_valid
            else None
            for key, value in pathway_signed.items()
        },
        "pathway_magnitude_fraction": {
            key: _finite(value / magnitude_total, f"magnitude fraction {key}")
            for key, value in pathway_magnitude.items()
        },
        "concentration": {
            "top1_share": _finite(float(sorted_shares[0].mean()), "top1 share"),
            "top5_share": _finite(float(sorted_shares[:5].sum(dim=0).mean()), "top5 share"),
            "normalised_entropy": _finite(
                float(entropy.mean()) / math.log(len(names)), "normalised entropy"
            ),
            "participation_ratio": _finite(
                float(participation.mean()), "participation ratio"
            ),
            "participation_fraction": _finite(
                float(participation.mean()) / len(names), "participation fraction"
            ),
        },
    }


# ---------------------------------------------------------- activation patching


#: Below this many resampling CLUSTERS a percentile interval on a fraction is
#: pinched inward rather than merely wide -- the resample distribution has too few
#: distinct atoms for its tails to mean anything. Measured coverage of a nominal
#: 95% interval is 0.74 / 0.82 / 0.89 / 0.94 at 3 / 4 / 8 / 400 units, so eight is
#: where the interval starts meaning roughly what it says.
#:
#: Named for clusters, not cases, because that is what it gates: a band with
#: twenty cases drawn from six sequences is refused. An earlier name said "cases"
#: and invited exactly the confusion the floor exists to prevent.
#:
#: This is the same number as ``statistics.MINIMUM_BOOTSTRAP_UNITS`` and is
#: imported from it rather than restated. An earlier version declared it locally
#: and justified that by "importing it would make the circuit census depend on the
#: homology module" -- which was wrong twice: the constant lives in ``statistics``,
#: which this module can depend on freely, and the claim that "the two are
#: declared to agree by test" named a test that did not exist.
MINIMUM_ELIGIBILITY_CLUSTERS = MINIMUM_BOOTSTRAP_UNITS

#: Minimum-effect thresholds the eligible fraction is reported at, beside the one
#: the run was scored on.
#:
#: The eligible fraction *is* the non-local propagation measurement, and it is a
#: fraction above a threshold — so Appendix B rule 8 applies directly: prefer a
#: threshold-free statistic, and where a threshold cannot be removed, sweep it and
#: show the ordering is invariant rather than asserting it. B6's first run reported
#: the fraction at one value (0.25 logits) and stored only the aggregate, so no
#: reader could recompute the curve or check the ordering at any other cut.
#: `absolute_effect_quantiles` below is the threshold-free companion: the
#: distribution the fraction is a single slice of.
ELIGIBILITY_THRESHOLD_LADDER: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 1.00, 2.00)


def _threshold_row(
    absolute: np.ndarray,
    sources: np.ndarray,
    threshold: float,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    """One row of the eligibility sweep: the fraction above ``threshold`` and its interval.

    Computed on the same float64 array the headline fraction uses, and from a
    generator seeded per threshold rather than one stream shared across the sweep,
    so the row at the run's own cut is bit-identical to
    ``eligible_fraction_interval`` instead of being a second estimate of it.
    """

    flags = absolute >= threshold
    return _case_resampled_interval(
        flags, sources, np.random.default_rng(seed), resamples
    ) | {"threshold": float(threshold), "fraction": float(flags.mean())}


def _case_resampled_interval(
    flags: np.ndarray,
    sources: np.ndarray,
    generator: np.random.Generator,
    resamples: int,
) -> dict[str, Any]:
    """Percentile interval for an eligibility fraction, clustered by source sequence.

    **The sampling unit is the sequence, not the case.** Cases are drawn with
    replacement from the cohort's usable rows, so at a production case count they
    are many corruptions of the *same* few sequences -- 512 cases per band from a
    24-record cohort is about twenty cases per sequence. Whether a single-token
    change propagates 33-64 positions is a property of the sequence at least as
    much as of the position, so resampling cases would treat twenty correlated
    observations as twenty independent ones and report an interval too narrow by
    roughly the square root of that factor. Resampling sequences and taking all
    of a drawn sequence's cases is the same rule the rest of the package applies.

    Returns ``degenerate`` rather than an interval below the floor: a percentile
    interval over four units reads *narrower* than one over eight, and that
    pinching decided two verdicts in this programme before (EXP-R2-061).
    """

    n = int(flags.size)
    clusters = np.unique(sources)
    if clusters.size < MINIMUM_ELIGIBILITY_CLUSTERS:
        return {
            "n_cases": n,
            "n_source_sequences": int(clusters.size),
            "degenerate": True,
            "reason": (
                f"{clusters.size} source sequences is below the "
                f"{MINIMUM_ELIGIBILITY_CLUSTERS}-cluster floor; a percentile interval "
                "here is pinched inward rather than wide, so none is reported"
            ),
        }
    members = [np.flatnonzero(sources == cluster) for cluster in clusters]
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        picked = generator.integers(0, clusters.size, size=clusters.size)
        taken = np.concatenate([members[int(choice)] for choice in picked])
        draws[index] = flags[taken].mean()
    return {
        "n_cases": n,
        "n_source_sequences": int(clusters.size),
        "resampling_unit": "source_sequence",
        "degenerate": False,
        "resamples": int(resamples),
        "q025": float(np.quantile(draws, 0.025)),
        "q975": float(np.quantile(draws, 0.975)),
    }


@dataclass(frozen=True)
class PatchCase:
    """One clean/corrupted pair with its perturbation and read-out positions."""

    clean_ids: tuple[int, ...]
    corrupt_ids: tuple[int, ...]
    position_p: int
    position_q: int
    band: str
    #: Index of the cohort row this case was cut from. Cases are drawn with
    #: replacement, so several cases share a source; that is the cluster any
    #: interval over cases has to resample. Defaulted so the frozen artefacts and
    #: the tests that construct cases by hand keep working, and -1 is a distinct
    #: cluster per the ``np.unique`` in :func:`_case_resampled_interval` only when
    #: every case carries it, which is the honest reading of "source unrecorded".
    source: int = -1

    def __post_init__(self) -> None:
        if len(self.clean_ids) != len(self.corrupt_ids):
            raise ValueError("clean and corrupted inputs must have the same length")
        if not 0 <= self.position_p < self.position_q < len(self.clean_ids):
            raise ValueError("patching requires 0 <= p < q < sequence length")
        differing = [
            index
            for index, (left, right) in enumerate(zip(self.clean_ids, self.corrupt_ids))
            if left != right
        ]
        if differing != [self.position_p]:
            raise ValueError("the corrupted input must differ at exactly position p")


def build_patch_cases(
    arm: Arm,
    strings: Sequence[str],
    unigram: Unigram,
    *,
    seq_len: int,
    bands: Sequence[tuple[int, int]] = DISTANCE_BANDS,
    cases_per_band: int,
    seed: int,
) -> list[PatchCase]:
    """Sample one single-token corruption per case, at a controlled ``q - p``.

    All cases are cut to the same token length so that the whole sweep runs as
    one batch and no padding position can enter the measurement.

    Both the perturbed and the read-out position are required to hold a token in
    the unigram support, which excludes layout tokens.  Corrupting a line break,
    or reading the next-token logit off one, would make the measurement partly
    about a rendering's layout for the one arm whose rendering has layout tokens.
    """

    if seq_len < 8 or cases_per_band < 1 or not bands:
        raise ValueError("invalid patch-case parameters")
    usable: list[list[int]] = []
    for text in strings:
        row = [int(token) for token in arm.tokenizer(text, return_tensors=None)["input_ids"]]
        if len(row) < seq_len:
            continue
        usable.append(row[:seq_len])
    if not usable:
        raise RuntimeError(f"{arm.name}: no cohort record reaches {seq_len} tokens")
    support = {int(token) for token in unigram.token_ids}
    rng = np.random.default_rng(seed)
    cases: list[PatchCase] = []
    for low, high in bands:
        if low < 1 or high < low:
            raise ValueError(f"invalid distance band ({low}, {high})")
        label = f"{low}-{high}"
        produced = 0
        for _ in range(cases_per_band * 64):
            source = int(rng.integers(0, len(usable)))
            row = usable[source]
            begin, end = content_bounds(arm, row, len(row))
            begin = max(begin, 1)
            distance = int(rng.integers(low, high + 1))
            if end - begin <= distance:
                continue
            position_p = int(rng.integers(begin, end - distance))
            position_q = position_p + distance
            if row[position_p] not in support or row[position_q] not in support:
                continue
            corrupted = list(row)
            corrupted[position_p] = unigram.sample_other(rng, row[position_p])
            cases.append(
                PatchCase(
                    clean_ids=tuple(row),
                    corrupt_ids=tuple(corrupted),
                    position_p=position_p,
                    position_q=position_q,
                    band=label,
                    source=source,
                )
            )
            produced += 1
            if produced >= cases_per_band:
                break
        if produced < cases_per_band:
            raise RuntimeError(
                f"{arm.name}: only {produced}/{cases_per_band} cases for band {label} at "
                f"seq_len={seq_len}; the band does not fit the content span"
            )
    return cases


def _component_modules(arm: Arm) -> dict[tuple[str, int], torch.nn.Module]:
    modules: dict[tuple[str, int], torch.nn.Module] = {}
    for layer, block in enumerate(arm.blocks()):
        modules[("attn_out", layer)] = arm.attention(layer)
        modules[("mlp_out", layer)] = arm.mlp(layer)
        modules[("resid_post", layer)] = block
    return modules


@torch.no_grad()
def activation_patching(
    arm: Arm,
    cases: Sequence[PatchCase],
    *,
    minimum_effect: float,
    component_kinds: Sequence[str] = COMPONENT_KINDS,
    batch_size: int = 64,
    eligibility_resamples: int = 2000,
    eligibility_seed: int = 20260730,
) -> dict[str, Any]:
    """Recovered fraction when one component at one position is restored.

    The read-out is the logit difference at ``q`` between the clean run's top-1
    token and the clean run's rank-2 token.  Both tokens are fixed once from the
    clean run and reused for the corrupted and patched runs, so the metric is a
    genuine difference-in-differences rather than a moving target.

    Cases whose corruption moves the metric by less than ``minimum_effect``
    logits are excluded from the recovered-fraction average, because the ratio is
    then a ratio of noise.  The exclusion rate is reported per band and is itself
    a measurement: a modality where a single-token change never propagates has no
    circuit for patching to find.

    **Forward passes run in chunks of ``batch_size`` cases.**  Every pass used to
    run the whole case set at once, and ``read_at_q`` materialises the model's
    full ``[cases, width, vocab]`` logit tensor before selecting the single
    position it needs.  At the historical 192 cases that is about 2.5 GB and
    invisible; at the case count the far bands actually need it is the binding
    constraint, because the non-local propagation figure rests on 2-16 eligible
    cases in the 33-64 band and reaching thirty per arm takes roughly sixteen
    times as many.  Chunking bounds the peak at one chunk's logits and changes no
    number: the hooks, the metric and the eligibility rule are per case and carry
    no cross-case state.

    ``eligible_fraction_interval`` accompanies each band's exclusion rate.  The
    fraction *is* the measurement here -- "a single-token change never
    propagates" is the claim -- and a bare ratio over as few as two cases cannot
    say whether two arms differ.
    """

    arm.require("circuits")
    if not cases:
        raise ValueError("no patching cases were supplied")
    if minimum_effect <= 0:
        raise ValueError("minimum_effect must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    unknown = set(component_kinds) - set(COMPONENT_KINDS)
    if unknown:
        raise ValueError(f"unknown component kinds {sorted(unknown)}")
    width = len(cases[0].clean_ids)
    if any(len(case.clean_ids) != width for case in cases):
        raise ValueError("all patching cases must share one sequence length")

    device = arm.device
    clean = torch.tensor([case.clean_ids for case in cases], dtype=torch.long, device=device)
    corrupt = torch.tensor([case.corrupt_ids for case in cases], dtype=torch.long, device=device)
    sites = {
        "p": torch.tensor([case.position_p for case in cases], dtype=torch.long, device=device),
        "q": torch.tensor([case.position_q for case in cases], dtype=torch.long, device=device),
    }
    chunks = [
        slice(start, min(start + batch_size, len(cases)))
        for start in range(0, len(cases), batch_size)
    ]

    modules = _component_modules(arm)
    cache: dict[tuple[str, int, str], torch.Tensor] = {}
    handles = []

    # Filled chunk by chunk on the clean pass. The read-out tokens are fixed
    # there and reused everywhere after, which is what makes every later pass a
    # difference against a constant rather than against a moving target.
    top_token = torch.zeros(len(cases), dtype=torch.long, device=device)
    alternative = torch.zeros(len(cases), dtype=torch.long, device=device)

    def capture(kind: str, layer: int, span: slice, local_rows: torch.Tensor):
        def hook(_module, _args, output: Any) -> None:
            tensor = _leading_tensor(output)
            for site, index in sites.items():
                key = (kind, layer, site)
                if key not in cache:
                    cache[key] = torch.zeros(
                        (len(cases), tensor.shape[-1]), dtype=tensor.dtype, device=device
                    )
                cache[key][span] = tensor[local_rows, index[span]].detach()

        return hook

    def logits_at_q(ids: torch.Tensor, span: slice, local_rows: torch.Tensor) -> torch.Tensor:
        """Logits at the read-out position only, for one chunk of cases."""

        logits = arm.model(input_ids=ids[span], use_cache=False).logits
        return logits[local_rows, sites["q"][span]].float()

    metric_clean = torch.zeros(len(cases), dtype=torch.float32, device=device)
    for span in chunks:
        local_rows = torch.arange(span.stop - span.start, device=device)
        handles = [
            module.register_forward_hook(capture(kind, layer, span, local_rows))
            for (kind, layer), module in modules.items()
            if kind in component_kinds
        ]
        try:
            read = logits_at_q(clean, span, local_rows)
        finally:
            for handle in handles:
                handle.remove()
        ranked = read.topk(2, dim=-1)
        top_token[span] = ranked.indices[:, 0]
        alternative[span] = ranked.indices[:, 1]
        metric_clean[span] = (
            read.gather(1, ranked.indices[:, :1]).squeeze(1)
            - read.gather(1, ranked.indices[:, 1:2]).squeeze(1)
        )

    def metric(read: torch.Tensor, span: slice) -> torch.Tensor:
        return read.gather(1, top_token[span].unsqueeze(1)).squeeze(1) - read.gather(
            1, alternative[span].unsqueeze(1)
        ).squeeze(1)

    def corrupt_metric(patch_site: tuple[str, int, str] | None) -> torch.Tensor:
        """The metric on the corrupted input, optionally with one component restored."""

        out = torch.zeros(len(cases), dtype=torch.float32, device=device)
        for span in chunks:
            local_rows = torch.arange(span.stop - span.start, device=device)
            handle = None
            if patch_site is not None:
                kind, layer, site = patch_site
                cached = cache[(kind, layer, site)][span]
                index = sites[site][span]

                def hook(_module, _args, output: Any) -> Any:
                    tensor = _leading_tensor(output)
                    patched = tensor.clone()
                    patched[local_rows, index] = cached.to(patched.dtype)
                    if isinstance(output, tuple):
                        return (patched,) + tuple(output[1:])
                    return patched

                handle = modules[(kind, layer)].register_forward_hook(hook)
            try:
                out[span] = metric(logits_at_q(corrupt, span, local_rows), span)
            finally:
                if handle is not None:
                    handle.remove()
        return out

    metric_corrupt = corrupt_metric(None)
    denominator = metric_clean - metric_corrupt
    eligible = denominator.abs() >= minimum_effect

    def patch(kind: str, layer: int, site: str) -> torch.Tensor:
        return corrupt_metric((kind, layer, site))

    bands = sorted({case.band for case in cases}, key=lambda label: int(label.split("-")[0]))
    band_index = {
        band: torch.tensor(
            [i for i, case in enumerate(cases) if case.band == band],
            dtype=torch.long,
            device=device,
        )
        for band in bands
    }

    case_sources = np.asarray([case.source for case in cases], dtype=np.int64)
    corruption: dict[str, Any] = {}
    for band, index in band_index.items():
        effects = denominator.index_select(0, index)
        # float64 on the host, once, and everything downstream reads it: the
        # headline fraction, its interval and every row of the sweep. Computing
        # the headline from a numpy bool array and the sweep from a float32 torch
        # reduction made the two disagree in the eighth decimal for any
        # non-dyadic fraction, which reads as a defect to anyone diffing them.
        absolute = effects.abs().detach().cpu().numpy().astype(np.float64)
        flags = absolute >= float(minimum_effect)
        sources = case_sources[index.detach().cpu().numpy()]
        corruption[band] = {
            "n_cases": int(index.numel()),
            "mean_absolute_effect": _finite(float(effects.abs().mean()), f"effect {band}"),
            "median_absolute_effect": _finite(
                float(effects.abs().median()), f"median effect {band}"
            ),
            "eligible_cases": int(flags.sum()),
            "eligible_fraction": float(flags.mean()),
            # The exclusion rate is the measurement, not bookkeeping: "a
            # single-token change does not propagate this far" is the claim the
            # non-local propagation result makes. A bare ratio over as few as two
            # cases cannot support a comparison between arms, and two of the four
            # arms had fewer than ten eligible far-band cases when that result was
            # first reported.
            "eligible_fraction_interval": _case_resampled_interval(
                flags, sources, np.random.default_rng(eligibility_seed), eligibility_resamples
            ),
            # Appendix B rule 8. The fraction at one cut cannot show that an
            # ordering across arms is a property of the arms rather than of the
            # cut, and an aggregate-only artefact cannot be re-cut afterwards.
            #
            # Three consistency properties are enforced rather than hoped for.
            # The run's own `minimum_effect` is *in* the ladder, so the swept
            # curve always contains the number reported beside it -- the ladder
            # is a fixed tuple and `minimum_effect` is a free float, so without
            # this the two could not be compared at all. The fractions are
            # computed on the same float64 numpy array as `eligible_fraction`,
            # not a float32 torch reduction, so the shared entry is bit-equal
            # rather than equal to eight decimals. And every threshold draws
            # from one generator seeded per threshold, so the interval at the
            # run's cut is the same interval as `eligible_fraction_interval`
            # rather than a second one computed off a diverged stream.
            "eligible_fraction_by_threshold": {
                f"{threshold:g}": _threshold_row(
                    absolute, sources, threshold, eligibility_seed, eligibility_resamples
                )
                for threshold in sorted(
                    set(ELIGIBILITY_THRESHOLD_LADDER) | {float(minimum_effect)}
                )
            },
            # The threshold-free companion: the distribution every fraction above
            # is one slice of, so a reader can re-cut at any value without a re-run.
            "absolute_effect_quantiles": {
                f"q{int(q * 1000):03d}": _finite(
                    float(np.quantile(absolute, q)), f"|effect| quantile {q} {band}"
                )
                for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
            },
            "mean_clean_metric": _finite(
                float(metric_clean.index_select(0, index).mean()), f"clean metric {band}"
            ),
        }

    recovered: dict[str, Any] = {}
    for kind in component_kinds:
        for layer in range(arm.n_layer):
            for site in ("p", "q"):
                patched = patch(kind, layer, site)
                fraction = (patched - metric_corrupt) / denominator
                entry: dict[str, Any] = {}
                for band, index in band_index.items():
                    keep = eligible.index_select(0, index)
                    values = fraction.index_select(0, index)[keep]
                    entry[band] = (
                        {
                            "n": int(values.numel()),
                            "mean": _finite(float(values.mean()), "recovered mean"),
                            "median": _finite(float(values.median()), "recovered median"),
                        }
                        if values.numel() > 0
                        else {"n": 0, "mean": None, "median": None}
                    )
                recovered[f"{kind}|{layer}|{site}"] = entry

    return {
        "sequence_length": width,
        "n_cases": len(cases),
        "minimum_effect_logits": float(minimum_effect),
        "eligible_cases": int(eligible.sum()),
        "component_kinds": list(component_kinds),
        "bands": bands,
        "corruption_effect": corruption,
        "recovered_fraction": recovered,
    }


def summarise_patching(result: Mapping[str, Any], *, arm: Arm) -> dict[str, Any]:
    """Collapse the per-(kind, layer, site) map into a per-pathway reading.

    ``best_layer`` is the layer index, resolved through the list of layers that
    actually produced a mean. Taking ``argmax`` of the compacted value list and
    publishing it as a layer was wrong whenever any layer was dropped: it would
    have named an earlier layer than the true peak and divided it by
    ``n_layer - 1`` to give a relative depth that no measurement supports. In the
    current design eligibility is a property of the case, not of the layer, so
    the compaction is all-or-nothing and the two indices coincide -- which is
    exactly the kind of latency that survives review and then stops being true.
    ``n_layers_with_a_mean`` is reported so the compaction is visible.
    """

    recovered = result["recovered_fraction"]
    summary: dict[str, Any] = {}
    for kind in result["component_kinds"]:
        for site in ("p", "q"):
            per_band: dict[str, Any] = {}
            for band in result["bands"]:
                measured = [
                    (layer, recovered[f"{kind}|{layer}|{site}"][band]["mean"])
                    for layer in range(arm.n_layer)
                    if recovered[f"{kind}|{layer}|{site}"][band]["mean"] is not None
                ]
                if not measured:
                    per_band[band] = {
                        "best_layer": None,
                        "best_mean": None,
                        "layer_mean": None,
                        "n_layers_with_a_mean": 0,
                    }
                    continue
                layers = [layer for layer, _ in measured]
                array = np.asarray([value for _, value in measured], dtype=np.float64)
                position = int(np.argmax(array))
                best_layer = layers[position]
                per_band[band] = {
                    "best_layer": best_layer,
                    "best_layer_fraction": _finite(
                        best_layer / max(arm.n_layer - 1, 1), "best layer fraction"
                    ),
                    "best_mean": _finite(float(array[position]), "best recovered mean"),
                    "layer_mean": _finite(float(array.mean()), "layer-mean recovered"),
                    "n_layers_with_a_mean": len(measured),
                }
            summary[f"{kind}|{site}"] = per_band
    return summary
