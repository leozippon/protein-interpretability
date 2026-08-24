"""D3.j variant (a): does a residue behave like a chemical entity, or like a mark?

Why this module exists
======================

Three campaigns in a row -- F10, F12 and D3.g's stage 35 -- ended at the same
wall, and §7.0 of the audit names the reason: each measured on an **agreement
set**, a population where evolutionary statistics and biological knowledge
predict the same outcome. There a pass is consistent with recombination and a
loss to a simple baseline is the expected behaviour of the weaker of two
estimators of one shared quantity, so neither outcome says anything about
knowledge, at any effect size and with any instrument.

EXP-R2-214 pre-registers three successors under that rule. This module is the
whole of **D3.j variant (a)**, the cheapest of them: its contradiction set is
computable on CPU from the k-mer background already on disk, so whether the set
exists at all is answerable before a checkpoint is loaded.

The operation
=============

For an ordered residue pair ``r -> s`` the input embedding row of ``r`` is
overwritten with that of ``s``, so the model reads every occurrence of ``r`` as
if it were ``s``, and the change in held-out NLL is measured. **The output head
and every other parameter are untouched**, so the model must still predict ``r``;
what the damage measures is how much the downstream computation depends on
``r``'s own input identity.

That last sentence is a requirement on the code, not a description of it. GPT-2
ties ``lm_head.weight`` to ``transformer.wte.weight`` -- ZymCTRL, ProtGPT2 and
gpt2-large all inherit the tie, ProGen2 and ByGPT5 do not -- and under a tie the
write also deletes ``r``'s output class, which is a different and much larger
intervention that would be reported under this stage's name.
:class:`ArmAlphabetModel` clones the head first and records that it did.

What is scored, and what is confounded
======================================

A swap at ``r`` can matter only where the model reads an ``r``. Scoring every
position would divide a fixed effect by sequence length and make per-token damage
proportional to the **corpus frequency of r** -- a corpus statistic, and exactly
the quantity this design holds apart from chemistry. The scored population is
therefore the next-token targets whose immediately preceding input token is
``r``: damage per read of ``r``.

Two positions are then excluded, and they are the ones the pair's own identity
reaches. ``target == r`` asks the model to predict the symbol whose reading it
has just lost. ``target == s`` is the opposite hazard: the corrupted context now
looks like an ``s``, and where ``s`` follows ``s`` often the prediction gets
*easier*, entering the measurement as negative damage produced by the label
alone. Both exclusions are per pair; both arms of the paired difference use the
identical mask; damage is per scored token, so masks of different sizes stay
commensurable; and the excluded count is recorded for every pair.

**What is not measured**: damage at longer range. A swap corrupts the residual
stream at every position after the first ``r``, and only the position
immediately after each ``r`` is read. That narrows the estimand, deliberately,
to the place where the effect is a function of the swap rather than of how far
the model has since recovered.

The two axes (D3.j-A2), and which side each belongs to
======================================================

*Chemical similarity* is the declared free physicochemical descriptor set:
Kyte-Doolittle hydropathy, side-chain volume and formal charge at pH 7 from
:data:`src.transfer.concept_lens.PROPERTY_BASIS`, plus Grantham polarity
(:data:`GRANTHAM_POLARITY`), z-scored, Euclidean. **No corpus is read.**

*Corpus-distributional similarity* is each residue's context profile over the
staged UniRef50 3-mer background: the normalised distribution over ``(left,
right)`` neighbour pairs, compared by cosine, with symmetric KL as the declared
alternative and both reported.

**BLOSUM62 is on the ceiling side and is never the chemical axis.** It is
estimated from aligned families, so under §7.0 clause 1 it *is* evolutionary
statistics; a design that used it as the chemistry axis would compare statistics
against statistics on both axes and would look like a chemistry result while
being a statistics result. It is vendored here as a second statistics estimator
and reported beside the k-mer axis, which is a check worth having and which has
already returned something: over the 190 unordered pairs the two statistics
estimators agree at Spearman **-0.010** while BLOSUM62 correlates with the
physicochemical axis at **+0.386**. Substitution scores and corpus co-occurrence
are not one quantity, and any design that treated them as interchangeable would
have been wrong about which side it was measuring.

A pair is admitted where its two similarity ranks fall in **opposite quantile
bands** of their own distributions, giving two quadrants, and the cut is swept at
terciles, quartiles and quintiles (rule 17). Admission is computed over the 190
**unordered** pairs, because the two similarities are symmetric; the intervention
is then run on both **ordered** directions of every admitted pair, which are
separate observations in separate resampling groups.

The estimand (D3.j-A4) and the ceiling (D3.j-A5)
================================================

``Delta_chem = mean D over chemically-dissimilar / distributionally-similar
pairs - mean D over chemically-similar / distributionally-dissimilar pairs``.

The chemical account predicts ``Delta_chem > 0``; the corpus-symbol account
predicts ``Delta_chem < 0``. Opposite signs on the same set is what admits the
design under §7.0 clause 4, and it is why a correlation is *not* the frozen
statistic here: the quadrants were built so that a mean difference between them
has a sign each hypothesis claims.

Clearing a shuffled null admits nothing. The thing to clear is the **recombination
ceiling**: the identical substitution applied to the UniRef50 3-mer conditional
(:class:`FragmentConditional`), scored on the same held-out sequences under the
same statistic. The fragment model has no chemistry, so its ``Delta_chem`` is
what the distributional account predicts, and the arm must clear it in the
chemical direction under the standing margin -- the paired group-bootstrap 95%
interval of the difference excluding zero over at least eight groups, and the
arm's own effect at least twice the ceiling's positive part. A norm-matched
random-substitute control runs beside it over at least eight distinct random
substitute **rows** -- more directions, not more positions (rule 39) -- and the
effect must exceed their 95th percentile.

Gates, in the order they bind
=============================

**D3.j-A0, arm admission, is a measurement.** :func:`symbol_token_coverage`
measures what fraction of the alphabet characters in the arm's own scored window
are carried by tokens that are exactly one alphabet character, and an arm below
:data:`MINIMUM_SYMBOL_TOKEN_COVERAGE` is *not measurable* with its coverage as
the reported result. No arm is excluded by name. The residue-tokenised arms
reach 1.000 by construction, which is why 99% is the right bar; a multi-residue
BPE arm falls far below it, and the twenty single-residue pieces its vocabulary
does contain would otherwise have been swapped to measure the rare positions
where a residue happened to be tokenised alone -- a cohort selected by local
context rather than at random (L31: on the panel's BPE protein arm a single
substitution leaves mutant and wild type token-aligned on 47.0-54.5% of
instances, and the survivors are the BPE-stable subset), reported in a unit that
is not the other arms' unit (L23).

**D3.j-A1, the byte-level text control, validates only the half text can
validate.** Text has no chemistry, so the control establishes only that the
readout detects *substitute similarity at all*: damage under a distributionally
similar substitute must be smaller than under a dissimilar one, with the paired
interval excluding zero over at least eight groups. If it does not, the readout
is VOID as a specification defect and no protein arm is read -- which is enforced
here by the protein cell requiring the text control's own artefact and refusing
on anything but a PASS, rather than by a promise.

**D3.j-A3, the contradiction set's own attainability, runs first and on CPU.**
At least eight pairs are required in each quadrant at the declared cut. Falling
short is a live outcome and is a statement about the alphabet and the corpus, not
about any model. Measured on the staged background it is live in exactly the way
the pre-registration anticipated: 17 and 18 pairs per quadrant at terciles, 9 and
11 at quartiles, and **5 and 9 at quintiles**, so the strictest rung of the sweep
does not reach the floor.

:func:`intervention_invariants` binds before either, because a write that does
not land passes every null. It requires an identity substitution to move the
likelihood by **exactly** zero and a seeded random direction of the same norm to
move it, and it applies a *constant* vector of that norm beside them as a
reported diagnostic -- never as the control -- for the reason
``src.transfer.das.invariants`` records: a layer norm's mean subtraction removes
a constant exactly, so a constant-vector positive control reports a bound write
as unbound.

No resampler is defined in this module. :func:`~src.transfer.statistics.
paired_group_bootstrap` is imported, its eight-unit floor is checked before it is
called, and a pair set resolving into fewer than eight substituted symbols is
reported with its point estimate and no interval.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy import stats

from .arms import AA20, Arm
from .io import sha256_file
from .concept_lens import PROPERTY_BASIS
from .kmer_background import ALPHABET as KMER_ALPHABET, KmerBackground
from .near_duplicates import near_duplicate_groups
from .statistics import (
    MINIMUM_BOOTSTRAP_UNITS,
    bootstrap_unit_floor,
    paired_group_bootstrap,
)

# --------------------------------------------------------------- declarations

#: The register entry every threshold here is quoted from, and the variant of it
#: this module implements.
PRE_REGISTRATION = "EXP-R2-214"
PRE_REGISTRATION_TRACK = "D3.j variant (a), embedding substitution"

#: D3.j-B is a separately identified successor. Selecting A, including by
#: default, must not write these names into an A artefact.
PRE_REGISTRATION_TRACK_B = "D3.j-B, fragment-substitution-damage admission axis"
EXPERIMENT_B = "D3.j-B"
B_STAGE_CONSTRUCT = "construct"
B_STAGE_CONFIRM = "confirm"
B_STAGES = (B_STAGE_CONSTRUCT, B_STAGE_CONFIRM)
B_CONFIRMATION_INDICES = (1, 2)
KIND_AXIS_CONSTRUCTION = "axis_construction"
AXIS_CONSTRUCTED = "AXIS_CONSTRUCTED"
CROSSED_INTERVAL_REFUSED = "CROSSED_INTERVAL_REFUSED"
PROTEIN_AXIS_CONTEXT_PROFILE = "context_profile"
PROTEIN_AXIS_FRAGMENT_SUBSTITUTION_DAMAGE = "fragment_substitution_damage"
PROTEIN_AXES = (
    PROTEIN_AXIS_CONTEXT_PROFILE,
    PROTEIN_AXIS_FRAGMENT_SUBSTITUTION_DAMAGE,
)
FRAGMENT_AXIS_SYMMETRIZATION = "arithmetic_mean_of_directional_damages"
CEILING_CONSTRUCTION_VOID = "CEILING_DOES_NOT_PREDICT_DISTRIBUTIONAL_SIDE"

#: Amendments to that entry which this module implements. Empty and *declared*:
#: an artefact carrying no amendment field and one produced under the unamended
#: text are indistinguishable to a reader, which ``36_concept_injection.py``
#: records the cost of.
PRE_REGISTRATION_AMENDMENTS: tuple[str, ...] = ()

#: The intervention's name. It is part of every artefact basename, because a
#: stage whose output file does not name its intervention cannot be told apart
#: from a successor that swaps something else.
INTERVENTION = "input_embedding_row_swap"

#: Inference dtype, fixed rather than exposed: rule 15b requires float32 for a
#: difference of order 0.01-0.1 nats, and this estimand is one.
DTYPE = "float32"

#: The quantile bands the admission cut is swept over (rule 17). A pair is
#: admitted when one similarity is below the band's lower quantile and the other
#: above its upper quantile.
CUTS: dict[str, float] = {"tercile": 1.0 / 3.0, "quartile": 0.25, "quintile": 0.2}

#: The two quadrants of the contradiction set, in the order D3.j-A4's contrast
#: subtracts them: ``Delta = mean over [1] - mean over [0]``.
QUADRANTS = (
    "chemically_similar_distributionally_dissimilar",
    "chemically_dissimilar_distributionally_similar",
)

#: The two ends of the agreement set, which the reachability check reads.
AGREEMENT_CLASSES = ("both_dissimilar", "both_similar")

#: D3.j-A3's floor: at least eight unordered pairs in each quadrant at the
#: declared cut. It is the shared unit floor because it is the same object -- the
#: quadrant mean is what the interval is taken over.
MINIMUM_QUADRANT_PAIRS = MINIMUM_BOOTSTRAP_UNITS

#: D3.j-A0's bar. Attainable at 1.000 by construction on a symbol-tokenised arm,
#: which is what makes it the right bar rather than a round number.
MINIMUM_SYMBOL_TOKEN_COVERAGE = 0.99

#: D3.j-A5, rule 39: the random-substitute control buys directions, not
#: positions, so its draw count is a count of distinct substitute rows.
MINIMUM_RANDOM_DIRECTIONS = 8

#: Every order the staged higher-order background supports. The ceiling is
#: reported as a **curve** over these rather than at one k, which is EXP-R2-214's
#: pre-data amendment of 2026-08-19: at k = 3 the fragment conditional's damage is
#: 1.7-3.0% of a decoder's on this estimand, so "at least twice the ceiling"
#: degenerates into "greater than zero" and audit §7.0's null stops binding. The
#: curve makes visible where it starts to bind and whether a verdict survives as
#: it does.
#:
#: **k = 1 is the curve's own reachability anchor.** A unigram conditional reads no
#: context, so no substitution can change it and its Delta is exactly zero by
#: construction. A curve whose first point is not exactly zero is a defect in the
#: indexing, not a fact about the corpus.
FRAGMENT_ORDERS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)

#: The order EXP-R2-214 froze, kept in the table so the amendment's effect is
#: visible rather than substituted.
PRE_REGISTERED_FRAGMENT_ORDER = 3

#: The share of the arm's own damage the ceiling must itself produce before
#: "twice the ceiling" is a bar with teeth. It is a **declared diagnostic and not
#: a gate**: EXP-R2-214 fixes the margin and inventing a second blocking clause
#: after seeing a number would be the failure this programme catalogues, so the
#: adequacy is reported beside every verdict instead. It exists because a ceiling
#: that does nothing is trivially cleared, and a CHEMISTRY verdict obtained
#: against a flat ceiling must not be quotable without that fact attached.
CEILING_ADEQUACY_FLOOR = 0.1

#: The declared distributional metrics. The first decides; the second is
#: reported beside it so that the cut is not an artefact of one metric.
DISTRIBUTIONAL_METRICS = ("cosine", "symmetric_kl")

#: Where the input embedding lives, per declared architecture. Declared and never
#: searched, for the reason ``arms._ATTENTION_PATH`` is: ProGen2's remote code
#: does not implement ``get_input_embeddings`` at all -- it raises
#: ``NotImplementedError`` -- so a stage reaching for the standard accessor would
#: fail on three of the four protein arms, and one that searched a module tree for
#: the first ``nn.Embedding`` would silently resolve a positional table.
INPUT_EMBEDDING_PATH: dict[str, tuple[str, ...]] = {
    "gpt2": ("transformer", "wte"),
    "progen": ("transformer", "wte"),
    "t5_decoder": ("shared",),
}

#: Where the output head lives, per declared architecture. Needed only to break
#: the tie, and declared beside the embedding it may be tied to.
OUTPUT_HEAD_PATH: dict[str, tuple[str, ...]] = {
    "gpt2": ("lm_head",),
    "progen": ("lm_head",),
    "t5_decoder": ("lm_head",),
}

#: Grantham polarity, the fourth descriptor D3.j-A2 declares beside
#: :data:`src.transfer.concept_lens.PROPERTY_BASIS`'s hydropathy, charge and
#: volume. A property of the side chain rather than of what it aligns with or
#: occurs beside, which is what puts it on the chemistry side of §7.0.
GRANTHAM_POLARITY: dict[str, float] = {
    "A": 8.1, "R": 10.5, "N": 11.6, "D": 13.0, "C": 5.5,
    "Q": 10.5, "E": 12.3, "G": 9.0, "H": 10.4, "I": 5.2,
    "L": 4.9, "K": 11.3, "M": 5.7, "F": 5.2, "P": 8.0,
    "S": 9.2, "T": 8.6, "W": 5.4, "Y": 6.2, "V": 5.9,
}

GRANTHAM_SOURCE = (
    "Grantham R. (1974), 'Amino acid difference formula to help explain protein "
    "evolution', Science 185:862-864, column p (polarity)"
)

CHEMICAL_AXIS_SOURCE = (
    "src.transfer.concept_lens.PROPERTY_BASIS (Kyte-Doolittle hydropathy, formal "
    "charge at pH 7, side-chain volume) extended by Grantham polarity; each "
    "z-scored across the alphabet, combined as a Euclidean distance. No corpus is "
    "read, which is what places this axis on the biology side of audit §7.0"
)

BLOSUM62_SOURCE = (
    "Henikoff S. and Henikoff J.G. (1992), 'Amino acid substitution matrices from "
    "protein blocks', PNAS 89:10915-10919; the matrix as distributed by NCBI, in "
    "its published ARNDCQEGHILKMFPSTWYV row order"
)

BLOSUM62_SIDE_NOTE = (
    "BLOSUM62 is estimated from aligned homologous families, so under audit §7.0 "
    "clause 1 it is evolutionary statistics and belongs on the CEILING side. It is "
    "never the chemical axis here: a design that used it as one would compare "
    "statistics against statistics on both axes and would read as a chemistry "
    "result. It is carried as a second statistics estimator beside the k-mer axis"
)

#: The published order of :data:`BLOSUM62_ROWS`, so the constant can be checked
#: against any published copy by eye; the reordering onto AA20 is code, where it
#: is testable.
BLOSUM62_ORDER = "ARNDCQEGHILKMFPSTWYV"

BLOSUM62_ROWS: tuple[tuple[int, ...], ...] = (
    (4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0),
    (-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3),
    (-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3),
    (-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3),
    (0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1),
    (-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2),
    (-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2),
    (0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3),
    (-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3),
    (-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3),
    (-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1),
    (-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2),
    (-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1),
    (-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1),
    (-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2),
    (1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2),
    (0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0),
    (-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3),
    (-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1),
    (0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4),
)


# ------------------------------------------------------------------ alphabets


@dataclass(frozen=True)
class Symbol:
    """One member of an arm's alphabet: a label and the single token that is it."""

    label: str
    token_id: int


def protein_alphabet(arm: Arm) -> tuple[Symbol, ...]:
    """The twenty residues and the twenty tokens that are them.

    Resolved by decoding the vocabulary rather than by encoding each residue, so
    a tokenizer that spells a residue with two pieces is caught here. Finding the
    twenty rows is **not** admission: a multi-residue BPE vocabulary contains the
    single-character pieces too, and D3.j-A0's coverage measurement is what
    decides whether those rows are the alphabet the model actually reads.
    """

    if arm.modality != "protein":
        raise ValueError(f"{arm.name} is a {arm.modality} arm; expected protein")
    return _single_character_alphabet(arm, set(AA20), ordered=AA20)


def text_alphabet(arm: Arm) -> tuple[Symbol, ...]:
    """The ASCII letters, for the byte-level text control of D3.j-A1."""

    if arm.modality != "text":
        raise ValueError(f"{arm.name} is a {arm.modality} arm; expected text")
    letters = "".join(chr(code) for code in range(ord("a"), ord("z") + 1))
    letters += "".join(chr(code) for code in range(ord("A"), ord("Z") + 1))
    return _single_character_alphabet(arm, set(letters), ordered=letters)


def _single_character_alphabet(
    arm: Arm, wanted: set[str], *, ordered: str
) -> tuple[Symbol, ...]:
    tokenizer = arm.tokenizer
    limit = min(int(arm.model.config.vocab_size), len(tokenizer))
    found: dict[str, list[int]] = {character: [] for character in ordered}
    for token_id in range(limit):
        piece = tokenizer.convert_ids_to_tokens(token_id)
        if piece is None:
            continue
        text = tokenizer.convert_tokens_to_string([piece])
        if len(text) == 1 and text in wanted:
            found[text].append(token_id)
    ambiguous = {character: ids for character, ids in found.items() if len(ids) != 1}
    if ambiguous:
        raise ValueError(
            f"{arm.name}: {sorted(ambiguous)} do not each resolve to exactly one "
            f"token ({ambiguous}); an embedding row of a symbol is not defined"
        )
    return tuple(Symbol(character, found[character][0]) for character in ordered)


def symbol_token_coverage(
    arm: Arm, texts: Sequence[str], *, alphabet: Sequence[Symbol], max_len: int
) -> dict[str, Any]:
    """D3.j-A0: what share of the scored alphabet characters own a token.

    Measured on the arm's own rendered cohort, truncated to the scored window,
    by decoding every token and asking whether it is exactly one alphabet
    character. A residue-tokenised arm returns 1.0 by construction; a
    multi-residue BPE arm returns a small number, and that number is the reported
    result for it rather than a negative about the model.
    """

    if not texts:
        raise ValueError("coverage cannot be measured on an empty cohort")
    characters = {symbol.label for symbol in alphabet}
    owned = {int(symbol.token_id) for symbol in alphabet}
    tokenizer = arm.tokenizer
    total = 0
    carried = 0
    tokens = 0
    for text in texts:
        ids = tokenizer(text, return_tensors=None)["input_ids"][:max_len]
        tokens += len(ids)
        for token_id in ids:
            piece = tokenizer.convert_ids_to_tokens(int(token_id))
            if piece is None:
                continue
            decoded = tokenizer.convert_tokens_to_string([piece])
            present = sum(1 for character in decoded if character in characters)
            total += present
            if int(token_id) in owned:
                carried += present
    if total == 0:
        raise ValueError(
            f"{arm.name}: the rendered cohort carries no alphabet character inside "
            f"the {max_len}-token window, so coverage is undefined"
        )
    return {
        "coverage": carried / total,
        "alphabet_characters_in_window": total,
        "alphabet_characters_in_single_symbol_tokens": carried,
        "tokens_examined": tokens,
        "max_len": int(max_len),
        "n_records": len(texts),
        "declared_tokenisation": arm.spec.tokenisation,
        "definition": (
            "share of the alphabet characters inside the scored window that are "
            "carried by a token whose whole decoded form is one alphabet character"
        ),
    }


def admit_arm(coverage: Mapping[str, Any], arm_name: str, *, minimum: float) -> dict[str, Any]:
    """D3.j-A0's verdict, returned rather than raised for a failing arm.

    A refusal here is a *measurement about the arm* -- its coverage is the
    reported result -- so it belongs in the artefact beside the number that
    decided it, not in a traceback.
    """

    measured = float(coverage["coverage"])
    admitted = measured >= minimum
    return {
        "admitted": bool(admitted),
        "measured_coverage": measured,
        "minimum_coverage": float(minimum),
        "arm": arm_name,
        "reason": (
            "at least the declared share of scored alphabet characters own a token, "
            "so replacing a symbol's embedding row replaces what the model reads"
            if admitted
            else (
                f"{measured:.4f} of the scored alphabet characters are carried by "
                "single-symbol tokens, below the declared "
                f"{minimum:.2f}. The substitution is undefined on this arm rather "
                "than noisy: the rows that do exist would measure the positions "
                "where a symbol happened to be tokenised alone, a cohort selected "
                "by local context rather than at random (L31), in a unit that is "
                "not the other arms' unit (L23). NOT MEASURABLE; this coverage is "
                "the reported result for this arm"
            )
        ),
    }


# --------------------------------------------------- chemical axis (biology side)


@dataclass(frozen=True)
class PropertyAxis:
    """A distance matrix over an alphabet, and the declaration it came from."""

    distance: np.ndarray
    properties_used: tuple[str, ...]
    properties_dropped: tuple[str, ...]
    source: str

    def record(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "properties_used": list(self.properties_used),
            "properties_dropped": list(self.properties_dropped),
            "dropped_reason": (
                "constant over the admitted alphabet, so it contributes no distance "
                "and its inclusion would only rescale every pair identically"
            ),
            "definition": (
                "each descriptor z-scored across the alphabet, then the Euclidean "
                "distance between symbols divided by the square root of the number "
                "of descriptors used"
            ),
        }


def chemical_property_table(residues: Sequence[str]) -> dict[str, list[float]]:
    """D3.j-A2's declared descriptor set, imported rather than restated.

    Three of the four come from :data:`src.transfer.concept_lens.PROPERTY_BASIS`,
    which was frozen for the concept lens before any run of that stage;
    re-declaring hydropathy here would give one table two sources (Appendix B
    rule 12). The fourth, polarity, is declared in this module with its own
    citation because no other module needed it.
    """

    table = {
        name: [float(values[residue]) for residue in residues]
        for name, values in PROPERTY_BASIS.items()
    }
    table["polarity"] = [float(GRANTHAM_POLARITY[residue]) for residue in residues]
    return table


def property_distance(table: Mapping[str, Sequence[float]], *, source: str) -> PropertyAxis:
    """Chemical distance from a declared descriptor table, and nothing else."""

    names = tuple(sorted(table))
    rows = np.asarray([list(table[name]) for name in names], dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] < 2:
        raise ValueError("a descriptor table needs at least two symbols")
    if not np.isfinite(rows).all():
        raise ValueError("the descriptor table carries a non-finite value")
    keep = [index for index in range(rows.shape[0]) if float(rows[index].std()) > 0.0]
    if not keep:
        raise ValueError("every declared descriptor is constant over this alphabet")
    kept = rows[keep]
    standardised = (kept - kept.mean(axis=1, keepdims=True)) / kept.std(axis=1, keepdims=True)
    difference = standardised[:, :, None] - standardised[:, None, :]
    return PropertyAxis(
        distance=np.sqrt((difference**2).sum(axis=0) / standardised.shape[0]),
        properties_used=tuple(names[index] for index in keep),
        properties_dropped=tuple(
            name for index, name in enumerate(names) if index not in keep
        ),
        source=source,
    )


def blosum62_distance(residues: Sequence[str]) -> np.ndarray:
    """BLOSUM62 as a distance, in the requested order. A CEILING-side estimator.

    ``d(x, y) = (S(x, x) + S(y, y)) / 2 - S(x, y)``, the standard conversion of a
    log-odds similarity into a non-negative dissimilarity with a zero diagonal.
    See :data:`BLOSUM62_SIDE_NOTE` for why it is not the chemical axis.
    """

    index = {residue: position for position, residue in enumerate(BLOSUM62_ORDER)}
    missing = sorted(set(residues) - set(index))
    if missing:
        raise ValueError(f"BLOSUM62 does not score {missing}")
    scores = np.asarray(BLOSUM62_ROWS, dtype=np.float64)
    if scores.shape != (20, 20) or not np.allclose(scores, scores.T):
        raise ValueError("the vendored BLOSUM62 is not a symmetric 20x20 matrix")
    selected = scores[np.ix_([index[r] for r in residues], [index[r] for r in residues])]
    diagonal = np.diag(selected)
    distance = 0.5 * (diagonal[:, None] + diagonal[None, :]) - selected
    if (distance < 0).any():
        raise ValueError("the BLOSUM62 distance conversion produced a negative value")
    return distance


# --------------------------------------------- distributional axis (ceiling side)


def residue_context_counts(
    background: KmerBackground, residues: Sequence[str]
) -> np.ndarray:
    """``(left, right)`` neighbour-pair counts of each residue, from the pinned k=3.

    The 3-mer vector is indexed base-20 over
    :data:`src.transfer.kmer_background.ALPHABET`, most significant symbol first,
    so reshaping to ``(a, s, b)`` and reading the centre index gives each
    residue's joint context distribution exactly. ``load`` has already refused the
    file if its digest moved.
    """

    if KMER_ALPHABET != AA20:
        raise ValueError(
            f"the k-mer background indexes {KMER_ALPHABET} and the panel declares "
            f"{AA20}; the reshape below would silently permute the alphabet"
        )
    if 3 not in background.counts:
        raise ValueError("the background carries no k = 3 vector")
    counts = background.counts[3].reshape(20, 20, 20).astype(np.float64)
    rows = [AA20.index(residue) for residue in residues]
    return counts.transpose(1, 0, 2).reshape(20, 400)[rows]


def residue_context_profiles_at_order(ordered: OrderedFragmentCounts) -> np.ndarray:
    """Each residue's context profile at an odd order, as one distribution.

    At order ``k`` the context is the ``(k-1)/2`` residues on each side, so the
    profile has ``20 ** (k - 1)`` cells and the k = 3 case is exactly
    :func:`residue_context_counts`'s joint ``(left, right)`` pair. Even orders have
    no symmetric split and are refused rather than given an arbitrary one.
    """

    order = ordered.order
    if order % 2 == 0 or order < 3:
        raise ValueError(
            f"a symmetric context profile is defined at odd orders from 3; got {order}"
        )
    side = len(AA20) ** ((order - 1) // 2)
    counts = ordered.counts
    profiles = np.empty((len(AA20), side * side), dtype=np.float64)
    for row in range(len(AA20)):
        if isinstance(counts, np.memmap):
            block = np.empty((side, side), dtype=np.float64)
            for left in range(side):
                base = (left * len(AA20) + row) * side
                block[left] = np.asarray(counts[base : base + side])
            profiles[row] = block.reshape(-1)
        else:
            profiles[row] = np.asarray(counts).reshape(side, len(AA20), side)[:, row, :].reshape(-1)
    totals = profiles.sum(axis=1)
    if (totals <= 0).any():
        raise ValueError("a residue has no observed context at this order")
    return profiles / totals[:, None]


def token_context_counts(
    sequences: Sequence[Sequence[int]],
    symbols: Sequence[Symbol],
    *,
    bucket_of_token: np.ndarray,
    n_buckets: int,
) -> np.ndarray:
    """The same joint ``(left, right)`` context counts over a tokenised corpus.

    ``bucket_of_token`` coarsens a neighbour onto a declared context alphabet.
    For the byte-level text control a neighbour is already one character, and the
    coarsening declared by :func:`lowercase_letter_buckets` folds case away so
    that 27 buckets rather than several hundred have to be estimated.
    """

    row_of = {symbol.token_id: index for index, symbol in enumerate(symbols)}
    counts = np.zeros((len(symbols), n_buckets * n_buckets), dtype=np.float64)
    for tokens in sequences:
        for position in range(1, len(tokens) - 1):
            row = row_of.get(int(tokens[position]))
            if row is None:
                continue
            left = int(bucket_of_token[int(tokens[position - 1])])
            right = int(bucket_of_token[int(tokens[position + 1])])
            counts[row, left * n_buckets + right] += 1.0
    return counts


def lowercase_letter_buckets(arm: Arm) -> tuple[np.ndarray, int, str]:
    """Coarsen every token onto the lowercased letter it starts with, or 'other'."""

    tokenizer = arm.tokenizer
    size = int(arm.model.config.vocab_size)
    limit = min(size, len(tokenizer))
    buckets = np.full(size, 26, dtype=np.int64)
    for token_id in range(limit):
        piece = tokenizer.convert_ids_to_tokens(token_id)
        if piece is None:
            continue
        text = tokenizer.convert_tokens_to_string([piece]).strip().lower()
        if text and "a" <= text[0] <= "z":
            buckets[token_id] = ord(text[0]) - ord("a")
    return buckets, 27, (
        "each neighbour token is coarsened onto the lowercased ASCII letter its "
        "decoded form starts with, with one further bucket for everything else"
    )


def context_profiles(counts: np.ndarray) -> np.ndarray:
    """Normalise each symbol's context counts to a distribution."""

    if counts.ndim != 2:
        raise ValueError("context counts must be one row per symbol")
    totals = counts.sum(axis=1)
    if (totals <= 0).any():
        empty = sorted(np.flatnonzero(totals <= 0).tolist())
        raise ValueError(
            f"symbols at rows {empty} were never observed with both neighbours, so "
            "they have no context profile; enlarge the background or drop them from "
            "the alphabet as a declared decision"
        )
    return counts / totals[:, None]


def cosine_distance(profiles: np.ndarray) -> np.ndarray:
    """One minus the cosine between context profiles: D3.j-A2's declared metric."""

    norms = np.linalg.norm(profiles, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError("a context profile has zero norm")
    unit = profiles / norms
    distance = 1.0 - unit @ unit.T
    np.fill_diagonal(distance, 0.0)
    return np.maximum(distance, 0.0)


def symmetric_kl_distance(profiles: np.ndarray) -> np.ndarray | None:
    """The declared alternative metric, or ``None`` where it is not defined.

    Symmetric KL is infinite wherever one profile puts mass on a cell the other
    leaves empty, which on the staged background never happens -- all 8,000
    3-mers are observed -- and on a text background of any affordable size always
    does. Returning ``None`` says so; substituting a smoothing constant would make
    the alternative metric a function of an undeclared choice.
    """

    if (profiles <= 0).any():
        return None
    logs = np.log(profiles)
    size = profiles.shape[0]
    distance = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        difference = profiles[i] - profiles
        log_difference = logs[i] - logs
        distance[i] = (difference * log_difference).sum(axis=1)
    np.fill_diagonal(distance, 0.0)
    return distance


# --------------------------------------------------- quadrants and the cut sweep


@dataclass(frozen=True)
class PairSet:
    """Ordered substitution pairs as index pairs into one alphabet, with classes.

    Ordered, because the intervention is: ``(r, s)`` writes ``s``'s row over
    ``r``'s and reads the positions that follow an ``r``, while ``(s, r)`` does
    the opposite on different positions. Both similarity axes are symmetric, so
    admission is decided on the 190 **unordered** pairs and both directions of an
    admitted pair inherit its quadrant; the two directions are then separate
    observations in separate resampling groups.
    """

    pairs: tuple[tuple[int, int], ...]
    classes: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.pairs) != len(self.classes):
            raise ValueError("every pair needs a class label")
        if any(x == y for x, y in self.pairs):
            raise ValueError("a substitution pair cannot name one symbol twice")

    def __len__(self) -> int:
        return len(self.pairs)

    @property
    def groups(self) -> np.ndarray:
        """The resampling unit: the substituted symbol."""

        return np.asarray([x for x, _ in self.pairs], dtype=np.int64)

    def codes(self, order: Sequence[str]) -> np.ndarray:
        """``-1`` for ``order[0]`` and ``+1`` for ``order[1]``, so that the
        contrast is ``mean over order[1] - mean over order[0]``."""

        if len(order) != 2:
            raise ValueError("a contrast has exactly two classes")
        lookup = {order[0]: -1, order[1]: 1}
        unknown = sorted(set(self.classes) - set(lookup))
        if unknown:
            raise ValueError(f"{unknown} are not classes of this contrast")
        return np.asarray([lookup[klass] for klass in self.classes], dtype=np.int64)

    def labelled(self, symbols: Sequence[Symbol]) -> list[dict[str, Any]]:
        return [
            {"substituted": symbols[x].label, "substitute": symbols[y].label, "class": klass}
            for (x, y), klass in zip(self.pairs, self.classes)
        ]


def _unordered(size: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(size, 1)


def _rank_bands(values: np.ndarray, quantile: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """The lower and upper quantile bands, taken on **ranks**.

    D3.j-A2 admits a pair where its two similarity *ranks* fall in opposite
    bands, and the distinction from a value threshold is not cosmetic. A cosine
    distance over sparse context profiles ties at 1.0 for every pair with
    disjoint support, and a strict value comparison against a quantile that lands
    inside such a tie block excludes the **whole** block -- an empty band, which
    reads exactly like an alphabet with no contradiction set. Average ranks assign
    a straddling block by its centre of mass instead, so it enters the band it
    mostly lies in.

    That is an improvement and not a guarantee: a tie block whose centre is on the
    wrong side of the threshold is still excluded entire, and it should be. The
    tie count is therefore returned with the bands, so a degenerate axis is
    visible in the artefact rather than reaching the reader as a quadrant that
    happened to come out small.
    """

    ranks = stats.rankdata(values)
    size = values.size
    low = ranks <= quantile * size
    high = ranks > (1.0 - quantile) * size
    return low, high, {
        "n_values": int(size),
        "n_distinct_values": int(np.unique(values).size),
        "n_tied_values": int(size - np.unique(values).size),
        "low_band_size": int(low.sum()),
        "high_band_size": int(high.sum()),
    }


def quadrants_at_cut(
    chemical: np.ndarray, distributional: np.ndarray, *, cut: str
) -> dict[str, Any]:
    """D3.j-A2's two quadrants at one rung of the sweep, over unordered pairs."""

    if cut not in CUTS:
        raise ValueError(f"unknown cut {cut!r}; declared: {sorted(CUTS)}")
    if chemical.shape != distributional.shape or chemical.ndim != 2:
        raise ValueError("the two axes must be square matrices over one alphabet")
    quantile = CUTS[cut]
    rows, columns = _unordered(chemical.shape[0])
    chem, dist = chemical[rows, columns], distributional[rows, columns]
    chem_low_band, chem_high_band, chem_ties = _rank_bands(chem, quantile)
    dist_low_band, dist_high_band, dist_ties = _rank_bands(dist, quantile)
    similar_dissimilar = chem_low_band & dist_high_band
    dissimilar_similar = chem_high_band & dist_low_band
    members = {
        QUADRANTS[0]: [
            (int(rows[index]), int(columns[index]))
            for index in np.flatnonzero(similar_dissimilar)
        ],
        QUADRANTS[1]: [
            (int(rows[index]), int(columns[index]))
            for index in np.flatnonzero(dissimilar_similar)
        ],
    }
    counts = {name: len(pairs) for name, pairs in members.items()}
    return {
        "cut": cut,
        "quantile": quantile,
        "chemical_band_values": [
            float(np.quantile(chem, quantile)), float(np.quantile(chem, 1.0 - quantile))
        ],
        "distributional_band_values": [
            float(np.quantile(dist, quantile)), float(np.quantile(dist, 1.0 - quantile))
        ],
        "chemical_band": chem_ties,
        "distributional_band": dist_ties,
        "unordered_pairs_scored": int(chem.size),
        "unordered_counts": counts,
        "minimum_required": int(MINIMUM_QUADRANT_PAIRS),
        "readable": bool(min(counts.values()) >= MINIMUM_QUADRANT_PAIRS),
        "members": members,
    }


def cut_sweep(chemical: np.ndarray, distributional: np.ndarray) -> dict[str, Any]:
    """D3.j-A3: every rung of the sweep, with its admitted count per quadrant.

    Falling short of eight per quadrant is a live outcome and a statement about
    the alphabet and the corpus rather than about any model, so it is reported as
    a measurement here and not raised.
    """

    rows, columns = _unordered(chemical.shape[0])
    correlation = stats.spearmanr(chemical[rows, columns], distributional[rows, columns])
    sweep = {cut: quadrants_at_cut(chemical, distributional, cut=cut) for cut in CUTS}
    return {
        "axis_spearman": float(correlation.statistic),
        "axis_spearman_p": float(correlation.pvalue),
        "n_unordered_pairs": int(rows.size),
        "per_cut": {
            cut: {key: value for key, value in record.items() if key != "members"}
            for cut, record in sweep.items()
        },
        "readable_cuts": [cut for cut, record in sweep.items() if record["readable"]],
        "rule": (
            "a pair is admitted where one similarity is below its band's lower "
            "quantile and the other above its upper quantile. At least "
            f"{MINIMUM_QUADRANT_PAIRS} unordered pairs are required in each quadrant "
            "at the declared cut (D3.j-A3); the sweep is reported at every rung so "
            "that the ordering is shown to survive it, or the dependence is itself "
            "the finding (rule 17)"
        ),
    }


def ordered_pair_set(quadrant_record: Mapping[str, Any]) -> PairSet:
    """Both directions of every admitted unordered pair."""

    pairs: list[tuple[int, int]] = []
    classes: list[str] = []
    for name in QUADRANTS:
        for x, y in quadrant_record["members"][name]:
            pairs.extend([(x, y), (y, x)])
            classes.extend([name, name])
    if not pairs:
        raise ValueError(f"the {quadrant_record['cut']} cut admits no pair at all")
    return PairSet(tuple(pairs), tuple(classes))


def symmetrize_directional_damage(forward: float, reverse: float) -> float:
    """Unordered axis value: the arithmetic mean of the two directed damages."""

    return 0.5 * (float(forward) + float(reverse))


def fragment_damage_axis(
    directional: Mapping[tuple[int, int], Mapping[str, Any]],
    *,
    size: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Build the unordered fragment-damage axis, refusing any missing direction.

    A pair enters only when both directed substitutions are measurable on the
    declared scored records. Nothing is smoothed or imputed.
    """

    distance = np.full((size, size), np.nan, dtype=np.float64)
    observed = np.zeros((size, size), dtype=bool)
    refusals: list[dict[str, Any]] = []
    rows, columns = _unordered(size)
    for x, y in zip(rows.tolist(), columns.tolist()):
        left = directional.get((int(x), int(y)))
        right = directional.get((int(y), int(x)))
        if left is None or right is None:
            refusals.append({
                "pair": (int(x), int(y)),
                "reason": "a directed substitution was never scored",
            })
            continue
        if not left.get("measurable") or not right.get("measurable"):
            refusals.append({
                "pair": (int(x), int(y)),
                "forward_measurable": bool(left.get("measurable")),
                "reverse_measurable": bool(right.get("measurable")),
                "forward_reason": left.get("unmeasurable_reason"),
                "reverse_reason": right.get("unmeasurable_reason"),
                "reason": "insufficient fragment coverage",
            })
            continue
        value = symmetrize_directional_damage(
            float(left["nats_per_scored_token"]),
            float(right["nats_per_scored_token"]),
        )
        distance[x, y] = distance[y, x] = value
        observed[x, y] = observed[y, x] = True
    np.fill_diagonal(distance, 0.0)
    np.fill_diagonal(observed, True)
    return distance, observed, refusals


def _observed_unordered(
    chemical: np.ndarray, distributional: np.ndarray, observed: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, columns = _unordered(chemical.shape[0])
    keep = observed[rows, columns]
    return rows[keep], columns[keep], chemical[rows, columns][keep], distributional[rows, columns][keep]


def quadrants_at_cut_observed(
    chemical: np.ndarray,
    distributional: np.ndarray,
    observed: np.ndarray,
    *,
    cut: str,
) -> dict[str, Any]:
    """D3.j-B quadrants: ranks are taken only over covered unordered pairs."""

    if cut not in CUTS:
        raise ValueError(f"unknown cut {cut!r}; declared: {sorted(CUTS)}")
    if chemical.shape != distributional.shape or chemical.ndim != 2:
        raise ValueError("the two axes must be square matrices over one alphabet")
    rows, columns, chem, dist = _observed_unordered(chemical, distributional, observed)
    if chem.size == 0:
        empty = {
            QUADRANTS[0]: [],
            QUADRANTS[1]: [],
        }
        return {
            "cut": cut,
            "quantile": CUTS[cut],
            "unordered_pairs_scored": 0,
            "unordered_counts": {name: 0 for name in QUADRANTS},
            "minimum_required": int(MINIMUM_QUADRANT_PAIRS),
            "readable": False,
            "members": empty,
            "chemical_band": {"n_values": 0},
            "distributional_band": {"n_values": 0},
        }
    quantile = CUTS[cut]
    chem_low_band, chem_high_band, chem_ties = _rank_bands(chem, quantile)
    dist_low_band, dist_high_band, dist_ties = _rank_bands(dist, quantile)
    similar_dissimilar = chem_low_band & dist_high_band
    dissimilar_similar = chem_high_band & dist_low_band
    members = {
        QUADRANTS[0]: [
            (int(rows[index]), int(columns[index]))
            for index in np.flatnonzero(similar_dissimilar)
        ],
        QUADRANTS[1]: [
            (int(rows[index]), int(columns[index]))
            for index in np.flatnonzero(dissimilar_similar)
        ],
    }
    counts = {name: len(pairs) for name, pairs in members.items()}
    return {
        "cut": cut,
        "quantile": quantile,
        "chemical_band_values": [
            float(np.quantile(chem, quantile)), float(np.quantile(chem, 1.0 - quantile))
        ],
        "distributional_band_values": [
            float(np.quantile(dist, quantile)), float(np.quantile(dist, 1.0 - quantile))
        ],
        "chemical_band": chem_ties,
        "distributional_band": dist_ties,
        "unordered_pairs_scored": int(chem.size),
        "unordered_counts": counts,
        "minimum_required": int(MINIMUM_QUADRANT_PAIRS),
        "readable": bool(min(counts.values()) >= MINIMUM_QUADRANT_PAIRS),
        "members": members,
    }


def cut_sweep_observed(
    chemical: np.ndarray, distributional: np.ndarray, observed: np.ndarray
) -> dict[str, Any]:
    """The D3.j-A3 sweep, restricted to pairs the fragment axis actually covers."""

    rows, columns, chem, dist = _observed_unordered(chemical, distributional, observed)
    correlation = (
        stats.spearmanr(chem, dist) if chem.size >= 2 and np.unique(chem).size > 1 and np.unique(dist).size > 1
        else None
    )
    sweep = {
        cut: quadrants_at_cut_observed(chemical, distributional, observed, cut=cut)
        for cut in CUTS
    }
    return {
        "axis_spearman": None if correlation is None else float(correlation.statistic),
        "axis_spearman_p": None if correlation is None else float(correlation.pvalue),
        "n_unordered_pairs": int(rows.size),
        "per_cut": {
            cut: {key: value for key, value in record.items() if key != "members"}
            for cut, record in sweep.items()
        },
        "readable_cuts": [cut for cut, record in sweep.items() if record["readable"]],
        "rule": (
            "a pair is admitted where one similarity is below its band's lower "
            "quantile and the other above its upper quantile, ranked only over "
            f"pairs with both directed fragment damages measurable. At least "
            f"{MINIMUM_QUADRANT_PAIRS} unordered pairs are required in each quadrant "
            "at the declared cut"
        ),
    }


def agreement_extremes_observed(
    chemical: np.ndarray,
    distributional: np.ndarray,
    observed: np.ndarray,
    *,
    cut: str,
    count: int,
) -> tuple[PairSet, dict[str, Any]]:
    """Reachability ends, ranked only over covered unordered pairs."""

    if count < 1:
        raise ValueError("the reachability check needs at least one pair per end")
    if cut not in CUTS:
        raise ValueError(f"unknown cut {cut!r}; declared: {sorted(CUTS)}")
    quantile = CUTS[cut]
    rows, columns, chem, dist = _observed_unordered(chemical, distributional, observed)
    if chem.size == 0:
        raise ValueError("no covered unordered pair remains for the agreement set")
    chem_low_band, chem_high_band, _ = _rank_bands(chem, quantile)
    dist_low_band, dist_high_band, _ = _rank_bands(dist, quantile)
    chem_rank = stats.rankdata(chem)
    dist_rank = stats.rankdata(dist)
    both_far = np.flatnonzero(chem_high_band & dist_high_band)
    both_near = np.flatnonzero(chem_low_band & dist_low_band)
    if both_far.size < count or both_near.size < count:
        raise ValueError(
            f"the {cut} agreement set holds {both_far.size} dissimilar and "
            f"{both_near.size} similar covered pairs, fewer than the {count} per end asked for"
        )
    far = sorted(both_far, key=lambda i: (-(chem_rank[i] + dist_rank[i]), i))[:count]
    near = sorted(both_near, key=lambda i: (chem_rank[i] + dist_rank[i], i))[:count]
    pairs = [(int(rows[i]), int(columns[i])) for i in far] + [
        (int(rows[i]), int(columns[i])) for i in near
    ]
    classes = [AGREEMENT_CLASSES[0]] * len(far) + [AGREEMENT_CLASSES[1]] * len(near)
    return PairSet(tuple(pairs), tuple(classes)), {
        "cut": cut,
        "pairs_per_end": int(count),
        "n_dissimilar_available": int(both_far.size),
        "n_similar_available": int(both_near.size),
        "ranking": "by the sum of the two axes' ranks over covered pairs",
    }


def matching_ceiling_predicts_distributional_side(
    codes: np.ndarray, ceiling_damage: Sequence[float]
) -> dict[str, Any]:
    """The matching ceiling must claim the distributional side of Delta.

    Admission places low fragment damage in the distributionally-similar
    quadrant, so the ceiling's own Delta is negative by construction. A
    non-negative value is a specification defect, not a model result.
    """

    delta = _quadrant_delta(codes, np.asarray(ceiling_damage, dtype=np.float64))
    if not np.isfinite(delta):
        return {
            "status": "VOID",
            "reason": CEILING_CONSTRUCTION_VOID,
            "detail": "the matching ceiling Delta is undefined on this pair set",
            "ceiling_delta": None,
        }
    if delta >= 0.0:
        return {
            "status": "VOID",
            "reason": CEILING_CONSTRUCTION_VOID,
            "detail": (
                "the matching fragment ceiling does not predict the distributional "
                f"side: Delta = {delta:.6g} is not strictly negative"
            ),
            "ceiling_delta": float(delta),
        }
    return {
        "status": "OK",
        "reason": None,
        "detail": (
            "the matching ceiling Delta is negative, so the admitted pair set "
            "has the predeclared opposite ordering"
        ),
        "ceiling_delta": float(delta),
    }


TEXT_BANDS = ("distributionally_similar", "distributionally_dissimilar")


def text_control_pair_set(
    distributional: np.ndarray, *, cut: str
) -> tuple[PairSet, dict[str, Any]]:
    """D3.j-A1's one-axis contrast: text has no chemistry, so only one axis exists.

    The same quantile bands as the protein cell, applied to the distributional
    axis alone. The control asks whether the readout can detect substitute
    similarity at all, which is the only half of the design text can validate.
    """

    if cut not in CUTS:
        raise ValueError(f"unknown cut {cut!r}; declared: {sorted(CUTS)}")
    quantile = CUTS[cut]
    rows, columns = _unordered(distributional.shape[0])
    values = distributional[rows, columns]
    low_band, high_band, ties = _rank_bands(values, quantile)
    members = {
        TEXT_BANDS[0]: [
            (int(rows[i]), int(columns[i])) for i in np.flatnonzero(low_band)
        ],
        TEXT_BANDS[1]: [
            (int(rows[i]), int(columns[i])) for i in np.flatnonzero(high_band)
        ],
    }
    pairs: list[tuple[int, int]] = []
    classes: list[str] = []
    for name in TEXT_BANDS:
        for x, y in members[name]:
            pairs.extend([(x, y), (y, x)])
            classes.extend([name, name])
    counts = {name: len(value) for name, value in members.items()}
    return PairSet(tuple(pairs), tuple(classes)), {
        "cut": cut,
        "quantile": quantile,
        "band_values": [
            float(np.quantile(values, quantile)), float(np.quantile(values, 1.0 - quantile))
        ],
        "band": ties,
        "unordered_counts": counts,
        "minimum_required": int(MINIMUM_QUADRANT_PAIRS),
        "readable": bool(min(counts.values()) >= MINIMUM_QUADRANT_PAIRS),
        "n_unordered_pairs": int(values.size),
    }


def agreement_extremes(
    chemical: np.ndarray, distributional: np.ndarray, *, cut: str, count: int
) -> tuple[PairSet, dict[str, Any]]:
    """The agreement set's two ends, for the reachability check (rule 40).

    An agreement pair is one on which the two axes concur -- exactly the
    population every earlier campaign measured on, and exactly the population that
    cannot discriminate the two hypotheses. It is the right population for a
    reachability check for the same reason: whatever the model computes on, a
    substitution that is both chemically and distributionally dissimilar should
    hurt more than one that is similar on both, and an instrument that cannot see
    that cannot be read on the hard case.
    """

    if count < 1:
        raise ValueError("the reachability check needs at least one pair per end")
    if cut not in CUTS:
        raise ValueError(f"unknown cut {cut!r}; declared: {sorted(CUTS)}")
    quantile = CUTS[cut]
    rows, columns = _unordered(chemical.shape[0])
    chem, dist = chemical[rows, columns], distributional[rows, columns]
    chem_low_band, chem_high_band, _ = _rank_bands(chem, quantile)
    dist_low_band, dist_high_band, _ = _rank_bands(dist, quantile)
    chem_rank = stats.rankdata(chem)
    dist_rank = stats.rankdata(dist)
    both_far = np.flatnonzero(chem_high_band & dist_high_band)
    both_near = np.flatnonzero(chem_low_band & dist_low_band)
    if both_far.size < count or both_near.size < count:
        raise ValueError(
            f"the {cut} agreement set holds {both_far.size} dissimilar and "
            f"{both_near.size} similar pairs, fewer than the {count} per end asked for"
        )
    far = sorted(both_far, key=lambda i: (-(chem_rank[i] + dist_rank[i]), i))[:count]
    near = sorted(both_near, key=lambda i: (chem_rank[i] + dist_rank[i], i))[:count]
    pairs = [(int(rows[i]), int(columns[i])) for i in far] + [
        (int(rows[i]), int(columns[i])) for i in near
    ]
    classes = [AGREEMENT_CLASSES[0]] * len(far) + [AGREEMENT_CLASSES[1]] * len(near)
    return PairSet(tuple(pairs), tuple(classes)), {
        "cut": cut,
        "pairs_per_end": int(count),
        "n_dissimilar_available": int(both_far.size),
        "n_similar_available": int(both_near.size),
        "ranking": "by the sum of the two axes' ranks, so neither end is entered on one axis alone",
    }


def cap_pairs(
    pairs: PairSet, *, strictness: Sequence[int], maximum: int, seed: int
) -> tuple[PairSet, dict[str, Any]]:
    """Subsample to a budget, spreading over class and substituted symbol.

    A uniform draw would concentrate on whichever substituted symbols carry the
    most pairs, and the substituted symbol is the resampling unit, so a
    concentrated draw buys pairs at the cost of the units the interval is taken
    over. The draw is round-robin over ``(class, substituted symbol)`` buckets,
    and inside a bucket the **strictest** pairs go first, so that a cap applied to
    the loosest rung of the sweep does not empty the strict rungs it contains.
    """

    if maximum < 1:
        raise ValueError("a pair budget below one measures nothing")
    if len(strictness) != len(pairs):
        raise ValueError("every pair needs a strictness rank")
    if len(pairs) <= maximum:
        return pairs, {
            "requested": int(maximum), "available": len(pairs),
            "selected": len(pairs), "subsampled": False, "seed": int(seed),
        }
    rng = np.random.default_rng(seed)
    keys = rng.random(len(pairs))
    buckets: dict[tuple[str, int], list[int]] = {}
    for index, ((x, _), klass) in enumerate(zip(pairs.pairs, pairs.classes)):
        buckets.setdefault((klass, x), []).append(index)
    for key in buckets:
        buckets[key].sort(key=lambda index: (strictness[index], keys[index]), reverse=True)
    order = sorted(buckets)
    selected: list[int] = []
    while len(selected) < maximum:
        drew = False
        for key in order:
            if not buckets[key]:
                continue
            selected.append(buckets[key].pop())
            drew = True
            if len(selected) >= maximum:
                break
        if not drew:
            break
    selected.sort()
    return PairSet(
        tuple(pairs.pairs[index] for index in selected),
        tuple(pairs.classes[index] for index in selected),
    ), {
        "requested": int(maximum), "available": len(pairs), "selected": len(selected),
        "subsampled": True, "seed": int(seed),
        "draw": (
            "round-robin over (class, substituted symbol) buckets under the declared "
            "seed, strictest sweep rung first inside a bucket"
        ),
    }


# --------------------------------------------------------------- intervention


def _walk(root: object, path: Sequence[str], *, what: str) -> Any:
    module = root
    for step in path:
        if not hasattr(module, step):
            raise TypeError(f"{what}: no {step!r} on the declared path {tuple(path)}")
        module = getattr(module, step)
    return module


class ArmAlphabetModel:
    """A panel arm exposed as an embedding matrix and a logit function.

    Constructing one **unties the output head** where the checkpoint ties it, and
    that is the whole reason this wrapper exists rather than two helper functions.
    Under a tie, overwriting an input embedding row also overwrites an unembedding
    row, and the measured quantity stops being "the model reads r as s" and
    becomes "the model can no longer emit r" -- a much larger effect that would be
    reported under this stage's name. D3.j's operation says the output head is
    untouched; this is where that is made true.
    """

    def __init__(self, arm: Arm) -> None:
        architecture = arm.spec.architecture
        if architecture not in INPUT_EMBEDDING_PATH:
            raise TypeError(
                f"{arm.name}: no input embedding is declared for {architecture!r}; "
                f"declared: {sorted(INPUT_EMBEDDING_PATH)}"
            )
        self.arm = arm
        self.embedding = _walk(
            arm.model, INPUT_EMBEDDING_PATH[architecture], what=f"{arm.name} embedding"
        )
        head = _walk(arm.model, OUTPUT_HEAD_PATH[architecture], what=f"{arm.name} head")
        self.was_tied = head.weight.data_ptr() == self.embedding.weight.data_ptr()
        if self.was_tied:
            head.weight = torch.nn.Parameter(head.weight.detach().clone(), requires_grad=False)
            if head.weight.data_ptr() == self.embedding.weight.data_ptr():
                raise RuntimeError(f"{arm.name}: the output head is still tied after cloning")
        self.head = head

    @property
    def weight(self) -> torch.Tensor:
        return self.embedding.weight

    def logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.arm.model(input_ids=input_ids, attention_mask=attention_mask).logits

    def record(self) -> dict[str, Any]:
        architecture = self.arm.spec.architecture
        return {
            "architecture": architecture,
            "embedding_path": ".".join(INPUT_EMBEDDING_PATH[architecture]),
            "output_head_path": ".".join(OUTPUT_HEAD_PATH[architecture]),
            "output_head_was_tied": bool(self.was_tied),
            "untie_reason": (
                "the intervention is on the INPUT embedding only. Where the "
                "checkpoint ties the unembedding to it, writing the substitute's row "
                "would also delete the substituted symbol's output class, which is a "
                "different and far larger intervention"
            ),
            "embedding_shape": list(self.weight.shape),
            "dtype": str(self.weight.dtype).removeprefix("torch."),
        }


@contextmanager
def substituted_row(weight: torch.Tensor, row: int, values: torch.Tensor) -> Iterator[None]:
    """Overwrite one embedding row for the duration of the block, then restore it."""

    if values.shape != weight[row].shape:
        raise ValueError(
            f"a substitute row of shape {tuple(values.shape)} cannot replace one of "
            f"shape {tuple(weight[row].shape)}"
        )
    original = weight.data[row].detach().clone()
    weight.data[row] = values.to(weight.dtype).to(weight.device)
    try:
        yield
    finally:
        weight.data[row] = original


@dataclass
class ScoringCohort:
    """A tokenised held-out cohort and the targets that belong to it."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    target_mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2 or self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("input ids and attention mask must share a 2-D shape")
        if self.target_mask.shape != (self.input_ids.shape[0], self.input_ids.shape[1] - 1):
            raise ValueError("the target mask must have one fewer column than the ids")
        if self.target_mask.dtype != torch.bool:
            raise ValueError("the target mask must be boolean")


def context_counts(cohort: ScoringCohort, token_ids: Sequence[int]) -> dict[int, int]:
    """How many scored targets each token is the immediately preceding input of."""

    preceding = cohort.input_ids[:, :-1].cpu()
    mask = cohort.target_mask.cpu()
    return {int(token): int(((preceding == int(token)) & mask).sum()) for token in token_ids}


def residue_runs_by_row(
    cohort: ScoringCohort, alphabet: Sequence[Symbol]
) -> list[list[str]]:
    """Contiguous alphabet runs inside each original scored sequence.

    A non-alphabet token ends a run, so a fragment never spans a marker or a
    forced FASTA wrap. Each outer entry is one cohort record.
    """

    label_of = {symbol.token_id: symbol.label for symbol in alphabet}
    by_row: list[list[str]] = []
    ids = cohort.input_ids.cpu()
    mask = cohort.attention_mask.cpu()
    for row in range(ids.shape[0]):
        runs: list[str] = []
        current: list[str] = []
        for position in range(ids.shape[1]):
            if not bool(mask[row, position]):
                break
            label = label_of.get(int(ids[row, position]))
            if label is None:
                if len(current) >= 3:
                    runs.append("".join(current))
                current = []
                continue
            current.append(label)
        if len(current) >= 3:
            runs.append("".join(current))
        by_row.append(runs)
    if not any(by_row):
        raise ValueError("the tokenised cohort yields no symbol run long enough to score")
    return by_row


class DamageScorer:
    """Held-out likelihood damage from one embedding-row substitution.

    The clean pass for a symbol is run over **exactly** the rows and in exactly
    the chunks its substituted passes use, and cached per symbol. One clean pass
    over the whole cohort would be cheaper and would break the identity
    invariant: a different batch composition is a different reduction order, so
    the two passes would differ in the last bits and "exactly zero" would become
    "about 1e-7" -- which is the tolerance a broken write hides in.
    """

    def __init__(self, model: Any, cohort: ScoringCohort, *, batch_size: int) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model = model
        self.cohort = cohort
        self.batch_size = int(batch_size)
        self._targets = cohort.input_ids[:, 1:].cpu()
        self._preceding = cohort.input_ids[:, :-1].cpu()
        self._scored = cohort.target_mask.cpu()
        self._clean_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        # Counted rather than estimated, because the campaign bound for this track
        # is computed from its own admitted counts and a bound derived from a
        # formula would be a bound derived from an assumption about how many rows
        # carry each symbol.
        self.n_damage_calls = 0
        self.n_forward_chunks = 0
        self.n_forward_tokens = 0

    def _context(self, token: int) -> torch.Tensor:
        return (self._preceding == int(token)) & self._scored

    def _forward(self, rows: torch.Tensor) -> torch.Tensor:
        width = self.cohort.input_ids.shape[1] - 1
        out = torch.empty((rows.numel(), width), dtype=torch.float32)
        with torch.no_grad():
            for start in range(0, rows.numel(), self.batch_size):
                index = rows[start : start + self.batch_size]
                ids = self.cohort.input_ids[index]
                self.n_forward_chunks += 1
                self.n_forward_tokens += int(ids.numel())
                logits = self.model.logits(ids, self.cohort.attention_mask[index])
                log_probabilities = torch.log_softmax(logits[:, :-1].float(), dim=-1)
                nll = -log_probabilities.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                out[start : start + index.numel()] = nll.float().cpu()
        return out

    def _clean(self, token: int) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._clean_cache.get(int(token))
        if cached is not None:
            return cached
        rows = torch.nonzero(self._context(token).any(dim=1), as_tuple=False).flatten()
        if rows.numel() == 0:
            raise ValueError(
                f"token {token} is the read symbol at no scored position of this "
                "cohort, so it has no damage to measure"
            )
        result = (rows, self._forward(rows))
        self._clean_cache[int(token)] = result
        return result

    def cost(self) -> dict[str, Any]:
        """What this cell actually spent, for the track's dispatch note."""

        return {
            "damage_calls": int(self.n_damage_calls),
            "forward_chunks": int(self.n_forward_chunks),
            "forward_tokens": int(self.n_forward_tokens),
            "batch_size": int(self.batch_size),
            "cached_clean_symbols": len(self._clean_cache),
            "note": (
                "one clean pass is cached per substituted symbol and reused by every "
                "substitute of that symbol; the count below therefore includes the "
                "clean passes and is the whole forward cost of the cell"
            ),
        }

    def damage(
        self, substituted: int, substitute: int, *, values: torch.Tensor | None = None
    ) -> dict[str, Any]:
        """Mean nats added per scored token by reading ``substituted`` as ``substitute``.

        ``measurable`` is false, with the counts beside it, when the pair's own
        two identities leave no scored target. That is a property of the cohort
        rather than a defect, so it is reported and the pair is dropped by the
        caller with its reason, not raised in the middle of a campaign.
        """

        self.n_damage_calls += 1
        rows, clean = self._clean(substituted)
        context = self._context(substituted)[rows]
        targets = self._targets[rows]
        keep = context & (targets != int(substituted)) & (targets != int(substitute))
        scored = int(keep.sum())
        record = {
            "n_scored_tokens": scored,
            "n_context_positions": int(context.sum()),
            "excluded_by_pair_identity": int(context.sum()) - scored,
            "n_sequences": int(rows.numel()),
            "measurable": scored > 0,
        }
        n_records = int(self.cohort.input_ids.shape[0])
        per_sum = np.zeros(n_records, dtype=np.float64)
        per_count = np.zeros(n_records, dtype=np.int64)
        if scored == 0:
            record["nats_per_scored_token"] = None
            record["unmeasurable_reason"] = (
                "every target following this symbol is one of the pair's own two "
                "identities, so no position is unaffected by the substitution's label"
            )
            record["per_record_nll_sum"] = per_sum
            record["per_record_n_scored"] = per_count
            return record
        source = (
            self.model.weight.data[substitute].detach().clone() if values is None else values
        )
        with substituted_row(self.model.weight, substituted, source):
            dirty = self._forward(rows)
        delta = (dirty - clean).to(torch.float64)
        record["nats_per_scored_token"] = float(delta[keep].mean())
        row_ids = rows.cpu().numpy()
        for index, sequence in enumerate(row_ids):
            mask = keep[index]
            count = int(mask.sum())
            if count:
                per_sum[int(sequence)] = float(delta[index][mask].sum())
                per_count[int(sequence)] = count
        record["per_record_nll_sum"] = per_sum
        record["per_record_n_scored"] = per_count
        return record


# ---------------------------------------------------- the recombination ceiling


@dataclass(frozen=True)
class OrderedFragmentCounts:
    """One order's k-mer counts, with what says which corpus produced them."""

    order: int
    counts: np.ndarray
    source: str
    sha256: str
    observed: int
    possible: int
    total_kmers: int

    def record(self) -> dict[str, Any]:
        return {
            "order": int(self.order),
            "source": self.source,
            "sha256": self.sha256,
            "observed_kmers": int(self.observed),
            "possible_kmers": int(self.possible),
            "coverage": self.observed / self.possible,
            "total_windows": int(self.total_kmers),
            "mean_observations_per_kmer": self.total_kmers / self.possible,
        }


def load_ordered_counts(
    directory: Path, orders: Sequence[int], *, pinned: Path
) -> dict[int, OrderedFragmentCounts]:
    """Read the requested orders, digest-checked, and prove they extend the pin.

    The higher-order directory is admissible as a ceiling only because it is a
    strict extension of the background the pre-registration names: its k = 3 and
    k = 4 vectors are byte-identical to ``uniref50/``'s. That is checked here by
    comparing the two manifests' digests rather than assumed from the README, so a
    curve cannot be computed against a second opinion about the corpus while
    reporting the pinned rung as its own k = 3.

    Orders whose vector exceeds :data:`_MEMMAP_BYTES` are memory-mapped. Nothing
    in this module needs a full pass over one: a context's total is the sum of
    twenty *contiguous* cells, so both the conditional and its denominator are
    gathers.
    """

    import json

    directory, pinned = Path(directory), Path(pinned)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    pinned_manifest = json.loads((pinned / "manifest.json").read_text(encoding="utf-8"))
    for name, payload in (("higher-order", manifest), ("pinned", pinned_manifest)):
        if payload.get("schema_version") != "kmer_background_v1":
            raise ValueError(f"the {name} background is not kmer_background_v1")
        if payload.get("alphabet") != AA20:
            raise ValueError(f"the {name} background indexes {payload.get('alphabet')!r}")
    shared = sorted(set(manifest["sha256"]) & set(pinned_manifest["sha256"]))
    if not shared:
        raise ValueError("the two backgrounds share no order, so the extension is unprovable")
    disagreeing = [k for k in shared if manifest["sha256"][k] != pinned_manifest["sha256"][k]]
    if disagreeing:
        raise ValueError(
            f"orders {disagreeing} differ between {directory} and the pinned {pinned}; "
            "the higher-order vectors are a second opinion about the corpus rather "
            "than an extension of the background this design is pre-registered on"
        )
    loaded: dict[int, OrderedFragmentCounts] = {}
    for order in sorted({int(value) for value in orders}):
        if order not in {int(value) for value in manifest["k"]}:
            raise ValueError(f"the background carries no k = {order} vector")
        path = directory / f"kmer_counts_k{order}.npy"
        digest = sha256_file(path)
        if digest != manifest["sha256"][str(order)]:
            raise RuntimeError(f"{path} hashes {digest}, manifest says {manifest['sha256'][str(order)]}")
        counts = np.load(path, mmap_mode=None if _fits_in_memory(path.stat().st_size) else "r")
        if counts.shape != (len(AA20) ** order,):
            raise ValueError(f"k = {order} has shape {counts.shape}, not {(len(AA20) ** order,)}")
        entry = manifest["counts"][str(order)]
        loaded[order] = OrderedFragmentCounts(
            order=order,
            counts=counts,
            source=str(path),
            sha256=digest,
            observed=int(entry["observed"]),
            possible=int(entry["possible"]),
            total_kmers=int(entry["total_kmers"]),
        )
    return loaded


#: Fraction of *available* memory a count vector may take before it is
#: memory-mapped instead of read. Decided against the host rather than against a
#: constant, because the same k = 7 vector is 10.24 GB on a workstation where that
#: is a large share of what is free and a rounding error on the 2 TB pod. The
#: difference is not cosmetic: at k = 7 nearly every context is distinct, so a
#: memory-mapped conditional turns each scored position into a random read and the
#: ceiling costs more than the model it is a null for -- measured at 62 s against
#: 212 s for one 48-record cell when only k = 7 was mapped.
_MEMORY_FRACTION = 0.4


def _fits_in_memory(size: int) -> bool:
    available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    return size <= _MEMORY_FRACTION * available


class FragmentConditional:
    """D3.j-A5: the UniRef50 fragment conditional at one order, as a substitutable model.

    ``P(c | the previous order-1 residues)`` from the pinned counts, as a plug-in
    maximum likelihood estimate with **no smoothing constant**, which is a choice
    this design does not have to make: a position is scored only where both the
    clean and the substituted k-gram were observed, and the covered fraction is
    reported per order. Substituting a smoothing constant would make the ceiling a
    function of an undeclared parameter exactly where the counts get thin.

    The substitution is the operation the arm receives: every ``r`` **in the
    context** is read as ``s`` while the target is untouched, scored at the same
    positions -- targets whose immediately preceding residue is ``r`` and whose own
    identity is neither ``r`` nor ``s``.

    **What rises with the order, and what that means.** A higher-order conditional
    reads more context, so it loses more when a context symbol is misread, and the
    ceiling climbs. It also becomes a weaker *estimator* and a stronger *lookup*:
    at k = 7 the staged corpus gives about thirteen observations per k-mer and 14%
    of them are unobserved. Under audit §7.0 that is not a defect of the ceiling --
    corpus statistics is what recombination means, and a lookup is the strongest
    form of it -- but it is why the curve reports each order's coverage and mean
    observations per k-mer beside its Delta, and why the k = 3 rung the
    pre-registration froze stays in the table.
    """

    def __init__(self, ordered: OrderedFragmentCounts, *, residues: Sequence[str] = AA20) -> None:
        self.ordered = ordered
        self.order = int(ordered.order)
        if self.order < 1:
            raise ValueError("a conditional needs a positive order")
        self.counts = ordered.counts
        self.index = {residue: AA20.index(residue) for residue in residues}
        self._powers = (len(AA20) ** np.arange(self.order - 1, -1, -1)).astype(np.int64)

    def _encoded(self, record: str) -> np.ndarray:
        return np.asarray([AA20.find(character) for character in record], dtype=np.int64)

    def _totals(self, contexts: np.ndarray) -> np.ndarray:
        """Total count of each context, as a gather over twenty contiguous cells."""

        unique, inverse = np.unique(contexts, return_inverse=True)
        block = unique[:, None] * len(AA20) + np.arange(len(AA20), dtype=np.int64)
        summed = np.asarray(self.counts[block.reshape(-1)]).reshape(block.shape).sum(axis=1)
        return summed[inverse]

    def damage(self, records: Sequence[str], substituted: str, substitute: str) -> dict[str, Any]:
        """The same contrast the arm is scored under, on the corpus model itself."""

        target_row, source_row = self.index[substituted], self.index[substitute]
        order = self.order
        deltas: list[np.ndarray] = []
        eligible = 0
        per_sum = np.zeros(len(records), dtype=np.float64)
        per_count = np.zeros(len(records), dtype=np.int64)
        for index, record in enumerate(records):
            codes = self._encoded(record)
            if codes.size <= max(order, 1) or (codes < 0).any():
                continue
            ends = np.arange(max(order - 1, 1), codes.size, dtype=np.int64)
            preceding, targets = codes[ends - 1], codes[ends]
            keep = (preceding == target_row) & (targets != target_row) & (targets != source_row)
            if not keep.any():
                continue
            ends = ends[keep]
            eligible += int(ends.size)
            offsets = np.arange(-(order - 1), 1, dtype=np.int64)
            windows = codes[ends[:, None] + offsets[None, :]]
            swapped = windows.copy()
            if order > 1:
                swapped[:, :-1] = np.where(windows[:, :-1] == target_row, source_row, windows[:, :-1])
            clean_flat = windows @ self._powers
            dirty_flat = swapped @ self._powers
            clean_count = np.asarray(self.counts[clean_flat], dtype=np.float64)
            dirty_count = np.asarray(self.counts[dirty_flat], dtype=np.float64)
            clean_total = self._totals(clean_flat // len(AA20)).astype(np.float64)
            dirty_total = self._totals(dirty_flat // len(AA20)).astype(np.float64)
            usable = (clean_count > 0) & (dirty_count > 0) & (clean_total > 0) & (dirty_total > 0)
            if not usable.any():
                continue
            clean_log = np.log(clean_count[usable] / clean_total[usable])
            dirty_log = np.log(dirty_count[usable] / dirty_total[usable])
            delta = clean_log - dirty_log
            deltas.append(delta)
            per_sum[index] = float(delta.sum())
            per_count[index] = int(delta.size)
        if not deltas:
            return {
                "order": order,
                "nats_per_scored_token": None,
                "n_scored_tokens": 0,
                "n_eligible_positions": eligible,
                "measurable": False,
                "unmeasurable_reason": (
                    "no position has both its clean and its substituted k-gram observed "
                    "in the corpus at this order, so the conditional is undefined there "
                    "and this design does not smooth"
                ),
                "per_record_nll_sum": per_sum,
                "per_record_n_scored": per_count,
            }
        pooled = np.concatenate(deltas)
        return {
            "order": order,
            "nats_per_scored_token": float(pooled.mean()),
            "n_scored_tokens": int(pooled.size),
            "n_eligible_positions": eligible,
            "scored_fraction": pooled.size / eligible if eligible else 0.0,
            "measurable": True,
            "per_record_nll_sum": per_sum,
            "per_record_n_scored": per_count,
        }

    def damage_by_sequence(
        self,
        runs_by_record: Sequence[Sequence[str]],
        substituted: str,
        substitute: str,
    ) -> dict[str, Any]:
        """Pool every residue run of one original sequence, then keep one row per sequence."""

        n_records = len(runs_by_record)
        per_sum = np.zeros(n_records, dtype=np.float64)
        per_count = np.zeros(n_records, dtype=np.int64)
        eligible = 0
        for index, runs in enumerate(runs_by_record):
            if not runs:
                continue
            record = self.damage(list(runs), substituted, substitute)
            eligible += int(record["n_eligible_positions"])
            if record["measurable"]:
                per_sum[index] = float(
                    record["nats_per_scored_token"] * record["n_scored_tokens"]
                )
                per_count[index] = int(record["n_scored_tokens"])
        total = int(per_count.sum())
        if total == 0:
            return {
                "order": self.order,
                "nats_per_scored_token": None,
                "n_scored_tokens": 0,
                "n_eligible_positions": eligible,
                "measurable": False,
                "unmeasurable_reason": (
                    "no position has both its clean and its substituted k-gram observed "
                    "in the corpus at this order, so the conditional is undefined there "
                    "and this design does not smooth"
                ),
                "per_record_nll_sum": per_sum,
                "per_record_n_scored": per_count,
            }
        return {
            "order": self.order,
            "nats_per_scored_token": float(per_sum.sum() / total),
            "n_scored_tokens": total,
            "n_eligible_positions": eligible,
            "scored_fraction": total / eligible if eligible else 0.0,
            "measurable": True,
            "per_record_nll_sum": per_sum,
            "per_record_n_scored": per_count,
        }


# ------------------------------------------------------------------ invariants


def intervention_invariants(
    scorer: DamageScorer,
    *,
    symbol_token: int,
    alphabet_tokens: Sequence[int],
    seed: int,
    tolerance: float,
) -> dict[str, Any]:
    """Refuse to measure unless the write does exactly what it claims.

    Two required checks and one reported diagnostic, in the shape
    ``src.transfer.das.invariants`` fixed for the residual-stream patch.

    The **null** check writes a symbol's own row over itself, and requires damage
    to be *exactly* zero. Not small: the two passes see identical weights,
    identical rows and identical chunking, so any difference at all is a defect,
    and a tolerance here would be a tolerance on correctness rather than on noise.

    The **positive** check *replaces* the row with a seeded random vector at the
    alphabet's mean row norm -- the same operation the measurement performs, with
    a substitute that carries no symbol -- and requires the likelihood to **move**,
    in either direction. Two choices in that sentence were made against their
    obvious alternatives. Replacement rather than addition, because adding a
    same-norm random vector leaves the original row at forty-five degrees and the
    model still largely reading the original symbol, which makes the control weak
    exactly where it has to be strong. Magnitude rather than sign, because a
    random row is not a worse reader of every context than the row it replaced,
    and on a symbol whose own row is unhelpful at the scored positions the
    substitution can *raise* the likelihood; a one-sided check would call that a
    dead hook. A null check alone cannot see a write that never lands -- that
    failure is on record in ``path_patching`` -- which is why this control exists
    at all.

    The **diagnostics** are a matched pair of *offsets* -- the row plus a constant
    vector, and the row plus a random direction of the same norm -- reported and
    never required. They exist because ``das.invariants`` records that a
    constant-vector control reports a bound intervention as unbound, measuring
    1.9e-5 against 0.62 for a random direction of equal norm on gpt2-large, and
    the reason is LayerNorm's mean subtraction, which removes a constant offset
    exactly. An RMSNorm decoder has no mean subtraction and does not. **Only an
    offset can show this**; a constant *replacement* destroys the symbol whatever
    the normalisation does, so the two diagnostics are offsets even though the
    control is a replacement, and the ratio between them is this arm's own
    instance of the fact ``src.transfer.circuits`` resolves per architecture
    rather than assuming.
    """

    if tolerance <= 0.0:
        raise ValueError("a positive-control tolerance must be positive")
    weight = scorer.model.weight
    rows = torch.as_tensor(list(alphabet_tokens), dtype=torch.long, device=weight.device)
    scale = float(weight.data[rows].to(torch.float32).norm(dim=-1).mean())
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("the alphabet's embedding rows have no usable norm")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    direction = torch.randn(weight.shape[-1], generator=generator, dtype=torch.float32)
    direction = direction / direction.norm() * scale
    constant = torch.full((weight.shape[-1],), scale / math.sqrt(weight.shape[-1]))
    base = weight.data[symbol_token].detach().to(torch.float32).cpu()
    identity = scorer.damage(symbol_token, symbol_token)
    if not identity["measurable"]:
        raise ValueError(
            f"token {symbol_token} has no scored position once its own identity is "
            "excluded, so it cannot carry the invariants; probe the most-read symbol"
        )
    perturbed = scorer.damage(symbol_token, symbol_token, values=direction)
    random_offset = scorer.damage(symbol_token, symbol_token, values=base + direction)
    constant_offset = scorer.damage(symbol_token, symbol_token, values=base + constant)
    record = {
        "probe_token": int(symbol_token),
        "identity_damage_nats": identity["nats_per_scored_token"],
        "identity_scored_tokens": identity["n_scored_tokens"],
        "random_replacement_damage_nats": perturbed["nats_per_scored_token"],
        "random_replacement_absolute_move_nats": abs(perturbed["nats_per_scored_token"]),
        "random_offset_damage_nats": random_offset["nats_per_scored_token"],
        "constant_offset_damage_nats": constant_offset["nats_per_scored_token"],
        "constant_over_random_offset_ratio": (
            abs(constant_offset["nats_per_scored_token"])
            / abs(random_offset["nats_per_scored_token"])
            if abs(random_offset["nats_per_scored_token"]) > 0.0
            else None
        ),
        "diagnostic_reading": (
            "a small constant-over-random offset ratio is LayerNorm's mean "
            "subtraction annihilating the constant, which is why a constant vector "
            "is the wrong positive control on such an arm; a ratio near one is an "
            "RMSNorm arm, which has no mean to subtract"
        ),
        "perturbation_norm": scale,
        "tolerance": float(tolerance),
        "seed": int(seed),
        "perturbation": (
            "CONTROL: the row is REPLACED by a seeded random vector at the alphabet's "
            "mean row norm -- the measurement's own operation with a substitute that "
            "carries no symbol. DIAGNOSTICS: the same random vector and a constant "
            "vector of the same norm are ADDED to the row instead, which is the only "
            "form in which a normalisation's mean subtraction is visible, and they "
            "are reported rather than required"
        ),
    }
    if identity["nats_per_scored_token"] != 0.0:
        raise RuntimeError(
            "writing a symbol's own embedding row over itself moved the held-out "
            f"likelihood by {identity['nats_per_scored_token']!r} nats/token. The "
            "intervention is not the identity when it must be, so every measured "
            "damage is partly the write itself"
        )
    if abs(perturbed["nats_per_scored_token"]) <= tolerance:
        raise RuntimeError(
            f"replacing an embedding row by a random vector of norm {scale:.4g} "
            f"moved the likelihood by {perturbed['nats_per_scored_token']:.3g} "
            f"nats/token, within the {tolerance:.3g} tolerance: the write is not "
            "reaching the forward pass. A null-only check cannot see this"
        )
    return record


def reachability_verdict(
    damages: Sequence[float], classes: Sequence[str], *, margin: float
) -> dict[str, Any]:
    """Rule 40 on the agreement set, read before any contradiction-set number.

    A failure is an instrument failure and voids the contradiction read; it is not
    a negative about the model, and criteria in this programme have been retracted
    for reporting one as the other.
    """

    values = np.asarray(damages, dtype=np.float64)
    labels = np.asarray(list(classes))
    if values.shape != labels.shape or values.size == 0:
        raise ValueError("every agreement pair needs a damage value and a class")
    if not np.isfinite(values).all():
        raise ValueError("an agreement pair produced a non-finite damage")
    dissimilar = values[labels == AGREEMENT_CLASSES[0]]
    similar = values[labels == AGREEMENT_CLASSES[1]]
    if dissimilar.size == 0 or similar.size == 0:
        raise ValueError("the check needs pairs at both ends of the agreement set")
    difference = float(dissimilar.mean() - similar.mean())
    return {
        "dissimilar_mean_nats": float(dissimilar.mean()),
        "similar_mean_nats": float(similar.mean()),
        "difference_nats": difference,
        "margin_nats": float(margin),
        "n_dissimilar": int(dissimilar.size),
        "n_similar": int(similar.size),
        "dissimilar_pair_damage": [float(value) for value in dissimilar],
        "similar_pair_damage": [float(value) for value in similar],
        "reachable": bool(dissimilar.mean() > 0.0 and difference >= margin),
        "criterion": (
            "on the agreement set, where the two axes concur, the most dissimilar "
            "pairs must damage the model more than the most similar pairs by the "
            "pre-registered margin, and the dissimilar end must do positive damage"
        ),
        "consequence_if_failed": (
            "the contradiction-set read is VOID as an instrument failure and is not "
            "measured; a null there would not be a statement about the model"
        ),
    }


# ------------------------------------------------------- the estimand and its null


def _quadrant_delta(truth: np.ndarray, prediction: np.ndarray) -> float:
    """``mean over the +1 class - mean over the -1 class``: D3.j-A4's contrast."""

    high = np.asarray(prediction)[np.asarray(truth) > 0]
    low = np.asarray(prediction)[np.asarray(truth) < 0]
    if high.size == 0 or low.size == 0:
        return float("nan")
    return float(high.mean() - low.mean())


def delta_contrast(
    *,
    codes: np.ndarray,
    damage: Sequence[float],
    groups: np.ndarray,
    seed: int,
    n_bootstrap: int,
    reference: Sequence[float] | None = None,
    reference_name: str = "zero",
) -> dict[str, Any]:
    """``Delta`` with an interval, and against a reference when one is given.

    Both prediction vectors are scored on the same resampled rows by
    :func:`~src.transfer.statistics.paired_group_bootstrap`, which is exactly what
    the ceiling comparison needs: ``Delta_arm`` and ``Delta_ceiling`` are
    quadrant means over the *same* pairs, so their difference must be resampled
    together or its interval describes two independent draws it never took. With
    no reference the right-hand vector is zero, which makes ``difference`` the
    arm's own ``Delta`` and gives it its own interval.
    """

    values = np.asarray(damage, dtype=np.float64)
    other = (
        np.zeros_like(values) if reference is None else np.asarray(reference, dtype=np.float64)
    )
    if values.shape != codes.shape or other.shape != values.shape:
        raise ValueError("codes, damage and reference must align pair for pair")
    if not np.isfinite(values).all() or not np.isfinite(other).all():
        raise ValueError("a damage vector carries a non-finite value")
    floor = bootstrap_unit_floor(int(np.unique(groups).size))
    block: dict[str, Any] = {
        "delta": _quadrant_delta(codes, values),
        "reference_delta": _quadrant_delta(codes, other),
        "reference_name": reference_name,
        "n_pairs": int(values.size),
        "n_pairs_positive_class": int((codes > 0).sum()),
        "n_pairs_negative_class": int((codes < 0).sum()),
        "unit_floor": floor,
        "resampling_unit": "substituted symbol",
        "definition": (
            "Delta = mean damage over the chemically-dissimilar / "
            "distributionally-similar quadrant minus mean damage over the "
            "chemically-similar / distributionally-dissimilar quadrant (D3.j-A4). "
            "The chemical account predicts Delta > 0, the corpus-symbol account "
            "Delta < 0"
        ),
    }
    if floor["degenerate"]:
        block["difference"] = block["delta"] - block["reference_delta"]
        block["difference_ci95"] = None
        block["interval_withheld_reason"] = floor["degenerate_reason"]
        return block
    resampled = paired_group_bootstrap(
        codes, values, other, groups, _quadrant_delta, seed=seed, n_bootstrap=n_bootstrap
    )
    block["difference"] = resampled["difference"]
    block["difference_ci95"] = resampled["difference_ci95"]
    block["group_bootstrap"] = {
        key: resampled[key]
        for key in (
            "n_bootstrap_requested", "n_finite_draws", "n_non_finite_draws",
            "n_groups", "minimum_groups",
        )
    }
    return block


def random_direction_delta_null(
    *, codes: np.ndarray, per_direction: Sequence[Sequence[float]], observed: float
) -> dict[str, Any]:
    """D3.j-A5's norm-matched control: directions, not positions (rule 39).

    Each draw replaces the *substitute* by a norm-matched random row while keeping
    the quadrant structure, so what survives is whatever contrast the composition
    of the two quadrants produces on its own. At least
    :data:`MINIMUM_RANDOM_DIRECTIONS` distinct rows, because the quantity being
    estimated is a spread over directions and more scored positions per direction
    would not estimate it.
    """

    if len(per_direction) < MINIMUM_RANDOM_DIRECTIONS:
        raise ValueError(
            f"{len(per_direction)} random substitute rows is below the declared "
            f"{MINIMUM_RANDOM_DIRECTIONS}; rule 39 asks for more directions, and a "
            "percentile over fewer is one of a handful of order statistics"
        )
    deltas = [_quadrant_delta(codes, np.asarray(values, dtype=np.float64)) for values in per_direction]
    if not all(math.isfinite(value) for value in deltas):
        raise ValueError("a random-direction draw produced a non-finite Delta")
    array = np.asarray(deltas)
    quantile = float(np.quantile(array, 0.95))
    return {
        "n_directions": int(array.size),
        "minimum_directions": int(MINIMUM_RANDOM_DIRECTIONS),
        "deltas": [float(value) for value in array],
        "q95": quantile,
        "median": float(np.median(array)),
        "observed_delta": float(observed),
        "observed_above_q95": bool(observed > quantile),
        "construction": (
            "the substitute row is a seeded random vector scaled to the mean "
            "embedding-row norm of the alphabet; the quadrant labels are unchanged, "
            "so the null retains the composition of the two quadrants and destroys "
            "only the substitute's identity"
        ),
    }


def ceiling_margin(
    *,
    delta_block: Mapping[str, Any],
    ceiling_block: Mapping[str, Any],
    random_null: Mapping[str, Any],
    factor: float,
) -> dict[str, Any]:
    """The standing §7.0 margin, clause by clause and named.

    ``twice the ceiling`` is stated against the ceiling's **positive part**. The
    standing rule is written for an excess over chance, which is non-negative by
    construction; ``Delta_ceiling`` is not, and it is predicted to be negative
    here, so doubling a negative number would make the clause weaker the more
    strongly the corpus account holds. Clamping at zero is the conservative
    reading and is declared rather than left to arithmetic.
    """

    delta = float(delta_block["delta"])
    # ``reference_delta`` and NOT ``delta``: both fields live on the same block,
    # because ``delta_contrast`` scores the arm and the ceiling on one resampled
    # set of rows, and ``delta`` there is the ARM's Delta. Reading the wrong one
    # turns the clause into ``delta >= factor * delta``, which no positive effect
    # can ever satisfy -- a margin that fails every arm and looks like a result.
    ceiling = float(ceiling_block["reference_delta"])
    interval = ceiling_block.get("difference_ci95")
    clauses = {
        "delta_positive": delta > 0.0,
        "difference_interval_excludes_zero_from_above": bool(
            interval is not None and interval[0] > 0.0
        ),
        "at_least_factor_times_ceiling": delta >= factor * max(ceiling, 0.0),
        "above_random_direction_q95": bool(random_null["observed_above_q95"]),
    }
    return {
        "arm_delta": delta,
        "ceiling_delta": ceiling,
        "difference": float(ceiling_block["difference"]),
        "difference_ci95": interval,
        "factor": float(factor),
        "clauses": clauses,
        "cleared": all(clauses.values()),
        "rule": (
            "audit §7.0's standing margin: the paired group-bootstrap 95% interval of "
            "the arm-minus-ceiling difference excludes zero over at least eight "
            f"groups, the arm's own Delta is at least {factor} times the ceiling's "
            "positive part, and it exceeds the 95th percentile of the norm-matched "
            "random-substitute directions"
        ),
    }


def ceiling_adequacy(
    arm_damage: Sequence[float], ceiling_damage: Sequence[float], *, floor: float
) -> dict[str, Any]:
    """Is the recombination ceiling doing anything on this estimand?

    Measured rather than assumed, because a corpus fragment model is a strong
    baseline for *sequence likelihood* and need not be one for *this* quantity.
    A 3-mer conditional's dependence on the identity of a single context residue
    is small -- adjacent residues in protein carry very little mutual information
    -- so its damage under a substitution can be a small fraction of a decoder's,
    and where it is, the standing margin's multiple of it is a bar anything
    clears.
    """

    arm = np.abs(np.asarray(arm_damage, dtype=np.float64))
    ceiling = np.abs(np.asarray(ceiling_damage, dtype=np.float64))
    if arm.shape != ceiling.shape or arm.size == 0:
        raise ValueError("the arm and the ceiling must be scored on the same pairs")
    if arm.mean() <= 0.0:
        raise ValueError("the arm produced no damage at all on this pair set")
    ratio = float(ceiling.mean() / arm.mean())
    return {
        "arm_mean_absolute_damage": float(arm.mean()),
        "ceiling_mean_absolute_damage": float(ceiling.mean()),
        "ratio": ratio,
        "floor": float(floor),
        "adequate": bool(ratio >= floor),
        # ``None`` and not NaN where the ceiling is constant: the k = 1 rung is
        # exactly zero for every pair by construction, so a rank correlation
        # against it has no value, and write_json refuses NaN three steps later
        # where the cause is no longer visible.
        "per_pair_spearman": (
            None
            if ceiling.std() == 0.0 or arm.std() == 0.0
            else float(stats.spearmanr(arm, ceiling).statistic)
        ),
        "status": "declared diagnostic, reported beside the verdict and never a gate",
        "reading": (
            "the ceiling produces a comparable amount of damage, so the standing "
            "margin's multiple of it is a bar with teeth"
            if ratio >= floor
            else (
                "the ceiling produces far less damage than the arm on this estimand, "
                "so 'at least the declared factor times the ceiling' is close to "
                "'greater than zero' and that clause carries little weight. The "
                "clauses doing the work are the norm-matched random-substitute "
                "control and the sign of the contrast on the contradiction set; a "
                "verdict from this cell must be quoted with this ratio"
            )
        ),
    }


def protein_verdict(
    *, margin: Mapping[str, Any], delta_block: Mapping[str, Any]
) -> dict[str, Any]:
    """What the contradiction-set read says, under §7.0's halt-and-classify clause."""

    delta = float(delta_block["delta"])
    interval = delta_block.get("difference_ci95")
    if margin["cleared"]:
        return {
            "verdict": "CHEMISTRY",
            "reason": (
                "on pairs where chemical and corpus-distributional similarity "
                "disagree, damage is larger for the chemically dissimilar substitute, "
                "and the effect clears the recombination ceiling under the standing "
                "margin. This licenses a candidate and nothing more: §8's causal, "
                "retrieval-aware and independent-biological clauses stay open"
            ),
            "delta": delta,
        }
    if interval is not None and interval[1] < 0.0:
        return {
            "verdict": "RECOMBINATION",
            "reason": (
                "damage tracks corpus context similarity rather than chemistry on the "
                "contradiction set, with the interval excluding zero from below. Under "
                "§7.0's halt-and-classify clause this is recorded as recombination and "
                "halts the line rather than being reported as a weak positive"
            ),
            "delta": delta,
        }
    if delta > 0.0:
        return {
            "verdict": "INSIDE_CEILING",
            "reason": (
                "the effect is in the chemical direction but does not clear the "
                "recombination ceiling under the standing margin; clearing a null "
                "admits nothing and this result is not a chemistry result"
            ),
            "delta": delta,
            "failed_clauses": [
                name for name, held in margin["clauses"].items() if not held
            ],
        }
    return {
        "verdict": "UNDECIDED",
        "reason": (
            "the contrast does not separate the two accounts at this cohort size and "
            "cut; the interval spans zero and the ceiling is not cleared"
        ),
        "delta": delta,
    }


def protein_verdict_b(
    *, margin: Mapping[str, Any], crossed: Mapping[str, Any]
) -> dict[str, Any]:
    """D3.j-B verdict: the crossed arm interval decides, never the symbol-only one."""

    if crossed.get("refused"):
        return {
            "verdict": "VOID",
            "reason": CROSSED_INTERVAL_REFUSED,
            "detail": crossed.get("refusal"),
            "delta": crossed.get("delta"),
        }
    delta = float(crossed["delta"])
    arm_interval = crossed.get("delta_ci95")
    if margin["cleared"]:
        return {
            "verdict": "CHEMISTRY",
            "reason": (
                "on pairs where chemical and corpus-distributional similarity "
                "disagree, damage is larger for the chemically dissimilar substitute, "
                "and the effect clears the recombination ceiling under the standing "
                "margin read from the crossed sequence-by-symbol interval"
            ),
            "delta": delta,
            "delta_ci95": arm_interval,
        }
    if arm_interval is not None and arm_interval[1] < 0.0:
        return {
            "verdict": "RECOMBINATION",
            "reason": (
                "the crossed arm interval lies entirely below zero, so damage tracks "
                "the fragment-damage axis rather than chemistry"
            ),
            "delta": delta,
            "delta_ci95": arm_interval,
        }
    if delta > 0.0:
        return {
            "verdict": "INSIDE_CEILING",
            "reason": (
                "the crossed arm point is in the chemical direction but does not "
                "clear the recombination ceiling under the standing margin"
            ),
            "delta": delta,
            "delta_ci95": arm_interval,
            "failed_clauses": [
                name for name, held in margin["clauses"].items() if not held
            ],
        }
    return {
        "verdict": "UNDECIDED",
        "reason": (
            "the crossed arm interval does not separate the two accounts and the "
            "ceiling is not cleared"
        ),
        "delta": delta,
        "delta_ci95": arm_interval,
    }


def tokenizer_identity(arm: Arm, *, max_tokens: int) -> dict[str, Any]:
    """What must match between a construction artefact and a confirmation arm."""

    tokenizer = arm.tokenizer
    vocab = int(getattr(getattr(arm.model, "config", None), "vocab_size", 0) or 0)
    return {
        "arm": arm.spec.name,
        "architecture": arm.spec.architecture,
        "tokenisation": arm.spec.tokenisation,
        "input_format": arm.spec.input_format,
        "vocab_size": vocab,
        "tokenizer_class": type(tokenizer).__name__,
        "max_tokens": int(max_tokens),
    }


def cohorts_independent(
    construction_records: Sequence[str], evaluation_records: Sequence[str]
) -> dict[str, Any]:
    """Exact-content and near-duplicate disjointness of two B draws."""

    construct = list(construction_records)
    evaluation = list(evaluation_records)
    shared = sorted(set(construct) & set(evaluation))
    union = construct + evaluation
    groups, grouping = near_duplicate_groups(union, unit="residues")
    construct_groups = {int(groups[index]) for index in range(len(construct))}
    evaluation_groups = {
        int(groups[index]) for index in range(len(construct), len(union))
    }
    shared_groups = sorted(construct_groups & evaluation_groups)
    independent = not shared and not shared_groups
    if shared:
        reason = "EXACT_CONTENT_OVERLAP"
    elif shared_groups:
        reason = "NEAR_DUPLICATE_OVERLAP"
    else:
        reason = None
    return {
        "independent": independent,
        "reason": reason,
        "n_shared_records": len(shared),
        "n_shared_near_duplicate_groups": len(shared_groups),
        "construction_grouping": grouping,
    }


def frozen_pair_set(
    members: Mapping[str, Sequence[str]], labels: Sequence[str]
) -> PairSet:
    """Rebuild the admitted ordered pairs from a frozen construction artefact."""

    index = {label: position for position, label in enumerate(labels)}
    pairs: list[tuple[int, int]] = []
    classes: list[str] = []
    for name in QUADRANTS:
        for token in members[name]:
            if len(token) != 2 or token[0] not in index or token[1] not in index:
                raise ValueError(f"frozen member {token!r} is not a residue pair")
            x, y = index[token[0]], index[token[1]]
            pairs.extend([(x, y), (y, x)])
            classes.extend([name, name])
    if not pairs:
        raise ValueError("the frozen contradiction set admits no pair")
    return PairSet(tuple(pairs), tuple(classes))


def text_control_verdict(delta_block: Mapping[str, Any]) -> dict[str, Any]:
    """D3.j-A1: the byte-level control passes only if it detects substitute similarity."""

    interval = delta_block.get("difference_ci95")
    passed = bool(
        float(delta_block["delta"]) > 0.0 and interval is not None and interval[0] > 0.0
    )
    return {
        "verdict": "PASS" if passed else "VOID",
        "delta": float(delta_block["delta"]),
        "difference_ci95": interval,
        "criterion": (
            "damage under a distributionally similar substitute must be smaller than "
            "under a dissimilar one, with the paired group-bootstrap interval "
            "excluding zero over at least eight groups (D3.j-A1)"
        ),
        "consequence": (
            "the readout detects substitute similarity, which is the only half of the "
            "design text can validate; it says nothing about chemistry"
            if passed
            else "the readout is VOID as a specification defect and no protein arm may be read"
        ),
    }


# -------------------------------------------- secondary, reported per-axis reading


def _pearson(truth: np.ndarray, prediction: np.ndarray) -> float:
    left = np.asarray(truth, dtype=np.float64)
    right = np.asarray(prediction, dtype=np.float64)
    if left.std() == 0.0 or right.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _centre_within(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    centred = np.asarray(values, dtype=np.float64).copy()
    for group in np.unique(groups):
        selected = groups == group
        centred[selected] -= centred[selected].mean()
    return centred


def _residual(target: np.ndarray, control: np.ndarray) -> np.ndarray:
    denominator = float(control @ control)
    if denominator <= 0.0:
        raise ValueError(
            "the embedding-distance control carries no variance once ranked and "
            "centred within the substituted symbol, which cannot happen on a real "
            "embedding and means the control vector was mis-assembled"
        )
    return target - (float(target @ control) / denominator) * control


def association_vectors(
    damage: Sequence[float],
    chemical: Sequence[float],
    distributional: Sequence[float],
    embedding: Sequence[float],
    groups: np.ndarray,
    *,
    control_for_embedding: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rank, centre within the substituted symbol, and optionally residualise."""

    columns = [
        _centre_within(stats.rankdata(np.asarray(values, dtype=np.float64)), groups)
        for values in (damage, chemical, distributional, embedding)
    ]
    if not control_for_embedding:
        return columns[0], columns[1], columns[2]
    control = columns[3]
    return tuple(_residual(column, control) for column in columns[:3])  # type: ignore[return-value]


def association(
    *,
    damage: Sequence[float],
    chemical: Sequence[float],
    distributional: Sequence[float],
    embedding: Sequence[float],
    groups: np.ndarray,
    seed: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    """Each axis's association with damage, and their difference. **Secondary.**

    D3.j-A4's frozen statistic is the quadrant contrast, not a correlation, and
    this block decides nothing. It is reported because the contrast collapses the
    pair set to two means and a reader who wants to know whether damage *orders*
    with either axis inside the quadrants cannot recover that from the contrast.

    The trivial explanation for any ordering is that a substitute whose embedding
    row is simply nearer does less harm, so embedding distance is regressed out
    and the controlled block is the one to read. Every vector is ranked across the
    pair set, centred within the substituted symbol -- a symbol whose reads are
    intrinsically more informative must not induce an association through its own
    offset -- and then residualised.
    """

    floor = bootstrap_unit_floor(int(np.unique(groups).size))
    blocks: dict[str, Any] = {}
    for name, controlled in (("raw", False), ("embedding_distance_controlled", True)):
        prepared = association_vectors(
            damage, chemical, distributional, embedding, groups,
            control_for_embedding=controlled,
        )
        chemical_score = _pearson(prepared[0], prepared[1])
        distributional_score = _pearson(prepared[0], prepared[2])
        block: dict[str, Any] = {
            "chemical_association": chemical_score,
            "distributional_association": distributional_score,
            "difference": chemical_score - distributional_score,
            "n_pairs": len(prepared[0]),
            "unit_floor": floor,
        }
        if floor["degenerate"]:
            block["difference_ci95"] = None
            block["interval_withheld_reason"] = floor["degenerate_reason"]
        else:
            resampled = paired_group_bootstrap(
                prepared[0], prepared[1], prepared[2], groups, _pearson,
                seed=seed, n_bootstrap=n_bootstrap,
            )
            block["difference_ci95"] = resampled["difference_ci95"]
        blocks[name] = block
    blocks["deciding"] = "embedding_distance_controlled"
    blocks["status"] = "reported beside D3.j-A4's contrast and decides nothing"
    return blocks


def shuffled_difference_null(
    *,
    damage: Sequence[float],
    codes: np.ndarray,
    groups: np.ndarray,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    """``Delta`` when damage is detached from the pair it came from. **A detector.**

    Damage is permuted within the substituted symbol, so the null keeps the group
    structure, the quadrant labels and the per-symbol damage scale and destroys
    only the pairing between a substitute and what it cost. Under §7.0 clearing
    this admits nothing -- the ceiling is the thing to clear -- and it is reported
    because a *failure* to clear it is informative about the instrument.
    """

    if draws < 20:
        raise ValueError(
            "a null reported as a distribution needs draws to be a distribution; "
            "below twenty its 97.5th percentile is one of two order statistics"
        )
    values = np.asarray(damage, dtype=np.float64)
    observed = _quadrant_delta(codes, values)
    rng = np.random.default_rng(seed)
    sampled: list[float] = []
    for _ in range(draws):
        permuted = values.copy()
        for group in np.unique(groups):
            selected = np.flatnonzero(groups == group)
            permuted[selected] = values[rng.permutation(selected)]
        candidate = _quadrant_delta(codes, permuted)
        if math.isfinite(candidate):
            sampled.append(candidate)
    if len(sampled) < draws // 2:
        raise RuntimeError(
            f"only {len(sampled)} of {draws} shuffled draws produced a finite Delta"
        )
    null = np.asarray(sampled)
    absolute = float(np.quantile(np.abs(null), 0.95))
    return {
        "draws_requested": int(draws),
        "draws_finite": int(null.size),
        "observed_delta": float(observed),
        "quantiles": {str(q): float(np.quantile(null, q)) for q in (0.025, 0.5, 0.975)},
        "absolute_q95": absolute,
        "observed_outside_null_q95": bool(abs(observed) > absolute),
        "two_sided_fraction_at_least_as_extreme": float((np.abs(null) >= abs(observed)).mean()),
        "status": (
            "a detection criterion. Under audit §7.0 clearing a shuffled null admits "
            "nothing; the recombination ceiling is the thing to clear"
        ),
        "permutation": "damage permuted within the substituted symbol",
    }


def embedding_distance(weight: torch.Tensor, tokens: Sequence[int]) -> np.ndarray:
    """Euclidean distance between the embedding rows of an alphabet, in float32."""

    rows = weight.data[torch.as_tensor(list(tokens), dtype=torch.long, device=weight.device)]
    matrix = rows.to(torch.float32).cpu().numpy().astype(np.float64)
    difference = matrix[:, None, :] - matrix[None, :, :]
    return np.sqrt((difference**2).sum(axis=-1))


def norm_matched_random_rows(
    weight: torch.Tensor, tokens: Sequence[int], *, count: int, seed: int
) -> list[torch.Tensor]:
    """``count`` distinct random substitute rows at the alphabet's mean row norm."""

    if count < 1:
        raise ValueError("at least one random substitute row is needed")
    rows = torch.as_tensor(list(tokens), dtype=torch.long, device=weight.device)
    scale = float(weight.data[rows].to(torch.float32).norm(dim=-1).mean())
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    drawn = []
    for _ in range(count):
        vector = torch.randn(weight.shape[-1], generator=generator, dtype=torch.float32)
        drawn.append(vector / vector.norm() * scale)
    return drawn


# ------------------------------------------------- synthetic known-answer world


class SyntheticAlphabetModel:
    """A bigram decoder whose next-token logits read one declared half of the row.

    Deliberately the smallest object that is still the *same* object the real path
    measures: an embedding matrix that can be written into, and a logit function
    that consumes it. The swap, the mask, the cross-entropy, the axes, the
    quadrants, the contrast, the interval and both nulls run unchanged, so the
    known-answer check exercises the pipeline rather than a model of it.
    """

    def __init__(self, embedding: torch.Tensor, head: torch.Tensor) -> None:
        if embedding.ndim != 2 or head.shape != embedding.shape:
            raise ValueError("the head must read the embedding it is paired with")
        self._weight = embedding.to(torch.float32).clone()
        self._head = head.to(torch.float32).clone()

    @property
    def weight(self) -> torch.Tensor:
        return self._weight

    def logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        del attention_mask  # the synthetic corpus has no padding
        return self._weight[input_ids] @ self._head.T


@dataclass(frozen=True)
class SyntheticWorld:
    """A corpus and a decoder built so that one of the two accounts is true."""

    planted: str
    model: SyntheticAlphabetModel
    symbols: tuple[Symbol, ...]
    property_table: dict[str, list[float]]
    scoring_ids: torch.Tensor
    background_ids: torch.Tensor
    settings: dict[str, Any]

    def context_counts(self) -> np.ndarray:
        """Joint context counts under the world's own declared coarsening.

        Coarsened for the reason the byte-level control is: an uncoarsened joint
        over twenty symbols is 400 cells estimated from a corpus that can be
        generated in a second, and the resulting profiles are so sparse that most
        pairs have disjoint support and identical cosine distance. The bucket
        count is part of the world's declaration and is recorded with it.
        """

        buckets = int(self.settings["n_context_buckets"])
        return token_context_counts(
            [row.tolist() for row in self.background_ids],
            self.symbols,
            bucket_of_token=np.arange(len(self.symbols), dtype=np.int64) % buckets,
            n_buckets=buckets,
        )

    def cohort(self) -> ScoringCohort:
        return ScoringCohort(
            input_ids=self.scoring_ids,
            attention_mask=torch.ones_like(self.scoring_ids),
            target_mask=torch.ones(
                (self.scoring_ids.shape[0], self.scoring_ids.shape[1] - 1), dtype=torch.bool
            ),
        )


#: The three worlds the known-answer self-test runs. ``neither`` is the one that
#: makes the check two-sided in the way that matters: its decoder reads a third,
#: nuisance block of the embedding row that **neither** axis measures, so the
#: substitutions do real damage and that damage is unrelated to both axes. An
#: instrument that returned "chemistry" there would be reading the quadrants'
#: composition rather than the substitute, which is the one failure this design's
#: own random-direction control exists to catch.
PLANTINGS = ("chemistry", "distribution", "neither")


def _markov_ids(transition: np.ndarray, *, sequences: int, length: int, rng: Any) -> torch.Tensor:
    size = transition.shape[0]
    out = np.zeros((sequences, length), dtype=np.int64)
    out[:, 0] = rng.integers(0, size, size=sequences)
    cumulative = transition.cumsum(axis=1)
    for position in range(1, length):
        draws = rng.random(sequences)
        out[:, position] = [
            int(np.searchsorted(cumulative[state], draw))
            for state, draw in zip(out[:, position - 1], draws)
        ]
    return torch.from_numpy(np.minimum(out, size - 1))


def synthetic_world(
    *,
    planted: str,
    seed: int,
    n_property: int = 4,
    n_context: int = 8,
    n_nuisance: int = 6,
    n_context_buckets: int = 5,
    sequences: int = 96,
    length: int = 384,
    property_weight: float = 0.6,
    context_weight: float = 0.9,
    nuisance_weight: float = 0.9,
) -> SyntheticWorld:
    """A world where the answer is known, and known not to be trivially recoverable.

    Every symbol carries two independent halves in its embedding row: a
    **property** vector, which the declared chemical axis is computed from, and a
    **context** code, which the generated corpus's statistics are driven by. The
    two are drawn independently, so the two axes disagree on a large share of
    pairs -- which is what makes the contradiction set non-empty here for the same
    reason it is non-empty on UniRef50.

    The corpus is generated from **both** halves, so chemistry is genuinely
    present in it and the distributional axis is not a relabelling of the
    chemical one. The decoder reads only the half named by ``planted``, and that
    is what the pipeline has to recover. Both plantings are run in the stage's
    self-test, because an instrument that could only ever return one of the two
    answers would pass a one-sided check.

    Twenty symbols labelled with the residue letters, so that the artefact's
    pair labels read the way a real one does; nothing here is a residue.
    """

    if planted not in PLANTINGS:
        raise ValueError(f"unknown planting {planted!r}; declared: {list(PLANTINGS)}")
    size = len(AA20)
    rng = np.random.default_rng(seed)
    properties = rng.standard_normal((size, n_property))
    context = rng.standard_normal((size, n_context))
    nuisance = rng.standard_normal((size, n_nuisance))
    property_readout = rng.standard_normal((size, n_property))
    context_readout = rng.standard_normal((size, n_context))
    nuisance_readout = rng.standard_normal((size, n_nuisance))

    true_logits = (
        property_weight * properties @ property_readout.T
        + context_weight * context @ context_readout.T
    )
    transition = np.exp(true_logits - true_logits.max(axis=1, keepdims=True))
    transition /= transition.sum(axis=1, keepdims=True)

    # The nuisance block is in every world's embedding, not only the world that
    # reads it, so that embedding distance carries the same amount of structure
    # neither axis explains in all three and the embedding-distance control is
    # doing the same work in each.
    embedding = np.concatenate([properties, context, nuisance], axis=1)
    blocks = {
        "chemistry": (property_weight * property_readout, 0.0, 0.0),
        "distribution": (0.0, context_weight * context_readout, 0.0),
        "neither": (0.0, 0.0, nuisance_weight * nuisance_readout),
    }[planted]
    head = np.concatenate(
        [
            blocks[0] if isinstance(blocks[0], np.ndarray) else np.zeros((size, n_property)),
            blocks[1] if isinstance(blocks[1], np.ndarray) else np.zeros((size, n_context)),
            blocks[2] if isinstance(blocks[2], np.ndarray) else np.zeros((size, n_nuisance)),
        ],
        axis=1,
    )
    return SyntheticWorld(
        planted=planted,
        model=SyntheticAlphabetModel(
            torch.from_numpy(embedding).float(), torch.from_numpy(head).float()
        ),
        symbols=tuple(Symbol(letter, index) for index, letter in enumerate(AA20)),
        property_table={
            f"synthetic_property_{axis}": [float(value) for value in properties[:, axis]]
            for axis in range(n_property)
        },
        scoring_ids=_markov_ids(transition, sequences=sequences, length=length, rng=rng),
        background_ids=_markov_ids(transition, sequences=sequences, length=length, rng=rng),
        settings={
            "planted": planted,
            "seed": int(seed),
            "n_symbols": int(size),
            "n_property": int(n_property),
            "n_context": int(n_context),
            "n_nuisance": int(n_nuisance),
            "n_context_buckets": int(n_context_buckets),
            "sequences": int(sequences),
            "length": int(length),
            "property_weight": float(property_weight),
            "context_weight": float(context_weight),
            "nuisance_weight": float(nuisance_weight),
            "construction": (
                "the corpus is generated from both the property and the context half, "
                "so chemistry is genuinely present in it; the decoder reads only the "
                "planted half, which is what the contradiction-set read must recover"
            ),
            "ceiling_not_exercised": (
                "the 3-mer recombination ceiling is not built here: a synthetic corpus "
                "of this size leaves most trigrams unobserved, and FragmentConditional "
                "refuses an incomplete background rather than smoothing it. The ceiling "
                "is exercised against the staged background in tests/"
            ),
        },
    )


__all__ = [
    "AGREEMENT_CLASSES",
    "ArmAlphabetModel",
    "BLOSUM62_ORDER",
    "BLOSUM62_ROWS",
    "BLOSUM62_SIDE_NOTE",
    "BLOSUM62_SOURCE",
    "CHEMICAL_AXIS_SOURCE",
    "CUTS",
    "DISTRIBUTIONAL_METRICS",
    "DTYPE",
    "DamageScorer",
    "FRAGMENT_ORDERS",
    "FragmentConditional",
    "OrderedFragmentCounts",
    "PRE_REGISTERED_FRAGMENT_ORDER",
    "GRANTHAM_POLARITY",
    "GRANTHAM_SOURCE",
    "INPUT_EMBEDDING_PATH",
    "INTERVENTION",
    "MINIMUM_QUADRANT_PAIRS",
    "MINIMUM_RANDOM_DIRECTIONS",
    "MINIMUM_SYMBOL_TOKEN_COVERAGE",
    "OUTPUT_HEAD_PATH",
    "PLANTINGS",
    "PRE_REGISTRATION",
    "PRE_REGISTRATION_AMENDMENTS",
    "PRE_REGISTRATION_TRACK",
    "QUADRANTS",
    "PairSet",
    "PropertyAxis",
    "ScoringCohort",
    "Symbol",
    "SyntheticAlphabetModel",
    "SyntheticWorld",
    "TEXT_BANDS",
    "admit_arm",
    "agreement_extremes",
    "association",
    "association_vectors",
    "blosum62_distance",
    "CEILING_ADEQUACY_FLOOR",
    "AXIS_CONSTRUCTED",
    "B_CONFIRMATION_INDICES",
    "B_STAGE_CONFIRM",
    "B_STAGE_CONSTRUCT",
    "B_STAGES",
    "CEILING_CONSTRUCTION_VOID",
    "CROSSED_INTERVAL_REFUSED",
    "EXPERIMENT_B",
    "FRAGMENT_AXIS_SYMMETRIZATION",
    "KIND_AXIS_CONSTRUCTION",
    "PRE_REGISTRATION_TRACK_B",
    "cohorts_independent",
    "frozen_pair_set",
    "PROTEIN_AXES",
    "PROTEIN_AXIS_CONTEXT_PROFILE",
    "PROTEIN_AXIS_FRAGMENT_SUBSTITUTION_DAMAGE",
    "agreement_extremes_observed",
    "cap_pairs",
    "ceiling_adequacy",
    "ceiling_margin",
    "chemical_property_table",
    "context_counts",
    "context_profiles",
    "cosine_distance",
    "cut_sweep",
    "cut_sweep_observed",
    "delta_contrast",
    "fragment_damage_axis",
    "embedding_distance",
    "intervention_invariants",
    "load_ordered_counts",
    "lowercase_letter_buckets",
    "matching_ceiling_predicts_distributional_side",
    "norm_matched_random_rows",
    "ordered_pair_set",
    "property_distance",
    "protein_alphabet",
    "protein_verdict",
    "protein_verdict_b",
    "tokenizer_identity",
    "quadrants_at_cut",
    "quadrants_at_cut_observed",
    "random_direction_delta_null",
    "reachability_verdict",
    "residue_runs_by_row",
    "residue_context_counts",
    "residue_context_profiles_at_order",
    "shuffled_difference_null",
    "substituted_row",
    "symbol_token_coverage",
    "symmetrize_directional_damage",
    "symmetric_kl_distance",
    "synthetic_world",
    "text_alphabet",
    "text_control_pair_set",
    "text_control_verdict",
    "token_context_counts",
]
