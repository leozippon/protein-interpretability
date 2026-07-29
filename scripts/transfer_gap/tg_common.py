"""Shared loaders for the text-vs-protein interpretability transfer screen.

**This module owns no input rendering and no model loading of its own.** Both are
imported from :mod:`src.transfer.arms`, which is the single declaration of what
each panel member is fed. That is not a stylistic preference: this file used to
carry a second, divergent renderer, and the divergence is how a 1.42 nats/token
defect survived long enough to reverse the sign of a headline result.

The defect, recorded so it is not reintroduced. ``protein_input`` returned the
plain amino-acid sequence for ProtGPT2, with the rationale that using one
rendering for all three protein arms held content constant. It does not. ProtGPT2
was pretrained on FASTA-formatted UniRef50 -- hard-wrapped at 60 residues,
end-of-text separated -- and its BPE merges were learned over exactly that byte
stream. Scoring it on one unwrapped line does not hold anything constant; it
penalises whichever arm is furthest from the rendering that happened to be
chosen. Measured on 80 Swiss-Prot sequences of 600-2000 residues (EXP-R2-028):
raw 8.046, end-of-text + raw 8.090, wrapped-at-60 6.652, end-of-text + wrapped
6.623 nats/token. ProtGPT2's context information moved from -1.31 to +2.23.

What this module does own is **which draw**, and it owns two halves of that.
Every cohort built here is a seeded draw from the complete eligible set, so a
cohort is a sample from the corpus rather than a sample from its first block, and
``skip`` partitions that permutation rather than walking further down the file --
file order is a documented hazard of this programme, worth +1.01 nats on ProGen2
when it was last measured (EXP-R2-059), three times over.

The second half is **which order**, and it was missing. ``arms.selected_positions``
returns a seeded draw sorted back into ascending corpus order, so the identity of
a cohort was seeded while its order stayed file order -- and six stages consume a
cohort positionally, by slicing it or by breaking out of it at a token cap. Each
of those was therefore reading the corpus-earliest part of a seeded set, which is
rule 1 wearing a seed. :func:`in_seeded_record_order` permutes once, at the point
of construction, so no stage has to remember.

Nothing here falls back silently: a missing corpus, a missing model, an arm whose
rendering needs metadata the cohort does not carry, or a loss-recovered
denominator too small to divide by all raise.

``analysis_layer`` and ``write_json`` are re-exported from ``src.transfer`` for
the same reason, and both replaced a local copy. The depth conversion existed
here six times in two mutually inconsistent forms -- four ``int(round(...))``
(round-half-to-even) and two ``floor(... + 0.5)`` (round-half-up) -- which agree
on every fraction any TG stage actually passes, so unifying them moved no
recorded number, and would not have stayed that way.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer.arms import (  # noqa: E402
    AA20,
    PANEL,
    REPO,
    Arm,
    Cohort,
    iter_fasta,
    protein_cohort,
    symbols_per_token,
    text_cohort,
)
from src.transfer.arms import load_arm as _load_arm  # noqa: E402
from src.transfer.arms import tokenize_batch as _tokenize_batch  # noqa: E402
from src.transfer.io import write_json as _write_json  # noqa: E402
from src.transfer.scoring import analysis_layer  # noqa: E402

__all__ = [
    "AA20",
    "REPO",
    "Arm",
    "Cohort",
    "DEFAULT_COHORT_SEED",
    "MIN_ABLATION_HEADROOM_NATS",
    "analysis_layer",
    "build_cohort",
    "cohort_for",
    "cohort_provenance",
    "in_seeded_record_order",
    "iter_fasta",
    "load_arm",
    "load_text",
    "loss_recovered",
    "protein_input",
    "symbol_position_mask",
    "symbols_per_token",
    "tokenize_batch",
    "write_json",
]

#: The four arms this screen was designed around. Kept as a declaration rather
#: than a filter: :func:`load_arm` accepts anything in ``PANEL``, but a result
#: written for an arm outside this set is not comparable to the recorded table.
TG_PANEL = ("gpt2-large", "protgpt2", "zymctrl", "progen2-medium")

#: Seed for every cohort permutation in the series. One constant rather than a
#: per-script default, so two stages of the same run draw from the same ordering
#: and ``skip`` is a genuine partition across scripts as well as within one.
DEFAULT_COHORT_SEED = 20260729

#: Below this, ``ce_mean_ablated - ce_clean`` is not a denominator. Loss recovered
#: is a ratio against the causal headroom of the ablated site, and the panel spans
#: headrooms from ~0.02 to ~7 nats/token; dividing a difference of order 0.1 nats
#: by a headroom of order 0.01 produces a number with the right units and no
#: content. 0.5 nats/token is the floor ``src.transfer.budget`` already applies to
#: context information for the same reason, reused here so the two agree.
MIN_ABLATION_HEADROOM_NATS = 0.5


# ------------------------------------------------------------------- models


def load_arm(
    name: str,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    attn_implementation: str | None = None,
) -> Arm:
    """Load a panel member through :func:`src.transfer.arms.load_arm`.

    Delegation is the point. The declared shape check, the observed-dtype check
    and the pad-token resolution all live in one place, and an arm loaded here is
    the same object ``scripts/transfer`` loads.
    """

    if name not in PANEL:
        raise KeyError(f"unknown arm {name!r}; panel is {sorted(PANEL)}")
    return _load_arm(
        name, device=device, dtype=dtype, attn_implementation=attn_implementation
    )


def tokenize_batch(arm: Arm, texts: list[str], max_len: int):
    """Right-padded ids and a validity mask, from the shared implementation.

    The local copy this replaces fell back to token id 0 when a tokenizer
    declared neither a pad nor an end-of-text token. Over ProGen2's 32-symbol
    vocabulary, id 0 is a real symbol.
    """

    return _tokenize_batch(arm, texts, max_len)


# ------------------------------------------------------------------ corpora


def cohort_for(
    arm: Arm,
    n: int,
    res_min: int,
    res_max: int,
    *,
    skip: int = 0,
    seed: int = DEFAULT_COHORT_SEED,
    min_chars: int = 800,
) -> Cohort:
    """A seeded-permutation cohort drawn from the arm's native corpus.

    ``skip`` partitions the permutation rather than advancing through the file:
    ``cohort_for(arm, 4000, ...)`` and ``cohort_for(arm, 120, ..., skip=4000)``
    are disjoint samples of the same corpus, which is what the callers that pass
    ``skip`` were asking for and not what file order gave them.

    **This function no longer selects anything itself.** It used to carry its own
    eligible-record enumeration for all three corpora -- a second
    ``set(sequence) <= AA20`` filter, a second parquet reader, a second
    ``<start>``/``<end>``/``<sep>`` parser -- plus its own permutation, because
    ``arms.protein_cohort`` and ``arms.text_cohort`` drew records in FASTA file
    order and offered no way to ask for anything else. They now take ``seed=`` and
    permute, so the whole layer is gone and the selection rule is the panel's.

    That is Appendix B rule 12 applied one step earlier than rendering. The
    rendering defect of the audit's section 0.1 survived a withdrawal precisely
    because two implementations existed and only one was fixed; a duplicated
    *eligibility* predicate is the same hazard, deciding which records exist
    before anything decides how they are drawn.

    The draw is returned in **seeded record order**, not corpus order, by
    :func:`in_seeded_record_order`. See that function for why; in short, the set
    is seeded but ``arms.selected_positions`` hands it back sorted, and half the
    stages in this series consume a cohort positionally.

    One consequence is recorded rather than hidden. ``Cohort.digest`` hashes an
    ordered list, so a digest recorded before 2026-07-29 will not reproduce byte
    for byte even though the sequences behind it are the same.
    """

    if n < 1:
        raise ValueError(f"cohort size must be positive, got {n}")
    source = PANEL[arm.name].source
    name = f"{source}_n{n}_skip{skip}_seed{seed}"
    if source == "openwebtext":
        drawn = text_cohort(n, min_chars=min_chars, skip=skip, name=name, seed=seed)
    elif source in ("zymctrl_ec", "swissprot"):
        drawn = protein_cohort(
            n,
            res_min,
            res_max,
            skip=skip,
            name=name,
            with_ec=source == "zymctrl_ec",
            seed=seed,
        )
    else:
        raise ValueError(f"{arm.name}: unsupported cohort source {source!r}")
    return in_seeded_record_order(drawn, seed)


def in_seeded_record_order(cohort: Cohort, seed: int) -> Cohort:
    """The same records, permuted under ``seed`` instead of left in corpus order.

    **A seeded set is not a seeded prefix, and this series kept confusing the
    two.** ``arms.selected_positions`` decides *which* records a draw contains
    from the seed and then returns them in ascending corpus order, because the
    collecting pass sweeps the corpus once. The identity of the set is therefore
    seeded; the order is file order. Every stage that then consumes part of a
    cohort -- and most of them do -- was consuming the corpus-earliest part of it:

    * ``tg01`` fitted its symbol-level Markov ladder on ``base_raw[:4000]`` of an
      8000-record cohort;
    * ``tg07`` and ``tg09`` stop ``collect`` at a token cap of 200k against a
      4000-sequence cohort, so they read roughly a fifth of it -- and an
      *arm-dependent* fifth, because ProtGPT2's multi-residue BPE yields about
      half as many tokens per protein as ProGen2's residue tokenizer, which
      compared four arms on prefixes of four different depths;
    * ``tg03`` and ``tg08`` slice ``eval_texts[:n]``;
    * ``tg08``'s data-axis low point trains on ``pool[:full // 16]``, the first
      sixteenth of the activation pool in sequence order -- a near-clonal
      homologue block, and that low point is what the "budget-limited" reading of
      limitation L3 rests on;
    * ``tg06`` draws 2000 sequences and keeps the first 400 usable windows.

    Swiss-Prot is sorted by accession, which groups by source organism and by
    curation date, so a corpus-order prefix is the rule-1 hazard that has
    manufactured an effect in this programme three times, once worth +1.01
    nats/token (EXP-R2-059). Permuting once, here, makes every one of those
    slices a sample of the drawn cohort instead, without any stage having to
    remember to do it -- which is the only version of this fix that stays fixed.

    The EC labels travel with their sequences. A cohort whose labels were
    permuted independently of its records would feed ZymCTRL another protein's
    conditioning tag, which is a 1.73-nat prompt (EXP-R2-034) attached to the
    wrong sequence.
    """

    if not cohort.records:
        raise ValueError(f"cohort {cohort.name!r} has no records to order")
    order = [int(i) for i in np.random.default_rng(seed).permutation(len(cohort.records))]
    metadata = dict(cohort.metadata)
    labels = metadata.get("ec_labels")
    if labels is not None:
        if len(labels) != len(cohort.records):
            raise ValueError(
                f"cohort {cohort.name!r} carries {len(labels)} EC labels for "
                f"{len(cohort.records)} records; they cannot be reordered together"
            )
        metadata["ec_labels"] = [labels[i] for i in order]
    metadata["record_order"] = {
        "mode": "seeded_permutation_of_the_drawn_set",
        "seed": int(seed),
        "reason": (
            "arms.selected_positions returns a seeded draw in ascending corpus "
            "order; a stage that slices or short-circuits over a cohort would "
            "otherwise be reading the corpus-earliest part of it"
        ),
    }
    return replace(
        cohort, records=[cohort.records[i] for i in order], metadata=metadata
    )


def build_cohort(
    arm: Arm,
    n: int,
    res_min: int,
    res_max: int,
    skip: int = 0,
    *,
    seed: int = DEFAULT_COHORT_SEED,
):
    """Return ``(model-input strings, symbol strings)`` for any arm.

    The input strings come from :meth:`src.transfer.arms.Cohort.input_strings`,
    which is the only renderer in this repository. The symbol strings are the
    tokenizer-independent axis: characters for text, residues for protein.
    """

    cohort = cohort_for(arm, n, res_min, res_max, skip=skip, seed=seed)
    return cohort.input_strings(arm), cohort.records


def load_text(n: int, min_chars: int = 800, skip: int = 0,
              *, seed: int = DEFAULT_COHORT_SEED) -> list[str]:
    """Documents drawn under a seeded permutation of the whole OpenWebText subset.

    Delegates to :func:`src.transfer.arms.text_cohort` for the same reason
    :func:`cohort_for` does: one eligibility predicate, one draw. The records
    come back in seeded order for the reason :func:`in_seeded_record_order`
    gives: a caller that keeps the first usable ``k`` of them would otherwise be
    keeping the earliest ``k`` shard entries.
    """

    return in_seeded_record_order(
        text_cohort(
            n,
            min_chars=min_chars,
            skip=skip,
            name=f"openwebtext_n{n}_skip{skip}",
            seed=seed,
        ),
        seed,
    ).records


def protein_input(arm: Arm, seq: str, ec_label: str | None = None) -> str:
    """Model-specific input string for one bare amino-acid sequence.

    A one-record cohort routed through the shared renderer, so a caller that
    reaches this function gets byte-for-byte what :func:`build_cohort` produces.
    ProtGPT2 receives the end-of-text prefix and the 60-column wrap; ProGen2 its
    N-to-C control token; ZymCTRL its EC tag, which must therefore be supplied.

    ZymCTRL without ``ec_label`` raises. That is the intended behaviour and not
    an inconvenience: the unconditioned rendering is off ZymCTRL's training
    distribution and its EC tag is separately measured at 1.73 nats of
    conditioning leak (EXP-R2-034), so a caller with no label is asking for a
    number that means neither thing.
    """

    metadata = {"ec_labels": [ec_label]} if ec_label is not None else {}
    cohort = Cohort(
        name="single_record",
        kind="text" if arm.modality == "text" else "protein",
        records=[seq],
        min_symbols=len(seq),
        max_symbols=len(seq),
        metadata=metadata,
    )
    return cohort.input_strings(arm)[0]


def cohort_provenance(cohort: Cohort, arm: Arm) -> dict:
    """The record that lets a reader tell two cohorts, or two renderings, apart.

    ``selection`` is :attr:`src.transfer.arms.Cohort.sampling`'s ``mode`` and
    nothing else. It used to be ``sampling.get("mode", "file_order")``, which
    substituted a *specific and hazardous* claim for an absence: a cohort built
    without a sampling record emitted ``selection: "file_order"`` beside
    ``sampling: null``, asserting the one draw this programme has been burned by
    three times about an artefact that recorded no draw at all. ``Cohort.sampling``
    already declares the correct answer -- ``mode: "unrecorded"`` with the hazard
    text -- so the default here was a second, wrong copy of a decision that had
    been made properly one import away. The Failure Principle inverted: the
    silent fallback was the dangerous value, not the safe one.
    """

    lengths = sorted(len(record) for record in cohort.records)
    sampling = cohort.sampling
    return {
        "name": cohort.name,
        "kind": cohort.kind,
        "digest": cohort.digest,
        "sequences": len(cohort.records),
        # arms.sampling_record is the source of these four; it carries the same
        # facts under `sampling` and adds the file-order hazard text when there
        # is no seed. The old keys are kept as the artefact spelling.
        "selection": sampling["mode"],
        "seed": sampling.get("seed"),
        "skip": sampling.get("skip"),
        "eligible_records": sampling.get("eligible_records"),
        "sampling": sampling,
        # Which records were drawn and which order they are consumed in are two
        # different facts, and only the first was ever recorded. See
        # `in_seeded_record_order`.
        "record_order": cohort.metadata.get("record_order"),
        "symbols_min": lengths[0],
        "symbols_median": lengths[len(lengths) // 2],
        "symbols_max": lengths[-1],
        "input_format": PANEL[arm.name].input_format,
        "tokenisation": PANEL[arm.name].tokenisation,
    }


# ------------------------------------------------------------- measurement


def loss_recovered(
    ce_clean: float,
    ce_ablated: float,
    ce_intervened: float,
    *,
    min_headroom: float = MIN_ABLATION_HEADROOM_NATS,
) -> dict:
    """Loss recovered with its denominator declared rather than assumed.

    ``(ce_ablated - ce_intervened) / (ce_ablated - ce_clean)``. The denominator is
    the causal headroom of the ablated site and it is arm-specific: across this
    panel it ranges from about 0.02 to about 7 nats/token, and on a de-leaked
    ZymCTRL it has been measured *negative*. A ratio against a headroom near zero
    is not a weak measurement, it is not a measurement, so this returns ``None``
    with the reason attached instead of a large number.
    """

    headroom = ce_ablated - ce_clean
    if headroom < min_headroom:
        return {
            "loss_recovered": None,
            "ablation_headroom_nats": headroom,
            "denominator_valid": False,
            "denominator_floor_nats": min_headroom,
            "denominator_refusal": (
                "mean-ablation headroom below the floor; loss recovered is a ratio "
                "against this quantity and is not defined here"
            ),
        }
    return {
        "loss_recovered": (ce_ablated - ce_intervened) / headroom,
        "ablation_headroom_nats": headroom,
        "denominator_valid": True,
        "denominator_floor_nats": min_headroom,
        "denominator_refusal": None,
    }


def symbol_position_mask(arm: Arm, ids: torch.Tensor) -> torch.Tensor:
    """True where a token carries at least one alphabet symbol.

    A native rendering puts non-symbol tokens into the scored stream: ProtGPT2's
    FASTA wrap contributes a newline every 60 residues plus an end-of-text
    prefix, and ZymCTRL's EC tag and ``<sep>``/``<start>``/``<end>`` markers are
    tokens too. Those positions belong in the model's own cross-entropy -- they
    are part of the distribution it was trained on -- but a claim about how well
    a dictionary reconstructs *protein* computation should be checkable against
    the residue-bearing positions alone. This mask makes that check possible
    instead of leaving the two conflated.
    """

    alphabet = set(AA20) if arm.modality == "protein" else None
    flat = ids.reshape(-1).tolist()
    cache: dict[int, bool] = {}
    keep = []
    for token in flat:
        hit = cache.get(token)
        if hit is None:
            piece = arm.tokenizer.decode([token])
            hit = (
                any(c in alphabet for c in piece)
                if alphabet is not None
                else bool(piece.strip())
            )
            cache[token] = hit
        keep.append(hit)
    return torch.tensor(keep, dtype=torch.bool, device=ids.device).reshape(ids.shape)


def write_json(path: Path, payload: dict) -> None:
    """Write one stage artefact through the shared writer, and say where.

    Delegation, for the same reason :func:`load_arm` delegates. The local copy
    this replaces was a third ``write_json`` in the repository and the only
    non-atomic one: it truncated the destination and then serialised into it, so
    a stage that raised mid-write left a syntactically broken file where a reader
    checking only the schema version would expect a complete one.

    The progress line is kept because these stages are run interactively and it
    is the only thing that says which of eleven arms has landed.
    """

    _write_json(path, payload)
    print(f"wrote {path}")
