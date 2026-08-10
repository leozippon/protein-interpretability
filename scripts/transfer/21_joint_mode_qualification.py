#!/usr/bin/env python3
"""Is one joint language-protein checkpoint measurable in BOTH of its modes?

D1.d asks which differences remain when text and protein modes share one
checkpoint, and it admits nothing until **one dense joint decoder is qualified in
both modes**. This stage is that qualification, and it is an external baseline
rather than a panel measurement: ``--checkpoint`` names a directory, not an arm,
because a checkpoint that has not passed this stage must not be in ``arms.py`` at
all.

**Estimand, one per mode.** Context information: the held-out unigram
cross-entropy on the scored symbols minus the model's own clean cross-entropy on
the same symbols. It is stage 01's estimand deliberately -- a joint checkpoint's
qualification has to be commensurable with the panel's -- and it carries stage
01's reading with it: an arm below the threshold is **unmeasurable on that
cohort**, not failing. That distinction is the whole verdict. It says the
evaluation interface cannot resolve anything on this cohort, and it says nothing
about what the checkpoint knows.

**What one scored symbol is, is the family's declaration.** Galactica and
InstructProtein reach a per-residue alphabet, so a protein symbol is a residue and
the magnitude is in nats per residue. ProLLaMA does not -- ``Seq=<...>`` over the
unmodified LLaMA-2 vocabulary merges residue runs, and that merged form is what
the lineage was trained on -- so its protein symbol is a token. The estimand
follows the declared unit rather than the other way round: the unigram support,
the clean cross-entropy's denominator and the reported unit all move together.

That mobility is exactly why every protein record carries its **measured residues
per scored token** and an explicit cross-arm comparability verdict. A magnitude in
nats per token is not commensurable with one in nats per residue, and it is not
commensurable with another arm's nats per token either when the arms differ in
residues per token (Appendix B rule 26, limitation L23). The artefact says so in
its own field rather than leaving a reader to notice.

**The unigram is held out and its support is declared.** It is fitted on a draw
that is disjoint from the scored one by construction (a later window of the same
seeded permutation) and then deduplicated by content, because Swiss-Prot carries
the same sequence under several accessions (Appendix B rule 3). Its support is
always the set of ids a scored target of that mode can take -- and that set is
read from ``joint_modes``, so it follows the declared symbol unit: the twenty
residue ids for a residue-unit family, every residue-spelling token (463 on the
staged LLaMA-2 vocabulary) for a token-unit one, and the model's whole vocabulary
for text. Laplace smoothing inflates a cross-entropy by roughly ``log(1 + s*V/N)``,
and on the 41682-target ProLLaMA reference this stage's defaults draw that is
0.0005 nats over twenty symbols, 0.011 over 463 and 0.570 over the full 32000 --
so fitting a protein reference over the whole vocabulary would put more smoothing
bias into the baseline than the effect being measured.

**Controls, because the headline number alone cannot be read.**

``reversed``      the same sequences scored backwards under the same rendering. A
                  decoder carrying real directional sequence structure cannot be
                  indifferent to reversal, so a small reversal cost is what
                  distinguishes "this cohort is hard" from "this mode is not
                  reading the sequence at all". It is a within-arm difference over
                  an identical residue multiset, which is why it survives the unit
                  problem above and why the decision rule leans on it hardest: the
                  cost is read **per residue** in both units. For a token-unit
                  family that is also the only readable form -- reversal permutes
                  the residues but not the tokenisation, so the reversed condition
                  is a different token population and neither a per-token cost nor
                  the forward reference applies to it.
``naive``         the same sequences under the rendering with the per-residue
                  escape removed -- what an unaided ``AutoTokenizer`` call
                  produces. It prices the rendering (Appendix B rule 4) in the
                  estimand's own per-residue units, and it is the one measurement
                  in this stage that is deliberately not verified against the
                  per-residue alphabet. **Withheld with its reason** for a family
                  that declares no escape: there "the escape removed" is the
                  declared rendering itself, the price is zero by construction,
                  and no substitute control is invented.
``residue_mass``  the model's probability mass on its residue tokens at every
                  scored TEXT position, where the family declares a residue
                  subspace disjoint from text. This is what identifies a
                  *collapsed* text mode -- a model treating arbitrary prose as
                  the prefix of a protein -- rather than merely a bad number.
                  Withheld with its reason where the residues are ordinary
                  letters of the text vocabulary, because there the statistic
                  would be a fact about twenty capital letters.

The rendering itself is declared in ``src.transfer.joint_modes`` and nowhere else
(Appendix B rule 12); this file names a family and reads the rest from there.
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer import joint_modes  # noqa: E402
from src.transfer.arms import (  # noqa: E402
    DEFAULT_CORPUS_DRAW_SEED,
    REPO,
    Cohort,
    protein_cohort,
    require_input_path,
    text_cohort,
)
from src.transfer.budget import MIN_CONTEXT_INFORMATION_NATS, power_status  # noqa: E402
from src.transfer.io import sha256_file, write_json  # noqa: E402
from src.transfer.pathways import (  # noqa: E402
    LAPLACE_SMOOTHING,
    assert_disjoint,
    disjoint_unigram_cross_entropy_nats,
    held_out_cohort,
)

SCHEMA_VERSION = "r2_transfer_joint_mode_qualification_v1"
DEFAULT_OUT = REPO / "results/transfer/joint_mode_qualification"

#: Modules whose content decides this stage's numbers, hashed into the artefact.
#: The rendering module is first because it is the one that has been worth 2.9
#: nats/token when wrong.
PROVENANCE_MODULES = (
    "src/transfer/joint_modes.py",
    "src/transfer/arms.py",
    "src/transfer/budget.py",
    "src/transfer/pathways.py",
    "src/transfer/io.py",
)

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

VERDICT_NOTE = (
    "a mode below --min-context-information is reported UNMEASURABLE ON THIS "
    "COHORT, not failing. It is a statement about this cohort and this evaluation "
    "interface; downstream analyses must exclude the mode rather than report a "
    "negative result from it (01_cohort_power.py's rule)"
)


# ------------------------------------------------------------------- checkpoint


def load_tokenizer(path: Path) -> tuple[Path, Any]:
    """The tokenizer alone, so a wrong checkpoint/family pair fails before the weights.

    Resolving the rendering needs nothing but the tokenizer, and it is the check
    most likely to refuse: a checkpoint whose delimiters or per-residue alphabet
    are not the declared family's is a configuration error, and paying a
    multi-gigabyte load to discover it is the shape stage 01 already moved its own
    refusals ahead of.
    """

    resolved = require_input_path(Path(path).resolve(), "--checkpoint")
    return resolved, AutoTokenizer.from_pretrained(str(resolved))


def load_model(
    resolved: Path, tokenizer: Any, *, device: str, dtype: str
) -> tuple[Any, dict[str, Any]]:
    """Load the weights and read the shape back from what was built.

    Every fact recorded here is read off the loaded object rather than echoed
    from the request, including the dtype: a build that ignored the requested
    dtype would otherwise be recorded as honouring it.
    """

    if dtype not in _DTYPES:
        raise ValueError(f"unsupported inference dtype {dtype!r}; known: {sorted(_DTYPES)}")
    model = AutoModelForCausalLM.from_pretrained(
        str(resolved),
        # ``torch_dtype`` rather than ``dtype`` for the reason
        # ``src.transfer.arms.load_arm`` records: it is the only spelling both the
        # workstation's transformers and the pod's honour, and the observed-dtype
        # check below is what actually enforces the outcome.
        torch_dtype=_DTYPES[dtype],
        device_map={"": device},
    )
    model.eval()
    config = model.config
    observed = sorted(
        {
            str(parameter.dtype).removeprefix("torch.")
            for parameter in model.parameters()
            if parameter.is_floating_point()
        }
    )
    if observed != [dtype]:
        raise ValueError(f"requested dtype {dtype}, observed {observed}")

    def fact(*names: str) -> int:
        for name in names:
            value = getattr(config, name, None)
            if value is not None:
                return int(value)
        raise ValueError(
            f"this checkpoint's config declares none of {names}, so its shape cannot "
            "be recorded and the artefact could not say what was measured"
        )

    facts = {
        "resolved_path": str(resolved),
        "model_type": str(getattr(config, "model_type", "undeclared")),
        "architectures": list(getattr(config, "architectures", []) or []),
        "n_layers": fact("num_hidden_layers", "n_layer"),
        "d_model": fact("hidden_size", "n_embd", "d_model"),
        "n_heads": fact("num_attention_heads", "n_head"),
        "vocab_size": fact("vocab_size"),
        "max_position_embeddings": fact("max_position_embeddings", "n_positions"),
        "dtype_requested": dtype,
        "dtype_observed": observed,
        "device": device,
        "facts_source": "read back from the loaded model's config and parameters, "
        "not echoed from the request",
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": int(len(tokenizer)),
    }
    return model, facts


# ---------------------------------------------------------------------- scoring


@torch.no_grad()
def score_positions(
    model: Any,
    token_ids: Sequence[int],
    positions: Sequence[int],
    *,
    device: str,
    subspace: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Per-position NLL in nats, and optionally the mass on a token subspace.

    The log-softmax is taken in float32 whatever the parameters are stored in,
    which is the convention EXP-R2-151 measured under; a bfloat16 log-softmax
    quantises at a step comparable with the effects this stage reports.
    """

    if not positions:
        raise ValueError("a scored record must have at least one scored position")
    ids = torch.tensor([list(token_ids)], dtype=torch.long, device=device)
    index = torch.tensor(list(positions), dtype=torch.long, device=device)
    logits = model(ids).logits[0].float()
    log_probabilities = torch.log_softmax(logits[index - 1], dim=-1)
    targets = ids[0, index]
    nll = -log_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
    mass = None
    if subspace is not None:
        columns = torch.tensor(list(subspace), dtype=torch.long, device=device)
        mass = log_probabilities[:, columns].exp().sum(1).double().cpu().numpy()
    return nll.double().cpu().numpy(), mass


def text_token_ids(tokenizer: Any, document: str, *, max_tokens: int) -> list[int]:
    """A text document's ids, truncated to the scored window."""

    return joint_modes.encode(tokenizer, document)[:max_tokens]


# --------------------------------------------------------------------- cohorts


def draw_cohort(args: argparse.Namespace, *, mode: str, n: int, skip: int, name: str) -> Cohort:
    """One draw from the corpus this mode is scored on.

    Both draws go through ``src.transfer.arms``: the seeded permutation, the
    eligibility filter and the sampling record are the panel's, not this stage's
    (Appendix B rules 1 and 12). ``--cohort-draw-seed 0`` still selects the
    historical file-order draw, and the mode reaches the artefact through
    ``Cohort.sampling`` either way.
    """

    seed = args.cohort_draw_seed or None
    if mode == "protein":
        return protein_cohort(
            n,
            args.protein_min_len,
            args.protein_max_len,
            skip=skip,
            name=name,
            seed=seed,
        )
    return text_cohort(n, args.text_min_chars, skip=skip, name=name, seed=seed)


def mode_cohorts(args: argparse.Namespace, mode: str) -> tuple[Cohort, Cohort, dict[str, int]]:
    """The scored cohort and a held-out reference that cannot overlap it.

    Disjointness is enforced twice because the two failures are different. The
    reference is drawn ``--sequences`` records further into the *same* seeded
    permutation, which makes the two windows disjoint by index; it is then
    deduplicated by content, because an index offset is not disjointness when the
    corpus carries the same sequence under several accessions. The final
    assertion is what turns a silent leak into a stopped run.
    """

    scored = draw_cohort(args, mode=mode, n=args.sequences, skip=0, name=f"{mode}_scored")
    candidate = draw_cohort(
        args,
        mode=mode,
        n=args.unigram_sequences,
        skip=args.sequences,
        name=f"{mode}_unigram_reference",
    )
    reference, overlap = held_out_cohort(candidate, scored)
    assert_disjoint(scored, reference)
    return scored, reference, overlap


def cohort_record(cohort: Cohort, *, band_unit: str) -> dict[str, Any]:
    return {
        "name": cohort.name,
        "kind": cohort.kind,
        "n_records": len(cohort),
        "digest": cohort.digest,
        "provenance_digest": cohort.provenance_digest,
        "sampling_record": cohort.sampling,
        "band": [cohort.min_symbols, cohort.max_symbols],
        "band_unit": band_unit,
    }


# --------------------------------------------------------------- unigram support


def scored_target_counts(
    tokenisation: joint_modes.JointTokenisation,
    records: Sequence[str],
    *,
    context: str | None,
) -> np.ndarray:
    """Protein-target counts over exactly the multiset the model is scored on.

    Counted through the same rendering the model sees and over the support the
    rendering declares, so the reference and the scored sample are counted over
    the same kind of symbol whichever symbol unit the family declares. The
    verification inside ``render`` is what guarantees every counted target is one
    the support can represent, so the lookup below cannot silently drop a target.
    """

    order = {value: index for index, value in enumerate(tokenisation.scored_target_ids)}
    counts = np.zeros(len(order), dtype=np.int64)
    for sequence in records:
        rendered = tokenisation.render(sequence, context=context)
        for position in rendered.scored_positions:
            counts[order[rendered.token_ids[position]]] += 1
    if counts.sum() < 1:
        raise RuntimeError("the protein records yielded no scored targets")
    return counts


def text_target_counts(
    tokenizer: Any, records: Sequence[str], *, vocab_size: int, max_tokens: int
) -> np.ndarray:
    """Next-token-target counts over the same window the model is scored on."""

    counts = np.zeros(vocab_size, dtype=np.int64)
    for document in records:
        targets = text_token_ids(tokenizer, document, max_tokens=max_tokens)[1:]
        if not targets:
            continue
        array = np.asarray(targets, dtype=np.int64)
        if array.min() < 0 or array.max() >= vocab_size:
            raise ValueError("a token id fell outside the checkpoint's declared vocabulary")
        counts += np.bincount(array, minlength=vocab_size)
    if counts.sum() < 1:
        raise RuntimeError("the text records yielded no scored targets")
    return counts


def unigram_record(
    reference_counts: np.ndarray,
    scored_counts: np.ndarray,
    *,
    support: str,
    reference: Cohort,
    overlap: dict[str, int],
) -> dict[str, Any]:
    """The held-out context-free baseline, with the support it was fitted over."""

    cross_entropy = disjoint_unigram_cross_entropy_nats(reference_counts, scored_counts)
    total = float(reference_counts.sum())
    size = int(reference_counts.size)
    return {
        "estimator": "disjoint_held_out",
        "support": support,
        "support_size": size,
        "smoothing": float(LAPLACE_SMOOTHING),
        "smoothing_mass_fraction": float(
            LAPLACE_SMOOTHING * size / (total + LAPLACE_SMOOTHING * size)
        ),
        "reference_tokens": int(total),
        "reference_cohort": cohort_record(reference, band_unit=(
            "residues" if reference.kind == "protein" else "characters"
        )),
        "reference_overlap_removed": overlap,
        "cross_entropy_nats": float(cross_entropy),
        "note": (
            "fitted on a draw disjoint from the scored one and deduplicated by "
            "content; never on the scored sample (Appendix B rule 3). The support "
            "is the set of ids a scored target of this mode can take, so the "
            "smoothing bias -- which scales with support size against reference "
            "size -- stays far below the effect being measured"
        ),
    }


def verdict_record(context_information: float, threshold: float) -> dict[str, Any]:
    """The measurability reading, in the vocabulary stage 01 declares."""

    _, status = power_status(context_information, threshold)
    return {
        "verdict": status,
        "minimum_context_information_nats": float(threshold),
        "verdict_note": VERDICT_NOTE,
    }


# ----------------------------------------------------------------- protein mode


def score_protein_records(
    model: Any,
    tokenisation: joint_modes.JointTokenisation,
    records: Sequence[str],
    *,
    device: str,
    context: str | None,
    variant: str,
    max_tokens: int,
) -> dict[str, Any]:
    """One protein condition: total NLL, tokens and residues, all denominators named."""

    total_nll = 0.0
    scored_tokens = 0
    residues = 0
    for sequence in records:
        rendered = tokenisation.render(sequence, context=context, variant=variant)
        if len(rendered.token_ids) > max_tokens:
            raise ValueError(
                f"a rendered protein needs {len(rendered.token_ids)} tokens and "
                f"--max-tokens is {max_tokens}; truncating it would drop the closing "
                "delimiter and silently change the scored span. Raise --max-tokens or "
                "lower --protein-max-len"
            )
        nll, _ = score_positions(
            model, rendered.token_ids, rendered.scored_positions, device=device
        )
        total_nll += float(nll.sum())
        scored_tokens += rendered.n_scored_tokens
        residues += rendered.n_residues
    return {
        "variant": variant,
        "document_context": context,
        "n_records": len(records),
        "n_scored_tokens": scored_tokens,
        "n_scored_residues": residues,
        "residues_per_scored_token": residues / scored_tokens,
        "clean_nll_nats_per_scored_token": total_nll / scored_tokens,
        "clean_nll_nats_per_residue": total_nll / residues,
        "symbol_unit": tokenisation.declaration.symbol_unit,
        "verified_against_declared_symbol_unit": variant == joint_modes.DECLARED,
    }


def comparability_record(
    declaration: joint_modes.JointRendering, residues_per_scored_token: float
) -> dict[str, Any]:
    """Whether this arm's magnitude may be placed beside another arm's, and why not.

    Appendix B rule 26 and limitation L23: an estimand whose unit is the token is
    not a cross-arm estimand when arms differ in symbols per token. The verdict is
    a field rather than a caveat in prose because the number it qualifies is a
    finite, plausible-looking magnitude that reads exactly like a comparable one.
    """

    if declaration.symbol_unit == joint_modes.RESIDUE_UNIT:
        return {
            "verdict": "COMPARABLE_ACROSS_ARMS",
            "symbol_unit": declaration.symbol_unit,
            "measured_residues_per_scored_token": float(residues_per_scored_token),
            "note": (
                "the scored unit is the residue and the declared rendering was "
                "verified at exactly one token per residue, so this magnitude is in "
                "the same units as every other per-residue arm's"
            ),
        }
    return {
        "verdict": "NOT_COMPARABLE_ACROSS_ARMS",
        "symbol_unit": declaration.symbol_unit,
        "measured_residues_per_scored_token": float(residues_per_scored_token),
        "note": (
            f"{declaration.name} declares the TOKEN as its protein symbol unit, and "
            f"one scored token carried {residues_per_scored_token:.3f} residues on "
            "this cohort. Nats per token is therefore not the unit any per-residue "
            "arm reports, and it is not this arm's unit on another cohort either, "
            "since residues per token is a property of the corpus and the tokenizer "
            "together. Do NOT place this magnitude beside a per-residue arm's or "
            "convert it by dividing: the held-out reference is fitted over token "
            "targets, so the quotient is not a per-residue cross-entropy (Appendix B "
            "rule 26, limitation L23). The within-arm controls below -- reversal "
            "above all -- are what this family's verdict rests on"
        ),
    }


def protein_mode(
    args: argparse.Namespace,
    model: Any,
    tokenisation: joint_modes.JointTokenisation,
) -> dict[str, Any]:
    """Context information on the declared protein symbol, with its controls beside it."""

    declaration = tokenisation.declaration
    per_residue = declaration.symbol_unit == joint_modes.RESIDUE_UNIT
    unit = "nats per scored residue" if per_residue else "nats per scored token"
    clean_key = (
        "clean_nll_nats_per_residue" if per_residue else "clean_nll_nats_per_scored_token"
    )

    scored, reference, overlap = mode_cohorts(args, "protein")
    print(f"[protein] {len(scored)} scored records, {len(reference)} held-out reference records")
    unigram = unigram_record(
        scored_target_counts(tokenisation, reference.records, context=args.protein_context),
        scored_target_counts(tokenisation, scored.records, context=args.protein_context),
        support=(
            "the declared residue token ids, the only ids a scored protein target "
            "can take under this rendering"
            if per_residue
            else "every token of this vocabulary spelled purely of canonical "
            "residues, which is the set of ids a scored protein target can take "
            "under a rendering whose symbol unit is the token"
        ),
        reference=reference,
        overlap=overlap,
    )
    declared = score_protein_records(
        model,
        tokenisation,
        scored.records,
        device=args.device,
        context=args.protein_context,
        variant=joint_modes.DECLARED,
        max_tokens=args.max_tokens,
    )
    context_information = unigram["cross_entropy_nats"] - declared[clean_key]
    print(
        f"  declared rendering  {declared[clean_key]:.4f} {unit} "
        f"against unigram {unigram['cross_entropy_nats']:.4f} "
        f"at {declared['residues_per_scored_token']:.3f} residues/token"
    )

    reversed_records = [sequence[::-1] for sequence in scored.records]
    reversed_score = score_protein_records(
        model,
        tokenisation,
        reversed_records,
        device=args.device,
        context=args.protein_context,
        variant=joint_modes.DECLARED,
        max_tokens=args.max_tokens,
    )
    reversed_score["cost_nats_per_residue"] = (
        reversed_score["clean_nll_nats_per_residue"] - declared["clean_nll_nats_per_residue"]
    )
    reversed_score["cost_unit"] = "nats per residue"
    reversed_score["note"] = (
        "the same sequences read C-to-N, and the control this stage's decision rule "
        "leans on hardest because it is a within-arm difference over an identical "
        "residue multiset -- it needs no cross-arm unit to be readable. A decoder "
        "carrying real directional sequence structure cannot be indifferent to it. "
        + (
            "Reversal preserves the residue multiset exactly, so the held-out "
            "reference above applies to this condition unchanged and the cost is a "
            "pure directional-information reading"
            if per_residue
            else "Reversal permutes the residues but NOT the tokenisation, so this "
            "condition is a different token population: the held-out token reference "
            "above does not apply to it, no context information is reported for it, "
            "and the cost is read per residue -- the denominator the two conditions "
            "do share (Appendix B rules 26, 27). Compare the two conditions' "
            "residues_per_scored_token to see how far the token populations differ"
        )
    )

    if declaration.naive_control_available:
        naive = score_protein_records(
            model,
            tokenisation,
            scored.records,
            device=args.device,
            context=args.protein_context,
            variant=joint_modes.NAIVE,
            max_tokens=args.max_tokens,
        )
        naive["price_nats_per_residue"] = (
            naive["clean_nll_nats_per_residue"] - declared["clean_nll_nats_per_residue"]
        )
        naive["price_nats_per_scored_token"] = (
            naive["clean_nll_nats_per_scored_token"]
            - declared["clean_nll_nats_per_scored_token"]
        )
        naive["context_information_nats_per_residue"] = (
            unigram["cross_entropy_nats"] - naive["clean_nll_nats_per_residue"]
        )
        naive["note"] = (
            "the same sequences with the per-residue escape removed -- what an unaided "
            "AutoTokenizer call produces. Deliberately NOT verified against the "
            "per-residue alphabet; it exists to be wrong. Read the per-residue price: "
            "the per-token one compares two different token populations, since this "
            "condition scores merged multi-residue pieces (Appendix B rules 26, 27)"
        )
        print(
            f"  naive rendering     {naive['clean_nll_nats_per_scored_token']:.4f} nats/token "
            f"at {naive['residues_per_scored_token']:.3f} residues/token"
        )
    else:
        naive = {
            "verdict": "WITHHELD",
            "reason": (
                f"{declaration.name} declares no per-residue escape. Its trained "
                "protein format writes residues straight into the checkpoint's own "
                "vocabulary, so 'the same block with the escape removed' IS the "
                "declared rendering: the price would be exactly 0.0 by construction "
                "and would report a control that measured nothing. The rendering cost "
                "this control exists to price does not exist for this family, and no "
                "substitute control is invented in its place (Appendix B rule 4)"
            ),
        }
        print(f"  naive rendering     WITHHELD ({declaration.name} has no escape to remove)")
    print(f"  reversed            cost {reversed_score['cost_nats_per_residue']:+.4f} nats/residue")

    comparability = comparability_record(declaration, declared["residues_per_scored_token"])
    record: dict[str, Any] = {
        "cohort": cohort_record(scored, band_unit="residues"),
        "unigram_reference": unigram,
        "declared_rendering": declared,
        "symbol_unit": declaration.symbol_unit,
        "context_information_nats": float(context_information),
        "context_information_unit": unit,
        "measured_residues_per_scored_token": float(declared["residues_per_scored_token"]),
        "cross_arm_comparability": comparability,
        "context_information_definition": (
            f"held-out unigram cross-entropy on the scored protein targets minus the "
            f"model's clean cross-entropy on the same targets, in {unit}. One scored "
            f"symbol is {joint_modes.SYMBOL_UNIT_DEFINITIONS[declaration.symbol_unit]}"
        ),
        "controls": {"reversed": reversed_score, "naive_rendering": naive},
    }
    record.update(verdict_record(context_information, args.min_context_information))
    return record


# -------------------------------------------------------------------- text mode


def text_mode(
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    tokenisation: joint_modes.JointTokenisation,
    *,
    vocab_size: int,
) -> dict[str, Any]:
    """Context information on text tokens, with the residue-mass control beside it."""

    scored, reference, overlap = mode_cohorts(args, "text")
    print(f"[text] {len(scored)} scored documents, {len(reference)} held-out reference documents")
    unigram = unigram_record(
        text_target_counts(
            tokenizer, reference.records, vocab_size=vocab_size, max_tokens=args.max_tokens
        ),
        text_target_counts(
            tokenizer, scored.records, vocab_size=vocab_size, max_tokens=args.max_tokens
        ),
        support="the checkpoint's full vocabulary, which is the set of ids a "
        "scored text target can take",
        reference=reference,
        overlap=overlap,
    )

    declaration = tokenisation.declaration
    disjoint = declaration.residue_subspace_disjoint_from_text
    subspace = (
        sorted(int(value) for value in tokenisation.residue_ids.values()) if disjoint else None
    )
    total_nll = 0.0
    scored_tokens = 0
    mass_total = 0.0
    dominated = 0
    for document in scored.records:
        token_ids = text_token_ids(tokenizer, document, max_tokens=args.max_tokens)
        if len(token_ids) < 2:
            raise ValueError("a text record tokenised to fewer than two tokens")
        positions = list(range(1, len(token_ids)))
        nll, mass = score_positions(
            model, token_ids, positions, device=args.device, subspace=subspace
        )
        total_nll += float(nll.sum())
        scored_tokens += len(positions)
        if mass is not None:
            mass_total += float(mass.sum())
            dominated += int((mass > 0.5).sum())
    clean = total_nll / scored_tokens
    context_information = unigram["cross_entropy_nats"] - clean
    print(
        f"  clean {clean:.4f} nats/token against unigram "
        f"{unigram['cross_entropy_nats']:.4f}"
    )

    if disjoint:
        control: dict[str, Any] = {
            "residue_subspace_ids": subspace,
            "mean_residue_probability_mass_per_position": mass_total / scored_tokens,
            "fraction_of_positions_with_residue_mass_above_half": dominated / scored_tokens,
            "n_scored_positions": scored_tokens,
            "uniform_reference_nats": math.log(vocab_size),
            "note": (
                "the mass this checkpoint places on its own residue tokens at every "
                "scored TEXT position. A high value identifies a COLLAPSED text mode "
                "-- the model treating arbitrary prose as the prefix of a protein -- "
                "which a cross-entropy alone cannot distinguish from a merely poor "
                "one. Compare the clean cross-entropy against uniform_reference_nats: "
                "worse than uniform is a broken interface until this control says "
                "otherwise"
            ),
        }
    else:
        control = {
            "verdict": "WITHHELD",
            "reason": (
                f"{declaration.name} declares no residue subspace disjoint from text: "
                "its residues are ordinary single-letter pieces of the text "
                "vocabulary, so a 'residue probability mass' here would be a "
                "statistic about twenty capital letters and would identify nothing"
            ),
        }

    record: dict[str, Any] = {
        "cohort": cohort_record(scored, band_unit="characters"),
        "unigram_reference": unigram,
        "clean_nll_nats_per_scored_token": clean,
        "n_scored_tokens": scored_tokens,
        "max_tokens": int(args.max_tokens),
        "context_information_nats": float(context_information),
        "context_information_unit": "nats per scored token",
        "context_information_definition": (
            "held-out unigram cross-entropy on the scored next-token targets minus "
            "the model's clean cross-entropy on the same targets"
        ),
        "controls": {"residue_subspace_mass": control},
    }
    record.update(verdict_record(context_information, args.min_context_information))
    return record


# ----------------------------------------------------------------------- driver


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="directory of the joint checkpoint to qualify. Required and not an "
        "arm name: a checkpoint that has not passed this stage must not be in the "
        "panel, so there is nothing for a default to point at",
    )
    parser.add_argument(
        "--rendering",
        required=True,
        choices=joint_modes.RENDERING_NAMES,
        help="which declared family's input format this checkpoint takes. The set "
        "is composed by src.transfer.joint_modes, which is the single place either "
        "mode's format is decided (Appendix B rule 12)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", default="bfloat16", choices=tuple(_DTYPES), help="inference dtype; "
        "the log-softmax is taken in float32 regardless"
    )
    parser.add_argument(
        "--modes",
        default="both",
        choices=("text", "protein", "both"),
        help="which modes to qualify. 'both' is the default because D1.d's "
        "admission rule needs both and a checkpoint measured in one is not "
        "qualified",
    )
    parser.add_argument("--sequences", type=int, default=64)
    parser.add_argument(
        "--unigram-sequences",
        type=int,
        default=400,
        help="records the held-out context-free baseline is fitted on, drawn past "
        "the scored cohort in the same permutation and then deduplicated by content",
    )
    parser.add_argument("--protein-min-len", type=int, default=64)
    parser.add_argument("--protein-max-len", type=int, default=246)
    parser.add_argument(
        "--text-min-chars",
        type=int,
        default=800,
        help="floor of the text cohort, in characters. src.transfer.arms.text_cohort's "
        "own default, so the population is the one every other text measurement in "
        "this repository uses",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="scored window for text. A protein rendering is never truncated: the "
        "stage raises instead, because dropping the closing delimiter would silently "
        "change the scored span",
    )
    parser.add_argument(
        "--protein-context",
        default=None,
        help="optional document context the protein block is embedded in, filled "
        "into the family's declared template. Omitted means the bare block, and "
        "whichever was used reaches the artefact",
    )
    parser.add_argument(
        "--cohort-draw-seed",
        type=int,
        default=DEFAULT_CORPUS_DRAW_SEED,
        help="seed for the permutation both draws are windows of; 0 selects the "
        "historical file-order draw, which is a declared choice and not a default "
        "(Appendix B rule 1)",
    )
    parser.add_argument(
        "--min-context-information",
        type=float,
        default=MIN_CONTEXT_INFORMATION_NATS,
        help="the threshold below which a mode is reported unmeasurable on this "
        "cohort. Stage 01's threshold, imported, so that a joint checkpoint is "
        "qualified against the level the panel was qualified against",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    modes = ("protein", "text") if args.modes == "both" else (args.modes,)

    declaration = joint_modes.rendering(args.rendering)
    print(f"[load] {args.checkpoint} as {declaration.name} on {args.device}")
    # Resolving the rendering against this tokenizer is what refuses a
    # checkpoint/family pair. It happens before the weights are read, and long
    # before any cross-entropy exists to be misread.
    resolved, tokenizer = load_tokenizer(args.checkpoint)
    tokenisation = joint_modes.resolve(tokenizer, declaration)
    model, checkpoint_facts = load_model(
        resolved, tokenizer, device=args.device, dtype=args.dtype
    )
    checkpoint_facts["requested_path"] = str(args.checkpoint)
    print(
        f"  {checkpoint_facts['n_layers']}L x {checkpoint_facts['d_model']}d x "
        f"{checkpoint_facts['n_heads']}h, vocab {checkpoint_facts['vocab_size']}; "
        f"rendering resolved to delimiters {tokenisation.start_id}/{tokenisation.end_id}"
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
                "path": "scripts/transfer/21_joint_mode_qualification.py",
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "modules": {
                name: sha256_file(REPO_ROOT / name) for name in PROVENANCE_MODULES
            },
        },
        "checkpoint": checkpoint_facts,
        "rendering": tokenisation.facts(),
        "seeds": {"cohort_draw": int(args.cohort_draw_seed)},
        "thresholds": {
            "minimum_context_information_nats": float(args.min_context_information)
        },
        "estimand": (
            "context information per mode: held-out unigram cross-entropy on the "
            "scored symbols minus the model's clean cross-entropy on the same "
            "symbols. A text symbol is a token; a protein symbol is whatever the "
            "family declares as its symbol unit in src.transfer.joint_modes, so "
            "each mode's record names its own unit, its own support, and -- for "
            "protein -- its measured residues per scored token and whether the "
            "magnitude may be placed beside another arm's at all"
        ),
    }

    modes_record: dict[str, Any] = {}
    if "protein" in modes:
        modes_record["protein"] = protein_mode(args, model, tokenisation)
    if "text" in modes:
        modes_record["text"] = text_mode(
            args,
            model,
            tokenizer,
            tokenisation,
            vocab_size=int(checkpoint_facts["vocab_size"]),
        )
    payload["modes"] = modes_record
    payload["verdicts"] = {name: record["verdict"] for name, record in modes_record.items()}
    payload["verdict_note"] = VERDICT_NOTE
    payload["modes_measured"] = list(modes)

    destination = args.out / "joint_mode_qualification.json"
    write_json(destination, payload)
    print()
    for name, record in modes_record.items():
        line = (
            f"[{name}] context information {record['context_information_nats']:+.4f} "
            f"({record['context_information_unit']})  {record['verdict']}"
        )
        comparability = record.get("cross_arm_comparability")
        if comparability is not None:
            line += (
                f"  {comparability['verdict']} at "
                f"{comparability['measured_residues_per_scored_token']:.3f} residues/token"
            )
        print(line)
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
