#!/usr/bin/env python3
"""Measure which candidate prefix rendering a checkpoint itself prefers.

The arm-rendering audit left nine questions that no file in the repository can
settle, each of them the shape of the ProtGPT3 defect: a checkpoint's official
recipe opens with a token that the published tokenizer config neither adds nor
mentions, and the repository's own comment reasoned from the config's silence
to a rendering the checkpoint was never trained behind. This script settles
each one by measurement on the checkpoint itself: a fixed held-out population
drawn by the repository's own cohort loaders, scored under each candidate
prefix, with the model's own next-token likelihood and, where a decisive
structural signal exists, the distribution it puts at the prefix.

It is a probe with one function per question and a ``--question`` switch, not a
harness. Nothing here is reusable infrastructure and nothing is checked that a
question below does not ask for; a question whose checkpoint does not fit on an
available card is reported as unmeasured by the caller rather than degraded to
a different dtype or a smaller model.

Population. Every protein number uses ``arms.protein_cohort(200, 64, 246,
seed=<--cohort-draw-seed>)`` and every text number uses
``arms.text_cohort(200, 800, seed=<--cohort-draw-seed>)`` truncated to 384
tokens: stage 01's ``--n-seq 200``, ``--res-min/max 64/246``,
``--cohort-draw-seed`` and ``--max-len 384``, so the measured population is the
one these questions bear on. The seed is the flag's and not a literal, so a
question can be re-read at stage 01's draw or at a disjoint window of it rather
than only on one region of the corpus. NLL comes from stage 21's
``score_positions`` (``log_softmax`` in float32 whatever the parameters are
stored in).

Run it with the host's checkpoint roots exported, from the repository root::

    export TRANSFER_MODEL_BASE_DIR=/Data/public/models_R2
    export TRANSFER_TEXT_MODEL_BASE_DIR=/Data/public
    python scripts/transfer/prefix_render_probe.py --question q1 --device cuda:4

Each question writes ``results/transfer/prefix_render_probe/<question>.json``
(ignored by Git) and prints the same record, and both carry the seed the cohort
was drawn under.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer import joint_modes  # noqa: E402
from src.transfer.arms import (  # noqa: E402
    DEFAULT_CORPUS_DRAW_SEED,
    MODEL_ROOT,
    PANEL,
    TEXT_MODEL_BASE,
    arm_spec,
    load_arm,
    load_arm_spec,
    protein_cohort,
    text_cohort,
)
from src.transfer.rita_fitness import native_encode_for_budget  # noqa: E402
from src.transfer.scoring import (  # noqa: E402
    per_sequence_scores,
    aggregate_variant,
    sequence_target_mask,
    target_rule,
)

#: Stage 01's defaults: the population these questions bear on.
N_RECORDS = 200
PROTEIN_BAND = (64, 246)
TEXT_MIN_CHARS = 800
MAX_TOKENS = 384

DEFAULT_OUT = REPO_ROOT / "results/transfer/prefix_render_probe"

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

_STAGE21 = None


def stage21():
    """Stage 21's ``score_positions``, imported the way stage 01 imports it."""

    global _STAGE21
    if _STAGE21 is None:
        path = Path(__file__).resolve().parent / "21_joint_mode_qualification.py"
        spec = importlib.util.spec_from_file_location("stage21_prefix_probe", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["stage21_prefix_probe"] = module
        spec.loader.exec_module(module)
        _STAGE21 = module
    return _STAGE21


# ------------------------------------------------------------------ primitives


def nll(model, ids, positions, *, device) -> np.ndarray:
    """Per-position NLL in nats, from stage 21's scorer."""

    values, _ = stage21().score_positions(model, ids, positions, device=device)
    return values


def row_probs(model, ids, index, *, device) -> torch.Tensor:
    """The distribution the model puts on the token at ``index``, as probabilities."""

    tensor = torch.tensor([list(ids)], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(tensor).logits[0, index - 1].float()
    return torch.log_softmax(logits, dim=-1).exp()


def top_k(probs: torch.Tensor, tokenizer, k: int = 5) -> list[dict]:
    values, indices = probs.topk(k)
    return [
        {
            "token": tokenizer.convert_ids_to_tokens(int(i)),
            "id": int(i),
            "prob": float(v),
        }
        for v, i in zip(values.tolist(), indices.tolist())
    ]


def summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "stderr": (
            float(array.std(ddof=1) / np.sqrt(array.size)) if array.size > 1 else None
        ),
    }


def mean_ce(model, rows, positions_of, *, device) -> tuple[float, list[float]]:
    """Token-weighted mean clean CE and the per-record means over one rendering."""

    sums: list[float] = []
    tokens: list[int] = []
    for ids in rows:
        positions = positions_of(ids)
        values = nll(model, ids, positions, device=device)
        sums.append(float(values.sum()))
        tokens.append(len(positions))
    return sum(sums) / sum(tokens), [s / t for s, t in zip(sums, tokens)]


def encode(tokenizer, parts: tuple) -> list[int]:
    """Ids for a rendering written as a sequence of token ids and literal strings.

    Strings are encoded with ``add_special_tokens=False``: a rendering is exactly
    the tokens it spells, and no postprocessor is allowed to add one.
    """

    ids: list[int] = []
    for part in parts:
        if isinstance(part, int):
            ids.append(part)
        else:
            ids.extend(int(v) for v in tokenizer(part, add_special_tokens=False)["input_ids"])
    return ids


def residue_ids(tokenizer, sequence: str) -> list[int]:
    """One id per residue, refused if the tokenizer merges residues."""

    ids = [int(v) for v in tokenizer(sequence, add_special_tokens=False)["input_ids"]]
    if len(ids) != len(sequence):
        raise ValueError(
            f"a per-residue unit was requested but the tokenizer spelled "
            f"{len(sequence)} residues as {len(ids)} tokens"
        )
    return ids


def load_hf(path: Path, dtype: str, device: str, *, trust_remote_code: bool = False):
    """The checkpoint and its tokenizer, at one dtype, on one device."""

    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        torch_dtype=_DTYPES[dtype],
        trust_remote_code=trust_remote_code,
        device_map={"": device},
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=trust_remote_code)
    return model, tokenizer


def protein_records(seed: int, n: int = N_RECORDS) -> list[str]:
    return list(protein_cohort(n, *PROTEIN_BAND, seed=seed).records)


def text_records(seed: int, n: int = N_RECORDS) -> list[str]:
    return list(text_cohort(n, TEXT_MIN_CHARS, seed=seed).records)


def base_record(question: str, dtype: str, device: str, seed: int, **extra) -> dict:
    record = {
        "question": question,
        "population": {
            "cohort_draw_seed": int(seed),
            "n_records": N_RECORDS,
            "protein_band_residues": list(PROTEIN_BAND),
            "text_min_chars": TEXT_MIN_CHARS,
            "max_tokens": MAX_TOKENS,
        },
        "dtype": dtype,
        "device": device,
    }
    record.update(extra)
    return record


# ---------------------------------------------------------------------- Q1


def q1(args) -> dict:
    """ProGen2: does a leading ``<|bos|>`` cost or save likelihood.

    ``tokenizer.bos_token`` is ``<|endoftext|>`` (id 30) on this tokenizer while
    ``config.bos_token_id`` is 1, which is the vocab's ``<|bos|>``. The
    rendering's BOS is therefore resolved from the config's id, not from the
    tokenizer's special-token mapping, and that difference is reported.
    """

    records = protein_records(args.cohort_draw_seed)
    result = base_record("q1", "float32", args.device, args.cohort_draw_seed, arms={})
    for name in ("progen2-small", "progen2-base", "progen2-medium"):
        arm = load_arm(name, args.device, dtype="float32")
        tokenizer = arm.tokenizer
        bos_id = int(arm.model.config.bos_token_id)
        marker_id = int(tokenizer.convert_tokens_to_ids("1"))
        rows_repo, rows_bos, rows_bos_only = [], [], []
        for sequence in records:
            residues = residue_ids(tokenizer, sequence)
            rows_repo.append([marker_id] + residues)
            rows_bos.append([bos_id, marker_id] + residues)
            rows_bos_only.append([bos_id] + residues)
        ce_repo, per_repo = mean_ce(
            arm.model, rows_repo, lambda ids: list(range(1, len(ids))), device=args.device
        )
        ce_bos, per_bos = mean_ce(
            arm.model, rows_bos, lambda ids: list(range(2, len(ids))), device=args.device
        )
        # The structural rows below show that after <|bos|> the model predicts a
        # residue rather than the direction marker, which makes "<|bos|> then
        # residues" a third candidate rather than a refuted one; it is priced for
        # the same reason ProtGPT3's two-token prefix was.
        ce_bos_only, _ = mean_ce(
            arm.model, rows_bos_only, lambda ids: list(range(1, len(ids))), device=args.device
        )
        deltas = [b - a for a, b in zip(per_repo, per_bos)]
        probs_after_bos = [
            row_probs(arm.model, ids, 1, device=args.device) for ids in rows_bos[:N_RECORDS]
        ]
        marker_mass = float(np.mean([float(p[marker_id]) for p in probs_after_bos]))
        probs_after_marker = [
            row_probs(arm.model, ids, 1, device=args.device) for ids in rows_repo[:N_RECORDS]
        ]
        result["arms"][name] = {
            "tokenizer_bos_token": tokenizer.bos_token,
            "tokenizer_bos_token_id": tokenizer.bos_token_id,
            "config_bos_token_id": bos_id,
            "direction_marker_id": marker_id,
            "residue_ce_nats_of_repo_rendering": ce_repo,
            "residue_ce_nats_of_bos_rendering": ce_bos,
            "residue_ce_nats_of_bos_then_residues_rendering": ce_bos_only,
            "paired_delta_bos_minus_repo": summary(deltas),
            "mass_on_direction_marker_after_bos": marker_mass,
            "top_k_after_bos": top_k(probs_after_bos[0], tokenizer),
            "top_k_after_marker_only": top_k(probs_after_marker[0], tokenizer),
            "mass_on_bos_after_marker_only": float(
                np.mean([float(p[bos_id]) for p in probs_after_marker])
            ),
            "n_residues": int(sum(len(residue_ids(tokenizer, s)) for s in records)),
        }
        del arm
        torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------- Q2


def q2(args) -> dict:
    """RITA-xl: should residue 1 be conditioned on the document-boundary ``<EOS>``."""

    arm = load_arm_spec(arm_spec("rita-xl"), args.device, dtype="float32")
    tokenizer = arm.tokenizer
    eos_id = 2
    records = protein_records(args.cohort_draw_seed)
    rows_raw, rows_prefix, rows_pad = [], [], []
    for sequence in records:
        native = native_encode_for_budget(tokenizer, sequence)
        rows_raw.append(native)
        rows_prefix.append([eos_id] + native)
        # A pad-prefix control, because "a boundary token belongs here" and "any
        # extra context helps" make the same prediction for the shared target set.
        rows_pad.append([1] + native)
    # The shared target set is every residue after the first plus the terminal
    # EOS: the positions the repository's own rendering already scores.
    shared_raw = lambda ids: list(range(1, len(ids)))  # noqa: E731
    shared_prefix = lambda ids: list(range(2, len(ids)))  # noqa: E731
    ce_raw, per_raw = mean_ce(arm.model, rows_raw, shared_raw, device=args.device)
    ce_prefix, per_prefix = mean_ce(arm.model, rows_prefix, shared_prefix, device=args.device)
    ce_pad, _ = mean_ce(arm.model, rows_pad, shared_prefix, device=args.device)
    # Residue 1 itself: available only under the EOS-prefixed rendering.
    first_residue = [float(nll(arm.model, ids, [1], device=args.device)[0]) for ids in rows_prefix]
    raw_residues = lambda ids: list(range(1, len(ids) - 1))  # noqa: E731
    prefix_residues = lambda ids: list(range(1, len(ids)))  # noqa: E731
    ce_raw_res, _ = mean_ce(arm.model, rows_raw, raw_residues, device=args.device)
    ce_prefix_res, _ = mean_ce(arm.model, rows_prefix, prefix_residues, device=args.device)
    probs_prefix0 = [row_probs(arm.model, ids, 1, device=args.device) for ids in rows_prefix[:64]]
    residue_mass = float(
        np.mean([float(p[3:23].sum()) for p in probs_prefix0])
    )
    probs_pad0 = [row_probs(arm.model, ids, 1, device=args.device) for ids in rows_pad[:64]]
    residue_mass_after_pad = float(np.mean([float(p[3:23].sum()) for p in probs_pad0]))
    terminus_mass_raw = float(
        np.mean(
            [
                float(row_probs(arm.model, ids, len(ids) - 1, device=args.device)[eos_id])
                for ids in rows_raw[:64]
            ]
        )
    )
    result = base_record(
        "q2",
        "float32",
        args.device,
        args.cohort_draw_seed,
        arm="rita-xl",
        eos_id=eos_id,
        shared_target_set="residues 2..L and the terminal <EOS>",
        shared_ce_nats_raw=ce_raw,
        shared_ce_nats_eos_prefixed=ce_prefix,
        shared_ce_nats_pad_prefixed_control=ce_pad,
        mass_on_residue_ids_after_eos_prefix=residue_mass,
        mass_on_residue_ids_after_pad_prefix_control=residue_mass_after_pad,
        shared_paired_delta_summary=summary([b - a for a, b in zip(per_raw, per_prefix)]),
        residue_ce_nats_residues_2_to_L_raw=ce_raw_res,
        residue_ce_nats_residues_1_to_L_prefixed=ce_prefix_res,
        first_residue_nll=summary(first_residue),
        mass_on_terminal_eos_raw=terminus_mass_raw,
        top_k_after_prefix=top_k(probs_prefix0[0], tokenizer),
        n_residues=int(sum(len(residue_ids(tokenizer, s)) for s in records)),
    )
    return result


# ---------------------------------------------------------------------- Q3


def q3(args) -> dict:
    """ProteinGLM-7B-CLM: is ``<eos>`` a continuation prompt or a terminator."""

    from src.transfer.proteinglm import (
        PREFIX_IDS,
        PREFIX_LENGTH,
        encode_budget_text,
        residue_token_ids,
    )

    arm = load_arm_spec(arm_spec("proteinglm-7b-clm"), args.device, dtype="float32")
    tokenizer = arm.tokenizer
    records = protein_records(args.cohort_draw_seed)
    rows = []
    for sequence in records:
        text = "<gmask><sop><eos>" + sequence
        rows.append(encode_budget_text(tokenizer, text, max_len=PREFIX_LENGTH + len(sequence)))
    # The repository's rule scores residues 2..L; the first residue's own
    # prediction is the column it drops.
    residues_2_to_L = lambda ids: list(range(PREFIX_LENGTH + 1, len(ids)))  # noqa: E731
    residues_1_to_L = lambda ids: list(range(PREFIX_LENGTH, len(ids)))  # noqa: E731
    ce_2L, _ = mean_ce(arm.model, rows, residues_2_to_L, device=args.device)
    ce_1L, _ = mean_ce(arm.model, rows, residues_1_to_L, device=args.device)
    first_residue = [
        float(nll(arm.model, ids, [PREFIX_LENGTH], device=args.device)[0]) for ids in rows
    ]
    probs = [row_probs(arm.model, ids, PREFIX_LENGTH, device=args.device) for ids in rows[:64]]
    residue_ids_set = list(residue_token_ids(tokenizer).values())
    residue_mass = float(np.mean([float(p[residue_ids_set].sum()) for p in probs]))
    eos_mass = float(np.mean([float(p[PREFIX_IDS[-1]]) for p in probs]))
    result = base_record(
        "q3",
        "float32",
        args.device,
        args.cohort_draw_seed,
        arm="proteinglm-7b-clm",
        prefix_ids=list(PREFIX_IDS),
        residue_ce_nats_residues_2_to_L=ce_2L,
        residue_ce_nats_residues_1_to_L=ce_1L,
        first_residue_nll=summary(first_residue),
        mass_on_residue_ids_after_eos_prefix=residue_mass,
        residue_ids_used_for_mass=residue_ids_set,
        mass_on_eos_at_the_position_that_predicts_residue_1=eos_mass,
        top_k_after_eos_prefix=top_k(probs[0], tokenizer),
        n_residues=int(sum(len(residue_ids(tokenizer, s)) for s in records)),
    )
    return result


# ---------------------------------------------------------------------- Q4


def q4(args) -> dict:
    """Galactica: should anything precede the document or the protein block.

    ``galai`` rewrites position 0 to ``eos_token_id`` (2, ``</s>``) when
    ``new_doc=True``, which defaults to False and appears nowhere here; the
    config also declares ``bos_token_id`` 0 (``<s>``) that nothing uses and the
    loaded tokenizer reports no ``bos_token``. One short forward over eight
    records in each mode, three renderings.
    """

    n = 8
    model, tokenizer = load_hf(
        Path(arm_path("galactica-1.3b")), "float32", args.device
    )
    eos_id, bos_id = 2, 0
    declaration = joint_modes.rendering("galactica")
    start_id = int(tokenizer.convert_tokens_to_ids(declaration.protein_start))
    end_id = int(tokenizer.convert_tokens_to_ids(declaration.protein_end))
    protein = protein_cohort(n, *PROTEIN_BAND, seed=args.cohort_draw_seed).records
    text = text_cohort(n, TEXT_MIN_CHARS, seed=args.cohort_draw_seed).records

    def span_positions(ids: list[int]) -> list[int]:
        mask = sequence_target_mask(
            torch.tensor([ids]),
            torch.ones((1, len(ids)), dtype=torch.long),
            rule="between_boundaries",
            start_token_id=start_id,
            end_token_id=end_id,
        )[0]
        return (torch.nonzero(mask, as_tuple=False).flatten() + 1).tolist()

    protein_rows: dict[str, dict] = {}
    protein_per_record: dict[str, list[float]] = {}
    for label, prefix in (("nothing", None), ("eos_2", eos_id), ("bos_0", bos_id)):
        rows = []
        for sequence in protein:
            block = declaration.render_protein(sequence)
            parts = (prefix, block) if prefix is not None else (block,)
            rows.append(encode(tokenizer, parts))
        ce, per_record = mean_ce(model, rows, span_positions, device=args.device)
        protein_per_record[label] = per_record
        first = [float(nll(model, ids, span_positions(ids)[:1], device=args.device)[0]) for ids in rows]
        protein_rows[label] = {
            "span_ce_nats_per_residue": ce,
            "first_residue_nll": summary(first),
        }
        if prefix is not None:
            probs = row_probs(model, rows[0], 1, device=args.device)
            protein_rows[label]["top_k_position_0"] = top_k(probs, tokenizer)
            protein_rows[label]["mass_on_start_delimiter_position_0"] = float(probs[start_id])
    for label in ("eos_2", "bos_0"):
        protein_rows[label]["paired_delta_vs_nothing"] = summary(
            [
                b - a
                for a, b in zip(protein_per_record["nothing"], protein_per_record[label])
            ]
        )

    # The shared target set is every document token after the first: the same
    # tokens under all three renderings, so the comparison is paired.
    raw_docs = [
        [int(v) for v in tokenizer(d, add_special_tokens=False)["input_ids"]][:MAX_TOKENS]
        for d in text
    ]
    text_rows: dict[str, dict] = {}
    text_per_record: dict[str, list[float]] = {}
    for label, prefix in (("nothing", None), ("eos_2", eos_id), ("bos_0", bos_id)):
        rows, firsts = [], []
        for raw, document in zip(raw_docs, text):
            ids = ([prefix] + raw) if prefix is not None else raw
            rows.append(ids)
            if prefix is not None:
                firsts.append(float(nll(model, ids, [1], device=args.device)[0]))
        sums, tokens = [], []
        for ids, raw in zip(rows, raw_docs):
            pos = list(range(len(ids) - len(raw) + 1, len(ids)))
            values = nll(model, ids, pos, device=args.device)
            sums.append(float(values.sum()))
            tokens.append(len(pos))
        per_record = [s / t for s, t in zip(sums, tokens)]
        text_per_record[label] = per_record
        text_rows[label] = {"document_token_ce_nats": sum(sums) / sum(tokens)}
        if prefix is not None:
            text_rows[label]["first_document_token_nll"] = summary(firsts)
            text_rows[label]["paired_delta_vs_nothing"] = summary(
                [b - a for a, b in zip(text_per_record["nothing"], per_record)]
            )
            probs = row_probs(model, rows[0], 1, device=args.device)
            text_rows[label]["top_k_position_0"] = top_k(probs, tokenizer)
    return base_record(
        "q4",
        "float32",
        args.device,
        args.cohort_draw_seed,
        arm="galactica-1.3b",
        n_protein_records=n,
        n_text_records=n,
        protein_mode=protein_rows,
        text_mode=text_rows,
    )


# ---------------------------------------------------------------------- Q5


def q5(args) -> dict:
    """InstructProtein text mode: is the tokenizer's auto-added BOS correct.

    ``joint_modes.encode`` calls the tokenizer with special tokens enabled, so the
    same BOS enters this family's protein rendering too; the protein block is
    priced here beside the text mode because a repair to the shared encoder has
    to know whether the BOS helps the mode that matters.
    """

    model, tokenizer = load_hf(Path(arm_path("InstructProtein")), "float32", args.device)
    documents = text_records(args.cohort_draw_seed)
    rows_auto, rows_none = [], []
    for document in documents:
        rows_auto.append(
            [int(v) for v in tokenizer(document)["input_ids"]][: MAX_TOKENS + 1]
        )
        rows_none.append(
            [int(v) for v in tokenizer(document, add_special_tokens=False)["input_ids"]][:MAX_TOKENS]
        )
    ce_auto, _ = mean_ce(model, rows_auto, lambda ids: list(range(1, len(ids))), device=args.device)
    ce_none, _ = mean_ce(model, rows_none, lambda ids: list(range(1, len(ids))), device=args.device)
    first = [float(nll(model, ids, [1], device=args.device)[0]) for ids in rows_auto]
    probs = row_probs(model, rows_auto[0], 1, device=args.device)
    declaration = joint_modes.rendering("instructprotein")
    start_id = int(tokenizer.convert_tokens_to_ids(declaration.protein_start))
    end_id = int(tokenizer.convert_tokens_to_ids(declaration.protein_end))
    protein = protein_cohort(8, *PROTEIN_BAND, seed=args.cohort_draw_seed).records

    def span_positions(ids: list[int]) -> list[int]:
        mask = sequence_target_mask(
            torch.tensor([ids]),
            torch.ones((1, len(ids)), dtype=torch.long),
            rule="between_boundaries",
            start_token_id=start_id,
            end_token_id=end_id,
        )[0]
        return (torch.nonzero(mask, as_tuple=False).flatten() + 1).tolist()

    protein_rows: dict[str, dict] = {}
    for label, special in (("auto_bos", True), ("no_bos", False)):
        rows = [
            [int(v) for v in tokenizer(declaration.render_protein(s), add_special_tokens=special)["input_ids"]]
            for s in protein
        ]
        ce, _ = mean_ce(model, rows, span_positions, device=args.device)
        first_residue = [
            float(nll(model, ids, span_positions(ids)[:1], device=args.device)[0]) for ids in rows
        ]
        protein_rows[label] = {
            "span_ce_nats_per_residue": ce,
            "first_residue_nll": summary(first_residue),
        }
        if special:
            first_probs = row_probs(model, rows[0], 1, device=args.device)
            protein_rows[label]["top_k_position_0"] = top_k(first_probs, tokenizer)
            protein_rows[label]["mass_on_protein_start_position_0"] = float(first_probs[start_id])
    return base_record(
        "q5",
        "float32",
        args.device,
        args.cohort_draw_seed,
        arm="InstructProtein",
        n_protein_records=len(protein),
        tokenizer_add_bos_token=bool(getattr(tokenizer, "add_bos_token", False)),
        bos_token=tokenizer.bos_token,
        bos_token_id=tokenizer.bos_token_id,
        auto_bos_ce_nats=ce_auto,
        no_bos_ce_nats=ce_none,
        auto_bos_scored_tokens=int(sum(len(r) - 1 for r in rows_auto)),
        no_bos_scored_tokens=int(sum(len(r) - 1 for r in rows_none)),
        first_document_token_nll_under_auto_bos=summary(first),
        top_k_position_0_after_auto_bos=top_k(probs, tokenizer),
        protein_mode=protein_rows,
    )


# ---------------------------------------------------------------------- Q6


def q6(args) -> dict:
    """ByGPT5: is the declared ``decoder_start`` conditioning prefix real."""

    model, tokenizer = load_hf(
        Path(text_model_path("bygpt5-small-en")),
        "float32",
        args.device,
        trust_remote_code=True,
    )
    documents = text_records(args.cohort_draw_seed)
    raw = [
        [int(v) for v in tokenizer(d, add_special_tokens=False)["input_ids"]][:MAX_TOKENS]
        for d in documents
    ]
    rows = {
        "bytes": raw,
        "decoder_start_0_then_bytes": [[0] + ids for ids in raw],
        "bytes_then_eos_1": [ids + [1] for ids in raw],
    }

    def positions(ids: list[int], width: int) -> list[int]:
        return list(range(len(ids) - width + 1, len(ids)))

    payload = {}
    for label, rendered in rows.items():
        sums, tokens = 0.0, 0
        for ids, width in zip(rendered, [len(r) for r in raw]):
            pos = positions(ids, width)
            sums += float(nll(model, ids, pos, device=args.device).sum())
            tokens += len(pos)
        payload[f"{label}_ce_nats_per_token"] = sums / tokens
        if label == "decoder_start_0_then_bytes":
            # The first document byte is predicted from the prefix here, and the
            # unprefixed rendering has no position that predicts it at all.
            payload[f"{label}_first_byte_nll"] = summary(
                [
                    float(nll(model, ids, [1], device=args.device)[0])
                    for ids in rendered
                ]
            )
        if label == "bytes_then_eos_1":
            # The first byte is unconditioned in this rendering too; what the
            # trailing <eos> adds is a scored terminator target.
            payload[f"{label}_trailing_eos_nll"] = summary(
                [
                    float(nll(model, ids, [len(ids) - 1], device=args.device)[0])
                    for ids in rendered
                ]
            )
    probs = row_probs(model, rows["decoder_start_0_then_bytes"][0], 1, device=args.device)
    payload["top_k_position_0_after_decoder_start"] = top_k(probs, tokenizer)
    return base_record(
        "q6",
        "float32",
        args.device,
        args.cohort_draw_seed,
        arm="bygpt5-small-en",
        decoder_start_token_id=int(model.config.decoder_start_token_id),
        eos_token_id=int(model.config.eos_token_id),
        scored_window_tokens=MAX_TOKENS,
        **payload,
    )


# ---------------------------------------------------------------------- Q7


def q7(args) -> dict:
    """Qwen2.5 and Qwen3-8B-Base: should a document be prefixed with 151643."""

    separator = 151643
    documents = text_records(args.cohort_draw_seed)
    result = base_record(
        "q7", args.dtype, args.device, args.cohort_draw_seed, separator_id=separator, arms={}
    )
    for name in args.q7_arms:
        if name in PANEL:
            arm = load_arm(name, args.device, dtype=args.dtype)
        else:
            arm = load_arm_spec(arm_spec(name), args.device, dtype=args.dtype)
        model, tokenizer = arm.model, arm.tokenizer
        del arm
        raw = [
            [int(v) for v in tokenizer(d, add_special_tokens=False)["input_ids"]][:MAX_TOKENS]
            for d in documents
        ]
        plain, prefixed = [], []
        for ids in raw:
            plain.append(ids)
            prefixed.append([separator] + ids)
        positions = lambda ids, width: list(range(len(ids) - width + 1, len(ids)))  # noqa: E731
        widths = [len(ids) for ids in raw]
        flat = sum(
            float(nll(model, ids, positions(ids, w), device=args.device).sum())
            for ids, w in zip(plain, widths)
        )
        pre = sum(
            float(nll(model, ids, positions(ids, w), device=args.device).sum())
            for ids, w in zip(prefixed, widths)
        )
        tokens = int(sum(widths)) - len(widths)
        first = [
            float(nll(model, ids, [1], device=args.device)[0]) for ids in prefixed
        ]
        probs = row_probs(model, prefixed[0], 1, device=args.device)
        result["arms"][name] = {
            "tokenizer_bos_token": tokenizer.bos_token,
            "tokenizer_bos_token_id": tokenizer.bos_token_id,
            "auto_bos_added_by_tokenizer": bool(
                tokenizer("x", add_special_tokens=True)["input_ids"][0] == separator
            ),
            "no_prefix_ce_nats": flat / tokens,
            "prefixed_ce_nats": pre / tokens,
            "delta_prefixed_minus_none": pre / tokens - flat / tokens,
            "first_token_nll_under_prefix": summary(first),
            "top_k_position_0_after_prefix": top_k(probs, tokenizer),
            "scored_tokens": tokens,
        }
        del model
        torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------- Q8


def q8(args) -> dict:
    """Llama-3.2-3B: bound the BOS the tokenizer adds silently."""

    arm = load_arm("llama-3.2-3b", args.device, dtype=args.dtype)
    tokenizer = arm.tokenizer
    documents = text_records(args.cohort_draw_seed)
    with_bos, without = [], []
    for document in documents:
        with_bos.append([int(v) for v in tokenizer(document)["input_ids"]][:MAX_TOKENS])
        without.append(
            [int(v) for v in tokenizer(document, add_special_tokens=False)["input_ids"]][:MAX_TOKENS]
        )
    ce_with, _ = mean_ce(arm.model, with_bos, lambda ids: list(range(1, len(ids))), device=args.device)
    ce_without, _ = mean_ce(arm.model, without, lambda ids: list(range(1, len(ids))), device=args.device)
    first = [float(nll(arm.model, ids, [1], device=args.device)[0]) for ids in with_bos]
    del arm
    torch.cuda.empty_cache()
    return base_record(
        "q8",
        args.dtype,
        args.device,
        args.cohort_draw_seed,
        arm="llama-3.2-3b",
        bos_token_id=int(tokenizer.bos_token_id),
        ce_nats_with_tokenizer_bos=ce_with,
        ce_nats_without_bos=ce_without,
        delta_with_minus_without=ce_with - ce_without,
        first_document_token_nll_under_bos=summary(first),
        scored_window_tokens=MAX_TOKENS,
    )


# ---------------------------------------------------------------------- Q9


def q9(args) -> dict:
    """GPT-2: the size of the bfloat16-versus-float32 scoring delta."""

    name = args.q9_arm
    arm32 = load_arm(name, args.device, dtype="float32")
    arm16 = load_arm(name, args.device, dtype=args.dtype)
    tokenizer = arm32.tokenizer
    if arm32.spec.modality == "protein":
        cohort = protein_cohort(N_RECORDS, *PROTEIN_BAND, seed=args.cohort_draw_seed)
    else:
        cohort = text_cohort(N_RECORDS, TEXT_MIN_CHARS, seed=args.cohort_draw_seed)
    texts = cohort.input_strings(arm32)
    rule = target_rule(arm32.spec.input_format)
    rows: list[dict] = []
    for text in texts:
        ids = [int(v) for v in tokenizer(text, return_tensors=None)["input_ids"]][:MAX_TOKENS]
        if len(ids) < 2:
            raise ValueError(f"{name}: a record tokenised to fewer than two tokens")
        mask = sequence_target_mask(
            torch.tensor([ids]), torch.ones((1, len(ids)), dtype=torch.long), rule=rule
        )
        if not bool(mask.any()):
            raise ValueError(f"{name}: a record has no scored target under rule {rule!r}")
        tensors = torch.tensor([ids], dtype=torch.long, device=args.device)
        with torch.no_grad():
            logits32 = arm32.model(tensors).logits.float()
            logits16 = arm16.model(tensors).logits.float()
        rows.extend(
            per_sequence_scores(logits16, logits32, tensors, mask)
        )
        del logits32, logits16, tensors
    aggregate = aggregate_variant(rows)
    del arm16
    torch.cuda.empty_cache()
    return base_record(
        "q9",
        f"{args.dtype}_versus_float32",
        args.device,
        args.cohort_draw_seed,
        arm=name,
        target_rule=rule,
        **aggregate,
    )


# --------------------------------------------------------------------- driver


def arm_path(directory: str) -> Path:
    return MODEL_ROOT / directory


def text_model_path(directory: str) -> Path:
    return TEXT_MODEL_BASE / directory


QUESTIONS = {
    "q1": q1,
    "q2": q2,
    "q3": q3,
    "q4": q4,
    "q5": q5,
    "q6": q6,
    "q7": q7,
    "q8": q8,
    "q9": q9,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True, choices=tuple(QUESTIONS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--cohort-draw-seed",
        type=int,
        default=DEFAULT_CORPUS_DRAW_SEED,
        help="the seed every question's protein and text cohort is drawn under, "
        "so a question can be re-read at stage 01's draw or at a disjoint window "
        "of the same permutation",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=tuple(_DTYPES))
    parser.add_argument("--q7-arms", nargs="+", default=["qwen2.5-0.5b"])
    parser.add_argument("--q9-arm", default="gpt2-large")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    record = QUESTIONS[args.question](args)
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / f"{args.question}.json"
    destination.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
