"""Synthetic CPU tests for native residue extraction.

Token IDs, hashes, inventory digests and the commit pin are fixtures, not
ProGen2/7B parity or scientific QA labels. No pretrained weights are loaded.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import hashlib
import os
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.transfer import latent_bridge as lb
from src.transfer.residue_extraction import ResidueSequence, extract_residue_batch
from src.transfer.residue_prefix import (
    IN_MEMORY_SCHEMA,
    MAX_TENSOR_BYTES_CEILING,
    ResidueBatch,
    ResidueProvenance,
    ResidueRecord,
    mean_from_residues,
)
from src.transfer.sequence_description import SequenceDescriptionRecord

SYN_INVENTORY = "a" * 64
SYN_COMMIT = "b" * 40
ALPHABET = "1ACDEFGHIKLMNPQRSTVWY"


class CompleteAlphabetTokenizer:
    pad_token_id: int | None = 0
    unk_token_id = 2

    def __init__(self) -> None:
        self._table = {symbol: index + 3 for index, symbol in enumerate(ALPHABET)}

    def convert_tokens_to_ids(self, symbol: str) -> int:
        return self._table.get(symbol, self.unk_token_id)

    def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
        return {"input_ids": [self.convert_tokens_to_ids(symbol) for symbol in text]}


class TinyBlock(nn.Module):
    def __init__(self, width: int, scale: float) -> None:
        nn.Module.__init__(self)  # pyright: ignore[reportUnknownMemberType]
        self.proj = nn.Linear(width, width, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(width) * scale)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden)


class TinyTransformer(nn.Module):
    def __init__(self, width: int) -> None:
        nn.Module.__init__(self)  # pyright: ignore[reportUnknownMemberType]
        self.h = nn.ModuleList([TinyBlock(width, 1.0), TinyBlock(width, 2.0)])


class TinyDonorModel(nn.Module):
    def __init__(self, vocab: int, width: int) -> None:
        nn.Module.__init__(self)  # pyright: ignore[reportUnknownMemberType]
        self.embed = nn.Embedding(vocab, width)
        self.transformer = TinyTransformer(width)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        hidden = self.embed(input_ids)
        for block in self.transformer.h:
            hidden = block(hidden)
        return hidden


class BoomBlock(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("forward exploded")


def _legacy_record(sequence: str = "ACDE") -> lb.PreparedRecord:
    source = SequenceDescriptionRecord(
        accession="P1",
        sequence=sequence,
        length=len(sequence),
        name="n",
        function_text="f",
        description_raw="raw",
        description_masked="masked",
        masked_terms=(),
        ec=("1.1.1.1",),
        go=("GO:1",),
        go_propagated=("GO:1",),
        pfam=(),
        cath=(),
        dup_group=1,
        family_group="famA",
        split="fit",
    )
    return lb.filter_records([source])[0][0]


def _sha(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def _ids(tokenizer: CompleteAlphabetTokenizer, sequence: str) -> tuple[int, ...]:
    return tuple(lb.native_donor_token_ids(tokenizer, sequence, max_tokens=32, max_residues=16))


def _record(accession: str, sequence: str, tokenizer: CompleteAlphabetTokenizer) -> ResidueRecord:
    return ResidueRecord(
        accession=accession,
        sequence_sha256=_sha(sequence),
        residue_count=len(sequence),
        native_token_ids=_ids(tokenizer, sequence),
    )


def _provenance(*, hidden_width: int, block_index: int = 0, capture_dtype: str = "float32") -> ResidueProvenance:
    tokenizer = CompleteAlphabetTokenizer()
    return ResidueProvenance(
        schema=IN_MEMORY_SCHEMA,
        donor_inventory_sha256=SYN_INVENTORY,
        code_commit=SYN_COMMIT,
        block_index=block_index,
        block_index_semantics=lb.BLOCK_INDEX_SEMANTICS,
        capture_dtype=capture_dtype,
        hidden_width=hidden_width,
        marker_id=int(tokenizer.convert_tokens_to_ids("1")),
        pad_id=0,
    )


def _donor(width: int = 4) -> tuple[lb.DonorHandle, CompleteAlphabetTokenizer, TinyDonorModel]:
    tokenizer = CompleteAlphabetTokenizer()
    model = TinyDonorModel(vocab=32, width=width)
    lb.freeze_module(model)
    donor = lb.DonorHandle(
        name="synthetic-donor",
        model=model,
        tokenizer=tokenizer,
        n_layer=2,
        d_model=width,
        kind="synthetic",
        checkpoint_digest={"note": "unauthenticated synthetic fixture"},
    )
    return donor, tokenizer, model


def _planned_bytes(*, batch: int, time: int, hidden_width: int, dtype: torch.dtype) -> int:
    return batch * time * (hidden_width * int(torch.empty((), dtype=dtype).element_size()) + 8 + 1 + 1)


def _extract(
    donor: Any,
    sequences: tuple[ResidueSequence, ...],
    records: tuple[ResidueRecord, ...],
    provenance: ResidueProvenance,
    *,
    max_tensor_bytes: int,
    max_residues: int = 16,
    max_tokens: int = 32,
    device: str | torch.device = "cpu",
    check: Callable[[], None] | None = None,
) -> ResidueBatch:
    return extract_residue_batch(
        donor,
        sequences,
        expected_records=records,
        expected_provenance=provenance,
        device=device,
        max_residues=max_residues,
        max_tokens=max_tokens,
        max_tensor_bytes=max_tensor_bytes,
        check=check if check is not None else (lambda: None),
    )


class NativeTokenHelperTests(unittest.TestCase):
    def test_helper_matches_legacy_wrapper_and_refuses_length_mismatch(self) -> None:
        tokenizer = CompleteAlphabetTokenizer()
        record = _legacy_record("ACDE")
        helper = lb.native_donor_token_ids(
            tokenizer, record.sequence, max_tokens=32, max_residues=10,
        )
        wrapper = lb.donor_token_ids(tokenizer, record, max_tokens=32, max_residues=10)
        self.assertEqual(helper, wrapper)
        self.assertEqual(len(helper), record.length + 1)
        mismatched = replace(record, length=record.length + 1)
        with self.assertRaisesRegex(ValueError, "matching length"):
            lb.donor_token_ids(tokenizer, mismatched, max_tokens=32, max_residues=10)
        self.assertEqual(
            lb.native_donor_token_ids(tokenizer, "ACDE", max_tokens=32, max_residues=10),
            helper,
        )


class ResidueExtractionTests(unittest.TestCase):
    def test_unequal_lengths_geometry_identity_and_mean(self) -> None:
        width = 4
        donor, tokenizer, model = _donor(width)
        sequences = (
            ResidueSequence("synthetic-A", "AC"),
            ResidueSequence("synthetic-B", "DEF"),
        )
        records = (
            _record("synthetic-A", "AC", tokenizer),
            _record("synthetic-B", "DEF", tokenizer),
        )
        provenance = _provenance(hidden_width=width)
        planned = _planned_bytes(batch=2, time=4, hidden_width=width, dtype=torch.float32)
        forwards = {"n": 0}

        def count(_module: nn.Module, _inputs: tuple[Any, ...]) -> None:
            forwards["n"] += 1

        hook = model.register_forward_pre_hook(count)
        try:
            batch = _extract(donor, sequences, records, provenance, max_tensor_bytes=planned)
        finally:
            hook.remove()
        self.assertEqual(forwards["n"], 1)
        self.assertEqual(batch.records, records)
        self.assertEqual(batch.provenance, provenance)
        self.assertEqual(tuple(batch.hidden.shape), (2, 4, width))
        self.assertEqual(batch.hidden.dtype, torch.float32)
        self.assertFalse(batch.hidden.requires_grad)
        self.assertIsNone(batch.hidden.grad_fn)
        marker = provenance.marker_id
        pad = provenance.pad_id
        self.assertEqual(int(batch.input_ids[0, 0]), marker)
        self.assertEqual(int(batch.input_ids[1, 0]), marker)
        self.assertEqual(tuple(int(v) for v in batch.input_ids[0, :3]), records[0].native_token_ids)
        self.assertEqual(tuple(int(v) for v in batch.input_ids[1, :4]), records[1].native_token_ids)
        self.assertEqual(int(batch.input_ids[0, 3]), pad)
        self.assertTrue(torch.equal(batch.attention_mask[0], torch.tensor([True, True, True, False])))
        self.assertTrue(torch.equal(batch.attention_mask[1], torch.tensor([True, True, True, True])))
        self.assertTrue(torch.equal(batch.content_mask[0], torch.tensor([False, True, True, False])))
        self.assertTrue(torch.equal(batch.content_mask[1], torch.tensor([False, True, True, True])))
        singles = [
            _extract(
                donor,
                (item,),
                (record,),
                provenance,
                max_tensor_bytes=_planned_bytes(
                    batch=1, time=record.residue_count + 1, hidden_width=width, dtype=torch.float32
                ),
            )
            for item, record in zip(sequences, records, strict=True)
        ]
        self.assertTrue(torch.equal(batch.hidden[0, :3], singles[0].hidden[0]))
        self.assertTrue(torch.equal(batch.hidden[1, :4], singles[1].hidden[0]))
        pooled = mean_from_residues(
            batch,
            expected_records=records,
            expected_provenance=provenance,
            max_tensor_bytes=planned,
        )
        self.assertTrue(torch.equal(pooled, lb.mean_content(batch.hidden.float(), batch.content_mask)))
        other = lb.capture_block_output(
            donor, layer=1, input_ids=batch.input_ids, attention_mask=batch.attention_mask.long()
        ).detach()
        same = lb.capture_block_output(
            donor, layer=0, input_ids=batch.input_ids, attention_mask=batch.attention_mask.long()
        ).detach()
        self.assertTrue(torch.equal(batch.hidden, same))
        self.assertFalse(torch.equal(batch.hidden, other))

    def test_grouped_refusals_and_cap_before_forward(self) -> None:
        width = 4
        donor, tokenizer, model = _donor(width)
        sequences = (
            ResidueSequence("synthetic-A", "AC"),
            ResidueSequence("synthetic-B", "DEF"),
        )
        records = (
            _record("synthetic-A", "AC", tokenizer),
            _record("synthetic-B", "DEF", tokenizer),
        )
        provenance = _provenance(hidden_width=width)
        planned = _planned_bytes(batch=2, time=4, hidden_width=width, dtype=torch.float32)
        forwards = {"n": 0}

        def count(_module: nn.Module, _inputs: tuple[Any, ...]) -> None:
            forwards["n"] += 1

        hook = model.register_forward_pre_hook(count)
        self.addCleanup(hook.remove)
        _extract(donor, sequences, records, provenance, max_tensor_bytes=planned)
        self.assertEqual(forwards["n"], 1)
        forwards["n"] = 0
        with self.assertRaisesRegex(ValueError, "refusing to allocate or forward"):
            _extract(donor, sequences, records, provenance, max_tensor_bytes=planned - 1)
        self.assertEqual(forwards["n"], 0)

        class TruncatingTokenizer(CompleteAlphabetTokenizer):
            def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
                ids = super().__call__(text)["input_ids"]
                return {"input_ids": ids[:-1]}

        class UnkTokenizer(CompleteAlphabetTokenizer):
            def convert_tokens_to_ids(self, symbol: str) -> int:
                if symbol == "A":
                    return self.unk_token_id
                return super().convert_tokens_to_ids(symbol)

        bad_alpha = ResidueSequence("synthetic-X", "ACX")
        bad_alpha_record = ResidueRecord(
            accession="synthetic-X",
            sequence_sha256=_sha("ACX"),
            residue_count=3,
            native_token_ids=(provenance.marker_id, 10, 11, 12),
        )
        long_seq = ResidueSequence("synthetic-L", "ACDE")
        long_record = _record("synthetic-L", "ACDE", tokenizer)
        swapped = (records[1], records[0])
        train_donor, _, train_model = _donor(width)
        train_model.train()
        grad_donor, _, grad_model = _donor(width)
        next(grad_model.parameters()).requires_grad_(True)
        wide_prov = _provenance(hidden_width=width + 1)
        layer_prov = _provenance(hidden_width=width, block_index=5)
        pad_tok = CompleteAlphabetTokenizer()
        pad_tok.pad_token_id = 7
        pad_donor = lb.DonorHandle(
            name="synthetic-donor",
            model=donor.model,
            tokenizer=pad_tok,
            n_layer=2,
            d_model=width,
            kind="synthetic",
            checkpoint_digest={"note": "unauthenticated synthetic fixture"},
        )
        trunc_donor = lb.DonorHandle(
            name="synthetic-donor",
            model=donor.model,
            tokenizer=TruncatingTokenizer(),
            n_layer=2,
            d_model=width,
            kind="synthetic",
            checkpoint_digest={"note": "unauthenticated synthetic fixture"},
        )
        unk_donor = lb.DonorHandle(
            name="synthetic-donor",
            model=donor.model,
            tokenizer=UnkTokenizer(),
            n_layer=2,
            d_model=width,
            kind="synthetic",
            checkpoint_digest={"note": "unauthenticated synthetic fixture"},
        )
        true_cap: Any = True

        def boom() -> None:
            raise RuntimeError("cancelled")

        cases: tuple[tuple[str, Callable[[], ResidueBatch]], ...] = (
            ("bad alphabet", lambda: _extract(
                donor, (bad_alpha,), (bad_alpha_record,), provenance, max_tensor_bytes=10_000
            )),
            ("overlength", lambda: _extract(
                donor, (long_seq,), (long_record,), provenance, max_tensor_bytes=10_000, max_residues=3
            )),
            ("unk", lambda: _extract(
                unk_donor, sequences[:1], records[:1], provenance, max_tensor_bytes=10_000
            )),
            ("token mismatch", lambda: _extract(
                trunc_donor, sequences[:1], records[:1], provenance, max_tensor_bytes=10_000
            )),
            ("train mode", lambda: _extract(
                train_donor, sequences, records, provenance, max_tensor_bytes=planned
            )),
            ("requires grad", lambda: _extract(
                grad_donor, sequences, records, provenance, max_tensor_bytes=planned
            )),
            ("order", lambda: _extract(donor, sequences, swapped, provenance, max_tensor_bytes=planned)),
            ("bool cap", lambda: _extract(
                donor, sequences, records, provenance, max_tensor_bytes=true_cap
            )),
            ("zero tokens", lambda: _extract(
                donor, sequences, records, provenance, max_tensor_bytes=planned, max_tokens=0
            )),
            ("ceiling", lambda: _extract(
                donor, sequences, records, provenance, max_tensor_bytes=MAX_TENSOR_BYTES_CEILING + 1
            )),
            ("width", lambda: _extract(donor, sequences, records, wide_prov, max_tensor_bytes=planned)),
            ("block bounds", lambda: _extract(
                donor, sequences, records, layer_prov, max_tensor_bytes=planned
            )),
            ("declared pad", lambda: _extract(
                pad_donor, sequences, records, provenance, max_tensor_bytes=planned
            )),
            ("wrong device", lambda: _extract(
                donor, sequences, records, provenance, max_tensor_bytes=planned, device="cuda"
            )),
            ("cancel", lambda: _extract(
                donor, sequences, records, provenance, max_tensor_bytes=planned, check=boom
            )),
        )
        before = forwards["n"]
        for name, factory in cases:
            with self.subTest(name):
                with self.assertRaises((TypeError, ValueError, RuntimeError)):
                    factory()
        self.assertEqual(forwards["n"], before)
        dtype_prov = _provenance(hidden_width=width, capture_dtype="bfloat16")
        with self.assertRaises(ValueError):
            _extract(donor, sequences, records, dtype_prov, max_tensor_bytes=planned)
        self.assertEqual(forwards["n"], before + 1)

    def test_explicit_execution_context_and_post_forward_cancellation(self) -> None:
        donor, tokenizer, model = _donor()
        sequences = (ResidueSequence("synthetic-A", "AC"),)
        records = (_record("synthetic-A", "AC", tokenizer),)
        provenance = _provenance(hidden_width=4)
        cap = _planned_bytes(batch=1, time=3, hidden_width=4, dtype=torch.float32)
        forwards = {"n": 0}

        def count(_module: nn.Module, _args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            self.assertEqual(kwargs["attention_mask"].dtype, torch.bool)
            forwards["n"] += 1

        pre_hook = model.register_forward_pre_hook(count, with_kwargs=True)
        self.addCleanup(pre_hook.remove)
        dropout = nn.Dropout(p=1.0)
        block = donor.blocks()[0]
        block.add_module("injected_dropout", dropout)

        def drop(_module: nn.Module, _args: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
            return dropout(output)

        drop_hook = block.register_forward_hook(drop)
        self.addCleanup(drop_hook.remove)
        self.assertFalse(model.training)
        self.assertTrue(dropout.training)
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        with self.assertRaisesRegex(ValueError, "train mode"):
            _extract(donor, sequences, records, provenance, max_tensor_bytes=cap)
        self.assertEqual(forwards["n"], 0)

        model.eval()
        # Meta safely exposes implicit default-device allocations without a GPU.
        with torch.device("meta"):
            batch = _extract(donor, sequences, records, provenance, max_tensor_bytes=cap)
            with patch("src.transfer.residue_extraction.pad_rows", side_effect=AssertionError("padded before cap")):
                with self.assertRaisesRegex(ValueError, "refusing to allocate or forward"):
                    _extract(donor, sequences, records, provenance, max_tensor_bytes=cap - 1)
        self.assertEqual(str(batch.hidden.device), "cpu")
        self.assertEqual(str(batch.input_ids.device), "cpu")
        self.assertEqual(batch.attention_mask.dtype, torch.bool)
        self.assertEqual(forwards["n"], 1)
        self.assertGreater(int(torch.count_nonzero(batch.hidden)), 0)

        def cancel_after_forward() -> None:
            if forwards["n"] > 1:
                raise RuntimeError("cancelled after forward")

        with patch("src.transfer.residue_extraction.content_mask_from_ids", side_effect=AssertionError("processed after cancellation")):
            with self.assertRaisesRegex(RuntimeError, "cancelled after forward"):
                _extract(donor, sequences, records, provenance, max_tensor_bytes=cap, check=cancel_after_forward)
        self.assertEqual(forwards["n"], 2)
        self.assertEqual(len(getattr(block, "_forward_hooks")), 1)  # Only the test's dropout hook remains.

    def test_hook_removed_when_forward_raises(self) -> None:
        width = 4
        donor, tokenizer, _model = _donor(width)
        sequences = (ResidueSequence("synthetic-A", "AC"),)
        records = (_record("synthetic-A", "AC", tokenizer),)
        provenance = _provenance(hidden_width=width)
        donor.blocks()[0] = BoomBlock().eval()
        with self.assertRaisesRegex(RuntimeError, "forward exploded"):
            _extract(
                donor,
                sequences,
                records,
                provenance,
                max_tensor_bytes=_planned_bytes(batch=1, time=3, hidden_width=width, dtype=torch.float32),
            )
        remaining = list(getattr(donor.blocks()[0], "_forward_hooks"))
        self.assertEqual(remaining, [])


if __name__ == "__main__":
    unittest.main()
