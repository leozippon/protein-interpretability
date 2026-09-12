"""Synthetic CPU tests for the in-memory residue-to-query prefix contract.

Token IDs, sequence hashes, inventory digests and the commit pin are synthetic
fixtures, not native ProGen2/Qwen 7B parity or scientific QA labels.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.transfer import latent_bridge as lb
from src.transfer import residue_prefix as rp
from src.transfer.residue_prefix import (
    IN_MEMORY_SCHEMA,
    MAX_TENSOR_BYTES_CEILING,
    QueryPrefix,
    ResidueBatch,
    ResidueProvenance,
    ResidueRecord,
    mean_from_residues,
    query_prefix_parameter_count,
    validate_residue_batch,
)

FP32_RTOL = 1e-5
FP32_ATOL = 1e-5
SYN_INVENTORY = "a" * 64
SYN_COMMIT = "b" * 40
SYN_HASH_A = "c" * 64
SYN_HASH_B = "d" * 64


def _record(
    *,
    accession: str,
    residue_count: int,
    native_token_ids: tuple[int, ...],
    sequence_sha256: str,
) -> ResidueRecord:
    return ResidueRecord(
        accession=accession,
        sequence_sha256=sequence_sha256,
        residue_count=residue_count,
        native_token_ids=native_token_ids,
    )


def _provenance(
    *,
    hidden_width: int,
    marker_id: int = 1,
    pad_id: int = 0,
    capture_dtype: str = "float32",
    block_index: int = 0,
) -> ResidueProvenance:
    return ResidueProvenance(
        schema=IN_MEMORY_SCHEMA,
        donor_inventory_sha256=SYN_INVENTORY,
        code_commit=SYN_COMMIT,
        block_index=block_index,
        block_index_semantics=lb.BLOCK_INDEX_SEMANTICS,
        capture_dtype=capture_dtype,
        hidden_width=hidden_width,
        marker_id=marker_id,
        pad_id=pad_id,
    )


def _geometry(
    records: tuple[ResidueRecord, ...],
    *,
    time: int,
    hidden_width: int,
    marker_id: int,
    pad_id: int,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = len(records)
    hidden = torch.randn(batch, time, hidden_width, dtype=dtype, generator=generator)
    input_ids = torch.full((batch, time), pad_id, dtype=torch.long)
    attention = torch.zeros(batch, time, dtype=torch.bool)
    content = torch.zeros(batch, time, dtype=torch.bool)
    for row, record in enumerate(records):
        attended = record.residue_count + 1
        input_ids[row, :attended] = torch.tensor(record.native_token_ids, dtype=torch.long)
        attention[row, :attended] = True
        content[row, 1:attended] = True
        if record.native_token_ids[0] != marker_id:
            raise AssertionError("fixture marker mismatch")
    return hidden, input_ids, attention, content


def _batch_from(
    records: tuple[ResidueRecord, ...],
    provenance: ResidueProvenance,
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    attention: torch.Tensor,
    content: torch.Tensor,
) -> ResidueBatch:
    return ResidueBatch(
        hidden=hidden,
        input_ids=input_ids,
        attention_mask=attention,
        content_mask=content,
        records=records,
        provenance=provenance,
    )


def _two_record_batch(
    *,
    time: int = 5,
    hidden_width: int = 4,
    marker_id: int = 1,
    pad_id: int = 0,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> tuple[ResidueBatch, int]:
    records = (
        _record(
            accession="synthetic-A",
            residue_count=2,
            native_token_ids=(marker_id, 10, 11),
            sequence_sha256=SYN_HASH_A,
        ),
        _record(
            accession="synthetic-B",
            residue_count=3,
            native_token_ids=(marker_id, 12, 13, 14),
            sequence_sha256=SYN_HASH_B,
        ),
    )
    provenance = _provenance(hidden_width=hidden_width, marker_id=marker_id, pad_id=pad_id)
    generator = torch.Generator()
    generator.manual_seed(seed)
    hidden, ids, attn, content = _geometry(
        records,
        time=time,
        hidden_width=hidden_width,
        marker_id=marker_id,
        pad_id=pad_id,
        dtype=dtype,
        generator=generator,
    )
    batch = _batch_from(records, provenance, hidden, ids, attn, content)
    used = (
        hidden.numel() * hidden.element_size()
        + ids.numel() * ids.element_size()
        + attn.numel() * attn.element_size()
        + content.numel() * content.element_size()
    )
    return batch, int(used)


def _query(
    *,
    input_width: int,
    output_width: int,
    n_queries: int = 2,
    n_heads: int = 2,
    internal_width: int | None = None,
    seed: int = 1,
) -> QueryPrefix:
    width = input_width if internal_width is None else internal_width
    budget = query_prefix_parameter_count(
        input_width=input_width,
        internal_width=width,
        output_width=output_width,
        n_queries=n_queries,
    )
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
    return QueryPrefix(
        input_width=input_width,
        internal_width=width,
        output_width=output_width,
        n_queries=n_queries,
        n_heads=n_heads,
        max_trainable_parameters=budget,
    )


class MetadataContractTests(unittest.TestCase):
    def test_record_and_provenance_refuse_invalid_fields(self) -> None:
        valid_ids = (1, 10, 11)
        cases: tuple[tuple[str, Any], ...] = (
            ("empty accession", lambda: ResidueRecord("", SYN_HASH_A, 2, valid_ids)),
            ("long accession", lambda: ResidueRecord("x" * 129, SYN_HASH_A, 2, valid_ids)),
            ("upper hash", lambda: ResidueRecord("synthetic-A", "C" * 64, 2, valid_ids)),
            ("bool length", lambda: ResidueRecord("synthetic-A", SYN_HASH_A, True, (1, 10))),
            ("short native ids", lambda: ResidueRecord("synthetic-A", SYN_HASH_A, 2, (1, 10))),
            ("negative id", lambda: ResidueRecord("synthetic-A", SYN_HASH_A, 1, (1, -1))),
        )
        for name, factory in cases:
            with self.subTest(name):
                with self.assertRaises((TypeError, ValueError)):
                    factory()
        with self.assertRaises(ValueError):
            ResidueProvenance(
                schema="latent_bridge_cache_v2",
                donor_inventory_sha256=SYN_INVENTORY,
                code_commit=SYN_COMMIT,
                block_index=0,
                block_index_semantics=lb.BLOCK_INDEX_SEMANTICS,
                capture_dtype="float32",
                hidden_width=4,
                marker_id=1,
                pad_id=0,
            )
        with self.assertRaises(ValueError):
            ResidueProvenance(
                schema=IN_MEMORY_SCHEMA,
                donor_inventory_sha256=SYN_INVENTORY,
                code_commit=SYN_COMMIT,
                block_index=0,
                block_index_semantics=lb.BLOCK_INDEX_SEMANTICS,
                capture_dtype="float32",
                hidden_width=4,
                marker_id=3,
                pad_id=3,
            )
        with self.assertRaises(TypeError):
            ResidueProvenance(
                schema=IN_MEMORY_SCHEMA,
                donor_inventory_sha256=SYN_INVENTORY,
                code_commit=SYN_COMMIT,
                block_index=False,
                block_index_semantics=lb.BLOCK_INDEX_SEMANTICS,
                capture_dtype="float32",
                hidden_width=4,
                marker_id=1,
                pad_id=0,
            )


class HandoffRefusalTests(unittest.TestCase):
    def test_consumer_refuses_malformed_batches(self) -> None:
        batch, used = _two_record_batch()
        records = batch.records
        provenance = batch.provenance
        query = _query(input_width=4, output_width=4)

        def consume(target: ResidueBatch, **kwargs: Any) -> None:
            validate_residue_batch(
                target,
                expected_records=kwargs.get("records", records),
                expected_provenance=kwargs.get("provenance", provenance),
                max_tensor_bytes=kwargs.get("max_tensor_bytes", used),
            )

        swapped = ResidueBatch(
            hidden=batch.hidden,
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            content_mask=batch.content_mask,
            records=(records[1], records[0]),
            provenance=provenance,
        )
        mean_like = ResidueBatch(
            hidden=batch.hidden.mean(dim=1),
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            content_mask=batch.content_mask,
            records=records,
            provenance=provenance,
        )
        truncated = _batch_from(
            records,
            provenance,
            batch.hidden[:, :3],
            batch.input_ids[:, :3],
            batch.attention_mask[:, :3],
            batch.content_mask[:, :3],
        )
        nan_hidden = batch.hidden.clone()
        nan_hidden[0, 1, 0] = torch.nan
        nan_batch = _batch_from(records, provenance, nan_hidden, batch.input_ids, batch.attention_mask, batch.content_mask)
        grad_hidden = batch.hidden.detach().clone().requires_grad_(True)
        grad_batch = _batch_from(records, provenance, grad_hidden, batch.input_ids, batch.attention_mask, batch.content_mask)
        base = batch.hidden.detach().clone().requires_grad_(True)
        graph_live = base * 2
        graph_batch = _batch_from(records, provenance, graph_live, batch.input_ids, batch.attention_mask, batch.content_mask)
        empty = ResidueBatch(
            hidden=batch.hidden[:0],
            input_ids=batch.input_ids[:0],
            attention_mask=batch.attention_mask[:0],
            content_mask=batch.content_mask[:0],
            records=(),
            provenance=provenance,
        )
        wrong_hash = (
            _record(
                accession="synthetic-A",
                residue_count=2,
                native_token_ids=(1, 10, 11),
                sequence_sha256=SYN_HASH_B,
            ),
            records[1],
        )
        other_prov = _provenance(hidden_width=4, marker_id=1, pad_id=0, block_index=1)
        residue_eq_pad = _record(
            accession="synthetic-A",
            residue_count=2,
            native_token_ids=(1, 0, 11),
            sequence_sha256=SYN_HASH_A,
        )
        bad_residue_ids = batch.input_ids.clone()
        bad_residue_ids[0, 1] = 0
        pad_eq_batch = _batch_from(
            (residue_eq_pad, records[1]),
            provenance,
            batch.hidden,
            bad_residue_ids,
            batch.attention_mask,
            batch.content_mask,
        )
        int_ids = _batch_from(
            records,
            provenance,
            batch.hidden,
            batch.input_ids.to(torch.int32),
            batch.attention_mask,
            batch.content_mask,
        )
        bf_decl = _provenance(hidden_width=4, capture_dtype="bfloat16")
        dtype_mismatch = _batch_from(
            records,
            bf_decl,
            batch.hidden,
            batch.input_ids,
            batch.attention_mask,
            batch.content_mask,
        )

        cases: tuple[tuple[str, Any, dict[str, Any]], ...] = (
            ("old 2-D mean", mean_like, {}),
            ("truncation", truncated, {"max_tensor_bytes": 10_000}),
            ("nonfinite", nan_batch, {}),
            ("requires_grad", grad_batch, {}),
            ("grad_fn", graph_batch, {}),
            ("empty batch", empty, {"records": (), "max_tensor_bytes": used}),
            ("order", swapped, {"records": records}),
            ("identity", batch, {"records": wrong_hash}),
            ("provenance", batch, {"provenance": other_prov}),
            ("cap overflow", batch, {"max_tensor_bytes": used - 1}),
            ("cap ceiling", batch, {"max_tensor_bytes": MAX_TENSOR_BYTES_CEILING + 1}),
            ("bool cap", batch, {"max_tensor_bytes": True}),
            ("residue equals pad", pad_eq_batch, {"records": pad_eq_batch.records}),
            ("int32 ids", int_ids, {}),
            ("dtype mismatch", dtype_mismatch, {"provenance": bf_decl}),
        )
        for name, target, kwargs in cases:
            with self.subTest(name):
                with self.assertRaises((TypeError, ValueError)):
                    consume(target, **kwargs)
                with self.assertRaises((TypeError, ValueError)):
                    mean_from_residues(
                        target,
                        expected_records=kwargs.get("records", records),
                        expected_provenance=kwargs.get("provenance", provenance),
                        max_tensor_bytes=kwargs.get("max_tensor_bytes", used),
                    )
                with self.assertRaises((TypeError, ValueError)):
                    query(
                        target,
                        expected_records=kwargs.get("records", records),
                        expected_provenance=kwargs.get("provenance", provenance),
                        max_tensor_bytes=kwargs.get("max_tensor_bytes", used),
                    )
        self.assertIsNotNone(graph_live.grad_fn)

    def test_over_budget_tiny_batch_refuses_before_isfinite(self) -> None:
        batch, used = _two_record_batch()
        records = batch.records
        provenance = batch.provenance
        query = _query(input_width=4, output_width=4)
        self.assertIs(
            validate_residue_batch(
                batch,
                expected_records=records,
                expected_provenance=provenance,
                max_tensor_bytes=used,
            ),
            batch,
        )
        pooled = mean_from_residues(
            batch,
            expected_records=records,
            expected_provenance=provenance,
            max_tensor_bytes=used,
        )
        self.assertEqual(tuple(pooled.shape), (2, 4))
        tokens = query(
            batch,
            expected_records=records,
            expected_provenance=provenance,
            max_tensor_bytes=used,
        )
        self.assertEqual(tuple(tokens.shape), (2, 2, 4))

        class FiniteScanEntered(Exception):
            pass

        over_budget = used - 1

        def refuse_validate() -> None:
            validate_residue_batch(
                batch,
                expected_records=records,
                expected_provenance=provenance,
                max_tensor_bytes=over_budget,
            )

        def refuse_mean() -> None:
            mean_from_residues(
                batch,
                expected_records=records,
                expected_provenance=provenance,
                max_tensor_bytes=over_budget,
            )

        def refuse_query() -> None:
            query(
                batch,
                expected_records=records,
                expected_provenance=provenance,
                max_tensor_bytes=over_budget,
            )

        with patch.object(rp.torch, "isfinite", side_effect=FiniteScanEntered):
            for name, call in (
                ("validate", refuse_validate),
                ("mean", refuse_mean),
                ("query", refuse_query),
            ):
                with self.subTest(consumer=name):
                    with self.assertRaises(ValueError) as ctx:
                        call()
                    self.assertIn("max_tensor_bytes", str(ctx.exception))


class MeanAndQueryInvarianceTests(unittest.TestCase):
    def test_mean_matches_mean_content_including_bfloat16(self) -> None:
        for dtype, capture in ((torch.float32, "float32"), (torch.bfloat16, "bfloat16")):
            with self.subTest(capture=capture):
                batch, used = _two_record_batch(dtype=dtype)
                if capture == "bfloat16":
                    provenance = _provenance(hidden_width=4, capture_dtype="bfloat16")
                    batch = _batch_from(
                        batch.records,
                        provenance,
                        batch.hidden,
                        batch.input_ids,
                        batch.attention_mask,
                        batch.content_mask,
                    )
                pooled = mean_from_residues(
                    batch,
                    expected_records=batch.records,
                    expected_provenance=batch.provenance,
                    max_tensor_bytes=used,
                )
                reference = lb.mean_content(batch.hidden.float(), batch.content_mask)
                self.assertEqual(pooled.dtype, torch.float32)
                self.assertTrue(torch.equal(pooled, reference))

    def test_marker_pad_values_and_right_padding_are_masked(self) -> None:
        batch, used = _two_record_batch(time=5, marker_id=1, pad_id=0, seed=2)
        query = _query(input_width=4, output_width=6, seed=3)
        mean_a = mean_from_residues(
            batch,
            expected_records=batch.records,
            expected_provenance=batch.provenance,
            max_tensor_bytes=used,
        )
        query_a = query(
            batch,
            expected_records=batch.records,
            expected_provenance=batch.provenance,
            max_tensor_bytes=used,
        )
        alt_marker, alt_pad = 9, 7
        alt_records = (
            _record(
                accession="synthetic-A",
                residue_count=2,
                native_token_ids=(alt_marker, 10, 11),
                sequence_sha256=SYN_HASH_A,
            ),
            _record(
                accession="synthetic-B",
                residue_count=3,
                native_token_ids=(alt_marker, 12, 13, 14),
                sequence_sha256=SYN_HASH_B,
            ),
        )
        alt_prov = _provenance(hidden_width=4, marker_id=alt_marker, pad_id=alt_pad)
        alt_ids = batch.input_ids.clone()
        alt_hidden = batch.hidden.clone()
        for row, record in enumerate(alt_records):
            attended = record.residue_count + 1
            alt_ids[row, 0] = alt_marker
            alt_ids[row, attended:] = alt_pad
            alt_hidden[row, 0] = torch.randn(4)
            if attended < alt_hidden.shape[1]:
                alt_hidden[row, attended:] = torch.randn(alt_hidden.shape[1] - attended, 4)
        alt_batch = _batch_from(
            alt_records,
            alt_prov,
            alt_hidden,
            alt_ids,
            batch.attention_mask,
            batch.content_mask,
        )
        mean_b = mean_from_residues(
            alt_batch,
            expected_records=alt_records,
            expected_provenance=alt_prov,
            max_tensor_bytes=used,
        )
        query_b = query(
            alt_batch,
            expected_records=alt_records,
            expected_provenance=alt_prov,
            max_tensor_bytes=used,
        )
        self.assertTrue(torch.equal(mean_a, mean_b))
        self.assertTrue(torch.allclose(query_a, query_b, rtol=FP32_RTOL, atol=FP32_ATOL))

        long_time = 8
        long_hidden = torch.zeros(2, long_time, 4, dtype=torch.float32)
        long_hidden[:, :5] = batch.hidden
        long_hidden[:, 5:] = torch.randn(2, long_time - 5, 4)
        long_ids = torch.full((2, long_time), 0, dtype=torch.long)
        long_ids[:, :5] = batch.input_ids
        long_attn = torch.zeros(2, long_time, dtype=torch.bool)
        long_attn[:, :5] = batch.attention_mask
        long_content = torch.zeros(2, long_time, dtype=torch.bool)
        long_content[:, :5] = batch.content_mask
        long_batch = _batch_from(
            batch.records,
            batch.provenance,
            long_hidden,
            long_ids,
            long_attn,
            long_content,
        )
        long_used = (
            long_hidden.numel() * long_hidden.element_size()
            + long_ids.numel() * long_ids.element_size()
            + long_attn.numel() * long_attn.element_size()
            + long_content.numel() * long_content.element_size()
        )
        mean_c = mean_from_residues(
            long_batch,
            expected_records=batch.records,
            expected_provenance=batch.provenance,
            max_tensor_bytes=long_used,
        )
        query_c = query(
            long_batch,
            expected_records=batch.records,
            expected_provenance=batch.provenance,
            max_tensor_bytes=long_used,
        )
        self.assertTrue(torch.equal(mean_a, mean_c))
        self.assertTrue(torch.allclose(query_a, query_c, rtol=FP32_RTOL, atol=FP32_ATOL))


class QueryPrefixBudgetAndGradTests(unittest.TestCase):
    def test_shape_budget_and_nonzero_adapter_grads(self) -> None:
        predicted = query_prefix_parameter_count(
            input_width=4, internal_width=8, output_width=6, n_queries=3
        )
        with self.assertRaises(ValueError):
            QueryPrefix(
                input_width=4,
                internal_width=8,
                output_width=6,
                n_queries=3,
                n_heads=3,
                max_trainable_parameters=predicted,
            )
        with self.assertRaises(ValueError):
            QueryPrefix(
                input_width=4,
                internal_width=8,
                output_width=6,
                n_queries=3,
                n_heads=2,
                max_trainable_parameters=predicted - 1,
            )
        module = QueryPrefix(
            input_width=4,
            internal_width=8,
            output_width=6,
            n_queries=3,
            n_heads=2,
            max_trainable_parameters=predicted,
        )
        self.assertEqual(module.trainable_parameter_count(), predicted)
        batch, used = _two_record_batch()
        train_out = module(
            batch,
            expected_records=batch.records,
            expected_provenance=batch.provenance,
            max_tensor_bytes=used,
        )
        eval_out = module.eval()(
            batch,
            expected_records=batch.records,
            expected_provenance=batch.provenance,
            max_tensor_bytes=used,
        )
        self.assertEqual(tuple(train_out.shape), (2, 3, 6))
        self.assertEqual(train_out.dtype, torch.float32)
        self.assertTrue(torch.allclose(train_out, eval_out, rtol=0.0, atol=0.0))
        module.train()
        loss = train_out.square().sum()
        torch.Tensor.backward(loss)  # pyright: ignore[reportUnknownMemberType]
        grads = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
        self.assertTrue(grads and all(grad is not None for grad in grads))
        self.assertTrue(all(torch.isfinite(grad).all() for grad in grads if grad is not None))
        self.assertTrue(any(float(grad.abs().sum()) > 0 for grad in grads if grad is not None))
        self.assertFalse(batch.hidden.requires_grad)
        self.assertIsNone(batch.hidden.grad)


class SoftPrefixIntegrationTests(unittest.TestCase):
    def test_two_record_answer_only_qwen2_and_embed_parity(self) -> None:
        donor_width = 8
        receiver_width = 16
        n_queries = 2
        batch, used = _two_record_batch(hidden_width=donor_width, time=5, seed=4)
        adapter = _query(
            input_width=donor_width,
            output_width=receiver_width,
            n_queries=n_queries,
            n_heads=2,
            internal_width=8,
            seed=5,
        )
        config = Qwen2Config(
            vocab_size=64,
            hidden_size=receiver_width,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=32,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            attention_dropout=0.0,
            use_cache=False,
        )
        torch.manual_seed(6)  # pyright: ignore[reportUnknownMemberType]
        receiver = Qwen2ForCausalLM(config)
        lb.freeze_module(receiver)
        embed = receiver.get_input_embeddings()

        # inject_soft_prefix concatenates soft + fixed-width prompt + answer
        # without compacting masked prompt slots. Shifted CE therefore predicts
        # the first answer from the token immediately before it. Left-pad prompts
        # so that token is the last real prompt token; right-pad answers, with EOS
        # inside both answer rows, so pad is not a label. This is a qualified
        # caller recipe, not a production QA collator or support for interior
        # mask holes.
        prompt_ids = torch.tensor([[0, 0, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
        prompt_mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]], dtype=torch.long)
        answer_ids = torch.tensor([[9, 2, 0], [10, 11, 2]], dtype=torch.long)
        answer_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long)
        soft = adapter(
            batch,
            expected_records=batch.records,
            expected_provenance=batch.provenance,
            max_tensor_bytes=used,
        )
        self.assertEqual(tuple(soft.shape), (2, n_queries, receiver_width))
        packed = lb.inject_soft_prefix(
            soft=soft,
            prompt_ids=prompt_ids,
            answer_ids=answer_ids,
            embed=embed,
            pad_id=0,
            prompt_mask=prompt_mask,
            answer_mask=answer_mask,
        )
        labels = packed["labels"]
        supervised = labels != lb.IGNORE_INDEX
        self.assertEqual(int(supervised.sum()), 5)
        self.assertTrue(
            torch.equal(
                supervised[0],
                torch.tensor([False] * (n_queries + 4) + [True, True, False]),
            )
        )
        self.assertTrue(
            torch.equal(
                supervised[1],
                torch.tensor([False] * (n_queries + 4) + [True, True, True]),
            )
        )
        self.assertTrue(torch.equal(labels[supervised], torch.tensor([9, 2, 10, 11, 2])))

        outputs = receiver(
            inputs_embeds=packed["inputs_embeds"],
            attention_mask=packed["attention_mask"],
            position_ids=packed["position_ids"],
        )
        selected = outputs.logits[:, :-1][labels[:, 1:] != lb.IGNORE_INDEX]

        def unpadded_selected(
            prefix: torch.Tensor,
            prompts: torch.Tensor,
            answers: torch.Tensor,
            pmask: torch.Tensor,
            amask: torch.Tensor,
        ) -> torch.Tensor:
            packed_row = lb.inject_soft_prefix(
                soft=prefix,
                prompt_ids=prompts,
                answer_ids=answers,
                embed=embed,
                pad_id=0,
                prompt_mask=pmask,
                answer_mask=amask,
            )
            with torch.no_grad():
                row_out = receiver(
                    inputs_embeds=packed_row["inputs_embeds"],
                    attention_mask=packed_row["attention_mask"],
                    position_ids=packed_row["position_ids"],
                )
            return row_out.logits[:, :-1][packed_row["labels"][:, 1:] != lb.IGNORE_INDEX]

        reference = torch.cat(
            [
                unpadded_selected(
                    soft[:1],
                    torch.tensor([[3, 4]], dtype=torch.long),
                    torch.tensor([[9, 2]], dtype=torch.long),
                    torch.ones(1, 2, dtype=torch.long),
                    torch.ones(1, 2, dtype=torch.long),
                ),
                unpadded_selected(
                    soft[1:],
                    torch.tensor([[5, 6, 7, 8]], dtype=torch.long),
                    torch.tensor([[10, 11, 2]], dtype=torch.long),
                    torch.ones(1, 4, dtype=torch.long),
                    torch.ones(1, 3, dtype=torch.long),
                ),
            ],
            dim=0,
        )
        self.assertEqual(int(selected.shape[0]), 5)
        self.assertTrue(torch.allclose(selected, reference, rtol=FP32_RTOL, atol=FP32_ATOL))

        loss = lb.shifted_cross_entropy(outputs.logits, labels)
        torch.Tensor.backward(loss)  # pyright: ignore[reportUnknownMemberType]
        self.assertFalse(batch.hidden.requires_grad)
        self.assertIsNone(batch.hidden.grad)
        for parameter in receiver.parameters():
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)
        adapter_grads = [parameter.grad for parameter in adapter.parameters()]
        self.assertTrue(all(grad is not None for grad in adapter_grads))
        self.assertTrue(all(torch.isfinite(grad).all() for grad in adapter_grads if grad is not None))
        self.assertTrue(any(float(grad.abs().sum()) > 0 for grad in adapter_grads if grad is not None))

        # inject_soft_prefix allows K=0: it checks rank 3, not positive K, and
        # torch.cat accepts the empty soft tensor. QueryPrefix still requires
        # positive n_queries.
        ids = torch.tensor([[3, 4, 5, 0], [6, 7, 0, 0]], dtype=torch.long)
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long)
        position_ids = (mask.cumsum(dim=1) - 1).clamp(min=0)
        packed_zero = lb.inject_soft_prefix(
            soft=torch.empty(ids.shape[0], 0, receiver_width),
            prompt_ids=ids,
            answer_ids=None,
            embed=embed,
            pad_id=0,
            prompt_mask=mask,
        )
        self.assertTrue(torch.equal(packed_zero["attention_mask"], mask))
        self.assertTrue(torch.equal(packed_zero["position_ids"], position_ids))
        with torch.no_grad():
            from_ids = receiver(input_ids=ids, attention_mask=mask, position_ids=position_ids).logits
            from_packer = receiver(
                inputs_embeds=packed_zero["inputs_embeds"],
                attention_mask=packed_zero["attention_mask"],
                position_ids=packed_zero["position_ids"],
            ).logits
        self.assertEqual(from_ids.dtype, from_packer.dtype)
        self.assertTrue(torch.allclose(from_ids, from_packer, rtol=FP32_RTOL, atol=FP32_ATOL))


if __name__ == "__main__":
    unittest.main()
