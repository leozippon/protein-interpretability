"""CPU tests for the M0/M1 EC-top-class latent bridge. No real weights."""

from __future__ import annotations

import copy
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.transfer import latent_bridge as lb
from src.transfer.io import write_json
from src.transfer.sequence_description import SequenceDescriptionRecord


def _record(
    *, accession: str = "P1", sequence: str = "ACDEFGHIKL", length: int | None = None,
    ec: tuple[str, ...] = ("1.1.1.1",), dup_group: int = 1,
    family_group: str = "famA", split: str = "fit",
) -> SequenceDescriptionRecord:
    return SequenceDescriptionRecord(
        accession=accession,
        sequence=sequence,
        length=len(sequence) if length is None else length,
        name="n",
        function_text="f",
        description_raw="raw",
        description_masked="masked",
        masked_terms=(),
        ec=ec,
        go=("GO:1",),
        go_propagated=("GO:1",),
        pfam=(),
        cath=(),
        dup_group=dup_group,
        family_group=family_group,
        split=split,
    )


class FilterTests(unittest.TestCase):
    def test_empty_ec_excluded(self) -> None:
        kept, counts = lb.filter_records([_record(ec=())])
        self.assertEqual(kept, [])
        self.assertEqual(counts["empty_ec"], 1)

    def test_cross_class_ec_dropped_same_class_kept(self) -> None:
        mixed = _record(accession="M", ec=("1.1.1.1", "2.1.1.1"), dup_group=2)
        same = _record(accession="S", ec=("1.1.1.1", "1.2.3.4"), dup_group=3)
        kept, counts = lb.filter_records([mixed, same])
        self.assertEqual(counts["multiple_top_level_classes"], 1)
        self.assertEqual([row.accession for row in kept], ["S"])
        self.assertEqual(kept[0].ec_class, 1)

    def test_illegal_aa_and_overlength(self) -> None:
        bad = _record(accession="X", sequence="ACDEX", length=5, dup_group=4)
        long = _record(accession="L", sequence="A" * 12, length=12, dup_group=5)
        kept, counts = lb.filter_records([bad, long], max_residues=10)
        self.assertEqual(counts["illegal_aa"], 1)
        self.assertEqual(counts["over_max_residues"], 1)
        self.assertEqual(kept, [])

    def test_target_json_is_canonical(self) -> None:
        self.assertEqual(lb.target_json(1), '{"class":1,"name":"oxidoreductase"}')
        self.assertEqual(lb.target_json(7), '{"class":7,"name":"translocase"}')


class SplitTests(unittest.TestCase):
    def test_dup_leak_raises(self) -> None:
        rows = [
            _record(accession="a", split="fit", dup_group=1, family_group="F"),
            _record(accession="b", split="eval", dup_group=1, family_group="G"),
        ]
        with self.assertRaises(ValueError):
            lb.verify_split_invariants(rows)

    def test_holdout_family_leak_raises(self) -> None:
        rows = [
            _record(accession="a", split="fit", dup_group=1, family_group="F"),
            _record(accession="b", split="family_holdout", dup_group=2, family_group="F"),
        ]
        with self.assertRaises(ValueError):
            lb.verify_split_invariants(rows)

    def test_fit_eval_same_family_ok(self) -> None:
        rows = [
            _record(accession="a", split="fit", dup_group=1, family_group="F"),
            _record(accession="b", split="eval", dup_group=2, family_group="F"),
        ]
        lb.verify_split_invariants(rows)

    def test_provenance_missing_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                lb.load_stage34_provenance(path, Path(tmp) / "cohort.json")

    def test_known_digest_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records = Path(tmp) / "records.jsonl"
            cohort = Path(tmp) / "cohort.json"
            rec = _record()
            records.write_text(json.dumps(rec.to_dict()) + "\n", encoding="utf-8")
            write_json(cohort, {"ok": True})
            with self.assertRaises(ValueError):
                lb.load_stage34_provenance(
                    records, cohort, expect_records_sha256="deadbeef"
                )

    def test_cap_uses_accession_hash_not_label(self) -> None:
        rows = [
            lb.filter_records([_record(accession=f"P{i}", dup_group=i, ec=("1.1.1.1",))])[0][0]
            for i in range(6)
        ]
        capped = lb.cap_split(rows, 3, seed=0)
        self.assertEqual(len(capped), 3)
        expected = [
            row.accession
            for row in sorted(rows, key=lambda r: (lb.cap_key(r.accession, 0), r.accession))[:3]
        ]
        self.assertEqual([row.accession for row in capped], expected)


class ParseTests(unittest.TestCase):
    def test_invalid_json_and_embedded_digit(self) -> None:
        self.assertFalse(lb.parse_generated_json("the class is 1")["valid"])
        self.assertFalse(lb.parse_generated_json('{"class":1}')["valid"])
        self.assertFalse(lb.parse_generated_json('{"class":"1","name":"oxidoreductase"}')["valid"])
        self.assertTrue(lb.parse_generated_json('{"class":1,"name":"oxidoreductase"}')["valid"])

    def test_metrics_count_invalid_as_failure(self) -> None:
        metrics = lb.confusion_and_scores([1, 1], [1, None])
        self.assertEqual(metrics["n_attempts"], 2)
        self.assertEqual(metrics["class_exact_accuracy"], 0.5)
        self.assertEqual(metrics["n_valid_json"], 1)

    def test_macro_f1_zero_support_policy(self) -> None:
        metrics = lb.confusion_and_scores([1, 1], [1, 1])
        self.assertIn("7", metrics["zero_support_classes"])
        self.assertNotIn("1", metrics["zero_support_classes"])
        self.assertIsNotNone(metrics["macro_f1_over_supported_classes"])
        self.assertLess(metrics["macro_f1_all_seven"], metrics["macro_f1_over_supported_classes"])


class TokenAndMaskTests(unittest.TestCase):
    def test_overlength_refuses(self) -> None:
        class T:
            def __call__(self, text, return_tensors=None, add_special_tokens=False):
                return {"input_ids": [1, 2, 3, 4]}

        with self.assertRaises(ValueError):
            lb.tokenize_full(T(), "ACDE", max_tokens=2)

    def test_content_mask_drops_marker_and_pad(self) -> None:
        ids = torch.tensor([[3, 10, 11, 0]])
        mask = torch.tensor([[1, 1, 1, 0]])
        content = lb.content_mask_from_ids(ids, mask, marker_id=3, pad_id=0)
        self.assertEqual(content.tolist(), [[False, True, True, False]])

    def test_module_does_not_call_tokenize_batch(self) -> None:
        source = Path(lb.__file__).read_text(encoding="utf-8")
        self.assertNotIn("tokenize_batch(", source)
        self.assertNotIn("import nnsight", source)
        self.assertNotIn("import peft", source)
        self.assertNotIn("admission_verdict", source)
        self.assertNotIn("assert_stage35_handoff", source)


class PrefixTests(unittest.TestCase):
    def test_labels_attention_position(self) -> None:
        embed = nn.Embedding(20, 4)
        soft = torch.ones(1, 2, 4)
        prompt = torch.tensor([[5, 6, 7]])
        answer = torch.tensor([[8, 9, 1]])
        packed = lb.inject_soft_prefix(
            soft=soft, prompt_ids=prompt, answer_ids=answer, embed=embed, pad_id=0
        )
        self.assertEqual(tuple(packed["inputs_embeds"].shape), (1, 8, 4))
        self.assertEqual(packed["labels"][0, :5].tolist(), [-100] * 5)
        self.assertEqual(packed["labels"][0, 5:].tolist(), [8, 9, 1])
        self.assertEqual(packed["attention_mask"].tolist(), [[1] * 8])
        self.assertEqual(packed["position_ids"].tolist(), [list(range(8))])

    def test_shifted_ce_matches_reference(self) -> None:
        class Tiny(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Embedding(12, 4)
                self.unembed = nn.Linear(4, 12, bias=False)

            def forward(self, inputs_embeds=None, attention_mask=None, position_ids=None, **_):
                logits = self.unembed(inputs_embeds)
                return SimpleNamespace(logits=logits)

        model = Tiny()
        soft = torch.randn(1, 2, 4)
        prompt = torch.tensor([[1, 2]])
        answer = torch.tensor([[3, 4]])
        packed = lb.inject_soft_prefix(
            soft=soft,
            prompt_ids=prompt,
            answer_ids=answer,
            embed=model.embed,
            pad_id=0,
        )
        logits = model(inputs_embeds=packed["inputs_embeds"]).logits
        loss = lb.shifted_cross_entropy(logits, packed["labels"])
        ref = F.cross_entropy(
            logits[:, :-1].reshape(-1, 12),
            packed["labels"][:, 1:].reshape(-1),
            ignore_index=-100,
        )
        self.assertTrue(torch.allclose(loss, ref))

    def test_frozen_receiver_only_bridge_updates(self) -> None:
        class Tiny(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Embedding(12, 8)
                self.unembed = nn.Linear(8, 12, bias=False)

            def get_input_embeddings(self):
                return self.embed

            def forward(self, inputs_embeds=None, attention_mask=None, position_ids=None, **_):
                if inputs_embeds is None:
                    raise ValueError("inputs_embeds required")
                mixed = inputs_embeds.cumsum(dim=1)
                return SimpleNamespace(logits=self.unembed(mixed))

        receiver = Tiny()
        lb.freeze_module(receiver)
        bridge = lb.PoolingMLPBridge(d_in=4, d_recv=8, k=2, hidden=8, output_scale=1.0)
        before_recv = {k: v.clone() for k, v in receiver.state_dict().items()}
        before_bridge = {
            name: parameter.detach().clone()
            for name, parameter in bridge.named_parameters()
            if parameter.requires_grad
        }
        pooled = torch.randn(1, 4)
        packed = lb.inject_soft_prefix(
            soft=bridge(pooled),
            prompt_ids=torch.tensor([[1, 2]]),
            answer_ids=torch.tensor([[3, 4]]),
            embed=receiver.embed,
            pad_id=0,
        )
        loss = lb.shifted_cross_entropy(
            receiver(inputs_embeds=packed["inputs_embeds"]).logits, packed["labels"]
        )
        loss.backward()
        for parameter in receiver.parameters():
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)
        self.assertTrue(
            any(
                p.grad is not None and float(p.grad.abs().sum()) > 0
                for p in bridge.parameters()
                if p.requires_grad
            )
        )
        opt = SGD([p for p in bridge.parameters() if p.requires_grad], lr=0.1)
        opt.step()
        for key, value in receiver.state_dict().items():
            self.assertTrue(torch.equal(value, before_recv[key]))
        self.assertTrue(
            any(
                not torch.equal(dict(bridge.named_parameters())[name], value)
                for name, value in before_bridge.items()
            )
        )


class HookTests(unittest.TestCase):
    def test_hook_is_last_block_not_post_ln_or_hidden_index(self) -> None:
        class AddBlock(nn.Module):
            def __init__(self, delta: float) -> None:
                super().__init__()
                self.delta = delta

            def forward(self, hidden, **_):
                return (hidden + self.delta, None)

        class ToyLM(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Embedding(6, 3)
                self.transformer = SimpleNamespace(
                    h=nn.ModuleList([AddBlock(1.0), AddBlock(10.0)])
                )
                self.post = 1000.0

            def forward(self, input_ids, attention_mask=None, **_):
                hidden = self.embed(input_ids).float()
                states = [hidden]
                for block in self.transformer.h:
                    hidden = block(hidden)[0]
                    states.append(hidden)
                post = hidden + self.post
                states.append(post)
                return SimpleNamespace(logits=post, hidden_states=tuple(states))

        class ToyDonor:
            def __init__(self) -> None:
                self.model = ToyLM()
                self.n_layer = 2

            def blocks(self):
                return self.model.transformer.h

        donor = ToyDonor()
        ids = torch.tensor([[1, 2, 3]])
        mask = torch.ones_like(ids)
        hooked = lb.capture_block_output(donor, layer=1, input_ids=ids, attention_mask=mask)
        outputs = donor.model(ids, attention_mask=mask)
        last_block = outputs.hidden_states[2]
        post_ln = outputs.hidden_states[-1]
        layer_index = outputs.hidden_states[1]
        self.assertTrue(torch.allclose(hooked, last_block))
        self.assertFalse(torch.allclose(hooked, post_ln))
        self.assertFalse(torch.allclose(hooked, layer_index))
        self.assertEqual(len(donor.blocks()[1]._forward_hooks), 0)

    def test_chat_without_template_raises(self) -> None:
        class Tok:
            chat_template = None

            def __call__(self, *a, **k):
                return {"input_ids": [1]}

        with self.assertRaises(ValueError):
            lb.encode_prompt(Tok(), "hello", prompt_mode="chat", max_tokens=8)


class CacheAndNnTests(unittest.TestCase):
    def test_cache_order_mismatch_raises(self) -> None:
        rows = lb.filter_records(
            [
                _record(accession="A", dup_group=1, ec=("1.1.1.1",)),
                _record(accession="B", dup_group=2, ec=("2.1.1.1",)),
            ]
        )[0]
        feats = np.zeros((2, 3), dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.npz"
            lb.save_feature_cache(path, feats, rows, {"layer": 0, "pooling": "mean_content"})
            with self.assertRaises(ValueError):
                lb.load_feature_cache(path, list(reversed(rows)))

    def test_checkpoint_digest_uses_weight_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "tokenizer.json").write_text("tok", encoding="utf-8")
            (root / "model.safetensors").write_bytes(b"aaaa")
            first = lb.checkpoint_content_digest(root)
            (root / "model.safetensors").write_bytes(b"bbbb")
            second = lb.checkpoint_content_digest(root)
            self.assertNotEqual(first["content_digest"], second["content_digest"])
            self.assertIn("model.safetensors", first["files"])

    def test_nn_gallery_is_fit_only(self) -> None:
        train = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        query = np.array([[0.9, 0.1, 0.0]])
        pred = lb.nearest_neighbor_predict(
            query, train, [1, 2], ["A", "B"], metric="cosine"
        )
        self.assertEqual(pred, [1])

    def test_bootstrap_common_ids_and_few_groups(self) -> None:
        left = [{"accession": "a", "family_group": "F", "correct": 1}]
        right = [{"accession": "b", "family_group": "F", "correct": 0}]
        with self.assertRaises(ValueError):
            lb.paired_family_bootstrap(left, right)
        tiny = [
            {"accession": "a", "family_group": "F", "correct": 1},
            {"accession": "b", "family_group": "G", "correct": 0},
        ]
        result = lb.paired_family_bootstrap(tiny, tiny, minimum_units=8)
        self.assertFalse(result["available"])
        self.assertIsNone(result["interval"])
        self.assertEqual(result["estimand"], "all_attempts_accuracy_difference")
        self.assertEqual(result["resampling_unit"], "family_group")
        mismatched = [dict(tiny[0], family_group="G"), tiny[1]]
        with self.assertRaisesRegex(ValueError, "mismatched family_group"):
            lb.paired_family_bootstrap(tiny, mismatched)

    def test_bootstrap_unequal_families_preserve_all_attempts(self) -> None:
        # The three-record family gains three hits; each singleton loses one.
        families = ["F", "F", "F", "G", "H"]
        left = []
        right = []
        for i, family in enumerate(families):
            for rows, correct in ((left, family == "F"), (right, family != "F")):
                rows.append({
                    "accession": str(i), "family_group": family, "correct": correct,
                    "true_class": 1, "predicted_class": 1 if correct else None,
                    "valid_json": correct,
                    "predicted_name": "oxidoreductase" if correct else None,
                })
        options = {"seed": 17, "n_resamples": 11, "minimum_units": 3}
        result = lb.paired_family_bootstrap(left, right, **options)
        metric_delta = (
            lb.prediction_metrics(left)["class_exact_accuracy"]
            - lb.prediction_metrics(right)["class_exact_accuracy"]
        )
        self.assertTrue(result["available"])
        self.assertAlmostEqual(result["delta"], metric_delta)
        self.assertAlmostEqual(result["delta"], 1 / 5)
        self.assertNotAlmostEqual(result["delta"], (1 - 1 - 1) / 3)
        self.assertEqual(result["estimand"], "all_attempts_accuracy_difference")
        self.assertEqual(result["resampling_unit"], "family_group")

        # Independent reference: expand every sampled family to its records,
        # counting a repeated family repeatedly rather than averaging families.
        record_deltas = {"F": [1, 1, 1], "G": [-1], "H": [-1]}
        rng = np.random.default_rng(options["seed"])
        samples = []
        for _ in range(options["n_resamples"]):
            drawn = rng.choice(["F", "G", "H"], size=3, replace=True)
            hits = [hit for family in drawn for hit in record_deltas[family]]
            samples.append(sum(hits) / len(hits))
        expected_interval = np.quantile(samples, [0.025, 0.975])
        np.testing.assert_allclose(result["interval"], expected_interval)
        self.assertEqual(result, lb.paired_family_bootstrap(left, right, **options))
        self.assertEqual(
            result, lb.paired_family_bootstrap(list(reversed(left)), right, **options)
        )
        self.assertEqual(
            result, lb.paired_family_bootstrap(left, list(reversed(right)), **options)
        )

    def test_bootstrap_duplicate_accessions_rejected(self) -> None:
        rows = [{"accession": "a", "family_group": "F", "correct": True}]
        for arm in ("left", "right"):
            with self.subTest(arm=arm):
                left = rows * 2 if arm == "left" else rows
                right = rows * 2 if arm == "right" else rows
                with self.assertRaisesRegex(ValueError, f"unique accessions in {arm} arm"):
                    lb.paired_family_bootstrap(left, right)

    def test_bootstrap_nonpositive_resample_count_rejected(self) -> None:
        rows = [{"accession": "a", "family_group": "F", "correct": True}]
        for n_resamples in (0, -1):
            with self.subTest(n_resamples=n_resamples):
                with self.assertRaisesRegex(ValueError, "n_resamples must be positive"):
                    lb.paired_family_bootstrap(rows, rows, n_resamples=n_resamples)


class BudgetAndCliTests(unittest.TestCase):
    def test_hidden_width_reports_actual_parameters(self) -> None:
        hidden, actual = lb.hidden_width_for_budget(d_in=16, d_recv=8, k=2, budget=200)
        bridge = lb.PoolingMLPBridge(
            d_in=16, d_recv=8, k=2, hidden=hidden, output_scale=1.0
        )
        self.assertEqual(actual, bridge.trainable_parameter_count())
        self.assertLessEqual(actual, 200)

    def test_prepare_cli_writes_report(self) -> None:
        script = REPO / "scripts/transfer/48_latent_bridge.py"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            recs = [
                _record(accession="a", split="fit", dup_group=1, family_group="F", ec=("1.1.1.1",)),
                _record(accession="b", split="eval", dup_group=2, family_group="G", ec=("2.1.1.1",)),
                _record(
                    accession="c",
                    split="family_holdout",
                    dup_group=3,
                    family_group="H",
                    ec=("3.1.1.1",),
                ),
            ]
            records_path = tmp_path / "records.jsonl"
            records_path.write_text(
                "".join(json.dumps(r.to_dict()) + "\n" for r in recs), encoding="utf-8"
            )
            cohort = tmp_path / "cohort.json"
            write_json(cohort, {"note": "toy"})
            out = tmp_path / "out"
            import runpy

            sys.path.insert(0, str(REPO / "scripts/transfer"))
            ns = runpy.run_path(str(script), run_name="not_main")
            code = ns["main"](
                [
                    "--device",
                    "cpu",
                    "--out",
                    str(out),
                    "--substage",
                    "prepare",
                    "--records",
                    str(records_path),
                    "--cohort-json",
                    str(cohort),
                    "--allow-unverified-cohort",
                    "--seed",
                    "0",
                ]
            )
            self.assertEqual(code, 0)
            report = json.loads((out / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["substage"], "prepare")
            self.assertIn("7", report["zero_support_classes_after_cap"]["train"])


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, **_):
        if text.startswith("{"):
            cid = json.loads(text)["class"]
            return {"input_ids": [3] * cid + [10 + cid]}
        return {"input_ids": [1, 4, 5]}

    def decode(self, ids, **_):
        return "invalid JSON"


class NativeTokenizer:
    pad_token_id = 0
    unk_token_id = 2

    def convert_tokens_to_ids(self, symbol):
        return {symbol: i + 3 for i, symbol in enumerate("1ACDEFGHIKLMNPQRSTVWY")}.get(symbol, 2)

    def __call__(self, text, **_):
        return {"input_ids": [self.convert_tokens_to_ids(symbol) for symbol in text]}


class TinyReceiver(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 8)
        self.out = nn.Linear(8, 32)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, inputs_embeds, attention_mask, position_ids, **_):
        hidden = (inputs_embeds * attention_mask.unsqueeze(-1)).cumsum(dim=1)
        hidden = hidden / (position_ids + 1).unsqueeze(-1)
        return SimpleNamespace(logits=self.out(hidden))


class RepairRegressionTests(unittest.TestCase):
    def test_class_correct_name_wrong_remains_primary_hit(self):
        record = lb.filter_records([_record()])[0][0]
        rows = [lb.prediction_row(record, '{"class":1,"name":"wrong"}')]
        self.assertTrue(rows[0]["valid_json"])
        self.assertTrue(rows[0]["correct"])
        self.assertFalse(rows[0]["name_consistent"])
        metrics = lb.prediction_metrics(rows)
        self.assertEqual(metrics["class_exact_accuracy"], 1)
        self.assertEqual(metrics["valid_json_rate"], 1)
        self.assertEqual(metrics["joint_accuracy"], 0)
        self.assertEqual(metrics["name_consistency_rate"], 0)
        for text in ('{"class":true,"name":"x"}', '{"class":1,"name":null}', '{"class":8,"name":"x"}'):
            with self.subTest(text=text):
                row = lb.prediction_row(record, text)
                self.assertFalse(row["correct"])
                self.assertFalse(row["valid_json"])

    def test_batched_greedy_matches_single_with_positions_and_eos(self):
        class PositionDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding.from_pretrained(torch.eye(64))

            def forward(self, inputs_embeds, attention_mask, position_ids, **_):
                ids = inputs_embeds.argmax(dim=-1)
                state = (ids * attention_mask).cumsum(dim=1)
                targets = (state + position_ids + 1) % 63 + 1
                return SimpleNamespace(logits=F.one_hot(targets, 64).float())

        from transformers import GPT2Config, GPT2LMHeadModel

        with lb.seeded_initialization(5):
            hf = GPT2LMHeadModel(GPT2Config(vocab_size=64, n_embd=16, n_layer=2, n_head=2,
                                          n_positions=32, attn_pdrop=0, resid_pdrop=0, embd_pdrop=0)).eval()
        position_model = PositionDecoder()
        for model, embed in ((position_model, position_model.embed), (hf, hf.get_input_embeddings())):
            for eos in (None, 8):
                def decode(sequences):
                    ids, mask = lb.pad_rows(sequences, 0)
                    with torch.no_grad():
                        return lb.greedy_continue(
                            model=model, embed=embed, inputs_embeds=embed(ids), attention_mask=mask,
                            position_ids=(mask.cumsum(1) - 1).clamp(min=0), max_new_tokens=4, eos_id=eos,
                        ).tolist()
                sequences = [[2, 3], [4, 5, 6]]
                batched = decode(sequences)
                for i, sequence in enumerate(sequences):
                    single = decode([sequence])[0]
                    self.assertEqual(batched[i][:len(single)], single)
                    if eos in single:
                        self.assertTrue(all(token == eos for token in batched[i][len(single):]))

    def test_seeded_bridge_init_restores_rng_and_is_reproducible(self):
        features = np.ones((2, 3), dtype=np.float32)
        def build(seed):
            return lb.build_bridge_for_features(features, d_recv=8, k=2, budget=300,
                                                output_scale=1, seed=seed)
        state = torch.random.get_rng_state().clone()
        first, second, different = build(7), build(7), build(8)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertEqual(lb.parameter_fingerprint(first), lb.parameter_fingerprint(second))
        self.assertNotEqual(lb.parameter_fingerprint(first), lb.parameter_fingerprint(different))

    def test_nonfinite_gradients_and_unsampled_weight_changes_are_detected(self):
        bridge = lb.PoolingMLPBridge(d_in=3, d_recv=8, k=2, hidden=4, output_scale=1)
        for parameter in bridge.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.ones_like(parameter)
        lb.require_bridge_gradients(bridge)
        grad = bridge.fc1.weight.grad
        assert grad is not None
        grad[0, 0] = float("inf")
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            lb.require_bridge_gradients(bridge)
        module = nn.Linear(32, 2, bias=False)
        before = lb.parameter_fingerprint(module)
        with torch.no_grad():
            module.weight[1, 20] += 1
            module.weight[1, 21] -= 1
        self.assertNotEqual(before, lb.parameter_fingerprint(module))

    def test_validation_loss_is_answer_token_weighted(self):
        records = lb.filter_records([_record(accession=str(i), dup_group=i, ec=(f"{i}.1.1.1",))
                                     for i in (1, 2, 7)])[0]
        features = np.arange(9, dtype=np.float32).reshape(3, 3)
        with lb.seeded_initialization(1):
            receiver = TinyReceiver()
        losses = []
        for batch_size in (1, 2, 3):
            bridge = lb.build_bridge_for_features(features, d_recv=8, k=2, budget=300,
                                                 output_scale=1, seed=3)
            result = lb.train_bridge(
                bridge=bridge, receiver=receiver, tokenizer=TinyTokenizer(), features=features,
                records=records, val_features=features, val_records=records, prompt_mode="plain",
                user_mode="latent", device="cpu", epochs=1, batch_size=batch_size, lr=0,
                grad_clip=0, seed=0, max_prompt_tokens=64,
            )
            losses.append(result["best_val_loss"])
            self.assertEqual(result["loss_unit"], lb.LOSS_UNIT)
        self.assertAlmostEqual(losses[0], losses[1], places=6)
        self.assertAlmostEqual(losses[0], losses[2], places=6)

    def test_donor_native_token_count_and_truncation_refusal(self):
        record = lb.filter_records([_record()])[0][0]
        tok = NativeTokenizer()
        ids = lb.donor_token_ids(tok, record, max_tokens=32, max_residues=10)
        self.assertEqual(len(ids), record.length + 1)
        for tokens, residues in ((10, 10), (32, 9)):
            with self.assertRaises(ValueError):
                lb.donor_token_ids(tok, record, max_tokens=tokens, max_residues=residues)
        class TruncatingTokenizer(NativeTokenizer):
            def __call__(self, text, **_):
                return {"input_ids": super().__call__(text)["input_ids"][:-1]}
        with self.assertRaisesRegex(ValueError, "every residue"):
            lb.donor_token_ids(TruncatingTokenizer(), record, max_tokens=32, max_residues=10)

    def test_interface_step_preserves_frozen_bytes_with_bfloat16_receiver(self):
        with lb.seeded_initialization(0):
            receiver = TinyReceiver().to(dtype=torch.bfloat16)
            donor = nn.Linear(3, 3)
        lb.freeze_module(receiver)
        lb.freeze_module(donor)
        features = np.array([[1, 2, 3]], dtype=np.float32)
        bridge = lb.build_bridge_for_features(features, d_recv=8, k=2, budget=300, output_scale=1, seed=1)
        result = lb.interface_optimizer_step(
            bridge=bridge, donor=donor, receiver=receiver, tokenizer=TinyTokenizer(), features=features,
            record=lb.filter_records([_record()])[0][0], prompt_mode="plain", max_prompt_tokens=64, device="cpu",
        )
        self.assertTrue(result["bridge_changed"])
        self.assertTrue(result["frozen_parameters_no_grad"])
        self.assertEqual(result["fingerprints_before"], result["fingerprints_after"])


class ArtifactAndDispatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cli = SimpleNamespace(**runpy.run_path(str(REPO / "scripts/transfer/48_latent_bridge.py")))
        raw = [_record(accession=str(i), dup_group=i, family_group=str(i), split=split,
                       ec=(f"{i}.1.1.1",)) for i, split in enumerate(("fit", "eval", "family_holdout"), 1)]
        (self.root / "records.jsonl").write_text("".join(json.dumps(r.to_dict()) + "\n" for r in raw))
        write_json(self.root / "cohort.json", {})
        self.prepared = lb.prepare_cohort(self.root / "records.jsonl", self.root / "cohort.json", seed=0)
        write_json(self.root / "prepared.json", self.prepared)
        self.args = self.cli.build_parser().parse_args([
            "--out", str(self.root), "--substage", "eval", "--condition", "kmer",
            "--prepared", str(self.root / "prepared.json"), "--receiver-path", str(self.root),
            "--cache", str(self.root / "cache.npz"), "--bridge-checkpoint", str(self.root / "bridge.pt"),
            "--max-new-tokens", "2", "--trainable-budget", "30000", "--k-soft", "2",
        ])
        self.records = lb.all_prepared_records(self.prepared)
        self.features = lb.kmer_features([r.sequence for r in self.records]).astype("float32")
        lb.save_feature_cache(self.args.cache, self.features, self.records, {
            "data_hash": self.prepared["prepared_sha256"], "donor_kind": "kmer",
            "checkpoint_digest": {"content_digest": "kmer_3_raw_frequency"},
            "layer": None, "pooling": None, "rendering": "sequence_only",
        })
        _, self.manifest = lb.load_feature_cache(self.args.cache, self.records)
        for name, text in (("config.json", "{}"), ("tokenizer.json", "{}"), ("model.safetensors", "weights")):
            (self.root / name).write_text(text)
        self.digest = lb.checkpoint_content_digest(self.root)
        self.bridge = lb.build_bridge_for_features(self.features, d_recv=8, k=2, budget=30000, output_scale=1, seed=0)
        self.identity = self.make_identity()
        lb.save_bridge(self.args.bridge_checkpoint, self.bridge, {"identity": self.identity})

    def make_identity(self, **changes):
        kwargs: dict[str, Any] = dict(condition="kmer", prepared_hash=self.prepared["prepared_sha256"],
                      cache_manifest=self.manifest, receiver_digest=self.digest, prompt_mode="plain",
                      bridge_config=self.bridge.config(), seed=0)
        kwargs.update(changes)
        return lb.bridge_identity(**kwargs)

    def test_identity_rejects_wrong_condition_cache_receiver_prompt_seed_and_missing_fields(self):
        _, payload = lb.load_bridge(self.args.bridge_checkpoint)
        changed_cache = copy.deepcopy(self.manifest)
        changed_cache["feature_sha256"] = "other"
        changed_receiver = copy.deepcopy(self.digest)
        changed_receiver["content_digest"] = "other"
        for changes in ({"cache_manifest": changed_cache}, {"receiver_digest": changed_receiver},
                        {"prompt_mode": "chat"}, {"seed": 1}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "identity"):
                lb.validate_bridge_identity(payload, self.make_identity(**changes))
        for condition in ("pretrained_donor", "random_donor", "oracle", "nn"):
            with self.assertRaises(ValueError):
                self.make_identity(condition=condition)
        with self.assertRaises(ValueError):
            self.make_identity(prepared_hash="wrong")
        missing = dict(self.manifest)
        del missing["feature_sha256"]
        with self.assertRaises(ValueError):
            self.make_identity(cache_manifest=missing)
        with self.assertRaises(ValueError):
            lb.validate_bridge_identity({"config": self.bridge.config(), "extra": {}}, self.identity)
        relocated = copy.deepcopy(self.digest)
        relocated["checkpoint"] = "/different/host/model"
        self.assertEqual(self.identity, self.make_identity(receiver_digest=relocated))

    def test_cli_kmer_consumes_cache_and_bridge_without_oracle_label(self):
        receiver, tokenizer = TinyReceiver(), TinyTokenizer()
        with patch.dict(self.cli.run_eval.__globals__, {"load_receiver": lambda *a, **k: (receiver, tokenizer)}):
            with patch.object(lb, "user_prompt", wraps=lb.user_prompt) as prompts:
                with patch.object(lb.PoolingMLPBridge, "forward", autospec=True,
                                  side_effect=lb.PoolingMLPBridge.forward) as forward:
                    report = self.cli.run_eval(self.args)
        self.assertEqual(report["condition"], "kmer")
        self.assertGreater(forward.call_count, 0)
        np.testing.assert_array_equal(forward.call_args.args[1].numpy(), self.features[-1:])
        for call in prompts.call_args_list:
            self.assertEqual(call.kwargs["mode"], "latent")
            self.assertIsNone(call.kwargs["oracle_class"])
            self.assertIsNone(call.kwargs["sequence"])
        with patch.dict(self.cli.run_eval.__globals__, {"load_receiver": lambda *a, **k: self.fail("loaded before identity check")}):
            (self.root / "model.safetensors").write_text("changed weights same width")
            with self.assertRaisesRegex(ValueError, "identity"):
                self.cli.run_eval(self.args)

    def test_cli_rejects_oracle_and_nn_eval_and_nn_emits_reparseable_json(self):
        for condition in ("oracle", "nn", "unknown"):
            self.args.condition = condition
            with self.assertRaises(ValueError):
                self.cli.run_eval(self.args)
        self.args.condition = "nn"
        self.cli.run_baselines(self.args)
        rows = json.loads((self.root / "predictions.json").read_text())["rows"]
        for row in rows:
            self.assertEqual(lb.parse_generated_json(row["text"])["class_id"], row["predicted_class"])
            self.assertTrue(row["name_consistent"])

    def test_prepared_hash_and_cache_dimensions_are_verified(self):
        changed = copy.deepcopy(self.prepared)
        changed["records"]["train"][0]["ec_class"] = 7
        write_json(self.root / "prepared.json", changed)
        with self.assertRaisesRegex(ValueError, "prepared_sha256"):
            lb.load_prepared_payload(self.root / "prepared.json")
        changed_manifest = dict(self.manifest, d=3)
        write_json(self.args.cache.with_suffix(".json"), changed_manifest)
        with self.assertRaisesRegex(ValueError, "shape"):
            lb.load_feature_cache(self.args.cache, self.records)

    def test_cli_m0_oracle_failure_is_nonzero_after_real_toy_chain(self):
        class ToyDonorModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(32, 3)
                self.block = nn.Linear(3, 3)
                self.transformer = SimpleNamespace(h=nn.ModuleList([self.block]))

            def forward(self, input_ids, attention_mask):
                return self.block(self.embed(input_ids))
        donor_model = ToyDonorModel()
        lb.freeze_module(donor_model)
        donor = lb.DonorHandle(name="toy", model=donor_model, tokenizer=NativeTokenizer(), n_layer=1,
                              d_model=3, kind="pretrained",
                              checkpoint_digest={"parameter_fingerprint": lb.parameter_fingerprint(donor_model)})
        receiver = TinyReceiver()
        lb.freeze_module(receiver)
        replacements = {"load_donor_arm": lambda *a, **k: donor,
                        "load_receiver": lambda *a, **k: (receiver, TinyTokenizer())}
        with patch.dict(self.cli.run_interface.__globals__, replacements):
            code = self.cli.main([
                "--out", str(self.root), "--substage", "interface", "--prepared", str(self.root / "prepared.json"),
                "--receiver-path", str(self.root), "--max-interface-records", "1", "--max-new-tokens", "2",
                "--trainable-budget", "300", "--k-soft", "2",
            ])
        self.assertEqual(code, 1)
        report = json.loads((self.root / "report.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["gates"]["native_marker_full_residues_content_mask"])
        self.assertTrue(report["gates"]["finite_bridge_gradient_optimizer_step_frozen_endpoints"])
        self.assertFalse(report["gates"]["oracle_all_seven_canonical_json"])
        self.assertEqual(len(report["content_audit"]), 1)
        self.assertEqual(report["content_audit"][0]["accession"], "1")
        self.assertEqual(len(report["oracle_rows"]), 7)


if __name__ == "__main__":
    unittest.main()
