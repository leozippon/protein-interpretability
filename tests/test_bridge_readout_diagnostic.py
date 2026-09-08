"""CPU tests for the read-only val readout diagnostic."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.transfer import bridge_readout_diagnostic as br
from src.transfer import classifier_handoff as ch
from src.transfer import latent_bridge as lb
from src.transfer.io import sha256_file, write_json
from src.transfer.sequence_description import SequenceDescriptionRecord

SCRIPT = REPO / "scripts/transfer/val_readout_diagnostic.py"


def _load_cli() -> Any:
    spec = importlib.util.spec_from_file_location("val_readout_diagnostic_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(
    *,
    accession: str = "P1",
    sequence: str = "ACDEFGHIKL",
    ec: tuple[str, ...] = ("1.1.1.1",),
    dup_group: int = 1,
    family_group: str = "famA",
    split: str = "fit",
) -> SequenceDescriptionRecord:
    return SequenceDescriptionRecord(
        accession=accession,
        sequence=sequence,
        length=len(sequence),
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


def _prepared(root: Path) -> dict[str, Any]:
    raw = [
        _record(accession="t1", split="fit", dup_group=1, family_group="F1", ec=("1.1.1.1",)),
        _record(accession="t2", split="fit", dup_group=2, family_group="F2", ec=("2.1.1.1",)),
        _record(accession="v1", split="eval", dup_group=3, family_group="F3", ec=("3.1.1.1",)),
        _record(accession="v2", split="eval", dup_group=4, family_group="F4", ec=("4.1.1.1",)),
        _record(accession="h1", split="family_holdout", dup_group=5, family_group="F5", ec=("5.1.1.1",)),
        _record(accession="h2", split="family_holdout", dup_group=6, family_group="F6", ec=("6.1.1.1",)),
    ]
    records_path = root / "records.jsonl"
    records_path.write_text(
        "".join(json.dumps(row.to_dict()) + "\n" for row in raw), encoding="utf-8"
    )
    write_json(root / "cohort.json", {})
    payload = lb.prepare_cohort(
        records_path,
        root / "cohort.json",
        seed=0,
        expect_records_sha256=None,
        expect_cohort_sha256=None,
    )
    write_json(root / "prepared.json", payload)
    return payload


def _donor_manifest(prepared: dict[str, Any]) -> dict[str, Any]:
    return {
        "data_hash": prepared["prepared_sha256"],
        "donor_kind": "pretrained",
        "donor_name": "progen2-medium",
        "checkpoint_digest": {"content_digest": "toy-pretrained"},
        "layer": 0,
        "n_layer": 1,
        "pooling": "mean_content",
        "rendering": "n_to_c_control",
        "direction_marker": "1",
        "max_tokens": 1024,
        "max_residues": 512,
        "dtype": "float32",
    }


def _cache(root: Path, prepared: dict[str, Any], *, d: int = 4) -> Path:
    rows = lb.all_prepared_records(prepared)
    rng = np.random.default_rng(0)
    features = rng.normal(size=(len(rows), d)).astype(np.float32)
    path = root / "donor_cache.npz"
    lb.save_feature_cache(path, features, rows, _donor_manifest(prepared))
    return path


TOY_RECEIVER = {
    "files": {"config.json": "a", "tokenizer.json": "b", "model.safetensors": "c"},
    "content_digest": "toy-receiver",
    "weight_files": ["model.safetensors"],
    "tokenizer_files": ["tokenizer.json"],
}


class DigitTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    PREFIX = '{"class":'
    PREFIX_IDS = [10, 11]
    DIGIT_BASE = 20

    def __call__(self, text, **_):
        if text.startswith(self.PREFIX) and text[len(self.PREFIX) : len(self.PREFIX) + 1].isdigit():
            digit = text[len(self.PREFIX)]
            suffix = text[len(self.PREFIX) + 1 :]
            ids = list(self.PREFIX_IDS) + [self.DIGIT_BASE + int(digit)]
            ids.extend(30 + (index % 20) for index in range(len(suffix)))
            return {"input_ids": ids}
        return {"input_ids": [3, 4, 5]}

    def decode(self, ids, **_):
        values = [int(token) for token in list(ids)]
        if len(values) == 1 and 21 <= values[0] <= 27:
            return str(values[0] - self.DIGIT_BASE)
        return "x"


class MergedTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, **_):
        if text.startswith("{"):
            return {"input_ids": [100 + json.loads(text)["class"]]}
        return {"input_ids": [3, 4]}

    def decode(self, ids, **_):
        values = [int(token) for token in list(ids)]
        if len(values) == 1 and values[0] > 100:
            return f"{values[0] - 100},"
        return "x"


class TinyCausalLM(nn.Module):
    def __init__(self, vocab: int = 64, dim: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.out = nn.Linear(dim, vocab)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, inputs_embeds, attention_mask, position_ids, **_):
        hidden = (inputs_embeds * attention_mask.unsqueeze(-1).to(inputs_embeds.dtype)).cumsum(dim=1)
        return SimpleNamespace(logits=self.out(hidden))


def _bridge(d_in: int = 4, d_recv: int = 8, k: int = 2, seed: int = 0) -> lb.PoolingMLPBridge:
    with lb.seeded_initialization(seed):
        return lb.PoolingMLPBridge(
            d_in=d_in, d_recv=d_recv, k=k, hidden=4, output_scale=1.0
        )


def _save_bridge(
    path: Path,
    bridge: lb.PoolingMLPBridge,
    *,
    prepared: Mapping,
    manifest: Mapping,
    seed: int,
    prompt_mode: str = "plain",
) -> str:
    identity = lb.bridge_identity(
        condition="pretrained_donor",
        prepared_hash=prepared["prepared_sha256"],
        cache_manifest=manifest,
        receiver_digest=TOY_RECEIVER,
        prompt_mode=prompt_mode,
        bridge_config=bridge.config(),
        seed=seed,
    )
    lb.save_bridge(path, bridge, extra={"identity": identity})
    return sha256_file(path)


def _bridge_seal(replicas, pin="pin") -> dict[str, Any]:
    return {"execution_pin": pin, "run_id": br.PRODUCTION_BRIDGE_RUN, "replicas": replicas}


def _classifier_seal(cells, pin="c") -> dict[str, Any]:
    return {
        "execution_pin": pin, "run_id": br.PRODUCTION_CLASSIFIER_RUN,
        "stage": br.PRODUCTION_CLASSIFIER_STAGE,
        "prepared_sha256": br.PRODUCTION_PREPARED_SHA256, "cells": cells,
    }


def _sealed_fixture(root: Path) -> tuple[dict[str, Any], Any]:
    """Nine tiny real checkpoint files and the real stage49 inventory contract."""
    prepared = _prepared(root)
    cache = _cache(root, prepared)
    _, manifest, *_ = ch.bound_pretrained_cache(cache, prepared)
    replicas = []
    for seed, label in zip(br.DEFAULT_SEEDS, br.DEFAULT_BRIDGE_LABELS, strict=True):
        digest = _save_bridge(
            root / label / "bridge.pt", _bridge(seed=seed),
            prepared=prepared, manifest=manifest, seed=seed,
        )
        replicas.append(dict(seed=seed, label=label, checkpoint_sha256=digest, best_val_loss=.1))
    write_json(root / "bridge_seal.json", _bridge_seal(replicas, br.PRODUCTION_BRIDGE_PIN))
    cells = []
    for kind in br.DEFAULT_HEADS:
        for seed in br.DEFAULT_SEEDS:
            head = ch.build_classifier_head(kind=kind, d_in=4, budget=20_000, seed=seed)
            identity = ch.classifier_identity(
                head=head, budget=20_000, seed=seed, prepared_hash=prepared["prepared_sha256"],
                cache_manifest=manifest,
                train_config=ch.train_config_record(
                    epochs=20, batch_size=32, lr=.001, weight_decay=.01,
                    betas=(.9, .999), eps=1e-8, grad_clip=1., seed=seed,
                ),
            )
            relative = ch.cell_relative_dir(kind, seed)
            checkpoint = root / "classifiers" / relative / "classifier.pt"
            ch.save_classifier(checkpoint, head, {"identity": identity})
            cells.append(dict(
                kind=kind, seed=seed, dir=relative, checkpoint_sha256=sha256_file(checkpoint),
                identity=identity,
            ))
    write_json(root / "classifier_seal.json", dict(
        _classifier_seal(cells, br.PRODUCTION_CLASSIFIER_PIN),
        prepared_sha256=prepared["prepared_sha256"],
    ))
    write_json(root / "classifiers" / "report.json", {
        "schema_version": ch.REPORT_SCHEMA, "substage": "train", "n_cells": 6, "cells": cells,
    })
    cli = _load_cli()
    args = cli.build_parser().parse_args([
        "--substage", "interface", "--out", str(root / "out"),
        "--prepared", str(root / "prepared.json"), "--cache", str(cache),
        "--bridge-root", str(root), "--classifier-dir", str(root / "classifiers"),
        "--bridge-acceptance", str(root / "bridge_seal.json"),
        "--classifier-acceptance", str(root / "classifier_seal.json"),
        "--receiver-path", str(root / "receiver"), "--prompt-mode", "plain",
        "--dtype", "float32", "--max-new-tokens", "2",
    ])
    return {"prepared": prepared, "cli": cli}, args


class LayoutAndShiftTests(unittest.TestCase):
    def test_digit_layout_and_merged_token_rejected(self) -> None:
        layout = br.class_digit_layout(DigitTokenizer(), 2)
        self.assertEqual(layout["prefix_len"], 2)
        self.assertEqual(layout["digit_token_by_class"][1], 21)
        self.assertEqual(DigitTokenizer().decode([27]), "7")
        with self.assertRaisesRegex(ValueError, "merged tokens"):
            br.class_digit_layout(MergedTokenizer(), 2)

    def test_off_by_one_toy_logits_are_asymmetric(self) -> None:
        proof = br.toy_off_by_one_proof()
        self.assertEqual(proof["logit_index"], proof["label_position"] - 1)
        self.assertNotEqual(proof["correct_token"], proof["off_by_one_token"])
        self.assertEqual(br.class_logit_index(3), 2)
        with self.assertRaisesRegex(ValueError, "t must be"):
            br.class_logit_index(0)


class SealAndSelectionTests(unittest.TestCase):
    def test_seal_missing_duplicate_and_sha_fail_before_load(self) -> None:
        replicas = [
            {
                "seed": seed,
                "label": label,
                "checkpoint_sha256": "a" * 64,
                "best_val_loss": 0.1,
            }
            for seed, label in zip(br.DEFAULT_SEEDS, br.DEFAULT_BRIDGE_LABELS)
        ]
        payload = _bridge_seal(replicas)
        br.validate_bridge_acceptance(payload, expected_pin="pin")
        missing = _bridge_seal(replicas[:2])
        with self.assertRaisesRegex(ValueError, "replica matrix"):
            br.validate_bridge_acceptance(missing, expected_pin="pin")
        duplicate = _bridge_seal([replicas[0], replicas[0], replicas[2]])
        with self.assertRaisesRegex(ValueError, "duplicate seed"):
            br.validate_bridge_acceptance(duplicate, expected_pin="pin")
        cells = [
            {
                "kind": kind,
                "seed": seed,
                "dir": ch.cell_relative_dir(kind, seed),
                "checkpoint_sha256": "b" * 64,
            }
            for kind in br.DEFAULT_HEADS
            for seed in br.DEFAULT_SEEDS
        ]
        br.validate_classifier_acceptance(_classifier_seal(cells), expected_pin="c")
        cells[0] = dict(cells[0], seed=cells[1]["seed"], dir=ch.cell_relative_dir("linear", cells[1]["seed"]))
        with self.assertRaisesRegex(ValueError, "duplicate cell|matrix"):
            br.validate_classifier_acceptance(
                _classifier_seal(cells), expected_pin="c"
            )

    def test_wrong_sha_does_not_deserialize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = _prepared(root)
            cache = _cache(root, prepared)
            _, manifest, *_ = ch.bound_pretrained_cache(cache, prepared)
            replicas = []
            for seed, label in zip(br.DEFAULT_SEEDS, br.DEFAULT_BRIDGE_LABELS, strict=True):
                path = root / label / "bridge.pt"
                digest = _save_bridge(path, _bridge(), prepared=prepared, manifest=manifest, seed=seed)
                replicas.append(dict(seed=seed, label=label, checkpoint_sha256=digest, best_val_loss=.1))
            replicas[-1]["checkpoint_sha256"] = "0" * 64
            acceptance = _bridge_seal(replicas)
            with patch.object(br, "load_bridge", side_effect=AssertionError("loaded")) as loader:
                with self.assertRaisesRegex(ValueError, "checkpoint sha256"):
                    br.qualify_bridges(
                        bridge_root=root,
                        acceptance=acceptance,
                        expected_pin="pin",
                        prepared_hash=prepared["prepared_sha256"],
                        cache_manifest=manifest,
                        receiver_digest=TOY_RECEIVER,
                        prompt_mode="plain",
                    )
            loader.assert_not_called()

    def test_classifier_wrong_sha_does_not_load_stage49(self) -> None:
        cli = _load_cli()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cells = []
            for kind in br.DEFAULT_HEADS:
                for seed in br.DEFAULT_SEEDS:
                    relative = ch.cell_relative_dir(kind, seed)
                    path = root / relative / "classifier.pt"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"not-a-checkpoint")
                    cells.append(
                        {
                            "kind": kind,
                            "seed": seed,
                            "dir": relative,
                            "checkpoint_sha256": sha256_file(path),
                        }
                    )
            cells[-1]["checkpoint_sha256"] = "0" * 64
            with patch.object(cli, "_load_stage49", side_effect=AssertionError("loaded 49")):
                with self.assertRaisesRegex(ValueError, "checkpoint sha256"):
                    cli._qualify_heads(
                        classifier_dir=root,
                        classifier_acceptance=cells,
                        prepared_hash="x",
                        cache_manifest={},
                        d_in=4,
                    )

    def test_interface_rejects_test_role_and_over_32(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prepared = _prepared(Path(tmp))
            val = lb.prepared_records(prepared, "val")
            train = lb.prepared_records(prepared, "train")
            with self.assertRaisesRegex(ValueError, "non-train"):
                br.select_interface_records(val, max_interface_records=2)
            with self.assertRaisesRegex(ValueError, "1..32"):
                br.select_interface_records(train, max_interface_records=33)
            selected = br.select_interface_records(train, max_interface_records=32)
            self.assertEqual(selected["n_mechanical"], 2)
            self.assertEqual(selected["mechanical_accessions"], ["t1", "t2"])


class TeacherForceAndGreedyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prepared = _prepared(self.root)
        self.cache = _cache(self.root, self.prepared)
        self.features, self.manifest, self.train, self.val, self.test = ch.bound_pretrained_cache(
            self.cache, self.prepared
        )
        self.train_feat, self.val_feat, self.test_feat = ch.split_cache_features(
            self.features, self.train, self.val, self.test
        )
        self.tokenizer = DigitTokenizer()
        self.receiver = TinyCausalLM()
        self.layout = br.class_digit_layout(self.tokenizer, self.tokenizer.eos_token_id)
        self.bridge = _bridge()

    def test_exact_masks_causal_shift_and_native_vs_fp32_paths(self) -> None:
        # Independent expected layout: two soft tokens, three prompt tokens,
        # then two common JSON tokens before the class digit. Do not use the
        # implementation's index helper to construct the asymmetric logits.
        answer_start, digit_position = 5, 7
        answers = [lb._answer_ids(self.tokenizer, row.target_json, 2) for row in self.train]
        lengths = [len(answer) for answer in answers]
        self.assertNotEqual(*lengths)
        expected_labels = torch.full((2, answer_start + max(lengths)), lb.IGNORE_INDEX)
        for i, answer in enumerate(answers):
            self.assertEqual(answer[-1], self.tokenizer.eos_token_id)
            self.assertNotEqual(answer[-2], self.tokenizer.eos_token_id)
            expected_labels[i, answer_start:answer_start + len(answer)] = torch.tensor(answer)
        packed_batches = []
        logits_batches = []

        def capture_pack(**kwargs):
            packed = lb.inject_soft_prefix(**kwargs)
            packed_batches.append(packed)
            return packed

        for output_dtype in (torch.bfloat16, torch.float32):
            with self.subTest(output_dtype=output_dtype):
                # Logits dtype, not the BF16 embedding weights, defines legacy CE.
                receiver = TinyCausalLM().to(dtype=torch.bfloat16)

                def asymmetric(inputs_embeds, **_):
                    logits = torch.zeros((*inputs_embeds.shape[:2], 64), dtype=output_dtype)
                    for i, record in enumerate(self.train):
                        logits[i, digit_position - 1, 20 + record.ec_class] = 7.25
                        logits[i, digit_position, 27] = 9.5
                    logits_batches.append(logits)
                    return SimpleNamespace(logits=logits)

                with patch.object(receiver, "forward", side_effect=asymmetric), patch.object(
                    br, "inject_soft_prefix", side_effect=capture_pack
                ):
                    result = br.teacher_forced_decomposition(
                        bridge=self.bridge, receiver=receiver, tokenizer=self.tokenizer,
                        features=self.train_feat, records=self.train, layout=self.layout,
                        prompt_mode="plain", device="cpu", batch_size=2, max_prompt_tokens=64,
                    )
                packed = packed_batches[-1]
                self.assertTrue(torch.equal(packed["labels"], expected_labels))
                self.assertEqual(int((packed["labels"] == lb.IGNORE_INDEX).sum()),
                                 2 * answer_start + 2 * max(lengths) - sum(lengths))
                self.assertEqual(int((packed["attention_mask"] == 0).sum()),
                                 2 * max(lengths) - sum(lengths))
                for i, row in enumerate(result["records"]):
                    self.assertEqual(row["total_token_count"], lengths[i])
                    self.assertEqual(row["remainder_token_count"], lengths[i] - 1)
                    self.assertEqual(row["digit_predicted_class"], self.train[i].ec_class)
                self.assertEqual(result["digit_logit_index"], digit_position - 1)
                self.assertEqual(result["fp32_unreduced"]["digit_accuracy"], 1.)
                self.assertEqual(result["legacy_native"]["token_count"], sum(lengths))
                self.assertEqual(result["fp32_unreduced"]["total_token_count"], sum(lengths))
                logits = logits_batches[-1]
                native = F.cross_entropy(
                    logits[:, :-1].reshape(-1, 64), expected_labels[:, 1:].reshape(-1),
                    ignore_index=lb.IGNORE_INDEX,
                )
                fp32 = F.cross_entropy(
                    logits.float()[:, :-1].reshape(-1, 64), expected_labels[:, 1:].reshape(-1),
                    ignore_index=lb.IGNORE_INDEX, reduction="none",
                ).double().sum().item()
                self.assertEqual(result["logits_dtype"], str(output_dtype))
                self.assertEqual(result["legacy_native"]["token_weighted_nll"], native.item())
                self.assertAlmostEqual(result["fp32_unreduced"]["total_ce_sum"], fp32)
                self.assertAlmostEqual(result["fp32_unreduced"]["digit_plus_rest_sum"], fp32)
                self.assertEqual(result["fp32_unreduced"]["token_ce_dtype"], "torch.float32")

    def test_device_placement_precedes_public_forwards(self) -> None:
        def checked(_module, inputs):
            self.assertGreater(move.call_count, 0)
            self.assertEqual(inputs[0].device, next(self.bridge.parameters()).device)
            self.assertEqual(next(self.bridge.parameters()).dtype, torch.float32)

        with patch.object(self.bridge, "to", wraps=self.bridge.to) as move:
            hook = self.bridge.register_forward_pre_hook(checked)
            try:
                br.teacher_forced_decomposition(
                    bridge=self.bridge, receiver=self.receiver, tokenizer=self.tokenizer,
                    features=self.train_feat, records=self.train, layout=self.layout,
                    prompt_mode="plain", device="cpu", batch_size=2, max_prompt_tokens=64,
                )
                move.reset_mock()
                br.suffix_invariance_check(
                    bridge=self.bridge, receiver=self.receiver, tokenizer=self.tokenizer,
                    feature_row=self.train_feat[0], record=self.train[0], layout=self.layout,
                    prompt_mode="plain", device="cpu", max_prompt_tokens=64,
                )
            finally:
                hook.remove()
        head = ch.build_classifier_head(kind="linear", d_in=4, budget=20_000, seed=1)
        with patch.object(head, "to", wraps=head.to) as head_move:
            br.direct_classifier_metrics(head=head, features=self.val_feat, records=self.val,
                                         device="cpu", batch_size=2)
        head_move.assert_called_with("cpu")
        self.assertTrue(all(p.dtype == torch.float32 for p in head.parameters()))

    def test_all_public_forwards_reject_test_and_shared_invalid_inputs(self) -> None:
        common: dict[str, Any] = dict(
            bridge=self.bridge, receiver=self.receiver, tokenizer=self.tokenizer,
            features=self.test_feat, records=self.test, prompt_mode="plain",
            device="cpu", batch_size=2, max_prompt_tokens=64,
        )
        head = ch.build_classifier_head(kind="linear", d_in=4, budget=20_000, seed=1)
        calls = [
            lambda: br.teacher_forced_decomposition(**common, layout=self.layout),
            lambda: br.free_greedy_predictions(**common, max_new_tokens=2),
            lambda: br.direct_classifier_metrics(head=head, features=self.test_feat,
                                                 records=self.test, device="cpu", batch_size=2),
            lambda: br.suffix_invariance_check(
                bridge=self.bridge, receiver=self.receiver, tokenizer=self.tokenizer,
                feature_row=self.test_feat[0], record=self.test[0], layout=self.layout,
                prompt_mode="plain", device="cpu", max_prompt_tokens=64,
            ),
        ]
        with patch.object(self.receiver, "forward", side_effect=AssertionError("forwarded test")):
            for call in calls:
                with self.assertRaisesRegex(ValueError, "never test"):
                    call()
        for features, records, batch, message in (
            (self.train_feat, self.train, 0, "positive"),
            (self.train_feat[:1], self.train, 1, "misaligned"),
            (self.train_feat, [self.train[0], self.val[0]], 1, "non-train"),
            (np.tile(self.train_feat[:1], (33, 1)), self.train[:1] * 33, 1, "32 rows"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                br.validate_forward_inputs(features, records, batch)

    def test_nonfinite_greedy_and_extreme_ce_fail_without_leaked_hook(self) -> None:
        common: dict[str, Any] = dict(
            bridge=self.bridge, receiver=self.receiver, tokenizer=self.tokenizer,
            features=self.train_feat, records=self.train, prompt_mode="plain",
            device="cpu", batch_size=2, max_prompt_tokens=64,
        )
        before_hooks = len(self.receiver._forward_hooks)

        def bad_logits(inputs_embeds, **_):
            return SimpleNamespace(logits=torch.full((*inputs_embeds.shape[:2], 64), float("nan")))

        with patch.object(self.receiver, "forward", side_effect=bad_logits):
            with self.assertRaisesRegex(ValueError, "non-finite greedy logits"):
                br.free_greedy_predictions(**common, max_new_tokens=2)
        self.assertEqual(len(self.receiver._forward_hooks), before_hooks)

        def extreme_logits(inputs_embeds, **_):
            logits = torch.full((*inputs_embeds.shape[:2], 64), -3e38)
            logits[:, :, 63] = 3e38
            self.assertTrue(bool(torch.isfinite(logits).all()))
            return SimpleNamespace(logits=logits)

        with patch.object(self.receiver, "forward", side_effect=extreme_logits):
            with self.assertRaisesRegex(ValueError, "non-finite.*NLL"):
                br.teacher_forced_decomposition(**common, layout=self.layout)
            # Even if a native kernel returns a finite scalar, FP32 unreduced
            # CE must independently reject overflow before sums/reconstruction.
            with patch.object(br, "shifted_cross_entropy", return_value=torch.tensor(0.)):
                with self.assertRaisesRegex(ValueError, "non-finite FP32 token CE"):
                    br.teacher_forced_decomposition(**common, layout=self.layout)

    def test_nan_features_rejected(self) -> None:
        bad = self.train_feat.copy()
        bad[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            br.teacher_forced_decomposition(
                bridge=self.bridge,
                receiver=self.receiver,
                tokenizer=self.tokenizer,
                features=bad,
                records=self.train,
                layout=self.layout,
                prompt_mode="plain",
                device="cpu",
                batch_size=2,
                max_prompt_tokens=64,
            )

    def test_greedy_ignores_gold_and_sequence_and_does_not_feed_answers(self) -> None:
        with patch.object(lb, "inject_soft_prefix", wraps=lb.inject_soft_prefix) as mocked:
            first = br.free_greedy_predictions(
                bridge=self.bridge,
                receiver=self.receiver,
                tokenizer=self.tokenizer,
                features=self.train_feat,
                records=self.train,
                prompt_mode="plain",
                device="cpu",
                batch_size=2,
                max_prompt_tokens=64,
                max_new_tokens=8,
            )
        for call in mocked.call_args_list:
            self.assertIsNone(call.kwargs.get("answer_ids"))
        mutated = []
        for row in self.train:
            payload = row.to_dict()
            payload["sequence"] = "WWWWWWWWWW"
            payload["ec_class"] = 7 if row.ec_class != 7 else 6
            payload["class_name"] = "translocase"
            payload["target_json"] = lb.target_json(payload["ec_class"])
            mutated.append(lb.PreparedRecord.from_dict(payload))
        second = br.free_greedy_predictions(
            bridge=self.bridge,
            receiver=self.receiver,
            tokenizer=self.tokenizer,
            features=self.train_feat,
            records=mutated,
            prompt_mode="plain",
            device="cpu",
            batch_size=2,
            max_prompt_tokens=64,
            max_new_tokens=8,
        )
        self.assertEqual(
            [row["text"] for row in first["rows"]],
            [row["text"] for row in second["rows"]],
        )

    def test_suffix_invariance_and_frozen_bytes(self) -> None:
        before = lb.parameter_fingerprint(self.bridge)
        receiver_before = lb.parameter_fingerprint(self.receiver)
        br.suffix_invariance_check(
            bridge=self.bridge,
            receiver=self.receiver,
            tokenizer=self.tokenizer,
            feature_row=self.train_feat[0],
            record=self.train[0],
            layout=self.layout,
            prompt_mode="plain",
            device="cpu",
            max_prompt_tokens=64,
        )
        br.teacher_forced_decomposition(
            bridge=self.bridge,
            receiver=self.receiver,
            tokenizer=self.tokenizer,
            features=self.train_feat[:1],
            records=self.train[:1],
            layout=self.layout,
            prompt_mode="plain",
            device="cpu",
            batch_size=1,
            max_prompt_tokens=64,
        )
        self.assertEqual(before, lb.parameter_fingerprint(self.bridge))
        self.assertEqual(receiver_before, lb.parameter_fingerprint(self.receiver))
        self.assertTrue(all(parameter.grad is None for parameter in self.bridge.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in self.receiver.parameters()))

    def test_direct_classifier_does_not_run_handoff(self) -> None:
        head = ch.build_classifier_head(kind="linear", d_in=4, budget=20_000, seed=1)
        with patch.object(ch, "generate_handoff", side_effect=AssertionError("handoff")):
            result = br.direct_classifier_metrics(
                head=head,
                features=self.val_feat,
                records=self.val,
                device="cpu",
                batch_size=2,
            )
        self.assertFalse(result["receiver_handoff_ran"])
        self.assertEqual(result["n"], 2)


class LifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        fixture, self.args = _sealed_fixture(self.root)
        self.cli = fixture["cli"]
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for module in (self.cli, br):
            self.stack.enter_context(patch.object(
                module, "PRODUCTION_PREPARED_SHA256", fixture["prepared"]["prepared_sha256"]
            ))
        self.stack.enter_context(patch.object(self.cli, "DEFAULT_BUDGET", 20_000))
        self.stack.enter_context(patch.object(
            self.cli, "PRODUCTION_RECEIVER_CONTENT_DIGEST", TOY_RECEIVER["content_digest"]
        ))
        self.digest_loader = self.stack.enter_context(patch.object(
            self.cli, "checkpoint_content_digest", return_value=dict(TOY_RECEIVER)
        ))
        self.receiver = TinyCausalLM()
        self.receiver_loader = self.stack.enter_context(patch.object(
            self.cli, "load_receiver", return_value=(self.receiver, DigitTokenizer())
        ))

    def test_last_of_nine_wrong_sha_prevents_every_model_loader(self) -> None:
        seal = br.load_acceptance(self.args.classifier_acceptance)
        last = seal["cells"][-1]
        checkpoint = self.args.classifier_dir / last["dir"] / "classifier.pt"
        checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
        with patch.object(br, "load_bridge") as bridge_loader, patch.object(
            self.cli, "_load_stage49"
        ) as stage_loader, patch.object(ch, "load_classifier") as head_loader:
            with self.assertRaisesRegex(ValueError, "checkpoint sha256"):
                self.cli._load_context(self.args)
        for loader in (bridge_loader, stage_loader, head_loader, self.receiver_loader):
            loader.assert_not_called()

    def test_invalid_preflight_inputs_stop_before_load(self) -> None:
        for field, value in (
            ("batch_size", 0), ("max_new_tokens", -1), ("max_prompt_tokens", 0),
            ("max_interface_records", 33), ("cache", None),
        ):
            with self.subTest(field=field), patch.object(self.args, field, value):
                with self.assertRaises(ValueError):
                    self.cli._load_context(self.args)
        original = br.load_acceptance(self.args.classifier_acceptance)
        for field, value in (("prepared_sha256", None), ("run_id", "wrong"), ("stage", "wrong")):
            write_json(self.args.classifier_acceptance, dict(original, **{field: value}))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                self.cli._load_context(self.args)
        write_json(self.args.classifier_acceptance, original)
        bridge_seal = br.load_acceptance(self.args.bridge_acceptance)
        for loss in (True, float("nan"), float("inf")):
            bridge_seal["replicas"][-1]["best_val_loss"] = loss
            # JSON NaN is deliberately parsed and explicitly rejected at validation.
            self.args.bridge_acceptance.write_text(json.dumps(bridge_seal))
            with self.subTest(loss=loss), self.assertRaisesRegex(ValueError, "best_val_loss"):
                self.cli._load_context(self.args)
        self.receiver_loader.assert_not_called()

    def test_receiver_digest_rejected_before_load(self) -> None:
        self.digest_loader.return_value = {"content_digest": "wrong"}
        with patch.object(br, "load_bridge") as bridge_loader, patch.object(
            self.cli, "_load_stage49"
        ) as stage_loader:
            with self.assertRaisesRegex(ValueError, "receiver content digest"):
                self.cli._load_context(self.args)
        self.receiver_loader.assert_not_called()
        bridge_loader.assert_not_called()
        stage_loader.assert_not_called()

    def test_real_six_head_interface_and_pre_forward_mutation_detection(self) -> None:
        # Real tiny checkpoint loaders, bridge forwards, six head forwards and
        # diagnostic CLI path. Only the unavailable receiver assets are replaced.
        with patch.object(self.cli, "direct_classifier_metrics", wraps=br.direct_classifier_metrics) as direct:
            report = self.cli.run_interface(self.args)
        self.assertEqual(direct.call_count, 6)
        self.assertTrue(report["gates"]["all_six_heads_exercised"])
        self.assertEqual(len(report["fingerprints"]), 10)
        self.assertEqual(len(report["input_files"]), 15)
        for item in report["heads"]:
            self.assertEqual(item["direct"]["n"], 2)
            self.assertEqual(item["direct"]["role"], "train")
            self.assertFalse(item["direct"]["receiver_handoff_ran"])
            self.assertGreater(item["actual_parameters"], 0)
        for proof in report["fingerprints"].values():
            self.assertEqual(proof["before"], proof["after"])
        # Mutation during the last real greedy receiver call must be caught by
        # the original baseline, not a freshly computed post-forward baseline.
        context = self.cli._load_context(self.args)
        calls = 0

        def mutate(_module, _args, _output):
            nonlocal calls
            calls += 1
            if calls == 1:
                with torch.no_grad():
                    self.receiver.out.bias[0] += .01

        def final_greedy(**kwargs):
            hook = self.receiver.register_forward_hook(mutate)
            try:
                return br.free_greedy_predictions(**kwargs)
            finally:
                hook.remove()

        original_greedy = br.free_greedy_predictions
        counter = 0

        def on_last_bridge(**kwargs):
            nonlocal counter
            counter += 1
            return final_greedy(**kwargs) if counter == 3 else original_greedy(**kwargs)

        with patch.object(self.cli, "_load_context", return_value=context), patch.object(
            self.cli, "free_greedy_predictions", side_effect=on_last_bridge
        ), self.assertRaisesRegex(RuntimeError, "receiver parameter fingerprint changed"):
            self.cli.run_interface(self.args)

    def test_raw_input_and_receiver_disk_digest_rechecked(self) -> None:
        context = self.cli._load_context(self.args)
        report_path = context["input_paths"]["classifier_train_report"]
        report_path.write_bytes(report_path.read_bytes() + b"\n")
        with self.assertRaisesRegex(RuntimeError, "input file changed: classifier_train_report"):
            self.cli._verify_unchanged(context)
        # Recompute a new fixture baseline only for the independent second case.
        context = self.cli._load_context(self.args)
        self.digest_loader.return_value = {"content_digest": "changed-on-disk"}
        with self.assertRaisesRegex(RuntimeError, "receiver content digest changed"):
            self.cli._verify_unchanged(context)


class QualifyAndReportTests(unittest.TestCase):
    def test_qualify_keeps_pre_freeze_parameter_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = _prepared(root)
            cache = _cache(root, prepared)
            _, manifest, *_ = ch.bound_pretrained_cache(cache, prepared)
            replicas = []
            for seed, label in zip(br.DEFAULT_SEEDS, br.DEFAULT_BRIDGE_LABELS):
                bridge = _bridge(seed=seed)
                actual = bridge.trainable_parameter_count()
                path = root / label / "bridge.pt"
                digest = _save_bridge(
                    path, bridge, prepared=prepared, manifest=manifest, seed=seed
                )
                replicas.append(
                    {
                        "seed": seed,
                        "label": label,
                        "checkpoint_sha256": digest,
                        "best_val_loss": 0.1,
                        "actual": actual,
                    }
                )
            qualified = br.qualify_bridges(
                bridge_root=root,
                acceptance=_bridge_seal(replicas),
                expected_pin="pin",
                prepared_hash=prepared["prepared_sha256"],
                cache_manifest=manifest,
                receiver_digest=TOY_RECEIVER,
                prompt_mode="plain",
            )
            self.assertEqual(len(qualified), 3)
            for item, replica in zip(qualified, replicas, strict=True):
                self.assertEqual(item["actual_parameters"], replica["actual"])
                self.assertGreater(item["actual_parameters"], 0)
                self.assertFalse(
                    any(parameter.requires_grad for parameter in item["bridge"].parameters())
                )
                self.assertEqual(sha256_file(root / replica["label"] / "bridge.pt"), replica["checkpoint_sha256"])

    def test_nonfinite_bridge_state_rejected_after_sha(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepared = _prepared(root)
            cache = _cache(root, prepared)
            _, manifest, *_ = ch.bound_pretrained_cache(cache, prepared)
            replicas = []
            for seed, label in zip(br.DEFAULT_SEEDS, br.DEFAULT_BRIDGE_LABELS):
                bridge = _bridge(seed=seed)
                path = root / label / "bridge.pt"
                _save_bridge(path, bridge, prepared=prepared, manifest=manifest, seed=seed)
                payload = torch.load(path, map_location="cpu")
                payload["state_dict"]["fc1.weight"][0, 0] = float("nan")
                torch.save(payload, path)
                replicas.append(
                    {
                        "seed": seed,
                        "label": label,
                        "checkpoint_sha256": sha256_file(path),
                        "best_val_loss": 0.1,
                    }
                )
            with self.assertRaisesRegex(ValueError, "non-finite bridge state"):
                br.qualify_bridges(
                    bridge_root=root,
                    acceptance=_bridge_seal(replicas),
                    expected_pin="pin",
                    prepared_hash=prepared["prepared_sha256"],
                    cache_manifest=manifest,
                    receiver_digest=TOY_RECEIVER,
                    prompt_mode="plain",
                )

    def test_write_report_json_sidecar_and_run_refuses_output(self) -> None:
        cli = _load_cli()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            digest = br.write_report_json(out, {"ok": True, "schema_version": "x"})
            self.assertEqual(digest, sha256_file(out / "report.json"))
            sidecar = (out / "report.json.sha256").read_text(encoding="utf-8")
            self.assertEqual(sidecar, f"{digest}  report.json\n")
            with self.assertRaises(FileExistsError):
                br.refuse_existing_output(out)
            empty = Path(tmp) / "empty"
            empty.mkdir()
            br.refuse_existing_output(empty)
            args = cli.build_parser().parse_args(
                ["--substage", "run", "--out", str(out), "--device", "cpu"]
            )
            for run in (cli.run_run, cli.run_interface):
                with patch.object(cli, "_load_context") as loader:
                    with self.assertRaises(FileExistsError):
                        run(args)
                    loader.assert_not_called()

    def test_cli_defaults(self) -> None:
        cli = _load_cli()
        args = cli.build_parser().parse_args(["--substage", "interface", "--out", "x"])
        self.assertEqual(args.prompt_mode, "chat")
        self.assertEqual(args.dtype, "bfloat16")
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.max_prompt_tokens, 1024)
        self.assertEqual(args.max_new_tokens, 64)
        self.assertEqual(args.max_interface_records, 32)


if __name__ == "__main__":
    unittest.main()
