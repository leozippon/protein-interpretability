"""CPU tests for the classifier → explicit-label → frozen-receiver handoff."""

from __future__ import annotations

import inspect
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

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import src.transfer.classifier_handoff as ch
from src.transfer import latent_bridge as lb
from src.transfer.io import write_json
from src.transfer.sequence_description import SequenceDescriptionRecord


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
    records_path.write_text("".join(json.dumps(row.to_dict()) + "\n" for row in raw), encoding="utf-8")
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


def _donor_manifest(prepared: dict[str, Any], **changes: Any) -> dict[str, Any]:
    manifest = {
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
        "dtype": "bfloat16",
    }
    manifest.update(changes)
    return manifest


def _cache(root: Path, prepared: dict[str, Any], *, d: int = 8, seed: int = 0) -> Path:
    rows = lb.all_prepared_records(prepared)
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(len(rows), d)).astype(np.float32)
    path = root / "donor_cache.npz"
    lb.save_feature_cache(path, features, rows, _donor_manifest(prepared))
    return path


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, **_):
        return {"input_ids": [1, 4, 5]}

    def decode(self, ids, **_):
        return "invalid JSON"


class ClassAwareTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, **_):
        marker = "The classifier predicted "
        blob = text[text.index(marker) + len(marker) :]
        payload = json.loads(blob[: blob.index("}") + 1])
        class_id = payload["class"]
        return {"input_ids": list(range(1, class_id + 2))}

    def decode(self, ids, **_):
        return "invalid JSON"


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


class HeadBudgetTests(unittest.TestCase):
    def test_linear_and_mlp_match_frozen_budget(self) -> None:
        linear = ch.build_classifier_head(kind="linear", d_in=1536, budget=3_000_000, seed=0)
        self.assertEqual(linear.kind, "linear")
        self.assertEqual(linear.trainable_parameter_count(), 10_759)
        self.assertEqual(tuple(linear(torch.zeros(2, 1536)).shape), (2, 7))
        mlp = ch.build_classifier_head(kind="mlp", d_in=1536, budget=3_000_000, seed=0)
        self.assertEqual(mlp.kind, "mlp")
        self.assertEqual(mlp.hidden, 1943)
        self.assertEqual(mlp.trainable_parameter_count(), 2_999_999)
        self.assertEqual(tuple(mlp(torch.zeros(2, 1536)).shape), (2, 7))
        hidden, actual = ch.mlp_hidden_width(d_in=1536, n_classes=7, budget=3_000_000)
        self.assertEqual((hidden, actual), (1943, 2_999_999))

    def test_seed_covers_init_and_restores_rng(self) -> None:
        state = torch.random.get_rng_state().clone()
        first = ch.build_classifier_head(kind="mlp", d_in=8, budget=500, seed=7)
        second = ch.build_classifier_head(kind="mlp", d_in=8, budget=500, seed=7)
        other = ch.build_classifier_head(kind="mlp", d_in=8, budget=500, seed=8)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertEqual(lb.parameter_fingerprint(first), lb.parameter_fingerprint(second))
        self.assertNotEqual(lb.parameter_fingerprint(first), lb.parameter_fingerprint(other))
        for parameter in first.input_norm.parameters():
            self.assertFalse(parameter.requires_grad)


class TrainAndIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prepared = _prepared(self.root)
        self.cache = _cache(self.root, self.prepared, d=8)
        self.features, self.manifest, self.train, self.val, self.test = ch.bound_pretrained_cache(
            self.cache, self.prepared
        )
        self.train_feat, self.val_feat, self.test_feat = ch.split_cache_features(
            self.features, self.train, self.val, self.test
        )

    def test_best_val_clone_and_first_tie(self) -> None:
        head = ch.build_classifier_head(kind="linear", d_in=8, budget=20_000, seed=1)
        snap = ch.cpu_state_clone(head)
        weight = dict(head.named_parameters())["fc.weight"]
        self.assertEqual(snap["fc.weight"].device.type, "cpu")
        with torch.no_grad():
            weight.add_(1)
        self.assertFalse(torch.equal(snap["fc.weight"], weight.detach().cpu()))
        head = ch.build_classifier_head(kind="linear", d_in=8, budget=20_000, seed=1)
        result = ch.train_classifier_head(
            head=head,
            train_features=self.train_feat,
            train_records=self.train,
            val_features=self.val_feat,
            val_records=self.val,
            device="cpu",
            epochs=3,
            batch_size=2,
            lr=0.0,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8,
            grad_clip=1.0,
            seed=1,
        )
        self.assertEqual(result["best_epoch"], 1)
        self.assertFalse(result["test_metrics_computed"])
        self.assertTrue(result["saw_nonzero_head_grad"])

    def test_restores_earlier_minimum_after_real_training_deteriorates(self) -> None:
        head = ch.build_classifier_head(kind="linear", d_in=8, budget=20_000, seed=1)
        train = lb.filter_records([_record(accession="fit", split="fit", ec=("1.1.1.1",))])[0]
        val = lb.filter_records([_record(accession="val", split="eval", ec=("2.1.1.1",))])[0]
        features = np.array([[1, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
        result = ch.train_classifier_head(
            head=head, train_features=features, train_records=train,
            val_features=features, val_records=val, device="cpu", epochs=3,
            batch_size=1, lr=0.03, weight_decay=0.0, betas=(0.9, 0.999),
            eps=1e-8, grad_clip=1.0, seed=1,
        )
        self.assertEqual(result["best_epoch"], 1)
        self.assertGreater(result["history"][-1]["val_ce"], result["best_val_ce"])
        with torch.no_grad():
            restored = nn.functional.cross_entropy(head(torch.from_numpy(features)), torch.tensor([1]))
        self.assertAlmostEqual(float(restored), result["best_val_ce"], places=6)

    def test_nonfinite_state_and_logits_cannot_become_valid_classes(self) -> None:
        head = ch.build_classifier_head(kind="linear", d_in=8, budget=20_000, seed=1)
        checkpoint = self.root / "nonfinite.pt"
        ch.save_classifier(checkpoint, head, extra={})
        clean = torch.load(checkpoint, map_location="cpu", weights_only=True)
        for bad in (float("nan"), float("inf")):
            with self.subTest(bad=bad):
                payload = {**clean, "state_dict": {k: v.clone() for k, v in clean["state_dict"].items()}}
                payload["state_dict"]["fc.weight"][0, 0] = bad
                torch.save(payload, checkpoint)
                with self.assertRaisesRegex(ValueError, "non-finite classifier state"):
                    ch.load_classifier(checkpoint)
                features = self.test_feat.copy()
                features[0, 0] = bad
                with self.assertRaisesRegex(ValueError, "non-finite classifier logits"):
                    ch.predict_classes(head=head, features=features, device="cpu", batch_size=2)
        with torch.no_grad():
            dict(head.named_parameters())["fc.weight"].fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "non-finite classifier logits"):
            ch.predict_classes(head=head, features=self.test_feat, device="cpu", batch_size=2)

    def test_duplicate_heads_are_rejected_before_writing_cells(self) -> None:
        out = self.root / "duplicate"
        with self.assertRaisesRegex(ValueError, "duplicate heads"):
            ch.train_all_cells(
                train_features=self.train_feat, train_records=self.train,
                val_features=self.val_feat, val_records=self.val,
                d_in=8, budget=20_000, heads=["linear", "linear"], seeds=[1],
                device="cpu", epochs=1, batch_size=2, lr=0.001, weight_decay=0.01,
                betas=(0.9, 0.999), eps=1e-8, grad_clip=1.0,
                prepared_hash=self.prepared["prepared_sha256"], cache_manifest=self.manifest,
                out_dir=out,
            )
        self.assertFalse(out.exists())

    def test_train_entry_rejects_test(self) -> None:
        sig = inspect.signature(ch.train_classifier_head)
        self.assertEqual(set(sig.parameters) & {"test", "test_features", "test_records"}, set())
        head = ch.build_classifier_head(kind="linear", d_in=8, budget=20_000, seed=0)
        extra = {"test_features": self.test_feat, "test_records": self.test}
        with self.assertRaises(TypeError):
            ch.train_classifier_head(
                head=head,
                train_features=self.train_feat,
                train_records=self.train,
                val_features=self.val_feat,
                val_records=self.val,
                device="cpu",
                epochs=1,
                batch_size=2,
                lr=1e-3,
                weight_decay=0.01,
                betas=(0.9, 0.999),
                eps=1e-8,
                grad_clip=1.0,
                seed=0,
                **extra,
            )
        with self.assertRaisesRegex(ValueError, "non-train"):
            ch.train_classifier_head(
                head=head,
                train_features=self.test_feat,
                train_records=self.test,
                val_features=self.val_feat,
                val_records=self.val,
                device="cpu",
                epochs=1,
                batch_size=2,
                lr=1e-3,
                weight_decay=0.01,
                betas=(0.9, 0.999),
                eps=1e-8,
                grad_clip=1.0,
                seed=0,
            )

    def test_identity_and_old_bridge_rejected(self) -> None:
        head = ch.build_classifier_head(kind="linear", d_in=8, budget=20_000, seed=20260908)
        result = ch.train_classifier_head(
            head=head,
            train_features=self.train_feat,
            train_records=self.train,
            val_features=self.val_feat,
            val_records=self.val,
            device="cpu",
            epochs=1,
            batch_size=2,
            lr=1e-3,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8,
            grad_clip=1.0,
            seed=20260908,
        )
        identity = ch.classifier_identity(
            head=head,
            budget=20_000,
            seed=20260908,
            prepared_hash=self.prepared["prepared_sha256"],
            cache_manifest=self.manifest,
            train_config=result["train_config"],
        )
        ckpt = self.root / "classifier.pt"
        ch.save_classifier(ckpt, head, extra={"identity": identity})
        loaded, payload = ch.load_classifier(ckpt)
        ch.validate_classifier_identity(payload, identity)
        self.assertEqual(lb.parameter_fingerprint(loaded), lb.parameter_fingerprint(head))
        wrong_seed_config = dict(result["train_config"])
        wrong_seed_config["seed"] = 20260909
        with self.assertRaisesRegex(ValueError, "identity"):
            ch.validate_classifier_identity(
                payload,
                ch.classifier_identity(
                    head=head,
                    budget=20_000,
                    seed=20260909,
                    prepared_hash=self.prepared["prepared_sha256"],
                    cache_manifest=self.manifest,
                    train_config=wrong_seed_config,
                ),
            )
        changed_cache = dict(self.manifest)
        changed_cache["feature_sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "identity"):
            ch.validate_classifier_identity(
                payload,
                ch.classifier_identity(
                    head=head,
                    budget=20_000,
                    seed=20260908,
                    prepared_hash=self.prepared["prepared_sha256"],
                    cache_manifest=changed_cache,
                    train_config=result["train_config"],
                ),
            )
        mlp = ch.build_classifier_head(kind="mlp", d_in=8, budget=20_000, seed=20260908)
        with self.assertRaisesRegex(ValueError, "identity"):
            ch.validate_classifier_identity(
                payload,
                ch.classifier_identity(
                    head=mlp,
                    budget=20_000,
                    seed=20260908,
                    prepared_hash=self.prepared["prepared_sha256"],
                    cache_manifest=self.manifest,
                    train_config=result["train_config"],
                ),
            )
        kmer = self.root / "kmer.npz"
        lb.save_feature_cache(
            kmer,
            self.features,
            lb.all_prepared_records(self.prepared),
            {
                "data_hash": self.prepared["prepared_sha256"],
                "donor_kind": "kmer",
                "checkpoint_digest": {"content_digest": "kmer_3_raw_frequency"},
                "layer": None,
                "pooling": None,
                "rendering": "sequence_only",
            },
        )
        with self.assertRaises(ValueError):
            ch.bound_pretrained_cache(kmer, self.prepared)
        bridge = lb.PoolingMLPBridge(d_in=8, d_recv=8, k=2, hidden=4, output_scale=1.0)
        lb.save_bridge(self.root / "bridge.pt", bridge, extra={"identity": {}})
        with self.assertRaisesRegex(ValueError, "latent-bridge"):
            ch.load_classifier(self.root / "bridge.pt")


class HandoffLeakTests(unittest.TestCase):
    def test_prompt_depends_on_pred_not_gold_or_sequence(self) -> None:
        sig = inspect.signature(ch.generate_handoff)
        forbidden = {"record", "records", "gold", "sequence", "ec_class", "prepared", "oracle"}
        self.assertEqual(forbidden & set(sig.parameters), set())
        self.assertNotIn("record", inspect.signature(ch.handoff_prompt).parameters)
        prompt_three = ch.handoff_prompt(3)
        self.assertEqual(prompt_three, ch.handoff_prompt(3))
        self.assertNotEqual(prompt_three, ch.handoff_prompt(4))
        self.assertIn(f"The classifier predicted {lb.target_json(3)}", prompt_three)
        self.assertNotIn(f"The classifier predicted {lb.target_json(4)}", prompt_three)
        gold_one = lb.filter_records([_record(ec=("1.1.1.1",), sequence="AAAAAAAACD")])[0][0]
        gold_seven = lb.filter_records([_record(ec=("7.1.1.1",), sequence="CCCCCCCCCD")])[0][0]
        receiver = TinyReceiver()
        lb.freeze_module(receiver)
        texts = ch.generate_handoff(
            predicted_classes=[2, 2],
            receiver=receiver,
            tokenizer=TinyTokenizer(),
            prompt_mode="plain",
            device="cpu",
            max_new_tokens=2,
            max_prompt_tokens=64,
            batch_size=2,
        )
        row_a = lb.prediction_row(gold_one, texts[0])
        row_b = lb.prediction_row(gold_seven, texts[0])
        self.assertEqual(row_a["text"], row_b["text"])
        self.assertNotEqual(gold_one.sequence, gold_seven.sequence)
        self.assertNotEqual(row_a["true_class"], row_b["true_class"])

    def test_illegal_predictions_and_length_errors(self) -> None:
        record = lb.filter_records([_record()])[0][0]
        for bad in (True, False, 0, 8, "1", 1.0, None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ch.require_class_id(bad)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ch.require_class_id_sequence([])
        with self.assertRaises(ValueError):
            ch.require_class_id_sequence("123")
        with self.assertRaises(ValueError):
            ch.classifier_direct_rows([record], [1, 2])
        with self.assertRaises(ValueError):
            ch.receiver_handoff_rows([record, record], ["a"], [1, 2])

    def test_padding_uses_class_dependent_prompts(self) -> None:
        receiver = TinyReceiver()
        lb.freeze_module(receiver)
        with patch.object(ch, "pad_rows", wraps=ch.pad_rows) as padded:
            ch.generate_handoff(
                predicted_classes=[1, 7],
                receiver=receiver,
                tokenizer=ClassAwareTokenizer(),
                prompt_mode="plain",
                device="cpu",
                max_new_tokens=2,
                max_prompt_tokens=64,
                batch_size=2,
            )
        rows = padded.call_args.args[0]
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(len(rows[0]), len(rows[1]))

    def test_error_decomposition_covers_mismatch_when_classifier_was_wrong(self) -> None:
        records = [
            lb.filter_records([_record(accession=f"A{i}", dup_group=i, ec=("1.1.1.1",))])[0][0]
            for i in range(1, 5)
        ]
        classifier_rows = ch.classifier_direct_rows(records, [1, 1, 2, 2])
        texts = [lb.target_json(1), "not json", lb.target_json(2), lb.target_json(1)]
        receiver_rows = ch.receiver_handoff_rows(records, texts, [1, 1, 2, 2])
        decomposed = ch.handoff_error_decomposition(classifier_rows, receiver_rows)
        self.assertEqual(decomposed["n_attempts"], 4)
        self.assertEqual(decomposed["n_receiver_matches_sent_class"], 2)
        self.assertEqual(decomposed["introduced_errors"], 1)
        self.assertEqual(decomposed["corrected_errors"], 1)
        self.assertEqual(
            decomposed["counts_2x2"],
            {
                "classifier_correct_handoff_preserved": 1,
                "classifier_correct_handoff_broken": 1,
                "classifier_incorrect_handoff_preserved": 1,
                "classifier_incorrect_handoff_broken": 1,
            },
        )
        self.assertFalse(receiver_rows[1]["valid_json"])
        self.assertEqual(receiver_rows[1]["sent_class"], 1)
        self.assertNotEqual(receiver_rows[3]["predicted_class"], receiver_rows[3]["sent_class"])


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cli = SimpleNamespace(
            **runpy.run_path(str(REPO / "scripts/transfer/49_classifier_handoff.py"))
        )
        self.prepared = _prepared(self.root)
        self.cache = _cache(self.root, self.prepared, d=8)
        for name, text in (
            ("config.json", "{}"),
            ("tokenizer.json", "{}"),
            ("model.safetensors", "weights"),
        ):
            (self.root / name).write_text(text)

    def test_cli_defaults_have_no_oracle_flag(self) -> None:
        args = self.cli.build_parser().parse_args(["--out", "x", "--substage", "train"])
        self.assertEqual(args.prompt_mode, "chat")
        self.assertEqual(args.dtype, "bfloat16")
        self.assertEqual(args.max_new_tokens, 64)
        self.assertEqual(args.max_prompt_tokens, 1024)
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.heads, ["linear", "mlp"])
        self.assertEqual(args.seeds, [20260908, 20260909, 20260910])
        self.assertEqual(args.epochs, 20)
        self.assertFalse(hasattr(args, "oracle"))
        with self.assertRaises(SystemExit):
            self.cli.build_parser().parse_args(["--out", "x", "--substage", "eval", "--oracle"])

    def test_train_cli_writes_cells_without_receiver_or_test_metrics(self) -> None:
        out = self.root / "train"
        with patch.object(ch, "train_classifier_head", wraps=ch.train_classifier_head) as wrapped:
            code = self.cli.main(
                [
                    "--device",
                    "cpu",
                    "--out",
                    str(out),
                    "--substage",
                    "train",
                    "--prepared",
                    str(self.root / "prepared.json"),
                    "--cache",
                    str(self.cache),
                    "--heads",
                    "linear",
                    "--seeds",
                    "20260908",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "2",
                    "--trainable-budget",
                    "20000",
                ]
            )
        self.assertEqual(code, 0)
        self.assertGreater(wrapped.call_count, 0)
        for call in wrapped.call_args_list:
            self.assertNotIn("test_features", call.kwargs)
            self.assertNotIn("test_records", call.kwargs)
            self.assertTrue(all(row.role == "train" for row in call.kwargs["train_records"]))
            self.assertTrue(all(row.role == "val" for row in call.kwargs["val_records"]))
        report = json.loads((out / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["substage"], "train")
        self.assertFalse(report["test_metrics_computed"])
        self.assertFalse(report["test_touched"])
        self.assertEqual(len(report["cells"]), 1)
        cell_dir = out / report["cells"][0]["dir"]
        self.assertTrue((cell_dir / "classifier.pt").is_file())
        cell_report = json.loads((cell_dir / "train_report.json").read_text(encoding="utf-8"))
        self.assertFalse(cell_report["test_metrics_computed"])

    def test_eval_is_exploratory_and_interface_failure_is_nonzero(self) -> None:
        train_out = self.root / "train"
        self.cli.main(
            [
                "--device",
                "cpu",
                "--out",
                str(train_out),
                "--substage",
                "train",
                "--prepared",
                str(self.root / "prepared.json"),
                "--cache",
                str(self.cache),
                "--heads",
                "linear",
                "--seeds",
                "20260908",
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--trainable-budget",
                "20000",
            ]
        )
        receiver = TinyReceiver()
        lb.freeze_module(receiver)
        tokenizer = TinyTokenizer()
        eval_out = self.root / "eval"
        with patch.dict(
            self.cli.run_eval.__globals__,
            {"load_receiver": lambda *a, **k: (receiver, tokenizer)},
        ):
            code = self.cli.main(
                [
                    "--device",
                    "cpu",
                    "--out",
                    str(eval_out),
                    "--substage",
                    "eval",
                    "--prepared",
                    str(self.root / "prepared.json"),
                    "--cache",
                    str(self.cache),
                    "--classifier-dir",
                    str(train_out),
                    "--receiver-path",
                    str(self.root),
                    "--prompt-mode",
                    "plain",
                    "--heads",
                    "linear",
                    "--seeds",
                    "20260908",
                    "--trainable-budget",
                    "20000",
                    "--batch-size",
                    "2",
                    "--max-new-tokens",
                    "2",
                ]
            )
        self.assertEqual(code, 0)
        report = json.loads((eval_out / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["evaluation_status"], "exploratory")
        self.assertEqual(len(report["cells"]), 1)
        cell = report["cells"][0]
        self.assertIn("counts_2x2", cell["error_decomposition"])
        self.assertEqual(cell["error_decomposition"]["n_attempts"], 2)
        direct = json.loads((eval_out / cell["dir"] / "classifier_direct.json").read_text())
        handoff = json.loads((eval_out / cell["dir"] / "receiver_handoff.json").read_text())
        self.assertEqual(direct["source"], "classifier_direct_template")
        self.assertEqual(handoff["source"], "receiver_handoff")
        self.assertEqual({row["accession"] for row in direct["rows"]}, {"h1", "h2"})
        self.assertEqual(
            [row["accession"] for row in direct["rows"]],
            [row["accession"] for row in handoff["rows"]],
        )
        self.assertTrue(all(row["valid_json"] for row in direct["rows"]))
        self.assertTrue(all(not row["valid_json"] for row in handoff["rows"]))
        self.assertTrue(all("sent_class" in row for row in handoff["rows"]))
        interface_out = self.root / "interface"
        with patch.dict(
            self.cli.run_interface.__globals__,
            {"load_receiver": lambda *a, **k: (receiver, tokenizer)},
        ):
            failed = self.cli.main(
                [
                    "--device",
                    "cpu",
                    "--out",
                    str(interface_out),
                    "--substage",
                    "interface",
                    "--prepared",
                    str(self.root / "prepared.json"),
                    "--cache",
                    str(self.cache),
                    "--receiver-path",
                    str(self.root),
                    "--prompt-mode",
                    "plain",
                    "--heads",
                    "linear",
                    "--seeds",
                    "20260908",
                    "--trainable-budget",
                    "20000",
                    "--max-interface-records",
                    "2",
                    "--max-new-tokens",
                    "2",
                ]
            )
        self.assertEqual(failed, 1)
        interface = json.loads((interface_out / "report.json").read_text())
        self.assertEqual(interface["status"], "failed")
        self.assertTrue(interface["gates"]["bounded_train_only"])
        self.assertTrue(interface["gates"]["linear_finite_nonzero_grad_and_ln_frozen"])
        self.assertFalse(interface["gates"]["handoff_preserves_all_seven_sent_classes"])
        self.assertFalse(interface["used_family_holdout"])


if __name__ == "__main__":
    unittest.main()
