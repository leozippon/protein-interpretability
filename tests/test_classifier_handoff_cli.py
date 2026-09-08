"""CLI admission must consume a complete frozen inventory, not self-validate a checkpoint."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch

from src.transfer.classifier_handoff import (
    REPORT_SCHEMA, build_classifier_head, cell_relative_dir, classifier_identity,
    save_classifier, train_config_record,
)
from src.transfer.latent_bridge import CACHE_SCHEMA, DIRECTION_MARKER, DONOR_RENDERING, POOLING

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/transfer/49_classifier_handoff.py"
spec = importlib.util.spec_from_file_location("classifier_handoff_cli_admission", SCRIPT)
assert spec is not None and spec.loader is not None
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class ClassifierHandoffCLIAdmissionTests(unittest.TestCase):
    def test_inventory_requires_exact_matrix_and_canonical_paths(self):
        cells = [
            {"kind": "linear", "seed": seed, "dir": cell_relative_dir("linear", seed),
             "identity": {"train_config": {}}}
            for seed in (8, 9)
        ]
        report = {"schema_version": REPORT_SCHEMA, "substage": "train", "n_cells": 2, "cells": cells}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "report.json"
            path.write_text(json.dumps(report))
            self.assertEqual(len(cli._load_train_cells(root, heads=["linear"], seeds=[8, 9])), 2)
            missing = copy.deepcopy(report)
            missing["cells"] = cells[:1]
            missing["n_cells"] = 1
            duplicate = copy.deepcopy(report)
            duplicate["cells"] = [cells[0], cells[0]]
            escaped = copy.deepcopy(report)
            escaped["cells"][0]["dir"] = "../stale-checkpoint"
            wrong_count = copy.deepcopy(report)
            wrong_count["n_cells"] = 1
            for changed in (missing, duplicate, escaped, wrong_count):
                with self.subTest(changed=changed):
                    path.write_text(json.dumps(changed))
                    with self.assertRaises(ValueError):
                        cli._load_train_cells(root, heads=["linear"], seeds=[8, 9])

    def test_checkpoint_training_config_is_bound_to_inventory(self):
        prepared_hash = "a" * 64
        manifest = {
            "schema_version": CACHE_SCHEMA, "data_hash": prepared_hash,
            "donor_kind": "pretrained", "checkpoint_digest": {"content_digest": "b" * 64},
            "layer": 0, "pooling": POOLING, "rendering": DONOR_RENDERING,
            "n": 1, "d": 4, "accessions": ["synthetic"],
            "accession_order_hash": "c" * 64, "feature_sha256": "d" * 64,
            "donor_name": "synthetic-donor", "n_layer": 1, "max_tokens": 64,
            "max_residues": 32, "dtype": "float32", "direction_marker": DIRECTION_MARKER,
        }
        head = build_classifier_head(kind="linear", d_in=4, budget=200, seed=8)
        config = train_config_record(epochs=20, batch_size=32, lr=0.001, weight_decay=0.01,
                                     betas=(0.9, 0.999), eps=1e-8, grad_clip=1.0, seed=8)
        identity = classifier_identity(head=head, budget=200, seed=8, prepared_hash=prepared_hash,
                                       cache_manifest=manifest, train_config=config)
        relative = cell_relative_dir("linear", 8)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / relative / "classifier.pt"
            save_classifier(checkpoint, head, extra={"identity": identity})
            report = {"schema_version": REPORT_SCHEMA, "substage": "train", "n_cells": 1,
                      "cells": [{"kind": "linear", "seed": 8, "dir": relative, "identity": identity}]}
            (root / "report.json").write_text(json.dumps(report))
            args = cli.build_parser().parse_args([
                "--substage", "eval", "--out", str(root / "eval"),
                "--classifier-dir", str(root), "--heads", "linear", "--seeds", "8",
                "--trainable-budget", "200",
            ])
            qualified = cli._qualified_eval_heads(args, prepared_hash=prepared_hash,
                                                  cache_manifest=manifest, d_in=4)
            self.assertEqual(qualified[0][2], hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            for name, value in head.state_dict().items():
                self.assertTrue(torch.equal(value, qualified[0][1].state_dict()[name]))
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            payload["extra"]["identity"]["train_config"]["epochs"] = 1
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, "identity"):
                cli._qualified_eval_heads(args, prepared_hash=prepared_hash,
                                          cache_manifest=manifest, d_in=4)


if __name__ == "__main__":
    unittest.main()
