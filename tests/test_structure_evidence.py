import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.transfer import structure_evidence as se
from src.transfer.io import write_json


class StructureEvidence(unittest.TestCase):
    def test_strict_sequences_never_silently_crop_or_clean(self):
        kwargs = dict(min_length=16, max_length=1024)
        for sequence, reason in [("", "empty_sequence"), ("A" * 15, "below_minimum_length"), ("A" * 1025, "above_maximum_length"), ("ACD X" * 4, "noncanonical_residues"), ("a" * 16, "noncanonical_residues")]:
            self.assertEqual(se.eligibility(sequence, **kwargs), reason)
        self.assertIsNone(se.eligibility("A" * 16, **kwargs))
        self.assertIsNone(se.eligibility("A" * 1024, **kwargs))

    def test_ca_confidence_does_not_count_absent_sidechains_or_average_atom_confidence(self):
        mask = np.zeros((5, 37))
        mask[:, :3] = 1
        scores = np.full((5, 37), 99.0)
        scores[:, 1] = [70, 70, 70, 70, 69]
        arrays = {"atom_plddt_0_100": scores, "atom37_atom_exists": mask, "atom37_positions_angstrom": np.zeros((5, 37, 3)), "predicted_aligned_error_angstrom": np.ones((5, 5)), "ptm": np.asarray(0.7)}
        result = se.summarize_arrays(arrays)
        self.assertAlmostEqual(result["mean_ca_plddt"], 69.8)
        self.assertAlmostEqual(result["fraction_ca_plddt_ge70"], 0.8)
        self.assertFalse(result["predicted_confidence_event"])
        scores[-1, 1] = 70
        self.assertTrue(se.summarize_arrays(arrays)["predicted_confidence_event"])
        scores[-1, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            se.summarize_arrays(arrays)

    def test_cohort_integrity_and_duplicate_denominators(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "cohort.jsonl"
            rows = [{"id": "a", "sequence": "A" * 16}, {"id": "b", "sequence": "A" * 16}, {"id": "invalid", "sequence": ""}]
            path.write_text("\n".join(json.dumps(row) for row in rows))
            self.assertEqual(len(se.load_cohort(path)), 3)
            sha = se.sequence_digest("")
            write_json(root / "objects" / sha / "result.json", {"evaluation_signature": "fixed", "sequence_sha256": sha, "status": "not_evaluable"})
            report = se.write_index(rows, root, "fixed", shard_index=0, num_shards=1)
            self.assertEqual(report["rows"], 3)
            self.assertEqual(report["unique_sequences"], 2)
            self.assertEqual(report["status_counts"], {"pending": 2, "not_evaluable": 1})
            rows[1]["id"] = "a"
            path.write_text("\n".join(json.dumps(row) for row in rows))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                se.load_cohort(path)
            path.write_text(json.dumps({"id": "a", "sequence": "AAA", "sequence_sha256": "wrong"}))
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                se.load_cohort(path)

    def test_exact_duplicates_remain_on_one_shard_and_union_is_complete(self):
        sequences = ["A" * i for i in range(1, 100)]
        for size in (1, 4, 8):
            partitions = [{s for s in sequences if se.shard_for(s, size) == shard} for shard in range(size)]
            self.assertEqual(set.union(*partitions), set(sequences))
            self.assertEqual(sum(map(len, partitions)), len(sequences))

    def test_resume_refuses_changed_configuration_and_missing_prediction_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "result.json", {"evaluation_signature": "a", "sequence_sha256": "b", "status": "ok", "files_sha256": {"prediction.pdb": "c"}})
            with self.assertRaisesRegex(ValueError, "incompatible"):
                se.load_result(root, "changed", "b")
            with self.assertRaisesRegex(ValueError, "incomplete/corrupted"):
                se.load_result(root, "a", "b")

    def test_installed_predictor_confidence_scale_and_pdb_b_factors(self):
        import torch
        from transformers.models.esm.modeling_esmfold import categorical_lddt

        raw = categorical_lddt(torch.zeros(1, 37, 50)).numpy()
        np.testing.assert_allclose(raw, 0.5, atol=1e-6)
        mask = np.zeros((1, 37))
        mask[:, :3] = 1
        arrays = {"aatype": np.zeros(1, dtype=np.int64), "residue_index": np.zeros(1, dtype=np.int64), "atom37_atom_exists": mask, "atom37_positions_angstrom": np.zeros((1, 37, 3)), "atom_plddt_0_100": raw * 100}
        pdb = se.pdb_from_arrays(arrays)
        ca = next(line for line in pdb.splitlines() if line.startswith("ATOM") and line[12:16].strip() == "CA")
        self.assertAlmostEqual(float(ca[60:66]), 50.0)


if __name__ == "__main__":
    unittest.main()
