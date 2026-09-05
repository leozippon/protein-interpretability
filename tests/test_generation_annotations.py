"""Reference distance and coverage must come from the same auditable HSP."""

import tempfile
import unittest
from pathlib import Path

from src.transfer import generation_annotations as ga
from src.transfer import homology


class ReferenceCoverage(unittest.TestCase):
    def test_best_identity_hit_retains_its_own_query_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            table = Path(directory) / "hits.tsv"
            table.write_text("q\ts1\t100\t40\t40\t1\t40\t100\t80\t1e-9\t60\n"
                             "q\ts2\t50\t90\t45\t3\t92\t100\t100\t1e-8\t50\n")
            hit = ga.best_hits(table, {"q": "A" * 100})["q"]
            fields = ga.reference_fields(hit, searched=True)
            self.assertEqual(fields["reference_identity"], 45)
            self.assertEqual(fields["reference_coverage"], 0.9)
            self.assertEqual(fields["reference_subject"], "s2")
            self.assertIsNone(fields["reference_target_coverage"])
            with self.assertRaisesRegex(ValueError, "query/length mismatch"):
                ga.best_hits(table, {"q": "A" * 99})

    def test_no_hit_is_not_zero_identity_or_unsearched(self):
        no_hit = ga.reference_fields(None, searched=True)
        missing = ga.reference_fields(None, searched=False)
        self.assertIsNone(no_hit["reference_identity"])
        self.assertIsNone(no_hit["reference_coverage"])
        self.assertNotEqual(no_hit["reference_search_status"], missing["reference_search_status"])

    def test_fresh_alignment_carries_target_coverage_from_the_same_hit(self):
        with tempfile.TemporaryDirectory() as directory:
            table = Path(directory) / "hits.tsv"
            table.write_text("q\ts\t60\t5\t3\t1\t4\t8\t10\t1e-9\t60\tAC-DE\tACGDE\n")
            hit = ga.best_hits(table, {"q": "ACDEACDE"}, fields=homology.ALIGNMENT_FIELDS)["q"]
            fields = ga.reference_fields(hit, searched=True)
            self.assertEqual(fields["reference_identity"], 37.5)
            self.assertEqual(fields["reference_coverage"], 0.5)
            self.assertEqual(fields["reference_target_coverage"], 0.5)
            self.assertEqual(fields["reference_target_coverage_status"], "aligned_subject_residue_fraction")


if __name__ == "__main__":
    unittest.main()
