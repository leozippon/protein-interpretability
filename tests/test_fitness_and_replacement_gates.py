"""Gates and draws of the external-baseline track, tested without a GPU.

Nothing here needs a checkpoint. The functions covered are the ones that decide
a verdict or select the units a verdict is computed over, and every one of them
was previously exercised only by running a campaign -- which is how a discrete
threshold ran at two different significance levels across two families of one
artefact without anything failing.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.transfer.fitness import BLOSUM62, PROGENMECH_ASSAYS, load_assay  # noqa: E402


def _load_stage(filename: str):
    """Import a stage whose module name starts with a digit."""

    path = REPO / "scripts/transfer" / filename
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


STAGE15 = _load_stage("15_replacement_faithfulness.py")
STAGE16 = _load_stage("16_fitness_recovery.py")


class Blosum62(unittest.TestCase):
    def test_it_is_the_full_symmetric_twenty_by_twenty_matrix(self):
        self.assertEqual(len(BLOSUM62), 400)
        for (left, right), value in BLOSUM62.items():
            self.assertEqual(value, BLOSUM62[(right, left)], f"{left}{right}")

    def test_identity_scores_beat_the_substitutions_of_the_same_residue(self):
        # A transcription error in the matrix would most likely break this,
        # because the diagonal is the largest entry of almost every row.
        for residue in "ARNDCQEGHILKMFPSTWYV":
            others = [BLOSUM62[(residue, other)] for other in "ARNDCQEGHILKMFPSTWYV"
                      if other != residue]
            self.assertGreater(BLOSUM62[(residue, residue)], max(others), residue)


class AssayDraw(unittest.TestCase):
    """The draw is where a fitness number is most easily made to say anything."""

    ASSAY = "GRB2_HUMAN_Faure_2021"

    @classmethod
    def setUpClass(cls):
        cls.path = (
            REPO / "data/proteingym/DMS_ProteinGym_substitutions" / f"{cls.ASSAY}.csv"
        )
        if not cls.path.is_file():
            raise unittest.SkipTest(f"{cls.path} absent on this host")

    def test_the_draw_is_seeded_and_reproducible(self):
        first = load_assay(self.ASSAY, n=64, seed=11)
        again = load_assay(self.ASSAY, n=64, seed=11)
        self.assertEqual(first.mutants, again.mutants)
        self.assertEqual(first.wildtype, again.wildtype)

    def test_a_different_seed_gives_a_different_draw(self):
        self.assertNotEqual(
            load_assay(self.ASSAY, n=64, seed=11).mutants,
            load_assay(self.ASSAY, n=64, seed=12).mutants,
        )

    def test_the_draw_is_not_a_prefix_of_the_file(self):
        # Appendix B rule 1. A ProteinGym CSV is ordered by residue position, so
        # a prefix draw is the protein's N-terminus rather than a sample of it.
        with self.path.open() as handle:
            head = [row["mutant"] for _, row in zip(range(64), csv.DictReader(handle))]
        self.assertNotEqual(load_assay(self.ASSAY, n=64, seed=11).mutants, head)

    def test_every_drawn_variant_reverts_to_one_wild_type(self):
        assay = load_assay(self.ASSAY, n=128, seed=7)
        for mutant, sequence in zip(assay.mutants, assay.sequences):
            for token in mutant.split(":"):
                position = int(token[1:-1])
                self.assertEqual(sequence[position - 1], token[-1])
                self.assertEqual(assay.wildtype[position - 1], token[0])

    def test_a_variant_disagreeing_with_its_own_mutation_string_is_refused(self):
        assay = load_assay(self.ASSAY, n=8, seed=3, include_multi=False)
        position = int(assay.mutants[0][1:-1])
        corrupted = list(assay.sequences[0])
        corrupted[position - 1] = "W" if corrupted[position - 1] != "W" else "A"
        from src.transfer.fitness import _revert

        with self.assertRaises(ValueError):
            _revert("".join(corrupted), assay.mutants[0])

    def test_the_stratified_draw_balances_the_score_bins(self):
        assay = load_assay(
            self.ASSAY, n=200, seed=5, stratify_by_score_bin=True, train_holdout=256
        )
        median = float(np.median(assay.scores))
        above = int((assay.scores > median).sum())
        # Balanced on DMS_score_bin, so the fit and unfit halves are equal by
        # construction rather than by the assay's natural class ratio.
        self.assertAlmostEqual(above / len(assay.scores), 0.5, delta=0.12)
        self.assertIn("score_bin_stratified", assay.sampling)

    def test_the_uniform_and_stratified_draws_are_different_populations(self):
        uniform = load_assay(self.ASSAY, n=200, seed=5)
        stratified = load_assay(
            self.ASSAY, n=200, seed=5, stratify_by_score_bin=True, train_holdout=256
        )
        self.assertNotEqual(uniform.mutants, stratified.mutants)

    def test_every_progenmech_assay_is_present_and_loadable(self):
        root = REPO / "data/proteingym/DMS_ProteinGym_substitutions"
        for name in PROGENMECH_ASSAYS:
            if not (root / f"{name}.csv").is_file():
                self.skipTest(f"{name} absent on this host")
            assay = load_assay(name, n=16, seed=1)
            self.assertEqual(len(assay.sequences), 16)
            self.assertGreater(len(assay.wildtype), 0)


class ExactTopKNull(unittest.TestCase):
    """The gate that moved between draws for an arithmetic reason."""

    @staticmethod
    def _paired(n_components: int, overlap: int, k: int, n_sequences: int = 24):
        """Two effect matrices whose top-k sets overlap in exactly ``overlap``."""

        original = np.zeros((n_components, n_sequences))
        replacement = np.zeros((n_components, n_sequences))
        original[np.arange(k)] = 10.0 - np.arange(k)[:, None]
        shared = list(range(overlap))
        fresh = list(range(k, k + (k - overlap)))
        for rank, index in enumerate(shared + fresh):
            replacement[index] = 10.0 - rank
        return original, replacement

    def test_the_p_value_is_the_hypergeometric_survival(self):
        from scipy import stats

        for n_components, k, overlap in ((60, 10, 4), (60, 10, 5), (30, 6, 2)):
            original, replacement = self._paired(n_components, overlap, k)
            record = STAGE15.causal_agreement(
                original, replacement, seed=1, replicates=200, top_k=k, alpha=0.05
            )
            self.assertEqual(record["top_k_overlap"], overlap)
            self.assertAlmostEqual(
                record["exact_null"]["p_value_one_sided"],
                float(stats.hypergeom(n_components, k, k).sf(overlap - 1)),
                places=12,
            )

    def test_the_recorded_cliff_no_longer_decides_the_verdict(self):
        # 4 of 10 in 60 is p = 0.052 and 5 is p = 0.0078. Against an empirical
        # q95 of exactly 4.0 these read False and True; against a declared alpha
        # they read as what they are, and the artefact carries the p-value.
        four = STAGE15.causal_agreement(
            *self._paired(60, 4, 10), seed=1, replicates=200, top_k=10, alpha=0.05
        )
        five = STAGE15.causal_agreement(
            *self._paired(60, 5, 10), seed=1, replicates=200, top_k=10, alpha=0.05
        )
        self.assertGreater(four["exact_null"]["p_value_one_sided"], 0.05)
        self.assertLess(five["exact_null"]["p_value_one_sided"], 0.05)
        self.assertFalse(four["exceeds_random_control"])
        self.assertTrue(five["exceeds_random_control"])

    def test_a_clamped_top_k_is_recorded_rather_than_silent(self):
        # The 10-component MoE family requested top-10 and tested top-3, so the
        # same named gate ran at a different attainable level than the 60-head
        # attention family. Both facts are now in the artefact.
        record = STAGE15.causal_agreement(
            *self._paired(10, 3, 3), seed=1, replicates=200, top_k=10, alpha=0.05
        )
        self.assertEqual(record["top_k"], 3)
        self.assertEqual(record["top_k_requested"], 10)
        self.assertTrue(record["top_k_clamped"])
        self.assertEqual(record["exact_null"]["smallest_significant_overlap"], 3)


class BackboneCoverage(unittest.TestCase):
    def test_agreement_alone_does_not_pass_the_gate(self):
        # A replacement embedding a strict subset of the backbone agrees
        # perfectly on what it carries while the weights it was fitted to go
        # uncompared. That is the failure the gate exists to catch.
        shared = torch.arange(6.0)
        released = {"a": shared.clone(), "b": shared.clone(), "c": shared.clone()}
        subset = {"a": shared.clone()}
        self.assertEqual(STAGE15.backbone_identity(subset, released)["verdict"], "FAIL")

    def test_the_keys_progen3_drops_by_design_do_not_fail_it(self):
        from src.transfer.progen3 import DROPPED_KEYS

        shared = torch.arange(6.0)
        released = {"a": shared.clone(), **{key: shared.clone() for key in DROPPED_KEYS}}
        embedded = {"a": shared.clone()}
        record = STAGE15.backbone_identity(embedded, released)
        self.assertEqual(record["verdict"], "PASS")
        self.assertEqual(record["released_keys_dropped_by_design"], sorted(DROPPED_KEYS))

    def test_a_disagreeing_tensor_fails(self):
        released = {"a": torch.arange(6.0)}
        embedded = {"a": torch.arange(6.0) + 1.0}
        self.assertEqual(STAGE15.backbone_identity(embedded, released)["verdict"], "FAIL")


class FreeBaselineGate(unittest.TestCase):
    @staticmethod
    def _assays(differences):
        return [
            {
                "model": {"spearman": 0.30 + d},
                "blosum62": {"spearman": 0.30},
                "model_minus_blosum62": d,
            }
            for d in differences
        ]

    def test_a_model_reliably_above_the_free_baseline_passes(self):
        record = STAGE16.free_baseline_gate(self._assays([0.2] * 8), gate=0.0)
        self.assertEqual(record["verdict"], "PASS")
        self.assertEqual(record["assays_where_model_wins"], 8)

    def test_a_model_indistinguishable_from_the_free_baseline_fails(self):
        record = STAGE16.free_baseline_gate(
            self._assays([0.05, -0.04, 0.03, -0.06, 0.02, -0.01, 0.04, -0.03]), gate=0.0
        )
        self.assertEqual(record["verdict"], "FAIL")

    def test_the_interval_is_paired_not_marginal(self):
        # Assays differ enormously in difficulty, so an unpaired comparison of
        # two means over eight of them reads assay selection. A constant
        # advantage on wildly different assays must still pass.
        record = STAGE16.free_baseline_gate(
            [
                {
                    "model": {"spearman": base + 0.10},
                    "blosum62": {"spearman": base},
                    "model_minus_blosum62": 0.10,
                }
                for base in (0.02, 0.10, 0.20, 0.31, 0.40, 0.48, 0.55, 0.62)
            ],
            gate=0.0,
        )
        self.assertEqual(record["verdict"], "PASS")

    def test_the_unit_floor_travels_with_the_interval(self):
        record = STAGE16.free_baseline_gate(self._assays([0.2, 0.2]), gate=0.0)
        self.assertTrue(record["unit_floor"]["degenerate"])
        self.assertIsNotNone(record["unit_floor"]["degenerate_reason"])


if __name__ == "__main__":
    unittest.main()
