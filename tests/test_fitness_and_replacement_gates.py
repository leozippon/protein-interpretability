"""Gates and draws of the external-baseline track, tested without a GPU.

Nothing here needs a checkpoint. The functions covered are the ones that decide
a verdict or select the units a verdict is computed over, and every one of them
was previously exercised only by running a campaign -- which is how a discrete
threshold ran at two different significance levels across two families of one
artefact without anything failing.
"""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import json
import sys
import tempfile
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

    def test_the_multi_mutant_count_describes_the_drawn_cohort(self):
        """The composition reported beside `n_variants` is the draw's own.

        It used to be accumulated while the CSV was read, so a shipped record
        carried `n_variants: 1000` beside a multi-mutant count of 14,015 -- a
        property of the eligible pool wearing the name of a property of the
        cohort. The count is bounded by the draw it describes, and the draw is
        what the ProGenMech design comparison is about.
        """

        assay = load_assay(self.ASSAY, n=200, seed=9)
        record = assay.record()
        self.assertEqual(record["n_variants"], len(assay.mutants))
        self.assertLessEqual(record["n_multi_mutant_drawn"], record["n_variants"])
        self.assertLess(record["n_variants"], record["n_eligible"])
        self.assertEqual(
            record["n_multi_mutant_drawn"],
            sum(1 for mutant in assay.mutants if ":" in mutant),
        )

    def test_a_single_substitution_draw_reports_no_multi_mutants(self):
        assay = load_assay(self.ASSAY, n=32, seed=9, include_multi=False)
        self.assertEqual(assay.record()["n_multi_mutant_drawn"], 0)

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


class _RecordingModel:
    """Just enough :class:`ReplaceableModel` to see what the control splices.

    The control must go through ``block_intercept`` -- the one primitive the
    replacement and the mean ablation already use -- rather than open a second
    forward path of its own, so this records the interceptor it is handed and
    nothing else.
    """

    device = torch.device("cpu")

    def __init__(self):
        self.interceptors = []

    @contextlib.contextmanager
    def block_intercept(self, fn):
        self.interceptors.append(fn)
        yield


class MatchedRandomPerturbation(unittest.TestCase):
    """The control that separates a LARGE replacement error from a DIRECTED one.

    Under sequential replacement a protein decoder recovers a fraction of the
    clean-to-ablated gap that a text decoder recovers almost all of, and the
    damage is 8-17x larger in absolute nats. That is compatible with two
    different stories -- an error that is merely bigger, and an error aimed at the
    subspace the downstream computation reads -- and only a perturbation matched
    to the error in both norm and angle tells them apart. If the matching is
    wrong the control silently answers a different question, so it is pinned here
    rather than trusted.
    """

    SHAPE = (2, 5, 16)

    @classmethod
    def _pair(cls, seed: int = 0):
        """A block output and a replacement whose error leans on it.

        The parallel component is deliberately large: an implementation that
        discarded it would still be norm-matched, and only the angle test would
        see the difference.
        """

        generator = torch.Generator().manual_seed(seed)
        clean = torch.randn(cls.SHAPE, generator=generator)
        error = 0.5 * clean + 0.3 * torch.randn(cls.SHAPE, generator=generator)
        return clean, clean + error

    @staticmethod
    def _cosine(left, right):
        return (left * right).sum(-1) / (left.norm(dim=-1) * right.norm(dim=-1))

    def test_the_perturbation_carries_the_errors_own_norm_at_every_position(self):
        clean, replacement = self._pair()
        perturbation = STAGE15.matched_perturbation(
            clean, replacement, torch.Generator().manual_seed(11)
        )
        self.assertEqual(perturbation.shape, clean.shape)
        torch.testing.assert_close(
            perturbation.norm(dim=-1),
            (replacement - clean).norm(dim=-1),
            rtol=1e-4,
            atol=1e-5,
        )

    def test_the_perturbation_makes_the_errors_own_angle_with_the_clean_output(self):
        clean, replacement = self._pair()
        error = replacement - clean
        perturbation = STAGE15.matched_perturbation(
            clean, replacement, torch.Generator().manual_seed(11)
        )
        reference = self._cosine(error, clean)
        # The angle has to be worth matching, or the test would pass on an
        # implementation that made r orthogonal to the block output.
        self.assertGreater(float(reference.min()), 0.2)
        torch.testing.assert_close(
            self._cosine(perturbation, clean), reference, rtol=1e-4, atol=1e-5
        )

    def test_the_orthogonal_direction_really_moves_between_seeds(self):
        clean, replacement = self._pair()
        first, second = (
            STAGE15.matched_perturbation(
                clean, replacement, torch.Generator().manual_seed(seed)
            )
            for seed in (1, 2)
        )
        again = STAGE15.matched_perturbation(
            clean, replacement, torch.Generator().manual_seed(1)
        )
        torch.testing.assert_close(first, again)
        # Matched in both quantities and different in the one thing left free,
        # which is what makes several draws a distribution rather than a repeat.
        torch.testing.assert_close(
            first.norm(dim=-1), second.norm(dim=-1), rtol=1e-4, atol=1e-5
        )
        unit = clean / clean.norm(dim=-1, keepdim=True)
        residuals = [
            value - (value * unit).sum(-1, keepdim=True) * unit
            for value in (first, second)
        ]
        self.assertLess(float(self._cosine(*residuals).max()), 0.99)

    def test_a_single_degenerate_position_refuses_the_whole_tensor(self):
        # Both degeneracies leave the angle undefined, and both would otherwise
        # return a zero perturbation at that position -- a control reported as
        # matched that was never matched to anything.
        clean, replacement = self._pair()
        zero_output = clean.clone()
        zero_output[0, 2] = 0.0
        with self.assertRaises(ValueError):
            STAGE15.matched_perturbation(
                zero_output, replacement, torch.Generator().manual_seed(3)
            )
        exact = replacement.clone()
        exact[1, 4] = clean[1, 4]
        with self.assertRaises(ValueError):
            STAGE15.matched_perturbation(
                clean, exact, torch.Generator().manual_seed(3)
            )

    def test_the_control_substitutes_the_clean_output_plus_the_matched_error(self):
        model = _RecordingModel()

        def transcoder(layer, x):
            return 1.5 * x + float(layer)

        factory = STAGE15.matched_perturbation_context(model, transcoder, seed=5)
        with factory():
            pass
        self.assertEqual(len(model.interceptors), 1, "one splice, not a second path")
        intercept = model.interceptors[0]
        block_input = torch.randn(1, 4, 8, generator=torch.Generator().manual_seed(2))
        block_output = torch.randn(1, 4, 8, generator=torch.Generator().manual_seed(4))
        substituted = intercept(3, block_input, block_output)
        # y_clean + r, not the replacement output, and r matched to the error the
        # replacement would have made at this layer.
        error = transcoder(3, block_input) - block_output
        torch.testing.assert_close(
            (substituted - block_output).norm(dim=-1),
            error.norm(dim=-1),
            rtol=1e-4,
            atol=1e-5,
        )

    def test_with_no_draws_the_control_is_withheld_with_a_reason_and_no_number(self):
        record = STAGE15.matched_perturbation_control({}, {}, replicates=16, seed=1)
        self.assertEqual(record["verdict"], "WITHHELD")
        self.assertIn("matched-perturbation-draws", record["reason"])
        self.assertEqual(sorted(record), ["reason", "verdict"])
        self.assertFalse(
            any(isinstance(value, (int, float)) for value in record.values()),
            "a withheld control must carry no number that could be read as a result",
        )

    def test_every_draw_is_reported_individually_beside_its_seed(self):
        # Three draws whose damage differs by a factor of two: a record that kept
        # only their mean would hide exactly that.
        n = 48
        clean = np.linspace(2.0, 2.4, n)
        scores = {
            "original": {"nll": clean, "kl": np.zeros(n)},
            "replacement": {"nll": clean + 1.0, "kl": np.full(n, 0.9)},
            "mean_ablated": {"nll": clean + 2.0, "kl": np.full(n, 2.0)},
        }
        seeds = {}
        for index, damage in enumerate((0.2, 0.3, 0.4)):
            name = f"matched_perturbation_draw{index}"
            scores[name] = {"nll": clean + damage, "kl": np.full(n, damage / 2.0)}
            seeds[name] = 100 + index
        record = STAGE15.matched_perturbation_control(
            scores, seeds, replicates=64, seed=3
        )
        self.assertEqual(record["verdict"], "REPORTED")
        self.assertEqual(record["matched_at"], "position")
        self.assertEqual([draw["seed"] for draw in record["draws"]], [100, 101, 102])
        for draw, damage in zip(record["draws"], (0.2, 0.3, 0.4)):
            self.assertAlmostEqual(draw["nll_minus_clean"], damage, places=6)
            # The same denominator the replacement's own recovery is read against.
            self.assertAlmostEqual(draw["recovery"], (2.0 - damage) / 2.0, places=6)
            self.assertAlmostEqual(draw["kl_nats_per_token"], damage / 2.0, places=6)
            self.assertAlmostEqual(draw["kl_recovery"], 1.0 - damage / 4.0, places=6)
            self.assertAlmostEqual(
                draw["replacement_nll_damage_over_this"], 1.0 / damage, places=6
            )
            self.assertIsNotNone(draw["paired_per_sequence"]["recovery_interval"])
        spread = record["across_draws"]["nll_minus_clean"]
        self.assertEqual(len(spread["values"]), 3)
        self.assertAlmostEqual(spread["min"], 0.2, places=6)
        self.assertAlmostEqual(spread["max"], 0.4, places=6)

    def test_the_flag_defaults_to_not_running_the_control(self):
        # Every invocation that predates the flag must compute what it computed
        # before, so the default cannot be anything but zero. Read from the
        # declaration rather than trusted, as the eligible-arm set is.
        source = Path(STAGE15.__file__).read_text(encoding="utf-8")
        declaration = source.split('"--matched-perturbation-draws",', 1)
        self.assertEqual(
            len(declaration), 2, "the stage declares no --matched-perturbation-draws"
        )
        self.assertIn("default=0", declaration[1][:200])


class _ToyDecoder(torch.nn.Module):
    """A residual stack whose sublayer output IS its contribution, as the arms' are."""

    def __init__(self, vocab: int = 13, width: int = 8, layers: int = 3):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, width)
        self.blocks = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.Linear(width, width),
                torch.nn.Tanh(),
                torch.nn.Linear(width, width),
            )
            for _ in range(layers)
        )
        self.head = torch.nn.Linear(width, vocab)

    def forward(self, ids):
        hidden = self.embed(ids)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.head(hidden)


class _ToyReplaceable:
    """The slice of :class:`ReplaceableModel` the behavioural sweep consumes."""

    def __init__(self, model: _ToyDecoder, vocab: int = 13):
        self.model = model.eval()
        self.vocab = vocab
        self.device = torch.device("cpu")
        self.n_layers = len(model.blocks)
        self.width = model.embed.embedding_dim

    def batch(self, inputs):
        ids = torch.tensor([[int(symbol) % self.vocab for symbol in row] for row in inputs])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def content_mask(self, batch):
        return batch["attention_mask"].bool()

    def run(self, batch):
        return self.model(batch["input_ids"])

    def scored_logits(self, batch):
        logits = self.model(batch["input_ids"])
        return (
            logits[..., :-1, :].float(),
            batch["input_ids"][..., 1:],
            batch["attention_mask"][..., 1:].bool(),
        )

    @contextlib.contextmanager
    def block_intercept(self, fn):
        handles = []
        for layer, block in enumerate(self.model.blocks):

            def hook(module, inputs, output, layer=layer):
                return fn(layer, inputs[0], output)

            handles.append(block.register_forward_hook(hook))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()


class MatchedPerturbationEndToEnd(unittest.TestCase):
    """The control as the stage composes it, on a CPU stub.

    No checkpoint exists on a host that only runs tests, so the sequential splice,
    the shared behavioural sweep and the artefact record are exercised here
    against a toy residual stack rather than left to a campaign to discover. What
    this establishes is composition, not a magnitude: the control is a condition
    of the *same* sweep, every draw moves, and what it produces serialises.
    """

    SEQUENCES = ["1234567890123", "9876543210987", "1122334455667", "5566778899001"]

    def setUp(self):
        torch.manual_seed(0)
        self.model = _ToyReplaceable(_ToyDecoder())
        # Briefly fitted to its own sequences, because on an untrained stack the
        # blocks are not load-bearing and the clean-to-ablated denominator every
        # recovery fraction divides by can come out negative -- which is a
        # property of the stub, not of the control.
        self._fit()
        width = self.model.width
        maps = [
            torch.eye(width) + 0.35 * torch.randn(width, width)
            for _ in range(self.model.n_layers)
        ]
        self.transcoder = lambda layer, x: x @ maps[layer]

    def _fit(self, steps: int = 400):
        batch = self.model.batch(self.SEQUENCES)
        optimiser = torch.optim.Adam(self.model.model.parameters(), lr=0.02)
        self.model.model.train()
        for _ in range(steps):
            logits, targets, _ = self.model.scored_logits(batch)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
            )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        self.model.model.eval()

    def _sweep(self, draws: int):
        conditions = {
            "original": None,
            "replacement": STAGE15.replacement_context(self.model, self.transcoder),
            "mean_ablated": STAGE15.mean_ablation_context(
                self.model, torch.zeros(self.model.n_layers, self.model.width)
            ),
        }
        seeds = {
            f"matched_perturbation_draw{index}": 40 + index for index in range(draws)
        }
        for name, seed in seeds.items():
            conditions[name] = STAGE15.matched_perturbation_context(
                self.model, self.transcoder, seed=seed
            )
        scores = STAGE15.behavioural_scores(
            self.model, self.SEQUENCES, conditions, batch_size=2
        )
        return scores, seeds

    def test_the_control_rides_the_same_sweep_and_every_draw_moves(self):
        scores, seeds = self._sweep(3)
        self.assertEqual(len(seeds), 3)
        for name in seeds:
            self.assertEqual(scores[name]["nll"].shape, scores["original"]["nll"].shape)
            self.assertTrue(np.isfinite(scores[name]["nll"]).all())
            # KL is taken against the same 'original' the replacement's is.
            self.assertTrue((scores[name]["kl"] >= 0).all())
        self.assertTrue((scores["original"]["kl"] == 0).all())
        means = [float(scores[name]["nll"].mean()) for name in seeds]
        self.assertEqual(len(set(means)), 3, "the draws are not independent")

    def test_the_record_is_complete_and_serialises_through_write_json(self):
        scores, seeds = self._sweep(2)
        record = STAGE15.matched_perturbation_control(
            scores, seeds, replicates=128, seed=5
        )
        self.assertEqual(record["verdict"], "REPORTED")
        self.assertEqual(record["n_draws"], 2)
        for draw in record["draws"]:
            for key in ("nll_minus_clean", "kl_nats_per_token", "recovery"):
                self.assertIsNotNone(draw[key], key)
            self.assertIn("resampling_unit", draw["paired_per_sequence"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "control.json"
            # write_json refuses NaN and infinity, which is the failure a ratio
            # against a zero-damage draw would produce.
            from src.transfer.io import write_json

            write_json(path, record)
            self.assertEqual(json.loads(path.read_text())["n_draws"], 2)


if __name__ == "__main__":
    unittest.main()
