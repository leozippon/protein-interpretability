"""The declarations that decide which decoder a replacement is measured on.

Nothing here needs a GPU or a checkpoint. What is covered is the part of the
dense-arm extension that can be wrong *silently*: which arms are admitted, what
the loader band accepts, whether the component grid the effect matrices are
indexed by still means the same thing on ProGen3, and whether a transcoder can be
spliced into a model it was not trained on.

The last of those is not hypothetical on this panel. ``gpt2-large`` and
``protgpt2`` are both 36 layers of width 1280, so the depth and width checks
``15_replacement_faithfulness.py`` has always carried cannot separate them.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts/transfer") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts/transfer"))

from panel_contract import CAMPAIGN_PANEL  # noqa: E402
from src.transfer import replaceable as R  # noqa: E402
from src.transfer.arms import PANEL, CORPUS_SOURCES, iter_corpus_records  # noqa: E402
from src.transfer.progen3 import component_grid, components  # noqa: E402
from src.transfer.transcoders import TranscoderConfig  # noqa: E402


def _load_stage(filename: str):
    """Import a stage whose module name starts with a digit."""

    path = REPO / "scripts/transfer" / filename
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


STAGE15 = _load_stage("15_replacement_faithfulness.py")
STAGE17 = _load_stage("17_train_transcoder.py")


class EligibleArms(unittest.TestCase):
    def test_the_progen3_baseline_is_always_reachable(self):
        # It is not a panel member, so no panel declaration can admit it, and the
        # default of both stages is this name.
        self.assertEqual(R.eligible_arms(CAMPAIGN_PANEL)[0], R.PROGEN3_ARM)
        self.assertNotIn(R.PROGEN3_ARM, PANEL)

    def test_the_matched_pair_and_the_smoke_arm_are_admitted(self):
        admitted = R.eligible_arms(CAMPAIGN_PANEL)
        for name in ("gpt2-large", "protgpt2", "gpt2"):
            self.assertIn(name, admitted)

    def test_an_architecture_this_estimand_does_not_cover_is_refused(self):
        # progen2 is a parallel-residual block: its feed-forward reads the same
        # normalisation as its attention, so a transcoder there predicts a
        # different object. llama/qwen2 are serial and simply unverified.
        admitted = R.eligible_arms(CAMPAIGN_PANEL)
        for name in ("progen2-medium", "llama-3.2-3b", "qwen2.5-0.5b", "bygpt5-medium-en"):
            self.assertNotIn(name, admitted)

    def test_a_conditioned_rendering_is_refused_with_its_reason(self):
        # zymctrl passes the architecture gate -- it is GPT-2 -- and is refused on
        # the conditioning leak instead. A refusal nobody can read is
        # indistinguishable from an arm nobody thought about.
        self.assertEqual(PANEL["zymctrl"].architecture, "gpt2")
        self.assertNotIn("zymctrl", R.eligible_arms(CAMPAIGN_PANEL))
        self.assertIn("zymctrl", R.DENSE_ARMS_WITHOUT_A_BAND)
        self.assertIn("1.73", R.DENSE_ARMS_WITHOUT_A_BAND["zymctrl"])

    def test_every_staged_dense_arm_is_either_measured_or_refused(self):
        # The half of the invariant the library cannot check on its own without
        # importing a stage script: an arm that is staged, architecturally
        # admissible and in neither table is one nobody decided about.
        undecided = sorted(
            name
            for name in CAMPAIGN_PANEL
            if PANEL[name].architecture in R.DENSE_ARCHITECTURES
            and name not in R.MEASURED_DENSE_SELF_CHECK_NLL
            and name not in R.DENSE_ARMS_WITHOUT_A_BAND
        )
        self.assertEqual(undecided, [])

    def test_an_arm_outside_the_panel_raises_rather_than_being_dropped(self):
        with self.assertRaises(KeyError):
            R.eligible_arms(["gpt2", "not-a-model"])

    def test_both_stages_offer_exactly_this_set(self):
        # Read from the argparse declaration rather than trusted: a stage that
        # kept its own tuple is the failure panel_contract exists to end.
        for stage in (STAGE15, STAGE17):
            source = Path(stage.__file__).read_text(encoding="utf-8")
            self.assertIn("choices=eligible_arms(CAMPAIGN_PANEL)", source)


class DenseLoaderBand(unittest.TestCase):
    def test_the_reference_value_passes(self):
        for arm, value in R.MEASURED_DENSE_SELF_CHECK_NLL.items():
            self.assertEqual(R.check_dense_nll(arm, value)["verdict"], "PASS")

    def test_every_recorded_corruption_is_refused_by_the_arm_it_was_measured_on(self):
        # The band exists to catch these two, so a band that admitted one would
        # be a band on nothing. protgpt2 rendered raw is L11; a randomly
        # initialised gpt2 is L24's shape on a dense arm.
        for arm, corruption in (
            ("protgpt2", "protgpt2_rendered_raw"),
            ("gpt2", "gpt2_randomly_initialised"),
        ):
            with self.assertRaises(RuntimeError):
                R.check_dense_nll(arm, R.MEASURED_DENSE_SELF_CHECK_CORRUPTIONS[corruption])

    def test_a_value_below_the_band_is_refused_too(self):
        # Below the band means the scored-target convention moved, which looks
        # like an improvement and corrupts everything downstream.
        with self.assertRaises(RuntimeError):
            R.check_dense_nll("gpt2-large", R.MEASURED_DENSE_SELF_CHECK_NLL["gpt2-large"] - 1.0)

    def test_an_unmeasured_arm_cannot_be_gated_at_all(self):
        with self.assertRaises(KeyError):
            R.check_dense_nll("zymctrl", 3.0)

    def test_the_band_is_narrower_than_the_distance_to_the_nearest_corruption(self):
        # The sizing argument, as an invariant rather than as a docstring: a band
        # that reached a corruption could pass on a broken arm.
        nearest = min(
            abs(corruption - value)
            for value in R.MEASURED_DENSE_SELF_CHECK_NLL.values()
            for corruption in R.MEASURED_DENSE_SELF_CHECK_CORRUPTIONS.values()
        )
        self.assertLess(R.DENSE_SELF_CHECK_HALF_WIDTH, nearest)

    def test_the_frozen_inputs_are_literals_and_shared_with_progen3(self):
        from src.transfer.progen3 import SELF_CHECK_SEQUENCES

        self.assertEqual(len(R.SELF_CHECK_DOCUMENTS), 8)
        # A protein dense arm is checked on exactly the records ProGen3 is checked
        # on, so the modality pair is gated on one text set and one protein set.
        self.assertIs(R.SELF_CHECK_SEQUENCES, SELF_CHECK_SEQUENCES)


class ComponentGrid(unittest.TestCase):
    def test_the_progen3_grid_is_unchanged_by_the_generalisation(self):
        # The saved effect matrices are indexed by this order, and every frozen
        # artefact under results/transfer/external_baseline names 'moe_block'.
        grid = component_grid(10, 6, block_kind="moe_block")
        self.assertEqual(len(grid), 70)
        self.assertEqual([c.label for c in grid[:2]], ["attention_head.L0H0", "attention_head.L0H1"])
        self.assertEqual(grid[60].label, "moe_block.L0")
        self.assertEqual(sorted({c.kind for c in grid}), ["attention_head", "moe_block"])

    def test_the_dense_grid_names_the_block_it_actually_replaces(self):
        grid = component_grid(12, 12, block_kind=R.DENSE_BLOCK_KIND)
        self.assertEqual(len(grid), 12 * 12 + 12)
        self.assertEqual(grid[-1].label, "mlp_block.L11")

    def test_the_families_are_derived_from_the_grid_in_its_order(self):
        self.assertEqual(
            STAGE15.families(component_grid(10, 6, block_kind="moe_block")),
            ("attention_head", "moe_block"),
        )
        self.assertEqual(
            STAGE15.families(component_grid(4, 2, block_kind=R.DENSE_BLOCK_KIND)),
            ("attention_head", "mlp_block"),
        )

    def test_the_progen3_accessor_still_delegates_to_the_shared_grid(self):
        class _Stub:
            n_layers = 10
            n_heads = 6

        self.assertEqual(
            [c.label for c in components(_Stub())],
            [c.label for c in component_grid(10, 6, block_kind="moe_block")],
        )


class ReplacementArmIsChecked(unittest.TestCase):
    """Depth and width do not identify an arm; on this panel they cannot."""

    def test_the_matched_pair_is_indistinguishable_by_shape(self):
        left, right = PANEL["gpt2-large"], PANEL["protgpt2"]
        self.assertEqual((left.n_layer, left.d_model), (right.n_layer, right.d_model))

    def test_a_transcoder_trained_on_another_arm_is_refused(self):
        with self.assertRaises(RuntimeError):
            STAGE15.require_matching_arm("gpt2-large", "protgpt2")

    def test_a_matching_arm_is_accepted_and_reported_as_declared(self):
        self.assertTrue(STAGE15.require_matching_arm("protgpt2", "protgpt2"))

    def test_a_checkpoint_predating_the_record_is_accepted_and_reported_as_such(self):
        # The four ProGen3 checkpoints already on disk declare no arm. They must
        # keep loading, and the artefact has to say the check did not run.
        self.assertFalse(STAGE15.require_matching_arm(None, "progen3"))

    def test_the_trainer_records_the_arm_and_it_round_trips(self):
        record = TranscoderConfig(num_layers=2, d_model=4, arm="gpt2-large").record()
        self.assertEqual(record["arm"], "gpt2-large")
        self.assertIsNone(TranscoderConfig(num_layers=2, d_model=4).record()["arm"])


class CorpusStreams(unittest.TestCase):
    def test_every_declared_corpus_has_a_band_in_the_trainer(self):
        # A source with no band would stream unfiltered, which is a different
        # population from the one the faithfulness stage scores.
        self.assertEqual(sorted(STAGE17.CORPUS_BAND), sorted(CORPUS_SOURCES))

    def test_the_text_floor_is_the_cohort_floor(self):
        import inspect

        from src.transfer.arms import text_cohort

        default = inspect.signature(text_cohort).parameters["min_chars"].default
        self.assertEqual(STAGE17.MIN_CHARACTERS, default)

    def test_an_unknown_source_raises(self):
        with self.assertRaises(KeyError):
            iter_corpus_records("uniprot", min_symbols=1, max_symbols=2)

    def test_a_relocation_that_would_be_ignored_is_refused(self):
        # swissprot and openwebtext are read through readers that consult their
        # own declared location, so an accepted path override would stream the
        # declared corpus under another name.
        for source in ("swissprot", "openwebtext"):
            with self.assertRaises(ValueError):
                iter_corpus_records(
                    source, min_symbols=1, max_symbols=None, path=Path("/nowhere")
                )

    def test_a_fasta_stream_needs_its_upper_bound_and_a_text_stream_refuses_one(self):
        with self.assertRaises(ValueError):
            iter_corpus_records("swissprot", min_symbols=32)
        with self.assertRaises(ValueError):
            iter_corpus_records("openwebtext", min_symbols=800, max_symbols=4000)

    def test_the_arms_declared_corpora_are_the_ones_that_can_be_streamed(self):
        for arm in R.eligible_arms(CAMPAIGN_PANEL):
            self.assertIn(R.arm_training_corpus(arm), CORPUS_SOURCES)
            self.assertIn(R.arm_evaluation_cohort_source(arm), CORPUS_SOURCES)

    def test_progen3_trains_and_evaluates_on_different_corpora_and_says_so(self):
        self.assertEqual(R.arm_training_corpus("progen3"), "uniref50")
        self.assertEqual(R.arm_evaluation_cohort_source("progen3"), "swissprot")

    def test_a_dense_arm_trains_on_the_corpus_it_is_scored_on(self):
        for arm in ("gpt2-large", "protgpt2"):
            self.assertEqual(
                R.arm_training_corpus(arm), R.arm_evaluation_cohort_source(arm)
            )
        self.assertEqual(R.arm_training_corpus("gpt2-large"), "openwebtext")
        self.assertEqual(R.arm_training_corpus("protgpt2"), "swissprot")


class HeldOutStream(unittest.TestCase):
    """What the trainer's held-out draw does and does not guarantee.

    The skip is counted in records read and the shuffle permutes *within* a block,
    so the held-out pool is disjoint from training when the skip clears whole
    blocks and **only then** -- a partially consumed final block hands training a
    random subset of it while the held-out pool starts partway into that same
    block. The first two tests pin that property of the streaming primitive, which
    is unchanged and correct.

    The third pins the stage's response to it: the offset is rounded up to a whole
    number of blocks, which removes the overlap exactly. EXP-R2-136 and
    EXP-R2-138 ran before that repair and carry a 6.25% leak at their own setting;
    it is symmetric across their arms, so it bounds their absolute held-out NMSE
    without touching the comparison they claim.
    """

    BLOCK = 8192

    @staticmethod
    def _records(n: int):
        return lambda: iter(f"record-{index}" for index in range(n))

    def _stream(self, total, *, skip, limit, seed=1):
        return list(
            STAGE17.stream_records(self._records(total), seed=seed, skip=skip, limit=limit)
        )

    def test_a_whole_block_training_budget_gives_a_disjoint_held_out_set(self):
        consumed, evaluation = self.BLOCK, 8
        offset = consumed + evaluation
        held_out = self._stream(2 * self.BLOCK, skip=offset, limit=evaluation)
        training = STAGE17.stream_records(
            self._records(2 * self.BLOCK), seed=1, skip=0, limit=None
        )
        seen = {next(training) for _ in range(consumed)}
        self.assertEqual(len(held_out), evaluation)
        self.assertEqual(set(held_out) & seen, set())

    def test_a_partial_final_block_leaks_and_the_leak_is_bounded_by_the_remainder(self):
        # The limitation, measured rather than described. Training consumes half a
        # block, so its records are a random half of that block while the held-out
        # pool is the same block minus a file-order prefix -- and they intersect.
        consumed, evaluation = self.BLOCK // 2, 64
        offset = consumed + evaluation
        held_out = self._stream(self.BLOCK, skip=offset, limit=evaluation)
        training = STAGE17.stream_records(
            self._records(self.BLOCK), seed=1, skip=0, limit=None
        )
        seen = {next(training) for _ in range(consumed)}
        overlap = len(set(held_out) & seen)
        self.assertGreater(
            overlap,
            0,
            "this test exists to pin a known leak; if it is gone the discipline "
            "changed and the published transcoder runs are no longer reproducible",
        )
        # Bounded by the fraction of the block training emitted, which is what
        # makes the real configuration's exposure small rather than absent.
        self.assertLess(overlap, evaluation)

    def test_the_stream_is_shuffled_rather_than_the_file_order(self):
        # Appendix B rule 1: a prefix of a corpus grouped by cluster or shard is a
        # region rather than a sample.
        drawn = list(STAGE17.stream_records(self._records(20), seed=7, skip=0, limit=20))
        self.assertEqual(sorted(drawn), sorted(f"record-{i}" for i in range(20)))
        self.assertNotEqual(drawn, [f"record-{i}" for i in range(20)])

    def test_the_same_seed_gives_the_same_stream(self):
        first = list(STAGE17.stream_records(self._records(64), seed=3, skip=0, limit=16))
        again = list(STAGE17.stream_records(self._records(64), seed=3, skip=0, limit=16))
        other = list(STAGE17.stream_records(self._records(64), seed=4, skip=0, limit=16))
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class HeldOutOffsetClearsWholeBlocks(unittest.TestCase):
    """The stage must not choose an offset that lands mid-block.

    Pinned on the arithmetic rather than on a run, because the failure it guards
    is silent: a mid-block offset produces a held-out set that looks disjoint,
    reports a plausible NMSE, and is partly the training set.
    """

    def test_the_offset_is_a_whole_number_of_shuffle_blocks(self):
        block = STAGE17.SHUFFLE_BLOCK
        for steps, batch in ((20_000, 16), (300, 4), (1, 1), (block, 1), (12_345, 7)):
            consumed = steps * batch
            offset = -(-consumed // block) * block
            self.assertEqual(offset % block, 0, f"{steps}x{batch} lands mid-block")
            self.assertGreaterEqual(offset, consumed, f"{steps}x{batch} overlaps training")
            self.assertLess(
                offset - consumed, block, f"{steps}x{batch} skips more than one block"
            )

    def test_the_campaign_setting_would_have_leaked_under_the_old_rule(self):
        block, consumed, evaluation = STAGE17.SHUFFLE_BLOCK, 20_000 * 16, 256
        self.assertNotEqual((consumed + evaluation) % block, 0)
        self.assertEqual(-(-consumed // block) * block, 327_680)
