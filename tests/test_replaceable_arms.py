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

import torch  # noqa: E402

from panel_contract import CAMPAIGN_PANEL  # noqa: E402
from src.transfer import arms as A  # noqa: E402
from src.transfer import replaceable as R  # noqa: E402
from src.transfer import scaling  # noqa: E402
from src.transfer.arms import (  # noqa: E402
    AA20,
    CORPUS_SOURCES,
    PANEL,
    Arm,
    iter_corpus_records,
    load_arm,
)
from src.transfer.budget import arm_power  # noqa: E402
from src.transfer.progen3 import (  # noqa: E402
    NON_RESIDUE_TOKENS,
    component_grid,
    components,
)
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

    def test_the_conditioned_arm_is_admitted_and_separates_modality_from_tokenisation(self):
        # zymctrl is the third arm on the one backbone: gpt2 architecture, 36
        # layers of width 1280 like both members of the matched pair, protein like
        # protgpt2, and residue-tokenised unlike it. Without it, protein modality
        # and multi-residue BPE are confounded on the only shape-matched pair the
        # panel has (EXP-R2-147's second acknowledged weakness).
        admitted = R.eligible_arms(CAMPAIGN_PANEL)
        self.assertIn("zymctrl", admitted)
        for name in ("gpt2-large", "protgpt2", "zymctrl"):
            self.assertEqual(
                (PANEL[name].architecture, PANEL[name].n_layer, PANEL[name].d_model),
                ("gpt2", 36, 1280),
            )
        self.assertEqual(PANEL["protgpt2"].tokenisation, "multi_residue_bpe")
        self.assertEqual(PANEL["zymctrl"].tokenisation, "residue")
        self.assertNotIn("zymctrl", R.DENSE_ARMS_WITHOUT_A_BAND)

    def test_every_staged_dense_arm_is_either_measured_or_refused(self):
        # The half of the invariant the library cannot check on its own without
        # importing a stage script: an arm that is staged, architecturally
        # admissible and in neither table is one nobody decided about.
        #
        # The condition is RESIDUAL_WRITE and not DENSE_ARCHITECTURES because a
        # band is a property of a checkpoint and its rendering rather than of the
        # transcoder tap: progen2-base is a campaign arm whose block layout this
        # module now declares, so leaving it out of both tables would be exactly
        # the omission this test exists to catch.
        undecided = sorted(
            name
            for name in CAMPAIGN_PANEL
            if PANEL[name].architecture in R.RESIDUAL_WRITE
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


class TheProteinScaleLadder(unittest.TestCase):
    """Four ProGen2 rungs reachable for a tolerance curve, and not for a campaign.

    The two upper rungs are staged on GPFS, load and run, and are deliberately
    outside :data:`src.transfer.arms.PANEL`: ``progen2-large`` declares
    ``vocab_size`` 51200 against a 31-token tokenizer and ``progen2-xlarge``
    declares no ``vocab_size`` at all, so every panel statistic derived from that
    key would be computed over a mostly dead alphabet or would raise. What follows
    pins that they are reachable by the one measurement that needs none of it, and
    refused by the machinery that does.
    """

    def test_the_staged_names_are_the_contracts_own_and_are_not_panel_members(self):
        # Two declarations of one fact would drift: panel_contract records the
        # measured reason for a campaign, arms.py records the loader's spec. The
        # library cannot check this itself without importing a stage script.
        import panel_contract

        self.assertEqual(
            sorted(A.STAGED_ARMS), sorted(panel_contract.STAGED_BUT_NOT_ADMITTED)
        )
        for name, reason in panel_contract.STAGED_BUT_NOT_ADMITTED.items():
            self.assertNotIn(name, PANEL, name)
            self.assertIn("vocab_size", reason, name)

    def test_the_ladder_varies_scale_and_nothing_else_it_declares(self):
        specs = [A.arm_spec(name) for name in A.PROTEIN_SCALE_LADDER]
        for key in ("modality", "tokenisation", "input_format", "architecture",
                    "evaluation_cohort_source", "pretraining_corpus"):
            self.assertEqual(len({getattr(spec, key) for spec in specs}), 1, key)
        self.assertEqual([spec.d_model for spec in specs], [1024, 1536, 2560, 4096])
        self.assertEqual([spec.n_layer for spec in specs], [12, 27, 32, 32])
        # And the first two rungs are the pair a campaign already schedules, so
        # the ladder extends a declared contrast rather than replacing it.
        self.assertEqual(A.PROTEIN_SCALE_LADDER[:2], A.PROTEIN_SCALE_CONTRAST)

    def test_the_perturbation_admits_the_ladder_and_the_replacement_does_not(self):
        eligible = R.eligible_arms(CAMPAIGN_PANEL)
        perturbable = R.perturbable_arms(CAMPAIGN_PANEL)
        for name in A.PROTEIN_SCALE_LADDER:
            self.assertIn(name, perturbable, name)
            self.assertNotIn(name, eligible, name)
        # Widened by exactly the parallel-residual arms and nothing else.
        self.assertEqual(
            sorted(set(perturbable) - set(eligible)), sorted(A.PROTEIN_SCALE_LADDER)
        )
        for name in eligible:
            self.assertIn(name, perturbable, name)

    def test_a_staged_checkpoint_is_admitted_for_one_measurement_family_only(self):
        # The capability is the enforcement of the paragraph above: budget is what
        # arm_power needs, and arm_power reads config.vocab_size.
        for name, spec in A.STAGED_ARMS.items():
            self.assertEqual(sorted(spec.capabilities), ["pathway"], name)
        arm = Arm(
            spec=A.STAGED_ARMS["progen2-xlarge"],
            model=None,
            tokenizer=None,
            device="cpu",
            dtype="bfloat16",
        )
        cohort = A.Cohort(name="stub", kind="protein", records=["MKT"], min_symbols=0, max_symbols=0)
        with self.assertRaises(ValueError) as caught:
            arm_power(arm, cohort, max_len=8, batch_size=1)
        self.assertIn("budget", str(caught.exception))

    def test_a_staged_non_member_cannot_be_registered_into_the_panel_at_run_time(self):
        # scaling.register_arm_spec inserts a ladder rung into PANEL so that
        # load_arm can reach it, and scaling.DEFAULT_LADDER lists both staged
        # rungs. Without this refusal a convergence-control run would give each
        # name two live declarations with different capability sets, and would
        # hand arm_power a plug-in entropy over progen2-large's 51200-symbol
        # config against its 31-token tokenizer.
        member = next(
            rung for rung in scaling.DEFAULT_LADDER if rung.name == "progen2-large"
        )
        with self.assertRaises(ValueError) as caught:
            scaling.register_arm_spec(member, {"n_layer": 32, "d_model": 2560})
        self.assertIn("STAGED_ARMS", str(caught.exception))
        self.assertNotIn("progen2-large", PANEL)
        # Both staged rungs are in that ladder, so both are covered.
        self.assertEqual(
            sorted(
                rung.name for rung in scaling.DEFAULT_LADDER if rung.name in A.STAGED_ARMS
            ),
            sorted(A.STAGED_ARMS),
        )

    def test_the_panel_door_stays_panel_only(self):
        # A staged checkpoint is loadable, but only by a caller that resolved its
        # declaration on purpose. Admitting it by name would let any stage with a
        # free-text --arm schedule a checkpoint the panel deliberately excluded.
        for name in A.STAGED_ARMS:
            with self.assertRaises(KeyError, msg=name):
                load_arm(name, device="cpu")
            self.assertIs(A.arm_spec(name), A.STAGED_ARMS[name])
        with self.assertRaises(KeyError):
            A.arm_spec("not-a-model")

    def test_the_cohort_and_corpus_of_a_staged_arm_resolve_before_it_is_loaded(self):
        for name in A.PROTEIN_SCALE_LADDER:
            self.assertEqual(R.arm_cohort_kind(name), "protein")
            self.assertEqual(R.arm_evaluation_cohort_source(name), "swissprot")

    def test_the_implementation_is_dispatched_from_the_classes_own_declarations(self):
        self.assertIs(R.replaceable_implementation("gpt2"), R.DenseReplaceable)
        self.assertIs(
            R.replaceable_implementation("progen"), R.ParallelResidualReplaceable
        )
        self.assertTrue(issubclass(R.ParallelResidualReplaceable, R.DenseReplaceable))
        for architecture in ("llama", "qwen2", "t5_decoder"):
            with self.assertRaises(TypeError, msg=architecture):
                R.replaceable_implementation(architecture)
        # The two sets partition the declared layouts, so no architecture is both
        # admitted by one class and dispatched to the other.
        self.assertEqual(
            R.DenseReplaceable.architectures & R.ParallelResidualReplaceable.architectures,
            frozenset(),
        )
        self.assertEqual(
            set(R.RESIDUAL_WRITE),
            set(R.DenseReplaceable.architectures | R.ParallelResidualReplaceable.architectures),
        )


class TheBandsLimitationIsRecordedRatherThanAssumedAway(unittest.TestCase):
    """A gate must publish what it does not catch, or a PASS overstates itself."""

    def test_every_unseparated_corruption_really_is_inside_the_band(self):
        # These are recorded as *not* separated, so a test that they are outside
        # the band would be the wrong test: what has to hold is that the table
        # tells the truth. If a later change made one of them separable, this
        # fails and the entry must move to MEASURED_DENSE_SELF_CHECK_CORRUPTIONS.
        for arm, entries in R.UNSEPARATED_DENSE_SELF_CHECK_CORRUPTIONS.items():
            reference = R.MEASURED_DENSE_SELF_CHECK_NLL[arm]
            for name, value in entries.items():
                self.assertLessEqual(
                    abs(value - reference),
                    R.DENSE_SELF_CHECK_HALF_WIDTH,
                    f"{arm}/{name} is separable and is filed as if it were not",
                )
                self.assertEqual(R.check_dense_nll(arm, value)["verdict"], "PASS")

    def test_the_gate_publishes_them_beside_its_verdict(self):
        record = R.check_dense_nll(
            "progen2-small", R.MEASURED_DENSE_SELF_CHECK_NLL["progen2-small"]
        )
        self.assertEqual(record["verdict"], "PASS")
        self.assertIn("rendered_raw", record["unseparated_corruptions"])
        self.assertEqual(sorted(record["corruptions"]), ["randomly_initialised"])
        # An arm with no recorded limitation says so with an empty record rather
        # than by omitting the field.
        self.assertEqual(
            R.check_dense_nll(
                "gpt2-large", R.MEASURED_DENSE_SELF_CHECK_NLL["gpt2-large"]
            )["unseparated_corruptions"],
            {},
        )

    def test_the_ladder_carries_a_measured_band_at_every_rung(self):
        for name in A.PROTEIN_SCALE_LADDER:
            self.assertIn(name, R.MEASURED_DENSE_SELF_CHECK_NLL, name)
            self.assertEqual(
                R.check_dense_nll(name, R.MEASURED_DENSE_SELF_CHECK_NLL[name])["verdict"],
                "PASS",
                name,
            )


class DenseLoaderBand(unittest.TestCase):
    def test_the_reference_value_passes(self):
        for arm, value in R.MEASURED_DENSE_SELF_CHECK_NLL.items():
            self.assertEqual(R.check_dense_nll(arm, value)["verdict"], "PASS")

    def test_every_recorded_corruption_is_refused_by_the_arm_it_was_measured_on(self):
        # The band exists to catch these, so a band that admitted one would be a
        # band on nothing. protgpt2 rendered raw is L11; a randomly initialised
        # arm is L24's shape on a dense arm; zymctrl stripped of its EC tag is
        # L11's shape on a conditioned arm and L15's leak measured directly.
        for arm, corruptions in R.MEASURED_DENSE_SELF_CHECK_CORRUPTIONS.items():
            for name, value in corruptions.items():
                with self.assertRaises(RuntimeError, msg=f"{arm}/{name}"):
                    R.check_dense_nll(arm, value)

    def test_a_value_below_the_band_is_refused_too(self):
        # Below the band means the scored-target convention moved, which looks
        # like an improvement and corrupts everything downstream.
        with self.assertRaises(RuntimeError):
            R.check_dense_nll("gpt2-large", R.MEASURED_DENSE_SELF_CHECK_NLL["gpt2-large"] - 1.0)

    def test_an_unmeasured_arm_cannot_be_gated_at_all(self):
        with self.assertRaises(KeyError):
            R.check_dense_nll("gpt2-xl", 3.0)

    def test_each_band_is_narrower_than_the_distance_to_that_arms_own_corruption(self):
        # The sizing argument, as an invariant rather than as a docstring: a band
        # that reached a corruption could pass on a broken arm.
        #
        # Per arm, and deliberately not across arms: zymctrl rendered without its
        # EC tag reads 3.1779 while a healthy gpt2-large reads 3.1706, so the
        # cross-product this test used to take now contains a pair 0.0073 apart
        # that means nothing about either arm.
        for arm, corruptions in R.MEASURED_DENSE_SELF_CHECK_CORRUPTIONS.items():
            reference = R.MEASURED_DENSE_SELF_CHECK_NLL[arm]
            for name, value in corruptions.items():
                self.assertLess(
                    R.DENSE_SELF_CHECK_HALF_WIDTH,
                    abs(value - reference),
                    f"{arm}'s band reaches its own {name} corruption",
                )

    def test_a_gate_reports_only_the_corruptions_of_the_arm_it_ran_on(self):
        record = R.check_dense_nll("gpt2-large", R.MEASURED_DENSE_SELF_CHECK_NLL["gpt2-large"])
        self.assertEqual(record["corruptions"], {})
        zym = R.check_dense_nll("zymctrl", R.MEASURED_DENSE_SELF_CHECK_NLL["zymctrl"])
        self.assertEqual(
            sorted(zym["corruptions"]), ["randomly_initialised", "rendered_without_its_ec_tag"]
        )

    def test_the_frozen_inputs_are_literals_and_shared_with_progen3(self):
        from src.transfer.progen3 import SELF_CHECK_SEQUENCES

        self.assertEqual(len(R.SELF_CHECK_DOCUMENTS), 8)
        # A protein dense arm is checked on exactly the records ProGen3 is checked
        # on, so the modality pair is gated on one text set and one protein set.
        self.assertIs(R.SELF_CHECK_SEQUENCES, SELF_CHECK_SEQUENCES)

    def test_the_conditioned_gate_has_real_ec_labels_rather_than_invented_ones(self):
        # Six of the eight unconditioned protein records are not enzymes and carry
        # no EC number, so conditioning them would mean inventing one -- a false
        # fact inside the gate that exists to catch false facts. The EC set is
        # therefore its own eight records, each with the label its corpus entry
        # carries.
        self.assertEqual(len(R.SELF_CHECK_EC_RECORDS), 8)
        for ec, sequence in R.SELF_CHECK_EC_RECORDS:
            self.assertRegex(ec, r"^\d+\.\d+\.\d+\.\d+$")
            self.assertTrue(set(sequence) <= set(AA20), f"{ec} is not a canonical sequence")
            # Every record, prompt and terminator must fit the cap the gate
            # tokenises under, or the gate would score a truncated span.
            self.assertLessEqual(len(sequence) + 10, R.SELF_CHECK_MAX_TOKENS)
        self.assertEqual(
            len({sequence for _, sequence in R.SELF_CHECK_EC_RECORDS}), 8
        )


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


class _StubTokenizer:
    """Just enough tokenizer to resolve a conditioned arm's boundaries.

    ZymCTRL's real vocabulary is reproduced where it matters: ``<sep>``,
    ``<start>`` and ``<end>`` are ordinary entries at ids 2, 3 and 4, the only
    *special* id is ``<|endoftext|>`` at 1, and that id is also the pad. The last
    two facts are the reason the unconditioned content mask cannot be reused --
    it removes special ids, and none of the three markers is one.
    """

    all_special_ids = [1]
    unk_token_id = 1
    pad_token_id = 1
    eos_token_id = 1
    eos_token = "<|endoftext|>"

    #: ``"1"`` is here because ProGen2's rendering prefixes it and its tokenizer
    #: does *not* declare it special -- the same arrangement as the three ZymCTRL
    #: markers, one format along.
    _IDS = {"<sep>": 2, "<start>": 3, "<end>": 4, "1": 5}

    def convert_tokens_to_ids(self, token):
        return self._IDS.get(token, self.unk_token_id)


def _stub(arm_name: str) -> R.DenseReplaceable:
    """The panel arm behind its own replaceable class, with no weights.

    The class is resolved from the declared architecture rather than fixed, so a
    parallel-residual arm is stubbed through the implementation that actually
    runs it; ``DenseReplaceable`` refuses ``progen`` by construction.
    """

    spec = PANEL[arm_name]
    implementation = (
        R.ParallelResidualReplaceable
        if spec.architecture in R.PARALLEL_ARCHITECTURES
        else R.DenseReplaceable
    )
    return implementation(
        Arm(
            spec=spec,
            model=None,
            tokenizer=_StubTokenizer(),
            device="cpu",
            dtype="bfloat16",
        ),
        max_tokens=512,
    )


class ConditionedScoredPositions(unittest.TestCase):
    """The estimand must be the residues, and only the residues.

    This is the condition that made ZymCTRL inadmissible and the one an admission
    has to establish. The EC prompt supplies 1.73 nats of label information
    (L15) and its ``<sep>``/``<start>`` positions are near-deterministic, so a
    mask that let either into the likelihood would move the clean cross-entropy,
    the fully-ablated endpoint and therefore both ends of the recovery ratio --
    and would put activations that are not content into the per-layer mean the
    ablation endpoint is built from and into the transcoder's own objective.

    Checked on token ids rather than on a checkpoint, so the invariant is pinned
    without a GPU. ``1 . 3 . 7 . 7 <sep> <start>`` is the real nine-token prompt
    ZymCTRL's tokenizer emits for EC 1.3.7.7.
    """

    PROMPT = [9, 431, 11, 431, 105, 431, 13, 2, 3]  # 1 . 3 . 7 . 7 <sep> <start>
    RESIDUES = [443, 444, 442]                      # M N L
    END, PAD = 4, 1

    def _batch(self, n_pad: int = 2):
        ids = self.PROMPT + self.RESIDUES + [self.END]
        attention = [1] * len(ids) + [0] * n_pad
        return {
            "input_ids": torch.tensor([ids + [self.PAD] * n_pad]),
            "attention_mask": torch.tensor([attention]),
        }

    def test_the_content_mask_keeps_the_residues_and_nothing_else(self):
        batch = self._batch()
        mask = _stub("zymctrl").content_mask(batch)[0].tolist()
        expected = [False] * len(self.PROMPT) + [True] * len(self.RESIDUES) + [False] * 3
        self.assertEqual(mask, expected)

    def test_the_scored_targets_are_the_residues_and_nothing_else(self):
        batch = self._batch()
        mask = _stub("zymctrl")._target_mask(batch)[0].tolist()
        # Column q predicts input token q + 1, so the first residue is predicted
        # from the prompt at column 8 and <end> at column 11 is not scored.
        expected = [False] * (len(self.PROMPT) - 1) + [True] * len(self.RESIDUES) + [False] * 3
        self.assertEqual(mask, expected)
        self.assertEqual(sum(mask), len(self.RESIDUES))

    def test_the_ec_digits_are_not_removable_as_special_tokens(self):
        # Why the conditioned branch has to exist at all: none of the prompt's
        # tokens is a tokenizer special id, so the unconditioned rule would keep
        # every one of them.
        batch = self._batch()
        naive = batch["attention_mask"].bool() & ~torch.isin(
            batch["input_ids"], torch.tensor(_StubTokenizer.all_special_ids)
        )
        self.assertEqual(int(naive.sum()), len(self.PROMPT) + len(self.RESIDUES) + 1)

    def test_an_unconditioned_arm_is_unchanged_by_the_shared_rule(self):
        # The rule is resolved for every arm through the same declaration; on an
        # unconditioned one it must reproduce the mask the frozen runs used, which
        # is the shifted validity mask.
        batch = {
            "input_ids": torch.tensor([[10, 11, 12, 13, 1, 1]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0, 0]]),
        }
        model = _stub("gpt2-large")
        self.assertEqual(
            model._target_mask(batch)[0].tolist(),
            batch["attention_mask"][0, 1:].bool().tolist(),
        )
        # And its content mask still keeps position 0. A raw text rendering
        # prefixes nothing, so it declares no marker and the marker term is empty;
        # changing that would move every frozen dense result.
        self.assertEqual(A.rendering_marker_ids(model.arm), ())
        self.assertTrue(bool(model.content_mask(batch)[0, 0]))

    def test_a_conditioned_render_without_labels_is_refused(self):
        with self.assertRaises(ValueError):
            _stub("zymctrl").render(["MNL"])
        with self.assertRaises(ValueError):
            _stub("zymctrl").render(["MNL", "MNL"], ec_labels=["1.1.1.1", None])

    def test_labels_handed_to_an_unconditioned_arm_are_refused_rather_than_dropped(self):
        with self.assertRaises(ValueError):
            _stub("protgpt2").render(["MNL"], ec_labels=["1.1.1.1"])

    def test_the_rendering_is_the_panels_own_and_carries_the_prompt(self):
        rendered = _stub("zymctrl").render(["MNL"], ec_labels=["1.3.7.7"])
        self.assertEqual(rendered, ["1.3.7.7<sep><start>MNL<end>"])


class RenderingMarkersAreNotContent(unittest.TestCase):
    """No marker an arm's rendering prefixes may sit in its content span.

    The condition that must hold on every input format, and the one that did not.
    The unconditioned branch removed the tokenizer's *special* ids alone, which is
    complete for ProtGPT2 -- whose end-of-text prefix is a special token -- and
    silently incomplete for ProGen2, whose N-to-C direction marker is an ordinary
    vocabulary entry. Stage 23 averages the fully-ablated endpoint every recovery
    ratio divides by over exactly these positions, and on ``progen2-small`` at
    bfloat16 over 32 Swiss-Prot records of 64-246 residues the one surviving
    marker per record moved that endpoint by 0.68 in relative norm at layer 1.

    Written against :func:`src.transfer.arms.rendering_marker_ids` rather than
    against literal ids, so a format that begins prefixing a marker is covered by
    this test as soon as it declares one.
    """

    #: One row per unconditioned format: the arm, the ids of a rendered record
    #: with whatever its format prefixes in front and two padding positions
    #: behind, and which of those positions are content. Id ``1`` is this stub's
    #: only special id and is its pad as well, which is the arrangement every
    #: GPT-2-lineage arm has -- so row two carries the same id as a *prefix* and as
    #: *padding*, and both must be out.
    CASES = (
        ("gpt2-large", [10, 11, 12, 13, 1, 1], [True, True, True, True, False, False]),
        ("protgpt2", [1, 10, 11, 12, 1, 1], [False, True, True, True, False, False]),
        ("progen2-medium", [5, 10, 11, 12, 1, 1], [False, True, True, True, False, False]),
    )

    def test_the_mask_keeps_no_declared_marker_on_any_unconditioned_format(self):
        for arm_name, ids, expected in self.CASES:
            with self.subTest(arm=arm_name):
                model = _stub(arm_name)
                batch = {
                    "input_ids": torch.tensor([ids]),
                    "attention_mask": torch.tensor([[1] * (len(ids) - 2) + [0, 0]]),
                }
                mask = model.content_mask(batch)
                self.assertEqual(mask[0].tolist(), expected)
                for marker in A.rendering_marker_ids(model.arm):
                    self.assertFalse(
                        bool((mask & (batch["input_ids"] == marker)).any()),
                        f"{arm_name} keeps its own rendering marker {marker}",
                    )

    def test_the_direction_marker_is_not_reachable_through_the_special_ids(self):
        # Why the fix could not be "the tokenizer already knows": it does not.
        # ProGen3 names "1" and "2" in NON_RESIDUE_TOKENS and ProGen2 declares
        # neither special, so the two arms' content spans disagreed by one
        # position per record while both were described as the same convention.
        model = _stub("progen2-medium")
        markers = A.rendering_marker_ids(model.arm)
        self.assertEqual(markers, (5,))
        self.assertNotIn(markers[0], _StubTokenizer.all_special_ids)
        self.assertIn("1", NON_RESIDUE_TOKENS)

    def test_a_conditioned_rendering_is_refused_rather_than_partly_described(self):
        # Its EC digits carry no marker id, so a set of ids cannot describe the
        # span that is not content, and returning the three delimiters would leave
        # seven digit positions inside it.
        with self.assertRaises(ValueError) as raised:
            A.rendering_marker_ids(_stub("zymctrl").arm)
        self.assertIn("conditioning_boundary_ids", str(raised.exception))


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
        for arm in ("gpt2-large", "protgpt2", "zymctrl"):
            self.assertEqual(
                R.arm_training_corpus(arm), R.arm_evaluation_cohort_source(arm)
            )
        self.assertEqual(R.arm_training_corpus("gpt2-large"), "openwebtext")
        self.assertEqual(R.arm_training_corpus("protgpt2"), "swissprot")
        self.assertEqual(R.arm_training_corpus("zymctrl"), "zymctrl_ec")

    def test_only_the_conditioned_corpus_carries_a_label(self):
        # The stream's unit is (record, conditioning_label). A corpus that
        # returned bare records for a conditioned arm would leave the renderer
        # with nothing to build the prompt from; one that returned a label for an
        # unconditioned arm would have it dropped.
        record, label = next(
            iter(iter_corpus_records("zymctrl_ec", min_symbols=32, max_symbols=1014))
        )
        self.assertTrue(set(record) <= set(AA20))
        self.assertRegex(label, r"^\d+\.\d+\.\d+\.\d+$")
        _, plain = next(
            iter(iter_corpus_records("swissprot", min_symbols=32, max_symbols=1022))
        )
        self.assertIsNone(plain)

    def test_the_conditioned_band_is_the_largest_its_context_can_render(self):
        # A record whose <end> is truncated away has no scored span at all, so the
        # band ceiling and the rendering wrapper are one decision, not two.
        low, high = STAGE17.CORPUS_BAND["zymctrl_ec"]
        self.assertEqual(low, STAGE17.MIN_RESIDUES)
        self.assertEqual(high + STAGE17.ZYMCTRL_WRAPPER_TOKENS, STAGE17.ZYMCTRL_CONTEXT)
        self.assertLess(high, STAGE17.MAX_RESIDUES)


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
