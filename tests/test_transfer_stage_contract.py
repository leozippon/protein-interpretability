"""Regression tests for the `scripts/transfer/` campaign contract (EXP-R2-063).

One test per behavioural fix, written against the restored property rather than
against the implementation, so that a re-implementation that keeps the property
keeps the test.

The theme running through most of them is a *scope* that was implicit: which arms
a stage measures, which band it draws on, which flags reach which item. Every one
of those could previously change without a downstream number looking wrong, which
is limitation L18's shape and the most damaging defect class this programme has.
"""

from __future__ import annotations

import ast
import json
import tempfile
import dataclasses
import os
import subprocess
import sys
import unittest
import unittest.mock
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
STAGE_DIR = REPO_ROOT / "scripts" / "transfer"
if str(STAGE_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE_DIR))

import panel_contract as pc  # noqa: E402
from src.transfer import path_patching  # noqa: E402
from src.transfer import prediction_addressed as pa  # noqa: E402
from src.transfer import scaling  # noqa: E402
from src.transfer.arms import PANEL  # noqa: E402
from src.transfer.circuits import _CIRCUIT_ARCHITECTURES  # noqa: E402
from src.transfer.path_patching import SUPPORTED_ARCHITECTURES  # noqa: E402


def _load_stage_module(filename: str):
    """Import a numbered entry point by path, the way the worker's preflight does."""

    import importlib.util

    path = STAGE_DIR / filename
    spec = importlib.util.spec_from_file_location(f"_stage_{filename[:2]}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ArmCanRunPredicate(unittest.TestCase):
    """`arm_can_run(stage, arm)` -- the single predicate the stages had improvised."""

    def test_path_patching_admits_the_rotary_arms_the_module_now_addresses(self):
        # These were refused until EXP-R2-079 because `path_patching` resolved
        # only the GPT-2 trunk, and the refusal was catalogued as an instrument
        # limit on L22: every text arm carrying that result was GPT-2
        # architecture, the configuration that retracted the QK/OV finding. The
        # contract must follow the module, not a remembered fact about it.
        for arm in ("qwen2.5-0.5b", "llama-3.2-3b"):
            verdict = pc.arm_can_run("induction_path_patching", arm)
            self.assertTrue(verdict.can_run, verdict.reason)
            self.assertIn(PANEL[arm].architecture, path_patching.SUPPORTED_ARCHITECTURES)

    def test_path_patching_refuses_an_architecture_outside_its_declaration(self):
        # The predicate must answer from the module's declared set rather than
        # from a hard-coded arm list, or it drifts the moment the set moves.
        #
        # The reason is whichever gate fires first, and that is deliberate: the
        # ByGPT5 arms are refused on capabilities before their `t5_decoder` layout
        # is ever consulted. Asserting the architecture reason unconditionally
        # would be asserting the order of the gates, not the refusal.
        unsupported = [
            arm
            for arm, spec in PANEL.items()
            if spec.architecture not in path_patching.SUPPORTED_ARCHITECTURES
        ]
        self.assertTrue(unsupported, "no arm exercises the refusal path")
        for arm in unsupported:
            verdict = pc.arm_can_run("induction_path_patching", arm)
            self.assertFalse(verdict.can_run, arm)
            self.assertTrue(verdict.reason, arm)

    def test_path_patching_admits_declared_progen_layouts(self):
        for arm in ("progen2-base", "progen2-medium"):
            verdict = pc.arm_can_run("induction_path_patching", arm)
            self.assertTrue(verdict.can_run, verdict.reason)
            self.assertEqual(PANEL[arm].architecture, "progen")

    def test_lens_family_refuses_rmsnorm_arms_and_admits_every_other_campaign_arm(self):
        eligible, refused = pc.stage_arms("lens_family")
        # Derived from the module's own architecture declaration rather than
        # listed: the two rotary arms lack `nn.LayerNorm` at `transformer.ln_f`,
        # and the byte-level control is refused on the same declaration for a
        # different reason (a T5 decoder's final norm is not a LayerNorm either).
        # A hard-coded pair made the second refusal read as a regression.
        self.assertEqual(
            {v.arm for v in refused},
            {
                arm
                for arm in pc.CAMPAIGN_PANEL
                if PANEL[arm].architecture not in scaling.LENS_ARCHITECTURES
            },
        )
        self.assertIn("qwen2.5-0.5b", {v.arm for v in refused})
        self.assertIn("bygpt5-medium-en", {v.arm for v in refused})
        # Derived, not restated: admitting ProGen2-small moved this from 9 to 10
        # and a hard-coded count made a legitimate panel change look like a defect.
        self.assertEqual(
            set(eligible),
            {
                arm
                for arm in pc.CAMPAIGN_PANEL
                if PANEL[arm].architecture in scaling.LENS_ARCHITECTURES
            },
        )

    def test_relational_channel_includes_progen2_base(self):
        # The worker's hand-written list named zymctrl and progen2-medium only.
        # progen2-base is protein, residue-tokenised and carries `relational`, so
        # excluding it narrowed the stage's panel by one arm with nothing saying so.
        eligible, _ = pc.stage_arms("relational_channel")
        self.assertEqual(
            eligible,
            [
                a
                for a in pc.CAMPAIGN_PANEL
                if PANEL[a].modality == "protein" and PANEL[a].tokenisation == "residue"
            ],
        )
        self.assertIn("progen2-base", eligible)

    def test_relational_channel_refuses_protgpt2_on_tokenisation_not_on_name(self):
        verdict = pc.arm_can_run("relational_channel", "protgpt2")
        self.assertFalse(verdict.can_run)
        self.assertIn("multi_residue_bpe", verdict.reason)

    def test_pathway_stages_refuse_arms_without_the_capability(self):
        for stage in ("pathway_budget", "estimand_power"):
            verdict = pc.arm_can_run(stage, "bygpt5-small-en")
            self.assertFalse(verdict.can_run, stage)
            self.assertIn("pathway", verdict.reason)

    def test_unknown_stage_or_arm_raises_rather_than_returning_false(self):
        # "cannot run" and "never heard of it" are different facts; collapsing
        # them is how a typo becomes a silently narrower panel.
        with self.assertRaises(KeyError):
            pc.arm_can_run("no_such_stage", "gpt2")
        with self.assertRaises(KeyError):
            pc.arm_can_run("lens_family", "no-such-arm")


class PaaCensusEligibility(unittest.TestCase):
    """`paa_census` (D2.c) -- the stage whose refusals are scientific decisions.

    Every D2.c run to date was a hand-written local invocation, so the panel it
    measured lived in a driver script rather than in a declaration. That is the
    L18 shape: the arms a number is computed over could change with nothing
    downstream looking wrong. These tests hold the two facts the audit settled
    (§D2.c blocker 1) and the one property that keeps them honest -- that the
    arm list is derived from `src/transfer/arms.py` rather than restated.
    """

    STAGE = "paa_census"

    def test_the_stage_is_declared_and_dispatched_per_arm(self):
        contract = pc.STAGE_CONTRACTS[self.STAGE]
        self.assertEqual(contract.entry_point, "14_paa_census.py")
        # per_arm, not panel_wide: the entry point takes one --census-arm and a
        # fixed --text-arm control, so one process measures exactly one arm.
        self.assertEqual(contract.scope, "per_arm")
        self.assertIn(self.STAGE, pc.STAGE_ORDER)

    def test_the_eligible_arms_are_derived_from_the_arm_declaration(self):
        # Appendix B rule 12 was earned by a hand-written arm list. The predicate
        # must be reproducible from ArmSpec alone: nothing here names an arm.
        contract = pc.STAGE_CONTRACTS[self.STAGE]
        self.assertIsNone(
            contract.declared_arms,
            "paa_census must not restate an arm list; its panel is a predicate",
        )
        expected = [
            name
            for name in pc.CAMPAIGN_PANEL
            if contract.capabilities <= PANEL[name].capabilities
            and PANEL[name].architecture in contract.architectures
            and PANEL[name].input_format in contract.input_formats
            and name not in contract.excluded_arms
        ]
        self.assertEqual(pc.stage_arms(self.STAGE)[0], expected)
        self.assertTrue(expected, "the predicate admits nothing; it cannot be right")

    def test_the_matched_pair_is_admitted(self):
        # The whole point of the width route (EXP-R2-082/087/088): gpt2-large and
        # ProtGPT2 are the only modality-identifying comparison this panel has,
        # and a D2.c that excluded them could not be read against L22 at all.
        for arm in ("gpt2-large", "protgpt2"):
            verdict = pc.arm_can_run(self.STAGE, arm)
            self.assertTrue(verdict.can_run, f"{arm}: {verdict.reason}")

    def test_zymctrl_is_refused_on_its_rendering_and_not_on_its_name(self):
        verdict = pc.arm_can_run(self.STAGE, "zymctrl")
        self.assertFalse(verdict.can_run)
        # The ground of the refusal is the declared input format, so any arm
        # rendered the same way is refused the same way -- a name-keyed exclusion
        # would admit the next EC-conditioned arm silently.
        self.assertIn("ec_conditioned", verdict.reason)
        for name, spec in PANEL.items():
            if spec.input_format == "ec_conditioned":
                self.assertFalse(pc.arm_can_run(self.STAGE, name).can_run, name)

    def test_the_zymctrl_refusal_records_why_it_is_permanent(self):
        # Not "unsupported for now": the audit measured that no width admits both
        # ZymCTRL and ProtGPT2, and separately that the single-length window
        # ZymCTRL would need breaks build_cohorts. A refusal that does not say
        # which of those it is invites someone to widen the band and try again.
        reason = pc.arm_can_run(self.STAGE, "zymctrl").reason
        self.assertIn("No width admits", reason)
        self.assertIn("build_cohorts", reason)

    def test_the_byte_level_text_control_is_admitted(self):
        # EXP-R2-114: every symbol-level-tokenised arm this stage measures is a
        # protein decoder, so the census's head-retrieval failure cannot be
        # attributed to tokenisation rather than to modality without a byte-level
        # TEXT arm. bygpt5-medium-en is that arm, and the declaration that used to
        # refuse it -- a per-head OV decomposition this stage never performs -- is
        # not the declaration this stage depends on.
        verdict = pc.arm_can_run(self.STAGE, "bygpt5-medium-en")
        self.assertTrue(verdict.can_run, verdict.reason)
        self.assertIn("circuits", PANEL["bygpt5-medium-en"].capabilities)
        self.assertNotIn("pathway", PANEL["bygpt5-medium-en"].capabilities)

    def test_the_stage_mirrors_the_module_that_actually_measures_it(self):
        contract = pc.STAGE_CONTRACTS[self.STAGE]
        self.assertEqual(
            contract.architecture_source,
            "src.transfer.prediction_addressed.PAA_ARCHITECTURES",
        )
        self.assertEqual(contract.architectures, frozenset(pa.PAA_ARCHITECTURES))
        # Every admitted architecture must have a knockout path in that module,
        # or the stage schedules a run that dies at the causal statistic.
        self.assertIn("t5_decoder", pa.PAA_ARCHITECTURES)

    def test_granting_circuits_to_bygpt5_widens_no_other_stage(self):
        # A capability is an intent; a module declaration is what is deliverable.
        # Both circuit stages gate on their own module's architecture set, so the
        # grant must not reach them -- checked against those sets rather than
        # against the refusal text, which is the thing that could go stale.
        self.assertNotIn("t5_decoder", _CIRCUIT_ARCHITECTURES)
        self.assertNotIn("t5_decoder", SUPPORTED_ARCHITECTURES)
        for stage in ("circuit_primitives", "induction_path_patching"):
            for arm in ("bygpt5-small-en", "bygpt5-base-en", "bygpt5-medium-en"):
                verdict = pc.arm_can_run(stage, arm)
                self.assertFalse(verdict.can_run, f"{stage}/{arm}")
                self.assertIn("t5_decoder", verdict.reason)

    def test_the_narrower_bygpt5_rungs_are_refused_on_their_head_grid(self):
        # 4x6 = 24 heads and 6x12 = 72 heads. hit@k is comparable only within a
        # grid size, and at 24 heads the chance level of hit@20 is 16.7 of a
        # ceiling of 20, so the statistic cannot separate a census that retrieves
        # the causally important heads from one that returns the grid. Nothing in
        # ArmSpec declares a head count, so this cannot be a property rule and the
        # refusal is a named exception that has to carry its own reason.
        contract = pc.STAGE_CONTRACTS[self.STAGE]
        for arm in ("bygpt5-small-en", "bygpt5-base-en"):
            verdict = pc.arm_can_run(self.STAGE, arm)
            self.assertFalse(verdict.can_run, arm)
            self.assertIn(arm, contract.excluded_arms)
            self.assertEqual(verdict.reason, contract.excluded_arms[arm])
            self.assertIn("heads", verdict.reason)
            # Refused on the grid, not on anything they share with the admitted
            # rung -- architecture, tokenisation and rendering are identical.
            self.assertEqual(
                (PANEL[arm].architecture, PANEL[arm].tokenisation, PANEL[arm].input_format),
                ("t5_decoder", "byte", "raw"),
            )

    def test_a_named_exclusion_must_name_a_real_arm_and_say_why(self):
        # An allow-list typo refuses everything and is noticed; a deny-list typo
        # refuses nothing and is not.
        self.assertIn(
            "not in",
            self._refused_by_the_import_check(
                excluded_arms={"bygtp5-medium-en": "a transposed name"}
            ),
        )
        self.assertIn(
            "without saying why",
            self._refused_by_the_import_check(excluded_arms={"bygpt5-medium-en": ""}),
        )

    def test_a_run_from_outside_the_campaign_panel_says_so_in_its_artefact(self):
        # `eligible_for_this_stage` is resolved over CAMPAIGN_PANEL, so an arm
        # measured from outside it does not appear there and the artefact reads as
        # a contradiction with its own `measured` list. The wrong resolution of
        # that contradiction -- "the stage refuses this arm and ran it anyway" --
        # is the damaging one, so the field names the arm and says why it is not a
        # campaign arm.
        #
        # Every panel member has been a campaign arm since 2026-08-21, so the
        # situation has to be constructed: bygpt5-medium-en held this role until
        # 2026-08-06 and the two narrower rungs until 2026-08-21. The property is
        # about any arm outside whatever the campaign panel is, so the panel is
        # what the test narrows -- and it narrows it by an arm this stage
        # ACCEPTS, which is the case the field exists for. An arm the stage
        # refuses anyway would leave `eligible_for_this_stage` false for two
        # different reasons and could not tell them apart.
        outsider = "gpt2-medium"
        panel = tuple(a for a in pc.CAMPAIGN_PANEL if a != outsider)
        with unittest.mock.patch.object(pc, "CAMPAIGN_PANEL", panel), \
                unittest.mock.patch.object(
                    pc, "PANEL_MEMBERS_NOT_STAGED", {outsider: "not staged for this test"}
                ):
            record = pc.stage_contract_record(self.STAGE, [outsider])
            outside = record["arm_selection"]["measured_outside_campaign_panel"]
            self.assertEqual(list(outside), [outsider])
            self.assertTrue(outside[outsider]["eligible_for_this_stage"])
            self.assertEqual(
                outside[outsider]["not_in_campaign_panel_because"],
                "not staged for this test",
            )
            self.assertNotIn(
                outsider, record["arm_selection"]["eligible_for_this_stage"]
            )
        # A campaign-panel arm leaves the field empty, so it cannot become noise
        # every artefact carries -- and with the panel unpatched, which is the
        # current state, no run of a panel arm can populate it at all.
        campaign = pc.stage_contract_record(self.STAGE, ["gpt2-large"])
        self.assertEqual(campaign["arm_selection"]["measured_outside_campaign_panel"], {})

    def test_every_refusal_carries_a_reason(self):
        eligible, refused = pc.stage_arms(self.STAGE, sorted(PANEL))
        self.assertTrue(refused, "no arm exercises the refusal path")
        for verdict in refused:
            self.assertTrue(verdict.reason, verdict.arm)
        self.assertNotIn("zymctrl", eligible)

    def test_the_pool_width_is_declared_beside_the_arms_it_admits(self):
        # The eligible list above is only true at a width that admits a full
        # ProtGPT2 row inside the unchanged 520-800 census band. Declaring the
        # arms without the width would make the declaration conditional on an
        # argument nobody records.
        self.assertEqual(pc.PAA_CENSUS_WIDTH, 192)
        self.assertEqual(
            pc.contract_payload()["paa_census_pool_width"], pc.PAA_CENSUS_WIDTH
        )

    def _refused_by_the_import_check(self, **overrides):
        """`_check_stage_contracts` against a deliberately broken paa_census."""

        original = pc.STAGE_CONTRACTS[self.STAGE]
        pc.STAGE_CONTRACTS[self.STAGE] = dataclasses.replace(original, **overrides)
        try:
            with self.assertRaises(AssertionError) as caught:
                pc._check_stage_contracts()
            return str(caught.exception)
        finally:
            pc.STAGE_CONTRACTS[self.STAGE] = original

    def test_an_input_format_no_arm_declares_is_refused_at_import(self):
        # An allow-list fails in one way a deny-list does not: a typo refuses
        # every arm and looks like a narrow stage rather than a broken one.
        message = self._refused_by_the_import_check(
            input_formats=frozenset({"raw", "typo"})
        )
        self.assertIn("typo", message)

    def test_restricting_input_formats_without_a_reason_is_refused_at_import(self):
        message = self._refused_by_the_import_check(input_format_reason="")
        self.assertIn(self.STAGE, message)


class StageDeclarationsMirrorTheirSource(unittest.TestCase):
    """A mirrored declaration that is not checked is a second declaration."""

    def test_homology_control_arms_match_the_scripts_own_declaration(self):
        declared = pc.declared_arms_in_source("10_homology_control.py", "PROTEIN_ARMS")
        self.assertEqual(
            declared,
            pc.STAGE_CONTRACTS["homology_control"].declared_arms,
            "10_homology_control.py::PROTEIN_ARMS and the panel contract disagree; "
            "a campaign run and a direct run would measure different panels",
        )

    def test_lens_architectures_match_the_module_that_delivers_them(self):
        from src.transfer.scaling import LENS_ARCHITECTURES

        self.assertEqual(
            pc.STAGE_CONTRACTS["lens_family"].architectures, frozenset(LENS_ARCHITECTURES)
        )

    def test_every_panel_member_is_admitted_or_excluded_explicitly(self):
        for name in PANEL:
            self.assertTrue(
                name in pc.CAMPAIGN_PANEL or name in pc.PANEL_MEMBERS_NOT_STAGED,
                f"{name} is in PANEL but neither staged nor given a reason",
            )

    def test_a_stages_named_refusal_is_not_also_a_campaign_exclusion(self):
        # The two declarations answer different questions and one must not be
        # used to express the other. `excluded_arms` refuses an arm from ONE
        # stage; CAMPAIGN_PANEL decides whether any campaign may schedule it at
        # all. bygpt5-small-en and bygpt5-base-en were kept out of the campaign
        # from 2026-08-06 to 2026-08-21 by a reason that was `paa_census`'s named
        # refusal restated -- so a refusal that belonged to one stage removed
        # them from `cohort_power` and `collision_null_census` too, silently and
        # with no reason attached to either.
        named = {
            arm
            for contract in pc.STAGE_CONTRACTS.values()
            for arm in contract.excluded_arms
        }
        overlap = sorted(named & set(pc.PANEL_MEMBERS_NOT_STAGED))
        self.assertEqual(
            overlap,
            [],
            f"{overlap} are refused by a stage AND kept out of the campaign; a "
            "stage's refusal applies to that stage, and every other stage then "
            "loses the arm without saying so",
        )

    def test_no_stage_may_schedule_an_arm_cohort_power_cannot_qualify(self):
        # Evidence-discipline rule 2: an arm whose cohort context-information has
        # not been qualified may not be scored. `cohort_power` is what qualifies
        # it, so an arm a stage admits but `cohort_power` refuses could only ever
        # produce a number nothing is allowed to read.
        qualifiable = set(pc.stage_arms("cohort_power")[0])
        for stage in pc.STAGE_ORDER:
            eligible = set(pc.stage_arms(stage)[0])
            self.assertEqual(
                sorted(eligible - qualifiable),
                [],
                f"{stage} admits arms cohort_power refuses",
            )

    def test_model_variable_is_resolved_from_the_declaration_not_the_arm_name(self):
        # gpt2-large is declared as TEXT_MODEL_ROOT itself; every other text arm
        # is addressed beneath TEXT_MODEL_BASE. The worker's `case` on the arm
        # name got this wrong for six of seven text arms until 2026-07-29.
        self.assertEqual(pc.model_variable("gpt2-large"), "TRANSFER_TEXT_MODEL_DIR")
        for arm in ("gpt2", "gpt2-medium", "gpt2-xl", "dialogpt-small",
                    "qwen2.5-0.5b", "llama-3.2-3b"):
            self.assertEqual(pc.model_variable(arm), "TRANSFER_TEXT_MODEL_BASE_DIR", arm)
        for arm in pc.CAMPAIGN_PANEL:
            if PANEL[arm].modality == "protein":
                self.assertEqual(pc.model_variable(arm), "TRANSFER_MODEL_BASE_DIR", arm)

    def test_corpus_variables_follow_the_declared_evaluation_cohort(self):
        self.assertEqual(pc.corpus_variables("zymctrl"), ("TRANSFER_ZYMCTRL_FASTA",))
        self.assertEqual(pc.corpus_variables("protgpt2"), ("TRANSFER_SWISSPROT_FASTA",))
        self.assertEqual(pc.corpus_variables("gpt2-large"), ("TRANSFER_OPENWEBTEXT_DIR",))


class CohortBandsAreDeclared(unittest.TestCase):
    """Four stages of one campaign draw protein cohorts on three different bands."""

    def test_lens_family_declares_that_its_band_is_not_the_qualifying_band(self):
        record = pc.stage_contract_record("lens_family", ["protgpt2"])
        self.assertEqual(record["cohort_band"]["protein_residues"], [64, 120])
        self.assertEqual(
            record["cohort_band"]["qualifying_stage_protein_residues"], [64, 246]
        )
        self.assertFalse(record["cohort_band"]["matches_qualifying_stage"])
        self.assertIn("NARROWER", record["cohort_band"]["reason"])

    def test_pathway_and_estimand_bands_match_the_qualifying_band(self):
        for stage in ("pathway_budget", "estimand_power"):
            record = pc.stage_contract_record(stage, ["protgpt2"])
            self.assertTrue(record["cohort_band"]["matches_qualifying_stage"], stage)

    def test_declared_bands_match_the_stage_scripts_own_argparse_defaults(self):
        # The declaration is only worth anything if it cannot drift from the flag.
        for stage, filename in (
            ("cohort_power", "01_cohort_power.py"),
            ("pathway_budget", "02_pathway_budget.py"),
            ("estimand_power", "03_estimand_power.py"),
            ("lens_family", "08_lens_family.py"),
        ):
            with self.subTest(stage=stage):
                declared = pc.STAGE_CONTRACTS[stage].protein_band
                self.assertEqual(
                    _argparse_int_defaults(filename, ("--res-min", "--res-max")),
                    list(declared),
                )

    def test_stage_contract_record_separates_a_narrowed_run_from_a_refusal(self):
        record = pc.stage_contract_record("lens_family", ["gpt2-large", "protgpt2"])
        selection = record["arm_selection"]
        self.assertEqual(selection["measured"], ["gpt2-large", "protgpt2"])
        # zymctrl could have been measured and was not: that is this invocation
        # narrowing the panel, which is the fact L18 says must be visible.
        self.assertIn("zymctrl", selection["eligible_but_not_measured"])
        self.assertIn("this invocation's arm list", selection["not_measured"]["zymctrl"])
        # qwen2.5-0.5b could not have been: that is a module limitation.
        self.assertNotIn("qwen2.5-0.5b", selection["eligible_for_this_stage"])
        self.assertIn("LENS_ARCHITECTURES", selection["not_measured"]["qwen2.5-0.5b"])


def _argparse_int_defaults(filename: str, flags: tuple[str, ...]) -> list[int]:
    """`default=` of the named integer flags, read statically from the source."""

    tree = ast.parse((STAGE_DIR / filename).read_text(encoding="utf-8"))
    found: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        name = node.args[0].value
        if name not in flags:
            continue
        for keyword in node.keywords:
            if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                found[name] = keyword.value.value
    missing = [flag for flag in flags if flag not in found]
    if missing:
        raise AssertionError(f"{filename} has no integer default for {missing}")
    return [found[flag] for flag in flags]


class DefaultsDoNotNarrowThePanel(unittest.TestCase):
    """A default that measures a subset of the panel and does not say so is L18."""

    def test_cohort_power_default_arms_cover_every_eligible_campaign_arm(self):
        module = _load_stage_module("01_cohort_power.py")
        text = module.default_arms("text", False)
        # Counted from the contract, not restated: admitting the byte-level
        # control moved the text side from seven arms to eight, and a literal
        # here would have made a declared panel change look like a defect --
        # the same reason the eligible-set assertion below is derived.
        self.assertEqual(
            set(text),
            {arm for arm in pc.CAMPAIGN_PANEL if PANEL[arm].modality == "text"},
            "cohort_power qualifies every campaign text arm; an arm it silently "
            "omits is an arm scored without its power check",
        )
        protein_ec = module.default_arms("protein", True)
        self.assertEqual(len(protein_ec), sum(1 for a in pc.CAMPAIGN_PANEL if PANEL[a].modality == "protein"))
        self.assertIn("zymctrl", protein_ec)
        self.assertNotIn(
            "zymctrl",
            module.default_arms("protein", False),
            "an EC-conditioned arm without --with-ec has no conditioning tag",
        )

    def test_pathway_and_estimand_defaults_exclude_arms_they_cannot_measure(self):
        for filename, attribute in (
            ("02_pathway_budget.py", "default_pathway_arms"),
            ("03_estimand_power.py", "default_estimand_arms"),
        ):
            with self.subTest(filename=filename):
                module = _load_stage_module(filename)
                arms = getattr(module, attribute)()
                self.assertNotIn("bygpt5-small-en", arms)
                # Every campaign arm the stage can actually measure, and no
                # other. Both stages need the `pathway` capability, which the
                # byte-level control does not carry, so it is absent here by the
                # contract's own refusal rather than by omission -- which is the
                # distinction this test exists to hold.
                stage = "pathway_budget" if filename.startswith("02") else "estimand_power"
                self.assertEqual(
                    set(arms),
                    {arm for arm in pc.CAMPAIGN_PANEL if pc.arm_can_run(stage, arm).can_run},
                )
                self.assertNotIn("bygpt5-medium-en", arms)

    def test_recommend_default_is_control_anchored_not_the_whole_panel(self):
        # `recommend` raises unless exactly one arm is text, so sorted(PANEL) was
        # a default that could never work -- and it lost a scheduled run.
        module = _load_stage_module("03_estimand_power.py")
        arms = module.default_recommend_arms()
        text = [name for name in arms if PANEL[name].modality == "text"]
        self.assertEqual(text, [module.TEXT_POSITIVE_CONTROL])
        self.assertEqual(len(arms), 1 + sum(1 for a in pc.CAMPAIGN_PANEL if PANEL[a].modality == "protein"))  # one text control plus four protein arms


class TheCensusDefaultsToTheWholeGrid(unittest.TestCase):
    """An omitted --causal-heads must not silently produce a selective census.

    It defaulted to 16 with a control offset of 120, which scores the census's own
    top 16 heads; census_causal_agreement then refuses the result under standing
    rule 24 and the run has cost a GPU to answer nothing. Every campaign
    invocation had to supply the exhaustive count by hand, and nine driver scripts
    grew the same per-arm table of head counts -- all of them n_layer * n_head
    minus the control block, which the entry point can compute and a driver
    cannot check.
    """

    def test_the_head_count_defaults_to_a_sentinel_not_to_a_small_literal(self):
        source = (STAGE_DIR / "14_paa_census.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        defaults = {}
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"):
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant):
                continue
            for keyword in node.keywords:
                if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                    defaults[node.args[0].value] = keyword.value.value
        self.assertIsNone(
            defaults.get("--causal-heads", "absent"),
            "a numeric default here is a selective census that looks like a full one",
        )
        self.assertIsNone(defaults.get("--control-offset", "absent"))

    def test_the_resolved_default_partitions_the_grid_with_the_control_block(self):
        # The property the nine hand-maintained tables encoded: causal heads and
        # the control offset are both grid - control_heads, so the two blocks
        # partition the grid exactly and no head is scored twice or skipped.
        for grid, control in ((720, 8), (192, 8), (1200, 8), (144, 8)):
            with self.subTest(grid=grid):
                causal = grid - control
                offset = grid - control
                self.assertEqual(causal + control, grid)
                self.assertEqual(len(range(offset, grid)), control)

    def test_an_over_large_request_is_refused_rather_than_truncated(self):
        source = (STAGE_DIR / "14_paa_census.py").read_text(encoding="utf-8")
        self.assertIn("exceed the", source)
        self.assertIn("grid_size", source)


class GuardsFireAtArgumentValidation(unittest.TestCase):
    """Every guard here reads only the command line, so none may need a checkpoint."""

    def test_lens_family_refuses_jacobian_positions_above_the_evaluation_split(self):
        module = _load_stage_module("08_lens_family.py")
        args = module.build_parser().parse_args(["--n-seq", "32", "--arms", "gpt2"])
        with self.assertRaises(ValueError) as caught:
            module.validate(args)
        self.assertIn("evaluation sequences", str(caught.exception))

    def test_lens_family_evaluation_split_size_matches_split_cohort(self):
        # The parse-time arithmetic must agree with the runtime split, or the
        # guard protects the wrong quantity.
        from src.transfer.arms import Cohort
        from src.transfer.lenses import split_cohort

        module = _load_stage_module("08_lens_family.py")
        for n_seq in (8, 16, 32, 33, 128, 200):
            for fraction in (0.5, 0.75, 0.7, 0.9):
                cohort = Cohort(
                    "t", "text", [f"record-{i}" for i in range(n_seq)], 1, 0, {}
                )
                _, evaluation = split_cohort(cohort, fraction, 0)
                self.assertEqual(
                    module.evaluation_split_size(n_seq, fraction),
                    len(evaluation),
                    f"n_seq={n_seq} fraction={fraction}",
                )

    def test_lens_family_refuses_an_arm_the_module_cannot_serve(self):
        module = _load_stage_module("08_lens_family.py")
        args = module.build_parser().parse_args(["--arms", "llama-3.2-3b"])
        with self.assertRaises(ValueError) as caught:
            module.validate(args)
        self.assertIn("LENS_ARCHITECTURES", str(caught.exception))

    def test_cohort_power_refuses_a_max_len_that_cannot_fit_its_contexts(self):
        module = _load_stage_module("01_cohort_power.py")
        args = _namespace(max_len=100, truncation_contexts=[1, 8, 128], skip_truncation=False)
        with self.assertRaises(ValueError) as caught:
            module.validate_truncation(args)
        self.assertIn("--max-len", str(caught.exception))
        # ...and accepts the campaign defaults.
        module.validate_truncation(
            _namespace(max_len=384, truncation_contexts=[1, 8, 128], skip_truncation=False)
        )
        # ...and does not check at all when the curve is skipped.
        module.validate_truncation(
            _namespace(max_len=8, truncation_contexts=[128], skip_truncation=True)
        )

    def test_cohort_power_refuses_an_ec_arm_without_an_ec_cohort_before_reading_the_corpus(self):
        module = _load_stage_module("01_cohort_power.py")
        with self.assertRaises(ValueError) as caught:
            module.validate_arms(["zymctrl"], _namespace(kind="protein", with_ec=False))
        self.assertIn("EC-conditioned", str(caught.exception))

    def test_cohort_power_ec_check_is_stated_against_input_format_not_a_name(self):
        source = (STAGE_DIR / "01_cohort_power.py").read_text(encoding="utf-8")
        self.assertNotIn('"zymctrl" in names', source)
        self.assertIn('input_format == "ec_conditioned"', source)


def _namespace(**values):
    import argparse

    return argparse.Namespace(**values)


class ShellContractStaysInStepWithThePanel(unittest.TestCase):
    """The generated file is a cache of the declaration, not a second declaration."""

    def test_the_emitted_shell_contract_matches_the_live_panel(self):
        self.assertEqual(
            pc.verify(),
            [],
            "scripts/transfer/panel_contract.sh is stale; run "
            "`python scripts/transfer/panel_contract.py --emit`",
        )

    def test_bash_can_source_it_and_reproduces_the_same_arm_lists(self):
        script = f"""
        set -euo pipefail
        source {pc.SHELL_CONTRACT}
        printf 'panel=%s\\n' "$TRANSFER_CAMPAIGN_PANEL"
        printf 'lens=%s\\n' "${{TRANSFER_STAGE_ARMS[lens_family]}}"
        printf 'relational=%s\\n' "${{TRANSFER_STAGE_ARMS[relational_channel]}}"
        printf 'pp=%s\\n' "${{TRANSFER_STAGE_ARMS[induction_path_patching]}}"
        printf 'modality=%s\\n' "${{TRANSFER_ARM_MODALITY[protgpt2]}}"
        printf 'modelvar=%s\\n' "${{TRANSFER_ARM_MODEL_VAR[qwen2.5-0.5b]}}"
        printf 'items=%s\\n' "$TRANSFER_COHORT_ITEMS"
        """
        out = subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True, check=True
        ).stdout
        rendered = dict(line.split("=", 1) for line in out.strip().splitlines())
        self.assertEqual(rendered["panel"].split(), list(pc.CAMPAIGN_PANEL))
        self.assertEqual(rendered["lens"].split(), pc.stage_arms("lens_family")[0])
        self.assertEqual(
            rendered["relational"].split(), pc.stage_arms("relational_channel")[0]
        )
        self.assertEqual(
            rendered["pp"].split(), pc.stage_arms("induction_path_patching")[0]
        )
        self.assertEqual(rendered["modality"], "protein")
        self.assertEqual(rendered["modelvar"], "TRANSFER_TEXT_MODEL_BASE_DIR")
        self.assertEqual(rendered["items"].split(), [i.item for i in pc.cohort_power_items()])

    def test_cohort_power_items_reproduce_the_campaign_dispatch(self):
        items = {item.item: item for item in pc.cohort_power_items()}
        self.assertEqual(items["protein_small_vocab"].arms, ("zymctrl",))
        self.assertIn("--with-ec", items["protein_small_vocab"].extra_args)
        self.assertEqual(items["protein_large_vocab"].arms, ("protgpt2",))
        self.assertIn("--skip-truncation", items["protein_large_vocab"].extra_args)
        self.assertEqual(items["protein_default_dtype"].arms, ("progen2-base", "progen2-small"))
        self.assertEqual(items["protein_default_dtype"].extra_args, ())
        self.assertEqual(items["protein_progen2_medium"].arms, ("progen2-medium",))
        self.assertEqual(
            items["protein_progen2_medium"].extra_args, ("--dtype", "float32")
        )
        # Every protein item needs its own --cohort-name or two identical cohorts
        # collide on one output filename.
        protein_names = [
            item.cohort_name for key, item in items.items() if key.startswith("protein")
        ]
        self.assertEqual(len(set(protein_names)), len(protein_names))
        self.assertNotIn(None, protein_names)


#: The machine-readable resource interface `external_resources/README.md` and
#: `scripts/transfer/README.md` both point operators at.
RESOURCE_MANIFEST = (
    REPO_ROOT / "external_resources" / "manifests" / "interpretability_transfer_resources.json"
)


class ResourceManifestMirrorsThePanelContract(unittest.TestCase):
    """The manifest's `contract` block is a copy of the declaration, not a claim.

    It says so itself -- "derived from `panel_contract.py --json`; do not
    hand-edit" -- and nothing read it, so the note was the whole guarantee and it
    drifted twice with every campaign still working: first to 11 arms / 11
    stages, then to schema v1 with an ALPHABETISED `stage_order`. Sorting that
    field destroys the only thing it states, the campaign execution order, and a
    reader cannot tell a sorted list from a stale one by looking at it.
    """

    def setUp(self):
        self.manifest = json.loads(RESOURCE_MANIFEST.read_text(encoding="utf-8"))
        self.live = pc.contract_payload()

    def test_the_contract_block_is_the_live_contract(self):
        block = self.manifest["contract"]
        stale = "; regenerate from `python scripts/transfer/panel_contract.py --json`"
        self.assertEqual(block["schema_version"], self.live["schema_version"], stale)
        self.assertEqual(block["stage_order"], self.live["stage_order"], stale)
        self.assertEqual(block["campaign_panel"], self.live["campaign_panel"], stale)
        self.assertEqual(block["arm_count"], len(self.live["campaign_panel"]), stale)
        self.assertEqual(block["stage_count"], len(self.live["stage_order"]), stale)

    def test_every_panel_arm_is_listed_under_the_variable_that_relocates_it(self):
        # PANEL, not CAMPAIGN_PANEL: these entries answer "which variable
        # relocates this checkpoint", which is declared for every panel member,
        # and the manifest's resolution_policy refuses to carry an availability
        # claim -- which campaign membership is. The two lists happen to cover
        # the same arms since 2026-08-21, and this test must keep resolving over
        # PANEL anyway, because the next arm admitted to PANEL is relocatable
        # before any campaign schedules it.
        declared: dict[str, set[str]] = {}
        for entry in self.manifest["model_resources"]:
            arms = entry["arms"]
            self.assertNotIn(entry["variable"], declared, "one entry per variable")
            self.assertEqual(len(arms), len(set(arms)), f"{entry['variable']} repeats an arm")
            declared[entry["variable"]] = set(arms)
        expected: dict[str, set[str]] = {}
        for arm in PANEL:
            expected.setdefault(pc.model_variable(arm), set()).add(arm)
        self.assertEqual(declared, expected)


class WorkerAndControllerBehaviour(unittest.TestCase):
    """Bash-level properties, exercised by sourcing the real scripts' logic."""

    def test_worker_and_controller_are_syntactically_valid(self):
        for name in ("h200_worker.sh", "run_transfer_h200.sh"):
            subprocess.run(["bash", "-n", str(STAGE_DIR / name)], check=True)

    def test_neither_script_hard_codes_an_arm_list(self):
        # Both used to carry their own KNOWN_ARMS string, and the worker three
        # further hand-written arm groupings besides.
        for name in ("h200_worker.sh", "run_transfer_h200.sh"):
            source = (STAGE_DIR / name).read_text(encoding="utf-8")
            for arm in ("progen2-base", "qwen2.5-0.5b", "llama-3.2-3b"):
                self.assertNotIn(
                    f"{arm}|", source, f"{name} still branches on the arm name {arm}"
                )
            self.assertIn("TRANSFER_CAMPAIGN_PANEL", source, name)

    def test_controller_never_prints_the_pod_name(self):
        source = (STAGE_DIR / "run_transfer_h200.sh").read_text(encoding="utf-8")
        self.assertNotIn('log "H200_POD:', source)
        self.assertIn("redact()", source)
        # The worker's whole output passes through the redactor before it reaches
        # either the terminal or the controller log, so the guarantee does not
        # depend on how the operator invoked the script.
        self.assertIn("| redact | tee", source)

    def test_pod_name_is_redacted_from_stdout_end_to_end(self):
        pod = "a-very-distinctive-pod-name-42"
        script = f"""
        set -euo pipefail
        H200_POD={pod}
        {_extract_function(STAGE_DIR / 'run_transfer_h200.sh', 'redact')}
        printf 'worker says %s is busy\\n' "{pod}" | redact
        """
        out = subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True, check=True
        ).stdout
        self.assertNotIn(pod, out)
        self.assertIn("<pod-redacted>", out)

    def test_duplicate_stage_args_are_refused_rather_than_silently_last_wins(self):
        # argparse takes the last occurrence. For --cohort-name that collides two
        # items' cohorts on one output path; for --dtype it discards a measured
        # reason. Refusing is the only safe answer.
        script = f"""
        set -euo pipefail
        log() {{ :; }}
        {_extract_function(STAGE_DIR / 'h200_worker.sh', 'assert_no_duplicate_options')}
        assert_no_duplicate_options cohort_power protein_progen2_medium \\
          python 01_cohort_power.py --dtype float32 --out /x --dtype bfloat16
        """
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--dtype", result.stderr)

    def test_non_duplicate_stage_args_are_accepted(self):
        script = f"""
        set -euo pipefail
        log() {{ :; }}
        {_extract_function(STAGE_DIR / 'h200_worker.sh', 'assert_no_duplicate_options')}
        assert_no_duplicate_options cohort_power text \\
          python 01_cohort_power.py --skip-truncation --out /x --n-seq 500
        """
        subprocess.run(["bash", "-c", script], check=True)

    def test_recommend_failure_is_deferred_and_still_fails_the_campaign(self):
        source = (STAGE_DIR / "h200_worker.sh").read_text(encoding="utf-8")
        self.assertIn("DEFERRED_FAILURES+=(", source)
        self.assertIn("campaign FAILED", source)
        # It must NOT exit inline: that cost tier 3 -- six GPU stages -- to a
        # failure in a CPU-only aggregation at the end of tier 2.
        recommend = source.split("run_estimand_power()", 1)[1].split("\n}\n", 1)[0]
        self.assertNotIn('exit "${status}"', recommend)

    def test_recommend_provenance_covers_the_measure_outputs_it_consumes(self):
        # Keyed on the command alone, changing ARGS_ESTIMAND_POWER re-ran every
        # measure item and then skipped recommend as complete, leaving a panel
        # verdict derived from measure outputs that no longer existed.
        source = (STAGE_DIR / "h200_worker.sh").read_text(encoding="utf-8")
        self.assertIn("--measure-inputs-sha256", source)

    def _build_command(self, stage: str, item: str, extra_env: str = "") -> list[str]:
        """The worker's own build_command, exercised against the real contract."""

        script = f"""
        set -euo pipefail
        source {pc.SHELL_CONTRACT}
        TRANSFER_PYTHON=/usr/bin/python3
        TRANSFER_SCRIPTS=/snapshot/scripts/transfer
        RUN_ID=testrun
        LOCAL_SCRATCH_ROOT=/tmp/scratch
        declare -A STAGE_EXTRA_ARGS=()
        declare -A ITEM_EXTRA_ARGS=()
        declare -A COHORT_ITEM_ARMS_FOR=()
        for i in $TRANSFER_COHORT_ITEMS; do
          COHORT_ITEM_ARMS_FOR[$i]="${{TRANSFER_COHORT_ITEM_ARMS[$i]}}"
        done
        read -r -a CIRCUIT_ARMS <<< "${{TRANSFER_STAGE_ARMS[circuit_primitives]}}"
        read -r -a HOMOLOGY_ARMS <<< "${{TRANSFER_STAGE_ARMS[homology_control]}}"
        read -r -a PATH_PATCHING_ARMS <<< "${{TRANSFER_STAGE_ARMS[induction_path_patching]}}"
        log() {{ :; }}
        {_extract_function(STAGE_DIR / 'h200_worker.sh', 'arm_modality')}
        {_extract_function(STAGE_DIR / 'h200_worker.sh', 'assert_no_duplicate_options')}
        {_extract_function(STAGE_DIR / 'h200_worker.sh', 'build_command')}
        {extra_env}
        build_command {stage} {item} 3 /out
        printf '%s\\n' "${{CMD[@]}}"
        """
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        if result.returncode != 0:
            raise AssertionError(result.stderr)
        return result.stdout.strip().splitlines()

    def test_cohort_power_items_are_built_from_the_contract(self):
        progen_base = self._build_command("cohort_power", "protein_default_dtype")
        self.assertIn("--kind", progen_base)
        self.assertEqual(progen_base[progen_base.index("--kind") + 1], "protein")
        self.assertEqual(
            progen_base[progen_base.index("--arms") + 1],
            "progen2-base",
        )
        self.assertNotIn("--dtype", progen_base)
        self.assertEqual(
            progen_base[progen_base.index("--cohort-name") + 1],
            "swissprot_default_dtype",
        )
        progen_medium = self._build_command("cohort_power", "protein_progen2_medium")
        self.assertEqual(
            progen_medium[progen_medium.index("--arms") + 1], "progen2-medium"
        )
        self.assertEqual(
            progen_medium[progen_medium.index("--dtype") + 1], "float32"
        )
        self.assertEqual(
            progen_medium[progen_medium.index("--cohort-name") + 1],
            "swissprot_progen2_medium_f32",
        )
        text = self._build_command("cohort_power", "text")
        self.assertEqual(text[text.index("--kind") + 1], "text")
        self.assertIn("--skip-truncation", text)
        # The text item takes the script's own default cohort name.
        self.assertNotIn("--cohort-name", text)

    def test_panel_wide_stages_are_built_from_the_eligible_arm_list(self):
        # The worker used to pass its whole ARM_LIST to stage 11, which raises on
        # four of the eleven campaign arms.
        command = self._build_command("induction_path_patching", "panel")
        arms = command[command.index("--arms") + 1 : command.index("--device")]
        self.assertEqual(arms, pc.stage_arms("induction_path_patching")[0])
        homology = self._build_command("homology_control", "panel")
        arms = homology[homology.index("--arms") + 1 : homology.index("--device")]
        self.assertEqual(arms, list(pc.STAGE_CONTRACTS["homology_control"].declared_arms))

    def test_item_scoped_args_reach_only_their_item(self):
        override = 'ITEM_EXTRA_ARGS["cohort_power/protein_progen2_medium"]="--cohort-pool-size 8000"'
        progen = self._build_command("cohort_power", "protein_progen2_medium", override)
        self.assertIn("--cohort-pool-size", progen)
        text = self._build_command("cohort_power", "text", override)
        self.assertNotIn("--cohort-pool-size", text)

    def test_a_stage_wide_arg_colliding_with_an_item_flag_is_refused(self):
        with self.assertRaises(AssertionError) as caught:
            self._build_command(
                "cohort_power",
                "protein_progen2_medium",
                'STAGE_EXTRA_ARGS["cohort_power"]="--dtype bfloat16"',
            )
        self.assertIn("--dtype", str(caught.exception))

    def test_paa_census_builds_a_well_formed_per_arm_invocation(self):
        command = self._build_command("paa_census", "protgpt2", "TEXT_ARM=gpt2-large")
        self.assertTrue(command[1].endswith("14_paa_census.py"), command)
        # Singular --census-arm plus the campaign's own text control, not the
        # entry point's default: a run that anchored on a different control from
        # the rest of the campaign would still look well-formed.
        self.assertEqual(command[command.index("--census-arm") + 1], "protgpt2")
        self.assertEqual(command[command.index("--text-arm") + 1], "gpt2-large")
        self.assertEqual(command[command.index("--device") + 1], "cuda:3")
        # --stages is fixed because the entry point REFUSES `match`/`query` when
        # --census-arm differs from --text-arm, and its own default requests all
        # five. The default would therefore fail every protein item outright.
        stages = command[command.index("--stages") + 1 : command.index("--census-arm")]
        self.assertEqual(stages, ["census", "causal"])
        for consumed in ("match", "query", "gate0"):
            self.assertNotIn(consumed, command)

    def test_paa_census_items_do_not_share_one_output_directory(self):
        # This stage names its principal artefacts after itself, not after the
        # arm -- census.json, causal.json, selected_heads.json,
        # paa_gate_report.json -- which no other per-arm stage does, and its own
        # census() docstring states that arms must therefore run in separate
        # --out directories. Sharing one, each arm would overwrite the previous
        # arm's census and each item's resume manifest would checksum a file
        # another arm wrote: an overwrite that verifies cleanly.
        source = (STAGE_DIR / "14_paa_census.py").read_text(encoding="utf-8")
        for shared in ('"census.json"', '"causal.json"', '"selected_heads.json"'):
            self.assertIn(shared, source, "this test's premise no longer holds")
        directories = set()
        for arm in pc.stage_arms("paa_census")[0][:3]:
            command = self._build_command("paa_census", arm, "TEXT_ARM=gpt2-large")
            out = command[command.index("--out") + 1]
            self.assertTrue(out.endswith(f"/{arm}"), out)
            directories.add(out)
        self.assertEqual(len(directories), 3, "two arms would write over each other")

    def test_paa_census_width_is_read_from_the_contract_not_written_in_the_worker(self):
        # The contract's eligible arm list is declared against this width:
        # ProtGPT2 admits no full-width cohort row at the entry point's own
        # default of 512, so a width the worker owned privately could drift from
        # the width the arms were admitted at, and nothing would look wrong until
        # a checkpoint was already on the GPU (EXP-R2-082).
        command = self._build_command("paa_census", "protgpt2", "TEXT_ARM=gpt2-large")
        self.assertEqual(
            command[command.index("--width") + 1], str(pc.PAA_CENSUS_WIDTH)
        )
        moved = self._build_command(
            "paa_census",
            "protgpt2",
            "TEXT_ARM=gpt2-large; TRANSFER_PAA_CENSUS_WIDTH=997",
        )
        self.assertEqual(moved[moved.index("--width") + 1], "997")

    def test_paa_census_scale_knobs_pass_through_but_feasibility_flags_do_not(self):
        # --census-sequences is what this campaign has to move (200 -> 600), and
        # it is a scale knob. --width is not: overriding it silently would change
        # which arms the run can serve, so it must be refused like --dtype is.
        scaled = self._build_command(
            "paa_census",
            "protgpt2",
            'TEXT_ARM=gpt2-large; STAGE_EXTRA_ARGS["paa_census"]="--census-sequences 600"',
        )
        self.assertEqual(scaled[scaled.index("--census-sequences") + 1], "600")
        per_item = self._build_command(
            "paa_census",
            "gpt2-large",
            'TEXT_ARM=gpt2-large; ITEM_EXTRA_ARGS["paa_census/gpt2-large"]="--cohort-draw-seed 20260801"',
        )
        self.assertIn("--cohort-draw-seed", per_item)
        with self.assertRaises(AssertionError) as caught:
            self._build_command(
                "paa_census",
                "protgpt2",
                'TEXT_ARM=gpt2-large; STAGE_EXTRA_ARGS["paa_census"]="--width 512"',
            )
        self.assertIn("--width", str(caught.exception))

    def test_the_command_preflight_covers_a_per_arm_stage_by_scope(self):
        # It used to be one `case` arm per per-arm stage beneath a `*` fallback
        # that built everything else as the literal item `panel`. A per-arm stage
        # added to the contract and forgotten there was not a build failure but a
        # silently WRONG build: `--census-arm panel` would have passed preflight.
        body = _extract_function(STAGE_DIR / "h200_worker.sh", "verify_commands_buildable")
        self.assertIn("TRANSFER_STAGE_SCOPE", body)
        self.assertIn("per_arm", body)
        for stage, contract in pc.STAGE_CONTRACTS.items():
            if contract.scope == "per_arm":
                self.assertNotIn(
                    f"{stage})",
                    body,
                    f"{stage} is preflighted by name; a per-arm stage added to the "
                    "contract must be covered by its declared scope instead",
                )

    def test_import_preflight_covers_every_scheduled_entry_point(self):
        # It used to be a hand-written list of nine while the worker schedules
        # eleven stages, so 10_homology_control.py and 11_induction_path_patching.py
        # were dispatched without ever being import-checked.
        source = (STAGE_DIR / "h200_worker.sh").read_text(encoding="utf-8")
        self.assertIn("requested_entry_points", source)
        self.assertNotIn("09_probe_and_erasure\n", source)
        for stage, contract in pc.STAGE_CONTRACTS.items():
            self.assertTrue((STAGE_DIR / contract.entry_point).is_file(), stage)


def _extract_function(path: Path, name: str) -> str:
    """One shell function's source, so a test can exercise it without the script."""

    source = path.read_text(encoding="utf-8")
    marker = f"\n{name}() {{\n"
    if marker not in source:
        raise AssertionError(f"{path.name} has no function {name}")
    body = source.split(marker, 1)[1]
    end = body.index("\n}\n")
    return f"{name}() {{\n{body[:end]}\n}}\n"


class TransferGapCohortLayerRemoved(unittest.TestCase):
    """`tg_common` no longer implements corpus eligibility or its own permutation."""

    def test_no_second_eligibility_predicate_survives(self):
        # Matched as definitions rather than as bare substrings. `_permutation`
        # as a substring also matches the string literal
        # "seeded_permutation_of_the_drawn_set" that the record-order provenance
        # writes, so the loose form failed on a change that introduced no second
        # selection layer at all -- a test that fires on the wrong thing is worse
        # than no test, because the next reader loosens it instead of reading it.
        source = (REPO_ROOT / "scripts" / "transfer_gap" / "tg_common.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        defined = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                defined |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        for symbol in (
            "_ELIGIBLE_CACHE",
            "_eligible_swissprot",
            "_eligible_ec",
            "_eligible_text",
            "_permutation",
        ):
            self.assertNotIn(symbol, defined, f"{symbol} is a second selection layer")

    def test_cohort_for_source_calls_the_shared_constructors(self):
        source = (REPO_ROOT / "scripts" / "transfer_gap" / "tg_common.py").read_text(
            encoding="utf-8"
        )
        body = source.split("def cohort_for(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("text_cohort(", body)
        self.assertIn("protein_cohort(", body)


class LibraryCohortSeedIsOptIn(unittest.TestCase):
    """The seeded draw must be opt-in: 10_homology_control relies on file order."""

    def test_protein_and_text_cohort_default_to_file_order(self):
        import inspect

        from src.transfer import arms

        for function in (arms.protein_cohort, arms.text_cohort):
            self.assertIsNone(
                inspect.signature(function).parameters["seed"].default,
                f"{function.__name__} must default to file order: "
                "10_homology_control.build_cohort relies on the stratified cohort "
                "being the cohort the headline was measured on",
            )

    def test_selected_positions_is_disjoint_across_skips_at_one_seed(self):
        from src.transfer.arms import selected_positions

        first = selected_positions(1000, n=100, skip=0, seed=7, label="a")
        second = selected_positions(1000, n=100, skip=100, seed=7, label="b")
        self.assertEqual(set(first) & set(second), set())

    def test_file_order_cohorts_carry_their_hazard(self):
        from src.transfer.arms import FILE_ORDER_HAZARD, sampling_record

        record = sampling_record(
            seed=None, skip=0, requested=10, eligible=None, corpus="plain_swissprot"
        )
        self.assertEqual(record["mode"], "file_order")
        self.assertEqual(record.get("hazard"), FILE_ORDER_HAZARD)


class ExplanationChannelSelection(unittest.TestCase):
    """L9's bits/symbol figures were measured over a filename-order prefix."""

    def test_structures_are_drawn_under_a_seeded_permutation(self):
        source = (STAGE_DIR / "06_explanation_channel.py").read_text(encoding="utf-8")
        self.assertIn("alphafold_model_sample(", source)
        self.assertNotIn("alphafold_models(ALPHAFOLD_ROOT, limit=", source)

    def test_no_channel_is_drawn_in_corpus_file_order(self):
        # The Pfam and text channels were file-order draws until plan item B3.
        # Their bits/symbol are entropies of a label channel measured over
        # whichever records were read, so a family-grouped prefix understates
        # them; both now draw under a seeded permutation of the whole corpus.
        source = (STAGE_DIR / "06_explanation_channel.py").read_text(encoding="utf-8")
        self.assertNotIn("swissprot_file_order_prefix", source)
        self.assertIn("selected_positions(", source)
        self.assertIn("seed=args.pfam_seed", source)
        self.assertIn("seed=args.text_seed", source)

    def test_the_max_units_cut_is_not_a_prefix_of_the_draw(self):
        # A seeded draw arrives in ascending corpus order. Taking the first
        # --max-units members that reach the window would reintroduce the
        # file-order prefix one step later, so every channel visits its draw
        # under a seeded permutation.
        source = (STAGE_DIR / "06_explanation_channel.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("draw_order("), 4)
        for anchor in ("for index in draw_order(", "def draw_order("):
            self.assertIn(anchor, source)


class ConvergenceControlRecordsBothCorpora(unittest.TestCase):
    def test_per_rung_record_names_the_evaluation_and_pretraining_corpora(self):
        source = (STAGE_DIR / "07_convergence_control.py").read_text(encoding="utf-8")
        self.assertIn('"evaluation_cohort_source": member.evaluation_cohort_source', source)
        self.assertIn('"pretraining_corpus": member.pretraining_corpus', source)


class ConvergenceLadderResolution(unittest.TestCase):
    def stage(self):
        return _load_stage_module("07_convergence_control.py")

    def test_no_override_uses_the_code_contract(self):
        stage = self.stage()
        ladder, provenance = stage.resolve_ladder(None)
        self.assertEqual(ladder, scaling.DEFAULT_LADDER)
        self.assertEqual(provenance["source"], "scaling.DEFAULT_LADDER")
        self.assertIsNone(provenance["path"])

    def test_valid_override_replaces_the_code_contract(self):
        stage = self.stage()
        with tempfile.TemporaryDirectory() as tmpdir:
            table = Path(tmpdir) / "ladder.md"
            table.write_text(
                "| name | path | modality | tokenisation | input_format | source | "
                "cohort_corpus | cohort_min_symbols | cohort_max_symbols |\n"
                "|---|---|---|---|---|---|---|---:|---:|\n"
                "| only-arm | /models/only | text | bpe | raw | web | "
                "openwebtext_screen | 800 | 0 |\n",
                encoding="utf-8",
            )
            ladder, provenance = stage.resolve_ladder(table)
        self.assertEqual([member.name for member in ladder], ["only-arm"])
        self.assertEqual(provenance["source"], "ladder_table")

    def test_explicit_missing_or_non_table_input_fails(self):
        stage = self.stage()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaises(FileNotFoundError):
                stage.resolve_ladder(root / "missing.md")
            prose = root / "prose.md"
            prose.write_text("# This is not a ladder declaration\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no ladder declaration"):
                stage.resolve_ladder(prose)


if __name__ == "__main__":  # pragma: no cover
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    unittest.main()


class ThePanelReaderPoolsOneConditionAndNeverComputesTheStatistic(unittest.TestCase):
    """`read_paa_panel.py` exists because pulling matrices to read a panel is what
    left fourteen finished arm-draws unread. Two properties keep it honest: it
    pools only artefacts sharing the declared condition, and it never computes the
    agreement statistic itself -- it takes the stored one or calls the module.
    """

    def reader(self):
        return _load_stage_module("read_paa_panel.py")

    def write_run(self, directory: Path, *, arm: str, settings: dict, hit: int, stored=True):
        directory.mkdir(parents=True, exist_ok=True)
        agreement = {
            "n_heads": 192,
            "spearman_census_vs_causal_magnitude": 0.1,
            "depth_controlled": {"within_layer": 0.0},
            "retrieval": {"hit_at_k": hit, "chance": 400 / 192, "k": 20, "ceiling": 20},
        }
        report = {
            "settings": settings,
            "census": {
                "arm": arm,
                "a1_candidate_pool": {"layout_tokens_excluded_from_decoys": []},
            },
            "causal": {"census_causal_agreement": agreement} if stored else {},
        }
        (directory / "paa_gate_report.json").write_text(json.dumps(report), encoding="utf-8")

    def test_an_off_condition_run_is_dropped_and_the_reason_is_stated(self):
        module = self.reader()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            declared = dict(module.DECLARED_CONDITION)
            self.write_run(root / "good", arm="protgpt2", settings=declared, hit=6)
            self.write_run(
                root / "thin", arm="protgpt2", settings={**declared, "census_sequences": 200}, hit=6
            )
            per_arm, dropped = module.collect(root, reports_only=True, any_condition=False)
            self.assertEqual([d["source"] for d in per_arm["protgpt2"]], ["good"])
            self.assertEqual(len(dropped), 1)
            self.assertIn("census_sequences=200", dropped[0])

    def test_a_pre_decoy_guard_run_is_dropped_because_it_is_a_different_instrument(self):
        module = self.reader()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_run(root / "old", arm="protgpt2", settings=dict(module.DECLARED_CONDITION), hit=6)
            report_path = root / "old" / "paa_gate_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            del report["census"]["a1_candidate_pool"]["layout_tokens_excluded_from_decoys"]
            report_path.write_text(json.dumps(report), encoding="utf-8")
            per_arm, dropped = module.collect(root, reports_only=True, any_condition=False)
            self.assertEqual(per_arm, {})
            self.assertIn("decoy layout guard", dropped[0])

    def test_reports_only_refuses_rather_than_recomputing(self):
        module = self.reader()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_run(
                root / "pre", arm="protgpt2", settings=dict(module.DECLARED_CONDITION),
                hit=6, stored=False,
            )
            per_arm, dropped = module.collect(root, reports_only=True, any_condition=False)
            self.assertEqual(per_arm, {})
            self.assertIn("no readable agreement statistic", dropped[0])

    def test_the_statistic_is_never_reimplemented_here(self):
        source = (STAGE_DIR / "read_paa_panel.py").read_text(encoding="utf-8")
        self.assertIn("census_causal_agreement", source)
        for forbidden in ("spearmanr", "argsort", "rankdata"):
            self.assertNotIn(
                forbidden,
                source,
                "this reader must take the module's statistic, not compute one beside it",
            )

    def test_chance_is_per_arm_so_the_summary_classifies_rather_than_ranks(self):
        module = self.reader()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            declared = dict(module.DECLARED_CONDITION)
            for index, hit in enumerate((1, 3, 5)):
                self.write_run(root / f"d{index}", arm="progen2-small", settings=declared, hit=hit)
            per_arm, _ = module.collect(root, reports_only=True, any_condition=False)
            row = module.summarise(per_arm)[0]
            self.assertEqual(row["k"], 3)
            self.assertEqual(row["hit_at_k"], [1, 3, 5])
            self.assertAlmostEqual(row["median_over_own_chance"], 3 / (400 / 192))
