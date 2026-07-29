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
import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
STAGE_DIR = REPO_ROOT / "scripts" / "transfer"
if str(STAGE_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE_DIR))

import panel_contract as pc  # noqa: E402
from src.transfer.arms import PANEL  # noqa: E402


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

    def test_path_patching_refuses_rotary_arms_with_the_module_declaration_named(self):
        # path_patching.require_supported_layout raises for these, after the
        # checkpoint is on the GPU. The worker used to hand it the whole arm list.
        for arm in ("qwen2.5-0.5b", "llama-3.2-3b"):
            verdict = pc.arm_can_run("induction_path_patching", arm)
            self.assertFalse(verdict.can_run, arm)
            self.assertIn("SUPPORTED_ARCHITECTURES", verdict.reason)

    def test_path_patching_admits_declared_progen_layouts(self):
        for arm in ("progen2-base", "progen2-medium"):
            verdict = pc.arm_can_run("induction_path_patching", arm)
            self.assertTrue(verdict.can_run, verdict.reason)
            self.assertEqual(PANEL[arm].architecture, "progen")

    def test_lens_family_refuses_rmsnorm_arms_and_admits_every_other_campaign_arm(self):
        eligible, refused = pc.stage_arms("lens_family")
        self.assertEqual({v.arm for v in refused}, {"qwen2.5-0.5b", "llama-3.2-3b"})
        self.assertEqual(len(eligible), 9)  # the nine arms EXP-R2-060 scored

    def test_relational_channel_includes_progen2_base(self):
        # The worker's hand-written list named zymctrl and progen2-medium only.
        # progen2-base is protein, residue-tokenised and carries `relational`, so
        # excluding it narrowed the stage's panel by one arm with nothing saying so.
        eligible, _ = pc.stage_arms("relational_channel")
        self.assertEqual(eligible, ["zymctrl", "progen2-base", "progen2-medium"])

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

    def test_model_variable_is_resolved_from_the_declaration_not_the_arm_name(self):
        # gpt2-large is declared as TEXT_MODEL_ROOT itself; every other text arm
        # is addressed beneath TEXT_MODEL_BASE. The worker's `case` on the arm
        # name got this wrong for six of seven text arms until 2026-07-29.
        self.assertEqual(pc.model_variable("gpt2-large"), "TRANSFER_TEXT_MODEL_DIR")
        for arm in ("gpt2", "gpt2-medium", "gpt2-xl", "dialogpt-small",
                    "qwen2.5-0.5b", "llama-3.2-3b"):
            self.assertEqual(pc.model_variable(arm), "TRANSFER_TEXT_MODEL_BASE_DIR", arm)
        for arm in ("protgpt2", "zymctrl", "progen2-base", "progen2-medium"):
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
        self.assertEqual(len(text), 7, "the campaign's text side is seven arms")
        protein_ec = module.default_arms("protein", True)
        self.assertEqual(len(protein_ec), 4)
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
                self.assertEqual(set(arms), set(pc.CAMPAIGN_PANEL))

    def test_recommend_default_is_control_anchored_not_the_whole_panel(self):
        # `recommend` raises unless exactly one arm is text, so sorted(PANEL) was
        # a default that could never work -- and it lost a scheduled run.
        module = _load_stage_module("03_estimand_power.py")
        arms = module.default_recommend_arms()
        text = [name for name in arms if PANEL[name].modality == "text"]
        self.assertEqual(text, [module.TEXT_POSITIVE_CONTROL])
        self.assertEqual(len(arms), 5)  # one text control plus four protein arms


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
        self.assertEqual(items["protein_progen2_base"].arms, ("progen2-base",))
        self.assertEqual(items["protein_progen2_base"].extra_args, ())
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
        progen_base = self._build_command("cohort_power", "protein_progen2_base")
        self.assertIn("--kind", progen_base)
        self.assertEqual(progen_base[progen_base.index("--kind") + 1], "protein")
        self.assertEqual(
            progen_base[progen_base.index("--arms") + 1],
            "progen2-base",
        )
        self.assertNotIn("--dtype", progen_base)
        self.assertEqual(
            progen_base[progen_base.index("--cohort-name") + 1],
            "swissprot_progen2_base",
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


if __name__ == "__main__":  # pragma: no cover
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    unittest.main()
