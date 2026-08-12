from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSFER_DIR = REPO_ROOT / "scripts" / "transfer"
CONTROLLER = TRANSFER_DIR / "run_transfer_h200.sh"
WORKER = TRANSFER_DIR / "h200_worker.sh"
README = TRANSFER_DIR / "README.md"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TRANSFER_DIR) not in sys.path:
    sys.path.insert(0, str(TRANSFER_DIR))

import panel_contract as pc  # noqa: E402


#: A stand-in for h200_pod_bash.sh that evaluates its argument on this host.
#:
#: The controller's remote predicates answer on stdout rather than by exit status,
#: because the real access layer returns 0 whatever the remote command exits with
#: (L20). A stub that merely succeeds therefore exercises none of them: it returns
#: an empty reply and pod_predicate correctly refuses it. Evaluating the command
#: locally is what makes these tests test the predicate rather than the stub.
_STUB_DIR = Path(tempfile.mkdtemp(prefix="h200_stub_"))
LOCAL_POD_BASH = _STUB_DIR / "pod_bash.sh"
LOCAL_POD_BASH.write_text('#!/usr/bin/env bash\nbash -c "$1"\n', encoding="utf-8")
LOCAL_POD_BASH.chmod(0o755)

#: A stand-in for h200_status.sh that states the verdict a healthy probe states.
#:
#: The default here used to be `/bin/true`, which is precisely what the
#: controller may no longer accept: the probe answers on its terminal `Health=`
#: line, because the layer carrying its exit status returns 0 for a command that
#: exited 7 (L20). A stub that only succeeds exercises none of that gate.
LOCAL_STATUS_OK = _STUB_DIR / "status_ok.sh"
LOCAL_STATUS_OK.write_text(
    "#!/usr/bin/env bash\nprintf 'Health=ok\\n'\n", encoding="utf-8"
)
LOCAL_STATUS_OK.chmod(0o755)


def extract_function(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    marker = f"\n{name}() {{\n"
    if marker not in source:
        raise AssertionError(f"{path.name} has no function {name}")
    body = source.split(marker, 1)[1]
    end = body.index("\n}\n")
    return f"{name}() {{\n{body[:end]}\n}}\n"


def worker_functions(*names: str) -> str:
    """Several of the worker's own functions, verbatim, for a harness to drive."""

    return "\n".join(extract_function(WORKER, name) for name in names)


def controller_env(**overrides: str) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("ARGS_")}
    env.update(
        {
            "H200_POD": "test-pod",
            "H200_STATUS_CHECK": str(LOCAL_STATUS_OK),
            "H200_SYNC": "/bin/true",
            "H200_GPFS_PUSH": "/bin/true",
            "H200_POD_BASH": str(LOCAL_POD_BASH),
            "H200_POD_EXEC": "/bin/true",
            "ARMS": "gpt2-large",
            "GPUS": "0",
            "STAGES": "cohort_power",
        }
    )
    env.update(overrides)
    return env


def make_project_copy(root: Path) -> Path:
    """A minimal checkout the controller can freeze, hash, push and log against.

    Every controller invocation that is not a ``--dry-run`` writes its own copy
    of the worker log to ``${PROJECT_ROOT}/logs/transfer_h200_controller/``, and
    ``PROJECT_ROOT`` defaults to the checkout the controller was started from. A
    test that overrides ``GPFS_*`` and not ``PROJECT_ROOT`` therefore files
    synthetic campaign records in the operational controller log directory,
    under run-ids carrying the real code hash -- indistinguishable from a real
    campaign by filename. Measured before this was fixed: 189 of the 294 files
    in that directory were this suite's, one of them recording a `campaign
    INCOMPLETE` verdict for a campaign that never ran.

    So it is a module-level helper rather than one class's method: every test
    that runs the controller for real builds its own project root from it.
    """

    project = root / "project"
    (project / "src").mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "src" / "__init__.py", project / "src" / "__init__.py")
    shutil.copytree(REPO_ROOT / "src" / "transfer", project / "src" / "transfer")
    shutil.copytree(TRANSFER_DIR, project / "scripts" / "transfer")
    return project


class RequestValidationTests(unittest.TestCase):
    def run_controller(self, **environment: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(CONTROLLER), "--dry-run"],
            cwd="/tmp",
            env=controller_env(**environment),
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_default_repo_root_comes_from_controller_location(self):
        result = self.run_controller()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"PROJECT_ROOT:      {REPO_ROOT}", result.stdout)

    def test_duplicate_stages_arms_and_gpus_are_rejected(self):
        cases = (
            ({"ARMS": "gpt2-large,gpt2-large"}, "ARMS contains duplicate value"),
            ({"STAGES": "cohort_power,cohort_power"}, "STAGES contains duplicate value"),
            ({"GPUS": "0,0"}, "GPUS contains duplicate value"),
        )
        for environment, message in cases:
            with self.subTest(environment=environment):
                result = self.run_controller(**environment)
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stderr)
        worker_source = WORKER.read_text(encoding="utf-8")
        self.assertIn('reject_duplicate_values STAGES "${REQUESTED_STAGES[@]}"', worker_source)
        self.assertIn('reject_duplicate_values ARMS "${ARM_LIST[@]}"', worker_source)
        self.assertIn('reject_duplicate_values GPUS "${GPU_LIST[@]}"', worker_source)

    def test_unknown_args_environment_variable_is_rejected(self):
        result = self.run_controller(ARGS_COHORT_POWRE="--n-seq 1")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown stage-argument environment variable: ARGS_COHORT_POWRE", result.stderr)

    def test_an_item_scoped_override_outside_arms_is_refused_not_recorded(self):
        # The controller enumerated a stage's items from the contract alone, so
        # an override naming an arm this invocation does not dispatch passed
        # validation, was written into RUN_MANIFEST.json's
        # parameters.stage_args, and was sent on as
        # `--item-args lens_family zymctrl <base64>` -- which the worker never
        # read, because its lens_family items are the contract's eligible arms
        # intersected with ARMS. The manifest then asserted a scale parameter the
        # campaign had not run at, so this is a refusal and not a silent drop.
        result = self.run_controller(
            ARMS="gpt2-large,protgpt2",
            STAGES="lens_family",
            ARGS_LENS_FAMILY__ZYMCTRL="--n-seq 64",
        )
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("lens_family/zymctrl", result.stderr)
        self.assertNotIn("--item-args", result.stdout)

    def test_a_cohort_item_whose_arms_are_all_outside_arms_is_refused(self):
        # cohort_power's items are cohort labels rather than arm names, and the
        # worker drops a label whose every arm falls outside ARMS. Same
        # falsehood, different item space.
        result = self.run_controller(
            ARMS="gpt2-large",
            STAGES="cohort_power",
            ARGS_COHORT_POWER__PROTEIN_SMALL_VOCAB="--n-seq 8",
        )
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("cohort_power/protein_small_vocab", result.stderr)

    def test_the_same_override_is_carried_when_arms_dispatches_the_item(self):
        # The refusal is about provenance, not about narrowing: ask for the arm
        # and the override reaches the worker untouched.
        result = self.run_controller(
            ARMS="gpt2-large,zymctrl",
            STAGES="lens_family",
            ARGS_LENS_FAMILY__ZYMCTRL="--n-seq 64",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("lens_family/zymctrl=[--n-seq 64]", result.stdout)
        self.assertIn("--item-args lens_family zymctrl", result.stdout)

    def test_text_arm_is_validated_against_the_contract(self):
        # ARMS, STAGES and GPUS were each validated and TEXT_ARM was not, so
        # TEXT_ARM=gpt2 instead of gpt2-large -- two valid text arms -- ran the
        # whole campaign and anchored the panel verdict on a different control
        # with nothing anywhere saying so. Evidence discipline rule 1 is about
        # which arm this is.
        unknown = self.run_controller(TEXT_ARM="gpt2-larg")
        self.assertEqual(unknown.returncode, 2, unknown.stdout)
        self.assertIn("unknown TEXT_ARM: gpt2-larg", unknown.stderr)

    def test_a_protein_text_arm_is_refused_before_the_freeze(self):
        wrong_modality = self.run_controller(TEXT_ARM="protgpt2", ARMS="gpt2-large,protgpt2")
        self.assertEqual(wrong_modality.returncode, 2, wrong_modality.stdout)
        self.assertIn("TEXT_ARM=protgpt2 has modality protein", wrong_modality.stderr)
        # Refused before anything was staged or hashed.
        self.assertNotIn("staging and hashing", wrong_modality.stdout)

    def test_a_different_but_valid_text_control_is_still_allowed(self):
        # Narrowing and re-anchoring remain available; they just have to be asked
        # for and be a text arm the panel declares.
        result = self.run_controller(TEXT_ARM="gpt2", ARMS="gpt2,protgpt2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("TEXT_ARM:          gpt2", result.stdout)

    def test_flag_equals_form_is_normalized_before_duplicate_check(self):
        script = f"""
        set -euo pipefail
        {extract_function(WORKER, "assert_no_duplicate_options")}
        assert_no_duplicate_options cohort_power text \
          python 01_cohort_power.py --out /safe --out=/unsafe
        """
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("repeat --out", result.stderr)


#: The globals the worker's campaign-accounting functions read, so a harness can
#: drive them without the surrounding script. Kept in one place because five tests
#: below need the same preamble and a drifting copy of it would test nothing.
LEDGER_PREAMBLE = """
        SKIPPED_FOR_DATA=()
        DEFERRED_FAILURES=()
        UNMEASURED_STAGES=()
        DISPATCHED_STAGES=()
        CAMPAIGN_LEDGER_PRINTED=0
        WORKER_EXIT_SENTINEL="TRANSFER_WORKER_EXIT="
        emit_exit_sentinel() { printf '%s%s\\n' "${WORKER_EXIT_SENTINEL}" "${1:-0}"; }
        SKIP_DATA_STATUS=75
        GPU_LIST=(0 1)
        RUN_ID=test_000000000000
        RESULTS_ROOT=/results
        LOGS_ROOT=/logs
        ARMS=gpt2-large,protgpt2
        STAGES=relational_channel
        log() { printf '%s\\n' "$*"; }
"""


class MissingDataAggregationTests(unittest.TestCase):
    def test_background_skips_reach_parent_and_prevent_success(self):
        functions = worker_functions(
            "run_stage_wave",
            "record_unmeasured_stage",
            "print_campaign_ledger",
            "finish_campaign",
        )
        script = f"""
        set -euo pipefail
        {LEDGER_PREAMBLE}
        {functions}
        run_item_atomic() {{
          return "$SKIP_DATA_STATUS"
        }}
        run_stage_wave test_stage item_a item_b
        printf 'parent_count=%s\n' "${{#SKIPPED_FOR_DATA[@]}}"
        finish_campaign
        """
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("parent_count=2", result.stdout)
        self.assertIn("test_stage incomplete", result.stdout)
        self.assertNotIn("campaign complete", result.stdout + result.stderr)
        self.assertIn("campaign INCOMPLETE", result.stderr)
        self.assertIn("test_stage/item_a", result.stderr)
        self.assertIn("test_stage/item_b", result.stderr)


class UnmeasuredStagesFailTheCampaign(unittest.TestCase):
    """A requested stage that measured nothing must not report success.

    `ARMS=gpt2-large,protgpt2 STAGES=relational_channel,homology_control` measured
    nothing at all -- relational_channel serves neither arm and homology_control
    serves neither -- and printed "campaign complete", exit 0. Three call sites
    logged one "skipping" line and returned 0, and nothing downstream could tell
    the difference between that and a stage that had run.
    """

    def run_ledger_script(self, body: str) -> subprocess.CompletedProcess[str]:
        functions = worker_functions(
            "record_unmeasured_stage",
            "print_campaign_ledger",
            "finish_campaign",
            "run_stage_wave",
            "run_panel_stage",
            "dispatch_stage",
            "reconcile_dispatched_stages",
        )
        script = f"""
        set -euo pipefail
        {LEDGER_PREAMBLE}
        {functions}
        {body}
        """
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True)

    def test_a_wave_with_no_items_is_recorded_and_fails_the_campaign(self):
        result = self.run_ledger_script(
            """
            run_stage_wave relational_channel
            finish_campaign
            """
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertNotIn("campaign complete", result.stdout)
        self.assertIn("UNMEASURED: relational_channel", result.stderr)
        self.assertIn("measured nothing", result.stderr)

    def test_a_panel_stage_with_no_eligible_arm_is_recorded(self):
        result = self.run_ledger_script(
            """
            run_panel_stage homology_control 0
            finish_campaign
            """
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("UNMEASURED: homology_control", result.stderr)
        # The skip itself is still correct and its reason must survive: an empty
        # --arms would fall back to the entry point's own default panel.
        self.assertIn("default panel", result.stderr)

    def test_estimand_power_with_no_eligible_arm_is_recorded(self):
        result = self.run_ledger_script(
            f"""
            {worker_functions("run_estimand_power")}
            stage_final_dir() {{ printf '%s\\n' "/results/$1"; }}
            TEXT_ARM=gpt2-large
            run_estimand_power
            finish_campaign
            """
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("UNMEASURED: estimand_power", result.stderr)

    def test_a_measured_stage_leaves_the_campaign_clean(self):
        # The negative path: the same accounting must not manufacture a failure
        # for a stage that did run.
        result = self.run_ledger_script(
            """
            run_item_atomic() { return 0; }
            run_stage_wave relational_channel zymctrl
            finish_campaign
            """
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("campaign complete", result.stdout)
        self.assertNotIn("UNMEASURED", result.stderr)

    def test_a_requested_stage_no_branch_dispatches_is_reconciled(self):
        # The dispatch chain is eleven hand-written branches and nothing checked
        # it against REQUESTED_STAGES, so a stage added to the panel contract and
        # forgotten there passed every preflight, never ran, and exited 0.
        result = self.run_ledger_script(
            """
            REQUESTED_STAGES=(cohort_power a_stage_nobody_wired)
            stage_requested() {
              case " ${REQUESTED_STAGES[*]} " in *" $1 "*) return 0 ;; *) return 1 ;; esac
            }
            ran=0
            dispatch_stage cohort_power eval 'ran=1'
            printf 'ran=%s dispatched=%s\n' "$ran" "${DISPATCHED_STAGES[*]}"
            reconcile_dispatched_stages
            finish_campaign
            """
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("ran=1 dispatched=cohort_power", result.stdout)
        self.assertIn("UNMEASURED: a_stage_nobody_wired", result.stderr)
        self.assertIn("no branch of this worker's tier chain dispatches it", result.stderr)

    def test_reconciliation_passes_when_every_requested_stage_dispatched(self):
        result = self.run_ledger_script(
            """
            REQUESTED_STAGES=(cohort_power lens_family)
            stage_requested() {
              case " ${REQUESTED_STAGES[*]} " in *" $1 "*) return 0 ;; *) return 1 ;; esac
            }
            dispatch_stage cohort_power true
            dispatch_stage lens_family true
            dispatch_stage probe_and_erasure false   # not requested: must not run
            reconcile_dispatched_stages
            finish_campaign
            """
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("campaign complete", result.stdout)

    def test_every_contract_stage_reaches_dispatch_stage_in_the_tier_chain(self):
        # The reconciliation above is the runtime guard; this is the static one,
        # and it is what makes adding a stage to panel_contract.py fail loudly
        # here rather than silently at the end of a scheduled campaign.
        source = WORKER.read_text(encoding="utf-8")
        dispatched = set(re.findall(r"^dispatch_stage (\S+) ", source, flags=re.MULTILINE))
        self.assertEqual(
            dispatched,
            set(pc.STAGE_ORDER),
            "h200_worker.sh's tier chain and panel_contract.py's stage order disagree",
        )


class EarlyExitPreservesTheLedger(unittest.TestCase):
    """finish_campaign is the only printer of the ledger and four call sites leave
    before it. A tier-1 data skip followed by a tier-3 item failure discarded the
    SKIP-DATA record that was added precisely so a data skip could not be lost."""

    def test_a_hard_failure_after_a_data_skip_still_reports_the_skip(self):
        functions = worker_functions(
            "record_unmeasured_stage",
            "print_campaign_ledger",
            "finish_campaign",
            "report_early_exit",
            "run_stage_wave",
        )
        script = f"""
        set -euo pipefail
        {LEDGER_PREAMBLE}
        {functions}
        trap report_early_exit EXIT
        run_item_atomic() {{
          case "$2" in
            skipped_item) return "$SKIP_DATA_STATUS" ;;
            *) return 1 ;;
          esac
        }}
        # tier 1: one item cannot run because its input is not staged.
        run_stage_wave tier1_stage skipped_item
        # tier 3: a genuine item failure, which exits the worker at once.
        run_stage_wave tier3_stage broken_item
        printf 'unreachable\n'
        """
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("unreachable", result.stdout)
        self.assertIn("SKIP-DATA: tier1_stage/skipped_item", result.stderr)
        self.assertIn("exited early with status 1", result.stderr)

    def test_the_trap_preserves_the_exit_status_exactly(self):
        functions = worker_functions(
            "print_campaign_ledger", "finish_campaign", "report_early_exit"
        )
        for status in (2, 75, 1):
            with self.subTest(status=status):
                script = f"""
                set -euo pipefail
                {LEDGER_PREAMBLE}
                {functions}
                trap report_early_exit EXIT
                DEFERRED_FAILURES+=("something/failed")
                exit {status}
                """
                result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
                self.assertEqual(result.returncode, status)
                self.assertIn("something/failed", result.stderr)

    def test_the_ledger_is_printed_once_not_twice(self):
        functions = worker_functions(
            "print_campaign_ledger", "finish_campaign", "report_early_exit"
        )
        script = f"""
        set -euo pipefail
        {LEDGER_PREAMBLE}
        {functions}
        trap report_early_exit EXIT
        SKIPPED_FOR_DATA+=("stage/item")
        finish_campaign
        """
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stderr.count("SKIP-DATA: stage/item"), 1)

    def test_a_preflight_refusal_is_not_given_a_spurious_second_voice(self):
        # Nothing accumulated: the refusal that exited already said why, and the
        # trap must not bury it under an empty ledger.
        functions = worker_functions(
            "print_campaign_ledger", "finish_campaign", "report_early_exit"
        )
        script = f"""
        set -euo pipefail
        {LEDGER_PREAMBLE}
        {functions}
        trap report_early_exit EXIT
        echo "GPU 0 is occupied; refusing to schedule" >&2
        exit 2
        """
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("exited early", result.stderr)
        self.assertNotIn("campaign", result.stderr)

    def test_the_worker_actually_installs_the_trap(self):
        source = WORKER.read_text(encoding="utf-8")
        self.assertIn("\ntrap report_early_exit EXIT\n", source)


class RecommendRefusesAbsentMeasureOutputs(unittest.TestCase):
    """`estimand_power`'s recommend aggregated `cat`-ed per-arm manifests with
    `2>/dev/null`; a missing one -- exactly what a legitimate SKIP-DATA produces --
    made the pipeline non-zero under pipefail and `set -e` killed the worker with
    NO message, before tiers 3 and 4 and before the SKIP-DATA summary was ever
    printed."""

    def build_script(self, out_dir: Path, python: str) -> str:
        functions = worker_functions(
            "record_unmeasured_stage",
            "print_campaign_ledger",
            "finish_campaign",
            "canonicalize_command",
            "provenance_record",
            "item_is_complete",
            "explain_incomplete",
            "run_estimand_power",
        )
        return f"""
        set -euo pipefail
        {LEDGER_PREAMBLE}
        FORCE=0
        TEXT_ARM=gpt2-large
        CODE_HASH_SHORT=abcdef123456
        TRANSFER_PYTHON={python}
        TRANSFER_SCRIPTS=/snapshot/scripts/transfer
        LOGS_ROOT={out_dir}/logs
        RESULTS_ROOT={out_dir}
        stage_final_dir() {{ printf '%s\\n' "{out_dir}/$1"; }}
        arm_modality() {{
          case "$1" in gpt2-large) printf 'text\\n' ;; *) printf 'protein\\n' ;; esac
        }}
        run_stage_wave() {{ printf 'measure %s\\n' "$*"; }}
        {functions}
        run_estimand_power gpt2-large protgpt2
        finish_campaign
        """

    def test_a_missing_measure_manifest_is_a_named_refusal_not_a_silent_death(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "estimand_power" / ".manifests").mkdir(parents=True)
            (out / "logs").mkdir()
            # Only the text control measured; the protein arm skipped for data.
            (out / "estimand_power" / ".manifests" / "gpt2-large.sha256").write_text(
                "deadbeef  gpt2-large.json\n", encoding="utf-8"
            )
            result = subprocess.run(
                ["bash", "-c", self.build_script(out, "/bin/false")],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("no measure output for: protgpt2", result.stdout)
            self.assertIn("estimand_power/recommend", result.stderr)
            self.assertIn("cannot anchor a panel verdict", result.stderr)
            # It must NOT have died at the aggregation: the campaign ledger was
            # reached and printed, which is the whole point.
            self.assertIn("campaign INCOMPLETE", result.stderr)

    def test_recommend_still_runs_when_every_measure_output_is_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "estimand_power" / ".manifests").mkdir(parents=True)
            (out / "logs").mkdir()
            for arm in ("gpt2-large", "protgpt2"):
                (out / "estimand_power" / ".manifests" / f"{arm}.sha256").write_text(
                    f"deadbeef  {arm}.json\n", encoding="utf-8"
                )
            fake_python = out / "fake_python"
            fake_python.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "while [ $# -gt 0 ]; do\n"
                '  if [ "$1" = --output ]; then printf \'{}\\n\' > "$2"; fi\n'
                "  shift\n"
                "done\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            result = subprocess.run(
                ["bash", "-c", self.build_script(out, str(fake_python))],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("done  stage=estimand_power item=recommend", result.stdout)
            self.assertTrue((out / "estimand_power" / "recommendation.json").is_file())
            self.assertIn("campaign complete", result.stdout)


class CheckpointPreflightIsPerCheckpoint(unittest.TestCase):
    """Six of seven text arms resolve TRANSFER_TEXT_MODEL_BASE_DIR, the models
    ROOT. It exists as soon as any text checkpoint is staged, so an arm whose own
    checkpoint was absent passed this check and raised inside load_arm -- and
    cohort_power runs all seven text arms in ONE process, so that lost the six that
    were fine."""

    def run_check(self, tmp: Path, stage: str, item: str) -> subprocess.CompletedProcess[str]:
        functions = worker_functions(
            "arm_modality",
            "model_var_for_arm",
            "model_path_for_arm",
            "corpus_vars_for_arms",
            "arms_for_item",
            "extra_vars_for_stage",
            "verify_item_data_paths",
        )
        script = f"""
        set -euo pipefail
        source {pc.SHELL_CONTRACT}
        export TRANSFER_TEXT_MODEL_BASE_DIR={tmp}/text_models
        export TRANSFER_TEXT_MODEL_DIR={tmp}/text_models/gpt2-large
        export TRANSFER_MODEL_BASE_DIR={tmp}/models
        export TRANSFER_OPENWEBTEXT_DIR={tmp}/openwebtext
        export TRANSFER_SWISSPROT_FASTA={tmp}/swissprot.fasta
        log() {{ printf '%s\\n' "$*"; }}
        declare -A COHORT_ITEM_ARMS_FOR=()
        {functions}
        if verify_item_data_paths {stage} {item}; then
          printf 'RESULT ok\\n'
        else
          printf 'RESULT skip %s\\n' "$MISSING_DATA_REASON"
        fi
        """
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True)

    def test_a_staged_root_with_an_absent_checkpoint_is_a_skip_not_a_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # The models ROOT is staged and holds one arm's checkpoint...
            (root / "text_models" / "gpt2-medium").mkdir(parents=True)
            (root / "openwebtext").mkdir()
            present = self.run_check(root, "pathway_budget", "gpt2-medium")
            self.assertEqual(present.returncode, 0, present.stderr)
            self.assertIn("RESULT ok", present.stdout)
            # ...but this arm's own checkpoint is not in it.
            absent = self.run_check(root, "pathway_budget", "gpt2")
            self.assertEqual(absent.returncode, 0, absent.stderr)
            self.assertIn("RESULT skip", absent.stdout)
            self.assertIn("missing checkpoint for arm gpt2", absent.stdout)
            self.assertIn(f"{root}/text_models/gpt2", absent.stdout)

    def test_the_arm_declared_as_the_variable_itself_resolves_to_the_variable(self):
        # gpt2-large is declared as TEXT_MODEL_ROOT, so its relative path is "."
        # and appending a leaf name would look for gpt2-large/gpt2-large.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "text_models" / "gpt2-large").mkdir(parents=True)
            (root / "openwebtext").mkdir()
            result = self.run_check(root, "pathway_budget", "gpt2-large")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("RESULT ok", result.stdout)

    def test_the_relative_paths_come_from_the_contract_not_from_the_worker(self):
        source = WORKER.read_text(encoding="utf-8")
        self.assertIn("TRANSFER_ARM_MODEL_REL", source)
        # Prose about a model is fine; a checkpoint directory name in executable
        # bash is a twelfth hand-maintained list waiting to drift from PANEL.
        code = [
            line
            for line in source.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        for leaf in ("ProtGPT2", "ZymCTRL", "DialoGPT-small", "Qwen2.5-0.5B", "Llama-3.2-3B"):
            for line in code:
                self.assertNotIn(
                    leaf, line, f"{leaf} is a checkpoint leaf name restated in bash: {line}"
                )
        self.assertEqual(pc.model_relative_path("gpt2-large"), ".")
        self.assertEqual(pc.model_relative_path("protgpt2"), "ProtGPT2")


class ReadmeStageTableMatchesTheContract(unittest.TestCase):
    """The operator guide's stage table is a hand-maintained copy of a generated
    declaration, which is the exact failure class panel_contract.py exists to end.
    It had already drifted: induction_path_patching was listed with seven eligible
    arms against the contract's nine, omitting both ProGen2 arms."""

    def stage_rows(self) -> list[list[str]]:
        rows = []
        for line in README.read_text(encoding="utf-8").splitlines():
            if not line.startswith("|"):
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) != 5 or not cells[0].isdigit():
                continue
            rows.append(cells)
        return rows

    def test_the_table_lists_every_contract_stage_in_order(self):
        rows = self.stage_rows()
        self.assertEqual(
            [cell[1].strip("`") for cell in rows],
            list(pc.STAGE_ORDER),
            "README stage table and panel_contract.py's STAGE_ORDER disagree",
        )
        self.assertEqual([int(cell[0]) for cell in rows], list(range(1, len(rows) + 1)))

    def test_each_row_names_the_contract_entry_point(self):
        for order, stage, entry, _scope, _arms in self.stage_rows():
            with self.subTest(stage=stage):
                self.assertEqual(
                    entry.strip("`"),
                    pc.STAGE_CONTRACTS[stage.strip("`")].entry_point,
                    f"README row {order} names the wrong entry point",
                )

    def test_each_row_names_the_contract_eligible_arms(self):
        for _order, stage, _entry, _scope, arms_cell in self.stage_rows():
            name = stage.strip("`")
            with self.subTest(stage=name):
                eligible = pc.stage_arms(name)[0]
                if pc.STAGE_CONTRACTS[name].scope == "armless":
                    self.assertEqual(arms_cell, "no arm dispatch")
                    self.assertEqual(eligible, [])
                    continue
                shorthand = re.fullmatch(r"all (\d+)", arms_cell)
                if shorthand:
                    self.assertEqual(int(shorthand.group(1)), len(pc.CAMPAIGN_PANEL))
                    self.assertEqual(eligible, list(pc.CAMPAIGN_PANEL))
                else:
                    self.assertEqual(re.findall(r"`([^`]+)`", arms_cell), eligible)

    def test_every_readme_section_the_shell_scripts_cite_exists(self):
        # Four comments pointed at sections that do not exist -- "Environment
        # contract" (four sites), "Atomicity and resume", "Known host-bound
        # quantities" -- so an operator following the pointer found nothing and
        # the fact the comment was deferring to was simply unavailable.
        headings = {
            line[3:].strip()
            for line in README.read_text(encoding="utf-8").splitlines()
            if line.startswith("## ")
        }
        cited: set[str] = set()
        for path in (WORKER, CONTROLLER):
            # A citation may wrap across two comment lines or two heredoc lines,
            # so the line structure is folded out before matching.
            source = re.sub(r"\n#\s*", " ", path.read_text(encoding="utf-8"))
            for pattern in (
                r"README\.md's \"([^\"]+)\"",
                r"\"([^\"]+)\" in scripts/transfer/README\.md",
            ):
                cited |= {" ".join(m.split()) for m in re.findall(pattern, source)}
        self.assertTrue(cited, "no README cross-reference found; the check would be vacuous")
        self.assertEqual(
            sorted(cited - headings),
            [],
            f"shell comments cite README sections that do not exist; headings are {sorted(headings)}",
        )

    def test_the_prose_counts_and_arm_list_match_the_contract(self):
        text = README.read_text(encoding="utf-8")
        self.assertIn(f"The active campaign has {len(pc.CAMPAIGN_PANEL)} arms:", text)
        self.assertIn(f"The contract declares {len(pc.STAGE_ORDER)} stages in this order:", text)
        arm_paragraph = text.split(f"has {len(pc.CAMPAIGN_PANEL)} arms:", 1)[1].split("\n\n", 2)[1]
        self.assertEqual(re.findall(r"`([^`]+)`", arm_paragraph), list(pc.CAMPAIGN_PANEL))


class SnapshotContractTests(unittest.TestCase):
    def write_executable(self, path: Path, body: str) -> None:
        path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body, encoding="utf-8")
        path.chmod(0o755)

    def test_transfer_and_reuse_verify_complete_staged_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = make_project_copy(root)
            helpers = root / "helpers"
            helpers.mkdir()
            marker = root / "worker-invocations"
            # The probe states its verdict on stdout; `:` says nothing and the
            # controller now refuses silence rather than reading exit status.
            self.write_executable(helpers / "status", "printf 'Health=ok\\n'\n")
            self.write_executable(
                helpers / "sync",
                '[ "$1" = push ]\nmkdir -p "$3"\ncp -a "$2/." "$3/"\n',
            )
            self.write_executable(
                helpers / "push",
                'mkdir -p "$(dirname "$2")"\ncp -p "$1" "$2"\n',
            )
            self.write_executable(helpers / "pod_bash", 'exec bash -c "$1"\n')
            self.write_executable(
                helpers / "pod_exec",
                # A worker that reaches its EXIT trap states its status on its last
                # line; the controller reads that rather than the transport's
                # exit code, which is always 0 (L20).
                'printf "called\\n" >> "$POD_EXEC_MARKER"\nprintf "TRANSFER_WORKER_EXIT=0\\n"\n',
            )
            package_root = root / "gpfs" / "packages"
            env = controller_env(
                PROJECT_ROOT=str(project),
                REPO_ROOT=str(project),
                H200_STATUS_CHECK=str(helpers / "status"),
                H200_SYNC=str(helpers / "sync"),
                H200_GPFS_PUSH=str(helpers / "push"),
                H200_POD_BASH=str(helpers / "pod_bash"),
                H200_POD_EXEC=str(helpers / "pod_exec"),
                GPFS_PACKAGE_ROOT=str(package_root),
                GPFS_RESULTS_ROOT=str(root / "gpfs" / "results"),
                GPFS_LOGS_ROOT=str(root / "gpfs" / "logs"),
                POD_EXEC_MARKER=str(marker),
            )
            first = subprocess.run(
                ["bash", str(CONTROLLER)],
                cwd="/tmp",
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            match = re.search(r"run_id=([0-9]{14}_[0-9a-f]{12}) code_hash=", first.stdout)
            self.assertIsNotNone(match, first.stdout)
            run_id = match.group(1)
            snapshot = package_root / run_id
            manifest = snapshot / "CODE_CONTENT_SHA256SUMS"
            self.assertTrue(manifest.is_file())
            self.assertNotIn("docs/", manifest.read_text(encoding="utf-8"))
            subprocess.run(
                ["sha256sum", "-c", "--", manifest.name],
                cwd=snapshot,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            invocations = list((snapshot / "INVOCATIONS").glob("*.json"))
            self.assertEqual(len(invocations), 1)
            self.assertEqual(
                invocations[0].stem,
                hashlib.sha256(invocations[0].read_bytes()).hexdigest(),
            )
            self.assertEqual(marker.read_text(encoding="utf-8").splitlines(), ["called"])

            with (snapshot / "scripts" / "transfer" / "h200_env.sh").open("a", encoding="utf-8") as handle:
                handle.write("\n# corruption injected by test\n")
            second_env = dict(env)
            second_env["RUN_ID"] = run_id
            second = subprocess.run(
                ["bash", str(CONTROLLER)],
                cwd="/tmp",
                env=second_env,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(second.returncode, 2)
            self.assertIn("snapshot checksum verification failed on GPFS", second.stderr)
            self.assertEqual(marker.read_text(encoding="utf-8").splitlines(), ["called"])

    def test_a_failing_worker_is_reported_with_its_run_id_and_log_pointer(self):
        # `set -e` fires at the pipeline inside invoke_worker, so `return
        # "${PIPESTATUS[0]}"` and the diagnostic that consumed it were both dead
        # code: the status propagated but the operator lost the run-id and the log
        # path at exactly the moment they are needed.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = make_project_copy(root)
            helpers = root / "helpers"
            helpers.mkdir()
            self.write_executable(helpers / "status", "printf 'Health=ok\\n'\n")
            self.write_executable(
                helpers / "sync",
                '[ "$1" = push ]\nmkdir -p "$3"\ncp -a "$2/." "$3/"\n',
            )
            self.write_executable(
                helpers / "push",
                'mkdir -p "$(dirname "$2")"\ncp -p "$1" "$2"\n',
            )
            self.write_executable(helpers / "pod_bash", 'exec bash -c "$1"\n')
            self.write_executable(
                helpers / "pod_exec",
                # The transport exits 7 here, but on the real cluster it would exit
                # 0; what makes this detectable is the worker's own sentinel.
                'printf "worker output\\n"\nprintf "TRANSFER_WORKER_EXIT=7\\n"\nexit 7\n',
            )
            env = controller_env(
                PROJECT_ROOT=str(project),
                REPO_ROOT=str(project),
                H200_STATUS_CHECK=str(helpers / "status"),
                H200_SYNC=str(helpers / "sync"),
                H200_GPFS_PUSH=str(helpers / "push"),
                H200_POD_BASH=str(helpers / "pod_bash"),
                H200_POD_EXEC=str(helpers / "pod_exec"),
                GPFS_PACKAGE_ROOT=str(root / "gpfs" / "packages"),
                GPFS_RESULTS_ROOT=str(root / "gpfs" / "results"),
                GPFS_LOGS_ROOT=str(root / "gpfs" / "logs"),
            )
            result = subprocess.run(
                ["bash", str(CONTROLLER)],
                cwd="/tmp",
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 7, result.stderr)
            match = re.search(r"worker failed with status 7 \(run_id=(\S+)\); see (\S+)", result.stderr)
            self.assertIsNotNone(match, result.stderr)
            run_id, log_path = match.group(1), match.group(2)
            self.assertTrue(Path(log_path).is_file(), f"{log_path} was named but not written")
            self.assertIn("worker output", Path(log_path).read_text(encoding="utf-8"))
            self.assertIn(run_id, log_path)
            self.assertNotIn("campaign complete", result.stdout)

    def test_worker_startup_rejects_corrupt_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp)
            runtime = snapshot / "runtime.txt"
            runtime.write_text("original\n", encoding="utf-8")
            digest = hashlib.sha256(runtime.read_bytes()).hexdigest()
            manifest = snapshot / "CODE_CONTENT_SHA256SUMS"
            manifest.write_text(f"{digest}  runtime.txt\n", encoding="utf-8")
            code_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
            function = extract_function(WORKER, "verify_snapshot_manifest")
            script = f"""
            set -euo pipefail
            SNAPSHOT_DIR={snapshot}
            RUN_ID=test_{code_hash[:12]}
            CODE_HASH_SHORT={code_hash[:12]}
            {function}
            verify_snapshot_manifest
            """
            valid = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
            self.assertEqual(valid.returncode, 0, valid.stderr)
            runtime.write_text("changed\n", encoding="utf-8")
            corrupt = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
            self.assertEqual(corrupt.returncode, 2)
            self.assertIn("snapshot content checksum verification failed", corrupt.stderr)
            source = WORKER.read_text(encoding="utf-8")
            self.assertLess(
                source.index("\nverify_snapshot_manifest\n"),
                source.index('source "${PANEL_CONTRACT_SH}"'),
            )


#: The healthy verdict, as the access-layer stub writes it.
HEALTHY_STATUS_STUB = "#!/usr/bin/env bash\nprintf 'Health=ok\\n'\n"


def access_layer_stubs(tmp, pod_exec_body, status_body=HEALTHY_STATUS_STUB) -> Path:
    """The five access-layer tools as local stubs, under an ssh_tunnel/ root."""

    access = Path(tmp) / "ssh_tunnel"
    access.mkdir(parents=True)
    (access / "h200_status.sh").write_text(status_body, encoding="utf-8")
    # The transfer stubs really move the bytes, because verify_remote_snapshot
    # can now fail and a stub that transferred nothing would fail it. That is
    # the point: before this change the verification could not fail, so a
    # do-nothing stub passed it and so, on the cluster, would a do-nothing
    # transfer.
    (access / "h200_sync.sh").write_text(
        '#!/usr/bin/env bash\nmkdir -p -- "$3"\ncp -r -- "$2"/. "$3"/\n',
        encoding="utf-8",
    )
    (access / "h200_gpfs_push.sh").write_text(
        '#!/usr/bin/env bash\nmkdir -p -- "$(dirname -- "$2")"\ncp -- "$1" "$2"\n',
        encoding="utf-8",
    )
    for name in ("h200_status", "h200_sync", "h200_gpfs_push"):
        (access / f"{name}.sh").chmod(0o755)
    # Every remote question this controller asks answers on stdout, and the
    # command string carries its own echo, so the stub evaluates it locally
    # rather than merely succeeding.
    pod_bash = access / "h200_pod_bash.sh"
    pod_bash.write_text('#!/usr/bin/env bash\nbash -c "$1"\n', encoding="utf-8")
    pod_bash.chmod(0o755)
    exec_helper = access / "h200_pod_exec.sh"
    exec_helper.write_text(pod_exec_body, encoding="utf-8")
    exec_helper.chmod(0o755)
    return access


def run_controller_with_stubs(tmp, access, **overrides) -> subprocess.CompletedProcess[str]:
    """One real controller run against local stubs, in its own project root.

    PROJECT_ROOT is the whole reason this is a shared helper. It is what the
    controller derives CONTROLLER_LOG_DIR from, so a run that leaves it at the
    default files its worker log in the operational
    logs/transfer_h200_controller/ of whatever checkout the suite is run from.
    """

    project = make_project_copy(Path(tmp))
    # H200_ACCESS_ROOT is the parent of ssh_tunnel/, which is where
    # access_layer_stubs put the stubs.
    environment = controller_env(
        PROJECT_ROOT=str(project),
        REPO_ROOT=str(project),
        H200_ACCESS_ROOT=str(access.parent),
        H200_STATUS_CHECK=str(access / "h200_status.sh"),
        H200_SYNC=str(access / "h200_sync.sh"),
        H200_GPFS_PUSH=str(access / "h200_gpfs_push.sh"),
        H200_POD_BASH=str(access / "h200_pod_bash.sh"),
        H200_POD_EXEC=str(access / "h200_pod_exec.sh"),
        GPFS_PACKAGE_ROOT=str(Path(tmp) / "packages"),
        GPFS_RESULTS_ROOT=str(Path(tmp) / "results"),
        GPFS_LOGS_ROOT=str(Path(tmp) / "logs"),
        **overrides,
    )
    return subprocess.run(
        ["bash", str(CONTROLLER)],
        capture_output=True,
        text=True,
        cwd=tmp,
        env=environment,
        timeout=180,
    )


class WorkerStatusDoesNotTravelOnTheTransport(unittest.TestCase):
    """The access layer returns 0 whatever the remote command exits with.

    Measured on this deployment: `h200_pod_exec.sh -- bash -c "exit 7"` returns
    0. So a worker that refused a campaign at preflight and scheduled no GPU came
    back to the operator as `campaign complete`, exit 0 -- the false-success shape
    the worker's whole ledger exists to prevent, defeated one layer above it. The
    worker now states its status on its last line and the controller reads that.
    """

    def test_the_sentinel_has_one_declaration_and_the_controller_reads_it(self):
        worker = WORKER.read_text(encoding="utf-8")
        controller = CONTROLLER.read_text(encoding="utf-8")
        declarations = [
            line
            for line in worker.splitlines()
            if line.startswith("WORKER_EXIT_SENTINEL=")
        ]
        self.assertEqual(len(declarations), 1, "the sentinel must be declared once")
        self.assertNotIn(
            'WORKER_EXIT_SENTINEL="TRANSFER_WORKER_EXIT="',
            controller,
            "the controller must read the sentinel out of the worker, not restate it",
        )
        self.assertIn("h200_worker.sh\" | head -1", controller.replace("\n", " ") + " ")

    def test_no_stage_source_prints_the_exit_sentinel(self):
        """The one fact that bounds the residual exposure invoke_worker accepts.

        The controller requires the sentinel to be UNIQUE, not to be the last
        line, because `kubectl exec` appends a trailer after the remote process's
        output and a position requirement mis-reported every genuine failure.
        Uniqueness is strictly weaker: a worker killed before its EXIT trap, plus
        a stage that had printed exactly one line beginning with the prefix,
        would be believed. invoke_worker's comment accepts that "because no stage
        prints this prefix today -- a fact a test asserts against the stage
        sources rather than something this comment assumes". No such test
        existed. The test above counts declarations in the worker, which is a
        different fact.
        """

        declaration = re.search(
            r'^WORKER_EXIT_SENTINEL="(.*)"$',
            WORKER.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
        self.assertIsNotNone(declaration, "the worker declares no sentinel")
        sentinel = declaration.group(1)
        sources = [
            *sorted(TRANSFER_DIR.glob("*.py")),
            *sorted((REPO_ROOT / "src" / "transfer").rglob("*.py")),
        ]
        self.assertTrue(sources, "no stage sources found; the check would be vacuous")
        offenders = [
            str(path.relative_to(REPO_ROOT))
            for path in sources
            if sentinel in path.read_text(encoding="utf-8")
        ]
        self.assertEqual(
            offenders,
            [],
            f"{sentinel!r} appears in a source the worker runs, so a line of stage "
            "output could be read as the campaign's exit status. Either stop "
            "printing it or make the controller require last-line position and "
            "accept the transport trailer some other way",
        )

    def test_a_nonzero_worker_status_fails_the_controller_though_the_transport_says_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            access = access_layer_stubs(
                tmp,
                # Exactly the observed behaviour: emit the worker's output,
                # including its sentinel, and then exit 0 regardless.
                "#!/usr/bin/env bash\n"
                "echo 'campaign INCOMPLETE: 1 requested stage(s) measured nothing'\n"
                "echo 'TRANSFER_WORKER_EXIT=1'\n"
                "exit 0\n",
            )
            result = run_controller_with_stubs(tmp, access)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("worker reported exit status 1", result.stdout + result.stderr)

    def test_a_missing_sentinel_is_itself_a_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            access = access_layer_stubs(
                tmp,
                # A worker killed mid-run, or a pod exec that never started:
                # output, no sentinel, and a transport that still says 0.
                "#!/usr/bin/env bash\necho 'partial output'\nexit 0\n",
            )
            result = run_controller_with_stubs(tmp, access)
            self.assertEqual(result.returncode, 90, result.stdout + result.stderr)
            self.assertIn("did not", result.stdout + result.stderr)

    def test_a_clean_worker_still_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            access = access_layer_stubs(
                tmp,
                "#!/usr/bin/env bash\necho 'TRANSFER_WORKER_EXIT=0'\nexit 0\n",
            )
            result = run_controller_with_stubs(tmp, access)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_a_transport_trailer_after_the_sentinel_is_tolerated(self):
        # kubectl exec appends "command terminated with exit code N" after the
        # remote process's output, so on every failing run the sentinel is
        # second-to-last. Requiring last-line position was tried and it turned a
        # correctly reported failure into "no sentinel" -- a true status traded for
        # a false one. Uniqueness is the invariant, not position.
        with tempfile.TemporaryDirectory() as tmp:
            access = access_layer_stubs(
                tmp,
                "#!/usr/bin/env bash\n"
                "echo 'TRANSFER_WORKER_EXIT=3'\n"
                "echo 'command terminated with exit code 3'\n"
                "exit 0\n",
            )
            result = run_controller_with_stubs(tmp, access)
            self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
            self.assertIn("worker reported exit status 3", result.stdout + result.stderr)

    def test_two_sentinels_are_refused_rather_than_resolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            access = access_layer_stubs(
                tmp,
                "#!/usr/bin/env bash\n"
                "echo 'TRANSFER_WORKER_EXIT=1'\n"
                "echo 'TRANSFER_WORKER_EXIT=0'\n"
                "exit 0\n",
            )
            result = run_controller_with_stubs(tmp, access)
            self.assertEqual(result.returncode, 91, result.stdout + result.stderr)
            self.assertIn("exactly one is expected", result.stdout + result.stderr)

    def test_a_non_numeric_status_is_not_a_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            access = access_layer_stubs(
                tmp,
                "#!/usr/bin/env bash\necho 'TRANSFER_WORKER_EXIT=ok'\nexit 0\n",
            )
            result = run_controller_with_stubs(tmp, access)
            self.assertEqual(result.returncode, 90, result.stdout + result.stderr)

    def test_a_trailing_blank_line_does_not_hide_the_sentinel(self):
        # tee and the access layer both add trailing whitespace; the sentinel is
        # the last non-empty line, not literally the last byte.
        with tempfile.TemporaryDirectory() as tmp:
            access = access_layer_stubs(
                tmp,
                "#!/usr/bin/env bash\nprintf 'TRANSFER_WORKER_EXIT=0\\n\\n   \\n'\nexit 0\n",
            )
            result = run_controller_with_stubs(tmp, access)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class ArmPathVariableIsDeclaredNotInferred(unittest.TestCase):
    """A contract that verifies here must verify inside the pod.

    The pod sets TRANSFER_TEXT_MODEL_BASE_DIR to TRANSFER_MODEL_BASE_DIR, because
    every checkpoint sits in one GPFS directory. The arm-to-variable map used to
    be recovered by comparing resolved paths, so under that alias six text arms
    classified as protein-root arms and the worker's own re-derivation refused the
    campaign -- correctly, and only because that check exists.
    """

    def test_the_contract_verifies_when_two_variables_alias(self):
        environment = dict(os.environ)
        environment.update(
            {
                "TRANSFER_MODEL_BASE_DIR": "/aliased/models",
                "TRANSFER_TEXT_MODEL_BASE_DIR": "/aliased/models",
                "TRANSFER_TEXT_MODEL_DIR": "/aliased/models/gpt2-large",
            }
        )
        result = subprocess.run(
            [sys.executable, str(TRANSFER_DIR / "panel_contract.py"), "--verify"],
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_every_arm_declares_its_variable(self):
        sys.path.insert(0, str(REPO_ROOT))
        from src.transfer.arms import PANEL

        import panel_contract as contract

        for arm, spec in PANEL.items():
            self.assertIn(spec.path_variable, contract.MODEL_PATH_VARIABLES, arm)
            self.assertEqual(contract.model_variable(arm), spec.path_variable, arm)


class ControllerLogsFollowTheProjectRoot(unittest.TestCase):
    """A synthetic campaign record must not land in an operational log directory.

    `CONTROLLER_LOG_DIR="${PROJECT_ROOT}/logs/transfer_h200_controller"` and
    `invoke_worker` writes `${CONTROLLER_LOG_DIR}/${RUN_ID}.log`, so a test that
    overrides `GPFS_*` and leaves `PROJECT_ROOT` at its default files its stub
    worker's output beside real campaigns -- under a run-id carrying the real
    code hash, which makes it indistinguishable from one by filename. Measured
    on this host: 189 of the 294 files in that directory were this suite's, and
    one of them recorded a `campaign INCOMPLETE` verdict that no campaign ever
    produced. The existing files are kept and reported, not deleted (Appendix B
    rule 18); what this test holds is that no new one is created.
    """

    def test_the_worker_log_is_written_under_project_root_not_the_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            access = access_layer_stubs(
                tmp, "#!/usr/bin/env bash\necho 'TRANSFER_WORKER_EXIT=0'\nexit 0\n"
            )
            result = run_controller_with_stubs(tmp, access)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            match = re.search(r"run_id=([0-9]{14}_[0-9a-f]{12}) code_hash=", result.stdout)
            self.assertIsNotNone(match, result.stdout)
            run_id = match.group(1)
            written = sorted(
                path.name
                for path in (
                    Path(tmp) / "project" / "logs" / "transfer_h200_controller"
                ).glob("*.log")
            )
            self.assertEqual(written, [f"{run_id}.log"])
            self.assertFalse(
                (
                    REPO_ROOT / "logs" / "transfer_h200_controller" / f"{run_id}.log"
                ).exists(),
                "this suite filed a synthetic campaign record in the checkout's "
                "operational controller log directory",
            )


class ClusterHealthIsDecidedOnStdout(unittest.TestCase):
    """The last controller decision that was still taken on an exit status.

    `if ! "${H200_STATUS_CHECK}"` cannot be true on this deployment. The health
    probe's final action is a pod check issued through the same `kubectl exec`
    path that L20 measured returning 0 for a command that exited 7, so under
    `set -e` the branch was unreachable and the gate could not refuse. CLAUDE.md
    and scripts/transfer/README.md name the probe's terminal `Health=` line as
    the authority and nothing read it. The blast radius was bounded -- the
    stdout-predicated checks that follow would have failed -- but a gate that
    cannot say no is the L20 shape surviving in the layer L20's repair claims to
    have closed.
    """

    def run_with_status(self, status_body, **overrides):
        with tempfile.TemporaryDirectory() as tmp:
            access = access_layer_stubs(
                tmp,
                "#!/usr/bin/env bash\necho 'TRANSFER_WORKER_EXIT=0'\nexit 0\n",
                status_body=status_body,
            )
            return run_controller_with_stubs(tmp, access, **overrides)

    def test_an_unhealthy_verdict_refuses_though_the_probe_exits_zero(self):
        result = self.run_with_status(
            "#!/usr/bin/env bash\necho 'Health=degraded'\nexit 0\n"
        )
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("Health=degraded", result.stderr)
        # Refused before anything crossed to GPFS.
        self.assertNotIn("pushing code snapshot", result.stdout)

    def test_a_probe_that_states_no_verdict_is_undecidable_not_healthy(self):
        # A probe killed mid-run, or one whose own transport died: output, no
        # verdict, exit 0. `/bin/true` is the degenerate case of this.
        result = self.run_with_status(
            "#!/usr/bin/env bash\necho 'tunnel up'\nexit 0\n"
        )
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("no Health= line", result.stderr)
        self.assertNotIn("pushing code snapshot", result.stdout)

    def test_the_verdict_and_not_the_exit_status_is_the_decision(self):
        # The negative path of the same rule: a probe that states ok and exits
        # non-zero still schedules, because on this transport the status it
        # carries is not the probe's own.
        result = self.run_with_status(
            "#!/usr/bin/env bash\nprintf 'Health=ok\\n'\nexit 7\n"
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Health=ok", result.stdout)

    def test_a_probe_that_hangs_is_bounded_and_reported_as_inconclusive(self):
        result = self.run_with_status(
            "#!/usr/bin/env bash\nsleep 30\nprintf 'Health=ok\\n'\n",
            H200_STATUS_TIMEOUT_SECONDS="1",
        )
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("INCONCLUSIVE", result.stderr)
        self.assertNotIn("pushing code snapshot", result.stdout)

    def test_the_default_bound_clears_the_documented_floor(self):
        # CLAUDE.md and scripts/transfer/README.md both require at least 90 s;
        # the call site had no caller-side bound at all.
        match = re.search(
            r'H200_STATUS_TIMEOUT_SECONDS="\$\{H200_STATUS_TIMEOUT_SECONDS:-(\d+)\}"',
            CONTROLLER.read_text(encoding="utf-8"),
        )
        self.assertIsNotNone(match, "the controller declares no health-probe timeout")
        self.assertGreaterEqual(int(match.group(1)), 90)


class EveryArmDispatchedStageDeclaresWhatItMeasured(unittest.TestCase):
    """An artefact must say which eligible arms its invocation left out.

    `panel_contract.stage_contract_record` is that declaration -- the arms
    measured, the arms eligible and not measured, the module refusals, and the
    cohort band beside the band the arms were qualified on -- and it is worth
    nothing in a stage that does not write it. `10_homology_control.py` wrote
    `panel_summary.json` with no such record, so a narrowed run of a panel-scoped
    stage and a full-panel run produced artefacts a reader cannot tell apart.
    That is L18 at the artefact level, in a stage whose declared scope is
    panel-wide.

    Generalised over the contract rather than written stage by stage, for the
    reason `tests/test_transfer_gap_contract.py` generalises its TG counterpart:
    a stage added to STAGE_CONTRACTS is covered without this file being edited,
    and a stage that starts writing the record cannot quietly stop.

    Armless stages are outside the scope: they dispatch no arm, so there is no
    arm selection for an invocation to narrow. Every other stage is in it,
    including those whose repairs are in flight -- this fails on them until those
    land, and that failure is the finding rather than a defect here.
    """

    def test_every_arm_dispatched_stage_writes_its_contract_record(self):
        missing = []
        for stage, contract in pc.STAGE_CONTRACTS.items():
            if contract.scope == "armless":
                continue
            source = (TRANSFER_DIR / contract.entry_point).read_text(encoding="utf-8")
            # Tolerant of line wrapping: a formatter may put the stage name on
            # the next line, and a source-text literal that a reformat can break
            # would be repaired by reformatting the source to suit the test --
            # the wrong direction. The check is still that this stage names
            # ITSELF, which is the fact worth holding.
            if not re.search(
                r'stage_contract_record\(\s*"' + re.escape(stage) + r'"', source
            ):
                missing.append(contract.entry_point)
        self.assertEqual(
            missing,
            [],
            "these stages write measurement artefacts without declaring which "
            "eligible arms the invocation omitted, so a narrowed run is "
            "indistinguishable from a full-panel one",
        )


class TheWorkerInvocationIsConstructedOnce(unittest.TestCase):
    """--dry-run must print the command the real path sends, not a second copy of it.

    The controller used to build POD_COMMAND for the dry run and re-list every
    flag inside invoke_worker. Nothing held the two in agreement, so a flag added
    to one and not the other makes --dry-run describe a campaign that does not
    happen -- a checking tool reporting something other than what runs, which is
    the L20 failure class one layer up.
    """

    def test_invoke_worker_sends_pod_command_rather_than_relisting_its_flags(self):
        source = extract_function(CONTROLLER, "invoke_worker")
        self.assertIn('"${POD_COMMAND[@]}"', source)
        relisted = [
            flag
            for flag in ("--run-id", "--snapshot-dir", "--results-root", "--arms", "--stages")
            if flag in source
        ]
        self.assertEqual(
            relisted,
            [],
            "these flags are listed a second time inside invoke_worker; the "
            "invocation is declared once, as POD_COMMAND, and --dry-run prints "
            "that same array",
        )

    def test_the_dry_run_prints_the_array_it_would_send(self):
        source = CONTROLLER.read_text(encoding="utf-8")
        self.assertIn('log "[dry-run]   ${POD_COMMAND[*]}"', source)


class ArmsForItemDispatchesOnDeclaredScope(unittest.TestCase):
    """The arms a (stage, item) pair touches are decided by the stage's declared
    SCOPE, not by its name.

    verify_commands_buildable decided the same question from a name list once, and
    a stage absent from that list fell through to a catch-all and was built with
    the literal item "panel" as though it were an arm. That was repaired to read
    TRANSFER_STAGE_SCOPE and arms_for_item was left with the identical shape --
    which matters because its consumer is the data-path preflight: a panel-wide
    stage misread as per-arm resolves model variables for an arm called "panel",
    finds none, and passes having checked nothing.
    """

    def run_dispatch(self, stage: str, item: str, *, scope: str | None = None) -> subprocess.CompletedProcess[str]:
        functions = worker_functions("arms_for_item")
        override = f"TRANSFER_STAGE_SCOPE['{stage}']='{scope}'" if scope is not None else ""
        script = f"""
        set -uo pipefail
        source {pc.SHELL_CONTRACT}
        declare -A COHORT_ITEM_ARMS_FOR=([text]='gpt2 gpt2-large')
        declare -A STAGE_ARMS_FOR=([circuit_primitives]='gpt2-large protgpt2')
        {override}
        {functions}
        arms_for_item {stage} {item}
        """
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True)

    def test_a_per_arm_stage_touches_the_item_itself(self):
        result = self.run_dispatch("pathway_budget", "protgpt2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split(), ["protgpt2"])

    def test_a_panel_wide_stage_touches_its_contract_arm_list_not_the_item(self):
        result = self.run_dispatch("circuit_primitives", "panel")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split(), ["gpt2-large", "protgpt2"])
        self.assertNotIn("panel", result.stdout.split())

    def test_an_armless_stage_touches_nothing(self):
        result = self.run_dispatch("explanation_channel", "panel")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "")

    def test_cohort_power_uses_its_declared_item_space(self):
        # Panel-wide, yet its items are neither "panel" nor arm names.
        result = self.run_dispatch("cohort_power", "text")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split(), ["gpt2", "gpt2-large"])

    def test_a_stage_with_no_declared_scope_is_refused_rather_than_guessed(self):
        result = self.run_dispatch("pathway_budget", "protgpt2", scope="")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("no declared scope", result.stderr)

    def test_the_dispatch_is_not_a_hand_maintained_stage_name_list(self):
        source = extract_function(WORKER, "arms_for_item")
        self.assertIn("TRANSFER_STAGE_SCOPE", source)
        named = [
            stage
            for stage in pc.STAGE_CONTRACTS
            if stage != "cohort_power" and f"{stage})" in source
        ]
        self.assertEqual(
            named,
            [],
            "these stages are dispatched by name; a stage's arm shape is its "
            "declared scope, and cohort_power is the one exception because its "
            "item space is declared separately as TRANSFER_COHORT_ITEM_ARMS",
        )


class ExternalBaselineDispatchTests(unittest.TestCase):
    """The four ways an external-baseline dispatch has silently reported success.

    Each of these happened. A stale snapshot ran code that predated a flag the
    caller passed; the launcher reported LAUNCHED over four stages that had
    already died; a bare ``wait`` printed "campaign complete" over a lane whose
    artefact was never pulled; and the access layer under all of it returns 0 for
    a remote command that exited non-zero.
    """

    DRIVER = TRANSFER_DIR / "run_external_baseline_h200.sh"

    def test_the_driver_refuses_a_run_id_minted_from_different_code(self):
        """The defect that produced four dead launches in one session."""

        result = subprocess.run(
            [
                "bash", str(self.DRIVER),
                "--run-id", "20260101000000_deadbeefcafe",
                "--snapshot-dir", "/gpfs/nowhere/packages/20260101000000_deadbeefcafe",
                "--stage", "15_replacement_faithfulness.py",
                "--label", "stale", "--gpu", "0",
            ],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
            env={**os.environ, "H200_POD": "unused", "H200_POD_BASH": str(LOCAL_POD_BASH)},
            timeout=600,
        )
        self.assertNotEqual(result.returncode, 0, "a stale run-id must not be adopted")
        self.assertIn("minted from different code", result.stderr)

    def test_the_controller_prints_a_code_hash_without_touching_the_pod(self):
        """The reuse check has to be answerable offline, or it will be skipped."""

        result = subprocess.run(
            ["bash", str(CONTROLLER), "--print-code-hash"],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
            env={**os.environ, "H200_POD": "unused",
                 "H200_ACCESS_ROOT": "/nonexistent-access-root"},
            timeout=600,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        printed = re.search(r"^CODE_HASH=([0-9a-f]{64})$", result.stdout, re.M)
        self.assertIsNotNone(printed, f"no CODE_HASH on stdout: {result.stdout[-500:]}")

    def test_run_id_and_snapshot_dir_must_be_given_together(self):
        """One alone writes this run's results beside another run's code."""

        result = subprocess.run(
            ["bash", str(self.DRIVER), "--run-id", "20260101000000_deadbeefcafe",
             "--stage", "15_replacement_faithfulness.py", "--label", "x", "--gpu", "0"],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
            env={**os.environ, "H200_POD": "unused"}, timeout=600,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be given together", result.stderr)

    def _dispatch_fixture(self) -> tuple[Path, str, Path, Path]:
        """A project root the driver will accept, with a reusable snapshot and out-dir.

        Shared by the poll tests and the dead-dispatch test below, because the
        run-id has to carry the *current* code hash or the driver refuses the
        reuse before it reaches the behaviour under test.
        """

        root = Path(tempfile.mkdtemp(prefix="expect_"))
        run_id_hash = subprocess.run(
            ["bash", str(CONTROLLER), "--print-code-hash"],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
            env={**os.environ, "H200_POD": "unused",
                 "H200_ACCESS_ROOT": "/nonexistent-access-root"},
            timeout=600,
        ).stdout
        code_hash = re.search(r"^CODE_HASH=([0-9a-f]{64})$", run_id_hash, re.M).group(1)
        run_id = f"20260101000000_{code_hash[:12]}"

        snapshot = root / "packages" / run_id
        (snapshot / "scripts" / "transfer").mkdir(parents=True)
        (snapshot / "scripts" / "transfer" / "20_retrieval_bound.py").write_text("", encoding="utf-8")
        out_dir = root / "results" / "external_baseline" / run_id / "score"
        out_dir.mkdir(parents=True)
        return root, run_id, snapshot, out_dir

    def test_a_stage_that_dies_at_startup_cannot_be_reported_as_success(self):
        """The invariant a swallowed exit status turned into a false success.

        2026-08-11: an ad-hoc driver dispatched the protein-mode diffing cell,
        this entry point detected the death and exited 6, and the caller reported
        ``controller exited 0`` because it read ``$?`` after a ``$(date -Is)``
        that had already reset it. The cell was recorded as complete and no
        measurement existed. Whatever a caller then does with the status, the
        sanctioned entry point must not be the place the failure is lost: it has
        to exit non-zero and say so on its own output.
        """

        root, run_id, snapshot, _ = self._dispatch_fixture()

        # The pod-side log the stage would have written before raising. The
        # liveness check reads it through H200_POD_BASH, so it must sit at the
        # path the driver derives from GPFS_PROJECT_ROOT.
        pod_log = root / "logs" / "external_baseline" / f"{run_id}_score.log"
        pod_log.parent.mkdir(parents=True, exist_ok=True)
        pod_log.write_text(
            "[cohort] drawing\n"
            "Traceback (most recent call last):\n"
            '  File "20_retrieval_bound.py", line 1, in <module>\n'
            "RuntimeError: the corpus carries duplicate records in this band\n",
            encoding="utf-8",
        )

        pod_exec = root / "pod_exec.sh"
        pod_exec.write_text('#!/usr/bin/env bash\necho LAUNCHED\n', encoding="utf-8")
        pod_exec.chmod(0o755)

        result = subprocess.run(
            ["bash", str(self.DRIVER), "--run-id", run_id,
             "--snapshot-dir", str(snapshot), "--stage", "20_retrieval_bound.py",
             "--label", "score", "--gpu", "0"],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
            env={**os.environ, "H200_POD": "unused",
                 "H200_POD_BASH": str(LOCAL_POD_BASH), "H200_POD_EXEC": str(pod_exec),
                 "GPFS_PROJECT_ROOT": str(root), "LOCAL_OUTPUT_ROOT": str(root),
                 "LIVENESS_SETTLE_SECONDS": "0", "GRACE_SECONDS": "0",
                 "POLL_SECONDS": "1", "TIMEOUT_SECONDS": "3"},
            timeout=600,
        )

        self.assertNotEqual(
            result.returncode, 0,
            "a stage that died at start-up exited 0; a caller reading the status "
            f"would record it as a measurement.\n{result.stdout[-800:]}",
        )
        self.assertIn("DIED AT DISPATCH", result.stdout)
        self.assertNotIn(
            "ADMITTED", result.stdout,
            "a dead dispatch must not reach the pull, let alone be admitted",
        )
        pulled = root / "results" / "transfer" / "external_baseline" / run_id / "score"
        self.assertFalse(
            pulled.exists(),
            f"a dead dispatch left a pulled result directory behind: {pulled}",
        )

    def _poll_verdict(self, staged: str, expect: str | None) -> str:
        """Run one dispatch to the point of its poll verdict and return it.

        The pull is never reached: this exercises the completion test alone,
        which is the part that decides whether a directory is finished.
        """

        root, run_id, snapshot, out_dir = self._dispatch_fixture()
        (out_dir / staged).write_text("{}", encoding="utf-8")

        pod_exec = root / "pod_exec.sh"
        pod_exec.write_text('#!/usr/bin/env bash\necho LAUNCHED\n', encoding="utf-8")
        pod_exec.chmod(0o755)

        command = [
            "bash", str(self.DRIVER), "--run-id", run_id,
            "--snapshot-dir", str(snapshot), "--stage", "20_retrieval_bound.py",
            "--label", "score", "--gpu", "0",
        ]
        if expect is not None:
            command += ["--expect", expect]

        result = subprocess.run(
            command, capture_output=True, text=True, cwd=str(REPO_ROOT),
            env={**os.environ, "H200_POD": "unused",
                 "H200_POD_BASH": str(LOCAL_POD_BASH), "H200_POD_EXEC": str(pod_exec),
                 "GPFS_PROJECT_ROOT": str(root),
                 # Not REPO_ROOT: the driver reads the stage file and computes the
                 # code hash from that, so it has to stay the real checkout.
                 "LOCAL_OUTPUT_ROOT": str(root),
                 "LIVENESS_SETTLE_SECONDS": "0", "GRACE_SECONDS": "0",
                 "POLL_SECONDS": "1", "TIMEOUT_SECONDS": "3"},
            timeout=600,
        )
        verdict = re.search(r"score (PRESENT|ABSENT|UNRESOLVED) after", result.stdout)
        self.assertIsNotNone(verdict, f"no poll verdict on stdout: {result.stdout[-800:]}")
        self._dispatch_root = root
        return verdict.group(1)

    def test_a_dispatch_under_test_does_not_write_the_operational_record(self):
        """A test run must not leave a dispatch record for a run that never happened.

        The driver writes `logs/external_baseline/<run-id>_<label>.dispatch`
        before launching, and that directory is what an operator reads to see
        what was actually dispatched. It was keyed on REPO_ROOT, which a test
        cannot move -- REPO_ROOT is where the stage file and the code hash are
        read from -- so exercising the dispatch path wrote into the live log
        twice. The location is now its own declaration; this pins it.
        """

        live = REPO_ROOT / "logs" / "external_baseline"
        before = set(live.glob("*.dispatch")) if live.exists() else set()
        self._poll_verdict(staged="wildtypes.json", expect="score.json")
        after = set(live.glob("*.dispatch")) if live.exists() else set()
        self.assertEqual(
            after - before, set(),
            "a dispatch under test wrote into the operational log directory",
        )
        written = list((self._dispatch_root / "logs" / "external_baseline").glob("*.dispatch"))
        self.assertEqual(
            len(written), 1,
            f"the dispatch record did not land under LOCAL_OUTPUT_ROOT: {written}",
        )

    def test_a_staged_input_does_not_read_as_a_finished_measurement(self):
        """The false success --expect exists to prevent.

        20_retrieval_bound.py's score stage reads wildtypes.json from the
        directory it writes score.json into. Under the default completion test
        -- any .json in the output directory -- the staged input satisfies the
        poll on its first tick, so the controller pulls a directory holding no
        measurement and reports it ADMITTED.
        """

        self.assertEqual(
            self._poll_verdict(staged="wildtypes.json", expect=None), "PRESENT",
            "the default test no longer accepts any .json; this test's premise is stale",
        )
        self.assertEqual(
            self._poll_verdict(staged="wildtypes.json", expect="score.json"), "UNRESOLVED",
            "a staged input satisfied --expect; a partial pull would be admitted",
        )

    def test_expect_accepts_the_artefact_it_names(self):
        """The guard must still recognise a real completion, or it is a hang."""

        self.assertEqual(
            self._poll_verdict(staged="score.json", expect="score.json"), "PRESENT",
        )

    def test_a_bare_wait_cannot_tell_a_failed_lane_from_a_successful_one(self):
        """Why the operator guide requires per-PID waiting.

        Pinned as an executable fact rather than left as advice, because the
        advice is what was ignored: a driver using bare ``wait`` reported a
        campaign complete while one lane had exited 7 with its artefact still on
        GPFS.
        """

        bare = subprocess.run(
            ["bash", "-c", "( exit 7 ) & ( exit 0 ) & wait; echo rc=$?"],
            capture_output=True, text=True, timeout=60,
        )
        self.assertIn("rc=0", bare.stdout, "bare wait no longer masks failure; update the guide")

        aggregated = subprocess.run(
            ["bash", "-c",
             "pids=(); ( exit 7 ) & pids+=($!); ( exit 0 ) & pids+=($!); "
             "failed=0; for p in \"${pids[@]}\"; do wait \"$p\" || failed=$((failed+1)); done; "
             "echo failed=$failed"],
            capture_output=True, text=True, timeout=60,
        )
        self.assertIn("failed=1", aggregated.stdout, "per-PID waiting must see the failure")

    def test_the_operator_guide_documents_the_states_that_mean_different_things(self):
        text = README.read_text(encoding="utf-8")
        for token in ("DIED AT DISPATCH", "ADMITTED", "ABSENT", "--print-code-hash"):
            self.assertIn(token, text, f"the operator guide does not document {token}")

    def test_every_external_baseline_stage_appears_in_the_research_plan(self):
        """The stage table fell behind by three stages before this test existed."""

        plan = (REPO_ROOT / "docs" / "RESEARCH_PLAN.md").read_text(encoding="utf-8")
        entry_points = sorted(
            p.name for p in TRANSFER_DIR.glob("[0-9][0-9]_*.py")
        )
        missing = [name for name in entry_points if f"`{name}`" not in plan]
        self.assertEqual(missing, [], f"stages absent from the measurement package table: {missing}")


if __name__ == "__main__":
    unittest.main()
