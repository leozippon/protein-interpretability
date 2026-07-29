from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSFER_DIR = REPO_ROOT / "scripts" / "transfer"
CONTROLLER = TRANSFER_DIR / "run_transfer_h200.sh"
WORKER = TRANSFER_DIR / "h200_worker.sh"


def extract_function(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    marker = f"\n{name}() {{\n"
    if marker not in source:
        raise AssertionError(f"{path.name} has no function {name}")
    body = source.split(marker, 1)[1]
    end = body.index("\n}\n")
    return f"{name}() {{\n{body[:end]}\n}}\n"


def controller_env(**overrides: str) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("ARGS_")}
    env.update(
        {
            "H200_POD": "test-pod",
            "H200_STATUS_CHECK": "/bin/true",
            "H200_SYNC": "/bin/true",
            "H200_GPFS_PUSH": "/bin/true",
            "H200_POD_BASH": "/bin/true",
            "H200_POD_EXEC": "/bin/true",
            "ARMS": "gpt2-large",
            "GPUS": "0",
            "STAGES": "cohort_power",
        }
    )
    env.update(overrides)
    return env


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


class MissingDataAggregationTests(unittest.TestCase):
    def test_background_skips_reach_parent_and_prevent_success(self):
        functions = "\n".join(
            extract_function(WORKER, name)
            for name in (
                "run_stage_wave",
                "finish_campaign",
            )
        )
        script = f"""
        set -euo pipefail
        SKIPPED_FOR_DATA=()
        DEFERRED_FAILURES=()
        SKIP_DATA_STATUS=75
        GPU_LIST=(0 1)
        RUN_ID=test_000000000000
        RESULTS_ROOT=/results
        LOGS_ROOT=/logs
        log() {{ printf '%s\n' "$*"; }}
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


class SnapshotContractTests(unittest.TestCase):
    def write_executable(self, path: Path, body: str) -> None:
        path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body, encoding="utf-8")
        path.chmod(0o755)

    def make_project_copy(self, root: Path) -> Path:
        project = root / "project"
        (project / "src").mkdir(parents=True)
        shutil.copy2(REPO_ROOT / "src" / "__init__.py", project / "src" / "__init__.py")
        shutil.copytree(REPO_ROOT / "src" / "transfer", project / "src" / "transfer")
        shutil.copytree(TRANSFER_DIR, project / "scripts" / "transfer")
        ladder = project / "docs" / "analysis" / "MODEL_LADDER_20260728.md"
        ladder.parent.mkdir(parents=True)
        shutil.copy2(REPO_ROOT / "docs" / "analysis" / ladder.name, ladder)
        return project

    def test_transfer_and_reuse_verify_complete_staged_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = self.make_project_copy(root)
            helpers = root / "helpers"
            helpers.mkdir()
            marker = root / "worker-invocations"
            self.write_executable(helpers / "status", ":\n")
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
                'printf "called\\n" >> "$POD_EXEC_MARKER"\n',
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
            self.assertIn(
                "docs/analysis/MODEL_LADDER_20260728.md",
                manifest.read_text(encoding="utf-8"),
            )
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


if __name__ == "__main__":
    unittest.main()
