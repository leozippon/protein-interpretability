"""End-to-end contracts for strict TG-99 campaign summarization."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE_DIR = REPO_ROOT / "scripts" / "transfer_gap"
for _path in (str(REPO_ROOT), str(STAGE_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import tg99_summarize as tg99  # noqa: E402
from tg_common import TG_PANEL  # noqa: E402
from tg_contract import TG_STAGES  # noqa: E402

SCRIPT = STAGE_DIR / "tg99_summarize.py"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def tg03_payload(arm: str, *, layer: int = 18, k: int = 32) -> dict[str, Any]:
    return {
        "arm": arm,
        "layer": layer,
        "n_layer": 36,
        "d_model": 1280,
        "d_sae": 10240,
        "k": k,
        "seed": 20260729,
        "train_tokens": 4_000_000,
        "eval_tokens": 400_000,
        "steps": 4_000,
        "train_cohort": {"selection_digest": "train"},
        "eval_cohort": {"selection_digest": "eval"},
        "fvu": 0.25,
        "dead_fraction": 0.1,
        "loss_recovered": 0.5,
        "ce_delta_nats": 0.1,
        "ce_clean_nats": 1.0,
        "denominator_valid": True,
        "feature_variance_explained_by_current_token": {"mean": 0.2},
        "feature_variance_explained_by_position": {"mean": 0.1},
    }


def stage_payload(stage: str, arm: str) -> dict[str, Any]:
    if stage == "tg01":
        return {
            "arm": arm,
            "symbols_per_token": 1.0,
            "unigram_entropy_nats": 2.0,
            "model_nll_nats": 1.0,
            "info_gain_over_unigram_bits": 1.4,
            "info_gain_bits_per_symbol": 1.4,
            "fraction_of_unigram_entropy_explained": 0.5,
            "top1_accuracy": 0.75,
            "local_fraction_within_8": 0.6,
            "gain_top_decile_share": 0.3,
            "gain_by_position_bin_nats": {"interior": 1.0},
        }
    if stage == "tg02":
        return {
            "arm": arm,
            "primary_shuffle": "within_sequence",
            "far_context_information_bits": 0.4,
            "within_sequence": {
                "far_order_information_bits": 0.2,
                "far_order_share": 0.5,
            },
        }
    if stage == "tg05":
        return {
            "arm": arm,
            "anchored_partner_auc": {
                "partner_marginal_only": 0.5,
                "single_concat": 0.6,
                "attention_pattern": 0.7,
            },
        }
    if stage == "tg06":
        return {
            "arm": arm,
            "transplant_cost_bits": 0.2,
            "uniform_cost_bits": 0.4,
            "transplant_cost_nats": 0.1,
            "ce_nats": {"clean": 1.0},
        }
    return {"arm": arm, "stage": stage}


def write_complete_campaign(root: Path) -> None:
    for stage, contract in TG_STAGES.items():
        if contract.scope == "summary":
            continue
        if contract.scope == "armless":
            write_json(root / stage / "explanation_channel.json", {"stage": stage})
            continue
        arms = contract.arms if contract.arms is not None else TG_PANEL
        for arm in arms:
            if stage == "tg03":
                layer = 13 if arm == "progen2-medium" else 18
                write_json(
                    root / stage / f"{arm}_L{layer}_k32.json",
                    tg03_payload(arm, layer=layer),
                )
            else:
                write_json(root / stage / f"{arm}.json", stage_payload(stage, arm))


def run_tg99(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def test_empty_campaign_fails_without_writing_summary(tmp_path: Path) -> None:
    root = tmp_path / "empty"

    result = run_tg99(root)

    assert result.returncode == 2
    assert "incomplete transfer-gap campaign" in result.stderr
    assert not (root / "SUMMARY.json").exists()


def test_partial_campaign_requires_opt_in_and_records_missing_matrix(
    tmp_path: Path,
) -> None:
    root = tmp_path / "partial"
    write_json(root / "tg01" / "gpt2-large.json", stage_payload("tg01", "gpt2-large"))

    refused = run_tg99(root)
    assert refused.returncode == 2
    assert not (root / "SUMMARY.json").exists()

    allowed = run_tg99(root, "--allow-partial")
    assert allowed.returncode == 0, allowed.stderr
    summary = json.loads((root / "SUMMARY.json").read_text(encoding="utf-8"))
    completeness = summary["completeness"]
    assert completeness["complete"] is False
    assert completeness["mode"] == "allow-partial"
    assert completeness["present_matrix"]["tg01"] == ["gpt2-large"]
    assert completeness["missing_matrix"]["tg01"] == [
        "protgpt2",
        "zymctrl",
        "progen2-medium",
    ]
    assert completeness["present_artifact_count"] == 1
    assert completeness["required_artifact_count"] == 39


def test_ambiguous_tg03_requires_stable_identity_selection(tmp_path: Path) -> None:
    root = tmp_path / "ambiguous"
    first = tg03_payload("gpt2-large", layer=18)
    second = tg03_payload("gpt2-large", layer=17)
    write_json(root / "tg03" / "gpt2-large_L18_k32.json", first)
    write_json(root / "tg03" / "gpt2-large_L17_k32.json", second)

    refused = run_tg99(root, "--allow-partial")
    assert refused.returncode == 2
    assert "ambiguous TG-03 configuration for gpt2-large" in refused.stderr
    assert not (root / "SUMMARY.json").exists()

    identity = tg99.tg03_stable_identity(first)
    selected = run_tg99(
        root,
        "--allow-partial",
        "--tg03-select",
        f"gpt2-large={identity}",
    )
    assert selected.returncode == 0, selected.stderr
    summary = json.loads((root / "SUMMARY.json").read_text(encoding="utf-8"))
    assert summary["tg03_selection"]["gpt2-large"] == {
        "identity": identity,
        "path": "tg03/gpt2-large_L18_k32.json",
    }
    assert summary["arms"]["gpt2-large"]["sae_layer"] == 18


def test_complete_campaign_is_accepted_with_contract_derived_matrix(
    tmp_path: Path,
) -> None:
    root = tmp_path / "complete"
    write_complete_campaign(root)

    result = run_tg99(root)

    assert result.returncode == 0, result.stderr
    summary = json.loads((root / "SUMMARY.json").read_text(encoding="utf-8"))
    completeness = summary["completeness"]
    assert completeness["complete"] is True
    assert completeness["mode"] == "strict"
    assert completeness["missing_matrix"] == {}
    assert completeness["present_artifact_count"] == 39
    assert (
        completeness["present_artifact_count"]
        == completeness["required_artifact_count"]
    )
    assert completeness["expected_matrix"]["tg00"] == [
        "protgpt2",
        "progen2-medium",
    ]
    assert set(completeness["expected_matrix"]) == {
        stage for stage, contract in TG_STAGES.items() if contract.scope != "summary"
    }
    assert set(summary["arms"]) == set(TG_PANEL)
    assert summary["arms"]["gpt2-large"]["sae_fvu"] == 0.25
    assert summary["explanation_channel"] == {"stage": "tg04"}
