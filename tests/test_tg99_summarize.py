"""End-to-end contracts for strict TG-99 campaign summarization.

Three of these encode corrections rather than the original behaviour:

* the expected matrix is 35 artefacts, not 39. TG-05 can produce one artefact of
  four and TG-06 three, and the contract now declares that instead of asking for
  four apiece from stages that refuse them -- strict mode was previously
  unsatisfiable by a *fully executed* campaign;
* TG-00's rendering and cohort deltas reach the summary. ``build_rows`` read
  TG-01, 02, 03, 05 and 06 and never the positive-control stage, so no
  ``SUMMARY.json`` has ever carried either delta;
* quantities combining two stages are refused when the stages measured
  different populations -- a different residue band, or a different scoring
  window over the same band. The second half was missing, and the test that
  asserted TG-01 over TG-06 was *not* refused was the test encoding the defect;
* an artefact that does not declare the contract it was produced under is
  refused rather than summarised, and the summary records the version its inputs
  declared instead of the version this code declares.
"""

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
from tg_contract import SCHEMA_VERSION, TG_STAGES, stage_contract_record  # noqa: E402

SCRIPT = STAGE_DIR / "tg99_summarize.py"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def tg03_payload(arm: str, *, layer: int = 18, k: int = 32) -> dict[str, Any]:
    return {
        "arm": arm,
        "contract": stage_contract_record("tg03", [arm]),
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
    """One stage artefact, carrying the contract block every stage now writes.

    Not decoration: ``inspect_campaign`` refuses an artefact that does not
    declare the contract this code enforces, and every artefact in the corrected
    tree except TG-01's fails that.
    """

    return {**_stage_body(stage, arm), "contract": stage_contract_record(stage, [arm])}


def _stage_body(stage: str, arm: str) -> dict[str, Any]:
    if stage == "tg00":
        # TG-00 is the positive-control stage; the summary read every other
        # stage and not it, so no SUMMARY.json ever carried a rendering or
        # cohort delta. A tg00 artefact missing these keys is a schema failure,
        # not a reason to skip the row.
        return {
            "arm": arm,
            "rendering_control": {
                "applicable": True,
                "rendering_delta_nats": 1.5,
                "wrong_control_token_delta_nats": 0.2,
            },
            "cohort_control": {"applicable": True, "cohort_delta_nats": 0.3},
        }
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
            write_json(
                root / stage / "explanation_channel.json",
                {"stage": stage, "contract": stage_contract_record(stage, [])},
            )
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
    # 35, not 39: TG-05 can produce one artefact and TG-06 three, and the
    # contract now says so instead of asking for four apiece from stages that
    # refuse them. Strict mode was unsatisfiable by a fully executed campaign.
    assert completeness["required_artifact_count"] == 35


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
    assert completeness["present_artifact_count"] == 35
    assert (
        completeness["present_artifact_count"]
        == completeness["required_artifact_count"]
    )
    assert completeness["expected_matrix"]["tg00"] == [
        "protgpt2",
        "progen2-medium",
    ]
    # The stages that refuse arms declare which ones, so a fully executed
    # campaign can satisfy strict mode. It could not before: these two expected
    # the whole four-arm panel from stages that raise on three and one of it.
    assert completeness["expected_matrix"]["tg05"] == ["progen2-medium"]
    assert completeness["expected_matrix"]["tg06"] == [
        "gpt2-large",
        "protgpt2",
        "zymctrl",
    ]
    assert set(completeness["expected_matrix"]) == {
        stage for stage, contract in TG_STAGES.items() if contract.scope != "summary"
    }
    assert set(summary["arms"]) == set(TG_PANEL)
    assert summary["arms"]["gpt2-large"]["sae_fvu"] == 0.25
    assert summary["explanation_channel"]["stage"] == "tg04"
    # Read off the artefacts, not stamped from this code's own constant.
    assert completeness["contract_schema_version"] == SCHEMA_VERSION


def test_the_positive_control_deltas_reach_the_summary(tmp_path: Path) -> None:
    """TG-00 prices the two defects that were each worth more than most of the
    effects measured on top of them, and ``build_rows`` read every stage but it."""

    root = tmp_path / "controls"
    write_complete_campaign(root)

    assert run_tg99(root).returncode == 0
    summary = json.loads((root / "SUMMARY.json").read_text(encoding="utf-8"))

    for arm in ("protgpt2", "progen2-medium"):
        row = summary["arms"][arm]
        assert row["rendering_delta_nats"] == 1.5
        assert row["cohort_delta_nats"] == 0.3
        assert row["wrong_control_token_delta_nats"] == 0.2
    # TG-00 is declared on two arms only, so the other two carry no control row
    # rather than a zero.
    assert "rendering_delta_nats" not in summary["arms"]["gpt2-large"]


def test_a_tg00_artefact_missing_its_controls_is_a_schema_failure(
    tmp_path: Path,
) -> None:
    root = tmp_path / "broken"
    write_complete_campaign(root)
    write_json(
        root / "tg00" / "protgpt2.json",
        {"arm": "protgpt2", "contract": stage_contract_record("tg00", ["protgpt2"])},
    )

    result = run_tg99(root)
    assert result.returncode == 2
    assert "incompatible stage artefact schema" in result.stderr
    assert not (root / "SUMMARY.json").exists()


def test_information_shares_are_refused_across_incommensurate_populations(
    tmp_path: Path,
) -> None:
    """Two stages measure one population only if they agree on both axes.

    TG-01 draws 400-1000 residues and TG-03 draws 120-1000: they share no protein
    below 400, and EXP-R2-060 prices protein cohort-block sensitivity at
    0.16-0.60 nats, larger than the 0.5-nat floor beside this division. That was
    checked. The scoring window was not: TG-01 scores positions 1..383 of every
    drawn sequence at ``--max-len 384`` and TG-03 positions 1..255 at 256, which
    is a second population difference, and one that applies to the text control
    as much as to a protein arm.
    """

    root = tmp_path / "populations"
    write_complete_campaign(root)

    assert run_tg99(root).returncode == 0
    summary = json.loads((root / "SUMMARY.json").read_text(encoding="utf-8"))

    for arm in ("protgpt2", "zymctrl", "progen2-medium"):
        row = summary["arms"][arm]
        assert row["sae_frac_information_lost"] is None
        refusal = row["sae_frac_information_lost_refusal"]
        assert "incommensurate protein cohort bands" in refusal
        assert "incommensurate scoring windows" in refusal
    # The text arm draws no protein band, so the band axis says nothing about it
    # -- which is exactly why the defect survived inspection. A token truncation
    # selects positions in text just as it does in protein, so the window axis
    # does. This assertion is the inverse of the one it replaces.
    text = summary["arms"]["gpt2-large"]
    assert text["sae_frac_information_lost"] is None
    assert "incommensurate scoring windows" in text["sae_frac_information_lost_refusal"]
    assert "cohort bands" not in text["sae_frac_information_lost_refusal"]


def test_the_attention_share_is_refused_across_the_window_it_crosses(
    tmp_path: Path,
) -> None:
    """The quantity that survived the band check and should not have.

    TG-01 and TG-06 share the 400-1000 residue band, so ``band_refusal`` licensed
    ``frac_information_from_attention_pattern`` on every arm the pair could
    produce. They do not share a scored population: TG-06 keeps only the
    sequences that reach 256 tokens and scores exactly their first 256 positions,
    while TG-01 scores every position of every drawn sequence up to 384.
    """

    root = tmp_path / "attention"
    write_complete_campaign(root)
    assert run_tg99(root).returncode == 0
    summary = json.loads((root / "SUMMARY.json").read_text(encoding="utf-8"))

    for arm in ("gpt2-large", "protgpt2", "zymctrl"):
        row = summary["arms"][arm]
        assert row["frac_information_from_attention_pattern"] is None
        assert "incommensurate scoring windows" in (
            row["frac_information_from_attention_pattern_refusal"]
        )


def test_two_stages_on_one_band_and_one_window_are_not_refused() -> None:
    """The negative path: the check must not refuse a commensurate pair.

    A guard that refused everything would satisfy every assertion above and
    measure nothing. TG-01 and TG-02 share both axes by declaration, and so do
    TG-03 and TG-07.
    """

    for arm in TG_PANEL:
        assert tg99.population_refusal(arm, "tg01", "tg02") is None, arm
        assert tg99.population_refusal(arm, "tg03", "tg07") is None, arm


def test_a_stage_that_declares_no_window_is_refused_rather_than_assumed() -> None:
    """TG-05 truncates nothing, which is not the same fact as truncating at 256."""

    refusal = tg99.population_refusal("progen2-medium", "tg03", "tg05")
    assert "undeclared" in refusal
    assert "incommensurate scoring windows" in refusal


def test_an_artefact_without_a_contract_block_is_refused(tmp_path: Path) -> None:
    """Fourteen of the eighteen artefacts in the corrected tree carry none.

    ``inspect_campaign`` validated "the file exists and parses as an object" and
    then recorded ``contract_schema_version`` from this code's own constant, so a
    strict ``complete: true`` summary could be assembled from a mixture of code
    generations and stamped with the current version.
    """

    root = tmp_path / "stale"
    write_complete_campaign(root)
    for arm in ("gpt2-large", "protgpt2"):
        payload = dict(stage_payload("tg01", arm))
        payload.pop("contract")
        write_json(root / "tg01" / f"{arm}.json", payload)

    result = run_tg99(root)

    assert result.returncode == 2
    # Every stale file is named, not just the first one found.
    assert "tg01/gpt2-large.json carries no `contract` block" in result.stderr
    assert "tg01/protgpt2.json carries no `contract` block" in result.stderr
    assert not (root / "SUMMARY.json").exists()


def test_an_artefact_from_a_superseded_contract_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "superseded"
    write_complete_campaign(root)
    payload = dict(stage_payload("tg06", "gpt2-large"))
    payload["contract"] = {**payload["contract"], "schema_version": "r2_transfer_gap_contract_v1"}
    write_json(root / "tg06" / "gpt2-large.json", payload)

    result = run_tg99(root)

    assert result.returncode == 2
    assert "declares contract schema 'r2_transfer_gap_contract_v1'" in result.stderr
    assert not (root / "SUMMARY.json").exists()


def test_the_refusal_survives_allow_partial(tmp_path: Path) -> None:
    """``--allow-partial`` licenses an incomplete campaign, not a stale one."""

    root = tmp_path / "partial_stale"
    write_json(root / "tg01" / "gpt2-large.json", _stage_body("tg01", "gpt2-large"))

    result = run_tg99(root, "--allow-partial")

    assert result.returncode == 2
    assert "carries no `contract` block" in result.stderr
    assert not (root / "SUMMARY.json").exists()


def test_a_partial_summary_announces_itself_on_stdout(tmp_path: Path) -> None:
    """It wrote the same SUMMARY.json and printed the same table as a strict run.

    The only thing separating the two was ``completeness.mode``, two levels down
    in a file nobody re-opens after reading the table.
    """

    root = tmp_path / "banner"
    write_json(root / "tg01" / "gpt2-large.json", stage_payload("tg01", "gpt2-large"))

    partial = run_tg99(root, "--allow-partial")
    assert partial.returncode == 0, partial.stderr
    first = partial.stdout.splitlines()[0]
    assert first.startswith("PARTIAL -- 1 of 35 artefacts, missing: ")
    assert "tg00: protgpt2, progen2-medium" in first

    complete = tmp_path / "complete"
    write_complete_campaign(complete)
    finished = run_tg99(complete)
    assert finished.returncode == 0, finished.stderr
    assert "PARTIAL" not in finished.stdout
