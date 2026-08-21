"""What the context-information re-analysis stage must do, including its refusals.

The fixtures here are real sidecars: synthetic protein records are tokenised,
reduced to the same per-record sufficient statistics ``01_cohort_power.py``
produces, and written through ``budget.write_power_records`` itself. Nothing is
mocked, so a test that passes is evidence about the file format the stage will
actually meet on disk rather than about a second format written in the test.

The load-bearing assertions are the negative ones. A re-analysis that quietly
drops a refused arm, quietly pairs two cohorts, or quietly resamples records
instead of near-duplicate groups still produces a complete-looking artefact full
of plausible intervals, so each of those is asserted against directly.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from src.transfer.arms import Cohort
from src.transfer.budget import (
    RecordStatistics,
    SparseCounts,
    write_power_records,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE_PATH = REPO_ROOT / "scripts" / "transfer" / "41_context_information_bootstrap.py"

RESIDUES = "ACDEFGHIKLMNPQRSTVWY"

#: Draws: 0.025 * 400 = 10 in each tail, which is the module's floor exactly.
DRAWS = 400


def _load_stage():
    spec = importlib.util.spec_from_file_location("_stage_41", STAGE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


stage = _load_stage()


# --------------------------------------------------------------------------- #
# Fixtures: synthetic cohorts written through the real sidecar writer
# --------------------------------------------------------------------------- #


def synthetic_records(
    seed: int, *, n_families: int, per_family: int, length: int, mutations: int
) -> list[str]:
    """Records with a declared near-duplicate structure.

    One family is one near-duplicate group: a founder and ``per_family - 1``
    point-mutated copies of it, which single-linkage 5-mer containment joins.
    Distinct families share nothing at 5-mer scale, because two independent draws
    of a 120-residue string share essentially none of their 3.2 million possible
    5-mers.
    """

    rng = np.random.default_rng(seed)
    records: list[str] = []
    for _ in range(n_families):
        founder = "".join(rng.choice(list(RESIDUES), size=length))
        records.append(founder)
        for _ in range(per_family - 1):
            letters = list(founder)
            for position in rng.choice(length, size=mutations, replace=False):
                letters[int(position)] = str(rng.choice(list(RESIDUES)))
            records.append("".join(letters))
    return records


def tokenise(record: str, *, unit: int, vocab_size: int) -> np.ndarray:
    """Residue ids, or paired-residue ids, inside the declared vocabulary."""

    indices = [RESIDUES.index(symbol) for symbol in record]
    if unit == 1:
        ids = indices
    else:
        ids = [a * len(RESIDUES) + b for a, b in zip(indices[::2], indices[1::2])]
    assert max(ids) < vocab_size
    return np.asarray(ids, dtype=np.int64)


def arm_statistics(
    name: str,
    cohort_records: list[str],
    reference_records: list[str],
    *,
    unit: int,
    vocab_size: int,
    information: float,
    smoothing: float = 1.0,
    empty_reference_rows: int = 0,
) -> RecordStatistics:
    """Per-record statistics whose context information is ``information`` by design.

    The model term is the record's cross-entropy under the smoothed reference
    unigram less ``information`` per token, so ``I`` recovers ``information`` up
    to the reference's own estimation error and the smoothing constant.
    """

    reference_blocks = [
        tokenise(record, unit=unit, vocab_size=vocab_size) for record in reference_records
    ]
    reference_blocks += [np.zeros(0, dtype=np.int64)] * empty_reference_rows
    counts = np.zeros(vocab_size, dtype=np.float64)
    for block in reference_blocks:
        counts += np.bincount(block, minlength=vocab_size)
    probabilities = (counts + smoothing) / (counts.sum() + smoothing * vocab_size)
    surprisal = -np.log(probabilities)

    cohort_blocks = [
        tokenise(record, unit=unit, vocab_size=vocab_size) for record in cohort_records
    ]
    return RecordStatistics(
        arm=name,
        vocab_size=vocab_size,
        record_index=np.arange(len(cohort_blocks), dtype=np.int64),
        clean_nll_sum=np.asarray(
            [float(surprisal[block].sum()) - information * block.size for block in cohort_blocks],
            dtype=np.float64,
        ),
        token_count=np.asarray([block.size for block in cohort_blocks], dtype=np.int64),
        n_symbols=np.asarray(
            [len(record) for record in cohort_records], dtype=np.int64
        ),
        target_counts=SparseCounts.from_records(cohort_blocks),
        reference_counts=SparseCounts.from_records(reference_blocks),
    )


def write_block(
    directory: Path,
    name: str,
    cohort_records: list[str],
    reference_records: list[str],
    *,
    arms: dict[str, dict],
    smoothing: float = 1.0,
    write_cohort_json: bool = True,
    write_reference_json: bool = False,
) -> dict[str, Path]:
    """One sidecar and its companions, named the way stage 01 names them."""

    cohort = Cohort(
        name=name,
        kind="protein",
        records=list(cohort_records),
        min_symbols=min(len(r) for r in cohort_records),
        max_symbols=max(len(r) for r in cohort_records),
    )
    reference = Cohort(
        name=f"{name}_reference",
        kind="protein",
        records=list(reference_records),
        min_symbols=min(len(r) for r in reference_records),
        max_symbols=max(len(r) for r in reference_records),
    )
    digest = cohort.digest
    directory.mkdir(parents=True, exist_ok=True)
    statistics = {
        arm: arm_statistics(
            arm, cohort_records, reference_records, smoothing=smoothing, **options
        )
        for arm, options in arms.items()
    }
    sidecar = directory / f"power_{name}_{digest[:12]}.records.npz"
    write_power_records(
        sidecar,
        statistics,
        cohort_digest=digest,
        reference_digest=reference.digest,
        smoothing=smoothing,
        seeds={"cohort_draw": 20260101},
        max_len=384,
    )
    paths = {"sidecar": sidecar}
    if write_cohort_json:
        cohort_json = directory / f"cohort_{name}_{digest[:12]}.json"
        cohort_json.write_text(
            json.dumps(
                {
                    "schema_version": "r2_transfer_cohort_power_v1",
                    "artifact": "frozen_cohort",
                    "cohort_digest": digest,
                    "cohort_name": name,
                    "cohort_kind": "protein",
                    "min_symbols": cohort.min_symbols,
                    "max_symbols": cohort.max_symbols,
                    "n_records": len(cohort_records),
                    "records": list(cohort_records),
                    "metadata": {},
                }
            ),
            encoding="utf-8",
        )
        paths["cohort_json"] = cohort_json
    if write_reference_json:
        reference_json = directory / f"reference_{name}_{digest[:12]}.json"
        reference_json.write_text(
            json.dumps({"records": list(reference_records)}), encoding="utf-8"
        )
        paths["reference_json"] = reference_json
    return paths


TWO_ARMS = {
    "progen2-base": {"unit": 1, "vocab_size": 32, "information": 0.55},
    "progen2-medium": {"unit": 1, "vocab_size": 32, "information": 0.50},
}


def standard_block(directory: Path, name: str, seed: int, **kwargs) -> dict[str, Path]:
    cohort = synthetic_records(seed, n_families=12, per_family=2, length=120, mutations=6)
    reference = synthetic_records(
        seed + 1000, n_families=40, per_family=2, length=200, mutations=10
    )
    return write_block(directory, name, cohort, reference, arms=dict(TWO_ARMS), **kwargs)


def run_stage(argv: list[str]) -> dict:
    stage.main(argv)
    out = Path(argv[argv.index("--out") + 1])
    name = argv[argv.index("--report-name") + 1]
    return json.loads((out / name).read_text(encoding="utf-8"))


def base_argv(paths: dict[str, Path], out: Path, **extra) -> list[str]:
    argv = [
        "--sidecar",
        str(paths["sidecar"]),
        "--n-bootstrap",
        str(DRAWS),
        "--seed",
        "31337",
        "--out",
        str(out),
        "--report-name",
        "report.json",
    ]
    if "reference_json" in paths:
        argv += ["--reference-json", str(paths["reference_json"])]
    for flag, value in extra.items():
        argv += [f"--{flag.replace('_', '-')}", *[str(v) for v in value]]
    return argv


def row_for(payload: dict, arm: str, block_id: str = "b0") -> dict:
    matches = [
        row
        for row in payload["arm_results"]
        if row["arm"] == arm and row["block_id"] == block_id
    ]
    assert len(matches) == 1, f"{arm} appears {len(matches)} times in {block_id}"
    return matches[0]


# --------------------------------------------------------------------------- #
# 1. The happy path exists, and says what it measured
# --------------------------------------------------------------------------- #


def test_a_well_grouped_cohort_produces_a_group_level_interval(tmp_path: Path) -> None:
    paths = standard_block(tmp_path / "block", "swissprot", seed=1)
    payload = run_stage(base_argv(paths, tmp_path / "out"))

    row = row_for(payload, "progen2-base")
    assert row["refused"] is False
    assert row["resampling_unit"] == "group"
    assert row["cohort_draw_shared_between_terms"] is True
    assert row["reference_resampled"] is True
    assert row["context_information_nats"] == pytest.approx(0.55, abs=0.12)
    low, high = row["bootstrap_ci_95"]
    assert low < row["context_information_nats"] < high
    # The grouping is the near-duplicate one, not one record per unit.
    assert payload["blocks"][0]["cohort_grouping"]["available"] is True
    assert row["n_groups"] < row["n_scored_records"]
    for field in (
        "bootstrap_fraction_nonpositive",
        "bootstrap_bias",
        "median_bias_z0",
        "interval_mc_se",
        "relative_information",
        "relative_information_ci_95",
        "information_bits_per_symbol",
        "information_bits_per_symbol_ci_95",
        "n_scored_records",
        "n_scored_tokens",
        "n_reference_records",
        "n_reference_tokens",
        "n_effective_groups",
        "largest_group_token_share",
        "n_singleton_groups",
        "top10_record_token_share",
        "cohort_token_share_unseen_in_reference",
        "seed",
        "n_bootstrap",
    ):
        assert field in row, field
        assert row[field] is not None, field


def test_the_sign_status_is_marked_non_evidential(tmp_path: Path) -> None:
    """It is expected to pass on every arm, so it must never read as a gate."""

    paths = standard_block(tmp_path / "block", "swissprot", seed=2)
    payload = run_stage(base_argv(paths, tmp_path / "out"))

    row = row_for(payload, "progen2-base")
    assert row["sign_status"] == "PASS"
    assert row["sign_status_is_evidential"] is False
    assert "NON-EVIDENTIAL" in row["sign_status_note"]
    assert payload["summary"]["conclusion"]["sign_rule_is_adopted"] is False


# --------------------------------------------------------------------------- #
# 2. Refusals propagate; they are never dropped
# --------------------------------------------------------------------------- #


def test_one_dominant_near_duplicate_group_refuses_the_interval(tmp_path: Path) -> None:
    """A cohort that chains into one component has one effective unit, not many."""

    cohort = synthetic_records(3, n_families=1, per_family=24, length=120, mutations=4)
    reference = synthetic_records(
        4, n_families=40, per_family=2, length=200, mutations=10
    )
    paths = write_block(
        tmp_path / "block", "swissprot", cohort, reference, arms=dict(TWO_ARMS)
    )
    payload = run_stage(base_argv(paths, tmp_path / "out"))

    row = row_for(payload, "progen2-base")
    assert row["refused"] is True
    assert row["n_groups"] == 1
    assert row["n_effective_groups"] < 8
    assert "below the 8-unit floor" in row["refusal_reason"]
    assert row["context_information_nats"] is None
    assert row["legacy_threshold_status"] == "REFUSED"
    # The arm is still in the table, and the refusal reaches the contrast.
    assert row["arm"] in {entry["arm"] for entry in payload["summary"]["arms"]}
    assert "progen2-base" in payload["summary"]["arms_with_a_refused_interval"]
    contrasts = payload["contrasts"]["within_cohort_paired"]
    assert contrasts and all(entry["refused"] for entry in contrasts)
    assert all(entry["statistics"] is None for entry in contrasts)


def test_an_arm_without_reference_counts_is_refused_rather_than_skipped(
    tmp_path: Path,
) -> None:
    cohort = synthetic_records(5, n_families=12, per_family=2, length=120, mutations=6)
    reference = synthetic_records(
        6, n_families=40, per_family=2, length=200, mutations=10
    )
    statistics = arm_statistics(
        "progen2-base", cohort, reference, unit=1, vocab_size=32, information=0.5
    )
    directory = tmp_path / "block"
    directory.mkdir(parents=True)
    frozen = Cohort(
        name="swissprot",
        kind="protein",
        records=cohort,
        min_symbols=min(len(r) for r in cohort),
        max_symbols=max(len(r) for r in cohort),
    )
    sidecar = directory / f"power_swissprot_{frozen.digest[:12]}.records.npz"
    write_power_records(
        sidecar,
        # reference_counts stripped: the held-out baseline never entered this run
        {"progen2-base": RecordStatistics(
            arm=statistics.arm,
            vocab_size=statistics.vocab_size,
            record_index=statistics.record_index,
            clean_nll_sum=statistics.clean_nll_sum,
            token_count=statistics.token_count,
            n_symbols=statistics.n_symbols,
            target_counts=statistics.target_counts,
        )},
        cohort_digest=frozen.digest,
        reference_digest=None,
        smoothing=None,
        seeds={"cohort_draw": 1},
        max_len=384,
    )
    payload = run_stage(
        base_argv({"sidecar": sidecar}, tmp_path / "out")
    )

    row = row_for(payload, "progen2-base")
    assert row["refused"] is True
    assert "no reference counts" in row["refusal_reason"]
    assert payload["blocks"][0]["arms_refused_at_load"]["progen2-base"]


def test_a_missing_cohort_file_declares_the_singleton_fallback(tmp_path: Path) -> None:
    """Falling back to record-level resampling is allowed only out loud."""

    paths = standard_block(
        tmp_path / "block", "swissprot", seed=7, write_cohort_json=False
    )
    payload = run_stage(base_argv(paths, tmp_path / "out"))

    grouping = payload["blocks"][0]["cohort_grouping"]
    assert grouping["available"] is False
    assert "DECLARED LIMITATION" in grouping["declared_limitation"]
    assert grouping["fallback"].startswith("singleton groups")
    row = row_for(payload, "progen2-base")
    assert row["cohort_grouping"]["available"] is False
    assert row["n_groups"] == row["n_scored_records"]


# --------------------------------------------------------------------------- #
# 3. Pairing is defined within a cohort and nowhere else
# --------------------------------------------------------------------------- #


def test_arms_on_different_cohorts_are_never_paired(tmp_path: Path) -> None:
    first = standard_block(tmp_path / "one", "swissprot", seed=11)
    second = standard_block(tmp_path / "two", "zymctrl_ec", seed=99)
    payload = run_stage(
        [
            "--sidecar",
            str(first["sidecar"]),
            str(second["sidecar"]),
            "--n-bootstrap",
            str(DRAWS),
            "--seed",
            "31337",
            "--out",
            str(tmp_path / "out"),
            "--report-name",
            "report.json",
        ]
    )

    assert {block["cohort_digest"] for block in payload["blocks"]} == {
        payload["blocks"][0]["cohort_digest"],
        payload["blocks"][1]["cohort_digest"],
    }
    assert payload["blocks"][0]["cohort_digest"] != payload["blocks"][1]["cohort_digest"]

    # Every paired contrast lives inside one block.
    for entry in payload["contrasts"]["within_cohort_paired"]:
        assert entry["paired"] is True
        assert entry["common_resample_indices"] is True
        assert entry["block_id"] in {"b0", "b1"}

    across = payload["contrasts"]["cross_cohort_unpaired"]
    assert across, "the same arm on two cohort draws must be contrasted"
    for entry in across:
        assert entry["paired"] is False
        assert entry["common_resample_indices"] is False
        assert entry["left_block_id"] != entry["right_block_id"]
        assert "independence_note" in entry
    assert any(entry["same_arm_across_draws"] for entry in across)

    # The two panels must not share a seed, or the unpaired difference would be
    # neither independent nor paired.
    seeds = [panel["seed"] for block in payload["blocks"] for panel in block["panels"]]
    assert len(set(seeds)) == len(seeds)


def test_the_declared_progen2_contrast_is_named_and_its_sign_tracked(
    tmp_path: Path,
) -> None:
    first = standard_block(tmp_path / "one", "swissprot", seed=13)
    second = standard_block(tmp_path / "two", "swissprot_skip", seed=77)
    payload = run_stage(
        [
            "--sidecar",
            str(first["sidecar"]),
            str(second["sidecar"]),
            "--n-bootstrap",
            str(DRAWS),
            "--seed",
            "4242",
            "--out",
            str(tmp_path / "out"),
            "--report-name",
            "report.json",
        ]
    )

    declared = payload["contrasts"]["declared"]
    assert len(declared) == 1
    entry = declared[0]
    assert {entry["left"], entry["right"]} == {"progen2-base", "progen2-medium"}
    assert entry["ordering_may_be_reported"] is False
    assert "NOT SUPPORTED" in entry["preregistration_note"]
    # One paired reading per block, plus the unpaired cross-block pair.
    assert entry["n_readings"] >= 3
    assert any(reading["paired"] is False for reading in entry["readings"])
    assert entry["sign_is_stable_across_readings"] in (True, False)


# --------------------------------------------------------------------------- #
# 4. Determinism
# --------------------------------------------------------------------------- #


def _without_timestamps(payload: dict) -> dict:
    trimmed = json.loads(json.dumps(payload))
    trimmed["metadata"].pop("started_at_utc")
    trimmed["metadata"].pop("finished_at_utc")
    return trimmed


def test_a_fixed_seed_reproduces_the_report_bit_for_bit(tmp_path: Path) -> None:
    """Same inputs, same seed, same command: the same bytes but for the clock."""

    paths = standard_block(tmp_path / "block", "swissprot", seed=17)
    argv = base_argv(paths, tmp_path / "out")
    first = run_stage(argv)
    second = run_stage(argv)

    assert json.dumps(_without_timestamps(first), sort_keys=True) == json.dumps(
        _without_timestamps(second), sort_keys=True
    )
    assert first["metadata"]["runner_sha256"] == second["metadata"]["runner_sha256"]


# --------------------------------------------------------------------------- #
# 5. E4, the smoothing sweep
# --------------------------------------------------------------------------- #


def test_the_alpha_sweep_stays_inside_the_analytic_bound(tmp_path: Path) -> None:
    """``I(a) - I(0) <= log(1 + aV/R)`` exactly, and the bound grows with V.

    The bound is what the sweep is checked against rather than a monotonicity
    claim: a token whose reference count is below ``R/V`` has its surprisal
    *reduced* by smoothing, so ``I`` is not monotone in ``a`` in general. The
    inequality is, and it is the whole content of the withdrawn figures.
    """

    cohort = synthetic_records(19, n_families=12, per_family=2, length=120, mutations=6)
    reference = synthetic_records(
        20, n_families=40, per_family=2, length=200, mutations=10
    )
    paths = write_block(
        tmp_path / "block",
        "swissprot",
        cohort,
        reference,
        arms={
            "progen2-base": {"unit": 1, "vocab_size": 32, "information": 0.5},
            "protgpt2": {"unit": 2, "vocab_size": 512, "information": 0.5},
        },
    )
    payload = run_stage(base_argv(paths, tmp_path / "out"))

    records = {entry["arm"]: entry for entry in payload["alpha_sensitivity"]["records"]}
    assert set(records) == {"progen2-base", "protgpt2"}
    for arm, entry in records.items():
        assert entry["unsmoothed_information_nats"] is not None, arm
        assert entry["unsmoothed_unavailable_reason"] is None, arm
        for reading in entry["by_alpha"]:
            assert reading["within_analytic_bound"] is True, (arm, reading)
            assert reading["excess_over_unsmoothed_nats"] <= reading["analytic_bound_nats"]
        ascending = sorted(entry["by_alpha"], key=lambda r: r["alpha"])
        bounds = [reading["analytic_bound_nats"] for reading in ascending]
        assert bounds == sorted(bounds)
        assert bounds[0] < bounds[-1]
        for reading in ascending:
            expected = math.log1p(
                reading["alpha"] * entry["vocab_size"] / entry["reference_tokens"]
            )
            assert reading["analytic_bound_nats"] == pytest.approx(expected)

    # The bias is vocabulary-dependent: the wider inventory carries the larger
    # bound at every rung of the ladder, which is why it falls unequally.
    small = {r["alpha"]: r for r in records["progen2-base"]["by_alpha"]}
    large = {r["alpha"]: r for r in records["protgpt2"]["by_alpha"]}
    for alpha in small:
        assert large[alpha]["analytic_bound_nats"] > small[alpha]["analytic_bound_nats"]
    assert (
        records["protgpt2"]["information_range_nats"]
        > records["progen2-base"]["information_range_nats"]
    )


# --------------------------------------------------------------------------- #
# 6. E5, the leakage-removed sensitivity
# --------------------------------------------------------------------------- #


def test_the_leakage_sensitivity_is_unavailable_without_reference_records(
    tmp_path: Path,
) -> None:
    paths = standard_block(tmp_path / "block", "swissprot", seed=23)
    payload = run_stage(base_argv(paths, tmp_path / "out"))

    records = payload["leakage_removed_sensitivity"]["records"]
    assert records
    for entry in records:
        assert entry["available"] is False
        # The reason must name the input that would fix it, not merely say
        # the screen is off -- an operator reads this line to act on it.
        assert "--reference-json" in entry["reason"]
        assert entry["information_nats"] is None


def test_the_leakage_sensitivity_drops_the_leaked_reference_records(
    tmp_path: Path,
) -> None:
    cohort = synthetic_records(29, n_families=12, per_family=2, length=120, mutations=6)
    clean = synthetic_records(30, n_families=40, per_family=2, length=200, mutations=10)
    # Ten reference records that are outright copies of cohort records, which
    # exact-content disjointness would catch but 30-mer leakage is defined to
    # include, plus the clean block.
    reference = clean + [record + record for record in cohort[:10]]
    paths = write_block(
        tmp_path / "block",
        "swissprot",
        cohort,
        reference,
        arms=dict(TWO_ARMS),
        write_reference_json=True,
    )
    payload = run_stage(base_argv(paths, tmp_path / "out"))

    records = {entry["arm"]: entry for entry in payload["leakage_removed_sensitivity"]["records"]}
    entry = records["progen2-base"]
    assert entry["available"] is True
    assert entry["n_reference_records_dropped"] >= 10
    assert entry["information_nats"] is not None
    assert entry["delta_vs_headline_nats"] is not None
    # The reference grouping is the near-duplicate one once the records exist.
    assert payload["blocks"][0]["reference_grouping"]["available"] is True


# --------------------------------------------------------------------------- #
# 7. Between-block variance
# --------------------------------------------------------------------------- #


def test_two_blocks_report_a_spread_and_refuse_an_interval(tmp_path: Path) -> None:
    first = standard_block(tmp_path / "one", "swissprot", seed=31)
    second = standard_block(tmp_path / "two", "swissprot_skip", seed=131)
    payload = run_stage(
        [
            "--sidecar",
            str(first["sidecar"]),
            str(second["sidecar"]),
            "--n-bootstrap",
            str(DRAWS),
            "--seed",
            "555",
            "--out",
            str(tmp_path / "out"),
            "--report-name",
            "report.json",
        ]
    )

    records = {entry["arm"]: entry for entry in payload["between_block_variance"]}
    entry = records["progen2-base"]
    assert entry["n_blocks"] == 2
    assert entry["degrees_of_freedom"] == 1
    assert entry["between_block_sd_nats"] is not None
    assert entry["between_block_interval"] is None
    assert "not an interval" in entry["between_block_interval_refusal"]
    assert entry["folded_into_the_bootstrap_interval"] is False
    assert entry["blocks_disjoint"]["checked"] is True
    assert entry["blocks_disjoint"]["all_disjoint"] is True


# --------------------------------------------------------------------------- #
# 8. The summary table names what is absent
# --------------------------------------------------------------------------- #


def test_an_arm_with_no_record_is_named_absent_rather_than_omitted(
    tmp_path: Path,
) -> None:
    paths = standard_block(tmp_path / "block", "swissprot", seed=37)
    payload = run_stage(base_argv(paths, tmp_path / "out"))

    entries = {entry["arm"]: entry for entry in payload["summary"]["arms"]}
    # Every declared panel arm is accounted for, measured or not.
    assert {"bygpt5-small-en", "bygpt5-base-en", "gpt2-large"} <= set(entries)
    absent = entries["bygpt5-small-en"]
    assert absent["present"] is False
    assert absent["status"] == "ABSENT"
    assert "not a pass" in absent["absent_reason"]
    assert "bygpt5-small-en" in payload["summary"]["arms_with_no_record"]
    assert entries["progen2-base"]["present"] is True
    assert entries["progen2-base"]["status"] == "PASS"
    conclusion = payload["summary"]["conclusion"]
    assert conclusion["sign_rule_is_adopted"] is False
    assert isinstance(conclusion["interval_alters_the_unmeasurable_set"], bool)
    assert isinstance(conclusion["interval_statement"], str)


def test_eight_blocks_earn_a_between_block_interval_that_is_still_separate(
    tmp_path: Path,
) -> None:
    """K = 8 is where the component stops being a one-degree-of-freedom quantity.

    It is still reported beside the bootstrap interval and never inside it: a
    variance component estimated from K blocks does not belong in a percentile
    interval computed from one.
    """

    blocks = [
        standard_block(tmp_path / f"block{index}", f"swissprot_{index}", seed=200 + index)
        for index in range(8)
    ]
    argv = ["--sidecar", *[str(block["sidecar"]) for block in blocks]]
    argv += [
        "--n-bootstrap",
        str(DRAWS),
        "--seed",
        "909",
        "--alpha-sweep",
        "1.0",
        "--out",
        str(tmp_path / "out"),
        "--report-name",
        "report.json",
    ]
    payload = run_stage(argv)

    entry = {row["arm"]: row for row in payload["between_block_variance"]}["progen2-base"]
    assert entry["n_blocks"] == 8
    assert entry["degrees_of_freedom"] == 7
    assert entry["between_block_interval_refusal"] is None
    low, high = entry["between_block_interval"]
    assert low < entry["mean_nats"] < high
    assert entry["folded_into_the_bootstrap_interval"] is False
    assert entry["blocks_disjoint"]["all_disjoint"] is True
