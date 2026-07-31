"""The far-band patching estimand, and the declarations it depends on.

Before this file the entire far-band code path was untested: no test in ``tests/``
referenced ``activation_patching``, ``build_patch_cases``, ``DISTANCE_BANDS``,
``_case_resampled_interval``, ``MINIMUM_ELIGIBILITY_CLUSTERS``, ``_threshold_sweep``
or ``ELIGIBILITY_THRESHOLD_LADDER`` -- including the clustered bootstrap, the
eight-cluster floor and the repr-key collision guard whose own docstring says a
collision would make every invariant this sweep exists to provide read as
satisfied.  The far-band eligible fraction is the measurement a published
modality ordering rests on.

Written against the property rather than the repair, in the style of the rest of
this suite: what is asserted is that a wrong assumption is now either impossible
or loudly declared.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
STAGE_DIR = REPO_ROOT / "scripts" / "transfer"
if str(STAGE_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE_DIR))

import panel_contract as pc  # noqa: E402

from src.transfer.arms import PANEL, Arm, Cohort  # noqa: E402
from src.transfer.circuits import (  # noqa: E402
    CLOSED_CONTENT_FORMATS,
    CONDITIONING_TOKEN_SLACK,
    DISTANCE_BANDS,
    DISTANCE_UNITS,
    ELIGIBILITY_THRESHOLD_LADDER,
    MINIMUM_ELIGIBILITY_CLUSTERS,
    MINIMUM_PROBE_RECORD_RETENTION,
    PATCHING_REFUSED_DTYPES,
    SYMBOL_DISTANCE_BANDS,
    DistanceBandPlan,
    PatchCase,
    RepeatProbe,
    Unigram,
    _case_resampled_interval,
    _threshold_sweep,
    activation_patching,
    build_patch_cases,
    natural_repeat_probes,
    content_symbol_name,
    patch_seq_len_refusal,
    probe_record_retention,
    resolve_distance_bands,
    summarise_patching,
)

#: Symbols per content token, measured on the five shipped far-band windows of
#: 2026-07-30. The whole point of the content-symbol geometry is these numbers.
MEASURED_SYMBOLS_PER_TOKEN = {
    "gpt2-large": 4.403,
    "gpt2-xl": 4.403,
    "protgpt2": 2.816,
    "progen2-small": 0.996,
    "progen2-base": 0.996,
    "progen2-medium": 0.996,
}


def _load_stage(filename: str):
    """Import a numbered entry point by path, the way the worker's preflight does."""

    spec = importlib.util.spec_from_file_location(f"_farband_{filename[:2]}", STAGE_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# =========================================================== fixtures on the CPU
#
# The panel's own checkpoints need a GPU and are not available to a unit test, so
# the patching plumbing is exercised on a real two-layer GPT-2 at eight
# dimensions: the same class, the same eager kernel, the same module names the
# hooks resolve.  Nothing here asserts a number the model produces; what is
# asserted is the geometry of the cases, the eligibility arithmetic and the
# refusals.

_VOCAB = 16
_CONTENT_TOKENS = tuple(range(2, 12))


class _CharTokenizer:
    """One token per character, over a ten-symbol alphabet."""

    is_fast = True
    bos_token_id = None
    eos_token_id = None
    pad_token_id = 0

    def __call__(self, text, return_tensors=None, **_kwargs):
        return {"input_ids": [_CONTENT_TOKENS[ord(c) - ord("a")] for c in text]}

    @staticmethod
    def decode(token_ids):
        return "".join(chr(ord("a") + int(t) - 2) for t in token_ids)


def _tiny_gpt2(*, dtype: str = "float32", seed: int = 3) -> Arm:
    config = GPT2Config(
        n_layer=2, n_head=2, n_embd=8, vocab_size=_VOCAB, n_positions=64,
        attn_implementation="eager",
    )
    torch.manual_seed(seed)
    return Arm(
        spec=replace(PANEL["gpt2-large"], name="tiny-gpt2", n_layer=2, d_model=8),
        model=GPT2LMHeadModel(config).eval(),
        tokenizer=_CharTokenizer(),
        device="cpu",
        dtype=dtype,
        attn_implementation="eager",
    )


def _unigram(support: tuple[int, ...] = _CONTENT_TOKENS) -> Unigram:
    return Unigram(
        token_ids=np.asarray(support, dtype=np.int64),
        counts=np.full(len(support), 10, dtype=np.int64),
        total_tokens=10 * len(support),
        scored_sequences=1,
    )


def _plan(bands, *, span: int = 1) -> DistanceBandPlan:
    return resolve_distance_bands(bands, unit="token", corruption_span=span)


def _records(n_records: int, length: int, seed: int = 11, symbols: int = 10) -> list[str]:
    rng = np.random.default_rng(seed)
    alphabet = [chr(ord("a") + i) for i in range(symbols)]
    return ["".join(rng.choice(alphabet, size=length)) for _ in range(n_records)]


# ===================================================== build_patch_cases geometry


def test_every_case_lands_inside_the_band_it_is_labelled_with():
    arm = _tiny_gpt2()
    bands = ((1, 1), (5, 8))
    cases = build_patch_cases(
        arm, _records(10, 40), _unigram(), seq_len=24, plan=_plan(bands),
        cases_per_band=8, seed=5,
    )

    assert len(cases) == 16
    for case in cases:
        low, high = (int(part) for part in case.band.split("-"))
        assert low <= case.position_q - case.position_p <= high
        assert len(case.clean_ids) == len(case.corrupt_ids) == 24
        differing = [
            i for i, (a, b) in enumerate(zip(case.clean_ids, case.corrupt_ids)) if a != b
        ]
        assert differing == [case.position_p]
        # The cluster an interval has to resample: cases are drawn with
        # replacement, so several share a source record.
        assert 0 <= case.source < 10
    assert len({case.source for case in cases}) > 1


def test_a_band_that_cannot_be_filled_is_refused_rather_than_thinned():
    """A band that yields fewer cases than asked for must not just yield them.

    Here the cohort carries no token the unigram supports, which is the shape the
    layout-token exclusion produces on a wrapped rendering.  An arm that returned a
    short band would enter the panel with a different sample size per band, which
    Appendix B rule 21 says is partly a reading of sample size.
    """

    arm = _tiny_gpt2()
    with pytest.raises(RuntimeError, match="only 0/4 cases for band"):
        build_patch_cases(
            arm, _records(4, 24, symbols=4), _unigram(support=(10, 11)), seq_len=16,
            plan=_plan(((5, 8),)), cases_per_band=4, seed=1,
        )


def test_a_band_wider_than_the_window_is_refused_before_any_sampling():
    arm = _tiny_gpt2()
    with pytest.raises(ValueError, match="does not exceed the widest resolved"):
        build_patch_cases(
            arm, _records(4, 24), _unigram(), seq_len=12, plan=_plan(((11, 11),)),
            cases_per_band=4, seed=1,
        )


def test_a_cohort_that_never_reaches_the_window_is_refused():
    arm = _tiny_gpt2()
    with pytest.raises(RuntimeError, match="no cohort record reaches"):
        build_patch_cases(
            arm, _records(4, 10), _unigram(), seq_len=24, plan=_plan(((1, 1),)),
            cases_per_band=2, seed=1,
        )


def test_a_case_must_differ_at_exactly_the_perturbed_position():
    with pytest.raises(ValueError, match="differ at exactly"):
        PatchCase(
            clean_ids=(2, 3, 4, 5), corrupt_ids=(2, 9, 9, 5),
            position_p=1, position_q=3, band="1-2",
        )
    with pytest.raises(ValueError, match=r"p \+ span <= q"):
        PatchCase(
            clean_ids=(2, 3, 4, 5), corrupt_ids=(2, 9, 4, 5),
            position_p=3, position_q=1, band="1-2",
        )


# ============================================== activation patching, end to end


def _patching_result(*, minimum_effect: float = 1e-6, n_records: int = 10):
    arm = _tiny_gpt2()
    cases = build_patch_cases(
        arm, _records(n_records, 40), _unigram(), seq_len=24,
        plan=_plan(((1, 1), (5, 8))), cases_per_band=8, seed=5,
    )
    return arm, cases, activation_patching(
        arm, cases, minimum_effect=minimum_effect, batch_size=4,
        eligibility_resamples=200,
    )


def test_patching_reports_every_band_every_component_and_a_summary():
    arm, cases, result = _patching_result()

    assert result["bands"] == ["1-1", "5-8"]
    assert result["n_cases"] == len(cases)
    for band, entry in result["corruption_effect"].items():
        assert entry["n_cases"] == 8
        assert 0.0 <= entry["eligible_fraction"] <= 1.0
        assert entry["eligible_cases"] == round(entry["eligible_fraction"] * 8)
    for kind in result["component_kinds"]:
        for layer in range(arm.n_layer):
            for site in ("p", "q"):
                assert set(result["recovered_fraction"][f"{kind}|{layer}|{site}"]) == {
                    "1-1", "5-8"
                }
    summary = summarise_patching(result, arm=arm)
    assert set(summary) == {
        f"{kind}|{site}" for kind in result["component_kinds"] for site in ("p", "q")
    }


def test_chunking_the_forward_passes_changes_no_number():
    """The docstring claims the hooks carry no cross-case state.  Check it."""

    arm = _tiny_gpt2()
    cases = build_patch_cases(
        arm, _records(10, 40), _unigram(), seq_len=24,
        plan=_plan(((1, 1), (5, 8))), cases_per_band=8, seed=5,
    )
    whole = activation_patching(arm, cases, minimum_effect=1e-6, batch_size=64)
    chunked = activation_patching(arm, cases, minimum_effect=1e-6, batch_size=3)
    assert whole == chunked


def test_patching_is_refused_in_a_dtype_coarser_than_the_effect():
    """Appendix B rule 15b, enforced against the dtype the arm was loaded with."""

    assert "bfloat16" in PATCHING_REFUSED_DTYPES
    arm = _tiny_gpt2(dtype="bfloat16")
    case = PatchCase(
        clean_ids=(2, 3, 4, 5), corrupt_ids=(2, 9, 4, 5),
        position_p=1, position_q=3, band="1-2",
    )
    with pytest.raises(ValueError, match="refused in bfloat16"):
        activation_patching(arm, [case], minimum_effect=0.25)


# ================================================== the eligibility arithmetic


def test_an_interval_over_too_few_clusters_is_refused_not_pinched():
    """A percentile interval over four units reads narrower than one over eight."""

    flags = np.array([True, False] * 8)
    sources = np.repeat(np.arange(MINIMUM_ELIGIBILITY_CLUSTERS - 1), 2)[: flags.size]
    pinched = _case_resampled_interval(flags, sources, np.random.default_rng(0), 200)
    assert pinched["degenerate"] is True
    assert "q025" not in pinched
    assert str(MINIMUM_ELIGIBILITY_CLUSTERS) in pinched["reason"]

    enough = _case_resampled_interval(
        flags, np.repeat(np.arange(MINIMUM_ELIGIBILITY_CLUSTERS), 2), np.random.default_rng(0), 200
    )
    assert enough["degenerate"] is False
    assert enough["resampling_unit"] == "source_sequence"
    assert enough["q025"] <= enough["q975"]


def test_the_resampling_unit_is_the_sequence_not_the_case():
    """Cases from one sequence are correlated; resampling them reads too narrow."""

    flags = np.repeat([True, False], 32)
    clustered = _case_resampled_interval(
        flags, np.repeat(np.arange(8), 8), np.random.default_rng(0), 400
    )
    per_case = _case_resampled_interval(
        flags, np.arange(flags.size), np.random.default_rng(0), 400
    )
    assert clustered["n_source_sequences"] == 8
    assert per_case["n_source_sequences"] == flags.size
    width = clustered["q975"] - clustered["q025"]
    assert width > (per_case["q975"] - per_case["q025"])


def test_the_sweep_carries_the_run_s_own_cut_bit_identically():
    absolute = np.linspace(0.0, 3.0, 64)
    sources = np.repeat(np.arange(8), 8)
    rows = _threshold_sweep(absolute, sources, 0.25, 7, 200)

    assert len(rows) == len(set(ELIGIBILITY_THRESHOLD_LADDER) | {0.25})
    assert set(rows) == {repr(float(t)) for t in ELIGIBILITY_THRESHOLD_LADDER}
    own = rows[repr(0.25)]
    assert own["fraction"] == float((absolute >= 0.25).mean())
    reference = _case_resampled_interval(
        absolute >= 0.25, sources, np.random.default_rng(7), 200
    )
    assert own["q025"] == reference["q025"] and own["q975"] == reference["q975"]
    fractions = [rows[repr(float(t))]["fraction"] for t in sorted(ELIGIBILITY_THRESHOLD_LADDER)]
    assert fractions == sorted(fractions, reverse=True)


def test_a_cut_that_nearly_collides_with_the_ladder_keeps_its_own_row():
    """``f"{t:g}"`` rendered 0.100000001 as ``"0.1"`` and silently dropped a row."""

    absolute = np.linspace(0.0, 3.0, 64)
    sources = np.repeat(np.arange(8), 8)
    rows = _threshold_sweep(absolute, sources, 0.1 + 1e-9, 7, 100)

    assert len(rows) == len(ELIGIBILITY_THRESHOLD_LADDER) + 1
    assert repr(0.1) in rows and repr(0.1 + 1e-9) in rows
    assert rows[repr(0.1)]["threshold"] != rows[repr(0.1 + 1e-9)]["threshold"]


def test_the_eligible_fraction_and_its_sweep_row_agree_exactly():
    _, _, result = _patching_result(minimum_effect=0.05)
    for entry in result["corruption_effect"].values():
        row = entry["eligible_fraction_by_threshold"][repr(0.05)]
        assert row["fraction"] == entry["eligible_fraction"]
        assert row["n_cases"] == entry["n_cases"]


# ================================= the conditioned window that cannot be patched


def test_a_banded_conditioned_cohort_has_no_admissible_sequence_length():
    """The queued ZymCTRL job: --patch-seq-len 816 on a 600-1000 residue band."""

    refusal = patch_seq_len_refusal(
        PANEL["zymctrl"], 816, min_symbols=600, max_symbols=1000
    )
    assert refusal is not None
    assert "600-1000" in refusal
    assert "no value of --patch-seq-len admits it" in refusal
    # ...and it is the band, not the integer, that refuses.
    for seq_len in (610, 816, 1010):
        assert patch_seq_len_refusal(
            PANEL["zymctrl"], seq_len, min_symbols=600, max_symbols=1000
        ) is not None


def test_a_single_length_conditioned_cohort_is_admitted_within_the_slack():
    for seq_len in (808, 806 + CONDITIONING_TOKEN_SLACK):
        assert patch_seq_len_refusal(
            PANEL["zymctrl"], seq_len, min_symbols=806, max_symbols=806
        ) is None
    for seq_len in (806, 806 + CONDITIONING_TOKEN_SLACK + 1):
        assert patch_seq_len_refusal(
            PANEL["zymctrl"], seq_len, min_symbols=806, max_symbols=806
        ) is not None


def test_an_unconditioned_arm_is_never_refused_by_the_conditioned_rule():
    assert PANEL["zymctrl"].input_format in CLOSED_CONTENT_FORMATS
    for arm in ("gpt2-large", "protgpt2", "progen2-medium"):
        assert PANEL[arm].input_format not in CLOSED_CONTENT_FORMATS
        assert patch_seq_len_refusal(
            PANEL[arm], 128, min_symbols=600, max_symbols=1000
        ) is None


def test_build_patch_cases_refuses_a_cohort_whose_rows_lose_their_closing_marker():
    """Rows longer than the window are refused, not silently selected by length."""

    class _ConditionedTokenizer(_CharTokenizer):
        @staticmethod
        def get_vocab():
            return {"<start>": 12, "<end>": 13}

        def __call__(self, text, return_tensors=None, **_kwargs):
            return {"input_ids": [12, *super().__call__(text)["input_ids"], 13]}

    arm = _tiny_gpt2()
    arm.spec = replace(arm.spec, input_format="ec_conditioned")
    arm.tokenizer = _ConditionedTokenizer()
    with pytest.raises(RuntimeError, match="closed by a trailing marker"):
        build_patch_cases(
            arm, _records(4, 30) + _records(2, 40, seed=3), _unigram(),
            seq_len=32, plan=_plan(((1, 1),)), cases_per_band=2, seed=1,
        )


# ============================================ record-level retention of probes


def _repeat_cohort(records: list[str], repeats: list[list[int]]) -> Cohort:
    return Cohort(
        "repeat_probe_cohort", "text", records, 1, 10_000,
        {
            "repeats": repeats,
            "criterion": {"kind": "approximate"},
            "repeat_stats": [
                {"identity_fraction": 1.0 if index % 2 == 0 else 0.5, "length": span}
                for index, (_, _, span) in enumerate(repeats)
            ],
        },
    )


class _PairTokenizer:
    """Two characters per token, with offsets: a stand-in for multi-symbol BPE.

    Enough to reproduce the mechanism that drops a record -- the two copies of a
    repeat are segmented identically only when the period is even.
    """

    is_fast = True
    bos_token_id = None

    def __call__(self, text, return_tensors=None, return_offsets_mapping=False, **_kw):
        ids, offsets = [], []
        for start in range(0, len(text), 2):
            piece = text[start : start + 2]
            ids.append(2 + (sum(ord(c) for c in piece) % 10))
            offsets.append((start, start + len(piece)))
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


def _pair_arm() -> Arm:
    arm = _tiny_gpt2()
    arm.tokenizer = _PairTokenizer()
    return arm


def _record_with_repeat(period: int, span: int = 16, seed: int = 2) -> tuple[str, list[int]]:
    rng = np.random.default_rng(seed)
    alphabet = [chr(ord("a") + i) for i in range(10)]
    unit = "".join(rng.choice(alphabet, size=span))
    filler = "".join(rng.choice(alphabet, size=period - span))
    tail = "".join(rng.choice(alphabet, size=8))
    return unit + filler + unit + tail, [0, period, span]


def test_a_probe_carries_the_cohort_record_it_was_built_from():
    aligned, coords = _record_with_repeat(period=16)
    cohort = _repeat_cohort([aligned, aligned], [coords, coords])
    probes = natural_repeat_probes(_pair_arm(), cohort, max_tokens=64)

    assert [probe.record_index for probe in probes] == [0, 1]
    retention = probe_record_retention(probes, cohort)
    assert retention["record_retention"] == 1.0
    assert retention["n_dropped"] == 0
    assert retention["dropped_identity_fraction_mean"] is None


def test_record_level_loss_is_reported_with_the_bias_it_carries():
    """A record that aligns nowhere has no probe, so no mean over probes sees it."""

    aligned, aligned_coords = _record_with_repeat(period=16, seed=2)
    dropped, dropped_coords = _record_with_repeat(period=17, seed=4)
    cohort = _repeat_cohort(
        [aligned, dropped, aligned], [aligned_coords, dropped_coords, aligned_coords]
    )
    probes = natural_repeat_probes(_pair_arm(), cohort, max_tokens=64)

    retention = probe_record_retention(probes, cohort)
    assert retention["n_records"] == 3
    assert retention["n_retained"] == 2
    assert retention["record_retention"] == pytest.approx(2 / 3)
    # The evidence, not a flag: the survivors differ from the dropped record on
    # exactly the axis the approximate criterion varies.
    assert retention["retained_identity_fraction_mean"] == pytest.approx(1.0)
    assert retention["dropped_identity_fraction_mean"] == pytest.approx(0.5)


def test_a_probe_set_that_lost_most_of_its_cohort_is_refused():
    aligned, aligned_coords = _record_with_repeat(period=16, seed=2)
    dropped, dropped_coords = _record_with_repeat(period=17, seed=4)
    cohort = _repeat_cohort(
        [aligned] + [dropped] * 3, [aligned_coords] + [dropped_coords] * 3
    )
    with pytest.raises(RuntimeError, match="reached a probe"):
        natural_repeat_probes(_pair_arm(), cohort, max_tokens=64)


def test_the_retention_floor_is_attainable_by_the_control_that_defines_it():
    """Appendix B rule 2. Measured: text 0.989, ProGen2 and ZymCTRL 1.000."""

    assert 0.0 < MINIMUM_PROBE_RECORD_RETENTION < 808 / 817


def test_retention_cannot_be_computed_from_a_probe_set_that_lost_its_indices():
    aligned, coords = _record_with_repeat(period=16)
    cohort = _repeat_cohort([aligned], [coords])
    anonymous = [
        RepeatProbe(
            kind="natural_repeat_approximate", input_ids=(1, 2, 3), query_positions=(2,),
            key_positions=(1,), coverage=1.0, repeat_symbols=4,
        )
    ]
    with pytest.raises(ValueError, match="record indices"):
        probe_record_retention(anonymous, cohort)


# ===================================== the stage's own argument-time refusals


def _stage_04():
    return _load_stage("04_circuit_primitives.py")


def test_the_stage_defaults_to_float32_and_refuses_patching_below_it():
    module = _stage_04()
    assert _argparse_default("04_circuit_primitives.py", "--dtype") == "float32"

    namespace = _stage_namespace(dtype="bfloat16", sections=["patching"])
    with pytest.raises(ValueError, match="Appendix B rule 15b"):
        module.validate_patching_arguments(namespace)
    module.validate_patching_arguments(_stage_namespace(dtype="float32"))


def test_the_stage_refuses_the_impossible_zymctrl_job_before_a_checkpoint_loads():
    module = _stage_04()
    namespace = _stage_namespace(arms=["zymctrl"], patch_seq_len=816)
    with pytest.raises(ValueError, match="no value of --patch-seq-len admits it"):
        module.validate_patching_arguments(namespace)


def test_every_alignment_statistic_reaches_per_head_for_every_probe():
    """The offset-two decoy was computed on the natural probes and discarded."""

    module = _stage_04()
    statistics = ("prefix_matching", "same_token", "offset_two")
    ranking = module.head_ranking(
        {name: np.zeros((2, 2)) for name in statistics},
        {label: {name: np.zeros((2, 2)) for name in statistics} for label in ("exact", "approximate")},
        {"diagonal_fraction": np.zeros((2, 2)), "mean_normalised_rank": np.zeros((2, 2))},
        {"diagonal_fraction": np.zeros((2, 2)), "mean_normalised_rank": np.zeros((2, 2))},
    )

    for name in statistics:
        assert f"{name}_synthetic" in ranking
        for label in ("exact", "approximate"):
            assert f"{name}_natural_{label}" in ranking
    # The spellings frozen artefacts already carry.
    assert {
        "prefix_matching_synthetic", "same_token_synthetic", "offset_two_synthetic",
        "prefix_matching_natural_exact", "same_token_natural_exact",
        "copy_diagonal_fraction", "copy_mean_normalised_rank",
        "copy_matched_diagonal_fraction", "copy_matched_mean_normalised_rank",
    } <= set(ranking)


def _stage_namespace(**overrides):
    from argparse import Namespace

    fields = {
        "arms": ["gpt2-large"],
        "sections": ["patching"],
        "dtype": "float32",
        "patch_seq_len": 128,
        "patch_minimum_effect": 0.25,
        "protein_min_len": 600,
        "protein_max_len": 1000,
        "patch_distance_unit": "token",
        "patch_distance_band": None,
        "patch_corruption_span": None,
    }
    fields.update(overrides)
    return Namespace(**fields)


def _argparse_default(filename: str, flag: str):
    tree = ast.parse((STAGE_DIR / filename).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"):
            continue
        if not node.args or getattr(node.args[0], "value", None) != flag:
            continue
        for keyword in node.keywords:
            if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                return keyword.value.value
    raise AssertionError(f"{filename} declares no constant default for {flag}")


# ============================================ the stage's declared cohort bands


def test_circuit_primitives_declares_both_of_the_bands_it_draws():
    record = pc.stage_contract_record("circuit_primitives", ["protgpt2"])
    bands = record["cohort_band"]["protein_residue_bands"]

    assert [band["protein_residues"] for band in bands] == [[600, 1000], [200, 800]]
    assert record["cohort_band"]["matches_qualifying_stage"] is False
    assert record["cohort_band"]["qualifying_stage_protein_residues"] == [64, 246]
    assert "QUALIFYING BAND" in record["cohort_band"]["reason"]


def test_a_null_band_means_the_stage_declares_it_draws_none():
    """It used to read identically for a stage that drew two and declared neither."""

    for stage in ("explanation_channel", "convergence_control"):
        band = pc.stage_contract_record(stage, [])["cohort_band"]
        assert band["matches_qualifying_stage"] is None
        assert band["protein_residue_bands"] == []
        assert band["reason"]
    for stage, contract in pc.STAGE_CONTRACTS.items():
        assert contract.protein_band_reason, stage
        if contract.protein_bands:
            assert pc.stage_contract_record(stage, [])["cohort_band"][
                "matches_qualifying_stage"
            ] is not None, stage


@pytest.mark.parametrize(
    ("stage", "filename"),
    [
        ("circuit_primitives", "04_circuit_primitives.py"),
        ("induction_path_patching", "11_induction_path_patching.py"),
        ("relational_channel", "05_relational_channel.py"),
        ("probe_and_erasure", "09_probe_and_erasure.py"),
        ("homology_control", "10_homology_control.py"),
    ],
)
def test_declared_bands_match_the_stage_scripts_own_argparse_defaults(stage, filename):
    """A declaration that can drift from the flag it mirrors is worth nothing."""

    for band in pc.STAGE_CONTRACTS[stage].protein_bands:
        observed = None
        for low_flag, high_flag in _band_flags(band.argument_prefix):
            try:
                observed = (
                    _argparse_default(filename, low_flag),
                    _argparse_default(filename, high_flag),
                )
            except AssertionError:
                continue
            break
        assert observed is not None, f"{filename} spells no band for {band.argument_prefix}"
        assert observed == band.residues, f"{stage}/{band.argument_prefix}"


def _band_flags(prefix: str) -> list[tuple[str, str]]:
    """Every way a stage in this campaign spells one band pair.

    Matched on shape rather than on a literal name, for the reason
    ``transfer_gap/tg_contract.py::residue_bound_prefixes`` gives: a name-based
    lookup keyed on ``res_min``/``res_max`` is blind to the two other spellings in
    use here (``--min-len``/``--max-len`` and ``--protein-min-len``), and being
    blind to a band is exactly the failure this declaration exists to prevent.
    """

    head, _, tail = prefix.rpartition("_")
    dashed = prefix.replace("_", "-")
    candidates = [(f"--{dashed}-min", f"--{dashed}-max")]
    stem = f"{head.replace('_', '-')}-" if head else ""
    candidates.append((f"--{stem}min-{tail}", f"--{stem}max-{tail}"))
    return candidates


def test_the_token_ladder_is_unchanged():
    """Every artefact shipped before the unit became declarable was drawn on it."""

    assert DISTANCE_BANDS == ((1, 1), (2, 4), (5, 8), (9, 16), (17, 32), (33, 64))


# ================================================ the declared distance geometry
#
# The far-band band was declared in TOKENS while the arms it compares differ by
# 4.4x in content symbols per token, so the sign of the modality contrast was a
# free parameter of the unit: at a matched token band 33-64 the protein arms read
# highest, and re-indexed to a matched content distance the ordering reverses.
# Neither alignment isolates the model -- matching the distance in tokens
# mismatches the perturbation and matching it in symbols mismatches it the other
# way -- so both legs are declarable and the unit is recorded with the number.


def test_a_token_geometry_resolves_to_itself_and_takes_no_scale():
    plan = resolve_distance_bands(DISTANCE_BANDS, unit="token")

    assert plan.token_bands == DISTANCE_BANDS
    assert plan.labels == tuple(f"{low}-{high}" for low, high in DISTANCE_BANDS)
    assert plan.corruption_span_tokens == 1
    assert plan.symbols_per_token is None
    assert plan.as_dict()["corruption_span"]["realised_content_symbols"] is None
    with pytest.raises(ValueError, match="not resolved through symbols_per_token"):
        resolve_distance_bands(DISTANCE_BANDS, unit="token", symbols_per_token=2.8)


@pytest.mark.parametrize(
    ("arm", "expected"),
    [
        ("gpt2-large", ((3, 3), (4, 7), (8, 14))),
        ("protgpt2", ((4, 5), (7, 11), (12, 22))),
        ("progen2-medium", ((10, 16), (18, 32), (34, 64))),
    ],
)
def test_the_measured_panel_scales_resolve_the_ladder_per_arm(arm, expected):
    plan = resolve_distance_bands(
        SYMBOL_DISTANCE_BANDS,
        unit="content_symbol",
        corruption_span=5,
        symbols_per_token=MEASURED_SYMBOLS_PER_TOKEN[arm],
    )
    assert plan.token_bands == expected


def test_every_token_distance_a_band_resolves_to_lies_inside_the_band():
    """Containment, not nearest: the resolved band may not spill outside the request."""

    for scale in MEASURED_SYMBOLS_PER_TOKEN.values():
        plan = resolve_distance_bands(
            SYMBOL_DISTANCE_BANDS, unit="content_symbol",
            corruption_span=5, symbols_per_token=scale,
        )
        for (low, high), (token_low, token_high) in zip(plan.requested, plan.token_bands):
            assert low <= token_low * scale
            assert token_high * scale <= high
            assert token_low >= 1


def test_a_band_narrower_than_one_token_is_refused_not_rounded():
    """At 4.4 characters per token GPT-2 cannot be asked for a 2-4 character step."""

    for band in ((1, 1), (2, 4), (5, 8)):
        with pytest.raises(ValueError, match="no token image"):
            resolve_distance_bands(
                (band,), unit="content_symbol", corruption_span=5,
                symbols_per_token=MEASURED_SYMBOLS_PER_TOKEN["gpt2-large"],
            )
    # ...and the same band is perfectly answerable on a residue-level arm.
    assert resolve_distance_bands(
        ((5, 8),), unit="content_symbol", corruption_span=1,
        symbols_per_token=MEASURED_SYMBOLS_PER_TOKEN["progen2-small"],
    ).token_bands == ((6, 8),)


def test_labels_come_from_the_request_so_arms_align_at_one_content_distance():
    """The point of a shared unit: one label, one content distance, every arm."""

    plans = {
        arm: resolve_distance_bands(
            SYMBOL_DISTANCE_BANDS, unit="content_symbol", corruption_span=5,
            symbols_per_token=scale,
        )
        for arm, scale in MEASURED_SYMBOLS_PER_TOKEN.items()
    }
    assert len({plan.labels for plan in plans.values()}) == 1
    assert plans["protgpt2"].labels == ("9-16", "17-32", "33-64")
    # ...while the token bands behind that one label are of course different.
    assert plans["protgpt2"].token_bands != plans["progen2-base"].token_bands


def test_a_content_geometry_needs_a_positive_measured_scale():
    with pytest.raises(ValueError, match="needs the arm's measured"):
        resolve_distance_bands(SYMBOL_DISTANCE_BANDS, unit="content_symbol", corruption_span=5)
    for scale in (0.0, -1.0):
        with pytest.raises(ValueError, match="must be positive"):
            resolve_distance_bands(
                SYMBOL_DISTANCE_BANDS, unit="content_symbol", corruption_span=5,
                symbols_per_token=scale,
            )
    with pytest.raises(ValueError, match="not finite"):
        resolve_distance_bands(
            SYMBOL_DISTANCE_BANDS, unit="content_symbol", corruption_span=5,
            symbols_per_token=float("nan"),
        )
    with pytest.raises(ValueError, match="unknown distance unit"):
        resolve_distance_bands(DISTANCE_BANDS, unit="residues")
    assert DISTANCE_UNITS == ("token", "content_symbol")


# ------------------------------------------------- the perturbation, the second leg


def test_a_perturbation_finer_than_one_token_is_refused_at_the_arms_own_floor():
    """ProtGPT2 cannot corrupt less than ~2.8 residues; it has no smaller unit."""

    with pytest.raises(ValueError, match="below this arm's floor"):
        resolve_distance_bands(
            SYMBOL_DISTANCE_BANDS, unit="content_symbol", corruption_span=1,
            symbols_per_token=MEASURED_SYMBOLS_PER_TOKEN["protgpt2"],
        )
    # A ProGen2 arm can, and the panel's floor is therefore ProtGPT2's, not its.
    assert resolve_distance_bands(
        SYMBOL_DISTANCE_BANDS, unit="content_symbol", corruption_span=1,
        symbols_per_token=MEASURED_SYMBOLS_PER_TOKEN["progen2-small"],
    ).corruption_span_tokens == 1


def test_the_realised_perturbation_and_its_rounding_are_reported_per_arm():
    """The rounding is a real mismatch; it is recorded rather than assumed away."""

    spans = {
        arm: resolve_distance_bands(
            SYMBOL_DISTANCE_BANDS, unit="content_symbol", corruption_span=3,
            symbols_per_token=scale,
        ).as_dict()["corruption_span"]
        for arm, scale in MEASURED_SYMBOLS_PER_TOKEN.items()
        if scale <= 3
    }
    assert set(spans) == {"protgpt2", "progen2-small", "progen2-base", "progen2-medium"}
    assert spans["protgpt2"]["tokens"] == 1
    assert spans["protgpt2"]["realised_content_symbols"] == pytest.approx(2.816)
    assert spans["protgpt2"]["arm_floor_content_symbols"] == pytest.approx(2.816)
    assert spans["progen2-base"]["tokens"] == 3
    assert spans["progen2-base"]["realised_content_symbols"] == pytest.approx(2.988)
    for record in spans.values():
        assert record["requested"] == 3
        assert record["rounding_relative_error"] < 0.07


def test_a_perturbation_may_not_reach_the_read_out():
    with pytest.raises(ValueError, match="reaches the read-out"):
        resolve_distance_bands(((2, 4), (5, 8)), unit="token", corruption_span=3)


def test_the_caveat_travels_with_every_geometry_and_names_what_is_not_identified():
    caveat = resolve_distance_bands(DISTANCE_BANDS, unit="token").as_dict()["caveat"]

    assert "not unit-free" in caveat
    assert "WITHIN a modality" in caveat
    assert content_symbol_name(_tiny_gpt2()) == "character"
    assert content_symbol_name(_progen_like_arm()) == "residue"


def _progen_like_arm() -> Arm:
    arm = _tiny_gpt2()
    arm.spec = replace(arm.spec, modality="protein")
    return arm


# ------------------------------------------- the perturbation, end to end on CPU


def test_a_multi_token_case_differs_over_exactly_its_declared_span():
    arm = _tiny_gpt2()
    cases = build_patch_cases(
        arm, _records(10, 40), _unigram(), seq_len=24,
        plan=_plan(((4, 8),), span=3), cases_per_band=8, seed=5,
    )

    for case in cases:
        assert case.corrupt_span == 3
        differing = [
            i for i, (a, b) in enumerate(zip(case.clean_ids, case.corrupt_ids)) if a != b
        ]
        assert differing == [case.position_p, case.position_p + 1, case.position_p + 2]
        assert case.position_p + 3 <= case.position_q


def test_a_span_that_is_not_the_declared_one_is_refused():
    kwargs = dict(clean_ids=(2, 3, 4, 5, 6), position_p=1, position_q=4, band="3-3")
    PatchCase(corrupt_ids=(2, 9, 8, 5, 6), corrupt_span=2, **kwargs)
    with pytest.raises(ValueError, match="exactly the declared span"):
        PatchCase(corrupt_ids=(2, 9, 8, 5, 6), corrupt_span=1, **kwargs)
    with pytest.raises(ValueError, match="exactly the declared span"):
        PatchCase(corrupt_ids=(2, 9, 4, 5, 6), corrupt_span=2, **kwargs)
    with pytest.raises(ValueError, match="at least one token"):
        PatchCase(corrupt_ids=(2, 9, 4, 5, 6), corrupt_span=0, **kwargs)


def test_restoring_the_final_residual_at_the_read_out_recovers_the_metric_exactly():
    """The end-to-end invariant of the whole patching path, at both spans.

    The read-out is a function of the residual leaving the last block at ``q``, so
    restoring exactly that has to return the clean metric. It is the one number in
    this machinery whose value is known in advance, and it exercises the hooks,
    the per-chunk cache and the (case, position) site indexing that carries a
    multi-token perturbation.
    """

    arm = _tiny_gpt2()
    for span in (1, 3):
        cases = build_patch_cases(
            arm, _records(10, 40), _unigram(), seq_len=24,
            plan=_plan(((4, 8),), span=span), cases_per_band=8, seed=5,
        )
        result = activation_patching(
            arm, cases, minimum_effect=1e-8, batch_size=3, eligibility_resamples=50,
        )
        assert result["corruption_span_tokens"] == span
        entry = result["recovered_fraction"][f"resid_post|{arm.n_layer - 1}|q"]["4-8"]
        assert entry["n"] > 0
        assert entry["mean"] == pytest.approx(1.0, abs=1e-4)


def test_cases_of_two_different_spans_may_not_be_measured_together():
    common = dict(clean_ids=(2, 3, 4, 5, 6), position_p=1, position_q=4, band="3-3")
    mixed = [
        PatchCase(corrupt_ids=(2, 9, 4, 5, 6), corrupt_span=1, **common),
        PatchCase(corrupt_ids=(2, 9, 8, 5, 6), corrupt_span=2, **common),
    ]
    with pytest.raises(ValueError, match="same number of tokens"):
        activation_patching(_tiny_gpt2(), mixed, minimum_effect=0.1)


# ------------------------------------------- the stage's own geometry refusals


def test_a_content_symbol_run_may_not_span_two_modalities():
    """A residue is not a character; one band label would name two quantities."""

    module = _stage_04()
    namespace = _stage_namespace(
        arms=["gpt2-large", "protgpt2"], patch_distance_unit="content_symbol",
        patch_corruption_span=5,
    )
    with pytest.raises(ValueError, match="within a modality"):
        module.validate_patching_arguments(namespace)
    module.validate_patching_arguments(
        _stage_namespace(
            arms=["protgpt2", "progen2-base"], patch_distance_unit="content_symbol",
            patch_corruption_span=3,
        )
    )


def test_a_content_symbol_run_must_declare_its_perturbation():
    module = _stage_04()
    with pytest.raises(ValueError, match="requires --patch-corruption-span"):
        module.validate_patching_arguments(
            _stage_namespace(arms=["protgpt2"], patch_distance_unit="content_symbol")
        )


def test_the_declared_ladder_defaults_to_the_unit_s_own():
    module = _stage_04()
    assert module.requested_distance_bands(_stage_namespace()) == DISTANCE_BANDS
    assert module.requested_distance_bands(
        _stage_namespace(patch_distance_unit="content_symbol")
    ) == SYMBOL_DISTANCE_BANDS
    assert module.requested_distance_bands(
        _stage_namespace(patch_distance_band=["33-64", "17-32"])
    ) == ((33, 64), (17, 32))
    for bad in ("33", "33-", "a-b", "64-33", "0-4"):
        with pytest.raises(ValueError):
            module.parse_distance_band(bad)
