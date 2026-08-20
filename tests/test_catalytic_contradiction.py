"""Conditions D3.k must always hold, and its negative paths.

Written against EXP-R2-214 and its amendment 1 rather than against the
implementation. Six of these are this programme's own lessons rather than hygiene:

* **rho must not read the record's own catalytic residues.** Both forced conditions
  overwrite the same positions with the same residues, so a record's motif state
  cancels; if it did not, this readout would be the motif reader with extra steps and
  the whole track would be circular. Tested by mutating a record's anchors and
  requiring rho to be bit-identical.
* **the ceiling curve's k = 1 rung is exactly zero**, because a unigram reads no
  context. A curve whose anchor is not exactly zero is an indexing defect, and the
  curve builder must refuse rather than report it.
* **the counter-stratum is a second contradiction set**, so a model that matches the
  biology reference on the primary contrast and fails it must be reported as reading
  motifs -- not as a partial pass.
* **a fitted readout on a held-out side is refused**, because 15 pairs split 8/7 and 7
  is below the shared unit floor. Tested as a raise, not as a docstring.
* **a joint checkpoint is refused with the annotation reason**, because the corpus
  annotation is mostly right and would hand the model the answer.
* **the margin must be shown reachable before it decides anything** (Appendix B rule
  2): against a ceiling above AUROC 0.75 the factor-2 rule demands an AUROC above 1,
  and a bar no result can reach classifies nothing.

``torch`` is never imported and no checkpoint is loaded: every scorer here is either
the fragment conditional or a planted decoder, which is what keeps the suite fast and
what the real cell's residue-to-token verification exists to cover separately.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer import catalytic_contradiction as cx  # noqa: E402
from src.transfer.arms import AA20, PANEL  # noqa: E402
from src.transfer.statistics import MINIMUM_BOOTSTRAP_UNITS  # noqa: E402

SEED = 20260819
COHORT = REPO_ROOT / "results/transfer/pseudokinase_contradiction/pseudokinase_contradiction.jsonl"
COHORT_SHA256 = "2767b884d5282aa44221f85fb891fd10b23e2f378eaf1fdd9b42bbb6c904c0b2"


def _load_stage():
    path = REPO_ROOT / "scripts/transfer/40_catalytic_contradiction.py"
    spec = importlib.util.spec_from_file_location("_stage_40", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE = _load_stage()


@pytest.fixture(scope="module")
def world():
    background = cx.synthetic_background(SEED)
    pairs, counter = cx.synthetic_cohort(background, seed=SEED)
    return background, pairs, counter


def _design(pairs, counter, *, radius=0):
    return cx.Design(
        pairs=pairs,
        counter=counter,
        radius=radius,
        max_residues=cx.SYNTHETIC_WINDOW,
        seed=SEED,
        shift_draws=cx.MINIMUM_RANDOM_ANCHOR_DRAWS,
    )


def _args(**overrides):
    namespace = argparse.Namespace(
        arm=None, joint_rendering=None, cohort=None, cohort_sha256=None,
        kmer_background=None, high_order_background=None, ceiling_orders=None,
        max_residues=None, exclusion_radius=0, shift_draws=8, ceiling_factor=2.0,
        split=cx.WHOLE_COHORT, seed=SEED, bootstrap_draws=200, batch_size=4,
        device="cpu", out=Path("/dev/null"), synthetic=True, synthetic_seed=SEED,
        synthetic_coupling=0.5,
    )
    for key, value in overrides.items():
        setattr(namespace, key, value)
    return namespace


# ------------------------------------------------ the readout's defining property


def test_rho_does_not_read_the_records_own_catalytic_residues(world):
    """The property that stops this being the motif reader run through a model.

    Both forced conditions overwrite the same positions with the same residues, so a
    record whose anchors are mutated -- from the live triad to the kinase-dead one and
    to a third state that is neither -- must produce a bit-identical rho.
    """

    background, pairs, _ = world
    likelihood = cx.FragmentLikelihood(background[3])
    record = pairs[0].active
    values = []
    for state in ({"vaik_lys": "K", "hrd_asp": "D", "dfg_asp": "D"},
                  {"vaik_lys": "R", "hrd_asp": "N", "dfg_asp": "N"},
                  {"vaik_lys": "W", "hrd_asp": "P", "dfg_asp": "C"}):
        sequence = list(record.sequence)
        for position, name in zip(record.anchor_positions, record.anchor_names):
            sequence[position - 1] = state[name]
        mutated = cx.Record(**{**record.__dict__, "sequence": "".join(sequence)})
        requests, _ = cx.site_requests([mutated], radius=0, max_residues=cx.SYNTHETIC_WINDOW)
        values.append(cx.rho_table(likelihood, requests)[0]["rho"])
    assert values[0] == values[1] == values[2]
    assert values[0] != 0.0


def test_the_two_forced_states_are_the_experimental_kinase_dead_substitutions():
    # Declared from the literature rather than derived from the labels they are read
    # against, and conservative in charge and volume so the fragment channel moves as
    # little as the intervention allows.
    assert cx.LIVE_STATE == {"vaik_lys": "K", "hrd_asp": "D", "dfg_asp": "D"}
    assert cx.DEAD_STATE == {"vaik_lys": "R", "hrd_asp": "N", "dfg_asp": "N"}
    assert "K72R" in cx.STATE_SOURCE and "D166N" in cx.STATE_SOURCE


def test_scored_positions_are_downstream_of_the_first_anchor_and_outside_the_radius(world):
    _, pairs, _ = world
    record = pairs[0].dead
    window = cx.window_bounds(record, max_residues=cx.SYNTHETIC_WINDOW)
    for radius in cx.EXCLUSION_RADII:
        positions = cx.scored_positions(
            record, record.anchor_positions, radius=radius, window=window
        )
        assert positions, radius
        assert min(positions) > min(record.anchor_positions)
        for anchor in record.anchor_positions:
            assert all(abs(position - anchor) > radius for position in positions)
        assert max(positions) <= record.domain_to


def test_the_write_invariant_holds_and_a_leftward_scorer_cannot_break_it(world):
    background, pairs, counter = world
    likelihood = cx.FragmentLikelihood(background[3])
    requests, _ = cx.site_requests(
        [pairs[0].dead, pairs[0].active], radius=0, max_residues=cx.SYNTHETIC_WINDOW
    )
    invariant = cx.upstream_invariance(likelihood, requests)
    assert invariant["holds"] is True
    assert invariant["max_absolute_difference"] == 0.0


# ------------------------------------------------------------ the ceiling curve


def test_the_k1_rung_of_the_ceiling_curve_is_exactly_zero(world):
    background, pairs, counter = world
    unigram = cx.FragmentLikelihood(background[1])
    requests, _ = cx.site_requests(
        _design(pairs, counter).all_records, radius=0, max_residues=cx.SYNTHETIC_WINDOW
    )
    rows = cx.rho_table(unigram, requests)
    assert rows
    assert all(row["rho"] == 0.0 for row in rows)


def test_the_curve_builder_refuses_a_k1_anchor_that_is_not_exactly_zero(world):
    background, pairs, counter = world
    labels = np.array([1] * 8 + [0] * 8)
    groups = np.arange(16)
    model = np.linspace(-1.0, 1.0, 16)
    # k = 1 must be exactly zero; a curve whose anchor drifts is an indexing defect and
    # the builder has to stop rather than report it as a fact about the corpus.
    with pytest.raises(RuntimeError, match="k = 1 rung"):
        cx.fragment_ceiling_curve(
            [1, 3],
            background,
            {1: np.full(16, 1e-9), 3: np.linspace(-0.1, 0.1, 16)},
            labels,
            model,
            groups,
            factor=2.0,
            seed=SEED,
            draws=64,
        )


def test_the_fragment_ceiling_is_structurally_zero_beyond_its_own_order(world):
    # A k-order conditional reads k-1 residues of context, so an exclusion radius at or
    # above k-1 puts every forced anchor outside every scored window and rho is exactly
    # zero. This is the axis that separates a local corpus effect from propagation.
    background, pairs, counter = world
    for order in (2, 3):
        likelihood = cx.FragmentLikelihood(background[order])
        requests, _ = cx.site_requests(
            [pairs[0].active], radius=order - 1, max_residues=cx.SYNTHETIC_WINDOW
        )
        assert cx.rho_table(likelihood, requests)[0]["rho"] == 0.0


def test_ceiling_adequacy_reports_both_means_even_when_the_ratio_is_undefined():
    block = cx.ceiling_adequacy([0.0, 0.0, 0.0], [0.0, 0.1, 0.2])
    assert block["mean_absolute_model_rho"] == 0.0
    assert block["mean_absolute_ceiling_rho"] == pytest.approx(0.1)
    assert block["adequacy_ratio"] is None and block["binds"] is None
    assert block["undefined_reason"]


# ------------------------------------------------------------------ the margin


def test_the_margin_is_shown_reachable_before_it_decides_anything():
    """Appendix B rule 2 on the admission rule itself.

    Against a ceiling at AUROC 0.90 the factor-2 rule demands 0.5 + 2 * 0.40 = 1.30,
    which no AUROC can reach. A bar no result can reach classifies nothing, so it has to
    be reported as a property of the ceiling.
    """

    labels = np.array([1] * 10 + [0] * 10)
    groups = np.arange(20)
    strong = np.concatenate([np.linspace(1.0, 0.6, 10), np.linspace(0.5, 0.0, 10)])
    row = cx.margin_record(labels, strong, strong, groups, factor=2.0, seed=SEED, draws=200)
    assert row["auroc_ceiling"] == 1.0
    assert row["required_auroc"] == pytest.approx(1.5)
    assert row["margin_attainable"] is False
    assert row["attainable_by_factor"]["1.0"] is True
    assert row["attainable_by_factor"]["2.0"] is False


def test_clause_two_needs_both_halves_and_a_tie_clears_neither():
    labels = np.array([1] * 10 + [0] * 10)
    groups = np.arange(20)
    scores = np.concatenate([np.linspace(1.0, 0.6, 10), np.linspace(0.5, 0.0, 10)])
    row = cx.margin_record(labels, scores, scores, groups, factor=2.0, seed=SEED, draws=200)
    assert row["difference"] == 0.0
    assert row["clears_difference"] is False
    assert row["clears"] is False


def test_a_single_class_resample_is_skipped_and_a_single_class_contrast_is_refused():
    # The counter-stratum contrast resamples 23 singleton groups of which 8 carry the
    # negative class, so about one draw in twenty thousand contains no negative. A raise
    # there would kill a campaign cell on a resample rather than on a fact about the
    # cohort; the shared group bootstrap already skips a non-finite draw and refuses when
    # too many are.
    import numpy as np

    assert np.isnan(cx.auroc(np.ones(6), np.arange(6.0)))
    assert cx.auroc(np.array([1, 1, 0, 0]), np.array([1.0, 0.9, 0.2, 0.1])) == 1.0
    with pytest.raises(ValueError, match="non-finite on the full sample"):
        cx.auroc_interval(np.ones(16), np.arange(16.0), np.arange(16), seed=SEED, draws=64)


def test_a_ceiling_factor_below_one_is_refused():
    with pytest.raises(ValueError, match="below one"):
        cx.margin_record(
            np.array([1, 1, 0, 0] * 4), np.arange(16.0), np.arange(16.0),
            np.arange(16), factor=0.5, seed=SEED, draws=64,
        )


# -------------------------------------------------------------------- refusals


def test_a_fitted_readout_on_a_held_out_side_is_refused():
    assert cx.refuse_fitted_probe(cx.WHOLE_COHORT, fit_units=8, eval_units=7) is None
    for side, units in (("fit", 8), ("eval", 7)):
        with pytest.raises(ValueError) as raised:
            cx.refuse_fitted_probe(side, fit_units=8, eval_units=7)
        message = str(raised.value)
        assert "amendment 1" in message and "item 5" in message
        assert f"carries {units}" in message
        assert str(MINIMUM_BOOTSTRAP_UNITS) in message


def test_the_stage_refuses_a_held_out_side_before_anything_is_opened():
    with pytest.raises(ValueError, match="8 fit / 7 eval"):
        STAGE.resolve(_args(split="eval"))


def test_a_joint_checkpoint_is_refused_with_the_annotation_reason():
    from src.transfer import joint_modes

    for name in joint_modes.RENDERING_NAMES:
        with pytest.raises(ValueError) as raised:
            cx.refuse_joint_annotation_channel(name)
        message = str(raised.value)
        assert "annotation channel carries the answer" in message
        assert "9 of the 18" in message and "5 of 18" in message
    with pytest.raises(ValueError, match="annotation channel"):
        STAGE.resolve(_args(joint_rendering="prollama", synthetic=True))


def test_an_annotation_conditioned_protein_arm_is_refused_and_a_sequence_only_one_is_not():
    # ZymCTRL's EC tag names protein-kinase activity for 15 of 15 matched actives and for
    # 5 of 18 dead records, so conditioning on it hands the model a partial label. The
    # refusal reads the declared input format, never the arm's name.
    with pytest.raises(ValueError, match="annotation"):
        cx.assert_sequence_only(PANEL["zymctrl"])
    with pytest.raises(ValueError, match="text arm"):
        cx.assert_sequence_only(PANEL["gpt2-large"])
    admitted = cx.assert_sequence_only(PANEL["progen2-small"])
    assert admitted["reads_annotation_channel"] is False


def test_every_pre_registered_decision_is_named_when_it_is_missing():
    for flag in STAGE.PRE_REGISTERED_DECISIONS:
        with pytest.raises(ValueError) as raised:
            STAGE.resolve(_args(**{flag: None}))
        assert f"--{flag.replace('_', '-')}" in str(raised.value)


def test_the_declared_headline_radius_must_be_a_rung_of_the_reported_sweep():
    with pytest.raises(ValueError, match="outside the declared sweep"):
        STAGE.resolve(_args(exclusion_radius=1))


def test_too_few_shifted_site_draws_are_refused():
    with pytest.raises(ValueError, match="below the declared"):
        STAGE.resolve(_args(shift_draws=cx.MINIMUM_RANDOM_ANCHOR_DRAWS - 1))


def test_a_campaign_run_names_every_campaign_flag_it_is_missing():
    with pytest.raises(ValueError) as raised:
        STAGE.resolve(_args(synthetic=False, arm="progen2-small"))
    message = str(raised.value)
    for flag in ("--cohort", "--cohort-sha256", "--ceiling-orders", "--max-residues"):
        assert flag in message


def test_this_module_adds_no_resampler_of_its_own():
    # A repository invariant test enumerates every function in `src.transfer` whose name
    # mentions resampling and requires it to reach the shared unit floor. This module must
    # add none: the group bootstrap has one implementation and this module calls it.
    import inspect

    offenders = [
        name
        for name, member in vars(cx).items()
        if inspect.isfunction(member)
        and member.__module__ == cx.__name__
        and ("bootstrap" in name.lower() or "resampl" in name.lower())
    ]
    assert offenders == []


# ------------------------------------------------------ the four planted worlds


@pytest.fixture(scope="module")
def synthetic_artefact(tmp_path_factory):
    out = tmp_path_factory.mktemp("d3k")
    argv = [
        "40_catalytic_contradiction.py",
        "--synthetic",
        "--exclusion-radius", "0",
        "--shift-draws", str(cx.MINIMUM_RANDOM_ANCHOR_DRAWS),
        "--ceiling-factor", "2.0",
        "--split", cx.WHOLE_COHORT,
        "--seed", str(SEED),
        "--bootstrap-draws", "200",
        "--device", "cpu",
        "--out", str(out),
    ]
    original = sys.argv
    sys.argv = argv
    try:
        STAGE.main()
    finally:
        sys.argv = original
    written = list(out.glob("catalytic_contradiction__synthetic__*.json"))
    assert len(written) == 1
    return json.loads(written[0].read_text(encoding="utf-8"))


def test_the_four_planted_decoders_return_four_different_correct_verdicts(synthetic_artefact):
    """The self-test checks the verdict, not the thresholds.

    A pipeline that could compute every number and reach only one verdict would pass a
    threshold-level self-test. These four are the four ways this readout can be wrong,
    and each has a different remedy.
    """

    certificate = synthetic_artefact["certificate"]
    assert certificate["recovered"] == certificate["expected"]
    assert certificate["recovered_every_verdict"] is True
    assert certificate["distinct_verdicts"] == len(cx.PLANTINGS)
    assert certificate["expected"] == {
        "catalysis": cx.CANDIDATE_KNOWLEDGE,
        "motif": cx.MOTIF_READING,
        "statistics": cx.RECOMBINATION,
        "null": cx.READOUT_DEGENERATE,
    }


def test_the_counter_stratum_is_read_separately_from_the_primary_contrast(synthetic_artefact):
    """A motif reader clears the primary contrast and dies on the counter-stratum.

    That is the whole reason ``active_despite_degradation`` is a second contradiction set
    and not a robustness check: both plantings reach the same primary verdict and the
    counter-stratum is what tells them apart.
    """

    catalysis = synthetic_artefact["worlds"]["catalysis"]
    motif = synthetic_artefact["worlds"]["motif"]
    assert catalysis["primary_contrast"]["verdict"]["verdict"] == cx.CLEARS_TOWARD_EXPERIMENT
    assert motif["primary_contrast"]["verdict"]["verdict"] == cx.CLEARS_TOWARD_EXPERIMENT
    assert catalysis["counter_stratum"]["verdict"]["verdict"] == cx.SEPARATES_COUNTER
    assert motif["counter_stratum"]["verdict"]["verdict"] == cx.DOES_NOT_SEPARATE_COUNTER
    assert catalysis["verdict"]["verdict"] == cx.CANDIDATE_KNOWLEDGE
    assert motif["verdict"]["verdict"] == cx.MOTIF_READING
    assert "motif" in motif["verdict"]["reason"]


def test_the_statistics_planting_lands_inside_its_own_ceiling_by_construction(
    synthetic_artefact,
):
    # The planted statistics decoder IS the synthetic world's fragment conditional, so
    # its adequacy against that rung is exactly 1 and its margin cannot clear. That is
    # what makes the RECOMBINATION verdict a property of the design rather than of a
    # tuned coupling.
    rows = synthetic_artefact["worlds"]["statistics"]["primary_contrast"]["ceiling"][
        "toward_experiment"
    ]["same_readout"]["rows"]
    top = [row for row in rows if row["order"] == max(cx.SYNTHETIC_ORDERS)][0]
    assert top["adequacy"]["adequacy_ratio"] == pytest.approx(1.0)
    assert top["clears"] is False


def test_the_synthetic_artefact_carries_its_pre_registration_and_its_limitations(
    synthetic_artefact,
):
    block = synthetic_artefact["pre_registration"]
    assert block["record"] == cx.PRE_REGISTRATION == "EXP-R2-214"
    assert block["amendments_implemented"] == list(cx.PRE_REGISTRATION_AMENDMENTS)
    assert "candidate" in block["scope"]
    assert set(block["what_each_account_predicts"]) == {
        "evolutionary_statistics", "catalytic_knowledge", "why_they_differ"
    }
    assert synthetic_artefact["provenance"]["runner"]["sha256"]
    assert set(STAGE.PROVENANCE_MODULES) <= set(synthetic_artefact["provenance"]["modules"])
    assert synthetic_artefact["limitations"]
    for flag in STAGE.CAMPAIGN_ONLY_FLAGS:
        assert flag not in synthetic_artefact["settings"]


def test_the_named_pre_registration_exists_in_the_experiment_log():
    # A stage may not name an identifier the log does not carry: that is the one failure
    # mode worse than naming none at all.
    log = (REPO_ROOT / "docs" / "EXPERIMENT_LOG.md").read_text(encoding="utf-8")
    assert f"## 2026-08-19 — {cx.PRE_REGISTRATION} pre-registered" in log
    assert f"{cx.PRE_REGISTRATION} amendment 1 (D3.k)" in log


def test_the_campaign_limitations_surface_the_uncertified_label_and_the_open_ceiling_row():
    assert "the_experimental_label_is_not_externally_certified" in STAGE.LIMITATIONS
    label = STAGE.LIMITATIONS["the_experimental_label_is_not_externally_certified"]
    assert "L-PK-1" in label and "curated" in label
    potts = STAGE.LIMITATIONS["a_potts_or_msa_coupling_ceiling_is_named_and_not_run"]
    assert "not run" in potts and "seven residues" in potts


# ---------------------------------------------------------------- the cohort


def _fixture_cohort(tmp_path, *, moderate=cx.MODERATE_CONFIDENCE_GENES,
                    contested=cx.CONTESTED_GENES):
    rng = np.random.default_rng(SEED)

    def entry(accession, gene, stratum, unit, partner, confidence):
        sequence = "".join(rng.choice(list(AA20), size=200))
        anchors = {"vaik_lys": 31, "hrd_asp": 131, "dfg_asp": 149}
        sequence = list(sequence)
        for name, position in anchors.items():
            sequence[position - 1] = cx.LIVE_STATE[name]
        sequence = "".join(sequence)
        return {
            "accession": accession, "gene": gene, "entry_name": gene + "_HUMAN",
            "stratum": stratum, "sequence": sequence, "split_unit": unit,
            "label_confidence": confidence, "annotation_stance": "silent",
            "matched_partner": partner,
            "kinase_domain": {
                "domain_from": 1, "domain_to": 200, "domain_bits": 100.0,
                "model": "Pkinase",
                "motifs": {
                    name: {
                        "residue": cx.LIVE_STATE[name], "position": position,
                        "intact": True, "motif": "XXX",
                    }
                    for name, position in anchors.items()
                },
            },
        }

    records = []
    for index in range(10):
        gene = moderate[index] if index < len(moderate) else f"DEAD{index}"
        confidence = "moderate" if index < len(moderate) else "high"
        records.append(entry(f"D{index:04d}", gene, cx.DEAD_STRATUM, index, f"A{index:04d}", confidence))
        records.append(entry(f"A{index:04d}", f"ACT{index}", cx.ACTIVE_STRATUM, index, f"D{index:04d}", ""))
    for index, gene in enumerate(contested):
        records.append(entry(f"C{index:04d}", gene, "contested", 100 + index, "", ""))
    path = tmp_path / "fixture.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "output": {"records_sha256": digest},
        "feasibility": {"matched_pairs_per_split_side": {"fit": 8, "eval": 7}},
    }
    path.with_name(path.name + ".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path, digest


def test_the_cohort_is_pinned_by_digest_and_a_rebuild_is_refused(tmp_path):
    path, digest = _fixture_cohort(tmp_path)
    cohort = cx.load_cohort(path, sha256=digest)
    assert cohort.sha256 == digest
    with pytest.raises(ValueError, match="pins"):
        cx.load_cohort(path, sha256="0" * 64)


def test_the_declared_gene_lists_are_checked_against_the_cohorts_own_fields(tmp_path):
    # The declaration exists so a reader sees the eight names; the check exists so the
    # declaration cannot drift from the cohort a sensitivity read would drop them from.
    path, digest = _fixture_cohort(tmp_path, moderate=("WRONG1", "WRONG2"))
    with pytest.raises(ValueError, match="moderate-confidence"):
        cx.load_cohort(path, sha256=digest)
    second = tmp_path / "second"
    second.mkdir()
    path, digest = _fixture_cohort(second, contested=("WRONG",))
    with pytest.raises(ValueError, match="contested"):
        cx.load_cohort(path, sha256=digest)


def test_the_moderate_confidence_drop_is_a_filter_over_the_frozen_cohort(tmp_path):
    path, digest = _fixture_cohort(tmp_path)
    cohort = cx.load_cohort(path, sha256=digest)
    every = cx.matched_pairs(cohort)
    high = cx.matched_pairs(cohort, high_confidence_only=True)
    assert len(every) == 10
    assert {pair.dead.label for pair in every} - {pair.dead.label for pair in high} == set(
        cx.MODERATE_CONFIDENCE_GENES
    )
    assert all(pair.dead.label_confidence == "high" for pair in high)
    # The contested records are held out of the positives by the cohort build; this stage
    # never has to exclude them, and that has to stay true.
    assert not (set(cx.CONTESTED_GENES) & {pair.dead.label for pair in every})


def test_a_matched_pair_is_one_split_unit_so_a_split_cannot_divide_the_contrast(tmp_path):
    path, digest = _fixture_cohort(tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    broken = []
    for line in lines:
        payload = json.loads(line)
        if payload["accession"] == "A0000":
            payload["split_unit"] = 999
        broken.append(json.dumps(payload))
    path.write_text("\n".join(broken) + "\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = json.loads(path.with_name(path.name + ".manifest.json").read_text())
    manifest["output"]["records_sha256"] = digest
    path.with_name(path.name + ".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    cohort = cx.load_cohort(path, sha256=digest)
    with pytest.raises(ValueError, match="one split unit"):
        cx.matched_pairs(cohort)


def test_an_anchor_that_does_not_index_its_own_sequence_is_refused(tmp_path):
    path, digest = _fixture_cohort(tmp_path)
    payloads = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    payloads[0]["kinase_domain"]["motifs"]["hrd_asp"]["residue"] = "W"
    path.write_text("\n".join(json.dumps(p) for p in payloads) + "\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = json.loads(path.with_name(path.name + ".manifest.json").read_text())
    manifest["output"]["records_sha256"] = digest
    path.with_name(path.name + ".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="does not describe this record"):
        cx.load_cohort(path, sha256=digest)


@pytest.mark.skipif(not COHORT.is_file(), reason="the staged D3.k cohort is absent")
def test_the_staged_cohort_carries_the_counts_the_amendment_records():
    cohort = cx.load_cohort(COHORT, sha256=COHORT_SHA256)
    pairs = cx.matched_pairs(cohort)
    high = cx.matched_pairs(cohort, high_confidence_only=True)
    counter = cohort.by_stratum(cx.COUNTER_STRATUM)
    assert len(cohort.records) == 461
    assert len(pairs) == 15
    assert len(counter) == 8
    # The sensitivity read has to remain readable: dropping the moderate-confidence dead
    # records must leave at least the shared unit floor.
    assert len(high) >= MINIMUM_BOOTSTRAP_UNITS
    assert {record.label for record in counter} == {
        "CASK", "PKDCC", "HASPIN", "WNK1", "WNK2", "WNK3", "WNK4", "POMK",
    }
    # PKDCC is the counter-stratum member that defeats the HMM channel rather than the
    # motif channel: all three catalytic columns intact, 17.9 bits.
    pkdcc = next(record for record in counter if record.label == "PKDCC")
    assert pkdcc.n_intact == 3 and pkdcc.domain_bits == pytest.approx(17.9)
