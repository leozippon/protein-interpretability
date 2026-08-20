"""Conditions D3.l must always hold, and its negative paths.

Written against EXP-R2-214 amendment 2's frozen text and against properties
rather than against the implementation. Five are this programme's own lessons
rather than hygiene:

* the ceiling curve's k = 1 rung must be **exactly** zero -- an order-1
  conditional reads no context, so a non-zero first point is an indexing defect
  and every higher order shares that indexing;
* arm admission is a **measurement**, so it is tested by measuring a tokenisation
  that fails it and one that passes at 1.000, and the refused arm is checked to
  have computed nothing behind its gate;
* the length control has to *do* something, so it is tested on a constructed case
  whose direction is known in advance rather than on whatever the cohort gives;
* the two planted worlds must come back with **opposite** verdicts and neither
  null may fire in either, which is the strongest form of a known-answer check;
* the standing margin reads the *ceiling's* contrast and not the arm's, which is
  one field name apart and produced a clause no positive effect could satisfy
  when D3.j made the same mistake.

``torch.set_num_threads(1)`` for the reason ``tests/test_alphabet_chemistry.py``
gives: the planted decoders are twenty-dimensional and thread launch dominates.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer import fold_discordance as fd  # noqa: E402

torch.set_num_threads(1)

COHORT = REPO_ROOT / "results/transfer/composition_matched_fold_set/composition_matched_fold_set.jsonl"
SEED = 20260819


def _load_stage():
    path = REPO_ROOT / "scripts/transfer/39_fold_discordance.py"
    spec = importlib.util.spec_from_file_location("_stage_39", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE = _load_stage()


def _args(**overrides) -> argparse.Namespace:
    settings = {
        "window": "raw",
        "junction_offset": 0,
        "min_window": 10,
        "resampling_unit": "anchor_group",
        "ceiling_orders": (1, 2, 3),
        "ceiling_factor": 2.0,
        "seed": SEED,
        "bootstrap_draws": 200,
        "batch_size": 16,
        "device": "cpu",
        "out": Path("/nonexistent"),
        "synthetic": True,
        "synthetic_seed": SEED,
        "arm": None,
        "cohort": None,
        "kmer_background": None,
        "high_order_background": None,
        "hmmer_bin": None,
        "pfam_hmm": None,
        "corpus_fasta": None,
        "profile_workdir": None,
        "hmmer_cpu": 8,
        "profile_parallel": 4,
    }
    settings.update(overrides)
    return argparse.Namespace(**settings)


@pytest.fixture(scope="module")
def planted():
    """The three worlds, measured once through the stage's own analysis path."""

    args = _args()
    out = {}
    for name in fd.PLANTINGS:
        world = fd.synthetic_world(planted=name, seed=SEED, device="cpu", ceiling_orders=(1, 2, 3))
        out[name] = STAGE.measure(
            triples=world.triples, arm=world.arm, ordered=world.ceiling, args=args
        )
    return out


# ------------------------------------------------- the curve's reachability anchor


def test_the_k_equals_one_rung_of_the_ceiling_curve_is_exactly_zero(planted):
    """An order-1 conditional reads no context, so the prefix cannot move it.

    Checked as exact equality and not as a tolerance. The point of the anchor is
    that it is zero *by construction*: anything else means the conditioned and the
    free pass are not reading the same positions, and the higher orders would be
    wrong in the same way without showing it.
    """

    world = fd.synthetic_world(planted="structure", seed=SEED, device="cpu", ceiling_orders=(1, 2, 3))
    member = fd.FragmentPrefixConditional(world.ceiling, order=1)
    for triple in world.triples[:4]:
        for label in fd.CANDIDATES:
            values, usable = member.advantage(triple.prefix, triple.continuations[label])
            assert values.shape == (len(triple.continuations[label]),)
            assert usable.all()
            assert np.array_equal(values, np.zeros_like(values))
    for body in planted.values():
        block = body["readings"]["raw"]["ceiling"]["1"] if "1" in body["readings"]["raw"]["ceiling"] else body["readings"]["raw"]["ceiling"]["curve"]["1"]
        assert block["ceiling_contrast"] == 0.0
        assert block["ceiling_contrast_is_exactly_zero"] is True


def test_a_higher_order_rung_is_not_zero_so_the_anchor_is_not_vacuous(planted):
    """The k = 1 check would pass on a ceiling that never computes anything."""

    curve = planted["structure"]["readings"]["raw"]["ceiling"]["curve"]
    assert curve["3"]["ceiling_contrast"] != 0.0
    assert curve[fd.PREFIX_COMPOSITION_MEMBER]["ceiling_contrast"] != 0.0


def test_a_markov_member_stops_reaching_the_window_once_the_offset_passes_its_order():
    """The structural fact that decides whether the fragment ceiling can bind."""

    world = fd.synthetic_world(planted="structure", seed=SEED, device="cpu", ceiling_orders=(1, 2, 3))
    member = fd.FragmentPrefixConditional(world.ceiling, order=3)
    assert member.reaches(0) is True
    assert member.reaches(1) is True
    assert member.reaches(2) is False
    triple = world.triples[0]
    values, _ = member.advantage(triple.prefix, triple.continuations["sequence_partner"])
    assert np.array_equal(values[2:], np.zeros_like(values[2:]))
    assert fd.PrefixAdaptedComposition(world.ceiling[1]).reaches(50) is True


# ----------------------------------------------------------------- the arm gate


def test_a_refused_arm_computes_nothing_behind_its_gate():
    """A multi-residue tokenisation is refused by measurement, with nothing behind it."""

    world = fd.synthetic_world(
        planted="structure", seed=SEED, device="cpu", ceiling_orders=(1, 2, 3),
        paired_tokenisation=True,
    )
    body = STAGE.measure(triples=world.triples, arm=world.arm, ordered=world.ceiling, args=_args())
    assert body["verdict"]["verdict"] == "NOT_MEASURABLE"
    assert body["admission"]["verdict"]["admitted"] is False
    assert body["admission"]["census"]["alignment"] < fd.MINIMUM_TOKEN_ALIGNMENT
    for computed in ("readings", "cost", "cohort_units"):
        assert computed not in body, f"{computed} was computed behind a closed gate"


def test_a_residue_tokenisation_passes_the_same_gate_at_one():
    """The gate has to be passable, or it is excluding by construction (rule 40)."""

    world = fd.synthetic_world(planted="neither", seed=SEED, device="cpu", ceiling_orders=(1, 2, 3))
    legs, _, _ = STAGE.build_legs(world.triples, seed=SEED)
    census = fd.alignment_census(world.arm, legs, sample=len(legs))
    assert census["alignment"] == 1.0
    assert census["both_passes_score_identical_tokens"] == 1.0
    assert fd.admit_arm(census, "synthetic", minimum=fd.MINIMUM_TOKEN_ALIGNMENT)["admitted"] is True


def test_the_scorer_is_invariant_to_how_legs_are_batched():
    """Right padding must not reach an earlier position, and it is checked rather than assumed.

    The legs of one triple have three different lengths, so they land in different
    batches at different batch sizes. If padding leaked -- through a non-causal
    operation, or through a position index taken from the padded width -- the
    conditioned and the free pass would be scored under different amounts of it
    and their difference would carry the batching.
    """

    world = fd.synthetic_world(planted="structure", seed=SEED, device="cpu", ceiling_orders=(1, 2, 3))
    legs, _, _ = STAGE.build_legs(world.triples[:6], seed=SEED)
    one = fd.ResidueSequenceScorer(world.arm, batch_size=1).logprobs(legs)
    many = fd.ResidueSequenceScorer(world.arm, batch_size=16).logprobs(legs)
    assert len(one) == len(many)
    for single, batched in zip(one, many):
        assert np.allclose(single, batched, atol=1e-6, rtol=0.0)


def test_an_arm_needing_a_conditioning_label_the_cohort_lacks_is_refused_with_its_coverage():
    """ZymCTRL's rendering carries an EC tag; the triple set defines none."""

    world = fd.synthetic_world(planted="neither", seed=SEED, device="cpu", ceiling_orders=(1, 2, 3))
    conditioned = fd.Arm(
        spec=type(world.arm.spec)(
            **{**vars(world.arm.spec), "input_format": "ec_conditioned"}
        ),
        model=world.arm.model,
        tokenizer=world.arm.tokenizer,
        device="cpu",
        dtype=fd.DTYPE,
    )
    coverage = fd.conditioning_label_coverage(conditioned, len(world.triples))
    assert coverage["requires_conditioning_label"] is True
    assert coverage["coverage"] == 0.0
    body = STAGE.measure(
        triples=world.triples, arm=conditioned, ordered=world.ceiling, args=_args()
    )
    assert body["verdict"]["verdict"] == "NOT_MEASURABLE"
    assert "admission" not in body and "readings" not in body


# ------------------------------------------------------------ the length control


def test_the_length_control_moves_the_contrast_in_the_expected_direction():
    """A constructed case whose answer is known before the code runs.

    The structure partner is the longer candidate and its extra tail carries no
    advantage at all, so the raw reading -- which averages that tail in -- must
    read *lower* than the length-controlled reading, which truncates both
    candidates to the shorter one. The two readings differing is the whole point
    of carrying both.
    """

    advantage = {
        "sequence_partner": np.full(20, 0.5),
        "structure_partner": np.concatenate([np.full(20, 1.0), np.zeros(20)]),
    }
    lengths = {label: advantage[label].size for label in fd.CANDIDATES}
    raw = fd.window_indices(lengths, mode="raw", offset=0, minimum=5)
    controlled = fd.window_indices(lengths, mode="length_controlled", offset=0, minimum=5)
    assert raw is not None and controlled is not None
    assert len(controlled["structure_partner"]) == len(controlled["sequence_partner"]) == 20
    assert len(raw["structure_partner"]) == 40
    raw_contrast = STAGE.windowed_contrast(advantage, raw)
    controlled_contrast = STAGE.windowed_contrast(advantage, controlled)
    assert raw_contrast == pytest.approx(0.0)
    assert controlled_contrast == pytest.approx(0.5)
    assert controlled_contrast > raw_contrast


def test_a_triple_with_no_window_left_is_dropped_rather_than_scored():
    lengths = {"sequence_partner": 12, "structure_partner": 12}
    assert fd.window_indices(lengths, mode="raw", offset=10, minimum=5) is None
    assert fd.window_indices(lengths, mode="raw", offset=0, minimum=5) is not None


# --------------------------------------------------------- the known-answer worlds


def test_the_two_planted_worlds_return_opposite_verdicts(planted):
    """A composition follower and a fold follower must not read the same way."""

    sequence = planted["sequence_statistics"]["verdict"]
    structure = planted["structure"]["verdict"]
    assert sequence["verdict"] == "RECOMBINATION"
    assert structure["verdict"] == "STRUCTURE_CANDIDATE"
    assert sequence["contrast"] < 0.0 < structure["contrast"]
    assert sequence["verdict"] != structure["verdict"]


def test_a_context_free_world_returns_an_exactly_zero_contrast_and_no_verdict(planted):
    """The third world is the one that must *not* produce a signed answer."""

    body = planted["neither"]
    assert body["verdict"]["verdict"] == "UNDECIDED"
    assert body["verdict"]["contrast"] == 0.0


def test_neither_null_fires_in_any_planted_world(planted):
    """A null that fires in a world with a real planted effect is a false positive."""

    for name, body in planted.items():
        nulls = body["readings"]["raw"]["nulls"]
        fired = sorted(key for key, block in nulls.items() if block["fires"])
        assert fired == [], f"{name}: {fired} fired"


def test_the_label_permutation_null_is_a_distribution_and_is_centred_at_zero():
    """One draw's interval excludes zero at chance, so the null is read as a distribution."""

    values = np.concatenate([np.full(40, 0.4), np.full(40, 0.1)])
    null = fd.sign_permutation_contrast(values, seed=SEED, draws=500)
    assert null["null_ci95"][0] < 0.0 < null["null_ci95"][1]
    assert null["fires"] is False
    assert null["observed_above_null_q95"] is True
    with pytest.raises(ValueError, match="distribution"):
        fd.sign_permutation_contrast(values, seed=SEED, draws=5)


def test_the_prefix_shuffle_null_fires_on_a_contrast_that_survives_the_shuffle():
    """Confound 3's model-side handle has to be able to fire, or it is decoration."""

    surviving = np.full(40, 0.3)
    groups = np.arange(40)
    fired = STAGE.shuffled_prefix_null(surviving, groups, seed=SEED, draws=200)
    assert fired["fires"] is True
    destroyed = np.concatenate([np.full(20, 0.02), np.full(20, -0.02)])
    assert STAGE.shuffled_prefix_null(destroyed, groups, seed=SEED, draws=200)["fires"] is False


def test_the_composition_world_lands_inside_the_channel_that_captures_it(planted):
    """Confound 3's ceiling member has to be the one that binds on a composition follower."""

    curve = planted["sequence_statistics"]["readings"]["raw"]["ceiling"]["curve"]
    composition = curve[fd.PREFIX_COMPOSITION_MEMBER]
    assert composition["adequacy"]["ratio"] > fd.CEILING_ADEQUACY_FLOOR
    assert composition["ceiling_contrast"] < 0.0
    fragment = max(curve[str(order)]["adequacy"]["ratio"] for order in (1, 2, 3))
    assert fragment < composition["adequacy"]["ratio"]


# ------------------------------------------------------------------ the margin


def test_the_margin_reads_the_ceilings_contrast_and_not_the_arms():
    """One field name apart, and it produced a clause nothing could satisfy in D3.j."""

    arm_block = {"contrast": 0.4}
    against = {"reference_contrast": 0.1, "difference": 0.3, "difference_ci95": [0.1, 0.5]}
    margin = fd.ceiling_margin(arm_block=arm_block, against_block=against, factor=2.0)
    assert margin["ceiling_contrast"] == 0.1
    assert margin["clauses"]["at_least_factor_times_ceiling"] is True
    assert margin["cleared"] is True
    assert margin["multiplicative_clause_binds"] is True
    tighter = fd.ceiling_margin(
        arm_block={"contrast": 0.15}, against_block=against, factor=2.0
    )
    assert tighter["clauses"]["at_least_factor_times_ceiling"] is False
    assert tighter["cleared"] is False


def test_a_negative_ceiling_is_clamped_and_the_clause_is_reported_as_not_binding():
    """The corpus account predicts a negative ceiling here; doubling one would weaken it."""

    margin = fd.ceiling_margin(
        arm_block={"contrast": 0.02},
        against_block={"reference_contrast": -0.9, "difference": 0.92, "difference_ci95": [0.5, 1.3]},
        factor=2.0,
    )
    assert margin["clauses"]["at_least_factor_times_ceiling"] is True
    assert margin["multiplicative_clause_binds"] is False
    assert "carries no weight" in margin["multiplicative_clause_note"]


def test_a_fired_null_voids_the_verdict_rather_than_reporting_it():
    verdict = fd.fold_verdict(
        margin={"cleared": True, "clauses": {}},
        arm_block={"contrast": 0.5, "difference_ci95": [0.2, 0.8]},
        nulls={"shuffled_prefix": {"fires": True}, "sign_permutation": {"fires": False}},
    )
    assert verdict["verdict"] == "VOID_NULL_FIRED"
    assert verdict["nulls_fired"] == ["shuffled_prefix"]


def test_a_contrast_toward_the_sequence_partner_is_recorded_as_recombination():
    verdict = fd.fold_verdict(
        margin={"cleared": False, "clauses": {"contrast_positive": False}},
        arm_block={"contrast": -0.3, "difference_ci95": [-0.5, -0.1]},
        nulls={},
    )
    assert verdict["verdict"] == "RECOMBINATION"


def test_a_positive_contrast_inside_the_ceiling_halts_rather_than_reading_as_partial():
    verdict = fd.fold_verdict(
        margin={"cleared": False, "clauses": {"at_least_factor_times_ceiling": False}},
        arm_block={"contrast": 0.3, "difference_ci95": [0.1, 0.5]},
        nulls={},
    )
    assert verdict["verdict"] == "INSIDE_CEILING"
    assert verdict["failed_clauses"] == ["at_least_factor_times_ceiling"]


# ------------------------------------------------------------------- the cohort


def test_the_pinned_cohort_loads_and_its_prefix_rule_reproduces_the_build():
    triples, record = fd.load_cohort(COHORT, expected_digest=fd.COHORT_DIGEST)
    assert len(triples) == 199
    assert record["sha256"] == fd.COHORT_DIGEST
    assert record["prefix_fraction"] == 0.5
    for triple in triples:
        assert triple.anchor_sequence.startswith(triple.prefix)
        assert triple.continuations["sequence_partner"] != triple.continuations["structure_partner"]


def test_a_cohort_whose_digest_moves_is_refused(tmp_path):
    altered = tmp_path / "composition_matched_fold_set.jsonl"
    altered.write_text(COHORT.read_text(encoding="utf-8")[:-1], encoding="utf-8")
    with pytest.raises(RuntimeError, match="pre-registered on"):
        fd.load_cohort(altered, expected_digest=fd.COHORT_DIGEST)


def test_both_resampling_units_are_measured_and_they_disagree_on_this_cohort():
    """The choice is declared because the two rules are not the same rule here."""

    triples, record = fd.load_cohort(COHORT, expected_digest=fd.COHORT_DIGEST)
    units = record["resampling_units"]
    assert units["anchor_group"]["n_units"] > units["shared_component"]["n_units"]
    assert units["shared_component"]["largest_unit_share"] > 0.5
    for unit in fd.RESAMPLING_UNITS:
        assert units[unit]["clears_unit_floor"] is True
        assert fd.triple_groups(triples, unit=unit).shape == (len(triples),)


# ---------------------------------------------------------------- the stage door


def test_the_stage_never_defaults_a_pre_registered_decision():
    parser = STAGE.build_parser()
    args = parser.parse_args(["--synthetic"])
    with pytest.raises(ValueError) as error:
        STAGE.resolve(args)
    for flag in STAGE.PRE_REGISTERED_DECISIONS:
        assert f"--{flag.replace('_', '-')}" in str(error.value)


def test_the_ceiling_orders_must_carry_the_anchor_and_the_frozen_rung():
    base = [
        "--window", "raw", "--junction-offset", "0", "--min-window", "10",
        "--resampling-unit", "anchor_group", "--ceiling-factor", "2.0",
        "--seed", str(SEED), "--synthetic",
    ]
    parser = STAGE.build_parser()
    with pytest.raises(ValueError, match="reachability anchor|k = 1"):
        STAGE.resolve(parser.parse_args([*base, "--ceiling-orders", "2,3"]))
    with pytest.raises(ValueError, match="froze"):
        STAGE.resolve(parser.parse_args([*base, "--ceiling-orders", "1,2"]))
    STAGE.resolve(parser.parse_args([*base, "--ceiling-orders", "1,2,3"]))


def test_campaign_flags_and_synthetic_are_refused_together():
    parser = STAGE.build_parser()
    base = [
        "--window", "raw", "--junction-offset", "0", "--min-window", "10",
        "--resampling-unit", "anchor_group", "--ceiling-orders", "1,2,3",
        "--ceiling-factor", "2.0", "--seed", str(SEED),
    ]
    with pytest.raises(ValueError, match="meaningless beside --synthetic"):
        STAGE.resolve(parser.parse_args([*base, "--synthetic", "--arm", "progen2-small"]))
    with pytest.raises(ValueError, match="a campaign run needs"):
        STAGE.resolve(parser.parse_args([*base, "--arm", "progen2-small"]))


def test_the_artefact_basename_carries_the_arm_the_cohort_and_the_seed():
    name = STAGE.artefact_name("progen2-small", fd.COHORT_DIGEST, SEED)
    assert "progen2-small" in name
    assert fd.COHORT_DIGEST[:12] in name
    assert f"seed{SEED}" in name


def test_the_superseded_background_is_refused_by_name(tmp_path):
    with pytest.raises(ValueError, match="superseded"):
        fd.load_ceiling(
            tmp_path / "kmer_background/uniref50_line_local_superseded_20260812",
            (1, 3),
            pinned=tmp_path / "uniref50",
        )


# ------------------------------------------------- the profile ceiling members


def _domtbl(tmp_path, rows):
    """A HMMER per-domain table with only the columns this parser reads."""

    lines = ["# a comment line HMMER always writes"]
    for target, profile, score, ali_from, ali_to in rows:
        fields = ["-"] * 23
        fields[0] = target
        fields[3] = profile
        fields[13] = str(score)
        fields[17] = str(ali_from)
        fields[18] = str(ali_to)
        lines.append(" ".join(fields))
    path = tmp_path / "domtbl.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_the_domain_table_is_parsed_by_position_and_ignores_comments(tmp_path):
    hits = fd.parse_domtblout(_domtbl(tmp_path, [("CND0_a", "PF00001", 42.5, 3, 12)]))
    assert len(hits) == 1
    assert hits[0].target == "CND0_a"
    assert hits[0].profile == "PF00001"
    assert hits[0].score == 42.5
    assert (hits[0].ali_from, hits[0].ali_to) == (3, 12)


def test_a_profile_member_scores_only_the_profiles_the_prefix_selected():
    """The prefix selects; the candidate is scored under exactly that selection.

    A candidate that hits a profile the prefix does not carry contributes nothing.
    Without this the member would measure "is this continuation in any family at
    all", which both candidates are, rather than "is it in the prefix's family".
    """

    prefix, candidate = "AAAA", "CDEFGHIK"
    member = fd.ProfileHomologyMember(
        "pfam_profile",
        selected={prefix: frozenset({"PF00001"})},
        hits={
            candidate: [
                fd.DomainHit(target="t", profile="PF00001", score=8.0, ali_from=1, ali_to=4),
                fd.DomainHit(target="t", profile="PF99999", score=800.0, ali_from=5, ali_to=8),
            ]
        },
        provenance={},
    )
    values, usable = member.advantage(prefix, candidate)
    assert usable.all()
    expected = fd.NATS_PER_BIT * 8.0 / 4.0
    assert np.allclose(values[:4], expected)
    assert np.array_equal(values[4:], np.zeros(4))


def test_a_prefix_with_no_profile_gives_exactly_zero_for_both_candidates():
    """The profile member's own reachability property, and it is the k = 1 property."""

    member = fd.ProfileHomologyMember(
        "corpus_profile", selected={"AAAA": frozenset()},
        hits={"CDEF": [fd.DomainHit(target="t", profile="CORPUS0", score=99.0, ali_from=1, ali_to=4)]},
        provenance={},
    )
    values, usable = member.advantage("AAAA", "CDEF")
    assert np.array_equal(values, np.zeros(4))
    assert usable.all()
    assert member.reaches(0) is True and member.reaches(500) is True


def test_a_profile_member_takes_the_maximum_over_covering_domains_and_clamps_at_zero():
    prefix, candidate = "AAAA", "CDEFGH"
    member = fd.ProfileHomologyMember(
        "pfam_profile",
        selected={prefix: frozenset({"PF1", "PF2"})},
        hits={
            candidate: [
                fd.DomainHit(target="t", profile="PF1", score=6.0, ali_from=1, ali_to=6),
                fd.DomainHit(target="t", profile="PF2", score=6.0, ali_from=1, ali_to=3),
                fd.DomainHit(target="t", profile="PF2", score=-50.0, ali_from=4, ali_to=6),
            ]
        },
        provenance={},
    )
    values, _ = member.advantage(prefix, candidate)
    # PF2 is denser over the first three residues, PF1 carries the rest, and the
    # negative domain is dropped rather than driving the maximum down.
    assert values[0] == pytest.approx(fd.NATS_PER_BIT * 6.0 / 3.0)
    assert values[4] == pytest.approx(fd.NATS_PER_BIT * 6.0 / 6.0)
    assert (values >= 0.0).all()


def test_a_domain_reaching_past_the_continuation_is_a_refusal_not_a_clip():
    member = fd.ProfileHomologyMember(
        "pfam_profile", selected={"AAAA": frozenset({"PF1"})},
        hits={"CDEF": [fd.DomainHit(target="t", profile="PF1", score=5.0, ali_from=2, ali_to=99)]},
        provenance={},
    )
    with pytest.raises(ValueError, match="residue continuation"):
        member.advantage("AAAA", "CDEF")


def test_the_profile_members_join_the_same_ceiling_family():
    world = fd.synthetic_world(planted="neither", seed=SEED, device="cpu", ceiling_orders=(1, 2, 3))
    member = fd.ProfileHomologyMember(
        "pfam_profile", selected={}, hits={}, provenance={"source": "test"}
    )
    family = STAGE.ceiling_members(world.ceiling, (1, 2, 3), [member])
    assert set(family) == {"1", "2", "3", fd.PREFIX_COMPOSITION_MEMBER, "pfam_profile"}
    assert fd.first_binding_order(
        {"1": {"adequacy": {"adequate": False}}, "pfam_profile": {"adequacy": {"adequate": True}}}
    ) == "pfam_profile"


def test_the_potts_gap_is_declared_rather_than_silent():
    """Clause 1 names couplings; an unbuilt member has to say so in the artefact."""

    assert "Potts" in fd.POTTS_MEMBER_ABSENT
    assert "cost is not the reason" in fd.POTTS_MEMBER_ABSENT.lower()
    limitations = STAGE.limitations_block(kind="protein")
    assert fd.POTTS_MEMBER_ABSENT in limitations["the_clause_one_family_and_what_of_it_is_built"]


def test_a_campaign_run_cannot_omit_the_profile_members():
    """The gap the coordinator closed must not be reopenable from the command line."""

    parser = STAGE.build_parser()
    args = parser.parse_args([
        "--arm", "progen2-small", "--cohort", "c", "--kmer-background", "k",
        "--high-order-background", "h", "--window", "raw", "--junction-offset", "0",
        "--min-window", "10", "--resampling-unit", "anchor_group",
        "--ceiling-orders", "1,2,3", "--ceiling-factor", "2.0", "--seed", str(SEED),
    ])
    with pytest.raises(ValueError) as error:
        STAGE.resolve(args)
    for flag in ("--hmmer-bin", "--pfam-hmm", "--corpus-fasta", "--profile-workdir"):
        assert flag in str(error.value)
