"""The collision null must be matched, must destroy the repeat, and must not be cheap to pass.

The census this null corrects counts a head when its attention on one
positionally-aligned key clears a fixed cut, and the number of *other* positions
holding the query's token is a property of the alphabet rather than of the model.
The repair is a per-arm baseline built by permuting each probe's own content
positions, so every collision statistic is held exactly while the planted
alignment is destroyed.

Three things therefore have to hold, and each is a way the repair could fail
without crashing:

* the null really is matched -- same length, same token multiset, same scored
  positions -- and the match is checked rather than inferred from the builder;
* the permutation really does remove the alignment, because a null that kept it
  would be unbeatable and every head count would read zero;
* the matching is load-bearing.  The last is the one worth a real construction:
  a head whose attention is driven by token *frequency* and which has no
  induction behaviour at all is correctly not counted against the permutation
  null, and IS counted against a null that redraws its tokens instead of
  permuting them.  That is the failure mode a composition-blind null has, and it
  runs in the direction of manufacturing induction heads.
"""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer import arms as A  # noqa: E402
from src.transfer import collision_null as CN  # noqa: E402
from src.transfer.circuits import RepeatProbe, attention_alignment_scores  # noqa: E402

VOCAB = 24
PAD, EOS = 0, 1
#: Content tokens: a deliberately small alphabet, which is the regime the
#: collision objection is about.
ALPHABET = np.arange(2, VOCAB, dtype=np.int64)
N_HEAD = 6
COPY_LEN = 24
N_PROBES = 64
SEED = 20260812

#: Head roles the stub model implements, in head order.  Only ``induction`` is an
#: induction head; every other role is a way of scoring highly on something that
#: is not induction.
HEAD_ROLES = (
    "uniform",
    "recency",
    "fixed_position",
    "duplicate_token",
    "collision_gated",
    "induction",
)


class _StubTokenizer:
    is_fast = True
    pad_token_id = PAD
    bos_token_id = None
    eos_token_id = EOS

    def decode(self, ids):
        return "".join("\n" if int(token) == EOS else chr(65 + int(token) % 26) for token in ids)


class _PrescribedAttention(torch.nn.Module):
    """A model whose attention pattern is a declared function of the tokens.

    Nothing is learned.  Each head implements one named mechanism exactly, so a
    census run over it has a known right answer and a wrong answer that is not a
    crash.
    """

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(n_head=N_HEAD)

    def forward(self, *, input_ids, attention_mask, output_attentions, use_cache):
        del attention_mask, use_cache
        if not output_attentions:
            raise AssertionError("the census must ask for attention patterns")
        batch, width = input_ids.shape
        pattern = torch.zeros(batch, N_HEAD, width, width, dtype=torch.float64)
        for row in range(batch):
            ids = input_ids[row].tolist()
            for query in range(width):
                allowed = list(range(query + 1))
                weights = {role: torch.zeros(width, dtype=torch.float64) for role in HEAD_ROLES}
                weights["uniform"][allowed] = 1.0
                weights["recency"][max(query - 1, 0)] = 1.0
                weights["fixed_position"][min(3, query)] = 1.0
                same = [j for j in allowed[:-1] if ids[j] == ids[query]]
                weights["duplicate_token"][same or [0]] = 1.0
                # No induction behaviour whatsoever, and no alignment: this head
                # only asks whether the query's token has occurred before, and
                # spreads over the whole prefix when it has. Its score therefore
                # moves with the collision multiplicity and with nothing else,
                # which is exactly the quantity a matched null holds fixed.
                weights["collision_gated"][allowed if same else [0]] = 1.0
                successors = [j + 1 for j in same if j + 1 <= query]
                weights["induction"][successors or [0]] = 1.0
                for head, role in enumerate(HEAD_ROLES):
                    pattern[row, head, query] = weights[role] / weights[role].sum()
        return SimpleNamespace(attentions=(pattern,))


def _arm(*, input_format: str = "raw") -> A.Arm:
    spec = A.ArmSpec(
        name="stub",
        path=Path("/nowhere"),
        path_variable="TRANSFER_TEXT_MODEL_BASE_DIR",
        modality="text",
        n_layer=1,
        d_model=8,
        tokenisation="byte",
        input_format=input_format,
        evaluation_cohort_source="openwebtext",
        architecture="t5_decoder",
        capabilities=frozenset({"budget", "lens", "circuits"}),
    )
    return A.Arm(
        spec=spec,
        model=_PrescribedAttention(),
        tokenizer=_StubTokenizer(),
        device="cpu",
        dtype="float32",
        attn_implementation="eager",
    )


def _probes(rng: np.random.Generator, *, n: int = N_PROBES) -> list[RepeatProbe]:
    """``[eos][random block][the same block]`` probes, the census's own geometry."""

    probes = []
    for _ in range(n):
        body = rng.choice(ALPHABET, size=COPY_LEN, replace=True).tolist()
        ids = [EOS] + body + body
        probes.append(
            RepeatProbe(
                kind="synthetic_repeat",
                input_ids=tuple(int(token) for token in ids),
                query_positions=tuple(1 + COPY_LEN + i for i in range(COPY_LEN - 1)),
                key_positions=tuple(1 + i + 1 for i in range(COPY_LEN - 1)),
                coverage=1.0,
                repeat_symbols=COPY_LEN,
            )
        )
    return probes


def _redrawn_null(rng: np.random.Generator, probes) -> list[RepeatProbe]:
    """A composition-BLIND null: same length, tokens redrawn from a wide alphabet.

    The naive null, and the comparison this suite exists to make.  It matches the
    probe in length and in scored positions and in nothing else, which is enough
    to manufacture a head out of a mechanism that has no induction behaviour at
    all.
    """

    out = []
    for probe in probes:
        ids = list(probe.input_ids)
        for position in range(1, len(ids)):
            ids[position] = int(rng.integers(1000, 60000))
        out.append(
            RepeatProbe(
                kind="collision_null",
                input_ids=tuple(ids),
                query_positions=probe.query_positions,
                key_positions=probe.key_positions,
                coverage=probe.coverage,
                repeat_symbols=probe.repeat_symbols,
            )
        )
    return out


class NullConstruction(unittest.TestCase):
    def setUp(self):
        self.arm = _arm()
        self.probes = _probes(np.random.default_rng(SEED))
        self.null = CN.collision_null_probes(self.arm, self.probes, seed=SEED)

    def test_the_null_is_matched_in_length_composition_and_scored_positions(self):
        report = CN.verify_null_match(self.arm, self.probes, self.null)
        self.assertTrue(report["length_matched"])
        self.assertTrue(report["token_multiset_matched"])
        self.assertTrue(report["scored_positions_matched"])
        # A permutation that moved nothing would pass every identity above. Half
        # the probe lies at or after the first query and is held byte-identical
        # by design, so the reachable ceiling here is just under one half.
        self.assertGreater(report["displaced_token_fraction"], 0.3)
        self.assertLess(report["displaced_token_fraction"], 0.5)

    def test_the_permutation_destroys_the_planted_alignment(self):
        self.assertEqual(CN.aligned_key_fraction(self.probes), 1.0)
        # Chance for a 22-symbol alphabet is about 1/22; the bound is loose on
        # purpose, because what must hold is that the alignment is GONE, not that
        # it lands on a particular number.
        self.assertLess(CN.aligned_key_fraction(self.null), 0.2)

    def test_the_null_holds_the_collision_statistic_the_objection_is_about(self):
        real = CN.antecedent_statistics(self.probes)
        null = CN.antecedent_statistics(self.null)
        self.assertAlmostEqual(
            real["same_token_antecedents_mean"], null["same_token_antecedents_mean"], places=6
        )

    def test_structural_positions_are_never_permuted(self):
        for probe, null in zip(self.probes, self.null):
            self.assertEqual(null.input_ids[0], EOS)
            self.assertEqual(probe.input_ids[0], null.input_ids[0])

    def test_a_probe_with_too_little_content_is_refused_rather_than_weakly_permuted(self):
        short = RepeatProbe(
            kind="synthetic_repeat",
            input_ids=(EOS, 5, 6, 7, 5, 6, 7),
            query_positions=(5, 6),
            key_positions=(2, 3),
            coverage=1.0,
            repeat_symbols=3,
        )
        with self.assertRaisesRegex(ValueError, "permutable content positions"):
            CN.collision_null_probes(self.arm, [short], seed=SEED)

    def test_a_mismatched_null_is_rejected_rather_than_scored(self):
        broken = list(self.null)
        ids = list(broken[0].input_ids)
        ids[5] = int(ids[5] + 1)
        broken[0] = RepeatProbe(
            kind="collision_null",
            input_ids=tuple(ids),
            query_positions=broken[0].query_positions,
            key_positions=broken[0].key_positions,
            coverage=1.0,
            repeat_symbols=broken[0].repeat_symbols,
        )
        with self.assertRaisesRegex(ValueError, "token multiset"):
            CN.verify_null_match(self.arm, self.probes, broken)


class PerProbeEstimator(unittest.TestCase):
    def test_per_probe_sums_reproduce_the_published_aggregate(self):
        arm = _arm()
        probes = _probes(np.random.default_rng(SEED), n=8)
        plain = attention_alignment_scores(arm, probes, batch_size=3)
        split = attention_alignment_scores(arm, probes, batch_size=3, per_probe=True)
        for name, matrix in plain["scores"].items():
            recomputed = split["per_probe_sums"][name].sum(axis=0) / split[
                "per_probe_counts"
            ].sum()
            np.testing.assert_allclose(recomputed, matrix, rtol=0, atol=1e-12)


class CensusVerdict(unittest.TestCase):
    """The census must count the induction head and refuse the four decoys."""

    def setUp(self):
        self.arm = _arm()
        self.probes = _probes(np.random.default_rng(SEED))
        self.payload = CN.collision_null_census(
            self.arm, self.probes, seed=SEED, batch_size=8, n_bootstrap=400
        )

    def _excess(self):
        return np.asarray(self.payload["statistics"]["prefix_matching"]["excess_per_head"])[0]

    def test_only_the_induction_head_clears_its_own_null(self):
        for label, cut in self.payload["statistics"]["prefix_matching"]["cuts"].items():
            self.assertEqual(cut["n_above_null"], 1, msg=f"at family-wise level {label}")
        self.assertEqual(
            int(np.argmax(self._excess())), HEAD_ROLES.index("induction")
        )

    def test_the_decoys_carry_no_excess_over_their_own_null(self):
        excess = self._excess()
        cut = self.payload["statistics"]["prefix_matching"]["cuts"]["0.95"]["null_cut"]
        for role in HEAD_ROLES:
            if role == "induction":
                continue
            self.assertLessEqual(excess[HEAD_ROLES.index(role)], cut, msg=role)
        # The collision-gated head is the one the matched null is FOR, and its
        # excess is not merely small: the permutation preserves its input exactly,
        # so it is zero to floating point.
        self.assertAlmostEqual(excess[HEAD_ROLES.index("collision_gated")], 0.0, places=12)

    def test_the_offset_two_decoy_statistic_counts_nothing(self):
        for label, cut in self.payload["statistics"]["offset_two"]["cuts"].items():
            self.assertEqual(cut["n_above_null"], 0, msg=f"at family-wise level {label}")

    def test_the_matched_null_is_what_stops_the_collision_head_being_counted(self):
        """The load-bearing test: a composition-blind null manufactures a head.

        ``collision_gated`` has no induction behaviour.  It asks only whether the
        query's token has occurred before, which a matched null holds fixed
        exactly and a redraw destroys, so against the permutation null its excess
        is zero and against the naive null it clears the cut.  That is the null
        being trivially passed, and it is the reason the matching is verified
        rather than assumed.
        """

        rng = np.random.default_rng(SEED + 99)
        wrong = _redrawn_null(rng, self.probes)
        real = attention_alignment_scores(self.arm, self.probes, batch_size=8)
        blind = attention_alignment_scores(self.arm, wrong, batch_size=8)
        head = HEAD_ROLES.index("collision_gated")
        blind_excess = float(
            real["scores"]["prefix_matching"][0, head]
            - blind["scores"]["prefix_matching"][0, head]
        )
        matched_excess = float(self._excess()[head])
        cut = self.payload["statistics"]["prefix_matching"]["cuts"]["0.95"]["null_cut"]
        self.assertLessEqual(matched_excess, cut)
        self.assertGreater(blind_excess, cut)

    def test_the_two_null_draws_are_independent_and_their_difference_is_the_floor(self):
        self.assertNotEqual(self.payload["seeds"]["null_a"], self.payload["seeds"]["null_b"])
        floor = self.payload["statistics"]["prefix_matching"]["null_noise_family_wise_median"]
        self.assertGreater(floor, 0.0)
        self.assertLess(floor, float(self._excess().max()))

    def test_the_studentised_cut_also_counts_only_the_induction_head(self):
        """The repaired cut must not buy its specificity back by counting more.

        Studentising exists to remove a variance mismatch, not to change which
        mechanism is detected, so the head it selects has to be the same one.
        """

        student = self.payload["statistics"]["prefix_matching"]["studentised"]
        for label in ("0.50", "0.90", "0.95", "0.99"):
            self.assertEqual(student[label]["n_above"], 1, msg=f"at level {label}")
        t = np.asarray(self.payload["statistics"]["prefix_matching"]["excess_t_per_head"])[0]
        self.assertEqual(int(np.argmax(t)), HEAD_ROLES.index("induction"))

    def test_the_studentised_decoy_counts_nothing(self):
        # The gate that was vacuous under a decoy-derived cut is live here.
        student = self.payload["statistics"]["offset_two"]["studentised"]
        for label in ("0.50", "0.90", "0.95", "0.99"):
            self.assertEqual(student[label]["n_above"], 0, msg=f"at level {label}")

    def test_a_token_independent_head_has_exactly_zero_excess_and_is_set_aside(self):
        """Why the "no spread, non-zero excess" refusal should be unreachable.

        The null differs from its probe only in token identities, so a head whose
        attention at the scored positions does not depend on the tokens scores
        identically on both and its excess is exactly zero.  Zero spread and a
        non-zero excess would need a head that responds identically across every
        probe and still moves between a probe and its permutation, which is a
        contradiction on this construction.  The refusal is kept as a guard
        against that reasoning being wrong, and this test pins the reasoning.
        """

        excess = self._excess()
        for role in ("uniform", "recency", "fixed_position"):
            self.assertEqual(excess[HEAD_ROLES.index(role)], 0.0, msg=role)
        degenerate = self.payload["statistics"]["prefix_matching"]["studentised"][
            "degenerate_heads"
        ]
        self.assertGreaterEqual(degenerate, 3)
        # Set aside, not counted, and not fatal: the run produced a verdict.
        self.assertEqual(
            self.payload["statistics"]["prefix_matching"]["studentised"]["0.95"]["n_above"], 1
        )

    def test_the_count_carries_an_interval_and_the_cut_is_swept(self):
        cuts = self.payload["statistics"]["prefix_matching"]["cuts"]
        self.assertEqual(sorted(cuts), ["0.50", "0.90", "0.95", "0.99"])
        levels = [cuts[label]["null_cut"] for label in sorted(cuts)]
        self.assertEqual(levels, sorted(levels))
        for cut in cuts.values():
            low, high = cut["n_above_null_ci"]
            self.assertLessEqual(low, cut["n_above_null"])
            self.assertLessEqual(cut["n_above_null"], high)

    def test_the_identity_ceiling_falls_as_the_alphabet_shrinks(self):
        """The diagnostic tracks the quantity the objection is about."""

        rng = np.random.default_rng(SEED + 7)
        wide = []
        for probe in _probes(rng, n=16):
            body = rng.integers(1000, 60000, size=COPY_LEN).tolist()
            wide.append(
                RepeatProbe(
                    kind="synthetic_repeat",
                    input_ids=tuple([EOS] + body + body),
                    query_positions=probe.query_positions,
                    key_positions=probe.key_positions,
                    coverage=1.0,
                    repeat_symbols=COPY_LEN,
                )
            )
        narrow = CN.antecedent_statistics(self.probes)
        broad = CN.antecedent_statistics(wide)
        self.assertGreater(
            narrow["same_token_antecedents_mean"], broad["same_token_antecedents_mean"]
        )
        self.assertLess(narrow["identity_ceiling"], broad["identity_ceiling"])


class SchedulingContract(unittest.TestCase):
    def test_the_byte_level_text_arms_are_admitted_by_this_stage(self):
        sys.path.insert(0, str(REPO_ROOT / "scripts/transfer"))
        from panel_contract import stage_arms  # noqa: PLC0415

        eligible, _ = stage_arms("collision_null_census", sorted(A.PANEL))
        for rung in ("bygpt5-small-en", "bygpt5-base-en", "bygpt5-medium-en"):
            self.assertIn(rung, eligible)
        # The point of admitting them: a text arm on each side of the alphabet.
        self.assertIn("gpt2-large", eligible)
        self.assertIn("protgpt2", eligible)
        self.assertIn("progen2-medium", eligible)

    def test_the_census_refuses_an_architecture_whose_pattern_is_not_declared(self):
        arm = _arm()
        arm.spec = replace(arm.spec, architecture="reformer")
        with self.assertRaisesRegex(TypeError, "no declared path"):
            CN.census_architecture(arm)

    def test_too_few_probes_is_refused_rather_than_bootstrapped(self):
        arm = _arm()
        with self.assertRaisesRegex(ValueError, "resampling units"):
            CN.collision_null_census(
                arm, _probes(np.random.default_rng(SEED), n=4), seed=SEED, batch_size=2
            )


if __name__ == "__main__":
    unittest.main()
