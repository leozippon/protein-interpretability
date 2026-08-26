"""Conditions EXP-R2-229's joint-mode lens must satisfy however it is invoked.

Three of these guard a way this measurement could be silently wrong rather than
loudly broken, which is the failure mode this repository treats as worst:

* the scored span. A protein window whose target mask does not sit exactly on the
  rendering's residue tokens produces a full trajectory over the wrong symbols.
* the memory schedule. ``blocked_trajectory`` exists so a 48-layer rung of width
  7168 does not need 7.2 GiB of host cache at once; if blocking changed the
  numbers it would be a second estimator wearing the name of a schedule.
* the resampling estimator. The bootstrap recomputes layer rates with a
  vectorised twin of ``lenses.lens_metrics``; the two must agree on the full
  sample or the published point estimate and its interval are two statistics.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from src.transfer import joint_lens as jl
from src.transfer import joint_modes
from src.transfer.arms import (
    INPUT_FORMAT_UNDECLARED,
    Arm,
    ArmSpec,
    Cohort,
    protein_cohort,
)
from src.transfer.lenses import (
    LensHead,
    cache_residuals,
    layer_grid,
    lens_metrics,
    lens_trajectory,
)

MODEL_BASE = Path(os.environ.get("TRANSFER_MODEL_BASE_DIR", "/Data/public/models_R2"))
GALACTICA = MODEL_BASE / "galactica-125m"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _stage():
    path = REPO_ROOT / "scripts" / "transfer" / "47_joint_mode_lens.py"
    spec = importlib.util.spec_from_file_location("_stage_47", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------ a tiny decoder


class _Block(nn.Module):
    """A block whose output is a deterministic, invertible function of its input."""

    def __init__(self, width: int, index: int) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width)
        torch.manual_seed(index)
        nn.init.normal_(self.linear.weight, std=0.1)
        nn.init.zeros_(self.linear.bias)

    def forward(self, hidden, **_kwargs):
        return (hidden + torch.tanh(self.linear(hidden)),)


class _Tiny(nn.Module):
    """The smallest object ``Arm.blocks`` and ``lenses.lens_head`` both resolve."""

    def __init__(self, *, n_layer: int, width: int, vocab: int) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.transformer = nn.Module()
        self.transformer.wte = nn.Embedding(vocab, width)
        self.transformer.h = nn.ModuleList(_Block(width, i) for i in range(n_layer))
        self.transformer.ln_f = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, vocab, bias=False)
        self.config = SimpleNamespace(vocab_size=vocab)

    def forward(self, input_ids=None, attention_mask=None, use_cache=False, **_kwargs):
        hidden = self.transformer.wte(input_ids)
        for block in self.transformer.h:
            hidden = block(hidden)[0]
        return SimpleNamespace(logits=self.lm_head(self.transformer.ln_f(hidden)))


def _tiny_arm(*, n_layer: int = 4, width: int = 8, vocab: int = 11) -> Arm:
    spec = ArmSpec(
        name="tiny",
        path=Path("/nonexistent/tiny"),
        path_variable="TRANSFER_MODEL_BASE_DIR",
        modality="text",
        n_layer=n_layer,
        d_model=width,
        tokenisation="bpe",
        input_format="raw",
        evaluation_cohort_source="synthetic",
        architecture="gpt2",
        capabilities=frozenset({"lens"}),
    )
    model = _Tiny(n_layer=n_layer, width=width, vocab=vocab)
    model.eval()
    return Arm(spec=spec, model=model, tokenizer=None, device="cpu", dtype="float32")


def _tiny_head(arm: Arm) -> LensHead:
    norm = arm.model.transformer.ln_f
    head = arm.model.lm_head
    return LensHead(
        weight=head.weight.detach().float(),
        bias=None,
        norm_weight=norm.weight.detach().float(),
        norm_bias=norm.bias.detach().float(),
        norm_eps=float(norm.eps),
        d_model=arm.d_model,
        vocab_size=int(arm.model.config.vocab_size),
    )


def _tiny_windows(arm: Arm, *, records: int, length: int, batch: int):
    generator = torch.Generator().manual_seed(7)
    vocab = int(arm.model.config.vocab_size)
    windows = []
    for offset in range(0, records, batch):
        rows = min(batch, records - offset)
        ids = torch.randint(0, vocab, (rows, length), generator=generator)
        mask = torch.ones_like(ids)
        target = torch.ones((rows, length - 1), dtype=torch.bool)
        windows.append(
            jl.ScoredWindow(
                input_ids=ids,
                attention_mask=mask,
                target_mask=target,
                sequence_indices=tuple(range(offset, offset + rows)),
            )
        )
    return windows


# ---------------------------------------------------------------- the tests


class DeclarationTests(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(num_hidden_layers=12, hidden_size=768, vocab_size=50000)

    def test_the_protein_declaration_cannot_be_rendered_by_the_panel_renderer(self):
        # The sentinel is not decoration: every renderer in ``arms`` falls through
        # to its own refusal on it, so a caller that reached for
        # ``Cohort.input_strings`` instead of ``joint_lens.protein_windows`` is
        # stopped rather than served a format this checkpoint never saw.
        spec = jl.joint_arm_spec(
            Path("/nonexistent/galactica"),
            name="galactica",
            mode="protein",
            config=self.config,
            architecture="opt",
        )
        self.assertEqual(spec.input_format, INPUT_FORMAT_UNDECLARED)
        arm = Arm(spec=spec, model=None, tokenizer=None, device="cpu", dtype="bfloat16")
        cohort = Cohort(name="c", kind="protein", records=["MKV"], min_symbols=1, max_symbols=9)
        with self.assertRaisesRegex(ValueError, "unsupported input format"):
            cohort.input_strings(arm)

    def test_only_the_lens_capability_is_granted(self):
        # e500d14 declared `opt` in two architecture tables and refused four. A
        # declaration that granted `pathway` or `circuits` would fail inside a
        # hook that emits (batch*token, d_model) instead of being refused here.
        for mode in ("text", "protein"):
            spec = jl.joint_arm_spec(
                Path("/nonexistent/galactica"),
                name="galactica",
                mode=mode,
                config=self.config,
                architecture="opt",
            )
            self.assertEqual(sorted(spec.capabilities), ["lens"])
            self.assertEqual((spec.n_layer, spec.d_model), (12, 768))
            arm = Arm(spec=spec, model=None, tokenizer=None, device="cpu", dtype="bfloat16")
            for capability in ("pathway", "circuits", "relational", "budget"):
                with self.assertRaises(ValueError):
                    arm.require(capability)

    def test_an_undeclared_mode_is_refused(self):
        with self.assertRaisesRegex(ValueError, "unknown joint mode"):
            jl.joint_arm_spec(
                Path("/nonexistent/galactica"),
                name="galactica",
                mode="protein_naive",
                config=self.config,
                architecture="opt",
            )


class BlockingIsAScheduleTests(unittest.TestCase):
    def test_blocking_reproduces_the_whole_cohort_row_set(self):
        arm = _tiny_arm()
        head = _tiny_head(arm)
        windows = _tiny_windows(arm, records=12, length=9, batch=2)
        layers = [point.layer for point in layer_grid(arm.n_layer, (0.0, 0.5, 1.0))]
        whole = lens_trajectory(
            head, cache_residuals(arm, windows, layers, max_bytes=2**30), device="cpu", chunk=64
        )
        blocked = jl.blocked_trajectory(
            arm, head, windows, layers, block_windows=2, metric_chunk=64, max_bytes=2**30
        )
        for layer in layers:
            self.assertEqual(len(whole[layer]), len(blocked[layer]))
            for reference, produced in zip(whole[layer], blocked[layer]):
                for key in reference:
                    self.assertAlmostEqual(
                        float(reference[key]), float(produced[key]), places=9, msg=f"{layer}/{key}"
                    )

    def test_a_grid_layer_that_loses_records_is_refused(self):
        with self.assertRaisesRegex(ValueError, "block_windows must be positive"):
            jl.blocked_trajectory(
                _tiny_arm(), None, [], [0], block_windows=0, metric_chunk=1, max_bytes=1
            )


class ResamplingEstimatorTests(unittest.TestCase):
    def _rows(self, n: int, *, kl: float, agreement: float):
        return [
            {
                "token_count": 5 + index,
                "ce_sum": 1.5 * (5 + index),
                "kl_sum": kl * (5 + index),
                "agreement_count": agreement * (5 + index),
                "entropy_sum": 2.0 * (5 + index),
            }
            for index in range(n)
        ]

    def test_the_vectorised_twin_agrees_with_the_published_estimator(self):
        rows = self._rows(9, kl=0.7, agreement=0.4)
        published = lens_metrics(rows)
        vectorised = jl._weighted_means(jl._row_matrix(rows), np.arange(len(rows)))
        self.assertAlmostEqual(published["ce_nats"], vectorised["ce_nats"], places=12)
        self.assertAlmostEqual(
            published["kl_to_final_nats"], vectorised[jl.KL_QUANTITY], places=12
        )
        self.assertAlmostEqual(
            published["top1_agreement_with_final"],
            vectorised[jl.AGREEMENT_QUANTITY],
            places=12,
        )
        self.assertAlmostEqual(
            1.0 - published["top1_agreement_with_final"],
            vectorised[jl.DISAGREEMENT_QUANTITY],
            places=12,
        )

    def test_below_the_unit_floor_no_draw_is_taken_and_every_interval_is_refused(self):
        rows = {0: self._rows(7, kl=0.9, agreement=0.1), 1: self._rows(7, kl=0.0, agreement=1.0)}
        out = jl.depth_bootstrap(
            rows, [0.5, 1.0], [0, 1], levels=[0.5], taus=[0.5], resamples=16, seed=1
        )
        self.assertTrue(out["degenerate"])
        self.assertIn("below the 8-unit floor", out["degenerate_reason"])
        for key, interval in out["intervals"].items():
            self.assertTrue(interval["refused"], key)
            self.assertIsNone(interval["q025"], key)

    def test_at_the_floor_an_interval_is_produced_and_bracketed(self):
        rows = {0: self._rows(8, kl=0.9, agreement=0.1), 1: self._rows(8, kl=0.0, agreement=1.0)}
        out = jl.depth_bootstrap(
            rows, [0.5, 1.0], [0, 1], levels=[0.5], taus=[0.5], resamples=64, seed=1
        )
        self.assertFalse(out["degenerate"])
        for key in (jl.agreement_key(0.5), jl.span_key(jl.KL_QUANTITY, 0.5)):
            interval = out["intervals"][key]
            self.assertFalse(interval["refused"], key)
            self.assertLessEqual(interval["q025"], interval["median"])
            self.assertLessEqual(interval["median"], interval["q975"])

    def test_an_agreement_level_is_never_an_undefined_draw(self):
        # The primary statistic is always defined because the deepest grid point
        # is the model itself, where agreement is exactly one. That is what keeps
        # MAX_UNDEFINED_DRAW_FRACTION off the primary and on the span family.
        rows = {0: self._rows(8, kl=0.9, agreement=0.1), 1: self._rows(8, kl=0.0, agreement=1.0)}
        out = jl.depth_bootstrap(
            rows, [0.5, 1.0], [0, 1], levels=[0.25, 0.5, 0.75], taus=[0.5],
            resamples=64, seed=3,
        )
        for level in (0.25, 0.5, 0.75):
            self.assertEqual(out["intervals"][jl.agreement_key(level)]["n_undefined_draws"], 0)

    def test_a_draw_set_thinned_past_the_cap_is_refused_rather_than_thinned_quietly(self):
        cap = jl.MAX_UNDEFINED_DRAW_FRACTION
        just_inside = [0.4] * 96 + [None] * 4
        just_outside = [0.4] * 90 + [None] * 10
        self.assertLessEqual(4 / 100, cap)
        self.assertGreater(10 / 100, cap)
        self.assertFalse(jl._draw_interval(just_inside)["refused"])
        refused = jl._draw_interval(just_outside)
        self.assertTrue(refused["refused"])
        self.assertEqual(refused["n_undefined_draws"], 10)
        self.assertIsNone(refused["q975"])


class LevelDepthTests(unittest.TestCase):
    def test_an_absolute_level_needs_no_normaliser(self):
        depths = [0.25, 0.5, 0.75, 1.0]
        self.assertAlmostEqual(
            jl.level_depth(depths, [0.0, 0.2, 0.4, 1.0], 0.3), 0.625, places=12
        )
        # Two trajectories with the same shape but different starting values reach
        # the same absolute level at different depths, which a span-normalised
        # depth would hide by dividing each by its own start.
        self.assertNotAlmostEqual(
            jl.level_depth(depths, [0.0, 0.2, 0.4, 1.0], 0.5),
            jl.level_depth(depths, [0.4, 0.5, 0.6, 1.0], 0.5),
        )

    def test_a_trajectory_already_above_the_level_returns_the_shallowest_depth(self):
        self.assertEqual(jl.level_depth([0.2, 1.0], [0.9, 1.0], 0.5), 0.2)

    def test_the_first_crossing_is_taken_on_a_non_monotone_trajectory(self):
        depths = [0.2, 0.4, 0.6, 0.8, 1.0]
        self.assertAlmostEqual(
            jl.level_depth(depths, [0.0, 0.6, 0.1, 0.2, 1.0], 0.5), 0.3667, places=3
        )

    def test_a_trajectory_that_never_reaches_the_level_is_refused(self):
        with self.assertRaisesRegex(ValueError, "never reached"):
            jl.level_depth([0.5, 1.0], [0.0, 0.4], 0.5)

    def test_a_level_outside_the_open_unit_interval_is_refused(self):
        for level in (0.0, 1.0, -0.1, 1.2):
            with self.assertRaises(ValueError):
                jl.level_depth([0.5, 1.0], [0.0, 1.0], level)


class GateTests(unittest.TestCase):
    def _interval(self, sign, *, excludes=True, refused=False):
        low, high = (0.05, 0.20) if sign > 0 else (-0.20, -0.05)
        if not excludes:
            low, high = -0.10, 0.20
        return {
            "refused": refused,
            "reason": None,
            "n_defined_draws": 100,
            "n_undefined_draws": 0,
            "q025": low,
            "median": (low + high) / 2,
            "q975": high,
            "point": 0.1 * sign,
            "excludes_zero": bool(excludes and not refused),
            "sign": sign,
        }

    def _contrast(self, level_signs, *, second_sign=1, excludes=True, refused=False):
        contrast = {
            jl.agreement_key(level): self._interval(sign, excludes=excludes, refused=refused)
            for level, sign in level_signs.items()
        }
        contrast[jl.SECOND_FUNCTIONAL_KEY] = self._interval(second_sign)
        return contrast

    def test_all_three_clauses_together_are_a_separation(self):
        gate = jl.mode_gate(self._contrast({0.25: 1, 0.50: 1, 0.75: 1}))
        self.assertEqual(gate["verdict"], "modality_depth_separation")
        self.assertEqual(gate["direction"], "protein_resolves_deeper")

    def test_the_other_direction_is_reachable(self):
        gate = jl.mode_gate(self._contrast({0.25: -1, 0.50: -1, 0.75: -1}, second_sign=-1))
        self.assertEqual(gate["verdict"], "modality_depth_separation")
        self.assertEqual(gate["direction"], "protein_resolves_shallower")

    def test_a_sign_that_flips_across_the_sweep_is_not_one_ordering(self):
        # Appendix B rule 17: a reading that holds at one level and reverses at
        # another is a threshold result, and every interval excluding zero does
        # not make it an ordering. This clause is not decorative -- the instrument
        # rung's span family flips sign across its own sweep.
        gate = jl.mode_gate(self._contrast({0.25: 1, 0.50: 1, 0.75: -1}))
        self.assertEqual(gate["verdict"], "no_modality_depth_separation")
        self.assertFalse(gate["clauses"]["agreement_sign_invariant_across_the_level_sweep"])

    def test_an_interval_straddling_zero_at_one_level_defeats_the_compound(self):
        gate = jl.mode_gate(self._contrast({0.25: 1, 0.50: 1, 0.75: 1}, excludes=False))
        self.assertEqual(gate["verdict"], "no_modality_depth_separation")
        self.assertFalse(gate["clauses"]["agreement_interval_excludes_zero_at_every_level"])

    def test_the_second_functional_must_agree(self):
        gate = jl.mode_gate(self._contrast({0.25: 1, 0.50: 1, 0.75: 1}, second_sign=-1))
        self.assertEqual(gate["verdict"], "no_modality_depth_separation")
        self.assertFalse(gate["clauses"]["kl_span_depth_agrees_at_tau_0.50"])

    def test_a_refused_interval_refuses_the_gate_rather_than_failing_it(self):
        gate = jl.mode_gate(self._contrast({0.25: 1, 0.50: 1, 0.75: 1}, refused=True))
        self.assertEqual(gate["verdict"], "refused")
        self.assertIsNone(gate["clauses"])

    def test_a_contrast_missing_a_gated_statistic_is_refused_loudly(self):
        contrast = self._contrast({0.25: 1, 0.50: 1, 0.75: 1})
        contrast.pop(jl.SECOND_FUNCTIONAL_KEY)
        with self.assertRaises(KeyError):
            jl.mode_gate(contrast)

    def test_the_verdict_vocabulary_is_closed(self):
        self.assertEqual(
            set(jl.VERDICTS),
            {"modality_depth_separation", "no_modality_depth_separation", "refused"},
        )


class ContrastTests(unittest.TestCase):
    KEY = jl.agreement_key(0.50)

    def _side(self, values, resamples=4):
        return {"resamples": resamples, "draws": {self.KEY: list(values)}}

    def _point(self, value):
        return {self.KEY: value}

    def test_a_difference_is_formed_draw_by_draw(self):
        contrast = jl.depth_contrast(
            self._side([0.9, 0.8, 0.85, 0.95]),
            self._side([0.3, 0.2, 0.25, 0.35]),
            point_protein=self._point(0.875),
            point_text=self._point(0.275),
        )
        interval = contrast[self.KEY]
        self.assertAlmostEqual(interval["point"], 0.6, places=12)
        self.assertTrue(interval["excludes_zero"])
        self.assertEqual(interval["sign"], 1)

    def test_a_draw_undefined_on_either_side_contributes_nothing(self):
        contrast = jl.depth_contrast(
            self._side([0.9, None, 0.85, 0.95]),
            self._side([0.3, 0.2, None, 0.35]),
            point_protein=self._point(0.9),
            point_text=self._point(0.3),
        )
        interval = contrast[self.KEY]
        self.assertEqual(interval["n_undefined_draws"], 2)
        self.assertTrue(interval["refused"])

    def test_two_sides_resampled_differently_cannot_be_contrasted(self):
        with self.assertRaisesRegex(ValueError, "different number of times"):
            jl.depth_contrast(
                self._side([0.9], resamples=4),
                self._side([0.3], resamples=8),
                point_protein=self._point(0.9),
                point_text=self._point(0.3),
            )

    def test_two_sides_publishing_different_statistics_cannot_be_contrasted(self):
        left = self._side([0.9])
        right = {"resamples": 4, "draws": {"something_else": [0.3]}}
        with self.assertRaisesRegex(ValueError, "different depth statistics"):
            jl.depth_contrast(
                left, right, point_protein=self._point(0.9), point_text=self._point(0.3)
            )


class TrajectoryRecordTests(unittest.TestCase):
    def _metrics(self, kl_last: float):
        return {
            0: {
                "kl_to_final_nats": 5.0,
                jl.DISAGREEMENT_QUANTITY: 0.9,
                "ce_nats": 6.0,
                "entropy_nats": 7.0,
                "top1_agreement_with_final": 0.1,
                "scored_tokens": 100,
                "sequences": 10,
            },
            1: {
                "kl_to_final_nats": kl_last,
                jl.DISAGREEMENT_QUANTITY: 0.0,
                "ce_nats": 2.0,
                "entropy_nats": 1.0,
                "top1_agreement_with_final": 1.0,
                "scored_tokens": 100,
                "sequences": 10,
            },
        }

    def test_the_deepest_grid_point_must_be_the_model_itself(self):
        grid = layer_grid(2, (0.0, 1.0))
        record = jl.trajectory_record(grid, self._metrics(0.0))
        self.assertTrue(record["falls_across_the_grid"][jl.KL_QUANTITY])
        self.assertTrue(record["kl_monotone_non_increasing_with_depth"])
        self.assertTrue(record["agreement_monotone_non_decreasing_with_depth"])
        self.assertEqual(record["agreement_at_shallowest_grid_point"], 0.1)
        with self.assertRaisesRegex(FloatingPointError, "KL to itself must be zero"):
            jl.trajectory_record(grid, self._metrics(0.4))

    def test_a_final_layer_that_does_not_agree_with_itself_is_refused(self):
        grid = layer_grid(2, (0.0, 1.0))
        metrics = self._metrics(0.0)
        metrics[1]["top1_agreement_with_final"] = 0.98
        with self.assertRaisesRegex(FloatingPointError, "agree with itself"):
            jl.trajectory_record(grid, metrics)


@unittest.skipUnless(GALACTICA.is_dir(), "galactica-125m is not staged on this host")
class RenderedSpanTests(unittest.TestCase):
    """The scored span, checked against the rendering on the real tokenizer."""

    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained(str(GALACTICA))
        if cls.tokenizer.pad_token_id is None:
            cls.tokenizer.pad_token = "<pad>"
        cls.tokenisation = joint_modes.resolve(cls.tokenizer, "galactica")
        cls.cohort = protein_cohort(6, 64, 246, name="protein_scored", seed=20260728)
        spec = jl.joint_arm_spec(
            GALACTICA,
            name="galactica-125m",
            mode="protein",
            config=SimpleNamespace(num_hidden_layers=12, hidden_size=768),
            architecture="opt",
        )
        cls.arm = Arm(
            spec=spec, model=None, tokenizer=cls.tokenizer, device="cpu", dtype="bfloat16"
        )

    def test_every_masked_target_is_a_declared_residue_token(self):
        built = jl.protein_windows(
            self.arm,
            self.tokenisation,
            self.cohort,
            protein_context=None,
            variant=joint_modes.DECLARED,
            batch_size=3,
        )
        residues = set(int(value) for value in self.tokenisation.residue_ids.values())
        seen = 0
        for window in built.windows:
            targets = window.input_ids[:, 1:][window.target_mask]
            seen += int(targets.numel())
            self.assertTrue(set(int(value) for value in targets) <= residues)
        self.assertEqual(seen, sum(len(record) for record in self.cohort.records))
        self.assertEqual(built.census["residues_per_scored_token"], 1.0)
        self.assertTrue(built.census["one_token_per_residue"])

    def test_the_naive_control_scores_merged_pieces_and_says_so(self):
        built = jl.protein_windows(
            self.arm,
            self.tokenisation,
            self.cohort,
            protein_context=None,
            variant=joint_modes.NAIVE,
            batch_size=3,
        )
        self.assertGreater(built.census["residues_per_scored_token"], 1.5)
        self.assertFalse(built.census["one_token_per_residue"])
        self.assertLess(built.census["n_scored_positions"], built.census["n_residues"])

    def test_the_position_cap_keeps_only_targets_the_text_window_can_match(self):
        capped = jl.protein_windows(
            self.arm,
            self.tokenisation,
            self.cohort,
            protein_context=None,
            variant=joint_modes.DECLARED,
            batch_size=3,
            position_cap=jl.POSITION_CAP,
        )
        for window in capped.windows:
            columns = torch.nonzero(window.target_mask, as_tuple=False)[:, 1]
            self.assertLessEqual(int(columns.max()) + 1, jl.POSITION_CAP)
        self.assertLess(capped.census["n_scored_positions"], capped.census["n_residues"])

    def test_a_text_cohort_is_refused(self):
        text = Cohort(name="t", kind="text", records=["hello"], min_symbols=1, max_symbols=0)
        with self.assertRaisesRegex(ValueError, "protein windows need a protein cohort"):
            jl.protein_windows(
                self.arm,
                self.tokenisation,
                text,
                protein_context=None,
                variant=joint_modes.DECLARED,
                batch_size=1,
            )

    def test_a_cap_that_leaves_a_record_with_no_target_is_refused(self):
        with self.assertRaisesRegex(ValueError, "no scored target at or below"):
            jl.protein_windows(
                self.arm,
                self.tokenisation,
                self.cohort,
                protein_context=None,
                variant=joint_modes.DECLARED,
                batch_size=3,
                position_cap=0,
            )


class IdentificationVerdictTests(unittest.TestCase):
    """The verdict is READ from stage 41's report, so a report it cannot read must stop.

    Restating the sign rule here would be a second declaration of the criterion
    that decides whether a protein reading may be taken at all (Appendix B rule
    12), which is why the stage parses the report instead -- and why a report
    that does not carry exactly one row for the arm, or exactly one block under
    it, has to raise rather than resolve to whichever row came first.
    """

    def setUp(self):
        self.stage = _stage()

    def _report(self, rows, path: Path):
        path.write_text(
            json.dumps(
                {
                    "summary": {"arms": rows},
                    "metadata": {"configuration": {"sidecar": ["s.npz"], "out": "o"}},
                }
            ),
            encoding="utf-8",
        )
        return path

    def _row(self, arm, sign, blocks=1):
        return {
            "arm": arm,
            "blocks": [
                {
                    "context_information_nats": 0.2,
                    "bootstrap_ci_95": [0.1, 0.3],
                    "cohort_digest": "abc",
                    "sign_status": sign,
                    "screening_status": "PASS",
                }
            ]
            * blocks,
        }

    def test_the_sign_rule_is_read_and_not_restated(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = self._report(
                [self._row("protein_declared", "PASS"), self._row("protein_reversed", "FAIL")],
                Path(tmp) / "r.json",
            )
            verdict = self.stage.identification_verdict(path, "protein_declared")
            self.assertTrue(verdict["identified"])
            self.assertEqual(verdict["cohort_digest"], "abc")
            self.assertIsNotNone(verdict["report_sha256"])
            self.assertEqual(verdict["report_inputs"]["sidecar"], ["s.npz"])

            failing = self._report([self._row("protein_declared", "FAIL")], Path(tmp) / "f.json")
            self.assertFalse(
                self.stage.identification_verdict(failing, "protein_declared")["identified"]
            )

    def test_a_report_without_exactly_one_row_for_the_arm_is_refused(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            absent = self._report([self._row("text_declared", "PASS")], Path(tmp) / "a.json")
            with self.assertRaisesRegex(ValueError, "rows name the arm"):
                self.stage.identification_verdict(absent, "protein_declared")
            twice = self._report(
                [self._row("protein_declared", "PASS"), self._row("protein_declared", "FAIL")],
                Path(tmp) / "t.json",
            )
            with self.assertRaisesRegex(ValueError, "rows name the arm"):
                self.stage.identification_verdict(twice, "protein_declared")

    def test_a_multi_block_row_is_refused_rather_than_reduced(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = self._report(
                [self._row("protein_declared", "PASS", blocks=2)], Path(tmp) / "b.json"
            )
            with self.assertRaisesRegex(ValueError, "blocks; one is required"):
                self.stage.identification_verdict(path, "protein_declared")


class StageArgumentTests(unittest.TestCase):
    def setUp(self):
        self.stage = _stage()

    def _args(self, **overrides):
        parser = self.stage.build_parser()
        namespace = parser.parse_args(["--checkpoint", "/nonexistent"])
        for key, value in overrides.items():
            setattr(namespace, key, value)
        return namespace

    def test_the_gate_stage_refuses_a_checkpoint_and_needs_its_two_inputs(self):
        with self.assertRaisesRegex(ValueError, "loads no checkpoint"):
            self.stage.validate(
                self._args(stage="gate", artefact=Path("a"), identification=Path("i"))
            )
        with self.assertRaisesRegex(ValueError, "--stage gate needs"):
            self.stage.validate(
                self._args(stage="gate", checkpoint=None, artefact=None, identification=None)
            )

    def test_the_measure_stage_needs_a_checkpoint_and_refuses_an_artefact(self):
        with self.assertRaisesRegex(ValueError, "--stage measure needs --checkpoint"):
            self.stage.validate(self._args(checkpoint=None))
        with self.assertRaisesRegex(ValueError, "belongs to --stage gate"):
            self.stage.validate(self._args(artefact=Path("a")))

    def test_a_cohort_below_the_bootstrap_unit_floor_is_refused_before_a_load(self):
        with self.assertRaisesRegex(ValueError, "bootstrap unit floor"):
            self.stage.validate(self._args(sequences=7))

    def test_the_frozen_defaults_are_the_qualification_draw(self):
        args = self._args()
        self.assertEqual(args.sequences, 128)
        self.assertEqual((args.protein_min_len, args.protein_max_len), (64, 246))
        self.assertEqual(args.text_min_chars, 800)
        self.assertEqual(args.cohort_draw_seed, 20260728)
        self.assertEqual(args.text_window_tokens, 164)
        self.assertEqual(args.position_cap, 163)
        self.assertEqual(args.bootstrap_resamples, 2000)
        self.assertEqual(args.bootstrap_seed, 20260826)
        self.assertEqual(args.dtype, "bfloat16")
        self.assertEqual(args.protein_context, None)


if __name__ == "__main__":
    unittest.main()
