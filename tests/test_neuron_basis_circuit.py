"""What the neuron-basis circuit stage must get right, on a CPU stub.

The stage answers F11's open question by asking whether a size-k circuit in the
model's own MLP neuron basis is as faithful on protein as on text. Its result is
only worth anything if three things hold, and each of them can be wrong
*silently*:

**The tensor.** A neuron is a coordinate of the ``d_mlp``-wide activation the
down-projection consumes. The ``d_model``-wide output the projection produces is
a dense mixture of every neuron and is far less sparse, so a circuit measured
there would understate what a sparse basis can do on every arm and would
manufacture a "protein needs a dictionary" conclusion out of our own error. The
stub below is deliberately built with ``n_inner != n_embd``, so a regression to
the output tensor changes a width and is caught.

**The estimand.** ``k = d_mlp`` substitutes every neuron by itself and must
reproduce the clean cross-entropy exactly; ``k = 0`` is the mean-ablated floor
and must recover nothing. And because the down-projection is affine, that floor
must equal the block-output floor ``15_replacement_faithfulness.py`` divides by
-- which is what makes the two stages' ratios the same measurement in different
bases.

**The control.** A random circuit is a control only if it is size-matched to the
circuit it controls.

Everything runs on a randomly initialised two-layer GPT-2 built from a config
object: no GPU, no network, no checkpoint, as ``tests/test_replaceable_arms.py``
and ``tests/test_transfer_core_regressions.py`` do.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts/transfer") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts/transfer"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402

from panel_contract import CAMPAIGN_PANEL  # noqa: E402
from src.transfer import arms as A  # noqa: E402
from src.transfer import replaceable as R  # noqa: E402


def _load_stage(filename: str):
    """Import a stage whose module name starts with a digit."""

    path = REPO / "scripts/transfer" / filename
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


STAGE22 = _load_stage("22_neuron_basis_circuit.py")

#: The stub's shape. ``D_MLP`` is deliberately neither ``D_MODEL`` nor a multiple
#: the implicit rule would produce, so a regression to the MLP *output* tensor
#: (width ``D_MODEL``) and a declaration that silently fell back to ``4 x
#: d_model`` are both visible as a wrong width.
D_MODEL, D_MLP, N_LAYER, N_HEAD, VOCAB = 8, 24, 2, 2, 16

INPUTS = [
    "the harbour opened onto a shallow bay",
    "a compiler translates one language into another",
    "rainfall is concentrated in two short seasons",
    "iron rusts when exposed to oxygen and water",
]


class _StubTokenizer:
    """Enough tokenizer for ``tokenize_batch`` and the unconditioned content mask.

    Id 0 is both the pad and the only special id, which is the GPT-2 lineage's
    own arrangement (its pad token *is* its end-of-text token); every other id is
    a content token.
    """

    all_special_ids = [0]
    unk_token_id = 0
    pad_token_id = 0
    eos_token = "<|endoftext|>"

    def __call__(self, text, return_tensors=None):
        return {"input_ids": [1 + (ord(character) % (VOCAB - 1)) for character in text]}

    def decode(self, ids):
        return "".join(chr(96 + int(i)) for i in ids)


def _spec(*, n_inner: int | None = D_MLP, architecture: str = "gpt2") -> A.ArmSpec:
    return A.ArmSpec(
        name="gpt2",
        path=Path("/nowhere"),
        path_variable="TRANSFER_TEXT_MODEL_BASE_DIR",
        modality="text",
        n_layer=N_LAYER,
        d_model=D_MODEL,
        tokenisation="bpe",
        input_format="raw",
        evaluation_cohort_source="openwebtext",
        architecture=architecture,
    )


def _arm(*, n_inner: int | None = D_MLP, architecture: str = "gpt2", seed: int = 11) -> A.Arm:
    torch.manual_seed(seed)
    config = GPT2Config(
        vocab_size=VOCAB,
        n_positions=64,
        n_embd=D_MODEL,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_inner=n_inner,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    model = GPT2LMHeadModel(config)
    # A randomly initialised MLP contributes so little that mean-ablating it can
    # *lower* the cross-entropy, which would leave the denominator negative and
    # the recovery ratio undefined for reasons that have nothing to do with the
    # code under test. Amplifying the second projection makes the block matter,
    # which is the regime a trained model is in.
    with torch.no_grad():
        for block in model.transformer.h:
            block.mlp.c_proj.weight.mul_(6.0)
    model.eval()
    return A.Arm(
        spec=_spec(n_inner=n_inner, architecture=architecture),
        model=model,
        tokenizer=_StubTokenizer(),
        device="cpu",
        dtype="float32",
    )


def _replaceable(**kwargs) -> R.DenseReplaceable:
    return R.DenseReplaceable(_arm(**kwargs), max_tokens=64)


class NeuronTensorDeclaration(unittest.TestCase):
    """The declaration decides which tensor is read, and it must be explicit."""

    def test_the_hooked_tensor_is_the_pre_down_projection_activation(self):
        model = _replaceable()
        seen: list[tuple[int, int]] = []

        def note(layer, activation):
            seen.append((layer, int(activation.shape[-1])))
            return None

        with STAGE22.neuron_intercept(model, note):
            model.run(model.batch(INPUTS[:2]))

        self.assertEqual(sorted({layer for layer, _ in seen}), list(range(N_LAYER)))
        widths = {width for _, width in seen}
        self.assertEqual(widths, {D_MLP})
        # The regression this pins: the MLP *output* is D_MODEL wide, so a hook
        # that drifted past the down-projection would report that instead.
        self.assertNotIn(D_MODEL, widths)

    def test_the_declared_width_is_the_configs_and_not_the_residual_width(self):
        arm = _arm()
        self.assertEqual(arm.d_mlp, D_MLP)
        self.assertNotEqual(arm.d_mlp, arm.d_model)
        facts = arm.mlp_neuron_facts()
        self.assertEqual(facts["declared_width"], D_MLP)
        self.assertEqual(facts["width_source"], "config.n_inner")
        self.assertEqual(facts["tensor"], "input of mlp.c_proj")

    def test_an_unset_inner_width_resolves_to_the_declared_implicit_multiple(self):
        # Every GPT-2 checkpoint on this panel leaves n_inner unset and means
        # 4 x n_embd by it, so the implicit path is the one the real arms take.
        arm = _arm(n_inner=None)
        self.assertEqual(arm.d_mlp, 4 * D_MODEL)
        self.assertEqual(arm.mlp_neuron_facts()["width_source"], "4 x d_model (n_inner/intermediate_size unset)")

    def test_a_measured_width_that_contradicts_the_declaration_is_refused(self):
        # The regression simulated at the declaration: point the path at the MLP
        # itself, whose input is the d_model-wide residual rather than the
        # d_mlp-wide hidden layer, and the reference pass must refuse it rather
        # than average a tensor of the wrong width.
        model = _replaceable()
        wrong = A.MlpNeuronTensor(
            down_projection=(),
            width_attributes=("n_inner",),
            implicit_width_multiple=4,
            activation_attribute="activation_function",
            activation_lower_bound={"gelu_new": -0.17},
        )
        original = dict(A._MLP_NEURON_TENSOR)
        A._MLP_NEURON_TENSOR["gpt2"] = wrong
        try:
            with self.assertRaises(RuntimeError) as caught:
                STAGE22.neuron_reference(
                    model, model.render(INPUTS), batch_size=2, declared_width=D_MLP
                )
        finally:
            A._MLP_NEURON_TENSOR.clear()
            A._MLP_NEURON_TENSOR.update(original)
        message = str(caught.exception)
        self.assertIn(f"is {D_MODEL} wide", message)
        self.assertIn(str(D_MLP), message)
        # The explanatory refusal must be what fires, not a shape error raised
        # further down by an accumulator of the declared width.
        self.assertIn("down-projection", message)

    def test_an_undeclared_architecture_is_refused_rather_than_duck_typed(self):
        # progen's block does have an mlp with an output projection; resolving it
        # by searching for a plausible attribute is exactly what must not happen,
        # because its parallel residual layout means the tensor is not the same
        # object. Same for the gated rotary lineages.
        for architecture in ("progen", "llama", "qwen2", "t5_decoder", "reformer"):
            with self.assertRaises(TypeError, msg=architecture):
                A.mlp_neuron_declaration(architecture)
        with self.assertRaises(TypeError):
            _arm(architecture="progen").mlp_down_projection(0)

    def test_the_moe_baseline_is_refused_by_name_with_its_reason(self):
        # progen3 is reachable through --arm because the eligible set is composed
        # rather than written down, and it must be refused before a checkpoint is
        # loaded rather than measured on some expert's projection.
        self.assertIn(R.PROGEN3_ARM, R.eligible_arms(CAMPAIGN_PANEL))
        with self.assertRaises(ValueError) as caught:
            STAGE22.declared_architecture(R.PROGEN3_ARM)
        self.assertIn("mixture of experts", str(caught.exception))

    def test_the_matched_triple_resolves_and_is_non_gated(self):
        for name in ("gpt2-large", "protgpt2", "zymctrl"):
            self.assertEqual(STAGE22.declared_architecture(name), "gpt2")
            A.mlp_neuron_declaration("gpt2")
        declared = A.mlp_neuron_declaration("gpt2")
        # Only non-gated spellings are declared, which is what holds the gating
        # confound fixed across the modality contrast.
        self.assertNotIn("silu", declared.activation_lower_bound)
        self.assertLessEqual(declared.activation_lower_bound["gelu_new"], 0.0)

    def test_an_undeclared_nonlinearity_is_refused_rather_than_bounded(self):
        arm = _arm()
        arm.model.config.activation_function = "silu"
        with self.assertRaises(TypeError):
            arm.mlp_neuron_facts()

    def test_a_pre_activation_tensor_fails_the_lower_bound(self):
        # The second half of the pin: a tensor of the right width can still be
        # the wrong one if it is read before the nonlinearity, and a GELU output
        # cannot go below -0.17 while a pre-activation can.
        facts = _arm().mlp_neuron_facts()
        with self.assertRaises(RuntimeError):
            STAGE22.verify_neuron_tensor(
                facts, {"measured_width": D_MLP, "measured_minimum": -3.5}
            )
        record = STAGE22.verify_neuron_tensor(
            facts, {"measured_width": D_MLP, "measured_minimum": -0.169}
        )
        self.assertEqual(record["verdict"], "PASS")
        self.assertTrue(record["width_differs_from_d_model"])

    def test_a_model_without_a_panel_arm_has_no_declaration_to_resolve(self):
        with self.assertRaises(TypeError):
            STAGE22.dense_arm(object())


class CircuitSizing(unittest.TestCase):
    """A control is only a control if it is the size of what it controls."""

    def test_the_selected_and_random_circuits_are_size_matched(self):
        scores = np.random.default_rng(0).normal(size=(N_LAYER, D_MLP))
        for k in (0, 1, 7, D_MLP):
            selected = STAGE22.top_k_mask(scores, k)
            control = STAGE22.random_mask((N_LAYER, D_MLP), k, seed=3)
            self.assertEqual(selected.sum(1).tolist(), [k] * N_LAYER)
            self.assertEqual(control.sum(1).tolist(), [k] * N_LAYER)
            self.assertEqual(selected.shape, control.shape)

    def test_a_control_of_the_wrong_size_is_refused(self):
        keep = torch.zeros((N_LAYER, D_MLP), dtype=torch.bool)
        keep[:, :5] = True
        STAGE22._sized(keep, 5, "a matched control")
        with self.assertRaises(RuntimeError):
            STAGE22._sized(keep, 6, "a mismatched control")

    def test_two_control_seeds_give_two_different_circuits(self):
        first = STAGE22.random_mask((N_LAYER, D_MLP), 6, seed=1)
        again = STAGE22.random_mask((N_LAYER, D_MLP), 6, seed=1)
        other = STAGE22.random_mask((N_LAYER, D_MLP), 6, seed=2)
        self.assertTrue(torch.equal(first, again))
        self.assertFalse(torch.equal(first, other))

    def test_the_grid_is_anchored_at_the_floor_and_the_whole_layer(self):
        grid = STAGE22.circuit_sizes([4, 8], d_mlp=D_MLP)
        self.assertEqual(grid, [0, 4, 8, D_MLP])
        self.assertEqual(STAGE22.circuit_sizes([0, D_MLP], d_mlp=D_MLP), [0, D_MLP])

    def test_a_circuit_larger_than_the_layer_is_refused(self):
        with self.assertRaises(ValueError):
            STAGE22.circuit_sizes([D_MLP + 1], d_mlp=D_MLP)
        with self.assertRaises(ValueError):
            STAGE22.circuit_sizes([-1], d_mlp=D_MLP)

    def test_the_default_grid_is_a_curve_rather_than_a_point(self):
        # The defect the audit's own limitation catalogue names: a single-point
        # ratio. The default must sweep, and must span the range over which a
        # sparse basis either does or does not exist on a 5120-neuron layer.
        self.assertGreaterEqual(len(STAGE22.DEFAULT_CIRCUIT_SIZES), 5)
        self.assertEqual(
            list(STAGE22.DEFAULT_CIRCUIT_SIZES), sorted(STAGE22.DEFAULT_CIRCUIT_SIZES)
        )
        self.assertLess(max(STAGE22.DEFAULT_CIRCUIT_SIZES), 5120)


class EstimandEndpoints(unittest.TestCase):
    """The curve is anchored at both ends by construction; both are measured."""

    @classmethod
    def setUpClass(cls):
        cls.model = _replaceable()
        cls.inputs = cls.model.render(INPUTS)
        cls.reference = STAGE22.neuron_reference(
            cls.model, cls.inputs, batch_size=2, declared_width=D_MLP
        )
        cls.clean = STAGE22.scored_cross_entropy(cls.model, cls.inputs, batch_size=2)
        cls.scores = STAGE22.attribution_scores(
            cls.model,
            cls.inputs,
            neuron_mean=cls.reference["neuron_mean"],
            batch_size=2,
            score="gradient_x_activation",
        )
        cls.picks = STAGE22.bootstrap_indices(len(cls.clean), replicates=200, seed=5)

    def _at(self, k: int) -> np.ndarray:
        return STAGE22.scored_cross_entropy(
            self.model,
            self.inputs,
            batch_size=2,
            factory=STAGE22.circuit_context(
                self.model, STAGE22.top_k_mask(self.scores, k), self.reference["neuron_mean"]
            ),
        )

    def test_keeping_every_neuron_reproduces_the_clean_model_exactly(self):
        full = self._at(D_MLP)
        np.testing.assert_allclose(full, self.clean, rtol=0, atol=0)

    def test_keeping_all_recovers_one_and_keeping_none_recovers_zero(self):
        ablated = self._at(0)
        full = self._at(D_MLP)
        denominator = float(ablated.mean() - self.clean.mean())
        self.assertGreater(
            denominator,
            0.0,
            "the stub's mean ablation must damage the model or the ratio is undefined",
        )
        top = STAGE22.recovery_record(self.clean, full, ablated, picks=self.picks)
        floor = STAGE22.recovery_record(self.clean, ablated, ablated, picks=self.picks)
        self.assertAlmostEqual(top["recovery"], 1.0, places=9)
        self.assertAlmostEqual(floor["recovery"], 0.0, places=9)
        self.assertAlmostEqual(floor["damage_nats_per_token"], denominator, places=9)

    def test_the_neuron_floor_is_the_block_output_floor_stage_15_divides_by(self):
        # The commensurability the whole comparison rests on: the down-projection
        # is affine, so mean-ablating every neuron and mean-ablating the block
        # output are one intervention. If they ever came apart, this stage's
        # ratios would have a different denominator from stage 15's.
        ablated = self._at(0)
        block = STAGE22.scored_cross_entropy(
            self.model,
            self.inputs,
            batch_size=2,
            factory=STAGE22.block_mean_context(
                self.model, self.reference["block_output_mean"]
            ),
        )
        record = STAGE22.endpoints_record(self.clean, ablated, self._at(D_MLP), block)
        self.assertEqual(record["verdict"], "PASS")
        self.assertLess(
            record["neuron_floor_minus_block_floor_fraction_of_denominator"],
            STAGE22.COMMENSURABILITY_TOLERANCE_FRACTION,
        )

    def test_intermediate_circuit_sizes_are_measured_rather_than_interpolated(self):
        # Not a scientific claim: only that intermediate sizes are measured and
        # produce finite numbers, so the artefact carries a curve rather than two
        # endpoints and an interpolation.
        for k in (1, D_MLP // 2):
            record = STAGE22.recovery_record(
                self.clean, self._at(k), self._at(0), picks=self.picks
            )
            self.assertTrue(np.isfinite(record["cross_entropy_nats_per_token"]))
            self.assertTrue(np.isfinite(record["damage_nats_per_token"]))

    def test_every_layer_is_attributed_and_not_only_the_last(self):
        # The regression this pins is silent and total: re-entering each layer's
        # activation as a detached leaf cuts the path from every earlier layer to
        # the loss, so all but the last layer score exactly zero and the
        # "selected" circuit becomes an arbitrary order.
        per_layer = self.scores.sum(axis=1)
        self.assertEqual(self.scores.shape, (N_LAYER, D_MLP))
        for layer, total in enumerate(per_layer):
            self.assertGreater(float(total), 0.0, f"layer {layer} scored nothing")

    def test_the_two_declared_rankings_are_both_available_and_differ(self):
        matched = STAGE22.attribution_scores(
            self.model,
            self.inputs,
            neuron_mean=self.reference["neuron_mean"],
            batch_size=2,
            score="mean_ablation_attribution",
        )
        self.assertEqual(matched.shape, self.scores.shape)
        self.assertFalse(np.allclose(matched, self.scores))
        with self.assertRaises(ValueError):
            STAGE22.attribution_scores(
                self.model,
                self.inputs,
                neuron_mean=self.reference["neuron_mean"],
                batch_size=2,
                score="not-a-score",
            )

    def test_the_reference_means_are_taken_over_content_positions_only(self):
        counted = self.reference["n_content_positions_per_layer"]
        expected = float(
            sum(
                int(self.model.content_mask(self.model.batch([text])).sum())
                for text in self.inputs
            )
        )
        self.assertEqual(counted, [expected] * N_LAYER)


class ArtefactCarriesTheNumerator(unittest.TestCase):
    """Standing rule 27: a ratio whose denominator is not published is not a
    measurement."""

    CLEAN = np.array([2.0, 2.2, 1.8, 2.0])
    ABLATED = np.array([4.0, 4.2, 3.8, 4.0])
    HALF = np.array([3.0, 3.2, 2.8, 3.0])

    def _picks(self):
        return STAGE22.bootstrap_indices(4, replicates=200, seed=7)

    def test_a_curve_point_carries_absolute_nats_beside_its_ratio(self):
        record = STAGE22.recovery_record(
            self.CLEAN, self.HALF, self.ABLATED, picks=self._picks()
        )
        self.assertAlmostEqual(record["recovery"], 0.5)
        self.assertAlmostEqual(record["damage_nats_per_token"], 1.0)
        self.assertAlmostEqual(record["denominator_nats_per_token"], 2.0)
        self.assertAlmostEqual(record["cross_entropy_nats_per_token"], 3.0)
        self.assertIn("damage_interval", record)
        self.assertIsNotNone(record["recovery_interval"])

    def test_the_endpoints_are_reported_in_nats_and_gated_on_both_checks(self):
        record = STAGE22.endpoints_record(
            self.CLEAN, self.ABLATED, self.CLEAN, self.ABLATED
        )
        self.assertEqual(record["verdict"], "PASS")
        self.assertAlmostEqual(record["clean_nats_per_token"], 2.0)
        self.assertAlmostEqual(record["mean_ablated_nats_per_token"], 4.0)
        self.assertAlmostEqual(record["denominator_nats_per_token"], 2.0)
        for key in ("clean_interval", "mean_ablated_interval"):
            self.assertIn("interval", record[key])

    def test_a_shifted_identity_point_fails_the_endpoint_gate(self):
        # k = d_mlp substitutes every neuron by itself, so a measurable
        # difference from clean means the intervention path is not a no-op and
        # every point of the curve is shifted with it.
        record = STAGE22.endpoints_record(
            self.CLEAN, self.ABLATED, self.CLEAN + 0.01, self.ABLATED
        )
        self.assertEqual(record["verdict"], "FAIL")

    def test_a_floor_that_is_not_stage_15s_floor_fails_the_endpoint_gate(self):
        record = STAGE22.endpoints_record(
            self.CLEAN, self.ABLATED, self.CLEAN, self.ABLATED + 1.0
        )
        self.assertEqual(record["verdict"], "FAIL")
        self.assertAlmostEqual(record["neuron_floor_minus_block_floor_nats"], 1.0)
        self.assertAlmostEqual(
            record["neuron_floor_minus_block_floor_fraction_of_denominator"], 0.5
        )

    def test_dtype_rounding_between_the_two_floors_does_not_fail_the_gate(self):
        # The tolerance is sized to absorb the two means being rounded on
        # opposite sides of an affine projection -- 0.72% of the denominator on
        # gpt2 at bfloat16 -- and to catch nothing smaller than a real
        # disagreement about what is being ablated.
        record = STAGE22.endpoints_record(
            self.CLEAN, self.ABLATED, self.CLEAN, self.ABLATED - 0.0144
        )
        self.assertEqual(record["verdict"], "PASS")
        self.assertLess(
            record["neuron_floor_minus_block_floor_fraction_of_denominator"],
            STAGE22.COMMENSURABILITY_TOLERANCE_FRACTION,
        )

    def test_a_non_positive_denominator_withholds_the_ratio_rather_than_dividing(self):
        record = STAGE22.recovery_record(
            self.ABLATED, self.HALF, self.CLEAN, picks=self._picks()
        )
        self.assertIsNone(record["recovery"])
        self.assertIsNone(record["recovery_interval"])
        # And the numerator survives, which is the point: the unnormalised
        # quantity is readable where the ratio is not.
        self.assertAlmostEqual(record["damage_nats_per_token"], -1.0)
        self.assertEqual(
            STAGE22.endpoints_record(
                self.ABLATED, self.CLEAN, self.ABLATED, self.CLEAN
            )["verdict"],
            "FAIL",
        )


class StageWiring(unittest.TestCase):
    def test_the_arm_choices_are_the_composed_eligible_set(self):
        # Read from the argparse declaration rather than trusted: a stage that
        # kept its own tuple is the failure panel_contract exists to end.
        source = Path(STAGE22.__file__).read_text(encoding="utf-8")
        self.assertIn("choices=eligible_arms(CAMPAIGN_PANEL)", source)

    def test_the_cohort_band_and_draw_are_arguments_with_the_declared_defaults(self):
        defaults = {
            action.dest: action.default for action in STAGE22.build_parser()._actions
        }
        self.assertEqual(defaults["protein_min_len"], 64)
        self.assertEqual(defaults["protein_max_len"], 246)
        self.assertEqual(defaults["cohort_draw_seed"], A.DEFAULT_CORPUS_DRAW_SEED)
        self.assertEqual(defaults["text_min_chars"], 800)

    def test_the_stage_is_not_registered_in_the_panel_contract(self):
        import panel_contract

        self.assertNotIn("neuron_basis_circuit", panel_contract.STAGE_CONTRACTS)

    def test_the_provenance_modules_exist_and_include_the_declaration(self):
        self.assertIn("src/transfer/arms.py", STAGE22.PROVENANCE_MODULES)
        for name in STAGE22.PROVENANCE_MODULES:
            self.assertTrue((REPO / name).exists(), name)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
