"""The ProGen3-112M loader, tested where it can actually go wrong.

``ProGen3ForCausalLM.from_pretrained(path, moe_implementation="eager")`` returns
a model whose every expert and router is random, without raising: the released
weights are in megablocks packing and carry no key the eager block recognises.
``src.transfer.progen3`` exists to make that impossible, and the three things
that could let it back in are tested here.

**The mapping.** A wrong expert mapping still produces a state dict that loads
with ``strict=True``, so nothing structural catches it. Two tests cover it: the
conversion is exactly invertible, and the converted per-expert weights compute
the same function as the packed ones under the megablocks reference math -- and
stop doing so the moment the gate and up projections are exchanged.

**The load.** A missing or unexpected key must abort, because a partial load of
this checkpoint is silent and plausible.

**The band.** The numerical tripwire is only worth its cost if it separates the
correct mapping from every corruption that survives ``strict=True``, which is a
property of the declared band and the recorded measurements, checkable without a
GPU or the 460 MB checkpoint. The measurements themselves come from
``external_resources/baselines/progen3_eager_probe`` and from the band
measurement recorded in ``src.transfer.progen3``.

Nothing here loads the real checkpoint, imports the third-party ``progen3``
package, or needs a GPU.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer.progen3 import (  # noqa: E402
    MEASURED_SELF_CHECK_NLL,
    SELF_CHECK_NLL_BAND,
    Component,
    ProGen3,
    ablated,
    check_nll,
    components,
    convert_megablocks_state_dict,
    moe_intercept,
    strict_load,
)

EXPERTS = 3
FFN = 4
HIDDEN = 5
TOP_K = 2


def tiny_checkpoint(seed: int = 0) -> dict[str, torch.Tensor]:
    """A megablocks-packed checkpoint the size of a unit test."""

    generator = torch.Generator().manual_seed(seed)

    def normal(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=torch.float64)

    return {
        "model.embed_tokens.weight": normal(7, HIDDEN),
        "model.layers.0.block_sparse_moe.experts.mlp.w1": normal(EXPERTS * FFN, HIDDEN),
        "model.layers.0.block_sparse_moe.experts.mlp.v1": normal(EXPERTS * FFN, HIDDEN),
        "model.layers.0.block_sparse_moe.experts.mlp.w2": normal(EXPERTS * FFN, HIDDEN),
        "model.layers.0.block_sparse_moe.router.layer.weight": normal(EXPERTS, HIDDEN),
        "mlm_head.weight": normal(7, HIDDEN),
    }


def convert(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return convert_megablocks_state_dict(
        raw, num_experts=EXPERTS, intermediate_size=FFN
    )


class TheMegablocksConversionIsExactlyInvertible(unittest.TestCase):
    """Repacking the converted tensors must give the released ones back."""

    def setUp(self) -> None:
        self.raw = tiny_checkpoint()
        self.converted = convert(self.raw)

    def test_every_packed_expert_tensor_is_recoverable(self):
        prefix = "model.layers.0.block_sparse_moe"
        for packed, target in (("w1", "w1"), ("v1", "w3"), ("w2", "w2")):
            slices = []
            for expert in range(EXPERTS):
                stored = self.converted[f"{prefix}.experts.{expert}.{target}.weight"]
                # w2 is the only transposed one, and it is the only one whose
                # stored shape is (hidden, ffn) rather than (ffn, hidden).
                expected_shape = (HIDDEN, FFN) if packed == "w2" else (FFN, HIDDEN)
                self.assertEqual(tuple(stored.shape), expected_shape, target)
                slices.append(stored.T if packed == "w2" else stored)
            repacked = torch.cat(slices, dim=0)
            self.assertTrue(
                torch.equal(repacked, self.raw[f"{prefix}.experts.mlp.{packed}"]),
                f"{packed} does not survive the round trip",
            )

    def test_the_router_is_renamed_and_the_mlm_head_is_dropped(self):
        prefix = "model.layers.0.block_sparse_moe"
        self.assertTrue(
            torch.equal(
                self.converted[f"{prefix}.gate.weight"],
                self.raw[f"{prefix}.router.layer.weight"],
            )
        )
        self.assertNotIn(f"{prefix}.router.layer.weight", self.converted)
        self.assertNotIn("mlm_head.weight", self.converted)

    def test_everything_outside_the_moe_is_passed_through_untouched(self):
        self.assertTrue(
            torch.equal(
                self.converted["model.embed_tokens.weight"],
                self.raw["model.embed_tokens.weight"],
            )
        )

    def test_a_packed_row_count_that_contradicts_the_config_is_refused(self):
        # The split is the one place a config/checkpoint disagreement turns into
        # silently wrong slices rather than an error.
        with self.assertRaises(ValueError) as caught:
            convert_megablocks_state_dict(
                self.raw, num_experts=EXPERTS + 1, intermediate_size=FFN
            )
        self.assertIn("packed row count", str(caught.exception))

    def test_an_undeclared_packed_expert_tensor_is_refused(self):
        raw = dict(self.raw)
        raw["model.layers.0.block_sparse_moe.experts.mlp.w9"] = torch.zeros(
            EXPERTS * FFN, HIDDEN
        )
        with self.assertRaises(ValueError) as caught:
            convert(raw)
        self.assertIn("no declared target", str(caught.exception))


class TheConvertedWeightsComputeTheMegablocksFunction(unittest.TestCase):
    """Invertibility is not enough: the split must land on the right projections.

    ``w1`` is the gate and ``v1`` is the up projection. Exchanging them
    round-trips perfectly, loads with ``strict=True``, and scores 3.18
    nats/token against 2.29 -- so only the arithmetic separates them.
    """

    def setUp(self) -> None:
        self.raw = tiny_checkpoint(seed=3)
        self.x = torch.randn(6, HIDDEN, generator=torch.Generator().manual_seed(9)).double()

    def packed_reference(self, raw: dict[str, torch.Tensor]) -> torch.Tensor:
        """megablocks 0.7.0: LearnedRouter + MemoryOptimizedGroupedGLU, run densely."""

        prefix = "model.layers.0.block_sparse_moe"
        w1 = raw[f"{prefix}.experts.mlp.w1"].view(EXPERTS, FFN, HIDDEN)
        v1 = raw[f"{prefix}.experts.mlp.v1"].view(EXPERTS, FFN, HIDDEN)
        w2 = raw[f"{prefix}.experts.mlp.w2"].view(EXPERTS, FFN, HIDDEN)
        scores = F.linear(self.x, raw[f"{prefix}.router.layer.weight"]).softmax(-1)
        weights, chosen = torch.topk(scores, TOP_K, dim=-1)
        weights = weights / weights.norm(p=1, dim=-1, keepdim=True)
        out = torch.zeros_like(self.x)
        for expert in range(EXPERTS):
            for slot in range(TOP_K):
                selected = chosen[:, slot] == expert
                if not selected.any():
                    continue
                rows = self.x[selected]
                activation = F.silu(rows @ w1[expert].T) * (rows @ v1[expert].T)
                out[selected] += (activation @ w2[expert]) * weights[selected, slot, None]
        return out

    def eager_from_converted(self, converted: dict[str, torch.Tensor]) -> torch.Tensor:
        """The same math through the eager ``GLUMLP``: w2(silu(w1(x)) * w3(x))."""

        prefix = "model.layers.0.block_sparse_moe"
        scores = F.linear(self.x, converted[f"{prefix}.gate.weight"]).softmax(-1)
        weights, chosen = torch.topk(scores, TOP_K, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        out = torch.zeros_like(self.x)
        for expert in range(EXPERTS):
            gate = converted[f"{prefix}.experts.{expert}.w1.weight"]
            up = converted[f"{prefix}.experts.{expert}.w3.weight"]
            down = converted[f"{prefix}.experts.{expert}.w2.weight"]
            for slot in range(TOP_K):
                selected = chosen[:, slot] == expert
                if not selected.any():
                    continue
                rows = self.x[selected]
                activation = F.silu(F.linear(rows, gate)) * F.linear(rows, up)
                out[selected] += F.linear(activation, down) * weights[selected, slot, None]
        return out

    def test_the_split_experts_reproduce_the_packed_grouped_glu(self):
        self.assertTrue(
            torch.allclose(
                self.eager_from_converted(convert(self.raw)),
                self.packed_reference(self.raw),
                atol=1e-12,
            )
        )

    def test_exchanging_the_gate_and_up_projections_changes_the_function(self):
        swapped = {}
        for key, value in self.raw.items():
            if key.endswith(".experts.mlp.w1"):
                swapped[key.replace(".mlp.w1", ".mlp.v1")] = value
            elif key.endswith(".experts.mlp.v1"):
                swapped[key.replace(".mlp.v1", ".mlp.w1")] = value
            else:
                swapped[key] = value
        corrupted = convert(swapped)
        # It round-trips and keeps every key, which is why strict loading cannot
        # see it...
        self.assertEqual(set(corrupted), set(convert(self.raw)))
        # ...and computes a different function, which is why the NLL can.
        self.assertFalse(
            torch.allclose(
                self.eager_from_converted(corrupted),
                self.packed_reference(self.raw),
                atol=1e-6,
            )
        )


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(3, 2, bias=False)
        self.second = nn.Linear(2, 3, bias=False)


class APartialLoadIsRefusedRatherThanRun(unittest.TestCase):
    """The failure this module exists to prevent is a load that half-works."""

    def test_a_complete_state_dict_loads(self):
        model = _TinyModel()
        state = {key: torch.ones_like(value) for key, value in model.state_dict().items()}
        strict_load(model, state)
        self.assertTrue(torch.equal(model.first.weight, torch.ones(2, 3)))

    def test_a_missing_key_raises_and_names_it(self):
        model = _TinyModel()
        state = {key: torch.ones_like(value) for key, value in model.state_dict().items()}
        del state["second.weight"]
        with self.assertRaises(RuntimeError) as caught:
            strict_load(model, state)
        message = str(caught.exception)
        self.assertIn("second.weight", message)
        self.assertIn("random initialisation", message)

    def test_an_unexpected_key_raises_and_names_it(self):
        model = _TinyModel()
        state = {key: torch.ones_like(value) for key, value in model.state_dict().items()}
        state["third.weight"] = torch.zeros(2, 2)
        with self.assertRaises(RuntimeError) as caught:
            strict_load(model, state)
        self.assertIn("third.weight", str(caught.exception))

    def test_a_refused_load_leaves_the_model_untouched(self):
        # A refusal that had already written half the tensors would leave the
        # caller holding exactly the object this module refuses to produce --
        # which is what happens if the key check runs after the assignment.
        model = _TinyModel()
        before = model.first.weight.clone()
        with self.assertRaises(RuntimeError):
            strict_load(model, {"first.weight": torch.ones(2, 3)})
        self.assertTrue(torch.equal(model.first.weight, before))


class TheSelfCheckBandSeparatesTheMappingsThatStrictLoadingCannot(unittest.TestCase):
    """The band is the only thing standing between a wrong mapping and a result.

    Checked against the recorded measurements rather than by running the model:
    the corrupted variants need the real checkpoint and a GPU, and the numbers
    they produced are the evidence the band was chosen from.
    """

    def test_the_measured_correct_mapping_is_accepted(self):
        record = check_nll(MEASURED_SELF_CHECK_NLL["correct_mapping"])
        self.assertEqual(record["verdict"], "PASS")

    def test_every_measured_corruption_is_rejected(self):
        for name, value in MEASURED_SELF_CHECK_NLL.items():
            if name == "correct_mapping":
                continue
            with self.subTest(corruption=name):
                with self.assertRaises(RuntimeError) as caught:
                    check_nll(value)
                self.assertIn("outside the declared band", str(caught.exception))

    def test_the_band_clears_the_nearest_corruption(self):
        # A band that touched the nearest corruption would pass a broken mapping
        # on any environment that shifted the number slightly.
        nearest = min(
            value
            for name, value in MEASURED_SELF_CHECK_NLL.items()
            if name != "correct_mapping"
        )
        self.assertLess(SELF_CHECK_NLL_BAND[1], nearest)
        self.assertGreater(nearest - SELF_CHECK_NLL_BAND[1], 0.25)

    def test_the_band_brackets_the_measured_value_symmetrically_enough_to_survive(self):
        measured = MEASURED_SELF_CHECK_NLL["correct_mapping"]
        low, high = SELF_CHECK_NLL_BAND
        self.assertLess(low, measured)
        self.assertLess(measured, high)
        # The observed spread across dtype, batch size and scoring direction is
        # 0.005 nats; anything under 0.05 would be a tripwire for the hardware.
        self.assertGreater(min(measured - low, high - measured), 0.05)

    def test_a_value_inside_the_band_is_reported_with_its_evidence(self):
        record = check_nll(MEASURED_SELF_CHECK_NLL["correct_mapping"])
        self.assertEqual(record["reference"], MEASURED_SELF_CHECK_NLL)
        self.assertEqual(record["band"], list(SELF_CHECK_NLL_BAND))


class _Block(nn.Module):
    """A stand-in for ``SparseMoeBlock``: returns (hidden, router probabilities)."""

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return hidden * 2.0, torch.zeros(hidden.shape[0], 3)


class _Layer(nn.Module):
    def __init__(self, heads: int, head_dim: int) -> None:
        super().__init__()
        self.block_sparse_moe = _Block()
        self.self_attn = nn.Module()
        self.self_attn.o_proj = nn.Linear(heads * head_dim, heads * head_dim, bias=False)


class _Backbone(nn.Module):
    def __init__(self, layers: int, heads: int, head_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer(heads, head_dim) for _ in range(layers)])


class _Model(nn.Module):
    def __init__(self, layers: int, heads: int, head_dim: int) -> None:
        super().__init__()
        self.model = _Backbone(layers, heads, head_dim)


def stub(layers: int = 2, heads: int = 3, head_dim: int = 4) -> ProGen3:
    return ProGen3(
        model=_Model(layers, heads, head_dim),
        config=SimpleNamespace(
            num_hidden_layers=layers,
            num_attention_heads=heads,
            hidden_size=heads * head_dim,
            pad_token_id=0,
        ),
        preparer=None,
        device=torch.device("cpu"),
        checkpoint=Path("/nonexistent"),
    )


class TheInterventionSurfaceHonoursTheBlockContract(unittest.TestCase):
    """The MoE block returns a pair, and a hook that forgets that is silent."""

    def test_returning_none_leaves_the_output_untouched(self):
        pg = stub()
        seen: list[int] = []
        x = torch.ones(2, 12)
        with moe_intercept(pg, lambda layer, i, o: seen.append(layer) or None):
            hidden, router = pg.moe_blocks[0](x)
        self.assertEqual(seen, [0])
        self.assertTrue(torch.equal(hidden, x * 2.0))
        self.assertEqual(tuple(router.shape), (2, 3))

    def test_a_replacement_keeps_the_router_term(self):
        pg = stub()
        x = torch.ones(2, 12)
        with moe_intercept(pg, lambda layer, i, o: torch.zeros_like(o)):
            hidden, router = pg.moe_blocks[1](x)
        self.assertTrue(torch.equal(hidden, torch.zeros(2, 12)))
        self.assertEqual(tuple(router.shape), (2, 3))

    def test_the_interceptor_sees_the_blocks_input_and_output(self):
        pg = stub()
        recorded: dict[str, torch.Tensor] = {}

        def tap(layer, block_input, block_output):
            recorded["in"] = block_input
            recorded["out"] = block_output
            return None

        x = torch.arange(12, dtype=torch.float32).reshape(1, 12)
        with moe_intercept(pg, tap):
            pg.moe_blocks[0](x)
        self.assertTrue(torch.equal(recorded["in"], x))
        self.assertTrue(torch.equal(recorded["out"], x * 2.0))

    def test_an_ablation_applies_on_top_of_a_replacement(self):
        # The composition the causal stage depends on: the replacement runs, and
        # the ablation removes the replacement's own output rather than the
        # original block's.
        pg = stub()
        x = torch.ones(1, 12)
        with moe_intercept(pg, lambda layer, i, o: o + 100.0):
            with ablated(pg, Component("moe_block", 0)):
                inner, _ = pg.moe_blocks[0](x)
            outer, _ = pg.moe_blocks[0](x)
        self.assertTrue(torch.equal(inner, torch.zeros(1, 12)))
        self.assertTrue(torch.equal(outer, x * 2.0 + 100.0))

    def test_every_hook_is_removed_when_the_context_exits(self):
        pg = stub()
        with moe_intercept(pg, lambda layer, i, o: torch.zeros_like(o)):
            pass
        hidden, _ = pg.moe_blocks[0](torch.ones(1, 12))
        self.assertTrue(torch.equal(hidden, torch.full((1, 12), 2.0)))


class HeadAblationRemovesExactlyOneHeadsColumns(unittest.TestCase):
    """Head ``h`` owns ``o_proj`` input columns ``h*head_dim .. (h+1)*head_dim``."""

    def test_only_the_named_heads_slice_is_zeroed(self):
        pg = stub(layers=1, heads=3, head_dim=4)
        projection = pg.model.model.layers[0].self_attn.o_proj
        x = torch.arange(12, dtype=torch.float32).reshape(1, 12)
        with ablated(pg, Component("attention_head", 0, 1)):
            observed = projection(x)
        expected_input = x.clone()
        expected_input[:, 4:8] = 0.0
        self.assertTrue(torch.allclose(observed, projection(expected_input)))
        # The caller's tensor must not have been modified in place.
        self.assertTrue(torch.equal(x, torch.arange(12, dtype=torch.float32).reshape(1, 12)))

    def test_the_hook_is_removed_afterwards(self):
        pg = stub(layers=1, heads=3, head_dim=4)
        projection = pg.model.model.layers[0].self_attn.o_proj
        x = torch.ones(1, 12)
        with ablated(pg, Component("attention_head", 0, 0)):
            pass
        self.assertTrue(torch.allclose(projection(x), projection(x.clone())))
        with ablated(pg, Component("attention_head", 0, 0)):
            ablated_output = projection(x)
        self.assertFalse(torch.allclose(ablated_output, projection(x)))

    def test_a_component_kind_with_no_ablation_path_is_refused(self):
        pg = stub()
        with self.assertRaises(ValueError) as caught:
            with ablated(pg, Component("expert", 0, 1)):
                pass
        self.assertIn("no ablation is implemented", str(caught.exception))

    def test_the_component_grid_is_every_head_then_every_block(self):
        pg = stub(layers=2, heads=3, head_dim=4)
        grid = components(pg)
        self.assertEqual(len(grid), 2 * 3 + 2)
        self.assertEqual(grid[0].label, "attention_head.L0H0")
        self.assertEqual(grid[-1].label, "moe_block.L1")
        self.assertEqual(
            [component.kind for component in grid],
            ["attention_head"] * 6 + ["moe_block"] * 2,
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
