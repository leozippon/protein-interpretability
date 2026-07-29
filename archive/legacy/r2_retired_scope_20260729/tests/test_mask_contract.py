"""CPU-only regression tests for the CLT valid-token mask contract."""

import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model_loader import ProteinModel  # noqa: E402
from src.training.clt_trainer import (  # noqa: E402
    CLTForTraining,
    CLTTrainer,
    _resolve_model_inference_contract,
)


def test_confirmatory_model_inference_contract_requires_bfloat16() -> None:
    verification = (
        "all_floating_model_parameters_exactly_declared_before_first_activation"
    )
    name, dtype = _resolve_model_inference_contract(
        {
            "inference_dtype": "bfloat16",
            "inference_dtype_verification": verification,
        },
        confirmatory=True,
    )
    assert (name, dtype) == ("bfloat16", torch.bfloat16)
    with pytest.raises(ValueError, match="requires declared bfloat16"):
        _resolve_model_inference_contract(
            {
                "inference_dtype": "float16",
                "inference_dtype_verification": verification,
            },
            confirmatory=True,
        )


def _configured_clt() -> CLTForTraining:
    clt = CLTForTraining(n_layers=1, d_model=2, d_clt=3, k=1, window=1)
    with torch.no_grad():
        clt.W_enc[0].copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]))
        clt.b_enc.zero_()
        clt.W_dec[0].copy_(
            torch.tensor(
                [
                    [[0.5, 0.0]],
                    [[0.0, 0.25]],
                    [[-0.1, -0.1]],
                ]
            )
        )
        clt.b_dec.zero_()
        clt.global_step.fill_(7)
    return clt


def _assert_same_forward(left: dict, right: dict) -> None:
    torch.testing.assert_close(left["loss"], right["loss"], rtol=0, atol=0)
    for key in ("fvu_mean", "l0_mean", "dead_mean"):
        assert left[key] == right[key]
    assert left["fvu_per_layer"] == right["fvu_per_layer"]


def test_training_ignores_padded_values_and_firing() -> None:
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    valid_resid = torch.tensor(
        [[[2.0, 0.5], [0.25, 3.0], [0.0, 0.0]], [[1.5, 0.75], [0.0, 0.0], [0.0, 0.0]]]
    )
    valid_target = torch.tensor(
        [[[1.0, -0.5], [0.25, 2.0], [0.0, 0.0]], [[-1.0, 0.75], [0.0, 0.0], [0.0, 0.0]]]
    )
    resid_a, resid_b = valid_resid.clone(), valid_resid.clone()
    target_a, target_b = valid_target.clone(), valid_target.clone()
    resid_a[~mask.bool()] = torch.tensor([100.0, 0.0])
    resid_b[~mask.bool()] = torch.tensor([0.0, 1000.0])
    target_a[~mask.bool()] = -500.0
    target_b[~mask.bool()] = 700.0

    clt_a = _configured_clt()
    clt_b = copy.deepcopy(clt_a)
    result_a = clt_a([resid_a], [target_a], mask)
    result_b = clt_b([resid_b], [target_b], mask)
    result_a["loss"].backward()
    result_b["loss"].backward()

    _assert_same_forward(result_a, result_b)
    torch.testing.assert_close(
        clt_a.feature_last_fired, clt_b.feature_last_fired, rtol=0, atol=0
    )
    for param_a, param_b in zip(clt_a.parameters(), clt_b.parameters()):
        torch.testing.assert_close(param_a.grad, param_b.grad, rtol=0, atol=0)


def test_padded_only_feature_does_not_fire() -> None:
    clt = CLTForTraining(n_layers=1, d_model=2, d_clt=2, k=1, window=1)
    with torch.no_grad():
        clt.W_enc[0].copy_(torch.eye(2))
        clt.b_enc.zero_()
        clt.W_dec[0].zero_()
        clt.b_dec.zero_()
        clt.global_step.fill_(11)

    resid = [torch.tensor([[[2.0, 0.0], [0.0, 100.0]]])]
    target = [torch.tensor([[[1.0, -1.0], [500.0, 500.0]]])]
    clt(resid, target, torch.tensor([[1, 0]]))

    assert clt.feature_last_fired[0].tolist() == [11, 0]


def test_all_valid_mask_is_exactly_equivalent_to_legacy_path() -> None:
    torch.manual_seed(123)
    resid = [torch.randn(2, 3, 2)]
    target = [torch.randn(2, 3, 2)]
    clt_unmasked = _configured_clt()
    clt_masked = copy.deepcopy(clt_unmasked)

    unmasked = clt_unmasked(resid, target)
    masked = clt_masked(resid, target, torch.ones(2, 3, dtype=torch.long))
    unmasked["loss"].backward()
    masked["loss"].backward()

    _assert_same_forward(unmasked, masked)
    torch.testing.assert_close(
        clt_unmasked.feature_last_fired,
        clt_masked.feature_last_fired,
        rtol=0,
        atol=0,
    )
    for param_a, param_b in zip(clt_unmasked.parameters(), clt_masked.parameters()):
        torch.testing.assert_close(param_a.grad, param_b.grad, rtol=0, atol=0)


def _resampling_trainer() -> CLTTrainer:
    clt = CLTForTraining(n_layers=1, d_model=2, d_clt=2, k=1, window=1)
    with torch.no_grad():
        for parameter in clt.parameters():
            parameter.zero_()
        clt.global_step.fill_(2)
    trainer = CLTTrainer.__new__(CLTTrainer)
    trainer.clt_module = clt
    trainer.dead_threshold = 0
    trainer.max_resample_fraction = 1.0
    trainer.device = torch.device("cpu")
    trainer.is_main = True
    trainer.world_size = 1
    trainer.optimizer = torch.optim.Adam(clt.parameters(), lr=1e-3)
    return trainer


def test_resampling_cap_is_deterministic_and_bounded() -> None:
    clt = CLTForTraining(n_layers=1, d_model=2, d_clt=10, k=1, window=1)
    with torch.no_grad():
        for parameter in clt.parameters():
            parameter.zero_()
        clt.global_step.fill_(2)
    trainer = CLTTrainer.__new__(CLTTrainer)
    trainer.clt_module = clt
    trainer.dead_threshold = 0
    trainer.max_resample_fraction = 0.2
    trainer.device = torch.device("cpu")
    trainer.is_main = True
    trainer.world_size = 1
    trainer.optimizer = torch.optim.Adam(clt.parameters(), lr=1e-3)
    count = trainer.resample_dead_features(
        [torch.zeros(1, 2, 2)],
        [torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])],
        torch.ones(1, 2, dtype=torch.long),
    )
    assert count == 2
    assert int((clt.feature_last_fired[0] == 2).sum()) == 2


def test_resampling_ignores_padded_values() -> None:
    mask = torch.tensor([[1, 1, 0, 0]])
    resid_a = torch.zeros(1, 4, 2)
    resid_b = resid_a.clone()
    resid_a[:, 2:] = 100.0
    resid_b[:, 2:] = -1000.0
    target_a = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [500.0, 0.0], [0.0, 500.0]]])
    target_b = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [-900.0, 1.0], [1.0, -900.0]]])
    trainer_a = _resampling_trainer()
    trainer_b = _resampling_trainer()

    torch.manual_seed(99)
    count_a = trainer_a.resample_dead_features([resid_a], [target_a], mask)
    torch.manual_seed(99)
    count_b = trainer_b.resample_dead_features([resid_b], [target_b], mask)

    assert count_a == count_b == 2
    torch.testing.assert_close(
        trainer_a.clt_module.W_enc,
        trainer_b.clt_module.W_enc,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        trainer_a.clt_module.W_dec[0],
        trainer_b.clt_module.W_dec[0],
        rtol=0,
        atol=0,
    )


class _BatchTokenizer:
    def __init__(self, pad_value: int = 0):
        self.pad_value = pad_value

    def __call__(self, sequences, **_kwargs):
        max_len = max(map(len, sequences))
        input_ids = torch.full((len(sequences), max_len), self.pad_value)
        attention_mask = torch.zeros_like(input_ids)
        for row, sequence in enumerate(sequences):
            length = len(sequence)
            input_ids[row, :length] = torch.arange(1, length + 1)
            attention_mask[row, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_batch_and_model_forward_preserve_attention_mask() -> None:
    trainer = CLTTrainer.__new__(CLTTrainer)
    trainer.rank = 0
    trainer.world_size = 1
    trainer.device = torch.device("cpu")
    trainer.data_cfg = {"max_seq_len": 8}
    trainer.tokenizer = _BatchTokenizer(pad_value=9)
    input_ids, attention_mask = trainer._make_batch(["AAA", "A"], 0, 2)
    assert input_ids.shape == attention_mask.shape == (2, 3)
    assert int(attention_mask.sum()) == 4

    class TinyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Identity()

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.Module()
            self.transformer.h = nn.ModuleList([TinyBlock()])
            self.seen_mask = None

        def forward(self, input_ids, attention_mask=None):
            self.seen_mask = attention_mask.clone()
            hidden = torch.nn.functional.one_hot(input_ids % 2, num_classes=2).float()
            return self.transformer.h[0].mlp(hidden)

    model = TinyLM()
    wrapped = ProteinModel(
        model,
        tokenizer=None,
        config=SimpleNamespace(n_layer=1, n_embd=2),
        model_name="tiny",
        device="cpu",
    )
    wrapped.get_activations(input_ids, attention_mask)
    torch.testing.assert_close(model.seen_mask, attention_mask, rtol=0, atol=0)


def test_activation_hooks_use_declared_layer_index_and_reject_duplicates() -> None:
    class OffsetMLP(nn.Module):
        def __init__(self, offset):
            super().__init__()
            self.offset = offset

        def forward(self, values):
            return values + self.offset

    class OutOfOrderLM(nn.Module):
        def __init__(self, duplicate=False):
            super().__init__()
            self.transformer = nn.Module()

            class Block(nn.Module):
                def __init__(self, offset):
                    super().__init__()
                    self.mlp = OffsetMLP(offset)

            self.transformer.h = nn.ModuleList([Block(1), Block(2)])
            self.duplicate = duplicate

        def forward(self, input_ids, attention_mask=None):
            hidden = torch.nn.functional.one_hot(input_ids % 2, num_classes=2).float()
            self.transformer.h[1].mlp(hidden + 10)
            self.transformer.h[0].mlp(hidden + 20)
            if self.duplicate:
                self.transformer.h[0].mlp(hidden + 30)

    input_ids = torch.tensor([[0, 1]])
    attention_mask = torch.ones_like(input_ids)
    model = OutOfOrderLM()
    wrapped = ProteinModel(
        model,
        tokenizer=None,
        config=SimpleNamespace(n_layer=2, n_embd=2),
        model_name="out-of-order",
        device="cpu",
    )
    activations = wrapped.get_activations(input_ids, attention_mask)
    assert activations.clt_input[0][0, 0, 0] == 21
    assert activations.clt_input[1][0, 0, 0] == 11

    duplicate = ProteinModel(
        OutOfOrderLM(duplicate=True),
        tokenizer=None,
        config=SimpleNamespace(n_layer=2, n_embd=2),
        model_name="duplicate",
        device="cpu",
    )
    with pytest.raises(RuntimeError, match="duplicate CLT-input capture for layer 0"):
        duplicate.get_activations(input_ids, attention_mask)


def test_evaluation_ignores_padded_values() -> None:
    script_path = PROJECT_ROOT / "scripts" / "05_evaluate_checkpoints.py"
    spec = importlib.util.spec_from_file_location("evaluate_checkpoints", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    class FakeProteinModel:
        def __init__(self, pad_value: int):
            self.tokenizer = _BatchTokenizer(pad_value)

        def get_activations(self, input_ids, attention_mask):
            assert attention_mask is not None
            values = input_ids.float()
            resid = torch.stack((values, values.square()), dim=-1)
            target = torch.stack((0.5 * values + 1.0, 2.0 - 0.25 * values), dim=-1)
            return SimpleNamespace(resid_pre=[resid], mlp_out=[target])

    clt_a = _configured_clt()
    clt_b = copy.deepcopy(clt_a)
    sequences = ["AAA", "A"]
    result_a = module.evaluate_clt(FakeProteinModel(100), clt_a, sequences, "cpu", 2)
    result_b = module.evaluate_clt(FakeProteinModel(900), clt_b, sequences, "cpu", 2)

    assert result_a == result_b
