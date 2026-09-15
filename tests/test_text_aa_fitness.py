"""CPU contract for the FP32 text-AA string-control scorer.

Real published weights are not loaded. Tokenisers may be read from local
``/Data/public`` files. Tiny Transformers configs stand in for GPT-2, Qwen2,
and Llama. ByGPT5 uses its local tokenizer plus a float32 stub so pad-id
conditioning can be checked without executing an unbounded remote-code path.
"""

from __future__ import annotations

import gc
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer.precision_policy import FP32_PER_TARGET_ABS  # noqa: E402
from src.transfer.scale_comparison import STRATUM_N_TO_C  # noqa: E402
from src.transfer.text_aa_fitness import (  # noqa: E402
    TEXT_AA_FP32_V1,
    TextAABoundary,
    TextAAFitnessScorer,
    encode_text_aa,
    load_text_aa_scorer,
    read_hard_context,
    resolve_text_aa_boundary,
)

PROBE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
GPT2_DIR = Path("/Data/public/gpt2")
QWEN2_DIR = Path("/Data/public/Qwen2.5-0.5B")
QWEN3_DIR = Path("/Data/public/Qwen3-8B-Base")
LLAMA_DIR = Path("/Data/public/Llama-3.2-3B")
BYGPT5_DIR = Path("/Data/public/bygpt5-small-en")


def _gpt2_boundary() -> TextAABoundary:
    return TextAABoundary(
        name="gpt2",
        model_type="gpt2",
        tokenizer_class="GPT2TokenizerFast",
        conditioning_kind="bos_same_id_as_eos",
        conditioning_id=50256,
        conditioning_token="<|endoftext|>",
        bos_token_id=50256,
        eos_token_id=50256,
        pad_token_id=None,
        unk_token_id=50256,
        decoder_start_token_id=None,
        documented_context=1024,
        forbid_token_ids=(),
        trust_remote_code=False,
    )


def _qwen_boundary(*, name: str = "qwen2.5-0.5b", model_type: str = "qwen2") -> TextAABoundary:
    return TextAABoundary(
        name=name,
        model_type=model_type,
        tokenizer_class="Qwen2TokenizerFast",
        conditioning_kind="eos_document_separator",
        conditioning_id=151643,
        conditioning_token="<|endoftext|>",
        bos_token_id=None,
        eos_token_id=151643,
        pad_token_id=151643,
        unk_token_id=None,
        decoder_start_token_id=None,
        documented_context=32768,
        forbid_token_ids=(151644,),
        trust_remote_code=False,
    )


def _llama_boundary() -> TextAABoundary:
    return TextAABoundary(
        name="llama-3.2-3b",
        model_type="llama",
        tokenizer_class="PreTrainedTokenizerFast",
        conditioning_kind="true_bos",
        conditioning_id=128000,
        conditioning_token="<|begin_of_text|>",
        bos_token_id=128000,
        eos_token_id=128001,
        pad_token_id=None,
        unk_token_id=None,
        decoder_start_token_id=None,
        documented_context=131072,
        forbid_token_ids=(),
        trust_remote_code=False,
    )


def _bygpt5_boundary() -> TextAABoundary:
    return TextAABoundary(
        name="bygpt5-small-en",
        model_type="bygpt5",
        tokenizer_class="ByGPT5Tokenizer",
        conditioning_kind="decoder_start",
        conditioning_id=0,
        conditioning_token="<pad>",
        bos_token_id=None,
        eos_token_id=1,
        pad_token_id=0,
        unk_token_id=2,
        decoder_start_token_id=0,
        documented_context=None,
        forbid_token_ids=(),
        trust_remote_code=True,
    )


def _require_dir(path: Path) -> Path:
    if not path.is_dir():
        pytest.skip(f"local tokenizer directory is absent: {path}")
    return path


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    path = _require_dir(GPT2_DIR)
    return AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False
    )


@pytest.fixture(scope="module")
def qwen2_tokenizer():
    path = _require_dir(QWEN2_DIR)
    return AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False
    )


@pytest.fixture(scope="module")
def qwen3_tokenizer():
    path = _require_dir(QWEN3_DIR)
    return AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False
    )


@pytest.fixture(scope="module")
def llama_tokenizer():
    path = _require_dir(LLAMA_DIR)
    return AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False
    )


@pytest.fixture(scope="module")
def bygpt5_tokenizer():
    path = _require_dir(BYGPT5_DIR)
    return AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=True
    )


class _StubLM(nn.Module):
    def __init__(self, vocab_size: int, *, logits_fn=None) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self.embed = nn.Embedding(vocab_size, 8)
        self.weight = nn.Parameter(torch.ones(1, dtype=torch.float32))
        self._logits_fn = logits_fn
        self.forward_ids: list[torch.Tensor] = []
        self.forward_masks: list[torch.Tensor | None] = []
        self.logit_tensors: list[torch.Tensor] = []

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids, attention_mask=None, use_cache=None):
        assert use_cache is False
        self.forward_ids.append(input_ids.detach().cpu().clone())
        self.forward_masks.append(
            None if attention_mask is None else attention_mask.detach().cpu().clone()
        )
        if self._logits_fn is not None:
            logits = self._logits_fn(input_ids)
        else:
            batch, width = input_ids.shape
            position = torch.arange(width, dtype=torch.float32, device=input_ids.device)
            vocab = torch.arange(
                self.embed.num_embeddings, dtype=torch.float32, device=input_ids.device
            )
            logits = (0.15 * position.view(1, width, 1) + 0.02 * vocab.view(1, 1, -1)).expand(
                batch, width, self.embed.num_embeddings
            ).contiguous()
        self.logit_tensors.append(logits)
        return SimpleNamespace(logits=logits)


def _tiny_gpt2(tokenizer, *, n_positions: int = 64) -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=n_positions,
        n_ctx=n_positions,
        n_embd=32,
        n_layer=1,
        n_head=4,
        n_inner=64,
        bos_token_id=50256,
        eos_token_id=50256,
        attn_pdrop=0.0,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


def _tiny_qwen2(tokenizer, *, max_position_embeddings: int = 64) -> Qwen2ForCausalLM:
    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=max_position_embeddings,
        bos_token_id=151643,
        eos_token_id=151643,
        pad_token_id=151643,
        tie_word_embeddings=True,
        attention_dropout=0.0,
    )
    model = Qwen2ForCausalLM(config)
    model.eval()
    return model


def _tiny_llama(tokenizer, *, max_position_embeddings: int = 64) -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=max_position_embeddings,
        bos_token_id=128000,
        eos_token_id=128001,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        attention_dropout=0.0,
    )
    model = LlamaForCausalLM(config)
    model.eval()
    return model


def _scorer(model, tokenizer, boundary, *, window: int = 64, hard=None):
    return TextAAFitnessScorer(
        model=model,
        tokenizer=tokenizer,
        boundary=boundary,
        application_window_tokens=window,
        facts={"hard_context": hard, "name": boundary.name},
    )


def _independent_shifted_ce_sum(model, ids: list[int]) -> tuple[float, int]:
    tokens = torch.tensor([ids], dtype=torch.long)
    mask = torch.ones_like(tokens)
    with torch.no_grad():
        logits = model(input_ids=tokens, attention_mask=mask, use_cache=False).logits
        assert logits.dtype == torch.float32
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            tokens[:, 1:].reshape(-1),
            reduction="sum",
        )
    n_target = len(ids) - 1
    return float((-loss).double().cpu()), n_target


def test_protocol_constant_and_stratum_are_declared():
    assert TEXT_AA_FP32_V1 == "text-aa-fp32-v1"
    assert TextAAFitnessScorer.scoring_stratum == STRATUM_N_TO_C
    assert "document-boundary" in TextAAFitnessScorer.score_description
    assert "PASS" not in TextAAFitnessScorer.score_description


def test_read_hard_context_ignores_generation_max_length():
    assert read_hard_context(SimpleNamespace(n_positions=1024, max_length=20)) == 1024
    assert read_hard_context(SimpleNamespace(max_position_embeddings=32768)) == 32768
    assert read_hard_context(SimpleNamespace(max_length=20, d_model=1472)) is None


def test_resolve_boundary_inheritance_and_refuse_remote_code_on_builtins():
    table = {
        "text_aa_checkpoints": {
            "gpt2": {
                "model_type": "gpt2",
                "tokenizer_class": "GPT2TokenizerFast",
                "eos_token_id": 50256,
                "bos_token_id": 50256,
                "unk_token_id": 50256,
                "pad_token_id": None,
                "context": 1024,
                "conditioning": {
                    "kind": "bos_same_id_as_eos",
                    "token": "<|endoftext|>",
                    "id": 50256,
                },
            },
            "gpt2-medium": {
                "inherits_boundary_from": "gpt2",
                "context": 1024,
            },
        }
    }
    resolved = resolve_text_aa_boundary("gpt2-medium", table)
    assert resolved.conditioning_id == 50256
    assert resolved.model_type == "gpt2"
    assert resolved.trust_remote_code is False
    with pytest.raises(ValueError, match="trust_remote_code"):
        TextAABoundary.from_mapping(
            {
                "model_type": "gpt2",
                "tokenizer_class": "GPT2TokenizerFast",
                "trust_remote_code": True,
                "eos_token_id": 50256,
                "conditioning": {"kind": "bos", "token": "x", "id": 1},
            },
            name="gpt2",
        )


def test_illegal_empty_and_non_aa20_are_refused_before_unk(gpt2_tokenizer):
    boundary = _gpt2_boundary()
    for illegal in ("", "MKTZ", "MKTJ", "mkt", "MKT A", "MKT\n", "MKT\t"):
        with pytest.raises(ValueError):
            encode_text_aa(gpt2_tokenizer, illegal, boundary)


def test_gpt2_encode_prepends_boundary_keeps_first_target_and_omits_eos(
    gpt2_tokenizer,
):
    boundary = _gpt2_boundary()
    ids = encode_text_aa(gpt2_tokenizer, PROBE, boundary)
    assert ids[0] == 50256
    targets = gpt2_tokenizer(PROBE, add_special_tokens=False)["input_ids"]
    assert ids[1:] == list(targets)
    assert len(targets) == 22
    assert ids[-1] != 50256
    assert 50256 not in ids[1:]
    assert gpt2_tokenizer.decode(targets, skip_special_tokens=False) == PROBE
    default = gpt2_tokenizer(PROBE, add_special_tokens=True)["input_ids"]
    assert list(default) == list(targets)


def test_qwen_uses_eos_separator_not_chat_or_bos(
    qwen2_tokenizer, qwen3_tokenizer
):
    for tokenizer, name, model_type in (
        (qwen2_tokenizer, "qwen2.5-0.5b", "qwen2"),
        (qwen3_tokenizer, "qwen3-8b-base", "qwen3"),
    ):
        boundary = _qwen_boundary(name=name, model_type=model_type)
        assert tokenizer.bos_token_id is None
        ids = encode_text_aa(tokenizer, PROBE, boundary)
        assert ids[0] == 151643
        assert 151644 not in ids
        targets = tokenizer(PROBE, add_special_tokens=False)["input_ids"]
        assert len(targets) == 21
        assert ids[-1] != 151643
        assert tokenizer.decode(targets, skip_special_tokens=False) == PROBE


def test_llama_true_bos_is_concatenated_without_double_prefix(llama_tokenizer):
    boundary = _llama_boundary()
    ids = encode_text_aa(llama_tokenizer, PROBE, boundary)
    assert ids[0] == 128000
    targets = llama_tokenizer(PROBE, add_special_tokens=False)["input_ids"]
    assert 128000 not in targets
    assert ids[1:] == list(targets)
    assert len(targets) == 21
    assert ids[-1] != 128001
    with_special = llama_tokenizer(PROBE, add_special_tokens=True)["input_ids"]
    assert list(with_special) == ids


def test_bygpt5_byte_targets_and_pad_conditioning(bygpt5_tokenizer):
    boundary = _bygpt5_boundary()
    ids = encode_text_aa(bygpt5_tokenizer, PROBE, boundary)
    assert ids[0] == 0
    assert ids[0] == bygpt5_tokenizer.pad_token_id
    targets = bygpt5_tokenizer(PROBE, add_special_tokens=False)["input_ids"]
    assert len(targets) == 33
    assert 0 not in targets
    assert bygpt5_tokenizer.decode(targets, skip_special_tokens=False) == PROBE


def test_missing_window_is_refused(gpt2_tokenizer):
    model = _StubLM(len(gpt2_tokenizer))
    with pytest.raises(TypeError):
        TextAAFitnessScorer(
            model=model,
            tokenizer=gpt2_tokenizer,
            boundary=_gpt2_boundary(),
            facts={"hard_context": 64},
        )
    with pytest.raises(ValueError, match="application_window_tokens"):
        _scorer(model, gpt2_tokenizer, _gpt2_boundary(), window=1)


def test_over_window_and_hard_context_are_refused(gpt2_tokenizer):
    boundary = _gpt2_boundary()
    stub = _StubLM(len(gpt2_tokenizer))
    encoded = encode_text_aa(gpt2_tokenizer, "MKT", boundary)
    scorer = _scorer(stub, gpt2_tokenizer, boundary, window=len(encoded) - 1)
    assert scorer.token_lengths(["MKT"]) == [len(encoded)]
    with pytest.raises(ValueError, match="application_window_tokens"):
        scorer.log_likelihood(["MKT"])
    with pytest.raises(ValueError, match="hard context"):
        _scorer(
            stub,
            gpt2_tokenizer,
            boundary,
            window=4096,
            hard=1024,
        )


def test_empty_list_mixed_length_order_and_repeats(gpt2_tokenizer):
    boundary = _gpt2_boundary()
    model = _tiny_gpt2(gpt2_tokenizer)
    scorer = _scorer(model, gpt2_tokenizer, boundary, window=64, hard=64)
    assert scorer.token_lengths([]) == []
    empty = scorer.log_likelihood([])
    assert empty.shape == (0,)
    assert empty.dtype == np.float64
    sequences = ["W", "MKT", "ACDEFGHIKLMNPQRSTVWY"]
    lengths = scorer.token_lengths(sequences)
    assert lengths[0] < lengths[1] < lengths[2]
    totals = scorer.log_likelihood(sequences)
    singles = np.array(
        [scorer.log_likelihood([sequence])[0] for sequence in sequences]
    )
    assert totals == pytest.approx(singles, abs=0)
    shuffled = list(reversed(sequences))
    reversed_totals = scorer.log_likelihood(shuffled)
    assert reversed_totals[0] == pytest.approx(totals[2], abs=0)
    assert reversed_totals[2] == pytest.approx(totals[0], abs=0)
    repeats = scorer.log_likelihood(["MKT", "MKT", "MKT"])
    assert repeats.shape == (3,)
    assert repeats[0] == pytest.approx(repeats[1], abs=1e-12)
    assert repeats[1] == pytest.approx(repeats[2], abs=1e-12)


def test_independent_shifted_ce_matches_public_api_and_keeps_first_target(
    gpt2_tokenizer,
):
    boundary = _gpt2_boundary()
    model = _tiny_gpt2(gpt2_tokenizer)
    scorer = _scorer(model, gpt2_tokenizer, boundary, window=64, hard=64)
    sequence = "MKTAYIAK"
    ids = scorer.encode(sequence)
    got = scorer.log_likelihood([sequence])[0]
    independent, n_target = _independent_shifted_ce_sum(model, ids)
    assert n_target == len(ids) - 1
    assert abs(got - independent) / n_target <= FP32_PER_TARGET_ABS
    tokens = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(
            input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False
        ).logits
        logp = torch.log_softmax(logits[:, :-1], dim=-1)
        skip_first = logp[:, 1:].gather(-1, tokens[:, 2:].unsqueeze(-1)).squeeze(-1)
        skip_total = float(skip_first.sum().double().cpu())
    assert abs(got - skip_total) / n_target > FP32_PER_TARGET_ABS
    meta = scorer.encoding_metadata(sequence)
    assert meta["n_prefix_tokens"] == 1
    assert meta["n_scored_tokens"] == n_target
    assert meta["n_residues"] == len(sequence)
    assert meta["trailing_eos_scored"] is False
    assert meta["prefix_scored"] is False


def test_qwen_conditioning_equals_pad_still_attends_position_zero(qwen2_tokenizer):
    boundary = _qwen_boundary()
    stub = _StubLM(len(qwen2_tokenizer))
    scorer = _scorer(stub, qwen2_tokenizer, boundary, window=64)
    scorer.log_likelihood(["MKT"])
    ids = stub.forward_ids[0]
    mask = stub.forward_masks[0]
    assert int(ids[0, 0]) == 151643
    assert mask is not None
    assert int(mask[0, 0]) == 1
    assert torch.equal(mask, torch.ones_like(mask))
    dropped = (ids != 151643).to(dtype=torch.long)
    assert int(dropped[0, 0]) == 0


def test_bygpt5_pad_conditioning_uses_position_mask(bygpt5_tokenizer):
    boundary = _bygpt5_boundary()
    stub = _StubLM(384)
    scorer = _scorer(stub, bygpt5_tokenizer, boundary, window=128)
    scorer.log_likelihood(["MKT"])
    ids = stub.forward_ids[0]
    mask = stub.forward_masks[0]
    assert int(ids[0, 0]) == 0
    assert mask is not None
    assert int(mask[0, 0]) == 1
    assert int((ids[0] == 0).sum()) >= 1
    assert torch.equal(mask, torch.ones_like(ids))


def test_llama_tiny_numeric_gate(llama_tokenizer):
    boundary = _llama_boundary()
    model = _tiny_llama(llama_tokenizer)
    scorer = _scorer(model, llama_tokenizer, boundary, window=64, hard=64)
    ids = scorer.encode("MKT")
    got = scorer.log_likelihood(["MKT"])[0]
    independent, n_target = _independent_shifted_ce_sum(model, ids)
    assert abs(got - independent) / n_target <= FP32_PER_TARGET_ABS


def test_qwen2_tiny_numeric_gate(qwen2_tokenizer):
    boundary = _qwen_boundary()
    model = _tiny_qwen2(qwen2_tokenizer)
    scorer = _scorer(model, qwen2_tokenizer, boundary, window=64, hard=64)
    ids = scorer.encode("ACDE")
    got = scorer.log_likelihood(["ACDE"])[0]
    independent, n_target = _independent_shifted_ce_sum(model, ids)
    assert abs(got - independent) / n_target <= FP32_PER_TARGET_ABS


def test_bf16_parameters_are_refused(gpt2_tokenizer):
    model = _tiny_gpt2(gpt2_tokenizer).to(torch.bfloat16)
    with pytest.raises(ValueError, match="float32"):
        _scorer(model, gpt2_tokenizer, _gpt2_boundary())


def test_fp32_matmul_context_restores_caller_state(gpt2_tokenizer):
    model = _tiny_gpt2(gpt2_tokenizer)
    scorer = _scorer(model, gpt2_tokenizer, _gpt2_boundary(), window=64, hard=64)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    scorer.log_likelihood(["MKT"])
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True
    assert torch.get_float32_matmul_precision() == "high"


def test_three_item_lifetime_releases_prior_logits(gpt2_tokenizer):
    boundary = _gpt2_boundary()
    model = _tiny_gpt2(gpt2_tokenizer)
    refs: list[weakref.ReferenceType[torch.Tensor]] = []
    original = model.forward

    def wrapped(*args, **kwargs):
        if refs:
            assert refs[-1]() is None
        outputs = original(*args, **kwargs)
        refs.append(weakref.ref(outputs.logits))
        return outputs

    model.forward = wrapped
    scorer = _scorer(model, gpt2_tokenizer, boundary, window=64, hard=64)
    sequences = ["W", "MKT", "ACDEFGHIKL"]
    totals = scorer.log_likelihood(sequences)
    assert totals.shape == (3,)
    gc.collect()
    assert len(refs) == 3
    assert refs[0]() is None
    assert refs[1]() is None
    assert refs[2]() is None


def test_load_text_aa_scorer_from_saved_tiny_gpt2(gpt2_tokenizer, tmp_path):
    model = _tiny_gpt2(gpt2_tokenizer, n_positions=64)
    checkpoint = tmp_path / "tiny-gpt2"
    checkpoint.mkdir()
    model.save_pretrained(checkpoint)
    gpt2_tokenizer.save_pretrained(checkpoint)
    scorer = load_text_aa_scorer(
        checkpoint,
        boundary=_gpt2_boundary(),
        application_window_tokens=64,
        device="cpu",
        name="gpt2",
    )
    try:
        assert scorer.facts["protocol"] == TEXT_AA_FP32_V1
        assert scorer.facts["dtype_observed"] == ["float32"]
        assert scorer.facts["hard_context"] == 64
        assert scorer.facts["experiment_admitted"] is False
        assert scorer.facts["production_batch_size"] == 1
        totals = scorer.log_likelihood(["MKT"])
        independent, n_target = _independent_shifted_ce_sum(scorer.model, scorer.encode("MKT"))
        assert abs(float(totals[0]) - independent) / n_target <= FP32_PER_TARGET_ABS
    finally:
        scorer.release()


def test_load_refuses_missing_path_and_window_beyond_hard_context(
    gpt2_tokenizer, tmp_path
):
    missing = tmp_path / "absent-checkpoint"
    with pytest.raises(FileNotFoundError):
        load_text_aa_scorer(
            missing,
            boundary=_gpt2_boundary(),
            application_window_tokens=64,
            device="cpu",
        )
    model = _tiny_gpt2(gpt2_tokenizer, n_positions=32)
    checkpoint = tmp_path / "tiny-gpt2-short"
    checkpoint.mkdir()
    model.save_pretrained(checkpoint)
    gpt2_tokenizer.save_pretrained(checkpoint)
    with pytest.raises(ValueError, match="hard context"):
        load_text_aa_scorer(
            checkpoint,
            boundary=_gpt2_boundary(),
            application_window_tokens=64,
            device="cpu",
            name="gpt2",
        )
