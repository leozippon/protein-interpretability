"""Budget-only candidate serving: opt-in doors, tokenizer bounds, tiny forwards.

Interface implemented and tested is not an 8-block identification interval and
not an experiment ADMITTED digest. These tests load no released 8B/1.3B weights.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import MixtralConfig, MixtralForCausalLM, Qwen3Config, Qwen3ForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts/transfer") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts/transfer"))

from src.transfer.amino_acids import AA20  # noqa: E402
from src.transfer.arms import (  # noqa: E402
    PANEL,
    STAGED_ARMS,
    STAGED_CANDIDATE_ARMS,
    STAGED_SCALE_ARMS,
    STAGED_SECOND_STAGE_ARMS,
    Arm,
    Cohort,
    _ATTENTION_PATH,
    _BUILTIN_CAUSAL_LM_ARCHITECTURES,
    _DECOMPOSABLE,
    _ROTARY_DECODERS,
    arm_spec,
    load_arm,
    load_arm_spec,
    require_input_path,
    tokenize_batch,
)
from src.transfer.budget import scored_tokens  # noqa: E402
from src.transfer.scoring import sequence_target_mask, target_rule  # noqa: E402

QWEN3 = "qwen3-8b-base"
PROTGPT3 = "protgpt3-1.3b"


def _load_stage(filename: str):
    path = REPO_ROOT / "scripts/transfer" / filename
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _closed(**extra):
    args = dict(
        kind="text",
        with_ec=False,
        allow_staged_scale_arms=False,
        allow_second_stage_arms=False,
        allow_candidate_arms=False,
    )
    args.update(extra)
    return argparse.Namespace(**args)


class _PadTokenizer:
    """Tiny alphabet tokenizer with a real pad id, no BOS prefix."""

    pad_token = "<pad>"
    pad_token_id = 0
    bos_token = None
    bos_token_id = None
    eos_token = "<eos>"
    eos_token_id = 1

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = int(vocab_size)

    def __call__(self, text: str, return_tensors=None) -> dict[str, list[int]]:
        ids = [2 + (ord(character) % (self.vocab_size - 2)) for character in text]
        if not ids:
            return {"input_ids": []}
        return {"input_ids": ids}

    def decode(self, ids) -> str:
        return "".join(chr(97 + (int(value) - 2) % 26) for value in ids if int(value) >= 2)


def _tiny_qwen3() -> Arm:
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    spec = replace(STAGED_ARMS[QWEN3], n_layer=1, d_model=32)
    return Arm(
        spec=spec,
        model=Qwen3ForCausalLM(config).eval(),
        tokenizer=_PadTokenizer(32),
        device="cpu",
        dtype="float32",
        attn_implementation="eager",
    )


def _tiny_mixtral() -> Arm:
    config = MixtralConfig(
        vocab_size=16,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=64,
    )
    torch.manual_seed(1)
    spec = replace(STAGED_ARMS[PROTGPT3], n_layer=1, d_model=32)
    return Arm(
        spec=spec,
        model=MixtralForCausalLM(config).eval(),
        tokenizer=_PadTokenizer(16),
        device="cpu",
        dtype="float32",
        attn_implementation="eager",
    )


# ------------------------------------------------------------------ doors


def test_candidate_tuple_is_exactly_the_two_opt_in_checkpoints():
    assert STAGED_CANDIDATE_ARMS == (QWEN3, PROTGPT3)
    for name in STAGED_CANDIDATE_ARMS:
        assert name in STAGED_ARMS
        assert name not in PANEL
        assert name not in STAGED_SCALE_ARMS
        assert name not in STAGED_SECOND_STAGE_ARMS
        assert arm_spec(name) is STAGED_ARMS[name]
        with pytest.raises(KeyError):
            load_arm(name, device="cpu")


def test_candidates_are_budget_only_and_are_not_rotary_family_members():
    for name in STAGED_CANDIDATE_ARMS:
        spec = STAGED_ARMS[name]
        assert spec.capabilities == frozenset({"budget"}), name
        assert spec.architecture in _BUILTIN_CAUSAL_LM_ARCHITECTURES, name
        assert spec.architecture not in _ROTARY_DECODERS, name
        assert spec.architecture not in _DECOMPOSABLE, name
        assert spec.architecture not in _ATTENTION_PATH, name
        arm = Arm(
            spec=spec,
            model=SimpleNamespace(config=SimpleNamespace(vocab_size=spec.scoring_target_alphabet_size)),
            tokenizer=object(),
            device="cpu",
            dtype="float32",
        )
        assert arm.supports("budget")
        for capability in ("lens", "pathway", "circuits", "relational"):
            assert not arm.supports(capability), capability
            with pytest.raises(ValueError, match="does not support"):
                arm.require(capability)
        with pytest.raises(TypeError, match="unsupported architecture"):
            arm.blocks()


def test_candidate_paths_use_the_existing_relocators_and_not_instruct():
    qwen = STAGED_ARMS[QWEN3]
    protein = STAGED_ARMS[PROTGPT3]
    assert qwen.path_variable == "TRANSFER_TEXT_MODEL_BASE_DIR"
    assert qwen.path.name == "Qwen3-8B-Base"
    assert qwen.path.name != "Qwen3-8B"
    assert protein.path_variable == "TRANSFER_MODEL_BASE_DIR"
    assert protein.path.name == "ProtGPT3-1.3B"
    assert qwen.modality == "text"
    assert protein.modality == "protein"
    assert qwen.input_format == protein.input_format == "raw"
    assert qwen.scoring_target_alphabet_size == 151936
    assert protein.scoring_target_alphabet_size == 31


def test_cohort_power_refuses_candidates_without_the_opt_in():
    stage = _load_stage("01_cohort_power.py")
    with pytest.raises(ValueError, match="unknown arms"):
        stage.validate_arms([QWEN3], _closed())
    with pytest.raises(ValueError, match="unknown arms"):
        stage.validate_arms([PROTGPT3], _closed(kind="protein"))
    with pytest.raises(ValueError, match="unknown arms"):
        stage.validate_arms([QWEN3], _closed(allow_second_stage_arms=True))
    with pytest.raises(ValueError, match="unknown arms"):
        stage.validate_arms([QWEN3], _closed(allow_staged_scale_arms=True))
    with pytest.raises(ValueError, match="unknown arms"):
        stage.validate_arms(["qwen2.5-7b"], _closed(allow_candidate_arms=True))
    with pytest.raises(ValueError, match="unknown arms"):
        stage.validate_arms(["progen2-large"], _closed(kind="protein", allow_candidate_arms=True))


def test_cohort_power_candidate_opt_in_admits_only_its_tuple():
    stage = _load_stage("01_cohort_power.py")
    stage.validate_arms([QWEN3], _closed(allow_candidate_arms=True))
    stage.validate_arms([PROTGPT3], _closed(kind="protein", allow_candidate_arms=True))
    with pytest.raises(ValueError, match="do not match"):
        stage.validate_arms([QWEN3], _closed(kind="protein", allow_candidate_arms=True))
    with pytest.raises(ValueError, match="do not match"):
        stage.validate_arms([PROTGPT3], _closed(allow_candidate_arms=True))
    record = stage._candidate_record([QWEN3], True)
    assert record["not_panel_admission"] is True
    assert record["not_experiment_admitted"] is True
    assert record["measured_candidate_arms"] == [QWEN3]
    assert record["scoring_target_alphabet"][QWEN3]["size"] == 151936
    contract = stage._cohort_power_stage_contract([QWEN3])
    assert contract["not_panel_admission"] is True
    assert contract["measured"] == []


def test_candidates_are_not_on_the_fitness_or_designed_referent_doors():
    retrieval = _load_stage("20_retrieval_bound.py")
    designed = _load_stage("29_designed_referent.py")
    for name in STAGED_CANDIDATE_ARMS:
        assert name not in retrieval.SCOREABLE_ARMS, name
        assert name not in retrieval.ARM_CORPUS, name
        assert name not in designed.DEFAULT_ARMS, name
        assert name not in designed.STAGED_SCALE_ARMS, name


def test_default_campaign_arms_are_unchanged():
    stage = _load_stage("01_cohort_power.py")
    protein = stage.default_arms("protein", with_ec=False)
    text = stage.default_arms("text", with_ec=False)
    assert QWEN3 not in text and QWEN3 not in protein
    assert PROTGPT3 not in text and PROTGPT3 not in protein
    assert all(name in PANEL for name in protein)
    assert all(name in PANEL for name in text)
    assert STAGED_SCALE_ARMS == ("progen2-large", "progen2-xlarge")
    assert STAGED_SECOND_STAGE_ARMS == (
        "qwen2.5-7b",
        "qwen2.5-32b",
        "proteinglm-7b-clm",
        "rita-xl",
    )


def test_candidate_overlap_with_an_older_door_is_refused_at_import():
    import src.transfer.arms as arms_mod

    original = arms_mod.STAGED_CANDIDATE_ARMS
    try:
        arms_mod.STAGED_CANDIDATE_ARMS = original + ("qwen2.5-7b",)
        with pytest.raises(AssertionError, match="not a widening of the second stage"):
            arms_mod._check_second_stage_arms()
        arms_mod.STAGED_CANDIDATE_ARMS = original + ("progen2-large",)
        with pytest.raises(AssertionError, match="not a widening of the first round"):
            arms_mod._check_second_stage_arms()
    finally:
        arms_mod.STAGED_CANDIDATE_ARMS = original
    arms_mod._check_second_stage_arms()


def test_a_missing_local_checkpoint_fails_closed():
    missing = replace(STAGED_ARMS[QWEN3], path=Path("/tmp/absent-qwen3-8b-base-candidate"))
    with pytest.raises(FileNotFoundError, match="TRANSFER_"):
        load_arm_spec(missing, device="cpu", dtype="float32")
    with pytest.raises(FileNotFoundError, match="TRANSFER_"):
        require_input_path(missing.path, missing.path_variable)


def test_truncation_scheduling_follows_declared_head_width():
    import panel_contract as pc

    assert pc.STAGED_OUTPUT_LOGIT_WIDTH[QWEN3] == 151936
    assert pc.STAGED_OUTPUT_LOGIT_WIDTH[PROTGPT3] == 31
    assert pc.skips_truncation_curve(QWEN3) is True
    assert pc.skips_truncation_curve(PROTGPT3) is False
    assert pc.staged_cohort_power_args(QWEN3) == ("--skip-truncation",)
    assert pc.staged_cohort_power_args(PROTGPT3) == ()


@pytest.mark.parametrize("name", list(STAGED_CANDIDATE_ARMS))
def test_candidate_declaration_matches_on_disk_config(name):
    spec = STAGED_ARMS[name]
    config_path = spec.path / "config.json"
    if not config_path.is_file():
        pytest.skip(f"{name} is not staged on this host")
    declared = json.loads(config_path.read_text(encoding="utf-8"))
    from src.transfer.arms import config_shape

    n_layer, d_model = config_shape(SimpleNamespace(**declared))
    assert (n_layer, d_model) == (spec.n_layer, spec.d_model)
    assert spec.scoring_target_alphabet_size == int(declared["vocab_size"])


# ------------------------------------------------------------------ tokenizers


def _local_tokenizer(name: str):
    spec = STAGED_ARMS[name]
    if not (spec.path / "tokenizer.json").is_file():
        pytest.skip(f"{name} tokenizer is not staged on this host")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(spec.path), trust_remote_code=False, local_files_only=True
    )


def test_qwen3_tokenizer_has_no_bos_prefix_and_pads_with_eos():
    tokenizer = _local_tokenizer(QWEN3)
    assert tokenizer.bos_token_id is None
    assert tokenizer.pad_token_id == tokenizer.eos_token_id == 151643
    arm = Arm(
        spec=STAGED_ARMS[QWEN3],
        model=SimpleNamespace(config=SimpleNamespace(vocab_size=151936)),
        tokenizer=tokenizer,
        device="cpu",
        dtype="float32",
    )
    short, long = "Short.", "A rather longer document about proteins."
    ids, mask = tokenize_batch(arm, [short, long], 32)
    assert ids.shape[0] == 2
    assert int(mask[0].sum()) < int(mask[1].sum())
    pad = int(tokenizer.pad_token_id)
    assert int(ids[0, -1]) == pad
    assert pad not in tokenizer(short, return_tensors=None)["input_ids"]
    assert tokenizer.bos_token_id not in tokenizer(short, return_tensors=None)["input_ids"]


def test_protgpt3_encodes_aa20_one_token_each_without_bos():
    tokenizer = _local_tokenizer(PROTGPT3)
    assert tokenizer.add_bos_token is False
    assert tokenizer.bos_token_id == 1
    assert tokenizer.eos_token_id == 2
    assert tokenizer.pad_token_id == 0
    encodings = {
        residue: tokenizer(residue, return_tensors=None)["input_ids"] for residue in AA20
    }
    assert all(len(ids) == 1 for ids in encodings.values())
    assert all(tokenizer.decode(ids) == residue for residue, ids in encodings.items())
    assert tokenizer("J", return_tensors=None)["input_ids"] == [3]
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    ids = tokenizer(sequence, return_tensors=None)["input_ids"]
    assert ids[0] != tokenizer.bos_token_id
    assert len(ids) == 20
    assert max(ids) < 31
    assert 33 not in ids
    arm = Arm(
        spec=STAGED_ARMS[PROTGPT3],
        model=SimpleNamespace(config=SimpleNamespace(vocab_size=31)),
        tokenizer=tokenizer,
        device="cpu",
        dtype="float32",
    )
    cohort = Cohort(
        name="stub", kind="protein", records=["ACDE", "MKTAYI"], min_symbols=0, max_symbols=8
    )
    rendered = cohort.input_strings(arm)
    assert rendered == ["ACDE", "MKTAYI"]
    batched, mask = tokenize_batch(arm, rendered, 16)
    assert batched.shape[0] == 2
    assert int(mask[0].sum()) == 4
    assert int(mask[1].sum()) == 6
    assert int(batched[0, 4]) == 0
    keep = sequence_target_mask(batched, mask, rule=target_rule("raw"))
    # First residue is context-only under raw causal scoring; pads are excluded.
    assert int(keep[0].sum()) == 3
    assert int(keep[1].sum()) == 5


class _NoPadTokenizer:
    pad_token_id = None

    def __call__(self, text: str, return_tensors=None) -> dict[str, list[int]]:
        return {"input_ids": [2, 3]}


def test_tokenize_batch_refuses_empty_and_unpadded_candidate_batches():
    arm = _tiny_qwen3()
    with pytest.raises(ValueError, match="empty batch"):
        tokenize_batch(arm, [], 8)
    unpadded = replace(arm, tokenizer=_NoPadTokenizer())
    with pytest.raises(ValueError, match="no pad token"):
        tokenize_batch(unpadded, ["ab"], 8)
    with pytest.raises(ValueError, match="no tokens"):
        tokenize_batch(arm, ["ab", ""], 8)
    protein = Cohort(name="stub", kind="protein", records=["ACDE"], min_symbols=0, max_symbols=8)
    with pytest.raises(ValueError, match="protein cohort given to a text arm"):
        protein.input_strings(arm)


# ------------------------------------------------------------------ tiny forwards


def _manual_nll(arm: Arm, texts: list[str], max_len: int) -> torch.Tensor:
    ids, mask = tokenize_batch(arm, texts, max_len)
    with torch.no_grad():
        logits = arm.model(input_ids=ids, attention_mask=mask).logits
    logprobs = torch.nn.functional.log_softmax(logits[:, :-1].float(), dim=-1)
    target = ids[:, 1:]
    nll = -logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    keep = sequence_target_mask(ids, mask, rule=target_rule(arm.spec.input_format))
    return nll[keep]


@pytest.mark.parametrize("factory", [_tiny_qwen3, _tiny_mixtral])
def test_tiny_candidate_forwards_match_the_scored_mask(factory):
    arm = factory()
    texts = ["ab", "abcd"]
    scored = scored_tokens(arm, texts, max_len=16, batch_size=2)
    manual = _manual_nll(arm, texts, 16)
    assert scored.target_ids.size > 0
    assert torch.isfinite(manual).all()
    assert scored.nll_nats == pytest.approx(manual.detach().cpu().numpy(), abs=1e-5)
    ids, mask = tokenize_batch(arm, texts, 16)
    assert int(mask[0].sum()) < int(mask[1].sum())
    assert int(ids[0, int(mask[0].sum())]) == arm.tokenizer.pad_token_id
