"""Stage-20 ProteinGym doors for remaining non-text main-table arms.

No weights, no GPU, no ρ. These tests pin format refusals, EC lookup, the
residues-2..L mask, and parser/default isolation.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer.arms import INPUT_FORMAT_GMASK_SOP_EOS, arm_spec  # noqa: E402
from src.transfer.joint_modes import RESIDUE_UNIT, resolve  # noqa: E402
from src.transfer.scale_comparison import STRATUM_N_TO_C  # noqa: E402
from src.transfer.scoring import sequence_target_mask, target_rule  # noqa: E402
from tests.test_joint_mode_qualification import instructprotein_stub  # noqa: E402


def _stage():
    path = REPO_ROOT / "scripts" / "transfer" / "20_retrieval_bound.py"
    spec = importlib.util.spec_from_file_location("_stage_20_nontxt", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DEFAULT_ARMS = ["progen2-medium", "progen3-112m", "protgpt2"]
NEW_DOORS = (
    "zymctrl",
    "proteinglm-7b-clm",
    "protgpt3-1.3b",
    "instructprotein",
)


def test_new_doors_are_explicit_and_do_not_widen_the_default_run():
    stage = _stage()
    assert sorted(stage.ARM_CORPUS) == DEFAULT_ARMS
    parser = stage.build_parser()
    assert parser.parse_args([]).arms == DEFAULT_ARMS
    for name in NEW_DOORS:
        assert name in stage.SCOREABLE_ARMS
        assert name not in stage.ARM_CORPUS
        assert parser.parse_args(["--arms", name]).arms == [name]


def test_text_and_unqualified_names_remain_refused():
    stage = _stage()
    parser = stage.build_parser()
    for name in ("qwen2.5-7b", "qwen3-8b-base", "gpt2", "galactica-125m-text"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--arms", name])
        with pytest.raises(KeyError):
            stage.corpus_record(name)
        assert name not in stage.SCOREABLE_ARMS


def test_corpus_records_are_not_retrieval_bounds():
    stage = _stage()
    zym = stage.corpus_record("zymctrl")
    assert zym["declared"] == arm_spec("zymctrl").pretraining_corpus
    assert "not a retrieval bound" in zym["identification"]
    glm = stage.corpus_record("proteinglm-7b-clm")
    assert "2..L" in glm["note"] or "residues 2" in glm["note"]
    gpt3 = stage.corpus_record("protgpt3-1.3b")
    assert gpt3["declared"] == arm_spec("protgpt3-1.3b").pretraining_corpus
    inst = stage.corpus_record("instructprotein")
    assert "protein mode only" in inst["note"]


def test_arm_scorer_refuses_ec_and_gmask_formats():
    stage = _stage()
    args = argparse.Namespace(device="cpu", dtype="bfloat16", batch_size=1)
    with pytest.raises(ValueError, match="ZymCTRL"):
        stage._ArmScorer("zymctrl", args)
    with pytest.raises(ValueError, match="residues 2..L"):
        stage._ArmScorer("proteinglm-7b-clm", args)


def test_ec_labels_by_sequence_refuses_conflict(tmp_path):
    stage = _stage()
    fasta = tmp_path / "ec.fasta"
    fasta.write_text(">P1|1.1.1.1\nACDE\n>P2|2.2.2.2\nACDE\n", encoding="utf-8")
    with pytest.raises(ValueError, match="maps to both"):
        stage.ec_labels_by_sequence(fasta)
    clean = tmp_path / "clean.fasta"
    clean.write_text(">P1|1.1.1.1\nACDE\n>P2|1.1.1.1\nACDE\n", encoding="utf-8")
    assert stage.ec_labels_by_sequence(clean) == {"ACDE": "1.1.1.1"}


def test_zymctrl_skips_missing_ec_and_refuses_unconditioned_render():
    stage = _stage()

    class _Tok:
        eos_token = None
        unk_token_id = 0
        pad_token_id = 1

        def convert_tokens_to_ids(self, token):
            return {"<start>": 10, "<end>": 11}.get(token, 0)

        def __call__(self, text, return_tensors=None):
            return {"input_ids": list(range(len(text)))}

    fake_arm = SimpleNamespace(
        name="zymctrl",
        modality="protein",
        spec=arm_spec("zymctrl"),
        tokenizer=_Tok(),
        device="cpu",
        model=SimpleNamespace(config=SimpleNamespace(n_positions=1024)),
    )
    args = argparse.Namespace(device="cpu", dtype="bfloat16", batch_size=1)
    scorer = object.__new__(stage._ZymCTRLScorer)
    scorer.torch = torch
    scorer.name = "zymctrl"
    scorer.batch_size = 1
    scorer.arm = fake_arm
    scorer.context = 1024
    scorer.start_id = 10
    scorer.end_id = 11
    scorer._ec_by_sequence = {"ACDE": "3.2.1.17"}
    scorer._bound_ec = None
    assert "absent from the EC-labelled" in scorer.bind_wildtype("MKTA")
    with pytest.raises(ValueError, match="unconditioned"):
        scorer._render(["ACDE"])
    assert scorer.bind_wildtype("ACDE") is None
    rendered = scorer._render(["ACDE", "ADDE"])
    assert rendered == [
        "3.2.1.17<sep><start>ACDE<end>",
        "3.2.1.17<sep><start>ADDE<end>",
    ]


def test_proteinglm_target_mask_drops_prefix_and_residue_one():
    ids = torch.tensor([[29, 32, 34, 5, 6, 7]], dtype=torch.long)
    mask = torch.ones_like(ids)
    keep = sequence_target_mask(
        ids, mask, rule=target_rule(INPUT_FORMAT_GMASK_SOP_EOS)
    )
    assert keep.shape == (1, 5)
    assert keep.tolist() == [[False, False, False, True, True]]
    bare = torch.tensor([[5, 6, 7, 8, 9]], dtype=torch.long)
    with pytest.raises(ValueError, match="gmask"):
        sequence_target_mask(
            bare, torch.ones_like(bare), rule=target_rule(INPUT_FORMAT_GMASK_SOP_EOS)
        )


def test_proteinglm_scorer_refuses_non_float32_and_wrong_name():
    stage = _stage()
    args = argparse.Namespace(device="cpu", dtype="bfloat16", batch_size=1)
    with pytest.raises(ValueError, match="FP32-only"):
        stage._ProteinGLMScorer("proteinglm-7b-clm", args)
    args.dtype = "float32"
    with pytest.raises(ValueError, match="proteinglm-7b-clm"):
        stage._ProteinGLMScorer("protgpt3-1.3b", args)


def test_instructprotein_scorer_uses_declared_residue_tokens_only():
    from src.transfer import instructprotein_fitness as IP

    tokenizer = instructprotein_stub()
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<protein>")
    tokenisation = resolve(tokenizer, "instructprotein")
    record = tokenisation.render("MK")
    assert "<protein>" in record.text
    assert record.n_residues == record.n_scored_tokens == 2
    start_id = tokenizer.convert_tokens_to_ids("<protein>")
    end_id = tokenizer.convert_tokens_to_ids("</protein>")
    assert start_id in record.token_ids
    assert record.token_ids[-1] == end_id
    scored_ids = {record.token_ids[i] for i in record.scored_positions}
    assert start_id not in scored_ids
    assert end_id not in scored_ids

    class _Rule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.device = torch.device("cpu")

        def forward(self, input_ids, attention_mask=None, use_cache=None):
            b, t = input_ids.shape
            logits = torch.zeros(b, t, 80, dtype=torch.float32)
            for row in range(b):
                for pos in range(t - 1):
                    logits[row, pos, int(input_ids[row, pos + 1])] = 1.0
            return SimpleNamespace(logits=logits)

    loaded = IP.LoadedInstructProtein(
        name="instructprotein",
        model=_Rule(),
        tokenizer=tokenizer,
        tokenisation=tokenisation,
        context=64,
        facts={"vocab_size": 80, "checkpoint": "stub"},
    )
    scorer = IP.InstructProteinFitnessScorer(loaded, batch_size=2)
    assert scorer.scoring_stratum == STRATUM_N_TO_C
    assert scorer.symbol_unit == RESIDUE_UNIT
    totals = scorer.log_likelihood(["MK", "AC"])
    assert totals.shape == (2,)
    assert np.all(np.isfinite(totals))
    accounting = scorer.residue_accounting()
    assert accounting["residues"] == 4
    assert accounting["scored_tokens"] == 4


def test_load_scorer_routes_new_doors(monkeypatch):
    stage = _stage()
    args = argparse.Namespace(device="cpu", dtype="bfloat16", batch_size=16)

    class _Stub:
        context = 128
        score_description = "stub"
        scoring_stratum = STRATUM_N_TO_C
        facts = {
            "checkpoint": "stub",
            "scientific_role": "stub",
            "native_eos_token_id": 2,
            "config_eos_token_id": None,
            "config_eos_token_id_status": "ignored",
        }

        def __init__(self, *a, **k):
            return

    monkeypatch.setattr(stage, "_ZymCTRLScorer", type("Z", (_Stub,), {}))
    monkeypatch.setattr(stage, "_ProteinGLMScorer", type("G", (_Stub,), {}))
    monkeypatch.setattr(stage, "_ArmScorer", type("A", (_Stub,), {}))

    class _Loaded:
        facts = {
            "checkpoint": "stub",
            "scientific_role": "stub",
            "vocab_size": 80,
        }
        context = 128

    class _IPScorer(_Stub):
        def __init__(self, loaded, *, batch_size):
            self.context = 128
            self.facts = loaded.facts

    monkeypatch.setattr(stage.IP, "load_instructprotein", lambda *a, **k: _Loaded())
    monkeypatch.setattr(stage.IP, "InstructProteinFitnessScorer", _IPScorer)

    glm_args = argparse.Namespace(device="cpu", dtype="float32", batch_size=16)
    _, _, record = stage._load_scorer("zymctrl", args)
    assert record["input_format"] == "ec_conditioned"
    _, _, record = stage._load_scorer("proteinglm-7b-clm", glm_args)
    assert record["input_format"] == INPUT_FORMAT_GMASK_SOP_EOS
    scorer, _, record = stage._load_scorer("protgpt3-1.3b", args)
    assert type(scorer).__name__ == "A"
    _, _, record = stage._load_scorer("instructprotein", args)
    assert record["input_format"].startswith("instructprotein")
