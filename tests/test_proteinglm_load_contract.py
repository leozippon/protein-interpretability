"""ProteinGLM load contracts: declared shape, serving config, no 7B weights.

These tests never load released 7B shards. Shape-mismatch uses a tiny derived
model returned at the ``load_pretrained`` boundary. Matching success materialises
a 2L/32d, seq_length=1024 checkpoint under ``/tmp``.
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer import arms as A  # noqa: E402
from src.transfer import proteinglm as pglm  # noqa: E402
from src.transfer.arms import (  # noqa: E402
    STAGED_ARMS,
    ArmSpec,
    load_arm_spec,
    require_declared_shape,
)

NAME = "proteinglm-7b-clm"
_FROZEN_SOURCE = Path("/Data/public/models_R2/proteinglm-7b-clm")
_TOKENIZER_FILES = (
    "tokenization_proteinglm.py",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def _require_source() -> Path:
    candidates = (STAGED_ARMS[NAME].path, _FROZEN_SOURCE)
    for path in candidates:
        if (path / "modeling_proteinglm.py").is_file() and (
            path / "tokenizer.model"
        ).is_file():
            return path
    pytest.skip("ProteinGLM source files are not staged on this host")


def _tiny_payload(
    source: Path, *, num_layers: int = 2, hidden_size: int = 32, seq_length: int = 1024
) -> dict:
    model = pglm.tiny_causal_lm(
        source,
        num_layers=num_layers,
        hidden_size=hidden_size,
        seq_length=seq_length,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(source), trust_remote_code=True, local_files_only=True
    )
    return {
        "model": model,
        "tokenizer": tokenizer,
        "config": model.config,
        "strict_load": None,
        "derived": {"package_name": "tiny-payload-not-from-disk"},
        "max_length": {
            "filled_max_length_from_seq_length": False,
            "max_length": 20,
            "seq_length": seq_length,
        },
        "attn_implementation": None,
    }


def _copy_serving_sidecar(source: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in pglm.SOURCE_FILENAMES + _TOKENIZER_FILES:
        shutil.copy2(source / name, dest / name)


def _materialize_tiny_checkpoint(
    dest: Path,
    source: Path,
    *,
    num_layers: int = 2,
    hidden_size: int = 32,
    seq_length: int = 1024,
) -> Path:
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    _copy_serving_sidecar(source, dest)
    model = pglm.tiny_causal_lm(
        source,
        num_layers=num_layers,
        hidden_size=hidden_size,
        seq_length=seq_length,
    )
    model.save_pretrained(dest, safe_serialization=True)
    return dest


def test_tiny_loaded_model_cannot_be_wrapped_as_declared_7b(monkeypatch):
    """The unfixed path returned Arm(spec=36L/4096d) around a 2L/32d model."""

    source = _require_source()
    spec = replace(STAGED_ARMS[NAME], path=source)
    payload = _tiny_payload(source)
    monkeypatch.setattr(A._proteinglm, "load_pretrained", lambda *args, **kwargs: payload)
    with pytest.raises(ValueError, match="declared 36L/4096d, loaded 2L/32d"):
        load_arm_spec(spec, device="cpu", dtype="float32")


def test_matching_tiny_spec_loads_a_1024_window_checkpoint(tmp_path):
    source = _require_source()
    dest = _materialize_tiny_checkpoint(tmp_path / "tiny-pglm", source)
    spec = replace(
        STAGED_ARMS[NAME],
        path=dest,
        n_layer=2,
        d_model=32,
    )
    arm = load_arm_spec(spec, device="cpu", dtype="float32")
    assert int(arm.model.config.num_layers) == 2
    assert int(arm.model.config.hidden_size) == 32
    assert int(arm.model.config.seq_length) == pglm.CONTEXT_LENGTH
    assert arm.n_layer == 2
    assert arm.d_model == 32
    pglm.require_serving_config(arm.model.config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("is_causal", False),
        ("rotary_embedding_2d", True),
        ("seq_length", 64),
        ("quantization_bit", 8),
        ("moe", True),
    ],
)
def test_serving_config_refuses_each_non_contract_field(field, value):
    source = _require_source()
    config = pglm.tiny_config(source, num_layers=2, hidden_size=32, seq_length=1024)
    pglm.require_serving_config(config)
    setattr(config, field, value)
    with pytest.raises(ValueError, match=field):
        pglm.require_serving_config(config)


def test_tiny_helper_may_use_a_short_window_but_pretrained_may_not():
    source = _require_source()
    short = pglm.tiny_config(source, num_layers=1, hidden_size=32, seq_length=64)
    assert int(short.seq_length) == 64
    with pytest.raises(ValueError, match="seq_length"):
        pglm.require_serving_config(short)
    legal = pglm.tiny_config(source, num_layers=1, hidden_size=32, seq_length=1024)
    pglm.require_serving_config(legal)


def test_load_pretrained_refuses_short_context_before_claiming_the_7b_contract(
    tmp_path,
):
    source = _require_source()
    dest = _materialize_tiny_checkpoint(
        tmp_path / "tiny-short", source, seq_length=64
    )
    with pytest.raises(ValueError, match="seq_length"):
        pglm.load_pretrained(dest, device="cpu", dtype="float32")


def test_ordinary_loader_still_refuses_a_shape_mismatch_before_weights(
    tmp_path, monkeypatch
):
    spec = ArmSpec(
        name="progen2-medium",
        path=tmp_path,
        path_variable="TRANSFER_MODEL_BASE_DIR",
        modality="protein",
        n_layer=27,
        d_model=1536,
        tokenisation="residue",
        input_format="n_to_c_control",
        evaluation_cohort_source="swissprot",
        architecture="progen",
        pretraining_corpus="uniref90_bfd30",
    )
    called = {"weights": False}

    monkeypatch.setattr(
        A.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(n_layer=2, n_embd=32),
    )

    def fake_model(*_args, **_kwargs):
        called["weights"] = True
        raise AssertionError("ordinary path must not load weights after a shape mismatch")

    monkeypatch.setattr(A.AutoModelForCausalLM, "from_pretrained", fake_model)
    with pytest.raises(ValueError, match="declared 27L/1536d, loaded 2L/32d"):
        load_arm_spec(spec, device="cpu", dtype="float32")
    assert called["weights"] is False


def test_require_declared_shape_is_the_shared_helper():
    spec = replace(STAGED_ARMS[NAME], n_layer=36, d_model=4096)
    require_declared_shape(spec, SimpleNamespace(num_layers=36, hidden_size=4096))
    with pytest.raises(ValueError, match="declared 36L/4096d, loaded 2L/32d"):
        require_declared_shape(spec, SimpleNamespace(num_layers=2, hidden_size=32))
