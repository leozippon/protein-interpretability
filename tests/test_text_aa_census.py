"""CPU contract for the tokenizer-only 13-text ProteinGym census.

No published weights are loaded. Real local tokenizers may be read. Tiny
synthetic ProteinGym CSVs stand in for LOOKUP replay. This is not a full
census and not a fitness result.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
import pytest
from transformers import AutoTokenizer, GPT2Config

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer.arms import arm_spec  # noqa: E402
from src.transfer.fitness import load_assay  # noqa: E402
from src.transfer.precision_policy import TEXT_AA_FP32_V1  # noqa: E402
from src.transfer.text_aa_cohort import (  # noqa: E402
    ENCODE_FAIL,
    EXCEEDS_HARD_CONTEXT,
    UNIMPLEMENTED_PHASES,
    census_one_model,
    load_text_aa_boundary_table,
    support_sets,
    text_aa_model_names,
)
from src.transfer.text_aa_fitness import (  # noqa: E402
    encode_text_aa,
    load_text_aa_scorer,
    load_text_aa_tokenizer,
    resolve_text_aa_boundary,
)

GPT2_DIR = Path("/Data/public/gpt2")
BYGPT5_DIR = Path("/Data/public/bygpt5-small-en")


def _require_dir(path: Path) -> Path:
    if not path.is_dir():
        pytest.skip(f"local tokenizer directory is absent: {path}")
    return path


def _load_cli():
    path = REPO_ROOT / "scripts/transfer/text_aa_dms.py"
    spec = importlib.util.spec_from_file_location("text_aa_dms", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_native():
    path = REPO_ROOT / "scripts/transfer/native_dms_extension.py"
    spec = importlib.util.spec_from_file_location("native_dms_extension_census_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_stage20():
    path = REPO_ROOT / "scripts/transfer/20_retrieval_bound.py"
    spec = importlib.util.spec_from_file_location("stage20_census_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_assay_csv(
    directory: Path,
    name: str,
    *,
    wildtype: str,
    mutants: list[tuple[str, str]],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["mutant", "mutated_sequence", "DMS_score", "DMS_score_bin"],
        )
        writer.writeheader()
        for mutant, sequence in mutants:
            writer.writerow(
                {
                    "mutant": mutant,
                    "mutated_sequence": sequence,
                    "DMS_score": 0.5,
                    "DMS_score_bin": "1",
                }
            )


def _write_wildtypes(path: Path, rows: list[tuple[str, str, str, int]]) -> Path:
    assay_to_wildtype = {name: ident for name, _sequence, ident, _cluster in rows}
    table: dict[str, dict] = {}
    for name, sequence, ident, cluster in rows:
        entry = table.setdefault(
            ident, {"length": len(sequence), "cluster": cluster, "assays": []}
        )
        entry["assays"].append(name)
    path.write_text(
        json.dumps(
            {
                "stage": "wildtypes",
                "assay_to_wildtype": assay_to_wildtype,
                "wildtypes": table,
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_fasta(path: Path, rows: list[tuple[str, str, str, int]]) -> Path:
    seen: dict[str, str] = {}
    chunks: list[str] = []
    for _name, sequence, ident, _cluster in rows:
        if ident in seen:
            continue
        seen[ident] = sequence
        chunks.append(f">{ident}\n{sequence}\n")
    path.write_text("".join(chunks), encoding="utf-8")
    return path


def _digest(directory: Path, name: str, *, seed: int) -> str:
    native = _load_native()
    assay = load_assay(name, n=1000, seed=seed, directory=directory)
    return native.mutant_digest(assay.mutants)


def _lookup_row(name: str, *, index: int, digest: str, n_variants: int) -> dict:
    return {
        "assay": name,
        "cluster": index,
        "mutant_digest": digest,
        "n_variants": n_variants,
        "wildtype_id": f"q{index:05d}",
        "spearman": {"lookup": 0.1, "blosum62": 0.05},
    }


def _freeze(tmp_path: Path, gym: Path, rows: list[tuple[str, str, str, int]]):
    native = _load_native()
    seed = native.VARIANT_SEED
    assays = []
    for index, (name, _sequence, _ident, _cluster) in enumerate(rows):
        digest = _digest(gym, name, seed=seed + index)
        n_variants = len(load_assay(name, n=1000, seed=seed + index, directory=gym).sequences)
        assays.append(
            _lookup_row(name, index=index, digest=digest, n_variants=n_variants)
        )
    lookup = {"assays": assays}
    wildtypes_path = _write_wildtypes(tmp_path / "wildtypes.json", rows)
    fasta_path = _write_fasta(tmp_path / "wildtypes.faa", rows)
    frozen = native.freeze_lookup_cohort(
        lookup,
        proteingym_dir=gym,
        wildtypes=json.loads(wildtypes_path.read_text(encoding="utf-8")),
        wildtypes_path=wildtypes_path,
        wildtypes_fasta_path=fasta_path,
    )
    return frozen, wildtypes_path, fasta_path, lookup


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    path = _require_dir(GPT2_DIR)
    return AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=False
    )


@pytest.fixture(scope="module")
def bygpt5_tokenizer():
    path = _require_dir(BYGPT5_DIR)
    return AutoTokenizer.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=True
    )


def test_boundary_table_has_thirteen_and_no_path_source():
    table = load_text_aa_boundary_table()
    names = text_aa_model_names(table)
    assert names == (
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl",
        "dialogpt-small",
        "qwen2.5-0.5b",
        "qwen2.5-7b",
        "qwen2.5-32b",
        "llama-3.2-3b",
        "qwen3-8b-base",
        "bygpt5-small-en",
        "bygpt5-base-en",
        "bygpt5-medium-en",
    )
    qwen3 = resolve_text_aa_boundary("qwen3-8b-base", table)
    assert qwen3.model_type == "qwen3"
    for spec in table["text_aa_checkpoints"].values():
        assert "local_dir" not in spec
        assert "path" not in spec
    assert arm_spec("qwen3-8b-base").path.name == "Qwen3-8B-Base"
    assert arm_spec("qwen3-8b-base").path_variable == "TRANSFER_TEXT_MODEL_BASE_DIR"


def test_load_text_aa_tokenizer_does_not_need_weights(gpt2_tokenizer, tmp_path):
    checkpoint = tmp_path / "tok-only"
    checkpoint.mkdir()
    for name in (
        "vocab.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        src = GPT2_DIR / name
        if src.is_file():
            (checkpoint / name).write_bytes(src.read_bytes())
    config = GPT2Config(
        vocab_size=len(gpt2_tokenizer),
        n_positions=64,
        n_embd=32,
        n_layer=1,
        n_head=4,
        bos_token_id=50256,
        eos_token_id=50256,
    )
    config.save_pretrained(checkpoint)
    table = load_text_aa_boundary_table()
    boundary = resolve_text_aa_boundary("gpt2", table)
    bundle = load_text_aa_tokenizer(checkpoint, boundary=boundary, name="gpt2")
    assert bundle.hard_context == 64
    assert bundle.tokenizer is not None
    assert not list(checkpoint.glob("*.safetensors"))
    assert not list(checkpoint.glob("pytorch_model.bin"))
    with pytest.raises(Exception):
        load_text_aa_scorer(
            checkpoint,
            boundary=boundary,
            application_window_tokens=64,
            device="cpu",
            name="gpt2",
        )


def test_mutant_over_window_excludes_whole_assay_and_keeps_digest(
    gpt2_tokenizer, tmp_path
):
    native = _load_native()
    gym = tmp_path / "gym"
    _write_assay_csv(
        gym,
        "short_wt",
        wildtype="AAA",
        mutants=[("A1W", "WAA"), ("A3G", "AAG")],
    )
    rows = [("short_wt", "AAA", "q00000", 0)]
    frozen, *_ = _freeze(tmp_path, gym, rows)
    table = load_text_aa_boundary_table()
    boundary = resolve_text_aa_boundary("gpt2", table)
    wt_ids = encode_text_aa(gpt2_tokenizer, "AAA", boundary)
    mut_ids = encode_text_aa(gpt2_tokenizer, "WAA", boundary)
    assert len(wt_ids) < len(mut_ids)
    payload = census_one_model(
        "gpt2",
        tokenizer=gpt2_tokenizer,
        boundary=boundary,
        hard_context=len(wt_ids),
        request=frozen,
        documented_context=len(wt_ids),
    )
    row = payload["assays"][0]
    assert row["admitted"] is False
    assert row["exclude_reason"] == EXCEEDS_HARD_CONTEXT
    assert row["n_variants"] == frozen["assays"][0]["n_variants"]
    assert row["mutant_digest"] == frozen["assays"][0]["mutant_digest"]
    assert row["seed"] == native.VARIANT_SEED
    assert payload["native_fixed"] == []
    assert frozen["sequences"]["short_wt"] == list(
        load_assay(
            "short_wt", n=1000, seed=native.VARIANT_SEED, directory=gym
        ).sequences
    )


def test_bygpt5_measures_length_then_window_and_wt_tie(
    bygpt5_tokenizer, tmp_path
):
    gym = tmp_path / "gym"
    _write_assay_csv(
        gym,
        "first",
        wildtype="MKT",
        mutants=[("M1A", "AKT"), ("K2G", "MGT")],
    )
    _write_assay_csv(
        gym,
        "second",
        wildtype="MKT",
        mutants=[("M1A", "AKT"), ("K2G", "MGT")],
    )
    rows = [
        ("first", "MKT", "q00000", 0),
        ("second", "MKT", "q00001", 1),
    ]
    frozen, *_ = _freeze(tmp_path, gym, rows)
    table = load_text_aa_boundary_table()
    boundary = resolve_text_aa_boundary("bygpt5-small-en", table)
    payload = census_one_model(
        "bygpt5-small-en",
        tokenizer=bygpt5_tokenizer,
        boundary=boundary,
        hard_context=None,
        request=frozen,
        documented_context=None,
    )
    assert payload["window_rule"] == "max_legal_request_input"
    assert payload["hard_context"] is None
    expected = 1 + len("MKT")
    assert payload["application_window_tokens"] == expected
    probe = payload["longest_probe_identity"]
    assert probe["assay"] == "first"
    assert probe["role"] == "wildtype"
    assert probe["index"] == 0
    assert probe["n_input_tokens"] == expected
    assert "sequence" not in probe
    assert payload["native_fixed"] == ["first", "second"]


def test_illegal_encoding_excludes_and_file_errors_fail_fast(
    gpt2_tokenizer, tmp_path
):
    table = load_text_aa_boundary_table()
    boundary = resolve_text_aa_boundary("gpt2", table)
    request = {
        "assays": [
            {
                "assay": "bad",
                "index": 0,
                "cluster": 0,
                "n_variants": 1,
                "mutant_digest": "d",
                "csv_sha256": "c",
                "seed": 1,
                "wildtype_id": "q00000",
                "wildtype_sequence": "AAA",
            }
        ],
        "sequences": {"bad": ["AAZ"]},
        "declared_assays": 1,
    }
    payload = census_one_model(
        "gpt2",
        tokenizer=gpt2_tokenizer,
        boundary=boundary,
        hard_context=1024,
        request=request,
        documented_context=1024,
    )
    assert payload["assays"][0]["exclude_reason"] == ENCODE_FAIL
    assert payload["assays"][0]["failed_sequences"][0]["role"] == "mutant"
    assert payload["assays"][0]["failed_sequences"][0]["mutant_index"] == 0

    def boom(*_args, **_kwargs):
        raise MemoryError("synthetic")

    monkey_request = dict(request)
    monkey_request["sequences"] = {"bad": ["AAA"]}
    import src.transfer.text_aa_cohort as cohort

    original = cohort.encode_text_aa
    cohort.encode_text_aa = boom  # type: ignore[method-assign]
    try:
        with pytest.raises(MemoryError):
            census_one_model(
                "gpt2",
                tokenizer=gpt2_tokenizer,
                boundary=boundary,
                hard_context=1024,
                request=monkey_request,
                documented_context=1024,
            )
    finally:
        cohort.encode_text_aa = original  # type: ignore[method-assign]

    cli = _load_cli()
    with pytest.raises(FileNotFoundError):
        cli.run_census(
            out=tmp_path / "out",
            lookup_path=tmp_path / "missing-lookup.json",
            wildtypes_path=tmp_path / "missing-wt.json",
            wildtypes_fasta_path=tmp_path / "missing.faa",
            proteingym_dir=tmp_path / "gym",
            models=["gpt2"],
            requested_device="cuda:0",
        )


def test_original_seed_index_is_kept(gpt2_tokenizer, tmp_path):
    native = _load_native()
    gym = tmp_path / "gym"
    _write_assay_csv(
        gym, "one", wildtype="AAA", mutants=[("A1W", "WAA"), ("A3G", "AAG")]
    )
    _write_assay_csv(
        gym, "two", wildtype="AAA", mutants=[("A1W", "WAA"), ("A3G", "AAG")]
    )
    rows = [
        ("one", "AAA", "q00000", 0),
        ("two", "AAA", "q00001", 1),
    ]
    frozen, *_ = _freeze(tmp_path, gym, rows)
    assert [row["seed"] for row in frozen["assays"]] == [
        native.VARIANT_SEED,
        native.VARIANT_SEED + 1,
    ]
    table = load_text_aa_boundary_table()
    payload = census_one_model(
        "gpt2",
        tokenizer=gpt2_tokenizer,
        boundary=resolve_text_aa_boundary("gpt2", table),
        hard_context=1024,
        request=frozen,
        documented_context=1024,
    )
    assert [row["seed"] for row in payload["assays"]] == [
        native.VARIANT_SEED,
        native.VARIANT_SEED + 1,
    ]
    assert [row["index"] for row in payload["assays"]] == [0, 1]


def test_incomplete_models_cannot_claim_all13_or_mixed_families():
    table = load_text_aa_boundary_table()

    def _stub(name: str, admitted: list[str]) -> dict:
        return {
            "native_fixed": admitted,
            "assays": [
                {"assay": assay} for assay in ["a", "b", "c"]
            ],
        }

    partial = {
        "gpt2": _stub("gpt2", ["a", "b"]),
        "dialogpt-small": _stub("dialogpt-small", ["a", "c"]),
        "qwen3-8b-base": _stub("qwen3-8b-base", ["a", "b"]),
        "qwen2.5-0.5b": _stub("qwen2.5-0.5b", ["a"]),
    }
    sets = support_sets(partial, table=table)
    assert sets["common_all13"]["complete"] is False
    assert sets["common_all13"]["assays"] is None
    assert "gpt2-medium" in sets["common_all13"]["missing_models"]
    assert sets["families"]["gpt2"]["complete"] is False
    assert "dialogpt-small" not in table["families"]["gpt2"]
    assert "qwen3-8b-base" not in table["families"]["qwen2.5"]
    full_gpt2 = {
        name: _stub(name, ["a", "b"] if name != "gpt2-xl" else ["a"])
        for name in table["families"]["gpt2"]
    }
    family = support_sets(full_gpt2, table=table)["families"]["gpt2"]
    assert family["complete"] is True
    assert family["assays"] == ["a"]


def test_same_token_length_different_ids_change_digest(gpt2_tokenizer):
    table = load_text_aa_boundary_table()
    boundary = resolve_text_aa_boundary("gpt2", table)
    base = {
        "assays": [
            {
                "assay": "x",
                "index": 0,
                "cluster": 0,
                "n_variants": 1,
                "mutant_digest": "d",
                "csv_sha256": "c",
                "seed": 1,
                "wildtype_id": "q00000",
                "wildtype_sequence": "WWW",
            }
        ],
        "declared_assays": 1,
    }
    first = census_one_model(
        "gpt2",
        tokenizer=gpt2_tokenizer,
        boundary=boundary,
        hard_context=1024,
        request={**base, "sequences": {"x": ["KKK"]}},
        documented_context=1024,
    )
    second = census_one_model(
        "gpt2",
        tokenizer=gpt2_tokenizer,
        boundary=boundary,
        hard_context=1024,
        request={**base, "sequences": {"x": ["MMM"]}},
        documented_context=1024,
    )
    wt = encode_text_aa(gpt2_tokenizer, "WWW", boundary)
    left = encode_text_aa(gpt2_tokenizer, "KKK", boundary)
    right = encode_text_aa(gpt2_tokenizer, "MMM", boundary)
    assert len(left) == len(right) == len(wt)
    assert left != right
    assert first["assays"][0]["encoded_ids_digest"] != second["assays"][0]["encoded_ids_digest"]
    assert first["assays"][0]["n_input_tokens_max_legal"] == second["assays"][0][
        "n_input_tokens_max_legal"
    ]


def test_cli_accepts_injected_device_and_refuses_later_phases(
    gpt2_tokenizer, tmp_path, monkeypatch
):
    gym = tmp_path / "gym"
    _write_assay_csv(
        gym, "one", wildtype="AAA", mutants=[("A1W", "WAA"), ("A3G", "AAG")]
    )
    rows = [("one", "AAA", "q00000", 0)]
    frozen, wildtypes_path, fasta_path, lookup = _freeze(tmp_path, gym, rows)
    lookup_path = tmp_path / "lookup.json"
    lookup_path.write_text(json.dumps(lookup), encoding="utf-8")
    frozen["lookup_sha256"] = "0" * 64
    frozen["lookup_path"] = str(lookup_path)
    frozen["proteingym_dir"] = str(gym)
    cli = _load_cli()
    monkeypatch.setattr(
        cli,
        "freeze_request",
        lambda **_kwargs: frozen,
    )
    monkeypatch.setattr(
        "src.transfer.text_aa_cohort.resolve_text_aa_checkpoint",
        lambda name: GPT2_DIR,
    )
    _require_dir(GPT2_DIR)
    out = tmp_path / "out"
    payload = cli.main(
        [
            "--device",
            "cuda:0",
            "--out",
            str(out),
            "census",
            "--lookup",
            str(lookup_path),
            "--wildtypes",
            str(wildtypes_path),
            "--wildtypes-fasta",
            str(fasta_path),
            "--proteingym-dir",
            str(gym),
            "--model",
            "gpt2",
        ]
    )
    assert payload["execution"]["tokenizer_only"] is True
    assert payload["execution"]["weights_loaded"] is False
    assert payload["execution"]["cuda_invoked"] is False
    assert payload["execution"]["requested_device"] == "cuda:0"
    assert payload["execution"]["executed_on"] == "cpu"
    assert payload["support_sets"]["common_all13"]["complete"] is False
    assert payload["support_sets"]["common_all13"]["assays"] is None
    assert payload["experiment_admitted"] is False
    assert (out / "text_aa_census.json").is_file()
    with pytest.raises(SystemExit, match="not implemented"):
        cli.main(
            [
                "--device",
                "cuda:0",
                "--out",
                str(out),
                "probe",
                "--lookup",
                str(lookup_path),
                "--wildtypes",
                str(wildtypes_path),
                "--wildtypes-fasta",
                str(fasta_path),
            ]
        )
    assert UNIMPLEMENTED_PHASES == ("probe", "score", "analyse")


def test_old_default_doors_are_unchanged():
    stage20 = _load_stage20()
    from src.transfer.arms import PANEL, STAGED_CANDIDATE_ARMS

    assert sorted(stage20.ARM_CORPUS) == ["progen2-medium", "progen3-112m", "protgpt2"]
    for name in text_aa_model_names():
        assert name not in stage20.SCOREABLE_ARMS
        assert name not in stage20.ARM_CORPUS
    assert "gpt2" in PANEL
    assert STAGED_CANDIDATE_ARMS == ("qwen3-8b-base", "protgpt3-1.3b")
    native = _load_native()
    with pytest.raises(ValueError, match="unknown native DMS protocol"):
        native.required_dtype("gpt2", TEXT_AA_FP32_V1)
