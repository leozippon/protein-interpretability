from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import torch
import torch.nn as nn

from src.revision.dictionary_controls import WindowedTranscoder
from src.revision.dictionary_fidelity import (
    analysis_layer,
    cluster_bootstrap,
    encode_source,
    fidelity_metrics,
    hash_payload,
    load_jsonl,
    reconstruct_target,
    sequence_target_mask,
    source_layers_for_target,
    verify_model_artifacts,
)
from src.revision.io import sha256_file


@cache
def _load_runner():
    path = Path(__file__).parents[1] / "scripts" / "79_run_dictionary_fidelity.py"
    spec = importlib.util.spec_from_file_location("dictionary_fidelity_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "method", ["topk_clt", "relu_l1_sae", "gated_sae", "dense_low_rank"]
)
def test_single_source_encoding_matches_full_encoder(method: str) -> None:
    torch.manual_seed(7)
    dictionary = WindowedTranscoder(
        method=method,
        n_layers=4,
        input_dim=5,
        target_dim=5,
        width=3 if method == "dense_low_rank" else 8,
        window=3,
        topk_k=2,
    )
    inputs = [torch.randn(6, 5) for _ in range(4)]
    expected = dictionary.encode(inputs)
    for layer, values in enumerate(inputs):
        observed = encode_source(dictionary, layer, values, activation_threshold=0.0)
        torch.testing.assert_close(observed, expected[layer])


@pytest.mark.parametrize(
    "method", ["topk_clt", "relu_l1_sae", "gated_sae", "dense_low_rank"]
)
def test_target_reconstruction_matches_full_windowed_decode(method: str) -> None:
    torch.manual_seed(11)
    dictionary = WindowedTranscoder(
        method=method,
        n_layers=5,
        input_dim=4,
        target_dim=4,
        width=2 if method == "dense_low_rank" else 7,
        window=3,
        topk_k=2,
    )
    sequence_inputs = [torch.randn(2, 3, 4) for _ in range(5)]
    flat_inputs = [values.reshape(-1, 4) for values in sequence_inputs]
    expected, _ = dictionary.reconstruct(flat_inputs)
    target = 3
    sources = source_layers_for_target(
        target, n_layers=dictionary.n_layers, window=dictionary.window
    )
    observed = reconstruct_target(
        dictionary,
        {layer: sequence_inputs[layer] for layer in sources},
        target_layer=target,
        activation_threshold=0.0,
    )
    torch.testing.assert_close(observed.reshape(-1, 4), expected[target])


def test_analysis_layer_uses_half_up_rounding() -> None:
    assert analysis_layer(36, 0.5) == 18
    assert analysis_layer(27, 0.5) == 13


def test_zymctrl_target_mask_excludes_prompt_and_end() -> None:
    ids = torch.tensor([[7, 2, 3, 40, 41, 4, 0]])
    attention = torch.tensor([[1, 1, 1, 1, 1, 1, 0]])
    mask = sequence_target_mask(
        ids,
        attention,
        model_name="zymctrl",
        start_token_id=3,
        end_token_id=4,
    )
    assert mask.tolist() == [[False, False, True, True, False, False]]


def test_fidelity_inputs_use_native_conditioning() -> None:
    runner = _load_runner()
    sequence = "ACDE"
    record = {
        "id": "P0|1.2.3.4",
        "source": "test",
        "sequence": sequence,
        "split": "evaluation",
        "family": "1.2.3.4",
        "sha256": hashlib.sha256(sequence.encode()).hexdigest(),
    }
    assert runner.format_fidelity_input(record, "sequence") == sequence
    assert (
        runner.format_fidelity_input(record, "zymctrl_ec")
        == "1.2.3.4<sep><start>ACDE<end>"
    )
    assert runner.format_fidelity_input(record, "progen2_n_to_c") == "1ACDE"


def test_cohort_builder_excludes_prior_rows_and_source_prefix(tmp_path: Path) -> None:
    runner = _load_runner()
    sequences = ["ACDE", "FGHI", "KLMN", "PQRS", "TVWY", "AAAA", "CCCC", "DDDD"]
    fasta = tmp_path / "source.fasta"
    fasta.write_text(
        "".join(
            f">P{index}|1.1.1.1\n1.1.1.1<sep><start>{sequence}<end>\n"
            for index, sequence in enumerate(sequences)
        )
    )
    excluded = tmp_path / "excluded.jsonl"
    excluded.write_text(
        f'{{"sha256":"{hashlib.sha256(sequences[2].encode()).hexdigest()}"}}\n'
    )
    output = tmp_path / "cohort"
    runner.prepare_cohort(
        SimpleNamespace(
            output_dir=output,
            count=2,
            min_length=4,
            max_length=4,
            exclude_source_prefix_records=2,
            fasta=fasta,
            source_sha256=None,
            exclude_jsonl=[excluded],
        )
    )
    selected = {row["sequence"] for row in load_jsonl(output / "evaluation.jsonl")}
    assert len(selected) == 2
    assert not selected & set(sequences[:3])


def test_cohort_builder_skips_noncanonical_sequences(tmp_path: Path) -> None:
    runner = _load_runner()
    fasta = tmp_path / "source.fasta"
    fasta.write_text(
        ">bad|1.1.1.1\n1.1.1.1<sep><start>ACDX<end>\n"
        ">good|1.1.1.1\n1.1.1.1<sep><start>ACDE<end>\n"
    )
    excluded = tmp_path / "excluded.jsonl"
    excluded.write_text(
        '{"sha256":"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"}\n'
    )
    output = tmp_path / "cohort"
    runner.prepare_cohort(
        SimpleNamespace(
            output_dir=output,
            count=1,
            min_length=4,
            max_length=4,
            exclude_source_prefix_records=0,
            fasta=fasta,
            source_sha256=None,
            exclude_jsonl=[excluded],
        )
    )
    rows = load_jsonl(output / "evaluation.jsonl")
    assert [row["sequence"] for row in rows] == ["ACDE"]


def test_prepare_mean_reads_model_identity_from_activation_provenance(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    target = np.asarray([[1.0, 3.0], [5.0, 7.0]], dtype=np.float16)
    shard = tmp_path / "train" / "layer_001" / "target.npy"
    shard.parent.mkdir(parents=True)
    np.save(shard, target, allow_pickle=False)
    manifest = {
        "activation_provenance": {
            "model_name": "protgpt2",
            "model_config_sha256": "1" * 64,
            "model_weights_sha256": "2" * 64,
            "tokenizer_sha256": "3" * 64,
        },
        "content_sha256": "4" * 64,
        "selected_layers": [0, 1, 2],
        "shards": [
            {
                "split": "train",
                "layer": 1,
                "target_path": "train/layer_001/target.npy",
                "target_sha256": sha256_file(shard),
                "rows": 2,
                "target_dim": 2,
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "mean"
    runner.prepare_mean(
        SimpleNamespace(
            output_dir=output,
            cache_manifest=manifest_path,
            model_name="protgpt2",
            analysis_layer_fraction=0.5,
            chunk_rows=1,
        )
    )
    np.testing.assert_array_equal(
        np.load(output / "target_mean.npy", allow_pickle=False),
        np.asarray([3.0, 5.0], dtype=np.float32),
    )


def test_target_splicer_reinjection_is_exact() -> None:
    runner = _load_runner()

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = nn.Linear(4, 4)

    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([Block() for _ in range(3)])
            self.output = nn.Linear(4, 7)

        def forward(self, input_ids, attention_mask):
            del attention_mask
            hidden = torch.nn.functional.one_hot(input_ids, num_classes=4).float()
            for block in self.blocks:
                hidden = hidden + block.mlp(hidden)
            return SimpleNamespace(logits=self.output(hidden))

    model = ToyModel().eval()
    wrapper = SimpleNamespace(
        model=model,
        _get_block=lambda layer: model.blocks[layer],
    )
    dictionary = WindowedTranscoder(
        method="dense_low_rank",
        n_layers=3,
        input_dim=4,
        target_dim=4,
        width=2,
        window=2,
    ).eval()
    ids = torch.tensor([[0, 1, 2, 3]])
    mask = torch.ones_like(ids)
    with runner.TargetSplicer(
        wrapper,
        dictionary,
        target_layer=1,
        activation_threshold=0.0,
        target_mean=torch.zeros(4),
    ) as splicer:
        clean = splicer.forward("clean", ids, mask)
        reinjected = splicer.forward("reinject", ids, mask)
        ablated = splicer.forward("mean_ablate", ids, mask)
    assert torch.equal(clean, reinjected)
    assert not torch.equal(clean, ablated)


def test_model_artifact_verification(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    weights = tmp_path / "pytorch_model.bin"
    tokenizer = tmp_path / "tokenizer.json"
    config.write_text("{}")
    weights.write_bytes(b"weights")
    tokenizer.write_text("{}")

    def tree(path: Path) -> str:
        return hash_payload(
            [
                {
                    "path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            ]
        )

    expected = {
        "model_config_sha256": sha256_file(config),
        "model_weights_sha256": tree(weights),
        "tokenizer_sha256": tree(tokenizer),
    }
    assert verify_model_artifacts(tmp_path, expected) == expected


def _row(clean: float, variant: float, kl: float) -> dict[str, float | int]:
    return {
        "token_count": 1,
        "clean_nll_sum": clean,
        "variant_nll_sum": variant,
        "kl_sum": kl,
        "argmax_agreement_count": 1,
    }


def test_fidelity_metrics_and_cluster_bootstrap_are_paired() -> None:
    dictionary = [_row(1.0, 1.2, 0.1), _row(2.0, 2.4, 0.2)]
    mean = [_row(1.0, 2.0, 0.5), _row(2.0, 4.0, 1.0)]
    metrics = fidelity_metrics(
        dictionary,
        mean,
        minimum_ce_denominator=0.1,
        minimum_kl_denominator=0.1,
    )
    assert metrics["denominators_valid"] is True
    assert math.isclose(float(metrics["loss_recovered"]), 0.8)
    assert math.isclose(float(metrics["kl_recovered"]), 0.8)
    first = cluster_bootstrap(
        dictionary,
        mean,
        samples=50,
        seed=17,
        minimum_ce_denominator=0.1,
        minimum_kl_denominator=0.1,
    )
    second = cluster_bootstrap(
        dictionary,
        mean,
        samples=50,
        seed=17,
        minimum_ce_denominator=0.1,
        minimum_kl_denominator=0.1,
    )
    assert first == second
    assert first["invalid_denominator_samples"] == 0


def test_small_mean_ablation_denominator_is_rejected() -> None:
    dictionary = [_row(1.0, 1.0, 0.0)]
    mean = [_row(1.0, 1.001, 0.001)]
    metrics = fidelity_metrics(
        dictionary,
        mean,
        minimum_ce_denominator=0.01,
        minimum_kl_denominator=0.01,
    )
    assert metrics["denominators_valid"] is False
    assert metrics["loss_recovered"] is None
    assert metrics["kl_recovered"] is None
