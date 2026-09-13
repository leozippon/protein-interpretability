"""CPU contract for the native ProteinGym extension helper.

No real weights, no GPU, no network. Probe uses injected scorers or tiny logit
tables. Analyse uses synthetic LOOKUP and stage-20-shaped score payloads.
"""

from __future__ import annotations

import csv
import importlib.util
import json
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

from src.transfer import galactica_fitness as G  # noqa: E402
from src.transfer.arms import AA20  # noqa: E402
from src.transfer.fitness import load_assay, wildtype_of  # noqa: E402
from src.transfer.io import sha256_file  # noqa: E402
from src.transfer.joint_modes import resolve  # noqa: E402
from src.transfer.rita_fitness import NATIVE_EOS_ID, NATIVE_PAD_ID, VOCAB_SIZE  # noqa: E402
from src.transfer.scale_comparison import STRATUM_N_TO_C  # noqa: E402
from tests.test_joint_mode_qualification import galactica_stub  # noqa: E402


def _load_script():
    path = REPO_ROOT / "scripts/transfer/native_dms_extension.py"
    spec = importlib.util.spec_from_file_location("native_dms_extension", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXT = _load_script()

AA20_IDS = {
    "L": 3,
    "A": 4,
    "G": 5,
    "V": 6,
    "E": 7,
    "S": 8,
    "I": 9,
    "K": 10,
    "R": 11,
    "D": 12,
    "T": 13,
    "P": 14,
    "N": 15,
    "Q": 16,
    "F": 17,
    "Y": 18,
    "M": 19,
    "H": 20,
    "C": 21,
    "W": 22,
}


def _write_assay_csv(directory: Path, name: str, wildtype: str = "MKT") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.csv"
    mutants = ["M1A", "K2G"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["mutant", "mutated_sequence", "DMS_score", "DMS_score_bin"]
        )
        writer.writeheader()
        for mutant, score in zip(mutants, (0.9, 0.1)):
            sequence = list(wildtype)
            for token in mutant.split(":"):
                sequence[int(token[1:-1]) - 1] = token[-1]
            writer.writerow(
                {
                    "mutant": mutant,
                    "mutated_sequence": "".join(sequence),
                    "DMS_score": score,
                    "DMS_score_bin": "1",
                }
            )


def _write_wildtypes(
    path: Path,
    rows: list[tuple[str, str, str, int]],
) -> Path:
    assay_to_wildtype = {name: ident for name, _sequence, ident, _cluster in rows}
    table: dict[str, dict] = {}
    for name, sequence, ident, cluster in rows:
        entry = table.setdefault(
            ident,
            {"length": len(sequence), "cluster": cluster, "assays": []},
        )
        if entry["length"] != len(sequence):
            raise AssertionError("test catalog length conflict")
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


def _write_wildtypes_fasta(
    path: Path, rows: list[tuple[str, str, str, int]]
) -> Path:
    seen: dict[str, str] = {}
    chunks: list[str] = []
    for _name, sequence, ident, _cluster in rows:
        if ident in seen:
            continue
        seen[ident] = sequence
        chunks.append(f">{ident}\n{sequence}\n")
    path.write_text("".join(chunks), encoding="utf-8")
    return path


def _lookup_row(name: str, *, index: int, digest: str, n_variants: int = 2) -> dict:
    return {
        "assay": name,
        "cluster": index,
        "mutant_digest": digest,
        "n_variants": n_variants,
        "wildtype_id": f"q{index:05d}",
        "spearman": {"lookup": 0.10 + 0.01 * index, "blosum62": 0.05},
    }


def _digest_from_csv(directory: Path, name: str, *, seed: int) -> str:
    assay = load_assay(name, n=1000, seed=seed, directory=directory)
    return EXT.mutant_digest(assay.mutants)


class _LengthScorer:
    def __init__(self, lengths: dict[object, list[int]], *, context: int, name: str) -> None:
        self._lengths = lengths
        self.context = context
        self.name = name
        self.batch_size = 2
        self.scoring_stratum = STRATUM_N_TO_C
        self.score_description = "test"
        self.facts = {"scientific_role": "injected test scorer", "context": context}

    def token_lengths(self, sequences):
        key = tuple(sequences)
        if key in self._lengths:
            return self._lengths[key]
        # keyed by first sequence as a stand-in when the test stored by assay
        for stored in self._lengths.values():
            if len(stored) == len(sequences):
                return stored
        return [len(sequence) for sequence in sequences]

    def log_likelihood(self, sequences):
        if sequences == ["MKTZ"] or any("Z" in sequence for sequence in sequences):
            raise ValueError("non-canonical amino-acid sequence")
        if max(self.token_lengths(sequences)) > self.context:
            raise ValueError("a rendering exceeds this checkpoint's context")
        return np.zeros(len(sequences), dtype=np.float64)

    def release(self) -> None:
        return None


class _PadTokenizer:
    def __init__(self, vocab_size: int = 50000) -> None:
        self._inner = galactica_stub()
        self._inner._vocab["<pad>"] = G.DECLARED_PAD_TOKEN_ID
        self._inner._inverse[G.DECLARED_PAD_TOKEN_ID] = "<pad>"
        self.pad_token = "<pad>"
        self._vocab_size = int(vocab_size)

    def __len__(self):
        return self._vocab_size

    def __call__(self, text, return_tensors=None):
        return self._inner(text, return_tensors=return_tensors)

    def convert_ids_to_tokens(self, index):
        return self._inner.convert_ids_to_tokens(index)

    def convert_tokens_to_ids(self, token):
        return self._inner.convert_tokens_to_ids(token)

    @property
    def pad_token_id(self):
        return self._inner.convert_tokens_to_ids(self.pad_token)


class _RuleLogits(nn.Module):
    def __init__(self, vocab_size: int, *, start_id: int, end_id: int, pad_id: int) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.start_id = int(start_id)
        self.end_id = int(end_id)
        self.pad_id = int(pad_id)
        self.device = torch.device("cpu")
        self.weight = nn.Parameter(torch.ones(1))
        self.omit_loss = False
        self.wrong_loss = False

    def forward(self, input_ids, attention_mask=None, use_cache=None, labels=None):
        assert use_cache is False
        batch, width = input_ids.shape
        position = torch.arange(width, dtype=torch.float32).view(1, width, 1)
        vocab = torch.arange(self.vocab_size, dtype=torch.float32).view(1, 1, -1)
        logits = (0.25 * position + 0.05 * vocab).expand(batch, width, self.vocab_size).contiguous()
        logits = logits + (0.0 * self.weight)
        logits[..., self.end_id] = 50.0
        logits[..., self.start_id] = 40.0
        logits[..., self.pad_id] = 100.0
        loss = None
        if labels is not None and not self.omit_loss:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:]
            loss = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, self.vocab_size),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )
            if self.wrong_loss:
                loss = loss + self.weight + 12.0
        return SimpleNamespace(logits=logits, loss=loss)


class _RitaTokenizer:
    def __init__(self) -> None:
        self._vocab = {"<unk>": 0, "<PAD>": NATIVE_PAD_ID, "<EOS>": NATIVE_EOS_ID, **AA20_IDS}
        self.eos_token_id = None
        self.pad_token_id = None

    def convert_tokens_to_ids(self, token: str) -> int:
        return int(self._vocab.get(str(token), 0))

    def convert_ids_to_tokens(self, token_id: int) -> str:
        inverse = {value: key for key, value in self._vocab.items()}
        return inverse[int(token_id)]

    def encode(self, text: str, add_special_tokens: bool = True):
        ids = [AA20_IDS[symbol] for symbol in text]
        if add_special_tokens:
            return ids + [NATIVE_EOS_ID]
        return ids


class _RitaLogits(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self.weight = nn.Parameter(torch.ones(1))
        self.forward_kwargs: list[dict] = []

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        self.forward_kwargs.append(
            {"attention_mask": attention_mask, "labels": labels, **kwargs}
        )
        batch, width = input_ids.shape
        position = torch.arange(width, dtype=torch.float32).view(1, width, 1)
        vocab = torch.arange(VOCAB_SIZE, dtype=torch.float32).view(1, 1, -1)
        logits = (0.2 * position + 0.03 * vocab).expand(batch, width, VOCAB_SIZE).contiguous()
        logits = logits + (0.0 * self.weight)
        logits[..., NATIVE_EOS_ID] = 50.0
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:]
            loss = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, VOCAB_SIZE),
                shift_labels.reshape(-1),
            )
        return SimpleNamespace(logits=logits, loss=loss)


def _galactica_scorer(sequences: list[str], *, context: int = 64, batch_size: int = 2):
    tokenizer = _PadTokenizer()
    tokenisation = resolve(tokenizer, "galactica")
    records = [tokenisation.render(sequence, context=None) for sequence in sequences]
    vocab = max(max(record.token_ids) for record in records) + 8
    tokenizer = _PadTokenizer(vocab)
    start_id = tokenizer.convert_tokens_to_ids("[START_AMINO]")
    end_id = tokenizer.convert_tokens_to_ids("[END_AMINO]")
    assert isinstance(start_id, int) and isinstance(end_id, int)
    model = _RuleLogits(vocab, start_id=start_id, end_id=end_id, pad_id=G.DECLARED_PAD_TOKEN_ID)
    loaded = G.LoadedGalactica(
        name="galactica-1.3b",
        model=model,
        tokenizer=tokenizer,
        tokenisation=resolve(tokenizer, "galactica"),
        context=context,
        facts={"rung": "galactica-1.3b", "vocab_size": vocab, "scientific_role": "test"},
    )
    return G.GalacticaFitnessScorer(loaded, batch_size=batch_size)


def _rita_scorer(*, context: int = 64, batch_size: int = 2):
    scorer = SimpleNamespace(
        name="rita-xl",
        tokenizer=_RitaTokenizer(),
        eos_id=NATIVE_EOS_ID,
        residue_ids=dict(AA20_IDS),
        model=_RitaLogits(),
        torch=torch,
        batch_size=batch_size,
        context=context,
        scoring_stratum=STRATUM_N_TO_C,
        score_description="raw sequence with tokenizer-native terminal EOS",
        facts={"scientific_role": "test rita"},
    )

    def token_lengths(sequences):
        return [len(sequence) + 1 for sequence in sequences]

    def log_likelihood(sequences):
        if any(any(symbol not in AA20 for symbol in sequence) for sequence in sequences):
            raise ValueError("non-canonical amino-acid sequence")
        encoded = [scorer.tokenizer.encode(sequence) for sequence in sequences]
        if len({len(ids) for ids in encoded}) != 1:
            raise ValueError("variable-length batch is refused")
        width = len(encoded[0])
        if width > scorer.context:
            raise ValueError("a native encoding exceeds this checkpoint's context")
        ids = torch.tensor(encoded, dtype=torch.long)
        logits = scorer.model(input_ids=ids).logits
        logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        token = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        return token.sum(1).double().cpu().numpy()

    def release():
        return None

    scorer.token_lengths = token_lengths
    scorer.log_likelihood = log_likelihood
    scorer.release = release
    return scorer


def test_dtype_gates_match_the_declared_native_interfaces():
    with pytest.raises(ValueError, match="bfloat16"):
        EXT._require_dtype("galactica-1.3b", "float32")
    with pytest.raises(ValueError, match="float32"):
        EXT._require_dtype("rita-xl", "bfloat16")
    assert EXT._require_dtype("galactica-30b", "bfloat16") == "bfloat16"
    assert EXT.REQUIRED_DTYPE["rita-xl"] == "float32"
    assert EXT.G.CHECKPOINTS["galactica-125m"].name == "galactica-125m"
    from src.transfer.arms import STAGED_ARMS

    assert STAGED_ARMS["rita-xl"].path.name == "RITA_xl"


def test_lookup_replay_keeps_original_index_and_refuses_a_digest_mismatch(tmp_path):
    gym = tmp_path / "gym"
    _write_assay_csv(gym, "assay_a")
    _write_assay_csv(gym, "assay_b")
    digest_a = _digest_from_csv(gym, "assay_a", seed=EXT.VARIANT_SEED)
    digest_b = _digest_from_csv(gym, "assay_b", seed=EXT.VARIANT_SEED + 1)
    lookup = {
        "assays": [
            _lookup_row("assay_a", index=0, digest=digest_a),
            _lookup_row("assay_b", index=1, digest=digest_b),
        ]
    }
    catalog_rows = [
        ("assay_a", "MKT", "q00000", 0),
        ("assay_b", "MKT", "q00001", 1),
    ]
    wildtypes_path = _write_wildtypes(tmp_path / "wildtypes.json", catalog_rows)
    fasta_path = _write_wildtypes_fasta(tmp_path / "wildtypes.faa", catalog_rows)
    wildtypes = json.loads(wildtypes_path.read_text(encoding="utf-8"))
    frozen = EXT.freeze_lookup_cohort(
        lookup,
        proteingym_dir=gym,
        wildtypes=wildtypes,
        wildtypes_path=wildtypes_path,
        wildtypes_fasta_path=fasta_path,
    )
    assert [row["seed"] for row in frozen["assays"]] == [
        EXT.VARIANT_SEED,
        EXT.VARIANT_SEED + 1,
    ]
    assert frozen["assays"][0]["wildtype_sequence"] == "MKT"
    lookup["assays"][1]["mutant_digest"] = "0" * 64
    with pytest.raises(ValueError, match="mutant_digest"):
        EXT.freeze_lookup_cohort(
            lookup,
            proteingym_dir=gym,
            wildtypes=wildtypes,
            wildtypes_path=wildtypes_path,
            wildtypes_fasta_path=fasta_path,
        )


def test_context_exclusion_does_not_reset_the_seed_index(tmp_path):
    gym = tmp_path / "gym"
    _write_assay_csv(gym, "long", wildtype="MKTAAAAAA")
    _write_assay_csv(gym, "short", wildtype="MKT")
    digest0 = _digest_from_csv(gym, "long", seed=EXT.VARIANT_SEED)
    digest1 = _digest_from_csv(gym, "short", seed=EXT.VARIANT_SEED + 1)
    lookup = {
        "assays": [
            _lookup_row("long", index=0, digest=digest0),
            _lookup_row("short", index=1, digest=digest1),
        ]
    }
    catalog_rows = [
        ("long", "MKTAAAAAA", "q00000", 0),
        ("short", "MKT", "q00001", 1),
    ]
    wildtypes_path = _write_wildtypes(tmp_path / "wildtypes.json", catalog_rows)
    fasta_path = _write_wildtypes_fasta(tmp_path / "wildtypes.faa", catalog_rows)
    frozen = EXT.freeze_lookup_cohort(
        lookup,
        proteingym_dir=gym,
        wildtypes=json.loads(wildtypes_path.read_text(encoding="utf-8")),
        wildtypes_path=wildtypes_path,
        wildtypes_fasta_path=fasta_path,
    )
    scorer = _LengthScorer(
        {
            tuple(frozen["sequences"]["long"]): [9, 9],
            tuple(frozen["sequences"]["short"]): [3, 3],
        },
        context=4,
        name="galactica-1.3b",
    )
    scorer.token_lengths = lambda sequences: (
        [9, 9] if sequences == frozen["sequences"]["long"] else [3, 3]
    )
    cohort = EXT.attach_token_lengths(frozen, scorer)
    assert cohort["context_excluded_assays"] == ["long"]
    assert cohort["analysis_assays"] == ["short"]
    assert frozen["assays"][1]["seed"] == EXT.VARIANT_SEED + 1
    assert cohort["skipped"][0]["reason"] == EXT.CONTEXT_EXCLUSION_REASON


def test_galactica_author_ce_and_batch_match_on_variable_length_probes():
    scorer = _galactica_scorer(["MKT", AA20], batch_size=2)
    record = EXT.check_author_alignment(
        scorer, ["MKT", AA20], arm="galactica-1.3b", dtype="bfloat16"
    )
    assert record["author_ce_max_abs_nats_per_target"] <= EXT.FP32_PER_TARGET_ABS
    assert record["native_api_loss_present"] is True
    assert record["native_api_vs_reference_max_abs_nats_per_target"] <= EXT.FP32_PER_TARGET_ABS
    assert record["batch_vs_single_tolerance_used"] == EXT.LOWPREC_BATCH_VS_SINGLE
    EXT.check_refusals(_galactica_scorer(["MKT"], context=3, batch_size=1))


def test_rita_author_ce_equal_length_probes_and_mask_agreement():
    scorer = _rita_scorer()
    record = EXT.check_author_alignment(
        scorer, ["MKT", "MRT"], arm="rita-xl", dtype="float32"
    )
    assert record["author_ce_max_abs_nats_per_target"] <= EXT.FP32_PER_TARGET_ABS
    assert record["native_api_loss_present"] is True
    assert record["rita_no_mask_vs_all_ones_max_abs_logit"] <= EXT.FP32_PER_TARGET_ABS
    assert record["batch_vs_single_tolerance_used"] == EXT.FP32_PER_TARGET_ABS
    EXT.check_refusals(_rita_scorer(context=4))


def test_probe_writes_no_pass_file_when_author_ce_fails(tmp_path, monkeypatch):
    gym = tmp_path / "gym"
    _write_assay_csv(gym, "assay_a")
    digest = _digest_from_csv(gym, "assay_a", seed=EXT.VARIANT_SEED)
    lookup_path = tmp_path / "lookup.json"
    lookup_path.write_text(
        json.dumps({"assays": [_lookup_row("assay_a", index=0, digest=digest)]}),
        encoding="utf-8",
    )
    catalog_rows = [("assay_a", "MKT", "q00000", 0)]
    wildtypes_path = _write_wildtypes(tmp_path / "wildtypes.json", catalog_rows)
    fasta_path = _write_wildtypes_fasta(tmp_path / "wildtypes.faa", catalog_rows)
    scorer = _galactica_scorer(["MKT", AA20])

    def boom(*args, **kwargs):
        raise RuntimeError("author shifted CE disagrees")

    monkeypatch.setattr(EXT, "check_author_alignment", boom)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(RuntimeError, match="author shifted CE"):
        EXT.run_probe(
            arm="galactica-1.3b",
            lookup_path=lookup_path,
            proteingym_dir=gym,
            wildtypes_path=wildtypes_path,
            wildtypes_fasta_path=fasta_path,
            out=out,
            dtype="bfloat16",
            batch_size=2,
            device="cpu",
            scorer=scorer,
        )
    assert not (out / "interface_check.json").exists()


def _score_payload(
    arm: str,
    rhos: dict[str, float],
    *,
    skipped=None,
    digest="d0",
    dtype="bfloat16",
    csv_sha256="csv0",
    wildtypes_sha256="wt0",
    batch_size=16,
):
    return {
        "arm": arm,
        "assays": [
            {
                "assay": name,
                "wildtype_id": f"q{index:05d}",
                "mutant_digest": digest,
                "n_variants": 2,
                "csv_sha256": csv_sha256,
                "spearman": value,
            }
            for index, (name, value) in enumerate(rhos.items())
        ],
        "skipped": list(skipped or []),
        "settings": {
            "dtype": dtype,
            "scoring_stratum": STRATUM_N_TO_C,
            "seed": EXT.VARIANT_SEED,
            "variants": EXT.VARIANT_CAP,
            "batch_size": batch_size,
        },
        "input_fingerprints": {"wildtypes_sha256": wildtypes_sha256},
    }


def _probe_payload(
    arm: str,
    rhos: dict[str, float],
    *,
    skipped=None,
    digest="d0",
    dtype="bfloat16",
    lookup_sha256="lookup0",
    csv_sha256="csv0",
    wildtypes_sha256="wt0",
    batch_size=16,
):
    excluded = [row["assay"] for row in (skipped or [])]
    analysis = [name for name in rhos if name not in excluded]
    names = list(rhos) + excluded
    return {
        "arm": arm,
        "probe_passed": True,
        "settings": {
            "dtype": dtype,
            "scoring_stratum": STRATUM_N_TO_C,
            "variant_seed": EXT.VARIANT_SEED,
            "variant_cap": EXT.VARIANT_CAP,
            "batch_size": batch_size,
        },
        "fingerprints": {
            "lookup_sha256": lookup_sha256,
            "wildtypes_sha256": wildtypes_sha256,
            "wildtypes_fasta_sha256": "fa0",
            "csv_sha256": {name: csv_sha256 for name in names},
        },
        "cohort": {
            "declared_assays": len(rhos) + len(excluded),
            "declared_clusters": len(rhos),
            "analysis_assays": analysis,
            "analysis_clusters": len(analysis),
            "context": 64,
            "context_excluded_assays": excluded,
            "assays": [
                {
                    "assay": name,
                    "wildtype_id": f"q{index:05d}",
                    "wildtype_sequence": "MKT",
                    "mutant_digest": digest,
                    "n_variants": 2,
                    "cluster": str(index),
                    "seed": EXT.VARIANT_SEED + index,
                    "csv_sha256": csv_sha256,
                    "max_tokens": 8,
                }
                for index, name in enumerate(names)
            ],
        },
        "skipped": list(skipped or []),
    }


def _eight_rhos(base: float) -> dict[str, float]:
    return {f"assay_{index}": base + 0.01 * index for index in range(8)}


def _write_group_inputs(tmp_path: Path, *, rita: bool = False):
    rhos = _eight_rhos(0.2)
    lookup = {
        "assays": [
            {
                "assay": name,
                "cluster": str(index),
                "mutant_digest": "d0",
                "n_variants": 2,
                "wildtype_id": f"q{index:05d}",
                "spearman": {"lookup": 0.11, "blosum62": 0.04},
            }
            for index, name in enumerate(rhos)
        ]
    }
    lookup_path = tmp_path / "lookup.json"
    lookup_path.write_text(json.dumps(lookup), encoding="utf-8")
    lookup_sha256 = sha256_file(lookup_path)
    probes = {}
    scores = {}
    names = ("rita-xl",) if rita else EXT.GALACTICA_GROUP
    dtype = "float32" if rita else "bfloat16"
    for offset, name in enumerate(names):
        shifted = {assay: value + 0.02 * offset for assay, value in rhos.items()}
        probe = _probe_payload(
            name, shifted, dtype=dtype, lookup_sha256=lookup_sha256
        )
        score = _score_payload(name, shifted, dtype=dtype)
        probe_path = tmp_path / f"probe_{name}.json"
        score_path = tmp_path / f"score_{name}.json"
        probe_path.write_text(json.dumps(probe), encoding="utf-8")
        score_path.write_text(json.dumps(score), encoding="utf-8")
        probes[name] = probe_path
        scores[name] = score_path
    return lookup_path, probes, scores, rhos


def test_analyse_refuses_missing_named_input_and_partial_galactica(tmp_path):
    lookup_path, probes, scores, _ = _write_group_inputs(tmp_path)
    with pytest.raises(ValueError, match="disagree"):
        EXT.run_analyse(
            lookup_path=lookup_path,
            probes={k: probes[k] for k in list(probes)[:3]},
            scores=scores,
            out=tmp_path / "out",
        )
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        EXT.run_analyse(
            lookup_path=lookup_path,
            probes=probes,
            scores={**scores, "galactica-30b": missing},
            out=tmp_path / "out",
        )


def test_analyse_refuses_galactica_support_disagreement(tmp_path):
    lookup_path, probes, scores, rhos = _write_group_inputs(tmp_path)
    skip = {
        "assay": "assay_0",
        "context": 64,
        "max_tokens": 80,
        "reason": EXT.CONTEXT_EXCLUSION_REASON,
    }
    broken = json.loads(probes["galactica-30b"].read_text(encoding="utf-8"))
    broken["cohort"]["context_excluded_assays"] = ["assay_0"]
    broken["cohort"]["analysis_assays"] = [name for name in rhos if name != "assay_0"]
    broken["skipped"] = [skip]
    probes["galactica-30b"].write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="support"):
        EXT.run_analyse(
            lookup_path=lookup_path,
            probes=probes,
            scores=scores,
            out=tmp_path / "out",
        )


def test_analyse_family_means_and_paired_delta_reuse_scale_comparison(tmp_path):
    lookup_path, probes, scores, rhos = _write_group_inputs(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    payload = EXT.run_analyse(
        lookup_path=lookup_path, probes=probes, scores=scores, out=out
    )
    group = payload["groups"]["galactica"]
    assert group["pairs"]["reference_only"] == [["galactica-125m", "galactica-1.3b"]]
    assert group["pairs"]["qualified_ladder"] == [
        ["galactica-1.3b", "galactica-6.7b"],
        ["galactica-6.7b", "galactica-30b"],
    ]
    lookup = json.loads(lookup_path.read_text(encoding="utf-8"))
    units = {row["assay"]: str(row["cluster"]) for row in lookup["assays"]}
    lookup_raw = {row["assay"]: row["spearman"]["lookup"] for row in lookup["assays"]}
    expected = EXT.unit_mean_record(
        lookup_raw,
        units,
        resamples=EXT.BOOTSTRAP_RESAMPLES,
        seed=EXT.BOOTSTRAP_SEED + EXT.CHANNEL_OFFSETS["lookup_raw"],
    )
    assert group["lookup_raw_mean"]["point"] == pytest.approx(expected["point"])
    assert group["lookup_raw_mean"]["interval"] == expected["interval"]
    per_rung = {
        name: json.loads(scores[name].read_text(encoding="utf-8"))
        for name in EXT.GALACTICA_GROUP
    }
    raw = {
        name: {row["assay"]: row["spearman"] for row in payload_row["assays"]}
        for name, payload_row in per_rung.items()
    }
    expected_endpoint = EXT.endpoint(
        raw,
        units,
        rungs=EXT.GALACTICA_GROUP,
        pairs=EXT.GALACTICA_PAIRS,
        resamples=EXT.BOOTSTRAP_RESAMPLES,
        seed=EXT.BOOTSTRAP_SEED,
    )
    assert group["raw_spearman"]["adjacent_delta_rho"][
        "galactica-1.3b__galactica-6.7b"
    ]["point"] == pytest.approx(
        expected_endpoint["adjacent_delta_rho"]["galactica-1.3b__galactica-6.7b"]["point"]
    )
    assert "acquired" not in payload
    assert payload["no_acquired_or_retrieval_bounded_verdict"] is True
    assert payload["blosum62_is_not_a_dms_gate"] is True
    assert payload["no_parameter_count_causal_claim"] is True
    assert (out / "native_dms_comparison.json").is_file()


def test_rita_group_uses_the_offset_seed_and_no_pairs(tmp_path):
    lookup_path, probes, scores, _ = _write_group_inputs(tmp_path, rita=True)
    out = tmp_path / "out"
    out.mkdir()
    payload = EXT.run_analyse(
        lookup_path=lookup_path, probes=probes, scores=scores, out=out
    )
    group = payload["groups"]["rita-xl"]
    assert group["rungs"] == ["rita-xl"]
    assert group["pairs"]["all_computed"] == []
    assert payload["bootstrap"]["rita_group_offset"] == 1000
    lookup = json.loads(lookup_path.read_text(encoding="utf-8"))
    units = {row["assay"]: str(row["cluster"]) for row in lookup["assays"]}
    lookup_raw = {row["assay"]: row["spearman"]["lookup"] for row in lookup["assays"]}
    expected = EXT.unit_mean_record(
        lookup_raw,
        units,
        resamples=EXT.BOOTSTRAP_RESAMPLES,
        seed=EXT.BOOTSTRAP_SEED + 1000 + EXT.CHANNEL_OFFSETS["lookup_raw"],
    )
    assert group["lookup_raw_mean"]["point"] == pytest.approx(expected["point"])


def test_analyse_refuses_nan_spearman(tmp_path):
    lookup_path, probes, scores, rhos = _write_group_inputs(tmp_path)
    broken = json.loads(scores["galactica-1.3b"].read_text(encoding="utf-8"))
    broken["assays"][0]["spearman"] = float("nan")
    scores["galactica-1.3b"].write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        EXT.run_analyse(
            lookup_path=lookup_path,
            probes=probes,
            scores=scores,
            out=tmp_path / "out",
        )


def test_cli_analyse_requires_named_paths(tmp_path):
    with pytest.raises(ValueError, match="NAME=PATH|requires --probe"):
        EXT.main(["analyse", "--lookup", str(tmp_path / "lookup.json"), "--out", str(tmp_path)])


def test_cli_probe_requires_wildtypes(tmp_path):
    with pytest.raises(ValueError, match="--wildtypes"):
        EXT.main(
            [
                "probe",
                "--lookup",
                str(tmp_path / "lookup.json"),
                "--out",
                str(tmp_path),
                "--arm",
                "galactica-1.3b",
                "--dtype",
                "bfloat16",
            ]
        )
    with pytest.raises(ValueError, match="--wildtypes-fasta"):
        EXT.main(
            [
                "probe",
                "--lookup",
                str(tmp_path / "lookup.json"),
                "--out",
                str(tmp_path),
                "--arm",
                "galactica-1.3b",
                "--dtype",
                "bfloat16",
                "--wildtypes",
                str(tmp_path / "wildtypes.json"),
            ]
        )


def test_author_alignment_restores_grad_state_on_a_parameter_tied_model():
    enabled_before = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    try:
        scorer = _galactica_scorer(["MKT", AA20], batch_size=2)
        record = EXT.check_author_alignment(
            scorer, ["MKT", AA20], arm="galactica-1.3b", dtype="bfloat16"
        )
        assert record["native_api_loss_present"] is True
        assert torch.is_grad_enabled() is True
    finally:
        torch.set_grad_enabled(enabled_before)
    torch.set_grad_enabled(False)
    try:
        scorer = _galactica_scorer(["MKT"], batch_size=1)
        EXT.check_author_alignment(
            scorer, ["MKT"], arm="galactica-1.3b", dtype="bfloat16"
        )
        assert torch.is_grad_enabled() is False
    finally:
        torch.set_grad_enabled(enabled_before)


def test_missing_or_wrong_native_loss_fails_closed():
    scorer = _galactica_scorer(["MKT"], batch_size=1)
    scorer.loaded.model.omit_loss = True
    with pytest.raises(RuntimeError, match="returned no loss"):
        EXT.check_author_alignment(
            scorer, ["MKT"], arm="galactica-1.3b", dtype="bfloat16"
        )
    scorer = _galactica_scorer(["MKT"], batch_size=1)
    scorer.loaded.model.wrong_loss = True
    with pytest.raises(RuntimeError, match="native CausalLM loss disagrees"):
        EXT.check_author_alignment(
            scorer, ["MKT"], arm="galactica-1.3b", dtype="bfloat16"
        )


def test_freeze_refuses_same_length_wildtype_that_disagrees_with_fasta(tmp_path):
    gym = tmp_path / "gym"
    _write_assay_csv(gym, "assay_a", wildtype="MKT")
    digest = _digest_from_csv(gym, "assay_a", seed=EXT.VARIANT_SEED)
    lookup = {"assays": [_lookup_row("assay_a", index=0, digest=digest)]}
    catalog_rows = [("assay_a", "MKT", "q00000", 0)]
    wildtypes_path = _write_wildtypes(tmp_path / "wildtypes.json", catalog_rows)
    fasta_path = _write_wildtypes_fasta(tmp_path / "wildtypes.faa", catalog_rows)
    wildtypes = json.loads(wildtypes_path.read_text(encoding="utf-8"))
    EXT.freeze_lookup_cohort(
        lookup,
        proteingym_dir=gym,
        wildtypes=wildtypes,
        wildtypes_path=wildtypes_path,
        wildtypes_fasta_path=fasta_path,
    )
    _write_assay_csv(gym, "assay_a", wildtype="MKS")
    assay = load_assay("assay_a", n=1000, seed=EXT.VARIANT_SEED, directory=gym)
    assert EXT.mutant_digest(assay.mutants) == digest
    assert len(assay.sequences) == 2
    assert len(wildtype_of("assay_a", gym)) == 3
    assert wildtype_of("assay_a", gym) == "MKS"
    with pytest.raises(ValueError, match="disagrees with FASTA"):
        EXT.freeze_lookup_cohort(
            lookup,
            proteingym_dir=gym,
            wildtypes=wildtypes,
            wildtypes_path=wildtypes_path,
            wildtypes_fasta_path=fasta_path,
        )


def test_freeze_refuses_missing_or_duplicate_fasta_ids(tmp_path):
    gym = tmp_path / "gym"
    _write_assay_csv(gym, "assay_a", wildtype="MKT")
    digest = _digest_from_csv(gym, "assay_a", seed=EXT.VARIANT_SEED)
    lookup = {"assays": [_lookup_row("assay_a", index=0, digest=digest)]}
    catalog_rows = [("assay_a", "MKT", "q00000", 0)]
    wildtypes_path = _write_wildtypes(tmp_path / "wildtypes.json", catalog_rows)
    wildtypes = json.loads(wildtypes_path.read_text(encoding="utf-8"))
    missing = tmp_path / "missing.faa"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        EXT.freeze_lookup_cohort(
            lookup,
            proteingym_dir=gym,
            wildtypes=wildtypes,
            wildtypes_path=wildtypes_path,
            wildtypes_fasta_path=missing,
        )
    other = tmp_path / "other.faa"
    other.write_text(">q00001\nMKT\n", encoding="utf-8")
    with pytest.raises(ValueError, match="has no id"):
        EXT.freeze_lookup_cohort(
            lookup,
            proteingym_dir=gym,
            wildtypes=wildtypes,
            wildtypes_path=wildtypes_path,
            wildtypes_fasta_path=other,
        )
    duplicate = tmp_path / "dup.faa"
    duplicate.write_text(">q00000\nMKT\n>q00000\nMKT\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate FASTA id"):
        EXT.freeze_lookup_cohort(
            lookup,
            proteingym_dir=gym,
            wildtypes=wildtypes,
            wildtypes_path=wildtypes_path,
            wildtypes_fasta_path=duplicate,
        )


def test_analyse_refuses_replaced_lookup_values_with_the_same_row_ids(tmp_path):
    lookup_path, probes, scores, _ = _write_group_inputs(tmp_path)
    lookup = json.loads(lookup_path.read_text(encoding="utf-8"))
    lookup["assays"][0]["spearman"]["lookup"] = 0.99
    lookup_path.write_text(json.dumps(lookup), encoding="utf-8")
    with pytest.raises(ValueError, match="lookup_sha256"):
        EXT.run_analyse(
            lookup_path=lookup_path,
            probes=probes,
            scores=scores,
            out=tmp_path / "out",
        )


def test_analyse_refuses_csv_hash_change_when_mutation_labels_match(tmp_path):
    lookup_path, probes, scores, _ = _write_group_inputs(tmp_path)
    broken = json.loads(scores["galactica-1.3b"].read_text(encoding="utf-8"))
    broken["assays"][0]["csv_sha256"] = "different-csv"
    scores["galactica-1.3b"].write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="csv_sha256"):
        EXT.run_analyse(
            lookup_path=lookup_path,
            probes=probes,
            scores=scores,
            out=tmp_path / "out",
        )


def test_analyse_refuses_missing_wildtype_id_and_variant_count(tmp_path):
    lookup_path, probes, scores, _ = _write_group_inputs(tmp_path)
    broken = json.loads(scores["galactica-1.3b"].read_text(encoding="utf-8"))
    del broken["assays"][0]["wildtype_id"]
    del broken["assays"][0]["n_variants"]
    scores["galactica-1.3b"].write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="missing wildtype_id"):
        EXT.run_analyse(
            lookup_path=lookup_path,
            probes=probes,
            scores=scores,
            out=tmp_path / "out",
        )


def test_analyse_refuses_galactica_wildtype_sequence_disagreement(tmp_path):
    lookup_path, probes, scores, _ = _write_group_inputs(tmp_path)
    broken = json.loads(probes["galactica-30b"].read_text(encoding="utf-8"))
    broken["cohort"]["assays"][0]["wildtype_sequence"] = "MKS"
    probes["galactica-30b"].write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="wildtype_sequence"):
        EXT.run_analyse(
            lookup_path=lookup_path,
            probes=probes,
            scores=scores,
            out=tmp_path / "out",
        )


def test_analyse_refuses_galactica_fasta_fingerprint_disagreement(tmp_path):
    lookup_path, probes, scores, _ = _write_group_inputs(tmp_path)
    broken = json.loads(probes["galactica-30b"].read_text(encoding="utf-8"))
    broken["fingerprints"]["wildtypes_fasta_sha256"] = "other-fa"
    probes["galactica-30b"].write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="wildtypes_fasta_sha256"):
        EXT.run_analyse(
            lookup_path=lookup_path,
            probes=probes,
            scores=scores,
            out=tmp_path / "out",
        )


def test_analyse_refuses_sampling_config_disagreement(tmp_path):
    lookup_path, probes, scores, _ = _write_group_inputs(tmp_path)
    broken = json.loads(scores["galactica-1.3b"].read_text(encoding="utf-8"))
    broken["settings"]["seed"] = EXT.VARIANT_SEED
    broken["settings"]["variants"] = 8
    scores["galactica-1.3b"].write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="variant_cap|variants"):
        EXT.run_analyse(
            lookup_path=lookup_path,
            probes=probes,
            scores=scores,
            out=tmp_path / "out",
        )


def test_longest_eligible_batch_matches_max_tokens_and_omits_loglik(tmp_path):
    gym = tmp_path / "gym"
    _write_assay_csv(gym, "assay_a")
    digest = _digest_from_csv(gym, "assay_a", seed=EXT.VARIANT_SEED)
    lookup_path = tmp_path / "lookup.json"
    lookup_path.write_text(
        json.dumps({"assays": [_lookup_row("assay_a", index=0, digest=digest)]}),
        encoding="utf-8",
    )
    catalog_rows = [("assay_a", "MKT", "q00000", 0)]
    wildtypes_path = _write_wildtypes(tmp_path / "wildtypes.json", catalog_rows)
    fasta_path = _write_wildtypes_fasta(tmp_path / "wildtypes.faa", catalog_rows)
    scorer = _galactica_scorer(["MKT", AA20], context=64, batch_size=2)
    out = tmp_path / "out"
    out.mkdir()
    payload = EXT.run_probe(
        arm="galactica-1.3b",
        lookup_path=lookup_path,
        proteingym_dir=gym,
        wildtypes_path=wildtypes_path,
        wildtypes_fasta_path=fasta_path,
        out=out,
        dtype="bfloat16",
        batch_size=2,
        device="cpu",
        scorer=scorer,
    )
    longest = payload["longest_eligible_batch"]
    assay_row = payload["cohort"]["assays"][0]
    assert longest["kind"] == "synthetic_longest_eligible_shape_not_dms_score"
    assert longest["not_a_dms_variant"] is True
    assert longest["synthetic_sequence"] == AA20[:3]
    assert longest["n_residues"] == 3
    assert longest["token_length"] == assay_row["max_tokens"]
    assert longest["expected_max_tokens"] == assay_row["max_tokens"]
    assert longest["batch_size"] == 2
    assert longest["all_finite"] is True
    assert longest["log_likelihood_retained"] is False
    assert longest["cuda_memory_measured"] is False
    assert longest["cuda_peak_allocated_bytes"] is None
    assert longest["cuda_peak_reserved_bytes"] is None
    assert "log_likelihood" not in longest
    saved = json.loads((out / "interface_check.json").read_text(encoding="utf-8"))
    assert "log_likelihood" not in saved["longest_eligible_batch"]
    assert saved["no_new_model_fitness_scores"] is True


def test_probe_refuses_when_no_assay_is_eligible(tmp_path):
    gym = tmp_path / "gym"
    wildtype = "MK" + ("T" * 78)
    _write_assay_csv(gym, "long", wildtype=wildtype)
    digest = _digest_from_csv(gym, "long", seed=EXT.VARIANT_SEED)
    lookup_path = tmp_path / "lookup.json"
    lookup_path.write_text(
        json.dumps({"assays": [_lookup_row("long", index=0, digest=digest)]}),
        encoding="utf-8",
    )
    catalog_rows = [("long", wildtype, "q00000", 0)]
    wildtypes_path = _write_wildtypes(tmp_path / "wildtypes.json", catalog_rows)
    fasta_path = _write_wildtypes_fasta(tmp_path / "wildtypes.faa", catalog_rows)
    scorer = _galactica_scorer(["MKT", AA20], context=32, batch_size=2)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(RuntimeError, match="no eligible assay"):
        EXT.run_probe(
            arm="galactica-1.3b",
            lookup_path=lookup_path,
            proteingym_dir=gym,
            wildtypes_path=wildtypes_path,
            wildtypes_fasta_path=fasta_path,
            out=out,
            dtype="bfloat16",
            batch_size=2,
            device="cpu",
            scorer=scorer,
        )
    assert not (out / "interface_check.json").exists()
    with pytest.raises(RuntimeError, match="no eligible assay"):
        EXT.check_longest_eligible_batch(
            scorer,
            {
                "analysis_assays": [],
                "assays": [],
                "context": 32,
            },
            arm="galactica-1.3b",
            batch_size=16,
        )
