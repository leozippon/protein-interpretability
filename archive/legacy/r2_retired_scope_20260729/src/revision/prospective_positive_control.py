"""Prospective unexposed long-range positive control for P0-7.

The benchmark trains small causal language models and TopK CLTs on a property
defined by equality of two distant amino acids.  Neither anchor alone reveals
the property, and no property, query, family, or endpoint-label token is given
to the model.  A fixed adapter supplies a known path and direction after the
second anchor.  This validates only the synthetic intervention pipeline.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

from src.training.clt_trainer import CLTForTraining

from .causal_positive_control import FAMILY_ALPHABETS, MOTIFS, POSITIONS
from .io import sha256_file, write_json, write_jsonl
from .statistics import benjamini_hochberg, mean_interval, tost_paired


SPEC_SCHEMA = "r2_p0_7_prospective_positive_control_spec_v2"
FREEZE_SCHEMA = "r2_p0_7_prospective_positive_control_freeze_v1"
RESULT_SCHEMA = "r2_p0_7_prospective_positive_control_result_v2"
MANIFEST_SCHEMA = "r2_p0_7_prospective_positive_control_manifest_v1"
CLAIM_SCOPE = "synthetic_pipeline_sensitivity_only_no_pretrained_causal_inference"
AA = "ACDEFGHIKLMNPQRSTVWY"
BODY_AA = tuple(residue for residue in AA if residue not in {"W", "Y"})
SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>")
TOKENS = SPECIAL_TOKENS + tuple(AA)
TOKEN_TO_ID = {token: index for index, token in enumerate(TOKENS)}
HEX = frozenset("0123456789abcdef")
SPLITS = ("train", "discovery", "assessment")
PROFILE_FIELDS = (
    "firing_frequency",
    "mean_activation",
    "decoder_norm",
    "direct_logit_effect_norm",
    "received_attention_mass",
    "reconstruction_contribution",
)
CALIPER_FIELDS = {
    "firing_frequency_abs",
    "mean_activation_abs_log_ratio",
    "decoder_norm_abs_log_ratio",
    "direct_logit_effect_norm_abs_log_ratio",
    "received_attention_mass_abs",
    "reconstruction_contribution_abs_log_ratio",
    "standardized_distance_max",
}


def _derived_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31)


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        observed = set(value) if isinstance(value, dict) else set()
        raise ValueError(
            f"{label} fields differ: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or not set(value) <= HEX:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(),
        parse_constant=lambda constant: (_ for _ in ()).throw(
            ValueError(f"{path} contains non-finite constant {constant}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _positive(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return number


def _probability(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 < number < 1.0:
        raise ValueError(f"{label} must lie strictly between zero and one")
    return number


def validate_prospective_spec(source: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete write-once v2 benchmark specification."""

    spec = _require_keys(
        dict(source),
        {
            "schema_version",
            "benchmark_id",
            "claim_scope",
            "cohort",
            "model",
            "training",
            "discovery",
            "intervention",
            "matching",
            "gates",
        },
        "prospective P0-7 spec",
    )
    if spec["schema_version"] != SPEC_SCHEMA:
        raise ValueError("unsupported prospective P0-7 spec schema")
    if not isinstance(spec["benchmark_id"], str) or not spec["benchmark_id"].strip():
        raise ValueError("benchmark_id must be non-empty")
    if spec["claim_scope"] != CLAIM_SCOPE:
        raise ValueError("prospective P0-7 claim_scope exceeds the synthetic boundary")

    cohort = _require_keys(
        spec["cohort"],
        {
            "split_seeds",
            "per_split",
            "sequence_lengths",
            "left_anchor_position",
            "right_anchor_position",
            "negative_endpoint_residue",
            "positive_endpoint_residue",
        },
        "cohort",
    )
    split_seeds = _require_keys(cohort["split_seeds"], set(SPLITS), "split_seeds")
    if any(type(value) is not int or value < 0 for value in split_seeds.values()):
        raise ValueError("split seeds must be non-negative integers")
    if len(set(split_seeds.values())) != len(SPLITS):
        raise ValueError("train/discovery/assessment seeds must be distinct")
    if type(cohort["per_split"]) is not int or cohort["per_split"] < 24:
        raise ValueError("per_split must be an integer of at least 24")
    if cohort["per_split"] % 12:
        raise ValueError("per_split must be divisible by 12 for marginal balance")
    lengths = cohort["sequence_lengths"]
    if (
        not isinstance(lengths, list)
        or len(lengths) < 2
        or len(set(lengths)) != len(lengths)
        or any(type(length) is not int or length < 40 for length in lengths)
    ):
        raise ValueError("sequence_lengths require at least two distinct integers >= 40")
    left = cohort["left_anchor_position"]
    right = cohort["right_anchor_position"]
    if (
        type(left) is not int
        or type(right) is not int
        or left < 4
        or right - left < 16
        or right > min(lengths) - 8
    ):
        raise ValueError("anchors require a frozen separation >=16 and suffix >=7")
    if (
        cohort["negative_endpoint_residue"] != "W"
        or cohort["positive_endpoint_residue"] != "Y"
    ):
        raise ValueError("v2 endpoint residues are frozen as W/Y")

    model = _require_keys(
        spec["model"],
        {
            "seeds",
            "d_content",
            "d_clt",
            "k",
            "nuisance_control_paths",
            "carrier_gain",
            "adapter_gain",
            "readout_gain",
        },
        "model",
    )
    seeds = model["seeds"]
    if (
        not isinstance(seeds, list)
        or len(seeds) < 3
        or len(set(seeds)) != len(seeds)
        or any(type(seed) is not int or seed < 0 for seed in seeds)
    ):
        raise ValueError("model seeds require at least three unique non-negative integers")
    for field in ("d_content", "d_clt", "k"):
        if type(model[field]) is not int or model[field] < 1:
            raise ValueError(f"model {field} must be a positive integer")
    if model["d_content"] < 4 or not 1 <= model["k"] < model["d_clt"]:
        raise ValueError("invalid d_content/d_clt/k geometry")
    if model["nuisance_control_paths"] != 2:
        raise ValueError("v2 freezes exactly two orthogonal nuisance control paths")
    for field in ("carrier_gain", "adapter_gain", "readout_gain"):
        _positive(model[field], f"model {field}")

    training = _require_keys(
        spec["training"],
        {
            "lm_steps",
            "clt_steps",
            "learning_rate",
            "clt_learning_rate",
            "device",
        },
        "training",
    )
    for field in ("lm_steps", "clt_steps"):
        if type(training[field]) is not int or training[field] < 1:
            raise ValueError(f"training {field} must be a positive integer")
    _positive(training["learning_rate"], "learning_rate")
    _positive(training["clt_learning_rate"], "clt_learning_rate")
    if training["device"] not in {"cpu", "cuda"}:
        raise ValueError("training device must be cpu or cuda")

    discovery = _require_keys(
        spec["discovery"],
        {"selection_alpha_bh", "ground_truth_direction_cosine_min"},
        "discovery",
    )
    _probability(discovery["selection_alpha_bh"], "selection_alpha_bh")
    cosine_min = float(discovery["ground_truth_direction_cosine_min"])
    if not 0.0 < cosine_min <= 1.0:
        raise ValueError("ground_truth_direction_cosine_min must lie in (0, 1]")

    intervention = _require_keys(
        spec["intervention"],
        {
            "target_layer",
            "target_site",
            "doses",
            "negative_equivalence_margin_log_odds",
        },
        "intervention",
    )
    if intervention["target_layer"] != 1 or intervention["target_site"] != (
        "layer_1_mlp_output_at_endpoint_predictor"
    ):
        raise ValueError("v2 known intervention path is frozen at layer 1/predictor")
    doses = intervention["doses"]
    if (
        not isinstance(doses, list)
        or len(doses) < 2
        or sorted(doses) != doses
        or len(set(doses)) != len(doses)
    ):
        raise ValueError("intervention doses must be distinct, increasing and prespecified")
    for dose in doses:
        _positive(dose, "intervention dose")
    _positive(
        intervention["negative_equivalence_margin_log_odds"],
        "negative equivalence margin",
    )

    matching = _require_keys(
        spec["matching"],
        {"control_count", "control_abs_ground_truth_cosine_max", "hard_calipers"},
        "matching",
    )
    if type(matching["control_count"]) is not int or matching["control_count"] < 1:
        raise ValueError("control_count must be a positive integer")
    max_cosine = float(matching["control_abs_ground_truth_cosine_max"])
    if not 0.0 <= max_cosine < cosine_min:
        raise ValueError("control cosine bound must be below the recovery threshold")
    calipers = _require_keys(
        matching["hard_calipers"], CALIPER_FIELDS, "matching hard_calipers"
    )
    for field, value in calipers.items():
        _positive(value, f"hard caliper {field}")

    gates = _require_keys(
        spec["gates"],
        {
            "sensitivity_min",
            "specificity_min",
            "false_discovery_rate_max",
            "endpoint_accuracy_min",
            "clt_fvu_max",
            "activation_alignment_spearman_min",
            "effect_recovery_ratio",
        },
        "gates",
    )
    for field in (
        "sensitivity_min",
        "specificity_min",
        "endpoint_accuracy_min",
        "activation_alignment_spearman_min",
    ):
        _probability(gates[field], f"gate {field}")
    fdr_max = float(gates["false_discovery_rate_max"])
    if not 0.0 <= fdr_max < 1.0:
        raise ValueError("false_discovery_rate_max must lie in [0, 1)")
    fvu_max = float(gates["clt_fvu_max"])
    if not 0.0 < fvu_max < 1.0:
        raise ValueError("clt_fvu_max must lie in (0, 1)")
    recovery = gates["effect_recovery_ratio"]
    if (
        not isinstance(recovery, list)
        or len(recovery) != 2
        or not 0.0 < float(recovery[0]) <= 1.0 <= float(recovery[1])
    ):
        raise ValueError("effect_recovery_ratio must be a positive interval around one")
    return spec


@dataclass(frozen=True)
class ProspectiveRecord:
    record_id: str
    source: str
    split: str
    sequence: str
    family: str
    motif: str
    motif_position: str
    n_terminal: str
    length: int
    left_anchor_position: int
    right_anchor_position: int
    left_anchor_residue: str
    right_anchor_residue: str
    nuisance1_left_residue: str
    nuisance1_right_residue: str
    nuisance1_equal: bool
    nuisance2_left_residue: str
    nuisance2_right_residue: str
    nuisance2_equal: bool
    long_range_equal: bool
    endpoint_residue: str
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _balanced(values: Sequence[Any], count: int, rng: np.random.Generator) -> list[Any]:
    if count % len(values):
        raise ValueError("balanced factor count is not divisible by its levels")
    result = list(values) * (count // len(values))
    order = rng.permutation(count)
    return [result[index] for index in order]


def generate_prospective_cohort(spec: Mapping[str, Any]) -> list[ProspectiveRecord]:
    """Generate immutable balanced splits without a property-label input token."""

    frozen = validate_prospective_spec(spec)
    cohort = frozen["cohort"]
    per_split = cohort["per_split"]
    left_position = cohort["left_anchor_position"]
    right_position = cohort["right_anchor_position"]
    pair_states = (("A", "A"), ("A", "C"), ("C", "A"), ("C", "C"))
    records: list[ProspectiveRecord] = []
    seen_sequences: set[str] = set()
    for split in SPLITS:
        rng = np.random.default_rng(cohort["split_seeds"][split])
        pairs = _balanced(pair_states, per_split, rng)
        nuisance1_pairs = _balanced(pair_states, per_split, rng)
        nuisance2_pairs = _balanced(pair_states, per_split, rng)
        families = _balanced(tuple(FAMILY_ALPHABETS), per_split, rng)
        motifs = _balanced(MOTIFS, per_split, rng)
        motif_positions = _balanced(POSITIONS, per_split, rng)
        n_terminals = _balanced(("A", "M"), per_split, rng)
        lengths = _balanced(tuple(cohort["sequence_lengths"]), per_split, rng)
        for index in range(per_split):
            family = families[index]
            length = int(lengths[index])
            alphabet = np.asarray(
                [residue for residue in FAMILY_ALPHABETS[family] if residue in BODY_AA]
            )
            sequence = rng.choice(alphabet, size=length).tolist()
            sequence[0] = n_terminals[index]
            motif = motifs[index]
            motif_position = motif_positions[index]
            motif_start = {
                "early": 4,
                "middle": length // 2,
                "late": length - len(motif) - 4,
            }[motif_position]
            occupied = {
                left_position,
                left_position + 2,
                left_position + 4,
                right_position,
                right_position + 2,
                right_position + 4,
                length - 1,
            }
            while any(
                position in occupied
                for position in range(motif_start, motif_start + len(motif))
            ):
                motif_start += len(motif) + 1
            if motif_start + len(motif) >= length - 1:
                raise RuntimeError("frozen motif placement collides with a causal site")
            sequence[motif_start : motif_start + len(motif)] = motif
            left_residue, right_residue = pairs[index]
            sequence[left_position] = left_residue
            sequence[right_position] = right_residue
            nuisance1_left, nuisance1_right = nuisance1_pairs[index]
            nuisance2_left, nuisance2_right = nuisance2_pairs[index]
            sequence[left_position + 2] = nuisance1_left
            sequence[right_position + 2] = nuisance1_right
            sequence[left_position + 4] = nuisance2_left
            sequence[right_position + 4] = nuisance2_right
            long_range = left_residue == right_residue
            endpoint = (
                cohort["positive_endpoint_residue"]
                if long_range
                else cohort["negative_endpoint_residue"]
            )
            sequence[-1] = endpoint
            value = "".join(sequence)
            if value in seen_sequences:
                raise RuntimeError("prospective grammar produced a duplicate sequence")
            seen_sequences.add(value)
            record_id = f"p07-v2-{split}-{index:05d}"
            records.append(
                ProspectiveRecord(
                    record_id=record_id,
                    source="prospective_unexposed_long_range_grammar_v2",
                    split=split,
                    sequence=value,
                    family=family,
                    motif=motif,
                    motif_position=motif_position,
                    n_terminal=n_terminals[index],
                    length=length,
                    left_anchor_position=left_position,
                    right_anchor_position=right_position,
                    left_anchor_residue=left_residue,
                    right_anchor_residue=right_residue,
                    nuisance1_left_residue=nuisance1_left,
                    nuisance1_right_residue=nuisance1_right,
                    nuisance1_equal=nuisance1_left == nuisance1_right,
                    nuisance2_left_residue=nuisance2_left,
                    nuisance2_right_residue=nuisance2_right,
                    nuisance2_equal=nuisance2_left == nuisance2_right,
                    long_range_equal=long_range,
                    endpoint_residue=endpoint,
                    sha256=hashlib.sha256(value.encode("utf-8")).hexdigest(),
                )
            )
    return records


@dataclass
class EncodedProspectiveCohort:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    predictor_positions: torch.Tensor
    records: list[ProspectiveRecord]

    def to(self, device: torch.device) -> "EncodedProspectiveCohort":
        return EncodedProspectiveCohort(
            self.input_ids.to(device),
            self.attention_mask.to(device),
            self.predictor_positions.to(device),
            self.records,
        )


def encode_prospective_cohort(
    records: Sequence[ProspectiveRecord],
) -> EncodedProspectiveCohort:
    if not records:
        raise ValueError("prospective records must be non-empty")
    encoded = [
        [
            TOKEN_TO_ID["<bos>"],
            *(TOKEN_TO_ID[residue] for residue in record.sequence),
            TOKEN_TO_ID["<eos>"],
        ]
        for record in records
    ]
    width = max(map(len, encoded))
    input_ids = torch.full(
        (len(records), width), TOKEN_TO_ID["<pad>"], dtype=torch.long
    )
    attention_mask = torch.zeros((len(records), width), dtype=torch.bool)
    predictor_positions = torch.empty(len(records), dtype=torch.long)
    for row, (record, token_ids) in enumerate(zip(records, encoded)):
        input_ids[row, : len(token_ids)] = torch.tensor(token_ids)
        attention_mask[row, : len(token_ids)] = True
        predictor_positions[row] = record.length - 1
        if token_ids[predictor_positions[row] + 1] != TOKEN_TO_ID[record.endpoint_residue]:
            raise RuntimeError("endpoint predictor/target alignment failed")
    return EncodedProspectiveCohort(
        input_ids, attention_mask, predictor_positions, list(records)
    )


class _CausalAttentionBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, 1, batch_first=True)
        self.projection = nn.Linear(width, width)

    def forward(
        self, values: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        length = values.shape[1]
        causal = torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=values.device),
            diagonal=1,
        )
        normalized = self.norm(values)
        attended, weights = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal,
            key_padding_mask=~attention_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        return 0.35 * torch.tanh(self.projection(attended)), weights


class ProspectiveLongRangeLM(nn.Module):
    """Causal LM with a fixed adapter for an unexposed anchor equality."""

    def __init__(self, model_seed: int, spec: Mapping[str, Any]) -> None:
        super().__init__()
        frozen = validate_prospective_spec(spec)
        model = frozen["model"]
        cohort = frozen["cohort"]
        self.model_seed = int(model_seed)
        self.d_content = int(model["d_content"])
        self.nuisance_paths = int(model["nuisance_control_paths"])
        self.d_model = self.d_content + self.nuisance_paths + 1
        self.left_token_position = int(cohort["left_anchor_position"]) + 1
        self.right_token_position = int(cohort["right_anchor_position"]) + 1
        self.carrier_gain = float(model["carrier_gain"])
        self.adapter_gain = float(model["adapter_gain"])
        self.readout_gain = float(model["readout_gain"])
        max_length = max(cohort["sequence_lengths"]) + 2
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(_derived_seed("prospective-lm", model_seed))
            self.embedding = nn.Embedding(
                len(TOKENS), self.d_content, padding_idx=TOKEN_TO_ID["<pad>"]
            )
            self.position = nn.Embedding(max_length, self.d_content)
            self.blocks = nn.ModuleList(
                [_CausalAttentionBlock(self.d_content) for _ in range(2)]
            )
            self.content_head = nn.Linear(self.d_content, len(TOKENS))
        rng = np.random.default_rng(_derived_seed("prospective-rotation", model_seed))
        rotation, _ = np.linalg.qr(rng.normal(size=(self.d_model, self.d_model)))
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1
        self.register_buffer("rotation", torch.tensor(rotation, dtype=torch.float32))
        self.register_buffer(
            "planted_direction", torch.tensor(rotation[:, -1], dtype=torch.float32)
        )
        self.register_buffer(
            "nuisance_directions",
            torch.tensor(rotation[:, -3:-1], dtype=torch.float32),
        )

    def _rotate(self, values: torch.Tensor) -> torch.Tensor:
        return values @ self.rotation.T

    def _unrotate(self, values: torch.Tensor) -> torch.Tensor:
        return values @ self.rotation

    @staticmethod
    def _patch(
        values: torch.Tensor,
        positions: torch.Tensor,
        direction: torch.Tensor,
        dose: float,
    ) -> torch.Tensor:
        changed = values.clone()
        rows = torch.arange(values.shape[0], device=values.device)
        changed[rows, positions] += float(dose) * direction
        return changed

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        intervention_layer: int | None = None,
        intervention_positions: torch.Tensor | None = None,
        intervention_direction: torch.Tensor | None = None,
        intervention_dose: float = 0.0,
        capture: bool = False,
    ) -> tuple[
        torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]
    ]:
        batch, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device)
        content = self.embedding(input_ids) + self.position(positions)[None]
        equality = torch.where(
            input_ids[:, self.left_token_position]
            == input_ids[:, self.right_token_position],
            1.0,
            0.0,
        ).to(content.dtype)
        nuisance1 = torch.where(
            input_ids[:, self.left_token_position + 2]
            == input_ids[:, self.right_token_position + 2],
            1.0,
            0.0,
        ).to(content.dtype)
        nuisance2 = torch.where(
            input_ids[:, self.left_token_position + 4]
            == input_ids[:, self.right_token_position + 4],
            1.0,
            0.0,
        ).to(content.dtype)
        available = positions[None] >= self.right_token_position + 4
        semantic = (
            torch.stack((nuisance1, nuisance2, equality), dim=1)[:, None, :]
            * available[..., None]
        )
        zeros = torch.zeros(
            (batch, length, self.nuisance_paths + 1), device=content.device
        )
        resid_pre: list[torch.Tensor] = []
        mlp_out: list[torch.Tensor] = []
        attention_rows: list[torch.Tensor] = []

        pre0 = self._rotate(torch.cat((content, zeros), dim=-1))
        content_delta0, attention0 = self.blocks[0](content, attention_mask)
        delta0 = self._rotate(torch.cat((content_delta0, zeros), dim=-1))
        observed0 = pre0 + delta0
        if intervention_layer == 0:
            if intervention_positions is None or intervention_direction is None:
                raise ValueError("an intervention requires positions and a direction")
            observed0 = self._patch(
                observed0,
                intervention_positions,
                intervention_direction,
                intervention_dose,
            )
        content = self._unrotate(observed0)[..., : self.d_content]
        resid_pre.append(pre0)
        mlp_out.append(delta0)
        attention_rows.append(attention0)

        carrier = self.carrier_gain * semantic
        pre1 = self._rotate(torch.cat((content, carrier), dim=-1))
        content_delta1, attention1 = self.blocks[1](content, attention_mask)
        adapter = self.adapter_gain * semantic
        delta1 = self._rotate(torch.cat((content_delta1, adapter), dim=-1))
        observed1 = pre1 + delta1
        if intervention_layer == 1:
            if intervention_positions is None or intervention_direction is None:
                raise ValueError("an intervention requires positions and a direction")
            observed1 = self._patch(
                observed1,
                intervention_positions,
                intervention_direction,
                intervention_dose,
            )
        resid_pre.append(pre1)
        mlp_out.append(delta1)
        attention_rows.append(attention1)

        final_base = self._unrotate(observed1)
        logits = self.content_head(final_base[..., : self.d_content])
        valid_lengths = attention_mask.sum(dim=1)
        predictor_positions = valid_lengths - 3
        rows = torch.arange(batch, device=input_ids.device)
        planted_score = observed1[rows, predictor_positions] @ self.planted_direction
        planted_score = planted_score - 0.5 * (
            self.carrier_gain + self.adapter_gain
        )
        logits[rows, predictor_positions, TOKEN_TO_ID["Y"]] = (
            self.readout_gain * planted_score
        )
        logits[rows, predictor_positions, TOKEN_TO_ID["W"]] = (
            -self.readout_gain * planted_score
        )
        nuisance_scores = (
            observed1[rows, predictor_positions] @ self.nuisance_directions
        )
        nuisance_scores = nuisance_scores - 0.5 * (
            self.carrier_gain + self.adapter_gain
        )
        for index, (positive, negative) in enumerate((("A", "C"), ("D", "E"))):
            logits[rows, predictor_positions, TOKEN_TO_ID[positive]] += (
                self.readout_gain * nuisance_scores[:, index]
            )
            logits[rows, predictor_positions, TOKEN_TO_ID[negative]] -= (
                self.readout_gain * nuisance_scores[:, index]
            )
        return (
            logits,
            resid_pre if capture else [],
            mlp_out if capture else [],
            attention_rows if capture else [],
        )


def _state_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _split_records(
    records: Sequence[ProspectiveRecord], spec: Mapping[str, Any]
) -> dict[str, list[ProspectiveRecord]]:
    frozen = validate_prospective_spec(spec)
    groups = {split: [row for row in records if row.split == split] for split in SPLITS}
    expected = frozen["cohort"]["per_split"]
    if any(len(rows) != expected for rows in groups.values()):
        raise ValueError("frozen cohort has an incomplete train/discovery/assessment split")
    all_ids: set[str] = set()
    all_hashes: set[str] = set()
    left = frozen["cohort"]["left_anchor_position"]
    right = frozen["cohort"]["right_anchor_position"]
    for split, rows in groups.items():
        for row in rows:
            digest = hashlib.sha256(row.sequence.encode("utf-8")).hexdigest()
            if digest != row.sha256 or row.record_id in all_ids or digest in all_hashes:
                raise ValueError("prospective cohort identity/hash integrity failed")
            if (
                row.left_anchor_position != left
                or row.right_anchor_position != right
                or row.sequence[left] != row.left_anchor_residue
                or row.sequence[right] != row.right_anchor_residue
                or row.sequence[left + 2] != row.nuisance1_left_residue
                or row.sequence[right + 2] != row.nuisance1_right_residue
                or row.nuisance1_equal
                != (row.nuisance1_left_residue == row.nuisance1_right_residue)
                or row.sequence[left + 4] != row.nuisance2_left_residue
                or row.sequence[right + 4] != row.nuisance2_right_residue
                or row.nuisance2_equal
                != (row.nuisance2_left_residue == row.nuisance2_right_residue)
                or row.long_range_equal
                != (row.left_anchor_residue == row.right_anchor_residue)
                or row.endpoint_residue != ("Y" if row.long_range_equal else "W")
                or row.sequence[-1] != row.endpoint_residue
            ):
                raise ValueError("prospective long-range ground truth is inconsistent")
            all_ids.add(row.record_id)
            all_hashes.add(digest)
        labels = np.asarray([row.long_range_equal for row in rows], dtype=np.int64)
        left_a = np.asarray([row.left_anchor_residue == "A" for row in rows])
        right_a = np.asarray([row.right_anchor_residue == "A" for row in rows])
        if (
            labels.mean() != 0.5
            or left_a.mean() != 0.5
            or right_a.mean() != 0.5
            or labels[left_a].mean() != 0.5
            or labels[right_a].mean() != 0.5
        ):
            raise ValueError(f"{split} anchor marginals expose the long-range property")
    return groups


def _lm_loss(
    model: ProspectiveLongRangeLM, cohort: EncodedProspectiveCohort
) -> torch.Tensor:
    logits, _, _, _ = model(cohort.input_ids, cohort.attention_mask)
    target = cohort.input_ids[:, 1:]
    valid = cohort.attention_mask[:, 1:]
    return F.cross_entropy(logits[:, :-1][valid], target[valid])


def _train_lm(
    model: ProspectiveLongRangeLM,
    cohort: EncodedProspectiveCohort,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    training = spec["training"]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training["learning_rate"])
    )
    initial = float(_lm_loss(model, cohort).detach())
    model.train()
    for _ in range(training["lm_steps"]):
        optimizer.zero_grad(set_to_none=True)
        loss = _lm_loss(model, cohort)
        loss.backward()
        optimizer.step()
    model.eval()
    final = float(_lm_loss(model, cohort).detach())
    return {
        "steps": int(training["lm_steps"]),
        "initial_loss": initial,
        "final_loss": final,
        "loss_improved": bool(final < initial),
    }


@torch.no_grad()
def _captures(
    model: ProspectiveLongRangeLM, cohort: EncodedProspectiveCohort
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    _, inputs, targets, attentions = model(
        cohort.input_ids, cohort.attention_mask, capture=True
    )
    return (
        [value.detach() for value in inputs],
        [value.detach() for value in targets],
        [value.detach() for value in attentions],
    )


def _train_clt(
    model: ProspectiveLongRangeLM,
    cohort: EncodedProspectiveCohort,
    spec: Mapping[str, Any],
    model_seed: int,
) -> tuple[CLTForTraining, dict[str, Any]]:
    inputs, targets, _ = _captures(model, cohort)
    model_spec = spec["model"]
    training = spec["training"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(_derived_seed("prospective-topk-clt", model_seed))
        clt = CLTForTraining(
            n_layers=2,
            d_model=model.d_model,
            d_clt=int(model_spec["d_clt"]),
            k=int(model_spec["k"]),
            window=1,
        ).to(cohort.input_ids.device)
    optimizer = torch.optim.Adam(
        clt.parameters(), lr=float(training["clt_learning_rate"])
    )
    initial = float(clt(inputs, targets, cohort.attention_mask)["loss"].detach())
    clt.train()
    for _ in range(training["clt_steps"]):
        optimizer.zero_grad(set_to_none=True)
        output = clt(inputs, targets, cohort.attention_mask)
        output["loss"].backward()
        optimizer.step()
    clt.eval()
    with torch.no_grad():
        final = clt(inputs, targets, cohort.attention_mask)
    return clt, {
        "steps": int(training["clt_steps"]),
        "initial_loss": initial,
        "final_loss": float(final["loss"]),
        "fvu_mean": float(final["fvu_mean"]),
        "fvu_per_layer": [float(value) for value in final["fvu_per_layer"]],
        "l0_mean": float(final["l0_mean"]),
    }


def _endpoint_log_odds(
    logits: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    rows = torch.arange(logits.shape[0], device=logits.device)
    selected = logits[rows, positions]
    return selected[:, TOKEN_TO_ID["Y"]] - selected[:, TOKEN_TO_ID["W"]]


def _endpoint_accuracy(
    model: ProspectiveLongRangeLM, cohort: EncodedProspectiveCohort
) -> float:
    with torch.no_grad():
        logits, _, _, _ = model(cohort.input_ids, cohort.attention_mask)
    predicted = _endpoint_log_odds(logits, cohort.predictor_positions) > 0
    truth = torch.tensor(
        [row.long_range_equal for row in cohort.records],
        dtype=torch.bool,
        device=predicted.device,
    )
    return float((predicted == truth).float().mean())


def _confound_matrix(records: Sequence[ProspectiveRecord]) -> np.ndarray:
    matrix = np.column_stack(
        (
            np.ones(len(records)),
            [row.motif == "HGG" for row in records],
            [row.motif_position == "early" for row in records],
            [row.motif_position == "middle" for row in records],
            [row.length for row in records],
            [row.sequence.count("GG") + row.sequence.count("CP") for row in records],
            [row.family == "alpha" for row in records],
            [row.family == "beta" for row in records],
            [row.family == "gly_pro" for row in records],
            [row.n_terminal == "M" for row in records],
            [row.left_anchor_residue == "A" for row in records],
            [row.right_anchor_residue == "A" for row in records],
            [row.nuisance1_left_residue == "A" for row in records],
            [row.nuisance1_right_residue == "A" for row in records],
            [row.nuisance1_equal for row in records],
            [row.nuisance2_left_residue == "A" for row in records],
            [row.nuisance2_right_residue == "A" for row in records],
            [row.nuisance2_equal for row in records],
        )
    ).astype(np.float64)
    if np.linalg.matrix_rank(matrix) != matrix.shape[1]:
        raise ValueError("prospective discovery adjustment matrix is rank deficient")
    return matrix


def _logit_effect_vector(
    model: ProspectiveLongRangeLM, direction: np.ndarray
) -> np.ndarray:
    rotation = model.rotation.detach().cpu().numpy()
    base = direction @ rotation
    content = base[: model.d_content]
    weights = model.content_head.weight.detach().cpu().numpy()
    effect = weights @ content
    projection = float(direction @ model.planted_direction.detach().cpu().numpy())
    effect[TOKEN_TO_ID["Y"]] = model.readout_gain * projection
    effect[TOKEN_TO_ID["W"]] = -model.readout_gain * projection
    nuisance = direction @ model.nuisance_directions.detach().cpu().numpy()
    for index, (positive, negative) in enumerate((("A", "C"), ("D", "E"))):
        effect[TOKEN_TO_ID[positive]] += model.readout_gain * nuisance[index]
        effect[TOKEN_TO_ID[negative]] -= model.readout_gain * nuisance[index]
    return effect


def _feature_table(
    model: ProspectiveLongRangeLM,
    clt: CLTForTraining,
    cohort: EncodedProspectiveCohort,
    spec: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[int], int]:
    inputs, _, attentions = _captures(model, cohort)
    with torch.no_grad():
        codes = clt.encode(inputs)[1]
    rows = torch.arange(codes.shape[0], device=codes.device)
    values = codes[rows, cohort.predictor_positions].detach().cpu().numpy()
    labels = np.asarray(
        [record.long_range_equal for record in cohort.records], dtype=np.float64
    )
    confounds = _confound_matrix(cohort.records)
    full = np.column_stack((confounds, labels))
    if np.linalg.matrix_rank(full) != full.shape[1]:
        raise ValueError("long-range equality is aliased with a discovery confound")
    planted = model.planted_direction.detach().cpu().numpy()
    predictor = cohort.predictor_positions
    attention_values = []
    for row, position in enumerate(predictor.tolist()):
        valid_length = int(cohort.attention_mask[row].sum().item())
        attention_values.append(
            float(
                attentions[1][row, position:valid_length, position]
                .sum()
                .detach()
                .cpu()
            )
        )
    received_attention = float(np.mean(attention_values))
    direction_cosine_min = float(
        spec["discovery"]["ground_truth_direction_cosine_min"]
    )
    endpoint_effect_min = 2.0 * model.readout_gain * direction_cosine_min
    table: list[dict[str, Any]] = []
    pvalues = []
    for feature in range(values.shape[1]):
        outcome = values[:, feature]
        reduced_fit = confounds @ np.linalg.lstsq(confounds, outcome, rcond=None)[0]
        coefficients = np.linalg.lstsq(full, outcome, rcond=None)[0]
        residual = outcome - full @ coefficients
        dof = len(outcome) - np.linalg.matrix_rank(full)
        inverse = np.linalg.pinv(full.T @ full)
        variance = float(residual @ residual / dof) if dof > 0 else 0.0
        standard_error = float(np.sqrt(max(variance * inverse[-1, -1], 0.0)))
        coefficient = float(coefficients[-1])
        if standard_error == 0.0:
            pvalue = 0.0 if coefficient > 0.0 else 1.0
        else:
            pvalue = float(stats.t.sf(coefficient / standard_error, dof))
        total = float(np.sum((outcome - outcome.mean()) ** 2))
        incremental_r2 = (
            float(np.sum((outcome - reduced_fit) ** 2) - residual @ residual) / total
            if total > 0.0
            else 0.0
        )
        decoder = clt.W_dec[1][feature, 0].detach().cpu().numpy().copy()
        decoder_norm = float(np.linalg.norm(decoder))
        unit = decoder / max(decoder_norm, 1e-12)
        cosine = float(unit @ planted)
        logit_effect = _logit_effect_vector(model, unit)
        mean_activation = float(outcome.mean())
        table.append(
            {
                "layer": 1,
                "feature": int(feature),
                "semantic_coefficient": coefficient,
                "semantic_p_positive": pvalue,
                "semantic_incremental_r2": incremental_r2,
                "firing_frequency": float(np.mean(outcome > 0.0)),
                "mean_activation": mean_activation,
                "decoder_norm": decoder_norm,
                "direct_logit_effect_norm": float(np.linalg.norm(logit_effect)),
                "received_attention_mass": received_attention,
                "reconstruction_contribution": mean_activation * decoder_norm,
                "direct_endpoint_effect_per_dose": float(
                    logit_effect[TOKEN_TO_ID["Y"]]
                    - logit_effect[TOKEN_TO_ID["W"]]
                ),
                "ground_truth_direction_cosine": cosine,
                "minimum_direct_endpoint_effect_per_dose": endpoint_effect_min,
                "active_ground_truth_positive": bool(
                    mean_activation > 0.0 and cosine >= direction_cosine_min
                ),
            }
        )
        pvalues.append(pvalue)
    adjusted = benjamini_hochberg(pvalues)
    for row, qvalue in zip(table, adjusted):
        row["semantic_q_bh"] = float(qvalue)
        row["selected_positive"] = bool(
            row["semantic_coefficient"] > 0.0
            and qvalue < spec["discovery"]["selection_alpha_bh"]
            and row["direct_endpoint_effect_per_dose"]
            >= row["minimum_direct_endpoint_effect_per_dose"]
        )
        row["discovery_score"] = float(
            max(row["semantic_incremental_r2"], 0.0)
            * max(row["direct_endpoint_effect_per_dose"], 0.0)
        )
    selected = [int(row["feature"]) for row in table if row["selected_positive"]]
    if not selected:
        raise RuntimeError("discovery selected no positive dictionary feature")
    target = max(
        (row for row in table if row["selected_positive"]),
        key=lambda row: (row["discovery_score"], -row["feature"]),
    )
    return table, selected, int(target["feature"])


def _abs_log_ratio(left: float, right: float) -> float:
    floor = 1e-8
    return abs(math.log(max(left, floor) / max(right, floor)))


def _match_controls(
    table: Sequence[Mapping[str, Any]],
    target_feature: int,
    spec: Mapping[str, Any],
    model: ProspectiveLongRangeLM,
) -> list[dict[str, Any]]:
    """Construct active, orthogonal nuisance-path controls under hard calipers."""

    matching = spec["matching"]
    calipers = matching["hard_calipers"]
    target = next(row for row in table if row["feature"] == target_feature)
    target_profile = {
        "firing_frequency": float(target["firing_frequency"]),
        "mean_activation": float(target["mean_activation"]),
        "decoder_norm": 1.0,
        "direct_logit_effect_norm": float(target["direct_logit_effect_norm"]),
        "received_attention_mass": float(target["received_attention_mass"]),
        "reconstruction_contribution": float(target["mean_activation"]),
    }
    controls = []
    for index in range(model.nuisance_paths):
        direction = model.nuisance_directions[:, index].detach().cpu().numpy()
        controls.append(
            {
                "control_path": f"orthogonal_nuisance_relation_{index + 1}",
                "control_direction_index": index,
                "ground_truth_direction_cosine": float(
                    direction @ model.planted_direction.detach().cpu().numpy()
                ),
                "profile": {
                    "firing_frequency": target_profile["firing_frequency"],
                    "mean_activation": target_profile["mean_activation"],
                    "decoder_norm": 1.0,
                    "direct_logit_effect_norm": float(
                        np.linalg.norm(_logit_effect_vector(model, direction))
                    ),
                    "received_attention_mass": target_profile[
                        "received_attention_mass"
                    ],
                    "reconstruction_contribution": target_profile[
                        "reconstruction_contribution"
                    ],
                },
                "activation_gate": "exact_discovery_target_activation_trace",
            }
        )
    matrix = np.asarray(
        [
            [profile[field] for field in PROFILE_FIELDS]
            for profile in (target_profile, *(row["profile"] for row in controls))
        ],
        dtype=np.float64,
    )
    scale = matrix.std(axis=0)
    scale[scale < 1e-8] = 1.0
    target_vector = matrix[0]
    eligible = []
    for row, vector in zip(controls, matrix[1:]):
        profile = row["profile"]
        diagnostics = {
            "firing_frequency_abs": abs(
                profile["firing_frequency"] - target_profile["firing_frequency"]
            ),
            "mean_activation_abs_log_ratio": _abs_log_ratio(
                profile["mean_activation"], target_profile["mean_activation"]
            ),
            "decoder_norm_abs_log_ratio": _abs_log_ratio(
                profile["decoder_norm"], target_profile["decoder_norm"]
            ),
            "direct_logit_effect_norm_abs_log_ratio": _abs_log_ratio(
                profile["direct_logit_effect_norm"],
                target_profile["direct_logit_effect_norm"],
            ),
            "received_attention_mass_abs": abs(
                profile["received_attention_mass"]
                - target_profile["received_attention_mass"]
            ),
            "reconstruction_contribution_abs_log_ratio": _abs_log_ratio(
                profile["reconstruction_contribution"],
                target_profile["reconstruction_contribution"],
            ),
            "standardized_distance": float(
                np.linalg.norm((vector - target_vector) / scale)
            ),
        }
        if (
            abs(row["ground_truth_direction_cosine"])
            <= matching["control_abs_ground_truth_cosine_max"]
            and all(
            diagnostics[field] <= calipers[field]
            for field in CALIPER_FIELDS - {"standardized_distance_max"}
            )
            and diagnostics["standardized_distance"]
            <= calipers["standardized_distance_max"]
        ):
            eligible.append({**row, "caliper_diagnostics": diagnostics})
    eligible.sort(
        key=lambda row: (
            row["caliper_diagnostics"]["standardized_distance"],
            row["control_path"],
        )
    )
    count = matching["control_count"]
    if len(eligible) < count:
        raise RuntimeError(
            "hard matched-control calipers failed: "
            f"required {count}, found {len(eligible)}; calipers are not widened"
        )
    return eligible[:count]


def _intervention_effects(
    model: ProspectiveLongRangeLM,
    cohort: EncodedProspectiveCohort,
    direction: torch.Tensor,
    *,
    layer: int,
    positions: torch.Tensor,
    dose: float,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        baseline, _, _, _ = model(cohort.input_ids, cohort.attention_mask)
        changed, _, _, _ = model(
            cohort.input_ids,
            cohort.attention_mask,
            intervention_layer=layer,
            intervention_positions=positions,
            intervention_direction=direction,
            intervention_dose=dose,
        )
    baseline_odds = _endpoint_log_odds(baseline, cohort.predictor_positions)
    changed_odds = _endpoint_log_odds(changed, cohort.predictor_positions)
    probability_change = torch.sigmoid(changed_odds) - torch.sigmoid(baseline_odds)
    return (
        (changed_odds - baseline_odds).detach().cpu().numpy(),
        probability_change.detach().cpu().numpy(),
    )


def _fit_and_select(
    train: Sequence[ProspectiveRecord],
    discovery: Sequence[ProspectiveRecord],
    *,
    model_seed: int,
    spec: Mapping[str, Any],
    device: torch.device,
) -> tuple[
    ProspectiveLongRangeLM,
    CLTForTraining,
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[int],
    int,
    list[dict[str, Any]],
]:
    """Fit/select without accepting assessment records as an argument."""

    train_cohort = encode_prospective_cohort(train).to(device)
    discovery_cohort = encode_prospective_cohort(discovery).to(device)
    model = ProspectiveLongRangeLM(model_seed, spec).to(device)
    lm_training = _train_lm(model, train_cohort, spec)
    clt, clt_training = _train_clt(model, train_cohort, spec, model_seed)
    table, selected, target = _feature_table(model, clt, discovery_cohort, spec)
    controls = _match_controls(table, target, spec, model)
    return (
        model,
        clt,
        lm_training,
        clt_training,
        table,
        selected,
        target,
        controls,
    )


def _all_seed_detection_gates(
    model_results: Sequence[Mapping[str, Any]], gates: Mapping[str, Any]
) -> bool:
    """Require every seed to pass; aggregate means are descriptive only."""

    return bool(
        model_results
        and all(
            float(row["sensitivity"]) >= float(gates["sensitivity_min"])
            and float(row["specificity"]) >= float(gates["specificity_min"])
            and float(row["false_discovery_rate"])
            <= float(gates["false_discovery_rate_max"])
            for row in model_results
        )
    )


def _detection_metrics(
    table: Sequence[Mapping[str, Any]], selected: Sequence[int]
) -> tuple[set[int], float, float, float]:
    """Score one planted mechanism and coordinate-level negative candidates."""

    truth = {
        int(row["feature"])
        for row in table
        if bool(row["active_ground_truth_positive"])
    }
    selected_set = set(selected)
    true_positive = len(truth & selected_set)
    false_positive = len(selected_set - truth)
    true_negative = len(table) - len(truth) - false_positive
    return (
        truth,
        float(true_positive > 0),
        true_negative / max(len(table) - len(truth), 1),
        false_positive / max(len(selected_set), 1),
    )


def run_prospective_positive_control(
    records: Sequence[ProspectiveRecord],
    spec: Mapping[str, Any],
    *,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the frozen v2 benchmark once and assess untouched records last."""

    frozen = validate_prospective_spec(spec)
    groups = _split_records(records, frozen)
    device = torch.device(frozen["training"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("frozen P0-7 spec requests CUDA but CUDA is unavailable")
    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_root is not None:
        checkpoint_root.mkdir(parents=True, exist_ok=False)
    split_hashes = {
        split: _canonical_hash([row.to_dict() for row in rows])
        for split, rows in groups.items()
    }
    model_results: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []
    assessment_signatures: list[np.ndarray] = []
    margin = frozen["intervention"]["negative_equivalence_margin_log_odds"]
    recovery_lower, recovery_upper = frozen["gates"]["effect_recovery_ratio"]

    for model_seed in frozen["model"]["seeds"]:
        (
            model,
            clt,
            lm_training,
            clt_training,
            table,
            selected,
            target_feature,
            controls,
        ) = _fit_and_select(
            groups["train"],
            groups["discovery"],
            model_seed=int(model_seed),
            spec=frozen,
            device=device,
        )
        for row in table:
            feature_rows.append({"model_seed": int(model_seed), **row})

        # Dead decoder rows are not recovered sparse features: counting their
        # random orientation would corrupt the sensitivity denominator.
        truth_positive, sensitivity, specificity, false_discovery_rate = (
            _detection_metrics(table, selected)
        )

        assessment = encode_prospective_cohort(groups["assessment"]).to(device)
        eval_inputs, _, _ = _captures(model, assessment)
        with torch.no_grad():
            codes = clt.encode(eval_inputs)[1]
        rows = torch.arange(codes.shape[0], device=device)
        endpoint_codes = codes[rows, assessment.predictor_positions]
        decoder_rows = clt.W_dec[1][:, 0].detach()
        decoder_cosines = (
            decoder_rows @ model.planted_direction
        ) / decoder_rows.norm(dim=1).clamp_min(1e-12)
        signature = (
            endpoint_codes[:, selected]
            * decoder_cosines[torch.as_tensor(selected, device=device)]
        ).sum(dim=1).detach().cpu().numpy()
        adjustment = _confound_matrix(assessment.records)
        signature -= adjustment @ np.linalg.lstsq(
            adjustment, signature, rcond=None
        )[0]
        assessment_signatures.append(signature)

        target_direction = clt.W_dec[1][target_feature, 0].detach()
        target_direction = target_direction / target_direction.norm().clamp_min(1e-12)
        target_cosine = float(target_direction @ model.planted_direction)
        dose_rows = []
        for dose in frozen["intervention"]["doses"]:
            effects, probability_changes = _intervention_effects(
                model,
                assessment,
                target_direction,
                layer=1,
                positions=assessment.predictor_positions,
                dose=float(dose),
            )
            known_effect = 2.0 * model.readout_gain * target_cosine * float(dose)
            observed = mean_interval(effects)
            recovery_ratio = float(observed["mean"] / known_effect)
            dose_rows.append(
                {
                    "dose": float(dose),
                    "known_effect": known_effect,
                    "observed_log_odds_effect": observed,
                    "mean_endpoint_probability_change": float(
                        probability_changes.mean()
                    ),
                    "effect_recovery_ratio": recovery_ratio,
                    "recovery_inside_frozen_interval": bool(
                        recovery_lower <= recovery_ratio <= recovery_upper
                    ),
                }
            )
            intervention_rows.extend(
                {
                    "model_seed": int(model_seed),
                    "record_id": record.record_id,
                    "arm": "selected_known_layer_position_direction",
                    "layer": 1,
                    "position": int(position),
                    "feature": target_feature,
                    "dose": float(dose),
                    "endpoint_log_odds_difference": float(effect),
                    "endpoint_probability_change": float(probability_change),
                }
                for record, position, effect, probability_change in zip(
                    assessment.records,
                    assessment.predictor_positions.tolist(),
                    effects,
                    probability_changes,
                )
            )

        wrong_positions = assessment.predictor_positions - 1
        site_controls = {}
        for dose in frozen["intervention"]["doses"]:
            for arm, layer, positions in (
                ("selected_wrong_layer", 0, assessment.predictor_positions),
                ("selected_wrong_position", 1, wrong_positions),
            ):
                effects, probability_changes = _intervention_effects(
                    model,
                    assessment,
                    target_direction,
                    layer=layer,
                    positions=positions,
                    dose=float(dose),
                )
                site_controls[f"{arm}_dose_{float(dose):g}"] = tost_paired(
                    effects, margin
                )
                intervention_rows.extend(
                    {
                        "model_seed": int(model_seed),
                        "record_id": record.record_id,
                        "arm": arm,
                        "layer": layer,
                        "position": int(position),
                        "feature": target_feature,
                        "dose": float(dose),
                        "endpoint_log_odds_difference": float(effect),
                        "endpoint_probability_change": float(probability_change),
                    }
                    for record, position, effect, probability_change in zip(
                        assessment.records,
                        positions.tolist(),
                        effects,
                        probability_changes,
                    )
                )

        control_results = []
        decoder_directions = clt.W_dec[1][:, 0].detach()
        decoder_directions = decoder_directions / decoder_directions.norm(
            dim=1, keepdim=True
        ).clamp_min(1e-12)
        for control in controls:
            direction = model.nuisance_directions[
                :, int(control["control_direction_index"])
            ]
            off_target = decoder_directions @ direction
            control_doses = []
            for dose in frozen["intervention"]["doses"]:
                effects, probability_changes = _intervention_effects(
                    model,
                    assessment,
                    direction,
                    layer=1,
                    positions=assessment.predictor_positions,
                    dose=float(dose),
                )
                control_doses.append(
                    {
                        "dose": float(dose),
                        "endpoint_log_odds_equivalence": tost_paired(
                            effects, margin
                        ),
                        "mean_endpoint_probability_change": float(
                            probability_changes.mean()
                        ),
                        "intended_direction_displacement": float(dose),
                        "reconstruction_displacement": float(dose),
                    }
                )
                intervention_rows.extend(
                    {
                        "model_seed": int(model_seed),
                        "record_id": record.record_id,
                        "arm": "hard_matched_known_nuisance_path_control",
                        "layer": 1,
                        "position": int(position),
                        "feature": None,
                        "control_path": control["control_path"],
                        "dose": float(dose),
                        "endpoint_log_odds_difference": float(effect),
                        "endpoint_probability_change": float(probability_change),
                    }
                    for record, position, effect, probability_change in zip(
                        assessment.records,
                        assessment.predictor_positions.tolist(),
                        effects,
                        probability_changes,
                    )
                )
            control_results.append(
                {
                    **control,
                    "off_target_decoder_projection_l2": float(off_target.norm()),
                    "dose_sweep": control_doses,
                    "all_doses_equivalent": bool(
                        all(
                            row["endpoint_log_odds_equivalence"]["equivalent"]
                            for row in control_doses
                        )
                    ),
                }
            )

        endpoint_accuracy = _endpoint_accuracy(model, assessment)
        checkpoint = None
        if checkpoint_root is not None:
            checkpoint_path = checkpoint_root / f"model_seed_{model_seed}.pt"
            torch.save(
                {
                    "schema_version": "r2_p0_7_prospective_checkpoint_v2",
                    "model_seed": int(model_seed),
                    "spec_sha256": _canonical_hash(frozen),
                    "model_state": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "clt_state": {
                        key: value.detach().cpu()
                        for key, value in clt.state_dict().items()
                    },
                },
                checkpoint_path,
            )
            checkpoint = {
                "path": f"{checkpoint_root.name}/{checkpoint_path.name}",
                "sha256": sha256_file(checkpoint_path),
            }
        negative_equivalence_passed = bool(
            all(row["equivalent"] for row in site_controls.values())
            and all(row["all_doses_equivalent"] for row in control_results)
        )
        model_results.append(
            {
                "model_seed": int(model_seed),
                "model_state_sha256": _state_hash(model),
                "clt_state_sha256": _state_hash(clt),
                "checkpoint": checkpoint,
                "lm_training": {
                    **lm_training,
                    "assessment_loss": float(_lm_loss(model, assessment).detach()),
                    "assessment_endpoint_accuracy": endpoint_accuracy,
                },
                "clt_training": clt_training,
                "known_path": {
                    "layer": 1,
                    "site": "layer_1_mlp_output_at_endpoint_predictor",
                    "direction_sha256": hashlib.sha256(
                        model.planted_direction.detach().cpu().numpy().tobytes()
                    ).hexdigest(),
                    "positive_direction": True,
                    "relation": "distant_anchor_residue_equality",
                },
                "selected_features": selected,
                "selected_feature": target_feature,
                "selected_ground_truth_cosine": target_cosine,
                "ground_truth_positive_features": sorted(truth_positive),
                "sensitivity_unit": "one_planted_mechanism_per_model_seed",
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "false_discovery_rate": float(false_discovery_rate),
                "matched_controls": control_results,
                "site_controls": site_controls,
                "dose_sweep": dose_rows,
                "negative_equivalence_passed": negative_equivalence_passed,
                "path_localized": bool(
                    target_feature in truth_positive and negative_equivalence_passed
                ),
            }
        )

    alignments = []
    seeds = frozen["model"]["seeds"]
    for left in range(len(seeds)):
        for right in range(left + 1, len(seeds)):
            correlation = float(
                stats.spearmanr(
                    assessment_signatures[left], assessment_signatures[right]
                ).statistic
            )
            alignments.append(
                {
                    "model_seed_a": int(seeds[left]),
                    "model_seed_b": int(seeds[right]),
                    "assessment_confound_adjusted_activation_spearman": correlation,
                }
            )

    sensitivity = float(np.mean([row["sensitivity"] for row in model_results]))
    specificity = float(np.mean([row["specificity"] for row in model_results]))
    fdr = float(np.mean([row["false_discovery_rate"] for row in model_results]))
    recovery = [
        dose["effect_recovery_ratio"]
        for row in model_results
        for dose in row["dose_sweep"]
    ]
    gates = frozen["gates"]
    passed = bool(
        _all_seed_detection_gates(model_results, gates)
        and all(row["path_localized"] for row in model_results)
        and all(row["lm_training"]["loss_improved"] for row in model_results)
        and all(
            row["lm_training"]["assessment_endpoint_accuracy"]
            >= gates["endpoint_accuracy_min"]
            for row in model_results
        )
        and all(
            row["clt_training"]["fvu_mean"] <= gates["clt_fvu_max"]
            and row["clt_training"]["fvu_per_layer"][1] <= gates["clt_fvu_max"]
            for row in model_results
        )
        and all(recovery_lower <= value <= recovery_upper for value in recovery)
        and all(
            row["assessment_confound_adjusted_activation_spearman"]
            >= gates["activation_alignment_spearman_min"]
            for row in alignments
        )
    )
    return {
        "schema_version": RESULT_SCHEMA,
        "status": (
            "prospective_synthetic_gate_passed"
            if passed
            else "prospective_synthetic_gate_failed"
        ),
        "claim_scope": CLAIM_SCOPE,
        "pretrained_model_causal_inference": False,
        "legacy_controls_upgraded": False,
        "input_contract": {
            "model_visible_tokens": "BOS + canonical amino acids + EOS only",
            "long_range_label_token_exposed": False,
            "endpoint_is_future_canonical_residue": True,
            "assessment_used_for_selection": False,
            "anchor_marginals_balanced": True,
            "alignment_signature": (
                "discovery-selected sparse activations weighted by frozen "
                "known-direction cosine"
            ),
        },
        "split_counts": {split: len(rows) for split, rows in groups.items()},
        "split_sha256": split_hashes,
        "vocabulary": list(TOKENS),
        "vocabulary_sha256": _canonical_hash(TOKENS),
        "frozen_gates": gates,
        "matching_contract": frozen["matching"],
        "aggregate": {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "false_discovery_rate": fdr,
            "mean_effect_recovery_ratio": float(np.mean(recovery)),
            "all_negative_metrics_equivalent": bool(
                all(row["negative_equivalence_passed"] for row in model_results)
            ),
            "all_paths_localized": bool(
                all(row["path_localized"] for row in model_results)
            ),
            "prospective_synthetic_gate_passed": passed,
        },
        "cross_model_alignment": alignments,
        "models": model_results,
        "feature_discovery_rows": feature_rows,
        "intervention_rows": intervention_rows,
    }


def _read_records(path: Path) -> list[ProspectiveRecord]:
    fields = set(ProspectiveRecord.__dataclass_fields__)
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(
            line,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"{path}:{line_number} non-finite constant {constant}")
            ),
        )
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError(f"{path}:{line_number} has an invalid record schema")
        records.append(ProspectiveRecord(**value))
    if not records:
        raise ValueError("frozen prospective cohort is empty")
    return records


def _write_exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        dict(value), sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
    with path.open("x") as handle:
        handle.write(payload)


def _execution_source_hashes(runner_path: Path) -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {
        Path(__file__).name: sha256_file(Path(__file__)),
        Path(runner_path).name: sha256_file(Path(runner_path)),
        "causal_positive_control.py": sha256_file(
            root / "src/revision/causal_positive_control.py"
        ),
        "statistics.py": sha256_file(root / "src/revision/statistics.py"),
        "clt_trainer.py": sha256_file(root / "src/training/clt_trainer.py"),
    }


def freeze_prospective_benchmark(
    spec_path: Path,
    expected_spec_sha256: str,
    output_dir: Path,
    *,
    runner_path: Path,
    command: Sequence[str],
) -> Path:
    """Publish a hash-bound cohort/spec freeze before any model execution."""

    spec_path = Path(spec_path).resolve()
    expected = _require_sha256(expected_spec_sha256, "prospective spec SHA-256")
    if not spec_path.is_file() or sha256_file(spec_path) != expected:
        raise ValueError("prospective spec is missing or its SHA-256 changed")
    spec = validate_prospective_spec(_strict_json(spec_path))
    output = Path(output_dir).resolve()
    staging = output.with_name(f".{output.name}.staging")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite benchmark freeze: {output}")
    if staging.exists():
        raise FileExistsError(f"stale benchmark freeze staging directory: {staging}")
    staging.mkdir(parents=True)
    try:
        frozen_spec = staging / "frozen_spec.json"
        shutil.copyfile(spec_path, frozen_spec)
        cohort_path = staging / "cohort.jsonl"
        records = generate_prospective_cohort(spec)
        write_jsonl(cohort_path, (record.to_dict() for record in records))
        artifacts = {
            frozen_spec.name: sha256_file(frozen_spec),
            cohort_path.name: sha256_file(cohort_path),
        }
        source_hashes = _execution_source_hashes(runner_path)
        freeze_id = _canonical_hash(
            {
                "spec_sha256": expected,
                "artifact_hashes": artifacts,
                "source_hashes": source_hashes,
            }
        )
        manifest_path = staging / "freeze_manifest.json"
        write_json(
            manifest_path,
            {
                "schema_version": FREEZE_SCHEMA,
                "status": "frozen_unexecuted",
                "freeze_id": freeze_id,
                "claim_scope": CLAIM_SCOPE,
                "command": list(command),
                "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "spec": {
                    "source_path": str(spec_path),
                    "sha256": expected,
                    "frozen_artifact": frozen_spec.name,
                },
                "cohort_contract": {
                    "splits": list(SPLITS),
                    "counts": {
                        split: sum(row.split == split for row in records)
                        for split in SPLITS
                    },
                    "assessment_used_for_selection": False,
                    "property_label_token_exposed": False,
                },
                "artifact_hashes": artifacts,
                "source_hashes": source_hashes,
            },
        )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output / "freeze_manifest.json"


def _verify_freeze(
    frozen_dir: Path,
    expected_manifest_sha256: str,
    *,
    runner_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[ProspectiveRecord], Path]:
    frozen = Path(frozen_dir).resolve()
    manifest_path = frozen / "freeze_manifest.json"
    expected = _require_sha256(
        expected_manifest_sha256, "prospective freeze-manifest SHA-256"
    )
    if not manifest_path.is_file() or sha256_file(manifest_path) != expected:
        raise ValueError("prospective freeze manifest is missing or its SHA-256 changed")
    manifest = _require_keys(
        _strict_json(manifest_path),
        {
            "schema_version",
            "status",
            "freeze_id",
            "claim_scope",
            "command",
            "frozen_at_utc",
            "spec",
            "cohort_contract",
            "artifact_hashes",
            "source_hashes",
        },
        "prospective freeze manifest",
    )
    if (
        manifest["schema_version"] != FREEZE_SCHEMA
        or manifest["status"] != "frozen_unexecuted"
        or manifest["claim_scope"] != CLAIM_SCOPE
    ):
        raise ValueError("prospective freeze manifest is incomplete")
    expected_sources = _execution_source_hashes(runner_path)
    if manifest["source_hashes"] != expected_sources:
        raise ValueError("prospective execution code changed after the benchmark freeze")
    artifacts = manifest["artifact_hashes"]
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "frozen_spec.json",
        "cohort.jsonl",
    }:
        raise ValueError("prospective freeze artifact set is incomplete")
    for filename, digest in artifacts.items():
        _require_sha256(digest, f"{filename} SHA-256")
        path = frozen / filename
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"frozen artifact changed: {filename}")
    spec_descriptor = _require_keys(
        manifest["spec"], {"source_path", "sha256", "frozen_artifact"}, "frozen spec"
    )
    if (
        spec_descriptor["frozen_artifact"] != "frozen_spec.json"
        or spec_descriptor["sha256"] != artifacts["frozen_spec.json"]
    ):
        raise ValueError("freeze manifest does not bind the exact prospective spec")
    spec = validate_prospective_spec(_strict_json(frozen / "frozen_spec.json"))
    records = _read_records(frozen / "cohort.jsonl")
    _split_records(records, spec)
    expected_freeze_id = _canonical_hash(
        {
            "spec_sha256": spec_descriptor["sha256"],
            "artifact_hashes": artifacts,
            "source_hashes": expected_sources,
        }
    )
    if manifest["freeze_id"] != expected_freeze_id:
        raise ValueError("prospective freeze_id is invalid")
    return manifest, spec, records, manifest_path


def execute_frozen_prospective_benchmark(
    frozen_dir: Path,
    expected_manifest_sha256: str,
    output_dir: Path,
    *,
    runner_path: Path,
    command: Sequence[str],
) -> Path:
    """Claim and execute one freeze; any claim permanently forbids retuning."""

    frozen = Path(frozen_dir).resolve()
    output = Path(output_dir).resolve()
    staging = output.with_name(f".{output.name}.staging")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite prospective result: {output}")
    if staging.exists():
        raise FileExistsError(f"stale prospective result staging directory: {staging}")
    manifest, spec, records, freeze_manifest_path = _verify_freeze(
        frozen,
        expected_manifest_sha256,
        runner_path=runner_path,
    )
    claim_path = frozen / "execution_claim.json"
    _write_exclusive_json(
        claim_path,
        {
            "schema_version": "r2_p0_7_prospective_execution_claim_v1",
            "status": "claimed_no_retry_or_retuning",
            "freeze_id": manifest["freeze_id"],
            "freeze_manifest_sha256": sha256_file(freeze_manifest_path),
            "output_path": str(output),
            "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    staging.mkdir(parents=True)
    try:
        result = run_prospective_positive_control(
            records,
            spec,
            checkpoint_dir=staging / "checkpoints",
        )
        feature_rows = result.pop("feature_discovery_rows")
        intervention_rows = result.pop("intervention_rows")
        cohort_path = staging / "cohort.jsonl"
        shutil.copyfile(frozen / "cohort.jsonl", cohort_path)
        feature_path = staging / "feature_discovery.jsonl"
        intervention_path = staging / "interventions.jsonl"
        summary_path = staging / "summary.json"
        write_jsonl(feature_path, feature_rows)
        write_jsonl(intervention_path, intervention_rows)
        result["freeze"] = {
            "freeze_id": manifest["freeze_id"],
            "freeze_manifest_sha256": sha256_file(freeze_manifest_path),
            "spec_sha256": manifest["spec"]["sha256"],
            "execution_claim_sha256": sha256_file(claim_path),
        }
        result["artifacts"] = {
            "cohort": cohort_path.name,
            "feature_discovery": feature_path.name,
            "interventions": intervention_path.name,
            "checkpoints": [row["checkpoint"] for row in result["models"]],
        }
        write_json(summary_path, result)
        artifacts = (
            cohort_path,
            feature_path,
            intervention_path,
            summary_path,
            *sorted((staging / "checkpoints").glob("*.pt")),
        )
        run_manifest_path = staging / "run_manifest.json"
        write_json(
            run_manifest_path,
            {
                "schema_version": MANIFEST_SCHEMA,
                "status": "complete",
                "claim_scope": CLAIM_SCOPE,
                "command": list(command),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "freeze": result["freeze"],
                "environment": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                    "torch": torch.__version__,
                    "device": spec["training"]["device"],
                },
                "source_hashes": manifest["source_hashes"],
                "artifact_hashes": {
                    str(path.relative_to(staging)): sha256_file(path)
                    for path in artifacts
                },
            },
        )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output / "run_manifest.json"
