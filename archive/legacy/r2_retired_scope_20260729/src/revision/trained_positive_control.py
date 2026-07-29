"""Trained autoregressive planted-mechanism control for P0-7.

This is intentionally separate from :mod:`causal_positive_control`, whose v1
artifact is an analysis-only simulator.  Here a causal recurrent language
model and the repository's TopK CLT are both optimized from immutable training
records.  Candidate discovery and confirmatory intervention evaluation use
disjoint records.  A pass validates this bounded synthetic pipeline only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

from src.training.clt_trainer import CLTForTraining

from .causal_positive_control import FAMILY_ALPHABETS, GrammarRecord
from .io import sha256_file
from .statistics import benjamini_hochberg, mean_interval, tost_paired


AA = "ACDEFGHIKLMNPQRSTVWY"
SPECIAL_TOKENS = (
    "<pad>",
    "<bos>",
    "<eos>",
    "<query>",
    "<lr0>",
    "<lr1>",
    "<endpoint0>",
    "<endpoint1>",
    *(f"<family:{name}>" for name in FAMILY_ALPHABETS),
)
TOKENS = SPECIAL_TOKENS + tuple(AA)
TOKEN_TO_ID = {token: index for index, token in enumerate(TOKENS)}


def _derived_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31)


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _state_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ControlConfig:
    d_content: int = 8
    d_clt: int = 32
    k: int = 4
    lm_steps: int = 20
    clt_steps: int = 150
    learning_rate: float = 0.02
    clt_learning_rate: float = 0.03
    carrier_gain: float = 0.5
    adapter_gain: float = 1.0
    readout_gain: float = 1.5
    selection_alpha: float = 0.05
    equivalence_margin: float = 0.50
    matched_controls: int = 2
    doses: tuple[float, ...] = (0.5, 1.0, 2.0)

    def validate(self) -> None:
        if self.d_content < 4 or self.d_clt < 4:
            raise ValueError("d_content and d_clt must each be at least four")
        if not 1 <= self.k < self.d_clt:
            raise ValueError("k must lie in [1, d_clt)")
        if self.lm_steps < 1 or self.clt_steps < 1:
            raise ValueError("training steps must be positive")
        if self.learning_rate <= 0 or self.clt_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if self.adapter_gain <= 0 or self.readout_gain <= 0:
            raise ValueError("planted gains must be positive")
        if not 0 < self.selection_alpha < 0.5:
            raise ValueError("selection_alpha must lie in (0, 0.5)")
        if self.equivalence_margin <= 0 or self.matched_controls < 1:
            raise ValueError("invalid control or equivalence specification")
        if not self.doses or any(dose <= 0 for dose in self.doses):
            raise ValueError("doses must be non-empty and positive")


@dataclass
class EncodedCohort:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    predictor_positions: torch.Tensor
    records: list[GrammarRecord]

    def to(self, device: torch.device) -> "EncodedCohort":
        return EncodedCohort(
            self.input_ids.to(device),
            self.attention_mask.to(device),
            self.predictor_positions.to(device),
            self.records,
        )


def _encode_record(record: GrammarRecord) -> tuple[list[int], int]:
    ids = [
        TOKEN_TO_ID["<bos>"],
        TOKEN_TO_ID["<lr1>" if record.long_range else "<lr0>"],
        TOKEN_TO_ID[f"<family:{record.family}>"],
        *(TOKEN_TO_ID[residue] for residue in record.sequence),
        TOKEN_TO_ID["<query>"],
        TOKEN_TO_ID["<endpoint1>" if record.long_range else "<endpoint0>"],
        TOKEN_TO_ID["<eos>"],
    ]
    return ids, len(ids) - 3


def encode_cohort(records: Sequence[GrammarRecord]) -> EncodedCohort:
    if not records:
        raise ValueError("records must be non-empty")
    encoded = [_encode_record(record) for record in records]
    width = max(len(ids) for ids, _ in encoded)
    input_ids = torch.full(
        (len(records), width), TOKEN_TO_ID["<pad>"], dtype=torch.long
    )
    mask = torch.zeros((len(records), width), dtype=torch.bool)
    predictors = torch.empty(len(records), dtype=torch.long)
    for row, (ids, predictor) in enumerate(encoded):
        input_ids[row, : len(ids)] = torch.tensor(ids)
        mask[row, : len(ids)] = True
        predictors[row] = predictor
    return EncodedCohort(input_ids, mask, predictors, list(records))


class _CausalBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.gru = nn.GRU(width, width, batch_first=True)
        self.projection = nn.Linear(width, width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        recurrent, _ = self.gru(values)
        return 0.35 * torch.tanh(self.projection(recurrent))


class TinyPlantedAutoregressiveLM(nn.Module):
    """Two-layer causal LM with one rotated, fixed sparse adapter path."""

    def __init__(self, model_seed: int, config: ControlConfig, max_length: int) -> None:
        super().__init__()
        self.model_seed = int(model_seed)
        self.config = config
        self.d_model = config.d_content + 1
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(_derived_seed("tiny-lm", model_seed))
            self.embedding = nn.Embedding(
                len(TOKENS), config.d_content, padding_idx=TOKEN_TO_ID["<pad>"]
            )
            self.position = nn.Embedding(max_length, config.d_content)
            self.blocks = nn.ModuleList(
                [_CausalBlock(config.d_content), _CausalBlock(config.d_content)]
            )
            self.content_head = nn.Linear(config.d_content, len(TOKENS))
        rng = np.random.default_rng(_derived_seed("rotation", model_seed))
        rotation, _ = np.linalg.qr(rng.normal(size=(self.d_model, self.d_model)))
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1
        self.register_buffer("rotation", torch.tensor(rotation, dtype=torch.float32))
        self.register_buffer(
            "planted_direction", torch.tensor(rotation[:, -1], dtype=torch.float32)
        )

    def _rotate(self, base: torch.Tensor) -> torch.Tensor:
        return base @ self.rotation.T

    def _unrotate(self, observed: torch.Tensor) -> torch.Tensor:
        return observed @ self.rotation

    @staticmethod
    def _patch(
        values: torch.Tensor,
        positions: torch.Tensor,
        direction: torch.Tensor,
        dose: float,
    ) -> torch.Tensor:
        patched = values.clone()
        patched[torch.arange(values.shape[0], device=values.device), positions] += (
            float(dose) * direction
        )
        return patched

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        intervention_layer: int | None = None,
        intervention_positions: torch.Tensor | None = None,
        intervention_direction: torch.Tensor | None = None,
        intervention_dose: float = 0.0,
        capture: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        batch, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device)
        content = self.embedding(input_ids) + self.position(positions)[None]
        semantic = torch.where(
            input_ids[:, 1] == TOKEN_TO_ID["<lr1>"], 1.0, -1.0
        ).to(content.dtype)
        causal_semantic = semantic[:, None] * (positions[None] >= 1)
        zeros = torch.zeros((batch, length, 1), device=content.device, dtype=content.dtype)
        resid_pre: list[torch.Tensor] = []
        mlp_out: list[torch.Tensor] = []

        pre0 = self._rotate(torch.cat((content, zeros), dim=-1))
        delta0 = self._rotate(torch.cat((self.blocks[0](content), zeros), dim=-1))
        observed0 = pre0 + delta0
        if intervention_layer == 0:
            if intervention_positions is None or intervention_direction is None:
                raise ValueError("an intervention requires positions and a direction")
            observed0 = self._patch(
                observed0, intervention_positions, intervention_direction, intervention_dose
            )
        content = self._unrotate(observed0)[..., : self.config.d_content]
        resid_pre.append(pre0)
        mlp_out.append(delta0)

        carrier = self.config.carrier_gain * causal_semantic[..., None]
        pre1 = self._rotate(torch.cat((content, carrier), dim=-1))
        adapter = self.config.adapter_gain * causal_semantic[..., None]
        delta1 = self._rotate(
            torch.cat((self.blocks[1](content), adapter), dim=-1)
        )
        resid_pre.append(pre1)
        mlp_out.append(delta1)
        observed1 = pre1 + delta1
        if intervention_layer == 1:
            if intervention_positions is None or intervention_direction is None:
                raise ValueError("an intervention requires positions and a direction")
            observed1 = self._patch(
                observed1, intervention_positions, intervention_direction, intervention_dose
            )
        final_base = self._unrotate(observed1)
        logits = self.content_head(final_base[..., : self.config.d_content])
        # The two held-out endpoint rows are reserved for the planted path.
        # This makes its direct effect identifiable instead of allowing the
        # learned content head to create an unrecorded parallel mechanism.
        logits[..., TOKEN_TO_ID["<endpoint1>"]] = 0.0
        logits[..., TOKEN_TO_ID["<endpoint0>"]] = 0.0
        planted_score = observed1 @ self.planted_direction
        query = input_ids == TOKEN_TO_ID["<query>"]
        planted_score = planted_score * query
        logits[..., TOKEN_TO_ID["<endpoint1>"]] += self.config.readout_gain * planted_score
        logits[..., TOKEN_TO_ID["<endpoint0>"]] -= self.config.readout_gain * planted_score
        return logits, resid_pre if capture else [], mlp_out if capture else []


def _lm_loss(model: TinyPlantedAutoregressiveLM, cohort: EncodedCohort) -> torch.Tensor:
    logits, _, _ = model(cohort.input_ids)
    target = cohort.input_ids[:, 1:]
    valid = cohort.attention_mask[:, 1:]
    return F.cross_entropy(logits[:, :-1][valid], target[valid])


def _train_lm(
    model: TinyPlantedAutoregressiveLM,
    cohort: EncodedCohort,
    config: ControlConfig,
) -> tuple[float, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    initial = float(_lm_loss(model, cohort).detach())
    model.train()
    for _ in range(config.lm_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = _lm_loss(model, cohort)
        loss.backward()
        optimizer.step()
    model.eval()
    return initial, float(_lm_loss(model, cohort).detach())


@torch.no_grad()
def _captures(
    model: TinyPlantedAutoregressiveLM, cohort: EncodedCohort
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    _, inputs, targets = model(cohort.input_ids, capture=True)
    return [value.detach() for value in inputs], [value.detach() for value in targets]


def _train_clt(
    model: TinyPlantedAutoregressiveLM,
    cohort: EncodedCohort,
    config: ControlConfig,
    model_seed: int,
) -> tuple[CLTForTraining, dict]:
    inputs, targets = _captures(model, cohort)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(_derived_seed("topk-clt", model_seed))
        clt = CLTForTraining(
            n_layers=2,
            d_model=model.d_model,
            d_clt=config.d_clt,
            k=config.k,
            window=1,
        ).to(cohort.input_ids.device)
    optimizer = torch.optim.Adam(clt.parameters(), lr=config.clt_learning_rate)
    initial = float(clt(inputs, targets, cohort.attention_mask)["loss"].detach())
    clt.train()
    for _ in range(config.clt_steps):
        optimizer.zero_grad(set_to_none=True)
        output = clt(inputs, targets, cohort.attention_mask)
        output["loss"].backward()
        optimizer.step()
    clt.eval()
    with torch.no_grad():
        final = clt(inputs, targets, cohort.attention_mask)
    return clt, {
        "initial_loss": initial,
        "final_loss": float(final["loss"]),
        "fvu_mean": float(final["fvu_mean"]),
        "fvu_per_layer": [float(value) for value in final["fvu_per_layer"]],
        "l0_mean": float(final["l0_mean"]),
    }


def _split_records(records: Sequence[GrammarRecord]) -> dict[str, list[GrammarRecord]]:
    required = {"train", "discovery", "evaluation"}
    groups = {split: [row for row in records if row.split == split] for split in required}
    if any(not rows for rows in groups.values()):
        raise ValueError("records require non-empty train, discovery and evaluation splits")
    for split, rows in groups.items():
        if len({row.record_id for row in rows}) != len(rows):
            raise ValueError(f"duplicate record identity within {split}")
        if len({row.sha256 for row in rows}) != len(rows):
            raise ValueError(f"duplicate sequence hash within {split}")
        if any(hashlib.sha256(row.sequence.encode()).hexdigest() != row.sha256 for row in rows):
            raise ValueError(f"sequence SHA-256 mismatch within {split}")
    ids = [{row.record_id for row in groups[split]} for split in sorted(required)]
    hashes = [{row.sha256 for row in groups[split]} for split in sorted(required)]
    if any(ids[i] & ids[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("record identities overlap across splits")
    if any(hashes[i] & hashes[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("sequence hashes overlap across splits")
    return groups


def _endpoint_log_odds(logits: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(logits.shape[0], device=logits.device)
    selected = logits[rows, positions]
    return selected[:, TOKEN_TO_ID["<endpoint1>"]] - selected[:, TOKEN_TO_ID["<endpoint0>"]]


def _confound_matrix(records: Sequence[GrammarRecord]) -> np.ndarray:
    """Frozen position, length, k-mer, family and N-terminal adjustment set."""

    return np.column_stack(
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
        )
    ).astype(np.float64)


def _feature_table(
    model: TinyPlantedAutoregressiveLM,
    clt: CLTForTraining,
    cohort: EncodedCohort,
    config: ControlConfig,
) -> tuple[list[dict], int]:
    inputs, _ = _captures(model, cohort)
    with torch.no_grad():
        codes = clt.encode(inputs)[1]
    rows = torch.arange(codes.shape[0], device=codes.device)
    values = codes[rows, cohort.predictor_positions].detach().cpu().numpy()
    labels = np.asarray([row.long_range for row in cohort.records], dtype=np.float64)
    confounds = _confound_matrix(cohort.records)
    full = np.column_stack((confounds, labels))
    direction = model.planted_direction.detach().cpu().numpy()
    endpoint_readout = 2.0 * config.readout_gain * direction
    table: list[dict] = []
    pvalues: list[float] = []
    for feature in range(config.d_clt):
        outcome = values[:, feature]
        reduced_fit = confounds @ np.linalg.lstsq(confounds, outcome, rcond=None)[0]
        coefficients = np.linalg.lstsq(full, outcome, rcond=None)[0]
        residual = outcome - full @ coefficients
        dof = len(outcome) - np.linalg.matrix_rank(full)
        inverse = np.linalg.pinv(full.T @ full)
        variance = float(residual @ residual / dof) if dof > 0 else 0.0
        standard_error = float(np.sqrt(max(variance * inverse[-1, -1], 0.0)))
        coefficient = float(coefficients[-1])
        if standard_error == 0:
            pvalue = 0.0 if coefficient > 0 else 1.0
        else:
            pvalue = float(stats.t.sf(coefficient / standard_error, dof))
        total = float(np.sum((outcome - outcome.mean()) ** 2))
        incremental_r2 = (
            float(np.sum((outcome - reduced_fit) ** 2) - residual @ residual) / total
            if total > 0
            else 0.0
        )
        decoder = clt.W_dec[1][feature, 0].detach().cpu().numpy()
        decoder /= max(float(np.linalg.norm(decoder)), 1e-12)
        cosine = float(decoder @ direction)
        direct_effect = float(decoder @ endpoint_readout)
        activation_rate = float(np.mean(outcome > 0))
        table.append(
            {
                "feature": feature,
                "semantic_coefficient": coefficient,
                "semantic_p_positive": pvalue,
                "semantic_incremental_r2": incremental_r2,
                "mean_activation": float(outcome.mean()),
                "firing_frequency": activation_rate,
                "decoder_norm": float(
                    clt.W_dec[1][feature, 0].detach().norm().cpu()
                ),
                "direct_endpoint_effect_per_dose": direct_effect,
                "ground_truth_direction_cosine": cosine,
            }
        )
        pvalues.append(pvalue)
    adjusted = benjamini_hochberg(pvalues)
    for row, qvalue in zip(table, adjusted):
        row["semantic_q_bh"] = float(qvalue)
        row["eligible"] = bool(
            row["semantic_coefficient"] > 0
            and qvalue < config.selection_alpha
            and row["direct_endpoint_effect_per_dose"] > 0
        )
        row["discovery_score"] = float(
            max(row["semantic_incremental_r2"], 0)
            * max(row["direct_endpoint_effect_per_dose"], 0)
        )
    eligible = [row for row in table if row["eligible"]]
    selected = max(eligible, key=lambda row: (row["discovery_score"], -row["feature"]))
    return table, int(selected["feature"])


def _intervention_effects(
    model: TinyPlantedAutoregressiveLM,
    cohort: EncodedCohort,
    direction: torch.Tensor,
    *,
    layer: int,
    positions: torch.Tensor,
    dose: float,
) -> np.ndarray:
    with torch.no_grad():
        baseline, _, _ = model(cohort.input_ids)
        changed, _, _ = model(
            cohort.input_ids,
            intervention_layer=layer,
            intervention_positions=positions,
            intervention_direction=direction,
            intervention_dose=dose,
        )
    return (
        _endpoint_log_odds(changed, cohort.predictor_positions)
        - _endpoint_log_odds(baseline, cohort.predictor_positions)
    ).detach().cpu().numpy()


def run_trained_positive_control(
    records: Sequence[GrammarRecord],
    *,
    model_seeds: Sequence[int],
    config: ControlConfig = ControlConfig(),
    device: str = "cpu",
    checkpoint_dir: Path | None = None,
) -> dict:
    """Train three or more LMs and TopK CLTs, then score held-out recovery."""

    config.validate()
    if len(model_seeds) < 3 or len(set(model_seeds)) != len(model_seeds):
        raise ValueError("at least three unique model seeds are required")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    groups = _split_records(records)
    cohorts = {split: encode_cohort(rows).to(torch_device) for split, rows in groups.items()}
    split_hashes = {
        split: _canonical_hash([asdict(row) for row in rows])
        for split, rows in sorted(groups.items())
    }
    if len(set(split_hashes.values())) != 3:
        raise RuntimeError("split hashes are not distinct")
    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_root is not None:
        checkpoint_root.mkdir(parents=True, exist_ok=False)

    model_results: list[dict] = []
    feature_rows: list[dict] = []
    intervention_rows: list[dict] = []
    evaluation_signatures: list[np.ndarray] = []
    max_length = max(cohort.input_ids.shape[1] for cohort in cohorts.values())
    for model_seed in model_seeds:
        model = TinyPlantedAutoregressiveLM(model_seed, config, max_length).to(torch_device)
        lm_initial, lm_final = _train_lm(model, cohorts["train"], config)
        lm_evaluation = float(_lm_loss(model, cohorts["evaluation"]).detach())
        clt, clt_quality = _train_clt(
            model, cohorts["train"], config, int(model_seed)
        )
        table, selected_feature = _feature_table(
            model, clt, cohorts["discovery"], config
        )
        for row in table:
            feature_rows.append({"model_seed": int(model_seed), **row})
        true_feature = int(
            max(table, key=lambda row: row["ground_truth_direction_cosine"])["feature"]
        )
        aligned_features = [
            int(row["feature"])
            for row in table
            if row["ground_truth_direction_cosine"] >= 0.70
        ]

        eval_inputs, _ = _captures(model, cohorts["evaluation"])
        with torch.no_grad():
            eval_codes = clt.encode(eval_inputs)[1]
        row_ids = torch.arange(eval_codes.shape[0], device=torch_device)
        signature = eval_codes[
            row_ids, cohorts["evaluation"].predictor_positions, selected_feature
        ].detach().cpu().numpy()
        adjustment = _confound_matrix(cohorts["evaluation"].records)
        evaluation_signatures.append(
            signature - adjustment @ np.linalg.lstsq(adjustment, signature, rcond=None)[0]
        )

        selected_direction = clt.W_dec[1][selected_feature, 0].detach()
        selected_direction = selected_direction / selected_direction.norm().clamp_min(1e-12)
        selected_cosine = float(
            selected_direction @ model.planted_direction.to(selected_direction)
        )
        profiles = np.asarray(
            [
                [row["firing_frequency"], row["mean_activation"], row["decoder_norm"]]
                for row in table
            ]
        )
        target_profile = profiles[selected_feature]
        scale = profiles.std(axis=0)
        scale[scale < 1e-8] = 1.0
        candidates = [
            row for row in table
            if row["feature"] != selected_feature
            and abs(row["ground_truth_direction_cosine"])
            < config.equivalence_margin / (2 * config.readout_gain)
        ]
        if len(candidates) < config.matched_controls:
            raise RuntimeError(
                "insufficient ground-truth-orthogonal features for matched controls: "
                f"required {config.matched_controls}, found {len(candidates)}"
            )
        candidates.sort(
            key=lambda row: (
                float(np.linalg.norm((profiles[row["feature"]] - target_profile) / scale)),
                row["feature"],
            )
        )
        controls = candidates[: config.matched_controls]

        dose_results = []
        for dose in config.doses:
            effects = _intervention_effects(
                model,
                cohorts["evaluation"],
                selected_direction,
                layer=1,
                positions=cohorts["evaluation"].predictor_positions,
                dose=dose,
            )
            expected = 2.0 * config.readout_gain * selected_cosine * dose
            interval = mean_interval(effects)
            dose_results.append(
                {
                    "dose": float(dose),
                    "known_effect": expected,
                    "observed_effect": interval,
                    "effect_recovery_ratio": float(interval["mean"] / expected),
                }
            )
            intervention_rows.extend(
                {
                    "model_seed": int(model_seed),
                    "record_id": record.record_id,
                    "arm": "selected_target_site",
                    "feature": selected_feature,
                    "layer": 1,
                    "dose": float(dose),
                    "log_odds_difference": float(effect),
                }
                for record, effect in zip(cohorts["evaluation"].records, effects)
            )

        wrong_layer = _intervention_effects(
            model,
            cohorts["evaluation"],
            selected_direction,
            layer=0,
            positions=cohorts["evaluation"].predictor_positions,
            dose=1.0,
        )
        wrong_positions = torch.clamp(cohorts["evaluation"].predictor_positions - 1, min=0)
        wrong_position = _intervention_effects(
            model,
            cohorts["evaluation"],
            selected_direction,
            layer=1,
            positions=wrong_positions,
            dose=1.0,
        )
        site_controls = {
            "wrong_layer": tost_paired(wrong_layer, config.equivalence_margin),
            "wrong_position": tost_paired(wrong_position, config.equivalence_margin),
        }
        for arm, effects, layer in (
            ("selected_wrong_layer", wrong_layer, 0),
            ("selected_wrong_position", wrong_position, 1),
        ):
            intervention_rows.extend(
                {
                    "model_seed": int(model_seed),
                    "record_id": record.record_id,
                    "arm": arm,
                    "feature": selected_feature,
                    "layer": layer,
                    "dose": 1.0,
                    "log_odds_difference": float(effect),
                }
                for record, effect in zip(cohorts["evaluation"].records, effects)
            )
        control_results = []
        for control in controls:
            feature = int(control["feature"])
            direction = clt.W_dec[1][feature, 0].detach()
            direction = direction / direction.norm().clamp_min(1e-12)
            effects = _intervention_effects(
                model,
                cohorts["evaluation"],
                direction,
                layer=1,
                positions=cohorts["evaluation"].predictor_positions,
                dose=1.0,
            )
            match_distance = float(np.linalg.norm((profiles[feature] - target_profile) / scale))
            control_results.append(
                {
                    "feature": feature,
                    "standardized_match_distance": match_distance,
                    "ground_truth_direction_cosine": control[
                        "ground_truth_direction_cosine"
                    ],
                    "equivalence": tost_paired(effects, config.equivalence_margin),
                }
            )
            intervention_rows.extend(
                {
                    "model_seed": int(model_seed),
                    "record_id": record.record_id,
                    "arm": "matched_dictionary_control",
                    "feature": feature,
                    "layer": 1,
                    "dose": 1.0,
                    "log_odds_difference": float(effect),
                }
                for record, effect in zip(cohorts["evaluation"].records, effects)
            )

        with torch.no_grad():
            logits, _, _ = model(cohorts["evaluation"].input_ids)
            odds = _endpoint_log_odds(logits, cohorts["evaluation"].predictor_positions)
            predicted = odds > 0
            truth = torch.tensor(
                [row.long_range for row in cohorts["evaluation"].records],
                device=torch_device,
            )
            endpoint_accuracy = float((predicted == truth).float().mean())
        selected_row = table[selected_feature]
        recovered = selected_feature in aligned_features
        model_hash = _state_hash(model)
        clt_hash = _state_hash(clt)
        checkpoint = None
        if checkpoint_root is not None:
            checkpoint_path = checkpoint_root / f"model_seed_{int(model_seed)}.pt"
            torch.save(
                {
                    "schema_version": "r2-trained-planted-checkpoint-v1",
                    "model_seed": int(model_seed),
                    "config": asdict(config),
                    "model_state": {
                        key: value.detach().cpu() for key, value in model.state_dict().items()
                    },
                    "clt_state": {
                        key: value.detach().cpu() for key, value in clt.state_dict().items()
                    },
                },
                checkpoint_path,
            )
            checkpoint = {
                "path": f"{checkpoint_root.name}/{checkpoint_path.name}",
                "sha256": sha256_file(checkpoint_path),
            }
        model_results.append(
            {
                "model_seed": int(model_seed),
                "model_state_sha256": model_hash,
                "clt_state_sha256": clt_hash,
                "checkpoint": checkpoint,
                "lm_training": {
                    "steps": config.lm_steps,
                    "initial_loss": lm_initial,
                    "final_loss": lm_final,
                    "evaluation_loss": lm_evaluation,
                    "loss_improved": bool(lm_final < lm_initial),
                    "evaluation_endpoint_accuracy": endpoint_accuracy,
                },
                "clt_training": {"steps": config.clt_steps, **clt_quality},
                "planted_layer": 1,
                "true_dictionary_feature": true_feature,
                "ground_truth_aligned_features_cosine_ge_0_70": aligned_features,
                "selected_feature": selected_feature,
                "selected_ground_truth_cosine": selected_cosine,
                "semantic_specificity": {
                    "partial_p_positive": selected_row["semantic_p_positive"],
                    "q_bh": selected_row["semantic_q_bh"],
                    "incremental_r2_beyond_position_length_kmer": selected_row[
                        "semantic_incremental_r2"
                    ],
                },
                "sensitivity": float(recovered),
                "specificity": (
                    1.0
                    if recovered
                    else float(
                        max(config.d_clt - len(aligned_features) - 1, 0)
                        / max(config.d_clt - len(aligned_features), 1)
                    )
                ),
                "false_discovery_rate": 0.0 if recovered else 1.0,
                "dose_sweep": dose_results,
                "site_controls": site_controls,
                "matched_controls": control_results,
                "path_localized": bool(
                    recovered
                    and all(row["equivalence"]["equivalent"] for row in control_results)
                    and all(row["equivalent"] for row in site_controls.values())
                ),
            }
        )

    pairwise_alignment = []
    for left in range(len(model_seeds)):
        for right in range(left + 1, len(model_seeds)):
            correlation = stats.spearmanr(
                evaluation_signatures[left], evaluation_signatures[right]
            ).statistic
            pairwise_alignment.append(
                {
                    "model_seed_a": int(model_seeds[left]),
                    "model_seed_b": int(model_seeds[right]),
                    "heldout_confound_adjusted_activation_spearman": float(correlation),
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
    passed = bool(
        sensitivity >= 0.8
        and specificity >= 0.95
        and fdr <= 0.1
        and all(row["path_localized"] for row in model_results)
        and all(row["lm_training"]["loss_improved"] for row in model_results)
        and all(row["lm_training"]["evaluation_endpoint_accuracy"] >= 0.8 for row in model_results)
        and all(row["clt_training"]["fvu_mean"] <= 0.5 for row in model_results)
        and all(row["clt_training"]["fvu_per_layer"][1] <= 0.5 for row in model_results)
        and all(0.8 <= value <= 1.2 for value in recovery)
        and all(
            row["heldout_confound_adjusted_activation_spearman"] >= 0.8
            for row in pairwise_alignment
        )
    )
    return {
        "schema_version": "r2-trained-planted-autoregressive-control-v1",
        "scope": (
            "Trained synthetic autoregressive sensitivity control only; a pass "
            "does not establish causality in any pretrained protein model."
        ),
        "architecture": (
            "two-layer causal GRU language model plus a rotated fixed adapter; "
            "repository CLTForTraining with per-token ReLU-TopK codes"
        ),
        "config": asdict(config),
        "device": str(torch_device),
        "split_counts": {split: len(rows) for split, rows in sorted(groups.items())},
        "split_sha256": split_hashes,
        "vocabulary_sha256": _canonical_hash(TOKENS),
        "frozen_gates": {
            "minimum_model_seeds": 3,
            "sensitivity_min": 0.8,
            "specificity_min": 0.95,
            "false_discovery_rate_max": 0.1,
            "endpoint_accuracy_min": 0.8,
            "clt_mean_and_planted_layer_fvu_max": 0.5,
            "activation_alignment_spearman_min": 0.8,
            "effect_recovery_ratio": [0.8, 1.2],
            "equivalence_margin_log_odds": config.equivalence_margin,
        },
        "aggregate": {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "false_discovery_rate": fdr,
            "mean_effect_recovery_ratio": float(np.mean(recovery)),
            "all_paths_localized": bool(all(row["path_localized"] for row in model_results)),
            "posthoc_development_smoke_passed": passed,
        },
        "cross_model_alignment": pairwise_alignment,
        "models": model_results,
        "feature_discovery_rows": feature_rows,
        "intervention_rows": intervention_rows,
    }
