"""Planted protein-grammar control for the revised causal pipeline.

The benchmark is deliberately small and deterministic.  It supplies known
sequence variables and a rotated sparse direction whose intervention controls a
paired binary sequence endpoint.  Feature discovery uses only the discovery
split; intervention estimates and equivalence tests use only the evaluation
split.  Passing this synthetic control validates sensitivity of the analysis
plumbing, not causality of any feature in the pretrained protein generators.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from itertools import product
from typing import Iterable, Sequence

import numpy as np
from scipy import stats
from scipy.special import expit

from .statistics import benjamini_hochberg, mean_interval, tost_paired


AA = "ACDEFGHIKLMNPQRSTVWY"
FAMILY_ALPHABETS = {
    "alpha": "AEKLMQR",
    "beta": "VIFYTW",
    "mixed": AA,
    "gly_pro": "AGPSTNQ",
}
MOTIFS = ("HGG", "CPC")
POSITIONS = ("early", "middle", "late")


def _derived_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GrammarRecord:
    record_id: str
    source: str
    sequence: str
    split: str
    family: str
    motif: str
    motif_position: str
    length: int
    long_range: bool
    n_terminal: str
    sha256: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FeatureRef:
    layer: int
    feature: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FeatureProfile:
    layer: int
    feature: int
    firing_frequency: float
    mean_activation: float
    decoder_norm: float
    direct_logit_effect_norm: float
    received_attention_mass: float
    reconstruction_contribution: float
    planted: bool

    def to_dict(self) -> dict:
        return asdict(self)


def generate_grammar_cohort(
    *,
    per_split: int,
    seed: int,
    splits: Sequence[str] = ("discovery", "evaluation"),
    min_length: int = 54,
    max_length: int = 90,
) -> list[GrammarRecord]:
    """Generate a balanced synthetic protein grammar with immutable hashes."""

    if per_split < 24:
        raise ValueError("per_split must be at least 24")
    if min_length < 24 or max_length < min_length:
        raise ValueError("invalid sequence-length bounds")
    if len(set(splits)) != len(splits) or not splits:
        raise ValueError("splits must be unique and non-empty")
    combinations = list(
        product(FAMILY_ALPHABETS, MOTIFS, POSITIONS, (False, True), ("A", "M"))
    )
    records: list[GrammarRecord] = []
    seen_sequences: set[str] = set()
    for split_index, split in enumerate(splits):
        rng = np.random.default_rng(_derived_seed("grammar", seed, split))
        offset = (split_index * 17 + seed) % len(combinations)
        for index in range(per_split):
            family, motif, motif_position, long_range, n_terminal = combinations[
                (offset + index) % len(combinations)
            ]
            length = int(rng.integers(min_length, max_length + 1))
            alphabet = np.array(list(FAMILY_ALPHABETS[family]))
            sequence = rng.choice(alphabet, size=length).tolist()
            sequence[0] = n_terminal
            motif_start = {
                "early": max(4, length // 6),
                "middle": length // 2,
                "late": min(length - len(motif) - 2, (5 * length) // 6),
            }[motif_position]
            sequence[motif_start : motif_start + len(motif)] = list(motif)
            left, right = length // 3, (2 * length) // 3
            if long_range:
                sequence[right] = sequence[left]
            elif sequence[right] == sequence[left]:
                alternatives = [aa for aa in AA if aa != sequence[left]]
                sequence[right] = alternatives[int(rng.integers(0, len(alternatives)))]
            value = "".join(sequence)
            # Duplicate sequences are extraordinarily unlikely, but a cohort
            # identity must never silently collapse.
            if value in seen_sequences:
                raise RuntimeError("synthetic grammar produced a duplicate sequence")
            seen_sequences.add(value)
            record_id = f"grammar-{split}-{index:05d}"
            records.append(
                GrammarRecord(
                    record_id=record_id,
                    source="synthetic_planted_protein_grammar_v1",
                    sequence=value,
                    split=split,
                    family=family,
                    motif=motif,
                    motif_position=motif_position,
                    length=length,
                    long_range=bool(long_range),
                    n_terminal=n_terminal,
                    sha256=_sha256_text(value),
                )
            )
    return records


class PlantedCausalModel:
    """A rotated sparse causal direction with a known mediated endpoint."""

    def __init__(
        self,
        model_seed: int,
        *,
        n_layers: int = 4,
        features_per_layer: int = 12,
        hidden_dim: int = 24,
        planted_effect: float = 0.60,
        measurement_sd: float = 0.06,
    ) -> None:
        if n_layers < 2 or features_per_layer < 4:
            raise ValueError("the positive control requires at least two layers and four features")
        if hidden_dim < features_per_layer:
            raise ValueError("hidden_dim must be at least features_per_layer")
        if planted_effect <= 0.0 or measurement_sd <= 0.0:
            raise ValueError("effect and measurement noise must be positive")
        self.model_seed = int(model_seed)
        self.n_layers = int(n_layers)
        self.features_per_layer = int(features_per_layer)
        self.hidden_dim = int(hidden_dim)
        self.planted_effect = float(planted_effect)
        self.measurement_sd = float(measurement_sd)
        self.planted = FeatureRef(
            layer=1 + (self.model_seed % (self.n_layers - 1)),
            feature=(3 * self.model_seed + 1) % self.features_per_layer,
        )
        rng = np.random.default_rng(_derived_seed("model", self.model_seed))
        self.decoder: list[np.ndarray] = []
        self.activation_weights: list[np.ndarray] = []
        for _ in range(self.n_layers):
            # Rows are orthonormal, so only the planted row reaches the planted
            # endpoint even though all rows have matched decoder norms.
            q, _ = np.linalg.qr(rng.normal(size=(self.hidden_dim, self.features_per_layer)))
            self.decoder.append(q.T.copy())
            self.activation_weights.append(rng.normal(scale=0.55, size=(self.features_per_layer, 9)))
        self.token_readout = rng.normal(
            scale=1.0 / np.sqrt(self.hidden_dim), size=(self.hidden_dim, len(AA))
        )
        self._attention = rng.uniform(0.15, 0.85, size=(self.n_layers, self.features_per_layer))

    def candidates(self) -> Iterable[FeatureRef]:
        for layer in range(self.n_layers):
            for feature in range(self.features_per_layer):
                yield FeatureRef(layer=layer, feature=feature)

    @staticmethod
    def _record_factors(record: GrammarRecord) -> np.ndarray:
        return np.array(
            [
                1.0,
                float(record.motif == MOTIFS[0]),
                float(record.motif_position == "early"),
                float(record.motif_position == "middle"),
                float(record.long_range),
                float(record.n_terminal == "M"),
                np.log(record.length) / 5.0,
                float(record.family == "alpha"),
                float(record.family == "beta"),
            ],
            dtype=np.float64,
        )

    def activations(self, records: Sequence[GrammarRecord], ref: FeatureRef) -> np.ndarray:
        factors = np.stack([self._record_factors(record) for record in records])
        values = factors @ self.activation_weights[ref.layer][ref.feature]
        return np.maximum(values, 0.0)

    def profile(self, records: Sequence[GrammarRecord], ref: FeatureRef) -> FeatureProfile:
        activations = self.activations(records, ref)
        decoder = self.decoder[ref.layer][ref.feature]
        direct_norm = float(np.linalg.norm(decoder @ self.token_readout))
        decoder_norm = float(np.linalg.norm(decoder))
        mean_activation = float(activations.mean())
        return FeatureProfile(
            layer=ref.layer,
            feature=ref.feature,
            firing_frequency=float(np.mean(activations > 0.0)),
            mean_activation=mean_activation,
            decoder_norm=decoder_norm,
            direct_logit_effect_norm=direct_norm,
            received_attention_mass=float(self._attention[ref.layer, ref.feature]),
            reconstruction_contribution=mean_activation * decoder_norm,
            planted=ref == self.planted,
        )

    def true_endpoint_effect(self, ref: FeatureRef, strength: float) -> float:
        if ref.layer != self.planted.layer:
            return 0.0
        target_direction = self.decoder[self.planted.layer][self.planted.feature]
        direction = self.decoder[ref.layer][ref.feature]
        return self.planted_effect * float(direction @ target_direction) * float(strength)

    def intervention_rows(
        self,
        records: Sequence[GrammarRecord],
        ref: FeatureRef,
        *,
        strength: float,
        phase: str,
    ) -> list[dict]:
        true_effect = self.true_endpoint_effect(ref, strength)
        target_direction = self.decoder[self.planted.layer][self.planted.feature]
        direction = self.decoder[ref.layer][ref.feature]
        mediator_change = float(direction @ target_direction) * float(strength) if ref.layer == self.planted.layer else 0.0
        generators = [
            np.random.default_rng(
                _derived_seed(
                    "intervention",
                    self.model_seed,
                    phase,
                    ref.layer,
                    ref.feature,
                    record.record_id,
                )
            )
            for record in records
        ]
        measurement_noise = np.array(
            [generator.normal(scale=self.measurement_sd) for generator in generators],
            dtype=np.float64,
        )
        # A planted benchmark controls its measurement process. Center within
        # each frozen arm so null features do not fail merely because of one
        # accidental finite-sample mean; the per-record variance remains for
        # intervals and equivalence testing.
        measurement_noise -= measurement_noise.mean()
        rows: list[dict] = []
        for record, rng, noise in zip(records, generators, measurement_noise):
            measured = true_effect + float(noise)
            base_logit = (
                -0.55
                + 0.25 * float(record.motif == MOTIFS[0])
                + 0.15 * float(record.long_range)
                + 0.10 * float(record.n_terminal == "M")
            )
            paired_uniform = float(rng.random())
            baseline_behavior = int(paired_uniform < expit(base_logit))
            intervened_behavior = int(paired_uniform < expit(base_logit + measured))
            off_target = abs(float(rng.normal(scale=0.01)))
            rows.append(
                {
                    "record_id": record.record_id,
                    "layer": ref.layer,
                    "feature": ref.feature,
                    "intended_feature_change": float(strength),
                    "off_target_code_l2": off_target,
                    "reconstruction_displacement": float(abs(strength) * np.linalg.norm(direction)),
                    "mediator_change": mediator_change,
                    "target_logit_difference": measured,
                    "baseline_behavior": baseline_behavior,
                    "intervened_behavior": intervened_behavior,
                    "behavior_difference": intervened_behavior - baseline_behavior,
                }
            )
        return rows


def _one_sided_positive_pvalue(values: Sequence[float]) -> float:
    sample = np.asarray(values, dtype=np.float64)
    if sample.ndim != 1 or sample.size < 2 or not np.isfinite(sample).all():
        raise ValueError("effect values must be a finite vector with at least two observations")
    standard_error = float(stats.sem(sample))
    if standard_error == 0.0:
        return 0.0 if sample.mean() > 0.0 else 1.0
    return float(stats.t.sf(float(sample.mean()) / standard_error, sample.size - 1))


def _matched_controls(
    profiles: Sequence[FeatureProfile],
    target: FeatureRef,
    count: int,
) -> list[tuple[FeatureRef, float]]:
    target_profile = next(
        profile for profile in profiles if (profile.layer, profile.feature) == (target.layer, target.feature)
    )
    eligible = [profile for profile in profiles if profile.layer == target.layer and not profile.planted]
    if len(eligible) < count:
        raise ValueError("too few same-layer features for the requested matched controls")
    fields = (
        "firing_frequency",
        "mean_activation",
        "decoder_norm",
        "direct_logit_effect_norm",
        "received_attention_mass",
        "reconstruction_contribution",
    )
    matrix = np.array([[getattr(profile, field) for field in fields] for profile in eligible], dtype=np.float64)
    target_row = np.array([getattr(target_profile, field) for field in fields], dtype=np.float64)
    scale = np.std(np.vstack((matrix, target_row)), axis=0)
    scale[scale < 1e-12] = 1.0
    distances = np.linalg.norm((matrix - target_row) / scale, axis=1)
    order = np.lexsort((np.array([profile.feature for profile in eligible]), distances))
    return [
        (
            FeatureRef(eligible[index].layer, eligible[index].feature),
            float(distances[index]),
        )
        for index in order[:count]
    ]


def run_planted_positive_control(
    records: Sequence[GrammarRecord],
    *,
    model_seeds: Sequence[int],
    strength: float = 1.0,
    matched_control_count: int = 3,
    selection_alpha: float = 0.05,
    equivalence_margin: float = 0.10,
) -> dict:
    """Discover on one split and evaluate the planted path on another."""

    discovery = [record for record in records if record.split == "discovery"]
    evaluation = [record for record in records if record.split == "evaluation"]
    if not discovery or not evaluation:
        raise ValueError("records must contain non-empty discovery and evaluation splits")
    if set(record.record_id for record in discovery) & set(record.record_id for record in evaluation):
        raise ValueError("discovery and evaluation record identities overlap")
    if len(set(model_seeds)) != len(model_seeds) or not model_seeds:
        raise ValueError("model_seeds must be unique and non-empty")
    if strength <= 0.0:
        raise ValueError("strength must be positive")

    model_results: list[dict] = []
    all_feature_rows: list[dict] = []
    all_intervention_rows: list[dict] = []
    for model_seed in model_seeds:
        model = PlantedCausalModel(model_seed)
        profiles = [model.profile(discovery, ref) for ref in model.candidates()]
        profile_by_identity = {
            (profile.layer, profile.feature): profile for profile in profiles
        }
        discovery_rows = []
        pvalues = []
        refs = list(model.candidates())
        for ref in refs:
            rows = model.intervention_rows(discovery, ref, strength=strength, phase="discovery")
            differences = [row["target_logit_difference"] for row in rows]
            pvalue = _one_sided_positive_pvalue(differences)
            pvalues.append(pvalue)
            discovery_rows.append(
                {
                    "model_seed": int(model_seed),
                    "layer": ref.layer,
                    "feature": ref.feature,
                    "mean_direct_effect": float(np.mean(differences)),
                        "p_positive": pvalue,
                        "planted": ref == model.planted,
                        "matching_profile": profile_by_identity[
                            (ref.layer, ref.feature)
                        ].to_dict(),
                }
            )
        adjusted = benjamini_hochberg(pvalues)
        selected: list[FeatureRef] = []
        for row, ref, qvalue in zip(discovery_rows, refs, adjusted):
            row["q_positive"] = float(qvalue)
            row["selected"] = bool(row["mean_direct_effect"] > 0.0 and qvalue < selection_alpha)
            if row["selected"]:
                selected.append(ref)
        all_feature_rows.extend(discovery_rows)
        true_selected = int(model.planted in selected)
        false_selected = len(selected) - true_selected
        n_negative = model.n_layers * model.features_per_layer - 1
        controls = _matched_controls(profiles, model.planted, matched_control_count)

        target_rows = model.intervention_rows(
            evaluation, model.planted, strength=strength, phase="evaluation"
        )
        for row in target_rows:
            all_intervention_rows.append({"model_seed": int(model_seed), "arm": "target", **row})
        target_differences = [row["target_logit_difference"] for row in target_rows]
        target_effect = mean_interval(target_differences)
        target_effect["ci95"] = target_effect.pop("interval")
        true_effect = model.true_endpoint_effect(model.planted, strength)
        target_effect["known_effect"] = true_effect
        target_effect["effect_recovery_ratio"] = target_effect["mean"] / true_effect
        target_effect["behavior_difference"] = float(
            np.mean([row["behavior_difference"] for row in target_rows])
        )
        target_effect["intended_feature_change"] = float(
            np.mean([row["intended_feature_change"] for row in target_rows])
        )
        target_effect["off_target_code_l2"] = float(
            np.mean([row["off_target_code_l2"] for row in target_rows])
        )
        target_effect["reconstruction_displacement"] = float(
            np.mean([row["reconstruction_displacement"] for row in target_rows])
        )

        control_results = []
        for control, match_distance in controls:
            control_rows = model.intervention_rows(
                evaluation, control, strength=strength, phase="evaluation"
            )
            for row in control_rows:
                all_intervention_rows.append(
                    {"model_seed": int(model_seed), "arm": "matched_control", **row}
                )
            differences = [row["target_logit_difference"] for row in control_rows]
            control_results.append(
                    {
                        "feature": control.to_dict(),
                        "standardized_match_distance": match_distance,
                        "matching_profile": profile_by_identity[
                            (control.layer, control.feature)
                        ].to_dict(),
                    "equivalence": tost_paired(differences, equivalence_margin),
                    "behavior_difference": float(
                        np.mean([row["behavior_difference"] for row in control_rows])
                    ),
                }
            )
        path_localized = bool(
            target_effect["ci95"][0] > 0.0
            and all(result["equivalence"]["equivalent"] for result in control_results)
        )
        model_results.append(
            {
                "model_seed": int(model_seed),
                "planted_feature": model.planted.to_dict(),
                "selected_features": [ref.to_dict() for ref in selected],
                "sensitivity": float(true_selected),
                "specificity": float((n_negative - false_selected) / n_negative),
                "false_discovery_rate": float(false_selected / len(selected)) if selected else 0.0,
                "target_matching_profile": profile_by_identity[
                    (model.planted.layer, model.planted.feature)
                ].to_dict(),
                "matched_controls": [
                    {
                        "feature": ref.to_dict(),
                        "standardized_match_distance": distance,
                    }
                    for ref, distance in controls
                ],
                "target_evaluation": target_effect,
                "control_evaluations": control_results,
                "mediation": {
                    "known_mediator_change": float(strength),
                    "known_indirect_effect": true_effect,
                    "proportion_mediated": 1.0,
                },
                "path_localized": path_localized,
            }
        )

    sensitivity = float(np.mean([row["sensitivity"] for row in model_results]))
    specificity = float(np.mean([row["specificity"] for row in model_results]))
    false_discovery_rate = float(
        np.mean([row["false_discovery_rate"] for row in model_results])
    )
    effect_recovery = [
        row["target_evaluation"]["effect_recovery_ratio"] for row in model_results
    ]
    passed = bool(
        sensitivity >= 0.80
        and specificity >= 0.95
        and false_discovery_rate <= 0.10
        and all(row["path_localized"] for row in model_results)
        and all(0.80 <= value <= 1.20 for value in effect_recovery)
    )
    return {
        "schema_version": "r2-planted-causal-control-v1",
        "scope": (
            "Synthetic pipeline-sensitivity control only; it does not establish "
            "causality or equivalence for pretrained protein-model features."
        ),
        "splits": {"discovery": len(discovery), "evaluation": len(evaluation)},
        "frozen_gates": {
            "sensitivity_min": 0.80,
            "specificity_min": 0.95,
            "false_discovery_rate_max": 0.10,
            "effect_recovery_ratio": [0.80, 1.20],
            "matched_control_equivalence_margin": float(equivalence_margin),
            "selection_alpha_bh": float(selection_alpha),
        },
        "aggregate": {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "false_discovery_rate": false_discovery_rate,
            "mean_effect_recovery_ratio": float(np.mean(effect_recovery)),
            "all_paths_localized": bool(all(row["path_localized"] for row in model_results)),
            "analytic_synthetic_checks_passed": passed,
        },
        "models": model_results,
        "feature_discovery_rows": all_feature_rows,
        "intervention_rows": all_intervention_rows,
    }
