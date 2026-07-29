"""Paired N-terminal counterfactuals with causal-opportunity normalization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from .statistics import tost_paired


AA = set("ACDEFGHIKLMNPQRSTVWY")
CONDITIONS = (
    "natural_mxx",
    "m_to_a",
    "internal_mxx_insertion",
    "artificial_truncation",
)
BOS_POLICIES = ("native", "removed")
CONTRASTS = {
    "initiator_m_to_a": (("m_to_a", "native"), ("natural_mxx", "native")),
    "motif_internal_vs_n_terminal": (
        ("internal_mxx_insertion", "native"),
        ("natural_mxx", "native"),
    ),
    "artificial_start_vs_internal": (
        ("artificial_truncation", "native"),
        ("internal_mxx_insertion", "native"),
    ),
    "bos_native_vs_removed": (("natural_mxx", "native"), ("natural_mxx", "removed")),
}
FEATURE_METRICS = ("feature_activation_pre", "normalized_received_attention")
ATTENTION_PATH_METRICS = (
    "suffix_nll_increase_key_masked",
    "suffix_observed_token_logit_change_key_masked",
)
INFERENCE_METRICS = (*FEATURE_METRICS, *ATTENTION_PATH_METRICS)


def _derived_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    """Hash an exact, ordered tokenization using canonical JSON."""

    normalized = []
    for token_id in token_ids:
        if isinstance(token_id, bool) or not isinstance(token_id, (int, np.integer)):
            raise ValueError("token_ids must contain integers, not booleans or floats")
        if int(token_id) < 0:
            raise ValueError("token_ids must be non-negative")
        normalized.append(int(token_id))
    if not normalized:
        raise ValueError("token_ids must be non-empty")
    payload = json.dumps(normalized, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CounterfactualVariant:
    protein_id: str
    condition: str
    sequence: str
    focal_start: int
    original_length: int
    sequence_length: int
    normalized_focal_position: float
    sha256: str

    def to_dict(self) -> dict:
        return asdict(self)


def received_attention_by_key(
    attention: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Normalize each key's received mass by eligible causal queries.

    ``attention[..., query, key]`` may include arbitrary leading dimensions
    (for example batch, layer and head). ``valid_mask`` is a one-dimensional
    token mask shared by those leading dimensions. Self-attention is eligible,
    so a valid key at index ``k`` has valid queries ``q >= k``.
    """

    values = np.asarray(attention, dtype=np.float64)
    if values.ndim < 2 or values.shape[-2] != values.shape[-1]:
        raise ValueError("attention must end in a square [query, key] matrix")
    if not np.isfinite(values).all() or np.any(values < -1e-12):
        raise ValueError("attention must be finite and non-negative")
    length = values.shape[-1]
    valid = np.ones(length, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != (length,):
        raise ValueError("valid_mask must have one entry per token")
    query = np.arange(length)[:, None]
    key = np.arange(length)[None, :]
    eligible = (query >= key) & valid[:, None] & valid[None, :]
    raw = np.sum(values * eligible, axis=-2)
    counts = eligible.sum(axis=0).astype(np.int64)
    normalized = np.zeros_like(raw, dtype=np.float64)
    np.divide(raw, counts, out=normalized, where=counts > 0)
    return {
        "raw_received_attention": raw,
        "eligible_query_count": counts,
        "normalized_received_attention": normalized,
    }


def _clean_natural_sequence(sequence: str) -> str:
    cleaned = "".join(str(sequence).upper().split())
    if len(cleaned) < 18 or any(residue not in AA for residue in cleaned):
        raise ValueError("natural sequences must contain at least 18 canonical amino acids")
    if cleaned[0] != "M":
        raise ValueError("the N-terminal counterfactual cohort requires natural MXX starts")
    return cleaned


def build_counterfactual_variants(
    records: Sequence[Mapping[str, object]],
    *,
    internal_fraction: float = 0.55,
) -> list[CounterfactualVariant]:
    """Build four paired sequence conditions for every natural MXX protein."""

    if not 0.25 <= internal_fraction <= 0.75:
        raise ValueError("internal_fraction must lie in [0.25, 0.75]")
    variants: list[CounterfactualVariant] = []
    seen_ids: set[str] = set()
    for record in records:
        protein_id = str(record.get("protein_id") or record.get("id") or "")
        if not protein_id or protein_id in seen_ids:
            raise ValueError("protein identifiers must be non-empty and unique")
        seen_ids.add(protein_id)
        natural = _clean_natural_sequence(str(record["sequence"]))
        motif = natural[:3]
        insertion = int(round((len(natural) - 3) * internal_fraction))
        insertion = min(max(insertion, 4), len(natural) - 4)
        internal = natural[:insertion] + motif + natural[insertion:]
        sequences = {
            "natural_mxx": (natural, 0),
            "m_to_a": ("A" + natural[1:], 0),
            "internal_mxx_insertion": (internal, insertion),
            "artificial_truncation": (internal[insertion:], 0),
        }
        for condition in CONDITIONS:
            sequence, focal_start = sequences[condition]
            variants.append(
                CounterfactualVariant(
                    protein_id=protein_id,
                    condition=condition,
                    sequence=sequence,
                    focal_start=focal_start,
                    original_length=len(natural),
                    sequence_length=len(sequence),
                    normalized_focal_position=focal_start / max(len(sequence) - 1, 1),
                    sha256=_sha256_text(sequence),
                )
            )
    return variants


def validate_disjoint_cohorts(
    evaluation_records: Sequence[Mapping[str, object]],
    discovery_records: Sequence[Mapping[str, object]],
) -> dict:
    """Fail if held-out evaluation proteins overlap discovery by ID or sequence."""

    def identities(records: Sequence[Mapping[str, object]], label: str) -> tuple[set[str], set[str]]:
        ids: set[str] = set()
        sequence_hashes: set[str] = set()
        for source in records:
            protein_id = str(source.get("protein_id") or source.get("id") or "").strip()
            sequence = "".join(str(source.get("sequence", "")).upper().split())
            if not protein_id or protein_id in ids:
                raise ValueError(f"{label} protein identifiers must be non-empty and unique")
            if not sequence or any(residue not in AA for residue in sequence):
                raise ValueError(f"{label} sequences must be non-empty canonical amino acids")
            digest = _sha256_text(sequence)
            if digest in sequence_hashes:
                raise ValueError(f"{label} contains duplicate protein sequences")
            ids.add(protein_id)
            sequence_hashes.add(digest)
        if not ids:
            raise ValueError(f"{label} cohort must be non-empty")
        return ids, sequence_hashes

    evaluation_ids, evaluation_sequences = identities(evaluation_records, "evaluation")
    discovery_ids, discovery_sequences = identities(discovery_records, "discovery")
    overlapping_ids = evaluation_ids & discovery_ids
    overlapping_sequences = evaluation_sequences & discovery_sequences
    if overlapping_ids or overlapping_sequences:
        raise ValueError(
            "discovery/evaluation cohort overlap detected: "
            f"protein_ids={sorted(overlapping_ids)}, sequence_hashes={sorted(overlapping_sequences)}"
        )
    return {
        "discovery_evaluation_disjoint": True,
        "n_evaluation_proteins": len(evaluation_ids),
        "n_discovery_proteins": len(discovery_ids),
        "overlap_by_id": 0,
        "overlap_by_exact_sequence": 0,
    }


def validate_equivalence_spec(spec: Mapping[str, object]) -> dict:
    """Freeze P0-5 equivalence margins and the multiplicity family."""

    if not isinstance(spec, Mapping):
        raise ValueError("equivalence specification must be a JSON object")
    alpha = float(spec.get("alpha", 0.0))
    if not 0.0 < alpha < 0.5:
        raise ValueError("equivalence alpha must lie strictly between zero and one half")
    margins = spec.get("margins")
    if not isinstance(margins, Mapping) or set(margins) != set(INFERENCE_METRICS):
        raise ValueError(f"equivalence margins must be specified for {list(INFERENCE_METRICS)}")
    normalized_margins = {metric: float(margins[metric]) for metric in INFERENCE_METRICS}
    if any(not np.isfinite(value) or value <= 0.0 for value in normalized_margins.values()):
        raise ValueError("all equivalence margins must be finite and positive")
    multiplicity = str(
        spec.get("multiplicity", "holm_all_feature_control_and_protein_pair_did_cells")
    )
    if multiplicity != "holm_all_feature_control_and_protein_pair_did_cells":
        raise ValueError(
            "P0-5 multiplicity must be "
            "holm_all_feature_control_and_protein_pair_did_cells"
        )
    return {"alpha": alpha, "margins": normalized_margins, "multiplicity": multiplicity}


def normalize_measurement_rows(
    rows: Sequence[Mapping[str, object]],
    variants: Sequence[CounterfactualVariant | Mapping[str, object]],
) -> list[dict]:
    """Validate measurements and join them to exact variants/tokenizations."""

    variant_lookup: dict[tuple[str, str], dict] = {}
    for source in variants:
        variant = source.to_dict() if isinstance(source, CounterfactualVariant) else dict(source)
        key = (str(variant.get("protein_id", "")), str(variant.get("condition", "")))
        if not key[0] or key[1] not in CONDITIONS or key in variant_lookup:
            raise ValueError("variants require unique protein_id x condition identities")
        sequence = str(variant.get("sequence", ""))
        digest = str(variant.get("sha256", ""))
        if digest != _sha256_text(sequence):
            raise ValueError(f"variant {key} has an invalid sequence SHA-256")
        variant_lookup[key] = variant
    variant_proteins = {key[0] for key in variant_lookup}
    if not variant_proteins or len(variant_lookup) != len(variant_proteins) * len(CONDITIONS):
        raise ValueError("variants must contain the complete four-condition protein cohort")

    required = {
        "protein_id",
        "condition",
        "bos_policy",
        "model",
        "layer",
        "feature",
        "feature_role",
        "feature_match_id",
        "feature_activation_pre",
        "received_attention_raw",
        "eligible_query_count",
        "normalized_focal_position",
        "sequence_length",
        "firing_frequency",
        "input_norm",
        "variant_sha256",
        "tokenizer_revision",
        "token_ids",
        "token_ids_sha256",
        "focal_token_index",
        "protein_pair_id",
        "protein_match_role",
        "matched_protein_id",
        "protein_match_focal_position",
        "protein_match_normalized_position",
        "baseline_suffix_nll",
        "key_masked_suffix_nll",
        "suffix_nll_increase_key_masked",
        "baseline_suffix_observed_token_logit_mean",
        "key_masked_suffix_observed_token_logit_mean",
        "suffix_observed_token_logit_change_key_masked",
        "attention_key_mask_max_abs_strict_suffix",
    }
    normalized: list[dict] = []
    identities: set[tuple] = set()
    tokenizations: dict[tuple, tuple] = {}
    for source in rows:
        missing = required - set(source)
        if missing:
            raise ValueError(f"measurement row is missing fields: {sorted(missing)}")
        row = dict(source)
        if row["condition"] not in CONDITIONS or row["bos_policy"] not in BOS_POLICIES:
            raise ValueError("unknown counterfactual condition or BOS policy")
        if row["feature_role"] not in {"target", "control"}:
            raise ValueError("feature_role must be target or control")
        feature_match_id = str(row["feature_match_id"])
        if len(feature_match_id) != 64 or any(
            character not in "0123456789abcdef" for character in feature_match_id
        ):
            raise ValueError("feature_match_id must be a lowercase SHA-256 digest")
        variant_key = (str(row["protein_id"]), str(row["condition"]))
        if variant_key not in variant_lookup:
            raise ValueError(f"measurement does not join to a frozen variant: {variant_key}")
        variant = variant_lookup[variant_key]
        if str(row["variant_sha256"]) != str(variant["sha256"]):
            raise ValueError(f"measurement variant hash mismatch: {variant_key}")
        identity = (
            str(row["protein_id"]),
            str(row["condition"]),
            str(row["bos_policy"]),
            str(row["model"]),
            int(row["layer"]),
            int(row["feature"]),
        )
        if identity in identities:
            raise ValueError(f"duplicate measurement identity: {identity}")
        identities.add(identity)
        integer_fields = (
            "layer",
            "feature",
            "eligible_query_count",
            "sequence_length",
            "focal_token_index",
            "protein_match_focal_position",
        )
        if any(
            isinstance(row[field], (bool, np.bool_))
            or not isinstance(row[field], (int, np.integer))
            for field in integer_fields
        ):
            raise ValueError(
                "measurement indices, lengths and protein_match_focal_position "
                "must be integers"
            )
        finite_fields = (
            "feature_activation_pre",
            "received_attention_raw",
            "normalized_focal_position",
            "sequence_length",
            "firing_frequency",
            "input_norm",
            "protein_match_focal_position",
            "protein_match_normalized_position",
            "baseline_suffix_nll",
            "key_masked_suffix_nll",
            "suffix_nll_increase_key_masked",
            "baseline_suffix_observed_token_logit_mean",
            "key_masked_suffix_observed_token_logit_mean",
            "suffix_observed_token_logit_change_key_masked",
            "attention_key_mask_max_abs_strict_suffix",
        )
        if not all(np.isfinite(float(row[field])) for field in finite_fields):
            raise ValueError("measurement row contains non-finite values")
        if float(row["received_attention_raw"]) < 0.0:
            raise ValueError("received_attention_raw must be non-negative")
        if (
            float(row["firing_frequency"]) <= 0.0
            or float(row["input_norm"]) <= 0.0
        ):
            raise ValueError("firing_frequency and input_norm must be positive")
        if not 0.0 <= float(row["protein_match_normalized_position"]) <= 1.0:
            raise ValueError("protein_match_normalized_position must lie in [0, 1]")
        count = int(row["eligible_query_count"])
        if count <= 0:
            raise ValueError("eligible_query_count must be positive")
        token_ids = [int(token_id) for token_id in row["token_ids"]]
        token_digest = token_ids_sha256(row["token_ids"])
        if str(row["token_ids_sha256"]) != token_digest:
            raise ValueError(f"measurement tokenization hash mismatch: {variant_key}")
        focal_token_index = int(row["focal_token_index"])
        if not 0 <= focal_token_index < len(token_ids):
            raise ValueError("focal_token_index must index token_ids")
        if count != len(token_ids) - focal_token_index:
            raise ValueError(
                "eligible_query_count must equal valid token count minus focal_token_index"
            )
        pair_id = str(row["protein_pair_id"])
        if len(pair_id) != 64 or any(character not in "0123456789abcdef" for character in pair_id):
            raise ValueError("protein_pair_id must be a lowercase SHA-256 digest")
        if row["protein_match_role"] not in {"target", "control"}:
            raise ValueError("protein_match_role must be target or control")
        matched_protein_id = str(row["matched_protein_id"]).strip()
        if not matched_protein_id or matched_protein_id == str(row["protein_id"]):
            raise ValueError("matched_protein_id must name a distinct protein")
        baseline_nll = float(row["baseline_suffix_nll"])
        key_masked_nll = float(row["key_masked_suffix_nll"])
        nll_effect = float(row["suffix_nll_increase_key_masked"])
        baseline_logit = float(row["baseline_suffix_observed_token_logit_mean"])
        key_masked_logit = float(
            row["key_masked_suffix_observed_token_logit_mean"]
        )
        logit_effect = float(
            row["suffix_observed_token_logit_change_key_masked"]
        )
        if baseline_nll < 0.0 or key_masked_nll < 0.0:
            raise ValueError("strict-suffix NLL values must be non-negative")
        if not np.isclose(
            nll_effect,
            key_masked_nll - baseline_nll,
            rtol=1e-6,
            atol=1e-7,
        ) or not np.isclose(
            logit_effect,
            key_masked_logit - baseline_logit,
            rtol=1e-6,
            atol=1e-7,
        ):
            raise ValueError(
                "focal-key effects must equal key-masked minus baseline endpoints"
            )
        key_mask_max = float(row["attention_key_mask_max_abs_strict_suffix"])
        if not 0.0 <= key_mask_max <= 1e-6:
            raise ValueError("focal-key intervention leaked attention into the strict suffix")
        tokenizer_revision = str(row["tokenizer_revision"]).strip()
        if not tokenizer_revision:
            raise ValueError("tokenizer_revision must be non-empty")
        tokenization_key = (*variant_key, str(row["bos_policy"]), str(row["model"]))
        tokenization = (tokenizer_revision, tuple(token_ids), token_digest, focal_token_index)
        previous = tokenizations.setdefault(tokenization_key, tokenization)
        if previous != tokenization:
            raise ValueError(
                "all features must use the identical tokenization for a protein/condition/BOS/model"
            )
        if int(row["sequence_length"]) != int(variant["sequence_length"]):
            raise ValueError(f"measurement sequence length mismatch: {variant_key}")
        if not np.isclose(
            float(row["normalized_focal_position"]),
            float(variant["normalized_focal_position"]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"measurement focal position mismatch: {variant_key}")
        row["protein_id"] = str(row["protein_id"])
        row["model"] = str(row["model"])
        row["layer"] = int(row["layer"])
        row["feature"] = int(row["feature"])
        row["feature_match_id"] = feature_match_id
        row["feature_activation_pre"] = float(row["feature_activation_pre"])
        row["received_attention_raw"] = float(row["received_attention_raw"])
        row["eligible_query_count"] = count
        row["normalized_received_attention"] = row["received_attention_raw"] / count
        row["normalized_focal_position"] = float(row["normalized_focal_position"])
        row["sequence_length"] = int(row["sequence_length"])
        row["firing_frequency"] = float(row["firing_frequency"])
        row["input_norm"] = float(row["input_norm"])
        row["variant_sha256"] = str(row["variant_sha256"])
        row["tokenizer_revision"] = tokenizer_revision
        row["token_ids"] = token_ids
        row["token_ids_sha256"] = token_digest
        row["focal_token_index"] = focal_token_index
        row["protein_pair_id"] = pair_id
        row["protein_match_role"] = str(row["protein_match_role"])
        row["matched_protein_id"] = matched_protein_id
        row["protein_match_focal_position"] = int(row["protein_match_focal_position"])
        row["protein_match_normalized_position"] = float(
            row["protein_match_normalized_position"]
        )
        for field in (
            "baseline_suffix_nll",
            "key_masked_suffix_nll",
            "suffix_nll_increase_key_masked",
            "baseline_suffix_observed_token_logit_mean",
            "key_masked_suffix_observed_token_logit_mean",
            "suffix_observed_token_logit_change_key_masked",
            "attention_key_mask_max_abs_strict_suffix",
        ):
            row[field] = float(row[field])
        normalized.append(row)
    measured_proteins = {row["protein_id"] for row in normalized}
    if measured_proteins != variant_proteins:
        missing = sorted(variant_proteins - measured_proteins)
        extra = sorted(measured_proteins - variant_proteins)
        raise ValueError(
            f"measurement and frozen-variant protein cohorts differ; missing={missing}, extra={extra}"
        )
    return normalized


def _validate_protein_pair_coverage(
    rows: Sequence[dict],
    variants: Sequence[CounterfactualVariant | Mapping[str, object]],
) -> list[dict]:
    """Require reciprocal target/control pairs and a complete factorial per role."""

    variant_rows = [
        variant.to_dict() if isinstance(variant, CounterfactualVariant) else dict(variant)
        for variant in variants
    ]
    internal_by_protein = {
        str(row["protein_id"]): row
        for row in variant_rows
        if row["condition"] == "internal_mxx_insertion"
    }
    metadata: dict[str, tuple[str, str, str, int, float]] = {}
    for row in rows:
        protein = row["protein_id"]
        current = (
            row["protein_pair_id"],
            row["protein_match_role"],
            row["matched_protein_id"],
            row["protein_match_focal_position"],
            row["protein_match_normalized_position"],
        )
        previous = metadata.setdefault(protein, current)
        if previous != current:
            raise ValueError("protein pair metadata changed across measurement rows")
    if set(metadata) != set(internal_by_protein):
        raise ValueError("protein pair metadata does not cover the frozen protein cohort")
    pair_members: dict[str, dict[str, str]] = {}
    for protein, (pair_id, role, matched, focal, normalized_position) in metadata.items():
        internal = internal_by_protein[protein]
        expected_focal = int(internal["focal_start"])
        expected_normalized = expected_focal / max(int(internal["original_length"]) - 1, 1)
        if focal != expected_focal or not np.isclose(
            normalized_position, expected_normalized, rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                "source focal_position must equal the frozen internal insertion site"
            )
        roles = pair_members.setdefault(pair_id, {})
        if role in roles:
            raise ValueError("protein_pair_id has duplicate target/control roles")
        roles[role] = protein
        if matched not in metadata:
            raise ValueError("matched protein is outside the frozen protein cohort")
    pairs: list[dict] = []
    for pair_id, members in sorted(pair_members.items()):
        if set(members) != {"target", "control"}:
            raise ValueError("every protein_pair_id requires one target and one control")
        target, control = members["target"], members["control"]
        if metadata[target][2] != control or metadata[control][2] != target:
            raise ValueError("target/control matched_protein_id values must be reciprocal")
        pairs.append(
            {
                "protein_pair_id": pair_id,
                "target_protein_id": target,
                "control_protein_id": control,
            }
        )

    cells: dict[tuple, set[tuple[str, str]]] = {}
    for row in rows:
        key = (
            row["model"],
            row["layer"],
            row["feature"],
            row["feature_role"],
            row["protein_pair_id"],
            row["protein_match_role"],
        )
        cells.setdefault(key, set()).add((row["condition"], row["bos_policy"]))
    expected_cells = {(condition, bos) for condition in CONDITIONS for bos in BOS_POLICIES}
    if any(value != expected_cells for value in cells.values()):
        raise ValueError(
            "every feature requires a complete condition/BOS factorial for both "
            "target/control protein-pair roles"
        )
    expected_keys = {
        (
            row["model"],
            row["layer"],
            row["feature"],
            row["feature_role"],
            pair["protein_pair_id"],
            role,
        )
        for row in rows
        for pair in pairs
        for role in ("target", "control")
    }
    if set(cells) != expected_keys:
        raise ValueError(
            "every feature must cover the identical frozen protein set and both "
            "roles of every protein_pair_id"
        )
    return pairs


def _bootstrap_mean(values: np.ndarray, *, seed: int, n_bootstrap: int) -> dict:
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("paired effects must be a finite vector with at least two proteins")
    rng = np.random.default_rng(seed)
    draws = np.array(
        [values[rng.integers(0, values.size, values.size)].mean() for _ in range(n_bootstrap)]
    )
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "n_proteins": int(values.size),
        "n_bootstrap": int(n_bootstrap),
    }


def _holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    values = np.asarray(pvalues, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Holm adjustment requires a non-empty finite p-value vector")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    adjusted_ranked = np.maximum.accumulate((values.size - np.arange(values.size)) * ranked)
    adjusted = np.empty_like(values)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def _paired_target_control_inference(
    values: np.ndarray,
    *,
    margin: float,
    alpha: float,
    seed: int,
    n_resamples: int,
) -> dict:
    summary = _bootstrap_mean(values, seed=seed, n_bootstrap=n_resamples)
    rng = np.random.default_rng(_derived_seed(seed, "sign_randomization"))
    null = np.array(
        [np.mean(values * rng.choice([-1.0, 1.0], values.size)) for _ in range(n_resamples)]
    )
    observed = float(values.mean())
    return {
        **summary,
        "two_sided_p": float((1 + np.sum(np.abs(null) >= abs(observed))) / (n_resamples + 1)),
        "equivalence": tost_paired(values, margin, alpha=alpha),
    }


def _feature_matches(rows: Sequence[dict], control_count: int) -> list[dict]:
    """Recover the extractor-frozen feature sets without rematching or ranking."""

    if type(control_count) is not int or control_count < 1:
        raise ValueError("control_count must be a positive integer")
    profiles: dict[tuple[str, int, int], dict] = {}
    for row in rows:
        identity = (row["model"], row["layer"], row["feature"])
        profile = {
            "model": row["model"],
            "layer": row["layer"],
            "feature": row["feature"],
            "role": row["feature_role"],
            "feature_match_id": row["feature_match_id"],
            "firing_frequency": row["firing_frequency"],
            "input_norm": row["input_norm"],
        }
        previous = profiles.setdefault(identity, profile)
        if previous != profile:
            raise ValueError(
                "feature role, frozen match ID, firing frequency and input norm "
                "must be invariant across all measurement rows"
            )

    by_match: dict[str, list[dict]] = {}
    for profile in profiles.values():
        by_match.setdefault(profile["feature_match_id"], []).append(profile)
    matches = []
    for match_id, members in sorted(by_match.items()):
        targets = [member for member in members if member["role"] == "target"]
        controls = [member for member in members if member["role"] == "control"]
        if len(targets) != 1 or len(controls) != control_count:
            raise ValueError(
                "every frozen feature_match_id must contain exactly one target and "
                f"exactly {control_count} controls"
            )
        target = targets[0]
        if any(
            (control["model"], control["layer"])
            != (target["model"], target["layer"])
            for control in controls
        ):
            raise ValueError("a frozen feature match must stay within one model/layer")
        matches.append(
            {
                "feature_match_id": match_id,
                "matching_source": "extractor_frozen_feature_match_id",
                "target": target,
                "controls": [
                    {"profile": control}
                    for control in sorted(controls, key=lambda item: item["feature"])
                ],
            }
        )
    return matches


def _conditional_coefficients(
    rows: Sequence[dict], *, seed: int, n_bootstrap: int
) -> dict:
    proteins = sorted({row["protein_id"] for row in rows})
    rows_by_protein = {
        protein: [row for row in rows if row["protein_id"] == protein]
        for protein in proteins
    }

    def design(selected: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        chosen = []
        group_labels = []
        for draw_index, protein in enumerate(selected):
            chosen.extend(rows_by_protein[protein])
            group_labels.extend([f"{protein}__draw_{draw_index}"] * len(rows_by_protein[protein]))
        x = np.array(
            [
                [
                    row["normalized_received_attention"],
                    row["normalized_focal_position"],
                    row["sequence_length"],
                    float(row["condition"] != "m_to_a"),
                    float(row["condition"] in {"natural_mxx", "m_to_a", "artificial_truncation"}),
                    float(row["bos_policy"] == "native"),
                ]
                for row in chosen
            ],
            dtype=np.float64,
        )
        y = np.array([row["feature_activation_pre"] for row in chosen], dtype=np.float64)
        group = np.array(group_labels)
        for protein in np.unique(group):
            mask = group == protein
            x[mask] -= x[mask].mean(axis=0)
            y[mask] -= y[mask].mean()
        return x, y

    x, y = design(proteins)
    if np.linalg.matrix_rank(x) < x.shape[1]:
        raise ValueError("conditional N-terminal design is rank deficient")
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(proteins, size=len(proteins), replace=True).tolist()
        xb, yb = design(sampled)
        if np.linalg.matrix_rank(xb) == xb.shape[1]:
            draws.append(np.linalg.lstsq(xb, yb, rcond=None)[0])
    if not draws:
        raise RuntimeError("no full-rank protein bootstrap draws were produced")
    boot = np.stack(draws)
    names = (
        "normalized_received_attention",
        "normalized_position",
        "sequence_length",
        "initiator_m",
        "sequence_start",
        "bos_token",
    )
    return {
        name: {
            "coefficient": float(coefficients[index]),
            "ci95": [
                float(np.percentile(boot[:, index], 2.5)),
                float(np.percentile(boot[:, index], 97.5)),
            ],
        }
        for index, name in enumerate(names)
    }


def _unique_attention_path_rows(rows: Sequence[dict]) -> list[dict]:
    """Collapse feature/layer duplicates after checking identical path effects."""

    unique: dict[tuple, dict] = {}
    fields = (
        "baseline_suffix_nll",
        "key_masked_suffix_nll",
        "suffix_nll_increase_key_masked",
        "baseline_suffix_observed_token_logit_mean",
        "key_masked_suffix_observed_token_logit_mean",
        "suffix_observed_token_logit_change_key_masked",
        "attention_key_mask_max_abs_strict_suffix",
        "normalized_focal_position",
        "sequence_length",
        "protein_pair_id",
        "protein_match_role",
    )
    for row in rows:
        key = (row["protein_id"], row["condition"], row["bos_policy"], row["model"])
        previous = unique.setdefault(key, row)
        mismatched = False
        for field in fields:
            left, right = previous[field], row[field]
            if isinstance(left, str) or isinstance(right, str):
                mismatched = left != right
            else:
                mismatched = not np.isclose(left, right, rtol=0.0, atol=1e-12)
            if mismatched:
                break
        if mismatched:
            raise ValueError(
                "attention-path intervention effects changed across feature/layer duplicates"
            )
    return list(unique.values())


def _conditional_attention_path_effects(
    rows: Sequence[dict], *, seed: int, n_bootstrap: int
) -> dict:
    """Condition key-mask effects on position/length without mediation claims."""

    unique = _unique_attention_path_rows(rows)
    pair_ids = sorted({row["protein_pair_id"] for row in unique})
    rows_by_pair = {
        pair_id: [row for row in unique if row["protein_pair_id"] == pair_id]
        for pair_id in pair_ids
    }

    def design(selected: Sequence[str], metric: str) -> tuple[np.ndarray, np.ndarray]:
        chosen: list[dict] = []
        groups: list[str] = []
        for draw, pair_id in enumerate(selected):
            pair_rows = rows_by_pair[pair_id]
            chosen.extend(pair_rows)
            groups.extend([f"{pair_id}__draw_{draw}"] * len(pair_rows))
        x = np.asarray(
            [
                [
                    row["normalized_focal_position"],
                    np.log(float(row["sequence_length"])),
                    float(row["condition"] != "m_to_a"),
                    float(
                        row["condition"]
                        in {"natural_mxx", "m_to_a", "artificial_truncation"}
                    ),
                    float(row["bos_policy"] == "native"),
                    float(row["protein_match_role"] == "target"),
                ]
                for row in chosen
            ],
            dtype=np.float64,
        )
        y = np.asarray([row[metric] for row in chosen], dtype=np.float64)
        group = np.asarray(groups)
        for value in np.unique(group):
            mask = group == value
            x[mask] -= x[mask].mean(axis=0)
            y[mask] -= y[mask].mean()
        return x, y

    names = (
        "normalized_position",
        "log_sequence_length",
        "initiator_m",
        "sequence_start",
        "bos_token",
        "target_protein_role",
    )
    result: dict[str, dict] = {}
    for metric in ATTENTION_PATH_METRICS:
        x, y = design(pair_ids, metric)
        if np.linalg.matrix_rank(x) < x.shape[1]:
            raise ValueError(f"conditional attention-path design is rank deficient: {metric}")
        coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
        rng = np.random.default_rng(_derived_seed(seed, metric, "attention_path"))
        draws = []
        for _ in range(n_bootstrap):
            sampled = rng.choice(pair_ids, size=len(pair_ids), replace=True).tolist()
            xb, yb = design(sampled, metric)
            if np.linalg.matrix_rank(xb) == xb.shape[1]:
                draws.append(np.linalg.lstsq(xb, yb, rcond=None)[0])
        if not draws:
            raise RuntimeError(
                f"no full-rank pair bootstrap draws for attention-path metric {metric}"
            )
        boot = np.stack(draws)
        result[metric] = {
            name: {
                "coefficient": float(coefficients[index]),
                "ci95": [
                    float(np.percentile(boot[:, index], 2.5)),
                    float(np.percentile(boot[:, index], 97.5)),
                ],
            }
            for index, name in enumerate(names)
        }
    return result


def _apply_joint_holm(results: Sequence[dict], alpha: float) -> None:
    separation = _holm_adjust([result["two_sided_p"] for result in results])
    equivalence = _holm_adjust(
        [result["equivalence"]["p_tost"] for result in results]
    )
    for result, separated_p, equivalent_p in zip(results, separation, equivalence):
        result["two_sided_p_holm"] = float(separated_p)
        result["equivalence"]["p_tost_holm"] = float(equivalent_p)
        result["equivalence"]["equivalent_holm"] = bool(equivalent_p < alpha)
        if result["equivalence"]["equivalent_holm"]:
            result["status"] = "equivalent"
        elif separated_p < alpha:
            result["status"] = "separated"
        else:
            result["status"] = "inconclusive"


def analyze_n_terminal_counterfactuals(
    rows: Sequence[Mapping[str, object]],
    *,
    variants: Sequence[CounterfactualVariant | Mapping[str, object]],
    equivalence_spec: Mapping[str, object],
    n_bootstrap: int = 1000,
    seed: int = 20260717,
    control_count: int = 2,
) -> dict:
    """Estimate paired motif, position, start and BOS effects."""

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    frozen_equivalence = validate_equivalence_spec(equivalence_spec)
    normalized = normalize_measurement_rows(rows, variants)
    protein_pairs = _validate_protein_pair_coverage(normalized, variants)
    expected_proteins = {variant.protein_id if isinstance(variant, CounterfactualVariant) else str(variant["protein_id"]) for variant in variants}
    proteins_by_feature: dict[tuple, set[str]] = {}
    roles_by_feature: dict[tuple, str] = {}
    for row in normalized:
        feature_identity = (row["model"], row["layer"], row["feature"])
        previous_role = roles_by_feature.setdefault(feature_identity, row["feature_role"])
        if previous_role != row["feature_role"]:
            raise ValueError("a model/layer/feature identity cannot be both target and control")
        proteins_by_feature.setdefault((*feature_identity, row["feature_role"]), set()).add(
            row["protein_id"]
        )
    for feature, proteins in proteins_by_feature.items():
        if proteins != expected_proteins:
            raise ValueError(
                f"target/control feature {feature} does not use the identical frozen protein set"
            )
    matches = _feature_matches(normalized, control_count)
    if not matches:
        raise ValueError("at least one target feature and matched controls are required")
    by_feature: dict[tuple, list[dict]] = {}
    for row in normalized:
        key = (row["model"], row["layer"], row["feature"], row["feature_role"])
        by_feature.setdefault(key, []).append(row)

    feature_results = []
    for key, feature_rows in sorted(by_feature.items(), key=lambda item: str(item[0])):
        lookup = {
            (row["protein_id"], row["condition"], row["bos_policy"]): row
            for row in feature_rows
        }
        proteins = sorted({row["protein_id"] for row in feature_rows})
        expected = len(proteins) * len(CONDITIONS) * len(BOS_POLICIES)
        if len(feature_rows) != expected:
            raise ValueError(f"feature {key} lacks the complete condition x BOS factorial")
        contrast_results = {}
        for contrast_name, (left_key, right_key) in CONTRASTS.items():
            contrast_results[contrast_name] = {}
            for metric in FEATURE_METRICS:
                values = np.array(
                    [
                        lookup[(protein, *left_key)][metric]
                        - lookup[(protein, *right_key)][metric]
                        for protein in proteins
                    ]
                )
                contrast_results[contrast_name][metric] = _bootstrap_mean(
                    values,
                    seed=_derived_seed(seed, key, contrast_name, metric),
                    n_bootstrap=n_bootstrap,
                )
        feature_results.append(
            {
                "model": key[0],
                "layer": key[1],
                "feature": key[2],
                "feature_role": key[3],
                "feature_match_id": feature_rows[0]["feature_match_id"],
                "protein_matching": "within_protein_exact",
                "n_proteins": len(proteins),
                "contrasts": contrast_results,
                "conditional_model": _conditional_coefficients(
                    feature_rows,
                    seed=_derived_seed(seed, key, "conditional"),
                    n_bootstrap=n_bootstrap,
                ),
            }
        )

    row_lookup = {
        key: {
            (row["protein_id"], row["condition"], row["bos_policy"]): row
            for row in normalized
            if (
                row["model"],
                row["layer"],
                row["feature"],
                row["feature_role"],
            )
            == key
        }
        for key in by_feature
    }
    target_control_results = []
    ordered_proteins = sorted(expected_proteins)
    for match in matches:
        target = match["target"]
        target_key = (
            target["model"],
            target["layer"],
            target["feature"],
            "target",
        )
        control_keys = [
            (
                item["profile"]["model"],
                item["profile"]["layer"],
                item["profile"]["feature"],
                "control",
            )
            for item in match["controls"]
        ]
        for contrast_name, (left_key, right_key) in CONTRASTS.items():
            for metric in FEATURE_METRICS:
                values = []
                for protein in ordered_proteins:
                    target_contrast = (
                        row_lookup[target_key][(protein, *left_key)][metric]
                        - row_lookup[target_key][(protein, *right_key)][metric]
                    )
                    control_contrasts = [
                        row_lookup[control_key][(protein, *left_key)][metric]
                        - row_lookup[control_key][(protein, *right_key)][metric]
                        for control_key in control_keys
                    ]
                    values.append(target_contrast - float(np.mean(control_contrasts)))
                target_control_results.append(
                    {
                        "model": target["model"],
                        "layer": target["layer"],
                        "feature_match_id": match["feature_match_id"],
                        "target_feature": target["feature"],
                        "control_features": [key[2] for key in control_keys],
                        "contrast": contrast_name,
                        "metric": metric,
                        "estimand": "target_contrast_minus_mean_matched_control_contrast",
                        **_paired_target_control_inference(
                            np.asarray(values, dtype=np.float64),
                            margin=frozen_equivalence["margins"][metric],
                            alpha=frozen_equivalence["alpha"],
                            seed=_derived_seed(seed, target_key, contrast_name, metric),
                            n_resamples=n_bootstrap,
                        ),
                    }
                )
    protein_pair_did_results: list[dict] = []
    for key, lookup in row_lookup.items():
        for contrast_name, (left_key, right_key) in CONTRASTS.items():
            for metric in FEATURE_METRICS:
                paired_effects = []
                for pair in protein_pairs:
                    target = pair["target_protein_id"]
                    control = pair["control_protein_id"]
                    target_contrast = (
                        lookup[(target, *left_key)][metric]
                        - lookup[(target, *right_key)][metric]
                    )
                    control_contrast = (
                        lookup[(control, *left_key)][metric]
                        - lookup[(control, *right_key)][metric]
                    )
                    paired_effects.append(
                        {
                            "protein_pair_id": pair["protein_pair_id"],
                            "difference_in_differences": target_contrast
                            - control_contrast,
                        }
                    )
                values = np.asarray(
                    [row["difference_in_differences"] for row in paired_effects],
                    dtype=np.float64,
                )
                protein_pair_did_results.append(
                    {
                        "model": key[0],
                        "layer": key[1],
                        "feature": key[2],
                        "feature_role": key[3],
                        "contrast": contrast_name,
                        "metric": metric,
                        "estimand": (
                            "target_protein_within_condition_contrast_minus_"
                            "matched_control_protein_within_condition_contrast"
                        ),
                        "paired_effects": paired_effects,
                        **_paired_target_control_inference(
                            values,
                            margin=frozen_equivalence["margins"][metric],
                            alpha=frozen_equivalence["alpha"],
                            seed=_derived_seed(seed, key, contrast_name, metric, "pair_did"),
                            n_resamples=n_bootstrap,
                        ),
                    }
                )

    path_rows = _unique_attention_path_rows(normalized)
    path_lookup = {
        (row["protein_id"], row["condition"], row["bos_policy"], row["model"]): row
        for row in path_rows
    }
    for model in sorted({row["model"] for row in path_rows}):
        for contrast_name, (left_key, right_key) in CONTRASTS.items():
            for metric in ATTENTION_PATH_METRICS:
                paired_effects = []
                for pair in protein_pairs:
                    target = pair["target_protein_id"]
                    control = pair["control_protein_id"]
                    target_contrast = (
                        path_lookup[(target, *left_key, model)][metric]
                        - path_lookup[(target, *right_key, model)][metric]
                    )
                    control_contrast = (
                        path_lookup[(control, *left_key, model)][metric]
                        - path_lookup[(control, *right_key, model)][metric]
                    )
                    paired_effects.append(
                        {
                            "protein_pair_id": pair["protein_pair_id"],
                            "difference_in_differences": target_contrast
                            - control_contrast,
                        }
                    )
                values = np.asarray(
                    [row["difference_in_differences"] for row in paired_effects],
                    dtype=np.float64,
                )
                protein_pair_did_results.append(
                    {
                        "model": model,
                        "layer": None,
                        "feature": None,
                        "feature_role": "attention_path_intervention",
                        "contrast": contrast_name,
                        "metric": metric,
                        "estimand": (
                            "target_protein_within_condition_key_mask_effect_minus_"
                            "matched_control_protein_within_condition_key_mask_effect"
                        ),
                        "paired_effects": paired_effects,
                        **_paired_target_control_inference(
                            values,
                            margin=frozen_equivalence["margins"][metric],
                            alpha=frozen_equivalence["alpha"],
                            seed=_derived_seed(seed, model, contrast_name, metric, "pair_did"),
                            n_resamples=n_bootstrap,
                        ),
                    }
                )
    multiplicity_results = [*target_control_results, *protein_pair_did_results]
    _apply_joint_holm(multiplicity_results, frozen_equivalence["alpha"])
    return {
        "schema_version": "r2-n-terminal-counterfactuals-v2",
        "scope": (
            "Paired counterfactual analysis with opportunity-normalized attention and "
            "a focal-key suffix intervention. The intervention is not formal feature "
            "mediation and does not by itself establish an attention-sink mechanism."
        ),
        "n_measurement_rows": len(normalized),
        "conditions": list(CONDITIONS),
        "bos_policies": list(BOS_POLICIES),
        "attention_normalization": "raw_received_mass / eligible_valid_causal_queries",
        "feature_measurement_timing": "before_any_attention_intervention",
        "protein_set_contract": "identical_frozen_proteins_for_every_target_and_control",
        "variant_join_contract": "exact_sequence_sha256_and_canonical_token_ids_sha256",
        "frozen_equivalence_spec": frozen_equivalence,
        "multiplicity_family_size": len(multiplicity_results),
        "protein_pair_contract": "complete_reciprocal_target_control_pairs",
        "protein_pairs": protein_pairs,
        "feature_matches": matches,
        "features": feature_results,
        "target_minus_control_inference": target_control_results,
        "protein_pair_difference_in_differences": protein_pair_did_results,
        "conditional_attention_path_effects": _conditional_attention_path_effects(
            normalized,
            seed=_derived_seed(seed, "conditional_attention_path"),
            n_bootstrap=n_bootstrap,
        ),
        "normalized_rows": normalized,
    }


def synthetic_n_terminal_fixture(
    *, seed: int = 20260717, n_proteins: int = 48
) -> tuple[list[dict], list[CounterfactualVariant], list[dict]]:
    """CPU-only fixture with separable planted M, position, start and BOS effects."""

    if n_proteins < 24 or n_proteins % 2:
        raise ValueError("synthetic fixture requires an even count of at least 24 proteins")
    rng = np.random.default_rng(seed)
    alphabet = np.array(sorted(AA))
    records = []
    for index in range(n_proteins):
        length = int(rng.integers(55, 91))
        sequence = rng.choice(alphabet, size=length).tolist()
        sequence[:3] = ["M", "A" if index % 2 else "S", "K" if index % 3 else "G"]
        insertion = int(round((length - 3) * 0.55))
        insertion = min(max(insertion, 4), length - 4)
        records.append(
            {
                "protein_id": f"synthetic-{index:04d}",
                "sequence": "".join(sequence),
                "focal_position": insertion,
            }
        )
    variants = build_counterfactual_variants(records)
    protein_noise = {record["protein_id"]: float(rng.normal(scale=0.02)) for record in records}
    feature_match_id = _sha256_text("synthetic-feature-match-101-201-202")
    feature_profiles = [
        (101, "target", 0.31, 1.20, 1.0),
        (201, "control", 0.30, 1.18, 0.15),
        (202, "control", 0.33, 1.23, 0.10),
    ]
    rows = []
    amino_token = {residue: index + 1 for index, residue in enumerate(sorted(AA))}
    for variant in variants:
        protein_index = int(variant.protein_id.rsplit("-", 1)[1])
        pair_index = protein_index // 2
        role = "target" if protein_index % 2 == 0 else "control"
        matched_index = protein_index + (1 if role == "target" else -1)
        pair_id = _sha256_text(f"synthetic-pair-{pair_index:04d}")
        path_strength = 1.0 if role == "target" else 0.35
        is_m = float(variant.condition != "m_to_a")
        is_start = float(
            variant.condition in {"natural_mxx", "m_to_a", "artificial_truncation"}
        )
        for bos_policy in BOS_POLICIES:
            bos = float(bos_policy == "native")
            token_ids = [amino_token[residue] for residue in variant.sequence]
            if bos_policy == "native":
                token_ids = [0, *token_ids]
            focal_token_index = variant.focal_start + int(bos)
            suffix_nll_increase = (
                0.025
                + path_strength
                * (
                    0.018 * is_m
                    + 0.026 * is_start
                    - 0.010 * variant.normalized_focal_position
                    + 0.008 * bos
                )
                + protein_noise[variant.protein_id]
            )
            suffix_logit_change = -0.8 * suffix_nll_increase
            baseline_suffix_nll = 2.4 + 0.05 * np.log(variant.sequence_length)
            baseline_suffix_logit = 1.1 - 0.02 * np.log(variant.sequence_length)
            for feature, role, firing, input_norm, strength in feature_profiles:
                normalized_attention = (
                    0.08
                    + strength * (
                        0.025 * is_m
                        + 0.035 * is_start
                        - 0.020 * variant.normalized_focal_position
                        + 0.015 * bos
                    )
                    + protein_noise[variant.protein_id]
                    + float(rng.normal(scale=0.004))
                )
                eligible = len(token_ids) - focal_token_index
                feature_activation = (
                    0.55
                    + strength
                    * (
                        0.42 * is_m
                        + 0.30 * is_start
                        - 0.20 * variant.normalized_focal_position
                        + 0.16 * bos
                        + 0.50 * normalized_attention
                    )
                    + protein_noise[variant.protein_id]
                )
                rows.append(
                    {
                        "protein_id": variant.protein_id,
                        "condition": variant.condition,
                        "bos_policy": bos_policy,
                        "model": "synthetic_rotated_decoder",
                        "layer": 2,
                        "feature": feature,
                        "feature_role": role,
                        "feature_match_id": feature_match_id,
                        "feature_activation_pre": feature_activation,
                        "received_attention_raw": normalized_attention * eligible,
                        "eligible_query_count": eligible,
                        "normalized_focal_position": variant.normalized_focal_position,
                        "sequence_length": variant.sequence_length,
                        "firing_frequency": firing,
                        "input_norm": input_norm,
                        "variant_sha256": variant.sha256,
                        "tokenizer_revision": "synthetic-character-tokenizer-v1",
                        "token_ids": token_ids,
                        "token_ids_sha256": token_ids_sha256(token_ids),
                        "focal_token_index": focal_token_index,
                        "protein_pair_id": pair_id,
                        "protein_match_role": (
                            "target" if protein_index % 2 == 0 else "control"
                        ),
                        "matched_protein_id": f"synthetic-{matched_index:04d}",
                        "protein_match_focal_position": records[protein_index][
                            "focal_position"
                        ],
                        "protein_match_normalized_position": records[protein_index][
                            "focal_position"
                        ]
                        / max(variant.original_length - 1, 1),
                        "baseline_suffix_nll": baseline_suffix_nll,
                        "key_masked_suffix_nll": baseline_suffix_nll
                        + suffix_nll_increase,
                        "suffix_nll_increase_key_masked": suffix_nll_increase,
                        "baseline_suffix_observed_token_logit_mean": (
                            baseline_suffix_logit
                        ),
                        "key_masked_suffix_observed_token_logit_mean": (
                            baseline_suffix_logit + suffix_logit_change
                        ),
                        "suffix_observed_token_logit_change_key_masked": (
                            suffix_logit_change
                        ),
                        "attention_key_mask_max_abs_strict_suffix": 0.0,
                    }
                )
    return records, variants, rows
