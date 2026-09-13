#!/usr/bin/env python3
"""Interface probe and post-score analysis for a native ProteinGym extension.

This is a pre-data contract helper, not a campaign freeze and not a scoring
stage. ``probe`` is a synthetic interface check plus a replay of the existing
LOOKUP queue. ``analyse`` consumes later stage-20 score artefacts. Neither phase
computes a new model Spearman, and neither writes a fitness PASS.

Status: protocol-frozen v1 before any new DMS ρ; no new model fitness scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer import galactica_fitness as G  # noqa: E402
from src.transfer import rita_fitness as RITA  # noqa: E402
from src.transfer import scale_comparison as C  # noqa: E402
from src.transfer.arms import AA20, STAGED_ARMS  # noqa: E402
from src.transfer.fitness import load_assay, wildtype_of  # noqa: E402
from src.transfer.generation_evidence import read_fasta  # noqa: E402
from src.transfer.io import sha256_file, write_json  # noqa: E402
from src.transfer.probes import PROTEINGYM_ROOT  # noqa: E402
from src.transfer.scale_comparison import (  # noqa: E402
    DmsCensus,
    STRATUM_N_TO_C,
    align_dms,
    endpoint,
    require_uniform_dtype,
    require_uniform_stratum,
    unit_mean_record,
)

SCHEMA_VERSION = "r2_native_dms_extension_v0_predata"
STATUS = "protocol-frozen v1 before any new DMS ρ; no new model fitness scores"
VARIANT_SEED = 20260807
VARIANT_CAP = 1000
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260913
FP32_PER_TARGET_ABS = 1e-4
LOWPREC_BATCH_VS_SINGLE = 0.02
CONTEXT_EXCLUSION_REASON = (
    "the rendered variant exceeds this arm's context; truncating would score a "
    "sequence that may not contain the mutated position"
)
ALIGN_EXCLUSION_REASON = "exceeds this arm's context"
GALACTICA_GROUP: tuple[str, ...] = G.GALACTICA_RUNGS
RITA_GROUP: tuple[str, ...] = (RITA.RITA_ARM,)
REFERENCE_PAIR = ("galactica-125m", "galactica-1.3b")
QUALIFIED_PAIRS: tuple[tuple[str, str], ...] = (
    ("galactica-1.3b", "galactica-6.7b"),
    ("galactica-6.7b", "galactica-30b"),
)
GALACTICA_PAIRS: tuple[tuple[str, str], ...] = (REFERENCE_PAIR, *QUALIFIED_PAIRS)
CHANNEL_OFFSETS = {
    "raw_spearman": 0,
    "model_minus_lookup": 100,
    "model_minus_blosum62": 200,
    "lookup_raw": 300,
    "blosum62_raw": 400,
}
RITA_SEED_OFFSET = 1000
GALACTICA_SYNTHETIC = ("MKT", AA20)
RITA_SYNTHETIC = ("MKT", "MRT")
DEFAULT_BATCH_SIZE = 16
BATCH_REDUCTION_LADDER: tuple[int, ...] = (16, 8, 4, 2, 1)
REQUIRED_DTYPE = {
    **{name: "bfloat16" for name in GALACTICA_GROUP},
    RITA.RITA_ARM: RITA.INFERENCE_DTYPE,
}


def mutant_digest(mutants: Sequence[str]) -> str:
    """Fingerprint of drawn mutation strings; matches stage 20's mutant_digest."""

    return hashlib.sha256("\n".join(mutants).encode()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{path} does not exist")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def _finite(value: float, *, what: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{what} is not finite: {value!r}")
    return number


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"expected NAME=PATH, got {value!r}")
    name, _, rest = value.partition("=")
    if not name or not rest:
        raise ValueError(f"expected NAME=PATH, got {value!r}")
    return name, Path(rest)


def _runtime_record() -> dict[str, Any]:
    missing: list[str] = []
    try:
        import torch
    except ImportError:
        missing.append("torch")
        torch = None  # type: ignore[assignment]
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
        transformers = None  # type: ignore[assignment]
    if missing:
        raise RuntimeError(
            "missing required dependencies "
            f"{missing}; will not auto-install nnsight, wandb, or anything else"
        )
    assert torch is not None
    assert transformers is not None
    return {
        "python": sys.version.split()[0],
        "python_full": " ".join(sys.version.split()),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "torch_cuda_compiled": getattr(torch.version, "cuda", None),
        "note": (
            "versions are whatever this process imported. Compute ct and the "
            "H200 image are different stacks; a CPU pytest pass is not an H200 pass"
        ),
    }


def _code_sources() -> dict[str, str]:
    return {
        "script": str(Path(__file__).resolve()),
        "galactica_fitness": str(Path(G.__file__).resolve()),
        "rita_fitness": str(Path(RITA.__file__).resolve()),
        "fitness.load_assay": "src.transfer.fitness.load_assay",
        "scale_comparison": str(Path(C.__file__).resolve()),
        "io.sha256_file": "src.transfer.io.sha256_file",
        "rita_checkpoint": str(STAGED_ARMS["rita-xl"].path),
        **{
            f"galactica_checkpoint_{name}": str(G.CHECKPOINTS[name])
            for name in GALACTICA_GROUP
        },
    }


def _arm_family(arm: str) -> str:
    if arm in GALACTICA_GROUP:
        return "galactica"
    if arm == RITA.RITA_ARM:
        return "rita"
    raise KeyError(
        f"unknown native DMS arm {arm!r}; declared: {list(GALACTICA_GROUP) + list(RITA_GROUP)}"
    )


def _require_dtype(arm: str, dtype: str) -> str:
    expected = REQUIRED_DTYPE[arm]
    if dtype != expected:
        raise ValueError(f"{arm} scoring dtype must be {expected!r}; got {dtype!r}")
    return dtype


def load_native_scorer(
    arm: str, *, device: str, dtype: str, batch_size: int
) -> Any:
    """Public scorers only. Does not copy stage-20 scoring or invent a pad token."""

    family = _arm_family(arm)
    _require_dtype(arm, dtype)
    if family == "galactica":
        loaded = G.load_galactica(arm, device=device, dtype=dtype)
        return G.GalacticaFitnessScorer(loaded, batch_size=batch_size)
    return RITA.RitaFitnessScorer(
        checkpoint=STAGED_ARMS["rita-xl"].path,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
    )


def freeze_lookup_cohort(
    lookup: Mapping[str, Any],
    *,
    proteingym_dir: Path,
    wildtypes: Mapping[str, Any],
    wildtypes_path: Path,
    wildtypes_fasta_path: Path,
) -> dict[str, Any]:
    """Replay LOOKUP's original assay order. Do not skip, then reset the seed index.

    ``wildtypes.json`` identifies wild types by ``qNNNNN`` and length, not by
    sequence. The sequence is recovered with :func:`wildtype_of` from the same
    CSV :func:`load_assay` reads, then checked against the original
    ``wildtypes.faa`` full string for that id. ``mutant_digest`` still
    fingerprints mutation strings only.
    """

    rows = list(lookup["assays"])
    if not rows:
        raise ValueError("LOOKUP carries no assays")
    fasta_path = Path(wildtypes_fasta_path)
    if not fasta_path.is_file():
        raise FileNotFoundError(f"wildtypes FASTA does not exist: {fasta_path}")
    fasta = read_fasta(fasta_path)
    assay_to_wildtype = wildtypes.get("assay_to_wildtype")
    catalog_rows = wildtypes.get("wildtypes")
    if not isinstance(assay_to_wildtype, Mapping):
        raise ValueError("wildtypes.json has no assay_to_wildtype map")
    if not isinstance(catalog_rows, Mapping):
        raise ValueError("wildtypes.json has no wildtypes table")
    assays: list[dict[str, Any]] = []
    sequences: dict[str, list[str]] = {}
    csv_hashes: dict[str, str] = {}
    clusters: set[str] = set()
    recovered_by_id: dict[str, str] = {}
    for index, row in enumerate(rows):
        name = str(row["assay"])
        csv_path = Path(proteingym_dir) / f"{name}.csv"
        csv_hashes[name] = sha256_file(csv_path)
        assay = load_assay(
            name,
            n=VARIANT_CAP,
            seed=VARIANT_SEED + index,
            directory=proteingym_dir,
        )
        recovered = wildtype_of(name, proteingym_dir)
        if assay.wildtype != recovered:
            raise ValueError(
                f"{name}: load_assay wild type disagrees with wildtype_of"
            )
        digest = mutant_digest(assay.mutants)
        if digest != row["mutant_digest"]:
            raise ValueError(
                f"{name}: mutant_digest {digest} disagrees with LOOKUP "
                f"{row['mutant_digest']}; the original index {index} was not reset"
            )
        n_variants = len(assay.sequences)
        if n_variants != int(row["n_variants"]):
            raise ValueError(
                f"{name}: n_variants {n_variants} disagrees with LOOKUP "
                f"{row['n_variants']}"
            )
        if "wildtype_id" not in row or row["wildtype_id"] in (None, ""):
            raise ValueError(f"{name}: LOOKUP row carries no wildtype_id")
        wildtype_id = str(row["wildtype_id"])
        if name not in assay_to_wildtype:
            raise ValueError(f"{name}: wildtypes.json has no assay_to_wildtype entry")
        catalog_id = str(assay_to_wildtype[name])
        if catalog_id != wildtype_id:
            raise ValueError(
                f"{name}: LOOKUP wildtype_id {wildtype_id!r} disagrees with "
                f"wildtypes.json {catalog_id!r}"
            )
        entry = catalog_rows.get(catalog_id)
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"{name}: wildtypes.json has no row for {catalog_id!r}"
            )
        if "length" not in entry:
            raise ValueError(
                f"{name}: wildtypes.json {catalog_id!r} has no length"
            )
        if int(entry["length"]) != len(recovered):
            raise ValueError(
                f"{name}: recovered wild type length {len(recovered)} disagrees "
                f"with wildtypes.json length {entry['length']}"
            )
        if catalog_id not in fasta:
            raise ValueError(
                f"{name}: wildtypes FASTA has no id {catalog_id!r}"
            )
        fasta_sequence = fasta[catalog_id]
        if fasta_sequence != recovered:
            raise ValueError(
                f"{name}: recovered wild type disagrees with FASTA id {catalog_id!r}"
            )
        cluster = str(row["cluster"])
        if "cluster" in entry and str(entry["cluster"]) != cluster:
            raise ValueError(
                f"{name}: LOOKUP cluster {cluster!r} disagrees with "
                f"wildtypes.json {entry['cluster']!r}"
            )
        if catalog_id in recovered_by_id and recovered_by_id[catalog_id] != recovered:
            raise ValueError(
                f"{name}: wild type {catalog_id!r} recovered as {recovered!r}, "
                f"but another assay recovered {recovered_by_id[catalog_id]!r}"
            )
        recovered_by_id[catalog_id] = recovered
        clusters.add(cluster)
        assays.append(
            {
                "assay": name,
                "index": int(index),
                "seed": int(VARIANT_SEED + index),
                "wildtype_id": wildtype_id,
                "wildtype_sequence": recovered,
                "cluster": cluster,
                "n_variants": n_variants,
                "mutant_digest": digest,
                "csv_sha256": csv_hashes[name],
            }
        )
        sequences[name] = list(assay.sequences)
    return {
        "declared_assays": len(assays),
        "declared_clusters": len(clusters),
        "assays": assays,
        "sequences": sequences,
        "csv_sha256": csv_hashes,
        "lookup_order": [row["assay"] for row in assays],
        "wildtypes_sha256": sha256_file(wildtypes_path),
        "wildtypes_fasta_sha256": sha256_file(fasta_path),
    }


def attach_token_lengths(
    frozen: Mapping[str, Any],
    scorer: Any,
) -> dict[str, Any]:
    """Exclude only assays whose rendered max length exceeds this scorer's context."""

    context = int(scorer.context)
    assays: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    analysis: list[str] = []
    analysis_clusters: set[str] = set()
    for row in frozen["assays"]:
        name = row["assay"]
        lengths = [int(value) for value in scorer.token_lengths(frozen["sequences"][name])]
        if not lengths:
            raise ValueError(f"{name}: token_lengths returned no values")
        max_tokens = int(max(lengths))
        entry = dict(row)
        entry["max_tokens"] = max_tokens
        entry["context"] = context
        if max_tokens > context:
            entry["exceeds_context"] = True
            skipped.append(
                {
                    "assay": name,
                    "max_tokens": max_tokens,
                    "context": context,
                    "reason": CONTEXT_EXCLUSION_REASON,
                }
            )
        else:
            entry["exceeds_context"] = False
            analysis.append(name)
            analysis_clusters.add(str(row["cluster"]))
        assays.append(entry)
    return {
        "declared_assays": int(frozen["declared_assays"]),
        "declared_clusters": int(frozen["declared_clusters"]),
        "analysis_assays": analysis,
        "analysis_clusters": len(analysis_clusters),
        "context": context,
        "context_excluded_assays": [row["assay"] for row in skipped],
        "assays": assays,
        "skipped": skipped,
        "csv_sha256": dict(frozen["csv_sha256"]),
        "lookup_order": list(frozen["lookup_order"]),
    }


def _n_scored_targets(scorer: Any, sequences: Sequence[str], *, arm: str) -> list[int]:
    family = _arm_family(arm)
    if family == "galactica":
        return [int(record.n_scored_tokens) for record in scorer.render(sequences)]
    encoded = [
        RITA._encode_native(
            scorer.tokenizer,
            sequence,
            eos_id=scorer.eos_id,
            residue_ids=scorer.residue_ids,
        )
        for sequence in sequences
    ]
    return [max(len(ids) - 1, 0) for ids in encoded]


def _require_native_loss(outputs: Any, *, arm: str) -> Any:
    loss = getattr(outputs, "loss", None)
    if loss is None:
        raise RuntimeError(
            f"{arm}: native CausalLM forward(labels=...) returned no loss; "
            "the independent shifted-CE reference is not a substitute"
        )
    return loss


def _native_vs_reference(
    *, arm: str, native_loss: Any, reference_nll_sum: float, n_targets: int
) -> float:
    if n_targets < 1:
        raise RuntimeError(f"{arm}: native-loss batch has no scored targets")
    native_total = float(native_loss) * n_targets
    per_target = abs(native_total - reference_nll_sum) / n_targets
    if per_target > FP32_PER_TARGET_ABS:
        raise RuntimeError(
            f"{arm}: native CausalLM loss disagrees with the independent FP32 "
            f"shifted-CE reference at {per_target} nats/target; FP32 tolerance "
            f"is {FP32_PER_TARGET_ABS}. The threshold is not moved to match the run"
        )
    return per_target


def _galactica_author_log_likelihood(
    scorer: Any, sequences: Sequence[str]
) -> tuple[np.ndarray, dict[str, Any]]:
    torch = scorer.torch
    import torch.nn.functional as F

    records = scorer.render(sequences)
    totals = np.empty(len(records), dtype=np.float64)
    device = scorer.loaded.model.device
    model = scorer.loaded.model
    native_max = 0.0
    with torch.inference_mode():
        for start in range(0, len(records), scorer.batch_size):
            chunk = records[start : start + scorer.batch_size]
            width = max(len(record.token_ids) for record in chunk)
            ids = torch.full((len(chunk), width), scorer.pad_id, dtype=torch.long)
            mask = torch.zeros((len(chunk), width), dtype=torch.long)
            labels = torch.full((len(chunk), width), -100, dtype=torch.long)
            for row, record in enumerate(chunk):
                length = len(record.token_ids)
                ids[row, :length] = torch.tensor(record.token_ids, dtype=torch.long)
                mask[row, :length] = 1
                labels[row, list(record.scored_positions)] = ids[
                    row, list(record.scored_positions)
                ]
            ids = ids.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            logits = model(
                input_ids=ids, attention_mask=mask, use_cache=False
            ).logits.float()
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:]
            per_token = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).reshape(len(chunk), width - 1)
            valid = shift_labels != -100
            nll = (per_token * valid).sum(1)
            totals[start : start + len(chunk)] = (-nll).cpu().numpy()
            outputs = model(
                input_ids=ids,
                attention_mask=mask,
                use_cache=False,
                labels=labels,
            )
            native_max = max(
                native_max,
                _native_vs_reference(
                    arm="galactica",
                    native_loss=_require_native_loss(outputs, arm="galactica"),
                    reference_nll_sum=float(nll.sum().item()),
                    n_targets=int(valid.sum().item()),
                ),
            )
    extras = {
        "native_api_loss_present": True,
        "native_api_vs_reference_max_abs_nats_per_target": native_max,
        "reference_ce_is_independent_fp32_shifted_ce": True,
    }
    return totals, extras


def _rita_author_log_likelihood(
    scorer: Any, sequences: Sequence[str]
) -> tuple[np.ndarray, dict[str, Any]]:
    torch = scorer.torch
    import torch.nn.functional as F

    encoded = [
        RITA._encode_native(
            scorer.tokenizer,
            sequence,
            eos_id=scorer.eos_id,
            residue_ids=scorer.residue_ids,
        )
        for sequence in sequences
    ]
    widths = {len(ids) for ids in encoded}
    if len(widths) != 1:
        raise ValueError("RITA author-CE probe requires equal-length encodings")
    width = widths.pop()
    device = scorer.model.device
    totals = np.empty(len(encoded), dtype=np.float64)
    mask_max_diff = 0.0
    native_max = 0.0
    with torch.inference_mode():
        for start in range(0, len(encoded), scorer.batch_size):
            chunk = encoded[start : start + scorer.batch_size]
            ids = torch.tensor(chunk, dtype=torch.long, device=device)
            logits = scorer.model(input_ids=ids).logits.float()
            ones = torch.ones_like(ids)
            masked = scorer.model(input_ids=ids, attention_mask=ones).logits.float()
            mask_max_diff = max(
                mask_max_diff, float((logits - masked).abs().max().cpu())
            )
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = ids[:, 1:]
            per_token = F.cross_entropy(
                shift_logits.reshape(-1, RITA.VOCAB_SIZE),
                shift_labels.reshape(-1),
                reduction="none",
            ).reshape(len(chunk), width - 1)
            nll = per_token.sum(1)
            totals[start : start + len(chunk)] = (-nll).cpu().numpy()
            outputs = scorer.model(input_ids=ids, labels=ids)
            native_max = max(
                native_max,
                _native_vs_reference(
                    arm="rita-xl",
                    native_loss=_require_native_loss(outputs, arm="rita-xl"),
                    reference_nll_sum=float(nll.sum().item()),
                    n_targets=int(per_token.numel()),
                ),
            )
    extras = {
        "rita_no_mask_vs_all_ones_max_abs_logit": mask_max_diff,
        "native_api_loss_present": True,
        "native_api_vs_reference_max_abs_nats_per_target": native_max,
        "reference_ce_is_independent_fp32_shifted_ce": True,
    }
    if mask_max_diff > FP32_PER_TARGET_ABS:
        raise RuntimeError(
            "RITA logits without a mask disagree with all-ones attention_mask: "
            f"max abs {mask_max_diff} > {FP32_PER_TARGET_ABS}"
        )
    return totals, extras


def check_author_alignment(
    scorer: Any, sequences: Sequence[str], *, arm: str, dtype: str
) -> dict[str, Any]:
    """Same-batch author CE vs scorer totals. Fail closed; do not loosen thresholds."""

    family = _arm_family(arm)
    torch = scorer.torch
    with torch.inference_mode():
        scorer_totals = np.asarray(scorer.log_likelihood(sequences), dtype=np.float64)
        targets = np.asarray(_n_scored_targets(scorer, sequences, arm=arm), dtype=np.int64)
        if np.any(targets <= 0):
            raise RuntimeError("a synthetic sequence produced no scored targets")
        extras: dict[str, Any] = {}
        if family == "galactica":
            author, extras = _galactica_author_log_likelihood(scorer, sequences)
        else:
            author, extras = _rita_author_log_likelihood(scorer, sequences)
        per_target = np.abs(scorer_totals - author) / targets
        max_author = float(per_target.max())
        if max_author > FP32_PER_TARGET_ABS:
            raise RuntimeError(
                "author shifted CE disagrees with the scorer at "
                f"{max_author} nats/target; FP32 tolerance is {FP32_PER_TARGET_ABS}. "
                "The threshold is not moved to match the run"
            )
        original_bs = int(scorer.batch_size)
        try:
            scorer.batch_size = max(len(sequences), 1)
            batched = np.asarray(scorer.log_likelihood(sequences), dtype=np.float64)
            scorer.batch_size = 1
            singles = np.array(
                [float(scorer.log_likelihood([sequence])[0]) for sequence in sequences],
                dtype=np.float64,
            )
        finally:
            scorer.batch_size = original_bs
        batch_per_target = np.abs(batched - singles) / targets
        max_batch = float(batch_per_target.max())
        batch_tol = FP32_PER_TARGET_ABS if dtype == "float32" else LOWPREC_BATCH_VS_SINGLE
        if max_batch > batch_tol:
            raise RuntimeError(
                "batch vs single-sequence totals disagree at "
                f"{max_batch} nats/target; tolerance is {batch_tol}. "
                "The threshold is not moved to match the run"
            )
        record = {
            "sequences": list(sequences),
            "n_targets": [int(value) for value in targets],
            "author_ce_max_abs_nats_per_target": max_author,
            "batch_vs_single_max_abs_nats_per_target": max_batch,
            "fp32_per_target_abs": FP32_PER_TARGET_ABS,
            "lowprec_batch_vs_single": LOWPREC_BATCH_VS_SINGLE,
            "batch_vs_single_tolerance_used": batch_tol,
            "author_ce_uses_fp32_logits": True,
            **extras,
        }
    return record


def check_refusals(scorer: Any) -> dict[str, str]:
    illegal = "MKTZ"
    try:
        scorer.log_likelihood([illegal])
    except ValueError as exc:
        illegal_ok = "non-canonical" in str(exc) or "canonical" in str(exc)
        if not illegal_ok:
            raise RuntimeError(f"illegal AA refused for the wrong reason: {exc}") from exc
    else:
        raise RuntimeError("illegal amino-acid sequence was accepted")
    over = "W" * (int(scorer.context) + 8)
    try:
        scorer.log_likelihood([over])
    except ValueError as exc:
        if "exceeds" not in str(exc) and "context" not in str(exc):
            raise RuntimeError(f"over-context refused for the wrong reason: {exc}") from exc
    else:
        raise RuntimeError("a rendering longer than context was accepted")
    return {
        "illegal_aa": "refused",
        "over_context": "refused",
    }


def _aa20_tile(n_residues: int) -> str:
    if n_residues < 1:
        raise ValueError(f"eligible wild type has no residues ({n_residues})")
    alphabet = AA20
    repeats = (n_residues + len(alphabet) - 1) // len(alphabet)
    return (alphabet * repeats)[:n_residues]


def _scorer_cuda_device(scorer: Any) -> Any:
    model = getattr(getattr(scorer, "loaded", None), "model", None)
    if model is None:
        model = getattr(scorer, "model", None)
    if model is None:
        return None
    device = getattr(model, "device", None)
    if device is not None and str(getattr(device, "type", "")) == "cuda":
        return device
    parameters = getattr(model, "parameters", None)
    if parameters is None:
        return None
    try:
        param = next(parameters())
    except (StopIteration, TypeError):
        return None
    if bool(getattr(param, "is_cuda", False)):
        return param.device
    return None


def check_longest_eligible_batch(
    scorer: Any,
    cohort: Mapping[str, Any],
    *,
    arm: str,
    batch_size: int,
) -> dict[str, Any]:
    """One configured-batch forward at the longest eligible wild-type length.

    The sequences are an AA20 tiling truncated to that wild type's residue
    count. They are not DMS variants, carry no fitness labels, and their
    log-likelihoods are not retained as scores.
    """

    analysis = [str(name) for name in (cohort.get("analysis_assays") or [])]
    if not analysis:
        raise RuntimeError(
            f"{arm}: no eligible assay remains after context exclusion; "
            "the longest-shape batch gate refuses to run"
        )
    by_name = {row["assay"]: row for row in cohort["assays"]}
    longest: dict[str, Any] | None = None
    for name in analysis:
        row = by_name[name]
        if longest is None or int(row["max_tokens"]) > int(longest["max_tokens"]):
            longest = row
    if longest is None:
        raise RuntimeError(
            f"{arm}: no eligible assay remains after context exclusion; "
            "the longest-shape batch gate refuses to run"
        )
    n_residues = len(str(longest["wildtype_sequence"]))
    synthetic = _aa20_tile(n_residues)
    expected = int(longest["max_tokens"])
    context = int(cohort["context"])
    measured = [int(value) for value in scorer.token_lengths([synthetic])]
    if measured != [expected]:
        raise RuntimeError(
            f"{arm}: synthetic longest-shape token length {measured} != "
            f"eligible max_tokens {expected} for assay {longest['assay']!r}"
        )
    if expected > context:
        raise RuntimeError(
            f"{arm}: eligible max_tokens {expected} exceeds context {context}"
        )
    configured = int(batch_size)
    if configured < 1:
        raise ValueError("batch size must be positive")
    batch = [synthetic] * configured
    batch_lengths = [int(value) for value in scorer.token_lengths(batch)]
    if batch_lengths != [expected] * configured:
        raise RuntimeError(
            f"{arm}: longest-shape batch token lengths {batch_lengths} != "
            f"{expected} x {configured}"
        )
    torch = getattr(scorer, "torch", None)
    if torch is None:
        raise RuntimeError(f"{arm}: longest-shape gate requires scorer.torch")
    cuda_device = _scorer_cuda_device(scorer)
    measured_cuda = cuda_device is not None
    original_bs = int(scorer.batch_size)
    try:
        scorer.batch_size = configured
        if measured_cuda:
            torch.cuda.reset_peak_memory_stats(cuda_device)
            torch.cuda.synchronize(cuda_device)
        started = time.perf_counter()
        with torch.inference_mode():
            totals = np.asarray(scorer.log_likelihood(batch), dtype=np.float64)
        if measured_cuda:
            torch.cuda.synchronize(cuda_device)
        elapsed = time.perf_counter() - started
    finally:
        scorer.batch_size = original_bs
    if totals.shape != (configured,):
        raise RuntimeError(
            f"{arm}: longest-shape log_likelihood shape {totals.shape} != "
            f"({configured},)"
        )
    if not np.all(np.isfinite(totals)):
        raise RuntimeError(
            f"{arm}: longest-shape batch produced a non-finite log-likelihood"
        )
    del totals
    record: dict[str, Any] = {
        "kind": "synthetic_longest_eligible_shape_not_dms_score",
        "not_a_dms_variant": True,
        "assay": longest["assay"],
        "wildtype_id": longest["wildtype_id"],
        "n_residues": n_residues,
        "synthetic_sequence": synthetic,
        "batch_size": configured,
        "token_length": expected,
        "expected_max_tokens": expected,
        "context": context,
        "elapsed_seconds": elapsed,
        "cuda_synchronized": measured_cuda,
        "all_finite": True,
        "log_likelihood_retained": False,
        "cuda_memory_measured": measured_cuda,
    }
    if measured_cuda:
        record["cuda_peak_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(cuda_device)
        )
        record["cuda_peak_reserved_bytes"] = int(
            torch.cuda.max_memory_reserved(cuda_device)
        )
    else:
        record["cuda_peak_allocated_bytes"] = None
        record["cuda_peak_reserved_bytes"] = None
        record["cuda_memory_note"] = (
            "CPU stub: CUDA memory unmeasured; values are not fabricated"
        )
    return record


def run_probe(
    *,
    arm: str,
    lookup_path: Path,
    proteingym_dir: Path,
    wildtypes_path: Path,
    wildtypes_fasta_path: Path,
    out: Path,
    dtype: str,
    batch_size: int,
    device: str = "cuda",
    scorer: Any | None = None,
) -> dict[str, Any]:
    """Synthetic interface check and LOOKUP-queue freeze. Not a DMS score."""

    _arm_family(arm)
    _require_dtype(arm, dtype)
    lookup = _read_json(lookup_path)
    wildtypes = _read_json(wildtypes_path)
    frozen = freeze_lookup_cohort(
        lookup,
        proteingym_dir=proteingym_dir,
        wildtypes=wildtypes,
        wildtypes_path=wildtypes_path,
        wildtypes_fasta_path=wildtypes_fasta_path,
    )
    owned = scorer is None
    if scorer is None:
        scorer = load_native_scorer(
            arm, device=device, dtype=dtype, batch_size=batch_size
        )
    assert scorer is not None
    try:
        family = _arm_family(arm)
        synthetic = GALACTICA_SYNTHETIC if family == "galactica" else RITA_SYNTHETIC
        alignment = check_author_alignment(scorer, synthetic, arm=arm, dtype=dtype)
        refusals = check_refusals(scorer)
        cohort = attach_token_lengths(frozen, scorer)
        longest = check_longest_eligible_batch(
            scorer, cohort, arm=arm, batch_size=int(batch_size)
        )
        facts = dict(getattr(scorer, "facts", {}))
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": STATUS,
            "phase": "probe",
            "kind": "synthetic_interface_check_not_dms_score",
            "arm": arm,
            "group": family,
            "scientific_role": facts.get("scientific_role")
            or (
                G.GALACTICA_CORPUS[arm]["note"]
                if arm in G.GALACTICA_CORPUS
                else RITA.RITA_CORPUS[arm]["note"]
            ),
            "probe_passed": True,
            "created_utc": _utc_now(),
            "settings": {
                "dtype": dtype,
                "batch_size": int(batch_size),
                "scoring_stratum": scorer.scoring_stratum,
                "score": scorer.score_description,
                "variant_seed": VARIANT_SEED,
                "variant_cap": VARIANT_CAP,
                "variant_draw": (
                    "seeded permutation of eligible rows, seed=20260807+original LOOKUP index"
                ),
            },
            "facts": facts,
            "runtime": _runtime_record(),
            "code_sources": _code_sources(),
            "fingerprints": {
                "lookup_sha256": sha256_file(lookup_path),
                "wildtypes_sha256": frozen["wildtypes_sha256"],
                "wildtypes_fasta_sha256": frozen["wildtypes_fasta_sha256"],
                "csv_sha256": cohort["csv_sha256"],
            },
            "synthetic_check": {**alignment, "refusals": refusals},
            "longest_eligible_batch": longest,
            "cohort": {
                "declared_assays": cohort["declared_assays"],
                "declared_clusters": cohort["declared_clusters"],
                "analysis_assays": cohort["analysis_assays"],
                "analysis_clusters": cohort["analysis_clusters"],
                "context": cohort["context"],
                "context_excluded_assays": cohort["context_excluded_assays"],
                "assays": cohort["assays"],
                "lookup_order": cohort["lookup_order"],
            },
            "skipped": cohort["skipped"],
            "blosum62_is_not_a_dms_gate": True,
            "no_new_model_fitness_scores": True,
        }
        destination = Path(out) / "interface_check.json"
        write_json(destination, payload)
        return payload
    finally:
        if owned and hasattr(scorer, "release"):
            scorer.release()


def _require_maps(
    probes: Mapping[str, Path], scores: Mapping[str, Path]
) -> tuple[dict[str, Path], dict[str, Path]]:
    if set(probes) != set(scores):
        raise ValueError(
            "probe and score names disagree: "
            f"{sorted(set(probes) ^ set(scores))}. Missing requested inputs are refused"
        )
    for name, path in list(probes.items()) + list(scores.items()):
        if not Path(path).is_file():
            raise FileNotFoundError(f"requested input {name}={path} does not exist")
    names = set(probes)
    gal = [name for name in GALACTICA_GROUP if name in names]
    if gal and list(gal) != list(GALACTICA_GROUP):
        raise ValueError(
            "Galactica is one common-support group; requested "
            f"{gal} but the group is {list(GALACTICA_GROUP)}. "
            "Partial ladders are not silently summarised"
        )
    extra = names - set(GALACTICA_GROUP) - set(RITA_GROUP)
    if extra:
        raise ValueError(f"unrecognised arms in analyse inputs: {sorted(extra)}")
    return dict(probes), dict(scores)


def _probe_census(probe: Mapping[str, Any], *, label: str) -> DmsCensus:
    cohort = probe["cohort"]
    return DmsCensus(
        label=label,
        declared_assays=int(cohort["declared_assays"]),
        declared_clusters=int(cohort["declared_clusters"]),
        analysis_assays=len(cohort["analysis_assays"]),
        analysis_clusters=int(cohort["analysis_clusters"]),
        context_excluded_assays=tuple(cohort["context_excluded_assays"]),
    )


def _require_mapping(payload: Mapping[str, Any], key: str, *, where: str) -> Mapping[str, Any]:
    if key not in payload or not isinstance(payload[key], Mapping):
        raise ValueError(f"{where} is missing {key}")
    return payload[key]


def _require_present(payload: Mapping[str, Any], key: str, *, where: str) -> Any:
    if key not in payload or payload[key] is None:
        raise ValueError(f"{where} is missing {key}")
    return payload[key]


def _verify_probe_against_score(
    name: str, probe: Mapping[str, Any], score: Mapping[str, Any]
) -> None:
    if probe.get("probe_passed") != True:
        raise ValueError(f"{name}: probe did not pass; analyse refuses it")
    if probe.get("arm") != name:
        raise ValueError(f"{name}: probe arm is {probe.get('arm')!r}")
    if score.get("arm") != name:
        raise ValueError(f"{name}: score arm is {score.get('arm')!r}")
    settings = _require_mapping(score, "settings", where=f"{name} score")
    probe_settings = _require_mapping(probe, "settings", where=f"{name} probe")
    required_dtype = REQUIRED_DTYPE[name]
    score_dtype = _require_present(settings, "dtype", where=f"{name} score settings")
    probe_dtype = _require_present(
        probe_settings, "dtype", where=f"{name} probe settings"
    )
    if score_dtype != required_dtype or probe_dtype != required_dtype:
        raise ValueError(
            f"{name}: dtype must be {required_dtype!r}; score has {score_dtype!r}, "
            f"probe has {probe_dtype!r}"
        )
    score_stratum = _require_present(
        settings, "scoring_stratum", where=f"{name} score settings"
    )
    probe_stratum = _require_present(
        probe_settings, "scoring_stratum", where=f"{name} probe settings"
    )
    if score_stratum != STRATUM_N_TO_C or probe_stratum != STRATUM_N_TO_C:
        raise ValueError(
            f"{name}: scoring_stratum must be {STRATUM_N_TO_C!r}; score has "
            f"{score_stratum!r}, probe has {probe_stratum!r}"
        )
    score_seed = _require_present(settings, "seed", where=f"{name} score settings")
    probe_seed = _require_present(
        probe_settings, "variant_seed", where=f"{name} probe settings"
    )
    if score_seed != VARIANT_SEED or probe_seed != VARIANT_SEED:
        raise ValueError(
            f"{name}: score seed {score_seed!r} / probe variant_seed {probe_seed!r} "
            f"must both equal {VARIANT_SEED}"
        )
    score_variants = _require_present(
        settings, "variants", where=f"{name} score settings"
    )
    probe_cap = _require_present(
        probe_settings, "variant_cap", where=f"{name} probe settings"
    )
    if score_variants != VARIANT_CAP or probe_cap != VARIANT_CAP:
        raise ValueError(
            f"{name}: score variants {score_variants!r} / probe variant_cap "
            f"{probe_cap!r} must both equal {VARIANT_CAP}"
        )
    score_batch = _require_present(
        settings, "batch_size", where=f"{name} score settings"
    )
    probe_batch = _require_present(
        probe_settings, "batch_size", where=f"{name} probe settings"
    )
    if score_batch != probe_batch:
        raise ValueError(
            f"{name}: score batch_size {score_batch!r} disagrees with probe "
            f"{probe_batch!r}"
        )
    probe_fp = _require_mapping(probe, "fingerprints", where=f"{name} probe")
    score_fp = _require_mapping(score, "input_fingerprints", where=f"{name} score")
    probe_wt = _require_present(
        probe_fp, "wildtypes_sha256", where=f"{name} probe fingerprints"
    )
    score_wt = _require_present(
        score_fp, "wildtypes_sha256", where=f"{name} score input_fingerprints"
    )
    if score_wt != probe_wt:
        raise ValueError(
            f"{name}: score wildtypes_sha256 disagrees with the probe"
        )
    by_probe = {row["assay"]: row for row in probe["cohort"]["assays"]}
    skipped = {row["assay"] for row in score.get("skipped") or []}
    expected_skip = set(probe["cohort"]["context_excluded_assays"])
    if skipped != expected_skip:
        raise ValueError(
            f"{name}: score skipped {sorted(skipped)} but probe excluded "
            f"{sorted(expected_skip)}. Support is not silently intersected"
        )
    scored = {row["assay"] for row in score["assays"]}
    expected_scored = set(probe["cohort"]["analysis_assays"])
    if scored != expected_scored:
        raise ValueError(
            f"{name}: score assays {sorted(scored ^ expected_scored)} disagree with the probe"
        )
    for row in score["assays"]:
        assay = row["assay"]
        expected = by_probe[assay]
        digest = _require_present(row, "mutant_digest", where=f"{name} {assay}")
        if digest != expected["mutant_digest"]:
            raise ValueError(f"{name} {assay}: mutant_digest disagrees with the probe")
        wildtype_id = _require_present(row, "wildtype_id", where=f"{name} {assay}")
        if wildtype_id != expected["wildtype_id"]:
            raise ValueError(f"{name} {assay}: wildtype_id disagrees with the probe")
        n_variants = _require_present(row, "n_variants", where=f"{name} {assay}")
        if n_variants != expected["n_variants"]:
            raise ValueError(f"{name} {assay}: n_variants disagrees with the probe")
        csv_hash = _require_present(row, "csv_sha256", where=f"{name} {assay}")
        if csv_hash != expected["csv_sha256"]:
            raise ValueError(
                f"{name} {assay}: csv_sha256 disagrees with the probe; "
                "mutant_digest does not bind full sequences or DMS labels"
            )
        _finite(float(row["spearman"]), what=f"{name} {assay} spearman")


_ASSAY_IDENTITY_FIELDS = (
    "csv_sha256",
    "wildtype_id",
    "wildtype_sequence",
    "n_variants",
    "mutant_digest",
    "cluster",
    "seed",
)


def _require_shared_probe_identity(
    names: Sequence[str], probes: Mapping[str, Mapping[str, Any]]
) -> None:
    first_name = names[0]
    first = probes[first_name]
    first_settings = _require_mapping(
        first, "settings", where=f"{first_name} probe"
    )
    first_fp = _require_mapping(
        first, "fingerprints", where=f"{first_name} probe"
    )
    first_rows = {row["assay"]: row for row in first["cohort"]["assays"]}
    for name in names[1:]:
        probe = probes[name]
        settings = _require_mapping(probe, "settings", where=f"{name} probe")
        fingerprints = _require_mapping(
            probe, "fingerprints", where=f"{name} probe"
        )
        for key in (
            "lookup_sha256",
            "wildtypes_sha256",
            "wildtypes_fasta_sha256",
            "csv_sha256",
        ):
            left = _require_present(
                first_fp, key, where=f"{first_name} probe fingerprints"
            )
            right = _require_present(
                fingerprints, key, where=f"{name} probe fingerprints"
            )
            if left != right:
                raise ValueError(
                    f"{name}: probe {key} disagrees with {first_name}"
                )
        for key in (
            "variant_seed",
            "variant_cap",
            "dtype",
            "scoring_stratum",
        ):
            if _require_present(
                settings, key, where=f"{name} probe settings"
            ) != _require_present(
                first_settings, key, where=f"{first_name} probe settings"
            ):
                raise ValueError(
                    f"{name}: probe setting {key} disagrees with {first_name}"
                )
        rows = {row["assay"]: row for row in probe["cohort"]["assays"]}
        if set(rows) != set(first_rows):
            raise ValueError(
                f"{name}: probe assay set disagrees with {first_name}"
            )
        for assay, expected in first_rows.items():
            got = rows[assay]
            for field in _ASSAY_IDENTITY_FIELDS:
                left = _require_present(
                    expected, field, where=f"{first_name} {assay}"
                )
                right = _require_present(got, field, where=f"{name} {assay}")
                if left != right:
                    raise ValueError(
                        f"{name} {assay}: {field} disagrees with {first_name}"
                    )


def _group_payload(
    names: Sequence[str],
    probes: Mapping[str, dict[str, Any]],
    scores: Mapping[str, dict[str, Any]],
    lookup: Mapping[str, Any],
    *,
    seed: int,
    pairs: Sequence[tuple[str, str]],
    label: str,
) -> dict[str, Any]:
    models = {name: scores[name] for name in names}
    require_uniform_dtype(models, rungs=names, label=label)
    stratum = require_uniform_stratum(models, rungs=names, label=label)
    first = probes[names[0]]
    for name in names[1:]:
        if probes[name]["cohort"]["context_excluded_assays"] != first["cohort"][
            "context_excluded_assays"
        ] or probes[name]["cohort"]["analysis_assays"] != first["cohort"]["analysis_assays"]:
            raise ValueError(
                f"{label}: {name} and {names[0]} disagree on support. "
                "A common-support group is not formed by taking an intersection"
            )
        if int(probes[name]["cohort"]["context"]) != int(first["cohort"]["context"]):
            raise ValueError(
                f"{label}: context disagrees between {name} and {names[0]}"
            )
    _require_shared_probe_identity(names, probes)
    for name in names:
        _verify_probe_against_score(name, probes[name], scores[name])
    census = _probe_census(first, label=label)
    alignment = align_dms(
        models,
        dict(lookup),
        rungs=names,
        context=int(first["cohort"]["context"]),
        exclusion_reason=ALIGN_EXCLUSION_REASON,
        census=census,
    )
    units = alignment["units"]
    lookup_by_assay = {row["assay"]: row for row in lookup["assays"]}
    lookup_raw = {
        assay: _finite(
            float(lookup_by_assay[assay]["spearman"]["lookup"]),
            what=f"{assay} LOOKUP Spearman",
        )
        for assay in alignment["analysis_assays"]
    }
    blosum_raw = {
        assay: _finite(
            float(lookup_by_assay[assay]["spearman"]["blosum62"]),
            what=f"{assay} BLOSUM62 Spearman",
        )
        for assay in alignment["analysis_assays"]
    }
    return {
        "rungs": list(names),
        "pairs": {
            "reference_only": [list(REFERENCE_PAIR)]
            if REFERENCE_PAIR in pairs
            else [],
            "qualified_ladder": [list(pair) for pair in pairs if pair in QUALIFIED_PAIRS],
            "all_computed": [list(pair) for pair in pairs],
        },
        "scoring_stratum": stratum,
        "dtype": scores[names[0]]["settings"]["dtype"],
        "declared_cohort": {
            "n_assays": len(alignment["declared_assays"]),
            "n_clusters": alignment["declared_clusters"],
        },
        "n_assays": len(alignment["analysis_assays"]),
        "n_clusters": len(set(units.values())),
        "context_excluded_assays": alignment["context_excluded_assays"],
        "unit": "wild-type family at 50% identity",
        "raw_spearman": endpoint(
            alignment["raw"],
            units,
            rungs=names,
            pairs=pairs,
            resamples=BOOTSTRAP_RESAMPLES,
            seed=seed + CHANNEL_OFFSETS["raw_spearman"],
        ),
        "model_minus_lookup": endpoint(
            alignment["contrasts"]["model_minus_lookup"],
            units,
            rungs=names,
            pairs=pairs,
            resamples=BOOTSTRAP_RESAMPLES,
            seed=seed + CHANNEL_OFFSETS["model_minus_lookup"],
        ),
        "model_minus_blosum62": endpoint(
            alignment["contrasts"]["model_minus_blosum62"],
            units,
            rungs=names,
            pairs=pairs,
            resamples=BOOTSTRAP_RESAMPLES,
            seed=seed + CHANNEL_OFFSETS["model_minus_blosum62"],
        ),
        "lookup_raw_mean": unit_mean_record(
            lookup_raw,
            units,
            resamples=BOOTSTRAP_RESAMPLES,
            seed=seed + CHANNEL_OFFSETS["lookup_raw"],
        ),
        "blosum62_raw_mean": unit_mean_record(
            blosum_raw,
            units,
            resamples=BOOTSTRAP_RESAMPLES,
            seed=seed + CHANNEL_OFFSETS["blosum62_raw"],
        ),
        "per_assay_spearman": alignment["raw"],
        "blosum62_is_not_a_dms_gate": True,
        "no_acquired_or_retrieval_bounded_verdict": True,
        "descriptive_not_causal": True,
    }


def run_analyse(
    *,
    lookup_path: Path,
    probes: Mapping[str, Path],
    scores: Mapping[str, Path],
    out: Path,
) -> dict[str, Any]:
    probe_paths, score_paths = _require_maps(probes, scores)
    lookup = _read_json(lookup_path)
    loaded_probes = {name: _read_json(path) for name, path in probe_paths.items()}
    loaded_scores = {name: _read_json(path) for name, path in score_paths.items()}
    lookup_hash = sha256_file(lookup_path)
    for name, probe in loaded_probes.items():
        fingerprints = _require_mapping(probe, "fingerprints", where=f"{name} probe")
        stored = _require_present(
            fingerprints, "lookup_sha256", where=f"{name} probe fingerprints"
        )
        if stored != lookup_hash:
            raise ValueError(
                f"{name}: probe lookup_sha256 {stored} disagrees with the current "
                f"LOOKUP file {lookup_hash}; replacing LOOKUP values while keeping "
                "assay identifiers is refused"
            )
    groups: dict[str, Any] = {}
    names = set(probe_paths)
    if set(GALACTICA_GROUP) <= names:
        groups["galactica"] = _group_payload(
            GALACTICA_GROUP,
            loaded_probes,
            loaded_scores,
            lookup,
            seed=BOOTSTRAP_SEED,
            pairs=GALACTICA_PAIRS,
            label="native_dms_extension_predata_galactica",
        )
        groups["galactica"]["reference_pair_note"] = (
            "galactica-125m → galactica-1.3b is a small-scale dual-mode-unidentified "
            "reference pair, not a qualified ladder step and not a parameter-count effect"
        )
    if RITA.RITA_ARM in names:
        groups[RITA.RITA_ARM] = _group_payload(
            RITA_GROUP,
            loaded_probes,
            loaded_scores,
            lookup,
            seed=BOOTSTRAP_SEED + RITA_SEED_OFFSET,
            pairs=(),
            label="native_dms_extension_predata_rita",
        )
    if not groups:
        raise ValueError("analyse was given no complete Galactica or RITA group")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "phase": "analyse",
        "created_utc": _utc_now(),
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "channel_offsets": dict(CHANNEL_OFFSETS),
            "rita_group_offset": RITA_SEED_OFFSET,
            "endpoint_positional_offsets": (
                "rung i draws at seed+i; pair j draws at seed+10+j, as in "
                "src.transfer.scale_comparison.endpoint"
            ),
        },
        "inputs": {
            "lookup_sha256": sha256_file(lookup_path),
            "probes": {name: sha256_file(path) for name, path in probe_paths.items()},
            "scores": {name: sha256_file(path) for name, path in score_paths.items()},
        },
        "groups": groups,
        "blosum62_is_not_a_dms_gate": True,
        "no_acquired_or_retrieval_bounded_verdict": True,
        "no_cross_stratum_ranking": True,
        "no_parameter_count_causal_claim": True,
        "descriptive_not_causal": True,
    }
    write_json(Path(out) / "native_dms_comparison.json", payload)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("probe", "analyse"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lookup", type=Path, required=True)
    parser.add_argument("--proteingym-dir", type=Path, default=None)
    parser.add_argument(
        "--wildtypes",
        type=Path,
        default=None,
        help="probe: path to the cell wildtypes.json catalogue",
    )
    parser.add_argument(
        "--wildtypes-fasta",
        type=Path,
        default=None,
        help="probe: path to the cell wildtypes.faa sequences",
    )
    parser.add_argument("--arm")
    parser.add_argument("--dtype")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--probe",
        action="append",
        default=[],
        help="analyse: NAME=PATH of a probe interface_check.json",
    )
    parser.add_argument(
        "--score",
        action="append",
        default=[],
        help="analyse: NAME=PATH of a stage-20 model score JSON",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lookup = Path(args.lookup)
    if args.phase == "probe":
        if not args.arm or not args.dtype:
            raise ValueError("probe requires --arm and --dtype")
        if args.wildtypes is None:
            raise ValueError("probe requires --wildtypes")
        if args.wildtypes_fasta is None:
            raise ValueError("probe requires --wildtypes-fasta")
        proteingym = Path(args.proteingym_dir) if args.proteingym_dir else PROTEINGYM_ROOT
        return run_probe(
            arm=args.arm,
            lookup_path=lookup,
            proteingym_dir=proteingym,
            wildtypes_path=Path(args.wildtypes),
            wildtypes_fasta_path=Path(args.wildtypes_fasta),
            out=out,
            dtype=args.dtype,
            batch_size=int(args.batch_size),
            device=str(args.device),
        )
    probes = dict(_parse_named_path(value) for value in args.probe)
    scores = dict(_parse_named_path(value) for value in args.score)
    if not probes or not scores:
        raise ValueError("analyse requires --probe NAME=PATH and --score NAME=PATH")
    return run_analyse(
        lookup_path=lookup, probes=probes, scores=scores, out=out
    )


if __name__ == "__main__":
    main()
