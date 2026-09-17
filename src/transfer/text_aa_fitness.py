"""FP32 text amino-acid string-control scorer.

This is a dedicated scoring core, not a panel door and not a stage-20 wire-up.
The estimand is the sum of conditional log-likelihoods of every target token of
a raw uppercase AA20 string after the document-boundary token its own family
declares -- or after nothing, for a family whose measured rendering is the byte
string itself. Mutant minus WT is applied upstream. There is no length
normalisation, no biological prompt, and no trailing EOS in the scored path.

Every scored span must carry exactly one token per residue. A text tokenizer
that merges neighbouring residues into multi-residue pieces is refused inside
:func:`encode_text_aa` -- the same rule ``joint_modes.verify_one_token_per_residue``
enforces per rendering -- so neither the scorer nor the census can measure a
merged unit. The property belongs to the encoding and not to the checkpoint, so
a string that does encode one-to-one is still permitted even on a tokenizer that
merges elsewhere; nothing re-splits, truncates, or substitutes a different unit.

Usage (core only; not a legal ProteinGym run until a later stage is wired)::

    from src.transfer.text_aa_fitness import load_text_aa_scorer

    scorer = load_text_aa_scorer(
        checkpoint,
        boundary=spec,  # resolved per-checkpoint mapping
        application_window_tokens=window,  # required; not a guessed hard cap
        device="cpu",
    )
    lengths = scorer.token_lengths(sequences)
    totals = scorer.log_likelihood(sequences)  # shape (n,), float64, batch_size 1

``application_window_tokens`` is the declared scoring envelope, not
``config.max_length`` and not a stage-01 budget. ByGPT5 has T5 relative
position bias and no absolute position table, so it has no hard context from
embeddings; a window is still required. Tokenizer loading is shared with the
census path and does not load weights. This module does not freeze that
window, run a census, or admit an experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .amino_acids import AA20
from .arms import (
    require_clean_loading_info,
    require_input_path,
    unpack_pretrained_loading_info,
)
from .precision_policy import (
    FP32_PER_TARGET_ABS,
    TEXT_AA_FP32_V1,
    fp32_matmul_context,
)
from .scale_comparison import STRATUM_N_TO_C

__all__ = [
    "BUILTIN_MODEL_TYPES",
    "BYGPT5_MODEL_TYPE",
    "CHECKPOINT_VARIABLE",
    "FP32_PER_TARGET_ABS",
    "HARD_CONTEXT_ATTRIBUTES",
    "NO_CONDITIONING_KIND",
    "TEXT_AA_FP32_V1",
    "TextAABoundary",
    "TextAAEncodingError",
    "TextAAFitnessScorer",
    "TextAATokenizerBundle",
    "encode_text_aa",
    "load_text_aa_scorer",
    "load_text_aa_tokenizer",
    "read_hard_context",
    "resolve_text_aa_boundary",
]

CHECKPOINT_VARIABLE = "TRANSFER_TEXT_MODEL_BASE_DIR"
BUILTIN_MODEL_TYPES = frozenset({"gpt2", "qwen2", "qwen3", "llama"})
BYGPT5_MODEL_TYPE = "bygpt5"
_ALLOWED_MODEL_TYPES = BUILTIN_MODEL_TYPES | {BYGPT5_MODEL_TYPE}
HARD_CONTEXT_ATTRIBUTES = (
    "n_positions",
    "max_position_embeddings",
    "max_seq_len",
    "seq_length",
)
_BYGPT5_AUTOMAP = {
    "AutoConfig": "configuration_bygpt5.ByGPT5Config",
    "AutoModelForCausalLM": "modeling_bygpt5.ByGPT5LMHeadModel",
}
_BYGPT5_LOCAL_FILES = (
    "configuration_bygpt5.py",
    "modeling_bygpt5.py",
    "tokenization_bygpt5.py",
)
_PRODUCTION_BATCH_SIZE = 1

#: ``conditioning.kind`` for a family whose rendering prefixes no conditioning
#: token at all. Spelled once: the boundary table declares it, the parser requires
#: a null id under it, and :func:`encode_text_aa` returns the target ids alone.
NO_CONDITIONING_KIND = "none_declared"


class TextAAEncodingError(ValueError):
    """Declared sequence restriction. Census may exclude the whole assay.

    Empty strings, characters outside AA20, empty targets, UNK or special
    target ids, and failed strict round-trips use this type. Tokenizer
    internals, schema, configuration, memory, and unknown backend failures
    must not be converted to it.
    """


def _as_int(value: Any, *, what: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} must be an int, got {value!r}") from exc


def _as_optional_int(value: Any, *, what: str) -> int | None:
    if value is None:
        return None
    return _as_int(value, what=what)


def _id_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        ids: list[int] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
                continue
            ids.append(int(item))
        return ids
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return [int(value)]
    return []


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def read_hard_context(config: Any) -> int | None:
    """Return a declared absolute position table size, or None.

    ByGPT5 declares none of these keys. ``max_length`` is a generation default
    and is not consulted.
    """

    for name in HARD_CONTEXT_ATTRIBUTES:
        value = getattr(config, name, None)
        if value is not None:
            return _as_int(value, what=name)
    return None


@dataclass(frozen=True)
class TextAABoundary:
    """Frozen document-boundary contract for one text-AA checkpoint family.

    ``conditioning_id`` is the token the rendering prefixes, and it is ``None``
    exactly when ``conditioning_kind`` is :data:`NO_CONDITIONING_KIND`: the family
    declares that its rendering carries no conditioning prefix, so
    :func:`encode_text_aa` returns the target ids alone. That is a declaration and
    not a missing field -- a boundary whose ``conditioning`` mapping is absent
    altogether is refused -- and a declared prefix is never chosen from a config
    default: ByGPT5's ``decoder_start_token_id`` (0, ``<pad>``) is T5Config's own
    default carried into a decoder-only config, and the rendering measured on that
    checkpoint scores lower without it (0.88939 nats/token against 0.90865 over
    200 OpenWebText documents of at least 800 characters at the 384-token window,
    draw seed 20260728, float32).
    """

    name: str
    model_type: str
    tokenizer_class: str
    conditioning_kind: str
    conditioning_id: int | None
    conditioning_token: str | None
    bos_token_id: int | None
    eos_token_id: int | None
    pad_token_id: int | None
    unk_token_id: int | None
    decoder_start_token_id: int | None
    documented_context: int | None
    forbid_token_ids: tuple[int, ...]
    trust_remote_code: bool

    @classmethod
    def from_mapping(cls, spec: Mapping[str, Any], *, name: str) -> "TextAABoundary":
        if not isinstance(spec, Mapping):
            raise ValueError(f"{name}: boundary spec must be a mapping")
        model_type = str(spec.get("model_type") or "")
        if model_type not in _ALLOWED_MODEL_TYPES:
            raise ValueError(
                f"{name}: unsupported model_type {model_type!r}; "
                f"allowed {sorted(_ALLOWED_MODEL_TYPES)}"
            )
        conditioning = spec.get("conditioning")
        if not isinstance(conditioning, Mapping):
            raise ValueError(f"{name}: boundary spec is missing conditioning")
        kind = str(conditioning.get("kind") or "")
        if not kind:
            raise ValueError(f"{name}: conditioning.kind is required")
        if conditioning.get("id") is None:
            if kind != NO_CONDITIONING_KIND:
                raise ValueError(
                    f"{name}: conditioning.id is required, and is null only under "
                    f"conditioning.kind {NO_CONDITIONING_KIND!r}, which declares "
                    "that this family's rendering prefixes no conditioning token"
                )
            conditioning_id: int | None = None
            conditioning_token: str | None = None
        else:
            if kind == NO_CONDITIONING_KIND:
                raise ValueError(
                    f"{name}: {NO_CONDITIONING_KIND!r} declares that no "
                    "conditioning token is prefixed, so it cannot also carry one"
                )
            conditioning_id = _as_int(conditioning.get("id"), what="conditioning.id")
            conditioning_token = str(conditioning.get("token") or "")
        forbid = tuple(
            _id_list(conditioning.get("do_not_use"))
            + _id_list(spec.get("forbid_token_ids"))
        )
        tokenizer_bos = spec.get("bos_token_id_tokenizer", spec.get("bos_token_id"))
        trust = model_type == BYGPT5_MODEL_TYPE
        if spec.get("trust_remote_code") and not trust:
            raise ValueError(
                f"{name}: trust_remote_code is only allowed for {BYGPT5_MODEL_TYPE}"
            )
        return cls(
            name=str(name),
            model_type=model_type,
            tokenizer_class=str(spec.get("tokenizer_class") or ""),
            conditioning_kind=kind,
            conditioning_id=conditioning_id,
            conditioning_token=conditioning_token,
            bos_token_id=_as_optional_int(tokenizer_bos, what="bos_token_id"),
            eos_token_id=_as_optional_int(spec.get("eos_token_id"), what="eos_token_id"),
            pad_token_id=_as_optional_int(spec.get("pad_token_id"), what="pad_token_id"),
            unk_token_id=_as_optional_int(spec.get("unk_token_id"), what="unk_token_id"),
            decoder_start_token_id=_as_optional_int(
                spec.get("decoder_start_token_id"),
                what="decoder_start_token_id",
            ),
            documented_context=_as_optional_int(
                spec.get("context"), what="documented context"
            ),
            forbid_token_ids=forbid,
            trust_remote_code=trust,
        )


@dataclass(frozen=True)
class TextAATokenizerBundle:
    """Tokenizer, config, and hard context for one text-AA checkpoint.

    Census uses this bundle and never constructs a scorer. Scoring still requires
    a separate float32 weight load.
    """

    tokenizer: Any
    config: Any
    boundary: TextAABoundary
    hard_context: int | None


def resolve_text_aa_boundary(name: str, table: Mapping[str, Any]) -> TextAABoundary:
    """Resolve ``inherits_boundary_from`` against a boundary table."""

    checkpoints = table.get("text_aa_checkpoints", table)
    if not isinstance(checkpoints, Mapping):
        raise ValueError("boundary table is missing text_aa_checkpoints")
    chain: list[Mapping[str, Any]] = []
    current: str | None = str(name)
    seen: set[str] = set()
    while current:
        if current in seen:
            raise ValueError(f"boundary inheritance cycle at {current!r}")
        seen.add(current)
        if current not in checkpoints:
            raise KeyError(f"unknown text-AA checkpoint {current!r}")
        spec = checkpoints[current]
        if not isinstance(spec, Mapping):
            raise ValueError(f"{current}: boundary spec must be a mapping")
        chain.append(spec)
        parent = spec.get("inherits_boundary_from")
        current = None if parent is None else str(parent)
    merged: dict[str, Any] = {}
    for spec in reversed(chain):
        merged = _deep_merge(merged, spec)
    return TextAABoundary.from_mapping(merged, name=name)


def _require_aa20(sequence: str) -> str:
    if not isinstance(sequence, str):
        raise TypeError(f"sequence must be a str, got {type(sequence)!r}")
    if sequence == "":
        raise TextAAEncodingError("empty amino-acid string is refused")
    illegal = sorted({character for character in sequence if character not in AA20})
    if illegal:
        raise TextAAEncodingError(
            "characters outside AA20 are refused before tokenisation "
            f"(including Z/J/whitespace): {illegal!r}"
        )
    return sequence


def _as_id_list(value: Any, *, what: str) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        raise ValueError(f"{what}: batched tokenizer output is refused")
    try:
        return [int(item) for item in value]
    except TypeError as exc:
        raise ValueError(f"{what}: tokenizer did not return a list of ids") from exc


def _special_ids(tokenizer: Any) -> set[int]:
    raw = getattr(tokenizer, "all_special_ids", None) or []
    return {int(item) for item in raw}


def encode_text_aa(
    tokenizer: Any,
    sequence: str,
    boundary: TextAABoundary,
) -> list[int]:
    """Prefix a declared conditioning token onto reversible AA20 target ids.

    The returned span must carry exactly one target token per residue. A
    tokenizer that merges neighbouring residues into multi-residue pieces is
    refused here, before any caller can score or count the merged unit, because
    a mutation changes the segmentation and token position stops being residue
    position. This is the rule ``joint_modes.verify_one_token_per_residue``
    already applies to the joint checkpoints.

    A boundary declaring :data:`NO_CONDITIONING_KIND` returns the target ids
    alone: its rendering is the byte string itself, and the first of those ids is
    then the sequence's first residue, which the scorer treats as context exactly
    as it treats a declared prefix. That declaration is not a missing
    conditioning -- the parser refuses a boundary that carries none.
    """

    sequence = _require_aa20(sequence)
    encoded = tokenizer(sequence, add_special_tokens=False)
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise ValueError("tokenizer did not return input_ids")
    target_ids = _as_id_list(encoded["input_ids"], what="target ids")
    if not target_ids:
        raise TextAAEncodingError("tokenizer produced no target tokens")
    forbidden = _special_ids(tokenizer)
    if boundary.conditioning_id is not None:
        forbidden.add(int(boundary.conditioning_id))
    if boundary.unk_token_id is not None:
        forbidden.add(int(boundary.unk_token_id))
    forbidden.update(boundary.forbid_token_ids)
    hits = [token_id for token_id in target_ids if token_id in forbidden]
    if hits:
        raise TextAAEncodingError(
            "target ids contain UNK, special, conditioning, or forbidden ids: "
            f"{hits!r}"
        )
    decoded = tokenizer.decode(target_ids, skip_special_tokens=False)
    if decoded != sequence:
        raise TextAAEncodingError(
            "decode(target_ids, skip_special_tokens=False) must equal the raw "
            f"AA20 string; got {decoded!r}"
        )
    if len(target_ids) != len(sequence):
        raise TextAAEncodingError(
            f"{boundary.name}: the text tokenizer produced {len(target_ids)} "
            f"scored tokens for {len(sequence)} residues "
            f"({len(sequence) / len(target_ids):.3f} residues per token). It "
            "merged residues into multi-residue pieces, so token position is "
            "not residue position and the summed log-likelihood is not the "
            "per-residue quantity this protocol declares -- measured at about "
            "2.9 nats/token on galactica-1.3b (EXP-R2-151, Appendix B rule 4). "
            "Re-splitting or truncating would score a different unit, so this "
            "encoding is refused by the same rule "
            "joint_modes.verify_one_token_per_residue applies"
        )
    prefix = boundary.conditioning_id
    if prefix is None:
        return list(target_ids)
    if prefix in target_ids:
        raise TextAAEncodingError("conditioning id must not appear in target ids")
    return [int(prefix), *target_ids]


def _require_application_window(value: Any) -> int:
    window = _as_int(value, what="application_window_tokens")
    if window < 2:
        raise ValueError(
            "application_window_tokens is required and must be >= 2 "
            f"(prefix plus at least one target); got {window}"
        )
    return window


def _check_special_id(actual: Any, expected: int | None, *, what: str) -> None:
    observed = _as_optional_int(actual, what=what)
    if observed != expected:
        raise ValueError(f"{what}: expected {expected!r}, got {observed!r}")


def _require_bygpt5_local_code(path: Path, config: Any) -> None:
    auto_map = getattr(config, "auto_map", None)
    if not isinstance(auto_map, Mapping):
        raise ValueError("ByGPT5 checkpoint must declare a local auto_map")
    for key, expected in _BYGPT5_AUTOMAP.items():
        got = auto_map.get(key)
        if got != expected:
            raise ValueError(
                f"ByGPT5 auto_map[{key!r}] must be {expected!r}, got {got!r}"
            )
    missing = [
        name for name in _BYGPT5_LOCAL_FILES if not (path / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "ByGPT5 remote code is only the already-vendored local files; "
            f"missing {missing} under {path}"
        )


def _observed_floating_dtypes(model: Any) -> list[str]:
    return sorted(
        {
            str(parameter.dtype).removeprefix("torch.")
            for parameter in model.parameters()
            if parameter.is_floating_point()
        }
    )


def _declared_storage_dtype(config: Any) -> str:
    """The dtype the checkpoint itself declares, or ``"undeclared"``.

    Read from the config's instance dict because ``dtype`` is the current
    transformers spelling and ``torch_dtype`` the older one, and because reading
    the attribute would log a deprecation warning for a value that is also the
    storage dtype. Absent keys mean the checkpoint declares nothing, which is
    the GPT-2 family's normal case and not a float32 claim.
    """

    fields = getattr(config, "__dict__", None)
    if not isinstance(fields, dict):
        return "undeclared"
    for name in ("dtype", "torch_dtype"):
        value = fields.get(name)
        if value is not None:
            return str(value).removeprefix("torch.")
    return "undeclared"


def _refuse_autocast(torch_mod: Any) -> None:
    enabled = bool(torch_mod.is_autocast_enabled())
    cuda_enabled = False
    cuda = getattr(torch_mod, "cuda", None)
    checker = getattr(cuda, "is_autocast_enabled", None) if cuda is not None else None
    if callable(checker):
        cuda_enabled = bool(checker())
    if enabled or cuda_enabled:
        raise RuntimeError("silent autocast is refused")


class TextAAFitnessScorer:
    """Summed per-residue log-likelihood under a declared text-AA boundary.

    Production scoring is always ``batch_size=1``. A public call may contain
    several sequences, including mixed lengths, repeats, and the empty list;
    each non-empty item is encoded and scored on its own. Padding is not used,
    so GPT-2/Qwen conditioning ids that coincide with EOS are never treated as
    pad. Spearman is not computed here.
    """

    score_description = (
        "raw uppercase AA20 string after one declared document-boundary token; "
        "summed conditional log-likelihood of every target token, which is "
        "exactly one per residue, no length normalisation, prefix and trailing "
        "EOS not scored"
    )
    scoring_stratum = STRATUM_N_TO_C

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        boundary: TextAABoundary,
        application_window_tokens: int,
        facts: Mapping[str, Any],
        torch_mod: Any | None = None,
    ) -> None:
        import torch

        self.torch = torch if torch_mod is None else torch_mod
        self.model = model
        self.tokenizer = tokenizer
        self.boundary = boundary
        self.application_window_tokens = _require_application_window(
            application_window_tokens
        )
        hard = facts.get("hard_context")
        self.hard_context = None if hard is None else _as_int(hard, what="hard_context")
        if (
            self.hard_context is not None
            and self.application_window_tokens > self.hard_context
        ):
            raise ValueError(
                "application_window_tokens "
                f"{self.application_window_tokens} exceeds this checkpoint's "
                f"hard context {self.hard_context}"
            )
        observed = _observed_floating_dtypes(model)
        if observed != ["float32"]:
            raise ValueError(
                "text-AA scoring runs in float32, so every floating parameter "
                f"of the bound model must be float32; observed {observed}. This "
                "checks the model handed to the scorer, not how the checkpoint "
                "was stored: a BF16/FP16-stored checkpoint that the loader "
                "upcast is permitted, and its declared storage dtype is "
                "recorded as facts['storage_torch_dtype']"
            )
        vocab = int(model.get_input_embeddings().num_embeddings)
        self.vocab_size = vocab
        self.facts = dict(facts)
        self.facts.setdefault("protocol", TEXT_AA_FP32_V1)
        self.facts.setdefault("production_batch_size", _PRODUCTION_BATCH_SIZE)
        self.facts.setdefault("application_window_tokens", self.application_window_tokens)
        self.facts.setdefault("experiment_admitted", False)

    def encode(self, sequence: str) -> list[int]:
        return encode_text_aa(self.tokenizer, sequence, self.boundary)

    def encoding_metadata(self, sequence: str) -> dict[str, Any]:
        ids = self.encode(sequence)
        return {
            "n_residues": len(sequence),
            # Read off the ids rather than restated from the declaration: the span
            # carries one token per residue, so what is left is the prefix, and
            # that is zero on a family declaring none.
            "n_prefix_tokens": len(ids) - len(sequence),
            "n_scored_tokens": len(ids) - 1,
            "n_input_tokens": len(ids),
            "conditioning_id": (
                None if self.boundary.conditioning_id is None
                else int(self.boundary.conditioning_id)
            ),
            "conditioning_kind": self.boundary.conditioning_kind,
            "trailing_eos_scored": False,
            "prefix_scored": False,
        }

    def token_lengths(self, sequences: Sequence[str]) -> list[int]:
        """Full prefix-plus-target length spent against the application window."""

        return [len(self.encode(str(sequence))) for sequence in sequences]

    def log_likelihood(self, sequences: Sequence[str]) -> np.ndarray:
        items = [str(sequence) for sequence in sequences]
        if not items:
            return np.zeros(0, dtype=np.float64)
        _refuse_autocast(self.torch)
        totals = np.empty(len(items), dtype=np.float64)
        torch = self.torch
        with fp32_matmul_context(torch):
            with torch.no_grad():
                for index, sequence in enumerate(items):
                    ids = self.encode(sequence)
                    width = len(ids)
                    if width > self.application_window_tokens:
                        raise ValueError(
                            f"{self.boundary.name}: encoding of {width} tokens "
                            "exceeds application_window_tokens "
                            f"{self.application_window_tokens}; truncating "
                            "would drop a mutation"
                        )
                    if self.hard_context is not None and width > self.hard_context:
                        raise ValueError(
                            f"{self.boundary.name}: encoding of {width} tokens "
                            f"exceeds hard context {self.hard_context}"
                        )
                    totals[index] = self._score_one(ids)
        if not np.all(np.isfinite(totals)):
            raise RuntimeError(
                f"{self.boundary.name}: a scored sequence returned a non-finite total"
            )
        return totals

    def _score_one(self, ids: list[int]) -> float:
        torch = self.torch
        device = self.model.device
        tokens = torch.tensor([ids], dtype=torch.long, device=device)
        # Position mask, not id==pad. Conditioning may equal pad (ByGPT5) or EOS
        # (GPT-2/Qwen); those positions still attend.
        mask = torch.ones((1, len(ids)), dtype=torch.long, device=device)
        outputs = self.model(
            input_ids=tokens, attention_mask=mask, use_cache=False
        )
        logits = outputs.logits
        if logits.dtype != torch.float32:
            raise RuntimeError(
                f"{self.boundary.name}: logits dtype {logits.dtype} is not float32"
            )
        expected = (1, len(ids), self.vocab_size)
        if tuple(logits.shape) != expected:
            raise RuntimeError(
                f"{self.boundary.name}: expected logits {expected}, "
                f"got {tuple(logits.shape)}"
            )
        logp = torch.log_softmax(logits[:, :-1], dim=-1)
        gathered = logp.gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
        total = float(gathered.sum().double().cpu().item())
        del outputs, logits, logp, gathered, tokens, mask
        return total

    def release(self) -> None:
        used_cuda = getattr(self.model.device, "type", None) == "cuda"
        del self.model
        del self.tokenizer
        if used_cuda:
            self.torch.cuda.empty_cache()


def _relative_position_facts(config: Any, *, model_type: str) -> dict[str, Any]:
    if model_type != BYGPT5_MODEL_TYPE:
        return {
            "position_encoding": "absolute_or_rotary",
            "relative_attention_num_buckets": None,
            "relative_attention_max_distance": None,
        }
    buckets = getattr(config, "relative_attention_num_buckets", None)
    distance = getattr(config, "relative_attention_max_distance", None)
    return {
        "position_encoding": "t5_relative_bias",
        "relative_attention_num_buckets": (
            None if buckets is None else int(buckets)
        ),
        "relative_attention_max_distance": (
            None if distance is None else int(distance)
        ),
        "hard_context_note": (
            "ByGPT5Stack uses T5 relative attention bias on block 0 and has no "
            "absolute position table. Distances beyond relative_attention_max_distance "
            "share a bucket; that is not a hard cap and not a scoring window. "
            "config.max_length is a generation default and is not used."
        ),
    }


def load_text_aa_tokenizer(
    checkpoint: Path | str,
    *,
    boundary: TextAABoundary | Mapping[str, Any],
    name: str | None = None,
) -> TextAATokenizerBundle:
    """Load tokenizer and config only. Weights are not read.

    Built-in GPT-2 / Qwen / Llama classes use ``trust_remote_code=False``.
    ByGPT5 is the only remote-code path, and only the already-vendored local
    ``configuration_bygpt5`` / ``modeling_bygpt5`` / ``tokenization_bygpt5``
    files are accepted. The checkpoint directory is not modified.
    """

    from transformers import AutoConfig, AutoTokenizer

    resolved = require_input_path(Path(checkpoint).resolve(), CHECKPOINT_VARIABLE)
    label = name or resolved.name
    if isinstance(boundary, TextAABoundary):
        spec = boundary
    else:
        spec = TextAABoundary.from_mapping(boundary, name=label)
    trust = bool(spec.trust_remote_code)
    if trust and spec.model_type != BYGPT5_MODEL_TYPE:
        raise ValueError("trust_remote_code is only allowed for ByGPT5")
    if spec.model_type not in _ALLOWED_MODEL_TYPES:
        raise ValueError(f"unsupported model_type {spec.model_type!r}")

    config = AutoConfig.from_pretrained(
        str(resolved),
        local_files_only=True,
        trust_remote_code=trust,
    )
    model_type = str(getattr(config, "model_type", "") or "")
    if model_type != spec.model_type:
        raise ValueError(
            f"{label}: expected model_type {spec.model_type!r}, got {model_type!r}"
        )
    auto_map = getattr(config, "auto_map", None)
    if spec.model_type == BYGPT5_MODEL_TYPE:
        _require_bygpt5_local_code(resolved, config)
    elif auto_map:
        raise ValueError(
            f"{label}: built-in text-AA checkpoints must not require auto_map"
        )

    hard_context = read_hard_context(config)
    tokenizer = AutoTokenizer.from_pretrained(
        str(resolved),
        local_files_only=True,
        trust_remote_code=trust,
    )
    tokenizer_class = type(tokenizer).__name__
    if spec.tokenizer_class and tokenizer_class != spec.tokenizer_class:
        aliases = {
            "GPT2TokenizerFast": {"GPT2Tokenizer", "GPT2TokenizerFast"},
            "Qwen2TokenizerFast": {"Qwen2Tokenizer", "Qwen2TokenizerFast"},
        }
        allowed = aliases.get(spec.tokenizer_class, {spec.tokenizer_class})
        if tokenizer_class not in allowed:
            raise ValueError(
                f"{label}: expected tokenizer class {spec.tokenizer_class!r}, "
                f"got {tokenizer_class!r}"
            )
    _check_special_id(tokenizer.bos_token_id, spec.bos_token_id, what="bos_token_id")
    _check_special_id(tokenizer.eos_token_id, spec.eos_token_id, what="eos_token_id")
    _check_special_id(tokenizer.pad_token_id, spec.pad_token_id, what="pad_token_id")
    _check_special_id(tokenizer.unk_token_id, spec.unk_token_id, what="unk_token_id")
    if spec.decoder_start_token_id is not None:
        # Checked against the loaded config as the checkpoint's own declared start
        # token, and deliberately NOT required to equal the conditioning id. On
        # ByGPT5 the two are different quantities: the config key is T5Config's
        # default 0 carried into a decoder-only config, while the conditioning id
        # is the token the rendering prefixes, and no measurement supports that
        # token there.
        config_start = getattr(config, "decoder_start_token_id", None)
        _check_special_id(
            config_start,
            spec.decoder_start_token_id,
            what="decoder_start_token_id",
        )
    if (
        spec.conditioning_id is not None
        and spec.conditioning_id in spec.forbid_token_ids
    ):
        raise ValueError(f"{label}: conditioning id is forbidden")
    if spec.model_type in {"qwen2", "qwen3"} and spec.bos_token_id is not None:
        raise ValueError(
            f"{label}: Qwen text-AA conditioning is the EOS document separator; "
            "tokenizer BOS must be absent"
        )
    return TextAATokenizerBundle(
        tokenizer=tokenizer,
        config=config,
        boundary=spec,
        hard_context=hard_context,
    )


def load_text_aa_scorer(
    checkpoint: Path | str,
    *,
    boundary: TextAABoundary | Mapping[str, Any],
    application_window_tokens: int,
    device: str = "cpu",
    name: str | None = None,
) -> TextAAFitnessScorer:
    """Load one local checkpoint in float32 and bind a text-AA boundary.

    Built-in GPT-2 / Qwen / Llama classes are loaded with
    ``trust_remote_code=False``. ByGPT5 is the only remote-code path, and only
    the already-vendored local ``configuration_bygpt5`` /
    ``modeling_bygpt5`` / ``tokenization_bygpt5`` files are accepted. Weights
    are not downloaded and the checkpoint directory is not modified.

    The checkpoint is read with ``torch_dtype=torch.float32``, so a checkpoint
    stored in BF16/FP16 is upcast and scored as float32. What is enforced is the
    loaded parameter dtype -- exactly ``float32`` -- not the storage dtype, which
    is recorded as ``facts['storage_torch_dtype']`` instead of being refused.
    """

    import torch
    from transformers import AutoModelForCausalLM

    resolved = require_input_path(Path(checkpoint).resolve(), CHECKPOINT_VARIABLE)
    label = name or resolved.name
    bundle = load_text_aa_tokenizer(resolved, boundary=boundary, name=label)
    spec = bundle.boundary
    config = bundle.config
    tokenizer = bundle.tokenizer
    hard_context = bundle.hard_context
    window = _require_application_window(application_window_tokens)
    if hard_context is not None and window > hard_context:
        raise ValueError(
            f"{label}: application_window_tokens {window} exceeds hard context "
            f"{hard_context}"
        )
    trust = bool(spec.trust_remote_code)
    model_type = str(getattr(config, "model_type", "") or "")
    tokenizer_class = type(tokenizer).__name__
    storage_dtype = _declared_storage_dtype(config)

    loaded = AutoModelForCausalLM.from_pretrained(
        str(resolved),
        torch_dtype=torch.float32,
        device_map={"": device},
        output_loading_info=True,
        local_files_only=True,
        trust_remote_code=trust,
    )
    model, loading_info = unpack_pretrained_loading_info(loaded)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    diagnostics = require_clean_loading_info(loading_info, arm=label)
    observed = _observed_floating_dtypes(model)
    if observed != ["float32"]:
        raise ValueError(
            f"{label}: requested float32 from_pretrained, observed {observed}"
        )
    facts = {
        "checkpoint": str(resolved),
        "name": label,
        "protocol": TEXT_AA_FP32_V1,
        "research_role": "text-aa-string-control",
        "scientific_role": (
            "text amino-acid string control; not native protein semantics, "
            "not a panel admission, and not an experiment PASS"
        ),
        "device": device,
        "dtype_requested": "float32",
        "dtype_observed": observed,
        "storage_torch_dtype": storage_dtype,
        "tokenizer_class": tokenizer_class,
        "model_type": model_type,
        "conditioning_kind": spec.conditioning_kind,
        "conditioning_id": spec.conditioning_id,
        "conditioning_token": spec.conditioning_token,
        "application_window_tokens": window,
        "hard_context": hard_context,
        "documented_context": spec.documented_context,
        "trust_remote_code": trust,
        "production_batch_size": _PRODUCTION_BATCH_SIZE,
        "trailing_eos_scored": False,
        "prefix_scored": False,
        "normalization": "none",
        "strict_loading_diagnostics": diagnostics,
        "experiment_admitted": False,
        "facts_source": (
            "read back from the loaded tokenizer, model config, and parameters; "
            "storage_torch_dtype is the checkpoint's declared dtype and is not "
            "the inference dtype"
        ),
        **_relative_position_facts(config, model_type=model_type),
    }
    return TextAAFitnessScorer(
        model=model,
        tokenizer=tokenizer,
        boundary=spec,
        application_window_tokens=window,
        facts=facts,
        torch_mod=torch,
    )
