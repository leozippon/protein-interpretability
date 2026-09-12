"""In-memory residue-batch handoff and learned-query soft-prefix compressor.

This is an explicit CPU-side interface unit. It does not extract donor states,
persist a cache, parse QA tasks, or run recoverable training. Callers supply
already-captured residue tensors and bound provenance assertions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.transfer.latent_bridge import BLOCK_INDEX_SEMANTICS, mean_content

IN_MEMORY_SCHEMA = "residue_query_prefix_in_memory_v1"
ACCESSION_MAX_CHARS = 128
MAX_TENSOR_BYTES_CEILING = 1024 ** 3
HEX64_LEN = 64
COMMIT_HEX_LEN = 40
_HEX_RE = re.compile(r"[0-9a-f]+")
CAPTURE_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
PROVENANCE_NOTE = (
    "ResidueProvenance fields are bound caller assertions about this in-memory "
    "handoff. They do not qualify a file, checkpoint, tokenizer, or model load."
)
TOKEN_ID_NOTE = (
    "Native token IDs are checked for consistency with the supplied record "
    "metadata and provenance marker/pad IDs. This does not authenticate that "
    "the IDs were obtained from a qualified tokenizer; extractor qualification "
    "remains open."
)
LIMITATIONS = (
    "Not a residue cache, save/load API, or live-extraction path.",
    "Frozen dataclass fields do not make caller-owned tensors immutable or hash-sealed.",
    "max_tensor_bytes is a per-batch input bound over hidden+ids+masks, not VRAM/activation cost.",
    "No filesystem or host-resource checks in this library.",
    "Contextual mean and the query resampler can both carry donor-encoded order; "
    "the resampler adds no positions and does not locate residues on its own.",
    "Learned queries are global latent slots, not a natural-language question encoder.",
    PROVENANCE_NOTE,
    TOKEN_ID_NOTE,
)


def _require_int(value: object, *, name: str, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a non-boolean int")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _require_hex(value: object, *, name: str, length: int) -> str:
    if type(value) is not str or len(value) != length or _HEX_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be {length} lowercase hex characters")
    return value


def query_prefix_parameter_count(
    *,
    input_width: int,
    internal_width: int,
    output_width: int,
    n_queries: int,
) -> int:
    """Exact trainable parameter count for this QueryPrefix implementation.

    Counts FP32 adapter tensors only: input projection, learned queries,
    four attention linears, two LayerNorms, a residual FFN with inner width
    equal to ``internal_width``, and the receiver projection. Head count
    reshapes attention and does not add parameters.
    """

    d_in = _require_int(input_width, name="input_width", minimum=1)
    d = _require_int(internal_width, name="internal_width", minimum=1)
    d_out = _require_int(output_width, name="output_width", minimum=1)
    k = _require_int(n_queries, name="n_queries", minimum=1)
    return d * d_in + k * d + d_out * d + 6 * d * d + 11 * d + d_out


@dataclass(frozen=True)
class ResidueRecord:
    """Per-row native-geometry metadata. Synthetic callers must mark that fact."""

    accession: str
    sequence_sha256: str
    residue_count: int
    native_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.accession) is not str or not self.accession or len(self.accession) > ACCESSION_MAX_CHARS:
            raise ValueError(
                f"accession must be a nonempty string of at most {ACCESSION_MAX_CHARS} characters"
            )
        _require_hex(self.sequence_sha256, name="sequence_sha256", length=HEX64_LEN)
        length = _require_int(self.residue_count, name="residue_count", minimum=1)
        tokens = self.native_token_ids
        if type(tokens) is not tuple:
            tokens = tuple(tokens)
            object.__setattr__(self, "native_token_ids", tokens)
        if len(tokens) != length + 1:
            raise ValueError(
                "native_token_ids must be exactly one forward marker then one token per residue"
            )
        checked: list[int] = []
        for index, token in enumerate(tokens):
            checked.append(_require_int(token, name=f"native_token_ids[{index}]", minimum=0))
        if tuple(checked) != tokens:
            object.__setattr__(self, "native_token_ids", tuple(checked))


@dataclass(frozen=True)
class ResidueProvenance:
    """Shared caller assertions for one in-memory residue batch."""

    schema: str
    donor_inventory_sha256: str
    code_commit: str
    block_index: int
    block_index_semantics: str
    capture_dtype: str
    hidden_width: int
    marker_id: int
    pad_id: int

    def __post_init__(self) -> None:
        if self.schema != IN_MEMORY_SCHEMA:
            raise ValueError(f"schema must be {IN_MEMORY_SCHEMA!r}, not a cache or mean-pool schema")
        _require_hex(self.donor_inventory_sha256, name="donor_inventory_sha256", length=HEX64_LEN)
        _require_hex(self.code_commit, name="code_commit", length=COMMIT_HEX_LEN)
        _require_int(self.block_index, name="block_index", minimum=0)
        if self.block_index_semantics != BLOCK_INDEX_SEMANTICS:
            raise ValueError("block_index_semantics must be the existing BLOCK_INDEX_SEMANTICS text")
        if self.capture_dtype not in CAPTURE_DTYPES:
            raise ValueError("capture_dtype must be 'float32' or 'bfloat16'")
        _require_int(self.hidden_width, name="hidden_width", minimum=1)
        marker = _require_int(self.marker_id, name="marker_id", minimum=0)
        pad = _require_int(self.pad_id, name="pad_id", minimum=0)
        if marker == pad:
            raise ValueError("marker_id and pad_id must be distinct")


@dataclass(frozen=True, eq=False)
class ResidueBatch:
    """Caller-owned residue tensors plus immutable metadata.

    Freezing the dataclass does not freeze storage: hidden, input_ids, and
    masks remain mutable tensors and are not hash-sealed.
    """

    hidden: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    content_mask: torch.Tensor
    records: tuple[ResidueRecord, ...]
    provenance: ResidueProvenance

    def __post_init__(self) -> None:
        records = self.records if type(self.records) is tuple else tuple(self.records)
        object.__setattr__(self, "records", records)
        if any(type(record) is not ResidueRecord for record in records):
            raise TypeError("records must be ResidueRecord values")
        if type(self.provenance) is not ResidueProvenance:
            raise TypeError("provenance must be ResidueProvenance")


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _require_expected(
    batch: ResidueBatch,
    *,
    expected_records: Sequence[ResidueRecord],
    expected_provenance: ResidueProvenance,
) -> tuple[tuple[ResidueRecord, ...], ResidueProvenance]:
    if type(batch) is not ResidueBatch:
        raise TypeError("batch must be ResidueBatch")
    records = tuple(expected_records)
    if any(type(record) is not ResidueRecord for record in records):
        raise TypeError("expected_records must be ResidueRecord values")
    if records != batch.records:
        raise ValueError("expected_records must equal the batch records in order")
    if type(expected_provenance) is not ResidueProvenance:
        raise TypeError("expected_provenance must be ResidueProvenance")
    if expected_provenance != batch.provenance:
        raise ValueError("expected_provenance must equal the batch provenance")
    if batch.provenance.schema != IN_MEMORY_SCHEMA:
        raise ValueError("wrong in-memory schema")
    return records, batch.provenance


def validate_residue_batch(
    batch: ResidueBatch,
    *,
    expected_records: Sequence[ResidueRecord],
    expected_provenance: ResidueProvenance,
    max_tensor_bytes: int,
) -> ResidueBatch:
    """Validate caller tensors at a public mean/query boundary. Does not copy."""

    records, provenance = _require_expected(
        batch,
        expected_records=expected_records,
        expected_provenance=expected_provenance,
    )
    cap = _require_int(max_tensor_bytes, name="max_tensor_bytes", minimum=1)
    if cap > MAX_TENSOR_BYTES_CEILING:
        raise ValueError(
            f"max_tensor_bytes {cap} exceeds the {MAX_TENSOR_BYTES_CEILING} byte hard ceiling"
        )
    hidden = batch.hidden
    input_ids = batch.input_ids
    attention_mask = batch.attention_mask
    content_mask = batch.content_mask
    if hidden.ndim == 2:
        raise ValueError(
            "residue hidden must be (batch, time, width); 2-D mean features are not residue states"
        )
    if hidden.ndim != 3:
        raise ValueError("residue hidden must be (batch, time, width)")
    batch_size, time, width = (int(hidden.shape[0]), int(hidden.shape[1]), int(hidden.shape[2]))
    for name, value in (("batch", batch_size), ("time", time), ("hidden_width", width)):
        _require_int(value, name=name, minimum=1)
    if len(records) != batch_size:
        raise ValueError("record count must match the hidden batch dimension")
    if width != provenance.hidden_width:
        raise ValueError("hidden width does not match provenance.hidden_width")
    if input_ids.shape != (batch_size, time) or input_ids.dtype != torch.long:
        raise ValueError("input_ids must be long with shape (batch, time)")
    if attention_mask.shape != (batch_size, time) or attention_mask.dtype != torch.bool:
        raise ValueError("attention_mask must be bool with shape (batch, time)")
    if content_mask.shape != (batch_size, time) or content_mask.dtype != torch.bool:
        raise ValueError("content_mask must be bool with shape (batch, time)")
    device = hidden.device
    if input_ids.device != device or attention_mask.device != device or content_mask.device != device:
        raise ValueError("hidden, input_ids, and masks must share one device")
    declared = CAPTURE_DTYPES[provenance.capture_dtype]
    if hidden.dtype != declared:
        raise ValueError(
            f"hidden dtype {hidden.dtype} does not match declared capture dtype {provenance.capture_dtype}"
        )
    if hidden.requires_grad or hidden.grad_fn is not None:
        raise ValueError("donor hidden must be detached frozen features")
    used = (
        _tensor_bytes(hidden)
        + _tensor_bytes(input_ids)
        + _tensor_bytes(attention_mask)
        + _tensor_bytes(content_mask)
    )
    if used > cap:
        raise ValueError(
            f"batch tensors occupy {used} bytes, exceeding max_tensor_bytes={cap}; refusing to slice"
        )
    if not torch.isfinite(hidden).all():
        raise ValueError("hidden contains non-finite values")
    marker = provenance.marker_id
    pad = provenance.pad_id
    for row, record in enumerate(records):
        length = record.residue_count
        attended = length + 1
        if attended > time:
            raise ValueError("native geometry exceeds the time axis; refusing truncation")
        attn_row = attention_mask[row]
        content_row = content_mask[row]
        ids_row = input_ids[row]
        if not bool(attn_row[:attended].all()) or bool(attn_row[attended:].any()):
            raise ValueError("attention must cover exactly the right-padded marker+residue prefix")
        expected_content = torch.zeros(time, dtype=torch.bool, device=device)
        expected_content[1:attended] = True
        if not torch.equal(content_row, expected_content):
            raise ValueError("content mask must be true only on residue positions 1..L")
        native = tuple(cast(int, ids_row[position].item()) for position in range(attended))
        if native != record.native_token_ids:
            raise ValueError("input_ids do not match supplied native_token_ids")
        if native[0] != marker:
            raise ValueError("first native token must be the provenance marker_id")
        if any(token == marker or token == pad for token in native[1:]):
            raise ValueError("native residue token ids must not equal marker_id or pad_id")
        if attended < time:
            pad_ids = ids_row[attended:]
            if bool((pad_ids != pad).any()):
                raise ValueError("right-pad positions must use pad_id with both masks false")
        if not bool(content_row.any()):
            raise ValueError("a record has no content positions")
    return batch


def mean_from_residues(
    batch: ResidueBatch,
    *,
    expected_records: Sequence[ResidueRecord],
    expected_provenance: ResidueProvenance,
    max_tensor_bytes: int,
) -> torch.Tensor:
    """Masked mean over residue positions. Pooling is FP32, matching old extraction."""

    validated = validate_residue_batch(
        batch,
        expected_records=expected_records,
        expected_provenance=expected_provenance,
        max_tensor_bytes=max_tensor_bytes,
    )
    return mean_content(validated.hidden.float(), validated.content_mask)


class QueryPrefix(nn.Module):
    """Single-block learned-query cross-attention resampler.

    Learned queries are global latent slots. Marker and pad positions are
    masked out of attention. Captured bfloat16 states are promoted to FP32
    before the adapter; adapter weights stay FP32. No dropout, LoRA, or
    question encoder. Train and eval paths are identical.
    """

    def __init__(
        self,
        *,
        input_width: int,
        internal_width: int,
        output_width: int,
        n_queries: int,
        n_heads: int,
        max_trainable_parameters: int,
    ) -> None:
        d_in = _require_int(input_width, name="input_width", minimum=1)
        d = _require_int(internal_width, name="internal_width", minimum=1)
        d_out = _require_int(output_width, name="output_width", minimum=1)
        k = _require_int(n_queries, name="n_queries", minimum=1)
        heads = _require_int(n_heads, name="n_heads", minimum=1)
        budget = _require_int(max_trainable_parameters, name="max_trainable_parameters", minimum=1)
        if d % heads != 0:
            raise ValueError("internal_width must be divisible by n_heads")
        predicted = query_prefix_parameter_count(
            input_width=d_in,
            internal_width=d,
            output_width=d_out,
            n_queries=k,
        )
        if predicted > budget:
            raise ValueError(
                f"QueryPrefix needs {predicted} parameters, exceeding max_trainable_parameters={budget}"
            )
        nn.Module.__init__(self)  # pyright: ignore[reportUnknownMemberType]
        self.input_width = d_in
        self.internal_width = d
        self.output_width = d_out
        self.n_queries = k
        self.n_heads = heads
        self.head_dim = d // heads
        self.predicted_parameters = predicted
        self.input_proj = nn.Linear(d_in, d)
        self.query_tokens = nn.Parameter(torch.empty(k, d))
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.attn_out = nn.Linear(d, d)
        self.attn_norm = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, d)
        self.ff2 = nn.Linear(d, d)
        self.ff_norm = nn.LayerNorm(d)
        self.output_proj = nn.Linear(d, d_out)
        for parameter in self.parameters():
            if parameter.dtype != torch.float32:
                raise ValueError("QueryPrefix parameters must be FP32")
            parameter.requires_grad_(True)
        actual = self.trainable_parameter_count()
        if actual != predicted:
            raise RuntimeError(f"parameter count {actual} != formula {predicted}")

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def train(self, mode: bool = True) -> QueryPrefix:
        super().train(mode)
        return self

    def _cross_attend(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        content_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, n_queries, width = queries.shape
        heads = self.n_heads
        head_dim = self.head_dim
        query = self.q_proj(queries).view(batch, n_queries, heads, head_dim).transpose(1, 2)
        key = self.k_proj(keys).view(batch, keys.shape[1], heads, head_dim).transpose(1, 2)
        value = self.v_proj(values).view(batch, values.shape[1], heads, head_dim).transpose(1, 2)
        scale = head_dim ** -0.5
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        scores = scores.masked_fill(~content_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        if not torch.isfinite(weights).all():
            raise ValueError("query attention produced non-finite weights")
        mixed = torch.matmul(weights, value).transpose(1, 2).contiguous().view(batch, n_queries, width)
        return self.attn_out(mixed)

    def forward(
        self,
        batch: ResidueBatch,
        *,
        expected_records: Sequence[ResidueRecord],
        expected_provenance: ResidueProvenance,
        max_tensor_bytes: int,
    ) -> torch.Tensor:
        validated = validate_residue_batch(
            batch,
            expected_records=expected_records,
            expected_provenance=expected_provenance,
            max_tensor_bytes=max_tensor_bytes,
        )
        hidden = validated.hidden
        if hidden.shape[-1] != self.input_width:
            raise ValueError(
                f"hidden width {int(hidden.shape[-1])} does not match QueryPrefix input_width {self.input_width}"
            )
        try:
            parameter = next(self.parameters())
        except StopIteration as exc:
            raise RuntimeError("QueryPrefix has no parameters") from exc
        if hidden.device != parameter.device:
            raise ValueError("batch device does not match QueryPrefix parameters; refusing silent relocate")
        if parameter.dtype != torch.float32:
            raise ValueError("QueryPrefix parameters must be FP32")
        hidden_fp32 = hidden.float()
        keys = self.input_proj(hidden_fp32)
        queries = self.query_tokens.unsqueeze(0).expand(hidden_fp32.shape[0], -1, -1)
        attended = self._cross_attend(queries, keys, keys, validated.content_mask)
        hidden_states = self.attn_norm(queries + attended)
        feedforward = self.ff2(F.gelu(self.ff1(hidden_states)))
        hidden_states = self.ff_norm(hidden_states + feedforward)
        tokens = self.output_proj(hidden_states)
        if tokens.dtype != torch.float32 or tokens.shape != (
            hidden_fp32.shape[0],
            self.n_queries,
            self.output_width,
        ):
            raise ValueError("query prefix must return FP32 (batch, n_queries, output_width)")
        if not torch.isfinite(tokens).all():
            raise ValueError("query prefix produced non-finite tokens")
        return tokens
