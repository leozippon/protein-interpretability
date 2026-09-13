"""Native residue-state extractor for the in-memory ResidueBatch handoff.

This library tokenises already-supplied literal sequences with the shared native
helper, captures one explicit donor block, and returns a validated ResidueBatch.
It does not load, move, freeze, or authenticate models, caches, or scientific
assets. ResidueProvenance remains a caller assertion bound outside this module.
Cooperative cancellation runs before input work and between records; a native
forward is not interruptible here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import torch

from src.transfer.latent_bridge import (
    BLOCK_INDEX_SEMANTICS,
    DIRECTION_MARKER,
    capture_block_output,
    content_mask_from_ids,
    native_donor_token_ids,
    pad_rows,
)
from src.transfer.residue_prefix import (
    ACCESSION_MAX_CHARS,
    CAPTURE_DTYPES,
    MAX_TENSOR_BYTES_CEILING,
    ResidueBatch,
    ResidueProvenance,
    ResidueRecord,
    validate_residue_batch,
)

ID_BYTES = 8
MASK_BYTES = 1


def _require_int(value: object, *, name: str, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a non-boolean int")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _explicit_device(device: object) -> torch.device:
    if type(device) is str:
        return torch.device(device)
    if isinstance(device, torch.device):
        return device
    raise TypeError("device must be a str or torch.device")


def _planned_tensor_bytes(*, batch: int, time: int, hidden_width: int, capture_dtype: torch.dtype) -> int:
    hidden_position = hidden_width * {torch.float32: 4, torch.bfloat16: 2}[capture_dtype]
    return batch * time * (hidden_position + ID_BYTES + MASK_BYTES + MASK_BYTES)


def _sequence_sha256(sequence: str) -> str:
    return sha256(sequence.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ResidueSequence:
    """Literal accession and residue string. Not an EC PreparedRecord."""

    accession: str
    sequence: str

    def __post_init__(self) -> None:
        if type(self.accession) is not str or not self.accession or len(self.accession) > ACCESSION_MAX_CHARS:
            raise ValueError(
                f"accession must be a nonempty string of at most {ACCESSION_MAX_CHARS} characters"
            )
        if type(self.sequence) is not str or not self.sequence:
            raise ValueError("sequence must be a nonempty string")


def extract_residue_batch(
    donor: Any,
    sequences: Sequence[ResidueSequence],
    *,
    expected_records: Sequence[ResidueRecord],
    expected_provenance: ResidueProvenance,
    device: str | torch.device,
    max_residues: int,
    max_tokens: int,
    max_tensor_bytes: int,
    check: Callable[[], None],
) -> ResidueBatch:
    """Capture right-padded marker+residue states for one already-loaded donor."""

    if not callable(check):
        raise TypeError("check must be callable")
    check()
    residues = _require_int(max_residues, name="max_residues", minimum=1)
    tokens = _require_int(max_tokens, name="max_tokens", minimum=1)
    cap = _require_int(max_tensor_bytes, name="max_tensor_bytes", minimum=1)
    if cap > MAX_TENSOR_BYTES_CEILING:
        raise ValueError(
            f"max_tensor_bytes {cap} exceeds the {MAX_TENSOR_BYTES_CEILING} byte hard ceiling"
        )
    requested = _explicit_device(device)
    items = tuple(sequences)
    if not items:
        raise ValueError("extract_residue_batch requires a nonempty sequence batch")
    if any(type(item) is not ResidueSequence for item in items):
        raise TypeError("sequences must be ResidueSequence values")
    records = tuple(expected_records)
    if any(type(record) is not ResidueRecord for record in records):
        raise TypeError("expected_records must be ResidueRecord values")
    if type(expected_provenance) is not ResidueProvenance:
        raise TypeError("expected_provenance must be ResidueProvenance")
    if len(items) != len(records):
        raise ValueError("sequence count must match expected_records")
    if expected_provenance.block_index_semantics != BLOCK_INDEX_SEMANTICS:
        raise ValueError("block_index_semantics must be the existing BLOCK_INDEX_SEMANTICS text")
    if not hasattr(donor, "blocks") or not hasattr(donor, "model") or not hasattr(donor, "tokenizer"):
        raise TypeError("donor must expose model, tokenizer, and Arm.blocks(); do not index hidden_states")
    model = donor.model
    tokenizer = donor.tokenizer
    blocks = donor.blocks()
    width = _require_int(donor.d_model, name="donor.d_model", minimum=1)
    if width != expected_provenance.hidden_width:
        raise ValueError("donor hidden width does not match provenance.hidden_width")
    n_layer = getattr(donor, "n_layer", None)
    if n_layer is not None and _require_int(n_layer, name="donor.n_layer", minimum=1) != len(blocks):
        raise ValueError("donor.n_layer does not match donor.blocks()")
    if not 0 <= expected_provenance.block_index < len(blocks):
        raise ValueError(
            f"block_index {expected_provenance.block_index} is outside 0..{len(blocks) - 1}"
        )
    if any(module.training for module in model.modules()):
        raise ValueError("donor contains modules in train mode; refusing silent eval()")
    parameters = list(model.parameters())
    if not parameters:
        raise ValueError("donor has no parameters")
    if any(parameter.requires_grad for parameter in parameters):
        raise ValueError("donor parameters require grad; refusing silent freeze")
    if any(parameter.device != requested for parameter in parameters):
        raise ValueError("donor is not on the explicit device; refusing silent relocate")
    marker = tokenizer.convert_tokens_to_ids(DIRECTION_MARKER)
    if marker is None or marker == tokenizer.unk_token_id:
        raise ValueError("donor tokenizer has no direction-marker id")
    if int(marker) != expected_provenance.marker_id:
        raise ValueError("tokenizer marker id does not match provenance.marker_id")
    declared_pad = getattr(tokenizer, "pad_token_id", None)
    if declared_pad is not None and int(declared_pad) != expected_provenance.pad_id:
        raise ValueError("tokenizer pad_token_id does not match provenance.pad_id")
    capture_dtype = CAPTURE_DTYPES[expected_provenance.capture_dtype]
    check()
    token_rows: list[list[int]] = []
    for item, record in zip(items, records, strict=True):
        check()
        if item.accession != record.accession:
            raise ValueError("sequence accession does not match expected_records")
        if _sequence_sha256(item.sequence) != record.sequence_sha256:
            raise ValueError("sequence SHA256 does not match expected_records")
        if len(item.sequence) != record.residue_count:
            raise ValueError("residue count does not match expected_records")
        native_ids = native_donor_token_ids(
            tokenizer, item.sequence, max_tokens=tokens, max_residues=residues,
        )
        if tuple(native_ids) != record.native_token_ids:
            raise ValueError("native token ids do not match expected_records")
        if native_ids[0] != expected_provenance.marker_id:
            raise ValueError("first native token must be the provenance marker_id")
        if any(token == expected_provenance.marker_id or token == expected_provenance.pad_id for token in native_ids[1:]):
            raise ValueError("native residue token ids must not equal marker_id or pad_id")
        token_rows.append(native_ids)
    check()
    time = max(len(row) for row in token_rows)
    planned = _planned_tensor_bytes(
        batch=len(token_rows),
        time=time,
        hidden_width=width,
        capture_dtype=capture_dtype,
    )
    if planned > cap:
        raise ValueError(
            f"planned batch tensors occupy {planned} bytes, exceeding max_tensor_bytes={cap}; "
            "refusing to allocate or forward"
        )
    check()
    # Staging must not allocate on an ambient default (possibly another GPU).
    with torch.device("cpu"):
        padded_ids, padded_mask = pad_rows(token_rows, expected_provenance.pad_id)
    padded_ids = padded_ids.to(requested)
    padded_mask = padded_mask.to(device=requested, dtype=torch.bool)
    hidden = capture_block_output(
        donor,
        layer=expected_provenance.block_index,
        input_ids=padded_ids,
        attention_mask=padded_mask,
    ).detach()
    check()
    if hidden.device != requested:
        raise ValueError("captured hidden is not on the explicit device; refusing silent relocate")
    if hidden.dtype != capture_dtype:
        raise ValueError(
            f"hidden dtype {hidden.dtype} does not match declared capture dtype "
            f"{expected_provenance.capture_dtype}"
        )
    if hidden.shape != (len(token_rows), time, width):
        raise ValueError("captured hidden shape is not (batch, max(L+1), hidden_width)")
    attention = padded_mask.bool()
    content = content_mask_from_ids(
        padded_ids,
        padded_mask,
        marker_id=expected_provenance.marker_id,
        pad_id=expected_provenance.pad_id,
    )
    check()
    batch = ResidueBatch(
        hidden=hidden,
        input_ids=padded_ids,
        attention_mask=attention,
        content_mask=content,
        records=records,
        provenance=expected_provenance,
    )
    return validate_residue_batch(
        batch,
        expected_records=records,
        expected_provenance=expected_provenance,
        max_tensor_bytes=cap,
    )
