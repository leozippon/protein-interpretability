"""Transplanting declared parameter groups of one checkpoint into another.

What this is for. The ProLLaMA lineage carries the panel's cleanest matched
capability difference: on the 211-assay native support the Llama-2-7B parent's
native likelihood ranks mutations at -0.00740 [-0.02713, +0.01226] within-assay
Spearman and ProLLaMA Stage 1's at +0.14604 [+0.11814, +0.17435], with the
architecture, the tokenizer, the rendering and the scale held fixed and only the
continued-pretraining step between them. Copying part of Stage 1's parameters
into the parent and re-measuring that same quantity asks which part of the
checkpoint difference is enough, in this context, to carry the gain.

What a transplant is and is not. A group whose transplant moves the endpoint
shows that those parameter values are *sufficient in this context*: sufficient
against this parent, on this support, under this scoring. It does not show they
are necessary, does not name a circuit, and does not say what the computation
is. Necessity is a separate measurement -- transplanting every group except one
-- and this module supports it through :func:`resolve_groups`' ``complement:``
spelling so that both claims are made with the same machinery and neither is
read off the other.

The partition. Every tensor in the checkpoint belongs to exactly one declared
group, and the groups are checked to cover the checkpoint rather than assumed
to. That is what makes the two endpoints exact rather than approximate: the
empty selection leaves the destination checkpoint untouched, and the full
selection overwrites every tensor, so the floor is the parent bit-for-bit and
the ceiling is the source bit-for-bit. A partition that quietly missed a tensor
would put an unmeasured mixture at both ends and every recovered fraction
between them would be read against the wrong scale.

Digests. Provenance here is tensor-by-tensor because a localisation claim is
only as good as the record of what was moved. :func:`fp32_digest` is the single
definition both the census and the post-transplant verification use, so a
destination tensor's digest is comparable with its source's by construction
rather than by two implementations agreeing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
from typing import Any

#: Depth blocks the transformer body is cut into. Quarters, because the screen
#: has to cross depth with component type inside a bounded arm budget and eight
#: crossed cells is what that budget buys; the number is declared here so the
#: block boundaries are derived from the layer count rather than written down.
DEPTH_BLOCKS = 4

#: Groups that are not per-layer. ``norm`` collects every RMSNorm weight, the
#: final one included, and ``rope`` the stored rotary frequency buffers: both
#: are declared groups rather than an unclassified remainder, so that a
#: checkpoint pair differing there is a finding instead of a silent omission.
FLAT_GROUPS = ("embedding", "head", "norm", "rope")

#: Groups a checkpoint may legitimately not store. The per-layer rotary
#: frequency buffer is serialized by the transformers versions these two
#: checkpoints were released under and is absent from ones saved by later
#: versions, so its absence is a serialization fact rather than a partition
#: defect. Every other declared group must be present, because an empty one
#: would mean the depth partition had been computed against the wrong layer
#: count and the selections would silently cover less than they name.
OPTIONAL_GROUPS = ("rope",)

_LAYER = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_ATTENTION = frozenset(f"self_attn.{k}_proj.weight" for k in ("q", "k", "v", "o"))
_MLP = frozenset(f"mlp.{k}_proj.weight" for k in ("gate", "up", "down"))
_NORM = frozenset(("input_layernorm.weight", "post_attention_layernorm.weight"))
_ROPE = "self_attn.rotary_emb.inv_freq"

#: Chunk size for digesting and comparing tensors, in elements. Bounds the peak
#: working set of a census over two 6.7 G-parameter checkpoints.
CHUNK_ELEMENTS = 1 << 20


def depth_block(layer: int, n_layers: int) -> int:
    """Which declared depth block a zero-based layer index falls in."""

    if n_layers < DEPTH_BLOCKS or not 0 <= layer < n_layers:
        raise ValueError(f"layer {layer} outside a {n_layers}-layer body")
    return min(layer * DEPTH_BLOCKS // n_layers, DEPTH_BLOCKS - 1)


def group_names() -> tuple[str, ...]:
    """Every declared group, in a fixed order."""

    depth = tuple(
        f"{kind}_q{index + 1}" for kind in ("attention", "mlp") for index in range(DEPTH_BLOCKS)
    )
    return FLAT_GROUPS + depth


def group_of(name: str, n_layers: int) -> str:
    """The one group a checkpoint tensor name belongs to, or a refusal.

    Refusing an unrecognised name is the load-bearing behaviour: a tensor this
    function cannot place would otherwise be left out of every selection, and
    the full transplant would stop being the source checkpoint.
    """

    if name == "model.embed_tokens.weight":
        return "embedding"
    if name == "lm_head.weight":
        return "head"
    if name == "model.norm.weight":
        return "norm"
    match = _LAYER.fullmatch(name)
    if match is None:
        raise ValueError(f"unclassified checkpoint tensor: {name}")
    layer, tail = int(match.group(1)), match.group(2)
    quarter = depth_block(layer, n_layers) + 1
    if tail in _NORM:
        return "norm"
    if tail == _ROPE:
        return "rope"
    if tail in _ATTENTION:
        return f"attention_q{quarter}"
    if tail in _MLP:
        return f"mlp_q{quarter}"
    raise ValueError(f"unclassified checkpoint tensor: {name}")


def partition(names: Iterable[str], n_layers: int) -> dict[str, list[str]]:
    """Group every name, and require the declared groups to be exactly covered."""

    result: dict[str, list[str]] = {group: [] for group in group_names()}
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ValueError(f"duplicate checkpoint tensor: {name}")
        seen.add(name)
        result[group_of(name, n_layers)].append(name)
    empty = [group for group, members in result.items()
             if not members and group not in OPTIONAL_GROUPS]
    if empty:
        raise ValueError(f"declared groups absent from the checkpoint: {empty}")
    return {group: sorted(members) for group, members in result.items()}


def resolve_groups(spec: str, *, allowed: Sequence[str] | None = None) -> tuple[str, ...]:
    """Parse a selection: ``none``, ``all``, a group list, or ``complement:<list>``.

    The complement spelling exists so that a necessity arm and a sufficiency arm
    are the same code path with the same partition. A complement computed some
    other way would be a second definition of what "everything else" means.
    """

    universe = tuple(allowed) if allowed is not None else group_names()
    unknown = [name for name in universe if name not in group_names()]
    if unknown:
        raise ValueError(f"undeclared groups: {unknown}")
    text = spec.strip()
    complement = text.startswith("complement:")
    if complement:
        text = text[len("complement:"):].strip()
        if text in ("", "none", "all"):
            raise ValueError("complement requires an explicit non-empty group list")
    if text == "none":
        selected: tuple[str, ...] = ()
    elif text == "all":
        selected = universe
    else:
        parts = [piece.strip() for piece in text.split(",") if piece.strip()]
        if not parts:
            raise ValueError("empty group selection; use 'none'")
        if len(set(parts)) != len(parts):
            raise ValueError("repeated group in selection")
        missing = [piece for piece in parts if piece not in universe]
        if missing:
            raise ValueError(f"undeclared groups: {missing}")
        selected = tuple(piece for piece in universe if piece in set(parts))
    if complement:
        selected = tuple(name for name in universe if name not in set(selected))
    return selected


def fp32_digest(tensor, *, chunk_elements: int = CHUNK_ELEMENTS) -> str:
    """SHA-256 over the tensor's FP32 bytes, chunked, in flat element order."""

    import torch

    if chunk_elements < 1:
        raise ValueError("chunk size must be positive")
    if not tensor.is_floating_point():
        raise ValueError("expected a floating-point model tensor")
    flat = tensor.detach().reshape(-1)
    digest = hashlib.sha256()
    for start in range(0, flat.numel(), chunk_elements):
        chunk = flat[start:start + chunk_elements].contiguous().to("cpu", torch.float32)
        if not torch.isfinite(chunk).all().item():
            raise ValueError("nonfinite FP32 tensor value")
        digest.update(chunk.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _weight_map(checkpoint: Path) -> dict[str, str]:
    index = checkpoint / "model.safetensors.index.json"
    return json.loads(index.read_text())["weight_map"]


def _n_layers(checkpoint: Path) -> int:
    return int(json.loads((checkpoint / "config.json").read_text())["num_hidden_layers"])


#: Semantic configuration fields, and the keys each architecture family declares
#: them under, in resolution order.
#:
#: Written because comparing raw key names is a blind spot rather than a
#: comparison. This function compared ``hidden_act`` and nothing else, which
#: Llama declares and OPT does not: an OPT pair differing at ``gelu`` against
#: ``relu`` read as *matching*, because ``config.get("hidden_act")`` returned
#: ``None`` on both sides. A census whose purpose is to refuse incompatible pairs
#: must compare the property in force, not the spelling one family happens to
#: use, or it passes the next pair for the same reason.
#:
#: Aliases are ordered, and the first key a config declares wins. ``enable_bias``
#: appears under both bias fields because OPT declares one flag where Llama
#: declares two; a family that separates them is compared field by field and one
#: that fuses them resolves both fields to the same flag.
CONFIG_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "model_type": ("model_type",),
    "hidden_width": ("hidden_size", "n_embd", "d_model"),
    "mlp_width": ("intermediate_size", "ffn_dim", "n_inner", "d_ff"),
    "depth": ("num_hidden_layers", "n_layer", "num_layers"),
    "attention_heads": ("num_attention_heads", "n_head", "num_heads"),
    "key_value_heads": ("num_key_value_heads",),
    "vocabulary": ("vocab_size",),
    "activation": ("hidden_act", "activation_function", "activation"),
    "norm_epsilon": ("rms_norm_eps", "layer_norm_epsilon", "layer_norm_eps"),
    "rope_theta": ("rope_theta",),
    "rope_scaling": ("rope_scaling",),
    "position_budget": ("max_position_embeddings", "n_positions"),
    "attention_bias": ("attention_bias", "enable_bias"),
    "mlp_bias": ("mlp_bias", "enable_bias"),
    "tied_embeddings": ("tie_word_embeddings",),
}

#: Fields a config may legitimately omit, with the value that omission means.
#: ``tie_word_embeddings`` defaults to ``True`` in the transformers base
#: configuration, so an absent field means *tied*, and the old truthiness test on
#: a missing key admitted a tied pair as untied. Defaulting to the refusing
#: direction is what makes the tie check a check.
CONFIG_FIELD_DEFAULTS: dict[str, Any] = {"tied_embeddings": True}


def resolve_config_field(config: Mapping[str, Any], field: str) -> dict[str, Any]:
    """The value of one semantic field, whichever key declares it.

    Returns the key that supplied it and whether it was declared, defaulted or
    absent, so a comparison can say *why* two configs agree.
    """

    if field not in CONFIG_FIELD_ALIASES:
        raise ValueError(f"undeclared configuration field {field!r}")
    for key in CONFIG_FIELD_ALIASES[field]:
        if key in config:
            return {"field": field, "key": key, "value": config[key], "source": "declared"}
    if field in CONFIG_FIELD_DEFAULTS:
        return {"field": field, "key": None, "value": CONFIG_FIELD_DEFAULTS[field],
                "source": "family default"}
    return {"field": field, "key": None, "value": None, "source": "absent"}


def compare_configs(configs: Sequence[Mapping[str, Any]],
                    fields: Sequence[str] = tuple(CONFIG_FIELD_ALIASES)) -> dict[str, Any]:
    """Field-by-field semantic comparison of two configurations.

    A field absent from both is not a mismatch -- ``key_value_heads`` is absent
    for every non-Llama family -- but it is reported in ``absent_on_both``
    instead of vanishing, because a field neither config declares is a property
    the comparison did not check and that has to be visible rather than silent.
    """

    if len(configs) != 2:
        raise ValueError("a configuration comparison takes exactly two configs")
    resolved = {field: [resolve_config_field(config, field) for config in configs]
                for field in fields}
    mismatched = sorted(field for field, pair in resolved.items()
                        if pair[0]["value"] != pair[1]["value"])
    absent = sorted(field for field, pair in resolved.items()
                    if all(entry["source"] == "absent" for entry in pair))
    return {
        "fields": {field: {"values": [entry["value"] for entry in pair],
                           "keys": [entry["key"] for entry in pair],
                           "sources": [entry["source"] for entry in pair]}
                   for field, pair in resolved.items()},
        "matching": sorted(field for field in resolved if field not in mismatched),
        "mismatched": mismatched,
        "absent_on_both": absent,
    }


def architecture_record(destination: Path, source: Path) -> dict[str, Any]:
    """Refuse a pair that cannot be transplanted, and record why it can be.

    Shapes alone are not enough: two checkpoints that disagree on the rendering
    their tokenizer produces, or that tie input and output weights, would make
    the endpoints of a transplant incomparable even with every tensor the same
    size. The comparison is over the semantic fields of
    :data:`CONFIG_FIELD_ALIASES` rather than over one family's key names, for the
    reason that table records.
    """

    configs = [json.loads((root / "config.json").read_text()) for root in (destination, source)]
    comparison = compare_configs(configs)
    if comparison["mismatched"]:
        raise ValueError("architecture mismatch: " + json.dumps(
            {field: comparison["fields"][field] for field in comparison["mismatched"]},
            sort_keys=True))
    if resolve_config_field(configs[0], "tied_embeddings")["value"]:
        raise ValueError("embedding/head transplant is undefined for tied weights")
    tokenizers = [hashlib.sha256((root / "tokenizer.model").read_bytes()).hexdigest()
                  for root in (destination, source)]
    if tokenizers[0] != tokenizers[1]:
        raise ValueError("tokenizer mismatch")
    maps = [_weight_map(root) for root in (destination, source)]
    if set(maps[0]) != set(maps[1]):
        raise ValueError("checkpoint tensor keys differ")
    return {
        "architecture_equal": True,
        "untied_embeddings": True,
        "tokenizer_model_sha256": tokenizers[0],
        "config_sha256": [hashlib.sha256((root / "config.json").read_bytes()).hexdigest()
                          for root in (destination, source)],
        "n_tensors": len(maps[0]),
        "num_hidden_layers": int(resolve_config_field(configs[0], "depth")["value"]),
        "configuration": comparison,
    }


def census(destination: Path, source: Path, *, chunk_elements: int = CHUNK_ELEMENTS,
           progress=None) -> dict[str, Any]:
    """Exact tensor-by-tensor comparison of the two checkpoints, by group.

    Both FP32 digests are recorded because FP32 is the precision the transplant
    and the scoring run in, and native byte equality is recorded separately so
    that an equality created by widening is never mistaken for one present in
    the stored checkpoints.
    """

    import torch
    from safetensors import safe_open

    interface = architecture_record(destination, source)
    n_layers = interface["num_hidden_layers"]
    roots = [destination, source]
    maps = [_weight_map(root) for root in roots]
    names = sorted(maps[0])
    members = partition(names, n_layers)
    rows: list[dict[str, Any]] = []
    for name in names:
        tensors = []
        for root, mapping in zip(roots, maps):
            with safe_open(root / mapping[name], framework="pt", device="cpu") as handle:
                tensors.append(handle.get_tensor(name))
        left, right = tensors
        if left.shape != right.shape:
            raise ValueError(f"{name}: shape mismatch")
        if not (left.is_floating_point() and right.is_floating_point()):
            raise ValueError(f"{name}: expected a floating-point model tensor")
        native = [hashlib.sha256(), hashlib.sha256()]
        differing = 0
        flat = [tensor.reshape(-1) for tensor in tensors]
        for start in range(0, left.numel(), chunk_elements):
            chunks = [tensor[start:start + chunk_elements].contiguous() for tensor in flat]
            widened = [chunk.to(torch.float32) for chunk in chunks]
            if not all(torch.isfinite(chunk).all().item() for chunk in widened):
                raise ValueError(f"{name}: nonfinite FP32 checkpoint value")
            differing += int(torch.count_nonzero(widened[0] != widened[1]).item())
            for index in range(2):
                native[index].update(chunks[index].view(torch.uint8).numpy().tobytes())
        dtypes = [str(tensor.dtype) for tensor in tensors]
        rows.append({
            "name": name,
            "group": group_of(name, n_layers),
            "shape": list(left.shape),
            "n_elements": int(left.numel()),
            "dtypes": dtypes,
            "native_sha256": [handle.hexdigest() for handle in native],
            "fp32_sha256": [fp32_digest(tensor, chunk_elements=chunk_elements) for tensor in tensors],
            "differing_elements_fp32": differing,
        })
        rows[-1]["native_bytes_equal"] = (
            dtypes[0] == dtypes[1] and rows[-1]["native_sha256"][0] == rows[-1]["native_sha256"][1]
        )
        rows[-1]["fp32_bytes_equal"] = rows[-1]["fp32_sha256"][0] == rows[-1]["fp32_sha256"][1]
        if (rows[-1]["differing_elements_fp32"] == 0) != rows[-1]["fp32_bytes_equal"]:
            raise ValueError(f"{name}: element comparison and FP32 digest disagree")
        if progress is not None:
            progress(rows[-1])
        del tensors, left, right, flat
    by_name = {row["name"]: row for row in rows}
    groups = {}
    for group, group_names_ in members.items():
        selected = [by_name[name] for name in group_names_]
        groups[group] = {
            "n_tensors": len(selected),
            "n_elements": sum(row["n_elements"] for row in selected),
            "changed_tensors_fp32": sum(1 for row in selected if row["differing_elements_fp32"]),
            "differing_elements_fp32": sum(row["differing_elements_fp32"] for row in selected),
            "fp32_bytes_equal": all(row["fp32_bytes_equal"] for row in selected),
            "native_bytes_equal": all(row["native_bytes_equal"] for row in selected),
            "tensor_names": group_names_,
        }
    return {
        "status": "complete",
        "schema_version": "d2_transplant_census_v1",
        "interface": interface,
        "comparison": (
            "exact elementwise equality after FP32 conversion, the precision the "
            "transplant and the scoring both run in; native byte equality recorded "
            "separately"
        ),
        "limitation": (
            "A census of stored values. Differing tensors do not establish which "
            "values carry a behaviour, and identical tensors are a no-op transplant "
            "rather than evidence of insensitivity."
        ),
        "sources": [
            {
                "checkpoint": str(root),
                "files": {
                    name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                    for name in ["model.safetensors.index.json", "config.json", "tokenizer.model",
                                 *sorted(set(mapping.values()))]
                },
            }
            for root, mapping in zip(roots, maps)
        ],
        "group_order": list(group_names()),
        "checkpoint_order": ["destination", "source"],
        "groups": groups,
        "tensors": rows,
    }


def census_expectations(census_record: dict[str, Any], selected: Sequence[str]) -> dict[str, str]:
    """Which FP32 digest each tensor must carry once ``selected`` is transplanted."""

    chosen = set(selected)
    undeclared = chosen - set(group_names())
    if undeclared:
        raise ValueError(f"undeclared groups: {sorted(undeclared)}")
    return {
        row["name"]: row["fp32_sha256"][1 if row["group"] in chosen else 0]
        for row in census_record["tensors"]
    }


def transplant(model, source: Path, census_record: dict[str, Any], selected: Sequence[str],
               *, chunk_elements: int = CHUNK_ELEMENTS) -> dict[str, Any]:
    """Copy the selected groups' source tensors into a loaded destination model.

    Every checkpoint tensor is then digested off the live model and compared
    with the digest the census says it must carry. That check is what turns
    "the code selected these names" into "these values are in the model": the
    empty selection has to come back as the destination checkpoint and the full
    selection as the source checkpoint, tensor for tensor, or the run refuses.
    """

    import torch
    from safetensors import safe_open

    chosen = set(selected)
    undeclared = chosen - set(group_names())
    if undeclared:
        raise ValueError(f"undeclared groups: {sorted(undeclared)}")
    mapping = _weight_map(source)
    live = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    rows = census_record["tensors"]
    copied, absent = [], []
    with torch.no_grad():
        for row in rows:
            name = row["name"]
            destination = live.get(name)
            if destination is None:
                absent.append(row)
                continue
            if row["group"] not in chosen:
                continue
            with safe_open(source / mapping[name], framework="pt", device="cpu") as handle:
                value = handle.get_tensor(name)
            if tuple(value.shape) != tuple(destination.shape):
                raise ValueError(f"{name}: shape mismatch against the loaded model")
            if destination.dtype != torch.float32:
                raise ValueError(f"{name}: destination is {destination.dtype}, not float32")
            destination.copy_(value.to(torch.float32))
            copied.append(row)
    unmoved = [row["name"] for row in absent if not row["fp32_bytes_equal"]]
    if unmoved:
        raise ValueError(
            f"checkpoint tensors differ but have no destination in the loaded model: {unmoved}"
        )
    expected = census_expectations(census_record, sorted(chosen))
    observed, mismatched = {}, []
    for row in rows:
        destination = live.get(row["name"])
        if destination is None:
            continue
        observed[row["name"]] = fp32_digest(destination, chunk_elements=chunk_elements)
        if observed[row["name"]] != expected[row["name"]]:
            mismatched.append(row["name"])
    if mismatched:
        raise ValueError(f"transplanted state does not match the census: {mismatched}")
    return {
        "selected_groups": sorted(chosen),
        "n_transplanted_tensors": len(copied),
        "n_transplanted_elements": sum(row["n_elements"] for row in copied),
        "n_transplanted_differing_elements": sum(row["differing_elements_fp32"] for row in copied),
        "transplanted_tensors": sorted(row["name"] for row in copied),
        "absent_destinations": sorted(row["name"] for row in absent),
        "absent_destinations_all_identical": True,
        "verified_tensor_fp32_sha256": observed,
        "verification": (
            "every live checkpoint tensor digested off the loaded model and matched "
            "against the census digest of the checkpoint its group selects"
        ),
    }


def select_screen_panel(assay_rows: Sequence[dict[str, Any]], *, families: int,
                        seed: int) -> dict[str, Any]:
    """A label-blind family subsample for the screen, one assay per family.

    Only assay and family identifiers reach the choice. The measured effects are
    never read, so the subsample cannot be selected towards an outcome, and the
    rule is replayed from the eligible support by every consumer rather than
    trusted from the artefact it is recorded in.
    """

    import numpy as np

    rows = {row["assay"]: row for row in assay_rows}
    if len(rows) != len(assay_rows):
        raise ValueError("duplicate eligible assay")
    if families < 2:
        raise ValueError("a family subsample needs at least two families")
    by_family: dict[Any, list[str]] = {}
    for assay in sorted(rows):
        by_family.setdefault(rows[assay]["cluster"], []).append(assay)
    ordered = sorted(by_family, key=str)
    if len(ordered) < families:
        raise ValueError(f"{families} families requested from {len(ordered)} eligible")
    rng = np.random.default_rng(seed)
    chosen = sorted([ordered[int(i)] for i in rng.permutation(len(ordered))[:families]], key=str)
    assays = sorted(by_family[family][int(rng.integers(len(by_family[family])))]
                    for family in chosen)
    return {
        "schema_version": "d2_screen_panel_v1",
        "seed": int(seed),
        "n_families": int(families),
        "selection": (
            "sort eligible assay IDs; group by family; sort families by str; seeded "
            "NumPy permutation takes the requested count; seeded integer takes one "
            "assay from each selected family's sorted list; sort the selected assay IDs"
        ),
        "uses_outcomes": False,
        "eligible_assay_ids": sorted(rows),
        "eligible_family_ids": ordered,
        "selected_family_ids": chosen,
        "selected_assay_ids": assays,
    }


def panel_digest(panel: dict[str, Any]) -> str:
    """One digest over what a panel selects, independent of how it is stored."""

    return hashlib.sha256(json.dumps(
        [panel["n_families"], panel["seed"], sorted(panel["selected_assay_ids"]),
         sorted(map(str, panel["selected_family_ids"]))],
        separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def support_digest(assay_ids: Iterable[str]) -> str:
    """One digest over an assay support, in the spelling the readout study used."""

    return hashlib.sha256(json.dumps(sorted(assay_ids), separators=(",", ":")).encode()).hexdigest()
