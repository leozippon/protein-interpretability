"""Caller-owned-stream loader for persisted metadata-census v1 seed artifacts.

Fixed v1 schema, relation policy, field names and digest construction follow the
archived producer ``logs/bridge_complementarity_20260909/pika_metadata_census_v2/census.py``
(``metadata_census_projection_v1``, ``metadata_census_component_v1``,
``metadata_census_summary_v1``). This module does not import that producer, compute
a new graph, global df, W, U, shingle intersections or roles, authenticate original
source assets, reread scientific paths, or select a cohort.

Caller owns the three distinct binary streams; this library neither opens nor
closes them. Filesystem, symlink and stable-stat checks, the 8 GiB MemAvailable /
5 GiB disk gate, address-space limits and execution deadlines are operator
responsibilities and are not claimed here.

``check`` is cooperative cancellation and is invoked before the first input access,
around bounded decode/hash work, at each record and member, and in validation
loops. Builtin JSON decoding and canonical digest encoding/sorting are not
immediately interruptible.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, fields
from hashlib import sha256
from typing import BinaryIO, Literal, Protocol, cast
import json
import math
import re

RELATION_POLICY = "accession_exactsequence_rawpfam_pikauniref_only_metadata_census_v1"
PROJECTION_SCHEMA = "metadata_census_projection_v1"
COMPONENT_SCHEMA = "metadata_census_component_v1"
SUMMARY_SCHEMA = "metadata_census_summary_v1"
SOURCE_NAMES = (
    "old_jsonl", "old_pfam_tsv", "pika_sequences", "pika_uniref_split",
    "pika_evogroup_split", "preview_selection", "current_xml",
)
LIMITATION_NAMES = (
    "near_duplicate_closure_complete", "homology_isolation_established",
    "eligibility_assigned", "roles_reserved",
)
LIBRARY_FLAGS = ("source_assets_verified", "fixed_universe_cardinalities_verified")
FLAG_NAMES = (
    "old_exposure", "preview_exposure", "either_test",
    "unresolved_current_xml", "unresolved_current_xml_pfam",
)
ACCESSION = re.compile(
    r"(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})\Z"
)
SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")
MAX_NODES = 300_000
MAX_DECLARED_MEMBERSHIPS = 250_000_000
MAX_SEQUENCE_CHARACTERS = 262_148
MAX_JSONL_RECORD_BYTES = 8 * 1024 * 1024
MAX_SUMMARY_BYTES = 1024 * 1024
MAX_COMBINED_INPUT_BYTES = 1024 ** 3
Pool = Literal["old", "pika"]
Key = tuple[Pool, str]
Partition = tuple[tuple[Key, ...], ...]
LIMITATION_STATEMENTS = (
    "no new near-duplicate closure, homology isolation, eligibility or roles",
    "original source assets remain unauthenticated",
    "real 277167-node cardinality is not qualified",
    "declared per-record shingle counts are not recomputed",
    "unconsumed projection metadata is not revalidated",
    "caller stream ownership, filesystem identity, resource gates, "
    "address-space limits and deadlines are not verified by this library",
)


class SeedLoaderError(ValueError):
    """Stable failure code without scientific payload or paths."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def require(condition: bool, code: str) -> None:
    if not condition:
        raise SeedLoaderError(code)


def _int_in_range(value: object, code: str, *, minimum: int, maximum: int) -> int:
    require(type(value) is int, code)
    number = cast(int, value)
    require(minimum <= number <= maximum, code)
    return number


@dataclass(frozen=True)
class ArtifactIdentity:
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        require(type(self.bytes) is int and self.bytes > 0, "invalid_expected_identity")
        require(
            type(self.sha256) is str and SHA256_HEX.fullmatch(self.sha256) is not None,
            "invalid_expected_identity",
        )


@dataclass(frozen=True)
class SeedInputIdentities:
    projection: ArtifactIdentity
    components: ArtifactIdentity
    summary: ArtifactIdentity

    def __post_init__(self) -> None:
        require(type(self.projection) is ArtifactIdentity, "invalid_expected_identity")
        require(type(self.components) is ArtifactIdentity, "invalid_expected_identity")
        require(type(self.summary) is ArtifactIdentity, "invalid_expected_identity")


@dataclass(frozen=True)
class LoaderLimits:
    nodes: int = MAX_NODES
    declared_memberships: int = MAX_DECLARED_MEMBERSHIPS
    sequence_characters: int = MAX_SEQUENCE_CHARACTERS
    projection_record_bytes: int = MAX_JSONL_RECORD_BYTES
    component_record_bytes: int = MAX_JSONL_RECORD_BYTES
    summary_bytes: int = MAX_SUMMARY_BYTES
    combined_input_bytes: int = MAX_COMBINED_INPUT_BYTES

    def __post_init__(self) -> None:
        ceilings = {
            "nodes": MAX_NODES,
            "declared_memberships": MAX_DECLARED_MEMBERSHIPS,
            "sequence_characters": MAX_SEQUENCE_CHARACTERS,
            "projection_record_bytes": MAX_JSONL_RECORD_BYTES,
            "component_record_bytes": MAX_JSONL_RECORD_BYTES,
            "summary_bytes": MAX_SUMMARY_BYTES,
            "combined_input_bytes": MAX_COMBINED_INPUT_BYTES,
        }
        minima = {
            "nodes": 1,
            "declared_memberships": 0,
            "sequence_characters": 1,
            "projection_record_bytes": 1,
            "component_record_bytes": 1,
            "summary_bytes": 1,
            "combined_input_bytes": 1,
        }
        for field in fields(self):
            value = getattr(self, field.name)
            require(type(value) is int, "invalid_limit")
            number = cast(int, value)
            require(
                minima[field.name] <= number <= ceilings[field.name],
                "limit_above_default_or_negative",
            )


@dataclass(frozen=True)
class DescriptiveFlags:
    old_exposure: bool
    preview_exposure: bool
    either_test: bool
    unresolved_current_xml: bool
    unresolved_current_xml_pfam: bool


@dataclass(frozen=True)
class SeedRecord:
    key: Key
    sequence: str
    declared_distinct_5mer_count: int
    flags: DescriptiveFlags


@dataclass(frozen=True)
class LoadedSeeds:
    records: tuple[SeedRecord, ...]
    seeds: Partition
    identities: SeedInputIdentities
    node_count: int
    component_count: int
    component_size_histogram: tuple[tuple[int, int], ...]
    limitations: tuple[str, ...]


class _Budget:
    def __init__(self, remaining: int) -> None:
        self.remaining = remaining


def _text(value: object, code: str) -> str:
    require(type(value) is str, code)
    return cast(str, value)


def _nat(value: object, code: str) -> int:
    require(type(value) is int and value >= 0, code)
    return cast(int, value)


def _false(value: object, code: str) -> None:
    require(type(value) is bool and value is False, code)


def _mapping(value: object, code: str) -> dict[str, object]:
    require(type(value) is dict, code)
    return cast(dict[str, object], value)


def _digest(value: object, check: Callable[[], None]) -> str:
    check()
    encoder = json.JSONEncoder(
        ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    hasher = sha256()
    for text in encoder.iterencode(value):
        hasher.update(text.encode("ascii"))
    hasher.update(b"\n")
    check()
    return hasher.hexdigest()


def _decode_json(text: str) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            require(key not in result, "duplicate_json_key")
            result[key] = value
        return result

    def constant(_: str) -> None:
        raise SeedLoaderError("nonfinite_json")

    def finite(value: str) -> float:
        try:
            number = float(value)
        except ValueError as exc:
            raise SeedLoaderError("nonfinite_json") from exc
        require(math.isfinite(number), "nonfinite_json")
        return number

    try:
        decoder = json.JSONDecoder(
            object_pairs_hook=pairs, parse_float=finite, parse_constant=constant,
        )
        parsed, end = decoder.raw_decode(text)
    except SeedLoaderError:
        raise
    except json.JSONDecodeError as exc:
        raise SeedLoaderError("malformed_json") from exc
    require(end == len(text), "trailing_json_garbage")
    return parsed


def _object(text: str, code: str) -> dict[str, object]:
    parsed = _decode_json(text)
    require(type(parsed) is dict, code)
    return cast(dict[str, object], parsed)


def _sha256_hex(value: object, code: str) -> str:
    require(type(value) is str, code)
    text = cast(str, value)
    require(SHA256_HEX.fullmatch(text) is not None, code)
    return text


def _artifact(value: object, code: str) -> ArtifactIdentity:
    mapping = _mapping(value, code)
    require("bytes" in mapping and "sha256" in mapping, code)
    count = _nat(mapping["bytes"], code)
    require(count > 0, code)
    return ArtifactIdentity(count, _sha256_hex(mapping["sha256"], code))


def _utf8(payload: bytes, code: str) -> str:
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SeedLoaderError(code) from exc


class _HashLike(Protocol):
    def update(self, data: bytes, /) -> None: ...


def _readline(
    stream: BinaryIO,
    record_limit: int,
    budget: _Budget,
    hasher: _HashLike,
    check: Callable[[], None],
    oversize_code: str,
) -> bytes | None:
    check()
    cap = record_limit if budget.remaining >= record_limit else budget.remaining
    line = stream.readline(cap + 1)
    if line == b"":
        return None
    if len(line) > cap or not line.endswith(b"\n"):
        if len(line) > cap and budget.remaining < record_limit:
            raise SeedLoaderError("combined_input_byte_cap")
        raise SeedLoaderError(oversize_code)
    check()
    hasher.update(line)
    budget.remaining -= len(line)
    payload = line[:-1]
    require(payload != b"", "empty_jsonl_record")
    return payload


def _jsonl_payloads(
    stream: BinaryIO,
    record_limit: int,
    budget: _Budget,
    hasher: _HashLike,
    check: Callable[[], None],
    oversize_code: str,
) -> Iterator[bytes]:
    while True:
        payload = _readline(stream, record_limit, budget, hasher, check, oversize_code)
        if payload is None:
            return
        yield payload


def _read_summary(
    stream: BinaryIO,
    summary_limit: int,
    budget: _Budget,
    check: Callable[[], None],
) -> tuple[bytes, int, str]:
    check()
    cap = summary_limit if budget.remaining >= summary_limit else budget.remaining
    chunk_limit = 64 * 1024
    parts: list[bytes] = []
    total = 0
    while True:
        check()
        remaining = cap - total
        request = min(chunk_limit, remaining + 1)
        piece = stream.read(request)
        if piece == b"":
            break
        if total + len(piece) > cap:
            if budget.remaining < summary_limit:
                raise SeedLoaderError("combined_input_byte_cap")
            raise SeedLoaderError("summary_oversize_or_unterminated")
        check()
        parts.append(piece)
        total += len(piece)
    data = b"".join(parts)
    if data == b"" or not data.endswith(b"\n"):
        raise SeedLoaderError("summary_oversize_or_unterminated")
    check()
    hasher = sha256()
    hasher.update(data)
    check()
    budget.remaining -= total
    payload = data[:-1]
    require(payload != b"", "empty_summary")
    return payload, total, hasher.hexdigest()


def _pool(value: object) -> Pool:
    require(value in ("old", "pika"), "pool_invalid")
    return cast(Pool, value)


def _qualified_key(value: object, pool: Pool, accession: str) -> Key:
    require(type(value) is list and len(cast(list[object], value)) == 2, "qualified_key_invalid")
    left, right = cast(list[object], value)
    require(left == pool and right == accession, "qualified_key_mismatch")
    return pool, accession


def _member_key(value: object) -> Key:
    require(type(value) is list and len(cast(list[object], value)) == 2, "component_members_invalid")
    left, right = cast(list[object], value)
    pool = _pool(left)
    accession = _text(right, "component_members_invalid")
    require(ACCESSION.fullmatch(accession) is not None, "component_members_invalid")
    return pool, accession


def _flags(value: object) -> DescriptiveFlags:
    mapping = _mapping(value, "descriptive_flags_invalid")
    require(set(mapping) == set(FLAG_NAMES), "descriptive_flags_invalid")
    parsed: dict[str, bool] = {}
    for name in FLAG_NAMES:
        flag = mapping[name]
        require(type(flag) is bool, "descriptive_flags_invalid")
        parsed[name] = cast(bool, flag)
    return DescriptiveFlags(
        parsed["old_exposure"], parsed["preview_exposure"], parsed["either_test"],
        parsed["unresolved_current_xml"], parsed["unresolved_current_xml_pfam"],
    )


def _histogram(
    value: object,
    code: str,
    *,
    minimum_size: int,
) -> tuple[tuple[int, int], ...]:
    require(type(value) is list, code)
    pairs: list[tuple[int, int]] = []
    previous: int | None = None
    for item in cast(list[object], value):
        require(type(item) is list and len(cast(list[object], item)) == 2, code)
        size_obj, count_obj = cast(list[object], item)
        size = _int_in_range(size_obj, code, minimum=minimum_size, maximum=MAX_DECLARED_MEMBERSHIPS)
        count = _int_in_range(count_obj, code, minimum=1, maximum=MAX_NODES)
        require(previous is None or size > previous, code)
        previous = size
        pairs.append((size, count))
    return tuple(pairs)


def _flag_count_map(value: object, code: str) -> dict[str, int]:
    mapping = _mapping(value, code)
    require(set(mapping) == set(FLAG_NAMES), code)
    parsed: dict[str, int] = {}
    for name in FLAG_NAMES:
        parsed[name] = _nat(mapping[name], code)
    return parsed


def _source_assertions(value: object, check: Callable[[], None]) -> tuple[tuple[str, str], ...]:
    require(type(value) is list, "source_assertions_invalid")
    items: list[tuple[str, str]] = []
    names: set[str] = set()
    for item in cast(list[object], value):
        check()
        require(type(item) is list and len(cast(list[object], item)) == 2, "source_assertions_invalid")
        name_obj, identity_obj = cast(list[object], item)
        name = _text(name_obj, "source_assertions_invalid")
        identity = _text(identity_obj, "source_assertions_invalid")
        require(bool(name.strip()) and bool(identity.strip()), "source_assertions_invalid")
        require(name not in names, "source_assertions_invalid")
        names.add(name)
        items.append((name, identity))
    ordered = tuple(items)
    require(ordered == tuple(sorted(ordered)), "source_assertions_invalid")
    require(names == set(SOURCE_NAMES) and len(ordered) == len(SOURCE_NAMES), "source_assertions_invalid")
    return ordered


def _projection_record(
    row: dict[str, object],
    *,
    limits: LoaderLimits,
    policy: str | None,
    binding: str | None,
    check: Callable[[], None],
) -> tuple[SeedRecord, str, str]:
    check()
    required = (
        "schema", "relation_policy", "source_binding_identity", "pool", "accession",
        "qualified_key", "sequence", "sequence_sha256", "distinct_5mer_count",
        "descriptive_flags", *LIMITATION_NAMES,
    )
    for name in required:
        require(name in row, "projection_field_missing")
    require(row["schema"] == PROJECTION_SCHEMA, "projection_schema_invalid")
    relation_policy = _text(row["relation_policy"], "relation_policy_invalid")
    require(relation_policy == RELATION_POLICY, "relation_policy_mismatch")
    if policy is not None:
        require(relation_policy == policy, "relation_policy_mismatch")
    source_binding = _sha256_hex(row["source_binding_identity"], "source_binding_invalid")
    if binding is not None:
        require(source_binding == binding, "source_binding_mismatch")
    for name in LIMITATION_NAMES:
        _false(row[name], "projection_limitation_not_false")
    pool = _pool(row["pool"])
    accession = _text(row["accession"], "accession_invalid")
    require(ACCESSION.fullmatch(accession) is not None, "accession_invalid")
    key = _qualified_key(row["qualified_key"], pool, accession)
    sequence = _text(row["sequence"], "sequence_invalid")
    require(sequence != "", "sequence_empty")
    require(len(sequence) <= limits.sequence_characters, "sequence_character_cap")
    digest = _sha256_hex(row["sequence_sha256"], "sequence_sha256_invalid")
    require(digest == sha256(sequence.encode("utf-8")).hexdigest(), "sequence_sha256_mismatch")
    declared = _nat(row["distinct_5mer_count"], "distinct_5mer_count_invalid")
    length = len(sequence)
    if length < 5:
        require(declared == 0, "distinct_5mer_count_invalid")
    else:
        require(1 <= declared <= length - 4, "distinct_5mer_count_invalid")
    flags = _flags(row["descriptive_flags"])
    return SeedRecord(key, sequence, declared, flags), relation_policy, source_binding


def load_seeds(
    projection: BinaryIO,
    components: BinaryIO,
    summary: BinaryIO,
    *,
    expected: SeedInputIdentities,
    limits: LoaderLimits | None = None,
    check: Callable[[], None],
) -> LoadedSeeds:
    """Load literal sequence records and the complete immutable seed partition.

    Expected stream identities are required. Parser ceilings are not a 16 GiB fit
    proof or resource reservation. Builtin JSON decoding/sorting is not immediately
    interruptible.
    """
    check()
    if limits is None:
        limits = LoaderLimits()
    require(type(limits) is LoaderLimits, "invalid_limit")
    require(type(expected) is SeedInputIdentities, "invalid_expected_identity")
    require(
        len({id(projection), id(components), id(summary)}) == 3,
        "distinct_input_streams_required",
    )
    budget = _Budget(limits.combined_input_bytes)

    projection_hasher = sha256()
    records: list[SeedRecord] = []
    by_key: dict[Key, SeedRecord] = {}
    previous: Key | None = None
    policy: str | None = None
    binding: str | None = None
    memberships = 0
    for payload in _jsonl_payloads(
        projection, limits.projection_record_bytes, budget, projection_hasher, check,
        "jsonl_record_oversize_or_unterminated",
    ):
        check()
        text = _utf8(payload, "jsonl_record_not_utf8")
        check()
        row = _object(text, "projection_row_not_object")
        record, policy, binding = _projection_record(
            row, limits=limits, policy=policy, binding=binding, check=check,
        )
        require(previous is None or record.key > previous, "qualified_key_order")
        previous = record.key
        require(len(records) < limits.nodes, "nodes_cap")
        require(
            record.declared_distinct_5mer_count <= limits.declared_memberships - memberships,
            "memberships_cap",
        )
        memberships += record.declared_distinct_5mer_count
        records.append(record)
        by_key[record.key] = record
    require(bool(records) and policy is not None and binding is not None, "empty_projection")
    projection_identity = ArtifactIdentity(
        limits.combined_input_bytes - budget.remaining, projection_hasher.hexdigest(),
    )
    require(projection_identity == expected.projection, "expected_projection_identity_mismatch")
    assert policy is not None and binding is not None

    node_flags: Counter[str] = Counter()
    for record in records:
        check()
        for name in FLAG_NAMES:
            if getattr(record.flags, name):
                node_flags[name] += 1
    binding_base = {
        "relation_policy": RELATION_POLICY,
        "projection_sha256": projection_identity.sha256,
        "source_binding_identity": binding,
        **{name: False for name in LIMITATION_NAMES},
    }
    remaining_after_projection = budget.remaining
    component_hasher = sha256()
    seeds: list[tuple[Key, ...]] = []
    seen_keys: set[Key] = set()
    previous_first: Key | None = None
    component_sizes: Counter[int] = Counter()
    affected: Counter[str] = Counter()
    for payload in _jsonl_payloads(
        components, limits.component_record_bytes, budget, component_hasher, check,
        "jsonl_record_oversize_or_unterminated",
    ):
        check()
        text = _utf8(payload, "jsonl_record_not_utf8")
        check()
        row = _object(text, "component_row_not_object")
        for name in (
            "schema", "relation_policy", "projection_sha256", "source_binding_identity",
            "component_sha256", "members", "descriptive_flags", *LIMITATION_NAMES,
        ):
            require(name in row, "component_field_missing")
        require(row["schema"] == COMPONENT_SCHEMA, "component_schema_invalid")
        require(
            _text(row["relation_policy"], "relation_policy_invalid") == RELATION_POLICY,
            "relation_policy_mismatch",
        )
        require(
            _sha256_hex(row["projection_sha256"], "component_projection_sha256_invalid")
            == projection_identity.sha256,
            "component_projection_sha256_mismatch",
        )
        require(
            _sha256_hex(row["source_binding_identity"], "source_binding_invalid") == binding,
            "source_binding_mismatch",
        )
        for name in LIMITATION_NAMES:
            _false(row[name], "component_limitation_not_false")
        members_value = row["members"]
        require(
            type(members_value) is list and len(cast(list[object], members_value)) >= 1,
            "component_members_invalid",
        )
        canonical: list[Key] = []
        flag_or = {name: False for name in FLAG_NAMES}
        previous_member: Key | None = None
        for raw in cast(list[object], members_value):
            check()
            key = _member_key(raw)
            require(key in by_key, "component_member_unknown")
            require(key not in seen_keys, "component_member_duplicate")
            require(previous_member is None or key > previous_member, "component_member_order")
            previous_member = key
            seen_keys.add(key)
            member = by_key[key]
            canonical.append(member.key)
            for name in FLAG_NAMES:
                if getattr(member.flags, name):
                    flag_or[name] = True
        require(previous_first is None or canonical[0] > previous_first, "component_first_key_order")
        previous_first = canonical[0]
        members = tuple(canonical)
        computed = _digest({**binding_base, "members": members}, check)
        require(
            _sha256_hex(row["component_sha256"], "component_sha256_invalid") == computed,
            "component_sha256_mismatch",
        )
        declared_flags = _flags(row["descriptive_flags"])
        for name in FLAG_NAMES:
            require(getattr(declared_flags, name) is flag_or[name], "component_flags_mismatch")
        for name, marked in flag_or.items():
            if marked:
                affected[name] += len(members)
        component_sizes[len(members)] += 1
        seeds.append(members)
    require(bool(seeds), "empty_components")
    require(seen_keys == set(by_key), "component_coverage_incomplete")
    components_identity = ArtifactIdentity(
        remaining_after_projection - budget.remaining, component_hasher.hexdigest(),
    )
    require(components_identity == expected.components, "expected_components_identity_mismatch")

    summary_payload, summary_size, summary_digest = _read_summary(
        summary, limits.summary_bytes, budget, check,
    )
    check()
    summary_text = _utf8(summary_payload, "summary_not_utf8")
    check()
    summary_row = _object(summary_text, "summary_object_required")
    summary_identity = ArtifactIdentity(summary_size, summary_digest)
    require(summary_identity == expected.summary, "expected_summary_identity_mismatch")

    for name in (
        "schema", "relation_policy", "source_assertions", "source_binding_identity",
        "projection_sha256", "projection_artifact", "components_artifact",
        "observed_records", "partial_relation_component_count",
        "partial_relation_component_size_histogram", "shingles",
        "affected_record_lower_bounds", "node_descriptive_flag_counts",
        *LIMITATION_NAMES, *LIBRARY_FLAGS,
    ):
        require(name in summary_row, "summary_field_missing")
    require(summary_row["schema"] == SUMMARY_SCHEMA, "summary_schema_invalid")
    require(
        _text(summary_row["relation_policy"], "relation_policy_invalid") == RELATION_POLICY,
        "relation_policy_mismatch",
    )
    for name in (*LIMITATION_NAMES, *LIBRARY_FLAGS):
        _false(summary_row[name], "summary_limitation_not_false")
    assertions = _source_assertions(summary_row["source_assertions"], check)
    computed_binding = _digest(assertions, check)
    require(
        _sha256_hex(summary_row["source_binding_identity"], "source_binding_invalid")
        == computed_binding,
        "source_binding_identity_mismatch",
    )
    require(computed_binding == binding, "source_binding_mismatch")
    require(
        _sha256_hex(summary_row["projection_sha256"], "summary_projection_sha256_invalid")
        == projection_identity.sha256,
        "summary_projection_sha256_mismatch",
    )
    require(
        _artifact(summary_row["projection_artifact"], "projection_artifact_invalid")
        == projection_identity,
        "summary_projection_artifact_mismatch",
    )
    require(
        _artifact(summary_row["components_artifact"], "components_artifact_invalid")
        == components_identity,
        "summary_components_artifact_mismatch",
    )
    observed_raw = _mapping(summary_row["observed_records"], "observed_records_invalid")
    require(set(observed_raw) == {"old", "pika"}, "observed_records_invalid")
    observed = {
        "old": _nat(observed_raw["old"], "observed_records_invalid"),
        "pika": _nat(observed_raw["pika"], "observed_records_invalid"),
    }
    pools = {"old": 0, "pika": 0}
    for record in records:
        check()
        pools[record.key[0]] += 1
    require(pools == observed, "observed_records_mismatch")
    total_nodes = len(records)
    require(observed["old"] + observed["pika"] == total_nodes, "observed_records_invalid")
    require(total_nodes > 0, "observed_records_invalid")
    component_count = len(seeds)
    require(
        _nat(summary_row["partial_relation_component_count"], "component_count_invalid")
        == component_count,
        "component_count_mismatch",
    )
    histogram = _histogram(
        summary_row["partial_relation_component_size_histogram"],
        "component_histogram_invalid",
        minimum_size=1,
    )
    computed_histogram = tuple(sorted(component_sizes.items()))
    require(histogram == computed_histogram, "component_histogram_mismatch")
    require(sum(size * count for size, count in histogram) == total_nodes, "component_histogram_mismatch")
    require(sum(count for _, count in histogram) == component_count, "component_histogram_mismatch")
    shingles = _mapping(summary_row["shingles"], "shingles_invalid")
    require(
        "per_record_distinct_count_histogram" in shingles and "total_postings" in shingles,
        "shingles_invalid",
    )
    declared_sizes = [record.declared_distinct_5mer_count for record in records]
    shingle_histogram = _histogram(
        shingles["per_record_distinct_count_histogram"],
        "shingle_histogram_invalid",
        minimum_size=0,
    )
    require(
        shingle_histogram == tuple(sorted(Counter(declared_sizes).items())),
        "shingle_histogram_mismatch",
    )
    require(
        _nat(shingles["total_postings"], "total_postings_invalid") == memberships,
        "total_postings_mismatch",
    )
    require(
        _flag_count_map(summary_row["node_descriptive_flag_counts"], "node_flag_counts_invalid")
        == {name: node_flags[name] for name in FLAG_NAMES},
        "node_flag_count_mismatch",
    )
    require(
        _flag_count_map(summary_row["affected_record_lower_bounds"], "affected_counts_invalid")
        == {name: affected[name] for name in FLAG_NAMES},
        "affected_count_mismatch",
    )
    return LoadedSeeds(
        records=tuple(records),
        seeds=tuple(seeds),
        identities=SeedInputIdentities(projection_identity, components_identity, summary_identity),
        node_count=total_nodes,
        component_count=component_count,
        component_size_histogram=histogram,
        limitations=LIMITATION_STATEMENTS,
    )


__all__ = (
    "ArtifactIdentity",
    "DescriptiveFlags",
    "Key",
    "LoadedSeeds",
    "LoaderLimits",
    "Partition",
    "SeedInputIdentities",
    "SeedLoaderError",
    "SeedRecord",
    "load_seeds",
)
