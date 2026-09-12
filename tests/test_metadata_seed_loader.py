"""Bounded synthetic checks for the persisted metadata-census seed loader."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Callable, Literal, cast
import io
import json
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer.metadata_seed_loader import (  # noqa: E402
    ArtifactIdentity,
    LoadedSeeds,
    LoaderLimits,
    SeedInputIdentities,
    SeedLoaderError,
    load_seeds,
)

Pool = Literal["old", "pika"]
POLICY = "accession_exactsequence_rawpfam_pikauniref_only_metadata_census_v1"
PROJECTION_SCHEMA = "metadata_census_projection_v1"
COMPONENT_SCHEMA = "metadata_census_component_v1"
SUMMARY_SCHEMA = "metadata_census_summary_v1"
SOURCE_NAMES = (
    "current_xml", "old_jsonl", "old_pfam_tsv", "pika_evogroup_split",
    "pika_sequences", "pika_uniref_split", "preview_selection",
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
ASSERTIONS = tuple((name, f"asserted-{name}") for name in SOURCE_NAMES)


@dataclass(frozen=True)
class Bundle:
    projection: bytes
    components: bytes
    summary: bytes
    expected: SeedInputIdentities
    component_digests: tuple[str, ...]
    source_binding: str


class Watch(io.BytesIO):
    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.touched = False

    def read(self, size: int | None = -1) -> bytes:
        self.touched = True
        return super().read(-1 if size is None else size)

    def readline(self, size: int | None = -1) -> bytes:
        self.touched = True
        return super().readline(-1 if size is None else size)


class ShortReads(io.BytesIO):
    """Finite size-respecting stream; nonempty short reads are not EOF."""

    def __init__(self, data: bytes, chunk: int) -> None:
        super().__init__(data)
        self.chunk = chunk
        self.eof_seen = False
        self.calls: list[tuple[int, int]] = []

    def read(self, size: int | None = -1) -> bytes:
        if type(size) is not int or size < 1:
            raise AssertionError("unbounded_or_nonpositive_read")
        data = super().read(min(size, self.chunk))
        self.calls.append((size, len(data)))
        if data == b"":
            self.eof_seen = True
        return data


def dump(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("ascii") + b"\n"


def digest_of(value: object) -> str:
    return sha256(dump(value)).hexdigest()


def declared_count(sequence: str) -> int:
    if len(sequence) < 5:
        return 0
    return len({sequence[index:index + 5] for index in range(len(sequence) - 4)})


def flags_for(
    pool: str,
    *,
    preview: bool = False,
    test: bool = False,
    unresolved_xml: bool = False,
    unresolved_pfam: bool = False,
) -> dict[str, bool]:
    return {
        "old_exposure": pool == "old",
        "preview_exposure": preview,
        "either_test": test,
        "unresolved_current_xml": unresolved_xml,
        "unresolved_current_xml_pfam": unresolved_pfam,
    }


def identities_for(projection: bytes, components: bytes, summary: bytes) -> SeedInputIdentities:
    return SeedInputIdentities(
        ArtifactIdentity(len(projection), sha256(projection).hexdigest()),
        ArtifactIdentity(len(components), sha256(components).hexdigest()),
        ArtifactIdentity(len(summary), sha256(summary).hexdigest()),
    )


def projection_row(
    pool: str,
    accession: str,
    sequence: str,
    binding: str,
    flags: dict[str, bool],
) -> dict[str, object]:
    row: dict[str, object] = {
        "accession": accession,
        "descriptive_flags": dict(flags),
        "distinct_5mer_count": declared_count(sequence),
        "pool": pool,
        "qualified_key": [pool, accession],
        "relation_policy": POLICY,
        "schema": PROJECTION_SCHEMA,
        "sequence": sequence,
        "sequence_sha256": sha256(sequence.encode("utf-8")).hexdigest(),
        "source_binding_identity": binding,
        "unretained_label": "discard-me",
        "xml_linkage": "absent",
    }
    for name in LIMITATION_NAMES:
        row[name] = False
    return row


def component_row(
    members: tuple[tuple[str, str], ...],
    records: dict[tuple[str, str], dict[str, object]],
    binding: str,
    projection_sha256: str,
) -> dict[str, object]:
    flag_or = {name: False for name in FLAG_NAMES}
    for key in members:
        flags = records[key]["descriptive_flags"]
        assert isinstance(flags, dict)
        for name in FLAG_NAMES:
            if flags[name]:
                flag_or[name] = True
    payload = {
        "relation_policy": POLICY,
        "projection_sha256": projection_sha256,
        "source_binding_identity": binding,
        **{name: False for name in LIMITATION_NAMES},
        "members": [list(key) for key in members],
    }
    identity = digest_of(payload)
    row: dict[str, object] = {
        "schema": COMPONENT_SCHEMA,
        "component_sha256": identity,
        "descriptive_flags": flag_or,
        **payload,
    }
    return row


def summary_row(
    records: list[dict[str, object]],
    components: list[dict[str, object]],
    binding: str,
    projection: bytes,
    components_bytes: bytes,
) -> dict[str, object]:
    pools = {"old": 0, "pika": 0}
    sizes: list[int] = []
    node_flags = {name: 0 for name in FLAG_NAMES}
    for row in records:
        pool = str(row["pool"])
        pools[pool] += 1
        count = row["distinct_5mer_count"]
        assert type(count) is int
        sizes.append(count)
        flags = row["descriptive_flags"]
        assert isinstance(flags, dict)
        flag_map = cast(dict[str, object], flags)
        for name in FLAG_NAMES:
            if flag_map[name]:
                node_flags[name] += 1
    affected = {name: 0 for name in FLAG_NAMES}
    component_sizes: Counter[int] = Counter()
    for row in components:
        raw_members = row["members"]
        assert isinstance(raw_members, list)
        members = cast(list[object], raw_members)
        component_sizes[len(members)] += 1
        flags = row["descriptive_flags"]
        assert isinstance(flags, dict)
        flag_map = cast(dict[str, object], flags)
        for name in FLAG_NAMES:
            if flag_map[name]:
                affected[name] += len(members)
    payload: dict[str, object] = {
        "schema": SUMMARY_SCHEMA,
        "relation_policy": POLICY,
        "source_assertions": [list(item) for item in ASSERTIONS],
        "source_binding_identity": binding,
        "projection_sha256": sha256(projection).hexdigest(),
        "projection_artifact": {
            "bytes": len(projection), "sha256": sha256(projection).hexdigest(),
        },
        "components_artifact": {
            "bytes": len(components_bytes), "sha256": sha256(components_bytes).hexdigest(),
        },
        "observed_records": {"old": pools["old"], "pika": pools["pika"]},
        "partial_relation_component_count": len(components),
        "partial_relation_component_size_histogram": [
            list(item) for item in sorted(component_sizes.items())
        ],
        "node_descriptive_flag_counts": node_flags,
        "affected_record_lower_bounds": affected,
        "shingles": {
            "k": 5,
            "W": 0,
            "total_postings": sum(sizes),
            "per_record_distinct_count_histogram": [
                list(item) for item in sorted(Counter(sizes).items())
            ],
            "posting_frequency_histogram": [[1, 1]],
        },
        "interpretation": "fabricated unrelated field",
    }
    for name in (*LIMITATION_NAMES, *LIBRARY_FLAGS):
        payload[name] = False
    return payload


def build_bundle(
    specs: list[tuple[str, str, str, dict[str, bool]]],
    groups: list[tuple[tuple[str, str], ...]],
) -> Bundle:
    binding = digest_of(ASSERTIONS)
    rows = [
        projection_row(pool, accession, sequence, binding, flags)
        for pool, accession, sequence, flags in specs
    ]
    rows.sort(key=lambda item: (str(item["pool"]), str(item["accession"])))
    projection = b"".join(dump(row) for row in rows)
    by_key = {
        (str(row["pool"]), str(row["accession"])): row for row in rows
    }
    ordered_groups = sorted(tuple(sorted(group)) for group in groups)
    component_rows = [
        component_row(group, by_key, binding, sha256(projection).hexdigest())
        for group in ordered_groups
    ]
    components = b"".join(dump(row) for row in component_rows)
    summary = dump(summary_row(rows, component_rows, binding, projection, components))
    return Bundle(
        projection,
        components,
        summary,
        identities_for(projection, components, summary),
        tuple(str(row["component_sha256"]) for row in component_rows),
        binding,
    )


def full_bundle() -> Bundle:
    return build_bundle(
        [
            ("old", "P12345", "ABCDE", flags_for("old")),
            ("pika", "P12345", "αβγδεζ", flags_for("pika")),
            ("pika", "P02345", "AB", flags_for("pika")),
            (
                "pika", "A0A023PGB6", "aA aA\nXYZ",
                flags_for("pika", preview=True, test=True, unresolved_xml=True, unresolved_pfam=True),
            ),
            ("pika", "Q9Y6K8", "AAAAAAA", flags_for("pika", unresolved_pfam=True)),
        ],
        [
            (("old", "P12345"), ("pika", "P12345")),
            (("pika", "P02345"),),
            (("pika", "A0A023PGB6"),),
            (("pika", "Q9Y6K8"),),
        ],
    )


def tiny_bundle() -> Bundle:
    return build_bundle(
        [
            ("old", "P00001", "ABCDE", flags_for("old")),
            ("pika", "P00001", "FGHIJ", flags_for("pika")),
        ],
        [(("old", "P00001"),), (("pika", "P00001"),)],
    )


def load_bundle(
    bundle: Bundle,
    *,
    limits: LoaderLimits | None = None,
    check: Callable[[], None] | None = None,
    projection: bytes | None = None,
    components: bytes | None = None,
    summary: bytes | None = None,
    expected: SeedInputIdentities | None = None,
) -> LoadedSeeds:
    return load_seeds(
        io.BytesIO(bundle.projection if projection is None else projection),
        io.BytesIO(bundle.components if components is None else components),
        io.BytesIO(bundle.summary if summary is None else summary),
        expected=bundle.expected if expected is None else expected,
        limits=limits,
        check=(lambda: None) if check is None else check,
    )


def refuse(
    bundle: Bundle,
    code: str,
    *,
    limits: LoaderLimits | None = None,
    check: Callable[[], None] | None = None,
    projection: bytes | None = None,
    components: bytes | None = None,
    summary: bytes | None = None,
    expected: SeedInputIdentities | None = None,
) -> None:
    returned: list[LoadedSeeds] = []
    try:
        returned.append(load_bundle(
            bundle, limits=limits, check=check, projection=projection,
            components=components, summary=summary, expected=expected,
        ))
    except SeedLoaderError as exc:
        if returned:
            raise AssertionError("partial success returned after failure") from exc
        if exc.code != code:
            raise AssertionError(f"expected {code}, got {exc.code}") from exc
        return
    raise AssertionError(f"expected {code}, succeeded")


class MetadataSeedLoaderTests(unittest.TestCase):
    def test_full_synthetic_partition_and_canonical_digests(self) -> None:
        bundle = full_bundle()
        loaded = load_bundle(bundle)
        self.assertEqual(loaded.node_count, 5)
        self.assertEqual(loaded.component_count, 4)
        self.assertEqual(loaded.component_size_histogram, ((1, 3), (2, 1)))
        self.assertEqual(loaded.identities, bundle.expected)
        self.assertEqual(loaded.limitations[2], "real 277167-node cardinality is not qualified")
        keys = [record.key for record in loaded.records]
        self.assertEqual(
            keys,
            [
                ("old", "P12345"),
                ("pika", "A0A023PGB6"),
                ("pika", "P02345"),
                ("pika", "P12345"),
                ("pika", "Q9Y6K8"),
            ],
        )
        self.assertIs(loaded.seeds[0][0], loaded.records[0].key)
        self.assertIs(loaded.seeds[0][1], loaded.records[3].key)
        self.assertEqual(len(loaded.seeds[0]), 2)
        by_key = {record.key: record for record in loaded.records}
        self.assertEqual(by_key[("old", "P12345")].sequence, "ABCDE")
        self.assertEqual(by_key[("pika", "P12345")].sequence, "αβγδεζ")
        self.assertEqual(by_key[("pika", "P02345")].declared_distinct_5mer_count, 0)
        self.assertEqual(by_key[("pika", "A0A023PGB6")].sequence, "aA aA\nXYZ")
        self.assertTrue(by_key[("old", "P12345")].flags.old_exposure)
        exposed = by_key[("pika", "A0A023PGB6")]
        self.assertTrue(exposed.flags.preview_exposure)
        self.assertTrue(exposed.flags.either_test)
        self.assertTrue(exposed.flags.unresolved_current_xml)
        self.assertTrue(exposed.flags.unresolved_current_xml_pfam)
        self.assertTrue(by_key[("pika", "Q9Y6K8")].flags.unresolved_current_xml_pfam)
        self.assertEqual(digest_of(ASSERTIONS), bundle.source_binding)
        self.assertEqual(len(bundle.component_digests), 4)
        self.assertTrue(all(len(item) == 64 for item in bundle.component_digests))
        self.assertIn("not recomputed", " ".join(loaded.limitations))
        self.assertIn("not verified by this library", " ".join(loaded.limitations))

    def test_caps_exact_versus_plus_one(self) -> None:
        bundle = tiny_bundle()
        projection_len = max(len(line) + 1 for line in bundle.projection.split(b"\n") if line)
        component_len = max(len(line) + 1 for line in bundle.components.split(b"\n") if line)
        combined = len(bundle.projection) + len(bundle.components) + len(bundle.summary)
        cases: list[tuple[str, LoaderLimits, str | None]] = [
            ("nodes_exact", LoaderLimits(nodes=2), None),
            ("nodes_plus_one", LoaderLimits(nodes=1), "nodes_cap"),
            ("memberships_exact", LoaderLimits(declared_memberships=2), None),
            ("memberships_plus_one", LoaderLimits(declared_memberships=1), "memberships_cap"),
            ("sequence_exact", LoaderLimits(sequence_characters=5), None),
            ("sequence_plus_one", LoaderLimits(sequence_characters=4), "sequence_character_cap"),
            ("projection_line_exact", LoaderLimits(projection_record_bytes=projection_len), None),
            (
                "projection_line_plus_one",
                LoaderLimits(projection_record_bytes=projection_len - 1),
                "jsonl_record_oversize_or_unterminated",
            ),
            ("component_line_exact", LoaderLimits(component_record_bytes=component_len), None),
            (
                "component_line_plus_one",
                LoaderLimits(component_record_bytes=component_len - 1),
                "jsonl_record_oversize_or_unterminated",
            ),
            ("summary_exact", LoaderLimits(summary_bytes=len(bundle.summary)), None),
            (
                "summary_plus_one",
                LoaderLimits(summary_bytes=len(bundle.summary) - 1),
                "summary_oversize_or_unterminated",
            ),
            ("combined_exact", LoaderLimits(combined_input_bytes=combined), None),
            ("combined_plus_one", LoaderLimits(combined_input_bytes=combined - 1), "combined_input_byte_cap"),
        ]
        for name, limits, code in cases:
            with self.subTest(name=name):
                if code is None:
                    loaded = load_bundle(bundle, limits=limits)
                    self.assertEqual(loaded.node_count, 2)
                else:
                    refuse(bundle, code, limits=limits)

    def test_malformed_envelopes(self) -> None:
        bundle = tiny_bundle()
        duplicate = b'{"schema":"metadata_census_projection_v1","schema":"dup"}\n'
        nonfinite = bundle.projection.replace(
            b'"distinct_5mer_count":1', b'"distinct_5mer_count":NaN', 1,
        )
        overflow = bundle.projection.replace(
            b'"distinct_5mer_count":1', b'"distinct_5mer_count":1e999', 1,
        )
        cases: list[tuple[str, bytes | None, bytes | None, bytes | None, str]] = [
            ("unterminated_projection", bundle.projection.rstrip(b"\n"), None, None,
             "jsonl_record_oversize_or_unterminated"),
            ("unterminated_summary", None, None, bundle.summary.rstrip(b"\n"),
             "summary_oversize_or_unterminated"),
            ("empty_projection_line", b"\n" + bundle.projection, None, None, "empty_jsonl_record"),
            ("duplicate_json_key", duplicate, None, None, "duplicate_json_key"),
            ("nonfinite_nan", nonfinite, None, None, "nonfinite_json"),
            ("nonfinite_overflow", overflow, None, None, "nonfinite_json"),
            ("malformed_json", b"{not-json\n", None, None, "malformed_json"),
        ]
        for name, projection, components, summary, code in cases:
            with self.subTest(name=name):
                used_p = bundle.projection if projection is None else projection
                used_c = bundle.components if components is None else components
                used_s = bundle.summary if summary is None else summary
                refuse(
                    bundle, code, projection=used_p, components=used_c, summary=used_s,
                    expected=identities_for(used_p, used_c, used_s),
                )

    def test_binding_order_coverage_and_histogram_refusals(self) -> None:
        bundle = tiny_bundle()
        wrong_expected = replace(
            bundle.expected,
            projection=ArtifactIdentity(bundle.expected.projection.bytes, "ab" * 32),
        )
        reversed_rows = b"".join(reversed(bundle.projection.splitlines(keepends=True)))
        summary_obj = json.loads(bundle.summary)
        summary_obj["observed_records"] = {"old": 2, "pika": 0}
        bad_observed = dump(summary_obj)
        summary_obj = json.loads(bundle.summary)
        summary_obj["partial_relation_component_size_histogram"] = [[1, 9]]
        bad_histogram = dump(summary_obj)
        summary_obj = json.loads(bundle.summary)
        summary_obj["node_descriptive_flag_counts"]["old_exposure"] = 9
        bad_flags = dump(summary_obj)
        first, second = bundle.components.splitlines(keepends=True)
        swapped = second + first
        extra_member = json.loads(bundle.components.splitlines()[0])
        extra_member["members"] = [["old", "P00001"], ["pika", "P00001"]]
        extra_member["component_sha256"] = digest_of({
            "relation_policy": POLICY,
            "projection_sha256": sha256(bundle.projection).hexdigest(),
            "source_binding_identity": bundle.source_binding,
            **{name: False for name in LIMITATION_NAMES},
            "members": extra_member["members"],
        })
        extra_bytes = dump(extra_member) + bundle.components.splitlines(keepends=True)[1]
        cases: list[tuple[str, dict[str, object], str]] = [
            ("wrong_expected_hash", {"expected": wrong_expected}, "expected_projection_identity_mismatch"),
            (
                "key_order",
                {
                    "projection": reversed_rows,
                    "expected": identities_for(reversed_rows, bundle.components, bundle.summary),
                },
                "qualified_key_order",
            ),
            (
                "component_first_key_order",
                {
                    "components": swapped,
                    "expected": identities_for(bundle.projection, swapped, bundle.summary),
                },
                "component_first_key_order",
            ),
            (
                "coverage_duplicate_and_missing",
                {
                    "components": extra_bytes,
                    "expected": identities_for(bundle.projection, extra_bytes, bundle.summary),
                },
                "component_member_duplicate",
            ),
            (
                "observed_mismatch",
                {"summary": bad_observed, "expected": identities_for(bundle.projection, bundle.components, bad_observed)},
                "observed_records_mismatch",
            ),
            (
                "histogram_mismatch",
                {"summary": bad_histogram, "expected": identities_for(bundle.projection, bundle.components, bad_histogram)},
                "component_histogram_mismatch",
            ),
            (
                "flag_count_mismatch",
                {"summary": bad_flags, "expected": identities_for(bundle.projection, bundle.components, bad_flags)},
                "node_flag_count_mismatch",
            ),
        ]
        for name, kwargs, code in cases:
            with self.subTest(name=name):
                refuse(bundle, code, **kwargs)  # type: ignore[arg-type]

    def test_component_flag_and_sha_mismatch(self) -> None:
        bundle = full_bundle()
        rows = [json.loads(line) for line in bundle.components.splitlines() if line]
        rows[0]["descriptive_flags"]["old_exposure"] = False
        flipped = b"".join(dump(row) for row in rows)
        refuse(
            bundle, "component_flags_mismatch", components=flipped,
            expected=identities_for(bundle.projection, flipped, bundle.summary),
        )
        rows = [json.loads(line) for line in bundle.components.splitlines() if line]
        rows[0]["component_sha256"] = "cd" * 32
        bad_sha = b"".join(dump(row) for row in rows)
        refuse(
            bundle, "component_sha256_mismatch", components=bad_sha,
            expected=identities_for(bundle.projection, bad_sha, bundle.summary),
        )

    def test_limit_and_identity_constructors(self) -> None:
        with self.assertRaises(SeedLoaderError) as raised:
            LoaderLimits(nodes=True)  # type: ignore[arg-type]
        self.assertEqual(raised.exception.code, "invalid_limit")
        with self.assertRaises(SeedLoaderError) as raised:
            LoaderLimits(nodes=300_001)
        self.assertEqual(raised.exception.code, "limit_above_default_or_negative")
        with self.assertRaises(SeedLoaderError) as raised:
            ArtifactIdentity(bytes=0, sha256="ab" * 32)
        self.assertEqual(raised.exception.code, "invalid_expected_identity")
        with self.assertRaises(SeedLoaderError) as raised:
            ArtifactIdentity(bytes=1, sha256="AB" * 32)
        self.assertEqual(raised.exception.code, "invalid_expected_identity")
        bundle = tiny_bundle()
        shared = io.BytesIO(bundle.projection)
        with self.assertRaises(SeedLoaderError) as raised:
            load_seeds(
                shared, shared, io.BytesIO(bundle.summary),
                expected=bundle.expected, check=lambda: None,
            )
        self.assertEqual(raised.exception.code, "distinct_input_streams_required")

    def test_cancellation_before_reads_and_during_membership(self) -> None:
        bundle = full_bundle()
        projection = Watch(bundle.projection)
        components = Watch(bundle.components)
        summary = Watch(bundle.summary)

        def before() -> None:
            raise RuntimeError("cancelled-before")

        with self.assertRaises(RuntimeError):
            load_seeds(
                projection, components, summary, expected=bundle.expected, check=before,
            )
        self.assertFalse(projection.touched)
        self.assertFalse(components.touched)
        self.assertFalse(summary.touched)

        first_line = bundle.components.split(b"\n", 1)[0] + b"\n"
        tracked = Watch(bundle.components)
        after_line = {"n": 0}
        returned: list[LoadedSeeds] = []

        def during() -> None:
            if tracked.tell() >= len(first_line):
                after_line["n"] += 1
                if after_line["n"] >= 4:
                    raise RuntimeError("cancelled-membership")

        with self.assertRaises(RuntimeError) as raised:
            returned.append(load_seeds(
                io.BytesIO(bundle.projection), tracked, io.BytesIO(bundle.summary),
                expected=bundle.expected, check=during,
            ))
        self.assertEqual(str(raised.exception), "cancelled-membership")
        self.assertEqual(returned, [])
        self.assertGreaterEqual(after_line["n"], 4)
        self.assertTrue(tracked.touched)

    def test_summary_short_reads_eof_caps_and_cancellation(self) -> None:
        bundle = tiny_bundle()
        chunk_limit = 64 * 1024
        combined = len(bundle.projection) + len(bundle.components) + len(bundle.summary)
        trailer = b'{"unexpected_trailer":true}\n'

        def bounded(stream: ShortReads, *, cap: int) -> None:
            accepted = 0
            for request, got in stream.calls:
                remaining = cap - accepted
                self.assertGreater(request, 0)
                self.assertEqual(request, min(chunk_limit, remaining + 1))
                self.assertLessEqual(got, request)
                self.assertLessEqual(got, stream.chunk)
                if got == 0 or accepted + got > cap:
                    continue
                accepted += got
            self.assertLessEqual(accepted, cap)

        def load_summary(
            stream: ShortReads,
            *,
            limits: LoaderLimits | None = None,
            check: Callable[[], None] | None = None,
        ) -> LoadedSeeds:
            return load_seeds(
                io.BytesIO(bundle.projection),
                io.BytesIO(bundle.components),
                stream,
                expected=bundle.expected,
                limits=limits,
                check=(lambda: None) if check is None else check,
            )

        for chunk in (1, 17, len(bundle.summary)):
            with self.subTest(case="valid_short_eof", chunk=chunk):
                summary = ShortReads(bundle.summary, chunk)
                loaded = load_summary(summary)
                self.assertEqual(loaded.identities.summary, bundle.expected.summary)
                self.assertEqual(loaded.node_count, 2)
                self.assertTrue(summary.eof_seen)
                self.assertEqual(summary.tell(), len(bundle.summary))
                self.assertEqual(summary.calls[-1][1], 0)
                bounded(summary, cap=LoaderLimits().summary_bytes)

        with self.subTest(case="valid_summary_plus_trailer"):
            summary = ShortReads(bundle.summary + trailer, len(bundle.summary))
            returned: list[LoadedSeeds] = []
            with self.assertRaises(SeedLoaderError) as raised:
                returned.append(load_summary(summary))
            self.assertEqual(returned, [])
            self.assertEqual(raised.exception.code, "trailing_json_garbage")
            self.assertEqual(bundle.expected.summary.bytes, len(bundle.summary))
            bounded(summary, cap=LoaderLimits().summary_bytes)

        with self.subTest(case="summary_exact"):
            summary = ShortReads(bundle.summary, 1)
            loaded = load_summary(summary, limits=LoaderLimits(summary_bytes=len(bundle.summary)))
            self.assertEqual(loaded.identities.summary, bundle.expected.summary)
            self.assertTrue(summary.eof_seen)
            self.assertEqual(summary.calls[-1], (1, 0))
            bounded(summary, cap=len(bundle.summary))

        with self.subTest(case="summary_plus_one"):
            summary = ShortReads(bundle.summary + b"X", 1)
            returned: list[LoadedSeeds] = []
            with self.assertRaises(SeedLoaderError) as raised:
                returned.append(load_summary(
                    summary, limits=LoaderLimits(summary_bytes=len(bundle.summary) - 1),
                ))
            self.assertEqual(returned, [])
            self.assertEqual(raised.exception.code, "summary_oversize_or_unterminated")
            self.assertFalse(summary.eof_seen)
            self.assertLess(summary.tell(), len(bundle.summary) + 1)
            bounded(summary, cap=len(bundle.summary) - 1)

        with self.subTest(case="combined_exact"):
            summary = ShortReads(bundle.summary, 17)
            loaded = load_summary(summary, limits=LoaderLimits(combined_input_bytes=combined))
            self.assertEqual(loaded.identities.summary, bundle.expected.summary)
            self.assertTrue(summary.eof_seen)
            bounded(summary, cap=len(bundle.summary))

        with self.subTest(case="combined_plus_one"):
            summary = ShortReads(bundle.summary + b"X", 1)
            returned: list[LoadedSeeds] = []
            with self.assertRaises(SeedLoaderError) as raised:
                returned.append(load_summary(
                    summary, limits=LoaderLimits(combined_input_bytes=combined - 1),
                ))
            self.assertEqual(returned, [])
            self.assertEqual(raised.exception.code, "combined_input_byte_cap")
            self.assertFalse(summary.eof_seen)
            bounded(summary, cap=len(bundle.summary) - 1)

        with self.subTest(case="cancel_during_accumulation"):
            summary = ShortReads(bundle.summary, 1)
            returned: list[LoadedSeeds] = []

            def during() -> None:
                if summary.calls:
                    raise RuntimeError("cancelled-summary-accumulation")

            with self.assertRaises(RuntimeError) as raised:
                returned.append(load_summary(summary, check=during))
            self.assertEqual(str(raised.exception), "cancelled-summary-accumulation")
            self.assertEqual(returned, [])
            self.assertFalse(summary.eof_seen)
            self.assertGreaterEqual(len(summary.calls), 1)
            self.assertLess(summary.tell(), len(bundle.summary))
            bounded(summary, cap=LoaderLimits().summary_bytes)


if __name__ == "__main__":
    unittest.main()
