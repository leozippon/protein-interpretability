"""Immutable sequence-oracle sidecars for the generated-output supplement.

Retained R227 query coordinates recover coverage without another search. New
unconditional attempts use the same Pfam gathering thresholds and DIAMOND
configuration. Neither a no-hit result nor distance to this corpus proves
training disjointness or biochemical function.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from . import conditioned_generation as cg
from . import concept_injection as ci
from . import generation_evidence as ge
from . import homology
from .io import sha256_file
from .progen3_generation import POLICY as NATIVE_POLICY


def read_attempts(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate attempt identifier")
    for row in rows:
        if ge.sequence_hash(row["sequence"]) != row["sequence_sha256"]:
            raise ValueError(f"attempt sequence hash mismatch: {row['id']}")
    return rows


def best_hits(table: Path, sequences: dict[str, str]) -> dict[str, homology.Hit]:
    best: dict[str, homology.Hit] = {}
    for hit in homology.parse_hits(table):
        if hit.query not in sequences or len(sequences[hit.query]) != hit.qlen:
            raise ValueError(f"alignment query/length mismatch: {hit.query}")
        if not 1 <= min(hit.qstart, hit.qend) <= max(hit.qstart, hit.qend) <= hit.qlen:
            raise ValueError(f"invalid alignment query coordinates: {hit.query}")
        previous = best.get(hit.query)
        # Identity over the full query is the original R227 covariate. Coverage
        # belongs to that same selected hit, not a different convenient HSP.
        rank = (hit.identity_over_query, hit.bitscore, -hit.evalue, hit.subject)
        old_rank = (previous.identity_over_query, previous.bitscore, -previous.evalue, previous.subject) if previous else None
        if old_rank is None or rank > old_rank:
            best[hit.query] = hit
    return best


def reference_fields(hit: homology.Hit | None, *, searched: bool, empty: bool = False) -> dict[str, Any]:
    return {
        "reference_search_status": "aligned_query_coverage_available" if hit else ("no_reported_alignment" if searched else ("empty_not_searched" if empty else "not_searched")),
        "reference_identity": hit.identity_over_query if hit else None,
        "reference_coverage": (abs(hit.qend - hit.qstart) + 1) / hit.qlen if hit else None,
        "reference_coverage_definition": "query_coordinate_span_fraction_of_full_query",
        "reference_aligned_region_identity_percent": hit.pident if hit else None,
        "reference_subject": hit.subject if hit else None,
        "reference_subject_length": hit.slen if hit else None,
        "reference_target_coverage": None,
        "reference_target_coverage_status": "subject_coordinates_not_retained",
        "reference_hit_selection": "max_identity_over_query_then_bitscore_then_evalue_then_subject",
    }


def write_sidecar(rows: list[dict], output: Path, provenance: dict) -> dict:
    content = ge.jsonl_bytes(rows)
    ge.write_immutable(output / "sequence_annotations.jsonl", content)
    result = {"records": len(rows), "search_status": dict(Counter(row["reference_search_status"] for row in rows)),
              "provenance": provenance, "annotations_sha256": sha256_file(output / "sequence_annotations.jsonl")}
    ge.write_immutable(output / "annotation_manifest.json", (json.dumps(result, sort_keys=True, indent=2) + "\n").encode())
    return result


def recover_r227(attempts: Path, table: Path, query_fasta: Path, report: Path, output: Path) -> dict:
    ledger = read_attempts(attempts)
    queries = ge.read_fasta(query_fasta)
    best = best_hits(table, queries)
    old_report = json.loads(report.read_text())
    corpus_records = {arm: block["max_identity_to_corpus"] for arm, block in old_report["protein_arms"].items()}
    output_rows = []
    accounted_queries = set()
    for row in ledger:
        name = f"{row['arm']}#{row['source_key']}#{row['source_sample_index']}".replace(" ", "_")
        originally_searched = row["role"] in ("generation", "unconditioned_floor") and bool(row["sequence"])
        searched = originally_searched and name in queries
        if searched:
            if queries.get(name) != row["sequence"]:
                raise ValueError(f"retained search input differs from frozen attempt: {name}")
            accounted_queries.add(name)
        hit = best.get(name) if searched else None
        fields = reference_fields(hit, searched=searched, empty=not row["sequence"])
        fields["reference_coverage_status"] = "query_coordinates_retained" if searched else "not_applicable"
        if originally_searched and not searched:
            # The R227 work path was reused for each arm; the final arm's table
            # survives while earlier per-attempt identities survive in report.
            fields.update({"reference_identity": row["reference_identity"],
                           "reference_search_status": row["reference_search_status"],
                           "reference_coverage_status": "raw_alignment_not_retained"})
        if searched and row["reference_identity"] != fields["reference_identity"]:
            raise ValueError(f"retained hit disagrees with original identity covariate: {name}")
        output_rows.append({"id": row["id"], "sequence_sha256": row["sequence_sha256"], **fields})
    if accounted_queries != set(queries):
        raise ValueError("retained query FASTA contains unaccounted sequences")
    provenance = {
        "campaign": "EXP-R2-232", "new_inference": False,
        "inputs": {label: sha256_file(path) for label, path in {"attempts": attempts, "diamond_table": table, "diamond_query_fasta": query_fasta, "original_report": report}.items()},
        "corpus_and_commands": {arm: {key: block.get(key) for key in ("corpus", "corpus_records", "command", "corpus_note")} for arm, block in corpus_records.items()},
        "note": "Original report omitted coverage. Retained final-arm qstart/qend/qlen permit partial recovery; earlier-arm raw table was overwritten. Frozen attempts/subsets remain unchanged.",
    }
    return write_sidecar(output_rows, output, provenance)


def annotate_native(attempts: Path, output: Path, *, hmmscan: Path, pfam_hmm: Path,
                    diamond: Path, diamond_db: Path, reference_metadata: Path,
                    threads: int = 8, shards: int = 4) -> dict:
    """Run already staged executables/databases; never install or build assets."""
    rows = read_attempts(attempts)
    if len(rows) != NATIVE_POLICY["attempts"] or {row["source_sample_index"] for row in rows} != set(range(NATIVE_POLICY["attempts"])):
        raise ValueError("native annotation requires every declared generation attempt")
    if any(row["arm"] != "progen3-3b" or row["class_key"] is not None for row in rows):
        raise ValueError("native annotation requires the unconditional ProGen3-3B ledger")
    paths = {"attempts": attempts, "hmmscan": hmmscan, "pfam_hmm": pfam_hmm,
             "diamond": diamond, "diamond_db": diamond_db, "reference_metadata": reference_metadata}
    paths.update({f"pfam_index{suffix}": Path(str(pfam_hmm) + suffix) for suffix in (".h3f", ".h3i", ".h3m", ".h3p")})
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    configuration = {"inputs": {label: sha256_file(path) for label, path in paths.items()},
                     "reference_metadata": json.loads(reference_metadata.read_text()),
                     "threads": threads, "shards": shards,
                     "pfam_threshold": cg.PFAM_THRESHOLD,
                     "diamond_sensitivity": "very-sensitive", "diamond_evalue": 1e-3,
                     "diamond_max_target_seqs": 5, "diamond_masking": 0,
                     "runner_sha256": sha256_file(Path(__file__))}
    ge.write_immutable(output / "search_configuration.json", (json.dumps(configuration, sort_keys=True, indent=2) + "\n").encode())
    manifest = output / "annotation_manifest.json"
    if manifest.is_file():
        saved = json.loads(manifest.read_text())
        if saved["annotations_sha256"] != sha256_file(output / "sequence_annotations.jsonl"):
            raise ValueError("completed annotation sidecar hash mismatch")
        return saved
    sequences = {row["id"]: row["sequence"] for row in rows if row["sequence"] and row["valid_aa20"]}
    hits, profile_receipt = cg.annotate(sequences, tool=SimpleNamespace(hmmscan=hmmscan),
                                      database=SimpleNamespace(path=pfam_hmm), workspace=output / "pfam",
                                      threads=threads, shards=shards, label="native_generation")
    fasta = ci.write_fasta(output / "queries.fasta", sequences)
    table = output / "reference_hits.tsv"
    command, log_tail = homology.run_diamond_blastp(SimpleNamespace(executable=diamond), SimpleNamespace(path=diamond_db),
                                                 fasta, table, threads=threads * shards,
                                                 sensitivity="very-sensitive", evalue=1e-3, max_target_seqs=5)
    best = best_hits(table, sequences)
    annotations = []
    for row in rows:
        families = ge._families(hits, row["id"])
        annotations.append({"id": row["id"], "sequence_sha256": row["sequence_sha256"],
                            "pfam_families": families, "any_profile_hit": bool(families),
                            "target_profile_hit": None, "profile_hit_classes": None,
                            "profile_search_status": "searched" if row["id"] in sequences else "empty_or_noncanonical_not_searched",
                            **reference_fields(best.get(row["id"]), searched=row["id"] in sequences, empty=not row["sequence"])})
    return write_sidecar(annotations, output, {"campaign": "EXP-R2-233", "new_inference": True,
                                              "configuration_sha256": sha256_file(output / "search_configuration.json"),
                                              "profile_receipt": profile_receipt, "diamond_command": command,
                                              "diamond_log_tail": log_tail, "diamond_table_sha256": sha256_file(table)})
