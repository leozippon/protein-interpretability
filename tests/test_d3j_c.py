"""D3.j-C group-disjoint construction, and the guarantee that D3.j-B is untouched.

C is a new campaign. It does not salvage, rename, or reinterpret the closed
D3.j-B run. These tests cover the new splitter and the C stage contract; B's
own tests remain the authority on B.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer import alphabet_chemistry as ac  # noqa: E402
from src.transfer.arms import (  # noqa: E402
    GROUP_DISJOINT_SAMPLING_MODE,
    group_disjoint_sampling_record,
    protein_cohort_from_records,
)
from src.transfer.near_duplicates import (  # noqa: E402
    ELIGIBLE_CORPUS_EXHAUSTED,
    EligibleCorpusExhausted,
    fill_group_disjoint_slots,
)


def _load_stage():
    path = REPO_ROOT / "scripts/transfer/37_alphabet_chemistry.py"
    spec = importlib.util.spec_from_file_location("_stage_37_c", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE = _load_stage()


def _args(**overrides) -> argparse.Namespace:
    parser = STAGE.build_parser()
    args = parser.parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _seq(letter: str, length: int = 40) -> str:
    return letter * length


def _fill(records, sizes=(1, 1, 1), names=("construction", "confirm1", "confirm2")):
    return fill_group_disjoint_slots(
        records,
        slot_sizes=sizes,
        slot_names=names,
        source_positions=list(range(len(records))),
    )


def test_the_fill_is_a_function_of_the_ordered_stream_alone():
    records = [_seq("A"), _seq("C"), _seq("D"), _seq("E"), _seq("F")]
    first = _fill(records, sizes=(2, 1, 1))
    second = _fill(records, sizes=(2, 1, 1))
    assert first.slots[0].records == second.slots[0].records
    assert first.slots[1].records == second.slots[1].records
    assert first.slots[2].records == second.slots[2].records
    assert first.record() == second.record()
    assert first.slots[0].source_positions == (0, 1)


def test_an_exact_duplicate_is_rejected_from_a_later_slot_and_kept_inside_one():
    records = [_seq("A"), _seq("A"), _seq("C"), _seq("D")]
    fill = _fill(records, sizes=(2, 1, 1))
    assert fill.slots[0].records == (_seq("A"), _seq("A"))
    assert fill.slots[1].records == (_seq("C"),)
    assert fill.rejected_exact == 0
    later = _fill([_seq("A"), _seq("C"), _seq("A"), _seq("D")], sizes=(1, 1, 1))
    assert later.rejected_exact == 1
    assert later.slots[1].records == (_seq("C"),)
    assert later.slots[2].records == (_seq("D"),)
    assert later.slots[2].rejected_exact == 1


def test_a_near_duplicate_is_rejected_from_a_later_slot_and_kept_inside_one():
    base = "ACDEFGHIKLMNPQRSTVWY" * 2
    near = base[:-1] + "A"
    fill = _fill([base, near, _seq("W"), _seq("L")], sizes=(2, 1, 1))
    assert fill.slots[0].records == (base, near)
    assert fill.rejected_near == 0
    later = _fill([base, near, _seq("W"), _seq("L")], sizes=(1, 1, 1))
    assert later.rejected_near == 1
    assert later.slots[0].records == (base,)
    assert later.slots[1].records == (_seq("W"),)
    assert later.slots[1].rejected_near == 1


def test_filled_slots_are_pairwise_exact_and_near_duplicate_disjoint():
    base = "ACDEFGHIKLMNPQRSTVWY" * 2
    records = [base, base[:-1] + "A", _seq("W"), _seq("L"), _seq("Y")]
    fill = _fill(records, sizes=(1, 1, 1))
    named = {slot.name: list(slot.records) for slot in fill.slots}
    check = ac.pairwise_cohorts_independent(named)
    assert check["independent"] is True
    assert check["pairs_checked"] == 3


def test_exhaustion_fails_explicitly_and_does_not_invent_a_slot():
    with pytest.raises(EligibleCorpusExhausted, match=ELIGIBLE_CORPUS_EXHAUSTED) as caught:
        _fill([_seq("A"), _seq("A"), _seq("A")], sizes=(1, 1, 1))
    assert caught.value.reason == ELIGIBLE_CORPUS_EXHAUSTED
    assert caught.value.detail["failed_slot"] == "confirm1"
    assert caught.value.detail["filled_slots"] == ["construction"]


def test_c_resolve_selects_a_new_schema_and_leaves_b_as_the_default():
    b_args = _args(
        arm="progen2-small", cut="tercile", seed=1,
        records=8, max_tokens=64, min_symbol_occurrences=10,
        kmer_background=Path("x"), high_order_background=Path("y"),
        ceiling_orders="1,3,5", fragment_axis_order=5,
        protein_axis=ac.PROTEIN_AXIS_FRAGMENT_SUBSTITUTION_DAMAGE,
        b_stage=ac.B_STAGE_CONSTRUCT,
    )
    STAGE.resolve(b_args)
    assert STAGE.is_variant_b(b_args) is True
    assert STAGE.is_variant_c(b_args) is False
    b_payload = STAGE.base_payload(b_args, kind=ac.KIND_AXIS_CONSTRUCTION)
    assert b_payload["schema_version"] == STAGE.SCHEMA_VERSION_B
    assert b_payload["experiment"] == ac.EXPERIMENT_B
    assert "experiment" not in b_payload["settings"]

    c_args = _args(
        arm="progen2-small", cut="tercile", seed=1,
        records=8, max_tokens=64, min_symbol_occurrences=10,
        kmer_background=Path("x"), high_order_background=Path("y"),
        ceiling_orders="1,3,5", fragment_axis_order=5,
        protein_axis=ac.PROTEIN_AXIS_FRAGMENT_SUBSTITUTION_DAMAGE,
        b_stage=ac.B_STAGE_CONSTRUCT, experiment=ac.EXPERIMENT_C,
    )
    STAGE.resolve(c_args)
    assert STAGE.is_variant_c(c_args) is True
    assert STAGE.is_variant_b(c_args) is False
    c_payload = STAGE.base_payload(c_args, kind=ac.KIND_AXIS_CONSTRUCTION)
    assert c_payload["schema_version"] == STAGE.SCHEMA_VERSION_C
    assert c_payload["experiment"] == ac.EXPERIMENT_C
    assert ac.EXPERIMENT_C in STAGE.artefact_name(
        ac.KIND_AXIS_CONSTRUCTION, "progen2-small", 1, variant=f"{ac.EXPERIMENT_C}-construct"
    )


def test_c_refuses_a_skip_window_and_a_cannot_use_a_c_flag():
    c_args = _args(
        arm="progen2-small", cut="tercile", seed=1,
        records=8, max_tokens=64, min_symbol_occurrences=10,
        kmer_background=Path("x"), high_order_background=Path("y"),
        ceiling_orders="1,3,5", fragment_axis_order=5,
        protein_axis=ac.PROTEIN_AXIS_FRAGMENT_SUBSTITUTION_DAMAGE,
        b_stage=ac.B_STAGE_CONSTRUCT, experiment=ac.EXPERIMENT_C, cohort_skip=10,
    )
    with pytest.raises(ValueError, match="not a D3.j-C setting"):
        STAGE.resolve(c_args)
    a_args = _args(
        arm="progen2-small", cut="tercile", max_pairs=40, null_draws=200, seed=1,
        reachability_pairs=4, reachability_margin=0.0, random_directions=8,
        ceiling_factor=2.0, records=8, max_tokens=64, min_symbol_occurrences=10,
        kmer_background=Path("x"), high_order_background=Path("y"),
        text_control=Path("z"), ceiling_orders="1,3", experiment=ac.EXPERIMENT_C,
    )
    with pytest.raises(ValueError, match="decided nothing"):
        STAGE.resolve(a_args)


def test_b_construct_source_is_still_a_skip_window_and_voids_on_overlap():
    source = inspect.getsource(STAGE.run_b_construct)
    assert "construction_skip + construction_records * confirmation_index" in source
    assert "_protein_cohort_at_skip" in source
    assert "THREE_WAY_COHORTS_NOT_INDEPENDENT" in source
    assert "build_group_disjoint_protein_cohorts" not in source


def _c_construction_payload(**overrides):
    records = ["ACDEFGHIKLMNPQRSTVWYACDE", "WWWWWWWWWWWWWWWWWWWWWWWW"]
    sampling = group_disjoint_sampling_record(
        seed=7, requested=2, eligible=10, corpus="plain_swissprot",
        algorithm=ac.GROUP_DISJOINT_ALGORITHM,
        algorithm_version=ac.GROUP_DISJOINT_ALGORITHM_VERSION,
        containment_threshold=0.5, shingle_length=5,
        slot="confirm1", source_positions=(3, 4),
    )
    cohort = protein_cohort_from_records(
        records, 64, 246, name="swissprot", sampling=sampling,
    )
    payload = {
        "schema_version": STAGE.SCHEMA_VERSION_C,
        "kind": ac.KIND_AXIS_CONSTRUCTION,
        "experiment": ac.EXPERIMENT_C,
        "verdict": {"verdict": ac.AXIS_CONSTRUCTED},
        "tokenizer_identity": {
            "arm": "progen2-small", "architecture": "progen",
            "tokenisation": "residue", "input_format": "progen2",
            "vocab_size": 30, "tokenizer_class": "Stub", "max_tokens": 64,
        },
        "axes": {
            "labels": list("ACDEFGHIKLMNPQRSTVWY"),
            "distributional": {
                "kind": ac.PROTEIN_AXIS_FRAGMENT_SUBSTITUTION_DAMAGE,
                "order": 5,
                "symmetrization": ac.FRAGMENT_AXIS_SYMMETRIZATION,
                "corpus": {"sha256": "abc", "order": 5},
            },
            "matrices": {"distributional_fragment_damage": [[0.0]]},
        },
        "contradiction_set": {"declared_cut": "tercile", "unordered_members": {}},
        "cohort": {"digest": "construct", "records": ["YYYY" * 10]},
        "evaluation_protocol": {
            "slots": {
                "1": {
                    "index": 1, "seed": 7, "digest": cohort.digest,
                    "provenance_digest": cohort.provenance_digest,
                    "n_records": 2, "records": records,
                    "content_hashes": STAGE._content_hashes(records),
                    "source_positions": [3, 4], "sampling": sampling,
                }
            }
        },
    }
    payload.update(overrides)
    return payload


def test_c_confirmation_reproduces_frozen_records_and_refuses_a_seed_override(tmp_path):
    from src.transfer.io import write_json

    payload = _c_construction_payload()
    artefact = tmp_path / "construct.json"
    write_json(artefact, payload)
    loaded = STAGE._load_construction_artefact(
        artefact, experiment=ac.EXPERIMENT_C, schema_version=STAGE.SCHEMA_VERSION_C
    )
    args = _args(confirmation_index=1, cohort_draw_seed=7, records=2, cohort_skip=None)
    slot = STAGE._select_frozen_c_confirmation_slot(args, loaded)
    spec = type("S", (), {"evaluation_cohort_source": "swissprot"})()
    cohort = STAGE._cohort_from_c_slot(args, spec, slot)
    assert list(cohort.records) == payload["evaluation_protocol"]["slots"]["1"]["records"]
    assert cohort.digest == slot["digest"]
    assert cohort.sampling["mode"] == GROUP_DISJOINT_SAMPLING_MODE
    bad_seed = _args(confirmation_index=1, cohort_draw_seed=8, records=2)
    with pytest.raises(ValueError, match="does not match frozen"):
        STAGE._select_frozen_c_confirmation_slot(bad_seed, loaded)
    override = _args(confirmation_index=1, cohort_draw_seed=7, records=3)
    with pytest.raises(ValueError, match="cannot override frozen record count"):
        STAGE._select_frozen_c_confirmation_slot(override, loaded)


def test_c_confirmation_refuses_tampered_records_hashes_and_a_b_artefact(tmp_path):
    from src.transfer.io import write_json

    payload = _c_construction_payload()
    artefact = tmp_path / "construct.json"
    write_json(artefact, payload)
    loaded = STAGE._load_construction_artefact(
        artefact, experiment=ac.EXPERIMENT_C, schema_version=STAGE.SCHEMA_VERSION_C
    )
    args = _args(confirmation_index=1, cohort_draw_seed=7, records=2)
    spec = type("S", (), {"evaluation_cohort_source": "swissprot"})()
    tampered = dict(loaded["evaluation_protocol"]["slots"]["1"])
    tampered["records"] = ["A" * 40, "C" * 40]
    with pytest.raises(ValueError, match="content hashes"):
        STAGE._cohort_from_c_slot(args, spec, tampered)
    hashed = dict(loaded["evaluation_protocol"]["slots"]["1"])
    hashed["content_hashes"] = ["0" * 64, "1" * 64]
    with pytest.raises(ValueError, match="content hashes"):
        STAGE._cohort_from_c_slot(args, spec, hashed)
    digest_only = dict(loaded["evaluation_protocol"]["slots"]["1"])
    digest_only["digest"] = "not-the-digest"
    with pytest.raises(ValueError, match="does not match frozen slot"):
        STAGE._cohort_from_c_slot(args, spec, digest_only)
    b_path = tmp_path / "b.json"
    b_payload = dict(payload)
    b_payload["schema_version"] = STAGE.SCHEMA_VERSION_B
    b_payload["experiment"] = ac.EXPERIMENT_B
    write_json(b_path, b_payload)
    with pytest.raises(ValueError, match="schema"):
        STAGE._load_construction_artefact(
            b_path, experiment=ac.EXPERIMENT_C, schema_version=STAGE.SCHEMA_VERSION_C
        )
    voided = dict(payload)
    voided["verdict"] = {"verdict": "VOID"}
    void_path = tmp_path / "void.json"
    write_json(void_path, voided)
    with pytest.raises(ValueError, match="not AXIS_CONSTRUCTED"):
        STAGE._load_construction_artefact(
            void_path, experiment=ac.EXPERIMENT_C, schema_version=STAGE.SCHEMA_VERSION_C
        )


def test_c_loader_defaults_do_not_accept_a_c_file_as_b(tmp_path):
    from src.transfer.io import write_json

    path = tmp_path / "c.json"
    write_json(path, _c_construction_payload())
    with pytest.raises(ValueError, match="schema"):
        STAGE._load_construction_artefact(path)


def test_the_campaign_manifest_is_fourteen_cells_on_healthy_cards():
    path = REPO_ROOT / "scripts/transfer/campaign_d3jc_confirmation.tsv"
    text = path.read_text(encoding="utf-8")
    assert "H200_POD" not in text
    assert "${TRANSFER_RESULTS_RUN_DIR}" in text
    assert "eval " not in text
    rows = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        rows.append(line.split("\t"))
    assert len(rows) == 14
    slots = {int(row[0]) for row in rows}
    assert slots == {0, 1, 2, 3, 4}
    gpus = {int(row[2]) for row in rows}
    assert gpus == {0, 1, 3}
    expects = [row[6] for row in rows]
    assert all(name.endswith(".json") and "/" not in name for name in expects)
    assert sum("D3.j-C-construct" in name for name in expects) == 4
    assert sum("D3.j-C-confirm1" in name for name in expects) == 4
    assert sum("D3.j-C-confirm2" in name for name in expects) == 4
    assert any("seed202608241" in name for name in expects)
    assert any("seed202608242" in name for name in expects)
    args = " ".join(row[7] for row in rows)
    assert "--experiment D3.j-C" in args
    assert "--cohort-draw-seed 20260728" in args
    assert "--seed 20260824" in args
    assert "--seed 20260825" in args
    assert "--seed 20260826" in args
    assert "--fragment-axis-order 7" in args
    assert "d3jc_construct_progen2_base" in args
    assert "eval(" not in args


def test_c_confirm_builds_scoring_state_once_from_the_frozen_cohort(tmp_path, monkeypatch):
    from src.transfer.io import write_json

    payload = _c_construction_payload()
    frozen_records = list(payload["evaluation_protocol"]["slots"]["1"]["records"])
    payload["cohort"]["records"] = frozen_records
    artefact = tmp_path / "construct.json"
    write_json(artefact, payload)

    captured: dict = {"state_calls": []}
    real_from_slot = STAGE._cohort_from_c_slot

    def from_slot(args, spec, slot):
        cohort = real_from_slot(args, spec, slot)
        captured["frozen"] = cohort
        return cohort

    def spy_state(args, spec, *, cohort=None):
        captured["state_calls"].append(cohort)
        if cohort is captured.get("frozen"):
            return {
                "arm": object(),
                "cohort": cohort,
                "texts": list(cohort.records),
                "alphabet": (),
                "coverage": {},
                "admission": {"admitted": True, "reason": ""},
                "scoring_batch": object(),
                "cohort_record": {},
                "occurrences": {},
                "occupancy": {},
                "groups": (),
                "grouping": {},
                "runs_by_record": [],
                "identity": payload["tokenizer_identity"],
            }

        class _Skip0:
            digest = "skip-0-redraw"
            provenance_digest = "skip-0-prov"
            records = ["X" * 40]
            name = "skip0"
            sampling = {"skip": 0}

        return {
            "arm": object(),
            "cohort": _Skip0(),
            "texts": ["X" * 40],
            "alphabet": (),
            "coverage": {},
            "admission": {"admitted": True, "reason": ""},
            "scoring_batch": object(),
            "cohort_record": {},
            "occurrences": {},
            "occupancy": {},
            "groups": (),
            "grouping": {},
            "runs_by_record": [],
            "identity": payload["tokenizer_identity"],
        }

    monkeypatch.setattr(STAGE, "_cohort_from_c_slot", from_slot)
    monkeypatch.setattr(STAGE, "_b_scoring_state", spy_state)
    monkeypatch.setattr(STAGE, "read_text_control", lambda path: {"verdict": "PASS"})
    monkeypatch.setattr(
        ac, "load_ordered_counts", lambda *a, **k: {5: type("O", (), {"sha256": "abc"})()}
    )
    args = _args(
        arm="progen2-small", cut="tercile", seed=1, records=2, max_tokens=64,
        min_symbol_occurrences=10, kmer_background=Path("x"),
        high_order_background=Path("y"), text_control=Path("z"),
        ceiling_orders="1,3,5", fragment_axis_order=5,
        protein_axis=ac.PROTEIN_AXIS_FRAGMENT_SUBSTITUTION_DAMAGE,
        b_stage=ac.B_STAGE_CONFIRM, experiment=ac.EXPERIMENT_C,
        construction_artefact=artefact, confirmation_index=1,
        cohort_draw_seed=7, out=tmp_path, cohort_skip=None,
    )
    result = STAGE.run_c_confirm(args)
    assert captured["state_calls"] == [captured["frozen"]]
    assert result["independence"]["independent"] is False
    assert result["independence"]["reason"] == "EXACT_CONTENT_OVERLAP"
    assert result["verdict"]["verdict"] == "VOID"


def test_group_disjoint_sampling_is_declared_and_is_not_a_skip_window():
    record = group_disjoint_sampling_record(
        seed=20260728, requested=4, eligible=20, corpus="plain_swissprot",
        algorithm=ac.GROUP_DISJOINT_ALGORITHM,
        algorithm_version=ac.GROUP_DISJOINT_ALGORITHM_VERSION,
        containment_threshold=0.5, shingle_length=5,
        slot="construction", source_positions=(0, 2, 5, 9),
    )
    assert record["mode"] == GROUP_DISJOINT_SAMPLING_MODE
    assert "skip" not in record
    assert record["source_positions"] == [0, 2, 5, 9]
