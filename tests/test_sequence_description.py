"""What the sequence-description cohort must get right for D3.g to be gateable.

The gate at audit §8 item 4 asks for three properties of the *data*, and each of
them can be false while the artefact looks complete.

**A masked description must not name its concept.** This is the local form of the
failure that closed R3.1: ZymCTRL's one positive cell was reading its own EC tag
(L15). If ``kinase`` survives in the text, an alignment scores on a string match
and the protein side contributes nothing. So the mask is tested on the cases that
actually occur -- a different case, a plural, and a longer phrase containing a
shorter surface form -- rather than on the exact string it was given.

**A record-level split is not a split on a protein corpus.** L30 measured 42.5%
of held-out records keeping a >=95%-identity relative. The test constructs a pool
where that is true by construction and checks that the group split removes it and
the record split does not, so the contrast is a property of the two procedures
and not of one corpus.

**A propagated annotation must be a closure.** A GO set that is not closed under
``is_a``/``part_of`` makes a concept's positive rule silently miss ancestors, and
nothing downstream can see it.

The last group of tests pins the coupling the Swiss-Prot parser now carries. It
is this repository's only one, and ``ops/build_zymctrl_ec_labeled_swissprot.py``
consumes it to produce an artefact other stages depend on; the rendering is
therefore asserted against a fixture rather than trusted to survive a change in
the iterator.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.transfer import sequence_description as sd  # noqa: E402
from src.transfer.near_duplicates import (  # noqa: E402
    group_disjoint_split,
    near_duplicate_groups,
)
from src.transfer.relational import homology_disjoint_split  # noqa: E402

AA20 = "ACDEFGHIKLMNPQRSTVWY"

MINIMAL_OBO = """format-version: 1.2
data-version: releases/test

[Term]
id: GO:0000001
name: root function
namespace: molecular_function

[Term]
id: GO:0000002
name: binding
namespace: molecular_function
is_a: GO:0000001 ! root function

[Term]
id: GO:0000003
name: cellular component root
namespace: cellular_component

[Term]
id: GO:0000004
name: nucleotide binding
namespace: molecular_function
synonym: "nucleotide binder" EXACT []
synonym: "chemical thing" BROAD []
is_a: GO:0000002 ! binding

[Term]
id: GO:0000005
name: ATP binding
namespace: molecular_function
alt_id: GO:0000099
is_a: GO:0000004 ! nucleotide binding
relationship: part_of GO:0000003 ! cellular component root

[Term]
id: GO:0000006
name: obsolete thing
namespace: molecular_function
is_a: GO:0000001 ! root function
is_obsolete: true
"""


def _ontology(text: str = MINIMAL_OBO) -> sd.GoOntology:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "go.obo"
        path.write_text(text, encoding="utf-8")
        return sd.load_go_ontology(path)


def _sequences(n: int, *, length: int = 150, seed: int = 0) -> list[str]:
    generator = np.random.default_rng(seed)
    return [
        "".join(AA20[int(i)] for i in generator.integers(0, len(AA20), size=length))
        for _ in range(n)
    ]


def _stage_module():
    path = REPO / "scripts" / "transfer" / "34_sequence_description_cohort.py"
    spec = importlib.util.spec_from_file_location("_stage_34", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AnnotationClosure(unittest.TestCase):
    """``go_propagated`` is a closure, or a concept's positive rule is partial."""

    def test_the_closure_is_transitive_through_is_a_and_part_of(self):
        ontology = _ontology()
        closure = set(ontology.close(["GO:0000005"]))
        self.assertEqual(
            closure,
            {"GO:0000005", "GO:0000004", "GO:0000002", "GO:0000001", "GO:0000003"},
            "an ancestor two is_a edges up, and the part_of parent, must both be "
            "in the closure",
        )

    def test_closing_a_closure_adds_nothing(self):
        ontology = _ontology()
        once = ontology.close(["GO:0000005"])
        self.assertEqual(once, ontology.close(once))

    def test_an_alternate_id_resolves_to_its_primary_term(self):
        ontology = _ontology()
        self.assertEqual(ontology.canonical("GO:0000099"), "GO:0000005")
        self.assertIn("GO:0000005", ontology.close(["GO:0000099"]))

    def test_an_unknown_id_is_dropped_from_the_closure_and_is_reportable(self):
        ontology = _ontology()
        self.assertEqual(ontology.close(["GO:9999999"]), ())
        self.assertEqual(
            sd.unresolved_go_ids(ontology, ["GO:0000005", "GO:9999999"]),
            ("GO:9999999",),
        )

    def test_a_truncated_ontology_raises_rather_than_closing_partially(self):
        truncated = MINIMAL_OBO.replace(
            "[Term]\nid: GO:0000002\nname: binding\nnamespace: molecular_function\n"
            "is_a: GO:0000001 ! root function\n\n",
            "",
        )
        with self.assertRaises(RuntimeError) as raised:
            _ontology(truncated)
        self.assertIn("truncated", str(raised.exception))

    def test_depth_to_root_is_the_shortest_path(self):
        ontology = _ontology()
        # GO:0000005 reaches a root in one edge through part_of and in three
        # through is_a; the coarseness statistic is the shorter of the two.
        self.assertEqual(ontology.min_depth_to_root("GO:0000005"), 1)
        self.assertEqual(ontology.min_depth_to_root("GO:0000004"), 2)
        self.assertEqual(ontology.min_depth_to_root("GO:0000001"), 0)


class DescriptionMasking(unittest.TestCase):
    """The control that stops an alignment scoring on a string match."""

    def test_the_term_is_gone_and_is_reported(self):
        masked, matched, spans = sd.mask_description(
            "Kinase activity on serine residues.", ["kinase activity"]
        )
        self.assertNotIn("inase activit", masked)
        self.assertIn(sd.MASK_PLACEHOLDER, masked)
        self.assertEqual(matched, ("kinase activity",))
        self.assertEqual(spans, 1)

    def test_case_and_plural_variants_are_masked_too(self):
        masked, matched, _ = sd.mask_description(
            "Two KINASES and one kinase.", ["kinase"]
        )
        self.assertNotIn("kinase", masked.lower())
        self.assertEqual(matched, ("kinase",))

    def test_a_longer_phrase_is_masked_as_one_span_not_around_its_substring(self):
        masked, _, spans = sd.mask_description(
            "Shows protein kinase activity.", ["kinase activity", "protein kinase activity"]
        )
        self.assertEqual(spans, 1, "the shorter form must not re-match inside the mask")
        self.assertNotIn("protein", masked)

    def test_the_placeholder_is_fixed_so_length_is_not_a_side_channel(self):
        short, _, _ = sd.mask_description("It is a ligase.", ["ligase"])
        long, _, _ = sd.mask_description(
            "It is a 1-aminocyclopropane-1-carboxylate deaminase.",
            ["1-aminocyclopropane-1-carboxylate deaminase"],
        )
        self.assertEqual(
            short.replace("It is a ", "").rstrip("."),
            long.replace("It is a ", "").rstrip("."),
            "two terms of very different length must leave the same trace",
        )

    def test_a_surface_form_inside_a_word_is_not_masked(self):
        masked, matched, _ = sd.mask_description("Kinaselike domain.", ["kinase"])
        self.assertEqual(masked, "Kinaselike domain.")
        self.assertEqual(matched, ())

    def test_a_term_below_the_floor_is_never_masked(self):
        masked, matched, _ = sd.mask_description("An ND domain.", ["ND"])
        self.assertEqual(masked, "An ND domain.")
        self.assertEqual(matched, ())

    def test_a_form_unblocked_by_an_earlier_mask_is_still_removed(self):
        """The defect a real 1,200-record run refused on, kept as a test.

        ``pyrophosphoryl-`` cannot match inside ``pyrophosphoryl-undecaprenol``
        and matches immediately once ``undecaprenol`` is a placeholder. A
        single-pass mask leaves the second form in the text and the artefact looks
        complete.
        """

        text = "The pyrophosphoryl-undecaprenol acceptor."
        masked, matched, _ = sd.mask_description(
            text, ["undecaprenol", "pyrophosphoryl-"]
        )
        self.assertNotIn("pyrophosphoryl", masked)
        self.assertNotIn("undecaprenol", masked)
        self.assertEqual(set(matched), {"undecaprenol", "pyrophosphoryl-"})

    def test_masking_is_idempotent_on_its_own_output(self):
        terms = ["undecaprenol", "pyrophosphoryl-", "kinase activity"]
        once, _, _ = sd.mask_description(
            "A pyrophosphoryl-undecaprenol kinase activity.", terms
        )
        twice, matched, spans = sd.mask_description(once, terms)
        self.assertEqual(twice, once)
        self.assertEqual((matched, spans), ((), 0))

    def test_a_form_that_matches_the_placeholder_is_refused(self):
        with self.assertRaises(ValueError):
            sd.mask_description("anything", [sd.MASK_PLACEHOLDER.strip("[]")])

    def test_the_record_identity_forms_reach_the_enzyme_and_family_names(self):
        ontology = _ontology()
        forms = sd.record_identity_forms(
            go_ids=["GO:0000004"],
            go_terms=["nucleotide binding"],
            ec=["1.1.1.1"],
            pfam_entries=[("PF00001", "Kringle")],
            interpro_entries=[("IPR000001", "Kringle")],
            ontology=ontology,
            enzyme={"1.1.1.1": ("alcohol dehydrogenase",)},
            interpro_names={"IPR000001": "Kringle domain"},
        )
        for expected in (
            "nucleotide binding",
            "nucleotide binder",
            "alcohol dehydrogenase",
            "oxidoreductase",
            "Kringle",
            "Kringle domain",
            "1.1.1.1",
        ):
            self.assertIn(expected, forms)
        self.assertNotIn(
            "chemical thing", forms, "a BROAD synonym is not a form of the term"
        )


ENZYME_FIXTURE = """CC   header line
//
ID   1.1.1.1
DE   alcohol dehydrogenase.
AN   aldehyde reductase.
CA   (1) a primary alcohol + NAD(+) = an aldehyde.
//
ID   2.4.1.227
DE   undecaprenyldiphospho-muramoylpentapeptide beta-N-
DE   acetylglucosaminyltransferase.
AN   MurG transferase.
AN   UDP-N-acetylglucosamine--N-acetylmuramyl-(pentapeptide) pyrophosphoryl-
AN   undecaprenol N-acetylglucosamine transferase.
CA   something + something else.
//
ID   1.1.1.-
DE   a partial number that is not a full EC.
//
"""


class EnzymeNomenclature(unittest.TestCase):
    """A wrapped ENZYME name is one name, and a fragment is a wrong one."""

    def _load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "enzyme.dat"
            path.write_text(ENZYME_FIXTURE, encoding="utf-8")
            return sd.load_enzyme_descriptions(path)

    def test_a_name_wrapped_across_lines_is_rejoined(self):
        names = self._load()["2.4.1.227"]
        self.assertIn(
            "undecaprenyldiphospho-muramoylpentapeptide beta-N-acetylglucosaminyltransferase",
            names,
            "a mid-word wrap after a hyphen is joined without a space",
        )
        self.assertIn(
            "UDP-N-acetylglucosamine--N-acetylmuramyl-(pentapeptide) "
            "pyrophosphoryl-undecaprenol N-acetylglucosamine transferase",
            names,
        )
        self.assertNotIn(
            "UDP-N-acetylglucosamine--N-acetylmuramyl-(pentapeptide) pyrophosphoryl-",
            names,
            "the fragment is not a surface form: it masks half a name out of a "
            "description that never carried the whole one",
        )

    def test_a_complete_name_is_read_without_its_full_stop(self):
        self.assertEqual(
            self._load()["1.1.1.1"], ("alcohol dehydrogenase", "aldehyde reductase")
        )

    def test_a_partial_ec_number_carries_no_entry(self):
        self.assertNotIn("1.1.1.-", self._load())

    def test_a_file_that_is_not_an_enzyme_release_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "enzyme.dat"
            path.write_text("CC   nothing here\n//\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                sd.load_enzyme_descriptions(path)


class VerbatimSequenceLeak(unittest.TestCase):
    def test_a_quoted_run_of_residues_is_detected(self):
        sequence = _sequences(1, length=200, seed=3)[0]
        report = sd.sequence_in_description(
            sequence, f"Contains the motif {sequence[40:60]} near the active site."
        )
        self.assertTrue(report["shared_runs"])

    def test_ordinary_prose_and_a_short_motif_do_not_fire(self):
        sequence = _sequences(1, length=200, seed=4)[0]
        report = sd.sequence_in_description(
            sequence, f"A serine protease with a {sequence[10:18]} motif."
        )
        self.assertFalse(report["contains_full_sequence"])
        self.assertEqual(report["shared_runs"], ())


class ConceptLabelling(unittest.TestCase):
    """Both sides are defined, or the concept is undefined for that record."""

    def setUp(self) -> None:
        self.ontology = _ontology()
        self.go_concept = sd.ConceptSpec("c", "go", "GO:0000004", "nucleotide binding")
        self.ec_concept = sd.ConceptSpec("e", "ec", "1", "oxidoreductase")

    def test_a_descendant_annotation_makes_a_positive_through_the_closure(self):
        closure = self.ontology.close(["GO:0000005"])
        self.assertEqual(
            sd.concept_label(
                self.go_concept, go_propagated=closure, ec=[], ontology=self.ontology
            ),
            1,
        )

    def test_an_annotation_in_the_same_aspect_makes_a_negative(self):
        closure = self.ontology.close(["GO:0000002"])
        self.assertEqual(
            sd.concept_label(
                self.go_concept, go_propagated=closure, ec=[], ontology=self.ontology
            ),
            0,
        )

    def test_no_annotation_in_the_aspect_is_undefined_and_not_a_negative(self):
        closure = self.ontology.close(["GO:0000003"])
        self.assertIsNone(
            sd.concept_label(
                self.go_concept, go_propagated=closure, ec=[], ontology=self.ontology
            ),
            "absence of curation is not evidence of absence; an open-world "
            "negative is the defect this rule exists to prevent",
        )

    def test_an_ec_concept_is_undefined_for_a_non_enzyme(self):
        self.assertIsNone(
            sd.concept_label(self.ec_concept, go_propagated=(), ec=[], ontology=self.ontology)
        )
        self.assertEqual(
            sd.concept_label(
                self.ec_concept, go_propagated=(), ec=["1.14.13.168"], ontology=self.ontology
            ),
            1,
        )
        self.assertEqual(
            sd.concept_label(
                self.ec_concept, go_propagated=(), ec=["3.5.99.7"], ontology=self.ontology
            ),
            0,
        )

    def test_an_ec_concept_outside_the_declared_classes_is_refused(self):
        with self.assertRaises(ValueError):
            sd.ConceptSpec("bad", "ec", "1.1", "not a class")


class ConceptAdmission(unittest.TestCase):
    """EXP-R2-213 C34-5/C34-6: counted in groups, per split, curve before floor."""

    def _cohort(self, per_split: dict[str, tuple[int, int]], *, groups_per_record=True):
        """One record per group by default; ``groups_per_record=False`` collapses
        every positive of a split into a single near-duplicate group."""

        splits: list[str] = []
        column: list[int | None] = []
        groups: list[int] = []
        counter = 0
        for name, (positive, negative) in per_split.items():
            for index in range(positive + negative):
                splits.append(name)
                column.append(1 if index < positive else 0)
                if groups_per_record or index >= positive:
                    counter += 1
                    groups.append(counter)
                else:
                    groups.append(-hash(name) % 1000)
        labels = {spec.concept_id: list(column) for spec in sd.CONCEPTS}
        return labels, splits, groups

    def test_a_concept_thin_in_one_deciding_split_is_rejected_though_the_pool_is_rich(self):
        labels, splits, groups = self._cohort(
            {"fit": (200, 200), "eval": (200, 200), "family_holdout": (1, 400)}
        )
        admission = sd.admit_concepts(
            labels, splits, groups, ontology=_ontology(), min_groups_per_cell=8
        )
        self.assertEqual(admission.admitted, ())
        reasons = {entry["concept_id"]: entry["reason"] for entry in admission.rejected}
        for spec in sd.CONCEPTS:
            if spec.kind == "ec":
                self.assertIn(
                    "family_holdout",
                    reasons[spec.concept_id],
                    "the rejection has to name the split that failed, or an "
                    "operator cannot tell a thin concept from a thin holdout",
                )
            else:
                # The declaration check runs before the counts: a concept the
                # ontology release does not carry cannot be admitted on counts.
                self.assertIn("not a term of ontology", reasons[spec.concept_id])

    def test_a_thin_fit_split_does_not_reject_a_concept(self):
        labels, splits, groups = self._cohort(
            {"fit": (1, 1), "eval": (20, 20), "family_holdout": (20, 20)}
        )
        admission = sd.admit_concepts(
            labels, splits, groups, ontology=_ontology(), min_groups_per_cell=8
        )
        self.assertEqual(
            [entry for entry in admission.admitted if entry.startswith("ec_")],
            [spec.concept_id for spec in sd.CONCEPTS if spec.kind == "ec"],
            "fit is not a deciding split: a map is fitted there and nothing is "
            "reported there",
        )

    def test_the_unit_is_the_near_duplicate_group_and_not_the_record(self):
        # Forty positives inside ONE group is one bootstrap unit, and admitting it
        # on the record count is the failure this rule exists to prevent.
        labels, splits, groups = self._cohort(
            {"fit": (40, 40), "eval": (40, 40), "family_holdout": (40, 40)},
            groups_per_record=False,
        )
        admission = sd.admit_concepts(
            labels, splits, groups, ontology=_ontology(), min_groups_per_cell=8
        )
        self.assertEqual(admission.admitted, ())
        for spec in sd.CONCEPTS:
            if spec.kind == "ec":
                cell = admission.counts[spec.concept_id]["eval"]
                self.assertEqual(cell["positive"], 40)
                self.assertEqual(cell["bearing_groups"], 1)

    def test_the_floor_curve_is_reported_at_the_declared_floors_and_is_monotone(self):
        labels, splits, groups = self._cohort(
            {"fit": (40, 40), "eval": (20, 20), "family_holdout": (10, 10)}
        )
        admission = sd.admit_concepts(
            labels, splits, groups, ontology=_ontology(), min_groups_per_cell=8
        )
        self.assertEqual(
            sorted(int(key) for key in admission.floor_curve),
            sorted(sd.ADMISSION_FLOOR_CURVE),
        )
        counts = [
            admission.floor_curve[str(floor)]["both"] for floor in sd.ADMISSION_FLOOR_CURVE
        ]
        self.assertEqual(counts, sorted(counts, reverse=True))
        self.assertGreater(counts[0], counts[-1], "the curve has to be informative")
        # 10 groups per side in family_holdout: survives at 4 and 8, not at 16.
        self.assertEqual(admission.floor_curve["16"]["family_holdout"], 0)

    def test_misaligned_labels_splits_or_groups_are_refused(self):
        labels, splits, groups = self._cohort(
            {"fit": (5, 5), "eval": (5, 5), "family_holdout": (5, 5)}
        )
        for bad in ((splits[:-1], groups), (splits, groups[:-1])):
            with self.assertRaises(ValueError):
                sd.admit_concepts(
                    labels, bad[0], bad[1], ontology=_ontology(), min_groups_per_cell=1
                )


class RecordSchema(unittest.TestCase):
    """Two other stages read ``records.jsonl``; the schema is a contract."""

    def _record(self, **overrides):
        payload = dict(
            accession="P00001",
            sequence="ACDEFGHIKL",
            length=10,
            name="Test protein",
            function_text="Does a thing that is described at some length.",
            description_raw="Test protein. Does a thing.",
            description_masked="[MASK]. Does a thing.",
            masked_terms=("Test protein",),
            ec=(),
            go=("GO:0000005",),
            go_propagated=("GO:0000001", "GO:0000005"),
            pfam=("PF00001",),
            cath=(),
            dup_group=0,
            family_group="pfam:g0",
            split="fit",
        )
        payload.update(overrides)
        return sd.SequenceDescriptionRecord(**payload)

    def test_a_round_trip_through_disk_preserves_every_field(self):
        record = self._record()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            sd.write_records(path, [record])
            self.assertEqual(sd.read_records(path), [record])
            first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(set(first), set(sd.RECORD_FIELDS))

    def test_a_propagated_set_that_is_not_a_closure_of_the_direct_set_is_refused(self):
        with self.assertRaises(ValueError):
            self._record(go=("GO:0000005",), go_propagated=("GO:0000001",))

    def test_a_declared_length_that_is_not_the_sequence_length_is_refused(self):
        with self.assertRaises(ValueError):
            self._record(length=11)

    def test_an_unknown_split_is_refused(self):
        with self.assertRaises(ValueError):
            self._record(split="test")

    def test_a_reader_refuses_a_line_outside_the_frozen_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            payload = self._record().to_dict()
            payload["extra_field"] = 1
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                sd.read_records(path)

    def test_an_empty_cohort_is_not_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                sd.write_records(Path(tmp) / "records.jsonl", [])


class HomologyLeakageContrast(unittest.TestCase):
    """The record split leaks and the group split does not, on one alignment.

    Constructed rather than sampled: sixteen unrelated records plus eight
    near-duplicate pairs. The "alignment" is synthesised at the identities the
    construction puts there, so the test is about the two split procedures and
    not about an aligner.
    """

    #: 120 unrelated records and 40 near-duplicate pairs. The pool is this size
    #: because whole groups are indivisible: at 32 records a group of two is 6%
    #: of the pool and no split lands inside ``group_disjoint_split``'s 2%
    #: tolerance, so the refusal fires before the leakage can be read. The
    #: granularity has to be finer than the tolerance for the comparison to exist.
    N_UNRELATED = 120
    N_PAIRS = 40

    def setUp(self) -> None:
        base = _sequences(self.N_UNRELATED, seed=11)
        pairs = _sequences(self.N_PAIRS, seed=12)
        # Each pair member is the same sequence with two substitutions: a
        # near-duplicate by any reading, and the unit L30 says must not be split.
        self.sequences = base + [
            sequence if not tail else sequence[:5] + "WW" + sequence[7:]
            for sequence in pairs
            for tail in (0, 1)
        ]
        self.stage = _stage_module()

    def _hits(self):
        hits = []
        for index in range(self.N_UNRELATED, len(self.sequences), 2):
            for left, right in ((index, index + 1), (index + 1, index)):
                length = len(self.sequences[left])
                hits.append((left, right, length - 2, length, length))
        return hits

    def test_the_group_split_removes_what_the_record_split_leaves(self):
        groups, _ = near_duplicate_groups(self.sequences, unit="residues")
        total = len(self.sequences)
        n_train = int(round(0.6 * total))
        # Read over several seeds: the record-level split leaks on most of them
        # and the group split must leak on none, which is the asymmetry the
        # design turns on rather than one lucky draw.
        record_leaks = []
        for seed in range(8):
            record_mask = homology_disjoint_split(
                np.arange(total), train_fraction=n_train / total, seed=seed, min_side=1
            )
            group_mask, _ = group_disjoint_split(groups, n_train=n_train, seed=seed)
            record_leak = self.stage.leakage_reading(
                self._hits(),
                train=record_mask,
                held_out=~record_mask,
                label="record_level",
                note="test",
            )
            group_leak = self.stage.leakage_reading(
                self._hits(),
                train=group_mask,
                held_out=~group_mask,
                label="near_duplicate_group",
                note="test",
            )
            record_leaks.append(record_leak["at_or_above"]["95"]["n"])
            self.assertEqual(
                group_leak["at_or_above"]["95"]["n"],
                0,
                f"seed {seed}: a near-duplicate group was divided",
            )
        self.assertGreater(
            sum(record_leaks),
            0,
            "the record-level split has to leak here, or the contrast is vacuous",
        )

    def test_a_pool_that_cannot_be_split_by_group_refuses(self):
        one_family = [self.sequences[self.N_UNRELATED]] * 20 + _sequences(2, seed=13)
        groups, _ = near_duplicate_groups(one_family, unit="residues")
        with self.assertRaises(RuntimeError) as raised:
            group_disjoint_split(groups, n_train=11, seed=0)
        self.assertIn("do not widen the tolerance", str(raised.exception))

    def test_a_reading_over_overlapping_sides_is_refused(self):
        mask = np.ones(len(self.sequences), dtype=bool)
        with self.assertRaises(ValueError):
            self.stage.leakage_reading(
                self._hits(), train=mask, held_out=mask, label="x", note="y"
            )


class StageContract(unittest.TestCase):
    def setUp(self) -> None:
        self.stage = _stage_module()

    def test_every_pre_registered_flag_is_named_when_it_is_missing(self):
        args = self.stage.build_parser().parse_args([])
        with self.assertRaises(ValueError) as raised:
            self.stage.resolve(args)
        message = str(raised.exception)
        for flag in self.stage.REQUIRED_FLAGS:
            self.assertIn(f"--{flag.replace('_', '-')}", message)

    def test_no_pre_registered_flag_carries_a_default(self):
        parser = self.stage.build_parser()
        defaults = {action.dest: action.default for action in parser._actions}
        for flag in self.stage.REQUIRED_FLAGS:
            self.assertIsNone(
                defaults[flag],
                f"--{flag.replace('_', '-')} has a default, so the decision it "
                "carries can change without the request changing",
            )

    def test_the_uniform_external_baseline_flags_are_accepted(self):
        args = self.stage.build_parser().parse_args(
            ["--device", "cuda:3", "--out", "/tmp/x"]
        )
        self.assertEqual(args.device, "cuda:3")
        self.assertEqual(str(args.out), "/tmp/x")

    #: A complete, pre-registration-conforming request, so that a refusal in the
    #: tests below is the refusal under test and not a missing flag.
    BASE = [
        "--family-source", "pfam", "--go-evidence", "all", "--pool-size", "10",
        "--holdout-fraction", "0.2", "--fit-fraction", "0.7", "--seed", "1",
        "--work", "/tmp/w", "--length-band", "10", "500",
        "--min-family-groups-per-side", "8", "--min-concept-groups-per-cell", "8",
        "--stop-min-concepts-eval", "8", "--stop-min-concepts-family-holdout", "4",
        "--straddling-refusal-boundary", "90",
    ]

    def test_the_complete_request_resolves(self):
        args = self.stage.build_parser().parse_args(self.BASE)
        self.stage.resolve(args)  # must not raise

    def test_an_inverted_band_and_an_out_of_range_fraction_are_refused(self):
        for override in (
            ["--length-band", "500", "10"],
            ["--holdout-fraction", "1.5"],
            ["--fit-fraction", "0.0"],
            ["--pool-size", "0"],
        ):
            args = self.stage.build_parser().parse_args(self.BASE + override)
            with self.assertRaises(ValueError):
                self.stage.resolve(args)

    def test_a_request_that_weakens_a_frozen_criterion_is_refused_by_name(self):
        for override, expected in (
            (["--min-concept-groups-per-cell", "4"], "min-concept-groups-per-cell"),
            (["--min-family-groups-per-side", "2"], "min-family-groups-per-side"),
            (["--stop-min-concepts-eval", "1"], "stop-min-concepts-eval"),
            (["--stop-min-concepts-family-holdout", "1"], "stop-min-concepts-family-holdout"),
            (["--straddling-refusal-boundary", "95"], "straddling-refusal-boundary"),
        ):
            args = self.stage.build_parser().parse_args(self.BASE + override)
            with self.assertRaises(ValueError) as raised:
                self.stage.resolve(args)
            self.assertIn("EXP-R2-213", str(raised.exception))
            self.assertIn(expected, str(raised.exception))

    def test_tightening_a_frozen_criterion_is_allowed(self):
        args = self.stage.build_parser().parse_args(
            self.BASE
            + ["--min-concept-groups-per-cell", "16", "--straddling-refusal-boundary", "70"]
        )
        self.stage.resolve(args)  # tightening cannot manufacture a pass

    def test_a_boundary_the_curve_does_not_report_is_refused(self):
        args = self.stage.build_parser().parse_args(
            self.BASE + ["--straddling-refusal-boundary", "85"]
        )
        with self.assertRaises(ValueError) as raised:
            self.stage.resolve(args)
        self.assertIn("declared boundaries", str(raised.exception))

    def test_the_five_pre_registered_identity_boundaries_are_the_ones_read(self):
        self.assertEqual(
            sorted(self.stage.IDENTITY_BOUNDARIES, reverse=True),
            [95.0, 90.0, 80.0, 70.0, 50.0],
        )

    def test_the_reservoir_draw_is_seeded_and_is_not_a_prefix(self):
        stream = list(range(1000))
        first, seen = self.stage.reservoir_sample(iter(stream), size=20, seed=7)
        again, _ = self.stage.reservoir_sample(iter(stream), size=20, seed=7)
        other, _ = self.stage.reservoir_sample(iter(stream), size=20, seed=8)
        self.assertEqual(seen, 1000)
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)
        self.assertNotEqual(
            sorted(first),
            list(range(20)),
            "a head-of-file prefix is what Appendix B rule 1 forbids",
        )


class Stop34(unittest.TestCase):
    """The three STOP-34 conditions, each on the quantity it is stated over."""

    def setUp(self) -> None:
        self.stage = _stage_module()
        self.args = self.stage.build_parser().parse_args(StageContract.BASE)
        self.stage.resolve(self.args)

    def _admission(self, per_split_groups: int):
        splits: list[str] = []
        column: list[int | None] = []
        groups: list[int] = []
        counter = 0
        for name in ("fit", "eval", "family_holdout"):
            for label in (1, 0):
                for _ in range(per_split_groups):
                    counter += 1
                    splits.append(name)
                    column.append(label)
                    groups.append(counter)
        labels = {spec.concept_id: list(column) for spec in sd.CONCEPTS}
        return sd.admit_concepts(
            labels,
            splits,
            groups,
            ontology=_ontology(),
            min_groups_per_cell=self.args.min_concept_groups_per_cell,
        )

    def _reading(self, straddling: dict[str, int]):
        return [
            {
                "split": "near_duplicate_group",
                "straddling_pairs": {"95": 0, "90": 0, "80": 0, "70": 0, "50": 0}
                | straddling,
            }
        ]

    def test_a_clean_cohort_with_enough_concepts_does_not_stop(self):
        # Only the six EC concepts can be admissible against the minimal test
        # ontology, which is fewer than the eval floor of eight -- so this asserts
        # the two conditions separately rather than pretending one cohort clears
        # both: here the straddling condition is the one under test.
        stops = self.stage.stop_34(
            self.args, admission=self._admission(20), readings=self._reading({})
        )
        self.assertEqual(
            [entry["condition"] for entry in stops],
            ["too_few_admissible_concepts_in_eval"],
        )

    def test_a_straddling_pair_at_the_refusal_boundary_stops(self):
        stops = self.stage.stop_34(
            self.args, admission=self._admission(20), readings=self._reading({"90": 1})
        )
        conditions = [entry["condition"] for entry in stops]
        self.assertIn("group_split_leaves_straddling_pairs", conditions)
        entry = stops[conditions.index("group_split_leaves_straddling_pairs")]
        self.assertEqual(entry["statement_about"], "the evaluation interface (L30)")

    def test_a_straddling_pair_below_the_boundary_does_not_stop(self):
        stops = self.stage.stop_34(
            self.args,
            admission=self._admission(20),
            readings=self._reading({"70": 5, "50": 9}),
        )
        self.assertNotIn(
            "group_split_leaves_straddling_pairs",
            [entry["condition"] for entry in stops],
            "the residual below the boundary is declared, not gated (L30)",
        )

    def test_a_thin_concept_panel_stops_on_both_reporting_splits(self):
        stops = self.stage.stop_34(
            self.args, admission=self._admission(2), readings=self._reading({})
        )
        self.assertEqual(
            [entry["condition"] for entry in stops],
            [
                "too_few_admissible_concepts_in_eval",
                "too_few_admissible_concepts_in_family_holdout",
            ],
        )


class StraddlingPairCounting(unittest.TestCase):
    def setUp(self) -> None:
        self.stage = _stage_module()

    def test_a_cross_boundary_pair_is_counted_once_at_every_boundary_it_clears(self):
        train = np.array([True, False])
        # One pair at 96% identity over the shorter sequence, reported in both
        # directions by the aligner as DIAMOND does.
        pairs = [(1, 0, 96, 100, 100), (0, 1, 96, 100, 100)]
        reading = self.stage.leakage_reading(
            pairs, train=train, held_out=~train, label="x", note="y"
        )
        self.assertEqual(reading["straddling_pairs"], {"95": 1, "90": 1, "80": 1, "70": 1, "50": 1})
        self.assertEqual(reading["at_or_above"]["95"]["n"], 1)
        self.assertEqual(reading["n_held_out_with_a_byte_identical_relative"], 0)

    def test_a_within_side_pair_is_not_leakage(self):
        train = np.array([True, True, False])
        pairs = [(0, 1, 100, 100, 100), (1, 0, 100, 100, 100)]
        reading = self.stage.leakage_reading(
            pairs, train=train, held_out=~train, label="x", note="y"
        )
        self.assertEqual(reading["straddling_pairs"]["50"], 0)


SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<uniprot xmlns="{namespace}">
<entry dataset="Swiss-Prot">
    <accession>P00001</accession>
    <accession>P90001</accession>
    <name>TEST1_TESTA</name>
    <protein>
        <recommendedName>
            <fullName>Test enzyme one</fullName>
            <ecNumber>1.1.1.1</ecNumber>
        </recommendedName>
        <alternativeName>
            <fullName>Not the recommended name</fullName>
        </alternativeName>
    </protein>
    <comment type="function">
        <text evidence="1">Catalyses a reaction.</text>
    </comment>
    <comment type="similarity">
        <text>Belongs to a family.</text>
    </comment>
    <dbReference type="GO" id="GO:0000005">
        <property type="term" value="F:ATP binding"/>
        <property type="evidence" value="ECO:0007669"/>
    </dbReference>
    <dbReference type="Pfam" id="PF00001">
        <property type="entry name" value="Kringle"/>
        <property type="match status" value="1"/>
    </dbReference>
    <sequence length="10" mass="1">ACDEFGHIKL</sequence>
</entry>
<entry dataset="Swiss-Prot">
    <accession>P00002</accession>
    <name>TEST2_TESTA</name>
    <protein>
        <recommendedName>
            <fullName>Test enzyme two</fullName>
        </recommendedName>
    </protein>
    <dbReference type="EC" id="2.7.11.1"/>
    <dbReference type="EC" id="1.1.1.-"/>
    <sequence length="8" mass="1">MKVLAACD</sequence>
</entry>
<entry dataset="Swiss-Prot">
    <accession>P00003</accession>
    <name>TEST3_TESTA</name>
    <protein>
        <recommendedName>
            <fullName>Not an enzyme</fullName>
        </recommendedName>
    </protein>
    <sequence length="6" mass="1">MKVLAA</sequence>
</entry>
</uniprot>
"""


class SwissProtIteration(unittest.TestCase):
    """The repository's one Swiss-Prot reader, and the defences it inherited."""

    def _entries(self, namespace: str = "https://uniprot.org/uniprot"):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.xml"
            path.write_text(SAMPLE_XML.format(namespace=namespace), encoding="utf-8")
            return list(sd.iter_swissprot_entries(path))

    def test_the_fields_a_description_cohort_reads_are_extracted(self):
        first, second, third = self._entries()
        self.assertEqual(first.accession, "P00001")
        self.assertEqual(first.protein_name, "Test enzyme one")
        self.assertEqual(first.function_texts, ("Catalyses a reaction.",))
        self.assertEqual(first.sequence, "ACDEFGHIKL")
        self.assertEqual(first.go[0], sd.GoAnnotation("GO:0000005", "ATP binding", "F", "ECO:0007669"))
        self.assertEqual(first.pfam, (("PF00001", "Kringle"),))
        self.assertEqual(
            second.function_texts, (), "an entry with no function comment is yielded"
        )
        self.assertEqual(third.ec, (), "an entry with no EC is yielded, not skipped")

    def test_both_ec_sources_are_read_and_partial_numbers_are_not(self):
        first, second, _ = self._entries()
        self.assertEqual(first.ec, ("1.1.1.1",), "the <ecNumber> element is a source")
        self.assertEqual(
            second.ec,
            ("2.7.11.1",),
            "the dbReference is the other source, and 1.1.1.- is not a full EC",
        )

    def test_the_namespace_is_read_from_the_document_and_not_hard_coded(self):
        # UniProt has shipped both spellings; against a literal, every entry is
        # silently skipped and the consumer writes an empty file and exits 0.
        for namespace in ("https://uniprot.org/uniprot", "http://uniprot.org/uniprot"):
            self.assertEqual(len(self._entries(namespace)), 3, namespace)


class OpsRenderingIsPinnedToTheSharedIterator(unittest.TestCase):
    """``ec_labeled_swissprot.fasta`` is an artefact other stages consume.

    The builder no longer owns its parser, so the coupling is pinned here: the
    rendering, the one-record-per-EC fan-out, the length cut, and the refusal to
    write an empty corpus are asserted against a fixture rather than left to be
    re-derived from whatever the iterator returns next.
    """

    def _build(self, *, max_seq_len: int = 1022, xml: str | None = None) -> str:
        script = REPO / "ops" / "build_zymctrl_ec_labeled_swissprot.py"
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "sample.xml"
            source.write_text(
                xml or SAMPLE_XML.format(namespace="https://uniprot.org/uniprot"),
                encoding="utf-8",
            )
            destination = Path(tmp) / "out.fasta"
            completed = subprocess.run(
                [
                    sys.executable, str(script),
                    "--xml", str(source),
                    "--out", str(destination),
                    "--max-seq-len", str(max_seq_len),
                ],
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                return f"FAILED\n{completed.stderr}"
            return destination.read_text(encoding="utf-8")

    def test_the_rendering_is_exactly_zymctrls(self):
        self.assertEqual(
            self._build(),
            ">P00001|1.1.1.1\n1.1.1.1<sep><start>ACDEFGHIKL<end>\n"
            ">P00002|2.7.11.1\n2.7.11.1<sep><start>MKVLAACD<end>\n",
        )

    def test_the_length_cut_is_still_this_scripts_own(self):
        self.assertEqual(
            self._build(max_seq_len=9),
            ">P00002|2.7.11.1\n2.7.11.1<sep><start>MKVLAACD<end>\n",
        )

    def test_a_corpus_with_no_enzyme_writes_nothing_and_fails(self):
        empty = SAMPLE_XML.format(namespace="https://uniprot.org/uniprot")
        empty = empty.replace("<ecNumber>1.1.1.1</ecNumber>", "").replace(
            '<dbReference type="EC" id="2.7.11.1"/>', ""
        )
        output = self._build(xml=empty)
        self.assertTrue(output.startswith("FAILED"))
        self.assertIn("An empty corpus is not a result", output)


if __name__ == "__main__":
    unittest.main()
