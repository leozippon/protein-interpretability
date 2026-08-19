"""Every campaign stage must draw its corpus under a declared seed.

Appendix B rule 1 of the transfer audit exists because iterating a biological
corpus in file order manufactured an effect three times, most recently worth
+1.01 nats. The library honoured it -- ``arms.protein_cohort`` and
``arms.text_cohort`` have taken a ``seed`` since EXP-R2-062 -- but until
EXP-R2-068 not one campaign stage passed it. The corpus pool was a head-of-file
prefix and only the *subsample of that prefix* was seeded, which is why the
EXP-R2-060 qualification records the measured 0.16-0.60 nat cohort-block
sensitivity as bounding *within-pool* rather than corpus-wide selection
uncertainty.

These tests are static rather than behavioural on purpose: the defect was never
a wrong number in a run, it was an argument nobody passed, and a test that reads
the call sites is the one that catches the next stage added without it.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE_DIR = REPO_ROOT / "scripts" / "transfer"

#: Every constructor that decides which records exist, which is the decision rule
#: 1 governs.
#:
#: The repeat constructors were missing from this tuple when it was first written,
#: and the omission cost more than the plain ones would have. A repeat cohort is
#: not a census unless ``n`` reaches the matching population: the induction stage
#: drew **32 of 817** matching proteins and **32 of 968** matching documents under
#: the approximate criterion -- a four per cent head-of-file prefix -- on the
#: cohort that carries the programme's headline part-1 result. A test that covered
#: only ``protein_cohort`` and ``text_cohort`` passed while that was true.
CORPUS_CONSTRUCTORS = (
    "protein_cohort",
    "text_cohort",
    "protein_repeat_cohort",
    "text_repeat_cohort",
)

#: Stages that legitimately do not call the corpus constructors at all, with the
#: reason, so that a stage which stops calling them is a visible change rather
#: than a silently passing test.
NON_DRAWING_STAGES: dict[str, str] = {
    "05_relational_channel.py": (
        "draws AlphaFold structures under its own seeded permutation of the "
        "catalogue, not sequences from a FASTA corpus"
    ),
    "09_probe_and_erasure.py": (
        "draws through src.transfer.probes: record_order is seeded by default and "
        "probes.text_units now passes its seed to the corpus draw as well. The "
        "second half of that was untrue when this exemption was written, and the "
        "exemption is what hid it -- probes.text_units took a seed and applied it "
        "only to positions, over a 150-of-396,000 file-order prefix"
    ),
    "12_induction_robustness.py": "reads artefacts from disk; loads no corpus",
    "24_component_swap.py": (
        "measures its chimera through 21_joint_mode_qualification.py's own "
        "protein_mode and text_mode, which draw both windows through "
        "arms.protein_cohort and arms.text_cohort under that stage's mode_cohorts. "
        "It defines --cohort-draw-seed and passes it in the namespace those "
        "functions read, so rule 1 is answered at the one call site rather than at "
        "a second copy of it -- a chimera's number is only readable beside the "
        "qualification figures it is compared with, which requires the identical "
        "draw and not merely a matching one"
    ),
    "panel_contract.py": "a declaration, not a measurement",
    "paa_failure_audit.py": (
        "stratifies retained census and causal artefacts that already exist; it "
        "loads no model and constructs no cohort, and the draw each stratum "
        "belongs to is whichever one the run it reads recorded"
    ),
    "read_paa_panel.py": (
        "reads paa_gate_report.json files already on disk and recomputes their "
        "statistic through prediction_addressed.census_causal_agreement; it loads "
        "no model and constructs no cohort, so the draw it reports is whichever "
        "one the run it is reading recorded"
    ),
    "17_train_transcoder.py": (
        "trains on a stream of millions of records -- UniRef50 for ProGen3, and "
        "for a panel arm the corpus its own evaluation cohort is drawn from -- "
        "which the cohort constructors cannot serve: they count the whole corpus "
        "and then select, which is right for a frozen 128-sequence cohort and not "
        "affordable for a training run. It draws through its own seeded "
        "block-shuffled stream instead, and answers rule 1 the same way -- a "
        "biological corpus is ordered by cluster and a web corpus by shard, so a "
        "prefix is a region rather than a sample, and the block size the shuffle "
        "operates over is recorded in the artefact rather than left implicit"
    ),
    "25_model_diffing_baselines.py": (
        "draws through 17_train_transcoder.py's seeded block-shuffled stream, "
        "imported rather than reimplemented, because it fits its maps on exactly "
        "the tensors and the population a transcoder is fitted to. It also needs "
        "something a cohort constructor cannot give it: ONE pool of train+eval "
        "records split under a seeded permutation, so the fitting and the "
        "reporting halves are samples of one population rather than two windows of "
        "a cluster-ordered corpus -- the gap that would otherwise read as a "
        "failure of the map. Rule 1 is answered with --seed and with --skip, which "
        "moves the pool through the corpus in file order and is the only way to "
        "produce the skip-offset sensitivity; the block size, the pool size and "
        "the skip all reach the artefact"
    ),
    "20_retrieval_bound.py": (
        "its units are the 187 distinct wild types of the ProteinGym substitution "
        "benchmark and the DMS variants of each, not corpus sequences, so it draws "
        "through src.transfer.fitness.load_assay under a seed and never as a "
        "prefix. It reads the corpus itself in full -- background frequencies, "
        "record count and byte-identical membership are census quantities over "
        "every record, so there is nothing to sample and no order to be biased by"
    ),
    "34_sequence_description_cohort.py": (
        "its records are Swiss-Prot XML *entries*, not FASTA sequences: it needs "
        "the accession, the curated name, the function comments and the GO "
        "cross-references, none of which a Cohort carries, so it reads the release "
        "through src.transfer.sequence_description.iter_swissprot_entries. Rule 1 "
        "is answered by a seeded reservoir over every eligible entry of the whole "
        "release -- a uniform draw in one pass, which is stronger than a permuted "
        "window and is what the 933 MB stream affords -- and --max-entries-scanned "
        "records in the artefact that a capped scan is a corpus prefix and not a "
        "sample"
    ),
    "35_concept_alignment.py": (
        "consumes the frozen cohort 34_sequence_description_cohort.py wrote and "
        "constructs nothing: its units are that stage's records, in that stage's "
        "splits, and a cohort of its own would be a second definition of what "
        "'held out' means on a protein corpus, which is L30's defect. It reads the "
        "file through src.transfer.concept_alignment.load_cohort, which re-checks "
        "the group disjointness the manifest certifies rather than trusting it. "
        "Rule 1's hazard is absent for the records and present for two draws this "
        "stage does make, and both are seeded by --seed and reach the artefact: the "
        "common retrieval gallery, drawn once per split from the near-duplicate "
        "grouping and reused by every arm and null draw so the paired comparisons "
        "are paired, and the pairing permutations of the shuffled-pair, "
        "shuffled-fit and rank-matched nulls. A second cohort draw is a second run "
        "of stage 34 followed by a second run of this one"
    ),
    "36_concept_injection.py": (
        "consumes the same frozen cohort as 35_concept_alignment.py and constructs "
        "nothing, for the same reason: its units are stage 34's records in stage "
        "34's splits, and a cohort of its own would be a second definition of what "
        "'held out' means on a protein corpus, which is L30's defect. It reads the "
        "file through src.transfer.concept_alignment.load_cohort and fits every "
        "direction on the 'fit' split while measuring on --eval-split, so the two "
        "are that stage's declaration and not this one's. Rule 1's hazard is absent "
        "for the records and present for three draws this stage does make, and all "
        "three are seeded and reach the artefact: A36-3(b)'s norm-matched random "
        "directions and A36-6's generation sampling from --seed, and A36-5's "
        "label permutations from --seed + 1. Its own reduction, when cost forces "
        "one, is not a draw either -- --max-concepts drops concepts in ascending "
        "order of their eval bearing-group count, which is a deterministic "
        "quantity of the cohort it was handed"
    ),
    "16_fitness_recovery.py": (
        "its units are DMS variants of one wild type, not corpus sequences, so it "
        "draws through src.transfer.fitness.load_assay rather than the FASTA "
        "constructors. That draw is seeded and is a permutation of the eligible "
        "rows, never a prefix -- a ProteinGym CSV is ordered by position, so the "
        "hazard rule 1 names is present here and is answered in the same way"
    ),
    "29_designed_referent.py": (
        "its units are the MegaScale wild types EXP-R2-190's certificate searched "
        "and the measured variants of each, not corpus sequences, and it takes "
        "ALL of them: a census over 478 wild types and every eligible variant, "
        "exactly as the certificate itself was a census. Rule 1's hazard is "
        "absent rather than answered -- there is no sample, so there is no draw "
        "to seed and no skip offset to be sensitive to -- and the cohort artefact "
        "records `sampling.mode = census` so a later reader does not have to "
        "infer it. It builds its own Cohort only to reach `input_strings`, which "
        "renders sequences it already holds and selects nothing"
    ),
    "31_basis_adequacy.py": (
        "re-reads dictionaries that already exist and must be scored on the "
        "population they were scored on, so it constructs no cohort of its own: "
        "it calls 17_train_transcoder.py's held_out_cohort with the corpus seed, "
        "step budget, batch size and evaluation budget that dictionary's own "
        "settings block recorded, and refuses the cell when any of them is "
        "absent. A fresh draw here would be the failure the near-duplicate "
        "screen exists to make visible -- a different population reported under "
        "the dictionary's name -- so rule 1 is answered by the run being "
        "reproduced rather than by a seed this stage chooses"
    ),
    "32_crosscoder.py": (
        "DRAWS A COHORT and delegates the draw rather than constructing one. It "
        "trains a dictionary on a stream of millions of records, which the cohort "
        "constructors cannot serve for the reason 17_train_transcoder.py gives, so "
        "it calls that stage's seeded block-shuffled stream for training and its "
        "screened held_out_cohort for evaluation -- imported, not reimplemented, "
        "so a Crosscoder's per-site reconstruction is readable against the "
        "per-layer transcoders scored on the same population. Rule 1 is answered "
        "by --corpus-seed on both, and the held-out draw is additionally taken "
        "past everything the step budget reaches and screened for near-duplicates "
        "against the training stream, because a record-level offset is not a "
        "held-out set on a protein corpus (L30). The offset, the screen and the "
        "block size all reach the artefact. It does NOT use "
        "25_model_diffing_baselines.draw_splits: that stage fits a map on one half "
        "of one pool while this one trains against a step budget, and two "
        "definitions of the held-out set would be two populations under one name"
    ),
    "30_activation_spectrum.py": (
        "measures the rank of the activation cloud a dictionary was fitted on, "
        "so it must sample the population that dictionary was scored on and not "
        "a cohort of its own: it draws through 17_train_transcoder.py's seeded "
        "block-shuffled stream and near-duplicate screen, imported rather than "
        "reimplemented, at the offset that run's --steps and --batch-size "
        "produce. The stream is prefix-stable, so the dictionary run's own "
        "candidates are the first of this draw in the same order and its screen "
        "reproduces as a prefix, which the artefact reports as a specification "
        "check. Rule 1 is answered twice over: by --corpus-seed on the stream "
        "the reproduced run used, and by --seed on this stage's own choice of "
        "which token positions within each record to take, which is a uniform "
        "draw under an equal per-record cap rather than a prefix of each record"
    ),
    "33_differential_reliance.py": (
        "DRAWS A COHORT and delegates the draw, for 31_basis_adequacy.py's reason "
        "rather than 32_crosscoder.py's: it reads a Crosscoder that already exists "
        "and must ablate on the population that dictionary was held out on, so a "
        "cohort of its own would be a different population reported under the "
        "dictionary's name. It calls 17_train_transcoder.py's held_out_cohort with "
        "the corpus seed, step budget, evaluation budget and -- the one that is "
        "easy to get wrong -- the DICTIONARY's fit batch size rather than this "
        "stage's own, because the held-out offset is steps x batch_size and "
        "re-deriving it at the measurement's batch size would silently move the "
        "window. Rule 1 is answered by --corpus-seed, and here it is also enforced "
        "rather than merely passed: save_crosscoder records those five fields in "
        "the dictionary and assert_dictionary_matches refuses the run when any of "
        "them disagrees, so a mismatched population cannot reach a number. That "
        "guard is active rather than dormant -- it compares only fields the "
        "artefact actually recorded, and all five are recorded and were verified "
        "to refuse one at a time on 2026-08-18"
    ),
}


#: Every directory that can construct a cohort. `src/transfer` is included
#: because `probes.text_units` drew a 150-document file-order prefix of ~396,000
#: eligible documents while taking a `seed` it applied only to positions, and a
#: scan restricted to `scripts/transfer` could not see it.
SEARCH_DIRS = (
    REPO_ROOT / "scripts" / "transfer",
    REPO_ROOT / "scripts" / "transfer_gap",
    REPO_ROOT / "src" / "transfer",
)

#: Call sites that draw in file order ON PURPOSE, with the reason. Declared as a
#: prefix list rather than silently excluded from the scan, so that adding one is
#: a visible decision and the reason is checkable.
DELIBERATE_FILE_ORDER: dict[str, str] = {
    # `src/transfer/arms.py` is deliberately NOT here. It defines the constructors
    # and calls none of them, so exempting it exempted nothing -- and because the
    # filter matches a path prefix rather than a line, it would have silently
    # pre-exempted any future file-order draw added anywhere in that file.
    "scripts/transfer_gap/tg00_input_contract.py": (
        "TG-00 is the positive control that PRICES a file-order draw. It has to "
        "make one: its cohort_delta_nats is the difference between a file-order "
        "cohort and a seeded one, and a seeded control would measure nothing"
    ),
}


def stage_sources() -> dict[str, ast.Module]:
    trees: dict[str, ast.Module] = {}
    for path in sorted(STAGE_DIR.glob("*.py")):
        trees[path.name] = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return trees


def corpus_calls(tree: ast.Module) -> list[ast.Call]:
    """Calls to a corpus constructor, however it was named at the call site.

    Both ``text_cohort(...)`` and ``arms.text_cohort(...)`` count: matching only
    ``ast.Name`` let an attribute-style call reintroduce the defect invisibly.
    """

    found: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name in CORPUS_CONSTRUCTORS:
            found.append(node)
    return found


def seed_keyword(call: ast.Call) -> ast.keyword | None:
    for keyword in call.keywords:
        if keyword.arg == "seed":
            return keyword
    return None


def is_literal_none(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def argparse_default(tree: ast.Module, option: str) -> ast.AST | None:
    """The ``default=`` expression of one ``add_argument`` call, or None.

    Read from the call rather than by splitting the source on the option string:
    three stages mention ``--cohort-draw-seed`` in a docstring *before* they
    define it, so a text split inspected prose and the literal check it guarded
    was blind for those three -- and passed over a live defect in the one stage
    that qualifies every other stage's cohort.
    """

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        attr = func.attr if isinstance(func, ast.Attribute) else None
        if attr != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        if node.args[0].value != option:
            continue
        for keyword in node.keywords:
            if keyword.arg == "default":
                return keyword.value
        return None
    return None


class EveryCorpusDrawDeclaresItsSeed(unittest.TestCase):
    def test_every_call_site_in_every_live_directory_passes_seed(self):
        # Scoped over every directory that can build a cohort, not just the stage
        # scripts, and rejecting `seed=None` as well as an absent keyword: passing
        # the keyword with a literal None is the same file-order prefix with the
        # test satisfied.
        offenders: list[str] = []
        for directory in SEARCH_DIRS:
            for path in sorted(directory.glob("*.py")):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for call in corpus_calls(tree):
                    where = f"{path.relative_to(REPO_ROOT)}:{call.lineno}"
                    keyword = seed_keyword(call)
                    if keyword is None:
                        offenders.append(f"{where} has no seed=")
                    elif is_literal_none(keyword.value):
                        offenders.append(f"{where} passes seed=None")
        offenders = [
            entry
            for entry in offenders
            if not any(entry.startswith(prefix) for prefix in DELIBERATE_FILE_ORDER)
        ]
        self.assertEqual(
            offenders,
            [],
            "a corpus draw without an explicit non-None seed= is a file-order "
            "prefix; Appendix B rule 1. Offenders:\n  " + "\n  ".join(offenders),
        )

    def test_a_stage_that_draws_nothing_is_declared(self):
        # Guards the test above against becoming vacuous: if a stage stops
        # calling the constructors, that has to be a declared fact rather than a
        # silently empty iteration.
        for name, tree in stage_sources().items():
            if corpus_calls(tree):
                self.assertNotIn(
                    name,
                    NON_DRAWING_STAGES,
                    f"{name} is declared as not drawing a corpus but calls a constructor",
                )
            else:
                self.assertIn(
                    name,
                    NON_DRAWING_STAGES,
                    f"{name} draws no corpus and gives no reason; add one to "
                    "NON_DRAWING_STAGES or restore the draw",
                )

    def test_the_drawing_stages_are_the_ones_we_think(self):
        drawing = {name for name, tree in stage_sources().items() if corpus_calls(tree)}
        self.assertEqual(
            drawing,
            {
                "01_cohort_power.py",
                "02_pathway_budget.py",
                "03_estimand_power.py",
                "04_circuit_primitives.py",
                "06_explanation_channel.py",
                "07_convergence_control.py",
                "08_lens_family.py",
                "10_homology_control.py",
                "11_induction_path_patching.py",
                "13_induction_probe_bootstrap.py",
                "14_paa_census.py",
                "15_replacement_faithfulness.py",
                "18_das_subspace.py",
                "19_routing_locality.py",
                "21_joint_mode_qualification.py",
                "22_neuron_basis_circuit.py",
                "23_perturbation_sensitivity.py",
                # Draws circuit_primitives' analysis cohort, at the same band and
                # under the same seeded permutation, because the collision null it
                # builds has to be read against a census whose probes came from
                # that same unigram.
                "27_collision_null_census.py",
                # Draws text_cohort directly and builds its own accession-bearing
                # protein cohort, because a family-disjoint split needs the
                # accession that protein_cohort discards. That constructor is not
                # in CORPUS_CONSTRUCTORS, so the seed discipline it is held to is
                # the one below -- the flag, its imported default, and the
                # skip-offset window -- plus the stage's own check that its
                # eligible set matches arms._eligible_protein_records exactly.
                "26_concept_lens.py",
                # Its `cohort` stage draws nothing -- its units are ProteinGym
                # position pairs and the corpus alignments already searched by
                # 20_retrieval_bound.py -- but its `attainability` stage draws
                # both constructors, because A1's planted-coupling probes are
                # carried by real sequences and real documents rather than by
                # sampled symbols. One drawing stage inside the file is what the
                # contract governs, so it is registered here rather than
                # exempted on the strength of the half that does not draw.
                "28_epistasis_coupling.py",
            },
        )

    def test_the_repeat_constructors_offer_a_seed_at_all(self):
        # The library half of the same defect: a stage cannot pass a seed the
        # constructor does not take, and these two took one only after the plain
        # constructors did.
        import inspect

        from src.transfer.circuits import protein_repeat_cohort, text_repeat_cohort

        for function in (protein_repeat_cohort, text_repeat_cohort):
            self.assertIn("seed", inspect.signature(function).parameters, function.__name__)


class TheDrawSeedHasOneDeclaration(unittest.TestCase):
    def test_the_constant_lives_in_the_panel_declaration(self):
        from src.transfer.arms import DEFAULT_CORPUS_DRAW_SEED

        self.assertIsInstance(DEFAULT_CORPUS_DRAW_SEED, int)
        self.assertNotEqual(
            DEFAULT_CORPUS_DRAW_SEED,
            0,
            "0 selects the file-order draw; it cannot be the declared default",
        )

    def test_no_stage_restates_the_seed_as_a_literal(self):
        # Appendix B rule 12: a single declaration, imported, never reimplemented.
        # A stage that hard-codes the integer would silently stop tracking the
        # panel the day the declaration moves -- and one did, in the stage that
        # qualifies the cohorts every other stage is told to match.
        offenders: list[str] = []
        for name, tree in stage_sources().items():
            default = argparse_default(tree, "--cohort-draw-seed")
            if default is None:
                continue
            if isinstance(default, ast.Constant):
                offenders.append(f"{name} hard-codes {default.value!r}")
            elif not (isinstance(default, ast.Name) and default.id == "DEFAULT_CORPUS_DRAW_SEED"):
                offenders.append(f"{name} defaults to {ast.dump(default)[:60]}")
        self.assertEqual(
            offenders,
            [],
            "these stages do not default to the imported DEFAULT_CORPUS_DRAW_SEED: "
            + "; ".join(offenders),
        )

    def test_every_drawing_stage_defines_the_flag(self):
        # Defined as an argparse option, not merely mentioned: an `assertIn` over
        # the raw source passed for a stage that only named the flag in prose.
        sources = stage_sources()
        expected = {name for name, tree in sources.items() if corpus_calls(tree)} - {
            "06_explanation_channel.py"  # per-channel seeds; see its own contract test
        }
        for name in sorted(expected):
            self.assertIsNotNone(
                argparse_default(sources[name], "--cohort-draw-seed"),
                f"{name} draws a corpus but defines no --cohort-draw-seed option, so "
                "the file-order variant is unreachable and the seeded one is undeclared",
            )

    def test_the_repeat_constructors_honour_the_seed_they_accept(self):
        # Behavioural, not signature-only: a constructor that accepted `seed` and
        # ignored it in the body satisfied the signature check.
        from src.transfer.circuits import _select_matching

        found = [object() for _ in range(40)]
        def pick(seed):
            return _select_matching(found, n=8, skip=0, seed=seed, name="probe")

        file_order = pick(None)
        seeded = pick(20260728)
        other = pick(99)
        self.assertEqual(file_order, list(range(8)))
        self.assertNotEqual(seeded, file_order)
        self.assertNotEqual(seeded, other)
        self.assertEqual(seeded, pick(20260728))


class ASubsampleCarriesItsParentsDraw(unittest.TestCase):
    """A seeded subsample of a file-order prefix is still a file-order prefix."""

    def _cohort(self, seed):
        from src.transfer.arms import Cohort, sampling_record

        return Cohort(
            name="pool",
            kind="protein",
            records=[f"AAAA{index}" for index in range(20)],
            min_symbols=1,
            max_symbols=10,
            metadata={
                "sampling": sampling_record(
                    seed=seed, skip=0, requested=20, eligible=100, corpus="plain_swissprot"
                )
            },
        )

    def test_a_file_order_parent_keeps_its_hazard_through_the_subsample(self):
        from src.transfer.arms import FILE_ORDER_HAZARD
        from src.transfer.pathways import subsample_cohort

        child = subsample_cohort(self._cohort(seed=None), 5, 7)
        self.assertEqual(child.sampling["mode"], "file_order")
        self.assertEqual(child.sampling["hazard"], FILE_ORDER_HAZARD)
        self.assertEqual(child.sampling["subsample_seed"], 7)

    def test_a_seeded_parent_is_recorded_as_seeded(self):
        from src.transfer.pathways import subsample_cohort

        child = subsample_cohort(self._cohort(seed=11), 5, 7)
        self.assertEqual(child.sampling["mode"], "seeded_permutation")
        self.assertEqual(child.sampling["seed"], 11)
        self.assertNotIn("hazard", child.sampling)

    def test_the_parent_is_identified_so_the_draw_can_be_reproduced(self):
        from src.transfer.pathways import subsample_cohort

        parent = self._cohort(seed=11)
        child = subsample_cohort(parent, 5, 7)
        self.assertEqual(child.sampling["subsample_parent_digest"], parent.digest)
        self.assertEqual(child.sampling["subsample_parent_size"], len(parent))
        self.assertEqual(child.sampling["subsample_size"], 5)


class ASeededSkipIsDisjoint(unittest.TestCase):
    """--cohort-skip has to index a disjoint window, or it is not a sensitivity."""

    def test_two_skips_at_one_seed_share_no_record(self):
        from src.transfer.arms import selected_positions

        first = selected_positions(5000, n=400, skip=0, seed=20260728, label="a")
        second = selected_positions(5000, n=400, skip=400, seed=20260728, label="b")
        self.assertEqual(set(first) & set(second), set())

    def test_a_seeded_window_is_not_the_file_order_window(self):
        from src.transfer.arms import selected_positions

        seeded = selected_positions(5000, n=400, skip=0, seed=20260728, label="a")
        self.assertNotEqual(seeded, list(range(400)))



class ConditionedRenderingKeepsItsScoredSpan(unittest.TestCase):
    """One declaration of the token budget a conditioned rendering needs.

    ``content_bounds`` refuses a row whose ``<end>`` was truncated away, and the
    stages' shipped defaults -- a 256-token unigram window against a 1000-residue
    protein band -- put ZymCTRL's rows past it. The resolver was first written
    inside ``04_circuit_primitives.py``; ``11_induction_path_patching.py`` then
    failed on the same arm for the same reason, which is what a second copy of a
    decision buys. It now lives beside ``fit_unigram``.
    """

    def _arm(self, name):
        from src.transfer.arms import PANEL

        class _Arm:
            spec = PANEL[name]

        return _Arm()

    def test_only_a_conditioned_arm_is_widened(self):
        from src.transfer.circuits import CONDITIONING_TOKEN_SLACK, conditioned_token_budget

        self.assertEqual(
            conditioned_token_budget(self._arm("zymctrl"), 256, 1000),
            1000 + CONDITIONING_TOKEN_SLACK,
        )
        for name in ("protgpt2", "progen2-medium", "gpt2-large"):
            self.assertEqual(conditioned_token_budget(self._arm(name), 256, 1000), 256)

    def test_an_already_wide_request_is_not_narrowed(self):
        from src.transfer.circuits import conditioned_token_budget

        self.assertEqual(conditioned_token_budget(self._arm("zymctrl"), 4096, 1000), 4096)

    def test_both_stages_resolve_the_budget_and_neither_restates_it(self):
        for name in ("04_circuit_primitives.py", "11_induction_path_patching.py"):
            source = (STAGE_DIR / name).read_text(encoding="utf-8")
            self.assertIn("conditioned_token_budget(", source, name)
            self.assertNotIn("def conditioned_token_budget", source, name)
            self.assertNotIn("CONDITIONING_TOKEN_SLACK = ", source, name)
            self.assertNotIn(
                "max_tokens=args.unigram_max_tokens",
                source,
                f"{name} still fits its unigram on the unresolved window",
            )


class StageConsumersMatchTheirLibraryContracts(unittest.TestCase):
    """A key a stage reads must be a key the library still returns.

    `path_patching.structural_invariants` published a `passed` flag that could
    never be false -- any failure raises -- and it was removed as an unfalsifiable
    guard. Correct, but `11_induction_path_patching.py` still read it, so the stage
    measured all four arms on the H200 and then died writing its panel summary.
    The cost of a library change is paid by its consumers, and nothing was checking
    them.
    """

    def test_no_stage_reads_the_removed_invariants_passed_flag(self):
        import ast as _ast

        for directory in SEARCH_DIRS:
            for path in sorted(directory.glob("*.py")):
                source = path.read_text(encoding="utf-8")
                if "structural_invariants" not in source:
                    continue
                for node in _ast.walk(_ast.parse(source)):
                    if not isinstance(node, _ast.Subscript):
                        continue
                    key = node.slice
                    if isinstance(key, _ast.Constant) and key.value == "passed":
                        inner = node.value
                        rendered = _ast.dump(inner)
                        self.assertNotIn(
                            "structural_invariants",
                            rendered,
                            f"{path.relative_to(REPO_ROOT)}:{node.lineno} reads a "
                            "'passed' key that structural_invariants no longer returns",
                        )

    def test_structural_invariants_really_does_not_return_that_key(self):
        # Guards the test above against becoming vacuous if the key comes back.
        source = (REPO_ROOT / "src" / "transfer" / "path_patching.py").read_text(
            encoding="utf-8"
        )
        body = source.split("def structural_invariants(", 1)[1].split("\n@", 1)[0]
        self.assertNotIn('"passed":', body)


class TheConstructorListIsComplete(unittest.TestCase):
    """`CORPUS_CONSTRUCTORS` is hand-maintained, and that is how this went wrong.

    The tuple started with the two plain constructors and omitted the two repeat
    ones, so the contract test passed while the census that carries the headline
    part-1 result drew a four per cent head-of-file prefix. The fix added the two
    names and added nothing that would catch a fifth constructor. This does.

    A corpus-reading constructor is identified by what it does, not by its name:
    it is a public function in `src.transfer` that returns a `Cohort` and reads a
    corpus through `iter_fasta`, a parquet shard, or another such constructor.
    """

    def test_every_cohort_returning_constructor_in_the_library_is_listed(self):
        import inspect

        from src.transfer import arms, circuits

        missing: list[str] = []
        for module in (arms, circuits):
            source = Path(inspect.getfile(module)).read_text(encoding="utf-8")
            tree = ast.parse(source)
            for node in tree.body:
                if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
                    continue
                returns = getattr(node, "returns", None)
                name = returns.id if isinstance(returns, ast.Name) else None
                if name != "Cohort":
                    continue
                body = ast.get_source_segment(source, node) or ""
                reads_corpus = any(
                    token in body
                    for token in ("iter_fasta", "_eligible_", "pq.read_table", "text_cohort(", "protein_cohort(")
                )
                if reads_corpus and node.name not in CORPUS_CONSTRUCTORS:
                    missing.append(f"{module.__name__}.{node.name}")
        self.assertEqual(
            missing,
            [],
            "these return a Cohort and read a corpus but are not in "
            f"CORPUS_CONSTRUCTORS, so no test checks their call sites: {missing}",
        )

    def test_every_listed_constructor_exists(self):
        # Guards the other direction: a renamed constructor must not leave a dead
        # name in the tuple, which would make the call-site scan silently narrower.
        from src.transfer import arms, circuits

        for name in CORPUS_CONSTRUCTORS:
            self.assertTrue(
                hasattr(arms, name) or hasattr(circuits, name),
                f"{name} is listed but exists in neither arms nor circuits",
            )

    def test_declared_exemptions_name_files_that_exist(self):
        for prefix in DELIBERATE_FILE_ORDER:
            self.assertTrue(
                (REPO_ROOT / prefix).is_file(),
                f"{prefix} is exempted but does not exist; a stale exemption is a "
                "silently widened hole",
            )

    def test_non_drawing_stage_declarations_name_files_that_exist(self):
        for name in NON_DRAWING_STAGES:
            self.assertTrue(
                (STAGE_DIR / name).is_file(),
                f"{name} is declared as non-drawing but does not exist",
            )


if __name__ == "__main__":  # pragma: no cover
    import sys as _sys

    if str(REPO_ROOT) not in _sys.path:
        _sys.path.insert(0, str(REPO_ROOT))
    unittest.main()
