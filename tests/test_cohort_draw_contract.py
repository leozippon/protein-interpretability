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
    "panel_contract.py": "a declaration, not a measurement",
    "read_paa_panel.py": (
        "reads paa_gate_report.json files already on disk and recomputes their "
        "statistic through prediction_addressed.census_causal_agreement; it loads "
        "no model and constructs no cohort, so the draw it reports is whichever "
        "one the run it is reading recorded"
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
