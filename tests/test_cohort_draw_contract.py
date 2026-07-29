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
        "draws through src.transfer.probes, whose record_order is seeded by default"
    ),
    "12_induction_robustness.py": "reads artefacts from disk; loads no corpus",
    "panel_contract.py": "a declaration, not a measurement",
}


def stage_sources() -> dict[str, ast.Module]:
    trees: dict[str, ast.Module] = {}
    for path in sorted(STAGE_DIR.glob("*.py")):
        trees[path.name] = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return trees


def corpus_calls(tree: ast.Module) -> list[ast.Call]:
    found: list[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in CORPUS_CONSTRUCTORS:
                found.append(node)
    return found


class EveryCorpusDrawDeclaresItsSeed(unittest.TestCase):
    def test_every_call_site_passes_seed(self):
        offenders: list[str] = []
        for name, tree in stage_sources().items():
            for call in corpus_calls(tree):
                keywords = {kw.arg for kw in call.keywords}
                if "seed" not in keywords:
                    offenders.append(f"{name}:{call.lineno} {call.func.id}(...) has no seed=")
        self.assertEqual(
            offenders,
            [],
            "a corpus draw without an explicit seed= is a file-order prefix; "
            "Appendix B rule 1. Offenders:\n  " + "\n  ".join(offenders),
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
        # panel the day the declaration moves.
        from src.transfer.arms import DEFAULT_CORPUS_DRAW_SEED

        literal = str(DEFAULT_CORPUS_DRAW_SEED)
        offenders: list[str] = []
        for path in sorted(STAGE_DIR.glob("*.py")):
            source = path.read_text(encoding="utf-8")
            if "--cohort-draw-seed" not in source:
                continue
            block = source.split("--cohort-draw-seed", 1)[1].split(")", 1)[0]
            if literal in block:
                offenders.append(path.name)
        self.assertEqual(
            offenders,
            [],
            f"these stages hard-code {literal} instead of importing "
            f"DEFAULT_CORPUS_DRAW_SEED: {offenders}",
        )

    def test_every_drawing_stage_offers_the_flag(self):
        expected = {
            name for name, tree in stage_sources().items() if corpus_calls(tree)
        } - {"06_explanation_channel.py"}  # per-channel seeds; see its own contract test
        for name in sorted(expected):
            source = (STAGE_DIR / name).read_text(encoding="utf-8")
            self.assertIn(
                "--cohort-draw-seed",
                source,
                f"{name} draws a corpus but the draw cannot be selected from the "
                "command line, so the file-order variant is unreachable and the "
                "seeded one is undeclared",
            )


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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
