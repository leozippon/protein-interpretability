"""CPU contract for the InstructProtein family-generation feasibility probe.

Written against the properties the probe depends on rather than against its
implementation: the prompt is the checkpoint's declared rendering, the residue
span is recovered from the token ids by the identity rule the family declares,
and the class draw is the campaign's seeded permutation rather than a list a
caller could hand-pick. No model, no GPU and no corpus is read here.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer import conditioned_generation as cg  # noqa: E402
from src.transfer import joint_modes as jm  # noqa: E402


def _load_probe():
    path = REPO_ROOT / "scripts/transfer/instructprotein_family_generation_probe.py"
    spec = importlib.util.spec_from_file_location("instructprotein_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PROBE = _load_probe()


class TheResidueSpan(unittest.TestCase):
    """One residue per declared token, taken by identity and not by position."""

    LETTERS = {50267: "A", 50268: "C", 50269: "D"}
    START = 50265
    END = 50266
    TEXT = 199

    def test_the_leading_residue_run_is_the_sequence(self):
        ids = [50267, 50268, 50269, self.END, self.TEXT]
        self.assertEqual(PROBE.residue_run(ids, self.LETTERS), "ACD")

    def test_a_generation_that_leaves_the_residue_space_stops_there(self):
        ids = [50267, self.TEXT, 50268]
        self.assertEqual(PROBE.residue_run(ids, self.LETTERS), "A")

    def test_a_generation_that_never_enters_the_residue_space_is_empty(self):
        # The prompt ends at the declared protein start, so a tail that opens with
        # a non-residue id is a failure of the interface and not a sequence.
        self.assertEqual(PROBE.residue_run([self.START, 50267], self.LETTERS), "")
        self.assertEqual(PROBE.residue_run([self.TEXT, 50267], self.LETTERS), "")
        self.assertEqual(PROBE.residue_run([], self.LETTERS), "")

    def test_an_unterminated_run_is_still_the_sequence(self):
        self.assertEqual(PROBE.residue_run([50267, 50268], self.LETTERS), "AC")


class ThePromptIsTheDeclaredRendering(unittest.TestCase):
    """The intervention is the prompt the checkpoint was trained to receive."""

    def test_the_prompt_is_the_declared_context_scaffold_then_the_protein_start(self):
        declaration = jm.rendering("instructprotein")
        template = declaration.protein_context_template
        assert template is not None, "this probe's intervention is the declared scaffold"
        instruction = PROBE.INSTRUCTION_TEMPLATE.format(
            label="Crossover junction endodeoxyribonuclease RuvC"
        )
        prompt = PROBE.prompt_for(declaration, "Crossover junction endodeoxyribonuclease RuvC")
        self.assertEqual(
            prompt,
            "Instruction: I would like a protein that is in "
            "Crossover junction endodeoxyribonuclease RuvC.\nOutput: <protein>",
        )
        self.assertEqual(
            prompt,
            template.format(context=instruction) + declaration.protein_start,
        )

    def test_the_instruction_is_the_published_one_and_only_the_label_moves(self):
        declaration = jm.rendering("instructprotein")
        left = PROBE.prompt_for(declaration, "family one")
        right = PROBE.prompt_for(declaration, "family two")
        self.assertNotEqual(left, right)
        self.assertEqual(
            left.replace("family one", "{label}"), right.replace("family two", "{label}")
        )


class TheClassDraw(unittest.TestCase):
    """The campaign's seeded permutation, not a prefix and not a hand-pick."""

    def test_eligibility_needs_both_enough_records_and_a_published_name(self):
        census = {
            "PF1": {f"a{index}" for index in range(200)},
            "PF2": {f"b{index}" for index in range(199)},
            "PF3": {f"c{index}" for index in range(500)},
        }
        labels = {"PF1": ("one", "family one"), "PF2": ("two", "family two")}
        self.assertEqual(
            PROBE.eligible_families(census, labels, minimum=cg.MIN_CLASS_RECORDS),
            ("PF1",),
            "a family below the record count, or one the release does not name, is "
            "not a class this probe can draw or instruct",
        )

    def test_the_draw_is_reproducible_and_the_two_classes_differ(self):
        candidates = tuple(f"PF{index:05d}" for index in range(200))
        first = PROBE.class_draw(candidates, seed=PROBE.PROBE_SEED, n=2)
        self.assertEqual(first, PROBE.class_draw(candidates, seed=PROBE.PROBE_SEED, n=2))
        self.assertEqual(len(set(first)), 2)
        self.assertNotEqual(first, PROBE.class_draw(candidates, seed=PROBE.PROBE_SEED + 1, n=2))

    def test_the_draw_is_not_the_head_of_the_candidate_list(self):
        candidates = tuple(f"PF{index:05d}" for index in range(200))
        self.assertNotEqual(PROBE.class_draw(candidates, seed=PROBE.PROBE_SEED, n=2), candidates[:2])

    def test_a_draw_larger_than_the_admissible_set_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "never topped up"):
            PROBE.class_draw(("PF1",), seed=PROBE.PROBE_SEED, n=2)


if __name__ == "__main__":
    unittest.main()
