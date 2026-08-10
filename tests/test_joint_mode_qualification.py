"""The declarations and guards behind the joint-checkpoint qualification stage.

Nothing here needs a GPU, a network or a 7B checkpoint. What is covered is the
part of ``src.transfer.joint_modes`` and
``scripts/transfer/21_joint_mode_qualification.py`` that can be wrong *silently*:
a rendering that merged residues, a residue alphabet the tokenizer does not
carry, a scored span that quietly included a delimiter, a held-out reference that
overlapped the sample it normalises, and a below-threshold mode reported as a
failure rather than as unmeasurable.

The tokenizers are stubs, and they are stubs of exactly the behaviour that caused
the defect: a vocabulary that merges adjacent residues into two-residue pieces
unless something stops it. ``StubTokenizer(split_marker=...)`` carries Galactica's
``Split``-and-remove pretokenizer rule, and the same stub without it is a
tokenizer that ignores the escape -- which is what an unaided ``AutoTokenizer``
call amounts to and what costs about 2.9 nats/token on the real checkpoint.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import inspect
import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from src.transfer import joint_modes as JM  # noqa: E402
from src.transfer.arms import AA20, Cohort, sampling_record, selected_positions  # noqa: E402
from src.transfer.budget import MEASURABLE, UNMEASURABLE  # noqa: E402


def _load_stage(filename: str):
    """Import a stage whose module name starts with a digit."""

    path = REPO / "scripts/transfer" / filename
    spec = importlib.util.spec_from_file_location(f"_stage_{filename[:2]}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE = _load_stage("21_joint_mode_qualification.py")


# ------------------------------------------------------------------ stub tokenizer


class StubTokenizer:
    """A tokenizer with the one behaviour that matters: it merges residues.

    Longest match wins over a vocabulary that contains every two-residue pair, so
    a bare protein string tokenises to roughly half as many pieces as it has
    residues -- the real defect, in twenty lines. ``split_marker`` reproduces
    Galactica's released ``Split``/``Removed`` pretokenizer rule: the literal is
    consumed and each fragment is tokenised on its own, which is what makes the
    escape work.
    """

    def __init__(
        self,
        *,
        specials: tuple[str, ...],
        split_marker: str | None = None,
        bos_id: int | None = None,
        alias: dict[str, str] | None = None,
    ) -> None:
        self.split_marker = split_marker
        self.bos_id = bos_id
        self._vocab: dict[str, int] = {}
        self._inverse: dict[int, str] = {}
        self._add("<unk>")
        for token in specials:
            self._add(token)
        for residue in AA20:
            self._add(residue)
        for left in AA20:
            for right in AA20:
                self._add(left + right)
        for character in "Ƥ abcdefghijklmnopqrstuvwxyz.,\n#":
            self._add(character)
        for token, target in (alias or {}).items():
            self._vocab[token] = self._vocab[target]
        self._longest = max(len(token) for token in self._vocab)
        self.unk_id = self._vocab["<unk>"]

    def _add(self, token: str) -> None:
        if token in self._vocab:
            return
        index = len(self._vocab) + 10
        self._vocab[token] = index
        self._inverse[index] = token

    def __len__(self) -> int:
        return len(self._vocab) + 10

    def convert_tokens_to_ids(self, token: str):
        return self._vocab.get(token)

    def convert_ids_to_tokens(self, index: int):
        return self._inverse.get(int(index))

    def _fragment(self, text: str) -> list[int]:
        ids: list[int] = []
        cursor = 0
        while cursor < len(text):
            for length in range(min(self._longest, len(text) - cursor), 0, -1):
                candidate = text[cursor : cursor + length]
                if candidate in self._vocab:
                    ids.append(self._vocab[candidate])
                    cursor += length
                    break
            else:
                ids.append(self.unk_id)
                cursor += 1
        return ids

    def __call__(self, text: str, return_tensors=None) -> dict[str, list[int]]:
        fragments = (
            text.split(self.split_marker) if self.split_marker else [text]
        )
        ids: list[int] = [] if self.bos_id is None else [self.bos_id]
        for fragment in fragments:
            ids.extend(self._fragment(fragment))
        return {"input_ids": ids}


def galactica_stub(*, honours_the_split_rule: bool = True) -> StubTokenizer:
    return StubTokenizer(
        specials=("[START_AMINO]", "[END_AMINO]"),
        split_marker=JM.GALACTICA_SPLIT_MARKER if honours_the_split_rule else None,
    )


def instructprotein_stub(*, drop: tuple[str, ...] = (), alias=None) -> StubTokenizer:
    residues = tuple(f"Ƥ{residue}" for residue in AA20 if residue not in drop)
    return StubTokenizer(
        specials=("<protein>", "</protein>") + residues,
        bos_id=2,
        alias=alias,
    )


SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"


# ------------------------------------------------------------------ declarations


class RenderingDeclaration(unittest.TestCase):
    def test_both_qualified_families_are_declared_in_one_place(self):
        self.assertEqual(sorted(JM.RENDERING_NAMES), ["galactica", "instructprotein"])
        self.assertIs(JM.rendering("galactica"), JM.JOINT_RENDERINGS["galactica"])

    def test_an_unknown_rendering_name_is_refused(self):
        with self.assertRaises(KeyError):
            JM.rendering("prollama")

    def test_the_stage_offers_exactly_the_declared_families_and_refuses_the_rest(self):
        parser = STAGE.build_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--checkpoint", "/nowhere", "--rendering", "prollama"])
        parsed = parser.parse_args(["--checkpoint", "/nowhere", "--rendering", "galactica"])
        self.assertEqual(parsed.rendering, "galactica")

    def test_the_checkpoint_is_required_because_it_is_not_a_panel_arm(self):
        parser = STAGE.build_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--rendering", "galactica"])

    def test_the_declared_strings_are_the_ones_the_checkpoints_were_trained_on(self):
        galactica = JM.rendering("galactica")
        marker = JM.GALACTICA_SPLIT_MARKER
        self.assertEqual(
            galactica.render_protein("MK"),
            f"[START_AMINO]{marker}M{marker}K{marker}[END_AMINO]",
        )
        self.assertEqual(
            galactica.render_protein("MK", variant=JM.NAIVE), "[START_AMINO]MK[END_AMINO]"
        )
        self.assertEqual(
            galactica.render_protein("MK", context="Serine protease"),
            f"# Serine protease\n\n[START_AMINO]{marker}M{marker}K{marker}[END_AMINO]",
        )
        instruct = JM.rendering("instructprotein")
        self.assertEqual(instruct.render_protein("MK"), "<protein>ƤMƤK</protein>")
        self.assertEqual(
            instruct.render_protein("MK", variant=JM.NAIVE), "<protein>MK</protein>"
        )
        self.assertEqual(
            instruct.render_protein("MK", context="I would like a protein."),
            "Instruction: I would like a protein.\nOutput: <protein>ƤMƤK</protein>",
        )

    def test_a_declaration_with_no_escape_is_refused_because_that_is_the_control(self):
        with self.assertRaises(ValueError):
            JM.JointRendering(
                name="broken",
                protein_start="<a>",
                protein_end="</a>",
                residue_escape="",
                escape_before_end_delimiter=False,
                protein_context_template=None,
                scored_target_rule=JM.BETWEEN_DELIMITERS,
                residue_subspace_disjoint_from_text=False,
                note="",
            )

    def test_a_declaration_with_an_undeclared_scoring_rule_is_refused(self):
        with self.assertRaises(ValueError):
            JM.JointRendering(
                name="broken",
                protein_start="<a>",
                protein_end="</a>",
                residue_escape="^",
                escape_before_end_delimiter=False,
                protein_context_template=None,
                scored_target_rule="whatever_looks_right",
                residue_subspace_disjoint_from_text=False,
                note="",
            )

    def test_a_non_canonical_sequence_is_refused_rather_than_rendered(self):
        with self.assertRaises(ValueError):
            JM.rendering("galactica").render_protein("MKTX")

    def test_an_unknown_variant_is_refused(self):
        with self.assertRaises(ValueError):
            JM.rendering("galactica").render_protein("MK", variant="escaped-ish")

    def test_only_the_family_with_dedicated_residue_tokens_declares_a_text_subspace(self):
        # This is what the text-mode control is gated on: Galactica's residues are
        # ordinary capital letters, so a "residue probability mass" on its text
        # mode would identify nothing.
        self.assertFalse(JM.rendering("galactica").residue_subspace_disjoint_from_text)
        self.assertTrue(JM.rendering("instructprotein").residue_subspace_disjoint_from_text)


# ----------------------------------------------------- resolution against a tokenizer


class ResolutionAgainstATokenizer(unittest.TestCase):
    def test_the_residue_alphabet_is_derived_from_the_declaration_itself(self):
        for name, tokenizer in (
            ("galactica", galactica_stub()),
            ("instructprotein", instructprotein_stub()),
        ):
            resolved = JM.resolve(tokenizer, name)
            self.assertEqual(sorted(resolved.residue_ids), sorted(AA20))
            self.assertEqual(len(set(resolved.residue_ids.values())), len(AA20))
            self.assertNotIn(resolved.start_id, set(resolved.residue_ids.values()))
            self.assertNotIn(resolved.end_id, set(resolved.residue_ids.values()))

    def test_a_tokenizer_that_does_not_carry_a_declared_residue_is_refused(self):
        # The declared alphabet is twenty tokens; a tokenizer missing one spells
        # it as two pieces, which is exactly the state a residue-level claim must
        # never be made in.
        with self.assertRaises(ValueError) as raised:
            JM.resolve(instructprotein_stub(drop=("W",)), "instructprotein")
        self.assertIn("per-residue alphabet", str(raised.exception))

    def test_two_residues_sharing_one_token_id_are_refused(self):
        with self.assertRaises(ValueError) as raised:
            JM.resolve(instructprotein_stub(alias={"ƤC": "ƤA"}), "instructprotein")
        self.assertIn("share a token id", str(raised.exception))

    def test_a_tokenizer_without_the_declared_delimiters_is_refused(self):
        with self.assertRaises(ValueError):
            JM.resolve(galactica_stub(), "instructprotein")


# -------------------------------------------------------- the per-residue guard


class PerResidueVerification(unittest.TestCase):
    """The defect this stage exists to prevent, from both directions."""

    def test_the_escaped_rendering_passes_and_is_one_token_per_residue(self):
        resolved = JM.resolve(galactica_stub(), "galactica")
        record = resolved.render(SEQUENCE)
        self.assertEqual(record.n_scored_tokens, len(SEQUENCE))
        self.assertEqual(record.residues_per_scored_token, 1.0)
        resolved.verify_one_token_per_residue(record)

    def test_the_naive_rendering_fails_verification_and_really_did_merge(self):
        resolved = JM.resolve(galactica_stub(), "galactica")
        naive = resolved.render(SEQUENCE, variant=JM.NAIVE)
        self.assertGreater(naive.residues_per_scored_token, 1.0)
        with self.assertRaises(ValueError) as raised:
            resolved.verify_one_token_per_residue(naive)
        self.assertIn("merged residues", str(raised.exception))

    def test_a_tokenizer_that_ignores_the_escape_raises_on_the_DECLARED_rendering(self):
        # The load-bearing case: the escape is present in the rendered string and
        # the tokenizer does not carry the Split rule that consumes it. The run
        # must stop, because the merged measurement looks exactly like a valid one.
        resolved = JM.resolve(
            galactica_stub(honours_the_split_rule=True), "galactica"
        )
        deaf = JM.JointTokenisation(
            declaration=resolved.declaration,
            tokenizer=galactica_stub(honours_the_split_rule=False),
            start_id=resolved.start_id,
            end_id=resolved.end_id,
            residue_ids=resolved.residue_ids,
        )
        with self.assertRaises(ValueError):
            deaf.render(SEQUENCE)

    def test_the_same_guard_holds_for_the_identity_rule_family(self):
        resolved = JM.resolve(instructprotein_stub(), "instructprotein")
        resolved.verify_one_token_per_residue(resolved.render(SEQUENCE))
        naive = resolved.render(SEQUENCE, variant=JM.NAIVE)
        with self.assertRaises(ValueError):
            resolved.verify_one_token_per_residue(naive)

    def test_the_declared_variant_is_verified_automatically_with_no_opt_out(self):
        # `render` takes no flag that skips the check, so a caller cannot obtain
        # an unverified declared record; only the naive control, whose name says
        # what it is, reaches a scored span without it.
        resolved = JM.resolve(galactica_stub(), "galactica")
        signature = inspect.signature(resolved.render)
        self.assertEqual(
            sorted(signature.parameters), ["context", "sequence", "variant"]
        )
        self.assertEqual(signature.parameters["variant"].default, JM.DECLARED)


# ------------------------------------------------------------- scored positions


class ScoredPositions(unittest.TestCase):
    def test_the_selector_returns_the_residues_and_excludes_both_delimiters(self):
        for name, tokenizer in (
            ("galactica", galactica_stub()),
            ("instructprotein", instructprotein_stub()),
        ):
            resolved = JM.resolve(tokenizer, name)
            record = resolved.render(SEQUENCE)
            ids = record.token_ids
            opening = ids.index(resolved.start_id)
            closing = ids.index(resolved.end_id)
            self.assertEqual(
                list(record.scored_positions),
                list(range(opening + 1, closing)),
                name,
            )
            self.assertNotIn(opening, record.scored_positions)
            self.assertNotIn(closing, record.scored_positions)
            self.assertEqual(
                [ids[position] for position in record.scored_positions],
                [resolved.residue_ids[residue] for residue in SEQUENCE],
                name,
            )

    def test_the_identity_rule_is_what_instructprotein_actually_uses(self):
        resolved = JM.resolve(instructprotein_stub(), "instructprotein")
        rendered = resolved.declaration.render_protein(SEQUENCE)
        by_identity = JM.scored_target_positions(
            resolved.tokenizer,
            resolved.declaration,
            rendered,
            residue_ids=resolved.residue_ids,
        )
        by_position = JM.scored_target_positions(
            resolved.tokenizer,
            resolved.declaration,
            rendered,
            rule=JM.BETWEEN_DELIMITERS,
        )
        self.assertEqual(by_identity, by_position)
        self.assertEqual(len(by_identity), len(SEQUENCE))

    def test_the_identity_rule_refuses_to_run_without_the_declared_alphabet(self):
        resolved = JM.resolve(instructprotein_stub(), "instructprotein")
        with self.assertRaises(ValueError):
            JM.scored_target_positions(
                resolved.tokenizer,
                resolved.declaration,
                resolved.declaration.render_protein(SEQUENCE),
            )

    def test_a_context_prefix_moves_the_span_rather_than_being_scored(self):
        resolved = JM.resolve(galactica_stub(), "galactica")
        bare = resolved.render(SEQUENCE)
        titled = resolved.render(SEQUENCE, context="a serine protease")
        self.assertEqual(titled.n_scored_tokens, bare.n_scored_tokens)
        self.assertGreater(min(titled.scored_positions), min(bare.scored_positions))

    def test_a_rendering_without_exactly_one_delimiter_pair_is_refused(self):
        resolved = JM.resolve(galactica_stub(), "galactica")
        with self.assertRaises(ValueError):
            JM.scored_target_positions(
                resolved.tokenizer, resolved.declaration, "[START_AMINO]"
            )


# ----------------------------------------------------------- the held-out draw


SCORED_RECORDS = ["".join(AA20) * 4, "".join(reversed(AA20)) * 4, ("MK" * 40)]
REFERENCE_RECORDS = [("AC" * 40), ("DE" * 40), ("FG" * 40)]


class HeldOutUnigramDraw(unittest.TestCase):
    """The reference must be a different sample, and the stage must prove it."""

    def _args(self) -> argparse.Namespace:
        return argparse.Namespace(
            sequences=len(SCORED_RECORDS),
            unigram_sequences=4,
            protein_min_len=8,
            protein_max_len=400,
            text_min_chars=8,
            cohort_draw_seed=20260728,
        )

    def _stub_draw(self, calls: list[dict]):
        def draw(n, min_len, max_len, *, skip=0, name="", with_ec=False, seed=None):
            calls.append({"n": n, "skip": skip, "seed": seed})
            # The reference block deliberately repeats one scored record, the way
            # Swiss-Prot repeats a sequence under several accessions.
            records = (
                list(SCORED_RECORDS)
                if skip == 0
                else [SCORED_RECORDS[0], *REFERENCE_RECORDS]
            )
            return Cohort(
                name,
                "protein",
                records[:n] if skip == 0 else records,
                min_len,
                max_len,
                {
                    "sampling": sampling_record(
                        seed=seed,
                        skip=skip,
                        requested=n,
                        eligible=64,
                        corpus="stub_swissprot",
                    )
                },
            )

        return draw

    def test_the_reference_is_drawn_past_the_scored_cohort_and_deduplicated(self):
        calls: list[dict] = []
        original = STAGE.protein_cohort
        STAGE.protein_cohort = self._stub_draw(calls)
        try:
            scored, reference, overlap = STAGE.mode_cohorts(self._args(), "protein")
        finally:
            STAGE.protein_cohort = original
        self.assertEqual([call["skip"] for call in calls], [0, len(SCORED_RECORDS)])
        self.assertEqual({call["seed"] for call in calls}, {20260728})
        self.assertEqual(set(scored.records) & set(reference.records), set())
        self.assertEqual(overlap["dropped_sequences_shared_with_cohort"], 1)
        self.assertEqual(reference.records, REFERENCE_RECORDS)
        # The reference is the denominator of the estimand, so how IT was drawn
        # has to reach the artefact too (Appendix B rule 1).
        self.assertEqual(reference.sampling["mode"], "seeded_permutation")
        self.assertEqual(reference.sampling["seed"], 20260728)
        self.assertEqual(reference.sampling["skip"], len(SCORED_RECORDS))

    def test_an_overlapping_reference_stops_the_run_rather_than_leaking(self):
        original_draw = STAGE.protein_cohort
        original_holdout = STAGE.held_out_cohort
        STAGE.protein_cohort = self._stub_draw([])
        # A deduplication step that silently kept the shared record: the final
        # assertion is what has to catch it.
        STAGE.held_out_cohort = lambda candidate, scored: (candidate, {})
        try:
            with self.assertRaises(ValueError):
                STAGE.mode_cohorts(self._args(), "protein")
        finally:
            STAGE.protein_cohort = original_draw
            STAGE.held_out_cohort = original_holdout

    def test_the_seeded_permutation_is_the_panels_own_and_two_windows_are_disjoint(self):
        # The draw itself is arms.selected_positions, imported rather than
        # reimplemented; this pins the property the stage relies on.
        first = selected_positions(500, n=64, skip=0, seed=20260728, label="scored")
        second = selected_positions(500, n=400, skip=64, seed=20260728, label="reference")
        self.assertEqual(set(first) & set(second), set())


# --------------------------------------------------------------- the verdicts


class Measurability(unittest.TestCase):
    def test_a_below_threshold_mode_is_unmeasurable_rather_than_failing(self):
        # galactica-1.3b's protein mode, at the value EXP-R2-151 measured.
        record = STAGE.verdict_record(0.1072, 0.30)
        self.assertEqual(record["verdict"], UNMEASURABLE)
        self.assertNotIn("FAIL", json.dumps(record))
        self.assertIn("not failing", record["verdict_note"])

    def test_a_mode_above_the_threshold_is_measurable(self):
        record = STAGE.verdict_record(4.6147, 0.30)
        self.assertEqual(record["verdict"], MEASURABLE)

    def test_a_collapsed_mode_is_still_reported_as_unmeasurable(self):
        # InstructProtein's text mode reads -12.03: worse than context-free, and
        # still a statement about this interface rather than a failed gate.
        record = STAGE.verdict_record(-12.0259, 0.30)
        self.assertEqual(record["verdict"], UNMEASURABLE)
        self.assertNotIn("FAIL", json.dumps(record))

    def test_the_threshold_defaults_to_the_one_the_panel_was_qualified_against(self):
        from src.transfer.budget import MIN_CONTEXT_INFORMATION_NATS

        parsed = STAGE.build_parser().parse_args(
            ["--checkpoint", "/nowhere", "--rendering", "galactica"]
        )
        self.assertEqual(parsed.min_context_information, MIN_CONTEXT_INFORMATION_NATS)


# ------------------------------------------------------------- unigram support


class UnigramSupport(unittest.TestCase):
    def test_the_protein_reference_is_fitted_over_the_residue_alphabet_only(self):
        resolved = JM.resolve(galactica_stub(), "galactica")
        counts = STAGE.residue_target_counts(resolved, [SEQUENCE], context=None)
        self.assertEqual(counts.size, len(AA20))
        self.assertEqual(int(counts.sum()), len(SEQUENCE))
        # Support size decides the smoothing bias, which is why it is recorded.
        record = STAGE.unigram_record(
            counts,
            counts,
            support="test",
            reference=Cohort("r", "protein", [SEQUENCE], 8, 400, {}),
            overlap={},
        )
        self.assertEqual(record["support_size"], len(AA20))
        self.assertLess(record["smoothing_mass_fraction"], 0.5)
        self.assertGreater(record["cross_entropy_nats"], 0.0)


class ScoringPrimitive(unittest.TestCase):
    """The two ways ``score_positions`` could be wrong without anything raising."""

    def _model(self, logits: "torch.Tensor"):
        class Fixed:
            def __call__(self, ids):
                return SimpleNamespace(logits=logits)

        return Fixed()

    def test_a_target_is_read_from_the_logits_of_the_position_before_it(self):
        torch.manual_seed(0)
        token_ids = [3, 1, 4, 1, 5, 9]
        logits = torch.randn(1, len(token_ids), 12)
        positions = [1, 3, 5]
        nll, mass = STAGE.score_positions(
            self._model(logits), token_ids, positions, device="cpu"
        )
        reference = torch.log_softmax(logits[0].float(), dim=-1)
        expected = [
            float(-reference[position - 1, token_ids[position]]) for position in positions
        ]
        self.assertEqual([round(value, 6) for value in nll], [round(v, 6) for v in expected])
        self.assertIsNone(mass)
        # The off-by-one this pins down is silent: it produces finite, plausible
        # cross-entropies for the wrong pairing.
        off_by_one = [
            float(-reference[position, token_ids[position]]) for position in positions
        ]
        self.assertNotEqual(
            [round(value, 6) for value in nll], [round(v, 6) for v in off_by_one]
        )

    def test_the_subspace_mass_is_the_probability_on_the_declared_residue_ids(self):
        torch.manual_seed(1)
        token_ids = [2, 7, 7, 3]
        logits = torch.randn(1, len(token_ids), 12)
        subspace = [0, 5, 11]
        _, mass = STAGE.score_positions(
            self._model(logits), token_ids, [1, 2, 3], device="cpu", subspace=subspace
        )
        probabilities = torch.softmax(logits[0].float(), dim=-1)
        expected = [float(probabilities[position - 1, subspace].sum()) for position in (1, 2, 3)]
        self.assertEqual([round(v, 6) for v in mass], [round(v, 6) for v in expected])

    def test_a_record_with_no_scored_position_is_refused(self):
        with self.assertRaises(ValueError):
            STAGE.score_positions(self._model(torch.zeros(1, 3, 5)), [1, 2, 3], [], device="cpu")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
