"""The bridge from a joint checkpoint's two modes to a faithfulness number.

The experiment this file guards is the only dictionary comparison in the
programme that is not also a comparison of models. Every other one -- CLT against
PLT, ``gpt2-large`` against ``protgpt2``, ProGen3 against anything -- varies the
weights as well as the thing under test, which is how L25 came about: a
cross-layer transcoder's win at equal width turned out to be a 3.25x parameter
advantage and reversed under a parameter budget. Two per-layer transcoders on the
**two modes of one ProLLaMA checkpoint** share architecture, scale, lineage and
weights by construction, so a difference between their behavioural numbers cannot
be any of those.

It cannot be any of those *only if nothing else moved*, and four ways for
something else to move are silent:

**The wrong tensors.** A mode's dictionary must be fitted on the tensors that
mode is scored on. In protein mode those are the rendering's own scored span --
the token run whose spellings are the sequence -- and not the delimiters, the
instruction prefix or the beginning-of-sequence token; in text mode they are
every non-padding, non-special position. The two counts differ, so a run that
took the wrong one would train on a different population and nothing would raise.

**A mode the rendering cannot express.** Galactica declares the residue as its
symbol unit and reaches it only through an escape its released tokenizer
consumes; on a tokenizer that ignores the escape the declared alphabet does not
exist, and a protein run there would fit a dictionary to merged multi-residue
pieces -- worth 2.9 nats/token when it happened to a likelihood (EXP-R2-151).

**An unmatched pair.** Layers, width, ``k``, training tokens and the held-out
budget must agree between the two runs, and the *checkpoint* must too:
``prollama:protein`` names a mode, and three checkpoints of one lineage answer to
it.

**A dictionary scored on the other mode.** The two conditions share a checkpoint,
a depth and a width, so every shape check this repository has passes when a text
dictionary is spliced into the protein mode of the same weights. Only the
declared name separates them, and the artefact would otherwise read as a
faithfulness result.

The stubs are the ones ``tests/test_perturbation_sensitivity.py`` already builds
-- a randomly initialised two-layer LLaMA briefly overfitted on its own corpus,
and a tokenizer that merges residues and spells ``Seq=<`` out of ordinary pieces,
which are the two properties of the staged ProLLaMA tokenizer that decide these
numbers. Imported rather than copied: a second copy of a fixture is a second
definition of what the stub IS. No GPU, no network, no 7B checkpoint.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for entry in (REPO, REPO / "scripts/transfer", Path(__file__).resolve().parent):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.transfer import joint_modes as JM  # noqa: E402
from src.transfer import replaceable as R  # noqa: E402
from src.transfer.progen3 import Component as R_Component  # noqa: E402
from src.transfer.transcoders import (  # noqa: E402
    MATCHED_TRAINING_KEY,
    load_trained_transcoder,
    matched_training_from_artefact,
)
from test_perturbation_sensitivity import (  # noqa: E402
    DOCUMENTS,
    SEQUENCES,
    _dense,
    _llama,
    galactica_stub_without_the_escape,
    prollama_stub,
)


def _load_stage(filename: str):
    """Import a stage whose module name starts with a digit."""

    path = REPO / "scripts/transfer" / filename
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


STAGE15 = _load_stage("15_replacement_faithfulness.py")
STAGE17 = _load_stage("17_train_transcoder.py")


def _joint(mode: str, *, tokenizer=None, max_tokens: int = 128) -> R.JointReplaceable:
    tokenizer = tokenizer or prollama_stub()
    declaration = JM.rendering("prollama")
    return R.JointReplaceable(
        model=_llama(tokenizer),
        tokenizer=tokenizer,
        checkpoint=Path("/nowhere"),
        declaration=declaration,
        mode=mode,
        tokenisation=R.joint_tokenisation(tokenizer, declaration, mode),
        max_tokens=max_tokens,
    )


# --------------------------------------------------- what a mode actually scores


class TheDictionaryIsFittedOnWhatTheModeScores(unittest.TestCase):
    """The tensors the trainer captures are the mode's own scored positions."""

    def test_protein_mode_captures_the_renderings_scored_span_and_nothing_else(self):
        model = _joint("protein")
        records = [(sequence, None) for sequence in SEQUENCES]
        x, y, mask = STAGE17.capture(model, records)
        kept_x, kept_y = STAGE17.flatten(x, y, mask)

        expected = sum(
            model.tokenisation.render(sequence, context=None).n_scored_tokens
            for sequence in SEQUENCES
        )
        self.assertEqual(kept_x.shape, (model.n_layers, expected, model.width))
        self.assertEqual(kept_y.shape, kept_x.shape)
        # And that really is fewer than the rendered strings carry: the
        # delimiters, the instruction prefix and the beginning-of-sequence token
        # are outside the span, so a run that took every position would fit the
        # dictionary to a different population.
        self.assertLess(expected, int(mask.numel()))

    def test_the_two_modes_capture_different_token_populations(self):
        tokenizer = prollama_stub()
        protein = _joint("protein", tokenizer=tokenizer)
        text = _joint("text", tokenizer=tokenizer)

        def kept(model, records):
            x, y, mask = STAGE17.capture(model, [(record, None) for record in records])
            return STAGE17.flatten(x, y, mask)[0].shape[1]

        protein_tokens = kept(protein, SEQUENCES)
        text_tokens = kept(text, DOCUMENTS)
        self.assertGreater(protein_tokens, 0)
        self.assertGreater(text_tokens, 0)
        self.assertNotEqual(protein_tokens, text_tokens)
        # Text mode keeps every non-special position of its own tokenisation,
        # which is the unconditioned convention every dense arm scores under.
        expected = sum(
            len([i for i in JM.encode(tokenizer, document) if i not in tokenizer.all_special_ids])
            for document in DOCUMENTS
        )
        self.assertEqual(text_tokens, expected)

    def test_the_captured_pair_is_the_block_this_estimand_is_defined_on(self):
        # The trainer reads block input and predicts block output; the identity
        # that certifies the interception is verified exactly on the live pass.
        record = _joint("protein").self_check()
        self.assertEqual(record["estimand"]["verdict"], "PASS")
        self.assertEqual(record["estimand"]["max_absolute_difference"], 0.0)

    def test_the_rendering_records_do_not_accumulate_across_batches(self):
        # Unbounded otherwise, and proportional to the step count: the joint
        # campaign renders of order 3e5 protein records over a run and no later
        # step reads any of them.
        model = _joint("protein")
        for _ in range(3):
            STAGE17.capture(model, [(sequence, None) for sequence in SEQUENCES])
            self.assertEqual(model._rendered, {})
        # And the implementations that keep no such state see a no-op.
        self.assertIsNone(_dense().forget_rendered())


class AModeTheRenderingCannotExpressIsRefused(unittest.TestCase):
    def test_a_tokenizer_without_the_declared_alphabet_refuses_protein_mode(self):
        with self.assertRaises(ValueError) as caught:
            R.joint_tokenisation(
                galactica_stub_without_the_escape(), JM.rendering("galactica"), "protein"
            )
        self.assertIn("per-residue alphabet", str(caught.exception))

    def test_the_same_tokenizer_is_admitted_in_text_mode_and_says_it_was_not_resolved(self):
        # Text mode's scored positions are the tokenizer's own next-token targets
        # and do not depend on the protein format, so resolving would refuse a
        # measurable mode on a property the run never reads.
        tokenizer = galactica_stub_without_the_escape()
        self.assertIsNone(
            R.joint_tokenisation(tokenizer, JM.rendering("galactica"), "text")
        )

    def test_an_undeclared_mode_is_refused_by_name(self):
        with self.assertRaises(ValueError):
            R.joint_tokenisation(prollama_stub(), JM.rendering("prollama"), "nucleotide")
        with self.assertRaises(ValueError):
            R.joint_mode_corpus("nucleotide")

    def test_an_undeclared_rendering_family_is_refused(self):
        with self.assertRaises(KeyError):
            JM.rendering("not-a-family")

    def test_each_mode_trains_and_scores_on_one_declared_corpus(self):
        # A dictionary trained on one population and scored on another is the
        # train/eval gap EXP-R2-135 priced at 4.1x in NLL recovery.
        self.assertEqual(R.joint_mode_corpus("protein"), "swissprot")
        self.assertEqual(R.joint_mode_corpus("text"), "openwebtext")
        for mode in R.JOINT_MODES:
            args = argparse.Namespace(
                joint_checkpoint=Path("/nowhere"), rendering="prollama", mode=mode, arm=None
            )
            self.assertEqual(STAGE15.cohort_source(args), R.joint_mode_corpus(mode))


class TheProteinBandCannotProduceARecordThatDoesNotFit(unittest.TestCase):
    """A rendered protein is never truncated, so the band has to guarantee it."""

    def _band(self, max_tokens: int, context: str | None = None):
        tokenizer = prollama_stub()
        tokenisation = JM.resolve(tokenizer, JM.rendering("prollama"))
        return STAGE17.joint_protein_band(
            tokenisation, max_tokens=max_tokens, protein_context=context
        ), tokenisation

    def test_the_ceiling_pays_for_the_measured_wrapper(self):
        (low, high), tokenisation = self._band(128)
        self.assertEqual(low, STAGE17.MIN_RESIDUES)
        self.assertLess(high, 128)
        # Every record at the ceiling renders inside the cap, because a merged
        # rendering only ever costs fewer tokens than residues.
        longest = "A" * high
        self.assertLessEqual(len(tokenisation.render(longest).token_ids), 128)

    def test_a_document_context_is_paid_for_too(self):
        (_, bare), _ = self._band(128)
        (_, with_context), tokenisation = self._band(128, context="Hydrolase")
        self.assertLess(with_context, bare)
        self.assertLessEqual(
            len(tokenisation.render("A" * with_context, context="Hydrolase").token_ids), 128
        )

    def test_a_cap_that_cannot_hold_the_floor_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self._band(4)
        self.assertIn("wrapper", str(caught.exception))

    def test_the_declared_ceiling_still_bounds_a_generous_cap(self):
        (_, high), _ = self._band(100_000)
        self.assertEqual(high, STAGE17.MAX_RESIDUES)


# ------------------------------------------------------- the trainer, end to end


class _StubCorpus:
    """A corpus large enough that the held-out draw clears the training budget.

    ``17_train_transcoder.py`` skips a whole number of shuffle blocks past
    everything training can reach, so the stub has to carry more than one block --
    which is the arithmetic that makes the two sets disjoint and is therefore
    worth exercising rather than patching away.
    """

    def __init__(self, records: list[str]) -> None:
        self.records = records

    def __call__(self, source, *, min_symbols, max_symbols=None, path=None):
        cycle = self.records
        return (
            (cycle[index % len(cycle)], None)
            for index in range(STAGE17.SHUFFLE_BLOCK + 8)
        )


@contextlib.contextmanager
def _trainer_on_stubs(tokenizer, backbone, records):
    """The trainer with its loaders and its corpus replaced, and nothing else."""

    saved = (
        STAGE17.STAGE21.load_tokenizer,
        STAGE17.STAGE21.load_model,
        STAGE17.corpus_location,
        STAGE17.iter_corpus_records,
        R.checkpoint_weights_digest,
        sys.argv,
    )
    facts = {
        "resolved_path": "/nowhere",
        "model_type": "llama",
        "n_layers": backbone.config.num_hidden_layers,
        "d_model": backbone.config.hidden_size,
        "n_heads": backbone.config.num_attention_heads,
        "vocab_size": len(tokenizer),
        "dtype_requested": "float32",
        "dtype_observed": ["float32"],
    }
    STAGE17.STAGE21.load_tokenizer = lambda path: (Path(path), tokenizer)
    STAGE17.STAGE21.load_model = lambda resolved, tok, *, device, dtype: (
        backbone,
        dict(facts),
    )
    STAGE17.corpus_location = lambda source, path=None: Path(f"/nowhere/{source}")
    STAGE17.iter_corpus_records = _StubCorpus(records)
    R.checkpoint_weights_digest = lambda path: "c" * 64
    try:
        yield
    finally:
        (
            STAGE17.STAGE21.load_tokenizer,
            STAGE17.STAGE21.load_model,
            STAGE17.corpus_location,
            STAGE17.iter_corpus_records,
            R.checkpoint_weights_digest,
            sys.argv,
        ) = saved


def _train(directory: Path, mode: str, tokenizer, backbone, **overrides) -> Path:
    """One dictionary, trained through the real ``main`` on stub loaders."""

    settings = {
        "--d-hidden": "8",
        "--k": "2",
        "--auxk": "2",
        "--steps": "4",
        "--batch-size": "2",
        "--eval-sequences": "4",
        "--eval-every": "100",
        "--max-tokens": "128",
        "--train-tokens": "10",
    }
    settings.update(overrides)
    records = SEQUENCES if mode == "protein" else DOCUMENTS
    with _trainer_on_stubs(tokenizer, backbone, list(records)):
        sys.argv = [
            "17_train_transcoder.py",
            "--architecture", "plt",
            "--joint-checkpoint", "/nowhere",
            "--rendering", "prollama",
            "--mode", mode,
            "--device", "cpu",
            "--out", str(directory),
        ] + [value for pair in settings.items() for value in pair]
        with contextlib.redirect_stdout(io.StringIO()):
            STAGE17.main()
    stem = f"prollama_{mode}_plt_d{settings['--d-hidden']}_k{settings['--k']}_s20260806"
    return directory / f"{stem}.pt"


class TheTrainerReachesAJointCheckpointByPathAndMode(unittest.TestCase):
    """The whole driver, on stubs: the part no CPU host can otherwise reach."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = prollama_stub()
        cls.backbone = _llama(cls.tokenizer)

    def test_both_modes_train_and_declare_a_matched_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protein = _train(root, "protein", self.tokenizer, self.backbone)
            text = _train(root, "text", self.tokenizer, self.backbone)
            self.assertTrue(protein.is_file() and text.is_file())
            self.assertNotEqual(protein, text, "the two modes must not overwrite each other")

            payload = json.loads(protein.with_suffix(".json").read_text())
            self.assertEqual(payload["target"]["kind"], "joint_checkpoint")
            self.assertEqual(payload["target"]["mode"], "protein")
            self.assertEqual(payload["target"]["name"], "prollama:protein")
            self.assertEqual(payload["target"]["rendering"]["name"], "prollama")
            self.assertEqual(payload["condition"]["corpus_source"], "swissprot")
            self.assertIn("joint_modes", payload["condition"]["input_rendering"])
            # The text mode records that it needed no protein rendering rather
            # than omitting the field.
            other = json.loads(text.with_suffix(".json").read_text())
            self.assertEqual(other["target"]["rendering"]["verdict"], "NOT_RESOLVED")
            self.assertEqual(other["condition"]["corpus_source"], "openwebtext")

            # Both artefacts carry every field the pair is refused on, and the
            # dictionaries carry it too, so a checkpoint handed to the
            # faithfulness stage states its own configuration.
            left = matched_training_from_artefact(protein.with_suffix(".json"))
            right = matched_training_from_artefact(text.with_suffix(".json"))
            self.assertEqual(left.training_token_budget, 10)
            self.assertEqual(left.digest(), right.digest())
            self.assertNotEqual(left.target, right.target)
            _, recorded, declared = load_trained_transcoder(protein)
            self.assertEqual(recorded["arm"], "prollama:protein")
            self.assertEqual(declared, left)

            # The realised token counts differ between the modes at an identical
            # budget -- which is exactly why the budget and not the step count is
            # what the two runs are matched on.
            self.assertNotEqual(left.training_tokens, right.training_tokens)

    def test_the_token_budget_stops_the_run_and_a_missed_one_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _train(root, "protein", self.tokenizer, self.backbone)
            payload = json.loads(path.with_suffix(".json").read_text())
            self.assertGreaterEqual(payload["training"]["tokens"], 10)
            self.assertLess(payload["training"]["steps"], 4, "the budget must stop it")

            with self.assertRaises(RuntimeError) as caught:
                _train(
                    root,
                    "protein",
                    self.tokenizer,
                    self.backbone,
                    **{"--train-tokens": "100000000"},
                )
            self.assertIn("scored tokens", str(caught.exception))

    def test_without_a_budget_the_loop_is_the_one_that_predates_it(self):
        # The regression the flag must not break: --train-tokens 0 runs --steps
        # steps, evaluates at the last one, and declares no budget.
        with tempfile.TemporaryDirectory() as directory:
            path = _train(
                Path(directory),
                "text",
                self.tokenizer,
                self.backbone,
                **{"--train-tokens": "0", "--steps": "3"},
            )
            payload = json.loads(path.with_suffix(".json").read_text())
            self.assertEqual(payload["training"]["steps"], 3)
            self.assertEqual(payload["training"]["history"][-1]["step"], 3)
            self.assertIsNone(payload[MATCHED_TRAINING_KEY]["training_token_budget"])
            self.assertIn("no token budget", payload["condition"]["training_budget"])


# ------------------------------------------------------- refusing an unmatched pair


class TheFaithfulnessStageRefusesAnUnmatchedPair(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = prollama_stub()
        cls.backbone = _llama(cls.tokenizer)
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        cls.protein = _train(root, "protein", cls.tokenizer, cls.backbone)
        cls.text = _train(root, "text", cls.tokenizer, cls.backbone)
        cls.wide = _train(
            root, "text", cls.tokenizer, cls.backbone, **{"--d-hidden": "16"}
        )

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def _declared(self, path: Path):
        return load_trained_transcoder(path)[2]

    def test_the_matched_pair_passes_and_reaches_the_artefact(self):
        record = STAGE15.matched_pair_record(
            self._declared(self.protein), self.text.with_suffix(".json"), kind="local"
        )
        self.assertEqual(record["verdict"], "MATCHED")
        self.assertEqual(record["comparison"]["verdict"], "MATCHED")
        self.assertTrue(record["comparison"]["distinct_targets"])
        self.assertEqual(record["this_run"]["target"], "prollama:protein")
        self.assertEqual(record["other_run"]["target"], "prollama:text")

    def test_a_checkpoint_and_artefact_predating_the_optimisation_fields_still_pair(self):
        """The exact shape of the four dictionaries already on GPFS.

        They were written before the matched declaration covered the optimiser,
        so their ``matched_training`` block carries none of those fields -- but
        their settings block carries every one, in the checkpoint under ``record``
        and in the JSON at the top level. Both readers recover them from there.

        This is the path a re-dispatch takes on the first file it opens. Requiring
        the fields instead would refuse four already-trained dictionaries over
        values recorded three lines away, and the refusal would land after the
        model was loaded rather than at a declaration.
        """

        widened = (
            "auxk", "learning_rate", "weight_decay", "grad_clip",
            "batch_size", "seed", "corpus_seed", "max_tokens",
        )

        def aged(path: Path) -> Path:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            for name in widened:
                checkpoint[MATCHED_TRAINING_KEY].pop(name, None)
            older = path.with_name("aged_" + path.name)
            torch.save(checkpoint, older)
            payload = json.loads(path.with_suffix(".json").read_text())
            for name in widened:
                payload[MATCHED_TRAINING_KEY].pop(name, None)
            older.with_suffix(".json").write_text(json.dumps(payload), encoding="utf-8")
            return older

        old_protein, old_text = aged(self.protein), aged(self.text)
        record = STAGE15.matched_pair_record(
            load_trained_transcoder(old_protein)[2],
            old_text.with_suffix(".json"),
            kind="local",
        )
        self.assertEqual(record["comparison"]["verdict"], "MATCHED")
        self.assertEqual(record["comparison"]["disagreements"], [])
        # Recovered, not defaulted: the values are this run's own.
        self.assertEqual(
            record["this_run"]["learning_rate"],
            json.loads(self.protein.with_suffix(".json").read_text())["settings"][
                "learning_rate"
            ],
        )
        # And a declaration with neither source is refused rather than guessed.
        stripped = json.loads(old_text.with_suffix(".json").read_text())
        stripped.pop("settings")
        orphan = old_text.with_name("orphan.json")
        orphan.write_text(json.dumps(stripped), encoding="utf-8")
        with self.assertRaises(KeyError):
            matched_training_from_artefact(orphan)

    def test_a_width_mismatch_refuses_the_run(self):
        with self.assertRaises(RuntimeError) as caught:
            STAGE15.matched_pair_record(
                self._declared(self.protein), self.wide.with_suffix(".json"), kind="local"
            )
        self.assertIn("d_hidden", str(caught.exception))

    def test_a_different_backbone_refuses_the_run(self):
        # 'prollama:protein' names a MODE, and three checkpoints of one lineage
        # answer to it, so the weight digest is the only field that says which.
        other = json.loads(self.text.with_suffix(".json").read_text())
        other[MATCHED_TRAINING_KEY]["backbone_sha256"] = "d" * 64
        forged = Path(self.directory.name) / "forged.json"
        forged.write_text(json.dumps(other), encoding="utf-8")
        with self.assertRaises(RuntimeError) as caught:
            STAGE15.matched_pair_record(
                self._declared(self.protein), forged, kind="local"
            )
        self.assertIn("backbone_sha256", str(caught.exception))

    def test_without_the_flag_the_pair_is_recorded_as_unchecked_rather_than_passed(self):
        record = STAGE15.matched_pair_record(
            self._declared(self.protein), None, kind="local"
        )
        self.assertEqual(record["verdict"], "NOT_CHECKED")
        self.assertIn("digest", record["this_run"])
        self.assertIn("--matched-against", record["reason"])

    def test_a_replacement_with_no_declaration_says_so(self):
        record = STAGE15.matched_pair_record(None, None, kind="linear")
        self.assertEqual(record["verdict"], "WITHHELD")
        self.assertIn("linear", record["reason"])

    def test_the_flag_is_refused_rather_than_ignored_when_there_is_nothing_to_check(self):
        # Accepting --matched-against against a replacement that declares nothing
        # would report an unchecked pair as a checked one.
        with self.assertRaises(RuntimeError) as caught:
            STAGE15.matched_pair_record(
                None, self.text.with_suffix(".json"), kind="linear"
            )
        self.assertIn("--matched-against", str(caught.exception))

    def test_the_scoring_budget_carries_its_own_digest(self):
        def budget(**overrides):
            settings = {
                "sequences": 128,
                "bootstrap": 1000,
                "max_tokens": 512,
                "protein_min_len": 64,
                "protein_max_len": 246,
                "text_min_chars": 800,
                "cohort_draw_seed": 20260728,
                "cohort_skip": 0,
                "gate_recovery": 0.8,
                "gate_rho": 0.5,
                "matched_perturbation_draws": 0,
            }
            settings.update(overrides)
            return STAGE15.scoring_budget(argparse.Namespace(**settings))

        self.assertEqual(budget()["digest"], budget()["digest"])
        self.assertNotEqual(budget()["digest"], budget(sequences=64)["digest"])
        self.assertNotEqual(budget()["digest"], budget(bootstrap=500)["digest"])
        self.assertEqual(sorted(budget()["fields"]), sorted(STAGE15.SCORING_BUDGET_FIELDS))


class ADictionaryScoredOnTheOtherModeIsRefused(unittest.TestCase):
    """The failure with no shape to catch it, and the reason `arm` is recorded."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = prollama_stub()
        cls.backbone = _llama(cls.tokenizer)

    def test_the_two_modes_are_indistinguishable_by_every_shape_check(self):
        protein = _joint("protein", tokenizer=self.tokenizer)
        text = _joint("text", tokenizer=self.tokenizer)
        self.assertEqual(protein.n_layers, text.n_layers)
        self.assertEqual(protein.width, text.width)
        self.assertEqual(protein.checkpoint, text.checkpoint)
        self.assertNotEqual(protein.name, text.name)

    def test_a_text_dictionary_scored_on_protein_is_refused_by_name(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _train(Path(directory), "text", self.tokenizer, self.backbone)
            _, recorded, _ = load_trained_transcoder(path)
            with self.assertRaises(RuntimeError) as caught:
                STAGE15.require_matching_arm(
                    recorded["arm"], _joint("protein", tokenizer=self.tokenizer).name,
                    required=True,
                )
            self.assertIn("prollama:text", str(caught.exception))
            # And the same dictionary on its own mode is accepted and said so.
            self.assertTrue(
                STAGE15.require_matching_arm(
                    recorded["arm"], _joint("text", tokenizer=self.tokenizer).name,
                    required=True,
                )
            )

    def test_an_undeclared_arm_is_refused_on_the_joint_path_only(self):
        # A joint pair shares depth, width and checkpoint, so an undeclared
        # dictionary could be either mode and the artefact would look identical.
        with self.assertRaises(RuntimeError) as caught:
            STAGE15.require_matching_arm(None, "prollama:protein", required=True)
        self.assertIn("declares no arm", str(caught.exception))
        # The four ProGen3 checkpoints written before the record existed keep
        # loading, and the artefact says the check did not run.
        self.assertFalse(STAGE15.require_matching_arm(None, "progen3"))


# ------------------------------------------------ the causal sweep, end to end


class TheCausalSweepRunsOnTheGridTheModelDeclares(unittest.TestCase):
    """A joint checkpoint must be scoreable to the end, or refuse before the cost.

    Nothing exercised the causal sweep on any model, and the gap cost a live
    campaign four scoring runs. ``components()`` emitted every attention head for
    every implementation, ``JointReplaceable.ablated`` refused every one of them,
    and the two only meet at the last step of the stage -- after the behavioural
    sweep is computed and before anything is written -- so each run died with its
    numbers in memory and nothing on disk, and re-dispatching reproduced it.

    So: the grid a joint model declares is a grid it will ablate, the sweep over
    that grid completes, a grid it will *not* ablate is refused where refusing is
    free rather than where it is expensive, and a dense arm's two families are
    untouched by the restriction.
    """

    def test_the_joint_grid_is_the_block_family_and_every_component_ablates(self):
        model = _joint("protein")
        grid = model.components()
        self.assertEqual(STAGE15.families(grid), (model.block_kind,))
        self.assertEqual(len(grid), model.n_layers)
        self.assertEqual({component.kind for component in grid}, {model.block_kind})
        for component in grid:
            with model.ablated(component):
                pass

    def test_the_causal_sweep_completes_on_a_joint_checkpoint(self):
        model = _joint("protein")
        grid = model.components()
        inputs = model.render(SEQUENCES[:2])
        effects = STAGE15.component_effects(model, inputs, grid, batch_size=2)
        self.assertEqual(effects.shape, (len(grid), len(inputs)))
        self.assertTrue(np.isfinite(effects).all())

    def test_a_component_the_model_refuses_is_caught_before_any_sweep(self):
        model = _joint("protein")
        smuggled = model.components() + [R_Component("attention_head", 0, 0)]
        with self.assertRaises(ValueError) as caught:
            STAGE15.check_grid_is_ablatable(model, smuggled)
        self.assertIn("attention_head", str(caught.exception))

    def test_the_artefact_says_which_comparison_this_grid_licenses(self):
        """Both halves of the scoping rule, on the side each applies to.

        A one-family artefact and a two-family artefact are not the same amount
        of evidence, so joint-against-dense is refused. The within-checkpoint pair
        is not that comparison: both modes carry the identical block-only grid, so
        the missing family cancels and the pair -- which holds the weights fixed
        and varies only the mode -- stays admissible. An artefact that stated only
        the refusal would read as if the whole measurement were compromised.
        """

        joint = STAGE15.component_family_comparability(
            STAGE15.families(_joint("protein").components())
        )
        self.assertIn("ONLY", joint)
        self.assertIn("NOT like-for-like", joint)
        self.assertIn("OTHER MODE", joint)
        self.assertIn("IS like-for-like", joint)

        # And the two modes really do carry the same grid, which is what makes
        # the second half true rather than merely asserted.
        tokenizer = prollama_stub()
        self.assertEqual(
            _joint("protein", tokenizer=tokenizer).components(),
            _joint("text", tokenizer=tokenizer).components(),
        )

        dense = STAGE15.component_family_comparability(
            STAGE15.families(_dense().components())
        )
        self.assertIn("both component families", dense)
        self.assertIn("NOT like-for-like", dense)

    def test_a_dense_arm_still_declares_and_sweeps_both_families(self):
        # The joint restriction must not have narrowed the panel path: a dense arm
        # carries an attention output projection and its grid is unchanged.
        model = _dense()
        grid = model.components()
        self.assertEqual(
            STAGE15.families(grid), ("attention_head", model.block_kind)
        )
        self.assertEqual(len(grid), model.n_layers * (model.n_heads + 1))
        STAGE15.check_grid_is_ablatable(model, grid)


# ---------------------------------------------------------------- the panel path


class ThePanelArmPathIsUnchanged(unittest.TestCase):
    """Every panel invocation must mean exactly what it meant before."""

    def test_both_stages_still_default_to_progen3(self):
        for stage, extra in ((STAGE15, []), (STAGE17, ["--architecture", "plt"])):
            args = stage.build_parser().parse_args(extra)
            self.assertEqual(args.arm, R.PROGEN3_ARM)
            self.assertIsNone(args.joint_checkpoint)
            self.assertIsNone(args.rendering)
            self.assertIsNone(args.mode)
            stage.resolve_target(args)
            self.assertEqual(args.arm, R.PROGEN3_ARM, "a panel run keeps its arm")

    def test_the_eligible_arm_set_is_still_the_composed_one(self):
        from panel_contract import CAMPAIGN_PANEL

        admitted = R.eligible_arms(CAMPAIGN_PANEL)
        for stage, extra in ((STAGE15, []), (STAGE17, ["--architecture", "plt"])):
            for action in stage.build_parser()._actions:
                if action.dest == "arm":
                    self.assertEqual(list(action.choices), admitted)

    def test_a_panel_arm_keeps_its_declared_cohort_source_and_label(self):
        args = argparse.Namespace(
            arm="protgpt2", joint_checkpoint=None, rendering=None, mode=None
        )
        self.assertEqual(STAGE15.target_label(args), "protgpt2")
        self.assertEqual(
            STAGE15.cohort_source(args), R.arm_evaluation_cohort_source("protgpt2")
        )
        # And the name the trainer records for a panel arm is the --arm value, so
        # one check covers both kinds of target.
        self.assertEqual(_dense().name, "gpt2")

    def test_a_panel_arm_refuses_the_joint_flags(self):
        for stage, extra in ((STAGE15, []), (STAGE17, ["--architecture", "plt"])):
            for flag, value in (
                ("--rendering", "prollama"),
                ("--mode", "protein"),
                ("--protein-context", "Hydrolase"),
            ):
                args = stage.build_parser().parse_args(extra + [flag, value])
                with self.assertRaises(ValueError, msg=f"{stage.__name__} {flag}"):
                    stage.resolve_target(args)

    def test_an_arm_and_a_joint_checkpoint_are_mutually_exclusive(self):
        for stage, extra in ((STAGE15, []), (STAGE17, ["--architecture", "plt"])):
            with self.assertRaises(SystemExit):
                with contextlib.redirect_stderr(io.StringIO()):
                    stage.build_parser().parse_args(
                        extra + ["--arm", "gpt2-large", "--joint-checkpoint", "/nowhere"]
                    )

    def test_progen3s_own_relocation_flag_is_refused_beside_a_joint_checkpoint(self):
        for stage, extra in ((STAGE15, []), (STAGE17, ["--architecture", "plt"])):
            args = stage.build_parser().parse_args(
                extra
                + [
                    "--joint-checkpoint", "/nowhere",
                    "--rendering", "prollama",
                    "--mode", "text",
                    "--checkpoint", "/elsewhere",
                ]
                + (["--replacement-kind", "local"] if stage is STAGE15 else [])
            )
            with self.assertRaises(ValueError) as caught:
                stage.resolve_target(args)
            self.assertIn("--checkpoint", str(caught.exception))

    def test_an_unreadable_matched_against_fails_before_anything_is_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            absent = Path(directory) / "missing.json"
            args = STAGE15.build_parser().parse_args(
                ["--arm", "gpt2-large", "--matched-against", str(absent)]
            )
            with self.assertRaises(FileNotFoundError):
                STAGE15.resolve_target(args)
            unrelated = Path(directory) / "unrelated.json"
            unrelated.write_text(json.dumps({"schema_version": "x"}), encoding="utf-8")
            args.matched_against = unrelated
            with self.assertRaises(KeyError):
                STAGE15.resolve_target(args)

    def test_the_released_replacement_is_refused_for_a_joint_checkpoint(self):
        args = STAGE15.build_parser().parse_args(
            ["--joint-checkpoint", "/nowhere", "--rendering", "prollama", "--mode", "protein"]
        )
        with self.assertRaises(ValueError) as caught:
            STAGE15.resolve_target(args)
        self.assertIn("ProGen3", str(caught.exception))

    def test_the_trained_transcoder_loader_still_serves_the_older_checkpoints(self):
        # A checkpoint written before the matched declaration existed must keep
        # loading, and must report the absence rather than an invented pair.
        from src.transfer.transcoders import Transcoder, TranscoderConfig

        config = TranscoderConfig(num_layers=2, d_model=4, d_hidden=8, k=2, cross_layer=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.pt"
            torch.save(
                {"config": config.record(), "state_dict": Transcoder(config).state_dict()},
                path,
            )
            replacement, recorded, declared = load_trained_transcoder(path)
            self.assertEqual(recorded["d_hidden"], 8)
            self.assertIsNone(recorded["arm"])
            self.assertIsNone(declared)
            self.assertEqual(replacement.num_layers, 2)
            self.assertEqual(
                STAGE15.matched_pair_record(declared, None, kind="local")["verdict"],
                "WITHHELD",
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
