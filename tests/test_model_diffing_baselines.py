"""What the model-diffing baselines must get right, on CPU stubs.

The stage decides whether a Crosscoder on the ProLLaMA lineage would buy anything,
by asking whether a shift, a rotation or a full linear map already carries one
checkpoint's representation onto another's. That is only an answer if five things
hold, and every one of them can be wrong while the run still produces a complete
artefact full of finite, plausible numbers.

**The two models saw the same input.** A position-by-position residual is defined
only when record *r* tokenises to the same ids in both checkpoints. Two different
tokenizers, two different renderings, or two different token caps all produce
activations that align by index and correspond to nothing, and no downstream
number looks wrong. The tokenizer digest is a refusal, and the rendered strings,
the batch tensors and the content masks are compared on every batch.

**The held-out split is held out.** A map with ``d^2`` free parameters reported on
positions it was fitted on is not a measurement. The two splits are one seeded
pool -- so they are samples of one population rather than two regions of a
cluster-ordered corpus -- split over near-duplicate *groups* rather than over
records, because on a protein corpus the record is not the unit of independence:
41.4% of the held-out records of the Swiss-Prot pool this stage draws have a
relative at 95% identity or above while only 17.4% are exact, so a record-level
split with an exact-string check is not a held-out split. Whichever unit the
corpus is made of, a group may not straddle the two halves, exact content overlap
still refuses, and a pool that cannot be partitioned refuses rather than reporting
a fraction it did not achieve.

**The estimators are the estimators.** A perfect linear relation must come back at
a near-zero ``ridge`` residual and a large ``identity`` one, a perfect rotation at
a near-zero ``procrustes`` one, and a pure offset at a near-zero ``mean_shift``
one. Each of the four is a strictly larger class than the one before it, so a
defect in any of them shows up as an ordering that is not monotone.

**The null is a null.** With ``n`` tokens and ``d^2`` parameters a linear map fits
noise, so a good ``ridge`` number means nothing without the shuffled pairing beside
it. The shuffled control must degrade a genuinely correlated pair and must leave
the denominator and the target's norms exactly unmoved, because a permutation
changes only the pairing.

**The unit is inside the reference.** The adjacent-layer residual is what one
layer of ordinary computation costs, and it must not depend on the target
checkpoint at all -- if it did, the scale the cross-checkpoint number is read
against would move with the number.

The stubs are the ones ``tests/test_perturbation_sensitivity.py`` already builds --
a randomly initialised two-layer LLaMA briefly overfitted on its own corpus, and a
tokenizer that merges residues and spells ``Seq=<`` out of ordinary pieces -- and
they are imported rather than copied, because a second copy of a fixture is a
second definition of what the stub IS. One local subclass adds ``get_vocab``,
which that stub does not carry because no stage it exercises digests a vocabulary.
No GPU, no network, no 7B checkpoint.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
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

import numpy  # noqa: E402
import torch  # noqa: E402

from src.transfer import arms as A  # noqa: E402
from src.transfer import joint_modes as JM  # noqa: E402
from src.transfer import replaceable as R  # noqa: E402
from test_perturbation_sensitivity import (  # noqa: E402
    DOCUMENTS,
    SEQUENCES,
    StubJointTokenizer,
    _llama,
)


def _load_stage(filename: str):
    """Import a stage whose module name starts with a digit.

    Registered in ``sys.modules`` before execution, which is not optional here:
    the stage declares a module-level dataclass, and ``@dataclass`` resolves its
    annotations through ``sys.modules[cls.__module__]``.
    """

    path = REPO / "scripts/transfer" / filename
    spec = importlib.util.spec_from_file_location(f"_diffing_stage_{filename[:2]}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE = _load_stage("25_model_diffing_baselines.py")

D_MODEL = 8
N_LAYER = 2


class _Vocabulary(StubJointTokenizer):
    """The staged tokenizer's ``get_vocab``, which the shared stub does not carry.

    Added by subclassing rather than by copying the stub or by editing it: the
    digest that decides whether two checkpoints' positions are comparable reads the
    whole id-to-token map, and no stage the shared stub was built for does.
    """

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)


def _tokenizer(**kwargs) -> _Vocabulary:
    return _Vocabulary(**kwargs)


def _joint(model, tokenizer, mode: str, **kwargs) -> R.JointReplaceable:
    declaration = JM.rendering("prollama")
    settings = {"max_tokens": 128, "protein_context": None}
    settings.update(kwargs)
    return R.JointReplaceable(
        model=model,
        tokenizer=tokenizer,
        checkpoint=Path("/nowhere"),
        declaration=declaration,
        mode=mode,
        tokenisation=R.joint_tokenisation(tokenizer, declaration, mode),
        **settings,
    )


def _perturbed(model, *, scale: float = 0.05, seed: int = 3):
    """A second checkpoint of the same lineage: same shape, different weights."""

    other = copy.deepcopy(model)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in other.parameters():
            parameter.add_(
                torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype)
                * scale
            )
    return other


# ------------------------------------------------------ the estimators alone


def _estimate(
    train: list[tuple[torch.Tensor, torch.Tensor]],
    held_out: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    ridge: float = 1e-8,
    seed: int = 11,
) -> list[dict]:
    """The whole estimator path on prepared ``(layers, tokens, d)`` batches."""

    n_layers, _, d_model = train[0][0].shape
    moments = STAGE.PairedMoments(
        n_layers=n_layers, d_model=d_model, device=torch.device("cpu")
    )
    generator = torch.Generator().manual_seed(seed)
    for a, b in train:
        moments.update(a, b, torch.randperm(a.shape[1], generator=generator))
    maps = STAGE.fit_all(moments, ridge=ridge, progress=lambda _: None)
    evaluation = STAGE.HeldOut(n_layers=n_layers, maps=maps)
    generator = torch.Generator().manual_seed(seed + 1)
    for a, b in held_out:
        evaluation.update(a, b, torch.randperm(a.shape[1], generator=generator))
    return [evaluation.layer_record(layer) for layer in range(n_layers)]


def _activations(
    *, n_layers: int = N_LAYER, tokens: int = 512, d_model: int = D_MODEL, seed: int = 0
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randn(n_layers, tokens, d_model, generator=generator, dtype=torch.float64) * 3.0
        + 5.0
    )


def _linear_image(a: torch.Tensor, *, seed: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    """``b[l] = (a[l] - mean) @ W[l] + offset``: exactly affine, layer by layer."""

    generator = torch.Generator().manual_seed(seed)
    weights = torch.randn(
        a.shape[0], a.shape[2], a.shape[2], generator=generator, dtype=torch.float64
    )
    offset = torch.randn(a.shape[0], 1, a.shape[2], generator=generator, dtype=torch.float64)
    return torch.stack([a[i] @ weights[i] for i in range(a.shape[0])]) + offset, weights


class TheFourMapsAreTheFourMapsTheyClaimToBe(unittest.TestCase):
    """Each class is a strict superset of the one before it, so the order must hold."""

    def test_a_perfect_linear_relation_is_recovered_by_ridge_and_not_by_identity(self):
        train = [(_activations(seed=s), None) for s in range(4)]
        train = [(a, _linear_image(a)[0]) for a, _ in train]
        held = [(a, _linear_image(a)[0]) for a in (_activations(seed=90),)]
        record = _estimate(train, held)[0]["cross"]["true"]
        self.assertLess(record["ridge"]["normalised_residual"], 1e-6)
        # And the raw difference between the two is large, so the near-zero above
        # is the map's doing rather than the two sides being the same to begin with.
        self.assertGreater(record["identity"]["normalised_residual"], 0.5)
        self.assertGreater(
            record["ridge"]["identity_residual_removed"], 0.999,
            "ridge must remove essentially all of the identity residual here",
        )
        self.assertGreater(record["ridge"]["mean_cosine"], 0.999)

    def test_a_pure_offset_is_recovered_by_the_shift_and_a_rotation_by_procrustes(self):
        # mean_shift is the smallest class that can carry a constant, and
        # procrustes the smallest that can carry a rotation with a scale. Each has
        # to be reached by its own method and not only by ridge, or the sequence
        # of four says nothing about WHICH simple map explains the difference.
        shifted = [(a, a + 11.0) for a in (_activations(seed=s) for s in range(4))]
        record = _estimate(shifted, [(a, a + 11.0) for a in (_activations(seed=91),)])[0]
        self.assertLess(record["cross"]["true"]["mean_shift"]["normalised_residual"], 1e-12)
        self.assertGreater(record["cross"]["true"]["identity"]["normalised_residual"], 1.0)

        generator = torch.Generator().manual_seed(5)
        rotation, _ = torch.linalg.qr(
            torch.randn(D_MODEL, D_MODEL, generator=generator, dtype=torch.float64)
        )

        def rotate(a: torch.Tensor) -> torch.Tensor:
            return 2.5 * a @ rotation + 7.0

        rotated = [(a, rotate(a)) for a in (_activations(seed=s) for s in range(4))]
        turned = _estimate(rotated, [(a, rotate(a)) for a in (_activations(seed=92),)])[0]
        cell = turned["cross"]["true"]["procrustes"]
        self.assertLess(cell["normalised_residual"], 1e-6)
        self.assertAlmostEqual(cell["procrustes_scale"], 2.5, places=3)
        # ... while a shift alone cannot carry it.
        self.assertGreater(
            turned["cross"]["true"]["mean_shift"]["normalised_residual"], 0.5
        )

    def test_the_reported_denominator_is_the_held_out_variance_about_the_training_mean(self):
        # Appendix B rule 27: a normalised effect is a within-arm quantity unless
        # its denominator travels with it, so the denominator has to be both
        # present and correct.
        train = [(_activations(seed=s), _activations(seed=s + 50)) for s in range(4)]
        held = [(_activations(seed=93), _activations(seed=94))]
        record = _estimate(train, held)[0]
        training_mean = torch.cat([b[0] for _, b in train], dim=0).mean(0)
        expected = float(
            (held[0][1][0] - training_mean).pow(2).sum() / held[0][1][0].shape[0]
        )
        self.assertAlmostEqual(
            record["cross"]["denominator_per_position"], expected, places=6
        )
        self.assertGreater(record["cross"]["mean_target_norm"], 0.0)
        self.assertGreater(record["cross"]["mean_predictor_norm"], 0.0)


class TheShuffledPairingIsTheNull(unittest.TestCase):
    """Without it a good ridge number is indistinguishable from d^2 free parameters."""

    def _correlated(self, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        a = _activations(seed=seed)
        image, _ = _linear_image(a)
        noise = torch.randn(
            image.shape, generator=torch.Generator().manual_seed(seed + 400), dtype=torch.float64
        )
        return a, image + noise

    def test_the_shuffled_control_degrades_the_maps_that_are_fitted_to_the_pairing(self):
        train = [self._correlated(seed) for seed in range(6)]
        held = [self._correlated(95)]
        record = _estimate(train, held)[0]["cross"]
        true = record["true"]["ridge"]["normalised_residual"]
        null = record["shuffled"]["ridge"]["normalised_residual"]
        self.assertLess(true, 0.1, "the true pairing carries a real correspondence")
        self.assertGreater(null, 0.9, "the shuffled pairing must have nothing to fit")
        self.assertGreater(null - true, 0.8)
        # Procrustes is fitted to the pairing too, so it must degrade as well --
        # and identity and mean_shift are NOT, so nothing is asserted about them
        # here: neither reads the cross-moment, so on a pair related by a general
        # linear map their two pairings differ only by sampling noise and the sign
        # of that difference is not a property of anything.
        self.assertGreater(
            record["shuffled"]["procrustes"]["normalised_residual"],
            record["true"]["procrustes"]["normalised_residual"] + 0.2,
        )

    def test_on_a_near_identity_pair_the_null_degrades_every_method(self):
        # Which is the regime two checkpoints of one lineage are actually in: the
        # target's activation at a position is close to the reference's at the SAME
        # position, so even the unfitted maps lose when the pairing is destroyed.
        def near(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
            a = _activations(seed=seed)
            noise = torch.randn(
                a.shape,
                generator=torch.Generator().manual_seed(seed + 700),
                dtype=torch.float64,
            )
            return a, a + 0.25 * noise

        record = _estimate([near(seed) for seed in range(6)], [near(95)])[0]["cross"]
        for method in STAGE.METHODS:
            self.assertGreater(
                record["shuffled"][method]["normalised_residual"],
                record["true"][method]["normalised_residual"],
                method,
            )
        self.assertLess(record["true"]["identity"]["normalised_residual"], 0.05)

    def test_a_permutation_moves_the_pairing_and_nothing_else(self):
        # The denominator and the target's mean norm are sums over the multiset of
        # target rows, which a permutation does not change. If either moved, the
        # null would be measuring something besides the pairing.
        train = [self._correlated(seed) for seed in range(4)]
        held = [self._correlated(96)]
        layers = _estimate(train, held)
        for record in layers:
            for comparison in ("cross", "adjacent"):
                if comparison not in record:
                    continue
                self.assertGreater(record[comparison]["denominator_per_position"], 0.0)
                self.assertEqual(
                    record[comparison]["true"]["identity"]["mean_prediction_norm"],
                    record[comparison]["shuffled"]["identity"]["mean_prediction_norm"],
                    "the identity map's prediction is the reference, which the "
                    "permutation does not touch",
                )

    def test_a_fit_with_fewer_positions_than_parameters_is_refused(self):
        # n <= d makes ridge interpolate the training split and makes the shuffled
        # control read near zero for a trivial reason, which would invert the
        # reading of the whole artefact.
        moments = STAGE.PairedMoments(
            n_layers=N_LAYER, d_model=D_MODEL, device=torch.device("cpu")
        )
        a = _activations(tokens=D_MODEL - 2)
        moments.update(a, _linear_image(a)[0], torch.randperm(a.shape[1]))
        with self.assertRaises(RuntimeError) as caught:
            STAGE.fit_all(moments, ridge=1e-6, progress=lambda _: None)
        self.assertIn("underdetermined", str(caught.exception))


class TheAdjacentLayerUnitIsMeasuredInsideTheReference(unittest.TestCase):
    """The scale the cross-checkpoint number is read against must not move with it."""

    def _reference_with_an_affine_step(self, seed: int) -> torch.Tensor:
        first = _activations(n_layers=1, seed=seed)[0]
        step = torch.randn(
            D_MODEL, D_MODEL, generator=torch.Generator().manual_seed(21), dtype=torch.float64
        )
        return torch.stack([first, first @ step + 4.0])

    def test_the_adjacent_residual_reads_the_reference_and_ignores_the_target(self):
        train = [self._reference_with_an_affine_step(seed) for seed in range(4)]
        held = [self._reference_with_an_affine_step(97)]
        first = _estimate(
            [(a, _activations(seed=index + 300)) for index, a in enumerate(train)],
            [(held[0], _activations(seed=301))],
        )
        second = _estimate(
            [(a, _activations(seed=index + 600) * 17.0) for index, a in enumerate(train)],
            [(held[0], _activations(seed=601) * 17.0)],
        )
        # Layer 0 -> layer 1 of the reference is exactly affine, so ridge finds it.
        self.assertLess(first[0]["adjacent"]["true"]["ridge"]["normalised_residual"], 1e-6)
        # And a completely different target checkpoint leaves it bit for bit alone.
        self.assertEqual(first[0]["adjacent"], second[0]["adjacent"])
        # ... while the cross-checkpoint numbers did move, so the two runs really
        # did differ in their target.
        self.assertNotEqual(first[0]["cross"], second[0]["cross"])

    def test_the_last_layer_carries_no_adjacent_comparison(self):
        train = [(a, _linear_image(a)[0]) for a in (_activations(seed=s) for s in range(4))]
        layers = _estimate(train, [(_activations(seed=98), _activations(seed=99))])
        self.assertIn("adjacent", layers[0])
        self.assertNotIn("adjacent", layers[-1], "layer L-1 has no successor")
        summary = STAGE.summarise(layers)
        self.assertEqual(summary["cross"]["n_layers"], N_LAYER)
        self.assertEqual(summary["adjacent"]["n_layers"], N_LAYER - 1)

    def test_a_single_layer_model_is_refused_because_it_has_no_unit(self):
        with self.assertRaises(ValueError) as caught:
            STAGE.PairedMoments(n_layers=1, d_model=D_MODEL, device=torch.device("cpu"))
        self.assertIn("adjacent", str(caught.exception))


# --------------------------------------------------------------- the refusals


class TwoCheckpointsMustBeComparableBeforeAnythingIsMeasured(unittest.TestCase):
    def test_two_different_vocabularies_are_refused_by_digest(self):
        with self.assertRaises(ValueError) as caught:
            STAGE.assert_identical_tokenizers(
                _tokenizer(), _tokenizer(specials=("[START_AMINO]", "[END_AMINO]"))
            )
        self.assertIn("same vocabulary", str(caught.exception))

    def test_one_vocabulary_passes_and_records_the_digest_it_compared(self):
        record = STAGE.assert_identical_tokenizers(_tokenizer(), _tokenizer())
        self.assertEqual(record["verdict"], "IDENTICAL")
        self.assertEqual(len(record["vocabulary_sha256"]), 64)
        self.assertIn("24_component_swap", record["digest_source"])
        # The digest is the swap stage's, so the two stages cannot drift on what
        # "the same vocabulary" means.
        self.assertEqual(
            record["vocabulary_sha256"], STAGE.STAGE24.vocabulary_digest(_tokenizer())
        )

    def test_a_shape_disagreement_is_refused_before_any_activation_is_compared(self):
        tokenizer = _tokenizer()
        model = _llama(tokenizer)
        reference = _joint(model, tokenizer, "protein")
        target = _joint(_perturbed(model), tokenizer, "protein")
        facts = {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "n_layers": reference.n_layers,
            "d_model": reference.width,
            "vocab_size": len(tokenizer),
        }
        record = STAGE.assert_comparable_shape(
            reference, target, reference_facts=facts, target_facts=dict(facts)
        )
        self.assertEqual(record["verdict"], "COMPARABLE")
        self.assertEqual(record["d_model"], reference.width)

        for field, value in (("d_model", 4096), ("n_layers", 32), ("model_type", "opt")):
            wrong = dict(facts)
            wrong[field] = value
            with self.assertRaises(ValueError, msg=field) as caught:
                STAGE.assert_comparable_shape(
                    reference, target, reference_facts=facts, target_facts=wrong
                )
            self.assertIn(field, str(caught.exception))

    def test_a_handle_that_did_not_build_what_its_config_declared_is_refused(self):
        # L24's shape: the two configs agree and one of the two models is not what
        # it says. The config check cannot see it; the handle check can.
        tokenizer = _tokenizer()
        deep = _joint(_llama(tokenizer), tokenizer, "protein")
        shallow = _joint(_llama(tokenizer), tokenizer, "protein")
        shallow._layers = lambda: list(deep._layers())[:1]  # type: ignore[method-assign]
        facts = {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "n_layers": deep.n_layers,
            "d_model": deep.width,
            "vocab_size": len(tokenizer),
        }
        with self.assertRaises(ValueError) as caught:
            STAGE.assert_comparable_shape(
                deep, shallow, reference_facts=facts, target_facts=dict(facts)
            )
        self.assertIn("did not build what it declared", str(caught.exception))

    def test_the_declared_tensor_reaches_the_artefact_by_name(self):
        # "the block output" names a different tensor on different block layouts,
        # so the name and the layout's own declaration of it both have to be
        # recorded rather than left for a reader to infer from the stage name.
        tokenizer = _tokenizer()
        model = _joint(_llama(tokenizer), tokenizer, "protein")
        output = STAGE.tensor_declaration(model, "block_output")
        self.assertEqual(output["selected"], "block_output")
        self.assertIn("mlp", output["is"])
        self.assertEqual(output["block_layout"], model.perturbation_target)
        entry = STAGE.tensor_declaration(model, "block_input")
        self.assertIn("post_attention_layernorm", entry["is"])
        with self.assertRaises(ValueError):
            STAGE.tensor_declaration(model, "residual_stream")


class TheHeldOutSplitMustBeHeldOut(unittest.TestCase):
    """Records are realistic, because the property under test is about content.

    A fixture of ``record-0000``, ``record-0001``, ... would be one near-duplicate
    group -- the strings differ in one character -- so it could only ever exercise
    the refusal. Random sequences over the canonical alphabet are unrelated by
    construction, and the near-duplicates the tests need are planted explicitly so
    that what each test is about is visible in the fixture.
    """

    ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

    def _corpus(self, records):
        return lambda: iter((record, None) for record in records)

    def _sequences(self, n, *, length=120, seed=0):
        rng = numpy.random.default_rng(seed)
        return [
            "".join(self.ALPHABET[int(i)] for i in rng.integers(0, 20, size=length))
            for _ in range(n)
        ]

    def _mutate(self, sequence, *, seed):
        rng = numpy.random.default_rng(seed)
        residues = list(sequence)
        for position in rng.choice(len(residues), size=3, replace=False):
            residues[int(position)] = self.ALPHABET[int(rng.integers(0, 20))]
        return "".join(residues)

    def test_the_two_splits_are_one_pool_and_are_disjoint(self):
        records = self._sequences(64, seed=1)
        train, evaluation, splits = STAGE.draw_splits(
            self._corpus(records), n_train=40, n_eval=16, seed=5, skip=0,
            symbol_unit="residues",
        )
        self.assertEqual(splits["verdict"], "NEAR_DUPLICATE_DISJOINT")
        self.assertEqual((len(train), len(evaluation)), (40, 16))
        self.assertEqual(set(record for record, _ in train) & set(r for r, _ in evaluation), set())
        # Unrelated records: every group is a singleton, so the group split IS the
        # record split and the requested sizes are met exactly.
        self.assertEqual(splits["grouping"]["n_groups"], 56)
        self.assertEqual(splits["boundary_containment"]["n_above_threshold"], 0)
        # One pool: the union is exactly the drawn records, and neither half is a
        # contiguous region of the corpus's own file order -- which is the failure
        # a prefix draw would produce (Appendix B rule 1).
        position = {record: index for index, record in enumerate(records)}
        drawn = {record for record, _ in train} | {record for record, _ in evaluation}
        self.assertEqual(len(drawn), 56)
        for half in (train, evaluation):
            indices = sorted(position[record] for record, _ in half)
            self.assertLess(indices[0], 32)
            self.assertGreaterEqual(indices[-1], 32)

    def test_near_duplicates_are_kept_on_one_side_rather_than_straddling(self):
        # The property the whole change exists for. Under a record-level split
        # these four would be scattered across both halves at this seed; under a
        # group split they cannot be, and the boundary audit confirms it on the
        # returned mask rather than on the construction.
        records = self._sequences(60, seed=2)
        planted = records[0]
        for index in (10, 25, 41):
            records[index] = self._mutate(planted, seed=index)
        train, evaluation, splits = STAGE.draw_splits(
            self._corpus(records), n_train=40, n_eval=16, seed=5, skip=0,
            symbol_unit="residues",
        )
        family = {records[index] for index in (0, 10, 25, 41)}
        left = family & {record for record, _ in train}
        right = family & {record for record, _ in evaluation}
        self.assertTrue(bool(left) != bool(right), "the planted group straddles the split")
        self.assertEqual(splits["grouping"]["largest_group_size"], 4)
        self.assertEqual(splits["boundary_containment"]["n_above_threshold"], 0)
        self.assertEqual(splits["group_split"]["verdict"], "GROUP_DISJOINT")

    def test_a_pool_of_one_near_duplicate_group_is_refused(self):
        # The guard is the feature. A corpus band that yields one family has no
        # held-out split, and saying so is the correct outcome.
        with self.assertRaises(RuntimeError) as caught:
            STAGE.draw_splits(
                self._corpus([self._sequences(1, seed=3)[0]] * 64),
                n_train=40, n_eval=16, seed=5, skip=0, symbol_unit="residues",
            )
        self.assertIn("homology cluster", str(caught.exception))

    def test_a_pool_dominated_by_one_group_is_refused_rather_than_rebalanced(self):
        base = self._sequences(1, length=200, seed=4)[0]
        records = self._sequences(11, seed=5) + [
            self._mutate(base, seed=100 + index) for index in range(45)
        ]
        with self.assertRaises(RuntimeError) as caught:
            STAGE.draw_splits(
                self._corpus(records), n_train=40, n_eval=16, seed=5, skip=0,
                symbol_unit="residues",
            )
        self.assertIn("cannot be partitioned at the requested fraction", str(caught.exception))

    def test_a_corpus_too_small_for_the_declared_splits_is_refused(self):
        with self.assertRaises(RuntimeError) as caught:
            STAGE.draw_splits(
                self._corpus(self._sequences(10, seed=6)),
                n_train=40,
                n_eval=16,
                seed=5,
                skip=0,
                symbol_unit="residues",
            )
        self.assertIn("ran out", str(caught.exception))

    def test_a_text_pool_of_singletons_splits_exactly_as_a_record_draw_would(self):
        # Attainability before application: the gate this stage now applies to a
        # protein arm has to be attainable on the text control under the same
        # procedure. It is, and more than that -- with every group a singleton the
        # group split reproduces the record-level permutation split exactly, which
        # is what keeps the completed text-mode cell comparable with the protein
        # one instead of superseding it.
        documents = [
            f"document {index} " + " ".join(
                f"w{index}x{word}" for word in range(40)
            )
            for index in range(64)
        ]
        train, evaluation, splits = STAGE.draw_splits(
            self._corpus(documents), n_train=40, n_eval=16, seed=5, skip=0,
            symbol_unit="characters",
        )
        self.assertEqual(splits["grouping"]["n_groups"], 56)
        pool = list(STAGE.STAGE17.stream_records(self._corpus(documents), seed=5, skip=0, limit=56))
        order = numpy.random.default_rng(5 + 1).permutation(len(pool))
        self.assertEqual(train, [pool[int(index)] for index in order[:40]])
        self.assertEqual(evaluation, [pool[int(index)] for index in order[40:]])

    def test_the_skip_moves_the_pool_through_the_corpus(self):
        # The seed permutes WITHIN blocks read in file order, so only --skip can
        # produce the skip-offset sensitivity Appendix B rule 1 requires.
        records = self._sequences(200, seed=7)
        position = {record: index for index, record in enumerate(records)}
        head, held, _ = STAGE.draw_splits(
            self._corpus(records), n_train=40, n_eval=16, seed=5, skip=0,
            symbol_unit="residues",
        )
        tail, tail_held, facts = STAGE.draw_splits(
            self._corpus(records), n_train=40, n_eval=16, seed=5, skip=120,
            symbol_unit="residues",
        )
        self.assertEqual(facts["skip_records"], 120)
        # Everything past the skip and nothing before it: the pool is a different
        # region of the corpus, which is what a sensitivity re-run needs and what
        # the seed alone cannot produce.
        self.assertTrue(all(position[record] >= 120 for record, _ in tail + tail_held))
        self.assertTrue(any(position[record] < 120 for record, _ in head + held))


class BothCheckpointsMustReceiveTheSameInput(unittest.TestCase):
    """A residual by index is meaningless unless index i is the same token in both."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = _tokenizer()
        cls.model = _llama(cls.tokenizer)
        cls.other = _perturbed(cls.model)

    def _records(self, mode: str):
        source = SEQUENCES if mode == "protein" else DOCUMENTS
        return [(record, None) for record in source]

    def test_two_checkpoints_of_one_lineage_render_and_batch_identically(self):
        for mode in R.JOINT_MODES:
            reference = _joint(self.model, self.tokenizer, mode)
            target = _joint(self.other, self.tokenizer, mode)
            STAGE.assert_identical_batches(reference, target, self._records(mode))

    def test_a_different_protein_context_is_caught_even_at_one_tokenizer(self):
        reference = _joint(self.model, self.tokenizer, "protein")
        target = _joint(self.other, self.tokenizer, "protein", protein_context="Hydrolase")
        with self.assertRaises(RuntimeError) as caught:
            STAGE.assert_identical_batches(reference, target, self._records("protein"))
        self.assertIn("different strings", str(caught.exception))

    def test_a_different_token_cap_is_caught_even_at_one_rendering(self):
        reference = _joint(self.model, self.tokenizer, "text", max_tokens=128)
        target = _joint(self.other, self.tokenizer, "text", max_tokens=6)
        with self.assertRaises(RuntimeError) as caught:
            STAGE.assert_identical_batches(reference, target, self._records("text"))
        self.assertIn("input_ids", str(caught.exception))

    def test_the_paired_capture_keeps_the_modes_own_scored_positions(self):
        reference = _joint(self.model, self.tokenizer, "protein")
        target = _joint(self.other, self.tokenizer, "protein")
        a, b = STAGE.paired_capture(
            reference, target, self._records("protein"), tensor="block_output"
        )
        expected = sum(
            reference.tokenisation.render(sequence, context=None).n_scored_tokens
            for sequence in SEQUENCES
        )
        self.assertEqual(a.shape, (reference.n_layers, expected, reference.width))
        self.assertEqual(a.shape, b.shape)
        # Two different checkpoints on the same input really do produce different
        # activations, so a zero residual downstream would mean something.
        self.assertFalse(torch.equal(a, b))

    def test_the_two_declared_tensors_are_different_objects(self):
        reference = _joint(self.model, self.tokenizer, "protein")
        target = _joint(self.other, self.tokenizer, "protein")
        records = self._records("protein")
        entry, _ = STAGE.paired_capture(reference, target, records, tensor="block_input")
        exit_, _ = STAGE.paired_capture(reference, target, records, tensor="block_output")
        self.assertEqual(entry.shape, exit_.shape)
        self.assertFalse(torch.equal(entry, exit_))


# ------------------------------------------------------------- the whole stage


@contextlib.contextmanager
def _stage_on_stubs(tokenizer, checkpoints: dict[str, object], corpus: list[str]):
    """The stage with its loaders and its corpus replaced, and nothing else."""

    saved = (
        STAGE.STAGE21.load_tokenizer,
        STAGE.STAGE21.load_model,
        STAGE.corpus_location,
        STAGE.iter_corpus_records,
        R.checkpoint_weights_digest,
        sys.argv,
    )

    def load_model(resolved, tok, *, device, dtype):
        model = checkpoints[Path(resolved).name]
        return model, {
            "resolved_path": str(resolved),
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "n_layers": model.config.num_hidden_layers,
            "d_model": model.config.hidden_size,
            "n_heads": model.config.num_attention_heads,
            "vocab_size": len(tok),
            "dtype_requested": dtype,
            "dtype_observed": ["float32"],
            "device": device,
        }

    STAGE.STAGE21.load_tokenizer = lambda path: (Path(path), tokenizer)
    STAGE.STAGE21.load_model = load_model
    STAGE.corpus_location = lambda source, path=None: Path(f"/nowhere/{source}")
    STAGE.iter_corpus_records = lambda source, *, min_symbols, max_symbols=None, path=None: (
        iter((record, None) for record in corpus)
    )
    R.checkpoint_weights_digest = lambda path: hashlib.sha256(
        str(path).encode("utf-8")
    ).hexdigest()
    try:
        yield
    finally:
        (
            STAGE.STAGE21.load_tokenizer,
            STAGE.STAGE21.load_model,
            STAGE.corpus_location,
            STAGE.iter_corpus_records,
            R.checkpoint_weights_digest,
            sys.argv,
        ) = saved


def _numbers(node) -> list[float]:
    """Every number anywhere inside a nested artefact fragment."""

    if isinstance(node, dict):
        return [value for entry in node.values() for value in _numbers(entry)]
    if isinstance(node, list):
        return [value for entry in node for value in _numbers(entry)]
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        return []
    return [float(node)]


def _protein_corpus(count: int = 64) -> list[str]:
    """Unrelated canonical sequences, drawn rather than derived from four bases.

    This fixture used to cycle ``SEQUENCES`` and append a poly-residue tail, and
    its docstring said that made the records distinct "so the two splits cannot
    overlap by content". Every record was distinct and every sixteen of them were
    near-copies of one base sequence, which is exactly the inference
    :func:`STAGE.draw_splits` now refuses -- so the end-to-end run on this corpus
    would refuse too, and rightly. Distinctness is not independence, which is the
    property the whole stage rests on. Drawn from the canonical alphabet, which
    the stub tokenizer carries whole as singles and pairs, so no draw can grow its
    vocabulary past the stub model's embedding.
    """

    rng = numpy.random.default_rng(20260728)
    return [
        "".join(A.AA20[int(index)] for index in rng.integers(0, 20, size=33 + (position % 5)))
        for position in range(count)
    ]


def _text_corpus(count: int = 64) -> list[str]:
    """Documents over the stub's own word set, in independent draws.

    Same repair and same reason: appending ``in the Nth year`` to one of four
    documents produced records that share every word five-gram but one.
    """

    rng = numpy.random.default_rng(20260729)
    vocabulary = sorted({word for document in DOCUMENTS for word in document.split()})
    return [
        " ".join(str(word) for word in rng.choice(vocabulary, size=11)) for _ in range(count)
    ]


def _run(directory: Path, mode: str, tokenizer, checkpoints, corpus, **overrides) -> dict:
    settings = {
        "--train-records": "40",
        "--eval-records": "16",
        "--batch-size": "4",
        "--max-tokens": "128",
        "--ridge": "1e-6",
        "--reference": "/models/base",
        "--target": "/models/adapted",
    }
    settings.update(overrides)
    with _stage_on_stubs(tokenizer, checkpoints, corpus):
        sys.argv = [
            "25_model_diffing_baselines.py",
            "--rendering", "prollama",
            "--mode", mode,
            "--device", "cpu",
            "--out", str(directory),
        ] + [value for pair in settings.items() for value in pair]
        with contextlib.redirect_stdout(io.StringIO()):
            STAGE.main()
    name = STAGE.artefact_name(
        Path(settings["--reference"]),
        Path(settings["--target"]),
        mode,
        overrides.get("--tensor", "block_output"),
    )
    return json.loads((directory / name).read_text(encoding="utf-8"))


class TheStageRunsEndToEndOnTwoCheckpointsOfOneLineage(unittest.TestCase):
    """The whole driver, on stubs: the part no CPU host can otherwise reach."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = _tokenizer()
        base = _llama(cls.tokenizer)
        cls.checkpoints = {"base": base, "adapted": _perturbed(base)}

    def test_the_protein_run_reports_every_quantity_its_verdict_needs(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = _run(
                Path(directory), "protein", self.tokenizer, self.checkpoints, _protein_corpus()
            )
        self.assertEqual(payload["schema_version"], STAGE.SCHEMA_VERSION)
        self.assertEqual(payload["tokenizer_vocabulary"]["verdict"], "IDENTICAL")
        self.assertEqual(payload["comparability"]["verdict"], "COMPARABLE")
        self.assertEqual(payload["cohort"]["corpus_source"], "swissprot")
        self.assertEqual(payload["cohort"]["splits"]["verdict"], "NEAR_DUPLICATE_DISJOINT")
        # The measured boundary, not the construction: a reader of the artefact
        # has to be able to see how close the two sides came without re-running
        # anything, and on a protein cohort that is the number that decides
        # whether the held-out split was held out.
        boundary = payload["cohort"]["splits"]["boundary_containment"]
        self.assertEqual(boundary["n_above_threshold"], 0)
        self.assertEqual(payload["cohort"]["symbol_unit"], "residues")
        self.assertGreater(payload["cohort"]["n_train_positions"], payload["comparability"]["d_model"])
        self.assertEqual(payload["reference"]["role"], "reference")
        self.assertNotEqual(
            payload["reference"]["weights_sha256"], payload["target"]["weights_sha256"]
        )
        self.assertEqual(payload["reference"]["loader_gate"]["estimand"]["verdict"], "PASS")

        self.assertEqual(len(payload["layers"]), N_LAYER)
        for record in payload["layers"]:
            for comparison in ("cross", "adjacent"):
                if comparison not in record:
                    continue
                self.assertGreater(record[comparison]["denominator_per_position"], 0.0)
                for pairing in STAGE.PAIRINGS:
                    for method in STAGE.METHODS:
                        cell = record[comparison][pairing][method]
                        self.assertGreaterEqual(cell["normalised_residual"], 0.0)
                        self.assertIn("mean_cosine", cell)
                        self.assertIn("mean_prediction_norm", cell)
        self.assertIn("ridge_absolute", payload["layers"][0]["cross"]["true"]["ridge"])
        self.assertIn("procrustes_scale", payload["layers"][0]["cross"]["true"]["procrustes"])

        verdict = payload["verdict"]
        self.assertEqual(
            sorted(verdict["quantities"]),
            ["adjacent_layer_unit", "shuffled_null", "true_pairing"],
        )
        self.assertEqual(len(verdict["readings"]), 3)
        self.assertIn("NOT MADE HERE", verdict["decision"])
        self.assertIn("REPRESENTATIONAL ONLY", verdict["statement"])
        # The verdict declares no number of its own: outside the measured
        # quantities it is words only, so there is no cut for "a Crosscoder is
        # warranted" for a reader to mistake for a calibrated one (Appendix B
        # rule 17).
        self.assertEqual(
            _numbers({key: value for key, value in verdict.items() if key != "quantities"}),
            [],
            "the verdict must state no threshold of its own",
        )
        self.assertNotEqual(_numbers(verdict["quantities"]), [])

    def test_the_protein_run_carries_the_pre_adaptation_base_limitation_and_text_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protein = _run(root, "protein", self.tokenizer, self.checkpoints, _protein_corpus())
            text = _run(root, "text", self.tokenizer, self.checkpoints, _text_corpus())
        self.assertIn("protein_mode_reference", protein["limitations"])
        # 24_component_swap.py's own words, imported. The claim rests on the
        # REVERSAL COST -- indifference to reading direction -- and not on the
        # context-information floor, which that mode clears.
        self.assertIn(
            "REVERSAL COST", protein["limitations"]["protein_mode_reference"]
        )
        self.assertIn(
            "PRE-ADAPTATION REFERENCE",
            protein["limitations"]["protein_mode_reference"],
        )
        self.assertIn(
            "does not require a measurable behavioural estimand",
            protein["limitations"]["protein_mode_reference"],
        )
        self.assertNotIn("protein_mode_reference", text["limitations"])
        self.assertEqual(text["cohort"]["corpus_source"], "openwebtext")
        self.assertEqual(text["rendering"]["verdict"], "NOT_RESOLVED")
        self.assertEqual(protein["rendering"]["name"], "prollama")
        # Two modes into one directory must not overwrite each other.
        self.assertNotEqual(
            protein["settings"]["mode"], text["settings"]["mode"]
        )

    def test_a_checkpoint_compared_with_itself_leaves_no_residual_at_all(self):
        # The identity anchor for the whole path: same weights, same inputs, so
        # every cross-checkpoint residual under true pairing must be exactly zero
        # and every cosine exactly one. Anything else is a defect in the capture,
        # the pairing or the arithmetic rather than a finding.
        with tempfile.TemporaryDirectory() as directory:
            payload = _run(
                Path(directory),
                "protein",
                self.tokenizer,
                self.checkpoints,
                _protein_corpus(),
                **{"--target": "/models/base"},
            )
        self.assertEqual(
            payload["reference"]["weights_sha256"], payload["target"]["weights_sha256"]
        )
        for record in payload["layers"]:
            cell = record["cross"]["true"]["identity"]
            self.assertEqual(cell["normalised_residual"], 0.0)
            self.assertAlmostEqual(cell["mean_cosine"], 1.0, places=9)
            self.assertIsNone(cell["identity_residual_removed"])
            # ... and the shuffled null at the same layer is not zero, so the run
            # did compare something.
            self.assertGreater(
                record["cross"]["shuffled"]["identity"]["normalised_residual"], 0.0
            )

    def test_the_declared_tensor_selects_a_different_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = _run(
                root, "protein", self.tokenizer, self.checkpoints, _protein_corpus()
            )
            entry = _run(
                root,
                "protein",
                self.tokenizer,
                self.checkpoints,
                _protein_corpus(),
                **{"--tensor": "block_input"},
            )
        self.assertEqual(output["tensor"]["selected"], "block_output")
        self.assertEqual(entry["tensor"]["selected"], "block_input")
        self.assertNotEqual(
            output["layers"][0]["cross"]["true"]["identity"]["normalised_residual"],
            entry["layers"][0]["cross"]["true"]["identity"]["normalised_residual"],
        )

    def test_two_different_vocabularies_stop_the_run_before_the_weights_are_read(self):
        loaded: list[str] = []
        tokenizers = {"base": _tokenizer(), "adapted": _tokenizer(specials=("[X]",))}
        saved = STAGE.STAGE21.load_tokenizer
        with tempfile.TemporaryDirectory() as directory:
            try:
                with _stage_on_stubs(self.tokenizer, self.checkpoints, _protein_corpus()):
                    STAGE.STAGE21.load_tokenizer = lambda path: (
                        Path(path),
                        tokenizers[Path(path).name],
                    )
                    inner = STAGE.STAGE21.load_model

                    def watched(resolved, tok, *, device, dtype):
                        loaded.append(Path(resolved).name)
                        return inner(resolved, tok, device=device, dtype=dtype)

                    STAGE.STAGE21.load_model = watched
                    sys.argv = [
                        "25_model_diffing_baselines.py",
                        "--reference", "/models/base",
                        "--target", "/models/adapted",
                        "--rendering", "prollama",
                        "--mode", "protein",
                        "--device", "cpu",
                        "--out", str(directory),
                    ]
                    with self.assertRaises(ValueError) as caught:
                        with contextlib.redirect_stdout(io.StringIO()):
                            STAGE.main()
            finally:
                STAGE.STAGE21.load_tokenizer = saved
        self.assertIn("same vocabulary", str(caught.exception))
        self.assertEqual(loaded, [], "no weights may be read before the digests agree")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
