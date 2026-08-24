"""Conditions D3.j variant (a) must always hold, and its negative paths.

Written against EXP-R2-214's frozen text and against properties rather than the
implementation. Five are this programme's own lessons rather than hygiene:

* the arm-admission bar is a **measurement**, so it is tested by measuring a
  tokenizer that fails it and one that passes at 1.000, never by asserting an
  arm's name;
* an intervention that does not land passes every null, so the identity
  invariant is tested as exactly zero and the positive control as strictly
  moving;
* the reachability gate must **void** rather than return a negative, so the null
  world is checked to stop at the gate with nothing measured behind it;
* the standing margin reads the *ceiling's* Delta and not the arm's, which is
  one field name apart and produced a clause no positive effect could satisfy;
* the chemical axis must read no corpus, so BLOSUM62 is checked to be on the
  ceiling side and the descriptor table to be the one ``concept_lens`` declared.

``torch.set_num_threads(1)`` for the reason ``tests/test_concept_injection.py``
gives: the toy decoder is eleven-dimensional and thread launch dominates.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer import alphabet_chemistry as ac  # noqa: E402
from src.transfer.arms import AA20  # noqa: E402
from src.transfer.concept_lens import PROPERTY_BASIS  # noqa: E402
from src.transfer.kmer_background import load as load_background  # noqa: E402

torch.set_num_threads(1)

BACKGROUND = REPO_ROOT / "data/kmer_background/uniref50"
SEED = 20260819


def _load_stage():
    path = REPO_ROOT / "scripts/transfer/37_alphabet_chemistry.py"
    spec = importlib.util.spec_from_file_location("_stage_37", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE = _load_stage()


# ------------------------------------------------------------------ fixtures


class _StubTokenizer:
    """A tokenizer over a declared piece list, longest piece first.

    Not a mock of the code under test: the quantity being tested is a property of
    a *vocabulary*, and the cheapest honest way to vary a vocabulary is to write
    one down. The greedy longest-match rule is what a BPE does to a residue run.
    """

    def __init__(self, pieces: list[str]) -> None:
        self.pieces = list(pieces)
        self._order = sorted(range(len(pieces)), key=lambda i: -len(pieces[i]))

    def __len__(self) -> int:
        return len(self.pieces)

    def __call__(self, text: str, return_tensors=None):
        ids: list[int] = []
        position = 0
        while position < len(text):
            for index in self._order:
                piece = self.pieces[index]
                if text.startswith(piece, position):
                    ids.append(index)
                    position += len(piece)
                    break
            else:
                raise ValueError(f"cannot tokenise {text[position]!r}")
        return {"input_ids": ids}

    def convert_ids_to_tokens(self, token_id: int):
        return self.pieces[token_id]

    def convert_tokens_to_string(self, pieces):
        return "".join(pieces)


class _StubSpec:
    def __init__(self, tokenisation: str, modality: str) -> None:
        self.tokenisation = tokenisation
        self.modality = modality
        self.name = "stub"


class _StubConfig:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size


class _StubModel:
    def __init__(self, vocab_size: int) -> None:
        self.config = _StubConfig(vocab_size)


class _StubArm:
    def __init__(self, pieces: list[str], *, tokenisation: str, modality: str = "protein") -> None:
        self.tokenizer = _StubTokenizer(pieces)
        self.model = _StubModel(len(pieces))
        self.spec = _StubSpec(tokenisation, modality)
        self.name = "stub-arm"
        self.modality = modality


def _residue_arm() -> _StubArm:
    return _StubArm(list(AA20), tokenisation="residue")


def _bpe_arm() -> _StubArm:
    """Twenty single-residue pieces AND longer merges, as a real BPE has."""

    merges = [first + second for first in AA20 for second in AA20]
    return _StubArm(list(AA20) + merges, tokenisation="multi_residue_bpe")


def _cohort_texts() -> list[str]:
    rng = np.random.default_rng(11)
    # Odd lengths on purpose: under the all-2-mer vocabulary below every pair
    # merges and only the trailing residue survives as its own token, which is
    # the small-but-non-zero coverage a real multi-residue BPE returns.
    return ["".join(rng.choice(list(AA20), size=241)) for _ in range(6)]


# ------------------------------------------- D3.j-A0, admission as a measurement


def test_a_symbol_tokenised_arm_reaches_the_admission_bar_by_construction():
    arm = _residue_arm()
    alphabet = ac.protein_alphabet(arm)
    coverage = ac.symbol_token_coverage(arm, _cohort_texts(), alphabet=alphabet, max_len=256)
    assert coverage["coverage"] == 1.0
    verdict = ac.admit_arm(coverage, arm.name, minimum=ac.MINIMUM_SYMBOL_TOKEN_COVERAGE)
    assert verdict["admitted"] is True


def test_a_multi_residue_bpe_arm_is_refused_on_its_measured_coverage():
    """The refusal is the number, not the arm's name.

    The vocabulary below contains all twenty single-residue pieces, so an
    implementation that resolved the alphabet by looking for them would find
    twenty rows and swap them. What decides is how much of the *scored* text
    those rows actually carry.
    """

    arm = _bpe_arm()
    alphabet = ac.protein_alphabet(arm)
    assert len(alphabet) == 20
    coverage = ac.symbol_token_coverage(arm, _cohort_texts(), alphabet=alphabet, max_len=256)
    assert 0.0 < coverage["coverage"] < ac.MINIMUM_SYMBOL_TOKEN_COVERAGE
    verdict = ac.admit_arm(coverage, arm.name, minimum=ac.MINIMUM_SYMBOL_TOKEN_COVERAGE)
    assert verdict["admitted"] is False
    assert f"{coverage['coverage']:.4f}" in verdict["reason"]
    assert "NOT MEASURABLE" in verdict["reason"]


def test_an_alphabet_whose_symbol_has_two_tokens_is_refused_rather_than_guessed():
    arm = _StubArm(list(AA20) + ["A"], tokenisation="residue")
    with pytest.raises(ValueError, match="exactly one token"):
        ac.protein_alphabet(arm)


# ------------------------------------------------ the two axes and their sides


def test_the_chemical_axis_is_the_declared_descriptor_set_and_reads_no_corpus():
    table = ac.chemical_property_table(AA20)
    assert set(table) == set(PROPERTY_BASIS) | {"polarity"}
    for name, values in PROPERTY_BASIS.items():
        assert table[name] == [float(values[residue]) for residue in AA20]
    assert sorted(ac.GRANTHAM_POLARITY) == sorted(AA20)


def test_a_constant_descriptor_is_dropped_and_named_rather_than_scaled_in():
    table = ac.chemical_property_table(AA20)
    table["flat"] = [1.0] * 20
    axis = ac.property_distance(table, source="test")
    assert axis.properties_dropped == ("flat",)
    reference = ac.property_distance(ac.chemical_property_table(AA20), source="test")
    assert np.allclose(axis.distance, reference.distance)


def test_blosum62_is_symmetric_with_the_published_diagonal_and_a_zero_self_distance():
    scores = np.asarray(ac.BLOSUM62_ROWS, dtype=float)
    assert scores.shape == (20, 20)
    assert np.array_equal(scores, scores.T)
    published = {"W": 11, "C": 9, "H": 8, "P": 7, "Y": 7, "A": 4, "L": 4}
    for residue, value in published.items():
        index = ac.BLOSUM62_ORDER.index(residue)
        assert scores[index, index] == value
    distance = ac.blosum62_distance(AA20)
    assert np.allclose(np.diag(distance), 0.0)
    assert (distance >= 0).all()
    assert distance[AA20.index("I"), AA20.index("L")] < distance[AA20.index("I"), AA20.index("D")]


def test_the_context_profile_is_normalised_and_a_symbol_without_context_is_refused():
    counts = np.array([[1.0, 3.0], [2.0, 2.0]])
    profiles = ac.context_profiles(counts)
    assert np.allclose(profiles.sum(axis=1), 1.0)
    with pytest.raises(ValueError, match="no context profile"):
        ac.context_profiles(np.array([[1.0, 1.0], [0.0, 0.0]]))


def test_symmetric_kl_is_withheld_rather_than_smoothed_where_a_cell_is_empty():
    dense = ac.context_profiles(np.array([[1.0, 3.0], [2.0, 2.0]]))
    assert ac.symmetric_kl_distance(dense) is not None
    sparse = ac.context_profiles(np.array([[1.0, 0.0], [2.0, 2.0]]))
    assert ac.symmetric_kl_distance(sparse) is None


# ------------------------------------ the contradiction set and its attainability


def _independent_axes(size: int = 20, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    chemical = rng.random((size, size))
    distributional = rng.random((size, size))
    for matrix in (chemical, distributional):
        matrix += matrix.T
        np.fill_diagonal(matrix, 0.0)
    return chemical, distributional


def test_every_admitted_pair_is_opposite_signed_on_the_two_axes():
    """The whole design: a pair is in only if the axes disagree about it."""

    chemical, distributional = _independent_axes()
    for cut in ac.CUTS:
        record = ac.quadrants_at_cut(chemical, distributional, cut=cut)
        rows, columns = np.triu_indices(chemical.shape[0], 1)
        chem_low, chem_high, _ = ac._rank_bands(chemical[rows, columns], ac.CUTS[cut])
        dist_low, dist_high, _ = ac._rank_bands(distributional[rows, columns], ac.CUTS[cut])
        lookup = {(int(rows[i]), int(columns[i])): i for i in range(rows.size)}
        for x, y in record["members"][ac.QUADRANTS[0]]:
            index = lookup[(x, y)]
            assert chem_low[index] and dist_high[index]
        for x, y in record["members"][ac.QUADRANTS[1]]:
            index = lookup[(x, y)]
            assert chem_high[index] and dist_low[index]
        assert not (
            set(map(tuple, record["members"][ac.QUADRANTS[0]]))
            & set(map(tuple, record["members"][ac.QUADRANTS[1]]))
        )


def test_a_straddling_tie_block_enters_a_band_that_a_value_threshold_would_empty():
    """A strict value comparison inside a tie block excludes the whole block.

    Which reads exactly like an alphabet with no contradiction set, so the bands
    are taken on average ranks, which assign a straddling block by its centre.
    """

    values = np.array([0.1, 0.2, 0.3, 0.4] + [1.0] * 8)
    _, high, record = ac._rank_bands(values, 1.0 / 3.0)
    assert (values > np.quantile(values, 2.0 / 3.0)).sum() == 0
    assert high.sum() == 8
    assert record["n_tied_values"] == 7


def test_a_tie_block_that_cannot_be_split_is_reported_rather_than_hidden():
    values = np.array([0.1, 0.2] + [1.0] * 10)
    _, high, record = ac._rank_bands(values, 1.0 / 3.0)
    assert high.sum() == 0
    assert record["n_tied_values"] == 9
    assert record["n_distinct_values"] == 3


def test_a_cut_that_cannot_fill_its_quadrants_is_reported_and_not_raised():
    size = 8
    chemical = np.zeros((size, size))
    distributional = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i != j:
                chemical[i, j] = abs(i - j)
                distributional[i, j] = abs(i - j)
    sweep = ac.cut_sweep(chemical, distributional)
    assert sweep["readable_cuts"] == []
    assert all(record["readable"] is False for record in sweep["per_cut"].values())


@pytest.mark.skipif(not BACKGROUND.is_dir(), reason="the staged k-mer background is absent")
def test_the_staged_alphabet_and_corpus_admit_a_contradiction_set_at_the_declared_cuts():
    """D3.j-A3 on the real inputs: whether the set exists is a measurement."""

    background = load_background(BACKGROUND)
    chemical = ac.property_distance(
        ac.chemical_property_table(AA20), source=ac.CHEMICAL_AXIS_SOURCE
    ).distance
    profiles = ac.context_profiles(ac.residue_context_counts(background, AA20))
    distributional = ac.cosine_distance(profiles)
    sweep = ac.cut_sweep(chemical, distributional)
    assert sweep["n_unordered_pairs"] == 190
    assert 0.0 < sweep["axis_spearman"] < 0.5
    assert "tercile" in sweep["readable_cuts"]
    # The two statistics estimators are not one quantity, which is why BLOSUM62
    # is on the ceiling side rather than standing in for either axis.
    rows, columns = np.triu_indices(20, 1)
    blosum = ac.blosum62_distance(AA20)[rows, columns]
    from scipy import stats

    assert abs(stats.spearmanr(blosum, distributional[rows, columns]).statistic) < 0.1
    assert stats.spearmanr(blosum, chemical[rows, columns]).statistic > 0.2


# --------------------------------------------------- the write and its invariants


def _world(planted: str = "chemistry", **kwargs):
    return ac.synthetic_world(planted=planted, seed=SEED, sequences=24, length=128, **kwargs)


def test_substituting_a_symbols_own_row_does_exactly_zero_damage():
    """The no-op invariant, exactly and not approximately.

    Both passes see identical weights, identical rows and identical chunking, so
    any difference at all is a defect in the write or the scorer, and a tolerance
    here would be a tolerance on correctness.
    """

    world = _world()
    scorer = ac.DamageScorer(world.model, world.cohort(), batch_size=8)
    for token in (0, 5, 11):
        record = scorer.damage(token, token)
        assert record["measurable"]
        assert record["nats_per_scored_token"] == 0.0


def test_the_positive_control_moves_and_the_row_is_restored_afterwards():
    world = _world()
    scorer = ac.DamageScorer(world.model, world.cohort(), batch_size=8)
    before = world.model.weight.data.clone()
    record = ac.intervention_invariants(
        scorer,
        symbol_token=0,
        alphabet_tokens=[symbol.token_id for symbol in world.symbols],
        seed=SEED,
        tolerance=1e-6,
    )
    assert record["identity_damage_nats"] == 0.0
    assert abs(record["random_replacement_damage_nats"]) > 1e-6
    assert record["constant_over_random_offset_ratio"] is not None
    assert torch.equal(world.model.weight.data, before)


def test_a_write_that_does_not_reach_the_forward_pass_is_refused():
    """A null-only check cannot see an unbound write, which is why this exists."""

    world = _world()

    class _DeafModel:
        """Reads a private copy of the embedding, so a write never reaches it."""

        def __init__(self, inner):
            self._weight = inner.weight
            self._frozen = inner.weight.detach().clone()
            self._head = inner._head

        @property
        def weight(self):
            return self._weight

        def logits(self, input_ids, attention_mask):
            del attention_mask
            return self._frozen[input_ids] @ self._head.T

    scorer = ac.DamageScorer(_DeafModel(world.model), world.cohort(), batch_size=8)
    with pytest.raises(RuntimeError, match="not reaching the forward pass"):
        ac.intervention_invariants(
            scorer,
            symbol_token=0,
            alphabet_tokens=[symbol.token_id for symbol in world.symbols],
            seed=SEED,
            tolerance=1e-6,
        )


def test_substituted_row_restores_the_row_even_when_the_block_raises():
    weight = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    original = weight.clone()
    with pytest.raises(ZeroDivisionError):
        with ac.substituted_row(weight, 2, torch.zeros(3)):
            assert not torch.equal(weight, original)
            raise ZeroDivisionError
    assert torch.equal(weight, original)


def test_the_pairs_own_two_identities_are_excluded_from_the_scored_positions():
    world = _world()
    scorer = ac.DamageScorer(world.model, world.cohort(), batch_size=8)
    record = scorer.damage(0, 1)
    assert record["n_scored_tokens"] < record["n_context_positions"]
    assert record["excluded_by_pair_identity"] == (
        record["n_context_positions"] - record["n_scored_tokens"]
    )


# ------------------------------------------------------- the ceiling and the margin


HIGH_ORDER = REPO_ROOT / "data/kmer_background/uniref50_high_order"


def _records(n: int = 4, size: int = 300):
    rng = np.random.default_rng(5)
    return ["".join(rng.choice(list(AA20), size=size)) for _ in range(n)]


@pytest.mark.skipif(not HIGH_ORDER.is_dir(), reason="the staged k-mer background is absent")
def test_the_fragment_ceiling_is_the_identity_on_a_self_substitution_and_moves_otherwise():
    ordered = ac.load_ordered_counts(HIGH_ORDER, [3], pinned=BACKGROUND)
    ceiling = ac.FragmentConditional(ordered[3])
    records = _records()
    assert ceiling.damage(records, "A", "A")["nats_per_scored_token"] == 0.0
    moved = ceiling.damage(records, "A", "W")
    assert moved["measurable"] and moved["n_scored_tokens"] > 0
    assert moved["nats_per_scored_token"] != 0.0


@pytest.mark.skipif(not HIGH_ORDER.is_dir(), reason="the staged k-mer background is absent")
def test_the_unigram_ceiling_is_exactly_zero_because_it_reads_no_context():
    """The curve's own reachability anchor (rule 40 applied to the null).

    A k = 1 conditional cannot see the symbol that was substituted, so its Delta
    is zero by construction. A curve whose first point is not exactly zero is a
    defect in the k-gram indexing rather than a fact about the corpus, and every
    higher order shares that indexing.
    """

    ordered = ac.load_ordered_counts(HIGH_ORDER, [1], pinned=BACKGROUND)
    ceiling = ac.FragmentConditional(ordered[1])
    records = _records()
    for substitute in ("W", "G", "D"):
        record = ceiling.damage(records, "A", substitute)
        assert record["measurable"]
        assert record["nats_per_scored_token"] == 0.0


def _sampled_records(ordered, order, *, n, size, seed):
    """Sequences drawn from the conditional itself, so high-order context is real."""

    rng = np.random.default_rng(seed)
    counts = ordered.counts
    out = []
    for _ in range(n):
        codes = list(rng.integers(0, 20, size=order - 1))
        while len(codes) < size:
            context = 0
            for value in codes[-(order - 1):]:
                context = context * 20 + int(value)
            row = np.asarray(counts[context * 20 : context * 20 + 20], dtype=np.float64)
            total = row.sum()
            codes.append(int(rng.integers(0, 20)) if total <= 0 else int(rng.choice(20, p=row / total)))
        out.append("".join(AA20[value] for value in codes))
    return out


@pytest.mark.skipif(not HIGH_ORDER.is_dir(), reason="the staged k-mer background is absent")
def test_the_ceiling_rises_with_order_only_where_the_sequence_carries_long_context():
    """The amendment's premise, and the control that stops it being circular.

    On sequence sampled from the k = 6 conditional the ceiling's damage rises with
    order, because the context genuinely predicts. On uniform random residues it
    does not, which is what says the rise measured on real protein is a property
    of protein and not of sparse counts at high order inflating noise.
    """

    ordered = ac.load_ordered_counts(HIGH_ORDER, [1, 2, 4, 6], pinned=BACKGROUND)
    pairs = [("A", "W"), ("D", "L"), ("G", "P"), ("K", "E")]

    def mean_damage(records, order):
        model = ac.FragmentConditional(ordered[order])
        return float(
            np.mean(
                [abs(model.damage(records, a, b)["nats_per_scored_token"]) for a, b in pairs]
            )
        )

    structured = _sampled_records(ordered[6], 6, n=4, size=400, seed=11)
    rising = [mean_damage(structured, order) for order in (2, 4, 6)]
    assert rising[0] < rising[1] < rising[2], rising
    assert mean_damage(structured, 1) == 0.0

    flat = _records(n=4, size=400)
    unstructured = [mean_damage(flat, order) for order in (2, 4, 6)]
    assert max(unstructured) < rising[-1], (unstructured, rising)


@pytest.mark.skipif(not HIGH_ORDER.is_dir(), reason="the staged k-mer background is absent")
def test_the_curve_must_extend_the_pinned_background_and_not_replace_it(tmp_path):
    """A curve computed against a second opinion about the corpus is refused."""

    import json
    import shutil

    forged = tmp_path / "forged"
    forged.mkdir()
    manifest = json.loads((HIGH_ORDER / "manifest.json").read_text())
    manifest["sha256"]["3"] = "0" * 64
    (forged / "manifest.json").write_text(json.dumps(manifest))
    shutil.copy(HIGH_ORDER / "kmer_counts_k3.npy", forged / "kmer_counts_k3.npy")
    with pytest.raises(ValueError, match="second opinion about the corpus"):
        ac.load_ordered_counts(forged, [3], pinned=BACKGROUND)


@pytest.mark.skipif(not HIGH_ORDER.is_dir(), reason="the staged k-mer background is absent")
def test_an_even_order_context_profile_is_refused_rather_than_split_arbitrarily():
    ordered = ac.load_ordered_counts(HIGH_ORDER, [4], pinned=BACKGROUND)
    with pytest.raises(ValueError, match="odd orders"):
        ac.residue_context_profiles_at_order(ordered[4])


def test_the_standing_margin_reads_the_ceilings_delta_and_not_the_arms():
    """One field name apart, and the wrong one fails every positive effect.

    ``delta_contrast`` scores the arm and the ceiling on one resampled set of
    rows, so both live on the same block and ``delta`` there is the ARM's. A
    margin built on it reads ``delta >= factor * delta``.
    """

    delta_block = {"delta": 0.5}
    ceiling_block = {"delta": 0.5, "reference_delta": 0.01, "difference": 0.49,
                     "difference_ci95": [0.2, 0.8]}
    random_null = {"observed_above_q95": True}
    margin = ac.ceiling_margin(
        delta_block=delta_block, ceiling_block=ceiling_block,
        random_null=random_null, factor=2.0,
    )
    assert margin["ceiling_delta"] == 0.01
    assert margin["clauses"]["at_least_factor_times_ceiling"] is True
    assert margin["cleared"] is True


def test_a_negative_ceiling_is_clamped_at_zero_rather_than_doubled():
    margin = ac.ceiling_margin(
        delta_block={"delta": 0.5},
        ceiling_block={"delta": 0.5, "reference_delta": -3.0, "difference": 3.5,
                       "difference_ci95": [3.0, 4.0]},
        random_null={"observed_above_q95": True},
        factor=2.0,
    )
    assert margin["clauses"]["at_least_factor_times_ceiling"] is True
    margin = ac.ceiling_margin(
        delta_block={"delta": 0.5},
        ceiling_block={"delta": 0.5, "reference_delta": 0.4, "difference": 0.1,
                       "difference_ci95": [0.05, 0.2]},
        random_null={"observed_above_q95": True},
        factor=2.0,
    )
    assert margin["clauses"]["at_least_factor_times_ceiling"] is False
    assert margin["cleared"] is False


def test_a_flat_ceiling_is_reported_as_inadequate_rather_than_silently_cleared():
    adequacy = ac.ceiling_adequacy([1.0, 0.8, 1.2], [0.01, 0.02, 0.0], floor=0.1)
    assert adequacy["adequate"] is False
    assert "little weight" in adequacy["reading"]
    assert ac.ceiling_adequacy([1.0, 0.8], [0.5, 0.4], floor=0.1)["adequate"] is True


def test_the_random_direction_control_needs_directions_and_not_positions():
    codes = np.array([1, 1, -1, -1])
    with pytest.raises(ValueError, match="rule 39"):
        ac.random_direction_delta_null(
            codes=codes, per_direction=[[1.0, 1.0, 1.0, 1.0]] * 4, observed=0.0
        )


# ------------------------------------------------------------- the unit floor


def test_a_pair_set_below_the_unit_floor_is_published_without_an_interval():
    codes = np.array([1, 1, 1, -1, -1, -1])
    groups = np.array([0, 0, 1, 1, 2, 2])
    block = ac.delta_contrast(
        codes=codes, damage=[1.0, 1.1, 0.9, 0.2, 0.3, 0.1], groups=groups,
        seed=SEED, n_bootstrap=100,
    )
    assert block["unit_floor"]["degenerate"] is True
    assert block["difference_ci95"] is None
    assert "8-unit floor" in block["interval_withheld_reason"]


def test_the_floor_is_the_shared_declaration_and_not_a_local_number():
    from src.transfer.statistics import MINIMUM_BOOTSTRAP_UNITS

    assert ac.MINIMUM_QUADRANT_PAIRS == MINIMUM_BOOTSTRAP_UNITS


def test_this_module_declares_no_resampler_of_its_own():
    """Appendix B rule 12 for resampling: one implementation, imported."""

    import inspect

    local = [
        name
        for name, member in vars(ac).items()
        if inspect.isfunction(member)
        and member.__module__ == ac.__name__
        and ("bootstrap" in name.lower() or "resampl" in name.lower())
    ]
    assert local == []


# ------------------------------------------ reachability voids rather than negates


def test_the_reachability_gate_voids_and_does_not_return_a_negative():
    failed = ac.reachability_verdict(
        [0.1, 0.1, 0.4, 0.5],
        [ac.AGREEMENT_CLASSES[0]] * 2 + [ac.AGREEMENT_CLASSES[1]] * 2,
        margin=0.05,
    )
    assert failed["reachable"] is False
    assert "VOID" in failed["consequence_if_failed"]
    passed = ac.reachability_verdict(
        [0.6, 0.7, 0.1, 0.2],
        [ac.AGREEMENT_CLASSES[0]] * 2 + [ac.AGREEMENT_CLASSES[1]] * 2,
        margin=0.05,
    )
    assert passed["reachable"] is True


def test_the_null_world_stops_at_the_reachability_gate_with_nothing_behind_it():
    """A decoder reading a block neither axis measures must not be read at all."""

    world = ac.synthetic_world(planted="neither", seed=SEED)
    chemical = ac.property_distance(world.property_table, source="test").distance
    distributional = ac.cosine_distance(ac.context_profiles(world.context_counts()))
    pairs, _ = ac.agreement_extremes(chemical, distributional, cut="tercile", count=4)
    scorer = ac.DamageScorer(world.model, world.cohort(), batch_size=16)
    damages = [
        scorer.damage(world.symbols[x].token_id, world.symbols[y].token_id)[
            "nats_per_scored_token"
        ]
        for x, y in pairs.pairs
    ]
    assert ac.reachability_verdict(damages, pairs.classes, margin=0.0)["reachable"] is False


# ------------------------------------------------- the known-answer certificate


@pytest.mark.parametrize("planted", ["chemistry", "distribution"])
def test_the_planted_world_is_recovered_and_the_shuffled_null_is_not_the_effect(planted):
    world = ac.synthetic_world(planted=planted, seed=SEED)
    chemical = ac.property_distance(world.property_table, source="test").distance
    distributional = ac.cosine_distance(ac.context_profiles(world.context_counts()))
    embedding = ac.embedding_distance(
        world.model.weight, [symbol.token_id for symbol in world.symbols]
    )
    quadrants = ac.quadrants_at_cut(chemical, distributional, cut="tercile")
    assert quadrants["readable"]
    pairs = ac.ordered_pair_set(quadrants)
    scorer = ac.DamageScorer(world.model, world.cohort(), batch_size=32)
    damages = [
        scorer.damage(world.symbols[x].token_id, world.symbols[y].token_id)[
            "nats_per_scored_token"
        ]
        for x, y in pairs.pairs
    ]
    codes = pairs.codes(ac.QUADRANTS)
    delta = ac.delta_contrast(
        codes=codes, damage=damages, groups=pairs.groups, seed=SEED, n_bootstrap=300
    )
    expected = 1.0 if planted == "chemistry" else -1.0
    assert np.sign(delta["delta"]) == expected
    assert np.sign(delta["difference_ci95"][0]) == np.sign(delta["difference_ci95"][1]) == expected
    controlled = ac.association(
        damage=damages,
        chemical=[chemical[x, y] for x, y in pairs.pairs],
        distributional=[distributional[x, y] for x, y in pairs.pairs],
        embedding=[embedding[x, y] for x, y in pairs.pairs],
        groups=pairs.groups, seed=SEED, n_bootstrap=300,
    )["embedding_distance_controlled"]
    low, high = controlled["difference_ci95"]
    assert np.sign(low) == np.sign(high) == expected
    null = ac.shuffled_difference_null(
        damage=damages, codes=codes, groups=pairs.groups, seed=SEED, draws=200
    )
    assert null["observed_outside_null_q95"] is True


# ---------------------------------------------------------------- stage contract


def _args(**overrides) -> argparse.Namespace:
    parser = STAGE.build_parser()
    args = parser.parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_the_ceiling_curve_must_retain_the_pre_registered_order():
    args = _args(
        arm="progen2-small", cut="tercile", max_pairs=40, null_draws=200, seed=1,
        reachability_pairs=4, reachability_margin=0.0, random_directions=8,
        ceiling_factor=2.0, records=8, max_tokens=64, min_symbol_occurrences=10,
        kmer_background=Path("x"), high_order_background=Path("y"),
        text_control=Path("z"), ceiling_orders="5,6,7",
    )
    with pytest.raises(ValueError, match="the rung EXP-R2-214 froze"):
        STAGE.resolve(args)


def test_an_even_axis_correlation_order_is_refused_before_anything_is_read():
    args = _args(
        synthetic=True, cut="tercile", max_pairs=40, null_draws=200, seed=1,
        reachability_pairs=4, reachability_margin=0.0, random_directions=8,
        axis_correlation_orders="3,4",
    )
    with pytest.raises(ValueError, match="symmetric"):
        STAGE.resolve(args)


def test_the_stage_names_every_missing_pre_registered_decision():
    with pytest.raises(ValueError) as error:
        STAGE.resolve(_args(arm="progen2-small"))
    message = str(error.value)
    for flag in STAGE.PRE_REGISTERED_DECISIONS:
        assert f"--{flag.replace('_', '-')}" in message


def test_a_protein_cell_without_the_text_control_artefact_is_refused():
    args = _args(
        arm="progen2-small", cut="tercile", max_pairs=40, null_draws=200, seed=1,
        reachability_pairs=4, reachability_margin=0.0, random_directions=8,
        ceiling_factor=2.0, records=8, max_tokens=64, min_symbol_occurrences=10,
        
    )
    with pytest.raises(ValueError, match="--text-control"):
        STAGE.resolve(args)


def test_a_text_arm_is_refused_the_decisions_that_decide_nothing_on_it():
    args = _args(
        arm="bygpt5-medium-en", cut="tercile", max_pairs=40, null_draws=200, seed=1,
        records=8, max_tokens=64, min_symbol_occurrences=10, 
        background_records=8, ceiling_factor=2.0,
    )
    with pytest.raises(ValueError, match="--ceiling-factor"):
        STAGE.resolve(args)


def test_the_synthetic_path_refuses_a_flag_that_names_a_real_campaign():
    args = _args(
        synthetic=True, cut="tercile", max_pairs=40, null_draws=200, seed=1,
        reachability_pairs=4, reachability_margin=0.0, random_directions=8,
        records=8,
    )
    with pytest.raises(ValueError, match="--records"):
        STAGE.resolve(args)


def test_too_few_random_directions_is_refused_before_a_model_is_loaded():
    args = _args(
        synthetic=True, cut="tercile", max_pairs=40, null_draws=200, seed=1,
        reachability_pairs=4, reachability_margin=0.0,
        random_directions=ac.MINIMUM_RANDOM_DIRECTIONS - 1,
    )
    with pytest.raises(ValueError, match="rule 39"):
        STAGE.resolve(args)


def test_the_artefact_basename_carries_the_arm_the_intervention_and_the_seed():
    name = STAGE.artefact_name("protein_cell", "progen2-small", 20260819)
    assert "progen2-small" in name
    assert ac.INTERVENTION in name
    assert "seed20260819" in name
    assert name != STAGE.artefact_name("protein_cell", "progen2-base", 20260819)
    assert name != STAGE.artefact_name("protein_cell", "progen2-small", 7)


def test_a_text_control_that_did_not_pass_stops_every_protein_cell(tmp_path):
    from src.transfer.io import write_json

    artefact = tmp_path / "control.json"
    write_json(
        artefact,
        {
            "schema_version": STAGE.SCHEMA_VERSION,
            "kind": "text_control",
            "arm": {"name": "bygpt5-medium-en"},
            "verdict": {"verdict": "VOID", "delta": -0.1, "difference_ci95": [-0.3, 0.1]},
        },
    )
    with pytest.raises(RuntimeError, match="protein arm is read"):
        STAGE.read_text_control(artefact)


def test_a_text_control_verdict_needs_both_a_positive_delta_and_an_interval():
    assert ac.text_control_verdict(
        {"delta": 0.5, "difference_ci95": [0.1, 0.9]}
    )["verdict"] == "PASS"
    assert ac.text_control_verdict(
        {"delta": 0.5, "difference_ci95": [-0.1, 0.9]}
    )["verdict"] == "VOID"
    assert ac.text_control_verdict({"delta": 0.5, "difference_ci95": None})["verdict"] == "VOID"


def test_the_protein_verdict_classifies_recombination_rather_than_calling_it_weak():
    margin = {"cleared": False, "clauses": {"delta_positive": False}}
    verdict = ac.protein_verdict(
        margin=margin, delta_block={"delta": -0.4, "difference_ci95": [-0.6, -0.2]}
    )
    assert verdict["verdict"] == "RECOMBINATION"
    inside = ac.protein_verdict(
        margin={"cleared": False, "clauses": {"above_random_direction_q95": False}},
        delta_block={"delta": 0.4, "difference_ci95": [0.1, 0.7]},
    )
    assert inside["verdict"] == "INSIDE_CEILING"
    assert inside["failed_clauses"] == ["above_random_direction_q95"]


# --------------------------------------------------------------- D3.j-B successor


def _toy_ordered(order: int, counts: np.ndarray) -> ac.OrderedFragmentCounts:
    return ac.OrderedFragmentCounts(
        order=order,
        counts=counts,
        source="toy",
        sha256="0" * 64,
        observed=int((counts > 0).sum()),
        possible=int(counts.size),
        total_kmers=int(counts.sum()),
    )


def _kmer_index(symbols: str) -> int:
    value = 0
    for character in symbols:
        value = value * 20 + AA20.index(character)
    return value


def _order2_conditional() -> ac.FragmentConditional:
    """A bigram table where A/C are interchangeable and A/D are not."""

    counts = np.ones(400, dtype=np.float64)
    for left, right, weight in (
        ("A", "A", 80.0),
        ("A", "C", 80.0),
        ("C", "A", 80.0),
        ("C", "C", 80.0),
        ("A", "D", 1.0),
        ("D", "A", 1.0),
        ("D", "D", 80.0),
    ):
        counts[_kmer_index(left + right)] = weight
    return ac.FragmentConditional(_toy_ordered(2, counts))


def test_a_omitted_protein_axis_is_still_variant_a():
    args = _args(
        arm="progen2-small", cut="tercile", max_pairs=40, null_draws=200, seed=1,
        reachability_pairs=4, reachability_margin=0.0, random_directions=8,
        ceiling_factor=2.0, records=8, max_tokens=64, min_symbol_occurrences=10,
        kmer_background=Path("x"), high_order_background=Path("y"),
        text_control=Path("z"), ceiling_orders="1,3,5",
    )
    STAGE.resolve(args)
    assert STAGE.selected_protein_axis(args) == ac.PROTEIN_AXIS_CONTEXT_PROFILE
    assert STAGE.is_variant_b(args) is False
    payload = STAGE.base_payload(args, kind="protein_cell")
    assert payload["schema_version"] == STAGE.SCHEMA_VERSION
    assert "experiment" not in payload
    assert "protein_axis" not in payload["settings"]
    assert "fragment_axis_order" not in payload["settings"]
    assert payload["pre_registration"]["track"] == ac.PRE_REGISTRATION_TRACK


def test_a_artefact_name_is_unchanged_when_variant_is_omitted():
    assert STAGE.artefact_name("protein_cell", "progen2-small", 20260819) == (
        f"alphabet_chemistry__progen2-small__{ac.INTERVENTION}__seed20260819.json"
    )


def test_b_requires_an_explicit_fragment_axis_order_among_the_ceiling_orders():
    args = _args(
        arm="progen2-small", cut="tercile", seed=1,
        records=8, max_tokens=64, min_symbol_occurrences=10,
        kmer_background=Path("x"), high_order_background=Path("y"),
        ceiling_orders="1,3,5",
        protein_axis=ac.PROTEIN_AXIS_FRAGMENT_SUBSTITUTION_DAMAGE,
        b_stage=ac.B_STAGE_CONSTRUCT,
    )
    with pytest.raises(ValueError, match="--fragment-axis-order"):
        STAGE.resolve(args)
    args.fragment_axis_order = 4
    with pytest.raises(ValueError, match="must be listed in --ceiling-orders"):
        STAGE.resolve(args)
    args.fragment_axis_order = 5
    STAGE.resolve(args)
    assert STAGE.is_variant_b(args) is True
    payload = STAGE.base_payload(args, kind=ac.KIND_AXIS_CONSTRUCTION)
    assert payload["schema_version"] == STAGE.SCHEMA_VERSION_B
    assert payload["experiment"] == ac.EXPERIMENT_B
    assert payload["settings"]["fragment_axis_order"] == 5
    assert ac.EXPERIMENT_B in STAGE.artefact_name(
        ac.KIND_AXIS_CONSTRUCTION, "progen2-small", 1, variant=ac.EXPERIMENT_B
    )


def test_a_refuses_a_fragment_axis_order_it_does_not_use():
    args = _args(
        arm="progen2-small", cut="tercile", max_pairs=40, null_draws=200, seed=1,
        reachability_pairs=4, reachability_margin=0.0, random_directions=8,
        ceiling_factor=2.0, records=8, max_tokens=64, min_symbol_occurrences=10,
        kmer_background=Path("x"), high_order_background=Path("y"),
        text_control=Path("z"), ceiling_orders="1,3",
        fragment_axis_order=3,
    )
    with pytest.raises(ValueError, match="decided nothing"):
        STAGE.resolve(args)


def test_the_unordered_axis_is_the_arithmetic_mean_of_the_two_directions():
    assert ac.symmetrize_directional_damage(2.0, 4.0) == 3.0


def test_b_axis_uses_fragment_damage_and_not_context_cosine():
    model = _order2_conditional()
    records = ["ACADCDCAACAD", "CDCADACDACAD"]
    directed = {}
    for x, source in enumerate(AA20):
        for y, target in enumerate(AA20):
            if x == y:
                continue
            directed[(x, y)] = model.damage(records, source, target)
    distance, observed, refusals = ac.fragment_damage_axis(directed, size=20)
    a, c, d = AA20.index("A"), AA20.index("C"), AA20.index("D")
    assert observed[a, c] and observed[a, d]
    assert distance[a, c] == ac.symmetrize_directional_damage(
        directed[(a, c)]["nats_per_scored_token"],
        directed[(c, a)]["nats_per_scored_token"],
    )
    assert distance[a, c] < distance[a, d]
    cosine = ac.cosine_distance(ac.context_profiles(np.eye(20) + 0.1))
    assert not np.allclose(np.nan_to_num(distance), cosine)
    assert any(item["reason"] == "insufficient fragment coverage" for item in refusals)


def test_a_pair_without_fragment_coverage_is_refused_not_imputed():
    model = ac.FragmentConditional(_toy_ordered(2, np.zeros(400)))
    directed = {(0, 1): model.damage(["A" * 8], "A", "C")}
    directed[(1, 0)] = model.damage(["C" * 8], "C", "A")
    distance, observed, refusals = ac.fragment_damage_axis(directed, size=20)
    assert not bool(observed[0, 1])
    assert np.isnan(distance[0, 1])
    assert refusals and refusals[0]["reason"] == "insufficient fragment coverage"


def test_the_matching_ceiling_must_have_the_opposite_ordering():
    codes = np.array([1, 1, -1, -1])
    ok = ac.matching_ceiling_predicts_distributional_side(codes, [0.1, 0.2, 0.8, 0.9])
    assert ok["status"] == "OK"
    void = ac.matching_ceiling_predicts_distributional_side(codes, [0.8, 0.9, 0.1, 0.2])
    assert void["status"] == "VOID"
    assert void["reason"] == ac.CEILING_CONSTRUCTION_VOID


def test_fragment_admission_cannot_see_model_damage():
    import inspect

    source = inspect.getsource(ac.fragment_damage_axis)
    assert "nats_per_scored_token" in source
    assert "arm" not in source.lower()
    directed = {
        (0, 1): {"measurable": True, "nats_per_scored_token": 0.2},
        (1, 0): {"measurable": True, "nats_per_scored_token": 0.4},
    }
    first, _, _ = ac.fragment_damage_axis(directed, size=4)
    directed[(0, 1)]["arm_damage"] = 99.0
    second, _, _ = ac.fragment_damage_axis(directed, size=4)
    assert first[0, 1] == pytest.approx(0.3)
    assert second[0, 1] == pytest.approx(0.3)


def test_a_historical_estimator_is_unchanged_on_the_old_path():
    codes = np.array([1, 1, 1, 1, -1, -1, -1, -1, 1, -1])
    groups = np.arange(10)
    block = ac.delta_contrast(
        codes=codes,
        damage=[1.0] * 5 + [0.0] * 5,
        groups=groups,
        seed=SEED,
        n_bootstrap=200,
    )
    assert block["resampling_unit"] == "substituted symbol"
    assert block["difference_ci95"] is not None


def _interval_payload(*, sequence_groups, n_pairs=16, n_records=16, seed=3):
    from src.transfer.crossed_group_interval import crossed_group_interval

    codes = np.array([1] * 8 + [-1] * 8)
    symbol_groups = np.arange(n_pairs)
    arm_sum = np.zeros((n_pairs, n_records))
    arm_count = np.ones((n_pairs, n_records))
    ceiling_sum = np.zeros((n_pairs, n_records))
    ceiling_count = np.ones((n_pairs, n_records))
    offset = np.arange(n_records) * 50.0
    for pair in range(n_pairs):
        signal = 1.0 if codes[pair] > 0 else 0.0
        arm_sum[pair] = signal + offset
        ceiling_sum[pair] = (1.0 - signal) + offset
    return crossed_group_interval(
        codes=codes,
        symbol_groups=symbol_groups,
        sequence_groups=np.asarray(sequence_groups),
        arm_sum=arm_sum,
        arm_count=arm_count,
        ceiling_sum=ceiling_sum,
        ceiling_count=ceiling_count,
        seed=seed,
        n_draws=2000,
    )


def test_the_crossed_interval_is_deterministic_paired_and_group_sensitive():
    groups = np.arange(16)
    first = _interval_payload(sequence_groups=groups)
    second = _interval_payload(sequence_groups=groups)
    assert first["difference_ci95"] == second["difference_ci95"]
    assert first["refused"] is False
    assert first["units"]["n_sequence_groups"] == 16
    assert first["units"]["n_symbol_groups"] == 16
    assert first["difference_ci95"][0] > 0.0
    width = first["difference_ci95"][1] - first["difference_ci95"][0]
    assert width < 1.0
    collapsed = _interval_payload(sequence_groups=np.zeros(16, dtype=int))
    assert collapsed["refused"] is True
    assert collapsed["difference_ci95"] is None


def test_the_crossed_interval_refuses_too_few_symbol_units():
    from src.transfer.crossed_group_interval import crossed_group_interval

    codes = np.array([1, 1, 1, -1, -1, -1])
    refused = crossed_group_interval(
        codes=codes,
        symbol_groups=np.array([0, 0, 1, 2, 3, 4]),
        sequence_groups=np.arange(8),
        arm_sum=np.ones((6, 8)),
        arm_count=np.ones((6, 8)),
        ceiling_sum=np.zeros((6, 8)),
        ceiling_count=np.ones((6, 8)),
        seed=1,
        n_draws=2000,
    )
    assert refused["refused"] is True
    assert refused["units"]["n_symbol_groups"] == 5


def test_duplicate_sequence_groups_are_not_treated_as_extra_units():
    from src.transfer.crossed_group_interval import crossed_group_interval

    n_pairs, n_unique = 16, 16
    codes = np.array([1] * 8 + [-1] * 8)
    arm_sum = np.zeros((n_pairs, n_unique))
    ceiling_sum = np.zeros((n_pairs, n_unique))
    for pair in range(n_pairs):
        arm_sum[pair] = 1.0 if codes[pair] > 0 else 0.0
        ceiling_sum[pair] = 0.0 if codes[pair] > 0 else 1.0
    singles = crossed_group_interval(
        codes=codes,
        symbol_groups=np.arange(n_pairs),
        sequence_groups=np.arange(n_unique),
        arm_sum=arm_sum,
        arm_count=np.ones_like(arm_sum),
        ceiling_sum=ceiling_sum,
        ceiling_count=np.ones_like(arm_sum),
        seed=9,
        n_draws=2000,
    )
    doubled_sum = np.concatenate([arm_sum, arm_sum], axis=1)
    doubled_ceil = np.concatenate([ceiling_sum, ceiling_sum], axis=1)
    doubled = crossed_group_interval(
        codes=codes,
        symbol_groups=np.arange(n_pairs),
        sequence_groups=np.concatenate([np.arange(n_unique), np.arange(n_unique)]),
        arm_sum=doubled_sum,
        arm_count=np.ones_like(doubled_sum),
        ceiling_sum=doubled_ceil,
        ceiling_count=np.ones_like(doubled_sum),
        seed=9,
        n_draws=2000,
    )
    assert singles["units"]["n_sequence_groups"] == doubled["units"]["n_sequence_groups"] == 16
    assert singles["difference_ci95"] == doubled["difference_ci95"]


def test_a_self_substitution_mean_is_still_exactly_zero_with_record_stats():
    world = _world()
    scorer = ac.DamageScorer(world.model, world.cohort(), batch_size=8)
    record = scorer.damage(0, 0)
    assert record["nats_per_scored_token"] == 0.0
    assert float(record["per_record_nll_sum"].sum()) == 0.0


def test_b_without_a_stage_is_refused():
    args = _args(
        arm="progen2-small", cut="tercile", seed=1,
        records=8, max_tokens=64, min_symbol_occurrences=10,
        kmer_background=Path("x"), high_order_background=Path("y"),
        ceiling_orders="1,3,5", fragment_axis_order=5,
        protein_axis=ac.PROTEIN_AXIS_FRAGMENT_SUBSTITUTION_DAMAGE,
    )
    with pytest.raises(ValueError, match="--b-stage"):
        STAGE.resolve(args)


def test_construct_refuses_model_measurement_flags():
    args = _args(
        arm="progen2-small", cut="tercile", seed=1,
        records=8, max_tokens=64, min_symbol_occurrences=10,
        kmer_background=Path("x"), high_order_background=Path("y"),
        ceiling_orders="1,3,5", fragment_axis_order=5,
        protein_axis=ac.PROTEIN_AXIS_FRAGMENT_SUBSTITUTION_DAMAGE,
        b_stage=ac.B_STAGE_CONSTRUCT, text_control=Path("z"),
    )
    with pytest.raises(ValueError, match="axis-construction"):
        STAGE.resolve(args)


def test_confirm_requires_construction_artefact_and_index():
    args = _args(
        arm="progen2-small", cut="tercile", max_pairs=40, null_draws=200, seed=1,
        reachability_pairs=4, reachability_margin=0.0, random_directions=8,
        ceiling_factor=2.0, records=8, max_tokens=64, min_symbol_occurrences=10,
        kmer_background=Path("x"), high_order_background=Path("y"),
        text_control=Path("z"), ceiling_orders="1,3,5", fragment_axis_order=5,
        protein_axis=ac.PROTEIN_AXIS_FRAGMENT_SUBSTITUTION_DAMAGE,
        b_stage=ac.B_STAGE_CONFIRM,
    )
    with pytest.raises(ValueError, match="--construction-artefact"):
        STAGE.resolve(args)
    args.construction_artefact = Path("frozen.json")
    with pytest.raises(ValueError, match="--confirmation-index"):
        STAGE.resolve(args)


def test_crossed_interval_reports_arm_and_ceiling_percentiles():
    block = _interval_payload(sequence_groups=np.arange(16))
    assert block["delta_ci95"] is not None
    assert block["reference_delta_ci95"] is not None
    assert block["difference_ci95"] is not None
    assert block["delta_ci95"][0] <= block["delta"] <= block["delta_ci95"][1]


def test_b_verdict_follows_the_crossed_arm_interval_not_the_symbol_only_one():
    from src.transfer.crossed_group_interval import crossed_group_interval

    n_pairs, n_records = 16, 16
    codes = np.array([1] * 8 + [-1] * 8)
    arm_sum = np.zeros((n_pairs, n_records))
    arm_count = np.ones((n_pairs, n_records))
    ceiling_sum = np.zeros((n_pairs, n_records))
    ceiling_count = np.ones((n_pairs, n_records))
    for pair in range(n_pairs):
        if codes[pair] > 0:
            arm_sum[pair, :4] = 5.0
            arm_sum[pair, 4:] = -2.0
        ceiling_sum[pair] = 0.5
    crossed = crossed_group_interval(
        codes=codes,
        symbol_groups=np.arange(n_pairs),
        sequence_groups=np.arange(n_records),
        arm_sum=arm_sum,
        arm_count=arm_count,
        ceiling_sum=ceiling_sum,
        ceiling_count=ceiling_count,
        seed=11,
        n_draws=2000,
    )
    pair_means = arm_sum.sum(axis=1) / arm_count.sum(axis=1)
    own = ac.delta_contrast(
        codes=codes, damage=pair_means, groups=np.arange(n_pairs),
        seed=11, n_bootstrap=2000,
    )
    assert own["difference_ci95"][1] < 0.0
    assert own["delta"] < 0.0
    symbol_only = ac.protein_verdict(
        margin={"cleared": False, "clauses": {"delta_positive": False}},
        delta_block=own,
    )
    assert symbol_only["verdict"] == "RECOMBINATION"
    assert crossed["delta_ci95"][0] < 0.0 < crossed["delta_ci95"][1]
    margin = {"cleared": False, "clauses": {"delta_positive": False}}
    crossed_verdict = ac.protein_verdict_b(margin=margin, crossed=crossed)
    assert crossed_verdict["verdict"] == "UNDECIDED"
    assert crossed_verdict["verdict"] != symbol_only["verdict"]


def test_a_refused_crossed_interval_is_void_not_inside_ceiling():
    from src.transfer.crossed_group_interval import crossed_group_interval

    refused = crossed_group_interval(
        codes=np.array([1, 1, 1, -1, -1, -1]),
        symbol_groups=np.array([0, 0, 1, 2, 3, 4]),
        sequence_groups=np.arange(8),
        arm_sum=np.ones((6, 8)),
        arm_count=np.ones((6, 8)),
        ceiling_sum=np.zeros((6, 8)),
        ceiling_count=np.ones((6, 8)),
        seed=1,
        n_draws=2000,
    )
    assert refused["refused"] is True
    fallback = ac.protein_verdict(
        margin={"cleared": False, "clauses": {}},
        delta_block={"delta": 0.4, "difference_ci95": None},
    )
    assert fallback["verdict"] == "INSIDE_CEILING"
    verdict = ac.protein_verdict_b(
        margin={"cleared": False, "clauses": {}}, crossed=refused
    )
    assert verdict["verdict"] == "VOID"
    assert verdict["reason"] == ac.CROSSED_INTERVAL_REFUSED


def _construction_payload(**overrides):
    members = {
        "tercile": {
            ac.QUADRANTS[0]: ["AC", "AD"],
            ac.QUADRANTS[1]: ["CW", "DY"],
        }
    }
    payload = {
        "schema_version": STAGE.SCHEMA_VERSION_B,
        "kind": ac.KIND_AXIS_CONSTRUCTION,
        "experiment": ac.EXPERIMENT_B,
        "verdict": {"verdict": ac.AXIS_CONSTRUCTED},
        "tokenizer_identity": {
            "arm": "progen2-small",
            "architecture": "progen",
            "tokenisation": "residue",
            "input_format": "progen2",
            "vocab_size": 30,
            "tokenizer_class": "Stub",
            "max_tokens": 64,
        },
        "axes": {
            "labels": list(AA20),
            "distributional": {
                "kind": ac.PROTEIN_AXIS_FRAGMENT_SUBSTITUTION_DAMAGE,
                "order": 5,
                "symmetrization": ac.FRAGMENT_AXIS_SYMMETRIZATION,
                "corpus": {"sha256": "abc", "order": 5},
            },
            "matrices": {"distributional_fragment_damage": [[0.0]]},
        },
        "contradiction_set": {
            "declared_cut": "tercile",
            "unordered_members": members,
        },
        "cohort": {
            "digest": "construct",
            "n_records": 2,
            "sampling": {"skip": 0, "seed": 1},
            "records": ["ACDEFGHIKLMNPQRSTVWYACDE", "ACDEFGHIKLMNPQRSTVWYAAAA"],
        },
    }
    payload.update(overrides)
    return payload


def test_construction_artefact_schema_mismatch_is_refused(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"schema_version": "nope", "kind": "axis_construction"}')
    with pytest.raises(ValueError, match="schema"):
        STAGE._load_construction_artefact(path)


def test_confirmation_refuses_an_artefact_that_is_not_constructed(tmp_path):
    from src.transfer.io import write_json

    path = tmp_path / "ok.json"
    write_json(path, _construction_payload())
    loaded = STAGE._load_construction_artefact(path)
    assert loaded["axes"]["distributional"]["order"] == 5
    broken = _construction_payload()
    broken["verdict"] = {"verdict": "NO_CONTRADICTION_SET_AT_DECLARED_CUT"}
    bad = tmp_path / "not_constructed.json"
    write_json(bad, broken)
    with pytest.raises(ValueError, match="not AXIS_CONSTRUCTED"):
        STAGE._load_construction_artefact(bad)
    wrong_order = _construction_payload()
    wrong_order["axes"]["distributional"]["order"] = 3
    assert wrong_order["axes"]["distributional"]["order"] != 5


def test_same_or_near_duplicate_cohorts_are_not_independent():
    records = ["ACDEFGHIKLMNPQRSTVWYACDE", "GGGGGGGGGGGGGGGGGGGGGGGG"]
    same = ac.cohorts_independent(records, list(records))
    assert same["independent"] is False
    assert same["reason"] == "EXACT_CONTENT_OVERLAP"
    near = ac.cohorts_independent(
        ["ACDEFGHIKLMNPQRSTVWYAAAA"],
        ["ACDEFGHIKLMNPQRSTVWYAAAV"],
    )
    assert near["independent"] is False
    assert near["reason"] == "NEAR_DUPLICATE_OVERLAP"
    far = ac.cohorts_independent(
        ["ACDEFGHIKLMNPQRSTVWY" * 2],
        ["WWWWWWWWWWWWWWWWWWWW" * 2],
    )
    assert far["independent"] is True


def test_confirmation_skip_defaults_away_from_the_construction_draw():
    args = _args(
        confirmation_index=1, cohort_draw_seed=7, cohort_skip=None,
    )
    construction = {
        "cohort": {"n_records": 10, "sampling": {"skip": 0, "seed": 7}}
    }
    STAGE._apply_confirmation_skip(args, construction)
    assert args.cohort_skip == 10
    args.cohort_skip = 0
    with pytest.raises(ValueError, match="same cohort cannot serve both roles"):
        STAGE._apply_confirmation_skip(args, construction)


def test_frozen_pair_membership_does_not_recompute_quantiles():
    members = {
        ac.QUADRANTS[0]: ["AC"],
        ac.QUADRANTS[1]: ["DW"],
    }
    pairs = ac.frozen_pair_set(members, list(AA20))
    assert {"AC", "CA", "DW", "WD"} == {
        f"{AA20[x]}{AA20[y]}" for x, y in pairs.pairs
    }
    live = ac.quadrants_at_cut(
        np.abs(np.arange(20)[:, None] - np.arange(20)[None, :]).astype(np.float64),
        np.abs(np.arange(20)[:, None] + np.arange(20)[None, :]).astype(np.float64),
        cut="tercile",
    )
    live_tokens = {
        name: [f"{AA20[x]}{AA20[y]}" for x, y in live["members"][name]]
        for name in ac.QUADRANTS
    }
    assert members[ac.QUADRANTS[0]] != live_tokens[ac.QUADRANTS[0]] or members[
        ac.QUADRANTS[1]
    ] != live_tokens[ac.QUADRANTS[1]]
