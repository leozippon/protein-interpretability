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
