"""What one cohort-power record has to hold together, and what it has to keep.

Two failures motivate this file, and both were silent.

*One record, two baselines.* ``budget.arm_power`` computed every figure against
the in-cohort plug-in unigram entropy and ``01_cohort_power.py`` then replaced
*some* of them with figures taken against a held-out cross-entropy. What
survived on disk was ZymCTRL's ``context_information_miller_madow_nats`` 2.027
sitting beside ``context_information_nats`` 2.029: two "context information"
numbers, two different baselines, and nothing in either name to say which.

*Aggregates only.* The stage published means and intervals and discarded the
per-record quantities they were computed from, so re-analysing its uncertainty
meant re-running the forward passes. The same bill was paid once already for the
induction census.

*The reference, unfrozen.* The sidecar's held-out counts are order-free, and the
block they came from was not persisted at all, so neither the near-duplicate
structure of the reference nor its k-mer overlap with the scored cohort could be
recovered from disk -- which is both halves of what
``41_context_information_bootstrap.py`` needs to run E5.

The tests below therefore pin behaviour rather than implementation: an identity
the held-out figure must satisfy exactly, the naming rule that makes a record
readable, the refusal that keeps an estimator from silently degrading, a round
trip that rebuilds the published figures from the sidecar with the model absent,
and the reference file that lets the consumer stage compute what it otherwise
declares unavailable.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer.arms import PANEL, Arm, Cohort  # noqa: E402
from src.transfer.budget import (  # noqa: E402
    POWER_RECORDS_SCHEMA_VERSION,
    SparseCounts,
    arm_power,
    arm_power_with_records,
    write_power_records,
)
from src.transfer.io import sha256_file, write_json  # noqa: E402
from src.transfer.pathways import (  # noqa: E402
    AblationScope,
    BaselineBank,
    ScoredBatch,
    Target,
    cohort_target_token_counts,
    disjoint_unigram_cross_entropy_nats,
    held_out_cohort,
    measure_pathways,
)
from src.transfer.prediction_addressed import (  # noqa: E402
    cohort_power_held_out,
    scored_target_counts,
    scored_target_records,
)

ALPHABET = "abcdefghij"
#: The alphabet plus the word separator, which is a content character of a text
#: record and not punctuation: ``near_duplicates`` shingles a text record by
#: *words*, so a record carrying no separator carries no shingles at all.
SYMBOLS = ALPHABET + " "
VOCAB = 16
MAX_LEN = 24
LN2 = math.log(2.0)


class _CharTokenizer:
    """One id per character, which is enough for every path exercised here.

    Ids start at one so that the pad id is not a content id; anything outside
    the alphabet is dropped on decode, which never happens below.
    """

    pad_token_id = 0

    def __call__(self, text: str, return_tensors=None) -> dict[str, list[int]]:
        return {"input_ids": [SYMBOLS.index(character) + 1 for character in text]}

    def decode(self, ids) -> str:
        return "".join(SYMBOLS[int(value) - 1] for value in ids if 1 <= int(value) <= len(SYMBOLS))


def _tiny_arm(*, capabilities: frozenset[str] | None = None, seed: int = 3) -> Arm:
    """A real two-layer GPT-2 over a ten-character alphabet, on the CPU.

    Every quantity under test comes out of a forward pass, so a stub that never
    runs one cannot reach them. Eight dimensions, sixteen vocabulary entries.
    """

    config = GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=8,
        vocab_size=VOCAB,
        n_positions=64,
        attn_implementation="eager",
    )
    torch.manual_seed(seed)
    spec = replace(PANEL["gpt2"], name="tiny-gpt2", n_layer=2, d_model=8)
    if capabilities is not None:
        spec = replace(spec, capabilities=capabilities)
    return Arm(
        spec=spec,
        model=GPT2LMHeadModel(config).eval(),
        tokenizer=_CharTokenizer(),
        device="cpu",
        dtype="float32",
        attn_implementation="eager",
    )


def _cohort(name: str, *, seed: int, n_records: int, weights: np.ndarray) -> Cohort:
    """Records drawn from a deliberately skewed alphabet distribution.

    Skewed, and differently skewed between the scored cohort and its reference,
    so that the plug-in and held-out baselines are far apart: a test in which
    the two coincide cannot tell which one a field was computed against.
    """

    generator = np.random.default_rng(seed)
    letters = np.array(list(ALPHABET))
    probabilities = weights / weights.sum()
    records = [
        "".join(generator.choice(letters, size=int(generator.integers(12, MAX_LEN)), p=probabilities))
        for _ in range(n_records)
    ]
    return Cohort(name=name, kind="text", records=records, min_symbols=12, max_symbols=MAX_LEN)


@pytest.fixture(scope="module")
def measurement():
    """One scored cohort, one disjoint reference, one arm, measured once."""

    arm = _tiny_arm()
    cohort = _cohort("scored", seed=11, n_records=6, weights=np.arange(1.0, 11.0))
    reference = _cohort("reference", seed=12, n_records=8, weights=np.arange(10.0, 0.0, -1.0))
    assert not set(cohort.records) & set(reference.records)
    counts, per_record = scored_target_records(
        arm, reference.input_strings(arm), max_len=MAX_LEN
    )
    report, records = arm_power_with_records(
        arm,
        cohort,
        max_len=MAX_LEN,
        batch_size=2,
        unigram_estimator="disjoint",
        reference_token_counts=counts,
        reference={"cohort": reference.name, "digest": reference.digest},
    )
    return {
        "arm": arm,
        "cohort": cohort,
        "reference": reference,
        "reference_counts": counts,
        "records": replace(records, reference_counts=per_record),
        "report": report,
    }


# ------------------------------------------------------- one record, one baseline


def test_held_out_context_information_is_the_cross_entropy_minus_the_clean_ce(measurement):
    """The published definition, held exactly rather than approximately.

    ``context_information_nats`` for a held-out run is
    ``disjoint_unigram_cross_entropy_nats(reference, targets) - clean_ce_nats``
    and nothing else. It is asserted against a target-count vector built by the
    tokenizer-only path, which also pins the claim that that path counts the
    same multiset the forward pass scores.
    """

    report = measurement["report"]
    arm, cohort = measurement["arm"], measurement["cohort"]
    targets = scored_target_counts(arm, cohort.input_strings(arm), max_len=MAX_LEN)
    assert int(targets.sum()) == report["n_scored_tokens"]
    expected = disjoint_unigram_cross_entropy_nats(measurement["reference_counts"], targets)
    assert report["unigram_entropy_held_out_nats"] == pytest.approx(expected, rel=1e-15)
    assert report["context_information_nats"] == pytest.approx(
        expected - report["clean_ce_nats"], rel=1e-15
    )
    assert report["unigram_entropy_used_for_verdict_nats"] == report[
        "unigram_entropy_held_out_nats"
    ]


def test_no_two_published_fields_come_from_different_baselines_unnamed(measurement):
    """The ZymCTRL defect, stated as the rule that would have caught it.

    A field whose name carries ``plug_in``, ``on_cohort`` or ``miller_madow`` is
    the in-cohort plug-in family; every other baseline-derived field is the
    estimator the record names. The two baselines are far apart here, so a field
    computed against the wrong one cannot pass by coincidence.
    """

    report = measurement["report"]
    plug_in = report["unigram_entropy_on_cohort_nats"]
    held_out = report["unigram_entropy_used_for_verdict_nats"]
    clean = report["clean_ce_nats"]
    expansion = report["symbols_per_token"]
    assert abs(held_out - plug_in) > 0.05, "the two baselines must differ for this to test anything"

    # Named for the held-out baseline, or named for nothing, so: held-out.
    assert report["context_information_nats"] == pytest.approx(held_out - clean, rel=1e-15)
    assert report["context_information_bits_per_symbol"] == pytest.approx(
        (held_out - clean) / LN2 / expansion, rel=1e-15
    )
    assert report["unigram_entropy_bits_per_symbol"] == pytest.approx(
        held_out / LN2 / expansion, rel=1e-15
    )
    interval = report["per_sequence_context_information_interval"]
    per_record = measurement["records"]
    per_sequence = per_record.clean_nll_sum / per_record.token_count
    assert interval["mean"] == pytest.approx(float(np.mean(held_out - per_sequence)), rel=1e-12)

    # Named for the plug-in, and computed against it.
    assert report["context_information_plug_in_nats"] == pytest.approx(plug_in - clean, rel=1e-15)
    miller_madow = report["unigram_entropy_plug_in_miller_madow_nats"]
    assert miller_madow > plug_in
    assert report["context_information_plug_in_miller_madow_nats"] == pytest.approx(
        miller_madow - clean, rel=1e-15
    )

    # The two names the defect wore are gone: a Miller-Madow correction is a
    # correction to the plug-in estimator and has no meaning against a held-out
    # cross-entropy, so the bare names cannot be re-introduced.
    assert "context_information_miller_madow_nats" not in report
    assert "unigram_entropy_miller_madow_nats" not in report

    # Every remaining figure the estimator could be read off travels with it.
    assert report["unigram_estimator"] == "disjoint"
    assert report["unigram_baseline"]["estimator"] == "disjoint"
    assert report["unigram_baseline"]["reference"]["digest"] == measurement["reference"].digest
    assert report["cross_arm_comparable"] is True


def test_each_verdict_is_taken_against_the_baseline_its_name_carries(measurement):
    """Both verdicts are published, and neither is the other's baseline."""

    report = measurement["report"]
    threshold = report["minimum_context_information_nats"]
    assert report["power_verdict"] == (
        "PASS" if report["context_information_nats"] >= threshold else "FAIL"
    )
    assert report["power_verdict_plug_in"] == (
        "PASS" if report["context_information_plug_in_nats"] >= threshold else "FAIL"
    )
    # The held-out verdict is the headline, so its value is not estimator-tagged;
    # the plug-in one is tagged by its key and is never overwritten.
    assert report["measurability"] in ("measurable", "unmeasurable_on_this_cohort")
    assert report["measurability_plug_in"] in ("measurable", "unmeasurable_on_this_cohort")


def test_the_plug_in_run_names_its_estimator_in_the_verdict_itself():
    """Without a reference, the verdict cannot be quoted as a cross-arm one."""

    arm = _tiny_arm()
    cohort = _cohort("scored", seed=11, n_records=6, weights=np.arange(1.0, 11.0))
    report = arm_power(arm, cohort, max_len=MAX_LEN, batch_size=2)
    assert report["unigram_estimator"] == "plugin"
    assert report["cross_arm_comparable"] is False
    assert report["measurability"].endswith("against_plug_in_baseline")
    assert report["context_information_nats"] == pytest.approx(
        report["context_information_plug_in_nats"], rel=1e-15
    )
    assert "unigram_entropy_held_out_nats" not in report
    assert "plug_in_bias_nats" not in report


# --------------------------------------------------------------- no silent fallback


def test_the_held_out_estimator_without_a_reference_raises():
    """A silent downgrade to the plug-in moves every figure and shows nothing."""

    arm = _tiny_arm()
    cohort = _cohort("scored", seed=11, n_records=4, weights=np.arange(1.0, 11.0))
    with pytest.raises(ValueError, match="no fallback"):
        arm_power(arm, cohort, max_len=MAX_LEN, batch_size=2, unigram_estimator="disjoint")
    with pytest.raises(ValueError, match="no fallback"):
        arm_power(
            arm,
            cohort,
            max_len=MAX_LEN,
            batch_size=2,
            unigram_estimator="disjoint",
            reference_token_counts=np.ones(VOCAB, dtype=np.int64),
        )


def test_a_reference_the_plug_in_would_ignore_raises():
    """A record that names a reference it did not use is worse than none."""

    arm = _tiny_arm()
    cohort = _cohort("scored", seed=11, n_records=4, weights=np.arange(1.0, 11.0))
    with pytest.raises(ValueError, match="plug-in estimator"):
        arm_power(
            arm,
            cohort,
            max_len=MAX_LEN,
            batch_size=2,
            unigram_estimator="plugin",
            reference_token_counts=np.ones(VOCAB, dtype=np.int64),
            reference={"cohort": "somewhere", "digest": "0" * 8},
        )


def test_an_unknown_estimator_is_refused_by_name():
    arm = _tiny_arm()
    cohort = _cohort("scored", seed=11, n_records=4, weights=np.arange(1.0, 11.0))
    with pytest.raises(ValueError, match="unknown unigram estimator"):
        arm_power(arm, cohort, max_len=MAX_LEN, batch_size=2, unigram_estimator="held_out")


# ------------------------------------------------- the sidecar, without the model


def test_the_sidecar_reproduces_the_report_without_re_running_the_model(
    measurement, tmp_path
):
    """The load-bearing property: a re-analysis needs the arrays and no GPU.

    ``clean_ce_nats``, ``n_scored_tokens`` and the held-out baseline are rebuilt
    from the persisted arrays alone. Nothing here touches the arm.
    """

    report = measurement["report"]
    destination = tmp_path / "power_scored_0123456789ab.records.npz"
    metadata = write_power_records(
        destination,
        {"tiny-gpt2": measurement["records"]},
        cohort_digest=measurement["cohort"].digest,
        reference_digest=measurement["reference"].digest,
        smoothing=report["unigram_baseline"]["smoothing"],
        seeds={"cohort_draw": 20260727, "truncation_curve": 7},
        max_len=MAX_LEN,
    )
    assert metadata["schema_version"] == POWER_RECORDS_SCHEMA_VERSION
    assert metadata["sha256"] == sha256_file(destination)
    assert metadata["path"] == destination.name

    stored = np.load(destination)
    assert str(stored["schema_version"]) == POWER_RECORDS_SCHEMA_VERSION
    assert list(stored["arms"]) == ["tiny-gpt2"]
    assert str(stored["cohort_digest"]) == measurement["cohort"].digest
    vocab = int(stored["tiny-gpt2::vocab_size"])

    tokens = stored["tiny-gpt2::token_count"]
    assert int(tokens.sum()) == report["n_scored_tokens"]
    clean_ce = float(stored["tiny-gpt2::clean_nll_sum"].sum() / tokens.sum())
    assert clean_ce == pytest.approx(report["clean_ce_nats"], rel=1e-12)

    def dense(prefix: str) -> np.ndarray:
        return SparseCounts(
            offsets=stored[f"{prefix}counts_offsets"],
            token_ids=stored[f"{prefix}unique_token_ids"],
            counts=stored[f"{prefix}counts"],
        ).vocabulary_totals(vocab)

    held_out = disjoint_unigram_cross_entropy_nats(
        dense("tiny-gpt2::reference_"),
        dense("tiny-gpt2::"),
        smoothing=float(stored["smoothing"]),
    )
    assert held_out == pytest.approx(report["unigram_entropy_held_out_nats"], rel=1e-12)

    # And the per-record cross-entropies the interval was taken over.
    per_sequence = stored["tiny-gpt2::clean_nll_sum"] / tokens
    interval = report["per_sequence_context_information_interval"]
    assert interval["mean"] == pytest.approx(
        float(np.mean(held_out - per_sequence)), rel=1e-10
    )
    assert int(stored["tiny-gpt2::reference_token_count"].sum()) == int(
        measurement["reference_counts"].sum()
    )


def test_the_sidecar_carries_no_sequence_text(measurement, tmp_path):
    """The cohort JSON beside it already holds every record; a second copy is a
    second source, and it would travel without the digest that identifies it."""

    destination = tmp_path / "power_scored_0123456789ab.records.npz"
    write_power_records(
        destination,
        {"tiny-gpt2": measurement["records"]},
        cohort_digest=measurement["cohort"].digest,
        reference_digest=measurement["reference"].digest,
        smoothing=1.0,
        seeds={"cohort_draw": 1},
        max_len=MAX_LEN,
    )
    stored = np.load(destination)
    textual = {name for name in stored.files if stored[name].dtype.kind in "US"}
    assert textual == {"schema_version", "arms", "cohort_digest", "reference_digest", "seed_names"}
    joined = " ".join(str(stored[name]) for name in textual)
    for record in measurement["cohort"].records + measurement["reference"].records:
        assert record not in joined


def test_the_sidecar_is_replaced_atomically_and_leaves_no_temporary(
    measurement, tmp_path
):
    """A partial array file is indistinguishable from a complete one to a reader
    that only checks the schema version, which is why the JSON writer beside it
    renames rather than writes in place."""

    destination = tmp_path / "power_scored_0123456789ab.records.npz"
    for smoothing in (1.0, 0.5):
        write_power_records(
            destination,
            {"tiny-gpt2": measurement["records"]},
            cohort_digest=measurement["cohort"].digest,
            reference_digest=None,
            smoothing=smoothing,
            seeds={"cohort_draw": 1},
            max_len=MAX_LEN,
        )
    assert [path.name for path in tmp_path.iterdir()] == [destination.name]
    assert float(np.load(destination)["smoothing"]) == 0.5
    assert str(np.load(destination)["reference_digest"]) == ""


def test_a_record_that_scored_no_token_keeps_its_row_in_the_reference_counts():
    """Rows are aligned with records, so a re-analysis can subset by record.

    ``np.add.reduceat`` returns the element at the offset for an empty span
    rather than zero, which would attribute one record's tokens to another; the
    compressed form is therefore summed by differencing a cumulative sum.
    """

    counts = SparseCounts.from_records(
        [np.array([1, 1, 2], dtype=np.int64), np.zeros(0, dtype=np.int64), np.array([3], dtype=np.int64)]
    )
    assert counts.n_records == 3
    assert list(counts.record_totals()) == [3, 0, 1]
    assert list(counts.vocabulary_totals(5)) == [0, 2, 1, 1, 0]


# ----------------------------------------------------------- capability guards


def _budget_less_arm() -> Arm:
    """An arm declared for the pathway family alone, as the staged ProGen2 rungs
    are: ``config.vocab_size`` is 51200 against a 31-token tokenizer on one and
    absent on the other, so a statistic taken over it is not commensurate."""

    return _tiny_arm(capabilities=frozenset({"pathway"}))


def test_scored_target_records_refuses_an_arm_without_the_budget_capability():
    arm = _budget_less_arm()
    with pytest.raises(ValueError, match="budget"):
        scored_target_records(arm, ["abc", "bcd"], max_len=MAX_LEN)
    with pytest.raises(ValueError, match="budget"):
        scored_target_counts(arm, ["abc", "bcd"], max_len=MAX_LEN)


def test_cohort_power_held_out_refuses_an_arm_without_the_budget_capability():
    arm = _budget_less_arm()
    cohort = _cohort("scored", seed=11, n_records=3, weights=np.arange(1.0, 11.0))
    reference = _cohort("reference", seed=12, n_records=3, weights=np.arange(10.0, 0.0, -1.0))
    with pytest.raises(ValueError, match="budget"):
        cohort_power_held_out(
            arm, cohort, reference, max_len=MAX_LEN, batch_size=2, threshold_nats=0.3
        )


def test_cohort_target_token_counts_refuses_an_arm_without_the_budget_capability():
    arm = _budget_less_arm()
    cohort = _cohort("scored", seed=11, n_records=3, weights=np.arange(1.0, 11.0))
    with pytest.raises(ValueError, match="budget"):
        cohort_target_token_counts(arm, cohort, max_len=MAX_LEN)


def test_measure_pathways_refuses_an_arm_without_the_budget_capability():
    arm = _budget_less_arm()
    ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    batch = ScoredBatch(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        target_mask=torch.ones((1, 3), dtype=torch.bool),
        sequence_indices=(0,),
    )
    scope = AblationScope(
        name="mlp0", family="single", submodules=("mlp",), anchor_layer=0, width=None
    )
    bank = BaselineBank(
        kind="zero", vectors={Target("mlp", 0): torch.zeros(8)}, provenance={}
    )
    with pytest.raises(ValueError, match="budget"):
        measure_pathways(arm, [batch], [scope], bank)


def test_arm_power_still_refuses_a_budget_less_arm_before_the_forward_pass():
    """The guard the other four now match, restated against the same fixture."""

    arm = _budget_less_arm()
    cohort = _cohort("scored", seed=11, n_records=3, weights=np.arange(1.0, 11.0))
    with pytest.raises(ValueError, match="budget"):
        arm_power(arm, cohort, max_len=MAX_LEN, batch_size=2)


# ------------------------------------------ the held-out reference, on disk


def _stage(filename: str, name: str):
    """A numerically named stage script, which is not on the import path."""

    path = REPO_ROOT / "scripts" / "transfer" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


LEAK_KMER = 10


def _worded(cohort: Cohort) -> Cohort:
    """The same records, cut into two-character words.

    ``near_duplicates`` shingles a *text* record by words -- five characters of
    English measure spelling rather than content -- so an undifferentiated
    character string carries no shingles and is its own group whatever it holds.
    A reference of such strings would pass a grouping test that never grouped
    anything, which is a test that cannot fail.
    """

    return replace(
        cohort,
        records=[
            " ".join(record[index : index + 2] for index in range(0, len(record), 2))
            for record in cohort.records
        ],
    )


@pytest.fixture
def frozen_run(tmp_path):
    """One held-out run's artefacts on disk, written the way the stage writes them.

    The reference carries two planted structures a real corpus supplies of its
    own accord and a random draw does not: a near-copy of another reference
    record, which is what the reference grouping exists to find, and a near-copy
    of a *scored* record, which ``pathways.assert_disjoint`` cannot see because
    it is not an exact repeat and which is exactly what E5's k-mer screen is for.
    Both survive ``held_out_cohort``, since neither is an exact duplicate.
    """

    stage = _stage("01_cohort_power.py", "_stage_01_reference")
    arm = _tiny_arm()
    cohort = _worded(_cohort("scored", seed=11, n_records=6, weights=np.arange(1.0, 11.0)))
    drawn = _worded(
        _cohort(
            "scored_unigram_reference",
            seed=12,
            n_records=8,
            weights=np.arange(10.0, 0.0, -1.0),
        )
    )
    candidate = replace(
        drawn,
        records=[
            *drawn.records,
            f"{drawn.records[0].rsplit(' ', 1)[0]} aa",
            f"{cohort.records[0].rsplit(' ', 1)[0]} jj",
        ],
    )
    reference, overlap = held_out_cohort(candidate, cohort)
    assert overlap["dropped_sequences_shared_with_cohort"] == 0

    counts, per_record = scored_target_records(
        arm, reference.input_strings(arm), max_len=MAX_LEN
    )
    report, records = arm_power_with_records(
        arm,
        cohort,
        max_len=MAX_LEN,
        batch_size=2,
        unigram_estimator="disjoint",
        reference_token_counts=counts,
        reference={"cohort": reference.name, "digest": reference.digest},
    )
    stem = f"{cohort.name}_{cohort.digest[:12]}"
    write_json(
        tmp_path / f"cohort_{stem}.json",
        {
            "schema_version": stage.SCHEMA_VERSION,
            "artifact": "frozen_cohort",
            "cohort_digest": cohort.digest,
            "cohort_name": cohort.name,
            "cohort_kind": cohort.kind,
            "min_symbols": cohort.min_symbols,
            "max_symbols": cohort.max_symbols,
            "n_records": len(cohort),
            "records": cohort.records,
            "metadata": cohort.metadata,
        },
    )
    sidecar = tmp_path / f"power_{stem}.records.npz"
    write_power_records(
        sidecar,
        {"tiny-gpt2": replace(records, reference_counts=per_record)},
        cohort_digest=cohort.digest,
        reference_digest=reference.digest,
        smoothing=report["unigram_baseline"]["smoothing"],
        seeds={"cohort_draw": 20260727},
        max_len=MAX_LEN,
    )
    return {
        "arm": arm,
        "cohort": cohort,
        "reference": reference,
        "reference_counts": counts,
        "sidecar": sidecar,
        "path": stage.write_reference_records(tmp_path, cohort, reference),
    }


def test_a_plug_in_run_writes_no_reference_file_at_all(tmp_path):
    """There is no reference to freeze, and an empty one would have to be read."""

    stage = _stage("01_cohort_power.py", "_stage_01_reference")
    destination = tmp_path / "out"
    cohort = _cohort("scored", seed=11, n_records=4, weights=np.arange(1.0, 11.0))
    assert stage.write_reference_records(destination, cohort, None) is None
    assert not destination.exists()


def test_the_reference_file_is_named_and_digested_as_the_sidecar_names_it(frozen_run):
    """The one check a consumer holding both files can make.

    The sidecar's ``reference_digest`` is what every context-information figure
    was measured against; a reference file under a different digest describes a
    different draw, and the two would be indistinguishable once the records were
    read out of it.
    """

    path, reference, cohort = frozen_run["path"], frozen_run["reference"], frozen_run["cohort"]
    assert path.name == f"reference_{cohort.name}_{reference.digest[:12]}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    stored = np.load(frozen_run["sidecar"])
    assert payload["reference_digest"] == str(stored["reference_digest"])
    assert payload["cohort_digest"] == str(stored["cohort_digest"])
    assert payload["artifact"] == "held_out_unigram_reference"
    assert payload["n_records"] == len(reference) == stored["tiny-gpt2::reference_token_count"].size
    # And the draw that produced the block travels with it, as it does for the
    # cohort: a baseline whose sampling mode is unknowable is Appendix B rule 1.
    assert payload["metadata"]["sampling"]["held_out_against_digest"] == cohort.digest


def test_the_persisted_records_retokenise_to_the_sidecar_reference_counts(frozen_run):
    """The round trip that makes the file a reference and not a note about one.

    Re-tokenising what was written reproduces the per-record token counts and the
    vocabulary totals the sidecar holds, so a re-analysis that subsets the
    reference by record -- which is the whole of E5 -- is subsetting the same
    objects the published baseline was fitted on.
    """

    payload = json.loads(frozen_run["path"].read_text(encoding="utf-8"))
    restored = Cohort(
        name=payload["reference_name"],
        kind=payload["reference_kind"],
        records=payload["records"],
        min_symbols=payload["min_symbols"],
        max_symbols=payload["max_symbols"],
        metadata=payload["metadata"],
    )
    assert restored.digest == payload["reference_digest"]
    counts, per_record = scored_target_records(
        frozen_run["arm"], restored.input_strings(frozen_run["arm"]), max_len=MAX_LEN
    )
    stored = np.load(frozen_run["sidecar"])
    assert list(per_record.record_totals()) == list(stored["tiny-gpt2::reference_token_count"])
    assert list(counts) == list(frozen_run["reference_counts"])
    vocab = int(stored["tiny-gpt2::vocab_size"])
    replayed = SparseCounts(
        offsets=stored["tiny-gpt2::reference_counts_offsets"],
        token_ids=stored["tiny-gpt2::reference_unique_token_ids"],
        counts=stored["tiny-gpt2::reference_counts"],
    ).vocabulary_totals(vocab)
    assert list(replayed) == list(per_record.vocabulary_totals(vocab))


def test_the_bootstrap_stage_refuses_explicitly_without_the_file_and_runs_with_it(
    frozen_run,
):
    """The gap this file closes, stated as the consumer's own two behaviours.

    Without it ``41_context_information_bootstrap.py`` must say so -- a singleton
    reference grouping with its limitation declared, and no k-mer screen -- and
    with it both must be computed from the records rather than assumed away.
    """

    bootstrap = _stage("41_context_information_bootstrap.py", "_stage_41_reference")
    # load_block takes one draw's (sidecar, cohort_json, reference_json) triples
    # since the 2026-08-21 pairing repair, which keys a block on the cohort and
    # reference digests rather than on the producing invocation.
    absent = bootstrap.load_block(
        0,
        [(frozen_run["sidecar"], None, None)],
        requested_arms=None,
        containment=bootstrap.NEAR_DUPLICATE_CONTAINMENT,
        shingle=None,
    )
    assert absent.reference_records is None
    assert absent.reference_grouping["available"] is False
    assert "DECLARED LIMITATION" in absent.reference_grouping["declared_limitation"]
    assert absent.reference_grouping["n_groups"] == len(frozen_run["reference"])

    present = bootstrap.load_block(
        0,
        [(frozen_run["sidecar"], None, frozen_run["path"])],
        requested_arms=None,
        containment=bootstrap.NEAR_DUPLICATE_CONTAINMENT,
        shingle=None,
    )
    grouping = present.reference_grouping
    assert grouping["available"] is True
    assert "declared_limitation" not in grouping
    assert present.reference_records == frozen_run["reference"].records
    # The planted near-copy is joined to the record it copies, so the grouping is
    # a measurement over the reference rather than the singleton fallback under
    # another name.
    assert grouping["n_groups"] < grouping["n_records"] == len(frozen_run["reference"])
    assert grouping["largest_group_size"] == 2

    # ...and the k-mer screen sees exactly the planted leak, which is a near-copy
    # of a scored record and therefore invisible to an exact-content check.
    leaked = bootstrap.leaked_reference_mask(
        present.cohort_records or frozen_run["cohort"].records,
        present.reference_records,
        LEAK_KMER,
    )
    assert int(leaked.sum()) == 1
    assert leaked[-1]
