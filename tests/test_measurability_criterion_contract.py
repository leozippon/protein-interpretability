"""The two measurability criteria, at the call sites that apply them.

EXP-R2-218 split one undeclared 0.30-nat floor into two criteria with different
answers: identification, decided on the point estimate against
``budget.SCREENING_CONTEXT_INFORMATION_NATS``, and ratio admissibility, decided
per arm by ``budget.ratio_denominator_admissibility`` against the denominator's
own bootstrap standard error. ``budget.MIN_CONTEXT_INFORMATION_NATS`` survives
as a reporting column and decides nothing.

**Two failures motivate this file, and the suite saw neither.**

*Three entry points raised.* ``pathways.pathway_metrics`` gained a required
``context_information_se_nats`` and refuses -- correctly, and with no fallback --
without it. Three stages called it bare, each of them immediately before the
``pathway_cluster_bootstrap`` that publishes the very standard error it needed.
Nothing caught it because no test ever executed those three functions: the
library-level tests call ``pathway_metrics`` directly, and the stage-level ones
read declarations. The tests below run the three stage functions on a real
two-layer GPT-2 over real cohort rows, because a call site that is only reasoned
about is a call site that is not tested.

*The retired constant kept deciding.* Seven sites still took it as an operative
default after the split. A stage added tomorrow can reintroduce that in one line,
so the static contracts below scope over every live directory rather than over
the seven that were known.

GPU accounting is monkeypatched, and nothing else is: ``measure_arm`` resets and
reads CUDA peak-memory counters, which have no CPU equivalent and enter no
measured quantity. Every cross-entropy, every bootstrap draw and every verdict
below comes out of a real forward pass.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
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

from src.transfer import budget, scaling  # noqa: E402
from src.transfer.arms import PANEL, Arm, Cohort  # noqa: E402
from src.transfer.pathways import (  # noqa: E402
    mlp_all,
    pathway_cluster_bootstrap,
    pathway_metrics,
    prepare_batches,
)

STAGE_DIR = REPO_ROOT / "scripts" / "transfer"

#: Every directory whose modules can apply a measurability criterion.
SEARCH_DIRS = (
    REPO_ROOT / "scripts" / "transfer",
    REPO_ROOT / "scripts" / "transfer_gap",
    REPO_ROOT / "src" / "transfer",
)


def _load_stage(filename: str):
    path = STAGE_DIR / filename
    spec = importlib.util.spec_from_file_location(f"_criterion_stage_{filename[:2]}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------- a real arm

ALPHABET = "ABCDEFGHIJ"
VOCAB = 16
MAX_LEN = 40


class _CharTokenizer:
    """One id per character, ids offset past the pad id."""

    pad_token_id = 0

    def __call__(self, text: str, return_tensors=None) -> dict[str, list[int]]:
        return {"input_ids": [ALPHABET.index(character) + 1 for character in text]}

    def decode(self, ids) -> str:
        return "".join(
            ALPHABET[int(value) - 1] for value in ids if 1 <= int(value) <= len(ALPHABET)
        )


@pytest.fixture(scope="module")
def tiny_arm() -> Arm:
    """A real two-layer GPT-2 on the CPU, declared as a panel text arm.

    Small enough to run a whole stage inside a unit test and real enough that
    every cross-entropy under test comes from a forward pass.
    """

    config = GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=8,
        vocab_size=VOCAB,
        n_positions=64,
        attn_implementation="eager",
    )
    torch.manual_seed(11)
    spec = replace(PANEL["gpt2"], name="tiny-gpt2", n_layer=2, d_model=8)
    return Arm(
        spec=spec,
        model=GPT2LMHeadModel(config).eval(),
        tokenizer=_CharTokenizer(),
        device="cpu",
        dtype="float32",
        attn_implementation="eager",
    )


@pytest.fixture(scope="module")
def tiny_pool() -> Cohort:
    generator = np.random.default_rng(5)
    letters = np.array(list(ALPHABET))
    weights = np.linspace(1.0, 4.0, len(ALPHABET))
    probabilities = weights / weights.sum()
    records = [
        "".join(
            generator.choice(
                letters, size=int(generator.integers(14, MAX_LEN)), p=probabilities
            )
        )
        for _ in range(16)
    ]
    return Cohort(
        name="tiny-text", kind="text", records=records, min_symbols=14, max_symbols=MAX_LEN
    )


@pytest.fixture
def cpu_accelerator_accounting(monkeypatch):
    """CUDA peak-memory bookkeeping, neutralised so a CPU test can run a GPU stage.

    These four calls record how much accelerator memory a run used. They are not
    inputs to any measured quantity, and a stage cannot be executed on the CPU
    with them live.
    """

    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *a, **k: 0)


# ------------------------------------------------- the three broken entry points


def _assert_metrics_carry_their_own_bootstrap_error(metrics, bootstrap) -> None:
    """The point metrics and the interval beside them share one standard error.

    The whole repair is that the bootstrap runs first and hands its estimate to
    ``pathway_metrics``. If the two ever disagree, a caller has estimated the
    denominator's precision twice and published a criterion read off one of them
    beside an interval built from the other.
    """

    error = bootstrap["context_information_se_nats"]
    assert error > 0.0
    assert metrics["context_information_se_nats"] == pytest.approx(error)
    admissibility = metrics["context_information_admissibility"]
    assert admissibility["criterion"] == "fieller_precondition_on_the_denominator"
    assert admissibility["minimum_admissible_context_information"][
        "nats_per_token"
    ] == pytest.approx(budget.FIELLER_DENOMINATOR_MULTIPLE * error)
    # The retired constant is present and inert.
    assert admissibility["legacy_minimum_context_information_nats"] == pytest.approx(0.30)


def test_pathway_budget_measures_an_arm_without_raising(
    tiny_arm, tiny_pool, cpu_accelerator_accounting
):
    """02_pathway_budget.measure_arm, executed. It raised before the reorder."""

    stage = _load_stage("02_pathway_budget.py")
    args = argparse.Namespace(
        depths=[0.5],
        window=1,
        unigram_entropy=[],
        max_len=MAX_LEN,
        n_seq=8,
        batch_size=4,
        baseline="cohort_mean",
        unigram_estimator="plugin",
        minimum_ce_delta_nats=0.001,
        minimum_kl_nats=0.0,
        bootstrap_samples=16,
        bootstrap_seed=7,
        seeds=[1, 2],
        seed_target_error_fraction=0.5,
        protein_source="ec_labelled_swissprot",
    )
    record = stage.measure_arm(tiny_arm, tiny_pool, None, args)

    assert record["scopes"]
    for scope in record["scopes"]:
        for entry in scope["seeds"]:
            _assert_metrics_carry_their_own_bootstrap_error(
                entry["metrics"], entry["cluster_bootstrap"]
            )


def test_estimand_power_measures_an_arm_without_raising(
    tiny_arm, tiny_pool, cpu_accelerator_accounting
):
    """03_estimand_power.measure_arm, executed. It raised before the reorder."""

    stage = _load_stage("03_estimand_power.py")
    args = argparse.Namespace(
        depths=[0.5],
        widths=[1],
        baselines=["cohort_mean"],
        max_len=MAX_LEN,
        n_seq=8,
        seed=1,
        batch_size=4,
        unigram_estimator="plugin",
        unigram_entropy_nats=None,
        minimum_ce_delta_nats=0.0,
        minimum_kl_nats=0.0,
        bootstrap_samples=16,
        bootstrap_seed=7,
        protein_source="ec_labelled_swissprot",
    )
    record = stage.measure_arm(tiny_arm, tiny_pool, None, args)

    assert record["estimands"]
    for row in record["estimands"]:
        _assert_metrics_carry_their_own_bootstrap_error(
            row["metrics"], row["cluster_bootstrap"]
        )


def test_convergence_control_measures_pathway_shares_without_raising(
    tiny_arm, tiny_pool
):
    """07_convergence_control.measure_pathway_shares, executed.

    It also has to publish the standard error at the record level, because
    ``scaling.analysis_frame`` cannot evaluate this rung's denominator without
    it and must not substitute a constant.
    """

    stage = _load_stage("07_convergence_control.py")
    args = argparse.Namespace(
        max_len=MAX_LEN,
        batch_size=4,
        unigram_estimator="plugin",
        bootstrap_samples=16,
        bootstrap_seed=7,
    )
    shares = stage.measure_pathway_shares(tiny_arm, tiny_pool, None, {}, args)

    for scope in ("mlp_all", "attn_all"):
        _assert_metrics_carry_their_own_bootstrap_error(
            shares["metrics"][scope], shares["cluster_bootstrap"][scope]
        )
    assert shares["context_information_se_nats"] == pytest.approx(
        shares["cluster_bootstrap"]["mlp_all"]["context_information_se_nats"]
    )
    # The sign test is kept, under a name that cannot be mistaken for the
    # Fieller flag that carries the same word elsewhere.
    assert "context_information_valid" not in shares
    assert isinstance(shares["context_information_positive"], bool)


# ------------------------------------------------ a ratio site with no SE refuses


def test_a_ratio_site_with_no_standard_error_refuses_rather_than_falling_back(
    tiny_arm, tiny_pool
):
    """Real rows, no standard error: the criterion says so instead of guessing.

    ``pathway_metrics`` raises, which is what broke the three stages, and the
    exception names the missing quantity rather than reaching for the constant.
    """

    cohort = Cohort(
        name="tiny-text",
        kind="text",
        records=tiny_pool.records[:8],
        min_symbols=tiny_pool.min_symbols,
        max_symbols=tiny_pool.max_symbols,
    )
    batches = prepare_batches(tiny_arm, cohort, max_len=MAX_LEN, batch_size=4)
    from src.transfer.pathways import build_baseline, measure_pathways

    scope = mlp_all()
    bank = build_baseline(
        tiny_arm,
        batches,
        scope.resolve(tiny_arm.n_layer),
        kind="cohort_mean",
        cohort_digest=cohort.digest,
    )
    run = measure_pathways(tiny_arm, batches, [scope], bank)
    rows = run.rows_by_scope[scope.name]

    with pytest.raises(ValueError, match="no fallback"):
        pathway_metrics(rows, unigram_entropy_nats=2.0)
    with pytest.raises(ValueError, match="finite and strictly"):
        pathway_metrics(rows, unigram_entropy_nats=2.0, context_information_se_nats=0.0)

    # And with one, the same rows produce a verdict read off that number.
    bootstrap = pathway_cluster_bootstrap(
        rows, samples=32, seed=4, unigram_entropy_nats=2.0
    )
    metrics = pathway_metrics(
        rows,
        unigram_entropy_nats=2.0,
        context_information_se_nats=bootstrap["context_information_se_nats"],
    )
    _assert_metrics_carry_their_own_bootstrap_error(metrics, bootstrap)


def _convergence_record(*, standard_error: float | None) -> dict:
    """One ``analysis_frame`` input row, built through ``scaling.convergence_row``."""

    convergence = scaling.convergence_row(
        {
            "nats": 3.0,
            "cohort_plug_in_entropy_nats": 2.9,
            "estimator": "disjoint",
            "source": "held_out",
            "reference": {},
        },
        clean_ce_nats=1.0,
        context_information_se_nats=standard_error,
        symbols_per_token=1.0,
        n_scored_tokens=512,
        vocab_size=VOCAB,
        n_parameters=1000,
        n_layer=2,
        d_model=8,
    )
    return {
        "name": "tiny-gpt2",
        "modality": "text",
        "tokenisation": "bpe",
        "input_format": "raw",
        "source": "openwebtext",
        "architecture": "gpt2",
        "capabilities": ["budget", "pathway"],
        "cohort_min_symbols": 14,
        "cohort_max_symbols": MAX_LEN,
        "cohort": {"source": "openwebtext", "digest": "0" * 8},
        "convergence": convergence,
    }


def test_the_convergence_frame_withholds_a_denominator_verdict_it_cannot_take():
    """``analysis_frame``'s ratio flag is Fieller, and has no constant to fall to.

    A rung with a standard error gets a verdict read off its own precision. A
    rung without one gets ``None`` and a named reason -- never ``True`` and never
    ``False``, because both would be a magnitude constant deciding a ratio, which
    is the failure EXP-R2-218 measured.
    """

    admissible = scaling.analysis_frame([_convergence_record(standard_error=0.05)])[0]
    assert admissible["measurable_denominator"] is True
    assert admissible["denominator_admissibility"]["fieller_denominator_multiple"] == (
        pytest.approx(budget.FIELLER_DENOMINATOR_MULTIPLE)
    )
    assert admissible["denominator_admissibility_unavailable_reason"] is None

    # 2.0 nats of context information against a 0.30-nat standard error: above
    # the retired constant by 6.6x and still inadmissible on its own precision.
    refused = scaling.analysis_frame([_convergence_record(standard_error=0.30)])[0]
    assert refused["context_information_nats"] == pytest.approx(2.0)
    assert refused["measurable_denominator"] is False
    assert refused["denominator_admissibility"]["clears_legacy_floor"] is True

    withheld = scaling.analysis_frame([_convergence_record(standard_error=None)])[0]
    assert withheld["measurable_denominator"] is None
    assert withheld["denominator_admissibility"] is None
    assert (
        withheld["denominator_admissibility_unavailable_reason"]
        == scaling.NO_DENOMINATOR_STANDARD_ERROR
    )
    assert "no fallback" in scaling.NO_DENOMINATOR_STANDARD_ERROR

    # A row no longer carries a floor in nats at all, because the criterion is a
    # per-arm bound and a constant cannot express it.
    assert "denominator_floor_nats" not in withheld

    with pytest.raises(ValueError, match="finite and strictly"):
        _convergence_record(standard_error=0.0)


# ----------------------------------------------- the screening sites take 0.05


#: Stages whose measurability flag screens an arm in, and the option that sets it.
SCREENING_SITES = {
    "01_cohort_power.py": "--threshold-nats",
    "08_lens_family.py": "--minimum-context-information-nats",
    "41_context_information_bootstrap.py": "--threshold-nats",
}

#: Stages that deliberately screen against a locally declared magnitude instead,
#: with the module-level name that declares it. Each must record its own
#: UNDERIVED status, because a number that decides a verdict and was never
#: derived has to say so where it is used.
LOCALLY_DECLARED_FLOORS = {
    "scripts/transfer/21_joint_mode_qualification.py": (
        "JOINT_MODE_QUALIFICATION_FLOOR_NATS",
        "JOINT_MODE_QUALIFICATION_FLOOR_STATUS",
    ),
    "src/transfer/mode_subspaces.py": (
        "MODE_BEHAVIOURAL_READ_FLOOR_NATS",
        "MODE_BEHAVIOURAL_READ_FLOOR_STATUS",
    ),
}


def _argparse_default(tree: ast.Module, option: str) -> ast.AST | None:
    """The ``default=`` expression of one ``add_argument`` call, or None.

    Read from the call rather than by splitting the source, following
    ``tests/test_cohort_draw_contract.py``: a stage that names its own option in
    a docstring would otherwise satisfy a text search without defining anything.
    """

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (func.attr if isinstance(func, ast.Attribute) else None) != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        if node.args[0].value != option:
            continue
        for keyword in node.keywords:
            if keyword.arg == "default":
                return keyword.value
    return None


@pytest.mark.parametrize("filename,option", sorted(SCREENING_SITES.items()))
def test_a_screening_site_defaults_to_the_calibrated_identification_floor(
    filename, option
):
    """Identification is a detection question and 0.30 was 20-30x too strict for it."""

    tree = ast.parse((STAGE_DIR / filename).read_text(encoding="utf-8"))
    default = _argparse_default(tree, option)
    assert default is not None, f"{filename} no longer defines {option}"
    assert isinstance(default, ast.Name), (
        f"{filename} {option} takes a literal default; the criterion must be named"
    )
    assert default.id == "SCREENING_CONTEXT_INFORMATION_NATS", (
        f"{filename} {option} defaults to {default.id}"
    )


@pytest.mark.parametrize("relative,names", sorted(LOCALLY_DECLARED_FLOORS.items()))
def test_a_locally_declared_floor_declares_that_it_is_underived(relative, names):
    """0.30 may survive only where it is declared, and only saying it is underived."""

    magnitude_name, status_name = names
    path = Path(relative)
    module = (
        _load_stage(path.name)
        if path.parts[0] == "scripts"
        else importlib.import_module("src.transfer.mode_subspaces")
    )
    assert getattr(module, magnitude_name) == pytest.approx(0.30)
    status = getattr(module, status_name)
    assert status.startswith("UNDERIVED")
    # A status that does not say what would retire it is a label, not a
    # declaration.
    assert "standard error" in status


# ------------------------------------------- the retired constant decides nothing


def _live_modules() -> dict[str, ast.Module]:
    trees: dict[str, ast.Module] = {}
    for directory in SEARCH_DIRS:
        for path in sorted(directory.glob("*.py")):
            relative = str(path.relative_to(REPO_ROOT))
            trees[relative] = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
    return trees


def _is_retired_constant(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "MIN_CONTEXT_INFORMATION_NATS"
    if isinstance(node, ast.Attribute):
        return node.attr == "MIN_CONTEXT_INFORMATION_NATS"
    return False


def test_the_retired_constant_is_never_an_operative_default():
    """It is a reporting column. A default is the operative form of a criterion.

    Seven sites took it as one after the split -- five screening flags, a
    denominator floor and a library keyword -- and every one of them was a
    criterion nobody had chosen, applied under a name that says it was retired.
    """

    offenders: list[str] = []
    for relative, tree in _live_modules().items():
        if relative == "src/transfer/budget.py":
            continue  # its own declaration
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if (func.attr if isinstance(func, ast.Attribute) else None) == (
                    "add_argument"
                ):
                    option = (
                        node.args[0].value
                        if node.args and isinstance(node.args[0], ast.Constant)
                        else "?"
                    )
                    for keyword in node.keywords:
                        if keyword.arg == "default" and _is_retired_constant(
                            keyword.value
                        ):
                            offenders.append(
                                f"{relative}:{node.lineno} {option} default="
                                "MIN_CONTEXT_INFORMATION_NATS"
                            )
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defaults = list(node.args.defaults) + [
                    value for value in node.args.kw_defaults if value is not None
                ]
                for value in defaults:
                    if _is_retired_constant(value):
                        offenders.append(
                            f"{relative}:{node.lineno} {node.name}() takes "
                            "MIN_CONTEXT_INFORMATION_NATS as a parameter default"
                        )
    assert offenders == [], (
        "MIN_CONTEXT_INFORMATION_NATS is a legacy reporting column and decides "
        "nothing (EXP-R2-218, transfer audit 5.08(f)). Offenders:\n  "
        + "\n  ".join(offenders)
    )


def _publication_label(tree: ast.Module, node: ast.AST) -> str:
    """The name a node's value is published under, found by walking outwards.

    Walking outwards rather than inspecting the immediate parent is what closes
    the obvious hole: ``bool(I >= C)`` inside a dict entry has a ``Call`` for a
    parent and the key that names it one level further out.
    """

    parents: dict[int, tuple[ast.AST, str, int]] = {}
    for parent in ast.walk(tree):
        for field, value in ast.iter_fields(parent):
            items = value if isinstance(value, list) else [value]
            for index, child in enumerate(items):
                if isinstance(child, ast.AST):
                    parents[id(child)] = (parent, field, index)

    current: ast.AST | None = node
    while current is not None and id(current) in parents:
        parent, field, index = parents[id(current)]
        if isinstance(parent, ast.Dict) and field == "values":
            key = parent.keys[index]
            if isinstance(key, ast.Constant):
                return str(key.value)
        if isinstance(parent, ast.Assign):
            targets = [t.id for t in parent.targets if isinstance(t, ast.Name)]
            if targets:
                return ",".join(targets)
        if isinstance(parent, ast.keyword) and parent.arg:
            return parent.arg
        current = parent
    return "<unpublished>"


def test_the_retired_constant_is_only_ever_compared_under_a_legacy_name():
    """Every surviving comparison against it must publish a legacy column.

    A default is not the only operative form. A bare ``I >= 0.30`` decides just
    as much, so the comparison is admitted only where its result is stored under
    a name that says which number produced it.
    """

    offenders: list[str] = []
    for relative, tree in _live_modules().items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            operands = [node.left] + list(node.comparators)
            if not any(_is_retired_constant(item) for item in operands):
                continue
            label = _publication_label(tree, node)
            if "legacy" not in label.lower():
                offenders.append(
                    f"{relative}:{node.lineno} compares against the retired "
                    f"constant under {label!r}"
                )
    assert offenders == [], (
        "a comparison against MIN_CONTEXT_INFORMATION_NATS decides something "
        "unless it is published as the legacy column it is. Offenders:\n  "
        + "\n  ".join(offenders)
    )


def test_every_pathway_metrics_call_site_supplies_the_denominators_error():
    """The static half of the repair, so stage N+1 cannot reintroduce it.

    ``pathway_metrics`` refuses without ``context_information_se_nats`` and the
    refusal is right, so the defect is not a wrong number in a run -- it is a
    keyword nobody passed, which is exactly the shape a call-site scan catches.
    """

    offenders: list[str] = []
    for relative, tree in _live_modules().items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr
                if isinstance(func, ast.Attribute)
                else None
            )
            if name != "pathway_metrics":
                continue
            if not any(
                keyword.arg == "context_information_se_nats" for keyword in node.keywords
            ):
                offenders.append(f"{relative}:{node.lineno}")
    assert offenders == [], (
        "pathway_metrics applies the Fieller precondition and has no fallback; "
        "a call site without context_information_se_nats raises at run time. "
        "Offenders:\n  " + "\n  ".join(offenders)
    )
