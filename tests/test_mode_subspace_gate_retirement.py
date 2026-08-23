"""The 0.30-nat behavioural-read pre-gate is retired, and may not come back.

``src.transfer.mode_subspaces`` used to refuse every behavioural cell of a mode
whose *declared* context information fell below a locally declared 0.30-nat floor,
before any of those cells was computed. The floor was underived -- it is the
constant catalogued at L41 and retired from the package by EXP-R2-218 -- and the
number it decided was measured on a different cohort from the one being ablated.

Retiring a guard is only safe if the failure it guarded against is caught
somewhere else. **That is the load-bearing claim these tests hold.** The failure
is "a licensed necessity claim on a mode whose ablation destroyed no
context-derived signal", and ``DECISION_RULES["residual_licensed_v1"]`` already
tests it by measurement: a licensed verdict requires, in *every* mode of the run,
that the residual non-unigram damage from ablating that mode's own necessary
subspace be positive with a paired group-bootstrap 95% interval excluding zero.

So the first test enumerates the whole clause space rather than picking a case,
and asserts the implication in the direction that matters: **no** assignment of
the clauses in which any mode's residual damage is unestablished may return
``DISTINCT_SUBSPACES`` or ``SHARED_SUBSPACE``. The rest hold the retirement itself
-- that the constant is gone from both files and cannot creep back as an operative
default, that the declared input is gone from the stage and from every campaign
that used to pass it, and that what replaced it is measured on this stage's own
cohort. The AST checks follow ``tests/test_measurability_criterion_contract.py``,
which is where this repository already reads criteria out of the syntax tree
rather than out of a text search: a stage that names a number in a docstring would
otherwise satisfy a text search without deciding anything, and a stage that
decides on one would evade a text search that looked for a name.
"""

from __future__ import annotations

import ast
import itertools
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer import mode_subspaces as ms  # noqa: E402

RULE = ms.decision_rule("residual_licensed_v1")

MODULE = REPO_ROOT / "src" / "transfer" / "mode_subspaces.py"
STAGE = REPO_ROOT / "scripts" / "transfer" / "38_mode_subspaces.py"

#: The campaigns that used to declare a per-mode figure on the command line.
CAMPAIGNS = sorted((REPO_ROOT / "scripts" / "transfer").glob("campaign_d2i_gpu*.tsv"))

#: The verdicts that assert a mode needs directions at a layer. Everything else
#: this rule can return names a clause that failed.
LICENSED = ("DISTINCT_SUBSPACES", "SHARED_SUBSPACE")


def _damage(*, difference: float, excludes_zero: bool) -> dict[str, object]:
    """One bootstrap interval, at a stated point estimate and a stated verdict."""

    if excludes_zero:
        interval = [difference * 0.5, difference * 1.5]
    else:
        interval = [-abs(difference) - 0.01, abs(difference) + 0.01]
    return {"difference": difference, "difference_ci95": interval}


def _own(*, total: bool, residual: bool, share: float) -> dict[str, object]:
    """One mode's own-subspace decomposition, at the clause values asked for."""

    return {
        "total": _damage(difference=0.4 if total else 0.0, excludes_zero=total),
        "residual": _damage(difference=0.3 if residual else 0.0, excludes_zero=residual),
        "residual_share": share,
    }


def _overlap(*, measured: float) -> dict[str, object]:
    upper = 0.2
    return {
        ms.MEAN_SQUARED_COSINE: measured,
        ms.FIRST_PRINCIPAL_ANGLE_COSINE: measured,
        "chance": {
            statistic: {"p97.5": upper, "closed_form_mean": 0.1}
            for statistic in ms.OVERLAP_STATISTICS
        },
    }


def _verdict(**kwargs) -> str:
    defaults = dict(
        layer=7,
        modes=("text", "protein"),
        invariants_held=True,
        rule=RULE,
        statistic=ms.MEAN_SQUARED_COSINE,
    )
    defaults.update(kwargs)
    return ms.layer_verdict(**defaults)["verdict"]


# ------------------------------- what must catch a mode with nothing to destroy


@pytest.mark.parametrize(
    "clauses",
    list(itertools.product([True, False], repeat=6)),
    ids=lambda clauses: "".join("1" if flag else "0" for flag in clauses),
)
def test_no_clause_assignment_licenses_a_verdict_without_residual_damage(clauses):
    """The implication the retired pre-gate is being traded for.

    Enumerated over every assignment of the six clauses the rule reads, in both
    modes, rather than demonstrated on one constructed case: the claim is that
    there is NO path to a licensed verdict without established residual damage in
    every mode, and a claim of that shape is not evidenced by an example.
    """

    text_total, text_residual, protein_total, protein_residual, asym, high = clauses
    own = {
        "text": _own(total=text_total, residual=text_residual, share=0.9),
        "protein": _own(total=protein_total, residual=protein_residual, share=0.9),
    }
    asymmetry = {
        mode: _damage(difference=0.2 if asym else 0.0, excludes_zero=asym)
        for mode in ("text", "protein")
    }
    verdict = _verdict(
        own=own,
        asymmetry=asymmetry,
        overlap=_overlap(measured=0.9 if high else 0.05),
        attainable={"text": True, "protein": True},
    )
    established = text_residual and protein_residual
    if not established:
        assert verdict not in LICENSED, (
            f"{verdict} was licensed with residual damage established in "
            f"{[m for m in own if own[m]['residual']['difference'] > 0]}"
        )
    assert verdict in ms.VERDICTS


def test_a_mode_with_no_damage_at_all_reads_no_measured_damage_and_not_a_subspace():
    """The verdict a signal-free mode actually lands on, and its reading.

    The pre-gate's job was to stop this cell from being computed. The rule's job
    is to say what it found, which is strictly more information: the artefact
    names the clause that failed instead of an admission decision taken before the
    measurement existed.
    """

    own = {
        "text": _own(total=True, residual=True, share=0.9),
        "protein": _own(total=False, residual=False, share=None),
    }
    asymmetry = {
        "text": _damage(difference=0.2, excludes_zero=True),
        "protein": _damage(difference=0.0, excludes_zero=False),
    }
    record = ms.layer_verdict(
        layer=7,
        modes=("text", "protein"),
        own=own,
        asymmetry=asymmetry,
        overlap=_overlap(measured=0.05),
        attainable={"text": True, "protein": True},
        invariants_held=True,
        rule=RULE,
        statistic=ms.MEAN_SQUARED_COSINE,
    )
    assert record["verdict"] == "NO_MEASURED_DAMAGE"
    assert record["clauses"]["total_damage_excludes_zero"] == {
        "text": True,
        "protein": False,
    }
    assert "is not a claim that the subspace is unnecessary" in record["reading"]


def test_a_mode_whose_whole_block_write_may_be_zeroed_harmlessly_is_void():
    """No attainable denominator is a statement about the site, before any clause.

    ``necessary_rank`` sets ``attainable`` from the full-block ablation, so this is
    the branch a mode reaches when there is nothing at the layer to destroy at all
    -- and it precedes the damage clauses, which is why a signal-free mode cannot
    reach a licensed verdict even with the residual clauses satisfied by noise.
    """

    withheld = ms.necessary_rank([1, 2, 4], [0.0, 0.0, 0.0], 0.0, RULE)
    assert withheld["attainable"] is False
    assert withheld["necessary_rank"] is None
    assert withheld["target_damage_nats"] is None
    assert "no attainable denominator" in withheld["withheld_reason"]

    verdict = _verdict(
        own={
            mode: _own(total=True, residual=True, share=0.9)
            for mode in ("text", "protein")
        },
        asymmetry={
            mode: _damage(difference=0.2, excludes_zero=True)
            for mode in ("text", "protein")
        },
        overlap=_overlap(measured=0.05),
        attainable={"text": True, "protein": withheld["attainable"]},
    )
    assert verdict == "VOID_INSTRUMENT"


def test_damage_that_is_mostly_the_marginal_shift_is_unigram_only_not_a_subspace():
    """The other way an ablation can destroy nothing that licenses a claim."""

    verdict = _verdict(
        own={
            mode: _own(total=True, residual=True, share=0.1)
            for mode in ("text", "protein")
        },
        asymmetry={
            mode: _damage(difference=0.2, excludes_zero=True)
            for mode in ("text", "protein")
        },
        overlap=_overlap(measured=0.05),
        attainable={"text": True, "protein": True},
    )
    assert verdict == "UNIGRAM_ONLY"
    assert "must not be reported as one" in ms.VERDICT_READINGS["UNIGRAM_ONLY"]


# ----------------------------------------- the constant is gone and stays gone


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


RETIRED_NAMES = (
    "MODE_BEHAVIOURAL_READ_FLOOR_NATS",
    "MODE_BEHAVIOURAL_READ_FLOOR_STATUS",
    "BEHAVIOURAL_READ_REFUSED",
    "UNMEASURABLE_MODE_EVIDENCE",
    "mode_measurability",
    "assert_behavioural_read",
)


@pytest.mark.parametrize("name", RETIRED_NAMES)
def test_no_retired_name_survives_in_the_module_namespace(name):
    assert not hasattr(ms, name), f"{name} is back in src.transfer.mode_subspaces"


@pytest.mark.parametrize("path", [MODULE, STAGE], ids=lambda path: path.name)
def test_no_retired_name_is_written_in_either_owned_file(path):
    # Including the sentinel's own string value: a verdict cell spelled
    # "BEHAVIOURAL_READ_REFUSED" would reintroduce the refusal into artefacts even
    # with the constant deleted.
    source = path.read_text(encoding="utf-8")
    for name in RETIRED_NAMES:
        assert name not in source, f"{path.name} still writes {name}"


@pytest.mark.parametrize("path", [MODULE, STAGE], ids=lambda path: path.name)
def test_the_retired_magnitude_is_never_an_operative_default_or_a_comparison(path):
    """A default and a comparison are the two operative forms of a criterion.

    The same reading ``test_the_retired_constant_is_never_an_operative_default``
    takes of ``MIN_CONTEXT_INFORMATION_NATS``, applied to the bare magnitude,
    because this gate's number was declared locally and a successor would declare
    it locally again rather than import a name.
    """

    tree = _tree(path)

    def is_the_magnitude(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Constant)
            and isinstance(node.value, float)
            and node.value == pytest.approx(0.30)
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if (func.attr if isinstance(func, ast.Attribute) else None) == "add_argument":
                for keyword in node.keywords:
                    if keyword.arg == "default":
                        assert not is_the_magnitude(keyword.value), (
                            f"{path.name} defaults a flag to 0.30"
                        )
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for default in [*node.args.defaults, *node.args.kw_defaults]:
                assert default is None or not is_the_magnitude(default), (
                    f"{path.name}: {node.name} takes 0.30 as a parameter default"
                )
        if isinstance(node, ast.Compare):
            for operand in [node.left, *node.comparators]:
                assert not is_the_magnitude(operand), (
                    f"{path.name} compares against a bare 0.30"
                )


def test_the_stage_declares_no_context_information_flag():
    """The declared per-mode figure is gone from the interface, not just unused.

    An accepted-but-ignored flag is the shape this repository refuses elsewhere:
    it keeps an unverifiable hand-typed literal in the artefact's settings block
    while nothing verifies it, which is worse than either using it or not
    accepting it.
    """

    tree = _tree(STAGE)
    declared = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (node.func.attr if isinstance(node.func, ast.Attribute) else None)
        == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    assert "--context-information" not in declared
    assert "--decision-rule" in declared, "the AST read found no flags at all"


@pytest.mark.parametrize("path", CAMPAIGNS, ids=lambda path: path.name)
def test_no_campaign_still_passes_the_retired_flag(path):
    # A campaign row that passes a flag the stage no longer declares dies at
    # argparse on the cluster rather than here.
    assert CAMPAIGNS, "no d2i campaign files were found to check"
    assert "--context-information" not in path.read_text(encoding="utf-8")


# -------------------------------- what is reported in the declared input's place


def test_context_information_is_computed_from_this_runs_own_two_operands():
    """Both estimators of the reference, and the correction between them.

    The quantity is the cohort's own target-count entropy minus the model's clean
    cross-entropy over the same scored positions. Reporting it at one estimator
    only would hide which one produced it, and L12 prices the plug-in estimator at
    up to +1.02 nats on a protein arm.
    """

    reference = {
        "plug_in_entropy_nats": 4.0,
        "miller_madow_entropy_nats": 4.25,
    }
    record = ms.cohort_context_information(reference, 3.5)
    assert record["context_information_plug_in_reference_nats"] == pytest.approx(0.5)
    assert record["context_information_miller_madow_reference_nats"] == pytest.approx(
        0.75
    )
    assert record["clean_cross_entropy_nats"] == pytest.approx(3.5)
    # The two differ by exactly the correction, so a reader can see the bias.
    assert (
        record["context_information_miller_madow_reference_nats"]
        - record["context_information_plug_in_reference_nats"]
    ) == pytest.approx(
        reference["miller_madow_entropy_nats"] - reference["plug_in_entropy_nats"]
    )
    # Negative is a real answer here and must not be clipped: a model worse than
    # the cohort's own symbol frequencies reads below zero, and that is the
    # finding rather than an error.
    assert ms.cohort_context_information(reference, 9.0)[
        "context_information_plug_in_reference_nats"
    ] == pytest.approx(-5.0)


def test_the_reported_quantity_says_it_is_neither_of_the_two_it_could_be_mistaken_for():
    note = ms.COHORT_CONTEXT_INFORMATION_NOTE
    assert "EXP-R2-152" in note and "is NOT that" in note
    assert "budget.arm_power" in note
    assert "gates on it" in note

    # The model-marginal quantity keeps its own prohibition, because
    # 21_joint_mode_qualification.py still holds a threshold on the corpus-
    # referenced one even though this stage holds a threshold on neither.
    marginal = ms.MODEL_MARGINAL_CONTEXT_INFORMATION_NOTE
    assert "different estimands" in marginal
    assert "21_joint_mode_qualification.py" in marginal
    assert "0.30" not in marginal


def test_the_stage_reports_a_measured_figure_per_mode_and_declares_none():
    source = STAGE.read_text(encoding="utf-8")
    assert '"cohort_context_information": ms.cohort_context_information(' in source
    assert "declared_context_information_nats" not in source
    # The model-marginal reading stays beside it; the two answer different
    # questions and dropping either would leave the other unqualified.
    assert "clean_context_information_against_model_marginal_nats" in source


# ------------------------------------------- the evidence states the catalogue


def test_the_low_signal_evidence_states_the_limitation_catalogue_correctly():
    """It claimed the catalogue ended at L32 and that there was no L33.

    Both halves were false when written into six published artefacts and into the
    message a refused run died with. The catalogue runs to L42, and the entry that
    covers the retired floor is L41.
    """

    audit = (REPO_ROOT / "docs" / "INTERPRETABILITY_TRANSFER_AUDIT.md").read_text(
        encoding="utf-8"
    )
    for row in ("| L33 |", "| L41 |", "| L42 |"):
        assert row in audit, f"the audit no longer carries {row.strip('| ')}"

    evidence = ms.LOW_SIGNAL_MODE_EVIDENCE
    assert "there is no L33" not in evidence
    assert "catalogue ends at L32" not in evidence
    assert "L41" in evidence and "L42" in evidence
    # It is evidence about a cohort, so it has to say it decides nothing.
    assert "decides nothing here" in evidence
    assert "0.0843" in evidence and "EXP-R2-152" in evidence


@pytest.mark.parametrize("path", [MODULE, STAGE], ids=lambda path: path.name)
def test_neither_owned_file_repeats_the_false_catalogue_claim(path):
    source = path.read_text(encoding="utf-8")
    assert "catalogue ends at L32" not in source
    assert "there is no L33" not in source


def test_the_evidence_reaches_the_artefact_as_a_limitation_and_not_as_a_gate():
    source = STAGE.read_text(encoding="utf-8")
    assert "ms.LOW_SIGNAL_MODE_EVIDENCE" in source
    limitations = source[source.index("LIMITATIONS: dict[str, Any] = {") :]
    assert "ms.LOW_SIGNAL_MODE_EVIDENCE" in limitations
