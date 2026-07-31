"""Regression tests for the 2026-07-29 full-scope audit of ``src.transfer``.

Every test here corresponds to one defect that was found and fixed, and each is
written against the *property* the fix restores rather than against the fix's
implementation.  The organising principle is the one the transfer audit's
Appendix B states: the failures this programme has paid for were not crashes,
they were plausible numbers produced under wrong assumptions, so what has to be
asserted is that the wrong assumption is now either impossible or loudly
declared.

Grouped by the module they defend.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import math
import sys
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer import (  # noqa: E402
    channels,
    circuits,
    homology,
    induction_robustness,
    path_patching,
    pathways,
    prediction_addressed,
    probes,
    relational,
    statistics,
)
from src.transfer.arms import (  # noqa: E402
    CONDITIONING_END,
    CONDITIONING_START,
    MATCHED_DATA_CONTRAST,
    PANEL,
    PRETRAINING_UNDECLARED,
    SAMPLING_MODES,
    TEXT_ARCHITECTURE_CONTRAST,
    TEXT_DATA_CONTRAST,
    Arm,
    Cohort,
    sampling_record,
    selected_positions,
    tokenize_batch,
)
from src.transfer.circuits import (  # noqa: E402
    PROTEIN_APPROXIMATE_CRITERION,
    RepeatHit,
    RepeatProbe,
    Unigram,
    _select_matching,
    summarise_head_matrix,
    summarise_patching,
)
from src.transfer.homology import (  # noqa: E402
    MINIMUM_BOOTSTRAP_UNITS,
    Hit,
    HomologyAssignment,
    assign_homology,
    bootstrap_stratum,
    count_fasta_records,
    stratum_integrity,
    sub_cohort,
    truncated_alignment,
)
from src.transfer.homology import (  # noqa: E402
    covariate_analysis as homology_covariate_analysis,
)
from src.transfer.homology import ov_over_heads  # noqa: E402
from src.transfer import budget, io as io_module  # noqa: E402
from src.transfer.io import write_json  # noqa: E402
from src.transfer.pathways import (  # noqa: E402
    disjoint_unigram_cross_entropy_nats,
    pathway_cluster_bootstrap,
    pathway_metrics,
    smoothing_diagnostics,
)
from src.transfer.prediction_addressed import cluster_bootstrap  # noqa: E402
from src.transfer.scaling import (  # noqa: E402
    DEFAULT_LADDER,
    nearest_neighbour_contrasts,
)
from src.transfer.scoring import (  # noqa: E402
    analysis_layer,
    analysis_layers,
    sequence_target_mask,
    target_rule,
)


# --------------------------------------------------------------------- helpers


def _stub_arm(name: str = "gpt2-large", **overrides) -> Arm:
    """An ``Arm`` with no model, for the code paths that only read declarations."""

    spec = replace(PANEL[name], **overrides) if overrides else PANEL[name]
    return Arm(
        spec=spec,
        model=None,
        tokenizer=None,
        device="cpu",
        dtype="float32",
        attn_implementation=overrides.pop("attn_implementation", "eager"),
    )


class _WordTokenizer:
    """Whitespace tokenizer: enough for ``tokenize_batch``'s contract."""

    pad_token_id = 0

    def __call__(self, text: str, return_tensors=None) -> dict[str, list[int]]:
        return {"input_ids": [len(word) for word in text.split()]}


# ------------------------------------------------------------- arms: the panel


def test_source_is_the_evaluation_cohort_and_pretraining_is_a_separate_field():
    """``source`` reads as provenance; it never was, so both facts are declared.

    The regression this guards: every text arm carried ``source="openwebtext"``,
    which is true of the evaluation cohort and false of the pretraining data for
    six of the seven text arms.
    """

    for name in ("gpt2-large", "dialogpt-small", "qwen2.5-0.5b", "llama-3.2-3b"):
        spec = PANEL[name]
        assert spec.source == spec.evaluation_cohort_source
        assert spec.evaluation_cohort_source == "openwebtext"
    assert PANEL["gpt2-large"].pretraining_corpus == "webtext"
    assert PANEL["dialogpt-small"].pretraining_corpus == "reddit_dialogue"
    assert PANEL["qwen2.5-0.5b"].pretraining_corpus != "openwebtext"
    assert PANEL["llama-3.2-3b"].pretraining_corpus != "openwebtext"


@pytest.mark.parametrize("pair", [MATCHED_DATA_CONTRAST, TEXT_DATA_CONTRAST])
def test_corpus_contrasts_vary_the_pretraining_corpus_and_nothing_else(pair):
    """A corpus contrast that varies nothing is not a contrast."""

    left, right = (PANEL[name] for name in pair)
    for field in ("modality", "n_layer", "d_model", "tokenisation", "architecture"):
        assert getattr(left, field) == getattr(right, field)
    assert left.pretraining_corpus != right.pretraining_corpus
    assert PRETRAINING_UNDECLARED not in (left.pretraining_corpus, right.pretraining_corpus)


def test_the_architecture_contrast_actually_leaves_the_gpt2_lineage():
    """The declaration behind §5.05(d), which was previously read by nothing.

    A GPT-2-architecture arm in this tuple would leave the claim "it survives
    replacing GPT-2-large with a Qwen2 and a Llama decoder" stated and
    unsupported. That failure mode is not hypothetical: it is what killed the
    QK/OV dissociation, which turned out to be a GPT-2-lineage property.
    """

    assert TEXT_ARCHITECTURE_CONTRAST
    for name in TEXT_ARCHITECTURE_CONTRAST:
        assert PANEL[name].modality == "text", name
        assert PANEL[name].architecture != "gpt2", name


def test_a_sampling_mode_outside_the_declaration_is_refused():
    """A mode with no declared hazard text is a draw no reader can interpret."""

    for seed in (None, 7):
        assert sampling_record(
            seed=seed, skip=0, requested=4, eligible=100, corpus="c"
        )["mode"] in SAMPLING_MODES


def test_undeclared_pretraining_corpus_is_a_sentinel_not_a_guess():
    """An arm with no model card gets ``undeclared``, never a plausible corpus."""

    assert PANEL["bygpt5-small-en"].pretraining_corpus == PRETRAINING_UNDECLARED


def test_ladder_declares_both_corpora():
    for member in DEFAULT_LADDER:
        assert member.source == member.evaluation_cohort_source
        assert member.pretraining_corpus


# --------------------------------------------------------- arms: cohort sampling


def test_seeded_draws_at_different_skips_are_disjoint():
    """A skip-offset sensitivity is only a sensitivity if the offsets do not overlap.

    Under file order ``skip`` walks further down one file; under a seeded
    permutation it must take a different, non-overlapping part of the same
    corpus.
    """

    first = selected_positions(200, n=20, skip=0, seed=11, label="t")
    second = selected_positions(200, n=20, skip=20, seed=11, label="t")
    assert len(first) == len(second) == 20
    assert not set(first) & set(second)
    assert set(first) | set(second) <= set(range(200))


def test_file_order_mode_reproduces_the_historical_prefix():
    """Frozen artefacts stay reproducible; the mode is what changes, not the maths."""

    assert selected_positions(50, n=5, skip=0, seed=None, label="t") == [0, 1, 2, 3, 4]
    assert selected_positions(50, n=5, skip=5, seed=None, label="t") == [5, 6, 7, 8, 9]


def test_a_seeded_draw_is_not_the_file_order_prefix():
    assert selected_positions(500, n=25, skip=0, seed=7, label="t") != list(range(25))


def test_selected_positions_refuses_a_corpus_that_cannot_supply_the_draw():
    with pytest.raises(RuntimeError, match="eligible records cannot supply"):
        selected_positions(10, n=8, skip=5, seed=1, label="t")


def test_file_order_sampling_carries_its_hazard_and_seeded_sampling_does_not():
    """The invisibility of the draw is what manufactured an effect three times."""

    file_order = sampling_record(
        seed=None, skip=0, requested=10, eligible=None, corpus="plain_swissprot"
    )
    seeded = sampling_record(
        seed=4, skip=0, requested=10, eligible=900, corpus="plain_swissprot"
    )
    assert file_order["mode"] == "file_order"
    assert "seeded permutation" in file_order["hazard"]
    assert seeded["mode"] == "seeded_permutation"
    assert "hazard" not in seeded
    assert seeded["eligible_records"] == 900


def test_a_cohort_without_a_sampling_record_says_so():
    cohort = Cohort("c", "protein", ["AAAA"], 1, 10, {})
    assert cohort.sampling["mode"] == "unrecorded"


def test_provenance_digest_separates_cohorts_that_the_content_digest_cannot():
    """Nested repeat cohorts can share records and still be different measurements."""

    records = ["ACDEFGHIKL", "MNPQRSTVWY"]
    exact = Cohort("c", "protein", records, 1, 100, {"criterion": {"kind": "exact"}})
    approximate = Cohort(
        "c", "protein", records, 1, 100, {"criterion": {"kind": "approximate"}}
    )
    assert exact.digest == approximate.digest
    assert exact.provenance_digest != approximate.provenance_digest


def test_provenance_digest_survives_unserialisable_metadata():
    cohort = Cohort("c", "protein", ["AAAA"], 1, 10, {"array": np.arange(3)})
    assert isinstance(cohort.provenance_digest, str)


# ------------------------------------------------------------ arms: batch shape


def test_tokenize_batch_refuses_a_row_that_produces_no_tokens():
    """A fully masked row contributes to every mean without appearing as dropped."""

    arm = _stub_arm()
    arm.tokenizer = _WordTokenizer()
    with pytest.raises(ValueError, match="tokenise to no tokens"):
        tokenize_batch(arm, ["one two", ""], 8)


def test_tokenize_batch_refuses_an_empty_batch():
    arm = _stub_arm()
    arm.tokenizer = _WordTokenizer()
    with pytest.raises(ValueError, match="empty batch"):
        tokenize_batch(arm, [], 8)


# ------------------------------------------------- arms: attention-kernel contract


@pytest.mark.parametrize("implementation", [None, "sdpa", "flash_attention_2"])
def test_a_non_eager_kernel_cannot_enter_a_pattern_measurement(implementation):
    """Reading a pattern already failed; *overriding* one would have been accepted."""

    arm = _stub_arm()
    arm.attn_implementation = implementation
    with pytest.raises(ValueError, match="eager"):
        arm.require_eager_attention("a pattern measurement")


def test_eager_passes_the_attention_contract():
    _stub_arm().require_eager_attention("a pattern measurement")


# ------------------------------------------------------------------- scoring


def test_the_scored_target_rule_follows_the_input_format_not_the_arm_name():
    """The predecessor of this rule dispatched on the literal string ``"zymctrl"``.

    For any *other* EC-conditioned arm it returned the plain validity mask and
    silently discarded the boundary ids it was handed, so the EC tag, the
    separator and the terminator -- the positions the model predicts most
    confidently -- were scored as cohort content. Clean cross-entropy falls,
    context information rises, and nothing raises.
    """

    for name, spec in PANEL.items():
        expected = (
            "between_boundaries"
            if spec.input_format == "ec_conditioned"
            else "all_valid"
        )
        assert target_rule(spec.input_format) == expected, name

    # A second EC-conditioned arm, whatever it is called, gets the boundary rule.
    future = replace(PANEL["zymctrl"], name="zymctrl-v2")
    assert target_rule(future.input_format) == "between_boundaries"


def test_an_unconditioned_rendering_takes_the_plain_rule():
    """The unconditioned mode strips the tag, so there are no boundaries to find."""

    assert target_rule("ec_conditioned", ec_conditioning="unconditioned") == "all_valid"
    assert target_rule("ec_conditioned", ec_conditioning="native") == "between_boundaries"
    assert target_rule("ec_conditioned", ec_conditioning="fixed") == "between_boundaries"


def test_the_boundary_rule_refuses_rather_than_scoring_the_whole_prompt():
    ids = torch.tensor([[9, 1, 5, 6, 2, 0]])
    mask = torch.ones_like(ids)
    with pytest.raises(ValueError, match="requires start and end token IDs"):
        sequence_target_mask(ids, mask, rule="between_boundaries")


def test_the_plain_rule_refuses_boundary_ids_instead_of_ignoring_them():
    """Accepting-and-discarding them is exactly how the predecessor failed."""

    ids = torch.tensor([[9, 1, 5, 6, 2, 0]])
    mask = torch.ones_like(ids)
    with pytest.raises(ValueError, match="ignores them"):
        sequence_target_mask(
            ids, mask, rule="all_valid", start_token_id=1, end_token_id=2
        )


def test_the_boundary_rule_scores_only_the_span_between_the_markers():
    #                   position: 0  1  2  3  4  5
    ids = torch.tensor([[9, 1, 5, 6, 2, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
    keep = sequence_target_mask(
        ids, mask, rule="between_boundaries", start_token_id=1, end_token_id=2
    )
    # Target column q predicts token q+1: columns 1 and 2 predict tokens 5 and 6.
    assert keep.tolist() == [[False, True, True, False, False]]


def test_an_unknown_target_rule_raises_rather_than_defaulting():
    ids = torch.tensor([[9, 1, 5, 6]])
    with pytest.raises(ValueError, match="unknown target rule"):
        sequence_target_mask(ids, torch.ones_like(ids), rule="whatever")


def test_one_depth_conversion_governs_every_stage():
    """Two conversions coexisted and disagreed on the ProGen2 arms at depth 0.25.

    ``src.transfer.relational`` carried ``int(round(f * (n_layer - 1)))``.
    Python's ``round`` is round-half-to-even, so at ``0.25 * 26 = 6.5`` -- exact
    in binary, and exactly the 27-layer ProGen2 case -- it returned layer 6 while
    ``analysis_layer`` returned 7. ``09_probe_and_erasure.py`` reached the first
    and ``02``/``03``/``08`` reached the second.
    """

    assert not hasattr(relational, "analysis_layers")
    assert analysis_layer(27, 0.25) == 7
    assert analysis_layers(27, (0.25, 0.5, 0.75)) == [7, 13, 20]
    # Half-integer depths round up, never to the even neighbour.
    for n_layers in (27, 35):
        for fraction in (0.25, 0.5, 0.75):
            exact = fraction * (n_layers - 1)
            if exact == int(exact) + 0.5:
                assert analysis_layer(n_layers, fraction) == int(exact) + 1


def _depth_conversion_sites(path: Path) -> list[str]:
    """Calls that turn a fractional depth into a layer index without ``analysis_layer``.

    Matches the shape rather than a spelling: any ``int``/``round``/``floor``
    call whose argument subtree mentions ``n_layer(s) - 1``. That catches all
    three conventions this programme has carried -- ``int(round(x))``,
    ``floor(x + 0.5)`` and bare ``int(x)`` -- and any fourth.
    """

    import ast

    def mentions_depth_span(node: ast.AST) -> bool:
        for child in ast.walk(node):
            if (
                isinstance(child, ast.BinOp)
                and isinstance(child.op, ast.Sub)
                and isinstance(child.right, ast.Constant)
                and child.right.value == 1
            ):
                target = child.left
                name = (
                    target.attr
                    if isinstance(target, ast.Attribute)
                    else target.id
                    if isinstance(target, ast.Name)
                    else ""
                )
                if name in ("n_layer", "n_layers", "num_layers"):
                    return True
        return False

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = (
            function.attr
            if isinstance(function, ast.Attribute)
            else function.id
            if isinstance(function, ast.Name)
            else ""
        )
        if name not in ("int", "round", "floor"):
            continue
        if any(mentions_depth_span(argument) for argument in node.args):
            found.append(f"{path.name}:{node.lineno}")
    return found


def test_no_module_carries_a_second_depth_conversion():
    """The invariant behind the test above: exactly one implementation exists.

    The previous version asserted that one *known* duplicate had been deleted,
    which is a statement about the last repair rather than about the property.
    It passed while two further conventions were live:
    ``probes.erasure_layer_for`` kept ``int(round(...))`` -- so stage 09
    resolved its probe grid and its erasure layer under different rules on the
    two ProGen2 arms -- and ``tg05_relational_channel.py`` kept bare truncation,
    under which the published artefact records layers ``[8, 13, 17]`` for
    progen2-medium where the panel's depth 0.33 is layer 9.
    """

    roots = [
        REPO_ROOT / "src" / "transfer",
        REPO_ROOT / "scripts" / "transfer",
        REPO_ROOT / "scripts" / "transfer_gap",
    ]
    offenders: list[str] = []
    for root in roots:
        for path in sorted(root.glob("*.py")):
            if path.name == "scoring.py":  # the one declaration
                continue
            offenders.extend(_depth_conversion_sites(path))
    assert offenders == [], (
        "these sites convert a relative depth to a layer index without going "
        f"through src.transfer.scoring.analysis_layer: {offenders}"
    )


def test_the_erasure_layer_and_the_probe_grid_agree_on_every_arm():
    """Stage 09 resolves both from one relative depth; they must name one layer.

    ``erasure_layer_for`` carried round-half-to-even and ``analysis_layer_grid``
    round-half-up, so on ``progen2-base``/``progen2-medium`` at depth 0.25 the
    erasure layer was 6 and the probe grid contained 7. The stage then refused
    the run with a message blaming the operator's chosen depth.
    """

    for name, spec in sorted(PANEL.items()):
        arm = _stub_arm(name)
        for fraction in (0.0, 0.25, 1.0 / 3.0, 0.5, 2.0 / 3.0, 0.75, 1.0):
            assert probes.erasure_layer_for(arm, fraction) == analysis_layer(
                spec.n_layer, fraction
            ), (name, fraction)
        assert probes.erasure_layer_for(arm, 0.5) in analysis_layers(
            spec.n_layer, (0.25, 0.5, 0.75)
        )


def test_the_erasure_layer_refuses_a_depth_outside_the_unit_interval():
    for fraction in (-0.01, 1.01):
        with pytest.raises(ValueError):
            probes.erasure_layer_for(_stub_arm("gpt2-large"), fraction)


def test_the_conditioning_markers_have_one_declaration():
    """Four modules used to spell ``<start>``/``<end>`` independently."""

    cohort = Cohort(
        name="one",
        kind="protein",
        records=["AAA"],
        min_symbols=3,
        max_symbols=3,
        metadata={"ec_labels": ["1.1.1.1"]},
    )
    rendered = cohort.input_strings(_stub_arm("zymctrl"))[0]
    assert rendered == f"1.1.1.1<sep>{CONDITIONING_START}AAA{CONDITIONING_END}"


# ------------------------------------------------------------------ circuits


def test_summarise_patching_reports_a_layer_index_not_a_position_in_a_filtered_list():
    """``best_layer`` was ``argmax`` of the compacted list of non-null means."""

    arm = SimpleNamespace(n_layer=4)
    recovered = {
        "mlp_out|0|q": {"1-1": {"n": 0, "mean": None, "median": None}},
        "mlp_out|1|q": {"1-1": {"n": 2, "mean": 0.1, "median": 0.1}},
        "mlp_out|2|q": {"1-1": {"n": 0, "mean": None, "median": None}},
        "mlp_out|3|q": {"1-1": {"n": 2, "mean": 0.9, "median": 0.9}},
    }
    recovered.update({key.replace("|q", "|p"): value for key, value in recovered.items()})
    summary = summarise_patching(
        {
            "recovered_fraction": recovered,
            "component_kinds": ["mlp_out"],
            "bands": ["1-1"],
        },
        arm=arm,
    )
    band = summary["mlp_out|q"]["1-1"]
    assert band["best_layer"] == 3
    assert band["n_layers_with_a_mean"] == 2
    assert band["best_layer_fraction"] == pytest.approx(1.0)


def test_summarise_patching_reports_an_absent_band_as_absent():
    arm = SimpleNamespace(n_layer=2)
    recovered = {
        f"mlp_out|{layer}|{site}": {"1-1": {"n": 0, "mean": None, "median": None}}
        for layer in range(2)
        for site in ("p", "q")
    }
    summary = summarise_patching(
        {"recovered_fraction": recovered, "component_kinds": ["mlp_out"], "bands": ["1-1"]},
        arm=arm,
    )
    assert summary["mlp_out|q"]["1-1"]["best_layer"] is None
    assert summary["mlp_out|q"]["1-1"]["n_layers_with_a_mean"] == 0


def test_an_unscored_copying_layer_cannot_be_summarised_as_a_measured_zero():
    """Zero is a legal copying score, so a zero-filled unscored row reads as data."""

    matrix = np.zeros((3, 4), dtype=np.float64)
    matrix[1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        summarise_head_matrix(matrix, "diagonal_fraction")


def test_repeat_cohort_selection_is_seeded_and_reproducible():
    """Filtering for a criterion first hid the head-of-file draw one level deeper."""

    hit = RepeatHit(0, 20, 16, 0, None)
    found = [hit if index % 2 == 0 else None for index in range(40)]
    matching = [index for index, value in enumerate(found) if value is not None]
    file_order = _select_matching(found, n=5, skip=0, seed=None, name="c")
    assert file_order == matching[:5]
    seeded = _select_matching(found, n=5, skip=0, seed=17, name="c")
    assert seeded != file_order
    assert set(seeded) <= set(matching)
    assert seeded == _select_matching(found, n=5, skip=0, seed=17, name="c")
    later = _select_matching(found, n=5, skip=5, seed=17, name="c")
    assert not set(seeded) & set(later)


def test_repeat_cohort_selection_refuses_an_impossible_draw():
    found = [RepeatHit(0, 20, 16, 0, None)] * 3
    with pytest.raises(RuntimeError, match="only 3 matching records"):
        _select_matching(found, n=5, skip=0, seed=1, name="c")


# ------------------------------------------------------------------- channels


def test_an_event_that_fires_everywhere_has_no_attainable_gate():
    """``h(1) = 0``; dividing the gate by it was a division by zero, not a ratio."""

    with pytest.raises(ValueError, match="selection entropy is zero"):
        channels.event_selection_ceiling(
            100,
            event_counts=[10],
            gate_nats=[0.1],
            realised_event_count=100,
            realised_gate_nats=0.1,
        )


def test_the_event_selection_ceiling_still_reports_an_attainable_design():
    record = channels.event_selection_ceiling(
        1000,
        event_counts=[10, 100],
        gate_nats=[0.1],
        realised_event_count=100,
        realised_gate_nats=0.01,
    )
    assert record["gate_is_attainable"] is True
    assert record["realised_max_possible_mi_nats"] > 0.0


def test_alphafold_sampling_is_seeded_and_records_its_draw(tmp_path):
    """Filename order is accession order, which front-loads proteome dumps."""

    for index in range(40):
        (tmp_path / f"AF-P{index:05d}-F1-model_v4.pdb.gz").write_bytes(b"")
    prefix = channels.alphafold_models(tmp_path, limit=8)
    sample, record = channels.alphafold_model_sample(tmp_path, limit=8, seed=3)
    assert len(sample) == 8
    assert sample != prefix
    assert record["mode"] == "seeded_permutation"
    assert record["catalogue_size"] == 40
    again, _ = channels.alphafold_model_sample(tmp_path, limit=8, seed=3)
    assert again == sample


# ------------------------------------------------------------------ relational


def test_a_partner_closer_than_the_first_decoy_band_is_refused():
    """The band lookup raised ``StopIteration`` from inside a generator."""

    contact = np.zeros((60, 60), dtype=np.int8)
    contact[0, 5] = contact[5, 0] = 1
    with pytest.raises(ValueError, match="decoy band"):
        relational.anchored_pairs(
            contact,
            np.random.default_rng(0),
            min_separation=2,
            n_decoys=2,
            max_groups=1,
        )


def test_within_anchor_auc_reports_a_protein_clustered_interval():
    """Anchors inside one protein are not independent draws."""

    labels = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    scores = np.array([0.9, 0.1, 0.8, 0.2, 0.4, 0.6, 0.3, 0.7])
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    proteins = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    plain = relational.within_anchor_auc(labels, scores, groups)
    assert plain["sem_protein_clustered"] is None
    assert plain["sem_cluster_unit"] == "anchor"
    clustered = relational.within_anchor_auc(labels, scores, groups, proteins=proteins)
    assert clustered["auc"] == pytest.approx(plain["auc"])
    assert clustered["n_proteins"] == 2
    assert clustered["sem_protein_clustered"] is not None


def test_within_anchor_auc_refuses_an_anchor_spanning_two_proteins():
    labels = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.1, 0.8, 0.2])
    groups = np.array([0, 0, 1, 1])
    proteins = np.array(["A", "B", "C", "C"])
    with pytest.raises(ValueError, match="more than one protein"):
        relational.within_anchor_auc(labels, scores, groups, proteins=proteins)


# ------------------------------------------------------------------- homology


def _hit(**overrides) -> Hit:
    fields = {
        "query": "q00000",
        "subject": "UniRef50_Q3E8Z8",
        "pident": 100.0,
        "length": 607,
        "nident": 607,
        "qstart": 1,
        "qend": 607,
        "qlen": 732,
        "slen": 732,
        "evalue": 0.0,
        "bitscore": 1196.0,
    }
    fields.update(overrides)
    return Hit(**fields)


def test_the_observed_masked_alignment_is_detected_as_truncated():
    """Verbatim ``UniRef50_Q3E8Z8`` was binned at 82.9% by DIAMOND repeat masking.

    Confirmed against the corpus: cohort record 0 of the 2026-07-28 run is
    byte-identical to that entry over all 732 residues.
    """

    hit = _hit()
    assert hit.identity_over_query == pytest.approx(82.9, abs=0.1)
    assert truncated_alignment(hit) is True


def test_a_genuine_partial_homologue_is_not_flagged_as_truncated():
    """A fragment subject really is shorter than the query; a masked one is not."""

    assert truncated_alignment(_hit(slen=400, nident=400, qend=400)) is False
    assert truncated_alignment(_hit(pident=70.0, nident=430)) is False
    assert truncated_alignment(_hit(nident=730, qend=730)) is False


def test_assign_homology_stops_on_a_truncated_alignment():
    """A stratification built on truncated alignments is a different measurement."""

    cohort = Cohort("c", "protein", ["A" * 732], 1, 1000, {"repeats": [[0, 20, 16]]})
    with pytest.raises(RuntimeError, match="--masking 0"):
        assign_homology(cohort, ["q00000"], [_hit()])


def test_assign_homology_accepts_an_untruncated_search():
    cohort = Cohort("c", "protein", ["A" * 732], 1, 1000, {"repeats": [[0, 20, 16]]})
    assignments = assign_homology(
        cohort,
        ["q00000"],
        [_hit(nident=732, length=732, qend=732)],
        max_target_seqs=100,
    )
    assert assignments[0].max_identity_over_query == pytest.approx(100.0)
    assert assignments[0].stratum.startswith("ge95")
    assert assignments[0].hit_list_saturated is False


def test_a_saturated_hit_list_is_recorded():
    """``max identity in the corpus`` is really ``max of the hits it chose to report``."""

    cohort = Cohort("c", "protein", ["A" * 732] * 1, 1, 1000, {"repeats": [[0, 20, 16]]})
    hits = [
        _hit(nident=732, length=732, qend=732, subject=f"UniRef50_{index}", bitscore=100.0 - index)
        for index in range(3)
    ]
    assignments = assign_homology(cohort, ["q00000"], hits, max_target_seqs=3)
    assert assignments[0].hit_list_saturated is True


def test_stratum_integrity_counts_records_the_stratum_does_not_support():
    """A near-duplicate that misses the repeat cannot explain induction on it."""

    def assignment(index: int, spans: bool, identity: float) -> HomologyAssignment:
        from src.transfer.homology import assign_stratum

        return HomologyAssignment(
            record_index=index,
            query_id=f"q{index}",
            query_length=100,
            n_hits=4,
            max_identity_over_query=identity,
            max_pident=identity,
            best_subject="s",
            best_bitscore=1.0,
            best_qstart=1,
            best_qend=50,
            best_hit_spans_repeat=spans,
            stratum=assign_stratum(identity),
        )

    report = stratum_integrity(
        [assignment(0, True, 99.0), assignment(1, False, 99.0), assignment(2, True, 50.0)]
    )
    high = report["ge95_near_duplicate"]
    assert high["records"] == 2
    assert high["best_hit_does_not_span_repeat"] == 1


def test_a_degenerate_stratum_bootstrap_returns_no_interval():
    """A four-unit percentile interval is *narrower* than a four-hundred-unit one."""

    scores = [
        SimpleNamespace(
            sums={"prefix_matching": np.full((2, 2), 0.1 + 0.01 * index)},
            scored_positions=10,
            uniform_sum=1.0,
        )
        for index in range(MINIMUM_BOOTSTRAP_UNITS - 1)
    ]
    report = bootstrap_stratum(
        scores, threshold=0.05, n_heads=4, resamples=64, seed=1
    )
    assert report["degenerate"] is True
    assert report["peak_over_uniform_ci"] is None
    assert report["n_units"] == MINIMUM_BOOTSTRAP_UNITS - 1


def test_a_populated_stratum_bootstrap_still_returns_an_interval():
    scores = [
        SimpleNamespace(
            sums={"prefix_matching": np.full((2, 2), 0.1 + 0.01 * index)},
            scored_positions=10,
            uniform_sum=1.0,
        )
        for index in range(MINIMUM_BOOTSTRAP_UNITS + 4)
    ]
    report = bootstrap_stratum(scores, threshold=0.05, n_heads=4, resamples=200, seed=1)
    assert report["degenerate"] is False
    low, high = report["peak_over_uniform_ci"]
    assert low <= high


def test_sub_cohort_refuses_per_record_metadata_it_would_mis_pair():
    """A one-record sub-cohort used to keep a full-length ``repeat_stats``."""

    cohort = Cohort(
        "c",
        "protein",
        ["AAAA", "CCCC", "DDDD"],
        1,
        10,
        {
            "repeats": [[0, 2, 2]] * 3,
            "repeat_stats": [{"length": 2}] * 3,
            "unknown_per_record": [1, 2, 3],
        },
    )
    with pytest.raises(ValueError, match="unknown_per_record"):
        sub_cohort(cohort, [0], name="one")


def test_sub_cohort_slices_every_declared_per_record_array():
    cohort = Cohort(
        "c",
        "protein",
        ["AAAA", "CCCC", "DDDD"],
        1,
        10,
        {"repeats": [[0, 2, 2], [1, 3, 2], [2, 4, 2]], "repeat_stats": [1, 2, 3]},
    )
    sliced = sub_cohort(cohort, [2], name="one")
    assert sliced.metadata["repeats"] == [[2, 4, 2]]
    assert sliced.metadata["repeat_stats"] == [3]


# --------------------------------------------------------------------- probes


def test_record_order_is_seeded_by_default_and_reproducible():
    """Swiss-Prot is entry-name ordered and ProteinGym is one file per assay."""

    seeded = probes.record_order(50, seed=5)
    assert seeded != list(range(50))
    assert sorted(seeded) == list(range(50))
    assert seeded == probes.record_order(50, seed=5)
    assert probes.record_order(50, seed=5, mode="file_order") == list(range(50))


def test_record_order_refuses_an_undeclared_mode():
    with pytest.raises(ValueError, match="unknown record selection"):
        probes.record_order(10, seed=1, mode="whatever")


def test_the_behaviour_cohort_draw_spans_its_groups():
    """A prefix of a test fold is one assay; a cap must not collapse onto it."""

    candidates = list(range(30))
    groups = [f"assay_{index // 10}" for index in candidates]
    drawn = probes.stratified_unit_draw(candidates, groups=groups, limit=6, seed=2)
    assert len(drawn) == 6
    assert len({groups[index] for index in drawn}) == 3


def test_the_behaviour_cohort_draw_is_reproducible_and_bounded():
    candidates = list(range(12))
    groups = ["a"] * 6 + ["b"] * 6
    first = probes.stratified_unit_draw(candidates, groups=groups, limit=20, seed=9)
    assert first == sorted(candidates)
    assert first == probes.stratified_unit_draw(
        candidates, groups=groups, limit=20, seed=9
    )


def test_a_clustered_excess_interval_is_wider_than_a_row_level_one():
    """Forty mutants of one wild type are not forty independent sequences."""

    values = [1.0] * 10 + [0.0] * 10
    weights = [1] * 20
    groups = ["protein_a"] * 10 + ["protein_b"] * 10
    rows = probes.sequence_bootstrap(values, weights, seed=3, n_bootstrap=400)
    clustered = probes.sequence_bootstrap(
        values, weights, seed=3, n_bootstrap=400, groups=groups
    )
    assert (rows[1] - rows[0]) < (clustered[1] - clustered[0])


def test_a_single_cluster_cannot_produce_an_interval():
    with pytest.raises(ValueError, match="at least two groups"):
        probes.sequence_bootstrap(
            [1.0, 2.0], [1, 1], seed=1, n_bootstrap=200, groups=["a", "a"]
        )


def test_a_unit_must_declare_arm_independent_content():
    with pytest.raises(ValueError, match="raw content"):
        probes.Unit(
            unit_id="u",
            input_string="<|endoftext|>\nACDEF",
            token_ids=(1, 2, 3),
            group="g",
            positions=np.asarray([1]),
            labels={"ss3": np.asarray(["H"])},
            pool_span=(0, 3),
        )


def test_the_cohort_digest_is_arm_independent():
    """Hashing the rendered strings made the cross-arm identity check unable to fire."""

    def sample_set(rendered: str) -> probes.SampleSet:
        unit = probes.Unit(
            unit_id="u",
            input_string=rendered,
            content="ACDEFGHIKL",
            token_ids=(1, 2, 3),
            group="g",
            positions=np.asarray([1]),
            labels={"ss3": np.asarray(["H"])},
            pool_span=(0, 3),
        )
        return probes.SampleSet(
            concept="ss3",
            task_type="classification",
            y=np.asarray(["H"]),
            groups=np.asarray(["g"]),
            unit_index=np.asarray([0]),
            states={0: np.zeros((1, 4), dtype=np.float32)},
            pooled={0: np.zeros((1, 4), dtype=np.float32)},
            units=[unit],
            label_values=["H"],
        )

    wrapped = sample_set("<|endoftext|>\nACDEFGHIKL")
    conditioned = sample_set("1.1.1.1<sep><start>ACDEFGHIKL<end>")
    assert wrapped.cohort().records == ["ACDEFGHIKL"]
    assert wrapped.cohort().digest == conditioned.cohort().digest
    assert wrapped.cohort().min_symbols == 10


def test_an_arm_without_the_pathway_capability_is_refused_a_probe():
    """``Arm.blocks()`` carries no capability gate, so nothing else stopped it."""

    arm = _stub_arm("bygpt5-small-en")
    reason = probes.refusal_reason("next_token_class", arm)
    assert reason is not None
    assert "pathway" in reason


def test_a_capable_text_arm_is_not_refused():
    assert probes.refusal_reason("next_token_class", _stub_arm("gpt2-large")) is None


# --------------------------------------------------------------- path patching


def _probe(kind: str, offset: int) -> RepeatProbe:
    body = list(range(2, 10))
    ids = [1, *body, *body]
    return RepeatProbe(
        kind=kind,
        input_ids=tuple(token + offset for token in ids),
        query_positions=(10, 11, 12),
        key_positions=(2, 3, 4),
        coverage=1.0,
        repeat_symbols=8,
    )


def _unigram() -> Unigram:
    tokens = np.arange(1, 40, dtype=np.int64)
    return Unigram(
        token_ids=tokens,
        counts=np.full(tokens.size, 10, dtype=np.int64),
        total_tokens=int(tokens.size * 10),
        scored_sequences=10,
    )


def test_path_cases_are_drawn_across_probes_under_a_seeded_permutation():
    """Probes were consumed in list order, so only the first few ever contributed."""

    arm = SimpleNamespace(name="gpt2-large")
    all_probes = [_probe("natural_repeat_exact", 0) for _ in range(20)]
    cases, provenance = path_patching.build_path_cases(
        arm, all_probes, _unigram(), n_cases=6, cases_per_probe=1, max_tokens=64, seed=5
    )
    visited = sorted({case.probe_index for case in cases})
    assert len(cases) == 6
    assert provenance["probe_visit_order"] == "seeded_permutation"
    assert visited != list(range(6))
    repeat, _ = path_patching.build_path_cases(
        arm, all_probes, _unigram(), n_cases=6, cases_per_probe=1, max_tokens=64, seed=5
    )
    assert sorted({case.probe_index for case in repeat}) == visited


def test_path_cases_refuse_a_probe_list_that_mixes_criteria():
    """The provenance labels the whole set by ``cases[0]``; the design crosses criteria."""

    arm = SimpleNamespace(name="gpt2-large")
    mixed = [_probe("natural_repeat_exact", 0), _probe("natural_repeat_approximate", 0)]
    with pytest.raises(ValueError, match="mix kinds"):
        path_patching.build_path_cases(
            arm, mixed, _unigram(), n_cases=2, cases_per_probe=1, max_tokens=64, seed=1
        )


def test_path_patching_refuses_an_architecture_it_cannot_address():
    """Admission is by declaration, and the declaration is a closed set.

    EXP-R2-079 added the two rotary lineages, so the arms this once refused are
    now admitted. The invariant under test is not which arms are in the set -- that
    is the panel's business and it moves -- but that an architecture *outside* the
    set is refused by name rather than reaching a GPU and failing at depth.
    """

    for name in ("llama-3.2-3b", "qwen2.5-0.5b", "gpt2-large", "progen2-medium"):
        arm = _stub_arm(name)
        assert PANEL[name].architecture in path_patching.SUPPORTED_ARCHITECTURES
        path_patching.require_supported_layout(arm)

    undeclared = _stub_arm("llama-3.2-3b", architecture="mamba")
    with pytest.raises(TypeError, match="path patching is implemented"):
        path_patching.require_supported_layout(undeclared)


def _arm_with_projection(name: str, attribute: str) -> Arm:
    """An ``Arm`` whose blocks carry their attention output projection at ``attribute``.

    Includes the complete trunk, final norm and unembedding contract checked by
    ``path_patching.require_supported_layout``, **in the layout this arm's
    architecture actually uses**: GPT-2 and ProGen keep the trunk at
    ``model.transformer`` with blocks at ``.h``, a final ``ln_f`` and attention at
    ``.attn``; the rotary lineages keep it at ``model.model`` with ``.layers``, a
    final ``norm`` and attention at ``.self_attn``. Building one shape for both
    would let a resolver that only ever saw GPT-2 pass this test.
    """

    architecture = PANEL[name].architecture
    rotary = architecture in ("llama", "qwen2")
    attention_attribute = "self_attn" if rotary else "attn"

    def make_block() -> torch.nn.Module:
        block = torch.nn.Module()
        attention = torch.nn.Module()
        attention.add_module(attribute, torch.nn.Linear(4, 4))
        block.add_module(attention_attribute, attention)
        return block

    inner = torch.nn.Module()
    inner.add_module(
        "layers" if rotary else "h",
        torch.nn.ModuleList([make_block() for _ in range(PANEL[name].n_layer)]),
    )
    inner.add_module("norm" if rotary else "ln_f", torch.nn.LayerNorm(4))
    model = torch.nn.Module()
    model.add_module("model" if rotary else "transformer", inner)
    model.add_module("lm_head", torch.nn.Linear(4, 7))
    return Arm(
        spec=PANEL[name],
        model=model,
        tokenizer=None,
        device="cpu",
        dtype="float32",
        attn_implementation="eager",
    )


def test_require_supported_layout_admits_exactly_what_it_can_resolve():
    """The guard's admission must mean the projection resolves, not that a field matches.

    The four admitted architectures fall into two trunk layouts and use three
    output-projection names between them. Admission requires both the declared
    projection and the complete trunk contract used by the patcher, on the layout
    that architecture really has.
    """

    for name in ("gpt2-large", "progen2-medium", "llama-3.2-3b", "qwen2.5-0.5b"):
        resolvable = _arm_with_projection(
            name, path_patching._OUTPUT_PROJECTION_ATTRIBUTE[PANEL[name].architecture]
        )
        path_patching.require_supported_layout(resolvable)
        assert path_patching.attention_output_projection(resolvable, 0) is not None

        unresolvable = _arm_with_projection(name, "some_other_projection")
        with pytest.raises(TypeError, match="has no"):
            path_patching.attention_output_projection(unresolvable, 0)
        with pytest.raises(TypeError, match="has no"):
            path_patching.require_supported_layout(unresolvable)


def test_require_supported_layout_refuses_a_rotary_arm_wearing_the_gpt2_trunk():
    """A declaration that does not match the checkpoint is refused, not resolved.

    The guard used to hard-code ``model.transformer``; it now asks
    ``circuits.inner_decoder``, which branches on the declared architecture. A
    llama arm whose checkpoint is laid out like GPT-2 is a declaration error, and
    it has to fail here rather than at an arbitrary depth of a GPU run.
    """

    mislabelled = _arm_with_projection("gpt2-large", "c_proj")
    mislabelled.spec = replace(PANEL["gpt2-large"], architecture="llama")
    with pytest.raises(TypeError, match="no model.model"):
        path_patching.require_supported_layout(mislabelled)


def _head_row(direct: float, total: float) -> dict[str, float]:
    return {
        "effects": {
            "direct": direct,
            "mlp_mediated": 0.0,
            "attn_mediated": 0.0,
            "interaction": 0.0,
            "mediated": total - direct,
            "total": total,
        }
    }


def test_the_aggregate_fraction_is_withheld_when_the_heads_cancel():
    """A per-head floor guarding a sum over fifty heads let a residual clear it."""

    heads = [_head_row(1.0, 1.0), _head_row(-1.0, -0.995)]
    summary = path_patching.summarise_senders(heads, min_head_effect=0.01)
    assert summary["aggregate_fraction_valid"] is False
    assert summary["aggregate_effect_weighted_fraction"]["direct"] is None
    assert summary["aggregate_cancellation_ratio"] < 0.01
    assert summary["aggregate_is_a_weighted_mean"] is False
    assert summary["aggregate_n_heads"] == 2


def test_the_aggregate_fraction_survives_an_uncancelled_panel():
    heads = [_head_row(0.6, 1.0), _head_row(0.4, 1.0), _head_row(0.5, 1.0)]
    summary = path_patching.summarise_senders(heads, min_head_effect=0.01)
    assert summary["aggregate_fraction_valid"] is True
    assert summary["aggregate_is_a_weighted_mean"] is True
    assert summary["aggregate_effect_weighted_fraction"]["direct"] == pytest.approx(0.5)


def test_head_heterogeneity_is_not_reported_as_a_confidence_interval():
    """An arm's induction heads are its whole head population, not a sample.

    Updated by the 2026-07-29 repair: the head counts were four a side, which is
    now below the shared unit floor, so the case that exercises the *naming*
    invariant has to clear the floor first.  The four-head case became
    ``test_a_head_population_below_the_unit_floor_publishes_no_spread``.
    """

    report = path_patching.bootstrap_difference(
        [0.4, 0.5, 0.6, 0.55, 0.45, 0.52, 0.58, 0.47],
        [0.1, 0.2, 0.15, 0.05, 0.12, 0.18, 0.08, 0.11],
        resamples=500,
        seed=2,
        left_criterion="prefix_matching_above_threshold",
        right_criterion="prefix_matching_above_threshold",
    )
    assert report["is_a_sampling_confidence_interval"] is False
    assert report["resampling_unit"] == "sender_head"
    assert "ci_low" not in report and "ci_high" not in report
    assert "excludes_zero" not in report
    assert report["degenerate"] is False
    assert report["spread_low"] <= report["spread_high"]
    assert 0.0 <= report["p_two_sided_uncorrected"] <= 1.0
    assert "multiplicity" in report


# --------------------------------------------------------------------- scaling


def _frame_row(name: str, modality: str, capabilities: list[str], value: float) -> dict:
    row = {
        "name": name,
        "modality": modality,
        "capabilities": capabilities,
        "in_distribution": True,
        "realized_information_fraction": value,
    }
    row.update({metric: value for metric in ("mlp_share_of_context_information",)})
    row["induction_natural_fraction_of_heads_above_threshold"] = value
    return row


def test_nearest_neighbour_contrasts_apply_the_same_capability_gate_as_the_fit():
    """One quantity under two eligibility rules in one artefact."""

    frame = [
        _frame_row("gpt2-large", "text", ["pathway", "circuits", "lens", "budget"], 0.5),
        _frame_row("bygpt5-small-en", "text", ["budget", "lens"], 0.4),
        _frame_row("protgpt2", "protein", ["pathway", "circuits", "lens", "budget"], 0.3),
    ]
    contrasts = nearest_neighbour_contrasts(
        frame,
        metric_key="mlp_share_of_context_information",
        axis_key="realized_information_fraction",
    )
    assert len(contrasts) == 1
    assert contrasts[0]["text_member"] == "gpt2-large"
    assert contrasts[0]["required_capability"] == "pathway"


def test_an_arm_without_the_capability_cannot_become_a_nearest_neighbour():
    frame = [
        _frame_row("bygpt5-small-en", "text", ["budget", "lens"], 0.31),
        _frame_row("protgpt2", "protein", ["pathway", "circuits", "lens", "budget"], 0.3),
    ]
    contrasts = nearest_neighbour_contrasts(
        frame,
        metric_key="mlp_share_of_context_information",
        axis_key="realized_information_fraction",
    )
    assert contrasts[0]["text_member"] is None


# ------------------------------------------------------- criteria still nested


def test_the_repeat_criteria_are_still_nested_after_the_cohort_changes():
    """The seeded draw must not have moved what counts as a repeat."""

    assert PROTEIN_APPROXIMATE_CRITERION.min_unit == 16
    assert PROTEIN_APPROXIMATE_CRITERION.max_gap_ratio == 2.0
    assert math.isclose(PROTEIN_APPROXIMATE_CRITERION.max_substitution_rate, 0.5)


# ==========================================================================
# The 2026-07-29 repair pass.  Each block names the defect it defends against
# and asserts the property the repair restores, not the repair's shape.
# ==========================================================================


@contextlib.contextmanager
def monkeypatched(target, name: str, value):
    """Temporarily replace an attribute, restoring it however the block exits."""

    original = getattr(target, name)
    setattr(target, name, value)
    try:
        yield
    finally:
        setattr(target, name, original)


def _tiny_gpt2(seed: int = 3) -> Arm:
    """A real two-layer GPT-2 on the CPU, so the hook plumbing can be exercised.

    Several of the defects in this block live in ``register_forward_hook``
    plumbing -- whether a hook binds, which head a mask reaches, whether a
    pinned value is a module's own output.  None of that can be asserted against
    a stub that never runs a forward pass, and a source-text assertion tests the
    repair rather than the property.  The panel's own checkpoints need a GPU and
    are not available to a unit test, so the arm is a genuine
    ``GPT2LMHeadModel`` at eight dimensions and random weights: the same class,
    the same eager attention kernel, the same module names the measurement
    resolves, small enough to run in milliseconds.
    """

    config = GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=8,
        vocab_size=16,
        n_positions=32,
        attn_implementation="eager",
    )
    torch.manual_seed(seed)
    model = GPT2LMHeadModel(config).eval()

    class _PadTokenizer:
        pad_token_id = 0

    return Arm(
        spec=replace(PANEL["gpt2-large"], name="tiny-gpt2", n_layer=2, d_model=8),
        model=model,
        tokenizer=_PadTokenizer(),
        device="cpu",
        dtype="float32",
        attn_implementation="eager",
    )


def _tiny_patcher():
    """A ``PathPatcher`` over four hand-built cases, and a sender head."""

    def case(tokens: list[int], k: int, corrupt_token: int, probe: int):
        corrupted = list(tokens)
        corrupted[k] = corrupt_token
        return path_patching.PathCase(
            tuple(tokens),
            tuple(corrupted),
            len(tokens) - 1,
            k,
            tokens[k],
            corrupt_token,
            probe,
            "exact",
        )

    cases = [
        case([3, 5, 7, 9, 4, 5, 8, 6, 2, 5], 2, 11, 0),
        case([4, 6, 7, 2, 9, 6, 3, 5, 1, 6], 3, 13, 0),
        case([2, 8, 5, 7, 3, 6, 9, 4, 1, 8], 4, 15, 1),
        case([5, 9, 4, 6, 8, 2, 7, 3, 5, 9], 5, 12, 1),
    ]
    patcher = path_patching.PathPatcher(
        _tiny_gpt2(), cases, batch_size=4, minimum_effect=1e-4
    )
    sender = path_patching.SenderHead(
        layer=0, head=1, prefix_matching=0.5, above_threshold=True
    )
    return patcher, sender


def _tiny_paa_arm():
    """A tiny arm plus two PAA instances whose predicted token recurs earlier."""

    rows = [
        [3, 5, 7, 5, 9, 4, 5, 8, 6, 2, 5, 7],
        [4, 6, 7, 6, 2, 9, 6, 3, 5, 1, 6, 8],
    ]
    pool = prediction_addressed.InstancePool(
        arm="tiny-gpt2",
        sequence=np.array([0, 1]),
        query=np.array([10, 10]),
        antecedent=np.array([6, 6]),
        predicted_token=np.array([5, 6]),
        confidence=np.array([0.5, 0.5]),
        distance=np.array([4, 4]),
        unigram_percentile=np.array([0.5, 0.5]),
        decoys=np.array([[2, 3], [2, 3]]),
        clean_logit_target=np.array([1.0, 1.0]),
        clean_logit_runner_up=np.array([0.5, 0.5]),
        cascade={},
    )
    return _tiny_gpt2(), rows, pool


# ------------------------------------------------- homology: FASTA line state


def test_a_fasta_record_count_does_not_depend_on_the_read_block_size(tmp_path):
    """C1: line state must survive a block boundary that lands on a newline.

    ``count_fasta_records`` decides whether an existing DIAMOND index may be
    adopted for a corpus, so a count that depends on the buffer size is a
    resume verdict that depends on the buffer size.  The inner scan ran
    ``position <= len(block)``, which took one extra turn on an empty trailing
    segment whenever a block ended exactly at a newline, found no newline in it
    and cleared ``at_line_start`` -- so the *next* block's header went uncounted
    and its bytes were added to the residue total.

    Every chunk size is exercised, including the ones that split the file
    exactly at a newline, because the defect was invisible at all the others.
    """

    body = b">r0\nAAAA\n>r1\nCCCC\n>r2\nGGGG\n"
    path = tmp_path / "three.fa"
    path.write_bytes(body)
    truth = (3, 12)
    observed = {count_fasta_records(path, chunk=size) for size in range(1, len(body) + 3)}
    assert observed == {truth}

    # The two boundaries that used to be wrong, named so a regression reports
    # which case broke: chunk 9 ends on the newline after the first record and
    # chunk 6 ends on the newline inside it.
    assert count_fasta_records(path, chunk=9) == truth
    assert count_fasta_records(path, chunk=6) == truth

    # No trailing newline, and CRLF line endings, are the other two shapes a
    # real corpus arrives in.
    unterminated = tmp_path / "unterminated.fa"
    unterminated.write_bytes(b">r0\nAAAA\n>r1\nCCCC")
    assert {count_fasta_records(unterminated, chunk=size) for size in range(1, 20)} == {
        (2, 8)
    }
    crlf = tmp_path / "crlf.fa"
    crlf.write_bytes(b">a\r\nAA\r\nBB\r\n>b\r\nCC\r\n")
    assert {count_fasta_records(crlf, chunk=size) for size in range(1, 24)} == {(2, 6)}


# --------------------------------------- pathways: the held-out baseline's own bias


def test_the_smoothing_bias_of_the_held_out_baseline_tracks_vocabulary_size():
    """C2: the remedy for the plug-in bias carries a smaller version of it.

    Appendix B rule 3 replaced a plug-in unigram baseline with a held-out one
    because the plug-in's bias grows with vocabulary size.  The held-out
    estimator's additive smoothing biases it in the opposite direction on the
    *same* axis: ``s*V`` pseudo-counts against ``N`` reference tokens.  Upwards
    bias is conservative for one arm's share and is not conservative for an
    ordering across arms, which is the quantity the panel exists to produce.

    Asserted as an ordering rather than against fixed nats, so the test does not
    encode the sample it was written on.
    """

    rng = np.random.default_rng(0)
    inflation = {}
    for vocabulary in (32, 50257):
        probabilities = rng.dirichlet(np.full(vocabulary, 0.3))
        reference = rng.multinomial(100_000, probabilities).astype(float)
        targets = rng.multinomial(50_000, probabilities).astype(float)
        truth = float(-(targets * np.log(probabilities)).sum() / targets.sum())
        estimate = disjoint_unigram_cross_entropy_nats(reference, targets)
        inflation[vocabulary] = estimate - truth
        report = smoothing_diagnostics(reference, targets)
        assert report["vocabulary_size"] == vocabulary
        # The reportable quantity is an upper bound on the per-token inflation
        # of every token the reference saw, so it must bracket what is observed.
        assert report["normaliser_inflation_nats"] >= inflation[vocabulary] > 0.0
        assert 0.0 < report["smoothing_mass_fraction"] < 1.0

    # The bias is not a constant offset across the panel: it is two hundred
    # times larger on the 50k-piece arm than on the residue-level one.
    assert inflation[50257] > 100 * inflation[32]


def test_the_smoothing_constant_is_swept_and_the_sweep_travels_with_the_number():
    """C2: an unavoidable constant is swept, per Appendix B rule 8.

    The constant cannot be eliminated -- an unsmoothed held-out unigram is
    infinite the moment a scored target is unseen in the reference -- so the
    remedy is to show what turns on it.  The sweep must therefore be present,
    must recompute the same estimator, and must show that a *smaller* constant
    is not automatically better, which is the reason the default did not move.
    """

    rng = np.random.default_rng(1)
    probabilities = rng.dirichlet(np.full(50257, 0.3))
    reference = rng.multinomial(100_000, probabilities).astype(float)
    targets = rng.multinomial(50_000, probabilities).astype(float)
    report = smoothing_diagnostics(reference, targets)

    assert set(report["cross_entropy_by_smoothing"]) == {"1", "0.5", "0.1", "0.01"}
    assert report["cross_entropy_by_smoothing"]["1"] == pytest.approx(
        disjoint_unigram_cross_entropy_nats(reference, targets, smoothing=1.0)
    )
    # Some scored targets are unseen in the reference, which is what makes a
    # small constant expensive rather than free.
    assert report["target_mass_unseen_in_reference"] > 0.0
    truth = float(-(targets * np.log(probabilities)).sum() / targets.sum())
    at = {
        key: abs(value - truth)
        for key, value in report["cross_entropy_by_smoothing"].items()
    }
    assert at["0.01"] > at["0.5"], "a smaller constant is not uniformly less biased"
    assert report["cross_entropy_sweep_range_nats"] > 0.0

    # And on a residue-level vocabulary the same sweep is inert, which is the
    # differential the panel has to know about.
    small = rng.dirichlet(np.full(32, 0.3))
    small_reference = rng.multinomial(100_000, small).astype(float)
    small_targets = rng.multinomial(50_000, small).astype(float)
    small_report = smoothing_diagnostics(small_reference, small_targets)
    assert small_report["cross_entropy_sweep_range_nats"] < 1e-3
    assert (
        small_report["smoothing_mass_fraction"] < report["smoothing_mass_fraction"] / 100
    )


# --------------------------------------- pathways: denominator guards on the share


def _pathway_rows(clean: float, variant: float, *, n: int = 8, tokens: int = 100) -> list[dict]:
    return [
        {
            "token_count": tokens,
            "clean_nll_sum": clean * tokens,
            "variant_nll_sum": variant * tokens,
            "kl_sum": 0.5 * tokens,
            "argmax_agreement_count": tokens // 2,
        }
        for _ in range(n)
    ]


def test_a_share_is_refused_on_a_denominator_too_small_to_divide_by():
    """C5: the share was guarded on the denominator's sign, not its magnitude.

    ``budget.MIN_CONTEXT_INFORMATION_NATS`` is the floor below which no ratio
    computed on an arm can be interpreted, and ``pathway_metrics`` -- which
    produces exactly such a ratio -- tested only ``context_information > 0``.
    A hundredth of a nat is positive.
    """

    # Denominator 0.01 nats: positive, and far below the floor.
    metrics = pathway_metrics(_pathway_rows(2.0, 2.4), unigram_entropy_nats=2.01)
    assert metrics["context_information_nats"] == pytest.approx(0.01)
    assert metrics["context_information_positive"] is True
    assert metrics["context_information_valid"] is False
    assert metrics["share_of_context_information"] is None
    assert metrics["measurable"] is False
    assert metrics["minimum_context_information_nats"] == pytest.approx(0.30)

    # A denominator above the floor still produces the share it always did.
    healthy = pathway_metrics(_pathway_rows(2.0, 2.4), unigram_entropy_nats=3.0)
    assert healthy["context_information_valid"] is True
    assert healthy["share_of_context_information"] == pytest.approx(0.4 / 1.0)

    # The floor is a parameter, so the ordering can be shown not to turn on it.
    swept = pathway_metrics(
        _pathway_rows(2.0, 2.4),
        unigram_entropy_nats=2.01,
        minimum_context_information_nats=0.001,
    )
    assert swept["context_information_valid"] is True


def test_a_share_interval_conditioned_on_its_denominator_is_not_published():
    """C4: dropping invalid draws changes the estimand and used to be silent.

    A draw is discarded exactly when its clean cross-entropy came out high, so
    the survivors carry the largest denominators and therefore the smallest
    shares.  ``statistics.MINIMUM_FINITE_DRAW_FRACTION`` exists for this and was
    not reached from here: the percentile over the survivors was published under
    the plain name with only a count beside it.
    """

    # Half the sequences have a clean CE above the baseline and half well below,
    # so resampling straddles the floor and a large minority of draws is invalid.
    rows = _pathway_rows(0.2, 0.6, n=6) + _pathway_rows(3.0, 3.4, n=6)
    report = pathway_cluster_bootstrap(
        rows, samples=400, seed=3, unigram_entropy_nats=2.0
    )
    assert 0 < report["share_valid_samples"] < 400
    assert report["share_of_context_information"] is None
    assert report["share_interval_refused_reason"] is not None
    assert "conditioned" in report["share_interval_refused_reason"]
    # The effect-scale intervals do not depend on the denominator and survive.
    assert report["ce_delta_nats"] is not None
    assert report["kl_clean_to_ablated_nats"] is not None

    # When every draw clears the floor the interval is published exactly as before.
    healthy = pathway_cluster_bootstrap(
        _pathway_rows(2.0, 2.4), samples=200, seed=3, unigram_entropy_nats=4.0
    )
    assert healthy["share_valid_samples"] == 200
    assert healthy["share_interval_refused_reason"] is None
    assert healthy["share_of_context_information"]["median"] == pytest.approx(0.4 / 2.0)


def test_the_clean_reference_is_shared_rather_than_compared():
    """C6a: a guard that cannot fail must not be advertised as an invariant.

    ``measure_pathways`` compared each scope's clean NLL against the first
    scope's and raised on a difference.  ``clean_nll_sum`` is a function of the
    clean logits, the token ids and the target mask alone, all of which are
    fixed outside the scope loop, so the comparison was ``x != x``.  The
    property is structural; what has to stay true is that the clean forward pass
    is not moved inside the loop, and the docstring must say which of the two it
    is.
    """

    source = inspect.getsource(pathways.measure_pathways)
    assert "clean reference changed between scopes" not in source
    assert "structural, not checked" in pathways.measure_pathways.__doc__
    # One clean forward per batch: it is established before the scope loop and
    # never re-established inside it.
    body = source.split("for batch in batches:", 1)[1]
    before, inside = body.split("for scope in scopes:", 1)
    assert "clean_logits = arm.model(" in before
    assert "clean_logits = arm.model(" not in inside


# ---------------------------------------------------- one shared bootstrap floor


def _stratum_scores(count: int) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            sums={"prefix_matching": np.full((2, 2), 0.1 + 0.01 * index)},
            scored_positions=10,
            uniform_sum=1.0,
        )
        for index in range(count)
    ]


def _mean_difference(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(prediction - truth))


def _case_flags(clusters: int) -> tuple[np.ndarray, np.ndarray]:
    sources = np.repeat(np.arange(clusters), 3)
    flags = (np.arange(sources.size) % 2).astype(bool)
    return flags, sources


#: Every resampler in ``src.transfer`` that reaches the shared unit floor, with a
#: call one unit below the floor and one exactly on it.  ``raises`` is required
#: of the functions whose unit count is a configuration choice, or whose return
#: value has nowhere to carry a verdict a caller would have to notice;
#: ``degenerate`` of the ones whose unit count is a measured property of a
#: checkpoint or a stratum, where the count is itself the finding and the point
#: estimate survives.  The distinction is the one ``bootstrap_unit_floor``'s
#: docstring draws, and it is asserted here rather than described.
FLOOR_RESPECTING_RESAMPLERS: dict[str, dict[str, object]] = {
    "statistics.paired_group_bootstrap": {
        "refusal": "raises",
        "below": lambda n: statistics.paired_group_bootstrap(
            np.zeros(n),
            np.ones(n),
            np.full(n, 2.0),
            np.arange(n),
            _mean_difference,
            seed=0,
            n_bootstrap=200,
        ),
    },
    "induction_robustness.cluster_bootstrap_fraction": {
        "refusal": "raises",
        "below": lambda n: induction_robustness.cluster_bootstrap_fraction(
            np.tile(np.array([[0.9, 0.0], [0.0, 0.0]]), (n, 1, 1)),
            threshold=0.1,
            resamples=200,
            seed=0,
        ),
    },
    "induction_robustness.contrast_ratio_bootstrap": {
        "refusal": "raises",
        "below": lambda n: induction_robustness.contrast_ratio_bootstrap(
            np.tile(np.array([[0.9, 0.9], [0.0, 0.0]]), (n, 1, 1)),
            np.tile(np.array([[0.9, 0.0], [0.0, 0.0]]), (n, 1, 1)),
            threshold=0.1,
            resamples=200,
            seed=0,
        ),
    },
    "prediction_addressed.cluster_bootstrap": {
        "refusal": "raises",
        "below": lambda n: prediction_addressed.cluster_bootstrap(
            np.linspace(0.0, 1.0, n).reshape(n, 1),
            np.ones(n),
            replicates=400,
            seed=0,
        ),
    },
    "homology.bootstrap_stratum": {
        "refusal": "degenerate",
        "below": lambda n: homology.bootstrap_stratum(
            _stratum_scores(n), threshold=0.05, n_heads=4, resamples=64, seed=1
        ),
    },
    "path_patching.bootstrap_difference": {
        "refusal": "degenerate",
        # Both criteria are required and named per side: EXP-R2-073 made the
        # resampled population a declared fact rather than a docstring, because
        # under the exhaustive criterion the population is the whole head grid
        # and the caveat that shipped with it described a threshold-selected set.
        "below": lambda n: path_patching.bootstrap_difference(
            list(np.linspace(0.4, 0.6, n)),
            list(np.linspace(0.0, 0.2, n)),
            resamples=200,
            seed=2,
            left_criterion="prefix_matching_above_threshold",
            right_criterion="prefix_matching_above_threshold",
        ),
    },
    "circuits._case_resampled_interval": {
        "refusal": "degenerate",
        "below": lambda n: circuits._case_resampled_interval(
            *_case_flags(n), np.random.default_rng(0), 200
        ),
    },
}

#: Resamplers that do NOT reach the floor. Recorded rather than quietly left out
#: of the loop above, because the previous version of this test asserted the
#: property in its name and exercised no resampler at all, so the five below were
#: invisible. Each is the same hazard in a module the repair that wrote this list
#: did not own; ``probes.sequence_bootstrap`` resamples sequences,
#: ``lenses.*_cluster_bootstrap`` and ``pathways.pathway_cluster_bootstrap``
#: resample clusters, and all five guard nothing or guard ``n < 2``.
RESAMPLERS_WITHOUT_A_UNIT_FLOOR = frozenset(
    {
        "probes.sequence_bootstrap",
        "lenses.lens_cluster_bootstrap",
        "lenses.residue_class_cluster_bootstrap",
        "lenses.jacobian_cluster_bootstrap",
        "pathways.pathway_cluster_bootstrap",
    }
)

#: The floor itself, which is not a resampler.
FLOOR_DECLARATION = "statistics.bootstrap_unit_floor"


def _package_resamplers() -> set[str]:
    """Every resampling entry point in ``src.transfer``, found rather than listed."""

    import importlib
    import pkgutil

    from src import transfer as package

    found: set[str] = set()
    for module_info in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"src.transfer.{module_info.name}")
        for name, member in vars(module).items():
            if not inspect.isfunction(member) or member.__module__ != module.__name__:
                continue
            lowered = name.lower()
            if "bootstrap" in lowered or "resampl" in lowered:
                found.add(f"{module_info.name}.{name}")
    return found


def test_one_bootstrap_unit_floor_is_declared_and_every_resampler_reaches_it():
    """C7: the floor is declared once, and the declaration is not the property.

    The previous version of this test asserted ``MINIMUM_BOOTSTRAP_UNITS == 8``
    and called ``bootstrap_unit_floor`` twice.  Both passed while three of the
    package's resamplers guarded ``n < 2`` or nothing, one of which returned an
    interval *narrower* at two units than at three -- the pinching pathology the
    constant's own comment is written about.  A test named for every resampler
    has to invoke every resampler, so this one enumerates them: one unit below
    the floor must refuse, and the floor itself must not.
    """

    assert MINIMUM_BOOTSTRAP_UNITS is statistics.MINIMUM_BOOTSTRAP_UNITS
    assert statistics.MINIMUM_BOOTSTRAP_UNITS == 8

    floor = statistics.bootstrap_unit_floor(3)
    assert floor["degenerate"] is True
    assert floor["n_units"] == 3 and floor["minimum_units"] == 8
    assert "coverage" in floor["degenerate_reason"]
    assert statistics.bootstrap_unit_floor(8)["degenerate"] is False
    assert statistics.bootstrap_unit_floor(8)["degenerate_reason"] is None

    for name, spec in FLOOR_RESPECTING_RESAMPLERS.items():
        call = spec["below"]
        if spec["refusal"] == "raises":
            with pytest.raises(ValueError, match="below the 8-unit floor"):
                call(MINIMUM_BOOTSTRAP_UNITS - 1)
        else:
            refused = call(MINIMUM_BOOTSTRAP_UNITS - 1)
            assert refused["degenerate"] is True, name
            # ``circuits`` names the same floor after its own unit -- "8-cluster"
            # against "8-unit" -- so the count is what is asserted, not the noun.
            assert f"below the {MINIMUM_BOOTSTRAP_UNITS}-" in (
                refused.get("degenerate_reason") or refused.get("reason") or ""
            ), name
        accepted = call(MINIMUM_BOOTSTRAP_UNITS)
        assert accepted is not None, name
        if spec["refusal"] == "degenerate":
            assert accepted["degenerate"] is False, name


def test_the_resampler_inventory_is_complete_and_its_gaps_are_named():
    """C7: a resampler cannot arrive, or be repaired, without a decision.

    The five in ``RESAMPLERS_WITHOUT_A_UNIT_FLOOR`` are an accepted limitation,
    not a defect that was overlooked: they live in modules outside the change
    that fixed the others, and the honest record of that is a named list this
    test holds to be exhaustive.  Adding a resampler fails here until it is
    placed in one list or the other; giving one of the five a floor fails here
    until it is moved.
    """

    discovered = _package_resamplers()
    declared = (
        set(FLOOR_RESPECTING_RESAMPLERS)
        | set(RESAMPLERS_WITHOUT_A_UNIT_FLOOR)
        | {FLOOR_DECLARATION}
    )
    assert discovered == declared, (
        "undeclared resamplers: "
        f"{sorted(discovered - declared)}; declared but absent: "
        f"{sorted(declared - discovered)}"
    )
    for name in RESAMPLERS_WITHOUT_A_UNIT_FLOOR:
        module_name, function_name = name.split(".")
        module = sys.modules[f"src.transfer.{module_name}"]
        source = inspect.getsource(getattr(module, function_name))
        assert "bootstrap_unit_floor" not in source, (
            f"{name} now applies the floor; move it into "
            "FLOOR_RESPECTING_RESAMPLERS with a below-floor call"
        )


def test_a_cluster_bootstrap_below_the_unit_floor_is_refused():
    """C7: ``cluster_bootstrap`` guarded ``n < 2`` and published everything above.

    It returns bare arrays with nowhere to carry a verdict, so the refusal is an
    exception rather than a flag: every caller reads ``q_low``/``q_high``
    straight into a report.
    """

    values = np.linspace(0.0, 1.0, 7).reshape(7, 1)
    with pytest.raises(ValueError, match="below the 8-unit floor"):
        cluster_bootstrap(values, np.ones(7), replicates=400, seed=0)

    ok = cluster_bootstrap(
        np.linspace(0.0, 1.0, 8).reshape(8, 1), np.ones(8), replicates=400, seed=0
    )
    assert ok["q_low"][0] <= ok["mean"][0] <= ok["q_high"][0]

    # Zero-weight clusters do not count towards the floor: they contribute
    # nothing to the estimate and must not be able to lift a run over it.
    weights = np.array([1.0] * 7 + [0.0] * 4)
    with pytest.raises(ValueError, match="below the 8-unit floor"):
        cluster_bootstrap(np.linspace(0.0, 1.0, 11).reshape(11, 1), weights, replicates=400, seed=0)


def test_a_percentile_the_replicate_count_cannot_resolve_is_refused():
    """An interval is refused when its own tail holds too few draws to be one.

    The census asks for a Bonferroni column at ``alpha = 0.05 / n_heads``. At the
    24 heads it screens, 1000 replicates put 1.04 draws below the lower
    percentile, so the published bound is the second-smallest draw and moves by
    0.0028 logits between seeds -- the size of the smallest effect in the table
    beside it. At the 720 heads an exhaustive census patches it is 0.035 draws.

    The refusal states the replicate count that *would* resolve the request,
    because that is the parameter a caller can change; and it subsumes the flat
    ``replicates >= 100`` it replaced, which at the default alpha admitted 2.5
    draws in the tail.
    """

    values = np.linspace(0.0, 1.0, 32).reshape(32, 1)
    weights = np.ones(32)

    with pytest.raises(ValueError, match="Use at least 9600 replicates"):
        cluster_bootstrap(values, weights, replicates=1000, seed=0, alpha=0.05 / 24)
    with pytest.raises(ValueError, match="Use at least 288000 replicates"):
        cluster_bootstrap(values, weights, replicates=1000, seed=0, alpha=0.05 / 720)
    # The old flat minimum admitted this; the rule that replaced it does not.
    with pytest.raises(ValueError, match="Use at least 400 replicates"):
        cluster_bootstrap(values, weights, replicates=100, seed=0)

    resolved = cluster_bootstrap(values, weights, replicates=9600, seed=0, alpha=0.05 / 24)
    assert resolved["q_low"][0] <= resolved["mean"][0] <= resolved["q_high"][0]

    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError, match="alpha must lie"):
            cluster_bootstrap(values, weights, replicates=1000, seed=0, alpha=bad)

    # The data problem is reported before the replicate problem: a caller with
    # both has to fix the cohort either way, and the count is the cheaper fix.
    with pytest.raises(ValueError, match="below the 8-unit floor"):
        cluster_bootstrap(
            np.linspace(0.0, 1.0, 4).reshape(4, 1), np.ones(4), replicates=100, seed=0
        )


def test_a_head_population_below_the_unit_floor_publishes_no_spread():
    """C7: the 2026-07-28 panel published spreads over three to six heads.

    Degenerate is returned rather than raised, because an arm having four
    induction heads is a measured property of the checkpoint and not a
    configuration error.  The point difference survives; the spread, the
    separation verdict and the p-value do not.
    """

    report = path_patching.bootstrap_difference(
        [0.4, 0.5, 0.6, 0.55],
        [0.1, 0.2, 0.15, 0.05],
        resamples=500,
        seed=2,
        left_criterion="prefix_matching_above_threshold",
        right_criterion="prefix_matching_above_threshold",
    )
    assert report["degenerate"] is True
    assert report["n_units"] == 4
    assert report["difference"] == pytest.approx(0.5125 - 0.125)
    assert report["spread_low"] is None and report["spread_high"] is None
    assert report["separated_across_heads"] is None
    assert report["p_two_sided_uncorrected"] is None
    assert "below the 8-unit floor" in report["interval_caveat"]

    # The smaller side sets the floor: eight against four is still four.
    lopsided = path_patching.bootstrap_difference(
        [0.4] * 12,
        [0.1, 0.2, 0.15, 0.05],
        resamples=500,
        seed=2,
        left_criterion="prefix_matching_above_threshold",
        right_criterion="prefix_matching_above_threshold",
    )
    assert lopsided["degenerate"] is True and lopsided["n_units"] == 4


# ------------------------------------------ path patching: estimator agreement


def test_a_clustered_standard_error_is_published_with_its_own_centre():
    """C8: the SEM was the SEM of an estimator that was not the mean beside it.

    ``mean`` is case-weighted; ``sem_probe_clustered`` is the standard error of
    an unweighted mean over probe means.  They differ whenever probes contribute
    unequal case counts, which the eligibility filter guarantees, and here the
    published mean lands outside its own interval.
    """

    values = torch.tensor([0.9] * 20 + [0.1] * 2 + [0.05] * 2, dtype=torch.float64)
    probes = torch.tensor([0] * 20 + [1] * 2 + [2] * 2)
    report = path_patching._probe_clustered_sem(values, probes, "direct")

    assert report["n_probes"] == 3
    assert report["mean_probe_clustered"] == pytest.approx((0.9 + 0.1 + 0.05) / 3.0)
    # The case-weighted mean is not the centre this standard error describes,
    # and the gap is larger than the standard error itself.
    case_weighted = float(values.mean())
    assert case_weighted == pytest.approx(0.7625)
    centre = report["mean_probe_clustered"]
    assert abs(case_weighted - centre) > report["sem_probe_clustered"]
    # Equal case counts is the one situation in which the two agree.
    equal = path_patching._probe_clustered_sem(
        torch.tensor([0.9, 0.9, 0.1, 0.1], dtype=torch.float64),
        torch.tensor([0, 0, 1, 1]),
        "direct",
    )
    assert equal["mean_probe_clustered"] == pytest.approx(0.5)


def test_the_head_write_linearity_check_is_read_per_case():
    """C9: a ratio of batch norms hides per-row violations behind one large row.

    The identity being checked -- under the direct condition the final residual
    moves by exactly the patched head's own write -- holds case by case, so one
    row whose write is an order of magnitude larger than the others must not be
    able to supply the denominator for all of them.  Shown first in closed form
    at the invariant's own 0.02 tolerance, then measured on a real forward pass.
    """

    write = torch.zeros(8, 4, dtype=torch.float64)
    write[0, 0] = 20.0           # one dominant row, correct
    write[1:, 0] = 0.1           # seven small rows
    error = torch.zeros_like(write)
    error[1:, 0] = 0.1           # ...each entirely wrong

    aggregate = float(error.norm() / write.norm())
    per_row = float((error.norm(dim=-1) / write.norm(dim=-1)).max())
    assert aggregate < 0.02, "the old aggregate form passes seven wrong rows in eight"
    assert per_row == pytest.approx(1.0)
    assert per_row > 0.02, "the per-case form fails, which is correct"

    # On a real model the per-case maximum is attainable at the same tolerance:
    # it must be of the same order as the aggregate, not orders above it, or the
    # stricter reading would be a gate the positive control cannot pass.
    patcher, sender = _tiny_patcher()
    report = path_patching.structural_invariants(
        patcher, sender, tolerance=1e-3, linearity_tolerance=0.02
    )
    rows_block = report["head_write_linearity_rows"]
    assert report["head_write_linearity"]["observed"] < 0.02
    assert rows_block["n_rows_scored"] == rows_block["n_rows"]
    assert rows_block["batch_aggregate_relative_error"] > 0.0
    assert (
        report["head_write_linearity"]["observed"]
        < 100 * rows_block["batch_aggregate_relative_error"]
    )
    # The write floor is relative, so it cannot be a threshold that admits every
    # case on one arm's residual scale and no case on another's.
    assert 0.0 < path_patching._HEAD_WRITE_RELATIVE_FLOOR < 1.0


def test_the_freeze_only_invariant_has_a_positive_control():
    """C6b: ``freeze_only`` pins each sublayer to its own output on its own input.

    Writing a module's deterministic output back into it is a no-op, so the
    condition recovers zero whether the hooks are exact or never bound at all --
    and a freeze that silently fails to bind makes the four pathway conditions
    one condition, so the arm reports a 100% direct effect with every invariant
    green.  The control pins the same sublayers to a value that is *not* their
    own output and requires the metric to move.
    """

    patcher, sender = _tiny_patcher()
    report = path_patching.structural_invariants(
        patcher, sender, tolerance=1e-3, linearity_tolerance=0.02
    )
    assert report["freeze_only"]["observed"] == pytest.approx(0.0, abs=1e-3)
    assert abs(report["freeze_only_perturbed"]["observed"]) > 1e-3

    # The negative path: freezing hooks that bind and do nothing.  ``freeze_only``
    # still reports zero and still passes -- which is the whole defect -- and the
    # run fails on the positive control instead.
    patcher, sender = _tiny_patcher()
    inert = lambda self, cached, batch, *, all_positions: (lambda *a, **k: None)  # noqa: E731
    with monkeypatched(path_patching.PathPatcher, "_freeze_hook", inert):
        with pytest.raises(RuntimeError) as failure:
            path_patching.structural_invariants(
                patcher, sender, tolerance=1e-3, linearity_tolerance=0.02
            )
    message = str(failure.value)
    assert "freeze_only_perturbed" in message
    assert "freeze_only:" not in message, "the old check cannot see an inert freeze"


def test_the_census_score_is_emitted_on_the_key_set_the_knockout_removes():
    """The selector scored one key; the causal statistic removes all of them.

    ``paa_specific`` scores attention onto ``pool.antecedent``, the nearest
    earlier occurrence. ``knockout_effects`` removes every earlier occurrence.
    That is a mismatch of *one* key against a median of 3 on gpt2-large and 13 to
    17 on ProGen2-medium, so a rank correlation between the two -- which is what
    plan item D2.c is -- would be attenuated harder on the small alphabet by
    construction, in the direction the modality hypothesis predicts.
    ``corruption_effects`` already fixed this for the matching gate.

    Both scores are emitted rather than one replacing the other: ``paa_specific``
    is what EXP-R2-059 published and L5/L6 quote. The assertions below are
    computed from the model's own attention pattern, not from the function's
    return value, so they test the definition rather than the implementation.
    """

    arm, rows, pool = _tiny_paa_arm()
    scores = prediction_addressed.paa_attention_scores(arm, rows, pool, batch_size=2)

    ids = torch.tensor(np.asarray(rows, dtype=np.int64), dtype=torch.long)
    with torch.no_grad():
        pattern = arm.model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
            output_attentions=True,
        ).attentions

    key_sets = prediction_addressed.antecedent_sets(rows, pool, np.array([0, 1]))
    # The fixture's predicted token recurs three times before the query, so this
    # test would pass vacuously if the sets were singletons.
    assert [len(keys) for keys in key_sets] == [3, 3]
    assert scores["keys_per_instance"].tolist() == [3.0, 3.0]

    for instance in (0, 1):
        sequence = int(pool.sequence[instance])
        query = int(pool.query[instance])
        nearest = int(pool.antecedent[instance])
        decoys = [int(value) for value in pool.decoys[instance]]
        for layer in range(arm.n_layer):
            weights = pattern[layer][sequence]
            decoy_mean = np.mean(
                [weights[:, query, decoy].numpy() for decoy in decoys], axis=0
            )
            expected_nearest = weights[:, query, nearest].numpy() - decoy_mean
            expected_matched = (
                np.sum([weights[:, query, key].numpy() for key in key_sets[instance]], axis=0)
                - len(key_sets[instance]) * decoy_mean
            )
            np.testing.assert_allclose(
                scores["paa_specific"][sequence, layer], expected_nearest, atol=1e-6
            )
            np.testing.assert_allclose(
                scores["paa_specific_matched"][sequence, layer],
                expected_matched,
                atol=1e-6,
            )

    # The two scores must actually differ here, or the test cannot fail for the
    # reason it exists: the nearest key carries only part of the removed mass.
    assert not np.allclose(scores["paa_specific"], scores["paa_specific_matched"])


def test_the_matched_census_score_collapses_onto_the_nearest_key_score():
    """With one antecedent the two definitions coincide, and must.

    The size-matched decoy baseline is ``n_keys`` times the per-key mean, so at
    ``n_keys == 1`` it is the per-key mean and the matched score is the original.
    A baseline that stayed at the per-key mean while the numerator became a sum
    would pass the previous test and fail this one.
    """

    arm, rows, pool = _tiny_paa_arm()
    # Row 0's predicted token 5 sits at 1, 3 and 6; blank the first two so only
    # the nearest occurrence survives, and leave row 1 alone as the contrast.
    single = [list(rows[0]), list(rows[1])]
    single[0][1] = 14
    single[0][3] = 14

    key_sets = prediction_addressed.antecedent_sets(single, pool, np.array([0, 1]))
    assert [len(keys) for keys in key_sets] == [1, 3]

    scores = prediction_addressed.paa_attention_scores(arm, single, pool, batch_size=2)
    np.testing.assert_allclose(
        scores["paa_specific"][0], scores["paa_specific_matched"][0], atol=1e-6
    )
    assert not np.allclose(scores["paa_specific"][1], scores["paa_specific_matched"][1])
    assert scores["keys_per_instance"].tolist() == [1.0, 3.0]


def test_the_antecedent_knockout_confirms_the_intervention_took_effect():
    """C6c: the antecedent mass was captured on the clean pass only.

    The taps are removed before any knockout runs, so nothing in the artefact
    showed a head's antecedent mass collapsing under the intervention.  A mask
    that broadcast over heads, or a head index off by one, would return
    ``delta_m_gap`` near zero for every head and be reported as the absence of a
    causally confirmable suppressive head population -- which this module states
    up front is "a complete answer".
    """

    arm, rows, pool = _tiny_paa_arm()
    selected = np.array([0, 1])
    heads = [(0, 0), (1, 1)]

    effects = prediction_addressed.knockout_effects(
        arm, rows, pool, selected, heads, batch_size=2
    )
    # Before: the heads do read the antecedents.  After: they do not.
    assert effects["antecedent_attention_mass"].min() > 0.01
    assert effects["knocked_antecedent_attention_mass"].max() == pytest.approx(0.0)
    assert effects["knockout_residual_mass_max"] < prediction_addressed.KNOCKOUT_RESIDUAL_MASS

    # The negative path: a mask that lands one head over.  Every delta comes back
    # near zero, which reads as "no suppressive head", so the run must refuse.
    real = prediction_addressed.build_knockout_mask

    def one_head_over(**kwargs):
        kwargs = dict(kwargs)
        kwargs["head"] = (kwargs["head"] + 1) % kwargs["heads"]
        return real(**kwargs)

    with monkeypatched(prediction_addressed, "build_knockout_mask", one_head_over):
        with pytest.raises(RuntimeError, match="attention mass on the antecedent keys"):
            prediction_addressed.knockout_effects(
                arm, rows, pool, selected, heads, batch_size=2
            )


def test_the_knockout_mask_is_per_head_and_per_query():
    """C6c: the shape the positive control exists to defend."""

    mask = prediction_addressed.build_knockout_mask(
        batch=2,
        heads=4,
        width=6,
        head=1,
        query_positions=torch.tensor([5, 5]),
        key_sets=[[1, 3], [2]],
        device="cpu",
        dtype=torch.float32,
    )
    assert mask.shape == (2, 4, 6, 6)
    assert float(mask[0, 1, 5, 1]) == pytest.approx(
        prediction_addressed.KNOCKOUT_LOGIT, rel=1e-6
    )
    assert float(mask[0, 0, 5, 1]) == 0.0, "the mask must not broadcast over heads"
    assert float(mask[0, 1, 4, 1]) == 0.0, "nor over query positions"


# ------------------------------------------- the adjudicating statistic's interval


def _covariate_scores(n: int, seed: int):
    """Probe scores whose induction response tracks identity, not repeat length."""

    rng = np.random.default_rng(seed)
    lengths = rng.integers(15, 60, size=n)
    identity = rng.uniform(0.0, 100.0, size=n)
    response = 0.02 * identity + rng.normal(0.0, 0.5, size=n)

    class _Score:
        def __init__(self, index: int, repeat: int, value: float) -> None:
            self.record_index = index
            self.scored_positions = 10
            self.repeat_symbols = repeat
            self.sums = {"prefix_matching": np.full((2, 2), value)}

    scores = [_Score(i, int(lengths[i]), float(response[i] * 10)) for i in range(n)]
    return scores, {i: float(identity[i]) for i in range(n)}


def test_the_memorisation_adjudication_carries_an_interval():
    """C19: two partial correlations at n=8 used to decide it, as bare points.

    This file refuses a percentile interval below eight units on the grounds
    that one taken there misleads, and then adjudicated between memorisation and
    a repeat-length artefact on two correlations over as few as eight probes
    with no interval, no p-value and no resampling.  "The identity term survives
    and the length term does not" is a comparison, and it now has a sampling
    fraction of its own rather than two point estimates to eyeball.
    """

    scores, identities = _covariate_scores(24, seed=5)
    report = homology_covariate_analysis(
        scores, identities, layer=0, head=0, resamples=500, seed=1
    )
    assert report["measured"] is True
    booted = report["bootstrap"]
    assert booted["unit"] == "probe"
    assert booted["usable_draws"] == 500 and booted["degenerate_draws"] == 0
    low, high = booted["partial_identity_given_repeat_length_ci"]
    assert low <= report["partial_identity_given_repeat_length"] <= high
    low, high = booted["partial_repeat_length_given_identity_ci"]
    assert low <= report["partial_repeat_length_given_identity"] <= high
    # The verdict's own quantity, not an eyeball of two intervals.
    assert booted["fraction_identity_term_larger"] > 0.5
    assert booted["refused_reason"] is None


def test_an_ill_conditioned_covariate_bootstrap_publishes_no_interval():
    """C19: resamples with a constant covariate are counted, not dropped silently."""

    class _Score:
        def __init__(self, index: int, value: float) -> None:
            self.record_index = index
            self.scored_positions = 10
            self.repeat_symbols = 20 + index          # perfectly ordered
            self.sums = {"prefix_matching": np.full((2, 2), value)}

    scores = [_Score(i, 0.1 + 0.02 * i) for i in range(10)]
    identities = {i: float(i * 7 % 50) for i in range(10)}
    booted = homology_covariate_analysis(
        scores, identities, layer=0, head=0, resamples=300, seed=1
    )["bootstrap"]
    assert booted["degenerate_draws"] > 0
    assert booted["usable_draws"] < booted["required_draws"]
    assert booted["partial_identity_given_repeat_length_ci"] is None
    assert booted["fraction_identity_term_larger"] is None
    assert "conditioned" in booted["refused_reason"]


# ------------------------------------------------ PAA: scaffolding, sink, matching


def test_scaffolding_and_the_attention_sink_supply_no_antecedent_or_decoy():
    """C11 and C10: queries excluded the prefix; antecedents and decoys did not.

    ``tokenised_rows`` computes the content bound and uses it to skip the
    scaffolding prefix when choosing queries.  ``build_instance_pool`` never
    received it and searched ``tokens[:q]`` from index 0, so ``k*`` could land on
    a FASTA newline or an EC tag -- tokens that recur in every row by
    construction -- and the decoy window started at ``max(0, q - high)``, so the
    attention sink was an eligible decoy for any instance in a far distance bin.
    """

    pool = prediction_addressed.InstancePool(
        arm="x",
        sequence=np.array([0]),
        query=np.array([9]),
        antecedent=np.array([5]),
        predicted_token=np.array([7]),
        confidence=np.array([0.5]),
        distance=np.array([4]),
        unigram_percentile=np.array([0.5]),
        decoys=np.array([[2, 3]]),
        clean_logit_target=np.array([1.0]),
        clean_logit_runner_up=np.array([0.5]),
        cascade={},
        content_low=3,
    )
    # Token 7 sits at 0, 1 and 2 -- all inside the scaffolding -- and at 5 and 8.
    rows = [[7, 7, 7, 4, 6, 7, 2, 9, 7, 1, 3, 8]]
    keys = prediction_addressed.antecedent_sets(rows, pool, np.array([0]))
    assert keys == [[5, 8]], "scaffolding occurrences must not become knockout keys"

    # With no declared scaffolding, position 0 is still excluded: it is the sink.
    plain = replace(pool, content_low=0)
    assert prediction_addressed.antecedent_sets(rows, plain, np.array([0])) == [
        [1, 2, 5, 8]
    ]


def _load_stage_module(filename: str):
    """Import a numbered entry point by path, the way the worker's preflight does."""

    import importlib.util

    path = REPO_ROOT / "scripts/transfer" / filename
    spec = importlib.util.spec_from_file_location(f"_stage_{filename[:2]}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_pool_round_trip_keeps_every_field_the_dataclass_declares(tmp_path):
    """``content_low`` was dropped by save/load and by the matched subset.

    ``antecedent_sets`` decides which keys the causal knockout removes and reads
    ``content_low`` from the pool, so a reloaded pool that silently took the
    dataclass default of 0 would add format scaffolding -- ProtGPT2's newline,
    ZymCTRL's ``<sep>`` -- to the removed key set and attribute the effect to the
    antecedent. It stayed latent because the only pool ever reloaded is the text
    control's, whose content bound is 0 either way; it becomes live the first time
    a protein pool is scored.

    The assertion is over ``dataclasses.fields`` rather than a written-out list,
    so a field added later cannot be dropped without failing here.
    """

    census = _load_stage_module("14_paa_census.py")
    pool = prediction_addressed.InstancePool(
        arm="protgpt2",
        sequence=np.array([0]),
        query=np.array([9]),
        antecedent=np.array([5]),
        predicted_token=np.array([7]),
        confidence=np.array([0.5]),
        distance=np.array([4]),
        unigram_percentile=np.array([0.5]),
        decoys=np.array([[2, 3]]),
        clean_logit_target=np.array([1.0]),
        clean_logit_runner_up=np.array([0.5]),
        cascade={"positions_scored": 11},
        content_low=3,
    )
    rows = [[7, 7, 7, 4, 6, 7, 2, 9, 7, 1, 3, 8]]
    path = tmp_path / "pool.npz"
    census.save_pool(path, rows, pool)
    reloaded_rows, reloaded = census.load_pool(path)

    assert reloaded_rows == rows
    for field in fields(prediction_addressed.InstancePool):
        original = getattr(pool, field.name)
        restored = getattr(reloaded, field.name)
        if isinstance(original, np.ndarray):
            assert np.array_equal(original, restored), field.name
        else:
            assert original == restored, field.name

    # The knockout key set is the thing that moved, so assert it directly rather
    # than only the field it is derived from.
    selected = np.array([0])
    assert prediction_addressed.antecedent_sets(
        reloaded_rows, reloaded, selected
    ) == prediction_addressed.antecedent_sets(rows, pool, selected)
    assert census._subset(reloaded, selected).content_low == pool.content_low


def test_loading_a_pool_written_before_a_field_existed_refuses(tmp_path):
    """A default is not a measurement, and downstream cannot tell them apart."""

    census = _load_stage_module("14_paa_census.py")
    path = tmp_path / "old_pool.npz"
    np.savez_compressed(
        path,
        rows=np.asarray([[1, 2, 3]], dtype=np.int64),
        arm="protgpt2",
        cascade=json.dumps({}),
        sequence=np.array([0]),
        query=np.array([2]),
        antecedent=np.array([1]),
        predicted_token=np.array([3]),
        confidence=np.array([0.5]),
        distance=np.array([1]),
        unigram_percentile=np.array([0.5]),
        decoys=np.array([[1]]),
        clean_logit_target=np.array([1.0]),
        clean_logit_runner_up=np.array([0.5]),
    )
    with pytest.raises(RuntimeError, match="content_low"):
        census.load_pool(path)


def test_the_matching_gate_corrupts_what_the_causal_statistic_removes():
    """C12: the gate replaced one occurrence; the statistic removes them all.

    Asymmetric by modality, which is what makes it a manufactured conclusion
    rather than noise: over a twenty-symbol alphabet the predicted token recurs
    dozens of times in a protein row, so replacing the nearest occurrence alone
    leaves the evidence intact and the instance lands in the "no effect" bin;
    in English the same token is often a once-occurring content word and the
    single replacement removes everything.  The matching then reports that the
    arms cannot be matched.
    """

    source = inspect.getsource(prediction_addressed.corruption_effects)
    assert "antecedent_sets(rows, pool, block)" in source
    assert "for position in key_sets[row]" in source
    # One draw per occurrence, so the corrupted row is not a run of one symbol,
    # which would itself be a readable pattern.
    corrupt_block = source.split("corrupted = base.copy()", 1)[1]
    assert corrupt_block.count("rng.choice") == 1
    assert corrupt_block.index("for position in key_sets[row]") < corrupt_block.index(
        "rng.choice"
    )


def test_the_alpha_sweep_control_is_measured_on_the_treatment_axis():
    """C14: the random control had no manipulation check of its own.

    Appendix B rule 5.  Recording p(X) only under the prediction nudge leaves the
    artefact unable to say whether a random direction of identical norm moved
    the target probability as much -- and at alpha = 2 the perturbation is twice
    the residual norm.
    """

    source = inspect.getsource(prediction_addressed.query_source_intervention)
    assert "p_target_by_alpha_random_control" in source
    assert "p_substitute_by_alpha_random_control" in source
    # Recorded for both conditions, not inside an ``if label == "prediction"``.
    assert 'if label == "prediction":\n                            probabilities' not in source


def test_the_per_sequence_cap_is_a_seeded_draw_not_a_prefix():
    """C15: ``max_per_sequence`` kept the first eligible queries in each row.

    Earlier queries carry less context, so a front cap shifts both the
    query-position and the antecedent-distance distributions, and distance is
    one of the covariates the CEM gate matches on.  Latent -- the campaign does
    not pass the cap -- but it is a "first N" selection inside a seeded function.
    """

    source = inspect.getsource(prediction_addressed.build_instance_pool)
    assert "kept_here" not in source
    assert "rng.choice(\n                        len(row_instances)" in source
    assert "instances_dropped_by_per_sequence_cap" in source


# ------------------------------------------------ smaller corrections, C20 group


def test_an_externally_supplied_baseline_does_not_claim_an_estimator():
    """C20: ``override_nats`` returned ``estimator: 'disjoint'`` for a CLI value."""

    arm = _stub_arm()
    record = pathways.unigram_baseline(
        arm,
        estimator="disjoint",
        target_counts=np.array([5, 3, 2], dtype=np.int64),
        override_nats=1.25,
    )
    assert record["source"] == "external_override"
    assert record["estimator"] == "external_override"
    assert record["requested_estimator"] == "disjoint"
    assert record["nats"] == pytest.approx(1.25)


def test_a_zero_width_interval_says_so():
    """C20: a zero standard error produced an interval that reads as precision."""

    degenerate = statistics.mean_interval([2.0, 2.0, 2.0, 2.0])
    assert degenerate["interval"] == [2.0, 2.0]
    assert degenerate["zero_width_from_zero_variance"] is True
    assert statistics.mean_interval([1.0, 2.0, 3.0])["zero_width_from_zero_variance"] is False


def test_a_single_class_test_fold_is_refused_like_a_single_class_training_fold():
    """C20: only the training fold's class coverage was checked."""

    # The positive class lives in two groups out of six, so with three folds one
    # fold's test set gets neither of them while every training fold still sees
    # one -- which is exactly the case the training-fold check cannot reach.
    truth = np.array([1, 1, 1, 1] + [0] * 8)
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    with pytest.raises(ValueError, match="test fold contains fewer than two classes"):
        statistics.make_group_splits(
            truth, groups, n_splits=3, seed=0, task_type="classification"
        )

    # A grouping that can be scored is still accepted.
    balanced_truth = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    balanced_groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    splits = statistics.make_group_splits(
        balanced_truth, balanced_groups, n_splits=2, seed=0, task_type="classification"
    )
    assert len(splits) == 2


def test_a_flat_or_falling_truncation_curve_is_distinguishable_from_a_missing_one():
    """C20: a negative span is a finding and collapsed to the same ``None``."""

    source = inspect.getsource(budget.truncation_curve)
    assert "negative_nll_rises_with_context" in source
    assert "flat_no_context_effect" in source
    assert "nll_reduction_status" in source


def test_an_empty_head_set_keeps_the_populated_schema():
    """C20: ``ov_over_heads`` changed shape when no head cleared the threshold.

    ZymCTRL selects no head at threshold 0.10, so the arm whose row changes
    shape is the one a modality reading turns on.
    """

    copying = {
        "diagonal_fraction": np.full((2, 2), 0.5),
        "mean_normalised_rank": np.full((2, 2), 0.25),
    }
    empty = ov_over_heads(copying, [])
    populated = ov_over_heads(copying, [(0, 1), (1, 0)])
    assert set(empty) == set(populated)
    assert empty["n_heads"] == 0
    assert all(empty[key] is None for key in empty if key != "n_heads")


def test_the_artefact_writer_commits_the_directory_entry(tmp_path):
    """C20: ``write_json``'s 'Atomic' headline fsynced the file, not the rename."""

    source = inspect.getsource(io_module._atomic_write)
    assert "os.fsync(directory)" in source
    target = tmp_path / "nested" / "artefact.json"
    write_json(target, {"b": 2, "a": 1})
    assert target.read_text() == '{\n  "a": 1,\n  "b": 2\n}\n'
    assert not list(tmp_path.glob("nested/.artefact*"))


def test_a_plug_in_measurability_verdict_names_its_own_estimator():
    """C3: the verdict came from the estimator the package calls diagnostic-only.

    ``pathways.UNIGRAM_ESTIMATORS`` describes the plug-in as "an explicit opt-in
    diagnostic" whose bias runs from +0.003 nats at 32 symbols to +1.65 at
    50257, which is several times the 0.30-nat floor the verdict is taken
    against -- so which arms pass is set mostly by vocabulary size.  No
    published verdict came from it (``01_cohort_power.py`` recomputes from the
    held-out estimator), and nothing stopped a caller inheriting one, because
    the estimator was recorded in a different field from the verdict.  The
    plug-in verdict now carries its estimator in its own value.
    """

    assert budget.MEASURABLE_PLUG_IN != budget.MEASURABLE
    assert budget.UNMEASURABLE_PLUG_IN != budget.UNMEASURABLE
    assert "plug_in" in budget.MEASURABLE_PLUG_IN
    signature = inspect.signature(budget.arm_power).parameters
    # The estimator can be made explicit at the call, and the default is not a
    # silent one.
    assert "held_out_unigram_nats" in signature
    assert signature["held_out_unigram_nats"].default is None
    source = inspect.getsource(budget.arm_power)
    assert "power_verdict_plug_in" in source
    assert "MEASURABLE_PLUG_IN if status == MEASURABLE else UNMEASURABLE_PLUG_IN" in source


def test_bits_per_symbol_divides_by_the_expansion_of_the_window_it_scored():
    """C16: the nats were per scored target, the expansion per rendered string.

    ZymCTRL's EC tag, ``<sep>``, ``<start>`` and ``<end>`` are excluded from the
    scored targets and counted by ``arms.symbols_per_token``.  They add tokens
    and no alphabet symbols, so they understate the expansion for that arm alone
    -- and the expansion is a divisor, so every ``*_bits_per_symbol`` figure came
    out inflated for exactly one arm, on the axis the module declares to be the
    cross-arm comparable one.
    """

    source = inspect.getsource(budget.arm_power)
    assert "expansion = scored_symbols_per_token(arm, scored)" in source
    assert '"symbols_per_token_rendered_string": rendered_expansion' in source
    # The scored-window expansion is derived from the scored targets themselves,
    # so it cannot drift from the multiset the cross-entropy was taken over.
    assert "scored.target_ids" in inspect.getsource(budget.scored_symbols_per_token)


def test_a_near_clonal_holdout_is_measured_rather_than_asserted():
    """C17: 'held out' was enforced by byte-identity over a near-clonal corpus.

    Two sequences differing at one residue share essentially every order-1 and
    order-2 statistic, and this ladder is described as the only
    tokenizer-independent axis the protein arms can be compared on.  Exact set
    disjointness cannot see that; a shared 30-mer over a twenty-letter alphabet
    can, and is homology rather than chance.
    """

    rng = np.random.default_rng(0)
    alphabet = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
    train = ["".join(rng.choice(alphabet, size=200)) for _ in range(40)]
    test = ["".join(rng.choice(alphabet, size=200)) for _ in range(20)]
    clean = budget.near_duplicate_fraction(train, test)
    assert clean["fraction_test_sequences_with_shared_kmer"] == 0.0

    near_clone = list(train[0])
    near_clone[100] = "W" if near_clone[100] != "W" else "A"
    contaminated = test[:19] + ["".join(near_clone)]
    # The check the module had says these sets are disjoint.
    assert not set(train) & set(contaminated)
    # The check it now has does not.
    leaked = budget.near_duplicate_fraction(train, contaminated)
    assert leaked["n_test_sequences_with_shared_kmer"] == 1
    assert leaked["fraction_test_sequences_with_shared_kmer"] == pytest.approx(0.05)
    assert budget.markov_baselines(train, contaminated)["held_out_strictness"][
        "n_test_sequences_with_shared_kmer"
    ] == 1
