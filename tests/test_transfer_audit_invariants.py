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

import math
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer import channels, path_patching, probes, relational  # noqa: E402
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
    stratum_integrity,
    sub_cohort,
    truncated_alignment,
)
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
    """``circuits`` and ``pathway`` are both granted to arms this module cannot reach."""

    rotary = _stub_arm("llama-3.2-3b")
    assert rotary.supports("circuits") and rotary.supports("pathway")
    with pytest.raises(TypeError, match="path patching is implemented"):
        path_patching.require_supported_layout(rotary)
    path_patching.require_supported_layout(_stub_arm("gpt2-large"))


def _arm_with_projection(name: str, attribute: str) -> Arm:
    """An ``Arm`` whose blocks carry their attention output projection at ``attribute``.

    Enough module structure for ``Arm.blocks()``/``Arm.attention()`` and
    ``path_patching.attention_output_projection`` to resolve, and nothing else.
    """

    block = torch.nn.Module()
    attention = torch.nn.Module()
    attention.add_module(attribute, torch.nn.Linear(4, 4))
    block.add_module("attn", attention)
    transformer = torch.nn.Module()
    transformer.add_module("h", torch.nn.ModuleList([block]))
    model = torch.nn.Module()
    model.add_module("transformer", transformer)
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

    This is the invariant behind the guard, and it did not hold. Until
    EXP-R2-067 ``require_supported_layout`` compared ``ArmSpec.architecture``
    against ``SUPPORTED_ARCHITECTURES`` and stopped, so it admitted the two
    ProGen2 arms -- which declare ``architecture='gpt2'`` by never setting the
    field, while their checkpoints keep ``out_proj`` where GPT-2 keeps
    ``c_proj``. ``attention_output_projection`` then raised partway through a
    scheduled GPU run, which is the failure this guard exists to replace.

    The previous test asserted the admission, so it defended the defect.
    """

    for name in ("gpt2-large", "progen2-medium"):
        resolvable = _arm_with_projection(
            name, path_patching._OUTPUT_PROJECTION_ATTRIBUTE[PANEL[name].architecture]
        )
        path_patching.require_supported_layout(resolvable)
        assert path_patching.attention_output_projection(resolvable, 0) is not None

        # The ProGen2 checkpoints' real layout. The guard must refuse it rather
        # than leave the refusal to a call made after the weights are resident.
        unresolvable = _arm_with_projection(name, "some_other_projection")
        with pytest.raises(TypeError, match="has no"):
            path_patching.attention_output_projection(unresolvable, 0)
        with pytest.raises(TypeError, match="has no"):
            path_patching.require_supported_layout(unresolvable)


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
    """An arm's induction heads are its whole head population, not a sample."""

    report = path_patching.bootstrap_difference(
        [0.4, 0.5, 0.6, 0.55], [0.1, 0.2, 0.15, 0.05], resamples=500, seed=2
    )
    assert report["is_a_sampling_confidence_interval"] is False
    assert report["resampling_unit"] == "sender_head"
    assert "ci_low" not in report and "ci_high" not in report
    assert "excludes_zero" not in report
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
