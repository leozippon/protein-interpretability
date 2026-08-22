#!/usr/bin/env python3
"""The control that separates modality from convergence, scale and tokenisation.

The programme's comparison - GPT-2-large against ProtGPT2, ZymCTRL and
ProGen2-medium - is matched on depth, width and vocabulary size and on nothing
else. The protein decoders were trained on different data, at a different scale,
to a different degree of convergence, with different tokenizers, and are scored
on a cohort none of them was validated against. So "protein decoders are harder
to interpret" and "these particular protein decoders are less converged, or are
off-distribution on this cohort" predict the same measurements, and the
programme has so far run only measurements that cannot tell them apart.

This script runs the measurement that can. Every rung of a size ladder - text and
protein - is scored on its own native cohort: the corpus and sequence-length band
it was pretrained for, declared beside the model rather than shared across the
panel, because a shared corpus penalises whichever model's training distribution
happens to sit furthest from it. Every protein rung is additionally scored on
every other protein cohort in the run, so that the in-distribution exclusion is
supported by evidence about the alternative bands rather than asserted from one.
From that scoring comes
a convergence axis (how much of the cohort's context-free entropy the model
actually removes, plus a per-symbol cross-entropy that survives the comparison of
BPE with residue-level tokenisation, plus parameter count). The same
interpretability metrics used elsewhere in the programme - mean-ablation pathway
shares, the induction-head census, direct-logit-attribution concentration - are
then read against that axis, separately by modality.

Three readings come out, and the script is built so that each of them can go
against the programme:

*the fit* of metric on convergence plus a modality indicator, whose indicator
coefficient is the residual modality offset at matched convergence;
*the tokenisation control*, ProtGPT2's multi-residue BPE against the
residue-level protein rungs, which isolates tokenisation inside one modality;
*the distribution control*, which reports every rung's in-distribution flag and
excludes from every fit any model that fails it.

The verdict rule is fixed in ``src.transfer.scaling`` before any data is seen and
defaults to ``underpowered``. Nothing here is a production sweep: this is a
validation-scale run of a control, and its point is that it could refute the
hypothesis it was built to test.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from dataclasses import replace as dataclass_replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer.scoring import aggregate_variant  # noqa: E402
from src.transfer.io import sha256_file, write_json  # noqa: E402
from src.transfer.arms import (  # noqa: E402
    DEFAULT_CORPUS_DRAW_SEED,
    Arm,
    Cohort,
    load_arm,
    protein_cohort,
    symbols_per_token,
    text_cohort,
    tokenize_batch,
)
from src.transfer.budget import arm_power  # noqa: E402
from src.transfer.circuits import (  # noqa: E402
    INDUCTION_THRESHOLDS,
    PROTEIN_EXACT_CRITERION,
    TEXT_EXACT_CRITERION,
    attention_alignment_scores,
    direct_logit_attribution,
    fit_unigram,
    head_census,
    n_head,
    natural_repeat_probes,
    protein_repeat_cohort,
    synthetic_repeat_probes,
    text_repeat_cohort,
)
from src.transfer.pathways import (  # noqa: E402
    TEXT_COHORT_SOURCE,
    UNIGRAM_ESTIMATORS,
    assert_disjoint,
    attn_all,
    build_baseline,
    cohort_composition,
    cohort_target_token_counts,
    held_out_cohort,
    measure_pathways,
    mlp_all,
    pathway_cluster_bootstrap,
    pathway_metrics,
    prepare_batches,
    scope_record,
    subsample_cohort,
    unigram_baseline,
)
from src.transfer.lenses import (  # noqa: E402
    activation_subspace,
    analysis_layer,
    cache_residuals,
    freeze_parameters,
    jacobian_alignment,
    jacobian_finite_difference_check,
    jacobian_formulation,
    jacobian_gram,
    jacobian_matrices,
    jacobian_probe_row,
    lens_head,
    prepare_windows,
    sample_jacobian_probes,
)
from src.transfer.scaling import (  # noqa: E402
    CONVERGENCE_AXES,
    LADDER_TABLE_COLUMNS,
    DEFAULT_EQUIVALENCE_MARGIN,
    DEFAULT_INDUCTION_THRESHOLD,
    DEFAULT_LADDER,
    DEFAULT_MIN_RESIDUAL_DOF,
    INTERPRETABILITY_METRICS,
    NO_DENOMINATOR_STANDARD_ERROR,
    PRIMARY_AXIS,
    PRIMARY_INDUCTION_PROBE,
    PRIMARY_METRIC,
    SCHEMA_VERSION,
    LadderMember,
    analysis_frame,
    aperture_summary,
    attribution_summary,
    circuits_supported,
    cohort_sensitivity_rows,
    conditioning_control,
    convergence_row,
    decide_verdict,
    distribution_control,
    fit_modality_offset,
    induction_summary,
    inspect_member,
    lens_supported,
    nearest_neighbour_contrasts,
    paired_architecture_contrast,
    parse_ladder_table,
    pathway_summary,
    register_arm_spec,
    renderable,
    tokenisation_contrast,
)

DEFAULT_OUTPUT = REPO_ROOT / "results/transfer/convergence_control"
DEFAULT_BACKUP = REPO_ROOT / "logs/convergence_control_backup"
DEFAULT_LADDER_TABLE: Path | None = None

#: Whole-pathway ablation scopes. Anchored single-layer scopes are swept
#: elsewhere; here the y-value has to be one number per model that does not
#: depend on where in a ladder-varying depth the anchor was placed, so only the
#: whole-pathway scopes are measured.
PATHWAY_SCOPES = ("mlp_all", "attn_all")


def resolve_ladder(
    table_path: Path | None,
) -> tuple[tuple[LadderMember, ...], dict[str, Any]]:
    """The configured ladder, from the operator's table when there is one.

    The ladder is assembled by a separate process, so a table on disk is
    authoritative and replaces the built-in list entirely rather than merging
    with it; merging would let a stale built-in entry survive an operator's
    deliberate removal.
    """

    if table_path is None:
        return DEFAULT_LADDER, {
            "source": "scaling.DEFAULT_LADDER",
            "path": None,
            "sha256": None,
            "note": "no operator ladder table supplied",
        }
    if not table_path.is_file():
        raise FileNotFoundError(f"ladder table does not exist: {table_path}")
    digest = sha256_file(table_path)
    parsed = parse_ladder_table(table_path)
    if parsed is None:
        raise ValueError(
            f"{table_path}: no ladder declaration with required columns "
            f"{list(LADDER_TABLE_COLUMNS)}"
        )
    return parsed, {
        "source": "ladder_table",
        "path": str(table_path),
        "sha256": digest,
    }


def build_pool(
    key: tuple[str, int, int], pool_size: int, *, skip: int = 0, seed: int | None
) -> Cohort:
    """The evaluation pool named by one ``(corpus, min symbols, max symbols)`` key.

    Pools are keyed rather than derived from modality because an arm is scored on
    the distribution it was trained for, and two arms of the same modality need
    not share one: ProtGPT2 and the ProGen2 rungs are full-length-protein models
    and ZymCTRL's window is capped by the token budget its ``<end>`` marker has to
    fit inside. Members whose keys agree share a pool and therefore a digest,
    which is what makes their convergence axes comparable.
    """

    corpus, low, high = key
    suffix = "" if skip == 0 else f"_skip{skip}"
    if corpus == TEXT_COHORT_SOURCE:
        return text_cohort(
            pool_size, min_chars=low, skip=skip, name=f"openwebtext_screen{suffix}",
            seed=seed,
        )
    if corpus == "ec_labelled_swissprot":
        return protein_cohort(
            pool_size, low, high, skip=skip, with_ec=True,
            name=f"swissprot_ec_{low}_{high}{suffix}",
            seed=seed,
        )
    if corpus == "plain_swissprot":
        return protein_cohort(
            pool_size, low, high, skip=skip, with_ec=False,
            name=f"swissprot_{low}_{high}{suffix}",
            seed=seed,
        )
    raise ValueError(f"unsupported cohort corpus {corpus!r}")


def measure_pathway_shares(
    arm: Arm,
    cohort: Cohort,
    reference_counts,
    reference: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Mean-ablation footprint of the whole MLP and whole attention pathways.

    The context-free denominator is estimated on a held-out corpus rather than on
    the tokens it normalises. A plug-in entropy computed on its own sample is
    biased downwards by roughly the distinct-token count over twice the sample
    size, which on a 32-sequence cohort is negligible for a 32-token protein
    alphabet and about 1.65 nats for a 50257-piece BPE one. Because the share is
    ``dCE / (H - CE)``, that bias inflates the share of exactly the
    large-vocabulary arms and leaves the residue-level arms alone - a differential
    distortion running along the tokenisation contrast this control measures. The
    plug-in figure is still recorded, so the correction is visible per rung
    instead of being asserted.

    The same denominator normalises the convergence axis, so the axis and the
    metric are built from one scored-token multiset and one estimator.
    """

    scopes = [mlp_all(), attn_all()]
    batches = prepare_batches(arm, cohort, max_len=args.max_len, batch_size=args.batch_size)
    targets = sorted({target for scope in scopes for target in scope.resolve(arm.n_layer)})
    bank = build_baseline(
        arm, batches, targets, kind="cohort_mean", cohort_digest=cohort.digest
    )
    run = measure_pathways(arm, batches, scopes, bank)
    baseline = unigram_baseline(
        arm,
        estimator=args.unigram_estimator,
        target_counts=run.target_token_counts,
        reference_counts=reference_counts,
        reference=reference,
    )
    entropy = float(baseline["nats"])
    clean_ce = float(aggregate_variant(run.rows_by_scope["mlp_all"])["clean_ce_nats"])

    metrics: dict[str, Any] = {}
    bootstraps: dict[str, Any] = {}
    for scope in scopes:
        rows = run.rows_by_scope[scope.name]
        # The bootstrap runs first because it is where the denominator's own
        # standard error comes from, and the Fieller precondition
        # ``pathway_metrics`` applies has no fallback without it.
        bootstrap = pathway_cluster_bootstrap(
            rows,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
            unigram_entropy_nats=entropy,
        )
        bootstraps[scope.name] = bootstrap
        metrics[scope.name] = pathway_metrics(
            rows,
            unigram_entropy_nats=entropy,
            context_information_se_nats=bootstrap["context_information_se_nats"],
        )
    # ``measure_pathways`` computes the clean forward once per batch and shares it
    # across scopes, and both bootstraps run from the same seed, so the two scopes
    # produce one standard error for the cohort's context information rather than
    # two. It is published at this level because it is a property of the cohort
    # and the arm, not of an ablation scope, and because
    # ``scaling.analysis_frame`` needs it to decide whether this rung's context
    # information may be divided by at all.
    context_information_se = float(bootstraps["mlp_all"]["context_information_se_nats"])
    return {
        "ablation_baseline_kind": "cohort_mean",
        "ablation_baseline_provenance": bank.provenance,
        "scored_tokens": run.scored_tokens,
        "scored_sequences": run.scored_sequences,
        "unigram_baseline": baseline,
        "clean_ce_nats": clean_ce,
        "context_information_nats": entropy - clean_ce,
        "context_information_se_nats": context_information_se,
        # The sign test, kept under its own name: "off distribution" and
        # "admissible as a denominator" are different findings, and the second is
        # ``context_information_admissibility`` below.
        "context_information_positive": bool(entropy - clean_ce > 0.0),
        "context_information_admissibility": metrics["mlp_all"][
            "context_information_admissibility"
        ],
        "scopes": {
            scope.name: scope_record(scope, run.targets_by_scope[scope.name])
            for scope in scopes
        },
        "metrics": metrics,
        "cluster_bootstrap": bootstraps,
        **pathway_summary(metrics["mlp_all"], metrics["attn_all"]),
    }


def build_repeat_cohort(modality: str, args: argparse.Namespace) -> Cohort:
    """Records containing a genuine repeated span, for the natural-repeat census.

    One repeat cohort per modality, shared by every rung of that modality, so the
    census compares arms on identical probes. The protein cohort is drawn from the
    EC-labelled source because that is the only protein source carrying the
    conditioning labels ZymCTRL's rendering needs.
    """

    if modality == "text":
        return text_repeat_cohort(
            args.repeat_cohort_size,
            max_chars=args.text_repeat_chars,
            criterion=dataclass_replace(
                TEXT_EXACT_CRITERION, min_unit=args.text_repeat_unit
            ),
            scan_documents=args.text_repeat_scan,
            seed=args.cohort_draw_seed or None,
        )
    if modality == "protein":
        # circuits.py now takes a RepeatCriterion value rather than loose
        # keyword arguments, so that every artefact records which probe produced
        # it. The exact criterion is the one this control has always used;
        # --protein-repeat-unit still overrides its minimum unit length.
        return protein_repeat_cohort(
            args.repeat_cohort_size,
            min_len=args.repeat_min_len,
            max_len=args.repeat_max_len,
            criterion=dataclass_replace(
                PROTEIN_EXACT_CRITERION, min_unit=args.protein_repeat_unit
            ),
            seed=args.cohort_draw_seed or None,
        )
    raise ValueError(f"unsupported modality {modality!r}")


def measure_induction(
    arm: Arm,
    cohort: Cohort,
    repeat_cohort: Cohort,
    strings: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Prefix-matching census on both probe families, natural reported as primary.

    The synthetic probe is a random token block repeated in token space. It is
    constructible identically at every rung, which is why it was used alone
    before, but it is off-distribution for a protein decoder and specifically so
    for ProtGPT2: its native rendering breaks a line every 60 residues and a
    128-token synthetic probe carries no break at all. The natural probe uses real
    repeated spans inside real records, so it is in-distribution everywhere.

    Both are measured for every arm rather than one being chosen per arm. Mixing
    probe families across rungs inside a single fit would make the modality
    coefficient partly a probe-family coefficient, which is the exact confound
    this control exists to avoid.
    """

    unigram = fit_unigram(arm, strings, max_tokens=args.unigram_max_tokens)
    ec_label = None
    if arm.spec.input_format == "ec_conditioned":
        labels = cohort.metadata.get("ec_labels")
        if not labels:
            raise ValueError(f"{arm.name}: cohort carries no EC label for a probe prefix")
        ec_label = labels[0]

    synthetic = synthetic_repeat_probes(
        arm,
        unigram,
        n_probes=args.synthetic_probes,
        copy_len=args.synthetic_copy_len,
        seed=args.seed,
        ec_label=ec_label,
    )
    synthetic_alignment = attention_alignment_scores(
        arm, synthetic, batch_size=args.probe_batch_size
    )
    synthetic_census = head_census(synthetic_alignment["scores"]["prefix_matching"])

    natural = natural_repeat_probes(arm, repeat_cohort, max_tokens=args.natural_max_tokens)
    natural_alignment = attention_alignment_scores(
        arm, natural, batch_size=args.natural_batch_size
    )
    natural_census = head_census(natural_alignment["scores"]["prefix_matching"])

    return {
        "primary_probe": PRIMARY_INDUCTION_PROBE,
        "unigram": unigram.summary(),
        "repeat_cohort": cohort_composition_repeat(repeat_cohort),
        "synthetic_census": synthetic_census,
        "natural_census": natural_census,
        "copy_len_tokens": int(args.synthetic_copy_len),
        **induction_summary(
            synthetic_alignment,
            synthetic_census,
            threshold=args.induction_threshold,
            prefix="induction_synthetic_",
        ),
        **induction_summary(
            natural_alignment,
            natural_census,
            threshold=args.induction_threshold,
            prefix="induction_natural_",
        ),
    }


def cohort_composition_repeat(cohort: Cohort) -> dict[str, Any]:
    """Identity of a repeat cohort, which has no ``pathways`` corpus label."""

    lengths = sorted(len(record) for record in cohort.records)
    return {
        "name": cohort.name,
        "kind": cohort.kind,
        "digest": cohort.digest,
        "sequences": len(cohort.records),
        "symbols_min": lengths[0],
        "symbols_median": lengths[len(lengths) // 2],
        "symbols_max": lengths[-1],
    }


def measure_aperture(
    arm: Arm, cohort: Cohort, args: argparse.Namespace
) -> dict[str, Any]:
    """Rank and alignment of the same-position output Jacobian at one mid layer.

    The estimand is ``J = d logits(q) / d h_l(q)``, the same-position block only:
    paths from earlier positions through attention are excluded, so every rank
    here is a lower bound on the full Jacobian rank. That approximation is the
    lens module's and is carried through verbatim in ``jacobian_formulation``.

    One layer is measured rather than a depth sweep because each probe costs
    ``d_model`` reverse-mode passes and this host is reserved for light work; the
    layer is the one nearest half depth, which is where the lens track's own grid
    is densest.
    """

    arm.require("lens")
    head = lens_head(arm)
    freeze_parameters(arm)
    # The Jacobian is a per-position property, so it needs a handful of positions
    # rather than the scoring cohort. It gets its own small window budget because
    # the residual cache and the ablation measurement would otherwise be resident
    # at once, which exhausts a 44 GiB card by the widest text rung.
    gc.collect()
    torch.cuda.empty_cache()
    lens_cohort = subsample_cohort(
        cohort, min(args.aperture_sequences, len(cohort)), args.seed
    )
    windows = prepare_windows(
        arm,
        lens_cohort,
        max_len=args.aperture_max_len,
        batch_size=args.aperture_batch_size,
    )
    layer = analysis_layer(arm.n_layer, args.aperture_depth)
    cache = cache_residuals(arm, windows, [layer], max_bytes=args.aperture_cache_bytes)
    subspace = activation_subspace(cache.residual[layer], device=arm.device)
    gram = jacobian_gram(head)
    probes = sample_jacobian_probes(
        windows,
        count=args.aperture_probes,
        relative_position=args.jacobian_relative_position,
        seed=args.seed,
    )
    rows: list[dict[str, float]] = []
    finite_difference: dict[str, Any] | None = None
    for index, probe in enumerate(probes):
        matrices = jacobian_matrices(arm, head, probe, [layer], chunk=args.aperture_chunk)
        if index == 0:
            finite_difference = jacobian_finite_difference_check(
                arm,
                head,
                probe,
                matrices[layer],
                layer,
                epsilon=args.jacobian_finite_difference_epsilon,
                seed=args.seed,
            )
            relative = finite_difference["relative_error"]
            if relative is None or relative > args.jacobian_finite_difference_tolerance:
                raise FloatingPointError(
                    f"{arm.name}: Jacobian disagrees with a central finite difference "
                    f"by {relative} at layer {layer}"
                )
        alignment = jacobian_alignment(
            matrices[layer],
            gram,
            subspace,
            rank=args.alignment_rank,
            floor_relative=args.spectrum_floor,
        )
        rows.append(jacobian_probe_row(alignment))
        del matrices
    layer_record = {
        "layer": layer,
        "relative_depth": (layer + 1) / arm.n_layer,
        "probes": len(rows),
        "chance_expressed_fraction": args.alignment_rank / arm.d_model,
        "probe_mean": {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]},
    }
    del cache
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "formulation": jacobian_formulation(head, layers=[layer]),
        "finite_difference_check": finite_difference,
        "alignment_rank": int(args.alignment_rank),
        "spectrum_floor": float(args.spectrum_floor),
        "layer_record": layer_record,
        **aperture_summary(
            layer_record, d_model=arm.d_model, vocab_size=int(head.vocab_size)
        ),
    }


def measure_attribution(
    arm: Arm, strings: list[str], args: argparse.Namespace
) -> dict[str, Any]:
    """Direct logit attribution of the correct-next-token logit onto components.

    The reconstruction tolerance matches the value ``src.transfer.circuits``
    declares and is exposed rather than widened. It is a real guard: in bfloat16
    the component sum reproduces ProGen2-medium's final residual only to a
    relative error of 0.070, against 1.6e-06 in float32, and that error grows with
    depth. Since depth varies across the ladder, a loosened tolerance would let a
    depth-correlated numerical artefact enter the very axis this control fits,
    which is why the default inference dtype here is float32 rather than the
    bfloat16 used elsewhere in the transfer suite.
    """

    selected = strings[: args.attribution_sequences]
    ids, mask = tokenize_batch(arm, selected, args.attribution_max_tokens)
    attribution = direct_logit_attribution(
        arm, ids, mask, reconstruction_tolerance=args.attribution_tolerance
    )
    return {
        "n_sequences": len(selected),
        "max_tokens": int(args.attribution_max_tokens),
        # Recorded on the axis itself, not only in the run configuration: the
        # reconstruction guard this decomposition depends on is dtype-sensitive,
        # so a reader has to be able to see the dtype beside the number.
        "inference_dtype": arm.dtype,
        "reconstruction_tolerance": float(args.attribution_tolerance),
        "mean_contribution": attribution["mean_contribution"],
        "pathway_signed_mean": attribution["pathway_signed_mean"],
        **attribution_summary(attribution),
    }


def measure_member(
    member: LadderMember,
    probe: dict[str, Any],
    pools: dict[tuple[str, int, int], Cohort],
    references: dict[tuple[str, int, int], Cohort],
    repeat_cohorts: dict[str, Cohort],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Every axis for one ladder rung, on one seeded draw from its native pool.

    Before the native cohort decides anything, the rung is also scored on every
    other cohort in the run that its input format can render. That sweep is
    budget-only and costs one forward pass per cohort, and it is what turns the
    in-distribution exclusion from an assertion into a measurement: a rung
    excluded on its native band is excluded on the evidence that it also failed
    or passed elsewhere, both of which are recorded.
    """

    started = time.time()
    register_arm_spec(member, probe)
    arm = load_arm(
        member.name,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    torch.cuda.reset_peak_memory_stats(arm.device)

    sweep: dict[tuple[str, int, int], dict[str, Any]] = {}
    draws: dict[tuple[str, int, int], Cohort] = {}
    held_out_counts: dict[tuple[str, int, int], Any] = {}
    held_out_records: dict[tuple[str, int, int], dict[str, Any]] = {}
    unrenderable: dict[str, str] = {}
    for key, pool in sorted(pools.items()):
        ok, reason = renderable(member, key[0])
        if not ok:
            unrenderable[str(list(key))] = str(reason)
            continue
        draws[key] = subsample_cohort(pool, args.n_seq, args.seed)
        held_out, retention = held_out_cohort(references[key], draws[key])
        assert_disjoint(draws[key], held_out)
        reference_counts = cohort_target_token_counts(arm, held_out, max_len=args.max_len)
        target_counts = cohort_target_token_counts(arm, draws[key], max_len=args.max_len)
        baseline = unigram_baseline(
            arm,
            estimator=args.unigram_estimator,
            target_counts=target_counts,
            reference_counts=reference_counts,
            reference={"cohort": held_out.name, "digest": held_out.digest, **retention},
        )
        sweep[key] = {
            "power": arm_power(
                arm, draws[key], max_len=args.max_len, batch_size=args.batch_size
            ),
            "unigram_entropy_nats": baseline["nats"],
            "unigram_entropy_plug_in_nats": baseline["cohort_plug_in_entropy_nats"],
            "unigram_estimator": baseline["estimator"],
            "baseline": baseline,
        }
        # Kept so the native cohort does not re-tokenise a 40k-sequence reference
        # corpus that has already been counted for this arm.
        held_out_counts[key] = reference_counts
        held_out_records[key] = {
            "cohort": held_out.name,
            "digest": held_out.digest,
            "corpus": key[0],
            **retention,
        }

    native = member.cohort_key
    if native not in sweep:
        raise RuntimeError(f"{member.name}: its own native cohort could not be scored")
    cohort = draws[native]
    strings = cohort.input_strings(arm)
    pathways = (
        measure_pathway_shares(
            arm, cohort, held_out_counts[native], held_out_records[native], args
        )
        if arm.supports("pathway")
        else None
    )
    # An arm without the pathway capability still needs a convergence axis, so the
    # baseline and clean cross-entropy come from the budget stage instead. The two
    # stages score the same multiset for every arm whose input format carries no
    # prompt scaffolding, which is every arm in that position.
    budget_native = sweep[native]
    convergence = convergence_row(
        pathways["unigram_baseline"] if pathways is not None else budget_native["baseline"],
        clean_ce_nats=(
            pathways["clean_ce_nats"]
            if pathways is not None
            else float(budget_native["power"]["clean_ce_nats"])
        ),
        # The only bootstrap of this denominator on this rung is the pathway
        # stage's, so an arm without the pathway capability records no standard
        # error at all. That is passed through as None: analysis_frame then
        # withholds the admissibility verdict rather than substituting the
        # retired constant for a measurement that was never made.
        context_information_se_nats=(
            None if pathways is None else pathways["context_information_se_nats"]
        ),
        symbols_per_token=symbols_per_token(arm, strings, args.max_len),
        n_scored_tokens=(
            pathways["scored_tokens"]
            if pathways is not None
            else int(budget_native["power"]["n_scored_tokens"])
        ),
        vocab_size=int(arm.model.config.vocab_size),
        n_parameters=sum(p.numel() for p in arm.model.parameters()),
        n_layer=arm.n_layer,
        d_model=arm.d_model,
    )

    record: dict[str, Any] = {
        "name": member.name,
        "modality": member.modality,
        "tokenisation": member.tokenisation,
        "input_format": member.input_format,
        # Three separate facts, all named. `source` is the legacy spelling of
        # `evaluation_cohort_source` and is kept because frozen artefacts use it;
        # writing only that one made scaling.analysis_frame fall back to the panel
        # declaration for the other two, so a rung's pretraining corpus -- the
        # field every corpus contrast is defined against -- was recovered rather
        # than recorded.
        "source": member.source,
        "evaluation_cohort_source": member.evaluation_cohort_source,
        "pretraining_corpus": member.pretraining_corpus,
        "path": str(member.path),
        "architecture": member.architecture,
        "capabilities": sorted(member.capabilities),
        "dtype": arm.dtype,
        # n_head validates the GPT-2 head decomposition, which is a circuits-family
        # assumption; a T5-derived arm need not satisfy it and is not asked to.
        "n_head": n_head(arm) if arm.supports("circuits") else None,
        "cohort_corpus": member.cohort_corpus,
        "cohort_min_symbols": member.cohort_min_symbols,
        "cohort_max_symbols": member.cohort_max_symbols,
        "cohort": cohort_composition(cohort, source=member.cohort_corpus),
        "cohort_sensitivity": cohort_sensitivity_rows(member, sweep),
        "cohorts_not_renderable": unrenderable,
        "budget": sweep[native]["power"],
        "convergence": convergence,
        "pathways": pathways,
    }

    # The induction census and the attribution decomposition do not divide by the
    # cohort's context information, so they remain well defined for an
    # off-distribution model and are measured for every rung whose input format
    # src.transfer.circuits can score. They are still kept out of the fits by
    # analysis_frame, which drops off-distribution rows.
    supported, unsupported_reason = circuits_supported(member)
    supported = supported and arm.supports("circuits")
    if not arm.supports("circuits"):
        unsupported_reason = (
            f"{member.name} ({member.architecture}) does not declare the 'circuits' "
            "capability; it admits no GPT-2-style sublayer decomposition"
        )
    record["circuits_axes_measured"] = supported
    record["circuits_axes_skipped_reason"] = unsupported_reason
    record["induction"] = (
        measure_induction(arm, cohort, repeat_cohorts[member.modality], strings, args)
        if supported
        else None
    )
    record["attribution"] = measure_attribution(arm, strings, args) if supported else None
    lens_ok, lens_reason = lens_supported(member)
    lens_ok = lens_ok and arm.supports("lens")
    record["aperture_measured"] = lens_ok
    record["aperture_skipped_reason"] = None if lens_ok else lens_reason
    record["aperture"] = measure_aperture(arm, cohort, args) if lens_ok else None
    record["runtime_seconds"] = round(time.time() - started, 2)
    record["peak_gpu_bytes"] = int(torch.cuda.max_memory_allocated(arm.device))

    del arm
    gc.collect()
    torch.cuda.empty_cache()
    return record


def build_analysis(frame: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """Fits, contrasts and controls over the measured rungs.

    Two fit families are produced. The deciding one uses every in-distribution
    rung, exactly as the exclusion rule declares. The sensitivity one additionally
    drops rungs whose context information is not admissible as the denominator of
    a ratio, because a share computed against a 0.18-nat denominator is a ratio to
    noise and one such rung can dominate an interval on its own. The criterion is
    ``budget.ratio_denominator_admissibility`` -- Fieller's precondition against
    each rung's own bootstrap standard error -- and not the retired 0.30-nat
    constant, which EXP-R2-218 measured to be up to 3.2x too lax for exactly this
    job. A rung that recorded no standard error is admitted to neither side and is
    named in its own list, because the criterion cannot be evaluated for it and a
    constant may not stand in.

    The sensitivity family is reported and never consulted for the verdict: it
    exists so that a reader can see whether the deciding interval is wide because
    the evidence is weak or because one denominator is small.
    """

    fits = {
        f"{metric}~{axis}": fit_modality_offset(frame, metric_key=metric, axis_key=axis)
        for metric in INTERPRETABILITY_METRICS
        for axis in CONVERGENCE_AXES
    }
    adjusted_fits = {
        f"{metric}~{axis}": fit_modality_offset(
            frame, metric_key=metric, axis_key=axis, include_tokenisation=True
        )
        for metric in INTERPRETABILITY_METRICS
        for axis in CONVERGENCE_AXES
    }
    admissible_frame = [row for row in frame if row["measurable_denominator"] is True]
    sensitivity_fits = {
        f"{metric}~{axis}": fit_modality_offset(
            admissible_frame, metric_key=metric, axis_key=axis
        )
        for metric in INTERPRETABILITY_METRICS
        for axis in CONVERGENCE_AXES
    }
    sensitivity_adjusted = {
        f"{metric}~{axis}": fit_modality_offset(
            admissible_frame, metric_key=metric, axis_key=axis, include_tokenisation=True
        )
        for metric in INTERPRETABILITY_METRICS
        for axis in CONVERGENCE_AXES
    }
    sensitivity_decision = decide_verdict(
        sensitivity_fits,
        tokenisation_adjusted_fits=sensitivity_adjusted,
        primary_metric=PRIMARY_METRIC,
        primary_axis=PRIMARY_AXIS,
        equivalence_margin=args.equivalence_margin,
        min_residual_dof=args.min_residual_dof,
    )
    contrasts = {
        metric: nearest_neighbour_contrasts(frame, metric_key=metric, axis_key=PRIMARY_AXIS)
        for metric in INTERPRETABILITY_METRICS
    }
    decision = decide_verdict(
        fits,
        tokenisation_adjusted_fits=adjusted_fits,
        primary_metric=PRIMARY_METRIC,
        primary_axis=PRIMARY_AXIS,
        equivalence_margin=args.equivalence_margin,
        min_residual_dof=args.min_residual_dof,
    )
    # ZymCTRL's EC tag very nearly determines the family, so anything
    # family-flavoured read from it may be reading the prompt, and its realized
    # information fraction is the highest in the panel partly for that reason.
    # Every fit is therefore repeated without the conditioned rungs, reported and
    # never consulted for the verdict, so a reader can see what it drives.
    unconditioned_frame = [row for row in frame if not row["conditioning_leak"]]
    unconditioned_fits = {
        f"{metric}~{axis}": fit_modality_offset(
            unconditioned_frame, metric_key=metric, axis_key=axis
        )
        for metric in INTERPRETABILITY_METRICS
        for axis in CONVERGENCE_AXES
    }
    unconditioned_adjusted = {
        f"{metric}~{axis}": fit_modality_offset(
            unconditioned_frame, metric_key=metric, axis_key=axis, include_tokenisation=True
        )
        for metric in INTERPRETABILITY_METRICS
        for axis in CONVERGENCE_AXES
    }
    unconditioned_decision = decide_verdict(
        unconditioned_fits,
        tokenisation_adjusted_fits=unconditioned_adjusted,
        primary_metric=PRIMARY_METRIC,
        primary_axis=PRIMARY_AXIS,
        equivalence_margin=args.equivalence_margin,
        min_residual_dof=args.min_residual_dof,
    )
    return {
        "tokenisation_adjusted_fits": adjusted_fits,
        "conditioning_leak_sensitivity": {
            "deciding": False,
            "rule": "drop every rung whose prompt carries a conditioning tag",
            "members_dropped": [
                row["name"] for row in frame if row["conditioning_leak"]
            ],
            "members_retained": [row["name"] for row in unconditioned_frame],
            "fits": unconditioned_fits,
            "tokenisation_adjusted_fits": unconditioned_adjusted,
            "would_be_verdict": unconditioned_decision,
        },
        "denominator_admissibility_sensitivity": {
            "deciding": False,
            "criterion": "fieller_precondition_on_the_denominator",
            "criterion_provenance": (
                "src.transfer.budget.ratio_denominator_admissibility: "
                "I_hat > FIELLER_DENOMINATOR_MULTIPLE * SE(I_hat), evaluated "
                "against each rung's own bootstrap standard error. It is a "
                "per-arm bound and not a constant, so no floor in nats is quoted "
                "here"
            ),
            "members_retained": [row["name"] for row in admissible_frame],
            "members_dropped": [
                row["name"] for row in frame if row["measurable_denominator"] is False
            ],
            "members_without_a_denominator_standard_error": [
                row["name"] for row in frame if row["measurable_denominator"] is None
            ],
            "unavailable_note": NO_DENOMINATOR_STANDARD_ERROR,
            "fits": sensitivity_fits,
            "tokenisation_adjusted_fits": sensitivity_adjusted,
            "would_be_verdict": sensitivity_decision,
        },
        "frame": frame,
        "fits": fits,
        "nearest_neighbour_contrasts_on_primary_axis": contrasts,
        "tokenisation_control": tokenisation_contrast(
            frame, keys=list(INTERPRETABILITY_METRICS) + list(CONVERGENCE_AXES)
        ),
        "conditioning_control": conditioning_control(
            frame, keys=list(INTERPRETABILITY_METRICS) + list(CONVERGENCE_AXES)
        ),
        "paired_data_contrast": paired_architecture_contrast(
            frame, keys=list(INTERPRETABILITY_METRICS) + list(CONVERGENCE_AXES)
        ),
        "distribution_control": distribution_control(frame),
        "verdict": decision["verdict"],
        "verdict_detail": decision,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ladder-table",
        type=Path,
        default=DEFAULT_LADDER_TABLE,
        help="optional operator-supplied ladder declaration; defaults to the code contract",
    )
    parser.add_argument(
        "--members",
        nargs="+",
        default=None,
        help="subset of configured ladder members to measure; default is all of them",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    # A concurrent process has twice deleted results/transfer between a
    # run finishing and its output being read. Nothing under scripts/transfer or
    # src/transfer removes it, so the second copy is written under logs/, which is
    # local-only, until the culprit is found.
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP)
    parser.add_argument("--seed", type=int, default=20260728)

    parser.add_argument("--pool-size", type=int, default=2000)
    parser.add_argument(
        "--cohort-draw-seed",
        type=int,
        default=DEFAULT_CORPUS_DRAW_SEED,
        help="seed for the permutation every ladder pool is drawn under; 0 "
        "selects the historical file-order prefix, which is a declared choice "
        "and not a default (transfer audit, Appendix B rule 1)",
    )
    parser.add_argument("--n-seq", type=int, default=32)
    # 512 rather than 256: ProtGPT2's context information keeps rising with the
    # scored window on full-length proteins (+0.61 nats at 256 tokens against
    # +0.98 at 512 on the 600-2000 band), and it is the rung whose denominator
    # decides whether the modality coefficient is identified at all.
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)

    parser.add_argument(
        "--unigram-estimator", default="disjoint", choices=list(UNIGRAM_ESTIMATORS)
    )
    parser.add_argument("--unigram-reference-size", type=int, default=40000)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=20260728)

    parser.add_argument("--unigram-max-tokens", type=int, default=256)
    parser.add_argument("--synthetic-probes", type=int, default=8)
    parser.add_argument("--synthetic-copy-len", type=int, default=64)
    parser.add_argument("--probe-batch-size", type=int, default=4)
    parser.add_argument(
        "--induction-threshold",
        type=float,
        default=DEFAULT_INDUCTION_THRESHOLD,
        choices=list(INDUCTION_THRESHOLDS),
    )

    parser.add_argument("--aperture-depth", type=float, default=0.5)
    parser.add_argument("--aperture-probes", type=int, default=4)
    parser.add_argument("--aperture-chunk", type=int, default=32)
    parser.add_argument("--aperture-cache-bytes", type=int, default=2 * 2**30)
    parser.add_argument("--aperture-sequences", type=int, default=8)
    parser.add_argument("--aperture-batch-size", type=int, default=2)
    parser.add_argument("--aperture-max-len", type=int, default=256)
    parser.add_argument("--alignment-rank", type=int, default=64)
    parser.add_argument("--spectrum-floor", type=float, default=1e-6)
    parser.add_argument("--jacobian-relative-position", type=float, default=0.6)
    parser.add_argument("--jacobian-finite-difference-epsilon", type=float, default=1e-1)
    parser.add_argument(
        "--jacobian-finite-difference-tolerance", type=float, default=2e-2
    )

    parser.add_argument("--repeat-cohort-size", type=int, default=16)
    parser.add_argument("--repeat-min-len", type=int, default=200)
    parser.add_argument("--repeat-max-len", type=int, default=600)
    parser.add_argument("--protein-repeat-unit", type=int, default=16)
    parser.add_argument("--text-repeat-chars", type=int, default=2000)
    parser.add_argument("--text-repeat-unit", type=int, default=40)
    parser.add_argument("--text-repeat-scan", type=int, default=3000)
    parser.add_argument("--natural-max-tokens", type=int, default=640)
    parser.add_argument("--natural-batch-size", type=int, default=2)

    parser.add_argument("--attribution-sequences", type=int, default=8)
    parser.add_argument("--attribution-max-tokens", type=int, default=256)
    parser.add_argument("--attribution-tolerance", type=float, default=0.02)

    parser.add_argument(
        "--equivalence-margin", type=float, default=DEFAULT_EQUIVALENCE_MARGIN
    )
    parser.add_argument("--min-residual-dof", type=int, default=DEFAULT_MIN_RESIDUAL_DOF)

    args = parser.parse_args()
    if args.n_seq > args.pool_size:
        raise ValueError("--n-seq cannot exceed --pool-size")
    if args.members is not None and len(set(args.members)) != len(args.members):
        raise ValueError("--members repeats a member")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required")
    torch.manual_seed(args.seed)

    ladder, ladder_provenance = resolve_ladder(args.ladder_table)
    configured = {member.name: member for member in ladder}
    if args.members is not None:
        unknown = sorted(set(args.members) - set(configured))
        if unknown:
            raise KeyError(
                f"requested members {unknown} are not in the configured ladder "
                f"{sorted(configured)}"
            )
        requested = [configured[name] for name in args.members]
    else:
        requested = list(ladder)

    probes = {member.name: inspect_member(member) for member in requested}
    available = [member for member in requested if probes[member.name]["available"]]
    for member in requested:
        state = probes[member.name]
        print(
            f"[ladder] {member.name:16s} {'available' if state['available'] else 'ABSENT'}"
            f"{'' if state['available'] else ': ' + str(state['unavailable_reason'])}",
            flush=True,
        )
    if not available:
        raise RuntimeError("no configured ladder member is available on this host")

    draw_seed = args.cohort_draw_seed or None
    pools = {
        key: build_pool(key, args.pool_size, seed=draw_seed)
        for key in sorted({member.cohort_key for member in available})
    }
    # The reference starts one whole pool past the measurement draw, and
    # held_out_cohort then removes anything still shared by content: Swiss-Prot
    # and the EC corpus both carry the same sequence under several accessions, so
    # skipping by position alone leaves a real leak. Under a seed the two are
    # disjoint windows of one permutation rather than adjacent blocks of the
    # file, which is what makes the reference a sample of the corpus rather than
    # of whichever families happen to follow the measurement block.
    references = {
        key: build_pool(key, args.unigram_reference_size, skip=args.pool_size, seed=draw_seed)
        for key in sorted(pools)
    }
    for key, pool in sorted(pools.items()):
        print(
            f"[cohort] {key[0]:22s} band={key[1]}-{key[2]} "
            f"n={len(pool)} digest={pool.digest[:12]} "
            f"reference_n={len(references[key])}",
            flush=True,
        )

    repeat_cohorts = {
        modality: build_repeat_cohort(modality, args)
        for modality in sorted({member.modality for member in available})
    }
    for modality, repeat in sorted(repeat_cohorts.items()):
        print(
            f"[repeat] {modality:8s} n={len(repeat)} digest={repeat.digest[:12]}",
            flush=True,
        )

    started = datetime.now(timezone.utc).isoformat()
    output_dir = args.output_dir.resolve()
    records: list[dict[str, Any]] = []
    backup_dir = args.backup_dir.resolve()
    for member in available:
        record = measure_member(
            member, probes[member.name], pools, references, repeat_cohorts, args
        )
        write_json(output_dir / "members" / f"{member.name}.json", record)
        write_json(backup_dir / "members" / f"{member.name}.json", record)
        records.append(record)
        convergence = record["convergence"]
        pathways = record["pathways"]
        share = (
            None if pathways is None else pathways["mlp_share_of_context_information"]
        )
        print(
            f"[{member.name}] cohort={member.cohort_corpus}"
            f"[{member.cohort_min_symbols}-{member.cohort_max_symbols}] "
            f"rif={convergence['realized_information_fraction']:+.4f} "
            f"ce={convergence['clean_ce_bits_per_symbol']:.4f} bits/symbol "
            f"in_distribution={convergence['in_distribution']} "
            f"mlp_share={'n/a' if share is None else f'{share:+.4f}'} "
            f"({record['runtime_seconds']}s, "
            f"{record['peak_gpu_bytes'] / 2**30:.1f} GiB)",
            flush=True,
        )

    frame = analysis_frame(records)
    analysis = build_analysis(frame, args)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": sha256_file(Path(__file__)),
        "scaling_module_sha256": sha256_file(REPO_ROOT / "src" / "transfer" / "scaling.py"),
        "arms_module_sha256": sha256_file(REPO_ROOT / "src" / "transfer" / "arms.py"),
        "ladder_provenance": ladder_provenance,
        "ladder_configured": [member.name for member in requested],
        "ladder_used": [member.name for member in available],
        "ladder_absent": {
            name: state["unavailable_reason"]
            for name, state in probes.items()
            if not state["available"]
        },
        "ladder_probe": probes,
        "configuration": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in sorted(vars(args).items())
        },
        "estimands": {
            "primary_metric": PRIMARY_METRIC,
            "primary_axis": PRIMARY_AXIS,
            "convergence_axes": list(CONVERGENCE_AXES),
            "interpretability_metrics": list(INTERPRETABILITY_METRICS),
            "pathway_scopes": list(PATHWAY_SCOPES),
        },
        "cohort_pools": [
            {
                "cohort_corpus": key[0],
                "cohort_min_symbols": key[1],
                "cohort_max_symbols": key[2],
                "members": [
                    member.name for member in available if member.cohort_key == key
                ],
                **cohort_composition(pool, source=key[0]),
            }
            for key, pool in sorted(pools.items())
        ],
        "members": records,
        **analysis,
    }
    write_json(output_dir / "convergence_control.json", payload)
    write_json(backup_dir / "convergence_control.json", payload)
    sensitivity = payload["denominator_admissibility_sensitivity"]
    print(
        f"verdict={payload['verdict']}: {payload['verdict_detail']['reason']}",
        flush=True,
    )
    print(
        "non-deciding sensitivity (dropping "
        f"{sensitivity['members_dropped']} as inadmissible denominators, "
        f"{sensitivity['members_without_a_denominator_standard_error']} unevaluated "
        "for want of a standard error): "
        f"{sensitivity['would_be_verdict']['verdict']} - "
        f"{sensitivity['would_be_verdict']['reason']}",
        flush=True,
    )
    print(f"wrote {output_dir / 'convergence_control.json'}", flush=True)


if __name__ == "__main__":
    main()
