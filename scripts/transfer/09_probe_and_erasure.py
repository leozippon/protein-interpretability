#!/usr/bin/env python3
"""Decodability versus reliance, per arm: probe suite plus LEACE concept erasure.

Two questions are routinely conflated when interpretability methods are carried
from text decoders to protein decoders. *Decodability* asks whether a concept is
present in the activations, and a probe answers it. *Reliance* asks whether the
next-token computation consults the concept, and only an intervention answers
it. A protein decoder can encode fold state or family identity that its
next-residue prediction never reads; attributing the transfer gap requires
telling those two cases apart rather than reporting the first and implying the
second.

This entry point runs, for one arm: linear and small-MLP probes on the residual
stream at a relative-depth grid, under group-disjoint cross-validation whose
grouping variable is named in the output; then a LEACE erasure of each concept
at one prespecified relative depth, verified by the collapse of the linear probe
to chance, and applied inside the forward pass to measure the cost in nats per
token against a mean-ablation reference and a dimension-matched random-direction
control.

Refusals are outputs, not failures, and they are per concept rather than per
arm. ProtGPT2's multi-residue BPE has no residue-to-token map, so it is refused
for the residue-level structure concepts and the variant-level fitness concept
and measured on the sequence-level ones, where only the token span covering the
sequence is needed. ZymCTRL is handed a functional tag in its own prompt, which
on an enzyme cohort nearly determines both EC class and Pfam family: under
``--ec-conditioning native`` those sequence-level concepts are refused rather
than reported with a caveat, and ``fixed`` (native format, one constant tag) or
``unconditioned`` (bare sequence, off the arm's training distribution) make them
measurable. Residue-level structure concepts are not affected by that leak and
are measured under native conditioning. Every refusal, and the conditioning mode
that would lift it, is written into the artifact.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer.io import write_json  # noqa: E402
from src.transfer.arms import PANEL, REPO, Arm, ArmSpec, load_arm  # noqa: E402
from src.transfer.probes import (  # noqa: E402
    CHANCE_MODEL,
    CONCEPTS,
    EC_CONDITIONING_MODES,
    ERASURE_CITATION,
    ERASURE_METHOD,
    LAYER_FRACTIONS,
    FIXED_EC_TAG,
    PRIMARY_CONTROL,
    SCHEMA_VERSION,
    SampleSet,
    Unit,
    analysis_layer_grid,
    collect_samples,
    combined_digest,
    concept_report,
    concepts_for_modality,
    ec_units,
    erasure_layer_for,
    fitness_units,
    format_skill,
    pfam_units,
    refusal_reason,
    restrict_labels,
    structure_units,
    text_units,
    token_budget,
)

DEFAULT_OUT = (
    REPO / "results/transfer_20260728/probe_erasure"
)

#: What each EC-conditioning mode buys and costs, recorded in every artifact.
EC_CONDITIONING_INTERPRETATION: dict[str, str] = {
    "native": (
        "the arm's own prompt format. It is in-distribution, and it hands the "
        "model a functional tag that nearly determines any sequence-level "
        "function or family label, so those concepts are refused on this arm "
        "rather than reported. Residue-level structure concepts are unaffected: "
        "the tag constrains a protein's chemistry but not which residue at which "
        "position is helical or buried."
    ),
    "fixed": (
        "the native format with one constant EC tag on every prompt. The tag "
        "carries zero information about any label, so sequence-level concepts "
        "become measurable, at the cost of conditioning each protein on a tag "
        "that is not its own."
    ),
    "unconditioned": (
        "the bare sequence, with no tag and none of the native delimiters. The "
        "leak is gone and so is the arm's training distribution; its "
        "cross-entropy is not comparable with its native-format numbers."
    ),
}

#: Concepts that share one collection pass because they share one cohort.
CONCEPT_GROUPS: dict[str, tuple[str, ...]] = {
    "structure": ("ss3", "burial"),
    "ec": ("ec_class",),
    "pfam": ("pfam_family",),
    "fitness": ("fitness",),
    "text": ("next_token_class", "next_token_rarity"),
}


def build_units(
    group: str, arm: Arm, args: argparse.Namespace
) -> tuple[list[Unit], dict[str, Any]]:
    """Build the units of one concept group in the arm's native input format."""

    if group == "structure":
        return structure_units(
            arm,
            n_proteins=args.n_structures,
            scan_models=args.scan_models,
            min_len=args.min_len,
            max_len=args.max_len,
            min_plddt=args.min_plddt,
            residues_per_protein=args.residues_per_protein,
            seed=args.seed,
            kmer=args.homology_kmer,
            kmer_jaccard=args.homology_kmer_jaccard,
            pfam_jaccard=args.homology_pfam_jaccard,
            ec_conditioning=args.ec_conditioning,
        )
    if group == "ec":
        return ec_units(
            arm,
            n_proteins=args.n_ec_proteins,
            min_len=args.min_len,
            max_len=args.max_len,
            positions_per_protein=args.positions_per_sequence,
            max_per_family=args.max_ec_proteins_per_family,
            seed=args.seed,
            ec_conditioning=args.ec_conditioning,
        )
    if group == "pfam":
        return pfam_units(
            arm,
            n_families=args.n_pfam_families,
            proteins_per_family=args.pfam_proteins_per_family,
            min_len=args.min_len,
            max_len=args.max_len,
            positions_per_protein=args.positions_per_sequence,
            redundancy_kmer=args.redundancy_kmer,
            redundancy_jaccard=args.redundancy_kmer_jaccard,
            seed=args.seed,
            ec_conditioning=args.ec_conditioning,
        )
    if group == "fitness":
        return fitness_units(
            arm,
            n_assays=args.n_assays,
            variants_per_assay=args.variants_per_assay,
            min_len=args.fitness_min_len,
            max_len=args.fitness_max_len,
            seed=args.seed,
            ec_conditioning=args.ec_conditioning,
        )
    if group == "text":
        return text_units(
            arm,
            n_documents=args.n_documents,
            positions_per_document=args.positions_per_document,
            max_tokens=args.max_tokens,
            min_chars=args.min_document_chars,
            seed=args.seed,
        )
    raise ValueError(f"unknown concept group {group!r}")


def groups_for(concepts: list[str]) -> list[str]:
    """The collection passes needed to cover exactly the requested concepts."""

    needed = []
    for group, members in CONCEPT_GROUPS.items():
        if any(concept in concepts for concept in members):
            if not set(members) <= set(concepts):
                raise ValueError(
                    f"concepts {sorted(set(members) - set(concepts))} share one "
                    f"cohort with {sorted(set(members) & set(concepts))} and cannot "
                    "be measured separately; request the whole group"
                )
            needed.append(group)
    return needed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="progen2-medium")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--concepts",
        nargs="+",
        default=None,
        help="default: every concept this arm is not refused for",
    )
    parser.add_argument("--layer-fractions", type=float, nargs="+", default=list(LAYER_FRACTIONS))
    parser.add_argument("--erasure-fraction", type=float, default=0.5)

    parser.add_argument("--n-structures", type=int, default=150)
    parser.add_argument("--scan-models", type=int, default=4000)
    parser.add_argument("--residues-per-protein", type=int, default=24)
    parser.add_argument("--min-plddt", type=float, default=70.0)
    parser.add_argument("--min-len", type=int, default=110)
    parser.add_argument("--max-len", type=int, default=320)
    parser.add_argument("--homology-kmer", type=int, default=3)
    parser.add_argument("--homology-kmer-jaccard", type=float, default=0.10)
    parser.add_argument("--homology-pfam-jaccard", type=float, default=0.50)

    parser.add_argument("--n-ec-proteins", type=int, default=400)
    parser.add_argument("--max-ec-proteins-per-family", type=int, default=4)
    parser.add_argument("--n-pfam-families", type=int, default=10)
    parser.add_argument("--pfam-proteins-per-family", type=int, default=30)
    parser.add_argument("--positions-per-sequence", type=int, default=8)
    parser.add_argument("--redundancy-kmer", type=int, default=5)
    parser.add_argument("--redundancy-kmer-jaccard", type=float, default=0.20)

    parser.add_argument("--n-assays", type=int, default=10)
    parser.add_argument("--variants-per-assay", type=int, default=80)
    parser.add_argument("--fitness-min-len", type=int, default=40)
    parser.add_argument("--fitness-max-len", type=int, default=300)

    parser.add_argument("--n-documents", type=int, default=150)
    parser.add_argument("--positions-per-document", type=int, default=24)
    parser.add_argument("--min-document-chars", type=int, default=800)

    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--probe-dim", type=int, default=64)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--mlp-hidden", type=int, nargs="+", default=[128])
    parser.add_argument("--mlp-max-iter", type=int, default=300)
    parser.add_argument("--max-scored-units", type=int, default=40)
    parser.add_argument("--max-post-erasure-linear-skill", type=float, default=0.05)
    parser.add_argument("--min-clean-linear-skill", type=float, default=0.10)
    parser.add_argument("--min-ce-denominator", type=float, default=0.05)
    parser.add_argument(
        "--ec-conditioning",
        choices=list(EC_CONDITIONING_MODES),
        default="native",
        help=(
            "how an EC-conditioned arm is prompted; 'native' leaks the tag into "
            "sequence-level probes and refuses them, 'fixed' keeps the format "
            f"with the constant tag {FIXED_EC_TAG}, 'unconditioned' drops the tag "
            "and takes the arm off its training distribution"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _spec_only_arm(spec: ArmSpec) -> Arm:
    """An ``Arm`` carrying only its declaration, for command-line-time checks.

    ``refusal_reason``, ``analysis_layer_grid`` and ``erasure_layer_for`` all read
    ``ArmSpec`` alone -- capabilities, modality, tokenisation, ``n_layer`` -- so
    the checks they support do not need a checkpoint. Passing ``None`` for the
    model and tokenizer makes that explicit: if any of them ever starts touching
    the weights, this raises ``AttributeError`` immediately rather than quietly
    validating something different from what it validated before.
    """

    return Arm(spec=spec, model=None, tokenizer=None, device="", dtype="")


def main() -> None:
    args = parse_args()
    if args.arm not in PANEL:
        raise ValueError(f"unknown arm {args.arm!r}; panel is {sorted(PANEL)}")
    spec = PANEL[args.arm]
    requested = list(args.concepts) if args.concepts else None
    if requested is not None:
        unknown = sorted(set(requested) - set(CONCEPTS))
        if unknown:
            raise ValueError(f"unknown concepts {unknown}; known are {sorted(CONCEPTS)}")

    # Everything below reads ArmSpec, not the checkpoint: refusal_reason consults
    # arm.spec and arm.supports(...), the concept list comes from the declared
    # modality, and the erasure grid is arithmetic on the declared n_layer. All of
    # it used to run after load_arm, so a --concepts/--arm pair that could never
    # work, or an --erasure-fraction outside --layer-fractions, cost a checkpoint
    # load first. A spec-only Arm makes them command-line-time checks.
    declared = _spec_only_arm(spec)
    candidates = list(requested) if requested is not None else list(
        concepts_for_modality(spec.modality)
    )
    declared_refusals = [
        {
            "concept": concept,
            "reason": refusal_reason(concept, declared, ec_conditioning=args.ec_conditioning),
        }
        for concept in candidates
        if refusal_reason(concept, declared, ec_conditioning=args.ec_conditioning) is not None
    ]
    if requested is not None and declared_refusals:
        raise ValueError(
            f"{spec.name} is refused for explicitly requested concepts: {declared_refusals}"
        )
    declared_layers = analysis_layer_grid(declared, args.layer_fractions)
    declared_erasure_layer = erasure_layer_for(declared, args.erasure_fraction)
    if declared_erasure_layer not in declared_layers:
        raise ValueError(
            f"erasure depth {args.erasure_fraction} resolves to layer "
            f"{declared_erasure_layer}, which is not in the probe grid "
            f"{declared_layers}; the erasure layer must also be probed so its clean "
            "and erased skills are the same measurement"
        )

    arm = load_arm(args.arm, device=args.device, dtype=args.dtype)
    candidates = list(requested) if requested is not None else list(
        concepts_for_modality(arm.modality)
    )
    refusals = [
        {
            "concept": concept,
            "reason": refusal_reason(concept, arm, ec_conditioning=args.ec_conditioning),
        }
        for concept in candidates
        if refusal_reason(concept, arm, ec_conditioning=args.ec_conditioning) is not None
    ]
    if requested is not None and refusals:
        raise ValueError(
            f"{arm.name} is refused for explicitly requested concepts: {refusals}"
        )
    concepts = [
        concept
        for concept in candidates
        if refusal_reason(concept, arm, ec_conditioning=args.ec_conditioning) is None
    ]

    created = datetime.now(timezone.utc).isoformat()
    arm_block = {
        "arm": arm.name,
        "path": str(spec.path),
        "modality": spec.modality,
        "n_layer": arm.n_layer,
        "d_model": arm.d_model,
        "tokenisation": spec.tokenisation,
        "input_format": spec.input_format,
        "source": spec.source,
        "dtype": args.dtype,
        "device": args.device,
        "ec_conditioning": args.ec_conditioning,
        "fixed_ec_tag": FIXED_EC_TAG if args.ec_conditioning == "fixed" else None,
        "native_distribution": args.ec_conditioning != "unconditioned",
    }
    # A non-native conditioning mode is a different measurement of the same
    # arm, so it gets its own file rather than overwriting the native one.
    suffix = "" if args.ec_conditioning == "native" else f"__ec_{args.ec_conditioning}"
    destination = args.out / f"{arm.name}{suffix}.json"
    if not concepts:
        write_json(
            destination,
            {
                "schema_version": SCHEMA_VERSION,
                "artifact": "probe_and_erasure_report",
                "status": "refused_all_concepts",
                "created_utc": created,
                "arm_spec": arm_block,
                "refused_concepts": refusals,
                "seeds": {"cohort_and_split": int(args.seed)},
            },
        )
        print(f"wrote {destination}")
        for row in refusals:
            print(f"refused {row['concept']}: {row['reason']}")
        return

    layers = analysis_layer_grid(arm, args.layer_fractions)
    erasure_layer = erasure_layer_for(arm, args.erasure_fraction)
    if erasure_layer not in layers:
        raise ValueError(
            f"erasure depth {args.erasure_fraction} resolves to layer {erasure_layer}, "
            f"which is not in the probe grid {layers}; the erasure layer must also be "
            "probed so its clean and erased skills are the same measurement"
        )

    sample_sets: dict[str, SampleSet] = {}
    for group in groups_for(concepts):
        units, construction = build_units(group, arm, args)
        collected = collect_samples(
            arm,
            units,
            layers=layers,
            max_tokens=args.max_tokens,
            construction=construction,
        )
        for concept in CONCEPT_GROUPS[group]:
            sample_sets[concept] = restrict_labels(
                collected[concept], min_groups_per_label=args.n_splits
            )
        print(
            f"collected {group}: {len(units)} units, "
            f"{sample_sets[CONCEPT_GROUPS[group][0]].n_samples} samples"
        )

    reports = {
        concept: concept_report(
            arm,
            samples,
            layers=layers,
            layer_fractions=args.layer_fractions,
            erasure_layer=erasure_layer,
            erasure_fraction=args.erasure_fraction,
            n_splits=args.n_splits,
            seed=args.seed,
            probe_dim=args.probe_dim,
            n_bootstrap=args.n_bootstrap,
            mlp_hidden=tuple(args.mlp_hidden),
            mlp_max_iter=args.mlp_max_iter,
            max_post_erasure_skill=args.max_post_erasure_linear_skill,
            min_clean_skill=args.min_clean_linear_skill,
            minimum_ce_denominator=args.min_ce_denominator,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            max_scored_units=args.max_scored_units,
            ec_conditioning=args.ec_conditioning,
        )
        for concept, samples in sample_sets.items()
    }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "probe_and_erasure_report",
        "status": "measured",
        "created_utc": created,
        "arm_spec": arm_block,
        "cohort_digest": {
            "combined": combined_digest(sample_sets),
            "per_concept": {
                concept: samples.cohort().digest
                for concept, samples in sample_sets.items()
            },
        },
        "seeds": {
            "cohort_and_split": int(args.seed),
            "derivation": "sha256_of_the_seed_and_the_fold_or_probe_identity",
        },
        "ec_conditioning": {
            "mode": args.ec_conditioning,
            "fixed_tag": FIXED_EC_TAG if args.ec_conditioning == "fixed" else None,
            "interpretation": EC_CONDITIONING_INTERPRETATION[args.ec_conditioning],
        },
        "split_policy": {
            "estimator": "grouped_out_of_fold_predictions_pooled_once",
            "classification_splitter": "sklearn_StratifiedGroupKFold_shuffled",
            "regression_splitter": "sklearn_GroupKFold_shuffled",
            "n_splits": int(args.n_splits),
            "leakage_check": "every_fold_is_verified_group_disjoint_before_use",
            "grouping_variable_by_concept": {
                concept: CONCEPTS[concept].grouping for concept in sample_sets
            },
            "family_disjoint_by_concept": {
                concept: CONCEPTS[concept].family_disjoint for concept in sample_sets
            },
            "record_level_fallback": "refused; an unusable grouping raises instead",
        },
        "design": {
            "layer_fractions": [float(value) for value in args.layer_fractions],
            "analysis_layers": [int(layer) for layer in layers],
            "erasure_fraction": float(args.erasure_fraction),
            "erasure_layer": int(erasure_layer),
            "probe_input_dimension": int(args.probe_dim),
            "probe_pipeline": "standardise_then_fold_fitted_pca_then_estimator",
            "linear_probe": "logistic_regression_balanced_or_ridge",
            "mlp_probe": {
                "hidden_layer_sizes": list(args.mlp_hidden),
                "max_iter": int(args.mlp_max_iter),
            },
            "chance_model": CHANCE_MODEL,
            "skill_definition": "(score - chance) / (1 - chance)",
            "erasure_method": ERASURE_METHOD,
            "erasure_citation": ERASURE_CITATION,
            "control_matching_criteria": (
                "a control is a matched cost only if it displaces activations "
                "no more than 3x as far as the erasure and costs less than half "
                "the mean-ablation reference"
            ),
            "matched_controls": [
                "variance_matched_random",
                "random_whitened_orthonormal",
                "random_raw_orthonormal",
            ],
            "primary_control": PRIMARY_CONTROL,
            "reference": "mean_ablation_of_the_residual_stream_at_the_erasure_layer",
            "forward_token_budget_by_concept": token_budget(sample_sets),
            "probe_sample_unit": (
                "one token per sample on every arm, so equal sampled positions "
                "are equal probe training tokens rather than equal sequences"
            ),
            "cross_modality_caveat": (
                "text and protein probe targets are analogues matched on "
                "granularity and difficulty, not the same concepts; a difference "
                "in probe skill between arms is not a difference in encoded "
                "structure"
            ),
        },
        "thresholds": {
            "max_post_erasure_linear_skill": float(args.max_post_erasure_linear_skill),
            "min_clean_linear_skill_for_an_informative_gate": float(
                args.min_clean_linear_skill
            ),
            "minimum_ce_denominator_nats": float(args.min_ce_denominator),
            "min_groups_per_label": int(args.n_splits),
            "min_plddt": float(args.min_plddt),
            "burial_contact_bands": list(
                sample_sets["ss3"].construction["burial_contact_bands"]
            )
            if "ss3" in sample_sets
            else None,
            "protein_length_band": [int(args.min_len), int(args.max_len)],
            "fitness_length_band": [int(args.fitness_min_len), int(args.fitness_max_len)],
            "max_tokens": int(args.max_tokens),
            "max_scored_units": int(args.max_scored_units),
            "n_bootstrap": int(args.n_bootstrap),
        },
        "refused_concepts": refusals,
        "concepts": reports,
    }
    write_json(destination, payload)
    print(f"wrote {destination}")

    for concept, report in sorted(reports.items()):
        cohort = report["cohort"]
        print(
            f"\n{concept}: {cohort['n_samples']} samples, {cohort['n_units']} units, "
            f"{cohort['n_groups']} groups of {cohort['grouping_variable']}"
        )
        for layer in report["layer_grid"]["layers"]:
            block = report["decodability"]["per_position"][str(layer)]
            print(f"  layer {layer:2d} linear {format_skill(block['linear'])}")
            print(f"  layer {layer:2d} mlp    {format_skill(block['mlp'])}")
        erasure = report["erasure"]
        gate = erasure["verification"]["gate"]
        print(
            f"  erasure layer {erasure['layer']} rank {erasure['erasers']['leace']['erased_rank']}: "
            f"linear skill {gate['clean_linear_skill']:+.4f} -> "
            f"{gate['observed_post_erasure_linear_skill']:+.4f} "
            f"(gate {'passed' if gate['passed'] else 'FAILED'}, "
            f"informative {gate['informative']})"
        )
        behaviour = erasure["behaviour"]
        for mode, block in sorted(behaviour["modes"].items()):
            eraser = erasure["erasers"][mode]
            print(
                f"    {mode:28s} ce_delta {block['ce_delta_nats']:+.4f} nats "
                f"kl {block['kl_nats']:.4f} nats  removed_var "
                f"{eraser['removed_variance_fraction']:.4f}  displacement "
                f"{eraser['mean_relative_displacement']:.4f}"
            )
        for control, excess in sorted(behaviour["excess_over_control"].items()):
            marker = "*" if control == behaviour["primary_control"] else " "
            matching = erasure["control_matching"][control]
            print(
                f"   {marker}excess over {control:28s} "
                f"{excess['ce_excess_nats']:+.4f} nats "
                f"[{excess['ce_excess_ci95'][0]:+.4f}, {excess['ce_excess_ci95'][1]:+.4f}]"
                f"  displacement_ratio {matching['displacement_ratio']:.2f}"
                f"{'' if matching['cost_is_a_matched_cost'] else '  UNMATCHED'}"
            )


if __name__ == "__main__":
    main()
