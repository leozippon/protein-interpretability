#!/usr/bin/env python3
"""Anchored partner identification from per-position states versus attention.

For each AlphaFold protein an anchor residue with a true long-range contact
partner is scored against decoys from the same sequence-separation band. Six
dimensionality-matched predictor arms are compared, each with a linear probe and
a small MLP, on a homology-disjoint split. If attention beats every per-position
arm under both estimators and on a non-leaky split, then the pathway a
frozen-attention attribution graph discards is where the relational computation
lives.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# See 08_lens_family.py for why the stage directory is added explicitly.
_STAGE_DIR = str(Path(__file__).resolve().parent)
if _STAGE_DIR not in sys.path:
    sys.path.insert(0, _STAGE_DIR)

from panel_contract import arm_can_run, stage_contract_record  # noqa: E402
from src.transfer.arms import (  # noqa: E402
    PANEL,
    REPO,
    ZYMCTRL_FASTA,
    Cohort,
    iter_fasta,
    load_arm,
)
from src.transfer.channels import (  # noqa: E402
    ALPHAFOLD_ROOT,
    PFAM_RESIDUE_TSV,
    alphafold_models,
    load_pfam_spans,
    read_alphafold_model,
)
from src.transfer.io import write_json  # noqa: E402
from src.transfer.relational import (  # noqa: E402
    CONTACT_ANGSTROM,
    MIN_SEPARATION,
    PREDICTOR_ARMS,
    anchored_pairs,
    build_feature_arms,
    contact_map,
    encode,
    evaluate_predictor,
    fit_position_projection,
    homology_clusters,
    homology_disjoint_split,
    random_protein_split,
    require_residue_token_map,
)
from src.transfer.scoring import analysis_layers  # noqa: E402

SCHEMA_VERSION = "r2_transfer_relational_channel_v1"
DEFAULT_OUT = REPO / "results/transfer/relational_channel"
LAYER_FRACTIONS = (0.33, 0.5, 0.67)


def ec_labels_by_accession() -> dict[str, str]:
    """EC conditioning tags keyed by UniProt accession.

    ZymCTRL is EC-conditioned; feeding it a bare sequence puts it
    off-distribution, which is exactly the failure the cohort-power measurement
    exists to catch. A structure with no EC label is therefore excluded rather
    than scored on an unconditioned prompt.
    """

    labels: dict[str, str] = {}
    for header, _ in iter_fasta(ZYMCTRL_FASTA):
        if "|" not in header:
            raise ValueError(f"{ZYMCTRL_FASTA}: header {header!r} has no accession|EC form")
        accession, ec = header.split("|", 1)
        labels[accession] = ec
    if not labels:
        raise RuntimeError(f"{ZYMCTRL_FASTA}: no EC-labelled records")
    return labels


def collect(args: argparse.Namespace, arm) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    """Encode anchored pair samples protein by protein, counting every exclusion."""

    layers = analysis_layers(arm.n_layer, LAYER_FRACTIONS)
    generator = np.random.default_rng(args.seed)
    conditioned = arm.spec.input_format == "ec_conditioned"
    ec_labels = ec_labels_by_accession() if conditioned else {}
    records: list[dict[str, Any]] = []
    excluded = {
        "length_out_of_range": 0,
        "low_plddt": 0,
        "non_canonical_residues": 0,
        "too_few_anchor_groups": 0,
        "no_ec_label": 0,
    }
    # Filename order is UniProt-accession order, which front-loads whole
    # proteome dumps of closely related entries; a seeded permutation of the
    # full model set is both reproducible and taxonomically diverse.
    catalogue = alphafold_models(ALPHAFOLD_ROOT)
    order = np.random.default_rng(args.seed).permutation(len(catalogue))[: args.scan_models]
    examined = 0
    for index in order:
        path = catalogue[int(index)]
        examined += 1
        structure = read_alphafold_model(path)
        if structure.n_non_canonical_residues > 0:
            excluded["non_canonical_residues"] += 1
            continue
        if not args.min_len <= len(structure) <= args.max_len:
            excluded["length_out_of_range"] += 1
            continue
        if float(structure.plddt.mean()) < args.min_plddt:
            excluded["low_plddt"] += 1
            continue
        if conditioned and structure.accession not in ec_labels:
            excluded["no_ec_label"] += 1
            continue
        pairs = anchored_pairs(
            contact_map(structure.ca, args.contact_angstrom),
            generator,
            min_separation=args.min_separation,
            n_decoys=args.n_decoys,
            max_groups=args.groups_per_protein,
        )
        if pairs.n_groups < args.min_groups_per_protein:
            excluded["too_few_anchor_groups"] += 1
            continue
        metadata = {"ec_labels": [ec_labels[structure.accession]]} if conditioned else {}
        cohort = Cohort(
            "relational",
            "protein",
            [structure.sequence],
            args.min_len,
            args.max_len,
            metadata,
        )
        input_string = cohort.input_strings(arm)[0]
        hidden, attention = encode(arm, input_string, structure.sequence, pairs, layers=layers)
        records.append(
            {
                "accession": structure.accession,
                "sequence": structure.sequence,
                "hidden": hidden,
                "attention": attention,
                "pairs": pairs,
            }
        )
        if len(records) >= args.n_proteins:
            break
    if len(records) < args.min_proteins:
        raise RuntimeError(
            f"{arm.name}: only {len(records)} usable structures out of "
            f"{examined} examined; need {args.min_proteins}. Exclusions: {excluded}"
        )
    return records, excluded, examined


def assemble(
    records: list[dict[str, Any]], train_mask: np.ndarray, *, pca_dim: int, seed: int
) -> dict[str, Any]:
    """Fit the projection on the training proteins and score every predictor arm."""

    train = [record for record, flag in zip(records, train_mask) if flag]
    test = [record for record, flag in zip(records, train_mask) if not flag]
    if not train or not test:
        raise ValueError("both split sides must contain proteins")

    projection = fit_position_projection(
        [record["hidden"] for record in train], n_components=pca_dim, seed=seed
    )

    def side(subset: list[dict[str, Any]], group_stride: int) -> dict[str, np.ndarray]:
        anchors = np.concatenate(
            [projection.transform(r["hidden"][r["pairs"].anchor]) for r in subset]
        )
        partners = np.concatenate(
            [projection.transform(r["hidden"][r["pairs"].partner]) for r in subset]
        )
        attention = np.concatenate([r["attention"] for r in subset])
        separation = np.concatenate([r["pairs"].separation for r in subset])
        labels = np.concatenate([r["pairs"].label for r in subset])
        groups = np.concatenate(
            [
                r["pairs"].group + group_stride * (index + 1)
                for index, r in enumerate(subset)
            ]
        )
        # The protein each pair came from. Anchors are drawn *within* a protein,
        # so an interval over anchors treats up to --groups-per-protein correlated
        # draws as independent; the sampling unit is the protein, and this is what
        # lets within_anchor_auc report the interval that unit supports.
        proteins = np.concatenate(
            [
                np.full(r["pairs"].anchor.size, index, dtype=np.int64)
                for index, r in enumerate(subset)
            ]
        )
        return {
            "features": build_feature_arms(anchors, partners, attention, separation),
            "label": labels,
            "group": groups,
            "protein": proteins,
        }

    train_side = side(train, 10**6)
    test_side = side(test, 10**6)
    scored = {
        name: evaluate_predictor(
            train_side["features"][name],
            train_side["label"],
            test_side["features"][name],
            test_side["label"],
            test_side["group"],
            seed=seed,
            test_proteins=test_side["protein"],
        )
        for name in PREDICTOR_ARMS
    }
    return {
        "n_train_proteins": len(train),
        "n_test_proteins": len(test),
        "n_train_pairs": int(train_side["label"].size),
        "n_test_pairs": int(test_side["label"].size),
        "test_positive_rate": float(test_side["label"].mean()),
        "pca_dim_per_position": int(pca_dim),
        "anchored_partner_identification": scored,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="progen2-medium")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--n-proteins", type=int, default=140)
    parser.add_argument("--min-proteins", type=int, default=60)
    parser.add_argument("--scan-models", type=int, default=4000)
    parser.add_argument("--min-len", type=int, default=110)
    parser.add_argument("--max-len", type=int, default=320)
    parser.add_argument("--min-plddt", type=float, default=75.0)
    parser.add_argument("--contact-angstrom", type=float, default=CONTACT_ANGSTROM)
    parser.add_argument("--min-separation", type=int, default=MIN_SEPARATION)
    parser.add_argument("--n-decoys", type=int, default=8)
    parser.add_argument("--groups-per-protein", type=int, default=24)
    parser.add_argument("--min-groups-per-protein", type=int, default=4)
    parser.add_argument("--pca-dim", type=int, default=0, help="0 matches the attention width")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--kmer", type=int, default=3)
    parser.add_argument("--kmer-jaccard", type=float, default=0.10)
    parser.add_argument("--pfam-jaccard", type=float, default=0.50)
    parser.add_argument("--min-test-proteins", type=int, default=10)
    parser.add_argument(
        "--random-split-contrast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also score a leaky random protein split, to size the homology leak",
    )
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.arm not in PANEL:
        raise ValueError(f"unknown arm {args.arm!r}; panel is {sorted(PANEL)}")
    # The tokenisation and capability requirements are properties of ArmSpec, so
    # they are checked before the checkpoint is loaded. require_residue_token_map
    # below needs the loaded tokenizer and stays where it is; this refuses the
    # arms it would refuse anyway, plus the ones with no `relational` capability,
    # without paying for a model load first.
    verdict = arm_can_run("relational_channel", args.arm)
    if not verdict.can_run:
        raise ValueError(f"05_relational_channel.py cannot measure {args.arm}: {verdict.reason}")
    # homology_disjoint_split raises when --train-fraction and --min-test-proteins
    # cannot both be satisfied -- after every protein has been encoded through the
    # model. The necessary condition is arithmetic on the command line.
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must lie strictly between zero and one")
    held_out = args.n_proteins - int(args.train_fraction * args.n_proteins)
    if args.min_test_proteins > held_out:
        raise ValueError(
            f"--min-test-proteins {args.min_test_proteins} cannot be satisfied: "
            f"--n-proteins {args.n_proteins} at --train-fraction {args.train_fraction} "
            f"leaves at most {held_out} test proteins, and homology-disjoint "
            "clustering can only reduce that further"
        )
    arm = load_arm(
        args.arm, device=args.device, dtype=args.dtype, attn_implementation="eager"
    )
    require_residue_token_map(arm)

    records, excluded, examined = collect(args, arm)
    accessions = [record["accession"] for record in records]
    sequences = [record["sequence"] for record in records]
    spans = load_pfam_spans(PFAM_RESIDUE_TSV, accessions=set(accessions))
    pfam_by_accession = {
        accession: {family for _, _, family in entries} for accession, entries in spans.items()
    }
    clusters, homology_summary = homology_clusters(
        accessions,
        sequences,
        pfam_by_accession=pfam_by_accession,
        kmer=args.kmer,
        kmer_jaccard_threshold=args.kmer_jaccard,
        pfam_jaccard_threshold=args.pfam_jaccard,
    )

    attention_width = records[0]["attention"].shape[1]
    pca_dim = args.pca_dim or max(8, attention_width // 2)
    homology_mask = homology_disjoint_split(
        clusters,
        train_fraction=args.train_fraction,
        seed=args.seed,
        min_side=args.min_test_proteins,
    )
    splits = {
        "homology_disjoint": assemble(records, homology_mask, pca_dim=pca_dim, seed=args.seed)
    }
    if args.random_split_contrast:
        splits["random_protein"] = assemble(
            records,
            random_protein_split(
                len(records), train_fraction=args.train_fraction, seed=args.seed
            ),
            pca_dim=pca_dim,
            seed=args.seed,
        )

    cohort = Cohort("alphafold_relational", "protein", sequences, args.min_len, args.max_len, {})
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "relational_channel_report",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_digest": cohort.digest,
        "arm_spec": {
            "arm": arm.name,
            "path": str(arm.spec.path),
            "n_layer": arm.n_layer,
            "d_model": arm.d_model,
            "tokenisation": arm.spec.tokenisation,
            "input_format": arm.spec.input_format,
            "attn_implementation": "eager",
            "dtype": args.dtype,
        },
        "seeds": {"sampling_and_split": int(args.seed)},
        "thresholds": {
            "contact_angstrom": float(args.contact_angstrom),
            "min_separation": int(args.min_separation),
            "min_plddt": float(args.min_plddt),
            "min_groups_per_protein": int(args.min_groups_per_protein),
            "train_fraction": float(args.train_fraction),
        },
        "design": {
            "analysis_layers": analysis_layers(arm.n_layer, LAYER_FRACTIONS),
            "layer_fractions": list(LAYER_FRACTIONS),
            "n_decoys_per_anchor": int(args.n_decoys),
            "groups_per_protein_cap": int(args.groups_per_protein),
            "hidden_width_before_pca": int(records[0]["hidden"].shape[1]),
            "attention_features": int(attention_width),
            "pca_dim_per_position": int(pca_dim),
            "predictor_arms": list(PREDICTOR_ARMS),
        },
        "cohort": {
            "n_proteins": len(records),
            "n_models_examined": examined,
            "model_scan_budget": int(args.scan_models),
            "model_selection": "seeded_permutation_of_the_full_alphafold_model_set",
            "min_len": int(args.min_len),
            "max_len": int(args.max_len),
            "excluded": excluded,
        },
        "homology": homology_summary,
        "splits": splits,
        # A per-arm stage narrows by running fewer processes, not by shrinking a
        # list, so its artefacts are identical whether the campaign covered the
        # eligible panel or one arm of it. The record is what makes the two
        # distinguishable after the fact.
        "stage_contract": stage_contract_record("relational_channel", [arm.name]),
    }
    destination = args.out / f"{arm.name}.json"
    write_json(destination, payload)
    print(f"wrote {destination}")
    for split_name, block in splits.items():
        print(f"{split_name}: {block['n_train_proteins']}/{block['n_test_proteins']} proteins")
        for name, result in sorted(
            block["anchored_partner_identification"].items(),
            key=lambda item: -item[1]["linear"]["auc"],
        ):
            print(
                f"  {name:22s} d={result['n_features']:4d}  "
                f"linear {result['linear']['auc']:.4f}+/-{result['linear']['sem']:.4f}  "
                f"mlp {result['mlp']['auc']:.4f}+/-{result['mlp']['sem']:.4f}"
            )


if __name__ == "__main__":
    main()
