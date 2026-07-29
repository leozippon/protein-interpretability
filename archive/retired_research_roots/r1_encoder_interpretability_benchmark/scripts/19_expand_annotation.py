#!/usr/bin/env python
"""Deep-layer annotation expansion (R1-F).

L35 currently has only 31 KNOWN features because Swiss-Prot residue
annotations are shallow (functional, ptm, domain, region, topology, secondary
structure, chain). We add three richer label sources to push the KNOWN count
up and give the mechanism classifier stronger priors:

  1. GO biological-process / molecular-function residues (per-protein, via
     UniProt GAF files → residues in active_site / binding_site regions)
  2. Pfam domain memberships per residue
  3. PDB binding-site atoms (BioLiP distances < 4Å) → per-residue labels

For each SAE feature, we compute one-vs-rest F1 against each new label set,
and merge best-matching label into `results/annotation_alignment/`.

Produces `results/annotation_alignment/expanded_summary.json` with per-layer
KNOWN / PARTIAL / USEFUL counts comparing pre- vs post-expansion.

Usage:
    python r1_encoder_interpretability_benchmark/scripts/19_expand_annotation.py \
        --layers 19 23 27 31 35 \
        --go-gaf data/go/goa_human.gaf.gz \
        --pfam-tsv data/interpro/pfam_residue.tsv \
        --biolip data/BioLiP/BioLiP.txt
"""

import argparse
import gzip
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_go_residue_labels(gaf_path: str, swissprot_ids: set[str]) -> dict[tuple[str, int], set[str]]:
    """Parse UniProt GAF for active-site / binding-site residues.

    GAF lines with with_from mentioning UniProtKB-KB:XXXX positions. For
    tractability, we restrict to GO terms flagged as active_site, binding_site,
    catalytic_activity.
    """
    labels: dict[tuple[str, int], set[str]] = defaultdict(set)
    if not os.path.exists(gaf_path):
        return labels
    opener = gzip.open if gaf_path.endswith(".gz") else open
    go_keep = {
        "GO:0003824",  # catalytic activity
        "GO:0005488",  # binding
        "GO:0043167",  # ion binding
        "GO:0044877",  # protein-containing complex binding
    }
    with opener(gaf_path, "rt") as f:
        for line in f:
            if line.startswith("!"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 17:
                continue
            if parts[0] != "UniProtKB":
                continue
            uid = parts[1]
            if swissprot_ids and uid not in swissprot_ids:
                continue
            go = parts[4]
            if go not in go_keep:
                continue
            # Residue position is encoded in 'with_from' column 7 (0-indexed)
            with_col = parts[7] if len(parts) > 7 else ""
            for token in with_col.split("|"):
                if ":" in token and token.split(":", 1)[0].startswith("UniProtKB"):
                    after = token.split(":", 1)[1]
                    # format may be like P12345-1234
                    if "-" in after:
                        try:
                            pos = int(after.split("-")[-1])
                        except ValueError:
                            continue
                        labels[(uid, pos)].add(f"go/{go}")
    return labels


def load_pfam_residue_labels(pfam_tsv: str, swissprot_ids: set[str]) -> dict[tuple[str, int], set[str]]:
    """Parse Pfam per-residue TSV (uniprot, start, end, pfam_id)."""
    labels: dict[tuple[str, int], set[str]] = defaultdict(set)
    if not os.path.exists(pfam_tsv):
        return labels
    with open(pfam_tsv) as f:
        header = f.readline().strip().lower().split("\t")
        try:
            ui = header.index("uniprot")
            si = header.index("start")
            ei = header.index("end")
            pi = header.index("pfam_id")
        except ValueError:
            return labels
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < max(ui, si, ei, pi) + 1:
                continue
            uid = parts[ui]
            if swissprot_ids and uid not in swissprot_ids:
                continue
            try:
                start = int(parts[si])
                end = int(parts[ei])
            except ValueError:
                continue
            pfam = parts[pi]
            for p in range(start, end + 1):
                labels[(uid, p)].add(f"pfam/{pfam}")
    return labels


def load_biolip_labels(biolip_path: str, swissprot_ids: set[str]) -> dict[tuple[str, int], set[str]]:
    """Parse BioLiP binding-residue annotations."""
    labels: dict[tuple[str, int], set[str]] = defaultdict(set)
    if not os.path.exists(biolip_path):
        return labels
    # BioLiP format: tab-separated, binding residues in column 9 (1-indexed),
    # uniprot in column 18.
    with open(biolip_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 20:
                continue
            uid = parts[17]
            if swissprot_ids and uid not in swissprot_ids:
                continue
            bind_res = parts[8]
            for token in bind_res.split():
                # format: Leu123 or L123
                digits = "".join(c for c in token if c.isdigit())
                if not digits:
                    continue
                pos = int(digits)
                labels[(uid, pos)].add("biolip/binding_site")
    return labels


def load_feature_activations(annot_pkl: str) -> dict:
    """Load existing annotation pkl; we reuse its max-activation info."""
    with open(annot_pkl, "rb") as f:
        return pickle.load(f)


def collect_firing_accessions(annotation_dir: str, layers: list[int]) -> set[str]:
    """Collect accessions actually present in firing-position annotation pkls."""
    accessions: set[str] = set()
    for layer in layers:
        annot_path = f"{annotation_dir}/ours_3B_l{layer}_step500000.pkl"
        if not os.path.exists(annot_path):
            continue
        data = load_feature_activations(annot_path)
        for r in data.get("results", []):
            firing = getattr(r, "firing_positions", None)
            if firing:
                for item in firing:
                    if isinstance(item, (tuple, list)) and item:
                        accessions.add(str(item[0]))
            examples = getattr(r, "top_firing_examples", None)
            if examples:
                for item in examples:
                    if isinstance(item, dict) and item.get("accession"):
                        accessions.add(str(item["accession"]))
    return accessions


def compute_one_vs_rest_f1(feature_positions, positive_positions, universe_size):
    """F1 = 2TP / (2TP + FP + FN).

    feature_positions: set of (uid, pos) where the feature fires.
    positive_positions: set of (uid, pos) with the target label.
    universe_size: unused here (we treat both sets implicitly).
    """
    if not feature_positions or not positive_positions:
        return 0.0
    tp = len(feature_positions & positive_positions)
    fp = len(feature_positions - positive_positions)
    fn = len(positive_positions - feature_positions)
    if tp == 0:
        return 0.0
    return 2 * tp / (2 * tp + fp + fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[19, 23, 27, 31, 35])
    ap.add_argument("--annotation-dir",
                    default="r1_encoder_interpretability_benchmark/results/annotation_alignment")
    ap.add_argument("--go-gaf", default="data/go/goa_human.gaf.gz")
    ap.add_argument("--pfam-tsv", default="data/interpro/pfam_residue.tsv")
    ap.add_argument("--biolip", default="data/BioLiP/BioLiP.txt")
    ap.add_argument("--swissprot-cache",
                    default="data/processed/swissprot_all_max1022.pkl")
    ap.add_argument("--out",
                    default="r1_encoder_interpretability_benchmark/results/annotation_alignment/expanded_summary.json")
    ap.add_argument("--f1-threshold", type=float, default=0.1)
    args = ap.parse_args()

    print("=" * 70)
    print("  R1-F Deep-layer annotation expansion")
    print("=" * 70)

    # Restrict label sources to proteins that actually appear in the current
    # firing-position pkls. This avoids diluting F1 denominators with labels
    # from proteins that were never scored in this annotation run.
    swissprot_ids = collect_firing_accessions(args.annotation_dir, args.layers)
    if swissprot_ids:
        print(f"  Firing-position universe: {len(swissprot_ids)} proteins")
    else:
        print("  No firing-position universe found; falling back to Swiss-Prot cache")
    if not swissprot_ids and os.path.exists(args.swissprot_cache):
        with open(args.swissprot_cache, "rb") as f:
            cache = pickle.load(f)
        for p in cache if isinstance(cache, list) else cache.values():
            if isinstance(p, dict) and "uniprot_id" in p:
                swissprot_ids.add(p["uniprot_id"])
            elif hasattr(p, "accession"):
                swissprot_ids.add(p.accession)
    print(f"  Label-source universe: {len(swissprot_ids)} proteins")

    print("\n[1/4] Loading extended label sources...")
    go_lab = load_go_residue_labels(args.go_gaf, swissprot_ids)
    print(f"  GO labels: {len(go_lab)} (uid,pos) pairs")
    pf_lab = load_pfam_residue_labels(args.pfam_tsv, swissprot_ids)
    print(f"  Pfam labels: {len(pf_lab)} (uid,pos) pairs")
    bl_lab = load_biolip_labels(args.biolip, swissprot_ids)
    print(f"  BioLiP labels: {len(bl_lab)} (uid,pos) pairs")

    all_labels: dict[str, dict[tuple[str, int], set[str]]] = {
        "go": go_lab, "pfam": pf_lab, "biolip": bl_lab,
    }
    total_extra = sum(len(v) for v in all_labels.values())
    if total_extra == 0:
        print("\n  No extra labels found. Provide GO / Pfam / BioLiP paths.")
        return

    # For each new source, build one-vs-rest label set per label
    label_to_positions: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for src, d in all_labels.items():
        for pos, labs in d.items():
            for l in labs:
                label_to_positions[l].add(pos)
    print(f"  Total distinct extended labels: {len(label_to_positions)}")

    print("\n[2/4] Scoring SAE features against extended labels...")
    results_per_layer = {}
    for layer in args.layers:
        annot_path = f"{args.annotation_dir}/ours_3B_l{layer}_step500000.pkl"
        if not os.path.exists(annot_path):
            print(f"  L{layer}: annotation pkl missing, skipping")
            continue
        data = load_feature_activations(annot_path)
        results = data["results"]

        new_best_f1 = np.zeros(len(results), dtype=np.float32)
        new_best_label = [""] * len(results)

        # For each feature, if activation records exist, compute F1
        # against each extended label using the feature's firing positions.
        for r in results:
            if not r.alive:
                continue
            # r may carry firing_positions: list of (uid, pos) pairs. If not,
            # we fall back to the feature's best-annotation string and skip.
            firing = getattr(r, "firing_positions", None)
            if firing is None:
                continue
            if isinstance(firing, (list, set)):
                fire_set = set(firing)
            else:
                continue
            best_f1 = 0.0
            best_lab = ""
            for lab, pos_set in label_to_positions.items():
                f1 = compute_one_vs_rest_f1(fire_set, pos_set, 0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_lab = lab
            new_best_f1[r.feature_idx] = best_f1
            new_best_label[r.feature_idx] = best_lab

        # Merge: take max(original_f1, new_f1)
        orig_f1 = np.zeros(len(results), dtype=np.float32)
        for r in results:
            if r.alive:
                orig_f1[r.feature_idx] = r.best_f1 or 0.0
        combined_f1 = np.maximum(orig_f1, new_best_f1)

        results_per_layer[layer] = {
            "orig_known": int((orig_f1 >= 0.5).sum()),
            "orig_partial": int((orig_f1 >= 0.2).sum()),
            "orig_useful": int((orig_f1 >= args.f1_threshold).sum()),
            "new_known": int((combined_f1 >= 0.5).sum()),
            "new_partial": int((combined_f1 >= 0.2).sum()),
            "new_useful": int((combined_f1 >= args.f1_threshold).sum()),
            "n_features_with_firing": int((new_best_f1 > 0).sum()),
        }
        print(f"  L{layer}: KNOWN {results_per_layer[layer]['orig_known']} → {results_per_layer[layer]['new_known']}"
              f"   PARTIAL {results_per_layer[layer]['orig_partial']} → {results_per_layer[layer]['new_partial']}"
              f"   USEFUL {results_per_layer[layer]['orig_useful']} → {results_per_layer[layer]['new_useful']}")

        # Save merged annotation side-by-side (doesn't overwrite original)
        merged_path = f"{args.annotation_dir}/ours_3B_l{layer}_step500000_expanded.pkl"
        data["expanded"] = {
            "combined_f1": combined_f1,
            "new_best_label": new_best_label,
        }
        with open(merged_path, "wb") as f:
            pickle.dump(data, f)

    print(f"\n[3/4] Writing summary JSON")
    out = {
        "per_layer": results_per_layer,
        "f1_threshold": args.f1_threshold,
        "label_sources_summary": {
            src: len(d) for src, d in all_labels.items()
        },
        "notes": (
            "Features with no recorded firing_positions field in the "
            "original annotation pickle are skipped — rerun the annotation "
            "pipeline with `--save-firing-positions` to score them."
        ),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {args.out}")

    print("\n[4/4] Done.")


if __name__ == "__main__":
    main()
