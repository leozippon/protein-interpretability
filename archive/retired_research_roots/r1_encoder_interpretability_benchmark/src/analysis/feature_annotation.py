"""SAE feature ↔ annotation alignment pipeline.

Given a trained SAE and annotated Swiss-Prot proteins:
1. Run ESM-2 → SAE inference to get per-residue feature activations
2. For each SAE feature × annotation type, compute precision/recall/F1
3. Classify features as KNOWN (F1>0.5), PARTIAL (0.2-0.5), NOVEL (F1<0.2)

Supports both:
  - InterProt pre-trained 650M SAEs (for development/baseline)
  - Our trained 3B SAEs (for final results)
"""

import os
import pickle
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoTokenizer, EsmModel

from src.data.swissprot_parser import ProteinAnnotation, build_residue_labels


@dataclass
class FeatureAnnotationResult:
    """F1 alignment result for one SAE feature."""
    feature_idx: int
    best_annotation: str       # annotation type with highest F1
    best_f1: float
    best_precision: float
    best_recall: float
    classification: str        # KNOWN, PARTIAL, or NOVEL
    alive: bool                # whether the feature fires at all
    mean_activation: float     # average activation when active
    num_tokens_active: int     # total tokens where this feature fires
    all_scores: dict           # annotation_type → (f1, precision, recall)
    firing_positions: list | None = None  # optional [(accession, pos)]
    top_firing_examples: list | None = None  # optional [{accession,pos,aa,activation}]


def load_interprot_sae(sae_path: str, d_model: int = 1280, d_sae: int = 4096, device: str = "cuda"):
    """Load InterProt pre-trained SAE from safetensors file.

    InterProt SAE uses LayerNorm + b_pre (not b_dec), TopK per-example.
    Weights: w_enc (d_model, d_sae), w_dec (d_sae, d_model), b_enc (d_sae), b_pre (d_model).
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../external/interprot"))
    from interprot.sae_model import SparseAutoencoder

    sae = SparseAutoencoder(d_model, d_sae)
    if sae_path.endswith(".safetensors"):
        from safetensors.torch import load_file as safe_load
        state = safe_load(sae_path)
    else:
        state = torch.load(sae_path, map_location=device, weights_only=True)
    sae.load_state_dict(state, strict=False)
    sae.to(device).eval()
    return sae


def load_our_sae(checkpoint_path: str, device: str = "cuda"):
    """Load our trained BatchTopK SAE from checkpoint."""
    import yaml

    config_path = os.path.join(checkpoint_path, "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    from src.training.sae import BatchTopKSAE

    scfg = config["sae"]
    sae = BatchTopKSAE(
        d_in=scfg["d_in"],
        d_sae=scfg["d_sae"],
        k=scfg["k"],
        topk_mode=scfg.get("topk_mode", "example"),
        normalize_activations=scfg.get("normalize_activations", False),
    )
    state = torch.load(os.path.join(checkpoint_path, "sae.pt"), map_location=device, weights_only=True)
    sae.load_state_dict(state)
    sae.to(device).eval()
    return sae


@torch.no_grad()
def extract_sae_activations(
    sequences: list[str],
    esm_model: EsmModel,
    tokenizer,
    sae,
    target_layer: int,
    device: str = "cuda",
    batch_size: int = 8,
    sae_type: str = "interprot",
) -> list[np.ndarray]:
    """Run ESM-2 → SAE inference on protein sequences.

    Args:
        sequences: List of amino acid strings
        esm_model: Loaded ESM-2 model (frozen)
        tokenizer: ESM-2 tokenizer
        sae: Loaded SAE model
        target_layer: ESM-2 layer to extract (0-indexed)
        sae_type: "interprot" or "ours" (different forward interface)

    Returns:
        List of (seq_len, d_sae) numpy arrays, one per sequence
    """
    all_activations = []

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i : i + batch_size]
        spaced = [" ".join(list(seq)) for seq in batch_seqs]

        encoded = tokenizer(
            spaced,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        ).to(device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = esm_model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                output_hidden_states=True,
            )

        # hidden_states[0] = embeddings, [i+1] = layer i output
        layer_acts = outputs.hidden_states[target_layer + 1].float()

        for j, seq in enumerate(batch_seqs):
            seq_len = len(seq)
            # Extract AA tokens only (positions 1..seq_len, skip CLS at 0)
            aa_acts = layer_acts[j, 1 : seq_len + 1, :]  # (seq_len, d_model)

            if sae_type == "interprot":
                # InterProt get_acts: LN → subtract b_pre → encode → topK
                sae_feats = sae.get_acts(aa_acts)  # (seq_len, d_sae)
            else:
                # Our BatchTopKSAE: forward returns (x_hat, f, ...)
                # Use forward to get sparse activations with topk applied
                output = sae(aa_acts)
                sae_feats = output.f  # sparse feature activations

            all_activations.append(sae_feats.cpu().numpy())

    return all_activations


def compute_feature_annotation_f1(
    annotations: list[ProteinAnnotation],
    sae_activations: list[np.ndarray],
    d_sae: int,
    threshold_percentile: float = 95.0,
    min_positive_residues: int = 50,
    save_firing_positions: bool = False,
    max_firing_positions_per_feature: int = 200,
) -> list[FeatureAnnotationResult]:
    """Compute F1 between each SAE feature and each annotation type.

    For each SAE feature:
      1. Binarize: feature is "active" when activation > threshold
         (threshold = 95th percentile of nonzero activations across all proteins)
      2. For each annotation type, compute precision/recall/F1
      3. Best F1 determines classification: KNOWN/PARTIAL/NOVEL

    Args:
        annotations: Swiss-Prot protein annotations
        sae_activations: Per-protein SAE activation arrays
        d_sae: SAE dictionary size
        threshold_percentile: Percentile for binarization threshold
        min_positive_residues: Minimum annotation occurrences to consider
        save_firing_positions: Store top thresholded residue positions per feature
        max_firing_positions_per_feature: Cap stored positions/examples per feature

    Returns:
        List of FeatureAnnotationResult, one per SAE feature
    """
    assert len(annotations) == len(sae_activations)

    # Step 1: Collect all annotation types and their residue-level labels
    # Two-pass approach: first collect per-protein labels, then concatenate
    print("Building annotation label matrices...")
    per_protein_labels = []  # list of (seq_len, labels_dict) tuples
    all_sae_acts = []
    residue_meta = [] if save_firing_positions else None

    for ann, acts in zip(annotations, sae_activations):
        labels = build_residue_labels(ann)
        seq_len = len(ann.sequence)
        assert acts.shape[0] == seq_len, f"Mismatch: acts {acts.shape[0]} vs seq {seq_len}"
        per_protein_labels.append((seq_len, labels))
        all_sae_acts.append(acts)
        if residue_meta is not None:
            residue_meta.extend(
                (ann.accession, pos + 1, ann.sequence[pos])
                for pos in range(seq_len)
            )

    # Collect all annotation type names
    all_label_names = set()
    for _, labels in per_protein_labels:
        all_label_names.update(labels.keys())

    # Build concatenated label arrays, padding missing annotations with False
    all_labels = {}
    for name in all_label_names:
        arrays = []
        for seq_len, labels in per_protein_labels:
            if name in labels:
                arrays.append(np.array(labels[name], dtype=bool))
            else:
                arrays.append(np.zeros(seq_len, dtype=bool))
        all_labels[name] = np.concatenate(arrays)

    # Filter annotation types with sufficient positive residues
    valid_annotations = {}
    for name, label_arr in all_labels.items():
        total_positive = int(label_arr.sum())
        if total_positive >= min_positive_residues:
            valid_annotations[name] = label_arr

    # Concatenate all SAE activations
    all_acts_concat = np.concatenate(all_sae_acts, axis=0)  # (total_residues, d_sae)
    total_residues = all_acts_concat.shape[0]

    print(f"Total residues: {total_residues:,}")
    print(f"Valid annotation types: {len(valid_annotations)}")

    # Step 2: Build annotation matrix
    ann_names = sorted(valid_annotations.keys())
    n_ann = len(ann_names)
    # (n_annotations, total_residues) float32 matrix
    ann_matrix = np.stack(
        [valid_annotations[name].astype(np.float32) for name in ann_names], axis=0
    )
    ann_positive_counts = ann_matrix.sum(axis=1)  # (n_annotations,)

    # Step 3: Compute per-feature thresholds and binarize ALL features at once
    print("Computing per-feature thresholds...")
    nonzero_counts = (all_acts_concat > 0).sum(axis=0).astype(np.int64)  # (d_sae,)
    alive_mask = nonzero_counts > 0  # (d_sae,)

    # Vectorized mean of nonzero activations
    mean_acts = np.where(
        alive_mask,
        all_acts_concat.sum(axis=0) / np.maximum(nonzero_counts, 1),
        0.0,
    ).astype(np.float32)

    # Compute per-feature thresholds in chunks to avoid sorting full matrix
    # For each feature, threshold = 95th percentile of nonzero activations
    print("  Computing thresholds (chunked)...")
    thresholds = np.zeros(d_sae, dtype=np.float32)
    chunk_size = 1024
    for start in range(0, d_sae, chunk_size):
        end = min(start + chunk_size, d_sae)
        chunk = all_acts_concat[:, start:end]
        for j in range(end - start):
            fi = start + j
            if not alive_mask[fi]:
                continue
            nonzero_vals = chunk[:, j][chunk[:, j] > 0]
            if len(nonzero_vals) > 0:
                thresholds[fi] = np.percentile(nonzero_vals, threshold_percentile)

    # Binarize all features at once: (total_residues, d_sae)
    feat_binary_matrix = (all_acts_concat > thresholds[np.newaxis, :]).astype(np.float32)
    num_active_per_feat = feat_binary_matrix.sum(axis=0)  # (d_sae,)

    # Step 4: Single matmul for ALL TPs: (n_ann, total_residues) @ (total_residues, d_sae) → (n_ann, d_sae)
    print(f"Computing F1 matrix: ({n_ann} annotations × {d_sae} features)...")
    tp_matrix = ann_matrix @ feat_binary_matrix  # (n_ann, d_sae)

    # Vectorized precision/recall/F1 for all features at once
    fp_matrix = num_active_per_feat[np.newaxis, :] - tp_matrix  # (n_ann, d_sae)
    fn_matrix = ann_positive_counts[:, np.newaxis] - tp_matrix   # (n_ann, d_sae)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision_matrix = np.where(tp_matrix + fp_matrix > 0,
                                    tp_matrix / (tp_matrix + fp_matrix), 0.0)
        recall_matrix = np.where(tp_matrix + fn_matrix > 0,
                                 tp_matrix / (tp_matrix + fn_matrix), 0.0)
        f1_matrix = np.where(precision_matrix + recall_matrix > 0,
                             2 * precision_matrix * recall_matrix / (precision_matrix + recall_matrix),
                             0.0)

    # Best annotation per feature
    best_indices = np.argmax(f1_matrix, axis=0)  # (d_sae,)
    best_f1s = f1_matrix[best_indices, np.arange(d_sae)]
    best_precs = precision_matrix[best_indices, np.arange(d_sae)]
    best_recs = recall_matrix[best_indices, np.arange(d_sae)]

    # Build results
    print("Building result objects...")
    results = []
    for feat_idx in range(d_sae):
        firing_positions = None
        top_firing_examples = None
        if save_firing_positions and alive_mask[feat_idx]:
            active_rows = np.where(feat_binary_matrix[:, feat_idx] > 0)[0]
            if active_rows.size > 0:
                acts = all_acts_concat[active_rows, feat_idx]
                order = np.argsort(-acts)[:max_firing_positions_per_feature]
                picked_rows = active_rows[order]
                picked_acts = acts[order]
                firing_positions = [
                    (residue_meta[int(row)][0], int(residue_meta[int(row)][1]))
                    for row in picked_rows
                ]
                top_firing_examples = [
                    {
                        "accession": residue_meta[int(row)][0],
                        "position": int(residue_meta[int(row)][1]),
                        "aa": residue_meta[int(row)][2],
                        "activation": float(act),
                    }
                    for row, act in zip(picked_rows, picked_acts)
                ]

        if not alive_mask[feat_idx]:
            results.append(FeatureAnnotationResult(
                feature_idx=feat_idx,
                best_annotation="",
                best_f1=0.0,
                best_precision=0.0,
                best_recall=0.0,
                classification="DEAD",
                alive=False,
                mean_activation=0.0,
                num_tokens_active=0,
                all_scores={},
                firing_positions=firing_positions,
                top_firing_examples=top_firing_examples,
            ))
            continue

        best_f1 = float(best_f1s[feat_idx])
        best_ann = ann_names[int(best_indices[feat_idx])]

        # Store non-trivial scores only
        f1_col = f1_matrix[:, feat_idx]
        nontrivial = f1_col > 0.1
        all_scores = {}
        if nontrivial.any():
            for i in np.where(nontrivial)[0]:
                all_scores[ann_names[i]] = (
                    float(f1_col[i]),
                    float(precision_matrix[i, feat_idx]),
                    float(recall_matrix[i, feat_idx]),
                )

        if best_f1 > 0.5:
            classification = "KNOWN"
        elif best_f1 > 0.2:
            classification = "PARTIAL"
        else:
            classification = "NOVEL"

        results.append(FeatureAnnotationResult(
            feature_idx=feat_idx,
            best_annotation=best_ann,
            best_f1=best_f1,
            best_precision=float(best_precs[feat_idx]),
            best_recall=float(best_recs[feat_idx]),
            classification=classification,
            alive=True,
            mean_activation=float(mean_acts[feat_idx]),
            num_tokens_active=int(num_active_per_feat[feat_idx]),
            all_scores=all_scores,
            firing_positions=firing_positions,
            top_firing_examples=top_firing_examples,
        ))

    return results


def summarize_results(results: list[FeatureAnnotationResult]) -> dict:
    """Summarize feature classification results."""
    from collections import Counter

    class_counts = Counter(r.classification for r in results)
    alive = [r for r in results if r.alive]
    alive_classified = Counter(r.classification for r in alive)

    best_known = sorted(
        [r for r in results if r.classification == "KNOWN"],
        key=lambda r: -r.best_f1,
    )[:20]

    return {
        "total_features": len(results),
        "alive_features": len(alive),
        "dead_features": class_counts.get("DEAD", 0),
        "classification": dict(alive_classified),
        "classification_pct": {
            k: v / max(len(alive), 1) * 100 for k, v in alive_classified.items()
        },
        "mean_best_f1_alive": np.mean([r.best_f1 for r in alive]) if alive else 0.0,
        "top_known_features": [
            (r.feature_idx, r.best_annotation, f"{r.best_f1:.3f}")
            for r in best_known
        ],
    }
