"""Variant effect prediction using SAE perturbation signatures.

Core R1 scientific contribution: classify missense variant mechanisms
(LOF/GOF/DN/neomorphic) by comparing SAE feature activations between
wild-type and mutant protein sequences.

Approach:
  1. Run ESM-2 → SAE on WT and mutant sequences
  2. Compute feature perturbation signature: Δf = f(mutant) - f(WT)
  3. Classify mechanism from perturbation pattern:
     - Many features ablated → LOF (loss-of-function)
     - Specific features amplified → GOF (gain-of-function)
     - New features activated + old preserved → neomorphic
     - Interface features disrupted → dominant-negative

References:
  - Badonyi & Marsh (2025) "Prevalence and mechanisms of non-LOF variants"
  - Badonyi et al. (2025) "PreMode: predicting disease mechanism"
"""

import gzip
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import AutoTokenizer, EsmModel

from src.training.sae import BatchTopKSAE


@dataclass
class MissenseVariant:
    """A missense variant with optional mechanism label."""
    gene: str
    uniprot_id: str
    position: int          # 1-indexed
    wt_residue: str        # single-letter AA
    mut_residue: str       # single-letter AA
    clinical_significance: str = ""
    mechanism: str = ""    # LOF, GOF, DN, neomorphic, or ""
    source: str = ""       # clinvar, mave, etc.


@dataclass
class PerturbationSignature:
    """SAE feature perturbation from a missense variant."""
    variant: MissenseVariant
    layer: int
    # Per-feature perturbation at the mutation site
    delta_local: np.ndarray   # (d_sae,) = f_mut[pos] - f_wt[pos]
    # Global perturbation (mean absolute change across all positions)
    delta_global: np.ndarray  # (d_sae,) = mean(|f_mut - f_wt|) across positions
    # Features significantly affected
    ablated_features: list[int]    # features that were active in WT but not in mut
    amplified_features: list[int]  # features with significantly higher activation in mut
    novel_features: list[int]      # features active in mut but not in WT
    # Summary statistics
    n_ablated: int = 0
    n_amplified: int = 0
    n_novel: int = 0
    total_perturbation: float = 0.0  # L1 norm of delta_local
    wt_active_count: int = 0
    mut_active_count: int = 0


def load_clinvar_missense(
    variant_summary_path: str,
    swissprot_map: dict[str, str],
    max_variants: int | None = None,
) -> list[MissenseVariant]:
    """Load missense variants from ClinVar variant_summary.txt.gz.

    Maps gene symbols to UniProt accessions using the Swiss-Prot mapping.

    Args:
        variant_summary_path: Path to variant_summary.txt.gz
        swissprot_map: gene_symbol → uniprot_accession mapping
        max_variants: Maximum variants to load

    Returns:
        List of MissenseVariant objects with UniProt IDs
    """
    # Parse p.XxxNNNYyy (HGVS protein notation) to extract position and AAs
    hgvs_pattern = re.compile(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})')

    aa3to1 = {
        'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
        'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
        'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
        'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
        'Ter': '*',
    }

    variants = []
    seen = set()

    opener = gzip.open if variant_summary_path.endswith('.gz') else open
    with opener(variant_summary_path, 'rt') as f:
        header = f.readline().strip().split('\t')
        # Find column indices
        cols = {h: i for i, h in enumerate(header)}
        gene_col = cols.get('GeneSymbol', cols.get('#GeneSymbol', 4))
        name_col = cols.get('Name', 2)
        type_col = cols.get('Type', 1)
        clin_col = cols.get('ClinicalSignificance', 6)

        for line in f:
            fields = line.strip().split('\t')
            if len(fields) <= max(gene_col, name_col, type_col, clin_col):
                continue

            # Only single nucleotide variants (missense)
            if fields[type_col] != 'single nucleotide variant':
                continue

            name = fields[name_col]
            gene = fields[gene_col]
            clin_sig = fields[clin_col]

            # Parse HGVS protein notation
            match = hgvs_pattern.search(name)
            if not match:
                continue

            wt_aa3, pos_str, mut_aa3 = match.groups()
            wt_aa = aa3to1.get(wt_aa3)
            mut_aa = aa3to1.get(mut_aa3)
            if not wt_aa or not mut_aa or mut_aa == '*':
                continue  # Skip nonsense/stop-gain
            pos = int(pos_str)

            # Map gene to UniProt
            uniprot_id = swissprot_map.get(gene, "")
            if not uniprot_id:
                continue

            # Deduplicate
            key = (uniprot_id, pos, wt_aa, mut_aa)
            if key in seen:
                continue
            seen.add(key)

            variants.append(MissenseVariant(
                gene=gene,
                uniprot_id=uniprot_id,
                position=pos,
                wt_residue=wt_aa,
                mut_residue=mut_aa,
                clinical_significance=clin_sig,
                source="clinvar",
            ))

            if max_variants and len(variants) >= max_variants:
                break

    return variants


def build_gene_to_uniprot_map(
    swissprot_xml_path: str | None = None,
    idmapping_path: str | None = None,
    swissprot_cache: str | None = None,
) -> dict[str, str]:
    """Build gene symbol → UniProt accession mapping from Swiss-Prot.

    Uses either the ID mapping file or parsed Swiss-Prot cache.
    Only includes human proteins.
    """
    gene_map = {}

    if idmapping_path and os.path.exists(idmapping_path):
        # Use UniProt ID mapping file (faster)
        opener = gzip.open if idmapping_path.endswith('.gz') else open
        with opener(idmapping_path, 'rt') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3 and parts[1] == 'Gene_Name':
                    gene_map[parts[2]] = parts[0]
        return gene_map

    if swissprot_cache and os.path.exists(swissprot_cache):
        import pickle
        with open(swissprot_cache, 'rb') as f:
            annotations = pickle.load(f)
        for ann in annotations:
            if 'Homo sapiens' in ann.organism:
                # Use accession as key; try to extract gene name from features
                gene_map[ann.accession] = ann.accession
        return gene_map

    return gene_map


def build_uniprot_sequence_map(swissprot_cache_path: str) -> dict[str, str]:
    """Build UniProt accession → sequence mapping from Swiss-Prot cache."""
    import pickle
    with open(swissprot_cache_path, 'rb') as f:
        annotations = pickle.load(f)

    seq_map = {}
    for ann in annotations:
        seq_map[ann.accession] = ann.sequence
    return seq_map


def create_mutant_sequence(wt_sequence: str, position: int, wt_aa: str,
                           mut_aa: str) -> str | None:
    """Create mutant sequence by substituting a single amino acid.

    Args:
        wt_sequence: Wild-type protein sequence
        position: 1-indexed mutation position
        wt_aa: Expected wild-type residue
        mut_aa: Mutant residue

    Returns:
        Mutant sequence, or None if WT residue doesn't match
    """
    idx = position - 1
    if idx < 0 or idx >= len(wt_sequence):
        return None
    if wt_sequence[idx] != wt_aa:
        return None
    return wt_sequence[:idx] + mut_aa + wt_sequence[idx + 1:]


@torch.no_grad()
def compute_perturbation_signature(
    wt_sequence: str,
    mut_sequence: str,
    variant: MissenseVariant,
    esm_model: EsmModel,
    tokenizer,
    sae: BatchTopKSAE,
    layer: int,
    device: str = "cuda",
    significance_threshold: float = 0.1,
) -> PerturbationSignature:
    """Compute SAE feature perturbation from a single missense variant.

    Args:
        wt_sequence: Wild-type protein sequence
        mut_sequence: Mutant protein sequence
        variant: Variant metadata
        esm_model: Frozen ESM-2 model
        tokenizer: ESM-2 tokenizer
        sae: Trained SAE for this layer
        layer: ESM-2 layer index (0-indexed)
        device: CUDA device
        significance_threshold: Minimum activation change to count as significant

    Returns:
        PerturbationSignature with per-feature perturbation analysis
    """
    def get_sae_features(sequence: str) -> np.ndarray:
        spaced = " ".join(list(sequence))
        encoded = tokenizer(
            spaced, return_tensors="pt", truncation=True, max_length=1024
        ).to(device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = esm_model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                output_hidden_states=True,
            )

        # hidden_states[0] = embeddings, [i+1] = layer i output
        layer_acts = outputs.hidden_states[layer + 1].float()
        # Skip CLS token, take only AA tokens
        seq_len = len(sequence)
        aa_acts = layer_acts[0, 1:seq_len + 1, :]  # (seq_len, d_model)

        output = sae(aa_acts)
        return output.f.cpu().numpy()  # (seq_len, d_sae)

    # Get WT and mutant SAE features
    f_wt = get_sae_features(wt_sequence)
    f_mut = get_sae_features(mut_sequence)

    pos_idx = variant.position - 1  # 0-indexed

    # Local perturbation at mutation site
    delta_local = f_mut[pos_idx] - f_wt[pos_idx]

    # Global perturbation (mean absolute change)
    min_len = min(f_wt.shape[0], f_mut.shape[0])
    delta_global = np.abs(f_mut[:min_len] - f_wt[:min_len]).mean(axis=0)

    # Classify feature changes at mutation site
    wt_active = f_wt[pos_idx] > 0
    mut_active = f_mut[pos_idx] > 0

    ablated = np.where(wt_active & ~mut_active)[0].tolist()
    novel = np.where(~wt_active & mut_active)[0].tolist()

    # Amplified: active in both but significantly stronger in mutant
    both_active = wt_active & mut_active
    if both_active.any():
        ratio = np.where(both_active & (f_wt[pos_idx] > 1e-6),
                         f_mut[pos_idx] / np.maximum(f_wt[pos_idx], 1e-6), 1.0)
        amplified = np.where(both_active & (ratio > 1.5))[0].tolist()
    else:
        amplified = []

    return PerturbationSignature(
        variant=variant,
        layer=layer,
        delta_local=delta_local,
        delta_global=delta_global,
        ablated_features=ablated,
        amplified_features=amplified,
        novel_features=novel,
        n_ablated=len(ablated),
        n_amplified=len(amplified),
        n_novel=len(novel),
        total_perturbation=float(np.abs(delta_local).sum()),
        wt_active_count=int(wt_active.sum()),
        mut_active_count=int(mut_active.sum()),
    )


def classify_mechanism(
    signatures: list[PerturbationSignature],
) -> str:
    """Classify variant mechanism from multi-layer perturbation signatures.

    With TopK SAEs, the number of active features is always k per position.
    So mechanism must be inferred from WHICH features change and by HOW MUCH,
    not from counting active features.

    Classification logic:
    - LOF: large perturbation, high feature turnover (many features replaced),
           net decrease in activation magnitude of retained features
    - GOF: moderate perturbation, few features replaced but some strongly
           amplified, net increase in activation magnitude
    - DN: moderate perturbation concentrated at specific positions,
          core features retained but with altered magnitudes
    - Neomorphic: novel features that are strongly activated (not just
                  replacing weak features due to TopK reranking)
    - Benign: small perturbation, features largely unchanged

    Args:
        signatures: PerturbationSignature from multiple SAE layers

    Returns:
        Predicted mechanism: "LOF", "GOF", "DN", "neomorphic", or "benign"
    """
    total_perturbation = sum(s.total_perturbation for s in signatures)
    total_ablated = sum(s.n_ablated for s in signatures)
    total_amplified = sum(s.n_amplified for s in signatures)

    # Feature turnover: fraction of WT features that were replaced
    wt_total = sum(s.wt_active_count for s in signatures)
    turnover = total_ablated / max(wt_total, 1)

    # Net magnitude change: do surviving features get stronger or weaker?
    # Compute from delta_local: positive = gained strength, negative = lost
    net_magnitude = sum(float(s.delta_local.sum()) for s in signatures)

    # Amplification strength: how much do amplified features increase?
    amp_strength = sum(
        float(s.delta_local[s.delta_local > 0].sum()) for s in signatures
    )
    abl_strength = sum(
        float(abs(s.delta_local[s.delta_local < 0].sum())) for s in signatures
    )

    # Global spread: ratio of global perturbation to local perturbation
    local_pert = sum(float(np.abs(s.delta_local).sum()) for s in signatures)
    global_pert = sum(float(np.abs(s.delta_global).sum()) for s in signatures)
    spread_ratio = global_pert / max(local_pert, 1e-6)

    if total_perturbation < 50.0:
        return "benign"

    # High turnover + net loss of activation = LOF
    if turnover > 0.3 and net_magnitude < 0:
        return "LOF"

    # Strong amplification + low turnover = GOF
    if amp_strength > abl_strength * 1.5 and turnover < 0.2:
        return "GOF"

    # Moderate perturbation with widespread effect = DN
    if spread_ratio > 0.3 and 0.1 < turnover < 0.4:
        return "DN"

    # High perturbation + high turnover = LOF (default for severe disruption)
    if turnover > 0.25:
        return "LOF"

    # Moderate, balanced perturbation
    if net_magnitude > 0:
        return "GOF"

    return "LOF"


def compute_perturbation_features(
    signatures: list[PerturbationSignature],
) -> np.ndarray:
    """Extract numerical features from multi-layer perturbation signatures
    for machine learning classification.

    Returns a feature vector capturing the perturbation pattern across layers.
    """
    features = []
    for sig in signatures:
        features.extend([
            sig.n_ablated,
            sig.n_amplified,
            sig.n_novel,
            sig.total_perturbation,
            sig.wt_active_count,
            sig.mut_active_count,
            sig.mut_active_count / max(sig.wt_active_count, 1),  # activity ratio
            np.abs(sig.delta_local).mean(),     # mean local perturbation
            np.abs(sig.delta_global).mean(),    # mean global perturbation
            np.abs(sig.delta_local).max(),      # max local perturbation
        ])
    return np.array(features, dtype=np.float32)
