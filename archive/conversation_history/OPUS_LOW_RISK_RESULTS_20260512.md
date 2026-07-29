# Low-Risk TODO Results for Opus Replanning (2026-05-12)

This packet summarizes the low-risk experiments run after the Opus universal
primitives pivot.  It is meant to support a new planning decision, not to argue
for a manuscript claim.

## Executive Summary

The R2 cross-model triplets remain statistically interesting, but the
annotation evidence is currently weak.  Cheap sequence labels produce only very
small mutual information, and the staged Pfam / Swiss-Prot / AlphaFold resources
do not cover the broad UniRef50 top-firing set.  R2 should not proceed directly
to causal intervention unless Opus explicitly wants a high-risk diagnostic.

R1 also narrowed further: gene-level mechanism prediction failed under a
Pfam-family holdout proxy, and dbNSFP/CADD/REVEL cannot be matched to the
current protein-HGVS-only IndelMissense records.

## Completed Low-Risk Runs

### R2: Cheap-Label Annotation of 38 Universal Triplets

Script:

- `Research2/scripts/29_universal_primitive_annotation.py`

Main output:

- `Research2/results/circuit_analysis/universal_primitives_uniref500_20260512/`

Setup:

- 38 balanced-200 wide-match universal triplets.
- 500 UniRef50 sequences.
- Top 100 firing positions per triplet.
- Labels: amino-acid identity, coarse residue chemistry, source, sequence-position bin.

Result:

- All 38 triplets have some weak amino-acid / chemistry enrichment.
- Best MI across simple labels:
  - min: 0.000116 nats
  - median: 0.000688 nats
  - max: 0.001924 nats
- This is far below the Opus interpretability gate of 0.1 nats.

Interpretation:

- The current data support "cross-model conserved latent features" better than
  "named biological primitives."
- Simple sequence-composition labels are not strong enough to name the triplets.
- Do not use these results as a positive biological-primitive claim.

### R2: Resource Coverage Annotation

Script:

- `Research2/scripts/32_universal_resource_annotation.py`

UniRef500 output:

- `Research2/results/circuit_analysis/universal_primitives_resource_annotation_20260512/`

UniRef500 result:

- Top-firing events: 3,800.
- Triplets: 38.
- Unique accessions: 452.
- Pfam-covered accessions: 0.
- Swiss-Prot-covered accessions: 0.
- AlphaFold-covered accessions: 1.

Balanced-200 output:

- `Research2/results/circuit_analysis/universal_primitives_balanced200_resource_annotation_20260512/`

Balanced-200 result:

- Top-firing events: 500.
- Triplets: 10.
- Unique accessions: 33.
- Pfam-covered accessions: 7.
- Swiss-Prot-covered accessions: 7.
- AlphaFold-covered accessions: 0.
- Most Swiss-Prot overlaps are chain/topology labels. Pfam event hits are sparse
  and concentrated in only a few triplets.

Interpretation:

- The broad UniRef500 top-firing set is not resource-ready for Pfam /
  Swiss-Prot / AlphaFold annotation using the currently staged local resources.
- The balanced calibration set has limited coverage, but it is biased toward
  lysozyme/random controls and does not solve the universal-primitives question.
- A resource-annotated cohort is needed before serious biological naming:
  either Swiss-Prot/Pfam-covered sequences, or additional UniRef-to-UniProt /
  InterPro / AlphaFold mapping.

### R1: Gene-Level Mechanism Gate

Script:

- `Research1/scripts/39_gene_level_mechanism.py`

Output:

- `Research1/results/variant_effect/gene_level_mechanism_20260512.{json,md}`

Setup:

- 253 genes.
- Classes: LOF 142, GOF 67, DN 44.
- Holdout: dominant Pfam family per UniProt as a proxy because a Pfam clan map
  is not staged.

Result:

- Macro-AUC: 0.5665.
- Macro-F1: 0.3769.
- Accuracy: 0.4743.
- Per-class AUC:
  - DN: 0.5279
  - GOF: 0.5879
  - LOF: 0.5837
- Gate: `drop_mechanism_headline`.

Interpretation:

- R1 should not claim robust gene-level mechanism prediction.
- Mechanism results can remain as a negative / limited diagnostic, but not as a
  central headline.

### R1: Indel Competitor Staging

Script:

- `Research1/scripts/40_indel_competitor_attempt.py`

Output:

- `Research1/results/variant_effect/indel_competitor_attempt_20260512.{json,md}`

Result:

- dbNSFP GRCh38 is staged.
- dbNSFP exposes CADD and REVEL columns.
- Current IndelMissense v1 records lack `chrom`, `pos`, `ref`, and `alt`.
- A first-10k dbNSFP row scan found 0 indel-like rows.
- Gate: `drop_head_to_head_for_current_pass`.

Interpretation:

- A valid CADD/REVEL indel head-to-head cannot be run against the current
  protein-HGVS-only benchmark.
- Reopening this requires rebuilding IndelMissense from genomic VCF-style
  records or adding genomic coordinate recovery.

## Recommended Decision Questions for Opus

1. Should R2 continue as a "universal biological primitives" paper, or be
   downgraded to "cross-model conserved latent features" until stronger
   annotation is available?
2. Should the next R2 experiment build a resource-annotated cohort first
   instead of running causal intervention?
3. Should R2 causal intervention be deferred unless Pfam / Swiss-Prot /
   structural annotation reaches a clear threshold?
4. Should R1 now freeze around the bounded IndelMissense benchmark plus
   negative diagnostics, dropping the gene-level mechanism section?
5. Is it worth rebuilding IndelMissense with genomic coordinates for CADD/REVEL,
   or should that be future work?

## Current Compute State

- H200 pod `jiaotongdamoxing-zhk-zip-final-1gpu-0511` remains reserved.
- No active experiment process is running.
- GPU memory/utilization at last check: 0 MiB / 0%.
