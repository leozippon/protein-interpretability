# ProteinInterpret Framework Notes

This directory records the shared benchmark schema before the framework grows
into a full Python package.

## Shared Result Schema

Every benchmark row should be reducible to:

- `track`: `encoder` or `decoder`.
- `task`: the concrete evaluation task.
- `dataset`: the dataset or cohort name.
- `method`: explanation or reference method.
- `metric`: metric name.
- `value`: scalar value or concise comparison.
- `ci95`: confidence interval when available.
- `n`: number of variants, assays, sequences or rows.
- `gate`: pass/fail/reference/diagnostic status.
- `interpretation`: one-sentence meaning of the result.
- `source`: local evidence path.

## Encoder Benchmark Axes

- **Predictive utility:** pathogenicity AUC, indel AUC, DMS Spearman rho.
- **Localization:** whether top explained residues overlap variant sites,
  functional residues, domains, active sites, PTMs or binding sites.
- **Faithfulness:** whether masking, occluding, mutating or feature-patching
  the explanation target changes the model score more than matched controls.
- **Mechanism alignment:** LOF/GOF/DN, abundance/stability, active-site vs
  structural-disruption separation.
- **Robustness:** explanation stability across seeds, layers, homologous
  backgrounds and sequence windows.
- **Protein-specific metric candidate:** CRPI, catalytic-residue pointing
  index, using 3D distance to curated catalytic/active-site residues.

## Decoder Benchmark Axes

- **Cross-model conservation:** matched sparse features or triplets above a
  correlation threshold with permutation-null controls.
- **Characterization:** k-mer, position, BPE boundary, attention sink and
  high-norm signatures.
- **Biological grounding:** Pfam, EC, Swiss-Prot, motif and structural labels.
- **Generation faithfulness:** steering or ablation should change target
  generation metrics more than matched controls.
- **Checkpoint/model diffing:** whether explanation statistics distinguish
  mature checkpoints from weak or early checkpoints.
- **Causal intervention:** CLT feature, attention-head and head-set ablations
  should pass pre-specified gates before any causal mechanism claim.

## Current Boundary

The v0 tables are evidence ledgers, not a public leaderboard. The next step is
to add method runners and dataset loaders that can emit the same row schema.
