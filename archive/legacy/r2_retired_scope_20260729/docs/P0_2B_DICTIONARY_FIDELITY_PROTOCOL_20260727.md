# P0-2b prospective dictionary-fidelity qualification

**Status:** frozen before evaluation  
**Frozen:** 2026-07-27  
**Scope:** revised downstream instrument qualification only

## Separation from P0-2

P0-2 remains governed by
`P0_2_DICTIONARY_PROTOCOL_20260717.md`. Its executed specification and
eligibility receipt are immutable. P0-2b does not replace its FVU,
dead-feature, L0 or firing-rate gates and cannot change atlas eligibility.

P0-2b asks a new question: which completed P0-2 dictionaries preserve
next-token behaviour well enough to be considered for a revised downstream
programme? Methods that terminated at P0-2's validation sparsity-match gate
have no full checkpoint and are recorded as unavailable, not rerun.

## Frozen inputs

- Models: ProtGPT2, ZymCTRL and ProGen2-medium, verified against the config,
  weight-tree and tokenizer-tree hashes recorded by the exact-cache manifests.
- Dictionaries: all 27 completed P0-2 full-run checkpoints, seeds 17, 29 and
  43, bound through the executed P0-2 gate specification.
- Evaluation cohort: 240 unique sequences selected from the EC-labelled
  ZymCTRL Swiss-Prot FASTA, restricted to the canonical 20 amino acids.
- Length: 64–246 amino acids. Metadata-only pre-freeze audits found 272
  sequences at 250 residues, but a tokenizer-only preflight observed a
  260-token ZymCTRL input. Tightening the ceiling to 246 produced 248 eligible
  sequences; 240 preserves a hash-selected reserve without weakening
  exclusions.
- Exclusions: sequence-hash union of the original P0-2 train, validation and
  test manifests for UniRef50 and ZymCTRL, plus the first 40,000 source FASTA
  records. The prefix exclusion reconstructs and exceeds every ZymCTRL source
  prefix used by TG-01, TG-02, TG-03, TG-06 and TG-07; the broadest such pilot
  prefix ended before source record 30,885.
- Selection: the 240 lowest values of
  `SHA256("p0_2b_20260727:" + sequence_sha256)` after exclusions.
- Mean-ablation vector: float32 mean of the frozen training-cache MLP-output
  target at the analysis layer, computed separately for each model.

The cohort, means, protocol, implementation files, original receipt, results,
manifests and checkpoints are SHA-256-bound in one executed specification
before any evaluation forward pass.

## Native inputs and scoring

- ProtGPT2: raw sequence.
- ZymCTRL: `EC<sep><start>sequence<end>`.
- ProGen2-medium: native N-to-C control token followed by sequence,
  `1sequence`.

Use the P0-2 tokenizer settings and a 256-token maximum. Before every forward
pass, tokenize without truncation and fail if any native input exceeds that
limit; the 246-residue ceiling retains the measured control/end-token overhead.
Score next-token targets within the protein sequence. Exclude ZymCTRL EC,
separator, start and end targets. ProtGPT2 is scored at its native BPE-token
level and must not support residue-level interpretation claims.

## Intervention

Use relative depth 0.5 with half-up rounding: layer 18 for 36-layer models and
layer 13 for ProGen2-medium. Capture the live architecture-specific inputs to
the trained eight-layer source window. Decode only the selected target MLP
output and allow subsequent model computation to recompute normally.

Evaluate four paths for every batch:

1. clean forward pass;
2. exact reinjection of the unmodified target MLP output;
3. replacement by the frozen training-target mean;
4. replacement by the checkpoint's dictionary reconstruction, including its
   frozen activation threshold.

## Metrics and gates

Let `CE_x` be next-token cross-entropy and `KL(clean || x)` the downstream
forward KL, both averaged over scored targets.

```text
loss recovered = (CE_mean - CE_dictionary) / (CE_mean - CE_clean)
KL recovered   = 1 - KL(clean || dictionary) / KL(clean || mean)
```

Both recovered metrics are undefined and the run fails qualification unless:

- `CE_mean - CE_clean >= 0.05` nats/token; and
- `KL(clean || mean) >= 0.01` nats/token.

The intervention implementation also fails unless exact reinjection has:

- maximum absolute logit difference exactly zero;
- absolute CE difference at most `1e-6` nats/token; and
- argmax agreement exactly 1.0.

Report paired 1,000-replicate sequence-cluster bootstrap intervals with seed
20260727. The intervals are descriptive and are not additional gates.

A model/method qualifies for the revised downstream programme only when every
available seed has loss recovered at least 0.80, KL recovered at least 0.80,
valid denominators and a passing reinjection check. Dense low-rank controls
are reported but cannot qualify as sparse instruments.

## Decision rule

- If no sparse model/method qualifies, stop P2–P8 and report an instrument
  failure for the tested checkpoints.
- If one or more qualify, downstream work may use only those exact
  model/methods and must preserve seed-level reporting.
- Do not start behavior-weighted training, pair transcoders, oracle labeling,
  dense semantic audits, DMS interventions, steering or new atlas work until
  the P0-2b output is complete and inspected.

P0-2b contains no matched text arm and no fixed-versus-recomputed-attention
comparison. It can qualify protein-side instruments but cannot by itself
establish why Anthropic-style interpretability transfers differently from text
to protein generators.
