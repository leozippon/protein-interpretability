# Review of `OPUS_BRILLIANT_FINAL_20260517.md`

This is a technical review of Opus's latest pivot proposal, checked against the
current repository results.

## Bottom Line

I agree with the main strategic direction:

1. R1 should stop targeting Nature Methods as a primary venue and should be
   reframed as a resource / calibrated audit paper.
2. R2's N-terminal attention-sink subset is the strongest current discovery in
   the project and should be promoted to the headline.
3. A causal ablation experiment is the right next high-value experiment.

However, several details in Opus's phrasing and proposed ablation design need
to be tightened before execution or manuscript writing.

## Critical Correction: What `first2_fraction = 1.00` Means

Opus states that T011/T018/T023 "fire at the first two residues with fraction
1.00 across 700 proteins." This is too strong.

The actual evidence is:

- The M-1 cohort has 700 sequences.
- For each triplet, the characterization script saved the top 100 firing rows.
- For T011/T018/T023, all 100 saved top-firing rows are within residues 1-2.
- These top rows come from 100 unique sequences per triplet, not all 700
  sequences.

Current files:

- `Research2/results/circuit_analysis/attention_sink_subset_20260516/attention_sink_subset.tsv`
- `Research2/results/circuit_analysis/attention_sink_biological_correlate_20260516/biological_correlates.tsv`

Observed top-row source distributions:

| Triplet | Top rows | Unique sequences | Source distribution | First2 fraction |
|---|---:|---:|---|---:|
| T011 | 100 | 100 | swissprot_n1:68; real_lysozyme:20; random_uniref50:12 | 1.00 |
| T018 | 100 | 100 | swissprot_n1:71; real_lysozyme:20; random_uniref50:9 | 1.00 |
| T023 | 100 | 100 | swissprot_n1:61; real_lysozyme:18; random_uniref50:21 | 1.00 |
| T025 | 100 | 11 | random_uniref50:52; swissprot_n1:48 | 0.00 |

Recommended wording:

> The top-firing events for T011/T018/T023 are completely concentrated at the
> N-terminal edge: 100/100 saved top-firing positions for each triplet fall
> within residues 1-2, compared with 202/3700 background top-firing rows
> (BH q = 2.72e-117).

Avoid:

> The features fire at residues 1-2 in every protein.

That claim has not been tested yet.

## Correction: "Initiator Methionine" Should Be Used Carefully

The current evidence supports "N-terminal edge / start-context sink" more
strongly than "initiator-methionine sink" in the strict biological sense.

Reasons:

- T011/T018/T023 are strongly N-terminal.
- Their top 3-mers are M-rich:
  - T011: MKA, MKI, MTA.
  - T018: MKK, MKI, MKA.
  - T023: MRI, MRS, MRA.
- But `context_starts_m` is not exactly 1.00 for all:
  - T011: 0.99.
  - T018: 0.95.
  - T023: 0.89.
- Some representative contexts do not visibly start with M because the saved
  context window can be centered at residue 2 or affected by edge truncation.

Recommended wording:

> N-terminal edge / initiator-context attention sinks.

Use "initiator-methionine" only after an explicit start-M stratification test
or after showing that the firing event is tied to the first methionine rather
than simply to the sequence boundary.

## R2 Pivot Assessment

The pivot is scientifically sound:

- The broad universal-biological-primitive claim failed.
- The narrow attention-sink subtype is strong and concrete.
- The analogy to NLP attention sinks is useful, but should be phrased as an
  empirical analog, not mechanistic identity.
- The quality diagnostic result is useful but preliminary:
  - Mature v2 atlas: 38 universal triplets.
  - Early10k atlas: 16 universal triplets.
  - Old v1 ProtGPT2/ZymCTRL checkpoints were not mounted in the current pod.

Recommended R2 headline:

> Cross-model conserved N-terminal attention-sink features in protein language
> models.

This is safer than:

> Initiator-methionine attention sinks emerge convergently across protein
> language models.

The latter may become valid after the causal and start-M stratified analyses.

## Causal Ablation Design: Needed Fixes

Opus's proposed causal ablation is directionally right, but the metric needs to
be defined carefully for decoder-only language models.

### Issue 1: Perplexity at positions 1-2 may be the wrong primary target

In a decoder-only LM, token `i` is predicted from tokens `< i`. A feature firing
at residues 1-2 may influence later predictions by acting as an attention sink,
not necessarily the likelihood of residues 1-2 themselves. Position 1 may be
conditioned only on BOS / prefix.

Better primary readouts:

1. Teacher-forced delta NLL for target tokens at positions 2-10 and 10+.
2. Delta NLL stratified by whether the ablated feature is active at residues
   1-2 in that sequence.
3. Attention-received redistribution away from residues 1-2 after ablation.
4. Optional generation readout: sequence starts, N-terminal motif stability,
   and downstream quality, but this should be secondary.

### Issue 2: Existing steering hook is MLP-output patching, not attention patching

`Research2/src/analysis/circuit_discovery.py` has a TopK-aware steering hook
that modifies the CLT-explained same-layer MLP component. This can be reused,
but it is not directly an attention-module ablation. Any attention redistribution
will be an indirect effect.

Therefore the causal claim should be:

> Ablating the sparse feature changes likelihood / attention statistics
> consistent with an attention-sink role.

Not initially:

> The feature is the attention sink mechanism.

The stronger phrasing is justified only if the ablation result is very clean.

### Issue 3: Controls are required

The ablation experiment must include controls, otherwise reviewers can argue
that removing any highly active feature damages the model.

Recommended controls:

- Random same-layer feature ablations matched for activation frequency.
- Non-N-terminal attention-associated T025 as a negative / specificity control.
- High-correlation but non-attention triplets if available.
- Sham hook with multiplier 1.0.

Minimum acceptable table:

| Condition | Position 2-10 delta NLL | Position 10+ delta NLL | Attention redistribution | Interpretation |
|---|---:|---:|---:|---|
| T011/T018/T023 ablated | positive | near zero or smaller | positive | supports sink role |
| T025 ablated | different profile | different profile | weaker / local | specificity |
| matched random features | near zero or nonspecific | near zero or nonspecific | near zero | specificity |

### Issue 4: Per-model feature mapping must be explicit

Each triplet consists of three model-specific features, not one universal
feature. The causal script should ablate per-model features separately:

- ProtGPT2 feature/layer from the triplet table.
- ZymCTRL feature/layer from the triplet table.
- ProGen2-medium feature/layer from the triplet table.

The report should show per-model effects and a cross-model meta-summary.

### Issue 5: Test start-M dependence explicitly

Add a cheap diagnostic before or inside the ablation:

- Start-M sequences versus non-start-M sequences.
- First-residue M versus other amino acids.
- Top-firing rank and activation at residues 1 and 2 per sequence.

If the effect only appears in start-M sequences, the "initiator-methionine"
phrasing becomes much stronger. If it appears regardless of residue identity,
"N-terminal edge sink" is the safer term.

## R1 Assessment

I agree with Opus's R1 decision:

- R1 should not continue as a Nature Methods push.
- Do not chase VUS reclassification right now.
- Freeze the core scientific conclusion unless a very targeted diagnostic is
  needed for internal confidence.

The one nuance: R1 should not be framed only as an IndelMissense paper unless
the manuscript can convincingly present the dataset as reusable despite its
size (6,649 records). The safer resource framing is:

> IndelMissense v1 plus a calibrated audit of what ESM-2 sparse-feature
> perturbations do and do not provide for clinical variant interpretation.

The negative diagnostics are a strength if presented compactly and honestly.

## Recommended Next Action

Proceed with R2 causal ablation, but implement it with corrected claims and
controls:

- New file: `Research2/scripts/40_attention_sink_causal_ablation.py`.
- Inputs:
  - `Research2/results/circuit_analysis/universal_atlas_balanced200_wide_triplets_20260512.tsv`.
  - `Research2/results/circuit_analysis/triplet_characterization_20260515_nperm2000/cohort.json`.
  - CLT checkpoints for ProtGPT2 v2, ZymCTRL v2, ProGen2-medium.
- Output:
  - Per-model TSV with intact vs ablated teacher-forced NLL by position bin.
  - Attention redistribution metrics.
  - Control-feature comparison.
  - Summary markdown with PASS/PARTIAL/FAIL gates.

Acceptance should be more conservative than Opus's current gates:

- PASS: T011/T018/T023 show consistent, activation-conditioned NLL or attention
  redistribution effects across at least two models and exceed matched random
  controls.
- PARTIAL: effect exists in one model or only in attention but not NLL.
- FAIL: no effect beyond controls; retain correlation-only framing.

## Final Recommendation

Adopt Opus's strategic pivot, but correct the overstatement about "across 700
proteins" and design the ablation around decoder-LM causal semantics. With
those corrections, this is the best current path: R2 becomes the priority paper,
R1 becomes a resource / audit paper, and no more broad R1 rescue experiments
should be launched unless Opus explicitly requests them.
