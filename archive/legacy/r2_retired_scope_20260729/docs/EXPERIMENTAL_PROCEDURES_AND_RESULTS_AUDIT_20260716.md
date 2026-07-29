# R2 Experimental Procedures and Results Audit

**Audit date:** 2026-07-16  
**Research direction:** Paper A, conserved sparse-readout and diagnostic audit  
**Scope:** all live R2 documentation, source/configuration files, locally
available results, compact recoverability evidence, manuscript source data, and
the B-side result tree  
**Disposition:** scientifically useful, evidence-corrected internal research
package; not yet an exactly reproducible public release

**Post-assessment package update (2026-07-17):** The independent assessment was
accepted as a major-revision specification. Immediate source-data remediation
added the complete direct-effect candidate/sign summary and converted four
historical JSON copies containing 18 non-finite constants into strict RFC 8259
release copies. Each replacement is `null` with its JSON path and original
token recorded, while the historical source hash is retained. A subsequent
receipt-bound Supplementary Table 11 added the prospectively frozen synthetic
P0-7 calibration without adding its sequence-bearing raw cohort or checkpoints.
The current verifier passes 65 manifest rows, 68 checksums and 1,649,686 package
bytes.
Counts below explicitly labelled “at audit time” remain the pre-remediation
snapshot.

## 1. Executive finding

R2 is a multi-stage audit of sparse internal readouts in three autoregressive
protein language models: ProtGPT2, ZymCTRL, and ProGen2-medium. It began as a
drug-design and enzyme-steering programme, but its strongest intervention and
biological-interpretation claims failed their gates. The defensible project is
now an audit that separates four questions:

1. Are sparse activation profiles recurrent across the three models under a
   fixed extraction and matching procedure?
2. What low-level, positional, or biological associations do the recurrent
   readouts retain?
3. Do specified feature, head, head-set, or generation interventions establish
   causal control?
4. Which selected linear signals present at the architecture-specific CLT
   inputs remain recoverable from the sparse codes?

The bounded answer is:

- **Recurrence:** yes under one fixed 200-sequence procedure: 38 triplets at
  absolute correlation ≥0.90 versus at most one in 30 sequence-assignment
  permutations.
- **Low-level association:** 37/38 triplets have at least one implemented
  association with local sequence, position, CLT-input norm, received
  attention, or a ProtGPT2 token boundary; the tests are exploratory and
  overlapping.
- **N-terminal subset:** T011, T018, and T023 select initiator-methionine
  contexts in their saved top events and have high **unnormalized** received
  attention. This is cross-model recurrent within the saved discovery and
  characterization outputs, not an independently replicated biological readout
  or a causal attention sink.
- **Checkpoint sensitivity:** one mature-versus-10k comparison yields 38 versus
  16 triplets and motivates a candidate diagnostic; one pair does not validate
  a general quality metric.
- **Causal control and steering:** every specified feature/head/head-set gate
  failed, and no raw comparison in the eight-target steering run met its
  positive gate. The steering selector's fallback included negative-attribution
  features in six class/layer cells, so that run is not a clean eight-class
  positive-direction test. These negatives apply only to the implemented
  sites, controls, strengths, feature selector, and scores.
- **Recoverability:** sparse codes retain selected linear signals from their
  own architecture-specific CLT inputs to varying degrees. The wider
  exploratory dictionaries produced mixed changes and missed the
  preregistered FVU criterion, so they do not rule out capacity or
  optimization.

### 1.1 Audit verdict by evidential level

| Evidential level | Current verdict | What may be claimed | What may not be claimed |
|---|---|---|---|
| Dictionary training/quality | Uneven, unmasked diagnostics | Reference and exploratory dictionaries were trained and evaluated; limitations are quantified | Matched mask-aware held-out quality, or uniformly high-fidelity dictionaries |
| Cross-model recurrence | Positive under the implemented procedure | Procedure-specific recurrent sparse readouts | Architecture-independent universal features or biological primitives |
| Semantic association | Low-level positive; matched rich-label analysis exploratory negative | Overlapping sequence/position/norm associations; three initiator-M readouts | General absence of biological information or a semantic dictionary |
| Checkpoint comparison | One-pair sensitivity result | Candidate checkpoint diagnostic | Validated prospective quality metric |
| Generation steering | No positive evidence under the implemented heuristic pipeline; feature-sign defect requires rerun | Pipeline diagnostic and separate older asymmetric lysozyme case study | Clean eight-class direct-effect failure, successful EC steering, therapeutic design, or a general lysozyme negative |
| Feature/head interventions | Negative for all specified gates | Bounded negative intervention evidence | Single-feature causality, causal attention sink, or exclusion of all mechanisms |
| Recoverability | Mixed, exploratory | Selected linear signals retained relative to each CLT's own input | Substrate-vs-dictionary attribution, capacity falsification, or distributed-mechanism proof |

### 1.2 Release-critical blockers

The current package cannot support exact independent end-to-end reproduction
because:

- the recovered, hash-verified 200-sequence atlas file and three reference CLTs
  are not yet in a persistent licensed public deposit;
- deployed-model provenance is incomplete at different levels: the complete
  ProtGPT2 tree is now verified to upstream commit
  `f71aa6cf063ad784ebd53881d11332fd098eaa58`; ZymCTRL's weight is verified
  and `3c532ef172b9cd2e95238baadf5167ebb89fbc32` is the best-supported snapshot
  but strict whole-tree proof remains incomplete; the deployed
  ProGen2-medium tree is a hybrid with no single matching upstream commit and
  requires a deposited local snapshot;
- full CLT checkpoints and several large caches remain on historical
  GPFS/OSS paths rather than a versioned public deposit;
- training and quick evaluation included padding in CLT loss, FVU, and feature
  statistics;
- the atlas null uses independent pair/layer sequence permutations rather than
  a coherent model-wise permutation;
- external database/tool versions are incomplete for some historical runs;
- discovery and evaluation are not fully separated for several analyses; and
- there is no release tag/commit plus DOI-backed data/code archive.

The manuscript source-data package is nevertheless internally strong for the
processed evidence it contains. After the evidence-correction and Table 11
rebuild its verifier passes 65 manifest rows and 68 checksums; the verifier's
total package-byte count is 1,649,686. That does not restore omitted
raw cohorts, checkpoints, generated sequences, or activation caches.

## 2. Audit scope, evidence hierarchy, and inventory

### 2.1 Materials reviewed

This audit reviewed:

- [`README.md`](../README.md), [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md), and the
  full [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md);
- the root `docs/PROJECT_STATUS.md` and `docs/PROJECT_LOG.md` rescue/evidence
  history;
- four training configurations, 61 live first-party files under `scripts/`,
  and seven live first-party files under `src/`, excluding bytecode;
- all 288 locally visible files under `results/` and 17 compact files under
  `evidence/`, with deeper numerical checks on canonical summaries and tables;
- the recoverability protocol, amendment log, and corrected-analysis plan;
- the evidence-corrected npj main text, eleven-table supplement, and 65-row
  source-data manifest; and
- historical/intermediate artifacts needed to identify superseded claims.

R2 has no project-local `PROJECT_LOG.md`; the chronological scientific record
is split between `docs/EXPERIMENT_LOG.md` and the repository-level project log.
The project-local experiment log also omits the major 13–18 May atlas,
characterization, N-terminal, and ablation runs; this audit consolidates them.

### 2.2 Evidence hierarchy used in this review

When records disagree, this document uses the following precedence:

1. current evidence-repair artifacts and their manifests;
2. exact canonical result JSON/TSV and current source-data copies;
3. the evidence-corrected July manuscript and supplement;
4. the dated experiment log and root project status; and
5. early interpretation notes, proposals, and legacy labels.

This ordering matters. For example, the May Swiss-Prot result file correctly
records what the old script produced, but its 0.1-nat decision gate is invalid;
the July re-audit is the appropriate interpretation layer. Likewise, the June
capacity-falsification narrative is superseded by the July evidence correction
even though the underlying wider-dictionary measurements remain valid.

### 2.3 Experimental flow

```text
Frozen protein generators
        │
        ├─> CLT training / unmasked quality diagnostics
        │          │
        │          ├─> fixed 200-sequence cross-model atlas + assignment null
        │          │         ├─> characterization / Swiss-Prot / basis probes
        │          │         ├─> N-terminal readout subset
        │          │         └─> candidate checkpoint comparison
        │          │
        │          ├─> feature, head, head-set and attention-output interventions
        │          └─> CLT-input versus sparse-code recoverability probes
        │
        └─> ZymCTRL direct-effect features + TopK-aware steering
                   ├─> heuristic eight-target benchmark
                   └─> Pfam / CLEAN / ESMFold / Foldseek controls
```

## 3. Models, hook points, and sparse dictionaries

### 3.1 Canonical reference panel

| Model | Transformer | Tokenization | Reference CLT | Training | Unmasked quality summary | Main use |
|---|---|---|---|---|---|---|
| ProtGPT2 | 36 layers, 1,280 hidden, 20 heads | multi-residue BPE | 36 layers, d=8,192, k=128, window=8 | 200k steps; UniRef50 | mean FVU 0.2013; dead 0.6197 | atlas and interventions |
| ZymCTRL | 36 layers, 1,280 hidden, 20 heads | residue-level vocabulary | 36 layers, d=8,192, k=128, window=8 | 200k steps; EC-labelled FASTA | mean FVU 0.3307; dead 0.6682 | atlas and steering |
| ProGen2-medium | 27 layers, 1,536 hidden, 16 heads | residue-level vocabulary | 27 layers, d=4,096, k=64, window=8 | 100k steps; UniRef50 | mean FVU ≈0.33; dead ≈0.57 from an older 200-sequence evaluation | atlas and interventions |

The April experiment log calls ProGen2-medium 1,024-dimensional. The staged
model configuration records `n_embd=1536`; the 1,024 value is a historical
documentation error.

### 3.2 What the CLTs encode and reconstruct

For source layer `l`, the CLT encodes
`z_l = TopK(ReLU(W_l x_l + b_l))`. Its windowed decoder reconstructs MLP
outputs over the following eight-layer window. The encoded tensor `x_l` is not
one homologous raw residual stream:

- ProtGPT2 and ZymCTRL use the layer-normalized post-attention MLP input.
- ProGen2-medium uses the layer-normalized block input shared by attention and
  the MLP.

Consequently, “MLP-output CLT feature” is imprecise: the feature is defined by
an architecture-specific normalized input while the decoder target is an MLP
output. Cross-model matches are activation-profile matches, not identical
directions in one shared representation space.

### 3.3 Padding-mask defect

The training/evaluation activation path did not pass `attention_mask`, and CLT
loss/FVU/firing calculations did not exclude padding. Reconstruction of the
saved 100-sequence quick evaluations found:

| Model | Padding positions / reported positions | Fraction |
|---|---:|---:|
| ProtGPT2 | 298 / 8,342 | 3.57% |
| ZymCTRL | 673 / 24,466 | 2.75% |

Training-like random-pair samples had larger estimated mean padding fractions:
20.81% for ProtGPT2, 20.23% for ProGen2-medium, and 10.66% for ZymCTRL. Those
are local sample estimates, not exact remote-training fractions. Atlas
activations were extracted sequence-by-sequence, so right padding did not
directly contaminate its real-token profiles; dictionary optimization was
still exposed to padding gradients.

## 4. Dictionary training and quality procedures

### 4.1 EXP-R2-001: initial d=4,096 training

**Procedure.** One H200 GPU; 100,000 steps; batch size 2; d=4,096; k=64;
window=8; learning rate 3×10^-4 with 2,000-step warmup and cosine decay. The
initial run had no dead-feature resampling or activation normalization.

| Model | Final FVU | Dead fraction | L0 | Audit result |
|---|---:|---:|---:|---|
| ProtGPT2 | ≈0.30 | 0.919 | 64 | Catastrophic feature death; loss scale much larger than the other models. |
| ZymCTRL | ≈0.35 | 0.430 | ≈62 | Moderate feature death. |
| ProGen2-medium | ≈0.32 | 0.292 | ≈57 | Best of the initial runs. |

ProtGPT2's dead fraction began rising sharply after the 10,000-step dead-unit
tracking threshold and never recovered. The training code was then extended
with error-direction resampling, resume support, shuffled ordering, and model
shape inference. These initial checkpoints are developmental, not the atlas
references.

### 4.2 EXP-R2-002: resampling rerun and layer map

**Procedure.** Evaluate the April rerun checkpoints after adding dead-feature
resampling. A feature was considered evaluation-dead if it fired on fewer than
0.1% of held-out tokens, a stricter rule than the training no-fire window.

| Model | Rerun FVU | Rerun dead | Mean alive count | Usable layers under alive≥20%, FVU<0.5 |
|---|---:|---:|---:|---:|
| ProtGPT2 | 0.328 | 0.819 | 743 / 4,096 | 11/36, contiguous L5–15 |
| ZymCTRL | 0.380 | 0.583 | 1,706 / 4,096 | 23/36, L0–16 and L25–30 |
| ProGen2-medium | 0.333 | 0.574 | 1,745 / 4,096 | 20/27, L2–14 and L20–26 |

Resampling helped ProtGPT2 relative to its 91.9% dead initial run but did not
make it a strong dictionary. The observed ZymCTRL/ProGen dead fractions rose
under the stricter evaluation definition, so training and evaluation dead-unit
rates must not be compared as identical quantities.

### 4.3 EXP-R2-003/004/006: EC feature analysis and architecture correction

The early ZymCTRL analysis counted 6,214 features specific to one of eight
enzyme-labelled prompts and 1,365 active across all eight, then reported raw
layer-35 EC separation `L2=564.51`. It interpreted this as a three-phase
biological architecture and recommended steering layers 33–35.

That interpretation is superseded. Layer 35 was about 90% dead, so its large
raw separation did not identify a usable CLT circuit. The quality-weighted
reconciliation selected:

| Role | Reconciled layer | Evidence |
|---|---:|---|
| EC prompt recognition | L3 | effective L2 19.06; alive ≈35% |
| shared generation readout | L12 | quality 0.341; alive ≈64% |
| deepest usable output readout | L30 | effective L2 13.34; alive ≈21% |

The terms “universal generation,” “enzyme-specific output,” catalytic motifs,
and evolutionary analogy in the early log are hypotheses, not validated
biological interpretations.

### 4.4 v2 reference dictionaries

ProtGPT2 and ZymCTRL were retrained to d=8,192, k=128 for 200,000 steps, with
resampling every 2,500 steps, a 5,000-step dead threshold, 3,000-step warmup,
and 300,000 source sequences. Their quick-evaluation results are the reference
values in Section 3.1. ProtGPT2 improved substantially relative to the April
d=4,096 checkpoint, while the ZymCTRL quick-evaluation dead fraction remained
high. These are unmasked 100-sequence diagnostics rather than a matched,
mask-aware held-out comparison.

Two advertised v2 configuration controls, `init_dec_norm_scale` and
`dec_norm_weight`, appear in the YAML files but are never read by the live
trainer, so the claimed decoder-initialization and norm-penalty fixes were not
implemented. The current evaluator also labels records after `skip=200000` as
held out even though ProtGPT2 v2 was configured for the first 300,000 eligible
UniRef50 sequences; without an executed-run cohort manifest, train/evaluation
overlap cannot be excluded. A retained v2 runtime path says `step_100000`
while loading an 8,192-wide checkpoint, leaving provenance internally
inconsistent.

## 5. Cross-model atlas and checkpoint diagnostic

### 5.1 Canonical atlas procedure

**Canonical scripts:** `15_cross_model_conservation.py`,
`26_universal_atlas_summary.py`, and `27_universal_atlas_null_control.py`  
**Canonical outputs:**
`results/circuit_analysis/cross_model_conservation_3model_balanced200_wide_20260512.json`
and
`results/circuit_analysis/universal_atlas_balanced200_wide_null_control_30x_20260513.json`

Procedure:

1. Use a fixed balanced set of 200 sequences—100 real lysozymes and 100 random
   UniRef50 controls—truncated to 256 residues.
2. Extract token-level sparse activations and mean-pool each feature within
   each sequence.
3. Use anchor layers 5, 12, and 30 for the 36-layer models and relative-depth
   layers 4, 9, and 22 for ProGen2-medium.
4. At each model/layer, retain the 4,096 features with highest variance over
   sequences.
5. Greedily select the top 300 one-to-one pairwise correlation matches.
6. Form an exact triplet only when all three model-pair edges exceed the
   absolute-correlation threshold.
7. For the null, permute sequence assignment and rerun discovery; use 30
   replicates with seed 20260513.

Discovery and evaluation use the same fixed cohort and saved feature pool; no
independent atlas-validation cohort was used.

### 5.2 Canonical result

| Threshold | Observed exact triplets | Null mean | Null SD | Null maximum |
|---|---:|---:|---:|---:|
| absolute r ≥0.90 | 38 | 0.0667 | ≈0.25 | 1 |
| absolute r ≥0.95 | 30 | 0.0667 | ≈0.25 | 1 |
| absolute r ≥0.98 | 8 | 0.0333 | ≈0.18 | 1 |

This establishes recurrence under the exact cohort, layer mapping, variance
filter, pool size, greedy matcher, and null. It does not establish universal
biology, invariance to alternative matching algorithms, or generalization to
other cohorts.

The 38 triplets are distributed across anchor layers as 14 at L5, 13 at L12,
and 11 at L30. The implemented null independently permutes the second
activation matrix for every model pair and layer; it does not apply one
coherent model-wise sequence permutation across all pairwise graphs. Triplet
closure is therefore evaluated across edges created from different random
assignments, which can suppress transitive null triplets more strongly than a
coherent permutation. No formal permutation p-value is stored; with 30
replicates, the smallest ordinary plus-one empirical p-value would be 1/31
≈0.0323. The safe statement is separation from this implemented
pair/layer-specific null.

### 5.3 Cohort and source sensitivity

Using a different 500-sequence UniRef50 cohort with the otherwise wide atlas
produced only 8 exact triplets at the 0.90 threshold and 3 at 0.95, rather than
38 and 30. None of those eight feature identities matched any of the 38
balanced-cohort triplets. A separate
real-lysozyme versus random-UniRef50 source-selectivity analysis classified all
38 balanced-cohort triplets as `weak/source-mixed`. These development results
do not negate the canonical fixed-procedure recurrence, but they show that the
triplet count is cohort-sensitive.

The named `calibration_lysozyme_balanced200_20260511.json` derived cohort is
absent. Its apparent 100 real and 100 random component records remain in
`results/ec_metrics/calibration_lysozyme_20260507/calibration_sequences.json`,
and historical logs describe the missing file as an interleaving of them.
However, the exact derived ordering and hash have not been reconstructed and
validated against the atlas. Byte-identical atlas regeneration therefore
cannot yet be certified.

### 5.4 Candidate checkpoint diagnostic

The same atlas procedure compared the mature ProtGPT2/ZymCTRL references with
their 10,000-step checkpoints while holding the mature ProGen2-medium
checkpoint fixed.

| Condition | Triplets | Mean layer CKA | Mean matched-feature absolute r |
|---|---:|---:|---:|
| Mature pair + mature ProGen2-medium | 38 | 0.5559 | 0.8259 |
| ProtGPT2/ZymCTRL at 10k + mature ProGen2-medium | 16 | 0.4504 | 0.6403 |

This is one selected checkpoint pair, one cohort, and one matching
configuration. It demonstrates sensitivity to training stage and motivates a
candidate diagnostic. It does not validate prediction of dictionary quality
across seeds, full trajectories, models, or independent cohorts.

## 6. Semantic, resource, and low-level characterization

### 6.1 Cheap-label annotation and resource coverage

An initial 500-sequence UniRef50 study took the 100 highest-firing positions
per triplet and tested amino-acid identity, coarse chemistry, sequence source,
and position bins. All 38 triplets had some weak simple-label enrichment, but
best MI ranged from 0.000116 to 0.001924 nats with median 0.000688. This did not
support named biological primitives.

Resource coverage was also weak:

| Cohort | Events | Unique accessions | Accessions with Pfam | Accessions with Swiss-Prot | Accessions with AlphaFold |
|---|---:|---:|---:|---:|---:|
| Broad UniRef500 set | 3,800 | 452 | 0 | 0 | 1 |
| Balanced-200, top 10 triplets | 500 | 33 | 7 | 7 | 0 |

Most Swiss-Prot overlaps in the smaller panel were chain/topology annotations,
not strong functional labels. At event level, the balanced panel had only 8
Pfam-overlapping and 165 Swiss-Prot-overlapping top positions. Neither resource
audit covers all claims implied by the early “universal primitive” terminology.

### 6.2 Original Swiss-Prot gate and July repair

The original analysis sampled 500 Swiss-Prot proteins of length 100–400 by
round-robin dominant-Pfam labels, yielding 122,671 positions. It selected the
top 100 positions for each of 38 triplets and computed plug-in MI against ten
label families. Its gate demanded MI ≥0.1 nats.

That gate is mathematically invalid. With event prevalence 100/122,671, the
event entropy and maximum possible MI are 0.00661255 nats; 0.1 nats is 15.12
times the theoretical maximum. The saved `0/38 FAIL` must not be interpreted
as evidence of absent biological alignment.

The July CPU-only repair reproduced all 3,800 retained top-event rows and their
ten label fields—38,000 label-field comparisons—and saved MI values to maximum
absolute error 4.99×10^-9 nats. For 38 triplets × ten label
families it used 2,000 permutations, plus-one p-values, and BH correction over
380 tests:

| Null | Sampling rule | BH q<0.05 |
|---|---|---:|
| Global | uniformly sample positions | 224 / 380 |
| Matched | within protein, matched amino acid and fine position/edge strata | 0 / 380 |

The largest matched excess normalized MI was 0.02180 for T006
`swiss_feature_type`, raw p=0.000500 and q=0.09495. The cohort has 500 distinct
dominant-Pfam labels for 500 proteins, so dominant-Pfam MI cannot demonstrate
replicated family generalization.

This is a post-audit exploratory repair of already-selected top events. It
lacks the continuous activation distribution and cannot establish absence of
all biological association. Its deterministic rerun reproduced both numeric
tables byte-for-byte; the canonical hashes are recorded in Section 12.

The original script-33 bootstrap was not a valid null test: it resampled
positions independently, capped each replicate at 20,000 positions, did not
hold the top-event count fixed, and ignored within-protein dependence.

### 6.3 Dimension-unmatched basis probes

The 38-dimensional triplet vector was compared with 2,560-dimensional
mean-pooled ESM-2 embeddings:

| Task | Triplet basis | ESM-2 | n |
|---|---:|---:|---:|
| Pfam-family macro-F1 | 0.3995 | 1.0000 | 240 |
| EC top-class macro-F1 | 0.2550 | 0.7080 | 280 |
| Secondary-fraction R² | −0.6049 | 0.3245 | 300 |

This quick probe shows that the selected 38-dimensional basis underperforms the
chosen dense comparator. It is not dimension-matched and is not a semantic
null test. Despite later family-aware wording, the live script uses record-level
`StratifiedKFold` for classification and record-level `KFold` for the
secondary-fraction regression; the triplet and ESM-2 arms also use different
random seeds and folds and report no confidence intervals.

### 6.4 Triplet characterization and synthesis

The final characterization cohort combined 500 Swiss-Prot proteins with 200
calibration sequences, totalling 153,628 retained residue positions. Per-model
activations were standardized across retained positions, aligned, and averaged
across the three matched features. The 100
highest consensus positions per triplet defined the saved events.

Five implemented tests evaluated centred 3/5-mer enrichment, normalized
position, ProtGPT2 BPE boundaries, same-layer received attention, and CLT-input
L2 norm. Each used 2,000 resampling draws under its implemented null—the
position test simulated Uniform(0,1) references, while the other tests used
label/value permutations—and BH adjustment over 38 triplets × five families.

| Association or synthesis category | Count |
|---|---:|
| k-mer significant | 27 / 38 |
| normalized-position significant | 35 / 38 |
| high CLT-input-norm significant | 25 / 38 |
| unnormalized received-attention significant | 4 / 38 |
| ProtGPT2 BPE-boundary significant | 1 / 38 |
| at least one implemented test | 37 / 38 |
| three or more implemented tests | 21 / 38 |

Post hoc overlapping clusters were: 17 k-mer+position+norm, 6 position+norm, 6
k-mer-dominant mixed, 4 position-only, 4 received-attention, and 1
unclassified.

The tests generally permute residue positions without preserving protein
identity; the more significant of 3-mer and 5-mer was selected without a
second correction. These are overlapping exploratory associations, not
mutually exclusive biological concepts or evidence that one covariate explains
another.

### 6.5 N-terminal initiator-methionine subset

| Triplet | Received-attention r | Top events in first 2 / first 5 | Context starts M | Top 3-mers | Role |
|---|---:|---:|---:|---|---|
| T011 | 0.921 | 100/100 / 100/100 | 0.99 | MKA, MKI, MTA | initiator-M readout |
| T018 | 0.914 | 100/100 / 100/100 | 0.95 | MKK, MKI, MKA | initiator-M readout |
| T023 | 0.885 | 100/100 / 100/100 | 0.89 | MRI, MRS, MRA | initiator-M readout |
| T025 | 0.079 | 0/100 / 0/100 | 0.02 | HNC, NHI, IHN | non-N-terminal comparator; only 11 unique sequences |

Each of T011/T018/T023 represents 100 distinct proteins, not repeated positions
from a few proteins. In the event-level context table, each has first-two
fraction 1.0 versus 202/3,700 = 0.0546 in the pooled background; the best BH
q-value is 2.72×10^-117. Although each target row comes from a distinct protein,
the Fisher analysis treats pooled background event rows as independent despite
repeated proteins and cross-triplet dependence. The top events were selected
from the same evidence, and causal opportunity is not normalized. Early key positions have more future queries
that can attend to them. The correct label is therefore **N-terminal
initiator-methionine sparse readout with high unnormalized received attention**,
not “causal attention-sink feature.”

## 7. Attention-output pilot and causal intervention procedures

### 7.1 Attention-output sparse dictionary pilot

**Script/output:** `scripts/43_attention_output_transcoder_pilot.py` and
`results/circuit_analysis/attention_output_transcoder_pilot_20260514_l23/`

The pilot trained a d=2,048, k=32 sparse dictionary for 3,000 steps on the
input to ZymCTRL layer 23's attention output projection, using 5,000 EC-labelled
records. It screened 2,048 features on 120 sequences and ablated the three
highest-ranked N-terminal features on 80 sequences against three random
feature sets.

| Diagnostic | Result |
|---|---:|
| Evaluation FVU | 0.4782 |
| Evaluation alive fraction | 0.08691 |
| Top feature | 1671 |
| Maximum received-attention correlation | 0.98145 |
| Maximum first-two activation delta | 8.0173 |
| Target ΔNLL, all positions | −0.00202 |
| Target ΔNLL, first two | −0.35771 |
| Target ΔNLL, positions 2–10 | −0.05081 |
| Random-set first-two ΔNLL values | +0.01580 / 0 / 0 |

The sparse readout criterion was descriptive-positive, but the target ablation
changed NLL in the opposite direction and did not support causality; the script
has no machine-evaluated causal-ablation gate. The two exact-zero random sets
also indicate inactive random features and weak, non-activity-matched controls.
This is a single-model, single-layer,
in-sample pilot. Training sampled all 5,000 records while evaluation/ablation
reused the first 120/80; there was no held-out split. Feature 1671 is a selected
maximum without a multiplicity-adjusted null. At step 3,000, 1,870 features
were resampled immediately before saving, so the training “alive=1” bookkeeping
does not describe the saved evaluation state. The evaluation alive fraction of
8.69% is the relevant observed diagnostic. The saved summaries are plain
means, not 1,000-bootstrap interval summaries.

### 7.2 TopK-aware CLT feature patch

**Script/output:** `scripts/40_attention_sink_causal_ablation.py` and
`results/circuit_analysis/attention_sink_causal_ablation_20260517/`

The intervention set the encoded component multiplier to zero, recomputed the
TopK code, and patched the attributed same-layer MLP-output contribution. It
tested T011/T018/T023 in each model on 200 Swiss-Prot proteins, with T025 and
two same-layer random features per target as controls. This is subtraction of
a CLT decoder contribution, not literal deletion of a neuron or the complete
model state.

| Model | Target T011/T018/T023 mean ΔNLL at positions 2–10 | Gate hits |
|---|---|---:|
| ProGen2-medium | −0.000041 / +0.000193 / +0.000327 | 0/3 |
| ProtGPT2 | +0.000164 / +0.000045 / +0.000082 | 0/3 |
| ZymCTRL | −0.04876 / +0.002956 / −0.04047 | 0/3 |

All nine model-triplet gates failed across 6,000 per-sequence rows. The tested
features were active at the N-terminal edge, but expected-direction likelihood
damage and attention redistribution were not consistent.

### 7.3 Direct single-head ablation

**Script/output:** `scripts/41_attention_head_sink_ablation.py` and
`results/circuit_analysis/attention_head_sink_ablation_20260518/`

Heads were selected on the first 100 proteins by unnormalized first-two
attention mass plus `max(correlation, 0)`; negative correlations were clipped
to zero, and some selected triplet–head pairs still had negative correlations. A contiguous
head-output slice was then zeroed before the attention output projection and
evaluated on the 200-protein cohort with one random head per target and a sham.

| Model | Selected first-two mass | Target mean ΔNLL positions 2–10 | Mean target-feature drop | Result |
|---|---:|---:|---:|---|
| ProGen2-medium | ≈0.943 | ≈−0.00108 | zero or negative | fail |
| ProtGPT2 | 0.954–0.966 | ≈+0.00044 to +0.00048 | zero or negative | fail |
| ZymCTRL | ≈0.896 | +0.00714 | zero or negative | fail |

All gates failed over 4,800 rows. High received-attention mass did not translate
into the specified feature damage or N-terminal likelihood effect.

### 7.4 Top-8 and top-32 head-set ablations

The top-8 experiment used three same-layer random sets per model; the top-32
experiment used only one. Each model had 200 evaluation proteins.

| Test | Rows | Largest/diagnostic target result | Control comparison | Strict / exploratory gate |
|---|---:|---|---|---|
| Top-8 set | 3,000 | ZymCTRL positions 2–10 ΔNLL +0.04065; feature drop −0.00352 | random mean ΔNLL −0.02188 | fail / fail |
| Top-32 set | 1,800 | ZymCTRL target ΔNLL +0.01614; feature drop −0.01050 | random ΔNLL +0.16524 | fail / fail |

The top-32 random control exceeding the targeted effect is a particularly clear
specificity failure. Across all intervention families, discovery and evaluation
were not fully independent. Random controls were layer-matched but not matched
on activation magnitude or received-attention mass, and the head-discovery
first 100 proteins overlap the 200-protein intervention cohort.

### 7.5 Causal conclusion

The feature-patch, single-head, and head-set analyses failed their implemented
gates. The attention-output pilot passed its descriptive readout gate but
produced an opposite-signed target-ablation effect and had no encoded causal
gate. The calibrated conclusion is limited to these features, heads, sites,
strengths, cohorts, and controls. These experiments do
not establish a causal attention sink, but neither do they exclude every
distributed, alternative-site, redundant, or nonlinear mechanism.

## 8. Steering, generated-sequence, and external-metric procedures

### 8.1 Hook sanity and artifact provenance

The April diagnostic selected an active feature, attached the same MLP-output
hook family used for steering, and measured teacher-forced logit changes.

| Diagnostic | Result |
|---|---:|
| ZymCTRL multiplier 1 versus no hook | identical |
| ZymCTRL multiplier 10 maximum logit shift | 12.93 |
| ZymCTRL multiplier 0 maximum logit shift | 0.885 |
| ProGen2-medium multiplier 10 maximum shift | 3.938 |
| ProGen2-medium multiplier 0 maximum shift | 1.250 |

The saved `ec_features.pkl` and ZymCTRL v2 checkpoint both have 36 layers and
d=8,192. Plumbing can change logits; earlier zero effects are not explained by
a universally disconnected hook.

### 8.2 Direct-effect feature construction

`scripts/16_direct_effect_features.py` used the ZymCTRL v2 checkpoint, eight EC
prompts, and one identical default reference sequence per prompt. A
teacher-forced backward pass ranked features by activation × likelihood
gradient × CLT decoder direction, summed across the decoder window. It saved
the top ten rows by absolute effect for each of 8 classes × 36 layers, producing
shape `(8, 36, 10)`.

This generates candidates, not a steering result. It uses one reference
sequence per class and has no stability analysis. Absolute-effect ranking
retains both positive and negative directions.

### 8.3 TopK-aware eight-target steering and feature-sign defect

**Procedure.** The hook modified five selected feature preactivations at each
of layers 3, 12, and 30 by multiplier 2.5, reapplied TopK, and replaced the
CLT-explained MLP component. For each enzyme-labelled prompt it generated 100
unsteered and 100 steered sequences. The current script/defaults and execution
runner reconstruct temperature 0.8, top-k 50, up to 200 new tokens, and
separate seed bases; the canonical result JSON does not retain those fields.
The score was an implemented/pre-existing class-specific motif/composition
heuristic, not a trained or externally validated EC classifier. Confidence
intervals used 10,000 independent bootstrap replicates and p-values used 10,000
two-sided label permutations.

| Target | Unsteered | Steered | Difference | 95% CI | p |
|---|---:|---:|---:|---|---:|
| Lysozyme | 0.890 | 0.894 | +0.004 | [−0.040, +0.047] | 0.8678 |
| Trypsin | 0.723 | 0.737 | +0.013 | [−0.047, +0.073] | 0.6743 |
| Alcohol dehydrogenase | 0.650 | 0.669 | +0.019 | [−0.060, +0.098] | 0.6373 |
| Catalase | 0.981 | 0.970 | −0.011 | [−0.023, +0.001] | 0.0904 |
| DNA polymerase | 0.967 | 0.959 | −0.008 | [−0.033, +0.019] | 0.5693 |
| Lipase | 0.769 | 0.826 | +0.057 | [−0.007, +0.122] | 0.0913 |
| Kinase | 0.510 | 0.617 | +0.107 | [+0.000, +0.210] | 0.0532 |
| Carbonic anhydrase | 0.900 | 0.855 | −0.045 | [−0.106, +0.014] | 0.1439 |

No raw comparison met the gate. However, the live selector uses five positive
features only when at least five positive rows occur among the ten
absolute-effect-ranked candidates; otherwise it falls back to the first five
absolute-ranked rows, including negative effects. Of 120 interventions, 19
were negative-attribution features. Six targets were affected in at least one
layer, and catalase L3 amplified five negative-effect features.

Therefore, the exact defensible statement is **no positive evidence under the
implemented feature-selection and heuristic-scoring pipeline**. This is not a
clean eight-class test of amplifying five positive direct-effect features. The
selector must be corrected and the run repeated before claiming a validated
eight-class direct-effect negative. The direct-effect summary is also absent
from the manuscript source-data package, so reviewers cannot audit the sign
problem from that package alone.

The saved benchmark retains only three example sequences per arm and omits
temperature, token limit, and seed bases. Its inferential statistics cannot be
recomputed from the result JSON alone.

### 8.4 Earlier steering runs

- The April v2 z-score-selected eight-class run used 200 sequences per arm,
  multiplier 2.5, and produced 0/8 positive classes.
- A three-target multiplier-5 sweep used 64 per arm and produced lipase
  +0.0717 (p=0.1066), kinase +0.0260 (p=0.6584), and carbonic anhydrase −0.0470
  (p=0.2077): 0/3 positive.
- The oldest `zymctrl_purity.json` contains all-zero scores caused by the
  pre-fix prompt/string path and is unusable legacy provenance.

These runs use different feature selectors and intervention implementations;
they are not interchangeable replications.

### 8.5 Separate lysozyme generation and lead selection

The external lysozyme case is a **different, earlier experiment** from the May
3 direct-effect/TopK-aware run. On 29 April it used z-score-selected features,
six per layer at L3/L12/L30, multiplier 2.5, maximum 400 tokens, and temperature
0.8 to generate 200 steered plus 200 independently seeded unsteered sequences.
The top ten steered leads were ranked 60% by length/E-before-D/aromatic rules
and 40% by model log likelihood.

Steered sequences had mean length 214.0 (range 89–400), and 97.5% passed the
permissive E-before-D proxy. There was no equivalently rank-selected unsteered
lead set. The lead filter uses motif-like criteria related to the downstream
question, and the output omits checkpoint/feature-source paths, temperature,
seeds, and token limit.

Crucially, Pfam, CLEAN, ESMFold, and Foldseek evaluate this older z-score/
off-manifold generation, **not** the later direct-effect benchmark. There is no
generation-wide external-metric evaluation of the May 3 intervention.

### 8.6 Generated-sequence external metrics

| Panel | Sequence n | Structure n | Pfam | CLEAN exact | CLEAN prefix | Foldseek mean top TM |
|---|---:|---:|---:|---:|---:|---:|
| Selected steered leads | 10 | 10 | 0.900 | 0.900 | 0.900 | 0.8833 |
| All steered | 200 | 0 | 0.860 | 0.775 | 0.865 | unavailable |
| All unsteered | 200 | 20 | 0.820 | 0.775 | 0.875 | 0.9384 |

Generation-wide tests were:

| Endpoint | Counts/difference | One-sided Fisher p |
|---|---|---:|
| Pfam lysozyme-like | 172/200 vs 164/200; +0.040 | 0.16989 |
| CLEAN exact | 155/200 vs 155/200; 0.000 | 0.54763 |
| CLEAN prefix | 173/200 vs 175/200; −0.010 | 0.67205 |

The selected steered structures had mean pLDDT 67.716 with 60% above 70; the
first 20 unsteered structures had mean 73.191 with 70% above 70. This is a
post-selected ten-versus-first-twenty descriptive comparison, not randomized or
symmetric, and no formal structure comparison test was run. Foldseek measures
nearest similarity to the staged PDB100 database, not lysozyme-specific
correctness. The broad Pfam regex includes generic `glyco` and `hydrolase`
terms. The generation-wide sequence endpoints and bounded structural subset do
not favour steering over prompt conditioning.

### 8.7 Real-lysozyme versus random-protein metric control

The retrospective control used 100 EC 3.2.1.17 Swiss-Prot/ZymCTRL sequences and
100 random UniRef50 proteins, seed 7, restricted to length 80–250. Mean lengths
were 154.44 and 155.13, but records were not individually length-matched.

| Metric | Real mean | Random mean | Effect size d | p |
|---|---:|---:|---:|---:|
| Pfam lysozyme-like | 0.910 | 0.000 | 4.4969 | 4.71×10^-47 |
| CLEAN exact 3.2.1.17 | 0.960 | 0.000 | 6.9282 | 5.08×10^-53 |
| CLEAN 3.2.1.x prefix | 0.990 | 0.050 | 5.5486 | 9.75×10^-50 |
| ESMFold mean pLDDT | 79.740 | 58.478 | 1.7858 | 5.90×10^-22 |
| ESMFold confident fraction | 0.865 | 0.336 | 2.0632 | 4.45×10^-23 |
| Foldseek top TM | 0.971 | 0.653 | 1.5927 | 2.75×10^-29 |

The generated metric triad completed before the calibration summary on 7 May,
so the manuscript statement that real/random proteins were evaluated “first”
is chronologically inaccurate. This retrospective control shows responsiveness
to obvious positives versus broad random proteins; it does not establish hard-
negative specificity, a decision threshold, generalization to other EC
classes, independence from reference/training sets, or steering efficacy.

For the binary Pfam/CLEAN rows, the reported `d` uses the project's custom
pooled-Bernoulli standardization; only the continuous ESMFold/Foldseek rows use
the conventional pooled-variance Cohen's d. Reproducibility metadata is also
internally inconsistent: the saved calibration manifest records
`max_random_candidates: null` and an eligible pool of 28,850,507, whereas the
runner documents `--max-random-candidates 50000`. The executed sampling path
must be resolved.

### 8.8 Site-mismatched mean-direction experiment

The recoverability pipeline's script 46 derived an eight-class mean-difference
direction from cached ZymCTRL layer-3 architecture-specific CLT-input tensors,
then added that direction at the MLP output. With alpha 8, 40 sequences per arm,
and the heuristic score it passed 0/8 gates.

This coordinate/site mismatch makes the artifact's `distributed_or_robust`
verdict mechanistically uninterpretable. It is not a valid oracle
controllability test and cannot show that computation is distributed or robust.

## 9. Representation-recoverability and wider-dictionary procedures

### 9.1 Protocol, cohort, and corrected analysis

The recoverability protocol was frozen on 4 June 2026, then amended after
validity problems were identified. The current corrected analysis used:

- 820 Swiss-Prot proteins of length 100–400;
- 280 EC-labelled proteins (40 × seven top classes);
- 240 proteins (12 × 20 dominant-Pfam labels);
- 300 proteins for secondary-structure fractions;
- 44,626 annotated residues from 319 proteins for residue structure; and
- only 48 ZymCTRL decoder-native sequences, six per prompt, reconstructed from
  three saved steered and three saved unsteered examples per class.

It cached architecture-specific CLT inputs (`C`), same-layer sparse codes
(`F`), reconstructions, composition baselines, and ESM-2. Protein arrays were
mean-pooled; token values were mapped to residues for the residue task.

The amended pipeline used fold-internal `StandardScaler → PCA(max 256) →`
L2-logistic/ridge probes. EC top class used fivefold Pfam-family-disjoint CV
plus a report-only stratified variant; Pfam necessarily used stratified CV;
residue structure used protein-grouped CV. Skill was chance-corrected macro-F1
or R². `C` was the best-layer CLT-input skill, `F` the sparse-code skill at that
same layer, and recovery was `rho=clip(F/C, 0, 1)`. CIs used 1,000 group
bootstraps of fixed out-of-fold predictions.

### 9.2 Corrected reference-dictionary results

| Model | Task | C | F | rho |
|---|---|---:|---:|---:|
| ProtGPT2 | EC family-disjoint | 0.260 | 0.262 | 1.000 |
|  | EC stratified, report-only | 0.557 | 0.547 | 0.982 |
|  | Pfam | 0.939 | 0.808 | 0.861 |
|  | residue structure | 0.221 | 0.217 | 0.981 |
| ZymCTRL | EC family-disjoint | 0.123 | 0.022 | 0.180 |
|  | EC stratified, report-only | 0.356 | 0.229 | 0.643 |
|  | Pfam | 0.902 | 0.768 | 0.851 |
|  | residue structure | 0.149 | 0.129 | 0.866 |
|  | decoder-native EC | 0.652 | 0.525 | 0.806 |
| ProGen2-medium | EC family-disjoint | 0.272 | 0.153 | 0.561 |
|  | EC stratified, report-only | 0.594 | 0.497 | 0.836 |
|  | Pfam | 0.952 | 0.910 | 0.956 |
|  | residue structure | 0.256 | 0.208 | 0.812 |

The protein secondary-fraction floor remained numerically invalid even after
PCA: ProtGPT2 C/F=0.621/−3.155; ZymCTRL 0.397/−184.669; ProGen2-medium
0.608/−917.465. It is report-only and excluded from decision gates.

The corrected decision was `NO-GO`: ProtGPT2 was rich and near-faithful on
gated tasks; ZymCTRL was rich and near-faithful on residue structure and the
small decoder-EC set; ProGen2-medium had no two-task bottleneck but was not
uniformly near-faithful because family-disjoint EC rho was 0.561.

### 9.3 Why the recoverability audit remains exploratory

The current results are useful selected linear-probe diagnostics, but they are
not a confirmatory substrate/dictionary attribution:

- “raw residual” in the frozen protocol is wrong; C is the architecture-
  specific tensor actually input to each CLT.
- The original v1 decision was invalidated by an information-preserving random
  up-projection, ≈99% EC/Pfam family confounding, and unstable secondary-
  fraction regression.
- Only one analysis seed was run; the preregistered five seed repetitions were
  not completed.
- The preregistered multiplicity analysis across layers was not completed.
- Best-layer selection and reported scoring use the same cross-validated
  cohort; there is no nested layer-selection evaluation.
- The implementation compares separate CI endpoints rather than the requested
  paired bootstrap skill differences for one richness route.
- Decoder-native EC uses only 48 prompt-correlated saved examples rather than
  the intended full generated cohort.
- The structure/contact stretch task was not run.
- The site-mismatched direction experiment cannot support a controllability
  verdict.

The supported conclusion is simply that selected collective linear signals are
retained to varying degrees from each CLT's own input tensor.

### 9.4 Exploratory wider dictionaries

Despite the preregistered no-go, an explicitly exploratory override trained all
three models sequentially on two H200 GPUs each at d=16,384, k=128, window=8,
300,000 steps, batch 1/GPU, learning rate 3×10^-4, resampling every 2,500 steps,
and a 5,000-step dead threshold. ProtGPT2/ProGen2 used UniRef50 and ZymCTRL used
the EC-labelled FASTA. All inherited the padding-mask defect.

| Model | Width change | Final training-log FVU | Dead | L0 | Loss | Preregistered quality gate |
|---|---:|---:|---:|---:|---:|---|
| ProtGPT2 | 2× | 0.2782 | 0.334 | 128.0 | 7.1176 | fails FVU and dead criteria |
| ZymCTRL | 2× | 0.3348 | 0.044 | 90.6 | 0.0201 | fails FVU criterion |
| ProGen2-medium | 4× | 0.3101 | 0.122 | 123.0 | 0.0138 | fails FVU criterion |

Selected changes were mixed:

| Model/task | Reference F / rho | Wider F / rho | Direction |
|---|---:|---:|---|
| ProtGPT2 EC | 0.262 / 1.000 | 0.204 / 0.785 | worse |
| ProtGPT2 Pfam | 0.808 / 0.861 | 0.789 / 0.840 | worse |
| ZymCTRL decoder EC | 0.525 / 0.806 | 0.623 / 0.955 | improved |
| ZymCTRL Pfam | 0.768 / 0.851 | 0.719 / 0.796 | worse |
| ProGen2-medium EC | 0.153 / 0.561 | 0.228 / 0.837 | improved |
| ProGen2-medium Pfam | 0.910 / 0.956 | 0.931 / 0.978 | improved |

None met the preregistered FVU <0.15 requirement for a demonstrably better
dictionary; ProtGPT2 also missed dead fraction <0.30. The logged quality values
are final training values rather than a matched held-out comparison, only
ProGen2 received a fourfold width increase, and the atlas/semantic/intervention
analyses were not rerun on a quality-gated wider dictionary.

The 12 June amendment's statement that capacity was “falsified by
demonstration” is explicitly superseded by the 16 July correction. The current
result is a mixed exploratory width comparison; it does not isolate model
substrate, dictionary capacity, optimization, feature geometry, or hook-site
mismatch.

## 10. Chronological procedure-to-result register

### 10.1 Numbered experiment log

| ID | Date | Procedure | Recorded result | Current status / canonical artifact |
|---|---|---|---|---|
| EXP-R2-001 | 2026-04-02 | Initial d=4,096 CLTs, no resampling | FVU ≈0.30–0.35; dead 29.2–91.9% | Developmental failure analysis; H200 logs/configs |
| EXP-R2-002 | 2026-04-05 | Resampling-rerun evaluation | FVU 0.328/0.380/0.333; dead 81.9/58.3/57.4% | `results/checkpoint_evaluation/rerun_evaluation.json` |
| EXP-R2-003 | 2026-04-05 | ZymCTRL EC feature/circuit analysis | 6,214 one-class and 1,365 eight-class active features; steering insufficient | Historical; biological interpretation superseded |
| EXP-R2-004 | 2026-04-05 | Layer-wise EC specificity | raw L35 L2=564.51 | Historical; L35 unusable due ≈90% dead CLT |
| EXP-R2-005 | 2026-04-13 | Layer quality map | usable 11/36, 23/36, 20/27 | `results/checkpoint_evaluation/layer_quality_map.json` |
| EXP-R2-006 | 2026-04-13 | Architecture reconciliation | usable ZymCTRL L3/L12/L30 | Descriptive layer-selection correction |
| EXP-R2-007 | 2026-04-29 | Hook sanity and pickle provenance | hooks move logits; artifact dimensions match | `results/diagnostics/` |
| EXP-R2-008 | 2026-05-03 | Direct-effect candidate construction | `(8,36,10)` candidates | Candidate artifact; one reference sequence/class |
| EXP-R2-009 | 2026-05-03 | TopK-aware heuristic steering | 0/8 raw positive gates | Pipeline negative with feature-sign defect |
| EXP-R2-010 | 2026-05-04 | External-metric readiness | tools/databases initially missing | Operational blocker, later partly resolved |
| EXP-R2-011 | 2026-05-04 | Steering viability decision | no-go for strong steering/drug design | Decision remains correct, reasons refined |
| EXP-R2-012 | 2026-05-04 | HMMER/Pfam/Foldseek/CLEAN staging | core tools staged; weights/DB/calibration incomplete | Operational resource record |
| EXP-R2-013 | 2026-05-04 | Pfam scan of older lysozyme run | 0.860 vs 0.820, p=0.1699 | No generation-wide lift |
| EXP-R2-014 | 2026-05-07 | CLEAN/Foldseek generated triad | sequence metrics no lift; selected structure subset | Separate older z-score steering run |
| EXP-R2-015 | 2026-05-07 | Real/random metric response control | effect sizes d=1.59–6.93 | Retrospective metric-responsiveness control |
| EXP-R2-016 | 2026-05-12 | Cheap-label triplet annotation | MI 0.000116–0.001924 | Weak low-level association only |
| EXP-R2-017 | 2026-05-12 | Resource coverage | broad cohort 0 Pfam/0 Swiss/1 AlphaFold accession | Resource-readiness negative |
| EXP-R2-018 | 2026-06-05 | Original recoverability run | no-go | Superseded by validity-corrected v2 |
| EXP-R2-019 | 2026-06-06 | Corrected recoverability reanalysis | no-go; selected signals often retained | Canonical `probes_v2` / `decision_v2b`, exploratory |
| EXP-R2-020 | 2026-06-06–12 | Wider d=16,384 retrain | all models completed 300k | Exploratory override of preregistered no-go |
| EXP-R2-021 | 2026-06-12 | Wider-dictionary probes | mixed changes; corrected no-go | Capacity-falsification interpretation superseded |
| EXP-R2-022 | 2026-07-16 | MI and padding evidence repair | 224/380 global, 0/380 matched; original gate impossible | Canonical repair artifact |
| EXP-R2-023 | 2026-07-16 | npj evidence/package build | 18-page main, 11-page supplement, checksums pass | Operational PASS; human/release blockers remain |
| EXP-R2-024 | 2026-07-16 | Canonical path migration | structure/build validation | Structural only; historical paths preserved |

### 10.2 Major procedures omitted from the numbered R2 log

| Date | Procedure | Result | Canonical path |
|---|---|---|---|
| 2026-05-12/13 | Balanced-200 atlas and 30-replicate null | 38/30/8 observed; null maxima 1 | `results/circuit_analysis/*balanced200_wide*` |
| 2026-05-12 | UniRef500 atlas sensitivity | 8 at 0.90; no identity overlap with canonical 38 | `results/circuit_analysis/*uniref500_wide*` |
| 2026-05-13 | Original Swiss-Prot gate | saved 0/38 under impossible threshold | `results/circuit_analysis/swissprot_triplet_annotation_20260513/` |
| 2026-05-13 | Triplet basis probes | triplet basis below ESM-2 on 3/3 tasks | `results/circuit_analysis/triplet_basis_probes_20260513/` |
| 2026-05-14/15 | Characterization and synthesis | 37/38 with ≥1 low-level test; 21/38 with ≥3 | `results/circuit_analysis/triplet_*20260515_nperm2000/` |
| 2026-05-14 | Attention-output pilot | high selected readout; causal ablation opposite-direction | `results/circuit_analysis/attention_output_transcoder_pilot_20260514_l23/` |
| 2026-05-16 | N-terminal subset/context | three initiator-M readouts; 18/24 event tests | `results/circuit_analysis/attention_sink_*20260516/` |
| 2026-05-16 | One-pair checkpoint sensitivity | mature 38 vs early 16 | `results/circuit_analysis/universal_atlas_quality_diagnostic_20260516/` |
| 2026-05-17 | CLT feature patch | all nine model-triplet gates failed | `results/circuit_analysis/attention_sink_causal_ablation_20260517/` |
| 2026-05-18 | Single-head, top-8, top-32 ablation | all strict/exploratory gates failed | `results/circuit_analysis/attention_*ablation*20260518/` |

The absence of these analyses from `docs/EXPERIMENT_LOG.md` is a provenance
gap. The root project log contains their narrative history, but a project-local
audit should not require reconstructing two separate chronologies.

**Post-audit amendment (2026-07-17):** this chronology gap is closed. The
omitted 12--18 May procedures are now restored in the project-local experiment
log. The table above is retained as the finding that prompted that repair.

## 11. Cross-project integrity and claim audit

### 11.1 Prioritized findings

| Severity | Finding | Consequence |
|---|---|---|
| **Critical** | The exact balanced-200 atlas file and three reference CLTs were recovered and hash-verified after the initial audit, but upstream model revisions, complete execution metadata and a persistent licensed deposit remain absent. | Byte-identical public central-atlas regeneration cannot yet be certified. |
| **Critical** | The original Swiss-Prot 0.1-nat gate exceeds the mathematical maximum by 15.12×; its iid, 20k-capped bootstrap neither fixes event count nor preserves proteins. | Its saved `FAIL` and uncertainty procedure are invalid, not an ordinary negative result. |
| **High** | The atlas null uses separate pair/layer permutations rather than one coherent model-wise assignment. | Separation is only from the implemented stronger/different null. |
| **High** | The 500-UniRef run recovers eight triplets with zero exact identity overlap with the canonical 38. | Feature identities are not independently replicated across cohorts. |
| **High** | The direct-effect steering fallback amplified 19 negative-attribution features in six class/layer cells. | The eight-class run is not a clean positive-direction intervention negative. |
| **High** | External lysozyme metrics evaluate the older z-score/off-manifold generation, not the May 3 direct-effect pipeline. | They cannot externally validate or refute the later intervention. |
| **High** | CLT training/quality metrics include padding and reference quality rows are not matched evaluations. | Dictionary quality and dead/FVU values are biased and not cross-model comparable. |
| **High** | Wider dictionaries missed the preregistered FVU criterion and use unmatched training-log quality. | They do not falsify capacity or optimization. |
| **High** | The attention-output pilot has no held-out split, selects maxima without multiplicity correction, and resamples most features at the final step. | Its positive-readout interpretation is highly exploratory. |
| **High** | The live basis-probe script uses ordinary stratified CV and different folds/seeds across representations despite stronger manuscript wording. | The stated family-aware comparison is not implemented. |
| **Moderate** | Characterization nulls are residue-exchangeable, 3/5-mer selection is uncorrected, and received attention is unnormalized for causal opportunity. | Association q-values do not establish protein-level or causal semantics. |
| **Moderate** | Feature/head discovery overlaps evaluation; random controls are not activation/mass matched. | Negative intervention specificity is bounded to the implemented controls. |
| **Moderate** | Recoverability uses one seed, non-nested layer selection, no completed multiplicity analysis, and a 48-sequence decoder-EC subset. | Results are descriptive, not confirmatory substrate/dictionary attribution. |
| **Moderate** | Several packaged JSON files contain Python `NaN`/`Infinity`. | They are not strict RFC 8259 JSON and may fail public tooling. |
| **Moderate** | Result metadata retains historical `Research2/` or `paper_r2_nature_mi/` paths. | Hash provenance is useful, but direct rerun paths are stale. |
| **Moderate** | `init_dec_norm_scale` and `dec_norm_weight` are declared but unused. | The v2 configuration overstates implemented training controls. |
| **Moderate** | The R2 log omits major May procedures. | Project-local chronological auditability is incomplete. |

### 11.2 Manuscript/package discrepancies

**Post-audit amendment (2026-07-17):** the current manuscript explicitly
discloses the mixed Figure 5 runs, the steering chronology and the
attention-pilot uncertainty limitation. The source-data package now verifies
65 rows, 68 checksums and 1,649,686 bytes. The bullets below are retained as
the dated pre-remediation audit trail rather than current discrepancies.

- Resolved after the assessment: the rebuilt source-data package verifies
  1,649,686 bytes; EXP-R2-023 remains the historical pre-remediation count.
- The Methods say the real/random metric control was evaluated first, but the
  generated triad predates the calibration summary on 7 May.
- Figure 5 panels a–c use legacy 17 April public-API ESMFold structures from an
  older generation/checkpoint run, while panel d uses the 29 April v2 local
  fp32 ESMFold aggregates. The caption calls the images post hoc but does not
  disclose the mixed runs.
- The attention-output pilot's saved summary reports means without the stated
  1,000-bootstrap interval analysis.
- Resolved after the assessment: the complete direct-effect candidate/sign
  summary is now in the reviewer source-data package.
- `main.tex` comments still cite the former project path, and copied evidence
  intentionally preserves several old absolute paths.

The package verifier establishes file/hash consistency, screens all packaged
files for sequence-like values, and applies credential-pattern checks to copied
model-config files. It does not establish strict JSON validity,
statistical correctness, cohort comparability, or agreement between every
LaTeX statement and executable implementation.

### 11.3 Superseded claims and artifacts

Treat the following as historical provenance only:

- L35 as the dominant usable biological EC circuit;
- 1,365 within-ZymCTRL “universal features” as cross-model primitives;
- the original Swiss-Prot `0/38` result as evidence of no biology;
- `attention_sink` labels as causal mechanisms;
- the automatically generated “primitive” names in
  `universal_primitives_uniref500_20260512/interpretation.md` and positive
  “attention-sink family/subtype” wording in the May subset/correlate summaries;
- selected steered leads as validated therapeutic or enzyme designs;
- the site-mismatched mean direction as an oracle/distributed-robustness test;
- wider-dictionary recovery as capacity falsification; and
- “raw residual stream” language for the architecture-specific CLT inputs.

## 12. Reproduction and artifact verification

### 12.1 Safe local verification commands

From repository root on B:

```bash
source /Data/lzp/BioInterpretebility-CC/.venv/bin/activate
PYTHONDONTWRITEBYTECODE=1 python \
  r2_interpretability_transfer/manuscript/source_data/build_source_data.py \
  --verify-only
python r2_interpretability_transfer/scripts/50_make_manuscript_figures.py

source ~/miniconda3/etc/profile.d/conda.sh
conda activate latex
cd r2_interpretability_transfer/manuscript
tectonic --reruns 2 main.tex
tectonic supplementary_information.tex
```

For the deterministic MI repair:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ct
python r2_interpretability_transfer/scripts/49_reaudit_triplet_mi.py \
  --seed 20260716 --n-perm 2000 \
  --out-dir /tmp/r2_mi_reaudit_verify_20260716
```

Script 49 fails unless it reconstructs the expected 500 proteins, 122,671
positions, 38×100 unique top events, all 3,800 labels, and the saved plug-in MI
within tolerance. It refuses to write into the canonical source directory.

The exact historical 200-sequence file was subsequently recovered from the
archived path named by the atlas provenance and hash-verified. Its ordered
records equal the prior reconstruction before added provenance fields. The
three exact reference CLTs were also located and hash-verified. Public
byte-identical regeneration remains blocked by immutable upstream revisions,
complete execution metadata and deposition of these recovered inputs—not by
cohort ordering or local checkpoint location.

### 12.2 Audit-time build state

- Source-data verifier: 58 manifest rows, 61 checksums, 1,412,592 bytes.
- Main Article: 18 A4 pages, six figures.
- Supplement: 11 A4 pages, exactly ten tables.
- Current logs contain no unresolved-reference or overfull-box matches.
- Scientific result artifacts were not rewritten during this documentation
  audit.

### 12.3 Key artifact hashes

| Artifact | SHA-256 |
|---|---|
| MI reanalysis table | `c144bfb16123bc774aaf5a22b291bba113ea0d5f2058a0f05e7fb11a39941556` |
| MI triplet summary | `96296e31170492f88b32ca709f54f6039a153e1a17fe66bc3827ca4d3ac94784` |
| Direct-effect steering JSON | `92362ec20dcf1090183e31c9bfdbaf664d2dad07a4c74c19c83088a79065c1bf` |
| Direct-effect candidate summary | `ef247bccbed60490ea35f5ca59c146c6e09d64ba7740ac521e3410dc98b265ca` |
| Older lysozyme v2 generations | `b2b262e9f519633716be902a25e0e338b9326379a20468ebdf3c5988ae8a5b53` |
| Generated metric triad | `bcfb7be4c9fec4222bf36f64c3c6e37facb32b97c1a5670e1d5bcdceefc0930f` |
| Real/random metric control | `1c5413039ffc610940efd44e64e0fb3ff994bae8a23b26c79cd4d4351b5e116a` |

Before public deposit, non-finite JSON constants should be replaced with
`null` plus explicit status/missingness fields, while retaining hashes of the
historical originals as provenance.

## 13. Audit disposition and required remediation

### 13.1 Present disposition

R2 is accepted as an **evidence-corrected internal audit of procedure-specific
sparse readouts and bounded intervention negatives**. It is not accepted as:

- an exactly reproducible public release;
- a biological-primitive dictionary;
- a causal attention-sink mechanism study;
- a successful or cleanly refuted EC-steering study;
- a demonstration that dictionary capacity/optimization is irrelevant; or
- a validated prospective checkpoint-quality benchmark.

The shortest defensible synthesis is:

> Under one fixed, cohort-sensitive matching procedure, three protein
> generators yielded 38 sparse-readout triplets that separated from the
> implemented pair/layer-specific assignment null. The triplets showed
> overlapping low-level associations, including three initiator-methionine
> readouts with high unnormalized received attention. In a post-audit
> reanalysis, no rich-label association survived BH correction under the
> specified within-protein amino-acid/position-matched null. The implemented
> feature/head gates failed or showed no expected-direction effect. The steering pipeline supplied no positive evidence but requires
> a corrected feature-sign rerun for a clean eight-target negative. Selected
> linear signals were retained by sparse codes to varying degrees; exploratory
> wider dictionaries did not meet the preregistered quality criterion.

### 13.2 Remediation order

1. Deposit the recovered exact 200-sequence cohort and three hash-verified
   reference CLTs with executed configs; recover or explicitly mark unavailable
   upstream model revisions, and publish complete cohort/execution manifests.
2. Recompute the atlas with a coherent model-wise permutation null, more
   replicates, multiple matching seeds/algorithms, and an independent cohort.
3. Make CLT training/evaluation mask-aware and rerun matched quality evaluation;
   retraining is required to remove padding gradients.
4. Remove or implement the unused v2 configuration fields and record exact
   checkpoint/config hashes.
5. Correct the steering selector to require positive direct effects, package
   the attribution table, and rerun all eight targets with immutable full
   sequence/statistic outputs and a validated biological endpoint.
6. State explicitly that the external lysozyme panel is a separate earlier
   intervention; if needed, evaluate the corrected direct-effect generations
   with Pfam/CLEAN and symmetric structure sampling.
7. Resolve the calibration manifest/runner sampling mismatch and record exact
   Pfam, CLEAN, ESMFold, Foldseek, and reference-database revisions.
8. Repeat semantic tests with protein-blocked nulls, pre-specified k-mer tests,
   normalized causal-opportunity controls, and held-out discovery/evaluation.
9. Re-run the basis comparison on identical folds with dimension controls and
   confidence intervals, or remove family-aware wording.
10. Treat the attention-output pilot as developmental until it has held-out data,
   no end-of-run mass resampling, and multiplicity-aware selection.
11. If recoverability remains in the paper, complete seed repetition, nested
    layer selection, paired-difference bootstrap, multiplicity handling, and a
    full decoder-native cohort.
12. Normalize non-finite JSON, current paths, statistical units, and run
    manifests; add the missing May procedures to `EXPERIMENT_LOG.md`.
13. Correct the manuscript chronology, Figure 5 mixed-run disclosure,
    intervention bootstrap wording, and source-data byte count.
14. Finish verified author/affiliation/funding metadata and create a versioned,
    DOI-backed code/data/checkpoint release before submission.

Until these items are resolved, this file is the authoritative human-readable
procedure/result audit layer over the live R2 project.
