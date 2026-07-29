# R0 Experimental Procedures and Results Audit

**Audit date:** 2026-07-16  
**Scope:** `r0_shared_interpretability_framework/` and every upstream artifact
represented in its current v0 evidence tables  
**Audited release directory:** `results/v0_20260515/`  
**Status:** internal evidence ledger; **not an audit-ready benchmark release or
leaderboard**

## 1. Executive finding

R0 does not contain an independent model-training or model-evaluation study.
It contains three executable procedures:

1. an IndelMissense coordinate/resource preflight;
2. a mechanism-feature position-mapping preflight; and
3. a compiler that projects saved R1 encoder and R2 decoder results into a
   shared table schema.

The two preflights perform descriptive calculations. The compiler performs no
new model inference, intervention, resampling, or hypothesis test. Therefore,
the 127 rows in the current result ledgers are **75 standardized encoder
evidence rows plus 52 standardized decoder evidence rows**, not 127 R0
experiments.

| Audit question | Finding |
|---|---|
| Are the R0-native preflights reproducible from their current inputs? | Yes. Both deterministic JSON payloads reproduced from the fixed inputs. |
| Does the builder reproduce the current stored table content? | Yes. All 75 encoder rows and 52 decoder rows match the current builder output. Timestamped summaries are necessarily different on each run. |
| Are the compiled values all faithful to their upstream evidence? | No. Three decoder steering-comparison rows are false because of source-schema drift; one encoder VAMP-like row inherits a score-orientation error; several sample sizes and labels are inaccurate. |
| Is the ledger current through the July 2026 evidence audit? | No. It still registers the invalid May Swiss-Prot MI gate and omits its July matched-null re-audit. |
| Can R0 be treated as a final benchmark or leaderboard? | No. It lacks validated referential integrity, controlled gate semantics, immutable provenance, complete uncertainty fields, and an R0 experiment log. |
| What is defensible now? | R0 is useful as an internal index of procedures, descriptive preflights, inherited positive readouts, and calibrated negative results, provided the corrections in this document are applied during interpretation. |

The most important audit conclusion is that **table reproducibility is not the
same as evidential correctness**. The deterministic row content can be
regenerated consistently while still carrying stale, malformed, or over-broad
claims from upstream artifacts and hard-coded compiler logic.

## 2. Evidence boundary and artifact flow

R0 is a shared framework between the encoder benchmark (R1) and decoder audit
(R2). The current implementation is an evidence ledger rather than the fuller
benchmark architecture proposed in
[`BENCHMARK_SCHEMA.md`](BENCHMARK_SCHEMA.md).

```text
R1 saved JSON results ─────────────┐
                                   │
IndelMissense records.tsv ──> R0 Indel resource preflight ──┐
                                                            │
R1 mechanism audit + Swiss-Prot ──> R0 position preflight ──┼─>
                                                            │
R2 saved JSON results ──────────────────────────────────────┘
                         build_v0_benchmark_tables.py
                                      │
             ┌────────────────────────┼───────────────────────┐
             │                        │                       │
       encoder tables           decoder tables       registry + summary
```

### 2.1 Live R0 inventory

| Class | Live contents | Audit interpretation |
|---|---|---|
| Documentation | `README.md`, `docs/README.md`, `docs/BENCHMARK_SCHEMA.md` | Scope and proposed schema; no project-specific experiment log. |
| Executable procedures | Three Python scripts under `scripts/` | Two descriptive preflights and one evidence compiler. |
| Encoder result products | 75-row method table, 7-row dataset table, 2-row resource table, two preflight JSON/Markdown pairs | Mixture of R0-native preflight results and R1-derived results. |
| Decoder result products | 52-row method table and 8-row dataset table | Entirely derived from R2 artifacts. |
| Shared products | 9-row method registry plus JSON/Markdown summaries | Manually curated metadata and generated counts. |
| Missing infrastructure | No `src/`, tests, configs, run manifest, research plan, or `docs/EXPERIMENT_LOG.md` | R0 is not yet a standalone benchmark implementation. |

The current summary declares 75 encoder rows, 52 decoder rows, 7 encoder
datasets, 8 decoder datasets, and 9 registered methods. There is no live
`benchmark_results.tsv`; the two track-specific `method_result_table.tsv`
files are the de facto ledgers.

### 2.2 Audit method

This review:

- read all live R0 documentation, scripts, and result files;
- traced all fixed script inputs and outputs;
- reconciled every encoder and decoder method-table row with the builder and
  its named upstream artifact;
- independently checked the two preflight inputs and outputs;
- checked source-path existence, dataset-name consistency, sample-size units,
  gate vocabulary, non-finite values, and registry links;
- compared the May ledger with the current R1/R2 claim boundaries and July
  re-audit evidence; and
- did not modify any result artifact or upstream scientific output.

All 27 JSON artifacts read by the builder were present on the B workstation at
audit time, and all source paths named by the 127 method-result rows existed.
This path check does not validate the internal methods or claims of every
upstream experiment; the checks below state where deeper reconciliation was
performed.

## 3. R0-native procedure P0-1: IndelMissense v1.1 resource preflight

**Script:**
[`indelmissense_v11_resource_preflight.py`](../scripts/indelmissense_v11_resource_preflight.py)  
**Classification:** coordinate/resource-compatibility preflight, not a
pathogenicity experiment  
**Compute:** CPU and Python standard library only

### 3.1 Inputs and outputs

- Fixed input: `data/indelmissense/v1.1_coordinates/records.tsv`.
- Fixed outputs:
  `results/v0_20260515/encoder/indelmissense_v11_resource_preflight.json` and
  `.md`, plus the JSON on standard output.
- There is no CLI, configuration file, seed, or output-version argument.

### 3.2 Procedure

The script loads the complete TSV, counts labels, splits, and variant classes,
and evaluates VCF-style coordinate availability separately for GRCh37 and
GRCh38. A record is counted as VCF-covered only when chromosome, position,
REF, and ALT are all nonempty. It then counts REF/ALT length pairs.

Its exact-SNV proxy is purely `len(REF) == len(ALT) == 1`. It does not query
dbNSFP, REVEL, a reference genome, or any pathogenicity model. Equal-length
alleles longer than one base are counted separately as MNVs for GRCh38.

### 3.3 Results

| Quantity | Result |
|---|---:|
| Total records | 6,649 |
| Pathogenic / benign | 5,274 / 1,375 |
| Train / validation / test | 5,356 / 653 / 640 |
| Deletion / insertion / delins / duplication | 2,530 / 2,504 / 1,187 / 428 |
| GRCh37 VCF-field coverage | 6,631 / 6,649 = 0.9973 |
| GRCh38 VCF-field coverage | 6,635 / 6,649 = 0.9979 |
| GRCh38 exact one-base proxy records | 0 |
| GRCh38 equal-length multi-base records | 194 |

The most frequent GRCh38 allele-length pairs were 4→1 (1,379), 2→1 (998),
3→1 (490), 1→4 (407), 7→1 (305), 10→1 (252), 13→1 (179), 1→7 (163),
16→1 (143), 5→1 (124), 1→10 (107), and 2→2 (106).

The operational conclusion is narrow: this coordinate table is dominated by
indel/delins alleles, so an SNV-centric exact dbNSFP/REVEL join is unsuitable
as the primary comparator without an actual coverage study. Exact joining was
not run. The resource table's claim that dbNSFP was visible on an H200 volume
is historical status, not revalidated by R0; the current B environment lacks a
local dbNSFP BGZF/index and `pysam`.

### 3.4 Audit limitations

- The proxy does not validate chromosome or position syntax, allele alphabet,
  `REF != ALT`, normalization, or reference-genome agreement.
- The two GRCh37 length-one proxy records have `REF == ALT` and are therefore
  not valid substitutions. This does not affect the GRCh38 result of zero.
- Missing columns usually become empty categories rather than schema errors.
- The full TSV is held in memory and outputs are written non-atomically.
- A missing input fails explicitly, but malformed biological coordinates are
  not detected.

## 4. R0-native procedure P0-2: mechanism-feature position preflight

**Script:**
[`encoder_mechanism_localization_preflight.py`](../scripts/encoder_mechanism_localization_preflight.py)  
**Classification:** parseability and sequence-position input preflight, not a
localization, occlusion, CRPI, or faithfulness experiment  
**Compute:** CPU and Python standard library only

### 4.1 Inputs and outputs

- Fixed feature input:
  `r1_encoder_interpretability_benchmark/results/variant_effect/mechanism_feature_audit_firing_20260504.json`.
- Fixed sequence input: `data/swissprot/uniprot_sprot.fasta.gz`.
- Fixed outputs:
  `results/v0_20260515/encoder/mechanism_localization_preflight.json` and
  `.md`, plus JSON on standard output.

### 4.2 Procedure

The script parses each `top_firing_examples` entry using the expected form
`accession:positionAA(activation)`, streams Swiss-Prot to obtain the lengths of
only requested accessions, and calculates:

- normalized position as `position / sequence_length`;
- N-terminal fractions at positions ≤20 and ≤50;
- C-terminal fractions within the last 20 and last 50 residues; and
- example-weighted global and LOF/GOF/DN means, medians, and fractions.

The input contains 150 selected rows: 3 mechanism classes × 5 layers × the top
10 classifier coefficients per class/layer. Six `(layer, feature)` pairs occur
in both GOF and LOF selections, so there are 144 unique `(layer, feature)`
identities rather than 150 independent selections.

### 4.3 Results

All 150 rows and 750 row-example instances parsed and mapped to Swiss-Prot.
The 750 instances reduce to 720 unique feature-position instances and 711
unique accession-position pairs. All 184 accessions mapped, all positions were
in range, and all supplied residue letters matched Swiss-Prot in this audit.

| Stratum | Row-example n | Mean normalized position | Median | N≤20 | N≤50 | C≤20 | C≤50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Overall | 750 | 0.5088 | 0.4987 | 0.0427 | 0.1653 | 0.0653 | 0.1573 |
| DN | 250 | 0.5455 | 0.5903 | 0.0040 | 0.1000 | 0.1080 | 0.1920 |
| GOF | 250 | 0.4486 | 0.4116 | 0.0920 | 0.2240 | 0.0560 | 0.1200 |
| LOF | 250 | 0.5323 | 0.5343 | 0.0320 | 0.1720 | 0.0320 | 0.1600 |

The source's recorded annotation classifications are 3 `KNOWN`, 38
`PARTIAL`, and 109 `NOVEL`. Applying the expanded F1 fields instead produces
4, 48, and 98, respectively. The five keyword-derived note categories contain
87 weak/family-level, 39 domain/fold plausible LOF, 15 regulatory/signaling
plausible GOF, 7 binding/interface, and 2 topology/localization entries.

These counts are input characterization only. They do not show that a feature
localizes a causal residue or a variant mechanism.

### 4.4 Audit limitations

- A missing Swiss-Prot file silently returns an empty length map; a missing
  `rows` field silently becomes an empty list.
- The regex excludes signed or scientific-notation activations and unexpected
  residue-token forms.
- The script itself does not test position bounds or residue identity, even
  though the present input passed those checks during this audit.
- Positions are assumed to be one-based; `position / length` is used without
  documenting the coordinate convention.
- Summary statistics are row-example-weighted. Duplicate selected features
  and positions therefore receive repeated weight.
- The labels `manual_audit`, `manual_interpretation`, and `completed_manual_audit`
  are inaccurate: the interpretations were produced automatically by keyword
  rules over annotation strings.
- The features were selected by mechanism-classifier coefficients, not by
  annotation quality. No inferential test is performed.

## 5. R0-native procedure P0-3: v0 evidence-ledger compilation

**Script:**
[`build_v0_benchmark_tables.py`](../scripts/build_v0_benchmark_tables.py)  
**Classification:** deterministic formatting and evidence aggregation, not an
experiment

### 5.1 Inputs and transformations

The builder uses fixed repository roots and has no command-line interface. It
reads 27 JSON artifacts distributed across R0, R1, and R2. Encoder sources
cover pathogenicity, gene-grouped evaluation, indel baselines and preflight,
ProteinGym, annotation expansion, mechanism-feature classification and
position preflight, plus three negative rescue gates. Decoder sources cover
cross-model matching, low-level characterization, checkpoint comparison,
attention-output readout, N-terminal context, resource coverage, metric
calibration, generated-sequence scoring, steering, basis probes, and causal
ablations.

The builder maps source values into the 11-column schema
`track, task, dataset, method, metric, value, ci95, n, gate, interpretation,
source`. It formats most numbers to four decimal places and writes fixed
dataset and method metadata. Ledger-side calculations include simple
counts/differences/summaries such as decoder null means/maxima, annotation
deltas and class counts, contextual set counts, and a minimum q-value. It does
not perform new model inference or a new inferential test, and most scientific
statistics and gates are copied or hard-coded.

### 5.2 Outputs and counts

| Output | Current content |
|---|---:|
| `encoder/method_result_table.tsv` | 75 rows |
| `encoder/dataset_table.tsv` | 7 rows |
| `encoder/resource_readiness.tsv` | 2 rows |
| `decoder/method_result_table.tsv` | 52 rows |
| `decoder/dataset_table.tsv` | 8 rows |
| `method_registry.tsv` | 9 rows |
| `summary.json` and `summary.md` | Counts, dataset copies, and notes |

The JSON summary timestamp is `2026-07-16T11:55:08.174260+00:00`; the Markdown
summary timestamp is 219 microseconds later. The containing directory remains
named `v0_20260515`, so it is best interpreted as a May evidence snapshot
regenerated after the July path migration, not an immutable May release.

### 5.3 Failure and reproducibility behavior

- A missing JSON source silently becomes `{}`. The build can succeed while
  omitting evidence.
- Widespread `.get(..., default)` use can manufacture plausible-looking
  fallback values. This caused the three false steering rows described below.
- Files are overwritten sequentially and non-atomically in the fixed dated
  directory, allowing a failed run to leave mixed generations.
- Gates are mostly copied or hard-coded labels rather than validated decisions.
- The two summaries read the wall clock separately, preventing byte-identical
  regeneration.
- The build records no evidence cutoff, schema version, Git revision, command,
  environment, source hash, source modification time, or immutable run ID.
- Four-decimal formatting discards source precision.
- Upstream JSON contains non-RFC `NaN` or `Infinity`, and the generated decoder
  TSV contains one literal `nan`.

## 6. Encoder evidence organized by procedure and result

Everything in this section except the two R0 preflights is inherited from R1.
The results are predictive, coverage, or input-quality evidence; none by itself
establishes explanation faithfulness.

### 6.1 Missense pathogenicity: available-score comparison

The source compares SAE-LR, ESM-2 LLR, their ensemble, and available external
scores. Coverage differs by method, so sample sizes must accompany AUCs.

| Cohort | SAE-LR | ESM-2 LLR | SAE+LLR | AlphaMissense | gMVP | ESM-1v |
|---|---:|---:|---:|---:|---:|---:|
| ClinVar2000 | 0.8782 [0.8619, 0.8923], n=2,000 | 0.8822 [0.8677, 0.8960], n=2,000 | 0.9143 [0.9010, 0.9259], n=2,000 | 0.9474 [0.9377, 0.9567], n=1,972 | 0.9369 [0.9257, 0.9474], n=1,788 | 0.9089 [0.8952, 0.9212], n=1,972 |
| CancerHoldout101 | 0.9079 [0.8471, 0.9590], n=101 | 0.8978 [0.8273, 0.9561], n=101 | 0.9193 [0.8617, 0.9646], n=101 | 0.9700 [0.9244, 0.9988], n=101 | 0.9400 [0.8878, 0.9806], n=98 | 0.8552 [0.7768, 0.9211], n=101 |

Identically named SAE methods are not methodologically identical across these
two rows. ClinVar `SAE-LR` uses annotation-selected, high-dimensional
perturbation features with variant-level cross-validation, whereas the cancer
holdout version uses 50 aggregate signature features trained on non-cancer
genes. ClinVar `SAE+LLR` is a simple z-sum; the cancer-holdout version is a
fitted logistic combination. Cross-cohort differences therefore should not be
interpreted as performance changes of one fixed method.

`CancerHoldout101` is not an external cancer cohort. It is a subset of 101
ClinVar variants from curated cancer genes, trained against 1,899 variants
from non-cancer genes; no COSMIC or other external cancer table was available.

### 6.2 Gene-aware ClinVar comparison

| Method | ROC AUC [95% CI] | n |
|---|---:|---:|
| AlphaMissense | 0.9474 [0.9368, 0.9567] | 1,972 |
| gMVP | 0.9369 [0.9252, 0.9481] | 1,788 |
| ESM-1v | 0.9089 [0.8927, 0.9234] | 1,972 |
| SAE-LR group-CV | 0.7559 [0.7300, 0.7809] | 1,972 |
| AlphaMissense+SAE stack | 0.8542 [0.8352, 0.8719] | 1,972 |
| AlphaMissense+SAE z-sum | 0.9210 [0.9072, 0.9330] | 1,972 |

Only SAE-family predictions came from gene-grouped cross-validation. The
external scores were evaluated directly with gene-bootstrap confidence
intervals, so “all methods under grouped CV” would be inaccurate. Neither SAE
combination exceeds AlphaMissense or gMVP.

### 6.3 IndelMissense protein-level pathogenicity

The source contains 6,649 variants from 1,894 proteins. The last two learned
models use five-fold `StratifiedGroupKFold` by UniProt ID.

| Method or feature | ROC AUC [95% CI] | n | Audit interpretation |
|---|---:|---:|---|
| SAE damage score | 0.7735 [0.7616, 0.7849] | 6,649 | Fails the saved conjunctive standalone gate: AUC ≥0.78 and ≥0.03 above at least one non-SAE baseline. |
| Truncating indicator | 0.7495 [0.7420, 0.7570] | 6,649 | Descriptive baseline. |
| Absolute length delta | 0.7343 [0.7230, 0.7456] | 6,649 | Descriptive baseline. |
| Relative absolute length delta | 0.7461 [0.7345, 0.7567] | 6,649 | Descriptive baseline. |
| Early-position score | 0.3974 [0.3777, 0.4156] | 6,649 | Below chance in the saved orientation. |
| Early-truncation score | 0.7508 [0.7436, 0.7580] | 6,649 | Descriptive baseline. |
| gnomAD pLI | 0.3525 [0.3356, 0.3713] | 5,875 | Coverage-limited gene constraint. |
| gnomAD LOEUF score | 0.4269 [0.4072, 0.4468] | 5,875 | Coverage-limited gene constraint. |
| gnomAD LoF-Z | 0.4437 [0.4252, 0.4620] | 5,875 | Coverage-limited gene constraint. |
| gnomAD missense-Z | 0.5100 [0.4898, 0.5308] | 5,887 | Coverage-limited gene constraint. |
| ESM region mean ΔNLL | 0.8037 [0.7924, 0.8148] | 6,649 | Stronger than SAE alone. |
| ESM region sum ΔNLL | 0.5059 [0.4912, 0.5199] | 6,649 | Near chance. |
| Cheap-feature grouped LR | 0.8447 [0.8344, 0.8545] | 6,649 | Predictive result. |
| SAE+ESM+cheap grouped LR | 0.9108 [0.9027, 0.9185] | 6,649 | Best listed model; not faithfulness evidence. |

All reported confidence intervals are ordinary row bootstraps rather than
protein-group bootstraps and therefore ignore within-protein dependence.
Missing-value medians for the learned models are computed before fold
splitting, creating minor unsupervised leakage. These qualifications are not
represented in the R0 row schema.

### 6.4 ProteinGym substitution assays

Of 217 assay records, 214 were usable; three had no SAE-scored mutants.
Reported values are unweighted means across usable assays.

| Score | Mean Spearman ρ [95% CI] |
|---|---:|
| ESM-2 LLR | 0.4341 [0.4091, 0.4594] |
| Raw SAE disruption | −0.2314 [−0.2588, −0.2035] |
| Sign-corrected SAE disruption | 0.2314 [0.2034, 0.2592] |
| Sign-corrected SAE+LLR | 0.4047 [0.3818, 0.4282] |

The ensemble beats LLR on 33.64% of usable assays, below its 40% gate. The
negative raw correlation largely reflects comparing a higher-is-damage score
with higher-is-fitness DMS measurements. The SAE family does not beat ESM-2
LLR in the aggregate.

### 6.5 Feature-annotation expansion

For each ESM-2 layer, R0 records known features, useful features, and gains
after GO/Pfam/BioLiP expansion.

| Layer | R0 `n` | Known | Useful | Δ known | Δ useful |
|---|---:|---:|---:|---:|---:|
| 19 | 13,868 | 381 | 7,481 | +235 | +1,433 |
| 23 | 13,832 | 301 | 6,773 | +166 | +1,432 |
| 27 | 13,456 | 163 | 4,423 | +91 | +1,039 |
| 31 | 13,649 | 86 | 3,197 | +37 | +876 |
| 35 | 12,895 | 45 | 1,935 | +13 | +513 |

The R0 `n` values count features with positive overlap against an added label;
they do not count all features with firing positions. The actual nonempty
firing-position counts are 15,731, 15,698, 15,518, 15,460, and 14,355. The
added-label F1 calculation tests many labels using at most 200 saved top firing
positions. The final combined score is the maximum of that expanded score and
the original F1 computed from the full thresholded activation set. There is no
null or multiple-testing correction. The results are descriptive coverage,
not evidence of biological primitives or causality.

### 6.6 Mechanism-feature classification and position preflight

R0 records 3 `KNOWN`, 38 `PARTIAL`, and 109 `NOVEL` source classifications,
then emits ten preflight rows: seven overall quantities (parsed-feature count,
example count, mapping fraction, mean, median, N≤20, and C≤20) plus three
class-specific mean-position rows. Section 4 gives the fuller position results
available in the preflight artifact.

The audit found that the source labels differ from classifications obtained
from the expanded F1 fields (4/48/98), six selected features are duplicated
across classes, and all note categories are rule-generated. These rows should
remain input-quality evidence only.

### 6.7 Pre-specified negative/rescue gates

| Gate | Current R0 result | Correct audit interpretation |
|---|---|---|
| VAMP-like abundance proxy | `1/9`, `FAIL` | The source used the obsolete plus-sign ensemble. Replacing it post hoc with already-saved sign-corrected assay scores produces point-estimate wins on 3/9 assays, exactly the nominal gate. This is an orientation defect, **not** a rescued claim: there is no paired uncertainty test, AlphaMissense is only a proxy abundance comparator, and no full VAMP rescoring was run. |
| Low-homology proxy | Δ AUC −0.0900 [−0.1253, −0.0568], `FAIL` | In the tested low-Q1 stratum, SAE+LLR=0.8727 versus AlphaMissense=0.9627. R0 incorrectly records parent-cohort n=1,972; tested n=450 across 250 genes. UniRef50 cluster size is a proxy, not MSA Meff. |
| AM/SAE disagreement typing | 0 significant contexts, `FAIL` | 473 disagreements exist, but inference uses 321 review-filtered rows (283 AM-right/SAE-wrong, 38 SAE-right/AM-wrong); 0/90 tests have BH q<0.05 and best q=0.0717. R0 records 473 rather than inferential n=321. |

The encoder conclusion remains a calibrated audit: available SAE evidence
supports coverage and benchmark construction, not superiority to
AlphaMissense, low-homology rescue, abundance advantage, or statistically
typed blind spots.

## 7. Decoder evidence organized by procedure and result

Every result in this section is inherited from R2. Current claim discipline
permits procedure-specific cross-model sparse readouts, a candidate checkpoint
diagnostic from one pair, and three N-terminal initiator-methionine readouts
with high unnormalized received attention. It does not permit single-feature
causality, an attention-sink mechanism, biological primitives, successful
steering, or a claim that wider dictionaries exclude capacity/optimization.

### 7.1 Cross-model sparse-readout discovery and characterization

| Procedure | Result | R0 gate | Audit interpretation |
|---|---|---|---|
| Fixed three-model matching on 200 sequences | 38, 30, and 8 triplets at absolute correlation ≥0.90, ≥0.95, and ≥0.98 | `PASS` ×3 | Across 30 assignment permutations, the null means were 0.0667, 0.0667, and 0.0333; maxima were 1. This is procedure-specific recurrence, not universality. |
| Low-level characterization of 38 triplets | 37/38 had at least one association | `PASS_readout_not_biology` | Descriptive readout evidence. Residue-level permutations do not fully model within-protein dependence. |
| Characterization signatures | received-attention 4; BPE boundary 1; high CLT-input norm 25; k-mer 27; position 35 | `diagnostic` ×5 | Multi-label counts; not biological-primitive evidence. `attention-sink` is a legacy artifact label. |
| One mature/early checkpoint pair | 38 versus 16 matched triplets | `PASS` ×2 | Candidate checkpoint diagnostic from one selected pair only; not a validated separator. R0 leaves n blank although the cohort has 200 sequences. |

The feature hooks are architecture-specific: the features encode normalized
CLT inputs selected for each model architecture, while the CLTs reconstruct
MLP outputs. They should not be described as one literally identical
“MLP-output feature” space across models.

The upstream CLT training and quick-evaluation paths discard `attention_mask`,
so padding entered gradients, FVU, and firing statistics. A July mask audit
reconstructed 298/8,342 padded positions (3.57%) for ProtGPT2 and 673/24,466
(2.75%) for ZymCTRL in the saved quick-evaluation totals. Single-sequence atlas
extraction itself had no batch padding. R0 does not encode this limitation.

### 7.2 Attention-output and N-terminal context pilot

The layer-23 attention-output dictionary used 5,000 training records and 120
evaluation sequences (`d_sae=2048`, top-k=32, 3,000 steps). It reported mean
evaluation FVU 0.4782, maximum received-attention correlation 0.9815, and
maximum first-two-position activation delta 8.0173. The alive-feature fraction
was only 0.0869, and the displayed metrics are selected maxima. The causal
ablation gate failed.

The legacy contextual test considered four selected candidates across 24
position/k-mer tests: 18 tests were significant, three candidates had
significant context, and the best BH q-value was 2.7189×10^-117. Only
T011/T018/T023 are the N-terminal initiator-methionine readouts; T025 is a
non-N-terminal comparator. Tests use events rather than proteins as the
statistical unit. These results support context association, not a causal
attention-sink mechanism.

### 7.3 Resource coverage

Among 33 unique accessions represented by 500 top events from 10 triplets,
seven joined Pfam, seven joined Swiss-Prot, and zero joined AlphaFold. The R0
dataset row's `n=500` is an event count, whereas result rows use `n=33`
accessions. The source does not cover all 38 triplets.

### 7.4 External generation-metric calibration

The real-lysozyme versus length-matched random UniRef50 control used 100+100
sequences. Standardized effect sizes were:

| Metric | Effect size d |
|---|---:|
| Pfam lysozyme-like hit | 4.4969 |
| CLEAN exact EC 3.2.1.17 | 6.9282 |
| CLEAN 3.2.1.x prefix | 5.5486 |
| ESMFold mean pLDDT | 1.7858 |
| ESMFold confident fraction | 2.0632 |
| Foldseek top TM | 1.5927 |

This validates that the metric stack distinguishes these particular real and
random panels. It is not evidence that steering works. R0 discards the source
p-values.

### 7.5 Generated-sequence metric triad and compiler defect

| Panel | Sequence/Foldseek n | Pfam | CLEAN exact | CLEAN prefix | Foldseek mean top TM |
|---|---:|---:|---:|---:|---:|
| Selected steered leads | 10 / 10 | 0.900 | 0.900 | 0.900 | 0.8833 |
| All steered generations | 200 / 0 | 0.860 | 0.775 | 0.865 | unavailable |
| Unsteered generations | 200 / 20 | 0.820 | 0.775 | 0.875 | 0.9384 |

The selected ten leads are a filtered subset and cannot estimate
generation-wide lift. The dataset manifest reports 410 panel entries by
counting those selected leads again, but there are at most 400 distinct,
non-double-counted generated-sequence records; statistical independence is not
established.

The source's valid all-steered versus unsteered tests are:

| Metric | Difference | One-sided Fisher p | n |
|---|---:|---:|---:|
| Pfam hit rate | +0.0400 | 0.16989 | 200+200 |
| CLEAN exact rate | 0.0000 | 0.54763 | 200+200 |
| CLEAN prefix rate | −0.0100 | 0.67205 | 200+200 |

**Critical compiler error:** the builder expects source keys `diff`,
`fisher_greater_p`, `a_n`, and `b_n`, but the current source exposes `table`
and `p`. R0 consequently publishes false values `0.0000`, `p=1.0000`, and
`n=+` for all three comparisons. The negative scientific conclusion happens
to remain unchanged, but the reported statistics are not source values.

The builder also expects obsolete Foldseek threshold keys. It omits six source
threshold fields—four finite values and two `NaN` values for the all-steered
panel with no Foldseek structures—writes literal `nan` for the all-steered
Foldseek mean, assigns that row n=200 despite zero structures, and assigns the
unsteered Foldseek result n=200 despite only 20 structures.

### 7.6 Steering results

- Multiplier 2.5: 0/8 enzyme-labelled prompt/heuristic-score targets showed a
  significant positive shift, with 100 sequences per condition.
- Multiplier 5.0: 0/3 targets showed a significant positive shift, with 64
  sequences per condition. This older sweep predates direct-effect feature
  selection and should not be labeled a direct-effect run.

The class score is a motif/composition heuristic rather than validated EC
purity. These are calibrated negative results and do not support successful
EC-class steering.

### 7.7 Swiss-Prot biological-label gate: stale and invalid in R0

R0 still compiles the May row `FAIL`, 0/38 from the original Swiss-Prot
annotation gate. Its 0.1-nat mutual-information threshold is mathematically
impossible for the defined top-100 binary event: with 122,671 positions, the
event entropy and maximum possible MI are 0.00661255 nats, 15.12 times below
the gate.

The current July re-audit, absent from R0, tested 380 triplet-label pairs with
2,000 permutations:

- 224/380 survived the global position null;
- 0/380 survived a within-protein null matched on amino acid and fine
  position/edge strata; and
- the maximum matched excess normalized MI was 0.0218028, raw p=0.000500,
  BH q=0.09495.

The defensible result is therefore an exploratory matched-null negative for
the saved, selected top-event analysis. It is not evidence that all biological
associations are absent. In addition, `dominant_pfam` has 500 levels for 500
proteins, so it supplies no replicated cross-protein family-generalization
evidence. The old row should be marked `INVALID_SUPERSEDED`, not treated as an
ordinary failed biological gate.

### 7.8 Basis probes and causal-ablation negatives

| Procedure | Result | Audit interpretation |
|---|---|---|
| Pfam probe | 38-D triplets macro-F1 0.3995 vs 2,560-D ESM-2 1.0000, n=240 | Dimension-unmatched exploratory comparison. |
| EC top-class probe | 0.2550 vs 0.7080, n=280 | Does not show a biological dictionary. |
| Secondary-fraction probe | R² −0.6049 vs 0.3245, n=300 | The 38-D triplet basis underperformed the 2,560-D comparator in this exploratory probe. |
| CLT feature intervention | Gate failed; source contains 6,000 per-sequence rows | R0 leaves n blank. The specified intervention gate failed. |
| Single-head intervention | Gate failed; 4,800 rows | The specified intervention gate failed. |
| Top-8 head-set intervention | Gate failed; 3,000 rows | The specified intervention gate failed. |
| Top-32 head-set intervention | Gate failed; 1,800 rows | The specified intervention gate failed. |

These failures neither establish nor exclude all distributed, alternative-site,
or nonlinear mechanisms.

## 8. Cross-ledger integrity and documentation findings

### 8.1 Prioritized defects

| Severity | Finding | Consequence |
|---|---|---|
| **Critical** | Decoder steering comparison reads nonexistent keys and manufactures three zero effects, p=1 values, and `n=+`. | The R0 ledger reports false statistics even though the negative decision is unchanged. |
| **Critical** | The impossible May Swiss-Prot 0.1-nat gate remains active; the July matched-null re-audit is absent. | The biological-label evidence is stale and the failure reason is misrepresented. |
| **High** | The VAMP-like encoder gate uses an obsolete plus-sign ensemble; the saved sign-corrected point estimates give 3/9 rather than 1/9 wins. | The current row is orientation-dependent and must be re-audited, without reviving an abundance-advantage claim. |
| **High** | Foldseek field-name drift omits six measurements and yields one non-finite value with wrong sample sizes. | The generation triad is incomplete and statistically ambiguous. |
| **High** | Missing JSON silently becomes `{}` and outputs are overwritten non-atomically. | A successful run cannot certify completeness or single-generation consistency. |
| **High** | Checkpoint rows describe one mature/early pair as a validated separator. | Claim exceeds evidence; it is only a candidate diagnostic. |
| **High** | Two registry rows cite missing live `docs/BENCHMARK_PROGRESS_20260515_CN.md`. | Registry path integrity fails; the document is only in frozen `archive/project_records/`. |
| **Moderate** | Low-homology and disagreement rows use parent or pre-filter sample sizes. | `n` does not identify the tested statistical unit. |
| **Moderate** | Dataset names lack referential integrity and `n` mixes variants, assays, layers, features, events, accessions, and result rows. | Cross-table joins and sample-size interpretation are unsafe. |
| **Moderate** | Gate tokens mix decision, workflow, evidence level, and claim scope. | Automated PASS/FAIL summaries are not meaningful. |
| **Moderate** | `ci95` is misused for p-values; 49/52 decoder rows have no interval. | Statistical fields cannot be parsed consistently. |
| **Moderate** | `manual_audit`, `attention_sink`, “universal,” and “MLP-output CLT triplets” retain over-broad or legacy wording. | Registry and task labels can be mistaken for supported claims. |
| **Moderate** | The fixed `v0_20260515` directory is regenerated with new timestamps but no immutable manifest. | Evidence cutoff and byte-level provenance are ambiguous. |
| **Moderate** | R0 has no project `EXPERIMENT_LOG.md`, tests, or configuration layer. | Procedure history and regression protection are incomplete. |

### 8.2 Dataset referential integrity

The encoder manifest has seven rows and all named source paths exist, but six
dataset/pseudo-dataset identifiers used by encoder result rows are absent. The
method table uses `IndelMissense_v1`, while the manifest lists
`IndelMissense_v1.1_coordinates`; metadata indicates the same records, but no
explicit version crosswalk is provided.

The decoder method table uses 18 dataset identifiers, 12 of which are absent
from its eight-row manifest. Conversely, umbrella entries such as
`N_terminal_sink_ablation_suite` and
`lysozyme_generated_steered_unsteered` are never used verbatim by result rows.
Specific manifest issues include:

- `balanced200_three_model_atlas` is marked `runnable`, but its upstream input
  still points to a missing legacy `Research2/...` cohort path;
- `ZymCTRL_layer23_attention_output` and the ablation suite use `n=0` to mean
  unpopulated even though source counts are available;
- `N_terminal_sink_triplets n=4` includes one non-N-terminal comparator;
- `balanced200_top_firing_positions n=500` means events, not proteins; and
- the generation manifest's `n=410` double-counts ten selected leads.

### 8.3 Method registry

The nine registry rows comprise four encoder and five decoder methods. Three
encoder registry sources and four decoder registry sources resolve; the two
not-started attribution rows share the broken progress-document link.

Terminology should be corrected as follows:

- `MLP-output CLT triplets` conflates the architecture-specific normalized CLT
  input encoded by a feature with the MLP output reconstructed by the CLT;
- `attention-head and sink-set` should use claim-safe N-terminal-readout
  terminology;
- `Pfam/CLEAN/Foldseek/ESMFold triad` conflates the four-metric calibration
  stack with the three-family generated-sequence triad; and
- not-started methods should remain registry placeholders, not evidence rows.

### 8.4 Gate vocabulary

The decoder table alone uses ten tokens: `PASS`, `PASS_readout_not_biology`,
`diagnostic`, `readout_PASS_causal_FAIL`, `PASS_context_not_causality`,
`coverage_audit`, `PASS_metric_validity`, `filter_pass`,
`benchmark_measurement`, and `FAIL`. Encoder rows add still more tokens.

A future schema should separate at least:

- `result_status`: available, missing, invalid, superseded;
- `gate_decision`: pass, fail, not_applicable;
- `evidence_level`: descriptive, predictive, associational, interventional;
- `claim_scope`: input quality, readout, diagnostic, causal, generation; and
- `statistical_unit` and `n_unit`.

## 9. Reproduction instructions

Run from repository root on the B workstation, in this order:

```bash
source /Data/lzp/BioInterpretebility-CC/.venv/bin/activate

python r0_shared_interpretability_framework/scripts/indelmissense_v11_resource_preflight.py
python r0_shared_interpretability_framework/scripts/encoder_mechanism_localization_preflight.py
python r0_shared_interpretability_framework/scripts/build_v0_benchmark_tables.py
```

The requested environment reported Python 3.13.5 during the audit. All three
scripts use the standard library, require no GPU, and passed Python syntax
parsing. Order matters because the builder consumes both R0 preflight JSONs.

These commands overwrite the current fixed output directory. Until the
builder is repaired, rerunning it will faithfully recreate the known malformed
rows described in Sections 7.5 and 8.1.

### 9.1 Audit-time SHA-256 inventory

The hashes below identify the executable and generated R0 artifacts audited
before this document was added. They are not a substitute for a builder-owned
manifest because upstream input hashes are not included.

| Artifact | SHA-256 |
|---|---|
| `scripts/build_v0_benchmark_tables.py` | `569639bd3ae8251bd772525cb08eab2a622e834b6a98582a6a13692cf0a218bd` |
| `scripts/encoder_mechanism_localization_preflight.py` | `e3aa2511f28ed87713c6eada8057a2188002a45049de3bf7ff21b0b58009b212` |
| `scripts/indelmissense_v11_resource_preflight.py` | `27f0a36b28969adc2601b6d2a0c0401c4c8fc23f031c49a7c2d51cec88f01709` |
| `encoder/method_result_table.tsv` | `7466bb0c14da35c2c5d96f479d85118e187ebe3aae9d4cfe9a96840ba48ffacf` |
| `encoder/dataset_table.tsv` | `f303f1c41a77d76a5a7e6c5b7f2083fe2f0e809aa58f3f9279a14814b70fffef` |
| `encoder/resource_readiness.tsv` | `8fcdbc6e2b2a1df7681351bca6f931f62ad816e37ddc401464c031619112582e` |
| `encoder/indelmissense_v11_resource_preflight.json` | `13f8eaf4a18002621ef5e01a5db19afd83acf83f70fefb4c5764fc2d55c5b39a` |
| `encoder/indelmissense_v11_resource_preflight.md` | `51c3a8f5cf94a6cba635c780f24410af9fba3d598a653b7a418b286b26d5c9a2` |
| `encoder/mechanism_localization_preflight.json` | `21d90724d5f39f9b6013c10bbbd2be9a370e05aa8b717152178efba11b05ee89` |
| `encoder/mechanism_localization_preflight.md` | `e255ebb7cd943708c5e82a0938c1ecd84c44b38101d0452ca4c52072069080ec` |
| `decoder/method_result_table.tsv` | `9c1fe432fbe61a81407a4c305f0c2e8e6d1e8c8ae9210600b83c09927e3bf3f4` |
| `decoder/dataset_table.tsv` | `26e95a14df1a5c00fcd3b40fe7ea6cbdf8330e26cb7d7ed37edcda1f0d0776a3` |
| `method_registry.tsv` | `07e08eadc8864153eb637105b3d5efb0b58aebd4ab21726cc840905969663664` |
| `summary.json` | `02f66292da09cd9212b518943ac5ab1692eb0cac9612bf4c096fad00aae2ec89` |
| `summary.md` | `2365857918c8e30feaa499e7c4e0a3837a02e34e060541327abc5d050a653329` |

## 10. Audit disposition and required remediation

### 10.1 Present disposition

R0 is accepted as an **internal, human-reviewed evidence index with known
exceptions**. It is rejected as a publication-ready benchmark release,
machine-verifiable experiment registry, or leaderboard.

The current evidence supports these bounded statements:

- the two R0 preflight input packages/artifacts are parseable and have the
  descriptive properties reported above;
- encoder evidence supports benchmark and coverage work but not the failed
  SAE superiority/rescue claims;
- decoder evidence supports procedure-specific sparse readouts, one candidate
  checkpoint diagnostic, and three N-terminal initiator-methionine readouts
  with high unnormalized received attention; and
- the tested steering scores, interventions, and gates remain negative.

### 10.2 Repair order before release

1. Correct the decoder steering/Foldseek extraction and add regression tests
   against the current source schema.
2. Replace the invalid Swiss-Prot row with an explicitly
   `INVALID_SUPERSEDED` record plus the matched-null re-audit.
3. Re-run the VAMP-like gate with an explicitly signed score definition,
   paired uncertainty analysis, and unchanged negative-claim discipline.
4. Correct statistical sample sizes and add `n_unit`/`statistical_unit`.
5. Enforce source existence, required keys, finite values, and atomic output;
   fail the build on any violation.
6. Normalize dataset identifiers and make every result row resolve to exactly
   one dataset-manifest entry.
7. Split status, gate, evidence level, and claim scope into controlled fields.
8. Add immutable run IDs, an evidence cutoff, schema version, source hashes,
   environment/command metadata, and one shared generation timestamp.
9. Correct registry links and claim-unsafe terminology.
10. Add R0 tests and `docs/EXPERIMENT_LOG.md`, then record all future preflight
    and ledger generations chronologically.

Until those changes are implemented, this document is the authoritative
human-readable audit layer over `results/v0_20260515/`.
