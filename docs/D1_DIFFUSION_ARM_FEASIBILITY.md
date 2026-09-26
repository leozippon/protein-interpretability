# Non-autoregressive generative arms: executability and score qualification

**Feasibility assessment, 2026-09-26. No pipeline run, no admitted-panel quantity changed, and nothing in `manuscript/` or `summary.md` touched.** The question asked here is narrow: can a non-autoregressive generative paradigm — discrete diffusion over sequence, or sequence–structure flow matching — enter the existing biological capability pipeline at all, and at what cost. Everything below is either an interface fact established by inspection or a quantity measured on a named exploratory support. Nothing here licenses a capability claim about any model, and no count, verdict or figure in the capability map moves because of it.

Two suggested candidates were audited, DPLM (sequence-level absorbing-state discrete diffusion) and La-Proteina (sequence–structure latent flow matching). They were treated as suggestions rather than requirements, and the audit also covered DPLM-2 as the nearer sequence–structure option.

## Admission rule for anything that follows

Any model admitted from this line goes into a **separate exploratory roster** with its own records. The panel-level counts that the capability map rests on — 33 admitted arms, 14 of 33 resolving over the strongest measured control, 4 of 33 on the stability endpoint, 11 residue-level against 19 BPE and 3 byte-level strata — are counts over that panel and must not change because a model was added. If the extension is ever integrated into the admitted panel, the counts change deliberately, once, and with the reason recorded. Until then an exploratory row is read beside the panel and never inside it.

## What is executable on this host

The route matters as much as the artefact, so it is recorded first. From Compute, `hf-mirror.com` and `pypi.org` respond (HTTP 200); `huggingface.co`, `github.com` and `raw.githubusercontent.com` do not respond within 20 s and are reported as blocked rather than absent. A third-party GitHub proxy, `ghproxy.net`, does respond and serves raw repository files (the DPLM README at 32,517 bytes and the La-Proteina README at 19,592 bytes), so **reference source code is reachable, through a third party**; a `gitclone.com` clone of the La-Proteina repository returned HTTP 502. `catalog.ngc.nvidia.com` responds at its root and `api.ngc.nvidia.com` returns 401 without credentials. A blocked direct route is therefore not the same as an unavailable artefact anywhere in this assessment.

**DPLM (sequence-level discrete diffusion) is executable in the validated `ct` environment with no new dependency.** The mirror carries `airkingbd/dplm_150m`, `dplm_650m` and `dplm_3b`; the 3B checkpoint is 11.37 GB in four fp32 shards and was fetched with the mirror and the token from the ignored local environment file, as was the 650M rung. The decisive fact is structural: the checkpoints' weight keys are exactly `esm.*` and `lm_head.*`, and `transformers.EsmForMaskedLM.from_pretrained` consumes them with **0 missing, 0 unexpected and 0 mismatched keys** at both sizes (2,839,005,634 and 651,043,254 parameters) under Python 3.11.14, PyTorch 2.9.1+cu128 and Transformers 4.57.3. The reference stack is not needed and must not be installed: `bytedance/dplm` pins Python 3.9, `lightning==2.2.0`, `torchtext==0.17.0` and `deepspeed==0.14.4`, which is incompatible with `ct`.

Skipping the reference loader is safe here, and that was checked rather than assumed. Reading `src/byprot/models/dplm/modules/dplm_modeling_esm.py`, `EsmForDPLM` subclasses `EsmForMaskedLM`, builds the stock `EsmEmbeddings` (so ESM-2's `token_dropout` rescaling applies identically), and its modified attention pre-scales the query by `head_size**-0.5` and calls `scaled_dot_product_attention` with `scale=1.0`, which is the same function of the same weights; the head is the stock `EsmLMHead`; the wrapper derives its attention mask from the pad id, as the tokenizer's mask does. The `hidden_dropout_prob` the wrapper overrides is inactive in evaluation mode. The one divergence is in the sampling path, not the forward: the reference sets the `<mask>` and `X` logits to negative infinity before decoding. That renormalisation cancels exactly in the contrast defined below, because the contrast is a difference of two log-probabilities at one position under one normalisation.

**DPLM-2's joint interface is not executable here; its sequence half is.** `airkingbd/dplm2_650m` and `dplm2_3b` are on the mirror under Apache-2.0 with the same ESM-shaped config and a vocabulary of 8,229. Structure input requires `airkingbd/struct_tokenizer`, which holds `dplm2_struct_tokenizer.ckpt` and a `config.yaml` and no architecture code; the coordinate-to-token encoder lives in the reference stack. The Synthyra redistribution bundles DPLM-2 modelling and token-vocabulary code but no coordinate encoder was found in its file listing. So a sequence–structure DPLM-2 arm would require the reference runtime, which is the same runtime that must stay out of `ct`.

### No sequence–structure generative arm is executable against this pipeline

This is a finding about a model class and its interface, not a failure to obtain an artefact, and it is worth stating on its own terms so that a later reader does not re-derive it. It holds independently of the runtime incompatibility recorded below, and it would still hold if that environment were built tomorrow.

La-Proteina's checkpoints are obtainable and licensed for use: `nvidia/NV-La-Proteina-Ucond-v1` and `-Motif-v1` are on the mirror under the NVIDIA Open Model License, as five Lightning checkpoints (two autoencoders, three latent diffusions). Its code is reachable. What it does not have is a counterpart to anything this pipeline measures. The repository's entry points are `proteinfoundation/generate.py` and `proteinfoundation/evaluate.py`; the inputs are a residue count, noise scales, a seed and optionally motif coordinates, and the output is a PDB file. **No entry point accepts a supplied sequence and returns a quantity.** Scoring a wild type against a mutant would need two things the release does not provide and the cohorts cannot supply: an all-atom structure for each variant, which none of the three cohorts carries, and a likelihood or evidence bound under a continuous flow in a partially latent space, which the released code does not expose. Obtaining the structures from a folding predictor would make the score a function of that predictor rather than of La-Proteina, and would import that instrument's licensed scope, which on this programme's record is a paired within-sequence pLDDT contrast with calibration attained on 2 of 19 arms.

The nearer alternative closes for a different reason, recorded with DPLM-2 above: its structure side needs a coordinate encoder whose architecture lives in the reference runtime. So the joint interface is out of reach for DPLM-2 on this host and absent by construction for La-Proteina.

The consequence is a scope statement rather than a result about any model: **the sequence–structure slot cannot be filled by adding an arm, because what is missing is an endpoint.** The three cohorts carry sequences and measured quantities and no per-variant structures, so a model that consumes structure has nothing to consume. If that slot matters, the work to do first is to build or source an endpoint with per-variant experimental structures, and only then to ask which model can read it.

The runtime is recorded for completeness and is not what decides this: a separate conda environment with `torch==2.7.0+cu118`, `torch_geometric`/`torch_scatter`/`torch_sparse`/`torch_cluster` compiled against that exact pair, `graphein==1.7.7`, `lightning==2.5.0` and `numpy==1.23.5`, incompatible with `ct`, with the cu118 wheel index returning 403 on a bare probe.

## The model-native score, and whether it qualifies

DPLM is an absorbing-state discrete diffusion model whose denoiser is a single bidirectional forward pass over a partially masked sequence; it carries no timestep input, which the weight listing confirms by containing nothing beyond the ESM-2 parameters. The corruption level enters only through how much of the input is in the absorbing state. That fixes the natural shape of a native score and rules out forcing an autoregressive likelihood on it.

For a sequence x of length L and a corruption level p, draw a mask set M of size m = max(1, round(pL)), replace those positions with `<mask>`, and take

s(x | M) = (1/|M|) · Σ_{i ∈ M} log p_θ(x_i | x_M),

one Monte-Carlo term of the model's own denoising objective at that level. Two contrasts are built from it, and they behave completely differently.

**(A) The mutation-anchored contrast** requires the substituted position j to be in M and reads both residues off the same forward pass: Δ_A = log p_θ(a | x_M) − log p_θ(w_j | x_M). Wild type and mutant see a bit-identical input, so **the corruption is matched by construction and not by agreement between two draws**. One forward pass serves all nineteen substitutions at every masked position at once. At p = 1/L, with M = {j}, the member carries no sampling at all.

**(B) The full-sequence contrast** is Δ_B = s(x_mut | M) − s(x_wt | M) under a shared M. It scores the mutant sequence itself, which is what a diffusion model's own evidence bound is about, and it needs two forward passes.

The reference release exposes no scoring entry point, so this score is a declaration of this programme's, computed from the reference forward pass.

### What was measured

All numbers in this section come from DPLM-650M on an exploratory slice of the MegaScale deposition: two natural backgrounds, `1O6X.pdb` and `1UBQ.pdb_L67S`, both of length 72, 120 single substitutions each, 32 mask realisations per cell, split seed 20260926. **That slice is not the frozen stability cohort and carries no resampling contract, so the quantities below are point measurements without intervals and none may be quoted as if it had one.** Their job is to decide whether the score can be used, not what it says.

Determinism first. Two independent evaluations of the same forward pass agree to a largest absolute difference of **0.0** across the full 33-token distribution at every position, and two independent computations of the deterministic member agree to **0.0** on all 240 variants. Its cost is 57 and 63 forward passes per background, its between-variant spread is 2.856 and 2.425 nats, and its rank correlation against the deposition's `ddG_ML` on this slice is +0.606 and +0.365.

Seed variance is then read against the spread it would have to resolve. Writing σ_seed for the within-variant standard deviation across realisations and σ_signal for the between-variant standard deviation of the 32-realisation mean, the ratio at a single realisation is:

| corruption level p | masked positions | (A) mutation-anchored | (B) full sequence, matched | (B) full sequence, unmatched |
| --- | --- | --- | --- | --- |
| 1/L (deterministic) | 1 | 0.000 / 0.000 | — | — |
| 0.05 | 4 | 0.164 / 0.128 | 3.019 / 4.869 | 9.228 / 7.300 |
| 0.15 | 11 | 0.320 / 0.255 | 2.128 / 5.190 | 8.501 / 10.901 |
| 0.3 | 22 | 0.648 / 0.400 | 1.817 / 4.332 | 6.124 / 7.648 |
| 0.5 | 36 | 1.264 / 0.516 | 2.445 / 3.216 | 7.332 / 5.721 |
| 0.8 | 58 | 0.499 / 0.698 | 3.245 / 2.988 | 8.601 / 3.697 |

Each cell gives `1O6X.pdb` / `1UBQ.pdb_L67S`. The "unmatched" column breaks the pairing between the mutant's realisation and the wild type's while holding the set of realisations fixed, which is what happens when a variant and its wild type are drawn separately; it inflates the per-realisation noise by a factor of **1.24 to 4.00** across the ten cells while leaving the many-realisation mean where it was, because that construction permutes rather than redraws. A design that redraws the wild type independently for each variant would additionally carry the wild-type term's own variance into every variant's estimate. **Matched realisations are therefore necessary and were verified to be matched, not assumed.**

Translating the ratio into the units the programme resolves is what decides admission. A score entering a fit as one scalar column with classical additive sampling error attenuates the increment it can produce by approximately 1/(1 + (σ_seed/σ_signal)²/N) at N realisations. For the deterministic member that factor is exactly 1. For (A) at p = 0.05 and N = 1 it is 0.974 and 0.984. For (B) at N = 1 it runs from **0.036 to 0.232**, so an increment of the size the stability gate actually resolves — +0.00764 to +0.02260 kcal²/mol² — would read as 0.00027 to 0.00525 kcal²/mol², at or below the smallest quantity that gate resolves. Bringing the sampling noise to a tenth of the between-variant spread needs **2 to 11 realisations for (A) at p ≤ 0.15**, rising to 160 at p = 0.5, and **331 to 2,694 realisations for (B)**.

### Verdict on the score

The mutation-anchored contrast qualifies. Its p = 1/L member is exactly reproducible, is matched by construction, needs one forward pass per background and mutated position, and attenuates nothing; its low-corruption members at p ≤ 0.15 qualify at 2 to 11 realisations. The full-sequence contrast **does not qualify at any affordable cost**, and the cost section below puts a number on that. This is the uncomfortable part of the finding and should not be smoothed over: the member that qualifies is the single-position denoising readout, which is the masked-marginal shape, while the member that is distinctively about the diffusion paradigm — the model's assignment to a whole sequence — is the one out of reach.

One further property is a choice and not a fact about the model: the corruption level changes what the score measures, and it does so unevenly across backgrounds. On this slice the deterministic member's rank correlation against `ddG_ML` degrades with corruption on one background (+0.606 at p = 1/L to −0.113 at p = 0.8) and holds or rises on the other (+0.365 to +0.414). A corruption level selected on one cohort should not be expected to transfer to another, which is the same lesson the readout reassessment has been recording about class and depth.

## Cost

Throughput was measured on one H200 in the 1-GPU pod, in fp32, on the DPLM-3B computation graph. The weights used were the `esm2_t36_3B_UR50D` checkpoint already staged on shared storage, whose config matches `airkingbd/dplm_3b` on every shape-determining field — 36 layers, hidden 2,560, intermediate 10,240, 40 heads, rotary positions, vocabulary 33 — so the graph and the shapes are the same and only the parameter values differ, which do not enter a timing. That substitution is declared rather than inferred. At batch 64 the measured rates are **106.7 sequences/s at length 72, 30.7 at 256, 20.1 at 384 and 7.1 at 1,022**; at length 72 batch 8 and 32 give 77.7 and 95.5. The four pods were checked immediately before dispatch and all sixteen cards reported 0 MiB in use and 0% utilisation; only the 1-GPU pod was used and nothing held by the four-card pods was claimed.

Under the deterministic member the cohorts cost almost nothing per checkpoint:

| cohort | forward passes | measured cost |
| --- | --- | --- |
| 25,856 single-mutant stability states over 5,664 mutated positions, lengths 37–72 | 5,664 | 53 s, 0.015 GPU-hours |
| 25,728 anchor variants, at most one pass per assay and position | ≤ 25,728 | 0.36 GPU-hours if every pass ran at length 384, 1.01 at 1,022 |
| 8,192 cycle states, first-order control only, 217 site pairs | ≤ 434 | 4 s |

Under the full-sequence member at its own qualifying N, the same stability cohort costs **22.4 GPU-hours at N = 331 and 182.0 GPU-hours at N = 2,694, per checkpoint** — 1,500 to 12,000 times the deterministic member for one cohort and one model. Cost is therefore not what blocks the qualifying score and is decisive against the non-qualifying one.

One interface limit belongs with the anchor figure. DPLM declares 1,026 positions. ProteinGym's wild-type lengths on this host have median 245, quartiles 69 and 536 and maximum 3,423, with 81 of 217 assays above 384 and **16 of 217 above 1,022**, so a DPLM arm would carry its own support subset of 201 assays for the same reason other arms carry theirs.

## The canonical state, declared

A diffusion model has no single hidden state, because the state is a function of the corruption level and of the mask realisation. The canonical state declared here is: **the uncorrupted input (p = 0, no absorbing token anywhere), fp32, all 34 hidden states retained, mean-pooled over residue positions with the two terminal tokens excluded, computed one sequence at a time.** It is chosen because it is the only corruption-free state, it is exactly reproducible, and it is the state the DPLM representation-learning results are read from. It is also off the diffusion trajectory, which is the honest cost of the choice.

How much that choice matters was measured on DPLM-650M, as the cosine between the pooled state at each level and the pooled state at p = 0, and as the cosine between realisations within a level:

| level | layer 17 of 34, to p = 0 | layer 17, within level | layer 33 of 34, to p = 0 | layer 33, within level |
| --- | --- | --- | --- | --- |
| 0.05 | 0.99977 | 0.99791 | 0.99970 | 0.99659 |
| 0.15 | 0.99874 | 0.98119 | 0.99501 | 0.97028 |
| 0.3 | 0.99752 | 0.99435 | 0.99275 | 0.99048 |
| 0.5 | 0.97330 | 0.89248 | 0.91716 | 0.91089 |
| 0.8 | 0.91332 | 0.88046 | 0.83117 | 0.94211 |

So a state at heavy corruption is a materially different vector from the canonical one, and a representation verdict computed at one level is conditional on that level exactly as today's verdicts are conditional on readout class and depth. **A canonical state is a declared choice with a measured magnitude, and it is not "the model's representation".** No representation readout should be fitted from this line until the readout reassessment closes, and when it does, the corruption level joins class and depth as a factor that must be varied one at a time.

## Which conclusions have a comparable interface

| conclusion | comparable interface | what is different or missing |
| --- | --- | --- |
| One: the first-order capability boundary | Yes, on 201 of 217 assays. The endpoint is a rank correlation over variants and the model side is one scalar column, so the profile and substitution-table comparators stay defined. | The column is a denoising contrast, not a sequence likelihood; nothing that reads nats per token has a counterpart. |
| Two: local sequence and single-residue constraints | Yes. The score enters the crossed-control fit as one scalar column on the same cohort and splits. | The 14-of-33 count stays as it is; a DPLM row is recorded separately. |
| Three: combinatorial residue dependence | Partly. The first-order single-mutant control that actually carries the conclusion has a direct counterpart at qualifying precision. | **Not applicable for the four-state interaction contrast.** That contrast is a double difference of four full-sequence scores, and the only full-sequence native score measured here fails determinism at any affordable N. The alternative — a two-position joint obtained by ordered unmasking — is a new estimator with no counterpart in the autoregressive panel, so the comparison would be between different estimators and would have to be qualified in its own right first. |
| Four: biophysics, and contact specificity | Yes for folding and stability: 25,856 single mutants, 5,664 forward passes, the same control set and the same three split seeds. | The contact gate stopped at part 1 for every arm, so there is nothing to compare. Any structure quantity inherits the folding instrument's licensed scope. |
| Five: biological context and distance | Yes for the retrieval and identity strata, which restratify scores already computed and need no new interface. | The supplied-homolog regime is different in kind: an autoregressive prefix against concatenation into a bidirectional 1,026-position window. It would need its own qualification and is marked partly comparable. |
| Six: generation and control | In kind, yes, and this is where the paradigm actually differs. | Two declared differences: the length is supplied rather than emitted, so the contiguous-corpus-fragment control's length matching has to be redefined, and there is no prefix, so the censored-prefix column of the attempt ledger has no counterpart. Class-conditioned generation has no counterpart at all, since DPLM takes no class label. **And the sampler is the one part that is not a bare forward pass**: reproducing DPLM's published generation would mean re-implementing a denoising-and-remasking loop whose reference code sits behind a blocked direct route, so anything produced that way is a re-implementation and should not be reported as DPLM's generation. |

## Recommendation

**A full pass across the panel is not worth spending; a bounded two-endpoint exploratory extension is, and only with a matched control arm.**

The compute case for the qualifying score is overwhelming — under one and a half GPU-hours for both scoring cohorts at both DPLM sizes — so cost is not what should decide this. Three things should.

First, the qualifying member is a single-position denoising readout, which is the masked-marginal shape, and the panel contains no masked language model. Two exploratory measurements bear on what that would buy, both on three MegaScale backgrounds and 4,063 variants with no interval and no claim attached: DPLM-650M's deterministic contrast agrees with the ESM-2-650M checkpoint it was initialised from at Pearson +0.656 and Spearman +0.658, and reaches +0.403 against `ddG_ML` where ESM-2 reaches +0.391; DPLM-3B agrees with DPLM-650M at Pearson +0.902 and Spearman +0.906, and reaches +0.400. Read plainly, continued diffusion training moved the score substantially away from its masked initialisation without moving this endpoint. **Without ESM-2 arms alongside it, a DPLM row cannot be read as evidence about discrete diffusion rather than about bidirectional masked scoring**, and adding those arms costs the same negligible amount.

Second, the object that would make this a paradigm comparison rather than another scorer — the model's own assignment to a whole sequence — is exactly the one that fails qualification, by one to two orders of magnitude on required realisations.

Third, no sequence–structure generative arm is executable here. La-Proteina has no scoring interface and no mutant structures to feed one; DPLM-2's coordinate encoder needs the runtime that must stay out of `ct`. If the sequence–structure slot is what the programme wants, the work to do first is not an arm but an endpoint: nothing in the three cohorts supplies the structures such a model would have to consume.

Concretely, if this line is pursued: admit `airkingbd/dplm_3b` and `airkingbd/dplm_650m` to a separate exploratory roster together with `esm2_t36_3B_UR50D` and an ESM-2 650M rung as matched masked-scoring controls; use the deterministic p = 1/L member only; run the mutation-ranking and single-mutant stability endpoints only; do not run the cycle interaction endpoint, and record it as not applicable with the reason above; fit no representation readout until the readout reassessment closes, and then only under the declared canonical state with the corruption level treated as a varied factor. That is roughly 1.5 to 3 GPU-hours in total and it changes nothing in the admitted panel. If the six conclusions are the deliverable and the panel is closing, the honest answer is that this extension would add a row to two of them and would not change any of them, and it can be skipped without loss.

The bounded extension above was authorized on 2026-09-25 and cancelled the same day, before it produced any record, when priority moved to closing Direction-1-a. Its detached driver and the single 1-GPU job it was running were terminated and the allocation released. Nothing from it was measured, admitted or written anywhere, and the two exploratory correlations quoted above remain the unqualified draw they always were: three MegaScale backgrounds, 4,063 variants, no interval. The manuscript's paradigm-coverage limitation rests on the interface finding in this document and on no quantity from this line, so the cancellation costs the paper nothing. Anyone resuming it should read the concrete roster above as a fresh authorization, not as work in progress.

## Provenance

Route probes, checkpoint downloads, load checks, the qualification measurement and the state measurement ran on Compute in the `ct` environment; the throughput measurement ran in the 1-GPU pod on shared storage. The exploratory slice was built directly from the MegaScale deposition held on this host and is not the frozen stability cohort; it carries no family grouping, no split contract and no resampling unit, and nothing derived from it is an estimate of anything the capability map reports. Working scripts and outputs are under the ignored `logs/dplm_feasibility/` directory, and the throughput artefact under the pod-side scratch path named in the experiment log entry for this date.
