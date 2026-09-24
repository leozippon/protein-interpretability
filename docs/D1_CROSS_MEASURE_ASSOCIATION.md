# Direction-1 cross-measure association

**Completed, 2026-09-21.** The frozen roster ran to a terminal state on CPU. The inventory resolved 176 available and 22 named unavailable axis cells with no seat left pending. Of the 51 declared pooled pair ids, 34 returned a coefficient; the seventeen conditional pairs `P-G1-C2`…`P-S3-C2` closed at n ≤ 1 and therefore without one, as the uncertainty rule below requires. Within-lineage coefficients closed at 34, 25, 25 and 17 of the same 51 ids for ProGen2, Galactica, ProGen3 and Llama-2 → ProLLaMA.

Three seat resolutions differ from the frozen table. The four eight-block envelopes recorded below as not located became named unavailable G1 cells for ProtGPT2, ZymCTRL, ProGen2-small and ProGen2-base, which lowers the n of every G1 pair. The Galactica and InstructProtein MegaScale cells resolved unavailable after their natural control failed, so the reserved pending-S3 pair ids hold no seat and S3 closed on the readable arms alone. Galactica-125M's U4 and U5 cells are named unavailable. ESMFold2 values replaced the retired v1 inputs for C3/C4 and for ProGen3-3B's U3/U4.

The coefficients are in `results/transfer/s50_cross_measure/cross_measure_association.json` (sha256 `72edbbef9c898b6853b3b97b0b04e655ca0803b9ab69673009565b8ea023d4e2`) and in `summary.md`. Consistent with the prohibition at the end of this file, they are not copied here; this document remains the frozen design.

Date: 2026-09-18 (Asia/Shanghai). Campaign: EXP-R2-247. This document freezes the association among the context-information gate, external sequence scoring, generation endpoints, and declared parameter count **before** remaining MegaScale Galactica / InstructProtein scores, homologous-context expansion scores, and EXP-R2-246 unconditional-generation numbers are written into it. It does not change [EXP-R2-232](D1_GENERATION_BIOLOGY_PREREGISTRATION.md), [EXP-R2-233](D1_PROGEN3_GENERATION_PREREGISTRATION.md), [EXP-R2-246](D1_UNCONDITIONAL_GENERATION_EXPANSION.md), or any already-recorded cell.

No coefficient is computed in this file. Homologous-context AUROC is scoring-with-context and is not a generation endpoint; stage 46 numbers do not enter any pair below.

## Question

After every currently queued Direction-1 cell that this association names has a terminal state, what is the rank association among the four axes, within each declared lineage and on the pooled protein/joint set? The pooled coefficient is descriptive. It is not a parameter-count effect. Galactica ProteinGym already forbids a parameter-count causal claim (`no_parameter_count_causal_claim` on `native_dms_comparison.json` sha256 `38cb2b5dd549ef3e4b9f9fbb9e362a025c9a61f7ac94b1e9c204bc48b6ff7d84`); that prohibition is not reopened.

## Units

No composite score is formed. Each axis is one already-recorded quantity.

### Context-information gate

Two recorded objects, not a third:

1. **Tightest-of-eight lower bound.** Minimum, over the eight blocks, of the displacement-corrected near-duplicate-group bootstrap 95% confidence-interval lower bound on context information. Unit: nats per scored token. Identification uses this same family of bounds: a checkpoint is identified if and only if every one of the eight corrected lower bounds is strictly greater than zero. The published eight-block envelope `[min lower, max upper]` uses this minimum as its left endpoint; that left endpoint is the continuous quantity, not a pooled 95% interval.
2. **Identified / not identified.** Binary companion label from the same eight bounds. An unidentified gate remains a completed measurement and does not drop the checkpoint from later pairs.

Where those numbers live:

- Extra-model protein/joint cells (EXP-R2-240): combined stage-41 report `results/external_baseline/20260916033459_68af0c6949cd/r240_stage41/context_information_bootstrap.json` and `identification_table.json`. The quoted envelopes are in the 2026-09-16 experiment-log entry “EXP-R2-240 combined stage 41 report” and in `summary.md` §共同测量前提. Protein-mode envelopes already quoted there include Llama-2-7B `[+0.0452,+0.1178]`, Galactica 125M `[−0.1873,−0.0660]` (not identified), 1.3B `[+0.0304,+0.0947]`, 6.7B `[+0.1196,+0.2129]`, 30B `[+0.2452,+0.4762]`, ProLLaMA Stage 1 `[+0.4868,+0.6424]`, Stage 2 `[+0.4598,+0.6040]`, ProGen2-large `[+1.1115,+1.4309]`, xlarge `[+1.4891,+1.8276]`, ProGen3 112M `[+0.8276,+1.1306]`, 3B `[+1.5228,+1.9188]`, ProteinGLM-7B-CLM `[+1.5648,+1.8978]`, RITA-xl `[+1.1942,+1.5878]`, InstructProtein protein `[+1.8885,+2.2812]`. Joint text-mode envelopes are not this axis.
- ProtGPT3-1.3B native format: stage-41 report sha256 `105efe5631ddbd5d78794bb95605cd0555cefe6768d4f8c8b58e271aa33be869` on freeze `20260917222641_07fae5c1614a`. Tightest lower bound **+0.8563327672967896** on block 7; identified. This does not rewrite the EXP-R2-240 raw-format envelope.
- ProGen2-medium / large / xlarge under EXP-R2-224: `results/transfer/r224_stage41/context_information_bootstrap.json`, min-lower-to-max-upper `+1.1712` to `+1.4783` / `+1.1813` to `+1.5173` / `+1.5291` to `+1.8406`. Those sidecars are not the EXP-R2-240 01 design. For large and xlarge the association reads the EXP-R2-240 envelope, not this earlier ladder report.
- Original fifteen-arm protein cells (ProtGPT2, ZymCTRL, ProGen2-small / base / medium): identification (all eight blocks, displacement-corrected sign rule) is recorded as identified in `summary.md` §共同测量前提. A published eight-block envelope of the same form as EXP-R2-240 was **not located** in tracked documents for ProtGPT2, ZymCTRL, or ProGen2-small / base. EXP-R2-216 E1 is one block (skip 0). E2 is a between-block mean of point estimates, not a minimum of eight corrected lower bounds. EXP-R2-221 states that displacement correction changed 0 of 120 fifteen-arm verdicts; it does not reprint the eight corrected lower bounds. The runner reads the original stage-41 sidecars if they are present at analysis time and records an unavailable G1 cell if they are not. ProGen2-medium’s continuous G1 value is the EXP-R2-224 minimum `+1.1712` until an 01-design eight-block envelope is located for that rung.

Earlier stage-21 single-draw bounds (for example Galactica-1.3B protein `+0.038694`, Llama-2-7B protein `+0.063641`) are not this quantity.

Nats per scored token are not cross-tokenisation comparable. The association is a rank correlation, not a difference of nats.

### External sequence scoring

ProteinGym and MegaScale stay two quantities. They are never averaged or otherwise merged.

- **S1.** Family-bootstrap MODEL − LOOKUP, dimensionless Spearman difference (family-mean ρ_model minus ρ_LOOKUP). Unit: Δρ. Source: each arm’s analyse artefact as already cited in `summary.md` §外部蛋白任务上的序列评分 and the 2026-09-17 experiment-log ProteinGym entries. Supports remain the arm’s own analysis set (201/163, 213/171, 214/172, or 217/174) and are never intersected to manufacture a common n.
- **S2.** Family-bootstrap MODEL − BLOSUM62 on the same artefact, same family bootstrap, same support. Unit: Δρ. Source: the 2026-09-17 entry “Add MODEL − BLOSUM62 beside the ProteinGym MODEL − LOOKUP row” and the same analyse files.
- **S3.** MegaScale **design-side** Spearman ρ of model likelihood against measured stability, **only where the natural control passed**. Unit: ρ. Readable negative findings already on disk: ProtGPT2 `+0.0837`; ProGen2-base / medium / large / xlarge `+0.0456` / `+0.0280` / `+0.0519` / `+0.0480`; ProGen3-3B `+0.1189`; ProteinGLM-7B-CLM `+0.1395`; RITA-xl `+0.0332` (`summary.md` MegaScale row; 2026-09-18 “MegaScale: fill remaining non-text arms”). Design-side numbers whose natural control failed are recorded and are not S3. Galactica and InstructProtein MegaScale scores are not in this file.

Raw ProteinGym Spearman is not an association axis. Winning LOOKUP or BLOSUM62 is not biological knowledge.

### Generation

Only endpoints already named in `summary.md` §条件生成 and §无条件生成. Conditional and unconditional stay separate tables. No “generation quality” index.

| Code | Quantity | Unit | Conditional table | Unconditional table |
| --- | --- | --- | --- | --- |
| C1 / U1 | Completion or native-terminal rate | attempts / attempts | **Not a row** in §条件生成; C1 is unavailable | Interface-accepted or native-closed outputs over 800 retained attempts. ProGen3-3B: 501/800 accepted, 299 budget-censored prefixes |
| C2 | Target-profile hit rate, requested condition, all-attempt | hits / attempts | ZymCTRL 2,488/2,800; ProLLaMA Stage 2 380/3,000 | Undefined (no requested class) |
| U2 | Any-Pfam recognition rate | hits / attempts | Not the conditional target-profile row | ProGen3-3B: 496/800; remaining arms pending EXP-R2-246 |
| C3 / U3 | Generation-minus-own-shuffle mean CA-pLDDT | pLDDT points (0–100 scale difference) | Pending ESMFold2 re-fold and re-attained natural/shuffle calibration | Pending ESMFold2 re-fold; remaining arms pending EXP-R2-246 |
| C4 / U4 | All-attempt confidence-event point estimate | dimensionless rate | Pending ESMFold2 re-fold | Pending ESMFold2 re-fold; remaining arms pending EXP-R2-246 |
| C5 / U5 | Median identity among UniRef50 hits | percent identity | Requested hits: 70.00% and 32.05% | ProGen3-3B 43.71% among aligned; remaining arms pending |

Requested-minus-mismatched selectivity is in the conditional table and is **not** an association axis (it is not in the endpoint list above). A budget-censored prefix is not a complete product. A Pfam or profile hit is not function. Median identity is computed among hits; a no-hit is not zero identity.

### Declared parameter count

Integer parameter count already declared on the model card or in the repository arms table, not a fitted size index.

| Checkpoint | Declared count | Kind | Source |
| --- | --- | --- | --- |
| ProtGPT2 | 774,030,080 | dense | `src/transfer/arms.py` (unit-matched to gpt2-large) |
| ZymCTRL | **not located as an integer** | dense, 36 × 1,280, residue vocab 458 | Shape in `arms.py` / `summary.md`. Do not copy ProtGPT2’s 774,030,080 |
| ProGen2-small | 151.1M | dense | `arms.py`, EXP-R2-068 load check |
| ProGen2-base | 764,803,616 | dense | `arms.py` |
| ProGen2-medium | 764,803,616 | dense | `arms.py`: architecturally identical to base down to this count |
| ProGen2-large | 2,779.4M | dense | `arms.py`, EXP-R2-068 |
| ProGen2-xlarge | 6,443.6M | dense | `arms.py` |
| RITA-xl | 1,208,655,872 | dense | `arms.py` |
| ProGen3-112M | 112M (card name) | MoE, 8 experts, top-2 | `summary.md`. Total versus active not separately declared |
| ProGen3-3B | 3B (card name) | MoE, 8 experts, top-2 | `summary.md`. Total versus active not separately declared |
| ProteinGLM-7B-CLM | 7B (card name) | dense, 36 × 4,096 | `summary.md` |
| ProtGPT3-1.3B | about 1.33B **total**; active undeclared | MoE, 8 experts, top-2 | `summary.md` |
| Galactica 125M | about 125M | dense OPT, 12 × 768 | `summary.md` |
| Galactica 1.3B | 1.3B (card name) | dense, 24 × 2,048 | `summary.md` |
| Galactica 6.7B | 6.7B (card name) | dense, 32 × 4,096 | `summary.md` |
| Galactica 30B | 30B (card name) | dense, 48 × 7,168 | `summary.md` |
| InstructProtein | **exact integer not separately declared**; same reported shape as Galactica 1.3B | dense, 24 × 2,048 | `summary.md` |
| Llama-2-7B / ProLLaMA Stage 1 / Stage 2 | 7B (card name), shared 32 × 4,096 | dense | `summary.md` |

ProGen3 3B and Galactica 30B are not the same kind of size: sparse MoE versus dense. ProtGPT3’s declared total is not an active-parameter count. A pooled Spearman against this column is still descriptive and is not a scale law. Within-lineage coefficients use only that lineage’s declared counts.

## Population

Protein and joint **generative** checkpoints that have, or are queued to have, the relevant cell. Text-only checkpoints are out of every pair.

| Checkpoint | Gate (protein mode) | ProteinGym S1/S2 | MegaScale S3 | Conditional C2–C5 | Unconditional U1–U5 | Parameters |
| --- | --- | --- | --- | --- | --- | --- |
| ProtGPT2 | identified; G1 from original sidecars or unavailable | available | readable | out | pending EXP-R2-246 | 774,030,080 |
| ZymCTRL | identified; G1 as above | unavailable (217 skipped, no ρ) | unavailable (no EC on a design) | available (native EC) | out | integer not located |
| ProGen2-small | identified; G1 as above | available (201/163) | recorded, not S3 (natural control failed) | out | pending | 151.1M |
| ProGen2-base | identified; G1 as above | available (213/171) | readable | out | pending | 764,803,616 |
| ProGen2-medium | identified; G1 = R224 min `+1.1712` until an 01 envelope is located | available | readable | out | pending | 764,803,616 |
| ProGen2-large | identified; G1 from EXP-R2-240 | available | readable | out | pending | 2,779.4M |
| ProGen2-xlarge | identified; G1 from EXP-R2-240 | available | readable | out | pending | 6,443.6M |
| ProGen3-112M | identified; G1 from EXP-R2-240 | available | recorded, not S3 | out | pending | 112M card, MoE |
| ProGen3-3B | identified; G1 from EXP-R2-240 | available | readable | out | available (EXP-R2-233; do not regenerate) | 3B card, MoE |
| ProteinGLM-7B-CLM | identified; G1 from EXP-R2-240 | available (201/163) | readable | out | pending | 7B card |
| ProtGPT3-1.3B | identified native; G1 `+0.8563` | available (201/163) | recorded, not S3 | out | pending | ~1.33B total, active undeclared |
| RITA-xl | identified; G1 from EXP-R2-240 | available (201/163) | readable | out | pending | 1,208,655,872 |
| Galactica 125M | not identified; stays in later pairs | available (213/171) | pending s29 | out | pending | ~125M dense |
| Galactica 1.3B / 6.7B / 30B | identified; G1 from EXP-R2-240 | available (213/171) | pending s29 | out | pending | card names, dense |
| InstructProtein | identified protein; G1 from EXP-R2-240 | available (213/171) | pending s29 | out | pending | shape-matched 1.3B; exact count not located |
| Llama-2-7B | identified protein; G1 from EXP-R2-240 | available (217/174) | recorded, not S3 | out | out (scoring wrap / text floor, not an unlabelled generator) | 7B card |
| ProLLaMA Stage 1 | identified protein; G1 from EXP-R2-240 | available (217/174) | recorded, not S3 | out | pending (`Seq=<`) | 7B card |
| ProLLaMA Stage 2 | identified protein; G1 from EXP-R2-240 | available (217/174) | recorded, not S3 | available (native superfamily) | pending (`Seq=<`) | 7B card |

Galactica uses `[START_AMINO]`, InstructProtein uses the protein block without an instruction slot, ProLLaMA uses `Seq=<`. Those are generation or scoring renders, not a licence to treat homologue-context AUROC as generation.

## Association, not cause

Statistic: Spearman rank correlation of two named quantities across checkpoints.

Uncertainty:

- n ≥ 5: Spearman, a checkpoint-bootstrap 95% interval (resample checkpoints with replacement), and the leave-one-out range of the coefficient.
- n = 4: Spearman and the four leave-one-out coefficients.
- n = 2 or 3: scatter, the rank coefficient, and n. No interval. At n = 2 the coefficient is only ±1 or undefined (ties).
- n ≤ 1: named single point or empty pair; no coefficient.

Within-lineage coefficients are reported separately from the pooled set:

- ProGen2 five rungs: small, base, medium, large, xlarge.
- Galactica four rungs: 125M, 1.3B, 6.7B, 30B.
- ProGen3 two rungs: 112M, 3B.
- Llama-2 → ProLLaMA three rungs: Llama-2-7B, Stage 1, Stage 2.

The pooled set is every protein/joint checkpoint that the pair admits. It is not a scale ladder. Depth, width, corpus, tokenisation, scoring direction, and analysis set co-vary. Galactica’s ProteinGym raw Spearman rises across rungs while every MODEL − LOOKUP interval stays below zero; that already forbids reading a parameter-count cause from a pooled or Galactica scoring–size coefficient.

G2 (identified / not) is a label printed beside G1. It is not itself a Spearman axis.

## Frozen pair list

A pair’s n is the number of checkpoints whose **both** cells are `available` or `pending` in the population table. A `pending` cell keeps its seat until it lands. If a pending cell becomes unavailable (interface failure, natural-control failure, empty payload), the pair is reported with that named drop. n is not quietly recomputed after seeing the other numbers.

Conditional pairs never share a table with unconditional pairs. S1 and S2 are never merged into one scoring number; they may each be associated with another axis.

### Pooled pairs

| Pair id | X | Y | Frozen n | Notes |
| --- | --- | --- | --- | --- |
| P-G1-S1 | G1 | S1 | 19 | ZymCTRL named unavailable for S1 |
| P-G1-S2 | G1 | S2 | 19 | same |
| P-G1-S3r | G1 | S3 readable | 9 | ProtGPT2; ProGen2-base/medium/large/xlarge; ProGen3-3B; ProteinGLM-7B-CLM; RITA-xl |
| P-G1-S3p | G1 | S3 pending | 5 | Galactica 125M/1.3B/6.7B/30B; InstructProtein. Coefficient after s29; natural-control failure becomes a named drop inside this pair |
| P-G1-P | G1 | parameters | 19 | ZymCTRL remains in the scatter with a missing parameter integer; it does not enter the coefficient until that integer is located |
| P-S1-S3r | S1 | S3 readable | 9 | association of two scoring families, not a merged score |
| P-S1-S3p | S1 | S3 pending | 5 | |
| P-S2-S3r | S2 | S3 readable | 9 | |
| P-S2-S3p | S2 | S3 pending | 5 | |
| P-S1-P | S1 | parameters | 19 | ZymCTRL out of S1; InstructProtein parameter integer missing → 18 in the coefficient, InstructProtein named drop on P |
| P-S2-P | S2 | parameters | 19 | same |
| P-S3r-P | S3 readable | parameters | 9 | all nine have a declared integer |
| P-S3p-P | S3 pending | parameters | 5 | InstructProtein parameter integer missing is a named drop on P inside this pair |
| P-G1-U1 … P-G1-U5 | G1 | U1–U5 | 18 | seventeen pending + ProGen3-3B. Llama-2-7B and ZymCTRL out of unlabelled generation |
| P-S1-U1 … P-S1-U5 | S1 | U1–U5 | 18 | |
| P-S2-U1 … P-S2-U5 | S2 | U1–U5 | 18 | |
| P-S3r-U1 … P-S3r-U5 | S3 readable | U1–U5 | 8 | readable S3 minus none of those eight is outside unlabelled generation |
| P-S3p-U1 … P-S3p-U5 | S3 pending | U1–U5 | 5 | same five Galactica / InstructProtein cells |
| P-P-U1 … P-P-U5 | parameters | U1–U5 | 18 | InstructProtein parameter integer missing is a named drop on P |
| P-G1-C2 … P-G1-C5 | G1 | C2–C5 | 2 | ZymCTRL and ProLLaMA Stage 2. C1 unavailable (not a §条件生成 row) |
| P-P-C2 … P-P-C5 | parameters | C2–C5 | 2 | ZymCTRL parameter integer missing → coefficient waits or is reported as n=1 named incomplete |
| P-S1-C2 … P-S1-C5 | S1 | C2–C5 | 1 | only ProLLaMA Stage 2 has both; ZymCTRL named unavailable for S1. No coefficient |
| P-S2-C2 … P-S2-C5 | S2 | C2–C5 | 1 | same |
| P-S3-C\* | S3 | C2–C5 | 0 | ZymCTRL unavailable for MegaScale; ProLLaMA Stage 2 natural control failed. Named empty |

S3-unreadable recorded design ρ (ProGen2-small, ProGen3-112M, ProtGPT3-1.3B, Llama-2-7B, both ProLLaMA stages) do not enter any S3 pair.

### Within-lineage pairs

Same X/Y codes, restricted to the lineage. Empty or n=1 lineage pairs stay named.

| Pair id prefix | Lineage | n on G1×S1 / G1×S2 / S1×P / S2×P | S3 | Unconditional U\* | Conditional C\* |
| --- | --- | --- | --- | --- | --- |
| L-PG2- | ProGen2 five | 5 | L-PG2-S3r: n=4 (small named not-S3) | n=5, all U pending | empty (no native class task in §条件生成) |
| L-GAL- | Galactica four | 4 | L-GAL-S3p: n=4 pending | n=4, all U pending | empty |
| L-PG3- | ProGen3 two | 2 | L-PG3-S3r: n=1 (112M not-S3). No S3 coefficient | n=2 (3B available, 112M pending) | empty |
| L-LLA- | Llama-2 → ProLLaMA three | 3 | all three not-S3. Named empty S3 | n=2 (Llama-2-7B out of U) | n=1 (Stage 2 only). No coefficient |

G1×P and S3×P / U×P / C×P inside a lineage use that lineage’s declared counts only. ProGen3’s two card names are both MoE; Galactica’s four are all dense. Those two lineages are not pooled into one size kind.

## Named unavailable or reserved cells

These seats are frozen now. They are not silent omissions.

- ZymCTRL × {S1, S2, S3, U1–U5}: completed refusals or population exclusions.
- Llama-2-7B × {U1–U5, C2–C5}: scoring wrap / text floor, not an unlabelled or native-class generator.
- Every text-only checkpoint × every pair.
- C1 on both conditional arms: completion/native-terminal rate is not a §条件生成 row.
- S3 on ProGen2-small, ProGen3-112M, ProtGPT3-1.3B, Llama-2-7B, ProLLaMA Stage 1, ProLLaMA Stage 2: design ρ recorded; natural control failed; not S3.
- S3 on Galactica four and InstructProtein: pending s29. Reserved pair ids `*-S3p-*`.
- U1–U5 on the seventeen EXP-R2-246 arms: pending. ProGen3-3B is not regenerated.
- Homologous-context AUROC (F16 and stage 46): not a pair, even after those numbers exist.
- Joint **text-mode** gate envelopes: not G1.
- ProteinGym raw Spearman: not S1.
- Requested-minus-mismatched: not C2.

A G1 that cannot be read as a tightest-of-eight corrected lower bound at analysis time is a named unavailable G1 cell for that checkpoint. It does not remove the checkpoint from S1, S3, U, or C pairs that do not need G1.

## What this document forbids

- Ranking models, publishing a leaderboard, or turning any coefficient into “better”.
- Claiming biology, function, folding, or knowledge from a rank association.
- Claiming that a larger model generates better, or that parameter count causes any gate, scoring, or generation difference. The pooled association is not a parameter-count effect. Galactica ProteinGym is not reopened as one.
- Treating gate identification as understanding, capability, or a licence to drop later cells.
- Merging conditional and unconditional endpoints, tables, or coefficients.
- Using homologue-context AUROC, or any other scoring-with-context statistic, as a generation metric.
- Inventing a composite “generation quality”, “scoring quality”, or “capability” index.
- Mixing ProteinGym and MegaScale into one number.
- Treating ProGen3 3B and Galactica 30B as the same kind of size.
- Filling this file with remaining MegaScale, stage-46, or EXP-R2-246 numbers as they arrive. Those belong in `summary.md`, the experiment log, and the analysis artefact this association writes later.

## Execution

CPU only, and only after `s29_galactica_instructprotein`, the stage-46 homologous-context expansion, and `s48` unconditional generation no longer occupy cards. This association does not take a GPU. It does not launch those queues.

The runner is `scripts/transfer/50_cross_measure_association.py` (`49_classifier_handoff.py` already occupies the 49 prefix). It prints every resolved path, every pair id, every named drop, and n before any coefficient, and it refuses to invent a missing cell. `--stage inventory` writes presence only. `--stage associate` computes a coefficient only for a pair whose frozen seats are all resolved; a still-pending pair stays pending.

s48 writes ledgers and the structure-parent draw. U2 needs HMMER, U3/U4 need the shared structure predictor, U5 needs the reconciled UniRef50 search. Those analysis cells are not invented here; a pair that needs them remains pending after generate.

## What could not be located

- A published eight-block displacement-corrected envelope (min lower bound) for ProtGPT2, ZymCTRL, ProGen2-small, and ProGen2-base under the same 01 design used by EXP-R2-240.
- An integer declared parameter count for ZymCTRL. The 36 × 1,280 shape is not that integer.
- An integer declared parameter count for InstructProtein distinct from “same shape as Galactica 1.3B”.
- Separate MoE **active** versus **total** parameter integers for ProGen3-112M and ProGen3-3B; only the card names and the 8-expert / top-2 routing are declared.
- A §条件生成 completion or native-terminal rate (C1).
- Any MegaScale Galactica or InstructProtein Spearman, any stage-46 homologous-context number, and any EXP-R2-246 unconditional-generation number. Those are pending or out of scope; they are not guessed here.
