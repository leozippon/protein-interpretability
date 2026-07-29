# TG series — transfer screen, text versus protein decoders

Ten entry points plus a positive-control stage and a collator. Every stage takes `--arm` and writes one JSON per arm under `results/transfer_gap_20260729_corrected/<stage>/`.

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ct
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd scripts/transfer_gap

python tg00_input_contract.py --arms protgpt2 progen2-medium --device cuda:2   # run first
python tg07_variance_behaviour.py --arm protgpt2 --device cuda:2
python tg99_summarize.py --root ../../results/transfer_gap_20260729_corrected
```

## Read this before quoting any number from `results/transfer_gap_20260724/`

That results tree was produced by a version of `tg_common.py` whose `protein_input` returned the plain amino-acid sequence for ProtGPT2, against a model pretrained on end-of-text-separated, 60-column-wrapped FASTA. The rendering is worth **1.78 nats/token** on a matched cohort. Every ProtGPT2 row in that tree is contaminated. The pre-correction snapshot and a per-number status table are at `archive/legacy/r2_transfer_gap_precorrection_20260729/`.

The corrected tree is `results/transfer_gap_20260729_corrected/`.

## Design rule this directory now enforces

**Rendering and model loading are not implemented here.** Both are imported from `src/transfer/arms.py`, which is the single declaration of what each panel member is fed. `tg_common.py` is an adapter, not a parallel implementation. The contamination above survived a withdrawal that covered other ProtGPT2 rows precisely because two renderers existed and only one was fixed.

`tg_common.py` does own **which draw**, but no longer **which records**. Every cohort here is a seeded permutation of the complete eligible set, and `skip` partitions that permutation rather than walking further down the corpus file — file order is a documented hazard of this programme worth up to +1.01 nats (EXP-R2-059, limitation L13). That used to require a local eligible-record enumeration, because `arms.protein_cohort` and `arms.text_cohort` drew in FASTA file order and offered no alternative. **They now take `seed=` and permute, and the local layer is gone**: `cohort_for` forwards, and the eligibility predicate is the panel's single copy.

Nor does it own the **depth grid** or the **artefact writer** any more. `analysis_layer` and `write_json` are re-exported from `src/transfer`. Both replaced local copies, and the depth conversion existed here six times in two mutually inconsistent forms — four `int(round(...))` (round-half-to-even) and two `floor(... + 0.5)` (round-half-up). They agree on every fraction any TG stage actually passes, so unifying them moved no recorded number; they would not have stayed that way.

## The stage contract

`tg_contract.py` declares, per stage, the arms it measures, the protein residue band it draws on, and the cohort seed — and `--verify` reads each entry point's `argparse` defaults back out of the source and refuses if any of them disagrees. `tests/test_transfer_gap_contract.py` runs it.

```bash
python tg_contract.py --verify   # 12 TG stages agree with tg_contract.py
python tg_contract.py --json     # the resolved contract
```

It exists because the TG series carried the defect `scripts/transfer/panel_contract.py` was built to close, unfixed. The stages draw protein cohorts on **three different bands** — 400–1000 (TG-01, TG-02, TG-06), 120–1000 (TG-03, TG-07, TG-08, TG-09), 64–246 (TG-10) — with none of them declared. Individually those are compute choices; together and silent, they let TG-01's information budget and TG-10's causal headroom read as two measurements of one cohort when they share no protein at all. That is Appendix B rule 13, and EXP-R2-060 prices protein cohort-block sensitivity at 0.16–0.60 nats.

Running it for the first time found two more (EXP-R2-066). Nine stages restated `DEFAULT_COHORT_SEED` as a literal, and one restated it **wrong**, as the pre-correction `20260724` — which makes `skip` a non-partition against every other stage. And `tg01_information_budget.py` registered `--seed` **twice** on one parser, so `argparse` raised on construction and **that stage could not run at all**. Nothing in the repository read those arguments until this file did.

## Stage status

| stage | what it measures | status |
|---|---|---|
| TG-00 | rendering and cohort positive controls | **new**; run before anything else |
| TG-01 | predictive-information budget, truncation curve, Markov ladder | corrected; **partly superseded** by `scripts/transfer/01_cohort_power.py` |
| TG-02 | far-context information split into order and composition | corrected; **unique to this series** |
| TG-03 | matched TopK dictionary fidelity | corrected and **re-run** |
| TG-04 | explanation-channel capacity | corrected; **superseded** by `src/transfer/channels.py` + `scripts/transfer/06_explanation_channel.py` |
| TG-05 | relational structure: per-position states versus attention | corrected; **superseded and retracted** — use `scripts/transfer/05_relational_channel.py` |
| TG-06 | price of freezing the attention pattern | corrected; **unique to this series** |
| TG-07 | variance versus behavioural fidelity | corrected and **re-run**; see below |
| TG-08 | dictionary budget sweep | corrected; **unique** (source of L3) |
| TG-09 | depth profile of the TG-07 statistic | corrected; **unique** |
| TG-10 | causal headroom of the P0-2b estimand | corrected; **superseded** by `scripts/transfer/03_estimand_power.py` |

"Superseded" means a better-controlled implementation exists in the current `src/transfer` package and should be preferred. The stage is kept and corrected rather than deleted, because a superseded script that still runs is a script that will still be run, and leaving it wrong is worse than either fixing or removing it.

## Invariants every stage now holds

1. **Native rendering.** Input strings come from `Cohort.input_strings`, never from a local branch on arm name.
2. **Seeded cohorts.** No stage takes the first N records of anything. Every artefact records its cohort digest, seed and eligible-record count.
3. **Declared denominators.** `loss_recovered` and every share divide by a quantity that is checked against a floor first, and return `None` with a reason when it fails. The panel's mean-ablation headroom spans 0.02 to 7 nats/token and has been measured negative; a ratio against the bottom of that range is not a weak measurement.
4. **Held-out estimators.** Information decompositions use held-out cross-entropy. The plug-in value is reported beside it so its bias is visible.
5. **Position accounting.** A native rendering puts separators and conditioning tags into the scored stream. Cross-entropies are reported over all scored positions and over alphabet-bearing positions separately.
6. **Loud instrument failures.** TG-06 asserts its monkeypatch reached the model; TG-05 refuses a conditioned arm it has no label for.

## TG-07, the load-bearing result

The variance–behaviour dissociation motivating a reliance-weighted-dictionary method line **does not reproduce as recorded**. Rank-512 loss recovered for ProtGPT2 moves from −0.105 to **+0.879**, and its alignment gap (variance explained − loss recovered) at rank 512 falls from +1.102 to +0.118 — below ProGen2-medium's +0.326. GPT-2-large is unchanged to three decimals, which is what confirms the movement is the rendering and not the cohort.

Separately, the variance-concentration premise was mismeasured for every arm. `PC1` and the participation ratio over all positions are dominated by structurally special tokens — GPT-2-large's first token at 24.8x the interior residual norm, ProtGPT2's FASTA newlines at 2.5x. On interior, alphabet-bearing positions the participation ratio reads 253 (gpt2-large), 4.9 (ProtGPT2), 577 (ZymCTRL) and 3.7 (ProGen2-medium): the protein arms do not cluster, and the highest effective dimension in the panel is a protein arm. Report the subsetted spectrum, never the all-position one.
