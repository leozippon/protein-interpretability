# R2 — text-to-protein interpretability transfer

The only live research root. It measures how mechanistic-interpretability
methods developed on text decoders behave when applied to protein generative
models, and what that says about the methods.

## Objective

1. **Differences.** Compare language-generative and protein-generative models.
2. **Transferability.** Analyse the limitations of text-derived interpretability
   methods applied to protein generative models.
3. **Adapted methods.** Design and validate protein-adapted methods.

Parts 2 and 3 are the deliverable; part 1 is instrumental.

## Canonical document

`docs/INTERPRETABILITY_TRANSFER_AUDIT.md`. It is the single source of truth for
what has been measured, what survived and what was withdrawn. **Where this file
or any other document disagrees with it, it wins.**

Read at minimum: §1 (the defensible position today), §3 (status of every claim),
§5 (the limitation catalogue L1–L19), §9 (the plan), and **Appendix B**, which
holds fourteen standing methodological rules — each earned by a failure in this
programme. Do not touch measurement code before reading Appendix B.

## Where the programme stands

- **Part 1 (differences).** One measured difference survives, in weakened form:
  on synthetic repeat probes, protein decoders have fewer heads in the upper
  tail of the induction-score distribution. It is not a distributional
  difference, it inverts at the headline threshold on the most defensible probe,
  and its modality increment is carried by a single protein model. It is not a
  modality finding. Pathway budget (MLP-to-attention ablation-cost ratio)
  separates text from protein with non-overlapping ranges and is the strongest
  part-1 result, but carries two caveats and is not yet a claim.
- **Part 2 (limitations).** Nineteen catalogued limitations, most of them
  demonstrated **on the text control** — which makes them properties of the
  method rather than of protein models. This is where the programme's yield is.
  The strongest single claim available is L5: the head-prevalence census has a
  narrow, previously unmeasured domain of applicability.
- **Part 3 (adapted methods).** Not yet earned. Two candidate directions carry
  standing rejections (§7) and no proposal is currently traceable to a measured
  limitation with adequate evidence.

Retracted, and not to be asserted anywhere: the induction gap as a *modality*
claim; the variance–behaviour dissociation; the QK/OV dissociation; the
relational channel; the tokenisation explanation; peak prefix-matching strength
as a memorisation-free statistic; the gpt2 / dialogpt-small corpus contrast.

## Panel

Eleven autoregressive decoders under one code path, each fed in the format it
was trained on: gpt2, gpt2-medium, gpt2-large, gpt2-xl, dialogpt-small,
qwen2.5-0.5b, llama-3.2-3b (text); protgpt2, zymctrl, progen2-base,
progen2-medium (protein). The matched pair is gpt2-large / protgpt2 — identical
depth, width, vocabulary size and parameter count. See the audit document §2 for
the identifying contrasts and the known structural limit.

## Layout

| Path | Contents |
|---|---|
| `docs/` | Canonical audit, research plan, experiment log, methods, dated analyses. |
| `src/transfer/` | The whole measurement library, and the whole of `src/`. Four *declarations* — `arms` (the panel, its cohorts and the one renderer), `scoring` (depth grid, scored-target rule, aggregation), `statistics`, `io` — and twelve measurement modules: budget, pathways, circuits, path_patching, prediction_addressed, lenses, probes, channels, relational, homology, induction_robustness, scaling. |
| `scripts/transfer/` | Campaign entry points `NN_*.py`, the H200 controller/worker, and `panel_contract.py` — the single declaration of which arms each stage may run, and why not. |
| `scripts/transfer_gap/` | The TG transfer screen, `tg_common.py` (an adapter over `src/transfer`, not a parallel implementation), and `tg_contract.py` — the single declaration of each TG stage's arms, residue band and cohort seed. |
| `evidence/` | Compact synchronized receipts. Synced; small by construction. |
| `configs/` | Model and training configurations. |
| `results/` | Generated artefacts. B-only, ignored by git. |
| `logs/` | Runtime logs. B-only. |
| `tests/` | `PYTHONPATH=. python -m pytest tests -q`. |

`src/` contains nothing but `src/transfer/`, and that is checked rather than
asserted: the H200 controller derives the code freeze by statically walking every
`src.*` import from `scripts/transfer/` and `src/transfer/`, and reports the
closure it found. It currently adds exactly one file outside those two
directories, `src/__init__.py`.

## Live evidence

| Pointer | What it establishes |
|---|---|
| `results/transfer_20260728/` | The 2026-07-28 campaign: pathway budget, estimand power, circuit primitives, convergence control, lens family, probes and erasure, homology control, path patching, induction robustness, PAA census. |
| `results/transfer_20260729/` | Homology / memorisation control re-run with `--masking 0` on full UniRef50 (EXP-R2-064, audit §0.05). |
| `results/transfer_20260729_instrument/`, `…_instrument_skip4000/` | Eleven-arm instrument-transfer campaign and its cohort-skip sensitivity arm (EXP-R2-060, EXP-R2-063; audit §5.05). |
| `results/transfer_gap_20260724/` | The TG series. Retained unmodified because the audit document cites it. |
| `results/transfer_gap_20260729_corrected/` | TG re-run after the rendering and cohort corrections (EXP-R2-062, audit §0.1). |
| `results/final_checkpoints/` | Four CLT checkpoints, 2026-04-03, `step_100000` — ProtGPT2, ZymCTRL, ProGen2-medium, ProGen2-xlarge. The only protein dictionaries held locally; an input to plan item C1. Training logs in `logs/r2_clt_*_20260402*`, manifest in `evidence/historical_reference_checkpoints_20260717/`. |
| `evidence/p0_2_adjudication_20260727/`, `evidence/p0_2b_fidelity_20260727/` | The P0-2 eligibility receipt and the P0-2b behavioural qualification — the evidence behind limitation **L1**, and the baseline that plan item **C1** re-qualifies. `p0_2b_fidelity_spec.executed.json` records all 27 dictionary checkpoint paths and SHA-256 digests on GPFS. |
| `evidence/p0_2_screening_20260722/`, `p0_2_full_launch_20260722/`, `p0_2_mask_validation_20260721/` | The provenance chain of those 27 runs, including the three terminal `sparsity_match_failure` model/methods that account for the nine intentionally absent runs. |
| `docs/analysis/P0_2B_DICTIONARY_FIDELITY_RESULTS_20260727.md` | The P0-2 / P0-2b results and their claim limits, in prose. |

## Running a campaign

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ct
# after any edit to src/transfer/arms.py:
python scripts/transfer/panel_contract.py --emit
# after any edit to a TG stage's argparse defaults:
PYTHONPATH=. python scripts/transfer_gap/tg_contract.py --verify
PYTHONPATH=. python -m pytest tests -q
```

See `scripts/transfer/README.md` for the H200 launcher. Check `nvidia-smi`
inside the pod before scheduling; cluster-wide allocation is not pod
utilization.

## Retired scope

The prior programme in this directory — cross-model conserved sparse-readout
atlases, EC-conditioned steering, enzyme design, CLT training, and the npj
Artificial Intelligence manuscript — is archived with its provenance at
`archive/legacy/r2_retired_scope_20260729/`. It reached its own terminal
negatives and is not being resumed. Do not build on it, and do not resurrect its
claims.

That archive holds the 79 numbered `scripts/`, `src/{models,training,analysis,revision}/`
and 25 test files, each with the reason it was archived and where its output
still lives. Nothing was deleted. The boundary was the **computed** import
closure of the objective's entry points, not a judgement about which directories
looked relevant (EXP-R2-066).
