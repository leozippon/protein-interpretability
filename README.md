# BioInterpretebility-CC

Mechanistic interpretability of protein generative models, measured against matched text decoders.

## Objective

One objective, three parts:

1. **Differences.** Compare language-generative and protein-generative models.
2. **Transferability.** Analyse how well existing interpretability methods — those developed on text decoders — transfer to protein generative models.
3. **Adapted methods.** Design and validate interpretability methods suited to protein generative models.

Parts 2 and 3 are the deliverable. Part 1 is instrumental: a difference between model families matters here only insofar as it explains why a method transfers badly.

Nothing else is in scope. There is no drug-design line, no enzyme-design line and no second paper.

## Canonical document

`docs/INTERPRETABILITY_TRANSFER_AUDIT.md` is the canonical findings, limitations catalogue and plan. **Where any other document disagrees with it, it wins.** It carries the programme's retractions (§0.05, §0.1) and qualifications (§5.05); a claim it has withdrawn must not be asserted as true anywhere else in this repository.

## Research roots

| ID | Scope | Home | Status |
|----|-------|------|--------|
| **R2** | Text-to-protein interpretability transfer | `` | **active — the only live root** |
| R0 | ProteinInterpret evaluation framework | `archive/retired_research_roots/` | retired 2026-07-29 |
| R1 | Encoder benchmark (ESM-2 SAE + IndelMissense) | `archive/retired_research_roots/` | retired 2026-07-29 |
| R3 | reserved | — | not assigned |

Retired roots keep their IDs and their full results trees so historical references stay resolvable. They are frozen provenance: do not build on, repair or resume them.

## Directory map

| Path | Purpose |
|------|---------|
| `` | **R2, the live research root.** Measurement library, campaign runners, results, evidence. |
| `data/` | Staged local datasets (Swiss-Prot, ProteinGym, AlphaFold, …). B-only, not synced. |
| `external_resources/` | Downloaded third-party baselines and tools. B-only. |
| `ops/` | Cluster, staging and operational helper scripts. |
| `docs/` | Repository-wide status, chronological log, navigation, naming conventions, dated audits. |
| `logs/` | Runtime logs for shared and cross-root operations. B-only. |
| `archive/` | Frozen provenance. Retired research roots, retired scope, superseded proposals and snapshots. |

## Read first

1. `docs/INTERPRETABILITY_TRANSFER_AUDIT.md` — canonical findings, limitations catalogue, plan.
2. `docs/RESEARCH_PLAN.md` — scope, evidence discipline, compute policy.
3. `docs/REPOSITORY_STRUCTURE.md` — naming rules, version-control policy, standard layout.
4. `CLAUDE.md` / `AGENTS.md` — environment and operational rules.
5. `docs/EXPERIMENT_LOG.md` — chronological experiment record (`EXP-R2-NNN`).
6. `docs/DOCUMENT_INDEX.md` — navigation, including frozen material.

Repository-root `check.md` is **frozen and superseded** by (1). Its amendment sections remain a useful record of how individual conclusions were reached and withdrawn, but its body asserts claims that have since been retracted. Do not cite it.

## Where the evidence is

Result trees live under `results/` and are B-only. The trees that carry current numbers:

| Tree | What it holds |
|---|---|
| `transfer_20260728/` | The 2026-07-28 campaign: pathway budget, estimand power, circuit primitives (synthetic / approximate / text control / text ladder), convergence control, lens family, probes and erasure, homology control, path patching, induction robustness, PAA census. |
| `transfer_20260729/` | Homology / memorisation control, re-run without repeat masking (EXP-R2-064). |
| `transfer_20260729_instrument/`, `…_instrument_skip4000/` | Eleven-arm instrument-transfer campaign and its cohort-skip sensitivity arm. |
| `transfer_gap_20260724/` | The TG series. Retained unmodified because the audit document cites it. |
| `transfer_gap_20260729_corrected/` | The TG series re-run after the rendering and cohort corrections (EXP-R2-062). |
| `final_checkpoints/` | Four CLT checkpoints (2026-04-03) — the only protein dictionaries held locally; an input to plan item C1. |

Compact synchronized receipts are under `evidence/`.

## Naming convention

- Research roots: `r<integer>_<research_content>`, lowercase snake case.
- Shared infrastructure reserves `r0`; future work reserves `r3`.
- Standard project folders keep conventional names: `docs/`, `scripts/`, `src/`, `results/`, `logs/`, `evidence/`, `configs/`.
- Root operational folders (`data/`, `external_resources/`, `ops/`, `logs/`, `docs/`, `archive/`) are not research directions and carry no `r<ID>` prefix.

## Claim discipline

The audit document's §1 states the defensible position and §3 the status of every claim that has been made. Two standing rules follow from it:

- A limitation demonstrated **on the text control** is a property of the method, not of protein models. Most of this programme's yield is of that kind.
- A modality coefficient in this panel is carried by ProtGPT2 alone. Do not present a single-model contrast as a modality finding.

Cluster access and the latest verified resource note are tracked outside the repository at `/home/lzp/hangzhou-remote/README.md`. Live capacity must be queried before scheduling because that note can become stale.
