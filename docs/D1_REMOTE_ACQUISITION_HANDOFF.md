# Handoff: acquiring and auditing remote-generalization candidates

This brief exists because the previous remote search concluded from what was on disk, and three of its four candidates had never been obtained. Routes to the public repositories are open — measured, below — so the candidates are acquirable and the earlier conclusions about them are not settled. This document carries what the incoming agent should not have to rediscover: what is already measured, the one methodological distinction that decides whether any of this supports a remote claim, the tooling and paths, and the stop rule.

It is a handoff, not a result. Nothing in it is an endpoint qualification.

## The distinction that decides everything, and the retroactive question it answers

**Two measurements, two references, orthogonal.** Conflating them turns a statement about independence into a statement about biological distance, or the reverse.

* **Within-library identity governs whether units are independent.** Exact Smith-Waterman between the cohort's own backgrounds, an edge at 30% identity and 80% coverage of both. It decides what may be held out from what. It says nothing about any corpus.
* **Distance to a reference corpus governs whether a background is remote.** Each background is searched against an external index and banded by the identity of its best hit. This is the only one of the two that can support a remote-generalization claim.

A library can hold backgrounds 30% identical to each other while every one sits at 99% identity to an entry the models saw. The first measurement would report many independent units and the second would report none remote, and only the second is about remoteness.

**Retroactive question, answered from the artifacts.** The existing [remote-homology gate's](D1_GATE_REMOTE_HOMOLOGY.md) remote stratum is banded against an **external reference corpus**, not by within-cohort banding. Its 100 entirely remote family groups are groups every member of which falls below 70% identity to its best hit in the full UniRef50 index — `uniref50_full.dmnd`, 60,315,044 clusters, 25.3 GB — under `diamond blastp --very-sensitive --masking 0 --evalue 1e-3`, identity as `100 * nident / qlen`. Both the frozen endpoint qualification and the gate record name that index and that setting. The grouping at 30%/80% is the separate independence axis. **So the claim is not qualified in the way feared: it is about biological distance from a reference corpus.**

### The sharper form of the question, and what the repository already answers

The open question is not whether the band was measured against a reference search — it was — but whether **that reference snapshot corresponds to any arm's actual training corpus**. A background below 70% identity to its best UniRef50 hit can still be close to something a given model saw, if that model's corpus is larger, differently filtered, or a different snapshot. `src.transfer.arms` declares each panel member's `pretraining_corpus` from its own model card, with an explicit `undeclared` sentinel rather than a guess, so part of the answer is already on disk:

| Declared pretraining corpus | Roster arms |
| --- | --- |
| `uniref50` | protgpt2 |
| `uniref90_bfd30` | progen2-medium, progen2-small |
| `progen2_base_mixture` | progen2-base |
| `webtext` | gpt2, gpt2-large, gpt2-medium, gpt2-xl |
| `reddit_dialogue` | dialogpt-small |
| `qwen2.5_pretraining_mixture` | qwen2.5-0.5b |
| `llama3_web_corpus_with_llama3.1_logit_distillation` | llama-3.2-3b |
| `undeclared` | bygpt5-base-en, bygpt5-medium-en, bygpt5-small-en |
| **not declared in `arms.PANEL` at all** | the other **19** of the 33, including the ProGen3, ProLLaMA, ProteinGLM, RITA, InstructProtein and Galactica families |

**They do not coincide, so the caveat is doing real work and should name what it is silent about.** Exactly **one** roster arm, ProtGPT2, declares the corpus family the searched snapshot belongs to. Two declare **UniRef90 + BFD30**, which is larger and differently filtered, so remoteness from UniRef50 does not imply remoteness from what those arms saw. Nineteen arms carry no corpus declaration in the repository at all — and they include every arm that resolved on the external-confirmation gate — so for those the question cannot be answered from repository declarations and needs each model card read. The protein-free corpora are silent in the other direction: a protein sequence's identity to a UniRef entry says nothing about whether a web corpus contained it, so the band is uninformative about the text arms rather than reassuring.

**What to measure, folded into the same operation.** When the corpus-distance search is built for the acquired backgrounds, run it over the existing remote-homology cohort's 305 wild types as well, against each reference collection that can be identified and staged — UniRef50 is already indexed, UniRef90 would need building — and report the band distribution per collection. If a stratum is remote from every identified collection, the caveat tightens into a claim for those arms. If it is remote only from the snapshot searched, the caveat stays and should name UniRef90/BFD30 and the undeclared nineteen as what it does not cover. Do not infer a corpus for an arm that declares none; the sentinel exists because inventing one would put a false fact into every artefact recording the panel.

The residual boundary is real, narrower, and was already recorded in both artifacts: **UniRef50 is a proxy for what protein-pretrained checkpoints saw, not any arm's own training set, and no band is per-arm.** The roster's arms were trained on different corpora — UniRef50, UniRef90, UniRef100, BFD and metagenomic sets among them — so "remote" means remote from one snapshot of one reference, and no detected alignment is not absence from any model's pretraining data. Anything acquired under this brief inherits that boundary unless it is banded per-arm, which nothing in the programme currently does.

## Reachability, measured 2026-09-25 from Compute

Report a blocked route as blocked; do not infer absence from a failed fetch.

| Host | HTTP | Reading |
| --- | ---: | --- |
| zenodo.org | 200 | open |
| hf-mirror.com (`HF_ENDPOINT`) | 200 | open, and the route for HuggingFace |
| huggingface.co | 000 | **blocked**, as the guidelines record; use the mirror |
| figshare.com | 202 | open |
| github.com | 200 | open |
| ncbi.nlm.nih.gov | 200 | open |
| biorxiv.org | 429 | rate-limited, not blocked; retry rather than conclude |

`cp .env.local.example .env.local && source .env.local` gives `HF_ENDPOINT` and `HF_TOKEN`; `hf download <id> --local-dir <dir> --token "$HF_TOKEN"` resumes automatically. Run `hf` from the `ct` environment.

## Priority, and what to count for each

Stop at the first failure in the declared sequence and read no model quantity before a control qualifies.

**1. DHFR broad mutational scanning.** Four things, and the fourth decides it.

- The count of **exact single-amino-acid** mutants in the released files — exact, not reconstructed from a multi-site library and not inferred from a mutation-count column.
- The number of **independent homolog backgrounds**, measured by grouping the backgrounds against each other at the frozen 30%/80% contract, against the eight-unit floor.
- **Replicate and barcode reliability**: whether a real per-variant dispersion channel exists — independent replicates, barcode counts, or a reported interval — rather than one pooled estimate. Without a dispersion channel there is no floor and no endpoint qualification is possible.
- **The distance of each background to the reference corpus**, by searching every background against `uniref50_full.dmnd` as above. **Do not read this off within-library identity fields.** This is the measurement that decides whether the dataset can support a remote claim at all, and it is the one the previous audit could not make because the files were absent.

What is already on disk and is *not* this dataset: ProteinGym's `DYR_ECOLI_Nguyen_2023` (2,916 exact single mutants) and `DYR_ECOLI_Thompson_2019` (2,363), which share **one** distinct target sequence, give one independent family, and are members of the 217-assay development panel so are not external whatever their support. Use them only as a cross-check on any newly acquired E. coli background, never as the audit's support.

**2. GROQ-seq and the TEV homolog data.** Assess the two datasets **separately** — each needs its own support count, dispersion channel and positive-control structure — and then **establish whether they are independent**, from the files and their metadata rather than the papers' framing. A shared selection system, barcode scheme or sequencing run makes them **one endpoint with two supports, not two endpoints**, and that must be stated explicitly either way. Concretely: compare barcode or index schemes, run identifiers, plasmid or vector fields, and selection-condition labels across the two releases.

**3. PET hydrolase**, a secondary whole-sequence functional option, only if the first two fail.

**4. Chorismate mutase**, exploratory, and only if a complete public **assay-level** release exists. A figure-level or summary-level release does not qualify.

## The machinery that already exists and should be reused rather than rebuilt

| Need | Where |
| --- | --- |
| Corpus-distance search | `scripts/transfer/measure_endpoint_remoteness.py` (bands, Wilson interval, grouping, unit floor) and `scripts/transfer/search_pairwise_homologs.py` (ALIGNMENT_FIELDS, for profiles) |
| Identity bands | `src.transfer.homology.STRATUM_NAMES` / `assign_stratum`; bands are `<30 / 30–70 / 70–95 / ≥95` and the frozen declaration, not to be restated |
| DIAMOND binary | `/Data/lzp/transfer_work/context_homologue/diamond/diamond`, version 2.1.24 |
| UniRef50 index | `/Data/lzp/homology_db/uniref50_full.dmnd`, 60,315,044 clusters |
| Grouping contract | `src.transfer.family_grouping` (`Union`, `batch_align`, `encode`), 30% identity and 80% coverage of both |
| Unit floor | `src.transfer.statistics.bootstrap_unit_floor`, eight units; a stratum below it carries no verdict in either direction |
| Endpoint qualification pattern | `scripts/transfer/qualify_remote_homology_endpoint.py` — two-channel decomposition with shared component as an upper bound and per-channel discordance as a lower bound, group-bootstrap intervals, per-stratum floors |
| Cohort and support declaration | `scripts/transfer/declare_remote_homology_support.py` |
| Control ladder | `scripts/transfer/qualify_nested_controls.py`, either cohort schema |
| Per-arm fit and panel | `scripts/transfer/fit_nested_singles.py`, `scripts/transfer/analyse_nested_panel.py` |
| Campaign manifests | `scripts/transfer/build_nested_gate_campaign.py`; distinct basename per wave |
| Registry | `scripts/transfer/build_dataset_registry.py`, `verify_dataset_registry.py` — register every acquisition with provenance, licence and digests |

Search cost, measured: 798 queries against the full index took 2,665 s at 20 threads, so a few hundred backgrounds is roughly an hour at 96 threads on Compute. Budget one hour per dataset for the corpus-distance measurement.

## Standing constraints

- **Sequence and stop rule.** Remote sequence support → endpoint qualification → control qualification → model readout. Stop at the first failure and report the outcome as **measurement-limited, not model-negative**. Do not force a positive.
- **A control must be a competent held-out predictor in its own right**, with its own contribution reported with an interval, before anything is measured over it. A control whose own contribution is indistinguishable from zero makes any increment over it uninformative; the catalogue records that as L48.
- **Units never mix.** A growth or fitness readout never sits beside a free-energy figure. MGnify is kcal/mol, Domainome is log2 enrichment on the replicate axis and dimensionless normalised fitness on the endpoint, and a functional assay will be something else again. `remote_homology.channel_decomposition` takes its channel names and unit as arguments for exactly this reason.
- **Independent units are family groups, backgrounds or homologs — never variant counts** — reported with effective (Kish) counts and the weighting convention named.
- Register each acquisition and record date, command and result in `docs/EXPERIMENT_LOG.md`, re-reading it immediately before appending. Never persist a pod name.

## What the previous search established, so it is not repeated

- **MGnify is closed, and the reason is a mechanism.** A tenfold tightening of the interval rule leaves the effect-scale shared-to-discordance ratio flat at 1.702 → 1.646 and falling to 1.447 while support collapses 98.3%; the shared component and the discordance shrink in near-exact proportion and the offset survives at every threshold. The two proteases report systematically different quantities, and no precision filter reaches the admitted MegaScale endpoint's 3.86 [3.61, 4.13]. Separately, on the close-identity stratum only 3 of 33 arms hold a positive increment at all three seeds and **none of the 11 residue-level arms is among them**. An expansion was refused on the control, not on cost: the arithmetic would have worked on the remote stratum, where ProGen2-large and ProGen2-xlarge needed 55 and 57 groups against 100 observed.
- **The grouping floor is not a source of slack.** 100, 147 and 176 entirely remote groups at 30%, 50% and 70% identity against a floor of 8. Keep 30%/80%: the 50% count is reached by leaving 1,013 of 1,098 alignment edges unmerged between pairs the contract itself calls homologous.
- **DHFR as held on disk is a failed qualification and TEV was an absent dataset** — different claims. The present brief reopens both because the files were never obtained, not because either judgement was wrong about the files that were there.
