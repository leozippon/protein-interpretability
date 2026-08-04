# Transfer Campaign Operator Guide

This directory contains the InterpretabilityTransfer measurement entry points. Scientific claims and decisions belong in `docs/INTERPRETABILITY_TRANSFER_AUDIT.md`; this file defines how to validate and operate the current campaign.

## Source Of Truth

`panel_contract.py` derives scheduling from `src/transfer/arms.py` and stage capability declarations. `panel_contract.sh` is generated from that contract and consumed by both controller and worker. Never maintain a separate runtime arm or stage list.

Validate the contract before a local or remote run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ct
python scripts/transfer/panel_contract.py --verify
```

The active campaign has 12 arms:

`gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`, `dialogpt-small`, `qwen2.5-0.5b`, `llama-3.2-3b`, `protgpt2`, `zymctrl`, `progen2-base`, `progen2-medium`, `progen2-small`.

The contract declares 12 stages in this order:

| Order | Stage | Entry point | Scope | Contract-eligible arms |
|---:|---|---|---|---|
| 1 | `cohort_power` | `01_cohort_power.py` | panel-wide | all 12 |
| 2 | `pathway_budget` | `02_pathway_budget.py` | per arm | all 12 |
| 3 | `estimand_power` | `03_estimand_power.py` | per arm, then control-anchored recommendation | all 12 |
| 4 | `circuit_primitives` | `04_circuit_primitives.py` | panel-wide | all 12 |
| 5 | `relational_channel` | `05_relational_channel.py` | per arm | `zymctrl`, `progen2-base`, `progen2-medium`, `progen2-small` |
| 6 | `explanation_channel` | `06_explanation_channel.py` | armless | no arm dispatch |
| 7 | `convergence_control` | `07_convergence_control.py` | armless | no arm dispatch |
| 8 | `lens_family` | `08_lens_family.py` | per arm | `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`, `dialogpt-small`, `protgpt2`, `zymctrl`, `progen2-base`, `progen2-medium`, `progen2-small` |
| 9 | `probe_and_erasure` | `09_probe_and_erasure.py` | per arm | all 12 |
| 10 | `homology_control` | `10_homology_control.py` | panel-wide | `protgpt2`, `zymctrl`, `progen2-medium` |
| 11 | `induction_path_patching` | `11_induction_path_patching.py` | panel-wide | all 12 |
| 12 | `paa_census` | `14_paa_census.py` | per arm | `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`, `dialogpt-small`, `qwen2.5-0.5b`, `llama-3.2-3b`, `protgpt2`, `progen2-base`, `progen2-medium`, `progen2-small` |

Eligibility is not uniform. Contract refusals are deliberate and include architecture, tokenization, modality, input-format, and stage-interface limits. Inspect them with `python scripts/transfer/panel_contract.py --json`; do not route around them.

`paa_census` is the one stage whose eligible arm list depends on a scheduling parameter, so that parameter is declared in the contract (`PAA_CENSUS_WIDTH`, rendered as `TRANSFER_PAA_CENSUS_WIDTH`) and passed by the worker rather than left to `ARGS_PAA_CENSUS`. The instance pool admits only cohort rows reaching exactly that width; at the entry point's own default of 512 ProtGPT2 admits none. `zymctrl` is refused permanently: no width admits both its EC-conditioned rendering and ProtGPT2's multi-residue BPE, so it cannot enter a shared-window panel. Its separately declared configuration is a direct invocation, not a campaign item.

This table is a hand-maintained copy of a generated declaration, which is the failure class `panel_contract.py` exists to end, so it is checked rather than trusted: `tests/test_h200_orchestration.py::ReadmeStageTableMatchesTheContract` parses it and fails if any row disagrees with the contract. It had already drifted — `induction_path_patching` was listed with seven eligible arms against the contract's nine, omitting both ProGen2 arms, whose `progen` layout `src.transfer.path_patching.SUPPORTED_ARCHITECTURES` declares.

## Controller And Worker

`run_transfer_h200.sh` is the controller and runs from the repository root on the B workstation. It freezes the complete transfer package and transitive local imports, hashes the snapshot, pushes it through `~/hangzhou-remote`, records a run manifest, and invokes the matching worker inside a selected pod.

`h200_worker.sh` runs only inside the pod from that frozen snapshot. It verifies the generated panel contract, imports selected entry points before scheduling GPUs, checks resources, dispatches stages according to their declared scope, writes through temporary directories, and records per-item checksums for resume.

Never invoke `h200_worker.sh` on B, push an ad hoc code tree for a real campaign, or run a campaign from a mutable working copy. A run result is attributable only to its frozen code hash and manifest.

## Dependencies And Resources

The workstation environment is declared in `requirements.txt`; use a CUDA-enabled PyTorch build appropriate to the host. H200 pods are offline and pre-provisioned, so do not install packages in a pod.

`external_resources/manifests/interpretability_transfer_resources.json` is the machine-readable resource interface. Runtime resource names use the `TRANSFER_` prefix.

| Variable | Resource |
|---|---|
| `TRANSFER_MODEL_BASE_DIR` | Protein model root for ProtGPT2, ZymCTRL, and ProGen2 checkpoints |
| `TRANSFER_TEXT_MODEL_BASE_DIR` | Named text-model root |
| `TRANSFER_TEXT_MODEL_DIR` | GPT-2-large checkpoint |
| `TRANSFER_OPENWEBTEXT_DIR` | OpenWebText cohort |
| `TRANSFER_SWISSPROT_FASTA` | Swiss-Prot FASTA |
| `TRANSFER_ZYMCTRL_FASTA` | EC-conditioned ZymCTRL FASTA |
| `TRANSFER_PFAM_RESIDUE_TSV` | Pfam residue annotations |
| `TRANSFER_ALPHAFOLD_DIR` | AlphaFold structures |
| `TRANSFER_PROTEINGYM_DIR` | ProteinGym assays |
| `TRANSFER_UNIREF50_FASTA` | UniRef50 homology reference |
| `TRANSFER_DIAMOND_TARBALL` and `TRANSFER_DIAMOND_CHECKSUM` | Verified DIAMOND distribution inputs |
| `TRANSFER_DIAMOND_DIR`, `TRANSFER_DIAMOND_DB`, and `TRANSFER_DIAMOND_TMPDIR` | DIAMOND extraction, generated database, and scratch outputs |

Every required path must fail explicitly at first use if absent. Do not substitute a nearby dataset, checkpoint, cohort, or model revision.

## Validate On B

Check current GPU and memory state, then run a small direct stage invocation rather than the H200 controller:

```bash
nvidia-smi
free -h
python scripts/transfer/01_cohort_power.py --help
python scripts/transfer/01_cohort_power.py \
  --kind text \
  --arms gpt2-large \
  --n-seq 20 \
  --device cuda:0 \
  --out /tmp/interpretability_transfer_smoke/cohort_power
```

Use each entry point's `--help` for its current interface. Keep validation outputs outside `results/` when they are disposable.

## Launch On H200

Pods are disposable and have no repository default. Query status, select a running pod in the current shell, inspect its actual GPUs, and preview the campaign:

```bash
~/hangzhou-remote/ssh_tunnel/h200_status.sh
~/hangzhou-remote/ssh_tunnel/h200_kubectl.sh get pods -o wide
export H200_POD=<running-pod-name>
~/hangzhou-remote/ssh_tunnel/h200_pod_exec.sh -- nvidia-smi
bash scripts/transfer/run_transfer_h200.sh --dry-run
```

`h200_status.sh` is an end-to-end probe across several SSH and Kubernetes
boundaries and normally takes 40–50 seconds. Give it a caller-side timeout of at
least 90 seconds. A timeout before the terminal `Health=` line is inconclusive,
not evidence that the cluster is unhealthy.

The status commands naturally display current pod names. Never persist a pod name in repository files, manifests, or durable logs.

Launch the full contract only after reviewing the dry run:

```bash
bash scripts/transfer/run_transfer_h200.sh
```

Scope a run explicitly when resources or scientific gates require it:

```bash
ARMS=gpt2-large,protgpt2 \
STAGES=cohort_power,pathway_budget,estimand_power \
GPUS=0,1 \
bash scripts/transfer/run_transfer_h200.sh --dry-run
```

`ARMS` defaults to the full 12-arm campaign panel and `STAGES` defaults to the full 12-stage order. The worker intersects requested arms with each stage contract and reports refusals; no unsupported arm is silently substituted.

## Controller Environment

| Variable | Meaning |
|---|---|
| `H200_POD` | Required shell-local pod selection |
| `H200_ACCESS_ROOT` | External access-helper root; defaults to `~/hangzhou-remote` |
| `ARMS` | Comma-separated requested arms; defaults to the 12-arm campaign panel |
| `STAGES` | Comma-separated requested stages; defaults to all 12 contract stages |
| `GPUS` | Comma-separated pod-relative GPU indices |
| `TEXT_ARM` | Text control for control-anchored aggregation |
| `ARGS_<STAGE>` | Extra arguments for every item in one stage |
| `ARGS_<STAGE>__<ITEM>` | Extra arguments for one stage item |
| `RUN_ID` | Resume an existing snapshot only when its embedded code hash matches |
| `EXPECTED_GPU_COUNT` | Optional minimum visible-GPU assertion |
| `MIN_FREE_MEM_MIB` | Worker warning threshold |
| `FORCE` | Re-run completed items when set to `1` |

Path overrides for the local checkout, remote package, results, and logs roots are supported by the controller and H200 environment scripts. Keep them in the shell or protected external configuration; never commit infrastructure endpoints.

## Resume And Output Safety

The controller binds each run ID to a content hash. Reusing `RUN_ID` is allowed only when the local frozen scope still matches that hash. The worker skips an item only when its checksum manifest verifies; a missing or invalid manifest causes that item to run again.

Results roots are shared, ignored, and not backed up by Git. Never delete or recreate a results root, never use `git clean -fdx` or `git clean -fdX`, and never treat an orphaned temporary directory as a completed result. Promote only compact, cited receipts to `evidence/`.

## Operator Checklist

1. Run `python scripts/transfer/panel_contract.py --verify`.
2. Confirm required models, datasets, tools, and environment variables.
3. Run `nvidia-smi` and `free -h` on the execution host.
4. Validate the changed stage on B with a small realistic cohort.
5. Query H200 status and set `H200_POD` only in the current shell.
6. Review `run_transfer_h200.sh --dry-run`, including resolved arms, stages, paths, GPUs, and extra arguments.
7. Launch through the controller and retain its run manifest.
8. After completion, verify item checksums and record the experiment in `docs/EXPERIMENT_LOG.md`.
