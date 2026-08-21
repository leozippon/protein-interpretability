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

The active campaign has 15 arms:

`gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`, `dialogpt-small`, `qwen2.5-0.5b`, `llama-3.2-3b`, `protgpt2`, `zymctrl`, `progen2-base`, `progen2-medium`, `progen2-small`, `bygpt5-medium-en`, `bygpt5-small-en`, `bygpt5-base-en`.

The contract declares 13 stages in this order:

| Order | Stage | Entry point | Scope | Contract-eligible arms |
|---:|---|---|---|---|
| 1 | `cohort_power` | `01_cohort_power.py` | panel-wide | all 15 |
| 2 | `pathway_budget` | `02_pathway_budget.py` | per arm | `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`, `dialogpt-small`, `qwen2.5-0.5b`, `llama-3.2-3b`, `protgpt2`, `zymctrl`, `progen2-base`, `progen2-medium`, `progen2-small` |
| 3 | `estimand_power` | `03_estimand_power.py` | per arm, then control-anchored recommendation | `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`, `dialogpt-small`, `qwen2.5-0.5b`, `llama-3.2-3b`, `protgpt2`, `zymctrl`, `progen2-base`, `progen2-medium`, `progen2-small` |
| 4 | `circuit_primitives` | `04_circuit_primitives.py` | panel-wide | `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`, `dialogpt-small`, `qwen2.5-0.5b`, `llama-3.2-3b`, `protgpt2`, `zymctrl`, `progen2-base`, `progen2-medium`, `progen2-small` |
| 5 | `relational_channel` | `05_relational_channel.py` | per arm | `zymctrl`, `progen2-base`, `progen2-medium`, `progen2-small` |
| 6 | `explanation_channel` | `06_explanation_channel.py` | armless | no arm dispatch |
| 7 | `convergence_control` | `07_convergence_control.py` | armless | no arm dispatch |
| 8 | `lens_family` | `08_lens_family.py` | per arm | `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`, `dialogpt-small`, `protgpt2`, `zymctrl`, `progen2-base`, `progen2-medium`, `progen2-small` |
| 9 | `probe_and_erasure` | `09_probe_and_erasure.py` | per arm | all 15 |
| 10 | `homology_control` | `10_homology_control.py` | panel-wide | `protgpt2`, `zymctrl`, `progen2-medium` |
| 11 | `induction_path_patching` | `11_induction_path_patching.py` | panel-wide | `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`, `dialogpt-small`, `qwen2.5-0.5b`, `llama-3.2-3b`, `protgpt2`, `zymctrl`, `progen2-base`, `progen2-medium`, `progen2-small` |
| 12 | `paa_census` | `14_paa_census.py` | per arm | `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`, `dialogpt-small`, `qwen2.5-0.5b`, `llama-3.2-3b`, `protgpt2`, `progen2-base`, `progen2-medium`, `progen2-small`, `bygpt5-medium-en` |
| 13 | `collision_null_census` | `27_collision_null_census.py` | panel-wide | all 15 |

Eligibility is not uniform. Contract refusals are deliberate and include architecture, tokenization, modality, input-format, and stage-interface limits. Inspect them with `python scripts/transfer/panel_contract.py --json`; do not route around them.

`paa_census` is the one stage whose eligible arm list depends on a scheduling parameter, so that parameter is declared in the contract (`PAA_CENSUS_WIDTH`, rendered as `TRANSFER_PAA_CENSUS_WIDTH`) and passed by the worker rather than left to `ARGS_PAA_CENSUS`. The instance pool admits only cohort rows reaching exactly that width; at the entry point's own default of 512 ProtGPT2 admits none. `zymctrl` is refused permanently: no width admits both its EC-conditioned rendering and ProtGPT2's multi-residue BPE, so it cannot enter a shared-window panel. Its separately declared configuration is a direct invocation, not a campaign item. `bygpt5-small-en` and `bygpt5-base-en` are refused here by name, and only here: their 24-head and 72-head grids leave `hit@20` with a chance level too close to its ceiling to read, and no arm property declares a head count. They are full campaign arms elsewhere.

This table is a hand-maintained copy of a generated declaration, which is the failure class `panel_contract.py` exists to end, so it is checked rather than trusted: `tests/test_h200_orchestration.py::ReadmeStageTableMatchesTheContract` parses it and fails if any row disagrees with the contract. It had already drifted — `induction_path_patching` was listed with seven eligible arms against the contract's nine, omitting both ProGen2 arms, whose `progen` layout `src.transfer.path_patching.SUPPORTED_ARCHITECTURES` declares.

## Controller And Worker

`run_transfer_h200.sh` is the controller and runs from the repository root on Compute. It freezes the complete transfer package and transitive local imports, hashes the snapshot, pushes it through `~/hangzhou-remote`, records a run manifest, and invokes the matching worker inside a selected pod.

`h200_worker.sh` runs only inside the pod from that frozen snapshot. It verifies the generated panel contract, imports selected entry points before scheduling GPUs, checks resources, dispatches stages according to their declared scope, writes through temporary directories, and records per-item checksums for resume.

Never invoke `h200_worker.sh` on Compute, push an ad hoc code tree for a real campaign, or run a campaign from a mutable working copy. A run result is attributable only to its frozen code hash and manifest.

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
export TRANSFER_TEXT_MODEL_BASE_DIR=/Data/public
export TRANSFER_MODEL_BASE_DIR=/Data/public/models_R2
python scripts/transfer/01_cohort_power.py --help
python scripts/transfer/01_cohort_power.py \
  --kind text \
  --arms gpt2-large \
  --n-seq 64 \
  --device cuda:0 \
  --out /tmp/interpretability_transfer_smoke/cohort_power
```

Both exports are required and neither has a working default on Compute: text arms resolve `TRANSFER_TEXT_MODEL_BASE_DIR` and protein arms `TRANSFER_MODEL_BASE_DIR`, and the declared defaults under the repository's parent do not exist on this host. The cohort size is not free either — `truncation_curve` needs 200 surviving 128-token windows, which 20 sequences do not supply on this arm.

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
bash scripts/transfer/run_transfer_h200.sh --pin HEAD
```

`--pin` freezes a named commit rather than the working tree, by checking it out into a temporary worktree the controller owns and removes. Use it whenever other agents may be committing: the freeze takes minutes, a hash that moves inside that window aborts the campaign, and one hash moved twice inside a single 4.5-minute freeze on 2026-08-12. Uncommitted work is not frozen under a pin, so commit first; without the flag the freeze reads the working tree exactly as before.

Scope a run explicitly when resources or scientific gates require it:

```bash
ARMS=gpt2-large,protgpt2 \
STAGES=cohort_power,pathway_budget,estimand_power \
GPUS=0,1 \
bash scripts/transfer/run_transfer_h200.sh --dry-run
```

`ARMS` defaults to the full 15-arm campaign panel and `STAGES` defaults to the full 13-stage order. The worker intersects requested arms with each stage contract and reports refusals; no unsupported arm is silently substituted.

## Controller Environment

| Variable | Meaning |
|---|---|
| `H200_POD` | Required shell-local pod selection |
| `H200_ACCESS_ROOT` | External access-helper root; defaults to `~/hangzhou-remote` |
| `ARMS` | Comma-separated requested arms; defaults to the 15-arm campaign panel |
| `STAGES` | Comma-separated requested stages; defaults to all 13 contract stages |
| `GPUS` | Comma-separated pod-relative GPU indices |
| `TEXT_ARM` | Text control for control-anchored aggregation |
| `ARGS_<STAGE>` | Extra arguments for every item in one stage |
| `ARGS_<STAGE>__<ITEM>` | Extra arguments for one stage item; refused when `ARMS` excludes that item, rather than recorded and never applied |
| `RUN_ID` | Resume an existing snapshot only when its embedded code hash matches |
| `EXPECTED_GPU_COUNT` | Optional minimum visible-GPU assertion |
| `MIN_FREE_MEM_MIB` | Worker warning threshold |
| `FORCE` | Re-run completed items when set to `1` |

Path overrides for the local checkout, remote package, results, and logs roots are supported by the controller and H200 environment scripts. Keep them in the shell or protected external configuration; never commit infrastructure endpoints.

## Resume And Output Safety

The controller binds each run ID to a content hash. Reusing `RUN_ID` is allowed only when the local frozen scope still matches that hash. The worker skips an item only when its checksum manifest verifies; a missing or invalid manifest causes that item to run again.

Results roots are shared, ignored, and not backed up by Git. Never delete or recreate a results root, never use `git clean -fdx` or `git clean -fdX`, and never treat an orphaned temporary directory as a completed result. Promote only compact, cited receipts to `evidence/`.

### Known operational limits

| Limit | Required practice |
|---|---|
| Snapshot publication has no distributed lease and is not atomic | Never run two controllers with the same run ID. If a push is interrupted, use a new run ID rather than repairing the partial snapshot in place. |
| Checkpoints and corpora are identified primarily by resolved paths, while code and result files are content-hashed | Preserve model revisions and corpus manifests with any claim that depends on exact upstream bytes. A matching code hash alone is not full input provenance. |
| A narrowed panel-wide run can replace fixed summary files in a shared results root | Give narrowed or exploratory runs a run-scoped `GPFS_RESULTS_ROOT`; do not point them at the canonical full-panel root. |
| The registered controller stops when the remote worker completes; it does not pull or scientifically read the result | Treat `campaign complete` as remote execution success only. Pull, digest-verify, inspect, and log the artefact before calling it admitted. |
| Text cohort qualification is one item covering all selected text arms | Stage every selected text checkpoint first. One missing checkpoint skips the text qualification item rather than producing a partial text baseline. |

## External Baseline Stages

Stages 15 to 26 are external to the registered panel. Each measures a checkpoint, a component, or an estimand the contract does not schedule, so none can name a registered stage or reach the controller's scheduling path; they launch through `run_external_baseline_h200.sh`, which freezes through the controller (never reimplementing the freeze) and then dispatches one stage to one GPU.

The driver is generic — `--stage <file>` accepts any `scripts/transfer/<file>` that exists — so this table records what has been written rather than what the driver enforces. A stage missing from it is undiscoverable rather than unreachable, which is how the table came to name only stages 15 to 20 while four more had already been run through the driver. The third column names what one invocation is *about*, because that is the axis a resume key does not have: each condition needs its own `--label` and its own results directory.

| Stage | What it measures | What one invocation names |
|---|---|---|
| `15_replacement_faithfulness.py` | Behavioural and diagnostic causal checks under sequential per-layer transcoder replacement | `--arm`, a dense panel arm (below), or one mode of a joint checkpoint |
| `16_fitness_recovery.py` | The fitness baseline and recovery interface | a sampling scheme and variant budget |
| `17_train_transcoder.py` | Local PLT and CLT training controls, which produce the dictionaries stage 15 scores | `--arm` as above, and `--architecture` |
| `18_das_subspace.py` | Closed design record; not scheduled and not run | — |
| `19_routing_locality.py` | Whether a mixture-of-experts replacement's residual is structured by the routing decision: router, expert-set, and boundary diagnostics | `--checkpoint` and the `--replacement` artefact to diagnose |
| `20_retrieval_bound.py` | Model fitness against a training-corpus profile retrieval bound | `--arms` and which of its six `--stages` to run; see the `--expect` note below |
| `21_joint_mode_qualification.py` | Whether one joint checkpoint is measurable in **both** modes, in stage 01's estimand | `--checkpoint` (a directory, never an arm name) and `--rendering` |
| `22_neuron_basis_circuit.py` | Faithfulness of a size-*k* circuit in the model's own raw-neuron basis, no dictionary trained | `--arm` |
| `23_perturbation_sensitivity.py` | Behavioural cost of a relative-magnitude MLP-output perturbation, swept over its size | `--arm` **or** `--checkpoint` with `--rendering`; one mode or both |
| `24_component_swap.py` | Which side of a lineage — vocabulary interface or body — carries a measured difference, in stage 21's estimand | `--host`, `--donor`, `--component-group`, `--rendering` |
| `25_model_diffing_baselines.py` | Whether a simple map already accounts for two checkpoints' representational difference, the baseline that gates Crosscoder training | `--reference`, `--target`, `--rendering`, and one `--mode` per run |
| `26_concept_lens.py` | Phase A of the concept-aligned lens: property decoding against its own nulls, with a final-layer positive control that must clear before any depth is read | `--arms` |

Three further entry points belong to neither list. Each re-reads a measurement that already exists rather than scheduling a new one, so none is a campaign item and each writes to its own output directory.

| Stage | What it re-analyses | What one invocation names |
|---|---|---|
| `12_induction_robustness.py` | The induction census's threshold and scale statements, recomputed from artefacts already on disk; it loads no model at all | the stored census to read |
| `13_induction_probe_bootstrap.py` | The same forward passes re-run on one GPU purely to retain the per-probe axis the stored artefacts averaged away | `--arms` and the stored census the recomputation must reproduce |
| `41_context_information_bootstrap.py` | The sampling uncertainty of cohort power, from the per-record sufficient statistics `01_cohort_power.py` persists: a group-level paired interval for context information, paired contrasts within a cohort draw and unpaired ones across draws, the between-block spread, the smoothing sweep and the leakage-removed sensitivity. Which arms are paired follows the cohort and reference digests rather than the file, so the several cohort items that can score one draw — `--dtype float32` is process-global, so an arm needing it is declared and produced on its own — are read under one resample index. `--unigram-null-control` adds one synthetic arm per measurable arm whose true context information is zero by construction, which is the only point in the artefact where the criteria are watched at a known value. CPU only — no model, no GPU, no cluster | `--sidecar`, one per cohort item and grouped into draws by digest; the `--cohort-json` whose records the near-duplicate grouping is computed from; and the `--reference-json` the leakage screen and the reference grouping need |

The `--arm` of stages 15 and 17 defaults to `progen3` and otherwise names a **dense** panel arm — the text control and the dense protein arm a replacement result needs before it can be attributed to protein, to mixture-of-experts, or to transcoder replacement in general. The eligible set is composed by `src.transfer.replaceable.eligible_arms` from `CAMPAIGN_PANEL`, the architectures that carry this estimand, and the arms with a measured loader band; run either stage with `--help` to see it. They are still not registered stages, so a dense-arm run is a direct invocation and not a campaign item.

Stage 20 builds and scores the training-corpus retrieval bound. Its scoring step stages `wildtypes.json` in the output directory before producing the result, so invoke the launcher with `--expect retrieval_bound.json`; otherwise the input file can be mistaken for completed output.

**Snapshot reuse, and the rule that governs it.** Several conditions of one comparison should share one snapshot: four controllers freezing at once collide on the shared relay's single temp script path, and the arms of one comparison must run one code hash or they are not the controlled comparison they are reported as. Freeze once, then pass `--run-id` and `--snapshot-dir` to each invocation:

```bash
export H200_POD=<running-pod-name>
eval "$(scripts/transfer/run_transfer_h200.sh --pin HEAD --freeze-only)"   # sets RUN_ID, SNAPSHOT_DIR, PIN_COMMIT
scripts/transfer/run_external_baseline_h200.sh --pin "$PIN_COMMIT" \
    --run-id "$RUN_ID" --snapshot-dir "$SNAPSHOT_DIR" \
    --stage 17_train_transcoder.py --label clt --gpu 0 -- --architecture clt &
```

**A reused snapshot runs the code it was frozen with, not the code on disk.** Editing a stage after freezing and then reusing the old run-id runs the old stage: four launches once died on `error: unrecognized arguments` for a flag added after the freeze. The driver refuses this — it asks the controller for the current hash (`--print-code-hash`, local only, no network) and requires the run-id's trailing segment to match, which is `resolve_run_id`'s rule applied at the one other place a snapshot can be adopted. **Edit a stage, freeze again.**

**Pass the same `--pin` to every cell of a campaign.** The hash the refusal compares against is read from the working tree unless a commit is named, and the tree moves under other agents: on 2026-08-12 that cost five-plus refused dispatches across three campaigns and two abandoned campaigns, once with the hash moving twice inside one freeze. Pinned, each cell reads the commit its snapshot was frozen from, so the comparison is stable for as long as the campaign runs. The refusal itself is unchanged and still applies to a pinned cell whose commit is not the one behind the snapshot.

**Dispatch each cell as its own command.** `export H200_POD=... && cell0 & cell1 &` backgrounds only the first list, so every later cell runs without the export and dies on the unset variable — eleven of twelve cells in one case, three in another five hours earlier. Export on its own line, then background each cell separately.

**The state vocabulary, because two of these mean very different things.**

| state | meaning |
|---|---|
| `LAUNCHED` | the launcher returned. It is not evidence the stage is running: the access layer returns 0 whatever the remote command did (L20) |
| `DIED AT DISPATCH` | the stage's own log shows a start-up failure. Nothing was scheduled. Exit 6 |
| `PRESENT` | a result JSON exists in the pod-side output directory |
| `ABSENT` | the GPU went idle and no result appeared — a *measurement* outcome, "ran and wrote nothing". Exit 4 |
| `ADMITTED` | pulled to B and the per-file digests taken on each side agree. **Only an ADMITTED result may be read.** Exit 5 on mismatch |

**A completed run whose pull fails is not a failed run.** The tunnel drops; `Connection timed out during banner exchange` during the pull leaves the artefact intact on GPFS and absent on Compute. Check the pod-side log for `[done]` before re-running anything, then re-pull and verify by hand:

```bash
R=<pod result dir>; L=<local result dir>
~/hangzhou-remote/ssh_tunnel/h200_pod_bash.sh "cd '$R' && find . -type f -printf '%P\n' | sort | xargs sha256sum" > /tmp/sums
~/hangzhou-remote/ssh_tunnel/h200_sync.sh pull "$R" "$L"
( cd "$L" && sha256sum -c /tmp/sums )
```

**Waiting on several conditions.** `wait` with no arguments returns 0 regardless of what its children exited with, so a driver that uses it prints "complete" over a lane that failed — which has happened, on a lane whose artefact had not been pulled. Wait per PID and aggregate:

```bash
pids=(); for spec in "${conditions[@]}"; do launch "$spec" & pids+=($!); done
failed=0; for pid in "${pids[@]}"; do wait "$pid" || failed=$((failed + 1)); done
[ "$failed" -eq 0 ] && echo "campaign complete" || echo "campaign INCOMPLETE: ${failed} lane(s) failed"
```

**One claim about health checks, stated here because the external README cannot be edited from this repository.** `Health=ok` is the probe's own terminal verdict and the controller is right to read that line rather than the probe's exit status. But the probe reports its GPFS check over the same channel that returns 0 for a remote command that exited 7, so `PodGPFS=read-write` is evidence that the check *ran*, not proof that a write succeeded. Treat it as a smoke signal; the thing that actually establishes a result crossed the boundary intact is the ADMITTED digest comparison above.

## Multi-Round Campaigns On One Dispatch

Everything above assumes dispatch is cheap: one invocation per cell, each polled and pulled from the workstation. When the transit link is unstable that assumption multiplies the exposure, because an N-round campaign then needs N successful dispatches and N successful polls. Two committed scripts remove most of those crossings. **Both are new and neither has been exercised in a pod; each carries an untested banner in its own header, and the checks to run first are listed there.**

`h200_campaign_queue.sh` runs a whole campaign inside the pod. It is dispatched once — detached with `setsid nohup`, the same detachment the external-baseline driver gives a stage and the one that survived a six-hour outage — and then reads a tab-separated manifest of cells and runs them slot by slot: cells in a slot run concurrently, one per card, and the next slot starts only when every cell of the previous one has exited. The per-cell invocation is the driver's, unchanged: the same interpreter, the same `h200_env.sh` sourcing from the cell's own snapshot, and the same `results/external_baseline/<run-id>/<label>` and `logs/external_baseline/<run-id>_<label>.log` paths. A campaign whose cells are pinned to different commits passes `--snapshot <key>=<dir>` more than once; the runner's header carries the worked case, where EXP-R2-207's trainer cells are pinned to a commit that predates `32_crosscoder.py` and so cannot share a freeze with the Crosscoder cells.

The observation channel is one small status file on GPFS, rewritten atomically on every state change, so `cat`ting it is the whole of "where is the campaign". It carries a metadata block — slot, tally by state, and always-present `# FAILURES` and `# NO-RECORD` lines — then one row per cell with its state, exit code, timestamps and artefact path. A failed cell is recorded and the campaign continues; a nonzero exit is never reported as completion, and because each cell is the runner's own child its exit status is read rather than inferred from a sentinel grep. Re-dispatching after an interruption skips any cell that already has a `*.json` in its own output directory, which is the same completion test the driver applies by default and is sound here because both stages write their record last and write it atomically.

`pull_records_h200.sh` retrieves the verdict without the payload. `h200_sync.sh pull` is a directory operation and cannot be asked for a subset, so a cell's ~46 KB record otherwise waits behind an 8.6 GB or 17.2 GB dictionary — a transfer that has failed twice on chunk-size mismatch. One level below it, `h200_gpfs_pull.sh` is already per-file and already digest-verified, so the only thing missing was file selection. By default this script pulls `*.json` only, verifies the delivered set against pod-side digests with the same comparison the driver admits a result on, and leaves the weights on GPFS where every downstream stage reads dictionaries from anyway; `--with-weights` pulls everything through the same path. Point it at a whole run root to collect every cell's record in one invocation.

## Operator Checklist

1. Run `python scripts/transfer/panel_contract.py --verify`.
2. Confirm required models, datasets, tools, and environment variables.
3. Run `nvidia-smi` and `free -h` on the execution host.
4. Validate the changed stage on Compute with a small realistic cohort.
5. Query H200 status and set `H200_POD` only in the current shell.
6. Review `run_transfer_h200.sh --dry-run`, including resolved arms, stages, paths, GPUs, and extra arguments.
7. Launch through the controller and retain its run manifest.
8. After completion, verify item checksums and record the experiment in `docs/EXPERIMENT_LOG.md`.
