# Transfer measurement package: operator guide

This directory holds the entry points for the R2 text-to-protein interpretability transfer programme. The scientific question, the four-stage decomposition, the model panel and the evidence-discipline rules are documented in `docs/methods/TRANSFER_MEASUREMENT_PROGRAMME.md` and `docs/RESEARCH_PLAN.md`; this file is the practical how-to-run guide. The library code lives in `src/transfer/`.

The entry points below were read directly from `scripts/transfer/*.py` while writing this guide. They were being developed concurrently with this document, so re-check `--help` before a real campaign in case an interface changed after this was written.

## Architecture: controller and worker

Full-scale H200 campaigns use two scripts with different jobs and different hosts, because the code lives on the local L20 host but the models, data and GPUs live behind the GPFS-mounted H200 pods:

- **`run_transfer_h200.sh` (controller)** runs on the local L20 host. It never touches a model or a GPU. It freezes a content-hashed snapshot of `src/transfer/` and `scripts/transfer/`, pushes that snapshot to a versioned GPFS path, writes a run manifest next to it, and hands off to the worker inside a pod via `~/hangzhou-remote/ssh_tunnel/h200_pod_exec.sh`.
- **`h200_worker.sh` (worker)** runs inside the H200 pod, as the copy that shipped inside the frozen snapshot (not a separately deployed copy). It verifies GPFS and the GPUs, then runs all nine stages against GPFS-mounted models and data, writing results atomically and resumably into the shared results root.

Never invoke `h200_worker.sh` directly from the L20 host -- it assumes GPFS-mounted paths that do not exist there. Never invoke the nine `0X_*.py` scripts through the controller for a small check -- see "Validate on L20 first" below.

**A real campaign always goes through the controller. Never push code to GPFS by hand.** `src/transfer/arms.py` and `src/transfer/scaling.py` are being actively edited by other agents throughout this package's development; the port agent's own validation snapshot was pushed manually and was, by its own report, not frozen against those concurrent edits. That is exactly the hazard the code freeze in "Run-id and code freeze" below exists to remove: a manual push captures whatever the working tree happens to contain at the moment someone runs the copy command, with no hash, no manifest and no way to later prove which code produced which result. Every real campaign must mint a run-id through `run_transfer_h200.sh`, which hashes the exact file set it pushes and refuses to let two different code states share one run-id.

## Run-id and code freeze

Every campaign is bound to a **run-id**: `<UTC-ish timestamp>_<first 12 hex chars of the code hash>`, e.g. `20260728102406_7e8eb6def3c0`. The code hash is `sha256` of a `sha256sum`-format manifest over the **frozen scope**, computed fresh by the controller on every invocation. The controller:

1. Refuses to run at all unless `H200_POD` is set and the documented cluster health check (`~/hangzhou-remote/ssh_tunnel/h200_status.sh`) passes.
2. Computes the frozen scope: every regular file under `src/transfer/` and `scripts/transfer/` (excluding `__pycache__`/`*.pyc`) as a baseline, **plus the transitive closure of every `src.*` import found by static analysis of that baseline** -- not a second hard-coded directory list. This exists because a directory-list-only freeze once shipped a snapshot missing `src/revision/` entirely (the transfer package then imported `src.revision.io`, `.statistics`, `.dictionary_fidelity`, `.dictionary_controls` and `.nested_recoverability`, several via relative imports, none of which lived under the two baseline directories) and failed a real nine-stage campaign at tier 1: every `cohort_power` item died two seconds in on `ModuleNotFoundError: No module named 'src.revision'`, after four GPUs were already scheduled. Two earlier campaigns had not caught it because they ran only `circuit_primitives`, which happened not to import `src.revision` -- an accident of stage selection, not evidence the freeze was complete.

   **As of EXP-R2-066 the closure is exactly `src/transfer/` plus `src/__init__.py`**: those twelve symbols were vendored into `src/transfer/{io,statistics,scoring}.py` and `src/revision/` was archived. The derivation stays, and is now the only thing standing between a future dependency and a repeat of the same failure — and it is what *proved* the closure clean rather than assumed. See `freeze_manifest` in `run_transfer_h200.sh` for the algorithm; it also includes one file outside both directories that `07_convergence_control.py` needs by default, `docs/analysis/MODEL_LADDER_20260728.md` (see `stage_snapshot`).
3. Copies exactly that frozen scope into a local staging directory and pushes it with `h200_sync.sh push` to `<GPFS_PACKAGE_ROOT>/<run-id>/`, refusing to proceed if that directory already exists with content that does not match (see below).
4. Writes and pushes `<run-id>/RUN_MANIFEST.json`: run-id, code hash, every frozen file's own sha256, the research root's git revision and dirty state, a UTC timestamp, the arm list, the stage list and every resolved parameter.
5. Invokes `<run-id>/scripts/transfer/h200_worker.sh` inside the pod -- the worker that runs is always the one bound to that run-id's own snapshot, so a later bug fix to the worker cannot retroactively change what an already-frozen run-id executed.

**Reusing a run-id (resume).** Pass `RUN_ID=<value>` to the controller to target a specific existing snapshot instead of minting a new one. The controller checks that `<value>` ends in the current code hash's first 12 hex characters; if the code on disk has changed since that run-id was minted, it refuses rather than silently resuming under different code. If the snapshot already exists on GPFS under that run-id, the push is skipped (it is by construction the same content); if it does not exist yet, the controller pushes it under the requested run-id as normal. A freshly auto-generated run-id whose GPFS directory unexpectedly already exists is treated as an anomaly and refused outright -- normal fresh runs never hit this path.

## Entry points

| script | stage | what it establishes | CLI arm flag |
|---|---|---|---|
| `01_cohort_power.py` | prerequisite | per-arm context-derived information on a frozen cohort; below-threshold arms are unmeasurable | `--arms A [A ...]` |
| `02_pathway_budget.py` | 1 | MLP-vs-attention share of next-token computation, swept over depth | `--arms A [A ...]` |
| `03_estimand_power.py` | prerequisite | which ablation estimands are powered, per arm and panel-wide | `measure --arms A [A ...]` / `recommend` |
| `04_circuit_primitives.py` | 1, 4 | induction/copying head census, direct logit attribution, activation-patching map | `--arms A [A ...]` |
| `05_relational_channel.py` | 3 | residue-pair structure: per-position states vs. attention only | `--arm A` (singular) |
| `06_explanation_channel.py` | 3 | bits of explanation from the annotation channel; event-selection ceiling | none |
| `07_convergence_control.py` | confound control | separates modality from convergence/scale/tokenisation by scoring a size ladder on each model's native cohort | none (`--members`, default all) |
| `08_lens_family.py` | 3 | logit lens / tuned lens / Jacobian alignment, one code path, matched cohort | `--arms A [A ...]` |
| `09_probe_and_erasure.py` | 3, 4 | decodability (probe) versus reliance (LEACE erasure) per concept | `--arm A` (singular) |

Stage numbers follow `docs/RESEARCH_PLAN.md`'s four-stage decomposition (1 substrate, 2 instrument, 3 semantics, 4 causal verification); it does not yet list 07/08/09, which were added after it was last updated.

A few things the table above does not have room for -- see the header comment in `h200_worker.sh` for the authoritative, most current version:

- `01`'s `--arms` are scored **together** against one shared cohort per invocation, so the worker cannot dispatch it purely per arm. It runs as up to four items, not two: `text` (gpt2-large), `protein_large_vocab` (ProtGPT2), `protein_small_vocab` (zymctrl) and `protein_progen2` (progen2-medium). gpt2-large and ProtGPT2 (vocab 50257) run with `--skip-truncation`: `truncation_curve` raises a hard, unhandled error above vocab 1024 without `logits_to_keep` support, which the pod's transformers 4.52.4 lacks, and since 01 writes its report only after its whole per-arm loop finishes, one arm raising would lose every other arm already computed in that invocation. zymctrl and progen2-medium (vocab 458 and 32) do **not** get `--skip-truncation`, since they can compute the curve and it is part of the measurement. progen2-medium is further isolated into its own `--dtype float32` invocation, because its `nll_reduction_shortest_to_longest_nats` was found to be host-bound under bfloat16 (see "Known host-bound quantities" below); `--dtype` is one flag for the whole invocation, so this applies float32 to the rest of that arm's cohort_power measurement too, not only the truncation curve. Each protein sub-item gets its own `--cohort-name` so that ProtGPT2's and progen2-medium's otherwise-identical non-EC cohorts do not collide on the same output filename.
- `02`, `03 measure`, `08` write one JSON per arm and are dispatched one arm per GPU.
- `03 recommend` aggregates already-written per-arm JSON; CPU-only, no `--device`, runs once after every `measure` job.
- `04` writes one JSON per arm **plus one combined `panel_summary.json` from the same process**, so the worker runs it once with every requested arm together, not per arm -- per-arm invocations would each overwrite `panel_summary.json` with a different, incomplete panel.
- **No stage is passed `--dtype` except `cohort_power`'s `protein_progen2` item, above.** Every other stage runs at whatever `--dtype` default its own author chose, unoverridden. This is a fixed policy, not an omission: run `20260728160933_83ff09d5a909` lost all four `lens_family` items to `FloatingPointError: Jacobian disagrees with a central finite difference by 1.0`, because the worker used to force `--dtype bfloat16` on every stage, including `08_lens_family.py`, whose author set `--dtype default="float32"` deliberately -- lens quantities are differences between near-identical distributions, bfloat16 rounding is comparable to the effect being measured, and the script's own tolerance guard correctly refused to emit a wrong Jacobian rather than silently return one. `09_probe_and_erasure.py`'s own default happens to be `bfloat16`, the same value the old blanket override forced, which is exactly why "it has not failed yet" was not evidence the override was safe there either -- it was unexamined, not justified. Overriding a script's own dtype default is done only where there is a specific measured reason, narrow in scope, and documented at the call site in `build_command` -- see the "DTYPE POLICY" comment in `h200_worker.sh`.
- `05` and `09` take a **singular** `--arm`. `05` is valid only for `zymctrl` and `progen2-medium` (ProtGPT2's multi-residue BPE has no residue-to-token map; `gpt2-large` fails the protein-cohort modality check). `09` is valid for all four arms -- per-concept refusals (e.g. ProtGPT2 on residue-level concepts) are written into its output rather than raised.
- `06` takes no `--arm`/`--arms`/`--device` at all: one CPU-only run for the whole panel.
- `07` takes no `--arm`/`--arms` either -- it sweeps a `--ladder-table` of named members (`--members` optionally restricts it; the worker does not pass `--members`, so it measures every configured member). Its `--backup-dir` is pointed at pod-local scratch, matching the script's own stated reason for that flag (the results root has been deleted by a concurrent process before).

Every script's own default output directory disagrees with the others: `01`, `05`, `06` and `09` default under `results/transfer/NN_stage_name/`; `02`, `03`, `04` and `08` default under `results/transfer_20260728/ stage_name/`; `07` defaults under both plus a `logs/`-relative backup. This is an observed inconsistency in the concurrently-written scripts, not a convention to imitate. `01`'s default is the most important one to override, not merely for consistency: it resolves via `R2_REPO_ROOT`, which on the pod points at the GPFS data root, so its default would land at `<R2_REPO_ROOT>/...` -- outside `biocc` and outside the run tree entirely, not just under a differently named directory inside it. The worker overrides every invocation's output flag (`--out`, `--output-root` or `--output-dir`, whichever that script uses) to land under the canonical `<results-root>/<stage>/`, where `<stage>` is `cohort_power`, `pathway_budget`, `estimand_power`, `circuit_primitives`, `relational_channel`, `explanation_channel`, `convergence_control`, `lens_family` or `probe_and_erasure`.

## Dependency order

1. **`cohort_power`** (01) must run first and pass its power check before anything consumes the cohort it freezes.
2. **`pathway_budget`** (02), then **`estimand_power`** (03). Within 03, `measure` runs the text arm (`gpt2-large`) alone first and only then the protein arms, because attainability must be demonstrated on the text control before a gate is applied to a protein arm (evidence discipline rule 1). `recommend` runs once, after every arm's `measure` output exists.
3. **`circuit_primitives`** (04), **`relational_channel`** (05), **`explanation_channel`** (06), **`convergence_control`** (07), **`lens_family`** (08), **`probe_and_erasure`** (09), in that order.

Neither the worker nor any of the nine scripts currently makes 02 onward read 01's unmeasurable-arms verdict automatically -- 01 only reports it. Excluding an arm that 01 marked unmeasurable from the later stages is a manual decision the operator makes (via `ARMS`), not something enforced in code today.

## Required data

`src/transfer/arms.py` and `src/transfer/channels.py` resolve every input path through `env_path("R2_...", local_default)`, so each one can be overridden by an environment variable and otherwise falls back to a local L20 default. A missing path fails at first use inside the relevant script (`src.transfer.arms.require_input_path`), naming the variable that relocates it -- nothing here silently substitutes a different input.

| variable | local (L20) default | what it is |
|---|---|---|
| `R2_REPO_ROOT` | `/Data/lzp/BioInterpretebility-CC` | repo root; several others resolve beneath it |
| `R2_MODEL_BASE_DIR` | `/Data/public/models_R2` | ProtGPT2/ZymCTRL/progen2-medium |
| `R2_TEXT_MODEL_DIR` | `/Data/public/gpt2-large` | the text arm |
| `R2_TEXT_MODEL_BASE_DIR` | `/Data/public` | named text-ladder checkpoints (07 only) |
| `R2_OPENWEBTEXT_DIR` | `/Data/public/datasets/openwebtext-screen/plain_text` | text cohort |
| `R2_SWISSPROT_FASTA` | `data/swissprot/uniprot_sprot.fasta.gz` | protein cohort |
| `R2_ZYMCTRL_FASTA` | `data/zymctrl/ec_labeled_swissprot.fasta` | EC conditioning / ZymCTRL cohort |
| `R2_PFAM_RESIDUE_TSV` | `data/interpro/pfam_residue.tsv` | Pfam residue spans (05, 06) |
| `R2_ALPHAFOLD_DIR` | `data/alphafold` | AlphaFold models (05, 06) |
| `R2_PROTEINGYM_DIR` | `data/proteingym/DMS_ProteinGym_substitutions` | fitness assays (09) |

For **H200 campaigns**, `scripts/transfer/h200_env.sh` (written by the port agent, sourced by the worker, not duplicated here) sets these to their GPFS equivalents, plus `R2_PYTHON` (the pod's interpreter -- there is no conda env in the pod) and `R2_PACKAGE_ROOT` (`PYTHONPATH`). `h200_env.sh` deliberately does not check that its paths exist -- see "Data-path and GPU preflight" below.

## Validate on L20 first

There is no separate L20 launcher. Validation means invoking one entry point directly for one arm with a small cohort, on a GPU you have checked with `nvidia-smi`:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ct
cd the repository root

python3 scripts/transfer/01_cohort_power.py \
  --kind text --arms gpt2-large --n-seq 20 --device cuda:0 \
  --out /tmp/transfer_l20_check/cohort_power

python3 scripts/transfer/02_pathway_budget.py \
  --arms gpt2-large --n-seq 20 --pool-size 40 --device cuda:0 \
  --output-root /tmp/transfer_l20_check/pathway_budget

python3 scripts/transfer/03_estimand_power.py measure \
  --arms gpt2-large --n-seq 20 --pool-size 40 --device cuda:0 \
  --output-root /tmp/transfer_l20_check/estimand_power

python3 scripts/transfer/04_circuit_primitives.py \
  --arms gpt2-large protgpt2 --cohort-size 8 --device cuda:0 \
  --output-dir /tmp/transfer_l20_check/circuit_primitives

python3 scripts/transfer/05_relational_channel.py \
  --arm progen2-medium --n-proteins 10 --min-proteins 6 --scan-models 200 \
  --device cuda:0 --out /tmp/transfer_l20_check/relational_channel

python3 scripts/transfer/06_explanation_channel.py \
  --n-pfam-proteins 200 --n-structures 50 --n-text-documents 50 \
  --out /tmp/transfer_l20_check/explanation_channel

python3 scripts/transfer/07_convergence_control.py \
  --members gpt2-large --pool-size 40 --n-seq 8 --device cuda:0 \
  --output-dir /tmp/transfer_l20_check/convergence_control \
  --backup-dir /tmp/transfer_l20_check/convergence_control_backup

python3 scripts/transfer/08_lens_family.py \
  --arms gpt2-large --n-seq 8 --pool-size 40 --device cuda:0 \
  --output-root /tmp/transfer_l20_check/lens_family

python3 scripts/transfer/09_probe_and_erasure.py \
  --arm progen2-medium --n-structures 10 --n-ec-proteins 10 \
  --device cuda:0 --out /tmp/transfer_l20_check/probe_and_erasure
```

Check each script's own `--help` for the current flag set and defaults before relying on the examples above -- they were current when this guide was written but the scripts were still being developed alongside it. Run `bash scripts/transfer/run_transfer_h200.sh --dry-run` (with `H200_POD` set to any placeholder value) to preview the full campaign's run-id, snapshot path and worker command without transferring or executing anything.

## Launch the H200 campaign

Pods on the H200 cluster are disposable; nothing here defaults to one. Per `~/hangzhou-remote/README.md`, set a pod once for the shell you launch from:

```bash
export H200_POD=<running-pod-name>
bash scripts/transfer/run_transfer_h200.sh --dry-run
bash scripts/transfer/run_transfer_h200.sh
```

A campaign scoped to one stage with a scale override -- the shape of the first real launch this package is meant for, powering up the induction-head finding from 16 natural probes at n=9 -- looks like:

```bash
export H200_POD=<running-pod-name>
STAGES=circuit_primitives \
ARGS_CIRCUIT_PRIMITIVES="--sections induction --repeat-cohort-size 400 --synthetic-probes 128 --cohort-size 120" \
bash scripts/transfer/run_transfer_h200.sh --dry-run
```

The controller runs the documented health check itself before doing anything else and aborts if it fails; you do not need to run it by hand first, though `~/hangzhou-remote/ssh_tunnel/h200_status.sh` is safe to run on its own at any time. Results land under `<GPFS_RESULTS_ROOT>/<stage>/`; worker logs land under `<GPFS_LOGS_ROOT>/<run-id>/`, one file per stage/item; the controller's own copy of the combined worker output streams to `logs/transfer_h200_controller/<run-id>.log` on the L20 host.

## Environment contract

The controller's own CLI is deliberately small (`--dry-run`, `--force`, `--help`); everything else -- which arms and stages run, at what scale, on which GPUs -- comes from environment variables, since none of it is discoverable from `--help` alone:

| variable | default | what it controls |
|---|---|---|
| `H200_POD` | none (required except for `--help`) | which pod the campaign targets. **Never printed** -- see "Pod-name redaction" below |
| `ARMS` | four panel arms | comma-separated arm list, validated against `R2_CAMPAIGN_PANEL` |
| `STAGES` | every stage the panel contract declares | comma-separated stage list (see below) |
| `GPUS` | `0,1,2,3` | comma-separated, pod-relative GPU indices |
| `TEXT_ARM` | `gpt2-large` | which arm must run first in `estimand_power`, and which one its panel verdict is anchored on |
| `ARGS_<STAGE>` | empty | raw extra CLI args appended to every item of that stage (see below) |
| `ARGS_<STAGE>__<ITEM>` | empty | the same, scoped to one item of one stage (see below) |
| `EXPECTED_GPU_COUNT` | empty (auto) | optional extra minimum-GPU-count assertion |
| `MIN_FREE_MEM_MIB` | `16000` | soft warning threshold only, never blocks |
| `FORCE` | `0` | same as passing `--force` |
| `RUN_ID` | empty (mint a new one) | resume a specific existing snapshot |

### The panel contract

`scripts/transfer/panel_contract.py` is the single declaration of which arms each stage may run and why not; `panel_contract.sh` is its rendering, generated by `panel_contract.py --emit` and sourced by **both** the controller and the worker.

Before it existed the same facts lived in five places that could disagree with `src/transfer/arms.py` and with each other: a hand-written `KNOWN_ARMS` in the controller, a second copy in the worker, plus the worker's own modality enumeration, lens-arm exclusion and relational-arm inclusion. Two were wrong. The relational list named `zymctrl` and `progen2-medium` and omitted `progen2-base`, which is protein, residue-tokenised and carries the `relational` capability -- a stage panel silently narrowed by one arm. The homology list was wrong the other way: the worker passed its own four-arm protein list to a script whose own `--arms` default names three, so a campaign run and a direct run measured different panels.

`arm_can_run(stage, arm)` composes four sources and never restates any of them: `ArmSpec.capabilities` (what the panel *intends*), the measuring module's own architecture declaration (what it can *deliver* -- `scaling.LENS_ARCHITECTURES`, `circuits._CIRCUIT_ARCHITECTURES`, `path_patching.SUPPORTED_ARCHITECTURES`), `ArmSpec.modality`/`tokenisation` where a stage's design needs them, and an explicit exclusion list for anything the first three cannot express. A refusal always names the declaration that made it, so a skip line tells an operator whether it was a panel decision, a module limitation or a staging fact.

**Regenerate after any change to `src/transfer/arms.py`:**

```bash
python scripts/transfer/panel_contract.py --emit
```

The worker runs `panel_contract.py --verify` in its preflight -- inside the pod, against the frozen snapshot, before any GPU is scheduled -- and refuses to run if the rendered file disagrees with the live panel. The controller only sources it, because the code-freeze step is required to work on a host without `torch` importable. `tests/test_transfer_stage_contract.py` runs the same verification.

### Pod-name redaction

Standing rule: no pod name in any file, log, commit or manifest. The controller never prints `H200_POD`, and the worker's entire merged stdout/stderr passes through `redact` before it reaches either the terminal or the controller log, so the guarantee does not depend on the operator remembering to pipe through `sed`.

`STAGES` exists because staging is normally partial. As of this writing, GPFS has the full text ladder, ProtGPT2/ZymCTRL/ProGen2-medium/ProGen2-xlarge, the ByGPT5 rungs, Swiss-Prot, UniRef50, ProteinGym, Pfam and the EC corpus, but **AlphaFold is empty** and ProGen2-small/base/large have not landed -- so `relational_channel`, `explanation_channel` and the structural concepts in `probe_and_erasure` cannot run today, while `cohort_power`, `pathway_budget`, `estimand_power`, `circuit_primitives` and `lens_family` all can. `STAGES=cohort_power,pathway_budget,estimand_power,circuit_primitives,lens_family` scopes a campaign to exactly that. An unknown stage name is a hard error naming the known nine, both in the controller (before anything is pushed) and again in the worker. `STAGES` is echoed in both scripts' startup banners alongside `ARMS` and recorded in `RUN_MANIFEST.json`'s `parameters.stages`, so a run's scope is part of its provenance, not just its console output.

A missing *stage script's own input data* (as opposed to a stage the operator did not request at all) is handled the same way, one level down: see "missing input vs. computation error" under "Atomicity and resume" below. Between the two, a campaign that requests all nine stages today, with AlphaFold still empty, correctly runs the five that can and records the rest as skipped with a reason -- and re-running the identical command once AlphaFold lands picks up exactly what was skipped, with no `--force` and no hand-edited `STAGES`.

**Stages do not currently expose their own scale parameters as controller flags** (`--n-seq`, `--pool-size`, `--seeds`, and each script's various other cohort/replicate-count knobs -- every one of the nine scripts names these differently, so there is no single uniform flag set to expose). Without `ARGS_<STAGE>`, every stage runs at whatever `scripts/transfer/0X_*.py` itself defaults to, which is validation scale, not the scale a production campaign needs (for example the 41-seed requirement derived for ProtGPT2). `ARGS_<STAGE>` is the only way to reach those knobs from the controller today: set the stage's raw extra arguments as a single string, one variable per stage --

```bash
ARGS_PATHWAY_BUDGET="--n-seq 500 --pool-size 1000" \
ARGS_COHORT_POWER="--n-seq 41" \
bash scripts/transfer/run_transfer_h200.sh
```

-- and the controller base64-encodes each one (so arbitrary flag values survive the controller -> `h200_pod_exec.sh` -> `kubectl exec` -> worker argv hop intact) and passes it through as `--stage-args STAGE BASE64`. The stage names are those in `R2_STAGE_ORDER`: `cohort_power`, `pathway_budget`, `estimand_power`, `circuit_primitives`, `relational_channel`, `explanation_channel`, `convergence_control`, `lens_family`, `probe_and_erasure`, `homology_control`, `induction_path_patching`.

`ARGS_<STAGE>` reaches **every item of a stage at once**, which is wrong for `cohort_power`: its four items differ in vocabulary regime, dtype and cohort name, so one scale knob is rarely right for all of them. `ARGS_<STAGE>__<ITEM>` scopes an override to one item -- the item name upper-cased with every non-alphanumeric character replaced by an underscore:

```bash
ARGS_COHORT_POWER="--n-seq 500" \
ARGS_COHORT_POWER__PROTEIN_PROGEN2="--cohort-pool-size 8000" \
ARGS_LENS_FAMILY__PROGEN2_MEDIUM="--n-seq 64" \
bash scripts/transfer/run_transfer_h200.sh
```

Either kind is **refused** if it repeats a flag the worker already sets for that item. argparse takes the last occurrence silently, and the flags the worker sets are not conveniences: `--cohort-name` decides the output filename, so overriding it collides two items' cohorts on one path, and `--dtype float32` and `--skip-truncation` each encode a measured reason. Changing one of those is a decision about what is measured and belongs in `build_command` beside its reason. The refusal happens in `verify_commands_buildable`, which builds every scheduled command before the import preflight and long before a GPU is touched.

The decoded, human-readable value is also recorded in `RUN_MANIFEST.json` and is part of what a resume check compares (see "Atomicity and resume" below), so a scale change is exactly the kind of thing that correctly forces a redo rather than a silent skip.

## Atomicity and resume

The worker never writes a script's real output path directly. Each stage/item runs with its output flag pointed at a fresh temporary directory that is itself a subdirectory of the item's final output directory (guaranteeing the same filesystem), and only after the process exits `0` does the worker move the produced files into place one at a time and write `<results-root>/<stage>/.manifests/<item>.sha256` (plain `sha256sum` format) for exactly what it moved. A killed run leaves an orphaned `.tmp.*` directory next to the real output -- safe to delete by hand, never read by anything -- and never a half-written file at the path a later stage or a human would treat as real.

**Resume is keyed on provenance, not only on file integrity.** The results root is shared across runs and outlives any one of them, so a checksum alone cannot tell a smaller, stale configuration from the one actually being asked for: a validation-scale run's output checksums just as cleanly as a production-scale run's. So alongside the checksum manifest, the worker also writes `<results-root>/<stage>/.manifests/<item>.provenance` -- one line recording this run's code hash (the trailing segment of the run-id) and the item's full argument vector, with only the GPU index and the instance-specific output path normalized out, since neither bears on what was measured. Before running an item, the worker checks that **both** the checksum manifest verifies **and** the stored provenance line exactly matches what this run would produce; an item is skipped only if both hold, and `--force` redoes it regardless. A checksum-valid item whose provenance does not match (different code, different arm/dtype/flag selection, or a different `ARGS_<STAGE>` scale override) is redone, not skipped -- this is what stops a later production-scale run from silently returning an earlier validation-scale run's numbers. Every skip and every redo is logged with the reason (`explain_incomplete` in `h200_worker.sh`), naming what matched or what did not, so an operator reading the log sees what was reused rather than having to infer it.

## Import preflight

Before touching GPFS or any GPU, the worker imports each of the nine wired entry points inside the snapshot -- as a module, so `main()` never runs and no model or GPU work happens, just the module's own top-level `import` statements -- and fails loudly, listing every failure rather than stopping at the first, if any of them cannot be imported (`verify_entry_points_importable` in `h200_worker.sh`). This exists because `bash -n` and `--dry-run` cannot catch a missing Python dependency or a syntax error: neither executes Python. A campaign that reached this point with an incomplete frozen scope (see "Run-id and code freeze" above) or a real syntax error in a file still being edited now fails in about two seconds, before any GPU is scheduled, rather than after four GPUs die simultaneously on the same `ModuleNotFoundError`.

**Each entry point runs in its own short-lived interpreter** (one `R2_PYTHON -c` subprocess per file, via `import_one_entry_point`), not one shared interpreter importing all nine in sequence. The first version did the latter and produced a false positive: run `20260728152900_02f91a55c9e7` failed `03_estimand_power.py` with `AttributeError: 'NoneType' object has no attribute '__dict__'`, even though that file imports cleanly on its own, on both the pod and the L20 host. The signature matches an earlier entry point's import leaving a `None` negative-cache marker behind in `sys.modules`, which a later, unrelated entry point's import then tripped over -- contamination between entry points sharing one interpreter, not a defect in either file. Nine separate subprocesses, each with its own private `sys.modules`, removes the possibility entirely: the check tests "does this file import in a clean interpreter" rather than "does it import after eight others have already been imported into the same one". Verified with a two-file reproduction (one fake entry point that leaves a `None` marker in `sys.modules`, a second that is clean alone but raises the exact reported `AttributeError` if imported after the first in one shared interpreter): confirmed the shared-interpreter design reproduces the false positive faithfully, and confirmed the subprocess-isolated design does not, on the identical pair.

**A second, narrower check follows the import check: `--help`, once per entry point** (`verify_entry_points_parse_args`, also one subprocess each). Argparse construction -- the `parser.add_argument(...)` calls -- happens inside `main()`/`parse_args()`, not at module import time, so the import check alone never exercises it; `03_estimand_power.py`'s two required subcommands are each checked (`measure --help`, `recommend --help`), matching how the worker actually invokes it. This is offered because it is cheap, not because it is comprehensive: it would **not** have caught the call-signature drift the port agent found and fixed by hand in `07_convergence_control.py` (`protein_repeat_cohort`/ `text_repeat_cohort` taking a `RepeatCriterion` value after `circuits.py` replaced the old loose `min_unit` keyword) -- that is a runtime call inside a function body, reached only once real measurement code executes past argument parsing, not an argparse-construction-time error, and nothing in this preflight claims to catch that class. Verified in isolation with three fakes: plain argparse (passes), a subcommand parser shaped like `03_estimand_power.py`'s `measure`/`recommend` (both subcommands checked, both pass), and a parser with a duplicate `add_argument("--arms")` registration (correctly fails with the real `argparse.ArgumentError` and full traceback).

## Data-path and GPU preflight

`h200_env.sh` (the pod environment file, sourced by the worker, owned by the port agent) states explicitly that sourcing it does **not** verify that its paths exist: `data/swissprot` and `data/alphafold` are still being staged, and a measurement needing neither must still be able to run. The worker honours that: instead of one blanket check over every input at startup, it checks only the variables the item about to run actually needs, right before that item runs (see `verify_item_data_paths` and the small per-stage tables above it in `h200_worker.sh`). Anything outside that scope is left to `src.transfer.arms.require_input_path`, which raises with the offending variable named in the message. This means a campaign restricted to arms/stages that only touch staged data (for example `pathway_budget` on `gpt2-large`, which needs only `R2_TEXT_MODEL_DIR` and `R2_OPENWEBTEXT_DIR`) can run today even while `data/swissprot`/`data/alphafold` are still being staged.

**Missing input vs. computation error.** These are deliberately different outcomes, because one is a scheduling fact and the other is a defect. When `verify_item_data_paths` finds a path missing, `run_item_atomic` logs it as `SKIP-DATA` with the reason, writes no manifest for that item, and returns success -- so the item is neither counted as a failure (the rest of the campaign continues) nor as complete (a later run of the same command retries it once the input lands, with no `--force` needed, since "not complete" is exactly what triggers a redo -- see above). This is distinct from an item whose script actually ran and raised: that remains a hard failure (`FAIL` in the log, non-zero exit, the stage's wave stops and the campaign aborts), because a crashed measurement needs to be looked at, not silently retried later. Combined with `STAGES` above, a campaign that requests every stage while `data/alphafold` is still empty runs the ones that do not need it and skips-and-records the rest with their reason, rather than requiring the operator to already know which five of nine can run today.

GPU checks are similarly non-blocking where the threshold is not a real requirement: occupancy (another process already using a GPU) is a hard failure, but the free-memory threshold (`MIN_FREE_MEM_MIB`, default 16000) is a logged warning only, since it is not a measured requirement for any of the nine scripts. The GPU *count* check is derived from `nvidia-smi` at run time rather than assumed; `EXPECTED_GPU_COUNT` is an optional extra minimum-count assertion, empty (unset) by default.

## Known host-bound quantities

The port agent's L20-versus-H200 numerical cross-check passed: zero verdict flips across 108 scope x seed comparisons, every H200 point estimate inside the L20 bootstrap interval, divergence 1.4e-4 to 2.9e-3 nats with no systematic bias. Two things it found are still worth knowing before comparing numbers across hosts, not because the campaign is unreliable but because they are genuine, understood properties of the measurement rather than bugs:

- **The trimmed/untrimmed unembedding path is host-dependent, and it is not numerically inert.** `truncation_curve` (`src/transfer/budget.py`) takes a `logits_to_keep`-trimmed path when the installed `transformers` build exposes it on that architecture's `forward`, and the untrimmed path otherwise -- a property of the library version, not of the measurement. transformers 4.57.3 (L20) exposes it for the GPT-2 family; 4.52.4 (the pod) does not. Measured on gpt2-large in bfloat16, the trimmed and untrimmed paths' last-position logits differ by up to 0.25 and the resulting per-token NLL by up to 0.12 nats. **ZymCTRL takes the trimmed path on L20 and the untrimmed one on the pod.** Every truncation curve records `logits_to_keep_used`; a cross-host comparison must check that field rather than assume it matches, and the guard that requires `--skip-truncation` above vocab 1024 without trimming support (see the `01` entry above) is exactly what stops the untrimmed path from running at all where it would dominate memory.
- **progen2-medium's truncation-derived `nll_reduction_shortest_to_longest_nats` is host-bound.** It moved 0.6266 -> 0.7293 (+16%) between float32 and bfloat16 in the cross-check, because it is a small difference between two endpoints that are each close to 2.9 nats -- everything else measured in the cross-check was well within tolerance, and a float32 rerun of this one statistic collapsed its L20-vs-H200 divergence to 2.6e-7. The worker runs progen2-medium's cohort_power measurement in float32 for exactly this reason (see the `01` entry above); this is the one arm/stage combination in the whole campaign that intentionally does not use the bfloat16 inference dtype the rest of the panel uses.

## Assumptions to verify

- **`h200_env.sh`'s contract is now resolved.** It exports `R2_REPO_ROOT`, `R2_MODEL_BASE_DIR`, `R2_TEXT_MODEL_DIR`, `R2_TEXT_MODEL_BASE_DIR`, `R2_OPENWEBTEXT_DIR`, `R2_SWISSPROT_FASTA`, `R2_ZYMCTRL_FASTA`, `R2_PFAM_RESIDUE_TSV`, `R2_PROTEINGYM_DIR`, `R2_ALPHAFOLD_DIR` (the variables `src/transfer/arms.py`/`channels.py`/`probes.py` actually read via `env_path`/`require_input_path`), plus `R2_PYTHON` (the pod's interpreter -- there is no conda env inside the pod) and `R2_PACKAGE_ROOT` (feeds `PYTHONPATH`). `h200_worker.sh` exports `R2_PACKAGE_ROOT=<snapshot-dir>` *before* sourcing `h200_env.sh` so that `PYTHONPATH` resolves to this run's own immutable snapshot rather than `h200_env.sh`'s generic, run-id-less default. `R2_GPFS_ROOT` and `R2_TRANSFER_GAP` are `h200_env.sh`'s own internal composition variables; the worker does not read them directly.
- **The per-stage required-variable mapping in `h200_worker.sh` (`extra_vars_for_stage`, `model_var_for_arm`, `corpus_vars_for_arms`) is a best-effort, deliberately conservative reading of what each script imports from `arms.py`/`channels.py`/`probes.py`, not an exhaustive trace.** It is confirmed by source for 01-06 and 08; for `07` and `09` it checks only what is directly confirmed (model/ladder base directories for 07; the arm's model and `R2_PROTEINGYM_DIR` for 09) and deliberately leaves the rest to `require_input_path`, because their real dependency is conditional on data this worker cannot inspect from bash (which ladder rungs are locally staged for 07; which probe concepts a run reaches for 09). If a run fails inside 07 or 09 on a missing path this worker did not catch early, that is `require_input_path` doing its job, not a bug -- but it is worth widening the table if the same path keeps coming up.
- **`h200_worker.sh` also expects `"$R2_PYTHON" -c 'import torch, transformers'` to work** once `h200_env.sh` is sourced; it does not activate conda itself (there is no conda env in the pod, per `h200_env.sh`'s own comments), on the assumption that the pod's environment setup is `h200_env.sh`'s job, not this worker's.
- The nine scripts' interfaces documented above were read from source at a specific point while they were still being written; re-verify with `--help`.
- The controller and worker were verified with `bash -n` and `--dry-run` against the real repository tree (a real code hash, run-id and worker command were produced), and the atomic-write/resume mechanism, the provenance-keyed skip/redo logic (including the specific validation-scale-then-production-scale scenario the resume fix exists for), the `ARGS_<STAGE>` base64 round trip end to end, the per-item data-path scoping and the four-way `cohort_power` command construction were separately verified in isolation against fake stage scripts, fake data paths and fake arm lists, but neither script has run against a real pod: no GPU job was run as part of writing this.
- A careful re-read while adding the provenance fix found that `build_command` had no `explanation_channel)` case at all, even though `run_explanation_channel` always calls it through `run_item_atomic` -- every invocation of stage 06 would have hit the "unknown stage" fallback and failed outright. Fixed as part of this change; flagged here because it is exactly the kind of defect that only a real run (or a careful line-by-line re-read, which is how it was actually found) turns up, not `bash -n` or `--dry-run`, since neither executes `build_command`.
- **A real nine-stage campaign (run `20260728150714_b613d3afe620`) found a second defect of the same class**: the code freeze covered only `src/transfer/` and `scripts/transfer/`, so every `cohort_power` item died on `ModuleNotFoundError: No module named 'src.revision'` after four GPUs were scheduled. Fixed by deriving the frozen scope from actual imports (see "Run-id and code freeze" above) and adding the import preflight (see "Import preflight" above), which would have caught this specific failure in about two seconds instead of after GPU scheduling. The closure algorithm was verified against the real tree as it then stood -- confirmed to find `src/revision/io.py`, `.statistics`, `.dictionary_fidelity`, `.dictionary_controls` and `.nested_recoverability` plus `src/__init__.py` and `src/revision/__init__.py` (34 files total, up from 23 before this fix) -- and the import preflight was verified in isolation with fake scripts covering a clean import, a missing dependency and a syntax error, and separately run against the real nine entry points, where it correctly surfaced `ModuleNotFoundError: No module named 'torch'`/`'transformers'` (this execution sandbox has neither installed). It was not verified with a working `torch` environment, i.e. the specific all-nine-pass path was not observed directly -- only its individual pass/fail branches, separately.
- **A real launch of the fixed preflight (run `20260728152900_02f91a55c9e7`) found a false positive in the preflight itself**, not in the package: the first version shared one interpreter across all nine imports, and `03_estimand_power.py` failed under it with `AttributeError: 'NoneType' object has no attribute '__dict__'` despite importing cleanly standalone on both the pod and the L20 host -- contamination via `sys.modules` between entry points, not a real defect. Fixed by isolating each entry point in its own subprocess (see "Import preflight" above). Verified with a two-file reproduction built to match the diagnosed mechanism (one fake file that leaves a `None` marker in `sys.modules`, a second that is clean alone but raises the identical `AttributeError` if imported after the first in one shared interpreter): confirmed the old shared-interpreter design reproduces the false positive on this pair, and confirmed the new subprocess-isolated design does not, on the identical pair. The original three-way test (clean / missing dependency / syntax error) was re-run under the new design and still correctly distinguishes all three.
- **A real launch past the fixed preflight (run `20260728160933_83ff09d5a909`) found a fourth defect: seven of nine stages completed, and `lens_family` failed all four items on `FloatingPointError: Jacobian disagrees with a central finite difference by 1.0` (tolerance `2e-2`; the same check passes at `2e-4` to `6e-3` on L20).** The cause was the worker, not the package: `build_command` forced `--dtype bfloat16` on every stage, including `08_lens_family.py`, whose author's own default is `float32` -- deliberately, since lens quantities are differences between near-identical distributions where bfloat16 rounding is comparable to the effect being measured, and the tolerance guard is documented as expected to fail under bfloat16 "by design". Fixed by removing every `--dtype` override except `cohort_power`'s `protein_progen2` item (see the bullet on dtype policy under "Entry points" above and the "DTYPE POLICY" comment in `h200_worker.sh`). Checked `09_probe_and_erasure.py`'s own default specifically, per the request not to assume an override is harmless just because it has not failed yet: it is `bfloat16`, the same value the old blanket override forced, so removing the override changes nothing there today but removes an unexamined assumption. Verified the resulting `build_command` output directly in isolation (all nine stage/item combinations checked; confirmed `--dtype` appears only for `cohort_power`'s `protein_progen2` item) rather than only reading the diff.
- **The port agent's L20-versus-H200 numerical cross-check has since reported and passed** (zero verdict flips across 108 scope x seed comparisons, every H200 point estimate inside the L20 bootstrap interval; see "Known host-bound quantities" above for what it found and how the worker responds to it). That gate is cleared. The remaining blocker to a real campaign is that the remaining ladder rungs and AlphaFold are still staging to GPFS -- not this preflight, which is designed to let a campaign scoped to already-staged data run without waiting for the rest (see "Data-path and GPU preflight" above).

## Do not run `git clean -fdx` in this research root

`` is its own git repository and `/results/` is listed in its `.gitignore`. `git clean -fdx` and `git clean -fdX` therefore delete every experiment artefact under `results/`, including completed runs that took hours of GPU time. This happened three times on 2026-07-28 before the cause was identified, and `04_circuit_primitives.py`'s and `07_convergence_control.py`'s own source comments independently record the same results root disappearing mid-run. The worker's atomic-write design (above) defends against a *concurrent* deletion corrupting an in-flight write, but nothing defends against the results root being deleted outright between runs -- only not running `git clean -fdx` there does.

Use `git clean -fd` (no `-x`), or clean an explicit subpath. Entry points write into `results/transfer_20260728/<stage>/` and must never clear or recreate that root. Snapshots are kept under `logs/results_snapshot_*`.
