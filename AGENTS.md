# InterpretabilityTransfer

## Research Objective

Study how and under what conditions mechanistic-interpretability methods developed on text decoders transfer to protein generative models. Meanwhile, analyses can also be performed from the interpretability perspective to examine the differences between protein‑generative models and language‑generative models, as well as whether protein‑generative models can truly acquire knowledge and generate novel proteins. Follow three directions in order:

1. **Compare model families.** Identify meaningful differences between text and protein generative models.
2. **Evaluate method transfer.** Determine where existing methods transfer, where they fail, and whether each limitation belongs to the method, model, data, or evaluation interface.
3. **Develop adapted methods.** Design and validate protein-specific methods only when the preceding evidence identifies a concrete failure mode. If feasible, leverage interpretability methods to uncover scientifically meaningful findings learned by the models, which correspond to real‑world phenomena or yield novel scientific insights.

`docs/INTERPRETABILITY_TRANSFER_AUDIT.md` is canonical for findings, limitations, retractions, and the current plan.

## Development Principles

- **Audit Principle**: Freeze the scope and define contracts and invariants first; require reproducible evidence of material impact, and distinguish defects from suggestions and accepted limitations.
- **Repair Principle**: Fix one root cause per small, self-contained change and leave overall code health better; redesign instead of stacking exceptions when complexity keeps growing.
- **Failure Principle**: Prefer explicit failure over silent fallback or false success when correctness cannot be guaranteed.
- **Test Principle**: Test invariants, negative paths, and realistic end-to-end behavior rather than only the current implementation's happy path.
- **Restraint Principle**: Record irreducible limitations honestly; do not add speculative abstractions, compatibility branches, or unsupported recovery behavior.

## Environment

- Bash runs on the B workstation. Activate the validated environment before using Python or GPU tools:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ct
```

- Validated workstation runtime: Python 3.11.14, PyTorch 2.9.1+cu128, Transformers 4.57.3, nnsight 0.5.15, and wandb 0.24.0. `requirements.txt` declares the active transfer package's direct Python dependencies; CUDA runtimes remain host-provisioned.
- LaTeX: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate latex && tectonic main.tex`.
- Before editing or running commands that write files, confirm the real path with `pwd -P` or `realpath`.

## Compute

- **Local B workstation:** 8 NVIDIA L20 GPUs, 46068 MiB reported each. Use for validation, small cohorts, and interface checks.
- **Remote H200 cluster:** 16 GPUs in total. A selected pod exposes only its current allocation; each H200 reports 143771 MiB in-pod. Use `scripts/transfer/run_transfer_h200.sh` for full campaigns. Make full use of H200 compute resources and reduce idle time.

### H200 Access

The cluster is offline and reached through `~/hangzhou-remote`. Check health, inspect disposable pods, and select one only for the current shell:

```bash
~/hangzhou-remote/ssh_tunnel/h200_status.sh
~/hangzhou-remote/ssh_tunnel/h200_kubectl.sh get pods -o wide
export H200_POD=<running-pod-name>
~/hangzhou-remote/ssh_tunnel/h200_pod_exec.sh -- nvidia-smi
```

The end-to-end status probe normally takes 40–50 seconds because it crosses
several SSH and Kubernetes boundaries. Give `h200_status.sh` a caller-side
timeout of at least 90 seconds. A timeout before its terminal `Health=` line is
inconclusive, not a failed cluster-health result.

Cluster allocation is not GPU utilization: `16/16` means all GPUs are assigned to pods, not necessarily computing. Inspect `nvidia-smi` inside the selected pod. Never persist pod names in repository files, manifests, or durable logs; status commands naturally display current names, and `H200_POD` is shell-local. Do not install dependencies in a pod or read the mode-600 `~/hangzhou-remote/config.sh`. Stage code and dependencies from B; the external README is authoritative for access, transfer, and recovery.

## Network And Downloads

B has no direct route to `huggingface.co`. Create the ignored local environment file from the placeholder, then use the mirror:

```bash
cp .env.local.example .env.local
source .env.local
MODEL_ID=organization/model-name
MODEL_DIR="$HOME/models/model-name"
hf download "$MODEL_ID" --local-dir "$MODEL_DIR" --token "$HF_TOKEN"
```

Run `hf` from the `ct` environment; downloads resume automatically.

## Git and GitHub

- The canonical remote repository is already configured; prefer SSH Git operations and keep `origin` aligned with it.
- Commit and push to the remote repository as‑needed. Maintain only the main branch of the repository. If other branches must be involved, all such branches shall ultimately be merged into the main branch.
- Keep commits focused and self-contained. Code, tests, and living documentation for the same behavior change should usually be committed together.
- Use concise imperative commit subjects; add a short body when validation commands or operational impact matter.
- Prefer concise English imperative commit subjects for tooling/search consistency. Chinese commit subjects are acceptable for human-facing milestones or domain-specific wording; commit bodies may use Chinese for context and validation details.
- Before committing, remove generated caches such as `__pycache__`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`, `*.pyc`, and `*.pyo`; never commit runtime logs, local state, data dumps, API keys, scratch notebooks, or ignored artifacts.
- Run `git status` before and after changes, review `git diff --cached` before committing, and leave unrelated local changes unstaged.
- Pull and rebase or merge carefully before pushing when the remote branch has moved. Do not rewrite shared history, force-push, or use destructive Git commands unless explicitly approved.

## Logging

Record each experiment's date, configuration or command, and result in `docs/EXPERIMENT_LOG.md`. Re-read it immediately before appending because agents write concurrently. Record repository chronology in `docs/PROJECT_LOG.md`; runtime logs stay under ignored `logs/`.

## Mutagen

- Files in @.mutagenignore is ignored from local repository, but you can check and read using terminal commands.

## Standing Rules

1. **Never take the first N records of a biological corpus.** Sample under a seeded permutation and report a skip-offset sensitivity. File-order iteration has manufactured an effect three times, once worth +1.01 nats.
2. **Check gate attainability on the text control before applying it to a protein arm.** A gate the positive control cannot pass is a specification defect, not a negative result. Violated twice, at real cost.
3. **Use held-out, never plug-in, estimators** for information decompositions; plug-in bias tracks vocabulary size and reached +1.02 nats.
4. **Feed every arm the format it was trained on**, and verify the rendering against the model's own likelihood. Rendering has been worth 1.42-1.78 nats/token.
5. **An intervention that moves everything needs a control for moving everything.**
6. **A reconstruction gate does not protect against wrong norm algebra**; RMSNorm-on-LayerNorm passes at 0.49% error while corrupting attribution.
7. **Verify environment defaults have not narrowed the panel;** print resolved paths before every campaign. One variable silently reduced a campaign's text side to a single model while every downstream number stayed well-formed.
8. **Prefer threshold-free statistics.** Where a threshold is unavoidable, sweep it and show the ordering is invariant.
9. **Report residual-stream spectra on interior, alphabet-bearing positions**; an all-position participation ratio measures the attention sink and separators.
10. **Search the literature by mechanism name, not by domain**, before designing a track.
11. Never delete a result artifact; move it. Frozen provenance is external at `/Data2/lzp/bio_archive`; never edit it in place.
13. Check `nvidia-smi` and `free -h` before and after GPU runs; on H200, run them inside the selected pod.
14. Report failures rather than routing around them. A refutation is a valid result; state it plainly.

## Operational Guardrails

- Fully read sufficient code and supporting documentation to form a sound design idea before writing or modifying any code.
- Treat resource checks and logging as mandatory steps, not optional cleanup.
- Keep the repository organized, clean and tidy.
- Perform audits at appropriate time.
- Sub‑agents may be spawned to assist with development and auditing.
- For difficult development and auditing tasks, spawn the highest‑performance sub‑agent.
- Include the Repository Principles and Development Principles above explicitly in every sub-agent's task prompt.
- Avoid indiscriminately adding new code. Clean up redundant code in a timely manner.
- When necessary, use abstractions, refactoring and other approaches to standardize code and reduce complexity.
- Dare to break thinking inertia and rethink when necessary.
