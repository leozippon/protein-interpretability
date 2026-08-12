# Research Objective

Study the mechanistic differences among pure-text, pure-protein, and joint language–protein generative models; determine when interpretability methods measure these systems faithfully; and develop causally validated interpretability methods to test whether protein-generative models have learned biological knowledge and, only after those methods prove reliable, use them to discover new biological knowledge. Follow three directions in order:

1. **Compare model families from first principles.** Characterize differences in tokenization, training corpora, and model architecture among pure-text, pure-protein, and joint language–protein generative models. Use matched checkpoint lineages and same-checkpoint text/protein modes where they improve identification.
2. **Use and audit interpretability methods.** Apply interpretability methods as controlled measurement tools to explain how those foundations affect model behavior, representations, and causal computation, while testing whether each method remains faithful across model families. Attribute limitations to the method itself or to transfer; for transfer-specific limitations, localize the responsible model, training data or stage, modality, or evaluation interface.
3. **Develop and validate methods for biological knowledge.** Propose and validate interpretability methods that determine whether protein-generative models have learned biological knowledge rather than merely reproduced corpus statistics or surface correlations. Only after a method passes causal, retrieval-aware, and independent biological validation may it be used to formulate and test hypotheses that could reveal new biological knowledge.

# Repository Principles

`docs/INTERPRETABILITY_TRANSFER_AUDIT.md` is canonical for detailed findings, limitations, retractions, and the current scientific plan. `summary.md` is the user-facing overview of the research direction.

# Repository-Specific Guidelines

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
- Prioritize computations on H200 instead of L20. Reserve L20 exclusively for basic correctness verification only.

### H200 Access

The cluster is offline and reached through `~/hangzhou-remote`. Check health, inspect disposable pods, and select one only for the current shell:

```bash
~/hangzhou-remote/ssh_tunnel/h200_status.sh
~/hangzhou-remote/ssh_tunnel/h200_kubectl.sh get pods -o wide
export H200_POD=<running-pod-name>
~/hangzhou-remote/ssh_tunnel/h200_pod_exec.sh -- nvidia-smi
```

The end-to-end status probe normally takes 40–50 seconds because it crosses several SSH and Kubernetes boundaries. Give `h200_status.sh` a caller-side timeout of at least 90 seconds. A timeout before its terminal `Health=` line is inconclusive, not a failed cluster-health result.

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

## Logging

Record each experiment's date, configuration or command, and result in `docs/EXPERIMENT_LOG.md`. Re-read it immediately before appending because agents write concurrently. Record repository chronology in `docs/PROJECT_LOG.md`; runtime logs stay under ignored `logs/`.

## Mutagen

- Files in @.mutagenignore is ignored from local repository, but you can check and read using terminal commands.

## Additional Operational Rules

- Treat resource checks and logging as mandatory steps, not optional cleanup.
- Include the Research Objective in every sub-agent task prompt in addition to the Development Principles.
- If experiments entail lengthy waiting periods, spawn a background task that retrieves experimental results and resumes this conversation once the projected experiment completion time arrives.

# Global Guidelines

## Development Principles

- **Implementation Principle**: Implement and retain the smallest complete solution justified by current requirements and actual use. Prefer simple, direct, elegant designs; avoid speculative generality, redundant guards, and features not justified by present evidence.
- **Audit Principle**: Freeze the scope and define required behavior and conditions that must always hold; require reproducible evidence of material impact, distinguish defects from suggestions and accepted limitations, and, unless instructed otherwise, weigh expected benefit against added complexity and redundancy. Do not turn low-impact risks into disproportionate machinery; still surface material defects and low-cost fixes.
- **Repair Principle**: Fix one root cause per small, self-contained change and leave overall code health better; redesign instead of stacking exceptions when complexity keeps growing.
- **Failure Principle**: Fail fast and explicitly rather than silently falling back or reporting false success when correctness cannot be guaranteed.
- **Test Principle**: Test conditions that must always hold, negative paths, and realistic end-to-end behavior rather than only the current implementation's happy path.
- **Restraint Principle**: Record irreducible limitations honestly; do not disguise unsupported behavior as compatibility or recovery.
- **Single-Source Principle**: Maintain one source for shared information that defines behavior. Duplicate it only when components cannot share it, and check consistency only when divergence would materially affect correctness or operation.

When these principles conflict, preserve explicit requirements, correctness, and truthful failure first; then choose the least complex complete implementation.

## Operational Guardrails

- Fully read sufficient code and supporting documentation to form a sound design idea before writing or modifying any code.
- Maintain independent judgment. When a request conflicts with evidence, a documented requirement, a safety constraint, or a higher-priority instruction, raise the conflict promptly.
- Use sub-agents when independent work or review materially improves speed or quality; use the highest-performance available sub-agent for high-risk or cross-component work.
- Include the Development Principles above explicitly in every sub-agent's task prompt.
- Treat sub-agent findings as advisory: reproduce material claims and arbitrate them against the frozen scope and required behavior before changing code. Repeat an audit only after a material change or new evidence.
- Place validation where data enters an independently controlled component; repeat it only for a distinct failure path.
- Before removing shared code, persisted data, a public interface, or an operational entry point, check where it is used.
- Keep the repository organized, clean and tidy.
- Match validation to risk. Use lightweight checks for documentation or mechanical changes, targeted negative tests and relevant regressions for behavior changes, and realistic end-to-end validation for authorized cross-environment, destructive, or deployment changes.

## Documentation

- Before a substantial addition or restructuring, identify the document's purpose and scope, then read the full affected document and any relevant neighboring documents. Integrate the change into the existing narrative; when accumulated patches have obscured it, reorganize instead of appending another fragment.
- Give each document a coherent hierarchy and reader-oriented flow. Use headings only for meaningful divisions at the same level, group related ideas together, and move from overview to detail and normal use to exceptions where that order fits.
- Apply the Single-Source Principle across documents. Keep each fact, design decision, and procedure in one authoritative section. Use links or brief pointers elsewhere; when sections overlap, clarify their boundaries instead of repeating the same content.
- Write Markdown prose as logical lines; do not hard-wrap it at 80 columns.
- Keep documentation aligned with current behavior.
- Verify commands and examples that readers may execute, and update or remove stale guidance with the related change.
- Use filename casing as a soft audience convention: retain ecosystem-standard names such as `README.md`, `LICENSE.md`, and `CHANGELOG.md`; use `lowercase-kebab-case.md` for ordinary user-facing documents and `UPPER_SNAKE_CASE.md` for agent, process, or internal-control documents.
- Structure user-facing documentation for human readability in plain, approachable language. Follow these practices:
  - Maintain a clear logical progression and fluent, natural language within and between sections.
  - Keep the content as concise as possible without sacrificing logical completeness.
  - Use tables and lists for genuinely parallel information; use concise prose for reasoning, sequences, and qualifications.
  - Avoid canned introductions, repetitive summaries, excessive headings, artificial parallelism, and unnecessary bold emphasis.
  - Avoid unnecessary redundant abstract concepts; retain necessary standard technical terminology.
  - Omit variable names, filenames, and similar details unless necessary.
  - Keep unnecessary cross-references to a minimum.

## Git and Delivery

- Keep commits focused and self-contained. Code, tests, and living documentation for the same behavior change should usually be committed together.
- Use concise imperative commit subjects, preferably in English for tooling and search consistency. Add a short body when validation or operational impact matters.
- Before committing, remove generated caches such as `__pycache__`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`, `*.pyc`, and `*.pyo`; never commit runtime logs, local state, data dumps, API keys, scratch notebooks, or ignored artifacts.
- Run `git status` before and after changes, review `git diff --cached` before committing, and leave unrelated local changes unstaged.
- Pull and rebase or merge carefully before pushing when the remote branch has moved. Do not rewrite shared history, force-push, or use destructive Git commands unless explicitly approved.
- Commit and push to the configured remote repository as needed.
- Treat commit, push, and deployment as separate actions. Deploy only when explicitly authorized, and follow the repository's authoritative runbook when one exists.
- The canonical remote repository is already configured; prefer SSH Git operations and keep `origin` aligned with it.
- Keep `main` as the only long-lived branch. Temporary branches are allowed when useful and must ultimately be merged into `main`.
