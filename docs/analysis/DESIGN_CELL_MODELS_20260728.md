# Staging models to fill the 2x2 modality x tokenisation design

**Date:** 2026-07-28 **Scope:** the modality (text/protein) x tokenisation (subword/symbol-level) design was degenerate — text x symbol-level was empty, and protein x subword had n=1 (`ProtGPT2`). This task investigates candidates to fill/strengthen those cells. Execution host: `/Data/lzp/BioInterpretebility-CC` (models staged under `/Data/public/` and `/Data/public/models_R2/`, outside this repo). This file is the only artifact written **inside this repo** (``) for this task; no `.py`/`.sh` file in this repo was touched. (Post-review addendum, same date: several `.py` files *were* added under `/Data/public/bygpt5-*-en/` to make those checkpoints self-contained — see "Durability fix" below. Those paths are outside this repo entirely.)

## Disk space

| | Available on `/Data` (`/dev/sdc`, 11T total, 94% used) |
|---|---|
| Before | 653 GB |
| After  | 651 GB |
| Net used | ~2.5 GB (`bygpt5-small-en` 281M + `bygpt5-base-en` 532M + `bygpt5-medium-en` 1.1G + `reformer-enwik8` 570M) |

Far above the 200 GB abort threshold throughout; no abort was necessary. Downloads were sequential via `hf-mirror.com`.

---

## Task A — decoder-only, byte/char-level TEXT language model

### Candidates investigated

| Repo id | Decoder-only? | Tokeniser | Verdict |
|---|---|---|---|
| `nllg/bygpt5-base`, `-medium`, `-large` (as literally named in the task prompt) | — | — | **do not exist** on the mirror; 404 even with an authenticated token. The real repo ids are `nllg/bygpt5-{small,base,medium}-{en,de}` (no "large" tier exists) plus poetry fine-tunes (`nllg/poetry-bygpt5-*`, `hgroener/bygpt5_poetry_*`). Found via `HfApi.list_models(search="bygpt5")`. |
| `nllg/bygpt5-small-en` | yes | byte-level (`tokenizer_class: ByGPT5Tokenizer`, a thin subclass of the built-in `ByT5Tokenizer`; vocab 384) | **staged, verified** |
| `nllg/bygpt5-base-en` | yes | byte-level | **staged, verified** |
| `nllg/bygpt5-medium-en` | yes | byte-level | **staged, verified** |
| `nllg/bygpt5-*-de` (German) | yes | byte-level | not staged — kept language consistent with the existing English text ladder (gpt2 family); architecturally identical to the `-en` checkpoints |
| `nllg/poetry-bygpt5-*`, `hgroener/bygpt5_poetry_*` | yes | byte-level | not staged — fine-tunes of the same base checkpoints on poetry corpora, redundant with the base `-en` models already staged |
| `google/byt5-small/base/large/xl/xxl` | **no** | byte-level | confirmed **encoder-decoder** (`T5ForConditionalGeneration`, `is_encoder_decoder: true` in `config.json`, fetched from mirror without download) — excluded per task instructions, ByGPT5 is the decoder-only sibling built on the same byte tokenizer |
| `google/reformer-enwik8` | yes | raw byte encoding (no HF tokenizer shipped at all; model card gives a 2-line `bytes+2` encode/decode function) | **staged, verified** — standard `transformers` architecture (`ReformerModelWithLMHead`), no custom code needed |
| `google/canine-c` | no | — | encoder-only (`feature-extraction` pipeline); excluded, not a causal LM |
| generic searches: "byte-level gpt", "char gpt", "charformer", "byte gpt2" | — | — | no other genuinely decoder-only byte/char-level pretrained *text* LM found on the mirror (hits were either finetunes of GPT-2's ordinary subword tokenizer, or unrelated) |

**ByGPT5 compatibility note:** the HF repos for `bygpt5-*-en` ship only weights + tokenizer files, no modeling code (`model_type: "bygpt5"` is not a built-in `transformers` architecture and the repo originally had no `modeling_bygpt5.py`, so `trust_remote_code=True` alone could not load it). The reference implementation lives in a separate GitHub repo, [`potamides/uniformers`](https://github.com/potamides/uniformers) (not on PyPI). Installing it as a package pins `transformers==4.43.3`, which would have forced a downgrade of the shared `ct` conda environment (currently 4.57.3) — this was judged too risky to the shared environment used by other R1/R2 work. The initial staging pass (first version of this report) instead imported the ~330 lines of source (`configuration.py`, `tokenizer.py`, `model.py`) from a **scratch directory outside `/Data/public/`**, which is not durable: it would not survive, and does not match how this programme's entry points actually load models (`AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)` against each checkpoint's own directory, offline). This was corrected — see "Durability fix" immediately below.

Independent of the durability question, there is also a real `transformers`-version compatibility issue: under current `transformers` internals (`Cache`-object based KV-cache), a forward pass with `use_cache=True` (the upstream default) throws `TypeError: list indices must be integers or slices, not NoneType` inside `T5Attention`'s cache update — a consequence of the vendored code's `layer_idx`-less custom attention blocks predating the `Cache` refactor. `use_cache=False` avoids the cache path entirely and the forward pass then runs cleanly with correctly-shaped logits. This is now baked into each checkpoint's `config.json` as a **deliberate, documented, permanent setting** (not a one-off runtime flag) — see below. It costs nothing for this programme's use case (single forward passes, no `generate()`/ incremental decoding), but it does mean these checkpoints cannot currently be used for autoregressive *generation* without further patching (documented in `modeling_bygpt5.py`'s header comment in each checkpoint directory).

**Reformer compatibility note:** `google/reformer-enwik8` uses LSH (locality-sensitive-hashing) attention, which requires the sequence length to be a multiple of `2 * lsh_attn_chunk_length` (512 here). The 60-byte probe was zero-padded to 512 for the forward-pass check; this is a standard, documented requirement of the Reformer architecture, not a defect.

### Verification table (CPU, one forward pass each)

| Repo id | Local path | Architecture class | Decoder-only | Layers | Hidden width | Vocab | Params | 60-char probe: chars/token | Logits shape | Verified |
|---|---|---|---|---|---|---|---|---|---|---|
| `nllg/bygpt5-small-en` | `/Data/public/bygpt5-small-en` | `ByGPT5LMHeadModel` | yes | 4 | 1472 | 384 | 73,495,680 | **1.000** | (1, 60, 384) | yes |
| `nllg/bygpt5-base-en` | `/Data/public/bygpt5-base-en` | `ByGPT5LMHeadModel` | yes | 6 | 1536 | 384 | 139,218,816 | **1.000** | (1, 60, 384) | yes |
| `nllg/bygpt5-medium-en` | `/Data/public/bygpt5-medium-en` | `ByGPT5LMHeadModel` | yes | 12 | 1536 | 384 | 289,052,672 | **1.000** | (1, 60, 384) | yes |
| `google/reformer-enwik8` | `/Data/public/reformer-enwik8` | `ReformerModelWithLMHead` | yes | 12 | 1024 | 258 | 149,182,722 | **1.000** | (1, 512, 258)\* | yes |

\*padded from 60 to 512 bytes to satisfy the LSH chunk-length constraint (see note above); the first 60 positions correspond to the probe string.

Probe string: `"The quick brown fox jumps over the lazy dog near the sleepy "` (60 chars). All four models tokenize it to exactly 60 tokens (chars/token = 1.000), confirming genuine byte-level granularity, as opposed to the 3-5 chars/token expected of a subword tokenizer.

### Task A conclusion

**Cell filled.** Four decoder-only, byte-level, natural-language-pretrained text LMs are now staged and verified: `bygpt5-small-en` (73M), `bygpt5-base-en` (139M), `bygpt5-medium-en` (289M), and `reformer-enwik8` (149M, pretrained on enwik8 = first 90M characters of English Wikipedia). The ByGPT5 family sits slightly below the "ideally 100M-1.5B" floor at the small end (73M) but base/medium and reformer-enwik8 (139-289M) are comfortably inside it. Two independent architecture families (T5-decoder-style ByGPT5, and Reformer) reduce the risk that any tokenisation-effect estimate from this cell is an architecture artifact rather than a byte-level-tokenizer effect.

### Durability fix — self-contained offline-loadable checkpoints (2026-07-28, post-review)

The first version of this report staged the three `bygpt5-*-en` checkpoints using modeling code imported from a scratch directory outside `/Data/public/`. That is not durable (the scratch dir will not survive) and does not match how this programme's entry points actually load models: `AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)` run directly against each checkpoint's own directory, offline. This has been corrected. Each of the three `bygpt5-*-en` directories is now fully self-contained, following the exact local-`auto_map` pattern already used for `progen2-base`/`progen2-medium` in `/Data/public/models_R2/` in this repo's model set:

1. **Local code copied in.** `configuration_bygpt5.py`, `modeling_bygpt5.py`, and `tokenization_bygpt5.py` (vendored from [`potamides/uniformers`](https://github.com/potamides/uniformers), unmodified except for added header comments) were copied into each of `/Data/public/bygpt5-small-en/`, `/Data/public/bygpt5-base-en/`, and `/Data/public/bygpt5-medium-en/`. (A tokenizer module was required after all: `tokenizer_config.json`'s `tokenizer_class` field is `ByGPT5Tokenizer`, not the built-in `ByT5Tokenizer` — this was mis-stated as already-built-in in the first version of this report, corrected above.)

2. **`config.json` `auto_map` made local, no repo qualifier.** Each `config.json` now has:
   ```json
   "auto_map": {
     "AutoConfig": "configuration_bygpt5.ByGPT5Config",
     "AutoModelForCausalLM": "modeling_bygpt5.ByGPT5LMHeadModel"
   }
   ```
   exactly the bare `module.Class` form `progen2-medium/config.json` uses — no `owner/name--` prefix, which is what sends `transformers` to the Hub and breaks offline loading (the defect repaired on `progen2-base` earlier the same day). Each `tokenizer_config.json` similarly got:
   ```json
   "auto_map": {"AutoTokenizer": ["tokenization_bygpt5.ByGPT5Tokenizer", null]}
   ```

3. **`"use_cache": false` set in each `config.json`.** Deliberate, permanent, documented (see compatibility note above and the header comment now in each `modeling_bygpt5.py`): the legacy T5 attention path in the vendored code breaks against `transformers` 4.57.3's `Cache` object when caching is on; `use_cache=False` runs cleanly for the single-forward-pass scoring this programme needs, at the cost of `generate()` support.

4. **Originals backed up before editing**, matching the `progen2-base` pattern: `config.json.orig_20260728` and `tokenizer_config.json.orig_20260728` sit alongside the edited files in each of the three directories.

#### Offline verification (required re-check)

Command, run identically for each of the three checkpoints (only the path changes):

```
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ct
cd /Data/lzp/BioInterpretebility-CC
HF_HUB_OFFLINE=1 python - <<'PY'
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

model_dir = "/Data/public/bygpt5-small-en"   # or -base-en / -medium-en
config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
model.eval()

text60 = "The quick brown fox jumps over the lazy dog near the sleepy "
enc = tokenizer(text60, return_tensors="pt")
with torch.no_grad():
    out = model(input_ids=enc["input_ids"])
print(type(config).__name__, type(model).__name__, config.use_cache,
      enc["input_ids"].shape[1], tuple(out.logits.shape))
PY
```

No manual `sys.path` edits, no imports from outside each checkpoint's own directory, `HF_HUB_OFFLINE=1` throughout (blocks all Hub network calls). Output, one run per checkpoint:

| Repo id | config/model class resolved from | `config.use_cache` | 60-char probe tokens | chars/token | Logits shape | Offline load + forward pass |
|---|---|---|---|---|---|---|
| `nllg/bygpt5-small-en` | `transformers_modules.bygpt5_hyphen_small_hyphen_en.{configuration_bygpt5,modeling_bygpt5}` | `False` | 60 | **1.000** | (1, 60, 384) | **SUCCESS** |
| `nllg/bygpt5-base-en` | `transformers_modules.bygpt5_hyphen_base_hyphen_en.{configuration_bygpt5,modeling_bygpt5}` | `False` | 60 | **1.000** | (1, 60, 384) | **SUCCESS** |
| `nllg/bygpt5-medium-en` | `transformers_modules.bygpt5_hyphen_medium_hyphen_en.{configuration_bygpt5,modeling_bygpt5}` | `False` | 60 | **1.000** | (1, 60, 384) | **SUCCESS** |

The `transformers_modules.bygpt5_hyphen_*` module names confirm `transformers` resolved the classes from each checkpoint's own local directory (a content/path-derived synthetic package name), not from any Hub lookup — this is the expected signature of a correctly-configured local `auto_map`. Param counts, layer counts, hidden widths, and vocab size are unchanged from the verification table above (this fix only changes *how* the code is found, not the model itself).

### Architecture comparability caveat — read before using this cell in any pathway/attention analysis

**ByGPT5 and Reformer are not architecturally comparable to the GPT-2-family text arms, and Reformer is not comparable to anything else in this design.** Filling the text x symbol-level cell was still worth doing — the protein side of this design already mixes architectures (ZymCTRL is GPT-2/CTRL-style, ProGen2 is GPT-J-style) — but architecture must be carried as a recorded covariate wherever this cell is used, not treated as a clean isolation of the tokenisation effect:

- **ByGPT5** is T5-derived: relative position biases (not GPT-2's learned absolute position embeddings), T5-style RMS-ish layer norm (no mean subtraction, no bias), gated-GELU feed-forward, and no causal-mask learned-position interaction of the GPT-2 kind. Its self-attention, layer-norm placement, and position encoding are all mechanistically different from GPT-2's.
- **Reformer** goes further: it uses locality-sensitive-hashing (LSH) attention (attention is computed only within hashed buckets, not densely over the full sequence) and reversible residual layers (activations are not stored/read the way a standard pre-norm or post-norm residual stream is). Its "attention weights" and residual-stream activations are not the same kind of object as GPT-2's dense causal self-attention and residual stream.

**Practical consequence:** any measurement that reads attention-share, attention-pattern, or residual/pathway structure (the kind of analysis this `the repository root` programme runs) should treat ByGPT5 as a different-architecture text arm requiring its own baseline, and should **not** run Reformer through any attention-pattern or pathway metric that assumes dense causal self-attention and a conventional residual stream at all — its LSH/reversible structure makes such a metric not commensurate with GPT-2's, ProGen2's, or ByGPT5's, and a naive comparison would silently mix incompatible measurements under one column of a plot. Metrics that only need next-token logits (e.g. perplexity, loss-based scoring) are unaffected by this caveat and remain valid for both models.

---

## Task B — second protein LM with subword tokenisation

### Candidates investigated

| Repo id | Local path | Decoder-only | Tokenisation granularity measured | Verdict |
|---|---|---|---|---|
| `hicai-zju/InstructProtein` (already local as `/Data/public/models_R2/InstructProtein`, OPT-1.3B based) | `/Data/public/models_R2/InstructProtein` | yes (`OPTForCausalLM`) | text prompt: **4.364** chars/token (standard English BPE). Protein sequence fed as **plain amino-acid letters** (no special formatting): **1.538** chars/token — some incidental BPE merging, but far short of ProtGPT2's genuine subword compression. Protein sequence fed in the model's **own documented format** (`Ƥ`-prefix per residue, `<protein>Ƥx1Ƥx2...</protein>`, taken verbatim from the official repo's `benchmarks/models/pretrained.py::instructprotein()` preprocessing function on GitHub): **0.984 residues/token** — i.e. essentially exactly one token per residue. | **does not fill the cell.** The model's own creators wrap every residue in a dedicated added token (`ƤA`, `ƤC`, ... 20 single-amino-acid tokens added to the OPT/GPT2 BPE vocab, confirmed in `added_tokens.json`) specifically so that protein spans tokenize one-token-per-residue — i.e. its bona fide protein representation is **symbol-level**, not subword, despite sharing a BPE vocab file with a text tokenizer. Using it on raw unprefixed amino-acid letters would be out-of-distribution relative to how it was trained/documented. |
| `lightonai/RITA_s` / `_m` / `_l` / `_xl` | not staged (tokenizer files only, 6 KB, no weights downloaded) | yes (`RITAModelForCausalLM`, custom code via `auto_map`) | **0.984** chars/token (61 tokens incl. EOS for a 60-residue probe) | **does not fill the cell.** `tokenizer.json` shows `vocab_size: 26` — every one of the 20 canonical amino acids (plus 2 pad/eos and a few extras) is a single-character added token; there is no larger merged vocabulary for BPE to draw on. Confirmed empirically without downloading the ~170 MB-2.4 GB weight files, since the tokenizer alone settles the question. |
| `nferruz/ProtGPT2` (reference — the existing n=1 cell occupant, already local) | `/Data/public/models_R2/ProtGPT2` | yes (`GPT2LMHeadModel`) | **3.000** chars/token (vocab 50,257) | reference point only, not a new candidate — shows what genuine protein subword tokenisation looks like in this cell, for calibration against the two candidates above |
| broad mirror search: "progen" (non-progen2), "amino acid bpe", "protein byte pair encoding", "protein subword", "prot-gpt2 distilled", "protein decoder language model", "protein causal lm" | — | — | no second genuinely subword-tokenized decoder-only protein LM found |

### Task B conclusion

**Cell not strengthened.** Neither of the two candidates named in the task brief actually uses subword tokenisation over amino acids when used as intended:

- `InstructProtein` is architecturally a decoder-only causal LM (`OPTForCausalLM`, 1,315,823,616 params, 24 layers, hidden 2048) and **does** run a clean forward pass on protein input in its documented format (logits shape `(1, 61, 50304)` for the 60-residue Ƥ-prefixed probe) — so it is usable as a protein LM in the causal-scoring sense the design needs. But its official prompting scheme (`<protein>` + `Ƥ`-prefix per residue, confirmed against the maintainers' own evaluation code) forces one-token-per-residue tokenisation, i.e. it is a **second symbol-level** protein model, not a subword one. It is also fundamentally a text-to-protein *instruction* model (bidirectional human-language / protein-language generation), not a plain pretrained protein LM, which is a further reason not to treat it as a like-for-like ProtGPT2 sibling.
- RITA is confirmed symbol-level from its 26-token vocabulary alone (no weights needed to determine this).

No substitute was staged in place of these; per instructions, a documented negative is reported rather than a mis-specified model. The protein x subword cell remains at n=1 (`ProtGPT2`).

---

## Summary: which cells were filled

| | subword tokenisation | symbol-level tokenisation |
|---|---|---|
| **text** | gpt2, gpt2-medium, gpt2-large, gpt2-xl (n=4, pre-existing) | **FILLED THIS TASK**: `bygpt5-small-en`, `bygpt5-base-en`, `bygpt5-medium-en`, `reformer-enwik8` (n=4) |
| **protein** | ProtGPT2 (n=1, unchanged — **not strengthened**, see Task B conclusion) | ZymCTRL, progen2-small/base/medium (n=4, pre-existing) |

Task A (higher priority) is fully resolved: the previously-empty text x symbol-level cell now has four verified decoder-only byte-level models across two independent architectures. Task B is an honest negative: both named candidates, and a broader mirror search, failed to turn up a second protein LM that is genuinely subword-tokenized when used as documented; nothing was staged for that cell and no model should be reported as filling it.

**Before using the text x symbol-level cell in any downstream analysis:** read "Architecture comparability caveat" above. ByGPT5 (T5-derived) and Reformer (LSH attention, reversible layers) are not architecturally comparable to the GPT-2-family text arms that occupy the other three cells of this design, and Reformer's attention/residual structure is not commensurate with anything else here. Architecture is a recorded covariate for this cell, not a controlled-away nuisance variable. All four ByGPT5/ Reformer checkpoints are staged as self-contained, offline-loadable directories under `/Data/public/` (see "Durability fix" above for the exact `auto_map`/`use_cache` changes and offline re-verification).
