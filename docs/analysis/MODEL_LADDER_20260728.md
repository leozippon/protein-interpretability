# Model size ladder — staging report

**Date:** 2026-07-28 **Scope:** stage a pretrained model size ladder in both modalities (text, protein) for plotting an interpretability metric against model scale. Execution host: `/Data/lzp/BioInterpretebility-CC` (models staged under `/Data/public/` and `/Data/public/models_R2/`, outside this repo). This file is the only artifact written inside the repo for this task.

## Disk space

| | Available on `/Data` |
|---|---|
| Before | 670 GB (`/dev/sdc`, 11T total, 94% used) |
| After  | 653 GB |
| Net used by ladder | ~17 GB (well under the ~30-40 GB estimate; each download used `--exclude` to skip redundant `onnx/`, `.tflite`, `.h5`, `.msgpack`, `.ot`, `.bin` weight-format duplicates and keep only the `safetensors` weights + tokenizer/config files, matching the file set already used for `gpt2-large`) |

Free space stayed far above the 200 GB abort threshold throughout; no abort was necessary.

## Downloads performed (sequential, via hf-mirror)

| Repo id | Local path | Result |
|---|---|---|
| `gpt2` | `/Data/public/gpt2` | downloaded, verified |
| `gpt2-medium` | `/Data/public/gpt2-medium` | downloaded, verified |
| `gpt2-xl` | `/Data/public/gpt2-xl` | downloaded, verified |
| `hugohrban/progen2-small` | `/Data/public/models_R2/progen2-small` | downloaded, verified |
| `hugohrban/progen2-base` | `/Data/public/models_R2/progen2-base` | downloaded, verified |
| `hugohrban/progen2-large` | `/Data/public/models_R2/progen2-large` | downloaded, verified (see vocab-head caveat below) |

`gpt2-large`, `progen2-medium`, `progen2-xlarge` were already present and were **not** re-downloaded, per instructions; they were re-verified in place (read-only, no files modified) so the ladder table below is fully evidence-based.

The exact HuggingFace ids requested for the protein ladder (`hugohrban/progen2-small`, `hugohrban/progen2-base`, `hugohrban/progen2-large`) were all available directly on hf-mirror — no substitute/equivalent conversion was needed.

## Verification method

For every row: loaded with `AutoConfig`/`AutoTokenizer`/`AutoModelForCausalLM`, `trust_remote_code=True`, `dtype=float32`, CPU only (`CUDA_VISIBLE_DEVICES=""`). Text models: tokenized `"The quick brown fox jumps over the lazy dog."`. Protein (ProGen2) models: tokenized `"1MKAILVVLL"` (`1` = N-to-C control token, per ProGen2 convention). One forward pass was run in each case; the values below (layers, width, vocab, param count, logits shape) are read directly from that run, not assumed from config files alone.

## Verification table

### Text ladder (GPT-2 family, byte-level BPE tokenizer, vocab 50257)

| Repo id | Local path | Layers | Hidden width | Vocab | Params | Logits shape | Verified |
|---|---|---|---|---|---|---|---|
| `gpt2` | `/Data/public/gpt2` | 12 | 768 | 50,257 | 124,439,808 | (1, 10, 50257) | yes |
| `gpt2-medium` | `/Data/public/gpt2-medium` | 24 | 1024 | 50,257 | 354,823,168 | (1, 10, 50257) | yes |
| `gpt2-large` (pre-existing, not re-downloaded) | `/Data/public/gpt2-large` | 36 | 1280 | 50,257 | 774,030,080 | (1, 10, 50257) | yes |
| `gpt2-xl` | `/Data/public/gpt2-xl` | 48 | 1600 | 50,257 | 1,557,611,200 | (1, 10, 50257) | yes |

### Protein ladder (ProGen2 family, character-level amino-acid tokenizer)

| Repo id | Local path | Layers | Hidden width (`embed_dim`) | Vocab (runtime) | Params | Logits shape | Verified |
|---|---|---|---|---|---|---|---|
| `hugohrban/progen2-small` | `/Data/public/models_R2/progen2-small` | 12 | 1024 | 32 | 151,148,576 | (1, 10, 32) | yes |
| `hugohrban/progen2-base` | `/Data/public/models_R2/progen2-base` | 27 | 1536 | 32 | 764,803,616 | (1, 10, 32) | yes |
| `hugohrban/progen2-medium` (pre-existing, not re-downloaded) | `/Data/public/models_R2/progen2-medium` | 27 | 1536 | 32 | 764,803,616 | (1, 10, 32) | yes |
| `hugohrban/progen2-large` | `/Data/public/models_R2/progen2-large` | 32 | 2560 | **51,200** (see caveat) | 2,779,356,160 | (1, 10, 51200) | yes, with caveat |
| `hugohrban/progen2-xlarge` (pre-existing, not re-downloaded) | `/Data/public/models_R2/progen2-xlarge` | 32 | 4096 | 32 (config has no `vocab_size`/`n_embd` keys; confirmed at runtime — see caveat) | 6,443,638,816 | (1, 10, 32) | yes |

Tokenizer used by every ProGen2 model above (byte-identical `tokenizer.json`, md5 `edd3465bb725a970cedda3299e02f3bf`, confirmed across all five checkpoints): 31 tokens — `<|pad|>`, `<|bos|>`, `<|eos|>`, digits `1`/`2` (N-to-C / C-to-N control tokens), 25 single-letter residue codes (including ambiguity codes `B`, `O`, `U`, `X`, `Z`), and `<|endoftext|>`. `tokenizer.vocab_size == 30`, `len(tokenizer) == 31`.

## Data-quality caveats (facts only — no files modified to "fix" these)

1. **`progen2-large` output vocabulary size mismatch (real, not a loader bug).** Its `config.json` declares `vocab_size` / `vocab_size_emb` / `vocab_size_lm_head` = 51,200, and the actual checkpoint weights match that: `transformer.wte.weight` and `lm_head.weight` are both `[51200, 2560]` (confirmed by reading the safetensors tensor shapes directly, not just the config). This is **inconsistent** with the 31-token tokenizer shipped in the same repo (identical tokenizer to every other ProGen2 size). It is not a download or loader defect — the `hugohrban/progen2-large` model card's own example code only interprets `logits[:, :tokenizer.get_vocab_size(with_added_tokens=False)]`, i.e. it treats the remaining ~51,169 output columns as unused. Forward passes succeed and produce valid finite logits over the full 51,200 columns; only dims 0-30 are meaningful for amino-acid decoding. Any downstream metric that assumes vocab size equals 32 (as it does for small/base/medium/xlarge) must special-case `progen2-large` or explicitly slice logits to the first 31 columns.

2. **`progen2-xlarge` config has no `vocab_size` or `n_embd` field.** `config.json` for `progen2-xlarge` only exposes `embed_dim` (4096), `vocab_size_emb` (32), and `vocab_size_lm_head` (32) — it lacks the `n_embd`/`vocab_size` keys that `progen2-small`, `progen2-base`, and `progen2-medium` all publish directly. This is a real schema inconsistency between hugohrban's per-size conversions, not an error on our side. Confirmed at runtime: forward-pass logits are `(1, 10, 32)`, and the tokenizer itself reports `vocab_size=30`/`len=31`, consistent with every other ProGen2 model except `progen2-large`. `config.json` was **not** edited.

3. **`progen2-base` and `progen2-medium` are the same architecture size** (27 layers, `embed_dim`/`n_embd` 1536, 764,803,616 parameters exactly, in both cases). This matches the published ProGen2 family design (base and medium share size but were trained on different corpora) — reported here so the "ladder" table is not misread as containing a duplicate/erroneous entry.

## Anything not obtained

None. All six ladder members (three text, three newly-downloaded protein) were downloaded/verified successfully, and both pre-existing entries in each modality (`gpt2-large`, `progen2-medium`, `progen2-xlarge`) were independently re-verified in place. The exact `hugohrban/progen2-*` repo ids requested were all available on hf-mirror; no substitute repos were needed.
