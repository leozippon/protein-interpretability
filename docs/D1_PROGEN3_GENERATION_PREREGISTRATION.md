# EXP-R2-233: one newer model's native generation point

Frozen specification date: 2026-09-05 (Asia/Shanghai). This is a separate supplement to [EXP-R2-232](D1_GENERATION_BIOLOGY_PREREGISTRATION.md); the EXP-R2-232 protocol and cohort hashes remain unchanged. It is declared before scientific ProGen3-3B outputs are generated. A 112M interface smoke does not qualify the 3B checkpoint and is not scientific evidence for this branch.

## Question and native task

Measure what the released ProGen3-3B native unconditional procedure produces at one fixed operating point, and evaluate its biological compatibility with the same bounded computational pipeline. This adds a current native-generation representative; it is neither a best-decoding search nor a claim about a frontier or upper limit. The EC and superfamily tasks in R227 are not the same task, so no difference of their label rates against this branch is defined.

Use the released `ProGen3Generator` interface, N-to-C unconditional direction prompt `"1"`, with its native BOS handling. Fixed configuration: 800 attempts, eight per batch, seed `20260905 + batch_index`, temperature 0.85, top-p 0.95, explicit top-k 50, repetition penalty 1, sampling enabled and KV caching enabled. Request `min_new_tokens=0`, `max_new_tokens=400` from the official API; that interface adds two delimiter positions, so the effective Hugging Face cap is 402. This is a bounded continuation budget, not an assertion of biological completeness.

The runner must pin the staged checkpoint configuration, shard index and all weights, and the author generator, model, tokenizer and preparation code through its manifest before the first scientific batch. It must preserve raw continuation/token information, the official compiled sequence or compilation failure, a strict leading AA20 run, and stopping/censoring information for every attempt. The primary evaluated sequence representation is the declared strict leading AA20 run, while official compilation success/failure is reported separately. Do not hide failed compilation, empty outputs or censored strings by retaining only successful compiled sequences. No cleaning/cropping may silently turn a failure into success.

Before scientific generation, the actual 3B checkpoint must pass strict loading, finite native likelihood, cached-versus-uncached interface checks and a bounded native-generation smoke under the same staged runtime. Smoke output is marked calibration and does not enter the 800 attempts. A failed gate stops the scientific branch with an explicit reason; no smaller checkpoint substitutes for the declared 3B measurement. Batch resumes must refuse changed source/checkpoint/configuration signatures and preserve completed raw batches unchanged.

## Full accounting and structural subset

The normalized schema uses `arm=progen3-3b`, `condition=unconditioned`, `class_key=null`, and `target_profile_hit=null`. There is no invented EC/IPR target. Report all 800 attempts, empty/invalid/compilation outcomes, length and budget censoring, any-Pfam recognition under the existing gathering-threshold oracle, exact and near-duplicate groups, and reference-search identity/coverage with explicit missing/no-hit states. Any profile recognition is a compatibility descriptor, not functional annotation of an experimentally tested candidate.

After the complete fixed batch exists, select 128 structure inputs by the EXP-R2-232 fixed score-independent length-stratified procedure, or all supported sequences if fewer, and attach one composition shuffle each. Strict AA20 length 16–1024 support and no cropping remain binding. Record every length-stratum population, selected count and inclusion probability. Selection must not depend on profile, reference distance, generation likelihood or structure predictions. The maximum is 256 structure rows before exact-hash cache reuse. Unsupported outputs remain in the original denominator and are not labelled biologically failed.

Use the identical pinned ESMFold weights, 0–100 CA confidence convention, four trunk passes, singleton inference, chunk 128 and mixed stem/trunk precision from EXP-R2-232. The existing natural/shuffled pilot and separately held-out natural controls calibrate predictor behavior on their sequence support. This is a shared control panel, not ProGen3-specific class-oracle validation. Report unmatched length/sequence regimes and do not borrow a claim of experimentally measured folder/nonfolder specificity.

## Endpoints, uncertainty and limits

The sole primary structural contrast is the design-weighted mean CA-pLDDT difference between each selected native output and its own composition shuffle, recovering the supported-output length distribution. Secondary continuous quantities are pTM and fraction of residues at CA pLDDT at least 70. The previously fixed mean-at-least-70 plus fraction-at-least-0.8 confidence event is descriptive and retains its explicit distinction from measured folding or function. Profile and structural availability are shown separately; no target-profile joint endpoint is invented for this unconditional task.

Uncertainty concerns this fixed native task and sampling procedure. Use 4,000 paired near-duplicate sequence-group bootstrap resamples, seed 20260905, retaining the inverse inclusion weights within resampled groups. Report the 95% interval for this one prespecified contrast. This unit differs from EXP-R2-232's class bootstrap because only one native task is measured here; it does not create a class-generalization experiment. Fewer than eight distinct sampled groups yields descriptive point estimates without a bootstrap inference. All pair statistics, sequence-group identifiers and weights are retained so intervals can be reconstructed.

The arm-level computational interpretation requires complete paired supported predictions and an attained shared EXP-R2-232 control calibration. Failure or uncertainty leaves an uncalibrated/mixed/bounded negative report, not evidence of no knowledge. Positive confidence relative to composition shuffle supports the limited predictor-based comparison; naturalness, matching a profile, or reference distance can strengthen contextual interpretation but do not establish actual expression, folding, activity or a specific learned biological rule. Surpassing every stronger reference and complete training-disjoint certification are not knowledge-existence gates.

Reference search is against its named, versioned corpus. Where only query-span coverage can be recovered, report that rather than inventing target coverage. A no-alignment record is distinct from a zero-percent alignment. All-attempt structural rates retain explicit support/missing-outcome bounds; their sampled estimates are not exact whole-cohort structural censuses. Observed qualifying distinct groups are reported without extrapolating the number of all unseen qualifying groups.

## Execution and finite completion

The generation entry point is:

```bash
python scripts/transfer/generate_progen3_evidence.py --checkpoint /gpfs/jiaotongdamoxing/zhk_zip/models/progen3-3b --out results/transfer/progen3_generation_evidence
```

The validated `ct` interpreter and staged author source are supplied explicitly by orchestration (`TRANSFER_PROGEN3_SRC` or the runner's `--source` option). The model/source/runtime manifests, exact command and resource receipts are recorded with the dispatch. Their contents must match the frozen settings above; no scientific output may precede this pinning. No access credentials or transient allocation identifiers are retained.

Completion is the 3B gate's honest terminal outcome, all 800 raw attempts if the gate passes, a complete immutable accounting ledger, the fixed structural subset to terminal outcomes, calibrated analyses, machine-readable tables and standalone figures. Do not extend samples, labels, model sizes or decoding settings based on results. A complete computational evidence package can support an honest publication assessment; it does not create functional laboratory validation or guarantee acceptance.
