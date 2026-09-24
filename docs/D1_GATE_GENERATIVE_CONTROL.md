# Generation and control gate

This gate asks one question of the Direction-1 panel: **do the sequences a frozen decoder samples satisfy a biological requirement, relative to generators that carry only declared sequence statistics?** It is an independent axis of the [capability map](D1_CAPABILITY_MAP_PLAN.md), not a restatement of the scoring gates. A checkpoint may rank mutations poorly and sample well, or the reverse, and the programme has already measured that scoring, generation and context utilisation do not collapse onto one axis. A negative here says nothing about any other gate, and no other gate's negative says anything about this one.

What is new is the comparison. Every existing generation endpoint in this programme compares a generated sequence either against its own composition shuffle, which is one point on a much longer axis, or against another condition of the same checkpoint, which holds the generator fixed. Neither answers the first objection a reader raises: a generator that carries nothing but the corpus's residue statistics would also produce sequences a curated profile recognises, so how much of the endpoint is that? This gate builds those generators, qualifies each one against the statistics it declares, and reads the model against them on identical denominators.

The design, the predictor settings and the calibration rule for the structure endpoint stay where they already live, in the [generation-biology preregistration](D1_GENERATION_BIOLOGY_PREREGISTRATION.md), the [expansion protocol](D1_UNCONDITIONAL_GENERATION_EXPANSION.md) and the [structure instrument reconciliation](D1_STRUCTURE_INSTRUMENT_RECONCILIATION.md). This document owns the matched generators, the endpoints they are read against, and the gate's verdict.

## The three endpoints and what each instrument licenses

| Endpoint | Instrument | Licensed scope | Noise behaviour |
| --- | --- | --- | --- |
| Any curated family | HMMER `hmmscan` against the staged Pfam-A at the release's own gathering thresholds | The sequence matches a curated profile. Per-arm licensed: one deterministic instrument, identical across arms, and the same call the conditioned-generation campaign used | Deterministic in the sequence. No seed, no repeat spread, nothing to propagate |
| Complete domain | The same call, read on the profile side through `--domtblout` | One assigned profile's best single domain instance covers at least 80% of that profile's own model length. Per-arm licensed for the same reason | Deterministic in the sequence |
| Structural plausibility | ESMFold2, paired within-sequence CA-pLDDT contrast, passed through from the existing analyses | The paired contrast in pLDDT points, and nothing per-arm on 17 of the 19 contrast-reporting arms | Not bitwise reproducible: 0.081 pLDDT points at the median of eight re-folded rows and 0.929 at the worst. The recorded intervals include none of it |

The first two endpoints are both read from one oracle call per sequence, and the two tables that call writes are not the same set. HMMER names a family on the sequence table when the full-sequence score clears the release's GA1 cut and writes a domain row only when one domain also clears GA2, so a family can appear on the sequence table with no domain row at all. That is a property of the oracle rather than a disagreement: the sequence table stays authoritative for the any-family endpoint, exactly as the existing profile endpoint reads it, the domain table alone supplies coverage, and the count of families with no domain row travels with every query. What must always hold is containment, and a query that breaks it refuses the endpoint.

**A complete domain is a stronger biological requirement than a recognised fragment, and it needs no new instrument.** A 40-residue alignment to a 300-residue profile is a statement that part of one domain's signal is present; a single domain instance spanning 80% of the profile is a statement that a whole curated domain is. The 80% threshold was declared before any coverage value was read, and the continuous distribution of best coverage is reported beside the event so the cutoff cannot carry the finding alone.

**The structure endpoint is where the binding limitation sits.** The instrument is admitted on a narrowed scope: the licensed quantity is the paired contrast of a sequence against its own composition shuffle, calibration is attained on the natural controls of two arms, and the other seventeen — every expansion arm and the ProGen3-3B native arm — inherit that attainment. On an inherited arm a positive contrast is evidence that the predictor assigns different CA-pLDDT to a sequence and to its own permutation, not evidence that the predictor was shown to separate natural from shuffled sequences in that arm's own class and length regime. This gate therefore reports the structure endpoint as **unresolved for instrument reasons on 17 of 19 arms** and does not convert a recorded contrast there into a claim that generated sequences are structurally plausible. The two arms with their own pilot controls, ProLLaMA and ZymCTRL, have them in the conditioned-class regime, so the licensed cell for each is its conditioned-generation cell and not its unconditioned expansion cell. Predictor confidence is not observed folding, expression or function, and no exact-sequence experimental measurement exists for any row in this programme.

**Three candidate endpoints were considered and declined, each for its own reason.** Native-format completion — whether an attempt reaches its own terminal inside the residue budget — is a formatting event rather than a biological requirement, and a short sequence can be biologically complete; it is carried as a denominator descriptor and not as an endpoint. Composition, entropy and within-sequence repetition are controls at this gate and cannot also be endpoints of it; they are what the matched generators hold fixed.

Assignment to the **requested** class, the sharpest form of the requirement, stays where it already is. It is measured and frozen for the two conditioned checkpoints against a within-arm mismatched-label negative, and this gate adds nothing to it, because none of the matched statistical generators carries a class label: a corpus fragment drawn without reference to a class reaches a nominated EC number or superfamily at a rate set by how many classes the ontology holds, so that contrast is uninformative rather than hard. The informative control there is a label-carrying one — corpus records of the requested class, drawn from a class-labelled corpus — which is a new instrument rather than a matched statistical generator, and it is recorded below as a recommendation rather than built here. What the two conditioned cells contribute at this gate is their any-family and complete-domain behaviour against the matched generators on their own denominators.

## The matched generators

Each control emits **one sequence per attempt of the arm's own ledger, at that attempt's exact length**, so the denominator of a contrast is the denominator of the ledger by construction and the comparison is paired. The declared amounts of sequence statistics form one axis, from the attempt's own residue multiset to a whole corpus record.

| Cohort | What it carries | Role |
| --- | --- | --- |
| `shuffle` | The attempt's own length and residue multiset, exactly; residue order destroyed | Control |
| `markov_0`, `markov_2`, `markov_4` | The staged UniRef50 conditional at conditioning order 0, 2 and 4, at the attempt's length | Control |
| `fragment` | A contiguous substring of one corpus record at the attempt's exact length — every corpus order at once, the limit of the `markov_k` axis | Control |
| `hydropathy` | The corpus composition exponentially tilted to the attempt's own mean Kyte–Doolittle hydropathy, at the attempt's length | Control |
| `natural` | A whole corpus record inside the attempt's length stratum | Reference, not a control |
| `profile` | Sequences emitted from the oracle's own profiles | Oracle-access ceiling, never a control the model is asked to beat |

**The hydropathy channel here is a generator and is not the one that beat a protein decoder on mutation ranking.** The frozen claim that a one-parameter Kyte–Doolittle change sum out-ranks ProtGPT2 on designed sequences is a statement about scoring a fixed variant set, and a ranking baseline cannot be read against a sampler at all. What this cohort supplies is the closest generative analogue: a sequence carrying the corpus composition and the attempt's own mean hydropathy and nothing else. It is a different object from that baseline, the two numbers are never differenced, and a result either way here neither supports nor disturbs that claim.

`natural` is excluded from every verdict deliberately: a decoder that does not out-recognise real proteins has not failed this gate, and the reference exists to say what satisfying the requirement looks like at that length. The `profile` ceiling is excluded for the opposite reason — it reads the endpoint with direct access to the database that scores it, so its rate is the endpoint's attainable maximum and a model's rate is read against that rather than against 1.000.

**That maximum is measurably below 1.000, which is worth having before any model rate is read.** Emitting one sequence from each of 1,000 Pfam-A profiles drawn under the gate seed, at a median consensus length of 125 residues, the same oracle recognises **0.892 [0.871, 0.910]** of them as carrying any family and assigns a complete domain to **0.835 [0.811, 0.857]** (Wilson 95%, denominator 1,000 of 1,000). A generator with direct access to the scoring database therefore misses roughly one attempt in nine, because a single draw from a profile's emission distribution need not clear that profile's own curated cut. The endpoint is a threshold on a bit score, not a test of membership.

**Competence before comparison.** The capability map binds every control with one rule: a control enters a comparison only if it is competent in its own right. For a generator that means reproducing the order statistics it declares, measured rather than asserted. `shuffle` and `fragment` are verified exactly, per sequence. `hydropathy` is verified as a cohort mean absolute deviation from its per-sequence target. The `markov_k` cohorts are placed on the corpus's order axis by their mean conditional log-likelihood in nats per residue under the staged UniRef50 conditional at each order, against the length-matched `fragment` cohort as the corpus reference — a like-for-like reference rather than the corpus's own entropy, because each cohort scores 4,617,020 positions at order 5 against 64,000,000 cells of that conditional, so any absolute distance there is dominated by sparsity rather than by order.

### Which controls qualified

Measured on all 14,467 searchable parents, before any recognition rate was read.

| Cohort | Declared statistic | Measured | Qualified |
| --- | --- | --- | --- |
| `shuffle` | Exact length and exact residue multiset | 0 length failures and 0 composition failures of 14,467 pairs | yes |
| `fragment` | A contiguous substring of its named corpus record at the exact length | 0 containment failures and 0 length failures of 14,467 | yes |
| `hydropathy` | The attempt's own mean Kyte–Doolittle hydropathy | 0.154 KD units mean absolute deviation, 0.117 at the median, against a declared tolerance of 0.25 | yes |
| `natural` | A whole record inside the attempt's length stratum | 0 band failures of 14,467 | yes |
| `markov_0` | The corpus conditional at order 0 | −0.0032 nats/residue from the corpus reference at order 0; captures a share of **−1.04** of the reference's gain one order up | under the amendment only |
| `markov_2` | The corpus conditional at order 2 | +0.0002 nats/residue at order 2; share **−0.92** | under the amendment only |
| `markov_4` | The corpus conditional at order 4 | −0.0007 nats/residue at order 4; share **−0.90** | yes |

All three Markov cohorts match their declared order to within 0.0032 nats per residue of the length-matched corpus reference, and all three capture a negative share of the reference's gain one order above it — that is, each fails to pick up any of the structure the corpus gains at the order above its own, which is what it means to be a point on this axis.

### These qualifications are exactness and order statistics, not intervals, and that is the right form here

Every other gate in the map qualifies a control by its own paired contribution to the baseline it augments, with an interval, so that a control indistinguishable from zero is visible rather than implied. This gate cannot use that form and does not need it, because its controls are *generators* rather than feature blocks: what has to be established is that each emits the statistics it declares and nothing more, and that is a property of every row rather than of an average over rows. Stating it as an interval would be a weaker claim about a stronger fact. What each of the three qualifications establishes, and why its form is the tight one:

**`fragment` is exact, per sequence, on all 14,467 searchable parents: 0 containment failures and 0 length failures.** The declared statistic is that every emitted sequence is a contiguous substring of its named corpus record at the parent attempt's exact length. That is a decidable property of each pair, checked on each pair, so the qualification is a census with no residual: there is no sequence in the cohort for which it might not hold, and an interval would describe sampling variation in a quantity that has none. This matters more than for the other two, because `fragment` is the binding control on both endpoints — the one 17 of 20 cells do not clear — so its competence is what those 17 verdicts rest on.

**`hydropathy` is a bounded deviation from a per-sequence target: 0.154 KD units mean absolute deviation, 0.117 at the median, against a declared tolerance of 0.25.** The declared statistic is the attempt's own mean Kyte–Doolittle hydropathy, and an exponentially tilted composition hits a per-sequence target only approximately, so here the qualification is a *bound* rather than a census. The tolerance was declared before the measurement and the realised deviation is below half of it. An interval on the cohort mean would answer a question nobody asked — whether the average miss differs from zero — when what the control has to satisfy is that no sequence misses its own target by more than the declared amount.

**`markov_k` is an order statistic on the corpus's own scale: −0.0007 nats/residue at order 4 against the length-matched corpus reference, with a share of −0.90 of the reference's gain one order up.** A Markov sampler's competence is not a magnitude to be bounded away from zero but a *position on an axis*: it must reproduce the conditional at its own order and must not reproduce the structure the corpus gains above it. Both halves are read as signed comparisons against a like-for-like reference, and the second is the one an interval would obscure — a negative share is a qualitative statement that the cohort picks up none of the next order's structure, and it is scale-free by construction, which is why the [scale-free amendment](#which-controls-qualified) is reported beside the pre-declared absolute margin.

So the contract's requirement that a control carry its own contribution with an interval is met here in a different and tighter currency, and the difference is declared rather than left to be inferred. What no form of this qualification establishes is that a generator's declared statistics are the only ones it carries: `fragment` inherits a curated family by descent, which is exactly why it is the binding control and not a null.

**The pre-declared second condition was declared against a wrong premise, and both readings are reported.** The order-axis condition required a cohort to sit at least 0.02 nats per residue *below* the reference one order up. That margin was chosen on the belief that the corpus's own adjacent-order gain runs from 0.1 nats upward. Measured on the length-matched reference it is **0.0062 nats from order 0 to 1, 0.0094 from 2 to 3 and 0.0246 from 4 to 5**, so an absolute 0.02-nat margin is larger than one order step below order 4 and the condition is unreachable there however faithful the sampler is. The declared value is not changed and its outcome is primary: under it `markov_0` and `markov_2` are unqualified and only `markov_4` enters the verdict. Beside it, a named post-hoc amendment asks the scale-free form of the same question — does the cohort capture a non-positive share of the reference's own gain one order up — under which all three qualify. Every endpoint carries a verdict under both sets, and where the two agree nothing hangs on the choice.

## The corpus draw

The `fragment`, `natural` and `markov_k` cohorts all rest on one seeded pass over the whole staged UniRef50, keeping a reservoir of 20,000 canonical records per length stratum together with each stratum's population count. The pass is not a convenience: it is what makes the draw a sample of the corpus rather than a window onto its head. A pilot over the first 200,000 records found **188,555 of 193,431 canonical records above 2,048 residues**, against **282,192 of 59,837,879 corpus-wide (0.47%)** and a corpus mean of 286 residues per record, so a file-order prefix of this corpus is not a sample of it.

The pass reports its own totals and three independent readers agree exactly: **60,315,044 records** and **17,277,105,157 canonical residues**, matching the staged k-mer background's independent count of both to the digit, and **17,282,055,793 residues over all symbols**, matching DIAMOND's separately recorded count of indexed letters for the same file. 477,165 records carry at least one non-canonical symbol and are excluded from the reservoir while their canonical residues remain in the totals.

## Attempt ledgers and denominators

Twenty cells, **16,000 attempts**, of which **14,467 carry a canonical searchable sequence**. Eighteen cells are the unconditioned native operating point at 800 of 800 attempts each: the seventeen expansion arms and the ProGen3-3B native arm. Two are the requested condition of the conditioned arms, at 800 of 3,200 attempts each, drawn class-stratified without replacement under the gate seed from all sixteen class keys the ledger holds — a selection that consults class identity and attempt id and nothing else. Because it uses all sixteen rather than the fourteen and fifteen admitted classes of the conditioned-generation campaign, these two cells are **not comparable to that campaign's published rates** and are not read against them.

The searchable parents run to a median of 400 residues with an interquartile range of 200 to 400, because the shared operating point stops at 400 new tokens and most arms reach it; the length-banded `natural` reference sits at a median of 303 residues [193, 389]. Every other cohort matches its parent's length exactly, verified on all 14,467 pairs with zero mismatches.

Three properties of the denominator are conditions rather than choices. An attempt whose output no oracle could search — empty, non-canonical or otherwise outside the alphabet — stays in the row set with an empty sequence in every cohort, counting as a non-recognition on both sides of every pair; dropping it would turn a rate into a survivor rate. An attempt the oracle searched and assigned nothing stays in the denominator as a measured zero. And unconditioned generation has no requested class, so no cell of it is read against a family-conditioned rate as though the two had solved one task.

Uncertainty is carried on the attempt ledger's own frozen near-duplicate sequence group, 4,000 resamples at seed 20260924, reported as 95% and 97.5% percentile intervals. **What that interval covers, and what it does not, is explicit.** It covers variation across the 800 attempts of one arm at one decoding configuration and one batch-seed stream. It does not cover the choice of decoding configuration, the seed stream as a whole, or retraining. The structure contrasts this gate passes through carry even less: they resample the structure subsample's units with the 800-attempt generation ledger held fixed, so they contain **no generation-seed uncertainty at all**, and that absence is declared here rather than inherited silently.

## Representation readout

None is applicable at this gate. The endpoints are properties of emitted sequences, and reading a frozen representation against them would need an instrument this gate does not build. The programme's representation-side questions are owned by the readout-reassessment gate, and nothing here bears on them.

## Results

The oracle ran once over every cohort of every cell: **115,736 queries in 252 shards**, one `hmmscan` call per shard at the release's gathering thresholds, and every shard bound its own query file by digest before any endpoint was read. Replaying the instrument on the generated sequences reproduces the frozen generation ledgers' own any-profile assignment on **16,000 of 16,000 attempts, with 0 disagreements**, so what is read below is the same assignment the earlier campaigns recorded, on a wider cohort set.

The ledger is 20 cells and **16,000 attempts of 16,000**, of which 14,467 carry a canonical searchable sequence and 13,980 distinct near-duplicate groups carry the resampling weight. The 1,533 attempts no oracle could search stay in every denominator as measured non-recognitions on both sides of every pair.

### The matched statistical generators place almost no mass on a recognisable family

Successes summed over all 20 cells, each cohort on the same 16,000 attempts:

| Cohort | Any curated family | Complete domain |
| --- | ---: | ---: |
| The model | 5,156 | 3,995 |
| `shuffle` | 0 | 0 |
| `markov_0` / `markov_2` / `markov_4` | 1 / 1 / 1 | 0 / 0 / 0 |
| `hydropathy` | 0 | 0 |
| `fragment` | 8,217 | 4,931 |
| `natural`, reference not control | 8,282 | 5,953 |

**This is the gate's first finding and it changes how the rest is read.** Four of the five qualified controls reach the endpoint at most once in 16,000 attempts, so their contrast against a model equals that model's own rate to within one attempt and carries no information the rate does not already carry. The question the `markov_k` axis was built to answer — how much of a recognition rate is corpus residue statistics at conditioning order 0 to 4 — has the answer 1 of 16,000 rather than a contest, and the same holds for the one-parameter hydropathy-matched sampler and for the composition shuffle. The one qualified generator that reaches the endpoint at scale is the corpus fragment, at 8,217 of 16,000 any-family and 4,931 complete-domain against the whole-record reference's 8,282 and 5,953, and it is the binding control on both endpoints.

It also settles the order-axis qualification dispute without a choice: because `markov_0` and `markov_2` reach 1 and 0 successes, every cell returns the identical verdict label under the pre-declared rule and under the named scale-free amendment, **20 of 20 on both endpoints**. Nothing in this gate hangs on which reading of that condition is taken.

### Any curated family

| Cell | Condition | Searchable of attempts | Model rate [Wilson 95%] | − `fragment` [97.5%] | − `natural`, reference [97.5%] | Verdict |
| --- | --- | ---: | --- | --- | --- | --- |
| galactica-1.3b | unconditioned | 598 of 800 | 0.1100 [0.0902, 0.1336] | -0.3813 [-0.5654, -0.2164] | -0.3700 [-0.5548, -0.2101] | beyond some, not the corpus fragment |
| galactica-125m | unconditioned | 9 of 800 | 0.0000 [0.0000, 0.0048] | -0.0013 [-0.3000, +0.0000] | -0.0013 [-0.3000, +0.0000] | not detected beyond any |
| galactica-30b | unconditioned | 678 of 800 | 0.1725 [0.1479, 0.2002] | -0.2650 [-0.3714, -0.1653] | -0.2563 [-0.3674, -0.1494] | beyond some, not the corpus fragment |
| galactica-6.7b | unconditioned | 587 of 800 | 0.1138 [0.0936, 0.1376] | -0.3325 [-0.5042, -0.1893] | -0.3575 [-0.5350, -0.2060] | beyond some, not the corpus fragment |
| instructprotein | unconditioned | 800 of 800 | 0.5875 [0.5530, 0.6211] | +0.0525 [-0.0088, +0.1099] | +0.0288 [-0.0275, +0.0865] | beyond some; fragment contrast unresolved |
| progen2-base | unconditioned | 739 of 800 | 0.2562 [0.2272, 0.2876] | -0.3050 [-0.3755, -0.2353] | -0.3575 [-0.4344, -0.2782] | beyond some, not the corpus fragment |
| progen2-large | unconditioned | 764 of 800 | 0.3725 [0.3397, 0.4065] | -0.2487 [-0.3096, -0.1899] | -0.2725 [-0.3299, -0.2165] | beyond some, not the corpus fragment |
| progen2-medium | unconditioned | 777 of 800 | 0.3750 [0.3421, 0.4091] | -0.2388 [-0.2961, -0.1817] | -0.2662 [-0.3337, -0.2008] | beyond some, not the corpus fragment |
| progen2-small | unconditioned | 784 of 800 | 0.2462 [0.2177, 0.2773] | -0.3850 [-0.4373, -0.3321] | -0.4025 [-0.4565, -0.3498] | beyond some, not the corpus fragment |
| progen2-xlarge | unconditioned | 731 of 800 | 0.4288 [0.3949, 0.4633] | -0.1537 [-0.2173, -0.0953] | -0.2012 [-0.2679, -0.1399] | beyond some, not the corpus fragment |
| progen3-112m | unconditioned | 800 of 800 | 0.3150 [0.2838, 0.3480] | -0.2675 [-0.3225, -0.2112] | -0.2550 [-0.3063, -0.2000] | beyond some, not the corpus fragment |
| progen3-3b | unconditioned | 800 of 800 | 0.6200 [0.5859, 0.6530] | +0.0337 [-0.0200, +0.0863] | +0.0800 [+0.0250, +0.1363] | beyond some; fragment contrast unresolved |
| prollama | requested | 800 of 800 | 0.1237 [0.1027, 0.1484] | -0.2688 [-0.3125, -0.2250] | -0.2262 [-0.2712, -0.1812] | beyond some, not the corpus fragment |
| prollama | unconditioned | 800 of 800 | 0.0450 [0.0327, 0.0617] | -0.3250 [-0.3672, -0.2850] | -0.3113 [-0.3550, -0.2697] | beyond some, not the corpus fragment |
| prollama-stage-1 | unconditioned | 800 of 800 | 0.1700 [0.1456, 0.1976] | -0.3250 [-0.3713, -0.2775] | -0.3512 [-0.3975, -0.3050] | beyond some, not the corpus fragment |
| proteinglm-7b-clm | unconditioned | 800 of 800 | 0.3513 [0.3190, 0.3850] | -0.2125 [-0.2672, -0.1610] | -0.1788 [-0.2313, -0.1277] | beyond some, not the corpus fragment |
| protgpt2 | unconditioned | 800 of 800 | 0.3862 [0.3531, 0.4205] | -0.1787 [-0.2367, -0.1202] | -0.1800 [-0.2396, -0.1199] | beyond some, not the corpus fragment |
| protgpt3-1.3b | unconditioned | 800 of 800 | 0.3950 [0.3617, 0.4293] | -0.1875 [-0.2412, -0.1336] | -0.1713 [-0.2253, -0.1163] | beyond some, not the corpus fragment |
| rita-xl | unconditioned | 800 of 800 | 0.4125 [0.3789, 0.4470] | -0.1863 [-0.2403, -0.1311] | -0.1875 [-0.2428, -0.1338] | beyond some, not the corpus fragment |
| zymctrl | requested | 800 of 800 | 0.9637 [0.9484, 0.9746] | +0.3500 [+0.3090, +0.3925] | +0.3300 [+0.2892, +0.3702] | beyond every qualified generator |

### Complete domain, at 80% of the profile's own model length

| Cell | Condition | Searchable of attempts | Model rate [Wilson 95%] | − `fragment` [97.5%] | − `natural`, reference [97.5%] | Verdict |
| --- | --- | ---: | --- | --- | --- | --- |
| galactica-1.3b | unconditioned | 598 of 800 | 0.0825 [0.0654, 0.1036] | -0.2312 [-0.3648, -0.1251] | -0.2925 [-0.4452, -0.1646] | beyond some, not the corpus fragment |
| galactica-125m | unconditioned | 9 of 800 | 0.0000 [0.0000, 0.0048] | -0.0013 [-0.3000, +0.0000] | -0.0013 [-0.3000, +0.0000] | not detected beyond any |
| galactica-30b | unconditioned | 678 of 800 | 0.1138 [0.0936, 0.1376] | -0.1500 [-0.2379, -0.0617] | -0.2062 [-0.3094, -0.1037] | beyond some, not the corpus fragment |
| galactica-6.7b | unconditioned | 587 of 800 | 0.0650 [0.0499, 0.0842] | -0.2288 [-0.3552, -0.1294] | -0.3050 [-0.4588, -0.1759] | beyond some, not the corpus fragment |
| instructprotein | unconditioned | 800 of 800 | 0.5100 [0.4754, 0.5445] | +0.2213 [+0.1625, +0.2804] | +0.1313 [+0.0702, +0.1898] | beyond every qualified generator |
| progen2-base | unconditioned | 739 of 800 | 0.1800 [0.1549, 0.2081] | -0.1950 [-0.2569, -0.1384] | -0.2800 [-0.3497, -0.2121] | beyond some, not the corpus fragment |
| progen2-large | unconditioned | 764 of 800 | 0.2600 [0.2308, 0.2915] | -0.1625 [-0.2176, -0.1070] | -0.2338 [-0.2913, -0.1757] | beyond some, not the corpus fragment |
| progen2-medium | unconditioned | 777 of 800 | 0.2737 [0.2440, 0.3057] | -0.1387 [-0.1929, -0.0820] | -0.2037 [-0.2595, -0.1448] | beyond some, not the corpus fragment |
| progen2-small | unconditioned | 784 of 800 | 0.1388 [0.1165, 0.1644] | -0.3000 [-0.3492, -0.2518] | -0.3287 [-0.3808, -0.2762] | beyond some, not the corpus fragment |
| progen2-xlarge | unconditioned | 731 of 800 | 0.3187 [0.2874, 0.3518] | -0.0738 [-0.1279, -0.0213] | -0.1438 [-0.2052, -0.0867] | beyond some, not the corpus fragment |
| progen3-112m | unconditioned | 800 of 800 | 0.2500 [0.2212, 0.2812] | -0.1000 [-0.1525, -0.0475] | -0.1275 [-0.1762, -0.0762] | beyond some, not the corpus fragment |
| progen3-3b | unconditioned | 800 of 800 | 0.5375 [0.5029, 0.5718] | +0.1950 [+0.1400, +0.2487] | +0.1475 [+0.0925, +0.2037] | beyond every qualified generator |
| prollama | requested | 800 of 800 | 0.0512 [0.0380, 0.0688] | -0.0888 [-0.1212, -0.0575] | -0.1375 [-0.1725, -0.1025] | beyond some, not the corpus fragment |
| prollama | unconditioned | 800 of 800 | 0.0213 [0.0133, 0.0338] | -0.1037 [-0.1350, -0.0738] | -0.1850 [-0.2248, -0.1475] | beyond some, not the corpus fragment |
| prollama-stage-1 | unconditioned | 800 of 800 | 0.0750 [0.0587, 0.0954] | -0.1725 [-0.2125, -0.1350] | -0.2725 [-0.3138, -0.2313] | beyond some, not the corpus fragment |
| proteinglm-7b-clm | unconditioned | 800 of 800 | 0.2750 [0.2452, 0.3070] | -0.0287 [-0.0811, +0.0225] | -0.1037 [-0.1550, -0.0537] | beyond some; fragment contrast unresolved |
| protgpt2 | unconditioned | 800 of 800 | 0.2875 [0.2572, 0.3198] | -0.0938 [-0.1520, -0.0368] | -0.1638 [-0.2205, -0.1065] | beyond some, not the corpus fragment |
| protgpt3-1.3b | unconditioned | 800 of 800 | 0.2850 [0.2548, 0.3173] | -0.0475 [-0.1000, +0.0025] | -0.1075 [-0.1589, -0.0573] | beyond some; fragment contrast unresolved |
| rita-xl | unconditioned | 800 of 800 | 0.3475 [0.3153, 0.3812] | -0.0063 [-0.0601, +0.0463] | -0.0988 [-0.1536, -0.0460] | beyond some; fragment contrast unresolved |
| zymctrl | requested | 800 of 800 | 0.9213 [0.9005, 0.9380] | +0.5363 [+0.4888, +0.5835] | +0.4650 [+0.4196, +0.5107] | beyond every qualified generator |

Rates are read against the profile sampler's ceiling of **0.892 [0.871, 0.910]** any-family and **0.835 [0.811, 0.857]** complete-domain (Wilson 95%, 1,000 of 1,000), not against 1.000. The ZymCTRL requested cell at 0.9637 [0.9484, 0.9746] and 0.9213 [0.9005, 0.9380] sits at or above that ceiling on both endpoints, which is a statement about a threshold on a bit score rather than about exceeding a maximum.

The 80% cutoff does not carry the complete-domain reading. Where the oracle assigns a profile at all, the best single domain instance covers a median **0.98** of that profile's model length for the model's own sequences, 0.9064 for `fragment` and 0.968 for `natural`, over 4,741, 7,905 and 8,027 assigned rows; `shuffle` contributes 0 rows to that distribution because it is assigned no profile to cover.

### Where the recognised attempts sit relative to the searched corpus

Each attempt carries a reported maximum identity to the searched UniRef50 snapshot, in the declared bands. Summed over all 20 cells:

| Band | Attempts | Any family | Complete domain |
| --- | ---: | ---: | ---: |
| 70% and over | 1,392 | 0.8901 | 0.8096 |
| 50% to 70% | 1,614 | 0.7224 | 0.6214 |
| 30% to 50% | 2,483 | 0.6214 | 0.4793 |
| Under 30% | 1,857 | 0.5310 | 0.3199 |
| No reported alignment | 7,121 | 0.0312 | 0.0114 |
| Not searched | 1,533 | 0.0000 | 0.0000 |

Recognition and corpus proximity co-occur strongly: 0.8901 of the attempts with at least 70% reported identity carry a family against 0.0312 of those with no reported alignment. This bounds what the endpoint means rather than measuring retrieval — no training set is accessible here, the searched snapshot is not any arm's training corpus, and `no reported alignment` is its own stratum and not an observed zero-percent identity. The unsearched stratum is 1,533 attempts whose output carried nothing searchable, which is not a distance measurement at all.

## Verdict

**Any curated family.** Of 20 cells, **1 satisfies the requirement beyond every qualified matched generator** — ZymCTRL at the requested condition, +0.3500 [+0.3090, +0.3925] over the corpus fragment. **18 clear every statistical generator and do not clear the corpus fragment**, and two of those 18, InstructProtein at +0.0525 [−0.0088, +0.1099] and ProGen3-3B at +0.0337 [−0.0200, +0.0863], have a fragment contrast whose interval contains zero, so their comparison against the binding control is unresolved rather than negative. **1 is not detected beyond any qualified generator**: Galactica-125M assigns 0 of 800 attempts a family on 9 searchable parents of 800, and its fragment contrast of −0.0013 [−0.3000, +0.0000] rests on one recognised fragment.

**Complete domain.** Of 20 cells, **3 satisfy the requirement beyond every qualified matched generator** — ZymCTRL requested at +0.5363 [+0.4888, +0.5835], InstructProtein at +0.2213 [+0.1625, +0.2804] and ProGen3-3B at +0.1950 [+0.1400, +0.2487] over the corpus fragment, each also above the whole-record reference. **16 clear every statistical generator and not the fragment**, and **1**, Galactica-125M, **is not detected beyond any**. The stronger biological requirement therefore separates the panel differently from the weaker one: two arms whose fragment comparison is unresolved on family assignment resolve above it once a whole curated domain is required.

**Structural plausibility is unresolved for instrument reasons on 18 of the 22 contrast cells, which is 17 of the 19 contrast-reporting arms.** The two counts are the same fact at two units: ProLLaMA and ZymCTRL each report a conditioned cell with its own controls, ProLLaMA also reports an unconditioned expansion cell that inherits, and the remaining 17 arms report only inheriting cells. The four cells that rest on their own natural controls are ProLLaMA and ZymCTRL in the pilot and main conditioned phases, at +53.525 [+45.289, +60.948], +5.924 [+2.491, +10.574], +63.377 [+60.027, +66.421] and +56.677 [+45.897, +62.573] pLDDT points at the class unit; those are licensed as paired within-sequence contrasts and nothing more. The other 18 — every expansion arm's unconditioned cell, including ProLLaMA's and ProLLaMA-Stage-1's, and the ProGen3-3B native cell — inherit calibration from that two-arm panel, and their recorded contrasts of +0.158 to +38.388 points are reported without a per-arm separation claim. Galactica-125M's cell supports one unit and has no estimable interval. No arm's structure cell contributes to this gate's endpoint verdicts.

**What no cell of this gate establishes.** Predictor confidence is not observed folding, expression or function; no exact-sequence experimental measurement exists for any row in this programme; and a curated-profile assignment is a threshold on a bit score against a curated model, not a demonstration that a sequence folds or works.

## Limitations

The intervals cover variation across the 800 attempts of one arm at one decoding configuration and one batch-seed stream, on the ledger's frozen near-duplicate groups at 4,000 resamples and seed 20260924. They do not cover the decoding configuration, the seed stream as a whole, or retraining. The structure contrasts passed through here carry less still: they resample the structure subsample's units with the 800-attempt generation ledger held fixed, so they contain **no generation-seed uncertainty at all**, and the predictor does not repeat bitwise — 0.081 pLDDT points at the median of eight re-folded rows and 0.929 at the worst.

No interval is adjusted for the 20 cells, the 2 endpoints, the 7 cohorts or the two qualification readings. The licensed cells alone are 20 cells times 2 endpoints, and the pattern across cells rather than any single cell's interval is the defensible reading.

Failing the corpus-fragment contrast is not evidence that a decoder carries nothing beyond corpus statistics. The fragment cohort is a substring of a real protein at the attempt's own length, so it inherits a curated family by descent; what 19 of 20 cells clear is the declared corpus-statistics axis, and what 3 cells clear is the fragment as well. The two comparisons answer different questions and neither substitutes for the other.

The two conditioned cells are 800 of 3,200 attempts drawn class-stratified from all sixteen class keys their ledgers hold, so they are **not comparable to the conditioned-generation campaign's published rates**, which admitted fourteen and fifteen classes, and they are not read against them. Assignment to the requested class stays where it already is, and the informative control there is a label-carrying one — corpus records of the requested class — which is a new instrument and is recorded here as a recommendation rather than built.

Galactica-125M's cell is a measured zero on a denominator of 800 with 9 searchable parents; it bounds nothing about the checkpoint beyond this operating point, and its structure and fragment comparisons rest on one unit each.

## Artifacts

The endpoint record is `results/transfer/d1_gate_generative_control/gate_endpoints.json` (SHA-256 `7e3d34d7ef2929eb0c1bcef9487b6e53c27490448aad938f78dd950b44e14246`) with the full seven-control per-arm table beside it as `gate_endpoints.md`. It binds its own inputs by digest: the build manifest `986975c2e45cdcb2…`, the competence receipts `551090838dd90fd3…`, the profile-sampler ceiling `b898309ea033e87e…` and the twenty structure analyses it passes through. The corpus reservoir receipt is `45203346ad7e858b…`. The control cohorts, the oracle tables and the 252 shard files stay under ignored `logs/d1_gate_generative_control/`, and `scripts/transfer/51_generative_control.py` regenerates any of them from the pool receipt and the staged corpus.
