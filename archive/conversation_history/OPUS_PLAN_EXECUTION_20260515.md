# Opus Plan Execution Status (2026-05-15 Plan)

This file tracks execution of `OPUS_NEXT_20260515.md`.

## Executive Status

- M-2 synthesis was implemented and rerun locally from the final M-1
  `n_perm=2000` outputs.
- The optional M-1 rerun with `n_perm=2000` completed on the 1-GPU H200
  hold pod, with top-100 firing rows saved and pulled back locally.
- R2 manuscript text was updated to the current Opus framing:
  cross-model statistical conservation, negative biological/downstream probes,
  and M-2 characterization synthesis.
- Added a manuscript evidence index linking the main R1/R2 numeric claims to
  their local result files.
- R1 and R2 manuscripts both compile successfully after the local edits.
- IndelMissense v1 now includes an explicit CC-BY-4.0 license file and README
  license note.
- The temporary Windows/company jump-host SSH outage recovered. Final
  `n_perm=2000` outputs were pulled back locally and used for the final M-2
  synthesis.
- The idle Opus H200 hold pod was released after final status checks.

## M-2 Synthesis

Script:

- `Research2/scripts/36_triplet_synthesis.py`

Final local output from M-1 `n_perm=2000` tables:

- `Research2/results/circuit_analysis/triplet_synthesis_20260515_nperm2000/summary.md`
- `Research2/results/circuit_analysis/triplet_synthesis_20260515_nperm2000/summary.json`
- `Research2/results/circuit_analysis/triplet_synthesis_20260515_nperm2000/cross_test_overlap.tsv`
- `Research2/results/circuit_analysis/triplet_synthesis_20260515_nperm2000/triplet_signatures.tsv`
- `Research2/results/circuit_analysis/triplet_synthesis_20260515_nperm2000/kmer_motifs.tsv`
- `Research2/results/circuit_analysis/triplet_synthesis_20260515_nperm2000/positional_profiles.tsv`

Key current results:

| Quantity | Result |
|---|---:|
| Triplets with at least one significant test | 37 / 38 |
| Triplets with three or more significant tests | 21 / 38 |
| k-mer significant | 27 |
| positional significant | 35 |
| high-norm significant | 25 |
| attention-sink significant | 4 |
| BPE-boundary significant | 1 |

Key overlaps:

| Pair | Both significant | Jaccard |
|---|---:|---:|
| k-mer + positional | 25 | 0.676 |
| positional + high-norm | 23 | 0.622 |
| k-mer + high-norm | 19 | 0.576 |

Attention-sink subset:

- T011, T018, T023, T025.

Important discrepancy versus Opus estimate:

- Opus estimated 28 / 38 triplets significant on three or more tests.
- The final executed M-2 result is 21 / 38 using the `n_perm=2000` M-1 rerun
  and the finalized v2 one-sided BPE-boundary test. The earlier two-sided BPE
  association would have counted boundary depletion as significant; v2
  correctly does not.

## M-1 `n_perm=2000` Rerun

Remote job:

- Pod: `jiaotongdamoxing-zhk-zip-opus-hold-1gpu-0513-master-0`
- Launch PID reported: `49`; active Python PID observed: `51`.
- Command:
  `CUDA_VISIBLE_DEVICES=0 python3 Research2/scripts/35_triplet_characterization.py --device cuda --out-dir Research2/results/circuit_analysis/triplet_characterization_20260515_nperm2000 --n-perm 2000 --top-position-rows 100`
- Log:
  `Research2/logs/runtime/r2_triplet_characterization_20260515_nperm2000.log`

Final status:

- ProtGPT2 700 / 700 complete.
- ZymCTRL 700 / 700 complete.
- ProGen2-medium 700 / 700 complete.
- Analysis completed all 38 triplets.
- Runtime: 583.8 seconds.
- New permutation floor is approximately 0.0005, fixing the old `n_perm=200`
  p-value floor.

Pulled local outputs:

- `Research2/results/circuit_analysis/triplet_characterization_20260515_nperm2000/summary.json`
- `Research2/results/circuit_analysis/triplet_characterization_20260515_nperm2000/summary.md`
- `Research2/results/circuit_analysis/triplet_characterization_20260515_nperm2000/triplet_characterization.tsv`
- `Research2/results/circuit_analysis/triplet_characterization_20260515_nperm2000/top_firing_positions.tsv`
- `Research2/results/circuit_analysis/triplet_characterization_20260515_nperm2000/cohort.json`
- `Research2/logs/runtime/r2_triplet_characterization_20260515_nperm2000.log`

## Manuscript Updates

R2 manuscript updated:

- `manuscripts/nature_methods_r2_circuit_diagnostics/main.tex`
- `manuscripts/nature_methods_r2_circuit_diagnostics/README.md`
- `manuscripts/README.md`

Main changes:

- Title and abstract now reflect the final thesis:
  "cross-model conservation without biological convergence."
- Added a three-model atlas section with:
  - 38 triplets at `|r| >= 0.90`
  - 30 at `|r| >= 0.95`
  - 8 at `|r| >= 0.98`
  - 30x null mean 0.067, max 1
  - Swiss-Prot rich-label MI failure
  - triplet-basis probe failures versus ESM-2
- Added M-2 characterization synthesis section with final `n_perm=2000`
  full-signature counts.
- Added methods for atlas discovery and triplet characterization.
- Added a top-level manuscript evidence index for the main R1/R2 numeric
  claims and result-file pointers.

Compilation:

- R1 manuscript compiled successfully to a 7-page PDF.
- R2 manuscript compiled successfully to an 8-page PDF.

## IndelMissense v1 Packaging

Updated:

- `data/indelmissense/v1/LICENSE`
- `data/indelmissense/v1/README.md`

The package now states CC-BY-4.0 for benchmark packaging, deterministic splits,
and BioCC-generated baseline scores, while noting that users must also comply
with upstream resource terms.

## Current Status

No experiment blocker remains for the Opus 2026-05-15 plan. After confirming
0 MiB / 0% GPU use and no active experiment process, the H200 hold pod
`jiaotongdamoxing-zhk-zip-opus-hold-1gpu-0513-master-0` was released.
