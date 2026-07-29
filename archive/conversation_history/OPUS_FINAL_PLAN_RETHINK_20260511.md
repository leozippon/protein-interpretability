# Prompt for Opus: Final No-Wet-Lab Plan Rethink

Date: 2026-05-11

## Scope Decision

For the two current manuscripts, we will **not include wet-lab experiments**.
Wet-lab validation may be considered in a later follow-up project or companion
paper, but it should not be required for the current two-paper plan.

Please rethink the next-phase strategy under this constraint and produce a
final executable plan.

## Current Intended Paper Split

1. **R1 paper:** variant perturbation, indel capability, AlphaMissense/gMVP/ESM-1v
   comparison, SAE residual interpretation, and honest mechanism diagnostics.
2. **R2 paper:** CLT/circuit-tracing diagnostics for protein language models,
   likely reframed away from steering/drug design toward interpretable sparse
   representations, feature atlases, and possibly safety/quality diagnostics.

## Hard Constraints

- No wet-lab claim, wet-lab validation, enzyme assay, MAVE pilot, or clinical
  partner-dependent result should be required for either current manuscript.
- Tier 3 wet-lab directions can be listed only as future work, not as acceptance
  criteria for the two current papers.
- PrimateAI-3D should be treated as unavailable and should not block R1.
- Do not reopen broad scalar pathogenicity competition as the main R1 story:
  AlphaMissense and gMVP outperform SAE+LLR on scalar pathogenicity.
- Do not claim robust variant-level LOF/GOF/DN prediction across proteins:
  protein-level CV is near random.
- Do not revive R2 steering/drug-design as a headline unless a purely
  computational result clearly justifies it; current steering evidence is
  negative.

## Facts That Must Be Respected

### R1

- SAE+LLR pathogenicity is useful but not state of the art versus AlphaMissense
  and gMVP.
- Protein-level mechanism CV is weak: SAE macro-AUC around 0.516.
- ProteinGym is a negative diagnostic: sign-corrected ensemble win rate is
  below the target.
- Channelopathy concordance is below target: 0.625 accuracy, DN variants often
  collapse to LOF.
- Indel prediction is the most unique computational capability: the current
  supported indel run scored 6,649 records with damage AUC around 0.7735, below
  the previous 0.85 target but not directly covered by AlphaMissense/gMVP/ESM-1v.
- Available baseline table should use SAE-LR, ESM-2 LLR, SAE+LLR,
  AlphaMissense, gMVP, and ESM-1v only.

### R2

- Hook plumbing works and interventions can change logits.
- Direct-effect feature selection and TopK-aware on-manifold steering were run.
- Steering result is negative: 0/8 EC classes show significant positive shifts.
- Lysozyme selected leads pass Pfam/CLEAN/Foldseek checks, but generation-wide
  steered-vs-unsteered lift is weak or absent.
- The EC metric stack is calibrated for lysozyme real-vs-random controls, so
  the steering null is not simply a broken-metric artifact.
- R2 should likely pivot to CLT features as interpretable sparse
  representations, universal feature atlases, and computational safety/quality
  diagnostics.

## Questions for Opus to Answer

1. What is the strongest **no-wet-lab** R1 manuscript story that remains
   scientifically honest and useful?
2. Which R1 experiments should still be run before freezing the manuscript?
   Prioritize only experiments that can materially change the paper.
3. Should the R1 indel work be expanded now, and if so, what is the minimal
   computational expansion that is worth the H200 cost?
4. How should SAE residual interpretation versus AlphaMissense be framed so it
   is not overstated?
5. Should gene-level mechanism prediction be tested as a replacement for the
   failed variant-level mechanism claim, or is that a distraction?
6. What is the strongest **no-wet-lab** R2 manuscript story after the steering
   no-go result?
7. Which R2 computational experiments are necessary to justify the pivot to
   interpretable sparse representations?
8. Are toxin detection and hallucination/quality detection strong enough for
   the R2 paper, or should they be deferred to a separate safety follow-up?
9. What should be removed from the current TODO plan because it depends on
   wet-lab validation or because it is unlikely to change the manuscripts?
10. What is the final three-week execution plan, with explicit stop/go criteria
    and no wet-lab dependencies?

## Desired Output From Opus

Please return:

- A final two-paper thesis statement.
- A prioritized list of experiments to run, with required inputs, expected
  outputs, acceptance criteria, and stop criteria.
- A list of experiments to explicitly drop or defer.
- A revised manuscript outline for R1 and R2.
- A risk register explaining which claims are fragile and how to phrase them
  conservatively.
- A final compute/resource plan for H200 usage.
