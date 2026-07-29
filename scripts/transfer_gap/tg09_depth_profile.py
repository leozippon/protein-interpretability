"""TG-09: was relative depth 0.5 an unfair place to measure?

TG-03 and TG-07 spliced at one site, relative depth 0.5. Loss recovered is
strongly depth-dependent in transformers, and mid-depth is usually the harshest
point, so a single site cannot support a cross-modality claim. This sweeps depth
with the training-free PCA-truncation splice from TG-07, which needs no
dictionary and so isolates the representation from the dictionary recipe.

At each depth it reports the variance captured by a fixed-rank projection and
the cross-entropy that projection recovers. The gap between the two curves is
the variance-behaviour misalignment at that site.

Carries the same three corrections as TG-07, whose machinery it reuses: native
per-arm rendering, seeded-permutation cohorts, and a declared floor on the
mean-ablation denominator. The 2026-07-24 ProtGPT2 profile -- loss recovered near
0.35 at every depth against a headroom near 1.0 nats/token -- is retracted on all
three counts.

A fixed rank is not a fixed *fraction* of the residual basis: 256 of ProGen2's
1536 dimensions is a sixth, against a fifth of GPT-2-large's 1280. The rank is
reported alongside ``d_model`` and the depth profile is read within an arm, never
as a cross-arm level.

**A fourth correction, and it is this stage's largest.** The profile had no
position-subset diagnostic at all: it called TG-07's ``collect`` as
``acts, _, _``, discarding the position indices and token ids, so its variance
curve was the all-position spectrum at every depth and ``alignment_gap`` -- the
whole output of the stage -- was variance-explained-of-the-attention-sink minus
loss recovered. Every depth now reports the four subset spectra TG-07 reports,
the primary names carry the interior alphabet-bearing spectrum, and the
all-position quantities are suffixed.
"""

from __future__ import annotations

import argparse
from pathlib import Path


from tg_common import (
    DEFAULT_COHORT_SEED,
    REPO,
    analysis_layer,
    cohort_for,
    cohort_provenance,
    load_arm,
    loss_recovered,
    write_json,
)
from tg07_variance_behaviour import (
    ALL_POSITION_SPECTRUM_HAZARD,
    PRIMARY_SUBSET,
    collect,
    covariance_of,
    encode_batches,
    position_subsets,
    spectrum_fields,
    spectrum_of,
    splice_ce,
    subset_spectra,
    variance_along,
)
from tg_contract import stage_contract_record

DEPTHS = [0.15, 0.33, 0.50, 0.67, 0.85]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rank", type=int, default=256)
    ap.add_argument("--fit-tokens", type=int, default=200_000)
    ap.add_argument("--fit-seqs", type=int, default=4000)
    ap.add_argument("--eval-seqs", type=int, default=120)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--res-min", type=int, default=120)
    ap.add_argument("--res-max", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=DEFAULT_COHORT_SEED)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    arm = load_arm(args.arm, device=args.device)
    fit_cohort = cohort_for(arm, args.fit_seqs, args.res_min, args.res_max, seed=args.seed)
    eval_cohort = cohort_for(
        arm, args.eval_seqs, args.res_min, args.res_max, skip=args.fit_seqs, seed=args.seed
    )
    fit_texts = fit_cohort.input_strings(arm)
    eval_batches = encode_batches(
        arm, eval_cohort.input_strings(arm), args.max_len, args.batch
    )

    rows = []
    for frac in DEPTHS:
        layer = analysis_layer(arm.n_layer, frac)
        # The positions and token ids are kept. This call used to read
        # `acts, _, _`, discarding both, so this stage had no position-subset
        # diagnostic at any depth and its variance curve -- and therefore every
        # `alignment_gap` in the profile -- was the all-position spectrum TG-07
        # retracted. On GPT-2-large that spectrum is the attention sink: PC1
        # 0.809 against 0.034 on interior alphabet-bearing positions.
        acts, positions, tokens = collect(
            arm, fit_texts, layer, args.max_len, args.batch, args.fit_tokens
        )
        mean = acts.mean(0)
        evals, evecs = spectrum_of(acts)
        subsets = position_subsets(arm, positions, tokens)
        spectra = subset_spectra(arm, acts, subsets)
        share_all = (evals / evals.sum()).numpy()
        primary_along = variance_along(
            evecs, covariance_of(acts[subsets[PRIMARY_SUBSET]])
        )
        share = (primary_along / primary_along.sum()).numpy()
        mean_d = mean.to(arm.device)
        basis = evecs[:, : args.rank].float().to(arm.device)
        del acts

        clean = splice_ce(arm, eval_batches, layer, None)
        ablated = splice_ce(
            arm, eval_batches, layer, lambda h: mean_d.to(h.dtype).expand_as(h)
        )

        def project(h, basis=basis):
            x = h.float() - mean_d
            return ((x @ basis) @ basis.T + mean_d).to(h.dtype)

        projected = splice_ce(arm, eval_batches, layer, project)
        recovered = loss_recovered(clean["all"], ablated["all"], projected["all"])
        row = dict(
            depth_fraction=frac,
            layer=layer,
            **spectrum_fields(spectra, share_all, evals),
            variance_explained=float(share[: args.rank].sum()),
            variance_explained_all_positions=float(share_all[: args.rank].sum()),
            ce_clean_nats=clean["all"],
            ce_mean_ablated_nats=ablated["all"],
            ce_projected_nats=projected["all"],
            ce_clean_nats_symbol_positions=clean["symbol"],
            ce_projected_nats_symbol_positions=projected["symbol"],
            loss_recovered_symbol_positions=loss_recovered(
                clean["symbol"], ablated["symbol"], projected["symbol"]
            )["loss_recovered"],
            **recovered,
        )
        # Built on the interior alphabet-bearing variance, which is the variance
        # of the representation. The all-position version is kept beside it
        # because the 2026-07-24 and 2026-07-29 profiles are that quantity and a
        # reader comparing against them needs to be able to.
        row["alignment_gap"] = (
            None
            if row["loss_recovered"] is None
            else row["variance_explained"] - row["loss_recovered"]
        )
        row["alignment_gap_all_positions"] = (
            None
            if row["loss_recovered"] is None
            else row["variance_explained_all_positions"] - row["loss_recovered"]
        )
        rows.append(row)
        shown = (
            "refused"
            if row["loss_recovered"] is None
            else f"{row['loss_recovered']:+.4f}"
        )
        gap = "     -  " if row["alignment_gap"] is None else f"{row['alignment_gap']:+.4f}"
        print(
            f"  depth {frac:.2f} (L{layer:2d})  var={row['variance_explained']:.4f}  "
            f"LR={shown}  gap={gap}  "
            f"headroom={row['ablation_headroom_nats']:.3f} nats",
            flush=True,
        )

    out = Path(args.out) if args.out else (
        REPO / "results/transfer_gap_20260729_corrected/tg09"
    )
    write_json(
        out / f"{arm.name}.json",
        dict(arm=arm.name, modality=arm.modality, rank=args.rank, seed=args.seed,
             contract=stage_contract_record("tg09", [arm.name]),
             d_model=arm.d_model, n_layer=arm.n_layer,
             spectrum_positions=PRIMARY_SUBSET,
             all_position_spectrum_hazard=ALL_POSITION_SPECTRUM_HAZARD,
             fit_cohort=cohort_provenance(fit_cohort, arm),
             eval_cohort=cohort_provenance(eval_cohort, arm),
             profile=rows),
    )
    scored = [r for r in rows if r["loss_recovered"] is not None]
    if not scored:
        print("  every depth refused its denominator; no profile is defined for this arm")
    else:
        best = max(scored, key=lambda r: r["loss_recovered"])
        print(f"  best depth {best['depth_fraction']} at LR {best['loss_recovered']:.4f}")


if __name__ == "__main__":
    main()
