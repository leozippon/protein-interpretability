"""TG-07: is reconstruction variance a valid proxy for behavioural fidelity?

Every dictionary-quality gate in this project (and in the wider SAE literature)
is a variance criterion: FVU, variance explained, MSE. That is only meaningful
if the directions carrying variance are the directions carrying behaviour.

This is a training-free test of exactly that. The residual stream at one layer is
projected onto its top-r principal components and spliced back into the forward
pass. Sweeping r traces variance explained against cross-entropy recovered. In a
representation where variance and behaviour align, the two curves track each
other; where they do not, a dictionary can look excellent on FVU while destroying
the computation.

**Three corrections against the 2026-07-24 run, whose ProtGPT2 row is retracted.**

1. *Rendering.* ProtGPT2 was scored on the plain sequence. It is now scored on
   the end-of-text-prefixed, 60-column FASTA stream it was pretrained on, via
   ``src.transfer.arms``. Worth 1.78 nats/token on this cohort (TG-00).
2. *Cohort.* Records were taken in FASTA file order. They are now a seeded
   permutation of the whole eligible set.
3. *Denominator.* ``loss_recovered`` divides by ``ce_mean_ablated - ce_clean``,
   which on the earlier ProtGPT2 row was about 1.0 nats/token against
   GPT-2-large's 6.9. A ratio against a small headroom is what produced the
   -0.105 at rank 512 and the -1.75 at rank 8; the quantity is now refused below
   a declared floor rather than reported.

A fourth quantity is added rather than corrected. A native rendering puts
non-residue tokens into the scored stream -- ProtGPT2's newline every 60
residues, ZymCTRL's EC tag and markers -- and a dissociation carried by those
positions would be a statement about FASTA formatting, not about protein
representation. Every cross-entropy is therefore reported twice: over all scored
positions, and over the positions whose target token carries at least one
residue.

**A fourth correction, against this stage's own corrected run.** The spectrum
this stage published under the names ``participation_ratio``, ``variance_top1``
and ``rank_for_90pct_variance`` was measured over *all* positions, and the audit
retracted the headline comparison built on it. Those names now carry the
interior, alphabet-bearing spectrum; the all-position numbers are suffixed
``_all_positions`` and carry their hazard text. See :data:`PRIMARY_SUBSET` for
the sizes involved -- they are not a rounding.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from tg_common import (
    DEFAULT_COHORT_SEED,
    REPO,
    analysis_layer,
    cohort_for,
    cohort_provenance,
    load_arm,
    loss_recovered,
    symbol_position_mask,
    tokenize_batch,
    write_json,
)
from tg_contract import stage_contract_record

RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def encode_batches(arm, texts: list[str], max_len: int, batch: int) -> list[dict]:
    """Tokenise once and keep the scored-position masks with the ids.

    The sweep runs a dozen forward conditions over the same evaluation set, and
    re-tokenising per condition both wastes time and leaves open the possibility
    that two conditions score different positions.
    """

    out = []
    for start in range(0, len(texts), batch):
        ids, mask = tokenize_batch(arm, texts[start : start + batch], max_len)
        ids, mask = ids.to(arm.device), mask.to(arm.device)
        valid = (mask[:, 1:] * mask[:, :-1]).bool()
        symbol = symbol_position_mask(arm, ids[:, 1:]) & valid
        out.append({"ids": ids, "mask": mask, "valid": valid, "symbol": symbol})
    return out


@torch.no_grad()
def collect(arm, texts: list[str], layer: int, max_len: int, batch: int,
            cap: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Residual stream after ``layer``, with each activation's position index.

    Tokenises as it goes and stops at ``cap``, so a fit cohort sized generously
    against an unknown tokens-per-sequence ratio costs nothing when the ratio
    turns out to be high.

    Positions and token ids are returned because the panel is not symmetric in
    what occupies its structurally special positions. Three of four arms receive
    a leading control token under their native rendering -- ProtGPT2 an
    end-of-text, ProGen2 its N-to-C token, ZymCTRL an EC tag -- while
    GPT-2-large's OpenWebText documents begin with content; and ProtGPT2's FASTA
    wrap puts a newline into roughly one scored position in seventeen. Both kinds
    of position carry anomalously large residual norms, so a spectrum over all
    positions can be dominated by tokens that one arm has and another does not.
    Whether that is happening is a fact to measure, not to assume either way.
    """

    store, acts, poss, toks, total = {}, [], [], [], 0
    handle = arm.blocks()[layer].register_forward_hook(
        lambda _m, _i, out: store.__setitem__(
            "h", out[0] if isinstance(out, tuple) else out
        )
    )
    try:
        for start in range(0, len(texts), batch):
            ids, mask = tokenize_batch(arm, texts[start : start + batch], max_len)
            ids, mask = ids.to(arm.device), mask.to(arm.device)
            arm.model(input_ids=ids, attention_mask=mask)
            keep = mask.bool()
            acts.append(store.pop("h")[keep].float().cpu())
            poss.append(
                torch.arange(ids.shape[1], device=arm.device).expand_as(ids)[keep].cpu()
            )
            toks.append(ids[keep].cpu())
            total += int(mask.sum())
            if total >= cap:
                break
    finally:
        handle.remove()
    out = torch.cat(acts)[:cap]
    positions = torch.cat(poss)[:cap]
    tokens = torch.cat(toks)[:cap]
    if out.shape[0] < cap:
        raise RuntimeError(f"{arm.name}: {out.shape[0]}/{cap} activations")
    return out, positions, tokens


def covariance_of(acts: torch.Tensor) -> torch.Tensor:
    """Centred covariance of one activation set."""

    centred = acts - acts.mean(0)
    return (centred.T @ centred) / centred.shape[0]


def spectrum_of(acts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Descending eigenvalues and eigenvectors of the centred covariance."""

    values, vectors = torch.linalg.eigh(covariance_of(acts).double())
    order = torch.argsort(values, descending=True)
    return values[order].clamp(min=0), vectors[:, order]


def participation_ratio(values: torch.Tensor) -> float:
    return float((values.sum() ** 2 / (values**2).sum()).item())


def variance_along(basis: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    """Variance of a subset along each column of an externally chosen basis.

    ``v.T @ C_subset @ v`` for every column ``v``, which is the variance the
    rank-r truncation of ``basis`` actually removes *from that subset*. The
    sweep's basis is the all-position one, because the splice is applied at
    every position; the variance the sweep reports against loss recovered must
    nonetheless be the variance of the representation, not of the separators.
    Those are two different questions and this function is what keeps them from
    being answered with one number.
    """

    return ((cov.double() @ basis.double()) * basis.double()).sum(0)


#: The positions a residual-stream spectrum is reported on, and the *only* ones a
#: number named ``participation_ratio`` or ``variance_top1`` in these artefacts
#: refers to. Appendix B rule 11, earned here.
#:
#: The all-position spectrum was the primary, plainly-named statistic of this
#: stage and of TG-09, and the audit retracted the comparison built on it. At
#: relative depth 0.5 over 200k activations it reads PC1 0.809 / participation
#: ratio 1.53 for GPT-2-large against 0.034 / 253.4 on interior alphabet-bearing
#: positions; 0.971 / 1.06 for ProtGPT2 against 0.439 / 4.86; 0.182 / 26.99 for
#: ZymCTRL against 0.008 / 577.3. ProtGPT2's headline
#: ``rank_for_90pct_variance = 1`` was the FASTA newline direction. The interior
#: values were present in the artefact all along, nested under a subset key,
#: while the retracted quantity kept the name a reader or a downstream script
#: picks up by default. Renaming is the fix: the numbers were never the problem.
PRIMARY_SUBSET = "interior_symbol_positions"

#: Attached to every all-position spectrum field so the hazard travels with the
#: number instead of living in this comment.
ALL_POSITION_SPECTRUM_HAZARD = (
    "measured over every scored position, including position 0 and every "
    "separator or conditioning token. Three of four TG arms receive a leading "
    "control token and one puts a FASTA newline into roughly one position in "
    "seventeen; both carry anomalously large residual norms, so this is a "
    "statistic about the attention sink and the format, not about the "
    "representation. Reported for traceability against the retracted 2026-07-24 "
    "table and for the sweep, whose splice is applied at every position. Read "
    f"the {PRIMARY_SUBSET!r} fields instead (Appendix B rule 11)"
)

#: Below this an eigendecomposition of a d_model-dimensional covariance is not
#: estimated, it is interpolated.
MIN_SUBSET_ACTIVATIONS = 1000


def position_subsets(arm, positions: torch.Tensor, tokens: torch.Tensor) -> dict:
    """The four position sets a spectrum can legitimately be read on.

    Shared with TG-09 rather than restated there: TG-09 called ``collect`` and
    discarded the positions and tokens outright (``acts, _, _``), so it had no
    subset diagnostic at any depth and its ``alignment_gap`` was built entirely
    on the all-position spectrum this stage retracted.
    """

    interior = positions > 0
    symbol = symbol_position_mask(arm, tokens)
    return {
        "all_positions": torch.ones_like(interior),
        "interior_positions": interior,
        "symbol_positions": symbol,
        "interior_symbol_positions": interior & symbol,
    }


def spectrum_fields(spectra: dict, share_all, evals: torch.Tensor) -> dict:
    """The named spectrum statistics an artefact of this family publishes.

    One declaration, used by TG-07's payload and TG-09's per-depth rows, because
    the defect this closes was a *naming* defect and a naming rule restated in
    two files is a naming rule that will diverge in one of them. The unqualified
    names carry :data:`PRIMARY_SUBSET`; the retracted all-position quantities are
    suffixed and travel with their hazard.
    """

    primary = spectra[PRIMARY_SUBSET]
    return {
        "spectrum_positions": PRIMARY_SUBSET,
        "n_spectrum_activations": primary["n_activations"],
        "participation_ratio": primary["participation_ratio"],
        "variance_top1": primary["variance_top1"],
        "variance_top10": primary["variance_top10"],
        "variance_top64": primary["variance_top64"],
        "participation_ratio_all_positions": participation_ratio(evals),
        "variance_top1_all_positions": float(share_all[0]),
        "variance_top10_all_positions": float(share_all[:10].sum()),
        "variance_top64_all_positions": float(share_all[:64].sum()),
        "all_position_spectrum_hazard": ALL_POSITION_SPECTRUM_HAZARD,
        "spectrum_by_position_subset": spectra,
    }


def subset_spectra(arm, acts: torch.Tensor, subsets: dict) -> dict:
    """One spectrum per position subset, each with the count it was measured on."""

    spectra = {}
    for label, selection in subsets.items():
        if int(selection.sum()) < MIN_SUBSET_ACTIVATIONS:
            raise RuntimeError(
                f"{arm.name}: subset {label!r} has {int(selection.sum())} activations"
            )
        values, _ = spectrum_of(acts[selection])
        share = (values / values.sum()).numpy()
        spectra[label] = {
            "n_activations": int(selection.sum()),
            "variance_top1": float(share[0]),
            "variance_top10": float(share[:10].sum()),
            "variance_top64": float(share[:64].sum()),
            "participation_ratio": participation_ratio(values),
        }
    return spectra


@torch.no_grad()
def splice_ce(arm, batches: list[dict], layer: int, project) -> dict:
    """Cross-entropy with the layer output passed through ``project``.

    Returns the all-position and residue-position means separately; ``project``
    of ``None`` is the clean pass.
    """

    def hook(_module, _inputs, out):
        if project is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        new = project(h)
        return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new

    handle = arm.blocks()[layer].register_forward_hook(hook)
    totals = {"all": 0.0, "symbol": 0.0}
    counts = {"all": 0, "symbol": 0}
    try:
        for item in batches:
            logits = arm.model(input_ids=item["ids"], attention_mask=item["mask"]).logits
            logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
            nll = -logp.gather(-1, item["ids"][:, 1:].unsqueeze(-1)).squeeze(-1)
            totals["all"] += float(nll[item["valid"]].sum())
            counts["all"] += int(item["valid"].sum())
            totals["symbol"] += float(nll[item["symbol"]].sum())
            counts["symbol"] += int(item["symbol"].sum())
    finally:
        handle.remove()
    if counts["symbol"] == 0:
        raise RuntimeError(f"{arm.name}: no residue-bearing scored positions")
    return {key: totals[key] / counts[key] for key in totals} | {
        "n_scored_all": counts["all"],
        "n_scored_symbol": counts["symbol"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--layer-frac", type=float, default=0.5)
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
    layer = analysis_layer(arm.n_layer, args.layer_frac)

    fit_cohort = cohort_for(arm, args.fit_seqs, args.res_min, args.res_max, seed=args.seed)
    eval_cohort = cohort_for(
        arm, args.eval_seqs, args.res_min, args.res_max, skip=args.fit_seqs, seed=args.seed
    )
    fit_texts = fit_cohort.input_strings(arm)
    eval_texts = eval_cohort.input_strings(arm)

    eval_batches = encode_batches(arm, eval_texts, args.max_len, args.batch)
    acts, positions, tokens = collect(
        arm, fit_texts, layer, args.max_len, args.batch, args.fit_tokens
    )

    mean = acts.mean(0)
    evals, evecs = spectrum_of(acts)
    mean_d, evecs_d = mean.to(arm.device), evecs.float().to(arm.device)

    # The position-subset spectra, and the reason the primary is not the
    # all-position one. See PRIMARY_SUBSET.
    subsets = position_subsets(arm, positions, tokens)
    spectra = subset_spectra(arm, acts, subsets)

    # Two cumulative variance curves against the *same* rank-r basis, because the
    # sweep splices at every position and so must project onto the all-position
    # eigenvectors, while the quantity worth comparing against loss recovered is
    # how much of the representation that projection keeps. `share` is therefore
    # the interior alphabet-bearing variance captured by the spliced basis, and
    # `share_all` is the all-position variance it captures -- the retracted
    # quantity, kept under its own name so the 2026-07-24 rows remain traceable.
    share_all = (evals / evals.sum()).numpy()
    primary_along = variance_along(evecs, covariance_of(acts[subsets[PRIMARY_SUBSET]]))
    share = (primary_along / primary_along.sum()).numpy()

    interior, symbol = subsets["interior_positions"], subsets["symbol_positions"]
    norms = acts.norm(dim=1)
    first_position_norm_ratio = float(norms[~interior].mean() / norms[interior].mean())
    separator_norm_ratio = (
        float(norms[~symbol].mean() / norms[symbol].mean())
        if int((~symbol).sum()) > 0
        else None
    )
    del acts

    clean = splice_ce(arm, eval_batches, layer, None)
    ablated = splice_ce(
        arm, eval_batches, layer, lambda h: mean_d.to(h.dtype).expand_as(h)
    )
    symbol_share = clean["n_scored_symbol"] / clean["n_scored_all"]
    print(
        f"[{arm.name}] L{layer}/{arm.n_layer}  clean {clean['all']:.4f} nats "
        f"(residue positions {clean['symbol']:.4f}, {symbol_share:.3f} of stream)",
        flush=True,
    )

    headroom = loss_recovered(clean["all"], ablated["all"], ablated["all"])
    print(
        f"  mean-ablation headroom {headroom['ablation_headroom_nats']:.4f} nats "
        f"-> denominator {'VALID' if headroom['denominator_valid'] else 'REFUSED'}",
        flush=True,
    )

    rows = []
    for rank in [r for r in RANKS if r <= arm.d_model]:
        basis = evecs_d[:, :rank]

        def project(h, basis=basis):
            x = h.float() - mean_d
            return ((x @ basis) @ basis.T + mean_d).to(h.dtype)

        scored = splice_ce(arm, eval_batches, layer, project)
        row = dict(
            rank=rank,
            variance_explained=float(share[:rank].sum()),
            variance_explained_all_positions=float(share_all[:rank].sum()),
            ce_nats=scored["all"],
            ce_nats_symbol_positions=scored["symbol"],
            **loss_recovered(clean["all"], ablated["all"], scored["all"]),
        )
        row["loss_recovered_symbol_positions"] = loss_recovered(
            clean["symbol"], ablated["symbol"], scored["symbol"]
        )["loss_recovered"]
        rows.append(row)
        recovered = row["loss_recovered"]
        print(
            f"  r={rank:4d}  var={row['variance_explained']:.4f} "
            f"(all pos {row['variance_explained_all_positions']:.4f})  "
            f"ce={scored['all']:.4f}  "
            f"loss_recovered=" + ("refused" if recovered is None else f"{recovered:+.4f}"),
            flush=True,
        )

    def first_rank(key, target):
        hit = [r for r in rows if r.get(key) is not None and r[key] >= target]
        return hit[0]["rank"] if hit else None

    payload = dict(
        arm=arm.name,
        modality=arm.modality,
        contract=stage_contract_record("tg07", [arm.name]),
        layer=layer,
        n_layer=arm.n_layer,
        d_model=arm.d_model,
        seed=args.seed,
        fit_tokens=int(args.fit_tokens),
        fit_cohort=cohort_provenance(fit_cohort, arm),
        eval_cohort=cohort_provenance(eval_cohort, arm),
        ce_clean_nats=clean["all"],
        ce_clean_nats_symbol_positions=clean["symbol"],
        ce_mean_ablated_nats=ablated["all"],
        ce_mean_ablated_nats_symbol_positions=ablated["symbol"],
        ablation_headroom_nats=headroom["ablation_headroom_nats"],
        denominator_valid=headroom["denominator_valid"],
        denominator_floor_nats=headroom["denominator_floor_nats"],
        symbol_position_share=symbol_share,
        n_scored_positions=clean["n_scored_all"],
        n_scored_symbol_positions=clean["n_scored_symbol"],
        # The unqualified spectrum names are the interior, alphabet-bearing ones.
        # They used to be the all-position ones, which is the quantity the audit
        # retracted; the interior values existed only nested inside
        # `spectrum_by_position_subset`, so the retracted number was still what a
        # reader or a downstream script picked up by default.
        **spectrum_fields(spectra, share_all, evals),
        first_position_norm_ratio=first_position_norm_ratio,
        separator_norm_ratio=separator_norm_ratio,
        rank_for_90pct_variance=first_rank("variance_explained", 0.90),
        rank_for_90pct_variance_all_positions=first_rank(
            "variance_explained_all_positions", 0.90
        ),
        rank_for_90pct_loss_recovered=first_rank("loss_recovered", 0.90),
        sweep=rows,
    )
    out_root = Path(args.out) if args.out else (
        REPO / "results/transfer_gap_20260729_corrected/tg07"
    )
    write_json(out_root / f"{arm.name}.json", payload)
    for label, values in spectra.items():
        print(
            f"  spectrum {label:26s} n={values['n_activations']:7d}  "
            f"PC1={values['variance_top1']:.4f}  PC64={values['variance_top64']:.4f}  "
            f"PR={values['participation_ratio']:8.2f}"
        )
    print(
        f"  first-position norm x{first_position_norm_ratio:.1f}  "
        + (
            f"separator norm x{separator_norm_ratio:.1f}"
            if separator_norm_ratio is not None
            else "no separator positions"
        )
    )
    print(
        f"  rank@90%var={payload['rank_for_90pct_variance']} ({PRIMARY_SUBSET})  "
        f"rank@90%var(all positions)="
        f"{payload['rank_for_90pct_variance_all_positions']}  "
        f"rank@90%loss={payload['rank_for_90pct_loss_recovered']}"
    )


if __name__ == "__main__":
    main()
