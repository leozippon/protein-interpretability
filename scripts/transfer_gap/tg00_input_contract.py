"""TG-00: certify the input contract before any TG number is believed.

Two defects in this programme were worth more than most of the effects measured
on top of them, and both are properties of how a sequence reaches a model rather
than of the model:

    rendering    ProtGPT2 scored on one unwrapped line instead of the
                 end-of-text-prefixed, 60-column FASTA stream it was pretrained
                 on. Measured at 1.42 nats/token in EXP-R2-028, enough to move
                 its context information from -1.31 to +2.23.
    cohort       Eligible records taken in FASTA file order rather than under a
                 seeded permutation. Measured at +1.01 nats/token in
                 EXP-R2-059, on ProGen2, by reading past the first 48 records.

This script measures both, as positive controls, on whatever arms are named. It
asserts nothing about protein models. It asserts that the instrument is wired up,
and it is meant to be run first and re-run whenever ``src.transfer.arms`` changes
a rendering.

Both quantities are reported as *deltas*, because the absolute level of either
depends on the cohort and the two controls are deliberately measured on different
cohorts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from tg_common import (
    DEFAULT_COHORT_SEED,
    REPO,
    Cohort,
    cohort_for,
    cohort_provenance,
    load_arm,
    write_json,
)
from src.transfer.arms import PANEL, iter_fasta, AA20, SWISSPROT_FASTA


@torch.no_grad()
def mean_cross_entropy(arm, strings: list[str], max_len: int) -> float:
    """Per-token cross-entropy over whole sequences, one at a time.

    Sequence-at-a-time rather than batched: this script compares renderings whose
    token counts differ by a third, and any padding scheme would make the scored
    position set depend on the rendering being compared.
    """

    total, count = 0.0, 0
    for text in strings:
        ids = arm.tokenizer(text, return_tensors=None)["input_ids"][:max_len]
        if len(ids) < 2:
            continue
        block = torch.tensor([ids], dtype=torch.long, device=arm.device)
        logits = arm.model(input_ids=block).logits
        logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
        nll = -logp.gather(-1, block[:, 1:].unsqueeze(-1)).squeeze(-1)
        total += float(nll.sum())
        count += int(nll.numel())
    if count == 0:
        raise RuntimeError(f"{arm.name}: nothing scored")
    return total / count


def file_order_cohort(arm, n: int, res_min: int, res_max: int) -> Cohort:
    """The historical selection rule, rebuilt here only to be measured against.

    This duplicates what ``src.transfer.arms.protein_cohort`` does, deliberately
    and in one place, because the whole point of this control is to quantify the
    rule it is a copy of. Nothing else in the series may select this way.
    """

    if PANEL[arm.name].source != "swissprot":
        raise ValueError(f"{arm.name}: the file-order control is defined on swissprot")
    allowed = set(AA20)
    records = []
    for _, sequence in iter_fasta(SWISSPROT_FASTA):
        if res_min <= len(sequence) <= res_max and set(sequence) <= allowed:
            records.append(sequence)
            if len(records) >= n:
                break
    if len(records) < n:
        raise RuntimeError(f"only {len(records)}/{n} eligible records")
    return Cohort(
        name=f"swissprot_fileorder_n{n}",
        kind="protein",
        records=records,
        min_symbols=res_min,
        max_symbols=res_max,
        metadata={"selection": "file_order"},
    )


def rendering_control(arm, cohort: Cohort, max_len: int) -> dict:
    """Cost of every rendering the panel could plausibly have chosen.

    Only defined where a wrong rendering is possible at all: an arm whose native
    format is ``raw`` has nothing to get wrong, and reporting a zero delta for it
    would read as a passed control rather than as an inapplicable one.
    """

    fmt = PANEL[arm.name].input_format
    if fmt != "fasta_wrapped":
        return {"applicable": False, "reason": f"input_format is {fmt!r}, not fasta_wrapped"}
    eot = arm.tokenizer.eos_token
    raw = list(cohort.records)
    wrapped = ["\n".join(s[i : i + 60] for i in range(0, len(s), 60)) for s in raw]
    variants = {
        "raw_single_line": raw,
        "eot_plus_raw": [eot + "\n" + s for s in raw],
        "wrapped_60": wrapped,
        "native_eot_plus_wrapped": cohort.input_strings(arm),
    }
    scored = {name: mean_cross_entropy(arm, rows, max_len) for name, rows in variants.items()}
    return {
        "applicable": True,
        "ce_nats_by_rendering": scored,
        "native_rendering": "native_eot_plus_wrapped",
        "rendering_delta_nats": scored["raw_single_line"] - scored["native_eot_plus_wrapped"],
        "reference_delta_nats": 1.42,
        "reference": "EXP-R2-028, 80 Swiss-Prot sequences 600-2000 aa, file-order cohort",
    }


def cohort_control(arm, n: int, res_min: int, res_max: int, max_len: int, seed: int) -> dict:
    """Cost of taking the first N eligible records instead of permuting.

    Both cohorts are rendered natively, so the only thing that varies is which
    records were chosen. A non-zero delta here is not noise: it is the corpus's
    first block being unrepresentative, which is what a curated FASTA sorted by
    accession will generally be.
    """

    if PANEL[arm.name].source != "swissprot":
        return {"applicable": False, "reason": f"source is {PANEL[arm.name].source!r}"}
    head = file_order_cohort(arm, n, res_min, res_max)
    permuted = cohort_for(arm, n, res_min, res_max, seed=seed)
    ce_head = mean_cross_entropy(arm, head.input_strings(arm), max_len)
    ce_permuted = mean_cross_entropy(arm, permuted.input_strings(arm), max_len)
    return {
        "applicable": True,
        "ce_file_order_nats": ce_head,
        "ce_seeded_permutation_nats": ce_permuted,
        "cohort_delta_nats": ce_head - ce_permuted,
        "file_order_digest": head.digest,
        "seeded_digest": permuted.digest,
        "reference_delta_nats": 1.01,
        "reference": "EXP-R2-059, ProGen2-medium context information, n=48 skip=0 vs skip=400",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["protgpt2", "progen2-medium"])
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--n-render", type=int, default=80)
    ap.add_argument("--render-min", type=int, default=600)
    ap.add_argument("--render-max", type=int, default=2000)
    ap.add_argument("--n-cohort", type=int, default=200)
    ap.add_argument("--cohort-min", type=int, default=200)
    ap.add_argument("--cohort-max", type=int, default=800)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=DEFAULT_COHORT_SEED)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_root = Path(args.out) if args.out else (
        REPO / "results/transfer_gap_20260729_corrected/tg00"
    )
    for name in args.arms:
        arm = load_arm(name, device=args.device)
        render_cohort = cohort_for(arm, args.n_render, args.render_min, args.render_max,
                                   seed=args.seed)
        payload = {
            "arm": arm.name,
            "modality": arm.modality,
            "input_format": PANEL[arm.name].input_format,
            "rendering_control": rendering_control(arm, render_cohort, args.max_len),
            "cohort_control": cohort_control(arm, args.n_cohort, args.cohort_min,
                                             args.cohort_max, args.max_len, args.seed),
            "rendering_cohort": cohort_provenance(render_cohort, arm),
        }
        write_json(out_root / f"{arm.name}.json", payload)
        rendering = payload["rendering_control"]
        cohort = payload["cohort_control"]
        print(f"[{arm.name}]")
        if rendering["applicable"]:
            for key, value in rendering["ce_nats_by_rendering"].items():
                print(f"    {key:28s} {value:.4f}")
            print(f"    rendering delta            {rendering['rendering_delta_nats']:+.4f} nats/token")
        else:
            print(f"    rendering control n/a: {rendering['reason']}")
        if cohort["applicable"]:
            print(f"    file order                 {cohort['ce_file_order_nats']:.4f}")
            print(f"    seeded permutation         {cohort['ce_seeded_permutation_nats']:.4f}")
            print(f"    cohort delta               {cohort['cohort_delta_nats']:+.4f} nats/token")
        else:
            print(f"    cohort control n/a: {cohort['reason']}")
        del arm
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
