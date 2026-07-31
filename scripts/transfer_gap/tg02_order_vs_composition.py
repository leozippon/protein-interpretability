"""TG-02: is the long-range information order-dependent or order-invariant?

Attribution graphs trace token-to-token computation. They only have a target if
the model's use of distant context depends on *which* symbol sits *where*. This
script splits the far-context contribution into an order-dependent part and an
order-invariant (composition/profile) part.

Every condition keeps the nearest `near` tokens intact, so the prediction point
is locally well formed in all arms and the manipulation is confined to the far
context:

    [ far block: 120 tokens, manipulated ][ near block: 8 tokens, intact ] -> target

    intact    true far block
    shuffled  same far tokens, permuted (composition preserved, order destroyed)
    foreign   far block taken from a different sequence in the cohort

    I_far          = NLL(foreign)  - NLL(intact)      total far-context information
    I_far_order    = NLL(shuffled) - NLL(intact)      order-dependent part
    I_far_compose  = NLL(foreign)  - NLL(shuffled)    order-invariant part

**Corrections against the 2026-07-24 run.** Rendering and cohort now come from
`src.transfer.arms` via `tg_common`, so ProtGPT2 is scored on its FASTA stream and
records are a seeded permutation rather than the corpus head.

That correction creates a manipulation problem this script did not previously
have, and it is handled rather than absorbed. Under a native rendering the far
block contains **separator tokens** -- a newline every 60 residues for ProtGPT2 --
and permuting the block destroys the line structure along with the residue order.
The resulting "order information" would then be partly the cost of malformed
FASTA, which is a fact about the format and not about whether the model's use of
distant context is order-dependent. Two shuffles are therefore scored:

    shuffled_all       every far token permuted (the original manipulation)
    shuffled_symbols   only alphabet-bearing tokens permuted, separators held in
                       place, so the block stays well-formed and only residue
                       order is destroyed

Which tokens are alphabet-bearing is `tg_common.token_carries_symbol`, imported
here through `symbol_token_ids`. It used to be restated in this file, deciding
which tokens the primary estimand permutes from a second copy of a predicate
`tg_common` already owned (Appendix B rule 12).

For an arm whose rendering has no separators the two are the same manipulation
and the two numbers agree; the difference between them is the price of the
format. `shuffled_symbols` is the primary, because it is the one that answers the
question the script was written to ask.

The order share `I_far_order / I_far` is also guarded: on ZymCTRL `I_far` was
0.31 nats, and a ratio against that is not a share.

**A third correction, to the foreign condition itself.** The donor offset was
uniform over the whole donor sequence while the intact far block always sits at
`[q - 128, q - 8)`, so a foreign block could begin at index 0 -- ProtGPT2's
end-of-text prefix and first FASTA line, or ZymCTRL's EC tag, `<sep>` and
`<start>` -- against an intact block that contains none of that. `I_far` was
then far-context information plus a format artefact, and it is the denominator
of `far_order_share`. The foreign block now occupies the same absolute index
range as the intact one, and every query is drawn far enough into its sequence
that neither block can start inside the conditioning prompt.
"""

from __future__ import annotations

import argparse
import inspect
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tg_common import (
    DEFAULT_COHORT_SEED,
    REPO,
    cohort_for,
    cohort_provenance,
    load_arm,
    symbol_token_ids,
    write_json,
)
from tg_contract import stage_contract_record
from src.transfer.arms import CONDITIONING_START, PANEL, conditioning_boundary_ids

LN2 = math.log(2.0)

#: Floor on ``I_far`` before ``I_far_order / I_far`` is reported as a share.
MIN_FAR_INFORMATION_NATS = 0.5


def conditioning_prefix_length(arm, ids: list[int]) -> int:
    """How many leading tokens of one rendering are the conditioning prompt.

    Zero for a raw rendering; one for ProtGPT2's end-of-text prefix and ProGen2's
    N-to-C control token; for ZymCTRL, everything up to and including ``<start>``
    -- the EC tag, the ``<sep>`` and the marker -- which is a variable number of
    tokens because EC numbers tokenise differently.

    The boundary token id comes from ``arms.conditioning_boundary_ids``, the
    declaration that emits the markers in the first place, rather than from a
    second pair of literals here.
    """

    start_id, _ = conditioning_boundary_ids(arm)
    if start_id is not None:
        if start_id not in ids:
            raise ValueError(
                f"{arm.name}: rendering carries no {CONDITIONING_START!r} token, so "
                "the conditioning prompt cannot be located and a far block cannot be "
                "kept clear of it"
            )
        return ids.index(start_id) + 1
    return 0 if PANEL[arm.name].input_format == "raw" else 1


def shuffle_symbols_only(block: list[int], symbols: set[int], rng) -> list[int]:
    """Permute the alphabet-bearing tokens of ``block``, holding the rest fixed."""

    slots = [i for i, token in enumerate(block) if token in symbols]
    if len(slots) < 2:
        return list(block)
    permuted = list(block)
    values = rng.permutation([block[i] for i in slots])
    for slot, value in zip(slots, values):
        permuted[slot] = int(value)
    return permuted


@torch.no_grad()
def score(arm, blocks: np.ndarray, batch: int) -> np.ndarray:
    """NLL of the final column given the preceding columns."""
    trimmed = "logits_to_keep" in inspect.signature(arm.model.forward).parameters
    if not trimmed and arm.model.config.vocab_size > 1024:
        raise RuntimeError(f"{arm.name}: large vocab without logits_to_keep support")
    out = []
    for start in range(0, len(blocks), batch):
        chunk = torch.tensor(blocks[start : start + batch], dtype=torch.long).to(arm.device)
        kwargs = {"logits_to_keep": 1} if trimmed else {}
        logits = arm.model(input_ids=chunk[:, :-1], **kwargs).logits
        logp = F.log_softmax(logits[:, -1].float(), dim=-1)
        out.append(-logp.gather(-1, chunk[:, -1:]).squeeze(-1).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-seq", type=int, default=400)
    ap.add_argument("--far", type=int, default=120)
    ap.add_argument("--near", type=int, default=8)
    ap.add_argument("--n-query", type=int, default=6)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--seed", type=int, default=DEFAULT_COHORT_SEED)
    ap.add_argument("--res-min", type=int, default=400)
    ap.add_argument("--res-max", type=int, default=1000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    arm = load_arm(args.arm, device=args.device)
    cohort = cohort_for(arm, args.n_seq, args.res_min, args.res_max, seed=args.seed)
    texts = cohort.input_strings(arm)
    symbols = symbol_token_ids(arm)

    ctx = args.far + args.near
    encoded = [
        arm.tokenizer(t, return_tensors=None)["input_ids"][: args.max_len] for t in texts
    ]
    encoded = [ids for ids in encoded if len(ids) >= ctx + 2]
    if len(encoded) < 50:
        raise RuntimeError(f"{arm.name}: only {len(encoded)} usable sequences")

    # How many leading tokens of each rendering are the conditioning prompt, so
    # that no block -- intact or foreign -- can start inside one.
    prompt = np.asarray([conditioning_prefix_length(arm, ids) for ids in encoded])
    lengths = np.asarray([len(ids) for ids in encoded])
    index = np.arange(len(encoded))

    rng = np.random.default_rng(args.seed)
    intact, shuffled_all, shuffled_symbols, foreign = [], [], [], []
    separator_tokens_in_far = 0
    skipped_no_donor = 0
    for i, ids in enumerate(encoded):
        # The first admissible query keeps the far block clear of the
        # conditioning prompt: the block spans [q - ctx, q - near), so q - ctx
        # must be at or past the end of the prompt.
        lo = ctx + int(prompt[i])
        if len(ids) <= lo:
            continue
        picks = rng.choice(
            np.arange(lo, len(ids)), size=min(args.n_query, len(ids) - lo), replace=False
        )
        for q in picks:
            # The foreign far block occupies the *same absolute index range* as
            # the intact one it replaces. The comment here has always said "same
            # position band" while the code drew a uniform offset over the whole
            # donor, so `off` could be 0: for ProtGPT2 that hands the model the
            # end-of-text prefix and the first FASTA line, and for ZymCTRL the EC
            # tag, `<sep>` and `<start>` -- a conditioning prompt priced at 1.73
            # nats of leak (EXP-R2-034) -- in a block whose intact counterpart
            # never contains any of it. `I_far = NLL(foreign) - NLL(intact)` was
            # then far-context information plus a format artefact, and `I_far` is
            # the denominator of `far_order_share`.
            off = int(q) - ctx
            eligible = np.flatnonzero(
                (lengths >= off + args.far) & (prompt <= off) & (index != i)
            )
            if eligible.size == 0:
                # No donor reaches this far into a sequence without either
                # running out or starting inside its own prompt. Dropping the
                # query is the only alternative to a mismatched block.
                skipped_no_donor += 1
                continue
            other = encoded[int(rng.choice(eligible))]

            window = ids[q - ctx : q + 1]
            far, near, target = window[: args.far], window[args.far : ctx], window[ctx:]
            separator_tokens_in_far += sum(1 for t in far if t not in symbols)
            intact.append(far + near + target)
            shuffled_all.append([int(x) for x in rng.permutation(far)] + near + target)
            shuffled_symbols.append(shuffle_symbols_only(far, symbols, rng) + near + target)
            foreign.append(other[off : off + args.far] + near + target)

    conditions = {
        "intact": intact,
        "shuffled_all": shuffled_all,
        "shuffled_symbols": shuffled_symbols,
        "foreign": foreign,
    }
    if not intact:
        raise RuntimeError(
            f"{arm.name}: every query was dropped for want of a donor long enough to "
            "supply a far block at the same absolute index range"
        )
    widths = {len(row) for rows in conditions.values() for row in rows}
    if widths != {ctx + 1}:
        raise ValueError(f"ragged conditions: {sorted(widths)}")
    counts = {name: len(rows) for name, rows in conditions.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"conditions are not paired query for query: {counts}")

    nll = {
        name: score(arm, np.asarray(rows, dtype=np.int64), args.batch)
        for name, rows in conditions.items()
    }
    n = len(intact)
    i_far = float((nll["foreign"] - nll["intact"]).mean())

    def sem(x):
        return float(x.std(ddof=1) / math.sqrt(x.size))

    def decomposition(shuffle_name: str) -> dict:
        order = float((nll[shuffle_name] - nll["intact"]).mean())
        compose = float((nll["foreign"] - nll[shuffle_name]).mean())
        return {
            "nll_shuffled_nats": float(nll[shuffle_name].mean()),
            "far_order_information_bits": order / LN2,
            "far_composition_information_bits": compose / LN2,
            # A share of a quantity that is itself near zero is not a share. On
            # the 2026-07-24 run ZymCTRL's far-context information was 0.31 nats
            # and its "order share" of 0.786 was arithmetic on that.
            "far_order_share": (
                order / i_far if i_far >= MIN_FAR_INFORMATION_NATS else None
            ),
            "sem_order_bits": sem(nll[shuffle_name] - nll["intact"]) / LN2,
            "frac_queries_order_helps": float(
                (nll[shuffle_name] > nll["intact"]).mean()
            ),
        }

    separator_share = separator_tokens_in_far / (n * args.far)
    payload = dict(
        arm=arm.name,
        modality=arm.modality,
        contract=stage_contract_record("tg02", [arm.name]),
        seed=args.seed,
        n_queries=n,
        n_sequences=len(encoded),
        far_tokens=args.far,
        near_tokens=args.near,
        cohort=cohort_provenance(cohort, arm),
        foreign_block_offset="same_absolute_index_range_as_the_intact_block",
        conditioning_prefix_tokens={
            "min": int(prompt.min()),
            "median": int(np.median(prompt)),
            "max": int(prompt.max()),
        },
        n_queries_dropped_for_want_of_a_donor=skipped_no_donor,
        nll_intact_nats=float(nll["intact"].mean()),
        nll_foreign_nats=float(nll["foreign"].mean()),
        far_context_information_bits=i_far / LN2,
        far_context_information_nats=i_far,
        far_information_floor_nats=MIN_FAR_INFORMATION_NATS,
        far_information_sufficient=i_far >= MIN_FAR_INFORMATION_NATS,
        sem_far_bits=sem(nll["foreign"] - nll["intact"]) / LN2,
        frac_queries_far_helps=float((nll["foreign"] > nll["intact"]).mean()),
        separator_share_of_far_block=separator_share,
        primary_shuffle="shuffled_symbols",
        shuffled_symbols=decomposition("shuffled_symbols"),
        shuffled_all=decomposition("shuffled_all"),
        format_cost_of_all_token_shuffle_bits=(
            float((nll["shuffled_all"] - nll["shuffled_symbols"]).mean()) / LN2
        ),
    )
    out = Path(args.out) if args.out else (
        REPO / "results/transfer_gap_20260729_corrected/tg02"
    )
    write_json(out / f"{arm.name}.json", payload)
    print(f"  {'nll_intact_nats':38s} {payload['nll_intact_nats']:.4f}")
    print(f"  {'far_context_information_bits':38s} {payload['far_context_information_bits']:.4f}")
    print(f"  {'separator_share_of_far_block':38s} {separator_share:.4f}")
    for name in ("shuffled_symbols", "shuffled_all"):
        block = payload[name]
        share = block["far_order_share"]
        print(
            f"  {name:24s} order={block['far_order_information_bits']:+.4f} bits  "
            f"compose={block['far_composition_information_bits']:+.4f} bits  "
            f"share=" + ("refused" if share is None else f"{share:.4f}")
        )
    print(
        f"  {'format cost of all-token shuffle':38s} "
        f"{payload['format_cost_of_all_token_shuffle_bits']:+.4f} bits"
    )


if __name__ == "__main__":
    main()
