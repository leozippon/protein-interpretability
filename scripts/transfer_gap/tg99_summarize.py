"""TG-99: collate the transfer-screen results into one comparison table.

Reads whichever results root is named on the command line, defaulting to the
corrected 2026-07-29 run. Quantities that the underlying stage refused -- a loss
recovered whose denominator was below its floor, a long-range share whose
information range was -- arrive as ``None`` and are printed as a dash. They are
not filled in, defaulted, or clamped: a refusal is the result.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from tg_common import REPO, write_json

DEFAULT_ROOT = (
    REPO / "results/transfer_gap_20260729_corrected"
)
ARMS = ["gpt2-large", "protgpt2", "zymctrl", "progen2-medium"]
LN2 = math.log(2.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    args = ap.parse_args()
    root = Path(args.root)

    def load(stage: str, name: str):
        path = root / stage / f"{name}.json"
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    rows = {}
    for arm in ARMS:
        row = {}
        d1 = load("tg01", arm)
        if d1:
            row |= dict(
                symbols_per_token=d1["symbols_per_token"],
                unigram_bits_per_token=d1["unigram_entropy_nats"] / LN2,
                model_bits_per_token=d1["model_nll_nats"] / LN2,
                gain_bits_per_token=d1["info_gain_over_unigram_bits"],
                gain_bits_per_symbol=d1["info_gain_bits_per_symbol"],
                frac_uncertainty_resolved=d1["fraction_of_unigram_entropy_explained"],
                top1_accuracy=d1["top1_accuracy"],
                # Taken from the stage rather than recomputed here: the stage
                # owns the denominator guard, and recomputing the ratio in the
                # summary would route around it.
                local_share_within_8=d1["local_fraction_within_8"],
                unigram_plug_in_bias_nats=d1.get("unigram_plug_in_bias_nats"),
                gain_top_decile_share=d1["gain_top_decile_share"],
                markov2_bits_per_residue=d1.get("markov_order2_bits_per_symbol"),
                gain_by_position_bin=d1["gain_by_position_bin_nats"],
            )
        d2 = load("tg02", arm)
        if d2:
            primary = d2[d2["primary_shuffle"]]
            row |= dict(
                far_bits=d2["far_context_information_bits"],
                far_order_bits=primary["far_order_information_bits"],
                far_order_share=primary["far_order_share"],
            )
        for path in sorted((root / "tg03").glob(f"{arm}_L*.json")):
            d3 = json.load(open(path))
            row |= dict(
                sae_layer=d3["layer"],
                sae_fvu=d3["fvu"],
                sae_dead=d3["dead_fraction"],
                sae_loss_recovered=d3["loss_recovered"],
                sae_delta_ce_nats=d3["ce_delta_nats"],
                sae_feature_var_by_token=d3[
                    "feature_variance_explained_by_current_token"
                ]["mean"],
                sae_feature_var_by_position=d3[
                    "feature_variance_explained_by_position"
                ]["mean"],
            )
            row["sae_denominator_valid"] = d3.get("denominator_valid")
            if d1:
                info = d1["unigram_entropy_nats"] - d3["ce_clean_nats"]
                # Same floor as every other share in the series. This one divides
                # the dictionary's cross-entropy cost by the information the model
                # holds over its context-free baseline, and that quantity is small
                # on arms whose baseline is close to their loss.
                row["sae_frac_information_lost"] = (
                    d3["ce_delta_nats"] / info if info >= 0.5 else None
                )
        d5 = load("tg05", arm)
        if d5:
            row |= {f"contact_{k}": v for k, v in d5["anchored_partner_auc"].items()}
        d6 = load("tg06", arm)
        if d6:
            row |= dict(
                transplant_cost_bits=d6["transplant_cost_bits"],
                uniform_cost_bits=d6["uniform_cost_bits"],
            )
            if d1:
                info = d1["unigram_entropy_nats"] - d6["ce_nats"]["clean"]
                # No clamp. The previous `min(1.0, ...)` hid the two ways this
                # ratio fails: a transplant cost exceeding the model's measured
                # context information means the two stages disagree about the
                # cohort, and a small `info` means there is no share to take.
                row["frac_information_from_attention_pattern"] = (
                    d6["transplant_cost_nats"] / info if info >= 0.5 else None
                )
        rows[arm] = row

    channel_path = root / "tg04" / "explanation_channel.json"
    out = {"arms": rows, "results_root": str(root)}
    if channel_path.exists():
        with channel_path.open(encoding="utf-8") as handle:
            out["explanation_channel"] = json.load(handle)
    write_json(root / "SUMMARY.json", out)

    keys = [
        "gain_bits_per_token", "gain_bits_per_symbol", "frac_uncertainty_resolved",
        "unigram_plug_in_bias_nats",
        "top1_accuracy", "local_share_within_8", "far_order_share",
        "sae_fvu", "sae_loss_recovered", "sae_frac_information_lost",
        "sae_feature_var_by_token", "frac_information_from_attention_pattern",
        "contact_partner_marginal_only", "contact_single_concat",
        "contact_attention_pattern",
    ]
    width = max(len(k) for k in keys) + 2
    print(f"{'metric':<{width}}" + "".join(f"{a:>17}" for a in ARMS))
    for key in keys:
        cells = []
        for arm in ARMS:
            v = rows[arm].get(key)
            cells.append(f"{v:>17.4f}" if isinstance(v, (int, float)) else f"{'-':>17}")
        print(f"{key:<{width}}" + "".join(cells))
    print(f"\nwrote {root / 'SUMMARY.json'}")


if __name__ == "__main__":
    main()
