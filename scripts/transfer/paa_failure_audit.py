#!/usr/bin/env python3
"""Stratified root-cause audit of the PAA census's retrieval gap.

**The question.** ``prediction_addressed.census_causal_agreement`` reports
``hit@20``: how many of the twenty causally largest heads the census's own top
twenty recovers, against that arm's chance level ``k^2/n_heads``.  Every text arm
and the one subword protein arm read well above their own chance; every
residue-tokenised protein arm reads at chance.  Architecture, conditioning,
scale, corpus and symbol granularity have each been excluded by a control.  What
has never been measured is the *proximal* cause -- which property of an instance
or of a head makes the ranking uninformative on the failing arms.

**Pre-registration.**  The factor list below, its predicted signs, the
discovery/held-out split and the pass criterion are fixed in this docstring
before any outcome was measured.  What had been inspected first is only the
*field inventory* of the retained artefacts (which is unavoidable -- one cannot
name a covariate the pool does not store) and, as a by-product of that
inspection, the marginal means of key multiplicity and decoy replacement on four
arms.  No stratified retrieval number was computed before this list was written.

Every census and causal statistic is computed by importing ``src.transfer``;
nothing here re-derives one.  Three defensible reductions of the census matrix
once disagreed by 0.05 on ProtGPT2, which is why the versioned functions exist.

Instance-level factors
----------------------
Measured per instance from ``pool_<arm>.npz`` and reduced to a **sequence**-level
value, because ``census_matrices.npz`` retains ``paa_specific_matched`` per
sequence and not per instance.  The sign is the predicted direction of the
association with retrieval quality (hit@20 above chance).

``margin`` (+)
    ``clean_logit_target - clean_logit_runner_up``.  A copy-suppression knockout
    is read out as a change in the target-versus-runner-up gap.  Where the gap is
    already ~0 the instance can carry no measurable effect, so a pool of
    zero-margin instances would give a causal ranking with nothing in it.
``confidence`` (+)
    ``p(predicted token)``.  The same argument on the probability scale, and the
    quantity the census's own matching gate bins on.
``distance`` (-)
    ``query - antecedent``.  Attention-addressed retrieval is expected to weaken
    with separation; if the failing arms' antecedents are systematically distant
    the census would be scoring an addressing event that does not happen.
``key_multiplicity`` (-)
    Earlier occurrences of the predicted symbol at or above the key floor,
    read from ``census_matrices['keys_per_instance']``.  This is the alphabet-size
    mechanism and the strongest a-priori candidate: over twenty residues the
    "antecedent" is one of many identical keys, so an antecedent-specific
    attention score has no unique addressee, while over a BPE vocabulary the
    antecedent is usually unique.
``decoy_replacement`` (-)
    Fraction of instances whose four drawn decoys are not distinct.  The decoy
    mean is the subtrahend of every head's ``paa_specific`` score, so a
    degenerate decoy set makes the subtrahend noisier.  **This is a proxy.**  The
    pool retains the four drawn decoys and a run-level counter, not the size of
    the eligible decoy pool at each query, so decoy-pool size itself is not
    computable and is not approximated.
``unigram_percentile`` (-)
    Frequency rank of the predicted symbol in the arm's own unigram
    distribution.  A high-frequency symbol has a less specific antecedent.
``local_entropy`` (-)
    Shannon entropy in bits of the 32-token window ending at the query.  A
    high-entropy neighbourhood offers no local structure for a head to address.
``relative_position`` (0, negative control)
    ``query / width``.  No mechanistic story.  It is in the list so that a
    stratification that lifts retrieval for every factor equally is exposed as
    measuring subset size rather than the factor.

Head-level factors -- properties of the causal target itself
------------------------------------------------------------
``snr`` (+)
    ``|delta_m_gap| / SE``, SE taken from the sequence-cluster percentile
    interval that ``cluster_bootstrap`` returns.  The cheapest and most likely
    hypothesis: if almost no head has an effect distinguishable from noise then
    "the twenty causally largest heads" is a ranking of noise, no selector can
    retrieve it, and the finding belongs to the **evaluation interface** rather
    than to the census.  EXP-R2-096 measured this for induction and never for
    copy suppression.
``fraction_ci_excludes_zero`` (+)
    The same question without a normal approximation.
``top20_concentration`` (+)
    Share of total ``|delta_m_gap|`` held by the top twenty heads, relative to
    ``20 / n_heads``.  A flat causal profile makes the top-20 boundary arbitrary
    even when individual effects are real.
``causal_top20_reliability`` (+)
    Intersection of the causal top-20 sets of two independent draws of the same
    arm, via ``top_set_jaccard``.  This is the **ceiling** on hit@20: a selector
    cannot recover a set the measurement does not reproduce, and it shares
    hit@20's chance level ``k^2/n_heads`` exactly.
``census_top20_reliability`` (+)
    The same for the census ranking, to attribute an unstable comparison to the
    side that is unstable.

Not computable from the retained artefacts
------------------------------------------
- Size of the eligible decoy pool per instance (see ``decoy_replacement``).
- Any instance-level stratification of the *census* score: the artefact retains
  per-sequence aggregates only.  Strata are therefore defined on sequence-level
  means of instance factors, and both sides of the comparison use the same
  sequence set.

Design
------
*Split.*  Within an arm the on-condition draws are sorted by path and assigned
alternately: even index to discovery, odd index to held-out.  Alternating rather
than splitting the list in half so that the two run families present for some
arms (``d2c_panel_dfix_*`` and the single-arm ``*_dfix_*`` directories) appear on
both sides.  Hypotheses are formed on discovery; every claim is reported on
held-out.

*Stratified retrieval.*  For a factor, sequences are split at the median of their
factor value.  The census side is recomputed by handing
``census_causal_agreement`` the stratum's rows of
``paa_specific_matched_per_sequence``; the causal side by re-running
``cluster_bootstrap`` over the stratum's clusters and substituting the resulting
per-head ``delta_m_gap`` into the causal records.  Because halving the sequence
set also halves the evidence, every stratum is read against a **size-matched
random-subset null** recomputed the same way; the factor's effect is the stratum
minus that null, not the stratum alone.

*Pass criterion.*  One factor accounting for roughly half the retrieval gap
across the failing protein arms, holding on held-out draws, while the text
positive controls stay at or above ~3.6x their own chance under the same
adjustment.  Anything less is a FAIL and ends iterative patching of this census.

Usage::

    python scripts/transfer/paa_failure_audit.py results --out <dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer.prediction_addressed import (  # noqa: E402
    CENSUS_RETRIEVAL_K,
    census_causal_agreement,
    census_head_scores,
    cluster_bootstrap,
    top_set_jaccard,
)


def _declared_condition() -> dict[str, Any]:
    """The panel condition, imported from the reader that defines it."""

    path = Path(__file__).resolve().with_name("read_paa_panel.py")
    spec = importlib.util.spec_from_file_location("_read_paa_panel", path)
    if spec is None or spec.loader is None:  # pragma: no cover - packaging error
        raise RuntimeError(f"cannot load the panel reader at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.DECLARED_CONDITION)


#: Bootstrap replicates for a per-head interval.  ``cluster_bootstrap`` refuses
#: fewer than 400 at alpha=0.05, and the stratified sweep calls it thousands of
#: times, so the point estimate runs at the floor and the reported SNR intervals
#: run wide.
STRATUM_REPLICATES = 400
SNR_REPLICATES = 2000

#: Half-width of the percentile interval in standard errors.
NORMAL_Z = 1.959963984540054

#: Window, in tokens, for the local-entropy factor.
ENTROPY_WINDOW = 32

#: Size-matched random subsets per draw for the stratification null.
NULL_SUBSETS = 40

#: Tie-breaks averaged over for the depth-only rival selector.
DEPTH_ONLY_SEEDS = 25

#: Predicted sign of each instance-level factor: +1 if a larger value is
#: predicted to help retrieval, -1 if smaller helps, 0 for the negative control.
FACTOR_SIGNS: dict[str, int] = {
    "margin": +1,
    "confidence": +1,
    "distance": -1,
    "key_multiplicity": -1,
    "decoy_replacement": -1,
    "unigram_percentile": -1,
    "local_entropy": -1,
    "relative_position": 0,
}

TEXT_ARMS = frozenset(
    {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "dialogpt-small", "llama-3.2-3b",
     "qwen2.5-0.5b", "bygpt5-medium-en"}
)
FAILING_ARMS = frozenset({"progen2-small", "progen2-base", "progen2-medium", "zymctrl"})


# ------------------------------------------------------------------ discovery


def discover(root: Path, *, any_condition: bool) -> list[dict[str, Any]]:
    """Every arm-draw under ``root`` carrying the four artefacts this needs."""

    condition = _declared_condition()
    found: list[dict[str, Any]] = []
    for report_path in sorted(root.rglob("paa_gate_report.json")):
        directory = report_path.parent
        report = json.loads(report_path.read_text(encoding="utf-8"))
        census = report.get("census")
        if not census:
            continue
        settings = report.get("settings", {})
        on_condition = all(settings.get(key) == value for key, value in condition.items())
        on_condition = on_condition and (
            census.get("a1_candidate_pool", {}).get("layout_tokens_excluded_from_decoys")
            is not None
        )
        if not on_condition and not any_condition:
            continue
        arm = census["arm"]
        needed = {
            "census": directory / "census_matrices.npz",
            "causal": directory / "causal_matrices.npz",
            "causal_json": directory / "causal.json",
            "pool": directory / f"pool_{arm}.npz",
        }
        if not all(path.is_file() for path in needed.values()):
            continue
        found.append(
            {
                "arm": arm,
                "directory": str(directory),
                "on_condition": bool(on_condition),
                "paths": {key: str(path) for key, path in needed.items()},
            }
        )
    return found


def split_draws(draws: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Alternating discovery/held-out assignment, by sorted path within an arm."""

    per_arm: dict[str, list[dict[str, Any]]] = {}
    for draw in draws:
        per_arm.setdefault(draw["arm"], []).append(draw)
    for arm, arm_draws in per_arm.items():
        arm_draws.sort(key=lambda entry: entry["directory"])
        for index, entry in enumerate(arm_draws):
            entry["fold"] = "discovery" if index % 2 == 0 else "held_out"
    return per_arm


# -------------------------------------------------------------- factor tables


def _local_entropy(rows: np.ndarray, sequence: np.ndarray, query: np.ndarray,
                   content_low: int) -> np.ndarray:
    """Shannon entropy (bits) of the ``ENTROPY_WINDOW`` tokens ending at q-1."""

    out = np.empty(sequence.size, dtype=np.float64)
    cache: dict[tuple[int, int], float] = {}
    for index in range(sequence.size):
        seq_id, q = int(sequence[index]), int(query[index])
        key = (seq_id, q)
        value = cache.get(key)
        if value is None:
            begin = max(content_low, q - ENTROPY_WINDOW)
            window = rows[seq_id, begin:q]
            if window.size == 0:
                value = 0.0
            else:
                counts = np.bincount(window)
                probabilities = counts[counts > 0] / window.size
                value = float(-(probabilities * np.log2(probabilities)).sum())
            cache[key] = value
        out[index] = value
    return out


def factor_table(draw: dict[str, Any]) -> dict[str, Any]:
    """Per-instance factor values plus their sequence-level reduction."""

    pool = np.load(draw["paths"]["pool"], allow_pickle=True)
    census = np.load(draw["paths"]["census"])
    rows = pool["rows"]
    n_sequences, width = rows.shape
    sequence = pool["sequence"].astype(np.int64)
    query = pool["query"].astype(np.int64)
    content_low = int(pool["content_low"])
    decoys = pool["decoys"]
    distinct = np.asarray([np.unique(row).size for row in decoys], dtype=np.float64)

    instance = {
        "margin": pool["clean_logit_target"] - pool["clean_logit_runner_up"],
        "confidence": pool["confidence"].astype(np.float64),
        "distance": pool["distance"].astype(np.float64),
        "decoy_replacement": (distinct < decoys.shape[1]).astype(np.float64),
        "unigram_percentile": pool["unigram_percentile"].astype(np.float64),
        "local_entropy": _local_entropy(rows, sequence, query, content_low),
        "relative_position": query.astype(np.float64) / float(width),
    }

    counts = census["instances_per_sequence"].astype(np.float64)
    if counts.size != n_sequences or not np.array_equal(
        counts, np.bincount(sequence, minlength=n_sequences).astype(np.float64)
    ):
        raise ValueError(
            f"{draw['directory']}: the pool and the census disagree on instances per "
            "sequence, so a sequence-level stratum would not be the same population "
            "on the two sides of the comparison"
        )
    per_sequence: dict[str, np.ndarray] = {}
    for name, values in instance.items():
        total = np.bincount(sequence, weights=values, minlength=n_sequences)
        with np.errstate(invalid="ignore", divide="ignore"):
            per_sequence[name] = np.where(counts > 0, total / np.maximum(counts, 1.0), np.nan)
    # Key multiplicity is retained per sequence only, by the census stage.
    keys = census["keys_per_instance"].astype(np.float64)
    per_sequence["key_multiplicity"] = np.where(counts > 0, keys, np.nan)
    instance["key_multiplicity"] = keys[sequence]

    cascade = json.loads(str(pool["cascade"]))
    return {
        "instance": instance,
        "per_sequence": per_sequence,
        "instances_per_sequence": counts,
        "n_instances": int(sequence.size),
        "cascade": {
            key: cascade.get(key)
            for key in (
                "positions_scored",
                "positions_with_eligible_candidate",
                "instances_retained",
                "decoys_drawn_with_replacement",
                "candidates_discarded_by_empty_decoy_pool",
                "content_low",
                "key_floor",
            )
        },
    }


def factor_summary(draw: dict[str, Any], table: dict[str, Any], *, seed: int) -> dict[str, Any]:
    """Sequence-clustered weighted mean and 95% interval for every factor."""

    names = sorted(table["per_sequence"])
    weights = table["instances_per_sequence"]
    keep = weights > 0
    matrix = np.column_stack([table["per_sequence"][name][keep] for name in names])
    finite = np.isfinite(matrix).all(axis=1)
    result = cluster_bootstrap(
        matrix[finite], weights[keep][finite], replicates=SNR_REPLICATES, seed=seed
    )
    return {
        name: {
            "mean": float(result["mean"][index]),
            "q_low": float(result["q_low"][index]),
            "q_high": float(result["q_high"][index]),
        }
        for index, name in enumerate(names)
    }


# ------------------------------------------------------------ head-level SNR


def head_diagnostics(draw: dict[str, Any], *, seed: int) -> dict[str, Any]:
    """Per-head effect SNR, detectability and concentration for one arm-draw."""

    causal = np.load(draw["paths"]["causal"])
    per_cluster = causal["per_cluster_delta_m_gap"]
    weights = causal["cluster_weights"]
    boot = cluster_bootstrap(per_cluster, weights, replicates=SNR_REPLICATES, seed=seed)
    mean = boot["mean"]
    error = (boot["q_high"] - boot["q_low"]) / (2.0 * NORMAL_Z)
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = np.abs(mean) / error
    snr = snr[np.isfinite(snr)]
    excludes_zero = (boot["q_low"] > 0.0) | (boot["q_high"] < 0.0)

    magnitude = np.abs(mean)
    k = min(CENSUS_RETRIEVAL_K, magnitude.size)
    top = np.sort(magnitude)[::-1][:k]
    share = float(top.sum() / magnitude.sum()) if magnitude.sum() > 0 else float("nan")

    rng = np.random.default_rng(seed + 1)
    head_boot = rng.integers(0, snr.size, size=(1000, snr.size))
    medians = np.median(snr[head_boot], axis=1)
    fractions = (snr[head_boot] > 2.0).mean(axis=1)
    return {
        "n_heads": int(mean.size),
        "n_clusters": int((weights > 0).sum()),
        "median_snr": float(np.median(snr)),
        "median_snr_q_low": float(np.quantile(medians, 0.025)),
        "median_snr_q_high": float(np.quantile(medians, 0.975)),
        "fraction_snr_gt_2": float((snr > 2.0).mean()),
        "fraction_snr_gt_2_q_low": float(np.quantile(fractions, 0.025)),
        "fraction_snr_gt_2_q_high": float(np.quantile(fractions, 0.975)),
        "fraction_ci_excludes_zero": float(excludes_zero.mean()),
        "top20_share": share,
        "top20_share_over_uniform": share / (k / magnitude.size) if magnitude.size else float("nan"),
        "max_abs_delta_m_gap": float(magnitude.max()),
    }


def retrieval_diagnostics(draw: dict[str, Any], *, seed: int) -> dict[str, Any]:
    """Where the causal top-20 sits in the census ranking, and a depth-only rival.

    Two readings of the same disagreement.  ``causal_top20_census_rank_percentile``
    is 0 when the census puts the causally largest heads first and 0.5 when it
    orders them no better than a coin.  ``depth_only_hit_at_k`` replaces the
    census score with the head's **layer index** and nothing else: it is the
    retrieval a selector achieves knowing only how deep a head is.  Ties inside a
    layer are broken by seeded jitter and the statistic is the median over
    ``DEPTH_ONLY_SEEDS`` tie-breaks, because a depth-only selector genuinely has
    no basis for choosing among the heads of one layer.
    """

    records = json.loads(Path(draw["paths"]["causal_json"]).read_text(encoding="utf-8"))["heads"]
    matched = np.load(draw["paths"]["census"])["paa_specific_matched_per_sequence"]
    agreement = census_causal_agreement(matched, records)
    grid = census_head_scores(matched)
    census = np.asarray([grid[row["layer"], row["head_index"]] for row in records])
    magnitude = np.asarray([abs(float(row["delta_m_gap"])) for row in records])
    ranks = np.argsort(np.argsort(-census))
    top = np.argsort(-magnitude)[: min(CENSUS_RETRIEVAL_K, magnitude.size)]

    n_layer, n_head = grid.shape
    layers = np.broadcast_to(np.arange(n_layer, dtype=np.float64)[:, None], (n_layer, n_head))
    depth_hits = []
    for offset in range(DEPTH_ONLY_SEEDS):
        rng = np.random.default_rng(seed + 7919 * offset)
        jittered = layers + rng.random((n_layer, n_head)) * 0.999
        depth_hits.append(
            census_causal_agreement(jittered[None, :, :], records)["retrieval"]["hit_at_k"]
        )
    return {
        "hit_at_k": agreement["retrieval"]["hit_at_k"],
        "chance": agreement["retrieval"]["chance"],
        "spearman": agreement["spearman_census_vs_causal_magnitude"],
        "within_layer": agreement["depth_controlled"]["within_layer"],
        "r_layer_census": agreement["depth_controlled"]["r_layer_census_score"],
        "r_layer_causal": agreement["depth_controlled"]["r_layer_causal_magnitude"],
        "causal_top20_census_rank_percentile": float(np.median(ranks[top])) / census.size,
        "depth_only_hit_at_k": float(np.median(depth_hits)),
        "depth_only_hit_at_k_range": [int(min(depth_hits)), int(max(depth_hits))],
    }


def _head_order(records: Sequence[dict[str, Any]]) -> list[int]:
    keyed = sorted(range(len(records)), key=lambda i: (records[i]["layer"], records[i]["head_index"]))
    return keyed


def reliability(draws: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cross-draw stability of the causal and census top-20 sets within an arm."""

    loaded = []
    for draw in draws:
        records = json.loads(Path(draw["paths"]["causal_json"]).read_text(encoding="utf-8"))["heads"]
        order = _head_order(records)
        magnitude = np.asarray([abs(float(records[i]["delta_m_gap"])) for i in order])
        matched = np.load(draw["paths"]["census"])["paa_specific_matched_per_sequence"]
        grid = np.nanmean(matched, axis=0)
        score = np.asarray([grid[records[i]["layer"], records[i]["head_index"]] for i in order])
        loaded.append((draw, magnitude, score))

    out = []
    for i in range(len(loaded)):
        for j in range(i + 1, len(loaded)):
            (left, mag_l, cen_l), (right, mag_r, cen_r) = loaded[i], loaded[j]
            if mag_l.size != mag_r.size or left["on_condition"] != right["on_condition"]:
                # A pair drawn under two conditions measures the conditions, not
                # the arm's reproducibility.
                continue
            k = min(CENSUS_RETRIEVAL_K, mag_l.size)
            out.append(
                {
                    "left": left["directory"],
                    "right": right["directory"],
                    "folds": sorted({left["fold"], right["fold"]}),
                    "n_heads": int(mag_l.size),
                    "chance": float(k * k / mag_l.size),
                    "causal_intersection": top_set_jaccard(mag_l, mag_r, count=k)["intersection"],
                    "census_intersection": top_set_jaccard(cen_l, cen_r, count=k)["intersection"],
                }
            )
    return out


# ------------------------------------------------- stratified retrieval sweep


def _retrieval(matched: np.ndarray, records: list[dict[str, Any]], per_cluster: np.ndarray,
               weights: np.ndarray, clusters: np.ndarray, order: list[int],
               keep_sequences: np.ndarray, *, seed: int) -> dict[str, Any] | None:
    """hit@20 recomputed on one subset of sequences, through the module."""

    member = np.isin(clusters, keep_sequences)
    if member.sum() < 8:
        return None
    subset_weights = np.where(member, weights, 0.0)
    try:
        boot = cluster_bootstrap(
            per_cluster, subset_weights, replicates=STRATUM_REPLICATES, seed=seed
        )
    except ValueError:
        return None
    stratum_records = [dict(record) for record in records]
    for position, index in enumerate(order):
        stratum_records[index]["delta_m_gap"] = float(boot["mean"][position])
    agreement = census_causal_agreement(matched[keep_sequences], stratum_records)
    return {
        "hit_at_k": agreement["retrieval"]["hit_at_k"],
        "chance": agreement["retrieval"]["chance"],
        "spearman": agreement["spearman_census_vs_causal_magnitude"],
    }


def stratified_sweep(draw: dict[str, Any], table: dict[str, Any], *, seed: int) -> dict[str, Any]:
    """Retrieval in the favourable and unfavourable half of every factor."""

    matched = np.load(draw["paths"]["census"])["paa_specific_matched_per_sequence"]
    causal = np.load(draw["paths"]["causal"])
    per_cluster = causal["per_cluster_delta_m_gap"]
    weights = causal["cluster_weights"].astype(np.float64)
    clusters = causal["clusters"].astype(np.int64)
    records = json.loads(Path(draw["paths"]["causal_json"]).read_text(encoding="utf-8"))["heads"]
    order = _head_order(records)
    # per_cluster columns follow causal_matrices['heads']; align them to the
    # (layer, head_index) order the records are read in.
    heads = causal["heads"]
    lookup = {(int(layer), int(head)): column for column, (layer, head) in enumerate(heads)}
    columns = [lookup[(records[i]["layer"], records[i]["head_index"])] for i in order]
    per_cluster = per_cluster[:, columns]

    counts = table["instances_per_sequence"]
    eligible = np.flatnonzero(counts > 0)
    rng = np.random.default_rng(seed)

    factors: dict[str, Any] = {}
    for name, predicted in FACTOR_SIGNS.items():
        values = table["per_sequence"][name][eligible]
        finite = np.isfinite(values)
        usable = eligible[finite]
        values = values[finite]
        if usable.size < 16 or np.unique(values).size < 2:
            factors[name] = {"skipped": "degenerate factor on this draw"}
            continue
        threshold = float(np.median(values))
        high = usable[values > threshold]
        low = usable[values <= threshold]
        favourable, unfavourable = (high, low) if predicted >= 0 else (low, high)
        entry = {
            "predicted_sign": predicted,
            "threshold": threshold,
            "n_favourable": int(favourable.size),
            "n_unfavourable": int(unfavourable.size),
            "favourable": _retrieval(matched, records, per_cluster, weights, clusters,
                                     order, favourable, seed=seed + 11),
            "unfavourable": _retrieval(matched, records, per_cluster, weights, clusters,
                                       order, unfavourable, seed=seed + 12),
        }
        factors[name] = entry

    sizes = sorted({int(entry["n_favourable"]) for entry in factors.values()
                    if "n_favourable" in entry})
    null: dict[str, Any] = {}
    for size in sizes:
        hits = []
        for replicate in range(NULL_SUBSETS):
            subset = rng.choice(eligible, size=size, replace=False)
            value = _retrieval(matched, records, per_cluster, weights, clusters, order,
                               np.sort(subset), seed=seed + 1000 + replicate)
            if value is not None:
                hits.append(value["hit_at_k"])
        null[str(size)] = {
            "n": len(hits),
            "median": float(np.median(hits)) if hits else float("nan"),
            "q_low": float(np.quantile(hits, 0.05)) if hits else float("nan"),
            "q_high": float(np.quantile(hits, 0.95)) if hits else float("nan"),
        }
    return {"factors": factors, "null_by_size": null}


# ---------------------------------------------------------------------- main


def _process(payload: tuple[dict[str, Any], int, bool]) -> dict[str, Any]:
    draw, seed, sweep = payload
    table = factor_table(draw)
    out: dict[str, Any] = {
        "arm": draw["arm"],
        "directory": draw["directory"],
        "fold": draw["fold"],
        "on_condition": draw["on_condition"],
        "n_instances": table["n_instances"],
        "cascade": table["cascade"],
        "factor_summary": factor_summary(draw, table, seed=seed),
        "heads": head_diagnostics(draw, seed=seed),
        "retrieval": retrieval_diagnostics(draw, seed=seed),
    }
    if sweep:
        out["stratified"] = stratified_sweep(draw, table, seed=seed)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="results root to scan")
    parser.add_argument("--out", type=Path, required=True, help="directory for the JSON record")
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--any-condition", action="store_true",
                        help="also read draws off the declared condition, flagged as such")
    parser.add_argument("--arms", default=None,
                        help="comma-separated arm names to keep; the arms with no "
                             "on-condition draw are read this way rather than by "
                             "re-running the whole tree off condition")
    parser.add_argument("--no-sweep", action="store_true",
                        help="head diagnostics and factor means only")
    args = parser.parse_args()

    draws = discover(args.root, any_condition=args.any_condition)
    if args.arms:
        wanted = {name.strip() for name in args.arms.split(",") if name.strip()}
        draws = [draw for draw in draws if draw["arm"] in wanted]
    if not draws:
        raise SystemExit(f"no readable arm-draw under {args.root}")
    per_arm = split_draws(draws)
    ordered = [draw for arm in sorted(per_arm) for draw in per_arm[arm]]

    payloads = [
        (draw, args.seed + 97 * index, not args.no_sweep)
        for index, draw in enumerate(ordered)
    ]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(_process, payloads))
    else:
        results = [_process(payload) for payload in payloads]

    record = {
        "root": str(args.root),
        "seed": args.seed,
        "declared_condition": _declared_condition(),
        "factor_signs": FACTOR_SIGNS,
        "draws": results,
        "reliability": {
            arm: reliability(arm_draws) for arm, arm_draws in sorted(per_arm.items())
            if len(arm_draws) > 1
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / "paa_failure_audit.json"
    destination.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"wrote {destination}")

    print(f"\n{'arm':16}{'K':>3}{'grid':>6}{'med SNR':>10}{'f(SNR>2)':>10}"
          f"{'f(CI!=0)':>10}{'top20share/unif':>17}")
    for arm in sorted(per_arm):
        rows = [entry for entry in results if entry["arm"] == arm]
        heads = [row["heads"] for row in rows]
        print(
            f"{arm:16}{len(rows):>3}{heads[0]['n_heads']:>6}"
            f"{np.median([h['median_snr'] for h in heads]):>10.2f}"
            f"{np.median([h['fraction_snr_gt_2'] for h in heads]):>10.3f}"
            f"{np.median([h['fraction_ci_excludes_zero'] for h in heads]):>10.3f}"
            f"{np.median([h['top20_share_over_uniform'] for h in heads]):>17.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
