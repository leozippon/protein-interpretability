#!/usr/bin/env python3
"""D3.d: does a protein decoder know a coupling its corpus does not already carry?

F10 (EXP-R2-143) bounded the *first-order* question: no protein decoder's
zero-shot fitness exceeds a position-independent profile lookup over its own
pretraining corpus. Both channels in that comparison are additive over
substitutions -- ``profiles.lookup_score`` is an exact sum of per-column
log-odds and ``fitness.Assay.blosum`` an exact sum over the mutation string --
so **both predict identically zero epistasis for every multi-substitution
variant**. The second-order question is therefore not bounded by the first, and
this stage asks it.

The decision rules are frozen in ``docs/EXPERIMENT_LOG.md`` under EXP-R2-177,
written before any model was loaded, and this entry point implements only the
two stages authorised there.

``cohort``          Stage 0, CPU only. The measured referent (specific
                    epistasis, global-epistasis-corrected out of fold, with its
                    split-half reliability), the corpus coupling channel with
                    its column-permutation null, that channel's **own** positive
                    control, and the family-disjoint split. Can refuse the
                    campaign on its own evidence.
``attainability``   Stage 1, one GPU. A1: can the estimator detect a coupling
                    planted for it? Standing rule 2 in the only form available
                    when the biological referent has no text analogue -- the
                    referent is synthetic, so the text control *can* be run.

**A1 is not a formality.** A null on arms that may simply lack the signal is
publishable only if the positive control proves detection was possible. If A1
fails, nothing downstream is interpreted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer import epistasis as E  # noqa: E402
from src.transfer.arms import (  # noqa: E402
    DEFAULT_CORPUS_DRAW_SEED,
    PANEL,
    REPO,
    Cohort,
    load_arm,
    protein_cohort,
    text_cohort,
    tokenize_batch,
)
from src.transfer.families import (  # noqa: E402
    boundary_leakage,
    family_assignment,
    family_disjoint_split,
    load_cath_superfamilies,
    load_pfam_families,
)
from src.transfer.fitness import available_assays  # noqa: E402
from src.transfer.homology import ALIGNMENT_FIELDS, parse_hits  # noqa: E402
from src.transfer.io import write_json  # noqa: E402
from src.transfer.profiles import AA20, build_profile  # noqa: E402
from src.transfer.statistics import bootstrap_unit_floor  # noqa: E402

SCHEMA_VERSION = E.SCHEMA_VERSION
DEFAULT_OUT = REPO / "results/transfer/epistasis"
RETRIEVAL_BOUND = REPO / "results/transfer/retrieval_bound"

#: The arms A1 is run on. gpt2-large is required by the frozen rule; the two
#: protein arms differ in tokenisation, which is the axis EXP-R2-176 measured as
#: load-bearing for anything position-level.
ATTAINABILITY_ARMS: tuple[str, ...] = ("gpt2-large", "protgpt2", "progen2-medium")


def _json_safe(value: Any) -> Any:
    """Non-finite floats to ``None``, recursively, before the artefact is written.

    A reliability is genuinely undefined on an assay carrying too few pairs to
    split in half, and NaN is the honest value for it -- but ``json`` refuses to
    serialise one, so an undefined statistic anywhere in the payload used to
    destroy the whole run's artefact after its work was done. ``None`` is the
    serialisable spelling of "not estimated here", and it is not the same as
    zero, which is what a clamp would have written.
    """

    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _resolved_paths() -> dict[str, Any]:
    """Print before every campaign; L18 cost a nine-stage run its whole text side."""

    record = {
        "repo": str(REPO),
        "proteingym": str(REPO / "data/proteingym/DMS_ProteinGym_substitutions"),
        "retrieval_bound": str(RETRIEVAL_BOUND),
        "panel_size": len(PANEL),
        "panel_arms": sorted(PANEL),
        "env": {
            key: os.environ.get(key)
            for key in (
                "TRANSFER_MODEL_BASE_DIR",
                "TRANSFER_TEXT_MODEL_BASE_DIR",
                "TRANSFER_PROTEINGYM_DIR",
                "TRANSFER_PFAM_RESIDUE_TSV",
                "TRANSFER_CATH_SUPERFAMILY_TSV",
            )
        },
    }
    print("[paths] " + json.dumps(record["env"]))
    print(f"[paths] PANEL carries {record['panel_size']} arms")
    return record


# ------------------------------------------------------------------- stage 0


def stage_cohort(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": "cohort",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "paths": _resolved_paths(),
    }

    wildtypes = json.loads((RETRIEVAL_BOUND / "wildtypes.json").read_text())
    assay_to_query: dict[str, str] = wildtypes["assay_to_wildtype"]
    query_meta: dict[str, Any] = wildtypes["wildtypes"]

    print("[cohort] building measured pair tables")
    tables: dict[str, E.PairTable] = {}
    refused: dict[str, str] = {}
    for name in available_assays():
        try:
            table = E.assay_pairs(
                name,
                seed=args.seed,
                min_doubles=args.min_doubles,
                mode=args.global_epistasis,
                max_doubles_per_pair=args.max_doubles_per_pair,
            )
        except (RuntimeError, ValueError) as error:
            refused[name] = f"{type(error).__name__}: {error}"
            continue
        if len(table.positions) < args.min_pairs:
            refused[name] = f"only {len(table.positions)} pairs at >= {args.min_doubles} doubles"
            continue
        tables[name] = table
        print(
            f"  {name:44s} pairs {len(table.positions):5d}  "
            f"doubles {int(table.n_doubles.sum()):7d}  "
            f"reliability {table.reliability:+.3f}  ceiling {table.attenuation_ceiling:.3f}"
        )
    if not tables:
        raise RuntimeError("no assay survived the pair-coverage floor")

    # One unit per identity cluster: the assay with the most pairs, so a protein
    # measured twice is one unit and not two (F10's unit, EXP-R2-143).
    by_cluster: dict[int, str] = {}
    for name, table in tables.items():
        query = assay_to_query.get(name)
        if query is None:
            refused[name] = "no retrieval-bound query id"
            continue
        cluster = int(query_meta[query]["cluster"])
        best = by_cluster.get(cluster)
        if best is None or len(tables[best].positions) < len(table.positions):
            by_cluster[cluster] = name
    print(f"[cohort] {len(tables)} assays over {len(by_cluster)} identity clusters")

    print("[couple] reading corpus alignments once")
    wanted = {assay_to_query[name] for name in by_cluster.values()}
    hits_by_query: dict[str, list] = defaultdict(list)
    for hit in parse_hits(RETRIEVAL_BOUND / "corpus_hits.tsv", fields=ALIGNMENT_FIELDS):
        if hit.query in wanted:
            hits_by_query[hit.query].append(hit)
    stored = np.load(RETRIEVAL_BOUND / "profiles.npz")

    units: list[dict[str, Any]] = []
    for cluster in sorted(by_cluster):
        name = by_cluster[cluster]
        table = tables[name]
        query = assay_to_query[name]
        record: dict[str, Any] = {
            "cluster": cluster,
            "query_id": query,
            **table.record(),
            "verbatim_in_corpus": bool(query_meta[query]["verbatim_in_corpus"]),
        }
        hits = hits_by_query.get(query, [])
        record["n_corpus_hits"] = len(hits)

        if len(hits) < args.min_hits:
            record["coupling"] = {
                "status": "refused",
                "reason": (
                    f"{len(hits)} corpus hits below the declared floor of {args.min_hits}; "
                    "a pairwise coupling estimated here would be reading its own bias"
                ),
            }
            units.append(record)
            continue

        rows, weights, neff = E.alignment_rows(
            table.wildtype, query, hits, max_sequences=args.max_profile_sequences
        )
        profile = build_profile(
            table.wildtype, query, hits, max_sequences=args.max_profile_sequences
        )
        E.verify_rows_against_profile(rows, weights, profile)
        if query in stored:
            drift = float(np.abs(profile.frequencies - stored[query]).max())
            record["profile_reproduces_f10"] = drift
            if drift > 1e-5:
                raise RuntimeError(
                    f"{query}: rebuilt profile differs from the frozen F10 artefact by "
                    f"{drift:.3e}; the corpus or the code under this channel has moved"
                )

        columns = np.unique(table.positions.reshape(-1) - 1)
        coupling, depth = E.coupling_apc(rows, weights, columns, seed=args.seed + cluster)
        index = {int(column): position for position, column in enumerate(columns)}
        pair_rows = np.array([index[i - 1] for i in table.positions[:, 0]])
        pair_cols = np.array([index[j - 1] for j in table.positions[:, 1]])
        observed = coupling[pair_rows, pair_cols]

        null_draws = E.coupling_column_null(
            rows, weights, columns, draws=args.coupling_null_draws, seed=args.seed + 7919 * cluster
        )
        null_pairs = null_draws[:, pair_rows, pair_cols]

        from scipy import stats

        control = float(stats.spearmanr(observed, table.epistasis).statistic)
        null_control = np.array(
            [stats.spearmanr(draw, table.epistasis).statistic for draw in null_pairs]
        )
        finite = null_control[np.isfinite(null_control)]
        q99 = float(np.quantile(finite, 0.99)) if len(finite) else float("nan")
        record["coupling"] = {
            "status": "measured",
            "neff": neff,
            "log10_neff": float(np.log10(neff)) if neff >= 1.0 else 0.0,
            "n_alignment_rows": int(rows.shape[0]),
            "n_columns": int(len(columns)),
            "mean_effective_depth": float(np.mean(depth[pair_rows, pair_cols])),
            "spearman_against_measured": control,
            "column_null_q99": q99,
            "column_null_mean": float(np.mean(finite)) if len(finite) else float("nan"),
            "positive_control": (
                "PASS" if np.isfinite(q99) and control > q99 else "FAIL"
            ),
            "note": (
                "PASS means this protein's corpus coupling predicts its measured "
                "specific epistasis above a column permutation that preserves every "
                "column's marginal and depth. Where it FAILS, A3 is not interpretable "
                "on this protein and must be reported as such rather than read as the "
                "model winning (F10's positive_control gate, one order up)"
            ),
        }
        units.append(record)

    payload["assays_refused"] = refused
    payload["units"] = units

    measured = [u for u in units if u["coupling"]["status"] == "measured"]
    passing = [u for u in measured if u["coupling"]["positive_control"] == "PASS"]
    payload["coupling_channel"] = {
        "units_total": len(units),
        "units_with_coupling": len(measured),
        "units_passing_positive_control": len(passing),
        "unit_floor": bootstrap_unit_floor(len(passing)),
        "verdict": "USABLE" if len(passing) >= 8 else "NOT_USABLE",
        "note": (
            "USABLE means enough proteins carry an interpretable retrieval channel "
            "for A3's cluster bootstrap. NOT_USABLE does not stop A1 or A2; it means "
            "the retrieval comparison is reported per protein and not pooled"
        ),
    }
    payload["family_split"] = _family_split(units, tables, assay_to_query, args)
    return payload


def _family_split(
    units: list[dict[str, Any]],
    tables: dict[str, E.PairTable],
    assay_to_query: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """The held-out-family protocol, through ``families`` and not around it."""

    import gzip
    import re

    entry_of: dict[str, str] = {}
    for unit in units:
        name = unit["assay"]
        entry_of[name] = "_".join(name.split("_")[:2])
    wanted = set(entry_of.values())
    accession: dict[str, str] = {}
    fasta = REPO / "data/swissprot/uniprot_sprot.fasta.gz"
    with gzip.open(fasta, "rt") as handle:
        for line in handle:
            if not line.startswith(">"):
                continue
            match = re.match(r">(?:sp|tr)\|([^|]+)\|(\S+)", line)
            if match and match.group(2) in wanted:
                accession[match.group(2)] = match.group(1)

    unit_ids: list[str] = []
    sequences: dict[str, str] = {}
    for unit in units:
        entry = entry_of[unit["assay"]]
        if entry not in accession:
            continue
        code = accession[entry]
        if code in sequences:
            continue
        unit_ids.append(code)
        sequences[code] = tables[unit["assay"]].wildtype

    out: dict[str, Any] = {
        "n_units": len(units),
        "n_entry_names_resolved": len(accession),
        "n_accessions": len(unit_ids),
        "sources": {},
    }
    for source, loader in (
        ("pfam", load_pfam_families),
        ("cath_superfamily", load_cath_superfamilies),
    ):
        table = loader(accessions=set(unit_ids))
        labelled = [unit for unit in unit_ids if table.get(unit)]
        entry: dict[str, Any] = {"n_labelled": len(labelled)}
        if len(labelled) >= 8:
            assignment = family_assignment(
                labelled, table, source=source, unlabelled="drop"
            )
            entry["n_groups"] = assignment.n_groups
            try:
                split = family_disjoint_split(assignment, seed=args.seed)
                entry["train_units"] = int(split.train.sum())
                entry["test_units"] = int(split.test.sum())
                entry["leakage"] = boundary_leakage(
                    split, sequences, seed=args.seed
                )
                entry["status"] = "split"
            except (RuntimeError, ValueError) as error:
                entry["status"] = f"refused: {type(error).__name__}: {error}"
        else:
            entry["status"] = "too few labelled units for a split"
        out["sources"][source] = entry
    return out


# ------------------------------------------------------------------- stage 1


class _Scorer:
    """Summed log-likelihood under one arm, in the panel's own rendering.

    Rendering is the panel's decision and not this stage's (Appendix B rule 12;
    ProtGPT2 moves 1.42 nats/token between raw sequence and its declared FASTA
    wrapping), so every string goes through ``Cohort.input_strings``.
    """

    def __init__(self, arm_name: str, args: argparse.Namespace) -> None:
        import torch

        self.torch = torch
        self.name = arm_name
        self.kind = "text" if PANEL[arm_name].modality == "text" else "protein"
        self.batch_size = args.batch_size
        self.arm = load_arm(arm_name, device=args.device, dtype=args.dtype)
        config = self.arm.model.config
        self.context = int(
            getattr(config, "n_positions", None)
            or getattr(config, "max_position_embeddings")
        )

    def _render(self, sequences: list[str]) -> list[str]:
        cohort = Cohort(
            name="coupling_probes",
            kind=self.kind,
            records=list(sequences),
            min_symbols=min(len(s) for s in sequences),
            max_symbols=max(len(s) for s in sequences),
            metadata={},
        )
        return cohort.input_strings(self.arm)

    def repeat_survives_tokenisation(self, probe: Any) -> bool | None:
        """Do the two copies of the planted motif carry identical tokens?

        Returns ``None`` where the tokenizer cannot report offsets, so an
        unmeasured diagnostic is never recorded as a passing one.
        """

        first, second, length = probe.motif_span
        text = self._render([probe.sequence])[0]
        body = probe.sequence
        start = text.find(body[:32])
        if start < 0:
            return None
        try:
            encoded = self.arm.tokenizer(text, return_offsets_mapping=True)
        except (TypeError, NotImplementedError, ValueError):
            return None
        offsets = encoded.get("offset_mapping")
        if not offsets:
            return None
        ids = encoded["input_ids"]

        def span_tokens(symbol_start: int) -> list[int]:
            prefix = probe.separator.join(probe.symbols[:symbol_start])
            lead = len(prefix) + (len(probe.separator) if symbol_start else 0)
            inner = probe.separator.join(probe.symbols[symbol_start : symbol_start + length])
            low, high = start + lead, start + lead + len(inner)
            return [
                token
                for token, (a, b) in zip(ids, offsets)
                if a >= low and b <= high and b > a
            ]

        return span_tokens(first) == span_tokens(second)

    def log_likelihood(self, sequences: list[str]) -> np.ndarray:
        torch = self.torch
        texts = self._render(sequences)
        totals = np.empty(len(texts), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                chunk = texts[start : start + self.batch_size]
                ids, mask = tokenize_batch(self.arm, chunk, self.context)
                ids = ids.to(self.arm.device)
                mask = mask.to(self.arm.device)
                logits = self.arm.model(input_ids=ids, attention_mask=mask).logits
                logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
                token = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                keep = (mask[:, 1:] * mask[:, :-1]).bool()
                totals[start : start + len(chunk)] = (
                    (token * keep).sum(1).double().cpu().numpy()
                )
        return totals

    def release(self) -> None:
        del self.arm
        self.torch.cuda.empty_cache()


def _records_for(arm_name: str, args: argparse.Namespace) -> tuple[list[list[str]], list[str], str]:
    """Per-modality symbol records, the substitute alphabet, and their separator."""

    if PANEL[arm_name].modality == "text":
        cohort = text_cohort(
            args.probes * 3, skip=args.cohort_skip, seed=args.cohort_draw_seed
        )
        records = []
        vocabulary: set[str] = set()
        for document in cohort.records:
            words = [w for w in document.split() if w.isalpha() and w.islower()]
            if len(words) >= args.symbols:
                records.append(words[: args.symbols])
                vocabulary.update(words[: args.symbols])
        alphabet = sorted(vocabulary)[: args.text_alphabet]
        return records[: args.probes], alphabet, " "
    cohort = protein_cohort(
        args.probes * 3,
        args.symbols,
        args.protein_max_len,
        skip=args.cohort_skip,
        seed=args.cohort_draw_seed,
    )
    records = [list(s[: args.symbols]) for s in cohort.records if len(s) >= args.symbols]
    return records[: args.probes], list(AA20), ""


def stage_attainability(args: argparse.Namespace) -> dict[str, Any]:
    from scipy import stats

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": "attainability",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "paths": _resolved_paths(),
        "rule": (
            "A1 passes when the 95% bootstrap interval of the pooled Spearman over "
            "probes lies wholly above the 99th percentile of the within-probe "
            "permutation null, on gpt2-large and on at least one protein arm "
            "(frozen in EXP-R2-177 before any model was loaded)"
        ),
        "arms": {},
    }

    for arm_name in args.arms:
        print(f"[a1] {arm_name}")
        records, alphabet, separator = _records_for(arm_name, args)
        probes = E.planted_coupling_probes(
            records,
            n_pairs=args.pairs,
            separation=args.separation,
            alphabet=alphabet,
            separator=separator,
            seed=args.seed,
        )
        print(f"  {len(probes)} probes, {args.pairs} pairs each, separation {args.separation}")
        scorer = _Scorer(arm_name, args)
        try:
            survived = [scorer.repeat_survives_tokenisation(probe) for probe in probes]
            flat: list[str] = []
            spans: list[tuple[int, list[str]]] = []
            for probe in probes:
                variants = probe.variants()
                spans.append((len(flat), [role for role, _ in variants]))
                flat.extend(text for _, text in variants)
            values = scorer.log_likelihood(flat)
        finally:
            scorer.release()

        referent: list[float] = []
        probe_id: list[int] = []
        by_scheme: dict[str, list[float]] = {"shared": [], "distinct": []}
        top1 = {"shared": 0, "distinct": 0}
        for index, (probe, (start, roles)) in enumerate(zip(probes, spans)):
            scores = {role: float(values[start + offset]) for offset, role in enumerate(roles)}
            for scheme in by_scheme:
                m = E.probe_epistasis(scores, probe, scheme=scheme)
                by_scheme[scheme].extend(m.tolist())
                if int(np.argmax(m)) == probe.planted:
                    top1[scheme] += 1
            referent.extend([1.0 if k == probe.planted else 0.0 for k in range(len(probe.pairs))])
            probe_id.extend([index] * len(probe.pairs))

        referent_array = np.asarray(referent)
        probe_array = np.asarray(probe_id)
        unique = np.unique(probe_array)
        entry: dict[str, Any] = {
            "modality": PANEL[arm_name].modality,
            "n_probes": len(probes),
            "n_pairs_per_probe": args.pairs,
            "separation_symbols": args.separation,
            "unit_floor": bootstrap_unit_floor(len(probes)),
            "planted_repeat_survives_tokenisation": (
                None
                if all(value is None for value in survived)
                else sum(1 for value in survived if value) / len(survived)
            ),
            "schemes": {},
        }
        for scheme, values_list in by_scheme.items():
            statistic_array = np.asarray(values_list)
            null = E.stratified_permutation_null(
                statistic_array,
                referent_array,
                probe_array,
                draws=args.null_draws,
                seed=args.seed + 11,
            )
            rng = np.random.default_rng(args.seed + 13)
            draws = np.empty(args.bootstrap, dtype=np.float64)
            for draw in range(args.bootstrap):
                picked = rng.integers(0, len(unique), size=len(unique))
                mask = np.concatenate([np.flatnonzero(probe_array == unique[p]) for p in picked])
                draws[draw] = stats.spearmanr(
                    statistic_array[mask], referent_array[mask]
                ).statistic
            finite = draws[np.isfinite(draws)]
            low, high = np.nanpercentile(finite, [2.5, 97.5])
            entry["schemes"][scheme] = {
                "spearman": null["observed"],
                "interval": [float(low), float(high)],
                "null": null,
                "top1_rate": top1[scheme] / len(probes),
                "top1_chance": 1.0 / args.pairs,
                "planted_epistasis_mean": float(statistic_array[referent_array > 0].mean()),
                "control_epistasis_mean": float(statistic_array[referent_array == 0].mean()),
                "passes": bool(low > null["null_q99"]),
            }
            read = entry["schemes"][scheme]
            print(
                f"  [{scheme:8s}] spearman {read['spearman']:+.4f} "
                f"CI [{read['interval'][0]:+.4f}, {read['interval'][1]:+.4f}] vs null q99 "
                f"{null['null_q99']:+.4f}  top1 {read['top1_rate']:.3f} "
                f"(chance {read['top1_chance']:.3f})  planted {read['planted_epistasis_mean']:+.4f} "
                f"control {read['control_epistasis_mean']:+.4f}"
            )
        entry["gated_scheme"] = "distinct"
        entry["verdict"] = "PASS" if entry["schemes"]["distinct"]["passes"] else "FAIL"
        payload["arms"][arm_name] = entry
        print(f"  {arm_name}: {entry['verdict']}")

    text_pass = any(
        entry["verdict"] == "PASS"
        for name, entry in payload["arms"].items()
        if entry["modality"] == "text"
    )
    protein_pass = any(
        entry["verdict"] == "PASS"
        for name, entry in payload["arms"].items()
        if entry["modality"] != "text"
    )
    payload["verdict"] = "PASS" if (text_pass and protein_pass) else "FAIL"
    payload["consequence"] = (
        "Stage 2 is authorised to be proposed."
        if payload["verdict"] == "PASS"
        else "The estimator cannot detect a coupling planted for it. Stage 2 is not "
        "run and no downstream number is interpreted (EXP-R2-177)."
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("cohort", "attainability"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument(
        "--cohort-draw-seed", type=int, default=DEFAULT_CORPUS_DRAW_SEED
    )
    parser.add_argument("--cohort-skip", type=int, default=0)
    # stage 0
    parser.add_argument("--min-doubles", type=int, default=E.MIN_DOUBLES_PER_PAIR)
    parser.add_argument("--min-pairs", type=int, default=2)
    parser.add_argument("--min-hits", type=int, default=500)
    parser.add_argument("--max-doubles-per-pair", type=int, default=None)
    parser.add_argument("--max-profile-sequences", type=int, default=5000)
    parser.add_argument("--coupling-null-draws", type=int, default=64)
    parser.add_argument(
        "--global-epistasis", default="isotonic", choices=E.GLOBAL_EPISTASIS_MODES
    )
    # stage 1
    parser.add_argument("--arms", nargs="+", default=list(ATTAINABILITY_ARMS))
    parser.add_argument("--probes", type=int, default=128)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--separation", type=int, default=12)
    parser.add_argument("--symbols", type=int, default=160)
    parser.add_argument("--protein-max-len", type=int, default=246)
    parser.add_argument("--text-alphabet", type=int, default=512)
    parser.add_argument("--null-draws", type=int, default=2000)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    payload = {"cohort": stage_cohort, "attainability": stage_attainability}[args.stage](args)
    write_json(args.out / f"{args.stage}.json", _json_safe(payload))
    print(f"[done] {args.out / f'{args.stage}.json'}")
    if "verdict" in payload:
        print(f"[verdict] {payload['verdict']}")


if __name__ == "__main__":
    main()
