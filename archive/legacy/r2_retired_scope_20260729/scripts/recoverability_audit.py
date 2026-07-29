#!/usr/bin/env python3
"""Engine for the representation-recoverability audit (R2-RECOV-AUDIT-v1).

Implements the shared machinery for the pre-registered protocol in
``r2_interpretability_transfer/preregistration/PROTOCOL.md``:

  * representation extraction  : R_raw (residual-stream ceiling),
                                 R_code (CLT sparse-code floor),
                                 R_recon (CLT reconstruction)
  * baselines                  : B_ngram (composition null), R_rand (matched-dim
                                 random projection of R_raw), ESM2 (rich ref.)
  * linear probes              : grouped (family-disjoint) cross-validation,
                                 chance-corrected *skill*, bootstrap CIs
  * ceiling / floor / gap      : C, F, gap = C-F, recovery rho = F/C, ref. phi
  * decision logic             : the per-model verdict and the single GO/NO-GO
                                 for the high-cost dictionary retrain (PROTOCOL §6)

The numbered driver scripts (44/45/46/47/48) are thin CLIs over this module.
Validated label loaders and the ESM-2 embedding path are reused from scripts
29/33/34 rather than re-implemented.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PKG_ROOT = SCRIPT_DIR.parent            # r2_interpretability_transfer/
REPO = PKG_ROOT.parent                  # repo root
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Reuse validated cohort/label/probe/ESM-2 code from the existing scripts.
S34 = load_module(SCRIPT_DIR / "34_triplet_basis_probes.py", "ra_probes34")
S33 = load_module(SCRIPT_DIR / "33_swissprot_triplet_annotation.py", "ra_swissprot33")
U29 = load_module(SCRIPT_DIR / "29_universal_primitive_annotation.py", "ra_annot29")

clean_sequence = U29.clean_sequence
EC_CLASS_NAMES = S34.EC_CLASS_NAMES

DEFAULT_MODEL_SPECS = [
    "protgpt2=/oss-pvc/zhk_zip/outputs/research2/clt_weights/protgpt2_v2/step_200000",
    "zymctrl=/oss-pvc/zhk_zip/outputs/research2/clt_weights/zymctrl_v2/step_200000",
    "progen2-medium=" + str(PKG_ROOT / "results/final_checkpoints/r2_clt_progen2_medium_rerun_20260403/clt_weights/progen2-medium/step_100000"),
]
DEFAULT_ESM2 = "/Data/public/esm2_t36_3B_UR50D"
PRIMARY_TASKS = ["ec_topclass", "ec_topclass_stratified", "pfam_family",
                 "secondary_fraction", "residue_ss", "decoder_ec"]
# Reported for context but excluded from the rich/bottleneck gate.
#  - ec_topclass_stratified: family leaks (quantifies the confound only).
#  - secondary_fraction: the sparse-code FLOOR regression is numerically
#    unstable (R^2 explodes negative under StandardScaler-on-sparse-codes even
#    with PCA), so its rho is not a trustworthy dictionary-loss signal. The
#    residue-level SS task (residue_ss) carries the structural claim. This is the
#    contingency pre-specified in NEXT_STEPS_v2 ("if it remains negative on the
#    full run, treat secondary_fraction as report-only").
REPORT_ONLY_TASKS = {"ec_topclass_stratified", "secondary_fraction"}

EC_CLASS_PROMPTS = {
    "lysozyme": "3.2.1.17",
    "trypsin": "3.4.21.4",
    "ADH": "1.1.1.1",
    "catalase": "1.11.1.6",
    "DNA_polymerase": "2.7.7.7",
    "lipase": "3.1.1.3",
    "kinase": "2.7.11.1",
    "carbonic_anh": "4.2.1.1",
}


# --------------------------------------------------------------------------
# Cohort + labels
# --------------------------------------------------------------------------

def build_cohort(args) -> tuple[list[dict], dict]:
    """Frozen Swiss-Prot cohort for T1-T4 (reuses script 34's loaders)."""
    records, meta = S34.build_records(
        swissprot_cache=args.swissprot_cache,
        pfam_residue=args.pfam_residue,
        ec_fasta=args.ec_fasta,
        goa_gaf=args.goa_gaf,
        go_obo=args.go_obo,
        min_len=args.min_len,
        max_len=args.max_len,
        pfam_classes=args.pfam_classes,
        pfam_per_class=args.pfam_per_class,
        ec_per_class=args.ec_per_class,
        ss_n=args.ss_n,
        seed=args.seed,
    )
    return records, meta


def residue_ss_labels(records: list[dict], swissprot_cache: Path, min_cov: int = 20) -> dict[str, np.ndarray]:
    """Per-residue 3-state SS (H/E/C) from Swiss-Prot features, by accession.

    helix -> H, strand -> E, turn/other -> C. Only positions with an explicit
    secondary-structure annotation are labelled; a protein is kept if it has
    >= min_cov annotated residues.
    """
    anns = {a.accession: a for a in S33.load_swissprot(swissprot_cache)}
    out: dict[str, np.ndarray] = {}
    for rec in records:
        ann = anns.get(rec["accession"])
        if ann is None:
            continue
        seq = clean_sequence(ann.sequence)
        n = len(seq)
        if n == 0:
            continue
        lab = np.full(n, "", dtype=object)
        for start, end, feat_type, _desc, category in ann.features:
            if category != "secondary_structure":
                continue
            code = {"helix": "H", "strand": "E", "turn": "C"}.get(feat_type)
            if code is None:
                continue
            for p in range(max(1, int(start)) - 1, min(n, int(end))):
                lab[p] = code
        if int((lab != "").sum()) >= min_cov:
            out[rec["accession"]] = lab
    return out


def load_decoder_ec_cohort(path: Path, max_per_class: int, min_len: int, max_len: int) -> list[dict]:
    """Decoder-native EC cohort (T5): ZymCTRL-generated sequences + their EC tag.

    Accepts a JSON produced by the steering benchmark / steered_generation. Looks
    for per-class sequence lists; tolerant of a few common shapes. Returns [] if
    nothing usable is found (T5 is then skipped and logged).
    """
    if not path or not Path(path).exists():
        return []
    data = json.loads(Path(path).read_text())
    per_class = data.get("per_class") or data.get("classes") or {}
    out: list[dict] = []
    for cls, payload in per_class.items():
        seqs: list = []
        if isinstance(payload, dict):
            for key in (
                "unsteered_sequences",
                "steered_sequences",
                "sequences",
                "unsteered",
                "steered",
                "generated",
                "example_unsteered",
                "example_steered",
            ):
                if isinstance(payload.get(key), list):
                    seqs.extend(payload[key])
        elif isinstance(payload, list):
            seqs = payload
        if not seqs:
            continue
        kept = 0
        for i, s in enumerate(seqs):
            seq = clean_sequence(s if isinstance(s, str) else s.get("sequence", ""))
            if min_len <= len(seq) <= max_len:
                label = str(cls)
                out.append({"id": f"{label}_{i}", "accession": f"{label}_{i}",
                            "sequence": seq, "decoder_ec": label})
                kept += 1
            if kept >= max_per_class:
                break
    return out


# --------------------------------------------------------------------------
# Representation extraction
# --------------------------------------------------------------------------

def _token_mask(pm, input_ids):
    """Boolean mask of non-special token positions (1D, length = seq)."""
    ids = input_ids.view(-1).tolist()
    special = set(getattr(pm.tokenizer, "all_special_ids", []) or [])
    return np.array([i not in special for i in ids], dtype=bool)


def extract_protein_matrices(pm, clt, sequences, layers, device, recon=True):
    """Mean-pooled (protein-level) R_raw / R_code / R_recon per layer.

    Returns dict: {"R_raw": {l: [n,d_model]}, "R_code": {l: [n,d_clt]},
                   "R_recon": {l: [n,d_model]}}  (R_recon omitted if recon=False)
    """
    import torch
    reps = {"R_raw": {l: [] for l in layers}, "R_code": {l: [] for l in layers}}
    if recon:
        reps["R_recon"] = {l: [] for l in layers}
    for k, rec in enumerate(sequences):
        ids = pm.tokenize(rec["sequence"])
        cache = pm.get_activations(ids)
        resid = [x.float() for x in cache.resid_pre]
        feats = clt.encode(resid)
        recon_l = clt.decode(feats) if recon else None
        mask = _token_mask(pm, ids)
        mask_t = torch.from_numpy(mask).to(device)
        for l in layers:
            def pool(t):  # t: [1, seq, dim] -> [dim] mean over real tokens
                v = t[0][mask_t] if mask_t.shape[0] == t.shape[1] else t[0]
                return v.mean(dim=0).detach().float().cpu().numpy()
            reps["R_raw"][l].append(pool(resid[l]))
            reps["R_code"][l].append(pool(feats[l]))
            if recon:
                reps["R_recon"][l].append(pool(recon_l[l]))
        if (k + 1) % 25 == 0 or k == len(sequences) - 1:
            print(f"    {pm.model_name}: {k+1}/{len(sequences)} seqs", flush=True)
    for name in reps:
        for l in layers:
            reps[name][l] = np.stack(reps[name][l]).astype(np.float32)
    return reps


def extract_residue_matrices(pm, clt, sequences, ss_by_acc, layers, device):
    """Per-residue R_raw / R_code for the residue-SS task (T4).

    Returns dict {"R_raw": {l: [N_res, d_model]}, "R_code": {l: [N_res, d_clt]}},
    plus parallel arrays y (labels), groups (protein index) aligned across layers.
    """
    import torch
    rows = {"R_raw": {l: [] for l in layers}, "R_code": {l: [] for l in layers}}
    y, groups, baseline_rows = [], [], []
    for gi, rec in enumerate(sequences):
        lab = ss_by_acc.get(rec["accession"])
        if lab is None:
            continue
        seq = rec["sequence"]
        ids = pm.tokenize(seq)
        spans = U29.token_residue_spans(pm.tokenizer, ids, seq)
        cache = pm.get_activations(ids)
        resid = [x.float() for x in cache.resid_pre]
        feats = clt.encode(resid)
        # residue -> covering token index
        res_tok = np.full(len(seq), -1, dtype=int)
        for tok_idx, span in enumerate(spans):
            for p in span:
                if 0 <= p < len(seq) and res_tok[p] < 0:
                    res_tok[p] = tok_idx
        valid = [p for p in range(min(len(seq), len(lab))) if lab[p] != "" and res_tok[p] >= 0]
        if not valid:
            continue
        for l in layers:
            raw_l = resid[l][0].detach().float().cpu().numpy()
            code_l = feats[l][0].detach().float().cpu().numpy()
            rows["R_raw"][l].append(raw_l[res_tok[valid]])
            rows["R_code"][l].append(code_l[res_tok[valid]])
        y.extend(lab[p] for p in valid)
        groups.extend([gi] * len(valid))
        for p in valid:
            row = np.zeros(len(AA) + 1, dtype=np.float32)
            aa = seq[p]
            if aa in _AA_IDX:
                row[_AA_IDX[aa]] = 1.0
            row[-1] = p / max(len(seq) - 1, 1)
            baseline_rows.append(row)
    out = {name: {l: (np.concatenate(rows[name][l]).astype(np.float32) if rows[name][l]
                      else np.zeros((0, 1), np.float32)) for l in layers} for name in rows}
    baseline = np.stack(baseline_rows).astype(np.float32) if baseline_rows else np.zeros((0, len(AA) + 1), np.float32)
    return out, np.array(y, dtype=object), np.array(groups, dtype=int), baseline


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------

AA = "ACDEFGHIKLMNPQRSTVWY"
_AA_IDX = {a: i for i, a in enumerate(AA)}


def ngram_baseline(sequences) -> np.ndarray:
    """Composition null: AA + dipeptide + tripeptide composition + length."""
    rows = []
    for rec in sequences:
        s = rec["sequence"]
        comp = np.zeros(20, np.float32)
        di = np.zeros(400, np.float32)
        tri = np.zeros(8000, np.float32)
        for a in s:
            if a in _AA_IDX:
                comp[_AA_IDX[a]] += 1
        for a, b in zip(s, s[1:]):
            if a in _AA_IDX and b in _AA_IDX:
                di[_AA_IDX[a] * 20 + _AA_IDX[b]] += 1
        for a, b, c in zip(s, s[1:], s[2:]):
            if a in _AA_IDX and b in _AA_IDX and c in _AA_IDX:
                tri[(_AA_IDX[a] * 20 + _AA_IDX[b]) * 20 + _AA_IDX[c]] += 1
        n = max(len(s), 1)
        rows.append(np.concatenate([comp / n, di / max(n - 1, 1), tri / max(n - 2, 1), [np.log(n)]]))
    return np.stack(rows).astype(np.float32)


def random_projection(X_raw: np.ndarray, out_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out_dim = max(1, int(out_dim))
    P = rng.standard_normal((X_raw.shape[1], out_dim)).astype(np.float32) / np.sqrt(X_raw.shape[1])
    return X_raw @ P


def alive_columns(X_code: np.ndarray) -> np.ndarray:
    """Indices of CLT features that fire (non-zero) for at least one protein."""
    return np.where(np.abs(X_code).sum(axis=0) > 0)[0]


# --------------------------------------------------------------------------
# Probes, skill, bootstrap
# --------------------------------------------------------------------------

def _make_groups(records_or_groups, idx):
    """Group id per sample = dominant Pfam family (homology-disjoint CV).

    Records without a family get a unique singleton group so they remain
    freely splittable.
    """
    groups = []
    for k, i in enumerate(idx):
        rec = records_or_groups[i]
        fam = rec.get("dominant_pfam")
        groups.append(fam if fam else f"__solo_{i}")
    return np.array(groups, dtype=object)


def _probe_pipeline(estimator, n_features, min_train, pca_dim, seed):
    """StandardScaler -> (optional) PCA -> estimator.

    PCA reduces every representation to a common, well-conditioned dimension so
    ceiling (R_raw, ~d_model) and floor (R_code, ~d_clt) are compared at matched
    dimensionality (this both fixes the high-dim regression blow-up and makes the
    old `R_rand` dimensionality gate unnecessary). `n_components` is capped to the
    smallest train-fold size, so it is always valid for grouped CV (whose folds
    are uneven).
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    steps = [StandardScaler(with_mean=True)]
    if pca_dim:
        from sklearn.decomposition import PCA
        d = min(int(pca_dim), int(n_features), int(min_train) - 1)
        if d >= 2:
            steps.append(PCA(n_components=d, svd_solver="randomized", random_state=seed))
    steps.append(estimator)
    return make_pipeline(*steps)


def cv_predict_classification(X, y, groups, seed, C=1.0, allow_fallback=False, pca_dim=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
    from collections import Counter
    labels = sorted(set(y))
    yi = np.array([labels.index(v) for v in y])
    n_splits = max(2, min(5, min(Counter(yi).values())))
    n_groups = len(set(groups))
    if n_groups >= n_splits:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(cv.split(X, yi, groups))
    elif allow_fallback:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(cv.split(X, yi))
    else:
        raise RuntimeError(f"too few groups for grouped CV: n_groups={n_groups}, n_splits={n_splits}")
    min_train = min(len(tr) for tr, _ in splits)
    clf = _probe_pipeline(LogisticRegression(C=C, max_iter=3000, class_weight="balanced"),
                          X.shape[1], min_train, pca_dim, seed)
    pred = cross_val_predict(clf, X, yi, cv=splits, method="predict")
    return yi, pred, labels


def cv_predict_regression(X, Y, groups, seed, alpha=1.0, allow_fallback=False, pca_dim=None):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
    n_splits = max(2, min(5, len(Y)))
    n_groups = len(set(groups))
    if n_groups >= n_splits:
        cv = GroupKFold(n_splits=n_splits)
        splits = list(cv.split(X, Y, groups))
    elif allow_fallback:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(cv.split(X, Y))
    else:
        raise RuntimeError(f"too few groups for grouped CV: n_groups={n_groups}, n_splits={n_splits}")
    min_train = min(len(tr) for tr, _ in splits)
    clf = _probe_pipeline(Ridge(alpha=alpha), X.shape[1], min_train, pca_dim, seed)
    pred = cross_val_predict(clf, X, Y, cv=splits, method="predict")
    return pred


def _macro_f1(yi, pred):
    from sklearn.metrics import f1_score
    return float(f1_score(yi, pred, average="macro"))


def _r2(Y, pred):
    from sklearn.metrics import r2_score
    return float(r2_score(Y, pred, multioutput="uniform_average"))


def _bootstrap_indices(n: int, rng: np.random.Generator, groups=None):
    if groups is None:
        return rng.integers(0, n, n)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    sampled = rng.choice(uniq, size=len(uniq), replace=True)
    return np.concatenate([np.where(groups == g)[0] for g in sampled])


def score_classification(yi, pred, seed, n_boot=1000, n_chance=25, groups=None):
    rng = np.random.default_rng(seed)
    metric = _macro_f1(yi, pred)
    chance = float(np.mean([_macro_f1(yi, rng.permutation(yi)) for _ in range(n_chance)]))
    n = len(yi)
    boot = []
    for _ in range(n_boot):
        idx = _bootstrap_indices(n, rng, groups)
        if len(set(yi[idx])) < 2:
            continue
        boot.append(_macro_f1(yi[idx], pred[idx]))
    lo, hi = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) if boot else (metric, metric)
    return {"metric": metric, "metric_name": "macro_f1", "chance": chance,
            "skill": metric - chance, "ci95": [lo - chance, hi - chance], "n": int(n)}


def score_regression(Y, pred, seed, n_boot=1000, groups=None):
    rng = np.random.default_rng(seed)
    metric = _r2(Y, pred)
    n = len(Y)
    boot = []
    for _ in range(n_boot):
        idx = _bootstrap_indices(n, rng, groups)
        boot.append(_r2(Y[idx], pred[idx]))
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    return {"metric": metric, "metric_name": "r2", "chance": 0.0,
            "skill": metric, "ci95": [lo, hi], "n": int(n)}


# --------------------------------------------------------------------------
# Ceiling / floor / gap
# --------------------------------------------------------------------------

def recoverability(ceiling_skill: float, floor_skill: float) -> dict:
    gap = ceiling_skill - floor_skill
    rho = float(np.clip(floor_skill / ceiling_skill, 0.0, 1.0)) if ceiling_skill > 1e-6 else float("nan")
    return {"gap": float(gap), "recovery_ratio": rho}


# --------------------------------------------------------------------------
# Decision logic (PROTOCOL §6)
# --------------------------------------------------------------------------

DECISION_THRESHOLDS = {
    "margin_macro_f1": 0.10,   # ceiling must beat baseline by this (or non-overlapping CI)
    "rho_lo": 0.50,            # dictionary-bottleneck threshold
    "rho_hi": 0.80,            # dictionary-near-faithful threshold
    "phi_rich": 0.50,          # decoder retains >=50% of ESM-2 skill
    "min_tasks": 2,
}


def per_model_verdict(model_tasks: dict, thr: dict = DECISION_THRESHOLDS) -> dict:
    """model_tasks[task] = {ceiling, floor, baseline, esm2, rho, ceiling_ci, baseline_ci}."""
    rich, bottleneck, faithful = [], [], []
    for task, m in model_tasks.items():
        if task in REPORT_ONLY_TASKS:
            continue  # stratified EC is family-leaking: report-only, not a gate
        c, b = m["ceiling"], m["baseline"]
        # Richness = ceiling beats the composition/chance baseline by margin or
        # by a non-overlapping CI. (The old `beats_rand` gate was dropped: the
        # PCA dimensionality control now handles the feature-count confound, and
        # the up-projecting R_rand was information-preserving, so it did no work.)
        beats = (c - b >= thr["margin_macro_f1"]) or (m.get("ceiling_ci", [0])[0] > m.get("baseline_ci", [0, 0])[1])
        if beats:
            rich.append(task)
            if not np.isnan(m["rho"]) and m["rho"] <= thr["rho_lo"]:
                bottleneck.append(task)
            if not np.isnan(m["rho"]) and m["rho"] >= thr["rho_hi"]:
                faithful.append(task)
    n = thr["min_tasks"]
    substrate_rich = len(rich) >= n
    substrate_thin = len(rich) == 0
    is_bottleneck = substrate_rich and len(bottleneck) >= n
    near_faithful = substrate_rich and len([t for t in rich if t in faithful]) >= len(rich)
    return {"rich_tasks": rich, "bottleneck_tasks": bottleneck, "faithful_tasks": faithful,
            "substrate_rich": substrate_rich, "substrate_thin": substrate_thin,
            "dictionary_bottleneck": is_bottleneck, "dictionary_near_faithful": near_faithful}


def retrain_go_nogo(model_verdicts: dict, model_gaps: dict, thr: dict = DECISION_THRESHOLDS) -> dict:
    """PROTOCOL §6.3. Returns the single GO/NO-GO and the retrain target."""
    go_models = [m for m, v in model_verdicts.items()
                 if v["substrate_rich"] and v["dictionary_bottleneck"]]
    if go_models:
        target = max(go_models, key=lambda m: model_gaps.get(m, 0.0))
        return {"decision": "GO", "reason": "rich substrate + dictionary bottleneck",
                "retrain_target": target, "candidates": go_models}
    if all(v["substrate_thin"] for v in model_verdicts.values()):
        return {"decision": "NO-GO", "reason": "substrate thin for all models (H2)",
                "retrain_target": None}
    if any(v["dictionary_near_faithful"] for v in model_verdicts.values()):
        return {"decision": "NO-GO", "reason": "dictionary already near-faithful on rich tasks",
                "retrain_target": None}
    return {"decision": "NO-GO", "reason": "no model meets the GO conditions",
            "retrain_target": None}
