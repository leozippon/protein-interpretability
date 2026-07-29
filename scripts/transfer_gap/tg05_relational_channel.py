"""TG-05: does protein-relevant structure live in per-position states or in the
attention pattern?

Attribution graphs freeze attention patterns and explain the feature-to-feature
pathway. That approximation is cheap in text, where most per-step computation is
resolvable from a short window. In proteins the functional unit is a *relation*
between residues that are far apart in sequence. This script measures, for the
same model and cohort, how well spatial contact is decodable from

    single   per-position hidden states of the two residues (concatenated)
    product  their elementwise product (a bilinear per-pair readout)
    attn     the attention weights between them across all layers and heads

with matched feature dimensionality and protein-disjoint evaluation. If `attn`
dominates, the pathway a frozen-attention attribution graph discards is where
the relational computation is.

**SUPERSEDED, and its headline is retracted.** `src/transfer/relational.py` with
`scripts/transfer/05_relational_channel.py` measures the same contrast with a
seeded structure order and a separation-only control. Under those the attention
margin over per-position marginals fell from about 0.10 to 0.03-0.05, and to
about 0.03 over the separation-only control, which withdrew the motivation for
the cross-position residue-pair transcoder line. Prefer that entry point.

Two defects are corrected here rather than left in place, because a superseded
script that still runs is a script that will still be run:

*ZymCTRL was scored unconditioned.* This module reached `protein_input` directly
instead of going through a cohort, and the old `protein_input` had no ZymCTRL
branch, so it fed the bare sequence to an EC-conditioned model -- the exact thing
`load_zymctrl` in the same file warned against, three functions away. The
rendering now comes from `src.transfer.arms`, which refuses a conditioned arm
without its label, so this arm raises instead of returning a number. It is
refused explicitly below, with a pointer, rather than by traceback.

*Structures were taken in filename order and split in that order.* The train/test
split was therefore by accession block, which is the cohort-selection artefact
the superseding run identified. Structures are now drawn under a seeded
permutation.
"""

from __future__ import annotations

import argparse
import gzip
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from tg_common import (
    DEFAULT_COHORT_SEED,
    REPO,
    analysis_layer,
    load_arm,
    protein_input,
    write_json,
)
from tg_contract import refuse_unless_eligible, stage_contract_record

ALPHAFOLD = REPO / "data/alphafold"
THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
    "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
    "TRP": "W", "TYR": "Y",
}


def read_structure(path: Path):
    coords, plddt, seq = [], [], []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                name = line[17:20].strip()
                if name not in THREE_TO_ONE:
                    continue
                coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
                plddt.append(float(line[60:66]))
                seq.append(THREE_TO_ONE[name])
    return np.asarray(coords), np.asarray(plddt), "".join(seq)


@torch.no_grad()
def encode(arm, seq: str, layers: list[int]):
    """Per-residue hidden states at `layers` and the full attention stack."""
    text = protein_input(arm, seq)
    ids = torch.tensor(
        [arm.tokenizer(text, return_tensors=None)["input_ids"]], dtype=torch.long
    ).to(arm.device)
    n_tok = ids.shape[1]
    offset = n_tok - len(seq)  # leading control token, if any
    if offset < 0:
        raise ValueError(f"{arm.name}: tokenization shorter than sequence")

    store = {}
    handles = [
        arm.blocks()[layer].register_forward_hook(
            lambda _m, _i, out, layer=layer: store.__setitem__(
                layer, (out[0] if isinstance(out, tuple) else out)[0].float().cpu()
            )
        )
        for layer in layers
    ]
    try:
        out = arm.model(input_ids=ids, output_attentions=True)
    finally:
        for h in handles:
            h.remove()
    hidden = np.concatenate([store[layer][offset:].numpy() for layer in layers], axis=1)
    attn = torch.stack([a[0] for a in out.attentions]).float().cpu().numpy()
    attn = attn[:, :, offset:, offset:]  # (layers, heads, L, L)
    if hidden.shape[0] != len(seq) or attn.shape[2] != len(seq):
        raise ValueError(f"{arm.name}: alignment failure for length {len(seq)}")
    return hidden, attn


def sample_pairs(n: int, contact: np.ndarray, rng, per_protein: int, n_neg: int = 8):
    """Anchored partner-identification groups.

    A model scored on unanchored pairs can reach high AUC using only additive
    marginals ("both residues are buried"), which says nothing about whether it
    represents *which* residue pairs with which. Each group here fixes the anchor
    i and one true partner j+, then draws `n_neg` non-partners j- from the same
    sequence-separation band. Anything that discriminates within a group beyond
    the partner's own marginal propensity is genuine pair information.
    """
    bands = ((12, 24), (24, 48), (48, 128), (128, 10**9))
    part_i, part_j, labels, seps, groups = [], [], [], [], []
    gid = 0
    order = rng.permutation(n)
    for i in order:
        sep_all = np.abs(np.arange(n) - i)
        far = sep_all >= 12
        pos = np.flatnonzero(far & (contact[i] == 1))
        if pos.size == 0:
            continue
        jpos = int(rng.choice(pos))
        sep = abs(jpos - i)
        lo, hi = next(b for b in bands if b[0] <= sep < b[1])
        cand = np.flatnonzero(far & (contact[i] == 0) & (sep_all >= lo) & (sep_all < hi))
        if cand.size < n_neg:
            continue
        jneg = rng.choice(cand, n_neg, replace=False)
        for j, lab in [(jpos, 1)] + [(int(x), 0) for x in jneg]:
            part_i.append(int(i))
            part_j.append(int(j))
            labels.append(lab)
            seps.append(abs(j - int(i)))
            groups.append(gid)
        gid += 1
        if gid * (n_neg + 1) >= per_protein:
            break
    if gid < 4:
        return None
    return (
        np.asarray(part_i),
        np.asarray(part_j),
        np.asarray(labels, dtype=int),
        np.asarray(seps),
        np.asarray(groups),
    )


def evaluate(train_x, train_y, test_x, test_y, test_groups, seed):
    """Mean within-anchor AUC: can the model pick the true partner?"""
    scaler = StandardScaler().fit(train_x)
    model = LogisticRegression(max_iter=3000, C=1.0, random_state=seed)
    model.fit(scaler.transform(train_x), train_y)
    score = model.decision_function(scaler.transform(test_x))
    aucs = []
    for g in np.unique(test_groups):
        sel = test_groups == g
        y = test_y[sel]
        if y.sum() == 0 or y.sum() == y.size:
            continue
        aucs.append(roc_auc_score(y, score[sel]))
    return float(np.mean(aucs)), float(np.std(aucs, ddof=1) / math.sqrt(len(aucs))), len(aucs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-proteins", type=int, default=140)
    # Spelled like every other residue band in the series. It was --min-len
    # /--max-len, which is the same spelling eight stages use for a *token*
    # truncation, and tg_contract's band check looked for --res-min/--res-max by
    # name -- so a live 110-320 residue band read as no band at all, under a
    # contract note asserting this stage "has no residue band".
    ap.add_argument("--res-min", type=int, default=110)
    ap.add_argument("--res-max", type=int, default=320)
    ap.add_argument("--min-plddt", type=float, default=75.0)
    ap.add_argument("--pairs-per-protein", type=int, default=240)
    ap.add_argument("--contact-angstrom", type=float, default=8.0)
    ap.add_argument("--pca-dim", type=int, default=0, help="0 = match attn width")
    ap.add_argument("--seed", type=int, default=DEFAULT_COHORT_SEED)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Before the model load, and from the contract rather than from arm names.
    # This block used to be two hand-written refusals *after* `load_arm`: one on
    # the literal string "protgpt2", two lines above a correctly written
    # `input_format` check, and none at all for gpt2-large, which instead died
    # inside `encode` on a negative token offset. Because none of it was declared
    # in tg_contract, tg99 expected four arms from a stage that can produce one.
    # The properties are ArmSpec fields -- residue-level tokenisation, and an
    # input format that is not ec_conditioned -- so an arm outside TG_PANEL with
    # the same properties is admitted for the same reason progen2-medium is.
    refuse_unless_eligible("tg05", args.arm)

    arm = load_arm(args.arm, device=args.device, attn_implementation="eager")
    # Through the panel's one depth convention. This line carried a *third*
    # convention -- bare truncation, `int(r * (n_layer - 1))` -- which survived
    # the EXP-R2-066 unification of `int(round(...))` and `floor(... + 0.5)`.
    # It rounds every non-integral product down, so it disagrees with
    # `analysis_layer` far more often than the two conventions that pass did:
    # the artefact in `results/transfer_gap_20260724/tg05/` records
    # `layers: [8, 13, 17]` for progen2-medium where the panel's depth 0.33 is
    # layer 9, and `[11, 17, 23]` for zymctrl where depths 0.33 and 0.5 are
    # layers 12 and 18. Those numbers were measured at depths this programme
    # names differently everywhere else; see the EXP-R2-067 log entry.
    layers = [analysis_layer(arm.n_layer, r) for r in (0.33, 0.5, 0.67)]
    rng = np.random.default_rng(args.seed)

    # Seeded permutation, not filename order: the train/test split below is a
    # prefix split, so a source-ordered scan makes it a split by accession block.
    available = sorted(ALPHAFOLD.glob("AF-*-model_v*.pdb.gz"))
    order = np.random.default_rng(args.seed).permutation(len(available))
    records = []
    for index in order:
        path = available[index]
        ca, plddt, seq = read_structure(path)
        if not (args.res_min <= len(seq) <= args.res_max):
            continue
        if len(ca) != len(seq) or plddt.mean() < args.min_plddt:
            continue
        dist = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
        contact = (dist < args.contact_angstrom).astype(np.int8)
        picks = sample_pairs(len(seq), contact, rng, args.pairs_per_protein)
        if picks is None:
            continue
        hidden, attn = encode(arm, seq, layers)
        i, j, y, sep, grp = picks
        # causal attention: only the later position can attend to the earlier
        late, early = np.maximum(i, j), np.minimum(i, j)
        rel = attn[:, :, late, early].reshape(-1, len(i)).T  # (pairs, layers*heads)
        records.append(
            dict(hi=hidden[i], hj=hidden[j], rel=rel, y=y, sep=sep,
                 grp=grp + 100_000 * len(records), protein=path.stem)
        )
        if len(records) >= args.n_proteins:
            break
    if len(records) < 60:
        raise RuntimeError(f"only {len(records)} usable structures")

    n_train = int(0.7 * len(records))
    train, test = records[:n_train], records[n_train:]
    attn_dim = train[0]["rel"].shape[1]
    pca_dim = args.pca_dim or max(8, attn_dim // 2)

    def stack(rows, key: str) -> np.ndarray:
        return np.concatenate([row[key] for row in rows], axis=0)

    y_tr, y_te = stack(train, "y"), stack(test, "y")

    pca = PCA(n_components=pca_dim, random_state=args.seed).fit(
        np.concatenate([stack(train, "hi"), stack(train, "hj")], axis=0)[::3]
    )
    tr_i, tr_j = pca.transform(stack(train, "hi")), pca.transform(stack(train, "hj"))
    te_i, te_j = pca.transform(stack(test, "hi")), pca.transform(stack(test, "hj"))

    grp_te = stack(test, "grp")
    arms_x = {
        "partner_marginal_only": (tr_j, te_j),
        "single_concat": (
            np.concatenate([tr_i, tr_j], axis=1),
            np.concatenate([te_i, te_j], axis=1),
        ),
        "bilinear_product": (tr_i * tr_j, te_i * te_j),
        "attention_pattern": (stack(train, "rel"), stack(test, "rel")),
        "separation_only": (
            np.log(stack(train, "sep"))[:, None],
            np.log(stack(test, "sep"))[:, None],
        ),
    }
    arms_x["single_plus_product"] = (
        np.concatenate([arms_x["single_concat"][0], arms_x["bilinear_product"][0]], axis=1),
        np.concatenate([arms_x["single_concat"][1], arms_x["bilinear_product"][1]], axis=1),
    )
    scored = {
        name: evaluate(xtr, y_tr, xte, y_te, grp_te, args.seed)
        for name, (xtr, xte) in arms_x.items()
    }
    auc = {name: value[0] for name, value in scored.items()}
    auc_sem = {name: value[1] for name, value in scored.items()}
    n_groups = next(iter(scored.values()))[2]

    payload = dict(
        arm=arm.name,
        contract=stage_contract_record("tg05", [arm.name]),
        residue_band=[args.res_min, args.res_max],
        layers=layers,
        n_proteins=len(records),
        n_train_proteins=len(train),
        n_test_proteins=len(test),
        n_train_pairs=int(y_tr.size),
        n_test_pairs=int(y_te.size),
        positive_rate=float(y_te.mean()),
        hidden_dim_before_pca=int(train[0]["hi"].shape[1]),
        pca_dim_per_position=pca_dim,
        attention_features=attn_dim,
        contact_angstrom=args.contact_angstrom,
        min_separation=12,
        n_test_anchor_groups=n_groups,
        anchored_partner_auc=auc,
        anchored_partner_auc_sem=auc_sem,
        seed=args.seed,
        structure_selection="seeded_permutation_of_all_alphafold_models",
        superseded_by="scripts/transfer/05_relational_channel.py",
    )
    out = Path(args.out) if args.out else (
        REPO / "results/transfer_gap_20260729_corrected/tg05"
    )
    write_json(out / f"{arm.name}.json", payload)
    for name, value in sorted(auc.items(), key=lambda kv: -kv[1]):
        print(f"  {name:24s} anchored AUC {value:.4f} +/- {auc_sem[name]:.4f}")


if __name__ == "__main__":
    main()
