"""Stage-3 semantics: is pair structure readable per-position, or only in attention?

Attribution graphs freeze the attention pattern and explain the residual
feature-to-feature pathway. In text that approximation is cheap because most of
what a step computes is resolvable from per-position states over a short window.
In proteins the functional unit is a *relation* between residues that are far
apart in sequence, so the question is whether that relation survives in the
per-position states at all.

Three design constraints make the answer trustworthy.

*Anchoring.* An unanchored contact classifier reaches roughly 0.85 AUC using
additive marginals alone - "both residues are buried" - without representing
which residue pairs with which. Every evaluation here fixes an anchor ``i`` with
one true long-range partner ``j+`` and scores it against decoys drawn from the
same sequence-separation band, so only within-anchor discrimination counts.

*Nonlinearity.* The pilot used linear probes only, so a per-position arm could
have lost to attention merely because the pair information is stored
nonlinearly. Every arm is therefore scored with both a linear probe and a small
MLP.

*Homology.* A random protein split leaks: homologues of a test protein sit in
the training set and a probe can memorise family-level contact patterns. The
split here is disjoint over homology clusters built from shared Pfam families
and k-mer similarity.

ProtGPT2 is refused outright. Its multi-residue BPE has no residue-to-token map,
so per-residue pair features are undefined for it; that refusal is part of the
finding rather than a limitation of this code.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .arms import Arm

#: Sequence-separation bands for decoy sampling. A decoy is drawn from the same
#: band as the true partner so that separation alone cannot identify the answer.
SEPARATION_BANDS: tuple[tuple[int, int], ...] = ((12, 24), (24, 48), (48, 128), (128, 1 << 30))

MIN_SEPARATION = 12
CONTACT_ANGSTROM = 8.0

PREDICTOR_ARMS = (
    "partner_marginal_only",
    "concat",
    "product",
    "concat_plus_product",
    "attention_pattern",
    "separation_only",
)


# ------------------------------------------------------------ token alignment


def require_residue_token_map(arm: Arm) -> None:
    """Refuse arms whose tokenizer cannot address individual residues."""

    if arm.modality != "protein":
        raise ValueError(f"{arm.name} is a {arm.modality} arm; this measurement is protein-only")
    if arm.spec.tokenisation != "residue":
        raise ValueError(
            f"{arm.name} tokenises as {arm.spec.tokenisation!r}: a multi-residue BPE "
            "vocabulary has no residue-to-token map, so per-residue pair features and "
            "residue-pair attention weights are undefined. This arm cannot be measured "
            "here and must not be approximated by a token-to-residue heuristic."
        )


def residue_token_offset(arm: Arm, input_string: str, sequence: str) -> tuple[list[int], int]:
    """Token ids for ``input_string`` and the index at which residue 1 sits.

    The alignment is verified token by token rather than inferred from a length
    difference, because a silent off-by-one would move every hidden state and
    every attention weight by one residue without any visible error.
    """

    require_residue_token_map(arm)
    ids = arm.tokenizer(input_string, return_tensors=None)["input_ids"]
    tokens = arm.tokenizer.convert_ids_to_tokens(ids)
    fmt = arm.spec.input_format
    if fmt == "ec_conditioned":
        if tokens.count("<start>") != 1:
            raise ValueError(f"{arm.name}: EC-conditioned input has no unique <start>")
        offset = tokens.index("<start>") + 1
    elif fmt == "n_to_c_control":
        offset = 1
    elif fmt == "raw":
        offset = 0
    else:
        raise ValueError(f"{arm.name}: unsupported input format {fmt!r}")
    window = tokens[offset : offset + len(sequence)]
    if len(window) != len(sequence) or any(a != b for a, b in zip(window, sequence)):
        raise ValueError(
            f"{arm.name}: residue tokens do not align with the sequence at offset {offset}"
        )
    return ids, offset


# ------------------------------------------------------------ anchored design


@dataclass(frozen=True)
class AnchoredPairs:
    """One anchor, one true partner and its decoys, repeated over anchors."""

    anchor: np.ndarray
    partner: np.ndarray
    label: np.ndarray
    separation: np.ndarray
    group: np.ndarray

    def __post_init__(self) -> None:
        shapes = {
            self.anchor.shape,
            self.partner.shape,
            self.label.shape,
            self.separation.shape,
            self.group.shape,
        }
        if len(shapes) != 1 or self.anchor.ndim != 1:
            raise ValueError("anchored-pair arrays must share one one-dimensional shape")

    @property
    def n_groups(self) -> int:
        return int(np.unique(self.group).size) if self.group.size else 0


def contact_map(ca: np.ndarray, angstrom: float = CONTACT_ANGSTROM) -> np.ndarray:
    if ca.ndim != 2 or ca.shape[1] != 3:
        raise ValueError("CA coordinates must have shape (n, 3)")
    if angstrom <= 0:
        raise ValueError("contact threshold must be positive")
    distance = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)
    return (distance < angstrom).astype(np.int8)


def anchored_pairs(
    contact: np.ndarray,
    generator: np.random.Generator,
    *,
    min_separation: int = MIN_SEPARATION,
    n_decoys: int = 8,
    max_groups: int,
) -> AnchoredPairs:
    """Sample anchor groups: one true long-range partner against band-matched decoys."""

    if contact.ndim != 2 or contact.shape[0] != contact.shape[1]:
        raise ValueError("contact map must be square")
    if min_separation < 1 or n_decoys < 1 or max_groups < 1:
        raise ValueError("min_separation, n_decoys and max_groups must be positive")
    if min_separation < SEPARATION_BANDS[0][0]:
        # Decoys are drawn from the band holding the true partner's separation.
        # A partner closer than the first band's floor belongs to no band, and
        # the band lookup below would raise StopIteration from inside a
        # generator expression -- a bare exception with no statement of what
        # went wrong.
        raise ValueError(
            f"min_separation {min_separation} falls below the first decoy band "
            f"{SEPARATION_BANDS[0]}, so a true partner could have no band to draw "
            "matched decoys from"
        )
    length = contact.shape[0]

    anchors: list[int] = []
    partners: list[int] = []
    labels: list[int] = []
    separations: list[int] = []
    groups: list[int] = []
    group_id = 0
    for anchor in generator.permutation(length):
        anchor = int(anchor)
        offsets = np.abs(np.arange(length) - anchor)
        distant = offsets >= min_separation
        true_partners = np.flatnonzero(distant & (contact[anchor] == 1))
        if true_partners.size == 0:
            continue
        partner = int(generator.choice(true_partners))
        separation = abs(partner - anchor)
        low, high = next(band for band in SEPARATION_BANDS if band[0] <= separation < band[1])
        candidates = np.flatnonzero(
            distant & (contact[anchor] == 0) & (offsets >= low) & (offsets < high)
        )
        if candidates.size < n_decoys:
            continue
        decoys = generator.choice(candidates, n_decoys, replace=False)
        for index, label in [(partner, 1)] + [(int(value), 0) for value in decoys]:
            anchors.append(anchor)
            partners.append(index)
            labels.append(label)
            separations.append(abs(index - anchor))
            groups.append(group_id)
        group_id += 1
        if group_id >= max_groups:
            break
    return AnchoredPairs(
        anchor=np.asarray(anchors, dtype=np.int64),
        partner=np.asarray(partners, dtype=np.int64),
        label=np.asarray(labels, dtype=np.int64),
        separation=np.asarray(separations, dtype=np.int64),
        group=np.asarray(groups, dtype=np.int64),
    )


# ------------------------------------------------------------------- encoding


@torch.no_grad()
def encode(
    arm: Arm,
    input_string: str,
    sequence: str,
    pairs: AnchoredPairs,
    *,
    layers: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-residue hidden states and per-pair attention weights, in one pass.

    Causal models only attend backwards, so a pair is read at ``a[later, earlier]``
    and the reverse entry is structurally zero.
    """

    if any(not 0 <= layer < arm.n_layer for layer in layers):
        raise ValueError(f"{arm.name}: analysis layer outside [0, {arm.n_layer})")
    ids, offset = residue_token_offset(arm, input_string, sequence)
    tensor = torch.tensor([ids], dtype=torch.long, device=arm.device)
    output = arm.model(
        input_ids=tensor, output_hidden_states=True, output_attentions=True
    )
    if output.attentions is None or output.attentions[0] is None:
        raise RuntimeError(
            f"{arm.name}: attention weights are None; load the arm with "
            'attn_implementation="eager"'
        )
    if len(output.hidden_states) != arm.n_layer + 1:
        raise RuntimeError(f"{arm.name}: expected {arm.n_layer + 1} hidden-state tensors")

    end = offset + len(sequence)
    hidden = torch.cat(
        [output.hidden_states[layer + 1][0, offset:end] for layer in layers], dim=-1
    )
    stacked = torch.stack([block[0] for block in output.attentions])
    if stacked.shape[-1] != tensor.shape[1]:
        raise RuntimeError(f"{arm.name}: attention tensor does not span the input")
    window = stacked[:, :, offset:end, offset:end]
    later = torch.tensor(
        np.maximum(pairs.anchor, pairs.partner), dtype=torch.long, device=arm.device
    )
    earlier = torch.tensor(
        np.minimum(pairs.anchor, pairs.partner), dtype=torch.long, device=arm.device
    )
    selected = window[:, :, later, earlier]
    attention = selected.permute(2, 0, 1).reshape(pairs.anchor.size, -1)
    return (
        hidden.float().cpu().numpy().astype(np.float32),
        attention.float().cpu().numpy().astype(np.float32),
    )


# ---------------------------------------------------------------- homology


def kmer_set(sequence: str, k: int) -> set[str]:
    if k < 1:
        raise ValueError("k must be positive")
    return {sequence[i : i + k] for i in range(max(0, len(sequence) - k + 1))}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = len(left | right)
    return len(left & right) / union if union else 0.0


def homology_clusters(
    accessions: Sequence[str],
    sequences: Sequence[str],
    *,
    pfam_by_accession: Mapping[str, set[str]],
    kmer: int = 3,
    kmer_jaccard_threshold: float = 0.10,
    pfam_jaccard_threshold: float = 0.50,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Single-linkage homology clusters over domain architecture and k-mer overlap.

    The Pfam relation is Jaccard overlap of the two proteins' family sets, not
    the presence of any shared family. Sharing one hub domain - an ABC
    transporter cassette, a kinase fold - does not make two multi-domain
    proteins homologous, but under single linkage it chains the entire cohort
    into one cluster and leaves no split at all. Requiring that most of the
    architecture agrees keeps the relation close to actual homology, and
    single-domain proteins that share their only family still merge at Jaccard
    1.0. The k-mer criterion covers proteins with no Pfam annotation, which
    would otherwise be treated as unrelated to everything.
    """

    if len(accessions) != len(sequences) or not accessions:
        raise ValueError("accessions and sequences must be non-empty and aligned")
    if not 0.0 < kmer_jaccard_threshold <= 1.0:
        raise ValueError("kmer_jaccard_threshold must lie in (0, 1]")
    if not 0.0 < pfam_jaccard_threshold <= 1.0:
        raise ValueError("pfam_jaccard_threshold must lie in (0, 1]")
    n = len(accessions)
    parent = list(range(n))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    families = [set(pfam_by_accession.get(accession, set())) for accession in accessions]
    kmers = [kmer_set(sequence, kmer) for sequence in sequences]
    pfam_links = 0
    kmer_links = 0
    for left in range(n):
        for right in range(left + 1, n):
            if find(left) == find(right):
                continue
            if (
                families[left]
                and families[right]
                and _jaccard(families[left], families[right]) >= pfam_jaccard_threshold
            ):
                union(left, right)
                pfam_links += 1
            elif _jaccard(kmers[left], kmers[right]) >= kmer_jaccard_threshold:
                union(left, right)
                kmer_links += 1

    labels = np.asarray([find(index) for index in range(n)], dtype=np.int64)
    _, cluster_ids, sizes = np.unique(labels, return_inverse=True, return_counts=True)
    unannotated = sum(1 for entry in families if not entry)
    return cluster_ids.astype(np.int64), {
        "n_proteins": n,
        "n_clusters": int(sizes.size),
        "largest_cluster_size": int(sizes.max()),
        "n_singleton_clusters": int((sizes == 1).sum()),
        "n_proteins_without_pfam": unannotated,
        "pfam_merges": pfam_links,
        "kmer_merges": kmer_links,
        "kmer_k": int(kmer),
        "kmer_jaccard_threshold": float(kmer_jaccard_threshold),
        "pfam_jaccard_threshold": float(pfam_jaccard_threshold),
    }


def homology_disjoint_split(
    cluster_ids: np.ndarray, *, train_fraction: float, seed: int, min_side: int = 2
) -> np.ndarray:
    """Boolean training mask that never splits a homology cluster."""

    if cluster_ids.ndim != 1 or cluster_ids.size < 2:
        raise ValueError("cluster_ids must be a one-dimensional array of at least two items")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie strictly between zero and one")
    if min_side < 1:
        raise ValueError("min_side must be positive")
    generator = np.random.default_rng(seed)
    unique = np.unique(cluster_ids)
    if unique.size < 2:
        raise RuntimeError(
            "every protein falls in one homology cluster; a homology-disjoint "
            "split does not exist for this selection"
        )
    target = train_fraction * cluster_ids.size
    train_clusters: set[int] = set()
    assigned = 0
    for cluster in generator.permutation(unique):
        if assigned >= target:
            break
        train_clusters.add(int(cluster))
        assigned += int((cluster_ids == cluster).sum())
    mask = np.isin(cluster_ids, list(train_clusters))
    if int(mask.sum()) < min_side or int((~mask).sum()) < min_side:
        raise RuntimeError(
            f"homology-disjoint split gave {int(mask.sum())} train and "
            f"{int((~mask).sum())} test proteins with {unique.size} clusters over "
            f"{cluster_ids.size} proteins; the selection is too homology-collapsed "
            f"for a {min_side}-protein minimum on both sides"
        )
    return mask


def random_protein_split(n: int, *, train_fraction: float, seed: int) -> np.ndarray:
    """Random protein-level split, retained only as the leaky contrast."""

    if n < 2:
        raise ValueError("at least two proteins are required")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie strictly between zero and one")
    generator = np.random.default_rng(seed)
    mask = np.zeros(n, dtype=bool)
    count = max(1, min(n - 1, int(round(train_fraction * n))))
    mask[generator.permutation(n)[:count]] = True
    return mask


# ------------------------------------------------------------------ scoring


def within_anchor_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    *,
    proteins: np.ndarray | None = None,
) -> dict[str, Any]:
    """Mean AUC computed inside each anchor group, with an honest interval.

    ``sem`` over anchor groups treats every anchor as an independent draw. Many
    anchors come from one protein and share its fold, its composition and its
    contact statistics, so that interval is anti-conservative by roughly the
    square root of the anchors-per-protein ratio. Supplying ``proteins`` -- one
    protein identifier per pair, aligned with ``groups`` -- adds
    ``sem_protein_clustered``, computed by averaging within protein first and
    taking the standard error over proteins, which is the interval the
    per-protein sampling design actually supports. The anchor-level figure is
    kept beside it rather than replaced, because several frozen artefacts quote
    it and the difference between the two is itself the diagnostic.
    """

    if labels.shape != scores.shape or labels.shape != groups.shape:
        raise ValueError("labels, scores and groups must align")
    if proteins is not None and proteins.shape != labels.shape:
        raise ValueError("protein identifiers must align with the pair list")
    values: list[float] = []
    owners: list[Any] = []
    skipped = 0
    for group in np.unique(groups):
        selected = groups == group
        y = labels[selected]
        if y.sum() == 0 or y.sum() == y.size:
            skipped += 1
            continue
        values.append(float(roc_auc_score(y, scores[selected])))
        if proteins is not None:
            owning = np.unique(proteins[selected])
            if owning.size != 1:
                raise ValueError(
                    "an anchor group spans more than one protein, so it is not a "
                    "within-protein anchor"
                )
            owners.append(owning[0])
    if len(values) < 2:
        raise RuntimeError("fewer than two scorable anchor groups")
    array = np.asarray(values, dtype=np.float64)
    report: dict[str, Any] = {
        "auc": float(array.mean()),
        "sem": float(array.std(ddof=1) / math.sqrt(array.size)),
        "n_groups": len(values),
        "n_groups_skipped": skipped,
        "sem_cluster_unit": "anchor",
        "sem_protein_clustered": None,
        "n_proteins": None,
    }
    if proteins is not None:
        owner_array = np.asarray(owners)
        unique_owners = np.unique(owner_array)
        report["n_proteins"] = int(unique_owners.size)
        if unique_owners.size >= 2:
            per_protein = np.asarray(
                [array[owner_array == owner].mean() for owner in unique_owners],
                dtype=np.float64,
            )
            report["sem_protein_clustered"] = float(
                per_protein.std(ddof=1) / math.sqrt(per_protein.size)
            )
    return report


def evaluate_predictor(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    test_groups: np.ndarray,
    *,
    seed: int,
    mlp_hidden: tuple[int, ...] = (128,),
    mlp_max_iter: int = 300,
    test_proteins: np.ndarray | None = None,
) -> dict[str, Any]:
    """Within-anchor AUC of a linear probe and of a small MLP on the same features.

    ``test_proteins`` is passed straight to :func:`within_anchor_auc`; supply it
    whenever the test side spans more than one protein, so that the reported
    interval is clustered on the sampling unit rather than on the anchor.
    """

    if train_x.ndim != 2 or test_x.ndim != 2 or train_x.shape[1] != test_x.shape[1]:
        raise ValueError("train and test feature matrices must share a width")
    if train_y.size != train_x.shape[0] or test_y.size != test_x.shape[0]:
        raise ValueError("labels do not align with feature matrices")
    if set(np.unique(train_y).tolist()) != {0, 1}:
        raise ValueError("training labels must contain both classes")

    linear = LogisticRegression(max_iter=3000, C=1.0, random_state=seed)
    nonlinear = MLPClassifier(
        hidden_layer_sizes=mlp_hidden,
        alpha=1e-3,
        max_iter=mlp_max_iter,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=seed,
    )
    scaler = StandardScaler().fit(train_x)
    train_z, test_z = scaler.transform(train_x), scaler.transform(test_x)
    linear.fit(train_z, train_y)
    nonlinear.fit(train_z, train_y)
    return {
        "n_features": int(train_x.shape[1]),
        "linear": within_anchor_auc(
            test_y,
            linear.decision_function(test_z),
            test_groups,
            proteins=test_proteins,
        ),
        "mlp": within_anchor_auc(
            test_y,
            nonlinear.predict_proba(test_z)[:, 1],
            test_groups,
            proteins=test_proteins,
        ),
        "mlp_hidden_layer_sizes": list(mlp_hidden),
        "mlp_iterations": int(nonlinear.n_iter_),
    }


def build_feature_arms(
    projected_anchor: np.ndarray,
    projected_partner: np.ndarray,
    attention: np.ndarray,
    separation: np.ndarray,
) -> dict[str, np.ndarray]:
    """Feature matrices for every predictor arm on one split side."""

    if projected_anchor.shape != projected_partner.shape:
        raise ValueError("anchor and partner projections must share a shape")
    if attention.shape[0] != projected_anchor.shape[0]:
        raise ValueError("attention features do not align with the pair list")
    if separation.shape[0] != projected_anchor.shape[0]:
        raise ValueError("separations do not align with the pair list")
    if np.any(separation < 1):
        raise ValueError("sequence separations must be positive")
    product = projected_anchor * projected_partner
    concat = np.concatenate([projected_anchor, projected_partner], axis=1)
    return {
        "partner_marginal_only": projected_partner,
        "concat": concat,
        "product": product,
        "concat_plus_product": np.concatenate([concat, product], axis=1),
        "attention_pattern": attention,
        "separation_only": np.log(separation.astype(np.float64))[:, None],
    }


def fit_position_projection(
    residue_states: Sequence[np.ndarray], *, n_components: int, seed: int, stride: int = 3
) -> PCA:
    """PCA on training residue states only, so the test side stays unseen."""

    if n_components < 1 or stride < 1:
        raise ValueError("n_components and stride must be positive")
    matrix = np.concatenate([block[::stride] for block in residue_states], axis=0)
    if matrix.shape[0] <= n_components:
        raise ValueError(
            f"PCA needs more than {n_components} training rows, found {matrix.shape[0]}"
        )
    return PCA(n_components=n_components, svd_solver="randomized", random_state=seed).fit(matrix)
