"""Tests for what the relational stage publishes about its own uncertainty.

The relational design draws many anchors from each protein, so an interval taken
over anchors treats up to `--groups-per-protein` correlated draws as independent.
`within_anchor_auc` will compute the interval the sampling unit supports, but only
if it is told which protein each pair came from; the stage did not tell it, so the
clustered interval was never published and the anchor-level one -- which the
function's own docstring calls anti-conservative -- stood alone. This is the
statistic the audit's power gate will read before the retracted relational effect
may be re-established, so the assertion here is that the assembled record carries
it, on the proteins the split actually held out.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.transfer.relational import PREDICTOR_ARMS, AnchoredPairs  # noqa: E402


def _load_stage(filename: str):
    """Import a stage whose module name starts with a digit."""

    path = REPO / "scripts/transfer" / filename
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


STAGE05 = _load_stage("05_relational_channel.py")

N_PROTEINS = 8
N_TRAIN_PROTEINS = 5
GROUPS_PER_PROTEIN = 4
DECOYS_PER_GROUP = 3
RESIDUES = 40
HIDDEN_WIDTH = 12
ATTENTION_WIDTH = 6


def _protein_record(generator: np.random.Generator) -> dict:
    """One protein's encoded anchors, in the shape `assemble` consumes.

    The states carry a weak pair signal so the probes have something to fit; what
    is under test is which sampling unit the interval is taken over, not how well
    anything separates.
    """

    anchors: list[int] = []
    partners: list[int] = []
    labels: list[int] = []
    separations: list[int] = []
    groups: list[int] = []
    for group in range(GROUPS_PER_PROTEIN):
        anchor = int(generator.integers(0, RESIDUES // 2))
        candidates = generator.choice(
            np.arange(RESIDUES // 2, RESIDUES), DECOYS_PER_GROUP + 1, replace=False
        )
        for rank, partner in enumerate(candidates):
            anchors.append(anchor)
            partners.append(int(partner))
            labels.append(int(rank == 0))
            separations.append(abs(int(partner) - anchor))
            groups.append(group)
    pairs = AnchoredPairs(
        anchor=np.asarray(anchors, dtype=np.int64),
        partner=np.asarray(partners, dtype=np.int64),
        label=np.asarray(labels, dtype=np.int64),
        separation=np.asarray(separations, dtype=np.int64),
        group=np.asarray(groups, dtype=np.int64),
    )
    hidden = generator.normal(size=(RESIDUES, HIDDEN_WIDTH)).astype(np.float32)
    hidden[pairs.partner[pairs.label == 1]] += 1.5
    attention = generator.normal(
        size=(pairs.anchor.size, ATTENTION_WIDTH)
    ).astype(np.float32)
    return {"hidden": hidden, "attention": attention, "pairs": pairs}


def test_the_assembled_record_carries_the_protein_clustered_interval() -> None:
    generator = np.random.default_rng(20260812)
    records = [_protein_record(generator) for _ in range(N_PROTEINS)]
    train_mask = np.zeros(N_PROTEINS, dtype=bool)
    train_mask[:N_TRAIN_PROTEINS] = True

    block = STAGE05.assemble(records, train_mask, pca_dim=4, seed=3)

    held_out = N_PROTEINS - N_TRAIN_PROTEINS
    assert block["n_test_proteins"] == held_out
    for name in PREDICTOR_ARMS:
        for estimator in ("linear", "mlp"):
            interval = block["anchored_partner_identification"][name][estimator]
            # Both are published: the anchor-level figure because frozen artefacts
            # quote it, the clustered one because it is the interval the draw
            # supports, and the gap between them is itself the diagnostic.
            assert interval["sem_cluster_unit"] == "anchor"
            assert interval["sem"] > 0.0
            assert interval["n_proteins"] == held_out, f"{name}/{estimator}"
            assert interval["sem_protein_clustered"] is not None, f"{name}/{estimator}"
            assert interval["n_groups"] == held_out * GROUPS_PER_PROTEIN
