#!/usr/bin/env python3
"""Build the pseudokinase contradiction set, and measure whether it is one.

Why this set exists
===================

Three measurements in this programme -- F10, F12 and D3.g -- were each matched or
beaten by a residue-statistics baseline. The diagnosis is not that the models are
empty; it is that all three were read on an **agreement set**, a cohort on which
evolutionary sequence statistics and biological knowledge predict the same label.
On an agreement set a model that recombines corpus fragments and a model that
knows biology score alike, so the measurement cannot separate them however
careful the statistics are.

Pseudokinases are the cleanest natural **contradiction set** a protein corpus
offers. They carry the eukaryotic protein-kinase fold, they score on the Pfam
kinase models, their nearest sequence neighbours are active kinases -- and they
are catalytically dead, because the catalytic machinery itself has degraded: the
beta3 VAIK lysine that positions the alpha/beta phosphates, the catalytic-loop
HRD aspartate that acts as the catalytic base, the activation-loop DFG aspartate
that chelates Mg2+. Global sequence statistics say "kinase"; experiment says "not
a kinase, catalytically". The two sides disagree, which is the property an
agreement set lacks.

What this script therefore does is not only assemble records. It **measures the
contradiction** rather than assuming it, because the premise can fail in three
separate ways and each failure changes what a downstream experiment may claim:

* the HMM score may already separate the strata, in which case the Pfam baseline
  is not at chance and the contradiction against *sequence statistics* is partial;
* the nearest active kinase may be far away, in which case a retrieval baseline
  is not fooled either;
* Swiss-Prot may annotate the pseudokinases correctly as inactive, in which case
  the contradiction is with sequence statistics but **not** with the annotation,
  and a corpus-memorisation account of the model is not excluded by this set.

All three are computed and written into the manifest. A verdict that the set does
not contradict is a legitimate output of this script.

The third of them has already come back negative, and the result is recorded
rather than absorbed. Swiss-Prot annotates most of the experimentally dead
stratum correctly: about half the records say "pseudokinase" or "catalytically
inactive" in the FUNCTION comment itself, and most of the rest carry no kinase EC
number at all, so an annotation-reading baseline is not fooled and in fact does
rather well. The contradiction this set carries is therefore with **sequence
statistics** and with **retrieval**, and not with the corpus annotation. What
follows for a downstream measurement is that a sequence-only protein model, whose
training corpus never contained the annotation, is measured cleanly here, while a
joint language-protein model is not: for that case the annotation channel has to
be closed with the description-masking machinery in
:mod:`src.transfer.sequence_description` before any separation means anything.
The five dead records whose annotation does side with "kinase" are listed in the
manifest, and five is below the resampling floor, so they are a note and not a
stratum.

The caveat that governs the whole design
========================================

**A motif-aware residue baseline solves this task almost by construction.** Read
the residue at the three catalytic columns and the pseudokinases fall out. That
is not a defect to be hidden; it is the reason the set is informative, and it
fixes what a downstream measurement is allowed to claim:

* against **global** sequence statistics -- HMM bit score, k-mer profile,
  nearest-neighbour retrieval, and the corpus annotation -- the set is a genuine
  contradiction, and those are precisely the baselines that beat the models in
  F10/F12/D3.g;
* against a **motif-aware** baseline it is not. A model that separates the strata
  has at most shown that it locates the catalytic machinery, which a strong
  sequence model could also learn from conservation statistics alone.

The stratum that contradicts even the motif baseline is
``active_despite_degradation``: kinases whose canonical motifs are degraded at
the canonical *positions* and which are nonetheless experimentally active -- the
WNKs, whose catalytic lysine sits on beta2 rather than beta3; CASK, which lacks
the DFG aspartate and phosphorylates neurexin-1 in a Mg2+-independent manner;
POMK and PKDCC, both once classified as pseudokinases and both shown to be active
kinases. That stratum is small, and its size is reported rather than padded.

Ground truth, and where it comes from
=====================================

Every record's *identity* is machine-readable: the gene symbol is resolved
through UniProt's own human ID mapping, the sequence, EC numbers, GO terms with
their evidence codes, Pfam and InterPro cross-references and the FUNCTION text
come from the Swiss-Prot release through this repository's one XML reader,
:func:`src.transfer.sequence_description.iter_swissprot_entries`, and the motif
status is computed from the sequence by aligning it to the Pfam kinase model.

The *label* -- catalytically dead, or active -- is not machine-readable, and this
script does not pretend otherwise. Swiss-Prot does not carry a "catalytically
dead" field, and inferring the label from the absence of an EC number would make
the label a function of the annotation, which is one of the three things the set
exists to test. So the label is curated, in :data:`CURATED_GENES`, from the
experimental literature, and every record carries ``label_provenance =
"curated_literature"`` together with the evidence kind and the citation it rests
on. The manifest says the same thing once more in its limitations block. A reader
who distrusts a particular call can drop that record by gene symbol.

The curation is stratified rather than merged, because the pseudokinase
literature has retracted itself repeatedly:

``dead_experimental``
    A published experiment addressed to catalysis: a negative in vitro kinase
    assay on purified protein, a structure showing a catalytically incompetent
    site, or a kinase-dead-equivalent genetic rescue. Each record also carries
    ``confidence``, ``high`` or ``moderate``, and moderate means the call rests
    on one study, on a study of the domain rather than the protein, or on
    orthologue work.

``dead_predicted``
    Motif degradation and kinome classification only. No assay addressed to
    catalysis that this curation can point to. These are **not** positives and
    are written to their own stratum.

``contested``
    Activity reported by some groups and absent in others, or a classification
    that has been reversed. ERBB3, whose intracellular domain does catalyse
    autophosphorylation at roughly a thousandth of EGFR's rate; ILK, reported as
    an active kinase in 1996 and shown to have a pseudoactive site in 2009;
    KSR2, TRIB2, ROR2, BUB1B, PIK3R4. None of them may enter the positives.

``active_despite_degradation``
    Experimentally active despite failing the canonical-position motif test.

``excluded_domain_level``
    Proteins that carry a pseudokinase *domain* inside an active kinase, or a
    pseudokinase outside the ePK fold. JAK1/2/3 and TYK2 have a JH2 pseudokinase
    domain and an active JH1 domain, so as whole-sequence records they are active
    kinases and belong to neither stratum; TRRAP and FAM20A are catalytically
    dead and are not ePK-fold, so including them would confound the fold this set
    holds constant. They are excluded from the automatic active pool as well,
    which is the point of listing them.

The active controls are **not** curated by hand. They are selected by
machine-readable criteria -- a Pfam kinase domain, a full kinase EC number, and a
GO protein-kinase-activity term with a non-IEA evidence code, so the activity
label has curated experimental support on the same footing the dead stratum is
held to -- and then matched. Their motif status is measured, never required, so
that the motif baseline's accuracy on this cohort is a reading and not a
construction.

Matching, and what could not be matched
=======================================

Two matchings are built, because no single one is right.

The primary pairing matches each contradiction-eligible ``dead_experimental``
record 1:1 to an active kinase on the Pfam model it hits (PF00069 or PF07714,
which separates the Ser/Thr branch from the Tyr branch), on kinase-domain **bit
score** inside a declared caliper, and on sequence length. Bit-score matching is
what makes a downstream positive interpretable: pseudokinases score lower than
active kinases on the kinase models, so an unmatched comparison lets the HMM
baseline succeed for a reason that has nothing to do with catalysis. A dead
record with no admissible partner inside the caliper is reported as unmatched
rather than paired with something the HMM can tell apart.

What that costs is family. Matching on score pairs TRIB1 with a TGF-beta
receptor because the two happen to score alike, and the realised pairs share a
Pfam model, a bit score, a length -- and little else. So a second pairing is
built on maximal kinase-domain identity, giving each pseudokinase its closest
active relative, which is the family match; it recovers EPHA10 against EPHA7,
IRAK2 against IRAK1, ULK4 against ULK3, VRK3 against VRK2. The HMM baseline is
reported on both, and the two numbers are the honest statement of the trade-off:
score-matched, the Pfam score is at chance by construction; family-matched, it is
not, because the family match restores the score gap that the pseudokinase's
degradation opened.

Only records the statistics side calls kinases enter either pairing. A
pseudokinase below Pfam's own gathering threshold is one Pfam does not claim, so
it carries no contradiction; it is written out with the flag rather than matched.

Matching on kinome group -- AGC, CAMK, CMGC, STE, TK, TKL -- was not possible:
no kinome classification table is on disk, and this host has no route to fetch
one. What stands in for it is the Pfam model, the InterPro entry overlap, and the
kinase-domain identity to both matched partners, all of which are reported per
record.

Leakage
=======

The split is by near-duplicate group, through
:mod:`src.transfer.near_duplicates`, never by record: limitation L30 measured
that a record-level split leaves 42.5% of held-out records with a relative at 95%
identity or above. The tension particular to this set is that a pseudokinase and
its matched active kinase are homologous *by construction*, so a homology
exclusion would empty the cohort. It is not applied, and the reason is the same
one that module argues at length: the relation used for splitting is
near-duplication, not homology. Homologues are free to fall on opposite sides and
the residual is measured and reported as a curve rather than gated.

Two further constraints follow from what the split is for. The unit of
independence is the **matched pair**, not the record, so a pair is never divided;
and the pseudokinase/active contrast is evaluated *within* a side, never across
the boundary, so homology between a pseudokinase and its own control is a
property of the contrast rather than leakage across the split.

Reproducing
===========

HMMER 3.4 and a pressed Pfam-A are required and are not defaulted to any path on
this host: pass ``--hmmsearch``, ``--hmmfetch`` and ``--pfam-a``. The run streams
the Swiss-Prot XML once, which is the dominant cost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer.near_duplicates import (  # noqa: E402
    boundary_containment,
    group_disjoint_split,
    near_duplicate_groups,
)
from src.transfer.sequence_description import (  # noqa: E402
    IEA_EVIDENCE,
    canonical_description,
    iter_swissprot_entries,
)
from src.transfer.statistics import MINIMUM_BOOTSTRAP_UNITS  # noqa: E402


# ------------------------------------------------------------ curated labels


@dataclass(frozen=True)
class CuratedGene:
    """One hand-curated label, with the evidence it rests on named."""

    gene: str
    stratum: str
    #: ``high`` or ``moderate``; ``""`` where the stratum does not grade.
    confidence: str
    #: What kind of experiment the call rests on, in the terms of the literature.
    evidence: str
    #: The studies the call rests on. Assistant domain knowledge, not a field of
    #: any file on disk -- see the ``limitations`` block of the manifest.
    citation: str
    note: str = ""
    #: A non-protein-kinase catalytic activity the same protein does have. A
    #: record carrying one is still a dead kinase, but it is not a dead *enzyme*,
    #: and a downstream contrast that reads "enzyme vs not" would be confounded.
    other_catalytic_activity: str = ""


DEAD_EXPERIMENTAL = "dead_experimental"
DEAD_PREDICTED = "dead_predicted"
CONTESTED = "contested"
ACTIVE_DESPITE_DEGRADATION = "active_despite_degradation"
EXCLUDED_DOMAIN_LEVEL = "excluded_domain_level"
ACTIVE_MATCHED = "active_matched"
ACTIVE_POOL = "active_pool"

CURATED_STRATA: tuple[str, ...] = (
    DEAD_EXPERIMENTAL,
    DEAD_PREDICTED,
    CONTESTED,
    ACTIVE_DESPITE_DEGRADATION,
    EXCLUDED_DOMAIN_LEVEL,
)

CURATED_GENES: tuple[CuratedGene, ...] = (
    # --- experimentally demonstrated catalytic deficiency -------------------
    CuratedGene(
        "STRADA", DEAD_EXPERIMENTAL, "high",
        "no activity on purified protein; crystal structure of the "
        "LKB1-STRADalpha-MO25 complex shows a closed, catalytically incompetent "
        "site that binds ATP without turnover",
        "Boudeau et al. 2003 EMBO J 22:5102; Zeqiraj et al. 2009 Science 326:1707",
        note="allosteric activator of LKB1",
    ),
    CuratedGene(
        "STRADB", DEAD_EXPERIMENTAL, "moderate",
        "assayed alongside STRADalpha without detectable activity; same degraded "
        "catalytic-loop and activation-loop aspartates",
        "Boudeau et al. 2003 EMBO J 22:5102",
    ),
    CuratedGene(
        "TRIB1", DEAD_EXPERIMENTAL, "high",
        "purified pseudokinase domain binds nucleotide only weakly and transfers "
        "no phosphate; structure with the C/EBPalpha degron",
        "Murphy et al. 2015 Structure 23:2111",
    ),
    CuratedGene(
        "TRIB3", DEAD_EXPERIMENTAL, "moderate",
        "degenerate catalytic loop, no detectable kinase activity; acts by "
        "substrate sequestration and degron presentation",
        "Hegedus et al. 2007 Cell Signal 19:238; Murphy et al. 2015 Structure 23:2111",
    ),
    CuratedGene(
        "VRK3", DEAD_EXPERIMENTAL, "high",
        "crystal structure of a degraded catalytic site that cannot bind ATP; "
        "no activity in vitro",
        "Scheeff et al. 2009 Structure 17:128",
    ),
    CuratedGene(
        "MLKL", DEAD_EXPERIMENTAL, "high",
        "human pseudokinase domain shows no detectable protein kinase activity; "
        "necroptosis is executed by its N-terminal four-helix bundle",
        "Murphy et al. 2013 Immunity 39:443",
    ),
    CuratedGene(
        "PEAK1", DEAD_EXPERIMENTAL, "high",
        "crystal structure of the pseudokinase domain; no nucleotide binding and "
        "no catalytic activity; functions as a dimerisation scaffold",
        "Patel et al. 2017 Nat Commun 8:1157; Ha & Boggon 2018 J Biol Chem 293:1642",
    ),
    CuratedGene(
        "PRAG1", DEAD_EXPERIMENTAL, "high",
        "SgK223/PRAG1 pseudokinase domain does not bind nucleotide and has no "
        "catalytic activity",
        "Patel et al. 2017 Nat Commun 8:1157",
    ),
    CuratedGene(
        "PEAK3", DEAD_EXPERIMENTAL, "moderate",
        "third PEAK-family pseudokinase; no catalytic activity reported, "
        "scaffolding function established",
        "Lopez et al. 2019 Biochem J 476:3241; Hou et al. 2021 J Mol Biol 433:166989",
    ),
    CuratedGene(
        "PTK7", DEAD_EXPERIMENTAL, "high",
        "receptor pseudokinase; the kinase domain neither binds nucleotide "
        "appreciably nor transfers phosphate",
        "Sheetz et al. 2020 Mol Cell 79:390",
    ),
    CuratedGene(
        "RYK", DEAD_EXPERIMENTAL, "high",
        "receptor pseudokinase with a degenerate catalytic loop; no activity",
        "Katso et al. 1999 Oncogene 18:3746; Sheetz et al. 2020 Mol Cell 79:390",
    ),
    CuratedGene(
        "EPHB6", DEAD_EXPERIMENTAL, "moderate",
        "catalytic-loop aspartate replaced; no kinase activity; signals by "
        "heterodimerisation with catalytically active EPH receptors",
        "Gurniak & Berg 1996 Oncogene 13:777; Sheetz et al. 2020 Mol Cell 79:390",
    ),
    CuratedGene(
        "EPHA10", DEAD_EXPERIMENTAL, "moderate",
        "degenerate catalytic and activation-loop motifs; no detectable activity",
        "Sheetz et al. 2020 Mol Cell 79:390",
    ),
    CuratedGene(
        "IRAK2", DEAD_EXPERIMENTAL, "moderate",
        "catalytic-loop aspartate replaced; no kinase activity; scaffolds "
        "Myddosome signalling",
        "Wesche et al. 1999 J Biol Chem 274:19403",
    ),
    CuratedGene(
        "IRAK3", DEAD_EXPERIMENTAL, "high",
        "IRAK-M shows no autophosphorylation and no kinase activity; negative "
        "regulator of TLR signalling",
        "Wesche et al. 1999 J Biol Chem 274:19403; Kobayashi et al. 2002 Cell 110:191",
        other_catalytic_activity=(
            "a guanylate cyclase activity has been reported (Freihat et al. 2019 "
            "Sci Signal 12:eaau0637); it is not protein-kinase activity and is not "
            "independently replicated"
        ),
    ),
    CuratedGene(
        "KSR1", DEAD_EXPERIMENTAL, "high",
        "arginine in place of the beta3 VAIK lysine in human and mouse KSR1; no "
        "catalytic activity; allosteric activator of RAF through side-to-side "
        "dimerisation",
        "Rajakulendran et al. 2009 Nature 461:542; Brennan et al. 2011 Nature 472:366",
    ),
    CuratedGene(
        "ULK4", DEAD_EXPERIMENTAL, "moderate",
        "structure shows nucleotide binding without a competent catalytic "
        "machinery; no phosphotransfer",
        "Preuss et al. 2020 Biochem J 477:3081",
    ),
    CuratedGene(
        "PAN3", DEAD_EXPERIMENTAL, "moderate",
        "ATP-binding pseudokinase subunit of the PAN2-PAN3 deadenylase; no "
        "phosphotransfer",
        "Christie et al. 2013 EMBO J 32:1084; Schafer et al. 2014 Nat Struct Mol Biol 21:591",
        note="the direct assays are largely on the yeast and Drosophila orthologues",
    ),
    # --- pseudokinase by motif degradation and classification only ----------
    CuratedGene(
        "SCYL1", DEAD_PREDICTED, "", "kinome classification and motif degradation",
        "Manning et al. 2002 Science 298:1912",
    ),
    CuratedGene(
        "SCYL2", DEAD_PREDICTED, "", "kinome classification and motif degradation",
        "Manning et al. 2002 Science 298:1912",
    ),
    CuratedGene(
        "SCYL3", DEAD_PREDICTED, "", "kinome classification and motif degradation",
        "Manning et al. 2002 Science 298:1912",
    ),
    CuratedGene(
        "NRBP1", DEAD_PREDICTED, "", "kinome classification and motif degradation",
        "Manning et al. 2002 Science 298:1912",
    ),
    CuratedGene(
        "NRBP2", DEAD_PREDICTED, "", "kinome classification and motif degradation",
        "Manning et al. 2002 Science 298:1912",
    ),
    CuratedGene(
        "TEX14", DEAD_PREDICTED, "",
        "degenerate kinase domain; required for intercellular bridge formation",
        "Manning et al. 2002 Science 298:1912; Greenbaum et al. 2006 PNAS 103:4982",
    ),
    CuratedGene(
        "CAMKV", DEAD_PREDICTED, "",
        "CaMK-family kinase-like domain with degraded catalytic motifs",
        "Manning et al. 2002 Science 298:1912",
    ),
    CuratedGene(
        "PXK", DEAD_PREDICTED, "", "kinase-like domain with catalytic residues absent",
        "Manning et al. 2002 Science 298:1912",
    ),
    CuratedGene(
        "STK40", DEAD_PREDICTED, "", "kinase-like domain with degraded motifs",
        "Manning et al. 2002 Science 298:1912",
    ),
    CuratedGene(
        "RPS6KC1", DEAD_PREDICTED, "", "SgK494 pseudokinase domain",
        "Manning et al. 2002 Science 298:1912",
    ),
    CuratedGene(
        "RPS6KL1", DEAD_PREDICTED, "", "SgK495 pseudokinase domain",
        "Manning et al. 2002 Science 298:1912",
    ),
    CuratedGene(
        "NPR1", DEAD_PREDICTED, "",
        "kinase homology domain of a receptor guanylate cyclase; catalytic motifs "
        "degraded, allosteric role established, no negative kinase assay curated here",
        "Koller et al. 1992 J Biol Chem 267:23063 (KHD deletion)",
        other_catalytic_activity="guanylate cyclase, EC 4.6.1.2",
    ),
    CuratedGene(
        "NPR2", DEAD_PREDICTED, "", "kinase homology domain of a receptor guanylate cyclase",
        "Manning et al. 2002 Science 298:1912",
        other_catalytic_activity="guanylate cyclase, EC 4.6.1.2",
    ),
    CuratedGene(
        "GUCY2C", DEAD_PREDICTED, "", "kinase homology domain of a receptor guanylate cyclase",
        "Manning et al. 2002 Science 298:1912",
        other_catalytic_activity="guanylate cyclase, EC 4.6.1.2",
    ),
    CuratedGene(
        "GUCY2D", DEAD_PREDICTED, "", "kinase homology domain of a receptor guanylate cyclase",
        "Manning et al. 2002 Science 298:1912",
        other_catalytic_activity="guanylate cyclase, EC 4.6.1.2",
    ),
    CuratedGene(
        "GUCY2F", DEAD_PREDICTED, "", "kinase homology domain of a receptor guanylate cyclase",
        "Manning et al. 2002 Science 298:1912",
        other_catalytic_activity="guanylate cyclase, EC 4.6.1.2",
    ),
    CuratedGene(
        "RNASEL", DEAD_PREDICTED, "",
        "pseudokinase domain fused to a ribonuclease; the protein's catalytic "
        "output is endoribonucleolytic",
        "Huang et al. 2014 Science 343:1244",
        other_catalytic_activity="endoribonuclease",
    ),
    # --- contested -----------------------------------------------------------
    CuratedGene(
        "ERBB3", CONTESTED, "",
        "long classified as a pseudokinase; the purified intracellular domain "
        "binds ATP and catalyses autophosphorylation at roughly a thousandth of "
        "EGFR's rate. Whether that activity is physiological is unsettled",
        "Shi et al. 2010 PNAS 107:7692; Steinkamp et al. 2014 Mol Cell Biol 34:965",
    ),
    CuratedGene(
        "TRIB2", CONTESTED, "",
        "autophosphorylation on Ser83 reported for TRIB2 and not reproduced for "
        "the other Tribbles; the family's structural work argues against "
        "nucleotide-dependent catalysis",
        "Bailey et al. 2015 J Biol Chem 290:1428; Murphy et al. 2015 Structure 23:2111",
    ),
    CuratedGene(
        "KSR2", CONTESTED, "",
        "reported to phosphorylate MEK1 within a KSR2-MEK1 complex; other work "
        "treats KSR2 as an allosteric activator only. Note that KSR2 carries "
        "arginine at the beta3 position exactly as KSR1 does -- this script reads "
        "VAIR in both -- so the activity claim is made against, not with, the motif "
        "evidence",
        "Brennan et al. 2011 Nature 472:366",
    ),
    CuratedGene(
        "ILK", CONTESTED, "",
        "reported as an active Ser/Thr kinase in 1996; the 2009 structure shows a "
        "pseudoactive site, and kinase-dead knock-in mice and flies are rescued",
        "Hannigan et al. 1996 Nature 379:91; Fukuda et al. 2009 Mol Cell 36:819; "
        "Lange et al. 2009 Nature 461:1002",
        note="current consensus leans pseudokinase; kept out of the positives",
    ),
    CuratedGene(
        "ROR1", CONTESTED, "",
        "commonly listed as a pseudokinase, and all three catalytic residues are "
        "present in the release sequence (VAIK / HKD / DLG). Activity is reported "
        "in some assays and absent in others. Demoted from the positives by this "
        "script's own motif reading, which is the sort of check the stratum exists "
        "to survive",
        "Gentile et al. 2011 Cancer Res 71:3132; Sheetz et al. 2020 Mol Cell 79:390",
    ),
    CuratedGene(
        "ROR2", CONTESTED, "",
        "tyrosine kinase activity reported in some assays and absent in others; "
        "catalytic residues intact, as for ROR1",
        "Sheetz et al. 2020 Mol Cell 79:390",
    ),
    CuratedGene(
        "BUB1B", CONTESTED, "",
        "vertebrate BUBR1 described as an unusual pseudokinase; earlier reports of "
        "activity are attributed to co-purifying BUB1",
        "Suijkerbuijk et al. 2012 Curr Biol 22:1962",
    ),
    CuratedGene(
        "PIK3R4", CONTESTED, "",
        "the VPS15 protein-kinase-like domain was reported to be an active kinase "
        "in yeast; structures of the human PI3KC3 complexes do not settle it",
        "Stack & Emr 1994 J Biol Chem 269:31552; Rostislavleva et al. 2015 Science 350:aac7365",
    ),
    # --- experimentally active despite canonical-position motif degradation --
    CuratedGene(
        "CASK", ACTIVE_DESPITE_DEGRADATION, "high",
        "lacks the DFG aspartate that chelates Mg2+ and nonetheless phosphorylates "
        "neurexin-1 in a Mg2+-independent manner",
        "Mukherjee et al. 2008 Cell 133:328",
    ),
    CuratedGene(
        "WNK1", ACTIVE_DESPITE_DEGRADATION, "high",
        "the catalytic lysine is relocated from the beta3 VAIK position to beta2; "
        "the canonical-position test calls it degraded and the kinase is fully active",
        "Xu et al. 2000 J Biol Chem 275:16795; Min et al. 2004 Structure 12:1303",
    ),
    CuratedGene(
        "WNK2", ACTIVE_DESPITE_DEGRADATION, "high",
        "same relocated catalytic lysine as WNK1",
        "Xu et al. 2000 J Biol Chem 275:16795",
    ),
    CuratedGene(
        "WNK3", ACTIVE_DESPITE_DEGRADATION, "high",
        "same relocated catalytic lysine as WNK1",
        "Xu et al. 2000 J Biol Chem 275:16795",
    ),
    CuratedGene(
        "WNK4", ACTIVE_DESPITE_DEGRADATION, "high",
        "same relocated catalytic lysine as WNK1",
        "Xu et al. 2000 J Biol Chem 275:16795",
    ),
    CuratedGene(
        "POMK", ACTIVE_DESPITE_DEGRADATION, "high",
        "SGK196 was classified as a pseudokinase and is an active protein "
        "O-mannose kinase acting on alpha-dystroglycan",
        "Yoshida-Moriguchi et al. 2013 Science 341:896",
    ),
    CuratedGene(
        "PKDCC", ACTIVE_DESPITE_DEGRADATION, "high",
        "SgK493/VLK was classified as a pseudokinase and is a secreted tyrosine "
        "kinase acting in the secretory pathway",
        "Bordoli et al. 2014 Cell 158:1033",
    ),
    CuratedGene(
        "HASPIN", ACTIVE_DESPITE_DEGRADATION, "high",
        "haspin has a kinase domain so divergent that Pfam gives it a family of "
        "its own rather than the ePK models, and it phosphorylates histone H3 Thr3",
        "Dai et al. 2005 Genes Dev 19:472; Eswaran et al. 2009 PNAS 106:20198",
        note="formerly GSG2; carries PF12330 and neither ePK model",
    ),
    # --- excluded ------------------------------------------------------------
    CuratedGene(
        "JAK1", EXCLUDED_DOMAIN_LEVEL, "",
        "carries a JH2 pseudokinase domain and an active JH1 kinase domain; as a "
        "whole-sequence record the protein is an active kinase",
        "Ungureanu et al. 2011 Nat Struct Mol Biol 18:971",
    ),
    CuratedGene(
        "JAK2", EXCLUDED_DOMAIN_LEVEL, "",
        "JH2 has low dual-specificity activity of its own; whole-sequence record "
        "is an active kinase",
        "Ungureanu et al. 2011 Nat Struct Mol Biol 18:971",
    ),
    CuratedGene(
        "JAK3", EXCLUDED_DOMAIN_LEVEL, "", "JH2 pseudokinase domain inside an active kinase",
        "Ungureanu et al. 2011 Nat Struct Mol Biol 18:971",
    ),
    CuratedGene(
        "TYK2", EXCLUDED_DOMAIN_LEVEL, "", "JH2 pseudokinase domain inside an active kinase",
        "Ungureanu et al. 2011 Nat Struct Mol Biol 18:971",
    ),
    CuratedGene(
        "TRRAP", EXCLUDED_DOMAIN_LEVEL, "",
        "catalytically dead PIKK; not the ePK fold this set holds constant",
        "Manning et al. 2002 Science 298:1912",
    ),
    CuratedGene(
        "FAM20A", EXCLUDED_DOMAIN_LEVEL, "",
        "pseudokinase that allosterically activates FAM20C; Fam20 fold, not ePK",
        "Cui et al. 2015 eLife 4:e06120; Cui et al. 2017 Science 355:1050",
    ),
    CuratedGene(
        "TTN", EXCLUDED_DOMAIN_LEVEL, "",
        "kinase domain embedded in a 34,000-residue protein; a whole-sequence "
        "record would not be comparable to a 400-residue kinase",
        "Mayans et al. 1998 Nature 395:863",
    ),
    CuratedGene(
        "OBSCN", EXCLUDED_DOMAIN_LEVEL, "",
        "two kinase domains in a very large modular protein",
        "Manning et al. 2002 Science 298:1912",
    ),
    CuratedGene(
        "SPEG", EXCLUDED_DOMAIN_LEVEL, "",
        "two kinase domains of unequal integrity in one modular protein",
        "Manning et al. 2002 Science 298:1912",
    ),
)


# ------------------------------------------------- catalytic-column anchors

#: Pfam models of the eukaryotic protein-kinase fold. The two branches of the
#: superfamily: PF00069 covers the Ser/Thr side, PF07714 the Tyr side. Both are
#: fetched by *name* because Pfam accessions carry a version suffix that changes
#: between releases while the names do not.
KINASE_MODELS: tuple[str, ...] = ("Pkinase", "PK_Tyr_Ser-Thr")

#: Accessions of active kinases whose three catalytic residues are textbook, used
#: to locate the model columns those residues occupy. Every one of them must map
#: to the same column and must carry the expected residue, or the run fails: the
#: point of using four references and three is that a single wrong recollection
#: cannot silently move a column, and a moved column mis-scores every record.
#:
#: Numbering is **UniProt canonical-sequence numbering**, which is not the
#: numbering the literature uses for three of these seven, and the difference is
#: exactly the sort of error this whole set would be destroyed by:
#:
#: * PKA is always cited on the *mature* chain, whose initiator methionine is
#:   removed -- the textbook K72/D166/D184 are 73/167/185 in P17612;
#: * ERK2 is cited on the *rat* sequence -- K52/D147/D165 are 54/149/167 in the
#:   human P28482;
#: * Src is cited on the *chicken* sequence -- K295/D386/D404 are 298/389/407 in
#:   the human P12931.
#:
#: Each corrected number was confirmed against the motif it names in the release
#: sequence itself (``YAMKIL``, ``IYRDLKPEN``, ``QVTDFGFAK`` for PKA and so on),
#: not by choosing whichever offset made the column check pass.
REFERENCE_CATALYTIC_RESIDUES: dict[str, dict[str, tuple[int, int, int]]] = {
    "Pkinase": {
        "P17612": (73, 167, 185),   # PRKACA, PKA catalytic subunit alpha
        "P24941": (33, 127, 145),   # CDK2
        "P28482": (54, 149, 167),   # MAPK1 / ERK2
        "P31749": (179, 274, 292),  # AKT1
    },
    "PK_Tyr_Ser-Thr": {
        "P12931": (298, 389, 407),  # SRC
        "P00533": (745, 837, 855),  # EGFR
        "P06239": (273, 364, 382),  # LCK
    },
}

MOTIF_NAMES: tuple[str, ...] = ("vaik_lys", "hrd_asp", "dfg_asp")
MOTIF_EXPECTED: dict[str, str] = {"vaik_lys": "K", "hrd_asp": "D", "dfg_asp": "D"}

#: Model columns around each anchor that spell the motif out, so the reading is
#: ``HRN`` or ``SLE`` or ``DNA`` and not only "the aspartate is missing". The
#: distinction is load-bearing: RYK keeps all three catalytic residues and is
#: still a pseudokinase, because its degradation is ``DFG`` -> ``DNA``, which a
#: single-residue test cannot see.
MOTIF_CONTEXT_OFFSETS: dict[str, tuple[int, int]] = {
    "vaik_lys": (-3, 0),
    "hrd_asp": (-2, 0),
    "dfg_asp": (0, 2),
}

#: EC top-level branches that name protein-kinase activity.
KINASE_EC_PREFIXES: tuple[str, ...] = ("2.7.10.", "2.7.11.", "2.7.12.")

#: A GO molecular-function term whose name contains this is a protein-kinase
#: activity term for the purposes of the active-control criterion.
GO_KINASE_ACTIVITY = re.compile(r"\bprotein\b.*\bkinase activity\b|\bkinase activity\b")

#: Wordings a curator uses when the entry itself says the protein is not a
#: catalytically competent kinase. Used only to *audit* the annotation, never to
#: assign a label.
#:
#: A screen, not a classifier, and it is reported as one. Free text does not say
#: which protein a clause is about: Swiss-Prot's FGR entry describes "receptors
#: devoid of kinase activity" that FGR signals *for*, and PDPK1's mentions a
#: catalytically inactive partner. Both fire here and neither is a pseudokinase.
#: What makes the screen usable is the ratio rather than any single hit -- it
#: fires on about half the experimentally dead stratum and on well under one
#: percent of the active pool -- and every active-pool hit is listed in the
#: manifest so the errors are inspectable rather than assumed absent.
INACTIVITY_WORDING = re.compile(
    r"pseudokinase"
    r"|catalytic(?:ally)?\s+(?:inactive|dead|deficient|incompetent)"
    r"|inactive\s+(?:protein\s+)?kinase"
    r"|kinase[- ](?:inactive|dead)"
    r"|(?:has|have|shows?|displays?|possess(?:es)?|exhibits?)\s+no\s+"
    r"(?:detectable\s+)?(?:protein\s+)?kinase\s+activity"
    r"|lacks?\s+(?:detectable\s+)?(?:protein\s+)?kinase\s+activity"
    r"|(?:devoid|deprived)\s+of\s+(?:protein\s+)?kinase\s+activity"
    r"|no\s+(?:detectable\s+)?(?:protein\s+)?kinase\s+activity"
    r"|not\s+(?:a\s+)?(?:functional|active|catalytically\s+active)\s+kinase"
    r"|lacks?\s+(?:the\s+)?(?:conserved\s+)?catalytic\s+"
    r"(?:lysine|aspartate|residues?)",
    re.IGNORECASE,
)

#: Longest record admitted. Kinase-domain-bearing proteins run from ~250 to a few
#: thousand residues; the cap keeps titin-scale records out of a cohort whose
#: contrast is a single domain, and it is reported rather than silent.
MAX_SEQUENCE_LENGTH = 3000
MIN_SEQUENCE_LENGTH = 150

#: Widest kinase-domain bit-score gap a matched pair may carry. The point of
#: matching at all is to take the Pfam score away from the baseline, so a pair
#: whose two halves differ by more than this is not a control and is dropped: the
#: dead record is reported as unmatched rather than paired with something the HMM
#: can tell apart for a reason that has nothing to do with catalysis. Twenty bits
#: is roughly a tenth of the median active-kinase domain score; the realised
#: median gap is written to the manifest and is far smaller.
MATCH_BIT_CALIPER = 20.0

#: The fraction of split units placed on the fit side.
DEFAULT_TRAIN_FRACTION = 0.5

#: Identity thresholds the leakage curve is reported at.
LEAKAGE_IDENTITY_GRID: tuple[float, ...] = (0.3, 0.4, 0.5, 0.7, 0.9, 0.95, 1.0)

RECORD_SCHEMA_VERSION = 1


# --------------------------------------------------------------- utilities


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_gene_index(path: Path) -> dict[str, tuple[str, ...]]:
    """``Gene_Name`` -> accessions, from UniProt's own human ID mapping.

    Isoform-suffixed accessions (``P31946-2``) are dropped: they name a splice
    variant and not a Swiss-Prot entry, and the release XML is keyed on the
    unsuffixed accession.
    """

    import gzip

    if not path.exists():
        raise FileNotFoundError(f"human ID mapping not found at {path}")
    index: dict[str, list[str]] = defaultdict(list)
    with gzip.open(path, "rt") as handle:
        for line in handle:
            accession, field, value = line.rstrip("\n").split("\t")
            if field != "Gene_Name" or "-" in accession:
                continue
            index[value.upper()].append(accession)
    return {gene: tuple(sorted(set(accessions))) for gene, accessions in index.items()}


def write_fasta(path: Path, sequences: dict[str, str]) -> None:
    with path.open("w") as handle:
        for name, sequence in sequences.items():
            handle.write(f">{name}\n")
            for start in range(0, len(sequence), 60):
                handle.write(sequence[start : start + 60] + "\n")


# ------------------------------------------------------------------- HMMER


@dataclass(frozen=True)
class DomainHit:
    """One reported domain of one sequence against one model."""

    accession: str
    model: str
    sequence_score: float
    sequence_evalue: float
    domain_score: float
    domain_ievalue: float
    ali_from: int
    ali_to: int
    env_from: int
    env_to: int


def run_hmmsearch(
    *,
    hmmsearch: Path,
    model_hmm: Path,
    fasta: Path,
    domtbl: Path,
    stockholm: Path,
    cpu: int,
) -> None:
    """Search one model, keeping everything reportable in the alignment.

    The inclusion thresholds are opened up to the reporting thresholds on
    purpose. A pseudokinase whose motifs have degraded can fall below Pfam's
    gathering cut, and a run that silently dropped it would remove exactly the
    records this set is about; whether a record clears the gathering threshold is
    recorded separately as a *reading*, not used as a filter.
    """

    command = [
        str(hmmsearch),
        "--cpu", str(cpu),
        "-E", "1e3", "--domE", "1e3", "--incE", "1e3", "--incdomE", "1e3",
        "--domtblout", str(domtbl),
        "-A", str(stockholm),
        "-o", "/dev/null",
        str(model_hmm),
        str(fasta),
    ]
    subprocess.run(command, check=True)


def parse_domtblout(path: Path, model: str) -> dict[str, DomainHit]:
    """Best-scoring domain per sequence, from a ``--domtblout`` table."""

    best: dict[str, DomainHit] = {}
    with path.open() as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 23:
                raise RuntimeError(
                    f"{path}: domtblout row has {len(fields)} fields, expected at "
                    "least 23; the table format is not the one this parser reads"
                )
            hit = DomainHit(
                accession=fields[0],
                model=model,
                sequence_evalue=float(fields[6]),
                sequence_score=float(fields[7]),
                domain_ievalue=float(fields[12]),
                domain_score=float(fields[13]),
                env_from=int(fields[19]),
                env_to=int(fields[20]),
                ali_from=int(fields[17]),
                ali_to=int(fields[18]),
            )
            previous = best.get(hit.accession)
            if previous is None or hit.domain_score > previous.domain_score:
                best[hit.accession] = hit
    return best


def parse_stockholm_rows(path: Path) -> dict[str, str]:
    """Sequence rows of an interleaved Stockholm alignment, concatenated in order."""

    rows: dict[str, list[str]] = defaultdict(list)
    if not path.exists():
        raise FileNotFoundError(f"hmmsearch wrote no alignment at {path}")
    with path.open() as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            name, _, block = line.partition(" ")
            if not block:
                raise RuntimeError(f"{path}: alignment row {line!r} has no residues")
            rows[name].append(block.strip())
    return {name: "".join(blocks) for name, blocks in rows.items()}


def match_state_map(row: str, start: int, stop: int) -> dict[int, tuple[int, str]]:
    """Model match state -> (sequence position, residue), for one aligned row.

    HMMER's alignment convention carries the model geometry in the *case* of the
    characters: upper case and ``-`` occupy match columns, lower case and ``.``
    occupy insert columns. That is per-row information, so no column-consistency
    assumption is needed and no ``#=GC RF`` line has to be present.

    The walk is checked against the row's own coordinate suffix: if the residues
    consumed do not land exactly on ``stop`` the mapping is wrong and the run
    stops rather than reporting a residue from the wrong position.
    """

    mapping: dict[int, tuple[int, str]] = {}
    state = 1
    position = start
    for character in row:
        if character == "-":
            state += 1
        elif character == ".":
            continue
        elif character.isupper():
            mapping[state] = (position, character)
            state += 1
            position += 1
        elif character.islower():
            position += 1
        else:
            raise RuntimeError(
                f"unexpected character {character!r} in a Stockholm alignment row"
            )
    if position - 1 != stop:
        raise RuntimeError(
            f"alignment row ends at sequence position {position - 1} but its "
            f"coordinate suffix says {stop}; the row walk is wrong"
        )
    return mapping


def aligned_domains(stockholm: Path) -> dict[tuple[str, int, int], dict[int, tuple[int, str]]]:
    """Every aligned domain of a search, keyed on (accession, from, to)."""

    parsed: dict[tuple[str, int, int], dict[int, tuple[int, str]]] = {}
    for name, row in parse_stockholm_rows(stockholm).items():
        accession, _, span = name.rpartition("/")
        if not accession or "-" not in span:
            raise RuntimeError(
                f"alignment row name {name!r} does not carry a /from-to suffix"
            )
        start_text, _, stop_text = span.partition("-")
        start, stop = int(start_text), int(stop_text)
        parsed[(accession, start, stop)] = match_state_map(row, start, stop)
    return parsed


def locate_catalytic_columns(
    model: str,
    hits: dict[str, DomainHit],
    alignments: dict[tuple[str, int, int], dict[int, tuple[int, str]]],
) -> dict[str, int]:
    """Which model columns hold VAIK-K, HRD-D and DFG-D, from reference kinases.

    Derived rather than declared: each reference kinase's known catalytic residue
    numbers are looked up in its own alignment, and the columns they fall in must
    agree across every reference and must carry the expected residue. Anything
    less stops the run -- a wrong column would silently mis-score every record.
    """

    references = REFERENCE_CATALYTIC_RESIDUES[model]
    columns: dict[str, set[int]] = {name: set() for name in MOTIF_NAMES}
    for accession, residues in references.items():
        hit = hits.get(accession)
        if hit is None:
            raise RuntimeError(
                f"reference kinase {accession} produced no {model} domain; the "
                "catalytic columns cannot be located"
            )
        mapping = alignments.get((accession, hit.ali_from, hit.ali_to))
        if mapping is None:
            raise RuntimeError(
                f"reference kinase {accession} has a {model} domain at "
                f"{hit.ali_from}-{hit.ali_to} with no alignment row"
            )
        reverse = {position: state for state, (position, _) in mapping.items()}
        for name, residue_number in zip(MOTIF_NAMES, residues, strict=True):
            state = reverse.get(residue_number)
            if state is None:
                raise RuntimeError(
                    f"{accession}: residue {residue_number} ({name}) is not inside "
                    f"the aligned {model} domain {hit.ali_from}-{hit.ali_to}"
                )
            observed = mapping[state][1]
            if observed != MOTIF_EXPECTED[name]:
                raise RuntimeError(
                    f"{accession}: residue {residue_number} is {observed}, not the "
                    f"{MOTIF_EXPECTED[name]} that {name} must be; the reference "
                    "numbering does not match this release's sequence"
                )
            columns[name].add(state)
    resolved: dict[str, int] = {}
    for name, states in columns.items():
        if len(states) != 1:
            raise RuntimeError(
                f"{model}: the reference kinases disagree about the {name} column "
                f"({sorted(states)}); one of the reference residue numbers is wrong"
            )
        resolved[name] = states.pop()
    return resolved


def motif_status(
    mapping: dict[int, tuple[int, str]], columns: dict[str, int]
) -> dict[str, Any]:
    """The residue each catalytic column holds in one record, and the verdict."""

    status: dict[str, Any] = {}
    intact = 0
    for name, state in columns.items():
        first, last = MOTIF_CONTEXT_OFFSETS[name]
        context = "".join(
            mapping.get(state + offset, (0, "-"))[1] for offset in range(first, last + 1)
        )
        entry = mapping.get(state)
        if entry is None:
            status[name] = {
                "residue": None,
                "position": None,
                "intact": False,
                "motif": context,
            }
            continue
        position, residue = entry
        is_intact = residue == MOTIF_EXPECTED[name]
        intact += int(is_intact)
        status[name] = {
            "residue": residue,
            "position": position,
            "intact": bool(is_intact),
            "motif": context,
        }
    status["n_intact"] = intact
    # A column the alignment left empty is not the same finding as a column
    # holding the wrong residue, and conflating them is how a divergent *active*
    # kinase acquires a false degradation: MKNK1, PINK1 and the EIF2AK family all
    # read as missing their beta3 lysine here because Pfam's alignment opens a gap
    # there, not because the lysine is gone. Both counts are reported so a reader
    # can require a substitution rather than an absence.
    status["n_substituted"] = sum(
        1
        for name in MOTIF_NAMES
        if status[name]["residue"] is not None and not status[name]["intact"]
    )
    status["n_unaligned"] = sum(
        1 for name in MOTIF_NAMES if status[name]["residue"] is None
    )
    status["all_intact"] = intact == len(MOTIF_NAMES)
    status["degraded_motifs"] = tuple(
        name for name in MOTIF_NAMES if not status[name]["intact"]
    )
    return status


# ------------------------------------------------------------- measurements


def domain_identity(
    left: dict[int, tuple[int, str]], right: dict[int, tuple[int, str]]
) -> float:
    """Percent identity over the model columns two records both occupy.

    Computed on the HMM alignment rather than on a pairwise aligner because it is
    the same geometry the motif reading uses, so the identity and the motif call
    cannot disagree about which residue is where. Columns only one record
    occupies are not counted, so this is identity over the shared core.
    """

    shared = left.keys() & right.keys()
    if not shared:
        return 0.0
    same = sum(1 for state in shared if left[state][1] == right[state][1])
    return same / len(shared)


def kmer_profile(sequence: str, k: int = 3) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for start in range(len(sequence) - k + 1):
        counts[sequence[start : start + k]] += 1
    return dict(counts)


def cosine(left: dict[str, int], right: dict[str, int]) -> float:
    shared = left.keys() & right.keys()
    numerator = sum(left[key] * right[key] for key in shared)
    left_norm = np.sqrt(sum(value * value for value in left.values()))
    right_norm = np.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(numerator / (left_norm * right_norm))


def bootstrap_auc(
    scores: np.ndarray,
    labels: np.ndarray,
    units: np.ndarray,
    *,
    seed: int,
    resamples: int = 2000,
) -> dict[str, float]:
    """AUROC with a percentile interval resampled over *units*, not records."""

    if np.unique(labels).size < 2:
        raise ValueError("an AUROC needs both classes present")
    point = float(roc_auc_score(labels, scores))
    unique_units = np.unique(units)
    if unique_units.size < MINIMUM_BOOTSTRAP_UNITS:
        raise ValueError(
            f"{unique_units.size} resampling units is below the floor of "
            f"{MINIMUM_BOOTSTRAP_UNITS}; no interval is reported"
        )
    rng = np.random.default_rng(seed)
    members = {unit: np.flatnonzero(units == unit) for unit in unique_units}
    draws: list[float] = []
    for _ in range(resamples):
        chosen = rng.choice(unique_units, size=unique_units.size, replace=True)
        index = np.concatenate([members[unit] for unit in chosen])
        if np.unique(labels[index]).size < 2:
            continue
        draws.append(float(roc_auc_score(labels[index], scores[index])))
    if len(draws) < resamples // 2:
        raise RuntimeError(
            f"only {len(draws)} of {resamples} bootstrap draws held both classes; "
            "the cohort is too unbalanced for a resampled interval"
        )
    array = np.asarray(draws)
    return {
        "auc": point,
        "ci_low": float(np.quantile(array, 0.025)),
        "ci_high": float(np.quantile(array, 0.975)),
        "n_units": int(unique_units.size),
        "n_effective_draws": len(draws),
    }


# ----------------------------------------------------------------- the build


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build and audit the pseudokinase contradiction set: catalytically "
            "dead protein kinases against matched active ones."
        )
    )
    parser.add_argument(
        "--xml", type=Path, default=REPO_ROOT / "data/swissprot/uniprot_sprot.xml.gz"
    )
    parser.add_argument(
        "--idmapping",
        type=Path,
        default=REPO_ROOT / "data/swissprot/HUMAN_9606_idmapping.dat.gz",
    )
    parser.add_argument("--hmmsearch", type=Path, required=True)
    parser.add_argument("--hmmfetch", type=Path, required=True)
    parser.add_argument(
        "--pfam-a",
        type=Path,
        required=True,
        help="Pfam-A.hmm; the kinase models are fetched from it by name",
    )
    parser.add_argument(
        "--hmmscan",
        type=Path,
        default=None,
        help=(
            "optional; when given, every record is scanned against all of Pfam-A "
            "so the manifest can report what the whole database, and not only the "
            "kinase models, calls each record"
        ),
    )
    parser.add_argument("--out", type=Path, required=True, help="JSONL destination")
    parser.add_argument(
        "--manifest", type=Path, default=None, help="defaults to <out>.manifest.json"
    )
    parser.add_argument("--work", type=Path, default=None, help="scratch directory")
    parser.add_argument("--cpu", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION
    )
    return parser


def collect_entries(xml: Path, wanted_accessions: set[str]) -> dict[str, Any]:
    """One pass of the release, keeping human entries that could matter.

    Kept: every entry whose Pfam cross-references include a kinase model, plus
    every accession named by the curation or by the catalytic-column references,
    so a pseudokinase too degraded for Pfam to annotate is still admitted and its
    absence from Pfam becomes a *finding* rather than a silent drop.
    """

    kept: dict[str, Any] = {}
    n_entries = 0
    n_human = 0
    kinase_pfam = {"PF00069", "PF07714"}
    for entry in iter_swissprot_entries(xml):
        n_entries += 1
        if not entry.entry_name.endswith("_HUMAN"):
            continue
        n_human += 1
        pfam_ids = {identifier for identifier, _ in entry.pfam}
        if not (pfam_ids & kinase_pfam) and entry.accession not in wanted_accessions:
            continue
        kept[entry.accession] = entry
    return {"entries": kept, "n_entries": n_entries, "n_human": n_human}


def main() -> None:
    args = build_parser().parse_args()
    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path: Path = args.manifest or out_path.with_suffix(out_path.suffix + ".manifest.json")
    work = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="pseudokinase-"))
    work.mkdir(parents=True, exist_ok=True)

    curated_by_gene = {gene.gene: gene for gene in CURATED_GENES}
    if len(curated_by_gene) != len(CURATED_GENES):
        raise RuntimeError("a gene symbol appears twice in the curation")
    declared = {gene.stratum for gene in CURATED_GENES} - set(CURATED_STRATA)
    if declared:
        raise RuntimeError(f"the curation uses undeclared strata {sorted(declared)}")

    # --- resolve curated gene symbols to Swiss-Prot accessions --------------
    gene_index = load_gene_index(args.idmapping)
    reference_accessions = {
        accession
        for model in REFERENCE_CATALYTIC_RESIDUES.values()
        for accession in model
    }
    curated_candidates: dict[str, tuple[str, ...]] = {}
    unresolved: list[str] = []
    for gene in curated_by_gene:
        accessions = gene_index.get(gene.upper(), ())
        if not accessions:
            unresolved.append(gene)
        else:
            curated_candidates[gene] = accessions
    if unresolved:
        raise RuntimeError(
            "these curated gene symbols are absent from the UniProt human ID "
            f"mapping and cannot be resolved to a record: {sorted(unresolved)}"
        )

    wanted = set(reference_accessions)
    for accessions in curated_candidates.values():
        wanted.update(accessions)

    print(f"streaming {args.xml} ...", flush=True)
    collected = collect_entries(args.xml, wanted)
    entries = collected["entries"]
    print(
        f"  {collected['n_entries']:,} entries, {collected['n_human']:,} human, "
        f"{len(entries):,} kinase-fold or named",
        flush=True,
    )

    # A curated gene may map to several accessions; exactly one must be a
    # reviewed human entry. Ambiguity is a curation failure, not something to
    # resolve by picking the first.
    curated_accession: dict[str, str] = {}
    for gene, accessions in curated_candidates.items():
        present = [accession for accession in accessions if accession in entries]
        if len(present) != 1:
            raise RuntimeError(
                f"{gene} resolves to {len(present)} Swiss-Prot human entries "
                f"({present}) out of candidates {list(accessions)}; the curation "
                "needs one entry per gene"
            )
        curated_accession[gene] = present[0]
    accession_curated = {value: key for key, value in curated_accession.items()}
    if len(accession_curated) != len(curated_accession):
        raise RuntimeError("two curated genes resolve to the same accession")

    for accession in reference_accessions:
        if accession not in entries:
            raise RuntimeError(
                f"catalytic-column reference {accession} is not in the release, or "
                "carries no kinase Pfam cross-reference"
            )

    # --- length admission ----------------------------------------------------
    length_rejected: dict[str, int] = {}
    pool: dict[str, Any] = {}
    for accession, entry in entries.items():
        length = len(entry.sequence)
        if MIN_SEQUENCE_LENGTH <= length <= MAX_SEQUENCE_LENGTH:
            pool[accession] = entry
        else:
            length_rejected[accession] = length
    curated_dropped_by_length = sorted(
        gene for gene, accession in curated_accession.items() if accession not in pool
    )
    # The length cap exists to keep titin-scale records out of a cohort whose
    # contrast is one domain. It must never be the thing that removes a labelled
    # record from a stratum a measurement is read on -- that would shrink the
    # positives for a reason unrelated to catalysis and leave no trace in the
    # counts, so a gene it removes has to be one the curation already excluded.
    silently_removed = [
        gene
        for gene in curated_dropped_by_length
        if curated_by_gene[gene].stratum != EXCLUDED_DOMAIN_LEVEL
    ]
    if silently_removed:
        raise RuntimeError(
            f"the length band {MIN_SEQUENCE_LENGTH}-{MAX_SEQUENCE_LENGTH} removes "
            f"{silently_removed}, which the curation places in a stratum a "
            "measurement is read on; widen the band or move the gene to "
            "excluded_domain_level with a reason"
        )
    for accession in reference_accessions:
        if accession not in pool:
            raise RuntimeError(f"reference {accession} fell outside the length band")

    fasta = work / "candidates.fasta"
    write_fasta(fasta, {accession: entry.sequence for accession, entry in pool.items()})
    print(f"  {len(pool):,} candidates written to {fasta}", flush=True)

    # --- kinase models -------------------------------------------------------
    hits: dict[str, dict[str, DomainHit]] = {}
    alignments: dict[str, dict[tuple[str, int, int], dict[int, tuple[int, str]]]] = {}
    columns: dict[str, dict[str, int]] = {}
    model_meta: dict[str, dict[str, Any]] = {}
    for model in KINASE_MODELS:
        model_hmm = work / f"{model}.hmm"
        with model_hmm.open("wb") as handle:
            subprocess.run(
                [str(args.hmmfetch), str(args.pfam_a), model], check=True, stdout=handle
            )
        header = {}
        with model_hmm.open() as handle:
            for line in handle:
                if line.startswith("HMM "):
                    break
                key, _, value = line.partition(" ")
                header.setdefault(key.strip(), value.strip())
        model_meta[model] = {
            "accession": header.get("ACC", ""),
            "length": int(header.get("LENG", "0")),
            "gathering_threshold": float(header.get("GA", "0 0").split()[0]),
        }
        domtbl = work / f"{model}.domtbl"
        stockholm = work / f"{model}.sto"
        run_hmmsearch(
            hmmsearch=args.hmmsearch,
            model_hmm=model_hmm,
            fasta=fasta,
            domtbl=domtbl,
            stockholm=stockholm,
            cpu=args.cpu,
        )
        hits[model] = parse_domtblout(domtbl, model)
        alignments[model] = aligned_domains(stockholm)
        columns[model] = locate_catalytic_columns(model, hits[model], alignments[model])
        print(
            f"  {model}: {len(hits[model]):,} sequences hit; catalytic columns "
            f"{columns[model]}",
            flush=True,
        )

    # --- per-record kinase-domain reading ------------------------------------
    best_model: dict[str, str] = {}
    for accession in pool:
        scored = [
            (hits[model][accession].domain_score, model)
            for model in KINASE_MODELS
            if accession in hits[model]
        ]
        if scored:
            best_model[accession] = max(scored)[1]

    def reading(accession: str) -> dict[str, Any] | None:
        model = best_model.get(accession)
        if model is None:
            return None
        hit = hits[model][accession]
        mapping = alignments[model].get((accession, hit.ali_from, hit.ali_to))
        if mapping is None:
            raise RuntimeError(
                f"{accession}: best {model} domain {hit.ali_from}-{hit.ali_to} has "
                "no alignment row"
            )
        return {
            "model": model,
            "model_accession": model_meta[model]["accession"],
            "sequence_bits": hit.sequence_score,
            "sequence_evalue": hit.sequence_evalue,
            "domain_bits": hit.domain_score,
            "domain_ievalue": hit.domain_ievalue,
            "domain_from": hit.ali_from,
            "domain_to": hit.ali_to,
            "passes_pfam_gathering": bool(
                hit.sequence_score >= model_meta[model]["gathering_threshold"]
            ),
            "motifs": motif_status(mapping, columns[model]),
            "_mapping": mapping,
        }

    readings = {accession: reading(accession) for accession in pool}

    # --- annotation audit ----------------------------------------------------
    def audit(entry: Any) -> dict[str, Any]:
        description = canonical_description(entry.protein_name, entry.function_texts)
        kinase_ec = [
            ec for ec in entry.ec if ec.startswith(KINASE_EC_PREFIXES)
        ]
        go_kinase = [
            {
                "go_id": annotation.go_id,
                "term": annotation.term,
                "evidence": annotation.evidence,
                "non_iea": annotation.evidence != IEA_EVIDENCE,
            }
            for annotation in entry.go
            if annotation.aspect == "F" and GO_KINASE_ACTIVITY.search(annotation.term)
        ]
        wording = INACTIVITY_WORDING.findall(description)
        return {
            "description": description,
            "description_chars": len(description),
            "ec": list(entry.ec),
            "kinase_ec": kinase_ec,
            "has_kinase_ec": bool(kinase_ec),
            "go_kinase_activity": go_kinase,
            "has_non_iea_kinase_go": any(item["non_iea"] for item in go_kinase),
            "function_text_flags_inactivity": bool(wording),
            "function_text_inactivity_wording": sorted({w.lower() for w in wording}),
            "interpro": [list(pair) for pair in entry.interpro],
            "pfam": [list(pair) for pair in entry.pfam],
        }

    audits = {accession: audit(entry) for accession, entry in pool.items()}

    def annotation_stance(accession: str) -> str:
        """What the corpus annotation itself says about this record's catalysis.

        Three answers, not two. The premise of a contradiction set is that the
        annotation sides with "kinase"; ``says_inactive`` means it does not, and a
        record in that class contradicts sequence statistics while agreeing with
        the corpus. ``silent`` -- no kinase EC, no curated kinase-activity term,
        no wording either way -- is neither, and is the largest class after
        ``says_inactive`` for the dead stratum.
        """

        record_audit = audits[accession]
        if record_audit["function_text_flags_inactivity"]:
            return "says_inactive"
        if record_audit["has_kinase_ec"] or record_audit["has_non_iea_kinase_go"]:
            return "says_kinase"
        return "silent"

    # --- strata --------------------------------------------------------------
    strata: dict[str, list[str]] = defaultdict(list)
    for gene, accession in curated_accession.items():
        if accession not in pool:
            continue
        strata[curated_by_gene[gene].stratum].append(accession)

    curated_accessions = set(accession_curated)
    active_pool: list[str] = []
    active_rejections: dict[str, int] = defaultdict(int)
    for accession, entry in pool.items():
        if accession in curated_accessions:
            active_rejections["curated_elsewhere"] += 1
            continue
        if readings[accession] is None:
            active_rejections["no_kinase_domain"] += 1
            continue
        record_audit = audits[accession]
        if not record_audit["has_kinase_ec"]:
            active_rejections["no_kinase_ec"] += 1
            continue
        if not record_audit["has_non_iea_kinase_go"]:
            active_rejections["no_non_iea_kinase_go"] += 1
            continue
        active_pool.append(accession)
    active_pool.sort()

    dead = sorted(strata[DEAD_EXPERIMENTAL])
    dead_with_domain = [a for a in dead if readings[a] is not None]
    # A dead record only *contradicts* the statistics side if the statistics side
    # calls it a kinase in the first place. One that falls below Pfam's own
    # gathering threshold is a pseudokinase Pfam does not claim, so it carries no
    # contradiction and is written out with the flag rather than matched.
    dead_eligible = [
        a for a in dead_with_domain if readings[a]["passes_pfam_gathering"]
    ]
    dead_not_eligible = {
        accession_curated[a]: (
            "no ePK domain reported"
            if readings[a] is None
            else f"{readings[a]['domain_bits']:.1f} bits, below Pfam gathering"
        )
        for a in dead
        if a not in dead_eligible
    }
    if len(dead_eligible) < MINIMUM_BOOTSTRAP_UNITS:
        raise RuntimeError(
            f"only {len(dead_eligible)} experimentally dead records are called a "
            f"kinase by Pfam, below the floor of {MINIMUM_BOOTSTRAP_UNITS}; the set "
            "cannot support a measurement and nothing has been written"
        )

    # --- 1:1 matching on model, bit score and length -------------------------
    def match_cost(dead_accession: str, active_accession: str) -> float | None:
        dead_read = readings[dead_accession]
        active_read = readings[active_accession]
        if dead_read["model"] != active_read["model"]:
            return None
        bits_gap = abs(dead_read["domain_bits"] - active_read["domain_bits"])
        length_gap = abs(
            np.log(len(pool[dead_accession].sequence))
            - np.log(len(pool[active_accession].sequence))
        )
        return bits_gap / 10.0 + length_gap

    remaining = set(active_pool)
    matches: dict[str, str] = {}
    unmatched: dict[str, str] = {}
    # Deterministic: the pseudokinase with the fewest admissible partners picks
    # first, ties broken on accession, so the assignment does not depend on dict
    # ordering and a scarce Tyr-branch record is not starved by a Ser/Thr one.
    order = sorted(
        dead_eligible,
        key=lambda accession: (
            sum(1 for other in active_pool if match_cost(accession, other) is not None),
            accession,
        ),
    )
    for dead_accession in order:
        costed = [
            (match_cost(dead_accession, other), other)
            for other in sorted(remaining)
            if match_cost(dead_accession, other) is not None
        ]
        if not costed:
            unmatched[accession_curated[dead_accession]] = (
                "no active kinase hits the same Pfam model"
            )
            continue
        _, chosen = min(costed)
        gap = abs(
            readings[dead_accession]["domain_bits"] - readings[chosen]["domain_bits"]
        )
        if gap > MATCH_BIT_CALIPER:
            unmatched[accession_curated[dead_accession]] = (
                f"nearest admissible active kinase is {gap:.1f} bits away, outside "
                f"the {MATCH_BIT_CALIPER:.0f}-bit caliper"
            )
            continue
        matches[dead_accession] = chosen
        remaining.discard(chosen)

    matched_actives = sorted(matches.values())
    if len(matches) < MINIMUM_BOOTSTRAP_UNITS:
        raise RuntimeError(
            f"only {len(matches)} matched pairs could be formed, below the floor of "
            f"{MINIMUM_BOOTSTRAP_UNITS}; nothing has been written"
        )

    # --- the cohort ----------------------------------------------------------
    # The whole active pool is carried, not only the matched half. Matching takes
    # the Pfam score away from the baseline and that is what makes a positive
    # interpretable; but it also decides for the reader how the control is drawn,
    # and a reader who wants the unmatched comparison -- or a different matching
    # -- can only have it if the pool is in the file.
    cohort: list[str] = sorted(
        set(dead)
        | set(active_pool)
        | set(strata[DEAD_PREDICTED])
        | set(strata[CONTESTED])
        | set(strata[ACTIVE_DESPITE_DEGRADATION])
        | {a for a in strata[EXCLUDED_DOMAIN_LEVEL] if a in pool}
    )
    matched_active_set = set(matched_actives)

    def stratum_of(accession: str) -> str:
        gene = accession_curated.get(accession)
        if gene is not None:
            return curated_by_gene[gene].stratum
        return ACTIVE_MATCHED if accession in matched_active_set else ACTIVE_POOL

    # --- near-duplicate grouping and the split -------------------------------
    sequences = [pool[accession].sequence for accession in cohort]
    groups, grouping_summary = near_duplicate_groups(sequences, unit="residues")
    position_of = {accession: index for index, accession in enumerate(cohort)}

    # The unit of independence is the matched pair. Two near-duplicate groups
    # joined by a pair become one split unit, so a pair is never divided and the
    # dead/active contrast is always evaluable inside a side.
    unit_parent = {int(group): int(group) for group in np.unique(groups)}

    def find(node: int) -> int:
        while unit_parent[node] != node:
            unit_parent[node] = unit_parent[unit_parent[node]]
            node = unit_parent[node]
        return node

    for dead_accession, active_accession in matches.items():
        left = find(int(groups[position_of[dead_accession]]))
        right = find(int(groups[position_of[active_accession]]))
        if left != right:
            unit_parent[right] = left
    relabel: dict[int, int] = {}
    units = np.empty(len(cohort), dtype=np.int64)
    for index in range(len(cohort)):
        root = find(int(groups[index]))
        if root not in relabel:
            relabel[root] = len(relabel)
        units[index] = relabel[root]

    n_units = int(np.unique(units).size)
    n_train_units = max(1, int(round(args.train_fraction * len(cohort))))
    train_mask, split_summary = group_disjoint_split(
        units, n_train=n_train_units, seed=args.seed, fraction_tolerance=0.12
    )
    leakage = boundary_containment(sequences, train_mask, unit="residues")

    # --- identity leakage curve, on the kinase domain ------------------------
    mappings = {
        accession: readings[accession]["_mapping"]
        for accession in cohort
        if readings[accession] is not None
    }
    train_accessions = [cohort[i] for i in np.flatnonzero(train_mask)]
    eval_accessions = [cohort[i] for i in np.flatnonzero(~train_mask)]
    max_identity_to_train: dict[str, float] = {}
    for accession in eval_accessions:
        if accession not in mappings:
            continue
        best = 0.0
        for other in train_accessions:
            if other in mappings:
                best = max(best, domain_identity(mappings[accession], mappings[other]))
        max_identity_to_train[accession] = best
    identity_values = np.asarray(sorted(max_identity_to_train.values()))
    identity_curve = {
        f"fraction_at_or_above_{threshold:g}": (
            float((identity_values >= threshold).mean()) if identity_values.size else 0.0
        )
        for threshold in LEAKAGE_IDENTITY_GRID
    }

    # --- does the statistics side get it wrong? ------------------------------
    def stratum_accessions(name: str) -> list[str]:
        return [a for a in cohort if stratum_of(a) == name and readings[a] is not None]

    # Every contradiction statistic is read on the records that carry a
    # contradiction: experimentally dead *and* called a kinase by Pfam.
    dead_set = sorted(dead_eligible)
    active_set = stratum_accessions(ACTIVE_MATCHED)

    pair_unit = {}
    for index, (dead_accession, active_accession) in enumerate(sorted(matches.items())):
        pair_unit[dead_accession] = index
        pair_unit[active_accession] = index

    matched_records = [a for a in dead_set + active_set if a in pair_unit]
    matched_labels = np.asarray(
        [1 if stratum_of(a) == DEAD_EXPERIMENTAL else 0 for a in matched_records]
    )
    matched_units = np.asarray([pair_unit[a] for a in matched_records])
    matched_bits = np.asarray([readings[a]["domain_bits"] for a in matched_records])

    unmatched_records = dead_set + [a for a in active_pool if a not in dead_set]
    unmatched_labels = np.asarray(
        [1 if a in set(dead_set) else 0 for a in unmatched_records]
    )
    unmatched_bits = np.asarray([readings[a]["domain_bits"] for a in unmatched_records])

    # A second, deliberately different matching. Bit-score matching buys HMM
    # neutrality and pays for it in family: it pairs TRIB1 with a TGF-beta
    # receptor because they happen to score alike. Matching instead on
    # kinase-domain identity gives each pseudokinase its closest active relative
    # -- the family match the brief asks for -- and gives the bit score back to
    # the baseline. Neither pairing is the right one on its own; the pair of them
    # is the honest statement of the trade-off, so both are built and both AUROCs
    # are reported.
    homology_remaining = set(active_pool)
    homology_matches: dict[str, str] = {}
    for dead_accession in sorted(dead_set):
        candidates = [
            (domain_identity(mappings[dead_accession], mappings[other]), other)
            for other in sorted(homology_remaining)
            if other in mappings
            and readings[other]["model"] == readings[dead_accession]["model"]
        ]
        if not candidates:
            continue
        _, chosen = max(candidates)
        homology_matches[dead_accession] = chosen
        homology_remaining.discard(chosen)
    homology_records = [
        accession
        for pair in sorted(homology_matches.items())
        for accession in pair
    ]
    homology_unit = {}
    for index, (dead_accession, active_accession) in enumerate(
        sorted(homology_matches.items())
    ):
        homology_unit[dead_accession] = index
        homology_unit[active_accession] = index
    homology_labels = np.asarray(
        [1 if a in set(dead_set) else 0 for a in homology_records]
    )
    homology_bits = np.asarray([readings[a]["domain_bits"] for a in homology_records])
    homology_units = np.asarray([homology_unit[a] for a in homology_records])
    homology_identities = [
        domain_identity(mappings[d], mappings[a]) for d, a in homology_matches.items()
    ]

    hmm_baseline = {
        "statistic": (
            "AUROC of the Pfam kinase-domain bit score for calling the "
            "experimentally dead stratum, oriented so that a *lower* bit score "
            "predicts dead. 0.5 is the value the contradiction premise assumes"
        ),
        "matched": bootstrap_auc(
            -matched_bits, matched_labels, matched_units, seed=args.seed
        ),
        "against_whole_active_pool": {
            "auc": float(roc_auc_score(unmatched_labels, -unmatched_bits)),
            "n_dead": int(unmatched_labels.sum()),
            "n_active": int((1 - unmatched_labels).sum()),
            "reading": (
                "what the Pfam score knows on its own, with no matching. This is "
                "the number that says how partial the contradiction against "
                "sequence statistics is; the matched value is 0.5 by construction"
            ),
        },
        "against_nearest_active_relative": {
            **bootstrap_auc(
                -homology_bits, homology_labels, homology_units, seed=args.seed
            ),
            "reading": (
                "the same statistic on the family-matched pairing, where the "
                "control is each pseudokinase's closest active relative rather "
                "than its bit-score twin"
            ),
        },
        "dead_bits": {
            "median": float(np.median([readings[a]["domain_bits"] for a in dead_set])),
            "min": float(min(readings[a]["domain_bits"] for a in dead_set)),
            "max": float(max(readings[a]["domain_bits"] for a in dead_set)),
        },
        "matched_active_bits": {
            "median": float(np.median([readings[a]["domain_bits"] for a in active_set])),
            "min": float(min(readings[a]["domain_bits"] for a in active_set)),
            "max": float(max(readings[a]["domain_bits"] for a in active_set)),
        },
        "n_dead_passing_pfam_gathering": sum(
            int(readings[a]["passes_pfam_gathering"]) for a in dead_set
        ),
        "n_matched_active_passing_pfam_gathering": sum(
            int(readings[a]["passes_pfam_gathering"]) for a in active_set
        ),
    }

    # motif baseline: the reading that succeeds by construction, reported so the
    # ceiling is explicit rather than discovered later.
    motif_scores = np.asarray(
        [-readings[a]["motifs"]["n_intact"] for a in matched_records], dtype=float
    )
    motif_baseline = {
        "statistic": (
            "AUROC of the number of intact catalytic residues, oriented so that "
            "fewer intact residues predicts dead. This baseline is expected to "
            "succeed: the strata are defined by catalysis and the motifs are its "
            "structural signature. It is the ceiling a model must be read against, "
            "not a baseline the set defeats"
        ),
        "matched": bootstrap_auc(
            motif_scores, matched_labels, matched_units, seed=args.seed
        ),
    }

    # Retrieval. The question is the one a nearest-neighbour baseline actually
    # asks: given a pseudokinase, what does the most similar sequence in the
    # cohort look like? If the answer is "an active kinase" for every record, a
    # retrieval baseline transfers the wrong label, and that is the second half of
    # the contradiction. Read over the whole cohort rather than over the matched
    # thirty, because the pool is what a retrieval baseline would search.
    active_strata = {ACTIVE_MATCHED, ACTIVE_POOL, EXCLUDED_DOMAIN_LEVEL}
    searchable = [a for a in cohort if a in mappings]
    profiles = {a: kmer_profile(pool[a].sequence) for a in searchable}
    similarities = {
        "kinase_domain_identity": lambda a, b: domain_identity(mappings[a], mappings[b]),
        "kmer3_cosine": lambda a, b: cosine(profiles[a], profiles[b]),
    }
    retrieval: dict[str, Any] = {}
    for name, similarity in similarities.items():
        nearest: dict[str, dict[str, Any]] = {}
        for accession in dead_set:
            scored = [
                (similarity(accession, other), other)
                for other in searchable
                if other != accession
            ]
            score, neighbour = max(scored)
            nearest[accession_curated[accession]] = {
                "neighbour": neighbour,
                "neighbour_gene": accession_curated.get(neighbour, ""),
                "neighbour_entry_name": pool[neighbour].entry_name,
                "neighbour_stratum": stratum_of(neighbour),
                "similarity": float(score),
                "neighbour_is_an_active_kinase": stratum_of(neighbour) in active_strata,
            }
        wrong = sum(
            1 for value in nearest.values() if value["neighbour_is_an_active_kinase"]
        )
        retrieval[name] = {
            "statistic": (
                "for each experimentally dead record, the most similar record in "
                "the whole cohort, and whether that neighbour is an active kinase. "
                "A one-nearest-neighbour baseline transfers its neighbour's label, "
                "so this fraction is the fraction it gets wrong"
            ),
            "n_dead_searched": len(nearest),
            "n_searchable": len(searchable),
            "fraction_whose_nearest_neighbour_is_active": wrong / len(nearest),
            "per_record": nearest,
        }

    # nearest active relative of each dead record, over the whole active pool
    nearest_active = {}
    for accession in dead_set:
        scored = [
            (domain_identity(mappings[accession], mappings[other]), other)
            for other in active_pool
            if other in mappings
        ]
        identity, neighbour = max(scored)
        nearest_active[accession] = {
            "gene": accession_curated[accession],
            "nearest_active": neighbour,
            "kinase_domain_identity": float(identity),
        }
    nearest_dead = {}
    dead_mapped = [a for a in dead_set if a in mappings]
    for accession in active_set:
        scored = [
            (domain_identity(mappings[accession], mappings[other]), other)
            for other in dead_mapped
        ]
        identity, neighbour = max(scored)
        nearest_dead[accession] = {
            "nearest_dead": neighbour,
            "nearest_dead_gene": accession_curated[neighbour],
            "kinase_domain_identity": float(identity),
        }

    annotation_audit = {
        "premise": (
            "the contradiction assumes the corpus annotation sides with 'kinase'. "
            "If Swiss-Prot in fact says the protein is catalytically inactive, the "
            "contradiction is with sequence statistics but not with the annotation, "
            "and a corpus-memorisation account of a model's success is not excluded"
        ),
        "by_stratum": {},
    }
    for name in sorted({stratum_of(a) for a in cohort}):
        members = [a for a in cohort if stratum_of(a) == name]
        stances = defaultdict(int)
        for accession in members:
            stances[annotation_stance(accession)] += 1
        annotation_audit["by_stratum"][name] = {
            "n": len(members),
            "n_with_kinase_ec": sum(int(audits[a]["has_kinase_ec"]) for a in members),
            "n_with_non_iea_kinase_go": sum(
                int(audits[a]["has_non_iea_kinase_go"]) for a in members
            ),
            "n_function_text_flags_inactivity": sum(
                int(audits[a]["function_text_flags_inactivity"]) for a in members
            ),
            "stance": dict(stances),
        }
    wording_hits = {
        pool[a].entry_name: audits[a]["function_text_inactivity_wording"]
        for a in cohort
        if stratum_of(a) in (ACTIVE_MATCHED, ACTIVE_POOL)
        and audits[a]["function_text_flags_inactivity"]
    }
    n_active_members = sum(
        1 for a in cohort if stratum_of(a) in (ACTIVE_MATCHED, ACTIVE_POOL)
    )
    annotation_audit["wording_screen"] = {
        "instrument": "src regex INACTIVITY_WORDING over name + FUNCTION comments",
        "hit_rate_dead_experimental": (
            sum(int(audits[a]["function_text_flags_inactivity"]) for a in dead_set)
            / len(dead_set)
        ),
        "hit_rate_active": len(wording_hits) / n_active_members,
        "active_hits_are_false_positives_to_inspect": wording_hits,
    }
    annotation_audit["dead_by_stance"] = {
        stance: sorted(
            accession_curated[a] for a in dead_set if annotation_stance(a) == stance
        )
        for stance in ("says_kinase", "silent", "says_inactive")
    }

    # --- optional whole-Pfam reading -----------------------------------------
    whole_pfam: dict[str, Any] = {"run": False}
    if args.hmmscan is not None:
        scan_fasta = work / "cohort.fasta"
        write_fasta(scan_fasta, {a: pool[a].sequence for a in cohort})
        scan_tbl = work / "cohort.pfam.domtbl"
        subprocess.run(
            [
                str(args.hmmscan), "--cpu", str(args.cpu), "--cut_ga",
                "--domtblout", str(scan_tbl), "-o", "/dev/null",
                str(args.pfam_a), str(scan_fasta),
            ],
            check=True,
        )
        best_family: dict[str, tuple[float, str]] = {}
        kinase_family_hit: set[str] = set()
        with scan_tbl.open() as handle:
            for line in handle:
                if line.startswith("#"):
                    continue
                fields = line.split()
                family, accession, score = fields[0], fields[3], float(fields[7])
                if family in KINASE_MODELS:
                    kinase_family_hit.add(accession)
                previous = best_family.get(accession)
                if previous is None or score > previous[0]:
                    best_family[accession] = (score, family)
        whole_pfam = {
            "run": True,
            "criterion": "hmmscan --cut_ga against all of Pfam-A",
            "best_family": {a: best_family.get(a, (0.0, ""))[1] for a in cohort},
            # The operative statistic. "Best family" is not it: a receptor
            # pseudokinase with seven immunoglobulin domains has Ig_3 as its
            # top-scoring family and is still called a kinase by Pfam, which is
            # what the contradiction premise needs.
            "n_with_a_kinase_family_above_gathering": len(kinase_family_hit & set(cohort)),
            "dead_without_a_kinase_family_above_gathering": sorted(
                accession_curated[a]
                for a in cohort
                if stratum_of(a) == DEAD_EXPERIMENTAL and a not in kinase_family_hit
            ),
            "n_whose_best_family_is_a_kinase_model": sum(
                1 for a in cohort if best_family.get(a, (0.0, ""))[1] in KINASE_MODELS
            ),
            "n_with_no_family_above_gathering": sum(
                1 for a in cohort if a not in best_family
            ),
        }

    # --- write records --------------------------------------------------------
    reverse_matches = {value: key for key, value in matches.items()}
    split_name = {}
    for index, accession in enumerate(cohort):
        split_name[accession] = "fit" if train_mask[index] else "eval"

    records: list[dict[str, Any]] = []
    for index, accession in enumerate(cohort):
        entry = pool[accession]
        gene = accession_curated.get(accession, "")
        curated = curated_by_gene.get(gene)
        read = readings[accession]
        record = {
            "schema_version": RECORD_SCHEMA_VERSION,
            "accession": accession,
            "entry_name": entry.entry_name,
            "gene": gene,
            "protein_name": entry.protein_name,
            "sequence": entry.sequence,
            "length": len(entry.sequence),
            "stratum": stratum_of(accession),
            "label_provenance": (
                "curated_literature" if curated is not None else "machine_readable_fields"
            ),
            "label_evidence": curated.evidence if curated is not None else (
                "kinase EC number and a non-IEA GO protein-kinase-activity term"
            ),
            "label_citation": curated.citation if curated is not None else "",
            "label_confidence": curated.confidence if curated is not None else "",
            "label_note": curated.note if curated is not None else "",
            "other_catalytic_activity": (
                curated.other_catalytic_activity if curated is not None else ""
            ),
            "matched_partner": matches.get(accession, "") or reverse_matches.get(accession, ""),
            "matched_partner_role": (
                "active_control"
                if accession in matches
                else ("dead_case" if accession in reverse_matches else "")
            ),
            "homology_matched_partner": homology_matches.get(accession, ""),
            # Two different questions, kept apart because PRAG1 answers them
            # differently: Pfam does call it a kinase, and no active kinase scores
            # close enough to control it.
            "statistics_side_calls_it_a_kinase": (
                bool(read["passes_pfam_gathering"]) if read is not None else False
            ),
            "in_matched_contrast": bool(
                accession in matches or accession in matched_active_set
            ),
            "near_duplicate_group": int(groups[index]),
            "split_unit": int(units[index]),
            "split": split_name[accession],
            "annotation": audits[accession],
            "annotation_stance": annotation_stance(accession),
            "kinase_domain": (
                {key: value for key, value in read.items() if key != "_mapping"}
                if read is not None
                else None
            ),
        }
        if accession in nearest_active:
            record["nearest_active_relative"] = nearest_active[accession]
        if accession in nearest_dead:
            record["nearest_dead_relative"] = nearest_dead[accession]
        records.append(record)

    temporary = out_path.with_name(f".{out_path.name}.partial")
    try:
        with temporary.open("w") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        temporary.replace(out_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    stratum_counts = defaultdict(int)
    stratum_units = defaultdict(set)
    for index, accession in enumerate(cohort):
        stratum_counts[stratum_of(accession)] += 1
        stratum_units[stratum_of(accession)].add(int(units[index]))

    pairs_per_side = {"fit": 0, "eval": 0}
    for dead_accession in matches:
        pairs_per_side[split_name[dead_accession]] += 1
    if any(
        split_name[dead_accession] != split_name[active_accession]
        for dead_accession, active_accession in matches.items()
    ):
        raise RuntimeError(
            "a matched pair was divided by the split; the pair is the unit of "
            "independence and merging its near-duplicate groups should have made "
            "that impossible"
        )
    feasibility = {
        "n_matched_pairs": len(matches),
        "bootstrap_floor": MINIMUM_BOOTSTRAP_UNITS,
        "clears_floor_unsplit": len(matches) >= MINIMUM_BOOTSTRAP_UNITS,
        "matched_pairs_per_split_side": pairs_per_side,
        "clears_floor_on_every_split_side": all(
            count >= MINIMUM_BOOTSTRAP_UNITS for count in pairs_per_side.values()
        ),
        "reading_this_supports": (
            "a paired, no-fit contrast -- one score per record, compared within "
            "each matched pair and resampled over pairs -- for which the whole "
            "cohort is the deciding side and the unit count is n_matched_pairs"
        ),
        "reading_this_does_not_support": (
            "a fitted probe reported on a group-disjoint held-out side, which needs "
            "the floor met on the deciding side alone. Splitting the pairs in two "
            "puts roughly half of them on each side"
        ),
    }
    manifest = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "feasibility": feasibility,
        "built_by": "ops/build_pseudokinase_contradiction_set.py",
        "seed": args.seed,
        "inputs": {
            "swissprot_xml": str(args.xml),
            "swissprot_xml_sha256": sha256_of(args.xml),
            "idmapping": str(args.idmapping),
            "pfam_a": str(args.pfam_a),
            "kinase_models": {model: model_meta[model] for model in KINASE_MODELS},
        },
        "output": {
            "records": str(out_path),
            "records_sha256": sha256_of(out_path),
            "n_records": len(records),
        },
        "population": {
            "n_swissprot_entries": collected["n_entries"],
            "n_human_entries": collected["n_human"],
            "n_human_kinase_fold_or_named": len(entries),
            "n_after_length_band": len(pool),
            "length_band": [MIN_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH],
            "n_dropped_by_length": len(length_rejected),
            "curated_genes_dropped_by_length": curated_dropped_by_length,
            "active_pool_size": len(active_pool),
            "active_pool_rejections": dict(active_rejections),
        },
        "strata": {
            name: {
                "n_records": stratum_counts[name],
                "n_split_units": len(stratum_units[name]),
                "clears_bootstrap_floor": len(stratum_units[name]) >= MINIMUM_BOOTSTRAP_UNITS,
            }
            for name in sorted(stratum_counts)
        },
        "matching": {
            "criterion": (
                "same Pfam kinase model, then minimal |bit-score gap| / 10 + "
                "|log length ratio|, assigned 1:1 without replacement"
            ),
            "bit_score_caliper": MATCH_BIT_CALIPER,
            "n_pairs": len(matches),
            "n_dead_eligible": len(dead_eligible),
            "dead_not_eligible": dead_not_eligible,
            "unmatched_dead": unmatched,
            "bit_gap": {
                "median": float(
                    np.median(
                        [
                            abs(
                                readings[d]["domain_bits"] - readings[a]["domain_bits"]
                            )
                            for d, a in matches.items()
                        ]
                    )
                ),
                "max": float(
                    max(
                        abs(readings[d]["domain_bits"] - readings[a]["domain_bits"])
                        for d, a in matches.items()
                    )
                ),
            },
            "could_not_match_on": (
                "kinome group (AGC/CAMK/CMGC/STE/TK/TKL): no classification table "
                "is on disk and this host has no route to fetch one. What matching "
                "on bit score costs instead is family: the realised pairs share a "
                "Pfam model and a score and little else, with a median "
                "kinase-domain identity reported below"
            ),
            "score_matched_pair_identity": {
                "median": float(
                    np.median(
                        [
                            domain_identity(mappings[d], mappings[a])
                            for d, a in matches.items()
                        ]
                    )
                ),
                "max": float(
                    max(
                        domain_identity(mappings[d], mappings[a])
                        for d, a in matches.items()
                    )
                ),
            },
            "homology_matched": {
                "criterion": (
                    "same Pfam kinase model, then maximal kinase-domain identity, "
                    "assigned 1:1 without replacement; no score caliper"
                ),
                "n_pairs": len(homology_matches),
                "identity": {
                    "median": float(np.median(homology_identities)),
                    "min": float(min(homology_identities)),
                    "max": float(max(homology_identities)),
                },
                "pairs": {
                    accession_curated[d]: {
                        "active": a,
                        "active_entry_name": pool[a].entry_name,
                        "kinase_domain_identity": float(
                            domain_identity(mappings[d], mappings[a])
                        ),
                        "bits_dead": readings[d]["domain_bits"],
                        "bits_active": readings[a]["domain_bits"],
                    }
                    for d, a in sorted(homology_matches.items())
                },
            },
            "pairs": {
                accession_curated[dead_accession]: {
                    "dead": dead_accession,
                    "active": active_accession,
                    "active_entry_name": pool[active_accession].entry_name,
                    "bits_dead": readings[dead_accession]["domain_bits"],
                    "bits_active": readings[active_accession]["domain_bits"],
                    "kinase_domain_identity": float(
                        domain_identity(
                            mappings[dead_accession], mappings[active_accession]
                        )
                    ),
                }
                for dead_accession, active_accession in sorted(matches.items())
            },
        },
        "contradiction": {
            "hmm_baseline": hmm_baseline,
            "motif_baseline": motif_baseline,
            "retrieval_baseline": retrieval,
            "nearest_active_relative_identity": {
                "median": float(
                    np.median([v["kinase_domain_identity"] for v in nearest_active.values()])
                ),
                "min": float(
                    min(v["kinase_domain_identity"] for v in nearest_active.values())
                ),
                "max": float(
                    max(v["kinase_domain_identity"] for v in nearest_active.values())
                ),
            },
            "annotation_audit": annotation_audit,
            "active_records_with_degraded_motifs": {
                pool[a].entry_name: {
                    "accession": a,
                    "stratum": stratum_of(a),
                    "motif": "".join(
                        readings[a]["motifs"][name]["motif"] for name in MOTIF_NAMES
                    ),
                    "n_intact": readings[a]["motifs"]["n_intact"],
                    "n_substituted": readings[a]["motifs"]["n_substituted"],
                    "n_unaligned": readings[a]["motifs"]["n_unaligned"],
                    "domain_bits": readings[a]["domain_bits"],
                }
                for a in cohort
                if stratum_of(a) in (ACTIVE_MATCHED, ACTIVE_POOL)
                and readings[a] is not None
                and not readings[a]["motifs"]["all_intact"]
            },
            "active_records_with_degraded_motifs_note": (
                "these are the motif baseline's errors on the active side, and "
                "they are two different things mixed together: divergent but "
                "genuinely active kinases whose beta3 column Pfam leaves unaligned "
                "(n_unaligned > 0), and candidates for pseudokinases this curation "
                "did not declare (n_substituted > 0). PSKH2 reads HRN, the same "
                "catalytic-loop substitution ERBB3 carries, and is the clearest of "
                "the latter"
            ),
            "whole_pfam": whole_pfam,
        },
        "leakage": {
            "grouping": grouping_summary,
            "split": split_summary,
            "n_split_units": n_units,
            "boundary_containment": leakage,
            "kinase_domain_identity_curve": identity_curve,
            "curve_statistic": (
                "per held-out record, the maximum kinase-domain identity to any "
                "record on the fit side; reported as the fraction at or above each "
                "threshold"
            ),
        },
        "limitations": {
            "L-PK-1": (
                "The catalytically-dead label is curated from the experimental "
                "literature by the assistant that wrote this file. It is not a "
                "field of any file on disk and this host cannot reach a citation "
                "database to verify it. Every curated record carries its evidence "
                "kind and citation; a reader who rejects a call can drop it by gene."
            ),
            "L-PK-2": (
                "A motif-aware residue baseline separates the strata almost "
                "perfectly, because the strata are defined by catalysis and the "
                "three catalytic residues are its structural signature. The set is "
                "a contradiction against global sequence statistics, retrieval and "
                "annotation, and not against a baseline that reads the catalytic "
                "columns. Its measured value is in the motif_baseline block."
            ),
            "L-PK-3": (
                "The number of independent units is bounded by the number of human "
                "genes with published catalysis experiments, which is of order "
                "twenty. Records can be multiplied by adding orthologues; units "
                "cannot, because orthologues join the same near-duplicate group."
            ),
            "L-PK-4": (
                "Active controls are selected on a kinase EC number and a non-IEA "
                "GO protein-kinase-activity term. That is curated experimental "
                "support for activity, not an assay this script ran, and an "
                "undeclared pseudokinase carrying both annotations would enter the "
                "active side. Active records whose catalytic motifs are degraded "
                "are written out with their motif reading so they can be inspected."
            ),
            "L-PK-5": (
                "Some dead records are dead kinases inside catalytically active "
                "proteins -- receptor guanylate cyclases, RNase L. They carry "
                "other_catalytic_activity and are in dead_predicted, not in the "
                "positives, but a contrast phrased as enzyme-versus-not would still "
                "be confounded by them."
            ),
            "L-PK-6": (
                "Homology between a pseudokinase and its matched active control is "
                "deliberate and is not leakage; leakage is controlled at the "
                "near-duplicate boundary only, and the residual homology across the "
                "fit/eval boundary is reported as a curve rather than gated. A "
                "homology gate would empty this cohort."
            ),
            "L-PK-8": (
                "The number of matched pairs is small enough that a fit/eval split "
                "leaves each side near or below the bootstrap floor. The set is "
                "built for a paired no-fit contrast; the split is reported so the "
                "leakage curve can be read, not because a fitted probe is powered."
            ),
            "L-PK-9": (
                "Bit-score matching forces the Pfam baseline towards chance on the "
                "matched contrast by construction. The number that says how much "
                "the Pfam score knows on its own is the unmatched AUROC against the "
                "whole active pool, and it is not 0.5."
            ),
            "L-PK-7": (
                "The annotation surface audited is the protein name plus the "
                "FUNCTION comments, because that is exactly what "
                "sequence_description.canonical_description builds this "
                "programme's descriptions from. Other comment types Swiss-Prot "
                "carries -- CAUTION in particular, where some pseudokinase "
                "statements live -- are outside that surface, are not exposed by "
                "the shared XML reader, and are not audited. The audit therefore "
                "measures what a model could have read here, which is the question, "
                "and not everything a curator wrote."
            ),
            "L-PK-10": (
                "The catalytic-residue reading is a reading of three model columns, "
                "and on a divergent kinase the column need not land on the "
                "structurally equivalent residue. Fourteen of the active pool read "
                "as degraded; inspection separates them into columns Pfam left "
                "unaligned (MKNK1, PINK1, the EIF2AK family, the SRPKs, all "
                "genuinely active) and true substitutions (PSKH2 reads HRN, the "
                "substitution ERBB3 carries, and is a candidate pseudokinase this "
                "curation did not declare). Each record carries n_substituted and "
                "n_unaligned separately so the distinction is available downstream."
            ),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"\nwrote {len(records)} records to {out_path}")
    print(f"wrote manifest to {manifest_path}")
    for name in sorted(stratum_counts):
        print(
            f"  {name}: {stratum_counts[name]} records, "
            f"{len(stratum_units[name])} split units"
        )
    print(f"  matched pairs: {len(matches)}")
    print(f"  HMM baseline AUROC (matched): {hmm_baseline['matched']['auc']:.3f}")
    print(f"  motif baseline AUROC (matched): {motif_baseline['matched']['auc']:.3f}")


if __name__ == "__main__":
    main()
