"""D3.k: is a kinase's catalytic machinery something the model reads, or something its family predicts?

**The cohort's contradiction, restated after it was measured.** Pseudokinases carry
the protein-kinase fold, score on the Pfam kinase models, and are experimentally
catalytically dead. EXP-R2-214 assumed the HMM therefore failed on them *by
construction*; amendment 1 (2026-08-19) measured that premise and it is false as
written. The Pfam bit score reaches AUROC 0.770 against the whole active pool and
0.762 [0.695, 0.879] against each pseudokinase's nearest active relative. It only
reaches chance -- 0.511 [0.444, 0.578] -- on the **20-bit-caliper-matched** contrast
over 15 pairs. The caliper is therefore not a convenience: it is what makes a
positive here interpretable rather than a restatement of bit score, and the
caliper-matched contrast is the admitted cohort.

**What the contradiction is with.** Of the 18 experimentally dead records, nine
already say "pseudokinase" or "catalytically inactive" in their own Swiss-Prot
FUNCTION text and only five carry a kinase EC number. The corpus *annotation* is
mostly right. The contradiction is with **sequence statistics and retrieval**, and
that is an admissibility rule rather than a preference: this track is admissible on
sequence-only protein arms and is **not** admissible on a joint language-protein
checkpoint, whose annotation channel carries the answer for exactly the records the
contrast turns on. :func:`refuse_joint_annotation_channel` and
:func:`assert_sequence_only` are that rule in code.

The readout, and why it is this one
===================================

The obvious readouts are both traps.

*Mean likelihood over the kinase domain* -- the readout EXP-R2-214 froze -- asks
whether the model finds a pseudokinase domain less typical of kinases than a
bit-score-matched active one. That is "the model scores kinase-like sequences as
kinase-like" with extra steps, which is the agreement set that closed F10, F12 and
D3.g's stage 35 (audit §7.0 clause 4).

*Reading the residue at the three catalytic columns* is not a model readout at all:
it is the **biology reference**, worth AUROC 0.922 [0.824, 0.989] on this cohort,
and knowing *which* three columns are catalytic is knowledge imported from
biochemistry rather than anything recoverable from co-occurrence. A model readout
that consists of pointing the model at those columns and reading back what it finds
there has measured the pointing.

What this module measures instead is **how the rest of the domain responds when the
catalytic anchors are forced**, with the record's own catalytic residues **erased
from the comparison**:

    rho(x) = [ NLL_rest(x | anchors forced to the kinase-dead state)
             - NLL_rest(x | anchors forced to the live catalytic state) ] / n_rest

Both conditions overwrite the same anchor positions with the same residues, so a
record's observed catalytic state never enters ``rho``. The two forced states are the
classical experimental kinase-dead substitutions -- VAIK lysine K->R, HRD aspartate
D->N, DFG aspartate D->N -- which are conservative in charge and volume, so the
composition and short-fragment displacement is as small as this intervention can be
made. ``NLL_rest`` is read only at domain positions **downstream of the first forced
anchor** and outside a declared radius of every forced anchor, so the quantity is not
the model's marginal preference for K over R at a conserved column but whether that
column's identity **propagates** into how the model reads the rest of the domain.

**What each account predicts, and why they differ.**

*Evolutionary statistics (audit §7.0 clause 1).* A profile HMM emits columns
independently, so forcing a residue at three columns changes nothing at any other
column: its ``rho`` is **exactly zero**, at every record, by construction. A k-order
fragment conditional sees a forced residue in at most ``k-1`` downstream windows, so
its ``rho`` is local, decays to exactly zero beyond an exclusion radius of ``k-1``,
and -- this is the operative half -- is a function of the *local sequence around the
anchors*, which the 20-bit caliper matches between a pseudokinase and its control.
The statistics account therefore predicts ``rho(dead) ~ rho(active)`` and an AUROC at
chance.

*Catalytic knowledge.* An active kinase's domain is a working catalytic machine:
the catalytic loop, the activation segment and the alphaC helix are configured around
a functioning K/D/D triad, so forcing that triad in is coherent with the rest of the
domain and forcing it out is not. A pseudokinase has lost the selection that
maintained that coupling. A model holding the architecture rather than the family
should therefore respond **more** to the forced state in an active kinase than in a
pseudokinase: ``rho(active) > rho(dead)``, and ``-rho`` separates the classes.

Opposite predictions on one contrast is what audit §7.0 clause 4 asks for, and the
three-way read amendment 1 fixes is the recombination ceiling near 0.51, the biology
reference near 0.92, and the model somewhere between.

The second contradiction set, which is not a robustness check
=============================================================

``active_despite_degradation`` (n = 8, exactly at the unit floor) holds kinases that
are experimentally **active despite degraded catalytic machinery**: POMK with none of
the three columns intact; WNK1-4, whose catalytic lysine sits on beta2 so beta3 reads
``VAWC``; CASK, which reads ``GFG`` at the DFG column and phosphorylates neurexin-1
magnesium-independently; HASPIN, a divergent fold; and PKDCC, which is **not**
motif-degraded at all -- it reads ``VALK``/``LLD``/``DLD``, all three intact -- and
instead defeats the HMM channel at 17.9 bits, below Pfam's gathering threshold. So a
motif reader is wrong on **seven** of the eight and the bit score on the eighth.

A model that matches the biology reference on the primary contrast **and also fails
to separate this stratum from the dead one** is reading motifs rather than structure.
That is frozen as its own verdict (:data:`MOTIF_READING`), not as a caveat, because a
robustness check that fails narrows a claim while this one changes what the claim is
about.

What is refused, structurally
=============================

* **A fitted readout reported on a held-out side.** The 15 matched pairs clear
  :data:`~src.transfer.statistics.MINIMUM_BOOTSTRAP_UNITS` only when the whole cohort
  is the deciding side; a 50/50 group-disjoint split gives 8 fit and 7 eval, and 7 is
  below the floor. :func:`refuse_fitted_probe` raises rather than documents it.
* **A joint language-protein checkpoint**, for the annotation-channel reason above.
* **An arm whose scored window is not carried by single-residue tokens**, measured
  through :func:`src.transfer.alphabet_chemistry.symbol_token_coverage` and never
  decided by an arm's name.
* **A verdict on a middling value.** The realised intervals at this n are +/-0.067
  near chance and -0.098/+0.067 near 0.92, so only separations of roughly
  AUROC >= :data:`RESOLVABLE_AUROC` are resolvable. Anything between is returned as
  :data:`NOT_RESOLVABLE`, and the unit count does not improve with effort: it is
  bounded by the number of human genes with published catalysis experiments.

Leakage, and why no homology exclusion is applied
=================================================

Audit §7.0 clause 3 excludes a test item on which a detectable homologue supplies the
answer. Here the answer is catalytic status, which no homologue supplies -- a
pseudokinase's closest relative is an active kinase and vice versa, and that is the
whole reason the set is a contradiction set. The split relation is **near-duplication,
not homology**, matched pairs are merged into one split unit so a pair is never
divided, and the contrast is evaluated **within** a side and never across it. A
homology exclusion would empty the cohort rather than clean it. The cohort manifest
carries the same reasoning as limitation L-PK-6 and this module carries it as
:data:`LEAKAGE_CLAUSE`.

The experimental label is not externally certified
==================================================

EXP-R2-214 required a digest-pinned published experimental compilation under
``external_resources/``. That is not what exists: the catalytically-dead label is
curated from the primary literature by the agent that wrote the cohort build, on a
host with no route to a citation database (manifest limitation L-PK-1). The
per-record ``label_provenance``, ``label_evidence``, ``label_citation`` and
``label_confidence`` fields are the audit trail. Every artefact this module produces
carries that deviation in its ``limitations`` block; it is not repaired by anything
here.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from src.transfer.alphabet_chemistry import (
    CEILING_ADEQUACY_FLOOR,
    FRAGMENT_ORDERS,
    MINIMUM_SYMBOL_TOKEN_COVERAGE,
    OrderedFragmentCounts,
    PRE_REGISTERED_FRAGMENT_ORDER,
    admit_arm,
)
from src.transfer.arms import AA20
from src.transfer.statistics import (
    MINIMUM_BOOTSTRAP_UNITS,
    bootstrap_unit_floor,
    paired_group_bootstrap,
)

# --------------------------------------------------------------- declarations

#: The register entry that froze this design, the track inside it, and the
#: amendments this module implements. Declared rather than implied: an artefact
#: produced under the amended design and one produced under the frozen text are
#: indistinguishable to a reader who is not told which.
PRE_REGISTRATION = "EXP-R2-214"
PRE_REGISTRATION_TRACK = "D3.k, pseudokinases"
PRE_REGISTRATION_AMENDMENTS: tuple[str, ...] = (
    "amendment 1 (D3.k), 2026-08-19: the HMM contradiction is partial and the "
    "admitted cohort is the 20-bit-caliper-matched contrast (item 1); the corpus "
    "annotation is mostly right, so sequence-only protein arms are admissible and a "
    "joint checkpoint is not (item 2); the motif reader is the biology-side reference "
    "and not a ceiling row (item 3); active_despite_degradation is a second, "
    "orthogonal contradiction set with its own verdict (item 4); the readout must be "
    "no-fit and only AUROC separations of about 0.75 are resolvable (item 5); the "
    "experimental label is curated rather than externally certified (item 6)"
)

#: What clearing anything here buys, stated so an artefact cannot be read as more.
PRE_REGISTRATION_SCOPE = (
    "EXP-R2-214 registers D3.k and audit §7.0 admits it; amendment 1 of 2026-08-19 "
    "corrects three of its premises and this module implements the amended design. "
    "Naming a pre-registration is not admission of a result: no number produced here "
    "may be cited as a programme finding until the run is recorded in "
    "docs/EXPERIMENT_LOG.md, and clearing the ceiling licenses a candidate and "
    "nothing more -- §8's causal, retrieval-aware and independent-biological clauses "
    "remain open (§7.0 clause 5)."
)

#: The intervention's name. Part of every artefact basename, so a successor that
#: forces something else cannot be mistaken for this one.
INTERVENTION = "catalytic_anchor_state_forcing"

#: Inference dtype, fixed rather than exposed: Appendix B rule 15b requires float32
#: for a difference of order 0.01-0.1 nats, and rho is one.
DTYPE = "float32"

SCHEMA_VERSION = "r2_transfer_catalytic_contradiction_v1"

#: The three catalytic columns, in sequence order along the kinase domain, as the
#: cohort build names them.
CATALYTIC_ANCHORS: tuple[str, ...] = ("vaik_lys", "hrd_asp", "dfg_asp")

#: The live catalytic state: the residues a competent protein kinase carries at the
#: three columns.
LIVE_STATE: dict[str, str] = {"vaik_lys": "K", "hrd_asp": "D", "dfg_asp": "D"}

#: The kinase-dead state. These are the classical experimental kinase-dead
#: substitutions rather than residues chosen here, which matters twice: they are what
#: an experimentalist means by "catalytically dead", and each is conservative in
#: charge and volume, so the composition and short-fragment displacement the
#: recombination ceiling can exploit is as small as this intervention can be made.
DEAD_STATE: dict[str, str] = {"vaik_lys": "R", "hrd_asp": "N", "dfg_asp": "N"}

STATE_SOURCE = (
    "the classical experimental kinase-dead substitutions: VAIK lysine K->R (PKA "
    "K72R, Gibbs and Zoller 1991), HRD aspartate D->N (PKA D166N, Gibbs and Zoller "
    "1991) and DFG aspartate D->N (PKA D184N). Declared here rather than derived from "
    "the cohort, so the intervention is not a function of the labels it is read "
    "against"
)

#: Exclusion radii swept around every forced anchor, in residues. The verdict is read
#: at the declared radius and the sweep is reported beside it (Appendix B rule 17).
#: The radius is what separates a local corpus effect from propagation: a k-order
#: fragment conditional's rho is exactly zero beyond radius k-1, so the ceiling curve
#: is read over (order, radius) and the verdict at the **binding** cell -- the most
#: demanding rung -- never the friendliest. That is amendment 3's rule for D3.j
#: applied to this design's own second axis.
EXCLUSION_RADII: tuple[int, ...] = (0, 3, 6, 12)

#: Random-anchor control draws. Rule 39 in this design's own terms: the control buys
#: **sites**, not positions -- the identical forced-state pattern is applied at a
#: shifted copy of the record's own anchor triple, which holds the number of anchors,
#: their spacing and the scored-set shape fixed and moves only *where* they sit. An
#: effect that survives the shift is specific to the catalytic columns; one that does
#: not is a response to conservative substitution anywhere in a kinase domain.
MINIMUM_RANDOM_ANCHOR_DRAWS = 8

#: Smallest displacement of the shifted anchor triple, in residues. Below this the
#: shifted anchors fall inside the real ones' exclusion window and the control stops
#: being a control.
MINIMUM_ANCHOR_SHIFT = 4

#: The AUROC a separation must reach before this cohort may call it. Amendment 1 item
#: 5: the realised intervals are +/-0.067 at 0.511 and -0.098/+0.067 at 0.922, so only
#: separations of roughly this size are resolvable at n = 15 -- adequate here only
#: because the two hypotheses predict 0.51 and 0.92, and inadequate for a subtler
#: contrast. The unit count does not improve with effort (manifest limitation L-PK-3).
RESOLVABLE_AUROC = 0.75

#: Multiples of the ceiling's excess over chance the margin is swept at, with the verdict
#: read at the declared one (Appendix B rule 17). 2.0 is the value in force for D3.g.
MARGIN_FACTOR_SWEEP: tuple[float, ...] = (1.0, 1.5, 2.0)

#: The one deciding side this design admits.
WHOLE_COHORT = "whole_cohort"
SPLIT_SIDES: tuple[str, ...] = (WHOLE_COHORT, "fit", "eval")

#: Renderings that put a natural-language or annotation channel in front of the
#: sequence. Refused here rather than run and discounted, for amendment 1 item 2's
#: reason: nine of the eighteen dead records state their own inactivity in the text a
#: joint checkpoint would read, and ZymCTRL's EC tag names protein-kinase activity for
#: 15 of 15 matched actives against 5 of 18 dead.
ANNOTATION_BEARING_INPUT_FORMATS = frozenset({"ec_conditioned"})

#: The eight moderate-confidence dead records, by gene symbol, so a sensitivity read
#: is possible without rebuilding the cohort. Declared here **and checked against the
#: cohort's own label_confidence field** at load: the declaration exists so a reader
#: sees the eight names, and the check exists so the declaration cannot drift.
MODERATE_CONFIDENCE_GENES: tuple[str, ...] = (
    "EPHA10", "EPHB6", "IRAK2", "PAN3", "PEAK3", "STRADB", "TRIB3", "ULK4",
)

#: The contested records, held out of the positives by the cohort build rather than
#: by this module. Named so a reader can see they are absent on purpose.
CONTESTED_GENES: tuple[str, ...] = (
    "BUB1B", "ERBB3", "ILK", "KSR2", "PIK3R4", "ROR1", "ROR2", "TRIB2",
)

DEAD_STRATUM = "dead_experimental"
ACTIVE_STRATUM = "active_matched"
COUNTER_STRATUM = "active_despite_degradation"

LEAKAGE_CLAUSE = (
    "No homology exclusion is applied and the omission is deliberate. Audit §7.0 "
    "clause 3 excludes a test item on which a detectable homologue supplies the "
    "answer; here the answer is catalytic status, which no homologue supplies -- a "
    "pseudokinase's closest relative is an active kinase and vice versa, which is the "
    "whole reason the set is a contradiction set, and the matched control is "
    "deliberately the closest admissible relative. The split relation is "
    "near-duplication rather than homology: 439 near-duplicate groups over 461 "
    "records at shingle length 5 and containment 0.5, matched pairs merged into one "
    "split unit so a pair is never divided, maximum boundary containment 0.4905 over "
    "230 held-out records. The contrast is evaluated WITHIN a side and never across "
    "it, so homology between a pseudokinase and its own control is a property of the "
    "contrast rather than leakage. A reader who sees 58.1% of held-out records "
    "keeping a >=0.5 relative and no homology filter would otherwise conclude the "
    "clause was waived. Manifest limitation L-PK-6 states the same."
)

#: Verdicts on the primary contrast.
CLEARS_TOWARD_EXPERIMENT = "CLEARS_CEILING_TOWARD_EXPERIMENT"
CLEARS_REVERSED = "CLEARS_CEILING_REVERSED"
RECOMBINATION = "RECOMBINATION"
NOT_RESOLVABLE = "NOT_RESOLVABLE_AT_THIS_N"
READOUT_DEGENERATE = "READOUT_DEGENERATE"
VOID_READOUT = "VOID_READOUT"
#: The model exceeds every ceiling row on the difference half of §7.0 clause 2, and fails
#: only the multiple half against a row whose bar no AUROC can reach. That is a statement
#: about the bar and the cohort, not about the model, and it is Appendix B rule 2 applied
#: to the admission rule itself.
MARGIN_UNATTAINABLE = "MARGIN_UNATTAINABLE"
PRIMARY_VERDICTS: tuple[str, ...] = (
    CLEARS_TOWARD_EXPERIMENT, CLEARS_REVERSED, RECOMBINATION, MARGIN_UNATTAINABLE,
    NOT_RESOLVABLE, READOUT_DEGENERATE, VOID_READOUT,
)

#: Verdicts on the counter-stratum, read as a second contradiction set in its own
#: right rather than as a robustness check on the first.
SEPARATES_COUNTER = "SEPARATES_COUNTER_STRATUM"
DOES_NOT_SEPARATE_COUNTER = "DOES_NOT_SEPARATE_COUNTER_STRATUM"
COUNTER_REVERSED = "COUNTER_STRATUM_REVERSED"
COUNTER_VERDICTS: tuple[str, ...] = (
    SEPARATES_COUNTER, DOES_NOT_SEPARATE_COUNTER, COUNTER_REVERSED,
    READOUT_DEGENERATE, VOID_READOUT,
)

#: The combined reading. ``MOTIF_READING`` is the one amendment 1 item 4 freezes: a
#: model that matches the biology reference on the primary contrast and fails the
#: counter-stratum is reading motifs rather than structure, and must be reported as
#: such rather than as a partial pass.
CANDIDATE_KNOWLEDGE = "CANDIDATE_CATALYTIC_KNOWLEDGE"
MOTIF_READING = "MOTIF_READING_NOT_STRUCTURE"
COMBINED_VERDICTS: tuple[str, ...] = (
    CANDIDATE_KNOWLEDGE, MOTIF_READING, RECOMBINATION, MARGIN_UNATTAINABLE,
    CLEARS_REVERSED, NOT_RESOLVABLE, READOUT_DEGENERATE, VOID_READOUT,
)


# ------------------------------------------------------------------- refusals


def refuse_fitted_probe(split: str, *, fit_units: int, eval_units: int) -> None:
    """Amendment 1 item 5, enforced rather than documented.

    The 15 matched pairs clear the shared unit floor **only when the whole cohort is
    the deciding side**. A 50/50 group-disjoint split gives 8 fit and 7 eval, and 7 is
    below the floor, so a fitted readout reported on a held-out side is not powered.
    This design's readout fits nothing, which is why the restriction changes no
    statistic -- but it converts an incidental property into a stated refusal and
    forecloses the adaptation a later reader would otherwise reach for.
    """

    if split not in SPLIT_SIDES:
        raise ValueError(f"unknown deciding side {split!r}; declared: {list(SPLIT_SIDES)}")
    if split == WHOLE_COHORT:
        return
    side = fit_units if split == "fit" else eval_units
    floor = bootstrap_unit_floor(int(side))
    raise ValueError(
        f"a readout decided on the {split!r} side is refused (EXP-R2-214 D3.k "
        f"amendment 1, item 5). The cohort splits {fit_units} fit / {eval_units} eval "
        f"matched pairs and this side carries {side}, against the "
        f"{floor['minimum_units']}-unit floor. The pairs clear the floor only when the "
        "whole cohort is the deciding side, so a fitted probe reported on a held-out "
        "side is not powered; this design's readout fits nothing and is decided on "
        f"{WHOLE_COHORT!r}"
    )


def refuse_joint_annotation_channel(name: str) -> None:
    """Amendment 1 item 2: D3.k is not admissible on a joint language-protein model.

    Always raises. The reason is measured rather than stylistic: the Swiss-Prot name
    and FUNCTION text already state inactivity for 9 of the 18 experimentally dead
    records and are silent for 4, and only 5 of 18 carry a kinase EC number against 15
    of 15 for the matched actives. A checkpoint that reads an annotation channel is
    handed the answer for exactly the records the contrast turns on, so it is refused
    before it is loaded rather than run and discounted afterwards.
    """

    raise ValueError(
        f"{name!r} is a joint language-protein rendering and D3.k is not admissible on "
        "one without description masking first, in the shape C34-1 specifies for D3.g "
        "(EXP-R2-214 amendment 1, item 2). Its annotation channel carries the answer: "
        "the Swiss-Prot FUNCTION text states inactivity for 9 of the 18 experimentally "
        "dead records and only 5 of 18 carry a kinase EC number, against 15 of 15 for "
        "the matched actives. The contradiction this cohort carries is with sequence "
        "statistics and retrieval, not with the annotation, so it is read on "
        "sequence-only protein arms"
    )


def assert_sequence_only(spec: Any) -> dict[str, Any]:
    """Refuse an arm that reads anything but residues, and say which channel it reads.

    Two refusals, in order. A text arm has no protein cohort to read at all. A protein
    arm whose rendering prefixes an annotation -- ZymCTRL's EC tag is the panel's one
    instance -- is refused for :func:`refuse_joint_annotation_channel`'s reason at a
    smaller scale: EC 2.7.11.- names protein-kinase activity for every matched active
    and for five of eighteen dead records, so the tag is a partial label.
    """

    if getattr(spec, "modality", None) != "protein":
        raise ValueError(
            f"{spec.name!r} is a {spec.modality!r} arm; D3.k reads residue sequences and "
            "has no text cohort. A text arm enters this design nowhere"
        )
    if spec.input_format in ANNOTATION_BEARING_INPUT_FORMATS:
        raise ValueError(
            f"{spec.name!r} renders its input as {spec.input_format!r}, which prefixes an "
            "annotation the model reads before the sequence. EXP-R2-214 amendment 1 item "
            "2 admits D3.k on sequence-only protein arms only: an EC tag names "
            "protein-kinase activity for 15 of 15 matched actives and for 5 of 18 dead "
            "records, so conditioning on it hands the model a partial label for exactly "
            "the contrast being measured"
        )
    return {
        "arm": spec.name,
        "modality": spec.modality,
        "input_format": spec.input_format,
        "reads_annotation_channel": False,
        "reason": (
            "a protein arm whose rendering carries no annotation prefix; the model reads "
            "residues and a direction or format marker and nothing that names catalysis"
        ),
    }


# --------------------------------------------------------------------- cohort


@dataclass(frozen=True)
class Record:
    """One cohort record, reduced to what this readout needs."""

    accession: str
    gene: str
    entry_name: str
    stratum: str
    sequence: str
    domain_from: int
    domain_to: int
    anchor_names: tuple[str, ...]
    anchor_positions: tuple[int, ...]
    observed_anchor_residues: tuple[str, ...]
    n_intact: int
    domain_bits: float | None
    split_unit: int
    label_confidence: str
    annotation_stance: str
    matched_partner: str

    @property
    def label(self) -> str:
        return self.gene or self.entry_name or self.accession

    @property
    def n_anchors(self) -> int:
        return len(self.anchor_positions)


@dataclass(frozen=True)
class MatchedPair:
    """One 20-bit-caliper-matched pair, which is one resampling unit."""

    dead: Record
    active: Record

    @property
    def split_unit(self) -> int:
        return self.dead.split_unit

    @property
    def label(self) -> str:
        return self.dead.label


@dataclass(frozen=True)
class Cohort:
    """The frozen contradiction set, with the checks that make it citable."""

    path: Path
    sha256: str
    records: tuple[Record, ...]
    manifest: dict[str, Any]

    def by_stratum(self, stratum: str) -> tuple[Record, ...]:
        return tuple(record for record in self.records if record.stratum == stratum)

    def record(self, accession: str) -> Record:
        for candidate in self.records:
            if candidate.accession == accession:
                return candidate
        raise KeyError(f"{accession} is not in this cohort")


def _anchor_reading(motifs: Mapping[str, Any]) -> tuple[
    tuple[str, ...], tuple[int, ...], tuple[str, ...], int
]:
    names: list[str] = []
    positions: list[int] = []
    residues: list[str] = []
    intact = 0
    for name in CATALYTIC_ANCHORS:
        entry = motifs.get(name)
        if not isinstance(entry, Mapping):
            continue
        if entry.get("intact"):
            intact += 1
        position = entry.get("position")
        if position is None:
            continue
        names.append(name)
        positions.append(int(position))
        residues.append(str(entry["residue"]))
    order = sorted(range(len(positions)), key=lambda index: positions[index])
    return (
        tuple(names[index] for index in order),
        tuple(positions[index] for index in order),
        tuple(residues[index] for index in order),
        intact,
    )


def load_cohort(path: Path, *, sha256: str) -> Cohort:
    """Read the frozen cohort, pinned by digest, and re-derive what it claims.

    The digest is a required argument rather than a check against the manifest's own
    copy: a cohort that has been rebuilt under the same filename is a different
    measurement, and the pin is what makes an artefact quoting this stage identify the
    records it was read on. Everything else here is re-derived rather than trusted --
    the anchor positions are checked against the sequence they index, the declared
    moderate-confidence and contested gene lists against the cohort's own fields, and
    the matched pairing against both members' ``matched_partner``.
    """

    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha256:
        raise ValueError(
            f"{path} hashes {digest}, and --cohort-sha256 pins {sha256}. A cohort "
            "quoted by an artefact has to be the one that was read"
        )
    manifest_path = path.with_name(path.name + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("output", {}).get("records_sha256") != digest:
        raise ValueError(
            f"{manifest_path} names records_sha256 "
            f"{manifest.get('output', {}).get('records_sha256')!r}, not {digest}; the "
            "manifest describes a different build than the records beside it"
        )
    records: list[Record] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        domain = payload.get("kinase_domain") or {}
        motifs = domain.get("motifs") or {}
        names, positions, residues, intact = _anchor_reading(motifs)
        sequence = payload["sequence"]
        for position, residue in zip(positions, residues):
            if sequence[position - 1] != residue:
                raise ValueError(
                    f"{payload['accession']}: the cohort places {residue!r} at position "
                    f"{position} and the sequence carries {sequence[position - 1]!r}; the "
                    "anchor indexing does not describe this record"
                )
        records.append(
            Record(
                accession=payload["accession"],
                gene=payload.get("gene", ""),
                entry_name=payload.get("entry_name", ""),
                stratum=payload["stratum"],
                sequence=sequence,
                domain_from=int(domain["domain_from"]) if domain.get("domain_from") else 0,
                domain_to=int(domain["domain_to"]) if domain.get("domain_to") else 0,
                anchor_names=names,
                anchor_positions=positions,
                observed_anchor_residues=residues,
                n_intact=intact,
                domain_bits=(
                    float(domain["domain_bits"]) if domain.get("domain_bits") is not None else None
                ),
                split_unit=int(payload["split_unit"]),
                label_confidence=payload.get("label_confidence", ""),
                annotation_stance=payload.get("annotation_stance", ""),
                matched_partner=payload.get("matched_partner", ""),
            )
        )
    cohort = Cohort(path=path, sha256=digest, records=tuple(records), manifest=manifest)
    _check_declared_gene_lists(cohort)
    return cohort


def _check_declared_gene_lists(cohort: Cohort) -> None:
    """The module's two declared gene lists against the cohort's own fields."""

    moderate = tuple(
        sorted(
            record.label
            for record in cohort.by_stratum(DEAD_STRATUM)
            if record.label_confidence == "moderate"
        )
    )
    if moderate != tuple(sorted(MODERATE_CONFIDENCE_GENES)):
        raise ValueError(
            f"the cohort's moderate-confidence dead records are {list(moderate)} and this "
            f"module declares {list(MODERATE_CONFIDENCE_GENES)}. One of the two is stale, "
            "and a sensitivity read that drops the wrong eight is worse than none"
        )
    contested = tuple(sorted(record.label for record in cohort.by_stratum("contested")))
    if contested != tuple(sorted(CONTESTED_GENES)):
        raise ValueError(
            f"the cohort's contested records are {list(contested)} and this module "
            f"declares {list(CONTESTED_GENES)}"
        )


def matched_pairs(cohort: Cohort, *, high_confidence_only: bool = False) -> tuple[MatchedPair, ...]:
    """The caliper-matched pairs, each one resampling unit.

    ``high_confidence_only`` drops the moderate-confidence dead records by gene symbol,
    which is amendment 1's sensitivity read. It is a filter over the frozen cohort and
    never a rebuild, and both readings are reported side by side so no flag can select
    the friendlier one.
    """

    dead = {record.accession: record for record in cohort.by_stratum(DEAD_STRATUM)}
    active = {record.accession: record for record in cohort.by_stratum(ACTIVE_STRATUM)}
    pairs: list[MatchedPair] = []
    for accession, record in sorted(dead.items()):
        partner = record.matched_partner
        if not partner:
            continue
        control = active.get(partner)
        if control is None:
            raise ValueError(
                f"{record.label} names matched partner {partner!r}, which is not in the "
                f"{ACTIVE_STRATUM} stratum"
            )
        if control.matched_partner != accession:
            raise ValueError(
                f"{record.label} and {control.label} do not name each other as partners"
            )
        if control.split_unit != record.split_unit:
            raise ValueError(
                f"{record.label} and its control sit in split units {record.split_unit} "
                f"and {control.split_unit}; a matched pair must be one split unit or the "
                "contrast can be divided by a split"
            )
        if high_confidence_only and record.label_confidence != "high":
            continue
        pairs.append(MatchedPair(dead=record, active=control))
    return tuple(pairs)


# -------------------------------------------------------------- the operation


def window_bounds(record: Record, *, max_residues: int) -> tuple[int, int]:
    """The residue window scored, 1-based inclusive.

    The window **ends at the kinase domain's last residue** and extends upstream. Both
    halves are deliberate. Nothing downstream of the domain can change a scored
    position's likelihood under an autoregressive model or under a fragment
    conditional, so including it would buy nothing and cost tokens; and the upstream
    context is what the model conditions the domain on, so it is kept up to the
    declared cap. A record whose sequence is longer than the cap is truncated at its N
    terminus, which is off-distribution -- but identically so in both forced
    conditions, and rho is a within-record difference, so the truncation cancels.
    """

    if max_residues < 1:
        raise ValueError("max_residues must be positive")
    if not record.domain_to:
        raise ValueError(f"{record.label} carries no kinase domain span")
    end = min(record.domain_to, len(record.sequence))
    start = max(1, end - max_residues + 1)
    return start, end


def scored_positions(
    record: Record, anchors: Sequence[int], *, radius: int, window: tuple[int, int]
) -> tuple[int, ...]:
    """Domain positions whose likelihood rho is read at, 1-based.

    Three conditions, each doing work. **Downstream of the first forced anchor**: an
    autoregressive model's likelihood at a position upstream of every intervention is
    bit-identical between the two conditions, so those positions carry no signal and
    would only dilute -- and the fact that they are identical is checked as a write
    invariant rather than assumed. **Outside ``radius`` of every anchor**: this is the
    axis that separates a local corpus effect from propagation, because a k-order
    fragment conditional's response is exactly zero beyond radius k-1. **Inside the
    domain and the window**: the estimand is the kinase domain, not the protein.
    """

    if radius < 0:
        raise ValueError("the exclusion radius is non-negative")
    if not anchors:
        raise ValueError(f"{record.label} has no forced anchor, so rho is undefined")
    start, end = window
    first = min(anchors)
    excluded = {
        position
        for anchor in anchors
        for position in range(anchor - radius, anchor + radius + 1)
    }
    low = max(first + 1, record.domain_from, start)
    high = min(record.domain_to, end)
    return tuple(position for position in range(low, high + 1) if position not in excluded)


def forced_windows(
    record: Record, anchors: Sequence[int], anchor_names: Sequence[str], *, window: tuple[int, int]
) -> dict[str, str]:
    """The two windows rho is the difference of, plus the record's own window.

    Both forced windows overwrite **the same positions with the same residues**, so the
    record's observed catalytic state cancels out of rho entirely. That is the property
    which stops this readout from being the motif reader implemented through a model,
    and it is asserted in the tests rather than only claimed here.
    """

    start, end = window
    if len(anchors) != len(anchor_names):
        raise ValueError("every forced anchor needs a declared column name")
    text = list(record.sequence[start - 1 : end])
    for position in anchors:
        if not start <= position <= end:
            raise ValueError(
                f"{record.label}: forced anchor at {position} is outside the scored "
                f"window {window}"
            )
    live = list(text)
    dead = list(text)
    for position, name in zip(anchors, anchor_names):
        live[position - start] = LIVE_STATE[name]
        dead[position - start] = DEAD_STATE[name]
    return {"observed": "".join(text), "live": "".join(live), "dead": "".join(dead)}


def anchor_shifts(
    record: Record, *, draws: int, radius: int, seed: int
) -> tuple[tuple[int, ...], dict[str, Any]]:
    """Displacements of the record's own anchor triple, for the site-specificity control.

    The triple is moved as a rigid body, so the number of anchors, their spacing and
    the shape of the scored set are held fixed and only the *site* moves. A free draw
    of three positions would change all four at once and the control would price the
    wrong thing.
    """

    if draws < MINIMUM_RANDOM_ANCHOR_DRAWS:
        raise ValueError(
            f"{draws} shifted-site draws is below the declared "
            f"{MINIMUM_RANDOM_ANCHOR_DRAWS}; a 95th percentile over fewer is one of a "
            "handful of order statistics"
        )
    anchors = record.anchor_positions
    low = record.domain_from - min(anchors)
    high = record.domain_to - max(anchors)
    candidates = [
        shift
        for shift in range(low, high + 1)
        if abs(shift) >= max(MINIMUM_ANCHOR_SHIFT, radius + 1)
    ]
    record_block = {
        "n_candidate_shifts": len(candidates),
        "shift_range": [int(low), int(high)],
        "minimum_shift": int(max(MINIMUM_ANCHOR_SHIFT, radius + 1)),
    }
    if len(candidates) < draws:
        record_block["admitted"] = False
        record_block["reason"] = (
            f"the domain admits {len(candidates)} displacements of the anchor triple at "
            f"this radius and {draws} were asked for; this record carries no "
            "site-specificity control rather than a control over repeated sites"
        )
        return (), record_block
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.asarray(candidates), size=draws, replace=False)
    record_block["admitted"] = True
    return tuple(int(value) for value in sorted(chosen)), record_block


def shuffled_control(record: Record, *, window: tuple[int, int], seed: int) -> str:
    """D3.k-A1's reachability arm: the same domain with its non-anchor residues permuted.

    A shuffled kinase domain keeps composition exactly and destroys the architecture, so
    a readout that measures whether the catalytic state is *coherent with the rest of the
    domain* must separate a real domain from its own shuffle. The anchors themselves are
    held in place, because the operation must remain defined at the same positions.
    Failing this check voids the readout as a specification defect and the contradiction
    contrast is **not measured** -- a null must not be producible by a broken instrument
    (Appendix B rule 40).
    """

    start, end = window
    text = list(record.sequence[start - 1 : end])
    held = {position - start for position in record.anchor_positions if start <= position <= end}
    movable = [index for index in range(len(text)) if index not in held]
    rng = np.random.default_rng(seed)
    permuted = rng.permutation(np.asarray(movable))
    shuffled = list(text)
    for source, destination in zip(movable, permuted):
        shuffled[destination] = text[source]
    return "".join(shuffled)


# ------------------------------------------------------------- the two scorers


class FragmentLikelihood:
    """The recombination ceiling, as a sequence likelihood at one fragment order.

    ``P(x_i | x_{i-k+1..i-1})`` from the pinned UniRef50 counts, as a plug-in maximum
    likelihood estimate with **no smoothing**, which this design does not have to
    choose: a position is scored only where both the numerator k-gram and its context
    are observed, and the covered fraction is reported per order.

    **k = 1 is the curve's own reachability anchor and is exactly zero.** A unigram
    reads no context, so forcing a residue at an anchor cannot change the likelihood at
    any other position: ``rho`` is 0.00000 for every record by construction. Every
    higher order shares the same indexing, so a curve whose first point is not exactly
    zero is an indexing defect caught before any verdict is read.
    """

    def __init__(self, ordered: OrderedFragmentCounts) -> None:
        self.ordered = ordered
        self.order = int(ordered.order)
        if self.order < 1:
            raise ValueError("a conditional needs a positive order")
        self.counts = ordered.counts
        self._powers = (len(AA20) ** np.arange(self.order - 1, -1, -1)).astype(np.int64)

    @property
    def name(self) -> str:
        return f"uniref50_fragment_k{self.order}"

    def _totals(self, contexts: np.ndarray) -> np.ndarray:
        unique, inverse = np.unique(contexts, return_inverse=True)
        block = unique[:, None] * len(AA20) + np.arange(len(AA20), dtype=np.int64)
        summed = np.asarray(self.counts[block.reshape(-1)]).reshape(block.shape).sum(axis=1)
        return summed[inverse]

    def nll(self, windows: Sequence[str], positions: Sequence[Sequence[int]]) -> list[np.ndarray]:
        """Per-position negative log-likelihood, ``nan`` where the corpus is silent.

        Returned per position rather than summed so that rho can be taken over the
        positions **both** forced conditions can score. At k = 7 the staged corpus
        leaves 14% of its k-mers unobserved, and a sum over two different position sets
        is not a difference of likelihoods.
        """

        rows: list[np.ndarray] = []
        for window, wanted in zip(windows, positions):
            codes = np.asarray([AA20.find(character) for character in window], dtype=np.int64)
            if (codes < 0).any():
                raise ValueError("the fragment ceiling reads canonical residues only")
            values = np.full(len(wanted), np.nan, dtype=np.float64)
            index = np.asarray(wanted, dtype=np.int64)
            conditioned = index >= self.order - 1
            if conditioned.any():
                selected = index[conditioned]
                offsets = np.arange(-(self.order - 1), 1, dtype=np.int64)
                grams = codes[selected[:, None] + offsets[None, :]] @ self._powers
                numerator = np.asarray(self.counts[grams], dtype=np.float64)
                denominator = self._totals(grams // len(AA20)).astype(np.float64)
                usable = (numerator > 0) & (denominator > 0)
                filled = np.full(selected.size, np.nan, dtype=np.float64)
                filled[usable] = -np.log(numerator[usable] / denominator[usable])
                values[np.flatnonzero(conditioned)] = filled
            rows.append(values)
        return rows


class ArmLikelihood:
    """One panel arm's per-residue next-token likelihood over a residue window.

    The window is rendered through :meth:`src.transfer.arms.Cohort.input_strings`, so the
    format an arm was trained on is decided once, in the panel declaration, and not a
    second time here. The residue-to-token map is then **verified rather than assumed**:
    every token is decoded, the single-residue tokens are collected in order, and their
    concatenation must equal the window exactly. An arm that spells a window any other
    way raises here rather than producing a plausible number at the wrong positions.
    """

    def __init__(self, arm: Any, *, batch_size: int) -> None:
        import torch

        self.arm = arm
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        self._torch = torch
        self.forward_tokens = 0
        self.forward_calls = 0

    def _rendered(self, windows: Sequence[str]) -> list[str]:
        from src.transfer.arms import Cohort

        cohort = Cohort(
            name="catalytic_contradiction_window",
            kind="protein",
            records=list(windows),
            min_symbols=min(len(window) for window in windows),
            max_symbols=max(len(window) for window in windows),
        )
        return cohort.input_strings(self.arm)

    def _residue_token_index(self, ids: Sequence[int], window: str) -> list[int]:
        tokenizer = self.arm.tokenizer
        index: list[int] = []
        spelled: list[str] = []
        for column, token_id in enumerate(ids):
            piece = tokenizer.convert_ids_to_tokens(int(token_id))
            if piece is None:
                continue
            decoded = tokenizer.convert_tokens_to_string([piece])
            if len(decoded) == 1 and decoded in AA20:
                index.append(column)
                spelled.append(decoded)
        if "".join(spelled) != window:
            raise ValueError(
                f"{self.arm.name}: the rendered window tokenises to "
                f"{len(spelled)} single-residue tokens spelling a string that is not the "
                f"{len(window)}-residue window. The residue-to-token map is undefined on "
                "this arm and no position-level quantity may be read from it"
            )
        if index and index[0] == 0:
            raise ValueError(
                f"{self.arm.name}: the first residue of the window is the first token, so "
                "its likelihood is not conditioned and the rendering carries no prefix"
            )
        return index

    def nll(self, windows: Sequence[str], positions: Sequence[Sequence[int]]) -> list[np.ndarray]:
        """Per-residue negative log-likelihood at the requested window positions."""

        torch = self._torch
        from src.transfer.arms import tokenize_batch

        rows: list[np.ndarray] = []
        for start in range(0, len(windows), self.batch_size):
            block = list(windows[start : start + self.batch_size])
            wanted = list(positions[start : start + self.batch_size])
            rendered = self._rendered(block)
            ids, mask = tokenize_batch(self.arm, rendered, 1_000_000)
            ids = ids.to(self.arm.device)
            mask = mask.to(self.arm.device)
            self.forward_tokens += int(mask.sum())
            self.forward_calls += 1
            with torch.no_grad():
                logits = self.arm.model(input_ids=ids, attention_mask=mask).logits
            logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
            targets = ids[:, 1:]
            token_nll = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            for row, (window, wanted_positions) in enumerate(zip(block, wanted)):
                kept = [
                    int(value)
                    for value, keep in zip(ids[row].tolist(), mask[row].tolist())
                    if keep
                ]
                columns = self._residue_token_index(kept, window)
                selected = [columns[position] - 1 for position in wanted_positions]
                if any(column < 0 for column in selected):
                    raise ValueError("a scored residue has no conditioning position")
                values = token_nll[row, torch.tensor(selected, device=token_nll.device)]
                rows.append(np.asarray(values.detach().cpu().numpy(), dtype=np.float64))
        return rows

    def cost(self) -> dict[str, Any]:
        return {
            "forward_calls": int(self.forward_calls),
            "forward_tokens": int(self.forward_tokens),
            "batch_size": int(self.batch_size),
            "dtype": DTYPE,
        }


# --------------------------------------------------------------- the estimand


@dataclass(frozen=True)
class SiteRequest:
    """One record, one set of forced anchors, and the positions rho is read at."""

    record: Record
    anchors: tuple[int, ...]
    anchor_names: tuple[str, ...]
    site: str
    window: tuple[int, int]
    positions: tuple[int, ...]


#: Fewest scored positions a record may contribute before its rho stops being a mean.
MINIMUM_SCORED_POSITIONS = 16


def site_requests(
    records: Sequence[Record], *, radius: int, max_residues: int, shift: int = 0, site: str | None = None
) -> tuple[tuple[SiteRequest, ...], list[dict[str, Any]]]:
    """Build one request per record, dropping and naming the ones rho is undefined on."""

    admitted: list[SiteRequest] = []
    dropped: list[dict[str, Any]] = []
    for record in records:
        anchors = tuple(position + shift for position in record.anchor_positions)
        window = window_bounds(record, max_residues=max_residues)
        outside = [position for position in anchors if not window[0] <= position <= window[1]]
        if outside:
            dropped.append(
                {"record": record.label, "reason": f"forced anchors {outside} outside the window"}
            )
            continue
        positions = scored_positions(record, anchors, radius=radius, window=window)
        if len(positions) < MINIMUM_SCORED_POSITIONS:
            dropped.append(
                {
                    "record": record.label,
                    "reason": (
                        f"{len(positions)} scored positions at radius {radius}, below the "
                        f"declared {MINIMUM_SCORED_POSITIONS}; a per-residue mean over "
                        "fewer is not one"
                    ),
                }
            )
            continue
        admitted.append(
            SiteRequest(
                record=record,
                anchors=anchors,
                anchor_names=record.anchor_names,
                site=site or ("catalytic" if shift == 0 else f"shift{shift:+d}"),
                window=window,
                positions=positions,
            )
        )
    return tuple(admitted), dropped


def rho_table(likelihood: Any, requests: Sequence[SiteRequest]) -> list[dict[str, Any]]:
    """rho for every request, scored on the positions both forced conditions can read.

    Both windows are handed to the scorer in one call so a batched implementation sees
    them together, and the per-position arrays are intersected on finiteness before the
    mean is taken: at high fragment order the corpus is silent at different positions in
    the two conditions, and a difference of two means over different position sets is
    not a difference of likelihoods.
    """

    windows: list[str] = []
    positions: list[list[int]] = []
    for request in requests:
        forced = forced_windows(
            request.record, request.anchors, request.anchor_names, window=request.window
        )
        offset = request.window[0]
        index = [position - offset for position in request.positions]
        windows.extend([forced["live"], forced["dead"]])
        positions.extend([index, index])
    values = likelihood.nll(windows, positions)
    if len(values) != 2 * len(requests):
        raise RuntimeError("the likelihood returned a row count that is not two per request")
    table: list[dict[str, Any]] = []
    for index, request in enumerate(requests):
        live = np.asarray(values[2 * index], dtype=np.float64)
        dead = np.asarray(values[2 * index + 1], dtype=np.float64)
        finite = np.isfinite(live) & np.isfinite(dead)
        n_scored = int(finite.sum())
        row: dict[str, Any] = {
            "record": request.record.label,
            "accession": request.record.accession,
            "stratum": request.record.stratum,
            "site": request.site,
            "split_unit": request.record.split_unit,
            "n_anchors": len(request.anchors),
            "n_requested": int(live.size),
            "n_scored": n_scored,
            "scored_fraction": (n_scored / live.size) if live.size else 0.0,
            "window": list(request.window),
            "anchors": list(request.anchors),
        }
        if n_scored < MINIMUM_SCORED_POSITIONS:
            row["rho"] = None
            row["unmeasurable_reason"] = (
                f"{n_scored} positions are scorable in both forced conditions, below the "
                f"declared {MINIMUM_SCORED_POSITIONS}"
            )
        else:
            row["rho"] = float((dead[finite] - live[finite]).mean())
            row["nll_live_per_residue"] = float(live[finite].mean())
            row["nll_dead_per_residue"] = float(dead[finite].mean())
        table.append(row)
    return table


def upstream_invariance(
    likelihood: Any, requests: Sequence[SiteRequest], *, positions_per_record: int = 16
) -> dict[str, Any]:
    """The write invariant: forcing an anchor cannot move an upstream likelihood.

    Both an autoregressive decoder and a fragment conditional read leftward only, so the
    likelihood at a position strictly upstream of every forced anchor is identical in
    the two conditions -- **exactly**, not approximately. Anything else means the
    intervention is writing where it was not asked to, or the residue-to-token map is
    off by a position, and either would land directly on rho. Checked rather than
    asserted, because a mapping defect produces a plausible number.
    """

    windows: list[str] = []
    positions: list[list[int]] = []
    checked: list[str] = []
    for request in requests:
        offset = request.window[0]
        first = min(request.anchors)
        upstream = [
            position
            for position in range(max(offset + 1, first - positions_per_record), first)
            if position > offset
        ]
        if not upstream:
            continue
        forced = forced_windows(
            request.record, request.anchors, request.anchor_names, window=request.window
        )
        index = [position - offset for position in upstream]
        windows.extend([forced["live"], forced["dead"]])
        positions.extend([index, index])
        checked.append(request.record.label)
    if not windows:
        return {"checked_records": 0, "max_absolute_difference": None, "holds": None}
    values = likelihood.nll(windows, positions)
    worst = 0.0
    for index in range(len(checked)):
        live = np.asarray(values[2 * index], dtype=np.float64)
        dead = np.asarray(values[2 * index + 1], dtype=np.float64)
        finite = np.isfinite(live) & np.isfinite(dead)
        if finite.any():
            worst = max(worst, float(np.abs(dead[finite] - live[finite]).max()))
    return {
        "checked_records": len(checked),
        "positions_per_record": int(positions_per_record),
        "max_absolute_difference": worst,
        "holds": bool(worst == 0.0),
        "requirement": (
            "exactly zero: a leftward-reading scorer cannot move an upstream likelihood "
            "when a downstream position is forced"
        ),
    }


# ------------------------------------------------------------------ statistics


def auroc(truth: np.ndarray, prediction: np.ndarray) -> float:
    """AUROC, with the positive class being the experimentally dead one.

    A resample carrying one class is returned as ``nan`` rather than raised. The
    counter-stratum contrast resamples 23 singleton groups of which 8 carry the negative
    class, so about one draw in twenty thousand contains no negative and a full
    thousand-draw run hits one about five per cent of the time. The shared group
    bootstrap already skips a non-finite draw and refuses when more than five per cent of
    them are, which is the correct handling; raising instead would kill a campaign cell on
    a resample rather than on a fact about the cohort. The full-sample score is still
    required to be finite there, so a genuinely single-class contrast is refused.
    """

    labels = np.asarray(truth)
    if np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, np.asarray(prediction)))


def is_degenerate(scores: Sequence[float]) -> bool:
    """True when the readout produced no variance at all, which is not chance.

    A model whose rho is identical on every record has not ranked the cohort at chance;
    it has not ranked it. An AUROC of 0.5 over an all-tied score reads as chance and is
    a different finding, so it is returned as its own verdict.
    """

    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        return True
    return bool(np.ptp(values) < 1e-12)


def auroc_interval(
    labels: Sequence[int],
    scores: Sequence[float],
    groups: Sequence[int],
    *,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    """One AUROC and its group interval, through the package's single resampler.

    The chance predictor enters as the right-hand vector: a constant score scores 0.5 on
    every draw, so the paired difference **is** the AUROC's own excess over chance and
    its percentile interval is the AUROC's. No second resampler is added -- there is one
    group bootstrap in this package and this is it.
    """

    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    groups = np.asarray(groups)
    floor = bootstrap_unit_floor(int(np.unique(groups).size))
    if floor["degenerate"]:
        return {"auroc": None, "interval": None, "floor": floor}
    result = paired_group_bootstrap(
        labels,
        scores,
        np.zeros_like(scores),
        groups,
        auroc,
        seed=seed,
        n_bootstrap=draws,
    )
    low, high = result["difference_ci95"]
    return {
        "auroc": result["left_score"],
        "interval": [0.5 + low, 0.5 + high],
        "excess_over_chance": result["difference"],
        "excess_ci95": result["difference_ci95"],
        "n_groups": result["n_groups"],
        "n_finite_draws": result["n_finite_draws"],
        "floor": floor,
    }


def margin_record(
    labels: Sequence[int],
    model_scores: Sequence[float],
    ceiling_scores: Sequence[float],
    groups: Sequence[int],
    *,
    factor: float,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    """Audit §7.0 clause 2's AUROC margin, evaluated inside the resampling.

    ``(AUROC_model - 0.5) >= factor * (AUROC_ceiling - 0.5)``, with the ceiling's excess
    **clamped at zero**. The clause is written for an excess over chance, which is
    non-negative; a ceiling that lands below chance would otherwise make the bar
    negative and let a model inside the ceiling be recorded as clearing it.
    """

    if factor < 1.0:
        raise ValueError(
            "a ceiling factor below one lets a model inside the recombination ceiling be "
            "recorded as clearing it"
        )

    def derived(model: float, ceiling: float) -> float:
        return (model - 0.5) - factor * max(ceiling - 0.5, 0.0)

    result = paired_group_bootstrap(
        np.asarray(labels),
        np.asarray(model_scores, dtype=np.float64),
        np.asarray(ceiling_scores, dtype=np.float64),
        np.asarray(groups),
        auroc,
        seed=seed,
        n_bootstrap=draws,
        derived_statistic=derived,
    )
    ceiling_excess = max(result["right_score"] - 0.5, 0.0)
    clears_difference = bool(result["difference_ci95"][0] > 0.0)
    clears_margin = bool(result["derived_ci95"][0] > 0.0)
    required = {
        str(swept): 0.5 + swept * ceiling_excess for swept in MARGIN_FACTOR_SWEEP
    }
    return {
        "auroc_model": result["left_score"],
        "auroc_ceiling": result["right_score"],
        "difference": result["difference"],
        "difference_ci95": result["difference_ci95"],
        "clears_difference": clears_difference,
        "margin": result["derived_score"],
        "margin_ci95": result["derived_ci95"],
        "clears_margin": clears_margin,
        "factor": float(factor),
        # Audit §7.0 clause 2 has two halves and both bind: the paired interval of the
        # difference excludes zero, AND the model's excess over chance is at least the
        # factor times the ceiling's.
        "clears": bool(clears_difference and clears_margin),
        # Appendix B rule 2, applied to the bar itself. Against a ceiling at AUROC c the
        # factor-f rule demands 0.5 + f*(c - 0.5), which exceeds 1 for any c above
        # 0.5 + 1/(2f) -- at f = 2 that is c > 0.75. A bar no result can reach decides
        # nothing about a model, so it is reported as a property of the ceiling rather
        # than left for a reader to work out from two numbers.
        "required_auroc": 0.5 + float(factor) * ceiling_excess,
        "margin_attainable": bool(0.5 + float(factor) * ceiling_excess <= 1.0),
        "required_auroc_by_factor": required,
        "attainable_by_factor": {key: bool(value <= 1.0) for key, value in required.items()},
        "point_margin_by_factor": {
            str(swept): (result["left_score"] - 0.5) - swept * ceiling_excess
            for swept in MARGIN_FACTOR_SWEEP
        },
        "n_groups": result["n_groups"],
        "rule": (
            "audit §7.0 clause 2, both halves: the paired group interval of "
            "AUROC_model - AUROC_ceiling excludes zero, and "
            "(AUROC_model - 0.5) - factor * max(AUROC_ceiling - 0.5, 0) > 0 over the same "
            "interval. The factor is swept and the point margin at each rung is carried, "
            "so a verdict that depends on the factor is visible as one (Appendix B rule 17)"
        ),
    }


def site_specificity(
    catalytic: float, shifted: Sequence[float]
) -> dict[str, Any]:
    """Whether the effect is specific to the catalytic columns or to substitution anywhere.

    The shifted-site draws move the record's own anchor triple as a rigid body, so an
    effect that survives them is a response to conservative substitution anywhere in a
    kinase domain rather than to the catalytic machinery. Read as the 95th percentile of
    the draws, with the draw count reported beside it because at the declared minimum of
    eight that percentile is the maximum of eight numbers.
    """

    values = np.asarray([value for value in shifted if value is not None], dtype=np.float64)
    if values.size < MINIMUM_RANDOM_ANCHOR_DRAWS:
        return {
            "n_draws": int(values.size),
            "minimum_draws": MINIMUM_RANDOM_ANCHOR_DRAWS,
            "percentile95": None,
            "exceeds": None,
            "reason": "too few shifted-site draws to read a 95th percentile",
        }
    bar = float(np.percentile(values, 95))
    return {
        "n_draws": int(values.size),
        "minimum_draws": MINIMUM_RANDOM_ANCHOR_DRAWS,
        "percentile95": bar,
        "catalytic": float(catalytic),
        "exceeds": bool(catalytic > bar),
        "draws": [float(value) for value in values],
    }


# --------------------------------------------------------------- the ceiling


def ceiling_row(
    name: str,
    labels: Sequence[int],
    model_scores: Sequence[float],
    ceiling_scores: Sequence[float],
    groups: Sequence[int],
    *,
    factor: float,
    seed: int,
    draws: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One ceiling estimator, its own AUROC, and the margin the model clears it by."""

    row = {"name": name, **margin_record(
        labels, model_scores, ceiling_scores, groups, factor=factor, seed=seed, draws=draws
    )}
    row["ceiling_interval"] = auroc_interval(
        labels, ceiling_scores, groups, seed=seed, draws=draws
    )["interval"]
    if extra:
        row.update(dict(extra))
    return row


def ceiling_adequacy(model_rho: Sequence[float], ceiling_rho: Sequence[float]) -> dict[str, Any]:
    """What share of the model's own rho the ceiling itself produces.

    A declared **diagnostic and not a gate**. EXP-R2-214 fixes the margin, and inventing
    a second blocking clause after seeing a number is the failure this programme
    catalogues. It is reported because a ceiling that moves nothing is trivially cleared,
    and a positive obtained against a flat ceiling must not be quotable without that fact
    attached -- D3.j's k = 3 rung did 1.7-3.0% of a decoder's own damage and its "twice
    the ceiling" degenerated into "greater than zero".
    """

    model = np.abs(np.asarray([value for value in model_rho if value is not None], dtype=np.float64))
    ceiling = np.abs(
        np.asarray([value for value in ceiling_rho if value is not None], dtype=np.float64)
    )
    # Both means are always reported, including when the ratio is undefined. A readout
    # that moved nothing is a finding about the model and the k = 1 anchor is a structural
    # check on the ceiling, so neither may disappear behind a null ratio.
    model_mean = float(model.mean()) if model.size else 0.0
    ceiling_mean = float(ceiling.mean()) if ceiling.size else 0.0
    ratio = None if model_mean == 0.0 else float(ceiling_mean / model_mean)
    return {
        "mean_absolute_model_rho": model_mean,
        "mean_absolute_ceiling_rho": ceiling_mean,
        "adequacy_ratio": ratio,
        "floor": CEILING_ADEQUACY_FLOOR,
        "binds": None if ratio is None else bool(ratio >= CEILING_ADEQUACY_FLOOR),
        "undefined_reason": (
            None
            if ratio is not None
            else "the model's own rho is identically zero, so a share of it is undefined"
        ),
    }


def fragment_ceiling_curve(
    orders: Sequence[int],
    backgrounds: Mapping[int, OrderedFragmentCounts],
    rho_by_order: Mapping[int, Sequence[float]],
    labels: Sequence[int],
    model_rho: Sequence[float],
    groups: Sequence[int],
    *,
    factor: float,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    """The recombination ceiling as a curve over fragment order, read at the binding rung.

    EXP-R2-214 amendment 3 fixed this shape for D3.j and it governs here: the verdict is
    read at the **most demanding** order on the curve, never the friendliest, and the
    whole curve is carried so a reader sees where the ceiling starts to bind rather than
    the chosen point. ``k = 1`` is the curve's own reachability anchor: a unigram reads no
    context, so no forcing can move it and its rho is exactly zero at every record. A
    curve whose first point is not exactly zero is an indexing defect, caught before any
    verdict is read.
    """

    rows: list[dict[str, Any]] = []
    for order in sorted(orders):
        counts = backgrounds[order]
        values = list(rho_by_order[order])
        row = ceiling_row(
            f"uniref50_fragment_k{order}",
            labels,
            model_rho,
            values,
            groups,
            factor=factor,
            seed=seed,
            draws=draws,
            extra={
                "order": int(order),
                "background": counts.record(),
                "adequacy": ceiling_adequacy(model_rho, values),
                "rho_is_zero_by_construction": bool(order == 1),
                "pre_registered_rung": bool(order == PRE_REGISTERED_FRAGMENT_ORDER),
            },
        )
        rows.append(row)
    anchor = next((row for row in rows if row["order"] == 1), None)
    if anchor is not None and abs(anchor["adequacy"]["mean_absolute_ceiling_rho"]) != 0.0:
        raise RuntimeError(
            "the k = 1 rung of the ceiling curve is not exactly zero. A unigram reads no "
            "context, so forcing a residue cannot move any other position's likelihood; a "
            "non-zero anchor is a defect in the indexing and not a fact about the corpus"
        )
    binding = max(rows, key=lambda row: row["auroc_ceiling"])
    return {
        "rows": rows,
        "binding_order": binding["order"],
        "binding_reason": (
            "the order at which the corpus fragment model itself separates the classes "
            "best; the verdict is read here and never at the friendliest rung"
        ),
        "verdict_by_ceiling_order": {
            str(row["order"]): ("CLEARS" if row["clears"] else "INSIDE") for row in rows
        },
        "survives_every_ceiling_order": bool(all(row["clears"] for row in rows)),
        "clears_binding_order": bool(binding["clears"]),
    }


# ---------------------------------------------------------------- verdicts


def architecture_response(measurement: Mapping[str, Any]) -> dict[str, Any]:
    """D3.k-A1 in this design's terms: does rho respond to domain architecture at all?

    The same records with their non-anchor residues permuted keep composition exactly
    and destroy the architecture, so a readout that measures whether a forced catalytic
    state is *coherent with the rest of the domain* must separate a real kinase domain
    from its own shuffle. It is reported as a qualifier rather than as a gate, and the
    distinction is deliberate: a failure here is a statement about the **arm** -- its rho
    does not track domain architecture -- and not a defect in the pipeline, so it must
    not void a run. What it does do is make a null on the contradiction contrast
    uninterpretable for that arm, which is Appendix B rule 40's requirement that a zero
    be shown reachable before it is reported, attached to the number rather than
    replacing it.
    """

    interval = measurement.get("interval")
    if interval is None:
        return {"separates": None, "reason": "below the unit floor", **dict(measurement)}
    return {
        "separates": bool(interval[0] > 0.5),
        "requirement": (
            "the group interval of the real-versus-shuffled AUROC excludes chance from "
            "above; the shuffle holds composition fixed and destroys the architecture"
        ),
        **dict(measurement),
    }


def primary_verdict(
    *,
    degenerate: bool,
    invariant: Mapping[str, Any],
    measurement: Mapping[str, Any],
    ceiling_rows: Sequence[Mapping[str, Any]],
    reversed_rows: Sequence[Mapping[str, Any]],
    specificity: Mapping[str, Any],
    response: Mapping[str, Any],
) -> dict[str, Any]:
    """The three-way read amendment 1 item 3 fixes, as one verdict.

    The order of the tests is the order in which a failure invalidates what follows. A
    readout with no variance has not ranked the cohort at chance and is returned as its
    own finding; a write invariant that fails means the intervention did not do what it
    says, and nothing downstream of it is a measurement; a value the interval cannot
    separate from chance at n = 15 is refused rather than called.
    """

    if degenerate:
        return {
            "verdict": READOUT_DEGENERATE,
            "reason": (
                "rho is identical on every record, so the intervention moved nothing this "
                "readout can rank. That is not chance and is not reported as chance"
            ),
        }
    if invariant.get("holds") is False:
        return {
            "verdict": VOID_READOUT,
            "reason": (
                "forcing a downstream anchor moved an upstream likelihood by "
                f"{invariant['max_absolute_difference']}, which a leftward-reading scorer "
                "cannot do. The intervention or the residue-to-token map is wrong and no "
                "number behind it is a measurement"
            ),
        }
    value = measurement["auroc"]
    interval = measurement["interval"]
    if value is None or interval is None:
        return {
            "verdict": NOT_RESOLVABLE,
            "reason": "the deciding side is below the shared unit floor",
        }
    clears_forward = bool(ceiling_rows) and all(row["clears"] for row in ceiling_rows)
    clears_reversed = bool(reversed_rows) and all(row["clears"] for row in reversed_rows)
    binding = max(ceiling_rows, key=lambda row: row["auroc_ceiling"]) if ceiling_rows else None
    qualifier = (
        None
        if response.get("separates")
        else (
            "NULL_NOT_SHOWN_REACHABLE: this arm's rho does not separate a real kinase "
            "domain from its own composition-preserving shuffle, so a null on the "
            "contradiction contrast cannot be attributed to what the model knows about "
            "catalysis rather than to a readout that does not respond to domain "
            "architecture on this arm (Appendix B rule 40)"
        )
    )
    common = {
        "auroc": value,
        "interval": interval,
        "binding_ceiling": None if binding is None else binding["name"],
        "binding_ceiling_auroc": None if binding is None else binding["auroc_ceiling"],
        "margin_factor": float(ceiling_rows[0]["factor"]) if ceiling_rows else None,
        "resolvable_at": RESOLVABLE_AUROC,
        "site_specific": specificity.get("exceeds"),
    }
    if clears_forward and value >= RESOLVABLE_AUROC and interval[0] > 0.5 and specificity.get("exceeds"):
        return {
            "verdict": CLEARS_TOWARD_EXPERIMENT,
            "reason": (
                "the model's separation of experimentally dead from bit-score-matched "
                "active clears the recombination ceiling at every rung under the standing "
                "margin, is resolvable at this n, and survives the shifted-site control. "
                "§7.0 clause 5: a candidate and nothing more"
            ),
            **common,
            "qualifier": qualifier,
        }
    if clears_reversed and value <= 1.0 - RESOLVABLE_AUROC and interval[1] < 0.5:
        return {
            "verdict": CLEARS_REVERSED,
            "reason": (
                "the model separates the classes in the direction opposite to the "
                "catalytic account -- the pseudokinase scaffolds respond more to the "
                "forced catalytic state than the active ones do. Registered as a positive "
                "result for a reading this design did not predict, not as a partial pass"
            ),
            **common,
            "qualifier": qualifier,
        }
    if not clears_forward and not clears_reversed:
        # Two ways to fail, and they are different findings. If the model exceeds every
        # ceiling row on the difference half of clause 2 and fails only the multiple half
        # against a row whose bar no AUROC can reach, that is Appendix B rule 2's failure
        # -- a gate that was never shown reachable -- and it is a statement about the bar
        # rather than about the model.
        beaten = [row for row in ceiling_rows if not row.get("clears_difference", False)]
        unreachable = [
            row["name"]
            for row in ceiling_rows
            if not row.get("margin_attainable", True) and not row.get("clears_margin", False)
        ]
        if not beaten and unreachable:
            return {
                "verdict": MARGIN_UNATTAINABLE,
                "reason": (
                    "the model exceeds every ceiling row on the difference half of §7.0 "
                    "clause 2, and fails only the multiple half against "
                    f"{unreachable}, whose bar at factor {common['margin_factor']} demands "
                    f"an AUROC of {max(row['required_auroc'] for row in ceiling_rows if row['name'] in unreachable):.4f}. "
                    "No result can reach it, so no model can be classified by it. This is "
                    "Appendix B rule 2 applied to the admission rule itself: the outcome is "
                    "a statement about the bar and about a cohort whose caliper did not "
                    "neutralise this channel, not about the model"
                ),
                **common,
                "unattainable_rows": unreachable,
                "qualifier": qualifier,
            }
        return {
            "verdict": RECOMBINATION,
            "reason": (
                "the model's AUROC does not clear the recombination ceiling under the "
                "standing margin at the binding rung. §7.0 clause 5: this line halts and "
                "is classified as recombination -- not narrowed to a subcohort, not re-run "
                "at more seeds, not reported as weak evidence of knowledge"
            ),
            **common,
            "ceiling_rows_the_model_does_not_exceed": [row["name"] for row in beaten],
            "unattainable_rows": unreachable,
            "qualifier": qualifier,
        }
    return {
        "verdict": NOT_RESOLVABLE,
        "reason": (
            f"the model clears the ceiling but its AUROC of {value:.4f} lies inside the "
            f"band this cohort cannot call. The realised interval at n = 15 is about "
            f"+/-0.07, so only separations of roughly {RESOLVABLE_AUROC} are resolvable, "
            "and the unit count is bounded by the number of human genes with published "
            "catalysis experiments rather than by effort (amendment 1, item 5)"
        ),
        **common,
        "qualifier": qualifier,
    }


def counter_stratum_verdict(
    *, degenerate: bool, measurement: Mapping[str, Any]
) -> dict[str, Any]:
    """The second contradiction set's own verdict, read separately from the first.

    ``active_despite_degradation`` holds kinases that are experimentally active with
    degraded catalytic machinery, so a motif reader calls seven of the eight dead. The
    contrast here is that stratum against the experimentally dead one, oriented so that
    dead ranks high. A model that separates them is not reading motifs; a model that
    cannot is, whatever it achieved on the primary contrast. This is a second reading of
    the same cohort and not a robustness check on the first: a robustness check that
    fails narrows a claim, and this one changes what the claim is about.
    """

    if degenerate:
        return {
            "verdict": READOUT_DEGENERATE,
            "reason": "rho is identical on every record of the counter-stratum contrast",
        }
    value = measurement["auroc"]
    interval = measurement["interval"]
    if value is None or interval is None:
        return {
            "verdict": DOES_NOT_SEPARATE_COUNTER,
            "reason": "the counter-stratum contrast is below the shared unit floor",
            "auroc": value,
            "interval": interval,
        }
    common = {"auroc": value, "interval": interval, "resolvable_at": RESOLVABLE_AUROC}
    if interval[0] > 0.5 and value >= RESOLVABLE_AUROC:
        return {
            "verdict": SEPARATES_COUNTER,
            "reason": (
                "the model ranks the experimentally active but motif-degraded stratum away "
                "from the experimentally dead one, which a motif reader cannot do on seven "
                "of these eight records"
            ),
            **common,
        }
    if interval[1] < 0.5 and value <= 1.0 - RESOLVABLE_AUROC:
        return {
            "verdict": COUNTER_REVERSED,
            "reason": (
                "the model ranks the motif-degraded actives as MORE dead than the "
                "experimentally dead stratum, which is a stronger motif reading than the "
                "motif reader itself"
            ),
            **common,
        }
    return {
        "verdict": DOES_NOT_SEPARATE_COUNTER,
        "reason": (
            "the model does not separate the experimentally active but motif-degraded "
            "stratum from the experimentally dead one. On seven of these eight records the "
            "catalytic columns are degraded and the protein is active, so a reading that "
            "groups them with the dead is a motif reading"
        ),
        **common,
    }


def combined_verdict(primary: Mapping[str, Any], counter: Mapping[str, Any]) -> dict[str, Any]:
    """The one line a reader takes away, with the motif freeze applied."""

    first, second = primary["verdict"], counter["verdict"]
    if first in (READOUT_DEGENERATE, VOID_READOUT):
        return {"verdict": first, "reason": primary["reason"]}
    if first == CLEARS_TOWARD_EXPERIMENT:
        if second == SEPARATES_COUNTER:
            return {
                "verdict": CANDIDATE_KNOWLEDGE,
                "reason": (
                    "the model clears the recombination ceiling on the caliper-matched "
                    "contrast AND separates the orthogonal counter-stratum, so the reading "
                    "is not a motif reading. §8's causal, retrieval-aware and independent "
                    "biological clauses are all open and none is discharged here"
                ),
            }
        return {
            "verdict": MOTIF_READING,
            "reason": (
                "the model clears the ceiling on the primary contrast and fails to separate "
                "the counter-stratum, whose members are experimentally active with degraded "
                "catalytic machinery. That is reading motifs rather than structure and is "
                "frozen as such by EXP-R2-214 amendment 1, item 4 -- it is not reported as a "
                "partial pass"
            ),
        }
    return {"verdict": first, "reason": primary["reason"]}


# ------------------------------------------------- known-answer synthetic world

#: The planted decoders the self-test runs, and the verdict each must return. Four
#: rather than three, because these are the four ways this readout can be wrong and
#: each has a different remedy: knowledge, a motif reading that survives the primary
#: contrast and dies on the counter-stratum, a corpus-statistical reading that lands
#: inside its own ceiling, and an intervention that moved nothing.
PLANTINGS: tuple[str, ...] = ("catalysis", "motif", "statistics", "null")
EXPECTED_SYNTHETIC_VERDICT: dict[str, str] = {
    "catalysis": CANDIDATE_KNOWLEDGE,
    "motif": MOTIF_READING,
    "statistics": RECOMBINATION,
    "null": READOUT_DEGENERATE,
}

SYNTHETIC_WINDOW = 240
#: Window indices, zero-based, of the three forced anchors in every synthetic record.
SYNTHETIC_ANCHOR_INDEX: tuple[int, ...] = (30, 130, 148)
#: Where the two planted features sit. Both are upstream of the first anchor, so neither
#: is ever forced and neither is ever scored: a planted world in which the feature the
#: decoder reads is inside the scored span would be certifying arithmetic rather than a
#: verdict.
SYNTHETIC_SCAFFOLD_INDEX = 10
SYNTHETIC_MOTIF_INDEX = 14
SYNTHETIC_PAIRS = 15
SYNTHETIC_COUNTER_RECORDS = 8
SYNTHETIC_ORDERS: tuple[int, ...] = (1, 2, 3)


def synthetic_background(seed: int) -> dict[int, OrderedFragmentCounts]:
    """A declared trigram corpus, and its own consistent lower-order marginals.

    The synthetic world's sequences are sampled from this table, so the fragment
    conditional built on it is the **exact** corpus-statistics predictor of that world
    rather than an estimate of it. That is what lets the self-test exercise the real
    ceiling code and the real margin rule, instead of declaring the ceiling untestable
    on synthetic data.
    """

    rng = np.random.default_rng(seed)
    size = len(AA20)
    joint = rng.integers(1, 1000, size=size**3).astype(np.int64)
    cube = joint.reshape(size, size, size)
    counts = {
        3: joint,
        2: cube.sum(axis=0).reshape(-1),
        1: cube.sum(axis=(0, 1)).reshape(-1),
    }
    return {
        order: OrderedFragmentCounts(
            order=order,
            counts=counts[order],
            source=f"synthetic_trigram_corpus(seed={seed})",
            sha256=hashlib.sha256(counts[order].tobytes()).hexdigest(),
            observed=int((counts[order] > 0).sum()),
            possible=int(counts[order].size),
            total_kmers=int(counts[order].sum()),
        )
        for order in SYNTHETIC_ORDERS
    }


def _sample_window(counts: np.ndarray, rng: Any, length: int) -> str:
    size = len(AA20)
    cube = counts.reshape(size, size, size)
    first, second = int(rng.integers(size)), int(rng.integers(size))
    out = [first, second]
    for _ in range(length - 2):
        weights = cube[out[-2], out[-1]].astype(np.float64)
        out.append(int(rng.choice(size, p=weights / weights.sum())))
    return "".join(AA20[index] for index in out)


def synthetic_cohort(
    background: Mapping[int, OrderedFragmentCounts], *, seed: int
) -> tuple[tuple[MatchedPair, ...], tuple[Record, ...]]:
    """A cohort with the real one's shape and the real one's disagreement planted in it.

    Fifteen matched pairs and an eight-record counter-stratum, carrying two features the
    planted decoders may read. The **scaffold** feature marks experimental catalytic
    competence and is true for the actives and for the counter-stratum. The **motif**
    feature marks an intact catalytic reading and is true for the actives only. They
    agree on the primary contrast and disagree on the counter-stratum, which is exactly
    the disagreement that makes ``active_despite_degradation`` a second contradiction set
    rather than a robustness check.
    """

    rng = np.random.default_rng(seed)
    trigram = background[3].counts

    def build(unit: int, stratum: str, *, scaffold: bool, motif: bool, partner: str) -> Record:
        text = list(_sample_window(trigram, rng, SYNTHETIC_WINDOW))
        text[SYNTHETIC_SCAFFOLD_INDEX] = "W" if scaffold else "Y"
        text[SYNTHETIC_MOTIF_INDEX] = "C" if motif else "A"
        state = LIVE_STATE if motif else DEAD_STATE
        for index, name in zip(SYNTHETIC_ANCHOR_INDEX, CATALYTIC_ANCHORS):
            text[index] = state[name]
        accession = f"SYN{stratum[:4].upper()}{unit:03d}"
        return Record(
            accession=accession,
            gene=accession,
            entry_name=accession,
            stratum=stratum,
            sequence="".join(text),
            domain_from=1,
            domain_to=SYNTHETIC_WINDOW,
            anchor_names=CATALYTIC_ANCHORS,
            anchor_positions=tuple(index + 1 for index in SYNTHETIC_ANCHOR_INDEX),
            observed_anchor_residues=tuple(state[name] for name in CATALYTIC_ANCHORS),
            n_intact=3 if motif else 0,
            domain_bits=100.0,
            split_unit=unit,
            label_confidence="high",
            annotation_stance="silent",
            matched_partner=partner,
        )

    pairs = tuple(
        MatchedPair(
            dead=build(unit, DEAD_STRATUM, scaffold=False, motif=False, partner=f"SYNACTI{unit:03d}"),
            active=build(unit, ACTIVE_STRATUM, scaffold=True, motif=True, partner=f"SYNDEAD{unit:03d}"),
        )
        for unit in range(SYNTHETIC_PAIRS)
    )
    counter = tuple(
        build(SYNTHETIC_PAIRS + unit, COUNTER_STRATUM, scaffold=True, motif=False, partner="")
        for unit in range(SYNTHETIC_COUNTER_RECORDS)
    )
    return pairs, counter


class PlantedLikelihood:
    """A decoder whose reading is planted, so the verdict it must produce is known.

    Every planting shares one base: the exact fragment conditional of the synthetic
    world's own corpus. ``statistics`` **is** that conditional, so it lands inside its own
    ceiling by construction rather than by tuning. ``catalysis`` and ``motif`` add a
    coupling term that fires only where the forced anchors carry the live state *and* the
    record carries the planted feature, and only at positions downstream of the first
    anchor -- so both remain leftward-consistent and pass the write invariant, which a
    planting that shifted every position would not.
    """

    def __init__(self, planting: str, background: Mapping[int, OrderedFragmentCounts], *, coupling: float) -> None:
        if planting not in PLANTINGS:
            raise ValueError(f"unknown planting {planting!r}; declared: {list(PLANTINGS)}")
        if coupling <= 0.0:
            raise ValueError("the planted coupling must be positive to be recoverable")
        self.planting = planting
        self.coupling = float(coupling)
        self._base = FragmentLikelihood(background[max(SYNTHETIC_ORDERS)])

    def _gate(self, window: str) -> float:
        live = all(
            window[index] == LIVE_STATE[name]
            for index, name in zip(SYNTHETIC_ANCHOR_INDEX, CATALYTIC_ANCHORS)
        )
        if self.planting == "catalysis":
            feature = window[SYNTHETIC_SCAFFOLD_INDEX] == "W"
        else:
            feature = window[SYNTHETIC_MOTIF_INDEX] == "C"
        return 1.0 if (live and feature) else 0.0

    def nll(self, windows: Sequence[str], positions: Sequence[Sequence[int]]) -> list[np.ndarray]:
        base = self._base.nll(windows, positions)
        if self.planting == "statistics":
            return base
        if self.planting == "null":
            return [np.zeros_like(values) for values in base]
        rows: list[np.ndarray] = []
        for window, wanted, values in zip(windows, positions, base):
            downstream = np.asarray(wanted, dtype=np.int64) > SYNTHETIC_ANCHOR_INDEX[0]
            rows.append(values - self.coupling * self._gate(window) * downstream)
        return rows

    def cost(self) -> dict[str, Any]:
        return {"planting": self.planting, "coupling": self.coupling}


# -------------------------------------------------------------- one measured cell


@dataclass(frozen=True)
class Design:
    """Everything about a cell that is fixed before any scorer is built."""

    pairs: tuple[MatchedPair, ...]
    counter: tuple[Record, ...]
    radius: int
    max_residues: int
    seed: int
    shift_draws: int

    @property
    def primary_records(self) -> tuple[Record, ...]:
        return tuple(
            record for pair in self.pairs for record in (pair.dead, pair.active)
        )

    @property
    def all_records(self) -> tuple[Record, ...]:
        return self.primary_records + tuple(self.counter)


def _rho_by_accession(table: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    return {row["accession"]: row["rho"] for row in table}


def measure(likelihood: Any, design: Design, *, with_shifts: bool, with_shuffle: bool) -> dict[str, Any]:
    """Every rho one scorer produces on one design, through one code path.

    The model and every ceiling estimator are measured by this function and differ only
    in the object handed to it, which is what makes "the same readout" a property of the
    code rather than a claim in a docstring (audit §7.0 clause 2).
    """

    requests, dropped = site_requests(
        design.all_records, radius=design.radius, max_residues=design.max_residues
    )
    catalytic = rho_table(likelihood, requests)
    block: dict[str, Any] = {
        "catalytic": _rho_by_accession(catalytic),
        "rows": catalytic,
        "dropped": dropped,
        "invariant": upstream_invariance(likelihood, requests),
    }
    if with_shifts:
        shifts: dict[str, tuple[int, ...]] = {}
        shift_notes: dict[str, Any] = {}
        for record in design.all_records:
            chosen, note = anchor_shifts(
                record,
                draws=design.shift_draws,
                radius=design.radius,
                seed=design.seed + record.split_unit,
            )
            shifts[record.accession] = chosen
            shift_notes[record.label] = note
        usable = [record for record in design.all_records if shifts[record.accession]]
        drawn: list[dict[str, float | None]] = []
        for index in range(design.shift_draws):
            shifted_requests: list[SiteRequest] = []
            for record in usable:
                built, _ = site_requests(
                    [record],
                    radius=design.radius,
                    max_residues=design.max_residues,
                    shift=shifts[record.accession][index],
                    site=f"shift{shifts[record.accession][index]:+d}",
                )
                shifted_requests.extend(built)
            drawn.append(_rho_by_accession(rho_table(likelihood, shifted_requests)))
        block["shifted"] = drawn
        block["shift_notes"] = shift_notes
    if with_shuffle:
        shuffled_rows: list[dict[str, Any]] = []
        for request in requests:
            if request.record.stratum not in (DEAD_STRATUM, ACTIVE_STRATUM):
                continue
            window = request.window
            permuted = shuffled_control(request.record, window=window, seed=design.seed)
            surrogate = Record(
                **{
                    **request.record.__dict__,
                    "accession": request.record.accession + "_shuffled",
                    "sequence": request.record.sequence[: window[0] - 1] + permuted,
                }
            )
            built, _ = site_requests(
                [surrogate], radius=design.radius, max_residues=design.max_residues
            )
            shuffled_rows.extend(rho_table(likelihood, built))
        block["shuffled"] = _rho_by_accession(shuffled_rows)
    return block


def contrast(
    rho: Mapping[str, float | None],
    positives: Sequence[Record],
    negatives: Sequence[Record],
) -> dict[str, Any]:
    """Labels, scores and resampling groups for one AUROC, with the orientation fixed.

    The score is ``-rho``: the catalytic account predicts that an active kinase's domain
    responds more to the forced catalytic state than a pseudokinase's, so a **smaller**
    rho predicts dead. The positive class is always the one the design predicts ranks
    high, and it is named in the returned record so no reader has to infer it.
    """

    labels: list[int] = []
    scores: list[float] = []
    groups: list[int] = []
    missing: list[str] = []
    for records, label in ((positives, 1), (negatives, 0)):
        for record in records:
            value = rho.get(record.accession)
            if value is None:
                missing.append(record.label)
                continue
            labels.append(label)
            scores.append(-float(value))
            groups.append(record.split_unit)
    return {
        "labels": np.asarray(labels),
        "scores": np.asarray(scores, dtype=np.float64),
        "groups": np.asarray(groups),
        "missing": missing,
        "n_positive": int(sum(labels)),
        "n_negative": int(len(labels) - sum(labels)),
        "n_groups": int(np.unique(groups).size) if groups else 0,
        "orientation": "score = -rho; the positive class is the experimentally dead one",
    }


# ------------------------------------------- the statistics family's own readouts

#: Fragment order of the composition/retrieval ceiling rows. Three, because that is the
#: order EXP-R2-214 names for D3.k's baseline (b) and the order the pinned background
#: was frozen at; the fragment *curve* is where the higher orders enter.
RETRIEVAL_ORDER = 3


def domain_kmer_vector(record: Record, *, order: int = RETRIEVAL_ORDER) -> np.ndarray:
    """The record's kinase-domain k-mer frequency vector, L1-normalised."""

    domain = record.sequence[record.domain_from - 1 : record.domain_to]
    size = len(AA20)
    vector = np.zeros(size**order, dtype=np.float64)
    for start in range(len(domain) - order + 1):
        index = 0
        valid = True
        for character in domain[start : start + order]:
            position = AA20.find(character)
            if position < 0:
                valid = False
                break
            index = index * size + position
        if valid:
            vector[index] += 1.0
    total = vector.sum()
    return vector / total if total else vector


def retrieval_scores(
    records: Sequence[Record], pool: Sequence[Record], *, order: int = RETRIEVAL_ORDER
) -> dict[str, dict[str, float]]:
    """Two corpus-side readouts on the same records: nearest neighbour and centroid.

    Both are members of audit §7.0 clause 1's family and neither reads a residue's
    meaning: the **nearest-neighbour** row is retrieval against the cohort's own active
    pool, which is EXP-R2-214's baseline (c) at the fragment scale this host can compute
    without an aligner, and the **centroid** row is the composition/fragment channel of
    baseline (b). They are reported beside the fragment curve as the statistics family's
    best on this contrast **by its own natural readout**, which is a strictly more
    demanding reading of clause 2 than the same-readout rows alone.
    """

    if not pool:
        raise ValueError("a retrieval readout needs a pool to retrieve from")
    pool_vectors = np.stack([domain_kmer_vector(record, order=order) for record in pool])
    norms = np.linalg.norm(pool_vectors, axis=1)
    centroid = pool_vectors.mean(axis=0)
    centroid_norm = float(np.linalg.norm(centroid))
    nearest: dict[str, float] = {}
    centred: dict[str, float] = {}
    pool_accessions = [record.accession for record in pool]
    for record in records:
        vector = domain_kmer_vector(record, order=order)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0 or centroid_norm == 0.0:
            raise ValueError(f"{record.label}: an empty domain has no fragment vector")
        similarity = (pool_vectors @ vector) / (norms * norm)
        mask = np.asarray([accession != record.accession for accession in pool_accessions])
        nearest[record.accession] = float(similarity[mask].max())
        centred[record.accession] = float(centroid @ vector / (centroid_norm * norm))
    return {"nearest_active": nearest, "active_centroid": centred}
