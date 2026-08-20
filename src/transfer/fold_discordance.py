"""D3.l: does a continuation follow the sequence neighbour or the structure neighbour?

**What this module is.** The measurement half of EXP-R2-214's third track, read on
the composition-matched / fold-discordant triple set that amendment 2 (D3.l)
admitted. Each record is ``(anchor, sequence partner, structure partner)``: the
sequence partner is composition-matched to the anchor, CATH-disjoint and
different in contact map; the structure partner carries the anchor's fold and was
selected as far as possible in sequence space. Evolutionary statistics and
biological structure therefore name **different records on every admitted
triple**, which is what §7.0 clause 4 requires of a contradiction set and what
neither F10, F12 nor D3.g's stage 35 had.

The estimand, and the three confounds it had to be built around
================================================================

The natural readout is a conditional likelihood contrast: score the sequence
partner's continuation and the structure partner's continuation under the
anchor's prefix and take the sign of the difference. Written down naively that
statistic is not identified, for three separate reasons, and each resolution is
recorded here rather than in prose somewhere else.

**1. The candidates differ in length.** Two things are done and both are
reported. Every score is a **mean over scored residue positions** rather than a
sum, so a longer continuation does not accumulate more evidence; and the
``length_controlled`` reading truncates both candidates to
``min(T_sequence, T_structure)`` positions, so the two are averaged over the same
*number* of positions **and the same position indices**. Per-token normalisation
alone does not do that: per-token likelihood falls with distance from the
conditioning context, so two candidates averaged over 104 and 93 positions are
averaged over different parts of that decay curve. Both readings are carried and
a verdict that differs between them is reported as differing.

**2. The candidates differ in composition, by construction.** The sequence
partner is composition-matched to the anchor and the structure partner is not, so
each candidate's *own* marginal likelihood is a nuisance quantity of exactly the
size the design is trying to measure. The estimand is therefore the **prefix
advantage** and not the bare conditional:

    A(c) = (1/|W|) * sum over the scored window of
           [ log p(c_i | x, c_<i) - log p(c_i | c_<i) ]
    P    = A(c_structure) - A(c_sequence)

Subtracting each candidate's own no-prefix score removes everything about the
candidate that does not depend on the anchor -- its composition, its length, its
own fragment typicality -- and leaves the quantity the design is about: how much
*this prefix* helps *this continuation*. The bare conditional contrast is
computed and reported beside it as ``conditional_contrast``, because it is the
statistic the pre-registration froze, but it is not the one a verdict is read
from, and :data:`ESTIMAND_DEVIATION` states why in the artefact.

The subtraction is also what gives the ceiling curve its reachability anchor: a
k = 1 corpus conditional reads no context, so its prefix advantage is **exactly**
zero for every position of every candidate and its contrast is exactly zero. A
curve whose first point is not exactly zero is an indexing defect, caught before
any verdict is read. On the bare conditional statistic k = 1 is *not* zero -- a
unigram already separates two continuations of different composition -- so that
statistic has no such anchor, which is a second reason it is not the one read.

**3. The prefix already favours the sequence partner.** The cohort measured the
prefix's own composition as closer to the sequence partner in 94.0% of triples.
That is the direction the design needs, and it means a model that merely
continues the prefix's composition scores as "sequence" for a reason that has
nothing to do with biology. Two independent handles are built for it. On the
statistics side :class:`PrefixAdaptedComposition` is a ceiling member that *is*
that model -- the plug-in residue distribution of the anchor's own prefix, with
the corpus unigram as its prior -- so the trivial account sits inside the ceiling
rather than being argued away. On the model side the composition-preserving
**prefix shuffle** null rescores both candidates under a permutation of the
anchor's prefix: a contrast that survives the shuffle is a contrast about
composition, and it fires as a defect rather than being reported as a finding.

**The junction.** Splicing a foreign continuation onto a prefix creates a
boundary that exists in neither protein. ``--junction-offset`` is a required
never-defaulted decision naming how many residues after the splice the scored
window begins, and the choice reaches the artefact. It is not a free parameter:
a Markov conditional of order k carries information across exactly ``k - 1``
positions, so at an offset of ``k - 1`` or more the fragment ceiling at that
order is **identically zero over the window**. That is reported per order as
``ceiling_reaches_window``, and it is the single most important thing a reader
needs in order to know whether the margin bound anything.

The recombination ceiling
=========================

§7.0 clause 2: the ceiling is the best the same contrast achieves, on the same
records at the same unit, by any member of the clause-1 family. Two kinds of
member are computed, both from the pinned UniRef50 background:

* :class:`FragmentPrefixConditional` at every order the staged
  ``uniref50_high_order`` background supports, read as a **curve** and never at
  one chosen rung. EXP-R2-214 amendment 3 froze this after measuring that a
  k = 3 ceiling did 1.7-3.0% of a decoder's own damage on D3.j's estimand, which
  turns "at least twice the ceiling" into "greater than zero". The verdict is
  read at the **binding** order -- the most demanding rung on the curve -- and
  ``verdict_by_ceiling_order`` and ``survives_every_ceiling_order`` are carried so
  a reader sees the whole curve. Coverage and observations per k-mer are reported
  at every order, because that is where the honest limit of the curve is.
* :class:`PrefixAdaptedComposition`, confound 3's answer. It reads the whole
  prefix at every position rather than ``k - 1`` of them, so it is the only
  ceiling member that survives a junction offset, and it is the member that
  captures the trivial account the cohort's own 94.0% figure warns about.

**The 2x margin belongs to the model-side effect and never to the cohort's
neighbour separation.** Amendment 2 froze that after measuring that requiring a
2x ordering margin of the cohort would have cut it from 199 triples to 4. Nothing
in this module gates on a cohort ordering margin; the realised separation is a
reported distribution of the cohort's own manifest.

**The ceiling is one family with three kinds of member, and the third kind is what
decides the answer.** §7.0 clause 1 names fragment and k-mer statistics,
profile-HMM scores *and* Potts/MSA couplings, and clause 2 makes the ceiling the
best any member achieves. The fragment conditional reads ``k - 1`` residues; the
composition channel reads residue counts; neither can detect homology below the
fragment scale. But the structure partner shares the anchor's CATH superfamily
**by construction**, and clause 3 states plainly that an alignment certificate
excludes whole-sequence retrieval and essentially nothing below the alignment
scale -- so a remote-homology detector, which is a corpus-statistics object under
clause 1, predicts a contrast toward the structure partner with no structural
knowledge whatever. Reading a positive result without such a member in the ceiling
would be an agreement set hiding inside a contradiction set.

Two profile members are therefore built. :func:`build_pfam_profile_member` selects
Pfam-A profiles with the anchor's prefix at Pfam's own gathering threshold and
scores the candidates under exactly those, filters off; its profiles are curated
over UniProt and contain none of this cohort's sequences, so it carries no
circularity. :func:`build_corpus_profile_member` builds a profile from each prefix
by jackhmmer against the staged corpus and scores the candidates under it, with
**both candidate accessions removed from the recruited alignment** so that no
candidate is scored against a profile containing it. That member also reports the
measurement the whole question turns on and which needs no model at all: how often
a corpus search seeded only by the anchor's prefix recruits the structure partner.
Both members are lower bounds in the same direction -- Pfam annotates only part of
a proteome, and the corpus member searches Swiss-Prot rather than the arms' own
pretraining mixtures -- and each reports the coverage that bounds it.

:data:`POTTS_MEMBER_ABSENT` states why no coupling member is built, and it is a
claim about this readout rather than about cost: detection is what a profile method
is for, a coupling term would be fitted from the same alignment the profile member
already uses, and the assertion that it would not become the binding member is
checkable against the profile members' own adequacy ratios in the artefact.

What this module does not do
============================

It fits nothing, so there is no train/test split to make group-disjoint; the
cohort's group-disjoint split exists for a stage that fits, and this one reads
all 199 triples as one set. It compares nothing across arms: a scored token is a
residue on a residue-tokenised arm and the estimand is undefined on an arm where
it is not, which is a refusal and not a comparison. And clearing the ceiling
licenses a candidate and nothing more -- §8's causal, retrieval-aware and
independent-biological clauses are untouched by anything here (§7.0 clause 5).
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from . import alphabet_chemistry as ac
from .arms import AA20, Arm, ArmSpec, Cohort, tokenize_batch
from .io import sha256_file
from .statistics import (
    MINIMUM_BOOTSTRAP_UNITS,
    bootstrap_unit_floor,
    paired_group_bootstrap,
)

# --------------------------------------------------------------- declarations

#: The log entry this stage is pre-registered under.
PRE_REGISTRATION = "EXP-R2-214"

#: The track, as the pre-registration names it.
PRE_REGISTRATION_TRACK = "D3.l, composition-matched sequence neighbour against fold neighbour"

#: The amendments implemented here, named so an artefact records which text it
#: was run against. Amendment 2 restated the estimand as an ordering contrast,
#: relocated the 2x margin off the cohort and made the unit a triple; amendment 3
#: turned the ceiling into a curve read at the binding order, and is applied to
#: this track for the reason it was written for D3.j.
PRE_REGISTRATION_AMENDMENTS: tuple[str, ...] = (
    "amendment 2 (D3.l): the estimand is an ordering contrast, the 2x margin is "
    "the model-side effect and never the cohort's neighbour separation, and the "
    "unit is a triple",
    "amendment 3 (D3.j), applied here: the ceiling is a curve over every staged "
    "order and the verdict is read at the binding order, never the friendliest",
)

#: The one departure from the frozen statistic, declared as such rather than
#: folded in silently. The pre-registration froze
#: ``P = log p(c_str | x) - log p(c_seq | x)``; this stage subtracts each
#: candidate's own no-prefix score before differencing.
ESTIMAND_DEVIATION = (
    "the frozen statistic is the bare conditional difference "
    "log p(c_str | x) - log p(c_seq | x). This stage reads the marginal-subtracted "
    "form [log p(c_str | x) - log p(c_str)] - [log p(c_seq | x) - log p(c_seq)] and "
    "reports the bare form beside it as conditional_contrast. Two reasons, both "
    "structural rather than cosmetic. The sequence partner is composition-matched "
    "to the anchor and the structure partner is not, so on the bare statistic each "
    "candidate's own marginal likelihood is a nuisance of the same size as the "
    "effect; and on the bare statistic a k = 1 corpus unigram already separates the "
    "two candidates, so the ceiling curve has no zero point and an indexing defect "
    "in it would be undetectable. The subtraction removes everything that does not "
    "depend on the anchor's prefix, which is what the ordering contrast of "
    "amendment 2 asks about"
)

#: Digest of the staged cohort this track is pre-registered on, from EXP-R2-214
#: amendment 2. Pinned in code rather than taken from the manifest beside the
#: file: a manifest travels with the artefact it describes, so a manifest alone
#: certifies internal consistency and not identity.
COHORT_DIGEST = "e4ba96f591beb5159cd19e33cf0a4cb60f52dbc916ca977f9420a7b349c4463c"

#: Inference dtype. float32 throughout: the estimand is a difference of two
#: log-likelihoods of the same tokens under two contexts, and at bfloat16 the
#: rounding of each term is of the same order as several of the per-position
#: differences being taken.
DTYPE = "float32"

#: The two candidates, in the order every code vector uses.
CANDIDATES: tuple[str, str] = ("sequence_partner", "structure_partner")

#: The two length readings, both computed and both reported.
WINDOWS: tuple[str, str] = ("raw", "length_controlled")

#: The two resampling units this cohort supports. Which one a verdict is read at
#: is a required command-line decision; both are always computed and reported.
RESAMPLING_UNITS: tuple[str, str] = ("anchor_group", "shared_component")

#: Resampling floor, inherited rather than restated (Appendix B rule 12).
MINIMUM_TRIPLE_GROUPS = MINIMUM_BOOTSTRAP_UNITS

#: Fraction of scored candidate legs whose continuation must tokenise to exactly
#: one token per residue **and** keep its tokenisation when spliced onto the
#: prefix. Below this the estimand is undefined on the arm and the arm is refused
#: with its measured figure as the reported result. Same floor and same shape as
#: D3.j's single-symbol coverage gate, for the same reason: L31 measured that a
#: multi-residue vocabulary leaves a position-level construction defined on
#: roughly half of a cohort, and the survivors are the BPE-stable subset rather
#: than a random one.
MINIMUM_TOKEN_ALIGNMENT = 0.99

#: Prior mass, in pseudo-residues, that :class:`PrefixAdaptedComposition` places
#: on the corpus unigram. One observation is the smallest mass that keeps the
#: model defined for a residue the prefix never shows, and it is declared rather
#: than tuned: a larger mass would make the ceiling weaker and is a parameter the
#: verdict would then depend on.
PREFIX_ADAPTED_PRIOR_MASS = 1.0

#: Name of the ceiling member that reads the whole prefix.
PREFIX_COMPOSITION_MEMBER = "prefix_adapted_composition"

#: The order whose prefix advantage is exactly zero by construction. Reported as
#: the curve's own reachability anchor.
REACHABILITY_ORDER = 1

#: The orders the pre-registration names and amendment 3 turned into a curve.
FRAGMENT_ORDERS: tuple[int, ...] = ac.FRAGMENT_ORDERS

#: The rung EXP-R2-214 froze, kept in every curve so the amendment's effect on
#: the verdict is visible rather than asserted.
PRE_REGISTERED_FRAGMENT_ORDER = ac.PRE_REGISTERED_FRAGMENT_ORDER

#: Below this, the ceiling is not doing enough on this estimand for "at least
#: twice the ceiling" to be a bar with teeth. Imported from D3.j rather than
#: restated: it is one declaration of one diagnostic.
CEILING_ADEQUACY_FLOOR = ac.CEILING_ADEQUACY_FLOOR

#: The background directory that is explicitly superseded and must not be read.
#: Named here so a mistaken path is refused by this module rather than producing
#: a plausible curve over a retired count.
SUPERSEDED_BACKGROUND_MARKER = "uniref50_line_local_superseded"

_ALPHABET_SET = frozenset(AA20)


# --------------------------------------------------------------------- cohort


@dataclass(frozen=True)
class Triple:
    """One admitted record: a prefix and the two continuations it must choose between."""

    anchor: str
    prefix: str
    anchor_sequence: str
    partner_ids: dict[str, str]
    continuations: dict[str, str]
    near_duplicate_groups: tuple[int, int, int]
    tm_structure_partner_verified: bool
    tm_ordering_holds: bool
    prefix_composition_closer_to_sequence_partner: bool

    def __post_init__(self) -> None:
        if set(self.continuations) != set(CANDIDATES):
            raise ValueError(f"a triple carries {sorted(self.continuations)}, not {list(CANDIDATES)}")
        if self.continuations[CANDIDATES[0]] == self.continuations[CANDIDATES[1]]:
            raise ValueError(
                f"{self.anchor}: the two candidate continuations are the same string, so "
                "the two hypotheses do not name different records on this triple"
            )
        for label, text in (("prefix", self.prefix), *self.continuations.items()):
            if not text:
                raise ValueError(f"{self.anchor}: {label} is empty")
            outside = sorted(set(text) - _ALPHABET_SET)
            if outside:
                raise ValueError(
                    f"{self.anchor}: {label} carries {outside}, outside the canonical "
                    f"alphabet {AA20}; the corpus background indexes nothing else"
                )


def prefix_length(sequence_length: int, fraction: float) -> int:
    """The residue count the cohort's own prefix rule produces.

    ``round`` and not ``floor(x + 0.5)``: this reproduces the rule the build used,
    which is checked against every stored ``anchor_prefix`` at load rather than
    assumed. The two rules disagree only at an exact half, which is exactly where
    a silent disagreement would move one residue of every odd-length continuation.
    """

    if sequence_length < 1 or not 0.0 < fraction < 1.0:
        raise ValueError("a prefix rule needs a positive length and a fraction in (0, 1)")
    return int(round(sequence_length * fraction))


def load_cohort(path: Path, *, expected_digest: str) -> tuple[list[Triple], dict[str, Any]]:
    """Read the staged triple set, refusing anything that is not the pinned cohort.

    The prefix rule is **re-derived and checked** against every stored
    ``anchor_prefix`` before it is applied to the two partners. The partners'
    continuations are not stored in the cohort -- only the full partner sequences
    are -- so the rule this module applies to them has to be the rule the build
    applied to the anchor, and the only way to know that is to reproduce it.
    """

    path = Path(path)
    digest = sha256_file(path)
    if digest != expected_digest:
        raise RuntimeError(
            f"{path} hashes {digest}, and this track is pre-registered on "
            f"{expected_digest}. A different triple set is a different contradiction "
            "set and needs its own admission"
        )
    manifest_path = path.with_name(f"{path.stem}_manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{manifest_path} does not exist; the cohort's manifest carries the "
            "realised ordering margins, the leakage screen and the declared rules, "
            "and a cohort cited without it is cited without its admission evidence"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stated = manifest.get("artefacts", {}).get("cohort_jsonl", {}).get("sha256")
    if stated != digest:
        raise RuntimeError(
            f"{manifest_path} states the cohort digest is {stated}, the file hashes "
            f"{digest}; the manifest does not describe this file"
        )

    triples: list[Triple] = []
    fractions: set[float] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            fraction = float(record["prefix_fraction"])
            fractions.add(fraction)
            anchor_sequence = record["anchor_sequence"]
            derived = anchor_sequence[: prefix_length(len(anchor_sequence), fraction)]
            if derived != record["anchor_prefix"]:
                raise RuntimeError(
                    f"line {line_number} ({record['anchor']}): the stored prefix is "
                    f"{len(record['anchor_prefix'])} residues and this module's rule "
                    f"derives {len(derived)}. The rule cannot then be applied to the "
                    "partners, whose continuations the cohort does not store"
                )
            continuations = {}
            partner_ids = {}
            for label in CANDIDATES:
                sequence = record[f"{label}_sequence"]
                continuations[label] = sequence[prefix_length(len(sequence), fraction):]
                partner_ids[label] = record[label]
            groups = record["near_duplicate_group"]
            triples.append(
                Triple(
                    anchor=record["anchor"],
                    prefix=record["anchor_prefix"],
                    anchor_sequence=anchor_sequence,
                    partner_ids=partner_ids,
                    continuations=continuations,
                    near_duplicate_groups=(
                        int(groups["anchor"]),
                        int(groups["sequence_partner"]),
                        int(groups["structure_partner"]),
                    ),
                    tm_structure_partner_verified=bool(
                        record["fold"]["structure_partner_at_or_above_same_fold_tm"]
                    ),
                    tm_ordering_holds=bool(record["fold"]["tm_score_ordering_holds"]),
                    prefix_composition_closer_to_sequence_partner=bool(
                        record["prefix_statistics"][
                            "composition_total_variation_to_sequence_partner"
                        ]
                        <= record["prefix_statistics"][
                            "composition_total_variation_to_structure_partner"
                        ]
                    ),
                )
            )
    if not triples:
        raise ValueError(f"{path} carries no records")
    if len(fractions) != 1:
        raise ValueError(
            f"{path} mixes prefix fractions {sorted(fractions)}; the prefix rule has to "
            "be one rule or the two partners' continuations are cut differently"
        )

    census = unit_census(triples)
    record = {
        "path": str(path),
        "sha256": digest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "n_triples": len(triples),
        "resampling_units": census,
        "n_distinct_members": len(
            {t.anchor for t in triples}
            | {t.partner_ids[label] for t in triples for label in CANDIDATES}
        ),
        "prefix_fraction": float(next(iter(fractions))),
        "prefix_rule": "round(len(sequence) * prefix_fraction), verified against every stored anchor_prefix",
        "n_tm_verified_structure_partner": sum(1 for t in triples if t.tm_structure_partner_verified),
        "n_tm_ordering_holds": sum(1 for t in triples if t.tm_ordering_holds),
        "share_prefix_composition_closer_to_sequence_partner": (
            sum(1 for t in triples if t.prefix_composition_closer_to_sequence_partner) / len(triples)
        ),
        "continuation_residues": {
            label: {
                "min": int(min(len(t.continuations[label]) for t in triples)),
                "median": int(np.median([len(t.continuations[label]) for t in triples])),
                "max": int(max(len(t.continuations[label]) for t in triples)),
            }
            for label in CANDIDATES
        },
        "declared_ordering_margins": manifest["distributions"]["ordering_margins"],
        "leakage_identity_at_or_above": manifest["leakage"]["identity_at_or_above"],
        "resampling_unit_note": (
            "L30's rule is the near-duplicate group and never the record. At triple "
            "level that rule has two implementations, they disagree on this cohort, "
            "and both are measured in resampling_units above"
        ),
    }
    return triples, record


def triple_groups(triples: Sequence[Triple], *, unit: str = "anchor_group") -> np.ndarray:
    """Resampling unit per triple, under the declared rule.

    Two rules are implementable on this cohort and they are not equivalent, so the
    choice is a pre-registered command-line decision and both readings reach the
    artefact rather than one being chosen quietly.

    ``anchor_group``
        the anchor's own near-duplicate group. A triple is identified by its
        anchor -- the prefix is the anchor's and the estimand is a statement about
        that prefix -- and this is the unit the cohort's own group-disjoint split
        is built from.
    ``shared_component``
        connected components of the graph joining two triples that share a
        near-duplicate group on **any** leg. Strictly more conservative, and on
        this cohort it is *degenerate*: 199 triples collapse into 21 components of
        which one holds 145, so a percentile interval over 21 units drops 73% of
        the records in 36% of its draws and is bimodal rather than wide. That is a
        property of the cohort's partner reuse, measured here rather than
        discovered afterwards, and it is why it is a reported sensitivity and not
        the default.
    """

    if unit not in RESAMPLING_UNITS:
        raise ValueError(f"unknown resampling unit {unit!r}; units are {list(RESAMPLING_UNITS)}")
    if unit == "anchor_group":
        labels = [triple.near_duplicate_groups[0] for triple in triples]
    else:
        parent: dict[int, int] = {}

        def find(node: int) -> int:
            parent.setdefault(node, node)
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(left: int, right: int) -> None:
            a, b = find(left), find(right)
            if a != b:
                parent[a] = b

        for triple in triples:
            first = triple.near_duplicate_groups[0]
            for other in triple.near_duplicate_groups[1:]:
                union(first, other)
        labels = [find(triple.near_duplicate_groups[0]) for triple in triples]
    unique = {value: index for index, value in enumerate(sorted(set(labels)))}
    return np.asarray([unique[value] for value in labels], dtype=np.int64)


def unit_census(triples: Sequence[Triple]) -> dict[str, Any]:
    """Both units' counts and the sharing that separates them, measured."""

    census: dict[str, Any] = {}
    for unit in RESAMPLING_UNITS:
        groups = triple_groups(triples, unit=unit)
        sizes = np.bincount(groups)
        census[unit] = {
            "n_units": int(sizes.size),
            "largest_unit_triples": int(sizes.max()),
            "largest_unit_share": float(sizes.max() / len(triples)),
            "clears_unit_floor": bool(sizes.size >= MINIMUM_TRIPLE_GROUPS),
        }
    seen: dict[int, int] = {}
    for triple in triples:
        for group in triple.near_duplicate_groups:
            seen[group] = seen.get(group, 0) + 1
    census["n_member_groups"] = len(seen)
    census["n_member_groups_in_more_than_one_triple"] = sum(1 for count in seen.values() if count > 1)
    census["minimum_units"] = int(MINIMUM_TRIPLE_GROUPS)
    return census


# ------------------------------------------------------ rendering and scoring


def render(arm: Arm, sequences: Sequence[str]) -> list[str]:
    """The arm's own rendering of a list of residue strings.

    Delegated to :meth:`src.transfer.arms.Cohort.input_strings` rather than
    reimplemented. There is one declaration of what a ProtGPT2 input looks like
    and one of what a ZymCTRL input looks like, and a stage that spelled either by
    hand would be the second (Appendix B rule 12); ProtGPT2's wrapping alone is
    worth 1.42 nats.
    """

    lengths = [len(s) for s in sequences]
    cohort = Cohort(
        name="fold_discordance_render",
        kind="protein",
        records=list(sequences),
        min_symbols=min(lengths),
        max_symbols=max(lengths),
    )
    return cohort.input_strings(arm)


def conditioning_label_coverage(arm: Arm, n_triples: int) -> dict[str, Any]:
    """Does the cohort define the conditioning label this arm's rendering requires?

    Measured against the cohort rather than decided from the arm's name. An
    EC-conditioned rendering needs an EC number per record; the triple set carries
    an accession, a sequence, a CATH superfamily and a near-duplicate group, and
    no enzyme classification. Scoring such an arm without its tag is off its
    training distribution by 1.73 nats (EXP-R2-034) and scoring it with a fabricated
    tag is worse, so the honest outcome is a refusal with the count.
    """

    required = arm.spec.input_format == "ec_conditioned"
    return {
        "input_format": arm.spec.input_format,
        "requires_conditioning_label": bool(required),
        "n_triples": int(n_triples),
        "n_triples_with_label": 0 if required else int(n_triples),
        "coverage": 0.0 if required else 1.0,
        "field": "ec_labels" if required else None,
        "note": (
            "the triple set carries no enzyme classification, so the conditioning tag "
            "this rendering requires is undefined on every record"
            if required
            else "this rendering carries no conditioning prompt"
        ),
    }


@dataclass(frozen=True)
class ScoringRequest:
    """One ``(context, continuation)`` leg the arm has to score."""

    context: str
    continuation: str


def alignment_census(
    arm: Arm, requests: Sequence[ScoringRequest], *, sample: int
) -> dict[str, Any]:
    """Measured token alignment of the splice, which is what admits an arm.

    Four properties are checked per leg, all on the arm's own renderings: the
    context's tokenisation is a **prefix** of the spliced tokenisation, the spliced
    tokenisation adds **exactly one token per continuation residue**, the
    continuation scored alone adds exactly one token per residue too, and the two
    passes carry **token-for-token identical** continuation spans. All four have to
    hold for "position i of the continuation" to name a residue in
    both the conditioned and the free pass, which is what the estimand differences.
    On a residue-tokenised arm they hold by construction and the census reads
    1.000; on a multi-residue vocabulary they do not, and the measured figure is
    the reported result for that arm rather than a negative.
    """

    if sample < 1:
        raise ValueError("the alignment census needs at least one leg")
    legs = list(requests)[:sample]
    contexts = sorted({leg.context for leg in legs})
    continuations = sorted({leg.continuation for leg in legs})
    rendered_context = dict(zip(contexts, render(arm, contexts))) if contexts else {}
    rendered_free = dict(zip(continuations, render(arm, continuations)))
    rendered_spliced = dict(
        zip(
            [(leg.context, leg.continuation) for leg in legs],
            render(arm, [leg.context + leg.continuation for leg in legs]),
        )
    )

    def ids(text: str) -> list[int]:
        return list(arm.tokenizer(text, return_tensors=None)["input_ids"])

    context_ids = {key: ids(value) for key, value in rendered_context.items()}
    free_ids = {key: ids(value) for key, value in rendered_free.items()}
    spliced_ids = {key: ids(value) for key, value in rendered_spliced.items()}
    empty_ids = ids(render(arm, [""])[0]) if arm.spec.input_format != "raw" else []

    aligned = 0
    prefix_ok = 0
    one_token_per_residue = 0
    free_ok = 0
    same_tokens = 0
    for leg in legs:
        spliced = spliced_ids[(leg.context, leg.continuation)]
        head = context_ids[leg.context]
        free = free_ids[leg.continuation]
        is_prefix = spliced[: len(head)] == head
        adds_one = len(spliced) - len(head) == len(leg.continuation)
        free_one = len(free) - len(empty_ids) == len(leg.continuation)
        # The property the estimand actually differences: the conditioned pass and
        # the free pass must score the SAME token at every continuation position.
        # Equal counts do not imply equal tokens, and a vocabulary that spelled a
        # residue differently after a prefix would pass the three counts above and
        # make every per-position difference a difference between two symbols.
        identical = spliced[len(head) :] == free[len(empty_ids) :]
        prefix_ok += int(is_prefix)
        one_token_per_residue += int(adds_one)
        free_ok += int(free_one)
        same_tokens += int(identical)
        aligned += int(is_prefix and adds_one and free_one and identical)
    total = len(legs)
    return {
        "n_legs_measured": total,
        "n_legs_total": len(requests),
        "context_tokenisation_is_a_prefix": prefix_ok / total,
        "one_token_per_continuation_residue": one_token_per_residue / total,
        "free_pass_one_token_per_residue": free_ok / total,
        "both_passes_score_identical_tokens": same_tokens / total,
        "alignment": aligned / total,
        "n_leading_render_tokens": len(empty_ids),
        "definition": (
            "a leg is aligned when the context's rendered tokenisation is a prefix of "
            "the spliced one, the splice adds exactly one token per continuation "
            "residue, the continuation scored alone does the same, and the two passes "
            "carry token-for-token identical continuation spans. All four are required "
            "for position i of the continuation to name the same residue, and the same "
            "token, in the conditioned and the free pass"
        ),
    }


def admit_arm(census: Mapping[str, Any], arm_name: str, *, minimum: float) -> dict[str, Any]:
    """D3.l's arm gate: by measurement, never by a list."""

    alignment = float(census["alignment"])
    admitted = alignment >= minimum
    return {
        "arm": arm_name,
        "alignment": alignment,
        "minimum": float(minimum),
        "admitted": bool(admitted),
        "reason": (
            f"{alignment:.6g} of scored legs keep one token per residue across the "
            f"splice, at or above the declared {minimum}"
            if admitted
            else (
                f"{alignment:.6g} of scored legs keep one token per residue across the "
                f"splice, below the declared {minimum}. A continuation whose "
                "tokenisation changes when it is spliced is not the same object in the "
                "conditioned and the free pass, so the per-position difference this "
                "estimand takes is undefined on this arm. The measured figure is the "
                "reported result and nothing is computed behind this gate"
            )
        ),
    }


class ResidueSequenceScorer:
    """Per-residue log-probabilities of a continuation, under a context or alone.

    One entry point, ``logprobs``, taking a batch of legs and returning one float
    array per leg with a value for every residue of that leg's continuation. The
    conditioned pass and the free pass differ only in the context string, so both
    go through the same rendering, the same tokenisation and the same gather --
    which is what makes their difference a difference rather than two
    measurements.
    """

    def __init__(self, arm: Arm, *, batch_size: int) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.arm = arm
        self.batch_size = int(batch_size)
        self._forward_tokens = 0
        self._forward_calls = 0

    def cost(self) -> dict[str, int]:
        return {
            "forward_tokens": int(self._forward_tokens),
            "forward_calls": int(self._forward_calls),
        }

    def logprobs(self, requests: Sequence[ScoringRequest]) -> list[np.ndarray]:
        legs = list(requests)
        if not legs:
            raise ValueError("nothing to score")
        contexts = sorted({leg.context for leg in legs})
        rendered_context = dict(zip(contexts, render(self.arm, contexts)))
        spliced = render(self.arm, [leg.context + leg.continuation for leg in legs])
        starts: list[int] = []
        for leg in legs:
            head = self.arm.tokenizer(
                rendered_context[leg.context], return_tensors=None
            )["input_ids"]
            if len(head) < 1:
                raise RuntimeError(
                    "the rendered context tokenises to nothing, so the first "
                    "continuation residue has no position to be predicted from"
                )
            starts.append(len(head))
        out: list[np.ndarray | None] = [None] * len(legs)
        order = sorted(range(len(legs)), key=lambda index: len(spliced[index]))
        for chunk_start in range(0, len(order), self.batch_size):
            chunk = order[chunk_start : chunk_start + self.batch_size]
            texts = [spliced[index] for index in chunk]
            width = max(
                starts[index] + len(legs[index].continuation) for index in chunk
            )
            ids, mask = tokenize_batch(self.arm, texts, width)
            ids = ids.to(self.arm.device)
            mask = mask.to(self.arm.device)
            with torch.inference_mode():
                logits = self.arm.model(input_ids=ids, attention_mask=mask).logits
            self._forward_calls += 1
            self._forward_tokens += int(mask.sum())
            logprob = torch.log_softmax(logits[:, :-1].float(), dim=-1)
            targets = ids[:, 1:]
            gathered = logprob.gather(-1, targets.unsqueeze(-1)).squeeze(-1).cpu().numpy()
            for row, index in enumerate(chunk):
                start = starts[index]
                length = len(legs[index].continuation)
                if start + length > ids.shape[1]:
                    raise RuntimeError(
                        f"leg {index} needs {start + length} tokens and the batch holds "
                        f"{ids.shape[1]}; the rendering truncated the continuation"
                    )
                values = gathered[row, start - 1 : start - 1 + length]
                if values.shape != (length,):
                    raise RuntimeError("the scored span does not match the continuation")
                out[index] = np.asarray(values, dtype=np.float64)
        missing = [index for index, value in enumerate(out) if value is None]
        if missing:
            raise RuntimeError(f"legs {missing} were never scored")
        return [value for value in out if value is not None]


# --------------------------------------------------- the recombination ceiling


def _encode(text: str) -> np.ndarray:
    return np.asarray([AA20.index(symbol) for symbol in text], dtype=np.int64)


class FragmentPrefixConditional:
    """The UniRef50 fragment conditional at one order, as a prefix-advantage model.

    ``P(residue | the previous k - 1 residues)`` as a plug-in maximum-likelihood
    estimate with **no smoothing constant**, exactly as D3.j's ceiling: a position
    is scored only where both the conditioned and the free k-gram were observed,
    and the covered fraction is reported. Smoothing would make the ceiling a
    function of an undeclared parameter precisely where the counts get thin.

    The quantity is the same one the arm is read on: what the anchor's prefix adds
    over scoring the continuation alone. Beyond position ``k - 2`` of the
    continuation the conditioning context lies entirely inside the continuation, so
    the two passes read identical contexts and the advantage is **exactly zero**;
    at k = 1 that is true at every position, which is the curve's reachability
    anchor. Below position ``k - 1`` the free pass backs off to the highest order
    the continuation's own start supports, which is the honest reading of "the same
    model with no prefix".
    """

    def __init__(self, ordered: Mapping[int, ac.OrderedFragmentCounts], *, order: int) -> None:
        self.order = int(order)
        if self.order < 1:
            raise ValueError("a conditional needs a positive order")
        missing = [k for k in range(1, self.order + 1) if k not in ordered]
        if missing:
            raise ValueError(
                f"the k = {self.order} conditional backs off through orders {missing}, "
                "which were not loaded; a backoff over an absent order is not a backoff"
            )
        self.ordered = {k: ordered[k] for k in range(1, self.order + 1)}
        self._counts = {k: ordered[k].counts for k in range(1, self.order + 1)}

    def record(self) -> dict[str, Any]:
        return self.ordered[self.order].record()

    def _logp(self, context: np.ndarray, symbol: int) -> tuple[float, bool]:
        width = len(AA20)
        order = context.size + 1
        counts = self._counts[order]
        base = 0
        for value in context:
            base = base * width + int(value)
        block = np.asarray(counts[base * width : base * width + width], dtype=np.float64)
        total = float(block.sum())
        numerator = float(block[symbol])
        if numerator <= 0.0 or total <= 0.0:
            return 0.0, False
        return math.log(numerator / total), True

    def advantage(self, context: str, continuation: str) -> tuple[np.ndarray, np.ndarray]:
        """Per-residue ``log p(c_i | x, c_<i) - log p(c_i | c_<i)`` and its usable mask."""

        codes = _encode(continuation)
        spliced = _encode(context + continuation)
        offset = len(context)
        values = np.zeros(codes.size, dtype=np.float64)
        usable = np.ones(codes.size, dtype=bool)
        reach = self.order - 1
        for index in range(min(reach, codes.size)):
            position = offset + index
            conditioned_context = spliced[max(0, position - reach) : position]
            free_context = codes[:index]
            if conditioned_context.size == free_context.size:
                # Reachable only when the context string is shorter than the
                # conditional's reach, in which case the prefix adds nothing at
                # this position and the two passes are the same computation.
                continue
            conditioned, ok_conditioned = self._logp(conditioned_context, int(codes[index]))
            free, ok_free = self._logp(free_context, int(codes[index]))
            if not (ok_conditioned and ok_free):
                usable[index] = False
                continue
            values[index] = conditioned - free
        return values, usable

    def reaches(self, offset: int) -> bool:
        """Does this order carry any information across a window starting at ``offset``?"""

        return offset < self.order - 1


class PrefixAdaptedComposition:
    """Confound 3's answer on the statistics side: continue the prefix's composition.

    The plug-in residue distribution of the context, with the corpus unigram as a
    prior of :data:`PREFIX_ADAPTED_PRIOR_MASS` pseudo-residues, scored against the
    corpus unigram itself as the no-prefix pass. It reads nothing but residue
    counts -- no order, no structure, no chemistry -- and it reads the **whole**
    prefix at every position rather than the ``k - 1`` a Markov conditional
    carries, which makes it the only ceiling member that survives a junction
    offset. The cohort measured the prefix as compositionally closer to the
    sequence partner on 94.0% of triples, so this member is where that fact sits
    in the ceiling rather than in a caveat.
    """

    def __init__(self, background: ac.OrderedFragmentCounts, *, prior_mass: float = PREFIX_ADAPTED_PRIOR_MASS) -> None:
        if background.order != 1:
            raise ValueError("the composition member is built from the order-1 background")
        if prior_mass <= 0.0:
            raise ValueError(
                "a non-positive prior mass leaves the model undefined for a residue the "
                "prefix never shows"
            )
        counts = np.asarray(background.counts, dtype=np.float64)
        total = counts.sum()
        if total <= 0.0 or (counts <= 0.0).any():
            raise ValueError("the corpus unigram is not a distribution over all twenty residues")
        self.unigram = counts / total
        self.prior_mass = float(prior_mass)
        self.background = background

    def record(self) -> dict[str, Any]:
        return {
            **self.background.record(),
            "member": PREFIX_COMPOSITION_MEMBER,
            "prior_mass_pseudo_residues": self.prior_mass,
            "definition": (
                "P(residue | prefix) = (count in the prefix + prior_mass * corpus "
                "unigram) / (|prefix| + prior_mass), against the corpus unigram as the "
                "no-prefix pass. Counts come from the context only and never from the "
                "continuation, so the model reads the prefix and nothing else"
            ),
        }

    def advantage(self, context: str, continuation: str) -> tuple[np.ndarray, np.ndarray]:
        codes = _encode(continuation)
        counts = np.bincount(_encode(context), minlength=len(AA20)).astype(np.float64)
        adapted = (counts + self.prior_mass * self.unigram) / (counts.sum() + self.prior_mass)
        values = np.log(adapted[codes]) - np.log(self.unigram[codes])
        return values, np.ones(codes.size, dtype=bool)

    def reaches(self, offset: int) -> bool:
        return True


def load_ceiling(
    directory: Path, orders: Sequence[int], *, pinned: Path
) -> dict[int, ac.OrderedFragmentCounts]:
    """The staged high-order background, digest-checked and refusing the retired one."""

    directory = Path(directory)
    if SUPERSEDED_BACKGROUND_MARKER in str(directory) or SUPERSEDED_BACKGROUND_MARKER in str(pinned):
        raise ValueError(
            f"{SUPERSEDED_BACKGROUND_MARKER} names the background superseded on "
            "2026-08-12; a curve computed over it would be a curve over a retired count"
        )
    needed = sorted(set(int(order) for order in orders) | {1})
    return ac.load_ordered_counts(directory, needed, pinned=pinned)


# ------------------------------------------------------- windows and contrasts


def window_indices(
    lengths: Mapping[str, int], *, mode: str, offset: int, minimum: int
) -> dict[str, np.ndarray] | None:
    """The scored positions of each candidate, or ``None`` when the triple is dropped.

    ``raw`` gives each candidate its own positions from ``offset`` to its end;
    ``length_controlled`` gives both the same positions, from ``offset`` to the
    shorter candidate's end, so the two means are taken over the same count of
    positions **and** the same indices.
    """

    if mode not in WINDOWS:
        raise ValueError(f"unknown window {mode!r}; windows are {list(WINDOWS)}")
    if offset < 0 or minimum < 1:
        raise ValueError("offset must be non-negative and the minimum window positive")
    if mode == "length_controlled":
        limit = min(lengths[label] for label in CANDIDATES)
        spans = {label: (offset, limit) for label in CANDIDATES}
    else:
        spans = {label: (offset, lengths[label]) for label in CANDIDATES}
    if any(stop - start < minimum for start, stop in spans.values()):
        return None
    return {label: np.arange(start, stop, dtype=np.int64) for label, (start, stop) in spans.items()}


def _mean(_: np.ndarray, prediction: np.ndarray) -> float:
    """The estimand's aggregate over triples: a plain mean of per-triple contrasts."""

    return float(np.mean(np.asarray(prediction, dtype=np.float64)))


def contrast_interval(
    values: Sequence[float],
    groups: Sequence[int],
    *,
    seed: int,
    draws: int,
    reference: Sequence[float] | None = None,
    reference_name: str = "zero",
) -> dict[str, Any]:
    """Mean per-triple contrast, and its difference from a reference, one draw set.

    Both vectors are scored on the same resampled groups by
    :func:`src.transfer.statistics.paired_group_bootstrap`, which is what the
    ceiling comparison needs: the arm's contrast and the ceiling's are means over
    the *same* triples, so their difference has to be resampled together. With no
    reference the right-hand vector is zero, which gives the arm's own contrast its
    own interval. No new resampler is introduced here and none may be: the package
    has one, and ``tests/test_transfer_audit_invariants.py`` enforces that.
    """

    left = np.asarray(values, dtype=np.float64)
    right = np.zeros_like(left) if reference is None else np.asarray(reference, dtype=np.float64)
    unit = np.asarray(groups)
    if left.ndim != 1 or right.shape != left.shape or unit.shape != left.shape:
        raise ValueError("contrast, reference and group vectors must align triple for triple")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("a per-triple contrast is non-finite")
    floor = bootstrap_unit_floor(int(np.unique(unit).size))
    block: dict[str, Any] = {
        "contrast": float(left.mean()),
        "reference_contrast": float(right.mean()),
        "reference_name": reference_name,
        "n_triples": int(left.size),
        "n_triples_positive": int((left > 0).sum()),
        "n_triples_negative": int((left < 0).sum()),
        "unit_floor": floor,
        "resampling_unit": "near-duplicate component of triples",
        "definition": (
            "P = A(structure partner) - A(sequence partner), where A(c) is the "
            "per-scored-residue prefix advantage log p(c | x) - log p(c) of that "
            "candidate. P > 0 follows the structure partner, P < 0 the sequence partner"
        ),
    }
    if floor["degenerate"]:
        block["difference"] = block["contrast"] - block["reference_contrast"]
        block["difference_ci95"] = None
        block["interval_withheld_reason"] = floor["degenerate_reason"]
        return block
    resampled = paired_group_bootstrap(
        np.zeros_like(left), left, right, unit, _mean, seed=seed, n_bootstrap=draws
    )
    block["difference"] = resampled["difference"]
    block["difference_ci95"] = resampled["difference_ci95"]
    block["group_draws"] = {
        key: resampled[key]
        for key in (
            "n_bootstrap_requested",
            "n_finite_draws",
            "n_non_finite_draws",
            "n_groups",
            "minimum_groups",
        )
    }
    return block


def sign_permutation_contrast(
    values: Sequence[float], *, seed: int, draws: int
) -> dict[str, Any]:
    """The false-positive control: relabel which candidate is the structural one.

    An independent seeded sign per triple, drawn ``draws`` times, which is the
    exact null of "the two candidates carry no label". Two things are read off it
    and they are different questions. The null must be **centred at zero** -- it is
    so by construction, and an interval that is not is a structural asymmetry
    between the two candidate slots rather than a fact about any model, which is
    the defect this control exists to catch. And the observed contrast is compared
    against the null's own upper tail, which is a *detection* statistic and admits
    nothing on its own under §7.0.

    A single permutation is not this test. One draw's interval excludes zero about
    as often as any 95% interval does, so the check would fire at chance and mean
    nothing; the null is a distribution and is read as one.
    """

    left = np.asarray(values, dtype=np.float64)
    if draws < 20:
        raise ValueError("a null reported as a distribution needs draws to be a distribution")
    rng = np.random.default_rng(seed)
    means = np.asarray(
        [
            float((left * rng.choice(np.asarray([-1.0, 1.0]), size=left.size)).mean())
            for _ in range(draws)
        ]
    )
    low, high = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    observed = float(left.mean())
    return {
        "observed_contrast": observed,
        "null_median": float(np.median(means)),
        "null_ci95": [low, high],
        "null_q95": float(np.percentile(means, 95.0)),
        "n_draws": int(draws),
        "observed_above_null_q95": bool(observed > float(np.percentile(means, 95.0))),
        "two_sided_share_at_or_beyond_observed": float(
            (np.abs(means) >= abs(observed)).mean()
        ),
        "fires": bool(not low <= 0.0 <= high),
        "criterion": (
            "the null fires when the distribution of the contrast under a random "
            "relabelling of the two candidates does not cover zero. It is centred at "
            "zero by construction, so a shift is an asymmetry between the two candidate "
            "slots and not a fact about the model. observed_above_null_q95 is reported "
            "beside it as a detection statistic and admits nothing on its own"
        ),
    }


def ceiling_margin(
    *,
    arm_block: Mapping[str, Any],
    against_block: Mapping[str, Any],
    factor: float,
) -> dict[str, Any]:
    """§7.0's standing margin on this estimand, clause by clause and named.

    ``reference_contrast`` and not ``contrast``: both live on ``against_block``,
    because one paired draw set scores the arm and the ceiling together, and
    ``contrast`` there is the **arm's**. Reading the wrong one turns the clause
    into ``P >= factor * P``, which no positive effect satisfies -- a margin that
    fails everything and looks like a result.

    ``twice the ceiling`` is taken against the ceiling's **positive part**. The
    standing rule is written for an excess over chance, which is non-negative; this
    contrast is signed and the corpus account predicts it negative, so doubling a
    negative number would weaken the clause exactly as the corpus account held more
    strongly. The clamp is declared, and ``multiplicative_clause_binds`` reports
    whether it left the clause doing any work at all.
    """

    arm = float(arm_block["contrast"])
    ceiling = float(against_block["reference_contrast"])
    interval = against_block.get("difference_ci95")
    clauses = {
        "contrast_positive": arm > 0.0,
        "difference_interval_excludes_zero_from_above": bool(
            interval is not None and interval[0] > 0.0
        ),
        "at_least_factor_times_ceiling": arm >= factor * max(ceiling, 0.0),
    }
    return {
        "arm_contrast": arm,
        "ceiling_contrast": ceiling,
        "difference": float(against_block["difference"]),
        "difference_ci95": interval,
        "factor": float(factor),
        "clauses": clauses,
        "cleared": all(clauses.values()),
        "multiplicative_clause_binds": bool(ceiling > 0.0),
        "multiplicative_clause_note": (
            "the ceiling's own contrast is at or below zero, so 'at least the declared "
            "factor times the ceiling' reduces to 'greater than zero' and that clause "
            "carries no weight here. The clauses doing the work are the sign of the "
            "contrast and the interval of the arm-minus-ceiling difference"
            if ceiling <= 0.0
            else "the ceiling's own contrast is positive, so the multiplicative clause binds"
        ),
        "rule": (
            "audit §7.0's standing margin: the paired group-bootstrap 95% interval of "
            "the arm-minus-ceiling difference excludes zero over at least "
            f"{MINIMUM_TRIPLE_GROUPS} groups, and the arm's own contrast is at least "
            f"{factor} times the ceiling's positive part"
        ),
    }


def first_binding_order(curve: Mapping[str, Mapping[str, Any]]) -> int | str | None:
    """The lowest ceiling member whose adequacy ratio reaches the declared floor."""

    def key(name: str) -> tuple[int, str]:
        return (0, name.zfill(4)) if name.isdigit() else (1, name)

    for name in sorted(curve, key=key):
        adequacy = curve[name].get("adequacy")
        if adequacy is not None and adequacy.get("adequate"):
            return int(name) if name.isdigit() else name
    return None


def fold_verdict(
    *,
    margin: Mapping[str, Any],
    arm_block: Mapping[str, Any],
    nulls: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """D3.l's pre-declared branch table, under §7.0's halt-and-classify clause."""

    contrast = float(arm_block["contrast"])
    interval = arm_block.get("difference_ci95")
    fired = sorted(name for name, block in nulls.items() if block.get("fires"))
    if fired:
        return {
            "verdict": "VOID_NULL_FIRED",
            "reason": (
                f"{fired} fired. A contrast that survives a random relabelling of the "
                "two candidates, or one that survives a composition-preserving "
                "permutation of the anchor's prefix, is not a statement about which "
                "partner the prefix prefers, and no verdict is read from it"
            ),
            "contrast": contrast,
            "nulls_fired": fired,
        }
    if margin["cleared"]:
        return {
            "verdict": "STRUCTURE_CANDIDATE",
            "reason": (
                "the continuation follows the partner that carries the anchor's fold, "
                "and the effect clears the recombination ceiling under the standing "
                "margin. §7.0 clause 5: this licenses a candidate and nothing more, and "
                "§8's causal, retrieval-aware and independent-biological clauses stay open"
            ),
            "contrast": contrast,
            "nulls_fired": [],
        }
    if interval is not None and interval[1] < 0.0:
        return {
            "verdict": "RECOMBINATION",
            "reason": (
                "the continuation follows the composition-matched sequence partner, with "
                "the interval excluding zero from below. This is the pre-declared "
                "'clears the ceiling toward c_seq' branch and is recorded as "
                "recombination explicitly; the line halts rather than being narrowed"
            ),
            "contrast": contrast,
            "nulls_fired": [],
        }
    if contrast > 0.0:
        return {
            "verdict": "INSIDE_CEILING",
            "reason": (
                "the contrast is toward the structure partner but does not clear the "
                "recombination ceiling under the standing margin. §7.0 clause 5 "
                "classifies this as recombination and halts the line; it is not a weak "
                "or partial structural result"
            ),
            "contrast": contrast,
            "failed_clauses": [name for name, held in margin["clauses"].items() if not held],
            "nulls_fired": [],
        }
    return {
        "verdict": "UNDECIDED",
        "reason": (
            "the contrast does not separate the two accounts at this cohort size, this "
            "window and this offset: the interval spans zero and the ceiling is not cleared"
        ),
        "contrast": contrast,
        "nulls_fired": [],
    }


# ------------------------------------------------------ known-answer self-test

#: The three worlds the self-test plants. ``sequence_statistics`` and
#: ``structure`` are the two accounts the design exists to separate and must come
#: back with opposite verdicts; ``neither`` is the world whose decoder reads no
#: context at all, where the contrast is exactly zero and neither null may fire.
PLANTINGS: tuple[str, str, str] = ("sequence_statistics", "structure", "neither")

#: Period of the synthetic fold signature. The marker residue sits at every
#: ``SYNTHETIC_PERIOD``-th position, so two folds differing only in **phase**
#: carry identical marker frequency: the signature is arrangement and not
#: composition, which is what makes the structure world unreachable by any
#: composition model.
SYNTHETIC_PERIOD = 5

#: Residue carrying the synthetic fold signature.
SYNTHETIC_MARKER_RESIDUE = "W"


class _ResidueTokenizer:
    """A residue-per-token vocabulary, and a paired one for the refusal path.

    The paired variant exists because the arm gate has to be exercised against a
    tokenisation that genuinely breaks the estimand rather than against a mock of
    one. Two residues per token is the smallest vocabulary on which "position i of
    the continuation" stops naming a residue.
    """

    def __init__(self, *, paired: bool = False) -> None:
        self.paired = bool(paired)
        self.symbols = list(AA20)
        self.marker_id = len(self.symbols)
        self.pad_token_id = len(self.symbols) + 1
        self.vocab_size = len(self.symbols) + 2

    def __call__(self, text: str, return_tensors: Any = None) -> dict[str, list[int]]:
        if return_tensors is not None:
            raise ValueError("this tokenizer only returns python lists")
        ids: list[int] = []
        index = 0
        while index < len(text):
            symbol = text[index]
            if symbol == "1":
                ids.append(self.marker_id)
                index += 1
                continue
            if self.paired and index + 1 < len(text) and text[index + 1] != "1":
                # One token for two residues: the id is the first residue's, which
                # is enough to make the leg unaligned and is never scored.
                ids.append(self.symbols.index(symbol))
                index += 2
                continue
            ids.append(self.symbols.index(symbol))
            index += 1
        return {"input_ids": ids}


@dataclass
class _Logits:
    logits: torch.Tensor


class _SyntheticDecoder(torch.nn.Module):
    """A decoder whose next-residue distribution implements one planted account.

    Not a transformer and not claimed to be one. It is a real ``forward`` over
    real token ids, so the rendering, the tokenisation, the alignment census, the
    batching, the log-softmax and the gather are all exercised end to end; what it
    is not is evidence that the same analysis behaves identically through
    thirty-six blocks.
    """

    def __init__(self, planted: str, unigram: np.ndarray, *, vocab_size: int, prior_mass: float) -> None:
        super().__init__()
        if planted not in PLANTINGS:
            raise ValueError(f"unknown planting {planted!r}; plantings are {list(PLANTINGS)}")
        self.planted = planted
        self.prior_mass = float(prior_mass)
        self.vocab_size = int(vocab_size)
        self.marker_index = AA20.index(SYNTHETIC_MARKER_RESIDUE)
        self.register_buffer("log_unigram", torch.log(torch.tensor(unigram, dtype=torch.float32)))
        # A weight so that ``.parameters()`` is non-empty and the module reports a
        # device the way a loaded arm does.
        self.scale = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> _Logits:
        batch, width = input_ids.shape
        device = input_ids.device
        base = torch.full((batch, width, self.vocab_size), -30.0, device=device)
        base[:, :, : len(AA20)] = self.log_unigram.to(device)
        if self.planted == "neither":
            return _Logits(base)
        residue = input_ids.clamp(max=len(AA20) - 1)
        is_residue = (input_ids < len(AA20)).float()
        if self.planted == "sequence_statistics":
            one_hot = torch.zeros(batch, width, len(AA20), device=device)
            one_hot.scatter_(2, residue.unsqueeze(-1), is_residue.unsqueeze(-1))
            counts = one_hot.cumsum(dim=1)
            seen = is_residue.cumsum(dim=1).unsqueeze(-1)
            prior = self.prior_mass * self.log_unigram.to(device).exp()
            adapted = (counts + prior) / (seen + self.prior_mass)
            base[:, :, : len(AA20)] = torch.log(adapted)
            return _Logits(base)
        # ``structure``: read the phase of the marker residue, which is a property
        # of arrangement and carries no composition information at all.
        period = SYNTHETIC_PERIOD
        positions = torch.arange(width, device=device)
        phase = ((positions - 1) % period).clamp(min=0)
        phase_one_hot = torch.zeros(batch, width, period, device=device)
        phase_one_hot.scatter_(2, phase.view(1, width, 1).expand(batch, width, 1), 1.0)
        is_marker = ((residue == self.marker_index).float() * is_residue).unsqueeze(-1)
        marker_by_phase = (phase_one_hot * is_marker).cumsum(dim=1)
        predicted_phase = marker_by_phase.argmax(dim=2)
        evidence = marker_by_phase.max(dim=2).values
        next_phase = ((positions + 1 - 1) % period).view(1, width).expand(batch, width)
        agrees = (predicted_phase == next_phase).float() * (evidence > 0).float()
        boost = 3.0 * (2.0 * agrees - 1.0)
        base[:, :, self.marker_index] = base[:, :, self.marker_index] + boost
        return _Logits(base)


@dataclass(frozen=True)
class SyntheticWorld:
    """One planted world, with everything the stage needs to read it."""

    planted: str
    triples: list[Triple]
    arm: Arm
    ceiling: dict[int, ac.OrderedFragmentCounts]
    settings: dict[str, Any]


def _synthetic_arm(planted: str, unigram: np.ndarray, *, paired: bool, device: str) -> Arm:
    tokenizer = _ResidueTokenizer(paired=paired)
    spec = ArmSpec(
        name=f"synthetic-{planted}",
        path=Path("/nonexistent/synthetic"),
        path_variable="TRANSFER_MODEL_BASE_DIR",
        modality="protein",
        n_layer=1,
        d_model=len(AA20),
        tokenisation="multi_residue_bpe" if paired else "residue",
        input_format="n_to_c_control",
        evaluation_cohort_source="synthetic",
        architecture="synthetic",
        pretraining_corpus="synthetic",
    )
    model = _SyntheticDecoder(
        planted, unigram, vocab_size=tokenizer.vocab_size, prior_mass=PREFIX_ADAPTED_PRIOR_MASS
    ).to(device)
    model.eval()
    return Arm(spec=spec, model=model, tokenizer=tokenizer, device=device, dtype=DTYPE)


def _draw_sequence(
    length: int, *, weights: np.ndarray, phase: int, start: int, rng: np.random.Generator
) -> str:
    """Residues from ``weights``, with the marker forced onto one phase class.

    The marker's density is ``1 / SYNTHETIC_PERIOD`` whatever the phase, so two
    folds differ in arrangement and not in composition.
    """

    marker = AA20.index(SYNTHETIC_MARKER_RESIDUE)
    others = np.asarray([index for index in range(len(AA20)) if index != marker])
    other_weights = weights[others] / weights[others].sum()
    symbols: list[str] = []
    for index in range(length):
        if (start + index) % SYNTHETIC_PERIOD == phase:
            symbols.append(AA20[marker])
        else:
            symbols.append(AA20[int(rng.choice(others, p=other_weights))])
    return "".join(symbols)


def _synthetic_counts(sequences: Sequence[str], orders: Sequence[int]) -> dict[int, ac.OrderedFragmentCounts]:
    """k-mer counts over the synthetic universe, in the same layout as the staged one."""

    width = len(AA20)
    loaded: dict[int, ac.OrderedFragmentCounts] = {}
    for order in sorted(set(int(value) for value in orders)):
        counts = np.zeros(width**order, dtype=np.int64)
        for sequence in sequences:
            codes = _encode(sequence)
            if codes.size < order:
                continue
            index = np.zeros(codes.size - order + 1, dtype=np.int64)
            for step in range(order):
                index = index * width + codes[step : codes.size - order + 1 + step]
            counts += np.bincount(index, minlength=width**order)
        loaded[order] = ac.OrderedFragmentCounts(
            order=order,
            counts=counts,
            source="synthetic universe",
            sha256="synthetic",
            observed=int((counts > 0).sum()),
            possible=width**order,
            total_kmers=int(counts.sum()),
        )
    return loaded


def synthetic_world(
    *,
    planted: str,
    seed: int,
    device: str = "cpu",
    n_triples: int = 32,
    continuation_residues: int = 60,
    length_jitter: int = 12,
    prefix_residues: int = 60,
    universe_records: int = 400,
    ceiling_orders: Sequence[int] = (1, 2, 3),
    paired_tokenisation: bool = False,
) -> SyntheticWorld:
    """A world whose answer is known, built through the same objects as a campaign.

    Every triple mirrors the real cohort's construction: the sequence partner's
    continuation is drawn from the anchor's own residue distribution and carries a
    **different** fold phase, and the structure partner's continuation carries the
    anchor's fold phase and a **different** residue distribution. A composition
    model therefore names the sequence partner and a fold model names the
    structure partner on every record, which is §7.0 clause 4's requirement made
    exact.
    """

    if planted not in PLANTINGS:
        raise ValueError(f"unknown planting {planted!r}; plantings are {list(PLANTINGS)}")
    if prefix_residues % SYNTHETIC_PERIOD:
        raise ValueError(
            "the prefix length must be a multiple of the fold period, or the phase "
            "does not survive the splice and the plant is not the plant"
        )
    if n_triples < MINIMUM_TRIPLE_GROUPS:
        raise ValueError(f"a world needs at least {MINIMUM_TRIPLE_GROUPS} resampling units")
    rng = np.random.default_rng(seed)
    anchor_weights = rng.dirichlet(np.full(len(AA20), 2.0))
    partner_weights = rng.dirichlet(np.full(len(AA20), 2.0))
    other_phase = SYNTHETIC_PERIOD // 2

    triples: list[Triple] = []
    universe: list[str] = []
    for index in range(n_triples):
        prefix = _draw_sequence(
            prefix_residues, weights=anchor_weights, phase=0, start=0, rng=rng
        )
        # The two candidates differ in length, as they do in the real cohort, so
        # the raw and the length-controlled readings are genuinely different
        # aggregations of the same scores rather than the same one twice.
        sequence_length = continuation_residues + int(rng.integers(0, length_jitter + 1))
        structure_length = continuation_residues - int(rng.integers(0, length_jitter + 1))
        tail = _draw_sequence(
            continuation_residues, weights=anchor_weights, phase=0, start=prefix_residues, rng=rng
        )
        sequence_continuation = _draw_sequence(
            sequence_length,
            weights=anchor_weights,
            phase=other_phase,
            start=prefix_residues,
            rng=rng,
        )
        structure_continuation = _draw_sequence(
            structure_length,
            weights=partner_weights,
            phase=0,
            start=prefix_residues,
            rng=rng,
        )
        triples.append(
            Triple(
                anchor=f"SYN{index:04d}",
                prefix=prefix,
                anchor_sequence=prefix + tail,
                partner_ids={
                    "sequence_partner": f"SEQ{index:04d}",
                    "structure_partner": f"STR{index:04d}",
                },
                continuations={
                    "sequence_partner": sequence_continuation,
                    "structure_partner": structure_continuation,
                },
                near_duplicate_groups=(3 * index, 3 * index + 1, 3 * index + 2),
                tm_structure_partner_verified=True,
                tm_ordering_holds=True,
                prefix_composition_closer_to_sequence_partner=True,
            )
        )
        universe.extend([prefix + tail, prefix + sequence_continuation, prefix + structure_continuation])
    for _ in range(universe_records):
        weights = anchor_weights if rng.random() < 0.5 else partner_weights
        phase = 0 if rng.random() < 0.5 else other_phase
        universe.append(
            _draw_sequence(
                prefix_residues + continuation_residues,
                weights=weights,
                phase=phase,
                start=0,
                rng=rng,
            )
        )

    ceiling = _synthetic_counts(universe, sorted(set(ceiling_orders) | {1}))
    unigram = np.asarray(ceiling[1].counts, dtype=np.float64)
    unigram = unigram / unigram.sum()
    arm = _synthetic_arm(planted, unigram, paired=paired_tokenisation, device=device)
    settings = {
        "planted": planted,
        "seed": int(seed),
        "n_triples": int(n_triples),
        "prefix_residues": int(prefix_residues),
        "continuation_residues": int(continuation_residues),
        "length_jitter": int(length_jitter),
        "universe_records": len(universe),
        "fold_period": SYNTHETIC_PERIOD,
        "fold_signature": (
            f"residue {SYNTHETIC_MARKER_RESIDUE} at every {SYNTHETIC_PERIOD}-th position; the "
            "two folds differ in phase and not in marker density, so the signature is "
            "arrangement and carries no composition information"
        ),
        "ceiling_orders": sorted(set(int(order) for order in ceiling_orders) | {1}),
        "tokenisation": "paired" if paired_tokenisation else "residue",
        "expected": {
            "sequence_statistics": "RECOMBINATION",
            "structure": "STRUCTURE_CANDIDATE",
            "neither": "UNDECIDED",
        }[planted],
    }
    return SyntheticWorld(planted=planted, triples=triples, arm=arm, ceiling=ceiling, settings=settings)


# ------------------------------- §7.0 clause 1's profile members, and the Potts gap

#: The two profile members. §7.0 clause 1 names "profile-HMM scores" as part of the
#: recombination family, and the fragment and composition members do not reach it:
#: a k-mer conditional cannot detect homology below the fragment scale and a
#: composition channel cannot detect it at all. These two can, which is why they
#: are the members that decide whether a contrast toward the structure partner is
#: structural knowledge or remote homology. The structure partner shares the
#: anchor's CATH superfamily **by construction**, so this is not a hypothetical.
PROFILE_MEMBERS: tuple[str, str] = ("pfam_profile", "corpus_profile")

#: Bits to nats. A HMMER bit score is ``log2 P(sequence | profile) / P(sequence |
#: null)`` against a background-composition null, which is the same shape as this
#: estimand's ``log p(c | x) - log p(c)``: a log-likelihood ratio of the same
#: residues under a context-informed model and a context-free one. The two nulls
#: are not the same object -- HMMER's is a fixed composition model and the arm's is
#: the arm scoring the continuation alone -- and the correspondence is declared
#: rather than asserted to be exact. It is also the reason the member needs no
#: separate free pass: with no prefix there is no profile, so its advantage is
#: exactly zero, which is the same reachability property k = 1 has.
NATS_PER_BIT = math.log(2.0)

#: How a prefix selects its profiles: Pfam's own curated gathering threshold. Not
#: a number chosen here, which is the point -- a tuned inclusion cut would make the
#: ceiling a function of an undeclared parameter exactly where it decides a verdict.
PROFILE_SELECTION_ARGUMENT = "--cut_ga"

#: How the candidates are then scored under those profiles: with HMMER's heuristic
#: filters **off**. The filters exist to make a database search fast and they cost
#: sensitivity, and a filtered-out remote homologue would read as a zero here --
#: which would understate the ceiling and flatter the model. The scoring pass runs
#: over the handful of profiles the prefixes selected, so turning the filters off
#: is affordable.
PROFILE_SCORING_ARGUMENTS: tuple[str, ...] = ("--max",)

#: Reporting cut for the scoring pass. Permissive on purpose, for the same reason:
#: the quantity wanted is the raw bit score, not a thresholded hit list.
PROFILE_SCORING_EVALUE = 1000.0

#: jackhmmer iterations for the corpus profile. Three is the usual profile-building
#: depth and is declared rather than swept: more iterations recruit more remote
#: homologues and would raise the ceiling, so this setting bounds the member from
#: below and the direction of that bound is stated in the artefact.
CORPUS_PROFILE_ITERATIONS = 3

#: What is removed from the recruited alignment before the corpus profile is built.
#: The two candidate accessions and nothing else. If a candidate were left in, the
#: member would score a sequence against a profile that contains it, which is a
#: lookup of the answer rather than a prediction from the corpus. The anchor's own
#: Swiss-Prot entry is deliberately **kept**: it is not a candidate, the model saw
#: it in training too, and removing it would weaken the ceiling for no reason.
CORPUS_PROFILE_EXCLUDES = "the two candidate accessions"

#: Why no Potts / MSA-coupling member is built, stated as a measurement rather than
#: as an omission. Clause 1 names couplings beside profiles, and the honest position
#: is that on **this** readout they are not the binding member and cannot be:
#: the readout is homology detection -- does the prefix's family contain this
#: continuation -- and a profile HMM is what that family's state of the art is.
#: Pairwise couplings add discriminative power for contact prediction and for
#: fitness, both of which are *within*-family quantities read at fixed alignment;
#: they add very little to the detection decision, which is why every homology
#: search tool in use is a profile method and none is a Potts method. A coupling
#: member would also have to be fitted per triple from the same jackhmmer alignment
#: the profile member already uses, so it would inherit that alignment's recall and
#: could only re-weight what the profile already found. The cost was measured and is
#: not the reason: one jackhmmer over the staged Swiss-Prot takes 5 s, so 199
#: plmDCA fits over those alignments would be hours of CPU and reachable. It is not
#: built because it would not change which member binds, and that claim is checkable
#: against the profile members' own adequacy ratios reported beside it.
POTTS_MEMBER_ABSENT = (
    "no Potts / MSA-coupling member is built. Clause 1 names couplings beside "
    "profiles, and on this readout -- does the anchor's prefix have detectable "
    "remote homology to this continuation -- the profile HMM is the family's state "
    "of the art and the coupling term is not what decides detection. A coupling "
    "member would be fitted from the same jackhmmer alignment the corpus profile "
    "member already uses, so it inherits that alignment's recall and can only "
    "re-weight what the profile already found. Cost is not the reason and was "
    "measured: one jackhmmer over the staged Swiss-Prot is 5 s, so 199 fits are "
    "reachable. The claim that it would not become the binding member is checkable "
    "against the two profile members' adequacy ratios in this artefact"
)

_HMMER_BINARIES = ("hmmsearch", "hmmfetch", "hmmbuild", "jackhmmer")


@dataclass(frozen=True)
class HmmerTool:
    """A verified HMMER installation and the provenance needed to reproduce it."""

    directory: Path
    version: str
    digests: dict[str, str]

    def path(self, program: str) -> Path:
        if program not in _HMMER_BINARIES:
            raise ValueError(f"{program!r} is not one of {list(_HMMER_BINARIES)}")
        return self.directory / program

    def record(self) -> dict[str, Any]:
        return {
            "directory": str(self.directory),
            "version": self.version,
            "sha256": dict(self.digests),
            "programs": list(_HMMER_BINARIES),
        }


def prepare_hmmer(directory: Path) -> HmmerTool:
    """Verify a staged HMMER build and hash every binary this stage will run.

    Mirrors ``homology.prepare_diamond``: the aligner is part of the measurement,
    so which build produced a number has to be recoverable from the artefact. The
    version is read back from the binary rather than taken from the directory name.
    """

    directory = Path(directory)
    digests: dict[str, str] = {}
    for program in _HMMER_BINARIES:
        executable = directory / program
        if not executable.is_file():
            raise FileNotFoundError(
                f"{executable} does not exist; build HMMER from the staged "
                "external_resources/tools/hmmer-3.4.tar.gz and point --hmmer-bin at "
                "its bin directory"
            )
        digests[program] = sha256_file(executable)
    completed = subprocess.run(
        [str(directory / "hmmsearch"), "-h"], capture_output=True, text=True, check=True
    )
    match = re.search(r"HMMER (\d[\w.]*)", completed.stdout)
    if match is None:
        raise RuntimeError(f"cannot parse a HMMER version from {completed.stdout[:200]!r}")
    return HmmerTool(directory=directory, version=match.group(1), digests=digests)


@dataclass(frozen=True)
class DomainHit:
    """One reported domain: which profile, on which target, where, and how strong."""

    target: str
    profile: str
    score: float
    ali_from: int
    ali_to: int


def parse_domtblout(path: Path) -> list[DomainHit]:
    """Read HMMER's per-domain table.

    Columns are positional and documented in the HMMER user guide: 1 target, 4
    query profile, 14 domain bit score, 18-19 the alignment's span on the target.
    Parsed by index rather than by header because the header is a comment line and
    the description field at the end contains spaces.
    """

    hits: list[DomainHit] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split()
            if len(fields) < 19:
                raise ValueError(f"{path} has a row with {len(fields)} fields, fewer than 19")
            hits.append(
                DomainHit(
                    target=fields[0],
                    profile=fields[3],
                    score=float(fields[13]),
                    ali_from=int(fields[17]),
                    ali_to=int(fields[18]),
                )
            )
    return hits


class ProfileHomologyMember:
    """§7.0 clause 1's profile-HMM member, as a prefix-advantage model.

    The construction, declared before any candidate was scored:

    1. the **anchor's prefix** selects a set of profiles -- Pfam-A entries at
       Pfam's own gathering threshold, or the single profile a jackhmmer search of
       the staged corpus from that prefix builds;
    2. each **candidate continuation** is then scored under exactly that set, with
       HMMER's heuristic filters off, and the raw per-domain bit score is kept;
    3. a domain contributes ``ln(2) * score / span`` nats to every residue of the
       span it aligns, and a residue takes the **maximum** over the domains
       covering it, or zero. The maximum rather than the sum because §7.0's
       ceiling is the best a member of the family achieves and not an ensemble of
       overlapping ones, and clamped at zero because a negative bit score means
       the profile fits worse than the composition null, which no detector would use.

    A prefix that selects no profile gives exactly zero for both candidates, so the
    member's contrast on that triple is exactly zero. That is the same reachability
    property the k = 1 fragment rung has, and it is reported as coverage rather
    than hidden: the member is *evaluable* everywhere and *informative* only where
    the prefix has detectable family membership.
    """

    def __init__(
        self,
        name: str,
        *,
        selected: Mapping[str, frozenset[str]],
        hits: Mapping[str, Sequence[DomainHit]],
        provenance: Mapping[str, Any],
    ) -> None:
        if name not in PROFILE_MEMBERS:
            raise ValueError(f"unknown profile member {name!r}; members are {list(PROFILE_MEMBERS)}")
        self.name = name
        self.selected = {key: frozenset(value) for key, value in selected.items()}
        self.hits = {key: tuple(value) for key, value in hits.items()}
        self.provenance = dict(provenance)

    def profiles_for(self, context: str) -> frozenset[str]:
        return self.selected.get(context, frozenset())

    def advantage(self, context: str, continuation: str) -> tuple[np.ndarray, np.ndarray]:
        values = np.zeros(len(continuation), dtype=np.float64)
        usable = np.ones(len(continuation), dtype=bool)
        profiles = self.profiles_for(context)
        if not profiles:
            return values, usable
        for hit in self.hits.get(continuation, ()):
            if hit.profile not in profiles:
                continue
            span = hit.ali_to - hit.ali_from + 1
            if span < 1 or hit.ali_to > len(continuation):
                raise ValueError(
                    f"{self.name}: a domain spans {hit.ali_from}-{hit.ali_to} of a "
                    f"{len(continuation)}-residue continuation"
                )
            per_residue = NATS_PER_BIT * hit.score / span
            if per_residue <= 0.0:
                continue
            window = slice(hit.ali_from - 1, hit.ali_to)
            values[window] = np.maximum(values[window], per_residue)
        return values, usable

    def reaches(self, offset: int) -> bool:
        """A profile reads the whole prefix, so no junction offset silences it."""

        return True

    def record(self) -> dict[str, Any]:
        return {
            **self.provenance,
            "member": self.name,
            "n_contexts_with_a_profile": sum(1 for value in self.selected.values() if value),
            "n_contexts": len(self.selected),
            "context_coverage": (
                sum(1 for value in self.selected.values() if value) / len(self.selected)
                if self.selected
                else 0.0
            ),
            "n_distinct_profiles": len({p for value in self.selected.values() for p in value}),
            "selection_argument": PROFILE_SELECTION_ARGUMENT,
            "scoring_arguments": list(PROFILE_SCORING_ARGUMENTS),
            "definition": (
                "the anchor's prefix selects the profiles; each candidate continuation "
                "is scored under exactly those, filters off; a domain gives "
                "ln(2)*score/span nats to each residue it aligns and a residue takes "
                "the maximum over covering domains, clamped at zero. A prefix with no "
                "profile gives exactly zero for both candidates"
            ),
        }


def _write_fasta(path: Path, records: Sequence[tuple[str, str]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for name, sequence in records:
            handle.write(f">{name}\n{sequence}\n")


def _run(command: Sequence[str], *, log: Path) -> None:
    """Run a HMMER program, failing loudly and keeping its output for the record."""

    completed = subprocess.run([str(part) for part in command], capture_output=True, text=True)
    Path(log).write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"{command[0]} exited {completed.returncode}: {completed.stderr.strip()[:400]}"
        )


def build_pfam_profile_member(
    *,
    tool: HmmerTool,
    pfam_hmm: Path,
    triples: Sequence[Triple],
    workdir: Path,
    cpu: int,
) -> ProfileHomologyMember:
    """Pfam-A as the profile member: curated families, and no circularity at all.

    Pfam profiles are estimated from curated seed alignments over all of UniProt.
    They do not contain the cohort's sequences as such, so this member cannot be
    accused of scoring a candidate against a profile built from it -- which is the
    objection the corpus member has to answer separately. Three passes: the
    prefixes select profiles at Pfam's gathering threshold, those profiles are
    fetched into a small file, and the candidates are scored under them with the
    heuristic filters off.
    """

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    pfam_hmm = Path(pfam_hmm)
    if not pfam_hmm.is_file():
        raise FileNotFoundError(f"{pfam_hmm} does not exist")
    prefixes = workdir / "prefixes.fasta"
    candidates = workdir / "candidates.fasta"
    _write_fasta(prefixes, [(f"PFX{index}", triple.prefix) for index, triple in enumerate(triples)])
    _write_fasta(
        candidates,
        [
            (f"CND{index}_{label}", triple.continuations[label])
            for index, triple in enumerate(triples)
            for label in CANDIDATES
        ],
    )
    selection_table = workdir / "prefix.domtbl"
    _run(
        [
            tool.path("hmmsearch"), "--cpu", str(cpu), PROFILE_SELECTION_ARGUMENT,
            "--domtblout", selection_table, "-o", "/dev/null", pfam_hmm, prefixes,
        ],
        log=workdir / "prefix_search.log",
    )
    selected_by_target: dict[str, set[str]] = {}
    for hit in parse_domtblout(selection_table):
        selected_by_target.setdefault(hit.target, set()).add(hit.profile)
    selected = {
        triple.prefix: frozenset(selected_by_target.get(f"PFX{index}", set()))
        for index, triple in enumerate(triples)
    }
    wanted = sorted({profile for value in selected.values() for profile in value})
    hits: dict[str, list[DomainHit]] = {}
    scoring_table = workdir / "candidate.domtbl"
    if wanted:
        # hmmfetch needs its own SSI index, which hmmpress does not create. Built
        # once beside the profile library rather than required of the operator: it is
        # a derived index over a file this stage was handed, and a missing one would
        # otherwise stop a campaign several minutes in.
        if not pfam_hmm.with_suffix(pfam_hmm.suffix + ".ssi").exists():
            _run([tool.path("hmmfetch"), "--index", pfam_hmm], log=workdir / "hmmfetch_index.log")
        names = workdir / "selected.names"
        names.write_text("\n".join(wanted) + "\n", encoding="utf-8")
        small = workdir / "selected.hmm"
        with small.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                [str(tool.path("hmmfetch")), "-f", str(pfam_hmm), str(names)],
                stdout=handle, stderr=subprocess.PIPE, text=True,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"hmmfetch exited {completed.returncode}: {completed.stderr[:400]}")
        _run(
            [
                tool.path("hmmsearch"), "--cpu", str(cpu), *PROFILE_SCORING_ARGUMENTS,
                "-E", str(PROFILE_SCORING_EVALUE), "--domE", str(PROFILE_SCORING_EVALUE),
                "--domtblout", scoring_table, "-o", "/dev/null", small, candidates,
            ],
            log=workdir / "candidate_search.log",
        )
        continuation_of = {
            f"CND{index}_{label}": triple.continuations[label]
            for index, triple in enumerate(triples)
            for label in CANDIDATES
        }
        for hit in parse_domtblout(scoring_table):
            hits.setdefault(continuation_of[hit.target], []).append(hit)
    return ProfileHomologyMember(
        "pfam_profile",
        selected=selected,
        hits=hits,
        provenance={
            "source": str(pfam_hmm),
            "sha256": sha256_file(pfam_hmm),
            "hmmer": tool.record(),
            "n_profiles_in_source": None,
            "n_profiles_selected": len(wanted),
            "circularity": (
                "none: Pfam profiles are estimated from curated seed alignments over "
                "UniProt and are not built from this cohort's sequences"
            ),
            "bound_direction": (
                "Pfam annotates a fraction of any proteome, so a prefix outside Pfam "
                "contributes exactly zero and this member is a LOWER bound on what the "
                "profile family achieves. context_coverage reports that fraction"
            ),
        },
    )


def _sequence_id(name: str) -> str:
    """The UniProt accession inside a Swiss-Prot FASTA name such as ``sp|P14174|MIF_HUMAN``."""

    parts = name.split("|")
    return parts[1] if len(parts) >= 2 else name


def _stockholm_names(path: Path) -> list[str]:
    names: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        names.append(line.split()[0])
    return names


def _filter_stockholm(source: Path, destination: Path, drop: set[str]) -> tuple[int, int]:
    """Copy an alignment, removing the rows whose accession is in ``drop``.

    Written line-wise rather than through an alignment library because HMMER's
    ``-A`` output is one block of full-length rows, which is the one Stockholm
    shape a line filter handles exactly. Returns the rows kept and dropped.
    """

    kept = dropped = 0
    with Path(source).open("r", encoding="utf-8") as reader, Path(destination).open(
        "w", encoding="utf-8"
    ) as writer:
        for line in reader:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("//"):
                writer.write(line)
                continue
            name = stripped.split()[0]
            if _sequence_id(name) in drop:
                dropped += 1
                continue
            kept += 1
            writer.write(line)
    return kept, dropped


def build_corpus_profile_member(
    *,
    tool: HmmerTool,
    corpus_fasta: Path,
    triples: Sequence[Triple],
    workdir: Path,
    cpu: int,
    parallel: int,
) -> ProfileHomologyMember:
    """A profile built from the anchor's prefix against the staged corpus.

    This is the member that covers what Pfam does not, and it is the one that has
    to answer the circularity objection: a search from the prefix can recruit the
    candidates themselves, and scoring a candidate against a profile that contains
    it is a lookup rather than a prediction. Both candidate accessions are
    therefore removed from the recruited alignment before the profile is built, and
    **whether each was recruited at all is recorded** -- that count is the most
    direct measurement of the remote-homology hypothesis this stage makes, because
    it is corpus statistics finding the structure partner from the prefix alone.
    """

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    corpus_fasta = Path(corpus_fasta)
    if not corpus_fasta.is_file():
        raise FileNotFoundError(f"{corpus_fasta} does not exist")
    if parallel < 1 or cpu < 1:
        raise ValueError("parallel and cpu must be positive")

    def one(index: int) -> dict[str, Any]:
        triple = triples[index]
        cell = workdir / f"triple{index:04d}"
        cell.mkdir(parents=True, exist_ok=True)
        query = cell / "prefix.fasta"
        _write_fasta(query, [(f"PFX{index}", triple.prefix)])
        alignment = cell / "recruited.sto"
        _run(
            [
                tool.path("jackhmmer"), "--cpu", str(cpu), "-N", str(CORPUS_PROFILE_ITERATIONS),
                "--noali", "-A", alignment, "-o", cell / "jackhmmer.out", query, corpus_fasta,
            ],
            log=cell / "jackhmmer.log",
        )
        recruited = {_sequence_id(name) for name in _stockholm_names(alignment)} if alignment.is_file() else set()
        drop = {triple.partner_ids[label] for label in CANDIDATES}
        filtered = cell / "filtered.sto"
        kept, dropped = (
            _filter_stockholm(alignment, filtered, drop) if alignment.is_file() else (0, 0)
        )
        row: dict[str, Any] = {
            "index": index,
            "anchor": triple.anchor,
            "n_recruited": len(recruited),
            "recruited_anchor": triple.anchor in recruited,
            "recruited_sequence_partner": triple.partner_ids["sequence_partner"] in recruited,
            "recruited_structure_partner": triple.partner_ids["structure_partner"] in recruited,
            "rows_kept": kept,
            "rows_dropped": dropped,
            "profile_built": False,
            "hits": [],
        }
        if kept < 1:
            return row
        profile = cell / "prefix.hmm"
        _run(
            [tool.path("hmmbuild"), "--amino", "-n", f"CORPUS{index}", profile, filtered],
            log=cell / "hmmbuild.log",
        )
        candidates = cell / "candidates.fasta"
        _write_fasta(
            candidates,
            [(f"CND{index}_{label}", triple.continuations[label]) for label in CANDIDATES],
        )
        table = cell / "candidate.domtbl"
        _run(
            [
                tool.path("hmmsearch"), "--cpu", "1", *PROFILE_SCORING_ARGUMENTS,
                "-E", str(PROFILE_SCORING_EVALUE), "--domE", str(PROFILE_SCORING_EVALUE),
                "--domtblout", table, "-o", "/dev/null", profile, candidates,
            ],
            log=cell / "candidate_search.log",
        )
        row["profile_built"] = True
        row["hits"] = parse_domtblout(table)
        return row

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        rows = list(pool.map(one, range(len(triples))))

    selected: dict[str, frozenset[str]] = {}
    hits: dict[str, list[DomainHit]] = {}
    for row in rows:
        triple = triples[row["index"]]
        selected[triple.prefix] = frozenset({f"CORPUS{row['index']}"}) if row["profile_built"] else frozenset()
        continuation_of = {
            f"CND{row['index']}_{label}": triple.continuations[label] for label in CANDIDATES
        }
        for hit in row["hits"]:
            hits.setdefault(continuation_of[hit.target], []).append(hit)
    recruited_structure = sum(1 for row in rows if row["recruited_structure_partner"])
    recruited_sequence = sum(1 for row in rows if row["recruited_sequence_partner"])
    return ProfileHomologyMember(
        "corpus_profile",
        selected=selected,
        hits=hits,
        provenance={
            "source": str(corpus_fasta),
            "sha256": sha256_file(corpus_fasta),
            "hmmer": tool.record(),
            "iterations": CORPUS_PROFILE_ITERATIONS,
            "excluded_from_the_alignment": CORPUS_PROFILE_EXCLUDES,
            "n_profiles_built": sum(1 for row in rows if row["profile_built"]),
            "recruited_alignment_rows": {
                "min": min(row["n_recruited"] for row in rows),
                "median": float(np.median([row["n_recruited"] for row in rows])),
                "max": max(row["n_recruited"] for row in rows),
            },
            # The measurement the whole ceiling question turns on, and it needs no
            # model: how often does a corpus search from the anchor's prefix alone
            # find the partner that carries the anchor's fold?
            "recruitment": {
                "n_triples": len(rows),
                "n_recruiting_the_structure_partner": recruited_structure,
                "n_recruiting_the_sequence_partner": recruited_sequence,
                "share_recruiting_the_structure_partner": recruited_structure / len(rows),
                "share_recruiting_the_sequence_partner": recruited_sequence / len(rows),
                "reading": (
                    "a jackhmmer search of the staged corpus, seeded only by the "
                    "anchor's prefix, that recruits the structure partner has detected "
                    "the remote homology this cohort's construction guarantees exists. "
                    "That is corpus statistics, not structural knowledge, and the share "
                    "bounds how much of any positive contrast it could explain"
                ),
            },
            "bound_direction": (
                "the corpus searched is the staged Swiss-Prot and not the arms' own "
                "UniRef50/UniRef90 pretraining mixtures, and jackhmmer runs at "
                f"{CORPUS_PROFILE_ITERATIONS} iterations. Both make this member a LOWER "
                "bound on what the profile family achieves"
            ),
            "circularity": (
                "both candidate accessions are removed from the recruited alignment "
                "before the profile is built, so no candidate is scored against a "
                "profile containing it. The anchor's own entry is kept: it is not a "
                "candidate and the arms saw it in training too"
            ),
        },
    )
