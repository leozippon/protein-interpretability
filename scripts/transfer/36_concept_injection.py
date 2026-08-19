#!/usr/bin/env python3
"""D3.g stage 36: does a text-derived concept direction causally steer protein mode?

The claim-bearing stage of D3.g, and the one that supplies the two hardest clauses of
the audit's §8 item 4 -- **transfer at least one graded protein-model intervention in
the predicted direction** and **preserve unrelated concepts**.

**Everything decidable here is frozen in `docs/EXPERIMENT_LOG.md` as EXP-R2-213.** This
stage implements A36-0 to A36-7 and adds no criterion of its own. Each frozen threshold
is a **required, never-defaulted flag** whose value :func:`resolve` checks against the
pre-registration and refuses to run without; a stage that defaulted any of them could
choose it after seeing which value worked, and would look exactly like one that had not.

* `--alphas` must be the nine frozen rungs `-4,-2,-1,-0.5,0,0.5,1,2,4` (sigma units).
* `--coherence-max-nll-inflation` must be `0.25` nats/token on **non-bearing** held-out
  sequences (A36-1), and the verdict is re-read at `0.10 / 0.25 / 0.50`.
* `--spearman-ceiling` must be `-0.8` (A36-3(c)).
* `--random-null-margin` must be `2.0` (A36-3(b)).
* `--random-directions` and `--permuted-directions` must be at least eight **distinct
  directions** (A36-3(b), A36-5).
* `--specificity-rule` must be `row_offdiagonal_p95` (A36-4).
* `--operating-alpha-rule` must be `smallest_admissible_positive_significant`.
* `--injection-site` must be the site the direction was estimated at.
* `--layer` is required: every criterion is per layer and never a cross-layer mean (L32).

**The sign convention, because it is the criterion.** Delta is the pre-registration's own
NLL shift -- `[NLL_bearing(a) - NLL_bearing(0)] - [NLL_nonbearing(a) - NLL_nonbearing(0)]`
-- so the predicted direction is `Delta < 0` at `alpha > 0` and A36-3(c)'s ceiling is
`rho <= -0.8`. This stage reports Delta in that convention and nowhere reports a
"benefit" with the opposite sign. A36-4's comparison is made on `-Delta`, the effect in
the predicted direction, so that "the diagonal exceeds its own row's off-diagonal 95th
percentile" means a larger movement rather than a larger signed number; both matrices
reach the artefact.

**Order of operations, and it is the pre-registration's.** A36-0's hook invariants run
before any effect is computed. A36-6's attainability -- the alpha=0 HMMER/Pfam-A
annotation rate -- is reported before any injected rate is read. A36-2's text-mode
positive control decides whether the protein side is read at all: if it does not fire the
stage returns VOID and the protein side is not read, because a protein null measured by
an instrument that has not been shown to work is uninterpretable rather than negative.

**Scope: the concepts are the cohort's admitted ones, and nothing else can be run.**
`--concepts` takes admitted concept **ids** -- `ec_hydrolase`, `go_atp_binding` -- which
are the coarse pre-declared concepts of `sequence_description.CONCEPTS`, each admitted
under C34-5 by measured per-cell group counts. The list is read from the cohort artefact
and anything outside it is refused, naming the offenders; there is no derived fallback. An
earlier version of this stage took `namespace:term` pairs, which would have let a specific
EC number like `3.2.1.4` be evaluated at group counts nobody checked against the floor.
The membership rule is likewise the cohort's own and is **three-valued**: a record with no
annotation of the relevant sort is undefined for the concept and enters neither arm.

**Reduction, if cost forces it.** The frozen rule is to drop concepts in ascending order
of their `eval` bearing-group count, a quantity known before any activation is captured,
and never to drop a control, a rung or a split. `--max-concepts` applies exactly that
rule and records which concepts were dropped with their counts. Note the interaction the
artefact also records: under the full split a pass costs the same whatever a concept
bears, so the rule drops the weakest concepts; under amendment 3's balanced draw a
concept's cost is proportional to what it bears, so the same rule drops the cheapest and
keeps the most expensive.

**The instrument check runs before any real number is trusted.** `--synthetic` plants
concept directions in a toy decoder whose response is known by construction and runs the
whole pipeline -- the same hook, the same scoring loop, the same bootstrap, the same
controls, the same verdict -- on two cells: one where the direction genuinely steers both
modes, and one where it steers text only. The first must be recovered and the second must
not.

**Amendment 3 (EXP-R2-213), decided before any campaign number existed:** A36-5 is
distributional rather than any-draw, a balanced per-concept evaluation draw is admitted
with a mandatory second-seed variance measurement, and A36-6's fit-split Pfam referent is
authorised by name. The layer is stage 35's own pre-registered decision layer, and
`--stage35-artefact` is read so that the layer, the site and the pooling are *checked*
against its causal hand-off rather than asserted.

**Cost, measured on the production cohort rather than estimated.** The scored pass count
is `K * (9 + 4*random + 9*permuted)` in protein mode and `K * (9 + 4*random)` in text: the
concept's own ladder runs at all nine rungs, the controls' *criteria* are read only at
positive rungs, and the permuted control keeps nine so that its per-draw A36-3 diagnostic
survives. One protein pass costs **0.161 s per scored record** on an L20 at float32
(measured on `ProLLaMA_Stage_1`, layer 16, the cohort's own 328-residue mean, 211 scored
tokens per record at 0.76 ms each, flat in batch size from 4 to 32, so the pass is
compute-bound); a text pass costs **0.142 s**. TF32 is deliberately not enabled: it would
be roughly fourfold faster and its 10-bit mantissa is not a safe substrate for a 0.01-nat
cross-entropy difference.

**What that comes to, and it does not fit.** The production cohort's `eval` split is
4,499 records in 3,752 near-duplicate groups, and the admitted concepts carry 15 to 985
bearing groups. Under `balanced_1to1` the union of all 17 concepts' subsets is 4,154 of
those 4,499 records -- `ec_transferase` alone draws 2,449 -- so the balanced draw buys
almost nothing at full breadth. Measured, for one layer across both checkpoints: **946
GPU-h** at all 17 concepts, **428** at the eight the frozen reduction rule keeps, and
**69** at the eight it drops. Against a 40 GPU-h bound, no `K = 8` selection fits. The
arithmetic and the options belong to the coordinator, not to this stage, which is why
nothing here reduces the design further on its own.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.transfer import concept_alignment as ca  # noqa: E402
from src.transfer import concept_injection as ci  # noqa: E402
from src.transfer import sequence_description as sd  # noqa: E402
from src.transfer import joint_modes  # noqa: E402
from src.transfer.arms import REPO  # noqa: E402
from src.transfer.io import sha256_file, write_json  # noqa: E402
from src.transfer.replaceable import JOINT_MODES, JointReplaceable  # noqa: E402


def _load_stage(filename: str) -> Any:
    """Import a stage whose module name starts with a digit.

    One of them, and for the reason ``32_crosscoder.py`` gives: stage 21 owns the
    only joint-checkpoint loader this programme has, and Appendix B rule 12 does
    not stop applying because the declaration lives in a file whose name starts
    with a digit.
    """

    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(f"_transfer_stage_{filename[:2]}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE21 = _load_stage("21_joint_mode_qualification.py")

SCHEMA_VERSION = ci.SCHEMA_VERSION
DEFAULT_OUT = REPO / "results/transfer/concept_injection"

#: float32 and nothing else (EXP-R2-213's standing condition). Declared here rather
#: than offered as a flag, because the alternative is not a legitimate setting for
#: an estimand that is a cross-entropy difference of order 0.01-0.1 nats.
INFERENCE_DTYPE = "float32"

#: The text side is the MASKED description and nothing else. A constant rather than
#: a flag because the whole design turns on it: a direction estimated from the
#: protein's name or its unmasked function text is a direction that can read the
#: concept's own word, and offering that as an option would offer the failure C34-1
#: exists to exclude.
TEXT_FIELD = "description_masked"

#: The split the direction is FITTED on; ``--eval-split`` is where it is measured.
#: Both come from the cohort's own declaration and neither is re-derived here -- L30
#: is what a second definition of a held-out set costs on a protein corpus.
FIT_SPLIT = "fit"

#: How the mandatory second-seed draw rides through the same sweeps: as a pseudo-
#: concept with the concept's own membership and an independently drawn subset. It is
#: never a measured concept, and it is stripped out of the concept set, the
#: specificity matrix and the verdicts.
DRAW_VARIANCE_SUFFIX = "::draw2"

CAMPAIGN_ONLY_FLAGS = (
    "checkpoint",
    "rendering",
    "cohort",
    "concepts",
    "pooling",
    "eval_split",
    "generation_readout",
    "stage35_artefact",
    "evaluation_draw",
)

#: Required on both paths: these are the frozen criteria, and supplying them under
#: ``--synthetic`` too is what makes the known-answer check a check of the decision
#: rule rather than only of the arithmetic.
PRE_REGISTERED_DECISIONS = (
    "layer",
    "alphas",
    "injection_site",
    "coherence_max_nll_inflation",
    "spearman_ceiling",
    "random_null_margin",
    "random_directions",
    "permuted_directions",
    "specificity_rule",
    "operating_alpha_rule",
)

PROVENANCE_MODULES = (
    "src/transfer/concept_injection.py",
    "src/transfer/concept_alignment.py",
    "src/transfer/replaceable.py",
    "src/transfer/joint_modes.py",
    "src/transfer/statistics.py",
    "src/transfer/io.py",
    "scripts/transfer/21_joint_mode_qualification.py",
)

LIMITATIONS = [
    {
        "id": "L31",
        "what": "a multi-residue BPE assigns tokens by sequence context, so a "
        "position-level intervention is undefined on part of any cohort and the "
        "survivors are the BPE-stable subset rather than a random one",
        "how_this_stage_answers_it": "the write is over every content position of a "
        "record rather than at a residue index, so no instance is selected on and "
        "nothing is computed on a survivor subcohort",
    },
    {
        "id": "L15",
        "what": "a conditioning leak makes a label lookup indistinguishable from a "
        "concept; D3.b's only positive protein cell dissolved under the pre-declared "
        "control of exactly that",
        "how_this_stage_answers_it": "the direction is estimated from descriptions "
        "masked of the concept's own term (C34-1) and the masking is re-checked here "
        "against the surviving text before a checkpoint is loaded",
    },
    {
        "id": "L32",
        "what": "a criterion stated per unit and read as a mean over units gave the "
        "D3.h diffing unit a verdict its own per-layer vector contradicted",
        "how_this_stage_answers_it": "one layer per run and no cross-layer mean; A36-4 "
        "is judged per concept against its own row's 95th percentile and the mean "
        "diagonal minus mean off-diagonal is reported as explicitly non-substitutable",
    },
    {
        "id": "L23",
        "what": "this family's protein symbol is a TOKEN spelling about 1.5 residues, "
        "so a magnitude in nats per token is not comparable with a per-residue arm's",
        "how_this_stage_answers_it": "every magnitude here is within one checkpoint and "
        "within one mode, so it needs no cross-arm conversion; it must not be quoted "
        "beside a per-residue figure",
    },
    {
        "id": "site_locality",
        "what": "the write is at the feed-forward's input, which is the site the "
        "direction and its sigma were estimated at, and that write is consumed by one "
        "block's feed-forward rather than persisting in the residual stream",
        "how_this_stage_answers_it": "stated rather than removed. A residual-stream "
        "write would be in a different space than the one sigma was measured in, "
        "because the post-attention RMSNorm rescales and applies a learned gain, so the "
        "coefficient would no longer be population-sigmas along the concept direction",
    },
    {
        "id": "undefined_is_a_third_cell",
        "what": "the cohort's membership rule is three-valued: a record with no "
        "annotation of the relevant sort is UNDEFINED for a concept and enters neither "
        "side. On the production cohort that is most of the split for some concepts -- "
        "ec_hydrolase has 508 bearing, 2,065 non-bearing and 1,292 UNDEFINED groups in "
        "eval",
        "how_this_stage_answers_it": "labels come from "
        "sequence_description.concept_label through the ontology the cohort itself "
        "recorded, so undefined records are excluded from both arms of Delta and from "
        "the coherence floor. A two-valued reading over the raw annotation column would "
        "have folded those 1,292 groups of unknown status into the non-bearing arm and "
        "silently changed the estimand",
    },
    {
        "id": "concept_scope",
        "what": "a stage that accepted a raw annotation term could evaluate concepts the "
        "cohort never admitted, at group counts nobody checked against C34-5's floor",
        "how_this_stage_answers_it": "--concepts takes ADMITTED concept ids and "
        "require_admitted_concepts refuses anything outside the list the cohort artefact "
        "records, naming the offenders and the artefact. There is no derived fallback",
    },
    {
        "id": "balanced_draw_variance",
        "what": "amendment 3's balanced draw turns the non-bearing arm from the whole "
        "declared split into a sample, which introduces draw variance the full-split "
        "design did not carry",
        "how_this_stage_answers_it": "measured, not assumed: one seed fixes the "
        "population for every rung, concept and checkpoint, and a second independent "
        "seed is run on one concept and reported beside the first as a first-class "
        "output. Because a subset selects rows of an already-scored pass, that "
        "measurement costs no additional forward pass",
    },
    {
        "id": "corpus_exposure",
        "what": "Swiss-Prot lies inside UniRef50 by construction, so every evaluation "
        "protein is a candidate pretraining member whatever family it belongs to",
        "how_this_stage_answers_it": "not answered, and EXP-R2-213 says so: passing "
        "every clause here establishes a graded, concept-specific causal effect and NOT "
        "that the checkpoint encodes biological knowledge",
    },
    {
        "id": "pfam_referent_is_derived",
        "what": "A36-6 needs a concept-consistent Pfam family set, and EXP-R2-213's "
        "concepts are GO terms and EC numbers, neither of which is a Pfam accession",
        "how_this_stage_answers_it": "the referent is derived from the cohort's own "
        "pfam column on the FIT split alone, so the evaluation split never defines the "
        "target it is scored against; a concept whose referent is empty has readout B "
        "refused with that reason rather than scored against an invented mapping. The "
        "derivation is authorised by name in EXP-R2-213 amendment 3 and was not in the "
        "frozen text",
    },
    {
        "id": "a36_5_against_a36_3b",
        "what": "A36-5 asks a permuted-label refit to FAIL A36-3, and A36-3(b)'s bar "
        "is set by ISOTROPIC random directions. Those are not the same population: a "
        "permuted refit lies in the span of the concept structure plus the "
        "representation cloud's own noise, so its component on any one concept "
        "direction is larger than an isotropic vector's, which falls only as one over "
        "the root of the width. The two clauses are therefore in tension, and the "
        "tension is tighter the lower the effective dimension of the cloud and the "
        "fewer the concepts",
        "how_this_stage_answers_it": "measured rather than assumed. On the "
        "known-answer fixture at three concepts and width 32 the worst of eight "
        "permuted refits carried 0.89 of the planted direction and PASSED every clause "
        "of A36-3; at eight concepts and width 128 none of 32 draws on one concept "
        "passes and the worst reaches 0.62 of the bar against the planted direction's "
        "1.19. The campaign's own margin is not this fixture's, so the artefact reports "
        "every permuted draw's verdict AND its distance from the bar rather than only "
        "the gate's boolean",
    },
    {
        "id": "a36_5_strictness_scales_with_its_control_size",
        "what": "A36-5 voids the readout if ANY permuted-label direction passes A36-3, "
        "so the gate gets strictly harder as --permuted-directions rises: the "
        "probability that some draw lands near enough to the concept direction grows "
        "with the number of draws. The control size is therefore not neutral to the "
        "verdict, which is unusual for a control and is worth stating before dispatch",
        "how_this_stage_answers_it": "not resolved here, because the rule is frozen. "
        "The stage runs the declared number, records each draw's distance to its bar, "
        "and reports the count that passed rather than only the boolean, so a void can "
        "be read as 'one draw of eight, at 1.02 of the bar' rather than as a flat "
        "failure. On the known-answer fixture with the effect PLANTED, one of eight "
        "concepts is voided this way at eight draws per concept -- which is the gate "
        "behaving as written on a fixture whose geometry allows it, and is the number "
        "a reader needs in order to weigh a void on the real cohort",
    },
    {
        "id": "clean_unavailable",
        "what": "CLEAN's EC prediction needs an ESM-1b encoder and CLEAN's own trained "
        "weights, and neither is staged on a host with no route to fetch them",
        "how_this_stage_answers_it": "probed and reported as unavailable with the files "
        "that are missing; no EC prediction is produced",
    },
]


# ------------------------------------------------------------------- arguments


def parse_alphas(argument: str) -> tuple[float, ...]:
    values = [part.strip() for part in str(argument).split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("the coefficient ladder is empty")
    try:
        ladder = tuple(sorted(float(value) for value in values))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"not a coefficient ladder: {argument!r}") from error
    return ladder


def parse_concepts(argument: str) -> tuple[str, ...]:
    """Concept ids the cohort ADMITTED, e.g. 'ec_hydrolase,go_atp_binding'.

    Ids and not ``namespace:term`` pairs, and the difference is not cosmetic. The
    cohort's concepts are the coarse pre-declared ones of
    ``sequence_description.CONCEPTS`` -- ``ec_hydrolase`` is EC class 3, not the EC
    number ``3.2.1.4`` -- each admitted under C34-5 by measured per-cell group counts.
    A stage that accepted a raw annotation term would happily evaluate a specific EC
    number that was never admitted, at group counts nobody checked against the floor.
    :func:`src.transfer.concept_injection.require_admitted_concepts` refuses anything
    outside the admitted list.
    """

    concepts = [part.strip() for part in str(argument).split(",") if part.strip()]
    if len(concepts) < 2:
        raise argparse.ArgumentTypeError(
            "at least two concepts are required: with one there is no off-diagonal and "
            "A36-4 is vacuous"
        )
    if len(set(concepts)) != len(concepts):
        raise argparse.ArgumentTypeError("the concept set carries a duplicate")
    return tuple(concepts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="directory of the joint checkpoint. A path and not an arm name, for the "
        "reason 21_joint_mode_qualification.py gives; both of its modes must be "
        "declared readable by concept_alignment.PROTEIN_MODE_BEHAVIOURAL_STATUS, which "
        "refuses Llama-2-7b-hf's protein mode on EXP-R2-152's -0.0013 nats bound",
    )
    parser.add_argument(
        "--rendering",
        default=None,
        choices=joint_modes.RENDERING_NAMES,
        help="which declared family's input format this checkpoint takes",
    )
    parser.add_argument(
        "--cohort",
        type=Path,
        default=None,
        help="the cohort directory written by 34_sequence_description_cohort.py; this "
        "stage consumes it through concept_alignment.load_cohort and never draws one",
    )
    parser.add_argument(
        "--concepts",
        type=parse_concepts,
        default=None,
        help="the concept set, as 'namespace|term' pairs written 'namespace:term'. At "
        "least two: A36-4 is read on the off-diagonal and one concept has none",
    )
    parser.add_argument(
        "--pooling",
        default=None,
        choices=ca.POOLINGS,
        help="how a record's content positions become one representation. Passed "
        "through to concept_alignment.mode_representations, which owns it",
    )
    parser.add_argument(
        "--eval-split",
        default=None,
        choices=[split for split in ca.SPLITS if split != FIT_SPLIT],
        help=f"the cohort split the intervention is MEASURED on. The direction is always "
        f"fitted on {FIT_SPLIT!r}, so this must differ from it. 'family_holdout' is the "
        "split §8's unseen-family clause is read on",
    )
    parser.add_argument(
        "--generation-readout",
        default=None,
        choices=("hmmer_pfam", "refused"),
        help="REQUIRED and never defaulted. 'hmmer_pfam' runs A36-6; 'refused' records "
        "that readout B was not run, with the operator's reason, rather than leaving an "
        "absent field a reader would have to notice",
    )
    parser.add_argument(
        "--generation-refusal-reason",
        default=None,
        help="why readout B was refused; required with --generation-readout refused",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="REQUIRED: the backbone layer the direction is estimated at and written "
        "at. One layer per run, because every criterion is per layer and a pass at one "
        "layer is a pass at that layer and no other (L32)",
    )
    parser.add_argument(
        "--alphas",
        type=parse_alphas,
        default=None,
        help="REQUIRED: the nine frozen rungs in sigma units. It begins with a minus "
        "sign, so write it as --alphas=-4,-2,-1,-0.5,0,0.5,1,2,4 rather than with a "
        "space, which argparse would read as an option. resolve() refuses any ladder "
        "other than EXP-R2-213's",
    )
    parser.add_argument(
        "--injection-site",
        default=None,
        choices=ci.INJECTION_SITES,
        help="REQUIRED, with exactly one legal value so that a second site cannot "
        "arrive without being named. It is the site the direction was estimated at; see "
        "src.transfer.concept_injection.INJECTION_SITE_NOTE",
    )
    parser.add_argument(
        "--coherence-max-nll-inflation",
        type=float,
        default=None,
        help="REQUIRED: A36-1's bound, in nats/token of inflation on the NON-BEARING "
        "held-out sequences. The frozen value is 0.25, anchored to these checkpoints' "
        "own protein-mode context information (0.5505 and 0.5215), so it admits damage "
        "of at most half the directional signal the mode has. resolve() refuses any "
        "other value on a real campaign. If no rung is admissible the readout is closed "
        "and the bound is not widened",
    )
    parser.add_argument(
        "--spearman-ceiling",
        type=float,
        default=None,
        help="REQUIRED: A36-3(c)'s ceiling, frozen at -0.8. It is negative because "
        "Delta is an NLL SHIFT and a working concept vector at alpha > 0 reduces the "
        "bearing sequences' cross-entropy",
    )
    parser.add_argument(
        "--random-null-margin",
        type=float,
        default=None,
        help="REQUIRED: A36-3(b)'s multiple of the norm-matched random-direction "
        "control's 95th percentile of |Delta|. Frozen at 2.0",
    )
    parser.add_argument(
        "--random-directions",
        type=int,
        default=None,
        help="REQUIRED: distinct norm-matched random directions in A36-3(b)'s control, "
        "at least eight. Distinct DIRECTIONS, not repeats of one and not more positions: "
        "with a random-direction control the detection floor is set by "
        "direction-to-direction variation",
    )
    parser.add_argument(
        "--permuted-directions",
        type=int,
        default=None,
        help="REQUIRED: distinct permuted-label concept vectors in A36-5's control, at "
        "least eight. Each is run over the whole ladder and put through A36-3, because "
        "A36-5 asks whether a permuted direction can PASS A36-3",
    )
    parser.add_argument(
        "--specificity-rule",
        default=None,
        choices=ci.SPECIFICITY_RULES,
        help="REQUIRED: A36-4's rule. One legal value, because a row mean may not "
        "substitute for the row's 95th percentile (L32, Appendix B rule 33)",
    )
    parser.add_argument(
        "--operating-alpha-rule",
        default=None,
        choices=ci.OPERATING_ALPHA_RULES,
        help="REQUIRED: which admissible rung a concept's specificity row and its "
        "generation are read at. The smallest admissible positive rung at which A36-3(a) "
        "and (b) both fire -- smallest rather than strongest, because picking the "
        "strongest would be selection on the outcome",
    )
    parser.add_argument(
        "--stage35-artefact",
        type=Path,
        default=None,
        help="35_concept_alignment.py's artefact for this checkpoint and cell. REQUIRED "
        "on a campaign: its causal hand-off carries the decision layer, the site and "
        "the pooling, and this stage refuses a --layer, --injection-site or --pooling "
        "that disagrees. Amendment 3 fixes the layer as stage 35's already "
        "pre-registered decision layer, and checking it is what makes that true rather "
        "than asserted",
    )
    parser.add_argument(
        "--evaluation-draw",
        default=None,
        choices=ci.EVALUATION_DRAWS,
        help="REQUIRED: which population Delta is computed over. 'full_split' is the "
        "frozen behaviour -- every record the cohort DEFINES the concept on. "
        "'balanced_1to1' is EXP-R2-213 amendment 3's addition: every near-duplicate "
        "group the concept bears, plus a seeded equal-size draw of groups it does not. "
        "Declared rather than defaulted so the authorisation is visible at the call site",
    )
    parser.add_argument(
        "--evaluation-draw-seed",
        type=int,
        default=None,
        help="REQUIRED with --evaluation-draw balanced_1to1: the ONE seed the draw is "
        "made from. One seeded permutation of the groups serves every concept, every "
        "rung and both checkpoints, so no two reported numbers rest on different "
        "populations",
    )
    parser.add_argument(
        "--evaluation-draw-cap",
        type=int,
        default=None,
        help="REQUIRED with --evaluation-draw balanced_1to1: EXP-R2-213 amendment 4's "
        "per-side bearing-group cap, frozen at 32. It is a CRITERION IMPROVEMENT and "
        "not an economy: within one row of A36-4's matrix each cell is Delta for the "
        "scored concept on its own subset at its own group count, so uncapped a "
        "15-group diagonal could be compared against a percentile over 985-group "
        "cells. The cap equalises the diagonal and every off-diagonal cell of a row at "
        "once. 32 rather than 16 because below about 24 groups per side the point "
        "estimate becomes draw-sensitive",
    )
    parser.add_argument(
        "--draw-variance-seed",
        type=int,
        default=None,
        help="REQUIRED with --evaluation-draw balanced_1to1, and it must differ from "
        "--evaluation-draw-seed: a second independent draw for one concept, whose "
        "result is reported beside the first. Balancing turns the non-bearing arm from "
        "the whole split into a SAMPLE, so it carries draw variance the full-split "
        "design did not, and that variance is measured rather than assumed small",
    )
    parser.add_argument(
        "--draw-variance-concept",
        default=None,
        help="extra concepts, comma-separated, to add to the second-seed draw-variance "
        "measurement. Every concept below the draw-stability threshold is covered "
        "automatically and cannot be opted out of: that check is the designated "
        "detector for point-estimate draw sensitivity, so it is wired to the concepts "
        "that need it. If no concept is below the threshold the smallest is still "
        "measured, because amendment 3 makes the check mandatory",
    )
    parser.add_argument(
        "--max-concepts",
        type=int,
        default=None,
        help="the frozen reduction rule, if cost forces one: keep this many concepts, "
        "dropping in ascending order of their eval bearing-group count. Never reduces a "
        "control, a rung or a split. The dropped concepts and their counts reach the "
        "artefact",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--protein-context",
        default=None,
        help="the declared family's document context. Naming none measures the format "
        "stage 1 was trained on; naming one measures stage 2's, and whichever was used "
        "reaches the artefact",
    )
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument(
        "--invariant-scale",
        type=float,
        default=10.0,
        help="norm of the random perturbation A36-0's positive control writes",
    )
    parser.add_argument("--generate-sequences", type=int, default=64)
    parser.add_argument("--generate-max-new-tokens", type=int, default=160)
    parser.add_argument("--generate-temperature", type=float, default=1.0)
    parser.add_argument("--generate-top-p", type=float, default=0.95)
    parser.add_argument("--hmmscan-evalue", type=float, default=1e-3)
    parser.add_argument("--hmmscan-threads", type=int, default=8)
    parser.add_argument(
        "--pfam-referent-min-bearing",
        type=int,
        default=2,
        help="a Pfam family enters a concept's A36-6 referent when at least this many "
        "of the concept's FIT-split bearing records carry it",
    )
    parser.add_argument(
        "--hmmer-root",
        type=Path,
        default=None,
        help="working location the HMMER build lives in, outside the repository. "
        "Required with --generation-readout hmmer_pfam",
    )
    parser.add_argument(
        "--pfam-root",
        type=Path,
        default=None,
        help="working location the pressed Pfam-A database lives in, outside the "
        "repository. Required with --generation-readout hmmer_pfam",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="run the whole pipeline on a toy decoder whose response to the planted "
        "directions is known by construction, and write the recovery certificate",
    )
    parser.add_argument("--synthetic-d-model", type=int, default=128)
    parser.add_argument("--synthetic-layers", type=int, default=3)
    parser.add_argument(
        "--synthetic-concepts",
        type=int,
        default=8,
        help="planted concepts. Eight rather than two or three, and A36-5 is the "
        "reason: a permuted-label refit's direction lies in the span of the planted "
        "directions, so its component on any one of them falls as one over the root "
        "of the count, while A36-3(b)'s bar falls only with the width. At three "
        "concepts a permuted refit passed every clause of A36-3 on this toy",
    )
    parser.add_argument(
        "--synthetic-records",
        type=int,
        default=192,
        help="records in the toy cohort. Membership is drawn independently per "
        "concept at one half, so each concept has about half the cohort on each side, "
        "and the draw is refused below the eight-group per-side floor",
    )
    parser.add_argument("--synthetic-span", type=int, default=6)
    parser.add_argument("--synthetic-text-gain", type=float, default=1.0)
    parser.add_argument("--synthetic-protein-gain", type=float, default=1.0)
    parser.add_argument("--synthetic-mlp-gain", type=float, default=0.1)
    parser.add_argument("--synthetic-head-scale", type=float, default=0.25)
    return parser


def resolve(args: argparse.Namespace) -> None:
    """Refuse an incoherent or unfrozen request before a cohort is opened.

    Every check here is EXP-R2-213 read back. The frozen numbers are validated
    rather than defaulted, so a run either declares them or does not start.
    """

    missing = [flag for flag in PRE_REGISTERED_DECISIONS if getattr(args, flag) is None]
    if missing:
        raise ValueError(
            "this stage needs "
            + ", ".join(f"--{flag.replace('_', '-')}" for flag in missing)
            + ". Every one is a frozen criterion of EXP-R2-213 and none has a default: "
            "the ladder, the coherence bound, the Spearman ceiling, the random-null "
            "margin, both control sizes, the specificity rule, the operating-rung rule, "
            "the site and the layer all fix what counts as a result before the result "
            "exists"
        )
    if args.layer < 0:
        raise ValueError("--layer must be a non-negative backbone layer index")
    if tuple(args.alphas) != ci.FROZEN_ALPHA_LADDER:
        raise ValueError(
            f"--alphas is {list(args.alphas)} and EXP-R2-213 freezes the ladder at "
            f"{list(ci.FROZEN_ALPHA_LADDER)}. The ladder is not re-derived here; if it "
            "has to change, the pre-registration changes first"
        )
    if float(args.spearman_ceiling) != ci.FROZEN_SPEARMAN_CEILING:
        raise ValueError(
            f"--spearman-ceiling is {args.spearman_ceiling} and A36-3(c) freezes it at "
            f"{ci.FROZEN_SPEARMAN_CEILING}. It is negative because Delta is an NLL shift"
        )
    if float(args.random_null_margin) != ci.FROZEN_RANDOM_NULL_MARGIN:
        raise ValueError(
            f"--random-null-margin is {args.random_null_margin} and A36-3(b) freezes it "
            f"at {ci.FROZEN_RANDOM_NULL_MARGIN}"
        )
    for flag in ("random_directions", "permuted_directions"):
        if getattr(args, flag) < ci.MINIMUM_CONTROL_DIRECTIONS:
            raise ValueError(
                f"--{flag.replace('_', '-')} is {getattr(args, flag)} and EXP-R2-213 "
                f"requires at least {ci.MINIMUM_CONTROL_DIRECTIONS} distinct directions. "
                "A control is never reduced for economy"
            )
    if args.max_concepts is not None and args.max_concepts < 2:
        raise ValueError(
            "--max-concepts below two leaves no off-diagonal, and A36-4 would be vacuous"
        )

    if args.synthetic:
        present = [flag for flag in CAMPAIGN_ONLY_FLAGS if getattr(args, flag) is not None]
        if present:
            raise ValueError(
                ", ".join(f"--{flag.replace('_', '-')}" for flag in present)
                + " name a real campaign and are meaningless beside --synthetic, which "
                "runs the same pipeline on data whose answer is known"
            )
        if args.synthetic_concepts < 2:
            raise ValueError("--synthetic-concepts must be at least two")
        if args.coherence_max_nll_inflation <= 0.0:
            raise ValueError("--coherence-max-nll-inflation must be positive")
        return

    if float(args.coherence_max_nll_inflation) != ci.COHERENCE_PRIMARY_BOUND:
        raise ValueError(
            f"--coherence-max-nll-inflation is {args.coherence_max_nll_inflation} and "
            f"A36-1 freezes the primary bound at {ci.COHERENCE_PRIMARY_BOUND} nats/token "
            "on non-bearing sequences, anchored to the checkpoints' own protein-mode "
            "context information. The verdict is re-read at "
            f"{list(ci.COHERENCE_SENSITIVITY_BOUNDS)} in the same run, so a sensitivity "
            "question does not need a different invocation"
        )
    absent = [flag for flag in CAMPAIGN_ONLY_FLAGS if getattr(args, flag) is None]
    if absent:
        raise ValueError(
            "this stage needs "
            + ", ".join(f"--{flag.replace('_', '-')}" for flag in absent)
            + " on a real campaign. --eval-split and --generation-readout are frozen "
            "decisions too: the split the effect is measured on is not the one the "
            "direction was fitted on, and a readout that was not run is recorded as "
            "refused rather than silently absent"
        )
    if args.generation_readout == "refused" and not args.generation_refusal_reason:
        raise ValueError(
            "--generation-readout refused needs --generation-refusal-reason; a refusal "
            "without its reason is indistinguishable from an omission"
        )
    if args.evaluation_draw == "balanced_1to1":
        if args.evaluation_draw_cap is None:
            raise ValueError(
                "--evaluation-draw balanced_1to1 needs --evaluation-draw-cap; "
                f"EXP-R2-213 amendment 4 freezes it at {ci.FROZEN_PER_SIDE_CAP}"
            )
        if int(args.evaluation_draw_cap) != ci.FROZEN_PER_SIDE_CAP:
            raise ValueError(
                f"--evaluation-draw-cap is {args.evaluation_draw_cap} and amendment 4 "
                f"freezes it at {ci.FROZEN_PER_SIDE_CAP}. 16 was refused despite "
                "fitting the earlier bound: below about "
                f"{ci.DRAW_STABILITY_GROUP_THRESHOLD} groups per side the point "
                "estimate becomes draw-sensitive, and a cap of 16 would put every "
                "concept there"
            )
        for flag in ("evaluation_draw_seed", "draw_variance_seed"):
            if getattr(args, flag) is None:
                raise ValueError(
                    f"--evaluation-draw balanced_1to1 needs --{flag.replace('_', '-')}. "
                    "The draw's seed fixes the population once, and the second seed is "
                    "the mandatory measurement of the draw variance balancing "
                    "introduces (EXP-R2-213 amendment 3)"
                )
        if args.draw_variance_seed == args.evaluation_draw_seed:
            raise ValueError(
                "--draw-variance-seed must differ from --evaluation-draw-seed; a second "
                "draw from the same seed is the same draw and measures nothing"
            )
    elif (
        args.evaluation_draw_seed is not None
        or args.draw_variance_seed is not None
        or args.evaluation_draw_cap is not None
    ):
        raise ValueError(
            "--evaluation-draw-seed, --draw-variance-seed and --evaluation-draw-cap "
            "belong to the balanced draw; under full_split there is no draw for them to "
            "seed or cap"
        )
    if args.generation_readout == "hmmer_pfam":
        if args.hmmer_root is None or args.pfam_root is None:
            raise ValueError(
                "--generation-readout hmmer_pfam needs --hmmer-root and --pfam-root, "
                "which are working locations outside the repository: a built binary and "
                "a pressed 2 GB database never enter version control"
            )
        if args.generate_sequences < 8:
            raise ValueError(
                "A36-6's interval is a bootstrap over generated sequences and is "
                "refused below eight per side"
            )


# ------------------------------------------------------------ the shared sweep


def sweep_direction(
    handle: Any,
    batches: Sequence[ci.PreparedBatch],
    baseline: ci.ScoredResponse,
    direction: ci.ConceptDirection,
    *,
    alphas: Sequence[float],
    layer: int,
    bearing: dict[str, np.ndarray],
    bootstrap_for: Sequence[str],
    seed: int,
    n_bootstrap: int,
    device: Any,
    dtype: torch.dtype,
    subsets: Mapping[str, Any] | None = None,
) -> dict[float, dict[str, Any]]:
    """One direction over the whole ladder, scored against every concept's labels.

    One forward pass per rung serves every concept, because the pass depends only on
    the injected direction while the split into bearing and non-bearing is a
    property of the labels.

    ``bootstrap_for`` names the concepts whose *interval* is computed, and only one
    interval is ever read: A36-3(a) is read on the injected concept at a positive
    admissible rung, A36-3(c) reads point values across the nine rungs, A36-4
    compares point effects against a row percentile, and A36-3(b) is a percentile
    over directions, which point values supply. So an interval is computed for the
    injected concept at the positive rungs and nowhere else. That drops no draw, no
    rung and no control -- every one is still run and still enters the statistic it
    belongs to -- and it is what keeps the bootstrap cost linear in the concept
    count rather than quadratic.
    """

    positive = {float(alpha) for alpha in alphas if float(alpha) > 0.0}
    out: dict[float, dict[str, Any]] = {}
    for alpha in alphas:
        response = ci.scored_response(
            handle,
            batches,
            layer=layer,
            delta=direction.delta(alpha, device=device, dtype=dtype),
        )
        deltas: dict[str, Any] = {}
        coherence: dict[str, Any] = {}
        for name, flags in bearing.items():
            subset = None if subsets is None else subsets[name]
            if name in bootstrap_for and float(alpha) in positive:
                deltas[name] = ci.delta_nll_shift(
                    baseline,
                    response,
                    flags,
                    seed=seed,
                    n_bootstrap=n_bootstrap,
                    label=f"{direction.concept}@{alpha}:{name}",
                    subset=subset,
                )
            else:
                deltas[name] = {
                    "delta_nats_per_token": ci.delta_point(
                        baseline, response, flags, subset=subset
                    )
                }
            coherence[name] = ci.coherence_record(
                baseline, response, flags, subset=subset
            )
        out[float(alpha)] = {"delta": deltas, "coherence": coherence}
    return out


def controls_by_alpha(
    sweeps: Sequence[dict[float, dict[str, Any]]], concept: str, *, alphas: Sequence[float]
) -> dict[float, dict[str, Any]]:
    """A36-3(b)'s control at every rung, from the random directions' sweeps."""

    return {
        float(alpha): ci.random_direction_control(
            [
                float(sweep[float(alpha)]["delta"][concept]["delta_nats_per_token"])
                for sweep in sweeps
            ]
        )
        for alpha in alphas
    }


def evaluate_concept(
    *,
    concept: str,
    sweep: dict[float, dict[str, Any]],
    controls: dict[float, dict[str, Any]],
    alphas: Sequence[float],
    bound: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """A36-1 and A36-3 for one direction in one mode, at one coherence bound."""

    coherence = {alpha: sweep[alpha]["coherence"][concept] for alpha in alphas}
    admissible = ci.admissible_alphas(coherence, max_nll_inflation=bound)
    deltas = {alpha: sweep[alpha]["delta"][concept] for alpha in alphas}
    evaluation = ci.evaluate_a36_3(
        deltas=deltas,
        controls=controls,
        admissible=admissible,
        margin=float(args.random_null_margin),
        spearman_ceiling=float(args.spearman_ceiling),
    )
    return {
        "coherence_bound": float(bound),
        "admissible_alphas": list(admissible),
        "coherence": {str(alpha): coherence[alpha] for alpha in alphas},
        "a36_3": evaluation,
        "operating_alpha": ci.operating_alpha(evaluation, rule=args.operating_alpha_rule),
    }


# ---------------------------------------------------------------- the campaign


def load_handles(args: argparse.Namespace) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Both modes of one checkpoint, over one set of loaded weights."""

    resolved, tokenizer = STAGE21.load_tokenizer(args.checkpoint)
    ci.require_behavioural_modes(resolved)
    declaration = joint_modes.rendering(args.rendering)
    model, facts = STAGE21.load_model(
        resolved, tokenizer, device=args.device, dtype=INFERENCE_DTYPE
    )
    tokenisation = joint_modes.resolve(tokenizer, declaration)
    handles = {
        mode: JointReplaceable(
            model=model,
            tokenizer=tokenizer,
            checkpoint=resolved,
            declaration=declaration,
            mode=mode,
            tokenisation=tokenisation if mode == "protein" else None,
            max_tokens=args.max_tokens,
            protein_context=args.protein_context,
        )
        for mode in JOINT_MODES
    }
    facts["dtype_enforced"] = ci.require_full_precision(handles["protein"])
    facts["rendering"] = tokenisation.facts()
    return resolved, facts, handles


ONTOLOGY: dict[str, Any] = {}


def load_concept_rules(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The cohort's own concept specs and GO ontology, loaded once from its manifest.

    The membership rule is three-valued and belongs to
    ``src.transfer.sequence_description``; it is imported rather than re-derived, and
    the ontology comes from the OBO path the cohort itself recorded, so this stage
    labels a record exactly as the cohort did. A second implementation here would be
    a second definition of what "does not bear this concept" means, and the
    difference is 1,292 groups of unknown status on one production concept.
    """

    if ONTOLOGY:
        return ONTOLOGY
    obo = Path(str(manifest["settings"]["go_obo"]))
    ONTOLOGY["specs"] = {spec.concept_id: spec for spec in sd.CONCEPTS}
    ONTOLOGY["ontology"] = sd.load_go_ontology(obo)
    ONTOLOGY["go_obo"] = str(obo)
    return ONTOLOGY


def concept_membership(
    name: str, records: Sequence[Mapping[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    """``(bearing, defined)`` for one concept, under the cohort's three-valued rule.

    ``defined`` is false where the cohort leaves the concept UNDEFINED for a record --
    no annotation of the relevant sort -- and such a record enters neither arm of
    Delta and neither side of the coherence floor. ``concept_alignment.concept_labels``
    is deliberately not used: it is two-valued over a raw annotation column and would
    fold every undefined record into the non-bearing arm.
    """

    spec = ONTOLOGY["specs"].get(name)
    if spec is None:
        raise ValueError(
            f"{name} is not a declared concept of src.transfer.sequence_description; "
            f"declared ids are {sorted(ONTOLOGY['specs'])}"
        )
    labels = [
        sd.concept_label(
            spec,
            go_propagated=record["go_propagated"] or (),
            ec=record["ec"] or (),
            ontology=ONTOLOGY["ontology"],
        )
        for record in records
    ]
    bearing = np.asarray([value == 1 for value in labels], dtype=bool)
    defined = np.asarray([value is not None for value in labels], dtype=bool)
    return bearing, defined


def select_concepts(
    args: argparse.Namespace, eval_records: Sequence[Any]
) -> tuple[list[str], dict[str, Any]]:
    """The concept set, reduced only by the frozen rule and only if asked.

    Ascending eval bearing-group count, which is known before any activation is
    captured. Nothing else is dropped: not a rung, not a control, not a split.
    """

    counts = {}
    for name in args.concepts:
        bearing, defined = concept_membership(name, eval_records)
        groups = ci.per_side_group_counts(
            [record["dup_group"] for record in eval_records], bearing, subset=defined
        )
        groups["undefined_records"] = int((~defined).sum())
        counts[name] = groups
    ordered = sorted(counts, key=lambda name: (counts[name]["bearing"], name))
    if args.max_concepts is None or args.max_concepts >= len(ordered):
        kept, dropped = ordered, []
    else:
        dropped = ordered[: len(ordered) - args.max_concepts]
        kept = ordered[len(ordered) - args.max_concepts :]
    return sorted(kept), {
        "eval_group_counts_per_side": counts,
        "kept": sorted(kept),
        "dropped": dropped,
        "rule": "drop concepts in ascending order of their eval bearing-group count; "
        "never drop a control, a rung or a split (EXP-R2-213)",
        "interaction_with_the_balanced_draw": "under the full split a pass costs the "
        "same whatever a concept bears, so the frozen rule drops the weakest concepts. "
        "Under amendment 3's balanced draw a concept's cost is proportional to what it "
        "bears, so the same rule now drops the CHEAPEST concepts and keeps the most "
        "expensive. The rule is applied as frozen and the interaction is recorded; it is "
        "not resolved here",
    }


def permuted_entry(label: str, evaluation: dict[str, Any]) -> dict[str, Any]:
    """One permuted-label draw's A36-5 record, with its distance to the bar.

    The boolean alone is not auditable: a control that fails by a hair and one that
    fails by an order of magnitude say different things about how much room the
    readout has, and A36-5 voids the concept on a single passing draw. The largest
    ratio of ``|Delta|`` to the control bar it had to clear therefore travels with
    every draw.
    """

    ratios = [
        abs(cell["delta_nats_per_token"]) / cell["b_control_bar"]
        for cell in evaluation["a36_3"]["per_alpha"].values()
    ]
    return {
        "direction": label,
        "passes_a36_3": bool(evaluation["a36_3"]["passed"]),
        "max_abs_delta_over_control_bar": max(ratios) if ratios else None,
        "spearman": evaluation["a36_3"]["graded"]["spearman"],
        "operating_alpha": evaluation["operating_alpha"],
    }


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    """The real measurement: two modes, one checkpoint, one frozen ladder."""

    cohort = ca.load_cohort(args.cohort)
    print(f"[cohort] {len(cohort)} records, splits {cohort.counts()}", flush=True)
    # The admitted concept set is READ, and anything outside it is refused. This
    # stage derives no concept set of its own; see require_admitted_concepts.
    admitted = ci.require_admitted_concepts(
        args.concepts, cohort.manifest, source=args.cohort
    )
    handoff = ci.assert_stage35_handoff(
        json.loads(Path(args.stage35_artefact).read_text())["causal_handoff"],
        layer=int(args.layer),
        site=args.injection_site,
        pooling=args.pooling,
        source=args.stage35_artefact,
    )
    print(f"[scope] {len(admitted)} admitted concepts; stage 35 hand-off {handoff}", flush=True)
    load_concept_rules(cohort.manifest)
    fit_records = [record for record in cohort.records if record["split"] == FIT_SPLIT]
    eval_records = [record for record in cohort.records if record["split"] == args.eval_split]
    if not fit_records or not eval_records:
        raise ValueError(
            f"the cohort needs records in {FIT_SPLIT!r} and {args.eval_split!r}; it has "
            f"{len(fit_records)} and {len(eval_records)}"
        )
    names, reduction = select_concepts(args, eval_records)
    print(f"[concepts] {names} (reduction: {reduction['dropped'] or 'none'})", flush=True)
    if len(names) < 2:
        raise ValueError("fewer than two concepts survive; A36-4 would be vacuous")

    specs = ONTOLOGY["specs"]
    masking = {
        name: ci.assert_descriptions_masked(cohort.records, specs[name].name)
        for name in names
    }
    membership = {
        split: {name: concept_membership(name, records) for name in names}
        for split, records in (("fit", fit_records), ("eval", eval_records))
    }
    bearing = {
        split: {name: pair[0] for name, pair in cells.items()}
        for split, cells in membership.items()
    }
    groups_eval = [record["dup_group"] for record in eval_records]
    if args.evaluation_draw == "balanced_1to1":
        included, draw_record = ci.balanced_evaluation_draw(
            groups_eval,
            {name: membership["eval"][name] for name in names},
            seed=args.evaluation_draw_seed,
            cap=args.evaluation_draw_cap,
        )
    else:
        included = {name: membership["eval"][name][1] for name in names}
        draw_record = {
            "rule": "full_split",
            "seed": None,
            "note": "every record the cohort DEFINES the concept on; undefined records "
            "enter neither side",
            "per_concept": {
                name: {"records_scored": int(included[name].sum())} for name in names
            },
        }
    # The mandatory draw-variance measurement rides along as a second pseudo-concept
    # with the same membership and an independently seeded subset. Because a subset
    # selects rows of an already-scored pass, it costs no extra forward pass at all --
    # provided its records are in the scored union, which is why it is folded in here
    # rather than run afterwards.
    variance_names: list[str] = []
    variance_included: dict[str, np.ndarray] = {}
    if args.evaluation_draw == "balanced_1to1":
        # Every concept the cap could not lift above the draw-stability threshold is
        # covered, and cannot be opted out of: this check is the designated detector
        # for point-estimate draw sensitivity, so it is wired to the concepts that need
        # it rather than to whichever one an operator names. On the production cohort
        # the rule catches go_atp_binding, whose 15 bearing groups no cap can equalise.
        below = list(draw_record["concepts_below_draw_stability_threshold"])
        extra = [
            part.strip()
            for part in str(args.draw_variance_concept or "").split(",")
            if part.strip()
        ]
        unknown = [name for name in extra if name not in names]
        if unknown:
            raise ValueError(
                f"--draw-variance-concept names {unknown}, which are not among the "
                f"measured concepts {names}"
            )
        variance_names = sorted(set(below) | set(extra))
        if not variance_names:
            # Amendment 3 makes the measurement mandatory, so it runs on the concept
            # with the fewest groups even when every concept clears the threshold.
            variance_names = [
                min(
                    names,
                    key=lambda name: draw_record["per_concept"][name][
                        "smaller_side_groups"
                    ],
                )
            ]
        second, second_record = ci.balanced_evaluation_draw(
            groups_eval,
            {name: membership["eval"][name] for name in variance_names},
            seed=args.draw_variance_seed,
            cap=args.evaluation_draw_cap,
        )
        variance_included = {name: second[name] for name in variance_names}
        draw_record["draw_variance"] = {
            "concepts": variance_names,
            "seed": int(args.draw_variance_seed),
            "coverage_rule": "every concept below the draw-stability threshold of "
            f"{ci.DRAW_STABILITY_GROUP_THRESHOLD} groups per side, plus any named by "
            "--draw-variance-concept; the threshold set cannot be opted out of",
            "concepts_below_threshold": below,
            "per_concept": second_record["per_concept"],
        }
        print(
            f"[draw variance] covering {variance_names} "
            f"(below threshold: {below or 'none'})",
            flush=True,
        )

    # A subset selects ROWS of a scored pass, so the balanced draw only reduces
    # compute if it also decides which records are scored. The scored population is
    # the union of every concept's drawn subset plus the variance draw's.
    scored_mask = np.zeros(len(eval_records), dtype=bool)
    for keep in included.values():
        scored_mask |= keep
    for keep in variance_included.values():
        scored_mask |= keep
    scored_records = [
        record for record, keep in zip(eval_records, scored_mask) if keep
    ]
    draw_record["records_scored_by_the_forward_passes"] = len(scored_records)
    draw_record["records_in_the_declared_split"] = len(eval_records)

    def reindex(mask: np.ndarray) -> np.ndarray:
        return np.asarray(mask, dtype=bool)[scored_mask]

    scored_groups = [record["dup_group"] for record in scored_records]
    scored_membership = {
        name: (
            reindex(bearing["eval"][name]),
            reindex(membership["eval"][name][1]),
        )
        for name in names
    }
    cells = {
        name: ci.ConceptCell(
            concept=name,
            bearing=scored_membership[name][0],
            defined=scored_membership[name][1],
            included=reindex(included[name]),
        )
        for name in names
    }
    for name in variance_names:
        pseudo = f"{name}{DRAW_VARIANCE_SUFFIX}"
        cells[pseudo] = ci.ConceptCell(
            concept=pseudo,
            bearing=scored_membership[name][0],
            defined=scored_membership[name][1],
            included=reindex(variance_included[name]),
        )
    for name in cells:
        counts = ci.require_per_side_group_floor(
            scored_groups,
            cells[name].bearing,
            label=f"{name} in {args.eval_split} under {args.evaluation_draw}",
            subset=cells[name].subset,
        )
        print(f"[draw] {name}: {counts} groups, {cells[name].counts()}", flush=True)

    resolved, facts, handles = load_handles(args)
    print(f"[load] {resolved} at {INFERENCE_DTYPE} on {args.device}", flush=True)

    representations = ca.mode_representations(
        handles["text"],
        args.rendering,
        "text",
        [str(record[TEXT_FIELD]) for record in fit_records],
        (int(args.layer),),
        args.pooling,
        args.device,
        args.batch_size,
        INFERENCE_DTYPE,
    )[int(args.layer)]
    directions = {
        name: ci.ConceptDirection.from_concept_vector(
            ca.concept_vector(representations, bearing["fit"][name]),
            concept=name,
            layer=int(args.layer),
            provenance=(
                f"concept_alignment.concept_vector on {len(fit_records)} {FIT_SPLIT} "
                f"records' TEXT-mode {args.pooling} representations at layer "
                f"{args.layer}, from {TEXT_FIELD}"
            ),
        )
        for name in names
    }

    batches = {
        mode: ci.prepare_batches(handles[mode], scored_records, batch_size=args.batch_size)
        for mode in JOINT_MODES
    }
    invariant_record = {
        mode: ci.invariants(
            handles[mode],
            batches[mode][0],
            layer=int(args.layer),
            scale=args.invariant_scale,
            seed=args.seed,
        )
        for mode in JOINT_MODES
    }
    print(
        f"[A36-0] protein null {invariant_record['protein']['null_patch_max_logit_gap']:.3g}"
        f" moved {invariant_record['protein']['perturbed_patch_max_logit_gap']:.3g}",
        flush=True,
    )
    baseline = {
        mode: ci.scored_response(
            handles[mode], batches[mode], layer=int(args.layer), delta=None
        )
        for mode in JOINT_MODES
    }
    device = handles["protein"].device
    dtype = next(handles["protein"].model.parameters()).dtype

    def sweep(handle_mode: str, direction: ci.ConceptDirection, bootstrap_for: Sequence[str]):
        return sweep_direction(
            handles[handle_mode],
            batches[handle_mode],
            baseline[handle_mode],
            direction,
            alphas=args.alphas,
            layer=int(args.layer),
            bearing={name: cells[name].bearing for name in cells},
            bootstrap_for=bootstrap_for,
            seed=args.seed,
            n_bootstrap=args.bootstrap_draws,
            device=device,
            dtype=dtype,
            subsets={name: cells[name].subset for name in cells},
        )

    random_sweeps: dict[str, dict[str, list[dict[float, dict[str, Any]]]]] = {}
    concept_sweeps: dict[str, dict[str, dict[float, dict[str, Any]]]] = {
        mode: {} for mode in JOINT_MODES
    }
    permuted_sweeps: dict[str, list[tuple[str, dict[float, dict[str, Any]]]]] = {}
    for name in names:
        # The variance pseudo-concept is bootstrapped too, so both draws carry an
        # interval and the comparison is between two intervals rather than two points.
        bootstrap_names = [name]
        if name in variance_names:
            bootstrap_names.append(f"{name}{DRAW_VARIANCE_SUFFIX}")
        for mode in JOINT_MODES:
            concept_sweeps[mode][name] = sweep(mode, directions[name], bootstrap_names)
        print(f"[ladder] {name} scored in both modes", flush=True)
        random_sweeps[name] = {
            mode: [
                # A36-3(b) is a percentile over directions, so a control draw
                # contributes a point value; its own interval is never read. No
                # draw and no control is dropped by that economy.
                sweep(mode, draw, ())
                for draw in ci.norm_matched_random_directions(
                    directions[name], n_draws=args.random_directions, seed=args.seed
                )
            ]
            for mode in JOINT_MODES
        }
        permuted_sweeps[name] = [
            (draw.concept, sweep("protein", draw, [name]))
            for draw in ci.permuted_label_directions(
                representations,
                bearing["fit"][name],
                concept=name,
                layer=int(args.layer),
                n_draws=args.permuted_directions,
                seed=args.seed + 1,
            )
        ]
        print(f"[controls] {name}: random and permuted sweeps complete", flush=True)

    per_concept: dict[str, Any] = {}
    matrix_cells: dict[tuple[str, str], float] = {}
    for name in names:
        controls = {
            mode: controls_by_alpha(random_sweeps[name][mode], name, alphas=args.alphas)
            for mode in JOINT_MODES
        }
        by_bound = {
            str(bound): {
                mode: evaluate_concept(
                    concept=name,
                    sweep=concept_sweeps[mode][name],
                    controls=controls[mode],
                    alphas=args.alphas,
                    bound=bound,
                    args=args,
                )
                for mode in JOINT_MODES
            }
            for bound in sorted({*ci.COHERENCE_SENSITIVITY_BOUNDS, args.coherence_max_nll_inflation})
        }
        primary = by_bound[str(float(args.coherence_max_nll_inflation))]
        permuted = []
        for label, sweep_result in permuted_sweeps[name]:
            evaluation = evaluate_concept(
                concept=name,
                sweep=sweep_result,
                controls=controls["protein"],
                alphas=args.alphas,
                bound=args.coherence_max_nll_inflation,
                args=args,
            )
            permuted.append(permuted_entry(label, evaluation))
        # A36-5 as amended: the permuted draws form a control DISTRIBUTION at each
        # rung, read the way A36-3(b) reads the isotropic one.
        permuted_controls = {
            float(alpha): ci.permuted_label_control(
                [
                    float(sweep_result[float(alpha)]["delta"][name]["delta_nats_per_token"])
                    for _, sweep_result in permuted_sweeps[name]
                ]
            )
            for alpha in args.alphas
        }
        a36_5 = ci.evaluate_a36_5(
            deltas={
                alpha: primary["protein"]["a36_3"]["per_alpha"][str(alpha)]
                for alpha in primary["protein"]["a36_3"]["a_and_b_firing_alphas"]
            },
            permuted_controls=permuted_controls,
            firing_alphas=primary["protein"]["a36_3"]["a_and_b_firing_alphas"],
            margin=float(args.random_null_margin),
            per_draw=permuted,
        )
        operating = primary["protein"]["operating_alpha"]
        if operating is not None:
            for scored in names:
                matrix_cells[(name, scored)] = float(
                    concept_sweeps["protein"][name][operating]["delta"][scored][
                        "delta_nats_per_token"
                    ]
                )
        per_concept[name] = {
            "direction": directions[name].record(),
            "masking_check": masking[name],
            "eval_groups_per_side": reduction["eval_group_counts_per_side"][name],
            "by_coherence_bound": by_bound,
            "primary_bound": float(args.coherence_max_nll_inflation),
            "random_direction_control": {
                mode: {str(alpha): controls[mode][alpha] for alpha in args.alphas}
                for mode in JOINT_MODES
            },
            "permuted_label_control": permuted,
            "permuted_label_control_distribution": {
                str(alpha): permuted_controls[float(alpha)] for alpha in args.alphas
            },
            "a36_5": a36_5,
            "evaluation_draw": {
                **draw_record["per_concept"][name],
                "cell": cells[name].counts(),
            },
            "operating_alpha": operating,
        }

    # A36-4 is read at each concept's OWN operating rung, so a concept with no
    # operating rung has no rung at which its row is defined. It is excluded from
    # the matrix with that reason rather than making the matrix uncomputable for
    # every other concept -- which would report a concept whose specificity was
    # never evaluated as one that failed it.
    complete = [name for name in names if per_concept[name]["operating_alpha"] is not None]
    excluded = sorted(set(names) - set(complete))
    if len(complete) >= 2:
        cells_for_matrix = {
            (injected, scored): matrix_cells[(injected, scored)]
            for injected in complete
            for scored in complete
        }
        specificity = ci.specificity_matrix(cells_for_matrix, complete, rule=args.specificity_rule)
        specificity["concepts_excluded_for_want_of_an_operating_alpha"] = excluded
        # A36-4's precision equalisation, and the rows it could not reach. The cap
        # equalises the diagonal and every off-diagonal cell of a row at once, which
        # is what makes the row's 95th percentile a like-for-like referent -- but a
        # concept that never had the groups cannot be lifted by any cap, so its row
        # is FLAGGED here rather than averaged in.
        specificity["precision_equalisation"] = {
            "per_side_cap": None
            if args.evaluation_draw != "balanced_1to1"
            else int(args.evaluation_draw_cap),
            "groups_per_side": {
                name: draw_record["per_concept"][name]["smaller_side_groups"]
                for name in complete
            },
            "rows_the_cap_could_not_equalise": [
                name
                for name in complete
                if not draw_record["per_concept"][name][
                    "above_draw_stability_threshold"
                ]
            ],
            "why_this_matters": "within one row, cell (A, B) is Delta for concept B on "
            "B's own subset at B's own group count, so the off-diagonal referent set "
            "itself mixes precisions and uncapped a 15-group diagonal could be compared "
            "against a percentile over 985-group cells. The heterogeneity is within-row "
            "as well as across-row and the cap addresses both",
            "what_it_does_not_do": "capping equalises NOISE and not SIGNAL. A36-4 still "
            "compares raw effects, so a concept with a genuinely larger effect still "
            "dominates its row, and this matrix must never be read as a matrix of "
            "t-statistics",
        }
    else:
        specificity = {
            "criterion": "A36-4",
            "status": "NOT_COMPUTED",
            "reason": "fewer than two concepts have an operating rung -- A36-3(a) and "
            "(b) fire at no admissible positive rung for "
            f"{excluded} -- so there is no off-diagonal and A36-4 is vacuous rather "
            "than failed",
            "concepts_with_an_operating_alpha": complete,
        }

    verdicts = {
        name: ci.verdict(
            concept=name,
            text_control=per_concept[name]["by_coherence_bound"][
                str(float(args.coherence_max_nll_inflation))
            ]["text"]["a36_3"],
            protein=per_concept[name]["by_coherence_bound"][
                str(float(args.coherence_max_nll_inflation))
            ]["protein"]["a36_3"],
            permuted=per_concept[name]["a36_5"],
            specificity_row=(
                specificity.get("per_concept", {}).get(name)
                if specificity.get("status") != "NOT_COMPUTED"
                else None
            ),
            admissible=per_concept[name]["by_coherence_bound"][
                str(float(args.coherence_max_nll_inflation))
            ]["protein"]["admissible_alphas"],
        )
        for name in names
    }
    for name in names:
        print(f"[verdict] {name}: {verdicts[name]['outcome']}", flush=True)

    draw_variance: dict[str, Any] | None = None
    if variance_names:
        threshold = ci.DRAW_STABILITY_GROUP_THRESHOLD
        positive = [alpha for alpha in args.alphas if float(alpha) > 0.0]
        per_concept_variance: dict[str, Any] = {}
        for name in variance_names:
            pseudo = f"{name}{DRAW_VARIANCE_SUFFIX}"
            ladder = concept_sweeps["protein"][name]
            rows = {}
            for alpha in positive:
                first = ladder[float(alpha)]["delta"][name]
                second = ladder[float(alpha)]["delta"][pseudo]
                half = (
                    None
                    if not first.get("delta_ci95")
                    else (first["delta_ci95"][1] - first["delta_ci95"][0]) / 2.0
                )
                rows[str(alpha)] = {
                    "draw_1": {
                        "delta_nats_per_token": first["delta_nats_per_token"],
                        "delta_ci95": first.get("delta_ci95"),
                        "n_groups_per_side": first.get("n_groups_per_side"),
                    },
                    "draw_2": {
                        "delta_nats_per_token": second["delta_nats_per_token"],
                        "delta_ci95": second.get("delta_ci95"),
                        "n_groups_per_side": second.get("n_groups_per_side"),
                    },
                    "difference": float(
                        second["delta_nats_per_token"] - first["delta_nats_per_token"]
                    ),
                    "difference_over_draw_1_ci_halfwidth": (
                        None
                        if not half
                        else float(
                            abs(
                                second["delta_nats_per_token"]
                                - first["delta_nats_per_token"]
                            )
                            / max(1e-12, half)
                        )
                    ),
                }
            entry = draw_record["per_concept"][name]
            per_concept_variance[name] = {
                "smaller_side_groups": entry["smaller_side_groups"],
                "above_draw_stability_threshold": entry[
                    "above_draw_stability_threshold"
                ],
                "covered_because": (
                    "below the draw-stability threshold"
                    if not entry["above_draw_stability_threshold"]
                    else "named by --draw-variance-concept, or the smallest concept when "
                    "every concept clears the threshold"
                ),
                "per_alpha": rows,
            }
            for alpha, row in rows.items():
                print(
                    f"[draw variance] {name} a={alpha}: "
                    f"{row['draw_1']['delta_nats_per_token']:+.6f} vs "
                    f"{row['draw_2']['delta_nats_per_token']:+.6f} "
                    f"(ratio to CI half-width "
                    f"{row['difference_over_draw_1_ci_halfwidth']})",
                    flush=True,
                )
        draw_variance = {
            "criterion": f"{ci.PRE_REGISTRATION} amendments 3 and 4, mandatory",
            "concepts": variance_names,
            "seeds": [int(args.evaluation_draw_seed), int(args.draw_variance_seed)],
            "per_side_cap": int(args.evaluation_draw_cap),
            "draw_stability_threshold_groups": threshold,
            "concepts_below_threshold": list(
                draw_record["concepts_below_draw_stability_threshold"]
            ),
            "per_concept": per_concept_variance,
            "reading": "balancing turns the non-bearing arm from the whole declared "
            "split into a SAMPLE, so it carries draw variance the full-split design did "
            "not. This is that variance, measured against a second independent seed on "
            "every concept the cap could not lift above the stability threshold. The "
            "ratio of the between-draw difference to the within-draw interval "
            "half-width is the number to read: at or below one the draw is not moving "
            "the result beyond its own sampling error. It is the designated detector "
            "for point-estimate draw sensitivity, which a wide interval does not show "
            f"-- measured on the fixture, at 12 groups per side a single subsample moved "
            "|Delta|/bar from about 1.2 to 0.78 across the decision boundary while its "
            "interval stayed at 28% of |Delta|",
            "cost_note": "a subset selects rows of an already-scored pass, so every "
            "second-draw record was folded into the scored union and this measurement "
            "cost no additional forward pass",
        }

    generation = run_generation_readout(
        args,
        handles["protein"],
        directions,
        fit_records=fit_records,
        bearing_fit={name: bearing["fit"][name] for name in names},
        operating={name: per_concept[name]["operating_alpha"] for name in names},
        device=device,
        dtype=dtype,
    )

    return {
        "kind": "concept_injection_campaign",
        "checkpoint": facts,
        "cohort": {
            "path": str(args.cohort),
            "counts": cohort.counts(),
            "fit_split": FIT_SPLIT,
            "eval_split": args.eval_split,
            "n_fit_records": len(fit_records),
            "n_eval_records": len(eval_records),
            "text_field": TEXT_FIELD,
            "representation_site": ca.REPRESENTATION_SITE,
            "representation_site_note": ca.REPRESENTATION_SITE_NOTE,
        },
        "concept_scope": {
            "admitted_by_the_cohort": list(admitted),
            "measured": list(names),
            "rule": "the admitted list is READ from the cohort artefact and anything "
            "outside it is refused; this stage derives no concept set of its own",
            "stage35_handoff": handoff,
        },
        "concept_reduction": reduction,
        "evaluation_draw": draw_record,
        "draw_variance": draw_variance,
        "invariants": invariant_record,
        "baseline": {
            mode: {
                "nll_nats_per_token": float(
                    np.average(
                        baseline[mode].nll_per_token,
                        weights=baseline[mode].scored_tokens.astype(float),
                    )
                ),
                "scored_tokens": int(baseline[mode].scored_tokens.sum()),
                "n_records": int(baseline[mode].nll_per_token.size),
            }
            for mode in JOINT_MODES
        },
        "per_concept": per_concept,
        "specificity": specificity,
        "readout_b": generation,
        "verdicts": verdicts,
        "attribution": {
            "rule": "A36-7: an effect on both checkpoints is not attributable to "
            "instruction tuning; on ProLLaMA alone it localises to instruction tuning; "
            "on ProLLaMA_Stage_1 alone it is the only configuration implicating "
            "continued pretraining, with the caveat that the two stages differ by only "
            "0.10 nats/token in text and 0.03 in protein, so a null between them may be "
            "a power result rather than a mechanism result",
            "this_run_measures": str(Path(args.checkpoint).resolve().name),
        },
    }


def run_generation_readout(
    args: argparse.Namespace,
    handle: Any,
    directions: dict[str, ci.ConceptDirection],
    *,
    fit_records: Sequence[Any],
    bearing_fit: dict[str, np.ndarray],
    operating: dict[str, float | None],
    device: Any,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """A36-6, attainability first: the alpha=0 annotation rate before anything else."""

    clean = ci.clean_availability(REPO_ROOT / "external_resources/ec_metrics/clean/CLEAN")
    if args.generation_readout == "refused":
        return {
            "criterion": "A36-6",
            "status": "REFUSED",
            "reason": args.generation_refusal_reason,
            "clean": clean,
        }
    tool = ci.prepare_hmmer(
        REPO_ROOT / "external_resources/tools/hmmer-3.4.tar.gz", args.hmmer_root
    )
    database = ci.prepare_pfam(
        REPO_ROOT / "external_resources/ec_metrics/pfam/Pfam-A.hmm.gz",
        REPO_ROOT / "external_resources/ec_metrics/pfam/Pfam-A.hmm.gz.sha256",
        args.pfam_root,
        tool=tool,
    )
    print(f"[A36-6] HMMER {tool.version}, Pfam-A {database.n_profiles} profiles", flush=True)
    declaration = handle.declaration
    prefix = (
        ""
        if args.protein_context is None
        else declaration.protein_context_template.format(context=args.protein_context)
    )
    prompt = prefix + declaration.protein_start
    workspace = Path(args.out) / "generated"
    workspace.mkdir(parents=True, exist_ok=True)

    def annotate(label: str, delta: torch.Tensor | None) -> tuple[dict[str, Any], dict[str, Any]]:
        raw = ci.generate_under_injection(
            handle,
            prompt,
            n_sequences=args.generate_sequences,
            max_new_tokens=args.generate_max_new_tokens,
            temperature=args.generate_temperature,
            top_p=args.generate_top_p,
            seed=args.seed,
            layer=int(args.layer),
            delta=delta,
        )
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", label)
        sequences = {
            f"{safe}_{index:04d}": sequence
            for index, sequence in enumerate(
                ci.extract_generated_sequence(text, end_delimiter=declaration.protein_end)
                for text in raw
            )
            if len(sequence) >= 20
        }
        record = {
            "n_requested": int(args.generate_sequences),
            "n_usable": len(sequences),
            "mean_length": (
                float(np.mean([len(value) for value in sequences.values()]))
                if sequences
                else 0.0
            ),
            "usable_rule": "the residue run before the closing delimiter, at least 20 "
            "residues; a continuation that wandered back into prose contributes a short "
            "sequence or none",
        }
        if not sequences:
            return record, {}
        fasta = ci.write_fasta(workspace / f"{safe}.fasta", sequences)
        table = workspace / f"{safe}.tbl"
        command, _ = ci.run_hmmscan(
            tool,
            database,
            fasta,
            table,
            evalue=args.hmmscan_evalue,
            threads=args.hmmscan_threads,
        )
        record["hmmscan_command"] = command
        record["sequence_names"] = list(sequences)
        return record, ci.parse_hmmscan_table(table)

    base_record, base_hits = annotate("alpha_0", None)
    base_names = base_record.get("sequence_names", [])
    attainability = (
        {"any_family_rate": 0.0, "n_sequences": 0}
        if not base_names
        else ci.annotation_rates(base_hits, base_names, [])
    )
    print(
        f"[A36-6] attainability: alpha=0 annotates "
        f"{attainability['any_family_rate']:.3f} of {attainability['n_sequences']}",
        flush=True,
    )
    if attainability["any_family_rate"] <= 0.0:
        return {
            "criterion": "A36-6",
            "status": "VOID_ZERO_BASE_RATE",
            "reason": "at alpha = 0 the generator annotates against Pfam-A at rate zero, "
            "so no increase is measurable against it and a null would be "
            "indistinguishable from an unreachable statistic. Readout A stands alone and "
            "the claim is worded as resting on the model's own likelihood with no "
            "independent instrument",
            "attainability": attainability,
            "alpha_0": base_record,
            "hmmer": tool.record(),
            "pfam": database.record(),
            "clean": clean,
        }

    per_concept: dict[str, Any] = {}
    for name, direction in directions.items():
        referent = ci.pfam_referent(
            fit_records,
            bearing_fit[name],
            min_bearing_records=args.pfam_referent_min_bearing,
        )
        alpha = operating.get(name)
        if not referent:
            per_concept[name] = {
                "status": "REFUSED_NO_REFERENT",
                "reason": "no Pfam family appears on at least "
                f"{args.pfam_referent_min_bearing} of this concept's fit-split bearing "
                "records, so there is no external referent and a mapping invented here "
                "would be the stage scoring itself",
            }
            continue
        if alpha is None:
            per_concept[name] = {
                "status": "REFUSED_NO_OPERATING_ALPHA",
                "reason": "A36-3(a) and (b) fire at no admissible positive rung, so "
                "there is no admissible alpha at which to generate",
                "referent": list(referent),
            }
            continue
        random_draw = ci.norm_matched_random_directions(
            direction, n_draws=args.random_directions, seed=args.seed
        )[0]
        entry: dict[str, Any] = {
            "status": "SCORED",
            "referent": list(referent),
            "operating_alpha": float(alpha),
            "baseline": ci.annotation_rates(base_hits, base_names, referent),
        }
        for label, drawn, coefficient in (
            ("injected", direction, float(alpha)),
            ("reversed", direction, -float(alpha)),
            ("random_direction", random_draw, float(alpha)),
        ):
            record, hits = annotate(
                f"{name}__{label}", drawn.delta(coefficient, device=device, dtype=dtype)
            )
            sequence_names = record.get("sequence_names", [])
            if not sequence_names:
                entry[label] = {"status": "NO_USABLE_SEQUENCE", "generation": record}
                continue
            rates = ci.annotation_rates(hits, sequence_names, referent)
            entry[label] = {"generation": record, "rates": rates}
            if len(sequence_names) >= 8 and len(base_names) >= 8:
                entry[label]["against_alpha_0"] = ci.annotation_rate_contrast(
                    rates,
                    entry["baseline"],
                    seed=args.seed,
                    n_bootstrap=args.bootstrap_draws,
                )
        if "injected" in entry and "rates" in entry["injected"]:
            random_rate = entry.get("random_direction", {}).get("rates")
            entry["clears_random_direction"] = (
                None
                if random_rate is None
                else bool(
                    entry["injected"]["rates"]["concept_family_rate"]
                    > random_rate["concept_family_rate"]
                )
            )
        per_concept[name] = entry
    return {
        "criterion": "A36-6",
        "status": "SCORED",
        "prompt": prompt,
        "attainability": attainability,
        "alpha_0": base_record,
        "hmmer": tool.record(),
        "pfam": database.record(),
        "clean": clean,
        "per_concept": per_concept,
        "referent_note": "the concept's Pfam referent is derived from the cohort's own "
        "pfam column on the FIT split alone; EXP-R2-213's concepts are GO terms and EC "
        "numbers, so no concept is itself a Pfam accession",
    }


# ------------------------------------------------------------ the known answer


def run_synthetic_check(args: argparse.Namespace) -> dict[str, Any]:
    """Run the whole pipeline on a toy decoder whose answer is known.

    Two cells. ``shared_concept`` plants a direction the head reads in BOTH modes,
    so A36-2, A36-3 and A36-4 must all pass and neither control may fire.
    ``text_only_concept`` plants the same direction and lets the head read it in
    text mode only, so A36-2 must still fire -- otherwise the check is measuring
    nothing -- and A36-3 must not: the outcome must be MEASURED_NEGATIVE. That is
    the false-positive case, and a pipeline that reports TRANSFERS there reports a
    transfer that does not exist.

    ``torch.set_num_threads(1)`` is set because the toy is 64-dimensional and the
    host has 96 cores: measured on B, one forward of this model costs 3.2 s across
    96 threads and 0.004 s on one, all of it thread launch. It changes no number.
    """

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    cells: dict[str, Any] = {}
    for cell, protein_gain in (
        ("shared_concept", float(args.synthetic_protein_gain)),
        ("text_only_concept", 0.0),
    ):
        model = ci.SyntheticConceptModel(
            d_model=args.synthetic_d_model,
            n_layers=args.synthetic_layers,
            n_concepts=args.synthetic_concepts,
            mlp_gain=float(args.synthetic_mlp_gain),
            head_scale=float(args.synthetic_head_scale),
            text_gain=float(args.synthetic_text_gain),
            protein_gain=protein_gain,
            seed=args.seed,
        )
        handles = {mode: ci.SyntheticJointHandle(model, mode=mode) for mode in JOINT_MODES}
        records, names = ci.synthetic_records(
            model,
            n_records=args.synthetic_records,
            span=args.synthetic_span,
            seed=args.seed,
        )
        batches = {
            mode: ci.synthetic_batches(handles[mode], records, batch_size=args.batch_size)
            for mode in JOINT_MODES
        }
        bearing = {name: ci.bearing_flags(records, name) for name in names}
        invariant_record = {
            mode: ci.invariants(
                handles[mode],
                batches[mode][0],
                layer=int(args.layer),
                scale=args.invariant_scale,
                seed=args.seed,
            )
            for mode in JOINT_MODES
        }
        representations, _ = ci.pooled_representations(
            handles["text"], batches["text"], layer=int(args.layer)
        )
        directions = {
            name: ci.ConceptDirection.from_concept_vector(
                ca.concept_vector(representations, bearing[name]),
                concept=name,
                layer=int(args.layer),
                provenance="concept_alignment.concept_vector on the toy model's "
                "TEXT-mode content-mean representations",
            )
            for name in names
        }
        planted_cosine = {
            name: float(
                abs(
                    np.dot(
                        directions[name].direction,
                        model.concept_directions[index].numpy().astype(np.float64),
                    )
                )
            )
            for index, name in enumerate(names)
        }
        baseline = {
            mode: ci.scored_response(
                handles[mode], batches[mode], layer=int(args.layer), delta=None
            )
            for mode in JOINT_MODES
        }
        dtype = model.embedding.dtype

        def sweep(mode: str, direction: ci.ConceptDirection, bootstrap_for: Sequence[str]):
            return sweep_direction(
                handles[mode],
                batches[mode],
                baseline[mode],
                direction,
                alphas=args.alphas,
                layer=int(args.layer),
                bearing=bearing,
                bootstrap_for=bootstrap_for,
                seed=args.seed,
                n_bootstrap=args.bootstrap_draws,
                device="cpu",
                dtype=dtype,
            )

        per_concept: dict[str, Any] = {}
        matrix_cells: dict[tuple[str, str], float] = {}
        concept_sweeps = {
            mode: {name: sweep(mode, directions[name], [name]) for name in names}
            for mode in JOINT_MODES
        }
        for name in names:
            controls = {
                mode: controls_by_alpha(
                    [
                        sweep(mode, draw, ())
                        for draw in ci.norm_matched_random_directions(
                            directions[name], n_draws=args.random_directions, seed=args.seed
                        )
                    ],
                    name,
                    alphas=args.alphas,
                )
                for mode in JOINT_MODES
            }
            evaluations = {
                mode: evaluate_concept(
                    concept=name,
                    sweep=concept_sweeps[mode][name],
                    controls=controls[mode],
                    alphas=args.alphas,
                    bound=args.coherence_max_nll_inflation,
                    args=args,
                )
                for mode in JOINT_MODES
            }
            permuted = []
            permuted_deltas: dict[float, list[float]] = {
                float(alpha): [] for alpha in args.alphas
            }
            for draw in ci.permuted_label_directions(
                representations,
                bearing[name],
                concept=name,
                layer=int(args.layer),
                n_draws=args.permuted_directions,
                seed=args.seed + 1,
            ):
                drawn = sweep("protein", draw, [name])
                evaluation = evaluate_concept(
                    concept=name,
                    sweep=drawn,
                    controls=controls["protein"],
                    alphas=args.alphas,
                    bound=args.coherence_max_nll_inflation,
                    args=args,
                )
                permuted.append(permuted_entry(draw.concept, evaluation))
                for alpha in args.alphas:
                    permuted_deltas[float(alpha)].append(
                        float(drawn[float(alpha)]["delta"][name]["delta_nats_per_token"])
                    )
            permuted_controls = {
                alpha: ci.permuted_label_control(values)
                for alpha, values in permuted_deltas.items()
            }
            a36_5 = ci.evaluate_a36_5(
                deltas={
                    alpha: evaluations["protein"]["a36_3"]["per_alpha"][str(alpha)]
                    for alpha in evaluations["protein"]["a36_3"]["a_and_b_firing_alphas"]
                },
                permuted_controls=permuted_controls,
                firing_alphas=evaluations["protein"]["a36_3"]["a_and_b_firing_alphas"],
                margin=float(args.random_null_margin),
                per_draw=permuted,
            )
            operating = evaluations["protein"]["operating_alpha"]
            if operating is not None:
                for scored in names:
                    matrix_cells[(name, scored)] = float(
                        concept_sweeps["protein"][name][operating]["delta"][scored][
                            "delta_nats_per_token"
                        ]
                    )
            per_concept[name] = {
                "planted_cosine": planted_cosine[name],
                "sigma": directions[name].sigma,
                "delta_by_alpha": {
                    str(alpha): concept_sweeps["protein"][name][alpha]["delta"][name][
                        "delta_nats_per_token"
                    ]
                    for alpha in args.alphas
                },
                "non_bearing_inflation_by_alpha": {
                    str(alpha): concept_sweeps["protein"][name][alpha]["coherence"][name][
                        "non_bearing_nll_inflation_nats_per_token"
                    ]
                    for alpha in args.alphas
                },
                "protein": evaluations["protein"],
                "text": evaluations["text"],
                "random_direction_control_at_operating_alpha": (
                    None
                    if operating is None
                    else controls["protein"][operating]
                ),
                "permuted_label_control": permuted,
                "permuted_label_control_distribution": {
                    str(alpha): permuted_controls[float(alpha)] for alpha in args.alphas
                },
                "a36_5": a36_5,
                "operating_alpha": operating,
            }
        complete = [name for name in names if per_concept[name]["operating_alpha"] is not None]
        specificity = (
            ci.specificity_matrix(
                {
                    (injected, scored): matrix_cells[(injected, scored)]
                    for injected in complete
                    for scored in complete
                },
                complete,
                rule=args.specificity_rule,
            )
            if len(complete) >= 2
            else {
                "criterion": "A36-4",
                "status": "NOT_COMPUTED",
                "reason": "fewer than two concepts have an operating rung",
                "concepts_with_an_operating_alpha": complete,
            }
        )
        verdicts = {
            name: ci.verdict(
                concept=name,
                text_control=per_concept[name]["text"]["a36_3"],
                protein=per_concept[name]["protein"]["a36_3"],
                permuted=per_concept[name]["a36_5"],
                specificity_row=(
                    specificity.get("per_concept", {}).get(name)
                    if specificity.get("status") != "NOT_COMPUTED"
                    else None
                ),
                admissible=per_concept[name]["protein"]["admissible_alphas"],
            )
            for name in names
        }
        cells[cell] = {
            "planted": {
                **model.declaration(),
                "n_records": len(records),
                "slice_span": int(args.synthetic_span),
                "bearing_records_per_concept": {
                    name: int(bearing[name].sum()) for name in names
                },
            },
            "invariants": invariant_record,
            "per_concept": per_concept,
            "specificity": specificity,
            "verdicts": verdicts,
            "expected_outcome": "TRANSFERS"
            if protein_gain > 0.0
            else "MEASURED_NEGATIVE",
            "outcome_counts": {
                outcome: sum(
                    1 for name in names if verdicts[name]["outcome"] == outcome
                )
                for outcome in ci.OUTCOMES
                if any(verdicts[name]["outcome"] == outcome for name in names)
            },
            "n_matching_expected": sum(
                1
                for name in names
                if verdicts[name]["outcome"]
                == ("TRANSFERS" if protein_gain > 0.0 else "MEASURED_NEGATIVE")
            ),
            "worst_permuted_distance_to_its_bar": max(
                (
                    entry["max_abs_delta_over_control_bar"]
                    for name in names
                    for entry in per_concept[name]["permuted_label_control"]
                    if entry["max_abs_delta_over_control_bar"] is not None
                ),
                default=None,
            ),
            "every_concept_matches_expected": all(
                verdicts[name]["outcome"]
                == ("TRANSFERS" if protein_gain > 0.0 else "MEASURED_NEGATIVE")
                for name in names
            ),
        }
        print(
            f"[{cell}] "
            + ", ".join(f"{name}={verdicts[name]['outcome']}" for name in names)
            + f" (expected {cells[cell]['expected_outcome']})",
            flush=True,
        )

    return {
        "kind": "synthetic_instrument_check",
        "torch_threads": int(torch.get_num_threads()),
        "cells": cells,
        "known_answer_recovered": all(
            cell["every_concept_matches_expected"] for cell in cells.values()
        ),
        "reading": {
            "shared_concept": "the planted direction is read by the head in both modes, "
            "so every frozen clause must pass: Delta negative at an admissible positive "
            "rung with an interval excluding zero, at least twice the random-direction "
            "control's 95th percentile of |Delta|, Spearman at most -0.8 over the nine "
            "rungs with a sign reversal below zero, and each concept's diagonal above its "
            "own row's off-diagonal 95th percentile",
            "text_only_concept": "the same direction, read by the head in TEXT mode "
            "only. A36-2 must still fire -- otherwise the check is measuring nothing -- "
            "and A36-3 must not, so the outcome is MEASURED_NEGATIVE. A pipeline that "
            "reports TRANSFERS here reports a transfer that does not exist",
            "controls": "a norm-matched random direction and a permuted-label concept "
            "vector steer nothing by construction, so neither may pass A36-3 in either "
            "cell",
        },
    }


# ------------------------------------------------------------------------ main


def main() -> None:
    args = build_parser().parse_args()
    resolve(args)
    args.out.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pre_registration": ci.PRE_REGISTRATION,
        "pre_registration_amendments": list(ci.PRE_REGISTRATION_AMENDMENTS),
        "amendment_note": ci.AMENDMENT_3_NOTE,
        "settings": {
            key: (
                str(value)
                if isinstance(value, Path)
                else list(value)
                if isinstance(value, tuple)
                else value
            )
            for key, value in vars(args).items()
            if not (args.synthetic and key in CAMPAIGN_ONLY_FLAGS)
        },
        "frozen_criteria": {
            "required_flags": list(PRE_REGISTERED_DECISIONS),
            "alpha_ladder": list(ci.FROZEN_ALPHA_LADDER),
            "coherence_primary_bound": ci.COHERENCE_PRIMARY_BOUND,
            "coherence_sensitivity_bounds": list(ci.COHERENCE_SENSITIVITY_BOUNDS),
            "spearman_ceiling": ci.FROZEN_SPEARMAN_CEILING,
            "random_null_margin": ci.FROZEN_RANDOM_NULL_MARGIN,
            "minimum_control_directions": ci.MINIMUM_CONTROL_DIRECTIONS,
            "delta_definition": ci.DELTA_DEFINITION,
            "specificity_rules": list(ci.SPECIFICITY_RULES),
            "operating_alpha_rules": list(ci.OPERATING_ALPHA_RULES),
            "rule": "every one of these is required and never defaulted; resolve() "
            "checks each against EXP-R2-213 and refuses to run otherwise, so what "
            "counts as a result is fixed before the result exists",
            "synthetic_scope": "on --synthetic the coherence bound is the toy's own and "
            "is recorded as such, because A36-1's 0.25 nats/token is anchored to the "
            "real checkpoints' measured protein-mode context information and the toy has "
            "no such anchor. Every dimensionless criterion -- the sigma ladder, the "
            "Spearman ceiling, the margin, both control sizes -- is validated on both "
            "paths",
        },
        "injection_site_note": ci.INJECTION_SITE_NOTE,
        "provenance": {
            "runner": {
                "path": "scripts/transfer/36_concept_injection.py",
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "modules": {name: sha256_file(REPO_ROOT / name) for name in PROVENANCE_MODULES},
        },
        "limitations": LIMITATIONS,
    }

    if args.synthetic:
        payload.update(run_synthetic_check(args))
        destination = args.out / f"concept_injection__synthetic_check__L{args.layer:02d}.json"
    else:
        payload.update(run_campaign(args))
        names = [str(concept) for concept in args.concepts]
        destination = args.out / (
            "concept_injection__"
            + re.sub(r"[^A-Za-z0-9._-]+", "-", Path(args.checkpoint).resolve().name)
            + f"__{args.rendering}__{args.eval_split}"
            + f"__L{args.layer:02d}"
            + f"__{len(names)}concepts-{ci.digest_of(names)}.json"
        )

    write_json(destination, payload)
    print(f"wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
