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

**Reduction, if cost forces it.** The frozen rule is to drop concepts in ascending order
of their `eval` bearing-group count, a quantity known before any activation is captured,
and never to drop a control, a rung or a split. `--max-concepts` applies exactly that
rule and records which concepts were dropped with their counts.

**The instrument check runs before any real number is trusted.** `--synthetic` plants
concept directions in a toy decoder whose response is known by construction and runs the
whole pipeline -- the same hook, the same scoring loop, the same bootstrap, the same
controls, the same verdict -- on two cells: one where the direction genuinely steers both
modes, and one where it steers text only. The first must be recovered and the second must
not.

**Cost, measured rather than estimated.** The scored pass count is
`K * 9 * (1 + random + permuted)` in protein mode and `K * 9 * (1 + random)` in text,
for `K` concepts: the controls are run at every rung and not only at the admissible ones,
because A36-1 re-reads the verdict at three coherence bounds and each admits a different
rung set, so a control that existed only at the primary bound's rungs could not serve the
other two. One protein pass costs **0.073 s per cohort record** on an L20 at float32
(measured on `ProLLaMA_Stage_1`, layer 16, 133-residue mean, 0.80 ms per scored token,
and flat in batch size from 4 to 32, so the pass is compute-bound). At `K = 8` and a
300-record evaluation split that is about **22 GPU-h across both checkpoints**, which
fits the campaign's 24 GPU-h bound with no margin; the frozen reduction to four concepts
brings it to about 11. TF32 is deliberately not enabled: it would be roughly fourfold
faster and its 10-bit mantissa is not a safe substrate for a 0.01-nat cross-entropy
difference.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.transfer import concept_alignment as ca  # noqa: E402
from src.transfer import concept_injection as ci  # noqa: E402
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

CAMPAIGN_ONLY_FLAGS = (
    "checkpoint",
    "rendering",
    "cohort",
    "concepts",
    "pooling",
    "eval_split",
    "generation_readout",
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
        "refused with that reason rather than scored against an invented mapping. This "
        "derivation is an implementation of A36-6 and is not itself pre-registered",
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


def parse_concepts(argument: str) -> tuple[tuple[str, str], ...]:
    """Concepts as 'namespace:term' pairs, e.g. 'go_propagated:GO:0016787,ec:3.2.1'."""

    concepts: list[tuple[str, str]] = []
    for part in str(argument).split(","):
        item = part.strip()
        if not item:
            continue
        if ":" not in item:
            raise argparse.ArgumentTypeError(
                f"{item!r} is not 'namespace:term'; a concept is a term of one of the "
                "cohort's annotation namespaces and the namespace decides what it means"
            )
        namespace, term = item.split(":", 1)
        concepts.append((namespace.strip(), term.strip()))
    if len(concepts) < 2:
        raise argparse.ArgumentTypeError(
            "at least two concepts are required: with one there is no off-diagonal and "
            "A36-4 is vacuous"
        )
    if len(set(concepts)) != len(concepts):
        raise argparse.ArgumentTypeError("the concept set carries a duplicate")
    return tuple(concepts)


def concept_name(concept: tuple[str, str]) -> str:
    return f"{concept[0]}|{concept[1]}"


def split_concept(name: str) -> tuple[str, str]:
    namespace, term = name.split("|", 1)
    return namespace, term


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
            if name in bootstrap_for and float(alpha) in positive:
                deltas[name] = ci.delta_nll_shift(
                    baseline,
                    response,
                    flags,
                    seed=seed,
                    n_bootstrap=n_bootstrap,
                    label=f"{direction.concept}@{alpha}:{name}",
                )
            else:
                deltas[name] = {
                    "delta_nats_per_token": ci.delta_point(baseline, response, flags)
                }
            coherence[name] = ci.coherence_record(baseline, response, flags)
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


def select_concepts(
    args: argparse.Namespace, eval_records: Sequence[Any]
) -> tuple[list[str], dict[str, Any]]:
    """The concept set, reduced only by the frozen rule and only if asked.

    Ascending eval bearing-group count, which is known before any activation is
    captured. Nothing else is dropped: not a rung, not a control, not a split.
    """

    counts = {}
    for concept in args.concepts:
        namespace, term = concept
        flags = ca.concept_labels(eval_records, namespace, term)
        groups = ci.per_side_group_counts(
            [record["dup_group"] for record in eval_records], flags
        )
        counts[concept_name(concept)] = groups
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

    masking = {
        name: ci.assert_descriptions_masked(cohort.records, split_concept(name)[1])
        for name in names
    }
    bearing = {
        mode_split: {
            name: ca.concept_labels(records, *split_concept(name)) for name in names
        }
        for mode_split, records in (("fit", fit_records), ("eval", eval_records))
    }
    for name in names:
        ci.require_per_side_group_floor(
            [record["dup_group"] for record in eval_records],
            bearing["eval"][name],
            label=f"{name} in {args.eval_split}",
        )

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
        mode: ci.prepare_batches(handles[mode], eval_records, batch_size=args.batch_size)
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
            bearing=bearing["eval"],
            bootstrap_for=bootstrap_for,
            seed=args.seed,
            n_bootstrap=args.bootstrap_draws,
            device=device,
            dtype=dtype,
        )

    random_sweeps: dict[str, dict[str, list[dict[float, dict[str, Any]]]]] = {}
    concept_sweeps: dict[str, dict[str, dict[float, dict[str, Any]]]] = {
        mode: {} for mode in JOINT_MODES
    }
    permuted_sweeps: dict[str, list[tuple[str, dict[float, dict[str, Any]]]]] = {}
    for name in names:
        for mode in JOINT_MODES:
            concept_sweeps[mode][name] = sweep(mode, directions[name], [name])
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
        cells = {
            (injected, scored): matrix_cells[(injected, scored)]
            for injected in complete
            for scored in complete
        }
        specificity = ci.specificity_matrix(cells, complete, rule=args.specificity_rule)
        specificity["concepts_excluded_for_want_of_an_operating_alpha"] = excluded
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
            permuted_passes=[
                entry["direction"]
                for entry in per_concept[name]["permuted_label_control"]
                if entry["passes_a36_3"]
            ],
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

    generation = run_generation_readout(
        args,
        handles["protein"],
        directions,
        fit_records=fit_records,
        bearing_fit=bearing["fit"],
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
        "concept_reduction": reduction,
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
            for draw in ci.permuted_label_directions(
                representations,
                bearing[name],
                concept=name,
                layer=int(args.layer),
                n_draws=args.permuted_directions,
                seed=args.seed + 1,
            ):
                evaluation = evaluate_concept(
                    concept=name,
                    sweep=sweep("protein", draw, [name]),
                    controls=controls["protein"],
                    alphas=args.alphas,
                    bound=args.coherence_max_nll_inflation,
                    args=args,
                )
                permuted.append(permuted_entry(draw.concept, evaluation))
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
                permuted_passes=[
                    entry["direction"]
                    for entry in per_concept[name]["permuted_label_control"]
                    if entry["passes_a36_3"]
                ],
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
            cell["recovered_expected_outcome"] for cell in cells.values()
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
        names = [concept_name(concept) for concept in args.concepts]
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
