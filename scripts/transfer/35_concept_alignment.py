#!/usr/bin/env python3
"""Can one joint checkpoint's protein mode be carried, linearly, to where its text mode puts the description?

**What this stage is for.** D3.g (§8 item 4; ``R3.2`` in ``summary.md``) asks
whether a protein-generative model's internal state can be mapped into a
pre-declared *text* concept subspace of the same checkpoint, using genuine
sequence-description pairs. Its admission bar is pre-registered and it is
explicit about order: mean, affine or orthogonal maps, nearest-neighbour
retrieval, shuffled-pair, rank-matched and bridge-specific-concept baselines
are run **first**, and a non-linear adapter is admitted only if the linear
ladder has already reported. This stage is the whole of that linear ladder and
every baseline beside it. It builds no adapter, and
``src.transfer.concept_alignment.NONLINEAR_ADAPTER_NOTE`` records why.

**The checkpoint is reached by path and never by an arm name.** The only
lineage qualified in both modes is ProLLaMA's (EXP-R2-152), and a joint
checkpoint is deliberately absent from ``arms.py``: a checkpoint that has not
passed ``21_joint_mode_qualification.py`` must not be in the panel at all. The
rendering is named on the command line and everything about it is read from
``src.transfer.joint_modes``, whose ``prollama`` family declares the **token**
as its symbol unit -- LLaMA-2's SentencePiece merges residue runs at about
1.536 residues per token, and that merged form *is* the trained format. So no
magnitude here is commensurable with a residue-unit arm's (limitation L23,
Appendix B rule 26), and nothing in this stage is compared across families.

**A35-0's gate is one baseline, and the artefact names it.** The raw-description
arm is read first at every layer, and what decides whether the masked arm is read
at all is A35-1's margin over ``shuffled_pair`` -- the baseline EXP-R2-213 names,
in the singular. The rest of the raw arm's table is reported beside it and gates
nothing, because the pre-registration's branch table gives the two outcomes
opposite subjects: a raw arm that cannot clear ``shuffled_pair`` voids the ladder
as a specification defect and is a statement about *the instrument*, while a raw
arm that clears it and loses to a composition or 3-mer surrogate is the
**surface-statistics** branch and is a statement about *the method*, read as a
measured negative on the masked arm. ``frozen_branch`` writes which of those
branches an outcome took into the artefact.

**Two cells, and the second is the attainability check rule 2 asks for.**
``--mode protein`` is the estimand: the sequence in protein mode against the
curated description in text mode. ``--mode text`` is the positive control: the
record's *name* in text mode against the same description, which is two text
views of one protein and where a linear alignment must therefore be attainable.
A gate the control cannot pass is a specification defect and not a protein
result -- six instances of that are on record -- so the control is a cell of
this stage rather than a promise about it. Its attainability is also cheap in
the other direction: a name is often a substring of its own description, so the
control certifies that the pipeline can find an alignment that exists, and not
that it can find a hard one.

**The pre-adaptation reference is representational and is fenced in code.**
``Llama-2-7b-hf`` may be run through this identical pipeline, because
activations exist whether or not a behavioural estimand does. Its protein mode
is behaviourally unmeasurable on this lineage -- reversal cost **-0.0013**
nats/residue against the adapted stage's +0.1442 (EXP-R2-152, re-measured at
EXP-R2-174) -- so :func:`~src.transfer.concept_alignment.admission_verdict`
returns ``REFERENCE_ONLY`` for it whatever its numbers say, and the
concept-vector hand-off the causal stage consumes is withheld with its reason
rather than emitted.

**The description-leakage control is not a second invocation.** Both the raw
and the masked description are captured in one pass over the same records, so
the cell that prices "how much of this is the concept term written in the
text" cannot be skipped, dispatched later, or run on a different draw. The
verdict is read on the **masked** variant; the raw one is reported beside it.

**The homology control is the cohort's own two splits.** ``eval`` is
group-disjoint at the near-duplicate group -- L30 measured what a record-level
split costs on Swiss-Prot: 42.5% of a held-out block keeps a >=95%-identity
relative and an exact-string guard reaches 18.1 of those points -- and
``family_holdout`` is disjoint at the family. Every number is reported on both,
and the gate's "reproduce on an unseen family" is the second of them.

**What this stage does not decide.** It states no behavioural quantity. The
graded protein-model intervention and the preservation of unrelated concepts
under it are causal and belong to ``36_concept_injection.py``, which imports
``concept_vector`` from this stage's module so that the direction measured here
and the direction steered along there are one object.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.transfer import concept_alignment as ca  # noqa: E402
from src.transfer import joint_modes  # noqa: E402
from src.transfer import sequence_description as sd  # noqa: E402
from src.transfer.arms import AA20, REPO, require_input_path  # noqa: E402
from src.transfer.io import sha256_file, write_json  # noqa: E402
from src.transfer.replaceable import JOINT_MODES, JointReplaceable, joint_tokenisation  # noqa: E402
from src.transfer.scoring import analysis_layers  # noqa: E402
from src.transfer.statistics import (  # noqa: E402
    bootstrap_unit_floor,
    paired_group_bootstrap,
)


def _load_stage(filename: str) -> Any:
    """Import a stage whose module name starts with a digit.

    One of them, and for the reason ``32_crosscoder.py`` gives: stage 21 owns
    the joint-checkpoint loader, and Appendix B rule 12 does not stop applying
    because the declaration lives in a file whose name starts with a digit.
    """

    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(f"_transfer_stage_{filename[:2]}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE21 = _load_stage("21_joint_mode_qualification.py")

SCHEMA_VERSION = "r2_transfer_concept_alignment_v1"
DEFAULT_OUT = REPO / "results/transfer/concept_alignment"

PROVENANCE_MODULES = (
    "src/transfer/concept_alignment.py",
    "src/transfer/joint_modes.py",
    "src/transfer/sequence_description.py",
    "src/transfer/replaceable.py",
    "src/transfer/scoring.py",
    "src/transfer/statistics.py",
    "src/transfer/kmer_background.py",
    "src/transfer/io.py",
    "scripts/transfer/21_joint_mode_qualification.py",
)

#: Flags naming a real campaign. ``--synthetic-check`` requires every one of
#: them to be absent, and they are omitted from the synthetic artefact's
#: ``settings`` rather than echoed as null -- 32_crosscoder.py records what an
#: echoed null cost it.
CAMPAIGN_ONLY_FLAGS = (
    "checkpoint", "rendering", "mode", "cohort", "protein_context", "go_obo",
)

#: Decisions this stage refuses to default. Each is a number or a name that
#: moves the answer, and a stage that supplied one silently would be reporting
#: a choice as a measurement. ``32_crosscoder.py:784`` is the same mechanism.
PRE_REGISTERED_FLAGS = (
    "layers",
    "decision_layer",
    "decision_threshold",
    "pca_components",
    "pooling",
    "alignment_method",
    "gallery_size",
    "excess_ratio",
)

#: The two description variants, captured in one pass. The masked one decides.
DESCRIPTION_VARIANTS = ("raw", "masked")

#: The retrieval direction EXP-R2-213 fixes: the description is the query and the
#: mapped sequence representations are the gallery. The reverse direction is
#: reported beside it, costs one more call, and decides nothing.
PRIMARY_DIRECTION = "description_to_sequence"
REVERSE_DIRECTION = "sequence_to_description"

#: Baselines that are given the strongest rung of the ladder rather than the
#: rung under test, so that beating them is conservative.
SURROGATE_METHOD = "affine"


# ----------------------------------------------------------------- arguments


def parse_layers(argument: str) -> tuple[int, ...]:
    """``"8,16-18,31"`` into ``(8, 16, 17, 18, 31)``; ranges inclusive both ends."""

    layers: list[int] = []
    for piece in argument.replace(" ", "").split(","):
        if not piece:
            continue
        if "-" in piece.lstrip("-"):
            low, _, high = piece.partition("-")
            start, stop = int(low), int(high)
            if stop < start:
                raise argparse.ArgumentTypeError(
                    f"{piece!r} is a descending range; write it low-high"
                )
            layers.extend(range(start, stop + 1))
        else:
            layers.append(int(piece))
    if not layers:
        raise argparse.ArgumentTypeError("a layer set cannot be empty")
    if len(set(layers)) != len(layers):
        raise argparse.ArgumentTypeError(f"{argument!r} names a layer twice")
    return tuple(sorted(layers))


def parse_fractions(argument: str) -> tuple[float, ...]:
    values = tuple(float(piece) for piece in argument.replace(" ", "").split(",") if piece)
    if not values:
        raise argparse.ArgumentTypeError("a depth grid cannot be empty")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="directory of the joint checkpoint. Required and not an arm name: a "
        "checkpoint that has not passed 21_joint_mode_qualification.py must not be "
        "in the panel, so there is nothing for a default to point at",
    )
    parser.add_argument(
        "--rendering",
        default=None,
        choices=joint_modes.RENDERING_NAMES,
        help="which declared family's input format this checkpoint takes. The set "
        "is composed by src.transfer.joint_modes, the single place either mode's "
        "format is decided",
    )
    parser.add_argument(
        "--mode",
        default=None,
        choices=JOINT_MODES,
        help="which mode supplies the SOURCE side. 'protein' is the estimand -- the "
        "sequence against its description. 'text' is the positive control -- the "
        "record's name against the same description, where a linear alignment must "
        "be attainable, which is the check Appendix B rule 2 requires before a gate "
        "is applied to a protein cell",
    )
    parser.add_argument(
        "--cohort",
        type=Path,
        default=None,
        help="the sequence-description cohort directory (or its records.jsonl) "
        "written by 34_sequence_description_cohort.py. Never synthesised here",
    )
    parser.add_argument(
        "--go-obo",
        type=Path,
        default=None,
        help="the go-basic.obo the cohort was built from. Defaults to the path the "
        "cohort's own manifest recorded, which is how 36_concept_injection.py "
        "reaches it too; name it explicitly when the cohort has been staged onto a "
        "host where that absolute path does not resolve. It is what turns a "
        "concept's identifier into the surface forms the cohort masked, and without "
        "it the bridge-specific arm of A35-1b cannot be built",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", default="float32", choices=("bfloat16", "float16", "float32"),
        help="inference dtype. float32 by default because EXP-R2-213's standing "
        "instruction names it; the pooling and every downstream statistic are "
        "float32 or better whatever this says, and the observed dtype is read back "
        "from the loaded parameters rather than echoed",
    )
    parser.add_argument(
        "--reference-artefact", type=Path, default=None,
        help="a previous artefact of this stage over the SAME cohort, for A35-4. "
        "The pre-adaptation reference is a separate invocation of this pipeline; "
        "pass its artefact here and the run reports whether attribution has to be "
        "withdrawn while the measurement stands. Absent is reported as "
        "REFERENCE_NOT_SUPPLIED rather than passed over in silence",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--protein-context",
        default=None,
        help="the declared family's document context, if the run wants stage 2's "
        "instruction form. Absent means the bare block, which is what stage 1 was "
        "trained on; whichever was used reaches the artefact",
    )

    parser.add_argument(
        "--layers",
        type=parse_layers,
        default=None,
        help="the absolute layers representations are taken at. REQUIRED and never "
        "defaulted: the depth a result is read at is a pre-registered decision, and "
        "--depth-grid is the way to name it in relative terms",
    )
    parser.add_argument(
        "--depth-grid",
        type=parse_fractions,
        default=None,
        help="relative depths to convert into --layers through "
        "src.transfer.scoring.analysis_layers, which is the one place this "
        "programme rounds a relative depth. Needs --n-layers or a loaded model",
    )
    parser.add_argument(
        "--decision-layer",
        type=int,
        default=None,
        help="the ONE layer the admission verdict is read at, which must be in "
        "--layers. REQUIRED: reading a verdict off whichever of several layers "
        "passed is a multiplicity this stage will not hide",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=None,
        help="the DETECTION FLOOR: the minimum excess of the primary statistic over "
        "chance, in its own units. REQUIRED and never defaulted. It sits underneath "
        "A35-1's two conditions rather than replacing either -- a method whose "
        "excess over chance is inside the floor has nothing for a comparison to be "
        "about. Pass 0.0 to run exactly EXP-R2-213's conditions and nothing more",
    )
    parser.add_argument(
        "--gallery-size",
        type=int,
        default=None,
        help="the COMMON gallery every target is ranked in, so the chance level of "
        "the primary statistic is the single number 1/gallery_size. REQUIRED and "
        "never defaulted: a per-query gallery makes the chance level an average "
        "over fields of different sizes and the primary statistic a different "
        "statistic per query",
    )
    parser.add_argument(
        "--excess-ratio",
        type=float,
        default=None,
        help="A35-1 condition (ii): the factor by which the method's excess over "
        "chance must exceed each baseline's. REQUIRED and never defaulted; "
        "EXP-R2-213 freezes it at 2.0. It is an effect-size bar and is deliberately "
        "separate from the interval condition, because significance alone is a "
        "detection criterion and does not license a comparative claim",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=None,
        help="width of the shared subspace both modes are compared in, fitted on the "
        "fit split alone. REQUIRED: it IS the linear ladder's capacity, which §8 "
        "item 4 asks to be reported beside every result. At d_model 4096 against a "
        "few hundred fit records an unreduced affine map has more free parameters "
        "than data",
    )
    parser.add_argument(
        "--pooling",
        default=None,
        choices=ca.POOLINGS,
        help="how a record's content positions become one vector. REQUIRED: it is "
        "the aggregation the estimand is defined by",
    )
    parser.add_argument(
        "--alignment-method",
        default=None,
        choices=ca.ALIGNMENT_METHODS,
        help="the rung of the ladder the verdict is read on. REQUIRED; every rung is "
        "fitted and reported whichever one this names",
    )

    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--null-draws", type=int, default=200)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument(
        "--rank-match-block",
        type=int,
        default=8,
        help="block width of the rank-matched null, in records of the length-rank "
        "order. Narrower is a stricter null; the achieved match is measured and "
        "reported beside every use of it",
    )
    parser.add_argument(
        "--concept-namespaces",
        default="ec,go_propagated,pfam",
        help="the cohort's annotation namespaces a concept may be declared from",
    )
    parser.add_argument(
        "--min-concept-positives",
        type=int,
        default=8,
        help="records a concept needs on EACH side in EVERY split before it is "
        "declared. Below the bootstrap unit floor a concept cannot carry an interval",
    )
    parser.add_argument(
        "--synthetic-check",
        action="store_true",
        help="run the whole analysis on data whose answer is known -- one cell with "
        "a linear correspondence planted and one with none -- and write the recovery "
        "certificate. No checkpoint and no cohort are loaded",
    )
    parser.add_argument("--synthetic-records", type=int, default=180)
    parser.add_argument("--synthetic-d-model", type=int, default=64)
    parser.add_argument("--synthetic-factors", type=int, default=6)
    parser.add_argument("--synthetic-noise", type=float, default=0.6)
    return parser


def resolve(args: argparse.Namespace) -> None:
    """Refuse an incoherent request before a cohort is opened or a model is loaded."""

    if args.synthetic_check:
        present = [
            flag for flag in CAMPAIGN_ONLY_FLAGS if getattr(args, flag) is not None
        ]
        if present:
            raise ValueError(
                ", ".join(f"--{flag.replace('_', '-')}" for flag in present)
                + " name a real campaign and are meaningless beside "
                "--synthetic-check, which runs the same analysis on data whose "
                "answer is known"
            )
    else:
        missing_inputs = [
            flag
            for flag in ("checkpoint", "rendering", "mode", "cohort")
            if getattr(args, flag) is None
        ]
        if missing_inputs:
            raise ValueError(
                "this stage needs "
                + ", ".join(f"--{flag.replace('_', '-')}" for flag in missing_inputs)
                + ". The checkpoint is named by path because a joint checkpoint is "
                "not a panel arm, and the cohort is never synthesised in the real path"
            )

    missing = [flag for flag in PRE_REGISTERED_FLAGS if getattr(args, flag) is None]
    if "layers" in missing and args.depth_grid is not None:
        missing.remove("layers")
    if missing:
        raise ValueError(
            "this stage never defaults "
            + ", ".join(f"--{flag.replace('_', '-')}" for flag in missing)
            + ". Each is a pre-registered decision: the layer set and the layer the "
            "verdict is read at, the margin the alignment must clear, the subspace "
            "width that is the ladder's capacity, the pooling that defines the "
            "estimand's aggregation, and the rung the verdict is read on"
        )
    if args.decision_threshold < 0.0:
        raise ValueError("a detection floor below zero admits a method below chance")
    if args.excess_ratio < 1.0:
        raise ValueError(
            "an excess ratio below one lets a baseline out-perform the method and "
            "still be recorded as beaten"
        )
    if args.gallery_size < 2:
        raise ValueError("a gallery of one has nothing to rank the target against")
    if args.alignment_method == "mean":
        raise ValueError(
            "the mean rung cannot be the decisive one under EXP-R2-213. A35-1 "
            "requires the shuffled-fit baseline, and a mean shift between two cloud "
            "centres never reads the pairing -- permuting the fit pairing leaves the "
            "fitted map identical, so the null is the truth under another name and "
            "the criterion is vacuous rather than passed. The mean rung is still "
            "fitted and reported as the rung below whichever one decides"
        )
    if args.pca_components < 2:
        raise ValueError("a subspace narrower than two dimensions carries no geometry")
    if args.null_draws < 20:
        raise ValueError(
            "a null reported as a distribution needs draws to be a distribution; "
            "below twenty its 97.5th percentile is one of two order statistics"
        )
    if args.batch_size < 1 or args.max_tokens < 2:
        raise ValueError("--batch-size and --max-tokens must leave a scorable record")


# -------------------------------------------------------------------- inputs


def open_modes(
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any], dict[str, JointReplaceable]]:
    """One load of the weights, one handle per mode over them.

    The tokenizer is read first and the rendering is resolved against it, which
    is what refuses a checkpoint/family pair before a multi-gigabyte load.
    """

    declaration = joint_modes.rendering(args.rendering)
    resolved, tokenizer = STAGE21.load_tokenizer(Path(args.checkpoint))
    model, facts = STAGE21.load_model(
        resolved, tokenizer, device=args.device, dtype=args.dtype
    )
    handles: dict[str, JointReplaceable] = {}
    for mode in ("protein", "text"):
        handles[mode] = JointReplaceable(
            model=model,
            tokenizer=tokenizer,
            checkpoint=resolved,
            declaration=declaration,
            mode=mode,
            tokenisation=joint_tokenisation(tokenizer, declaration, mode),
            max_tokens=args.max_tokens,
            protein_context=args.protein_context,
        )
    return resolved, facts, handles


def resolve_go_obo(cohort: ca.Cohort, args: argparse.Namespace) -> Path:
    """The ``go-basic.obo`` this cohort was built from, or a refusal naming it.

    The cohort's manifest records the ontology the cohort stage actually used, and
    ``36_concept_injection.py`` reaches it the same way, so the default is the
    recorded path rather than this module's own idea of where an ontology lives.
    The recorded path is absolute on the host the cohort was built on, which is not
    always the host this stage runs on -- so ``--go-obo`` names the staged copy,
    and a path that does not resolve is a refusal rather than a bridge arm silently
    built from nothing.
    """

    if args.go_obo is not None:
        return require_input_path(Path(args.go_obo).resolve(), "--go-obo")
    recorded = cohort.manifest.get("settings", {}).get("go_obo")
    if recorded is None:
        raise ValueError(
            f"{cohort.path.parent / 'cohort.json'} records no settings.go_obo, so "
            "the ontology this cohort's surface forms were derived from is unknown. "
            "Pass --go-obo, or rebuild the cohort with a stage that records it"
        )
    path = Path(str(recorded))
    if not path.exists():
        raise FileNotFoundError(
            f"the cohort records its GO ontology at {path}, which does not exist "
            "here; it is an absolute path on the host the cohort was built on. Pass "
            "--go-obo to name the staged copy. The ontology is what turns a "
            "concept's identifier into the surface forms the cohort masked out of "
            "the descriptions, and A35-1b's bridge arm is not defined without it"
        )
    return path


def declared_surface_forms(obo: Path) -> dict[tuple[str, str], tuple[str, ...]]:
    """Every declared concept's surface forms, keyed as this stage keys a concept.

    One object, read from the module that wrote ``masked_terms``:
    ``sequence_description.concept_surface_forms`` is what the cohort stage derives
    the shared mask vocabulary from, and it is what decides here whether a concept
    is named in the description. Keying the question on the concept's *identifier*
    instead -- a GO id, an EC number, a Pfam accession -- asks it of a vocabulary
    that cannot answer: the masked terms are English names, and on the production
    cohort the identifier-keyed test returned zero bridge concepts of 1,174.

    A GO concept is reached from either annotation column it can be declared from,
    because a cohort concept key is ``(namespace, term)`` and ``go_propagated`` and
    ``go`` are two columns carrying the same identifiers.
    """

    ontology = sd.load_go_ontology(obo)
    forms: dict[tuple[str, str], tuple[str, ...]] = {}
    for spec in sd.CONCEPTS:
        surface = sd.concept_surface_forms(spec, ontology=ontology)
        namespaces = ("go", "go_propagated") if spec.kind == "go" else ("ec",)
        for namespace in namespaces:
            forms[(namespace, spec.identifier)] = surface
    return forms


def source_strings(cohort: ca.Cohort, mode: str) -> list[str]:
    """The strings the source side is represented from, for one mode."""

    field = "sequence" if mode == "protein" else "name"
    values = [str(record[field]) for record in cohort.records]
    empty = [
        record["accession"]
        for record, value in zip(cohort.records, values)
        if not value.strip()
    ]
    if empty:
        raise ValueError(
            f"{len(empty)} records carry an empty {field!r} (first: {empty[:3]}); the "
            "source side of a pair cannot be an empty string"
        )
    return values


def description_strings(cohort: ca.Cohort, variant: str) -> list[str]:
    field = "description_raw" if variant == "raw" else "description_masked"
    values = [str(record[field]) for record in cohort.records]
    empty = [
        record["accession"]
        for record, value in zip(cohort.records, values)
        if not value.strip()
    ]
    if empty:
        raise ValueError(
            f"{len(empty)} records carry an empty {field!r} (first: {empty[:3]})"
        )
    return values


# ------------------------------------------------------------------ analysis


@dataclass(frozen=True)
class CellInputs:
    """Everything one (layer, description variant) cell is computed from."""

    source: np.ndarray
    target: np.ndarray
    sequences: tuple[str, ...]
    lengths: np.ndarray
    dup_group: np.ndarray
    family_group: np.ndarray
    split: np.ndarray
    accession: tuple[str, ...]
    concepts: tuple[tuple[str, str], ...]
    bridge: tuple[tuple[str, str], ...]
    labels: Mapping[tuple[str, str], np.ndarray]

    def index(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.split == split)


def _retrieval(
    query: np.ndarray, gallery_rows: np.ndarray, dup: np.ndarray, field: np.ndarray
) -> tuple[dict[str, Any], np.ndarray]:
    """The reported block and the per-record top-1 hits the bootstrap resamples."""

    ranks, sizes = ca.retrieval_ranks(query, gallery_rows, dup, gallery=field)
    return ca.metrics_from_ranks(ranks, sizes), ca.top1_indicators(ranks)


def _null_top1(
    query: np.ndarray, gallery_rows: np.ndarray, dup: np.ndarray, field: np.ndarray
) -> np.ndarray:
    """One null draw's per-record top-1 hits.

    Only the indicators, because a null draw permutes the query side and leaves
    the gallery: its field, its chance level and its cut-offs are the true run's by
    construction, so recomputing them per draw is the same number a few hundred
    times over.
    """

    ranks, _ = ca.retrieval_ranks(query, gallery_rows, dup, gallery=field)
    return ca.top1_indicators(ranks)


def _paired_interval(
    left: np.ndarray, right: np.ndarray, groups: np.ndarray, *, args: argparse.Namespace
) -> dict[str, Any]:
    """The paired group bootstrap, or the record of why there could not be one.

    The resampling unit is the near-duplicate group, as EXP-R2-213 fixes it, and
    ``src.transfer.statistics.paired_group_bootstrap`` is the only resampler in
    the loop -- this stage adds none of its own. The floor is consulted before the
    call rather than caught after it: below eight units the refusal is a finding
    about the split and belongs in the artefact, which is what
    ``bootstrap_unit_floor``'s own docstring asks a caller to do.
    """

    labels = np.asarray([str(value) for value in groups])
    floor = bootstrap_unit_floor(int(np.unique(labels).size))
    if floor["degenerate"]:
        return {"publishable": False, "floor": floor, "bootstrap": None}
    return {
        "publishable": True,
        "floor": floor,
        "bootstrap": paired_group_bootstrap(
            np.zeros(labels.size),
            np.asarray(left, dtype=np.float64),
            np.asarray(right, dtype=np.float64),
            labels,
            ca.mean_metric,
            seed=args.seed,
            n_bootstrap=args.bootstrap_draws,
        ),
    }


def _concept_axis(
    vectors: Mapping[tuple[str, str], ca.ConceptVector],
    source_fit: np.ndarray,
    source_block: np.ndarray,
    labels: Mapping[tuple[str, str], np.ndarray],
    block_index: np.ndarray,
) -> dict[str, Any]:
    """One AUC over every declared concept, and the per-concept AUCs beside it.

    A reported diagnostic and never a decisive quantity: it is a *classification*
    accuracy, and A35-1's conditions are stated on the primary retrieval
    statistic. Comparing an AUC excess against a retrieval excess would put two
    incommensurable statistics on one criterion, which is the reading this stage
    is required not to make.

    Each concept's projection is standardised on the fit split before the concepts
    are pooled, so one number describes the declared concept set rather than
    whichever concept happens to have the widest projections.
    """

    pooled_scores: list[np.ndarray] = []
    pooled_labels: list[np.ndarray] = []
    per_concept: list[dict[str, Any]] = []
    for key, vector in vectors.items():
        fit_scores = source_fit @ vector.direction
        centre = float(fit_scores.mean())
        spread = float(fit_scores.std(ddof=1))
        if spread <= 0.0:
            raise ValueError(
                f"concept {key} has zero spread on the fit split under this source "
                "representation, so its scores cannot be standardised"
            )
        scores = (source_block @ vector.direction - centre) / spread
        truth = labels[key][block_index]
        per_concept.append(
            {
                "namespace": key[0],
                "term": key[1],
                "auc": ca.concept_auc(scores, truth),
                "n_positive": int(truth.sum()),
                "n_negative": int((~truth).sum()),
            }
        )
        pooled_scores.append(scores)
        pooled_labels.append(truth)
    scores = np.concatenate(pooled_scores)
    truth = np.concatenate(pooled_labels)
    return {
        "pooled_auc": ca.concept_auc(scores, truth),
        "n_concepts": len(per_concept),
        "mean_per_concept_auc": float(np.mean([row["auc"] for row in per_concept])),
        "min_per_concept_auc": float(np.min([row["auc"] for row in per_concept])),
        "per_concept": per_concept,
        "is_decisive": False,
    }


def evaluate_cell(
    inputs: CellInputs,
    args: argparse.Namespace,
    *,
    with_intervals: bool,
    bridge_decisive: bool = False,
    bridge_inapplicable_reason: str | None = None,
) -> dict[str, Any]:
    """The ladder, every pre-registered baseline, both evaluation splits, one layer.

    ``bridge_decisive`` carries A35-1b's decision (EXP-R2-213 amendment 1) into the
    one flag :func:`src.transfer.concept_alignment.baseline_row` keeps it in, so
    that the verdict is still computed in exactly one place. It is false on the
    raw arm because that arm is where A35-1b's attainability is *measured*, and it
    is the measured value on the arm the verdict is read on.
    """

    fit = inputs.index("fit")
    subspace = ca.fit_subspace(
        np.vstack([inputs.source[fit], inputs.target[fit]]), args.pca_components
    )
    projected_source = ca.apply_subspace(subspace, inputs.source)
    projected_target = ca.apply_subspace(subspace, inputs.target)

    maps = {
        method: ca.fit_alignment(
            projected_source[fit],
            projected_target[fit],
            method,
            groups=inputs.dup_group[fit],
            seed=args.seed,
        )
        for method in ca.ALIGNMENT_METHODS
    }
    # A35-3: the ladder is an order. Every rung at or below the deciding one has
    # to be fitted and reported before the deciding one's numbers are emitted.
    ca.assert_ladder_reported(list(maps), args.alignment_method)
    primary = maps[args.alignment_method]

    # Concept directions are declared on the TEXT side, on the fit split, and
    # every read below uses these same directions. A baseline scored against its
    # own fitted direction would be a different question wearing the same name.
    vectors = {
        key: ca.concept_vector(projected_target[fit], inputs.labels[key][fit])
        for key in inputs.concepts
    }
    # The bridge coordinates are the span of the concepts declared on BOTH sides,
    # and the restriction is the orthogonal projector onto that span rather than
    # the concept coordinates themselves: the concept directions are not
    # orthonormal, so a cosine in their coordinates would be read under a metric
    # nothing declared. The projector keeps the ambient inner product and keeps
    # every block the same width, so the restricted retrieval is the SAME task on
    # the SAME gallery -- which is what makes it commensurable with the ladder.
    bridge_projector = None
    if inputs.bridge:
        basis = np.stack([vectors[key].direction for key in inputs.bridge], axis=1)
        bridge_projector = basis @ np.linalg.pinv(basis)

    # The surface surrogates are the identical retrieval task with the
    # protein-side representation swapped for surface features, and they are given
    # the STRONGEST rung rather than the rung under test, so beating them is
    # conservative.
    surrogates = {
        "composition": ca.composition_features(inputs.sequences),
        "kmer": ca.kmer_features(inputs.sequences, k=3),
    }
    surrogate_maps: dict[str, tuple[ca.Subspace, ca.AlignmentMap]] = {}
    for name, features in surrogates.items():
        width = min(args.pca_components, features.shape[1], fit.size)
        space = ca.fit_subspace(features[fit], width)
        surrogate_maps[name] = (
            space,
            ca.fit_alignment(
                ca.apply_subspace(space, features[fit]),
                projected_target[fit],
                SURROGATE_METHOD,
                groups=inputs.dup_group[fit],
                seed=args.seed,
            ),
        )

    # The description-only arm's length prior: a ridge from the description
    # representation to log length, fitted on the fit split with its penalty
    # chosen inside it. This is what "text-side self-information plus gallery
    # structure" comes to in the retrieval task itself -- the description says
    # what kind of protein it is, and sizes do the rest.
    log_length = np.log(inputs.lengths).reshape(-1, 1)
    length_prior = ca.fit_alignment(
        projected_target[fit], log_length[fit], "affine",
        groups=inputs.dup_group[fit], seed=args.seed,
    )

    fit_permutations = ca.shuffled_pairing(
        fit.size, draws=min(args.null_draws, 64), seed=args.seed + 2
    )
    refit_maps = [
        ca.fit_alignment(
            projected_source[fit][order],
            projected_target[fit],
            args.alignment_method,
            groups=inputs.dup_group[fit][order],
            seed=args.seed,
        )
        for order in fit_permutations
    ]

    cell: dict[str, Any] = {
        "subspace": subspace.record(),
        "maps": {method: fitted.record() for method, fitted in maps.items()},
        "ladder_order": list(ca.ALIGNMENT_METHODS),
        "primary_method": args.alignment_method,
        "primary_statistic": ca.PRIMARY_STATISTIC,
        "primary_direction": PRIMARY_DIRECTION,
        "surrogate_maps": {
            name: {"subspace": space.record(), "map": fitted.record()}
            for name, (space, fitted) in surrogate_maps.items()
        },
        "length_prior_map": length_prior.record(),
        "concept_vectors": {
            f"{namespace}:{term}": vectors[(namespace, term)].record()
            for namespace, term in inputs.concepts
        },
        "bridge_concepts": [f"{ns}:{term}" for ns, term in inputs.bridge],
        "splits": {},
    }

    for split in ("eval", "family_holdout"):
        index = inputs.index(split)
        dup = inputs.dup_group[index]
        field = ca.common_gallery(dup, gallery_size=args.gallery_size, seed=args.seed)
        query = projected_target[index]

        ladder: dict[str, Any] = {}
        top1: dict[str, np.ndarray] = {}
        for method, fitted in maps.items():
            gallery_rows = ca.apply_alignment(fitted, projected_source[index])
            forward, hits = _retrieval(query, gallery_rows, dup, field)
            reverse, _ = _retrieval(gallery_rows, query, dup, field)
            ladder[method] = {
                PRIMARY_DIRECTION: forward,
                REVERSE_DIRECTION: reverse,
            }
            top1[method] = hits
        gallery_primary = ca.apply_alignment(primary, projected_source[index])
        observed = ladder[args.alignment_method][PRIMARY_DIRECTION]
        chance = float(observed["top1_chance"])
        primary_hits = top1[args.alignment_method]

        # -- the ladder's foot: no fit at all, in the ambient activation space
        # both modes share. Reported, and not one of the frozen decisive set.
        nn_metrics, nn_hits = _retrieval(
            inputs.target[index], inputs.source[index], dup, field
        )

        # -- the three nulls. Each returns a per-record mean indicator, so the
        # paired interval condition (i) is computable against it, and its own
        # draw distribution, so the accuracy condition (ii) reads a mean rather
        # than a single draw.
        nulls: dict[str, dict[str, Any]] = {}
        null_hits: dict[str, np.ndarray] = {}
        shuffled = ca.shuffled_pairing(index.size, draws=args.null_draws, seed=args.seed)
        matched = ca.rank_matched_pairing(
            inputs.lengths[index],
            draws=args.null_draws,
            seed=args.seed + 1,
            block=args.rank_match_block,
        )
        for name, orders in (("shuffled_pair", shuffled), ("rank_matched", matched)):
            draws = [_null_top1(query[order], gallery_primary, dup, field) for order in orders]
            null_hits[name] = np.mean(draws, axis=0)
            nulls[name] = ca.null_distribution([float(row.mean()) for row in draws])
        refit_draws = [
            _null_top1(query, ca.apply_alignment(refit, projected_source[index]), dup, field)
            for refit in refit_maps
        ]
        null_hits["shuffled_fit"] = np.mean(refit_draws, axis=0)
        nulls["shuffled_fit"] = ca.null_distribution([float(row.mean()) for row in refit_draws])

        # -- surrogates, the bridge and the description-only arm: the same task,
        # the same gallery, the same statistic.
        arms: dict[str, dict[str, Any]] = {}
        arm_hits: dict[str, np.ndarray] = {}
        for name, (space, fitted) in surrogate_maps.items():
            gallery_rows = ca.apply_alignment(
                fitted, ca.apply_subspace(space, surrogates[name][index])
            )
            metrics, hits = _retrieval(query, gallery_rows, dup, field)
            arms[name] = metrics
            arm_hits[name] = hits
        if bridge_projector is not None:
            metrics, hits = _retrieval(
                query @ bridge_projector, gallery_primary @ bridge_projector, dup, field
            )
            arms["bridge_specific"] = {**metrics, "n_axes": len(inputs.bridge)}
            arm_hits["bridge_specific"] = hits

        description_only = _description_only_arm(
            query=query,
            gallery_primary=gallery_primary,
            gallery_fit=ca.apply_alignment(primary, projected_source[fit]),
            length_prior=length_prior,
            log_length=log_length,
            index=index,
            dup=dup,
            field=field,
        )
        arms["description_only"] = description_only["reported"]
        arm_hits["description_only"] = description_only["hits"]

        # -- the concept axis, reported and non-decisive throughout.
        mapped_fit_primary = ca.apply_alignment(primary, projected_source[fit])
        concept = {
            "aligned": _concept_axis(
                vectors, mapped_fit_primary, gallery_primary, inputs.labels, index
            ),
            "unmapped_source": _concept_axis(
                vectors, projected_source[fit], projected_source[index],
                inputs.labels, index,
            ),
            "description_only_ceiling": _concept_axis(
                vectors, projected_target[fit], query, inputs.labels, index
            ),
            "note": (
                "a classification accuracy, reported beside the ladder and decisive "
                "for nothing. A35-1's conditions are stated on the primary retrieval "
                "statistic, and an AUC excess is not commensurable with a retrieval "
                "excess"
            ),
        }
        concept["preservation"] = {
            "mapped_minus_unmapped_pooled_auc": float(
                concept["aligned"]["pooled_auc"]
                - concept["unmapped_source"]["pooled_auc"]
            ),
            "per_concept_minimum_delta": float(
                min(
                    mapped["auc"] - unmapped["auc"]
                    for mapped, unmapped in zip(
                        concept["aligned"]["per_concept"],
                        concept["unmapped_source"]["per_concept"],
                    )
                )
            ),
            "note": (
                "the representational half of the gate's 'preserve unrelated "
                "concepts'. The causal half -- unrelated concepts under a graded "
                "intervention -- belongs to 36_concept_injection.py"
            ),
        }

        rows = _primary_rows(
            args=args,
            split=split,
            observed=observed,
            chance=chance,
            primary_hits=primary_hits,
            dup=dup,
            nn_metrics=nn_metrics,
            nn_hits=nn_hits,
            nulls=nulls,
            null_hits=null_hits,
            arms=arms,
            arm_hits=arm_hits,
            bridge_available=bridge_projector is not None,
            bridge_decisive=bridge_decisive,
            bridge_inapplicable_reason=bridge_inapplicable_reason,
            source_is_sequence=args.mode != "text",
            with_intervals=with_intervals,
        )

        cell["splits"][split] = {
            "n_records": int(index.size),
            "n_dup_groups": int(np.unique(dup).size),
            "n_family_groups": int(np.unique(inputs.family_group[index]).size),
            "gallery": {
                "size": int(args.gallery_size),
                "chance_top1": chance,
                "seed": int(args.seed),
                "note": (
                    "one common field for every target, drawn once from the "
                    "near-duplicate grouping and reused by every arm, every null "
                    "draw and any reference run over this cohort, so the paired "
                    "comparisons are paired"
                ),
            },
            "ladder": ladder,
            "nearest_neighbour_ambient": nn_metrics,
            "nulls": nulls,
            "null_quality": {
                "rank_matched": ca.pairing_match_quality(inputs.lengths[index], matched),
                "shuffled_pair": ca.pairing_match_quality(inputs.lengths[index], shuffled),
            },
            "arms": arms,
            "description_only_components": description_only["components"],
            "concept": concept,
            "baseline_rows": rows,
            "primary_per_record": {
                "accessions": [inputs.accession[position] for position in index],
                "dup_groups": [str(value) for value in dup],
                "top1": [int(value) for value in primary_hits],
                "note": (
                    "the per-record top-1 hits of the deciding rung, kept so that a "
                    "separate reference run over this cohort can be compared under "
                    "A35-1's paired interval condition and not only on its point "
                    "estimate. The near-duplicate group travels with it because that "
                    "is the resampling unit the pre-registration fixes"
                ),
            },
        }
    return cell


def _description_only_arm(
    *,
    query: np.ndarray,
    gallery_primary: np.ndarray,
    gallery_fit: np.ndarray,
    length_prior: ca.AlignmentMap,
    log_length: np.ndarray,
    index: np.ndarray,
    dup: np.ndarray,
    field: np.ndarray,
) -> dict[str, Any]:
    """Retrieval attainable without any sequence-derived pairing signal.

    Two constructions, both the identical retrieval task on the identical gallery
    -- which is what makes this baseline commensurable with the ladder under
    A35-1(ii); a concept-classification accuracy would not be:

    ``gallery_typicality``
        a query-independent ranking. Each gallery item is scored by its cosine to
        the fit split's gallery centroid, so no query information enters at all
        and what is measured is the top-1 accuracy attainable from the gallery's
        own structure. It exceeds chance whenever the cohort's typical items are
        unevenly distributed across the drawn fields, and on a small common
        gallery that is not a negligible amount.
    ``length_prior``
        the description predicts the target's log length through a ridge fitted on
        the fit split, and the gallery is ranked by agreement with the observed
        length. This is the sharpest form of the objection: a description that
        says what kind of protein it is has said something about how long it is,
        and no model-internal protein representation is involved.

    The stronger of the two is what the criterion is read against, because a
    baseline is only worth reporting at its best.
    """

    centroid = gallery_fit.mean(axis=0)
    norms = np.linalg.norm(gallery_primary, axis=1)
    centroid_norm = float(np.linalg.norm(centroid))
    if not np.all(norms > 0.0) or centroid_norm <= 0.0:
        raise ValueError(
            "a mapped gallery row, or the fit split's gallery centroid, has zero "
            "norm, so it has no direction and no typicality; a zero here would "
            "read as a legitimate score"
        )
    typicality = (gallery_primary / norms[:, None]) @ (centroid / centroid_norm)
    typicality_scores = np.tile(typicality, (index.size, 1))
    typicality_ranks, sizes = ca.ranks_from_scores(typicality_scores, dup, gallery=field)

    predicted = ca.apply_alignment(length_prior, query).reshape(-1)
    observed_length = log_length[index].reshape(-1)
    length_scores = -np.abs(predicted[:, None] - observed_length[None, :])
    length_ranks, _ = ca.ranks_from_scores(length_scores, dup, gallery=field)

    components = {
        "gallery_typicality": ca.metrics_from_ranks(typicality_ranks, sizes),
        "length_prior": ca.metrics_from_ranks(length_ranks, sizes),
        "note": (
            "the reported description_only arm is whichever of these two reaches "
            "the higher top-1 accuracy; a baseline is read at its best"
        ),
    }
    stronger = max(
        ("gallery_typicality", typicality_ranks), ("length_prior", length_ranks),
        key=lambda item: float(np.mean(item[1] <= 1.0)),
    )
    return {
        "reported": {**ca.metrics_from_ranks(stronger[1], sizes), "construction": stronger[0]},
        "hits": ca.top1_indicators(stronger[1]),
        "components": components,
    }


def _primary_rows(
    *,
    args: argparse.Namespace,
    split: str,
    observed: Mapping[str, Any],
    chance: float,
    primary_hits: np.ndarray,
    dup: np.ndarray,
    nn_metrics: Mapping[str, Any],
    nn_hits: np.ndarray,
    nulls: Mapping[str, Mapping[str, Any]],
    null_hits: Mapping[str, np.ndarray],
    arms: Mapping[str, Mapping[str, Any]],
    arm_hits: Mapping[str, np.ndarray],
    bridge_available: bool,
    bridge_decisive: bool,
    bridge_inapplicable_reason: str | None,
    source_is_sequence: bool,
    with_intervals: bool,
) -> list[dict[str, Any]]:
    """Every baseline as one row on the primary statistic, under both conditions."""

    accuracy = float(observed["top1_accuracy"])
    ratio = float(args.excess_ratio)
    rows: list[dict[str, Any]] = []

    def interval(other: np.ndarray) -> dict[str, Any] | None:
        if not with_intervals:
            return None
        return _paired_interval(primary_hits, other, dup, args=args)

    def row(name: str, value: float | None, hits: np.ndarray | None, **extra) -> None:
        rows.append(
            ca.baseline_row(
                name, "primary_top1", accuracy, value,
                chance=chance, excess_ratio=ratio,
                interval=None if hits is None else interval(hits),
                **extra,
            )
        )

    for name in ("shuffled_pair", "rank_matched", "shuffled_fit"):
        row(
            name, float(nulls[name]["mean"]), null_hits[name],
            note="the null's mean accuracy over its declared draws; its 97.5th "
            "percentile is reported in the nulls block and is the stricter reading",
        )
    for name in ("composition", "kmer"):
        # A surface surrogate is the same task with the SOURCE-side representation
        # swapped for surface features of the source. In the text-control cell the
        # source is a curated name, which has no residue composition, so a
        # sequence-derived surrogate there is not a surrogate for that cell's
        # source -- it is a different predictor with strictly more information, and
        # requiring the control to beat it would make the attainability check a
        # test of something the control never claimed. Reported either way.
        row(
            name, float(arms[name]["top1_accuracy"]),
            arm_hits[name] if source_is_sequence else None,
            applicable=source_is_sequence,
            inapplicable_reason=(
                None
                if source_is_sequence
                else "this cell's source side is the curated name, not the sequence, "
                "so a residue-composition surrogate is not the same arm with its "
                "representation swapped; it is a predictor with access to the "
                "sequence the source does not carry. Reported, and it decides "
                "nothing in a text-control cell"
            ),
            note="the identical retrieval task with the protein-side representation "
            "swapped for surface features and mapped with the strongest rung",
        )
    row(
        "description_only", float(arms["description_only"]["top1_accuracy"]),
        arm_hits["description_only"],
        note="retrieval attainable with no sequence-derived pairing signal: the "
        "stronger of a query-independent gallery ranking and a text-predicted "
        "length prior. It is the same statistic on the same gallery, which is what "
        "makes A35-1(ii) well posed for it",
    )
    if bridge_available:
        # A35-1b (amendment 1): reported always, decisive only where the raw arm has
        # demonstrated the margin against it is reachable at these settings. The
        # measured value is carried either way -- an inapplicable baseline that
        # dropped its number would be indistinguishable from one that was not run.
        row(
            ca.A35_1B_BASELINE, float(arms[ca.A35_1B_BASELINE]["top1_accuracy"]),
            arm_hits[ca.A35_1B_BASELINE],
            decisive=bridge_decisive,
            applicable=bridge_inapplicable_reason is None,
            inapplicable_reason=bridge_inapplicable_reason,
            note="the same retrieval restricted to the span of the concepts declared "
            "on both sides. " + ca.A35_1B_NOTE,
        )
    else:
        rows.append(
            ca.baseline_row(
                ca.A35_1B_BASELINE, "primary_top1", accuracy, None,
                chance=chance, excess_ratio=ratio, decisive=bridge_decisive,
                applicable=False,
                inapplicable_reason="no declared concept is named in this cohort's "
                "masked_terms, so no concept is defined on both sides and the bridge "
                "coordinates are empty",
            )
        )

    # Reported and non-decisive: widening a frozen criterion is as much a change
    # to it as softening one. EXP-R2-213 amendment 1 names six decisive baselines
    # and neither of these two is among them.
    rows.append(
        ca.baseline_row(
            "chance", "primary_top1", accuracy, chance, chance=chance,
            excess_ratio=ratio, decisive=False,
            note="the analytic chance level of this common gallery, 1/gallery_size. "
            "It is built into the primary statistic as its offset and is carried as "
            "a row so the arithmetic is visible",
        )
    )
    rows.append(
        ca.baseline_row(
            "nearest_neighbour", "primary_top1", accuracy,
            float(nn_metrics["top1_accuracy"]), chance=chance, excess_ratio=ratio,
            decisive=False, interval=interval(nn_hits),
            note="cosine retrieval in the ambient activation space with no fit at "
            "all. The audit gate lists it; EXP-R2-213's decisive set does not, so it "
            "is reported and decides nothing here",
        )
    )
    for name in ("shuffled_pair", "rank_matched", "shuffled_fit"):
        rows.append(
            ca.baseline_row(
                f"{name}_q975", "primary_top1", accuracy,
                float(nulls[name]["decision_level"]), chance=chance,
                excess_ratio=ratio, decisive=False,
                note="the same null read at its 97.5th percentile rather than its "
                "mean: stricter than the frozen criterion and reported beside it",
            )
        )
    for row_record in rows:
        row_record["split"] = split
    return rows


def cell_verdict(
    cell: Mapping[str, Any],
    args: argparse.Namespace,
    *,
    split: str,
    status: Mapping[str, Any],
) -> dict[str, Any]:
    """The frozen decision for one cell on one split, read off the primary statistic."""

    block = cell["splits"][split]
    return ca.admission_verdict(
        block["baseline_rows"],
        excess_ratio=args.excess_ratio,
        detection_floor=args.decision_threshold,
        observed_excess=block["ladder"][args.alignment_method][PRIMARY_DIRECTION][
            "top1_excess"
        ],
        behavioural_status=status,
        mode=args.mode,
    )


ATTAINABILITY_SPLIT = "eval"


def attainability_verdict(
    raw_cell: Mapping[str, Any],
    args: argparse.Namespace,
    *,
    status: Mapping[str, Any],
) -> dict[str, Any]:
    """A35-0: the raw-description arm decides whether the masked arm is read at all.

    Attainability before control. A bar the arm with the concept name present
    cannot reach is a specification defect and not a protein result, so its
    failure voids the whole ladder rather than being reported beside a masked
    number -- two of D3.h's criteria were voided for skipping exactly this
    ordering, one of them unreachable at any sample size.

    **The gate is one baseline and the artefact says which.** EXP-R2-213 names it
    in the singular: A35-1's margin over ``shuffled_pair``. The rest of the raw
    arm's table is reported here in full and decides nothing at this step, because
    a raw arm that clears ``shuffled_pair`` and loses to a composition or 3-mer
    surrogate is the pre-declared **surface-statistics** branch -- a statement
    about the method, read as a measured negative on the masked arm -- while a raw
    arm that cannot clear ``shuffled_pair`` is a statement about the instrument.
    Gating on the whole decisive set files the first outcome under the second, and
    ``REFERENCE_ONLY`` -- which is not a failure at all -- would make the gate
    unreachable for the pre-adaptation reference whatever its numbers said.
    """

    gate = ca.attainability_gate(
        raw_cell["splits"][ATTAINABILITY_SPLIT]["baseline_rows"]
    )
    verdict = cell_verdict(raw_cell, args, split=ATTAINABILITY_SPLIT, status=status)
    attainable = bool(gate["attainable"])
    return {
        "verdict": "ATTAINABLE" if attainable else "VOID_SPECIFICATION_DEFECT",
        "attainable": attainable,
        "gate_baseline": gate["gate_baseline"],
        "split": ATTAINABILITY_SPLIT,
        "gate": gate,
        "raw_arm_verdict": verdict,
        "reason": (
            f"the raw-description arm, with the concept name present, clears A35-1's "
            f"two conditions over {gate['gate_baseline']} on the "
            f"{ATTAINABILITY_SPLIT} split, so the masked arm is a control on a bar "
            f"the design can reach. The remaining baselines are reported in "
            f"raw_arm_verdict and gate nothing here"
            if attainable
            else f"the raw-description arm does NOT clear A35-1's two conditions "
            f"over {gate['gate_baseline']} on the {ATTAINABILITY_SPLIT} split "
            f"(status {gate['status']}). A bar the attainability arm cannot reach is "
            "a specification defect and not a protein result, so the whole ladder is "
            "VOID and the masked arm is not read at all"
        ),
    }


def _bridge_row(cell: Mapping[str, Any], split: str) -> Mapping[str, Any]:
    matches = [
        row
        for row in cell["splits"][split]["baseline_rows"]
        if row["baseline"] == ca.A35_1B_BASELINE and row["axis"] == "primary_top1"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"A35-1b is read on {ca.A35_1B_BASELINE!r} and the {split} split carries "
            f"{len(matches)} such rows"
        )
    return matches[0]


def _bridge_over_full(row: Mapping[str, Any]) -> dict[str, Any]:
    """A35-1b's interpretable quantity, on the primary statistic, or why there is none."""

    full = row["observed_excess"]
    bridge = row["baseline_excess"]
    ratio = None
    reason = None
    if bridge is None:
        reason = "the bridge arm carries no value on this split"
    elif full is None or full <= 0.0:
        reason = (
            "the unrestricted arm has no excess over chance on this split, so a "
            "restriction's share of it is not defined"
        )
    else:
        ratio = float(bridge) / float(full)
    return {
        "bridge_top1_excess": bridge,
        "full_top1_excess": full,
        "bridge_over_full": ratio,
        "undefined_reason": reason,
    }


def a35_1b_restriction(
    raw_cell: Mapping[str, Any],
    deciding_cell: Mapping[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """A35-1b: the concept restriction, reported, and gating only where it can.

    EXP-R2-213 amendment 1 moved ``bridge_specific`` out of A35-1's decisive set
    for two reasons, and the second is the one that fixes the criterion rather
    than merely relaxing it: at the limit the original bar was backwards, because
    a genuine cross-modal alignment carried by the concepts declared on both sides
    *predicts* ``bridge ~ full``. So the reported quantity is the ratio
    ``bridge / full`` on the primary statistic on both splits, and clause (ii)
    becomes decisive against this baseline only where the **raw-description arm**
    demonstrates at the run's own settings that the declared margin against it is
    reachable -- the same attainability-before-control ordering A35-0 applies, on
    the same arm and the same split.
    """

    attainability_row = _bridge_row(raw_cell, ATTAINABILITY_SPLIT)
    reachable = attainability_row["meets_excess_ratio"]
    ratio = _bridge_over_full(attainability_row)
    if reachable is True:
        reason = (
            f"the raw-description arm clears {args.excess_ratio}x over "
            f"{ca.A35_1B_BASELINE} on the {ATTAINABILITY_SPLIT} split, so the margin "
            "is reachable at these settings and clause (ii) is decisive against it "
            "on the arm the verdict is read on"
        )
    elif reachable is False:
        reason = (
            f"the raw-description arm does not clear {args.excess_ratio}x over "
            f"{ca.A35_1B_BASELINE} on the {ATTAINABILITY_SPLIT} split: the achieved "
            f"bridge/full ratio is {ratio['bridge_over_full']}, so the margin is not "
            "reachable at these settings. Clause (ii) is non-applicable for this "
            "baseline and A35-1b is reported rather than gating (amendment 1)"
        )
    else:
        reason = (
            f"the raw-description arm's {ca.A35_1B_BASELINE} row carries no "
            "effect-size reading "
            f"({attainability_row['inapplicable_reason'] or 'no baseline value'}), "
            "so the margin cannot be shown reachable and clause (ii) is "
            "non-applicable for this baseline"
        )
    gating = reachable is True
    ratios = {"raw": {
        split: _bridge_over_full(_bridge_row(raw_cell, split))
        for split in ("eval", "family_holdout")
    }}
    if deciding_cell is not None:
        ratios[str(ca.DECIDING_DESCRIPTION_VARIANT)] = {
            split: _bridge_over_full(_bridge_row(deciding_cell, split))
            for split in ("eval", "family_holdout")
        }
    return {
        "criterion": "A35-1b",
        "baseline": ca.A35_1B_BASELINE,
        "amended_by": f"{ca.PRE_REGISTRATION} amendment 1",
        "gating": gating,
        "margin_reachable_on_the_raw_arm": reachable,
        "attainability_split": ATTAINABILITY_SPLIT,
        "attainability_row": dict(attainability_row),
        "reason": reason,
        "bridge_over_full": ratios,
        "inapplicable_reason": None if gating else reason,
        "note": ca.A35_1B_NOTE,
    }


def frozen_branch(
    stage_verdict: str, verdicts: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Which of EXP-R2-213's pre-declared branches this outcome is, named in the artefact.

    The pre-registration's branch table gives each outcome its own row and its own
    subject, and the two that are easiest to confuse have opposite ones: a raw arm
    failing A35-0 is about *the instrument, not the modality*, while a composition
    or 3-mer surrogate breaking A35-1(ii) is about *the method* and is a measured
    negative rather than a void. Naming the branch in the artefact is what stops a
    reader having to re-derive it from the verdict string.
    """

    failing = sorted(
        {
            name
            for record in verdicts.values()
            for name in record["baselines_failing_a_condition"]
        }
    )
    eval_verdict = verdicts.get("eval", {}).get("verdict")
    holdout_verdict = verdicts.get("family_holdout", {}).get("verdict")
    surrogates = sorted(set(failing) & {"composition", "kmer"})
    if stage_verdict == "VOID_SPECIFICATION_DEFECT":
        branch, statement, reading = (
            "a35_0_specification_defect",
            "the instrument, not the modality",
            "the raw-description arm did not clear A35-1's margin over "
            f"{ca.A35_0_GATE_BASELINE}, so the ladder is void and the masked arm was "
            "not read",
        )
    elif stage_verdict == "REFERENCE_ONLY":
        branch, statement, reading = (
            "pre_adaptation_reference",
            "attribution",
            "this checkpoint's protein mode is behaviourally unmeasurable, so the "
            "run is a representational reference and admits nothing",
        )
    elif stage_verdict == "PASS":
        branch, statement, reading = (
            "authorised",
            "existence, on this lineage at this site",
            "every pre-registered condition holds on both splits",
        )
    elif stage_verdict == "UNDERPOWERED":
        branch, statement, reading = (
            "underpowered",
            "the resampling unit count, not the model",
            "a condition holds but a decisive baseline carries no publishable "
            "interval",
        )
    elif eval_verdict == "PASS" and holdout_verdict != "PASS":
        branch, statement, reading = (
            "within_corpus_not_across_families",
            "generalisation, not existence",
            "A35-1 holds on the group-disjoint eval split and fails on the "
            "family-disjoint one; stage 36 is not authorised",
        )
    elif surrogates:
        branch, statement, reading = (
            "surface_statistics",
            "the method",
            "a composition or 3-mer surrogate broke A35-1(ii) ("
            + ", ".join(surrogates)
            + "), so the alignment is reported as an alignment of amino-acid "
            "composition rather than of concept. Where the surrogate also exceeds "
            "the masked arm, the model adds nothing over composition",
        )
    else:
        branch, statement, reading = (
            "measured_negative",
            "the model, at this site and on this lineage",
            "the masked arm did not clear a decisive baseline; STOP-35 registers it "
            "as a result rather than a failed round",
        )
    return {
        "branch": branch,
        "statement_about": statement,
        "reading": reading,
        "decisive_baselines_failing_a_condition": failing,
        "surrogate_baselines_failing": surrogates,
        "source": f"{ca.PRE_REGISTRATION}'s pre-declared branch table",
    }


def pre_registration_block(args: argparse.Namespace) -> dict[str, Any]:
    """What this run was produced under, declared from the constants that decide it.

    The amendment list is read from
    :data:`src.transfer.concept_alignment.PRE_REGISTRATION_AMENDMENTS` rather than
    written out here, so the artefact's claim about which text the code implements
    and the code's own frozen constants have one source. An amendment recorded in
    the register while the executing instrument does not implement it is the gap
    that produced EXP-R2-213's first mis-filed stage-35 read, and
    ``tests/test_concept_alignment.py`` checks this declaration against the
    constants it implies.
    """

    return {
        "record": ca.PRE_REGISTRATION,
        "amendments_implemented": list(ca.PRE_REGISTRATION_AMENDMENTS),
        "amendment_note": ca.AMENDMENT_1_NOTE,
        "primary_statistic": ca.PRIMARY_STATISTIC,
        "primary_statistic_note": ca.PRIMARY_STATISTIC_NOTE,
        "primary_direction": PRIMARY_DIRECTION,
        "decisive_baselines": list(ca.A35_1_BASELINES),
        "a35_0_gate_baseline": ca.A35_0_GATE_BASELINE,
        "a35_0_gate_note": ca.A35_0_GATE_NOTE,
        "a35_1b_baseline": ca.A35_1B_BASELINE,
        "a35_1b_note": ca.A35_1B_NOTE,
        "conditions": (
            "A35-1(i) the paired group-bootstrap 95% interval of the difference "
            "excludes zero; A35-1(ii) the excess over chance is at least "
            f"{args.excess_ratio}x the baseline's. Both required, on both splits"
        ),
        "detection_floor": float(args.decision_threshold),
        "resampling_unit": "dup_group (near-duplicate group)",
        "nonlinear_adapter_locked": True,
    }


def stop_35_record(stage_verdict: str) -> dict[str, Any]:
    """STOP-35, emitted explicitly rather than left to a reader.

    Any A35-1 failure on the eval split ends the campaign and stage 36 is
    unauthorised. Only ``PASS`` authorises it: ``UNDERPOWERED`` is a margin
    without an interval and ``REFERENCE_ONLY`` is not an admission at all.
    """

    stopped = stage_verdict != "PASS"
    return {
        "triggered": bool(stopped),
        "stage36_authorised": not stopped,
        "stage_verdict": stage_verdict,
        "reason": (
            "every pre-registered condition holds on the group-disjoint eval split "
            "and on the unseen-family split, so the causal stage is authorised"
            if not stopped
            else f"the stage verdict is {stage_verdict}. Any A35-1 failure on eval "
            "ends the campaign: stage 36 is UNAUTHORISED and no concept-vector "
            "hand-off is emitted"
        ),
    }


def audit_gate_all_baselines(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Whether the alignment also clears the baselines outside the frozen set.

    The audit's own wording is "beat every applicable baseline", and EXP-R2-213
    amendment 1 freezes a decisive set of six that does not include the ambient
    nearest-neighbour read, the nulls at their 97.5th percentile, or A35-1b's
    restriction where its margin was not shown reachable. Reported as a separate,
    non-binding reading so the frozen verdict is neither softened nor widened by
    it.
    """

    extra = [
        row
        for row in rows
        if not row["decisive"] and row["applicable"] and row["baseline"] != "chance"
    ]
    unmet = sorted(
        {row["baseline"] for row in extra if row["passes_both_conditions"] is not True}
    )
    return {
        "clears_every_reported_baseline": not unmet,
        "reported_baselines_not_cleared": unmet,
        "binding": False,
        "note": (
            "non-binding. EXP-R2-213's decisive set is "
            f"{list(ca.A35_1_BASELINES)}; this row records how the alignment stands "
            "against the baselines outside it"
        ),
    }


# ------------------------------------------------------------------ synthetic


def synthetic_inputs(args: argparse.Namespace, *, planted: bool) -> CellInputs:
    """Paired representations whose answer is known before the analysis runs.

    ``planted`` builds two views of one set of latent factors, so a linear map
    between them exists by construction and the ladder must find it. Without it
    the source side is drawn independently, so every baseline is the truth and the
    stage must return ``FAIL``. One of the latent factors also drives the
    sequences' composition, which is what makes the composition surrogate a live
    competitor in the check rather than a formality.
    """

    rng = np.random.default_rng(args.seed)
    n = int(args.synthetic_records)
    d = int(args.synthetic_d_model)
    k = int(args.synthetic_factors)
    latent = rng.normal(size=(n, k))
    source_basis = rng.normal(size=(k, d)) / np.sqrt(k)
    target_basis = rng.normal(size=(k, d)) / np.sqrt(k)
    noise = float(args.synthetic_noise)
    driver = latent if planted else rng.normal(size=(n, k))
    source = driver @ source_basis + noise * rng.normal(size=(n, d))
    target = latent @ target_basis + noise * rng.normal(size=(n, d))

    # One factor tilts the residue composition, so the surrogate has something
    # real to recover and the check prices it rather than assuming it fails.
    sequences: list[str] = []
    for row in range(n):
        weights = np.ones(len(AA20))
        weights[:5] *= float(np.exp(latent[row, 0]))
        weights = weights / weights.sum()
        length = int(rng.integers(64, 220))
        sequences.append("".join(rng.choice(list(AA20), size=length, p=weights)))

    # Two records per near-duplicate group and four per family, with both cut
    # points forced onto a multiple of the family block: the splits must not
    # straddle a group, which is exactly what the cohort's own certificate
    # requires and what load_cohort re-checks on a real file.
    dup = np.array([f"dup{row // 2}" for row in range(n)])
    family = np.array([f"fam{row // 4}" for row in range(n)])
    n_fit = (n // 2 // 4) * 4
    n_eval = ((n - n_fit) // 2 // 4) * 4
    if n_fit < 8 or n_eval < 8 or n - n_fit - n_eval < 8:
        raise ValueError("--synthetic-records is too small to carry three splits")
    split = np.array(
        ["fit"] * n_fit + ["eval"] * n_eval + ["family_holdout"] * (n - n_fit - n_eval)
    )
    concepts = tuple(("ec", f"1.{factor}.-.-") for factor in range(min(3, k)))
    labels = {key: latent[:, factor] > 0.0 for factor, key in enumerate(concepts)}
    return CellInputs(
        source=source,
        target=target,
        sequences=tuple(sequences),
        lengths=np.array([len(sequence) for sequence in sequences], dtype=np.float64),
        dup_group=dup,
        family_group=family,
        split=split,
        accession=tuple(f"SYN{row:05d}" for row in range(n)),
        concepts=concepts,
        bridge=concepts[:1],
        labels=labels,
    )


def run_synthetic_check(args: argparse.Namespace) -> dict[str, Any]:
    """Both cells, and the verdict each of them must produce."""

    cells: dict[str, Any] = {}
    for name, planted in (("planted_alignment", True), ("no_alignment", False)):
        inputs = synthetic_inputs(args, planted=planted)
        cell = evaluate_cell(inputs, args, with_intervals=True)
        cell["verdict"] = ca.admission_verdict(
            cell["splits"]["eval"]["baseline_rows"],
            excess_ratio=args.excess_ratio,
            detection_floor=args.decision_threshold,
            observed_excess=cell["splits"]["eval"]["ladder"][args.alignment_method][
                PRIMARY_DIRECTION
            ]["top1_excess"],
            behavioural_status={"checkpoint_name": "synthetic", "measurable": True},
            mode="text",
        )
        cell["stop_35"] = stop_35_record(cell["verdict"]["verdict"])
        cell["expected_verdict"] = "PASS" if planted else "FAIL"
        cells[name] = cell
    return {
        "kind": "synthetic_instrument_check",
        "cells": cells,
        "certificate": {
            name: {
                "expected": cell["expected_verdict"],
                "observed": cell["verdict"]["verdict"],
                "agrees": cell["expected_verdict"] == cell["verdict"]["verdict"],
                "eval_top1": cell["splits"]["eval"]["ladder"][args.alignment_method][
                    PRIMARY_DIRECTION
                ]["top1_accuracy"],
                "eval_top1_chance": cell["splits"]["eval"]["gallery"]["chance_top1"],
            }
            for name, cell in cells.items()
        },
        "note": (
            "a linear alignment's retrieval number is an unsupervised claim that "
            "nothing in a real run can falsify, so this is the only place it is "
            "falsifiable: one cell where the correspondence exists by construction "
            "and one where it does not"
        ),
    }


# ----------------------------------------------------------------------- main


def artefact_name(checkpoint: Path, mode: str, rendering: str) -> str:
    """Basename from checkpoint, mode and rendering -- never a fixed string.

    ``21_joint_mode_qualification.py:919`` writes every run to one fixed
    ``joint_mode_qualification.json``, so a second checkpoint in one output
    directory overwrites the first without a word. This stage's three campaign
    axes are all in the name.
    """

    def safe(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "-", value)

    return f"concept_alignment__{safe(checkpoint.name)}__{safe(mode)}__{safe(rendering)}.json"


def reference_comparison(
    args: argparse.Namespace,
    *,
    cell: Mapping[str, Any],
    arm: str,
) -> dict[str, Any]:
    """A35-4, as a reported verdict rather than a comment.

    If the pre-adaptation reference reaches the same primary statistic, the
    attribution to protein adaptation is withdrawn while the measurement stands.
    Both of A35-1's conditions are evaluable because the reference artefact
    carries its own per-record top-1 vector over the same cohort in the same
    order, drawn against the same common gallery.
    """

    if args.reference_artefact is None:
        return {
            "verdict": "REFERENCE_NOT_SUPPLIED",
            "reason": (
                "no --reference-artefact was given, so A35-4 was not evaluated. It "
                "is reported rather than passed over: the pre-adaptation reference "
                "is a separate invocation of this pipeline and its absence is a "
                "state of this run, not a property of the result"
            ),
            "binding": False,
        }
    reference = json.loads(
        require_input_path(Path(args.reference_artefact).resolve(), "--reference-artefact")
        .read_text(encoding="utf-8")
    )
    if int(reference["settings"]["gallery_size"]) != int(args.gallery_size) or int(
        reference["settings"]["seed"]
    ) != int(args.seed):
        raise ValueError(
            "the reference artefact was drawn against a different common gallery "
            "(size or seed), so its accuracy is not the same statistic as this run's"
        )
    key = f"layer{args.decision_layer}__{arm}"
    if key not in reference["cells"] or reference["cells"][key].get("withheld"):
        return {
            "verdict": "REFERENCE_ARM_UNAVAILABLE",
            "reason": (
                f"the reference artefact carries no readable {key} cell, so there is "
                "nothing to compare on. If its raw arm failed A35-0 its masked arm "
                "was withheld by the same rule that governs this run"
            ),
            "binding": False,
        }
    # A35-4 is binding only where it is read on the arm the verdict is read on.
    # A void ladder has no masked cell to compare, and the raw-arm reading that
    # remains is reported rather than passed over -- but it is not the attribution
    # decision, because the deciding variant is the masked one.
    binding = arm == ca.DECIDING_DESCRIPTION_VARIANT
    verdicts: dict[str, Any] = {
        "binding": binding,
        "arm": arm,
        "deciding_arm": ca.DECIDING_DESCRIPTION_VARIANT,
        "splits": {},
    }
    if not binding:
        verdicts["non_binding_reason"] = (
            f"read on the {arm!r} arm, which is not the deciding description "
            f"variant ({ca.DECIDING_DESCRIPTION_VARIANT!r}). The attribution "
            "question is about the arm the verdict is read on"
        )
    for split in ("eval", "family_holdout"):
        here = cell["splits"][split]
        there = reference["cells"][key]["splits"][split]
        if here["primary_per_record"]["accessions"] != there["primary_per_record"]["accessions"]:
            raise ValueError(
                f"the reference artefact's {split} split is a different record set, "
                "so no paired comparison against it is defined"
            )
        chance = float(here["gallery"]["chance_top1"])
        row = ca.baseline_row(
            "pre_adaptation_reference", "primary_top1",
            float(here["ladder"][args.alignment_method][PRIMARY_DIRECTION]["top1_accuracy"]),
            float(there["ladder"][args.alignment_method][PRIMARY_DIRECTION]["top1_accuracy"]),
            chance=chance, excess_ratio=args.excess_ratio, decisive=False,
            interval=_paired_interval(
                np.asarray(here["primary_per_record"]["top1"], dtype=np.float64),
                np.asarray(there["primary_per_record"]["top1"], dtype=np.float64),
                np.asarray(here["primary_per_record"]["dup_groups"]),
                args=args,
            ),
            note="the pre-adaptation checkpoint through this identical pipeline. It "
            "is representational on both sides and carries no behavioural claim",
        )
        verdicts["splits"][split] = {
            "row": row,
            "verdict": (
                "ATTRIBUTION_HELD"
                if row["passes_both_conditions"] is True
                else "ATTRIBUTION_WITHDRAWN_MEASUREMENT_STANDS"
            ),
            "reason": (
                "the adapted checkpoint clears the reference under both conditions, "
                "so the measurement may be attributed to what adaptation added"
                if row["passes_both_conditions"] is True
                else "the pre-adaptation reference reaches this statistic too, so the "
                "attribution to protein adaptation is withdrawn while the measurement "
                "stands. Nothing measured here is retracted; what is withdrawn is the "
                "claim about where it came from"
            ),
        }
    verdicts["verdict"] = (
        "ATTRIBUTION_HELD"
        if all(row["verdict"] == "ATTRIBUTION_HELD" for row in verdicts["splits"].values())
        else "ATTRIBUTION_WITHDRAWN_MEASUREMENT_STANDS"
    )
    return verdicts


def limitations_block(*, mode: str, status: Mapping[str, Any]) -> dict[str, Any]:
    block: dict[str, Any] = {
        "representational_only": (
            "every quantity here is a pooled activation, a linear map between two "
            "clouds of them, or a ranking derived from those. Nothing in this "
            "artefact is a behavioural claim about either mode, and the gate's "
            "graded intervention is not run here"
        ),
        "sequence_level_only": (
            "the estimand is one vector per record per side. Position-level "
            "correspondence is out of scope and is not merely unmeasured: L31 "
            "measures ProLLaMA's tokenizer leaving a single substitution "
            "token-aligned on 47.0-54.5% of instances, and the surviving half is "
            "the BPE-stable half rather than a random one, so a residue-to-word "
            "estimand would be computed on a composition-selected cohort"
        ),
        "token_unit_not_residue_unit": (
            "the prollama family declares the TOKEN as its symbol unit at about "
            "1.536 residues per token. No magnitude here is commensurable with a "
            "residue-unit arm's (L23, Appendix B rule 26), and this stage compares "
            "nothing across families"
        ),
        "resampling_unit_is_the_near_duplicate_group": (
            "EXP-R2-213 fixes the paired bootstrap's unit at the near-duplicate "
            "group, and on this cohort those groups are close to singletons, so the "
            "interval is close to a record bootstrap. The family group is the wider "
            "unit of independence for proteins and its count is reported beside "
            "every interval; an interval over near-duplicate groups is narrower than "
            "one over families would be, and that is a property of the frozen "
            "criterion rather than of the data"
        ),
        "one_checkpoint_one_cohort_one_draw": (
            "one checkpoint, one rendering, one cohort, one seed. The skip-offset "
            "and second-draw sensitivity Appendix B rule 1 asks for is a second run "
            "of this stage against a second cohort draw"
        ),
        "concept_axis_is_a_diagnostic": (
            "the concept axis is a classification accuracy and decides nothing. "
            "A35-1's two conditions are stated on the primary retrieval statistic, "
            "and comparing an AUC excess against a retrieval excess would be a "
            "comparison across statistics"
        ),
        "nonlinear_adapter": ca.NONLINEAR_ADAPTER_NOTE,
        "verdict_scope": (
            "the verdict is read at ONE pre-registered layer on the "
            f"{ca.DECIDING_DESCRIPTION_VARIANT!r} description variant, and every "
            "quantity is per layer with no cross-layer mean anywhere (L32). Other "
            "layers and the raw arm are reported and decide nothing, which is what "
            "stops a verdict being read off whichever cell happened to pass"
        ),
        "bridge_baseline_reading": (
            "the gate names a 'bridge-specific-concept baseline' without fixing its "
            "construction. It is implemented as the SAME retrieval task restricted "
            "to the span of the concepts declared on both sides -- annotated on the "
            "protein side and named in the description, which the cohort's "
            "masked_terms records -- so that it is commensurable with the ladder "
            "under A35-1(ii). That is an interpretation of the gate's wording and is "
            "recorded as one. EXP-R2-213 amendment 1 makes it A35-1b: reported as "
            "the bridge/full ratio and decisive only where the raw arm shows the "
            "margin against it is reachable"
        ),
        "bridge_concepts_are_the_cohorts_declared_concepts": (
            "'named in the description' is decided against "
            "sequence_description.concept_surface_forms, which is the object the "
            "cohort stage built masked_terms from, so only concepts the cohort "
            "DECLARES surface forms for can enter the bridge span. A concept read "
            "off an annotation column alone -- most of this stage's declared "
            "concepts, and every Pfam accession -- carries no declared text side "
            "and is not in it. The count of concepts carrying declared surface "
            "forms is reported beside the bridge count rather than left to be "
            "inferred from the difference"
        ),
        "description_only_reading": (
            "likewise implemented as a RETRIEVAL arm rather than a concept "
            "classification: the stronger of a query-independent gallery ranking "
            "and a ridge from the description representation to log length. A "
            "classification statistic could not be compared against a retrieval "
            "excess under one criterion"
        ),
    }
    if mode == "text":
        block["text_control_is_easy_by_construction"] = (
            "the control pairs a record's name with its own description, and a name "
            "is frequently a substring of its description. It certifies that this "
            "pipeline can find an alignment that exists; it does not certify that it "
            "can find a hard one, and it is not evidence about the protein cell's "
            "difficulty"
        )
    if status.get("measurable") is not True and mode == "protein":
        block["pre_adaptation_reference"] = str(status.get("reason", ""))
    return block


def _cell_inputs(
    captured: Mapping[str, Mapping[int, np.ndarray]],
    variant: str,
    layer: int,
    columns: Mapping[str, Any],
) -> CellInputs:
    return CellInputs(
        source=captured["source"][layer],
        target=captured[variant][layer],
        **columns,
    )


def main() -> None:
    args = build_parser().parse_args()
    resolve(args)
    args.out.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).isoformat()
    provenance = {
        "runner": {
            "path": "scripts/transfer/35_concept_alignment.py",
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "modules": {name: sha256_file(REPO_ROOT / name) for name in PROVENANCE_MODULES},
    }
    settings = {
        key: (str(value) if isinstance(value, Path) else list(value) if isinstance(value, tuple) else value)
        for key, value in vars(args).items()
    }

    if args.synthetic_check:
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": created,
            "settings": {
                key: value for key, value in settings.items()
                if key not in CAMPAIGN_ONLY_FLAGS
            },
            "pre_registration": pre_registration_block(args),
            "provenance": provenance,
            **run_synthetic_check(args),
            "limitations": {
                "synthetic": (
                    "these cells carry no model and no cohort. They check that the "
                    "analysis recovers a correspondence that exists and refuses one "
                    "that does not; they say nothing about any checkpoint"
                ),
                "nonlinear_adapter": ca.NONLINEAR_ADAPTER_NOTE,
            },
        }
        destination = args.out / "concept_alignment__synthetic_check.json"
        write_json(destination, payload)
        for name, row in payload["certificate"].items():
            print(
                f"[{name:18s}] expected {row['expected']:14s} observed "
                f"{row['observed']:14s} agrees={row['agrees']}  top-1 "
                f"{row['eval_top1']:.4f} against chance {row['eval_top1_chance']:.4f}"
            )
        print(f"wrote {destination}")
        return

    checkpoint = require_input_path(Path(args.checkpoint).resolve(), "--checkpoint")
    cohort = ca.load_cohort(Path(args.cohort))
    status = ca.protein_mode_behavioural_status(checkpoint)
    # Resolved before the weights are loaded, so a missing ontology costs seconds
    # rather than a model load: it decides which concepts are defined on both
    # sides, which is an input to the run and not a detail of its reporting.
    go_obo = resolve_go_obo(cohort, args)
    surface_forms = declared_surface_forms(go_obo)
    print(f"[paths] checkpoint {checkpoint}")
    print(f"[paths] cohort     {cohort.path}")
    print(f"[paths] go         {go_obo}")
    print(f"[paths] out        {args.out.resolve()}")
    print(f"[cohort] {cohort.counts()}")

    resolved, facts, handles = open_modes(args)
    if args.depth_grid is not None and args.layers is None:
        args.layers = tuple(analysis_layers(int(facts["n_layers"]), args.depth_grid))
        settings["layers"] = list(args.layers)
        print(f"[depth] {list(args.depth_grid)} -> layers {list(args.layers)}")
    if args.decision_layer not in args.layers:
        raise ValueError(
            f"--decision-layer {args.decision_layer} is not in --layers "
            f"{list(args.layers)}; the verdict must be read at a layer the run measured"
        )

    namespaces = tuple(piece for piece in args.concept_namespaces.split(",") if piece)
    concepts = tuple(
        ca.declared_concepts(
            cohort.records, namespaces, min_positive=args.min_concept_positives
        )
    )
    if not concepts:
        raise ValueError(
            f"no concept in namespaces {list(namespaces)} carries "
            f"{args.min_concept_positives} records on each side in every split, so "
            "the concept axis has nothing to read. Lower --min-concept-positives "
            "only with the power consequence stated: below the eight-unit floor a "
            "concept cannot carry an interval"
        )
    bridge = tuple(
        ca.bridge_concepts(cohort.records, concepts, surface_forms=surface_forms)
    )
    n_with_forms = sum(1 for key in concepts if key in surface_forms)
    print(
        f"[concepts] {len(concepts)} declared, {n_with_forms} carrying declared "
        f"surface forms, {len(bridge)} of those named in the cohort's masked_terms"
    )

    captured: dict[str, dict[int, np.ndarray]] = {
        "source": ca.mode_representations(
            handles[args.mode], args.rendering, args.mode,
            source_strings(cohort, args.mode), args.layers,
            args.pooling, args.device, args.batch_size, args.dtype,
        )
    }
    print(f"[capture] source ({args.mode}) done", flush=True)
    for variant in DESCRIPTION_VARIANTS:
        captured[variant] = ca.mode_representations(
            handles["text"], args.rendering, "text",
            description_strings(cohort, variant), args.layers,
            args.pooling, args.device, args.batch_size, args.dtype,
        )
        print(f"[capture] description ({variant}) done", flush=True)

    columns = {
        "sequences": tuple(str(record["sequence"]) for record in cohort.records),
        "lengths": np.asarray(
            [int(record["length"]) for record in cohort.records], dtype=np.float64
        ),
        "dup_group": np.asarray([str(record["dup_group"]) for record in cohort.records]),
        "family_group": np.asarray(
            [str(record["family_group"]) for record in cohort.records]
        ),
        "split": np.asarray([str(record["split"]) for record in cohort.records]),
        "accession": tuple(str(record["accession"]) for record in cohort.records),
        "concepts": concepts,
        "bridge": bridge,
        "labels": {
            key: ca.concept_labels(cohort.records, key[0], key[1]) for key in concepts
        },
    }

    # ---- A35-0: attainability BEFORE control. The raw arm is read first at every
    # layer, and the masked arm is not read at all unless the raw arm at the
    # decision layer clears the frozen criteria on the group-disjoint eval split.
    cells: dict[str, Any] = {}
    for layer in args.layers:
        cell = evaluate_cell(
            _cell_inputs(captured, "raw", layer, columns),
            args,
            with_intervals=layer == args.decision_layer,
        )
        cell.update({"layer": int(layer), "description_variant": "raw"})
        cells[f"layer{layer}__raw"] = cell
        print(
            f"[cell] layer {layer:2d} raw    eval top-1 "
            f"{cell['splits']['eval']['ladder'][args.alignment_method][PRIMARY_DIRECTION]['top1_accuracy']:.4f} "
            f"(chance {cell['splits']['eval']['gallery']['chance_top1']:.4f})",
            flush=True,
        )

    raw_deciding = cells[f"layer{args.decision_layer}__raw"]
    a35_0 = attainability_verdict(raw_deciding, args, status=status)
    attainable = bool(a35_0["attainable"])
    # A35-1b's attainability is measured on the SAME arm and split as A35-0's, and
    # it decides whether the bridge restriction gates the arm the verdict is read
    # on or is reported beside it (EXP-R2-213 amendment 1).
    a35_1b = a35_1b_restriction(raw_deciding, None, args)

    if attainable:
        for layer in args.layers:
            cell = evaluate_cell(
                _cell_inputs(captured, "masked", layer, columns),
                args,
                with_intervals=layer == args.decision_layer,
                bridge_decisive=True,
                bridge_inapplicable_reason=a35_1b["inapplicable_reason"],
            )
            cell.update({"layer": int(layer), "description_variant": "masked"})
            cells[f"layer{layer}__masked"] = cell
            print(
                f"[cell] layer {layer:2d} masked eval top-1 "
                f"{cell['splits']['eval']['ladder'][args.alignment_method][PRIMARY_DIRECTION]['top1_accuracy']:.4f}",
                flush=True,
            )
    else:
        for layer in args.layers:
            cells[f"layer{layer}__masked"] = {
                "layer": int(layer),
                "description_variant": "masked",
                "withheld": True,
                "reason": a35_0["reason"],
            }

    deciding_key = f"layer{args.decision_layer}__{ca.DECIDING_DESCRIPTION_VARIANT}"
    deciding_cell = cells[deciding_key]
    if attainable:
        a35_1b = a35_1b_restriction(raw_deciding, deciding_cell, args)
        verdicts = {
            split_name: cell_verdict(
                deciding_cell, args, split=split_name, status=status
            )
            for split_name in ("eval", "family_holdout")
        }
        audit_gate = {
            split_name: audit_gate_all_baselines(
                deciding_cell["splits"][split_name]["baseline_rows"]
            )
            for split_name in ("eval", "family_holdout")
        }
        precedence = ("REFERENCE_ONLY", "FAIL", "UNDERPOWERED", "PASS")
        stage_verdict = next(
            name
            for name in precedence
            if any(record["verdict"] == name for record in verdicts.values())
        )
        reference = reference_comparison(
            args, cell=deciding_cell, arm=ca.DECIDING_DESCRIPTION_VARIANT
        )
    else:
        verdicts = {}
        audit_gate = {}
        stage_verdict = "VOID_SPECIFICATION_DEFECT"
        reference = reference_comparison(args, cell=raw_deciding, arm="raw")

    stop_35 = stop_35_record(stage_verdict)
    branch = frozen_branch(stage_verdict, verdicts)
    stopped = bool(stop_35["triggered"])

    permitted = (
        not stopped
        and (args.mode != "protein" or status.get("measurable") is True)
    )
    if not permitted:
        handoff = {
            "emitted": False,
            "refusal": (
                stop_35["reason"]
                if stopped
                else f"{status['checkpoint_name']}: {status.get('reason', '')} No "
                "concept direction measured here is handed to "
                "36_concept_injection.py for this checkpoint, because the graded "
                "response it would read does not exist on this mode"
            ),
        }
    else:
        # The guard is called on the permitted branch too, so that it -- and not
        # the branch above it -- is what decides. A disagreement between the two
        # raises here rather than emitting a hand-off nobody checked.
        ca.assert_behavioural_read_permitted(checkpoint, args.mode)
        handoff = {
            "emitted": True,
            "layer": int(args.decision_layer),
            "site": ca.REPRESENTATION_SITE,
            "pooling": args.pooling,
            "description_variant": ca.DECIDING_DESCRIPTION_VARIANT,
            "concept_vectors": deciding_cell["concept_vectors"],
            "note": (
                "the concept directions 36_concept_injection.py steers along, with "
                "the sigma each graded step is a multiple of. They are declared on "
                "the TEXT side of this checkpoint and live in this cell's fitted "
                "subspace, whose basis is recorded under 'subspace'"
            ),
        }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": created,
        "kind": "concept_alignment",
        "pre_registration": pre_registration_block(args),
        "settings": settings,
        "provenance": provenance,
        "checkpoint": {**facts, "resolved_path": str(resolved)},
        "protein_mode_behavioural_status": status,
        "cohort": cohort.facts(),
        "declared_concepts": [f"{ns}:{term}" for ns, term in concepts],
        "bridge_concepts": [f"{ns}:{term}" for ns, term in bridge],
        "concept_declaration": {
            "namespaces": list(namespaces),
            "n_declared": len(concepts),
            "n_with_declared_surface_forms": n_with_forms,
            "n_bridge": len(bridge),
            "go_obo": str(go_obo),
            "note": (
                "a concept is defined on both sides when the cohort declares surface "
                "forms for it and one of them is in the cohort's own masked_terms. "
                "The forms come from sequence_description.concept_surface_forms, "
                "which is what the cohort stage built masked_terms from; the "
                "concept's identifier is not what a curated description calls it, "
                "and keying this test on the identifier compares two vocabularies "
                "that cannot meet"
            ),
        },
        "representation": {
            "site": ca.REPRESENTATION_SITE,
            "site_note": ca.REPRESENTATION_SITE_NOTE,
            "pooling": args.pooling,
            "source_mode": args.mode,
            "source_field": "sequence" if args.mode == "protein" else "name",
            "target_mode": "text",
            "rendering": handles[args.mode].rendering_note,
            "content_positions": handles[args.mode].scoring_note,
        },
        "cells": cells,
        "a35_0_attainability": a35_0,
        "a35_1b_concept_restriction": a35_1b,
        "a35_4_pre_adaptation_reference": reference,
        "audit_gate_all_baselines": audit_gate,
        "verdicts": verdicts,
        "stage_verdict": stage_verdict,
        "stop_35": stop_35,
        "frozen_branch": branch,
        "deciding_cell": deciding_key,
        "causal_handoff": handoff,
        "limitations": limitations_block(mode=args.mode, status=status),
    }
    ca.assert_per_layer_only(payload, args.layers)

    destination = args.out / artefact_name(checkpoint, args.mode, args.rendering)
    write_json(destination, payload)
    print()
    print(
        f"[A35-0] {a35_0['verdict']} on {a35_0['gate_baseline']}: "
        f"{a35_0['reason'][:110]}"
    )
    print(f"[A35-1b] gating={a35_1b['gating']}: {a35_1b['reason'][:110]}")
    for split_name, record in verdicts.items():
        print(f"[{split_name:15s}] {record['verdict']}: {record['reason'][:110]}")
    print(f"[A35-4] {reference['verdict']}")
    print(f"[stage] {stage_verdict}")
    print(f"[branch] {branch['branch']} -- a statement about {branch['statement_about']}")
    print(f"[STOP-35] triggered={stop_35['triggered']} stage36_authorised={stop_35['stage36_authorised']}")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
