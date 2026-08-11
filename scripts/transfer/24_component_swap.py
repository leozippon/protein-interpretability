#!/usr/bin/env python3
"""Does continued protein pretraining's text cost travel with the vocabulary interface, or with the body?

**The measured fact this stage exists to attribute.** On the ProLLaMA lineage --
``Llama-2-7b-hf`` -> ``ProLLaMA_Stage_1`` -> ``ProLLaMA``, all ``LlamaForCausalLM``
32 x 4096 over one unmodified 32000-piece vocabulary with no added tokens -- the
text mode's context information falls **+5.52 -> +0.83 -> +0.74** nats while the
protein mode goes from unmeasurable to measurable at the same step (EXP-R2-152).
Reading the lineage's own training scripts, stage 1 carries
``modules_to_save="embed_tokens,lm_head"``: it fully retrains the input embedding
and the output head -- 262.1 M parameters, 45 per cent of its 582 M trainable
budget -- alongside LoRA r=128 on all seven projections, while stage 2 carries no
``modules_to_save`` and is pure LoRA r=64. So the stage that costs 4.69 nats of
text is exactly the stage that retrains the vocabulary interface, and the stage
that costs 0.10 nats does not touch it.

That is a correlation across two release points, and two points cannot separate
"the vocabulary interface carries the loss" from "the body carries the loss and
the interface was retrained beside it". **The three checkpoints are
architecturally identical with an identical tokenizer, so their weights are
interchangeable tensor for tensor**, and the experiment that turns the
correlation into an attribution is a component swap.

**What one run is.** One chimera and one artefact: a declared *component group*
taken from the donor checkpoint, everything else kept from the host, then measured
in one or both modes. Four runs answer the question for host/donor drawn from the
base and stage 1:

1. base host, base donor -- the reference, and its own identity anchor
2. stage 1 host, stage 1 donor -- the degraded endpoint, likewise
3. base host, stage 1 donor, ``vocabulary_interface`` -- base body, stage 1 interface
4. stage 1 host, base donor, ``vocabulary_interface`` -- stage 1 body, base interface

If the text-mode loss follows the vocabulary interface, cell 3 is degraded and
cell 4 restored; if it follows the body, the reverse. **If neither holds the loss
is not decomposable this way, and that is the finding** -- the artefact records
four independent measurements and does not compose them into an attribution.

**The estimand is 21_joint_mode_qualification.py's, imported rather than
restated.** Context information: the held-out unigram cross-entropy on the scored
symbols minus the model's own clean cross-entropy on the same symbols, per mode,
with that stage's cohort machinery, its held-out reference, its controls and its
measurability threshold. A chimera's number is only worth anything if it sits
directly beside the qualification figures it is read against, and Appendix B rule
12 -- a single declaration, imported, never reimplemented -- is what puts it
there.

**Component groups are declared, never inferred.** ``embedding`` is the input
embedding, ``lm_head`` the output head, ``vocabulary_interface`` both together --
exactly what ``modules_to_save`` retrained -- and ``body`` the complement. The
group names are located through the model's own ``get_input_embeddings`` and
``get_output_embeddings``, so the names moved are read off the loaded object
rather than spelled for one architecture; a group nobody declared is refused by
name.

**Two silent failures this stage refuses instead of measuring.** A tensor whose
shape differs between donor and host would, if coerced, produce a model that
loads, runs and generates plausible text while being meaningless -- this
repository's L24 failure shape, where a checkpoint loaded "successfully" into a
randomly initialised feed-forward stack and only scoring caught it. And an
embedding row moved between two *different* vocabularies indexes a different
symbol in each, which is the same failure reached from the other side. Shapes and
dtypes are compared before any write, the two tokenizers' vocabularies are
compared by digest rather than assumed identical, and both refuse loudly.

**One limitation travels with every protein-mode number here.** The base model's
protein mode is unmeasurable -- context information +0.084 nats per token,
reversal cost -0.001 nats per residue (EXP-R2-152) -- so on the protein side the
base is a *pre-adaptation reference*, not a behavioural control. A protein
magnitude from a cell built wholly or partly from it is not comparable with an
adapted checkpoint's, and the artefact says so in its own field.

An external baseline, not a registered panel stage: a checkpoint is reached by
path, so it cannot be scheduled through ``panel_contract.STAGE_CONTRACTS``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer import joint_modes  # noqa: E402
from src.transfer.arms import DEFAULT_CORPUS_DRAW_SEED, REPO  # noqa: E402
from src.transfer.budget import MIN_CONTEXT_INFORMATION_NATS  # noqa: E402
from src.transfer.io import sha256_file, write_json  # noqa: E402
from src.transfer.replaceable import checkpoint_weights_digest  # noqa: E402


def _load_stage(filename: str) -> Any:
    """Import a stage whose module name starts with a digit.

    ``21_joint_mode_qualification.py`` is imported rather than copied because this
    stage's numbers are only readable if they are the *same* computation that
    stage performs: the same estimand, the same held-out reference, the same
    cohort draw and the same measurability threshold. Appendix B rule 12 does not
    stop applying because the declaration lives in a file whose name starts with a
    digit.
    """

    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(f"_transfer_stage_{filename[:2]}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE21 = _load_stage("21_joint_mode_qualification.py")

SCHEMA_VERSION = "r2_transfer_component_swap_v1"
DEFAULT_OUT = REPO / "results/transfer/component_swap"

#: Modules and stages whose content decides these numbers, hashed into the
#: artefact. The qualification stage is first because it owns the estimand; the
#: rendering module second because it has been worth 2.9 nats/token when wrong.
PROVENANCE_MODULES = (
    "scripts/transfer/21_joint_mode_qualification.py",
    "src/transfer/joint_modes.py",
    "src/transfer/arms.py",
    "src/transfer/budget.py",
    "src/transfer/pathways.py",
    "src/transfer/replaceable.py",
    "src/transfer/io.py",
)

# ---------------------------------------------------------- component groups

EMBEDDING = "embedding"
LM_HEAD = "lm_head"
VOCABULARY_INTERFACE = "vocabulary_interface"
BODY = "body"

#: What each declared group IS, in the artefact's own words. The set is closed:
#: a group nobody declared is refused by name rather than resolved by pattern,
#: because a swap is only an attribution if what moved was decided in advance.
COMPONENT_GROUPS: dict[str, str] = {
    EMBEDDING: (
        "the input embedding, located through the model's own "
        "get_input_embeddings(). Separable only when the embedding and the output "
        "head are distinct tensors"
    ),
    LM_HEAD: (
        "the output head, located through the model's own get_output_embeddings(). "
        "Separable only when the embedding and the output head are distinct tensors"
    ),
    VOCABULARY_INTERFACE: (
        "the input embedding and the output head together -- exactly the two "
        "modules the ProLLaMA lineage's stage 1 lists in modules_to_save and fully "
        "retrains, and the group this stage exists to attribute"
    ),
    BODY: (
        "every tensor of the state dict that is NOT in the vocabulary interface: "
        "attention, feed-forward, norms and position information. The exact "
        "complement of vocabulary_interface, so the two groups partition the "
        "state dict and a body swap is the mirror image of an interface swap"
    ),
}

VERDICT_NOTE = STAGE21.VERDICT_NOTE

#: The protein-side reading rule, recorded in every artefact rather than left to
#: a reader who sees a finite, plausible-looking magnitude.
PROTEIN_REFERENCE_LIMITATION = (
    "the base model's protein mode is UNMEASURABLE on this cohort -- context "
    "information +0.084 nats per token and a reversal cost of -0.001 nats per "
    "residue, which is indifference to reading direction (EXP-R2-152). On the "
    "protein side the base checkpoint is therefore a PRE-ADAPTATION REFERENCE and "
    "not a behavioural control: a protein-mode magnitude from a cell built wholly "
    "or partly out of it may not be read as comparable with an adapted "
    "checkpoint's, and a difference between two such cells does not identify what "
    "a difference between two measurable modes would. The text mode carries no "
    "such caveat: all three checkpoints of the lineage are measurable in it. "
    "Whether the caveat applies to THIS cell is decided by which checkpoints "
    "--host and --donor named: the stage records their resolved paths and weight "
    "digests and infers no lineage role from them, and this cell's own protein "
    "verdict below is a measurement of the chimera rather than an inheritance"
)

#: What a swap can get wrong while still producing a complete artefact, and what
#: this stage does about each. Recorded because every one of them yields a model
#: that loads, runs and generates plausible text.
SWAP_LIMITATIONS = (
    "a moved tensor whose shape differs between donor and host would, if coerced, "
    "produce a model that loads, runs and generates plausible text while being "
    "meaningless (limitation L24). Shapes and dtypes are compared for every moved "
    "tensor before any write, and a mismatch raises",
    "an embedding row moved between two different vocabularies indexes a different "
    "symbol in each. The two tokenizers' vocabularies are compared by digest, not "
    "assumed identical, and a difference raises",
    "a swap between two checkpoints whose moved tensors are already equal changes "
    "nothing while reporting a chimera. Every moved tensor is compared with its "
    "target before the write and swap_changed_weights records the outcome",
    "the swap is a substitution of weights, not of training. It answers which "
    "COMPONENT the measured loss travels with; it does not say that the component "
    "is where the loss was caused, because the two groups were trained together "
    "and the body's LoRA updates were fitted against the embedding stage 1 was "
    "simultaneously moving",
)


@dataclass(frozen=True)
class VocabularyInterfaceLocation:
    """Where one loaded model's vocabulary interface is, and whether it is one tensor.

    ``tied`` is read off the *tensors* rather than off the config, for the reason
    every other fact in this repository's checkpoint records is read back from the
    loaded object: a config field states an intention and a shared storage is what
    a swap would actually write through.
    """

    input_prefix: str
    output_prefix: str
    tied: bool
    declared_tie_word_embeddings: bool | None

    @property
    def separable(self) -> bool:
        """Whether ``embedding`` and ``lm_head`` are distinct objects at all."""

        return not self.tied

    def record(self) -> dict[str, Any]:
        return {
            "input_embedding_module": self.input_prefix,
            "output_head_module": self.output_prefix,
            "tied": self.tied,
            "declared_tie_word_embeddings": self.declared_tie_word_embeddings,
            "declaration_matches_observation": (
                self.declared_tie_word_embeddings is None
                or bool(self.declared_tie_word_embeddings) == self.tied
            ),
            "separable_groups": (
                sorted(COMPONENT_GROUPS)
                if self.separable
                else sorted({VOCABULARY_INTERFACE, BODY})
            ),
            "note": (
                "tying is read from the tensors -- the input embedding's weight and "
                "the output head's weight are the same object or share storage -- "
                "and not from the config field, which states an intention rather "
                "than what a write would go through. Under tying the embedding and "
                "the lm_head are ONE tensor, so they cannot be moved separately: "
                "writing either writes both, and a run that reported them as two "
                "swaps would report two cells that are the same cell. Those two "
                "groups are refused by name and only vocabulary_interface and body "
                "remain available"
            ),
        }


def _module_prefix(model: Any, module: Any, *, role: str) -> str:
    """The state-dict prefix of one submodule, by identity rather than by name."""

    for name, candidate in model.named_modules():
        if candidate is module:
            if not name:
                raise ValueError(
                    f"this model's {role} is the model itself, so the tensors that "
                    "belong to it cannot be separated from the tensors that do not"
                )
            return name
    raise ValueError(
        f"this model's {role} is not one of its own submodules, so the state-dict "
        "names that belong to it cannot be located and no group can be declared "
        "over it"
    )


def locate_vocabulary_interface(model: Any) -> VocabularyInterfaceLocation:
    """The two modules the vocabulary interface consists of, and whether they are one.

    Both are located through the transformers interface -- ``get_input_embeddings``
    and ``get_output_embeddings`` -- rather than by spelling ``embed_tokens`` and
    ``lm_head``, so the declaration follows the loaded architecture instead of one
    family's naming.
    """

    embedding = model.get_input_embeddings()
    head = model.get_output_embeddings()
    if embedding is None or head is None:
        raise ValueError(
            "this model does not expose both an input embedding and an output head "
            "through get_input_embeddings/get_output_embeddings, so its vocabulary "
            "interface cannot be located and no component group can be declared "
            "over it"
        )
    for module, role in ((embedding, "input embedding"), (head, "output head")):
        if getattr(module, "weight", None) is None:
            raise ValueError(
                f"this model's {role} carries no weight tensor, so tying cannot be "
                "decided and a swap could not say what it moved"
            )
    tied = embedding is head or (
        embedding.weight is head.weight
        or embedding.weight.data_ptr() == head.weight.data_ptr()
    )
    declared = getattr(model.config, "tie_word_embeddings", None)
    return VocabularyInterfaceLocation(
        input_prefix=_module_prefix(model, embedding, role="input embedding"),
        output_prefix=_module_prefix(model, head, role="output head"),
        tied=bool(tied),
        declared_tie_word_embeddings=None if declared is None else bool(declared),
    )


def _under(names: tuple[str, ...], prefix: str) -> tuple[str, ...]:
    return tuple(name for name in names if name == prefix or name.startswith(prefix + "."))


def group_tensor_names(model: Any, group: str) -> tuple[str, ...]:
    """Exactly the state-dict names one declared group covers, or a refusal.

    Refuses in two ways, and both are by name rather than by silence: a group
    nobody declared, and a separable group on a checkpoint whose embedding and
    output head are one tied tensor. The second refusal is the honest answer to an
    inseparable pair -- moving "the embedding" there also moves the head, so a run
    that accepted it would report a cell it did not measure.
    """

    if group not in COMPONENT_GROUPS:
        raise ValueError(
            f"unknown component group {group!r}; the declared groups are "
            f"{sorted(COMPONENT_GROUPS)}. A component swap is an attribution only "
            "if what moved was decided in advance, so a group is declared here and "
            "never inferred from a name"
        )
    location = locate_vocabulary_interface(model)
    names = tuple(model.state_dict().keys())
    embedding_names = _under(names, location.input_prefix)
    head_names = _under(names, location.output_prefix)
    for found, role, prefix in (
        (embedding_names, "input embedding", location.input_prefix),
        (head_names, "output head", location.output_prefix),
    ):
        if not found:
            raise ValueError(
                f"the {role} module {prefix!r} contributes no tensor to this model's "
                "state dict, so the group it defines would be empty and the swap "
                "would report a move that never happened"
            )
    if location.tied and group in (EMBEDDING, LM_HEAD):
        raise ValueError(
            f"this checkpoint ties its input embedding {location.input_prefix!r} to "
            f"its output head {location.output_prefix!r}: they are one tensor, so "
            f"the group {group!r} cannot be moved without also moving the other. "
            f"Only {sorted({VOCABULARY_INTERFACE, BODY})} are separable here, and "
            "the tied pair travels as vocabulary_interface"
        )
    if group == EMBEDDING:
        return embedding_names
    if group == LM_HEAD:
        return head_names
    interface = tuple(sorted(set(embedding_names) | set(head_names)))
    if group == VOCABULARY_INTERFACE:
        return interface
    return tuple(name for name in names if name not in set(interface))


# ------------------------------------------------------ interchangeability

def vocabulary_digest(tokenizer: Any) -> str:
    """SHA-256 of a tokenizer's whole id-to-token map.

    The identity that a swapped embedding row depends on. Two checkpoints of one
    lineage are only interchangeable at the vocabulary interface if row *i* means
    the same symbol in both, and nothing about a mismatch is visible downstream:
    the chimera loads, runs and produces fluent text over the wrong alphabet.
    """

    vocabulary = tokenizer.get_vocab()
    digest = hashlib.sha256()
    for token, index in sorted(vocabulary.items(), key=lambda item: (item[1], item[0])):
        digest.update(f"{int(index)}\t{token}\n".encode("utf-8"))
    return digest.hexdigest()


def assert_same_vocabulary(host: Any, donor: Any) -> dict[str, Any]:
    """Refuse two checkpoints whose tokenizers are not the same vocabulary."""

    sizes = (int(len(host)), int(len(donor)))
    digests = (vocabulary_digest(host), vocabulary_digest(donor))
    if sizes[0] != sizes[1] or digests[0] != digests[1]:
        raise ValueError(
            "the host and donor tokenizers are not the same vocabulary "
            f"(sizes {sizes}, digests {digests[0][:12]}.. / {digests[1][:12]}..). "
            "An embedding row moved between two vocabularies indexes a different "
            "symbol in each, and the resulting model would load, run and generate "
            "plausible text while being meaningless. The ProLLaMA lineage carries "
            "one unmodified vocabulary at all three points with no added tokens, "
            "which is the premise this stage rests on -- it is checked, not assumed"
        )
    return {
        "verdict": "IDENTICAL",
        "vocabulary_size": sizes[0],
        "vocabulary_sha256": digests[0],
        "host_tokenizer_class": type(host).__name__,
        "donor_tokenizer_class": type(donor).__name__,
        "note": (
            "the full id-to-token map of both tokenizers, digested and compared. "
            "This is the premise that makes two checkpoints' embedding rows "
            "interchangeable at all"
        ),
    }


#: Config facts that must agree before any tensor is moved. Read back from each
#: loaded model by ``21_joint_mode_qualification.load_model``, never echoed from a
#: request.
INTERCHANGEABLE_FACTS = (
    "model_type",
    "architectures",
    "n_layers",
    "d_model",
    "n_heads",
    "vocab_size",
    "max_position_embeddings",
    "dtype_observed",
)


def assert_interchangeable(
    host: Any, donor: Any, *, host_facts: dict[str, Any], donor_facts: dict[str, Any]
) -> dict[str, Any]:
    """Refuse two models whose tensors are not interchangeable, before any is moved.

    Three refusals, in the order that costs least to discover: the declared shape,
    the state-dict key set, and the tying status. The last is its own refusal
    rather than a consequence of the second, because a tied host and an untied
    donor carry the same keys and mean different things by them -- the donor's two
    tensors would both be written into the host's one.
    """

    differing = {
        name: [host_facts.get(name), donor_facts.get(name)]
        for name in INTERCHANGEABLE_FACTS
        if host_facts.get(name) != donor_facts.get(name)
    }
    if differing:
        raise ValueError(
            f"host and donor disagree on {sorted(differing)}: {differing}. A "
            "component swap is defined only between checkpoints of one architecture "
            "and one shape, so there is nothing here to interchange"
        )
    host_state, donor_state = host.state_dict(), donor.state_dict()
    only_host = sorted(set(host_state) - set(donor_state))
    only_donor = sorted(set(donor_state) - set(host_state))
    if only_host or only_donor:
        raise ValueError(
            f"the two state dicts do not carry the same tensors: {len(only_host)} "
            f"only in the host ({only_host[:4]}), {len(only_donor)} only in the "
            f"donor ({only_donor[:4]}). Swapping a group between them would move a "
            "different set of tensors in each direction"
        )
    host_location = locate_vocabulary_interface(host)
    donor_location = locate_vocabulary_interface(donor)
    if (host_location.input_prefix, host_location.output_prefix) != (
        donor_location.input_prefix,
        donor_location.output_prefix,
    ):
        raise ValueError(
            "the host and donor locate their vocabulary interface at different "
            f"modules ({host_location.input_prefix}/{host_location.output_prefix} "
            f"against {donor_location.input_prefix}/{donor_location.output_prefix}), "
            "so the same group name would name different tensors in each"
        )
    if host_location.tied != donor_location.tied:
        raise ValueError(
            f"the host ties its vocabulary interface: {host_location.tied}; the "
            f"donor: {donor_location.tied}. One of them holds the embedding and the "
            "output head as a single tensor and the other as two, so a group moved "
            "between them is not the same group and the write would silently "
            "collapse or duplicate a component"
        )
    return {
        "verdict": "INTERCHANGEABLE",
        "n_tensors": len(host_state),
        "shared_facts": {name: host_facts.get(name) for name in INTERCHANGEABLE_FACTS},
        "host_vocabulary_interface": host_location.record(),
        "donor_vocabulary_interface": donor_location.record(),
        "note": (
            "the declared shape, the full state-dict key set and the tying status "
            "all agree, which is what makes these two checkpoints interchangeable "
            "tensor for tensor. Per-tensor shapes and dtypes are compared again for "
            "every tensor actually moved"
        ),
    }


# ------------------------------------------------------------------- the swap


@torch.no_grad()
def swap_component(host: Any, donor: Any, group: str) -> dict[str, Any]:
    """Write the donor's copy of one declared group into the host, in place.

    The values are **copied**, never aliased. Assigning the donor's tensor objects
    into the host would leave the chimera sharing storage with a model the caller
    is about to release, so "everything else from the host" would hold only until
    something touched the donor.

    Every moved tensor's shape and dtype are compared before the first write, so a
    mismatch anywhere in the group stops the run with the host still intact rather
    than half-written. Under tying the same storage appears under two names; it is
    written once and both names are still reported, because what was *declared* to
    move is what the artefact has to say.
    """

    moved = group_tensor_names(host, group)
    host_state, donor_state = host.state_dict(), donor.state_dict()
    for name in moved:
        if name not in donor_state:
            raise ValueError(
                f"the donor carries no tensor named {name!r}, so the group {group!r} "
                "cannot be moved from it"
            )
        target, source = host_state[name], donor_state[name]
        if tuple(target.shape) != tuple(source.shape):
            raise ValueError(
                f"{name}: host shape {tuple(target.shape)} against donor shape "
                f"{tuple(source.shape)}. A component swap moves tensors between two "
                "checkpoints of one architecture; coercing a shape here would "
                "produce a model that loads, runs and generates plausible text "
                "while being meaningless (limitation L24), so it is refused"
            )
        if target.dtype != source.dtype:
            raise ValueError(
                f"{name}: host dtype {target.dtype} against donor dtype "
                f"{source.dtype}. Both checkpoints are loaded at the declared "
                "--dtype, so a difference means one of them did not honour it and "
                "the moved values would be silently requantised"
            )

    written: set[int] = set()
    identical: list[str] = []
    for name in moved:
        target = host_state[name]
        storage = target.data_ptr()
        if storage in written:
            continue
        written.add(storage)
        source = donor_state[name].to(device=target.device)
        if torch.equal(target, source):
            identical.append(name)
        target.copy_(source)

    kept = tuple(name for name in host_state if name not in set(moved))
    return {
        "component_group": group,
        "component_group_definition": COMPONENT_GROUPS[group],
        "declared_component_groups": sorted(COMPONENT_GROUPS),
        "tensors_moved": list(moved),
        "n_tensors_moved": len(moved),
        "n_distinct_tensors_written": len(written),
        "n_tensors_kept_from_host": len(kept),
        "tensors_identical_before_swap": identical,
        "swap_changed_weights": len(identical) < len(written),
        "verification": (
            "every moved tensor's shape and dtype were compared in donor and host "
            "before the first write, so a mismatch stops the run with the host "
            "intact; values are copied rather than aliased, so the chimera shares "
            "no storage with the donor"
        ),
        "identity_note": (
            "tensors_identical_before_swap names the moved tensors whose donor and "
            "host copies were already equal. When that covers the whole group the "
            "swap changed nothing: either the donor IS the host -- the identity "
            "anchor -- or the group was never retrained, and in both cases the cell "
            "is not the chimera its name suggests"
        ),
    }


# ----------------------------------------------------------------- the artefact


def checkpoint_record(
    resolved: Path, requested: Path, facts: dict[str, Any], *, role: str
) -> dict[str, Any]:
    """One end of the swap, identified by path and by the bytes on disk."""

    record = dict(facts)
    record.update(
        {
            "role": role,
            "requested_path": str(requested),
            "name": resolved.name,
            "weights_sha256": checkpoint_weights_digest(resolved),
        }
    )
    return record


def artefact_name(host: Path, donor: Path, group: str) -> str:
    """A file name that says which cell this is.

    Derived rather than fixed, because the four cells that answer the question
    differ only in their host, their donor and their group: a constant name would
    let the second run of a campaign silently overwrite the first inside one
    ``--out`` directory.
    """

    def safe(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "unnamed"

    return f"component_swap__host-{safe(host.name)}__{safe(group)}-from-{safe(donor.name)}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        type=Path,
        required=True,
        help="directory of the checkpoint that keeps everything OUTSIDE the "
        "component group. A path and not an arm name, for the reason "
        "21_joint_mode_qualification.py gives: a checkpoint that has not passed "
        "that stage must not be in the panel",
    )
    parser.add_argument(
        "--donor",
        type=Path,
        required=True,
        help="directory of the checkpoint the component group is taken FROM. Naming "
        "the host again is the reference cell: it measures that checkpoint "
        "unmodified and proves the swap machinery is an identity in the same run",
    )
    parser.add_argument(
        "--component-group",
        required=True,
        choices=tuple(COMPONENT_GROUPS),
        help="which declared group moves from the donor into the host. "
        "vocabulary_interface is the group the ProLLaMA lineage's stage 1 lists in "
        "modules_to_save, and body is its exact complement",
    )
    parser.add_argument(
        "--rendering",
        required=True,
        choices=joint_modes.RENDERING_NAMES,
        help="which declared family's input format these checkpoints take. The set "
        "is composed by src.transfer.joint_modes, the single place either mode's "
        "format is decided (Appendix B rule 12)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0", help="device the chimera is scored on")
    parser.add_argument(
        "--donor-device",
        default="cpu",
        help="device the donor is loaded on. The donor is a source of tensors and "
        "never runs a forward pass, so it does not need the accelerator; keeping it "
        "off the device halves the peak memory a swap between two 7B checkpoints "
        "needs. A cross-device copy is what moves the values",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=tuple(STAGE21._DTYPES),
        help="inference dtype for BOTH checkpoints, read back from the loaded "
        "parameters. One dtype, because a swap between two precisions would "
        "requantise the moved values",
    )
    parser.add_argument(
        "--modes",
        default="both",
        choices=("text", "protein", "both"),
        help="which modes to measure on the chimera. The text mode is what the "
        "attribution question is about; the protein mode is measured beside it and "
        "carries the pre-adaptation limitation the artefact records",
    )
    parser.add_argument("--sequences", type=int, default=64)
    parser.add_argument(
        "--unigram-sequences",
        type=int,
        default=400,
        help="records the held-out context-free baseline is fitted on, drawn past "
        "the scored cohort in the same permutation and then deduplicated by content",
    )
    parser.add_argument("--protein-min-len", type=int, default=64)
    parser.add_argument("--protein-max-len", type=int, default=246)
    parser.add_argument(
        "--text-min-chars",
        type=int,
        default=800,
        help="floor of the text cohort, in characters; src.transfer.arms.text_cohort's "
        "own default, so the population is the one every other text measurement in "
        "this repository uses",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="scored window for text. A protein rendering is never truncated: the "
        "stage raises instead, because dropping the closing delimiter would silently "
        "change the scored span",
    )
    parser.add_argument(
        "--protein-context",
        default=None,
        help="optional document context the protein block is embedded in, filled "
        "into the family's declared template. Omitted means the bare block, and "
        "whichever was used reaches the artefact",
    )
    parser.add_argument(
        "--cohort-draw-seed",
        type=int,
        default=DEFAULT_CORPUS_DRAW_SEED,
        help="seed for the permutation both draws are windows of; 0 selects the "
        "historical file-order draw, which is a declared choice and not a default "
        "(Appendix B rule 1)",
    )
    parser.add_argument(
        "--min-context-information",
        type=float,
        default=MIN_CONTEXT_INFORMATION_NATS,
        help="the threshold below which a mode is reported unmeasurable on this "
        "cohort. Stage 01's threshold, imported through "
        "21_joint_mode_qualification.py, so a chimera is read against the level the "
        "panel and the lineage were qualified against",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    modes = ("protein", "text") if args.modes == "both" else (args.modes,)

    declaration = joint_modes.rendering(args.rendering)
    # The tokenizers and the rendering first, so a wrong checkpoint/family pair or
    # two different vocabularies fail before two multi-gigabyte loads.
    host_path, host_tokenizer = STAGE21.load_tokenizer(args.host)
    donor_path, donor_tokenizer = STAGE21.load_tokenizer(args.donor)
    vocabulary = assert_same_vocabulary(host_tokenizer, donor_tokenizer)
    tokenisation = joint_modes.resolve(host_tokenizer, declaration)
    print(
        f"[load] host {host_path} on {args.device}, donor {donor_path} on "
        f"{args.donor_device}, as {declaration.name}"
    )

    host_model, host_facts = STAGE21.load_model(
        host_path, host_tokenizer, device=args.device, dtype=args.dtype
    )
    donor_model, donor_facts = STAGE21.load_model(
        donor_path, donor_tokenizer, device=args.donor_device, dtype=args.dtype
    )
    host = checkpoint_record(host_path, args.host, host_facts, role="host")
    donor = checkpoint_record(donor_path, args.donor, donor_facts, role="donor")
    interchange = assert_interchangeable(
        host_model, donor_model, host_facts=host_facts, donor_facts=donor_facts
    )
    print(
        f"  {host_facts['n_layers']}L x {host_facts['d_model']}d x "
        f"{host_facts['n_heads']}h, vocab {host_facts['vocab_size']}; "
        f"{interchange['n_tensors']} tensors, tied "
        f"{interchange['host_vocabulary_interface']['tied']}"
    )

    chimera = swap_component(host_model, donor_model, args.component_group)
    reference_cell = host["weights_sha256"] == donor["weights_sha256"]
    if reference_cell and chimera["swap_changed_weights"]:
        raise RuntimeError(
            "the host and donor weight files are byte-identical, so the swap must "
            "have been an identity, and it changed "
            f"{chimera['n_distinct_tensors_written'] - len(chimera['tensors_identical_before_swap'])} "
            "tensors. Either the loader is not deterministic on this checkpoint -- "
            "the L24 shape, where part of a model is newly initialised at load -- or "
            "the swap moved something other than what it named"
        )
    del donor_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        f"[swap] {chimera['component_group']}: {chimera['n_tensors_moved']} tensors "
        f"from the donor, {chimera['n_tensors_kept_from_host']} kept from the host; "
        f"changed weights: {chimera['swap_changed_weights']}"
    )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "provenance": {
            "runner": {
                "path": "scripts/transfer/24_component_swap.py",
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "modules": {name: sha256_file(REPO_ROOT / name) for name in PROVENANCE_MODULES},
        },
        "estimand": (
            "context information per mode, imported unchanged from "
            "21_joint_mode_qualification.py: held-out unigram cross-entropy on the "
            "scored symbols minus the chimera's clean cross-entropy on the same "
            "symbols, with that stage's cohort draw, held-out reference, controls "
            "and measurability threshold. A chimera's number is only readable "
            "beside the qualification figures of the checkpoints it was built from, "
            "which is why the estimand is imported rather than restated"
        ),
        "cell": {
            "composition": (
                f"{host_path.name} supplies every tensor outside "
                f"{args.component_group}; {donor_path.name} supplies "
                f"{args.component_group}"
            ),
            "is_reference_cell": reference_cell,
            "reference_cell_note": (
                "the host and donor weight files are byte-identical, so this run "
                "measures that checkpoint unmodified AND proves the swap machinery "
                "is an identity on it"
                if reference_cell
                else "the host and donor are different checkpoints, so this run "
                "measures a chimera"
            ),
        },
        "host": host,
        "donor": donor,
        "tokenizer_vocabulary": vocabulary,
        "interchangeability": interchange,
        "chimera": chimera,
        "rendering": tokenisation.facts(),
        "seeds": {"cohort_draw": int(args.cohort_draw_seed)},
        "thresholds": {
            "minimum_context_information_nats": float(args.min_context_information)
        },
        "limitations": {
            "protein_mode_reference": PROTEIN_REFERENCE_LIMITATION,
            "swap": list(SWAP_LIMITATIONS),
        },
    }

    modes_record: dict[str, Any] = {}
    if "protein" in modes:
        modes_record["protein"] = STAGE21.protein_mode(args, host_model, tokenisation)
        modes_record["protein"]["reference_limitation"] = PROTEIN_REFERENCE_LIMITATION
    if "text" in modes:
        modes_record["text"] = STAGE21.text_mode(
            args,
            host_model,
            host_tokenizer,
            tokenisation,
            vocab_size=int(host_facts["vocab_size"]),
        )
    payload["modes"] = modes_record
    payload["verdicts"] = {name: record["verdict"] for name, record in modes_record.items()}
    payload["verdict_note"] = VERDICT_NOTE
    payload["modes_measured"] = list(modes)

    destination = args.out / artefact_name(host_path, donor_path, args.component_group)
    write_json(destination, payload)
    print()
    for name, record in modes_record.items():
        print(
            f"[{name}] context information {record['context_information_nats']:+.4f} "
            f"({record['context_information_unit']})  {record['verdict']}"
        )
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
