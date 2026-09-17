#!/usr/bin/env python3
"""Does InstructProtein's native family instruction move its generations toward the requested family?

A feasibility probe, not a campaign, and not a widening of EXP-R2-227. The
checkpoint's native conditioning interface was established from primary sources
before this script existed: the paper's Table 5 carries the forward instruction
``Can you provide me with a protein belonging to the secretoglobin family?
Output: Sure, here's a protein from the secretoglobin family: {protein}.`` and a
domain form, and its Appendix A.4 applies the label-generic form

    Instruction: I would like a protein that is in {label}. Output: One of the protein that meets the demand is {protein}

to the Fold, Family and Superfamily label spaces. So a **family/domain name to
protein sequence** condition exists in the released instruction data, and the
question that follows is whether a request written in that form moves this
checkpoint's generations.

The checkpoint emits text positions badly (EXP-R2-151: 92.8% of its probability
mass on the twenty residue tokens at every text position, a clean text NLL of
19.08 against ln(50304) = 10.83). That is a readout failure at text positions.
Here the instruction is **supplied by us and never sampled**: the prompt is the
checkpoint's declared instruction scaffold, and only the protein span is
generated.

One class, one contrast, n = 200 per cell:

* **requested** -- the prompt names class A and the generations are scored for A;
* **mismatched** -- the prompt names class B and the same generations are scored
  for A.

Both cells are one arm, one sampling configuration and one oracle. The oracle is
EXP-R2-227's own: HMMER 3.4 against the pressed Pfam-A release at the release's
gathering thresholds, with the class's Pfam referent derived exactly as that
campaign derives it -- from a disjoint referent draw of the class's own real
Swiss-Prot exemplars, never from the anchor draw and never from a generation.
The per-class instrument anchor (real exemplars against length-matched random
UniRef50) runs **before any generation is scored**, as that campaign requires.

A low hit rate is only readable if the oracle recognises the class at all, which
is what the anchor prices. The class's exemplars are identified by the staged
`protein2ipr`-derived Pfam span table, so the anchor's real side partly
re-establishes the labelling channel rather than pricing class specificity from
an independent channel; that limitation is recorded per class and carried in the
report rather than papered over.

Stages, in order:

``--stage class``      the eligible family census and the seeded draw of the
                       requested and mismatched classes. No GPU, no oracle.
``--stage anchor``     the per-class instrument price, through HMMER and Pfam-A.
``--stage generate``   the only GPU stage: 200 + 200 samples at one frozen
                       configuration, with the generated ids kept.
``--stage score``      the oracle on the generations, the two rates, their
                       difference with an interval, and the residue accounting.

The oracle is staged host-locally and is not in the repository, so it is built
under ``--work`` unless it is already there: if ``--work/hmmer`` and
``--work/pfam`` already hold a built HMMER and a pressed Pfam-A -- a symlink to
an existing build is enough -- nothing is rebuilt. The repository's own
``conditioned_generation`` module supplies the classifier: the class draw, the
referent derivation, the anchor record, the near-duplicate grouping, the rate
contrast and the frozen sampling constants are all imported from it rather than
restated here.

Re-run:

    python scripts/transfer/instructprotein_family_generation_probe.py --stage class
    python scripts/transfer/instructprotein_family_generation_probe.py --stage anchor
    python scripts/transfer/instructprotein_family_generation_probe.py --stage generate --device cuda:4 --dtype bfloat16
    python scripts/transfer/instructprotein_family_generation_probe.py --stage score
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from src.transfer import concept_injection as ci  # noqa: E402
from src.transfer import conditioned_generation as cg  # noqa: E402
from src.transfer import joint_modes as jm  # noqa: E402
from src.transfer.arms import REPO, SWISSPROT_FASTA, UNIREF50_FASTA  # noqa: E402
from src.transfer.channels import PFAM_RESIDUE_TSV, load_pfam_spans  # noqa: E402
from src.transfer.instructprotein_fitness import (  # noqa: E402
    INSTRUCTPROTEIN_ARM,
    load_instructprotein,
)
from src.transfer.io import write_json  # noqa: E402

PROBE = "instructprotein_family_generation_probe"

DEFAULT_OUT = REPO / "results/transfer/instructprotein_family_generation_probe"
DEFAULT_WORK = REPO.parent / "work/instructprotein_family_generation_probe"

#: The probe's one seed: the class draw, the referent and anchor draws, the
#: length-matched random side, the per-cell sampling seeds and the bootstrap.
#: Declared once and never moved after a rate exists.
PROBE_SEED = 20260916

#: One requested class and one mismatched class, which is the whole contrast.
CLASSES = 2

#: Generations per cell. EXP-R2-227's own sample size, and the only prior
#: generation measurement in the repository is at the same n.
GENERATIONS_PER_CELL = 200

#: UniRef50 records held for the length-matched random side of the anchor. The
#: campaign's own default; 100 targets at a 10% length tolerance have no use for
#: more, and the scan is over the whole corpus either way.
RESERVOIR = 200_000

#: The paper's label-generic forward instruction, verbatim from Appendix A.4:
#: ``Instruction: I would like a protein that is in {label}. Output: One of the
#: protein that meets the demand is {protein}``. The declared rendering supplies
#: the ``Instruction:``/``Output:`` scaffold around it, so only the sentence is
#: spelled here; the label is the Pfam family's own released description.
INSTRUCTION_TEMPLATE = "I would like a protein that is in {label}."

#: The oracle's own thresholds, imported from the campaign rather than chosen
#: here: Pfam-A at its curated per-family gathering thresholds.
PFAM_THRESHOLD = cg.PFAM_THRESHOLD

#: Pfam's own release, so a family name is the release's and not a paraphrase.
PFAM_ARCHIVE = REPO / "external_resources/ec_metrics/pfam/Pfam-A.hmm.gz"
PFAM_CHECKSUM = REPO / "external_resources/ec_metrics/pfam/Pfam-A.hmm.gz.sha256"
HMMER_TARBALL = REPO / "external_resources/tools/hmmer-3.4.tar.gz"

HMMER_BINARY = "bin/hmmscan"


# ----------------------------------------------------------------- primitives


def residue_run(generated_ids: Sequence[int], residue_letters: Mapping[int, str]) -> str:
    """The leading run of declared residue ids, as residues.

    InstructProtein's rendering declares one dedicated token per residue and
    identifies its residues by *identity* rather than by position, so the scored
    span is decidable from the ids themselves. A generation that leaves the
    residue subspace at its first token yields the empty string, which is a
    genuine failure of the interface and stays in the denominator.
    """

    residues: list[str] = []
    for value in generated_ids:
        letter = residue_letters.get(int(value))
        if letter is None:
            break
        residues.append(letter)
    return "".join(residues)


def prompt_for(declaration: jm.JointRendering, label: str) -> str:
    """The string this checkpoint is fed to produce one sample.

    The declared context template with the instruction in it, then the declared
    protein start delimiter -- the same shape ``conditioned_generation.prompt_for``
    gives ProLLaMA's conditioned arm, so the sampled span begins in the residue
    subspace rather than at a delimiter the model would have to emit.
    """

    instruction = INSTRUCTION_TEMPLATE.format(label=label)
    if declaration.protein_context_template is None:
        raise ValueError(f"{declaration.name} declares no context template")
    return (
        declaration.protein_context_template.format(context=instruction)
        + declaration.protein_start
    )


def eligible_families(
    census: Mapping[str, set[str]], labels: Mapping[str, tuple[str, str]], *, minimum: int
) -> tuple[str, ...]:
    """The canonically ordered admissible class list.

    A class is admissible when the staged span table carries at least
    ``minimum`` Swiss-Prot records for it -- a 100-record referent draw and a
    disjoint 100-record anchor draw is what a class is priced on -- and the Pfam
    release carries a name for it, because the instruction names a family and a
    name nothing publishes is not one.
    """

    admissible = [
        key for key, accessions in census.items() if len(accessions) >= minimum and key in labels
    ]
    return tuple(sorted(admissible))


def class_draw(candidates: Sequence[str], *, seed: int, n: int) -> tuple[str, str]:
    """The requested class and the mismatched class, from one seeded permutation.

    ``seeded_draw`` is the campaign's own rule: a permutation of the canonically
    ordered admissible list, never a prefix of a frequency table and never the
    head of the file. The mismatched class is the next entry of the same
    permutation, which is the two-class case of that campaign's fixed-point-free
    pairing.
    """

    drawn = cg.seeded_draw(candidates, n=n, seed=seed)
    return drawn[0], drawn[1]


# --------------------------------------------------------------------- stages


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _preamble(artifact: str) -> dict[str, Any]:
    return {
        "probe": PROBE,
        "artifact": artifact,
        "created_utc": _timestamp(),
        "is_a_feasibility_probe": (
            "one class, one contrast, n = 200 per cell. This is not EXP-R2-227, it "
            "does not touch that campaign's class queue, compound or results, and no "
            "campaign verdict is issued from it"
        ),
        "readout_failure_is_not_this_estimand": (
            "InstructProtein's text positions are unmeasurable (EXP-R2-151: 92.8% of "
            "the probability mass on the twenty residue tokens at every text position; "
            "clean text NLL 19.08 against ln(50304) = 10.83). This probe supplies the "
            "instruction instead of sampling it and samples only the protein span, so "
            "that readout failure is not the quantity under test"
        ),
    }


def pfam_labels(path: Path) -> dict[str, tuple[str, str]]:
    """Unversioned Pfam accession to ``(NAME, DESC)``, from the release's own file."""

    if not Path(path).is_file():
        raise FileNotFoundError(f"{path} does not exist; the family names come from the release")
    labels: dict[str, tuple[str, str]] = {}
    accession = name = None
    desc = ""
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("NAME  "):
                name = line[6:].strip()
            elif line.startswith("ACC   "):
                accession = line[6:].strip().split(".", 1)[0]
            elif line.startswith("DESC  "):
                desc = line[6:].strip()
            elif line.startswith("HMMER3"):
                if accession is not None:
                    labels[accession] = (name or "", desc)
                accession = name = None
                desc = ""
    if not labels:
        raise RuntimeError(f"{path} carries no profile")
    return labels


def family_census(path: Path) -> dict[str, set[str]]:
    """Accessions per Pfam family, from the staged span table."""

    spans = load_pfam_spans(Path(path))
    census: dict[str, set[str]] = {}
    for accession, entries in spans.items():
        for _, _, family in entries:
            census.setdefault(family, set()).add(accession)
    return census


def class_pool(accessions: Sequence[str], *, path: Path) -> list[str]:
    """The class's real Swiss-Prot sequences, in canonical accession order."""

    sequences = cg.swissprot_sequences(accessions, path=path)
    return [sequences[accession] for accession in sorted(accessions) if accession in sequences]


def run_class(args: argparse.Namespace) -> dict[str, Any]:
    labels = pfam_labels(args.pfam_hmm)
    census = family_census(args.pfam_residue)
    candidates = eligible_families(census, labels, minimum=cg.MIN_CLASS_RECORDS)
    requested, mismatched = class_draw(candidates, seed=args.draw_seed, n=CLASSES)
    declaration = jm.rendering("instructprotein")
    classes = {
        role: {
            "key": key,
            "pfam_name": labels[key][0],
            "label": labels[key][1] or labels[key][0],
            "n_swissprot_records": len(census[key]),
            "prompt": prompt_for(declaration, labels[key][1] or labels[key][0]),
        }
        for role, key in (("requested", requested), ("mismatched", mismatched))
    }
    payload = _preamble("instructprotein_family_generation_class")
    payload.update(
        {
            "draw_seed": int(args.draw_seed),
            "draw": {
                "rule": (
                    "seeded permutation of the canonically ordered admissible family list "
                    "(conditioned_generation.seeded_draw); the mismatched class is the "
                    "next entry of the same permutation"
                ),
                "classes": CLASSES,
            },
            "eligibility": {
                "staged_span_table": str(args.pfam_residue),
                "staged_pfam_release": str(args.pfam_hmm),
                "minimum_records": int(cg.MIN_CLASS_RECORDS),
                "minimum_records_reason": (
                    "a class needs a 100-record referent draw and a disjoint 100-record "
                    "anchor draw; a referent fitted on the sequences the anchor is "
                    "priced on would return a real-side rate of one whatever the oracle "
                    "does"
                ),
                "families_in_the_span_table": len(census),
                "families_with_a_name_in_the_release": len(
                    [key for key in census if key in labels]
                ),
                "admissible_families": len(candidates),
            },
            "instruction_template": INSTRUCTION_TEMPLATE,
            "instruction_source": (
                "Appendix A.4's label-generic forward template, applied there to Fold, "
                "Family and Superfamily labels; Table 5 carries the same direction as "
                "Family Generation and Domain Generation"
            ),
            "prompt_source": (
                "joint_modes' declared rendering for this checkpoint: "
                "'Instruction: {context}\\nOutput: ' then the declared protein start "
                "delimiter"
            ),
            "classes": classes,
        }
    )
    return payload


def _reservoir(args: argparse.Namespace, band: tuple[int, int]) -> list[str]:
    """A cached length band of UniRef50, drawn once under this probe's seed."""

    path = args.work / f"uniref50_reservoir_{band[0]}_{band[1]}_{args.reservoir}_{args.draw_seed}.json"
    if path.is_file():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if tuple(cached["band"]) != band or int(cached["seed"]) != int(args.draw_seed):
            raise RuntimeError(
                f"{path} was drawn at band {cached['band']} under seed {cached['seed']}, "
                f"not at {list(band)} under {args.draw_seed}"
            )
        return cached["sequences"]
    sequences = cg.uniref50_pool(
        path=args.uniref50,
        size=args.reservoir,
        seed=args.draw_seed,
        min_len=band[0],
        max_len=band[1],
    )
    write_json(path, {"sequences": sequences, "seed": int(args.draw_seed), "band": list(band)})
    return sequences


def _oracle(args: argparse.Namespace) -> tuple[Any, Any]:
    tool = ci.prepare_hmmer(HMMER_TARBALL, args.work / "hmmer")
    database = ci.prepare_pfam(PFAM_ARCHIVE, PFAM_CHECKSUM, args.work / "pfam", tool=tool)
    return tool, database


def run_anchor(args: argparse.Namespace) -> dict[str, Any]:
    payload = _read(args.class_artifact)
    request = payload["classes"]["requested"]
    key = request["key"]
    pool = class_pool(sorted(family_census(args.pfam_residue)[key]), path=args.swissprot)
    if len(pool) < cg.MIN_CLASS_RECORDS:
        raise RuntimeError(
            f"{key} yielded {len(pool)} sequences from the staged Swiss-Prot file, below "
            f"the {cg.MIN_CLASS_RECORDS} a disjoint referent and anchor draw need"
        )
    referent_draw, real_draw = cg.split_draw(pool, seed=args.draw_seed)
    lengths = [len(sequence) for sequence in real_draw]
    band = (
        max(1, int(min(lengths) * (1.0 - cg.LENGTH_MATCH_TOLERANCE))),
        int(max(lengths) * (1.0 + cg.LENGTH_MATCH_TOLERANCE)) + 1,
    )
    reservoir = _reservoir(args, band)
    random_draw = cg.length_matched(real_draw, reservoir, seed=args.draw_seed)

    named: dict[str, str] = {}
    for kind, sequences in (("referent", referent_draw), ("real", real_draw), ("random", random_draw)):
        for index, sequence in enumerate(sequences):
            named[f"{key}|{kind}|{index}"] = sequence

    tool, database = _oracle(args)
    hits, scan = cg.annotate(
        named,
        tool=tool,
        database=database,
        workspace=args.work / "anchor",
        threads=args.hmmscan_threads,
        shards=args.hmmscan_shards,
        label="instructprotein_probe_anchor",
    )
    referent = cg.referent_from_draw(hits, [f"{key}|referent|{i}" for i in range(len(referent_draw))])
    record = {"key": key, "label": request["label"]}
    record.update(
        cg.anchor_record(
            real=cg.assigned(hits, [f"{key}|real|{i}" for i in range(len(real_draw))], referent),
            random=cg.assigned(hits, [f"{key}|random|{i}" for i in range(len(random_draw))], referent),
            referent=referent,
        )
    )
    record["referent_derivation"] = (
        "Pfam families carried by at least "
        f"{cg.REFERENT_FAMILY_SHARE:.2f} of a disjoint {len(referent_draw)}-record draw of the "
        "class's own real exemplars, by this same HMMER/Pfam-A oracle. The exemplars are "
        "identified by the staged protein2ipr-derived Pfam span table, so this real side "
        "re-establishes the labelling channel rather than pricing class specificity from "
        "an independent one; the random side is the part that prices the oracle's "
        "willingness to fire on proteins of the class's length that are not of its class"
    )
    result = _preamble("instructprotein_family_generation_anchor")
    result.update(
        {
            "draw_seed": int(args.draw_seed),
            "class": record,
            "class_admitted": bool(record["admitted"]),
            "draws": {
                "referent": len(referent_draw),
                "real": len(real_draw),
                "random": len(random_draw),
                "real_length_band": [min(lengths), max(lengths)],
            },
            "random_side": {
                "corpus": str(args.uniref50),
                "reservoir_size": int(args.reservoir),
                "reservoir_band": list(band),
                "tolerance": cg.LENGTH_MATCH_TOLERANCE,
                "note": (
                    "real length-matched UniRef50 proteins, which is what EXP-R2-015's "
                    "control was; a synthetic residue string would fail every profile for "
                    "reasons that have nothing to do with class specificity"
                ),
            },
            "hmmscan": scan,
            "pfam_threshold": PFAM_THRESHOLD,
        }
    )
    return result


def _sample(
    model: Any, tokenizer: Any, prompt: str, *, n: int, seed: int, batch_size: int
) -> list[tuple[str, list[int], int]]:
    """``n`` sampled continuations with the generated ids kept.

    ``conditioned_generation.sample_continuations`` is the declared door and this
    is its sampling configuration, imported from that module's own constants so
    the two cannot drift. It is re-spelled here for one reason: that function
    returns decoded text only, and the residue accounting this probe owes -- how
    many generated tokens are residue tokens -- needs the ids, which the decode
    does not carry back.

    Returns ``(decoded text with specials kept, generated ids, prompt length)``.
    """

    import torch

    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    ids = encoded["input_ids"].to(model.device)
    prompt_length = int(ids.shape[1])
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    rows: list[tuple[str, list[int], int]] = []
    index = 0
    with torch.no_grad():
        while len(rows) < n:
            size = min(batch_size, n - len(rows))
            torch.manual_seed(seed + index)
            generated = model.generate(
                input_ids=ids.repeat(size, 1),
                attention_mask=torch.ones((size, prompt_length), dtype=torch.long, device=ids.device),
                do_sample=True,
                temperature=cg.TEMPERATURE,
                top_p=cg.TOP_P,
                top_k=cg.TOP_K,
                repetition_penalty=cg.REPETITION_PENALTY,
                max_new_tokens=cg.MAX_NEW_TOKENS,
                pad_token_id=pad,
                use_cache=True,
            )
            for row in generated:
                tail = [int(value) for value in row[prompt_length:]]
                rows.append((tokenizer.decode(tail, skip_special_tokens=False), tail, prompt_length))
            index += 1
    return rows[:n]


def run_generate(args: argparse.Namespace) -> dict[str, Any]:
    class_payload = _read(args.class_artifact)
    loaded = load_instructprotein(INSTRUCTPROTEIN_ARM, device=args.device, dtype=args.dtype)
    declaration = loaded.tokenisation.declaration
    residue_letters = {int(value): residue for residue, value in loaded.tokenisation.residue_ids.items()}
    end_id = int(loaded.tokenizer.convert_tokens_to_ids(declaration.protein_end))
    if loaded.tokenizer.convert_ids_to_tokens(end_id) != declaration.protein_end:
        raise RuntimeError(
            f"{declaration.protein_end!r} does not round-trip through this tokenizer; the "
            "terminator this probe counts cannot be identified"
        )
    prefix_ids = jm.prefix_marker_ids(loaded.tokenizer, declaration)

    cells: dict[str, Any] = {}
    for role, block in class_payload["classes"].items():
        prompt = block["prompt"]
        observed_prefix = tuple(jm.encode(loaded.tokenizer, prompt)[: len(prefix_ids)])
        if observed_prefix != prefix_ids:
            raise RuntimeError(
                f"this prompt encodes with prefix {observed_prefix} where the rendering "
                f"declares {prefix_ids}; the declared rendering is not the one under test"
            )
        seed = cg.cell_seed(
            seed=args.sampling_seed, arm_name=INSTRUCTPROTEIN_ARM, class_key=block["key"], condition=role
        )
        rows = _sample(
            loaded.model,
            loaded.tokenizer,
            prompt,
            n=args.generations,
            seed=seed,
            batch_size=args.batch_size,
        )
        samples = [residue_run(ids, residue_letters) for _, ids, _ in rows]
        generated_tokens = sum(len(ids) for _, ids, _ in rows)
        residue_tokens = sum(
            sum(1 for value in ids if value in residue_letters) for _, ids, _ in rows
        )
        terminated = sum(1 for _, ids, _ in rows if end_id in ids)
        cells[role] = {
            "key": block["key"],
            "label": block["label"],
            "prompt": prompt,
            "seed": int(seed),
            "samples": samples,
            "raw": [text for text, _, _ in rows],
            "statistics": {
                "n": len(samples),
                "n_empty": int(sum(1 for sample in samples if not sample)),
                "mean_length": float(np.mean([len(sample) for sample in samples])),
                "n_terminated": int(terminated),
                "termination_rate": terminated / len(samples),
                "n_generated_tokens": int(generated_tokens),
                "n_residue_token_ids": int(residue_tokens),
                "residue_id_share": residue_tokens / generated_tokens,
                "residues_per_generated_token": sum(len(s) for s in samples) / generated_tokens,
            },
        }
        print(
            f"{role} {block['key']}: {len(samples)} samples, {cells[role]['statistics']['n_empty']} "
            f"empty, {terminated} reached {declaration.protein_end}",
            flush=True,
        )

    payload = _preamble("instructprotein_family_generation_samples")
    payload.update(
        {
            "arm": INSTRUCTPROTEIN_ARM,
            "checkpoint_facts": loaded.facts,
            "sampling": {
                "top_p": cg.TOP_P,
                "temperature": cg.TEMPERATURE,
                "top_k": cg.TOP_K,
                "repetition_penalty": cg.REPETITION_PENALTY,
                "max_new_tokens": cg.MAX_NEW_TOKENS,
                "sampling_seed": int(args.sampling_seed),
                "per_cell_seed_rule": (
                    "conditioned_generation.cell_seed: sha256 of probe, arm, class and "
                    "condition added to the seed, so the two cells never share a sample"
                ),
                "batch_size": int(args.batch_size),
                "batch_size_note": (
                    "a feasibility parameter, not a scientific one; the per-batch seed "
                    "makes the sample depend on it, so it is recorded with the run"
                ),
                "post_selection_filter": cg.POST_SELECTION_FILTER,
                "post_selection_note": cg.POST_SELECTION_NOTE,
            },
            "terminator": {
                "token": declaration.protein_end,
                "id": end_id,
                "note": (
                    "the declared protein end delimiter. It is not this checkpoint's "
                    "end-of-sequence token, so a sample that reaches it and one that "
                    "runs to the token budget are different objects and both are counted"
                ),
            },
            "residue_space": {
                "residue_tokens": len(residue_letters),
                "scored_target_rule": declaration.scored_target_rule,
                "prefix_marker_ids": list(prefix_ids),
                "note": (
                    "the prompt ends at the declared protein start delimiter, so sampling "
                    "begins in the residue subspace; residues_per_generated_token and "
                    "residue_id_share are how much of the sampled span stayed there"
                ),
            },
            "cells": cells,
        }
    )
    return payload


def run_score(args: argparse.Namespace) -> dict[str, Any]:
    anchor = _read(args.anchor_artifact)
    generations = _read(args.generation_artifact)
    referent = tuple(anchor["class"]["referent"])

    named: dict[str, str] = {}
    names: dict[str, list[str]] = {}
    for role, cell in generations["cells"].items():
        names[role] = []
        for index, sequence in enumerate(cell["samples"]):
            name = f"{role}|{index}"
            names[role].append(name)
            if sequence:
                named[name] = sequence

    tool, database = _oracle(args)
    hits, scan = cg.annotate(
        named,
        tool=tool,
        database=database,
        workspace=args.work / "generations",
        threads=args.hmmscan_threads,
        shards=args.hmmscan_shards,
        label="instructprotein_probe_generations",
    )

    flags: dict[str, list[bool]] = {}
    groups: dict[str, np.ndarray] = {}
    grouping: dict[str, Any] = {}
    for role, cell in generations["cells"].items():
        flags[role] = cg.assigned(hits, names[role], referent)
        groups[role], grouping[role] = cg.near_duplicate_group_ids(cell["samples"], unit="residues")

    rates = {
        role: {
            "p_grouped": cg.grouped_rate(flags[role], groups[role]),
            "p_ungrouped": float(np.mean([1.0 if value else 0.0 for value in flags[role]])),
            "n": len(flags[role]),
            "n_hits": int(sum(flags[role])),
            "n_with_any_family": int(sum(1 for name in names[role] if cg.families_of(hits, name))),
        }
        for role in flags
    }
    for block in rates.values():
        # The exact one-sided 95% bound for zero successes in n trials, from
        # (1-p)^n = 0.05. It is reported only where no hit was observed, because
        # that identity is the case it holds in: a nonzero count needs the beta
        # quantile and this probe does not add one to report it.
        block["zero_hit_upper_bound_95"] = (
            1.0 - 0.05 ** (1.0 / block["n"]) if block["n_hits"] == 0 else None
        )
    contrast = cg.two_sample_rate_contrast(
        flags["requested"],
        groups["requested"],
        flags["mismatched"],
        groups["mismatched"],
        resamples=args.bootstrap,
        seed=args.bootstrap_seed,
    )

    statistics = {role: cell["statistics"] for role, cell in generations["cells"].items()}
    pooled_tokens = sum(block["n_generated_tokens"] for block in statistics.values())
    pooled_residue_ids = sum(block["n_residue_token_ids"] for block in statistics.values())
    pooled_extracted = sum(sum(len(sample) for sample in cell["samples"]) for cell in generations["cells"].values())

    admitted = bool(anchor["class_admitted"])
    lower_bound_positive = None if contrast["ci95"] is None else bool(contrast["ci95"][0] > 0.0)
    if not admitted:
        reading = "not_interpretable_anchor_failed"
    elif lower_bound_positive is None:
        reading = "not_scored_interval_degenerate"
    elif lower_bound_positive:
        reading = "requested_label_moves_generation_toward_the_requested_family"
    else:
        reading = "no_measured_movement_toward_the_requested_family"

    payload = _preamble("instructprotein_family_generation_report")
    payload.update(
        {
            "draw_seed": int(anchor["draw_seed"]),
            "class": anchor["class"],
            "class_admitted": admitted,
            "anchor": {
                "referent": list(referent),
                "real_rate": anchor["class"]["real_rate"],
                "random_rate": anchor["class"]["random_rate"],
                "n_real": anchor["class"]["n_real"],
                "n_random": anchor["class"]["n_random"],
                "real_floor": anchor["class"]["real_floor"],
                "random_ceiling": anchor["class"]["random_ceiling"],
                "referent_derivation": anchor["class"]["referent_derivation"],
                "limitation": (
                    "the real side of this anchor and the class's referent share a "
                    "labelling channel (the staged Pfam span table), so the real rate "
                    "re-establishes that the pipeline extracts, writes and searches real "
                    "sequences and that the release recognises the class. It does not "
                    "price class specificity from an independent channel; the random "
                    "side is what prices the oracle's willingness to fire on proteins of "
                    "the class's length that are not of its class"
                ),
            },
            "oracle": {
                "engine": "HMMER 3.4 hmmscan",
                "database": "staged Pfam-A release",
                "threshold": PFAM_THRESHOLD,
                "threshold_note": (
                    "Pfam-A's own curated per-family gathering thresholds (--cut_ga), the "
                    "same invocation EXP-R2-227 uses, so a family call is a statement of "
                    "the release and not a cut chosen inside this probe"
                ),
                "hmmscan": scan,
            },
            "sampling": generations["sampling"],
            "values": {
                "p_requested": rates["requested"]["p_grouped"],
                "p_mismatched": rates["mismatched"]["p_grouped"],
                "difference": contrast["difference"],
                "difference_ci95": contrast["ci95"],
                "n_groups": contrast["n_groups"],
                "degenerate": contrast["degenerate"],
                "degenerate_reason": contrast["degenerate_reason"],
                "resampling_unit": "near-duplicate group",
                "interval_note": (
                    "an unpaired percentile bootstrap of the difference between two "
                    "separate samples of 200, clustered on near-duplicate groups, so a "
                    "burst of near-identical generations contributes once. One class is "
                    "one class: no class-clustered interval is available or claimed, and "
                    "at zero hits in both cells the bootstrap interval is the point "
                    "[0, 0], whose content is the same zero the counts already state"
                ),
            },
            "rates_by_condition": rates,
            "rates_by_condition_note": (
                "p_requested and p_mismatched are the rates at which the oracle assigns a "
                "generation to the requested class's referent; n_with_any_family is how "
                "many generations of that cell carry any Pfam family at all, which is "
                "what separates an oracle that recognises nothing from a cell whose "
                "generations are simply not of the requested class; "
                "zero_hit_upper_bound_95 is reported where no hit was observed"
            ),
            "grouping": grouping,
            "termination": {
                role: {
                    "n_terminated": block["n_terminated"],
                    "rate": block["termination_rate"],
                }
                for role, block in statistics.items()
            },
            "residue_space": {
                "per_cell": {
                    role: {
                        "residue_id_share": block["residue_id_share"],
                        "residues_per_generated_token": block["residues_per_generated_token"],
                        "n_generated_tokens": block["n_generated_tokens"],
                    }
                    for role, block in statistics.items()
                },
                "pooled_residue_id_share": pooled_residue_ids / pooled_tokens,
                "pooled_residues_per_generated_token": pooled_extracted / pooled_tokens,
                "pooled_extracted_residues": int(pooled_extracted),
                "note": (
                    "two different quantities, reported together because they answer "
                    "different questions. The residue-id share is of the whole sampled "
                    "span, including any continuation past the terminator, and is what "
                    "shows the span was decoded in the residue space rather than through "
                    "text. Residues per generated token counts only the extracted run, "
                    "which stops at the first terminator, so it is lower wherever a "
                    "sample kept generating after closing its protein block"
                ),
            },
            "reading": reading,
            "reading_note": (
                "one class under one oracle at n = 200 per cell. A positive interval says "
                "the request moved this checkpoint's generations toward the requested "
                "family on this class and this instrument, which is what makes a third "
                "conditioned arm worth considering; it is not a rate this probe has "
                "measured on any other class"
                if admitted
                else "the instrument anchor did not admit this class, so the two rates "
                "above are not readable as a class-selective statement"
            ),
        }
    )
    return payload


# ----------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", required=True, choices=("class", "anchor", "generate", "score"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--pfam-residue", type=Path, default=PFAM_RESIDUE_TSV)
    parser.add_argument("--pfam-hmm", type=Path, default=None, help="the pressed Pfam-A.hmm; defaults to --work/pfam/Pfam-A.hmm")
    parser.add_argument("--swissprot", type=Path, default=SWISSPROT_FASTA)
    parser.add_argument("--uniref50", type=Path, default=UNIREF50_FASTA)
    parser.add_argument("--class-artifact", type=Path, default=None)
    parser.add_argument("--anchor-artifact", type=Path, default=None)
    parser.add_argument("--generation-artifact", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--generations", type=int, default=GENERATIONS_PER_CELL)
    parser.add_argument("--draw-seed", type=int, default=PROBE_SEED)
    parser.add_argument("--sampling-seed", type=int, default=PROBE_SEED)
    parser.add_argument("--bootstrap", type=int, default=cg.BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=PROBE_SEED)
    parser.add_argument("--reservoir", type=int, default=RESERVOIR)
    parser.add_argument("--hmmscan-threads", type=int, default=8)
    parser.add_argument("--hmmscan-shards", type=int, default=16)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.pfam_hmm is None:
        args.pfam_hmm = args.work / "pfam/Pfam-A.hmm"
    if args.class_artifact is None:
        args.class_artifact = args.out / "class.json"
    if args.anchor_artifact is None:
        args.anchor_artifact = args.out / "anchor.json"
    if args.generation_artifact is None:
        args.generation_artifact = args.out / "generations.json"

    stages = {
        "class": (run_class, args.out / "class.json"),
        "anchor": (run_anchor, args.out / "anchor.json"),
        "generate": (run_generate, args.out / "generations.json"),
        "score": (run_score, args.out / "report.json"),
    }
    function, destination = stages[args.stage]
    payload = function(args)
    write_json(destination, payload)
    print(json.dumps({k: v for k, v in payload.items() if k in ("reading", "class", "classes")}, indent=2, default=str))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
