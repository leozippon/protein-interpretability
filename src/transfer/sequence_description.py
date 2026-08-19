"""Genuine sequence-description pairs, and the leak that would make them useless.

Why this module exists
======================

D3.g -- language-mediated causal concept alignment -- is admitted only if its
alignment "beat[s] every applicable baseline, survive[s] description and homology
leakage checks, preserve[s] unrelated concepts, transfer[s] at least one graded
protein-model intervention in the predicted direction, and reproduce[s] on an
unseen protein family" (audit §8 item 4). Three of those five clauses are
properties of the *data* rather than of the method, and they have to be built and
measured before any activation is captured. This module and
``scripts/transfer/34_sequence_description_cohort.py`` build them.

The failure this is written against has already happened once in this programme.
D3.b's single positive protein cell was ZymCTRL's, and it dissolved when the
pre-declared test held its EC conditioning tag constant: the arm was reading its
own prompt (L15, 1.73 nats). A sequence-description cohort has the same shape of
hazard one level up. **If the description names the concept, a "concept alignment"
is a string match.** ``kinase`` in a curated description makes ``GO:0016301``
trivially decodable from the text side, and an alignment fitted between a protein
activation and that text embedding can score arbitrarily well while the protein
side contributes nothing. So the concept's own surface forms are removed from
every description, the removal is *recorded* per record, and the residual leak is
measured rather than asserted.

What is controlled here, and what is not
========================================

*Description leakage.* Every surface form of a concept's identity -- its GO term
name and its EXACT/NARROW synonyms, the EC number string and the ENZYME
nomenclature description of that EC, the InterPro and Pfam entry names the record
carries, and the accession-style identifiers themselves -- is replaced by one
fixed placeholder, so neither the string nor its length survives. Both the raw and
the masked description are kept, because the *difference* between a result on the
two is the measurement that says whether an alignment was reading the concept
name.

*Verbatim sequence leakage.* A description that quotes its own sequence would let
the text side carry the protein side. Records whose description contains the
sequence, or any run of :data:`VERBATIM_SHINGLE` residues of it, are excluded
from the pool and counted.

*Homology leakage* is **not** this module's; it is the stage's, and it is
:mod:`.near_duplicates`' relation and :mod:`.families`' curated split, measured
against DIAMOND. Splitting is a property of a pool and belongs where the pool is
built.

What is **not** controlled, and cannot be here: *paraphrase*. Masking
``kinase activity`` does not remove ``catalyses the transfer of a phosphate group
from ATP``, which is the same fact in other words, and no lexical procedure
removes it. That residual is what the stage's per-concept raw-versus-masked leak
rates bound from one side and what an alignment's own baselines have to bound
from the other. It is recorded as a limitation in every artefact this module
feeds, and it is the reason masking is a *control* rather than a *fix*.

Swiss-Prot XML iteration lives here
===================================

:func:`iter_swissprot_entries` is this repository's one Swiss-Prot XML reader.
``ops/build_zymctrl_ec_labeled_swissprot.py`` carried the only other copy and now
consumes this one; the namespace resolution, the two EC sources and the
``//`` -- entry -- ``clear()`` streaming discipline are its, preserved verbatim,
because that parser had already earned two of its three defensive checks the hard
way. Appendix B rule 12 applies to *which records exist* as much as to what a
model is fed.

The GO cross-references inside the XML are used in preference to
``data/go/goa_uniprot_all.gaf.gz``. That is a deliberate choice and not an
oversight: the XML carries the annotation, its term name and its evidence code on
the same entry as the sequence and the description, so the join is by
construction and one pass reads everything; the 17.7 GB GAF would have to be
streamed and joined on accession to recover exactly the same triples for
Swiss-Prot entries, and it adds only annotations of entries this cohort does not
contain. What the GAF would add and the XML does not is *nothing this cohort
reads*; if a future estimand needs the full evidence provenance of an annotation
-- multiple evidence codes per (accession, term), or the assigning database --
the GAF becomes necessary and this choice must be revisited.
"""

from __future__ import annotations

import gzip
import json
import os
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple
from xml.etree.ElementTree import iterparse

from .arms import REPO, env_path, require_input_path

# ----------------------------------------------------------------- locations

SWISSPROT_XML = env_path(
    "TRANSFER_SWISSPROT_XML", REPO / "data/swissprot/uniprot_sprot.xml.gz"
)
GO_OBO = env_path("TRANSFER_GO_OBO", REPO / "data/go/go-basic.obo")
ENZYME_DAT = env_path("TRANSFER_ENZYME_DAT", REPO / "data/zymctrl/enzyme.dat")
INTERPRO_ENTRY_LIST = env_path(
    "TRANSFER_INTERPRO_ENTRY_LIST", REPO / "data/interpro/entry.list"
)

# ------------------------------------------------------------- record schema

#: The frozen field order of ``records.jsonl``. Two further stages read this file
#: and neither may guess at it, so the schema is declared once, here, and both
#: the writer and the reader check against this tuple rather than against each
#: other.
RECORD_FIELDS: tuple[str, ...] = (
    "accession",
    "sequence",
    "length",
    "name",
    "function_text",
    "description_raw",
    "description_masked",
    "masked_terms",
    "ec",
    "go",
    "go_propagated",
    "pfam",
    "cath",
    "dup_group",
    "family_group",
    "split",
)

#: The three sides of the cohort. ``fit`` and ``eval`` are near-duplicate-group
#: disjoint and answer L30; ``family_holdout`` is curated-family disjoint from
#: both and answers §8 item 4's "reproduce on an unseen protein family".
SPLIT_NAMES: tuple[str, ...] = ("fit", "eval", "family_holdout")

#: One placeholder for every masked surface form, whatever its length. A
#: length-preserving mask (``xxxxxx``) leaks the term's length, and a
#: per-concept placeholder leaks its identity; both have been used elsewhere and
#: both are side channels of exactly the kind this cohort exists to close.
MASK_PLACEHOLDER = "[MASK]"

#: Shortest surface form that may be masked. Below three characters a "term" is
#: a fragment that matches inside ordinary words even under boundary anchoring.
#: ``ATP`` sits exactly at the floor and is masked, which is the case that fixes
#: the value: it is the whole of ``GO:0005524``'s identity in a description.
MIN_MASKED_TERM_CHARS = 3

#: A description quoting this many consecutive residues of its own sequence is
#: carrying the protein side inside the text side. Fifteen residues is far
#: beyond any motif a curator writes out (PROSITE patterns reach ten or so) and
#: far below a domain, so the check separates a quoted sequence from a quoted
#: motif without a tuned threshold.
VERBATIM_SHINGLE = 15

#: The twenty standard residues. A record carrying ``X``, ``B``, ``Z``, ``U`` or
#: ``O`` is excluded rather than substituted: an ambiguity code is a statement
#: that the sequence is not known, and a cohort that silently maps it to a
#: residue reports a sequence that was never observed.
STANDARD_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWY")

#: Four-field EC numbers only. A partial number (``1.1.1.-``) names a subclass
#: rather than an activity, and the ENZYME nomenclature file has no entry for it.
#: Moved here from ``ops/build_zymctrl_ec_labeled_swissprot.py``, which now
#: imports it.
FULL_EC_PATTERN = re.compile(r"^\d+\.\d+\.\d+\.\d+$")

#: GO synonym scopes whose strings are surface forms of the term *itself*.
#: ``BROAD`` and ``RELATED`` are excluded: a broad synonym of a specific term is
#: a generic word (``metabolism``), and masking it removes ordinary description
#: content while protecting nothing.
GO_SYNONYM_SCOPES = ("EXACT", "NARROW")

#: The evidence code UniProt writes for an electronically inferred annotation.
#: An IEA GO term assigned by a UniRule that fires on a family signature is not
#: independent of the family label, so a run may exclude them -- and has to say
#: which it did, because the two populations are different.
IEA_EVIDENCE = "ECO:0007669"

GO_EVIDENCE_POLICIES = ("all", "non_iea")

#: EC top-level class names, from the ENZYME nomenclature's class list. Carried
#: as a literal because ``enzyme.dat`` holds only leaf entries and the class file
#: is not staged; these seven strings are stable nomenclature, and they are here
#: to be *masked out* of descriptions rather than to label anything.
EC_CLASS_NAMES: dict[str, str] = {
    "1": "oxidoreductase",
    "2": "transferase",
    "3": "hydrolase",
    "4": "lyase",
    "5": "isomerase",
    "6": "ligase",
    "7": "translocase",
}


# ----------------------------------------------------------- Swiss-Prot XML


class GoAnnotation(NamedTuple):
    """One ``<dbReference type="GO">`` of an entry, as the XML carries it."""

    go_id: str
    #: The curated term name with UniProt's aspect prefix stripped: the XML
    #: writes ``F:kinase activity`` and the aspect is kept separately so that a
    #: surface form is never masked with a stray ``F:`` attached to it.
    term: str
    #: ``F``, ``P`` or ``C``.
    aspect: str
    evidence: str


@dataclass(frozen=True)
class SwissProtEntry:
    """One Swiss-Prot entry, reduced to what a description cohort reads."""

    accession: str
    entry_name: str
    protein_name: str
    function_texts: tuple[str, ...]
    sequence: str
    ec: tuple[str, ...]
    go: tuple[GoAnnotation, ...]
    interpro: tuple[tuple[str, str], ...]
    pfam: tuple[tuple[str, str], ...]


def iter_swissprot_entries(path: Path = SWISSPROT_XML) -> Iterator[SwissProtEntry]:
    """Stream Swiss-Prot entries out of the release XML.

    The namespace is read from the document's own root rather than hard-coded.
    That check is inherited from ``ops/build_zymctrl_ec_labeled_swissprot.py``,
    which earned it: UniProt has shipped both ``http://uniprot.org/uniprot`` and
    ``https://uniprot.org/uniprot`` as the default namespace, and against the
    wrong literal every tag comparison fails, every entry is skipped, and the
    consumer writes an empty output and exits 0.

    EC numbers come from **both** places the release puts them -- the
    ``<ecNumber>`` elements inside ``<protein>`` and the ``<dbReference
    type="EC">`` cross-references -- deduplicated and sorted, because an entry
    can carry one and not the other.

    An entry without an accession or without a sequence is skipped, exactly as
    the parser this replaces skipped it: those two fields are the identity of the
    record, and there is nothing to yield without them. Everything else is
    yielded as found, including an empty ``protein_name`` or an empty ``ec``, so
    that eligibility is decided by the caller and is visible in the caller's own
    rejection counts.
    """

    path = Path(path)
    require_input_path(path, "TRANSFER_SWISSPROT_XML")
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as handle:
        namespace: str | None = None
        for event, payload in iterparse(handle, events=("start-ns", "end")):
            if event == "start-ns":
                prefix, uri = payload
                if prefix == "":
                    namespace = f"{{{uri}}}"
                continue
            element = payload
            if namespace is None:
                raise RuntimeError(
                    f"{path} declares no default XML namespace; this parser "
                    "resolves UniProt element names through it"
                )
            if element.tag != f"{namespace}entry":
                continue
            entry = _entry_from_element(element, namespace)
            if entry is not None:
                yield entry
            element.clear()


def _entry_from_element(element: Any, namespace: str) -> SwissProtEntry | None:
    accession_element = element.find(f"{namespace}accession")
    sequence_element = element.find(f"{namespace}sequence")
    if accession_element is None or sequence_element is None:
        return None
    if sequence_element.text is None:
        return None
    accession = (accession_element.text or "").strip()
    sequence = "".join(sequence_element.text.split())
    if not accession or not sequence:
        return None

    name_element = element.find(f"{namespace}name")
    entry_name = (name_element.text or "").strip() if name_element is not None else ""

    protein_name = ""
    protein = element.find(f"{namespace}protein")
    if protein is not None:
        recommended = protein.find(f"{namespace}recommendedName")
        if recommended is not None:
            full = recommended.find(f"{namespace}fullName")
            if full is not None and full.text:
                protein_name = " ".join(full.text.split())

    function_texts: list[str] = []
    for comment in element.findall(f"{namespace}comment[@type='function']"):
        for text in comment.findall(f"{namespace}text"):
            if text.text:
                collapsed = " ".join(text.text.split())
                if collapsed:
                    function_texts.append(collapsed)

    ec: set[str] = set()
    for number in element.findall(f".//{namespace}ecNumber"):
        if number.text and FULL_EC_PATTERN.fullmatch(number.text.strip()):
            ec.add(number.text.strip())
    for reference in element.findall(f".//{namespace}dbReference[@type='EC']"):
        identifier = reference.attrib.get("id", "").strip()
        if FULL_EC_PATTERN.fullmatch(identifier):
            ec.add(identifier)

    go: list[GoAnnotation] = []
    interpro: list[tuple[str, str]] = []
    pfam: list[tuple[str, str]] = []
    for reference in element.findall(f"{namespace}dbReference"):
        kind = reference.attrib.get("type")
        identifier = reference.attrib.get("id", "").strip()
        if not identifier:
            continue
        if kind == "GO":
            term, aspect, evidence = "", "", ""
            for prop in reference.findall(f"{namespace}property"):
                if prop.attrib.get("type") == "term":
                    value = prop.attrib.get("value", "")
                    aspect, _, term = value.partition(":")
                    term = term.strip()
                    aspect = aspect.strip()
                elif prop.attrib.get("type") == "evidence":
                    evidence = prop.attrib.get("value", "").strip()
            go.append(GoAnnotation(identifier, term, aspect, evidence))
        elif kind in ("InterPro", "Pfam"):
            entry_names = [
                prop.attrib.get("value", "").strip()
                for prop in reference.findall(f"{namespace}property")
                if prop.attrib.get("type") == "entry name"
            ]
            pair = (identifier, entry_names[0] if entry_names else "")
            (interpro if kind == "InterPro" else pfam).append(pair)

    return SwissProtEntry(
        accession=accession,
        entry_name=entry_name,
        protein_name=protein_name,
        function_texts=tuple(function_texts),
        sequence=sequence,
        ec=tuple(sorted(ec)),
        go=tuple(go),
        interpro=tuple(interpro),
        pfam=tuple(pfam),
    )


# --------------------------------------------------------------- Gene Ontology


@dataclass(frozen=True)
class GoOntology:
    """``go-basic.obo``, reduced to identity, parenthood and surface forms.

    ``go-basic`` is the filtered release whose only relations are ``is_a`` and
    ``part_of`` and which is guaranteed acyclic, which is what makes
    :meth:`close` a finite ancestor walk rather than a fixed-point iteration.
    """

    names: Mapping[str, str]
    namespaces: Mapping[str, str]
    synonyms: Mapping[str, tuple[str, ...]]
    parents: Mapping[str, tuple[str, ...]]
    obsolete: frozenset[str]
    alt_ids: Mapping[str, str]
    version: str

    def canonical(self, go_id: str) -> str | None:
        """The primary id of a term, following ``alt_id``, or ``None``."""

        if go_id in self.names:
            return go_id
        return self.alt_ids.get(go_id)

    def close(self, go_ids: Iterable[str]) -> tuple[str, ...]:
        """The ancestor closure under ``is_a`` and ``part_of``, sorted.

        Unknown ids are dropped rather than raising, because the ontology release
        and the Swiss-Prot release are versioned independently and a term
        obsoleted between the two is a data fact rather than a program fault. The
        caller counts them; see :func:`unresolved_go_ids`.
        """

        seen: set[str] = set()
        stack = [primary for go_id in go_ids if (primary := self.canonical(go_id))]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self.parents.get(current, ()))
        return tuple(sorted(seen))

    def min_depth_to_root(self, go_id: str) -> int:
        """Shortest parent path to a term with no parents.

        The coarseness statistic a concept declaration is admitted on. Depth is
        computed rather than eyeballed, because "coarse enough for an independent
        annotator to score" is the property §8 item 4 asks for and a comment
        asserting it is exactly the shape of defect §0.05 records.
        """

        primary = self.canonical(go_id)
        if primary is None:
            raise KeyError(f"{go_id} is not a term of this ontology release")
        frontier = {primary}
        seen = set(frontier)
        depth = 0
        while frontier:
            if any(not self.parents.get(term, ()) for term in frontier):
                return depth
            frontier = {
                parent
                for term in frontier
                for parent in self.parents.get(term, ())
                if parent not in seen
            }
            seen |= frontier
            depth += 1
        raise RuntimeError(
            f"{go_id} has no path to a root term; go-basic is acyclic by "
            "construction, so this release is not go-basic"
        )

    def surface_forms(self, go_id: str) -> tuple[str, ...]:
        """Every string that *is* this term: its id, its name, its synonyms."""

        primary = self.canonical(go_id)
        if primary is None:
            return ()
        forms = [primary, self.names[primary], *self.synonyms.get(primary, ())]
        return tuple(dict.fromkeys(form for form in forms if form))


_OBO_SYNONYM = re.compile(r'^synonym:\s+"((?:[^"\\]|\\.)*)"\s+(\w+)')


def load_go_ontology(path: Path = GO_OBO) -> GoOntology:
    """Parse ``go-basic.obo``. Obsolete terms are kept and flagged, not dropped.

    An obsolete term still appears in older Swiss-Prot cross-references, and a
    parser that dropped it would report the annotation as *unknown* -- which is
    the same signal a release mismatch produces, and the two have different
    remedies.
    """

    path = Path(path)
    require_input_path(path, "TRANSFER_GO_OBO")
    names: dict[str, str] = {}
    namespaces: dict[str, str] = {}
    synonyms: dict[str, list[str]] = {}
    parents: dict[str, list[str]] = {}
    obsolete: set[str] = set()
    alt_ids: dict[str, str] = {}
    version = ""

    current: str | None = None
    in_term = False
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith("data-version:"):
                version = line.split(":", 1)[1].strip()
                continue
            if line.startswith("["):
                in_term = line.strip() == "[Term]"
                current = None
                continue
            if not in_term or not line:
                continue
            if line.startswith("id: "):
                current = line[4:].strip()
                names.setdefault(current, "")
                continue
            if current is None:
                continue
            if line.startswith("name: "):
                names[current] = line[6:].strip()
            elif line.startswith("namespace: "):
                namespaces[current] = line[11:].strip()
            elif line.startswith("alt_id: "):
                alt_ids[line[8:].strip()] = current
            elif line.startswith("is_obsolete: true"):
                obsolete.add(current)
            elif line.startswith("is_a: "):
                parents.setdefault(current, []).append(line[6:].split("!")[0].strip())
            elif line.startswith("relationship: part_of "):
                parents.setdefault(current, []).append(
                    line[len("relationship: part_of ") :].split("!")[0].strip()
                )
            else:
                match = _OBO_SYNONYM.match(line)
                if match is not None and match.group(2) in GO_SYNONYM_SCOPES:
                    synonyms.setdefault(current, []).append(
                        match.group(1).replace('\\"', '"')
                    )

    if not names:
        raise RuntimeError(f"{path} yielded no GO terms; it is not an OBO release")
    unknown_parents = {
        parent
        for entries in parents.values()
        for parent in entries
        if parent not in names
    }
    if unknown_parents:
        raise RuntimeError(
            f"{path}: {len(unknown_parents)} is_a/part_of targets are not terms of "
            f"this file (first: {sorted(unknown_parents)[:5]}); the ontology is "
            "truncated and an ancestor closure over it would be silently partial"
        )
    return GoOntology(
        names=names,
        namespaces=namespaces,
        synonyms={key: tuple(value) for key, value in synonyms.items()},
        parents={key: tuple(dict.fromkeys(value)) for key, value in parents.items()},
        obsolete=frozenset(obsolete),
        alt_ids=alt_ids,
        version=version,
    )


def unresolved_go_ids(ontology: GoOntology, go_ids: Iterable[str]) -> tuple[str, ...]:
    """Ids the ontology release does not know at all, sorted and deduplicated."""

    return tuple(
        sorted({go_id for go_id in go_ids if ontology.canonical(go_id) is None})
    )


# ------------------------------------------------------- other label sources


def load_enzyme_descriptions(path: Path = ENZYME_DAT) -> dict[str, tuple[str, ...]]:
    """EC number to its ENZYME nomenclature names: the ``DE`` and every ``AN``.

    Both are surface forms of the same identity -- ``DE`` is the accepted name
    and each ``AN`` an alternative one -- so a description naming either has
    named the EC. The trailing full stop ENZYME writes is stripped, because it is
    punctuation of the record rather than of the name.

    **Wrapped names are rejoined, and that is not cosmetic.** ENZYME wraps a long
    name across successive ``DE``/``AN`` lines and marks the end of an entry with
    the full stop, so ``AN`` 2.4.1.227 arrives as
    ``UDP-...-pyrophosphoryl-`` and ``undecaprenol N-acetylglucosamine
    transferase.``. Treating each line as its own name produced a *fragment*, and
    a fragment ending in a hyphen is a surface form that matches nothing until
    its neighbour is masked and then matches -- which is how it was found, by
    :func:`mask_description`'s own fixed-point check refusing three records of a
    1,200-record pool. A fragment is also wrong in the other direction: it would
    mask half a name out of a description that never carried the whole one.

    The join is empty after a trailing hyphen -- the wrap is mid-word there -- and
    a single space otherwise.
    """

    path = Path(path)
    require_input_path(path, "TRANSFER_ENZYME_DAT")
    descriptions: dict[str, list[str]] = {}
    current: str | None = None
    pending: str | None = None

    def flush() -> None:
        nonlocal pending
        if current is not None and pending:
            descriptions[current].append(pending.rstrip(".").strip())
        pending = None

    with path.open(encoding="latin-1") as handle:
        for line in handle:
            tag, _, value = line.partition("   ")
            value = value.strip()
            if tag == "ID":
                flush()
                current = value if FULL_EC_PATTERN.fullmatch(value) else None
                if current is not None:
                    descriptions.setdefault(current, [])
            elif tag in ("DE", "AN") and current is not None and value:
                if pending is None:
                    pending = value
                else:
                    pending += "" if pending.endswith("-") else " "
                    pending += value
                if pending.endswith("."):
                    flush()
            else:
                # Any other tag, or the record separator, ends an unterminated
                # name rather than letting it absorb the next entry's text.
                flush()
                if line.startswith("//"):
                    current = None
    flush()
    if not descriptions:
        raise RuntimeError(f"{path} yielded no EC entries; it is not an ENZYME release")
    return {
        key: tuple(dict.fromkeys(name for name in value if name))
        for key, value in descriptions.items()
    }


def load_interpro_entry_names(path: Path = INTERPRO_ENTRY_LIST) -> dict[str, str]:
    """InterPro accession to its descriptive entry name, from ``entry.list``."""

    path = Path(path)
    require_input_path(path, "TRANSFER_INTERPRO_ENTRY_LIST")
    names: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        header = tuple(next(handle).rstrip("\n").split("\t"))
        if header != ("ENTRY_AC", "ENTRY_TYPE", "ENTRY_NAME"):
            raise ValueError(f"{path}: unexpected columns {header}")
        for number, line in enumerate(handle, 2):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3:
                raise ValueError(f"{path}:{number}: expected 3 columns, found {len(fields)}")
            names[fields[0]] = fields[2]
    if not names:
        raise RuntimeError(f"{path} holds no entries")
    return names


# ------------------------------------------------------------- the concepts


@dataclass(frozen=True)
class ConceptSpec:
    """One pre-declared concept, with the rule that decides both of its sides.

    ``kind`` fixes where the label comes from and therefore what "negative"
    means. Neither rule is open-world: a record with no annotation of the
    relevant sort is *undefined* for the concept and enters neither side, which
    is the only honest reading of a curated database where absence of an
    annotation is absence of curation.
    """

    concept_id: str
    kind: str
    identifier: str
    name: str

    def __post_init__(self) -> None:
        if self.kind not in ("go", "ec"):
            raise ValueError(f"unknown concept kind {self.kind!r}")
        if self.kind == "ec" and self.identifier not in EC_CLASS_NAMES:
            raise ValueError(
                f"{self.concept_id}: EC concepts are declared at the top-level "
                f"class; {self.identifier!r} is not one of {sorted(EC_CLASS_NAMES)}"
            )

    @property
    def positive_rule(self) -> str:
        if self.kind == "go":
            return f"{self.identifier} is in the record's is_a/part_of-closed GO set"
        return f"the record carries an EC number in class {self.identifier}"

    @property
    def negative_rule(self) -> str:
        if self.kind == "go":
            return (
                "the record carries at least one GO term in the same aspect and "
                f"{self.identifier} is not in its closure"
            )
        return "the record carries at least one EC number and none is in that class"

    def record(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "kind": self.kind,
            "identifier": self.identifier,
            "name": self.name,
            "positive_rule": self.positive_rule,
            "negative_rule": self.negative_rule,
        }


#: The declared candidate concepts. This tuple is frozen; which of its members a
#: run *admits* is computed against the realised splits by
#: :func:`admit_concepts` and recorded, never asserted here. The candidates are
#: coarse on purpose -- an independent annotator has to be able to score them
#: from a structure or an assay, which is what makes a downstream biological
#: validation possible at all -- and they deliberately span molecular function,
#: biological process, cellular component and enzyme class, so that a method that
#: works on one kind of concept and not another is visible rather than averaged.
CONCEPTS: tuple[ConceptSpec, ...] = (
    ConceptSpec("go_dna_binding", "go", "GO:0003677", "DNA binding"),
    ConceptSpec("go_rna_binding", "go", "GO:0003723", "RNA binding"),
    ConceptSpec("go_atp_binding", "go", "GO:0005524", "ATP binding"),
    ConceptSpec("go_metal_ion_binding", "go", "GO:0046872", "metal ion binding"),
    ConceptSpec("go_kinase_activity", "go", "GO:0016301", "kinase activity"),
    ConceptSpec("go_hydrolase_activity", "go", "GO:0016787", "hydrolase activity"),
    ConceptSpec("go_oxidoreductase_activity", "go", "GO:0016491", "oxidoreductase activity"),
    ConceptSpec("go_transmembrane_transport", "go", "GO:0055085", "transmembrane transport"),
    ConceptSpec("go_translation", "go", "GO:0006412", "translation"),
    ConceptSpec("go_dna_repair", "go", "GO:0006281", "DNA repair"),
    ConceptSpec("go_membrane", "go", "GO:0016020", "membrane"),
    ConceptSpec("ec_oxidoreductase", "ec", "1", "oxidoreductase"),
    ConceptSpec("ec_transferase", "ec", "2", "transferase"),
    ConceptSpec("ec_hydrolase", "ec", "3", "hydrolase"),
    ConceptSpec("ec_lyase", "ec", "4", "lyase"),
    ConceptSpec("ec_isomerase", "ec", "5", "isomerase"),
    ConceptSpec("ec_ligase", "ec", "6", "ligase"),
)

#: Deepest a GO concept may sit below its aspect root. A term twelve edges down
#: is a curator's distinction rather than a concept an independent annotator
#: could score, and the whole point of the declaration is that the label can be
#: checked outside the corpus that supplied it.
CONCEPT_MAX_ROOT_DEPTH = 5

#: The two splits EXP-R2-213's C34-5 reads admission on. ``fit`` is not one of
#: them: a map is fitted there and nothing is reported there, so a thin concept
#: on the fitting side costs power and does not invalidate a reading, while a thin
#: concept on either reporting side does.
DECIDING_SPLITS: tuple[str, ...] = ("eval", "family_holdout")

#: Per-cell group floors the surviving-concept count is reported at before the
#: declared floor is applied (C34-6). This is Appendix B rule 2's arithmetic half:
#: show that the statistic can reach the number before reading a verdict off it.
ADMISSION_FLOOR_CURVE: tuple[int, ...] = (4, 8, 16, 32)


def concept_label(
    spec: ConceptSpec,
    *,
    go_propagated: Sequence[str],
    ec: Sequence[str],
    ontology: GoOntology,
) -> int | None:
    """1, 0, or ``None`` when the concept is undefined for this record."""

    if spec.kind == "ec":
        if not ec:
            return None
        return int(any(number.split(".", 1)[0] == spec.identifier for number in ec))
    aspect = ontology.namespaces.get(spec.identifier)
    if aspect is None:
        raise KeyError(f"{spec.identifier} carries no namespace in this ontology")
    if spec.identifier in go_propagated:
        return 1
    in_aspect = any(
        ontology.namespaces.get(term) == aspect for term in go_propagated
    )
    return 0 if in_aspect else None


def concept_surface_forms(spec: ConceptSpec, *, ontology: GoOntology) -> tuple[str, ...]:
    """Every string that names the *concept* itself.

    Deliberately narrow. The wider set -- for an EC class, the ENZYME name of
    every enzyme in it -- is not returned here, because those strings belong to
    the *record* rather than to the concept: a description leaks EC 1 by naming
    its own ``alcohol dehydrogenase``, and that string is removed by
    :func:`record_identity_forms`, which is where the record's own identity is
    handled. Keeping the two apart is what lets the per-concept leak rate mean
    "the description named the concept" rather than "the description named
    something in the concept's extension", and it is also what keeps masking a
    per-record operation over tens of forms instead of thousands.
    """

    if spec.kind == "go":
        return ontology.surface_forms(spec.identifier)
    return (EC_CLASS_NAMES[spec.identifier], f"EC {spec.identifier}")


def record_identity_forms(
    *,
    go_ids: Sequence[str],
    go_terms: Sequence[str],
    ec: Sequence[str],
    pfam_entries: Sequence[tuple[str, str]],
    interpro_entries: Sequence[tuple[str, str]],
    ontology: GoOntology,
    enzyme: Mapping[str, tuple[str, ...]],
    interpro_names: Mapping[str, str],
) -> tuple[str, ...]:
    """Every string that names *this record's* curated identity.

    The record's own annotations are its concept membership stated in words:
    ``alcohol dehydrogenase`` is EC 1 written out, ``Protein kinase domain`` is
    the Pfam family whose presence is why the GO kinase term is there at all.
    Masking the concept name alone would leave all of it in place.

    ``go_terms`` is UniProt's own spelling of the term, kept beside the
    ontology's, because the two releases are versioned independently and a term
    renamed between them would otherwise leave the older spelling unmasked.
    """

    forms: list[str] = []
    for go_id, term in zip(go_ids, go_terms, strict=True):
        forms.append(go_id)
        forms.append(term)
        forms.extend(ontology.surface_forms(go_id))
    for number in ec:
        forms.append(number)
        forms.append(f"EC {number}")
        forms.extend(enzyme.get(number, ()))
        forms.append(EC_CLASS_NAMES.get(number.split(".", 1)[0], ""))
    for identifier, name in pfam_entries:
        forms.extend((identifier, name))
    for identifier, name in interpro_entries:
        forms.extend((identifier, name, interpro_names.get(identifier, "")))
    return tuple(sorted({form.strip() for form in forms if form and form.strip()}))


# ------------------------------------------------------------------ masking


def _variants(term: str) -> tuple[str, ...]:
    """A surface form and its regular English inflections.

    Only the forms a curated description actually uses. ``activity``/
    ``activities``, ``kinase``/``kinases``, ``synthase``/``synthases``: the
    plural of a term is the same term, and a mask that removed only the singular
    would leave the concept named in every description that happens to use the
    plural.
    """

    forms = {term}
    lowered = term.lower()
    if lowered.endswith("y") and not lowered.endswith(("ay", "ey", "iy", "oy", "uy")):
        forms.add(term[:-1] + "ies")
    elif lowered.endswith(("s", "x", "z", "ch", "sh")):
        forms.add(term + "es")
    else:
        forms.add(term + "s")
    if lowered.endswith("ies"):
        forms.add(term[:-3] + "y")
    elif lowered.endswith("es"):
        forms.add(term[:-2])
    if lowered.endswith("s"):
        forms.add(term[:-1])
    return tuple(sorted({form for form in forms if len(form) >= MIN_MASKED_TERM_CHARS}))


#: Passes :func:`mask_description` may take before it declares non-convergence.
#: Each pass that replaces anything strictly reduces the unmasked alphanumeric
#: content, so a fixed point exists; the bound turns a hypothetical cycle into a
#: refusal instead of a hang.
MAX_MASK_PASSES = 8


def mask_description(
    description: str, terms: Iterable[str]
) -> tuple[str, tuple[str, ...], int]:
    """Replace every surface form by :data:`MASK_PLACEHOLDER`, to a fixed point.

    Returns the masked description, the canonical terms that actually matched,
    and the number of spans replaced. Longest form first, so that
    ``protein kinase activity`` is masked as one span rather than leaving
    ``protein`` beside two placeholders -- the shorter form is then inside an
    already-replaced region and cannot match, which is why the order is part of
    the contract and not a detail.

    Boundaries are non-alphanumeric rather than ``\\b`` because surface forms
    carry hyphens, commas and digits (``1-aminocyclopropane-1-carboxylate
    deaminase``), and ``\\b`` anchors inside those.

    **One pass is not enough, and that was measured rather than reasoned.**
    Masking changes the neighbours of the text it leaves behind, so a form whose
    match was blocked by an alphanumeric neighbour becomes matchable once that
    neighbour becomes a placeholder: ``pyrophosphoryl-`` cannot match inside
    ``pyrophosphoryl-undecaprenol``, and matches immediately in
    ``pyrophosphoryl-[MASK]``. Three records of a 1,200-record pool survived a
    single-pass mask that way. So the substitution is iterated until a pass
    replaces nothing, which is what "no declared surface form survives" means.

    A term that would match inside the placeholder is refused rather than
    masked: replacing text *inside* ``[MASK]`` with ``[MASK]`` grows the string
    on every pass, and the fixed point would not exist.
    """

    ordered = sorted(
        {term for term in terms if len(term) >= MIN_MASKED_TERM_CHARS},
        key=lambda term: (-len(term), term),
    )
    patterns: list[tuple[str, re.Pattern[str]]] = []
    for term in ordered:
        pattern = re.compile(
            r"(?<![0-9A-Za-z])(?:"
            + "|".join(re.escape(form) for form in _variants(term))
            + r")(?![0-9A-Za-z])",
            flags=re.IGNORECASE,
        )
        if pattern.search(MASK_PLACEHOLDER):
            raise ValueError(
                f"surface form {term!r} matches the mask placeholder itself, so "
                "masking would not converge"
            )
        patterns.append((term, pattern))

    matched: set[str] = set()
    spans = 0
    text = description
    for _ in range(MAX_MASK_PASSES):
        replaced = 0
        for term, pattern in patterns:
            text, count = pattern.subn(MASK_PLACEHOLDER, text)
            if count:
                matched.add(term)
                replaced += count
        spans += replaced
        if not replaced:
            return text, tuple(sorted(matched)), spans
    raise RuntimeError(
        f"masking did not converge in {MAX_MASK_PASSES} passes on "
        f"{description[:80]!r}; a surface form is re-creating a match"
    )


def canonical_description(name: str, function_texts: Sequence[str]) -> str:
    """The one description string a record carries.

    Name first, then the function comments in release order. Both halves are
    kept because they carry different things -- the name is the curator's
    identity for the protein and the function text is what it does -- and a
    cohort that kept only one would make a downstream negative uninterpretable:
    an alignment that fails on names alone has not been shown to fail on
    descriptions.
    """

    head = name.strip().rstrip(".")
    body = " ".join(text.strip() for text in function_texts if text.strip())
    if head and body:
        return f"{head}. {body}"
    return head or body


def sequence_in_description(
    sequence: str, description: str, *, shingle: int = VERBATIM_SHINGLE
) -> dict[str, Any]:
    """Does the description quote the sequence, verbatim or in part?

    The uppercase-run restriction is what makes this specific rather than
    paranoid: residues are written in upper case, English prose is not, and a
    fifteen-character uppercase run in a curated description is a quoted
    sequence or an accession and nothing else.
    """

    if shingle < 1:
        raise ValueError("shingle must be positive")
    upper = sequence.upper()
    kmers = {upper[i : i + shingle] for i in range(len(upper) - shingle + 1)}
    runs = [
        run
        for run in re.findall(rf"[A-Z]{{{shingle},}}", description)
        if any(run[i : i + shingle] in kmers for i in range(len(run) - shingle + 1))
    ]
    return {
        "contains_full_sequence": upper in description.upper(),
        "shingle": int(shingle),
        "shared_runs": tuple(sorted(set(runs))),
    }


# ------------------------------------------------------------- the records


@dataclass(frozen=True)
class SequenceDescriptionRecord:
    """One cohort record, in the frozen schema :data:`RECORD_FIELDS` declares."""

    accession: str
    sequence: str
    length: int
    name: str
    function_text: str
    description_raw: str
    description_masked: str
    masked_terms: tuple[str, ...]
    ec: tuple[str, ...]
    go: tuple[str, ...]
    go_propagated: tuple[str, ...]
    pfam: tuple[str, ...]
    cath: tuple[str, ...]
    dup_group: int
    family_group: str
    split: str

    def __post_init__(self) -> None:
        if self.split not in SPLIT_NAMES:
            raise ValueError(f"unknown split {self.split!r}; splits are {SPLIT_NAMES}")
        if self.length != len(self.sequence):
            raise ValueError(
                f"{self.accession}: declared length {self.length} against a "
                f"{len(self.sequence)}-residue sequence"
            )
        if not set(self.go).issubset(self.go_propagated):
            raise ValueError(
                f"{self.accession}: go_propagated does not contain every direct "
                "term, so it is not a closure of them"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "accession": self.accession,
            "sequence": self.sequence,
            "length": self.length,
            "name": self.name,
            "function_text": self.function_text,
            "description_raw": self.description_raw,
            "description_masked": self.description_masked,
            "masked_terms": list(self.masked_terms),
            "ec": list(self.ec),
            "go": list(self.go),
            "go_propagated": list(self.go_propagated),
            "pfam": list(self.pfam),
            "cath": list(self.cath),
            "dup_group": int(self.dup_group),
            "family_group": self.family_group,
            "split": self.split,
        }
        if tuple(payload) != RECORD_FIELDS:
            raise RuntimeError(
                "the serialised field order has drifted from RECORD_FIELDS, which "
                "two downstream stages read"
            )
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SequenceDescriptionRecord:
        missing = [name for name in RECORD_FIELDS if name not in payload]
        extra = [name for name in payload if name not in RECORD_FIELDS]
        if missing or extra:
            raise ValueError(
                f"record does not match the frozen schema: missing {missing}, "
                f"unexpected {extra}"
            )
        return cls(
            accession=str(payload["accession"]),
            sequence=str(payload["sequence"]),
            length=int(payload["length"]),
            name=str(payload["name"]),
            function_text=str(payload["function_text"]),
            description_raw=str(payload["description_raw"]),
            description_masked=str(payload["description_masked"]),
            masked_terms=tuple(payload["masked_terms"]),
            ec=tuple(payload["ec"]),
            go=tuple(payload["go"]),
            go_propagated=tuple(payload["go_propagated"]),
            pfam=tuple(payload["pfam"]),
            cath=tuple(payload["cath"]),
            dup_group=int(payload["dup_group"]),
            family_group=str(payload["family_group"]),
            split=str(payload["split"]),
        )


def write_records(path: Path, records: Sequence[SequenceDescriptionRecord]) -> None:
    """One JSON object per line, written atomically.

    Atomic for the reason :mod:`.io` is: a partial JSONL left by an interrupted
    write is a valid JSONL of fewer records, and a reader that only counts lines
    cannot tell it from a complete cohort.
    """

    if not records:
        raise ValueError("refusing to write an empty cohort")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_records(path: Path) -> list[SequenceDescriptionRecord]:
    """Read ``records.jsonl`` back, checking every line against the schema."""

    records: list[SequenceDescriptionRecord] = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(SequenceDescriptionRecord.from_dict(json.loads(line)))
            except (ValueError, KeyError) as error:
                raise ValueError(f"{path}:{number}: {error}") from error
    if not records:
        raise RuntimeError(f"{path} holds no records")
    return records


# ------------------------------------------------------------ concept admission


def declaration_reason(
    spec: ConceptSpec,
    ontology: GoOntology,
    *,
    max_root_depth: int = CONCEPT_MAX_ROOT_DEPTH,
) -> str | None:
    """Why a declared concept cannot be scored at all, or ``None``.

    Checked before any label is computed, because the two failures are different
    facts: a concept the ontology release does not carry is a declaration that
    has gone stale, and a concept with too few records is a statement about the
    cohort. Reporting the second where the first is true would blame the pool for
    a broken declaration.
    """

    if spec.kind != "go":
        return None
    if ontology.canonical(spec.identifier) is None:
        return f"{spec.identifier} is not a term of ontology {ontology.version}"
    if spec.identifier in ontology.obsolete:
        return f"{spec.identifier} is obsolete in ontology {ontology.version}"
    depth = ontology.min_depth_to_root(spec.identifier)
    if depth > max_root_depth:
        return (
            f"minimum depth to root is {depth}, beyond the declared maximum "
            f"{max_root_depth}: too fine for an independent annotator to score"
        )
    return None


@dataclass(frozen=True)
class ConceptAdmission:
    """Which declared concepts a realised cohort can carry, and why not."""

    admitted: tuple[str, ...]
    counts: dict[str, dict[str, dict[str, int]]]
    admissible_per_split: dict[str, tuple[str, ...]]
    floor_curve: dict[str, dict[str, int]]
    min_groups_per_cell: int
    deciding_splits: tuple[str, ...]
    rejected: tuple[dict[str, Any], ...] = field(default=())

    def record(self) -> dict[str, Any]:
        return {
            "declared": [spec.record() for spec in CONCEPTS],
            "admitted": list(self.admitted),
            "rejected": list(self.rejected),
            "admissible_per_split": {
                name: list(entries) for name, entries in self.admissible_per_split.items()
            },
            "counts_per_split": self.counts,
            "floor_curve": self.floor_curve,
            "rule": (
                "a concept is admitted where, in EVERY deciding split "
                f"{list(self.deciding_splits)}, at least {self.min_groups_per_cell} "
                "distinct near-duplicate groups bear it and at least "
                f"{self.min_groups_per_cell} do not. The unit is the near-duplicate "
                "group and not the record, because that is the unit every interval "
                "downstream resamples over; the floor is MINIMUM_BOOTSTRAP_UNITS "
                "and is a statistical requirement rather than a tuned number "
                "(EXP-R2-213 C34-5)"
            ),
            "floor_curve_statistic": (
                "surviving concepts as a function of the per-cell group floor, "
                "computed before the declared floor is applied, so the count at "
                "the declared value is read off a curve rather than asserted "
                "(EXP-R2-213 C34-6). 'both' is the intersection over the deciding "
                "splits, which is what admission takes"
            ),
        }


def admit_concepts(
    labels: Mapping[str, Sequence[int | None]],
    splits: Sequence[str],
    groups: Sequence[int],
    *,
    ontology: GoOntology,
    min_groups_per_cell: int,
    deciding_splits: Sequence[str] = DECIDING_SPLITS,
    floor_curve: Sequence[int] = ADMISSION_FLOOR_CURVE,
    max_root_depth: int = CONCEPT_MAX_ROOT_DEPTH,
) -> ConceptAdmission:
    """Apply EXP-R2-213's C34-5 selection rule, and record what it decided.

    Counted in **near-duplicate groups**, not records. Forty records of one
    near-clonal group are one unit for every interval this campaign will report,
    so a concept with forty positives in one group has one bearing unit and
    cannot support a group bootstrap; counting records would admit it.

    A group is *bearing* in a split when at least one of its records in that
    split is positive, and *non-bearing* when at least one is negative. A group
    can be both -- near-duplicates usually share annotations, so this is rare --
    and the overlap is reported rather than resolved by a tie-break, because
    which way it were broken would change a floor comparison silently.
    """

    if min_groups_per_cell < 1:
        raise ValueError("a concept needs at least one group on each side")
    for spec in CONCEPTS:
        if spec.concept_id not in labels:
            raise KeyError(f"no labels were computed for {spec.concept_id}")
    if any(len(column) != len(splits) for column in labels.values()):
        raise ValueError("labels and split assignments must be aligned")
    if len(groups) != len(splits):
        raise ValueError("group ids and split assignments must be aligned")
    unknown = sorted({name for name in splits if name not in SPLIT_NAMES})
    if unknown:
        raise ValueError(f"unknown split names {unknown}; splits are {SPLIT_NAMES}")
    outside = [name for name in deciding_splits if name not in SPLIT_NAMES]
    if outside:
        raise ValueError(f"deciding splits {outside} are not splits of this cohort")

    counts: dict[str, dict[str, dict[str, int]]] = {}
    declaration: dict[str, str | None] = {}
    for spec in CONCEPTS:
        declaration[spec.concept_id] = declaration_reason(
            spec, ontology, max_root_depth=max_root_depth
        )
        bearing: dict[str, set[int]] = {name: set() for name in SPLIT_NAMES}
        non_bearing: dict[str, set[int]] = {name: set() for name in SPLIT_NAMES}
        records = {
            name: {"positive": 0, "negative": 0, "undefined": 0} for name in SPLIT_NAMES
        }
        for value, split, group in zip(
            labels[spec.concept_id], splits, groups, strict=True
        ):
            if value is None:
                records[split]["undefined"] += 1
                continue
            if value:
                records[split]["positive"] += 1
                bearing[split].add(int(group))
            else:
                records[split]["negative"] += 1
                non_bearing[split].add(int(group))
        counts[spec.concept_id] = {
            name: {
                "bearing_groups": len(bearing[name]),
                "non_bearing_groups": len(non_bearing[name]),
                "groups_on_both_sides": len(bearing[name] & non_bearing[name]),
                **records[name],
            }
            for name in SPLIT_NAMES
        }

    def survives(concept_id: str, split: str, floor: int) -> bool:
        if declaration[concept_id] is not None:
            return False
        cell = counts[concept_id][split]
        return (
            cell["bearing_groups"] >= floor and cell["non_bearing_groups"] >= floor
        )

    admissible_per_split = {
        split: tuple(
            spec.concept_id
            for spec in CONCEPTS
            if survives(spec.concept_id, split, min_groups_per_cell)
        )
        for split in deciding_splits
    }
    admitted = tuple(
        spec.concept_id
        for spec in CONCEPTS
        if all(
            spec.concept_id in admissible_per_split[split] for split in deciding_splits
        )
    )

    curve: dict[str, dict[str, int]] = {}
    for floor in floor_curve:
        cell = {
            split: sum(
                1 for spec in CONCEPTS if survives(spec.concept_id, split, int(floor))
            )
            for split in deciding_splits
        }
        cell["both"] = sum(
            1
            for spec in CONCEPTS
            if all(survives(spec.concept_id, split, int(floor)) for split in deciding_splits)
        )
        curve[str(int(floor))] = cell

    rejected: list[dict[str, Any]] = []
    for spec in CONCEPTS:
        if spec.concept_id in admitted:
            continue
        reason = declaration[spec.concept_id]
        if reason is None:
            short = [
                f"{split}({counts[spec.concept_id][split]['bearing_groups']} bearing"
                f"/{counts[spec.concept_id][split]['non_bearing_groups']} non-bearing"
                " groups)"
                for split in deciding_splits
                if not survives(spec.concept_id, split, min_groups_per_cell)
            ]
            reason = (
                f"below the declared floor of {min_groups_per_cell} bearing and "
                f"{min_groups_per_cell} non-bearing near-duplicate groups in every "
                "deciding split: " + ", ".join(short)
            )
        rejected.append({"concept_id": spec.concept_id, "reason": reason})

    return ConceptAdmission(
        admitted=admitted,
        counts=counts,
        admissible_per_split=admissible_per_split,
        floor_curve=curve,
        min_groups_per_cell=int(min_groups_per_cell),
        deciding_splits=tuple(deciding_splits),
        rejected=tuple(rejected),
    )
