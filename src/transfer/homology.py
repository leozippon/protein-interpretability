"""Training-set-overlap control for the induction-head finding.

The finding this module exists to attack
=======================================

The programme reports that autoregressive protein decoders carry induction
heads at least as sharp as a matched text decoder's but roughly five times
fewer of them, at matched convergence and after tokenisation adjustment.  That
reading assumes the prefix-matching score measures a *computation*: a head that,
on seeing a token it has seen before, attends to whatever followed the earlier
occurrence, and copies it.

There is a competing explanation that produces the same number.  The protein
decoders were pretrained on UniRef50 (ProtGPT2 directly; ProGen2 on UniRef90 and
BFD, of which UniRef50 representatives are members) and the natural-repeat
cohort is drawn from Swiss-Prot, which is largely *inside* UniRef50.  A model
that has memorised a near-duplicate of the probe sequence can produce
induction-shaped attention by retrieval rather than by copying: the earlier
occurrence is a cue into a stored sequence, not an operand of a general
algorithm.  If that is what is happening, the head-count comparison is a
statement about training-set overlap, and the modality contrast that the
programme draws from it does not follow.

The design
==========

For every sequence in the natural-repeat cohort we find its closest homologue in
the pretraining corpus with DIAMOND, stratify the cohort by maximum sequence
identity to that corpus, and recompute the induction quantities within each
stratum using exactly the estimators in :mod:`.circuits`.  The synthetic-repeat
probe is the negative control: it is constructed in token space from the arm's
own unigram distribution and appears in no corpus, so it cannot be memorised.

Interpretation, fixed before any result was seen
================================================

These three readings were written into this docstring before the first search
was run, and the report is made against them without amendment:

1. **Memorisation.**  Induction concentrated in the high-homology strata and
   weak at low homology.  The head-count finding would then be confounded by
   training-set overlap and could not be quoted as a statement about protein
   computation.
2. **General mechanism.**  Induction stable across strata *and* present at
   comparable strength on synthetic repeats.  The finding survives: a
   computation that runs on sequences the model cannot have stored is a
   computation.
3. **Neither.**  Intermediate or non-monotone across strata.  Reported as such.
   No verdict is to be forced from a non-monotone gradient, and no stratum
   boundary is to be moved after the fact to produce one.

The boundaries below are fixed for the same reason and must not be retuned once
results exist.  Bins are reported with their achieved counts even when empty.

What this control cannot do
===========================

**GPT-2's training corpus is not public.**  WebText was never released;
OpenWebText is an independent reconstruction of the collection procedure, not
the corpus GPT-2 saw.  Searching a protein cohort against UniRef50 and a text
cohort against OpenWebText would not be the same measurement, because a miss on
the text side could mean "GPT-2 never saw this" or "the reconstruction happens
not to contain it", and there is no way to tell which.  No text-side equivalent
is constructed here.  The consequence is precise and worth stating twice: this
control can establish whether the *protein* induction signal is general or
retrieved, which is what determines whether the protein head count means
anything mechanistically; it cannot support a matched cross-modal claim of the
form "both arms were checked for memorisation and both came back clean".

Two further limitations are structural rather than incidental:

*Swiss-Prot is inside UniRef50 by construction.*  Every Swiss-Prot entry belongs
to some UniRef50 cluster, and that cluster's representative is in the database,
at >=50% identity over >=80% of the shorter sequence by the clustering
definition.  The <30% stratum is therefore expected to be empty or near-empty on
a Swiss-Prot cohort, and its emptiness is a property of the cohort rather than
evidence about the model.  The contrast this control can actually resolve is
between cohort members that *are* UniRef50 representatives (present verbatim in
the corpus, identity ~100) and members that are not (present only through a
diverged relative).  That contrast is the memorisation contrast, and it is the
one the strata are cut to expose.

*The local UniRef50 snapshot is newer than ProtGPT2's training release.*
ProtGPT2 was trained on UniRef50 2021_04.  A hit found here may be a sequence
deposited after that release, which the model never saw.  This biases the
measured identity *upward* relative to the true training corpus, so records are
if anything assigned to strata that are too high, which makes a memorisation
gradient easier to see rather than harder.  A conservative direction for
interpretation 1 and an anti-conservative one for interpretation 2.

Direction of bias from an incomplete database
=============================================

This module records the exact database searched.  When the full local UniRef50
snapshot is indexed -- which is what the accompanying script does, and what the
JSON records -- there is no subset caveat.  Were a subset used instead, the bias
would not be symmetric and should not be described as simply "conservative":

* a *found* high-identity hit is always real, because DIAMOND identity is a
  lower bound on identity to the whole corpus.  The high strata stay pure and a
  memorisation gradient found in them is trustworthy;
* a *missing* hit is not evidence of absence.  A subset contaminates the
  low-identity strata with records whose true near-duplicate was simply not
  indexed, which inflates apparent induction at low homology and pushes the
  reading towards interpretation 2.

So an incomplete database is conservative for concluding memorisation and
anti-conservative for clearing the finding.  Since clearing the finding is the
outcome the programme wants, a subset weakens exactly the conclusion it would
most like to draw, and the synthetic-repeat negative control -- which no
database can contaminate -- is what has to carry that side of the argument.

Estimators
==========

Nothing statistical is reimplemented here.  Prefix-matching scores come from
:func:`.circuits.attention_alignment_scores`, head counts from
:func:`.circuits.head_census`, distribution summaries from
:func:`.circuits.summarise_head_matrix`, OV copying from
:func:`.circuits.ov_copying_scores`, and the probes themselves from
:func:`.circuits.natural_repeat_probes` and
:func:`.circuits.synthetic_repeat_probes`.  This module scores one probe at a
time and pools the per-probe sums, which is *arithmetically identical* to
calling the estimator on a probe list -- an equality this module verifies rather
than asserts, in :func:`verify_pooling` -- and which buys the ability to
re-partition the same forward passes by stratum and to bootstrap over probes
without spending a second pass on the GPU.
"""

from __future__ import annotations

import hashlib
import math
import re
import subprocess
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .arms import Arm, Cohort
from .circuits import (
    INDUCTION_THRESHOLDS,
    RepeatProbe,
    attention_alignment_scores,
    head_census,
    n_head,
    natural_repeat_probes,
    summarise_head_matrix,
)
from .statistics import (
    MINIMUM_BOOTSTRAP_UNITS,
    MINIMUM_FINITE_DRAW_FRACTION,
    bootstrap_unit_floor,
)

SCHEMA_VERSION = "r2_transfer_homology_control_v1"

#: Percent-identity band edges, fixed before the first search was run.  A record
#: is assigned to the unique band with ``low <= identity < high``; the top band
#: is closed at 100 so that an exact match has a home.  Do not retune.
STRATUM_EDGES: tuple[float, ...] = (0.0, 30.0, 70.0, 95.0, 100.000001)

#: Names carry their own boundaries so that a stratum label in a downstream table
#: cannot be read without them.
STRATUM_NAMES: tuple[str, ...] = (
    "lt30_no_detectable_homology",
    "id30_to_70_remote_homology",
    "id70_to_95_close_homology",
    "ge95_near_duplicate",
)

#: DIAMOND tabular fields, in the order they are requested and parsed.
#: ``nident`` and ``qlen`` are what the stratification actually uses; ``pident``
#: is kept because it is what a reader expects to see and the two differ sharply
#: when the corpus holds a fragment of the query (100% identical over 80% of the
#: query is not a near-duplicate of the query).  ``qstart``/``qend`` are kept
#: because a near-duplicate that does not cover the repeat span cannot explain
#: induction *on that span*.
DIAMOND_FIELDS: tuple[str, ...] = (
    "qseqid",
    "sseqid",
    "pident",
    "length",
    "nident",
    "qstart",
    "qend",
    "qlen",
    "slen",
    "evalue",
    "bitscore",
)

#: :data:`DIAMOND_FIELDS` plus the two aligned strings. A stratification needs
#: only the counts above; building a position-specific profile from the same
#: search needs the alignment itself, and re-deriving it from the counts is not
#: possible. Declared here rather than at the one call site that asks for it so
#: that the fields, their order and their parse stay in one place -- the search
#: command, the parser and the ``Hit`` record all read this module.
#:
#: ``qseq_gapped``/``sseq_gapped`` and not ``qseq``/``sseq``: DIAMOND's ``qseq``
#: is the aligned part of the query *with its gaps removed*, so on any alignment
#: carrying an indel the two strings have different lengths and cannot be walked
#: together. Measured on the first real search run through this module -- a
#: 541-column HSP returned a 485-character ``qseq`` against a 512-character
#: ``sseq`` -- and caught by :func:`~.profiles.build_profile`'s length check
#: rather than by reading the manual, which is the only reason it is recorded
#: here as a fact instead of shipped as a silent column shift.
ALIGNMENT_FIELDS: tuple[str, ...] = (*DIAMOND_FIELDS, "qseq_gapped", "sseq_gapped")

#: The one documented failure of :func:`.circuits.natural_repeat_probes` that is
#: a property of the record rather than of the configuration.  That function
#: raises ``RuntimeError`` in exactly one place and ``ValueError``/``TypeError``
#: everywhere else, so matching this fragment keeps every other fault fatal.  The
#: fragment excludes the probe-kind name, which the message interpolates and
#: which changes when a new repeat criterion is added.
_NO_PROBE_MESSAGE = "probe survived token alignment"


def _finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def sha256_file(path: Path, *, chunk: int = 1 << 22) -> str:
    """Streamed digest; the inputs here run to tens of gigabytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


# ------------------------------------------------------------------ the tool


@dataclass(frozen=True)
class DiamondTool:
    """A verified DIAMOND binary and the provenance needed to reproduce it."""

    executable: Path
    version: str
    binary_sha256: str
    tarball: Path
    tarball_sha256: str

    def record(self) -> dict[str, Any]:
        return {
            "executable": str(self.executable),
            "version": self.version,
            "binary_sha256": self.binary_sha256,
            "tarball": str(self.tarball),
            "tarball_sha256": self.tarball_sha256,
        }


def prepare_diamond(tarball: Path, checksum_file: Path, destination: Path) -> DiamondTool:
    """Verify the staged tarball against its checksum and extract it.

    The checksum is verified rather than assumed because two DIAMOND tarballs are
    staged side by side and only one has a published digest; extracting the wrong
    one would change the aligner without changing anything visible in the output.
    Extraction goes to a working location outside the repository so that a 25 GB
    database and a binary never enter version control.
    """

    tarball = Path(tarball)
    checksum_file = Path(checksum_file)
    destination = Path(destination)
    for path in (tarball, checksum_file):
        if not path.is_file():
            raise FileNotFoundError(f"{path} does not exist")

    fields = checksum_file.read_text(encoding="utf-8").split()
    if len(fields) < 1 or len(fields[0]) != 64:
        raise ValueError(f"{checksum_file} does not begin with a sha256 digest")
    expected = fields[0].lower()
    observed = sha256_file(tarball)
    if observed != expected:
        raise RuntimeError(
            f"{tarball} sha256 {observed} does not match the published {expected}"
        )

    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        binaries = [member for member in members if Path(member.name).name == "diamond"]
        if len(binaries) != 1:
            raise RuntimeError(
                f"{tarball} holds {len(binaries)} files named 'diamond'; expected exactly one"
            )
        archive.extractall(path=destination, members=binaries, filter="data")
    executable = destination / binaries[0].name
    if not executable.is_file():
        raise RuntimeError(f"extraction did not produce {executable}")
    executable.chmod(0o755)

    completed = subprocess.run(
        [str(executable), "version"], capture_output=True, text=True, check=True
    )
    match = re.search(r"diamond version ([0-9][0-9.]*)", completed.stdout)
    if match is None:
        raise RuntimeError(f"cannot parse a version from {completed.stdout!r}")
    return DiamondTool(
        executable=executable,
        version=match.group(1),
        binary_sha256=sha256_file(executable),
        tarball=tarball,
        tarball_sha256=observed,
    )


# --------------------------------------------------------------- the database


@dataclass(frozen=True)
class DiamondDatabase:
    """Exactly what was searched, and how much of the source it covers.

    ``coverage_fraction`` is the whole point of this record.  A partial database
    can only ever miss homology, and the direction that error pushes the reading
    is asymmetric (see the module docstring), so a report that does not state
    coverage cannot be interpreted at all.
    """

    path: Path
    source_fasta: Path
    source_records: int
    sequences: int
    letters: int
    makedb_command: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.source_records < 1 or self.sequences < 1 or self.letters < 1:
            raise ValueError("database record, sequence and letter counts must be positive")
        if self.sequences > self.source_records:
            raise ValueError(
                f"database holds {self.sequences} sequences but the source FASTA has "
                f"{self.source_records} records"
            )

    @property
    def coverage_fraction(self) -> float:
        return self.sequences / self.source_records

    def record(self) -> dict[str, Any]:
        source = Path(self.source_fasta)
        return {
            "database_path": str(self.path),
            "source_fasta": str(source),
            "source_fasta_bytes": source.stat().st_size,
            "source_fasta_records": self.source_records,
            "indexed_sequences": self.sequences,
            "indexed_letters": self.letters,
            "coverage_fraction": _finite(self.coverage_fraction, "database coverage"),
            "is_complete": self.sequences == self.source_records,
            "makedb_command": list(self.makedb_command),
        }


def count_fasta_records(path: Path, *, chunk: int = 1 << 24) -> tuple[int, int]:
    """``(records, residues)`` for a FASTA, in one pass and without decoding it.

    Both halves are returned because :func:`build_database` needs both to decide
    whether an existing index was built from *this* file. The record count alone
    cannot tell a half-built index from an index of a different corpus that
    happens to hold the same number of entries -- including the case that
    actually occurs, a source FASTA edited in place while its entry count stays
    put.

    ``residues`` counts every non-newline byte on a non-header line, which is
    what DIAMOND reports as ``Letters``. A sequence DIAMOND refuses to index
    also disappears from its ``Sequences`` count, so the two checks together
    admit an index only when it covers the same entries *and* the same residues
    as the file named beside it.

    Line state is carried across block boundaries, so a header whose newline
    ends one block and whose ``>`` starts the next is still counted once. On a
    24 GB corpus that boundary case happens a handful of times, which is small
    enough to look like a rounding difference and large enough to make the
    coverage fraction wrong.

    That invariant was *claimed* here and not delivered until EXP-R2-068. The
    inner scan ran ``while position <= len(block)``, so a block ending exactly
    on a newline took one extra iteration over an empty trailing segment, found
    no newline in it, and cleared ``at_line_start``. The next block's ``>`` then
    read as a continuation line: its record went uncounted and its header bytes
    were added to ``residues``. On ``'>r0\\nAAAA\\n>r1\\nCCCC\\n>r2\\nGGGG\\n'``
    -- true answer ``(3, 12)`` -- chunk 6 returned ``(2, 15)``, chunk 9 ``(1, 18)``
    and chunk 18 ``(2, 15)``. Read the loop bound as the invariant it enforces:
    a block is a sequence of complete lines plus at most one partial tail, and
    only a partial tail may clear ``at_line_start``.

    **No published number moved.** The shipped
    ``homology_control_unmasked/homology_assignment.json`` records
    ``source_fasta_records == indexed_sequences == 60315044`` at
    ``coverage_fraction 1.0``, so the 16 MiB default chunk never landed on a
    newline in the real UniRef50 file and EXP-R2-064's stratification stands.
    The defect was reachable, not reached.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist")
    records = 0
    residues = 0
    in_header = False
    at_line_start = True
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            position = 0
            while position < len(block):
                newline = block.find(b"\n", position)
                segment = block[position:newline] if newline != -1 else block[position:]
                if at_line_start and segment[:1] == b">":
                    in_header = True
                    records += 1
                if not in_header:
                    residues += len(segment) - segment.count(b"\r")
                if newline == -1:
                    at_line_start = False
                    break
                in_header = False
                at_line_start = True
                position = newline + 1
    if records < 1:
        raise RuntimeError(f"{path} contains no FASTA records")
    return records, residues


def _dbinfo(tool: DiamondTool, database: Path) -> tuple[int, int]:
    completed = subprocess.run(
        [str(tool.executable), "dbinfo", "--db", str(database)],
        capture_output=True,
        text=True,
        check=True,
    )
    values: dict[str, int] = {}
    for key in ("Sequences", "Letters"):
        match = re.search(rf"^\s*{key}\s+(\d+)\s*$", completed.stdout, flags=re.MULTILINE)
        if match is None:
            raise RuntimeError(f"cannot parse {key} from `diamond dbinfo` output")
        values[key] = int(match.group(1))
    return values["Sequences"], values["Letters"]


def build_database(
    tool: DiamondTool,
    source_fasta: Path,
    database: Path,
    *,
    threads: int,
    tmpdir: Path,
    rebuild: bool = False,
) -> DiamondDatabase:
    """Index the corpus, or adopt an existing index after checking it.

    An existing index is adopted only if its own ``dbinfo`` counts match the
    source FASTA on **both** axes: sequences against records, and letters
    against residues.  A half-built or differently-built database would
    otherwise be searched silently and would answer a different question from
    the one the JSON claims was asked.

    The letter check was added by EXP-R2-067.  ``_dbinfo`` has always returned
    ``letters`` and ``DiamondDatabase.record`` has always published it as
    ``indexed_letters``, but nothing compared it, so adoption turned on the
    record count alone -- a key that cannot separate "this index was built from
    this file" from "this index was built from a different corpus with the same
    number of entries", which is what a source FASTA edited in place looks like.
    That is the shape this programme has already paid for once: a resume verdict
    that reported complete against inputs that no longer matched and still
    passed its own checksum.  Here the artefact would have named the new FASTA
    while every stratum came from the old index.
    """

    source_fasta = Path(source_fasta)
    database = Path(database)
    tmpdir = Path(tmpdir)
    if threads < 1:
        raise ValueError("threads must be positive")
    if not source_fasta.is_file():
        raise FileNotFoundError(f"{source_fasta} does not exist")
    source_records, source_residues = count_fasta_records(source_fasta)

    command = (
        str(tool.executable),
        "makedb",
        "--in",
        str(source_fasta),
        "--db",
        str(database),
        "--threads",
        str(threads),
        "--tmpdir",
        str(tmpdir),
    )
    if rebuild or not database.is_file():
        database.parent.mkdir(parents=True, exist_ok=True)
        tmpdir.mkdir(parents=True, exist_ok=True)
        subprocess.run(list(command), check=True, capture_output=True, text=True)

    sequences, letters = _dbinfo(tool, database)
    if sequences != source_records:
        raise RuntimeError(
            f"{database} indexes {sequences} of {source_records} source records; it was "
            "not built from this FASTA in full. Rebuild it, or record the subset "
            "explicitly -- an unrecorded partial database makes the strata "
            "uninterpretable."
        )
    if letters != source_residues:
        raise RuntimeError(
            f"{database} indexes {sequences} sequences holding {letters} residues, but "
            f"{source_fasta} holds {source_residues} residues across the same "
            f"{source_records} records. The record counts agree and the contents do "
            "not, so this index was built from a different corpus than the one named "
            "beside it. Rebuild it with --rebuild-db; adopting it would attribute "
            "every homology stratum to a file that did not produce it."
        )
    return DiamondDatabase(
        path=database,
        source_fasta=source_fasta,
        source_records=source_records,
        sequences=sequences,
        letters=letters,
        makedb_command=command,
    )


# ------------------------------------------------------------------ the search


def write_query_fasta(cohort: Cohort, path: Path) -> list[str]:
    """One FASTA record per cohort record, named by its cohort index.

    The identifier is positional because the stratification has to be joined back
    onto the cohort by position; accession-based naming would make that join
    depend on a header format the cohort does not promise to preserve.
    """

    if cohort.kind != "protein":
        raise ValueError(f"cohort {cohort.name!r} is not a protein cohort")
    if not cohort.records:
        raise ValueError(f"cohort {cohort.name!r} is empty")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    identifiers = [f"q{index:05d}" for index in range(len(cohort.records))]
    with path.open("w", encoding="utf-8") as handle:
        for identifier, record in zip(identifiers, cohort.records):
            handle.write(f">{identifier}\n{record}\n")
    return identifiers


@dataclass(frozen=True)
class Hit:
    """One DIAMOND HSP, with identity expressed over the query rather than the HSP."""

    query: str
    subject: str
    pident: float
    length: int
    nident: int
    qstart: int
    qend: int
    qlen: int
    slen: int
    evalue: float
    bitscore: float
    #: The aligned query and subject strings *including* gap characters, so the
    #: two are the same length and can be walked column by column. Present only
    #: when the search was asked for :data:`ALIGNMENT_FIELDS`; optional because
    #: every existing caller stratifies on the counts above and pays nothing for
    #: an alignment it does not read, and a consumer that needs them must check.
    qseq_gapped: str | None = None
    sseq_gapped: str | None = None

    @property
    def identity_over_query(self) -> float:
        """Percent of the *query* that is identically matched.

        ``pident`` is identity within the aligned region, so a corpus entry that
        is a 60%-length fragment of the query scores 100 on ``pident`` while
        being nothing like a stored copy of the query.  Stratifying on ``pident``
        would put such a record in the near-duplicate bin and destroy the
        contrast this control depends on.
        """

        return 100.0 * self.nident / self.qlen


def run_diamond_blastp(
    tool: DiamondTool,
    database: DiamondDatabase,
    query_fasta: Path,
    output_tsv: Path,
    *,
    threads: int,
    sensitivity: str,
    evalue: float,
    max_target_seqs: int,
    fields: Sequence[str] = DIAMOND_FIELDS,
) -> tuple[list[str], str]:
    """Search the cohort against the corpus; return the command and the log tail.

    ``--very-sensitive`` is the default at the call site because the claim that
    matters most is a *negative* one -- that a record has no close relative in the
    corpus -- and a fast search cannot support a negative.  The query set is tens
    of sequences against tens of millions, so runtime is set by the database scan
    and sensitivity is nearly free.

    ``--masking 0`` is not a tuning choice; it is the difference between this
    control measuring what it claims to and measuring the opposite.  DIAMOND
    masks low-complexity and *repetitive* query regions by default, and this
    cohort is selected for containing internal tandem repeats.  With masking on,
    the HSP stops at the repeat: cohort record 0 of the 2026-07-28 run is
    byte-identical to ``UniRef50_Q3E8Z8`` over all 732 residues, and DIAMOND
    reported ``pident 100`` over 607 aligned residues, giving
    ``identity_over_query`` 82.9 and placing a verbatim member of ProtGPT2's
    pretraining corpus in the *close homology* bin rather than the near-duplicate
    one.  Five of forty-eight exact-cohort records were mis-binned that way, all
    in the same direction, and all into the bin with the highest measured
    induction -- which is exactly the pattern that would manufacture the
    "memorisation does not explain induction" reading this control exists to
    test.  The bias is also the reverse of the upward bias the module docstring
    declares, and it grows with repeat content, the one property the cohort is
    selected on.

    ``fields`` selects the tabular columns and defaults to
    :data:`DIAMOND_FIELDS`, which is what a stratification consumes. A caller
    that needs the alignments themselves -- a position-specific profile cannot
    be rebuilt from the counts -- passes :data:`ALIGNMENT_FIELDS`. The same list
    has to reach :func:`parse_hits`, so it is a parameter of both rather than a
    literal in either.
    """

    query_fasta = Path(query_fasta)
    output_tsv = Path(output_tsv)
    if not query_fasta.is_file():
        raise FileNotFoundError(f"{query_fasta} does not exist")
    if threads < 1 or evalue <= 0 or max_target_seqs < 1:
        raise ValueError("invalid DIAMOND search parameters")
    _checked_fields(fields)
    allowed = {"fast", "default", "sensitive", "mid-sensitive", "more-sensitive",
               "very-sensitive", "ultra-sensitive"}
    if sensitivity not in allowed:
        raise ValueError(f"unknown sensitivity {sensitivity!r}; known: {sorted(allowed)}")

    command = [
        str(tool.executable),
        "blastp",
        "--db",
        str(database.path),
        "--query",
        str(query_fasta),
        "--out",
        str(output_tsv),
        "--outfmt",
        "6",
        *fields,
        "--evalue",
        repr(evalue),
        "--max-target-seqs",
        str(max_target_seqs),
        "--threads",
        str(threads),
        # See the docstring: repeat masking truncates the alignment of exactly
        # the records this cohort is built from.
        "--masking",
        "0",
    ]
    if sensitivity != "default":
        command.append(f"--{sensitivity}")
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    log = (completed.stdout + completed.stderr).strip().splitlines()
    return command, "\n".join(log[-12:])


#: How each DIAMOND column becomes a :class:`Hit` attribute. One table, so the
#: search command, the parser and the record cannot drift apart.
_FIELD_ATTRIBUTES: dict[str, tuple[str, Any]] = {
    "qseqid": ("query", str),
    "sseqid": ("subject", str),
    "pident": ("pident", float),
    "length": ("length", int),
    "nident": ("nident", int),
    "qstart": ("qstart", int),
    "qend": ("qend", int),
    "qlen": ("qlen", int),
    "slen": ("slen", int),
    "evalue": ("evalue", float),
    "bitscore": ("bitscore", float),
    "qseq_gapped": ("qseq_gapped", str),
    "sseq_gapped": ("sseq_gapped", str),
}


def _checked_fields(fields: Sequence[str]) -> tuple[str, ...]:
    """Refuse a field list this module cannot parse into a complete ``Hit``.

    Every column in :data:`DIAMOND_FIELDS` is required because every consumer
    reads them; anything outside :data:`_FIELD_ATTRIBUTES` has no home on the
    record and would be silently discarded, which is the shape that lets a
    search be asked for a column nothing ever reads.
    """

    requested = tuple(str(field) for field in fields)
    if len(set(requested)) != len(requested):
        raise ValueError(f"duplicate DIAMOND output fields in {requested}")
    unknown = [field for field in requested if field not in _FIELD_ATTRIBUTES]
    if unknown:
        raise ValueError(
            f"DIAMOND fields {unknown} have no place on a Hit; known fields are "
            f"{sorted(_FIELD_ATTRIBUTES)}"
        )
    missing = [field for field in DIAMOND_FIELDS if field not in requested]
    if missing:
        raise ValueError(
            f"DIAMOND fields {missing} are required by every consumer of a Hit and "
            "are not in the requested output"
        )
    return requested


def parse_hits(output_tsv: Path, *, fields: Sequence[str] = DIAMOND_FIELDS) -> list[Hit]:
    """Read the tabular output, failing on any row that is not the declared shape.

    ``fields`` must be the list the search was run under; it defaults to
    :data:`DIAMOND_FIELDS`, so an existing caller reads exactly what it did
    before.
    """

    requested = _checked_fields(fields)
    rows: list[Hit] = []
    with Path(output_tsv).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != len(requested):
                raise ValueError(
                    f"{output_tsv}:{number} has {len(parts)} fields, expected "
                    f"{len(requested)}"
                )
            values = {}
            for field, part in zip(requested, parts):
                attribute, cast = _FIELD_ATTRIBUTES[field]
                values[attribute] = cast(part)
            rows.append(Hit(**values))
    return rows


# ------------------------------------------------------------ stratification


def assign_stratum(identity: float) -> str:
    """Band a percent identity, using the boundaries fixed in this module."""

    value = _finite(identity, "percent identity")
    if not 0.0 <= value <= 100.0:
        raise ValueError(f"percent identity {value} is outside [0, 100]")
    for index, name in enumerate(STRATUM_NAMES):
        if STRATUM_EDGES[index] <= value < STRATUM_EDGES[index + 1]:
            return name
    raise RuntimeError(f"percent identity {value} fell through every stratum")


@dataclass(frozen=True)
class HomologyAssignment:
    """One cohort record's closest relative in the corpus, and its stratum."""

    record_index: int
    query_id: str
    query_length: int
    n_hits: int
    max_identity_over_query: float
    max_pident: float
    best_subject: str | None
    best_bitscore: float | None
    best_qstart: int | None
    best_qend: int | None
    best_hit_spans_repeat: bool | None
    stratum: str
    #: The best hit is essentially exact over a length-matched subject yet
    #: covers well under the whole query. See :func:`truncated_alignment`.
    best_hit_looks_truncated: bool | None = None
    #: The hit list for this query reached ``--max-target-seqs``, so
    #: ``max_identity_over_query`` is a maximum over the reported hits and not
    #: over the corpus.
    hit_list_saturated: bool | None = None

    def record(self) -> dict[str, Any]:
        return {
            "record_index": self.record_index,
            "query_id": self.query_id,
            "query_length": self.query_length,
            "n_hits": self.n_hits,
            "max_identity_over_query": _finite(
                self.max_identity_over_query, "max identity"
            ),
            "max_pident": _finite(self.max_pident, "max pident"),
            "best_subject": self.best_subject,
            "best_bitscore": (
                None if self.best_bitscore is None else _finite(self.best_bitscore, "bitscore")
            ),
            "best_qstart": self.best_qstart,
            "best_qend": self.best_qend,
            "best_hit_spans_repeat": self.best_hit_spans_repeat,
            "best_hit_looks_truncated": self.best_hit_looks_truncated,
            "hit_list_saturated": self.hit_list_saturated,
            "stratum": self.stratum,
        }


#: Cohort metadata that is one entry per record and must be sliced whenever the
#: records are. Written by ``src.transfer.circuits`` when it builds a repeat
#: cohort; named here because :func:`sub_cohort` is the only place the pairing
#: can be broken, and it refuses any other per-record list rather than copying
#: it through.
PER_RECORD_METADATA_KEYS = ("repeats", "repeat_stats", "ec_labels")

#: Query coverage below which an alignment is treated as truncated when it is
#: otherwise a perfect, length-matched match. A genuine partial homologue whose
#: subject is the same length as the query and which is 100% identical over the
#: aligned part is not a thing that happens; a masked-out repeat is.
TRUNCATION_COVERAGE_LIMIT = 95.0

#: How close two lengths must be to count as length-matched, as a fraction.
TRUNCATION_LENGTH_TOLERANCE = 0.02

#: ``pident`` at or above which the aligned region is treated as exact.
TRUNCATION_PIDENT_FLOOR = 99.0


#: How :func:`assign_homology` responds to a truncated-looking alignment.
#:
#: ``any``
#:     stop on any hit at any rank that :func:`truncated_alignment` flags. The
#:     original behaviour and the default, so no existing caller moves.
#: ``stratum_changing``
#:     stop only on a flagged hit whose repair *could* move its record into a
#:     higher stratum, by :func:`truncation_raises_stratum`.
#:
#: **The second rule exists because the first one's premise was measured false.**
#: :func:`truncated_alignment`'s docstring argues that an alignment which is
#: exact over a length-matched subject and covers well under the whole query
#: "does not describe any biological relationship". At 48 queries against 100
#: targets that held. At 12 ProteinGym wild types against 5000 targets each
#: (2026-08-07, 22399 HSPs, ``--masking 0`` throughout) it does not: **11 of the
#: 22399 alignments are flagged, all of them against one query -- human
#: calmodulin -- and every one is ordinary biology.** They are other calmodulin
#: entries of length 147-151 that are 100% identical over 139 of the query's 149
#: residues with a terminal offset of about ten residues, which is exactly what a
#: hyper-conserved protein with variable termini looks like. The same query's own
#: verbatim corpus record, ``UniRef50_P0DP23``, is found at 100% identity over all
#: 149 residues in the same search, so nothing was truncated and the run would
#: have stopped on a false alarm.
#:
#: The refinement keeps every case the guard was earned on. EXP-R2-061's record
#: aligned 607 of 732 residues against a 732-residue subject, giving observed
#: identity 82.9 in ``id70_to_70..95`` against a potential 100.0 in
#: ``ge95_near_duplicate`` -- a stratum change, so it still stops the run under
#: both rules.
TRUNCATION_RULES = ("any", "stratum_changing")


def potential_identity_over_query(hit: Hit) -> float:
    """Identity over the query this alignment would reach if it were not truncated.

    An upper bound, and deliberately the most generous one: every query residue
    the alignment does not cover is assumed to have matched. That is what makes
    it safe to compare against a stratum boundary -- a truncation that cannot
    reach a higher stratum even under the most favourable repair cannot have
    caused a mis-binning.
    """

    if hit.qlen < 1:
        raise ValueError("a hit against a zero-length query has no identity")
    aligned = hit.qend - hit.qstart + 1
    if aligned < 1 or aligned > hit.qlen:
        raise ValueError(
            f"alignment covers {aligned} of a {hit.qlen}-residue query, which is not "
            "a query span"
        )
    return 100.0 * (hit.nident + hit.qlen - aligned) / hit.qlen


def truncation_raises_stratum(hit: Hit, observed_identity: float) -> bool:
    """Could repairing this truncation move its record into a higher stratum?"""

    potential = potential_identity_over_query(hit)
    if potential <= observed_identity:
        return False
    return assign_stratum(potential) != assign_stratum(observed_identity)


def truncated_alignment(hit: Hit) -> bool:
    """Does this alignment look truncated rather than partial?

    The signature is: the aligned region is essentially exact, the subject is
    essentially the same length as the query, and yet the alignment covers well
    under the whole query. A real partial homologue whose subject happens to
    match the query's length would have to be identical over a fragment and
    unalignable over the rest, which does not describe any biological
    relationship. It does describe an aligner that stopped at a masked region --
    the failure that put five verbatim corpus members into the close-homology bin
    and left no trace in any artefact.

    Reported rather than corrected: the repair is to search without masking, and
    silently re-binning a truncated hit would be inventing an alignment.
    """

    if hit.qlen < 1 or hit.slen < 1:
        return False
    length_ratio = abs(hit.slen - hit.qlen) / hit.qlen
    return bool(
        hit.pident >= TRUNCATION_PIDENT_FLOOR
        and length_ratio <= TRUNCATION_LENGTH_TOLERANCE
        and hit.identity_over_query < TRUNCATION_COVERAGE_LIMIT
    )


def assign_homology(
    cohort: Cohort,
    identifiers: Sequence[str],
    hits: Sequence[Hit],
    *,
    max_target_seqs: int | None = None,
    truncation_rule: str = "any",
) -> list[HomologyAssignment]:
    """Join hits back onto the cohort and band every record.

    A record with no hit is assigned identity 0 and lands in the lowest stratum.
    That is the correct reading of an ``e``-value-filtered miss, but it is also
    the only place where an incomplete database could put a memorised sequence in
    the wrong bin, so ``n_hits`` is carried through to the report rather than
    collapsed into the identity.

    A best hit that is essentially exact over a length-matched subject but covers
    well under the whole query stops the run. That combination is not a partial
    homologue, it is a truncated alignment, and it puts a verbatim member of the
    pretraining corpus into a lower stratum -- the direction that makes
    memorisation look like a weaker explanation than it is, which is the reading
    this control exists to test. It is refused rather than flagged because a
    stratification built on truncated alignments is not a weaker measurement, it
    is a different one.

    ``truncation_rule`` selects which flagged alignments stop the run; see
    :data:`TRUNCATION_RULES`. The default is the original behaviour, so no
    existing caller changes. ``"stratum_changing"`` is for a caller that searches
    thousands of targets per query, where the flag's measured false-positive
    class -- a hyper-conserved protein whose relatives are exact over a
    terminally-offset span -- makes the strict rule unrunnable without weakening
    what it protects.
    """

    if truncation_rule not in TRUNCATION_RULES:
        raise ValueError(
            f"unknown truncation rule {truncation_rule!r}; rules are {list(TRUNCATION_RULES)}"
        )
    if len(identifiers) != len(cohort.records):
        raise ValueError("identifier list does not match the cohort length")
    repeats = cohort.metadata.get("repeats")
    if repeats is None or len(repeats) != len(cohort.records):
        raise ValueError(f"cohort {cohort.name!r} carries no per-record repeat coordinates")

    by_query: dict[str, list[Hit]] = {identifier: [] for identifier in identifiers}
    for hit in hits:
        if hit.query not in by_query:
            raise ValueError(f"hit for unknown query {hit.query!r}")
        by_query[hit.query].append(hit)

    assignments: list[HomologyAssignment] = []
    for index, identifier in enumerate(identifiers):
        record = cohort.records[index]
        found = by_query[identifier]
        for hit in found:
            if hit.qlen != len(record):
                raise ValueError(
                    f"{identifier}: DIAMOND reports qlen {hit.qlen} for a "
                    f"{len(record)}-residue record"
                )
        if not found:
            assignments.append(
                HomologyAssignment(
                    record_index=index,
                    query_id=identifier,
                    query_length=len(record),
                    n_hits=0,
                    max_identity_over_query=0.0,
                    max_pident=0.0,
                    best_subject=None,
                    best_bitscore=None,
                    best_qstart=None,
                    best_qend=None,
                    best_hit_spans_repeat=None,
                    best_hit_looks_truncated=None,
                    hit_list_saturated=False,
                    stratum=assign_stratum(0.0),
                )
            )
            continue
        best = max(found, key=lambda hit: (hit.identity_over_query, hit.bitscore))
        first, second, span = (int(value) for value in repeats[index])
        # DIAMOND coordinates are 1-based inclusive; the repeat coordinates are
        # 0-based half-open over the record.
        spans_repeat = best.qstart - 1 <= first and best.qend >= second + span
        assignments.append(
            HomologyAssignment(
                record_index=index,
                query_id=identifier,
                query_length=len(record),
                n_hits=len(found),
                max_identity_over_query=best.identity_over_query,
                max_pident=max(hit.pident for hit in found),
                best_subject=best.subject,
                best_bitscore=best.bitscore,
                best_qstart=best.qstart,
                best_qend=best.qend,
                best_hit_spans_repeat=bool(spans_repeat),
                best_hit_looks_truncated=truncated_alignment(best),
                # ``n_hits`` counts reported HSP rows and ``--max-target-seqs``
                # caps *subject sequences*, so comparing one against the other
                # compares two different things: a subject reported under three
                # HSPs contributes three rows and one sequence, and the list then
                # reads as saturated at a third of the cap. Counted over distinct
                # subjects, which is the unit the cap is expressed in. On the
                # 2026-07-29 unmasked run every saturated record has exactly one
                # HSP per subject, so the two agree there and no published
                # saturation flag moves; the fix is for the search that does not.
                hit_list_saturated=(
                    None
                    if max_target_seqs is None
                    else len({hit.subject for hit in found}) >= max_target_seqs
                ),
                stratum=assign_stratum(best.identity_over_query),
            )
        )
    # Every hit, not only the best one. A truncated near-duplicate is a hit whose
    # alignment stopped at a masked region, and masking is exactly what can push
    # it *below* an untruncated but genuinely more distant relative -- so the
    # record whose stratification is wrong is precisely the record whose best hit
    # is not the truncated one. Inspecting only the best hit therefore looked
    # hardest at the cases that need it least. The 2026-07-29 unmasked run has no
    # truncated alignment at any rank, so no published stratification moves.
    observed = {assignment.query_id: assignment.max_identity_over_query
                for assignment in assignments}
    truncated = [
        (identifier, hit.subject, hit.identity_over_query)
        for identifier in identifiers
        for hit in by_query[identifier]
        if truncated_alignment(hit)
        and (
            truncation_rule == "any"
            or truncation_raises_stratum(hit, observed[identifier])
        )
    ]
    if truncated:
        raise RuntimeError(
            f"{len(truncated)} alignments over {len(assignments)} records are "
            f"exact over a length-matched subject yet cover under "
            f"{TRUNCATION_COVERAGE_LIMIT}% of the query, e.g. {truncated[:3]}. That is "
            "a truncated alignment, not a partial homologue, and it under-bins a "
            "verbatim corpus member. Re-run the search with --masking 0: DIAMOND masks "
            "repetitive query regions by default and this cohort is selected for "
            f"internal tandem repeats (truncation_rule={truncation_rule!r})"
        )
    return assignments


def stratum_integrity(
    assignments: Sequence[HomologyAssignment],
) -> dict[str, dict[str, int]]:
    """Per stratum, how many records the stratum's own evidence does not support.

    ``best_hit_spans_repeat`` is computed for every record and was, until this
    function existed, never used: :data:`DIAMOND_FIELDS` explains that
    ``qstart``/``qend`` are kept because "a near-duplicate that does not cover the
    repeat span cannot explain induction *on that span*", and then the record was
    stratified on identity alone. Twenty-seven per cent of the 2026-07-28 exact
    cohort had a best hit that does not cover its repeat, and every one of them
    was placed in a stratum as though its corpus match explained the repeat.

    This does not re-stratify -- that would change the estimand mid-programme --
    it reports the count so that a stratum gradient can be read against how much
    of each stratum the memorisation hypothesis actually reaches.
    """

    report = {
        name: {
            "records": 0,
            "best_hit_does_not_span_repeat": 0,
            "hit_list_saturated": 0,
            "no_hit": 0,
        }
        for name in STRATUM_NAMES
    }
    for assignment in assignments:
        entry = report[assignment.stratum]
        entry["records"] += 1
        if assignment.n_hits == 0:
            entry["no_hit"] += 1
        elif assignment.best_hit_spans_repeat is False:
            entry["best_hit_does_not_span_repeat"] += 1
        if assignment.hit_list_saturated:
            entry["hit_list_saturated"] += 1
    return report


def stratum_counts(assignments: Sequence[HomologyAssignment]) -> dict[str, int]:
    """Achieved bin counts for every declared stratum, including the empty ones."""

    counts = {name: 0 for name in STRATUM_NAMES}
    for assignment in assignments:
        counts[assignment.stratum] += 1
    return counts


def sequence_groups(cohort: Cohort) -> list[int]:
    """Group identifier per record; records with an identical sequence share one.

    The EC-labelled Swiss-Prot source carries one record per (protein, EC number)
    pair, so a protein annotated with several EC numbers appears several times
    with a byte-identical sequence.  Those copies are one observation, not
    several: they produce the same tokens, the same attention and the same probe.
    Counting them as independent would narrow every interval in this control by
    roughly the square root of the duplication factor, and the duplication is not
    spread evenly across the strata -- which is precisely the direction that would
    manufacture a stratum contrast out of nothing.

    Group identifiers are assigned in order of first appearance, so they are a
    deterministic function of the cohort.
    """

    order: dict[str, int] = {}
    groups: list[int] = []
    for record in cohort.records:
        key = hashlib.sha256(record.encode()).hexdigest()
        if key not in order:
            order[key] = len(order)
        groups.append(order[key])
    return groups


def distinct_stratum_counts(
    assignments: Sequence[HomologyAssignment], groups: Sequence[int]
) -> dict[str, int]:
    """Bin counts after collapsing byte-identical sequences to one observation."""

    if len(assignments) != len(groups):
        raise ValueError("assignments and group identifiers are not aligned")
    seen: dict[str, set[int]] = {name: set() for name in STRATUM_NAMES}
    for assignment, group in zip(assignments, groups):
        seen[assignment.stratum].add(int(group))
    return {name: len(values) for name, values in seen.items()}


def sub_cohort(cohort: Cohort, indices: Sequence[int], *, name: str) -> Cohort:
    """A cohort restricted to ``indices``, carrying its per-record metadata along.

    ``repeats``, ``repeat_stats`` and ``ec_labels`` are positional parallel
    arrays; slicing the records without slicing them would silently mis-pair a
    repeat coordinate, its statistics or an EC conditioning tag with the wrong
    sequence.

    Any *other* list-valued metadata whose length equals the record count is
    refused rather than carried through unsliced. The previous version named two
    keys and copied everything else verbatim, so a one-record sub-cohort carried
    a full-length ``repeat_stats`` whose entry zero described a different
    protein. Nothing read it, which is the only reason it was not a live defect.
    """

    if not indices:
        raise ValueError("cannot build an empty sub-cohort")
    total = len(cohort.records)
    if any(not 0 <= index < total for index in indices):
        raise ValueError("sub-cohort index out of range")
    metadata = dict(cohort.metadata)
    for key in PER_RECORD_METADATA_KEYS:
        values = cohort.metadata.get(key)
        if values is None:
            continue
        if len(values) != total:
            raise ValueError(f"cohort {cohort.name!r}: {key} is not aligned to its records")
        metadata[key] = [values[index] for index in indices]
    unsliced = sorted(
        key
        for key, value in cohort.metadata.items()
        if key not in PER_RECORD_METADATA_KEYS
        and isinstance(value, list)
        and len(value) == total
    )
    if unsliced:
        raise ValueError(
            f"cohort {cohort.name!r}: metadata {unsliced} is one entry per record but "
            "is not in PER_RECORD_METADATA_KEYS, so a sub-cohort would carry it "
            "mis-paired with the records it kept"
        )
    return Cohort(
        name,
        cohort.kind,
        [cohort.records[index] for index in indices],
        cohort.min_symbols,
        cohort.max_symbols,
        metadata,
    )


def probe_for_record(
    arm: Arm, cohort: Cohort, index: int, *, max_tokens: int
) -> RepeatProbe | None:
    """The natural-repeat probe for one record, or ``None`` if it has none.

    :func:`.circuits.natural_repeat_probes` drops a record whose two repeat
    copies are tokenised differently and raises when nothing survives.  Called on
    a one-record cohort that raise carries a *result* -- this record is not
    measurable on this arm -- rather than a fault, and it is converted here and
    counted in the report.  Every other ``RuntimeError`` is re-raised: the
    conversion is keyed to the one documented message, so a genuine fault cannot
    be absorbed by it.

    Scoring record by record rather than cohort by cohort is what lets the same
    forward passes be re-partitioned by stratum and bootstrapped, and it is exact
    -- :func:`verify_pooling` checks that against the estimator itself.
    """

    single = sub_cohort(cohort, [index], name=f"{cohort.name}#{index}")
    try:
        probes = natural_repeat_probes(arm, single, max_tokens=max_tokens)
    except RuntimeError as error:
        if _NO_PROBE_MESSAGE not in str(error):
            raise
        return None
    if len(probes) != 1:
        raise RuntimeError(f"{arm.name}: one record yielded {len(probes)} probes")
    return probes[0]


# ------------------------------------------------------------------- scoring


@dataclass(frozen=True)
class ProbeScore:
    """Per-probe attention *sums*, kept unnormalised so that subsets pool exactly."""

    record_index: int
    scored_positions: int
    uniform_sum: float
    coverage: float
    repeat_symbols: int
    sums: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        if self.scored_positions < 1:
            raise ValueError("a probe must contribute at least one scored position")
        if set(self.sums) != {"prefix_matching", "same_token", "offset_two"}:
            raise ValueError("probe score is missing one of the three alignment statistics")


def score_probe(arm: Arm, probe: RepeatProbe, record_index: int) -> ProbeScore:
    """Score one probe with the shared estimator and undo its normalisation."""

    result = attention_alignment_scores(arm, [probe], batch_size=1)
    scored = int(result["scored_query_positions"])
    return ProbeScore(
        record_index=record_index,
        scored_positions=scored,
        uniform_sum=float(result["uniform_baseline"]) * scored,
        coverage=float(result["mean_coverage"]),
        # Rounded, not truncated. ``int()`` floors, so a 39.9-residue repeat was
        # recorded as 39 and a 39.1-residue one also as 39, which is a
        # half-symbol downward bias on the covariate ``covariate_analysis``
        # partials induction strength against. It is a rank statistic, so the
        # bias matters only where it changes an ordering -- and truncation
        # changes orderings only by collapsing neighbours, which is precisely
        # where a partial correlation is decided.
        repeat_symbols=int(round(float(result["mean_repeat_symbols"]))),
        sums={key: np.asarray(value, dtype=np.float64) * scored
              for key, value in result["scores"].items()},
    )


def pool_scores(scores: Sequence[ProbeScore]) -> dict[str, Any]:
    """Combine per-probe sums into the same means the estimator would have returned."""

    if not scores:
        raise ValueError("cannot pool an empty score list")
    scored = sum(score.scored_positions for score in scores)
    if scored < 1:
        raise RuntimeError("pooled probes contributed no scored positions")
    keys = ("prefix_matching", "same_token", "offset_two")
    means = {
        key: sum(score.sums[key] for score in scores) / scored for key in keys
    }
    return {
        "n_probes": len(scores),
        "scored_query_positions": scored,
        "uniform_baseline": _finite(
            sum(score.uniform_sum for score in scores) / scored, "uniform baseline"
        ),
        "mean_coverage": _finite(
            float(np.mean([score.coverage for score in scores])), "probe coverage"
        ),
        "mean_repeat_symbols": _finite(
            float(np.mean([score.repeat_symbols for score in scores])), "repeat length"
        ),
        "scores": means,
    }


def verify_pooling(
    arm: Arm, probes: Sequence[RepeatProbe], scores: Sequence[ProbeScore], *, tolerance: float
) -> float:
    """Check per-probe pooling against a direct estimator call on the same probes.

    Pooling sums and renormalising *should* be the identity, but "should" is how
    an off-by-one in a weighting turns into a stratum effect that looks real.
    Both sides run the same probes at batch size one, so the only difference is
    the order of the additions; anything above floating-point noise is a defect.
    Returns the observed relative error.
    """

    if not probes or len(probes) != len(scores):
        raise ValueError("verification needs a non-empty, aligned probe and score list")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    direct = attention_alignment_scores(arm, list(probes), batch_size=1)
    pooled = pool_scores(list(scores))
    error = 0.0
    for key, reference in direct["scores"].items():
        reference = np.asarray(reference, dtype=np.float64)
        scale = float(np.abs(reference).max())
        if scale <= 0:
            raise RuntimeError(f"{arm.name}: reference {key} matrix is all zero")
        error = max(error, float(np.abs(pooled["scores"][key] - reference).max()) / scale)
    if error > tolerance:
        raise RuntimeError(
            f"{arm.name}: per-probe pooling disagrees with the shared estimator "
            f"(relative error {error:.3e} > {tolerance})"
        )
    return error


# ------------------------------------------------------------ stratum reports


def _selected_heads(prefix_matching: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    return [
        (int(layer), int(head))
        for layer, head in np.argwhere(prefix_matching >= threshold)
    ]


def ov_over_heads(
    copying: Mapping[str, np.ndarray], heads: Sequence[tuple[int, int]]
) -> dict[str, Any]:
    """Mean OV copying score over a named set of heads.

    The OV matrices themselves are a function of the weights and the sampled
    token support alone, so they do not vary by stratum: what varies is *which*
    heads a stratum's prefix-matching picks out.  Reporting the OV score of the
    stratum's own induction heads is therefore the only stratum-specific reading
    the OV circuit admits, and it is the one that answers the question -- a
    retrieval head need not have a copying OV circuit, a real induction head
    must.
    """

    if not heads:
        # The same keys as the populated branch, all null. The empty branch used
        # to emit three keys where the populated one emits five, so a consumer
        # reading ``diagonal_fraction_max`` got a ``KeyError`` for exactly the
        # arms that select no head -- ZymCTRL selects none at threshold 0.10 --
        # and a consumer using ``.get`` silently read ``None`` as a value rather
        # than as an absent column. A schema that changes shape with the result
        # is not a schema.
        return {
            "n_heads": 0,
            "diagonal_fraction_mean": None,
            "diagonal_fraction_max": None,
            "mean_normalised_rank_mean": None,
            "mean_normalised_rank_max": None,
        }
    rows = np.asarray([layer for layer, _ in heads], dtype=np.int64)
    columns = np.asarray([head for _, head in heads], dtype=np.int64)
    out: dict[str, Any] = {"n_heads": len(heads)}
    for key in ("diagonal_fraction", "mean_normalised_rank"):
        values = np.asarray(copying[key], dtype=np.float64)[rows, columns]
        out[f"{key}_mean"] = _finite(float(values.mean()), f"{key} over selected heads")
        out[f"{key}_max"] = _finite(float(values.max()), f"{key} max over selected heads")
    return out


def representative_scores(
    scores: Sequence[ProbeScore], group_of: Mapping[int, int]
) -> list[ProbeScore]:
    """One probe per distinct sequence, taken in cohort order.

    Byte-identical records give byte-identical probes, so the choice of
    representative is not a choice at all; what it changes is the weight the
    duplicated sequence carries.  See :func:`sequence_groups` for why that
    weight has to be one.
    """

    seen: dict[int, ProbeScore] = {}
    for score in scores:
        if score.record_index not in group_of:
            raise KeyError(f"no sequence group recorded for record {score.record_index}")
        group = int(group_of[score.record_index])
        if group not in seen:
            seen[group] = score
    return list(seen.values())


def bootstrap_stratum(
    scores: Sequence[ProbeScore],
    *,
    threshold: float,
    n_heads: int,
    resamples: int,
    seed: int,
    alpha: float = 0.05,
    minimum_units: int = MINIMUM_BOOTSTRAP_UNITS,
) -> dict[str, Any]:
    """Percentile intervals over the probe set for the two headline quantities.

    The resampling unit is the probe, not the query position, because query
    positions within one sequence share a repeat and are not independent.
    Callers pass the *representative* probe set (see
    :func:`representative_scores`), so the effective unit is the distinct
    sequence and duplicated records cannot narrow the interval.

    **A percentile interval is refused below ``minimum_units``, and it used to be
    the opposite of wide.**  The docstring previously claimed that a handful of
    sequences gives an interval that is "wide and honest rather than
    informative".  That is not what a percentile bootstrap does at small ``n``:
    measured relative widths of 9-15% at four units against 27% at four hundred,
    in the same artefact, and two ``consistent_with_memorisation`` verdicts in
    the 2026-07-28 run were decided by non-overlap of a four-unit interval
    against a four-hundred-unit one.  A degenerate stratum is now reported as
    degenerate, with its unit count, instead of publishing a number that reads
    as precision.

    The floor itself, and the coverage measurement that fixes it at eight, live
    in :data:`~src.transfer.statistics.MINIMUM_BOOTSTRAP_UNITS`.  The derivation
    once stated here selected four rather than eight and is corrected there.
    """

    if resamples < 1 or n_heads < 1 or not 0 < alpha < 1:
        raise ValueError("invalid bootstrap parameters")
    if minimum_units < 2:
        raise ValueError("a percentile interval needs at least two units")
    if not scores:
        raise ValueError("cannot bootstrap an empty stratum")
    count = len(scores)
    # One declaration of the floor and of the words that go with it, in
    # ``statistics``. This module held the only copy of both while two sibling
    # bootstraps guarded nothing but ``n < 2``.
    floor = bootstrap_unit_floor(count, minimum_units=minimum_units)
    if floor["degenerate"]:
        return {
            "resamples": int(resamples),
            "alpha": float(alpha),
            "unit": "probe",
            **floor,
            "peak_over_uniform_ci": None,
            "fraction_above_threshold_ci": None,
        }

    # Drawing ``count`` probes with replacement and adding their score matrices is
    # the same thing as multiplying the matrix stack by a multinomial count
    # vector, and the second form is a single BLAS call instead of a Python loop
    # over a million array additions. The distribution is identical; only the
    # arithmetic order changes.
    stack = np.stack([score.sums["prefix_matching"].reshape(-1) for score in scores])
    positions = np.asarray([score.scored_positions for score in scores], dtype=np.float64)
    uniform_sums = np.asarray([score.uniform_sum for score in scores], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(count, np.full(count, 1.0 / count), size=resamples).astype(
        np.float64
    )
    scored = draws @ positions
    if not np.all(scored > 0):
        raise RuntimeError("a bootstrap resample contributed no scored query positions")
    means = (draws @ stack) / scored[:, None]
    uniform = (draws @ uniform_sums) / scored
    peaks = means.max(axis=1) / uniform
    fractions = (means >= threshold).sum(axis=1) / n_heads
    if not (np.isfinite(peaks).all() and np.isfinite(fractions).all()):
        raise RuntimeError("bootstrap produced a non-finite statistic")
    low, high = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "resamples": int(resamples),
        "alpha": float(alpha),
        "unit": "probe",
        **floor,
        "peak_over_uniform_ci": [
            _finite(float(np.percentile(peaks, low)), "peak CI low"),
            _finite(float(np.percentile(peaks, high)), "peak CI high"),
        ],
        "fraction_above_threshold_ci": [
            _finite(float(np.percentile(fractions, low)), "fraction CI low"),
            _finite(float(np.percentile(fractions, high)), "fraction CI high"),
        ],
    }


def alignment_block(
    arm: Arm,
    scores: Sequence[ProbeScore],
    copying: Mapping[str, np.ndarray],
    *,
    headline_threshold: float,
    thresholds: Sequence[float] = INDUCTION_THRESHOLDS,
) -> dict[str, Any]:
    """Every induction quantity the headline reports, for one set of probes."""

    total_heads = n_head(arm) * arm.n_layer
    pooled = pool_scores(scores)
    prefix = pooled["scores"]["prefix_matching"]
    peak = float(prefix.max())
    uniform = pooled["uniform_baseline"]
    selected = _selected_heads(prefix, headline_threshold)
    return {
        # Which heads, not just how many. Two strata that agree on the count but
        # disagree on the identity of the heads would be two different
        # mechanisms reported as one number; the memorisation account in
        # particular predicts that the heads recruited on a stored sequence need
        # not be the heads recruited on an unseen one.
        "induction_head_set": [[layer, head] for layer, head in selected],
        "n_probes": int(pooled["n_probes"]),
        "scored_query_positions": int(pooled["scored_query_positions"]),
        "mean_coverage": pooled["mean_coverage"],
        "mean_repeat_symbols": pooled["mean_repeat_symbols"],
        "uniform_baseline": uniform,
        "peak_prefix_matching": _finite(peak, "peak prefix matching"),
        "peak_over_uniform": _finite(peak / uniform, "peak over uniform"),
        "n_heads": total_heads,
        "fraction_above_threshold": {
            f"{value:.2f}": _finite(
                float((prefix >= value).sum()) / total_heads, f"fraction above {value}"
            )
            for value in thresholds
        },
        "census": head_census(prefix),
        "same_token_distribution": summarise_head_matrix(
            pooled["scores"]["same_token"], "same_token"
        ),
        "offset_two_distribution": summarise_head_matrix(
            pooled["scores"]["offset_two"], "offset_two"
        ),
        "ov_over_induction_heads": ov_over_heads(copying, selected),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, so that Spearman is defined on tied covariates."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    for index in range(1, values.size + 1):
        if index == values.size or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    centred_left = left - left.mean()
    centred_right = right - right.mean()
    denominator = float(np.linalg.norm(centred_left) * np.linalg.norm(centred_right))
    if denominator <= 0:
        raise RuntimeError("a covariate has zero variance; correlation is undefined")
    return float(centred_left @ centred_right) / denominator


def covariate_analysis(
    scores: Sequence[ProbeScore],
    identities: Mapping[int, float],
    *,
    layer: int,
    head: int,
    minimum_n: int = 8,
    resamples: int = 2000,
    seed: int = 20260729,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Separate homology from repeat length as explanations of induction strength.

    Binning by identity is confounded: sequences that are present verbatim in the
    corpus also carry systematically *longer* internal repeats than sequences
    that are not, and a longer repeat gives a prefix-matching head more to lock
    onto.  A stratum gradient is therefore not by itself evidence about
    memorisation, and the synthetic probe -- whose repeat is longer than any
    natural stratum's -- sits on the same axis.

    This runs the comparison without bins.  The response is one number per probe:
    the mean attention paid by the arm's strongest induction head, which is the
    per-sequence version of the peak statistic.  Partial Spearman coefficients
    then ask whether identity still tracks induction once repeat length is held
    at its rank, and whether repeat length still does once identity is.  If the
    identity term survives and the length term does not, memorisation is the
    better explanation; if the reverse, the stratum gradient is a length artefact.

    **Those two partials are the module's adjudicating statistic and they used
    to be published as bare point estimates.**  ``minimum_n`` is eight, and this
    file refuses a percentile interval below eight units on the grounds that one
    computed there would mislead -- so the same file was willing to decide
    between memorisation and a length artefact on two correlations over eight
    probes with no interval, no p-value and no resampling of any kind.
    "Survives" and "does not" are comparisons, and a comparison of two point
    estimates at n = 8 is not one.  A probe-level bootstrap now accompanies both
    partials, along with the fraction of resamples in which the identity term is
    the larger of the two, which is the quantity the verdict actually rests on.
    The bootstrap resamples probes, the same unit :func:`bootstrap_stratum` uses
    and the only independent one here.
    """

    if minimum_n < 4:
        raise ValueError("a partial correlation on fewer than four points is not meaningful")
    if len(scores) < minimum_n:
        return {
            "measured": False,
            "n": len(scores),
            "reason": f"fewer than {minimum_n} probes; no correlation is reported",
        }
    response = np.asarray(
        [float(score.sums["prefix_matching"][layer, head]) / score.scored_positions
         for score in scores],
        dtype=np.float64,
    )
    lengths = np.asarray([float(score.repeat_symbols) for score in scores], dtype=np.float64)
    identity = np.asarray(
        [float(identities[score.record_index]) for score in scores], dtype=np.float64
    )
    # A cohort in which every record is a UniRef50 representative has no identity
    # variance at all, which the module docstring says to expect. That is a
    # property of the cohort, not an error, and it must not abort an arm's run
    # from four frames down after the GPU work is finished.
    constant = [
        name
        for name, values in (
            ("response", response), ("repeat_length", lengths), ("identity", identity)
        )
        if float(values.std()) <= 0.0
    ]
    if constant:
        return {
            "measured": False,
            "n": len(scores),
            "reason": (
                f"{constant} has no variance in this cohort, so a rank correlation "
                "against it is undefined"
            ),
            "constant_covariates": constant,
        }
    def partial(primary: float, secondary: float, between: float) -> float | None:
        denominator = math.sqrt(max(1.0 - secondary**2, 0.0) * max(1.0 - between**2, 0.0))
        if denominator <= 0:
            return None
        return _finite(
            (primary - secondary * between) / denominator, "partial correlation"
        )

    def partials(
        block_response: np.ndarray, block_lengths: np.ndarray, block_identity: np.ndarray
    ) -> tuple[float, float, float, float | None, float | None]:
        rank_response = _average_ranks(block_response)
        rank_length = _average_ranks(block_lengths)
        rank_identity = _average_ranks(block_identity)
        ry = _correlation(rank_response, rank_identity)
        rl = _correlation(rank_response, rank_length)
        yl = _correlation(rank_identity, rank_length)
        return ry, rl, yl, partial(ry, rl, yl), partial(rl, ry, yl)

    r_ry, r_rl, r_yl, partial_identity, partial_length = partials(
        response, lengths, identity
    )

    if resamples < 100 or not 0 < alpha < 1:
        raise ValueError("invalid bootstrap parameters for the covariate analysis")
    generator = np.random.default_rng(seed)
    identity_draws: list[float] = []
    length_draws: list[float] = []
    identity_larger = 0
    degenerate_draws = 0
    for _ in range(resamples):
        index = generator.integers(0, response.size, size=response.size)
        block = (response[index], lengths[index], identity[index])
        # A resample can have no variance in a covariate, which is a fact about
        # this cohort rather than a glitch: at eight probes an all-identical
        # draw is not rare. Counted, not skipped silently, and the interval is
        # withheld if too many of them occur -- the same rule
        # ``statistics.MINIMUM_FINITE_DRAW_FRACTION`` states for every other
        # bootstrap in this package.
        if any(float(values.std()) <= 0.0 for values in block):
            degenerate_draws += 1
            continue
        _, _, _, draw_identity, draw_length = partials(*block)
        if draw_identity is None or draw_length is None:
            degenerate_draws += 1
            continue
        identity_draws.append(draw_identity)
        length_draws.append(draw_length)
        identity_larger += int(abs(draw_identity) > abs(draw_length))

    required = int(math.ceil(MINIMUM_FINITE_DRAW_FRACTION * resamples))
    usable = len(identity_draws) >= required
    low, high = 100 * alpha / 2, 100 * (1 - alpha / 2)

    def interval(draws: list[float]) -> list[float] | None:
        if not usable:
            return None
        return [
            _finite(float(np.percentile(draws, low)), "partial CI low"),
            _finite(float(np.percentile(draws, high)), "partial CI high"),
        ]

    return {
        "measured": True,
        "n": len(scores),
        "head": [int(layer), int(head)],
        "response": "mean attention of the arm's strongest induction head, per probe",
        "spearman_identity_vs_induction": _finite(r_ry, "identity correlation"),
        "spearman_repeat_length_vs_induction": _finite(r_rl, "length correlation"),
        "spearman_identity_vs_repeat_length": _finite(r_yl, "identity/length correlation"),
        "partial_identity_given_repeat_length": partial_identity,
        "partial_repeat_length_given_identity": partial_length,
        "bootstrap": {
            "unit": "probe",
            "resamples": int(resamples),
            "seed": int(seed),
            "alpha": float(alpha),
            "usable_draws": len(identity_draws),
            "degenerate_draws": degenerate_draws,
            "required_draws": required,
            "partial_identity_given_repeat_length_ci": interval(identity_draws),
            "partial_repeat_length_given_identity_ci": interval(length_draws),
            # The verdict this analysis exists to reach is "the identity term
            # survives and the length term does not", which is a comparison of
            # two magnitudes. This is that comparison's own sampling fraction,
            # rather than two intervals a reader has to eyeball against each
            # other -- overlapping intervals do not settle a paired comparison.
            "fraction_identity_term_larger": (
                identity_larger / len(identity_draws) if usable else None
            ),
            "refused_reason": (
                None
                if usable
                else (
                    f"only {len(identity_draws)} of {resamples} resamples admitted a "
                    "partial correlation; the rest had a covariate with no variance, "
                    "so an interval over the survivors is conditioned on the cohort "
                    "being well conditioned and is not the requested distribution"
                )
            ),
        },
    }


def head_set_overlap(
    left: Sequence[Sequence[int]], right: Sequence[Sequence[int]]
) -> dict[str, Any]:
    """Do two probe families recruit the same heads, or merely the same number?

    Equal head counts across strata are consistent with one mechanism running
    everywhere and also with two mechanisms of similar size running in different
    places.  The Jaccard index separates them, and it is the sharpest single
    number this control produces: a head set identified on sequences the model
    has memorised that coincides with the head set identified on synthetic
    sequences it cannot have seen is one mechanism.
    """

    first = {tuple(int(value) for value in head) for head in left}
    second = {tuple(int(value) for value in head) for head in right}
    union = first | second
    return {
        "n_left": len(first),
        "n_right": len(second),
        "n_shared": len(first & second),
        "jaccard": None if not union else _finite(len(first & second) / len(union), "jaccard"),
    }


def stratum_report(
    arm: Arm,
    scores: Sequence[ProbeScore],
    copying: Mapping[str, np.ndarray],
    group_of: Mapping[int, int],
    *,
    n_records: int,
    headline_threshold: float,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """The full per-stratum row, reported at the level of distinct sequences.

    The top-level numbers are computed over one probe per distinct sequence and
    the bootstrap resamples those same units, because byte-identical records are
    one observation (:func:`sequence_groups`).  ``duplicate_weighted`` repeats the
    calculation over every record, which is what the headline census did and is
    kept so that the two can be compared rather than confused.

    A stratum with records but no probes is reported with ``measured: false`` and
    its record count intact.  It is not merged into a neighbour and it is not
    dropped: an unmeasurable bin is a fact about the cohort that a reader needs in
    order to judge how much the surviving bins can carry.
    """

    if n_records < 0 or headline_threshold <= 0:
        raise ValueError("invalid stratum-report parameters")
    if not scores:
        return {
            "measured": False,
            "n_records": int(n_records),
            "n_probes": 0,
            "n_distinct_sequences": 0,
            "reason": "no natural-repeat probe survives token alignment in this stratum",
        }

    representatives = representative_scores(scores, group_of)
    primary = alignment_block(
        arm, representatives, copying, headline_threshold=headline_threshold
    )
    # ``primary["n_probes"]`` is the number of representative probes the
    # top-level statistics were computed from, which is by construction the
    # number of distinct sequences; the record-level count is named separately so
    # that neither can be mistaken for the other.
    return {
        **primary,
        "measured": True,
        "n_records": int(n_records),
        "n_probes_all_records": len(scores),
        "n_distinct_sequences": len(representatives),
        "record_indices": sorted(score.record_index for score in scores),
        "distinct_record_indices": sorted(score.record_index for score in representatives),
        "duplicate_weighted": alignment_block(
            arm, scores, copying, headline_threshold=headline_threshold
        ),
        "bootstrap": bootstrap_stratum(
            representatives,
            threshold=headline_threshold,
            n_heads=n_head(arm) * arm.n_layer,
            resamples=resamples,
            seed=seed,
        ),
    }
