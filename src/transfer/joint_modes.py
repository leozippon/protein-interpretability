"""How one joint language-protein checkpoint is rendered in each of its modes.

A joint decoder has two input formats and one set of weights, so Appendix B rule
4 -- feed every arm the format it was trained on, and verify the rendering
against the model's own likelihood -- applies twice to the same checkpoint. This
module is the single place either format is decided (Appendix B rule 12);
``scripts/transfer/21_joint_mode_qualification.py`` names a family on the command
line and reads everything else from here.

**Why the per-residue escape is load-bearing rather than cosmetic.** Galactica's
protein format is ``[START_AMINO]...[END_AMINO]``, and its released
``tokenizer.json`` carries a ``Split`` pretokenizer on the literal
``SPL1T-TH1S-Pl3A5E`` with ``behavior: "Removed"`` -- so the marker is consumed by
the tokenizer itself, and inserting it before every residue is the *supported*
way to reach the model's own one-token-per-residue protein alphabet rather than a
workaround. Without it ``AutoTokenizer`` silently merges residues into
multi-residue BPE pieces at about 1.79 residues per token, and the measured cost
of that merge is 2.886 nats/token on ``galactica-1.3b`` (EXP-R2-151). Nothing
about a merged rendering looks wrong in an artefact -- the run completes, the
numbers are finite, and they are a measurement of a different object. That is why
:meth:`JointTokenisation.render` **raises** on a declared rendering that did not
produce one token per residue instead of recording a warning field nobody reads.

**One derivation, two rules.** The two families locate their scored positions
differently: Galactica's residues are ordinary single-letter pieces of a text
vocabulary and are identified by *position* (strictly between the delimiter ids),
while InstructProtein declares twenty dedicated ``ƤA``..``ƤY`` tokens and its
residues are identified by *identity*. Both rules are read off the same
declaration, and the residue token ids the identity rule needs are derived by
rendering a one-residue sequence through the very same declaration and reading
what falls between the delimiters -- so a family cannot end up with one alphabet
for selection and another for verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.transfer.arms import AA20

#: The literal Galactica's released pretokenizer splits on and removes. Declared
#: here because it is part of the rendering, not a property of any one model
#: directory; the ``Split`` rule that consumes it lives in the checkpoint's own
#: ``tokenizer.json``, which is what makes this the supported path.
GALACTICA_SPLIT_MARKER = "SPL1T-TH1S-Pl3A5E"

#: How a rendering locates the positions whose target token is a residue.
BETWEEN_DELIMITERS = "between_delimiters"
DECLARED_RESIDUE_IDS = "declared_residue_ids"
SCORED_TARGET_RULES = (BETWEEN_DELIMITERS, DECLARED_RESIDUE_IDS)

#: ``declared`` is the format the checkpoint was trained on. ``naive`` is the
#: same block with the per-residue escape removed -- the rendering an unaided
#: ``AutoTokenizer`` call produces, kept as a *control* so that a stage can price
#: the rendering in the estimand's own units rather than assert its value.
DECLARED = "declared"
NAIVE = "naive"
RENDERING_VARIANTS = (DECLARED, NAIVE)


@dataclass(frozen=True)
class JointRendering:
    """One family's declaration of how a protein is written for a joint decoder.

    ``residue_escape`` is the string inserted into the *rendered text* before
    every residue. It is not necessarily part of the resulting token's spelling:
    Galactica's marker is removed by the pretokenizer, so the token is the bare
    letter, while InstructProtein's ``Ƥ`` is the first character of a dedicated
    added token. Nothing here needs to know which, because the residue token ids
    are derived from the tokenizer rather than spelled a second time.

    ``protein_context_template`` is the optional document context the block is
    embedded in, with a single ``{context}`` field. It is optional in the strong
    sense: a rendering with no context is the bare block, and the context string
    that was used reaches the artefact rather than being implied by a flag.
    """

    name: str
    protein_start: str
    protein_end: str
    residue_escape: str
    escape_before_end_delimiter: bool
    protein_context_template: str | None
    scored_target_rule: str
    residue_subspace_disjoint_from_text: bool
    note: str

    def __post_init__(self) -> None:
        if self.scored_target_rule not in SCORED_TARGET_RULES:
            raise ValueError(
                f"{self.name}: scored target rule {self.scored_target_rule!r} is not "
                f"declared; known rules are {SCORED_TARGET_RULES}"
            )
        if not self.residue_escape:
            raise ValueError(
                f"{self.name}: a joint rendering must declare the per-residue escape "
                "that produces one token per residue; an empty one IS the naive "
                "control and cannot also be the declared format"
            )
        if not self.protein_start or not self.protein_end:
            raise ValueError(f"{self.name}: both protein delimiters must be declared")
        if self.protein_start == self.protein_end:
            raise ValueError(
                f"{self.name}: the two delimiters are the same string, so the scored "
                "span cannot be located"
            )
        if self.protein_context_template is not None and (
            "{context}" not in self.protein_context_template
        ):
            raise ValueError(
                f"{self.name}: the context template carries no {{context}} field, so "
                "the context string a run used could not reach the artefact"
            )

    def render_protein(
        self, sequence: str, *, context: str | None = None, variant: str = DECLARED
    ) -> str:
        """The string this family feeds a joint decoder for one sequence."""

        if variant not in RENDERING_VARIANTS:
            raise ValueError(
                f"unknown rendering variant {variant!r}; declared: {RENDERING_VARIANTS}"
            )
        if not sequence:
            raise ValueError("cannot render an empty sequence")
        illegal = sorted(set(sequence) - set(AA20))
        if illegal:
            raise ValueError(
                f"{self.name}: sequence carries non-canonical symbols {illegal}; the "
                "residue alphabet this rendering is declared over is src.transfer.arms.AA20"
            )
        escape = self.residue_escape if variant == DECLARED else ""
        body = "".join(escape + residue for residue in sequence)
        if escape and self.escape_before_end_delimiter:
            body += escape
        block = f"{self.protein_start}{body}{self.protein_end}"
        if context is None:
            return block
        if self.protein_context_template is None:
            raise ValueError(
                f"{self.name} declares no document-context template, so it cannot be "
                f"rendered with the context {context!r}"
            )
        return self.protein_context_template.format(context=context) + block


JOINT_RENDERINGS: dict[str, JointRendering] = {
    "galactica": JointRendering(
        name="galactica",
        protein_start="[START_AMINO]",
        protein_end="[END_AMINO]",
        residue_escape=GALACTICA_SPLIT_MARKER,
        # galai's own form, and the one verified at EXP-R2-151: the marker is
        # repeated once before the closing delimiter.
        escape_before_end_delimiter=True,
        protein_context_template="# {context}\n\n",
        # The residues are ordinary single-letter pieces of a 50000-piece text
        # vocabulary, so identity cannot separate a residue from the same letter
        # occurring in prose. Position can, and the delimiters are added tokens.
        scored_target_rule=BETWEEN_DELIMITERS,
        residue_subspace_disjoint_from_text=False,
        note=(
            "protein is [START_AMINO]...[END_AMINO] with the SPL1T-TH1S-Pl3A5E "
            "marker before every residue and once before the closing delimiter. "
            "The released tokenizer.json carries a Split pretokenizer on that "
            "literal with behavior 'Removed', so the marker is consumed by the "
            "tokenizer and this is the supported path rather than a workaround. "
            "The residue tokens are ordinary letters of the text vocabulary, so "
            "this family declares NO disjoint residue subspace and a "
            "residue-probability-mass statistic on its text mode would be a "
            "statistic about twenty capital letters"
        ),
    ),
    "instructprotein": JointRendering(
        name="instructprotein",
        protein_start="<protein>",
        protein_end="</protein>",
        residue_escape="Ƥ",
        escape_before_end_delimiter=False,
        protein_context_template="Instruction: {context}\nOutput: ",
        # added_tokens.json declares <protein> 50265, </protein> 50266 and
        # ƤA..ƤY 50267-50286, so the residue output space is enumerable and
        # disjoint from the text vocabulary and a scored position is decidable by
        # inspection rather than by inference from position.
        scored_target_rule=DECLARED_RESIDUE_IDS,
        residue_subspace_disjoint_from_text=True,
        note=(
            "protein is <protein>...</protein> with every residue written as its "
            "own dedicated added token 'Ƥ'+A..Y. Those twenty ids are disjoint "
            "from the text vocabulary, which makes the residue probability mass "
            "at a text position a measurable statistic and is what distinguishes "
            "a collapsed text mode from a merely poor one"
        ),
    ),
}

RENDERING_NAMES: tuple[str, ...] = tuple(JOINT_RENDERINGS)


def rendering(name: str) -> JointRendering:
    """The declaration for one family, refusing a name nobody declared."""

    if name not in JOINT_RENDERINGS:
        raise KeyError(
            f"unknown joint rendering {name!r}; declared families are "
            f"{sorted(JOINT_RENDERINGS)}. A joint checkpoint whose format is not "
            "declared here cannot be measured: rule 4 requires the rendering to be "
            "established before any number is read from it"
        )
    return JOINT_RENDERINGS[name]


# --------------------------------------------------------------- tokenisation


def encode(tokenizer: Any, text: str) -> list[int]:
    """Token ids for one string, as a plain list."""

    return [int(value) for value in tokenizer(text, return_tensors=None)["input_ids"]]


def _declared_token_id(tokenizer: Any, token: str, *, role: str) -> int:
    """The id of a token the rendering names, refusing an absent one.

    Round-tripped rather than compared against the unknown-token id: on
    InstructProtein the unknown token *is* the beginning-of-sequence token, so an
    ``== unk_token_id`` test would refuse a legitimate id on one family and
    accept a bogus one on another.
    """

    resolved = tokenizer.convert_tokens_to_ids(token)
    if resolved is None:
        raise ValueError(f"this tokenizer has no id for the {role} {token!r}")
    resolved = int(resolved)
    if tokenizer.convert_ids_to_tokens(resolved) != token:
        raise ValueError(
            f"the {role} {token!r} resolves to id {resolved}, which decodes back to "
            f"{tokenizer.convert_ids_to_tokens(resolved)!r}; this tokenizer does not "
            "carry it as a token of its own"
        )
    return resolved


def _delimited_span(token_ids: Sequence[int], start_id: int, end_id: int) -> tuple[int, ...]:
    """Positions strictly between the one start delimiter and the one end delimiter."""

    ids = list(token_ids)
    if ids.count(start_id) != 1 or ids.count(end_id) != 1:
        raise ValueError(
            f"a rendered protein must carry exactly one start delimiter (id "
            f"{start_id}) and one end delimiter (id {end_id}); this one carries "
            f"{ids.count(start_id)} and {ids.count(end_id)}"
        )
    opening, closing = ids.index(start_id), ids.index(end_id)
    if closing <= opening + 1:
        raise ValueError("the rendered protein has no content between its delimiters")
    return tuple(range(opening + 1, closing))


def scored_target_positions(
    tokenizer: Any,
    declaration: JointRendering,
    rendered: str,
    *,
    rule: str | None = None,
    residue_ids: Mapping[str, int] | None = None,
) -> tuple[int, ...]:
    """Positions of the rendered string whose target token is a residue.

    ``rule`` defaults to the family's own declaration. It is an argument at all
    because the naive control has no residue tokens by construction, so the
    identity rule selects nothing there and the delimited span is the only
    definable scored set for it; :meth:`JointTokenisation.render` is what makes
    that substitution, and it makes it in one place.

    A position of zero is refused: nothing predicts the first token of a
    sequence, so scoring it would read a logit that does not exist.
    """

    rule = declaration.scored_target_rule if rule is None else rule
    if rule not in SCORED_TARGET_RULES:
        raise ValueError(f"unknown scored target rule {rule!r}; declared: {SCORED_TARGET_RULES}")
    token_ids = encode(tokenizer, rendered)
    start_id = _declared_token_id(tokenizer, declaration.protein_start, role="start delimiter")
    end_id = _declared_token_id(tokenizer, declaration.protein_end, role="end delimiter")
    span = _delimited_span(token_ids, start_id, end_id)
    if rule == DECLARED_RESIDUE_IDS:
        if residue_ids is None:
            raise ValueError(
                f"{declaration.name} identifies its scored positions by token "
                "identity, so the declared residue ids must be supplied"
            )
        allowed = set(int(value) for value in residue_ids.values())
        positions = tuple(
            index for index, value in enumerate(token_ids) if value in allowed
        )
    else:
        positions = span
    if 0 in positions:
        raise ValueError(
            "position 0 was selected as a scored target, but nothing predicts the "
            "first token of a sequence"
        )
    return positions


@dataclass(frozen=True)
class RenderedProtein:
    """One sequence rendered, tokenised and reduced to its scored positions."""

    rendering: str
    variant: str
    text: str
    token_ids: tuple[int, ...]
    scored_positions: tuple[int, ...]
    n_residues: int

    @property
    def n_scored_tokens(self) -> int:
        return len(self.scored_positions)

    @property
    def residues_per_scored_token(self) -> float:
        """Residues per token **over the scored span**, never over the whole string.

        The denominator is named because it is the quantity that differs between
        the declared and the naive rendering: exactly 1.0 for a rendering that
        passed the per-residue verification, and about 1.79 for Galactica's
        unescaped form. A ratio taken over the whole rendered string would fold
        the delimiters and any document context into it (Appendix B rule 27).
        """

        return self.n_residues / len(self.scored_positions)


@dataclass(frozen=True)
class JointTokenisation:
    """A rendering resolved against one tokenizer, with its ids verified.

    Building this object is where a tokenizer that cannot carry the declaration
    is refused: both delimiters must exist as tokens of their own, and each of
    the twenty canonical residues must render to exactly one token, all twenty
    distinct. Every later call reuses the resolved ids, so the twenty probe
    encodings happen once per run rather than once per sequence.
    """

    declaration: JointRendering
    tokenizer: Any
    start_id: int
    end_id: int
    residue_ids: Mapping[str, int]

    def render(
        self, sequence: str, *, context: str | None = None, variant: str = DECLARED
    ) -> RenderedProtein:
        """Render, tokenise, select the scored positions and verify the result.

        The declared variant is verified against the model's own alphabet and
        **raises** when the tokenizer merged residues: one token per residue,
        every scored target inside the declared residue set. The naive variant is
        deliberately not verified -- it exists to be wrong, and its scored span is
        the delimited one because the identity rule would select nothing in it.
        """

        text = self.declaration.render_protein(sequence, context=context, variant=variant)
        token_ids = tuple(encode(self.tokenizer, text))
        rule = self.declaration.scored_target_rule if variant == DECLARED else BETWEEN_DELIMITERS
        positions = scored_target_positions(
            self.tokenizer,
            self.declaration,
            text,
            rule=rule,
            residue_ids=self.residue_ids,
        )
        record = RenderedProtein(
            rendering=self.declaration.name,
            variant=variant,
            text=text,
            token_ids=token_ids,
            scored_positions=positions,
            n_residues=len(sequence),
        )
        if variant == DECLARED:
            self.verify_one_token_per_residue(record)
        return record

    def verify_one_token_per_residue(self, record: RenderedProtein) -> None:
        """Refuse a rendering that did not reach the model's per-residue alphabet.

        This is the guard the whole module exists for. Galactica's tokenizer
        merges amino acids into multi-residue BPE pieces without the split-marker
        escape, and the merged measurement is wrong by about 2.9 nats/token while
        looking exactly like a valid one in the artefact (EXP-R2-151). It raises;
        it does not warn, and there is no flag that downgrades it.
        """

        if record.n_scored_tokens != record.n_residues:
            raise ValueError(
                f"{self.declaration.name}: the rendering produced "
                f"{record.n_scored_tokens} scored tokens for {record.n_residues} "
                f"residues ({record.residues_per_scored_token:.3f} residues per "
                "token). The tokenizer merged residues into multi-residue pieces, "
                "so token position is not residue position and the cross-entropy "
                "is not the quantity this stage reports -- measured at about 2.9 "
                "nats/token on galactica-1.3b (EXP-R2-151, Appendix B rule 4)"
            )
        allowed = set(int(value) for value in self.residue_ids.values())
        stray = sorted(
            {record.token_ids[position] for position in record.scored_positions} - allowed
        )
        if stray:
            raise ValueError(
                f"{self.declaration.name}: scored target ids {stray} are outside the "
                "declared residue alphabet, so the scored positions are not residues"
            )

    def facts(self) -> dict[str, Any]:
        """The resolved declaration, as it reaches the artefact."""

        declaration = self.declaration
        return {
            "name": declaration.name,
            "protein_start": declaration.protein_start,
            "protein_end": declaration.protein_end,
            "residue_escape": declaration.residue_escape,
            "escape_before_end_delimiter": declaration.escape_before_end_delimiter,
            "protein_context_template": declaration.protein_context_template,
            "scored_target_rule": declaration.scored_target_rule,
            "residue_subspace_disjoint_from_text": (
                declaration.residue_subspace_disjoint_from_text
            ),
            "start_token_id": int(self.start_id),
            "end_token_id": int(self.end_id),
            "residue_token_ids": {
                residue: int(value) for residue, value in sorted(self.residue_ids.items())
            },
            "residue_alphabet": AA20,
            "declaration_module": "src/transfer/joint_modes.py",
            "note": declaration.note,
        }


def resolve(tokenizer: Any, declaration: JointRendering | str) -> JointTokenisation:
    """Resolve a declared rendering against a tokenizer, or refuse the pair.

    The residue ids are *derived* rather than spelled a second time: each
    canonical residue is rendered as a one-residue sequence through the same
    declaration, and the token between the delimiters is that residue's id. A
    tokenizer that cannot produce one token per residue, or that maps two
    residues onto one id, is refused here -- before any checkpoint is loaded and
    long before any cross-entropy exists to be misread.
    """

    if isinstance(declaration, str):
        declaration = rendering(declaration)
    start_id = _declared_token_id(
        tokenizer, declaration.protein_start, role="start delimiter"
    )
    end_id = _declared_token_id(tokenizer, declaration.protein_end, role="end delimiter")
    residue_ids: dict[str, int] = {}
    for residue in AA20:
        probe = encode(tokenizer, declaration.render_protein(residue))
        span = _delimited_span(probe, start_id, end_id)
        if len(span) != 1:
            raise ValueError(
                f"{declaration.name}: the declared rendering of the single residue "
                f"{residue!r} produced {len(span)} tokens between its delimiters, so "
                "this tokenizer does not carry the per-residue alphabet the "
                "declaration claims"
            )
        residue_ids[residue] = probe[span[0]]
    if len(set(residue_ids.values())) != len(AA20):
        collisions = sorted(
            residue
            for residue, value in residue_ids.items()
            if list(residue_ids.values()).count(value) > 1
        )
        raise ValueError(
            f"{declaration.name}: residues {collisions} share a token id, so a scored "
            "position cannot be attributed to one residue"
        )
    return JointTokenisation(
        declaration=declaration,
        tokenizer=tokenizer,
        start_id=start_id,
        end_id=end_id,
        residue_ids=residue_ids,
    )
