"""What the perturbation-sensitivity stage must get right, on CPU stubs.

The stage asks whether a joint checkpoint's protein mode is more fragile than its
text mode under a perturbation of *equivalent relative magnitude*. That question
is only answerable if four things hold, and each of them can be wrong silently:

**The perturbation is the size it claims.** Epsilon is a fraction of the block
output's norm at that position, so a perturbation whose norm drifted -- by being
scaled against a batch mean, a layer mean, or nothing at all -- would compare two
modes under two different manipulations and report the difference as modality.

**The two anchors are real.** Epsilon = 0 adds the zero vector and must reproduce
the clean cross-entropy exactly; anything measurable means the splice path is not
a no-op and every point of the sweep is shifted with it. And the denominator every
ratio is divided by is the clean-to-mean-ablated gap, which must be positive.

**A degenerate position is refused.** A block output of zero norm has no scale for
epsilon to be relative to, and a zero perturbation there would report an
unperturbed position as a perturbed one.

**The joint arm is the checkpoint it says it is.** A rendering nobody declared, a
tokenizer that cannot carry the declared residue alphabet, and a block layout that
is not the serial post-attention feed-forward are all refused rather than measured.

**The perturbed tensor is the declared one on a PARALLEL residual block.**
ProGen2's block is GPT-J-style: attention and feed-forward read one ``ln_1`` and
both sum into the residual, so ``residual + ff`` is *not* the block's output and
the serial identity the GPT-2 arms are verified under would be checking the wrong
equality. ``TheParallelBlockTargetIsDeclared`` builds a parallel stub, pins that
the serial reconstruction misses while the declared one is exact, refuses a stub
whose declared identity does not hold, and refuses an architecture nobody
declared. Its config **omits ``vocab_size`` entirely and raises if anything reads
it**, which is ``progen2-xlarge``'s shape: that checkpoint carries only
``vocab_size_emb``/``vocab_size_lm_head``, and ``progen2-large`` carries a
``vocab_size`` of 51200 against a 31-token tokenizer, so a quantity derived from
that key would be over a mostly dead alphabet or would raise.

One correction to the specification is pinned here rather than left implicit: the
recovery fraction is **not** bounded below by zero and a large epsilon does not
approach the mean-ablated floor. Mean ablation removes the block's information; a
perturbation removes it *and* injects norm the residual stream never carried, which
damages the attention and embedding pathways too. So a large epsilon falls
*through* the floor. ``TheSweepAnchors`` measures that rather than asserting the
comfortable version, and the artefact records it as a property of the manipulation.

Everything runs on randomly initialised two-layer models built from config objects
-- a GPT-2 for the panel-arm path and a LLaMA for the joint path -- with stub
tokenizers: no GPU, no network, no 7B checkpoint, as ``tests/test_replaceable_arms.py``
and ``tests/test_neuron_basis_circuit.py`` do.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts/transfer") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts/transfer"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import (  # noqa: E402
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
)
from transformers.modeling_outputs import CausalLMOutputWithPast  # noqa: E402

from src.transfer import arms as A  # noqa: E402
from src.transfer import joint_modes as JM  # noqa: E402
from src.transfer import replaceable as R  # noqa: E402
from src.transfer.io import write_json  # noqa: E402


def _load_stage(filename: str):
    """Import a stage whose module name starts with a digit."""

    path = REPO / "scripts/transfer" / filename
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


STAGE = _load_stage("23_perturbation_sensitivity.py")


# --------------------------------------------------------------- the text stub

D_MODEL, D_MLP, N_LAYER, N_HEAD, VOCAB = 8, 24, 2, 2, 16

DOCUMENTS = [
    "the harbour opened onto a shallow bay",
    "a compiler translates one language into another",
    "rainfall is concentrated in two short seasons",
    "iron rusts when exposed to oxygen and water",
]


class _StubTextTokenizer:
    """Enough tokenizer for ``tokenize_batch`` and the unconditioned content mask."""

    all_special_ids = [0]
    unk_token_id = 0
    pad_token_id = 0
    eos_token = "<|endoftext|>"

    def __call__(self, text, return_tensors=None):
        return {"input_ids": [1 + (ord(character) % (VOCAB - 1)) for character in text]}

    def decode(self, ids):
        return "".join(chr(96 + int(i)) for i in ids)


def _overfit(model, rows: list[list[int]], *, steps: int = 200, lr: float = 5e-2) -> None:
    """Overfit a stub on a handful of rows so its MLP carries something.

    A *randomly initialised* decoder is already at its own chance rate, so
    mean-ablating its MLP moves the cross-entropy by a thousandth of a nat and a
    perturbation large enough to saturate it moves back towards chance -- which
    makes the recovery ratio a ratio of two rounding errors and lets a large
    epsilon read as *less* damaging than the floor. Every invariant this file
    pins is about the regime a trained model is in, so the stub is put in it.
    """

    width = max(len(row) for row in rows)
    ids = torch.zeros(len(rows), width, dtype=torch.long)
    labels = torch.full((len(rows), width), -100, dtype=torch.long)
    attention = torch.zeros(len(rows), width, dtype=torch.long)
    for index, row in enumerate(rows):
        ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
        labels[index, : len(row)] = torch.tensor(row, dtype=torch.long)
        attention[index, : len(row)] = 1
    model.train()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        optimiser.zero_grad()
        model(input_ids=ids, attention_mask=attention, labels=labels).loss.backward()
        optimiser.step()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


_TEXT_MODEL: GPT2LMHeadModel | None = None


def _text_model(seed: int = 11) -> GPT2LMHeadModel:
    """One trained text stub, built once: the tests read it and never mutate it."""

    global _TEXT_MODEL
    if _TEXT_MODEL is None:
        torch.manual_seed(seed)
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=VOCAB,
                n_positions=64,
                n_embd=D_MODEL,
                n_layer=N_LAYER,
                n_head=N_HEAD,
                n_inner=D_MLP,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
            )
        )
        tokenizer = _StubTextTokenizer()
        _overfit(model, [tokenizer(document)["input_ids"] for document in DOCUMENTS])
        _TEXT_MODEL = model
    return _TEXT_MODEL


def _text_arm() -> A.Arm:
    return A.Arm(
        spec=A.ArmSpec(
            name="gpt2",
            path=Path("/nowhere"),
            path_variable="TRANSFER_TEXT_MODEL_BASE_DIR",
            modality="text",
            n_layer=N_LAYER,
            d_model=D_MODEL,
            tokenisation="bpe",
            input_format="raw",
            evaluation_cohort_source="openwebtext",
            architecture="gpt2",
        ),
        model=_text_model(),
        tokenizer=_StubTextTokenizer(),
        device="cpu",
        dtype="float32",
    )


def _dense(**kwargs) -> R.DenseReplaceable:
    return R.DenseReplaceable(_text_arm(**kwargs), max_tokens=64)


# ----------------------------------------------------- the parallel-block stub


#: ProGen2's released vocabulary, in the order its tokenizer declares it: two
#: control tokens for the generation direction, then one token per residue, then
#: the single special id. Reproduced exactly because two of this file's claims are
#: about it -- that a residue tokenizer's expansion is 1.0 symbols per token, and
#: that the N-to-C control token is an ordinary vocabulary entry rather than a
#: special one, which is why the content mask has to resolve it from the rendering
#: declaration and cannot find it in ``all_special_ids``.
PROGEN_TOKENS: tuple[str, ...] = (
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "1",
    "2",
    *"ABCDEFGHIKLMNOPQRSTUVWXYZ",
    "<|endoftext|>",
)


class StubResidueTokenizer:
    """One token per residue, which is what ``tokenisation="residue"`` declares.

    The special-token bookkeeping reproduces the released ProGen2 tokenizer rather
    than being simplified: ``<|endoftext|>`` is the *only* special id and is also
    the pad and the unknown id, exactly as ``GPT2TokenizerFast`` reports for every
    ProGen2 rung. That coincidence is what makes ``all_special_ids`` an incomplete
    answer to "which positions are not content" on this lineage, so a stub that
    declared the direction marker special would test a tokenizer nobody has.
    """

    all_special_ids = [len(PROGEN_TOKENS) - 1]
    pad_token_id = len(PROGEN_TOKENS) - 1
    eos_token_id = len(PROGEN_TOKENS) - 1
    unk_token_id = len(PROGEN_TOKENS) - 1
    eos_token = "<|endoftext|>"

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._ids.get(token, self.unk_token_id)

    def __init__(self) -> None:
        self._ids = {token: index for index, token in enumerate(PROGEN_TOKENS)}

    def __len__(self) -> int:
        return len(PROGEN_TOKENS)

    def __call__(self, text, return_tensors=None):
        return {"input_ids": [self._ids[character] for character in text]}

    def decode(self, ids):
        return "".join(PROGEN_TOKENS[int(index)] for index in ids)


class StubProGenConfig:
    """A ProGen2-shaped config that declares no ``vocab_size`` at all.

    ``progen2-xlarge``'s config carries ``vocab_size_emb`` and
    ``vocab_size_lm_head`` and nothing else, so any code that reached for
    ``config.vocab_size`` on it would raise inside a run rather than at a
    declaration; ``progen2-large`` carries a ``vocab_size`` of 51200 against a
    31-token tokenizer, so the same code would silently compute over 51169
    unreachable symbols. Reading it here is an ``AssertionError`` and not an
    ``AttributeError`` on purpose: a ``getattr(config, "vocab_size", None)``
    swallows the second and would let the failure this stub exists to catch pass
    as a ``None``.
    """

    model_type = "progen"

    def __init__(self, n_head: int) -> None:
        self.n_head = n_head

    def __getattr__(self, name: str):
        if name == "vocab_size":
            raise AssertionError(
                "config.vocab_size was read on the ProGen2 path: progen2-xlarge "
                "declares no such key and progen2-large declares 51200 against a "
                "31-token tokenizer, so nothing here may depend on it"
            )
        raise AttributeError(name)


class _StubProGenAttention(torch.nn.Module):
    """Causal attention returning ProGen2's ``(output, ...)`` tuple.

    Separate ``q``/``k``/``v`` projections, deliberately: ProGen2's real
    ``qkv_proj`` is sharded in q, v, k order across eight model-parallel
    partitions, ``src.transfer.circuits`` already resolves that layout and
    verifies it against the live forward, and a stub with a fused projection here
    would invite a second implementation of the same split.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.q = torch.nn.Linear(width, width, bias=False)
        self.k = torch.nn.Linear(width, width, bias=False)
        self.v = torch.nn.Linear(width, width, bias=False)
        self.out_proj = torch.nn.Linear(width, width, bias=False)

    def forward(self, hidden_states, **kwargs):
        scores = self.q(hidden_states) @ self.k(hidden_states).transpose(-1, -2)
        length = hidden_states.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal, float("-inf")) / (hidden_states.shape[-1] ** 0.5)
        return (self.out_proj(scores.softmax(-1) @ self.v(hidden_states)), None)


class _StubProGenMLP(torch.nn.Module):
    def __init__(self, width: int, inner: int) -> None:
        super().__init__()
        self.fc_in = torch.nn.Linear(width, inner)
        self.fc_out = torch.nn.Linear(inner, width)

    def forward(self, hidden_states):
        return self.fc_out(torch.nn.functional.gelu(self.fc_in(hidden_states)))


class _StubProGenBlock(torch.nn.Module):
    """ProGen2's block, transcribed from its released ``modeling_progen.py``::

        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states, ...)[0]
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_output + feed_forward_hidden_states + residual

    ``leak`` is the corruption: a term the block adds that no interceptor sees, so
    the declared identity no longer holds and the arm must be refused.
    """

    def __init__(self, width: int, inner: int, *, leak: float = 0.0) -> None:
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(width)
        self.attn = _StubProGenAttention(width)
        self.mlp = _StubProGenMLP(width, inner)
        self.leak = float(leak)

    def forward(self, hidden_states, **kwargs):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states)[0]
        feed_forward_hidden_states = self.mlp(hidden_states)
        return (attn_output + feed_forward_hidden_states + residual + self.leak,)


class StubProGenForCausalLM(torch.nn.Module):
    """A ProGen2-shaped decoder: ``transformer.h`` of parallel-residual blocks.

    The embedding width is a constructor argument and never ``config.vocab_size``,
    which is how the released model reads it too (``config.vocab_size_emb``).
    Right padding plus causal attention means the padding mask changes nothing,
    so it is accepted and not consulted.
    """

    def __init__(self, *, leak: float = 0.0) -> None:
        super().__init__()
        self.config = StubProGenConfig(N_HEAD)
        self.transformer = torch.nn.Module()
        self.transformer.wte = torch.nn.Embedding(len(PROGEN_TOKENS), D_MODEL)
        self.transformer.h = torch.nn.ModuleList(
            _StubProGenBlock(D_MODEL, D_MLP, leak=leak) for _ in range(N_LAYER)
        )
        self.transformer.ln_f = torch.nn.LayerNorm(D_MODEL)
        self.lm_head = torch.nn.Linear(D_MODEL, len(PROGEN_TOKENS), bias=False)

    def forward(
        self, input_ids, attention_mask=None, labels=None, use_cache=False, return_dict=True
    ):
        hidden = self.transformer.wte(input_ids)
        for block in self.transformer.h:
            hidden = block(hidden)[0]
        logits = self.lm_head(self.transformer.ln_f(hidden))
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits[..., :-1, :].reshape(-1, logits.shape[-1]),
                labels[..., 1:].reshape(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)


_PARALLEL_MODELS: dict[float, StubProGenForCausalLM] = {}


def _progen_model(leak: float = 0.0, seed: int = 17) -> StubProGenForCausalLM:
    """One trained parallel stub per leak, for the reason ``_text_model`` is cached."""

    if leak not in _PARALLEL_MODELS:
        torch.manual_seed(seed)
        model = StubProGenForCausalLM(leak=leak)
        tokenizer = StubResidueTokenizer()
        _overfit(model, [tokenizer("1" + sequence)["input_ids"] for sequence in SEQUENCES])
        _PARALLEL_MODELS[leak] = model
    return _PARALLEL_MODELS[leak]


def _progen_arm(leak: float = 0.0) -> A.Arm:
    """A ProGen2 arm at stub scale, under the real arm's name and architecture.

    The name is real because the loader band is keyed by it; the shape is the
    stub's because a two-layer model is what a CPU test can train. Exactly the
    arrangement ``_text_arm`` uses for ``gpt2``.
    """

    return A.Arm(
        spec=A.ArmSpec(
            name="progen2-small",
            path=Path("/nowhere"),
            path_variable="TRANSFER_MODEL_BASE_DIR",
            modality="protein",
            n_layer=N_LAYER,
            d_model=D_MODEL,
            tokenisation="residue",
            input_format="n_to_c_control",
            evaluation_cohort_source="swissprot",
            architecture="progen",
            pretraining_corpus="uniref90_bfd30",
        ),
        model=_progen_model(leak),
        tokenizer=StubResidueTokenizer(),
        device="cpu",
        dtype="float32",
    )


def _parallel(leak: float = 0.0) -> R.ParallelResidualReplaceable:
    return R.ParallelResidualReplaceable(_progen_arm(leak), max_tokens=64)


# -------------------------------------------------------------- the joint stub


SEQUENCES = [
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
    "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMN",
    "MSDTLTRLAEVLEARKGAAPDSSYVASLYHKG",
    "MQPNDITFFQRFQNDILAGRKTITIRDASESH",
]


class StubJointTokenizer:
    """A vocabulary that merges residues, whose delimiters are not its tokens.

    The two properties of the staged ProLLaMA tokenizer that decide this stage's
    protein numbers: ``Seq=<`` is spelled out of ordinary pieces rather than
    carried as one, and a residue run comes back as multi-residue pieces, so the
    family's declared symbol unit is the token. ``split_marker`` reproduces
    Galactica's released ``Split``/``Removed`` rule; without it, a family that
    declares a per-residue alphabet cannot reach one -- which is the state a
    protein-mode run must be refused in.
    """

    def __init__(
        self,
        *,
        specials: tuple[str, ...] = (),
        split_marker: str | None = None,
        bos_id: int = 2,
        declare_ids: bool = True,
    ) -> None:
        self.split_marker = split_marker
        self.bos_id = bos_id
        self._vocab: dict[str, int] = {}
        self._inverse: dict[int, str] = {bos_id: "<s>"}
        self._add("<unk>")
        for token in specials:
            self._add(token)
        # 'Seq' is one piece, as it is on the staged LLaMA-2 vocabulary. Spelling
        # it out of 'S', 'e', 'q' would make the residue S occur inside the
        # prefix, and the spelled-run rule would then refuse every rendering as
        # ambiguous -- a property of this stub, not of the family.
        for token in ("Seq", "Generate", "by", "superfamily", "Superfamily"):
            self._add(token)
        for residue in A.AA20:
            self._add(residue)
        for left in A.AA20:
            for right in A.AA20:
                self._add(left + right)
        for character in "=<>[] abcdefghijklmnopqrstuvwxyz.,\n#":
            self._add(character)
        self._longest = max(len(token) for token in self._vocab)
        self.unk_id = self._vocab["<unk>"]
        self.all_special_ids = [bos_id, self.unk_id]
        # Deliberately no pad_token_id: LLaMA-2's tokenizer declares none either,
        # so the padding id has to be resolved from the declared fallback order.
        self.eos_token_id = bos_id if declare_ids else None
        self.unk_token_id = self.unk_id if declare_ids else None

    def _add(self, token: str) -> None:
        if token in self._vocab:
            return
        index = len(self._vocab) + 10
        self._vocab[token] = index
        self._inverse[index] = token

    def __len__(self) -> int:
        return len(self._vocab) + 10

    def convert_tokens_to_ids(self, token: str):
        return self._vocab.get(token)

    def convert_ids_to_tokens(self, index: int):
        return self._inverse.get(int(index))

    def decode(self, ids) -> str:
        return "".join(self._inverse.get(int(value), "") for value in ids)

    def _fragment(self, text: str) -> list[int]:
        ids: list[int] = []
        cursor = 0
        while cursor < len(text):
            for length in range(min(self._longest, len(text) - cursor), 0, -1):
                candidate = text[cursor : cursor + length]
                if candidate in self._vocab:
                    ids.append(self._vocab[candidate])
                    cursor += length
                    break
            else:
                ids.append(self.unk_id)
                cursor += 1
        return ids

    def __call__(self, text: str, return_tensors=None) -> dict[str, list[int]]:
        fragments = text.split(self.split_marker) if self.split_marker else [text]
        ids: list[int] = [self.bos_id]
        for fragment in fragments:
            ids.extend(self._fragment(fragment))
        return {"input_ids": ids}


def prollama_stub(**kwargs) -> StubJointTokenizer:
    return StubJointTokenizer(**kwargs)


def galactica_stub_without_the_escape() -> StubJointTokenizer:
    """Carries Galactica's delimiters and ignores its per-residue escape."""

    return StubJointTokenizer(specials=("[START_AMINO]", "[END_AMINO]"))


_JOINT_MODELS: dict[int, LlamaForCausalLM] = {}


def _llama(tokenizer, seed: int = 5) -> LlamaForCausalLM:
    """One trained LLaMA stub per vocabulary size, for the same reason as above."""

    size = len(tokenizer)
    if size not in _JOINT_MODELS:
        torch.manual_seed(seed)
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=size,
                hidden_size=D_MODEL,
                intermediate_size=D_MLP,
                num_hidden_layers=N_LAYER,
                num_attention_heads=N_HEAD,
                num_key_value_heads=N_HEAD,
                max_position_embeddings=256,
                tie_word_embeddings=False,
            )
        )
        corpus = list(DOCUMENTS) + [f"Seq=<{sequence}>" for sequence in SEQUENCES]
        _overfit(model, [JM.encode(tokenizer, text) for text in corpus])
        _JOINT_MODELS[size] = model
    return _JOINT_MODELS[size]


def _joint(mode: str, *, tokenizer=None, max_tokens: int = 128) -> R.JointReplaceable:
    tokenizer = tokenizer or prollama_stub()
    declaration = JM.rendering("prollama")
    return R.JointReplaceable(
        model=_llama(tokenizer),
        tokenizer=tokenizer,
        checkpoint=Path("/nowhere"),
        declaration=declaration,
        mode=mode,
        tokenisation=JM.resolve(tokenizer, declaration) if mode == "protein" else None,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------- the estimand


def _args(**overrides) -> argparse.Namespace:
    args = argparse.Namespace(
        sequences=len(DOCUMENTS),
        text_min_chars=8,
        protein_min_len=8,
        protein_max_len=400,
        cohort_skip=0,
        cohort_draw_seed=A.DEFAULT_CORPUS_DRAW_SEED,
        batch_size=2,
        bootstrap=64,
        seed=7,
        epsilons=[0.1, 4.0],
        draws=2,
        max_tokens=128,
    )
    vars(args).update(overrides)
    return args


def _stub_cohort(records, kind):
    def draw(n, *positional, skip=0, name="", seed=None, **keywords):
        return A.Cohort(
            name,
            kind,
            list(records)[:n],
            8,
            400,
            {
                "sampling": A.sampling_record(
                    seed=seed, skip=skip, requested=n, eligible=64, corpus=f"stub_{kind}"
                )
            },
        )

    return draw


@contextlib.contextmanager
def _patched_corpus(records, kind):
    attribute = "text_cohort" if kind == "text" else "protein_cohort"
    original = getattr(STAGE, attribute)
    setattr(STAGE, attribute, _stub_cohort(records, kind))
    try:
        yield
    finally:
        setattr(STAGE, attribute, original)


class ThePerturbationIsTheSizeItClaims(unittest.TestCase):
    """Epsilon is a fraction of THIS position's block output norm, or it is nothing."""

    def _y(self, seed: int = 3) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        # Deliberately heteroscedastic across positions: a perturbation scaled by
        # a batch or layer mean would pass a homoscedastic check and fail here.
        base = torch.randn((2, 5, D_MODEL), generator=generator)
        scale = torch.tensor([0.01, 0.1, 1.0, 10.0, 100.0]).view(1, 5, 1)
        return base * scale

    def test_the_norm_is_epsilon_times_the_block_output_norm_at_every_position(self):
        y = self._y()
        for epsilon in (0.05, 0.2, 0.8, 4.0):
            generator = torch.Generator().manual_seed(1)
            r = STAGE.relative_perturbation(y, epsilon, generator)
            torch.testing.assert_close(
                r.norm(dim=-1),
                epsilon * y.norm(dim=-1),
                rtol=1e-5,
                atol=0.0,
                msg=f"epsilon {epsilon}",
            )

    def test_epsilon_zero_is_exactly_the_zero_vector(self):
        y = self._y()
        generator = torch.Generator().manual_seed(1)
        r = STAGE.relative_perturbation(y, 0.0, generator)
        self.assertTrue(bool((r == 0).all()))

    def test_two_seeds_give_two_directions_and_one_norm(self):
        y = self._y()
        first = STAGE.relative_perturbation(y, 0.3, torch.Generator().manual_seed(1))
        again = STAGE.relative_perturbation(y, 0.3, torch.Generator().manual_seed(1))
        other = STAGE.relative_perturbation(y, 0.3, torch.Generator().manual_seed(2))
        self.assertTrue(torch.equal(first, again), "one seed must reproduce one draw")
        self.assertFalse(torch.equal(first, other), "two seeds must differ in direction")
        torch.testing.assert_close(
            first.norm(dim=-1), other.norm(dim=-1), rtol=1e-5, atol=0.0
        )
        # Different *directions*, not merely different numbers: the cosine between
        # two independent draws in 8 dimensions is nowhere near 1.
        cosine = torch.nn.functional.cosine_similarity(first, other, dim=-1)
        self.assertLess(float(cosine.abs().max()), 0.999)

    def test_a_zero_norm_block_output_is_refused_rather_than_perturbed_by_zero(self):
        y = self._y()
        y[0, 2] = 0.0
        for epsilon in (0.0, 0.5):
            with self.assertRaises(ValueError) as caught:
                STAGE.relative_perturbation(y, epsilon, torch.Generator().manual_seed(1))
            message = str(caught.exception)
            self.assertIn("zero norm", message)
            self.assertIn("1 positions", message)

    def test_a_negative_magnitude_is_refused(self):
        with self.assertRaises(ValueError):
            STAGE.relative_perturbation(
                self._y(), -0.1, torch.Generator().manual_seed(1)
            )

    def test_the_dtype_of_the_block_output_survives_the_perturbation(self):
        y = self._y().to(torch.bfloat16)
        r = STAGE.relative_perturbation(y, 0.25, torch.Generator().manual_seed(1))
        self.assertEqual(r.dtype, torch.bfloat16)


class TheSpliceIsTheOneSplice(unittest.TestCase):
    """The perturbation reaches the forward pass through block_intercept and nothing else."""

    @classmethod
    def setUpClass(cls):
        cls.model = _dense()
        cls.inputs = cls.model.render(DOCUMENTS)

    def test_every_layer_is_perturbed_and_at_the_declared_relative_size(self):
        clean: dict[int, torch.Tensor] = {}
        perturbed: dict[int, torch.Tensor] = {}

        def record(store):
            def tap(layer, x, y):
                store[layer] = y.detach().clone()
                return None

            return tap

        batch = self.model.batch(self.inputs[:2])
        with self.model.block_intercept(record(clean)):
            self.model.run(batch)
        # Registration order is entry order, so the inner tap reads the tensor the
        # perturbation produced -- which is what says the splice reached the pass.
        factory = STAGE.perturbation_context(self.model, epsilon=0.25, seed=4)
        with factory(), self.model.block_intercept(record(perturbed)):
            self.model.run(batch)

        self.assertEqual(sorted(perturbed), list(range(N_LAYER)))
        # Layer 0 sees the same clean output in both passes, so its perturbation is
        # exactly epsilon of its own norm. Later layers are perturbed on top of
        # what earlier ones already did, which is the sequential property.
        difference = perturbed[0] - clean[0]
        torch.testing.assert_close(
            difference.norm(dim=-1), 0.25 * clean[0].norm(dim=-1), rtol=1e-5, atol=0.0
        )
        self.assertFalse(torch.equal(perturbed[1], clean[1]))

    def test_the_hooks_are_removed_when_the_context_exits(self):
        # A splice that outlived its context would perturb every later sweep of
        # the same run -- including the clean endpoint the denominator is built
        # from -- without anything raising.
        before = STAGE22_scored(self.model, self.inputs)
        with STAGE.perturbation_context(self.model, epsilon=0.5, seed=4)():
            during = STAGE22_scored(self.model, self.inputs)
        after = STAGE22_scored(self.model, self.inputs)
        self.assertFalse(np.allclose(before, during))
        np.testing.assert_allclose(before, after, rtol=0, atol=0)


def STAGE22_scored(model, inputs, factory=None):
    return STAGE.STAGE22.scored_cross_entropy(
        model, inputs, batch_size=2, factory=factory
    )


def _choices(destination: str):
    """One argparse option's declared choices."""

    for action in STAGE.build_parser()._actions:
        if action.dest == destination:
            return action.choices
    raise AssertionError(f"the stage declares no --{destination.replace('_', '-')}")


class TheSweepAnchors(unittest.TestCase):
    """Both ends of the sweep are measured, and neither is what a reader assumes."""

    @classmethod
    def setUpClass(cls):
        cls.model = _dense()
        cls.inputs = cls.model.render(DOCUMENTS)
        cls.reference = STAGE.block_output_reference(
            cls.model, cls.inputs, batch_size=2
        )
        cls.clean = STAGE22_scored(cls.model, cls.inputs)
        cls.ablated = STAGE22_scored(
            cls.model,
            cls.inputs,
            factory=STAGE.STAGE15.mean_ablation_context(
                cls.model, cls.reference["block_output_mean"]
            ),
        )

    def _at(self, epsilon: float, seed: int = 3) -> np.ndarray:
        return STAGE22_scored(
            self.model,
            self.inputs,
            factory=STAGE.perturbation_context(self.model, epsilon=epsilon, seed=seed),
        )

    def test_the_denominator_is_positive_on_this_stub(self):
        self.assertGreater(
            float(self.ablated.mean() - self.clean.mean()),
            0.0,
            "the stub's mean ablation must damage the model or every ratio is undefined",
        )

    def test_epsilon_zero_reproduces_the_clean_model_exactly(self):
        np.testing.assert_allclose(self._at(0.0), self.clean, rtol=0, atol=0)

    def test_a_large_epsilon_falls_THROUGH_the_mean_ablated_floor(self):
        # The correction this class exists to pin. A perturbation is not bounded
        # by the mean-ablated floor: it removes the block's information AND
        # injects norm the residual stream never carried. Recovery therefore goes
        # negative rather than approaching zero from above.
        picks = STAGE.STAGE22.bootstrap_indices(len(self.clean), replicates=64, seed=1)
        huge = STAGE.STAGE22.recovery_record(
            self.clean, self._at(64.0), self.ablated, picks=picks
        )
        self.assertLessEqual(huge["recovery"], 0.0)
        self.assertGreater(
            huge["damage_nats_per_token"], huge["denominator_nats_per_token"]
        )

    def test_recovery_decreases_as_the_perturbation_grows(self):
        picks = STAGE.STAGE22.bootstrap_indices(len(self.clean), replicates=64, seed=1)
        curve = [
            STAGE.STAGE22.recovery_record(
                self.clean, self._at(epsilon), self.ablated, picks=picks
            )["recovery"]
            for epsilon in (0.0, 0.1, 0.4, 1.6, 6.4)
        ]
        self.assertEqual(curve, sorted(curve, reverse=True), curve)
        self.assertAlmostEqual(curve[0], 1.0, places=9)

    def test_the_reference_records_the_norm_epsilon_is_anchored_on(self):
        norms = self.reference["mean_block_output_norm_per_layer"]
        self.assertEqual(len(norms), N_LAYER)
        for value in norms:
            self.assertGreater(value, 0.0)
        counted = self.reference["n_content_positions_per_layer"]
        expected = float(
            sum(
                int(self.model.content_mask(self.model.batch([text])).sum())
                for text in self.inputs
            )
        )
        self.assertEqual(counted, [expected] * N_LAYER)


class TheArtefactCarriesTheNumerator(unittest.TestCase):
    """Standing rule 27: a ratio whose denominator is not published is not a
    measurement."""

    CLEAN = np.array([2.0, 2.2, 1.8, 2.0])
    ABLATED = np.array([4.0, 4.2, 3.8, 4.0])

    def test_the_endpoints_are_reported_in_nats_with_their_intervals(self):
        record = STAGE.endpoints_record(self.CLEAN, self.ABLATED, self.CLEAN)
        self.assertEqual(record["verdict"], "PASS")
        self.assertAlmostEqual(record["clean_nats_per_token"], 2.0)
        self.assertAlmostEqual(record["mean_ablated_nats_per_token"], 4.0)
        self.assertAlmostEqual(record["denominator_nats_per_token"], 2.0)
        for key in ("clean_interval", "mean_ablated_interval"):
            self.assertIn("interval", record[key])

    def test_a_shifted_zero_epsilon_point_fails_the_endpoint_gate(self):
        record = STAGE.endpoints_record(self.CLEAN, self.ABLATED, self.CLEAN + 0.01)
        self.assertEqual(record["verdict"], "FAIL")
        self.assertAlmostEqual(record["zero_epsilon_minus_clean_nats"], 0.01)

    def test_a_non_positive_denominator_fails_the_endpoint_gate(self):
        record = STAGE.endpoints_record(self.ABLATED, self.CLEAN, self.ABLATED)
        self.assertEqual(record["verdict"], "FAIL")

    def test_the_draw_spread_is_reported_before_its_summary(self):
        draws = [{"recovery": 0.9}, {"recovery": 0.3}, {"recovery": 0.6}]
        record = STAGE.across_draws(draws, "recovery")
        self.assertEqual(record["values"], [0.9, 0.3, 0.6])
        self.assertAlmostEqual(record["min"], 0.3)
        self.assertAlmostEqual(record["max"], 0.9)
        self.assertIsNone(STAGE.across_draws([{"recovery": None}], "recovery"))

    def test_a_measured_mode_carries_absolute_nats_and_its_own_denominator(self):
        model = _dense()
        # The band gate needs a real checkpoint; what is under test here is the
        # record, so the loader gate is stubbed rather than measured.
        model.self_check = lambda: {"verdict": "PASS", "note": "stubbed for the test"}
        with _patched_corpus(DOCUMENTS, "text"):
            with contextlib.redirect_stdout(io.StringIO()):
                record = STAGE.measure_mode(
                    _args(), model, source="openwebtext", label="stub"
                )

        endpoints = record["gates"]["endpoints"]
        for key in (
            "clean_nats_per_token",
            "mean_ablated_nats_per_token",
            "denominator_nats_per_token",
        ):
            self.assertTrue(np.isfinite(endpoints[key]), key)
        self.assertGreater(endpoints["denominator_nats_per_token"], 0.0)
        self.assertEqual(endpoints["verdict"], "PASS")

        # 0 is prepended to the declared grid and every point reports every draw.
        self.assertEqual([point["epsilon"] for point in record["sweep"]], [0.0, 0.1, 4.0])
        seeds = []
        for point in record["sweep"]:
            # Three draws of a random direction, reported individually -- except
            # at the identity anchor, where the zero vector has only one.
            self.assertEqual(len(point["draws"]), 1 if point["epsilon"] == 0.0 else 2)
            self.assertEqual(point["n_draws"], len(point["draws"]))
            for draw in point["draws"]:
                seeds.append(draw["seed"])
                # The numerator, not only the ratio.
                self.assertIn("damage_nats_per_token", draw)
                self.assertIn("denominator_nats_per_token", draw)
                self.assertIn("recovery", draw)
                self.assertAlmostEqual(
                    draw["denominator_nats_per_token"],
                    endpoints["denominator_nats_per_token"],
                    places=12,
                )
        self.assertEqual(len(set(seeds)), len(seeds), "every draw needs its own seed")

        self.assertAlmostEqual(record["sweep"][0]["across_draws"]["recovery"]["mean"], 1.0)
        self.assertGreater(record["symbols_per_token"]["value"], 0.0)
        self.assertIn("per scored token", record["symbols_per_token"]["unit"])
        self.assertEqual(
            len(record["block_output_norm"]["mean_per_layer"]), model.n_layers
        )

        # And it serialises: write_json refuses NaN and infinity, so a non-finite
        # intermediate would be caught here rather than in a campaign.
        with tempfile.TemporaryDirectory() as directory:
            write_json(Path(directory) / "record.json", record)

    def test_the_cross_mode_record_licenses_the_ratio_and_refuses_the_magnitude(self):
        def mode(denominator, expansion):
            return {
                "gates": {
                    "endpoints": {
                        "clean_nats_per_token": 1.0,
                        "mean_ablated_nats_per_token": 1.0 + denominator,
                        "denominator_nats_per_token": denominator,
                    }
                },
                "symbols_per_token": {"value": expansion, "unit": "residues per scored token"},
                "sweep": [
                    {
                        "epsilon": 0.1,
                        "across_draws": {"recovery": {"values": [0.9], "mean": 0.9}},
                    }
                ],
            }

        record = STAGE.cross_mode_record(
            {"text": mode(2.0, 4.2), "protein": mode(0.5, 1.54)}
        )
        self.assertEqual(record["magnitude_comparison"]["verdict"], "NOT_LICENSED")
        self.assertIn("L23", record["magnitude_comparison"]["reason"])
        self.assertIn("recovery fraction", record["licensed_comparison"])
        self.assertEqual(
            record["per_mode"]["protein"]["denominator_nats_per_token"], 0.5
        )
        self.assertEqual(record["per_mode"]["text"]["symbols_per_token"], 4.2)
        self.assertTrue(record["readable_only_with_two_modes"])
        self.assertFalse(
            STAGE.cross_mode_record({"text": mode(2.0, 4.2)})["readable_only_with_two_modes"]
        )


# --------------------------------------------------------------- the joint arm


class TheJointArmIsTheCheckpointItSaysItIs(unittest.TestCase):
    def test_an_unknown_rendering_is_refused_by_name(self):
        with self.assertRaises(KeyError) as caught:
            STAGE.resolve_protein_rendering(prollama_stub(), "not-a-family")
        self.assertIn("prollama", str(caught.exception))
        # And it is unreachable from the command line at all: the choices are
        # composed by src.transfer.joint_modes rather than listed by the stage.
        self.assertEqual(tuple(_choices("rendering")), tuple(JM.RENDERING_NAMES))

    def test_protein_mode_on_a_tokenizer_without_the_declared_alphabet_is_refused(self):
        # Galactica declares the residue as its symbol unit and reaches it only
        # through a per-residue escape the released tokenizer consumes. A
        # tokenizer that ignores that escape merges residues, so the declared
        # alphabet does not exist on it and protein mode must not be measured.
        with self.assertRaises(ValueError) as caught:
            STAGE.resolve_protein_rendering(
                galactica_stub_without_the_escape(), "galactica"
            )
        self.assertIn("per-residue alphabet", str(caught.exception))

    def test_protein_mode_without_a_resolved_rendering_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            R.JointReplaceable(
                model=_llama(prollama_stub()),
                tokenizer=prollama_stub(),
                checkpoint=Path("/nowhere"),
                declaration=JM.rendering("prollama"),
                mode="protein",
                tokenisation=None,
            )
        self.assertIn("resolved against", str(caught.exception))

    def test_text_mode_does_not_need_the_protein_alphabet_and_says_so(self):
        model = _joint("text", tokenizer=galactica_stub_without_the_escape())
        inputs = model.render(DOCUMENTS)
        with _patched_corpus(DOCUMENTS, "text"):
            with contextlib.redirect_stdout(io.StringIO()):
                record = STAGE.measure_mode(
                    _args(epsilons=[0.2], draws=1),
                    model,
                    source="openwebtext",
                    label="joint:text",
                )
        self.assertEqual(record["rendering"]["verdict"], "NOT_RESOLVED")
        self.assertEqual(record["rendering"]["declared_family"], "prollama")
        self.assertGreater(len(inputs), 0)

    def test_an_undeclared_block_layout_is_refused_rather_than_duck_typed(self):
        for architecture in ("gpt2", "opt", "progen", "qwen2"):
            with self.assertRaises(TypeError, msg=architecture):
                R.joint_block_layout(architecture)
        self.assertEqual(R.joint_block_layout("llama").feed_forward, "mlp")

    def test_the_declared_mode_set_is_the_one_the_stage_offers(self):
        self.assertEqual(R.JOINT_MODES, ("text", "protein"))
        with self.assertRaises(ValueError):
            _joint("nucleotide")

    def test_the_estimand_identity_holds_exactly_on_the_declared_layout(self):
        record = _joint("protein").self_check()
        self.assertEqual(record["estimand"]["verdict"], "PASS")
        self.assertEqual(record["estimand"]["max_absolute_difference"], 0.0)
        self.assertIn("post_attention_layernorm", record["estimand"]["identity"])

    def test_the_joint_arm_declares_the_gate_it_does_not_have(self):
        # A dense panel arm is gated on a measured likelihood band; no such band
        # exists for a checkpoint reached by path, and the honest record is a
        # withheld verdict with its reason rather than an invented one.
        record = _joint("text").self_check()
        self.assertEqual(record["likelihood_band"]["verdict"], "WITHHELD")
        self.assertIn("21_joint_mode_qualification.py", record["likelihood_band"]["reason"])

    def test_the_scored_span_is_the_renderings_own_and_excludes_the_delimiters(self):
        model = _joint("protein")
        inputs = model.render(SEQUENCES[:2])
        batch = model.batch(inputs)
        for row, text in enumerate(inputs):
            rendered = model.tokenisation.render(
                SEQUENCES[:2][row], context=None
            )
            positions = set(rendered.scored_positions)
            self.assertEqual(
                {int(index) for index in batch["content_mask"][row].nonzero().flatten()},
                positions,
            )
            self.assertEqual(
                {
                    int(index) + 1
                    for index in batch["target_mask"][row].nonzero().flatten()
                },
                positions,
            )
            # The rendering really did merge residues, which is why this family
            # declares the token as its symbol unit.
            self.assertGreater(rendered.residues_per_scored_token, 1.0)
            self.assertEqual(text, rendered.text)

    def test_a_string_that_did_not_come_from_render_has_no_scored_span(self):
        model = _joint("protein")
        with self.assertRaises(KeyError):
            model.batch(["Seq=<MKTA>"])

    def test_a_rendering_that_does_not_fit_the_token_budget_is_refused(self):
        model = _joint("protein", max_tokens=4)
        with self.assertRaises(ValueError) as caught:
            model.render(SEQUENCES[:1])
        self.assertIn("closing delimiter", str(caught.exception))

    def test_the_padding_id_is_resolved_from_a_declared_order_or_refused(self):
        model = _joint("text")
        self.assertEqual(model._pad_source, "eos_token_id")
        with self.assertRaises(ValueError):
            R.JointReplaceable(
                model=_llama(prollama_stub()),
                tokenizer=prollama_stub(declare_ids=False),
                checkpoint=Path("/nowhere"),
                declaration=JM.rendering("prollama"),
                mode="text",
            )

    def test_both_modes_run_through_the_identical_sweep(self):
        tokenizer = prollama_stub()
        declaration = JM.rendering("prollama")
        backbone = _llama(tokenizer)
        tokenisation = JM.resolve(tokenizer, declaration)
        modes = {}
        for mode, records, kind, source in (
            ("protein", SEQUENCES, "protein", "swissprot"),
            ("text", DOCUMENTS, "text", "openwebtext"),
        ):
            model = R.JointReplaceable(
                model=backbone,
                tokenizer=tokenizer,
                checkpoint=Path("/nowhere"),
                declaration=declaration,
                mode=mode,
                tokenisation=tokenisation if mode == "protein" else None,
                max_tokens=128,
            )
            with _patched_corpus(records, kind):
                with contextlib.redirect_stdout(io.StringIO()):
                    modes[mode] = STAGE.measure_mode(
                        _args(epsilons=[0.2], draws=1),
                        model,
                        source=source,
                        label=f"joint:{mode}",
                    )
        for mode, record in modes.items():
            self.assertAlmostEqual(
                record["sweep"][0]["across_draws"]["recovery"]["mean"], 1.0, msg=mode
            )
            self.assertTrue(
                np.isfinite(record["gates"]["endpoints"]["denominator_nats_per_token"]), mode
            )
        self.assertGreater(modes["protein"]["symbols_per_token"]["value"], 1.0)
        self.assertIn("residues", modes["protein"]["symbols_per_token"]["unit"])
        self.assertIn("characters", modes["text"]["symbols_per_token"]["unit"])
        cross = STAGE.cross_mode_record(modes)
        self.assertEqual(cross["magnitude_comparison"]["verdict"], "NOT_LICENSED")
        self.assertEqual(sorted(cross["per_mode"]), ["protein", "text"])
        with tempfile.TemporaryDirectory() as directory:
            write_json(Path(directory) / "modes.json", {"modes": modes, "cross": cross})


# --------------------------------------------------------- the parallel block


class TheParallelBlockTargetIsDeclared(unittest.TestCase):
    """What is perturbed on a GPT-J-style block, and how that is established.

    The scale ladder this stage reaches -- progen2-small through progen2-xlarge --
    is one ProGen2 lineage, and ProGen2's residual block is parallel. Everything
    below is about the one way that can go wrong without raising: perturbing a
    tensor whose relationship to the residual stream nobody checked, and reporting
    the number beside a serial arm's as though the two were the same manipulation.
    """

    def test_the_declared_target_is_the_feed_forward_output(self):
        model = _parallel()
        target = model.perturbation_target
        self.assertIn("feed-forward output", target["tensor"])
        self.assertTrue(target["block_layout"].startswith("parallel"))
        self.assertEqual(
            target["identity_verified"],
            "block output == (attention output + intercepted feed-forward output) "
            "+ ln_1 input",
        )
        # The attention writes the same residual on this layout and is left alone,
        # which is what makes the manipulation the same object it is on gpt2.
        self.assertIn("attention contribution", target["not_perturbed"])
        self.assertTrue(
            _dense().perturbation_target["block_layout"].startswith("serial")
        )

    def test_the_intercepted_tensor_IS_the_feed_forward_output(self):
        model = _parallel()
        batch = model.batch(model.render(SEQUENCES[:2]))
        seen: dict[int, torch.Tensor] = {}
        direct: dict[int, torch.Tensor] = {}
        handles = [
            block.mlp.register_forward_hook(
                lambda module, inputs, output, layer=layer: direct.__setitem__(
                    layer, output.detach().clone()
                )
            )
            for layer, block in enumerate(model.arm.blocks())
        ]

        def tap(layer, x, y):
            seen[layer] = y.detach().clone()
            return None

        try:
            with model.block_intercept(tap):
                model.run(batch)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(sorted(seen), list(range(N_LAYER)))
        for layer in range(N_LAYER):
            self.assertTrue(torch.equal(seen[layer], direct[layer]), layer)

    def test_the_declared_identity_holds_exactly_and_the_serial_one_does_not(self):
        # The whole reason the declaration exists. On a parallel block the block's
        # output is attn + ff + residual, so the serial reconstruction the GPT-2
        # arms are verified under is checking an equality that is simply false --
        # by 14.25 at bfloat16 on the real progen2-small, and by a wide margin here.
        model = _parallel()
        record = model.estimand_identity()
        self.assertEqual(record["verdict"], "PASS")
        self.assertEqual(record["max_absolute_difference"], 0.0)
        self.assertEqual(record["block_layout"], "parallel")
        self.assertEqual(record["n_layers"], N_LAYER)

        residual, attention, feed_forward, produced = {}, {}, {}, {}
        handles = []
        for layer, block in enumerate(model.arm.blocks()):
            handles.append(
                block.ln_1.register_forward_pre_hook(
                    lambda module, inputs, layer=layer: residual.__setitem__(
                        layer, inputs[0].detach()
                    )
                )
            )
            handles.append(
                block.attn.register_forward_hook(
                    lambda module, inputs, output, layer=layer: attention.__setitem__(
                        layer, output[0].detach()
                    )
                )
            )
            handles.append(
                block.mlp.register_forward_hook(
                    lambda module, inputs, output, layer=layer: feed_forward.__setitem__(
                        layer, output.detach()
                    )
                )
            )
            handles.append(
                block.register_forward_hook(
                    lambda module, inputs, output, layer=layer: produced.__setitem__(
                        layer, output[0].detach()
                    )
                )
            )
        try:
            model.run(model.batch(model.render(SEQUENCES[:2])))
        finally:
            for handle in handles:
                handle.remove()
        for layer in range(N_LAYER):
            declared = (attention[layer] + feed_forward[layer]) + residual[layer]
            serial = residual[layer] + feed_forward[layer]
            self.assertEqual(
                float((declared - produced[layer]).abs().max()), 0.0, f"layer {layer}"
            )
            self.assertGreater(
                float((serial - produced[layer]).abs().max()), 1e-3, f"layer {layer}"
            )

    def test_a_stub_whose_declared_identity_does_not_hold_is_refused(self):
        # A block that adds a term no interceptor sees. Nothing about it raises on
        # its own -- it loads, it runs, its cross-entropy is finite -- and the
        # identity check is the only thing between it and a complete artefact.
        with self.assertRaises(RuntimeError) as caught:
            _parallel(leak=0.25).estimand_identity()
        message = str(caught.exception)
        self.assertIn("ln_1 input", message)
        self.assertIn("not the residual write it is declared to be", message)

    def test_an_architecture_with_no_declaration_is_refused_rather_than_duck_typed(self):
        for architecture in ("llama", "qwen2", "opt", "t5_decoder", "reformer"):
            with self.assertRaises(TypeError, msg=architecture) as caught:
                R.residual_write(architecture)
            self.assertIn("no residual write is declared", str(caught.exception))
        self.assertTrue(R.residual_write("progen").parallel_attention)
        self.assertFalse(R.residual_write("gpt2").parallel_attention)
        # And the refusal reaches the constructor: an arm whose architecture this
        # implementation does not cover cannot be built at all.
        arm = _progen_arm()
        with self.assertRaises(TypeError):
            R.DenseReplaceable(arm, max_tokens=64)

    def test_nothing_on_this_path_reads_config_vocab_size(self):
        # The stub's config omits vocab_size the way progen2-xlarge's does and
        # raises AssertionError if anything reaches for it, so this is a run of the
        # real path rather than a search of the source.
        model = _parallel()
        with self.assertRaises(AssertionError):
            _ = model.arm.model.config.vocab_size
        inputs = model.render(SEQUENCES)
        self.assertEqual(model.n_heads, N_HEAD)
        STAGE.block_output_reference(model, inputs, batch_size=2)
        STAGE.symbols_per_token(model, inputs)
        model.estimand_identity()
        STAGE22_scored(
            model,
            inputs,
            factory=STAGE.perturbation_context(model, epsilon=0.2, seed=5),
        )

    def test_symbols_per_token_comes_from_the_tokenizer_and_is_one_per_residue(self):
        arm = _progen_arm()
        # A residue tokenizer emits one token per residue, so the expansion is
        # exactly 1.0 -- the property that makes a nats-per-token figure on this
        # lineage readable as nats per residue.
        self.assertEqual(A.symbols_per_token(arm, list(SEQUENCES), 512), 1.0)
        # Under the arm's own rendering the N-to-C control token adds one token and
        # no residue, so the measured expansion is n / (n + 1) -- read off the
        # tokenizer and the rendering, never off a vocabulary size.
        model = _parallel()
        inputs = model.render(SEQUENCES)
        residues = sum(len(sequence) for sequence in SEQUENCES)
        self.assertAlmostEqual(
            STAGE.symbols_per_token(model, inputs)["value"],
            residues / (residues + len(SEQUENCES)),
            places=12,
        )
        self.assertIn("residues", STAGE.symbols_per_token(model, inputs)["unit"])

    def test_the_control_token_is_context_and_the_residues_are_the_content(self):
        """The direction marker is context the model reads and content it is not.

        Both halves matter and they are different masks. The target rule never
        scores position 0 because nothing predicts the first position of a
        sequence, and every later position is scored -- the marker is part of the
        context those predictions are made from. The *content* mask is what stage
        23 averages the fully-ablated endpoint over, and the marker must be out of
        it: ProGen3's own residue mask names ``"1"`` and ``"2"`` in
        ``NON_RESIDUE_TOKENS``, so an endpoint that included this position would
        not be the quantity the arm it is compared against reports.

        This test asserted the opposite until it was measured. ProGen2 declares
        only ``<|pad|>``/``<|bos|>``/``<|eos|>`` special and the mask was built
        from ``all_special_ids`` alone, so the marker stayed in. On
        ``progen2-small`` at bfloat16 over 32 Swiss-Prot records of 64-246
        residues, that one position per record displaced the endpoint by 0.68 in
        relative norm at layer 1 and 0.59 at layer 2, and moved the layer-1 norm
        anchor from 1.1267 to 1.4993.
        """

        model = _parallel()
        marker = PROGEN_TOKENS.index("1")
        batch = model.batch(model.render(SEQUENCES[:1]))
        ids = batch["input_ids"][0].tolist()
        self.assertEqual(ids[0], marker)
        self.assertNotIn(marker, StubResidueTokenizer.all_special_ids)
        self.assertEqual(A.rendering_marker_ids(model.arm), (marker,))

        content = model.content_mask(batch)
        self.assertFalse(bool(content[0, 0]))
        self.assertEqual(int(content.sum()), len(SEQUENCES[0]))
        self.assertEqual(
            model._target_mask(batch)[0].tolist(),
            [True] * (len(ids) - 1),
        )

    def test_no_declared_rendering_marker_reaches_the_content_span(self):
        """The invariant, over the whole cohort and every marker the format adds.

        Written against :func:`src.transfer.arms.rendering_marker_ids` rather than
        against a literal id, so a format that starts prefixing a second marker is
        covered by this test the moment it declares it.
        """

        model = _parallel()
        batch = model.batch(model.render(SEQUENCES))
        content = model.content_mask(batch)
        markers = A.rendering_marker_ids(model.arm)
        self.assertTrue(markers)
        for marker in markers:
            self.assertFalse(bool((content & (batch["input_ids"] == marker)).any()))
        # And what remains is exactly the residues: nothing else was dropped.
        self.assertEqual(
            content.sum(1).tolist(), [len(sequence) for sequence in SEQUENCES]
        )

    def test_the_whole_sweep_runs_on_a_parallel_arm_and_serialises(self):
        model = _parallel()
        model.self_check = lambda: {"verdict": "PASS", "note": "stubbed for the test"}
        with _patched_corpus(SEQUENCES, "protein"):
            with contextlib.redirect_stdout(io.StringIO()):
                record = STAGE.measure_mode(
                    _args(), model, source="swissprot", label="progen2-small:protein"
                )
        endpoints = record["gates"]["endpoints"]
        self.assertEqual(endpoints["verdict"], "PASS")
        self.assertGreater(endpoints["denominator_nats_per_token"], 0.0)
        self.assertAlmostEqual(
            record["sweep"][0]["across_draws"]["recovery"]["mean"], 1.0
        )
        self.assertEqual(record["cohort"]["kind"], "protein")
        self.assertTrue(
            record["perturbation_target"]["block_layout"].startswith("parallel")
        )
        self.assertEqual(len(record["block_output_norm"]["mean_per_layer"]), N_LAYER)
        with tempfile.TemporaryDirectory() as directory:
            write_json(Path(directory) / "parallel.json", record)

    def test_the_ladder_is_reachable_and_the_transcoder_stages_still_refuse_it(self):
        import panel_contract

        admissible = R.perturbable_arms(panel_contract.CAMPAIGN_PANEL)
        self.assertEqual(admissible, STAGE.ADMISSIBLE_ARMS)
        for name in A.PROTEIN_SCALE_LADDER:
            self.assertIn(name, admissible, name)
            self.assertIn(name, _choices("arm"), name)
            self.assertNotIn(name, R.eligible_arms(panel_contract.CAMPAIGN_PANEL), name)


# ------------------------------------------------------------------ the target


class TheTargetIsResolvedBeforeAnythingIsLoaded(unittest.TestCase):
    def _resolve(self, **overrides):
        args = argparse.Namespace(
            arm=None,
            checkpoint=None,
            rendering=None,
            modes=None,
            epsilons=[0.1],
            draws=3,
        )
        vars(args).update(overrides)
        return STAGE.resolve_target(args)

    def test_a_checkpoint_without_a_rendering_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self._resolve(checkpoint=Path("/nowhere"))
        self.assertIn("--rendering", str(caught.exception))

    def test_a_checkpoint_defaults_to_both_modes(self):
        self.assertEqual(
            set(self._resolve(checkpoint=Path("/nowhere"), rendering="prollama")),
            {"text", "protein"},
        )
        self.assertEqual(
            self._resolve(
                checkpoint=Path("/nowhere"), rendering="prollama", modes="protein"
            ),
            ("protein",),
        )

    def test_a_panel_arm_takes_neither_a_rendering_nor_a_mode(self):
        self.assertEqual(self._resolve(arm="gpt2-large"), ())
        with self.assertRaises(ValueError):
            self._resolve(arm="gpt2-large", rendering="prollama")
        with self.assertRaises(ValueError):
            self._resolve(arm="gpt2-large", modes="both")

    def test_an_arm_and_a_checkpoint_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                STAGE.build_parser().parse_args(
                    ["--arm", "gpt2-large", "--checkpoint", "/nowhere"]
                )
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                STAGE.build_parser().parse_args([])

    def test_a_degenerate_sweep_is_refused(self):
        with self.assertRaises(ValueError):
            self._resolve(arm="gpt2-large", draws=0)
        with self.assertRaises(ValueError):
            self._resolve(arm="gpt2-large", epsilons=[-0.1])

    def test_the_matched_standalone_pair_runs_through_the_identical_path(self):
        # Without them a joint-model difference could not be attributed between
        # joint training, modality, tokenizer and lineage.
        for name in ("gpt2-large", "protgpt2"):
            self.assertIn(name, _choices("arm"))

    def test_an_arm_with_no_symbols_per_token_convention_is_refused(self):
        with self.assertRaises(TypeError) as caught:
            STAGE.symbols_per_token(object(), ["a"])
        self.assertIn("symbols-per-token", str(caught.exception))


class TheDriverWritesOneArtefactForOneCheckpoint(unittest.TestCase):
    """The whole driver, on stubs: the part no CPU host can otherwise reach.

    A 7B joint checkpoint does not fit on the hardware this repository validates
    on, so the driver's wiring -- the loaders, the two-mode loop, the cross-mode
    record and the serialisation -- would otherwise be exercised for the first
    time on a cluster. The loaders and the corpus are replaced by stubs; every
    line between them is the real one.
    """

    def test_both_modes_reach_one_artefact_with_their_own_denominators(self):
        tokenizer = prollama_stub()
        backbone = _llama(tokenizer)
        facts = {
            "resolved_path": "/nowhere",
            "model_type": "llama",
            "n_layers": N_LAYER,
            "d_model": D_MODEL,
            "n_heads": N_HEAD,
            "vocab_size": len(tokenizer),
            "dtype_requested": "float32",
            "dtype_observed": ["float32"],
        }
        saved = (
            STAGE.STAGE21.load_tokenizer,
            STAGE.STAGE21.load_model,
            STAGE.checkpoint_weights_digest,
            sys.argv,
        )
        STAGE.STAGE21.load_tokenizer = lambda path: (Path(path), tokenizer)
        STAGE.STAGE21.load_model = lambda resolved, tok, *, device, dtype: (
            backbone,
            dict(facts),
        )
        STAGE.checkpoint_weights_digest = lambda path: "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            sys.argv = [
                "23_perturbation_sensitivity.py",
                "--checkpoint", "/nowhere",
                "--rendering", "prollama",
                "--device", "cpu",
                "--dtype", "float32",
                "--sequences", "4",
                "--batch-size", "2",
                "--bootstrap", "32",
                "--epsilons", "0.2",
                "--draws", "2",
                "--max-tokens", "128",
                "--out", directory,
            ]
            try:
                with _patched_corpus(SEQUENCES, "protein"), _patched_corpus(
                    DOCUMENTS, "text"
                ):
                    with contextlib.redirect_stdout(io.StringIO()):
                        STAGE.main()
                payload = json.loads(
                    (Path(directory) / "perturbation_sensitivity.json").read_text()
                )
            finally:
                (
                    STAGE.STAGE21.load_tokenizer,
                    STAGE.STAGE21.load_model,
                    STAGE.checkpoint_weights_digest,
                    sys.argv,
                ) = saved

        self.assertEqual(payload["schema_version"], STAGE.SCHEMA_VERSION)
        self.assertEqual(sorted(payload["modes"]), ["protein", "text"])
        self.assertEqual(payload["target"]["kind"], "joint_checkpoint")
        self.assertEqual(payload["target"]["rendering_family"], "prollama")
        self.assertEqual(payload["target"]["weights_sha256"], "0" * 64)
        self.assertEqual(
            sorted(payload["provenance"]["modules"]), sorted(STAGE.PROVENANCE_MODULES)
        )
        self.assertIn("epsilon_anchored_at", payload["perturbation"])
        self.assertIn("NOT bounded below by zero", payload["perturbation"]["range_note"])

        # Each mode carries its own denominator in nats and its own expansion, and
        # only the ratio is licensed between them.
        denominators = set()
        for mode, record in payload["modes"].items():
            self.assertEqual(record["cohort"]["kind"], mode)
            denominators.add(record["gates"]["endpoints"]["denominator_nats_per_token"])
            self.assertGreater(record["symbols_per_token"]["value"], 0.0)
            self.assertEqual(record["sweep"][0]["epsilon"], 0.0)
            self.assertEqual(record["sweep"][0]["n_draws"], 1)
            self.assertEqual(record["sweep"][1]["n_draws"], 2)
        self.assertEqual(len(denominators), 2, "the two modes must not share a floor")
        self.assertEqual(
            payload["cross_mode"]["magnitude_comparison"]["verdict"], "NOT_LICENSED"
        )
        self.assertTrue(payload["cross_mode"]["readable_only_with_two_modes"])
        # The protein rendering is resolved and its facts travel with the run; the
        # text mode says in its own field that it did not need one.
        self.assertEqual(payload["modes"]["protein"]["rendering"]["name"], "prollama")
        self.assertEqual(
            payload["modes"]["protein"]["rendering"]["symbol_unit"], JM.TOKEN_UNIT
        )
        self.assertEqual(payload["modes"]["text"]["rendering"]["verdict"], "NOT_RESOLVED")
        self.assertEqual(
            payload["settings"]["cohort_draw_seed"], A.DEFAULT_CORPUS_DRAW_SEED
        )


class StageWiring(unittest.TestCase):
    def test_the_stage_is_not_registered_in_the_panel_contract(self):
        import panel_contract

        self.assertNotIn("perturbation_sensitivity", panel_contract.STAGE_CONTRACTS)

    def test_the_provenance_modules_exist_and_include_the_splice(self):
        self.assertIn("src/transfer/replaceable.py", STAGE.PROVENANCE_MODULES)
        for name in STAGE.PROVENANCE_MODULES:
            self.assertTrue((REPO / name).exists(), name)

    def test_the_cohort_band_and_draw_are_arguments_with_the_declared_defaults(self):
        defaults = {action.dest: action.default for action in STAGE.build_parser()._actions}
        self.assertEqual(defaults["protein_min_len"], 64)
        self.assertEqual(defaults["protein_max_len"], 246)
        self.assertEqual(defaults["cohort_draw_seed"], A.DEFAULT_CORPUS_DRAW_SEED)
        self.assertEqual(defaults["text_min_chars"], 800)

    def test_the_default_sweep_is_a_curve_with_several_draws_at_each_point(self):
        self.assertGreaterEqual(len(STAGE.DEFAULT_EPSILONS), 5)
        self.assertEqual(list(STAGE.DEFAULT_EPSILONS), sorted(STAGE.DEFAULT_EPSILONS))
        self.assertGreater(min(STAGE.DEFAULT_EPSILONS), 0.0)
        self.assertGreaterEqual(STAGE.DEFAULT_DRAWS_PER_EPSILON, 3)

    def test_the_shared_endpoints_are_imported_and_not_restated(self):
        # Appendix B rule 12. The fully-ablated endpoint, the per-sequence sweep,
        # the paired recovery record and the joint loader all exist already; a
        # second copy of any of them would make this stage's ratios a different
        # measurement from the ones they are meant to be compared with.
        source = Path(STAGE.__file__).read_text(encoding="utf-8")
        for name in (
            "def mean_ablation_context",
            "def scored_cross_entropy",
            "def recovery_record",
            "def bootstrap_indices",
            "def load_model",
        ):
            self.assertNotIn(name, source, f"{name} is restated rather than imported")
        for attribute in ("mean_ablation_context", "matched_perturbation_context"):
            self.assertTrue(hasattr(STAGE.STAGE15, attribute), attribute)
        for attribute in ("load_tokenizer", "load_model"):
            self.assertTrue(hasattr(STAGE.STAGE21, attribute), attribute)
        for attribute in ("scored_cross_entropy", "recovery_record", "bootstrap_indices"):
            self.assertTrue(hasattr(STAGE.STAGE22, attribute), attribute)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
