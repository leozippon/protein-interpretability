"""``opt`` as an interpretability-method architecture: what it may enter, and what it may not.

Galactica is the only checkpoint family available to this programme that spans
both modalities at four scale rungs, so it is the one addition that can put a
modality-by-scale grid under an interpretability method instead of leaving every
modality coefficient resting on one arm. This file pins what that support is and,
just as deliberately, what it is not.

``opt`` is declared in exactly two places: :meth:`src.transfer.arms.Arm.blocks`
and :data:`src.transfer.lenses.FINAL_LAYER_NORM_PATH`, which together are what a
lens needs -- the block outputs, and the model's own output head. It is declared
in none of the tables that carry a *decomposition*, and the tests below assert
those absences against the measured architectural facts that produce them rather
than against a list somebody wrote: an ``OPTDecoderLayer`` has no feed-forward
submodule and flattens batch and time before ``fc2``, and an ``OPTDecoder``
builds its initial residual inline from two embeddings, so neither the panel's
MLP tensor nor its embedding tensor exists as a module output.

The checkpoint-backed tests skip when ``galactica-125m`` is not staged; the rest
run everywhere.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.transfer import arms as A  # noqa: E402
from src.transfer import circuits, lenses, scaling  # noqa: E402
from src.transfer.arms import Arm, ArmSpec  # noqa: E402

#: The staged checkpoint every checkpoint-backed test below runs on. The smallest
#: rung of the ladder, because these assert layout and identity rather than
#: capability and the layout is the same at every rung.
CHECKPOINT = A.MODEL_ROOT / "galactica-125m"

#: Its config, restated so that a shape disagreement fails as a declaration
#: mismatch inside ``load_arm_spec`` rather than as a confusing tensor error.
N_LAYER = 12
D_MODEL = 768
VOCAB = 50000


def _spec(**overrides: object) -> ArmSpec:
    """A galactica-125m declaration, as a ladder rung would build one."""

    fields: dict[str, object] = {
        "name": "galactica-125m",
        "path": CHECKPOINT,
        "path_variable": "TRANSFER_MODEL_BASE_DIR",
        "modality": "text",
        "n_layer": N_LAYER,
        "d_model": D_MODEL,
        "tokenisation": "bpe",
        "input_format": "raw",
        "evaluation_cohort_source": "openwebtext",
        "architecture": "opt",
        "pretraining_corpus": "galactica_scientific_corpus",
        "capabilities": frozenset({"budget", "lens"}),
        "scoring_target_alphabet_size": VOCAB,
    }
    fields.update(overrides)
    return ArmSpec(**fields)  # type: ignore[arg-type]


def _stub_arm(model: object, **overrides: object) -> Arm:
    return Arm(
        spec=_spec(**overrides),
        model=model,
        tokenizer=object(),
        device="cpu",
        dtype="float32",
    )


# ----------------------------------------------------------------- declarations


def test_opt_is_declared_only_in_the_tables_it_was_verified_against():
    """Two entries, and four deliberate absences with architectural reasons."""

    assert "opt" in lenses.FINAL_LAYER_NORM_PATH
    assert "opt" in scaling.LENS_ARCHITECTURES
    # The block list is not a table; it is a branch, and this is what it resolves.
    assert lenses.FINAL_LAYER_NORM_PATH["opt"] == ("model", "decoder", "final_layer_norm")

    assert "opt" not in A._ATTENTION_PATH
    assert "opt" not in A._DECOMPOSABLE
    assert "opt" not in A._MLP_NEURON_TENSOR
    assert "opt" not in circuits._CIRCUIT_ARCHITECTURES


def test_the_lens_architecture_set_is_the_lens_modules_own_table():
    """One declaration, read in two modules, so neither can drift from the other."""

    assert scaling.LENS_ARCHITECTURES == tuple(sorted(lenses.FINAL_LAYER_NORM_PATH))


def test_lens_support_is_decided_by_that_table_and_names_it_when_refusing():
    member = scaling.LadderMember(
        name="galactica-125m",
        path=CHECKPOINT,
        modality="text",
        tokenisation="bpe",
        input_format="raw",
        evaluation_cohort_source="openwebtext",
        cohort_corpus=scaling.TEXT_CORPUS,
        cohort_min_symbols=64,
        cohort_max_symbols=0,
        architecture="opt",
    )
    supported, reason = scaling.lens_supported(member)
    assert supported and reason is None

    refused, reason = scaling.lens_supported(replace(member, architecture="rita"))
    assert not refused
    assert reason is not None
    assert "FINAL_LAYER_NORM_PATH" in reason and "rita" in reason


def test_a_decomposition_refuses_opt_even_when_the_capability_is_granted():
    """The capability is an intent; the table is what the module can serve."""

    arm = _stub_arm(
        SimpleNamespace(config=SimpleNamespace(vocab_size=VOCAB)),
        capabilities=frozenset(A.CAPABILITIES),
    )
    for accessor in (arm.mlp, arm.attention):
        with pytest.raises(TypeError, match="not defined for 'opt'"):
            accessor(0)
    with pytest.raises(TypeError, match="no attention submodule is declared"):
        arm.attention_pattern_module(0)
    with pytest.raises(TypeError, match="not defined for 'opt'"):
        circuits.circuit_architecture(arm)
    with pytest.raises(TypeError, match="no MLP hidden-activation tensor"):
        A.mlp_neuron_declaration("opt")


# ------------------------------------------------------------- module resolution


def test_the_block_list_is_refused_when_the_checkpoint_is_not_laid_out_like_opt():
    """A declaration that does not match the loaded object stops here.

    ``OPTForCausalLM`` keeps its decoder two attributes down. A checkpoint that
    stops one level short is a rotary decoder wearing an ``opt`` declaration, and
    resolving it to whatever ``.model`` happened to hold is the failure the
    branch exists to prevent.
    """

    rotary_shaped = SimpleNamespace(model=SimpleNamespace(layers=nn.ModuleList()))
    with pytest.raises(TypeError, match="no model.decoder.layers"):
        _stub_arm(rotary_shaped).blocks()
    with pytest.raises(TypeError, match="no model.decoder.layers"):
        _stub_arm(SimpleNamespace()).blocks()


def test_the_lens_head_refuses_a_post_norm_opt():
    """``do_layer_norm_before`` false leaves no final normalisation, and must refuse.

    The missing module is the mechanism; the reason is stronger than that. A
    post-norm block normalises its own output, so the tensor a lens would read at
    depth is not the residual stream the head was trained to consume.
    """

    decoder = SimpleNamespace(final_layer_norm=None)
    model = SimpleNamespace(
        model=SimpleNamespace(decoder=decoder),
        lm_head=nn.Linear(D_MODEL, VOCAB, bias=False),
        config=SimpleNamespace(vocab_size=VOCAB),
    )
    with pytest.raises(TypeError, match="no model.decoder.final_layer_norm"):
        lenses.lens_head(_stub_arm(model))


def test_the_lens_head_refuses_an_opt_that_projects_out_of_the_residual_basis():
    """``word_embed_proj_dim != hidden_size`` puts a projection under the head.

    That projection sits between the final normalisation and the unembedding, so
    a head built from those two alone would read a basis the unembedding is never
    applied to. It is refused by the width the head itself declares, which is
    what makes the check independent of the architecture name.
    """

    decoder = SimpleNamespace(final_layer_norm=nn.LayerNorm(D_MODEL))
    model = SimpleNamespace(
        model=SimpleNamespace(decoder=decoder),
        lm_head=nn.Linear(512, VOCAB, bias=False),
        config=SimpleNamespace(vocab_size=VOCAB),
    )
    with pytest.raises(ValueError, match="lm_head reads width 512"):
        lenses.lens_head(_stub_arm(model))


def test_the_final_normalisation_of_an_undeclared_architecture_is_refused_by_name():
    arm = _stub_arm(SimpleNamespace(), architecture="rita")
    with pytest.raises(TypeError, match="no final normalisation is declared"):
        lenses.final_layer_norm(arm)


def test_a_final_normalisation_of_the_wrong_form_is_refused():
    """This head applies a centring and a learned bias; RMSNorm has neither."""

    decoder = SimpleNamespace(final_layer_norm=nn.Identity())
    model = SimpleNamespace(
        model=SimpleNamespace(decoder=decoder),
        lm_head=nn.Linear(D_MODEL, VOCAB, bias=False),
        config=SimpleNamespace(vocab_size=VOCAB),
    )
    with pytest.raises(TypeError, match="not a\\s+LayerNorm"):
        lenses.lens_head(_stub_arm(model))


# --------------------------------------------------------------- the pad token


class _Vocabulary:
    """The two calls the pad resolution makes, over a fixed id-to-token map."""

    def __init__(self, tokens: dict[int, str]) -> None:
        self._tokens = tokens
        self.pad_token: str | None = None

    def convert_ids_to_tokens(self, index: int):
        return self._tokens.get(int(index))

    @property
    def pad_token_id(self):
        for index, token in self._tokens.items():
            if token == self.pad_token:
                return index
        return None


def test_a_config_declared_pad_token_is_adopted_only_after_the_vocabulary_confirms_it():
    tokenizer = _Vocabulary({0: "<s>", 1: "<pad>", 2: "</s>"})
    A.adopt_config_declared_pad_token(
        tokenizer, SimpleNamespace(pad_token_id=1), arm="galactica"
    )
    assert tokenizer.pad_token == "<pad>"
    assert tokenizer.pad_token_id == 1


def test_a_pad_id_the_vocabulary_cannot_name_stops_the_load():
    """The declared id must exist as a token, or nothing is padded with it."""

    tokenizer = _Vocabulary({0: "<s>", 1: "<pad>", 2: "</s>"})
    with pytest.raises(ValueError, match="does not map to a token"):
        A.adopt_config_declared_pad_token(
            tokenizer, SimpleNamespace(pad_token_id=50256), arm="rita-like"
        )
    assert tokenizer.pad_token is None


def test_a_config_that_declares_no_pad_id_leaves_the_tokenizer_alone():
    """Nothing to establish, so nothing is invented; the batch door still refuses.

    This is ``rita-xl``: no ``pad_token_id`` in the config at all, so the loader
    adds nothing and :func:`src.transfer.arms.tokenize_batch` refuses every batch
    exactly as it did before this step existed.
    """

    tokenizer = _Vocabulary({0: "<s>", 1: "<pad>"})
    A.adopt_config_declared_pad_token(tokenizer, SimpleNamespace(), arm="rita-xl")
    assert tokenizer.pad_token is None
    A.adopt_config_declared_pad_token(
        tokenizer, SimpleNamespace(pad_token_id=None), arm="rita-xl"
    )
    assert tokenizer.pad_token is None


def test_a_pad_assignment_that_does_not_read_back_stops_the_load():
    """The id is read back, so an ambiguous vocabulary cannot pass silently."""

    # Two ids spell the same token, so the assignment resolves to the first.
    tokenizer = _Vocabulary({1: "<pad>", 7: "<pad>"})
    with pytest.raises(ValueError, match="read back as 1"):
        A.adopt_config_declared_pad_token(
            tokenizer, SimpleNamespace(pad_token_id=7), arm="ambiguous"
        )


# ------------------------------------------------------ against the checkpoint

pytestmark_checkpoint = pytest.mark.skipif(
    not (CHECKPOINT / "config.json").is_file(),
    reason="galactica-125m is not staged on this host",
)


@pytest.fixture(scope="module")
def galactica():
    if not (CHECKPOINT / "config.json").is_file():
        pytest.skip("galactica-125m is not staged on this host")
    return A.load_arm_spec(_spec(), device="cpu", dtype="float32", strict=True)


@pytestmark_checkpoint
def test_the_block_list_is_the_decoders_own_layers(galactica):
    blocks = galactica.blocks()
    assert isinstance(blocks, torch.nn.ModuleList)
    assert len(blocks) == N_LAYER == galactica.n_layer
    inner = galactica.model.model.decoder
    assert blocks is inner.layers
    assert all(blocks[i] is inner.layers[i] for i in range(len(blocks)))
    assert all(type(block).__name__ == "OPTDecoderLayer" for block in blocks)


@pytestmark_checkpoint
def test_the_pad_token_is_resolved_from_the_checkpoints_own_config(galactica):
    """Galactica declares no special token anywhere the tokenizer reads."""

    assert galactica.tokenizer.pad_token == "<pad>"
    assert galactica.tokenizer.pad_token_id == 1
    ids, mask = A.tokenize_batch(
        galactica, ["A short document.", "A rather longer document about proteins."], 64
    )
    assert ids.shape[0] == 2
    assert int(mask[0].sum()) < int(mask[1].sum())
    assert int(ids[0, -1]) == 1


@pytestmark_checkpoint
def test_the_lens_head_reproduces_the_models_own_final_distribution(galactica):
    """The self-consistency check: the head at the last layer *is* the model.

    Nothing downstream of a lens means what it claims if this fails, so it is
    asserted on real cohort positions at a tolerance a wrong head cannot meet.
    """

    head = lenses.lens_head(galactica)
    assert head.d_model == D_MODEL
    assert head.vocab_size == VOCAB
    assert head.bias is None
    assert lenses.final_layer_norm(galactica) is galactica.model.model.decoder.final_layer_norm

    ids, mask = A.tokenize_batch(
        galactica,
        [
            "The mitochondrion is the powerhouse of the cell, and its inner "
            "membrane hosts the electron transport chain.",
            "Title: A study of protein folding kinetics under thermal stress",
        ],
        64,
    )
    window = lenses.ScoredWindow(
        input_ids=ids,
        attention_mask=mask,
        target_mask=mask[:, 1:].bool(),
        sequence_indices=(0, 1),
    )
    report = lenses.verify_lens_head(galactica, head, window, tolerance_nats=1e-6)
    assert report["positions"] > 0
    assert report["max_kl_nats"] < 1e-6


@pytestmark_checkpoint
def test_a_lens_read_at_depth_is_finite_and_ends_on_the_models_own_prediction(galactica):
    head = lenses.lens_head(galactica)
    ids, mask = A.tokenize_batch(
        galactica, ["Proteins fold into structures determined by their sequence."], 64
    )
    targets = ids[:, 1:]
    scored = mask[:, 1:].bool()
    with torch.no_grad(), lenses._BlockCapture(
        galactica, tuple(range(galactica.n_layer))
    ) as capture:
        logits = galactica.model(
            input_ids=ids, attention_mask=mask, use_cache=False
        ).logits.float()

    def mean_nll(log_probs: torch.Tensor) -> float:
        rows = log_probs.gather(-1, targets[scored].unsqueeze(-1)).squeeze(-1)
        return float(-rows.mean())

    depths = []
    for layer in range(galactica.n_layer):
        state = capture.captured[layer][:, :-1, :][scored]
        log_probs = head.log_probs(state)
        assert torch.isfinite(log_probs).all()
        depths.append(mean_nll(log_probs))

    model_nll = mean_nll(
        torch.log_softmax(logits[:, :-1], dim=-1)[scored]
    )
    # The lens at the last block is the model, and the trajectory descends to it
    # rather than wandering: an aperture that is not the model's own would not
    # land on the model's own number.
    assert depths[-1] == pytest.approx(model_nll, abs=1e-4)
    assert depths[0] > depths[-1]


@pytestmark_checkpoint
def test_the_sublayer_decomposition_is_exact_but_not_in_the_panels_shape(galactica):
    """Why ``opt`` is out of ``_DECOMPOSABLE``, asserted rather than described.

    The values close exactly -- block input plus attention output plus
    feed-forward output *is* the block output -- so the refusal is not about
    arithmetic. It is that an ``OPTDecoderLayer`` has no feed-forward submodule
    at all, and the one module whose output is that term emits a tensor of a
    different rank from every other arm's, because the block flattens batch and
    time before it. A hook that read it as a panel MLP output would be reading a
    differently shaped object under the same name.
    """

    block = galactica.blocks()[galactica.n_layer // 2]
    assert not hasattr(block, "mlp")
    assert {"fc1", "fc2", "self_attn"} <= {name for name, _ in block.named_children()}

    captured: dict[str, torch.Tensor] = {}
    handles = [
        block.register_forward_pre_hook(
            lambda _m, args: captured.__setitem__("in", args[0].detach().float())
        ),
        block.register_forward_hook(
            lambda _m, _a, out: captured.__setitem__(
                "out", (out[0] if isinstance(out, tuple) else out).detach().float()
            )
        ),
        block.self_attn.register_forward_hook(
            lambda _m, _a, out: captured.__setitem__(
                "attn", (out[0] if isinstance(out, tuple) else out).detach().float()
            )
        ),
        block.fc2.register_forward_hook(
            lambda _m, _a, out: captured.__setitem__("ff", out.detach().float())
        ),
    ]
    ids, mask = A.tokenize_batch(
        galactica, ["Alpha helices and beta sheets.", "Enzyme catalysis rates."], 32
    )
    try:
        galactica.model(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    block_out = captured["out"]
    assert captured["attn"].shape == block_out.shape
    # The feed-forward term is flattened over batch and token; that is the fact
    # the absence from _DECOMPOSABLE rests on.
    assert captured["ff"].shape == (block_out.shape[0] * block_out.shape[1], D_MODEL)
    residual = captured["in"] + captured["attn"] + captured["ff"].view(block_out.shape)
    assert float((residual - block_out).abs().max()) < 1e-5


@pytestmark_checkpoint
def test_no_module_emits_the_residual_entering_block_zero(galactica):
    """Why ``opt`` is out of ``circuits``, asserted rather than described.

    ``circuits.component_reconstruction`` sums an embedding term with every
    sublayer term and requires the sum to close on the final residual. On an OPT
    decoder the embedding term is not a module output: the token and learned
    position embeddings are added inline, so hooking the token table alone misses
    the position term entirely.
    """

    decoder = galactica.model.model.decoder
    captured: dict[str, torch.Tensor] = {}
    handles = [
        decoder.embed_tokens.register_forward_hook(
            lambda _m, _a, out: captured.__setitem__("tokens", out.detach().float())
        ),
        galactica.blocks()[0].register_forward_pre_hook(
            lambda _m, args: captured.__setitem__("residual", args[0].detach().float())
        ),
    ]
    ids, mask = A.tokenize_batch(galactica, ["Ribosomes translate messenger RNA."], 32)
    try:
        galactica.model(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    gap = float((captured["tokens"] - captured["residual"]).abs().max())
    assert gap > 1e-3, "the position term would have to be absent for this to be zero"
    assert decoder.project_in is None and decoder.project_out is None
