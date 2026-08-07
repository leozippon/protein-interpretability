"""ProGen3-112M, loaded so that it cannot come back silently random.

**Why this module exists.** The entry point everyone calls,
``ProGen3ForCausalLM.from_pretrained(path, moe_implementation="eager")``,
*succeeds* on the released checkpoint. It emits a "newly initialized" warning,
returns a model whose embeddings, attention and norms are loaded correctly, and
whose every expert and every router is freshly random -- because the released
weights are in **megablocks packing** and carry no key the eager
``SparseMoeBlock`` recognises. Nothing raises. The model generates. It scores
Swiss-Prot at **17.15 nats/token** against the **1.983** the same checkpoint
reaches once the experts are actually loaded (EXP: ``progen3_eager_probe``,
steps 08-09, one L20). A measurement taken on that object is a measurement of
noise wearing a checkpoint's name, and no exception separates the two.

:func:`load_progen3` is therefore the only supported way into this checkpoint in
this repository. It converts the packed tensors, loads them with
``strict=True``, and refuses on any missing or unexpected key.

**The conversion**, verified against megablocks 0.7.0's ``LearnedRouter`` and
``MemoryOptimizedGroupedGLU`` and against the eager block's own forward, with
``9216 = 8 experts x 1152``:

===========================================  ===========================================
released (megablocks packing)                eager ``SparseMoeBlock``
===========================================  ===========================================
``...block_sparse_moe.experts.mlp.w1``       ``...experts.{e}.w1.weight``  split
``...block_sparse_moe.experts.mlp.v1``       ``...experts.{e}.w3.weight``  split
``...block_sparse_moe.experts.mlp.w2``       ``...experts.{e}.w2.weight``  split + transpose
``...block_sparse_moe.router.layer.weight``  ``...block_sparse_moe.gate.weight``  rename
``mlm_head.weight``                          dropped: ``ProGen3ForCausalLM`` has no MLM head
===========================================  ===========================================

**The mapping is load-bearing and the evidence says a wrong one is unmissable.**
Under the mapping above: UniRef50 2.588 (the paper reports ~2.50), Swiss-Prot
1.983, residue-shuffled Swiss-Prot 2.940, uniform over 20 residues 2.996.
Corrupting the mapping while keeping ``strict=True`` clean: ``w1``/``v1``
swapped 3.201, gate rows rolled by one expert 3.173. Both are *worse than
shuffled protein*, so a mapping error cannot hide inside a plausible number --
provided something looks. :func:`self_check` is the thing that looks.

**What this module deliberately does not do.** It imports the patched
third-party ``progen3`` package by path and nothing else from that tree; the
tree is CC BY-NC-ND, git-ignored, and must not be vendored. It needs neither
megablocks (the eager block replaces it) nor flash-attn (a pure-PyTorch fp32
RMSNorm replaces the fused kernel); both substitutions are local patches to the
ignored copy, not to this repository.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
import torch.nn.functional as F

from .arms import REPO

#: The released ProGen3-112M checkpoint directory.
PROGEN3_CHECKPOINT = Path(
    os.environ.get("TRANSFER_PROGEN3_DIR", "/Data/public/progen3-112m")
)

#: Import root of the patched third-party ``progen3`` package. Resolved at call
#: time rather than at import, so this module can be imported -- and its pure
#: functions tested -- on a host that carries neither the tree nor the weights.
PROGEN3_SOURCE = Path(
    os.environ.get(
        "TRANSFER_PROGEN3_SRC",
        REPO / "external_resources/baselines/ProGenMech/external/progen3/src",
    )
)

#: Where a packed expert tensor's rows go, keyed by the released tensor's name.
#: ``v1`` is the *up* projection and lands on ``w3``; ``w1`` is the gate. Getting
#: those two the wrong way round is the 3.201-nat control above.
EXPERT_TARGETS = {"w1": "w1", "v1": "w3", "w2": "w2"}

#: Released tensors with no counterpart in ``ProGen3ForCausalLM``.
DROPPED_KEYS = ("mlm_head.weight",)

#: Eight Swiss-Prot sequences, 85-138 residues, taken at a fixed stride from
#: ``uniprot_sprot.fasta.gz`` and frozen here as literals. Literals rather than a
#: cohort draw because the self-check has to work in an environment that has the
#: weights and no corpus, and because a self-check whose input can move is not a
#: check. Accessions, in order: Q8YYH3, B2TK01, Q9CQM1, Q8SPI2, A7ZV06, B7JXS8,
#: P12104, Q4QMV0.
SELF_CHECK_SEQUENCES: tuple[str, ...] = (
    "MPNSTPQSQLIRAHVFVTGRVQGVGFRYSTVDTASQLGLTGWVRNLPDGRVEAVFEGVRDIVEDMVRWCHAGPPAAVVQDVAVEYEEPEGLRGFEVKRLVK",
    "MADTFLLKIVTPDKDIFNGNIKRIFLKNSVGRLEILANHANMVTSTISSIVEFTDADGKDRKLFISKGIASIFNNEMTIFSESAEFSDNIDLNRAEKAKERAEKRLLEGNKYDKERAELALLRSIERINLKKMN",
    "MIGGNTTIISGAINASTEAPGLGTGGRAWPVLVGVVLGAVVLSILIALAAKCHLCRRYHASYRHRPLSSAGGGNRPPVGEDEDDDGFIEDNYIQPGAGEMETTGSRDHFSL",
    "MPVVTGRLRDPDINPCLSESDASTRCLDENNYDKERCSTYFLKYKNCRKFWHSIMMQRRRNGVKPCMPTAAERDEILRAMGKMPY",
    "MLDEKSSNTASVVVLCTAPDEATAQDLAAKVLAEKLAACATLIPGATSLYYWEGKLEQEYEVQMILKTTVSHQQALLECLKSHHPYQTPELLVLPVTHGDTDYLSWLNASLR",
    "MNDYTRTIRLSDTDAAGVVYFASLLSICHEAYEASLEASGIDLKSFFRDSEVVIPIVHAEIDFFRPLYSGDRIIITLTTLQLKDTEFEITYQVGLVAPQSSLIAKAKTRHVAINPQTRQRTPLSESLMQWLKSTENSE",
    "MAFDSTWKVDRSENYDKFMEKMGVNIVKRKLAAHDNLKLTITQEGNKFTVKESSTFRNIEVVFELGVTFNYNLADGTELRGTWSLEGNKLIGKFKRTDNGNELNTVREIIGDELVQTYVYEGVEAKRIFKKD",
    "MQALLFISYGAILGASLRWAIGLLFNPLFSSFAFGTLIANLLGCLIIGVLLGFFWQFPQISSEWRLFLITGFLGSLTTFSSFSSEVVELFFNDKWLNGFCVLMMHLFGCLAMTVLGIWIYKICSQLLS",
)

#: Mean per-token NLL of :data:`SELF_CHECK_SEQUENCES` under this module's own
#: scoring convention (both directions, every non-pad target, bfloat16, one L20),
#: for the correct mapping and for every corruption that survives
#: ``strict=True``. Measured, not quoted: the panel-wide figures in the module
#: docstring were taken on 64 Swiss-Prot records at 60-400 residues and do not
#: transfer to this eight-record set.
MEASURED_SELF_CHECK_NLL = {
    "correct_mapping": 2.2867,
    "w1_v1_swapped": 3.1793,
    "gate_rows_rolled_by_one": 3.3055,
    "from_pretrained_eager_random_moe": 18.4764,
}

#: The band :func:`check_nll` accepts, in nats/token.
#:
#: Sized from two measurements rather than from taste. The **spread of the
#: correct model** across everything an environment can plausibly change is
#: 0.005 nats: bfloat16 and float16 at batch sizes 8, 4 and 1 give 2.2867-2.2879,
#: and scoring N->C only instead of both directions gives 2.2912. The **distance
#: to the nearest corruption** that ``strict=True`` cannot see is 0.89 nats
#: (``w1``/``v1`` swapped, 3.1793). A band of +/-0.3 nats around 2.2867 is
#: therefore ~60x the observed spread and still leaves 0.58 nats of clearance
#: below the nearest corruption, so it cannot fail on hardware and cannot pass on
#: a broken mapping.
#:
#: The lower end is not a numerical tolerance. A value materially *below* the
#: measured one means the scored-target convention moved -- a mask that stopped
#: scoring the hard positions, say -- which corrupts every downstream number just
#: as thoroughly as a wrong expert mapping and would otherwise look like an
#: improvement.
SELF_CHECK_NLL_BAND = (2.00, 2.60)

#: ProGen3's attention calls ``F.scaled_dot_product_attention`` inside
#: ``sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])``, and that backend has
#: no float32 kernel: a float32 model raises ``RuntimeError: No available
#: kernel`` from inside the first layer. Half precision is not a performance
#: choice here, it is the only precision this checkpoint runs in, so
#: :func:`load_progen3` refuses the others up front rather than letting an
#: unrelated-looking kernel error surface ten frames down.
SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)


# ------------------------------------------------------------------ conversion


def convert_megablocks_state_dict(
    raw: dict[str, torch.Tensor], *, num_experts: int, intermediate_size: int
) -> dict[str, torch.Tensor]:
    """Rewrite released megablocks-packed weights for the eager ``SparseMoeBlock``.

    Pure and checkpoint-free, so the mapping can be tested without the weights.
    ``num_experts`` and ``intermediate_size`` come from the model config and are
    checked against every packed tensor: a packed row count that is not
    ``num_experts * intermediate_size`` would otherwise be split into
    wrong-sized slices and reported by ``load_state_dict`` as a shape mismatch
    on a key whose name says nothing about which of the two numbers was wrong.
    """

    expected_rows = num_experts * intermediate_size
    converted: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if key in DROPPED_KEYS:
            continue
        if ".block_sparse_moe.router.layer.weight" in key:
            converted[key.replace(".router.layer.weight", ".gate.weight")] = value
            continue
        if ".block_sparse_moe.experts.mlp." not in key:
            converted[key] = value
            continue
        which = key.rsplit(".", 1)[1]
        if which not in EXPERT_TARGETS:
            raise ValueError(
                f"{key}: packed expert tensor {which!r} has no declared target; "
                f"this module knows {sorted(EXPERT_TARGETS)}"
            )
        if value.shape[0] != expected_rows:
            raise ValueError(
                f"{key}: packed row count {value.shape[0]} is not num_experts * "
                f"intermediate_size = {num_experts} * {intermediate_size} = "
                f"{expected_rows}; the expert split would be silently wrong"
            )
        target = EXPERT_TARGETS[which]
        for expert in range(num_experts):
            block = value[expert * intermediate_size : (expert + 1) * intermediate_size, :]
            # w2 is the DOWN projection: packed as (ffn, hidden), wanted as
            # (hidden, ffn). The other two are (ffn, hidden) either way.
            payload = block.T.contiguous() if which == "w2" else block
            converted[
                key.replace(
                    f".experts.mlp.{which}", f".experts.{expert}.{target}.weight"
                )
            ] = payload
    return converted


def released_state_dict(
    checkpoint: Path | str = PROGEN3_CHECKPOINT,
) -> dict[str, torch.Tensor]:
    """The released tensors exactly as shipped, in megablocks packing.

    Exposed because the audit has to compare them against the backbone embedded
    in a replacement checkpoint, and the file layout of the release is this
    module's business rather than every caller's.
    """

    from safetensors.torch import load_file  # noqa: PLC0415

    return load_file(str(Path(checkpoint) / "model.safetensors"))


def strict_load(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load ``state`` into ``model``, refusing any missing or unexpected key.

    ``load_state_dict(strict=True)`` would also raise, and two things make this
    worth writing anyway. The refusal names the module's own failure mode rather
    than listing keys, because the whole point is that a *partial* load of this
    checkpoint is silent and plausible. And the key check happens **before**
    anything is assigned, so a refused load leaves the model untouched instead of
    leaving the caller holding the half-loaded object this module exists to
    prevent.
    """

    expected = set(model.state_dict())
    missing = sorted(expected - set(state))
    unexpected = sorted(set(state) - expected)
    if missing or unexpected:
        raise RuntimeError(
            "ProGen3 state dict does not match the model exactly, so part of the "
            "model would keep its random initialisation and the model would still "
            "run and still score. Refusing.\n"
            f"  missing ({len(missing)}): {missing}\n"
            f"  unexpected ({len(unexpected)}): {unexpected}"
        )
    model.load_state_dict(state, strict=True)


# ---------------------------------------------------------------------- loader


@dataclass(frozen=True)
class ProGen3:
    """A loaded ProGen3-112M and the handful of handles the audit needs."""

    model: Any
    config: Any
    preparer: Any
    device: torch.device
    checkpoint: Path

    @property
    def tokenizer(self) -> Any:
        return self.preparer.tokenizer

    @property
    def n_layers(self) -> int:
        return int(self.config.num_hidden_layers)

    @property
    def n_heads(self) -> int:
        return int(self.config.num_attention_heads)

    @property
    def moe_blocks(self) -> list[torch.nn.Module]:
        """The per-layer MoE blocks, in layer order.

        This is the module a replacement model replaces, so it is the one place
        the evaluation needs to reach into. See :func:`moe_intercept` for reading
        and substituting its input and output.
        """

        return [layer.block_sparse_moe for layer in self.model.model.layers]

    @property
    def attention_blocks(self) -> list[torch.nn.Module]:
        """The per-layer attention modules, in layer order.

        The counterpart of :attr:`moe_blocks`: one declaration of where a layer's
        attention lives, so :func:`ablated` addresses ``o_proj`` through it rather
        than by walking the module tree itself. The released config sets
        ``fused_attention_norm`` false, so the attention sits directly on the
        decoder layer; a checkpoint that set it true would put the attention under
        ``norm_attn_norm`` and this property would raise ``AttributeError`` rather
        than silently address the wrong module.
        """

        return [layer.self_attn for layer in self.model.model.layers]

    def batch(self, sequences: list[str], *, reverse: bool = False) -> dict[str, torch.Tensor]:
        """Model kwargs for a batch of raw residue strings, on this model's device."""

        return self.preparer.get_batch_kwargs(
            sequences, device=self.device, reverse=reverse
        )


def load_progen3(
    checkpoint: Path | str = PROGEN3_CHECKPOINT,
    *,
    device: str | torch.device = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    source: Path | str = PROGEN3_SOURCE,
) -> ProGen3:
    """Load ProGen3-112M with every expert and router actually loaded.

    Every step that could fail silently is made to fail loudly instead: the
    third-party package must be present at ``source``, the checkpoint must be
    present, the packed shapes must match the config, and the converted state
    dict must match the model exactly. What is left is checked numerically by
    :func:`self_check`.
    """

    checkpoint = Path(checkpoint)
    source = Path(source)
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"ProGen3 cannot run in {dtype}: its attention is pinned to the "
            f"flash SDPA backend, which has no kernel for it. Supported: "
            f"{list(SUPPORTED_DTYPES)}"
        )
    if not (source / "progen3").is_dir():
        raise FileNotFoundError(
            f"no progen3 package under {source}; set TRANSFER_PROGEN3_SRC to the "
            "src/ directory of the patched third-party copy"
        )
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(
            f"{checkpoint} holds no model.safetensors; set TRANSFER_PROGEN3_DIR"
        )
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

    from progen3.batch_preparer import ProGen3BatchPreparer  # noqa: PLC0415
    from progen3.config import ProGen3Config  # noqa: PLC0415
    from progen3.modeling import ProGen3ForCausalLM  # noqa: PLC0415
    from safetensors.torch import load_file  # noqa: PLC0415

    settings = json.loads((checkpoint / "config.json").read_text())
    settings["moe_implementation"] = "eager"
    config = ProGen3Config(**settings)
    converted = convert_megablocks_state_dict(
        load_file(str(checkpoint / "model.safetensors")),
        num_experts=config.num_experts,
        intermediate_size=config.intermediate_size,
    )
    model = ProGen3ForCausalLM(config)
    strict_load(model, converted)
    model = model.to(device=device, dtype=dtype).eval()
    return ProGen3(
        model=model,
        config=config,
        preparer=ProGen3BatchPreparer(),
        device=torch.device(device),
        checkpoint=checkpoint,
    )


# --------------------------------------------------------------------- scoring


def forward(pg: ProGen3, batch: dict[str, torch.Tensor]) -> Any:
    """One forward pass on a prepared batch: the single spelling of these kwargs.

    ``sequence_ids`` and ``position_ids`` are not optional on this model and are
    not what a generic decoder call passes, so every place that ran the model
    used to restate them -- scoring here, and two capture sweeps in
    ``15_replacement_faithfulness.py``. Three copies of one call is three places
    a keyword can be dropped, and a dropped ``sequence_ids`` is a plausible
    number rather than an error.
    """

    return pg.model(
        input_ids=batch["input_ids"],
        position_ids=batch["position_ids"],
        sequence_ids=batch["sequence_ids"],
        use_cache=False,
        return_dict=True,
    )


def scored_logits(
    pg: ProGen3, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(logits, targets, mask)`` aligned for next-token scoring, logits in fp32.

    The shift and the pad mask are declared once here because every downstream
    number -- NLL, KL, every ablation effect -- is taken through them, and an
    off-by-one on either produces a plausible number rather than an error.
    ``targets`` is every non-pad label after the first position, which is the
    convention the checkpoint's published figures were measured under.
    """

    output = forward(pg, batch)
    labels = batch["labels"]
    targets = labels[..., 1:]
    mask = (labels != pg.config.pad_token_id)[..., 1:]
    return output.logits[..., :-1, :].float(), targets, mask


def token_nll(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-position negative log-likelihood of ``targets`` under ``logits``."""

    flat = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
    )
    return flat.view(targets.shape)


@torch.no_grad()
def sequence_nll(
    pg: ProGen3,
    sequences: list[str],
    *,
    batch_size: int = 8,
    directions: tuple[bool, ...] = (False, True),
) -> torch.Tensor:
    """Per-sequence mean NLL, one row per (sequence, direction).

    ``directions`` holds the ``reverse`` flags to score. ProGen3 is trained in
    both N->C and C->N, and its published perplexities are the mean of the two;
    a stage that scores one direction is measuring something else and has to say
    so, which is why this is an argument and not a constant.
    """

    rows: list[torch.Tensor] = []
    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start : start + batch_size]
        for reverse in directions:
            batch = pg.batch(chunk, reverse=reverse)
            logits, targets, mask = scored_logits(pg, batch)
            nll = token_nll(logits, targets)
            rows.append(((nll * mask).sum(1) / mask.sum(1)).float().cpu())
    return torch.cat(rows)


def mean_nll(pg: ProGen3, sequences: list[str], **kwargs: Any) -> float:
    """Mean per-token NLL over :func:`sequence_nll`'s rows."""

    return float(sequence_nll(pg, sequences, **kwargs).mean())


# ------------------------------------------------------------------ self-check


def check_nll(
    value: float, *, band: tuple[float, float] = SELF_CHECK_NLL_BAND
) -> dict[str, Any]:
    """Refuse an NLL outside the declared sanity band.

    Separated from :func:`self_check` so that the band can be tested against the
    recorded corruption values without a checkpoint or a GPU.
    """

    low, high = band
    inside = low <= value <= high
    record = {
        "nll": float(value),
        "band": [float(low), float(high)],
        "n_sequences": len(SELF_CHECK_SEQUENCES),
        "reference": dict(MEASURED_SELF_CHECK_NLL),
        "verdict": "PASS" if inside else "FAIL",
    }
    if not inside:
        raise RuntimeError(
            f"ProGen3 self-check NLL {value:.4f} nats/token is outside the declared "
            f"band [{low}, {high}]. Reference values on the same eight sequences: "
            f"{MEASURED_SELF_CHECK_NLL}. Above the band the most likely cause is a "
            "wrong expert or router mapping, which load_state_dict(strict=True) "
            "cannot see; below it, a change to the scored-target convention. "
            "Either way the model must not be measured until this is resolved."
        )
    return record


def self_check(pg: ProGen3, *, batch_size: int = 8) -> dict[str, Any]:
    """Score the frozen sequences and refuse unless the NLL is in band."""

    return check_nll(mean_nll(pg, list(SELF_CHECK_SEQUENCES), batch_size=batch_size))


# ------------------------------------------------------- interventions and taps


#: Tokens that are not residues. Padding and the two terminus markers are what a
#: CLM batch actually contains; ``<mask>``, ``<bos_glm>``, ``<eos_span>`` and the
#: hundred span tokens belong to the infilling objective and never appear in one,
#: so the list is a subset of ProGen3's special tokens that coincides with it for
#: every batch this repository builds.
NON_RESIDUE_TOKENS = ("<pad>", "<bos>", "<eos>", "<mask>", "1", "2")


def content_mask(pg: ProGen3, input_ids: torch.Tensor) -> torch.Tensor:
    """Residue positions: everything but padding and the sequence markers.

    The single copy of this decision. Both the replacement-faithfulness stage
    and the transcoder trainer score reconstructions, and a reconstruction
    scored on padding is scored on the easiest positions in the batch -- so the
    two must not be allowed to disagree about which positions those are
    (Appendix B rule 12).
    """

    vocabulary = pg.tokenizer.get_vocab()
    absent = [name for name in NON_RESIDUE_TOKENS if name not in vocabulary]
    if absent:
        raise RuntimeError(
            f"the ProGen3 tokenizer carries no {absent}; the residue mask cannot "
            "be built and the reconstruction statistics would silently include "
            "non-residue positions"
        )
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    for name in NON_RESIDUE_TOKENS:
        mask &= input_ids != vocabulary[name]
    return mask


@contextmanager
def moe_intercept(
    pg: ProGen3, fn: Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor | None]
) -> Iterator[None]:
    """Read or replace every MoE block's output while the model runs.

    ``fn(layer, block_input, block_output)`` returns a tensor to substitute for
    the block's output, or ``None`` to leave it alone -- so the same primitive
    serves capture, transcoder replacement and mean ablation.

    Two facts about the block make a hand-written hook a trap, and they are the
    reason this is a function rather than a comment. The eager block returns
    ``(hidden_states, router_probabilities)``, so a hook that returns a bare
    tensor silently drops the router term; and a capture hook that returns the
    value it captured replaces the output with itself, which works until the
    captured tensor is detached or moved. Interceptors compose in entry order:
    the innermost ``with`` sees the output the outer one produced, which is what
    lets an ablation be applied on top of a replacement.
    """

    handles = []
    for layer, block in enumerate(pg.moe_blocks):

        def hook(
            module: torch.nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
            layer: int = layer,
        ) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            replacement = fn(layer, inputs[0], hidden)
            if replacement is None:
                return None
            if isinstance(output, tuple):
                return (replacement,) + tuple(output[1:])
            return replacement

        handles.append(block.register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def router_probabilities(pg: ProGen3, layer: int, block_input: torch.Tensor) -> torch.Tensor:
    """The router distribution a MoE block **selects on**, recomputed in float32.

    The block returns a distribution alongside its hidden states, and that one is
    not this one: it is ``softmax`` in the hidden states' dtype, while the
    ``topk`` that picks the experts runs on a float32 ``softmax`` of the same
    logits. EXP-R2-130 measured the two disagreeing on the selected pair for 3.3%
    of tokens. Anything that labels a token by which experts ran must therefore
    recompute rather than read, which is what this does -- through the block's own
    ``gate``, so there is one declaration of where the router lives.

    Returns ``(..., num_experts)`` on ``block_input``'s leading shape.
    """

    gate = pg.moe_blocks[layer].gate
    return torch.softmax(gate(block_input).float(), dim=-1)


def router_probabilities_agree(pg: ProGen3, layer: int, block_input: torch.Tensor, returned: torch.Tensor) -> dict[str, Any]:
    """Whether the recomputed router matches the one the block handed back.

    Not a duplicate of the computation but a check on the *addressing*: if
    :attr:`ProGen3.moe_blocks` or the gate's name ever moved, the recomputation
    would silently describe a different module, and every routing cell derived
    from it would be a label for something else. The two are expected to differ
    only by the dtype of their softmax, so the tolerance is a dtype tolerance and
    the selected-set disagreement is reported rather than asserted away.
    """

    recomputed = router_probabilities(pg, layer, block_input).reshape(-1, returned.shape[-1])
    reference = returned.float().reshape(-1, returned.shape[-1])
    top_k = int(pg.config.num_experts_per_tok)
    chosen = recomputed.topk(top_k, dim=-1).indices.sort(dim=-1).values
    reference_choice = reference.topk(top_k, dim=-1).indices.sort(dim=-1).values
    return {
        "max_absolute_difference": float((recomputed - reference).abs().max()),
        "selected_set_disagreement": float((chosen != reference_choice).any(dim=-1).float().mean()),
        "n_tokens": int(reference.shape[0]),
    }


@dataclass(frozen=True)
class Component:
    """One ablatable unit, named the same way in the original and a replacement.

    Only units that exist unchanged in both models qualify, which is what makes
    a cross-model comparison of their effects meaningful. Experts do not
    qualify: a transcoder replacement has no experts, so an expert ablation has
    no counterpart to be compared against.
    """

    kind: str
    layer: int
    index: int | None = None

    @property
    def label(self) -> str:
        if self.index is None:
            return f"{self.kind}.L{self.layer}"
        return f"{self.kind}.L{self.layer}H{self.index}"


def component_grid(n_layers: int, n_heads: int, *, block_kind: str) -> list[Component]:
    """Every attention head, then every replaceable block, in layer order.

    Architecture-neutral because the grid is: what differs between a MoE decoder
    and a dense one is the *name* of the block being replaced, not the shape of
    the set. Declared once here, beside :class:`Component`, so that the dense
    adapter in :mod:`src.transfer.replaceable` builds the same object rather than
    a parallel copy of it that could come to disagree about the ordering the
    saved effect matrices are indexed by.
    """

    heads = [
        Component("attention_head", layer, head)
        for layer in range(n_layers)
        for head in range(n_heads)
    ]
    blocks = [Component(block_kind, layer) for layer in range(n_layers)]
    return heads + blocks


def components(pg: ProGen3) -> list[Component]:
    """Every attention head, then every MoE block, in layer order."""

    return component_grid(pg.n_layers, pg.n_heads, block_kind="moe_block")


@contextmanager
def ablated(pg: ProGen3, component: Component) -> Iterator[None]:
    """Zero one component's contribution to the residual stream.

    An attention head is zeroed at the input of ``o_proj``, which is the only
    place its own slice still exists: ``_sdpa_attn`` reshapes ``(bsz, heads,
    len, head_dim)`` to ``(bsz, len, heads * head_dim)`` with heads contiguous,
    so head ``h`` owns columns ``h * head_dim`` to ``(h + 1) * head_dim``. A MoE
    block is zeroed at its output, which in a replacement model is the
    replacement's output -- the matched intervention.
    """

    if component.kind == "moe_block":
        target = component.layer

        def zero(layer: int, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor | None:
            return torch.zeros_like(y) if layer == target else None

        with moe_intercept(pg, zero):
            yield
        return

    if component.kind != "attention_head":
        raise ValueError(f"no ablation is implemented for component kind {component.kind!r}")

    head_dim = pg.config.hidden_size // pg.n_heads
    low = component.index * head_dim
    high = low + head_dim
    projection = pg.attention_blocks[component.layer].o_proj

    def pre(module: torch.nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        masked = inputs[0].clone()
        masked[..., low:high] = 0
        return (masked,) + tuple(inputs[1:])

    handle = projection.register_forward_pre_hook(pre)
    try:
        yield
    finally:
        handle.remove()
