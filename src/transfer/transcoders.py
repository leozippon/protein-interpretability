"""Cross-layer and per-layer transcoders, in plain torch.

**Why this exists.** ProGenMech's central claim is that a cross-layer transcoder
(CLT) beats a per-layer transcoder (PLT) at replacing ProGen3-112M's MoE blocks.
Their CLT weights are unobtainable -- the mirror serves the PLT directory and
returns HTTP 403 for the CLT one on the same token -- and their trainer cannot
run in an offline pod, which needs ``pytorch_lightning``, ``polars`` and
``wandb``. So the claim could be gated on their baseline and not on their
headline (EXP-R2-130).

This module removes that bound the only way left: **train both, ourselves, under
identical conditions.** That is a stronger comparison than their released weights
could have supported in any case. Their PLT was trained on their data with their
schedule; ours are trained on the same data, for the same steps, from the same
seed, differing in exactly the thing under test -- whether a latent may write to
layers downstream of the one it was read from.

**What is reproduced, and what is not.** The architecture and objective follow
their ``training/clt_model.py`` and ``training_transcoder/plt_model.py`` as read
from the released code: a hand-rolled layer norm whose statistics are kept for
de-normalisation, a bias subtracted *after* that norm and added back before it,
TopK on the pre-activations followed by ReLU on the selected values, a per-layer
NMSE objective with no L1 term, and an auxiliary loss that revives dead latents
through the self-layer decoder only. Three deviations are deliberate and are
recorded rather than smoothed:

1. **Decoder initialisation.** Their CLT re-runs ``kaiming_uniform_`` on the
   *encoder* weight inside the decoder-init loop, so each of the 55 decoders is
   a different random draw and the surviving encoder weight is only the last of
   them. That is a defect, not a design; we initialise each decoder directly and
   leave the encoders alone. Their PLT's init loop does not have it.
2. **Decoder normalisation.** Their ``norm_weights``/``norm_grad`` are defined
   and never called anywhere in their repository, so decoders are unconstrained
   during training. We match that -- unconstrained -- rather than adding a
   constraint they did not train under.
3. **Scale.** This is a bounded reproduction: a declared step budget, not their
   full 5M-sequence epoch. The comparison it supports is CLT against PLT at
   equal budget, not an absolute loss against their published number.

The estimand a downstream stage measures is the same either way: the transcoder
reads a MoE block's **input** (the output of ``post_attention_layernorm``) and
predicts that block's **output** before the residual add.

**Nothing in this module is specific to a MoE block, and that is now used.** The
objective, the normalisation and the splicing see a ``(layers, tokens, d_model)``
tensor pair and never ask what produced it, so the same code trains a transcoder
for a dense decoder's feed-forward -- ``GPT2Block`` reads ``ln_2``'s output and
adds the result to the residual, which is the same two tensors.
:mod:`src.transfer.replaceable` holds the adapter and verifies that identity on
the loaded model. What that buys is a text control and a dense protein control
for a replacement result that until now had neither.

**The released PLT lives here too**, in :class:`PerLayerTranscoder`. It was
written inside ``15_replacement_faithfulness.py`` and could not leave it: a
module whose name begins with a digit cannot be imported, so the second stage
that needed to load the same checkpoint would have had to carry its own copy of
the forward pass -- which is the shape of the rendering defect Appendix B rule 12
exists to stop. One declaration of what a transcoder is, trained or released.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .arms import REPO

#: The released ProGenMech per-layer transcoder. Under ``external_resources``,
#: which is git-ignored and CC BY-NC-ND: this repository reads its tensors and
#: imports none of its code.
#: Overridable, because the staged copy does not sit where the B-local one does:
#: the pod's tree carries no ``baselines/`` segment, so a hard-coded default
#: resolves to a path that does not exist there and a run dies after the process
#: is already up. ``h200_env.sh`` exports the pod-side location.
DEFAULT_REPLACEMENT = Path(
    os.environ.get(
        "TRANSFER_PROGENMECH_PLT",
        REPO
        / "external_resources/baselines/ProGenMechModels/ProGen3_PLT_L10_D4608"
        / "checkpoints/last.ckpt",
    )
)

#: Their dead-latent threshold, in the unit their config states it in. The model
#: divides by the batch size to reach a step count, so a trainer that hard-codes
#: the step count makes ``--batch-size`` silently change how long a latent may
#: stay silent. Declared here once and divided where the batch is known.
DEAD_STEPS_SEQUENCES = 10_000


@dataclass(frozen=True)
class TranscoderConfig:
    """Everything that defines a transcoder, recorded in every checkpoint."""

    num_layers: int = 10
    d_model: int = 384
    d_hidden: int = 4608
    k: int = 64
    #: Dead latents revived per layer per step. Their *scripts* declare 128 and
    #: their *models* never read it: ``clt_model.py`` and ``plt_model.py`` both
    #: use ``min(d_model // 2, n_dead)``, which is 192 here. The effective value
    #: is the one reproduced, because the declared one is dead code in their
    #: repository and reproducing it would be reproducing a file rather than a
    #: computation.
    auxk: int = 192
    #: Steps a latent may stay silent before the auxiliary loss revives it.
    #: Their ``dead_steps_threshold`` is in *sequences*; the quotient with the
    #: batch size is what the model uses, so the trainer derives this from the
    #: batch it actually runs (see :data:`DEAD_STEPS_SEQUENCES`) rather than
    #: letting a fixed step count silently mean different things at different
    #: batch sizes.
    dead_steps: int = 625
    aux_weight: float = 1.0 / 32.0
    cross_layer: bool = True
    #: Which model this transcoder was trained against, as ``--arm`` names it.
    #:
    #: ``None`` for the four ProGen3 checkpoints written before the trainer could
    #: reach any other model, and the faithfulness stage says so rather than
    #: assuming. It matters because the shape checks a replacement is spliced
    #: under -- depth and width -- do not separate the arms this now covers:
    #: gpt2-large and ProtGPT2 are both 36 layers of width 1280, so a transcoder
    #: trained on one would splice into the other without raising and produce a
    #: complete artefact for a replacement fitted to a different model.
    arm: str | None = None

    def n_parameters(self) -> int:
        """Trainable parameters, in closed form, so an arm can be sized before it runs.

        Encoders ``L*(d*h + h)``, decoders ``pairs*h*d``, ``b_pre`` ``L*d``. The
        CLT's ``pairs`` is ``L(L+1)/2`` against the PLT's ``L`` -- a 3.25x
        parameter advantage at ``L=10``, which is why a parameter-matched PLT is
        a control this comparison needs rather than a refinement it can skip.
        """

        pairs = (
            self.num_layers * (self.num_layers + 1) // 2 if self.cross_layer else self.num_layers
        )
        encoders = self.num_layers * (self.d_model * self.d_hidden + self.d_hidden)
        decoders = pairs * self.d_hidden * self.d_model
        return encoders + decoders + self.num_layers * self.d_model

    def record(self) -> dict[str, Any]:
        return {
            "architecture": "CLT" if self.cross_layer else "PLT",
            "arm": self.arm,
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "d_hidden": self.d_hidden,
            "k": self.k,
            "auxk": self.auxk,
            "dead_steps": self.dead_steps,
            "aux_weight": self.aux_weight,
            "n_parameters": self.n_parameters(),
            "active_latents_per_token": self.k,
            "active_fraction_of_dictionary": self.k / self.d_hidden,
        }


def normalise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Layer norm that hands back its statistics, so the output can be undone.

    ``std`` is torch's default (unbiased, n-1) and the epsilon sits *outside* the
    root and *inside* the divisor, matching the released implementation. The
    de-normalisation divides by ``std`` alone rather than ``std + eps``, which is
    an asymmetry in their code and is reproduced because the trained weights
    depend on it.
    """

    mean = x.mean(dim=-1, keepdim=True)
    centred = x - mean
    std = centred.std(dim=-1, keepdim=True)
    return centred / (std + 1e-5), mean, std


def topk_relu(pre: torch.Tensor, k: int) -> torch.Tensor:
    """TopK on the pre-activations, then ReLU on the values that survived.

    In that order, which is theirs and is not the same as ReLU-then-TopK: a
    selected pre-activation that is negative becomes exactly zero, so a latent
    can be in the top-k and still contribute nothing.
    """

    values, indices = torch.topk(pre, k=k, dim=-1)
    out = torch.zeros_like(pre)
    return out.scatter_(-1, indices, F.relu(values))


class Transcoder(nn.Module):
    """A CLT or a PLT; the only difference is which decoders exist.

    One class rather than two because the objective, the normalisation, the
    activation, the dead-latent bookkeeping and the auxiliary loss are identical
    -- and because the comparison this module exists for is only meaningful if
    everything except the decoder connectivity is literally the same code.
    """

    def __init__(self, config: TranscoderConfig):
        super().__init__()
        self.config = config
        L, d, h = config.num_layers, config.d_model, config.d_hidden

        self.encoders = nn.ModuleList([nn.Linear(d, h) for _ in range(L)])
        self.b_pre = nn.Parameter(torch.zeros(L, d))
        pairs = (
            [(s, t) for s in range(L) for t in range(s, L)]
            if config.cross_layer
            else [(layer, layer) for layer in range(L)]
        )
        self.pairs = pairs
        self.decoders = nn.ParameterDict(
            {
                f"{s}_{t}": nn.Parameter(torch.empty(h, d))
                for s, t in pairs
            }
        )
        for parameter in self.decoders.values():
            nn.init.kaiming_uniform_(parameter, a=5 ** 0.5)
        # Steps since each latent last fired, for the auxiliary revival loss.
        self.register_buffer("silent_steps", torch.zeros(L, h, dtype=torch.long))

    @property
    def writes_to(self) -> dict[int, list[int]]:
        """Target layers each source layer may write to, for the record."""

        out: dict[int, list[int]] = {}
        for source, target in self.pairs:
            out.setdefault(source, []).append(target)
        return out

    def encode_layer(
        self, layer: int, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One layer's pre-activations and normalisation statistics.

        The single place a layer is encoded. :meth:`encode` calls it for the
        batched training path and :class:`TranscoderReplacement` calls it one
        block at a time inside a live forward pass, so a replacement cannot come
        to disagree with the objective it was trained against.
        """

        hat, mean, std = normalise(inputs)
        return self.encoders[layer](hat - self.b_pre[layer]), mean, std

    def decode_target(
        self,
        target: int,
        latents: dict[int, torch.Tensor],
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        """One target layer's reconstruction, from whichever sources have fired.

        ``latents`` is keyed by source layer. Every pair reaching ``target`` must
        be present: a CLT source that has not been encoded yet would silently
        contribute nothing and the reconstruction would be a different, better-
        looking object than the one that was trained. Since every pair runs
        ``source <= target`` and MoE blocks execute in layer order, a live
        forward pass always has them -- so a missing one is a defect, and it
        raises.
        """

        total = None
        for source, other in self.pairs:
            if other != target:
                continue
            if source not in latents:
                raise KeyError(
                    f"decoding layer {target} needs source layer {source}, which has "
                    "not been encoded; a cross-layer reconstruction missing a source "
                    "is not the model that was trained"
                )
            contribution = latents[source] @ self.decoders[f"{source}_{target}"]
            total = contribution if total is None else total + contribution
        assert total is not None
        return (total + self.b_pre[target]) * std + mean

    def encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``inputs`` is ``(L, tokens, d_model)``; returns latents and statistics."""

        means, stds = [], []
        pre_activations = []
        for layer in range(self.config.num_layers):
            pre_layer, mean, std = self.encode_layer(layer, inputs[layer])
            pre_activations.append(pre_layer)
            means.append(mean)
            stds.append(std)
        pre = torch.stack(pre_activations)
        latents = topk_relu(pre, self.config.k)
        return latents, pre, torch.stack(means), torch.stack(stds)

    def decode(
        self, latents: torch.Tensor, means: torch.Tensor, stds: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct every layer's MoE output from the latents that may reach it."""

        by_source = {layer: latents[layer] for layer in range(self.config.num_layers)}
        return torch.stack(
            [
                self.decode_target(target, by_source, means[target], stds[target])
                for target in range(self.config.num_layers)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        latents, _, means, stds = self.encode(inputs)
        return self.decode(latents, means, stds)

    # ------------------------------------------------------------------ losses

    def objective(
        self, inputs: torch.Tensor, targets: torch.Tensor, *, training: bool
    ) -> dict[str, Any]:
        """Per-layer NMSE, plus the dead-latent auxiliary term.

        There is no sparsity penalty: sparsity is TopK's, entirely. A reader
        expecting an L1 term should not go looking for one.
        """

        latents, pre, means, stds = self.encode(inputs)
        reconstruction = self.decode(latents, means, stds)

        variance = targets.var(dim=(1, 2), unbiased=False)
        per_layer = ((reconstruction - targets) ** 2).mean(dim=(1, 2)) / (variance + 1e-8)
        nmse = per_layer.sum()

        aux = torch.zeros((), device=inputs.device, dtype=nmse.dtype)
        n_dead = 0
        if training:
            fired = (latents > 0).any(dim=1)
            self.silent_steps = torch.where(
                fired, torch.zeros_like(self.silent_steps), self.silent_steps + 1
            )
            dead = self.silent_steps > self.config.dead_steps
            n_dead = int(dead.sum())
            if n_dead:
                residual = (targets - reconstruction).detach()
                for layer in range(self.config.num_layers):
                    layer_dead = dead[layer]
                    if not bool(layer_dead.any()):
                        continue
                    k_aux = int(min(self.config.auxk, int(layer_dead.sum())))
                    masked = pre[layer].masked_fill(~layer_dead, float("-inf"))
                    revived = topk_relu(masked, k_aux)
                    # Self-layer decoder only: a dead latent is revived where it
                    # was read, not by borrowing another layer's output.
                    #
                    # The bias and the de-normalisation are not decoration. The
                    # target is a residual in raw activation space; a decoder
                    # output is in the normalised space the encoder read from, and
                    # the two differ by this layer's own scale. Comparing them
                    # directly -- which this did until the defect was measured --
                    # makes the auxiliary gradient 1.8x to 19.3x too large and
                    # points it in the wrong direction (cosine 0.70-0.94 against
                    # the correct gradient in the first six layers), worst where
                    # the activation scale is smallest. It is also asymmetric
                    # across the comparison this module exists for: this decoder
                    # is the PLT's only contributor to its layer and one of up to
                    # ten in the CLT. Their ``clt_model.py`` and ``plt_model.py``
                    # both de-normalise here, and so does this now.
                    predicted = (
                        revived @ self.decoders[f"{layer}_{layer}"] + self.b_pre[layer]
                    ) * stds[layer] + means[layer]
                    aux = aux + F.mse_loss(predicted, residual[layer])

        return {
            "loss": nmse + self.config.aux_weight * aux,
            "nmse_sum": nmse,
            "nmse_per_layer": per_layer.detach(),
            "aux": aux.detach(),
            "n_dead": n_dead,
            "active_fraction": float((latents > 0).float().mean()),
        }


@dataclass
class TrainingRecord:
    """What a run reports about itself, beyond its weights."""

    steps: int = 0
    tokens: int = 0
    sequences: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    def record(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "tokens": self.tokens,
            "sequences": self.sequences,
            "history": self.history,
        }


# ------------------------------------------------- matching two training runs


#: What two dictionaries must agree on before a difference between their
#: behavioural numbers may be attributed to what they were trained on.
#:
#: **The list is the experiment.** Two per-layer transcoders on the *same*
#: ProLLaMA weights -- one on its text mode, one on its protein mode -- can only
#: separate modality from everything else if nothing else moves: the same layers,
#: the same dictionary width, the same active fraction, the same amount of data
#: and the same held-out budget. Every one of those can differ without anything
#: raising, and a width or token difference between the two arms would read as
#: modality in exactly the way L25 records a capacity difference reading as
#: cross-layer connectivity.
#:
#: ``backbone_sha256`` is in the list and is the field the *name* cannot supply.
#: A joint checkpoint is reached by path, so ``prollama:protein`` names a mode
#: and not a checkpoint: ``Llama-2-7b-hf``, ``ProLLaMA_Stage_1`` and ``ProLLaMA``
#: all answer to it, and a pair drawn from two of them would be a comparison
#: across training stages wearing the label of a comparison within one set of
#: weights.
#:
#: ``training_token_budget`` and not the realised count. The trainer stops at the
#: first step that reaches the budget, so the realised totals differ by up to one
#: batch; the budget is the declared quantity and is what must be equal, while
#: the realised counts are recorded beside it and reported as a relative
#: difference.
#:
#: **The criterion, so that a later field is decided rather than guessed.** A
#: setting belongs here when it changes the fitted dictionary, nothing about a
#: mode requires it to differ, and a run can set it. That admits the whole
#: optimisation block -- learning rate, weight decay, gradient clip, batch size,
#: the auxiliary-revival width, and both seeds -- and it excludes ``steps``, which
#: a budget-matched pair *must* be free to differ on because a text record and a
#: protein record carry different numbers of scored positions, and ``eval_every``,
#: which is a measurement cadence and does not touch the fitted object.
#:
#: The optimisation block was absent until two artefacts a hundredfold apart in
#: learning rate, at different seeds and at batch 8 against 64, registered
#: ``MATCHED`` with no disagreements. The failure was one level below a short
#: list: :class:`MatchedTraining` carried no such field at all, so widening this
#: tuple alone would have named fields that did not exist.
#:
#: ``corpus_seed`` is matched even though the two modes read *different* corpora
#: and the same seed therefore selects different records. What it equalises is the
#: draw procedure -- the block-shuffled stream and the held-out offset -- and two
#: modes drawn under different procedures differ on an axis nobody chose.
#:
#: ``max_tokens`` is deliberately **not** here and is reported instead; see
#: :attr:`MatchedTraining.max_tokens`.
MATCHED_TRAINING_FIELDS = (
    "backbone_sha256",
    "architecture",
    "num_layers",
    "d_model",
    "d_hidden",
    "k",
    "auxk",
    "training_token_budget",
    "evaluation_sequences",
    "learning_rate",
    "weight_decay",
    "grad_clip",
    "batch_size",
    "seed",
    "corpus_seed",
)


@dataclass(frozen=True)
class MatchedTraining:
    """One training run's declaration of everything a matched pair must share.

    Written by ``17_train_transcoder.py`` into both its JSON record and its
    checkpoint, and read back by ``15_replacement_faithfulness.py``, so the
    faithfulness artefact states the training configuration of the dictionary it
    scored rather than leaving a reader to find the training artefact.

    ``target`` is the one field that must **differ** across the pair -- it is the
    arm, or ``rendering:mode`` for a joint checkpoint -- and it is recorded here
    so that :func:`compare_matched_training` can say whether it was handed two
    conditions or the same one twice.
    """

    target: str
    backbone_sha256: str
    architecture: str
    num_layers: int
    d_model: int
    d_hidden: int
    k: int
    auxk: int
    #: ``None`` for a run whose length was set in steps rather than in tokens.
    #: Not zero: a zero budget and an undeclared one are different states, and
    #: only the second may be compared as if it were a number.
    training_token_budget: int | None
    #: What the run actually consumed, in scored tokens.
    training_tokens: int
    evaluation_sequences: int
    learning_rate: float
    weight_decay: float
    grad_clip: float
    batch_size: int
    seed: int
    corpus_seed: int
    #: The token cap each dictionary's inputs were rendered under. **Reported,
    #: not matched, and the distinction is the point.** It changes what a
    #: dictionary saw -- an activation at position 900 is not an activation at
    #: position 400 -- so it cannot be left out of the record; but it is the one
    #: setting a mode genuinely constrains, because a protein rendering pays a
    #: measured wrapper and ``17_train_transcoder.py`` derives that mode's residue
    #: ceiling from this cap. Forcing the two equal would either truncate the
    #: protein corpus below its band or move the text window for no reason.
    #:
    #: So it is declared here, compared in
    #: :func:`compare_matched_training`, and reported beside the verdict rather
    #: than folded into it. The live ProLLaMA pilot runs 512 for text against 1024
    #: for protein: that is a real unmatched axis, and a reader of its ``MATCHED``
    #: verdict has to be able to see it without opening the settings block.
    max_tokens: int

    def matched(self) -> dict[str, Any]:
        """Exactly the fields a pair is refused on, in the declared order."""

        values = asdict(self)
        return {name: values[name] for name in MATCHED_TRAINING_FIELDS}

    def digest(self) -> str:
        """SHA-256 over the matched fields, so a mismatch is visible in one line.

        Over :data:`MATCHED_TRAINING_FIELDS` alone and not over the whole record:
        a digest that moved with the realised token count or with the target name
        would differ between the two modes by construction and would therefore
        certify nothing.
        """

        return hashlib.sha256(
            json.dumps(self.matched(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def record(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "matched_fields": list(MATCHED_TRAINING_FIELDS),
            "digest": self.digest(),
            "note": (
                "the fields two dictionaries must agree on before a difference "
                "between their behavioural numbers may be attributed to what they "
                "were trained on. 'target' is the field that must DIFFER. Three "
                "fields are recorded and NOT matched: 'training_tokens', what the "
                "run consumed, because the loop stops at the first step to reach "
                "'training_token_budget' and so overshoots it by up to one batch; "
                "'max_tokens', the token cap the inputs were rendered under, "
                "because a protein rendering pays a wrapper this stage measures "
                "and derives its residue ceiling from, so a mode may genuinely "
                "need a different cap -- it is reported with its own agreement "
                "flag in the comparison instead; and 'target' itself"
            ),
        }


#: The declaration's optimisation fields, and the name each one carries in the
#: ``17_train_transcoder.py`` argument block they are projected from. Identical
#: names, because ``argparse`` produces them -- listed rather than assumed so that
#: :func:`matched_training` walks one sequence instead of repeating the same
#: two-source lookup eight times.
_OPTIMISATION_FIELDS = (
    "auxk",
    "learning_rate",
    "weight_decay",
    "grad_clip",
    "batch_size",
    "seed",
    "corpus_seed",
    "max_tokens",
)


def matched_training(
    record: Mapping[str, Any], *, settings: Mapping[str, Any] | None = None
) -> MatchedTraining:
    """Read a matched-training declaration back from an artefact or a checkpoint.

    ``settings`` is the same run's ``17_train_transcoder.py`` argument block, and
    it is where the optimisation fields above are read from when the declaration
    itself does not carry them. Dictionaries were written before the declaration
    was widened to cover the optimiser, and every one of those runs recorded the
    values in its own settings block -- so reading them there recovers the run's
    configuration rather than assuming one, and does not force a campaign to
    retrain in order to be certified on fields it did in fact hold fixed. The
    settings block is the source and this declaration is its projection, so the
    two cannot disagree.

    A record carrying neither raises. A pair certified on fields nobody wrote down
    is the state this widening exists to end.
    """

    supplied = dict(settings or {})
    values = {
        name: record.get(name, supplied.get(name)) for name in _OPTIMISATION_FIELDS
    }
    missing = sorted(
        name
        for name in (
            "target",
            "backbone_sha256",
            "architecture",
            "num_layers",
            "d_model",
            "d_hidden",
            "k",
            "training_tokens",
            "evaluation_sequences",
        )
        if name not in record
    ) + sorted(name for name, value in values.items() if value is None)
    if missing:
        raise KeyError(
            f"this matched-training record is missing {missing}; it was not written "
            "by 17_train_transcoder.py, so the configuration a comparison would be "
            "refused on is unknown"
        )
    budget = record.get("training_token_budget")
    return MatchedTraining(
        target=str(record["target"]),
        backbone_sha256=str(record["backbone_sha256"]),
        architecture=str(record["architecture"]),
        num_layers=int(record["num_layers"]),
        d_model=int(record["d_model"]),
        d_hidden=int(record["d_hidden"]),
        k=int(record["k"]),
        auxk=int(values["auxk"]),
        training_token_budget=None if budget is None else int(budget),
        training_tokens=int(record["training_tokens"]),
        evaluation_sequences=int(record["evaluation_sequences"]),
        learning_rate=float(values["learning_rate"]),
        weight_decay=float(values["weight_decay"]),
        grad_clip=float(values["grad_clip"]),
        batch_size=int(values["batch_size"]),
        seed=int(values["seed"]),
        corpus_seed=int(values["corpus_seed"]),
        max_tokens=int(values["max_tokens"]),
    )


def compare_matched_training(
    left: MatchedTraining, right: MatchedTraining
) -> dict[str, Any]:
    """Whether two dictionaries may be read against each other, field by field.

    Three verdicts, and the middle one is the point of having three. ``MATCHED``
    means every field of :data:`MATCHED_TRAINING_FIELDS` agrees. ``MISMATCH``
    names the ones that do not. ``UNMATCHED_BUDGET`` is the case where the two
    runs agree on everything they declared and at least one of them declared no
    token budget at all -- their step counts may be equal while their token
    counts are not, because a text record and a protein record carry different
    numbers of scored positions, and "the same number of steps" is then a
    matched *schedule* over two different amounts of data. It is reported
    separately rather than folded into either of the other two, because it is a
    statement about what was declared and not about what disagreed.

    The realised token counts are reported with their relative difference
    whatever the verdict, so a pair that agrees on a budget and diverged in
    practice is visible rather than certified by the digest alone. The token cap
    is reported the same way and for a stronger reason: it is the one setting a
    mode may legitimately need to differ on, so ``MATCHED`` does not cover it, and
    a reader who is not shown it would take the verdict for more than it claims.
    """

    fields = {
        name: {"values": [values[0], values[1]], "agree": values[0] == values[1]}
        for name, values in (
            (name, (left.matched()[name], right.matched()[name]))
            for name in MATCHED_TRAINING_FIELDS
        )
    }
    disagreements = sorted(name for name, entry in fields.items() if not entry["agree"])
    budget_declared = (
        left.training_token_budget is not None and right.training_token_budget is not None
    )
    realised = [left.training_tokens, right.training_tokens]
    largest = max(abs(value) for value in realised) or 1
    if disagreements:
        verdict = "MISMATCH"
    elif not budget_declared:
        verdict = "UNMATCHED_BUDGET"
    else:
        verdict = "MATCHED"
    return {
        "targets": [left.target, right.target],
        "distinct_targets": left.target != right.target,
        "digests": [left.digest(), right.digest()],
        "digests_agree": left.digest() == right.digest(),
        "matched_fields": list(MATCHED_TRAINING_FIELDS),
        "fields": fields,
        "disagreements": disagreements,
        "training_token_budget_declared": budget_declared,
        "training_tokens_realised": realised,
        "training_tokens_relative_difference": abs(realised[0] - realised[1]) / largest,
        "max_tokens": [left.max_tokens, right.max_tokens],
        "max_tokens_agree": left.max_tokens == right.max_tokens,
        "verdict": verdict,
        "note": (
            "MATCHED means every field two dictionaries must share agrees. "
            "UNMATCHED_BUDGET means they agree on what they declared and at least "
            "one declared no --train-tokens, so equal step counts do not imply "
            "equal data: a text record and a protein record carry different "
            "numbers of scored positions. distinct_targets says whether this is a "
            "pair of conditions at all rather than one condition twice. "
            "max_tokens is REPORTED and not matched: a protein rendering pays a "
            "measured wrapper and derives its residue ceiling from this cap, so a "
            "mode may need a different one -- but the two dictionaries then saw "
            "different context lengths, which no verdict here certifies, and "
            "max_tokens_agree is what says so"
        ),
    }


# ------------------------------------------------------------- the free baseline


class LinearReplacement:
    """The replacement that needs no dictionary: one affine map per block.

    **Why this exists.** Standing rule 28 says a method is scored against the
    trivial baseline available from its own coordinates, and it has never been
    applied to the replacement stage -- which scored exactly three conditions,
    the original, the transcoder, and the mean-ablated floor that is the
    *denominator*. Nothing stood between the floor and the method. Twice before,
    applying this rule changed what a result meant: a selector that knows only a
    head's layer index beat a census (EXP-R2-131), and a BLOSUM62 lookup was
    not separable from a model's zero-shot fitness on the panel its recovery
    ratios were quoted against (EXP-R2-133/134).

    A per-layer least-squares map ``y ~= (x - mu_x) W + mu_y`` carries
    ``d_model^2 + d_model`` parameters -- **147,840** against a cross-layer
    transcoder's 115,065,600, a factor of 778 -- has no sparsity, no latents, no
    training loop and no hyper-parameters beyond a ridge term. It is what a MoE
    block would be if it were a linear function of its input.

    If this recovers what the transcoders recover, then nothing measured on this
    estimand is evidence that a sparse dictionary captured anything: the
    quantity would be a property of how linear the block is, not of the method.
    That is the outcome this class exists to be able to report.
    """

    def __init__(self, weight: torch.Tensor, x_mean: torch.Tensor, y_mean: torch.Tensor) -> None:
        self.weight = weight  # (layers, d_model, d_model)
        self.x_mean = x_mean  # (layers, d_model)
        self.y_mean = y_mean
        self.num_layers = int(weight.shape[0])

    @property
    def n_parameters(self) -> int:
        layers, d_model, _ = self.weight.shape
        return layers * (d_model * d_model + d_model)

    def to(self, device: torch.device | str) -> LinearReplacement:
        self.weight = self.weight.to(device=device, dtype=torch.float32)
        self.x_mean = self.x_mean.to(device=device, dtype=torch.float32)
        self.y_mean = self.y_mean.to(device=device, dtype=torch.float32)
        return self

    def reset(self) -> None:
        """No cross-token or cross-layer state. Present so callers need not care."""

    @torch.no_grad()
    def __call__(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).float()
        out = (flat - self.x_mean[layer]) @ self.weight[layer] + self.y_mean[layer]
        return out.reshape(shape).to(x.dtype)

    def record(self) -> dict[str, Any]:
        return {
            "architecture": "LINEAR",
            "num_layers": self.num_layers,
            "d_model": int(self.weight.shape[1]),
            "n_parameters": self.n_parameters,
            "note": "per-layer affine least-squares map from block input to block "
            "output; the free baseline standing rule 28 requires, carrying no "
            "dictionary, no sparsity and no trained latents",
        }


class LinearReplacementFitter:
    """Accumulates the Gram matrices one affine map per layer is solved from.

    Streaming rather than storing activations: the normal equations need only
    ``X^T X`` and ``X^T Y`` per layer, which are ``d_model`` square regardless of
    how many tokens are seen. Accumulated in float64 because ``X^T X`` on
    activations whose scale spans two orders of magnitude across layers loses
    conditioning in float32 long before it loses it in float64.
    """

    def __init__(self, num_layers: int, d_model: int) -> None:
        self.num_layers, self.d_model = num_layers, d_model
        self.xtx = torch.zeros(num_layers, d_model, d_model, dtype=torch.float64)
        self.xty = torch.zeros(num_layers, d_model, d_model, dtype=torch.float64)
        self.x_sum = torch.zeros(num_layers, d_model, dtype=torch.float64)
        self.y_sum = torch.zeros(num_layers, d_model, dtype=torch.float64)
        self.count = torch.zeros(num_layers, dtype=torch.float64)

    def update(self, layer: int, x: torch.Tensor, y: torch.Tensor) -> None:
        """``x`` and ``y`` are ``(tokens, d_model)``, already masked to scored positions."""

        xd, yd = x.double(), y.double()
        self.xtx[layer] += (xd.T @ xd).cpu()
        self.xty[layer] += (xd.T @ yd).cpu()
        self.x_sum[layer] += xd.sum(0).cpu()
        self.y_sum[layer] += yd.sum(0).cpu()
        self.count[layer] += float(x.shape[0])

    def solve(self, *, ridge: float = 1e-6) -> LinearReplacement:
        """Centred normal equations, one layer at a time.

        The ridge is relative to each layer's own scale (a fraction of
        ``trace(X^T X)/d``), not absolute, because block-input scale varies by a
        factor of seven across ProGen3's depth and one absolute constant would be
        a different regulariser at every layer.
        """

        if float(self.count.min()) <= self.d_model:
            raise ValueError(
                f"a layer saw {float(self.count.min()):.0f} tokens for {self.d_model} "
                "features; the normal equations are underdetermined and the fit "
                "would be memorising the cohort rather than estimating a map"
            )
        weights, x_means, y_means = [], [], []
        for layer in range(self.num_layers):
            n = self.count[layer]
            mu_x = self.x_sum[layer] / n
            mu_y = self.y_sum[layer] / n
            centred_xtx = self.xtx[layer] - n * torch.outer(mu_x, mu_x)
            centred_xty = self.xty[layer] - n * torch.outer(mu_x, mu_y)
            scale = float(torch.diagonal(centred_xtx).mean())
            regularised = centred_xtx + ridge * scale * torch.eye(self.d_model, dtype=torch.float64)
            weights.append(torch.linalg.solve(regularised, centred_xty))
            x_means.append(mu_x)
            y_means.append(mu_y)
        return LinearReplacement(
            torch.stack(weights).float(),
            torch.stack(x_means).float(),
            torch.stack(y_means).float(),
        )


# -------------------------------------------------- a trained transcoder, spliced


class TranscoderReplacement:
    """A locally trained CLT or PLT, callable one MoE block at a time.

    The faithfulness stage substitutes a block's output through
    ``moe_intercept``, which sees one layer at a time and in layer order. A PLT
    fits that shape directly. A **CLT does not**, because layer ``t``'s
    reconstruction needs the latents of every source ``s <= t`` -- which is why
    the trained checkpoints had no path through that stage at all, and why the
    only thing the first four runs could deliver was reconstruction NMSE, the one
    quantity EXP-R2-132 established is not sufficient.

    The resolution is that ``s <= t`` for every pair by construction, so a
    forward pass has already encoded every source it needs by the time a target
    fires. This accumulates them and :meth:`decode_target` refuses if one is
    missing rather than quietly reconstructing from a subset.

    **The accumulator clears itself when layer 0 fires**, because that is what
    beginning a forward pass *is* on this model: MoE blocks execute in layer
    order and every pass starts at layer 0. That is the same fact the
    accumulation already depends on for correctness, so using it here adds no
    new assumption -- whereas requiring every caller to remember an explicit
    reset would add a way to be silently wrong. Stale latents from a previous
    batch usually differ in token count and would raise on the matmul, but two
    batches of equal length would not, and the reconstruction would quietly mix
    them. :meth:`reset` remains available for a caller that wants to be explicit.
    """

    def __init__(self, model: Transcoder) -> None:
        self.model = model.eval()
        self.config = model.config
        self._latents: dict[int, torch.Tensor] = {}

    @property
    def num_layers(self) -> int:
        return self.config.num_layers

    def to(self, device: torch.device | str) -> TranscoderReplacement:
        self.model = self.model.to(device=device, dtype=torch.float32)
        return self

    def reset(self) -> None:
        self._latents.clear()

    @torch.no_grad()
    def __call__(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        if layer == 0:
            self._latents.clear()
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).float()
        pre, mean, std = self.model.encode_layer(layer, flat)
        self._latents[layer] = topk_relu(pre, self.config.k)
        reconstruction = self.model.decode_target(layer, self._latents, mean, std)
        return reconstruction.reshape(shape).to(x.dtype)


#: The key a matched-training declaration is stored under, in a checkpoint and in
#: a training artefact alike. One spelling, because two consumers read it.
MATCHED_TRAINING_KEY = "matched_training"


def matched_training_from_artefact(path: Path) -> MatchedTraining:
    """The matched-training declaration of a ``17_train_transcoder.py`` JSON record.

    The JSON and not the checkpoint, because this reads the *other* arm of a
    comparison and the other arm's weights are of no interest: a matched pair's
    dictionaries run to gigabytes each, and loading one to read a dozen numbers
    would make the check cost more than the run it guards.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    declaration = payload.get(MATCHED_TRAINING_KEY)
    if declaration is None:
        raise KeyError(
            f"{path} carries no {MATCHED_TRAINING_KEY!r} block, so it is not a "
            "training record from 17_train_transcoder.py -- or it predates the "
            "matched-configuration declaration and the pair cannot be checked"
        )
    return matched_training(declaration, settings=payload.get("settings"))


def load_trained_transcoder(
    path: Path,
) -> tuple[TranscoderReplacement, dict[str, Any], MatchedTraining | None]:
    """Read a checkpoint written by ``17_train_transcoder.py``.

    Separate from :func:`load_replacement` because they are different objects:
    that one reads a third-party Lightning checkpoint carrying an embedded
    backbone, this one reads ours, which carries none. Conflating them behind one
    function would mean guessing which is on disk, and a wrong guess produces a
    shape error a long way from its cause.

    The third element is the run's :class:`MatchedTraining` declaration, or
    ``None`` for a checkpoint written before there was one. It travels with the
    weights rather than beside them so that a dictionary handed to the
    faithfulness stage carries the configuration it must be matched on, whatever
    happened to its training artefact.
    """

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    missing = sorted({"config", "state_dict"} - set(checkpoint))
    if missing:
        raise RuntimeError(
            f"{path} is not a checkpoint from 17_train_transcoder.py (missing "
            f"{missing}). A released ProGenMech checkpoint is read by "
            "load_replacement instead."
        )
    recorded = checkpoint["config"]
    declared = checkpoint.get(MATCHED_TRAINING_KEY)
    config = TranscoderConfig(
        num_layers=int(recorded["num_layers"]),
        d_model=int(recorded["d_model"]),
        d_hidden=int(recorded["d_hidden"]),
        k=int(recorded["k"]),
        auxk=int(recorded["auxk"]),
        dead_steps=int(recorded["dead_steps"]),
        aux_weight=float(recorded["aux_weight"]),
        cross_layer=recorded["architecture"] == "CLT",
        # Absent from the four checkpoints written before the trainer could reach
        # a second model. Read back as None rather than defaulted to ProGen3, so
        # that the consumer can say "this checkpoint does not declare its arm"
        # instead of being told an arm nobody recorded.
        arm=recorded.get("arm"),
    )
    model = Transcoder(config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return (
        TranscoderReplacement(model),
        recorded,
        None
        if declared is None
        else matched_training(
            declared, settings=(checkpoint.get("record") or {}).get("settings")
        ),
    )


# ------------------------------------------------------- the released transcoder


class PerLayerTranscoder:
    """ProGenMech's released per-layer transcoder, re-implemented from its weights.

    Their class lives in a CC BY-NC-ND tree and imports ``pytorch_lightning``,
    which is absent here and not installable in the offline pods. The weights
    themselves need neither: ``torch.load(weights_only=True)`` reads them once
    ``argparse.Namespace`` is allowlisted. So the forward pass is restated here,
    line for line against ``training_transcoder/plt_model.py``:

        centre and scale over the feature axis, subtract ``b_pre``, encode,
        keep the top ``k`` pre-activations under ReLU, decode, add ``b_pre``,
        undo the scaling.

    The scale is ``Tensor.std``'s **unbiased** estimate and the epsilon is added
    to it rather than inside the square root, because that is what their code
    does and this is a re-implementation, not an improvement. Each layer is
    independent, so a per-layer hook reproduces it exactly.
    """

    def __init__(self, state: dict[str, torch.Tensor], hyperparameters: Any) -> None:
        self.num_layers = int(hyperparameters.num_layers)
        self.d_model = int(hyperparameters.d_model)
        self.d_hidden = int(hyperparameters.d_hidden)
        self.k = int(hyperparameters.k)
        expected = {"stats_last_nonzero"}
        for layer in range(self.num_layers):
            expected |= {
                f"encoders.{layer}.weight",
                f"encoders.{layer}.bias",
                f"decoders.{layer}",
                f"b_pre.{layer}",
            }
        missing = sorted(expected - set(state))
        unexpected = sorted(set(state) - expected)
        if missing or unexpected:
            raise RuntimeError(
                "released transcoder state dict does not match the declared "
                f"architecture (L={self.num_layers}, d_model={self.d_model}, "
                f"d_hidden={self.d_hidden}, k={self.k}).\n"
                f"  missing ({len(missing)}): {missing}\n"
                f"  unexpected ({len(unexpected)}): {unexpected}"
            )
        self.encoder_weight = [state[f"encoders.{i}.weight"] for i in range(self.num_layers)]
        self.encoder_bias = [state[f"encoders.{i}.bias"] for i in range(self.num_layers)]
        self.decoder = [state[f"decoders.{i}"] for i in range(self.num_layers)]
        self.b_pre = [state[f"b_pre.{i}"] for i in range(self.num_layers)]
        for name, tensors, shape in (
            ("encoders", self.encoder_weight, (self.d_hidden, self.d_model)),
            ("decoders", self.decoder, (self.d_hidden, self.d_model)),
            ("b_pre", self.b_pre, (self.d_model,)),
        ):
            wrong = [i for i, t in enumerate(tensors) if tuple(t.shape) != shape]
            if wrong:
                raise RuntimeError(
                    f"released transcoder {name} at layers {wrong} do not have "
                    f"shape {shape}; the hyper-parameters and the weights disagree"
                )

    def to(self, device: torch.device) -> PerLayerTranscoder:
        for group in (self.encoder_weight, self.encoder_bias, self.decoder, self.b_pre):
            group[:] = [tensor.to(device=device, dtype=torch.float32) for tensor in group]
        return self

    def __call__(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        source = x.float()
        mu = source.mean(dim=-1, keepdim=True)
        centred = source - mu
        std = centred.std(dim=-1, keepdim=True)
        normalised = centred / (std + 1e-5) - self.b_pre[layer]
        pre = F.linear(normalised, self.encoder_weight[layer], self.encoder_bias[layer])
        top = torch.topk(pre, k=self.k, dim=-1, sorted=False)
        latents = torch.zeros_like(pre).scatter_(-1, top.indices, F.relu(top.values))
        recon = latents @ self.decoder[layer] + self.b_pre[layer]
        return (recon * std + mu).to(x.dtype)


def load_replacement(path: Path) -> tuple[PerLayerTranscoder, dict[str, torch.Tensor], Any]:
    """The released checkpoint's transcoder weights and its embedded backbone."""

    torch.serialization.add_safe_globals([argparse.Namespace])
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    hyperparameters = checkpoint["hyper_parameters"]["args"]
    transcoder = PerLayerTranscoder(
        {key[len("plt.") :]: value for key, value in state.items() if key.startswith("plt.")},
        hyperparameters,
    )
    backbone = {
        key[len("progen3_model.") :]: value
        for key, value in state.items()
        if key.startswith("progen3_model.")
    }
    return transcoder, backbone, hyperparameters
