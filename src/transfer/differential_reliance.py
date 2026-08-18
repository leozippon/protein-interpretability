"""Causal differential reliance: does ablating one crosscoder latent move one checkpoint more than the other?

**What this is for.** EXP-R2-210 registered a causal successor to the relative
decoder norm. The decoder norm is a property of the *dictionary's parameters*;
the definition R2.4's admission list actually needs is causal -- *a latent is
model-specific if ablating it changes behaviour in one checkpoint and not the
other* -- and that is measured on the models. This module is the instrument for
that measurement. It consumes a fitted :class:`~src.transfer.crosscoder.Crosscoder`
and two :class:`~src.transfer.replaceable.ReplaceableModel` handles and returns,
per site and per live latent, how much the ablation moved each checkpoint's
next-token likelihood and how much a matched random direction of the same norm
moved it.

**The four constraints EXP-R2-210 froze, and where each one lives in this file.**

1. **Additive single-latent perturbation, never replacement.** The intervention
   subtracts ``f_i * scale_m * W_dec[m, l, i]`` from the layer's own output and
   leaves the model otherwise intact -- :func:`ablation_deltas` and
   :func:`subtracted_at`. It must never splice the crosscoder's
   reconstruction in as a replacement model, and the reason is measured: R2.3
   found behavioural recovery on this joint checkpoint negative in all four cells
   and negative in text on the base checkpoint too, so the spliced model is worse
   than mean-ablating the whole block and there is no dynamic range inside it in
   which to read one latent of thousands. The additive form needs only the
   decoder direction to be meaningful.

2. **A matched random-direction control, same norm, same site, both
   checkpoints.** :func:`matched_random_directions`. The precedent is L17, where
   raw effects read -0.82 against -0.83 -- indistinguishable -- until a matched
   random control at -0.441 against -0.208 supplied the scale. The
   shuffled-pairing null is **retired for this statistic**: it is a null for
   pairing correspondence in a representational readout, it says nothing about
   the size of an intervention, and it is a ceiling rather than the floor an
   effect size has to be read against.

3. **The statistic is differential reliance, not differential possession.** At
   the fitted lambda = 0 dictionary ``polarised`` is 0.000 -- every live latent
   decodes into both checkpoints with comparable weight -- so ablating one
   removes near-identical vectors from both residual streams and any difference
   in effect is a property of the *downstream models*. What this measures is
   "the same feature carries more of the computation in one checkpoint than the
   other": the **retained-but-reweighted** case. It **structurally cannot see**
   features *introduced* or *removed* by a training stage, because there are no
   latents belonging to one model alone for a stage to introduce or remove. The
   sentence travels into every artefact as :data:`RELIANCE_BLIND_SPOT`.

4. **Disjoint-support packing by greedy colouring; no subset, no attribution
   screen.** :func:`pack_disjoint_supports`. No gradient or attribution proxy may
   select which latents are measured: L5 is exactly the case where a plausible
   selector failed to rank causal importance, at Spearman -0.062, p = 0.71.

**The one place this file departs from the pre-registration, and why.**
EXP-R2-210 packs latents whose *firing positions* are disjoint. Position-level
disjointness is **not** sufficient for the packed pass to equal the single-latent
passes, and the gap is not numerical noise. The intervention lands on layer
``l``'s output at position ``q``; every layer above ``l`` mixes across positions
through attention, so the ablation reaches every later position of the **same
sequence**. Pack latent ``j`` firing at ``q`` beside latent ``i`` firing at
``p > q`` in one sequence and ``i`` is no longer scored under "ablate ``i``" but
under "ablate ``i`` and ``j``", which is a different estimand carrying an
interaction. The conflict relation is therefore taken at **row** granularity --
two latents may share a pass only if no sequence of the batch carries firing
positions of both -- which is the coarsest granularity at which the model does
not mix, and is exactly the granularity at which packing is bitwise identical to
ablating one latent at a time. :data:`PACKING_RULE` states it in the artefact and
``tests/test_differential_reliance.py`` checks both halves: packed equals
individual bitwise under the row rule, and *fails* to under the position rule.

The cost consequence is reported rather than absorbed. Row-disjointness packs
densely only when each latent is measured on a slice of the cohort rather than on
all of it, so ``--rows-per-latent`` is the knob that buys the packing, and its
default is the whole cohort -- the pre-registered behaviour, at the
pre-registered cost.

**What this module does not own.** It never loads a checkpoint, never draws a
cohort and never decides admissibility. It consumes a backbone handle satisfying
:class:`RelianceBackbone` -- which :class:`~src.transfer.replaceable.JointReplaceable`
satisfies unchanged -- so the whole instrument is certifiable on a synthetic
paired backbone with known ground truth before it is pointed at 6.74B parameters
of anything, exactly as ``crosscoder.SyntheticGroundTruth`` lets the dictionary
be.
"""

from __future__ import annotations

import copy
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .crosscoder import Crosscoder, CrosscoderConfig
from .io import sha256_file

#: Schema of the fitted-dictionary file this stage consumes. A Crosscoder's
#: weights are not currently written by any stage -- ``32_crosscoder.py`` writes
#: its readout and drops the object -- so the causal readout needs a persisted
#: dictionary and this is the one format it will accept.
DICTIONARY_SCHEMA = "r2_transfer_crosscoder_state_v1"

#: Fields of the dictionary manifest that must match the run that reads it. A
#: dictionary fitted to a different checkpoint pair, mode, tensor or width is not
#: a dictionary for this measurement, and every one of those mismatches would
#: otherwise produce finite, plausible numbers about the wrong thing.
DICTIONARY_IDENTITY_FIELDS: tuple[str, ...] = (
    "backbone_pair_sha256",
    "mode",
    "tensor",
)

#: Keys the artefact must carry once per fitted site, checked for presence as
#: well as for length. Absence and collapse are the same failure to a reader and
#: absence is the easier one to miss.
REQUIRED_PER_SITE_FIELDS: tuple[str, ...] = (
    "site_per_site",
    "live_latents_per_site",
    "measured_latents_per_site",
    "passes_per_site",
    # Required rather than optional because it is a SIZING INPUT and not a
    # diagnostic: the mean number of cohort sequences a live latent fires
    # somewhere in is what decides whether disjoint-support packing packs at all,
    # and an artefact that omits it invites the next campaign to re-derive a
    # packing saving that does not exist.
    "mean_cohort_rows_per_live_latent_per_site",
)

#: Suffix marking a per-site field, the same convention
#: :func:`src.transfer.crosscoder.assert_per_layer_fields` enforces recursively.
PER_SITE_SUFFIX = "_per_site"

RELIANCE_BLIND_SPOT = (
    "this is differential RELIANCE and not differential possession. It measures "
    "whether ablating a latent changes behaviour more in one checkpoint than in "
    "the other, which is the RETAINED-BUT-REWEIGHTED case: the same feature "
    "carrying more of the computation on one side. It STRUCTURALLY CANNOT see a "
    "feature INTRODUCED or REMOVED by a training stage. At the lambda = 0 "
    "dictionary this readout is defined on, polarised is 0.000 -- every live "
    "latent decodes into both checkpoints with comparable weight -- so there are "
    "no model-exclusive latents whose presence or absence could be read, and "
    "ablating one removes near-identical vectors from both residual streams. A "
    "large differential reliance is therefore a statement about the DOWNSTREAM "
    "MODELS and not about the dictionary having allocated a latent to one of them"
)

PACKING_RULE = (
    "two latents share a forward pass only when no SEQUENCE of the batch carries "
    "firing positions of both. Row granularity and not position granularity, and "
    "the difference is not conservatism: the intervention lands on layer l's "
    "output at position q, every layer above l mixes across positions through "
    "attention, so the ablation reaches every later position of the same "
    "sequence. Two position-disjoint latents in one sequence would each be scored "
    "under 'ablate both', which is a different estimand carrying an interaction. "
    "At row granularity a packed pass is BITWISE identical to ablating each of "
    "its latents in its own forward pass, because rows of a batch do not "
    "interact; that identity is checked in tests/test_differential_reliance.py, "
    "together with its failure under the position rule. Packs are built by greedy "
    "colouring (Welsh-Powell order, first fit) on the row-overlap graph, computed "
    "implicitly rather than materialised: the graph over 11,691 latents has 68M "
    "possible edges"
)

CONTROL_RULE = (
    "for every latent, a direction drawn uniformly on the unit sphere and scaled "
    "to that latent's OWN decoder norm in that role, subtracted with the SAME "
    "activation coefficients at the SAME positions of the SAME site in BOTH "
    "checkpoints. The direction is keyed to (seed, site, latent) so it does not "
    "depend on which pack the latent landed in or on the order the cohort was "
    "streamed. It is a FLOOR: the effect a perturbation of this size at this site "
    "has when it points nowhere in particular, which is the scale a raw "
    "differential effect is unreadable without (L17: -0.82 against -0.83 became "
    "readable only beside -0.441 against -0.208). The shuffled-pairing null is "
    "NOT used and is retired for this statistic -- it controls pairing "
    "correspondence in a representational readout, not the size of an "
    "intervention, and it is a ceiling rather than a floor"
)

_UNIT_SPHERE_TINY = 1e-12


# ------------------------------------------------------------- the backbone


class RelianceBackbone(Protocol):
    """The three things this instrument needs from a model, and nothing else.

    :class:`~src.transfer.replaceable.JointReplaceable` satisfies this
    structurally and unchanged; so does the synthetic paired backbone below. The
    protocol is deliberately narrower than ``ReplaceableModel``: a readout that
    only reads next-token likelihood under a block-output perturbation should not
    be able to reach a checkpoint's renderer, its component grid or its cohort.
    """

    def block_intercept(
        self, fn: Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor | None]
    ) -> Any:
        """Context manager reading or replacing every block's output."""

    def scored_logits(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(logits, targets, mask)`` aligned for next-token scoring, fp32 logits."""

    def content_mask(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Which positions of the batch are content, in ``(rows, width)``."""


def assert_backbone(model: Any, *, role: str) -> None:
    """Refuse a handle that does not carry the three methods this readout needs."""

    missing = [
        name
        for name in ("block_intercept", "scored_logits", "content_mask")
        if not callable(getattr(model, name, None))
    ]
    if missing:
        raise TypeError(
            f"the {role} handle is missing {missing}, so it cannot be perturbed at "
            "a block output or scored for next-token likelihood. This readout is "
            "defined on a ReplaceableModel and duck-typing a partial one would "
            "measure something else without saying so"
        )


# ------------------------------------------------------ dictionary on disk


def save_crosscoder(
    path: Any,
    model: Crosscoder,
    *,
    backbone_pair_sha256: str,
    mode: str,
    tensor: str,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Persist a fitted Crosscoder together with what it was fitted to.

    **Why this exists here and not in the trainer.** ``32_crosscoder.py`` writes a
    JSON readout and lets the fitted object fall out of scope, so no crosscoder
    weights exist on disk anywhere in this repository. A causal readout cannot be
    computed from a readout: it needs ``W_dec`` and the frozen normalisation
    scales. The format is a plain ``torch.save`` of the config, the state dict and
    the identity of the pair -- and the identity is not decoration, because a
    dictionary silently read against the wrong checkpoints produces finite
    numbers about nothing.

    ``extra`` should carry the fit's cohort parameters -- ``steps``,
    ``fit_batch_size``, ``eval_sequences``, ``corpus_seed``, ``max_tokens`` --
    because the held-out offset is ``steps x batch_size`` and a reader that
    re-derives the cohort from different ones measures a different population
    under the same name. :func:`assert_dictionary_matches` checks whichever of
    them are present.
    """

    if not bool(model.scale_is_set):
        raise ValueError(
            "this Crosscoder's normalisation scales were never frozen, so its "
            "decoder norms are in an undefined gauge and the ablation vectors "
            "derived from them would not be the ones it was fitted with"
        )
    payload = {
        "schema": DICTIONARY_SCHEMA,
        "config": {
            "sites": list(model.config.sites),
            "d_model": int(model.config.d_model),
            "d_hidden": int(model.config.d_hidden),
            "k": int(model.config.k),
            "auxk": int(model.config.auxk),
            "dead_steps": int(model.config.dead_steps),
            "aux_weight": float(model.config.aux_weight),
            "decoder_norm_penalty": float(model.config.decoder_norm_penalty),
            "pairing": str(model.config.pairing),
            "role_names": list(model.config.role_names),
        },
        "backbone_pair_sha256": str(backbone_pair_sha256),
        "mode": str(mode),
        "tensor": str(tensor),
        "state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "extra": dict(extra or {}),
    }
    torch.save(payload, str(path))


def load_crosscoder(path: Any, *, device: str | torch.device = "cpu") -> tuple[Crosscoder, dict[str, Any]]:
    """A fitted Crosscoder and its manifest, refusing anything this stage cannot read.

    ``weights_only=False`` is required because the payload carries the config
    mapping beside the tensors; the file is one this programme wrote and its
    digest reaches the artefact.
    """

    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema") != DICTIONARY_SCHEMA:
        raise ValueError(
            f"{path} is not a {DICTIONARY_SCHEMA} dictionary; a Crosscoder for this "
            "readout must be written by differential_reliance.save_crosscoder so "
            "that the checkpoint pair it was fitted to travels with its weights"
        )
    config = CrosscoderConfig(
        sites=tuple(int(value) for value in payload["config"]["sites"]),
        d_model=int(payload["config"]["d_model"]),
        d_hidden=int(payload["config"]["d_hidden"]),
        k=int(payload["config"]["k"]),
        auxk=int(payload["config"]["auxk"]),
        dead_steps=int(payload["config"]["dead_steps"]),
        aux_weight=float(payload["config"]["aux_weight"]),
        decoder_norm_penalty=float(payload["config"]["decoder_norm_penalty"]),
        pairing=str(payload["config"]["pairing"]),
        role_names=tuple(payload["config"]["role_names"]),
    )
    model = Crosscoder(config)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    if not bool(model.scale_is_set):
        raise ValueError(
            f"{path} carries a Crosscoder whose normalisation scales were never "
            "frozen; every decoder norm read from it would be in an undefined gauge"
        )
    manifest = {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema": DICTIONARY_SCHEMA,
        "config": model.config.record(),
        "backbone_pair_sha256": str(payload["backbone_pair_sha256"]),
        "mode": str(payload["mode"]),
        "tensor": str(payload["tensor"]),
        "extra": dict(payload.get("extra") or {}),
    }
    return model, manifest


def assert_dictionary_matches(
    manifest: Mapping[str, Any],
    *,
    backbone_pair_sha256: str,
    mode: str,
    tensor: str,
    d_model: int,
    n_layers: int,
    cohort: Mapping[str, Any] | None = None,
) -> None:
    """Refuse a dictionary that does not describe the checkpoints it is read against.

    Four independent ways to be wrong and all four are fatal rather than warned
    about: the wrong checkpoint pair, the wrong mode, the wrong captured tensor,
    or a site index the backbone does not have. Each of them yields a finite
    number, which is exactly why none of them may be a warning.

    ``cohort`` is the fifth and is checked only against fields the dictionary
    actually recorded, because the trainer does not record them today. It is
    written this way rather than left out so the check turns itself on the moment
    :func:`save_crosscoder` is called with them: the held-out offset is
    ``steps x batch_size``, so a mismatch there measures a different population
    under the same name and nothing downstream would show it.
    """

    cohort = dict(cohort or {})

    observed = {
        "backbone_pair_sha256": str(backbone_pair_sha256),
        "mode": str(mode),
        "tensor": str(tensor),
    }
    wrong = {
        field: (manifest.get(field), observed[field])
        for field in DICTIONARY_IDENTITY_FIELDS
        if str(manifest.get(field)) != observed[field]
    }
    if wrong:
        raise ValueError(
            f"this dictionary was fitted to {wrong} (recorded, observed). A "
            "Crosscoder read against a checkpoint pair, mode or tensor other than "
            "the one it was fitted to still yields finite decoder norms, so this "
            "is a refusal and not a warning"
        )
    recorded = dict(manifest.get("extra") or {})
    disagreeing = {
        field: (recorded[field], value)
        for field, value in cohort.items()
        if field in recorded and recorded[field] != value
    }
    if disagreeing:
        raise ValueError(
            f"this dictionary records {disagreeing} (fitted, requested) for the "
            "cohort draw. The held-out offset is steps x batch size, so a "
            "disagreement here means the readout would run on a different "
            "population from the one the dictionary was held out on"
        )
    config = manifest["config"]
    if int(config["d_model"]) != int(d_model):
        raise ValueError(
            f"this dictionary is {config['d_model']}-dimensional and the backbone "
            f"is {d_model}-dimensional"
        )
    outside = sorted(layer for layer in config["sites"] if not 0 <= int(layer) < n_layers)
    if outside:
        raise ValueError(
            f"this dictionary carries sites {outside}, which are outside the "
            f"backbone's 0..{n_layers - 1}"
        )


# ----------------------------------------------------- ablation directions


def ablation_directions(model: Crosscoder, *, site: int) -> torch.Tensor:
    """``(2, d_hidden, d_model)``: what one unit of each latent writes, per role, in raw space.

    The Crosscoder's objective lives in the scaled space and
    :meth:`Crosscoder.reconstruct` is the only place the scaling is undone, so
    latent ``i``'s contribution to role ``m``'s reconstruction of the raw
    activation is ``f_i * scale[m, l] * W_dec[m, l, i]``. That product is the
    vector this readout subtracts, and taking ``W_dec`` without the scale would
    subtract a vector the dictionary never wrote.
    """

    index = _site_index(model.config, site)
    with torch.no_grad():
        return model.W_dec[:, index] * model.scale[:, index].view(2, 1, 1)


def matched_random_directions(
    directions: torch.Tensor, *, seed: int, site: int, latents: Sequence[int]
) -> torch.Tensor:
    """A random direction per latent, at that latent's own per-role decoder norm.

    One unit vector per latent, shared by the two roles and rescaled to each
    role's own norm, so the control differs from the measurement in **direction
    alone** -- same site, same positions, same activation coefficients, same
    magnitude on each side. Sharing the unit vector across roles is the tighter
    match: two independent draws would add a second difference between the roles
    that the measurement does not have.

    Keyed to ``(seed, site, latent)`` rather than drawn in stream order, so a
    latent's control direction does not depend on which pack it landed in, on the
    cohort's batching, or on how many latents were measured beside it.
    """

    if directions.ndim != 3 or directions.shape[0] != 2:
        raise ValueError(
            f"expected (2, n_latents, d_model) decoder directions, got "
            f"{tuple(directions.shape)}"
        )
    n_latents, d_model = directions.shape[1], directions.shape[2]
    if len(latents) != n_latents:
        raise ValueError(
            f"{len(latents)} latent indices for {n_latents} direction rows; the "
            "control is keyed to the latent index and the two must correspond"
        )
    norms = directions.norm(dim=2)
    out = torch.empty_like(directions)
    for row, latent in enumerate(latents):
        generator = torch.Generator().manual_seed(
            (int(seed) * 1_000_003 + int(site) * 1_000_033 + int(latent)) % (2**63 - 1)
        )
        draw = torch.randn(d_model, generator=generator, dtype=torch.float32)
        unit = draw / draw.norm().clamp_min(_UNIT_SPHERE_TINY)
        unit = unit.to(device=directions.device, dtype=directions.dtype)
        out[0, row] = unit * norms[0, row]
        out[1, row] = unit * norms[1, row]
    return out


def _site_index(config: CrosscoderConfig, site: int) -> int:
    if int(site) not in config.sites:
        raise ValueError(
            f"this Crosscoder carries no parameters for layer {site}; its fitted "
            f"sites are {list(config.sites)}"
        )
    return config.sites.index(int(site))


# -------------------------------------------------------- supports and packs


@dataclass(frozen=True)
class LatentSupport:
    """Where one latent fires on one batch, and how hard.

    ``rows`` and ``positions`` index the batch's ``(rows, width)`` grid and are
    the positions the intervention is applied at **and** the positions the effect
    is read at -- deliberately the same set. EXP-R2-210 scores each latent "on the
    positions where it actually fires", and applying the perturbation anywhere the
    effect is not read would put an unmeasured intervention into the pass.
    """

    latent: int
    rows: torch.Tensor
    positions: torch.Tensor
    coefficients: torch.Tensor

    def __post_init__(self) -> None:
        if not (self.rows.shape == self.positions.shape == self.coefficients.shape):
            raise ValueError(
                f"latent {self.latent}'s support is ragged: rows "
                f"{tuple(self.rows.shape)}, positions {tuple(self.positions.shape)}, "
                f"coefficients {tuple(self.coefficients.shape)}"
            )
        if self.rows.ndim != 1:
            raise ValueError("a support is a flat list of (row, position) firings")

    @property
    def size(self) -> int:
        return int(self.rows.numel())

    @property
    def row_set(self) -> frozenset[int]:
        return frozenset(int(value) for value in self.rows.tolist())


def latent_supports(
    latents: torch.Tensor,
    *,
    rows: torch.Tensor,
    positions: torch.Tensor,
    keep: Sequence[int],
) -> dict[int, LatentSupport]:
    """Turn one site's latent activations into one support per kept latent.

    ``latents`` is ``(n_positions, d_hidden)`` for the positions this readout is
    defined on, and ``rows``/``positions`` say where in the batch each of those
    rows of the matrix sits. Flat rather than a ``(rows, width, d_hidden)`` grid
    because the grid is the one object in this pipeline that does not fit: at
    ``d_hidden`` 16,384 and a 4,096-position batch it is 268 MB per site while the
    positions actually scored are a small fraction of it.

    Only latents in ``keep`` are returned, and only those that fire here appear at
    all: a latent with no firing on this batch has no intervention on it, and
    scheduling an empty pass would be a forward pass measuring nothing.
    """

    if latents.ndim != 2:
        raise ValueError(
            f"expected (n_positions, d_hidden) latents, got {tuple(latents.shape)}"
        )
    if not (rows.shape == positions.shape == latents.shape[:1]):
        raise ValueError(
            f"{tuple(rows.shape)} rows and {tuple(positions.shape)} positions were "
            f"given for {latents.shape[0]} rows of latent activations"
        )
    kept = torch.zeros(latents.shape[1], dtype=torch.bool, device=latents.device)
    if len(keep):
        kept[torch.as_tensor(list(keep), dtype=torch.long, device=latents.device)] = True
    entries, columns = ((latents > 0) & kept[None, :]).nonzero(as_tuple=True)
    values = latents[entries, columns]
    rows = rows.to(latents.device)[entries]
    positions = positions.to(latents.device)[entries]
    supports: dict[int, LatentSupport] = {}
    order = torch.argsort(columns, stable=True)
    columns, rows, positions, values = (
        columns[order],
        rows[order],
        positions[order],
        values[order],
    )
    if columns.numel():
        unique, counts = torch.unique_consecutive(columns, return_counts=True)
        start = 0
        for latent, count in zip(unique.tolist(), counts.tolist()):
            stop = start + count
            supports[int(latent)] = LatentSupport(
                latent=int(latent),
                rows=rows[start:stop].cpu(),
                positions=positions[start:stop].cpu(),
                coefficients=values[start:stop].detach().float().cpu(),
            )
            start = stop
    return supports


def pack_disjoint_supports(supports: Sequence[LatentSupport]) -> list[tuple[int, ...]]:
    """Greedy colouring on the row-overlap graph: which latents may share a pass.

    Welsh-Powell order -- widest support first, ties broken by latent index -- and
    first fit into the earliest pass whose rows it does not touch. Deterministic
    for a given input order, and the graph is never materialised: at 11,691
    latents it would carry 68M possible edges while the colouring only ever asks
    whether one support's rows meet a pass's.

    Returns packs as tuples of **indices into** ``supports``, ascending inside a
    pack and with packs in creation order, so that the schedule is reproducible
    and can be recorded.
    """

    row_sets = [support.row_set for support in supports]
    empty = [index for index, rows in enumerate(row_sets) if not rows]
    if empty:
        raise ValueError(
            f"supports {empty[:8]} touch no row, so they name an intervention with "
            "no positions; drop them before packing rather than scheduling a "
            "forward pass that measures nothing"
        )
    order = sorted(
        range(len(supports)), key=lambda index: (-len(row_sets[index]), supports[index].latent)
    )
    packs: list[list[int]] = []
    occupied: list[set[int]] = []
    for index in order:
        rows = row_sets[index]
        for pack, used in zip(packs, occupied):
            if used.isdisjoint(rows):
                pack.append(index)
                used |= rows
                break
        else:
            packs.append([index])
            occupied.append(set(rows))
    return [tuple(sorted(pack)) for pack in packs]


# ------------------------------------------------------------ the intervention


def ablation_deltas(
    support: LatentSupport, direction: torch.Tensor
) -> torch.Tensor:
    """``(n_firings, d_model)``: what is subtracted at each of one latent's positions."""

    if direction.ndim != 1:
        raise ValueError(
            f"one latent's ablation direction is a vector, got {tuple(direction.shape)}"
        )
    # Supports are held on the host -- there are tens of thousands of them and
    # they outlive any one pass -- while the decoder rows stay where the model is.
    coefficients = support.coefficients.to(device=direction.device, dtype=direction.dtype)
    return coefficients.unsqueeze(1) * direction.unsqueeze(0)


@contextmanager
def subtracted_at(
    model: RelianceBackbone,
    *,
    site: int,
    rows: torch.Tensor,
    positions: torch.Tensor,
    deltas: torch.Tensor,
) -> Iterator[None]:
    """Subtract a per-position vector from one layer's block output, and nothing else.

    The additive form of constraint 1, and the whole of the intervention: the
    block's own output is reduced by the latents' contributions at the positions
    they fire on, every other position and every other layer passes through
    untouched, and no reconstruction is spliced in anywhere.

    Refuses a repeated ``(row, position)``. Two latents writing to one position
    would make the pass an interaction rather than a set of independent single-
    latent ablations, and the packer's row rule already forbids it -- so reaching
    here means the schedule and the intervention disagree, which is the failure
    mode a silent ``index_put_`` accumulate would hide.
    """

    if not (rows.shape == positions.shape == deltas.shape[:1]):
        raise ValueError(
            f"the perturbation is ragged: rows {tuple(rows.shape)}, positions "
            f"{tuple(positions.shape)}, deltas {tuple(deltas.shape)}"
        )
    if rows.numel() == 0:
        raise ValueError("an intervention with no positions is not an intervention")
    flat = rows.to(torch.long) * (int(positions.max()) + 1) + positions.to(torch.long)
    if int(torch.unique(flat).numel()) != int(flat.numel()):
        raise ValueError(
            "two of this pass's latents fire at the same (row, position), so the "
            "pass would measure their interaction rather than either ablation"
        )
    seen = {"hit": False}

    def fn(layer: int, block_input: torch.Tensor, block_output: torch.Tensor):
        if int(layer) != int(site):
            return None
        seen["hit"] = True
        out = block_output.clone()
        index_rows = rows.to(out.device)
        index_positions = positions.to(out.device)
        out[index_rows, index_positions] = (
            out[index_rows, index_positions] - deltas.to(device=out.device, dtype=out.dtype)
        )
        return out

    with model.block_intercept(fn):
        yield
    if not seen["hit"]:
        raise RuntimeError(
            f"layer {site} was never intercepted during the forward pass, so the "
            "ablation this measurement reports was never applied"
        )


@torch.no_grad()
def position_nll(
    logits: torch.Tensor,
    targets: torch.Tensor,
    rows: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Next-token negative log-likelihood at named ``(row, position)`` pairs, in nats.

    Gathered at the positions asked for rather than computed over the whole grid:
    the logits of one batch of this lineage are half a gigabyte in fp32 and one
    ablated pass only ever reads a few thousand of their rows.
    """

    device = logits.device
    selected = logits[rows.to(device), positions.to(device)].float()
    wanted = targets[rows.to(device), positions.to(device)].to(device)
    return -torch.log_softmax(selected, dim=-1).gather(1, wanted.unsqueeze(1)).squeeze(1)


@torch.no_grad()
def grid_nll(
    logits: torch.Tensor, targets: torch.Tensor, *, chunk_rows: int = 1
) -> torch.Tensor:
    """Per-position next-token NLL over the whole ``(rows, width - 1)`` grid.

    Row-chunked because the intermediate is the logits again: at this lineage's
    vocabulary one batch is half a gigabyte in fp32 and the log-softmax would
    double it. Computed once per batch so that every ablated pass reads a clean
    reference of a few thousand floats rather than holding the clean logits.
    """

    out = torch.empty(targets.shape, dtype=torch.float32, device=logits.device)
    for start in range(0, logits.shape[0], max(1, int(chunk_rows))):
        stop = min(start + max(1, int(chunk_rows)), logits.shape[0])
        block = torch.log_softmax(logits[start:stop].float(), dim=-1)
        out[start:stop] = -block.gather(
            2, targets[start:stop].to(logits.device).unsqueeze(2)
        ).squeeze(2)
        del block
    return out


@dataclass(frozen=True)
class CleanPass:
    """One checkpoint's unperturbed forward on one batch: what every ablation is read against."""

    #: ``(rows, width - 1)`` next-token targets, as ``scored_logits`` aligns them.
    targets: torch.Tensor
    #: ``(rows, width - 1)`` boolean: positions this readout is defined on.
    scored: torch.Tensor
    #: Per-site block outputs, keyed by backbone layer index, ``(rows, width, d_model)``.
    block_outputs: dict[int, torch.Tensor]
    #: ``(rows, width - 1)`` clean next-token NLL in nats.
    nll: torch.Tensor


@torch.no_grad()
def clean_pass(
    model: RelianceBackbone, batch: dict[str, torch.Tensor], *, sites: Sequence[int]
) -> CleanPass:
    """The unperturbed forward: block outputs at the fitted sites and the clean likelihood.

    One forward for both, because ``scored_logits`` runs the model and
    ``block_intercept`` taps it while it runs -- the same composition
    ``17_train_transcoder.capture`` uses, with the logits kept instead of
    discarded.
    """

    assert_backbone(model, role="scored")
    wanted = {int(site) for site in sites}
    captured: dict[int, torch.Tensor] = {}

    def tap(layer: int, block_input: torch.Tensor, block_output: torch.Tensor) -> None:
        if int(layer) in wanted:
            captured[int(layer)] = block_output.detach()
        return None

    with model.block_intercept(tap):
        logits, targets, scored = model.scored_logits(batch)
    absent = sorted(wanted - set(captured))
    if absent:
        raise RuntimeError(
            f"layers {absent} were never intercepted, so this backbone has no block "
            f"output at every fitted site of the dictionary"
        )
    content = model.content_mask(batch)
    if content.shape[0] != scored.shape[0] or content.shape[1] != scored.shape[1] + 1:
        raise ValueError(
            f"the content mask is {tuple(content.shape)} and the scored mask is "
            f"{tuple(scored.shape)}; a scored position j predicts token j+1, so the "
            "two must differ by exactly one column"
        )
    # A position is measurable when it is content -- the population the dictionary
    # was fitted on -- AND carries a scored next token, because the effect of an
    # ablation at position j is read on the prediction j makes.
    measurable = content[:, : scored.shape[1]] & scored
    return CleanPass(
        targets=targets,
        scored=measurable,
        block_outputs=captured,
        nll=grid_nll(logits, targets),
    )


@torch.no_grad()
def measure_pack(
    model: RelianceBackbone,
    batch: dict[str, torch.Tensor],
    reference: CleanPass,
    *,
    site: int,
    pack: Sequence[LatentSupport],
    directions: Mapping[int, torch.Tensor],
) -> dict[int, torch.Tensor]:
    """One forward pass carrying one pass-worth of row-disjoint ablations.

    Returns, per latent, the per-position ``ablated - clean`` next-token NLL in
    nats at that latent's own firing positions. The subtraction is per position
    and is never reduced here; the caller accumulates.
    """

    if not pack:
        raise ValueError("an empty pass has no ablation to measure")
    rows = torch.cat([support.rows for support in pack])
    positions = torch.cat([support.positions for support in pack])
    deltas = torch.cat(
        [ablation_deltas(support, directions[support.latent]) for support in pack]
    )
    with subtracted_at(
        model, site=site, rows=rows, positions=positions, deltas=deltas
    ):
        logits, targets, _ = model.scored_logits(batch)
    if not torch.equal(targets, reference.targets):
        raise RuntimeError(
            "the ablated pass produced different next-token targets from the clean "
            "pass on the same batch, so the two are not scoring the same positions"
        )
    ablated = position_nll(logits, targets, rows, positions)
    clean = reference.nll[rows.to(reference.nll.device), positions.to(reference.nll.device)]
    del logits
    out: dict[int, torch.Tensor] = {}
    start = 0
    for support in pack:
        stop = start + support.size
        out[support.latent] = (ablated[start:stop] - clean[start:stop]).float().cpu()
        start = stop
    return out


# ------------------------------------------------------------- accumulation


class RelianceAccumulator:
    """Per-latent sums of the per-position effect, for one role and one arm.

    Sums and counts rather than means, so that batches of different widths
    contribute in proportion to the positions they carry, and so that the mean is
    formed once at the end from a quantity that was never rounded through an
    intermediate average.
    """

    def __init__(self, latents: Sequence[int]) -> None:
        self.latents = tuple(int(value) for value in latents)
        self._index = {latent: row for row, latent in enumerate(self.latents)}
        self.total = torch.zeros(len(self.latents), dtype=torch.float64)
        self.total_square = torch.zeros(len(self.latents), dtype=torch.float64)
        self.count = torch.zeros(len(self.latents), dtype=torch.long)

    def update(self, effects: Mapping[int, torch.Tensor]) -> None:
        for latent, values in effects.items():
            row = self._index[int(latent)]
            doubled = values.double()
            self.total[row] += float(doubled.sum())
            self.total_square[row] += float((doubled**2).sum())
            self.count[row] += int(values.numel())

    def mean(self) -> torch.Tensor:
        """Mean effect per latent, ``nan`` where a latent was never measured."""

        out = torch.full((len(self.latents),), float("nan"), dtype=torch.float64)
        seen = self.count > 0
        out[seen] = self.total[seen] / self.count[seen].double()
        return out

    def standard_error(self) -> torch.Tensor:
        """Standard error of that mean over the latent's own firing positions.

        Positions inside one sequence are not independent, so this is a
        within-latent dispersion and not a confidence interval; it is carried
        because a latent measured on nine positions and one measured on nine
        hundred should not read as equally resolved.
        """

        out = torch.full((len(self.latents),), float("nan"), dtype=torch.float64)
        seen = self.count > 1
        n = self.count[seen].double()
        mean = self.total[seen] / n
        variance = (self.total_square[seen] / n - mean**2).clamp_min(0.0) * (n / (n - 1))
        out[seen] = (variance / n).sqrt()
        return out


def differential_reliance(
    base: torch.Tensor, adapted: torch.Tensor
) -> torch.Tensor:
    """``adapted - base``: how much more the ablation moved one checkpoint than the other.

    Positive means the adapted checkpoint relies on the latent more. A latent
    never measured on either side stays ``nan`` rather than reading as zero,
    because "no difference" and "no measurement" are the two answers this readout
    must never confuse.
    """

    if base.shape != adapted.shape:
        raise ValueError(
            f"the two roles carry {tuple(base.shape)} and {tuple(adapted.shape)} "
            "effects; differential reliance is defined latent by latent"
        )
    return adapted.double() - base.double()


# ------------------------------------- the synthetic paired ground-truth check


class _CausalBlock(nn.Module):
    """One attention + feed-forward block whose feed-forward output IS the residual write."""

    def __init__(self, d_model: int, generator: torch.Generator) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(d_model)
        self.norm_feed_forward = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model, bias=False),
        )
        for parameter in self.parameters():
            if parameter.ndim >= 2:
                with torch.no_grad():
                    parameter.uniform_(-0.5, 0.5, generator=generator)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normed = self.norm_attention(hidden)
        query, key, value = self.qkv(normed).chunk(3, dim=-1)
        attended = F.scaled_dot_product_attention(
            query.unsqueeze(1), key.unsqueeze(1), value.unsqueeze(1), is_causal=True
        ).squeeze(1)
        hidden = hidden + self.proj(attended)
        return hidden + self.mlp(self.norm_feed_forward(hidden))


class SyntheticPairedBackbone(nn.Module):
    """A tiny causal transformer with a declared, analytically exact differential reliance.

    **Why this is in ``src`` and not only in a test.** The claim this instrument
    makes -- "ablating latent ``i`` moves the adapted checkpoint more than the
    base one" -- is unfalsifiable on a real pair, exactly as the Crosscoder's
    specificity claim is, which is why ``crosscoder.SyntheticGroundTruth`` lives
    beside the Crosscoder. The construction that makes it falsifiable is part of
    the instrument.

    **The injection, and why it reads the site rather than the final layer.** Two
    backbones share **every weight**; they differ only in a scalar
    ``reliance_gain``, which weights a channel from the residual stream *at the
    intercepted site* into one token's logit:
    ``logit[readout] += gain * (direction . residual_after_site)``. Subtracting
    ``c * direction`` from that site's block output lowers the residual along
    ``direction`` by exactly ``c``, so the adapted model's readout logit falls by
    exactly ``gain * c`` and the base model's does not fall at all through this
    channel. The ground truth is therefore analytic: with ``p`` the readout
    token's clean probability, the adapted model's likelihood at that position
    worsens by ``gain * c * (1 - p)`` to first order, and differential reliance
    is **proportional to the injected gain**.

    An earlier construction injected the same channel from the *final* hidden
    state instead, and it is recorded here because it looked equivalent and was
    not: LayerNorm renormalises, and the blocks above the site are nonlinear, so
    subtracting a direction at the site changed the final-layer projection by an
    amount whose *sign* varied with the seed. A ground truth whose sign depends on
    the draw is not a ground truth.

    Everything not on that channel is an ordinary transformer with real causal
    attention, which is the property the packing check needs: an ablation at one
    position of a sequence must actually reach the later positions of that
    sequence, or the row rule would be untestable.
    """

    def __init__(
        self,
        *,
        vocab: int = 48,
        d_model: int = 32,
        n_layers: int = 3,
        seed: int = 0,
        reliance_site: int = 0,
        reliance_gain: float = 0.0,
        reliance_direction: torch.Tensor | None = None,
        reliance_reference: float = 0.0,
        readout_token: int = 0,
    ) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(int(seed))
        self.vocab = int(vocab)
        self.d_model = int(d_model)
        self.embedding = nn.Embedding(vocab, d_model)
        self.position = nn.Embedding(512, d_model)
        with torch.no_grad():
            self.embedding.weight.uniform_(-0.5, 0.5, generator=generator)
            self.position.weight.uniform_(-0.1, 0.1, generator=generator)
        self.blocks = nn.ModuleList(
            [_CausalBlock(d_model, generator) for _ in range(n_layers)]
        )
        self.norm_final = nn.LayerNorm(d_model)
        self.unembedding = nn.Linear(d_model, vocab, bias=False)
        with torch.no_grad():
            self.unembedding.weight.uniform_(-0.5, 0.5, generator=generator)
        if not 0 <= int(reliance_site) < n_layers:
            raise ValueError(
                f"the reliance channel reads layer {reliance_site} of a "
                f"{n_layers}-layer backbone"
            )
        self.reliance_site = int(reliance_site)
        self.reliance_gain = float(reliance_gain)
        # A constant subtracted from the channel before it is weighted. It cannot
        # change the ablation's effect -- the channel still falls by exactly the
        # ablated coefficient -- and it moves the *clean* operating point off
        # saturation. Without it the injected logit swamps the vocabulary, the
        # readout token's clean probability sits at 0.9998, and the closed-form
        # ground truth degenerates to a null the check would then "recover"
        # while demonstrating nothing.
        self.reliance_reference = float(reliance_reference)
        self.readout_token = int(readout_token)
        direction = (
            torch.zeros(d_model)
            if reliance_direction is None
            else reliance_direction.detach().clone().float()
        )
        if reliance_direction is not None:
            direction = direction / direction.norm().clamp_min(_UNIT_SPHERE_TINY)
        self.register_buffer("reliance_direction", direction)
        self.eval()

    def paired_with(
        self, *, reliance_gain: float, reliance_reference: float | None = None
    ) -> "SyntheticPairedBackbone":
        """A copy differing in the injected reliance gain and in nothing else."""

        twin = copy.deepcopy(self)
        twin.reliance_gain = float(reliance_gain)
        if reliance_reference is not None:
            twin.reliance_reference = float(reliance_reference)
        return twin

    @property
    def n_layers(self) -> int:
        return len(self.blocks)

    @torch.no_grad()
    def reliance_channel(self, input_ids: torch.Tensor) -> torch.Tensor:
        """``(rows, width)``: the scalar the injected readout reads at each position."""

        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = self.embedding(input_ids) + self.position(positions)[None]
        for index, block in enumerate(self.blocks):
            hidden = block(hidden)
            if index == self.reliance_site:
                return hidden @ self.reliance_direction
        raise RuntimeError("the reliance site is past the last block")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = self.embedding(input_ids) + self.position(positions)[None]
        channel = None
        for index, block in enumerate(self.blocks):
            hidden = block(hidden)
            if index == self.reliance_site:
                channel = hidden @ self.reliance_direction
        logits = self.unembedding(self.norm_final(hidden))
        if self.reliance_gain != 0.0:
            assert channel is not None
            logits = logits.index_add(
                2,
                torch.tensor([self.readout_token], device=logits.device),
                (self.reliance_gain * (channel - self.reliance_reference)).unsqueeze(2),
            )
        return logits

    # -- the RelianceBackbone surface --------------------------------------

    @contextmanager
    def block_intercept(
        self, fn: Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor | None]
    ) -> Iterator[None]:
        """The same contract ``JointReplaceable`` declares, over each block's feed-forward."""

        handles = []
        for layer, block in enumerate(self.blocks):

            def hook(module, inputs, output, layer: int = layer):
                return fn(layer, inputs[0], output)

            handles.append(block.mlp.register_forward_hook(hook))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def scored_logits(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self(batch["input_ids"])
        return (
            logits[..., :-1, :].float(),
            batch["input_ids"][..., 1:],
            batch["target_mask"],
        )

    def content_mask(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return batch["content_mask"]


def synthetic_batch(
    *, rows: int, width: int, vocab: int, seed: int, device: str | torch.device = "cpu"
) -> dict[str, torch.Tensor]:
    """A batch in the layout every ``ReplaceableModel`` produces, for the ground-truth check."""

    generator = torch.Generator().manual_seed(int(seed))
    ids = torch.randint(0, vocab, (rows, width), generator=generator)
    return {
        "input_ids": ids.to(device),
        "attention_mask": torch.ones(rows, width, dtype=torch.long, device=device),
        "content_mask": torch.ones(rows, width, dtype=torch.bool, device=device),
        "target_mask": torch.ones(rows, width - 1, dtype=torch.bool, device=device),
    }


def synthetic_supports(
    *,
    rows_per_latent: Sequence[Sequence[int]],
    positions_per_latent: Sequence[Sequence[int]],
    coefficient: float = 1.0,
) -> list[LatentSupport]:
    """Hand-built supports, so a packing or recovery check names its own firing pattern."""

    if len(rows_per_latent) != len(positions_per_latent):
        raise ValueError("one row list and one position list per latent")
    supports: list[LatentSupport] = []
    for latent, (rows, positions) in enumerate(zip(rows_per_latent, positions_per_latent)):
        if len(rows) != len(positions):
            raise ValueError(f"latent {latent}'s rows and positions differ in length")
        supports.append(
            LatentSupport(
                latent=latent,
                rows=torch.tensor(list(rows), dtype=torch.long),
                positions=torch.tensor(list(positions), dtype=torch.long),
                coefficients=torch.full((len(rows),), float(coefficient)),
            )
        )
    return supports


# ---------------------------------------------------------------- artefact


def assert_required_per_site_fields(payload: Any) -> None:
    """Every field of :data:`REQUIRED_PER_SITE_FIELDS` appears somewhere in the artefact.

    The presence half of the per-layer discipline.
    :func:`src.transfer.crosscoder.assert_required_per_site_fields` enforces the
    same property against the *Crosscoder's* required list, which is frozen and
    names fields this artefact does not carry; the shape half --
    :func:`crosscoder.assert_per_layer_fields` -- is generic and is imported
    rather than restated.
    """

    found: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if isinstance(key, str) and key in REQUIRED_PER_SITE_FIELDS:
                    found.add(key)
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(payload)
    missing = sorted(set(REQUIRED_PER_SITE_FIELDS) - found)
    if missing:
        raise ValueError(
            f"this artefact carries no {missing}; a per-layer quantity that is "
            "absent is as unreadable as one that was averaged"
        )


def packed_cost(
    *,
    live_latents: int,
    cohort_rows: int,
    rows_per_latent: int,
    batch_rows: int,
    roles: int = 2,
    arms: int = 2,
) -> dict[str, Any]:
    """Forward passes a round costs, packed and naive, from the run's own numbers.

    Arithmetic rather than a quoted estimate, because two factors are easy to drop
    and both are large. The unit of work is a ``(latent, sequence-row)`` cell and
    every cell must be run **four** times -- two checkpoints times measurement and
    matched control -- so a naive figure that counts only latents times the cohort
    understates the round by 4x.

    Packing does not reduce the number of cells. What it does is stop a pass from
    wasting rows: a latent occupying one row of a batch leaves ``batch_rows - 1``
    rows idle unless other latents fill them, so the achievable density is
    ``batch_rows`` latents per pass exactly when each latent takes at most one row
    per batch -- which is what spreading a latent's assigned rows across batches
    buys. The saving against naive is then ``cohort_rows / rows_per_latent``, and
    it comes from measuring each latent on a slice of the cohort, not from the
    packing itself.
    """

    for name, value in (
        ("live_latents", live_latents),
        ("cohort_rows", cohort_rows),
        ("rows_per_latent", rows_per_latent),
        ("batch_rows", batch_rows),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive; got {value}")
    if rows_per_latent > cohort_rows:
        raise ValueError(
            f"a latent cannot be measured on {rows_per_latent} of {cohort_rows} rows"
        )
    n_batches = math.ceil(cohort_rows / batch_rows)
    cells = live_latents * rows_per_latent * roles * arms
    naive_cells = live_latents * cohort_rows * roles * arms
    spread = rows_per_latent <= n_batches
    latents_per_pass = (
        batch_rows if spread else max(1, batch_rows // math.ceil(rows_per_latent / n_batches))
    )
    return {
        "naive_forward_passes": math.ceil(naive_cells / batch_rows),
        "packed_forward_passes": math.ceil(cells / batch_rows),
        "latents_per_pass_ceiling": int(latents_per_pass),
        "speedup_over_naive": cohort_rows / rows_per_latent,
        "rows_per_latent": int(rows_per_latent),
        "cohort_rows": int(cohort_rows),
        "batch_rows": int(batch_rows),
        "roles": int(roles),
        "arms": int(arms),
        "one_row_per_batch_achievable": bool(spread),
        "note": (
            "a cell is one (latent, sequence-row) and there are "
            "live x rows_per_latent x 2 checkpoints x 2 arms of them, the arms "
            "being the ablation and its matched random control. One forward pass "
            "covers batch_rows cells when the row rule lets batch_rows different "
            "latents share it, which needs each latent to occupy at most one row "
            "per batch"
        ),
    }


ESTIMAND = (
    "for each live latent i of a fitted Crosscoder at each admissible site l, the "
    "mean change in next-token negative log-likelihood, in nats, at the positions "
    "where i fires, when f_i * scale_m * W_dec[m, l, i] is SUBTRACTED from layer "
    "l's own output of checkpoint m and the model is otherwise left intact; and "
    "the difference of those two means, adapted minus base, beside the same "
    "quantity for a random direction of the same per-role norm ablated with the "
    "same coefficients at the same positions of the same site in both checkpoints"
)
