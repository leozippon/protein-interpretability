"""D3.h's adequacy criteria, as committed code rather than as arithmetic by hand.

**Why this exists.** EXP-R2-191 fixed two gates before R2.4's four independent
dictionaries existed -- Criterion A on the alignment artefacts and Criterion B on
the dictionaries -- and both were then applied by reading numbers out of JSON and
computing medians in a shell. Nothing about the layer window, the ratio, the
denominator or the verdict rule was executable, so the gate that blocks R2.4
could not be re-derived from the repository, and a reader could not check that
the published verdict follows from the published artefacts. This module is that
derivation: it consumes the artefacts already on disk and returns a verdict, and
:mod:`tests.test_basis_criteria` holds it against the numbers EXP-R2-191 and
EXP-R2-194 published.

**What is a pre-registration and what is a reading.** Only the parts marked as
such below were fixed before the numbers existed. Criterion A's cut (median
R > 1) was; its ``UNRESOLVED`` band was not -- that was the reading EXP-R2-191
gave the text mode when the median sat 4.6% below the cut with an interquartile
range straddling it, and it is codified here rather than re-decided each time.
Criterion B2's threshold (``d_model``) was pre-registered; the *layer window* it
is read over was not, and :func:`criterion_b2` therefore reports every reading
side by side instead of choosing one silently (EXP-R2-203).

**Criterion B1 is void and this module refuses to issue a verdict for it.** It
divides a dictionary's held-out reconstruction NMSE by the cross-checkpoint
ridge residual, which is small at the shallow layers, so the ratio is unstable
exactly where neither quantity is large -- and both text control cells fail it.
Under the refusal condition the pre-declaration named in advance, a criterion its
own control cannot pass is a specification defect and not a result.
:func:`criterion_b1_descriptive` computes its numbers, because they are worth
seeing, and returns ``VOID`` in place of a verdict.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

#: Layers dropped at each end before an interior statistic is taken.
#:
#: Pre-registered in EXP-R2-191 as "layers 0, 1, 30 and 31" on a 32-layer model,
#: for two reasons that generalise to any depth rather than to that one: the top
#: layer has no adjacent successor, so its adjacent-layer denominator does not
#: exist, and the outermost layers carry the degenerate denominators EXP-R2-175
#: refused to aggregate over -- the text side's layer-1 adjacent identity
#: residual alone reads 6,347.7. Written as a margin so the window is derived
#: from the model's depth instead of being a pair of literals that silently mean
#: something else on a 12-layer arm.
INTERIOR_MARGIN = 2

#: The alignment map Criterion A is read on. The four maps this stage fits are
#: nested and all affine, and the ridge is the largest of them, so its residual
#: is what "no linear map explains it" is measured with.
ALIGNMENT_MAP = "ridge"


def interior_layers(num_layers: int) -> list[int]:
    """The layer window an interior statistic is taken over.

    Raises rather than returning an empty window on a model too shallow to have
    one: a median over no layers is not a smaller version of this statistic.
    """

    if num_layers <= 2 * INTERIOR_MARGIN:
        raise ValueError(
            f"a {num_layers}-layer model has no interior window at margin "
            f"{INTERIOR_MARGIN}: the criterion is not defined on it"
        )
    return list(range(INTERIOR_MARGIN, num_layers - INTERIOR_MARGIN))


def _interior(values: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one value per layer")
    window = interior_layers(len(vector))
    selected = vector[window]
    if not np.isfinite(selected).all():
        raise ValueError(
            f"{name} carries a non-finite value inside the interior window "
            f"{window[0]}-{window[-1]}; a median over it would be meaningless"
        )
    return selected


# ------------------------------------------------------- reading the artefacts


def alignment_residuals(
    artefact: Mapping[str, Any], *, alignment_map: str = ALIGNMENT_MAP
) -> tuple[list[float], list[float]]:
    """Cross-checkpoint and adjacent-layer true-pairing residuals, per layer.

    Read from a ``25_model_diffing_baselines.py`` artefact. Returns two vectors
    of one value per layer; the adjacent-layer residual of the top layer does not
    exist and is returned as ``nan``, which the interior window excludes by
    construction rather than by a special case at the point of use.
    """

    layers = artefact["layers"]
    cross: list[float] = []
    adjacent: list[float] = []
    for entry in layers:
        cross.append(float(entry["cross"]["true"][alignment_map]["normalised_residual"]))
        neighbour = entry.get("adjacent")
        adjacent.append(
            float("nan")
            if not neighbour
            else float(neighbour["true"][alignment_map]["normalised_residual"])
        )
    return cross, adjacent


def held_out_nmse_per_layer(record: Mapping[str, Any]) -> list[float]:
    """A ``17_train_transcoder.py`` record's final held-out NMSE, per layer."""

    return [float(value) for value in record["held_out"]["nmse_per_layer"]]


# ------------------------------------------------------------------ criterion A


def criterion_a(
    cross: Sequence[float], adjacent: Sequence[float]
) -> dict[str, Any]:
    """When simple alignment is insufficient to explain a checkpoint difference.

    ``R(l)`` is the cross-checkpoint true-pairing ridge residual at layer ``l``
    divided by the adjacent-layer true-pairing ridge residual at the same layer,
    so it is stated in the unit ``25_model_diffing_baselines.py`` declared before
    it ran: what one layer of the reference model's own ordinary computation
    costs.

    Three verdicts. ``INSUFFICIENT`` when the interior median exceeds 1, which is
    the pre-registered cut. ``UNRESOLVED`` when the median is at or below 1 but
    the interquartile range straddles it -- the reading EXP-R2-191 gave the text
    mode, held there rather than promoted, because a median 4.6% below a cut with
    an IQR across it is not a verdict. ``SUFFICIENT`` otherwise.
    """

    numerator = _interior(cross, "cross-checkpoint residual")
    denominator = _interior(adjacent, "adjacent-layer residual")
    if (denominator <= 0).any():
        raise ValueError(
            "an adjacent-layer residual inside the interior window is not "
            "positive; R would not be a ratio of two costs"
        )
    ratio = numerator / denominator
    median = float(np.median(ratio))
    q1, q3 = (float(value) for value in np.quantile(ratio, [0.25, 0.75]))
    if median > 1.0:
        verdict = "INSUFFICIENT"
    elif q1 <= 1.0 <= q3:
        verdict = "UNRESOLVED"
    else:
        verdict = "SUFFICIENT"
    return {
        "criterion": "A",
        "statistic": "R(l) = cross-checkpoint true-pairing ridge residual / "
        "adjacent-layer true-pairing ridge residual, at the same layer",
        "interior_layers": [interior_layers(len(cross))[0], interior_layers(len(cross))[-1]],
        "ratio_per_layer": [float(value) for value in ratio],
        "median": median,
        "iqr": [q1, q3],
        "fraction_above_one": float((ratio > 1.0).mean()),
        "verdict": verdict,
        "note": (
            "INSUFFICIENT when the interior median exceeds 1: the residual the "
            "best linear map leaves costs more than one layer of the model's own "
            "ordinary computation. UNRESOLVED when the median is at or below 1 "
            "and the interquartile range straddles it -- that band is EXP-R2-191's "
            "reading and not part of its pre-registration, which fixed only the cut"
        ),
    }


# ------------------------------------------------------------- criterion B1 (void)


def criterion_b1_descriptive(
    nmse_per_layer: Sequence[float], cross: Sequence[float]
) -> dict[str, Any]:
    """B1's numbers, and no verdict, because B1 is void as a gate.

    ``D(l)`` is a dictionary's held-out reconstruction NMSE at layer ``l`` divided
    by that mode's cross-checkpoint true-pairing ridge residual at the same
    layer. The pre-declaration required median D < 0.5 and max D < 1 over the
    interior window, and named the refusal condition itself: if the text control
    fails, the criterion is a specification defect rather than a protein result.
    Both text cells failed (EXP-R2-191, EXP-R2-194), so it is void, and this
    returns ``VOID`` where a verdict would go.

    The numbers are still computed. They are descriptive -- the dictionary's own
    unexplained variance beside the difference it would describe -- and the
    reason the criterion is badly formed is visible in them: D divides by the
    cross-checkpoint residual, which is small at the shallow layers, so the ratio
    is unstable exactly where neither quantity is large.
    """

    numerator = _interior(nmse_per_layer, "held-out NMSE")
    denominator = _interior(cross, "cross-checkpoint residual")
    if (denominator <= 0).any():
        raise ValueError("a cross-checkpoint residual in the interior window is not positive")
    ratio = numerator / denominator
    return {
        "criterion": "B1",
        "statistic": "D(l) = held-out reconstruction NMSE / cross-checkpoint "
        "true-pairing ridge residual, at the same layer",
        "median": float(np.median(ratio)),
        "max": float(ratio.max()),
        "verdict": "VOID",
        "note": (
            "VOID as a gate on the refusal condition its own pre-declaration "
            "named: both text control cells fail it, which makes it a badly-formed "
            "statistic rather than a badly-chosen threshold. No cell may be judged "
            "by it and no threshold is applied here; the numbers are descriptive"
        ),
    }


# ------------------------------------------------------------------ criterion B2


def criterion_b2(
    live_per_layer: Sequence[int], d_model: int
) -> dict[str, Any]:
    """Whether a dictionary's live basis can span the space it decomposes.

    A dictionary whose live latents number fewer than ``d_model`` is not
    decomposing the space into more features than it has dimensions, which is the
    premise a sparse over-complete dictionary is read under.

    **Three readings, reported side by side, because the pre-declaration carries
    two and they can disagree.** Its preamble requires Criterion B to hold "at the
    layers a difference is reported on", which is a per-layer condition; the B2
    clause states the statistic as "mean live latents per layer", which is one
    number for the whole dictionary. A mean over 32 layers is not a statement
    about any layer -- a dictionary can average far above ``d_model`` while
    individual layers sit far below it -- so this returns the mean reading for
    continuity with EXP-R2-191 and EXP-R2-194's published figures, the per-layer
    reading over every layer, and the per-layer reading over the interior window
    Criterion A is read on.

    ``verdict`` is the **all-layers** per-layer reading, which is the literal form
    of "at the layers a difference is reported on" when a diff is reported on the
    whole depth. The interior reading is reported and is *not* the verdict:
    Criterion A's window exists because the outermost layers carry degenerate
    *adjacent-layer denominators*, and B2 has no denominator, so importing that
    window here would narrow a criterion for a reason that does not apply to it.
    A reader who restricts a diff to the interior window has
    ``per_layer_reading_interior`` to read it by. ``mean_reading_agrees`` says
    whether the historical cross-layer figure would have given the same answer.
    """

    counts = np.asarray(live_per_layer, dtype=np.int64)
    if counts.ndim != 1 or counts.size == 0:
        raise ValueError("live latent counts must be one non-negative integer per layer")
    if (counts < 0).any():
        raise ValueError("a live latent count cannot be negative")
    if d_model <= 0:
        raise ValueError("d_model must be positive")
    window = interior_layers(int(counts.size))
    interior = counts[window]
    mean = float(counts.mean())
    mean_verdict = "PASS" if mean > d_model else "FAIL"
    all_verdict = "PASS" if (counts > d_model).all() else "FAIL"
    interior_verdict = "PASS" if (interior > d_model).all() else "FAIL"
    failing = [int(layer) for layer in np.flatnonzero(counts <= d_model)]
    return {
        "criterion": "B2",
        "statistic": "live latents per layer against d_model",
        "d_model": int(d_model),
        "live_per_layer": [int(value) for value in counts],
        "mean_live_per_layer": mean,
        "min_live_per_layer": int(counts.min()),
        "argmin_layer": int(counts.argmin()),
        "failing_layers": failing,
        "n_failing_layers": len(failing),
        "interior_layers": [window[0], window[-1]],
        "mean_reading": mean_verdict,
        "per_layer_reading_all_layers": all_verdict,
        "per_layer_reading_interior": interior_verdict,
        "verdict": all_verdict,
        "mean_reading_agrees": mean_verdict == all_verdict,
        "note": (
            "verdict is the per-layer reading over EVERY layer, which is what the "
            "pre-declaration's preamble asks for -- Criterion B must hold 'at the "
            "layers a difference is reported on'. The interior reading is beside it "
            "for a diff restricted to layers 2-29, and is not the verdict because "
            "Criterion A's window exists for degenerate adjacent-layer denominators "
            "that B2 does not have. mean_reading is the cross-layer mean EXP-R2-191 "
            "and EXP-R2-194 published; it is kept for continuity and is not the "
            "verdict, because a mean over layers is not a statement about any layer"
        ),
    }


# ------------------------------------------------------------------- the gate


def basis_gate(cells: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """One verdict over a set of cells, from their individual B2 readings.

    Each cell is ``{"cell": name, "b2": <criterion_b2 output>}``. The gate passes
    only if every cell passes, which is what "in all four cells" means; the
    failing cells are named rather than counted, and the cells whose per-layer
    reading disagrees with their mean are named separately, because that
    disagreement is a statement about how the gate was read and not about the
    dictionaries.
    """

    entries = list(cells)
    if not entries:
        raise ValueError("a gate over no cells has no verdict")
    failing = [str(entry["cell"]) for entry in entries if entry["b2"]["verdict"] != "PASS"]
    disagreeing = [
        str(entry["cell"]) for entry in entries if not entry["b2"]["mean_reading_agrees"]
    ]
    return {
        "n_cells": len(entries),
        "failing_cells": failing,
        "cells_where_the_mean_reading_disagrees": disagreeing,
        "verdict": "PASS" if not failing else "FAIL",
        "note": (
            "B2 must hold in every cell; a gate that passes on three of four is a "
            "failed gate. B1 is void and issues no verdict, so it is not composed "
            "here. Criterion A classifies the alignment artefacts and is per mode "
            "rather than per cell, so it is reported beside this rather than in it"
        ),
    }
