"""Read every arm's prefix-matching census against its own collision null.

The induction census scores attention from a repeated token onto a *single*
positionally-aligned key (:func:`~.circuits.attention_alignment_scores`).  A head
that disambiguates on single-token identity cannot know which of several earlier
copies of the query token is the aligned one, so its mass is split across every
earlier occurrence -- and how many of those there are is a property of the
alphabet, not of the model.  Measured on this panel's own corpora and tokenizers
over 256 tokens (EXP-R2-155): GPT-2 1.387, ProtGPT2 0.068, ZymCTRL and
ProGen2-medium 8.081.  A fixed 0.10 cut therefore asks a residue-tokenised arm
for a much larger share of an attainable maximum than it asks a BPE text arm for,
and a head count read off it is in part a reading of the tokenizer.

The correction is a per-arm null rather than a per-arm threshold.  Each real
probe is paired with a **collision null probe**: the same length, the same token
multiset, the same query and key positions scored -- and the planted repeat
destroyed by a seeded permutation of the content positions that precede the
first query.  Because the permutation is of the probe's *own* tokens and touches
only what lies before every scored query, the number of earlier positions holding
a query's token is preserved exactly, position by position, and the only thing
removed is the alignment the census is trying to detect.  A head is counted when
its excess over its own null exceeds what the null can produce by chance on that
arm, which is a detection criterion of equal power at every alphabet size.

**Why not aggregate over antecedents.** The obvious repair -- score attention
summed over every earlier position holding the query token -- manufactures the
opposite bias, because the number of summed keys is itself the alphabet
statistic: on a synthetic probe a null head with no induction behaviour rises
from 0.011 to 0.058 against a fixed 0.10 cut on a residue arm.  Any such sum
needs a size-matched decoy, which is what
:func:`~.prediction_addressed.paa_specific_matched` pairs its sum with.  This
module takes the other route: the numerator is left exactly as the published
census computes it, and the *baseline* is matched instead.

**Scope.** This module reads attention patterns and nothing else.  It never
rebuilds an OV circuit, splits a block into sublayers or touches a position
table, so it admits every architecture whose pattern is addressable rather than
only those whose sublayers are commensurate -- which is what lets the byte-level
text arms enter, and they are the point: their collision rate sits in the residue
range while their modality is text, so they separate alphabet from modality
exactly as EXP-R2-129 did for the prediction-addressed census.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .arms import Arm
from .circuits import (
    INDUCTION_THRESHOLDS,
    RepeatProbe,
    attention_alignment_scores,
    head_census,
    layout_token_ids,
    n_head,
    prefix_ids,
)
from .prediction_addressed import PAA_ARCHITECTURES
from .statistics import MINIMUM_BOOTSTRAP_UNITS

SCHEMA_VERSION = "r2_transfer_collision_null_v1"

#: Architectures this census can be scheduled on.
#:
#: Imported from :data:`~.prediction_addressed.PAA_ARCHITECTURES` rather than
#: restated.  That set answers "can this module tap the attention pattern and
#: remove a key from it before the softmax"; this census only *reads* the
#: pattern, through the model's own ``output_attentions``, so anything that set
#: admits this one admits and a divergence between the two would be a defect
#: rather than a distinction.  It is deliberately **not**
#: ``circuits._CIRCUIT_ARCHITECTURES``, which answers the stronger question of
#: whether a per-head OV circuit can be rebuilt; that question is not asked here
#: and answering it would exclude the byte-level text control this census exists
#: to admit.
CENSUS_ARCHITECTURES = PAA_ARCHITECTURES

#: Family-wise levels the null cut is read at, from permissive to strict.
#:
#: A count above a cut is a count above a threshold, so Appendix B rule 8 applies
#: and the level is swept rather than chosen: the reading is the *ordering* of
#: the arms, and an ordering that moves across this sweep is not a result.
DEFAULT_ALPHAS: tuple[float, ...] = (0.5, 0.9, 0.95, 0.99)

#: Percentile interval reported on every resampled count.
BOOTSTRAP_INTERVAL = (2.5, 97.5)

#: Fractions of an arm's own identity ceiling the *raw* census is re-read at.
#:
#: The companion reading, and the most direct answer to the objection: the
#: published census counts a head above a raw 0.10, and 0.10 is a different
#: fraction of what a single-token-identity head can reach on a 20-residue
#: alphabet than on a 50,257-piece one.  Dividing by the ceiling measured on the
#: arm's own probes puts the cut at the same fraction of the attainable maximum
#: everywhere; 0.10 raw against gpt2-large's measured ceiling is about 0.11 here,
#: so the second rung is the published cut, corrected.
#:
#: This reading assumes the mechanism the objection assumes -- a head that
#: disambiguates on single-token identity and splits evenly across every earlier
#: copy -- and is therefore a diagnostic beside the null-calibrated count, never
#: a replacement for it.  A head that uses multi-token context is not bounded by
#: this ceiling at all.
CEILING_FRACTIONS: tuple[float, ...] = (0.05, 0.11, 0.20, 0.40)

#: Spread below which a head's resampled excess is treated as having none.
#:
#: Attention is a softmax output in [0, 1] and every quantity here is a mean of
#: such values, so this is a numerical-zero test rather than a tuning parameter.
DEGENERATE_SPREAD = 1e-12

#: Fraction of a probe's own chance attention level below which an excess counts
#: as numerically zero rather than small.
#:
#: The companion to :data:`DEGENERATE_SPREAD`, and it has to be *relative* where
#: that one is absolute. A saturated head -- one whose softmax has collapsed onto
#: a single key, of which every arm has several -- can carry an excess of 1e-12
#: to 1e-8 with a bootstrap spread that underflows to zero, and an absolute test
#: at the same value as the spread's splits that single dead quantity across the
#: two tests: the spread reads as zero, the excess reads as non-zero, and the
#: measurement refuses an arm over a head nine orders of magnitude below chance.
#: Measured on ProGen2-small, whose uniform baseline is 0.0107: sixteen offset-two
#: heads sit below 1e-6 of it, the smallest at 1.45e-12.
#:
#: One part in a million of chance attention is not an effect-size threshold --
#: nothing this programme measures lives within six orders of magnitude of it --
#: and it is deliberately far below any value that could change a verdict. The
#: maximum excess among the heads it sets aside is reported with every census so
#: that claim is checkable rather than asserted.
NEGLIGIBLE_EXCESS_FRACTION = 1e-6

#: Content positions a probe must offer before its permutation means anything.
#:
#: Below this the permutation has too few arrangements for the null to be a null:
#: at four positions one draw in twenty-four is the identity.  A refusal, not a
#: fallback -- a probe too short to permute is a probe that cannot be scored
#: against its own null.
MINIMUM_PERMUTABLE_POSITIONS = 16


def _finite(value: float, label: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def census_architecture(arm: Arm) -> str:
    """The arm's architecture, refused unless its pattern is readable here."""

    architecture = arm.spec.architecture
    if architecture not in CENSUS_ARCHITECTURES:
        raise TypeError(
            f"{arm.name}: the collision-null census reads per-layer attention "
            f"patterns and has no declared path for {architecture!r}; declared: "
            f"{sorted(CENSUS_ARCHITECTURES)}"
        )
    return architecture


# ------------------------------------------------------------------ null probes


def structural_token_ids(arm: Arm, *, ec_label: str | None = None) -> frozenset[int]:
    """Token ids a permutation must leave where they are.

    Everything an arm's *rendering* puts around its content: the prompt prefix
    :func:`~.circuits.prefix_ids` builds, the tokenizer's own boundary markers,
    and for a conditioned rendering the tags that close the content span.  Moving
    one of these into the middle of a probe would be a different perturbation
    from the one this null makes, and it would be a different one *per arm*,
    which is precisely what the null exists to avoid.
    """

    ids = {int(token) for token in prefix_ids(arm, ec_label=ec_label)}
    for special in (
        arm.tokenizer.bos_token_id,
        arm.tokenizer.eos_token_id,
        arm.tokenizer.pad_token_id,
    ):
        if special is not None:
            ids.add(int(special))
    if arm.spec.input_format == "ec_conditioned":
        vocabulary = arm.tokenizer.get_vocab()
        for marker in ("<sep>", "<start>", "<end>"):
            if marker not in vocabulary:
                raise ValueError(f"{arm.name}: tokenizer lacks the {marker!r} marker")
            ids.add(int(vocabulary[marker]))
    return frozenset(ids)


def permutable_positions(
    arm: Arm, probe: RepeatProbe, structural: frozenset[int]
) -> np.ndarray:
    """Content positions strictly before the first scored query.

    The restriction to the earlier copy is what makes the match *exact* rather
    than close, and it was put here by a failing test.  A permutation of the
    whole probe moves tokens across the query positions, so the number of earlier
    positions holding a query's token -- the collision multiplicity the whole
    objection is about -- changes by a tenth, in the direction that flatters the
    null.  Permuting only what lies before every query cannot do that: a query's
    token is untouched, every permuted position is earlier than every query
    whatever order it ends up in, and the positions between the two are unchanged,
    so the multiplicity at every scored query is preserved token for token.  The
    query side of the probe is left byte-identical as well, which is a stronger
    match than the design asked for and free.

    Layout tokens are excluded for the reason :func:`~.circuits.layout_token_ids`
    records: one arm's rendering hard-wraps at sixty residues, and permuting a
    line break would perturb that arm's layout while perturbing no other arm's,
    so the null would stop being matched exactly where the panel is least
    symmetric.

    Every scored key's predecessor must fall inside the permuted region, because
    that predecessor holding the query's token *is* the alignment being removed.
    A probe that does not satisfy it is refused rather than half-nulled.
    """

    ids = np.asarray(probe.input_ids, dtype=np.int64)
    horizon = min(probe.query_positions)
    blocked = set(structural) | layout_token_ids(arm, {int(token) for token in np.unique(ids)})
    free = np.asarray(
        [
            index
            for index, token in enumerate(ids[:horizon])
            if int(token) not in blocked
        ],
        dtype=np.int64,
    )
    if free.size < MINIMUM_PERMUTABLE_POSITIONS:
        raise ValueError(
            f"{arm.name}: a probe of {ids.size} tokens offers only {free.size} "
            f"permutable content positions before its first scored query, below the "
            f"{MINIMUM_PERMUTABLE_POSITIONS} a collision null needs to be a null"
        )
    missing = sorted({int(key) - 1 for key in probe.key_positions} - set(free.tolist()))
    if missing:
        raise ValueError(
            f"{arm.name}: positions {missing[:8]} precede a scored key but are not "
            "permutable, so the alignment at those keys would survive into the null"
        )
    return free


def collision_null_probe(
    arm: Arm,
    probe: RepeatProbe,
    rng: np.random.Generator,
    *,
    structural: frozenset[int],
) -> RepeatProbe:
    """One probe with its planted repeat permuted away and everything else held."""

    free = permutable_positions(arm, probe, structural)
    ids = list(int(token) for token in probe.input_ids)
    drawn = [ids[position] for position in free[rng.permutation(free.size)]]
    for position, token in zip(free, drawn):
        ids[position] = token
    return RepeatProbe(
        kind="collision_null",
        input_ids=tuple(ids),
        query_positions=probe.query_positions,
        key_positions=probe.key_positions,
        coverage=probe.coverage,
        repeat_symbols=probe.repeat_symbols,
        record_index=probe.record_index,
    )


def collision_null_probes(
    arm: Arm,
    probes: Sequence[RepeatProbe],
    *,
    seed: int,
    ec_label: str | None = None,
) -> list[RepeatProbe]:
    """A collision null for every probe, one seeded permutation each."""

    if not probes:
        raise ValueError("no probes were supplied")
    structural = structural_token_ids(arm, ec_label=ec_label)
    rng = np.random.default_rng(seed)
    return [collision_null_probe(arm, probe, rng, structural=structural) for probe in probes]


# ------------------------------------------------------------------- the checks


def aligned_key_fraction(probes: Sequence[RepeatProbe]) -> float:
    """Fraction of scored pairs whose key really does follow a copy of the query.

    One for an exact repeat by construction, and at the arm's chance collision
    rate once the repeat has been permuted away.  This is the single number that
    says whether a null is a null, and it is checked rather than assumed: a
    permutation that left the repeat in place would produce a null the real
    probes cannot beat, and every head count read against it would be zero for a
    reason that has nothing to do with the model.
    """

    hits = 0
    total = 0
    for probe in probes:
        ids = probe.input_ids
        for query, key in zip(probe.query_positions, probe.key_positions):
            total += 1
            hits += int(ids[key - 1] == ids[query])
    if total < 1:
        raise ValueError("probe set scores no positions")
    return hits / total


def antecedent_statistics(probes: Sequence[RepeatProbe]) -> dict[str, Any]:
    """How many earlier copies of the query token each scored position has.

    ``same_token_antecedents`` is the multiplicity the vocabulary-collision
    objection is about, measured on the probes that were actually scored rather
    than on a corpus at some other length.  ``identity_ceiling`` is what it costs:
    the mean of ``1 / multiplicity``, which is the attention a head that
    disambiguates on single-token identity alone and splits evenly across every
    earlier copy would place on the one aligned key.  It is an upper bound on
    such a head and says nothing about a head that uses multi-token context, so
    it is reported as a diagnostic beside the null-calibrated count and never in
    place of it.
    """

    multiplicities: list[int] = []
    for probe in probes:
        ids = np.asarray(probe.input_ids, dtype=np.int64)
        for query in probe.query_positions:
            multiplicities.append(int((ids[:query] == ids[query]).sum()))
    counts = np.asarray(multiplicities, dtype=np.float64)
    if counts.size < 1:
        raise ValueError("probe set scores no positions")
    reachable = counts > 0
    return {
        "scored_positions": int(counts.size),
        "same_token_antecedents_mean": _finite(float(counts.mean()), "antecedent mean"),
        "positions_with_no_antecedent": int((~reachable).sum()),
        "identity_ceiling": _finite(
            float(np.where(reachable, 1.0 / np.maximum(counts, 1.0), 0.0).mean()),
            "identity ceiling",
        ),
    }


def verify_null_match(
    arm: Arm, probes: Sequence[RepeatProbe], null: Sequence[RepeatProbe]
) -> dict[str, Any]:
    """Prove, pair by pair, that the null matches its probe and lost the repeat.

    Length, token multiset and scored positions are checked as identities and
    raise on any difference.  Matching the null to the real probes is the whole
    design, so it is verified rather than argued from how the null was built --
    the builder is one edit away from being wrong, and a mismatched null would
    produce a plausible number rather than a crash.
    """

    if len(probes) != len(null):
        raise ValueError(f"{arm.name}: {len(null)} null probes for {len(probes)} real probes")
    displaced = 0
    positions = 0
    for index, (real, other) in enumerate(zip(probes, null)):
        if len(real.input_ids) != len(other.input_ids):
            raise ValueError(
                f"{arm.name}: null probe {index} is {len(other.input_ids)} tokens "
                f"against the real probe's {len(real.input_ids)}"
            )
        if Counter(real.input_ids) != Counter(other.input_ids):
            raise ValueError(
                f"{arm.name}: null probe {index} does not carry its probe's token multiset"
            )
        if (
            real.query_positions != other.query_positions
            or real.key_positions != other.key_positions
        ):
            raise ValueError(f"{arm.name}: null probe {index} scores different positions")
        positions += len(real.input_ids)
        displaced += sum(
            int(left != right) for left, right in zip(real.input_ids, other.input_ids)
        )
    return {
        "n_probes": len(null),
        "length_matched": True,
        "token_multiset_matched": True,
        "scored_positions_matched": True,
        "displaced_token_fraction": _finite(displaced / positions, "displaced fraction"),
        "aligned_key_fraction_real": _finite(
            aligned_key_fraction(probes), "real aligned-key fraction"
        ),
        "aligned_key_fraction_null": _finite(
            aligned_key_fraction(null), "null aligned-key fraction"
        ),
        "antecedents_real": antecedent_statistics(probes),
        "antecedents_null": antecedent_statistics(null),
    }


# ----------------------------------------------------------------- the estimate


def _weighted_means(
    sums: np.ndarray, counts: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """``sum(w * probe sums) / sum(w * probe counts)`` for a stack of weightings.

    ``weights`` is ``(draws, probes)`` and the result is ``(draws, layer, head)``.
    A multinomial weight vector is exactly a bootstrap resample of the probes, so
    this is the published position-weighted estimator recomputed on a resample
    rather than a different, probe-weighted one.
    """

    n_probes, layers, heads = sums.shape
    denominator = weights @ counts.astype(np.float64)
    if (denominator <= 0).any():
        raise ValueError("a resample scored no positions")
    flat = weights @ sums.reshape(n_probes, layers * heads)
    return (flat / denominator[:, None]).reshape(-1, layers, heads)


def collision_null_census(
    arm: Arm,
    probes: Sequence[RepeatProbe],
    *,
    seed: int,
    batch_size: int,
    n_bootstrap: int = 2000,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    thresholds: Sequence[float] = INDUCTION_THRESHOLDS,
    ec_label: str | None = None,
) -> dict[str, Any]:
    """Count this arm's induction heads against its own collision null.

    Two independent null draws are scored, not one.  The first is the baseline
    the real probes are read against; the second exists only to say how large a
    difference between two *nulls* this arm's heads produce by sampling alone, and
    it is the cut the count is taken above.  Both nulls are permutations of the
    same real probes, so the null-versus-null difference carries the same probe
    count, the same lengths and the same compositions as the real-versus-null
    difference and is a like-for-like noise floor rather than a smaller one
    rescaled.

    The ``offset_two`` decoy is carried through the identical arithmetic.  It is
    the same probes, the same nulls and the same cut applied to a key one position
    off the aligned one, so a procedure that would count heads on any strong
    positional structure shows it there.  A count that is large on
    ``prefix_matching`` and large on ``offset_two`` is not an induction count.
    """

    census_architecture(arm)
    if len(probes) < MINIMUM_BOOTSTRAP_UNITS:
        raise ValueError(
            f"{arm.name}: {len(probes)} probes is below the {MINIMUM_BOOTSTRAP_UNITS} "
            "resampling units an interval on a count needs to mean what it says"
        )
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be at least 100")
    if not alphas or any(not 0.0 < float(alpha) < 1.0 for alpha in alphas):
        raise ValueError("every family-wise level must lie strictly inside (0, 1)")

    null_a = collision_null_probes(arm, probes, seed=seed, ec_label=ec_label)
    null_b = collision_null_probes(arm, probes, seed=seed + 1, ec_label=ec_label)
    if [probe.input_ids for probe in null_a] == [probe.input_ids for probe in null_b]:
        raise RuntimeError(
            f"{arm.name}: the two null draws are identical, so the noise floor "
            "would be exactly zero and every head would clear it"
        )
    matching = {
        "null_a": verify_null_match(arm, probes, null_a),
        "null_b": verify_null_match(arm, probes, null_b),
    }

    scored = {
        "real": attention_alignment_scores(arm, probes, batch_size=batch_size, per_probe=True),
        "null_a": attention_alignment_scores(
            arm, null_a, batch_size=batch_size, per_probe=True
        ),
        "null_b": attention_alignment_scores(
            arm, null_b, batch_size=batch_size, per_probe=True
        ),
    }
    counts = scored["real"]["per_probe_counts"]
    for label in ("null_a", "null_b"):
        if not np.array_equal(scored[label]["per_probe_counts"], counts):
            raise RuntimeError(f"{arm.name}: {label} scores a different position count")

    n_probes = len(probes)
    rng = np.random.default_rng(seed + 2)
    weights = np.vstack(
        (
            np.ones((1, n_probes), dtype=np.float64),
            rng.multinomial(
                n_probes, np.full(n_probes, 1.0 / n_probes), size=n_bootstrap
            ).astype(np.float64),
        )
    )

    statistics: dict[str, Any] = {}
    resampled_real: dict[str, np.ndarray] = {}
    for statistic in ("prefix_matching", "offset_two"):
        real = _weighted_means(scored["real"]["per_probe_sums"][statistic], counts, weights)
        resampled_real[statistic] = real
        first = _weighted_means(scored["null_a"]["per_probe_sums"][statistic], counts, weights)
        second = _weighted_means(scored["null_b"]["per_probe_sums"][statistic], counts, weights)
        excess = real - first
        noise = first - second
        # Row zero is the unweighted sample; rows one and beyond are the resample.
        point = excess[0]
        resampled = excess[1:]
        family_wise = noise[1:].reshape(n_bootstrap, -1).max(axis=1)
        cuts: dict[str, Any] = {}
        for alpha in alphas:
            cut = float(np.quantile(family_wise, float(alpha)))
            above = resampled > cut
            low, high = np.percentile(
                above.reshape(n_bootstrap, -1).sum(axis=1), BOOTSTRAP_INTERVAL
            )
            n_heads = int(point.size)
            cuts[f"{float(alpha):.2f}"] = {
                "family_wise_level": float(alpha),
                "null_cut": _finite(cut, "null cut"),
                "n_above_null": int((point > cut).sum()),
                "n_above_null_ci": [float(low), float(high)],
                "fraction_above_null": _finite(
                    float((point > cut).sum()) / n_heads, "fraction above null"
                ),
                "fraction_above_null_ci": [float(low) / n_heads, float(high) / n_heads],
            }
        # The studentised family-wise maximum, which is the cut the raw one above
        # should have been. The raw cut takes its floor from the null-versus-null
        # difference, and that difference is more strongly paired than the one it
        # gates -- both nulls are permutations of the same probes and share the
        # query side byte-identically -- so it estimates a variance the gated
        # difference does not have. Studentising removes the mismatch because the
        # mismatch is a variance ratio. The floor is then taken from the CENTRED
        # bootstrap of the excess itself, which carries not only the right scale
        # but the right cross-head correlation structure, because it is the same
        # statistic. Centring removes each head's own effect, so a head with a
        # real alignment effect contributes its noise and not its signal.
        standard_error = resampled.std(axis=0, ddof=1)
        if not np.isfinite(standard_error).all():
            raise RuntimeError(f"{arm.name}: {statistic} produced a non-finite standard error")
        # A head can have exactly no spread, and the right answer depends on what
        # its excess is. A purely positional head -- one whose attention at the
        # scored positions does not depend on the tokens at all -- scores
        # identically on the real probe and on its permutation, so its excess is
        # zero on every probe and its spread is zero too. Its verdict is not in
        # doubt: it has no alignment effect, it cannot clear a positive cut, and
        # it contributes no noise to the family-wise maximum. Refusing the whole
        # arm for it would be a fail-fast in the wrong place. A head with no
        # spread and a NON-zero excess is the case where the answer does matter
        # and cannot be formed, so that one still refuses.
        degenerate = standard_error <= DEGENERATE_SPREAD
        negligible = NEGLIGIBLE_EXCESS_FRACTION * float(scored["real"]["uniform_baseline"])
        undecidable = degenerate & (np.abs(point) > negligible)
        if undecidable.any():
            raise RuntimeError(
                f"{arm.name}: {statistic} has {int(undecidable.sum())} head(s) whose "
                f"resampled excess has no spread but whose excess exceeds {negligible:.3e}, "
                "so a studentised statistic cannot be formed where its value could "
                "decide the count"
            )
        safe = np.where(degenerate, 1.0, standard_error)
        t_point = np.where(degenerate, 0.0, point / safe)
        centred = np.where(degenerate, -np.inf, (resampled - point) / safe)
        family_wise_t = centred.reshape(n_bootstrap, -1).max(axis=1)
        if not np.isfinite(family_wise_t).all():
            raise RuntimeError(
                f"{arm.name}: {statistic} has no head with a measurable spread, so "
                "there is no noise floor to read a count against"
            )
        resampled_t = np.where(degenerate, 0.0, resampled / safe)
        studentised: dict[str, Any] = {}
        for alpha in alphas:
            cut = float(np.quantile(family_wise_t, float(alpha)))
            low, high = np.percentile(
                (resampled_t > cut).reshape(n_bootstrap, -1).sum(axis=1), BOOTSTRAP_INTERVAL
            )
            n_heads = int(point.size)
            studentised[f"{float(alpha):.2f}"] = {
                "family_wise_level": float(alpha),
                "studentised_cut": _finite(cut, "studentised cut"),
                "n_above": int((t_point > cut).sum()),
                "n_above_ci": [float(low), float(high)],
                "fraction_above": _finite(
                    float((t_point > cut).sum()) / n_heads, "studentised fraction"
                ),
                "fraction_above_ci": [float(low) / n_heads, float(high) / n_heads],
            }
        studentised["degenerate_heads"] = int(degenerate.sum())
        # The evidence that setting those heads aside is hygiene and not a thumb
        # on the scale: how large the largest of them actually was.
        studentised["degenerate_max_abs_excess"] = _finite(
            float(np.abs(point[degenerate]).max()) if degenerate.any() else 0.0,
            "degenerate excess",
        )
        studentised["negligible_excess_bound"] = _finite(negligible, "negligible bound")

        statistics[statistic] = {
            "n_heads": int(point.size),
            "excess_max": _finite(float(point.max()), "peak excess"),
            "studentised": studentised,
            "excess_standard_error_per_head": standard_error.tolist(),
            "excess_t_per_head": t_point.tolist(),
            "excess_positive_total": _finite(
                float(point.clip(min=0.0).sum()), "positive excess total"
            ),
            "null_noise_family_wise_median": _finite(
                float(np.median(family_wise)), "family-wise noise median"
            ),
            "cuts": cuts,
            "excess_per_head": point.tolist(),
            "null_noise_per_head": noise[0].tolist(),
        }

    ceiling = float(matching["null_a"]["antecedents_real"]["identity_ceiling"])
    if ceiling <= 0.0:
        raise ValueError(f"{arm.name}: identity ceiling is not positive")
    raw = resampled_real["prefix_matching"]
    ceiling_counts: dict[str, Any] = {}
    for fraction in CEILING_FRACTIONS:
        cut = float(fraction) * ceiling
        above = raw[1:] >= cut
        low, high = np.percentile(
            above.reshape(n_bootstrap, -1).sum(axis=1), BOOTSTRAP_INTERVAL
        )
        ceiling_counts[f"{float(fraction):.2f}"] = {
            "ceiling_fraction": float(fraction),
            "raw_cut": _finite(cut, "ceiling-normalised cut"),
            "n_above": int((raw[0] >= cut).sum()),
            "n_above_ci": [float(low), float(high)],
        }

    real_scores = scored["real"]["scores"]
    return {
        "schema_version": SCHEMA_VERSION,
        "probe_kind": probes[0].kind,
        "n_probes": n_probes,
        "scored_query_positions": int(scored["real"]["scored_query_positions"]),
        "uniform_baseline": _finite(
            float(scored["real"]["uniform_baseline"]), "uniform baseline"
        ),
        "n_heads": int(n_head(arm) * arm.n_layer),
        "seeds": {"null_a": seed, "null_b": seed + 1, "bootstrap": seed + 2},
        "n_bootstrap": int(n_bootstrap),
        "bootstrap_interval_percentiles": list(BOOTSTRAP_INTERVAL),
        "matching": matching,
        "statistics": statistics,
        "fixed_threshold_census": {
            "real": head_census(real_scores["prefix_matching"], thresholds=thresholds),
            "null_a": head_census(
                scored["null_a"]["scores"]["prefix_matching"], thresholds=thresholds
            ),
        },
        "ceiling_normalised_census": {
            "identity_ceiling": _finite(ceiling, "identity ceiling"),
            "assumes": (
                "a head that disambiguates on single-token identity and splits "
                "evenly across every earlier copy of the query token; a head using "
                "multi-token context is not bounded by this ceiling"
            ),
            "counts": ceiling_counts,
        },
        "mean_scores": {
            label: {
                name: _finite(float(matrix.mean()), f"{label} {name} mean")
                for name, matrix in scored[label]["scores"].items()
            }
            for label in ("real", "null_a", "null_b")
        },
    }


def census_row(payload: Mapping[str, Any], *, alpha: float) -> dict[str, Any]:
    """The one line per arm a panel reading is taken from, at one level.

    Derived here rather than in the stage so that the per-arm artefact, the panel
    summary and any later re-reading cannot compute the headline three slightly
    different ways.
    """

    label = f"{float(alpha):.2f}"
    prefix = payload["statistics"]["prefix_matching"]
    decoy = payload["statistics"]["offset_two"]
    if label not in prefix["cuts"]:
        raise KeyError(f"level {label} is not among {sorted(prefix['cuts'])}")
    cut = prefix["cuts"][label]
    student = prefix["studentised"][label]
    return {
        "studentised_cut": student["studentised_cut"],
        "n_above_studentised": student["n_above"],
        "n_above_studentised_ci": student["n_above_ci"],
        "fraction_above_studentised": student["fraction_above"],
        "fraction_above_studentised_ci": student["fraction_above_ci"],
        "offset_two_decoy_n_above_studentised": decoy["studentised"][label]["n_above"],
        "n_heads": int(prefix["n_heads"]),
        "family_wise_level": float(alpha),
        "null_cut": cut["null_cut"],
        "n_above_null": cut["n_above_null"],
        "n_above_null_ci": cut["n_above_null_ci"],
        "fraction_above_null": cut["fraction_above_null"],
        "fraction_above_null_ci": cut["fraction_above_null_ci"],
        "offset_two_decoy_n_above_null": decoy["cuts"][label]["n_above_null"],
        "excess_max": prefix["excess_max"],
        "identity_ceiling": payload["matching"]["null_a"]["antecedents_real"][
            "identity_ceiling"
        ],
        "same_token_antecedents_mean": payload["matching"]["null_a"]["antecedents_real"][
            "same_token_antecedents_mean"
        ],
        "aligned_key_fraction_null": payload["matching"]["null_a"][
            "aligned_key_fraction_null"
        ],
        "n_above_fixed_0.10": int(
            payload["fixed_threshold_census"]["real"]["count_above_threshold"]["0.10"]
        ),
        "n_above_ceiling_0.11": int(
            payload["ceiling_normalised_census"]["counts"]["0.11"]["n_above"]
        ),
        "n_above_ceiling_0.11_ci": payload["ceiling_normalised_census"]["counts"]["0.11"][
            "n_above_ci"
        ],
        "uniform_baseline": payload["uniform_baseline"],
    }
