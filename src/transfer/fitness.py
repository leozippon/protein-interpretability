"""Zero-shot DMS fitness scoring, and the free baseline it has to be read against.

**Why this module exists.** ProGenMech (arXiv:2606.16044) reports that a sparse
circuit over a cross-layer transcoder recovers "~95%" and "~80%" of ProGen3's
zero-shot fitness performance. Both figures are ratios of Spearman correlations
against a base of **0.29**, and neither the base nor the ratio carries a floor.
Standing rule 28 requires a selector to be scored against the trivial baseline
available from its own coordinates; the analogue for a fitness predictor is the
score computable from the mutation string alone, before any model exists. This
module supplies that baseline (BLOSUM62), the cohort machinery for the eight
assays their released artefacts name, and the scoring convention all arms share.

**One fact that removes a candidate explanation, recorded because it is easy to
get wrong.** A ProteinGym substitution assay has a single wild type, so the
mutant-minus-wildtype log-likelihood differs from the raw mutant log-likelihood
by a constant. Spearman is invariant to it: the two conventions give *bit
identical* correlations, measured (0.5021144742520806 both ways on 400 single
mutants of GRB2_HUMAN_Faure_2021). Sequence length is likewise constant within an
assay, so summing and averaging over positions also rank-tie. Neither choice can
explain a disagreement between two reported fitness numbers on the same assay,
and this module therefore does not expose them as options.

BLOSUM62 is free of the *model and the method*, which is what rule 28 asks for.
It is not free of biology -- it is estimated from aligned protein blocks -- and
a claim built on it must say so rather than calling it uninformed.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .probes import PROTEINGYM_ROOT

#: The eight ProteinGym substitution assays ProGenMech evaluates on. Not taken
#: from the paper's prose: read from the directory names inside their released
#: ``ProGenMechData/functions.tar.gz``, which holds one circuit-discovery output
#: per assay, so this list is what their own artefacts were produced over.
PROGENMECH_ASSAYS: tuple[str, ...] = (
    "A4_HUMAN_Seuma_2022",
    "CAPSD_AAV2S_Sinai_2021",
    "F7YBW8_MESOW_Ding_2023",
    "GFP_AEQVI_Sarkisyan_2016",
    "GRB2_HUMAN_Faure_2021",
    "RASK_HUMAN_Weng_2022_abundance",
    "SPG1_STRSG_Olson_2014",
    "YAP1_HUMAN_Araya_2012",
)

#: Their ``function_circuit/discover_circuits.py`` defaults, so a run of this
#: module can be set to their sampling condition by name rather than by memory.
PROGENMECH_TEST_SEQUENCES = 1000

#: Their training-split size, removed from the pool before the test draw.
PROGENMECH_TRAIN_SEQUENCES = 256

#: ``seed = 42 + fold`` for five folds (``discover_circuits.py:200-202``).
PROGENMECH_FOLD_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)

_AA = "ARNDCQEGHILKMFPSTWYV"
_BLOSUM62_ROWS = """
4 -1 -2 -2 0 -1 -1 0 -2 -1 -1 -1 -1 -2 -1 1 0 -3 -2 0
-1 5 0 -2 -3 1 0 -2 0 -3 -2 2 -1 -3 -2 -1 -1 -3 -2 -3
-2 0 6 1 -3 0 0 0 1 -3 -3 0 -2 -3 -2 1 0 -4 -2 -3
-2 -2 1 6 -3 0 2 -1 -1 -3 -4 -1 -3 -3 -1 0 -1 -4 -3 -3
0 -3 -3 -3 9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
-1 1 0 0 -3 5 2 -2 0 -3 -2 1 0 -3 -1 0 -1 -2 -1 -2
-1 0 0 2 -4 2 5 -2 0 -3 -3 1 -2 -3 -1 0 -1 -3 -2 -2
0 -2 0 -1 -3 -2 -2 6 -2 -4 -4 -2 -3 -3 -2 0 -2 -2 -3 -3
-2 0 1 -1 -3 0 0 -2 8 -3 -3 -1 -2 -1 -2 -1 -2 -2 2 -3
-1 -3 -3 -3 -1 -3 -3 -4 -3 4 2 -3 1 0 -3 -2 -1 -3 -1 3
-1 -2 -3 -4 -1 -2 -3 -4 -3 2 4 -2 2 0 -3 -2 -1 -2 -1 1
-1 2 0 -1 -3 1 1 -2 -1 -3 -2 5 -1 -3 -1 0 -1 -3 -2 -2
-1 -1 -2 -3 -1 0 -2 -3 -2 1 2 -1 5 0 -2 -1 -1 -1 -1 1
-2 -3 -3 -3 -2 -3 -3 -3 -1 0 0 -3 0 6 -4 -2 -2 1 3 -1
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4 7 -1 -1 -4 -3 -2
1 -1 1 0 -1 0 0 0 -1 -2 -2 0 -1 -2 -1 4 1 -3 -2 -2
0 -1 0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1 1 5 -2 -2 0
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1 1 -4 -3 -2 11 2 -3
-2 -2 -2 -3 -2 -1 -2 -3 2 -1 -1 -2 -1 3 -3 -2 -2 2 7 -1
0 -3 -3 -3 -1 -2 -2 -3 -3 3 1 -2 1 -1 -2 -2 0 -3 -1 4
"""

#: BLOSUM62, as ``(wild_type, mutant) -> score``. Written out rather than pulled
#: from Biopython because the ``ct`` environment does not carry it and a
#: measurement dependency that has to be installed on the pod is not one this
#: repository can take (the pods are offline).
BLOSUM62: dict[tuple[str, str], int] = {
    (_AA[i], _AA[j]): int(value)
    for i, row in enumerate(_BLOSUM62_ROWS.strip().splitlines())
    for j, value in enumerate(row.split())
}


@dataclass(frozen=True)
class Assay:
    """One DMS assay, drawn to a declared size under a declared seed."""

    name: str
    wildtype: str
    mutants: list[str]
    sequences: list[str]
    scores: np.ndarray
    n_eligible: int
    n_multi: int
    seed: int
    sampling: str

    @property
    def blosum(self) -> np.ndarray:
        """Summed BLOSUM62 score of each variant's substitutions."""

        return np.array(
            [sum(BLOSUM62[(t[0], t[-1])] for t in m.split(":")) for m in self.mutants],
            dtype=np.float64,
        )

    def record(self) -> dict:
        return {
            "assay": self.name,
            "wildtype_length": len(self.wildtype),
            "n_variants": len(self.sequences),
            "n_eligible": self.n_eligible,
            "n_multi_mutant": self.n_multi,
            "seed": self.seed,
            "sampling": self.sampling,
        }


def _revert(sequence: str, mutant: str) -> str:
    """The wild type implied by one variant, reverting every substitution in it."""

    out = list(sequence)
    for token in mutant.split(":"):
        wild, position, mutated = token[0], int(token[1:-1]), token[-1]
        if out[position - 1] != mutated:
            raise ValueError(
                f"{mutant}: variant carries {out[position - 1]!r} at position "
                f"{position}, but the mutation string says {mutated!r}"
            )
        out[position - 1] = wild
    return "".join(out)


def load_assay(
    name: str,
    *,
    n: int,
    seed: int,
    include_multi: bool = True,
    stratify_by_score_bin: bool = False,
    train_holdout: int = 0,
    directory: Path | None = None,
) -> Assay:
    """Draw ``n`` variants of one ProteinGym substitution assay under ``seed``.

    The draw is a seeded permutation of the eligible rows, never a prefix: a
    ProteinGym CSV is ordered by position, so the first ``n`` rows are the
    protein's N-terminus rather than a sample of it (Appendix B rule 1).

    ``stratify_by_score_bin`` reproduces **ProGenMech's sampling design**: an
    equal number of variants from each ``DMS_score_bin``, drawn from a pool with
    a ``train_holdout`` split -- itself equally stratified -- removed first
    (``function_circuit/prepare_data.py:7-36``). This is not cosmetic. Their
    reported ProGen3 base of 0.29 is measured on a class-balanced resample,
    where ProteinGym's own benchmark records 0.497 for the identical checkpoint
    on six of the same assays; a recovery ratio is only readable against the
    base its own sampling produced, so both draws have to be available here.

    **The design is matched; the exact rows are not.** Their draw goes through
    ``pandas.DataFrame.sample(random_state=...)``, and pandas is deliberately
    not a dependency of this package -- the measurement harness has to run in an
    offline pod on torch, transformers, numpy and scipy alone. Reproducing their
    row selection bit-for-bit would buy a false precision anyway: the statistic
    moves by 0.013-0.035 across their own five fold seeds, so what reproduces is
    the design and its spread, not a single draw.

    Every retained variant is checked to revert to the *same* wild type. An
    assay whose rows disagree is a different protein per row, and the
    single-wildtype invariant that makes the likelihood-ratio convention
    rank-equivalent would not hold.
    """

    root = Path(directory) if directory is not None else PROTEINGYM_ROOT
    path = root / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"no ProteinGym assay at {path}")

    mutants: list[str] = []
    sequences: list[str] = []
    scores: list[float] = []
    bins: list[str] = []
    n_multi = 0
    with path.open() as handle:
        for row in csv.DictReader(handle):
            mutant = row["mutant"]
            tokens = mutant.split(":")
            if not include_multi and len(tokens) > 1:
                continue
            if any((t[0], t[-1]) not in BLOSUM62 for t in tokens):
                continue
            mutants.append(mutant)
            sequences.append(row["mutated_sequence"])
            scores.append(float(row["DMS_score"]))
            bins.append(row["DMS_score_bin"])
            n_multi += len(tokens) > 1

    if not mutants:
        raise RuntimeError(f"{name}: no eligible variants")
    eligible = len(mutants)
    wildtype = _revert(sequences[0], mutants[0])
    for mutant, sequence in zip(mutants, sequences):
        if _revert(sequence, mutant) != wildtype:
            raise RuntimeError(
                f"{name}: variant {mutant} reverts to a different wild type, so "
                "this assay is not a single-wildtype substitution set"
            )

    rng = np.random.default_rng(seed)
    if stratify_by_score_bin:
        strata: dict[str, list[int]] = {}
        for index, label in enumerate(bins):
            strata.setdefault(label, []).append(index)
        if len(strata) < 2:
            raise RuntimeError(
                f"{name}: DMS_score_bin has {len(strata)} level(s), so a "
                "class-balanced draw is not defined on this assay"
            )
        per_stratum = n // len(strata)
        holdout_per_stratum = train_holdout // len(strata)
        picked_list: list[int] = []
        for label in sorted(strata):
            shuffled = rng.permutation(strata[label])
            # The train split is removed from the pool first, exactly as their
            # design does, so the evaluated variants are disjoint from the ones
            # a circuit-selection step would have seen.
            pool = shuffled[holdout_per_stratum:]
            picked_list.extend(pool[:per_stratum].tolist())
        picked = np.array(sorted(picked_list), dtype=int)
    else:
        picked = np.sort(rng.permutation(eligible)[: min(n, eligible)])
    return Assay(
        name=name,
        wildtype=wildtype,
        mutants=[mutants[i] for i in picked],
        sequences=[sequences[i] for i in picked],
        scores=np.array([scores[i] for i in picked], dtype=np.float64),
        n_eligible=eligible,
        n_multi=n_multi,
        seed=seed,
        sampling=(
            f"score_bin_stratified, train_holdout={train_holdout} "
            "(ProGenMech's design, not their exact rows)"
            if stratify_by_score_bin
            else "seeded_uniform_permutation"
        ),
    )
