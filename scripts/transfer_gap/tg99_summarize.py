"""TG-99: validate and collate one transfer-gap campaign.

Strict mode is the default. It writes nothing unless every non-summary stage in
``tg_contract.TG_STAGES`` has produced the arm matrix declared by that contract.
``--allow-partial`` is an exploratory escape hatch, not a silent fallback: its
JSON output records the complete expected, present, and missing matrices.

**Strict mode is now attainable.** It was not: the expected matrix fell back to
the whole four-arm TG panel wherever a stage declared no arms, and two stages
that declared none nonetheless refused arms inside their own bodies -- TG-05 can
produce one artefact of four and TG-06 three. A fully executed campaign could
therefore never satisfy the default mode, which made ``--allow-partial``
mandatory and the default decorative. The eligibility is declared in
``tg_contract`` now, derived from ``ArmSpec`` fields, and the matrix follows it.

TG-03 artefacts carry configuration in their filenames and payloads. A single
candidate per arm remains backward compatible. Multiple candidates are refused
unless each ambiguous arm is selected explicitly by a stable semantic identity.

Two quantities are combined across stages, and both now check that the stages
measured one population first; see :func:`population_refusal`. TG-00's two
control deltas are surfaced, which they never were.

**Every artefact must declare the contract it was produced under.** This
validated only that a file exists and parses as a JSON object, and then stamped
``contract_schema_version`` with the version *this code* declares -- the code's
opinion of itself, written onto a summary whose inputs may have been produced by
several generations of the stage code. It was: fourteen of the eighteen
non-summary artefacts in the corrected tree carry no ``contract`` key at all, one
of them predating the branch that decides whether its own control applies,
another trained under a different normalisation order, and two publishing the
retracted all-position residual spectrum under the primary field names. Strict
mode would have called that campaign complete. A missing or superseded
``contract`` block is now a refusal that names the files.

A partial summary announces itself on stdout as well as inside the JSON. It
wrote the same ``SUMMARY.json`` as a strict run and printed the same table with
``-`` in the absent cells, so the only thing separating an exploratory summary
from a complete one was a field two levels down in a file nobody re-opens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from tg_common import REPO, TG_PANEL, write_json
from tg_contract import SCHEMA_VERSION as CONTRACT_SCHEMA_VERSION
from tg_contract import TG_STAGES
from src.transfer.arms import PANEL

DEFAULT_ROOT = REPO / "results/transfer_gap_20260729_corrected"
SUMMARY_SCHEMA_VERSION = "transfer_gap_summary_v2"
GLOBAL_ARTEFACT = "__global__"
LN2 = math.log(2.0)

TG03_IDENTITY_FIELDS = (
    "arm",
    "layer",
    "n_layer",
    "d_model",
    "d_sae",
    "k",
    "seed",
    "train_tokens",
    "eval_tokens",
    "steps",
    "train_cohort",
    "eval_cohort",
)
_ABSENT = "<absent>"


class SummaryError(RuntimeError):
    """An input contract violation that must prevent summary creation."""


def canonical_matrix() -> dict[str, tuple[str, ...]]:
    """Expected artefact identities, derived from the existing TG contract."""

    matrix: dict[str, tuple[str, ...]] = {}
    for name, stage in TG_STAGES.items():
        if stage.scope == "summary":
            continue
        if stage.scope == "armless":
            matrix[name] = (GLOBAL_ARTEFACT,)
        else:
            matrix[name] = stage.arms if stage.arms is not None else tuple(TG_PANEL)
    return matrix


def load_json(path: Path) -> dict[str, Any]:
    """Read one JSON object with an error that identifies the bad artefact."""

    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise SummaryError(f"cannot read JSON artefact {path}: {error}") from error
    if not isinstance(value, dict):
        raise SummaryError(f"JSON artefact {path} must contain an object")
    return value


def contract_refusal(payload: dict[str, Any]) -> str | None:
    """Why an artefact is not a record of the contract this code enforces.

    ``None`` means it is. The check is deliberately on the artefact rather than
    on the tree: a results directory is written stage by stage over days, and
    "these files sit in one directory" has never implied "these files came from
    one generation of the code". The tg01 artefacts of the corrected tree were
    rewritten on 2026-07-30 and carry a contract block; the tg00, tg03, tg07 and
    tg09 artefacts beside them are from 2026-07-29 and carry none, and three
    substantive drifts between the two dates were confirmed.
    """

    contract = payload.get("contract")
    if not isinstance(contract, dict):
        return "carries no `contract` block"
    version = contract.get("schema_version")
    if version != CONTRACT_SCHEMA_VERSION:
        return f"declares contract schema {version!r}"
    return None


def tg03_stable_identity(payload: dict[str, Any]) -> str:
    """A filename-independent identity for a TG-03 training configuration."""

    identity = {
        key: payload[key] if key in payload else _ABSENT for key in TG03_IDENTITY_FIELDS
    }
    try:
        encoded = json.dumps(
            identity,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    except (TypeError, ValueError) as error:
        raise SummaryError(
            f"TG-03 configuration is not canonical JSON: {error}"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def parse_tg03_selectors(values: list[str]) -> dict[str, str]:
    """Parse repeated ``ARM=IDENTITY`` selectors without accepting shorthand."""

    selectors: dict[str, str] = {}
    for value in values:
        arm, separator, identity = value.partition("=")
        if not separator or arm not in TG_PANEL:
            raise SummaryError(
                f"invalid --tg03-select {value!r}; expected ARM=IDENTITY for an arm "
                f"in {list(TG_PANEL)}"
            )
        if arm in selectors:
            raise SummaryError(f"duplicate --tg03-select for {arm}")
        if len(identity) != 64 or any(
            char not in "0123456789abcdef" for char in identity
        ):
            raise SummaryError(
                f"invalid TG-03 identity for {arm}; use the full 64-character identity "
                "reported by the ambiguity error"
            )
        selectors[arm] = identity
    return selectors


def resolve_tg03(
    root: Path, selectors: dict[str, str]
) -> tuple[dict[str, Path], dict[str, dict[str, str]]]:
    """Resolve at most one TG-03 artefact per expected arm."""

    expected_arms = canonical_matrix()["tg03"]
    unknown = sorted(set(selectors) - set(expected_arms))
    if unknown:
        raise SummaryError(f"TG-03 selectors name undeclared arms: {unknown}")

    selected_paths: dict[str, Path] = {}
    selection_record: dict[str, dict[str, str]] = {}
    for arm in expected_arms:
        candidates: list[tuple[str, Path]] = []
        for path in sorted((root / "tg03").glob(f"{arm}_*.json")):
            payload = load_json(path)
            if payload.get("arm") != arm:
                raise SummaryError(
                    f"TG-03 artefact {path} is named for {arm!r} but declares "
                    f"arm={payload.get('arm')!r}"
                )
            candidates.append((tg03_stable_identity(payload), path))

        requested = selectors.get(arm)
        if requested is not None:
            matches = [
                (identity, path)
                for identity, path in candidates
                if identity == requested
            ]
            if len(matches) != 1:
                available = (
                    ", ".join(
                        f"{identity} ({path.name})" for identity, path in candidates
                    )
                    or "none"
                )
                raise SummaryError(
                    f"TG-03 selector for {arm} matched {len(matches)} artefacts; "
                    f"available identities: {available}"
                )
            chosen = matches[0]
        elif len(candidates) > 1:
            available = ", ".join(
                f"{identity} ({path.name})" for identity, path in candidates
            )
            raise SummaryError(
                f"ambiguous TG-03 configuration for {arm}; candidates: {available}. "
                f"Select one with --tg03-select {arm}=IDENTITY"
            )
        elif candidates:
            chosen = candidates[0]
        else:
            continue

        identity, path = chosen
        selected_paths[arm] = path
        selection_record[arm] = {
            "identity": identity,
            "path": str(path.relative_to(root)),
        }
    return selected_paths, selection_record


def inspect_campaign(
    root: Path, tg03_paths: dict[str, Path]
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Resolve and validate the canonical matrix without writing anything."""

    expected = canonical_matrix()
    paths: dict[str, Path] = {}
    present: dict[str, list[str]] = {}
    missing: dict[str, list[str]] = {}
    # Accumulated rather than raised at the first offender: a reader fixing a
    # stale tree needs the whole list, and fourteen of eighteen were stale.
    superseded: list[str] = []
    observed_versions: set[str] = set()

    def accept(path: Path) -> None:
        payload = load_json(path)
        refusal = contract_refusal(payload)
        if refusal is None:
            observed_versions.add(payload["contract"]["schema_version"])
        else:
            superseded.append(f"{path} {refusal}")

    for stage, artefacts in expected.items():
        stage_present: list[str] = []
        stage_missing: list[str] = []
        if artefacts == (GLOBAL_ARTEFACT,):
            candidates = sorted((root / stage).glob("*.json"))
            if len(candidates) > 1:
                names = ", ".join(path.name for path in candidates)
                raise SummaryError(
                    f"ambiguous armless stage {stage}; expected one JSON artefact, "
                    f"found {len(candidates)}: {names}"
                )
            if candidates:
                path = candidates[0]
                accept(path)
                paths[f"{stage}:{GLOBAL_ARTEFACT}"] = path
                stage_present.append(GLOBAL_ARTEFACT)
            else:
                stage_missing.append(GLOBAL_ARTEFACT)
        else:
            for artefact in artefacts:
                path = (
                    tg03_paths.get(artefact)
                    if stage == "tg03"
                    else root / stage / f"{artefact}.json"
                )
                if path is not None and path.is_file():
                    accept(path)
                    paths[f"{stage}:{artefact}"] = path
                    stage_present.append(artefact)
                else:
                    stage_missing.append(artefact)
        present[stage] = stage_present
        if stage_missing:
            missing[stage] = stage_missing

    if superseded:
        raise SummaryError(
            f"these artefacts under {root} were not produced under the contract "
            f"this code enforces ({CONTRACT_SCHEMA_VERSION}), so a summary built "
            "from them would carry numbers from several generations of the stage "
            "code under one schema stamp: "
            + "; ".join(superseded)
            + ". Re-run the stages named, or point --root at a tree measured "
            "under this contract"
        )

    required_count = sum(len(artefacts) for artefacts in expected.values())
    present_count = sum(len(artefacts) for artefacts in present.values())
    completeness = {
        "complete": not missing,
        # Observed, not stamped. This used to be CONTRACT_SCHEMA_VERSION
        # unconditionally, which asserted of every input the one thing none of
        # them had been asked. `None` is the honest answer for a campaign with no
        # artefacts at all; anything present has been checked against it above.
        "contract_schema_version": next(iter(observed_versions), None),
        "expected_matrix": {
            stage: list(artefacts) for stage, artefacts in expected.items()
        },
        "present_matrix": present,
        "missing_matrix": missing,
        "required_artifact_count": required_count,
        "present_artifact_count": present_count,
    }
    return paths, completeness


def _loaded(paths: dict[str, Path], stage: str, artefact: str) -> dict[str, Any] | None:
    path = paths.get(f"{stage}:{artefact}")
    return None if path is None else load_json(path)


def population_refusal(arm: str, *stages: str) -> str | None:
    """Why two stages' numbers may not be divided, or ``None`` if they may.

    Combining a numerator from one stage with a denominator from another asserts
    that the two measured one population, and a population is fixed by two
    declarations rather than one.

    *Which sequences.* TG-01 draws protein cohorts on 400-1000 residues and TG-03
    on 120-1000: they share no protein below 400 residues, and EXP-R2-060 prices
    protein cohort-block sensitivity at 0.16-0.60 nats -- larger than the 0.5-nat
    floor the guard beside this division applies. Text arms are unaffected by
    this axis, because a protein residue band does not select their cohort, and
    that asymmetry is what let the defect pass inspection: the number was well
    formed on the control arm.

    *Which positions of them.* A shared band still leaves the scored position
    distribution free, and this half was checked by nothing. TG-01 truncates at
    ``--max-len 384`` and scores every position 1..383 of every drawn sequence;
    TG-06 keeps only the sequences that reach 256 tokens and scores exactly
    positions 1..255 of those, at ``--window 256``. TG-01's own artefact prices
    the difference: its information gain by position bin runs from 1.89 nats in
    the first twentieth of the window to 4.51 at its widest on ProtGPT2, against
    that same 0.5-nat floor. Unlike a residue band, a token truncation selects
    positions in text exactly as it does in protein, so this axis is checked for
    every arm. ``frac_information_from_attention_pattern`` combined TG-01 and
    TG-06 across it and was published on every arm the pair could produce.

    A consequence worth stating plainly rather than engineering around: under the
    contract as it stands there is no arm for which TG-01 may be divided by TG-03
    or TG-06 at all. Making one of those ratios available means re-running a
    stage on the other's window, not relaxing this function.
    """

    reasons: list[str] = []
    if PANEL[arm].modality == "protein":
        bands = {stage: TG_STAGES[stage].protein_band for stage in stages}
        if len(set(bands.values())) != 1 or None in bands.values():
            described = ", ".join(f"{stage} {band}" for stage, band in bands.items())
            reasons.append(
                f"incommensurate protein cohort bands ({described}); a ratio "
                "across them would attribute one population's information to "
                "another's. EXP-R2-060 prices protein cohort-block sensitivity at "
                "0.16-0.60 nats"
            )
    windows = {
        stage: (
            None
            if TG_STAGES[stage].scoring_window is None
            else (
                TG_STAGES[stage].scoring_window.option,
                TG_STAGES[stage].scoring_window.tokens,
            )
        )
        for stage in stages
    }
    if len(set(windows.values())) != 1 or None in windows.values():
        described = ", ".join(
            f"{stage} {'undeclared' if window is None else f'{window[0]} {window[1]}'}"
            for stage, window in windows.items()
        )
        reasons.append(
            f"incommensurate scoring windows ({described}); the two stages scored "
            "different position distributions of their sequences, and a "
            "cross-entropy is position-dependent by more than the 0.5-nat floor "
            "beside this division"
        )
    return "; ".join(reasons) or None


def build_rows(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    """Build the legacy metric rows after the campaign contract has been checked."""

    rows: dict[str, dict[str, Any]] = {}
    try:
        for arm in TG_PANEL:
            row: dict[str, Any] = {}

            # TG-00 is the positive-control stage `tg_contract` calls "the stage
            # the audit's plan item B1 names; it must be run before any TG number
            # is quoted", and this summary read every other stage and not it: no
            # SUMMARY.json has ever carried a rendering or cohort delta. The two
            # controls price the two defects that were each worth more than most
            # of the effects measured on top of them.
            d0 = _loaded(paths, "tg00", arm)
            if d0 is not None:
                rendering, cohort_control = d0["rendering_control"], d0["cohort_control"]
                row |= {
                    "rendering_delta_nats": (
                        rendering["rendering_delta_nats"]
                        if rendering["applicable"]
                        else None
                    ),
                    "rendering_control_applicable": rendering["applicable"],
                    "wrong_control_token_delta_nats": rendering.get(
                        "wrong_control_token_delta_nats"
                    ),
                    "cohort_delta_nats": (
                        cohort_control["cohort_delta_nats"]
                        if cohort_control["applicable"]
                        else None
                    ),
                    "cohort_control_applicable": cohort_control["applicable"],
                }

            d1 = _loaded(paths, "tg01", arm)
            if d1 is not None:
                row |= {
                    "symbols_per_token": d1["symbols_per_token"],
                    "unigram_bits_per_token": d1["unigram_entropy_nats"] / LN2,
                    "model_bits_per_token": d1["model_nll_nats"] / LN2,
                    "gain_bits_per_token": d1["info_gain_over_unigram_bits"],
                    "gain_bits_per_symbol": d1["info_gain_bits_per_symbol"],
                    "frac_uncertainty_resolved": d1[
                        "fraction_of_unigram_entropy_explained"
                    ],
                    "top1_accuracy": d1["top1_accuracy"],
                    "local_share_within_8": d1["local_fraction_within_8"],
                    "unigram_plug_in_bias_nats": d1.get("unigram_plug_in_bias_nats"),
                    "gain_top_decile_share": d1["gain_top_decile_share"],
                    "markov2_bits_per_residue": d1.get("markov_order2_bits_per_symbol"),
                    "gain_by_position_bin": d1["gain_by_position_bin_nats"],
                }

            d2 = _loaded(paths, "tg02", arm)
            if d2 is not None:
                primary = d2[d2["primary_shuffle"]]
                row |= {
                    "far_bits": d2["far_context_information_bits"],
                    "far_order_bits": primary["far_order_information_bits"],
                    "far_order_share": primary["far_order_share"],
                }

            d3 = _loaded(paths, "tg03", arm)
            if d3 is not None:
                row |= {
                    "sae_layer": d3["layer"],
                    "sae_fvu": d3["fvu"],
                    "sae_dead": d3["dead_fraction"],
                    "sae_loss_recovered": d3["loss_recovered"],
                    "sae_delta_ce_nats": d3["ce_delta_nats"],
                    "sae_feature_var_by_token": d3[
                        "feature_variance_explained_by_current_token"
                    ]["mean"],
                    "sae_feature_var_by_position": d3[
                        "feature_variance_explained_by_position"
                    ]["mean"],
                    "sae_denominator_valid": d3.get("denominator_valid"),
                }
                if d1 is not None:
                    refusal = population_refusal(arm, "tg01", "tg03")
                    information = d1["unigram_entropy_nats"] - d3["ce_clean_nats"]
                    row["sae_frac_information_lost"] = (
                        None
                        if refusal is not None or information < 0.5
                        else d3["ce_delta_nats"] / information
                    )
                    row["sae_frac_information_lost_refusal"] = refusal or (
                        None
                        if information >= 0.5
                        else "denominator below the 0.5-nat floor"
                    )

            d5 = _loaded(paths, "tg05", arm)
            if d5 is not None:
                row |= {
                    f"contact_{key}": value
                    for key, value in d5["anchored_partner_auc"].items()
                }

            d6 = _loaded(paths, "tg06", arm)
            if d6 is not None:
                row |= {
                    "transplant_cost_bits": d6["transplant_cost_bits"],
                    "uniform_cost_bits": d6["uniform_cost_bits"],
                }
                if d1 is not None:
                    refusal = population_refusal(arm, "tg01", "tg06")
                    information = d1["unigram_entropy_nats"] - d6["ce_nats"]["clean"]
                    row["frac_information_from_attention_pattern"] = (
                        None
                        if refusal is not None or information < 0.5
                        else d6["transplant_cost_nats"] / information
                    )
                    row["frac_information_from_attention_pattern_refusal"] = refusal or (
                        None
                        if information >= 0.5
                        else "denominator below the 0.5-nat floor"
                    )
            rows[arm] = row
    except (KeyError, TypeError, ZeroDivisionError) as error:
        raise SummaryError(f"incompatible stage artefact schema: {error}") from error
    return rows


def print_table(rows: dict[str, dict[str, Any]]) -> None:
    """Print the existing compact comparison table."""

    keys = [
        "rendering_delta_nats",
        "cohort_delta_nats",
        "gain_bits_per_token",
        "gain_bits_per_symbol",
        "frac_uncertainty_resolved",
        "unigram_plug_in_bias_nats",
        "top1_accuracy",
        "local_share_within_8",
        "far_order_share",
        "sae_fvu",
        "sae_loss_recovered",
        "sae_frac_information_lost",
        "sae_feature_var_by_token",
        "frac_information_from_attention_pattern",
        "contact_partner_marginal_only",
        "contact_single_concat",
        "contact_attention_pattern",
    ]
    width = max(len(key) for key in keys) + 2
    print(f"{'metric':<{width}}" + "".join(f"{arm:>17}" for arm in TG_PANEL))
    for key in keys:
        cells = []
        for arm in TG_PANEL:
            value = rows[arm].get(key)
            cells.append(
                f"{value:>17.4f}" if isinstance(value, (int, float)) else f"{'-':>17}"
            )
        print(f"{key:<{width}}" + "".join(cells))


def partial_banner(completeness: dict[str, Any]) -> str:
    """The one line that separates an exploratory summary from a complete one.

    ``--allow-partial`` wrote the same ``root/SUMMARY.json`` as strict mode and
    printed the same table, with ``-`` in the cells no artefact backed. The mode
    was recorded two levels down in a file nobody re-opens after reading the
    table, so a screenshot of a partial run and of a complete one were the same
    picture.
    """

    missing = "; ".join(
        f"{stage}: {', '.join(artefacts)}"
        for stage, artefacts in completeness["missing_matrix"].items()
    )
    return (
        f"PARTIAL -- {completeness['present_artifact_count']} of "
        f"{completeness['required_artifact_count']} artefacts, missing: {missing}"
    )


def summarize(
    root: Path,
    *,
    allow_partial: bool,
    tg03_selectors: dict[str, str],
) -> Path:
    """Validate, build, and atomically write one summary."""

    tg03_paths, tg03_selection = resolve_tg03(root, tg03_selectors)
    paths, completeness = inspect_campaign(root, tg03_paths)
    completeness["mode"] = "allow-partial" if allow_partial else "strict"
    if not completeness["complete"] and not allow_partial:
        missing = "; ".join(
            f"{stage}: {', '.join(artefacts)}"
            for stage, artefacts in completeness["missing_matrix"].items()
        )
        raise SummaryError(
            f"incomplete transfer-gap campaign under {root}; missing {missing}. "
            "Use --allow-partial only for exploratory summaries"
        )

    rows = build_rows(paths)
    out: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "arms": rows,
        "results_root": str(root),
        "completeness": completeness,
        "tg03_selection": tg03_selection,
    }
    channel = _loaded(paths, "tg04", GLOBAL_ARTEFACT)
    if channel is not None:
        out["explanation_channel"] = channel

    destination = root / "SUMMARY.json"
    # Before anything else this run prints, including write_json's own progress
    # line: a banner under the table is a banner an operator scrolls past.
    if not completeness["complete"]:
        print(partial_banner(completeness))
    write_json(destination, out)
    print_table(rows)
    print(f"\nwrote {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="write an explicitly incomplete exploratory summary",
    )
    parser.add_argument(
        "--tg03-select",
        action="append",
        default=[],
        metavar="ARM=IDENTITY",
        help="select one ambiguous TG-03 artefact by its reported stable identity",
    )
    args = parser.parse_args()
    try:
        selectors = parse_tg03_selectors(args.tg03_select)
        summarize(
            Path(args.root),
            allow_partial=args.allow_partial,
            tg03_selectors=selectors,
        )
    except SummaryError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
