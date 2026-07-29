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
measured one population first; see :func:`band_refusal`. TG-00's two control
deltas are surfaced, which they never were.
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
                load_json(path)
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
                    load_json(path)
                    paths[f"{stage}:{artefact}"] = path
                    stage_present.append(artefact)
                else:
                    stage_missing.append(artefact)
        present[stage] = stage_present
        if stage_missing:
            missing[stage] = stage_missing

    required_count = sum(len(artefacts) for artefacts in expected.values())
    present_count = sum(len(artefacts) for artefacts in present.values())
    completeness = {
        "complete": not missing,
        "contract_schema_version": CONTRACT_SCHEMA_VERSION,
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


def band_refusal(arm: str, *stages: str) -> str | None:
    """Why two stages' numbers may not be divided, or ``None`` if they may.

    Combining a numerator from one stage with a denominator from another asserts
    that the two measured one population. TG-01 draws protein cohorts on 400-1000
    residues at ``max_len`` 384 and TG-03 on 120-1000 at 256: they share no
    protein below 400 residues, and EXP-R2-060 prices protein cohort-block
    sensitivity at 0.16-0.60 nats -- larger than the 0.5-nat floor the guard
    beside this division applies. The bands are already declared in
    ``tg_contract``; nothing was reading them.

    Text arms are unaffected, because a protein residue band does not select
    their cohort. That is exactly the asymmetry that let this pass inspection:
    the number was well formed on the control arm.
    """

    if PANEL[arm].modality != "protein":
        return None
    bands = {stage: TG_STAGES[stage].protein_band for stage in stages}
    if len(set(bands.values())) == 1 and None not in bands.values():
        return None
    described = ", ".join(f"{stage} {band}" for stage, band in bands.items())
    return (
        f"incommensurate protein cohort bands ({described}); a ratio across them "
        "would attribute one population's information to another's. EXP-R2-060 "
        "prices protein cohort-block sensitivity at 0.16-0.60 nats"
    )


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
                    refusal = band_refusal(arm, "tg01", "tg03")
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
                    refusal = band_refusal(arm, "tg01", "tg06")
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
