#!/usr/bin/env python3
"""Fail-closed verification for the registered family-holdout 3x3 campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import verify_family_holdout_experiment as cell_verifier


ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "DepPLM-family-holdout-multiseed-campaign-v1"
SUMMARY_SCHEMA = "DepPLM-family-holdout-multiseed-summary-v1"
RELEASE_SCHEMA = "DepPLM-family-holdout-multiseed-release-v1"
SPLIT_SEEDS = (42, 43, 44)
TRAINING_SEEDS = (42, 43, 44)
FOLDS = (0, 1, 2, 3, 4)
CONDITIONS = ("DepPLM", "C-PLM-MLP", "TRAIN-GENE-CELL-MEAN")
SUMMARY_METRICS = (
    "mse",
    "mae",
    "residual_r2_vs_train_gene_set_cell_mean",
    "cellwise_pearson_mean",
    "gene_wise_pearson_mean",
    "cellwise_spearman_mean",
)
EXPECTED_SOURCE_PATHS = (
    "code/run_family_holdout_campaign.py",
    "code/verify_family_holdout_campaign.py",
    "code/run_family_holdout_experiment.py",
    "code/verify_family_holdout_experiment.py",
    "code/family_holdout.py",
)


class CampaignVerificationError(RuntimeError):
    """Raised when a multiseed artifact violates the registered contract."""


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def expected_cells() -> list[dict[str, Any]]:
    return [
        {
            "split_seed": split_seed,
            "training_seed": training_seed,
            "path": f"cells/split-{split_seed}-training-{training_seed}",
        }
        for split_seed in SPLIT_SEEDS
        for training_seed in TRAINING_SEEDS
    ]


def validate_cell_inventory(records: Sequence[Mapping[str, Any]]) -> None:
    observed: list[tuple[int, int]] = []
    for record in records:
        try:
            observed.append(
                (int(record["split_seed"]), int(record["training_seed"]))
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CampaignVerificationError("cell inventory has an invalid seed key") from error
    if len(set(observed)) != len(observed):
        raise CampaignVerificationError("cell inventory contains a duplicate seed cell")
    expected = {
        (split_seed, training_seed)
        for split_seed in SPLIT_SEEDS
        for training_seed in TRAINING_SEEDS
    }
    missing = sorted(expected - set(observed))
    unexpected = sorted(set(observed) - expected)
    if missing:
        raise CampaignVerificationError(f"cell inventory is missing seed cells: {missing}")
    if unexpected:
        raise CampaignVerificationError(
            f"cell inventory contains unexpected seed cells: {unexpected}"
        )
    if len(records) != 9:
        raise CampaignVerificationError("cell inventory must contain exactly nine records")


def validate_child_device_binding(
    campaign_device: str,
    child_report: Mapping[str, Any],
    *,
    cell_key: tuple[int, int],
) -> None:
    """Reject a finalized child produced on a different requested device."""
    if child_report.get("device") != campaign_device:
        raise CampaignVerificationError(
            f"{cell_key}: child device differs from the top-level campaign"
        )


def verify_release_inventory(campaign: Path) -> None:
    release_path = campaign / "release-manifest.json"
    if not release_path.is_file():
        raise CampaignVerificationError("top-level release manifest is missing")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if release.get("schema") != RELEASE_SCHEMA or release.get("status") != "complete":
        raise CampaignVerificationError("top-level release manifest is not complete")
    listed = release.get("files")
    if not isinstance(listed, list):
        raise CampaignVerificationError("top-level release file inventory is missing")
    records = {str(record.get("path")): record for record in listed}
    if len(records) != len(listed):
        raise CampaignVerificationError("top-level release inventory has duplicate paths")
    actual = {
        path.relative_to(campaign).as_posix()
        for path in campaign.rglob("*")
        if path.is_file()
        and path.name != "release-manifest.json"
        and not any(part.startswith(".") for part in path.relative_to(campaign).parts)
    }
    if set(records) != actual:
        raise CampaignVerificationError(
            "top-level release inventory does not exactly match campaign files"
        )
    for relative, record in records.items():
        path = campaign / relative
        if path.stat().st_size != int(record.get("bytes", -1)):
            raise CampaignVerificationError(f"release byte count differs: {relative}")
        if sha256_file(path) != record.get("sha256"):
            raise CampaignVerificationError(f"release hash differs: {relative}")


def _finite_descriptive(values: Sequence[float | None]) -> dict[str, Any]:
    normalized = [
        None if value is None or not math.isfinite(float(value)) else float(value)
        for value in values
    ]
    finite = [value for value in normalized if value is not None]
    if len(finite) != len(normalized) or not finite:
        return {
            "status": "undefined_nonfinite_value",
            "n": len(normalized),
            "n_finite": len(finite),
            "mean": None,
            "sd": None,
            "minimum": None,
            "maximum": None,
            "values": normalized,
        }
    array = np.asarray(finite, dtype=np.float64)
    return {
        "status": "descriptive_dispersion",
        "n": int(array.size),
        "n_finite": int(array.size),
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)) if array.size > 1 else None,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "values": array.tolist(),
    }


def _repeat_summary(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[int, int], float | None] = {}
    for item in values:
        key = (int(item["split_seed"]), int(item["training_seed"]))
        if key in by_key:
            raise CampaignVerificationError(f"duplicate repeat-summary cell: {key}")
        value = item.get("value")
        by_key[key] = None if value is None else float(value)
    expected = {
        (split_seed, training_seed)
        for split_seed in SPLIT_SEEDS
        for training_seed in TRAINING_SEEDS
    }
    if set(by_key) != expected:
        raise CampaignVerificationError("repeat summary lacks the exact registered grid")
    ordered = [
        {
            "split_seed": split_seed,
            "training_seed": training_seed,
            "value": by_key[(split_seed, training_seed)],
        }
        for split_seed in SPLIT_SEEDS
        for training_seed in TRAINING_SEEDS
    ]
    per_split = []
    for split_seed in SPLIT_SEEDS:
        split_values = [
            by_key[(split_seed, training_seed)] for training_seed in TRAINING_SEEDS
        ]
        per_split.append(
            {
                "split_seed": split_seed,
                "training_seeds": list(TRAINING_SEEDS),
                "training_repeat_summary": _finite_descriptive(split_values),
            }
        )
    return {
        "inference": "descriptive_only",
        "analysis_unit": "complete_five_fold_oof_seed_cell",
        "n_cells": 9,
        "n_split_seeds": 3,
        "training_repeats_per_split": 3,
        "cell_values": ordered,
        "per_split": per_split,
        "across_split_means": _finite_descriptive(
            [item["training_repeat_summary"]["mean"] for item in per_split]
        ),
        "independence_warning": (
            "Outer folds and fit seeds are not independent biological replicates; "
            "all summaries are descriptive and preserve paired seed cells."
        ),
    }


def _expected_summary(
    aggregates: Mapping[tuple[int, int], Mapping[str, Any]],
) -> dict[str, Any]:
    condition_summaries: dict[str, Any] = {}
    for condition in CONDITIONS:
        condition_summaries[condition] = {}
        for metric in SUMMARY_METRICS:
            condition_summaries[condition][metric] = _repeat_summary(
                [
                    {
                        "split_seed": split_seed,
                        "training_seed": training_seed,
                        "value": aggregates[(split_seed, training_seed)]["conditions"][
                            condition
                        ]["pooled_outer_predictions"][metric],
                    }
                    for split_seed in SPLIT_SEEDS
                    for training_seed in TRAINING_SEEDS
                ]
            )
    paired: dict[str, Any] = {}
    for comparator in ("C-PLM-MLP", "TRAIN-GENE-CELL-MEAN"):
        name = f"DepPLM_minus_{comparator}"
        paired[name] = {}
        for metric in SUMMARY_METRICS:
            values = []
            for split_seed in SPLIT_SEEDS:
                for training_seed in TRAINING_SEEDS:
                    aggregate = aggregates[(split_seed, training_seed)]
                    left = aggregate["conditions"]["DepPLM"][
                        "pooled_outer_predictions"
                    ][metric]
                    right = aggregate["conditions"][comparator][
                        "pooled_outer_predictions"
                    ][metric]
                    values.append(
                        {
                            "split_seed": split_seed,
                            "training_seed": training_seed,
                            "value": (
                                None
                                if left is None or right is None
                                else float(left) - float(right)
                            ),
                        }
                    )
            paired[name][metric] = _repeat_summary(values)
    return {
        "schema": SUMMARY_SCHEMA,
        "status": "complete_descriptive_summary",
        "registered_split_seeds": list(SPLIT_SEEDS),
        "registered_training_seeds": list(TRAINING_SEEDS),
        "outer_folds_per_cell": list(FOLDS),
        "conditions": condition_summaries,
        "paired_complete_oof_differences": paired,
        "inference": "descriptive_only",
        "inference_note": (
            "Each value first pools one complete five-fold OOF prediction surface. "
            "Training-seed repeats are summarized within split seed before the three "
            "split means are described. Folds and fit seeds are not independent "
            "biological replicates; no p-value or confidence interval is reported."
        ),
    }


def _assert_nested_equal(observed: Any, expected: Any, *, context: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping) or set(observed) != set(expected):
            raise CampaignVerificationError(f"{context}: mapping keys differ")
        for key, value in expected.items():
            _assert_nested_equal(observed[key], value, context=f"{context}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise CampaignVerificationError(f"{context}: list length differs")
        for index, value in enumerate(expected):
            _assert_nested_equal(observed[index], value, context=f"{context}[{index}]")
        return
    if isinstance(expected, float):
        if not isinstance(observed, (int, float)) or not math.isclose(
            float(observed), expected, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise CampaignVerificationError(f"{context}: numeric value differs")
        return
    if observed != expected:
        raise CampaignVerificationError(f"{context}: value differs")


def _validate_completion(completion: Mapping[str, Any]) -> None:
    expected = {
        "schema": SCHEMA,
        "status": "registered_multiseed_campaign_complete",
        "completed_cells": expected_cells(),
        "n_completed_cells": 9,
        "folds_per_cell": 5,
        "all_child_verifiers_verified": True,
        "prediction_policy": "complete_oof_exact_once_and_hash_bound_per_cell",
        "inference": "descriptive_only",
        "source_stability": "verified",
    }
    for key, value in expected.items():
        if completion.get(key) != value:
            raise CampaignVerificationError(f"completion contract differs: {key}")
    if not isinstance(completion.get("completed_utc"), str):
        raise CampaignVerificationError("completion timestamp is missing")


def verify_campaign(campaign: Path) -> dict[str, Any]:
    campaign = campaign.resolve(strict=True)
    verify_release_inventory(campaign)
    expected_root_entries = {
        "run-manifest.json",
        "campaign-summary.json",
        "cell-index.json",
        "completion.json",
        "release-manifest.json",
        "cells",
    }
    if {path.name for path in campaign.iterdir()} != expected_root_entries:
        raise CampaignVerificationError("top-level campaign entry inventory differs")
    run = json.loads((campaign / "run-manifest.json").read_text(encoding="utf-8"))
    completion = json.loads((campaign / "completion.json").read_text(encoding="utf-8"))
    _validate_completion(completion)
    expected_configuration = {
        "split_seeds": list(SPLIT_SEEDS),
        "training_seeds": list(TRAINING_SEEDS),
        "outer_folds": list(FOLDS),
        "conditions": list(CONDITIONS),
    }
    configuration = run.get("configuration")
    if (
        run.get("schema") != SCHEMA
        or run.get("status") != "initialized"
        or not isinstance(configuration, Mapping)
        or any(configuration.get(key) != value for key, value in expected_configuration.items())
        or configuration.get("device") not in ("cpu", "cuda")
        or run.get("expected_cells") != expected_cells()
    ):
        raise CampaignVerificationError("run manifest identity differs")
    campaign_device = str(configuration["device"])
    sources = run.get("sources")
    if not isinstance(sources, list) or [item.get("path") for item in sources] != list(
        EXPECTED_SOURCE_PATHS
    ):
        raise CampaignVerificationError("source inventory differs")
    if len({item.get("path") for item in sources}) != len(sources):
        raise CampaignVerificationError("source inventory contains duplicates")
    for record in sources:
        path = ROOT / str(record["path"])
        if path.stat().st_size != int(record.get("bytes", -1)):
            raise CampaignVerificationError(f"source byte count differs: {record['path']}")
        if sha256_file(path) != record.get("sha256"):
            raise CampaignVerificationError(f"source hash differs: {record['path']}")

    cell_root = campaign / "cells"
    actual_cell_directories = {
        path.relative_to(campaign).as_posix()
        for path in cell_root.iterdir()
        if path.is_dir()
    }
    expected_cell_directories = {item["path"] for item in expected_cells()}
    if actual_cell_directories != expected_cell_directories:
        raise CampaignVerificationError("cell directory inventory differs from exact 3x3 grid")

    index = json.loads((campaign / "cell-index.json").read_text(encoding="utf-8"))
    records = index.get("cells")
    if not isinstance(records, list):
        raise CampaignVerificationError("cell index records are missing")
    validate_cell_inventory(records)
    if index.get("schema") != SCHEMA or index.get("status") != "all_cells_independently_verified":
        raise CampaignVerificationError("cell index identity differs")
    if index.get("cells_sha256") != canonical_json_sha256({"cells": records}):
        raise CampaignVerificationError("cell index hash differs")
    record_by_key = {
        (int(record["split_seed"]), int(record["training_seed"])): record
        for record in records
    }

    aggregates: dict[tuple[int, int], Mapping[str, Any]] = {}
    fold_hashes: dict[int, set[str]] = {seed: set() for seed in SPLIT_SEEDS}
    baseline_prediction_hashes: dict[tuple[int, int], list[str]] = {}
    for specification in expected_cells():
        key = (int(specification["split_seed"]), int(specification["training_seed"]))
        record = record_by_key[key]
        expected_record_keys = {
            "split_seed",
            "training_seed",
            "path",
            "release_manifest_sha256",
            "aggregate_summary_sha256",
            "fold_manifest_sha256",
            "verification_status",
        }
        if set(record) != expected_record_keys or record["path"] != specification["path"]:
            raise CampaignVerificationError(f"cell index record differs for {key}")
        cell_dir = campaign / record["path"]
        report = cell_verifier.verify_campaign(cell_dir)
        if (
            report.get("status") != "verified"
            or report.get("split_seed") != key[0]
            or report.get("training_seed") != key[1]
            or record.get("verification_status") != "verified"
        ):
            raise CampaignVerificationError(f"child verification identity differs for {key}")
        validate_child_device_binding(campaign_device, report, cell_key=key)
        bindings = (
            ("release_manifest_sha256", cell_dir / "release-manifest.json"),
            ("aggregate_summary_sha256", cell_dir / "aggregate-summary.json"),
            ("fold_manifest_sha256", cell_dir / "fold-manifest.tsv"),
        )
        for field, path in bindings:
            if record.get(field) != sha256_file(path) or report.get(field) != record.get(field):
                raise CampaignVerificationError(f"{key}: {field} binding differs")
        fold_hashes[key[0]].add(str(record["fold_manifest_sha256"]))
        aggregate = json.loads(
            (cell_dir / "aggregate-summary.json").read_text(encoding="utf-8")
        )
        aggregates[key] = aggregate
        baseline_prediction_hashes[key] = []
        for fold in FOLDS:
            result = json.loads(
                (
                    cell_dir
                    / f"fold-{fold:02d}"
                    / "TRAIN-GENE-CELL-MEAN"
                    / "result.json"
                ).read_text(encoding="utf-8")
            )
            baseline_prediction_hashes[key].append(
                str(result["outer_test_predictions"]["arrays"]["prediction"]["sha256"])
            )

    if any(len(values) != 1 for values in fold_hashes.values()):
        raise CampaignVerificationError("a split seed has inconsistent fold manifests")
    if len({next(iter(values)) for values in fold_hashes.values()}) != len(SPLIT_SEEDS):
        raise CampaignVerificationError("registered split seeds did not produce distinct manifests")
    for split_seed in SPLIT_SEEDS:
        baseline_sets = {
            tuple(baseline_prediction_hashes[(split_seed, training_seed)])
            for training_seed in TRAINING_SEEDS
        }
        if len(baseline_sets) != 1:
            raise CampaignVerificationError(
                f"split {split_seed}: deterministic baseline changes with training seed"
            )

    observed_summary = json.loads(
        (campaign / "campaign-summary.json").read_text(encoding="utf-8")
    )
    expected_summary = _expected_summary(aggregates)
    _assert_nested_equal(observed_summary, expected_summary, context="campaign-summary")
    summary_text = json.dumps(observed_summary, sort_keys=True).lower()
    if "p_value" in summary_text or "ci95" in summary_text:
        raise CampaignVerificationError("descriptive summary contains inferential fields")

    return {
        "status": "verified",
        "campaign": str(campaign),
        "grid_cells_verified": 9,
        "split_seeds": list(SPLIT_SEEDS),
        "training_seeds": list(TRAINING_SEEDS),
        "folds_per_cell": 5,
        "conditions": list(CONDITIONS),
        "child_campaigns_independently_verified": True,
        "predictions_exact_once_and_hash_bound": True,
        "repeat_aware_inference": "descriptive_only",
        "release_manifest_sha256": sha256_file(campaign / "release-manifest.json"),
        "campaign_summary_sha256": sha256_file(campaign / "campaign-summary.json"),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    report = verify_campaign(parse_args(argv).campaign)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
