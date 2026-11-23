#!/usr/bin/env python3
"""Run the registered 3x3 family-holdout split/training-seed campaign.

Each grid cell is a complete five-fold OOF experiment.  This orchestrator is
resume-safe: it verifies finalized cells, resumes incomplete cells, and only
publishes its own release manifest after all nine cells pass the independent
cell verifier and the repeat-aware descriptive aggregate has been written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import run_family_holdout_experiment as cell_runner
import verify_family_holdout_experiment as cell_verifier


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "family_holdout_multiseed_v1"
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


class CampaignError(RuntimeError):
    """Raised when the registered multiseed campaign cannot be continued safely."""


def _sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = json.dumps(
        _json_safe(dict(value)),
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    try:
        temporary.write_text(payload, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_or_validate_json(path: Path, value: Mapping[str, Any]) -> None:
    expected = _json_safe(dict(value))
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != expected:
            raise CampaignError(f"resumed campaign metadata changed: {path}")
        return
    _atomic_write_json(path, expected)


def _write_or_validate_completion(
    path: Path, value: Mapping[str, Any]
) -> dict[str, Any]:
    expected = _json_safe(dict(value))
    if not path.exists():
        _atomic_write_json(path, expected)
        return expected
    observed = json.loads(path.read_text(encoding="utf-8"))
    without_time = lambda item: {  # noqa: E731 - compact stable projection
        key: entry for key, entry in item.items() if key != "completed_utc"
    }
    if without_time(observed) != without_time(expected):
        raise CampaignError("resumed multiseed completion metadata changed")
    return observed


def _file_record(path: Path, *, relative_to: Path = ROOT) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(relative_to.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _source_paths() -> list[Path]:
    return [
        Path(__file__).resolve(),
        ROOT / "code" / "verify_family_holdout_campaign.py",
        ROOT / "code" / "run_family_holdout_experiment.py",
        ROOT / "code" / "verify_family_holdout_experiment.py",
        ROOT / "code" / "family_holdout.py",
    ]


def expected_cells() -> list[dict[str, Any]]:
    """Return the immutable registered grid in deterministic execution order."""
    return [
        {
            "split_seed": split_seed,
            "training_seed": training_seed,
            "path": f"cells/split-{split_seed}-training-{training_seed}",
        }
        for split_seed in SPLIT_SEEDS
        for training_seed in TRAINING_SEEDS
    ]


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


def repeat_aware_summary(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Describe nine complete OOF cells while preserving the two seed axes."""
    expected_keys = {
        (split_seed, training_seed)
        for split_seed in SPLIT_SEEDS
        for training_seed in TRAINING_SEEDS
    }
    observed_keys = [
        (int(item["split_seed"]), int(item["training_seed"])) for item in values
    ]
    if len(set(observed_keys)) != len(observed_keys):
        raise CampaignError("repeat summary contains a duplicate seed cell")
    if set(observed_keys) != expected_keys:
        raise CampaignError("repeat summary does not contain the exact registered 3x3 grid")
    by_key = {
        (int(item["split_seed"]), int(item["training_seed"])): item.get("value")
        for item in values
    }
    ordered_cells = [
        {
            "split_seed": split_seed,
            "training_seed": training_seed,
            "value": (
                None
                if by_key[(split_seed, training_seed)] is None
                else float(by_key[(split_seed, training_seed)])
            ),
        }
        for split_seed in SPLIT_SEEDS
        for training_seed in TRAINING_SEEDS
    ]
    per_split: list[dict[str, Any]] = []
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
    split_means = [
        item["training_repeat_summary"]["mean"] for item in per_split
    ]
    return {
        "inference": "descriptive_only",
        "analysis_unit": "complete_five_fold_oof_seed_cell",
        "n_cells": len(ordered_cells),
        "n_split_seeds": len(SPLIT_SEEDS),
        "training_repeats_per_split": len(TRAINING_SEEDS),
        "cell_values": ordered_cells,
        "per_split": per_split,
        "across_split_means": _finite_descriptive(split_means),
        "independence_warning": (
            "Outer folds and fit seeds are not independent biological replicates; "
            "all summaries are descriptive and preserve paired seed cells."
        ),
    }


def build_campaign_summary(
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    aggregates: dict[tuple[int, int], Mapping[str, Any]] = {}
    for cell in cells:
        key = (int(cell["split_seed"]), int(cell["training_seed"]))
        if key in aggregates:
            raise CampaignError(f"duplicate aggregate for seed cell {key}")
        aggregate = cell.get("aggregate")
        if not isinstance(aggregate, Mapping):
            raise CampaignError(f"missing aggregate for seed cell {key}")
        aggregates[key] = aggregate
    if set(aggregates) != {
        (split_seed, training_seed)
        for split_seed in SPLIT_SEEDS
        for training_seed in TRAINING_SEEDS
    }:
        raise CampaignError("cannot aggregate anything except the exact registered 3x3 grid")

    condition_summaries: dict[str, Any] = {}
    for condition in CONDITIONS:
        condition_summaries[condition] = {}
        for metric in SUMMARY_METRICS:
            values = [
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
            condition_summaries[condition][metric] = repeat_aware_summary(values)

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
                    difference = (
                        None if left is None or right is None else float(left) - float(right)
                    )
                    values.append(
                        {
                            "split_seed": split_seed,
                            "training_seed": training_seed,
                            "value": difference,
                        }
                    )
            paired[name][metric] = repeat_aware_summary(values)

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


def _release_manifest(campaign_dir: Path) -> dict[str, Any]:
    files = [
        _file_record(path, relative_to=campaign_dir)
        for path in sorted(campaign_dir.rglob("*"))
        if path.is_file()
        and path.name != "release-manifest.json"
        and not any(
            part.startswith(".") for part in path.relative_to(campaign_dir).parts
        )
    ]
    return {
        "schema": RELEASE_SCHEMA,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }


def _verify_published_campaign(campaign_dir: Path) -> None:
    from verify_family_holdout_campaign import verify_campaign

    verify_campaign(campaign_dir)


def _publish_verified_release(
    campaign_dir: Path,
    *,
    verifier: Any = _verify_published_campaign,
) -> None:
    """Publish last and withdraw the marker if post-publish verification fails."""
    release_path = campaign_dir / "release-manifest.json"
    _atomic_write_json(release_path, _release_manifest(campaign_dir))
    try:
        verifier(campaign_dir)
    except BaseException:
        release_path.unlink(missing_ok=True)
        raise


def _resolved_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise CampaignError("--device cuda requested but CUDA is unavailable")
    return requested


def _new_campaign_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = output_root / f"registered-3x3-{timestamp}-{uuid.uuid4().hex[:8]}"
    path.mkdir(exist_ok=False)
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        help="Explicit create-or-resume directory for the complete registered grid.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    device = _resolved_device(args.device)
    campaign_dir = (
        _new_campaign_dir(args.output_root.resolve())
        if args.campaign_dir is None
        else args.campaign_dir.resolve()
    )
    campaign_dir.mkdir(parents=True, exist_ok=True)

    release_path = campaign_dir / "release-manifest.json"
    if release_path.exists():
        try:
            _verify_published_campaign(campaign_dir)
        except BaseException:
            # A release marker may survive process interruption between atomic
            # publication and final verification. Withdraw only that derived
            # marker so the next invocation can resume the immutable campaign.
            release_path.unlink(missing_ok=True)
            raise
        else:
            return campaign_dir

    run_path = campaign_dir / "run-manifest.json"
    if not run_path.exists() and any(campaign_dir.iterdir()):
        raise CampaignError(
            "refusing to initialize a non-empty directory without run-manifest.json"
        )

    source_records = [_file_record(path) for path in _source_paths()]
    run_manifest = {
        "schema": SCHEMA,
        "status": "initialized",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "split_seeds": list(SPLIT_SEEDS),
            "training_seeds": list(TRAINING_SEEDS),
            "outer_folds": list(FOLDS),
            "conditions": list(CONDITIONS),
            "device": device,
        },
        "expected_cells": expected_cells(),
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "sources": source_records,
        "aggregation_policy": (
            "complete OOF per split/training cell, then paired repeat-aware "
            "descriptive summaries; folds and fit seeds are not biological replicates"
        ),
    }
    if run_path.exists():
        observed = json.loads(run_path.read_text(encoding="utf-8"))
        stable_keys = (
            "schema",
            "status",
            "configuration",
            "expected_cells",
            "environment",
            "sources",
            "aggregation_policy",
        )
        if any(observed.get(key) != run_manifest[key] for key in stable_keys):
            raise CampaignError("resume manifest differs from the registered campaign")
    else:
        _atomic_write_json(run_path, run_manifest)

    cell_records: list[dict[str, Any]] = []
    for specification in expected_cells():
        split_seed = int(specification["split_seed"])
        training_seed = int(specification["training_seed"])
        cell_dir = campaign_dir / str(specification["path"])
        print(
            f"[family-campaign] split_seed={split_seed} "
            f"training_seed={training_seed}",
            flush=True,
        )
        if not (cell_dir / "release-manifest.json").is_file():
            cell_runner.main(
                [
                    "--split-seed",
                    str(split_seed),
                    "--training-seed",
                    str(training_seed),
                    "--device",
                    device,
                    "--campaign-dir",
                    str(cell_dir),
                ]
            )
        verification = cell_verifier.verify_campaign(cell_dir)
        if verification.get("device") != device:
            raise CampaignError(
                f"split_seed={split_seed} training_seed={training_seed}: "
                "child device differs from the top-level campaign"
            )
        aggregate_path = cell_dir / "aggregate-summary.json"
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        cell_records.append(
            {
                **specification,
                "release_manifest_sha256": _sha256_file(
                    cell_dir / "release-manifest.json"
                ),
                "aggregate_summary_sha256": _sha256_file(aggregate_path),
                "fold_manifest_sha256": verification["fold_manifest_sha256"],
                "verification_status": verification["status"],
                "aggregate": aggregate,
            }
        )

    current_sources = [_file_record(path) for path in _source_paths()]
    if current_sources != source_records:
        raise CampaignError("campaign source inventory changed during execution")

    summary = build_campaign_summary(cell_records)
    _write_or_validate_json(campaign_dir / "campaign-summary.json", summary)
    published_cells = [
        {key: value for key, value in cell.items() if key != "aggregate"}
        for cell in cell_records
    ]
    cells_payload = {
        "schema": SCHEMA,
        "status": "all_cells_independently_verified",
        "cells": published_cells,
        "cells_sha256": _canonical_json_sha256({"cells": published_cells}),
    }
    _write_or_validate_json(campaign_dir / "cell-index.json", cells_payload)
    completion = {
        "schema": SCHEMA,
        "status": "registered_multiseed_campaign_complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "completed_cells": expected_cells(),
        "n_completed_cells": 9,
        "folds_per_cell": 5,
        "all_child_verifiers_verified": True,
        "prediction_policy": "complete_oof_exact_once_and_hash_bound_per_cell",
        "inference": "descriptive_only",
        "source_stability": "verified",
    }
    _write_or_validate_completion(campaign_dir / "completion.json", completion)

    # Publish last.  Its exact inventory makes any missing, duplicate, or stale
    # child artifact fail closed in the independent top-level verifier.
    _publish_verified_release(campaign_dir)
    print(f"[family-campaign] complete campaign={campaign_dir}", flush=True)
    return campaign_dir


if __name__ == "__main__":
    main()
