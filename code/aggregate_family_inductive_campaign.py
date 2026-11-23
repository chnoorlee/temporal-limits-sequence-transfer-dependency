#!/usr/bin/env python3
"""Fail-closed validation and aggregation for the family-transfer campaign.

The scientific unit produced here is the frozen consensus of the complete
5-fold x 3-split-seed x 3-training-seed primary grid.  This module never reads
external outcomes.  It validates every run from its completion envelope,
replays campaign metrics from hash-bound arrays, and publishes a new output
directory with completion written last.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from corrected_protocol import derive_seed
from run_family_holdout_experiment import (
    FamilyExperimentError,
    derive_seeded_family_assignments,
    make_family_split,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOCK_PATH = ROOT / "docs" / "revision_v4" / "family_transfer_lock.json"
DEFAULT_PROTOCOL_PATH = ROOT / "docs" / "revision_v4" / "FAMILY_TRANSFER_PROTOCOL.md"
DEFAULT_RUNS_ROOT = ROOT / "results" / "family_inductive_v2" / "formal-runs"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "family_inductive_v2" / "campaigns"

ROSTER_SCHEMA = "DepPLM-family-inductive-campaign-roster-v2"
RUN_SCHEMA = "DepPLM-family-inductive-run-v2"
PREDICTION_SCHEMA = "DepPLM-family-inductive-predictions-v2"
METRICS_SCHEMA = "DepPLM-family-inductive-metrics-v2"
COMPLETION_SCHEMA = "DepPLM-family-inductive-completion-v2"
INPUT_MANIFEST_SCHEMA = "DepPLM-family-inductive-campaign-inputs-v2"
CAMPAIGN_SCHEMA = "DepPLM-family-inductive-campaign-v2"
CONSENSUS_SCHEMA = "DepPLM-family-inductive-consensus-v2"
CONSENSUS_PREDICTIONS_SCHEMA = "DepPLM-family-transfer-consensus-predictions-v2"
CAMPAIGN_COMPLETION_SCHEMA = "DepPLM-family-inductive-campaign-completion-v2"

BASELINE_METHODS = (
    "GLOBAL-TRAIN-MEAN",
    "SEQUENCE-KNN",
    "SEQUENCE-WEIGHTED-KNN",
    "PLM-RIDGE",
    "STATIC-CONVEX-STACK",
    "IDFREE-PLM-MEAN-MLP",
    "DECOMP-MEAN-ONLY",
)
PRIMARY_METHOD = "FAMILY-DISTANCE-STACK"
SECONDARY_CONTEXT_METHOD = "FAMILY-INDUCTIVE-DECOMP"
ALL_METHODS = (
    "GLOBAL-TRAIN-MEAN",
    "SEQUENCE-KNN",
    "SEQUENCE-WEIGHTED-KNN",
    "PLM-RIDGE",
    "STATIC-CONVEX-STACK",
    PRIMARY_METHOD,
    "IDFREE-PLM-MEAN-MLP",
    "DECOMP-MEAN-ONLY",
    SECONDARY_CONTEXT_METHOD,
)
SELECTED_BASELINE_CONSENSUS = "CONSENSUS-PER-JOB-SELECTED-BASELINE"
STACK_METHOD_ORDER = (
    "GLOBAL-TRAIN-MEAN",
    "SEQUENCE-WEIGHTED-KNN",
    "PLM-RIDGE",
)
STACK_BASE_PREDICTION_ARRAY_KEYS: dict[str, dict[str, str]] = {
    PRIMARY_METHOD: {
        "GLOBAL-TRAIN-MEAN": "primary_stack_base_global_prediction",
        "SEQUENCE-WEIGHTED-KNN": "primary_stack_base_weighted_knn_prediction",
        "PLM-RIDGE": "primary_stack_base_ridge_prediction",
    },
    "STATIC-CONVEX-STACK": {
        "GLOBAL-TRAIN-MEAN": "static_stack_base_global_prediction",
        "SEQUENCE-WEIGHTED-KNN": "static_stack_base_weighted_knn_prediction",
        "PLM-RIDGE": "static_stack_base_ridge_prediction",
    },
}
CONSENSUS_PREDICTION_KEYS = (
    "schema",
    "gene_ids",
    "canonical_gene_indices",
    "component_ids",
    "family_distance_stack_gene_mean_prediction",
    "consensus_per_job_selected_baseline_gene_mean_prediction",
    "consensus_global_train_mean_prediction",
)
RUN_CODE_PATHS = frozenset({
    "code/run_family_inductive_experiment.py",
    "code/family_inductive.py",
    "code/run_family_holdout_experiment.py",
    "code/run_corrected_experiments.py",
    "code/corrected_protocol.py",
    "code/embedding_provenance.py",
    "code/family_holdout.py",
    "code/models.py",
    "docs/revision_v4/family_transfer_lock.json",
    "docs/revision_v4/FAMILY_TRANSFER_PROTOCOL.md",
})
RUN_ADDITIONAL_INPUT_PATHS = frozenset({
    "data/processed/lineages.txt",
    "data/processed/ppi_graph.npz",
    "data/depmap/OmicsExpressionProteinCodingGenesTPMLogp1.csv",
    "data/depmap/OmicsSomaticMutations.csv",
})
HEX64 = re.compile(r"[0-9a-f]{64}")
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class CampaignValidationError(RuntimeError):
    """Raised when any input violates the frozen campaign contract."""


class AtomicPublishError(RuntimeError):
    """Raised when an immutable campaign destination cannot be published."""


@dataclass(frozen=True, order=True)
class JobKey:
    identity_percent: int
    outer_fold: int
    split_seed: int
    training_seed: int

    def as_dict(self) -> dict[str, int]:
        return {
            "identity_percent": int(self.identity_percent),
            "outer_fold": int(self.outer_fold),
            "split_seed": int(self.split_seed),
            "training_seed": int(self.training_seed),
        }

    def label(self) -> str:
        return (
            f"identity-{self.identity_percent:g}|fold-{self.outer_fold}|"
            f"split-{self.split_seed}|train-{self.training_seed}"
        )


@dataclass
class ValidatedRun:
    key: JobKey
    run_id: str
    path: Path
    arrays: dict[str, np.ndarray]
    baseline_inner_validation: dict[str, dict[str, float]]
    job_selected_baseline: str
    manifest: dict[str, Any]
    source_records: list[dict[str, Any]]


@dataclass(frozen=True)
class ReplayedFamilyPartition:
    genes: tuple[str, ...]
    component_ids: tuple[str, ...]
    fold_ids: tuple[int, ...]
    inner_train: np.ndarray
    inner_validation: np.ndarray
    outer_train: np.ndarray
    outer_test: np.ndarray


@dataclass
class Consensus:
    arrays: dict[str, np.ndarray]
    job_baseline_selection: dict[str, Any]
    seed_pair_metrics: dict[str, Any]


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise CampaignValidationError(f"duplicate JSON key: {key!r}")
        output[key] = value
    return output


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    _reject_external_path(path)
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CampaignValidationError(f"cannot read valid {label} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CampaignValidationError(f"{label} JSON root must be an object: {path}")
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _json_safe(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    _reject_external_path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _hash_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _hash_strings(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _file_record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    base = relative_to.resolve(strict=True)
    try:
        display = resolved.relative_to(base).as_posix()
    except ValueError as exc:
        raise CampaignValidationError(f"artifact escapes its root: {path}") from exc
    return {
        "path": display,
        "bytes": int(resolved.stat().st_size),
        "sha256": _sha256_file(resolved),
    }


def _reject_external_path(path: Path) -> None:
    """Reject any attempted read under the repository external-outcome tree."""
    try:
        resolved = path.resolve(strict=False)
        external = (ROOT / "data" / "external").resolve(strict=False)
        resolved.relative_to(external)
    except ValueError:
        return
    raise CampaignValidationError(f"external outcome reads are forbidden: {path}")


def _validate_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise CampaignValidationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_record(
    record: Any,
    path: Path,
    *,
    expected_path: str,
    label: str,
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise CampaignValidationError(f"{label} record is missing")
    expected_keys = {"path", "bytes", "sha256"}
    if set(record) != expected_keys:
        raise CampaignValidationError(
            f"{label} record keys mismatch: expected {sorted(expected_keys)!r}, "
            f"observed {sorted(record)!r}"
        )
    if record.get("path") != expected_path:
        raise CampaignValidationError(f"{label} path must be {expected_path!r}")
    expected_digest = _validate_sha256(record.get("sha256"), label=f"{label}.sha256")
    size = record.get("bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise CampaignValidationError(f"{label}.bytes must be a non-negative integer")
    actual = _file_record(path, relative_to=path.parent)
    if actual != dict(record):
        raise CampaignValidationError(
            f"{label} file record mismatch for {path}; expected digest "
            f"{expected_digest}, observed {actual['sha256']}"
        )
    return dict(record)


def _safe_run_file(run_dir: Path, filename: str) -> Path:
    if PurePosixPath(filename).name != filename or filename in {".", ".."}:
        raise CampaignValidationError(f"unsafe run artifact name: {filename!r}")
    if run_dir.is_symlink():
        raise CampaignValidationError(f"run directory may not be a symlink: {run_dir}")
    path = run_dir / filename
    if path.is_symlink():
        raise CampaignValidationError(f"run artifact may not be a symlink: {path}")
    try:
        path.resolve(strict=True).relative_to(run_dir.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise CampaignValidationError(f"run artifact is missing or escapes run root: {path}") from exc
    if not path.is_file():
        raise CampaignValidationError(f"run artifact is not a regular file: {path}")
    return path


def load_analysis_lock(path: Path = DEFAULT_LOCK_PATH) -> tuple[dict[str, Any], dict[str, Any]]:
    lock = _load_json(path, label="analysis lock")
    if lock.get("schema") != "DepPLM-family-transfer-analysis-lock-v2":
        raise CampaignValidationError("analysis lock schema mismatch")
    if lock.get("protocol_version") != "1.4":
        raise CampaignValidationError("analysis lock protocol version must be 1.4")
    if lock.get("implementation_namespace") != "family_inductive_v2":
        raise CampaignValidationError("analysis lock implementation namespace mismatch")
    if lock.get("execution_scope") != "remote-server-only":
        raise CampaignValidationError("analysis lock execution scope is not remote-server-only")
    if lock.get("primary_method") != PRIMARY_METHOD:
        raise CampaignValidationError("analysis lock primary method mismatch")
    if lock.get("primary_metric") != "gene_mean_r2_vs_consensus_per_job_outer_training_global_prediction":
        raise CampaignValidationError("analysis lock primary metric mismatch")
    if lock.get("positive_result_is_not_guaranteed") is not True:
        raise CampaignValidationError("analysis lock must preserve the negative-result boundary")
    primary_contract = lock.get("primary_method_contract")
    expected_primary_contract = {
        "representation": "frozen_esm_only_no_gene_identifier",
        "base_methods": [
            "GLOBAL-TRAIN-MEAN",
            "SEQUENCE-WEIGHTED-KNN",
            "PLM-RIDGE",
        ],
        "formula": "sum_m((a_m+b_m*z)*p_m)",
        "distance": "one_minus_maximum_outer_training_cosine_similarity",
        "distance_scaling": "clip((d-q10)/(q90-q10),0,1)",
        "distance_quantiles": [0.1, 0.9],
        "degenerate_quantile_action": "fail_closed",
        "constraints": [
            "sum_a_equals_one",
            "sum_b_equals_zero",
            "endpoint_weights_nonnegative",
            "weighted_knn_slope_nonpositive",
            "global_slope_nonnegative",
        ],
        "objective": "component_cross_fitted_mse_plus_lambda_sum_squared_slopes",
        "inner_crossfit_folds": 3,
        "outer_refit_crossfit_folds": 4,
        "crossfit_assignment": "deterministic_component_hash_with_training_seed",
        "candidate_grid": {
            "knn_neighbors": [5, 20, 50],
            "knn_temperature": [0.03, 0.1, 0.3],
            "ridge_alpha": [0.1, 1.0, 10.0, 100.0],
            "gate_lambda": [0.0, 0.01, 0.1],
        },
        "implementation_contract": {
            "solver": "scipy.optimize.SLSQP",
            "max_iterations": 2000,
            "ftol": 1e-12,
            "constraint_tolerance": 1e-8,
            "initialization": "equal_endpoint_weights",
            "objective_weighting": "gene_weighted_component_cross_fitted_mse",
            "failure_conditions": [
                "solver_unsuccessful",
                "nonfinite_objective_or_coefficient",
                "constraint_violation_above_tolerance",
            ],
            "component_fold_hash": (
                "sha256(utf8(decimal_training_seed)+NUL+utf8(component_id))_"
                "first_8_bytes_big_endian_mod_n_folds"
            ),
            "distance_stack_weight_arrays": (
                "one_weight_per_query_gene_equal_to_a_m_plus_b_m_times_z"
            ),
            "selected_baseline_name_array": "length_one_fixed_width_unicode",
            "static_stack_grid": (
                "one_static_fit_per_knn_k_temperature_and_ridge_alpha_without_"
                "lambda_duplication"
            ),
            "audit": (
                "record_scipy_version_coefficients_constraint_residuals_and_"
                "solver_status"
            ),
        },
        "selection_rule": "max_inner_component_heldout_r2_then_min_mse_then_canonical_config",
        "required_ablation": "STATIC-CONVEX-STACK",
    }
    if primary_contract != expected_primary_contract:
        raise CampaignValidationError("primary method contract differs from the frozen v1.4 design")
    split = lock.get("primary_split")
    campaign = lock.get("campaign_aggregation")
    if not isinstance(split, Mapping) or not isinstance(campaign, Mapping):
        raise CampaignValidationError("analysis lock lacks split or campaign contract")
    expected_grid = {
        "outer_folds": [0, 1, 2, 3, 4],
        "split_seeds": [42, 43, 44],
        "training_seeds": [42, 43, 44],
    }
    if float(split.get("identity_percent", -1)) != 30.0:
        raise CampaignValidationError("primary identity threshold must be 30 percent")
    if split.get("outer_folds") != 5:
        raise CampaignValidationError("primary split must contain five outer folds")
    if split.get("split_seeds") != [42, 43, 44] or split.get("training_seeds") != [42, 43, 44]:
        raise CampaignValidationError("primary split seed grid differs from the frozen contract")
    if campaign.get("job_grid") != expected_grid:
        raise CampaignValidationError("campaign job grid differs from the locked 5x3x3 grid")
    if campaign.get("required_primary_jobs") != 45:
        raise CampaignValidationError("campaign must require exactly 45 jobs")
    if campaign.get("baseline_family_candidates") != list(BASELINE_METHODS):
        raise CampaignValidationError("baseline candidate order or membership differs from lock")
    if campaign.get("baseline_family_selection") != "independent_within_each_job_from_that_jobs_inner_validation_only":
        raise CampaignValidationError("per-job baseline selection rule differs from lock")
    if campaign.get("cross_job_baseline_family_selection_forbidden") is not True:
        raise CampaignValidationError("cross-job baseline selection must be forbidden")
    if campaign.get("selected_baseline_consensus") != "arithmetic_mean_of_each_jobs_explicit_selected_baseline_outer_prediction":
        raise CampaignValidationError("selected-baseline consensus rule differs from lock")
    if campaign.get("per_gene_prediction_count") != 9:
        raise CampaignValidationError("campaign must require nine predictions per gene")
    if campaign.get("prediction_aggregation") != "arithmetic_mean_across_three_split_seeds_by_three_training_seeds":
        raise CampaignValidationError("campaign prediction aggregation is not arithmetic mean")
    if campaign.get("campaign_roster_digest_required") is not True:
        raise CampaignValidationError("campaign roster digest is not required by lock")
    if campaign.get("external_outcomes_forbidden_until_consensus_artifact_is_frozen") is not True:
        raise CampaignValidationError("external-outcome boundary is not locked")
    keys = campaign.get("run_prediction_array_keys")
    if not isinstance(keys, Mapping):
        raise CampaignValidationError("prediction array-key map is missing")
    method_keys = keys.get("method_gene_mean_predictions")
    if not isinstance(method_keys, Mapping) or tuple(method_keys) != ALL_METHODS:
        raise CampaignValidationError("prediction method-key map differs from registered methods")
    expected_scalar_keys = {
        "gene_ids",
        "canonical_gene_indices",
        "source_gene_indices",
        "component_ids",
        "component_sizes",
        "protein_lengths",
        "cell_ids",
        "source_cell_indices",
        "observed_gene_effect",
        "observed_gene_mean",
        "observed_gene_centered_residual",
        "nearest_training_distance",
        "distance_gate_coordinate",
        "distance_stack_weight_global",
        "distance_stack_weight_knn",
        "distance_stack_weight_ridge",
        "selective_risk_score",
        "selected_baseline_name",
        "selected_baseline_prediction",
        "secondary_centered_residual_prediction",
        "secondary_total_prediction",
    }
    if set(keys) != expected_scalar_keys | {"method_gene_mean_predictions"}:
        raise CampaignValidationError("prediction scalar-key roles differ from the v1.4 lock")
    registered_array_names = [
        *(str(keys[role]) for role in expected_scalar_keys),
        *(str(value) for value in method_keys.values()),
    ]
    if len(registered_array_names) != len(set(registered_array_names)):
        raise CampaignValidationError("prediction array names are not globally unique")
    expected_consensus_prediction_artifact = {
        "filename": "consensus-predictions.npz",
        "schema": CONSENSUS_PREDICTIONS_SCHEMA,
        "exact_keys": list(CONSENSUS_PREDICTION_KEYS),
        "completion_binding": (
            "completion.json.artifacts.consensus_predictions_records_file_bytes_"
            "sha256_and_per_array_dtype_shape_sha256"
        ),
    }
    if campaign.get("consensus_prediction_artifact") != expected_consensus_prediction_artifact:
        raise CampaignValidationError("consensus prediction artifact contract differs from v1.4 lock")
    if lock.get("required_baselines") != list(BASELINE_METHODS):
        raise CampaignValidationError("required baseline list differs from campaign candidates")
    bootstrap = lock.get("bootstrap")
    if bootstrap != {
        "unit": "sequence_component",
        "replicates": 5000,
        "confidence_level": 0.95,
        "seed": 20261027,
    }:
        raise CampaignValidationError("campaign bootstrap contract mismatch")
    if lock.get("coverage_levels") != [1.0, 0.9, 0.75, 0.5]:
        raise CampaignValidationError("coverage levels differ from the frozen contract")
    if campaign.get("primary_metrics_and_gates_use_consensus_predictions") is not True:
        raise CampaignValidationError("campaign gates must use consensus predictions")
    if campaign.get("missing_failed_or_duplicate_job_blocks_aggregation") is not True:
        raise CampaignValidationError("incomplete campaigns must block aggregation")
    return lock, {
        "path": path.resolve(strict=True).relative_to(ROOT.resolve(strict=True)).as_posix(),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def expected_job_keys(lock: Mapping[str, Any]) -> tuple[JobKey, ...]:
    grid = lock["campaign_aggregation"]["job_grid"]
    identity = int(lock["primary_split"]["identity_percent"])
    return tuple(
        JobKey(identity, int(fold), int(split_seed), int(training_seed))
        for split_seed in grid["split_seeds"]
        for training_seed in grid["training_seeds"]
        for fold in grid["outer_folds"]
    )


def _run_id(campaign_id: str, key: JobKey) -> str:
    return (
        f"{campaign_id}-i{key.identity_percent:g}-of{key.outer_fold}-"
        f"ss{key.split_seed}-ts{key.training_seed}"
    )


def build_roster(
    campaign_id: str,
    *,
    lock_path: Path = DEFAULT_LOCK_PATH,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
) -> dict[str, Any]:
    if IDENTIFIER.fullmatch(campaign_id) is None:
        raise CampaignValidationError("campaign id must be 1-128 safe ASCII characters")
    lock, lock_record = load_analysis_lock(lock_path)
    protocol_path = protocol_path.resolve(strict=True)
    protocol_record = _file_record(protocol_path, relative_to=ROOT)
    jobs = [
        {"run_id": _run_id(campaign_id, key), "run_key": key.as_dict()}
        for key in expected_job_keys(lock)
    ]
    if len(jobs) != 45 or len({job["run_id"] for job in jobs}) != 45:
        raise CampaignValidationError("internal roster construction did not yield 45 unique jobs")
    return {
        "schema": ROSTER_SCHEMA,
        "campaign_id": campaign_id,
        "protocol_version": "1.4",
        "scientific_use": True,
        "pilot": False,
        "analysis_lock": lock_record,
        "protocol": protocol_record,
        "required_job_count": 45,
        "job_grid": lock["campaign_aggregation"]["job_grid"],
        "jobs": jobs,
    }


def write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise AtomicPublishError(f"refusing to overwrite existing file: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def load_roster(
    path: Path,
    *,
    campaign_id: str,
    lock: Mapping[str, Any],
    lock_record: Mapping[str, Any],
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
) -> tuple[dict[str, Any], str, dict[str, JobKey]]:
    roster = _load_json(path, label="campaign roster")
    roster_sha256 = _sha256_file(path)
    expected_roster_fields = {
        "schema",
        "campaign_id",
        "protocol_version",
        "scientific_use",
        "pilot",
        "analysis_lock",
        "protocol",
        "required_job_count",
        "job_grid",
        "jobs",
    }
    if set(roster) != expected_roster_fields:
        raise CampaignValidationError("campaign roster root fields mismatch")
    if roster.get("schema") != ROSTER_SCHEMA:
        raise CampaignValidationError("campaign roster schema mismatch")
    if roster.get("campaign_id") != campaign_id:
        raise CampaignValidationError("campaign roster id mismatch")
    if roster.get("protocol_version") != "1.4":
        raise CampaignValidationError("campaign roster protocol mismatch")
    if roster.get("scientific_use") is not True or roster.get("pilot") is not False:
        raise CampaignValidationError("campaign roster is not a formal scientific roster")
    if roster.get("analysis_lock") != dict(lock_record):
        raise CampaignValidationError("campaign roster is bound to a different analysis lock")
    protocol_record = _file_record(protocol_path.resolve(strict=True), relative_to=ROOT)
    if roster.get("protocol") != protocol_record:
        raise CampaignValidationError("campaign roster is bound to a different protocol document")
    if roster.get("required_job_count") != 45:
        raise CampaignValidationError("campaign roster job count is not 45")
    if roster.get("job_grid") != lock["campaign_aggregation"]["job_grid"]:
        raise CampaignValidationError("campaign roster grid differs from analysis lock")
    raw_jobs = roster.get("jobs")
    if not isinstance(raw_jobs, list) or len(raw_jobs) != 45:
        raise CampaignValidationError("campaign roster must contain exactly 45 jobs")
    indexed: dict[str, JobKey] = {}
    expected = set(expected_job_keys(lock))
    for position, item in enumerate(raw_jobs):
        if not isinstance(item, Mapping) or set(item) != {"run_id", "run_key"}:
            raise CampaignValidationError(f"invalid roster job at position {position}")
        run_id = item.get("run_id")
        if not isinstance(run_id, str) or IDENTIFIER.fullmatch(run_id) is None:
            raise CampaignValidationError(f"invalid roster run id at position {position}")
        key = _parse_job_key(item.get("run_key"), label=f"roster jobs[{position}].run_key")
        if run_id != _run_id(campaign_id, key):
            raise CampaignValidationError(f"noncanonical roster run id: {run_id!r}")
        if run_id in indexed:
            raise CampaignValidationError(f"duplicate roster run id: {run_id!r}")
        indexed[run_id] = key
    if set(indexed.values()) != expected or len(set(indexed.values())) != 45:
        raise CampaignValidationError("roster jobs are not the exact locked Cartesian product")
    return roster, roster_sha256, indexed


def _parse_job_key(value: Any, *, label: str) -> JobKey:
    expected = {"identity_percent", "outer_fold", "split_seed", "training_seed"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise CampaignValidationError(f"{label} must contain exactly {sorted(expected)!r}")
    numeric = (
        value["identity_percent"],
        value["outer_fold"],
        value["split_seed"],
        value["training_seed"],
    )
    if any(type(item) is not int for item in numeric):
        raise CampaignValidationError(f"{label} values must be JSON integers")
    return JobKey(*numeric)


def _parse_run_key(value: Any, *, roster_sha256: str, label: str) -> JobKey:
    expected = {
        "identity_percent",
        "outer_fold",
        "split_seed",
        "training_seed",
        "campaign_roster_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise CampaignValidationError(f"{label} must contain exactly {sorted(expected)!r}")
    if value.get("campaign_roster_sha256") != roster_sha256:
        raise CampaignValidationError(f"{label} roster digest mismatch")
    return _parse_job_key(
        {
            "identity_percent": value["identity_percent"],
            "outer_fold": value["outer_fold"],
            "split_seed": value["split_seed"],
            "training_seed": value["training_seed"],
        },
        label=label,
    )


def _assert_designated_server(lock: Mapping[str, Any]) -> dict[str, Any]:
    registered = lock.get("designated_server")
    if not isinstance(registered, Mapping):
        raise CampaignValidationError("analysis lock designated server is missing")
    hostname = platform.node()
    if hostname != registered.get("hostname"):
        raise CampaignValidationError(
            f"aggregator hostname mismatch: expected {registered.get('hostname')!r}, observed {hostname!r}"
        )
    machine_id_path = Path("/etc/machine-id")
    if not machine_id_path.is_file():
        raise CampaignValidationError("designated-server /etc/machine-id is unavailable")
    machine_digest = hashlib.sha256(machine_id_path.read_bytes()).hexdigest()
    if machine_digest != registered.get("machine_id_sha256"):
        raise CampaignValidationError("aggregator machine-id digest mismatch")
    observed: set[str] = set()
    try:
        output = subprocess.run(
            ["ip", "-o", "-4", "addr", "show"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        observed.update(re.findall(r"\binet\s+([0-9.]+)/", output))
    except (OSError, subprocess.CalledProcessError):
        pass
    try:
        observed.update(
            item[4][0]
            for item in socket.getaddrinfo(hostname, None, socket.AF_INET)
        )
    except OSError:
        pass
    required_ip = str(registered.get("ipv4"))
    if required_ip not in observed:
        raise CampaignValidationError(
            f"aggregator designated IPv4 {required_ip!r} not observed; observed={sorted(observed)!r}"
        )
    return {
        "hostname": hostname,
        "machine_id_sha256": machine_digest,
        "verified_ipv4": required_ip,
        "observed_ipv4_addresses": sorted(observed),
    }


def _prediction_key_map(lock: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    registered = lock["campaign_aggregation"]["run_prediction_array_keys"]
    scalar = {
        str(name): str(key)
        for name, key in registered.items()
        if name != "method_gene_mean_predictions"
    }
    methods = {
        str(name): str(key)
        for name, key in registered["method_gene_mean_predictions"].items()
    }
    return scalar, methods


def _read_npz(
    path: Path,
    *,
    prediction_record: Mapping[str, Any],
    required_keys: set[str],
) -> dict[str, np.ndarray]:
    _reject_external_path(path)
    if prediction_record.get("schema") != PREDICTION_SCHEMA:
        raise CampaignValidationError(f"prediction record schema mismatch: {path}")
    array_records = prediction_record.get("arrays")
    if not isinstance(array_records, Mapping) or set(array_records) != required_keys:
        raise CampaignValidationError(f"prediction array manifest is not the exact registered allowlist: {path}")
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = archive.namelist()
            expected_members = [f"{name}.npy" for name in required_keys]
            if len(members) != len(set(members)):
                raise CampaignValidationError(
                    f"prediction NPZ has duplicate ZIP members: {path}"
                )
            if set(members) != set(expected_members):
                raise CampaignValidationError(
                    f"prediction NPZ member allowlist mismatch: {path}"
                )
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != required_keys:
                raise CampaignValidationError(f"prediction NPZ key allowlist mismatch: {path}")
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise CampaignValidationError(f"cannot safely load predictions {path}: {exc}") from exc
    for name, array in arrays.items():
        if array.dtype.hasobject:
            raise CampaignValidationError(f"object dtype is forbidden for prediction array {name!r}")
        record = array_records[name]
        if not isinstance(record, Mapping) or set(record) != {"dtype", "shape", "sha256"}:
            raise CampaignValidationError(f"invalid prediction array record for {name!r}")
        if record.get("dtype") != str(array.dtype) or record.get("shape") != list(array.shape):
            raise CampaignValidationError(f"prediction array dtype/shape mismatch for {name!r}")
        digest = _validate_sha256(record.get("sha256"), label=f"arrays.{name}.sha256")
        if _hash_array(array) != digest:
            raise CampaignValidationError(f"prediction array hash mismatch for {name!r}")
    return arrays


def _validate_baseline_evidence(value: Any, *, run_id: str) -> dict[str, dict[str, float]]:
    if not isinstance(value, Mapping) or set(value) != set(BASELINE_METHODS):
        raise CampaignValidationError(f"{run_id} baseline inner-validation evidence is incomplete")
    output: dict[str, dict[str, float]] = {}
    for method in BASELINE_METHODS:
        record = value[method]
        if not isinstance(record, Mapping) or set(record) != {"r2", "mse"}:
            raise CampaignValidationError(f"{run_id} lacks compact evidence for {method}")
        try:
            r2 = float(record["r2"])
            mse = float(record["mse"])
        except (TypeError, ValueError) as exc:
            raise CampaignValidationError(f"{run_id} has nonnumeric compact evidence for {method}") from exc
        if not math.isfinite(r2) or not math.isfinite(mse) or mse < 0.0:
            raise CampaignValidationError(f"{run_id} has invalid compact evidence for {method}")
        output[method] = {"r2": r2, "mse": mse}
    return output


def _validate_job_selected_baseline_evidence(
    value: Any,
    *,
    baseline: Mapping[str, Mapping[str, float]],
    lock: Mapping[str, Any],
    run_id: str,
) -> tuple[str, dict[str, Any]]:
    expected_fields = {
        "name",
        "selection_scope",
        "selection_rule",
        "selected_validation",
        "name_sha256",
        "outer_prediction_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise CampaignValidationError(
            f"{run_id} job-selected baseline evidence fields mismatch"
        )
    name = value.get("name")
    if name not in BASELINE_METHODS:
        raise CampaignValidationError(f"{run_id} job-selected baseline name is invalid")
    expected_name = min(
        BASELINE_METHODS,
        key=lambda method: (
            -float(baseline[method]["r2"]),
            float(baseline[method]["mse"]),
            method,
        ),
    )
    if name != expected_name:
        raise CampaignValidationError(
            f"{run_id} job-selected baseline does not replay from inner validation"
        )
    if value.get("selection_scope") != "per_outer_job_inner_validation_only":
        raise CampaignValidationError(
            f"{run_id} job-selected baseline selection scope mismatch"
        )
    if value.get("selection_rule") != lock["training_contract"]["baseline_selection_rule"]:
        raise CampaignValidationError(
            f"{run_id} job-selected baseline selection rule mismatch"
        )
    if value.get("selected_validation") != baseline[name]:
        raise CampaignValidationError(
            f"{run_id} job-selected baseline validation evidence mismatch"
        )
    if value.get("name_sha256") != _hash_strings([name]):
        raise CampaignValidationError(
            f"{run_id} job-selected baseline name hash mismatch"
        )
    _validate_sha256(
        value.get("outer_prediction_sha256"),
        label=f"{run_id}.job_selected_baseline.outer_prediction_sha256",
    )
    return str(name), dict(value)


def _finite_float(value: Any, *, label: str, nonnegative: bool = False) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError) as exc:
        raise CampaignValidationError(f"{label} must be numeric") from exc
    if not math.isfinite(output) or (nonnegative and output < 0.0):
        qualifier = "finite and non-negative" if nonnegative else "finite"
        raise CampaignValidationError(f"{label} must be {qualifier}")
    return output


def _stack_solver_contract(lock: Mapping[str, Any]) -> dict[str, Any]:
    implementation = lock["primary_method_contract"]["implementation_contract"]
    return {
        name: implementation[name]
        for name in (
            "solver",
            "max_iterations",
            "ftol",
            "constraint_tolerance",
            "initialization",
            "objective_weighting",
        )
    }


def _validate_crossfit_audit(
    value: Any,
    *,
    component_ids: Sequence[str],
    expected_folds: int,
    training_seed: int,
    hash_contract: str,
    run_id: str,
    label: str,
) -> None:
    expected_fields = {
        "n_folds",
        "training_seed",
        "assignment_contract",
        "fold_ids_sha256",
        "folds",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise CampaignValidationError(f"{run_id} {label} cross-fit audit fields mismatch")
    if value.get("n_folds") != expected_folds or value.get("training_seed") != training_seed:
        raise CampaignValidationError(f"{run_id} {label} cross-fit design mismatch")
    if value.get("assignment_contract") != hash_contract:
        raise CampaignValidationError(f"{run_id} {label} cross-fit hash contract mismatch")
    components = np.asarray(component_ids, dtype=str)
    if components.ndim != 1 or components.size == 0 or any(not item for item in components):
        raise CampaignValidationError(f"{run_id} {label} cross-fit components are malformed")
    unique_components = sorted(set(components.tolist()))
    if len(unique_components) < expected_folds:
        raise CampaignValidationError(f"{run_id} {label} has too few cross-fit components")
    prefix = str(int(training_seed)).encode("utf-8") + b"\x00"
    component_fold = {
        component_id: int.from_bytes(
            hashlib.sha256(prefix + component_id.encode("utf-8")).digest()[:8],
            "big",
        )
        % expected_folds
        for component_id in unique_components
    }
    fold_ids = np.asarray(
        [component_fold[component_id] for component_id in components.tolist()],
        dtype=np.int64,
    )
    if set(fold_ids.tolist()) != set(range(expected_folds)):
        raise CampaignValidationError(f"{run_id} {label} deterministic cross-fit has an empty fold")
    expected_records: list[dict[str, int]] = []
    for fold in range(expected_folds):
        validation = fold_ids == fold
        training = ~validation
        expected_records.append(
            {
                "fold": fold,
                "n_training_genes": int(np.sum(training)),
                "n_validation_genes": int(np.sum(validation)),
                "n_training_components": len(set(components[training].tolist())),
                "n_validation_components": len(set(components[validation].tolist())),
            }
        )
    expected_audit = {
        "n_folds": int(expected_folds),
        "training_seed": int(training_seed),
        "assignment_contract": hash_contract,
        "fold_ids_sha256": _hash_array(fold_ids.astype("<i8")),
        "folds": expected_records,
    }
    if dict(value) != expected_audit:
        raise CampaignValidationError(
            f"{run_id} {label} cross-fit audit does not replay from component membership"
        )


def _validate_distance_stack_fit(
    value: Any,
    *,
    static: bool,
    lock: Mapping[str, Any],
    scipy_version: str,
    run_id: str,
    label: str,
) -> dict[str, Any]:
    fields = {
        "schema",
        "static",
        "method_order",
        "q10",
        "q90",
        "a",
        "b",
        "endpoint_weights_z0",
        "endpoint_weights_z1",
        "gate_lambda",
        "objective_value",
        "solver",
        "solver_scipy_version",
        "solver_success",
        "solver_status",
        "solver_message",
        "solver_iterations",
        "solver_contract",
        "constraint_residuals",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise CampaignValidationError(f"{run_id} {label} fit fields mismatch")
    if value.get("schema") != "DepPLM-distance-stack-fit-v1" or value.get("static") is not static:
        raise CampaignValidationError(f"{run_id} {label} fit schema/static flag mismatch")
    if value.get("method_order") != list(STACK_METHOD_ORDER):
        raise CampaignValidationError(f"{run_id} {label} method order mismatch")
    q10 = _finite_float(value.get("q10"), label=f"{run_id}.{label}.q10")
    q90 = _finite_float(value.get("q90"), label=f"{run_id}.{label}.q90")
    if q10 < 0.0 or q90 > 2.0 or q90 <= q10:
        raise CampaignValidationError(f"{run_id} {label} distance quantiles are invalid")
    vectors: dict[str, np.ndarray] = {}
    for field in ("a", "b", "endpoint_weights_z0", "endpoint_weights_z1"):
        try:
            vector = np.asarray(value.get(field), dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise CampaignValidationError(f"{run_id} {label} {field} is not numeric") from exc
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise CampaignValidationError(f"{run_id} {label} {field} must be a finite length-three vector")
        vectors[field] = vector
    if not np.array_equal(vectors["a"], vectors["endpoint_weights_z0"]):
        raise CampaignValidationError(f"{run_id} {label} z=0 endpoint does not equal a")
    if not np.allclose(
        vectors["a"] + vectors["b"],
        vectors["endpoint_weights_z1"],
        rtol=0.0,
        atol=1e-15,
    ):
        raise CampaignValidationError(f"{run_id} {label} z=1 endpoint does not equal a+b")
    gate_lambda = _finite_float(
        value.get("gate_lambda"), label=f"{run_id}.{label}.gate_lambda", nonnegative=True
    )
    if static and gate_lambda != 0.0:
        raise CampaignValidationError(f"{run_id} {label} static fit has a lambda dimension")
    allowed_lambdas = {
        float(item) for item in lock["primary_method_contract"]["candidate_grid"]["gate_lambda"]
    }
    if not static and gate_lambda not in allowed_lambdas:
        raise CampaignValidationError(f"{run_id} {label} gate lambda is outside the frozen grid")
    _finite_float(value.get("objective_value"), label=f"{run_id}.{label}.objective_value", nonnegative=True)
    solver_contract = _stack_solver_contract(lock)
    if value.get("solver_contract") != solver_contract or value.get("solver") != solver_contract["solver"]:
        raise CampaignValidationError(f"{run_id} {label} solver contract mismatch")
    if value.get("solver_scipy_version") != scipy_version:
        raise CampaignValidationError(f"{run_id} {label} SciPy version mismatch")
    if value.get("solver_success") is not True:
        raise CampaignValidationError(f"{run_id} {label} solver did not succeed")
    _strict_nonnegative_int(value.get("solver_status"), label=f"{run_id}.{label}.solver_status")
    _strict_nonnegative_int(
        value.get("solver_iterations"), label=f"{run_id}.{label}.solver_iterations", positive=True
    )
    if not isinstance(value.get("solver_message"), str) or not value["solver_message"]:
        raise CampaignValidationError(f"{run_id} {label} solver message is missing")
    residual_fields = {
        "abs_sum_a_minus_one",
        "abs_sum_b",
        "minimum_endpoint_weight",
        "global_slope",
        "weighted_knn_slope",
    }
    residuals = value.get("constraint_residuals")
    if not isinstance(residuals, Mapping) or set(residuals) != residual_fields:
        raise CampaignValidationError(f"{run_id} {label} constraint residual fields mismatch")
    observed = {
        name: _finite_float(residuals[name], label=f"{run_id}.{label}.{name}")
        for name in residual_fields
    }
    expected = {
        "abs_sum_a_minus_one": float(abs(np.sum(vectors["a"]) - 1.0)),
        "abs_sum_b": float(abs(np.sum(vectors["b"]))),
        "minimum_endpoint_weight": float(
            min(np.min(vectors["endpoint_weights_z0"]), np.min(vectors["endpoint_weights_z1"]))
        ),
        "global_slope": float(vectors["b"][0]),
        "weighted_knn_slope": float(vectors["b"][1]),
    }
    if any(not math.isclose(observed[name], expected[name], rel_tol=0.0, abs_tol=1e-15) for name in expected):
        raise CampaignValidationError(f"{run_id} {label} constraint residuals do not replay")
    tolerance = float(solver_contract["constraint_tolerance"])
    valid = (
        observed["abs_sum_a_minus_one"] <= tolerance
        and observed["abs_sum_b"] <= tolerance
        and observed["minimum_endpoint_weight"] >= -tolerance
        and (static or observed["global_slope"] >= -tolerance)
        and (static or observed["weighted_knn_slope"] <= tolerance)
    )
    if not valid or (static and not np.allclose(vectors["b"], 0.0, rtol=0.0, atol=tolerance)):
        raise CampaignValidationError(f"{run_id} {label} constrained solution violates the lock")
    return dict(value)


def _candidate_selection_key(candidate: Mapping[str, Any], *, run_id: str, label: str) -> tuple[Any, ...]:
    metrics = candidate.get("validation_metrics")
    if not isinstance(metrics, Mapping):
        raise CampaignValidationError(f"{run_id} {label} validation metrics are missing")
    r2 = _finite_float(metrics.get("r2"), label=f"{run_id}.{label}.r2")
    mse = _finite_float(metrics.get("mse"), label=f"{run_id}.{label}.mse", nonnegative=True)
    epoch = _strict_nonnegative_int(candidate.get("epoch"), label=f"{run_id}.{label}.epoch")
    config = candidate.get("config")
    if not isinstance(config, Mapping):
        raise CampaignValidationError(f"{run_id} {label} config is missing")
    return (-r2, mse, _canonical(config), epoch)


def _validate_distance_stack_audits(
    manifest: Mapping[str, Any],
    *,
    lock: Mapping[str, Any],
    run_id: str,
    training_seed: int,
    scipy_version: str,
    traces: Mapping[str, Any],
    inner_component_ids: Sequence[str],
    outer_component_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    selection = manifest["selection"]
    expected_selection_fields = {
        "primary_method",
        "job_selected_baseline",
        "baseline_inner_validation",
        "neural_selection_rule",
        "traces",
        "traces_sha256",
        "distance_stack_audit",
        "selected_decomposition",
        "selected_matched_mean_only_ablation",
    }
    if set(selection) != expected_selection_fields:
        raise CampaignValidationError(f"{run_id} selection envelope keys mismatch")
    implementation = lock["primary_method_contract"]["implementation_contract"]
    hash_contract = implementation["component_fold_hash"]
    audit = selection.get("distance_stack_audit")
    expected_audit_fields = {
        "implementation_contract",
        "inner_crossfit",
        "selected_primary_fit",
        "selected_static_ablation_fit",
    }
    if not isinstance(audit, Mapping) or set(audit) != expected_audit_fields:
        raise CampaignValidationError(f"{run_id} inner distance-stack audit fields mismatch")
    if audit.get("implementation_contract") != implementation:
        raise CampaignValidationError(f"{run_id} distance-stack implementation contract mismatch")
    inner_crossfit = audit.get("inner_crossfit")
    expected_inner_crossfit_fields = {
        "inner_component_crossfit",
        "solver_contract",
        "crossfit_hash_contract",
    }
    if not isinstance(inner_crossfit, Mapping) or set(inner_crossfit) != expected_inner_crossfit_fields:
        raise CampaignValidationError(
            f"{run_id} distance stack inner-crossfit evidence contract mismatch"
        )
    if inner_crossfit.get("solver_contract") != _stack_solver_contract(lock):
        raise CampaignValidationError(
            f"{run_id} distance stack inner-crossfit solver contract mismatch"
        )
    if inner_crossfit.get("crossfit_hash_contract") != hash_contract:
        raise CampaignValidationError(
            f"{run_id} distance stack inner-crossfit hash contract mismatch"
        )
    _validate_crossfit_audit(
        inner_crossfit.get("inner_component_crossfit"),
        component_ids=inner_component_ids,
        expected_folds=int(lock["primary_method_contract"]["inner_crossfit_folds"]),
        training_seed=training_seed,
        hash_contract=hash_contract,
        run_id=run_id,
        label="inner",
    )

    grid = lock["primary_method_contract"]["candidate_grid"]
    primary_trace = traces.get(PRIMARY_METHOD)
    static_trace = traces.get("STATIC-CONVEX-STACK")
    if not isinstance(primary_trace, Mapping) or set(primary_trace) != {"candidates", "selected", "selection_rule"}:
        raise CampaignValidationError(f"{run_id} primary selection trace fields mismatch")
    if primary_trace.get("selection_rule") != lock["primary_method_contract"]["selection_rule"]:
        raise CampaignValidationError(f"{run_id} primary selection rule mismatch")
    if not isinstance(static_trace, Mapping) or set(static_trace) != {"candidates", "selected", "lambda_dimension"}:
        raise CampaignValidationError(f"{run_id} static selection trace fields mismatch")
    if static_trace.get("lambda_dimension") != "absent_by_registered_design":
        raise CampaignValidationError(f"{run_id} static stack has an illicit lambda dimension")

    expected_primary_configs = {
        _canonical({
            "neighbors": int(neighbors),
            "temperature": float(temperature),
            "alpha": float(alpha),
            "gate_lambda": float(gate_lambda),
            "static": False,
        })
        for neighbors in grid["knn_neighbors"]
        for temperature in grid["knn_temperature"]
        for alpha in grid["ridge_alpha"]
        for gate_lambda in grid["gate_lambda"]
    }
    expected_static_configs = {
        _canonical({
            "neighbors": int(neighbors),
            "temperature": float(temperature),
            "alpha": float(alpha),
            "static": True,
        })
        for neighbors in grid["knn_neighbors"]
        for temperature in grid["knn_temperature"]
        for alpha in grid["ridge_alpha"]
    }
    selected_fits: dict[str, dict[str, Any]] = {}
    for method, trace, expected_configs, static in (
        (PRIMARY_METHOD, primary_trace, expected_primary_configs, False),
        ("STATIC-CONVEX-STACK", static_trace, expected_static_configs, True),
    ):
        candidates = trace.get("candidates")
        selected = trace.get("selected")
        if not isinstance(candidates, list) or not candidates or not isinstance(selected, Mapping):
            raise CampaignValidationError(f"{run_id} {method} candidates/selection are malformed")
        if any(not isinstance(candidate, Mapping) or set(candidate) != {"config", "epoch", "validation_metrics", "fit"} for candidate in candidates):
            raise CampaignValidationError(f"{run_id} {method} candidate fields mismatch")
        configs = [_canonical(candidate["config"]) for candidate in candidates]
        if len(configs) != len(set(configs)) or set(configs) != expected_configs:
            raise CampaignValidationError(f"{run_id} {method} candidate grid mismatch")
        for position, candidate in enumerate(candidates):
            _candidate_selection_key(candidate, run_id=run_id, label=f"{method}.candidate[{position}]")
            _validate_distance_stack_fit(
                candidate["fit"],
                static=static,
                lock=lock,
                scipy_version=scipy_version,
                run_id=run_id,
                label=f"{method}.candidate[{position}].fit",
            )
            if not math.isclose(
                float(candidate["fit"]["gate_lambda"]),
                0.0 if static else float(candidate["config"]["gate_lambda"]),
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise CampaignValidationError(f"{run_id} {method} candidate lambda mismatch")
        expected_selected = min(
            candidates,
            key=lambda candidate: _candidate_selection_key(
                candidate, run_id=run_id, label=f"{method}.selection"
            ),
        )
        if dict(selected) != dict(expected_selected):
            raise CampaignValidationError(f"{run_id} {method} selected candidate does not replay")
        selected_fits[method] = dict(selected["fit"])
    if audit.get("selected_primary_fit") != selected_fits[PRIMARY_METHOD]:
        raise CampaignValidationError(f"{run_id} selected primary fit differs across envelopes")
    if audit.get("selected_static_ablation_fit") != selected_fits["STATIC-CONVEX-STACK"]:
        raise CampaignValidationError(f"{run_id} selected static fit differs across envelopes")

    primary_selection = selection.get("primary_method")
    if not isinstance(primary_selection, Mapping) or primary_selection != {
        "name": PRIMARY_METHOD,
        "selection_scope": "inner_component_heldout_validation_only",
        "selection_rule": lock["primary_method_contract"]["selection_rule"],
        "selected": primary_trace["selected"],
    }:
        raise CampaignValidationError(f"{run_id} primary selection envelope mismatch")

    final_refits = manifest.get("final_refits")
    if not isinstance(final_refits, Mapping):
        raise CampaignValidationError(f"{run_id} final refits are missing")
    final_audit = final_refits.get("distance_stack_audit")
    expected_final_fields = {
        "outer_component_crossfit",
        "primary_config",
        "primary_fit",
        "static_ablation_config",
        "static_ablation_fit",
        "per_query_weight_semantics",
        "weight_column_order",
    }
    if not isinstance(final_audit, Mapping) or set(final_audit) != expected_final_fields:
        raise CampaignValidationError(f"{run_id} final distance-stack audit fields mismatch")
    if final_audit.get("per_query_weight_semantics") != "a_m_plus_b_m_times_distance_gate_z":
        raise CampaignValidationError(f"{run_id} per-query weight semantics mismatch")
    if final_audit.get("weight_column_order") != list(STACK_METHOD_ORDER):
        raise CampaignValidationError(f"{run_id} final weight column order mismatch")
    _validate_crossfit_audit(
        final_audit.get("outer_component_crossfit"),
        component_ids=outer_component_ids,
        expected_folds=int(lock["primary_method_contract"]["outer_refit_crossfit_folds"]),
        training_seed=training_seed,
        hash_contract=hash_contract,
        run_id=run_id,
        label="outer",
    )
    final_primary_fit = _validate_distance_stack_fit(
        final_audit.get("primary_fit"),
        static=False,
        lock=lock,
        scipy_version=scipy_version,
        run_id=run_id,
        label="outer.primary_fit",
    )
    final_static_fit = _validate_distance_stack_fit(
        final_audit.get("static_ablation_fit"),
        static=True,
        lock=lock,
        scipy_version=scipy_version,
        run_id=run_id,
        label="outer.static_fit",
    )
    selected_primary_config = primary_trace["selected"]["config"]
    selected_static_config = static_trace["selected"]["config"]
    if final_audit.get("primary_config") != selected_primary_config:
        raise CampaignValidationError(f"{run_id} outer primary config differs from selection")
    if final_audit.get("static_ablation_config") != selected_static_config:
        raise CampaignValidationError(f"{run_id} outer static config differs from selection")
    if float(final_primary_fit["gate_lambda"]) != float(selected_primary_config["gate_lambda"]):
        raise CampaignValidationError(f"{run_id} outer primary lambda differs from inner selection")
    if float(final_static_fit["gate_lambda"]) != 0.0:
        raise CampaignValidationError(f"{run_id} outer static fit has a lambda dimension")
    return {
        PRIMARY_METHOD: final_primary_fit,
        "STATIC-CONVEX-STACK": final_static_fit,
    }


def _strict_nonnegative_int(value: Any, *, label: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CampaignValidationError(f"{label} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise CampaignValidationError(f"{label} must be {qualifier}")
    return value


def _expand_locked_grid(grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    names = sorted(grid)
    return [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*(grid[name] for name in names))
    ]


def _validate_gene_mean_metrics(value: Any, *, label: str) -> dict[str, Any]:
    fields = {"mse", "mae", "r2", "spearman", "pearson", "n"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise CampaignValidationError(f"{label} gene-mean metric fields mismatch")
    parsed: dict[str, Any] = {
        "mse": _finite_float(value["mse"], label=f"{label}.mse", nonnegative=True),
        "mae": _finite_float(value["mae"], label=f"{label}.mae", nonnegative=True),
        "r2": _finite_float(value["r2"], label=f"{label}.r2"),
        "n": _strict_nonnegative_int(value["n"], label=f"{label}.n", positive=True),
    }
    for name in ("spearman", "pearson"):
        parsed[name] = (
            None
            if value[name] is None
            else _finite_float(value[name], label=f"{label}.{name}")
        )
    return {name: parsed[name] for name in value}


def _validate_mean_mlp_training_record(
    value: Any,
    *,
    expected_config: Mapping[str, Any],
    expected_seed: int,
    expected_epochs: int,
    lock: Mapping[str, Any],
    validation_required: bool,
    run_id: str,
    label: str,
) -> dict[str, Any]:
    expected_fields = {
        "config",
        "seed",
        "epochs_requested",
        "epochs_executed",
        "selected_epoch",
        "validation_metrics",
        "history",
        "gene_subsample",
        "optimizer",
        "objective",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise CampaignValidationError(f"{run_id} {label} mean-MLP training fields mismatch")
    if value.get("config") != expected_config:
        raise CampaignValidationError(f"{run_id} {label} mean-MLP config mismatch")
    seed = _strict_nonnegative_int(value.get("seed"), label=f"{run_id}.{label}.seed")
    if seed != int(expected_seed):
        raise CampaignValidationError(f"{run_id} {label} mean-MLP seed mismatch")
    epochs_requested = _strict_nonnegative_int(
        value.get("epochs_requested"),
        label=f"{run_id}.{label}.epochs_requested",
        positive=True,
    )
    epochs_executed = _strict_nonnegative_int(
        value.get("epochs_executed"),
        label=f"{run_id}.{label}.epochs_executed",
        positive=True,
    )
    selected_epoch = _strict_nonnegative_int(
        value.get("selected_epoch"),
        label=f"{run_id}.{label}.selected_epoch",
        positive=True,
    )
    if (
        epochs_requested != expected_epochs
        or epochs_executed > epochs_requested
        or selected_epoch > epochs_executed
    ):
        raise CampaignValidationError(f"{run_id} {label} mean-MLP epoch accounting failed")
    if value.get("gene_subsample") != lock["training_contract"]["gene_subsample"]:
        raise CampaignValidationError(f"{run_id} {label} mean-MLP gene subsample mismatch")
    if value.get("optimizer") != lock["training_contract"]["optimizer"]:
        raise CampaignValidationError(f"{run_id} {label} mean-MLP optimizer mismatch")
    if value.get("objective") != "gene_mean_mse":
        raise CampaignValidationError(f"{run_id} {label} mean-MLP objective mismatch")
    history = value.get("history")
    if not isinstance(history, list) or len(history) != epochs_executed:
        raise CampaignValidationError(f"{run_id} {label} mean-MLP history length mismatch")
    expected_history_fields = (
        {"epoch", "train_mean_mse", "validation"}
        if validation_required
        else {"epoch", "train_mean_mse"}
    )
    for position, record in enumerate(history, start=1):
        if not isinstance(record, Mapping) or set(record) != expected_history_fields:
            raise CampaignValidationError(
                f"{run_id} {label} mean-MLP history[{position - 1}] fields mismatch"
            )
        if record.get("epoch") != position:
            raise CampaignValidationError(f"{run_id} {label} mean-MLP history epoch mismatch")
        _finite_float(
            record.get("train_mean_mse"),
            label=f"{run_id}.{label}.history[{position - 1}].train_mean_mse",
            nonnegative=True,
        )
        if validation_required:
            _validate_gene_mean_metrics(
                record.get("validation"),
                label=f"{run_id}.{label}.history[{position - 1}].validation",
            )
    if validation_required:
        selected_metrics = value.get("validation_metrics")
        if (
            not isinstance(selected_metrics, Mapping)
            or selected_metrics != history[selected_epoch - 1]["validation"]
        ):
            raise CampaignValidationError(
                f"{run_id} {label} mean-MLP selected validation does not replay"
            )
    elif value.get("validation_metrics") is not None or not (
        selected_epoch == epochs_executed == epochs_requested
    ):
        raise CampaignValidationError(f"{run_id} {label} mean-MLP final refit did not run fully")
    return dict(value)


def _validate_decomp_training_record(
    value: Any,
    *,
    expected_config: Mapping[str, Any],
    expected_mode: str,
    expected_seed: int,
    expected_epochs_requested: int,
    expected_epochs_executed: int | None,
    lock: Mapping[str, Any],
    validation_required: bool,
    expected_early_stopping: bool,
    run_id: str,
    label: str,
) -> dict[str, Any]:
    expected_fields = {
        "config",
        "training_mode",
        "seed",
        "epochs_requested",
        "epochs_executed",
        "selected_epoch",
        "validation_metrics",
        "history",
        "batch_size",
        "gene_subsample",
        "optimizer",
        "objective",
        "context_residual_loss_weight_effective",
        "model_architecture_sha256",
        "active_trainable_parameter_count",
        "active_trainable_parameter_names",
        "mean_path_trainable_parameter_count",
        "residual_path_trainable_parameter_count",
        "optimizer_steps_per_epoch",
        "optimizer_steps_requested",
        "optimizer_steps_executed",
        "gene_sampling_schedule_sha256",
        "cell_batch_schedule_sha256",
        "early_stopping_enabled",
        "raw_residual_centered_during_training",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise CampaignValidationError(f"{run_id} {label} decomposition training fields mismatch")
    if value.get("config") != expected_config or value.get("training_mode") != expected_mode:
        raise CampaignValidationError(f"{run_id} {label} decomposition config/mode mismatch")
    seed = _strict_nonnegative_int(value.get("seed"), label=f"{run_id}.{label}.seed")
    if seed != int(expected_seed):
        raise CampaignValidationError(f"{run_id} {label} decomposition seed mismatch")
    epochs_requested = _strict_nonnegative_int(
        value.get("epochs_requested"),
        label=f"{run_id}.{label}.epochs_requested",
        positive=True,
    )
    epochs_executed = _strict_nonnegative_int(
        value.get("epochs_executed"),
        label=f"{run_id}.{label}.epochs_executed",
        positive=True,
    )
    selected_epoch = _strict_nonnegative_int(
        value.get("selected_epoch"),
        label=f"{run_id}.{label}.selected_epoch",
        positive=True,
    )
    if (
        epochs_requested != int(expected_epochs_requested)
        or epochs_executed > epochs_requested
        or selected_epoch > epochs_executed
        or (
            expected_epochs_executed is not None
            and epochs_executed != int(expected_epochs_executed)
        )
    ):
        raise CampaignValidationError(f"{run_id} {label} decomposition epoch accounting failed")

    training_contract = lock["training_contract"]
    if value.get("batch_size") != training_contract["batch_size"]:
        raise CampaignValidationError(f"{run_id} {label} decomposition batch size mismatch")
    if value.get("gene_subsample") != training_contract["gene_subsample"]:
        raise CampaignValidationError(f"{run_id} {label} decomposition gene subsample mismatch")
    if value.get("optimizer") != training_contract["optimizer"]:
        raise CampaignValidationError(f"{run_id} {label} decomposition optimizer mismatch")

    mean_only = expected_mode == "DECOMP-MEAN-ONLY"
    if expected_mode not in {"DECOMP-MEAN-ONLY", SECONDARY_CONTEXT_METHOD}:
        raise CampaignValidationError(f"{run_id} {label} unregistered decomposition mode")
    expected_objective = (
        "gene_mean_mse_with_residual_loss_fixed_zero"
        if mean_only
        else "separate_gene_mean_mse_plus_gene_centered_context_residual_mse"
    )
    expected_residual_weight = (
        0.0 if mean_only else float(expected_config["context_residual_loss_weight"])
    )
    if value.get("objective") != expected_objective:
        raise CampaignValidationError(f"{run_id} {label} decomposition objective mismatch")
    if _finite_float(
        value.get("context_residual_loss_weight_effective"),
        label=f"{run_id}.{label}.context_residual_loss_weight_effective",
        nonnegative=True,
    ) != expected_residual_weight:
        raise CampaignValidationError(f"{run_id} {label} effective residual loss weight mismatch")
    for field in (
        "model_architecture_sha256",
        "gene_sampling_schedule_sha256",
        "cell_batch_schedule_sha256",
    ):
        _validate_sha256(value.get(field), label=f"{run_id}.{label}.{field}")
    active = _strict_nonnegative_int(
        value.get("active_trainable_parameter_count"),
        label=f"{run_id}.{label}.active_trainable_parameter_count",
        positive=True,
    )
    active_names = value.get("active_trainable_parameter_names")
    if (
        not isinstance(active_names, list)
        or not active_names
        or any(
            not isinstance(name, str)
            or not name
            or any(
                not part or re.fullmatch(r"[A-Za-z0-9_]+", part) is None
                for part in name.split(".")
            )
            for name in active_names
        )
        or active_names != sorted(set(active_names))
    ):
        raise CampaignValidationError(
            f"{run_id} {label} active trainable parameter names are noncanonical"
        )
    mean_parameters = _strict_nonnegative_int(
        value.get("mean_path_trainable_parameter_count"),
        label=f"{run_id}.{label}.mean_path_trainable_parameter_count",
        positive=True,
    )
    residual_parameters = _strict_nonnegative_int(
        value.get("residual_path_trainable_parameter_count"),
        label=f"{run_id}.{label}.residual_path_trainable_parameter_count",
    )
    mean_prefixes = ("sequence_encoder.", "mean_head.")
    residual_prefixes = ("cell_encoder.", "gene_factor.", "cell_factor.")
    if any(
        not name.startswith((*mean_prefixes, *residual_prefixes))
        for name in active_names
    ):
        raise CampaignValidationError(
            f"{run_id} {label} active trainable parameter path is unregistered"
        )
    mean_names = [name for name in active_names if name.startswith(mean_prefixes)]
    residual_names = [
        name for name in active_names if name.startswith(residual_prefixes)
    ]
    if (
        active != mean_parameters + residual_parameters
        or not mean_names
        or (
            mean_only
            and (
                residual_parameters != 0
                or active != mean_parameters
                or residual_names
                or len(mean_names) != len(active_names)
            )
        )
        or (
            not mean_only
            and (residual_parameters <= 0 or not residual_names)
        )
    ):
        raise CampaignValidationError(f"{run_id} {label} trainable parameter isolation failed")
    steps_per_epoch = _strict_nonnegative_int(
        value.get("optimizer_steps_per_epoch"),
        label=f"{run_id}.{label}.optimizer_steps_per_epoch",
        positive=True,
    )
    requested_steps = _strict_nonnegative_int(
        value.get("optimizer_steps_requested"),
        label=f"{run_id}.{label}.optimizer_steps_requested",
        positive=True,
    )
    executed_steps = _strict_nonnegative_int(
        value.get("optimizer_steps_executed"),
        label=f"{run_id}.{label}.optimizer_steps_executed",
        positive=True,
    )
    if (
        requested_steps != epochs_requested * steps_per_epoch
        or executed_steps != epochs_executed * steps_per_epoch
    ):
        raise CampaignValidationError(f"{run_id} {label} optimizer-step accounting failed")
    if value.get("early_stopping_enabled") is not expected_early_stopping:
        raise CampaignValidationError(f"{run_id} {label} early-stopping flag mismatch")
    if value.get("raw_residual_centered_during_training") is not False:
        raise CampaignValidationError(f"{run_id} {label} raw residual centering is forbidden")

    history = value.get("history")
    if not isinstance(history, list) or len(history) != epochs_executed:
        raise CampaignValidationError(f"{run_id} {label} decomposition history length mismatch")
    history_fields = {
        "epoch",
        "train_loss",
        "train_mean_mse",
        "train_raw_residual_mse",
        *( ["validation"] if validation_required else [] ),
    }
    mean_weight = float(expected_config["gene_mean_loss_weight"])
    for position, record in enumerate(history, start=1):
        if not isinstance(record, Mapping) or set(record) != history_fields:
            raise CampaignValidationError(
                f"{run_id} {label} decomposition history[{position - 1}] fields mismatch"
            )
        if record.get("epoch") != position:
            raise CampaignValidationError(f"{run_id} {label} decomposition history epoch mismatch")
        train_loss = _finite_float(
            record.get("train_loss"),
            label=f"{run_id}.{label}.history[{position - 1}].train_loss",
            nonnegative=True,
        )
        mean_loss = _finite_float(
            record.get("train_mean_mse"),
            label=f"{run_id}.{label}.history[{position - 1}].train_mean_mse",
            nonnegative=True,
        )
        residual_loss = _finite_float(
            record.get("train_raw_residual_mse"),
            label=f"{run_id}.{label}.history[{position - 1}].train_raw_residual_mse",
            nonnegative=True,
        )
        if mean_only and residual_loss != 0.0:
            raise CampaignValidationError(f"{run_id} {label} mean-only residual loss is nonzero")
        if not math.isclose(
            train_loss,
            mean_weight * mean_loss + expected_residual_weight * residual_loss,
            rel_tol=1e-6,
            abs_tol=1e-8,
        ):
            raise CampaignValidationError(f"{run_id} {label} training objective does not replay")
        if validation_required:
            _validate_gene_mean_metrics(
                record.get("validation"),
                label=f"{run_id}.{label}.history[{position - 1}].validation",
            )
    if validation_required:
        validation_metrics = value.get("validation_metrics")
        if (
            not isinstance(validation_metrics, Mapping)
            or validation_metrics != history[selected_epoch - 1]["validation"]
        ):
            raise CampaignValidationError(
                f"{run_id} {label} selected validation does not replay from history"
            )
    elif value.get("validation_metrics") is not None or not (
        selected_epoch == epochs_executed == epochs_requested
    ):
        raise CampaignValidationError(f"{run_id} {label} final refit did not execute fully")
    return dict(value)


def _validate_baseline_trace_evidence(
    baseline: Mapping[str, Mapping[str, float]],
    traces: Mapping[str, Any],
    *,
    lock: Mapping[str, Any],
    run_id: str,
    training_seed: int,
) -> None:
    for method in BASELINE_METHODS:
        trace = traces.get(method)
        selected = trace.get("selected") if isinstance(trace, Mapping) else None
        metrics = selected.get("validation_metrics") if isinstance(selected, Mapping) else None
        if not isinstance(metrics, Mapping):
            raise CampaignValidationError(f"{run_id} {method} selected validation is missing")
        compact = {
            "r2": _finite_float(metrics.get("r2"), label=f"{run_id}.{method}.selected.r2"),
            "mse": _finite_float(
                metrics.get("mse"),
                label=f"{run_id}.{method}.selected.mse",
                nonnegative=True,
            ),
        }
        if compact != baseline[method]:
            raise CampaignValidationError(
                f"{run_id} {method} compact baseline evidence differs from its selected trace"
            )

    primary_grid = lock["primary_method_contract"]["candidate_grid"]
    standard_configs: dict[str, set[str]] = {
        "GLOBAL-TRAIN-MEAN": {
            _canonical({"definition": "mean_of_inner_training_gene_means"})
        },
        "SEQUENCE-KNN": {
            _canonical(
                {
                    "neighbors": int(neighbors),
                    "weights": "uniform",
                    "metric": "cosine",
                }
            )
            for neighbors in lock["training_contract"]["baseline_grids"][
                "uniform_knn_neighbors"
            ]
        },
        "SEQUENCE-WEIGHTED-KNN": {
            _canonical(
                {
                    "neighbors": int(neighbors),
                    "temperature": float(temperature),
                    "weights": "softmax_cosine",
                    "formula": "exp((similarity-max_similarity)/temperature)",
                    "metric": "cosine",
                }
            )
            for neighbors in primary_grid["knn_neighbors"]
            for temperature in primary_grid["knn_temperature"]
        },
        "PLM-RIDGE": {
            _canonical(
                {
                    "alpha": float(alpha),
                    "standardization": "inner_train_only",
                    "solver": "lsqr",
                    "tol": 1e-8,
                    "max_iter": 10000,
                }
            )
            for alpha in primary_grid["ridge_alpha"]
        },
    }
    for method, expected_configs in standard_configs.items():
        trace = traces[method]
        if not isinstance(trace, Mapping) or set(trace) != {"candidates", "selected"}:
            raise CampaignValidationError(f"{run_id} {method} trace fields mismatch")
        candidates = trace.get("candidates")
        selected = trace.get("selected")
        if not isinstance(candidates, list) or not candidates or not isinstance(selected, Mapping):
            raise CampaignValidationError(f"{run_id} {method} trace is malformed")
        configs: list[str] = []
        for position, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping) or set(candidate) != {
                "config",
                "epoch",
                "validation_metrics",
            }:
                raise CampaignValidationError(
                    f"{run_id} {method} candidate[{position}] fields mismatch"
                )
            if candidate.get("epoch") != 0:
                raise CampaignValidationError(f"{run_id} {method} candidate epoch mismatch")
            _candidate_selection_key(
                candidate, run_id=run_id, label=f"{method}.candidate[{position}]"
            )
            configs.append(_canonical(candidate["config"]))
        if len(configs) != len(set(configs)) or set(configs) != expected_configs:
            raise CampaignValidationError(f"{run_id} {method} candidate grid mismatch")
        expected_selected = min(
            candidates,
            key=lambda candidate: _candidate_selection_key(
                candidate, run_id=run_id, label=f"{method}.selection"
            ),
        )
        if dict(selected) != dict(expected_selected):
            raise CampaignValidationError(f"{run_id} {method} selected candidate does not replay")

    mean_trace = traces["IDFREE-PLM-MEAN-MLP"]
    if not isinstance(mean_trace, Mapping) or set(mean_trace) != {"candidates", "selected"}:
        raise CampaignValidationError(f"{run_id} mean-MLP trace fields mismatch")
    mean_candidates = mean_trace.get("candidates")
    mean_selected = mean_trace.get("selected")
    expected_mean_configs = _expand_locked_grid(lock["training_contract"]["mean_mlp_grid"])
    if (
        not isinstance(mean_candidates, list)
        or len(mean_candidates) != len(expected_mean_configs)
        or not isinstance(mean_selected, Mapping)
    ):
        raise CampaignValidationError(f"{run_id} mean-MLP candidate grid cardinality mismatch")
    for position, (candidate, expected_config) in enumerate(
        zip(mean_candidates, expected_mean_configs, strict=True)
    ):
        if not isinstance(candidate, Mapping) or set(candidate) != {
            "config",
            "epoch",
            "validation_metrics",
            "training",
            "grid_index",
        }:
            raise CampaignValidationError(f"{run_id} mean-MLP candidate[{position}] fields mismatch")
        if candidate.get("grid_index") != position or candidate.get("config") != expected_config:
            raise CampaignValidationError(f"{run_id} mean-MLP candidate grid mismatch")
        _candidate_selection_key(
            candidate, run_id=run_id, label=f"IDFREE-PLM-MEAN-MLP.candidate[{position}]"
        )
        training = _validate_mean_mlp_training_record(
            candidate.get("training"),
            expected_config=expected_config,
            expected_seed=derive_seed(
                int(training_seed),
                "inner-mean-mlp",
                position,
                _canonical(expected_config),
            ),
            expected_epochs=int(lock["training_contract"]["inner_max_epochs"]),
            lock=lock,
            validation_required=True,
            run_id=run_id,
            label=f"IDFREE-PLM-MEAN-MLP.candidate[{position}]",
        )
        if (
            candidate.get("epoch") != training["selected_epoch"]
            or candidate.get("validation_metrics") != training["validation_metrics"]
        ):
            raise CampaignValidationError(
                f"{run_id} mean-MLP candidate[{position}] training envelope mismatch"
            )
    expected_mean_selected = min(
        mean_candidates,
        key=lambda candidate: _candidate_selection_key(
            candidate, run_id=run_id, label="IDFREE-PLM-MEAN-MLP.selection"
        ),
    )
    expected_mean_selected = {
        name: value for name, value in expected_mean_selected.items() if name != "training"
    }
    if dict(mean_selected) != expected_mean_selected:
        raise CampaignValidationError(f"{run_id} mean-MLP selected candidate does not replay")


def _validate_training_pair(
    proposed: Any,
    mean_only: Any,
    audit: Any,
    *,
    run_id: str,
    label: str,
    candidate_metadata: Mapping[str, Any] | None = None,
) -> None:
    if not isinstance(proposed, Mapping) or not isinstance(mean_only, Mapping):
        raise CampaignValidationError(f"{run_id} {label} training records are missing")
    if not isinstance(audit, Mapping):
        raise CampaignValidationError(f"{run_id} {label} matched optimizer audit is missing")
    exact_pairs = (
        "model_architecture_sha256",
        "optimizer_steps_per_epoch",
        "optimizer_steps_requested",
        "epochs_executed",
        "optimizer_steps_executed",
        "gene_sampling_schedule_sha256",
        "cell_batch_schedule_sha256",
        "mean_path_trainable_parameter_count",
    )
    for field in exact_pairs:
        if field not in proposed or proposed[field] != mean_only.get(field):
            raise CampaignValidationError(f"{run_id} {label} training mismatch: {field}")
    for record_name, record in (("proposed", proposed), ("mean_only", mean_only)):
        steps_per_epoch = _strict_nonnegative_int(
            record.get("optimizer_steps_per_epoch"),
            label=f"{run_id}.{label}.{record_name}.optimizer_steps_per_epoch",
            positive=True,
        )
        requested = _strict_nonnegative_int(
            record.get("optimizer_steps_requested"),
            label=f"{run_id}.{label}.{record_name}.optimizer_steps_requested",
            positive=True,
        )
        epochs = _strict_nonnegative_int(
            record.get("epochs_executed"),
            label=f"{run_id}.{label}.{record_name}.epochs_executed",
            positive=True,
        )
        executed = _strict_nonnegative_int(
            record.get("optimizer_steps_executed"),
            label=f"{run_id}.{label}.{record_name}.optimizer_steps_executed",
            positive=True,
        )
        if executed != epochs * steps_per_epoch or executed > requested:
            raise CampaignValidationError(f"{run_id} {label} optimizer-step accounting failed")
    for field in (
        "model_architecture_sha256",
        "gene_sampling_schedule_sha256",
        "cell_batch_schedule_sha256",
    ):
        _validate_sha256(proposed[field], label=f"{run_id}.{label}.{field}")
    proposed_active = _strict_nonnegative_int(
        proposed.get("active_trainable_parameter_count"),
        label=f"{run_id}.{label}.proposed.active_trainable_parameter_count",
        positive=True,
    )
    proposed_residual = _strict_nonnegative_int(
        proposed.get("residual_path_trainable_parameter_count"),
        label=f"{run_id}.{label}.proposed.residual_path_trainable_parameter_count",
        positive=True,
    )
    mean_path = _strict_nonnegative_int(
        proposed.get("mean_path_trainable_parameter_count"),
        label=f"{run_id}.{label}.mean_path_trainable_parameter_count",
        positive=True,
    )
    mean_only_active = _strict_nonnegative_int(
        mean_only.get("active_trainable_parameter_count"),
        label=f"{run_id}.{label}.mean_only.active_trainable_parameter_count",
        positive=True,
    )
    mean_only_residual = _strict_nonnegative_int(
        mean_only.get("residual_path_trainable_parameter_count"),
        label=f"{run_id}.{label}.mean_only.residual_path_trainable_parameter_count",
    )
    if proposed_residual <= 0 or mean_only_residual != 0 or mean_only_active != mean_path:
        raise CampaignValidationError(f"{run_id} {label} mean-only parameter isolation failed")
    try:
        effective_weight = float(mean_only["context_residual_loss_weight_effective"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CampaignValidationError(f"{run_id} {label} mean-only residual-loss audit is invalid") from exc
    if effective_weight != 0.0:
        raise CampaignValidationError(f"{run_id} {label} mean-only residual loss is nonzero")
    expected_audit: dict[str, Any] = {
        "status": "matched",
        "architecture_sha256": proposed["model_architecture_sha256"],
        "optimizer_steps_per_epoch": proposed["optimizer_steps_per_epoch"],
        "optimizer_steps_requested_each": proposed["optimizer_steps_requested"],
        "optimizer_steps_executed_each": proposed["optimizer_steps_executed"],
        "gene_sampling_schedule_sha256": proposed["gene_sampling_schedule_sha256"],
        "cell_batch_schedule_sha256": proposed["cell_batch_schedule_sha256"],
        "mean_path_trainable_parameter_count": mean_path,
        "proposed_active_trainable_parameter_count": proposed_active,
        "mean_only_active_trainable_parameter_count": mean_only_active,
    }
    if candidate_metadata is not None:
        expected_audit = {**dict(candidate_metadata), **expected_audit}
    if dict(audit) != expected_audit:
        raise CampaignValidationError(f"{run_id} {label} matched optimizer audit does not replay")


def _validate_matched_training_audits(
    manifest: Mapping[str, Any],
    *,
    lock: Mapping[str, Any],
    run_id: str,
    training_seed: int,
) -> Mapping[str, Any]:
    selection = manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise CampaignValidationError(f"{run_id} selection record is missing")
    if selection.get("neural_selection_rule") != (
        "max_inner_component_heldout_gene_mean_r2_then_min_mse_then_"
        "canonical_config"
    ):
        raise CampaignValidationError(f"{run_id} neural selection rule mismatch")
    traces = selection.get("traces")
    if not isinstance(traces, Mapping) or set(traces) != set(ALL_METHODS):
        raise CampaignValidationError(f"{run_id} selection trace method allowlist mismatch")
    proposed_trace = traces.get(SECONDARY_CONTEXT_METHOD)
    mean_only_trace = traces.get("DECOMP-MEAN-ONLY")
    if (
        not isinstance(proposed_trace, Mapping)
        or set(proposed_trace) != {"candidates", "selected"}
        or not isinstance(mean_only_trace, Mapping)
        or set(mean_only_trace)
        != {"candidates", "selected", "matched_candidate_audits", "selection_basis"}
    ):
        raise CampaignValidationError(f"{run_id} decomposition selection traces are missing")
    if mean_only_trace.get("selection_basis") != (
        "matched_to_selected_proposed_config_and_epoch"
    ):
        raise CampaignValidationError(f"{run_id} mean-only selection basis mismatch")
    proposed_candidates = proposed_trace.get("candidates")
    mean_only_candidates = mean_only_trace.get("candidates")
    audits = mean_only_trace.get("matched_candidate_audits")
    expected_configs = _expand_locked_grid(lock["training_contract"]["decomposition_grid"])
    if (
        not isinstance(proposed_candidates, list)
        or len(proposed_candidates) != len(expected_configs)
        or not isinstance(mean_only_candidates, list)
        or not isinstance(audits, list)
        or len(proposed_candidates) != len(mean_only_candidates)
        or len(proposed_candidates) != len(audits)
    ):
        raise CampaignValidationError(f"{run_id} matched candidate audit cardinality mismatch")
    for position, (proposed_candidate, mean_only_candidate, audit, expected_config) in enumerate(
        zip(
            proposed_candidates,
            mean_only_candidates,
            audits,
            expected_configs,
            strict=True,
        )
    ):
        if (
            not isinstance(proposed_candidate, Mapping)
            or set(proposed_candidate)
            != {"config", "epoch", "validation_metrics", "training", "grid_index"}
            or not isinstance(mean_only_candidate, Mapping)
            or set(mean_only_candidate)
            != {
                "config",
                "epoch",
                "validation_metrics",
                "training",
                "grid_index",
                "paired_proposed_grid_index",
                "selection_basis",
            }
        ):
            raise CampaignValidationError(f"{run_id} candidate {position} is malformed")
        grid_index = proposed_candidate.get("grid_index")
        config = proposed_candidate.get("config")
        if (
            grid_index != position
            or config != expected_config
            or grid_index != mean_only_candidate.get("grid_index")
            or config != mean_only_candidate.get("config")
        ):
            raise CampaignValidationError(f"{run_id} candidate {position} config/grid mismatch")
        if mean_only_candidate.get("paired_proposed_grid_index") != grid_index:
            raise CampaignValidationError(f"{run_id} candidate {position} paired grid mismatch")
        if mean_only_candidate.get("selection_basis") != (
            "matched_to_proposed_candidate_config_and_selected_epoch"
        ):
            raise CampaignValidationError(f"{run_id} candidate {position} selection basis mismatch")
        _candidate_selection_key(
            proposed_candidate,
            run_id=run_id,
            label=f"{SECONDARY_CONTEXT_METHOD}.candidate[{position}]",
        )
        _candidate_selection_key(
            mean_only_candidate,
            run_id=run_id,
            label=f"DECOMP-MEAN-ONLY.candidate[{position}]",
        )
        expected_seed = derive_seed(
            int(training_seed),
            "inner-decomp",
            position,
            _canonical(expected_config),
        )
        proposed_training = _validate_decomp_training_record(
            proposed_candidate.get("training"),
            expected_config=expected_config,
            expected_mode=SECONDARY_CONTEXT_METHOD,
            expected_seed=expected_seed,
            expected_epochs_requested=int(lock["training_contract"]["inner_max_epochs"]),
            expected_epochs_executed=None,
            lock=lock,
            validation_required=True,
            expected_early_stopping=True,
            run_id=run_id,
            label=f"{SECONDARY_CONTEXT_METHOD}.candidate[{position}]",
        )
        mean_only_training = _validate_decomp_training_record(
            mean_only_candidate.get("training"),
            expected_config=expected_config,
            expected_mode="DECOMP-MEAN-ONLY",
            expected_seed=expected_seed,
            expected_epochs_requested=int(lock["training_contract"]["inner_max_epochs"]),
            expected_epochs_executed=int(proposed_training["epochs_executed"]),
            lock=lock,
            validation_required=True,
            expected_early_stopping=False,
            run_id=run_id,
            label=f"DECOMP-MEAN-ONLY.candidate[{position}]",
        )
        proposed_epoch = int(proposed_training["selected_epoch"])
        if (
            proposed_candidate.get("epoch") != proposed_epoch
            or proposed_candidate.get("validation_metrics")
            != proposed_training["validation_metrics"]
        ):
            raise CampaignValidationError(
                f"{run_id} proposed candidate {position} training envelope mismatch"
            )
        if (
            mean_only_candidate.get("epoch") != proposed_epoch
            or mean_only_candidate.get("validation_metrics")
            != mean_only_training["history"][proposed_epoch - 1]["validation"]
        ):
            raise CampaignValidationError(
                f"{run_id} mean-only candidate {position} is not bound to the proposed epoch"
            )
        _validate_training_pair(
            proposed_training,
            mean_only_training,
            audit,
            run_id=run_id,
            label=f"candidate[{position}]",
            candidate_metadata={"grid_index": grid_index, "config": config},
        )
    proposed_selected = proposed_trace.get("selected")
    mean_only_selected = mean_only_trace.get("selected")
    expected_proposed_candidate = min(
        proposed_candidates,
        key=lambda candidate: _candidate_selection_key(
            candidate, run_id=run_id, label=f"{SECONDARY_CONTEXT_METHOD}.selection"
        ),
    )
    selected_position = int(expected_proposed_candidate["grid_index"])
    expected_proposed_selected = {
        name: value
        for name, value in expected_proposed_candidate.items()
        if name != "training"
    }
    expected_mean_only_selected = {
        name: value
        for name, value in mean_only_candidates[selected_position].items()
        if name != "training"
    }
    if not isinstance(proposed_selected, Mapping) or dict(proposed_selected) != expected_proposed_selected:
        raise CampaignValidationError(f"{run_id} decomposition selected candidate does not replay")
    if not isinstance(mean_only_selected, Mapping) or dict(mean_only_selected) != expected_mean_only_selected:
        raise CampaignValidationError(f"{run_id} mean-only selected candidate does not replay")
    if selection.get("selected_decomposition") != expected_proposed_selected:
        raise CampaignValidationError(f"{run_id} selected decomposition envelope mismatch")
    if selection.get("selected_matched_mean_only_ablation") != expected_mean_only_selected:
        raise CampaignValidationError(f"{run_id} selected mean-only envelope mismatch")
    final_refits = manifest.get("final_refits")
    expected_final_refit_fields = {
        "distance_stack_audit",
        "IDFREE-PLM-MEAN-MLP",
        "DECOMP-MEAN-ONLY",
        SECONDARY_CONTEXT_METHOD,
        "matched_optimizer_audit",
    }
    if not isinstance(final_refits, Mapping) or set(final_refits) != expected_final_refit_fields:
        raise CampaignValidationError(f"{run_id} final refit envelope keys mismatch")
    selected_epochs = int(expected_proposed_selected["epoch"])
    final_seed = derive_seed(int(training_seed), "outer-decomp-refit")
    final_proposed = _validate_decomp_training_record(
        final_refits.get(SECONDARY_CONTEXT_METHOD),
        expected_config=expected_proposed_selected["config"],
        expected_mode=SECONDARY_CONTEXT_METHOD,
        expected_seed=final_seed,
        expected_epochs_requested=selected_epochs,
        expected_epochs_executed=selected_epochs,
        lock=lock,
        validation_required=False,
        expected_early_stopping=False,
        run_id=run_id,
        label=f"final_refit.{SECONDARY_CONTEXT_METHOD}",
    )
    final_mean_only = _validate_decomp_training_record(
        final_refits.get("DECOMP-MEAN-ONLY"),
        expected_config=expected_mean_only_selected["config"],
        expected_mode="DECOMP-MEAN-ONLY",
        expected_seed=final_seed,
        expected_epochs_requested=selected_epochs,
        expected_epochs_executed=selected_epochs,
        lock=lock,
        validation_required=False,
        expected_early_stopping=False,
        run_id=run_id,
        label="final_refit.DECOMP-MEAN-ONLY",
    )
    _validate_training_pair(
        final_proposed,
        final_mean_only,
        final_refits.get("matched_optimizer_audit"),
        run_id=run_id,
        label="final_refit",
    )
    mean_selected = traces.get("IDFREE-PLM-MEAN-MLP", {}).get("selected")
    if not isinstance(mean_selected, Mapping):
        raise CampaignValidationError(f"{run_id} selected mean-MLP trace is missing")
    _validate_mean_mlp_training_record(
        final_refits.get("IDFREE-PLM-MEAN-MLP"),
        expected_config=mean_selected.get("config", {}),
        expected_seed=derive_seed(int(training_seed), "outer-mean-mlp-refit"),
        expected_epochs=_strict_nonnegative_int(
            mean_selected.get("epoch"),
            label=f"{run_id}.IDFREE-PLM-MEAN-MLP.selected.epoch",
            positive=True,
        ),
        lock=lock,
        validation_required=False,
        run_id=run_id,
        label="final_refit.IDFREE-PLM-MEAN-MLP",
    )
    return traces


def _manifest_campaign_evidence(
    manifest: Mapping[str, Any],
    *,
    run_id: str,
    expected_key: JobKey,
    roster_sha256: str,
    lock: Mapping[str, Any],
    scipy_version: str,
    partition: ReplayedFamilyPartition,
) -> tuple[
    dict[str, dict[str, float]],
    str,
    Mapping[str, Any],
    dict[str, dict[str, Any]],
]:
    evidence = manifest.get("campaign_evidence")
    if not isinstance(evidence, Mapping):
        raise CampaignValidationError(f"{run_id} lacks campaign_evidence")
    expected_fields = {
        "run_key",
        "campaign_roster_sha256",
        "campaign_aggregation_contract_sha256",
        "baseline_inner_validation",
        "baseline_inner_validation_sha256",
        "job_selected_baseline",
        "selection_traces_sha256",
        "outer_test_gene_ids_sha256",
        "outer_test_component_ids_sha256",
        "method_gene_mean_prediction_sha256",
        "stack_base_prediction_sha256",
        "nearest_outer_train_cosine_distance_sha256",
        "distance_gate_z_sha256",
        "distance_stack_weight_sha256",
        "validation_fitted_distance_risk_score_sha256",
        "prediction_array_manifest_sha256",
        "aggregation_status",
    }
    if set(evidence) != expected_fields:
        raise CampaignValidationError(f"{run_id} campaign_evidence keys mismatch")
    if _parse_run_key(
        evidence.get("run_key"),
        roster_sha256=roster_sha256,
        label=f"{run_id} campaign_evidence.run_key",
    ) != expected_key:
        raise CampaignValidationError(f"{run_id} campaign evidence run key mismatch")
    if evidence.get("campaign_roster_sha256") != roster_sha256:
        raise CampaignValidationError(f"{run_id} campaign evidence roster digest mismatch")
    expected_contract_hash = hashlib.sha256(_json_bytes(lock["campaign_aggregation"])).hexdigest()
    if evidence.get("campaign_aggregation_contract_sha256") != expected_contract_hash:
        raise CampaignValidationError(f"{run_id} aggregation contract hash mismatch")
    baseline = _validate_baseline_evidence(evidence["baseline_inner_validation"], run_id=run_id)
    if evidence.get("baseline_inner_validation_sha256") != hashlib.sha256(
        _json_bytes(baseline)
    ).hexdigest():
        raise CampaignValidationError(f"{run_id} compact baseline evidence hash mismatch")
    selected_baseline, selected_evidence = _validate_job_selected_baseline_evidence(
        evidence.get("job_selected_baseline"),
        baseline=baseline,
        lock=lock,
        run_id=run_id,
    )
    traces = _validate_matched_training_audits(
        manifest,
        lock=lock,
        run_id=run_id,
        training_seed=expected_key.training_seed,
    )
    _validate_baseline_trace_evidence(
        baseline,
        traces,
        lock=lock,
        run_id=run_id,
        training_seed=expected_key.training_seed,
    )
    traces_hash = hashlib.sha256(_json_bytes(traces)).hexdigest()
    if evidence.get("selection_traces_sha256") != traces_hash:
        raise CampaignValidationError(f"{run_id} selection trace hash mismatch")
    selection = manifest["selection"]
    if selection.get("traces_sha256") != traces_hash:
        raise CampaignValidationError(f"{run_id} selection envelope trace hash mismatch")
    if selection.get("baseline_inner_validation") != baseline:
        raise CampaignValidationError(f"{run_id} selection compact evidence mismatch")
    if selection.get("job_selected_baseline") != selected_evidence:
        raise CampaignValidationError(
            f"{run_id} job-selected baseline differs across evidence envelopes"
        )
    stack_fits = _validate_distance_stack_audits(
        manifest,
        lock=lock,
        run_id=run_id,
        training_seed=expected_key.training_seed,
        scipy_version=scipy_version,
        traces=traces,
        inner_component_ids=[
            partition.component_ids[int(index)]
            for index in partition.inner_train
        ],
        outer_component_ids=[
            partition.component_ids[int(index)]
            for index in partition.outer_train
        ],
    )
    method_hashes = evidence.get("method_gene_mean_prediction_sha256")
    if not isinstance(method_hashes, Mapping) or set(method_hashes) != set(ALL_METHODS):
        raise CampaignValidationError(f"{run_id} method prediction hash allowlist mismatch")
    for field in (
        "outer_test_gene_ids_sha256",
        "outer_test_component_ids_sha256",
        "nearest_outer_train_cosine_distance_sha256",
        "distance_gate_z_sha256",
        "validation_fitted_distance_risk_score_sha256",
        "prediction_array_manifest_sha256",
    ):
        _validate_sha256(evidence.get(field), label=f"{run_id}.{field}")
    weight_hashes = evidence.get("distance_stack_weight_sha256")
    if not isinstance(weight_hashes, Mapping) or set(weight_hashes) != set(STACK_METHOD_ORDER):
        raise CampaignValidationError(f"{run_id} distance-stack weight hash allowlist mismatch")
    for method, digest in weight_hashes.items():
        _validate_sha256(digest, label=f"{run_id}.{method}.distance_stack_weight_sha256")
    for method, digest in method_hashes.items():
        _validate_sha256(digest, label=f"{run_id}.{method}.prediction_sha256")
    stack_base_hashes = evidence.get("stack_base_prediction_sha256")
    if (
        not isinstance(stack_base_hashes, Mapping)
        or set(stack_base_hashes) != set(STACK_BASE_PREDICTION_ARRAY_KEYS)
    ):
        raise CampaignValidationError(
            f"{run_id} stack-base prediction hash stack allowlist mismatch"
        )
    for stack_name, method_map in STACK_BASE_PREDICTION_ARRAY_KEYS.items():
        stack_hashes = stack_base_hashes.get(stack_name)
        if not isinstance(stack_hashes, Mapping) or set(stack_hashes) != set(method_map):
            raise CampaignValidationError(
                f"{run_id} {stack_name} stack-base prediction hash method allowlist mismatch"
            )
        for method, digest in stack_hashes.items():
            _validate_sha256(
                digest,
                label=f"{run_id}.{stack_name}.{method}.stack_base_prediction_sha256",
            )
    if evidence.get("aggregation_status") != "single_job_input_only_not_aggregated":
        raise CampaignValidationError(f"{run_id} single-job aggregation status mismatch")
    return baseline, selected_baseline, evidence, stack_fits


def _validate_campaign_array_evidence(
    evidence: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    lock: Mapping[str, Any],
    *,
    run_id: str,
    job_selected_baseline: str,
) -> None:
    scalar, methods = _prediction_key_map(lock)
    expected = {
        "outer_test_gene_ids_sha256": _hash_array(
            np.asarray(arrays[scalar["gene_ids"]])
        ),
        "outer_test_component_ids_sha256": _hash_array(
            np.asarray(arrays[scalar["component_ids"]])
        ),
        "nearest_outer_train_cosine_distance_sha256": _hash_array(
            np.asarray(arrays[scalar["nearest_training_distance"]], dtype="<f8")
        ),
        "distance_gate_z_sha256": _hash_array(
            np.asarray(arrays[scalar["distance_gate_coordinate"]], dtype="<f8")
        ),
        "validation_fitted_distance_risk_score_sha256": _hash_array(
            np.asarray(arrays[scalar["selective_risk_score"]], dtype="<f8")
        ),
    }
    for field, digest in expected.items():
        if evidence.get(field) != digest:
            raise CampaignValidationError(f"{run_id} campaign evidence hash mismatch: {field}")
    method_hashes = evidence["method_gene_mean_prediction_sha256"]
    for method, key in methods.items():
        digest = _hash_array(np.asarray(arrays[key], dtype="<f8"))
        if method_hashes.get(method) != digest:
            raise CampaignValidationError(
                f"{run_id} campaign evidence prediction hash mismatch: {method}"
            )
    stack_base_hashes = evidence["stack_base_prediction_sha256"]
    for stack_name, method_map in STACK_BASE_PREDICTION_ARRAY_KEYS.items():
        for method, key in method_map.items():
            digest = _hash_array(np.asarray(arrays[key], dtype="<f8"))
            if stack_base_hashes[stack_name].get(method) != digest:
                raise CampaignValidationError(
                    f"{run_id} campaign evidence stack-base prediction hash mismatch: "
                    f"{stack_name}.{method}"
                )
    weight_hashes = evidence["distance_stack_weight_sha256"]
    for method, role in zip(
        STACK_METHOD_ORDER,
        (
            "distance_stack_weight_global",
            "distance_stack_weight_knn",
            "distance_stack_weight_ridge",
        ),
        strict=True,
    ):
        digest = _hash_array(np.asarray(arrays[scalar[role]], dtype="<f8"))
        if weight_hashes.get(method) != digest:
            raise CampaignValidationError(
                f"{run_id} campaign evidence distance-stack weight hash mismatch: {method}"
            )
    selected_name_array = arrays[scalar["selected_baseline_name"]]
    if selected_name_array.size != 1 or str(selected_name_array.reshape(-1)[0]) != job_selected_baseline:
        raise CampaignValidationError(
            f"{run_id} job-selected baseline name differs from campaign evidence"
        )
    selected_prediction_digest = _hash_array(
        np.asarray(arrays[scalar["selected_baseline_prediction"]], dtype="<f8")
    )
    if (
        evidence["job_selected_baseline"].get("outer_prediction_sha256")
        != selected_prediction_digest
    ):
        raise CampaignValidationError(
            f"{run_id} job-selected baseline prediction hash mismatch"
        )


def _validate_remote_identity(manifest: Mapping[str, Any], lock: Mapping[str, Any], *, run_id: str) -> None:
    remote = manifest.get("remote_execution_enforcement")
    if not isinstance(remote, Mapping):
        raise CampaignValidationError(f"{run_id} lacks remote execution enforcement")
    if remote.get("required_scope") != "remote-server-only":
        raise CampaignValidationError(f"{run_id} remote execution scope mismatch")
    registered = lock["designated_server"]
    if remote.get("registered_identity") != registered:
        raise CampaignValidationError(f"{run_id} registered server identity mismatch")
    check = remote.get("identity_check")
    if not isinstance(check, Mapping):
        raise CampaignValidationError(f"{run_id} server identity check is missing")
    if check.get("hostname") != registered["hostname"]:
        raise CampaignValidationError(f"{run_id} verified hostname mismatch")
    if check.get("machine_id_sha256") != registered["machine_id_sha256"]:
        raise CampaignValidationError(f"{run_id} verified machine-id mismatch")
    if check.get("verified_ipv4") != registered["ipv4"]:
        raise CampaignValidationError(f"{run_id} verified IPv4 mismatch")


def _validate_workspace_records(
    records: Any,
    *,
    workspace_root: Path,
    label: str,
    verify_files: bool,
) -> dict[str, tuple[int, str]]:
    if not isinstance(records, list) or not records:
        raise CampaignValidationError(f"{label} must be a non-empty record list")
    output: dict[str, tuple[int, str]] = {}
    root = workspace_root.resolve(strict=True)
    for position, record in enumerate(records):
        if not isinstance(record, Mapping) or set(record) != {"path", "bytes", "sha256"}:
            raise CampaignValidationError(f"invalid {label}[{position}] record")
        raw = record.get("path")
        if not isinstance(raw, str) or not raw or "\\" in raw:
            raise CampaignValidationError(f"invalid {label}[{position}].path")
        pure = PurePosixPath(raw)
        if (
            pure.is_absolute()
            or any(part in {"", ".", ".."} for part in pure.parts)
            or raw in output
        ):
            raise CampaignValidationError(f"unsafe or duplicate {label} path: {raw!r}")
        if pure.parts[:2] == ("data", "external"):
            raise CampaignValidationError(f"external outcomes are forbidden from {label}: {raw!r}")
        size = record.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise CampaignValidationError(f"invalid byte count for {label} {raw!r}")
        digest = _validate_sha256(record.get("sha256"), label=f"{label}.{raw}.sha256")
        output[raw] = (size, digest)
        if verify_files:
            target = root.joinpath(*pure.parts)
            if target.is_symlink():
                raise CampaignValidationError(f"workspace input may not be symlinked: {raw!r}")
            try:
                target.resolve(strict=True).relative_to(root)
            except (OSError, ValueError) as exc:
                raise CampaignValidationError(f"workspace input is missing or escapes root: {raw!r}") from exc
            actual = _file_record(target, relative_to=root)
            if actual != dict(record):
                raise CampaignValidationError(f"workspace {label} changed: {raw!r}")
    return output


def _locked_workspace_path(
    workspace_root: Path,
    value: Any,
    *,
    label: str,
    allow_windows_separators: bool = False,
) -> tuple[str, Path]:
    if not isinstance(value, str) or not value or ("\\" in value and not allow_windows_separators):
        raise CampaignValidationError(f"invalid locked {label} path")
    normalized = value.replace("\\", "/") if allow_windows_separators else value
    pure = PurePosixPath(normalized)
    if (
        pure.is_absolute()
        or pure.as_posix() != normalized
        or any(part in {"", ".", ".."} for part in normalized.split("/"))
        or pure.parts[:2] == ("data", "external")
    ):
        raise CampaignValidationError(f"unsafe locked {label} path: {value!r}")
    root = workspace_root.resolve(strict=True)
    path = root.joinpath(*pure.parts)
    if path.is_symlink():
        raise CampaignValidationError(f"locked {label} may not be symlinked: {value!r}")
    try:
        path.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise CampaignValidationError(f"locked {label} is missing or escapes root: {value!r}") from exc
    return normalized, path


def _expected_run_workspace_inventory(
    lock: Mapping[str, Any], *, workspace_root: Path
) -> tuple[set[str], set[str], str, str]:
    """Derive the runner's complete source inventory from hash-bound inputs."""
    assets = lock["locked_assets"]
    protein = assets["protein_representation"]
    family = assets["family_release"]
    manifest_relative, manifest_path = _locked_workspace_path(
        workspace_root, protein["manifest_path"], label="embedding manifest"
    )
    expected_manifest_hash = _validate_sha256(
        protein["manifest_sha256"], label="locked embedding manifest SHA-256"
    )
    if _sha256_file(manifest_path) != expected_manifest_hash:
        raise CampaignValidationError("frozen embedding manifest hash differs from analysis lock")
    embedding_manifest = _load_json(manifest_path, label="frozen embedding manifest")
    embedding_input = embedding_manifest.get("input")
    if not isinstance(embedding_input, Mapping):
        raise CampaignValidationError("frozen embedding manifest lacks input provenance")
    fasta_record = embedding_input.get("fasta")
    if not isinstance(fasta_record, Mapping):
        raise CampaignValidationError("frozen embedding manifest lacks FASTA provenance")
    fasta_relative, _ = _locked_workspace_path(
        workspace_root,
        fasta_record.get("path"),
        label="protein FASTA",
        allow_windows_separators=True,
    )
    fasta_hash = _validate_sha256(
        fasta_record.get("sha256"), label="frozen protein FASTA SHA-256"
    )
    input_paths = [
        protein["embedding_path"],
        protein["names_path"],
        manifest_relative,
        fasta_relative,
        assets["dependency_matrix"]["path"],
        assets["gene_manifest"]["path"],
        assets["cell_manifest"]["path"],
        family["release_manifest_path"],
        family["primary_fold_manifest_path"],
        *sorted(RUN_ADDITIONAL_INPUT_PATHS),
        *(item["path"] for item in family["sensitivity_fold_manifests"].values()),
    ]
    if not all(isinstance(path, str) for path in input_paths):
        raise CampaignValidationError("analysis lock contains a non-string input path")
    if len(set(input_paths)) != len(input_paths):
        raise CampaignValidationError("analysis lock contains duplicate input paths")
    return set(RUN_CODE_PATHS), set(input_paths), fasta_relative, fasta_hash


def _require_exact_workspace_inventory(
    records: Mapping[str, tuple[int, str]], expected: set[str], *, label: str
) -> None:
    actual = set(records)
    if actual != expected:
        raise CampaignValidationError(
            f"{label} inventory differs from runner contract: "
            f"missing={sorted(expected - actual)!r}, unexpected={sorted(actual - expected)!r}"
        )


def _family_fold_asset(
    lock: Mapping[str, Any], identity_percent: int
) -> tuple[str, str]:
    family = lock["locked_assets"]["family_release"]
    if int(identity_percent) == 30:
        return (
            str(family["primary_fold_manifest_path"]),
            str(family["primary_fold_manifest_sha256"]),
        )
    sensitivity = family["sensitivity_fold_manifests"].get(str(int(identity_percent)))
    if not isinstance(sensitivity, Mapping) or set(sensitivity) != {"path", "sha256"}:
        raise CampaignValidationError(
            f"identity {identity_percent!r} has no locked family fold manifest"
        )
    return str(sensitivity["path"]), str(sensitivity["sha256"])


def _validate_family_partition(
    manifest: Mapping[str, Any],
    *,
    lock: Mapping[str, Any],
    expected_key: JobKey,
    workspace_root: Path,
    run_id: str,
) -> ReplayedFamilyPartition:
    """Recompute the registered nested family split from hash-bound source files."""
    assets = lock["locked_assets"]
    family = assets["family_release"]
    gene_asset = assets["gene_manifest"]
    gene_relative, gene_path = _locked_workspace_path(
        workspace_root, gene_asset["path"], label="gene manifest"
    )
    gene_digest = _validate_sha256(
        gene_asset["sha256"], label="locked gene manifest SHA-256"
    )
    if _sha256_file(gene_path) != gene_digest:
        raise CampaignValidationError(f"{run_id} frozen gene manifest hash mismatch")
    try:
        genes = tuple(
            line.strip()
            for line in gene_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except (OSError, UnicodeError) as exc:
        raise CampaignValidationError(f"{run_id} cannot read frozen gene manifest") from exc
    if not genes or len(set(genes)) != len(genes):
        raise CampaignValidationError(f"{run_id} frozen gene manifest is empty or duplicated")

    fold_asset_path, fold_asset_digest = _family_fold_asset(
        lock, expected_key.identity_percent
    )
    fold_relative, fold_path = _locked_workspace_path(
        workspace_root, fold_asset_path, label="family fold manifest"
    )
    fold_digest = _validate_sha256(
        fold_asset_digest, label="locked family fold manifest SHA-256"
    )
    if _sha256_file(fold_path) != fold_digest:
        raise CampaignValidationError(f"{run_id} frozen family fold manifest hash mismatch")
    expected_family_record = _file_record(fold_path, relative_to=workspace_root)
    family_record = manifest.get("family_manifest")
    if (
        not isinstance(family_record, Mapping)
        or set(family_record) != {"path", "bytes", "sha256"}
        or dict(family_record) != expected_family_record
        or family_record.get("path") != fold_relative
    ):
        raise CampaignValidationError(f"{run_id} family manifest record mismatch")

    release_relative, release_path = _locked_workspace_path(
        workspace_root, family["release_manifest_path"], label="family release manifest"
    )
    release_digest = _validate_sha256(
        family["release_manifest_sha256"],
        label="locked family release manifest SHA-256",
    )
    if _sha256_file(release_path) != release_digest:
        raise CampaignValidationError(f"{run_id} frozen family release manifest hash mismatch")
    release = _load_json(release_path, label="family release manifest")
    if (
        release.get("schema") != "depplm-family-holdout-v1"
        or release.get("status") != "complete"
    ):
        raise CampaignValidationError(f"{run_id} family release is not complete")
    try:
        release_member = Path(fold_relative).relative_to(Path(release_relative).parent).as_posix()
    except ValueError as exc:
        raise CampaignValidationError(
            f"{run_id} family fold manifest is outside the registered release"
        ) from exc
    release_records = release.get("files")
    if not isinstance(release_records, list):
        raise CampaignValidationError(f"{run_id} family release file inventory is missing")
    matching_records = [
        record
        for record in release_records
        if isinstance(record, Mapping) and record.get("path") == release_member
    ]
    expected_release_record = {
        "path": release_member,
        "bytes": expected_family_record["bytes"],
        "sha256": fold_digest,
    }
    if len(matching_records) != 1 or dict(matching_records[0]) != expected_release_record:
        raise CampaignValidationError(
            f"{run_id} family fold manifest is not bound by the frozen release"
        )

    try:
        assignments = derive_seeded_family_assignments(
            fold_path,
            genes,
            split_seed=int(expected_key.split_seed),
        )
        split = make_family_split(assignments, int(expected_key.outer_fold))
    except (FamilyExperimentError, OSError, UnicodeError, ValueError) as exc:
        raise CampaignValidationError(
            f"{run_id} cannot replay the registered family partition: {exc}"
        ) from exc
    if tuple(item.gene for item in assignments) != genes:
        raise CampaignValidationError(f"{run_id} family assignment changed the gene order")
    component_ids = tuple(item.component_id for item in assignments)
    fold_ids = tuple(int(item.fold) for item in assignments)
    component_folds: dict[str, set[int]] = {}
    for component_id, fold_id in zip(component_ids, fold_ids, strict=True):
        component_folds.setdefault(component_id, set()).add(fold_id)
    if any(len(folds) != 1 for folds in component_folds.values()):
        raise CampaignValidationError(f"{run_id} a sequence component spans outer folds")

    partitions = {
        "inner_train": np.asarray(split.inner_train, dtype=np.int64),
        "inner_validation": np.asarray(split.inner_validation, dtype=np.int64),
        "outer_train": np.asarray(split.outer_train, dtype=np.int64),
        "outer_test": np.asarray(split.outer_test, dtype=np.int64),
    }
    component_sets = {
        name: {component_ids[int(index)] for index in indices}
        for name, indices in partitions.items()
    }
    expected_partition_audit = {
        name: {
            "n_genes": int(indices.size),
            "n_components": len(component_sets[name]),
            "gene_indices_sha256": _hash_array(indices.astype("<i8")),
            "component_ids_sha256": _hash_strings(sorted(component_sets[name])),
        }
        for name, indices in partitions.items()
    }
    partition_audit = manifest.get("partition_audit")
    if (
        not isinstance(partition_audit, Mapping)
        or set(partition_audit) != set(expected_partition_audit)
        or dict(partition_audit) != expected_partition_audit
    ):
        raise CampaignValidationError(
            f"{run_id} partition audit does not replay from the locked family manifest"
        )
    return ReplayedFamilyPartition(
        genes=genes,
        component_ids=component_ids,
        fold_ids=fold_ids,
        inner_train=partitions["inner_train"],
        inner_validation=partitions["inner_validation"],
        outer_train=partitions["outer_train"],
        outer_test=partitions["outer_test"],
    )


def _stable_environment(manifest: Mapping[str, Any], *, run_id: str) -> dict[str, Any]:
    environment = manifest.get("environment")
    if not isinstance(environment, Mapping):
        raise CampaignValidationError(f"{run_id} environment record is missing")
    required = (
        "hostname",
        "platform",
        "python",
        "numpy",
        "scipy",
        "sklearn",
        "torch",
        "cuda_runtime",
        "cuda_available",
        "device",
        "cuda_visible_devices",
        "deterministic_algorithms_enabled",
        "deterministic_debug_mode",
        "cudnn_version",
        "cudnn_deterministic",
        "cudnn_benchmark",
    )
    missing = [name for name in required if name not in environment]
    if missing:
        raise CampaignValidationError(f"{run_id} environment is missing {missing!r}")
    # Device indices and CUDA visibility may legitimately vary when 45 jobs are
    # scheduled across otherwise equivalent GPUs.  They remain in each source
    # manifest, while the cross-run fingerprint covers host/software/determinism.
    stable_fields = tuple(
        name
        for name in required
        if name not in {"device", "cuda_visible_devices"}
    )
    stable = {name: environment[name] for name in stable_fields}
    git = environment.get("git")
    if not isinstance(git, Mapping):
        raise CampaignValidationError(f"{run_id} git environment record is missing")
    stable["git"] = {"commit": git.get("commit"), "branch": git.get("branch")}
    return stable


def _validate_array_shapes(
    arrays: Mapping[str, np.ndarray],
    lock: Mapping[str, Any],
    *,
    run_id: str,
    stack_fits: Mapping[str, Mapping[str, Any]],
) -> None:
    scalar, methods = _prediction_key_map(lock)
    gene_ids = arrays[scalar["gene_ids"]]
    cells = arrays[scalar["cell_ids"]]
    if gene_ids.ndim != 1 or cells.ndim != 1 or gene_ids.size == 0 or cells.size == 0:
        raise CampaignValidationError(f"{run_id} gene/cell identifiers must be non-empty vectors")
    if gene_ids.dtype.kind not in {"U", "S"} or cells.dtype.kind not in {"U", "S"}:
        raise CampaignValidationError(f"{run_id} gene/cell identifiers must be fixed-width strings")
    component_ids = arrays[scalar["component_ids"]]
    if component_ids.dtype.kind not in {"U", "S"}:
        raise CampaignValidationError(f"{run_id} component identifiers must be fixed-width strings")
    n_genes = int(gene_ids.size)
    n_cells = int(cells.size)
    gene_vector_roles = (
        "canonical_gene_indices",
        "source_gene_indices",
        "component_ids",
        "component_sizes",
        "protein_lengths",
        "observed_gene_mean",
        "nearest_training_distance",
        "distance_gate_coordinate",
        "distance_stack_weight_global",
        "distance_stack_weight_knn",
        "distance_stack_weight_ridge",
        "selective_risk_score",
        "selected_baseline_prediction",
    )
    for role in gene_vector_roles:
        if arrays[scalar[role]].shape != (n_genes,):
            raise CampaignValidationError(f"{run_id} {role} does not align to genes")
    if arrays[scalar["source_cell_indices"]].shape != (n_cells,):
        raise CampaignValidationError(f"{run_id} source cell indices do not align")
    matrix_roles = (
        "observed_gene_effect",
        "observed_gene_centered_residual",
        "secondary_centered_residual_prediction",
        "secondary_total_prediction",
    )
    for role in matrix_roles:
        if arrays[scalar[role]].shape != (n_cells, n_genes):
            raise CampaignValidationError(
                f"{run_id} {role} must have shape cells x genes ({n_cells}, {n_genes})"
            )
    for method, key in methods.items():
        if arrays[key].shape != (n_genes,):
            raise CampaignValidationError(f"{run_id} prediction {method} does not align to genes")
    for stack_name, method_map in STACK_BASE_PREDICTION_ARRAY_KEYS.items():
        for method, key in method_map.items():
            if arrays[key].shape != (n_genes,):
                raise CampaignValidationError(
                    f"{run_id} {stack_name} base prediction {method} does not align to genes"
                )
    integer_roles = ("canonical_gene_indices", "source_gene_indices", "component_sizes", "protein_lengths", "source_cell_indices")
    for role in integer_roles:
        if arrays[scalar[role]].dtype.kind not in {"i", "u"}:
            raise CampaignValidationError(f"{run_id} {role} must use integer dtype")
    for role in ("canonical_gene_indices", "source_gene_indices", "source_cell_indices"):
        if np.any(arrays[scalar[role]] < 0):
            raise CampaignValidationError(f"{run_id} {role} contains negative values")
    for role in ("component_sizes", "protein_lengths"):
        if np.any(arrays[scalar[role]] <= 0):
            raise CampaignValidationError(f"{run_id} {role} must contain positive values")
    if len(set(arrays[scalar["canonical_gene_indices"]].tolist())) != n_genes:
        raise CampaignValidationError(f"{run_id} repeats a canonical gene index")
    if len(set(gene_ids.tolist())) != n_genes:
        raise CampaignValidationError(f"{run_id} repeats a gene identifier")
    if len(set(cells.tolist())) != n_cells:
        raise CampaignValidationError(f"{run_id} repeats a cell identifier")
    if len(set(arrays[scalar["source_gene_indices"]].tolist())) != n_genes:
        raise CampaignValidationError(f"{run_id} repeats a source gene index")
    if len(set(arrays[scalar["source_cell_indices"]].tolist())) != n_cells:
        raise CampaignValidationError(f"{run_id} repeats a source cell index")
    if any(len(value) == 0 for value in gene_ids.tolist()):
        raise CampaignValidationError(f"{run_id} contains an empty gene identifier")
    if any(len(value) == 0 for value in cells.tolist()):
        raise CampaignValidationError(f"{run_id} contains an empty cell identifier")
    if any(len(value) == 0 for value in component_ids.tolist()):
        raise CampaignValidationError(f"{run_id} contains an empty component identifier")
    selected_name_array = arrays[scalar["selected_baseline_name"]]
    if selected_name_array.shape != (1,) or selected_name_array.dtype.kind != "U":
        raise CampaignValidationError(
            f"{run_id} selected baseline name must be a length-one fixed-width Unicode array"
        )
    selected_name = str(selected_name_array[0])
    if selected_name not in BASELINE_METHODS:
        raise CampaignValidationError(f"{run_id} selected baseline name is not registered")
    finite_keys = [
        *methods.values(),
        *(
            key
            for method_map in STACK_BASE_PREDICTION_ARRAY_KEYS.values()
            for key in method_map.values()
        ),
        scalar["observed_gene_mean"],
        scalar["nearest_training_distance"],
        scalar["distance_gate_coordinate"],
        scalar["distance_stack_weight_global"],
        scalar["distance_stack_weight_knn"],
        scalar["distance_stack_weight_ridge"],
        scalar["selective_risk_score"],
        scalar["selected_baseline_prediction"],
        scalar["secondary_centered_residual_prediction"],
        scalar["secondary_total_prediction"],
    ]
    for key in finite_keys:
        array = arrays[key]
        if array.dtype.kind not in {"f", "i", "u"} or not np.all(np.isfinite(array)):
            raise CampaignValidationError(f"{run_id} array {key!r} must be finite numeric data")
    distance = arrays[scalar["nearest_training_distance"]]
    risk = arrays[scalar["selective_risk_score"]]
    if np.any(distance < 0.0) or np.any(risk < 0.0):
        raise CampaignValidationError(f"{run_id} distance and risk scores must be nonnegative")
    gate_coordinate = arrays[scalar["distance_gate_coordinate"]]
    if np.any(gate_coordinate < 0.0) or np.any(gate_coordinate > 1.0):
        raise CampaignValidationError(f"{run_id} distance gate coordinate is outside [0, 1]")
    constraint_tolerance = float(
        lock["primary_method_contract"]["implementation_contract"][
            "constraint_tolerance"
        ]
    )
    if set(stack_fits) != set(STACK_BASE_PREDICTION_ARRAY_KEYS):
        raise CampaignValidationError(f"{run_id} final distance-stack fit allowlist mismatch")
    distance = np.asarray(
        arrays[scalar["nearest_training_distance"]], dtype=np.float64
    )
    for stack_name, method_map in STACK_BASE_PREDICTION_ARRAY_KEYS.items():
        fit = stack_fits[stack_name]
        q10 = float(fit["q10"])
        q90 = float(fit["q90"])
        gate = np.clip((distance - q10) / (q90 - q10), 0.0, 1.0)
        a = np.asarray(fit["a"], dtype=np.float64)
        b = np.asarray(fit["b"], dtype=np.float64)
        query_weights = a[None, :] + gate[:, None] * b[None, :]
        if (
            np.any(query_weights < -constraint_tolerance)
            or not np.all(np.isfinite(query_weights))
            or not np.allclose(
                np.sum(query_weights, axis=1),
                1.0,
                rtol=0.0,
                atol=constraint_tolerance,
            )
        ):
            raise CampaignValidationError(
                f"{run_id} {stack_name} replayed weights violate convexity"
            )
        stack_bases = np.column_stack(
            [
                np.asarray(arrays[method_map[method]], dtype=np.float64)
                for method in STACK_METHOD_ORDER
            ]
        )
        replayed = np.sum(query_weights * stack_bases, axis=1)
        if not np.allclose(
            replayed,
            arrays[methods[stack_name]],
            rtol=1e-7,
            atol=1e-8,
        ):
            raise CampaignValidationError(
                f"{run_id} {stack_name} does not replay from its bound base predictions"
            )
        if stack_name == PRIMARY_METHOD:
            stored_gate = np.asarray(
                arrays[scalar["distance_gate_coordinate"]], dtype=np.float64
            )
            stored_weights = np.column_stack(
                [
                    arrays[scalar["distance_stack_weight_global"]],
                    arrays[scalar["distance_stack_weight_knn"]],
                    arrays[scalar["distance_stack_weight_ridge"]],
                ]
            ).astype(np.float64, copy=False)
            if not np.allclose(stored_gate, gate, rtol=0.0, atol=1e-15):
                raise CampaignValidationError(
                    f"{run_id} primary distance gate does not replay from its final fit"
                )
            if not np.allclose(
                stored_weights, query_weights, rtol=0.0, atol=1e-15
            ):
                raise CampaignValidationError(
                    f"{run_id} primary distance-stack weights do not replay from its final fit"
                )
    selected_prediction = arrays[scalar["selected_baseline_prediction"]]
    if not np.array_equal(selected_prediction, arrays[methods[selected_name]]):
        raise CampaignValidationError(
            f"{run_id} explicit selected-baseline prediction differs from {selected_name}"
        )
    observed = arrays[scalar["observed_gene_effect"]].astype(np.float64)
    observed_mean = arrays[scalar["observed_gene_mean"]].astype(np.float64)
    observed_residual = arrays[scalar["observed_gene_centered_residual"]].astype(np.float64)
    if np.any(np.isinf(observed)) or np.any(np.isinf(observed_residual)):
        raise CampaignValidationError(f"{run_id} observed matrices contain infinity")
    if np.any(np.all(~np.isfinite(observed), axis=0)):
        raise CampaignValidationError(f"{run_id} has a gene without any finite observed outcome")
    if not np.array_equal(np.isnan(observed), np.isnan(observed_residual)):
        raise CampaignValidationError(f"{run_id} observed residual missingness differs from outcomes")
    with np.errstate(invalid="ignore"):
        replay_mean = np.nanmean(observed, axis=0)
    if not np.allclose(replay_mean, observed_mean, rtol=1e-7, atol=1e-7):
        raise CampaignValidationError(f"{run_id} observed gene means do not replay")
    if not np.allclose(observed - observed_mean[None, :], observed_residual, rtol=1e-6, atol=1e-6, equal_nan=True):
        raise CampaignValidationError(f"{run_id} observed centered residual does not replay")
    proposed_mean = arrays[methods[SECONDARY_CONTEXT_METHOD]].astype(np.float64)
    proposed_residual = arrays[scalar["secondary_centered_residual_prediction"]].astype(np.float64)
    proposed_total = arrays[scalar["secondary_total_prediction"]].astype(np.float64)
    if not np.allclose(proposed_mean[None, :] + proposed_residual, proposed_total, rtol=1e-5, atol=1e-5):
        raise CampaignValidationError(f"{run_id} proposed total prediction does not replay")
    if not np.allclose(np.mean(proposed_residual, axis=0), 0.0, rtol=0.0, atol=2e-5):
        raise CampaignValidationError(f"{run_id} proposed residual is not centered by gene")


def validate_run_directory(
    run_dir: Path,
    *,
    expected_run_id: str,
    expected_key: JobKey,
    roster_sha256: str,
    lock: Mapping[str, Any],
    lock_record: Mapping[str, Any],
    workspace_root: Path,
    verify_workspace_files: bool,
) -> ValidatedRun:
    if run_dir.is_symlink():
        raise CampaignValidationError(f"run directory may not be a symlink: {run_dir}")
    run_dir = run_dir.resolve(strict=True)
    completion_path = _safe_run_file(run_dir, "completion.json")
    completion = _load_json(completion_path, label="run completion")
    if completion.get("schema") != COMPLETION_SCHEMA or completion.get("status") != "complete":
        raise CampaignValidationError(f"{expected_run_id} completion schema/status mismatch")
    if completion.get("run_id") != expected_run_id:
        raise CampaignValidationError(f"{expected_run_id} completion run id mismatch")
    if completion.get("scientific_use") is not True:
        raise CampaignValidationError(f"{expected_run_id} completion is not a formal scientific run")
    if _parse_run_key(
        completion.get("run_key"),
        roster_sha256=roster_sha256,
        label=f"{expected_run_id} completion.run_key",
    ) != expected_key:
        raise CampaignValidationError(f"{expected_run_id} completion run key mismatch")
    if completion.get("campaign_roster_sha256") != roster_sha256:
        raise CampaignValidationError(f"{expected_run_id} completion roster digest mismatch")
    if completion.get("campaign_roster_digest_source") != "command_line":
        raise CampaignValidationError(f"{expected_run_id} completion roster digest source mismatch")
    integrity = completion.get("integrity")
    required_integrity_fields = {
        "code_and_lock_unchanged_during_run",
        "inputs_unchanged_during_run",
        "outer_test_evaluated_once",
        "prediction_array_exact_allowlist_validated",
        "prediction_array_manifest_sha256",
        "campaign_evidence_sha256",
        "job_selected_baseline_outer_prediction_sha256",
        "stack_base_prediction_sha256",
        "distance_stack_solver_fail_closed",
    }
    if not isinstance(integrity, Mapping) or set(integrity) != required_integrity_fields:
        raise CampaignValidationError(f"{expected_run_id} completion integrity flags failed")
    for field in (
        "code_and_lock_unchanged_during_run",
        "inputs_unchanged_during_run",
        "outer_test_evaluated_once",
        "prediction_array_exact_allowlist_validated",
        "distance_stack_solver_fail_closed",
    ):
        if integrity.get(field) is not True:
            raise CampaignValidationError(
                f"{expected_run_id} completion integrity flag failed: {field}"
            )
    for field in (
        "prediction_array_manifest_sha256",
        "campaign_evidence_sha256",
        "job_selected_baseline_outer_prediction_sha256",
    ):
        _validate_sha256(integrity.get(field), label=f"{expected_run_id}.integrity.{field}")
    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {"predictions", "metrics", "run_manifest"}:
        raise CampaignValidationError(f"{expected_run_id} completion artifact allowlist mismatch")
    paths = {
        "predictions": _safe_run_file(run_dir, "predictions.npz"),
        "metrics": _safe_run_file(run_dir, "metrics.json"),
        "run_manifest": _safe_run_file(run_dir, "run-manifest.json"),
    }
    expected_names = {"predictions": "predictions.npz", "metrics": "metrics.json", "run_manifest": "run-manifest.json"}
    source_records = [_file_record(completion_path, relative_to=run_dir)]
    completion_prediction_record = artifacts["predictions"]
    if not isinstance(completion_prediction_record, Mapping):
        raise CampaignValidationError(f"{expected_run_id} completion prediction record is missing")
    basic_completion_prediction_record = {
        name: completion_prediction_record.get(name) for name in ("path", "bytes", "sha256")
    }
    source_records.append(
        _validate_record(
            basic_completion_prediction_record,
            paths["predictions"],
            expected_path=expected_names["predictions"],
            label=f"{expected_run_id}.predictions",
        )
    )
    for role in ("metrics", "run_manifest"):
        source_records.append(
            _validate_record(
                artifacts[role],
                paths[role],
                expected_path=expected_names[role],
                label=f"{expected_run_id}.{role}",
            )
        )

    manifest = _load_json(paths["run_manifest"], label="run manifest")
    if manifest.get("schema") != RUN_SCHEMA or manifest.get("status") != "complete":
        raise CampaignValidationError(f"{expected_run_id} run manifest schema/status mismatch")
    if manifest.get("run_id") != expected_run_id:
        raise CampaignValidationError(f"{expected_run_id} run manifest id mismatch")
    if manifest.get("scientific_use") is not True or manifest.get("pilot") is not False:
        raise CampaignValidationError(f"{expected_run_id} run manifest is pilot/non-scientific")
    if _parse_run_key(
        manifest.get("run_key"),
        roster_sha256=roster_sha256,
        label=f"{expected_run_id} manifest.run_key",
    ) != expected_key:
        raise CampaignValidationError(f"{expected_run_id} manifest run key mismatch")
    arguments = manifest.get("arguments")
    if not isinstance(arguments, Mapping):
        raise CampaignValidationError(f"{expected_run_id} arguments are missing")
    expected_arguments = {
        "identity_percent": expected_key.identity_percent,
        "outer_fold": expected_key.outer_fold,
        "split_seed": expected_key.split_seed,
        "training_seed": expected_key.training_seed,
        "run_id": expected_run_id,
        "pilot": False,
        "campaign_roster_sha256": roster_sha256,
        "campaign_roster_digest_source": "command_line",
    }
    if set(arguments) != set(expected_arguments) | {"device"}:
        raise CampaignValidationError(f"{expected_run_id} argument key allowlist mismatch")
    for field, expected in expected_arguments.items():
        if arguments.get(field) != expected:
            raise CampaignValidationError(f"{expected_run_id} argument mismatch: {field}")
    device = arguments.get("device")
    if not isinstance(device, str) or re.fullmatch(r"cuda(?::[0-9]+)?", device) is None:
        raise CampaignValidationError(f"{expected_run_id} formal run device is not CUDA")
    if manifest.get("lock_provenance", {}).get("lock") != dict(lock_record):
        raise CampaignValidationError(f"{expected_run_id} lock provenance mismatch")
    if manifest.get("campaign_aggregation_contract") != lock["campaign_aggregation"]:
        raise CampaignValidationError(f"{expected_run_id} aggregation contract mismatch")
    if manifest.get("training_contract") != lock["training_contract"]:
        raise CampaignValidationError(f"{expected_run_id} training contract mismatch")
    if manifest.get("robustness_strata_contract") != lock["robustness_strata"]:
        raise CampaignValidationError(f"{expected_run_id} robustness strata contract mismatch")
    if manifest.get("selective_risk_score_contract") != lock["selective_risk_score"]:
        raise CampaignValidationError(f"{expected_run_id} selective-risk contract mismatch")
    if manifest.get("external_3d_contract") != lock["external_3d_analysis"]:
        raise CampaignValidationError(f"{expected_run_id} external-3D contract mismatch")
    expected_effective_training = {
        "inner_max_epochs": lock["training_contract"]["inner_max_epochs"],
        "patience": lock["training_contract"]["patience"],
        "batch_size": lock["training_contract"]["batch_size"],
        "gene_subsample": lock["training_contract"]["gene_subsample"],
        "bootstrap_replicates": lock["bootstrap"]["replicates"],
        "pilot_subset_seed": None,
    }
    if manifest.get("effective_training") != expected_effective_training:
        raise CampaignValidationError(f"{expected_run_id} effective training contract mismatch")
    _validate_remote_identity(manifest, lock, run_id=expected_run_id)
    stable_environment = _stable_environment(manifest, run_id=expected_run_id)
    if manifest["environment"].get("device") != device:
        raise CampaignValidationError(f"{expected_run_id} argument/environment device mismatch")
    expected_code, expected_input, fasta_path, fasta_hash = _expected_run_workspace_inventory(
        lock, workspace_root=workspace_root
    )
    code_records = _validate_workspace_records(
        manifest.get("code_files"),
        workspace_root=workspace_root,
        label="code_files",
        verify_files=verify_workspace_files,
    )
    input_records = _validate_workspace_records(
        manifest.get("input_files"),
        workspace_root=workspace_root,
        label="input_files",
        verify_files=verify_workspace_files,
    )
    _require_exact_workspace_inventory(code_records, expected_code, label="code_files")
    _require_exact_workspace_inventory(input_records, expected_input, label="input_files")
    if input_records[fasta_path][1] != fasta_hash:
        raise CampaignValidationError("run FASTA record differs from frozen embedding provenance")
    partition = _validate_family_partition(
        manifest,
        lock=lock,
        expected_key=expected_key,
        workspace_root=workspace_root,
        run_id=expected_run_id,
    )
    data = manifest.get("data")
    expected_data_fields = {
        "n_genes",
        "n_cells",
        "gene_ids_sha256",
        "cell_ids_sha256",
        "source_gene_indices_sha256",
        "source_cell_indices_sha256",
    }
    if not isinstance(data, Mapping) or set(data) != expected_data_fields:
        raise CampaignValidationError(f"{expected_run_id} data evidence fields mismatch")
    if (
        data.get("n_genes") != len(partition.genes)
        or data.get("gene_ids_sha256") != _hash_strings(partition.genes)
    ):
        raise CampaignValidationError(f"{expected_run_id} data gene universe mismatch")
    # Formal runs load every locked gene in manifest order (pilot runs are
    # rejected above), so BaseData.source_gene_indices is this full mapping.
    expected_full_source_gene_indices = np.arange(
        len(partition.genes), dtype="<i8"
    )
    if data.get("source_gene_indices_sha256") != _hash_array(
        expected_full_source_gene_indices
    ):
        raise CampaignValidationError(
            f"{expected_run_id} source gene index mapping mismatch"
        )
    _strict_nonnegative_int(
        data.get("n_cells"), label=f"{expected_run_id}.data.n_cells", positive=True
    )
    for field in (
        "cell_ids_sha256",
        "source_gene_indices_sha256",
        "source_cell_indices_sha256",
    ):
        _validate_sha256(data.get(field), label=f"{expected_run_id}.data.{field}")
    (
        baseline_evidence,
        job_selected_baseline,
        campaign_evidence,
        stack_fits,
    ) = _manifest_campaign_evidence(
        manifest,
        run_id=expected_run_id,
        expected_key=expected_key,
        roster_sha256=roster_sha256,
        lock=lock,
        scipy_version=str(stable_environment["scipy"]),
        partition=partition,
    )

    metrics = _load_json(paths["metrics"], label="run metrics")
    if metrics.get("schema") != METRICS_SCHEMA or metrics.get("run_id") != expected_run_id:
        raise CampaignValidationError(f"{expected_run_id} metrics schema/id mismatch")
    if _parse_run_key(
        metrics.get("run_key"),
        roster_sha256=roster_sha256,
        label=f"{expected_run_id} metrics.run_key",
    ) != expected_key:
        raise CampaignValidationError(f"{expected_run_id} metrics run key mismatch")
    if metrics.get("campaign_evidence") != campaign_evidence:
        raise CampaignValidationError(f"{expected_run_id} campaign evidence differs across envelopes")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {
        "predictions",
        "metrics",
        "prediction_array_keys",
        "stack_base_prediction_array_keys",
        "prediction_array_names_exact",
        "prediction_array_manifest_sha256",
    }:
        raise CampaignValidationError(f"{expected_run_id} manifest output allowlist mismatch")
    if outputs.get("metrics") != artifacts["metrics"]:
        raise CampaignValidationError(f"{expected_run_id} metrics record differs across envelopes")
    if outputs.get("prediction_array_keys") != lock["campaign_aggregation"]["run_prediction_array_keys"]:
        raise CampaignValidationError(f"{expected_run_id} prediction key map differs from lock")
    if outputs.get("stack_base_prediction_array_keys") != STACK_BASE_PREDICTION_ARRAY_KEYS:
        raise CampaignValidationError(
            f"{expected_run_id} stack-base prediction key map differs from runner contract"
        )
    locked_array_keys = lock["campaign_aggregation"]["run_prediction_array_keys"]
    expected_array_names = [
        value
        for key, value in locked_array_keys.items()
        if key != "method_gene_mean_predictions"
    ] + list(locked_array_keys["method_gene_mean_predictions"].values()) + [
        array_name
        for method_map in STACK_BASE_PREDICTION_ARRAY_KEYS.values()
        for array_name in method_map.values()
    ]
    if len(expected_array_names) != len(set(expected_array_names)):
        raise CampaignValidationError("locked run prediction array names are not unique")
    if outputs.get("prediction_array_names_exact") != expected_array_names:
        raise CampaignValidationError(
            f"{expected_run_id} exact prediction array name list differs from lock"
        )
    prediction_record = outputs.get("predictions")
    if not isinstance(prediction_record, Mapping):
        raise CampaignValidationError(f"{expected_run_id} prediction record is missing")
    if set(prediction_record) != {"path", "bytes", "sha256", "schema", "arrays"}:
        raise CampaignValidationError(f"{expected_run_id} prediction record fields mismatch")
    if dict(prediction_record) != dict(completion_prediction_record):
        raise CampaignValidationError(f"{expected_run_id} prediction record differs across envelopes")
    array_records = prediction_record.get("arrays")
    if not isinstance(array_records, Mapping) or set(array_records) != set(expected_array_names):
        raise CampaignValidationError(
            f"{expected_run_id} prediction array manifest differs from exact name allowlist"
        )
    prediction_array_manifest_sha256 = hashlib.sha256(
        _json_bytes(array_records)
    ).hexdigest()
    if outputs.get("prediction_array_manifest_sha256") != prediction_array_manifest_sha256:
        raise CampaignValidationError(
            f"{expected_run_id} output prediction array manifest hash mismatch"
        )
    if campaign_evidence.get("prediction_array_manifest_sha256") != prediction_array_manifest_sha256:
        raise CampaignValidationError(
            f"{expected_run_id} campaign prediction array manifest hash mismatch"
        )
    if integrity.get("prediction_array_manifest_sha256") != prediction_array_manifest_sha256:
        raise CampaignValidationError(
            f"{expected_run_id} completion prediction array manifest hash mismatch"
        )
    campaign_evidence_sha256 = hashlib.sha256(_json_bytes(campaign_evidence)).hexdigest()
    if integrity.get("campaign_evidence_sha256") != campaign_evidence_sha256:
        raise CampaignValidationError(
            f"{expected_run_id} completion campaign evidence hash mismatch"
        )
    selected_prediction_sha256 = campaign_evidence["job_selected_baseline"][
        "outer_prediction_sha256"
    ]
    if (
        integrity.get("job_selected_baseline_outer_prediction_sha256")
        != selected_prediction_sha256
    ):
        raise CampaignValidationError(
            f"{expected_run_id} completion job-selected baseline hash mismatch"
        )
    if (
        integrity.get("stack_base_prediction_sha256")
        != campaign_evidence["stack_base_prediction_sha256"]
    ):
        raise CampaignValidationError(
            f"{expected_run_id} completion stack-base prediction hashes mismatch"
        )
    basic_prediction_record = {name: prediction_record.get(name) for name in ("path", "bytes", "sha256")}
    _validate_record(basic_prediction_record, paths["predictions"], expected_path="predictions.npz", label=f"{expected_run_id}.predictions")
    scalar, methods = _prediction_key_map(lock)
    required_keys = (
        set(scalar.values())
        | set(methods.values())
        | {
            key
            for method_map in STACK_BASE_PREDICTION_ARRAY_KEYS.values()
            for key in method_map.values()
        }
    )
    arrays = _read_npz(paths["predictions"], prediction_record=prediction_record, required_keys=required_keys)
    expected_outer = np.asarray(partition.outer_test, dtype=np.int64)
    if not np.array_equal(
        np.asarray(arrays[scalar["canonical_gene_indices"]], dtype=np.int64),
        expected_outer,
    ):
        raise CampaignValidationError(
            f"{expected_run_id} canonical outer-test indices do not replay from the family split"
        )
    expected_gene_ids = np.asarray(
        [partition.genes[int(index)] for index in expected_outer], dtype=str
    )
    expected_component_ids = np.asarray(
        [partition.component_ids[int(index)] for index in expected_outer], dtype=str
    )
    if not np.array_equal(arrays[scalar["gene_ids"]].astype(str), expected_gene_ids):
        raise CampaignValidationError(
            f"{expected_run_id} outer-test gene IDs do not replay from the family split"
        )
    if not np.array_equal(
        arrays[scalar["component_ids"]].astype(str), expected_component_ids
    ):
        raise CampaignValidationError(
            f"{expected_run_id} outer-test component IDs do not replay from the family split"
        )
    if not np.array_equal(
        np.asarray(arrays[scalar["source_gene_indices"]], dtype=np.int64),
        expected_full_source_gene_indices[expected_outer],
    ):
        raise CampaignValidationError(
            f"{expected_run_id} outer-test source gene indices do not replay "
            "from the locked gene manifest"
        )
    if (
        data["n_cells"] != int(arrays[scalar["cell_ids"]].size)
        or data["cell_ids_sha256"]
        != _hash_strings(arrays[scalar["cell_ids"]].astype(str).tolist())
        or data["source_cell_indices_sha256"]
        != _hash_array(
            np.asarray(arrays[scalar["source_cell_indices"]], dtype="<i8")
        )
    ):
        raise CampaignValidationError(f"{expected_run_id} data cell universe mismatch")
    _validate_array_shapes(
        arrays,
        lock,
        run_id=expected_run_id,
        stack_fits=stack_fits,
    )
    _validate_campaign_array_evidence(
        campaign_evidence,
        arrays,
        lock,
        run_id=expected_run_id,
        job_selected_baseline=job_selected_baseline,
    )
    return ValidatedRun(
        key=expected_key,
        run_id=expected_run_id,
        path=run_dir,
        arrays=arrays,
        baseline_inner_validation=baseline_evidence,
        job_selected_baseline=job_selected_baseline,
        manifest=dict(manifest),
        source_records=source_records,
    )


def _canonical(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _require_run_consistency(runs: Sequence[ValidatedRun], *, workspace_root: Path, verify_workspace_files: bool) -> None:
    if not runs:
        raise CampaignValidationError("campaign has no validated runs")
    first = runs[0].manifest
    invariants = (
        "lock_provenance",
        "training_contract",
        "robustness_strata_contract",
        "selective_risk_score_contract",
        "external_3d_contract",
        "effective_training",
        "data",
    )
    for field in invariants:
        reference = _canonical(first.get(field))
        if any(_canonical(run.manifest.get(field)) != reference for run in runs[1:]):
            raise CampaignValidationError(f"completed runs do not share one {field}")
    reference_environment = _canonical(_stable_environment(first, run_id=runs[0].run_id))
    for run in runs[1:]:
        if _canonical(_stable_environment(run.manifest, run_id=run.run_id)) != reference_environment:
            raise CampaignValidationError("completed runs do not share one stable environment fingerprint")
    for field in ("code_files", "input_files"):
        reference = _validate_workspace_records(first.get(field), workspace_root=workspace_root, label=field, verify_files=verify_workspace_files)
        for run in runs[1:]:
            current = _validate_workspace_records(run.manifest.get(field), workspace_root=workspace_root, label=field, verify_files=False)
            if current != reference:
                raise CampaignValidationError(f"completed runs do not share identical {field}")

    partition_groups: dict[tuple[int, int, int], list[ValidatedRun]] = {}
    for run in runs:
        group_key = (
            int(run.key.identity_percent),
            int(run.key.outer_fold),
            int(run.key.split_seed),
        )
        partition_groups.setdefault(group_key, []).append(run)
    for group_key, group in partition_groups.items():
        reference = group[0]
        reference_scalar = reference.manifest["outputs"]["prediction_array_keys"]
        for current in group[1:]:
            current_scalar = current.manifest["outputs"]["prediction_array_keys"]
            if (
                current.manifest.get("family_manifest")
                != reference.manifest.get("family_manifest")
                or current.manifest.get("partition_audit")
                != reference.manifest.get("partition_audit")
            ):
                raise CampaignValidationError(
                    "training seeds do not share one replayed family partition for "
                    f"identity/fold/split={group_key!r}"
                )
            for role in ("canonical_gene_indices", "gene_ids", "component_ids"):
                if not _same_array(
                    reference.arrays[reference_scalar[role]],
                    current.arrays[current_scalar[role]],
                ):
                    raise CampaignValidationError(
                        "training seeds do not share identical outer-test membership for "
                        f"identity/fold/split={group_key!r}: {role}"
                    )

    component_folds: dict[tuple[int, int, str], int] = {}
    for run in runs:
        scalar = run.manifest["outputs"]["prediction_array_keys"]
        for component_id in np.asarray(
            run.arrays[scalar["component_ids"]]
        ).astype(str):
            key = (
                int(run.key.identity_percent),
                int(run.key.split_seed),
                str(component_id),
            )
            previous_fold = component_folds.setdefault(key, int(run.key.outer_fold))
            if previous_fold != int(run.key.outer_fold):
                raise CampaignValidationError(
                    "one family component spans outer folds for "
                    f"identity/split/component={key!r}"
                )


def select_job_baseline(evidence: Mapping[str, Mapping[str, float]]) -> str:
    """Replay one job's comparator choice without consulting any other job."""
    validated = _validate_baseline_evidence(evidence, run_id="baseline-selection-replay")
    return min(
        BASELINE_METHODS,
        key=lambda method: (
            -validated[method]["r2"],
            validated[method]["mse"],
            method,
        ),
    )


def _same_array(left: np.ndarray, right: np.ndarray) -> bool:
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    if left.dtype.kind in {"f", "c"}:
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray, *, rank: bool = False) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2:
        return None
    if rank:
        x, y = _average_ranks(x), _average_ranks(y)
    x = x - np.mean(x)
    y = y - np.mean(y)
    denominator = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    if denominator == 0.0:
        return None
    return float(np.dot(x, y) / denominator)


def regression_metrics(prediction: np.ndarray, target: np.ndarray, global_reference: np.ndarray) -> dict[str, Any]:
    predicted = np.asarray(prediction, dtype=np.float64)
    observed = np.asarray(target, dtype=np.float64)
    reference = np.asarray(global_reference, dtype=np.float64)
    if predicted.ndim != 1 or observed.shape != predicted.shape or reference.shape != predicted.shape:
        raise CampaignValidationError("campaign metric vectors do not align")
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(observed)) or not np.all(np.isfinite(reference)):
        raise CampaignValidationError("campaign metric vectors must be finite")
    error = predicted - observed
    reference_error = reference - observed
    sse = float(np.dot(error, error))
    denominator = float(np.dot(reference_error, reference_error))
    return {
        "n_genes": int(predicted.size),
        "mse": float(np.mean(np.square(error))),
        "mae": float(np.mean(np.abs(error))),
        "pearson": _correlation(predicted, observed),
        "spearman": _correlation(predicted, observed, rank=True),
        "r2_vs_consensus_global_train_mean": None if denominator <= 0.0 else float(1.0 - sse / denominator),
        "reference_sse": denominator,
    }


def _quantile_strata(values: np.ndarray, quantiles: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise CampaignValidationError("quartile source must be a non-empty finite vector")
    edges = np.quantile(vector, np.asarray(quantiles, dtype=np.float64))
    labels = np.searchsorted(edges[1:-1], vector, side="right").astype(np.int8)
    return edges, labels


def aggregate_validated_runs(runs: Sequence[ValidatedRun], lock: Mapping[str, Any]) -> Consensus:
    if len(runs) != 45:
        raise CampaignValidationError(f"campaign requires 45 runs, observed {len(runs)}")
    indexed = {run.key: run for run in runs}
    if len(indexed) != 45 or set(indexed) != set(expected_job_keys(lock)):
        raise CampaignValidationError("validated runs are not the exact 45-job Cartesian product")
    scalar, method_keys = _prediction_key_map(lock)
    grid = lock["campaign_aggregation"]["job_grid"]
    pairs = [(int(s), int(t)) for s in grid["split_seeds"] for t in grid["training_seeds"]]
    per_pair: dict[tuple[int, int], dict[int, tuple[ValidatedRun, int]]] = {}
    canonical_set: set[int] | None = None
    for split_seed, training_seed in pairs:
        genes: dict[int, tuple[ValidatedRun, int]] = {}
        for fold in grid["outer_folds"]:
            run = indexed[JobKey(30.0, int(fold), split_seed, training_seed)]
            indices = run.arrays[scalar["canonical_gene_indices"]].astype(np.int64)
            for local, raw_index in enumerate(indices.tolist()):
                if raw_index in genes:
                    raise CampaignValidationError(
                        f"gene {raw_index} appears more than once across folds for split={split_seed}, train={training_seed}"
                    )
                genes[raw_index] = (run, local)
        observed = set(genes)
        if canonical_set is None:
            canonical_set = observed
        elif observed != canonical_set:
            raise CampaignValidationError(
                f"OOF gene universe differs for split={split_seed}, train={training_seed}"
            )
        per_pair[(split_seed, training_seed)] = genes
    if not canonical_set:
        raise CampaignValidationError("campaign OOF gene universe is empty")
    canonical_indices = np.asarray(sorted(canonical_set), dtype=np.int64)
    n_genes = int(canonical_indices.size)
    appearances: dict[int, list[tuple[ValidatedRun, int]]] = {int(index): [] for index in canonical_indices}
    for genes in per_pair.values():
        for index, location in genes.items():
            appearances[index].append(location)
    if any(len(value) != 9 for value in appearances.values()):
        raise CampaignValidationError("each gene must have exactly nine OOF appearances")

    first_run, first_local = appearances[int(canonical_indices[0])][0]
    cell_ids = first_run.arrays[scalar["cell_ids"]].copy()
    source_cells = first_run.arrays[scalar["source_cell_indices"]].copy()
    for run in runs:
        if not _same_array(run.arrays[scalar["cell_ids"]], cell_ids) or not _same_array(run.arrays[scalar["source_cell_indices"]], source_cells):
            raise CampaignValidationError("cell identity/order differs across campaign runs")
    n_cells = int(cell_ids.size)
    gene_id_values: list[str] = []
    source_indices = np.empty(n_genes, dtype=np.int64)
    component_id_values: list[str] = []
    component_sizes = np.empty(n_genes, dtype=np.int64)
    protein_lengths = np.empty(n_genes, dtype=np.int64)
    observed_mean = np.empty(n_genes, dtype=np.float64)
    observed_effect = np.empty((n_cells, n_genes), dtype=np.float32)
    observed_residual = np.empty((n_cells, n_genes), dtype=np.float32)
    distances = np.empty((9, n_genes), dtype=np.float64)
    risks = np.empty((9, n_genes), dtype=np.float64)
    gate_coordinates = np.empty((9, n_genes), dtype=np.float64)
    gate_weights = {
        role: np.empty((9, n_genes), dtype=np.float64)
        for role in (
            "distance_stack_weight_global",
            "distance_stack_weight_knn",
            "distance_stack_weight_ridge",
        )
    }
    predictions = {method: np.empty((9, n_genes), dtype=np.float64) for method in ALL_METHODS}
    selected_baseline_predictions = np.empty((9, n_genes), dtype=np.float64)
    proposed_residuals = np.empty((9, n_cells, n_genes), dtype=np.float64)
    proposed_totals = np.empty((9, n_cells, n_genes), dtype=np.float64)

    for position, canonical_index in enumerate(canonical_indices.tolist()):
        copies = appearances[canonical_index]
        reference_run, reference_local = copies[0]
        reference_values = {
            "gene_id": reference_run.arrays[scalar["gene_ids"]][reference_local],
            "source_index": reference_run.arrays[scalar["source_gene_indices"]][reference_local],
            "component_id": reference_run.arrays[scalar["component_ids"]][reference_local],
            "component_size": reference_run.arrays[scalar["component_sizes"]][reference_local],
            "protein_length": reference_run.arrays[scalar["protein_lengths"]][reference_local],
            "observed_mean": reference_run.arrays[scalar["observed_gene_mean"]][reference_local],
            "observed_effect": reference_run.arrays[scalar["observed_gene_effect"]][:, reference_local],
            "observed_residual": reference_run.arrays[scalar["observed_gene_centered_residual"]][:, reference_local],
        }
        for copy_index, (run, local) in enumerate(copies):
            comparisons = {
                "gene_id": run.arrays[scalar["gene_ids"]][local],
                "source_index": run.arrays[scalar["source_gene_indices"]][local],
                "component_id": run.arrays[scalar["component_ids"]][local],
                "component_size": run.arrays[scalar["component_sizes"]][local],
                "protein_length": run.arrays[scalar["protein_lengths"]][local],
                "observed_mean": run.arrays[scalar["observed_gene_mean"]][local],
                "observed_effect": run.arrays[scalar["observed_gene_effect"]][:, local],
                "observed_residual": run.arrays[scalar["observed_gene_centered_residual"]][:, local],
            }
            for field, reference in reference_values.items():
                candidate = comparisons[field]
                if isinstance(reference, np.ndarray):
                    equal = np.array_equal(candidate, reference, equal_nan=True)
                else:
                    equal = bool(candidate == reference)
                if not equal:
                    raise CampaignValidationError(
                        f"gene metadata/outcome disagreement for canonical index {canonical_index}: {field}"
                    )
            distances[copy_index, position] = run.arrays[scalar["nearest_training_distance"]][local]
            risks[copy_index, position] = run.arrays[scalar["selective_risk_score"]][local]
            gate_coordinates[copy_index, position] = run.arrays[scalar["distance_gate_coordinate"]][local]
            for role, values in gate_weights.items():
                values[copy_index, position] = run.arrays[scalar[role]][local]
            for method, key in method_keys.items():
                predictions[method][copy_index, position] = run.arrays[key][local]
            selected_baseline_predictions[copy_index, position] = run.arrays[
                scalar["selected_baseline_prediction"]
            ][local]
            proposed_residuals[copy_index, :, position] = run.arrays[
                scalar["secondary_centered_residual_prediction"]
            ][:, local]
            proposed_totals[copy_index, :, position] = run.arrays[
                scalar["secondary_total_prediction"]
            ][:, local]
        gene_id_values.append(str(reference_values["gene_id"]))
        source_indices[position] = int(reference_values["source_index"])
        component_id_values.append(str(reference_values["component_id"]))
        component_sizes[position] = int(reference_values["component_size"])
        protein_lengths[position] = int(reference_values["protein_length"])
        observed_mean[position] = float(reference_values["observed_mean"])
        observed_effect[:, position] = reference_values["observed_effect"]
        observed_residual[:, position] = reference_values["observed_residual"]

    gene_ids = np.asarray(gene_id_values, dtype=f"<U{max(map(len, gene_id_values))}")
    component_ids = np.asarray(
        component_id_values,
        dtype=f"<U{max(map(len, component_id_values))}",
    )
    if len(set(gene_id_values)) != n_genes:
        raise CampaignValidationError("campaign maps multiple canonical indices to one gene id")
    if len(set(source_indices.tolist())) != n_genes:
        raise CampaignValidationError("campaign maps multiple canonical genes to one source index")

    consensus_predictions = {method: np.mean(values, axis=0) for method, values in predictions.items()}
    consensus_selected_baseline = np.mean(selected_baseline_predictions, axis=0)
    consensus_residual = np.mean(proposed_residuals, axis=0)
    consensus_total = np.mean(proposed_totals, axis=0)
    if not np.allclose(
        consensus_total,
        consensus_predictions[SECONDARY_CONTEXT_METHOD][None, :] + consensus_residual,
        rtol=1e-6,
        atol=1e-6,
    ):
        raise CampaignValidationError("consensus total prediction is not arithmetic mean plus residual")
    distance = np.mean(distances, axis=0)
    risk = np.mean(risks, axis=0)
    quantiles = lock["campaign_aggregation"]["consensus_risk_and_distance"]["distance_quartile_quantiles"]
    distance_edges, distance_quartile = _quantile_strata(distance, quantiles)
    length_edges, length_quartile = _quantile_strata(protein_lengths.astype(np.float64), lock["robustness_strata"]["protein_length"]["quantiles"])
    selective_order = np.lexsort((canonical_indices, risk)).astype(np.int64)
    arrays: dict[str, np.ndarray] = {
        "canonical_gene_indices": canonical_indices,
        "gene_ids": gene_ids,
        "source_gene_indices": source_indices,
        "component_ids": component_ids,
        "component_size_full_release": component_sizes,
        "protein_length_residues": protein_lengths,
        "cell_ids": cell_ids,
        "source_cell_indices": source_cells,
        "observed_gene_effect": observed_effect,
        "observed_gene_mean": observed_mean,
        "observed_gene_centered_residual": observed_residual,
        "consensus_nearest_training_cosine_distance": distance,
        "consensus_validation_fitted_distance_risk_score": risk,
        "consensus_distance_gate_z": np.mean(gate_coordinates, axis=0),
        "consensus_job_selected_baseline_gene_mean_prediction": consensus_selected_baseline,
        "distance_quartile_zero_based": distance_quartile,
        "distance_quartile_edges": distance_edges,
        "protein_length_quartile_zero_based": length_quartile,
        "protein_length_quartile_edges": length_edges,
        "selective_order": selective_order,
        "family_inductive_decomp_centered_residual_prediction": consensus_residual.astype(np.float32),
        "family_inductive_decomp_total_prediction": consensus_total.astype(np.float32),
    }
    for role, values in gate_weights.items():
        arrays[f"consensus_{role}"] = np.mean(values, axis=0)
    for method, key in method_keys.items():
        arrays[key] = consensus_predictions[method]

    seed_pair_metrics: dict[str, Any] = {}
    for split_seed, training_seed in pairs:
        genes = per_pair[(split_seed, training_seed)]
        order = [genes[int(index)] for index in canonical_indices]
        pair_predictions = {
            method: np.asarray([run.arrays[key][local] for run, local in order], dtype=np.float64)
            for method, key in method_keys.items()
        }
        pair_selected_baseline = np.asarray(
            [
                run.arrays[scalar["selected_baseline_prediction"]][local]
                for run, local in order
            ],
            dtype=np.float64,
        )
        label = f"split-{split_seed}|train-{training_seed}"
        seed_pair_metrics[label] = {
            "methods": {
                method: regression_metrics(
                    values, observed_mean, pair_predictions["GLOBAL-TRAIN-MEAN"]
                )
                for method, values in pair_predictions.items()
            },
            "per_job_selected_baseline": regression_metrics(
                pair_selected_baseline,
                observed_mean,
                pair_predictions["GLOBAL-TRAIN-MEAN"],
            ),
        }
    counts = {method: 0 for method in BASELINE_METHODS}
    jobs: list[dict[str, Any]] = []
    for run in sorted(runs, key=lambda item: item.key):
        counts[run.job_selected_baseline] += 1
        jobs.append(
            {
                "run_id": run.run_id,
                "run_key": run.key.as_dict(),
                "selected": run.job_selected_baseline,
                "selected_inner_validation": run.baseline_inner_validation[
                    run.job_selected_baseline
                ],
            }
        )
    job_baseline_selection = {
        "selection_scope": "independent_within_each_job_inner_validation_only",
        "cross_job_selection_used": False,
        "rule": "max_r2_then_min_mse_then_lexicographic_method_name",
        "job_count": len(jobs),
        "identity_distribution_descriptive_only": counts,
        "jobs": jobs,
    }
    return Consensus(arrays, job_baseline_selection, seed_pair_metrics)


def _component_groups(component_ids: np.ndarray) -> list[np.ndarray]:
    grouped: dict[str, list[int]] = {}
    for index, value in enumerate(component_ids.astype(str).tolist()):
        grouped.setdefault(value, []).append(index)
    return [np.asarray(indices, dtype=np.int64) for indices in grouped.values()]


def _bootstrap_primary(
    proposed: np.ndarray,
    baseline: np.ndarray,
    global_reference: np.ndarray,
    target: np.ndarray,
    component_ids: np.ndarray,
    *,
    replicates: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    groups = _component_groups(component_ids)
    proposed_loss = np.square(proposed - target)
    baseline_loss = np.square(baseline - target)
    global_loss = np.square(global_reference - target)
    improvement_global = global_loss - proposed_loss
    improvement_baseline = baseline_loss - proposed_loss
    sums_global = np.asarray([np.sum(improvement_global[group]) for group in groups])
    sums_baseline = np.asarray([np.sum(improvement_baseline[group]) for group in groups])
    sums_denominator = np.asarray([np.sum(global_loss[group]) for group in groups])
    sizes = np.asarray([group.size for group in groups], dtype=np.int64)
    rng = np.random.default_rng(seed)
    boot_mse = np.empty(replicates, dtype=np.float64)
    boot_delta_r2 = np.empty(replicates, dtype=np.float64)
    chunk = max(1, 1_000_000 // len(groups))
    for start in range(0, replicates, chunk):
        stop = min(replicates, start + chunk)
        sampled = rng.integers(0, len(groups), size=(stop - start, len(groups)))
        count = np.sum(sizes[sampled], axis=1)
        denominator = np.sum(sums_denominator[sampled], axis=1)
        boot_mse[start:stop] = np.sum(sums_global[sampled], axis=1) / count
        boot_delta_r2[start:stop] = np.divide(
            np.sum(sums_baseline[sampled], axis=1),
            denominator,
            out=np.full(stop - start, np.nan),
            where=denominator > 0.0,
        )
    if not np.all(np.isfinite(boot_mse)) or not np.all(np.isfinite(boot_delta_r2)):
        raise CampaignValidationError("primary component bootstrap produced undefined replicates")
    alpha = (1.0 - confidence) / 2.0
    mse_low, mse_high = np.quantile(boot_mse, [alpha, 1.0 - alpha])
    delta_low, delta_high = np.quantile(boot_delta_r2, [alpha, 1.0 - alpha])
    denominator = float(np.sum(global_loss))
    return {
        "unit": "sequence_component",
        "replicates": replicates,
        "confidence_level": confidence,
        "seed": seed,
        "n_components": len(groups),
        "mse_improvement_vs_consensus_global": {
            "direction": "positive_favors_proposed",
            "estimate": float(np.mean(improvement_global)),
            "ci_low": float(mse_low),
            "ci_high": float(mse_high),
        },
        "delta_r2_vs_consensus_per_job_selected_baseline": {
            "direction": "positive_favors_proposed",
            "estimate": None if denominator <= 0 else float(np.sum(improvement_baseline) / denominator),
            "ci_low": float(delta_low),
            "ci_high": float(delta_high),
        },
    }


def _context_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(prediction) & np.isfinite(target)
    if not np.any(valid):
        raise CampaignValidationError("context endpoint has no finite observations")
    error = prediction[valid] - target[valid]
    denominator = float(np.dot(target[valid], target[valid]))
    pearson = [_correlation(prediction[:, index], target[:, index]) for index in range(target.shape[1])]
    spearman = [_correlation(prediction[:, index], target[:, index], rank=True) for index in range(target.shape[1])]
    finite_pearson = [value for value in pearson if value is not None]
    finite_spearman = [value for value in spearman if value is not None]
    return {
        "n_finite_entries": int(np.sum(valid)),
        "mse": float(np.mean(np.square(error))),
        "r2_vs_zero": None if denominator <= 0.0 else float(1.0 - np.dot(error, error) / denominator),
        "mean_within_gene_pearson": None if not finite_pearson else float(np.mean(finite_pearson)),
        "mean_within_gene_spearman": None if not finite_spearman else float(np.mean(finite_spearman)),
        "n_genes_with_defined_pearson": len(finite_pearson),
        "n_genes_with_defined_spearman": len(finite_spearman),
    }


def _bootstrap_context(
    prediction: np.ndarray,
    target: np.ndarray,
    component_ids: np.ndarray,
    *,
    replicates: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    groups = _component_groups(component_ids)
    error = np.square(prediction - target)
    truth = np.square(target)
    valid = np.isfinite(error) & np.isfinite(truth)
    error_sums = np.asarray([np.sum(error[:, group][valid[:, group]]) for group in groups])
    truth_sums = np.asarray([np.sum(truth[:, group][valid[:, group]]) for group in groups])
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    chunk = max(1, 1_000_000 // len(groups))
    for start in range(0, replicates, chunk):
        stop = min(replicates, start + chunk)
        sampled = rng.integers(0, len(groups), size=(stop - start, len(groups)))
        denominator = np.sum(truth_sums[sampled], axis=1)
        values[start:stop] = np.divide(
            np.sum(error_sums[sampled], axis=1),
            denominator,
            out=np.full(stop - start, np.nan),
            where=denominator > 0.0,
        )
        values[start:stop] = 1.0 - values[start:stop]
    if not np.all(np.isfinite(values)):
        raise CampaignValidationError("context component bootstrap produced undefined replicates")
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(values, [alpha, 1.0 - alpha])
    point = _context_metrics(prediction, target)["r2_vs_zero"]
    return {"estimate": point, "ci_low": float(low), "ci_high": float(high)}


def _subset_metrics(consensus: Consensus, indices: np.ndarray, method_keys: Mapping[str, str]) -> dict[str, Any]:
    arrays = consensus.arrays
    if indices.size == 0:
        return {
            "n_genes": 0,
            "status": "undefined_empty",
            "methods": {method: None for method in ALL_METHODS},
            "per_job_selected_baseline": None,
        }
    target = arrays["observed_gene_mean"][indices]
    reference = arrays[method_keys["GLOBAL-TRAIN-MEAN"]][indices]
    methods = {
        method: regression_metrics(arrays[key][indices], target, reference)
        for method, key in method_keys.items()
    }
    return {
        "n_genes": int(indices.size),
        "status": "evaluated",
        "methods": methods,
        "per_job_selected_baseline": regression_metrics(
            arrays["consensus_job_selected_baseline_gene_mean_prediction"][indices],
            target,
            reference,
        ),
    }


def campaign_metrics(consensus: Consensus, lock: Mapping[str, Any]) -> dict[str, Any]:
    arrays = consensus.arrays
    _, method_keys = _prediction_key_map(lock)
    target = arrays["observed_gene_mean"].astype(np.float64)
    global_prediction = arrays[method_keys["GLOBAL-TRAIN-MEAN"]].astype(np.float64)
    proposed = arrays[method_keys[PRIMARY_METHOD]].astype(np.float64)
    baseline = arrays["consensus_job_selected_baseline_gene_mean_prediction"].astype(
        np.float64
    )
    primary = {
        method: regression_metrics(arrays[key], target, global_prediction)
        for method, key in method_keys.items()
    }
    selected_baseline_metrics = regression_metrics(
        baseline, target, global_prediction
    )
    bootstrap_contract = lock["bootstrap"]
    bootstrap = _bootstrap_primary(
        proposed,
        baseline,
        global_prediction,
        target,
        arrays["component_ids"],
        replicates=int(bootstrap_contract["replicates"]),
        confidence=float(bootstrap_contract["confidence_level"]),
        seed=int(bootstrap_contract["seed"]),
    )
    context = _context_metrics(
        arrays["family_inductive_decomp_centered_residual_prediction"].astype(np.float64),
        arrays["observed_gene_centered_residual"].astype(np.float64),
    )
    context_bootstrap = _bootstrap_context(
        arrays["family_inductive_decomp_centered_residual_prediction"].astype(np.float64),
        arrays["observed_gene_centered_residual"].astype(np.float64),
        arrays["component_ids"],
        replicates=int(bootstrap_contract["replicates"]),
        confidence=float(bootstrap_contract["confidence_level"]),
        seed=int(bootstrap_contract["seed"]),
    )
    component_size = arrays["component_size_full_release"]
    singleton = _subset_metrics(consensus, np.flatnonzero(component_size == 1), method_keys)
    multi = _subset_metrics(consensus, np.flatnonzero(component_size > 1), method_keys)
    component_bins: dict[str, Any] = {}
    for definition in lock["robustness_strata"]["component_size_bins"]:
        minimum = int(definition["minimum"])
        maximum = definition["maximum"]
        mask = component_size >= minimum
        if maximum is not None:
            mask &= component_size <= int(maximum)
        component_bins[str(definition["label"])] = _subset_metrics(consensus, np.flatnonzero(mask), method_keys)
    distance_quartiles = {
        f"q{quartile + 1}": _subset_metrics(consensus, np.flatnonzero(arrays["distance_quartile_zero_based"] == quartile), method_keys)
        for quartile in range(4)
    }
    length_quartiles = {
        f"q{quartile + 1}": _subset_metrics(consensus, np.flatnonzero(arrays["protein_length_quartile_zero_based"] == quartile), method_keys)
        for quartile in range(4)
    }
    selective: dict[str, Any] = {}
    n_genes = target.size
    for coverage in lock["coverage_levels"]:
        retained = max(1, min(n_genes, int(math.ceil(float(coverage) * n_genes))))
        indices = arrays["selective_order"][:retained]
        selective[f"{float(coverage):g}"] = {
            "requested_fraction": float(coverage),
            "retained_gene_count": retained,
            "realized_fraction": retained / n_genes,
            "role": "descriptive_secondary_only",
            "coverage_guarantee": "none",
            "metrics": _subset_metrics(consensus, indices, method_keys),
        }
    gates = lock["gates"]
    development = gates["development_signal"]
    method_gate = gates["method"]
    context_gate = gates["context"]
    proposed_r2 = primary[PRIMARY_METHOD]["r2_vs_consensus_global_train_mean"]
    baseline_r2 = selected_baseline_metrics["r2_vs_consensus_global_train_mean"]
    multi_r2 = multi["methods"][PRIMARY_METHOD]["r2_vs_consensus_global_train_mean"] if multi["status"] == "evaluated" else None
    farthest = distance_quartiles["q4"]
    farthest_r2 = farthest["methods"][PRIMARY_METHOD]["r2_vs_consensus_global_train_mean"] if farthest["status"] == "evaluated" else None
    development_pass = bool(
        proposed_r2 is not None
        and proposed_r2 > float(development["full_coverage_gene_mean_r2_min_exclusive"])
        and bootstrap["mse_improvement_vs_consensus_global"]["ci_low"]
        > float(development["mse_improvement_vs_global_bootstrap_lower_min_exclusive"])
        and multi_r2 is not None
        and multi_r2 > float(development["multi_gene_component_r2_min_exclusive"])
        and farthest_r2 is not None
        and farthest_r2 > float(development["farthest_distance_quartile_r2_min_exclusive"])
    )
    method_pass = bool(
        proposed_r2 is not None
        and baseline_r2 is not None
        and proposed_r2 - baseline_r2
        >= float(method_gate["delta_r2_vs_per_job_selected_baseline_min"])
        and bootstrap["delta_r2_vs_consensus_per_job_selected_baseline"]["ci_low"]
        > float(method_gate["delta_r2_bootstrap_lower_min_exclusive"])
    )
    context_pass = bool(
        context["r2_vs_zero"] is not None
        and context["r2_vs_zero"] > float(context_gate["residual_r2_min_exclusive"])
        and context_bootstrap["ci_low"] > float(context_gate["residual_r2_bootstrap_lower_min_exclusive"])
        and context["mean_within_gene_pearson"] is not None
        and context["mean_within_gene_pearson"] >= float(context_gate["mean_within_gene_pearson_min"])
    )
    return {
        "schema": CAMPAIGN_SCHEMA,
        "status": "complete_and_replayed",
        "primary_analysis": {
            "identity_percent": 30.0,
            "prediction_count_per_gene": 9,
            "aggregation": "arithmetic_mean",
            "r2_reference": "per_gene_consensus_GLOBAL-TRAIN-MEAN_prediction",
            "primary_method": PRIMARY_METHOD,
            "per_job_selected_baseline": selected_baseline_metrics,
            "job_baseline_selection": consensus.job_baseline_selection,
            "methods": primary,
        },
        "component_block_bootstrap": bootstrap,
        "secondary_context": {**context, "component_block_bootstrap_r2": context_bootstrap},
        "robustness": {
            "singleton_components": singleton,
            "multi_gene_components": multi,
            "component_size_bins": component_bins,
            "distance_quartiles": distance_quartiles,
            "distance_quartile_edges": arrays["distance_quartile_edges"].tolist(),
            "protein_length_quartiles": length_quartiles,
            "protein_length_quartile_edges": arrays["protein_length_quartile_edges"].tolist(),
        },
        "selective_prediction": selective,
        "seed_pair_dispersion": {
            "role": "descriptive_not_biological_replicates",
            "pairs": consensus.seed_pair_metrics,
        },
        "locked_gates": {
            "development_signal": development_pass,
            "method": method_pass,
            "context": context_pass,
            "external_3d": "not_evaluated_no_external_outcomes_read",
            "temporal_26q3": "not_evaluated_no_external_outcomes_read",
            "intended_nar_mainline_internal_gates_passed": development_pass and method_pass,
            "nar_submission_ready": False,
        },
        "claim_boundary": (
            "Passing internal gates does not establish novelty, external validation, "
            "NAR submission readiness, or acceptance. Failed and undefined gates remain reported."
        ),
    }


def validate_campaign(
    *,
    campaign_id: str,
    roster_path: Path,
    runs_root: Path,
    workspace_root: Path = ROOT,
    lock_path: Path = DEFAULT_LOCK_PATH,
    verify_current_host: bool = True,
    verify_workspace_files: bool = True,
) -> tuple[dict[str, Any], Consensus, dict[str, Any]]:
    workspace_root = workspace_root.resolve(strict=True)
    lock, lock_record = load_analysis_lock(lock_path)
    host_record = _assert_designated_server(lock) if verify_current_host else {"verification_bypassed_for_pure_test": True}
    roster, roster_sha256, indexed_roster = load_roster(
        roster_path,
        campaign_id=campaign_id,
        lock=lock,
        lock_record=lock_record,
    )
    if runs_root.is_symlink():
        raise CampaignValidationError("formal runs root may not be a symlink")
    runs_root = runs_root.resolve(strict=True)
    if not runs_root.is_dir():
        raise CampaignValidationError("formal runs root must be a real directory")
    observed_dirs = {
        path.name: path
        for path in runs_root.iterdir()
        if path.is_dir() or path.is_symlink()
    }
    if set(observed_dirs) != set(indexed_roster):
        missing = sorted(set(indexed_roster) - set(observed_dirs))
        extra = sorted(set(observed_dirs) - set(indexed_roster))
        raise CampaignValidationError(f"formal run directory set mismatch; missing={missing!r}, extra={extra!r}")
    runs = [
        validate_run_directory(
            observed_dirs[run_id],
            expected_run_id=run_id,
            expected_key=key,
            roster_sha256=roster_sha256,
            lock=lock,
            lock_record=lock_record,
            workspace_root=workspace_root,
            verify_workspace_files=verify_workspace_files,
        )
        for run_id, key in sorted(indexed_roster.items())
    ]
    _require_run_consistency(runs, workspace_root=workspace_root, verify_workspace_files=verify_workspace_files)
    consensus = aggregate_validated_runs(runs, lock)
    metrics = campaign_metrics(consensus, lock)
    input_manifest = {
        "schema": INPUT_MANIFEST_SCHEMA,
        "campaign_id": campaign_id,
        "status": "complete_inputs_validated_before_external_outcome_access",
        "external_outcomes_read": False,
        "analysis_lock": lock_record,
        "roster": {
            **_file_record(roster_path.resolve(strict=True), relative_to=roster_path.resolve(strict=True).parent),
            "sha256": roster_sha256,
            "schema": roster["schema"],
        },
        "designated_server": host_record,
        "validated_run_count": len(runs),
        "job_keys": [run.key.as_dict() for run in sorted(runs, key=lambda item: item.key)],
        "source_artifacts": [
            {"run_id": run.run_id, "run_key": run.key.as_dict(), **record}
            for run in sorted(runs, key=lambda item: item.key)
            for record in run.source_records
        ],
    }
    return metrics, consensus, input_manifest


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    normalized = {
        name: np.asarray(arrays[name])
        for name in sorted(arrays)
    }
    if any(value.dtype.hasobject for value in normalized.values()):
        raise AtomicPublishError("object arrays are forbidden in campaign outputs")
    with path.open("xb") as handle:
        np.savez_compressed(handle, **normalized)
        handle.flush()
        os.fsync(handle.fileno())


def _npz_artifact_record(
    path: Path,
    *,
    relative_to: Path,
    schema: str,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    return {
        **_file_record(path, relative_to=relative_to),
        "schema": schema,
        "arrays": {
            name: {
                "dtype": str(np.asarray(arrays[name]).dtype),
                "shape": list(np.asarray(arrays[name]).shape),
                "sha256": _hash_array(np.asarray(arrays[name])),
            }
            for name in sorted(arrays)
        },
    }


def publish_campaign(
    *,
    campaign_id: str,
    metrics: Mapping[str, Any],
    consensus: Consensus,
    input_manifest: Mapping[str, Any],
    output_root: Path,
) -> Path:
    if IDENTIFIER.fullmatch(campaign_id) is None:
        raise AtomicPublishError("invalid campaign id")
    if output_root.is_symlink():
        raise AtomicPublishError("campaign output root may not be a symlink")
    output_root.mkdir(parents=True, exist_ok=True)
    output_root = output_root.resolve(strict=True)
    destination = output_root / campaign_id
    if destination.exists() or destination.is_symlink():
        raise AtomicPublishError(f"refusing to overwrite immutable campaign output: {destination}")
    staging = output_root / f".{campaign_id}.partial-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o700)
    try:
        (staging / "campaign-input-manifest.json").write_bytes(_json_bytes(input_manifest))
        (staging / "campaign-metrics.json").write_bytes(_json_bytes(metrics))
        consensus_arrays = dict(consensus.arrays)
        consensus_arrays["schema"] = np.asarray(CONSENSUS_SCHEMA)
        _write_npz(staging / "consensus-evidence.npz", consensus_arrays)
        _, method_keys = _prediction_key_map_from_consensus(consensus)
        frozen_prediction_arrays = {
            "schema": np.asarray(CONSENSUS_PREDICTIONS_SCHEMA),
            "gene_ids": consensus.arrays["gene_ids"],
            "canonical_gene_indices": consensus.arrays["canonical_gene_indices"],
            "component_ids": consensus.arrays["component_ids"],
            "family_distance_stack_gene_mean_prediction": consensus.arrays[
                method_keys[PRIMARY_METHOD]
            ],
            "consensus_per_job_selected_baseline_gene_mean_prediction": consensus.arrays[
                "consensus_job_selected_baseline_gene_mean_prediction"
            ],
            "consensus_global_train_mean_prediction": consensus.arrays[
                method_keys["GLOBAL-TRAIN-MEAN"]
            ],
        }
        _write_npz(staging / "consensus-predictions.npz", frozen_prediction_arrays)
        artifacts = {
            name: _file_record(staging / filename, relative_to=staging)
            for name, filename in (
                ("campaign_input_manifest", "campaign-input-manifest.json"),
                ("campaign_metrics", "campaign-metrics.json"),
            )
        }
        artifacts["consensus_evidence"] = _npz_artifact_record(
            staging / "consensus-evidence.npz",
            relative_to=staging,
            schema=CONSENSUS_SCHEMA,
            arrays=consensus_arrays,
        )
        artifacts["consensus_predictions"] = _npz_artifact_record(
            staging / "consensus-predictions.npz",
            relative_to=staging,
            schema=CONSENSUS_PREDICTIONS_SCHEMA,
            arrays=frozen_prediction_arrays,
        )
        completion = {
            "schema": CAMPAIGN_COMPLETION_SCHEMA,
            "status": "complete",
            "campaign_id": campaign_id,
            "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "external_outcomes_read": False,
            "artifacts": artifacts,
            "integrity": {
                "complete_45_job_cartesian_product": True,
                "each_gene_exactly_once_per_seed_pair": True,
                "nine_predictions_per_gene": True,
                "consensus_is_arithmetic_mean": True,
                "each_job_baseline_selected_from_its_own_inner_validation_only": True,
                "cross_job_baseline_identity_selection_used": False,
                "external_outcomes_read": False,
                "completion_written_last": True,
            },
        }
        (staging / "completion.json").write_bytes(_json_bytes(completion))
        for path in staging.iterdir():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        os.rename(staging, destination)
        directory_fd = os.open(output_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return destination


def _prediction_key_map_from_consensus(consensus: Consensus) -> tuple[dict[str, str], dict[str, str]]:
    methods = {
        "GLOBAL-TRAIN-MEAN": "global_train_mean_prediction",
        "SEQUENCE-KNN": "sequence_knn_gene_mean_prediction",
        "SEQUENCE-WEIGHTED-KNN": "sequence_weighted_knn_gene_mean_prediction",
        "PLM-RIDGE": "plm_ridge_gene_mean_prediction",
        "STATIC-CONVEX-STACK": "static_convex_stack_gene_mean_prediction",
        "FAMILY-DISTANCE-STACK": "family_distance_stack_gene_mean_prediction",
        "IDFREE-PLM-MEAN-MLP": "idfree_plm_mean_mlp_prediction",
        "DECOMP-MEAN-ONLY": "decomp_mean_only_gene_mean_prediction",
        "FAMILY-INDUCTIVE-DECOMP": "family_inductive_decomp_gene_mean_prediction",
    }
    missing = set(methods.values()) - set(consensus.arrays)
    if missing:
        raise AtomicPublishError(f"consensus lacks registered prediction arrays: {sorted(missing)!r}")
    return {}, methods


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    roster = subparsers.add_parser("build-roster", help="freeze the exact 45-job prelaunch roster")
    roster.add_argument("--campaign-id", required=True)
    roster.add_argument("--output", type=Path, required=True)
    roster.add_argument("--lock", type=Path, default=DEFAULT_LOCK_PATH)
    roster.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    aggregate = subparsers.add_parser("aggregate", help="validate, aggregate, and atomically publish consensus")
    aggregate.add_argument("--campaign-id", required=True)
    aggregate.add_argument("--roster", type=Path, required=True)
    aggregate.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    aggregate.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    aggregate.add_argument("--workspace-root", type=Path, default=ROOT)
    aggregate.add_argument("--lock", type=Path, default=DEFAULT_LOCK_PATH)
    aggregate.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "build-roster":
            roster = build_roster(args.campaign_id, lock_path=args.lock, protocol_path=args.protocol)
            write_new_json(args.output, roster)
            print(f"{args.output.resolve()} sha256={_sha256_file(args.output)}")
            return 0
        metrics, consensus, input_manifest = validate_campaign(
            campaign_id=args.campaign_id,
            roster_path=args.roster,
            runs_root=args.runs_root,
            workspace_root=args.workspace_root,
            lock_path=args.lock,
        )
        if args.validate_only:
            print(json.dumps(_json_safe(metrics), indent=2, sort_keys=True, allow_nan=False))
            return 0
        destination = publish_campaign(
            campaign_id=args.campaign_id,
            metrics=metrics,
            consensus=consensus,
            input_manifest=input_manifest,
            output_root=args.output_root,
        )
        print(destination)
        return 0
    except (OSError, ValueError, CampaignValidationError, AtomicPublishError) as exc:
        print(f"[family-inductive-aggregate] ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
