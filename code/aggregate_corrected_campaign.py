"""Validate and aggregate the complete corrected-v2 evaluation campaign.

Individual condition-by-seed runs are deliberately not scientific results.  This
module revalidates their immutable artifacts and only promotes the prespecified
complete protocol x condition x seed matrix to a campaign-level scientific
result.  Missing, duplicate, non-canonical, pilot, or hash-inconsistent runs make
the campaign fail closed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import struct
import sys
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, NoReturn, Sequence

import numpy as np

from corrected_protocol import (
    atomic_write_json,
    dependency_metrics,
    derive_seed,
    fit_gene_mean_baseline,
    make_inner_split,
    make_kfold_outer_splits,
)
from embedding_provenance import (
    EMBEDDING_RELATIVE_PATH,
    MANIFEST_RELATIVE_PATH,
    NAMES_RELATIVE_PATH,
    NORMALIZED_RELATIVE_PATH,
    validate_embedding_bundle,
)
from run_corrected_experiments import (
    CANONICAL_ARGUMENTS,
    CANONICAL_CONDITIONS as CONDITIONS,
    CANONICAL_SEEDS,
    FULL_GRID,
    TRANSFER_HOLDOUT_COUNTS,
    TRANSFER_HOLDOUT_LINEAGES,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = ROOT / "results" / "corrected_v2"
PROTOCOLS = ("p1", "transfer-forward", "transfer-reverse")
REFERENCE_CONDITION = "DepPLM"
CANONICAL_N_GENES = 17_768
CANONICAL_N_CELLS = 1_103

EXPRESSION_INPUT_PATH = "data/depmap/OmicsExpressionProteinCodingGenesTPMLogp1.csv"
COMMON_INPUT_PATHS = (
    "data/processed/gene_effect.npy",
    EMBEDDING_RELATIVE_PATH,
    NORMALIZED_RELATIVE_PATH,
    NAMES_RELATIVE_PATH,
    MANIFEST_RELATIVE_PATH,
    "data/processed/gene_seqs.fa",
    "data/processed/ppi_graph.npz",
    "data/processed/genes.txt",
    "data/processed/cell_lines.txt",
    "data/processed/lineages.txt",
    EXPRESSION_INPUT_PATH,
    "data/depmap/OmicsSomaticMutations.csv",
)
TRANSFER_INPUT_PATH = "data/processed/splits.json"
CODE_PATHS = (
    "code/run_corrected_experiments.py",
    "code/corrected_protocol.py",
    "code/models.py",
    "code/03_embed_proteins.py",
    "code/embedding_provenance.py",
)
FIT_SCOPE = {
    "expression_pca": "training_cells_only",
    "mutation_vocabulary": "training_cells_only",
    "mutation_token_order": "training_cells_only",
    "comutation_pmi": "training_cells_only",
    "selectivity_label": "training_cells_only",
    "lineage_categories": "training_cells_only_unseen_is_zero",
}

REQUIRED_METRICS = (
    "mse",
    "mae",
    "gene_mean_baseline_mse",
    "gene_mean_baseline_mae",
    "residual_r2_vs_train_gene_mean",
    "residual_cellwise_pearson_mean",
)
REPORT_METRICS = (
    *REQUIRED_METRICS,
    "residual_pearson_pooled",
    "gene_wise_pearson_mean",
    "gene_wise_mse_mean",
    "gene_wise_residual_r2_mean",
    "fraction_genes_positive_residual_r2",
    "cellwise_pearson_mean",
    "cellwise_spearman_mean",
)
OUTER_TEST_PREDICTION_SCHEMA = "DepPLM-outer-test-predictions-v1"
OUTER_TEST_PREDICTION_ARRAYS = (
    "prediction",
    "train_gene_mean",
    "outer_test_indices",
    "source_cell_indices",
    "source_gene_indices",
    "cell_ids",
    "gene_ids",
)

SELECTION_METRIC = "residual_r2_vs_train_gene_mean"
SCHEDULER_NAME = "CosineAnnealingLR"
RUNTIME_VERSION_KEYS = ("numpy", "pandas", "scipy", "scikit_learn", "torch")
RUNTIME_ACCELERATOR_KEYS = (
    "cuda_available",
    "torch_cuda_version",
    "cudnn_version",
    "cuda_device_name",
    "cuda_device_capability",
)
AGGREGATION_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "torch")


@dataclass(frozen=True, order=True)
class RunKey:
    protocol: str
    condition: str
    seed: int

    def label(self) -> str:
        return f"{self.protocol}|{self.condition}|seed-{self.seed}"


@dataclass(frozen=True)
class CampaignSpec:
    protocols: tuple[str, ...]
    conditions: tuple[str, ...]
    seeds: tuple[int, ...]
    canonical_arguments: Mapping[str, int | float]
    tuning_grid: tuple[Mapping[str, Any], ...]
    condition_configs: Mapping[str, Mapping[str, Any]] | None = None

    def expected_keys(self) -> tuple[RunKey, ...]:
        return tuple(
            RunKey(protocol, condition, seed)
            for protocol in self.protocols
            for condition in self.conditions
            for seed in self.seeds
        )


@dataclass(frozen=True)
class SplitEvidence:
    sha256: str
    fold_seed: int
    outer_train: tuple[int, ...]
    inner_train: tuple[int, ...]
    inner_validation: tuple[int, ...]
    outer_test: tuple[int, ...]


@dataclass
class ValidatedRun:
    key: RunKey
    path: Path
    cohort_sha256: str
    input_hashes: dict[str, str]
    code_hashes: dict[str, str]
    runtime_fingerprint: dict[str, Any]
    runtime_fingerprint_sha256: str
    fold_metrics: dict[str, dict[str, float]]
    splits: dict[str, SplitEvidence]
    n_cells: int
    source_artifacts: list[dict[str, Any]]
    embedding_provenance_sha256: str


class RunValidationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def canonical_spec() -> CampaignSpec:
    """Return the campaign contract directly from the corrected runner constants."""
    return CampaignSpec(
        protocols=PROTOCOLS,
        conditions=tuple(sorted(CONDITIONS)),
        seeds=tuple(int(seed) for seed in CANONICAL_SEEDS),
        canonical_arguments=dict(CANONICAL_ARGUMENTS),
        tuning_grid=tuple(dict(config) for config in FULL_GRID),
        condition_configs={name: dict(config) for name, config in CONDITIONS.items()},
    )


def _sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity(path: Path) -> dict[str, int | str]:
    """Read one coherent source snapshot for execution-provenance checks."""
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise RunValidationError(
            "aggregator_source_changed",
            f"cannot read aggregation source {path}: {exc}",
        ) from exc
    return {
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


_AGGREGATOR_SOURCE_PATH = Path(__file__).resolve()
_AGGREGATOR_SOURCE_AT_IMPORT = _source_identity(_AGGREGATOR_SOURCE_PATH)


def _capture_aggregator_source_identity() -> dict[str, int | str]:
    """Bind validation to the same source bytes from which this module loaded."""
    current = _source_identity(_AGGREGATOR_SOURCE_PATH)
    if current != _AGGREGATOR_SOURCE_AT_IMPORT:
        raise RunValidationError(
            "aggregator_source_changed",
            "aggregation source differs from the source snapshot captured at import",
        )
    return current


def _assert_aggregator_source_unchanged(expected: Mapping[str, Any]) -> None:
    required = {"size_bytes", "sha256"}
    if not isinstance(expected, Mapping) or set(expected) != required:
        raise RunValidationError(
            "invalid_aggregation_provenance",
            "aggregation source provenance must contain exactly size_bytes and sha256",
        )
    current = _source_identity(_AGGREGATOR_SOURCE_PATH)
    normalized = {
        "size_bytes": expected.get("size_bytes"),
        "sha256": expected.get("sha256"),
    }
    if current != normalized:
        raise RunValidationError(
            "aggregator_source_changed",
            "aggregation source changed after validation began; refusing publication",
        )


def _report_source_identity(report: Mapping[str, Any]) -> dict[str, Any]:
    provenance = report.get("aggregation_provenance")
    if not isinstance(provenance, Mapping):
        raise RunValidationError(
            "invalid_aggregation_provenance",
            "campaign report is missing aggregation_provenance",
        )
    source = provenance.get("source")
    if not isinstance(source, Mapping):
        raise RunValidationError(
            "invalid_aggregation_provenance",
            "campaign report is missing aggregation_provenance.source",
        )
    if source.get("path") != "code/aggregate_corrected_campaign.py":
        raise RunValidationError(
            "invalid_aggregation_provenance",
            "campaign report has an unexpected aggregation source path",
        )
    return {
        "size_bytes": source.get("size_bytes"),
        "sha256": source.get("sha256"),
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RunValidationError(
            "invalid_json", f"cannot read valid JSON from {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RunValidationError("invalid_json", f"expected a JSON object in {path}")
    return payload


def _required_nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RunValidationError(
            "invalid_runtime_fingerprint", f"{field} must be a non-empty string"
        )
    return value


def _optional_nonempty_string(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    return _required_nonempty_string(value, field=field)


def _optional_nonnegative_integer(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RunValidationError(
            "invalid_runtime_fingerprint",
            f"{field} must be null or a non-negative integer",
        )
    return value


def _cuda_device_capability(value: Any, *, device_type: str) -> list[int] | None:
    field = "run_manifest.runtime.cuda_device_capability"
    if device_type == "cpu":
        if value is not None:
            raise RunValidationError(
                "invalid_runtime_fingerprint",
                f"{field} must be null when run_manifest.device is 'cpu'",
            )
        return None
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(
            isinstance(part, bool) or not isinstance(part, int) or part < 0
            for part in value
        )
    ):
        raise RunValidationError(
            "invalid_runtime_fingerprint",
            f"{field} must be a two-element list of non-negative integers for CUDA runs",
        )
    return list(value)


def _runtime_fingerprint(run_manifest: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Return the canonical execution fingerprint recorded before training."""
    device_type = run_manifest.get("device")
    if device_type not in {"cpu", "cuda"}:
        raise RunValidationError(
            "invalid_runtime_fingerprint",
            "run_manifest.device must be the resolved device type 'cpu' or 'cuda'",
        )

    runtime = run_manifest.get("runtime")
    if not isinstance(runtime, dict):
        raise RunValidationError(
            "invalid_runtime_fingerprint", "run_manifest.runtime is missing"
        )
    required_runtime_fields = {
        "amp_enabled",
        "amp_precision",
        *RUNTIME_ACCELERATOR_KEYS,
    }
    missing_runtime_fields = required_runtime_fields - set(runtime)
    if missing_runtime_fields:
        raise RunValidationError(
            "invalid_runtime_fingerprint",
            "run_manifest.runtime is missing fingerprint fields: "
            f"{sorted(missing_runtime_fields)!r}",
        )
    amp_enabled = runtime.get("amp_enabled")
    if not isinstance(amp_enabled, bool):
        raise RunValidationError(
            "invalid_runtime_fingerprint",
            "run_manifest.runtime.amp_enabled must be a boolean",
        )
    amp_precision = runtime.get("amp_precision")
    expected_amp_enabled = device_type == "cuda"
    expected_amp_precision = "bfloat16" if expected_amp_enabled else None
    if (
        amp_enabled is not expected_amp_enabled
        or amp_precision != expected_amp_precision
    ):
        raise RunValidationError(
            "invalid_runtime_fingerprint",
            "run_manifest AMP settings do not match the resolved device and training policy: "
            f"device={device_type!r}, amp_enabled={amp_enabled!r}, "
            f"amp_precision={amp_precision!r}",
        )

    cuda_available = runtime.get("cuda_available")
    if not isinstance(cuda_available, bool):
        raise RunValidationError(
            "invalid_runtime_fingerprint",
            "run_manifest.runtime.cuda_available must be a boolean",
        )
    torch_cuda_version = _optional_nonempty_string(
        runtime.get("torch_cuda_version"),
        field="run_manifest.runtime.torch_cuda_version",
    )
    cudnn_version = _optional_nonnegative_integer(
        runtime.get("cudnn_version"),
        field="run_manifest.runtime.cudnn_version",
    )
    cuda_device_name = _optional_nonempty_string(
        runtime.get("cuda_device_name"),
        field="run_manifest.runtime.cuda_device_name",
    )
    cuda_device_capability = _cuda_device_capability(
        runtime.get("cuda_device_capability"), device_type=device_type
    )
    if device_type == "cpu" and cuda_device_name is not None:
        raise RunValidationError(
            "invalid_runtime_fingerprint",
            "run_manifest.runtime.cuda_device_name must be null for CPU runs",
        )
    if device_type == "cuda":
        if not cuda_available:
            raise RunValidationError(
                "invalid_runtime_fingerprint",
                "run_manifest.runtime.cuda_available must be true for CUDA runs",
            )
        missing_cuda_values = [
            field
            for field, value in (
                ("torch_cuda_version", torch_cuda_version),
                ("cudnn_version", cudnn_version),
                ("cuda_device_name", cuda_device_name),
            )
            if value is None
        ]
        if missing_cuda_values:
            raise RunValidationError(
                "invalid_runtime_fingerprint",
                "CUDA runs require non-null accelerator fingerprint fields: "
                f"{missing_cuda_values!r}",
            )

    versions = run_manifest.get("versions")
    if not isinstance(versions, dict):
        raise RunValidationError(
            "invalid_runtime_fingerprint", "run_manifest.versions is missing"
        )
    selected_versions = {
        name: _required_nonempty_string(
            versions.get(name), field=f"run_manifest.versions.{name}"
        )
        for name in RUNTIME_VERSION_KEYS
    }
    fingerprint = {
        "device_type": device_type,
        "amp_enabled": amp_enabled,
        "amp_precision": amp_precision,
        "cuda_available": cuda_available,
        "torch_cuda_version": torch_cuda_version,
        "cudnn_version": cudnn_version,
        "cuda_device_name": cuda_device_name,
        "cuda_device_capability": cuda_device_capability,
        "python": _required_nonempty_string(
            run_manifest.get("python"), field="run_manifest.python"
        ),
        "platform": _required_nonempty_string(
            run_manifest.get("platform"), field="run_manifest.platform"
        ),
        "versions": selected_versions,
    }
    canonical = json.dumps(
        fingerprint, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return fingerprint, hashlib.sha256(canonical).hexdigest()


def _relative_path(base: Path, value: Any, *, field: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value:
        raise RunValidationError("invalid_path", f"{field} must be a non-empty string")
    portable = value.replace("\\", "/")
    relative = PurePosixPath(portable)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise RunValidationError(
            "unsafe_path", f"unsafe relative path in {field}: {value}"
        )
    normalized = relative.as_posix()
    resolved_base = base.resolve()
    resolved = (resolved_base / Path(*relative.parts)).resolve()
    if resolved != resolved_base and resolved_base not in resolved.parents:
        raise RunValidationError("unsafe_path", f"path escapes {base}: {value}")
    return normalized, resolved


def _validate_file_record(
    record: Any,
    *,
    base: Path,
    field: str,
) -> tuple[str, str, int]:
    if not isinstance(record, dict):
        raise RunValidationError("invalid_file_record", f"{field} must be an object")
    normalized, path = _relative_path(base, record.get("path"), field=f"{field}.path")
    expected_hash = record.get("sha256")
    expected_bytes = record.get("bytes")
    if not _valid_sha256(expected_hash):
        raise RunValidationError("invalid_file_record", f"{field}.sha256 is invalid")
    if (
        not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes < 0
    ):
        raise RunValidationError("invalid_file_record", f"{field}.bytes is invalid")
    if not path.is_file():
        raise RunValidationError(
            "missing_artifact", f"missing file referenced by {field}: {path}"
        )
    observed_bytes = path.stat().st_size
    if observed_bytes != expected_bytes:
        raise RunValidationError(
            "artifact_size_mismatch",
            f"{field} expected {expected_bytes} bytes but found {observed_bytes}: {path}",
        )
    observed_hash = _sha256_file(path)
    if observed_hash != expected_hash:
        raise RunValidationError(
            "artifact_hash_mismatch",
            f"{field} SHA-256 mismatch for {path}: expected {expected_hash}, found {observed_hash}",
        )
    return normalized, observed_hash, observed_bytes


def _validate_record_list(
    records: Any,
    *,
    base: Path,
    field: str,
) -> dict[str, tuple[str, int]]:
    if not isinstance(records, list):
        raise RunValidationError("invalid_file_record", f"{field} must be a list")
    validated: dict[str, tuple[str, int]] = {}
    for index, record in enumerate(records):
        path, sha256, size = _validate_file_record(
            record, base=base, field=f"{field}[{index}]"
        )
        if path in validated:
            raise RunValidationError(
                "duplicate_artifact_record", f"{field} lists {path} more than once"
            )
        validated[path] = (sha256, size)
    return validated


def _require_equal(actual: Any, expected: Any, *, field: str) -> None:
    if actual != expected:
        raise RunValidationError(
            "metadata_mismatch", f"{field} expected {expected!r}, found {actual!r}"
        )


def _require_false(value: Any, *, field: str) -> None:
    if value is not False:
        raise RunValidationError(
            "premature_scientific_result", f"{field} must be false, found {value!r}"
        )


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunValidationError("invalid_metric", f"{field} is not numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise RunValidationError("invalid_metric", f"{field} is not finite")
    return numeric


def _protocol_error(field: str, message: str) -> None:
    raise RunValidationError("protocol_evidence", f"{field} {message}")


def _require_protocol_equal(actual: Any, expected: Any, *, field: str) -> None:
    if actual != expected:
        _protocol_error(field, f"expected {expected!r}, found {actual!r}")


def _require_protocol_keys(
    value: Any, expected: set[str], *, field: str
) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        found = sorted(value) if isinstance(value, dict) else type(value).__name__
        _protocol_error(
            field, f"must contain exactly {sorted(expected)!r}; found {found!r}"
        )
    return value


def _protocol_integer(
    value: Any,
    *,
    field: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _protocol_error(field, f"must be an integer, found {value!r}")
    if minimum is not None and value < minimum:
        _protocol_error(field, f"must be at least {minimum}, found {value}")
    if maximum is not None and value > maximum:
        _protocol_error(field, f"must be at most {maximum}, found {value}")
    return int(value)


def _protocol_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _protocol_error(field, f"must be numeric, found {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric):
        _protocol_error(field, f"must be finite, found {value!r}")
    return numeric


def _require_protocol_number_equal(actual: Any, expected: float, *, field: str) -> None:
    numeric = _protocol_number(actual, field=field)
    if not math.isclose(numeric, expected, rel_tol=1e-12, abs_tol=1e-12):
        _protocol_error(field, f"expected {expected!r}, found {numeric!r}")


def _require_protocol_sha256(value: Any, *, field: str) -> str:
    if not _valid_sha256(value):
        _protocol_error(field, "must be a lowercase SHA-256 digest")
    return str(value)


def _hash_array(values: np.ndarray) -> str:
    """Match the runner's dtype- and shape-bound array hashing convention."""
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _load_gene_effect(
    workspace_root: Path, *, n_genes: int, n_cells: int
) -> np.ndarray:
    path = workspace_root / "data" / "processed" / "gene_effect.npy"
    try:
        values = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise RunValidationError(
            "protocol_evidence",
            f"cannot load training-reference labels from {path}: {exc}",
        ) from exc
    if not isinstance(values, np.ndarray) or values.shape != (n_genes, n_cells):
        found = getattr(values, "shape", None)
        raise RunValidationError(
            "protocol_evidence",
            f"gene_effect.npy shape must be {(n_genes, n_cells)!r}, found {found!r}",
        )
    return values


def _gene_mean_reference_sha256(
    gene_effect: np.ndarray,
    fit_indices: Sequence[int],
    cache: dict[tuple[int, ...], str],
) -> str:
    key = tuple(int(index) for index in fit_indices)
    if key not in cache:
        reference = fit_gene_mean_baseline(gene_effect, np.asarray(key, dtype=np.int64))
        cache[key] = _hash_array(reference)
    return cache[key]


def _prediction_artifact_error(field: str, message: str) -> NoReturn:
    raise RunValidationError("outer_test_prediction_artifact", f"{field} {message}")


def _array_metadata(values: np.ndarray) -> dict[str, Any]:
    return {"dtype": str(values.dtype), "shape": list(values.shape)}


def _validate_array_metadata(value: Any, *, array: np.ndarray, field: str) -> None:
    if not isinstance(value, dict) or set(value) != {"dtype", "shape"}:
        _prediction_artifact_error(field, "must contain exactly dtype and shape")
    shape = value.get("shape")
    if (
        not isinstance(shape, list)
        or any(isinstance(size, bool) or not isinstance(size, int) for size in shape)
        or value.get("dtype") != str(array.dtype)
        or shape != list(array.shape)
    ):
        _prediction_artifact_error(
            field,
            f"does not match the stored array metadata {_array_metadata(array)!r}",
        )


def _safe_rank_correlation(prediction: np.ndarray, target: np.ndarray) -> float:
    finite = np.isfinite(prediction) & np.isfinite(target)
    if int(finite.sum()) < 10:
        return float("nan")
    from scipy.stats import rankdata

    x = rankdata(prediction[finite])
    y = rankdata(target[finite])
    if np.std(x) <= 0.0 or np.std(y) <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _cellwise_spearman_metrics(
    prediction: np.ndarray, target: np.ndarray
) -> dict[str, float | int]:
    correlations = np.asarray(
        [
            _safe_rank_correlation(prediction[row], target[row])
            for row in range(prediction.shape[0])
        ],
        dtype=np.float64,
    )
    finite = correlations[np.isfinite(correlations)]
    return {
        "cellwise_spearman_mean": (
            float(np.mean(finite)) if finite.size else float("nan")
        ),
        "cellwise_spearman_sd": (
            float(np.std(finite)) if finite.size else float("nan")
        ),
        "n_finite_cellwise_spearman": int(finite.size),
    }


def _require_prediction_array(
    values: np.ndarray,
    *,
    field: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any] | None = None,
    unicode: bool = False,
) -> None:
    if not isinstance(values, np.ndarray):
        _prediction_artifact_error(field, "is not an ndarray")
    if values.shape != shape:
        _prediction_artifact_error(
            field, f"shape must be {shape!r}, found {values.shape!r}"
        )
    if unicode:
        if values.dtype.kind != "U":
            _prediction_artifact_error(
                field, f"dtype must be Unicode, found {values.dtype}"
            )
    elif dtype is None or values.dtype != dtype:
        _prediction_artifact_error(
            field, f"dtype must be {dtype}, found {values.dtype}"
        )
    if not values.flags.c_contiguous:
        _prediction_artifact_error(field, "must be stored in C row-major order")


def _validate_outer_test_predictions(
    result: Mapping[str, Any],
    *,
    run_dir: Path,
    fold_name: str,
    split: SplitEvidence,
    gene_effect: np.ndarray,
    cell_ids: Sequence[str],
    gene_ids: Sequence[str],
    source_cell_indices: Sequence[int],
    source_gene_indices: Sequence[int],
    artifact_records: Mapping[str, tuple[str, int]],
) -> dict[str, float]:
    field = f"{fold_name}-result.outer_test_predictions"
    record = result.get("outer_test_predictions")
    expected_record_fields = {"schema", "path", "bytes", "sha256", "arrays"}
    if not isinstance(record, dict) or set(record) != expected_record_fields:
        found = sorted(record) if isinstance(record, dict) else type(record).__name__
        _prediction_artifact_error(
            field,
            f"must contain exactly {sorted(expected_record_fields)!r}; found {found!r}",
        )
    if record.get("schema") != OUTER_TEST_PREDICTION_SCHEMA:
        _prediction_artifact_error(
            f"{field}.schema",
            f"must be {OUTER_TEST_PREDICTION_SCHEMA!r}",
        )
    relative_path, artifact_sha256, artifact_bytes = _validate_file_record(
        record, base=run_dir, field=field
    )
    expected_path = f"{fold_name}-predictions.npz"
    if relative_path != expected_path:
        _prediction_artifact_error(
            f"{field}.path", f"must be {expected_path!r}, found {relative_path!r}"
        )
    if artifact_records.get(relative_path) != (artifact_sha256, artifact_bytes):
        _prediction_artifact_error(
            field, "does not match the summary/completion artifact record"
        )
    _, artifact_path = _relative_path(run_dir, relative_path, field=f"{field}.path")

    try:
        with np.load(artifact_path, allow_pickle=False) as archive:
            stored_names = tuple(archive.files)
            if stored_names != OUTER_TEST_PREDICTION_ARRAYS:
                _prediction_artifact_error(
                    field,
                    "must contain the seven canonical arrays in canonical order; "
                    f"found {stored_names!r}",
                )
            arrays = {
                name: np.asarray(archive[name]) for name in OUTER_TEST_PREDICTION_ARRAYS
            }
    except RunValidationError:
        raise
    except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile) as exc:
        _prediction_artifact_error(
            field, f"cannot be loaded with allow_pickle=False: {exc}"
        )

    n_test = len(split.outer_test)
    n_genes = len(gene_ids)
    expected_shapes = {
        "prediction": (n_test, n_genes),
        "train_gene_mean": (n_genes,),
        "outer_test_indices": (n_test,),
        "source_cell_indices": (n_test,),
        "source_gene_indices": (n_genes,),
        "cell_ids": (n_test,),
        "gene_ids": (n_genes,),
    }
    for name in ("prediction", "train_gene_mean"):
        _require_prediction_array(
            arrays[name],
            field=f"{field}.arrays.{name}",
            shape=expected_shapes[name],
            dtype=np.dtype(np.float32),
        )
    for name in (
        "outer_test_indices",
        "source_cell_indices",
        "source_gene_indices",
    ):
        _require_prediction_array(
            arrays[name],
            field=f"{field}.arrays.{name}",
            shape=expected_shapes[name],
            dtype=np.dtype(np.int64),
        )
    for name in ("cell_ids", "gene_ids"):
        _require_prediction_array(
            arrays[name],
            field=f"{field}.arrays.{name}",
            shape=expected_shapes[name],
            unicode=True,
        )

    metadata = record.get("arrays")
    if not isinstance(metadata, dict) or set(metadata) != set(
        OUTER_TEST_PREDICTION_ARRAYS
    ):
        _prediction_artifact_error(
            f"{field}.arrays",
            "must describe exactly the seven canonical arrays",
        )
    for name in OUTER_TEST_PREDICTION_ARRAYS:
        _validate_array_metadata(
            metadata[name], array=arrays[name], field=f"{field}.arrays.{name}"
        )

    expected_outer_test = np.asarray(split.outer_test, dtype=np.int64)
    expected_source_cells = np.asarray(
        [source_cell_indices[index] for index in split.outer_test], dtype=np.int64
    )
    expected_source_genes = np.asarray(source_gene_indices, dtype=np.int64)
    expected_cell_ids = np.asarray(
        [cell_ids[index] for index in split.outer_test], dtype=str
    )
    expected_gene_ids = np.asarray(gene_ids, dtype=str)
    expected_values = {
        "outer_test_indices": expected_outer_test,
        "source_cell_indices": expected_source_cells,
        "source_gene_indices": expected_source_genes,
        "cell_ids": expected_cell_ids,
        "gene_ids": expected_gene_ids,
    }
    for name, expected in expected_values.items():
        if not np.array_equal(arrays[name], expected):
            _prediction_artifact_error(
                f"{field}.arrays.{name}",
                "does not match the validated cohort/split order",
            )

    expected_reference = fit_gene_mean_baseline(
        gene_effect, np.asarray(split.outer_train, dtype=np.int64)
    )
    if not np.array_equal(
        arrays["train_gene_mean"], expected_reference, equal_nan=True
    ):
        _prediction_artifact_error(
            f"{field}.arrays.train_gene_mean",
            "does not equal the outer-train reference recomputed from gene_effect.npy",
        )
    expected_reference_sha256 = _hash_array(expected_reference)
    test_metrics = result.get("test_metrics")
    if (
        not isinstance(test_metrics, Mapping)
        or test_metrics.get("gene_mean_reference_sha256") != expected_reference_sha256
    ):
        _prediction_artifact_error(
            f"{field}.arrays.train_gene_mean",
            "hash does not match test_metrics.gene_mean_reference_sha256",
        )

    prediction = arrays["prediction"]
    if not np.isfinite(prediction).all():
        _prediction_artifact_error(
            f"{field}.arrays.prediction", "must contain only finite values"
        )
    target = np.asarray(gene_effect[:, expected_outer_test].T, dtype=np.float32)
    try:
        recomputed: dict[str, float | int] = {
            **dependency_metrics(prediction, target, expected_reference),
            **_cellwise_spearman_metrics(prediction, target),
        }
    except ValueError as exc:
        _prediction_artifact_error(field, f"cannot be scored: {exc}")

    verified: dict[str, float] = {}
    for metric in REPORT_METRICS:
        expected = recomputed.get(metric)
        if isinstance(expected, bool) or not isinstance(expected, (int, float)):
            _prediction_artifact_error(
                field, f"recomputation did not produce numeric metric {metric!r}"
            )
        expected_number = float(expected)
        if not math.isfinite(expected_number):
            _prediction_artifact_error(
                field, f"recomputed metric {metric!r} is not finite"
            )
        actual = test_metrics.get(metric) if isinstance(test_metrics, Mapping) else None
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            _prediction_artifact_error(
                f"{fold_name}-result.test_metrics.{metric}",
                "must be numeric",
            )
        actual_number = float(actual)
        if not math.isfinite(actual_number) or not math.isclose(
            actual_number,
            expected_number,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise RunValidationError(
                "outer_test_metric_mismatch",
                f"{fold_name}-result.test_metrics.{metric} expected "
                f"{expected_number!r} from the bound predictions, found "
                f"{actual_number!r}",
            )
        verified[metric] = expected_number
    return verified


def _protocol_values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _protocol_values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _protocol_values_equal(a, b) for a, b in zip(left, right)
        )
    if (
        not isinstance(left, bool)
        and not isinstance(right, bool)
        and isinstance(left, (int, float))
        and isinstance(right, (int, float))
    ):
        left_number = float(left)
        right_number = float(right)
        if math.isnan(left_number) and math.isnan(right_number):
            return True
    return left == right


def _validate_reference_metrics(
    value: Any,
    *,
    field: str,
    expected_scope: str,
    expected_reference_sha256: str,
    expected_n_cells: int,
    expected_target_mode: str | None = None,
    expected_residual_scale_scope: str | None = None,
    expected_residual_scale: float | None = None,
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _protocol_error(field, "must be an object")
    required = {
        SELECTION_METRIC,
        "residual_cellwise_pearson_mean",
        "residual_cellwise_pearson_median",
        "residual_cellwise_pearson_sd",
        "n_finite_residual_cellwise_pearson",
        "n_cells",
        "reference_scope",
        "gene_mean_reference_sha256",
        "raw_cellwise_correlation_role",
    }
    missing = sorted(required - set(value))
    if missing:
        _protocol_error(field, f"is missing required fields {missing!r}")
    if expected_target_mode is None:
        unexpected = sorted(
            {"prediction_target_mode", "residual_scale", "residual_scale_scope"}
            & set(value)
        )
        if unexpected:
            _protocol_error(field, f"has unexpected research fields {unexpected!r}")
    else:
        _require_protocol_equal(
            value.get("prediction_target_mode"),
            expected_target_mode,
            field=f"{field}.prediction_target_mode",
        )
        if expected_target_mode == "absolute":
            unexpected = sorted(
                {"residual_scale", "residual_scale_scope"} & set(value)
            )
            if unexpected:
                _protocol_error(field, f"has invalid absolute-target fields {unexpected!r}")
        elif expected_target_mode == "train_gene_mean_residual":
            _require_protocol_equal(
                value.get("residual_scale_scope"),
                expected_residual_scale_scope,
                field=f"{field}.residual_scale_scope",
            )
            residual_scale = _protocol_number(
                value.get("residual_scale"), field=f"{field}.residual_scale"
            )
            if not 0.0 <= residual_scale <= 1.0:
                _protocol_error(f"{field}.residual_scale", "must be in [0, 1]")
            if expected_residual_scale is not None:
                _require_protocol_number_equal(
                    residual_scale,
                    expected_residual_scale,
                    field=f"{field}.residual_scale",
                )
        else:
            _protocol_error(
                f"{field}.prediction_target_mode",
                f"has unsupported target mode {expected_target_mode!r}",
            )
    _require_protocol_equal(
        value.get("reference_scope"), expected_scope, field=f"{field}.reference_scope"
    )
    _require_protocol_equal(
        value.get("gene_mean_reference_sha256"),
        expected_reference_sha256,
        field=f"{field}.gene_mean_reference_sha256",
    )
    _require_protocol_equal(
        value.get("n_cells"), expected_n_cells, field=f"{field}.n_cells"
    )
    _require_protocol_equal(
        value.get("raw_cellwise_correlation_role"),
        "secondary_descriptive_only",
        field=f"{field}.raw_cellwise_correlation_role",
    )
    _protocol_number(value.get(SELECTION_METRIC), field=f"{field}.{SELECTION_METRIC}")
    n_finite = _protocol_integer(
        value.get("n_finite_residual_cellwise_pearson"),
        field=f"{field}.n_finite_residual_cellwise_pearson",
        minimum=0,
        maximum=expected_n_cells,
    )
    for name in (
        "residual_cellwise_pearson_mean",
        "residual_cellwise_pearson_median",
        "residual_cellwise_pearson_sd",
    ):
        statistic = value.get(name)
        if n_finite:
            _protocol_number(statistic, field=f"{field}.{name}")
        elif statistic is not None:
            _protocol_error(
                f"{field}.{name}",
                "must be null when no residual cellwise correlations exist",
            )
    return value


def _validate_preprocessing_manifest(
    value: Any,
    *,
    field: str,
    fit_indices: Sequence[int],
    evaluation_indices: Sequence[int],
    lineages: Sequence[str],
    expression_input_features: int,
    n_string_edges: int,
    n_expr_pc: int,
    top_mut: int,
) -> None:
    record = _require_protocol_keys(
        value,
        {
            "fit_indices_sha256",
            "n_fit_cells",
            "expression",
            "mutation",
            "graph",
            "lineage",
            "selectivity_labels_sha256",
        },
        field=field,
    )
    expected_fit_hash = _hash_array(np.asarray(fit_indices, dtype="<i8"))
    _require_protocol_equal(
        record.get("fit_indices_sha256"),
        expected_fit_hash,
        field=f"{field}.fit_indices_sha256",
    )
    _require_protocol_equal(
        record.get("n_fit_cells"), len(fit_indices), field=f"{field}.n_fit_cells"
    )
    _require_protocol_sha256(
        record.get("selectivity_labels_sha256"),
        field=f"{field}.selectivity_labels_sha256",
    )

    expression = _require_protocol_keys(
        record.get("expression"),
        {
            "input_features",
            "output_components",
            "fitted_components",
            "mean_sha256",
            "scale_sha256",
            "components_sha256",
        },
        field=f"{field}.expression",
    )
    _require_protocol_equal(
        expression.get("input_features"),
        expression_input_features,
        field=f"{field}.expression.input_features",
    )
    _require_protocol_equal(
        expression.get("output_components"),
        n_expr_pc,
        field=f"{field}.expression.output_components",
    )
    _require_protocol_equal(
        expression.get("fitted_components"),
        min(n_expr_pc, len(fit_indices), expression_input_features),
        field=f"{field}.expression.fitted_components",
    )
    for name in ("mean_sha256", "scale_sha256", "components_sha256"):
        _require_protocol_sha256(
            expression.get(name), field=f"{field}.expression.{name}"
        )

    mutation = _require_protocol_keys(
        record.get("mutation"),
        {
            "vocabulary",
            "vocabulary_sha256",
            "token_order_sha256",
            "n_vocabulary",
            "n_token_genes",
            "k_tokens",
        },
        field=f"{field}.mutation",
    )
    vocabulary = mutation.get("vocabulary")
    if (
        not isinstance(vocabulary, list)
        or any(not isinstance(gene, str) or not gene for gene in vocabulary)
        or len(set(vocabulary)) != len(vocabulary)
    ):
        _protocol_error(
            f"{field}.mutation.vocabulary", "must be a list of unique non-empty strings"
        )
    _require_protocol_equal(
        mutation.get("vocabulary_sha256"),
        _hash_strings(vocabulary),
        field=f"{field}.mutation.vocabulary_sha256",
    )
    _require_protocol_sha256(
        mutation.get("token_order_sha256"),
        field=f"{field}.mutation.token_order_sha256",
    )
    _require_protocol_equal(
        mutation.get("n_vocabulary"),
        len(vocabulary),
        field=f"{field}.mutation.n_vocabulary",
    )
    n_token_genes = _protocol_integer(
        mutation.get("n_token_genes"),
        field=f"{field}.mutation.n_token_genes",
        minimum=0,
        maximum=len(vocabulary),
    )
    if len(vocabulary) > top_mut or n_token_genes > top_mut:
        _protocol_error(
            f"{field}.mutation", f"exceeds the configured top_mut limit of {top_mut}"
        )
    _require_protocol_equal(
        mutation.get("k_tokens"), 64, field=f"{field}.mutation.k_tokens"
    )

    graph = _require_protocol_keys(
        record.get("graph"),
        {
            "n_edges",
            "edge_weight_sha256",
            "edge_weight_mean",
            "n_non_neutral_weights",
        },
        field=f"{field}.graph",
    )
    _require_protocol_equal(
        graph.get("n_edges"), n_string_edges, field=f"{field}.graph.n_edges"
    )
    _require_protocol_sha256(
        graph.get("edge_weight_sha256"), field=f"{field}.graph.edge_weight_sha256"
    )
    non_neutral = _protocol_integer(
        graph.get("n_non_neutral_weights"),
        field=f"{field}.graph.n_non_neutral_weights",
        minimum=0,
        maximum=n_string_edges,
    )
    del non_neutral
    if n_string_edges:
        _protocol_number(
            graph.get("edge_weight_mean"), field=f"{field}.graph.edge_weight_mean"
        )
    else:
        _require_protocol_equal(
            graph.get("edge_weight_mean"), None, field=f"{field}.graph.edge_weight_mean"
        )

    lineage = _require_protocol_keys(
        record.get("lineage"),
        {"categories", "n_unknown_evaluation_cells", "unknown_encoding"},
        field=f"{field}.lineage",
    )
    expected_categories = sorted({lineages[index] for index in fit_indices})
    _require_protocol_equal(
        lineage.get("categories"),
        expected_categories,
        field=f"{field}.lineage.categories",
    )
    expected_unknown = sum(
        lineages[index] not in set(expected_categories) for index in evaluation_indices
    )
    _require_protocol_equal(
        lineage.get("n_unknown_evaluation_cells"),
        expected_unknown,
        field=f"{field}.lineage.n_unknown_evaluation_cells",
    )
    _require_protocol_equal(
        lineage.get("unknown_encoding"),
        "all_zero",
        field=f"{field}.lineage.unknown_encoding",
    )


def _validate_tuning_entry(
    value: Any,
    *,
    field: str,
    expected_config: Mapping[str, Any],
    expected_seed: int,
    epoch_budget: int,
    patience: int,
    validation_n_cells: int,
    reference_sha256: str,
    target_mode: str | None,
) -> tuple[float, int, dict[str, Any]]:
    expected_fields = {
        "config",
        "selection_score",
        "best_epoch",
        "seed",
        "trained_epochs",
        "scheduler",
        "scheduler_horizon_epochs",
        "history",
        "selection_metric",
        "best_validation_metrics",
    }
    if target_mode is not None:
        expected_fields.add("target_mode")
    if target_mode == "train_gene_mean_residual":
        expected_fields.add("selected_config_updates")
    record = _require_protocol_keys(
        value,
        expected_fields,
        field=field,
    )
    _require_protocol_equal(
        record.get("config"), dict(expected_config), field=f"{field}.config"
    )
    _require_protocol_equal(record.get("seed"), expected_seed, field=f"{field}.seed")
    if target_mode is not None:
        _require_protocol_equal(
            record.get("target_mode"), target_mode, field=f"{field}.target_mode"
        )
    _require_protocol_equal(
        record.get("scheduler"), SCHEDULER_NAME, field=f"{field}.scheduler"
    )
    _require_protocol_equal(
        record.get("scheduler_horizon_epochs"),
        epoch_budget,
        field=f"{field}.scheduler_horizon_epochs",
    )
    _require_protocol_equal(
        record.get("selection_metric"),
        SELECTION_METRIC,
        field=f"{field}.selection_metric",
    )
    trained_epochs = _protocol_integer(
        record.get("trained_epochs"),
        field=f"{field}.trained_epochs",
        minimum=1,
        maximum=epoch_budget,
    )
    history = record.get("history")
    if not isinstance(history, list) or len(history) != trained_epochs:
        _protocol_error(
            f"{field}.history", f"must contain exactly {trained_epochs} epoch records"
        )

    history_best_score = -math.inf
    history_best_epoch = 0
    history_best_metrics: Mapping[str, Any] | None = None
    bad_epochs = 0
    for epoch_number, raw_epoch in enumerate(history, start=1):
        epoch = _require_protocol_keys(
            raw_epoch,
            {"epoch", "train_loss", "validation"},
            field=f"{field}.history[{epoch_number - 1}]",
        )
        _require_protocol_equal(
            epoch.get("epoch"),
            epoch_number,
            field=f"{field}.history[{epoch_number - 1}].epoch",
        )
        train_loss = _protocol_number(
            epoch.get("train_loss"),
            field=f"{field}.history[{epoch_number - 1}].train_loss",
        )
        if train_loss < 0.0:
            _protocol_error(
                f"{field}.history[{epoch_number - 1}].train_loss",
                "must be non-negative",
            )
        validation = _validate_reference_metrics(
            epoch.get("validation"),
            field=f"{field}.history[{epoch_number - 1}].validation",
            expected_scope="inner_train_gene_mean",
            expected_reference_sha256=reference_sha256,
            expected_n_cells=validation_n_cells,
            expected_target_mode=target_mode,
            expected_residual_scale_scope=(
                "fit_on_current_inner_validation"
                if target_mode == "train_gene_mean_residual"
                else None
            ),
        )
        score = _protocol_number(
            validation.get(SELECTION_METRIC),
            field=(
                f"{field}.history[{epoch_number - 1}].validation.{SELECTION_METRIC}"
            ),
        )
        if score > history_best_score + 1e-8:
            history_best_score = score
            history_best_epoch = epoch_number
            history_best_metrics = validation
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience and epoch_number != trained_epochs:
            _protocol_error(
                f"{field}.history",
                f"continues after early stopping should trigger at epoch {epoch_number}",
            )
    if trained_epochs < epoch_budget and bad_epochs < patience:
        _protocol_error(
            f"{field}.trained_epochs",
            "ends before the epoch budget without satisfying early stopping",
        )

    selection_score = _protocol_number(
        record.get("selection_score"), field=f"{field}.selection_score"
    )
    best_epoch = _protocol_integer(
        record.get("best_epoch"),
        field=f"{field}.best_epoch",
        minimum=1,
        maximum=trained_epochs,
    )
    _require_protocol_number_equal(
        selection_score, history_best_score, field=f"{field}.selection_score"
    )
    _require_protocol_equal(best_epoch, history_best_epoch, field=f"{field}.best_epoch")
    best_metrics = _validate_reference_metrics(
        record.get("best_validation_metrics"),
        field=f"{field}.best_validation_metrics",
        expected_scope="inner_train_gene_mean",
        expected_reference_sha256=reference_sha256,
        expected_n_cells=validation_n_cells,
        expected_target_mode=target_mode,
        expected_residual_scale_scope=(
            "fit_on_current_inner_validation"
            if target_mode == "train_gene_mean_residual"
            else None
        ),
    )
    if history_best_metrics is None or not _protocol_values_equal(
        best_metrics, history_best_metrics
    ):
        _protocol_error(
            f"{field}.best_validation_metrics",
            "does not equal the validation record at best_epoch",
        )
    selected_updates: dict[str, Any] = {}
    if target_mode == "train_gene_mean_residual":
        raw_updates = record.get("selected_config_updates")
        if not isinstance(raw_updates, dict) or set(raw_updates) != {"residual_scale"}:
            _protocol_error(
                f"{field}.selected_config_updates",
                "must contain exactly residual_scale",
            )
        residual_scale = _protocol_number(
            raw_updates.get("residual_scale"),
            field=f"{field}.selected_config_updates.residual_scale",
        )
        if not 0.0 <= residual_scale <= 1.0:
            _protocol_error(
                f"{field}.selected_config_updates.residual_scale", "must be in [0, 1]"
            )
        _require_protocol_number_equal(
            residual_scale,
            best_metrics.get("residual_scale"),
            field=f"{field}.selected_config_updates.residual_scale",
        )
        selected_updates["residual_scale"] = residual_scale
    return selection_score, best_epoch, selected_updates


def _validate_fold_protocol_evidence(
    result: Mapping[str, Any],
    *,
    fold_name: str,
    condition: str,
    split: SplitEvidence,
    spec: CampaignSpec,
    lineages: Sequence[str],
    expression_input_features: int,
    n_string_edges: int,
    gene_effect: np.ndarray,
    reference_hash_cache: dict[tuple[int, ...], str],
) -> None:
    field = f"{fold_name}-result"
    _require_protocol_keys(
        result,
        {
            "fold",
            "fold_number",
            "condition",
            "protocol",
            "seed",
            "fold_seed",
            "best_config",
            "best_epoch",
            "selection",
            "optimization",
            "tuning",
            "test_metrics",
            "test_evaluation_calls",
            "preprocessing",
            "outer_test_predictions",
            "wall_seconds",
            "pilot",
            "scientific_result",
            "result_scope",
        },
        field=field,
    )
    wall_seconds = _protocol_number(
        result.get("wall_seconds"), field=f"{field}.wall_seconds"
    )
    if wall_seconds < 0.0:
        _protocol_error(f"{field}.wall_seconds", "must be non-negative")

    condition_configs = spec.condition_configs or CONDITIONS
    try:
        condition_config = condition_configs[condition]
    except KeyError:
        _protocol_error(
            f"{field}.condition", f"has no runner configuration for {condition!r}"
        )
    expected_configs = [
        {**condition_config, **dict(candidate)} for candidate in spec.tuning_grid
    ]
    tuning = result.get("tuning")
    if not isinstance(tuning, list) or len(tuning) != len(expected_configs):
        _protocol_error(
            f"{field}.tuning",
            f"must contain the {len(expected_configs)} canonical grid candidates in order",
        )

    epoch_budget = _protocol_integer(
        spec.canonical_arguments.get("inner_max_epochs"),
        field="campaign.inner_max_epochs",
        minimum=1,
    )
    patience = _protocol_integer(
        spec.canonical_arguments.get("patience"), field="campaign.patience", minimum=1
    )
    inner_reference_sha = _gene_mean_reference_sha256(
        gene_effect, split.inner_train, reference_hash_cache
    )
    tuning_seed = derive_seed(split.fold_seed, "tune")
    tuning_outcomes = [
        _validate_tuning_entry(
            entry,
            field=f"{field}.tuning[{index}]",
            expected_config=expected_config,
            expected_seed=tuning_seed,
            epoch_budget=epoch_budget,
            patience=patience,
            validation_n_cells=len(split.inner_validation),
            reference_sha256=inner_reference_sha,
            target_mode=(
                str(expected_config["target_mode"])
                if "target_mode" in expected_config
                else None
            ),
        )
        for index, (entry, expected_config) in enumerate(zip(tuning, expected_configs))
    ]
    winning_index = max(
        range(len(tuning_outcomes)), key=lambda index: tuning_outcomes[index][0]
    )
    winning_score, winning_epoch, winning_updates = tuning_outcomes[winning_index]
    expected_best_config = {
        **expected_configs[winning_index],
        **winning_updates,
    }
    _require_protocol_equal(
        result.get("best_config"),
        expected_best_config,
        field=f"{field}.best_config",
    )
    _require_protocol_equal(
        result.get("best_epoch"), winning_epoch, field=f"{field}.best_epoch"
    )

    selection = _require_protocol_keys(
        result.get("selection"),
        {"metric", "maximize", "score"},
        field=f"{field}.selection",
    )
    _require_protocol_equal(
        selection.get("metric"), SELECTION_METRIC, field=f"{field}.selection.metric"
    )
    _require_protocol_equal(
        selection.get("maximize"), True, field=f"{field}.selection.maximize"
    )
    _require_protocol_number_equal(
        selection.get("score"), winning_score, field=f"{field}.selection.score"
    )

    optimization = _require_protocol_keys(
        result.get("optimization"),
        {
            "scheduler",
            "scheduler_horizon_epochs",
            "refit_epochs",
            "selection_epoch_budget",
        },
        field=f"{field}.optimization",
    )
    _require_protocol_equal(
        optimization.get("scheduler"),
        SCHEDULER_NAME,
        field=f"{field}.optimization.scheduler",
    )
    _require_protocol_equal(
        optimization.get("scheduler_horizon_epochs"),
        epoch_budget,
        field=f"{field}.optimization.scheduler_horizon_epochs",
    )
    _require_protocol_equal(
        optimization.get("selection_epoch_budget"),
        epoch_budget,
        field=f"{field}.optimization.selection_epoch_budget",
    )
    _require_protocol_equal(
        optimization.get("refit_epochs"),
        winning_epoch,
        field=f"{field}.optimization.refit_epochs",
    )

    n_expr_pc = _protocol_integer(
        spec.canonical_arguments.get("n_expr_pc"), field="campaign.n_expr_pc", minimum=1
    )
    top_mut = _protocol_integer(
        spec.canonical_arguments.get("top_mut"), field="campaign.top_mut", minimum=1
    )
    preprocessing = _require_protocol_keys(
        result.get("preprocessing"),
        {"inner_train_fit", "outer_train_refit"},
        field=f"{field}.preprocessing",
    )
    _validate_preprocessing_manifest(
        preprocessing.get("inner_train_fit"),
        field=f"{field}.preprocessing.inner_train_fit",
        fit_indices=split.inner_train,
        evaluation_indices=split.inner_validation,
        lineages=lineages,
        expression_input_features=expression_input_features,
        n_string_edges=n_string_edges,
        n_expr_pc=n_expr_pc,
        top_mut=top_mut,
    )
    _validate_preprocessing_manifest(
        preprocessing.get("outer_train_refit"),
        field=f"{field}.preprocessing.outer_train_refit",
        fit_indices=split.outer_train,
        evaluation_indices=split.outer_test,
        lineages=lineages,
        expression_input_features=expression_input_features,
        n_string_edges=n_string_edges,
        n_expr_pc=n_expr_pc,
        top_mut=top_mut,
    )

    outer_reference_sha = _gene_mean_reference_sha256(
        gene_effect, split.outer_train, reference_hash_cache
    )
    _validate_reference_metrics(
        result.get("test_metrics"),
        field=f"{field}.test_metrics",
        expected_scope="outer_train_gene_mean",
        expected_reference_sha256=outer_reference_sha,
        expected_n_cells=len(split.outer_test),
        expected_target_mode=(
            str(expected_best_config["target_mode"])
            if "target_mode" in expected_best_config
            else None
        ),
        expected_residual_scale_scope=(
            "frozen_from_inner_validation"
            if expected_best_config.get("target_mode")
            == "train_gene_mean_residual"
            else None
        ),
        expected_residual_scale=(
            float(expected_best_config["residual_scale"])
            if "residual_scale" in expected_best_config
            else None
        ),
    )


def _validate_direct_hash(
    payload: Mapping[str, Any],
    key: str,
    path: Path,
    *,
    field: str,
) -> str:
    expected = payload.get(key)
    if not _valid_sha256(expected):
        raise RunValidationError("invalid_hash", f"{field}.{key} is invalid")
    observed = _sha256_file(path)
    if observed != expected:
        raise RunValidationError(
            "artifact_hash_mismatch",
            f"{field}.{key} expected {expected}, found {observed} for {path}",
        )
    return observed


def _expected_fold_names(protocol: str) -> tuple[str, ...]:
    if protocol == "p1":
        return tuple(f"fold-{index:02d}" for index in range(5))
    if protocol == "transfer-forward":
        return ("complement-to-bkm",)
    if protocol == "transfer-reverse":
        return ("bkm-to-complement",)
    raise RunValidationError("noncanonical_run", f"unsupported protocol: {protocol!r}")


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _hash_strings(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _read_manifest_lines(path: Path, *, field: str) -> list[str]:
    try:
        values = [
            line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        ]
    except (OSError, UnicodeError) as exc:
        raise RunValidationError(
            "missing_artifact", f"cannot read {field} from {path}: {exc}"
        ) from exc
    if not values or any(not value for value in values):
        raise RunValidationError(
            "invalid_cohort_manifest", f"{field} must contain non-empty lines"
        )
    return values


def _read_expression_gene_names(path: Path, genes: Sequence[str]) -> list[str]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle))
    except (OSError, UnicodeError, StopIteration, csv.Error) as exc:
        raise RunValidationError(
            "invalid_cohort_manifest",
            f"cannot read the expression header from {path}: {exc}",
        ) from exc
    # DepMap serializes the row-index column with an empty header.  Pandas still
    # reads it as the first (identifier) column, so the independent verifier must
    # accept that canonical representation while requiring at least one feature.
    if len(header) < 2:
        raise RunValidationError(
            "invalid_cohort_manifest", "expression input has no usable feature columns"
        )
    allowed = set(genes)
    selected: list[str] = []
    seen: set[str] = set()
    for column in header[1:]:
        symbol = column.split(" (")[0]
        if symbol in allowed and symbol not in seen:
            selected.append(symbol)
            seen.add(symbol)
    if not selected:
        raise RunValidationError(
            "invalid_cohort_manifest",
            "expression input has no columns mapped to the processed gene set",
        )
    return selected


def _integer_sequence(
    value: Any,
    *,
    field: str,
    expected_length: int | None = None,
    upper_bound: int | None = None,
) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise RunValidationError(
            "split_semantics", f"{field} must be a non-empty integer list"
        )
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise RunValidationError(
            "split_semantics", f"{field} must contain only integers"
        )
    values = tuple(int(item) for item in value)
    if expected_length is not None and len(values) != expected_length:
        raise RunValidationError(
            "split_semantics",
            f"{field} has {len(values)} entries; expected {expected_length}",
        )
    if len(set(values)) != len(values):
        raise RunValidationError("split_semantics", f"{field} contains duplicates")
    if any(
        item < 0 or (upper_bound is not None and item >= upper_bound) for item in values
    ):
        raise RunValidationError(
            "split_semantics", f"{field} contains out-of-range indices"
        )
    return values


def _validate_cohort_manifest(
    cohort: Mapping[str, Any], *, workspace_root: Path
) -> tuple[int, list[str], list[str], tuple[int, ...]]:
    expected_fields = {
        "n_genes",
        "n_cells",
        "n_expression_features_raw",
        "n_string_edges",
        "genes_sha256",
        "cell_ids_sha256",
        "expression_gene_names_sha256",
        "source_cell_indices",
        "source_gene_indices",
    }
    if set(cohort) != expected_fields:
        raise RunValidationError(
            "invalid_cohort_manifest",
            "cohort-manifest.json fields do not match the corrected-v2.2 schema",
        )
    genes = _read_manifest_lines(
        workspace_root / "data" / "processed" / "genes.txt", field="genes.txt"
    )
    cell_ids = _read_manifest_lines(
        workspace_root / "data" / "processed" / "cell_lines.txt",
        field="cell_lines.txt",
    )
    lineages = _read_manifest_lines(
        workspace_root / "data" / "processed" / "lineages.txt", field="lineages.txt"
    )
    if len(genes) != CANONICAL_N_GENES or len(cell_ids) != CANONICAL_N_CELLS:
        raise RunValidationError(
            "invalid_cohort_manifest",
            "canonical cohort dimensions must be "
            f"{CANONICAL_N_GENES} genes by {CANONICAL_N_CELLS} cells; found "
            f"{len(genes)} by {len(cell_ids)}",
        )
    if len(set(genes)) != len(genes) or len(set(cell_ids)) != len(cell_ids):
        raise RunValidationError(
            "invalid_cohort_manifest", "gene and cell identifiers must be unique"
        )
    if len(lineages) != len(cell_ids):
        raise RunValidationError(
            "invalid_cohort_manifest", "lineages.txt does not align with cell_lines.txt"
        )
    for field, expected in (("n_genes", len(genes)), ("n_cells", len(cell_ids))):
        value = cohort.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise RunValidationError(
                "invalid_cohort_manifest",
                f"cohort.{field} expected {expected}, found {value!r}",
            )
    expression_names = _read_expression_gene_names(
        workspace_root / EXPRESSION_INPUT_PATH, genes
    )
    n_expression_features = cohort.get("n_expression_features_raw")
    if (
        isinstance(n_expression_features, bool)
        or not isinstance(n_expression_features, int)
        or n_expression_features != len(expression_names)
    ):
        raise RunValidationError(
            "invalid_cohort_manifest",
            "cohort.n_expression_features_raw does not match the current expression header",
        )
    n_string_edges = cohort.get("n_string_edges")
    if (
        isinstance(n_string_edges, bool)
        or not isinstance(n_string_edges, int)
        or n_string_edges <= 0
    ):
        raise RunValidationError(
            "invalid_cohort_manifest",
            "cohort.n_string_edges must be a positive integer",
        )
    expected_hashes = {
        "genes_sha256": _hash_strings(genes),
        "cell_ids_sha256": _hash_strings(cell_ids),
    }
    for field, expected in expected_hashes.items():
        if cohort.get(field) != expected:
            raise RunValidationError(
                "invalid_cohort_manifest",
                f"cohort.{field} does not match the current processed manifest",
            )
    if cohort.get("expression_gene_names_sha256") != _hash_strings(expression_names):
        raise RunValidationError(
            "invalid_cohort_manifest",
            "cohort.expression_gene_names_sha256 does not match the current expression header",
        )
    source_cells = _integer_sequence(
        cohort.get("source_cell_indices"),
        field="cohort.source_cell_indices",
        expected_length=len(cell_ids),
    )
    source_genes = _integer_sequence(
        cohort.get("source_gene_indices"),
        field="cohort.source_gene_indices",
        expected_length=len(genes),
    )
    if source_cells != tuple(range(len(cell_ids))) or source_genes != tuple(
        range(len(genes))
    ):
        raise RunValidationError(
            "invalid_cohort_manifest",
            "non-pilot canonical runs must use the complete processed cohort in source order",
        )
    return len(cell_ids), cell_ids, lineages, source_cells


def _index_sha256(indices: Sequence[int]) -> str:
    payload = struct.pack(f"<{len(indices)}q", *(int(index) for index in indices))
    return hashlib.sha256(payload).hexdigest()


def _validate_index_record(
    record: Any,
    *,
    field: str,
    n_cells: int,
    cell_ids: Sequence[str],
) -> tuple[int, ...]:
    if not isinstance(record, dict) or set(record) != {"indices", "cell_ids", "sha256"}:
        raise RunValidationError(
            "split_semantics", f"{field} does not match the index-record schema"
        )
    indices = _integer_sequence(
        record.get("indices"), field=f"{field}.indices", upper_bound=n_cells
    )
    expected_ids = [cell_ids[index] for index in indices]
    if record.get("cell_ids") != expected_ids:
        raise RunValidationError(
            "split_semantics", f"{field}.cell_ids do not match the indexed cohort rows"
        )
    if record.get("sha256") != _index_sha256(indices):
        raise RunValidationError(
            "split_semantics",
            f"{field}.sha256 does not match its little-endian int64 indices",
        )
    return indices


def _lineage_counts(indices: Sequence[int], lineages: Sequence[str]) -> dict[str, int]:
    return dict(sorted(Counter(lineages[index] for index in indices).items()))


def _load_transfer_contract(
    workspace_root: Path,
    *,
    n_cells: int,
    lineages: Sequence[str],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    payload = _load_json(workspace_root / TRANSFER_INPUT_PATH)
    expected_fields = {"P1_folds", "P2_train", "P2_test", "P3", "lineage_idx"}
    if set(payload) != expected_fields:
        raise RunValidationError(
            "split_semantics",
            "splits.json fields do not match the canonical processed-data schema",
        )
    complement = _integer_sequence(
        payload.get("P2_train"), field="splits.P2_train", upper_bound=n_cells
    )
    bkm = _integer_sequence(
        payload.get("P2_test"), field="splits.P2_test", upper_bound=n_cells
    )
    if set(complement).intersection(bkm) or set(complement).union(bkm) != set(
        range(n_cells)
    ):
        raise RunValidationError(
            "split_semantics",
            "stored transfer partitions must be disjoint and cover the cohort",
        )
    observed = _lineage_counts(bkm, lineages)
    if observed != dict(sorted(TRANSFER_HOLDOUT_COUNTS.items())):
        raise RunValidationError(
            "split_semantics",
            f"stored BKM holdout counts {observed!r} do not match {TRANSFER_HOLDOUT_COUNTS!r}",
        )
    leaked = set(lineages[index] for index in complement).intersection(
        TRANSFER_HOLDOUT_LINEAGES
    )
    if leaked:
        raise RunValidationError(
            "split_semantics",
            f"BKM lineages appear in the transfer complement: {sorted(leaked)!r}",
        )
    return complement, bkm


def _validate_split_manifest(
    split: Mapping[str, Any],
    *,
    split_path: Path,
    protocol: str,
    fold_name: str,
    run_seed: int,
    result_fold_seed: Any,
    n_cells: int,
    cell_ids: Sequence[str],
    lineages: Sequence[str],
    source_cell_indices: Sequence[int],
    transfer_contract: tuple[tuple[int, ...], tuple[int, ...]] | None,
    n_outer_folds: int,
    inner_val_fraction: float,
) -> SplitEvidence:
    expected_fields = {
        "seed",
        "outer_train",
        "inner_train",
        "inner_validation",
        "outer_test",
        "fit_scope",
        "k_tokens",
        "fold",
        "protocol",
        "source_cell_indices",
        "lineage_counts",
    }
    if set(split) != expected_fields:
        raise RunValidationError(
            "split_semantics", f"{fold_name}-split fields do not match the v2.2 schema"
        )
    expected_seed = derive_seed(run_seed, fold_name)
    _require_equal(split.get("protocol"), protocol, field=f"{fold_name}-split.protocol")
    _require_equal(split.get("fold"), fold_name, field=f"{fold_name}-split.fold")
    _require_equal(split.get("seed"), expected_seed, field=f"{fold_name}-split.seed")
    _require_equal(
        result_fold_seed, expected_seed, field=f"{fold_name}-result.fold_seed"
    )
    _require_equal(
        split.get("fit_scope"), FIT_SCOPE, field=f"{fold_name}-split.fit_scope"
    )
    _require_equal(split.get("k_tokens"), 64, field=f"{fold_name}-split.k_tokens")

    partitions = {
        name: _validate_index_record(
            split.get(name),
            field=f"{fold_name}-split.{name}",
            n_cells=n_cells,
            cell_ids=cell_ids,
        )
        for name in ("outer_train", "inner_train", "inner_validation", "outer_test")
    }
    outer_train = set(partitions["outer_train"])
    inner_train = set(partitions["inner_train"])
    inner_validation = set(partitions["inner_validation"])
    outer_test = set(partitions["outer_test"])
    if outer_train.intersection(outer_test) or outer_train.union(outer_test) != set(
        range(n_cells)
    ):
        raise RunValidationError(
            "split_semantics",
            f"{fold_name} outer partitions must cover the cohort once",
        )
    if (
        inner_train.intersection(inner_validation)
        or inner_train.union(inner_validation) != outer_train
        or inner_train.intersection(outer_test)
        or inner_validation.intersection(outer_test)
    ):
        raise RunValidationError(
            "split_semantics",
            f"{fold_name} inner train/validation must partition outer train and exclude test",
        )

    source_records = split.get("source_cell_indices")
    partition_names = tuple(partitions)
    if not isinstance(source_records, dict) or set(source_records) != set(
        partition_names
    ):
        raise RunValidationError(
            "split_semantics", f"{fold_name}-split.source_cell_indices is incomplete"
        )
    for name in partition_names:
        expected = [source_cell_indices[index] for index in partitions[name]]
        if source_records.get(name) != expected:
            raise RunValidationError(
                "split_semantics",
                f"{fold_name}-split.source_cell_indices.{name} is inconsistent",
            )
    expected_lineage_counts = {
        "outer_train": _lineage_counts(partitions["outer_train"], lineages),
        "outer_test": _lineage_counts(partitions["outer_test"], lineages),
    }
    if split.get("lineage_counts") != expected_lineage_counts:
        raise RunValidationError(
            "split_semantics", f"{fold_name}-split.lineage_counts is inconsistent"
        )

    if protocol == "p1":
        expected_fold_names = _expected_fold_names(protocol)
        if n_outer_folds != len(expected_fold_names):
            raise RunValidationError(
                "split_semantics",
                f"P1 requires {len(expected_fold_names)} outer folds, found {n_outer_folds}",
            )
        fold_number = expected_fold_names.index(fold_name)
        expected_outer = make_kfold_outer_splits(n_cells, n_outer_folds, run_seed)[
            fold_number
        ]
        expected_outer_train = tuple(int(index) for index in expected_outer[0])
        expected_outer_test = tuple(int(index) for index in expected_outer[1])
    elif transfer_contract is not None:
        complement, bkm = transfer_contract
        expected_outer_train, expected_outer_test = (
            (complement, bkm) if protocol == "transfer-forward" else (bkm, complement)
        )
    else:
        raise RunValidationError(
            "split_semantics", f"{fold_name} has no canonical outer-split contract"
        )
    if (
        partitions["outer_train"] != expected_outer_train
        or partitions["outer_test"] != expected_outer_test
    ):
        raise RunValidationError(
            "split_semantics",
            f"{fold_name} does not match the seed-derived canonical outer split",
        )

    expected_inner_train, expected_inner_validation = make_inner_split(
        np.asarray(expected_outer_train, dtype=np.int64),
        np.asarray(lineages, dtype=str),
        val_fraction=inner_val_fraction,
        seed=derive_seed(expected_seed, "inner-split"),
    )
    if partitions["inner_train"] != tuple(
        int(index) for index in expected_inner_train
    ) or partitions["inner_validation"] != tuple(
        int(index) for index in expected_inner_validation
    ):
        raise RunValidationError(
            "split_semantics",
            f"{fold_name} does not match the seed-derived canonical inner split",
        )

    return SplitEvidence(
        sha256=_sha256_file(split_path),
        fold_seed=expected_seed,
        outer_train=partitions["outer_train"],
        inner_train=partitions["inner_train"],
        inner_validation=partitions["inner_validation"],
        outer_test=partitions["outer_test"],
    )


def _validate_provenance_inventory(
    records: Mapping[str, tuple[str, int]], *, protocol: str, field: str
) -> None:
    expected = set(CODE_PATHS) if field == "code" else set(COMMON_INPUT_PATHS)
    if field == "inputs" and protocol != "p1":
        expected.add(TRANSFER_INPUT_PATH)
    if set(records) != expected:
        missing = sorted(expected - set(records))
        unexpected = sorted(set(records) - expected)
        raise RunValidationError(
            "noncanonical_provenance",
            f"run_manifest.{field} inventory mismatch; missing={missing!r}, unexpected={unexpected!r}",
        )


def _compare_float(actual: Any, expected: float, *, field: str) -> None:
    numeric = _finite_number(actual, field=field)
    if not math.isclose(numeric, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise RunValidationError(
            "summary_metric_mismatch",
            f"{field} expected {expected!r} from fold artifacts, found {numeric!r}",
        )


def validate_run_directory(
    run_dir: Path,
    *,
    workspace_root: Path,
    spec: CampaignSpec,
    _gene_effect: np.ndarray | None = None,
    _reference_hash_cache: dict[tuple[int, ...], str] | None = None,
    _expected_embedding_provenance: Mapping[str, Any] | None = None,
) -> ValidatedRun:
    """Revalidate a completion-bearing run and return its immutable evidence."""
    completion_path = run_dir / "completion.json"
    summary_path = run_dir / "summary.json"
    run_manifest_path = run_dir / "run-manifest.json"
    cohort_path = run_dir / "cohort-manifest.json"
    for required_path in (
        completion_path,
        summary_path,
        run_manifest_path,
        cohort_path,
    ):
        if not required_path.is_file():
            raise RunValidationError(
                "missing_artifact", f"completion-bearing run lacks {required_path.name}"
            )

    completion = _load_json(completion_path)
    summary = _load_json(summary_path)
    run_manifest = _load_json(run_manifest_path)
    cohort = _load_json(cohort_path)

    _require_equal(
        completion.get("schema"),
        "DepPLM-corrected-v2.2-completion",
        field="completion.schema",
    )
    _require_equal(
        completion.get("run_status"), "protocol_complete", field="completion.run_status"
    )
    _require_equal(
        completion.get("eligible_for_scientific_aggregation"),
        True,
        field="completion.eligible_for_scientific_aggregation",
    )
    _require_false(
        completion.get("scientific_result"), field="completion.scientific_result"
    )
    _require_equal(
        completion.get("summary_file"), "summary.json", field="completion.summary_file"
    )
    summary_sha = _validate_direct_hash(
        completion, "summary_sha256", summary_path, field="completion"
    )
    run_manifest_sha = _validate_direct_hash(
        completion, "run_manifest_sha256", run_manifest_path, field="completion"
    )
    cohort_sha = _validate_direct_hash(
        completion, "cohort_manifest_sha256", cohort_path, field="completion"
    )

    _require_equal(
        summary.get("schema"), "DepPLM-corrected-v2.2", field="summary.schema"
    )
    protocol = summary.get("protocol")
    condition = summary.get("condition")
    seed = summary.get("seed")
    if protocol not in spec.protocols:
        raise RunValidationError(
            "noncanonical_run", f"unsupported protocol: {protocol!r}"
        )
    if condition not in spec.conditions:
        raise RunValidationError(
            "noncanonical_run", f"unsupported condition: {condition!r}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in spec.seeds:
        raise RunValidationError("noncanonical_run", f"unsupported seed: {seed!r}")
    key = RunKey(str(protocol), str(condition), int(seed))

    _require_equal(summary.get("pilot"), False, field="summary.pilot")
    _require_equal(
        summary.get("run_status"), "protocol_complete", field="summary.run_status"
    )
    _require_equal(
        summary.get("protocol_complete"), True, field="summary.protocol_complete"
    )
    _require_equal(
        summary.get("eligible_for_scientific_aggregation"),
        True,
        field="summary.eligible_for_scientific_aggregation",
    )
    _require_false(summary.get("scientific_result"), field="summary.scientific_result")
    _require_equal(
        summary.get("primary_metric"),
        "residual_r2_vs_train_gene_mean",
        field="summary.primary_metric",
    )
    _require_equal(
        summary.get("required_finite_metrics"),
        list(REQUIRED_METRICS),
        field="summary.required_finite_metrics",
    )
    _require_equal(
        summary.get("source_stability"),
        {"checked_after_training": True, "unchanged": True, "reasons": []},
        field="summary.source_stability",
    )
    canonical = summary.get("canonical_campaign")
    if not isinstance(canonical, dict):
        raise RunValidationError(
            "noncanonical_run", "summary.canonical_campaign is missing"
        )
    _require_equal(
        canonical.get("seeds"),
        list(spec.seeds),
        field="summary.canonical_campaign.seeds",
    )
    _require_equal(
        canonical.get("arguments"),
        dict(spec.canonical_arguments),
        field="summary.canonical_campaign.arguments",
    )
    _require_equal(
        canonical.get("tuning_grid"),
        [dict(config) for config in spec.tuning_grid],
        field="summary.canonical_campaign.tuning_grid",
    )

    _require_equal(
        run_manifest.get("schema"), "DepPLM-corrected-v2.2", field="run_manifest.schema"
    )
    _require_equal(run_manifest.get("pilot"), False, field="run_manifest.pilot")
    _require_equal(
        run_manifest.get("run_status"), "in_progress", field="run_manifest.run_status"
    )
    _require_equal(
        run_manifest.get("protocol_complete"),
        False,
        field="run_manifest.protocol_complete",
    )
    _require_equal(
        run_manifest.get("eligible_for_scientific_aggregation"),
        False,
        field="run_manifest.eligible_for_scientific_aggregation",
    )
    _require_equal(
        run_manifest.get("output_policy"),
        "unique_run_directory_atomic_json_no_overwrite",
        field="run_manifest.output_policy",
    )
    _require_false(
        run_manifest.get("scientific_result"), field="run_manifest.scientific_result"
    )
    runtime = run_manifest.get("runtime")
    if not isinstance(runtime, dict):
        raise RunValidationError(
            "nondeterministic_runtime", "run_manifest.runtime is missing"
        )
    deterministic_contract = {
        "deterministic_algorithms_enabled": True,
        "deterministic_debug_mode": 2,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }
    for name, expected in deterministic_contract.items():
        if runtime.get(name) != expected:
            raise RunValidationError(
                "nondeterministic_runtime",
                f"run_manifest.runtime.{name} expected {expected!r}, found {runtime.get(name)!r}",
            )
    runtime_fingerprint, runtime_fingerprint_sha256 = _runtime_fingerprint(run_manifest)
    arguments = run_manifest.get("arguments")
    if not isinstance(arguments, dict):
        raise RunValidationError(
            "noncanonical_run", "run_manifest.arguments is missing"
        )
    _require_equal(arguments.get("protocol"), protocol, field="arguments.protocol")
    _require_equal(arguments.get("condition"), condition, field="arguments.condition")
    _require_equal(arguments.get("seed"), seed, field="arguments.seed")
    _require_equal(arguments.get("pilot"), False, field="arguments.pilot")
    _require_equal(arguments.get("fold_id"), -1, field="arguments.fold_id")
    for name, expected in spec.canonical_arguments.items():
        actual = arguments.get(name)
        if isinstance(expected, float):
            if (
                isinstance(actual, bool)
                or not isinstance(actual, (int, float))
                or not math.isclose(
                    float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12
                )
            ):
                raise RunValidationError(
                    "noncanonical_run",
                    f"arguments.{name} expected {expected!r}, found {actual!r}",
                )
        else:
            _require_equal(actual, expected, field=f"arguments.{name}")

    try:
        expected_embedding_provenance = (
            validate_embedding_bundle(workspace_root)
            if _expected_embedding_provenance is None
            else dict(_expected_embedding_provenance)
        )
    except (OSError, ValueError) as exc:
        raise RunValidationError(
            "invalid_embedding_provenance",
            f"canonical ESM embedding bundle failed verification: {exc}",
        ) from exc
    run_embedding_provenance = run_manifest.get("embedding_provenance")
    if not isinstance(run_embedding_provenance, dict):
        raise RunValidationError(
            "invalid_embedding_provenance",
            "run_manifest.embedding_provenance is missing",
        )
    _require_equal(
        run_embedding_provenance,
        expected_embedding_provenance,
        field="run_manifest.embedding_provenance",
    )
    embedding_provenance_sha256 = run_embedding_provenance.get("provenance_sha256")
    if (
        not isinstance(embedding_provenance_sha256, str)
        or len(embedding_provenance_sha256) != 64
    ):
        raise RunValidationError(
            "invalid_embedding_provenance",
            "run_manifest.embedding_provenance.provenance_sha256 is invalid",
        )

    n_cells, cell_ids, lineages, source_cell_indices = _validate_cohort_manifest(
        cohort, workspace_root=workspace_root
    )
    gene_ids = _read_manifest_lines(
        workspace_root / "data" / "processed" / "genes.txt", field="genes.txt"
    )
    source_gene_indices = tuple(int(index) for index in cohort["source_gene_indices"])
    gene_effect = (
        _load_gene_effect(
            workspace_root,
            n_genes=int(cohort["n_genes"]),
            n_cells=n_cells,
        )
        if _gene_effect is None
        else _gene_effect
    )
    if gene_effect.shape != (int(cohort["n_genes"]), n_cells):
        raise RunValidationError(
            "protocol_evidence",
            "shared gene-effect evidence does not align with the validated cohort",
        )
    reference_hash_cache = (
        {} if _reference_hash_cache is None else _reference_hash_cache
    )
    transfer_contract = (
        None
        if protocol == "p1"
        else _load_transfer_contract(workspace_root, n_cells=n_cells, lineages=lineages)
    )

    expected_names = _expected_fold_names(str(protocol))
    expected_numbers = list(range(len(expected_names)))
    expected_result_files = [f"{name}-result.json" for name in expected_names]
    expected_artifact_files = {
        *(f"{name}-result.json" for name in expected_names),
        *(f"{name}-split.json" for name in expected_names),
        *(f"{name}-predictions.npz" for name in expected_names),
    }
    _require_equal(
        summary.get("n_completed_folds"),
        len(expected_names),
        field="summary.n_completed_folds",
    )
    _require_equal(
        summary.get("fold_files"), expected_result_files, field="summary.fold_files"
    )
    _require_equal(
        summary.get("planned_fold_numbers"),
        expected_numbers,
        field="summary.planned_fold_numbers",
    )
    _require_equal(
        summary.get("completed_fold_numbers"),
        expected_numbers,
        field="summary.completed_fold_numbers",
    )

    completion_artifacts = _validate_record_list(
        completion.get("artifacts"), base=run_dir, field="completion.artifacts"
    )
    summary_artifacts = _validate_record_list(
        summary.get("artifacts"), base=run_dir, field="summary.artifacts"
    )
    _require_equal(
        completion_artifacts,
        summary_artifacts,
        field="completion/summary artifact records",
    )
    _require_equal(
        set(completion_artifacts),
        expected_artifact_files,
        field="fold artifact file set",
    )

    metrics_by_fold: dict[str, dict[str, float]] = {}
    split_evidence: dict[str, SplitEvidence] = {}
    for fold_number, fold_name in enumerate(expected_names):
        result_path = run_dir / f"{fold_name}-result.json"
        split_path = run_dir / f"{fold_name}-split.json"
        result = _load_json(result_path)
        split = _load_json(split_path)
        for field, expected in (
            ("protocol", protocol),
            ("condition", condition),
            ("seed", seed),
            ("fold", fold_name),
            ("fold_number", fold_number),
            ("pilot", False),
            ("scientific_result", False),
            ("result_scope", "single_outer_fold"),
            ("test_evaluation_calls", 1),
        ):
            _require_equal(
                result.get(field), expected, field=f"{fold_name}-result.{field}"
            )
        split_evidence[fold_name] = _validate_split_manifest(
            split,
            split_path=split_path,
            protocol=str(protocol),
            fold_name=fold_name,
            run_seed=int(seed),
            result_fold_seed=result.get("fold_seed"),
            n_cells=n_cells,
            cell_ids=cell_ids,
            lineages=lineages,
            source_cell_indices=source_cell_indices,
            transfer_contract=transfer_contract,
            n_outer_folds=int(spec.canonical_arguments["folds"]),
            inner_val_fraction=float(spec.canonical_arguments["inner_val_fraction"]),
        )
        _validate_fold_protocol_evidence(
            result,
            fold_name=fold_name,
            condition=str(condition),
            split=split_evidence[fold_name],
            spec=spec,
            lineages=lineages,
            expression_input_features=int(cohort["n_expression_features_raw"]),
            n_string_edges=int(cohort["n_string_edges"]),
            gene_effect=gene_effect,
            reference_hash_cache=reference_hash_cache,
        )
        metrics_by_fold[fold_name] = _validate_outer_test_predictions(
            result,
            run_dir=run_dir,
            fold_name=fold_name,
            split=split_evidence[fold_name],
            gene_effect=gene_effect,
            cell_ids=cell_ids,
            gene_ids=gene_ids,
            source_cell_indices=source_cell_indices,
            source_gene_indices=source_gene_indices,
            artifact_records=completion_artifacts,
        )

    if protocol == "p1":
        outer_test_counts = Counter(
            index
            for evidence in split_evidence.values()
            for index in evidence.outer_test
        )
        expected_counts = {index: 1 for index in range(n_cells)}
        if dict(outer_test_counts) != expected_counts:
            raise RunValidationError(
                "split_semantics",
                "the five P1 outer-test folds must cover every cohort row exactly once",
            )

    summary_metrics = summary.get("metrics")
    if not isinstance(summary_metrics, dict):
        raise RunValidationError("invalid_metric", "summary.metrics is missing")
    for metric in REPORT_METRICS:
        values = [metrics_by_fold[fold][metric] for fold in expected_names]
        record = summary_metrics.get(metric)
        if not isinstance(record, dict):
            raise RunValidationError(
                "invalid_metric", f"summary.metrics.{metric} is missing"
            )
        _require_equal(
            record.get("n"), len(values), field=f"summary.metrics.{metric}.n"
        )
        _compare_float(
            record.get("mean"),
            statistics.fmean(values),
            field=f"summary.metrics.{metric}.mean",
        )
        _compare_float(
            record.get("sd"),
            statistics.pstdev(values),
            field=f"summary.metrics.{metric}.sd",
        )

    input_records = _validate_record_list(
        run_manifest.get("inputs"), base=workspace_root, field="run_manifest.inputs"
    )
    code_records = _validate_record_list(
        run_manifest.get("code"), base=workspace_root, field="run_manifest.code"
    )
    _validate_provenance_inventory(
        input_records, protocol=str(protocol), field="inputs"
    )
    _validate_provenance_inventory(code_records, protocol=str(protocol), field="code")
    resolved_workspace = workspace_root.resolve(strict=True)
    resolved_run_dir = run_dir.resolve(strict=True)
    try:
        portable_run_path = resolved_run_dir.relative_to(resolved_workspace).as_posix()
    except ValueError as exc:
        raise RunValidationError(
            "unsafe_run_path",
            f"validated run directory is outside the workspace: {resolved_run_dir}",
        ) from exc

    source_artifacts = [
        {
            "path": portable_run_path,
            "completion_sha256": _sha256_file(completion_path),
            "summary_sha256": summary_sha,
            "run_manifest_sha256": run_manifest_sha,
            "cohort_manifest_sha256": cohort_sha,
            "runtime_fingerprint_sha256": runtime_fingerprint_sha256,
            "fold_artifacts": [
                {"path": path, "sha256": value[0], "bytes": value[1]}
                for path, value in sorted(completion_artifacts.items())
            ],
        }
    ]
    return ValidatedRun(
        key=key,
        path=resolved_run_dir,
        cohort_sha256=cohort_sha,
        input_hashes={path: value[0] for path, value in input_records.items()},
        code_hashes={path: value[0] for path, value in code_records.items()},
        runtime_fingerprint=runtime_fingerprint,
        runtime_fingerprint_sha256=runtime_fingerprint_sha256,
        fold_metrics=metrics_by_fold,
        splits=split_evidence,
        n_cells=n_cells,
        source_artifacts=source_artifacts,
        embedding_provenance_sha256=embedding_provenance_sha256,
    )


def _issue(code: str, message: str, *, path: Path | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"code": code, "message": message}
    if path is not None:
        payload["path"] = str(path.resolve())
    return payload


def _discover_run_directories(
    results_root: Path,
) -> tuple[list[Path], list[dict[str, Any]]]:
    candidates: list[Path] = []
    excluded: list[dict[str, Any]] = []
    if not results_root.is_dir():
        return candidates, [
            _issue(
                "missing_results_root", f"results root does not exist: {results_root}"
            )
        ]
    for path in sorted(results_root.iterdir(), key=lambda item: item.name):
        if not path.is_dir():
            continue
        run_like = any(
            (path / name).exists()
            for name in ("run-manifest.json", "summary.json", "completion.json")
        )
        if not run_like:
            continue
        if not (path / "completion.json").is_file():
            excluded.append(
                _issue(
                    "no_completion_marker",
                    "run is partial, pilot, failed, or predates the fail-closed completion gate",
                    path=path,
                )
            )
            continue
        candidates.append(path)
    return candidates, excluded


def _mean_sd_range(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty value sequence")
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _aggregate_complete_campaign(
    runs: Mapping[RunKey, ValidatedRun], spec: CampaignSpec
) -> tuple[dict[str, Any], dict[str, Any]]:
    aggregate: dict[str, Any] = {}
    seed_means: dict[tuple[str, str, str, int], float] = {}
    for protocol in spec.protocols:
        protocol_payload: dict[str, Any] = {}
        for condition in spec.conditions:
            condition_payload: dict[str, Any] = {}
            for metric in REPORT_METRICS:
                by_seed: dict[str, Any] = {}
                values: list[float] = []
                for seed in spec.seeds:
                    run = runs[RunKey(protocol, condition, seed)]
                    fold_values = [
                        run.fold_metrics[fold][metric]
                        for fold in sorted(run.fold_metrics)
                    ]
                    seed_mean = statistics.fmean(fold_values)
                    seed_means[(protocol, condition, metric, seed)] = seed_mean
                    values.append(seed_mean)
                    by_seed[str(seed)] = {
                        "n_outer_folds": len(fold_values),
                        "outer_fold_values": fold_values,
                        "mean_across_outer_folds": seed_mean,
                    }
                condition_payload[metric] = {
                    "unit_of_replication": "campaign_seed",
                    "replication_interpretation": (
                        "Each campaign seed defines a repeated nested-split-and-training "
                        "rerun; it is not an independently sampled biological unit."
                    ),
                    "fold_handling": (
                        "outer folds are averaged within seed and are not treated as "
                        "independent replicates"
                    ),
                    "by_seed": by_seed,
                    "across_seed_summary": _mean_sd_range(values),
                }
            protocol_payload[condition] = condition_payload
        aggregate[protocol] = protocol_payload

    comparisons: dict[str, Any] = {}
    primary = "residual_r2_vs_train_gene_mean"
    for protocol in spec.protocols:
        protocol_comparisons: dict[str, Any] = {}
        for condition in spec.conditions:
            if condition == REFERENCE_CONDITION:
                continue
            differences = [
                seed_means[(protocol, condition, primary, seed)]
                - seed_means[(protocol, REFERENCE_CONDITION, primary, seed)]
                for seed in spec.seeds
            ]
            protocol_comparisons[condition] = {
                "contrast": f"{condition} minus {REFERENCE_CONDITION}",
                "metric": primary,
                "direction": "positive favors the contrasted condition",
                "paired_seed_differences": {
                    str(seed): difference
                    for seed, difference in zip(spec.seeds, differences)
                },
                "descriptive_summary": _mean_sd_range(differences),
                "sign_pattern": {
                    "positive": sum(value > 0 for value in differences),
                    "negative": sum(value < 0 for value in differences),
                    "zero": sum(value == 0 for value in differences),
                },
                "inferential_p_value": None,
                "inference_reason": (
                    "No inferential p-value is reported. The three campaign seeds are "
                    "repeated nested-split-and-training reruns, not independently sampled "
                    "biological units."
                ),
            }
        comparisons[protocol] = protocol_comparisons
    return aggregate, comparisons


def _manifest_conflicts(
    runs: Iterable[ValidatedRun], attribute: str
) -> list[dict[str, Any]]:
    observed: dict[str, dict[str, list[str]]] = {}
    for run in runs:
        records: Mapping[str, str] = getattr(run, attribute)
        for path, sha256 in records.items():
            observed.setdefault(path, {}).setdefault(sha256, []).append(run.key.label())
    return [
        {
            "path": path,
            "hashes": [
                {"sha256": sha256, "runs": sorted(labels)}
                for sha256, labels in sorted(by_hash.items())
            ],
        }
        for path, by_hash in sorted(observed.items())
        if len(by_hash) > 1
    ]


def _runtime_fingerprint_groups(
    runs: Iterable[ValidatedRun],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for run in runs:
        by_hash = grouped.setdefault((run.key.protocol, run.key.seed), {})
        record = by_hash.setdefault(
            run.runtime_fingerprint_sha256,
            {
                "sha256": run.runtime_fingerprint_sha256,
                "fingerprint": dict(run.runtime_fingerprint),
                "runs": [],
            },
        )
        record["runs"].append(run.key.label())
    return {
        f"{protocol}|seed-{seed}": [
            {
                **record,
                "runs": sorted(record["runs"]),
            }
            for _, record in sorted(by_hash.items())
        ]
        for (protocol, seed), by_hash in sorted(grouped.items())
    }


def _cross_run_runtime_issues(
    runs: Mapping[RunKey, ValidatedRun], spec: CampaignSpec
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for protocol in spec.protocols:
        for seed in spec.seeds:
            matching = [
                run
                for key, run in runs.items()
                if key.protocol == protocol and key.seed == seed
            ]
            by_fingerprint: dict[str, list[str]] = {}
            for run in matching:
                by_fingerprint.setdefault(run.runtime_fingerprint_sha256, []).append(
                    run.key.label()
                )
            if len(by_fingerprint) > 1:
                details = "; ".join(
                    f"{sha256}: {', '.join(sorted(labels))}"
                    for sha256, labels in sorted(by_fingerprint.items())
                )
                issues.append(
                    _issue(
                        "runtime_fingerprint_inconsistency",
                        f"{protocol}|seed-{seed} conditions do not share one runtime "
                        f"fingerprint: {details}",
                    )
                )
    return issues


def _cross_run_split_issues(
    runs: Mapping[RunKey, ValidatedRun], spec: CampaignSpec
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for protocol in spec.protocols:
        if protocol not in PROTOCOLS:
            continue
        for seed in spec.seeds:
            for fold_name in _expected_fold_names(protocol):
                observed = {
                    run.splits[fold_name].sha256: run.key.label()
                    for key, run in runs.items()
                    if key.protocol == protocol
                    and key.seed == seed
                    and fold_name in run.splits
                }
                if len(observed) > 1:
                    issues.append(
                        _issue(
                            "split_manifest_inconsistency",
                            f"{protocol}|seed-{seed}|{fold_name} differs across conditions: "
                            + ", ".join(sorted(observed.values())),
                        )
                    )

    for condition in spec.conditions:
        for seed in spec.seeds:
            forward = runs.get(RunKey("transfer-forward", condition, seed))
            reverse = runs.get(RunKey("transfer-reverse", condition, seed))
            if forward is None or reverse is None:
                continue
            forward_split = forward.splits["complement-to-bkm"]
            reverse_split = reverse.splits["bkm-to-complement"]
            if (
                forward_split.outer_train != reverse_split.outer_test
                or forward_split.outer_test != reverse_split.outer_train
            ):
                issues.append(
                    _issue(
                        "transfer_direction_inconsistency",
                        f"transfer outer partitions are not exact reversals for "
                        f"{condition}|seed-{seed}",
                    )
                )
    return issues


def validate_campaign(
    results_root: Path,
    *,
    workspace_root: Path = ROOT,
    spec: CampaignSpec | None = None,
) -> dict[str, Any]:
    """Return a fail-closed validation report without modifying source runs."""
    aggregator_source_identity = _capture_aggregator_source_identity()
    active_spec = canonical_spec() if spec is None else spec
    candidates, excluded = _discover_run_directories(results_root)
    issues: list[dict[str, Any]] = []
    if active_spec != canonical_spec():
        issues.append(
            _issue(
                "noncanonical_campaign_spec",
                "caller-supplied campaign specification differs from the fixed 99-run contract",
            )
        )
    expected_embedding_provenance: dict[str, Any] | None = None
    try:
        expected_embedding_provenance = validate_embedding_bundle(workspace_root)
    except (OSError, ValueError) as exc:
        issues.append(
            _issue(
                "invalid_embedding_provenance",
                f"canonical ESM embedding bundle failed verification: {exc}",
            )
        )
    valid_runs: list[ValidatedRun] = []
    shared_gene_effect: np.ndarray | None = None
    reference_hash_cache: dict[tuple[int, ...], str] = {}
    for run_dir in candidates:
        try:
            if shared_gene_effect is None:
                shared_gene_effect = _load_gene_effect(
                    workspace_root,
                    n_genes=CANONICAL_N_GENES,
                    n_cells=CANONICAL_N_CELLS,
                )
            valid_runs.append(
                validate_run_directory(
                    run_dir,
                    workspace_root=workspace_root,
                    spec=active_spec,
                    _gene_effect=shared_gene_effect,
                    _reference_hash_cache=reference_hash_cache,
                    _expected_embedding_provenance=expected_embedding_provenance,
                )
            )
        except RunValidationError as exc:
            issues.append(_issue(exc.code, str(exc), path=run_dir))

    by_key: dict[RunKey, list[ValidatedRun]] = {}
    for run in valid_runs:
        by_key.setdefault(run.key, []).append(run)
    duplicate_keys = {key: runs for key, runs in by_key.items() if len(runs) > 1}
    for key, duplicates in sorted(duplicate_keys.items()):
        issues.append(
            _issue(
                "duplicate_matrix_cell",
                f"{key.label()} has {len(duplicates)} completion-bearing runs: "
                + ", ".join(str(run.path) for run in duplicates),
            )
        )

    selected = {key: runs[0] for key, runs in by_key.items() if len(runs) == 1}
    expected_keys = set(active_spec.expected_keys())
    missing = sorted(expected_keys - set(by_key))
    unexpected = sorted(set(by_key) - expected_keys)
    for key in unexpected:
        issues.append(
            _issue("unexpected_matrix_cell", f"unexpected canonical key: {key.label()}")
        )

    cohort_hashes = sorted({run.cohort_sha256 for run in valid_runs})
    if len(cohort_hashes) > 1:
        issues.append(
            _issue(
                "cohort_hash_inconsistency",
                f"validated runs contain {len(cohort_hashes)} distinct cohort manifests",
            )
        )
    input_conflicts = _manifest_conflicts(valid_runs, "input_hashes")
    code_conflicts = _manifest_conflicts(valid_runs, "code_hashes")
    if input_conflicts:
        issues.append(
            _issue(
                "input_hash_inconsistency",
                f"{len(input_conflicts)} input paths have conflicting hashes across runs",
            )
        )
    if code_conflicts:
        issues.append(
            _issue(
                "code_hash_inconsistency",
                f"{len(code_conflicts)} code paths have conflicting hashes across runs",
            )
        )

    protocol_input_sets: dict[str, list[tuple[tuple[str, str], ...]]] = {}
    protocol_code_sets: dict[str, list[tuple[tuple[str, str], ...]]] = {}
    for run in valid_runs:
        protocol_input_sets.setdefault(run.key.protocol, []).append(
            tuple(sorted(run.input_hashes.items()))
        )
        protocol_code_sets.setdefault(run.key.protocol, []).append(
            tuple(sorted(run.code_hashes.items()))
        )
    for protocol, manifests in sorted(protocol_input_sets.items()):
        if len(set(manifests)) > 1:
            issues.append(
                _issue(
                    "input_manifest_inconsistency",
                    f"{protocol} runs do not share an identical input manifest",
                )
            )
    for protocol, manifests in sorted(protocol_code_sets.items()):
        if len(set(manifests)) > 1:
            issues.append(
                _issue(
                    "code_manifest_inconsistency",
                    f"{protocol} runs do not share an identical code manifest",
                )
            )
    all_code_manifests = {
        manifest for manifests in protocol_code_sets.values() for manifest in manifests
    }
    if len(all_code_manifests) > 1:
        issues.append(
            _issue(
                "code_manifest_inconsistency",
                "runs do not share one identical campaign-wide code manifest",
            )
        )
    forward_inputs = set(protocol_input_sets.get("transfer-forward", []))
    reverse_inputs = set(protocol_input_sets.get("transfer-reverse", []))
    if forward_inputs and reverse_inputs and forward_inputs != reverse_inputs:
        issues.append(
            _issue(
                "input_manifest_inconsistency",
                "transfer-forward and transfer-reverse do not share one input manifest",
            )
        )

    issues.extend(_cross_run_runtime_issues(selected, active_spec))
    issues.extend(_cross_run_split_issues(selected, active_spec))
    runtime_fingerprints = _runtime_fingerprint_groups(valid_runs)

    protocol_validation_passed = (
        not issues
        and not missing
        and not duplicate_keys
        and set(selected) == expected_keys
    )
    aggregates: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    if protocol_validation_passed:
        aggregates, comparisons = _aggregate_complete_campaign(selected, active_spec)

    source_runs = [
        {
            "key": run.key.label(),
            **run.source_artifacts[0],
        }
        for run in sorted(valid_runs, key=lambda item: item.key)
    ]
    aggregation_packages: dict[str, str | None] = {}
    for package in AGGREGATION_PACKAGES:
        try:
            aggregation_packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            aggregation_packages[package] = None
    aggregation_provenance = {
        "source": {
            "path": "code/aggregate_corrected_campaign.py",
            **aggregator_source_identity,
        },
        "runtime": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "packages": aggregation_packages,
        },
        "embedding_bundle": expected_embedding_provenance,
    }
    scientific_readiness_gates = {
        "verified_embedding_generation_provenance": (
            expected_embedding_provenance is not None
            and all(
                run.embedding_provenance_sha256
                == expected_embedding_provenance.get("provenance_sha256")
                for run in valid_runs
            )
        ),
        "outer_test_metrics_recomputed_from_bound_predictions": (
            protocol_validation_passed
        ),
    }
    scientific_result = protocol_validation_passed and all(
        scientific_readiness_gates.values()
    )
    if scientific_result:
        scientific_result_reason = (
            "The exact prespecified matrix and both scientific-readiness gates "
            "passed validation."
        )
    elif protocol_validation_passed:
        scientific_result_reason = (
            "The exact prespecified matrix passed protocol validation, but at least "
            "one scientific-readiness evidence gate did not pass. No model conclusion "
            "may be drawn from this report."
        )
    else:
        scientific_result_reason = (
            "Campaign validation is incomplete or invalid. No model conclusion may "
            "be drawn from this report."
        )
    report = {
        "schema": "DepPLM-corrected-v2.2-campaign",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "aggregation_provenance": aggregation_provenance,
        "protocol_validation_passed": protocol_validation_passed,
        "scientific_readiness_gates": scientific_readiness_gates,
        "scientific_result": scientific_result,
        "scientific_result_reason": scientific_result_reason,
        "prespecified_matrix": {
            "protocols": list(active_spec.protocols),
            "conditions": list(active_spec.conditions),
            "seeds": list(active_spec.seeds),
            "expected_runs": len(expected_keys),
            "p1_outer_folds_per_run": 5,
            "transfer_outer_folds_per_run": 1,
            "canonical_arguments": dict(active_spec.canonical_arguments),
            "tuning_grid": [dict(config) for config in active_spec.tuning_grid],
        },
        "validation": {
            "completion_bearing_directories": len(candidates),
            "validated_runs": len(valid_runs),
            "unique_matrix_cells": len(by_key),
            "missing_matrix_cells": [key.label() for key in missing],
            "duplicate_matrix_cells": {
                key.label(): [str(run.path) for run in runs]
                for key, runs in sorted(duplicate_keys.items())
            },
            "issues": issues,
            "excluded_noncomplete_runs": excluded,
            "cohort_hashes": cohort_hashes,
            "input_hash_conflicts": input_conflicts,
            "code_hash_conflicts": code_conflicts,
            "runtime_fingerprints": runtime_fingerprints,
        },
        "statistical_policy": {
            "unit_of_replication": "campaign_seed",
            "replication_interpretation": (
                "Each campaign seed defines a repeated nested-split-and-training rerun; "
                "it is not an independently sampled biological unit."
            ),
            "outer_folds": "averaged within seed, never counted as independent replicates",
            "uncertainty": (
                "descriptive mean, sample standard deviation, minimum, and maximum across "
                "campaign-seed reruns"
            ),
            "inferential_tests": (
                "not reported; n=3 campaign-seed reruns do not establish independent "
                "biological replication"
            ),
        },
        "source_runs": source_runs,
        "aggregates": aggregates,
        "paired_descriptive_comparisons": comparisons,
    }
    _assert_aggregator_source_unchanged(aggregator_source_identity)
    return report


def _report_without_creation_time(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return the deterministic portion used for pre-publication replay."""
    normalized = json.loads(
        json.dumps(report, sort_keys=True, ensure_ascii=True, allow_nan=False)
    )
    normalized.pop("created_utc", None)
    return normalized


def _revalidate_report_for_publication(
    report: Mapping[str, Any],
    *,
    results_root: Path,
    workspace_root: Path,
) -> None:
    """Replay the complete campaign validator immediately before publication."""
    replayed = validate_campaign(
        results_root.resolve(), workspace_root=workspace_root.resolve()
    )
    if _report_without_creation_time(replayed) != _report_without_creation_time(report):
        raise RunValidationError(
            "campaign_changed_before_publication",
            "fresh validation of the source runs differs from the staged campaign report",
        )


def write_campaign_report(
    report: Mapping[str, Any],
    output_root: Path,
    *,
    results_root: Path,
    workspace_root: Path = ROOT,
) -> Path:
    """Atomically publish a freshly replayed report in a new directory."""
    expected_source = _report_source_identity(report)
    _assert_aggregator_source_unchanged(expected_source)
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    identifier = f"{timestamp}.{time.time_ns() % 1_000_000_000:09d}Z-campaign-{uuid.uuid4().hex[:8]}"
    destination = output_root / identifier
    staging = output_root / f".{identifier}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(exist_ok=False)
    staged_report = staging / "campaign.json"
    try:
        atomic_write_json(staged_report, dict(report))
        if destination.exists():
            raise FileExistsError(
                f"refusing to overwrite campaign directory: {destination}"
            )
        _assert_aggregator_source_unchanged(expected_source)
        staged_payload = _load_json(staged_report)
        _revalidate_report_for_publication(
            staged_payload,
            results_root=results_root,
            workspace_root=workspace_root,
        )
        _assert_aggregator_source_unchanged(expected_source)
        os.replace(staging, destination)
    except Exception:
        if staged_report.exists():
            staged_report.unlink()
        if staging.exists() and not any(staging.iterdir()):
            staging.rmdir()
        raise
    return destination / "campaign.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="default: <results-root>/campaigns",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print JSON without writing a campaign directory",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = validate_campaign(args.results_root, workspace_root=args.workspace_root)
    _assert_aggregator_source_unchanged(_report_source_identity(report))
    if args.dry_run:
        json.dump(report, sys.stdout, indent=2, sort_keys=True, ensure_ascii=True)
        sys.stdout.write("\n")
    else:
        output_root = args.output_root or (args.results_root / "campaigns")
        report_path = write_campaign_report(
            report,
            output_root,
            results_root=args.results_root,
            workspace_root=args.workspace_root,
        )
        print(report_path)
    return 0 if report["scientific_result"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
