"""Independent core-evidence audit for a completed K/token/PMI campaign.

This verifier intentionally does not import the training runner or its metric
implementation.  It checks the detached primary-verification receipt, hashes
the saved predictions and per-gene sidecars, verifies exact-once outer-test
coverage, and independently recomputes the four metrics used by the paired
descriptive analysis.  It does not replay training or preprocessing and does
not establish source-data truth, biological validity, or clinical validity.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import io
import json
import math
import mmap
import os
import platform
import stat as stat_module
import struct
import sys
import time
import uuid
import zipfile
import zlib
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence


RECEIPT_SCHEMA = "DepPLM-k-token-pmi-detached-verification-receipt-v1"
PRIMARY_REPORT_SCHEMA = "DepPLM-k-token-pmi-campaign-verification-v1"
AUDIT_SCHEMA = "DepPLM-k-token-pmi-independent-core-audit-v1"
PRIMARY_VERIFIER_MODE = "read_only_replay_using_training_runner_module"
VARIANTS = (
    "k16-frequency",
    "k32-frequency",
    "k64-frequency",
    "k64-vep-impact",
    "k64-driver-lof",
    "k64-frequency-unit",
    "k64-frequency-valid-only",
)
SEEDS = (42, 43, 44)
FOLDS = tuple(f"fold-{number:02d}" for number in range(5))
PAIRED_METRICS = (
    "residual_r2_vs_train_gene_mean",
    "residual_pearson_pooled",
    "mse",
    "mae",
)
PREDICTION_MEMBERS = {
    "prediction",
    "train_gene_mean",
    "outer_test_indices",
    "source_cell_indices",
    "source_gene_indices",
    "cell_ids",
    "gene_ids",
}
PER_GENE_COLUMNS = (
    "gene_index",
    "gene",
    "n_outer_test_observations",
    "mse",
    "mae",
    "train_gene_mean_baseline_mse",
    "mse_improvement_over_train_gene_mean",
)
MAX_NPY_HEADER_BYTES = 16 * 1024
MAX_NPZ_CONTAINER_OVERHEAD_BYTES = 1024 * 1024
MAX_SIDECAR_ROW_BYTES = 1024
MAX_SIDECAR_CONTAINER_OVERHEAD_BYTES = 64 * 1024
MAX_RECEIPT_BYTES = 16 * 1024**2
MAX_REPORT_BYTES = 512 * 1024**2
MAX_METADATA_BYTES = 64 * 1024**2
MAX_SOURCE_BYTES = 16 * 1024**2
MAX_IDENTITY_BYTES = 64 * 1024**2
MAX_GENE_EFFECT_BYTES = 4 * 1024**3
MAX_AUDIT_OUTPUT_BYTES = 512 * 1024**2
NPY_DECODE_CHUNK_BYTES = 1024 * 1024
PRIMARY_GATES = {
    "campaign_metadata",
    "source_and_input_hashes",
    "exact_105_identity_coverage",
    "training_run_bindings",
    "per_fold_preprocessing_manifests",
    "artifact_hashes_and_array_contracts",
    "per_gene_sidecars_recomputed",
    "outer_test_exact_once_and_index_integrity",
    "metrics_recomputed_from_predictions",
    "paired_descriptive_aggregation",
    "end_of_verification_stability",
}
RECEIPT_FIELDS = {
    "schema",
    "created_utc",
    "verification_report",
    "campaign_directory",
    "artifact_root",
    "campaign_plan",
    "campaign_completion",
    "protocol_complete",
    "scientific_result",
    "primary_verifier_source",
    "primary_verifier_mode",
    "limitations",
}
SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parent.parent


class AuditError(RuntimeError):
    """Raised when an independent core-evidence gate fails."""


class _MappedFloat32Values:
    """Read-only float32 sequence backed by one file mapping, not a RAM copy."""

    def __init__(self, mapping: mmap.mmap, *, offset: int, length: int) -> None:
        self._mapping = mapping
        self._offset = offset
        self._length = length
        self._closed = False

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> float:
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("mapped NPY indices must be integers")
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        if self._closed:
            raise AuditError("mapped NPY payload is closed")
        return float(
            struct.unpack_from("<f", self._mapping, self._offset + index * 4)[0]
        )

    def close(self) -> None:
        if not self._closed:
            self._mapping.close()
            self._closed = True


@dataclass(frozen=True)
class NpyArray:
    """Protocol-limited NPY value parsed without third-party dependencies."""

    descr: str
    shape: tuple[int, ...]
    values: array | tuple[str, ...] | _MappedFloat32Values


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuditError(f"JSON object contains duplicate key: {key}")
        result[key] = value
    return result


def _parse_json(payload: bytes | bytearray, *, label: str) -> Any:
    try:
        return json.loads(payload.decode("utf-8"), object_pairs_hook=_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"{label} is not valid duplicate-free UTF-8 JSON") from exc


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_link_or_reparse(value: os.stat_result) -> bool:
    attributes = int(getattr(value, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat_module, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return stat_module.S_ISLNK(value.st_mode) or bool(attributes & reparse_flag)


def _identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
    )


def _ancestor_identity_snapshot(path: Path) -> tuple[tuple[Any, ...], ...]:
    absolute = Path(os.path.abspath(path))
    chain = list(reversed((absolute.parent, *absolute.parent.parents)))
    snapshot: list[tuple[Any, ...]] = []
    for ancestor in chain:
        try:
            observed = ancestor.lstat()
        except OSError as exc:
            raise AuditError(
                f"artifact ancestor cannot be stat'ed: {ancestor}"
            ) from exc
        if _is_link_or_reparse(observed):
            raise AuditError(
                f"symbolic link or reparse ancestor is forbidden: {ancestor}"
            )
        if not stat_module.S_ISDIR(observed.st_mode):
            raise AuditError(f"artifact ancestor is not a directory: {ancestor}")
        snapshot.append(
            (
                str(ancestor),
                int(observed.st_dev),
                int(observed.st_ino),
                int(stat_module.S_IFMT(observed.st_mode)),
                int(getattr(observed, "st_file_attributes", 0)),
            )
        )
    return tuple(snapshot)


def _resolve_verified_directory_root(path: Path, *, label: str) -> Path:
    lexical = Path(os.path.abspath(path))
    before = _ancestor_identity_snapshot(lexical / ".root-identity-probe")
    try:
        root_before = lexical.lstat()
    except OSError as exc:
        raise AuditError(f"{label} cannot be stat'ed: {lexical}") from exc
    if _is_link_or_reparse(root_before):
        raise AuditError(f"{label} may not be a symbolic link or reparse point")
    if not stat_module.S_ISDIR(root_before.st_mode):
        raise AuditError(f"{label} is not a directory: {lexical}")
    try:
        canonical = lexical.resolve(strict=True)
    except OSError as exc:
        raise AuditError(f"{label} cannot be resolved: {lexical}") from exc
    after = _ancestor_identity_snapshot(lexical / ".root-identity-probe")
    try:
        root_after = lexical.lstat()
    except OSError as exc:
        raise AuditError(f"{label} disappeared during validation") from exc
    if (
        before != after
        or _is_link_or_reparse(root_after)
        or _identity(root_before) != _identity(root_after)
    ):
        raise AuditError(f"{label} changed during validation")
    return canonical


def _open_readonly_nofollow(path: Path) -> Any:
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    descriptor = os.open(path, flags)
    return os.fdopen(descriptor, "rb", closefd=True)


def _read_stable_bytes(
    path: Path,
    *,
    max_bytes: int,
) -> tuple[bytearray, dict[str, Any]]:
    # Keep the lexical leaf for lstat; resolve() here would erase evidence that
    # the caller supplied a symlink or Windows reparse point.
    path = Path(os.path.abspath(path))
    ancestors_before = _ancestor_identity_snapshot(path)
    try:
        path_before = path.lstat()
    except OSError as exc:
        raise AuditError(f"artifact cannot be stat'ed: {path}") from exc
    if _is_link_or_reparse(path_before):
        raise AuditError(f"symbolic link or reparse artifact is forbidden: {path}")
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1:
        raise AuditError("artifact byte limit must be a positive integer")
    if int(path_before.st_size) > max_bytes:
        raise AuditError(f"artifact exceeds its pre-materialization byte limit: {path}")
    digest = hashlib.sha256()
    payload = bytearray()
    byte_count = 0
    try:
        with _open_readonly_nofollow(path) as handle:
            before = os.fstat(handle.fileno())
            if not stat_module.S_ISREG(before.st_mode):
                raise AuditError(f"artifact is not a regular file: {path}")
            if _identity(path_before) != _identity(before):
                raise AuditError(
                    f"artifact path changed before its handle was opened: {path}"
                )
            if int(before.st_size) > max_bytes:
                raise AuditError(
                    f"artifact exceeds its pre-materialization byte limit: {path}"
                )
            while chunk := handle.read(min(4 * 1024 * 1024, max_bytes + 1 - byte_count)):
                payload.extend(chunk)
                digest.update(chunk)
                byte_count += len(chunk)
                if byte_count > max_bytes:
                    raise AuditError(
                        f"artifact exceeds its pre-materialization byte limit: {path}"
                    )
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise AuditError(f"artifact cannot be read: {path}") from exc
    if _identity(before) != _identity(after) or byte_count != int(after.st_size):
        raise AuditError(f"artifact changed while being read: {path}")
    try:
        path_after = path.lstat()
    except OSError as exc:
        raise AuditError(f"artifact disappeared while being read: {path}") from exc
    if _is_link_or_reparse(path_after) or _identity(path_after) != _identity(after):
        raise AuditError(f"artifact path changed while being read: {path}")
    if _ancestor_identity_snapshot(path) != ancestors_before:
        raise AuditError(f"artifact ancestor changed while being read: {path}")
    if not payload:
        raise AuditError(f"artifact is empty: {path}")
    return payload, {
        "path": str(path),
        "bytes": byte_count,
        "mtime_ns": int(after.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def _hash_stable_file(
    path: Path,
    *,
    max_bytes: int,
) -> dict[str, Any]:
    """Hash a stable file handle without retaining its contents."""
    path = Path(os.path.abspath(path))
    ancestors_before = _ancestor_identity_snapshot(path)
    try:
        path_before = path.lstat()
    except OSError as exc:
        raise AuditError(f"artifact cannot be stat'ed: {path}") from exc
    if _is_link_or_reparse(path_before):
        raise AuditError(f"symbolic link or reparse artifact is forbidden: {path}")
    if (
        not isinstance(max_bytes, int)
        or isinstance(max_bytes, bool)
        or max_bytes < 1
        or int(path_before.st_size) > max_bytes
    ):
        raise AuditError(f"artifact exceeds its streaming byte limit: {path}")
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with _open_readonly_nofollow(path) as handle:
            before = os.fstat(handle.fileno())
            if not stat_module.S_ISREG(before.st_mode):
                raise AuditError(f"artifact is not a regular file: {path}")
            if _identity(path_before) != _identity(before):
                raise AuditError(f"artifact changed before hashing: {path}")
            if int(before.st_size) > max_bytes:
                raise AuditError(f"artifact exceeds its streaming byte limit: {path}")
            while chunk := handle.read(4 * 1024 * 1024):
                byte_count += len(chunk)
                if byte_count > max_bytes:
                    raise AuditError(f"artifact exceeds its streaming byte limit: {path}")
                digest.update(chunk)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise AuditError(f"artifact cannot be hashed: {path}") from exc
    if _identity(before) != _identity(after) or byte_count != int(after.st_size):
        raise AuditError(f"artifact changed while being hashed: {path}")
    try:
        path_after = path.lstat()
    except OSError as exc:
        raise AuditError(f"artifact disappeared while being hashed: {path}") from exc
    if _is_link_or_reparse(path_after) or _identity(path_after) != _identity(after):
        raise AuditError(f"artifact path changed while being hashed: {path}")
    if _ancestor_identity_snapshot(path) != ancestors_before:
        raise AuditError(f"artifact ancestor changed while being hashed: {path}")
    if byte_count < 1:
        raise AuditError(f"artifact is empty: {path}")
    return {
        "path": str(path),
        "bytes": byte_count,
        "mtime_ns": int(after.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def _require_file_binding(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
    require_path: bool = False,
) -> None:
    expected_size = expected.get("bytes")
    expected_digest = expected.get("sha256")
    if (
        not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size < 1
        or not _is_sha256(expected_digest)
    ):
        raise AuditError(f"{label} has a malformed byte/hash binding")
    if (
        observed.get("bytes") != expected_size
        or observed.get("sha256") != expected_digest
    ):
        raise AuditError(f"{label} byte/hash binding does not match")
    expected_mtime = expected.get("mtime_ns")
    if (
        not isinstance(expected_mtime, int)
        or isinstance(expected_mtime, bool)
        or observed.get("mtime_ns") != expected_mtime
    ):
        raise AuditError(f"{label} mtime binding does not match")
    if require_path:
        raw_path = expected.get("path")
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            raise AuditError(f"{label} must bind an absolute path")
        if Path(raw_path).resolve() != Path(str(observed["path"])).resolve():
            raise AuditError(f"{label} path binding does not match")


def _require_outside(path: Path, directory: Path, *, label: str) -> None:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return
    raise AuditError(f"{label} must be outside {directory}")


def _safe_relative_path(raw_path: Any, *, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise AuditError(f"{label} path is missing")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise AuditError(f"{label} path must be a contained relative path")
    return relative


def _resolve_bound_path(
    record: Mapping[str, Any],
    *,
    root: Path,
    label: str,
) -> Path:
    relative = _safe_relative_path(record.get("path"), label=label)
    root = Path(os.path.abspath(root))
    _ancestor_identity_snapshot(root / ".bound-root-probe")
    current = root
    try:
        if _is_link_or_reparse(current.lstat()):
            raise AuditError(f"{label} root cannot be a link or reparse point")
        for part in relative.parts:
            if part in ("", "."):
                continue
            current = current / part
            if _is_link_or_reparse(current.lstat()):
                raise AuditError(f"{label} traverses a link or reparse point")
    except OSError as exc:
        raise AuditError(f"{label} artifact is missing: {relative}") from exc
    resolved_root = root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise AuditError(f"{label} path escapes its trusted root") from exc
    return resolved


def _read_bound_bytes(
    record: Mapping[str, Any],
    *,
    root: Path,
    label: str,
    max_bytes: int,
) -> tuple[Path, bytearray, dict[str, Any]]:
    if not isinstance(record, Mapping):
        raise AuditError(f"{label} binding must be a JSON object")
    expected_size = record.get("bytes")
    if (
        not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size < 1
        or expected_size > max_bytes
    ):
        raise AuditError(f"{label} exceeds its protocol byte limit")
    path = _resolve_bound_path(record, root=root, label=label)
    payload, observed = _read_stable_bytes(path, max_bytes=expected_size)
    _require_file_binding(observed, record, label=label)
    observed["path"] = str(path.relative_to(root.resolve()))
    return path, payload, observed


def _open_bound_float32_npy(
    record: Mapping[str, Any],
    *,
    root: Path,
    label: str,
    max_bytes: int,
) -> tuple[Path, NpyArray, dict[str, Any], _MappedFloat32Values]:
    """Validate, hash, and map one bound float32 NPY without copying its payload."""
    if not isinstance(record, Mapping):
        raise AuditError(f"{label} binding must be a JSON object")
    expected_size = record.get("bytes")
    expected_digest = record.get("sha256")
    if (
        not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size < 1
        or expected_size > max_bytes
        or not _is_sha256(expected_digest)
    ):
        raise AuditError(f"{label} exceeds its protocol byte limit")

    path = _resolve_bound_path(record, root=root, label=label)
    lexical_path = Path(os.path.abspath(path))
    ancestors_before = _ancestor_identity_snapshot(lexical_path)
    try:
        path_before = lexical_path.lstat()
    except OSError as exc:
        raise AuditError(f"{label} cannot be stat'ed") from exc
    if _is_link_or_reparse(path_before):
        raise AuditError(f"{label} cannot be a link or reparse point")
    if int(path_before.st_size) != expected_size:
        raise AuditError(f"{label} byte binding differs before parsing")

    mapping: mmap.mmap | None = None
    try:
        with _open_readonly_nofollow(lexical_path) as handle:
            before = os.fstat(handle.fileno())
            if not stat_module.S_ISREG(before.st_mode):
                raise AuditError(f"{label} is not a regular file")
            if _identity(path_before) != _identity(before):
                raise AuditError(f"{label} changed before its handle was opened")
            if int(before.st_size) != expected_size:
                raise AuditError(f"{label} byte binding differs before parsing")

            shape, descr, itemsize = _read_npy_header(handle)
            data_offset = int(handle.tell())
            if descr != "<f4" or itemsize != 4 or len(shape) != 2:
                raise AuditError(f"{label} must be a two-dimensional little-endian float32 NPY")
            element_count = math.prod(shape)
            if data_offset + element_count * itemsize != expected_size:
                raise AuditError(f"{label} byte size differs from its NPY header")

            digest = hashlib.sha256()
            byte_count = 0
            handle.seek(0)
            while chunk := handle.read(4 * 1024 * 1024):
                byte_count += len(chunk)
                if byte_count > expected_size:
                    raise AuditError(f"{label} exceeds its bound byte count")
                digest.update(chunk)
            after = os.fstat(handle.fileno())
            if (
                _identity(before) != _identity(after)
                or byte_count != expected_size
            ):
                raise AuditError(f"{label} changed while being hashed")
            mapping = mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ)
    except OSError as exc:
        if mapping is not None:
            mapping.close()
        raise AuditError(f"{label} cannot be read or mapped") from exc

    try:
        path_after = lexical_path.lstat()
        if (
            _is_link_or_reparse(path_after)
            or _identity(path_after) != _identity(after)
            or _ancestor_identity_snapshot(lexical_path) != ancestors_before
        ):
            raise AuditError(f"{label} path changed while being mapped")
    except Exception:
        mapping.close()
        raise

    observed = {
        "path": str(lexical_path.relative_to(root.resolve())),
        "bytes": byte_count,
        "mtime_ns": int(after.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }
    try:
        _require_file_binding(observed, record, label=label)
    except Exception:
        mapping.close()
        raise
    values = _MappedFloat32Values(
        mapping,
        offset=data_offset,
        length=element_count,
    )
    return path, NpyArray(descr=descr, shape=shape, values=values), observed, values


def _strict_equal(left: Any, right: Any) -> bool:
    if isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and set(left) == set(right)
            and all(_strict_equal(left[key], right[key]) for key in right)
        )
    if isinstance(right, list):
        return (
            isinstance(left, list)
            and len(left) == len(right)
            and all(_strict_equal(a, b) for a, b in zip(left, right))
        )
    if right is None:
        return left is None
    if isinstance(right, bool):
        return left is right
    if isinstance(right, int):
        return isinstance(left, int) and not isinstance(left, bool) and left == right
    if isinstance(right, float):
        return isinstance(left, float) and left == right
    if isinstance(right, str):
        return isinstance(left, str) and left == right
    return type(left) is type(right) and left == right


def _expected_identities() -> set[tuple[str, int, str]]:
    return {
        (variant, seed, fold)
        for variant in VARIANTS
        for seed in SEEDS
        for fold in FOLDS
    }


def _find_input_record(records: Any, relative_path: str) -> Mapping[str, Any]:
    if not isinstance(records, list):
        raise AuditError("primary report verified_inputs is not a list")
    matches: list[Mapping[str, Any]] = []
    normalized_path = relative_path.replace("\\", "/")
    for record in records:
        if not isinstance(record, Mapping):
            raise AuditError("primary report contains a malformed input binding")
        raw_path = record.get("path")
        if isinstance(raw_path, str) and raw_path.replace("\\", "/") == normalized_path:
            matches.append(record)
    if len(matches) != 1:
        raise AuditError(
            f"primary report must bind exactly one {normalized_path} input"
        )
    return matches[0]


def _parse_identity_lines(
    payload: bytes | bytearray, *, label: str
) -> tuple[str, ...]:
    try:
        identities = tuple(
            line.strip() for line in payload.decode("utf-8").splitlines()
        )
    except UnicodeDecodeError as exc:
        raise AuditError(f"{label} is not valid UTF-8") from exc
    if not identities or any(not value for value in identities):
        raise AuditError(f"{label} contains no identities or an empty identity")
    if len(set(identities)) != len(identities):
        raise AuditError(f"{label} contains duplicate identities")
    return identities


class _FsumAccumulator:
    """Bounded-memory, high-accuracy summation for large protocol arrays."""

    def __init__(self) -> None:
        self._buffer: list[float] = []
        self._partials: list[float] = []

    def add(self, value: float) -> None:
        self._buffer.append(value)
        if len(self._buffer) == 8192:
            self._partials.append(math.fsum(self._buffer))
            self._buffer.clear()

    def total(self) -> float:
        if self._buffer:
            self._partials.append(math.fsum(self._buffer))
            self._buffer.clear()
        return math.fsum(self._partials)


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _core_metrics(
    prediction: NpyArray,
    gene_effect: NpyArray,
    test_indices: Sequence[int],
    train_gene_mean: NpyArray,
) -> dict[str, float]:
    n_test, n_genes = prediction.shape
    if gene_effect.shape[0] != n_genes or len(test_indices) != n_test:
        raise AuditError("prediction and target shapes differ")
    if train_gene_mean.shape != (n_genes,):
        raise AuditError("train_gene_mean shape differs from prediction genes")
    predicted = prediction.values
    observed = gene_effect.values
    reference = train_gene_mean.values
    if (
        not isinstance(predicted, array)
        or not isinstance(observed, (array, _MappedFloat32Values))
        or not isinstance(reference, array)
    ):
        raise AuditError("numeric prediction arrays are malformed")
    sse = _FsumAccumulator()
    absolute = _FsumAccumulator()
    baseline_sse = _FsumAccumulator()
    sum_x = _FsumAccumulator()
    sum_y = _FsumAccumulator()
    sum_xx = _FsumAccumulator()
    sum_yy = _FsumAccumulator()
    sum_xy = _FsumAccumulator()
    n_pairs = 0
    n_cells = gene_effect.shape[1]
    for test_row, cell_index in enumerate(test_indices):
        for gene_index in range(n_genes):
            pred = float(predicted[test_row * n_genes + gene_index])
            target = float(observed[gene_index * n_cells + cell_index])
            mean = float(reference[gene_index])
            if not math.isfinite(pred):
                raise AuditError("prediction contains non-finite values")
            if not math.isfinite(target) or not math.isfinite(mean):
                continue
            error = _f32(pred - target)
            baseline_error = _f32(mean - target)
            x = _f32(pred - mean)
            y = _f32(target - mean)
            sse.add(error * error)
            absolute.add(abs(error))
            baseline_sse.add(baseline_error * baseline_error)
            sum_x.add(x)
            sum_y.add(y)
            sum_xx.add(x * x)
            sum_yy.add(y * y)
            sum_xy.add(x * y)
            n_pairs += 1
    if n_pairs == 0:
        raise AuditError("prediction has no finite target/reference pairs")
    sse_value = sse.total()
    baseline_value = baseline_sse.total()
    sx = sum_x.total()
    sy = sum_y.total()
    covariance = sum_xy.total() - sx * sy / n_pairs
    variance_x = sum_xx.total() - sx * sx / n_pairs
    variance_y = sum_yy.total() - sy * sy / n_pairs
    denominator = math.sqrt(max(variance_x, 0.0) * max(variance_y, 0.0))
    return {
        "mse": sse_value / n_pairs,
        "mae": absolute.total() / n_pairs,
        "residual_r2_vs_train_gene_mean": (
            1.0 - sse_value / baseline_value
            if baseline_value > 0.0
            else float("nan")
        ),
        "residual_pearson_pooled": (
            covariance / denominator if n_pairs >= 3 and denominator > 0.0 else float("nan")
        ),
    }


def _assert_metrics(recorded: Any, recomputed: Mapping[str, float]) -> None:
    if not isinstance(recorded, Mapping) or set(recorded) != set(PAIRED_METRICS):
        raise AuditError("outer-fold paired metric contract is incomplete")
    for name in PAIRED_METRICS:
        observed = recorded[name]
        expected = float(recomputed[name])
        if (
            not isinstance(observed, (int, float))
            or isinstance(observed, bool)
            or not math.isfinite(float(observed))
            or not math.isfinite(expected)
            or not math.isclose(float(observed), expected, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise AuditError(f"paired metric differs from independent replay: {name}")


def _expected_train_mean(
    gene_effect: NpyArray, test_indices: Sequence[int]
) -> array:
    values = gene_effect.values
    if not isinstance(values, (array, _MappedFloat32Values)):
        raise AuditError("gene_effect.npy numeric payload is malformed")
    n_genes, n_cells = gene_effect.shape
    held_out = set(test_indices)
    means = array("f")
    for gene_index in range(n_genes):
        accumulator = _FsumAccumulator()
        n_finite = 0
        for cell_index in range(n_cells):
            value = float(values[gene_index * n_cells + cell_index])
            if cell_index not in held_out and math.isfinite(value):
                accumulator.add(value)
                n_finite += 1
        means.append(
            _f32(accumulator.total() / n_finite)
            if n_finite
            else float("nan")
        )
    return means


def _prediction_shape_contract(
    *,
    n_genes: int,
    n_test: int,
) -> dict[str, tuple[int, ...]]:
    return {
        "prediction": (n_test, n_genes),
        "train_gene_mean": (n_genes,),
        "outer_test_indices": (n_test,),
        "source_cell_indices": (n_test,),
        "source_gene_indices": (n_genes,),
        "cell_ids": (n_test,),
        "gene_ids": (n_genes,),
    }


def _prediction_archive_byte_limit(
    *,
    n_genes: int,
    n_test: int,
    expected_gene_ids: Sequence[str],
    expected_cell_ids: Sequence[str],
) -> int:
    max_gene_characters = max(len(value) for value in expected_gene_ids)
    max_cell_characters = max(len(value) for value in expected_cell_ids)
    data_bytes = (
        (n_test * n_genes * 4)
        + (n_genes * 4)
        + ((2 * n_test + n_genes) * 8)
        + (n_test * max_cell_characters * 4)
        + (n_genes * max_gene_characters * 4)
    )
    return (
        data_bytes
        + len(PREDICTION_MEMBERS) * MAX_NPY_HEADER_BYTES
        + MAX_NPZ_CONTAINER_OVERHEAD_BYTES
    )


def _read_exact(stream: BinaryIO, size: int, *, label: str) -> bytes:
    payload = bytearray()
    while len(payload) < size:
        chunk = stream.read(min(1024 * 1024, size - len(payload)))
        if not chunk:
            raise AuditError(f"{label} is truncated")
        payload.extend(chunk)
    return bytes(payload)


def _read_npy_header(stream: BinaryIO) -> tuple[tuple[int, ...], str, int]:
    if _read_exact(stream, 6, label="NPY magic") != b"\x93NUMPY":
        raise AuditError("NPY magic differs")
    version = tuple(_read_exact(stream, 2, label="NPY version"))
    if version == (1, 0):
        header_size = struct.unpack("<H", _read_exact(stream, 2, label="NPY header size"))[0]
    elif version == (2, 0):
        header_size = struct.unpack("<I", _read_exact(stream, 4, label="NPY header size"))[0]
    else:
        raise AuditError(f"unsupported NPY version: {version}")
    if not 1 <= header_size <= MAX_NPY_HEADER_BYTES:
        raise AuditError("NPY header exceeds its protocol limit")
    raw_header = _read_exact(stream, header_size, label="NPY header")
    try:
        header = ast.literal_eval(raw_header.decode("latin1").strip())
    except (SyntaxError, ValueError, UnicodeDecodeError) as exc:
        raise AuditError("NPY header is malformed") from exc
    if not isinstance(header, dict) or set(header) != {"descr", "fortran_order", "shape"}:
        raise AuditError("NPY header field contract differs")
    shape = header["shape"]
    if (
        not isinstance(shape, tuple)
        or not shape
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in shape)
    ):
        raise AuditError("NPY shape is invalid")
    if header["fortran_order"] is not False:
        raise AuditError("NPY arrays must be C-contiguous on disk")
    descr = header["descr"]
    if not isinstance(descr, str):
        raise AuditError("NPY dtype descriptor is invalid")
    if descr == "<f4":
        itemsize = 4
    elif descr == "<i8":
        itemsize = 8
    elif descr.startswith("<U") and descr[2:].isdigit() and int(descr[2:]) >= 1:
        itemsize = 4 * int(descr[2:])
    else:
        raise AuditError("NPY dtype is outside the little-endian protocol")
    return tuple(shape), descr, itemsize


def _read_npy_array(stream: BinaryIO, *, member_size: int) -> NpyArray:
    shape, descr, itemsize = _read_npy_header(stream)
    header_bytes = int(stream.tell())
    data_size = math.prod(shape) * itemsize
    if header_bytes > MAX_NPY_HEADER_BYTES + 12 or member_size != header_bytes + data_size:
        raise AuditError("NPY byte size differs from its header")
    if descr == "<f4":
        values = array("f")
        if values.itemsize != 4:
            raise AuditError("host float32 array width is unsupported")
        remaining = data_size
        while remaining:
            chunk_size = min(NPY_DECODE_CHUNK_BYTES, remaining)
            chunk_size -= chunk_size % itemsize
            raw = _read_exact(stream, chunk_size, label="NPY payload chunk")
            values.frombytes(raw)
            remaining -= len(raw)
        if sys.byteorder != "little":
            values.byteswap()
    elif descr == "<i8":
        values = array("q")
        if values.itemsize != 8:
            raise AuditError("host int64 array width is unsupported")
        remaining = data_size
        while remaining:
            chunk_size = min(NPY_DECODE_CHUNK_BYTES, remaining)
            chunk_size -= chunk_size % itemsize
            raw = _read_exact(stream, chunk_size, label="NPY payload chunk")
            values.frombytes(raw)
            remaining -= len(raw)
        if sys.byteorder != "little":
            values.byteswap()
    else:
        decoded: list[str] = []
        remaining = data_size
        while remaining:
            chunk_size = min(NPY_DECODE_CHUNK_BYTES, remaining)
            chunk_size -= chunk_size % itemsize
            if chunk_size < itemsize:
                chunk_size = itemsize
            raw = _read_exact(stream, chunk_size, label="NPY payload chunk")
            decoded.extend(
                raw[offset : offset + itemsize]
                .decode("utf-32-le")
                .rstrip("\x00")
                for offset in range(0, len(raw), itemsize)
            )
            remaining -= len(raw)
        values = tuple(decoded)
    if stream.read(1):
        raise AuditError("NPY member contains trailing bytes")
    return NpyArray(descr=descr, shape=shape, values=values)


def _validate_prediction_container(
    payload: bytes | bytearray,
    *,
    n_genes: int,
    n_test: int,
    expected_gene_ids: Sequence[str],
    expected_cell_ids: Sequence[str],
) -> dict[str, NpyArray]:
    """Validate and parse the bounded ZIP/NPY structure using only stdlib."""
    archive_limit = _prediction_archive_byte_limit(
        n_genes=n_genes,
        n_test=n_test,
        expected_gene_ids=expected_gene_ids,
        expected_cell_ids=expected_cell_ids,
    )
    if len(payload) > archive_limit:
        raise AuditError("prediction archive exceeds its shape-derived file-size limit")
    expected_shapes = _prediction_shape_contract(n_genes=n_genes, n_test=n_test)
    expected_members = {f"{name}.npy" for name in PREDICTION_MEMBERS}
    max_gene_itemsize = max(len(value) for value in expected_gene_ids) * 4
    max_cell_itemsize = max(len(value) for value in expected_cell_ids) * 4
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(infos) != len(expected_members) or set(names) != expected_members:
                raise AuditError("prediction ZIP member contract differs")
            if len(names) != len(set(names)):
                raise AuditError("prediction ZIP contains duplicate member names")
            total_uncompressed = 0
            arrays: dict[str, NpyArray] = {}
            for info in infos:
                if (
                    info.is_dir()
                    or info.flag_bits & 0x1
                    or info.compress_type
                    not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                ):
                    raise AuditError("prediction ZIP member flags are not permitted")
                total_uncompressed += int(info.file_size)
                if total_uncompressed > archive_limit:
                    raise AuditError(
                        "prediction ZIP exceeds its cumulative uncompressed-size limit"
                    )
                name = info.filename[:-4]
                with archive.open(info, mode="r") as member:
                    parsed = _read_npy_array(member, member_size=int(info.file_size))
                if parsed.shape != expected_shapes[name]:
                    raise AuditError(
                        f"prediction member shape differs before load: {name}"
                    )
                if name in ("prediction", "train_gene_mean"):
                    if parsed.descr != "<f4":
                        raise AuditError(
                            f"prediction member dtype differs before load: {name}"
                        )
                elif name in (
                    "outer_test_indices",
                    "source_cell_indices",
                    "source_gene_indices",
                ):
                    if parsed.descr != "<i8":
                        raise AuditError(
                            f"prediction member dtype differs before load: {name}"
                        )
                else:
                    itemsize_limit = (
                        max_cell_itemsize if name == "cell_ids" else max_gene_itemsize
                    )
                    if not parsed.descr.startswith("<U") or not 4 <= int(parsed.descr[2:]) * 4 <= itemsize_limit:
                        raise AuditError(
                            f"prediction identity dtype exceeds its canonical bound: {name}"
                        )
                arrays[name] = parsed
            return arrays
    except AuditError:
        raise
    except (EOFError, OSError, ValueError, zipfile.BadZipFile) as exc:
        raise AuditError(
            "prediction archive structure cannot be parsed safely"
        ) from exc


def _sidecar_byte_limits(
    *,
    n_genes: int,
    expected_gene_ids: Sequence[str],
) -> tuple[int, int]:
    header_bytes = len(",".join(PER_GENE_COLUMNS).encode("utf-8")) + 2
    longest_gene = max(len(value.encode("utf-8")) for value in expected_gene_ids)
    uncompressed = header_bytes + n_genes * (longest_gene + MAX_SIDECAR_ROW_BYTES)
    return uncompressed + MAX_SIDECAR_CONTAINER_OVERHEAD_BYTES, uncompressed


def _validate_sidecar_payload(
    payload: bytes | bytearray,
    *,
    expected_gene_ids: Sequence[str],
) -> None:
    """Bound gzip expansion and require the fixed CSV schema and row count."""
    compressed_limit, uncompressed_limit = _sidecar_byte_limits(
        n_genes=len(expected_gene_ids), expected_gene_ids=expected_gene_ids
    )
    if len(payload) > compressed_limit:
        raise AuditError("per-gene sidecar exceeds its compressed-size limit")
    expanded = bytearray()
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as stream:
            while chunk := stream.read(64 * 1024):
                expanded.extend(chunk)
                if len(expanded) > uncompressed_limit:
                    raise AuditError(
                        "per-gene sidecar exceeds its cumulative decompressed-size limit"
                    )
    except AuditError:
        raise
    except (EOFError, OSError, zlib.error) as exc:
        raise AuditError("per-gene sidecar gzip stream is invalid") from exc
    try:
        text = expanded.decode("utf-8")
        rows = csv.reader(io.StringIO(text, newline=""), strict=True)
        header = next(rows)
        if tuple(header) != PER_GENE_COLUMNS:
            raise AuditError("per-gene sidecar CSV schema differs")
        row_count = 0
        for row in rows:
            if len(row) != len(PER_GENE_COLUMNS):
                raise AuditError("per-gene sidecar contains a malformed CSV row")
            row_count += 1
            if row_count > len(expected_gene_ids):
                raise AuditError("per-gene sidecar exceeds its row-count limit")
    except (csv.Error, StopIteration, UnicodeDecodeError) as exc:
        raise AuditError("per-gene sidecar is not valid UTF-8 CSV") from exc
    if row_count != len(expected_gene_ids):
        raise AuditError("per-gene sidecar row count differs from the gene cohort")


def _read_prediction(
    payload: bytes | bytearray,
    *,
    gene_effect: NpyArray,
    expected_gene_ids: Sequence[str],
    expected_cell_ids: Sequence[str],
    n_outer_test_cells: Any,
) -> tuple[dict[str, NpyArray], dict[str, float]]:
    if (
        not isinstance(n_outer_test_cells, int)
        or isinstance(n_outer_test_cells, bool)
        or not 1 <= n_outer_test_cells <= gene_effect.shape[1]
    ):
        raise AuditError("reported outer-test cell count is invalid")
    arrays = _validate_prediction_container(
        payload,
        n_genes=int(gene_effect.shape[0]),
        n_test=n_outer_test_cells,
        expected_gene_ids=expected_gene_ids,
        expected_cell_ids=expected_cell_ids,
    )
    prediction = arrays["prediction"]
    train_mean = arrays["train_gene_mean"]
    test_indices = arrays["outer_test_indices"]
    if prediction.descr != "<f4" or len(prediction.shape) != 2:
        raise AuditError("prediction must be a two-dimensional float32 array")
    if train_mean.descr != "<f4" or train_mean.shape != (
        prediction.shape[1],
    ):
        raise AuditError("train_gene_mean must be a matching float32 vector")
    for name in ("outer_test_indices", "source_cell_indices", "source_gene_indices"):
        if arrays[name].descr != "<i8" or len(arrays[name].shape) != 1:
            raise AuditError(f"{name} must be a one-dimensional int64 array")
    if not arrays["cell_ids"].descr.startswith("<U") or not arrays[
        "gene_ids"
    ].descr.startswith("<U"):
        raise AuditError("cell_ids and gene_ids must be non-object Unicode arrays")
    if (
        not isinstance(n_outer_test_cells, int)
        or isinstance(n_outer_test_cells, bool)
        or n_outer_test_cells != prediction.shape[0]
    ):
        raise AuditError("reported outer-test cell count differs from prediction")
    n_cells = int(gene_effect.shape[1])
    n_genes = int(gene_effect.shape[0])
    expected_shapes = {
        "outer_test_indices": (prediction.shape[0],),
        "source_cell_indices": (prediction.shape[0],),
        "source_gene_indices": (n_genes,),
        "cell_ids": (prediction.shape[0],),
        "gene_ids": (n_genes,),
    }
    if prediction.shape[1] != n_genes:
        raise AuditError("prediction gene dimension differs from gene_effect.npy")
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise AuditError(f"prediction identity array shape differs: {name}")
    index_values = tuple(int(value) for value in test_indices.values)
    if not index_values or len(set(index_values)) != len(index_values):
        raise AuditError("outer-test indices are empty or duplicated within a fold")
    if min(index_values) < 0 or max(index_values) >= n_cells:
        raise AuditError("outer-test index is outside gene_effect.npy")
    if tuple(arrays["source_cell_indices"].values) != index_values:
        raise AuditError("source-cell indices differ from canonical cohort indices")
    if tuple(arrays["source_gene_indices"].values) != tuple(range(n_genes)):
        raise AuditError("source-gene indices differ from canonical cohort indices")
    if tuple(arrays["gene_ids"].values) != tuple(expected_gene_ids):
        raise AuditError("prediction gene identities differ from genes.txt")
    expected_fold_cell_ids = tuple(expected_cell_ids[index] for index in index_values)
    if tuple(arrays["cell_ids"].values) != expected_fold_cell_ids:
        raise AuditError("prediction cell identities differ from cell_lines.txt")
    expected_mean = _expected_train_mean(gene_effect, index_values)
    observed_mean = train_mean.values
    if not isinstance(observed_mean, array) or len(observed_mean) != len(expected_mean):
        raise AuditError("train_gene_mean payload is malformed")
    if any(
        not (
            float(observed) == float(expected)
            or (math.isnan(float(observed)) and math.isnan(float(expected)))
        )
        for observed, expected in zip(observed_mean, expected_mean)
    ):
        raise AuditError("train_gene_mean was not fitted on the outer-test complement")
    return arrays, _core_metrics(prediction, gene_effect, index_values, train_mean)


def _write_exclusive_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    forbidden_directories: Sequence[Path] = (),
) -> dict[str, Any]:
    """Publish once with best-effort ancestor and destination race checks."""
    path = Path(os.path.abspath(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    for directory in forbidden_directories:
        _require_outside(path, directory, label="audit output")
    ancestors = _ancestor_identity_snapshot(path)
    parent_stat = path.parent.lstat()
    if _is_link_or_reparse(parent_stat):
        raise AuditError("audit output parent cannot be a link or reparse point")
    if path.exists() or path.is_symlink():
        raise AuditError(f"refusing to overwrite audit output: {path}")
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if len(encoded) > MAX_AUDIT_OUTPUT_BYTES:
        raise AuditError("audit output exceeds its protocol size limit")
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            written = os.fstat(handle.fileno())
            if not stat_module.S_ISREG(written.st_mode):
                raise AuditError("temporary audit output is not a regular file")
        if _ancestor_identity_snapshot(path) != ancestors:
            raise AuditError("audit output ancestors changed before publish")
        for directory in forbidden_directories:
            _require_outside(path.resolve(), directory, label="audit output")
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise AuditError(f"refusing to overwrite audit output: {path}") from exc
        if _ancestor_identity_snapshot(path) != ancestors:
            raise AuditError("audit output ancestors changed during publish")
        for directory in forbidden_directories:
            _require_outside(path.resolve(), directory, label="audit output")
    finally:
        temporary.unlink(missing_ok=True)
    record = _hash_stable_file(path, max_bytes=MAX_AUDIT_OUTPUT_BYTES)
    return record


def _audit_core_impl(
    receipt_path: Path,
    report_path: Path,
    *,
    mapped_resources: list[_MappedFloat32Values],
) -> dict[str, Any]:
    _, source_binding = _read_stable_bytes(SCRIPT_PATH, max_bytes=MAX_SOURCE_BYTES)
    stability_bindings: list[tuple[Path, Mapping[str, Any], str]] = []
    receipt_payload, receipt_binding = _read_stable_bytes(
        receipt_path, max_bytes=MAX_RECEIPT_BYTES
    )
    receipt = _parse_json(receipt_payload, label="detached receipt")
    if not isinstance(receipt, Mapping) or set(receipt) != RECEIPT_FIELDS:
        raise AuditError("detached receipt field contract differs")
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise AuditError("detached receipt schema differs")
    if receipt.get("primary_verifier_mode") != PRIMARY_VERIFIER_MODE:
        raise AuditError("detached receipt primary verifier mode differs")
    if not isinstance(receipt.get("created_utc"), str) or not receipt["created_utc"]:
        raise AuditError("detached receipt creation time is missing")
    if (
        receipt.get("protocol_complete") is not True
        or receipt.get("scientific_result") is not True
    ):
        raise AuditError("detached receipt does not bind a successful primary report")
    if not isinstance(receipt.get("limitations"), list) or not receipt["limitations"]:
        raise AuditError("detached receipt limitations are missing")

    artifact_root_raw = receipt.get("artifact_root")
    campaign_raw = receipt.get("campaign_directory")
    if (
        not isinstance(artifact_root_raw, str)
        or not Path(artifact_root_raw).is_absolute()
        or not isinstance(campaign_raw, str)
        or not Path(campaign_raw).is_absolute()
    ):
        raise AuditError("receipt artifact/campaign roots must be absolute paths")
    artifact_root = _resolve_verified_directory_root(
        Path(artifact_root_raw), label="receipt artifact root"
    )
    campaign_directory = _resolve_verified_directory_root(
        Path(campaign_raw), label="receipt campaign directory"
    )
    try:
        campaign_directory.relative_to(artifact_root)
    except ValueError as exc:
        raise AuditError("receipt campaign directory is outside artifact root") from exc
    _require_outside(receipt_path, artifact_root, label="detached receipt")
    _require_outside(receipt_path, campaign_directory, label="detached receipt")
    _require_outside(report_path, artifact_root, label="primary verification report")
    _require_outside(
        report_path, campaign_directory, label="primary verification report"
    )

    report_payload, report_binding = _read_stable_bytes(
        report_path, max_bytes=MAX_REPORT_BYTES
    )
    report_record = receipt.get("verification_report")
    if not isinstance(report_record, Mapping):
        raise AuditError("receipt verification_report binding is malformed")
    _require_file_binding(
        report_binding,
        report_record,
        label="primary verification report",
        require_path=True,
    )
    report = _parse_json(report_payload, label="primary verification report")
    if not isinstance(report, Mapping) or report.get("schema") != PRIMARY_REPORT_SCHEMA:
        raise AuditError("primary verification report schema differs")
    if (
        report.get("protocol_complete") is not True
        or report.get("scientific_result") is not True
    ):
        raise AuditError("primary verification report is not successful")
    gates = report.get("verification_gates")
    if (
        not isinstance(gates, Mapping)
        or set(gates) != PRIMARY_GATES
        or any(value is not True for value in gates.values())
    ):
        raise AuditError("primary verification gates are incomplete or false")
    if Path(str(report.get("artifact_root"))).resolve() != artifact_root:
        raise AuditError("report and receipt artifact roots differ")
    if Path(str(report.get("campaign_directory"))).resolve() != campaign_directory:
        raise AuditError("report and receipt campaign directories differ")
    for field in ("campaign_plan", "campaign_completion"):
        if not _strict_equal(report.get(field), receipt.get(field)):
            raise AuditError(f"report and receipt {field} bindings differ")
        metadata_path, _, metadata_binding = _read_bound_bytes(
            receipt[field],
            root=artifact_root,
            label=field,
            max_bytes=MAX_METADATA_BYTES,
        )
        stability_bindings.append((metadata_path, metadata_binding, field))

    primary_source = receipt.get("primary_verifier_source")
    if not isinstance(primary_source, Mapping) or not _strict_equal(
        report.get("verified_verifier_source"), primary_source
    ):
        raise AuditError("primary verifier source bindings differ")
    primary_source_path, _, primary_source_binding = _read_bound_bytes(
        primary_source,
        root=REPOSITORY_ROOT,
        label="primary verifier source",
        max_bytes=MAX_SOURCE_BYTES,
    )
    stability_bindings.append(
        (
            primary_source_path,
            primary_source_binding,
            "primary verifier source",
        )
    )
    if report.get("n_verified_variant_seed_jobs") != 21:
        raise AuditError("primary report does not contain exactly 21 jobs")
    if report.get("n_verified_outer_fold_evaluations") != 105:
        raise AuditError("primary report does not contain exactly 105 outer folds")
    if report.get("n_verified_prediction_and_sidecar_artifacts") != 210:
        raise AuditError("primary report does not contain exactly 210 result artifacts")

    verified_inputs = report.get("verified_inputs")
    gene_effect_record = _find_input_record(
        verified_inputs, "data/processed/gene_effect.npy"
    )
    genes_record = _find_input_record(verified_inputs, "data/processed/genes.txt")
    cells_record = _find_input_record(verified_inputs, "data/processed/cell_lines.txt")
    (
        gene_effect_path,
        gene_effect,
        gene_effect_binding,
        gene_effect_mapping,
    ) = _open_bound_float32_npy(
        gene_effect_record,
        root=REPOSITORY_ROOT,
        label="gene_effect.npy",
        max_bytes=MAX_GENE_EFFECT_BYTES,
    )
    mapped_resources.append(gene_effect_mapping)
    genes_path, genes_payload, genes_binding = _read_bound_bytes(
        genes_record,
        root=REPOSITORY_ROOT,
        label="genes.txt",
        max_bytes=MAX_IDENTITY_BYTES,
    )
    cells_path, cells_payload, cells_binding = _read_bound_bytes(
        cells_record,
        root=REPOSITORY_ROOT,
        label="cell_lines.txt",
        max_bytes=MAX_IDENTITY_BYTES,
    )
    stability_bindings.extend(
        [
            (gene_effect_path, gene_effect_binding, "gene_effect.npy"),
            (genes_path, genes_binding, "genes.txt"),
            (cells_path, cells_binding, "cell_lines.txt"),
        ]
    )
    if (
        gene_effect.descr != "<f4"
        or len(gene_effect.shape) != 2
        or min(gene_effect.shape) < 1
    ):
        raise AuditError("gene_effect.npy must be float32 with shape (genes, cells)")
    expected_gene_ids = _parse_identity_lines(genes_payload, label="genes.txt")
    expected_cell_ids = _parse_identity_lines(cells_payload, label="cell_lines.txt")
    if gene_effect.shape != (len(expected_gene_ids), len(expected_cell_ids)):
        raise AuditError(
            "gene_effect.npy shape differs from genes.txt and cell_lines.txt"
        )

    rows = report.get("outer_fold_metrics")
    if not isinstance(rows, list) or len(rows) != 105:
        raise AuditError("primary report outer_fold_metrics must contain 105 rows")
    expected_identities = _expected_identities()
    observed_identities: set[tuple[str, int, str]] = set()
    artifact_paths: set[Path] = set()
    coverage: dict[tuple[str, int], list[tuple[int, ...]]] = {}
    cell_identity: dict[int, str] = {}
    observed_gene_ids: tuple[str, ...] | None = None
    audit_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise AuditError("primary metric row is not a JSON object")
        identity = (row.get("variant"), row.get("seed"), row.get("fold"))
        if identity not in expected_identities or identity in observed_identities:
            raise AuditError(f"unexpected or duplicate outer-fold identity: {identity}")
        observed_identities.add(identity)
        prediction_record = row.get("prediction_artifact")
        sidecar_record = row.get("per_gene_sidecar")
        n_outer_test_cells = row.get("n_outer_test_cells")
        if (
            not isinstance(n_outer_test_cells, int)
            or isinstance(n_outer_test_cells, bool)
            or not 1 <= n_outer_test_cells <= gene_effect.shape[1]
        ):
            raise AuditError("reported outer-test cell count is invalid")
        prediction_byte_limit = _prediction_archive_byte_limit(
            n_genes=int(gene_effect.shape[0]),
            n_test=n_outer_test_cells,
            expected_gene_ids=expected_gene_ids,
            expected_cell_ids=expected_cell_ids,
        )
        sidecar_byte_limit, _ = _sidecar_byte_limits(
            n_genes=int(gene_effect.shape[0]),
            expected_gene_ids=expected_gene_ids,
        )
        prediction_path, prediction_payload, prediction_binding = _read_bound_bytes(
            prediction_record,
            root=artifact_root,
            label=f"prediction {identity}",
            max_bytes=prediction_byte_limit,
        )
        sidecar_path, sidecar_payload, sidecar_binding = _read_bound_bytes(
            sidecar_record,
            root=artifact_root,
            label=f"per-gene sidecar {identity}",
            max_bytes=sidecar_byte_limit,
        )
        if prediction_path.suffix.lower() != ".npz":
            raise AuditError(f"prediction artifact is not NPZ: {identity}")
        if not str(sidecar_path).lower().endswith(".csv.gz"):
            raise AuditError(f"per-gene sidecar is not gzip CSV: {identity}")
        for path in (prediction_path, sidecar_path):
            if path in artifact_paths:
                raise AuditError("prediction/sidecar artifact path is reused")
            artifact_paths.add(path)
        stability_bindings.extend(
            [
                (prediction_path, prediction_binding, f"prediction {identity}"),
                (sidecar_path, sidecar_binding, f"per-gene sidecar {identity}"),
            ]
        )
        try:
            _validate_sidecar_payload(
                sidecar_payload,
                expected_gene_ids=expected_gene_ids,
            )
        except AuditError as exc:
            raise AuditError(f"per-gene sidecar is invalid: {identity}: {exc}") from exc
        arrays, metrics = _read_prediction(
            prediction_payload,
            gene_effect=gene_effect,
            expected_gene_ids=expected_gene_ids,
            expected_cell_ids=expected_cell_ids,
            n_outer_test_cells=n_outer_test_cells,
        )
        _assert_metrics(row.get("metrics"), metrics)
        test_indices = tuple(int(value) for value in arrays["outer_test_indices"].values)
        coverage.setdefault((str(identity[0]), int(identity[1])), []).append(
            test_indices
        )
        for index, cell_id in zip(test_indices, arrays["cell_ids"].values):
            key = int(index)
            value = str(cell_id)
            if key in cell_identity and cell_identity[key] != value:
                raise AuditError("cell identity is inconsistent across predictions")
            cell_identity[key] = value
        if observed_gene_ids is None:
            observed_gene_ids = tuple(str(value) for value in arrays["gene_ids"].values)
        elif observed_gene_ids != tuple(arrays["gene_ids"].values):
            raise AuditError("gene identities are inconsistent across predictions")
        indices_digest = hashlib.sha256()
        for index in test_indices:
            indices_digest.update(struct.pack("<q", index))
        audit_rows.append(
            {
                "variant": identity[0],
                "seed": identity[1],
                "fold": identity[2],
                "n_outer_test_cells": len(test_indices),
                "outer_test_indices_sha256": indices_digest.hexdigest(),
                "prediction_artifact": prediction_binding,
                "per_gene_sidecar": sidecar_binding,
                "independently_recomputed_metrics": metrics,
            }
        )
    if observed_identities != expected_identities:
        raise AuditError("outer-fold identity coverage differs from the fixed matrix")
    if len(coverage) != len(VARIANTS) * len(SEEDS):
        raise AuditError("variant/seed coverage differs from the fixed matrix")
    if len(artifact_paths) != 2 * len(expected_identities):
        raise AuditError(
            "prediction/sidecar path coverage differs from the fixed matrix"
        )
    expected_cells = tuple(range(gene_effect.shape[1]))
    for identity, fold_indices in coverage.items():
        if len(fold_indices) != 5:
            raise AuditError(f"variant/seed does not contain five folds: {identity}")
        combined = tuple(index for fold in fold_indices for index in fold)
        if len(combined) != len(expected_cells) or tuple(sorted(combined)) != expected_cells:
            raise AuditError(f"outer-test coverage is not exact-once: {identity}")
    if set(cell_identity) != set(range(gene_effect.shape[1])):
        raise AuditError("cell identities do not cover the complete cohort")

    for path, binding, label in stability_bindings:
        observed = _hash_stable_file(path, max_bytes=int(binding["bytes"]))
        _require_file_binding(
            observed,
            binding,
            label=f"{label} end-of-audit stability",
        )
    receipt_end = _hash_stable_file(
        receipt_path, max_bytes=int(receipt_binding["bytes"])
    )
    _require_file_binding(
        receipt_end, receipt_binding, label="detached receipt stability"
    )
    report_end = _hash_stable_file(
        report_path, max_bytes=int(report_binding["bytes"])
    )
    _require_file_binding(report_end, report_binding, label="primary report stability")
    source_end = _hash_stable_file(
        SCRIPT_PATH, max_bytes=int(source_binding["bytes"])
    )
    _require_file_binding(
        source_end, source_binding, label="independent verifier source stability"
    )
    return {
        "schema": AUDIT_SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "audit_mode": "implementation_independent_read_only_core_replay",
        "audit_scope": (
            "Receipt/report binding, campaign metadata hashes, prediction and sidecar "
            "hashes, exact fixed identity coverage, exact-once outer-test coverage, "
            "outer-training gene means, and four paired metrics."
        ),
        "limitations": [
            "This implementation does not replay model training or fold-local preprocessing.",
            "Hash identity does not establish source-data truth or biological provenance.",
            "This audit is not independent biological, external-cohort, or clinical validation.",
            "Passing this audit does not make folds or seeds IID and adds no inferential claim.",
            "Path, opened-handle, and ancestor identity checks are best-effort; portable "
            "user-space checks cannot eliminate every same-account concurrent filesystem race.",
        ],
        "artifact_root": str(artifact_root),
        "campaign_directory": str(campaign_directory),
        "independent_verifier_source": source_binding,
        "detached_receipt": receipt_binding,
        "primary_verification_report": report_binding,
        "campaign_plan": receipt["campaign_plan"],
        "campaign_completion": receipt["campaign_completion"],
        "gene_effect_input": gene_effect_binding,
        "gene_identity_input": genes_binding,
        "cell_identity_input": cells_binding,
        "runtime": {
            "python": sys.version,
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "implementation": "python-standard-library-only",
        },
        "verification_gates": {
            "detached_receipt_report_binding": True,
            "primary_report_contract": True,
            "campaign_plan_and_completion_hashes": True,
            "primary_verifier_source_hash": True,
            "gene_effect_input_hash": True,
            "gene_and_cell_identity_hashes": True,
            "exact_105_identity_coverage": True,
            "all_105_prediction_hashes": True,
            "prediction_zip_npy_shape_and_resource_bounds": True,
            "all_105_sidecar_hashes_and_gzip_integrity": True,
            "sidecar_gzip_csv_schema_and_resource_bounds": True,
            "outer_test_exact_once": True,
            "outer_train_gene_means_recomputed": True,
            "four_paired_metrics_recomputed": True,
            "end_of_audit_stability": True,
        },
        "n_verified_variant_seed_jobs": len(coverage),
        "n_verified_outer_fold_evaluations": len(audit_rows),
        "n_verified_prediction_and_sidecar_artifacts": len(artifact_paths),
        "outer_fold_core_evidence": sorted(
            audit_rows,
            key=lambda item: (item["variant"], item["seed"], item["fold"]),
        ),
        "audit_passed": True,
        "protocol_complete": True,
        "scientific_result": False,
        "scientific_result_reason": (
            "This artifact is an independent core-evidence audit, not a new or "
            "independent scientific validation result."
        ),
        "errors": [],
    }


def _audit_core(receipt_path: Path, report_path: Path) -> dict[str, Any]:
    mapped_resources: list[_MappedFloat32Values] = []
    try:
        return _audit_core_impl(
            receipt_path,
            report_path,
            mapped_resources=mapped_resources,
        )
    finally:
        for resource in reversed(mapped_resources):
            resource.close()


def verify(
    receipt_path: Path,
    report_path: Path,
    *,
    output_path: Path | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    receipt_path = Path(os.path.abspath(receipt_path))
    report_path = Path(os.path.abspath(report_path))
    if output_path is None:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        output_path = receipt_path.parent / (
            f"independent-k-token-pmi-audit-{stamp}-{uuid.uuid4().hex[:8]}.json"
        )
    output_path = Path(os.path.abspath(output_path))
    report = _audit_core(receipt_path, report_path)
    _require_outside(output_path, Path(report["artifact_root"]), label="audit output")
    _require_outside(
        output_path, Path(report["campaign_directory"]), label="audit output"
    )
    output_binding = _write_exclusive_json(
        output_path,
        report,
        forbidden_directories=(
            Path(report["artifact_root"]),
            Path(report["campaign_directory"]),
        ),
    )
    return output_path, report, output_binding


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--verification-report", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="exclusive audit JSON path; default is a unique file beside the receipt",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output_path, report, binding = verify(
            args.receipt,
            args.verification_report,
            output_path=args.output,
        )
    except Exception as exc:
        print(
            f"[k-token-pmi-independent] audit_passed=False "
            f"error={type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    print(
        f"[k-token-pmi-independent] output={output_path} "
        f"sha256={binding['sha256']} audit_passed={report['audit_passed']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
