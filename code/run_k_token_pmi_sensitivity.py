"""Run leakage-controlled K, mutation-token, and PMI sensitivity analyses.

The ``audit`` command evaluates fitted preprocessing quantities over prespecified
outer folds without training a model.  It is useful for checking K enforcement,
token-order changes, PMI distributions, and affected STRING edges, but it is not
model-performance evidence.  The ``train`` command delegates the nested training
and exact-once outer-test evaluation to the corrected runner after replacing only
the fold-local mutation-token and PMI fit with a prespecified sensitivity variant.
The ``campaign`` command executes the fixed 7-variant by 3-seed by 5-fold matrix
and resumes only from complete, code-bound, hash-verified variant-seed runs.  The
``verify`` command replays the saved outer-test evaluation in a read-only process
and emits paired descriptive summaries only after every protocol gate passes.
"""

from __future__ import annotations

import argparse
import ast
import functools
import gzip
import hashlib
import importlib.metadata as importlib_metadata
import io
import json
import os
import platform
import stat as stat_module
import subprocess
import sys
import tempfile
import time
import uuid
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from scipy.stats import rankdata

import run_corrected_experiments as corrected
from corrected_protocol import (
    atomic_write_json,
    derive_seed,
    fit_expression_pca,
    fit_lineage_one_hot,
    make_kfold_outer_splits,
    make_unique_run_dir,
    selectivity_labels_from_train,
    set_global_seed,
)
from k_token_pmi_protocol import (
    IMPACT_RANK,
    MAX_TOKEN_SLOTS,
    VARIANTS,
    PmiFit,
    SensitivityMutationFeatures,
    SensitivityVariant,
    TokenAnnotation,
    feature_manifest,
    fit_pmi_sensitivity,
    fit_sensitivity_mutation_features,
    load_token_annotations,
    pmi_per_gene_summary,
    variant_by_name,
)
from models import normalized_adjacency


ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
DEPMAP = ROOT / "data" / "depmap"
OUTPUT_ROOT = ROOT / "results" / "k_token_pmi_v1"
CANONICAL_SEEDS = (42, 43, 44)
CANONICAL_FOLDS = 5
REFERENCE_VARIANT = "k64-frequency"
VERIFICATION_SCHEMA = "DepPLM-k-token-pmi-campaign-verification-v1"
DETACHED_RECEIPT_SCHEMA = "DepPLM-k-token-pmi-detached-verification-receipt-v1"
CAMPAIGN_RESUME_POLICY = (
    "Reuse only a unique canonical variant-seed run whose arguments and "
    "training-execution, dependency, and canonical-input hashes match this "
    "plan and whose five prediction plus five per-gene artifacts still match "
    "recorded byte counts and SHA-256 values."
)
CAMPAIGN_SCIENTIFIC_RESULT_REASON = (
    "The prediction matrix is complete and hash-bound but requires the "
    "prespecified paired descriptive aggregation before interpretation."
)
CRITICAL_RUNTIME_DISTRIBUTIONS = frozenset(
    {"numpy", "pandas", "scipy", "scikit-learn", "torch"}
)
SOURCE_TREE_EXCLUDED_DIRECTORIES = frozenset(
    {
        ".git",
        ".hg",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "venv",
    }
)
MAX_ENVIRONMENT_HASH_FILES = 100_000
ENVIRONMENT_HASH_CHUNK_BYTES = 4 * 1024**2
MAX_ENVIRONMENT_HASH_CHUNK_BYTES = 64 * 1024**2
MAX_JSON_ARTIFACT_BYTES = 256 * 1024**2
MAX_BOUND_ARTIFACT_BYTES = 4 * 1024**3
MAX_SOURCE_BYTES = 16 * 1024**2
MAX_IDENTITY_BYTES = 64 * 1024**2
MAX_GENE_EFFECT_BYTES = 4 * 1024**3
PAIRED_METRICS = (
    "residual_r2_vs_train_gene_mean",
    "residual_pearson_pooled",
    "mse",
    "mae",
)
FILE_RECORD_FIELDS = frozenset({"path", "bytes", "mtime_ns", "sha256"})
TRAINING_RUN_RECORD_FIELDS = frozenset(
    {"variant", "seed", "source", "run_directory", "run_manifest", "completion"}
)
PREDICTION_RECORD_FIELDS = FILE_RECORD_FIELDS | frozenset(
    {"variant", "seed", "fold", "schema", "arrays"}
)
SIDECAR_RECORD_FIELDS = FILE_RECORD_FIELDS | frozenset(
    {"variant", "seed", "fold"}
)
TRAINING_PREDICTION_RECORD_FIELDS = FILE_RECORD_FIELDS | frozenset(
    {"fold", "schema", "arrays"}
)
TRAINING_SIDECAR_RECORD_FIELDS = FILE_RECORD_FIELDS | frozenset({"fold"})
FINAL_CAMPAIGN_JOB_FIELDS = frozenset(
    {
        "job_number",
        "source",
        "variant",
        "seed",
        "run_directory",
        "run_manifest",
        "completion",
        "outer_test_predictions",
        "per_gene_prediction_sidecars",
        "cohort_manifest",
        "cohort_payload",
        "fold_protocol_artifacts",
    }
)
VERIFIER_CODE_PATH = Path(__file__).resolve()
TRAINING_CODE_PATHS = (
    ROOT / "code" / "k_token_pmi_protocol.py",
    ROOT / "code" / "run_corrected_experiments.py",
    ROOT / "code" / "corrected_protocol.py",
    ROOT / "code" / "embedding_provenance.py",
    ROOT / "code" / "models.py",
)
TRAINING_EXECUTION_SIGNATURE_SCHEMA = "DepPLM-k-token-pmi-training-signature-v2"
TRAINING_EXECUTION_COMPONENTS = (
    "_canonical_input_paths",
    "_sha256_file",
    "_stable_file_record",
    "_file_record",
    "_hash_strings",
    "_is_sha256",
    "_atomic_write_gzip_csv",
    "_runtime_record",
    "_canonical_training_namespace",
    "_campaign_jobs",
    "_normalised_record_path",
    "_code_signature",
    "_verify_current_bindings",
    "_variant_fold_data",
    "_patched_corrected_runner",
    "_per_gene_prediction_summary_from_arrays",
    "_per_gene_prediction_summary",
    "_training_completion",
    "run_training",
)
TRAINING_EXECUTION_CONSTANTS = (
    "CANONICAL_SEEDS",
    "CANONICAL_FOLDS",
    "CANONICAL_TRAINING_SETTINGS",
)
PER_GENE_COLUMNS = (
    "gene_index",
    "gene",
    "n_outer_test_observations",
    "mse",
    "mae",
    "train_gene_mean_baseline_mse",
    "mse_improvement_over_train_gene_mean",
)
CANONICAL_TRAINING_SETTINGS = {
    "protocol": "p1",
    "fold_id": -1,
    "folds": CANONICAL_FOLDS,
    "inner_val_fraction": 0.2,
    "inner_max_epochs": 20,
    "patience": 5,
    "batch_size": 64,
    "gene_subsample": 6_000,
    "top_mut": 5_000,
    "n_expr_pc": 128,
    "pilot": False,
    "pilot_cells": 96,
    "pilot_genes": 128,
}


@dataclass(frozen=True)
class VerificationBase:
    """Minimum immutable cohort state needed to replay outer-test evaluation."""

    gene_effect: np.ndarray
    genes: tuple[str, ...]
    cell_ids: tuple[str, ...]
    lineages: np.ndarray
    mutations_by_cell: tuple[frozenset[str], ...]
    token_annotations: tuple[Mapping[str, Any], ...]
    edge_index: np.ndarray
    source_cell_indices: np.ndarray
    source_gene_indices: np.ndarray


class VerificationError(RuntimeError):
    """Raised when a campaign cannot satisfy a fail-closed verification gate."""


def _is_link_or_reparse(stat_result: os.stat_result) -> bool:
    file_attributes = int(getattr(stat_result, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat_module, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return stat_module.S_ISLNK(stat_result.st_mode) or bool(
        file_attributes & reparse_flag
    )


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
    )


def _ancestor_identity_snapshot(path: Path) -> tuple[tuple[Any, ...], ...]:
    """Capture lexical ancestor identities without following link-like entries."""
    absolute = Path(os.path.abspath(path))
    chain = list(reversed((absolute.parent, *absolute.parent.parents)))
    snapshot: list[tuple[Any, ...]] = []
    for ancestor in chain:
        try:
            observed = ancestor.lstat()
        except OSError as exc:
            raise VerificationError(
                f"artifact ancestor cannot be stat'ed: {ancestor}"
            ) from exc
        if _is_link_or_reparse(observed):
            raise VerificationError(
                f"symbolic link or reparse ancestor is forbidden: {ancestor}"
            )
        if not stat_module.S_ISDIR(observed.st_mode):
            raise VerificationError(f"artifact ancestor is not a directory: {ancestor}")
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
    """Reject link-like lexical roots before obtaining their canonical path."""
    lexical = Path(os.path.abspath(path))
    before = _ancestor_identity_snapshot(lexical / ".root-identity-probe")
    try:
        root_before = lexical.lstat()
    except OSError as exc:
        raise VerificationError(f"{label} cannot be stat'ed: {lexical}") from exc
    if _is_link_or_reparse(root_before):
        raise VerificationError(f"{label} may not be a symbolic link or reparse point")
    if not stat_module.S_ISDIR(root_before.st_mode):
        raise VerificationError(f"{label} is not a directory: {lexical}")
    try:
        canonical = lexical.resolve(strict=True)
    except OSError as exc:
        raise VerificationError(f"{label} cannot be resolved: {lexical}") from exc
    after = _ancestor_identity_snapshot(lexical / ".root-identity-probe")
    try:
        root_after = lexical.lstat()
    except OSError as exc:
        raise VerificationError(f"{label} disappeared during validation") from exc
    if (
        before != after
        or _is_link_or_reparse(root_after)
        or _file_identity(root_before) != _file_identity(root_after)
    ):
        raise VerificationError(f"{label} changed during validation")
    return canonical


def _open_readonly_nofollow(path: Path) -> Any:
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    descriptor = os.open(path, flags)
    return os.fdopen(descriptor, "rb", closefd=True)


def _canonical_input_paths() -> tuple[Path, ...]:
    """Return every source input or ESM provenance artifact used by training."""
    return (
        PROCESSED / "gene_effect.npy",
        ROOT / corrected.EMBEDDING_RELATIVE_PATH,
        ROOT / corrected.NORMALIZED_RELATIVE_PATH,
        ROOT / corrected.NAMES_RELATIVE_PATH,
        ROOT / corrected.MANIFEST_RELATIVE_PATH,
        PROCESSED / "gene_seqs.fa",
        PROCESSED / "ppi_graph.npz",
        PROCESSED / "genes.txt",
        PROCESSED / "cell_lines.txt",
        PROCESSED / "lineages.txt",
        DEPMAP / "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
        DEPMAP / "OmicsSomaticMutations.csv",
    )


def _read_lines(path: Path) -> list[str]:
    payload, _ = _read_file_bytes(path, max_bytes=MAX_IDENTITY_BYTES)
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VerificationError(f"identity file is not UTF-8: {path}") from exc
    return [line.strip() for line in text.splitlines()]


def _stable_file_record(
    path: Path,
    *,
    relative_to: Path = ROOT,
    chunk_size: int = 4 * 1024 * 1024,
    snapshot_handle: Any | None = None,
    max_bytes: int = MAX_BOUND_ARTIFACT_BYTES,
) -> dict[str, Any]:
    """Hash one stable handle and optionally copy those exact bytes."""
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1:
        raise VerificationError("artifact byte limit must be a positive integer")
    max_bytes = min(max_bytes, MAX_BOUND_ARTIFACT_BYTES)
    path = Path(os.path.abspath(path))
    ancestors_before = _ancestor_identity_snapshot(path)
    try:
        path_before = path.lstat()
    except OSError as exc:
        raise VerificationError(f"artifact cannot be stat'ed: {path}") from exc
    if _is_link_or_reparse(path_before):
        raise VerificationError(
            f"symbolic link or reparse artifact is forbidden: {path}"
        )
    if not stat_module.S_ISREG(path_before.st_mode):
        raise VerificationError(f"artifact is not a regular file: {path}")
    if int(path_before.st_size) > max_bytes:
        raise VerificationError(
            f"artifact exceeds the {max_bytes}-byte protocol limit: "
            f"{path}"
        )

    digest = hashlib.sha256()
    byte_count = 0
    try:
        if snapshot_handle is not None:
            snapshot_handle.seek(0)
            snapshot_handle.truncate(0)
        with _open_readonly_nofollow(path) as handle:
            before = os.fstat(handle.fileno())
            if not stat_module.S_ISREG(before.st_mode):
                raise VerificationError(f"artifact is not a regular file: {path}")
            if _file_identity(path_before) != _file_identity(before):
                raise VerificationError(
                    f"artifact path changed before its handle was opened: {path}"
                )
            if int(before.st_size) > max_bytes:
                raise VerificationError(
                    f"artifact exceeds the {max_bytes}-byte protocol limit: {path}"
                )
            while chunk := handle.read(int(chunk_size)):
                digest.update(chunk)
                byte_count += len(chunk)
                if byte_count > max_bytes:
                    raise VerificationError(
                        f"artifact exceeds the {max_bytes}-byte protocol limit: {path}"
                    )
                if snapshot_handle is not None:
                    snapshot_handle.write(chunk)
            after = os.fstat(handle.fileno())
        if snapshot_handle is not None:
            snapshot_handle.flush()
            os.fsync(snapshot_handle.fileno())
            snapshot_handle.seek(0)
    except OSError as exc:
        raise VerificationError(f"artifact cannot be read: {path}") from exc

    if (
        _file_identity(before) != _file_identity(after)
        or byte_count != int(after.st_size)
        or byte_count < 1
    ):
        raise VerificationError(f"artifact changed while being hashed: {path}")
    try:
        path_after = path.lstat()
    except OSError as exc:
        raise VerificationError(
            f"artifact disappeared while being hashed: {path}"
        ) from exc
    if _is_link_or_reparse(path_after) or _file_identity(path_after) != _file_identity(
        after
    ):
        raise VerificationError(f"artifact path changed while being hashed: {path}")
    if _ancestor_identity_snapshot(path) != ancestors_before:
        raise VerificationError(f"artifact ancestor changed while being hashed: {path}")
    try:
        display_path = path.relative_to(relative_to)
    except ValueError:
        display_path = path
    return {
        "path": str(display_path),
        "bytes": byte_count,
        "mtime_ns": int(after.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def _sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    return str(
        _stable_file_record(
            path,
            relative_to=path.parent,
            chunk_size=chunk_size,
        )["sha256"]
    )


def _file_record(path: Path, *, relative_to: Path = ROOT) -> dict[str, Any]:
    return _stable_file_record(path, relative_to=relative_to)


def _hash_strings(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _hash_indices(values: Sequence[int] | np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype="<i8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _atomic_write_npz(path: Path, **arrays: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(
                handle,
                **{name: np.ascontiguousarray(value) for name, value in arrays.items()},
            )
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RuntimeError(
                f"refusing to overwrite existing output: {path}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return _file_record(path, relative_to=path.parent.parent)


def _atomic_write_gzip_csv(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    """Write a deterministic gzip CSV without embedding a temporary file name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
                with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                    frame.to_csv(text, index=False, lineterminator="\n")
            raw.flush()
            os.fsync(raw.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RuntimeError(
                f"refusing to overwrite existing output: {path}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return _file_record(path, relative_to=path.parent.parent)


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalise_distribution_name(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def _hash_environment_file(path: Path) -> dict[str, Any]:
    """Hash one regular file exactly with memory bounded by one fixed-size chunk."""
    chunk_bytes = ENVIRONMENT_HASH_CHUNK_BYTES
    if (
        not isinstance(chunk_bytes, int)
        or isinstance(chunk_bytes, bool)
        or chunk_bytes < 1
        or chunk_bytes > MAX_ENVIRONMENT_HASH_CHUNK_BYTES
    ):
        raise VerificationError(
            "environment hash chunk size must be an integer between 1 and "
            f"{MAX_ENVIRONMENT_HASH_CHUNK_BYTES} bytes"
        )
    path = Path(os.path.abspath(path))
    ancestors_before = _ancestor_identity_snapshot(path)
    try:
        path_before = path.lstat()
    except OSError as exc:
        raise VerificationError(f"environment file cannot be stat'ed: {path}") from exc
    if _is_link_or_reparse(path_before):
        raise VerificationError(
            f"environment file cannot be a link or reparse point: {path}"
        )
    if not stat_module.S_ISREG(path_before.st_mode):
        raise VerificationError(
            f"environment content is not a regular file: {path}"
        )
    digest = hashlib.sha256()
    byte_count = 0
    trailing_byte = b""
    try:
        with _open_readonly_nofollow(path) as handle:
            before = os.fstat(handle.fileno())
            if not stat_module.S_ISREG(before.st_mode):
                raise VerificationError(
                    f"environment content is not a regular file: {path}"
                )
            if _file_identity(path_before) != _file_identity(before):
                raise VerificationError(
                    f"environment file changed before opening: {path}"
                )
            expected_bytes = int(before.st_size)
            while byte_count < expected_bytes:
                chunk = handle.read(min(chunk_bytes, expected_bytes - byte_count))
                if not chunk:
                    break
                byte_count += len(chunk)
                digest.update(chunk)
            trailing_byte = handle.read(1)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise VerificationError(f"environment file cannot be read: {path}") from exc
    try:
        path_after = path.lstat()
    except OSError as exc:
        raise VerificationError(f"environment file disappeared: {path}") from exc
    if (
        _file_identity(before) != _file_identity(after)
        or _file_identity(path_after) != _file_identity(after)
        or _is_link_or_reparse(path_after)
        or byte_count != int(after.st_size)
        or trailing_byte != b""
        or _ancestor_identity_snapshot(path) != ancestors_before
    ):
        raise VerificationError(f"environment file changed while being hashed: {path}")
    return {"bytes": byte_count, "sha256": digest.hexdigest()}


def _aggregate_environment_content(
    entries: Sequence[tuple[str, Path | None]],
    *,
    label: str,
) -> dict[str, Any]:
    """Create a deterministic exact digest with bounded memory and file count.

    Each regular file is streamed through a fixed-size buffer.  The bytes read
    are bounded by the size captured from the opened file descriptor, with an
    extra byte read to fail closed if the file grows during hashing.  A total
    byte cap would make exact binding impossible for legitimate large runtime
    distributions such as CUDA-enabled PyTorch wheels.
    """
    chunk_bytes = ENVIRONMENT_HASH_CHUNK_BYTES
    if (
        not isinstance(chunk_bytes, int)
        or isinstance(chunk_bytes, bool)
        or chunk_bytes < 1
        or chunk_bytes > MAX_ENVIRONMENT_HASH_CHUNK_BYTES
    ):
        raise VerificationError(
            "environment hash chunk size must be an integer between 1 and "
            f"{MAX_ENVIRONMENT_HASH_CHUNK_BYTES} bytes"
        )
    if len(entries) > MAX_ENVIRONMENT_HASH_FILES:
        raise VerificationError(f"{label} exceeds the environment file-count budget")
    digest = hashlib.sha256()
    total_bytes = 0
    present_count = 0
    missing_count = 0
    for relative_name, path in sorted(entries, key=lambda item: item[0]):
        if not relative_name or "\0" in relative_name:
            raise VerificationError(f"{label} contains an invalid relative path")
        path_missing = path is None
        if path is not None:
            try:
                Path(path).lstat()
            except FileNotFoundError:
                path_missing = True
            except OSError as exc:
                raise VerificationError(
                    f"{label} entry cannot be stat'ed: {path}"
                ) from exc
        if path_missing:
            record: dict[str, Any] = {"path": relative_name, "missing": True}
            missing_count += 1
        else:
            assert path is not None
            binding = _hash_environment_file(path)
            total_bytes += int(binding["bytes"])
            present_count += 1
            record = {"path": relative_name, **binding}
        digest.update(
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        )
        digest.update(b"\n")
    if not entries:
        raise VerificationError(f"{label} does not expose any files to hash")
    return {
        "algorithm": "sha256-of-sorted-path-size-content-digests-v2",
        "n_files": len(entries),
        "n_present_files": present_count,
        "n_missing_files": missing_count,
        "total_present_bytes": total_bytes,
        "tree_sha256": digest.hexdigest(),
        "file_count_limit": MAX_ENVIRONMENT_HASH_FILES,
        "stream_chunk_bytes": ENVIRONMENT_HASH_CHUNK_BYTES,
    }


def _git_command(root: Path, arguments: Sequence[str]) -> bytes | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout if completed.returncode == 0 else None


def _source_tree_binding(root: Path) -> dict[str, Any]:
    """Bind live editable/local source content, including Git state when present."""
    root = Path(os.path.abspath(root))
    root_stat = root.lstat()
    if _is_link_or_reparse(root_stat) or not stat_module.S_ISDIR(root_stat.st_mode):
        raise VerificationError("editable/local source root must be a real directory")
    _ancestor_identity_snapshot(root / ".source-tree-probe")

    git_root_payload = _git_command(root, ("rev-parse", "--show-toplevel"))
    entries: list[tuple[str, Path | None]] = []
    git_record: dict[str, Any] | None = None
    if git_root_payload is not None:
        git_root = Path(os.fsdecode(git_root_payload).strip()).resolve()
        try:
            root.resolve().relative_to(git_root)
        except ValueError as exc:
            raise VerificationError(
                "Git source root is outside its reported worktree"
            ) from exc
        listed = _git_command(
            root,
            (
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
                "--full-name",
                "--",
                ".",
            ),
        )
        commit = _git_command(root, ("rev-parse", "HEAD"))
        status = _git_command(
            root,
            ("status", "--porcelain=v1", "-z", "--untracked-files=all", "--", "."),
        )
        if listed is None or commit is None or status is None:
            raise VerificationError("Git source state cannot be captured consistently")
        for raw_name in listed.split(b"\0"):
            if not raw_name:
                continue
            repository_relative = Path(os.fsdecode(raw_name))
            candidate = git_root / repository_relative
            try:
                source_relative = candidate.relative_to(root).as_posix()
            except ValueError as exc:
                raise VerificationError(
                    "Git listed a source path outside the local root"
                ) from exc
            entries.append((source_relative, candidate))
        git_record = {
            "vcs": "git",
            "commit": os.fsdecode(commit).strip(),
            "dirty": bool(status),
            "status_sha256": hashlib.sha256(status).hexdigest(),
        }
    else:
        for directory, directory_names, file_names in os.walk(
            root, topdown=True, followlinks=False
        ):
            current = Path(directory)
            retained_directories: list[str] = []
            for name in sorted(directory_names):
                if name in SOURCE_TREE_EXCLUDED_DIRECTORIES:
                    continue
                candidate = current / name
                candidate_stat = candidate.lstat()
                if _is_link_or_reparse(candidate_stat):
                    raise VerificationError(
                        f"editable/local source contains a link directory: {candidate}"
                    )
                retained_directories.append(name)
            directory_names[:] = retained_directories
            for name in sorted(file_names):
                candidate = current / name
                relative_name = candidate.relative_to(root).as_posix()
                entries.append((relative_name, candidate))

    content = _aggregate_environment_content(
        entries, label="editable/local source tree"
    )
    return {
        "source_location_sha256": hashlib.sha256(
            os.fsencode(str(root.resolve()))
        ).hexdigest(),
        "content": content,
        "vcs_state": git_record,
    }


def _direct_url_record(
    direct_url_text: str | None,
) -> tuple[dict[str, Any] | None, Path | None]:
    if direct_url_text is None:
        return None, None
    try:
        value = json.loads(direct_url_text)
    except json.JSONDecodeError as exc:
        raise VerificationError(
            "installed distribution has invalid direct_url.json"
        ) from exc
    if not isinstance(value, Mapping) or not isinstance(value.get("url"), str):
        raise VerificationError("installed distribution has malformed direct_url.json")
    directory_info = value.get("dir_info")
    vcs_info = value.get("vcs_info")
    archive_info = value.get("archive_info")
    editable = (
        isinstance(directory_info, Mapping) and directory_info.get("editable") is True
    )
    descriptor: dict[str, Any] = {
        "editable": editable,
        "kind": (
            "vcs"
            if isinstance(vcs_info, Mapping)
            else (
                "directory"
                if isinstance(directory_info, Mapping)
                else "archive" if isinstance(archive_info, Mapping) else "direct"
            )
        ),
        "vcs": vcs_info.get("vcs") if isinstance(vcs_info, Mapping) else None,
        "requested_revision": (
            vcs_info.get("requested_revision")
            if isinstance(vcs_info, Mapping)
            else None
        ),
        "commit_id": (
            vcs_info.get("commit_id") if isinstance(vcs_info, Mapping) else None
        ),
    }
    parsed = urlparse(value["url"])
    local_root: Path | None = None
    if parsed.scheme.lower() == "file":
        local_text = url2pathname(unquote(parsed.path))
        if parsed.netloc and parsed.netloc.lower() != "localhost":
            local_text = f"//{parsed.netloc}{local_text}"
        if (
            os.name == "nt"
            and len(local_text) >= 3
            and local_text[0] in ("/", "\\")
            and local_text[2] == ":"
        ):
            local_text = local_text[1:]
        candidate = Path(local_text)
        descriptor["local"] = True
        if candidate.is_dir():
            local_root = candidate
    else:
        descriptor["local"] = False
    return descriptor, local_root


def _distribution_content_binding(distribution: Any) -> dict[str, Any]:
    files = distribution.files
    if not files:
        raise VerificationError(
            f"distribution {distribution.metadata.get('Name')} exposes no file inventory"
        )
    entries = [
        (Path(str(item)).as_posix(), Path(distribution.locate_file(item)))
        for item in files
    ]
    return _aggregate_environment_content(
        entries,
        label=f"installed distribution {distribution.metadata.get('Name')}",
    )


def _installed_distribution_inventory() -> list[dict[str, Any]]:
    """Return a deterministic, exact installed-distribution inventory.

    Names, versions, installers, and direct-URL metadata are exact. Critical
    runtime distributions and every direct installation bind installed bytes;
    editable/local directories additionally bind the live source tree and Git
    commit/dirty state. URLs themselves are retained only through a digest so
    local paths or credentials are not copied into the manifest.
    """
    inventory: list[dict[str, Any]] = []
    for distribution in importlib_metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        name = str(raw_name or getattr(distribution, "name", "") or "<unknown>")
        normalized_name = _normalise_distribution_name(name)
        version = str(distribution.version)
        installer_text = distribution.read_text("INSTALLER")
        direct_url_text = distribution.read_text("direct_url.json")
        direct_url, local_source_root = _direct_url_record(direct_url_text)
        bind_installed_content = (
            normalized_name in CRITICAL_RUNTIME_DISTRIBUTIONS
            or direct_url_text is not None
        )
        record = {
            "name": name,
            "normalized_name": normalized_name,
            "version": version,
            "installer": installer_text.strip() if installer_text is not None else None,
            "direct_url_sha256": (
                hashlib.sha256(direct_url_text.encode("utf-8")).hexdigest()
                if direct_url_text is not None
                else None
            ),
            "direct_url": direct_url,
            "installed_content": (
                _distribution_content_binding(distribution)
                if bind_installed_content
                else None
            ),
            "live_source": (
                _source_tree_binding(local_source_root)
                if local_source_root is not None
                else None
            ),
        }
        if (
            direct_url is not None
            and direct_url.get("editable") is True
            and (record["live_source"] is None)
        ):
            raise VerificationError(
                f"editable distribution {name} has no hashable local source directory"
            )
        inventory.append(record)
    inventory.sort(
        key=lambda record: (
            record["normalized_name"],
            record["name"],
            record["version"],
            record["installer"] or "",
            record["direct_url_sha256"] or "",
            _canonical_json_sha256(record),
        )
    )
    return inventory


def _nvidia_driver_inventory() -> dict[str, Any]:
    """Query driver versions when ``nvidia-smi`` is available.

    Failure details are intentionally reduced to stable booleans.  Volatile
    localized stderr must not become part of a resumability fingerprint.
    """
    command = [
        "nvidia-smi",
        "--query-gpu=driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return {"nvidia_smi_available": False, "driver_versions": []}
    versions = sorted(
        {line.strip() for line in completed.stdout.splitlines() if line.strip()}
    )
    return {
        "nvidia_smi_available": completed.returncode == 0 and bool(versions),
        "driver_versions": versions if completed.returncode == 0 else [],
    }


def _gpu_runtime_inventory() -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    devices: list[dict[str, Any]] = []
    if cuda_available:
        for index in range(int(torch.cuda.device_count())):
            properties = torch.cuda.get_device_properties(index)
            device: dict[str, Any] = {
                "index": index,
                "name": str(properties.name),
                "total_memory_bytes": int(properties.total_memory),
                "compute_capability": [int(properties.major), int(properties.minor)],
                "multiprocessor_count": int(properties.multi_processor_count),
            }
            device_uuid = getattr(properties, "uuid", None)
            if device_uuid is not None:
                device["uuid"] = str(device_uuid)
            devices.append(device)
    return {
        "cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": len(devices),
        "devices": devices,
        **_nvidia_driver_inventory(),
    }


def _runtime_record() -> dict[str, Any]:
    packages = _installed_distribution_inventory()
    record: dict[str, Any] = {
        "schema": "DepPLM-training-runtime-environment-v3",
        "python": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "python_implementation": platform.python_implementation(),
        "python_cache_tag": sys.implementation.cache_tag,
        "byteorder": sys.byteorder,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "installed_distributions": packages,
        "installed_distributions_sha256": _canonical_json_sha256(packages),
        "gpu_runtime": _gpu_runtime_inventory(),
        "torch_build_config": torch.__config__.show(),
        "training_environment_variables": {
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG", ":4096:8"
            ),
            **{
                name: os.environ.get(name)
                for name in (
                    "CUDA_VISIBLE_DEVICES",
                    "MKL_NUM_THREADS",
                    "NVIDIA_TF32_OVERRIDE",
                    "OMP_NUM_THREADS",
                    "PYTHONHASHSEED",
                )
            },
        },
    }
    record["environment_sha256"] = _canonical_json_sha256(record)
    return record


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_file_bytes(
    path: Path,
    *,
    max_bytes: int,
) -> tuple[bytes, os.stat_result]:
    """Read one regular, non-link file handle and bind metadata to those bytes."""
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1:
        raise VerificationError("artifact read limit must be a positive integer")
    path = Path(os.path.abspath(path))
    ancestors_before = _ancestor_identity_snapshot(path)
    try:
        path_before = path.lstat()
    except OSError as exc:
        raise VerificationError(f"artifact cannot be stat'ed: {path}") from exc
    if _is_link_or_reparse(path_before):
        raise VerificationError(
            f"symbolic link or reparse artifact is forbidden: {path}"
        )
    if not stat_module.S_ISREG(path_before.st_mode):
        raise VerificationError(f"artifact is not a regular file: {path}")
    if int(path_before.st_size) > max_bytes:
        raise VerificationError(
            f"artifact exceeds the {max_bytes}-byte protocol limit: {path}"
        )
    try:
        with _open_readonly_nofollow(path) as handle:
            before = os.fstat(handle.fileno())
            if not stat_module.S_ISREG(before.st_mode):
                raise VerificationError(f"artifact is not a regular file: {path}")
            if _file_identity(path_before) != _file_identity(before):
                raise VerificationError(
                    f"artifact path changed before its handle was opened: {path}"
                )
            if int(before.st_size) > max_bytes:
                raise VerificationError(
                    f"artifact exceeds the {max_bytes}-byte protocol limit: {path}"
                )
            payload = bytearray()
            while True:
                chunk = handle.read(min(1024 * 1024, max_bytes + 1 - len(payload)))
                if not chunk:
                    break
                payload.extend(chunk)
                if len(payload) > max_bytes:
                    raise VerificationError(
                        f"artifact exceeds the {max_bytes}-byte protocol limit: {path}"
                    )
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise VerificationError(f"artifact cannot be read: {path}") from exc

    if _file_identity(before) != _file_identity(after) or len(payload) != int(
        after.st_size
    ):
        raise VerificationError(f"artifact changed while being read: {path}")
    try:
        path_after = path.lstat()
    except OSError as exc:
        raise VerificationError(
            f"artifact disappeared while being read: {path}"
        ) from exc
    if _is_link_or_reparse(path_after) or _file_identity(path_after) != _file_identity(
        after
    ):
        raise VerificationError(f"artifact path changed while being read: {path}")
    if _ancestor_identity_snapshot(path) != ancestors_before:
        raise VerificationError(f"artifact ancestor changed while being read: {path}")
    return bytes(payload), after


def _record_from_bytes(
    path: Path,
    payload: bytes,
    stat_result: os.stat_result,
    *,
    relative_to: Path,
) -> dict[str, Any]:
    try:
        display_path = path.relative_to(relative_to)
    except ValueError:
        display_path = path
    return {
        "path": str(display_path),
        "bytes": len(payload),
        "mtime_ns": int(stat_result.st_mtime_ns),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _read_stable_json(
    path: Path,
    *,
    relative_to: Path,
) -> tuple[Any, dict[str, Any]]:
    """Hash and parse the exact same immutable byte snapshot."""
    payload, stat_result = _read_file_bytes(
        path, max_bytes=MAX_JSON_ARTIFACT_BYTES
    )
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"JSON artifact cannot be parsed: {path}") from exc
    return parsed, _record_from_bytes(
        path, payload, stat_result, relative_to=relative_to
    )


def _canonical_training_namespace(
    *,
    variant: str,
    seed: int,
    device: str,
    mutation_chunksize: int,
) -> argparse.Namespace:
    return argparse.Namespace(
        command="train",
        variant=variant,
        seed=int(seed),
        device=device,
        mutation_chunksize=int(mutation_chunksize),
        **CANONICAL_TRAINING_SETTINGS,
    )


def _campaign_jobs(
    *,
    device: str,
    mutation_chunksize: int,
) -> list[argparse.Namespace]:
    return [
        _canonical_training_namespace(
            variant=variant.name,
            seed=seed,
            device=device,
            mutation_chunksize=mutation_chunksize,
        )
        for variant in VARIANTS
        for seed in CANONICAL_SEEDS
    ]


def _normalised_record_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("artifact record path must be a non-empty string")
    return value.replace("\\", "/")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validated_file_record_signature(
    record: Any,
    *,
    label: str,
    expected_fields: frozenset[str] = FILE_RECORD_FIELDS,
) -> tuple[str, tuple[int, str]]:
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    if set(record) != expected_fields:
        raise ValueError(f"{label} fields do not match the canonical schema")
    path = _normalised_record_path(record.get("path"))
    size = record.get("bytes")
    mtime_ns = record.get("mtime_ns")
    digest = record.get("sha256")
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError(f"{label} byte count must be a positive integer")
    if (
        not isinstance(mtime_ns, int)
        or isinstance(mtime_ns, bool)
        or mtime_ns < 0
    ):
        raise ValueError(f"{label} mtime_ns must be a non-negative integer")
    if not _is_sha256(digest):
        raise ValueError(f"{label} SHA-256 is malformed")
    return path, (size, digest)


def _code_signature(records: Sequence[Mapping[str, Any]]) -> dict[str, tuple[int, str]]:
    signature: dict[str, tuple[int, str]] = {}
    for record in records:
        path, value = _validated_file_record_signature(
            record,
            label="bound artifact record",
        )
        if path in signature:
            raise ValueError(f"duplicate bound artifact path: {path}")
        signature[path] = value
    return signature


def _strict_json_equal(observed: Any, expected: Any) -> bool:
    """Compare JSON-shaped values without Python's bool/int coercion."""
    if isinstance(expected, Mapping):
        return (
            isinstance(observed, Mapping)
            and set(observed) == set(expected)
            and all(
                _strict_json_equal(observed[key], expected[key]) for key in expected
            )
        )
    if isinstance(expected, list):
        return (
            isinstance(observed, list)
            and len(observed) == len(expected)
            and all(
                _strict_json_equal(left, right)
                for left, right in zip(observed, expected)
            )
        )
    if expected is None:
        return observed is None
    if isinstance(expected, bool):
        return observed is expected
    if isinstance(expected, int):
        return (
            isinstance(observed, int)
            and not isinstance(observed, bool)
            and observed == expected
        )
    if isinstance(expected, float):
        return isinstance(observed, float) and observed == expected
    if isinstance(expected, str):
        return isinstance(observed, str) and observed == expected
    return type(observed) is type(expected) and observed == expected


def _training_execution_signature_from_source(source: str) -> dict[str, Any]:
    """Hash only runner definitions that can affect training outputs.

    Full-file hashes of imported training dependencies remain mandatory.  The
    local runner is represented by a canonical AST subset so changes confined
    to verification/reporting code do not invalidate completed model fits.
    """
    tree = ast.parse(source)
    functions: dict[str, ast.AST] = {}
    classes: dict[str, ast.AST] = {}
    constants: dict[str, ast.AST] = {}
    imports: list[ast.AST] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = node
        elif isinstance(node, ast.ClassDef):
            classes[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = node
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)

    missing_functions = set(TRAINING_EXECUTION_COMPONENTS).difference(functions)
    missing_constants = set(TRAINING_EXECUTION_CONSTANTS).difference(constants)
    if missing_functions or missing_constants:
        raise RuntimeError(
            "training signature source is missing declared components: "
            f"functions={sorted(missing_functions)}, "
            f"constants={sorted(missing_constants)}"
        )
    selected_functions = set(TRAINING_EXECUTION_COMPONENTS)
    selected_classes: set[str] = set()
    selected_constants = set(TRAINING_EXECUTION_CONSTANTS)
    while True:
        selected_nodes = [functions[name] for name in selected_functions]
        selected_nodes.extend(classes[name] for name in selected_classes)
        selected_nodes.extend(constants[name] for name in selected_constants)
        referenced_names = {
            child.id
            for node in selected_nodes
            for child in ast.walk(node)
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
        }
        added_functions = referenced_names.intersection(functions).difference(
            selected_functions
        )
        added_classes = referenced_names.intersection(classes).difference(
            selected_classes
        )
        added_constants = referenced_names.intersection(constants).difference(
            selected_constants
        )
        if not (added_functions or added_classes or added_constants):
            break
        selected_functions.update(added_functions)
        selected_classes.update(added_classes)
        selected_constants.update(added_constants)

    function_order = list(TRAINING_EXECUTION_COMPONENTS) + sorted(
        selected_functions.difference(TRAINING_EXECUTION_COMPONENTS)
    )
    constant_order = list(TRAINING_EXECUTION_CONSTANTS) + sorted(
        selected_constants.difference(TRAINING_EXECUTION_CONSTANTS)
    )
    class_order = sorted(selected_classes)
    selected = [functions[name] for name in function_order]
    selected.extend(classes[name] for name in class_order)
    selected.extend(constants[name] for name in constant_order)
    import_bindings: list[str] = []
    for node in imports:
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                if bound in referenced_names:
                    import_bindings.append(f"import:{alias.name}:{alias.asname or ''}")
        else:
            module = node.module or ""
            for alias in node.names:
                bound = alias.asname or alias.name
                if bound in referenced_names:
                    import_bindings.append(
                        f"from:{module}:{node.level}:{alias.name}:{alias.asname or ''}"
                    )
    payload = {
        "components": list(TRAINING_EXECUTION_COMPONENTS),
        "constants": list(TRAINING_EXECUTION_CONSTANTS),
        "transitive_functions": function_order,
        "transitive_classes": class_order,
        "transitive_constants": constant_order,
        "imports": sorted(import_bindings),
        "ast": [ast.dump(node, include_attributes=False) for node in selected],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return {
        "schema": TRAINING_EXECUTION_SIGNATURE_SCHEMA,
        "components": list(TRAINING_EXECUTION_COMPONENTS),
        "constants": list(TRAINING_EXECUTION_CONSTANTS),
        "transitive_functions": function_order,
        "transitive_classes": class_order,
        "transitive_constants": constant_order,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


@functools.lru_cache(maxsize=1)
def _training_execution_signature() -> dict[str, Any]:
    payload, _ = _read_file_bytes(VERIFIER_CODE_PATH, max_bytes=MAX_SOURCE_BYTES)
    try:
        source = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VerificationError("training verifier source is not UTF-8") from exc
    return _training_execution_signature_from_source(source)


def _verify_current_bindings(
    recorded: Any,
    *,
    current_records: Sequence[Mapping[str, Any]],
    label: str,
) -> None:
    if not isinstance(recorded, list):
        raise VerificationError(f"{label} bindings are missing")
    try:
        recorded_signature = _code_signature(recorded)
        current_signature = _code_signature(current_records)
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"{label} bindings are malformed: {exc}") from exc
    if recorded_signature != current_signature:
        raise VerificationError(f"{label} bindings differ from current files")


def _resolve_bound_artifact(record: Mapping[str, Any], *, artifact_root: Path) -> Path:
    if not isinstance(record, Mapping):
        raise VerificationError("artifact record must be a JSON object")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise VerificationError("artifact path is missing")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise VerificationError(
            "artifact path must be relative to and contained in the output root"
        )
    root = artifact_root.resolve()
    path = root.joinpath(relative)
    current = root
    try:
        for part in relative.parts:
            if part in ("", "."):
                continue
            current = current / part
            current_stat = current.lstat()
            if _is_link_or_reparse(current_stat):
                raise VerificationError(
                    f"symbolic link or reparse artifact is forbidden: {raw_path}"
                )
    except OSError as exc:
        raise VerificationError(f"artifact is missing: {raw_path}") from exc
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise VerificationError("artifact path escapes the output root") from exc
    return resolved


def _read_verified_artifact_bytes(
    record: Mapping[str, Any],
    *,
    artifact_root: Path,
) -> tuple[Path, bytes, dict[str, Any]]:
    expected_size = record.get("bytes")
    expected_digest = record.get("sha256")
    if (
        not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size < 1
    ):
        raise VerificationError("artifact byte count is invalid")
    if not _is_sha256(expected_digest):
        raise VerificationError("artifact SHA-256 is invalid")
    path = _resolve_bound_artifact(record, artifact_root=artifact_root)
    if expected_size > MAX_BOUND_ARTIFACT_BYTES:
        raise VerificationError("artifact exceeds the protocol byte limit")
    payload, stat_result = _read_file_bytes(path, max_bytes=expected_size)
    raw_path = str(record["path"])
    if len(payload) != expected_size:
        raise VerificationError(f"artifact byte count changed: {raw_path}")
    if hashlib.sha256(payload).hexdigest() != expected_digest:
        raise VerificationError(f"artifact SHA-256 changed: {raw_path}")
    binding = _record_from_bytes(
        path, payload, stat_result, relative_to=artifact_root.resolve()
    )
    return path, payload, binding


def _verified_artifact_binding(
    record: Mapping[str, Any],
    *,
    artifact_root: Path,
) -> dict[str, Any]:
    expected_size = record.get("bytes")
    expected_digest = record.get("sha256")
    if (
        not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size < 1
        or expected_size > MAX_BOUND_ARTIFACT_BYTES
    ):
        raise VerificationError("artifact byte count is invalid")
    if not _is_sha256(expected_digest):
        raise VerificationError("artifact SHA-256 is invalid")
    path = _resolve_bound_artifact(record, artifact_root=artifact_root)
    observed = _stable_file_record(
        path,
        relative_to=artifact_root.resolve(),
        max_bytes=expected_size,
    )
    raw_path = str(record.get("path", ""))
    if observed["bytes"] != expected_size:
        raise VerificationError(f"artifact byte count changed: {raw_path}")
    if observed["sha256"] != expected_digest:
        raise VerificationError(f"artifact SHA-256 changed: {raw_path}")
    return observed


def _verify_artifact_record(
    record: Mapping[str, Any],
    *,
    artifact_root: Path,
) -> tuple[bool, str]:
    try:
        _verified_artifact_binding(record, artifact_root=artifact_root)
    except (TypeError, ValueError, VerificationError) as exc:
        return False, str(exc)
    return True, "verified"


def _read_verified_json_artifact(
    record: Mapping[str, Any],
    *,
    artifact_root: Path,
    label: str,
) -> tuple[Any, dict[str, Any]]:
    path, payload, binding = _read_verified_artifact_bytes(
        record, artifact_root=artifact_root
    )
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"{label} is not valid JSON: {path}") from exc
    return parsed, binding


def _verification_base_input_paths() -> tuple[Path, ...]:
    return (
        PROCESSED / "gene_effect.npy",
        PROCESSED / "ppi_graph.npz",
        PROCESSED / "genes.txt",
        PROCESSED / "cell_lines.txt",
        PROCESSED / "lineages.txt",
        DEPMAP / "OmicsSomaticMutations.csv",
    )


def _snapshot_canonical_inputs(
    paths: Sequence[Path],
) -> tuple[list[dict[str, Any]], dict[Path, Any]]:
    """Hash each source once and retain anonymous snapshots used by replay."""
    capture = {path.resolve() for path in _verification_base_input_paths()}
    snapshots: dict[Path, Any] = {}
    records: list[dict[str, Any]] = []
    seen: set[Path] = set()
    try:
        for raw_path in paths:
            path = Path(raw_path)
            key = path.resolve()
            if key in seen:
                raise VerificationError(f"duplicate canonical input path: {path}")
            seen.add(key)
            snapshot = None
            if key in capture:
                snapshot = tempfile.TemporaryFile(mode="w+b")
                snapshots[key] = snapshot
            records.append(
                _stable_file_record(
                    path,
                    relative_to=ROOT,
                    snapshot_handle=snapshot,
                )
            )
    except Exception as exc:
        close_errors = _close_input_snapshots(snapshots)
        if close_errors and hasattr(exc, "add_note"):
            exc.add_note("Snapshot cleanup also failed: " + "; ".join(close_errors))
        raise
    return records, snapshots


def _close_input_snapshots(snapshots: Mapping[Path, Any]) -> list[str]:
    """Attempt every close and return deterministic errors without masking callers."""
    errors: list[str] = []
    for path, handle in snapshots.items():
        try:
            handle.close()
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    return errors


def _bound_input_handle(
    snapshots: Mapping[Path, Any],
    path: Path,
    *,
    max_bytes: int = MAX_BOUND_ARTIFACT_BYTES,
) -> Any:
    handle = snapshots.get(path.resolve())
    if handle is None or handle.closed:
        raise VerificationError(f"bound input snapshot is missing: {path}")
    observed_size = int(os.fstat(handle.fileno()).st_size)
    if observed_size > max_bytes:
        raise VerificationError(
            f"bound input exceeds the {max_bytes}-byte protocol limit: {path}"
        )
    handle.seek(0)
    return handle


def _read_bound_lines(snapshots: Mapping[Path, Any], path: Path) -> list[str]:
    handle = _bound_input_handle(snapshots, path, max_bytes=MAX_IDENTITY_BYTES)
    try:
        text = handle.read().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VerificationError(f"bound text input is not UTF-8: {path}") from exc
    return [line.strip() for line in text.splitlines()]


def _load_bound_mutation_inputs(
    snapshots: Mapping[Path, Any],
    cell_ids: Sequence[str],
) -> tuple[list[set[str]], list[dict[str, TokenAnnotation]]]:
    """Parse mutation identities and token annotations from one bound snapshot."""
    ordered_cells = [str(value) for value in cell_ids]
    wanted = set(ordered_cells)
    required = [
        "ModelID",
        "HugoSymbol",
        "VepImpact",
        "Hotspot",
        "LikelyLoF",
        "TranscriptLikelyLof",
        "OncogeneHighImpact",
        "TumorSuppressorHighImpact",
    ]
    aggregate: dict[tuple[str, str], TokenAnnotation] = {}
    mutation_sets = {cell_id: set() for cell_id in ordered_cells}
    handle = _bound_input_handle(snapshots, DEPMAP / "OmicsSomaticMutations.csv")

    def truthy(values: pd.Series) -> pd.Series:
        normalized = values.astype("string").str.strip().str.lower()
        return normalized.isin({"1", "1.0", "true", "t", "yes", "y"})

    try:
        chunks = pd.read_csv(
            handle,
            usecols=required,
            dtype=str,
            chunksize=250_000,
            low_memory=False,
        )
        for chunk in chunks:
            frame = chunk.loc[chunk["ModelID"].isin(wanted)].copy()
            frame = frame.dropna(subset=["ModelID", "HugoSymbol"])
            if frame.empty:
                continue
            for model_id, symbol in zip(frame["ModelID"], frame["HugoSymbol"]):
                if symbol:
                    mutation_sets[str(model_id)].add(str(symbol))
            frame["impact_rank"] = (
                frame["VepImpact"]
                .astype("string")
                .str.upper()
                .map(IMPACT_RANK)
                .fillna(0)
            )
            frame["hotspot"] = truthy(frame["Hotspot"])
            frame["likely_lof"] = truthy(frame["LikelyLoF"]) | truthy(
                frame["TranscriptLikelyLof"]
            )
            frame["oncogene"] = truthy(frame["OncogeneHighImpact"])
            frame["tumour_suppressor"] = truthy(frame["TumorSuppressorHighImpact"])
            grouped = frame.groupby(
                ["ModelID", "HugoSymbol"], sort=False, observed=True
            )[
                [
                    "impact_rank",
                    "hotspot",
                    "likely_lof",
                    "oncogene",
                    "tumour_suppressor",
                ]
            ].max()
            for row in grouped.itertuples():
                key = (str(row.Index[0]), str(row.Index[1]))
                current = TokenAnnotation(
                    impact_rank=int(row.impact_rank),
                    hotspot=bool(row.hotspot),
                    likely_lof=bool(row.likely_lof),
                    oncogene_high_impact=bool(row.oncogene),
                    tumour_suppressor_high_impact=bool(row.tumour_suppressor),
                )
                aggregate[key] = aggregate.get(key, TokenAnnotation()).merge(current)
    except (KeyError, OSError, UnicodeError, ValueError) as exc:
        raise VerificationError(
            "bound mutation input cannot be parsed with the protocol schema"
        ) from exc

    annotations = {cell_id: {} for cell_id in ordered_cells}
    for (cell_id, gene), annotation in aggregate.items():
        annotations[cell_id][gene] = annotation
    mutations = [mutation_sets[cell_id] for cell_id in ordered_cells]
    annotation_rows = [annotations[cell_id] for cell_id in ordered_cells]
    if [set(item) for item in annotation_rows] != mutations:
        raise VerificationError(
            "annotation-derived mutation identities do not match source mutations"
        )
    return mutations, annotation_rows


def _load_verification_base(
    snapshots: Mapping[Path, Any],
) -> VerificationBase:
    genes = tuple(_read_bound_lines(snapshots, PROCESSED / "genes.txt"))
    cell_ids = tuple(_read_bound_lines(snapshots, PROCESSED / "cell_lines.txt"))
    lineages = np.asarray(
        _read_bound_lines(snapshots, PROCESSED / "lineages.txt"), dtype=str
    )
    gene_effect_handle = _bound_input_handle(
        snapshots,
        PROCESSED / "gene_effect.npy",
        max_bytes=MAX_GENE_EFFECT_BYTES,
    )
    try:
        gene_effect = np.array(
            np.load(gene_effect_handle, allow_pickle=False),
            dtype=np.float32,
            copy=True,
        )
    except (OSError, ValueError) as exc:
        raise VerificationError("bound gene_effect.npy cannot be parsed") from exc
    if gene_effect.shape != (len(genes), len(cell_ids)):
        raise VerificationError(
            "gene_effect.npy does not align with genes.txt and cell_lines.txt"
        )
    if lineages.shape != (len(cell_ids),):
        raise VerificationError("lineages.txt does not align with cell_lines.txt")
    mutations, annotations = _load_bound_mutation_inputs(snapshots, cell_ids)
    graph_handle = _bound_input_handle(snapshots, PROCESSED / "ppi_graph.npz")
    try:
        with np.load(graph_handle, allow_pickle=False) as graph:
            full_edge_index = np.array(graph["edge_index"], dtype=np.int64, copy=True)
    except (KeyError, OSError, ValueError) as exc:
        raise VerificationError("bound ppi_graph.npz cannot be parsed") from exc
    if full_edge_index.ndim != 2 or full_edge_index.shape[1] != 2:
        raise VerificationError("ppi_graph.npz edge_index must have shape (E, 2)")
    try:
        edge_index = corrected._subgraph(
            full_edge_index,
            np.arange(len(genes), dtype=np.int64),
            len(genes),
        )
    except (IndexError, ValueError) as exc:
        raise VerificationError(
            "bound ppi_graph.npz cannot be transformed by the training protocol"
        ) from exc
    return VerificationBase(
        gene_effect=gene_effect,
        genes=genes,
        cell_ids=cell_ids,
        lineages=lineages,
        mutations_by_cell=tuple(frozenset(item) for item in mutations),
        token_annotations=tuple(dict(item) for item in annotations),
        edge_index=edge_index,
        source_cell_indices=np.arange(len(cell_ids), dtype=np.int64),
        source_gene_indices=np.arange(len(genes), dtype=np.int64),
    )


def _expected_campaign_identities() -> set[tuple[str, int, str]]:
    return {
        (variant.name, seed, f"fold-{fold:02d}")
        for variant in VARIANTS
        for seed in CANONICAL_SEEDS
        for fold in range(CANONICAL_FOLDS)
    }


def _index_campaign_artifacts(
    records: Any,
    *,
    label: str,
    expected_fields: frozenset[str],
) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    expected = _expected_campaign_identities()
    expected_order = [
        (variant.name, seed, f"fold-{fold:02d}")
        for variant in VARIANTS
        for seed in CANONICAL_SEEDS
        for fold in range(CANONICAL_FOLDS)
    ]
    if not isinstance(records, list) or len(records) != len(expected):
        raise VerificationError(f"{label} must contain exactly {len(expected)} records")
    indexed: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    observed_order: list[tuple[str, int, str]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise VerificationError(f"{label} contains a non-object record")
        try:
            _validated_file_record_signature(
                record,
                label=f"{label} record",
                expected_fields=expected_fields,
            )
        except ValueError as exc:
            raise VerificationError(f"{label} record is malformed: {exc}") from exc
        variant = record.get("variant")
        seed = record.get("seed")
        fold = record.get("fold")
        if (
            not isinstance(variant, str)
            or not isinstance(seed, int)
            or isinstance(seed, bool)
            or not isinstance(fold, str)
        ):
            raise VerificationError(f"{label} identity is malformed")
        identity = (variant, seed, fold)
        if identity in indexed:
            raise VerificationError(f"{label} contains duplicate identity {identity}")
        indexed[identity] = record
        observed_order.append(identity)
    missing = expected.difference(indexed)
    unexpected = set(indexed).difference(expected)
    if missing or unexpected:
        raise VerificationError(
            f"{label} identity coverage differs from the fixed campaign: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    if observed_order != expected_order:
        raise VerificationError(f"{label} are not in canonical campaign order")
    return indexed


def _recompute_outer_test_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    train_gene_mean: np.ndarray,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        **corrected.dependency_metrics(prediction, target, train_gene_mean),
        "n_cells": int(prediction.shape[0]),
        "reference_scope": "outer_train_gene_mean",
        "gene_mean_reference_sha256": corrected._hash_array(train_gene_mean),
        "raw_cellwise_correlation_role": "secondary_descriptive_only",
    }
    spearman: list[float] = []
    for predicted, observed in zip(prediction, target):
        finite = np.isfinite(predicted) & np.isfinite(observed)
        if int(finite.sum()) < 10:
            spearman.append(float("nan"))
        else:
            spearman.append(
                corrected._safe_corr(
                    rankdata(predicted[finite]), rankdata(observed[finite])
                )
            )
    finite_spearman = np.asarray(spearman, dtype=np.float64)
    finite_spearman = finite_spearman[np.isfinite(finite_spearman)]
    metrics["cellwise_spearman_mean"] = (
        float(np.mean(finite_spearman)) if finite_spearman.size else float("nan")
    )
    metrics["cellwise_spearman_sd"] = (
        float(np.std(finite_spearman)) if finite_spearman.size else float("nan")
    )
    metrics["n_finite_cellwise_spearman"] = int(finite_spearman.size)
    return metrics


def _json_metric_value(value: Any) -> Any:
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _assert_metrics_match(recorded: Any, recomputed: Mapping[str, Any]) -> None:
    if not isinstance(recorded, Mapping):
        raise VerificationError("fold test_metrics are missing")
    if set(recorded) != set(recomputed):
        raise VerificationError(
            "recorded test metric fields differ from independently recomputed fields"
        )
    for name, expected_raw in recomputed.items():
        expected = _json_metric_value(expected_raw)
        observed = recorded[name]
        if expected is None:
            if observed is not None:
                raise VerificationError(f"test metric {name} should be null")
        elif isinstance(expected, float):
            if (
                not isinstance(observed, (int, float))
                or isinstance(observed, bool)
                or not np.isfinite(float(observed))
                or not np.isclose(float(observed), expected, rtol=1e-12, atol=1e-12)
            ):
                raise VerificationError(
                    f"test metric {name} differs from prediction replay"
                )
        elif isinstance(expected, int):
            if (
                not isinstance(observed, int)
                or isinstance(observed, bool)
                or observed != expected
            ):
                raise VerificationError(
                    f"test metric {name} differs from prediction replay"
                )
        elif observed != expected:
            raise VerificationError(
                f"test metric {name} differs from prediction replay"
            )


def _verify_split_manifest(
    split: Any,
    *,
    fold_name: str,
    fold_seed: int,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    base: VerificationBase,
) -> None:
    if not isinstance(split, Mapping):
        raise VerificationError("outer-fold split manifest is not an object")
    if (
        split.get("fold") != fold_name
        or split.get("protocol") != "p1"
        or split.get("seed") != fold_seed
    ):
        raise VerificationError("split manifest fold identity or protocol changed")
    expected_inner_train, expected_inner_validation = corrected.make_inner_split(
        outer_train,
        base.lineages,
        val_fraction=CANONICAL_TRAINING_SETTINGS["inner_val_fraction"],
        seed=derive_seed(fold_seed, "inner-split"),
    )
    expected_splits = (
        ("outer_train", outer_train),
        ("inner_train", expected_inner_train),
        ("inner_validation", expected_inner_validation),
        ("outer_test", outer_test),
    )
    for name, expected in expected_splits:
        record = split.get(name)
        if not isinstance(record, Mapping):
            raise VerificationError(f"split manifest is missing {name}")
        indices = record.get("indices")
        cell_ids = record.get("cell_ids")
        if not _strict_json_equal(indices, [int(value) for value in expected]):
            raise VerificationError(f"split manifest {name} indices changed")
        if not _strict_json_equal(
            cell_ids, [base.cell_ids[int(value)] for value in expected]
        ):
            raise VerificationError(f"split manifest {name} cell identities changed")
        if record.get("sha256") != _hash_indices(expected):
            raise VerificationError(f"split manifest {name} hash changed")
    source = split.get("source_cell_indices")
    if not isinstance(source, Mapping):
        raise VerificationError("split manifest source-cell identities are missing")
    for name, expected in expected_splits:
        source_expected = [
            int(base.source_cell_indices[int(value)]) for value in expected
        ]
        if not _strict_json_equal(source.get(name), source_expected):
            raise VerificationError(f"split manifest {name} source indices changed")


def _verify_prediction_artifact(
    record: Mapping[str, Any],
    *,
    artifact_root: Path,
    base: VerificationBase,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    if record.get("schema") != "DepPLM-outer-test-predictions-v1":
        raise VerificationError("prediction artifact schema is not recognised")
    path, payload, binding = _read_verified_artifact_bytes(
        record, artifact_root=artifact_root
    )
    expected_names = {
        "prediction",
        "train_gene_mean",
        "outer_test_indices",
        "source_cell_indices",
        "source_gene_indices",
        "cell_ids",
        "gene_ids",
    }
    metadata = record.get("arrays")
    if not isinstance(metadata, Mapping) or set(metadata) != expected_names:
        raise VerificationError("prediction artifact array contract is incomplete")
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as artifact:
            if (
                len(artifact.files) != len(expected_names)
                or set(artifact.files) != expected_names
            ):
                raise VerificationError(
                    "prediction archive members differ from the recorded contract"
                )
            arrays = {name: np.asarray(artifact[name]) for name in expected_names}
    except (OSError, ValueError) as exc:
        raise VerificationError(f"prediction archive cannot be read: {path}") from exc

    for name, array in arrays.items():
        observed = {"dtype": str(array.dtype), "shape": list(array.shape)}
        if not _strict_json_equal(metadata.get(name), observed):
            raise VerificationError(f"prediction array metadata changed for {name}")
    n_test = int(outer_test.size)
    n_genes = len(base.genes)
    expected_shapes = {
        "prediction": (n_test, n_genes),
        "train_gene_mean": (n_genes,),
        "outer_test_indices": (n_test,),
        "source_cell_indices": (n_test,),
        "source_gene_indices": (n_genes,),
        "cell_ids": (n_test,),
        "gene_ids": (n_genes,),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise VerificationError(f"prediction array shape changed for {name}")
    if arrays["prediction"].dtype != np.dtype("float32"):
        raise VerificationError("prediction dtype must be float32")
    if arrays["train_gene_mean"].dtype != np.dtype("float32"):
        raise VerificationError("train_gene_mean dtype must be float32")
    for name in ("outer_test_indices", "source_cell_indices", "source_gene_indices"):
        if arrays[name].dtype != np.dtype("int64"):
            raise VerificationError(f"{name} dtype must be int64")
    if arrays["cell_ids"].dtype.kind != "U" or arrays["gene_ids"].dtype.kind != "U":
        raise VerificationError(
            "prediction identities must be non-object Unicode arrays"
        )
    if not np.array_equal(arrays["outer_test_indices"], outer_test):
        raise VerificationError("saved outer-test indices differ from the fixed split")
    if np.unique(arrays["outer_test_indices"]).size != n_test:
        raise VerificationError("saved outer-test indices contain duplicates")
    if not np.array_equal(
        arrays["source_cell_indices"], base.source_cell_indices[outer_test]
    ):
        raise VerificationError("saved source-cell indices changed")
    if not np.array_equal(arrays["source_gene_indices"], base.source_gene_indices):
        raise VerificationError("saved source-gene indices changed")
    if not np.array_equal(
        arrays["cell_ids"], np.asarray([base.cell_ids[int(i)] for i in outer_test])
    ):
        raise VerificationError("saved cell identities changed")
    if not np.array_equal(arrays["gene_ids"], np.asarray(base.genes)):
        raise VerificationError("saved gene identities changed")
    prediction = np.asarray(arrays["prediction"], dtype=np.float32)
    if not np.isfinite(prediction).all():
        raise VerificationError("saved predictions contain non-finite values")
    expected_mean = corrected.fit_gene_mean_baseline(base.gene_effect, outer_train)
    train_mean = np.asarray(arrays["train_gene_mean"], dtype=np.float32)
    if not np.isfinite(train_mean).all():
        raise VerificationError("saved train_gene_mean contains non-finite values")
    if not np.allclose(train_mean, expected_mean, rtol=0.0, atol=0.0, equal_nan=True):
        raise VerificationError(
            "saved train_gene_mean differs from the fixed outer-training cells"
        )
    target = np.asarray(base.gene_effect[:, outer_test].T, dtype=np.float32)
    metrics = _recompute_outer_test_metrics(prediction, target, train_mean)
    per_gene = _per_gene_prediction_summary_from_arrays(
        predictions=prediction,
        test_indices=np.asarray(arrays["outer_test_indices"], dtype=np.int64),
        train_mean=train_mean,
        gene_ids=np.asarray(arrays["gene_ids"], dtype=str),
        base=base,
    )
    return metrics, binding, per_gene


def _required_sha256(record: Mapping[str, Any], field: str, *, label: str) -> str:
    value = record.get(field)
    if not _is_sha256(value):
        raise VerificationError(f"{label} {field} is not a canonical SHA-256")
    return str(value)


def _verify_preprocessing_fit(
    record: Any,
    *,
    expected_fit_indices: np.ndarray,
    base: VerificationBase,
    variant: SensitivityVariant,
    label: str,
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise VerificationError(f"{label} preprocessing manifest is missing")
    expected_fit = np.asarray(expected_fit_indices, dtype="<i8")
    expected_identity = {
        "fit_indices_sha256": corrected._hash_array(expected_fit),
        "n_fit_cells": int(expected_fit.size),
    }
    for field, expected in expected_identity.items():
        if not _strict_json_equal(record.get(field), expected):
            raise VerificationError(f"{label} preprocessing {field} changed")

    mutation = record.get("mutation")
    graph = record.get("graph")
    sensitivity = record.get("k_token_pmi_sensitivity")
    if not isinstance(mutation, Mapping):
        raise VerificationError(f"{label} mutation preprocessing is missing")
    if not isinstance(graph, Mapping):
        raise VerificationError(f"{label} graph preprocessing is missing")
    if not isinstance(sensitivity, Mapping):
        raise VerificationError(
            f"{label} K/token/PMI sensitivity preprocessing is missing"
        )
    features = fit_sensitivity_mutation_features(
        base.mutations_by_cell,
        base.token_annotations,
        base.genes,
        expected_fit,
        top_n=CANONICAL_TRAINING_SETTINGS["top_mut"],
        effective_k=variant.k_tokens,
        token_strategy=variant.token_strategy,
    )
    pmi_fit = fit_pmi_sensitivity(
        base.edge_index.T.astype(np.int64, copy=False),
        base.genes,
        features.matrix,
        features.vocabulary,
        expected_fit,
        mode=variant.pmi_mode,
    )
    expected_sensitivity = feature_manifest(features, pmi_fit)
    expected_mutation = {
        "vocabulary": list(features.vocabulary),
        "vocabulary_sha256": expected_sensitivity["vocabulary_sha256"],
        "token_order_sha256": corrected._hash_strings(features.token_order),
        "n_vocabulary": len(features.vocabulary),
        "n_token_genes": len(features.token_order),
        "k_tokens": MAX_TOKEN_SLOTS,
        "allocated_token_slots": MAX_TOKEN_SLOTS,
        "effective_k": int(variant.k_tokens),
        "token_strategy": variant.token_strategy,
    }
    if not _strict_json_equal(mutation, expected_mutation):
        raise VerificationError(
            f"{label} mutation preprocessing differs from an independent "
            f"fit for variant {variant.name}"
        )
    if not _strict_json_equal(sensitivity, expected_sensitivity):
        raise VerificationError(
            f"{label} sensitivity preprocessing differs from an independent "
            f"fit for variant {variant.name}"
        )
    if mutation["vocabulary_sha256"] != sensitivity["vocabulary_sha256"]:
        raise VerificationError(
            f"{label} mutation and sensitivity vocabulary hashes disagree"
        )
    expected_graph = {
        "n_edges": int(pmi_fit.weights.size),
        "edge_weight_sha256": expected_sensitivity["pmi"]["weights_sha256"],
        "edge_weight_mean": (
            float(torch.from_numpy(pmi_fit.weights).float().mean().item())
            if pmi_fit.weights.size
            else None
        ),
        "n_non_neutral_weights": int((~np.isclose(pmi_fit.weights, 1.0)).sum()),
        "pmi_mode": variant.pmi_mode,
    }
    if not _strict_json_equal(graph, expected_graph):
        raise VerificationError(
            f"{label} graph preprocessing differs from an independent PMI fit"
        )
    hashes = {
        field: str(expected_sensitivity[field])
        for field in (
            "vocabulary_sha256",
            "matrix_sha256",
            "tokens_sha256",
            "token_mask_sha256",
        )
    }
    for field in ("weights_sha256", "valid_mask_sha256", "raw_pmi_sha256"):
        hashes[f"pmi_{field}"] = str(expected_sensitivity["pmi"][field])
    return {
        "fit_indices_sha256": expected_identity["fit_indices_sha256"],
        "effective_k": int(variant.k_tokens),
        "allocated_token_slots": MAX_TOKEN_SLOTS,
        "token_strategy": variant.token_strategy,
        "pmi_mode": variant.pmi_mode,
        **hashes,
    }


def _verify_fold_preprocessing(
    result: Mapping[str, Any],
    *,
    outer_train: np.ndarray,
    fold_seed: int,
    base: VerificationBase,
    variant: SensitivityVariant,
) -> dict[str, Any]:
    preprocessing = result.get("preprocessing")
    if not isinstance(preprocessing, Mapping) or set(preprocessing) != {
        "inner_train_fit",
        "outer_train_refit",
    }:
        raise VerificationError(
            "fold preprocessing must contain exact inner-train and outer-train records"
        )
    inner_train, _ = corrected.make_inner_split(
        outer_train,
        base.lineages,
        val_fraction=CANONICAL_TRAINING_SETTINGS["inner_val_fraction"],
        seed=derive_seed(fold_seed, "inner-split"),
    )
    return {
        "inner_train_fit": _verify_preprocessing_fit(
            preprocessing["inner_train_fit"],
            expected_fit_indices=inner_train,
            base=base,
            variant=variant,
            label="inner-train",
        ),
        "outer_train_refit": _verify_preprocessing_fit(
            preprocessing["outer_train_refit"],
            expected_fit_indices=outer_train,
            base=base,
            variant=variant,
            label="outer-train",
        ),
    }


def _verify_per_gene_sidecar(
    record: Mapping[str, Any],
    *,
    expected: pd.DataFrame,
    artifact_root: Path,
) -> dict[str, Any]:
    sidecar_path, payload, binding = _read_verified_artifact_bytes(
        record, artifact_root=artifact_root
    )
    try:
        frame = pd.read_csv(io.BytesIO(payload), compression="gzip")
    except (OSError, EOFError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
        raise VerificationError(
            f"per-gene sidecar is not a readable gzip CSV: {sidecar_path}"
        ) from exc
    if list(frame.columns) != list(PER_GENE_COLUMNS):
        raise VerificationError("per-gene sidecar schema or column order changed")
    if len(frame) != len(expected):
        raise VerificationError("per-gene sidecar row count changed")

    try:
        observed_indices = pd.to_numeric(frame["gene_index"], errors="raise").to_numpy(
            dtype=np.float64
        )
        observed_counts = pd.to_numeric(
            frame["n_outer_test_observations"], errors="raise"
        ).to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise VerificationError(
            "per-gene sidecar integer identity columns are malformed"
        ) from exc
    expected_indices = expected["gene_index"].to_numpy(dtype=np.int64)
    expected_counts = expected["n_outer_test_observations"].to_numpy(dtype=np.int64)
    if (
        not np.isfinite(observed_indices).all()
        or not np.array_equal(observed_indices, expected_indices)
        or not np.isfinite(observed_counts).all()
        or not np.array_equal(observed_counts, expected_counts)
    ):
        raise VerificationError(
            "per-gene sidecar index/order or observation counts changed"
        )
    if frame["gene"].isna().any():
        raise VerificationError(
            "per-gene sidecar gene identities contain missing values"
        )
    observed_genes = frame["gene"].astype(str).to_numpy()
    expected_genes = expected["gene"].astype(str).to_numpy()
    if len(set(observed_genes.tolist())) != len(observed_genes) or not np.array_equal(
        observed_genes, expected_genes
    ):
        raise VerificationError(
            "per-gene sidecar gene identities are duplicated or out of order"
        )

    for field in (
        "mse",
        "mae",
        "train_gene_mean_baseline_mse",
        "mse_improvement_over_train_gene_mean",
    ):
        try:
            observed = pd.to_numeric(frame[field], errors="raise").to_numpy(
                dtype=np.float64
            )
        except (TypeError, ValueError) as exc:
            raise VerificationError(
                f"per-gene sidecar metric {field} is malformed"
            ) from exc
        expected_values = expected[field].to_numpy(dtype=np.float64)
        if not np.allclose(
            observed,
            expected_values,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        ):
            raise VerificationError(
                f"per-gene sidecar metric {field} differs from prediction replay"
            )
    return binding


def _finite_descriptive_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise VerificationError("descriptive aggregation requires finite values")
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "minimum": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
    }


def _paired_descriptive_aggregation(
    metric_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected = _expected_campaign_identities()
    indexed: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for row in metric_rows:
        try:
            identity = (str(row["variant"]), int(row["seed"]), str(row["fold"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise VerificationError("metric-row identity is malformed") from exc
        if identity in indexed:
            raise VerificationError(f"duplicate metric-row identity {identity}")
        indexed[identity] = row
    if set(indexed) != expected:
        raise VerificationError("metric rows do not cover the fixed 105-fold matrix")

    variants: dict[str, Any] = {}
    for variant in (item.name for item in VARIANTS):
        variant_rows = [
            indexed[(variant, seed, f"fold-{fold:02d}")]
            for seed in CANONICAL_SEEDS
            for fold in range(CANONICAL_FOLDS)
        ]
        variants[variant] = {
            metric: _finite_descriptive_summary(
                [float(row["metrics"][metric]) for row in variant_rows]
            )
            for metric in PAIRED_METRICS
        }

    comparisons: dict[str, Any] = {}
    for variant in (item.name for item in VARIANTS):
        if variant == REFERENCE_VARIANT:
            continue
        metric_comparisons: dict[str, Any] = {}
        for metric in PAIRED_METRICS:
            deltas: list[float] = []
            for seed in CANONICAL_SEEDS:
                for fold in range(CANONICAL_FOLDS):
                    fold_name = f"fold-{fold:02d}"
                    candidate = float(
                        indexed[(variant, seed, fold_name)]["metrics"][metric]
                    )
                    reference = float(
                        indexed[(REFERENCE_VARIANT, seed, fold_name)]["metrics"][metric]
                    )
                    delta = (
                        candidate - reference
                        if metric
                        in {
                            "residual_r2_vs_train_gene_mean",
                            "residual_pearson_pooled",
                        }
                        else reference - candidate
                    )
                    deltas.append(delta)
            summary = _finite_descriptive_summary(deltas)
            tolerance = 1e-12
            summary.update(
                {
                    "positive_direction": "candidate_better_than_reference",
                    "wins": int(sum(value > tolerance for value in deltas)),
                    "ties": int(sum(abs(value) <= tolerance for value in deltas)),
                    "losses": int(sum(value < -tolerance for value in deltas)),
                }
            )
            metric_comparisons[metric] = summary
        comparisons[variant] = metric_comparisons

    return {
        "reference_variant": REFERENCE_VARIANT,
        "pairing_key": ["seed", "outer_fold"],
        "n_pairs_per_comparison": len(CANONICAL_SEEDS) * CANONICAL_FOLDS,
        "inference": "conditional_descriptive_only",
        "p_values": None,
        "confidence_intervals": None,
        "dependence_warning": (
            "Outer folds and repeated seeds reuse the same cohort. Summaries are "
            "paired protocol diagnostics, not IID sampling inference."
        ),
        "variant_metrics": variants,
        "paired_comparisons": comparisons,
    }


def _split_record(
    train: np.ndarray,
    test: np.ndarray,
    cell_ids: Sequence[str],
) -> dict[str, Any]:
    if set(train).intersection(test):
        raise ValueError("outer training and test indices overlap")
    if set(train).union(test) != set(range(len(cell_ids))):
        raise ValueError("outer training and test indices do not partition the cohort")
    return {
        "outer_train": {
            "n": int(train.size),
            "indices_sha256": _hash_indices(train),
            "cell_ids_sha256": _hash_strings([cell_ids[int(idx)] for idx in train]),
        },
        "outer_test": {
            "n": int(test.size),
            "indices_sha256": _hash_indices(test),
            "cell_ids_sha256": _hash_strings([cell_ids[int(idx)] for idx in test]),
        },
    }


def _token_comparison(
    candidate: SensitivityMutationFeatures,
    reference: SensitivityMutationFeatures,
) -> dict[str, Any]:
    if candidate.tokens.shape != reference.tokens.shape:
        raise ValueError("candidate and reference token arrays do not align")
    jaccard: list[float] = []
    identical_sequence = 0
    for row in range(candidate.tokens.shape[0]):
        left = candidate.tokens[row, candidate.token_mask[row]].tolist()
        right = reference.tokens[row, reference.token_mask[row]].tolist()
        left_set, right_set = set(left), set(right)
        union = left_set.union(right_set)
        jaccard.append(
            len(left_set.intersection(right_set)) / len(union) if union else 1.0
        )
        identical_sequence += int(left == right)
    return {
        "reference_variant": "k64-frequency",
        "mean_selected_token_set_jaccard": float(np.mean(jaccard)),
        "median_selected_token_set_jaccard": float(np.median(jaccard)),
        "fraction_cells_identical_ordered_token_sequence": float(
            identical_sequence / candidate.tokens.shape[0]
        ),
    }


def _load_audit_inputs(
    *, mutation_chunksize: int
) -> tuple[list[str], list[str], list[set[str]], list[dict[str, Any]], np.ndarray]:
    genes = _read_lines(PROCESSED / "genes.txt")
    cells = _read_lines(PROCESSED / "cell_lines.txt")
    lineages = _read_lines(PROCESSED / "lineages.txt")
    if len(cells) != len(lineages):
        raise RuntimeError("cell and lineage manifests do not align")
    print(
        f"[k-token-pmi] loading mutation annotations for {len(cells)} cells",
        flush=True,
    )
    annotations = load_token_annotations(
        DEPMAP / "OmicsSomaticMutations.csv",
        cells,
        chunksize=mutation_chunksize,
    )
    mutations = [set(per_cell) for per_cell in annotations]
    with np.load(PROCESSED / "ppi_graph.npz") as graph:
        edge_index = np.asarray(graph["edge_index"], dtype=np.int64)
    return genes, cells, mutations, annotations, edge_index


def _fit_features(
    mutations: Sequence[set[str]],
    annotations: Sequence[Mapping[str, Any]],
    genes: Sequence[str],
    train: np.ndarray,
    variant: SensitivityVariant,
    *,
    top_mut: int,
) -> SensitivityMutationFeatures:
    return fit_sensitivity_mutation_features(
        mutations,
        annotations,
        genes,
        train,
        top_n=top_mut,
        effective_k=variant.k_tokens,
        token_strategy=variant.token_strategy,
    )


def _fit_pmi(
    edge_index: np.ndarray,
    genes: Sequence[str],
    features: SensitivityMutationFeatures,
    train: np.ndarray,
    *,
    mode: str,
    pmi_chunk_size: int,
) -> PmiFit:
    return fit_pmi_sensitivity(
        edge_index,
        genes,
        features.matrix,
        features.vocabulary,
        train,
        mode=mode,
        chunk_size=pmi_chunk_size,
    )


def run_audit(args: argparse.Namespace) -> Path:
    started = time.time()
    run_dir = make_unique_run_dir(
        OUTPUT_ROOT,
        protocol="audit-p1",
        condition="all-variants",
        seed=int(args.seeds[0]),
    )
    input_paths = [
        PROCESSED / "genes.txt",
        PROCESSED / "cell_lines.txt",
        PROCESSED / "lineages.txt",
        PROCESSED / "ppi_graph.npz",
        DEPMAP / "OmicsSomaticMutations.csv",
    ]
    code_paths = [Path(__file__).resolve(), ROOT / "code" / "k_token_pmi_protocol.py"]
    atomic_write_json(
        run_dir / "run-manifest.json",
        {
            "schema": "DepPLM-k-token-pmi-audit-v1",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "argv": sys.argv,
            "arguments": vars(args),
            "runtime": _runtime_record(),
            "inputs": [_file_record(path) for path in input_paths],
            "code": [_file_record(path) for path in code_paths],
            "variants": [vars(variant) for variant in VARIANTS],
            "leakage_control": (
                "Mutation prevalence, vocabulary, token order and PMI are fitted on "
                "the outer-training cells separately in every seed/fold. Outer-test "
                "labels are not loaded by this audit."
            ),
            "result_scope": "preprocessing_audit_only",
            "model_trained": False,
            "outer_test_predictions": None,
            "scientific_result": False,
            "run_status": "in_progress",
        },
    )
    print(f"[k-token-pmi] audit output={run_dir}", flush=True)
    genes, cells, mutations, annotations, edge_index = _load_audit_inputs(
        mutation_chunksize=args.mutation_chunksize
    )
    atomic_write_json(
        run_dir / "cohort-manifest.json",
        {
            "n_genes": len(genes),
            "n_cells": len(cells),
            "n_edge_records": int(edge_index.shape[0]),
            "genes_sha256": _hash_strings(genes),
            "cell_ids_sha256": _hash_strings(cells),
            "n_mutated_genes_per_cell": {
                "min": int(min(map(len, mutations))),
                "median": float(np.median([len(item) for item in mutations])),
                "max": int(max(map(len, mutations))),
            },
        },
    )

    records: list[dict[str, Any]] = []
    artifact_records: list[dict[str, Any]] = []
    completed: list[str] = []
    for seed in args.seeds:
        splits = make_kfold_outer_splits(len(cells), args.folds, seed)
        fold_numbers = range(len(splits)) if args.fold_id < 0 else [int(args.fold_id)]
        if args.fold_id >= len(splits):
            raise IndexError(
                f"--fold-id {args.fold_id} is outside 0..{len(splits) - 1}"
            )
        for fold_number in fold_numbers:
            fold_started = time.time()
            train, test = splits[int(fold_number)]
            prefix = f"seed-{seed}-fold-{int(fold_number):02d}"
            print(f"[k-token-pmi] audit {prefix}", flush=True)
            atomic_write_json(
                run_dir / f"{prefix}-split.json",
                {
                    "seed": int(seed),
                    "fold_number": int(fold_number),
                    **_split_record(train, test, cells),
                },
            )

            reference_variant = variant_by_name("k64-frequency")
            reference = _fit_features(
                mutations,
                annotations,
                genes,
                train,
                reference_variant,
                top_mut=args.top_mut,
            )
            pmi_by_mode = {
                mode: _fit_pmi(
                    edge_index,
                    genes,
                    reference,
                    train,
                    mode=mode,
                    pmi_chunk_size=args.pmi_chunk_size,
                )
                for mode in ("neutral-one", "valid-only", "unit")
            }
            np.testing.assert_allclose(
                pmi_by_mode["neutral-one"].raw_pmi,
                pmi_by_mode["valid-only"].raw_pmi,
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )
            np.testing.assert_allclose(
                pmi_by_mode["neutral-one"].raw_pmi,
                pmi_by_mode["unit"].raw_pmi,
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )
            pmi_npz = _atomic_write_npz(
                run_dir / f"{prefix}-pmi-edge-records.npz",
                raw_pmi=pmi_by_mode["neutral-one"].raw_pmi,
                valid_mask=pmi_by_mode["neutral-one"].valid_mask,
                neutral_one_weights=pmi_by_mode["neutral-one"].weights,
                valid_only_weights=pmi_by_mode["valid-only"].weights,
                unit_weights=pmi_by_mode["unit"].weights,
            )
            artifact_records.append(pmi_npz)
            for mode, pmi in pmi_by_mode.items():
                per_gene = pmi_per_gene_summary(edge_index, genes, pmi)
                artifact_records.append(
                    _atomic_write_gzip_csv(
                        run_dir / f"{prefix}-pmi-{mode}-per-gene.csv.gz",
                        per_gene,
                    )
                )

            fold_manifests: list[dict[str, Any]] = []
            for variant in VARIANTS:
                features = (
                    reference
                    if variant.name == reference_variant.name
                    else _fit_features(
                        mutations,
                        annotations,
                        genes,
                        train,
                        variant,
                        top_mut=args.top_mut,
                    )
                )
                pmi = pmi_by_mode[variant.pmi_mode]
                manifest = {
                    "variant": vars(variant),
                    "preprocessing": feature_manifest(features, pmi),
                    "token_comparison": _token_comparison(features, reference),
                }
                fold_manifests.append(manifest)
                count_summary = manifest["preprocessing"][
                    "token_count_before_truncation"
                ]
                records.append(
                    {
                        "seed": int(seed),
                        "fold_number": int(fold_number),
                        "variant": variant.name,
                        "axis": variant.axis,
                        "effective_k": variant.k_tokens,
                        "token_strategy": variant.token_strategy,
                        "pmi_mode": variant.pmi_mode,
                        "n_train_cells": int(train.size),
                        "n_test_cells": int(test.size),
                        "vocabulary_size": len(features.vocabulary),
                        "n_cells_truncated": count_summary["n_truncated"],
                        **manifest["token_comparison"],
                        **{
                            f"pmi_{key}": value
                            for key, value in pmi.statistics.items()
                            if key
                            in {
                                "n_edge_records",
                                "n_valid_edge_records",
                                "fraction_valid_edge_records",
                                "n_nonunit_edge_records",
                                "fraction_nonunit_edge_records",
                                "n_zero_weight_edge_records",
                                "n_valid_lower_clipped",
                                "n_valid_upper_clipped",
                                "weight_mean",
                            }
                        },
                    }
                )
            atomic_write_json(
                run_dir / f"{prefix}-variant-manifests.json",
                {
                    "seed": int(seed),
                    "fold_number": int(fold_number),
                    "fit_scope": "outer_train_only",
                    "variants": fold_manifests,
                    "pmi_edge_artifact": pmi_npz,
                    "wall_seconds": time.time() - fold_started,
                    "scientific_result": False,
                },
            )
            completed.append(prefix)

    summary_frame = pd.DataFrame.from_records(records).sort_values(
        ["seed", "fold_number", "variant"], ignore_index=True
    )
    summary_record = _atomic_write_gzip_csv(
        run_dir / "audit-summary.csv.gz", summary_frame
    )
    artifact_records.append(summary_record)
    expected_folds = len(args.seeds) * (args.folds if args.fold_id < 0 else 1)
    protocol_complete = (
        len(completed) == expected_folds
        and len(records) == expected_folds * len(VARIANTS)
        and args.fold_id == -1
    )
    canonical_complete = (
        protocol_complete
        and tuple(args.seeds) == CANONICAL_SEEDS
        and args.folds == CANONICAL_FOLDS
        and args.fold_id == -1
        and args.top_mut == 5_000
    )
    completion = {
        "schema": "DepPLM-k-token-pmi-audit-completion-v1",
        "run_status": "protocol_complete" if protocol_complete else "partial",
        "protocol_complete": protocol_complete,
        "canonical_audit_complete": canonical_complete,
        "completed_seed_folds": completed,
        "n_completed_seed_folds": len(completed),
        "n_variant_records": len(records),
        "summary": summary_record,
        "artifacts": artifact_records,
        "wall_seconds": time.time() - started,
        "model_trained": False,
        "outer_test_predictions": None,
        "eligible_for_scientific_aggregation": False,
        "scientific_result": False,
        "scientific_result_reason": (
            "This is a label-free preprocessing sensitivity audit. It does not "
            "estimate predictive performance or an observed transfer gain."
        ),
    }
    atomic_write_json(run_dir / "completion.json", completion)
    print(f"[k-token-pmi] audit complete={protocol_complete}", flush=True)
    return run_dir


def _variant_fold_data(
    base: corrected.BaseData,
    fit_indices: np.ndarray,
    *,
    seed: int,
    n_expr_pc: int,
    top_mut: int,
    annotations: Sequence[Mapping[str, Any]],
    variant: SensitivityVariant,
) -> corrected.FoldData:
    expression_fit = fit_expression_pca(
        base.expression_raw,
        fit_indices,
        n_components=n_expr_pc,
        seed=derive_seed(seed, "pca"),
    )
    expression = expression_fit.transform(base.expression_raw)
    mutation = fit_sensitivity_mutation_features(
        base.mutations_by_cell,
        annotations,
        base.genes,
        fit_indices,
        top_n=top_mut,
        effective_k=variant.k_tokens,
        token_strategy=variant.token_strategy,
    )
    lineage_fit = fit_lineage_one_hot(base.lineages, fit_indices)
    lineage = lineage_fit.transform(base.lineages)
    edge_index_np = base.edge_index.T.astype(np.int64, copy=False)
    pmi = fit_pmi_sensitivity(
        edge_index_np,
        base.genes,
        mutation.matrix,
        mutation.vocabulary,
        fit_indices,
        mode=variant.pmi_mode,
    )
    edge_index = torch.from_numpy(edge_index_np.copy()).long().contiguous()
    edge_weight = torch.from_numpy(pmi.weights.copy()).float()
    adjacency = normalized_adjacency(edge_index, edge_weight, len(base.genes))
    selectivity, _ = selectivity_labels_from_train(base.gene_effect, fit_indices)
    fold_data = corrected.FoldData(
        expression=expression,
        mutation=mutation,
        lineage=lineage,
        expression_fit=expression_fit,
        lineage_fit=lineage_fit,
        edge_index=edge_index,
        edge_weight=edge_weight,
        adjacency=adjacency,
        selectivity_labels=selectivity,
        fit_indices=np.asarray(fit_indices, dtype=np.int64).copy(),
    )
    fold_data.k_token_pmi_manifest = feature_manifest(mutation, pmi)
    return fold_data


@contextmanager
def _patched_corrected_runner(
    *,
    annotations: Sequence[Mapping[str, Any]],
    variant: SensitivityVariant,
) -> Iterator[None]:
    """Patch process-local runner hooks while preserving their original values."""
    original_k = corrected.K_TOKENS
    original_fit = corrected.fit_fold_data
    original_manifest = corrected.preprocessing_manifest

    def fit_fold_data(base, fit_indices, *, seed, n_expr_pc, top_mut):
        return _variant_fold_data(
            base,
            fit_indices,
            seed=seed,
            n_expr_pc=n_expr_pc,
            top_mut=top_mut,
            annotations=annotations,
            variant=variant,
        )

    def preprocessing_manifest(fold_data, evaluation_idx):
        manifest = original_manifest(fold_data, evaluation_idx)
        manifest["k_token_pmi_sensitivity"] = fold_data.k_token_pmi_manifest
        manifest["mutation"]["vocabulary_sha256"] = fold_data.k_token_pmi_manifest[
            "vocabulary_sha256"
        ]
        manifest["mutation"]["allocated_token_slots"] = int(
            fold_data.mutation.tokens.shape[1]
        )
        manifest["mutation"]["effective_k"] = int(variant.k_tokens)
        manifest["mutation"]["token_strategy"] = variant.token_strategy
        manifest["graph"]["pmi_mode"] = variant.pmi_mode
        return manifest

    corrected.K_TOKENS = int(variant.k_tokens)
    corrected.fit_fold_data = fit_fold_data
    corrected.preprocessing_manifest = preprocessing_manifest
    try:
        yield
    finally:
        corrected.K_TOKENS = original_k
        corrected.fit_fold_data = original_fit
        corrected.preprocessing_manifest = original_manifest


def _per_gene_prediction_summary_from_arrays(
    *,
    predictions: np.ndarray,
    test_indices: np.ndarray,
    train_mean: np.ndarray,
    gene_ids: np.ndarray,
    base: corrected.BaseData | VerificationBase,
) -> pd.DataFrame:
    predictions = np.asarray(predictions, dtype=np.float64)
    test_indices = np.asarray(test_indices, dtype=np.int64)
    train_mean = np.asarray(train_mean, dtype=np.float64)
    gene_ids = np.asarray(gene_ids, dtype=str)
    targets = np.asarray(base.gene_effect[:, test_indices].T, dtype=np.float64)
    if predictions.shape != targets.shape:
        raise ValueError("saved predictions and outer-test targets do not align")
    finite = np.isfinite(predictions) & np.isfinite(targets)
    error = predictions - targets
    baseline_error = train_mean[None, :] - targets
    n = finite.sum(axis=0)
    squared = np.where(finite, error**2, 0.0).sum(axis=0)
    absolute = np.where(finite, np.abs(error), 0.0).sum(axis=0)
    baseline_squared = np.where(finite, baseline_error**2, 0.0).sum(axis=0)
    mse = np.divide(squared, n, out=np.full(n.size, np.nan), where=n > 0)
    mae = np.divide(absolute, n, out=np.full(n.size, np.nan), where=n > 0)
    baseline_mse = np.divide(
        baseline_squared, n, out=np.full(n.size, np.nan), where=n > 0
    )
    return pd.DataFrame(
        {
            "gene_index": np.arange(gene_ids.size, dtype=np.int64),
            "gene": gene_ids,
            "n_outer_test_observations": n,
            "mse": mse,
            "mae": mae,
            "train_gene_mean_baseline_mse": baseline_mse,
            "mse_improvement_over_train_gene_mean": baseline_mse - mse,
        }
    )


def _per_gene_prediction_summary(
    prediction_path: Path,
    base: corrected.BaseData,
) -> pd.DataFrame:
    with np.load(prediction_path, allow_pickle=False) as artifact:
        return _per_gene_prediction_summary_from_arrays(
            predictions=np.asarray(artifact["prediction"]),
            test_indices=np.asarray(artifact["outer_test_indices"]),
            train_mean=np.asarray(artifact["train_gene_mean"]),
            gene_ids=np.asarray(artifact["gene_ids"]),
            base=base,
        )


def _training_completion(
    *,
    args: argparse.Namespace,
    variant: SensitivityVariant,
    completed: Sequence[int],
    expected: Sequence[int],
    sidecars: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    cohort_manifest: Mapping[str, Any],
    fold_artifacts: Sequence[Mapping[str, Any]],
    wall_seconds: float,
) -> dict[str, Any]:
    protocol_complete = (
        list(completed) == list(expected) and not args.pilot and args.fold_id == -1
    )
    canonical = (
        protocol_complete
        and args.protocol == "p1"
        and args.fold_id == -1
        and args.folds == CANONICAL_FOLDS
        and args.seed in CANONICAL_SEEDS
        and args.inner_val_fraction == 0.2
        and args.inner_max_epochs == 20
        and args.patience == 5
        and args.batch_size == 64
        and args.gene_subsample == 6_000
        and args.top_mut == 5_000
        and args.n_expr_pc == 128
    )
    predictions_complete = len(predictions) == len(completed) and all(
        isinstance(record.get("sha256"), str)
        and len(str(record["sha256"])) == 64
        and int(record.get("bytes", 0)) > 0
        for record in predictions
    )
    sidecars_complete = len(sidecars) == len(completed) and all(
        isinstance(record.get("sha256"), str)
        and len(str(record["sha256"])) == 64
        and int(record.get("bytes", 0)) > 0
        for record in sidecars
    )

    def record_complete(record: Any) -> bool:
        return (
            isinstance(record, Mapping)
            and isinstance(record.get("bytes"), int)
            and not isinstance(record.get("bytes"), bool)
            and int(record["bytes"]) > 0
            and _is_sha256(record.get("sha256"))
            and isinstance(record.get("path"), str)
            and bool(record["path"])
        )

    expected_fold_names = {f"fold-{fold:02d}" for fold in completed}
    protocol_artifacts_complete = (
        record_complete(cohort_manifest)
        and len(fold_artifacts) == len(completed)
        and {record.get("fold") for record in fold_artifacts} == expected_fold_names
        and all(
            isinstance(record, Mapping)
            and set(record) == {"fold", "split_manifest", "fold_result"}
            and record_complete(record.get("split_manifest"))
            and record_complete(record.get("fold_result"))
            for record in fold_artifacts
        )
    )
    if (
        not predictions_complete
        or not sidecars_complete
        or not protocol_artifacts_complete
    ):
        protocol_complete = False
        canonical = False
    reasons: list[str] = []
    if args.pilot:
        reasons.append("pilot mode is a structural smoke test")
    if list(completed) != list(expected):
        reasons.append("not every planned outer fold completed")
    if not predictions_complete:
        reasons.append("not every completed fold has a hash-bound prediction artifact")
    if not sidecars_complete:
        reasons.append("not every completed fold has a hash-bound per-gene sidecar")
    if not protocol_artifacts_complete:
        reasons.append(
            "cohort, split, fold-result, or preprocessing artifacts are not "
            "completely hash-bound"
        )
    if not canonical:
        reasons.append("run differs from at least one prespecified canonical setting")
    return {
        "schema": "DepPLM-k-token-pmi-training-completion-v2",
        "variant": vars(variant),
        "run_status": "protocol_complete" if protocol_complete else "partial",
        "protocol_complete": protocol_complete,
        "canonical_variant_seed_complete": canonical,
        "planned_fold_numbers": list(expected),
        "completed_fold_numbers": list(completed),
        "per_gene_prediction_sidecars": list(sidecars),
        "outer_test_predictions": list(predictions),
        "cohort_manifest": dict(cohort_manifest),
        "fold_protocol_artifacts": list(fold_artifacts),
        "wall_seconds": wall_seconds,
        "model_trained": bool(completed),
        "outer_test_predictions_saved_and_hashed": predictions_complete,
        "per_gene_prediction_sidecars_saved_and_hashed": sidecars_complete,
        "protocol_artifacts_saved_and_hashed": protocol_artifacts_complete,
        "eligible_for_scientific_aggregation": canonical,
        "scientific_result": False,
        "scientific_result_reason": (
            "A complete canonical variant-by-seed run still requires paired "
            "aggregation across all variants and all three seeds."
            if canonical
            else "; ".join(reasons)
        ),
    }


def _finalise_training_artifact_records(
    *,
    run_dir: Path,
    cohort_manifest: Mapping[str, Any],
    fold_artifacts: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    sidecars: Sequence[Mapping[str, Any]],
    artifact_root: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Re-read every training output immediately before completion publication."""
    artifact_root = artifact_root.resolve()
    run_dir = run_dir.resolve()
    try:
        run_relative = run_dir.relative_to(artifact_root)
    except ValueError as exc:
        raise VerificationError("training run directory is outside the artifact root") from exc
    expected_cohort_path = _normalised_record_path(
        str(run_relative / "cohort-manifest.json")
    )
    if _normalised_record_path(cohort_manifest.get("path")) != expected_cohort_path:
        raise VerificationError("final cohort manifest path is not canonical")
    cohort_payload, verified_cohort = _read_verified_json_artifact(
        cohort_manifest,
        artifact_root=artifact_root,
        label="final cohort manifest",
    )
    if not isinstance(cohort_payload, Mapping):
        raise VerificationError("final cohort manifest is not a JSON object")

    verified_folds: list[dict[str, Any]] = []
    artifact_paths = {expected_cohort_path}
    for fold_record in fold_artifacts:
        fold_name = fold_record.get("fold")
        split_record = fold_record.get("split_manifest")
        result_record = fold_record.get("fold_result")
        if (
            not isinstance(fold_name, str)
            or not isinstance(split_record, Mapping)
            or not isinstance(result_record, Mapping)
        ):
            raise VerificationError("final fold protocol artifact record is malformed")
        expected_split_path = _normalised_record_path(
            str(run_relative / f"{fold_name}-split.json")
        )
        expected_result_path = _normalised_record_path(
            str(run_relative / f"{fold_name}-result.json")
        )
        if (
            _normalised_record_path(split_record.get("path")) != expected_split_path
            or _normalised_record_path(result_record.get("path"))
            != expected_result_path
        ):
            raise VerificationError(
                f"final split or fold-result path changed for {fold_name}"
            )
        split_payload, verified_split = _read_verified_json_artifact(
            split_record,
            artifact_root=artifact_root,
            label=f"final {fold_name} split manifest",
        )
        result_payload, verified_result = _read_verified_json_artifact(
            result_record,
            artifact_root=artifact_root,
            label=f"final {fold_name} result",
        )
        preprocessing = (
            result_payload.get("preprocessing")
            if isinstance(result_payload, Mapping)
            else None
        )
        if not isinstance(split_payload, Mapping) or split_payload.get("fold") != fold_name:
            raise VerificationError(f"final split identity changed for {fold_name}")
        if (
            not isinstance(result_payload, Mapping)
            or result_payload.get("fold") != fold_name
            or not isinstance(preprocessing, Mapping)
            or set(preprocessing) != {"inner_train_fit", "outer_train_refit"}
            or not all(isinstance(value, Mapping) for value in preprocessing.values())
        ):
            raise VerificationError(
                f"final fold result or preprocessing changed for {fold_name}"
            )
        artifact_paths.update((expected_split_path, expected_result_path))
        verified_folds.append(
            {
                "fold": fold_name,
                "split_manifest": verified_split,
                "fold_result": verified_result,
            }
        )

    def refreshed_record(record: Mapping[str, Any]) -> dict[str, Any]:
        binding = _verified_artifact_binding(record, artifact_root=artifact_root)
        normalised_path = _normalised_record_path(binding["path"])
        resolved_path = (artifact_root / Path(normalised_path)).resolve()
        try:
            resolved_path.relative_to(run_dir)
        except ValueError as exc:
            raise VerificationError(
                "final training artifact is outside its immutable run directory"
            ) from exc
        if normalised_path in artifact_paths:
            raise VerificationError("final training artifact paths are not unique")
        artifact_paths.add(normalised_path)
        metadata = {
            key: value
            for key, value in record.items()
            if key not in {"path", "bytes", "mtime_ns", "sha256"}
        }
        return {**metadata, **binding}

    fold_names = [record["fold"] for record in verified_folds]
    if len(fold_names) != len(set(fold_names)):
        raise VerificationError("final fold protocol identities are duplicated")
    if len(predictions) != len(fold_names) or len(sidecars) != len(fold_names):
        raise VerificationError("final training artifact matrix is incomplete")
    if (
        {record.get("fold") for record in predictions} != set(fold_names)
        or {record.get("fold") for record in sidecars} != set(fold_names)
    ):
        raise VerificationError("final prediction or sidecar fold identities differ")
    required_prediction_arrays = {
        "prediction",
        "outer_test_indices",
        "cell_ids",
        "gene_ids",
        "source_cell_indices",
        "source_gene_indices",
        "train_gene_mean",
    }
    for record in predictions:
        if record.get("schema") != "DepPLM-outer-test-predictions-v1":
            raise VerificationError("final prediction artifact schema is not recognised")
        arrays = record.get("arrays")
        if not isinstance(arrays, Mapping) or set(arrays) != required_prediction_arrays:
            raise VerificationError("final prediction artifact array contract is incomplete")
    verified_predictions = [refreshed_record(record) for record in predictions]
    verified_sidecars = [refreshed_record(record) for record in sidecars]
    if len(artifact_paths) != 1 + 4 * len(fold_names):
        raise VerificationError("final training artifact path count is not canonical")
    return (
        verified_cohort,
        verified_folds,
        verified_predictions,
        verified_sidecars,
    )


def run_training(
    args: argparse.Namespace,
    *,
    input_records: Sequence[Mapping[str, Any]] | None = None,
    transaction_runtime: Mapping[str, Any] | None = None,
) -> Path:
    started = time.time()
    owns_runtime_transaction = transaction_runtime is None
    launch_runtime = (
        _runtime_record() if owns_runtime_transaction else transaction_runtime
    )
    variant = variant_by_name(args.variant)
    if args.pilot:
        args.inner_max_epochs = 1
        args.patience = 1
        args.batch_size = min(args.batch_size, 32)
        args.gene_subsample = 0
        args.top_mut = min(args.top_mut, 128)
        args.n_expr_pc = min(args.n_expr_pc, 32)
    device = corrected._device_from_arg(args.device)
    set_global_seed(args.seed)
    bound_inputs = (
        [dict(record) for record in input_records]
        if input_records is not None
        else [_file_record(path) for path in _canonical_input_paths()]
    )
    _verify_current_bindings(
        bound_inputs,
        current_records=[_file_record(path) for path in _canonical_input_paths()],
        label="training input preflight",
    )
    bound_code = [_file_record(path) for path in TRAINING_CODE_PATHS]
    bound_verifier_source = _file_record(VERIFIER_CODE_PATH)
    embedding_provenance = corrected.validate_embedding_bundle(ROOT)
    run_dir = make_unique_run_dir(
        OUTPUT_ROOT,
        protocol=args.protocol + ("-pilot" if args.pilot else ""),
        condition=variant.name,
        seed=args.seed,
    )
    atomic_write_json(
        run_dir / "run-manifest.json",
        {
            "schema": "DepPLM-k-token-pmi-training-v1",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "argv": sys.argv,
            "arguments": vars(args),
            "variant": vars(variant),
            "runtime": {**launch_runtime, "device": device},
            "inputs": bound_inputs,
            "embedding_provenance": embedding_provenance,
            "training_execution": _training_execution_signature(),
            "code": bound_code,
            "verifier_source_at_launch": bound_verifier_source,
            "output_policy": "unique_directory_atomic_outputs_no_overwrite",
            "run_status": "in_progress",
            "scientific_result": False,
        },
    )
    print(f"[k-token-pmi] training output={run_dir}", flush=True)
    base = corrected.load_base_data(
        pilot=args.pilot,
        seed=args.seed,
        pilot_cells=args.pilot_cells,
        pilot_genes=args.pilot_genes,
    )
    annotations = load_token_annotations(
        DEPMAP / "OmicsSomaticMutations.csv",
        base.cell_ids,
        chunksize=args.mutation_chunksize,
    )
    if [set(item) for item in annotations] != base.mutations_by_cell:
        raise RuntimeError(
            "annotation-derived mutation identities do not match corrected base data"
        )
    _verify_current_bindings(
        bound_inputs,
        current_records=[_file_record(path) for path in _canonical_input_paths()],
        label="post-load training input",
    )
    cohort_path = run_dir / "cohort-manifest.json"
    atomic_write_json(
        cohort_path,
        {
            "n_genes": len(base.genes),
            "n_cells": len(base.cell_ids),
            "n_edge_records": int(base.edge_index.shape[0]),
            "genes_sha256": _hash_strings(base.genes),
            "cell_ids_sha256": _hash_strings(base.cell_ids),
            "source_cell_indices": base.source_cell_indices,
            "source_gene_indices": base.source_gene_indices,
        },
    )
    cohort_record = _file_record(cohort_path, relative_to=OUTPUT_ROOT)

    grid: list[Mapping[str, Any]] = (
        [{"hidden": 32, "lr": 1e-3, "wd": 1e-5, "lamb_sel": 0.1}]
        if args.pilot
        else corrected.FULL_GRID
    )
    outer_splits = corrected._outer_splits(args, base)
    if args.fold_id >= 0:
        if args.fold_id >= len(outer_splits):
            raise IndexError(f"--fold-id {args.fold_id} is outside the planned folds")
        selected = [(args.fold_id, outer_splits[args.fold_id])]
    elif args.pilot:
        selected = [(0, outer_splits[0])]
    else:
        selected = list(enumerate(outer_splits))
    expected = [number for number, _ in selected]
    completed: list[int] = []
    sidecars: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    fold_artifacts: list[dict[str, Any]] = []
    with _patched_corrected_runner(annotations=annotations, variant=variant):
        for fold_number, (fold_name, outer_train, outer_test) in selected:
            result = corrected.run_fold(
                fold_name=fold_name,
                fold_number=fold_number,
                outer_train=outer_train,
                outer_test=outer_test,
                base=base,
                condition=variant.name,
                base_config=dict(corrected.CONDITIONS["DepPLM"]),
                grid=grid,
                args=args,
                device=device,
                run_dir=run_dir,
            )
            prediction_path = run_dir / result["outer_test_predictions"]["path"]
            if not prediction_path.is_file():
                prediction_path = run_dir / f"{fold_name}-predictions.npz"
            prediction_record = _file_record(
                prediction_path,
                relative_to=OUTPUT_ROOT,
            )
            expected_prediction = result["outer_test_predictions"]
            if (
                prediction_record["sha256"] != expected_prediction["sha256"]
                or prediction_record["bytes"] != expected_prediction["bytes"]
            ):
                raise RuntimeError(
                    f"{fold_name} outer-test prediction no longer matches its fold record"
                )
            predictions.append(
                {
                    "fold": fold_name,
                    "schema": expected_prediction["schema"],
                    "arrays": expected_prediction["arrays"],
                    **prediction_record,
                }
            )
            per_gene = _per_gene_prediction_summary(prediction_path, base)
            sidecar = _atomic_write_gzip_csv(
                run_dir / f"{fold_name}-per-gene-prediction-metrics.csv.gz",
                per_gene,
            )
            sidecars.append({"fold": fold_name, **sidecar})
            fold_artifacts.append(
                {
                    "fold": fold_name,
                    "split_manifest": _file_record(
                        run_dir / f"{fold_name}-split.json",
                        relative_to=OUTPUT_ROOT,
                    ),
                    "fold_result": _file_record(
                        run_dir / f"{fold_name}-result.json",
                        relative_to=OUTPUT_ROOT,
                    ),
                }
            )
            completed.append(fold_number)
    (
        verified_cohort,
        verified_fold_artifacts,
        verified_predictions,
        verified_sidecars,
    ) = _finalise_training_artifact_records(
        run_dir=run_dir,
        cohort_manifest=cohort_record,
        fold_artifacts=fold_artifacts,
        predictions=predictions,
        sidecars=sidecars,
        artifact_root=OUTPUT_ROOT,
    )
    _verify_current_bindings(
        bound_inputs,
        current_records=[_file_record(path) for path in _canonical_input_paths()],
        label="final post-training input",
    )
    _verify_current_bindings(
        bound_code,
        current_records=[_file_record(path) for path in TRAINING_CODE_PATHS],
        label="final post-training code",
    )
    _verify_current_bindings(
        [bound_verifier_source],
        current_records=[_file_record(VERIFIER_CODE_PATH)],
        label="final post-training verifier source",
    )
    if owns_runtime_transaction and not _strict_json_equal(
        launch_runtime, _runtime_record()
    ):
        raise VerificationError("training runtime changed during model execution")
    completion = _training_completion(
        args=args,
        variant=variant,
        completed=completed,
        expected=expected,
        sidecars=verified_sidecars,
        predictions=verified_predictions,
        cohort_manifest=verified_cohort,
        fold_artifacts=verified_fold_artifacts,
        wall_seconds=time.time() - started,
    )
    atomic_write_json(run_dir / "completion.json", completion)
    print(
        f"[k-token-pmi] training complete folds={completed} "
        f"canonical={completion['canonical_variant_seed_complete']}",
        flush=True,
    )
    return run_dir


def _validate_completed_training_run(
    run_dir: Path,
    *,
    expected_args: argparse.Namespace,
    current_code_records: Sequence[Mapping[str, Any]],
    current_input_records: Sequence[Mapping[str, Any]],
    artifact_root: Path = OUTPUT_ROOT,
    transaction_runtime: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    validation_runtime = (
        _runtime_record() if transaction_runtime is None else transaction_runtime
    )
    manifest_path = run_dir / "run-manifest.json"
    completion_path = run_dir / "completion.json"
    if not manifest_path.is_file() or not completion_path.is_file():
        return None, "run manifest or completion marker is missing"
    try:
        manifest, manifest_record = _read_stable_json(
            manifest_path, relative_to=artifact_root
        )
        completion, completion_record = _read_stable_json(
            completion_path, relative_to=artifact_root
        )
    except (OSError, json.JSONDecodeError, VerificationError) as exc:
        return None, f"run metadata cannot be read: {exc}"
    if not isinstance(manifest, Mapping) or not isinstance(completion, Mapping):
        return None, "run manifest and completion marker must be JSON objects"
    if manifest.get("schema") != "DepPLM-k-token-pmi-training-v1":
        return None, "training manifest schema is not recognised"
    if completion.get("schema") != "DepPLM-k-token-pmi-training-completion-v2":
        return None, (
            "training completion schema lacks mandatory cohort/split/fold-result "
            "bindings; the run must be retrained"
        )
    expected_device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if expected_args.device == "auto"
        else expected_args.device
    )
    expected_runtime = {**validation_runtime, "device": expected_device}
    if not _strict_json_equal(manifest.get("runtime"), expected_runtime):
        return None, "training runtime differs from the current campaign"
    if manifest.get("output_policy") != "unique_directory_atomic_outputs_no_overwrite":
        return None, "training output policy is not immutable"
    if manifest.get("run_status") != "in_progress":
        return None, "training manifest run status is not canonical"
    if manifest.get("scientific_result") is not False:
        return None, "training manifest must not claim a scientific result"
    embedding_provenance = manifest.get("embedding_provenance")
    if (
        not isinstance(embedding_provenance, Mapping)
        or embedding_provenance.get("schema")
        != "DepPLM-verified-embedding-provenance-v1"
        or embedding_provenance.get("verification_status") != "verified"
        or not _is_sha256(embedding_provenance.get("provenance_sha256"))
    ):
        return None, "verified ESM embedding provenance is missing or malformed"
    recorded_execution = manifest.get("training_execution")
    if recorded_execution is None:
        return None, (
            "legacy training run lacks a training-execution signature and is "
            "not eligible for canonical reuse"
        )
    if not _strict_json_equal(recorded_execution, _training_execution_signature()):
        return None, "training execution signature differs from the current protocol"
    recorded_verifier = manifest.get("verifier_source_at_launch")
    if not isinstance(recorded_verifier, Mapping):
        return None, "training verifier source binding is missing"
    try:
        _verify_current_bindings(
            [recorded_verifier],
            current_records=[_file_record(VERIFIER_CODE_PATH)],
            label="training verifier source",
        )
    except VerificationError as exc:
        return None, str(exc)

    recorded_args = manifest.get("arguments")
    if not isinstance(recorded_args, Mapping):
        return None, "training arguments are missing"
    expected_arguments = vars(expected_args)
    if not _strict_json_equal(recorded_args, expected_arguments):
        return None, "training arguments differ from the canonical campaign job"
    try:
        if _code_signature(manifest.get("code", [])) != _code_signature(
            current_code_records
        ):
            return None, "training code bindings differ from the current campaign"
        if _code_signature(manifest.get("inputs", [])) != _code_signature(
            current_input_records
        ):
            return None, "training input bindings differ from the current campaign"
    except (TypeError, ValueError):
        return None, "training code or input bindings are malformed"

    variant = variant_by_name(expected_args.variant)
    if not _strict_json_equal(manifest.get("variant"), vars(variant)):
        return None, "training manifest variant definition differs"
    if not _strict_json_equal(completion.get("variant"), vars(variant)):
        return None, "training completion variant definition differs"
    if completion.get("run_status") != "protocol_complete":
        return None, "variant-seed training run status is not protocol complete"
    if completion.get("protocol_complete") is not True:
        return None, "variant-seed training run is not protocol complete"
    if completion.get("canonical_variant_seed_complete") is not True:
        return None, "variant-seed training run is not canonical"
    if completion.get("eligible_for_scientific_aggregation") is not True:
        return None, "variant-seed training run is not aggregation eligible"
    if completion.get("scientific_result") is not False:
        return None, "training completion must not claim a scientific result"
    expected_folds = list(range(CANONICAL_FOLDS))
    if not _strict_json_equal(completion.get("planned_fold_numbers"), expected_folds):
        return None, "planned outer folds differ from the canonical plan"
    if not _strict_json_equal(completion.get("completed_fold_numbers"), expected_folds):
        return None, "completed outer folds differ from the canonical plan"

    predictions = completion.get("outer_test_predictions")
    sidecars = completion.get("per_gene_prediction_sidecars")
    if not isinstance(predictions, list) or len(predictions) != CANONICAL_FOLDS:
        return None, "exactly five outer-test prediction artifacts are required"
    if not isinstance(sidecars, list) or len(sidecars) != CANONICAL_FOLDS:
        return None, "exactly five per-gene sidecars are required"
    if completion.get("outer_test_predictions_saved_and_hashed") is not True:
        return None, "prediction artifact completion flag is false"
    if completion.get("per_gene_prediction_sidecars_saved_and_hashed") is not True:
        return None, "per-gene sidecar completion flag is false"
    if completion.get("protocol_artifacts_saved_and_hashed") is not True:
        return None, "cohort/split/fold-result artifact completion flag is false"

    expected_fold_names = {f"fold-{fold:02d}" for fold in expected_folds}
    if not all(isinstance(record, Mapping) for record in predictions):
        return None, "prediction artifact records are malformed"
    if not all(isinstance(record, Mapping) for record in sidecars):
        return None, "per-gene sidecar records are malformed"
    if {record.get("fold") for record in predictions} != expected_fold_names:
        return None, "prediction fold identities are incomplete or duplicated"
    if {record.get("fold") for record in sidecars} != expected_fold_names:
        return None, "per-gene sidecar fold identities are incomplete or duplicated"
    required_prediction_arrays = {
        "prediction",
        "outer_test_indices",
        "cell_ids",
        "gene_ids",
        "source_cell_indices",
        "source_gene_indices",
        "train_gene_mean",
    }
    artifact_paths: set[str] = set()

    cohort_record = completion.get("cohort_manifest")
    fold_protocol_records = completion.get("fold_protocol_artifacts")
    if not isinstance(cohort_record, Mapping):
        return None, "cohort manifest binding is missing"
    if (
        not isinstance(fold_protocol_records, list)
        or len(fold_protocol_records) != CANONICAL_FOLDS
        or not all(isinstance(record, Mapping) for record in fold_protocol_records)
    ):
        return None, "exactly five fold protocol artifact bindings are required"
    if {record.get("fold") for record in fold_protocol_records} != expected_fold_names:
        return None, "fold protocol artifact identities are incomplete or duplicated"
    try:
        run_relative = run_dir.resolve().relative_to(artifact_root.resolve())
    except ValueError:
        return None, "training run directory is outside the artifact root"
    expected_cohort_path = _normalised_record_path(
        str(run_relative / "cohort-manifest.json")
    )
    try:
        if _normalised_record_path(cohort_record.get("path")) != expected_cohort_path:
            return None, "cohort manifest binding path is not canonical"
        cohort_payload, verified_cohort_record = _read_verified_json_artifact(
            cohort_record,
            artifact_root=artifact_root,
            label="cohort manifest",
        )
    except (TypeError, ValueError, VerificationError) as exc:
        return None, f"cohort manifest binding is invalid: {exc}"
    if not isinstance(cohort_payload, Mapping):
        return None, "cohort manifest must be a JSON object"
    artifact_paths.add(expected_cohort_path)

    verified_fold_protocol: dict[str, dict[str, Any]] = {}
    for fold_record in fold_protocol_records:
        fold_name = str(fold_record.get("fold"))
        if set(fold_record) != {"fold", "split_manifest", "fold_result"}:
            return None, f"fold protocol binding is malformed for {fold_name}"
        split_record = fold_record.get("split_manifest")
        result_record = fold_record.get("fold_result")
        if not isinstance(split_record, Mapping) or not isinstance(
            result_record, Mapping
        ):
            return None, f"split or fold-result binding is missing for {fold_name}"
        expected_split_path = _normalised_record_path(
            str(run_relative / f"{fold_name}-split.json")
        )
        expected_result_path = _normalised_record_path(
            str(run_relative / f"{fold_name}-result.json")
        )
        try:
            if (
                _normalised_record_path(split_record.get("path")) != expected_split_path
                or _normalised_record_path(result_record.get("path"))
                != expected_result_path
            ):
                return (
                    None,
                    f"split or fold-result binding path changed for {fold_name}",
                )
            split_payload, verified_split_record = _read_verified_json_artifact(
                split_record,
                artifact_root=artifact_root,
                label=f"{fold_name} split manifest",
            )
            result_payload, verified_result_record = _read_verified_json_artifact(
                result_record,
                artifact_root=artifact_root,
                label=f"{fold_name} result",
            )
        except (TypeError, ValueError, VerificationError) as exc:
            return None, f"fold protocol binding is invalid for {fold_name}: {exc}"
        if (
            not isinstance(split_payload, Mapping)
            or split_payload.get("fold") != fold_name
        ):
            return None, f"split manifest identity changed for {fold_name}"
        preprocessing = (
            result_payload.get("preprocessing")
            if isinstance(result_payload, Mapping)
            else None
        )
        if (
            not isinstance(result_payload, Mapping)
            or result_payload.get("fold") != fold_name
            or not isinstance(preprocessing, Mapping)
            or set(preprocessing) != {"inner_train_fit", "outer_train_refit"}
            or not all(isinstance(value, Mapping) for value in preprocessing.values())
        ):
            return None, f"fold result or preprocessing binding changed for {fold_name}"
        artifact_paths.update((expected_split_path, expected_result_path))
        verified_fold_protocol[fold_name] = {
            "split_manifest": verified_split_record,
            "split_payload": split_payload,
            "fold_result": verified_result_record,
            "result_payload": result_payload,
        }

    for record in predictions:
        if record.get("schema") != "DepPLM-outer-test-predictions-v1":
            return None, "prediction artifact schema is not recognised"
        arrays = record.get("arrays")
        if not isinstance(arrays, Mapping) or set(arrays) != required_prediction_arrays:
            return None, "prediction artifact array contract is incomplete"
        verified, reason = _verify_artifact_record(record, artifact_root=artifact_root)
        if not verified:
            return None, reason
        artifact_paths.add(_normalised_record_path(record["path"]))
    for record in sidecars:
        verified, reason = _verify_artifact_record(record, artifact_root=artifact_root)
        if not verified:
            return None, reason
        artifact_paths.add(_normalised_record_path(record["path"]))
    if len(artifact_paths) != 1 + 4 * CANONICAL_FOLDS:
        return None, "training artifact paths are not unique"

    run_root = run_dir.resolve()
    for raw_path in artifact_paths:
        path = (artifact_root.resolve() / Path(raw_path)).resolve()
        try:
            path.relative_to(run_root)
        except ValueError:
            return (
                None,
                "training artifact is not contained in its immutable run directory",
            )

    if transaction_runtime is None and not _strict_json_equal(
        validation_runtime, _runtime_record()
    ):
        return None, "runtime changed while accepting the training candidate"
    return (
        {
            "variant": expected_args.variant,
            "seed": int(expected_args.seed),
            "run_directory": str(run_dir.relative_to(artifact_root)),
            "run_manifest": manifest_record,
            "completion": completion_record,
            "outer_test_predictions": predictions,
            "per_gene_prediction_sidecars": sidecars,
            "cohort_manifest": verified_cohort_record,
            "cohort_payload": cohort_payload,
            "fold_protocol_artifacts": verified_fold_protocol,
        },
        "verified",
    )


def _verify_final_campaign_artifact_inventory(
    records: Any,
    *,
    artifact_root: Path,
) -> int:
    """Rehash every canonical training artifact immediately before publication."""
    expected_jobs = [
        (variant.name, seed) for variant in VARIANTS for seed in CANONICAL_SEEDS
    ]
    if not isinstance(records, list) or len(records) != len(expected_jobs):
        raise VerificationError(
            "final campaign inventory must contain the complete canonical job matrix"
        )

    root = artifact_root.resolve()
    expected_folds = [f"fold-{fold:02d}" for fold in range(CANONICAL_FOLDS)]
    observed_paths: set[str] = set()

    def verify_binding(
        record: Any,
        *,
        label: str,
        expected_fields: frozenset[str],
        expected_path: str,
        run_dir: Path,
    ) -> None:
        try:
            recorded_path, _ = _validated_file_record_signature(
                record,
                label=label,
                expected_fields=expected_fields,
            )
        except (TypeError, ValueError) as exc:
            raise VerificationError(f"{label} is malformed: {exc}") from exc
        normalised_expected = _normalised_record_path(expected_path)
        if recorded_path != normalised_expected:
            raise VerificationError(f"{label} path is not canonical")
        if recorded_path in observed_paths:
            raise VerificationError(
                f"final campaign inventory contains a duplicate path: {recorded_path}"
            )
        observed = _verified_artifact_binding(record, artifact_root=root)
        observed_path = _normalised_record_path(observed.get("path"))
        if observed_path != recorded_path:
            raise VerificationError(f"{label} resolved path changed")
        resolved = (root / Path(recorded_path)).resolve()
        try:
            resolved.relative_to(run_dir)
        except ValueError as exc:
            raise VerificationError(
                f"{label} is outside its immutable training directory"
            ) from exc
        observed_paths.add(recorded_path)

    for job_number, ((expected_variant, expected_seed), record) in enumerate(
        zip(expected_jobs, records, strict=True)
    ):
        if not isinstance(record, Mapping) or set(record) != FINAL_CAMPAIGN_JOB_FIELDS:
            raise VerificationError(
                f"final campaign job {job_number} fields are not canonical"
            )
        if (
            record.get("job_number") != job_number
            or record.get("variant") != expected_variant
            or record.get("seed") != expected_seed
            or record.get("source")
            not in {"reused_verified_run", "trained_by_this_invocation"}
        ):
            raise VerificationError(
                f"final campaign job {job_number} identity or source is not canonical"
            )
        if not isinstance(record.get("cohort_payload"), Mapping):
            raise VerificationError(
                f"final campaign job {job_number} cohort payload is malformed"
            )

        run_dir = _safe_training_directory(record, artifact_root=root)
        run_relative = _normalised_record_path(str(run_dir.relative_to(root)))
        if _normalised_record_path(record.get("run_directory")) != run_relative:
            raise VerificationError(
                f"final campaign job {job_number} directory is not canonical"
            )

        verify_binding(
            record.get("run_manifest"),
            label=f"job {job_number} run manifest",
            expected_fields=FILE_RECORD_FIELDS,
            expected_path=f"{run_relative}/run-manifest.json",
            run_dir=run_dir,
        )
        verify_binding(
            record.get("completion"),
            label=f"job {job_number} training completion",
            expected_fields=FILE_RECORD_FIELDS,
            expected_path=f"{run_relative}/completion.json",
            run_dir=run_dir,
        )
        verify_binding(
            record.get("cohort_manifest"),
            label=f"job {job_number} cohort manifest",
            expected_fields=FILE_RECORD_FIELDS,
            expected_path=f"{run_relative}/cohort-manifest.json",
            run_dir=run_dir,
        )

        predictions = record.get("outer_test_predictions")
        sidecars = record.get("per_gene_prediction_sidecars")
        fold_protocol = record.get("fold_protocol_artifacts")
        if (
            not isinstance(predictions, list)
            or len(predictions) != CANONICAL_FOLDS
            or [item.get("fold") for item in predictions if isinstance(item, Mapping)]
            != expected_folds
        ):
            raise VerificationError(
                f"job {job_number} prediction inventory is not canonical"
            )
        if (
            not isinstance(sidecars, list)
            or len(sidecars) != CANONICAL_FOLDS
            or [item.get("fold") for item in sidecars if isinstance(item, Mapping)]
            != expected_folds
        ):
            raise VerificationError(
                f"job {job_number} sidecar inventory is not canonical"
            )
        if (
            not isinstance(fold_protocol, Mapping)
            or list(fold_protocol) != expected_folds
        ):
            raise VerificationError(
                f"job {job_number} fold protocol inventory is not canonical"
            )

        for fold_name, prediction, sidecar in zip(
            expected_folds, predictions, sidecars, strict=True
        ):
            verify_binding(
                prediction,
                label=f"job {job_number} {fold_name} prediction",
                expected_fields=TRAINING_PREDICTION_RECORD_FIELDS,
                expected_path=f"{run_relative}/{fold_name}-predictions.npz",
                run_dir=run_dir,
            )
            verify_binding(
                sidecar,
                label=f"job {job_number} {fold_name} sidecar",
                expected_fields=TRAINING_SIDECAR_RECORD_FIELDS,
                expected_path=(
                    f"{run_relative}/{fold_name}-per-gene-prediction-metrics.csv.gz"
                ),
                run_dir=run_dir,
            )
            protocol_record = fold_protocol[fold_name]
            if not isinstance(protocol_record, Mapping) or set(protocol_record) != {
                "split_manifest",
                "split_payload",
                "fold_result",
                "result_payload",
            }:
                raise VerificationError(
                    f"job {job_number} {fold_name} protocol record is malformed"
                )
            if not isinstance(protocol_record.get("split_payload"), Mapping) or not (
                isinstance(protocol_record.get("result_payload"), Mapping)
            ):
                raise VerificationError(
                    f"job {job_number} {fold_name} protocol payload is malformed"
                )
            verify_binding(
                protocol_record.get("split_manifest"),
                label=f"job {job_number} {fold_name} split manifest",
                expected_fields=FILE_RECORD_FIELDS,
                expected_path=f"{run_relative}/{fold_name}-split.json",
                run_dir=run_dir,
            )
            verify_binding(
                protocol_record.get("fold_result"),
                label=f"job {job_number} {fold_name} fold result",
                expected_fields=FILE_RECORD_FIELDS,
                expected_path=f"{run_relative}/{fold_name}-result.json",
                run_dir=run_dir,
            )

    expected_artifact_count = len(expected_jobs) * (3 + 4 * CANONICAL_FOLDS)
    if len(observed_paths) != expected_artifact_count:
        raise VerificationError(
            "final campaign artifact inventory is incomplete or contains aliases"
        )
    return len(observed_paths)


def _find_reusable_training_run(
    expected_args: argparse.Namespace,
    *,
    current_code_records: Sequence[Mapping[str, Any]],
    current_input_records: Sequence[Mapping[str, Any]],
    artifact_root: Path = OUTPUT_ROOT,
    transaction_runtime: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    if not artifact_root.exists():
        return None, rejected
    for candidate in sorted(artifact_root.iterdir()):
        try:
            candidate_stat = candidate.lstat()
        except OSError as exc:
            rejected.append(
                {
                    "run_directory": candidate.name,
                    "reason": f"candidate directory cannot be stat'ed: {exc}",
                }
            )
            continue
        file_attributes = int(getattr(candidate_stat, "st_file_attributes", 0))
        reparse_flag = int(getattr(stat_module, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
        if stat_module.S_ISLNK(candidate_stat.st_mode) or bool(
            file_attributes & reparse_flag
        ):
            rejected.append(
                {
                    "run_directory": candidate.name,
                    "reason": (
                        "symbolic link or reparse candidate directory is forbidden"
                    ),
                }
            )
            continue
        if not stat_module.S_ISDIR(candidate_stat.st_mode):
            continue
        manifest_path = candidate / "run-manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest, _ = _read_stable_json(manifest_path, relative_to=artifact_root)
        except (OSError, json.JSONDecodeError, VerificationError) as exc:
            rejected.append(
                {
                    "run_directory": candidate.name,
                    "reason": f"run manifest cannot be read: {exc}",
                }
            )
            continue
        if not isinstance(manifest, Mapping):
            rejected.append(
                {
                    "run_directory": candidate.name,
                    "reason": "run manifest must be a JSON object",
                }
            )
            continue
        recorded = manifest.get("arguments")
        if not isinstance(recorded, Mapping):
            rejected.append(
                {
                    "run_directory": candidate.name,
                    "reason": "training arguments are malformed",
                }
            )
            continue
        if (
            manifest.get("schema") != "DepPLM-k-token-pmi-training-v1"
            or recorded.get("variant") != expected_args.variant
            or recorded.get("seed") != expected_args.seed
        ):
            continue
        result, reason = _validate_completed_training_run(
            candidate,
            expected_args=expected_args,
            current_code_records=current_code_records,
            current_input_records=current_input_records,
            artifact_root=artifact_root,
            transaction_runtime=transaction_runtime,
        )
        if result is None:
            rejected.append({"run_directory": candidate.name, "reason": reason})
        else:
            valid.append(result)
    if len(valid) > 1:
        names = [record["run_directory"] for record in valid]
        raise RuntimeError(
            "multiple fully verified runs match one canonical matrix cell; "
            f"refusing outcome selection: {names}"
        )
    return (valid[0] if valid else None), rejected


def run_campaign(args: argparse.Namespace) -> Path:
    """Run or audit the fixed 7-variant x 3-seed x 5-fold model matrix."""
    started = time.time()
    campaign_runtime = _runtime_record()
    jobs = _campaign_jobs(
        device=args.device,
        mutation_chunksize=args.mutation_chunksize,
    )
    current_code_records = [_file_record(path) for path in TRAINING_CODE_PATHS]
    current_input_records = [_file_record(path) for path in _canonical_input_paths()]
    bound_verifier_source = _file_record(VERIFIER_CODE_PATH)
    run_dir = make_unique_run_dir(
        OUTPUT_ROOT,
        protocol="campaign-p1" + ("-plan" if args.plan_only else ""),
        condition="7variants-3seeds-5folds",
        seed=CANONICAL_SEEDS[0],
    )
    plan = {
        "schema": "DepPLM-k-token-pmi-campaign-plan-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arguments": vars(args),
        "runtime": campaign_runtime,
        "training_execution": _training_execution_signature(),
        "code": current_code_records,
        "verifier_source_at_launch": bound_verifier_source,
        "inputs": current_input_records,
        "matrix_definition": {
            "variants": [variant.name for variant in VARIANTS],
            "seeds": list(CANONICAL_SEEDS),
            "outer_folds_per_variant_seed": CANONICAL_FOLDS,
            "n_variant_seed_jobs": len(jobs),
            "n_outer_fold_model_evaluations": len(jobs) * CANONICAL_FOLDS,
            "unit_of_model_evaluation": "one outer-fold fit and exact-once prediction",
        },
        "jobs": [vars(job) for job in jobs],
        "resume_policy": CAMPAIGN_RESUME_POLICY,
        "run_status": "plan_only" if args.plan_only else "in_progress",
    }
    atomic_write_json(run_dir / "campaign-plan.json", plan)
    print(f"[k-token-pmi] campaign output={run_dir}", flush=True)

    initial_reusable: list[dict[str, Any]] = []
    initial_rejected: list[dict[str, Any]] = []
    reusable_by_job: dict[tuple[str, int], dict[str, Any]] = {}
    for job in jobs:
        reusable, rejected = _find_reusable_training_run(
            job,
            current_code_records=current_code_records,
            current_input_records=current_input_records,
            transaction_runtime=campaign_runtime,
        )
        if reusable is not None:
            reusable_by_job[(job.variant, job.seed)] = reusable
            initial_reusable.append(
                {
                    "variant": job.variant,
                    "seed": job.seed,
                    "run_directory": reusable["run_directory"],
                }
            )
        initial_rejected.extend(
            {
                "variant": job.variant,
                "seed": job.seed,
                **record,
            }
            for record in rejected
        )
    atomic_write_json(
        run_dir / "resume-scan.json",
        {
            "schema": "DepPLM-k-token-pmi-resume-scan-v1",
            "reusable_jobs": initial_reusable,
            "rejected_candidates": initial_rejected,
            "n_reusable_jobs": len(initial_reusable),
            "n_reusable_outer_fold_evaluations": (
                len(initial_reusable) * CANONICAL_FOLDS
            ),
        },
    )
    if args.plan_only:
        _verify_current_bindings(
            current_code_records,
            current_records=[_file_record(path) for path in TRAINING_CODE_PATHS],
            label="post-campaign-planning code",
        )
        _verify_current_bindings(
            current_input_records,
            current_records=[
                _file_record(path) for path in _canonical_input_paths()
            ],
            label="post-campaign-planning input",
        )
        _verify_current_bindings(
            [bound_verifier_source],
            current_records=[_file_record(VERIFIER_CODE_PATH)],
            label="post-campaign-planning verifier source",
        )
        if not _strict_json_equal(campaign_runtime, _runtime_record()):
            raise VerificationError("runtime changed during campaign planning")
        atomic_write_json(
            run_dir / "completion.json",
            {
                "schema": "DepPLM-k-token-pmi-campaign-completion-v1",
                "run_status": "plan_only",
                "protocol_complete": False,
                "canonical_campaign_complete": False,
                "n_completed_variant_seed_jobs": len(initial_reusable),
                "n_completed_outer_fold_model_evaluations": (
                    len(initial_reusable) * CANONICAL_FOLDS
                ),
                "n_planned_variant_seed_jobs": len(jobs),
                "n_planned_outer_fold_model_evaluations": (len(jobs) * CANONICAL_FOLDS),
                "model_training_started_by_this_invocation": False,
                "eligible_for_scientific_aggregation": False,
                "scientific_result": False,
                "scientific_result_reason": (
                    "Plan-only mode validates resumability but does not fill missing "
                    "matrix cells or estimate predictive performance."
                ),
                "wall_seconds": time.time() - started,
            },
        )
        return run_dir

    completed_jobs: list[dict[str, Any]] = []
    for job_number, job in enumerate(jobs):
        key = (job.variant, job.seed)
        result = reusable_by_job.get(key)
        source = "reused_verified_run"
        if result is None:
            print(
                f"[k-token-pmi] campaign train variant={job.variant} seed={job.seed}",
                flush=True,
            )
            training_dir = run_training(
                job,
                input_records=current_input_records,
                transaction_runtime=campaign_runtime,
            )
            result, reason = _validate_completed_training_run(
                training_dir,
                expected_args=job,
                current_code_records=current_code_records,
                current_input_records=current_input_records,
                artifact_root=OUTPUT_ROOT,
                transaction_runtime=campaign_runtime,
            )
            if result is None:
                raise RuntimeError(
                    f"new training run failed post-write validation: {reason}"
                )
            source = "trained_by_this_invocation"
        job_record = {"job_number": job_number, "source": source, **result}
        completed_jobs.append(job_record)

    _verify_current_bindings(
        current_code_records,
        current_records=[_file_record(path) for path in TRAINING_CODE_PATHS],
        label="post-campaign code",
    )
    _verify_current_bindings(
        current_input_records,
        current_records=[_file_record(path) for path in _canonical_input_paths()],
        label="post-campaign input",
    )
    if not _strict_json_equal(campaign_runtime, _runtime_record()):
        raise VerificationError("runtime changed during campaign execution")

    final_completed_jobs: list[dict[str, Any]] = []
    for job_number, (job, prior_record) in enumerate(
        zip(jobs, completed_jobs, strict=True)
    ):
        training_dir = _safe_training_directory(
            prior_record,
            artifact_root=OUTPUT_ROOT,
        )
        verified, reason = _validate_completed_training_run(
            training_dir,
            expected_args=job,
            current_code_records=current_code_records,
            current_input_records=current_input_records,
            artifact_root=OUTPUT_ROOT,
            transaction_runtime=campaign_runtime,
        )
        if verified is None:
            raise VerificationError(
                "final campaign artifact revalidation failed for "
                f"variant={job.variant}, seed={job.seed}: {reason}"
            )
        if (
            verified.get("variant") != job.variant
            or verified.get("seed") != job.seed
            or verified.get("run_directory") != prior_record.get("run_directory")
        ):
            raise VerificationError(
                "final campaign artifact revalidation returned a different job identity"
            )
        final_completed_jobs.append(
            {
                "job_number": job_number,
                "source": prior_record["source"],
                **verified,
            }
        )
    completed_jobs = final_completed_jobs
    for job, record in zip(jobs, completed_jobs, strict=True):
        atomic_write_json(
            run_dir
            / f"job-{int(record['job_number']):02d}-{job.variant}-s{int(job.seed)}.json",
            record,
        )

    post_write_completed_jobs: list[dict[str, Any]] = []
    for job_number, (job, prior_record) in enumerate(
        zip(jobs, completed_jobs, strict=True)
    ):
        training_dir = _safe_training_directory(
            prior_record,
            artifact_root=OUTPUT_ROOT,
        )
        verified, reason = _validate_completed_training_run(
            training_dir,
            expected_args=job,
            current_code_records=current_code_records,
            current_input_records=current_input_records,
            artifact_root=OUTPUT_ROOT,
            transaction_runtime=campaign_runtime,
        )
        if verified is None:
            raise VerificationError(
                "post-job-write campaign artifact revalidation failed for "
                f"variant={job.variant}, seed={job.seed}: {reason}"
            )
        post_write_record = {
            "job_number": job_number,
            "source": prior_record["source"],
            **verified,
        }
        if not _strict_json_equal(post_write_record, prior_record):
            raise VerificationError(
                "post-job-write campaign artifact revalidation changed the "
                f"canonical record for variant={job.variant}, seed={job.seed}"
            )
        post_write_completed_jobs.append(post_write_record)
    completed_jobs = post_write_completed_jobs

    predictions = [
        {
            "variant": record["variant"],
            "seed": record["seed"],
            **artifact,
        }
        for record in completed_jobs
        for artifact in record["outer_test_predictions"]
    ]
    sidecars = [
        {
            "variant": record["variant"],
            "seed": record["seed"],
            **artifact,
        }
        for record in completed_jobs
        for artifact in record["per_gene_prediction_sidecars"]
    ]
    matrix_complete = (
        len(completed_jobs) == len(jobs)
        and len(predictions) == len(jobs) * CANONICAL_FOLDS
        and len(sidecars) == len(jobs) * CANONICAL_FOLDS
    )
    completion = {
        "schema": "DepPLM-k-token-pmi-campaign-completion-v1",
        "run_status": "protocol_complete" if matrix_complete else "partial",
        "protocol_complete": matrix_complete,
        "canonical_campaign_complete": matrix_complete,
        "n_completed_variant_seed_jobs": len(completed_jobs),
        "n_completed_outer_fold_model_evaluations": len(predictions),
        "n_planned_variant_seed_jobs": len(jobs),
        "n_planned_outer_fold_model_evaluations": len(jobs) * CANONICAL_FOLDS,
        "verified_training_runs": [
            {
                key: record[key]
                for key in (
                    "variant",
                    "seed",
                    "source",
                    "run_directory",
                    "run_manifest",
                    "completion",
                )
            }
            for record in completed_jobs
        ],
        "outer_test_predictions": predictions,
        "per_gene_prediction_sidecars": sidecars,
        "wall_seconds": None,
        "model_training_started_by_this_invocation": any(
            record["source"] == "trained_by_this_invocation"
            for record in completed_jobs
        ),
        "eligible_for_scientific_aggregation": matrix_complete,
        "scientific_result": False,
        "scientific_result_reason": CAMPAIGN_SCIENTIFIC_RESULT_REASON,
    }
    _verify_current_bindings(
        current_code_records,
        current_records=[_file_record(path) for path in TRAINING_CODE_PATHS],
        label="final post-campaign code",
    )
    _verify_current_bindings(
        current_input_records,
        current_records=[_file_record(path) for path in _canonical_input_paths()],
        label="final post-campaign input",
    )
    _verify_current_bindings(
        [bound_verifier_source],
        current_records=[_file_record(VERIFIER_CODE_PATH)],
        label="final post-campaign verifier source",
    )
    if not _strict_json_equal(campaign_runtime, _runtime_record()):
        raise VerificationError("runtime changed after final campaign revalidation")
    _verify_final_campaign_artifact_inventory(
        completed_jobs,
        artifact_root=OUTPUT_ROOT,
    )
    completion["wall_seconds"] = time.time() - started
    atomic_write_json(run_dir / "completion.json", completion)
    print(
        f"[k-token-pmi] campaign complete outer-folds={len(predictions)}/"
        f"{len(jobs) * CANONICAL_FOLDS}",
        flush=True,
    )
    return run_dir


def _index_training_runs(
    records: Any,
) -> dict[tuple[str, int], Mapping[str, Any]]:
    expected_order = [
        (variant.name, seed) for variant in VARIANTS for seed in CANONICAL_SEEDS
    ]
    expected = set(expected_order)
    if not isinstance(records, list) or len(records) != len(expected):
        raise VerificationError(
            f"campaign completion must contain exactly {len(expected)} training runs"
        )
    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    observed_order: list[tuple[str, int]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise VerificationError("campaign training-run record is not an object")
        if set(record) != TRAINING_RUN_RECORD_FIELDS:
            raise VerificationError(
                "campaign training-run fields do not match the canonical schema"
            )
        variant = record.get("variant")
        seed = record.get("seed")
        if (
            not isinstance(variant, str)
            or not isinstance(seed, int)
            or isinstance(seed, bool)
        ):
            raise VerificationError("campaign training-run identity is malformed")
        if record.get("source") not in {
            "reused_verified_run",
            "trained_by_this_invocation",
        }:
            raise VerificationError("campaign training-run source is not recognised")
        run_directory = record.get("run_directory")
        if not isinstance(run_directory, str) or not run_directory:
            raise VerificationError("campaign training-run directory is malformed")
        for field in ("run_manifest", "completion"):
            try:
                _validated_file_record_signature(
                    record.get(field),
                    label=f"campaign training-run {field}",
                )
            except ValueError as exc:
                raise VerificationError(
                    f"campaign training-run {field} is malformed: {exc}"
                ) from exc
        identity = (variant, seed)
        if identity in indexed:
            raise VerificationError(f"duplicate campaign training run {identity}")
        indexed[identity] = record
        observed_order.append(identity)
    if set(indexed) != expected:
        raise VerificationError("campaign training runs do not cover the fixed matrix")
    if observed_order != expected_order:
        raise VerificationError("campaign training runs are not in canonical order")
    return indexed


def _safe_training_directory(record: Mapping[str, Any], *, artifact_root: Path) -> Path:
    raw = record.get("run_directory")
    if (
        not isinstance(raw, str)
        or not raw
        or Path(raw).is_absolute()
        or ".." in Path(raw).parts
    ):
        raise VerificationError("training run directory must be a relative path")
    root = artifact_root.resolve()
    candidate = root
    try:
        for part in Path(raw).parts:
            if part in ("", "."):
                continue
            candidate = candidate / part
            if _is_link_or_reparse(candidate.lstat()):
                raise VerificationError(
                    "training run directory may not traverse a symbolic link "
                    "or reparse point"
                )
    except OSError as exc:
        raise VerificationError(f"training run directory is missing: {raw}") from exc
    run_dir = candidate.resolve()
    try:
        run_dir.relative_to(root)
    except ValueError as exc:
        raise VerificationError(
            "training run directory escapes the artifact root"
        ) from exc
    if not run_dir.is_dir():
        raise VerificationError(f"training run directory is missing: {raw}")
    return run_dir


def _assert_record_matches(
    recorded: Any,
    observed: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if not isinstance(recorded, Mapping):
        raise VerificationError(f"{label} record is missing")
    for field in ("path", "bytes", "sha256"):
        if not _strict_json_equal(recorded.get(field), observed.get(field)):
            raise VerificationError(f"{label} {field} differs from the verified file")


def _verify_campaign_metadata(
    plan: Any,
    completion: Any,
    *,
    current_code_records: Sequence[Mapping[str, Any]],
    current_input_records: Sequence[Mapping[str, Any]],
    current_runtime: Mapping[str, Any],
) -> tuple[str, int]:
    if not isinstance(plan, Mapping) or not isinstance(completion, Mapping):
        raise VerificationError("campaign plan or completion is not a JSON object")
    if plan.get("schema") != "DepPLM-k-token-pmi-campaign-plan-v1":
        raise VerificationError("campaign plan schema is not recognised")
    if completion.get("schema") != "DepPLM-k-token-pmi-campaign-completion-v1":
        raise VerificationError("campaign completion schema is not recognised")
    expected_plan_fields = {
        "schema",
        "created_utc",
        "arguments",
        "runtime",
        "training_execution",
        "code",
        "verifier_source_at_launch",
        "inputs",
        "matrix_definition",
        "jobs",
        "resume_policy",
        "run_status",
    }
    if set(plan) != expected_plan_fields:
        raise VerificationError("campaign plan fields do not match the canonical schema")
    created_utc = plan.get("created_utc")
    if not isinstance(created_utc, str):
        raise VerificationError("campaign creation timestamp is missing")
    try:
        parsed_created_utc = time.strptime(created_utc, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise VerificationError("campaign creation timestamp is not canonical UTC") from exc
    if time.strftime("%Y-%m-%dT%H:%M:%SZ", parsed_created_utc) != created_utc:
        raise VerificationError("campaign creation timestamp is not canonical UTC")
    if plan.get("run_status") != "in_progress":
        raise VerificationError("campaign plan run status is not canonical")
    if plan.get("resume_policy") != CAMPAIGN_RESUME_POLICY:
        raise VerificationError("campaign resume policy is not canonical")
    arguments = plan.get("arguments")
    if not isinstance(arguments, Mapping):
        raise VerificationError("campaign arguments are missing")
    if (
        arguments.get("command") != "campaign"
        or arguments.get("plan_only") is not False
    ):
        raise VerificationError("verification requires an executed, non-plan campaign")
    device = arguments.get("device")
    chunksize = arguments.get("mutation_chunksize")
    if device not in {"auto", "cpu", "cuda"}:
        raise VerificationError("campaign device argument is invalid")
    if not isinstance(chunksize, int) or isinstance(chunksize, bool) or chunksize < 1:
        raise VerificationError("campaign mutation chunksize is invalid")
    expected_arguments = {
        "command": "campaign",
        "device": device,
        "mutation_chunksize": chunksize,
        "plan_only": False,
    }
    if not _strict_json_equal(arguments, expected_arguments):
        raise VerificationError("campaign arguments differ from the fixed protocol")

    expected_jobs = [
        vars(job) for job in _campaign_jobs(device=device, mutation_chunksize=chunksize)
    ]
    expected_matrix = {
        "variants": [variant.name for variant in VARIANTS],
        "seeds": list(CANONICAL_SEEDS),
        "outer_folds_per_variant_seed": CANONICAL_FOLDS,
        "n_variant_seed_jobs": len(expected_jobs),
        "n_outer_fold_model_evaluations": len(expected_jobs) * CANONICAL_FOLDS,
        "unit_of_model_evaluation": "one outer-fold fit and exact-once prediction",
    }
    if not _strict_json_equal(plan.get("matrix_definition"), expected_matrix):
        raise VerificationError("campaign matrix definition is not the fixed protocol")
    if not _strict_json_equal(plan.get("jobs"), expected_jobs):
        raise VerificationError("campaign jobs differ from the fixed protocol")
    recorded_execution = plan.get("training_execution")
    if recorded_execution is None:
        raise VerificationError(
            "legacy campaign plan lacks a training-execution signature"
        )
    if not _strict_json_equal(recorded_execution, _training_execution_signature()):
        raise VerificationError(
            "campaign training execution signature differs from the current protocol"
        )
    _verify_current_bindings(
        plan.get("code"), current_records=current_code_records, label="campaign code"
    )
    _verify_current_bindings(
        plan.get("inputs"),
        current_records=current_input_records,
        label="campaign input",
    )
    verifier_source = plan.get("verifier_source_at_launch")
    if not isinstance(verifier_source, Mapping):
        raise VerificationError("campaign verifier source binding is missing")
    _verify_current_bindings(
        [verifier_source],
        current_records=[_file_record(VERIFIER_CODE_PATH)],
        label="campaign verifier source",
    )
    if not _strict_json_equal(plan.get("runtime"), current_runtime):
        raise VerificationError(
            "campaign runtime differs from the verification runtime"
        )

    expected_completion_fields = {
        "schema",
        "run_status",
        "protocol_complete",
        "canonical_campaign_complete",
        "n_completed_variant_seed_jobs",
        "n_completed_outer_fold_model_evaluations",
        "n_planned_variant_seed_jobs",
        "n_planned_outer_fold_model_evaluations",
        "verified_training_runs",
        "outer_test_predictions",
        "per_gene_prediction_sidecars",
        "wall_seconds",
        "model_training_started_by_this_invocation",
        "eligible_for_scientific_aggregation",
        "scientific_result",
        "scientific_result_reason",
    }
    if set(completion) != expected_completion_fields:
        raise VerificationError(
            "campaign completion fields do not match the canonical schema"
        )
    required_completion = {
        "run_status": "protocol_complete",
        "protocol_complete": True,
        "canonical_campaign_complete": True,
        "n_completed_variant_seed_jobs": len(expected_jobs),
        "n_completed_outer_fold_model_evaluations": (
            len(expected_jobs) * CANONICAL_FOLDS
        ),
        "n_planned_variant_seed_jobs": len(expected_jobs),
        "n_planned_outer_fold_model_evaluations": (
            len(expected_jobs) * CANONICAL_FOLDS
        ),
        "eligible_for_scientific_aggregation": True,
    }
    for field, expected in required_completion.items():
        observed = completion.get(field)
        if isinstance(expected, bool):
            matches = observed is expected
        elif isinstance(expected, int):
            matches = (
                isinstance(observed, int)
                and not isinstance(observed, bool)
                and observed == expected
            )
        else:
            matches = observed == expected
        if not matches:
            raise VerificationError(
                f"campaign completion field {field} is not canonical"
            )
    if completion.get("scientific_result") is not False:
        raise VerificationError(
            "unverified campaign completion must not claim a scientific result"
        )
    wall_seconds = completion.get("wall_seconds")
    if (
        not isinstance(wall_seconds, (int, float))
        or isinstance(wall_seconds, bool)
        or not np.isfinite(float(wall_seconds))
        or float(wall_seconds) < 0.0
    ):
        raise VerificationError("campaign completion wall time is invalid")
    verified_runs = completion.get("verified_training_runs")
    if not isinstance(verified_runs, list) or len(verified_runs) != len(expected_jobs):
        raise VerificationError("campaign completion training-run inventory is incomplete")
    allowed_sources = {"reused_verified_run", "trained_by_this_invocation"}
    sources = [
        record.get("source") if isinstance(record, Mapping) else None
        for record in verified_runs
    ]
    if any(source not in allowed_sources for source in sources):
        raise VerificationError("campaign completion training-run source is invalid")
    expected_training_started = "trained_by_this_invocation" in sources
    if completion.get("model_training_started_by_this_invocation") is not expected_training_started:
        raise VerificationError(
            "campaign completion training-start flag differs from its run inventory"
        )
    if completion.get("scientific_result_reason") != CAMPAIGN_SCIENTIFIC_RESULT_REASON:
        raise VerificationError("campaign completion scientific-result reason is not canonical")
    return str(device), int(chunksize)


def verify_campaign(
    campaign_dir: Path,
    *,
    artifact_root: Path = OUTPUT_ROOT,
    output_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Replay a complete campaign and write a non-overwriting primary verdict."""
    campaign_dir = _resolve_verified_directory_root(
        campaign_dir, label="campaign directory"
    )
    artifact_root = _resolve_verified_directory_root(
        artifact_root, label="artifact root"
    )
    if output_path is None:
        output_path = _default_detached_report_path(artifact_root)
    else:
        output_path = Path(os.path.abspath(output_path))
    _require_detached_location(
        output_path,
        campaign_dir=campaign_dir,
        artifact_root=artifact_root,
        kind="verification report",
    )
    gates = {
        "campaign_metadata": False,
        "source_and_input_hashes": False,
        "exact_105_identity_coverage": False,
        "training_run_bindings": False,
        "per_fold_preprocessing_manifests": False,
        "artifact_hashes_and_array_contracts": False,
        "per_gene_sidecars_recomputed": False,
        "outer_test_exact_once_and_index_integrity": False,
        "metrics_recomputed_from_predictions": False,
        "paired_descriptive_aggregation": False,
        "end_of_verification_stability": False,
    }
    report: dict[str, Any] = {
        "schema": VERIFICATION_SCHEMA,
        "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "campaign_directory": str(campaign_dir),
        "artifact_root": str(artifact_root),
        "verification_gates": gates,
        "protocol_complete": False,
        "scientific_result": False,
        "scientific_result_scope": (
            "Canonical K/token/PMI outer-test performance with conditional paired "
            "descriptive summaries only; not clinical, biological, causal, or IID "
            "inferential validation."
        ),
        "limitations": [
            "Hash verification establishes byte identity, not independent truth of "
            "the source data or biological provenance.",
            "This primary replay shares implementation code with the training runner; "
            "it is not an implementation-independent verifier.",
            "The verifier intentionally reports no p-values or confidence intervals "
            "because folds and seeds are not treated as IID observations.",
            "Clinical, external-cohort, and independent biological validation are "
            "outside this K/token/PMI verification scope.",
        ],
        "errors": [],
    }
    input_snapshots: dict[Path, Any] = {}
    try:
        verification_runtime = _runtime_record()
        try:
            campaign_dir.relative_to(artifact_root)
        except ValueError as exc:
            raise VerificationError(
                "campaign directory is outside the artifact root"
            ) from exc
        plan_path = campaign_dir / "campaign-plan.json"
        completion_path = campaign_dir / "completion.json"
        plan, plan_record = _read_stable_json(plan_path, relative_to=artifact_root)
        completion, completion_record = _read_stable_json(
            completion_path, relative_to=artifact_root
        )
        verifier_record = _file_record(
            VERIFIER_CODE_PATH, relative_to=VERIFIER_CODE_PATH.parent.parent
        )
        code_records = [_file_record(path) for path in TRAINING_CODE_PATHS]
        input_records, input_snapshots = _snapshot_canonical_inputs(
            _canonical_input_paths()
        )
        device, chunksize = _verify_campaign_metadata(
            plan,
            completion,
            current_code_records=code_records,
            current_input_records=input_records,
            current_runtime=verification_runtime,
        )
        gates["campaign_metadata"] = True
        gates["source_and_input_hashes"] = True

        prediction_records = _index_campaign_artifacts(
            completion.get("outer_test_predictions"),
            label="outer-test predictions",
            expected_fields=PREDICTION_RECORD_FIELDS,
        )
        sidecar_records = _index_campaign_artifacts(
            completion.get("per_gene_prediction_sidecars"),
            label="per-gene prediction sidecars",
            expected_fields=SIDECAR_RECORD_FIELDS,
        )
        training_records = _index_training_runs(
            completion.get("verified_training_runs")
        )
        gates["exact_105_identity_coverage"] = True

        verification_base = _load_verification_base(input_snapshots)
        expected_gene_hash = _hash_strings(verification_base.genes)
        expected_cell_hash = _hash_strings(verification_base.cell_ids)
        metric_rows: list[dict[str, Any]] = []
        fold_metadata_records: list[dict[str, Any]] = []
        coverage: dict[tuple[str, int], list[np.ndarray]] = {}
        verified_artifact_paths: set[str] = set()

        for variant_definition in VARIANTS:
            variant = variant_definition.name
            for seed in CANONICAL_SEEDS:
                job_identity = (variant, seed)
                campaign_run = training_records[job_identity]
                run_dir = _safe_training_directory(
                    campaign_run, artifact_root=artifact_root
                )
                expected_args = _canonical_training_namespace(
                    variant=variant,
                    seed=seed,
                    device=device,
                    mutation_chunksize=chunksize,
                )
                validated, reason = _validate_completed_training_run(
                    run_dir,
                    expected_args=expected_args,
                    current_code_records=code_records,
                    current_input_records=input_records,
                    artifact_root=artifact_root,
                    transaction_runtime=verification_runtime,
                )
                if validated is None:
                    raise VerificationError(
                        f"training run {job_identity} failed validation: {reason}"
                    )
                _assert_record_matches(
                    campaign_run.get("run_manifest"),
                    validated["run_manifest"],
                    label=f"{job_identity} run manifest",
                )
                _assert_record_matches(
                    campaign_run.get("completion"),
                    validated["completion"],
                    label=f"{job_identity} completion",
                )
                manifest, manifest_record = _read_stable_json(
                    run_dir / "run-manifest.json", relative_to=artifact_root
                )
                _assert_record_matches(
                    campaign_run.get("run_manifest"),
                    manifest_record,
                    label=f"{job_identity} stable run manifest",
                )
                runtime = manifest.get("runtime")
                if not isinstance(runtime, Mapping):
                    raise VerificationError(
                        f"training runtime is missing for {job_identity}"
                    )
                for field, expected in verification_runtime.items():
                    if not _strict_json_equal(runtime.get(field), expected):
                        raise VerificationError(
                            f"training runtime field {field} changed for {job_identity}"
                        )
                if runtime.get("device") != corrected._device_from_arg(device):
                    raise VerificationError(
                        f"training runtime device changed for {job_identity}"
                    )
                cohort = validated["cohort_payload"]
                cohort_record = validated["cohort_manifest"]
                expected_cohort_identity = {
                    "n_genes": len(verification_base.genes),
                    "n_cells": len(verification_base.cell_ids),
                    "n_edge_records": int(verification_base.edge_index.shape[0]),
                    "genes_sha256": expected_gene_hash,
                    "cell_ids_sha256": expected_cell_hash,
                    "source_cell_indices": (
                        verification_base.source_cell_indices.tolist()
                    ),
                    "source_gene_indices": (
                        verification_base.source_gene_indices.tolist()
                    ),
                }
                if any(
                    not _strict_json_equal(cohort.get(field), expected)
                    for field, expected in expected_cohort_identity.items()
                ):
                    raise VerificationError(
                        f"cohort identity changed for {job_identity}"
                    )
                fold_metadata_records.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "kind": "cohort_manifest",
                        **cohort_record,
                    }
                )

                completion_predictions = {
                    str(record["fold"]): record
                    for record in validated["outer_test_predictions"]
                }
                completion_sidecars = {
                    str(record["fold"]): record
                    for record in validated["per_gene_prediction_sidecars"]
                }
                splits = make_kfold_outer_splits(
                    len(verification_base.cell_ids), CANONICAL_FOLDS, seed
                )
                coverage[job_identity] = []
                for fold, (outer_train, outer_test) in enumerate(splits):
                    fold_name = f"fold-{fold:02d}"
                    identity = (variant, seed, fold_name)
                    campaign_prediction = prediction_records[identity]
                    campaign_sidecar = sidecar_records[identity]
                    expected_campaign_prediction = {
                        "variant": variant,
                        "seed": seed,
                        **completion_predictions[fold_name],
                    }
                    expected_campaign_sidecar = {
                        "variant": variant,
                        "seed": seed,
                        **completion_sidecars[fold_name],
                    }
                    if not _strict_json_equal(
                        campaign_prediction, expected_campaign_prediction
                    ):
                        raise VerificationError(
                            f"campaign prediction record changed for {identity}"
                        )
                    if not _strict_json_equal(
                        campaign_sidecar, expected_campaign_sidecar
                    ):
                        raise VerificationError(
                            f"campaign sidecar record changed for {identity}"
                        )
                    fold_protocol = validated["fold_protocol_artifacts"][fold_name]
                    split_manifest = fold_protocol["split_payload"]
                    split_record = fold_protocol["split_manifest"]
                    result = fold_protocol["result_payload"]
                    result_record = fold_protocol["fold_result"]
                    _verify_split_manifest(
                        split_manifest,
                        fold_name=fold_name,
                        fold_seed=derive_seed(seed, fold_name),
                        outer_train=outer_train,
                        outer_test=outer_test,
                        base=verification_base,
                    )
                    expected_result_identity = {
                        "fold": fold_name,
                        "fold_number": fold,
                        "condition": variant,
                        "protocol": "p1",
                        "seed": seed,
                        "fold_seed": derive_seed(seed, fold_name),
                        "test_evaluation_calls": 1,
                        "pilot": False,
                    }
                    if any(
                        not _strict_json_equal(result.get(field), expected)
                        for field, expected in expected_result_identity.items()
                    ):
                        raise VerificationError(
                            f"fold result identity/protocol changed for {identity}"
                        )
                    preprocessing_bindings = _verify_fold_preprocessing(
                        result,
                        outer_train=outer_train,
                        fold_seed=derive_seed(seed, fold_name),
                        base=verification_base,
                        variant=variant_definition,
                    )
                    result_prediction = result.get("outer_test_predictions")
                    if not isinstance(result_prediction, Mapping):
                        raise VerificationError(
                            f"fold result prediction binding is missing for {identity}"
                        )
                    result_prediction_path = result_prediction.get("path")
                    if (
                        not isinstance(result_prediction_path, str)
                        or not result_prediction_path
                        or Path(result_prediction_path).is_absolute()
                        or Path(result_prediction_path).parent != Path(".")
                        or Path(result_prediction_path).name
                        != Path(str(campaign_prediction["path"])).name
                    ):
                        raise VerificationError(
                            f"fold result prediction path changed for {identity}"
                        )
                    for field in ("schema", "bytes", "sha256", "arrays"):
                        if not _strict_json_equal(
                            result_prediction.get(field), campaign_prediction.get(field)
                        ):
                            raise VerificationError(
                                f"fold result prediction {field} changed for {identity}"
                            )
                    (
                        metrics,
                        prediction_binding,
                        expected_per_gene,
                    ) = _verify_prediction_artifact(
                        campaign_prediction,
                        artifact_root=artifact_root,
                        base=verification_base,
                        outer_train=outer_train,
                        outer_test=outer_test,
                    )
                    _assert_metrics_match(result.get("test_metrics"), metrics)
                    sidecar_binding = _verify_per_gene_sidecar(
                        campaign_sidecar,
                        expected=expected_per_gene,
                        artifact_root=artifact_root,
                    )
                    coverage[job_identity].append(
                        np.asarray(outer_test, dtype=np.int64)
                    )
                    verified_artifact_paths.add(
                        str(
                            (
                                artifact_root / Path(str(campaign_prediction["path"]))
                            ).resolve()
                        )
                    )
                    verified_artifact_paths.add(
                        str(
                            (
                                artifact_root / Path(str(campaign_sidecar["path"]))
                            ).resolve()
                        )
                    )
                    fold_metadata_records.extend(
                        [
                            {
                                "variant": variant,
                                "seed": seed,
                                "fold": fold_name,
                                "kind": "split_manifest",
                                **split_record,
                            },
                            {
                                "variant": variant,
                                "seed": seed,
                                "fold": fold_name,
                                "kind": "fold_result",
                                **result_record,
                            },
                        ]
                    )
                    paired_metrics: dict[str, float] = {}
                    for name in PAIRED_METRICS:
                        value = float(metrics[name])
                        if not np.isfinite(value):
                            raise VerificationError(
                                f"paired metric {name} is non-finite for {identity}"
                            )
                        paired_metrics[name] = value
                    metric_rows.append(
                        {
                            "variant": variant,
                            "seed": seed,
                            "fold": fold_name,
                            "n_outer_test_cells": int(outer_test.size),
                            "prediction_artifact": prediction_binding,
                            "per_gene_sidecar": sidecar_binding,
                            "preprocessing": preprocessing_bindings,
                            "metrics": paired_metrics,
                        }
                    )

        gates["training_run_bindings"] = True
        gates["per_fold_preprocessing_manifests"] = True
        if len(verified_artifact_paths) != 2 * len(_expected_campaign_identities()):
            raise VerificationError(
                "prediction and sidecar paths are not globally unique"
            )
        gates["artifact_hashes_and_array_contracts"] = True
        gates["per_gene_sidecars_recomputed"] = True
        expected_indices = np.arange(len(verification_base.cell_ids), dtype=np.int64)
        for identity, folds in coverage.items():
            combined = np.concatenate(folds)
            if combined.size != expected_indices.size or not np.array_equal(
                np.sort(combined), expected_indices
            ):
                raise VerificationError(
                    f"outer-test rows are not exact-once for variant/seed {identity}"
                )
        gates["outer_test_exact_once_and_index_integrity"] = True
        gates["metrics_recomputed_from_predictions"] = True
        aggregation = _paired_descriptive_aggregation(metric_rows)
        gates["paired_descriptive_aggregation"] = True

        # Re-hash mutable dependencies and result artifacts at the end so a file
        # changed during verification cannot inherit an earlier successful check.
        _verify_current_bindings(
            code_records,
            current_records=[_file_record(path) for path in TRAINING_CODE_PATHS],
            label="post-verification code",
        )
        _verify_current_bindings(
            [verifier_record],
            current_records=[
                _file_record(
                    VERIFIER_CODE_PATH,
                    relative_to=VERIFIER_CODE_PATH.parent.parent,
                )
            ],
            label="post-verification verifier source",
        )
        _verify_current_bindings(
            input_records,
            current_records=[_file_record(path) for path in _canonical_input_paths()],
            label="post-verification input",
        )
        for label, records in (
            ("outer-test prediction", prediction_records.values()),
            ("per-gene prediction sidecar", sidecar_records.values()),
            ("fold metadata", fold_metadata_records),
        ):
            for record in records:
                _, reason = _verify_artifact_record(record, artifact_root=artifact_root)
                if reason != "verified":
                    identity = (
                        record.get("variant"),
                        record.get("seed"),
                        record.get("fold"),
                    )
                    raise VerificationError(
                        f"{label} changed during verification for {identity}: {reason}"
                    )
        for label, record in (
            ("campaign plan", plan_record),
            ("campaign completion", completion_record),
        ):
            _, reason = _verify_artifact_record(record, artifact_root=artifact_root)
            if reason != "verified":
                raise VerificationError(
                    f"{label} changed during verification: {reason}"
                )
        for identity, record in training_records.items():
            for field in ("run_manifest", "completion"):
                binding = record.get(field)
                if not isinstance(binding, Mapping):
                    raise VerificationError(
                        f"training {field} binding is missing for {identity}"
                    )
                _, reason = _verify_artifact_record(
                    binding, artifact_root=artifact_root
                )
                if reason != "verified":
                    raise VerificationError(
                        f"training {field} changed during verification for "
                        f"{identity}: {reason}"
                    )
        if not _strict_json_equal(verification_runtime, _runtime_record()):
            raise VerificationError("runtime changed during campaign verification")
        gates["end_of_verification_stability"] = True
        if not all(gates.values()):
            raise VerificationError("not every verification gate passed")

        report.update(
            {
                "campaign_plan": plan_record,
                "campaign_completion": completion_record,
                "verified_training_execution": _training_execution_signature(),
                "verified_code": code_records,
                "verified_verifier_source": verifier_record,
                "verified_inputs": input_records,
                "n_verified_variant_seed_jobs": len(training_records),
                "n_verified_outer_fold_evaluations": len(metric_rows),
                "n_verified_prediction_and_sidecar_artifacts": len(
                    verified_artifact_paths
                ),
                "fold_metadata_records": fold_metadata_records,
                "outer_fold_metrics": metric_rows,
                "paired_descriptive_aggregation": aggregation,
                "protocol_complete": True,
                "scientific_result": True,
                "scientific_result_reason": (
                    "All fixed-protocol, identity, training-execution, preprocessing, "
                    "hash, exact-once, metric/sidecar-replay, and paired-descriptive "
                    "gates passed."
                ),
            }
        )
    except Exception as exc:
        report["errors"] = [f"{type(exc).__name__}: {exc}"]
        report["protocol_complete"] = False
        report["scientific_result"] = False
        report["scientific_result_reason"] = (
            "At least one fail-closed verification gate did not pass."
        )
    finally:
        close_errors = _close_input_snapshots(input_snapshots)
    if close_errors:
        report.setdefault("errors", []).extend(
            f"SnapshotCloseError: {error}" for error in close_errors
        )
        gates["end_of_verification_stability"] = False
        report["protocol_complete"] = False
        report["scientific_result"] = False
        report["scientific_result_reason"] = (
            "At least one fail-closed verification gate did not pass."
        )
    _write_detached_json_exclusive(
        output_path,
        report,
        campaign_dir=campaign_dir,
        artifact_root=artifact_root,
        kind="verification report",
    )
    return output_path, report


def _path_is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _default_detached_receipt_path(artifact_root: Path) -> Path:
    root = artifact_root.resolve()
    receipt_root = root.parent / f"{root.name}-detached-verification-receipts"
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return receipt_root / f"receipt-{stamp}-{uuid.uuid4().hex[:8]}.json"


def _default_detached_report_path(artifact_root: Path) -> Path:
    root = artifact_root.resolve()
    report_root = root.parent / f"{root.name}-detached-verification-reports"
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return report_root / f"verification-{stamp}-{uuid.uuid4().hex[:8]}.json"


def _require_detached_location(
    target: Path,
    *,
    campaign_dir: Path,
    artifact_root: Path,
    kind: str,
) -> None:
    if _path_is_within(target, campaign_dir):
        raise VerificationError(
            f"{kind} must be outside the verified campaign directory"
        )
    if _path_is_within(target, artifact_root):
        raise VerificationError(f"{kind} must be outside the campaign artifact root")


def _require_detached_receipt_location(
    target: Path,
    *,
    campaign_dir: Path,
    artifact_root: Path,
) -> None:
    _require_detached_location(
        target,
        campaign_dir=campaign_dir,
        artifact_root=artifact_root,
        kind="detached receipt",
    )


def _write_detached_json_exclusive(
    target: Path,
    payload: Mapping[str, Any],
    *,
    campaign_dir: Path,
    artifact_root: Path,
    kind: str,
) -> None:
    """Publish once and reject any source/target identity drift observed in-call."""
    target = Path(os.path.abspath(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    _require_detached_location(
        target,
        campaign_dir=campaign_dir,
        artifact_root=artifact_root,
        kind=kind,
    )
    ancestors = _ancestor_identity_snapshot(target)
    if target.exists() or target.is_symlink():
        raise VerificationError(f"refusing to overwrite {kind}: {target}")
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
    temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
    published = False
    committed = False
    temporary_identity: os.stat_result | None = None
    published_identity: os.stat_result | None = None
    primary_error: BaseException | None = None
    primary_cause: BaseException | None = None
    try:
        with temporary.open("x+b") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            written = os.fstat(handle.fileno())
            temporary_identity = written
            if (
                not stat_module.S_ISREG(written.st_mode)
                or int(written.st_size) != len(encoded)
            ):
                raise VerificationError(f"temporary {kind} is not a regular file")
            source_before_link = temporary.lstat()
            if (
                _is_link_or_reparse(source_before_link)
                or not stat_module.S_ISREG(source_before_link.st_mode)
                or not os.path.samestat(written, source_before_link)
            ):
                raise VerificationError(f"temporary {kind} identity changed")
            if _ancestor_identity_snapshot(target) != ancestors:
                raise VerificationError(f"{kind} destination ancestors changed")
            _require_detached_location(
                target.resolve(),
                campaign_dir=campaign_dir,
                artifact_root=artifact_root,
                kind=kind,
            )
            try:
                os.link(temporary, target, follow_symlinks=False)
            except FileExistsError as exc:
                raise VerificationError(
                    f"refusing to overwrite {kind}: {target}"
                ) from exc
            published = True
            published_identity = written
            for label, observed in (
                ("temporary source", temporary.lstat()),
                ("published target", target.lstat()),
            ):
                if (
                    _is_link_or_reparse(observed)
                    or not stat_module.S_ISREG(observed.st_mode)
                    or not os.path.samestat(written, observed)
                ):
                    raise VerificationError(f"{kind} {label} identity changed")
            handle.seek(0)
            if handle.read(len(encoded) + 1) != encoded:
                raise VerificationError(f"temporary {kind} content changed")
            after_content_check = os.fstat(handle.fileno())
            if (
                not os.path.samestat(written, after_content_check)
                or int(after_content_check.st_size) != len(encoded)
            ):
                raise VerificationError(f"temporary {kind} identity changed")
            if _ancestor_identity_snapshot(target) != ancestors:
                raise VerificationError(f"{kind} destination ancestors changed")
            _require_detached_location(
                target.resolve(),
                campaign_dir=campaign_dir,
                artifact_root=artifact_root,
                kind=kind,
            )
            final_target = target.lstat()
            if (
                _is_link_or_reparse(final_target)
                or not stat_module.S_ISREG(final_target.st_mode)
                or not os.path.samestat(written, final_target)
            ):
                raise VerificationError(f"{kind} published target identity changed")
            committed = True
    except BaseException as exc:
        if isinstance(exc, OSError):
            primary_error = VerificationError(f"{kind} cannot be published: {target}")
            primary_cause = exc
        else:
            primary_error = exc
        if published and published_identity is not None and hasattr(
            primary_error, "add_note"
        ):
            primary_error.add_note(
                "The target is not removed automatically after a failed post-link "
                "check because pathname deletion could remove a concurrent replacement."
            )

    cleanup_error: OSError | None = None
    try:
        if committed and temporary_identity is not None and (
            temporary.exists() or temporary.is_symlink()
        ):
            observed_temporary = temporary.lstat()
            if (
                temporary_identity is not None
                and (
                    _is_link_or_reparse(observed_temporary)
                    or not stat_module.S_ISREG(observed_temporary.st_mode)
                    or not os.path.samestat(temporary_identity, observed_temporary)
                )
            ):
                raise OSError(f"temporary {kind} path was replaced before cleanup")
            temporary.unlink(missing_ok=True)
    except OSError as exc:
        cleanup_error = exc

    if primary_error is not None:
        if cleanup_error is not None and hasattr(primary_error, "add_note"):
            primary_error.add_note(
                f"Temporary {kind} cleanup also failed: {cleanup_error}"
            )
        if primary_cause is not None:
            raise primary_error from primary_cause
        raise primary_error
    if not committed:
        raise VerificationError(f"{kind} publication did not reach commit")
    if cleanup_error is not None:
        print(
            f"[{kind}] published successfully; temporary cleanup failed: "
            f"{cleanup_error}",
            file=sys.stderr,
            flush=True,
        )


def _write_detached_receipt_exclusive(
    target: Path,
    receipt: Mapping[str, Any],
    *,
    campaign_dir: Path,
    artifact_root: Path,
) -> None:
    """Best-effort race-hardened, no-overwrite receipt publication."""
    _write_detached_json_exclusive(
        target,
        receipt,
        campaign_dir=campaign_dir,
        artifact_root=artifact_root,
        kind="detached receipt",
    )


def _write_detached_verification_receipt(
    *,
    report_path: Path,
    report: Mapping[str, Any],
    campaign_dir: Path,
    artifact_root: Path,
    receipt_path: Path | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Bind a saved verification report from outside the mutable campaign tree."""
    campaign_dir = _resolve_verified_directory_root(
        campaign_dir, label="campaign directory"
    )
    artifact_root = _resolve_verified_directory_root(
        artifact_root, label="artifact root"
    )
    report_path = Path(os.path.abspath(report_path))
    _require_detached_location(
        report_path,
        campaign_dir=campaign_dir,
        artifact_root=artifact_root,
        kind="verification report",
    )
    target = Path(
        os.path.abspath(
            _default_detached_receipt_path(artifact_root)
            if receipt_path is None
            else receipt_path
        )
    )
    _require_detached_receipt_location(
        target, campaign_dir=campaign_dir, artifact_root=artifact_root
    )
    saved_report, report_record = _read_stable_json(
        report_path,
        relative_to=report_path.parent,
    )
    if not isinstance(saved_report, Mapping) or not _strict_json_equal(
        saved_report, report
    ):
        raise VerificationError(
            "saved verification report differs from the in-memory verdict"
        )
    report_record["path"] = str(report_path)
    verifier_record = _file_record(
        VERIFIER_CODE_PATH,
        relative_to=VERIFIER_CODE_PATH.parent.parent,
    )
    receipt: dict[str, Any] = {
        "schema": DETACHED_RECEIPT_SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verification_report": report_record,
        "campaign_directory": str(campaign_dir),
        "artifact_root": str(artifact_root),
        "campaign_plan": saved_report.get("campaign_plan"),
        "campaign_completion": saved_report.get("campaign_completion"),
        "protocol_complete": saved_report.get("protocol_complete") is True,
        "scientific_result": saved_report.get("scientific_result") is True,
        "primary_verifier_source": verifier_record,
        "primary_verifier_mode": "read_only_replay_using_training_runner_module",
        "limitations": [
            "This detached receipt binds exact report bytes but does not make the "
            "storage immutable or establish independent source-data truth.",
            "The primary verifier replays saved evidence in a read-only command but "
            "shares implementation code with the training runner; implementation-"
            "independent core checks require verify_k_token_pmi_sensitivity.py.",
            "Path, opened-handle, and ancestor identity checks are best-effort; "
            "portable user-space checks cannot eliminate every same-account "
            "concurrent filesystem race.",
        ],
    }
    _write_detached_receipt_exclusive(
        target,
        receipt,
        campaign_dir=campaign_dir,
        artifact_root=artifact_root,
    )
    receipt_record = _file_record(target, relative_to=target.parent)
    receipt_record["path"] = str(target)
    return target, receipt, receipt_record


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _fold_id(value: str) -> int:
    parsed = int(value)
    if parsed < -1:
        raise argparse.ArgumentTypeError("must be -1 or a non-negative integer")
    return parsed


def _inner_fraction(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or not 0.0 < parsed < 0.5:
        raise argparse.ArgumentTypeError("must be finite and between 0 and 0.5")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="run label-free preprocessing audit")
    audit.add_argument("--seeds", type=int, nargs="+", default=list(CANONICAL_SEEDS))
    audit.add_argument("--folds", type=_positive_int, default=CANONICAL_FOLDS)
    audit.add_argument("--fold-id", type=_fold_id, default=-1)
    audit.add_argument("--top-mut", type=_positive_int, default=5_000)
    audit.add_argument("--mutation-chunksize", type=_positive_int, default=250_000)
    audit.add_argument("--pmi-chunk-size", type=_positive_int, default=20_000)

    train = subparsers.add_parser(
        "train", help="train one variant and save predictions"
    )
    train.add_argument(
        "--variant", choices=[item.name for item in VARIANTS], required=True
    )
    train.add_argument(
        "--protocol",
        choices=["p1", "transfer-forward", "transfer-reverse"],
        default="p1",
    )
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--folds", type=_positive_int, default=CANONICAL_FOLDS)
    train.add_argument("--fold-id", type=_fold_id, default=-1)
    train.add_argument("--inner-val-fraction", type=_inner_fraction, default=0.2)
    train.add_argument("--inner-max-epochs", type=_positive_int, default=20)
    train.add_argument("--patience", type=_positive_int, default=5)
    train.add_argument("--batch-size", type=_positive_int, default=64)
    train.add_argument("--gene-subsample", type=int, default=6_000)
    train.add_argument("--top-mut", type=_positive_int, default=5_000)
    train.add_argument("--n-expr-pc", type=_positive_int, default=128)
    train.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    train.add_argument("--pilot", action="store_true")
    train.add_argument("--pilot-cells", type=_positive_int, default=96)
    train.add_argument("--pilot-genes", type=_positive_int, default=128)
    train.add_argument("--mutation-chunksize", type=_positive_int, default=250_000)

    campaign = subparsers.add_parser(
        "campaign",
        help="run the fixed 7-variant x 3-seed x 5-fold model campaign",
    )
    campaign.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="cuda",
    )
    campaign.add_argument(
        "--mutation-chunksize",
        type=_positive_int,
        default=250_000,
    )
    campaign.add_argument(
        "--plan-only",
        action="store_true",
        help="write and validate the resume plan without starting model training",
    )
    verify = subparsers.add_parser(
        "verify",
        help=(
            "replay and aggregate a complete fixed campaign in a read-only process "
            "without training"
        ),
    )
    verify.add_argument("--campaign-dir", type=Path, required=True)
    verify.add_argument("--artifact-root", type=Path, default=OUTPUT_ROOT)
    verify.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "non-overwriting verification JSON path outside the artifact root "
            "(default: unique sibling report directory)"
        ),
    )
    verify.add_argument(
        "--receipt-output",
        type=Path,
        default=None,
        help=(
            "detached receipt path outside the artifact root "
            "(default: unique sibling receipt directory)"
        ),
    )
    args = parser.parse_args(argv)
    if getattr(args, "folds", 2) < 2:
        parser.error("--folds must be at least 2")
    if getattr(args, "gene_subsample", 0) < 0:
        parser.error("--gene-subsample must be non-negative")
    if len(set(getattr(args, "seeds", []))) != len(getattr(args, "seeds", [])):
        parser.error("--seeds must be unique")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "audit":
            run_audit(args)
        elif args.command == "campaign":
            run_campaign(args)
        elif args.command == "train":
            run_training(args)
        else:
            output_path, report = verify_campaign(
                args.campaign_dir,
                artifact_root=args.artifact_root,
                output_path=args.output,
            )
            receipt_path, _, receipt_record = _write_detached_verification_receipt(
                report_path=output_path,
                report=report,
                campaign_dir=args.campaign_dir,
                artifact_root=args.artifact_root,
                receipt_path=args.receipt_output,
            )
            report_record = _file_record(output_path, relative_to=output_path.parent)
            print(
                f"[k-token-pmi] verification output={output_path} "
                f"report_sha256={report_record['sha256']} "
                f"detached_receipt={receipt_path} "
                f"receipt_sha256={receipt_record['sha256']} "
                f"scientific_result={report['scientific_result']}",
                flush=True,
            )
            return 0 if report["scientific_result"] else 2
    except Exception as exc:
        message = " ".join(str(exc).split()) or "unspecified error"
        print(
            f"[k-token-pmi] {args.command} failed: {type(exc).__name__}: {message}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
