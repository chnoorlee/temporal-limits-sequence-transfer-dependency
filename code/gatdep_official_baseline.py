"""Audit GATDep and run a leakage-resistant corrected secondary benchmark.

This module deliberately separates two objects:

* ``official_reproduction`` is the upstream workflow at its pinned commit.  It
  uses cohort-wide assets and evaluates the held-out fold every epoch, so it is
  not eligible for the corrected DepPLM comparison.
* ``GATDep-corrected-adaptation`` reuses the pinned graph, gene-set membership,
  and exact upstream model class, but trains on the common DepMap endpoints
  under nested outer evaluation.  Its pathway features are an explicitly named
  train-z-scored pathway-mean approximation, not official GSVA.

Audit and smoke modes are CPU-only.  Formal local training is never started
unless the pinned source, runtime, common cohort, and README-license-evidence
gate pass the fail-closed preflight checks.  The fixed README's explicit MIT
badge and statement are recorded only as the basis for a non-redistributive
local scientific audit; the absent license text remains a separate,
non-overridable redistribution blocker.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import types
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


OFFICIAL_REPOSITORY = "https://github.com/dming1024/GATDep"
OFFICIAL_COMMIT = "047ffdc2df8fafe58dd50d91293cd0e0cb7cb0a1"
OFFICIAL_PAPER_DOI = "10.1186/s12967-025-07501-3"
OFFICIAL_PAPER_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC12805727/"
AUDIT_SCHEMA = "GATDep-official-audit-v2"
RUN_SCHEMA = "GATDep-corrected-adaptation-run-v2"
PREDICTION_SCHEMA = "GATDep-corrected-outer-test-predictions-v1"
FEATURE_METHOD = "outer-train-zscore-pathway-mean-v1"
LOCAL_EXECUTION_LICENSE_STATUS = (
    "eligible_for_local_nonredistributive_audit_from_readme_mit_assertion"
)
REDISTRIBUTION_LICENSE_STATUS = "blocked_missing_license_text"
LOCAL_EXECUTION_LICENSE_BLOCKER = (
    "fixed official README must retain both its MIT badge and explicit MIT "
    "statement before a local non-redistributive scientific audit can run"
)
REDISTRIBUTION_LICENSE_BLOCKER = (
    "fixed official tree has no LICENSE/COPYING/NOTICE text at the pinned commit; "
    "author/legal clarification is required before redistributing upstream code, "
    "weights, or derivative release materials"
)
SCHEDULER_CONFIGURATION = {
    "name": "ReduceLROnPlateau",
    "monitor": "inner_validation_mse",
    "mode": "min",
    "factor": 0.5,
    "patience": 3,
    "refit_policy": "freeze_pre_epoch_learning_rates_through_selected_epoch",
}

EXPECTED_RUNTIME = {
    "python": "3.10.14",
    "numpy": "2.2.5",
    "pandas": "2.2.3",
    "scikit-learn": "1.6.1",
    "scipy": "1.15.3",
    "torch": "2.7.0",
    "torch-geometric": "2.6.1",
}
TORCH_VERSION_PATTERN = re.compile(
    r"(?P<base>(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*))"
    r"(?:\+(?P<tag>[a-z0-9]+(?:[._-][a-z0-9]+)*))?",
    re.ASCII,
)
UPSTREAM_PLATFORM_CONTRACT = {
    "operating_system": "linux-64",
    "cuda_runtime_major_minor": "12.6",
    "source": "pinned environment.yml",
}

EXPECTED_BLOBS: dict[str, tuple[str, int]] = {
    "README.md": ("e58b6a5c8ce5df43103801a3db7ccbd23e29a64f", 2_836),
    "GATDep.ipynb": ("c76458ac8993b52435fc5163059b54b9ea0b5524", 53_868),
    "GATDep.py": ("3a22469d710d7ce77ec71bf0ac2522922d711aa1", 6_406),
    "GeneDependencyGAT.py": (
        "4a9ba8ed73fcd00cf1c2fccad9753e3edb0f5c89",
        3_134,
    ),
    "geneExpression2GSVA.R": (
        "c818d2a1312a4adff891f455f8d8e16906d86a77",
        999,
    ),
    "environment.yml": ("765a410ca959bddddffe046db5f3546a5d62438e", 4_138),
    "requirements.txt": ("70baa53c62b7c33fa3b5d3359fd0d620b148bf20", 5_221),
    "datasets/GSE272107_gsva.txt": (
        "55e1f6204a8eb42dc248167a5513091fcf8d135e",
        1_533_557,
    ),
    "datasets/GSE272107_predictions.csv": (
        "80ac2fc1ce20f14fd5cda888c24e62574da61bd6",
        373_645,
    ),
    "datasets/adj_gene_geneset.npy": (
        "fb3ebcdc4bec600d58526b38f14900a0894e054d",
        61_673_528,
    ),
    "datasets/crispr_score_v4_20250423_scale_in_each_sample.txt": (
        "1b8848fa26351142d29d87cbc9e255ed94d12606",
        77_624_333,
    ),
    "datasets/genes2ID_2k.json": (
        "c926cac3f6b04d37e5f12d408ffcbcbc86f7bdf0",
        43_745,
    ),
    "datasets/genesets_select_6k.json": (
        "1d153ad142d38ca1aa5a87b2e87e5143d4f70f75",
        381_299,
    ),
    "datasets/ggi_id.txt": (
        "36d40f98ae7394f168ca8650b6dbe52ccd8fa0a4",
        7_162_751,
    ),
    "datasets/gsva_score_go_kegg_v3_format.npy": (
        "bb0962e1868804a29f91bd75de2847cdc1b4563f",
        28_553_600,
    ),
    "datasets/samples2ID.json": (
        "2a9541f1a27390624062c12a31de66b6b829ae54",
        25_004,
    ),
    "datasets/signature_go_kegg_set_v2.rds": (
        "2ec6fb740e1276dfcf555b914d48d59941f62582",
        1_889_130,
    ),
    "trained_models/GATDep/best_model_v1_basedon_v4_scale_data.pt": (
        "5e6dca26572453f874e8e6ea131fdfd6fac155de",
        13_749_453,
    ),
    "trained_models/GATDep/models/best_model_k_1_basedon_v4_scale_data.pt": (
        "3764ab5360ed55f73d7cec30e5e3549e3da69da5",
        13_749_477,
    ),
    "trained_models/GATDep/models/best_model_k_2_basedon_v4_scale_data.pt": (
        "62b5670229f242f87f89561470bef63b7d385e8e",
        13_749_477,
    ),
    "trained_models/GATDep/models/best_model_k_3_basedon_v4_scale_data.pt": (
        "cb03fade0a477a94556ede01a8141e512fef11ea",
        13_749_477,
    ),
    "trained_models/GATDep/models/best_model_k_4_basedon_v4_scale_data.pt": (
        "464b2970702eabf181fd95c23ab1cbfce6d0a70e",
        13_749_477,
    ),
    "trained_models/GATDep/models/best_model_k_5_basedon_v4_scale_data.pt": (
        "4b77a64d1f35e16bb69fa9d478df9f93947b16fc",
        13_749_477,
    ),
    "trained_models/GATDep/models/best_model_k_6_basedon_v4_scale_data.pt": (
        "23770a33cc1cdb1e0faddaa8e5da498e17e046b6",
        13_749_477,
    ),
    "trained_models/GATDep/models/best_model_k_7_basedon_v4_scale_data.pt": (
        "42d0ce7f46809cb83764a7d242f794411b7630a9",
        13_749_477,
    ),
    "trained_models/GATDep/models/best_model_k_8_basedon_v4_scale_data.pt": (
        "7a26a9b4fd1af9559fb962cbe4ce8a32814c014d",
        13_749_477,
    ),
    "trained_models/GATDep/models/best_model_k_9_basedon_v4_scale_data.pt": (
        "10d4d7e609802ad1998b8ade1e67355d135c0018",
        13_749_477,
    ),
    "trained_models/GATDep/models/best_model_k_10_basedon_v4_scale_data.pt": (
        "10a29e21d31b4b67bd39cb1a3e308e0a6910ee8a",
        13_749_501,
    ),
}

PROJECT_INPUTS = (
    "data/processed/cell_lines.txt",
    "data/processed/genes.txt",
    "data/processed/lineages.txt",
    "data/processed/gene_effect.npy",
    "data/depmap/OmicsExpressionProteinCodingGenesTPMLogp1.csv",
)
RUN_SOURCE_PATHS = (
    "code/gatdep_official_baseline.py",
    "code/verify_gatdep_official_baseline.py",
)
OFFICIAL_RUN_INPUTS = (
    "datasets/adj_gene_geneset.npy",
    "datasets/genes2ID_2k.json",
    "datasets/genesets_select_6k.json",
    "datasets/samples2ID.json",
    "datasets/ggi_id.txt",
    "GeneDependencyGAT.py",
)


class AuditError(RuntimeError):
    """Raised when immutable upstream evidence does not match the contract."""


class AtomicWriteError(RuntimeError):
    """Raised when an immutable output already exists."""


@dataclass(frozen=True)
class CommonCohort:
    cell_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]
    lineages: np.ndarray
    project_cell_indices: np.ndarray
    project_gene_indices: np.ndarray
    official_cell_indices: np.ndarray
    official_gene_indices: np.ndarray

    def contract(self) -> dict[str, Any]:
        record = {
            "ordering": "project-manifest-order-filtered-to-official-intersection",
            "n_cells": len(self.cell_ids),
            "n_genes": len(self.gene_ids),
            "cell_ids_sha256": hash_strings(self.cell_ids),
            "gene_ids_sha256": hash_strings(self.gene_ids),
            "project_cell_indices_sha256": hash_array(
                self.project_cell_indices.astype("<i8", copy=False)
            ),
            "project_gene_indices_sha256": hash_array(
                self.project_gene_indices.astype("<i8", copy=False)
            ),
            "official_cell_indices_sha256": hash_array(
                self.official_cell_indices.astype("<i8", copy=False)
            ),
            "official_gene_indices_sha256": hash_array(
                self.official_gene_indices.astype("<i8", copy=False)
            ),
        }
        return {**record, "sha256": canonical_hash(record)}


@dataclass(frozen=True)
class FittedPathwayTransform:
    """Train-fitted expression scaler plus fixed annotation aggregation."""

    mean_: np.ndarray
    scale_: np.ndarray
    pathway_weights_: np.ndarray
    fit_indices_: np.ndarray

    def transform(self, expression: np.ndarray) -> np.ndarray:
        values = np.asarray(expression, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.mean_.size:
            raise ValueError("expression has an incompatible feature dimension")
        clean = np.where(np.isfinite(values), values, self.mean_)
        standardized = (clean - self.mean_) / self.scale_
        return (standardized @ self.pathway_weights_).astype(np.float32, copy=False)

    def provenance(self) -> dict[str, Any]:
        return {
            "method": FEATURE_METHOD,
            "fit_scope": "provided_training_indices_only",
            "fit_indices_sha256": hash_array(
                self.fit_indices_.astype("<i8", copy=False)
            ),
            "mean_sha256": hash_array(self.mean_),
            "scale_sha256": hash_array(self.scale_),
            "pathway_weights_sha256": hash_array(self.pathway_weights_),
            "n_expression_genes": int(self.mean_.size),
            "n_pathways": int(self.pathway_weights_.shape[1]),
            "material_deviation": (
                "This is a train-z-scored pathway mean over expression-available "
                "official GATDep genes. It is not GSVA and must not be labeled as "
                "the upstream feature generator."
            ),
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1_bytes(source: bytes) -> str:
    """Hash already-read source bytes exactly as Git stores a blob."""

    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(source)}\0".encode("ascii"))
    digest.update(source)
    return digest.hexdigest()


def git_blob_sha1_file(path: str | Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    """Hash checkout bytes exactly as a Git blob, without clean/smudge filters."""

    source = Path(path)
    size = source.stat().st_size
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {size}\0".encode("ascii"))
    with source.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_pinned_text_source(path: Path, expected_blob: str) -> bytes:
    """Return canonical Git bytes, allowing only Git's CRLF checkout smudge."""

    source = path.read_bytes()
    if git_blob_sha1_bytes(source) == expected_blob:
        return source
    normalized = source.replace(b"\r\n", b"\n")
    if normalized != source and git_blob_sha1_bytes(normalized) == expected_blob:
        return normalized
    raise AuditError("loaded official model source bytes differ from pinned blob")


def file_record(path: str | Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = Path(path)
    display = resolved
    if relative_to is not None:
        try:
            display = resolved.relative_to(relative_to)
        except ValueError:
            pass
    return {
        "path": display.as_posix(),
        "bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def hash_strings(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def hash_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp-{uuid.uuid4().hex}")
    payload = json.dumps(
        _json_compatible(value),
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise AtomicWriteError(f"refusing to overwrite {target}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise AtomicWriteError(f"refusing to overwrite {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch_save(path: Path, state: Mapping[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise AtomicWriteError(f"refusing to overwrite {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AuditError(f"git {' '.join(args)} failed in {root}: {exc}") from exc


def ordered_mapping(path: Path) -> tuple[list[str], dict[str, int]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise AuditError(f"{path} must contain a non-empty object")
    mapping = {str(key): int(value) for key, value in raw.items()}
    if len(mapping) != len(raw) or len(set(mapping.values())) != len(mapping):
        raise AuditError(f"{path} contains duplicate names or indices")
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise AuditError(f"{path} indices are not contiguous from zero")
    ordered = [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])]
    return ordered, mapping


def build_common_cohort(project_root: Path, official_root: Path) -> CommonCohort:
    processed = project_root / "data" / "processed"
    project_cells = read_lines(processed / "cell_lines.txt")
    project_genes = read_lines(processed / "genes.txt")
    project_lineages = np.asarray(read_lines(processed / "lineages.txt"), dtype=str)
    official_cells, official_cell_map = ordered_mapping(
        official_root / "datasets" / "samples2ID.json"
    )
    official_genes, official_gene_map = ordered_mapping(
        official_root / "datasets" / "genes2ID_2k.json"
    )
    if len(project_lineages) != len(project_cells):
        raise AuditError("project lineages do not align with cell_lines.txt")

    official_cell_set = set(official_cells)
    official_gene_set = set(official_genes)
    project_cell_indices = np.asarray(
        [
            index
            for index, cell in enumerate(project_cells)
            if cell in official_cell_set
        ],
        dtype=np.int64,
    )
    project_gene_indices = np.asarray(
        [
            index
            for index, gene in enumerate(project_genes)
            if gene in official_gene_set
        ],
        dtype=np.int64,
    )
    cell_ids = tuple(project_cells[index] for index in project_cell_indices)
    gene_ids = tuple(project_genes[index] for index in project_gene_indices)
    return CommonCohort(
        cell_ids=cell_ids,
        gene_ids=gene_ids,
        lineages=project_lineages[project_cell_indices],
        project_cell_indices=project_cell_indices,
        project_gene_indices=project_gene_indices,
        official_cell_indices=np.asarray(
            [official_cell_map[cell] for cell in cell_ids], dtype=np.int64
        ),
        official_gene_indices=np.asarray(
            [official_gene_map[gene] for gene in gene_ids], dtype=np.int64
        ),
    )


def _parse_torch_version(value: Any) -> tuple[str, str | None] | None:
    """Parse the exact PyTorch version forms accepted by the runtime contract."""

    if not isinstance(value, str):
        return None
    match = TORCH_VERSION_PATTERN.fullmatch(value)
    if match is None:
        return None
    return match.group("base"), match.group("tag")


def current_runtime() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in EXPECTED_RUNTIME:
        if name == "python":
            continue
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    torch_build: dict[str, Any]
    try:
        import torch

        module_version = str(torch.__version__)
        module_parts = _parse_torch_version(module_version)
        cuda_available = bool(torch.cuda.is_available())
        get_arch_list = getattr(torch.cuda, "get_arch_list", None)
        compiled_architectures = (
            sorted(set(str(value) for value in get_arch_list()))
            if callable(get_arch_list)
            else []
        )
        devices = []
        for index in range(torch.cuda.device_count()):
            capability = list(torch.cuda.get_device_capability(index))
            architecture = f"sm_{capability[0]}{capability[1]}"
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": capability,
                    "architecture": architecture,
                    "kernel_supported_by_build": (
                        architecture in compiled_architectures
                    ),
                }
            )
        torch_build = {
            "module_version": module_version,
            "base_version": module_parts[0] if module_parts is not None else None,
            "build_tag": module_parts[1] if module_parts is not None else None,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": cuda_available,
            "device_count": int(torch.cuda.device_count()),
            "compiled_architectures": compiled_architectures,
            "devices": devices,
            "cuda_execution_compatible": bool(
                cuda_available
                and any(device["kernel_supported_by_build"] for device in devices)
            ),
        }
    except (ImportError, RuntimeError) as exc:
        torch_build = {"error": f"{type(exc).__name__}: {exc}"}
    runtime = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "platform_system": platform.system(),
        "machine": platform.machine(),
        "packages": packages,
        "rscript": shutil.which("Rscript"),
        "torch_build": torch_build,
    }
    runtime["fidelity"] = runtime_fidelity(runtime)
    return runtime


def runtime_fidelity(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Classify runtime fidelity without equating a local build with upstream."""
    packages = runtime.get("packages", {})
    distribution_versions_exact = not runtime_mismatches(runtime)
    torch_build = runtime.get("torch_build", {})
    module_base = (
        torch_build.get("base_version") if isinstance(torch_build, Mapping) else None
    )
    cuda_runtime = (
        torch_build.get("cuda_runtime") if isinstance(torch_build, Mapping) else None
    )
    platform_system = str(runtime.get("platform_system", ""))
    upstream_platform_compatible = (
        platform_system == "Linux"
        and module_base == EXPECTED_RUNTIME["torch"]
        and cuda_runtime == UPSTREAM_PLATFORM_CONTRACT["cuda_runtime_major_minor"]
    )
    cuda_device_kernel_compatible = bool(
        isinstance(torch_build, Mapping)
        and torch_build.get("cuda_execution_compatible") is True
    )
    return {
        "distribution_versions_exact": distribution_versions_exact,
        "torch_distribution_version": (
            packages.get("torch") if isinstance(packages, Mapping) else None
        ),
        "upstream_platform_contract": dict(UPSTREAM_PLATFORM_CONTRACT),
        "upstream_platform_compatible": upstream_platform_compatible,
        "cuda_device_kernel_compatible": cuda_device_kernel_compatible,
        "classification": (
            "upstream_declared_platform_compatible_not_bit_exact"
            if upstream_platform_compatible
            else "platform_adapted_local_runtime_not_upstream_bit_exact"
        ),
        "upstream_bit_exact_environment_claimed": False,
        "interpretation": (
            "Exact distribution base versions are an execution gate. PyTorch build "
            "tags, CUDA runtime, operating system, and GPU are recorded separately; "
            "a platform-adapted run is never described as the upstream bit-exact "
            "environment."
        ),
    }


def _official_source_audit(official_root: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    head = _git(official_root, "rev-parse", "HEAD")
    if head != OFFICIAL_COMMIT:
        errors.append(f"HEAD is {head}, expected {OFFICIAL_COMMIT}")
    dirty = bool(_git(official_root, "status", "--porcelain", "--untracked-files=no"))
    if dirty:
        errors.append("official checkout has tracked modifications")

    blobs: dict[str, Any] = {}
    for relative, (expected_blob, expected_bytes) in EXPECTED_BLOBS.items():
        path = official_root / relative
        record: dict[str, Any] = {
            "expected_git_blob": expected_blob,
            "expected_repository_bytes": expected_bytes,
            "exists_in_checkout": path.is_file(),
        }
        try:
            observed_blob = _git(
                official_root, "rev-parse", f"{OFFICIAL_COMMIT}:{relative}"
            )
            observed_bytes = int(_git(official_root, "cat-file", "-s", observed_blob))
            record.update(
                {
                    "git_blob": observed_blob,
                    "repository_bytes": observed_bytes,
                    "git_blob_match": observed_blob == expected_blob,
                    "repository_bytes_match": observed_bytes == expected_bytes,
                }
            )
            if observed_blob != expected_blob or observed_bytes != expected_bytes:
                errors.append(f"immutable blob mismatch: {relative}")
        except (AuditError, ValueError) as exc:
            record["error"] = str(exc)
            errors.append(f"cannot verify {relative}: {exc}")
        if path.is_file():
            record["checkout"] = file_record(path, relative_to=official_root)
        else:
            errors.append(f"checkout file is missing: {relative}")
        blobs[relative] = record

    root_names = set(
        _git(official_root, "ls-tree", "--name-only", OFFICIAL_COMMIT).splitlines()
    )
    license_names = sorted(
        name
        for name in root_names
        if name.upper().startswith(("LICENSE", "COPYING", "NOTICE"))
    )
    readme = _git(official_root, "show", f"{OFFICIAL_COMMIT}:README.md")
    license_record = {
        "repository_license_files": license_names,
        "readme_mit_badge": "License-MIT" in readme,
        "readme_mit_statement": "licensed under the MIT License" in readme,
        "status": "ambiguous_missing_license_text" if not license_names else "present",
        "local_execution_status": (
            LOCAL_EXECUTION_LICENSE_STATUS
            if not license_names
            and "License-MIT" in readme
            and "licensed under the MIT License" in readme
            else "blocked_insufficient_license_evidence"
        ),
        "redistribution_status": (
            REDISTRIBUTION_LICENSE_STATUS if not license_names else "not_blocked"
        ),
        "interpretation": (
            "The fixed tree asserts MIT in README.md but contains no license text. "
            "This audit makes no legal determination. The README assertion is used "
            "only as the project-policy basis for local, non-redistributive scientific "
            "execution; redistribution and derivative release require clarification "
            "from the upstream authors."
        ),
    }
    return {
        "repository": OFFICIAL_REPOSITORY,
        "commit": OFFICIAL_COMMIT,
        "observed_head": head,
        "tracked_worktree_dirty": dirty,
        "commit_metadata": {
            "author_date": _git(official_root, "show", "-s", "--format=%aI", head),
            "committer_date": _git(official_root, "show", "-s", "--format=%cI", head),
            "subject": _git(official_root, "show", "-s", "--format=%s", head),
        },
        "blobs": blobs,
        "license": license_record,
    }, errors


def _asset_audit(official_root: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    datasets = official_root / "datasets"
    gene_names, _ = ordered_mapping(datasets / "genes2ID_2k.json")
    pathway_names, _ = ordered_mapping(datasets / "genesets_select_6k.json")
    cell_names, _ = ordered_mapping(datasets / "samples2ID.json")
    adjacency = np.load(datasets / "adj_gene_geneset.npy", mmap_mode="r")
    gsva = np.load(datasets / "gsva_score_go_kegg_v3_format.npy", mmap_mode="r")
    if adjacency.shape != (2_350, 6_561) or str(adjacency.dtype) != "float32":
        errors.append(
            f"unexpected adjacency shape/dtype: {adjacency.shape}/{adjacency.dtype}"
        )
    if gsva.shape != (1_088, 6_561) or str(gsva.dtype) != "float32":
        errors.append(f"unexpected GSVA shape/dtype: {gsva.shape}/{gsva.dtype}")
    graph = pd.read_csv(datasets / "ggi_id.txt", sep=r"\s+")
    expected_columns = ["g1_id", "g2_id", "combined_score"]
    if graph.columns.tolist() != expected_columns:
        errors.append(f"unexpected graph columns: {graph.columns.tolist()}")
    bounds = graph[["g1_id", "g2_id"]].to_numpy(dtype=np.int64, copy=False)
    if bounds.size and (int(bounds.min()) < 0 or int(bounds.max()) >= len(gene_names)):
        errors.append("official graph contains a gene index outside genes2ID_2k.json")
    return {
        "genes": {
            "count": len(gene_names),
            "ordered_names_sha256": hash_strings(gene_names),
        },
        "gene_sets": {
            "count": len(pathway_names),
            "ordered_names_sha256": hash_strings(pathway_names),
        },
        "cells": {
            "count": len(cell_names),
            "ordered_names_sha256": hash_strings(cell_names),
        },
        "gene_gene_set_adjacency": {
            "shape": list(adjacency.shape),
            "dtype": str(adjacency.dtype),
            "nonzero": int(np.count_nonzero(adjacency)),
        },
        "precomputed_gsva": {"shape": list(gsva.shape), "dtype": str(gsva.dtype)},
        "gene_graph": {
            "rows": int(len(graph)),
            "columns": graph.columns.tolist(),
            "minimum_endpoint": int(bounds.min()),
            "maximum_endpoint": int(bounds.max()),
            "self_edges": int((bounds[:, 0] == bounds[:, 1]).sum()),
            "duplicate_directed_edges": int(graph.duplicated(["g1_id", "g2_id"]).sum()),
        },
    }, errors


def _protocol_evidence(official_root: Path) -> dict[str, Any]:
    notebook = json.loads((official_root / "GATDep.ipynb").read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    r_source = (official_root / "geneExpression2GSVA.R").read_text(encoding="utf-8")
    return {
        "paper": {"doi": OFFICIAL_PAPER_DOI, "url": OFFICIAL_PAPER_URL},
        "notebook_static_checks": {
            "ten_cluster_holdout_loop": "for k in np.arange(10)" in source,
            "fixed_inner_random_split": (
                "train_test_split(gene_feature_list, score_list, test_size=0.2, "
                "random_state=42)" in source
            ),
            "adamw_lr_1e_3_weight_decay_1e_2": (
                "AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)" in source
            ),
            "max_epochs_150": "num_epochs=150" in source,
            "test_evaluated_inside_epoch_loop": (
                "test_metrics = evaluate(model, test_loader)" in source
            ),
            "checkpoint_selected_by_validation_loss": "if val_loss < best_val_loss"
            in source,
        },
        "gsva_script_static_checks": {
            "gaussian_kernel": 'kcdf="Gaussian"' in r_source,
            "single_cohort_call": "gsva(as.matrix(gene_expression)" in r_source,
            "serialized_fit_state": False,
            "held_out_transform_api": False,
        },
        "code_paper_discrepancy": (
            "The pinned notebook uses AdamW(lr=1e-3, weight_decay=1e-2), up to "
            "150 epochs, whereas the article methods describe Adam(lr=5e-3) and "
            "75 epochs. The corrected adaptation follows the pinned code optimizer "
            "and records this discrepancy rather than resolving it silently."
        ),
    }


def audit_official_and_project(
    project_root: Path, official_root: Path
) -> dict[str, Any]:
    source, source_errors = _official_source_audit(official_root)
    assets, asset_errors = _asset_audit(official_root)
    cohort = build_common_cohort(project_root, official_root)
    project_cells = read_lines(project_root / "data/processed/cell_lines.txt")
    project_genes = read_lines(project_root / "data/processed/genes.txt")
    official_genes, _ = ordered_mapping(official_root / "datasets/genes2ID_2k.json")
    official_cells, _ = ordered_mapping(official_root / "datasets/samples2ID.json")
    project_gene_set = set(project_genes)
    project_cell_set = set(project_cells)
    missing_genes = sorted(set(official_genes) - project_gene_set)
    missing_cells = sorted(set(official_cells) - project_cell_set)
    errors = [*source_errors, *asset_errors]
    if len(cohort.cell_ids) != 1_088:
        errors.append(f"common-cell count is {len(cohort.cell_ids)}, expected 1088")
    if len(cohort.gene_ids) != 2_341:
        errors.append(f"common-gene count is {len(cohort.gene_ids)}, expected 2341")

    runtime = current_runtime()
    runtime_blockers = runtime_mismatches(runtime)
    preflight_blockers = list(runtime_blockers)
    if source["license"]["local_execution_status"] != LOCAL_EXECUTION_LICENSE_STATUS:
        preflight_blockers.append(LOCAL_EXECUTION_LICENSE_BLOCKER)
    return {
        "schema": AUDIT_SCHEMA,
        "generated_at_utc": utc_now(),
        "scientific_result": False,
        "official_source": source,
        "official_assets": assets,
        "official_protocol": _protocol_evidence(official_root),
        "project_intersection": {
            "project_cells": len(project_cells),
            "project_genes": len(project_genes),
            "official_cells": len(official_cells),
            "official_genes": len(official_genes),
            "common_cells": len(cohort.cell_ids),
            "common_genes": len(cohort.gene_ids),
            "official_cells_missing_from_project": missing_cells,
            "official_genes_missing_from_project": missing_genes,
            "cohort_contract": cohort.contract(),
        },
        "runtime": runtime,
        "expected_runtime_from_fixed_environment_yml": EXPECTED_RUNTIME,
        "comparability": {
            "official_reproduction": {
                "status": "not_eligible_for_strict_corrected_comparison",
                "reasons": [
                    "The released GSVA matrix was computed before outer splitting.",
                    "The released label scaling and K-means fold assignment are cohort-wide.",
                    "The notebook evaluates outer-test data at every epoch.",
                    "The official endpoint has 2,350 genes, while the exact common "
                    "DepPLM endpoint has 2,341 genes.",
                    "The Gaussian GSVA script exposes no train-fit/held-out-transform state.",
                ],
                "allowed_use": (
                    "isolated descriptive replay only; not a primary corrected result"
                ),
            },
            "corrected_adaptation": {
                "label": "GATDep-corrected-adaptation",
                "implementation": "code/gatdep_official_baseline.py",
                "feature_method": FEATURE_METHOD,
                "status": "runner_available_not_scientifically_executed",
                "material_deviations": [
                    "raw common DepMap gene-effect targets replace the released "
                    "per-sample scaled labels",
                    "2,341 common output genes replace the 2,350 official outputs",
                    "the official graph is induced and remapped to those 2,341 genes",
                    "train-z-scored pathway means over expression-available official "
                    "genes replace cohort-wide Gaussian GSVA",
                    "project corrected five-fold nested evaluation replaces official "
                    "K-means cluster holdout",
                    "outer test is transformed/evaluated exactly once after selection",
                ],
                "current_preflight_blockers": preflight_blockers,
                "local_execution_license_gate": source["license"][
                    "local_execution_status"
                ],
                "redistribution_license_gate": source["license"][
                    "redistribution_status"
                ],
                "redistribution_blocker": REDISTRIBUTION_LICENSE_BLOCKER,
                "redistribution_gate_overridable": False,
            },
        },
        "audit_errors": errors,
        "audit_passed": not errors,
    }


def runtime_mismatches(runtime: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if runtime.get("python") != EXPECTED_RUNTIME["python"]:
        blockers.append(
            f"python={runtime.get('python')!r}; expected {EXPECTED_RUNTIME['python']}"
        )
    packages = runtime.get("packages", {})
    for name, expected in EXPECTED_RUNTIME.items():
        if name == "python":
            continue
        observed = packages.get(name) if isinstance(packages, Mapping) else None
        compared: Any
        if name == "torch" and isinstance(observed, str):
            distribution_parts = _parse_torch_version(observed)
            if distribution_parts is None:
                blockers.append(
                    f"torch distribution version is malformed: {observed!r}"
                )
                continue
            compared = distribution_parts[0]
        else:
            compared = observed
        if compared != expected:
            blockers.append(f"{name}={observed!r}; expected {expected}")

    torch_build = runtime.get("torch_build", {})
    if not isinstance(torch_build, Mapping):
        torch_build = {}
    module_version = torch_build.get("module_version")
    module_parts = _parse_torch_version(module_version)
    if isinstance(module_version, str) and module_parts is None:
        blockers.append(f"torch.__version__ is malformed: {module_version!r}")
    module_base = module_parts[0] if module_parts is not None else None
    module_tag = module_parts[1] if module_parts is not None else None
    if module_base != EXPECTED_RUNTIME["torch"]:
        blockers.append(
            f"torch.__version__={module_version!r}; expected base "
            f"{EXPECTED_RUNTIME['torch']}"
        )
    if torch_build.get("base_version") != module_base:
        blockers.append(
            "torch_build.base_version is inconsistent with torch.__version__"
        )
    if torch_build.get("build_tag") != module_tag:
        blockers.append("torch_build.build_tag is inconsistent with torch.__version__")

    distribution_version = (
        packages.get("torch") if isinstance(packages, Mapping) else None
    )
    distribution_parts = _parse_torch_version(distribution_version)
    distribution_tag = distribution_parts[1] if distribution_parts is not None else None
    if distribution_tag is not None and distribution_tag != module_tag:
        blockers.append(
            "torch distribution local build tag differs from torch.__version__: "
            f"{distribution_tag!r} != {module_tag!r}"
        )
    return blockers


def fit_pathway_transform(
    expression: np.ndarray,
    train_indices: Sequence[int] | np.ndarray,
    support_membership: np.ndarray,
) -> FittedPathwayTransform:
    values = np.asarray(expression, dtype=np.float64)
    membership = np.asarray(support_membership, dtype=np.float64)
    train = np.asarray(train_indices, dtype=np.int64)
    if values.ndim != 2 or membership.ndim != 2:
        raise ValueError("expression and support_membership must be two-dimensional")
    if values.shape[1] != membership.shape[0]:
        raise ValueError("expression genes and membership rows do not align")
    if train.ndim != 1 or train.size == 0 or np.unique(train).size != train.size:
        raise ValueError("train_indices must be a non-empty unique vector")
    if int(train.min()) < 0 or int(train.max()) >= values.shape[0]:
        raise IndexError("train_indices are outside expression rows")
    fit_values = values[train]
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(fit_values, axis=0)
        scale = np.nanstd(fit_values, axis=0, ddof=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    denominator = membership.sum(axis=0)
    safe_denominator = np.where(denominator > 0, denominator, 1.0)
    weights = membership / safe_denominator
    return FittedPathwayTransform(
        mean_=mean.astype(np.float64, copy=False),
        scale_=scale.astype(np.float64, copy=False),
        pathway_weights_=weights.astype(np.float64, copy=False),
        fit_indices_=train.copy(),
    )


def derive_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(int(seed)), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def make_outer_splits(n_cells: int, n_splits: int, seed: int) -> list[np.ndarray]:
    if n_splits < 2 or n_splits > n_cells:
        raise ValueError("n_splits must be between 2 and n_cells")
    rng = np.random.default_rng(int(seed))
    return [
        np.sort(part).astype(np.int64)
        for part in np.array_split(rng.permutation(n_cells), n_splits)
    ]


def make_inner_split(
    outer_train: np.ndarray,
    lineages: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    outer = np.asarray(outer_train, dtype=np.int64)
    categories = np.asarray(lineages, dtype=str)
    if outer.ndim != 1 or outer.size == 0 or np.unique(outer).size != outer.size:
        raise ValueError("outer_train must be a non-empty unique vector")
    if not 0.0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be between zero and 0.5")
    rng = np.random.default_rng(int(seed))
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for category in sorted(set(categories[outer].tolist())):
        members = outer[categories[outer] == category].copy()
        rng.shuffle(members)
        if members.size < 2:
            train_parts.append(members)
            continue
        n_val = min(members.size - 1, max(1, round(members.size * val_fraction)))
        val_parts.append(members[:n_val])
        train_parts.append(members[n_val:])
    inner_train = np.sort(np.concatenate(train_parts)).astype(np.int64)
    if val_parts:
        inner_val = np.sort(np.concatenate(val_parts)).astype(np.int64)
    elif inner_train.size >= 2:
        inner_val = inner_train[-1:].copy()
        inner_train = inner_train[:-1]
    else:
        raise ValueError("outer training split is too small")
    return inner_train, inner_val


def split_record(
    cohort: CommonCohort,
    outer_test: np.ndarray,
    inner_train: np.ndarray,
    inner_val: np.ndarray,
    *,
    seed: int,
    fold_number: int,
) -> dict[str, Any]:
    outer_test = np.asarray(outer_test, dtype=np.int64)
    outer_train = np.asarray(
        sorted(set(range(len(cohort.cell_ids))) - set(outer_test.tolist())),
        dtype=np.int64,
    )
    if set(inner_train).union(inner_val) != set(outer_train):
        raise ValueError("inner splits do not partition outer train")
    if set(inner_train).intersection(inner_val) or set(outer_train).intersection(
        outer_test
    ):
        raise ValueError("nested split overlap detected")

    def describe(indices: np.ndarray) -> dict[str, Any]:
        return {
            "indices": indices.tolist(),
            "cell_ids": [cohort.cell_ids[int(index)] for index in indices],
            "indices_sha256": hash_array(indices.astype("<i8", copy=False)),
        }

    record = {
        "schema": "GATDep-corrected-split-v1",
        "seed": int(seed),
        "fold_number": int(fold_number),
        "outer_train": describe(outer_train),
        "inner_train": describe(np.asarray(inner_train, dtype=np.int64)),
        "inner_validation": describe(np.asarray(inner_val, dtype=np.int64)),
        "outer_test": describe(outer_test),
        "cohort_contract_sha256": cohort.contract()["sha256"],
        "fit_scope": {
            "expression_scaling": "inner_train_then_outer_train_refit",
            "pathway_aggregation": "fixed_annotation_transform",
            "target_scaling": "none_raw_gene_effect",
            "outer_test": "transform_and_evaluate_exactly_once_after_selection",
        },
    }
    return {**record, "split_sha256": canonical_hash(record)}


def outer_split_contract(
    cohort: CommonCohort, *, seed: int, n_splits: int
) -> dict[str, Any]:
    folds = make_outer_splits(len(cohort.cell_ids), n_splits, seed)
    payload = {
        "generator": "numpy.default_rng-permutation-array_split-v1",
        "seed": int(seed),
        "n_splits": int(n_splits),
        "cohort_contract_sha256": cohort.contract()["sha256"],
        "outer_test_indices": [fold.tolist() for fold in folds],
        "outer_test_cell_ids": [
            [cohort.cell_ids[int(index)] for index in fold] for fold in folds
        ],
    }
    return {**payload, "sha256": canonical_hash(payload)}


def load_corrected_data(
    project_root: Path,
    official_root: Path,
    cohort: CommonCohort,
) -> dict[str, Any]:
    expression_path = (
        project_root / "data/depmap/OmicsExpressionProteinCodingGenesTPMLogp1.csv"
    )
    header = pd.read_csv(expression_path, nrows=0).columns.tolist()
    id_column = header[0]
    expression_columns: dict[str, str] = {}
    for column in header[1:]:
        symbol = column.split(" (")[0]
        expression_columns.setdefault(symbol, column)

    official_genes, _ = ordered_mapping(official_root / "datasets/genes2ID_2k.json")
    support_names = [gene for gene in official_genes if gene in expression_columns]
    selected_columns = [expression_columns[gene] for gene in support_names]
    frame = pd.read_csv(
        expression_path,
        usecols=[id_column, *selected_columns],
        index_col=0,
        dtype={column: np.float32 for column in selected_columns},
    )
    missing_cells = [cell for cell in cohort.cell_ids if cell not in frame.index]
    if missing_cells:
        raise AuditError(f"expression is missing {len(missing_cells)} common cells")
    expression = frame.reindex(cohort.cell_ids).to_numpy(dtype=np.float32, copy=True)

    adjacency = np.load(
        official_root / "datasets/adj_gene_geneset.npy", allow_pickle=False
    ).astype(np.float32, copy=False)
    _, official_gene_map = ordered_mapping(official_root / "datasets/genes2ID_2k.json")
    support_indices = np.asarray(
        [official_gene_map[gene] for gene in support_names], dtype=np.int64
    )
    support_membership = np.asarray(adjacency[support_indices], dtype=np.float32)
    node_membership = np.asarray(
        adjacency[cohort.official_gene_indices], dtype=np.float32
    )

    target_source = np.load(
        project_root / "data/processed/gene_effect.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    expected_shape = (
        len(read_lines(project_root / "data/processed/genes.txt")),
        len(read_lines(project_root / "data/processed/cell_lines.txt")),
    )
    if target_source.shape != expected_shape:
        raise AuditError(
            f"gene_effect.npy shape {target_source.shape} != manifests {expected_shape}"
        )
    targets = np.asarray(
        target_source[
            np.ix_(cohort.project_gene_indices, cohort.project_cell_indices)
        ].T,
        dtype=np.float32,
    )

    graph = pd.read_csv(official_root / "datasets/ggi_id.txt", sep=r"\s+")
    remap = np.full(len(official_genes), -1, dtype=np.int64)
    remap[cohort.official_gene_indices] = np.arange(len(cohort.gene_ids))
    original_edges = graph[["g1_id", "g2_id"]].to_numpy(dtype=np.int64, copy=False)
    keep = (remap[original_edges[:, 0]] >= 0) & (remap[original_edges[:, 1]] >= 0)
    edges = np.stack(
        [remap[original_edges[keep, 0]], remap[original_edges[keep, 1]]], axis=0
    ).astype(np.int64, copy=False)
    return {
        "expression": expression,
        "expression_gene_names": tuple(support_names),
        "expression_official_gene_indices": support_indices,
        "support_membership": support_membership,
        "node_membership": node_membership,
        "targets": targets,
        "edge_index": edges,
    }


def set_deterministic_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_official_model_class(official_root: Path) -> type[Any]:
    relative = "GeneDependencyGAT.py"
    expected = EXPECTED_BLOBS[relative][0]
    observed = _git(
        official_root,
        "rev-parse",
        f"{OFFICIAL_COMMIT}:{relative}",
    )
    if observed != expected:
        raise AuditError("official model source blob changed")
    if _git(official_root, "rev-parse", "HEAD") != OFFICIAL_COMMIT:
        raise AuditError("official checkout is not at the pinned commit")
    if _git(official_root, "status", "--porcelain", "--untracked-files=no"):
        raise AuditError("official checkout has tracked modifications")
    path = official_root / relative
    if not path.is_file():
        raise AuditError("official model source is missing from the checkout")
    source = read_pinned_text_source(path, expected)
    module = types.ModuleType("gatdep_pinned_model")
    module.__file__ = str(path)
    module.__package__ = ""
    code = compile(source, str(path), "exec", dont_inherit=True)
    exec(code, module.__dict__)
    model_class = getattr(module, "GeneDependencyGAT", None)
    if model_class is None:
        raise AuditError("pinned source does not define GeneDependencyGAT")
    return model_class


def train_gene_mean(targets: np.ndarray, indices: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        reference = np.nanmean(np.asarray(targets)[indices], axis=0)
    if not np.isfinite(reference).all():
        raise ValueError("outer training data cannot define a finite gene mean")
    return reference.astype(np.float32, copy=False)


def dependency_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    reference: np.ndarray,
) -> dict[str, Any]:
    predicted = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    baseline = np.asarray(reference, dtype=np.float64)
    if predicted.shape != truth.shape or predicted.ndim != 2:
        raise ValueError("prediction and target must have the same 2D shape")
    if baseline.shape != (truth.shape[1],):
        raise ValueError("reference must contain one value per gene")
    valid = np.isfinite(truth)
    if np.any(valid & ~np.isfinite(predicted)):
        raise ValueError("non-finite prediction for an evaluable target")
    error = np.where(valid, predicted - truth, 0.0)
    base_error = np.where(valid, baseline[None, :] - truth, 0.0)
    count = int(valid.sum())
    if count == 0:
        raise ValueError("no finite targets")
    sse = float(np.square(error).sum())
    baseline_sse = float(np.square(base_error).sum())
    flat_valid = valid.ravel()
    x = predicted.ravel()[flat_valid]
    y = truth.ravel()[flat_valid]
    pearson = None
    if x.size > 1 and float(x.std()) > 0 and float(y.std()) > 0:
        pearson = float(np.corrcoef(x, y)[0, 1])
    return {
        "n_cells": int(truth.shape[0]),
        "n_genes": int(truth.shape[1]),
        "n_evaluable_pairs": count,
        "mse": sse / count,
        "mae": float(np.abs(error).sum()) / count,
        "gene_mean_baseline_mse": baseline_sse / count,
        "residual_r2_vs_train_gene_mean": (
            1.0 - sse / baseline_sse if baseline_sse > 0 else None
        ),
        "flattened_pearson": pearson,
    }


def _graph_dataset(
    indices: np.ndarray,
    expression: np.ndarray,
    targets: np.ndarray,
    fitted: FittedPathwayTransform,
    node_membership: np.ndarray,
    edge_index: np.ndarray,
) -> Any:
    import torch
    from torch_geometric.data import Data

    class LazyGraphDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return int(indices.size)

        def __getitem__(self, item: int) -> Any:
            index = int(indices[int(item)])
            pathway = fitted.transform(expression[index : index + 1])[0]
            node_features = node_membership * pathway[None, :]
            return Data(
                x=torch.from_numpy(np.asarray(node_features, dtype=np.float32)),
                edge_index=torch.from_numpy(edge_index).long(),
                y=torch.from_numpy(np.asarray(targets[index], dtype=np.float32)),
                source_index=torch.tensor([index], dtype=torch.long),
            )

    return LazyGraphDataset()


def _evaluate_model(
    model: Any,
    indices: np.ndarray,
    data: Mapping[str, Any],
    fitted: FittedPathwayTransform,
    *,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from torch_geometric.loader import DataLoader

    dataset = _graph_dataset(
        indices,
        data["expression"],
        data["targets"],
        fitted,
        data["node_membership"],
        data["edge_index"],
    )
    loader = DataLoader(
        dataset, batch_size=int(batch_size), shuffle=False, num_workers=0
    )
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.batch)
            predictions.append(
                output.reshape(batch.num_graphs, -1).float().cpu().numpy()
            )
            targets.append(batch.y.reshape(batch.num_graphs, -1).float().cpu().numpy())
    return (
        np.concatenate(predictions, axis=0).astype(np.float32, copy=False),
        np.concatenate(targets, axis=0).astype(np.float32, copy=False),
    )


def _train_with_validation(
    model: Any,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    data: Mapping[str, Any],
    fitted: FittedPathwayTransform,
    *,
    device: str,
    batch_size: int,
    max_epochs: int,
    patience: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch
    from torch_geometric.loader import DataLoader

    set_deterministic_seed(seed)
    model.to(device)
    dataset = _graph_dataset(
        train_indices,
        data["expression"],
        data["targets"],
        fitted,
        data["node_membership"],
        data["edge_index"],
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    reference = train_gene_mean(data["targets"], train_indices)
    best_score = -np.inf
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(max_epochs) + 1):
        learning_rate_used = float(optimizer.param_groups[0]["lr"])
        model.train()
        losses: list[float] = []
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch.x, batch.edge_index, batch.batch).reshape(-1)
            truth = batch.y.reshape(-1)
            finite = torch.isfinite(truth)
            loss = torch.nn.functional.huber_loss(
                output[finite], truth[finite], delta=1.0
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        prediction, target = _evaluate_model(
            model,
            validation_indices,
            data,
            fitted,
            device=device,
            batch_size=batch_size,
        )
        metrics = dependency_metrics(prediction, target, reference)
        score_raw = metrics["residual_r2_vs_train_gene_mean"]
        score = float(score_raw) if score_raw is not None else -np.inf
        scheduler.step(float(metrics["mse"]))
        history.append(
            {
                "epoch": epoch,
                "train_huber_loss": float(np.mean(losses)),
                "validation": metrics,
                "learning_rate_used": learning_rate_used,
                "post_scheduler_learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if score > best_score + 1e-12:
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(patience):
                break
    if best_epoch < 1 or not np.isfinite(best_score):
        raise RuntimeError("inner validation did not produce a finite selected model")
    return {
        "best_epoch": best_epoch,
        "selection_score": best_score,
        "refit_learning_rates": [
            float(item["learning_rate_used"]) for item in history[:best_epoch]
        ],
    }, history


def _refit_fixed_epochs(
    model: Any,
    train_indices: np.ndarray,
    data: Mapping[str, Any],
    fitted: FittedPathwayTransform,
    *,
    device: str,
    batch_size: int,
    learning_rates: Sequence[float],
    seed: int,
) -> list[dict[str, Any]]:
    import torch
    from torch_geometric.loader import DataLoader

    set_deterministic_seed(seed)
    model.to(device)
    dataset = _graph_dataset(
        train_indices,
        data["expression"],
        data["targets"],
        fitted,
        data["node_membership"],
        data["edge_index"],
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    frozen_learning_rates = [float(value) for value in learning_rates]
    if not frozen_learning_rates or any(
        not np.isfinite(value) or value <= 0.0 for value in frozen_learning_rates
    ):
        raise ValueError("refit learning-rate schedule must be finite and positive")
    history: list[dict[str, Any]] = []
    for epoch, learning_rate in enumerate(frozen_learning_rates, start=1):
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        model.train()
        losses: list[float] = []
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch.x, batch.edge_index, batch.batch).reshape(-1)
            truth = batch.y.reshape(-1)
            finite = torch.isfinite(truth)
            loss = torch.nn.functional.huber_loss(
                output[finite], truth[finite], delta=1.0
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(
            {
                "epoch": epoch,
                "train_huber_loss": float(np.mean(losses)),
                "learning_rate_used": learning_rate,
            }
        )
    return history


def write_prediction_artifact(
    path: Path,
    *,
    prediction: np.ndarray,
    target: np.ndarray,
    train_reference: np.ndarray,
    outer_test_indices: np.ndarray,
    cohort: CommonCohort,
) -> dict[str, Any]:
    expected = (outer_test_indices.size, len(cohort.gene_ids))
    arrays = {
        "prediction": np.ascontiguousarray(prediction, dtype=np.float32),
        "target_raw": np.ascontiguousarray(target, dtype=np.float32),
        "train_gene_mean": np.ascontiguousarray(train_reference, dtype=np.float32),
        "outer_test_indices_common": np.ascontiguousarray(
            outer_test_indices, dtype=np.int64
        ),
        "source_cell_indices_project": np.ascontiguousarray(
            cohort.project_cell_indices[outer_test_indices], dtype=np.int64
        ),
        "source_gene_indices_project": np.ascontiguousarray(
            cohort.project_gene_indices, dtype=np.int64
        ),
        "official_cell_indices": np.ascontiguousarray(
            cohort.official_cell_indices[outer_test_indices], dtype=np.int64
        ),
        "official_gene_indices": np.ascontiguousarray(
            cohort.official_gene_indices, dtype=np.int64
        ),
        "cell_ids": np.asarray(
            [cohort.cell_ids[int(index)] for index in outer_test_indices], dtype=str
        ),
        "gene_ids": np.asarray(cohort.gene_ids, dtype=str),
    }
    if arrays["prediction"].shape != expected or arrays["target_raw"].shape != expected:
        raise ValueError(f"prediction/target shape does not match {expected}")
    if arrays["train_gene_mean"].shape != (expected[1],):
        raise ValueError("train_gene_mean shape is invalid")
    if not np.isfinite(arrays["prediction"]).all():
        raise ValueError("refusing to save non-finite predictions")
    _atomic_save_npz(path, arrays)
    record = file_record(path, relative_to=path.parent)
    return {
        "schema": PREDICTION_SCHEMA,
        **record,
        "arrays": {
            name: {"shape": list(array.shape), "dtype": str(array.dtype)}
            for name, array in arrays.items()
        },
    }


def _instantiate_model(model_class: type[Any]) -> Any:
    return model_class(in_channels=6_561, hidden_channels=64, out_channels=1)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_or_match(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        if _load_json(path) != _json_compatible(value):
            raise AtomicWriteError(f"existing immutable record differs: {path}")
        return
    atomic_write_json(path, value)


def _immutable_created_at(path: Path) -> str:
    """Reuse an immutable record's original timestamp across fold invocations."""

    if not path.exists():
        return utc_now()
    value = _load_json(path).get("created_at_utc")
    if not isinstance(value, str) or not value.strip():
        raise AtomicWriteError(
            f"existing immutable record lacks created_at_utc: {path}"
        )
    return value


def _fold_paths(run_dir: Path, fold_number: int) -> dict[str, Path]:
    prefix = run_dir / f"fold-{fold_number:02d}"
    return {
        "split": prefix.with_name(prefix.name + "-split.json"),
        "pretest": prefix.with_name(prefix.name + "-pretest.json"),
        "model": prefix.with_name(prefix.name + "-model.pt"),
        "intent": prefix.with_name(prefix.name + "-test-intent.json"),
        "prediction": prefix.with_name(prefix.name + "-predictions.npz"),
        "index": prefix.with_name(prefix.name + "-predictions.index.json"),
        "result": prefix.with_name(prefix.name + "-result.json"),
    }


def _recover_or_write_result(
    paths: Mapping[str, Path],
    pretest: Mapping[str, Any],
    split: Mapping[str, Any],
    cohort: CommonCohort,
) -> dict[str, Any]:
    if paths["result"].exists():
        return _load_json(paths["result"])
    if not paths["prediction"].exists() or not paths["index"].exists():
        raise RuntimeError("cannot finalize a fold without prediction and index")
    with np.load(paths["prediction"], allow_pickle=False) as bundle:
        prediction = bundle["prediction"]
        target = bundle["target_raw"]
        reference = bundle["train_gene_mean"]
    metrics = dependency_metrics(prediction, target, reference)
    result = {
        "schema": "GATDep-corrected-fold-result-v1",
        "fold_number": int(split["fold_number"]),
        "selection": pretest["selection"],
        "test_metrics": metrics,
        "test_evaluation_calls": 1,
        "prediction_index": file_record(
            paths["index"], relative_to=paths["result"].parent
        ),
        "prediction_npz": file_record(
            paths["prediction"], relative_to=paths["result"].parent
        ),
        "split_sha256": split["split_sha256"],
        "cohort_contract_sha256": cohort.contract()["sha256"],
        "scientific_result": False,
        "result_scope": "single_outer_fold_pending_full_verified_aggregation",
    }
    atomic_write_json(paths["result"], result)
    return result


def run_one_fold(
    *,
    fold_number: int,
    outer_test: np.ndarray,
    cohort: CommonCohort,
    data: Mapping[str, Any],
    model_class: type[Any],
    run_dir: Path,
    seed: int,
    device: str,
    batch_size: int,
    max_epochs: int,
    patience: int,
) -> dict[str, Any]:
    paths = _fold_paths(run_dir, fold_number)
    outer_train = np.asarray(
        sorted(set(range(len(cohort.cell_ids))) - set(outer_test.tolist())),
        dtype=np.int64,
    )
    fold_seed = derive_seed(seed, f"fold-{fold_number:02d}")
    inner_split_seed = derive_seed(fold_seed, "inner-split")
    tune_seed = derive_seed(fold_seed, "tune")
    refit_seed = derive_seed(fold_seed, "refit")
    inner_train, inner_val = make_inner_split(
        outer_train,
        cohort.lineages,
        0.2,
        inner_split_seed,
    )
    split = split_record(
        cohort,
        outer_test,
        inner_train,
        inner_val,
        seed=fold_seed,
        fold_number=fold_number,
    )
    _write_or_match(paths["split"], split)
    if paths["result"].exists():
        return _load_json(paths["result"])

    final_fit = fit_pathway_transform(
        data["expression"], outer_train, data["support_membership"]
    )
    if paths["pretest"].exists():
        pretest = _load_json(paths["pretest"])
        if not paths["model"].exists():
            raise RuntimeError("pretest record exists but model checkpoint is missing")
    else:
        inner_fit = fit_pathway_transform(
            data["expression"], inner_train, data["support_membership"]
        )
        set_deterministic_seed(tune_seed)
        selected_model = _instantiate_model(model_class)
        selection, tuning_history = _train_with_validation(
            selected_model,
            inner_train,
            inner_val,
            data,
            inner_fit,
            device=device,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            seed=tune_seed,
        )
        set_deterministic_seed(refit_seed)
        final_model = _instantiate_model(model_class)
        refit_history = _refit_fixed_epochs(
            final_model,
            outer_train,
            data,
            final_fit,
            device=device,
            batch_size=batch_size,
            learning_rates=selection["refit_learning_rates"],
            seed=refit_seed,
        )
        state = {
            key: value.detach().cpu() for key, value in final_model.state_dict().items()
        }
        _atomic_torch_save(paths["model"], state)
        pretest = {
            "schema": "GATDep-corrected-pretest-v1",
            "fold_number": fold_number,
            "selection": {
                "metric": "residual_r2_vs_train_gene_mean",
                "maximize": True,
                "score": selection["selection_score"],
                "best_epoch": selection["best_epoch"],
                "fixed_configuration": {
                    "optimizer": "AdamW",
                    "learning_rate": 1e-3,
                    "weight_decay": 1e-2,
                    "loss": "Huber(delta=1.0)",
                    "scheduler": dict(SCHEDULER_CONFIGURATION),
                    "inner_max_epochs": int(max_epochs),
                    "patience": int(patience),
                },
                "refit_learning_rates": selection["refit_learning_rates"],
            },
            "inner_preprocessing": inner_fit.provenance(),
            "outer_train_preprocessing": final_fit.provenance(),
            "tuning_history": tuning_history,
            "refit_history": refit_history,
            "model_checkpoint": file_record(paths["model"], relative_to=run_dir),
            "outer_test_accessed": False,
            "split_sha256": split["split_sha256"],
            "seeds": {
                "fold": fold_seed,
                "inner_split": inner_split_seed,
                "tune": tune_seed,
                "refit": refit_seed,
            },
        }
        atomic_write_json(paths["pretest"], pretest)
        del selected_model, final_model

    if paths["prediction"].exists():
        if not paths["intent"].exists():
            raise RuntimeError("prediction exists without a prior test-intent record")
        if not paths["index"].exists():
            with np.load(paths["prediction"], allow_pickle=False) as bundle:
                arrays = {name: bundle[name] for name in bundle.files}
            index = {
                "schema": PREDICTION_SCHEMA,
                **file_record(paths["prediction"], relative_to=run_dir),
                "arrays": {
                    name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for name, value in arrays.items()
                },
                "test_evaluation_calls": 1,
                "split_sha256": split["split_sha256"],
                "cohort_contract_sha256": cohort.contract()["sha256"],
                "test_intent_sha256": sha256_file(paths["intent"]),
            }
            atomic_write_json(paths["index"], index)
        return _recover_or_write_result(paths, pretest, split, cohort)

    if paths["intent"].exists():
        raise RuntimeError(
            "test intent exists without predictions; whether outer test was already "
            "evaluated is indeterminate, so this run cannot be resumed"
        )

    intent = {
        "schema": "GATDep-corrected-test-intent-v1",
        "created_at_utc": utc_now(),
        "fold_number": fold_number,
        "nonce": uuid.uuid4().hex,
        "model_checkpoint_sha256": sha256_file(paths["model"]),
        "pretest_sha256": sha256_file(paths["pretest"]),
        "policy": "one_intent_allows_one_outer_test_evaluation",
    }
    atomic_write_json(paths["intent"], intent)

    import torch

    final_model = _instantiate_model(model_class)
    state = torch.load(paths["model"], map_location=device, weights_only=True)
    final_model.load_state_dict(state)
    final_model.to(device)
    prediction, target = _evaluate_model(
        final_model,
        outer_test,
        data,
        final_fit,
        device=device,
        batch_size=batch_size,
    )
    reference = train_gene_mean(data["targets"], outer_train)
    prediction_record = write_prediction_artifact(
        paths["prediction"],
        prediction=prediction,
        target=target,
        train_reference=reference,
        outer_test_indices=outer_test,
        cohort=cohort,
    )
    index = {
        **prediction_record,
        "test_evaluation_calls": 1,
        "split_sha256": split["split_sha256"],
        "cohort_contract_sha256": cohort.contract()["sha256"],
        "test_intent_sha256": sha256_file(paths["intent"]),
    }
    atomic_write_json(paths["index"], index)
    return _recover_or_write_result(paths, pretest, split, cohort)


def formal_preflight(audit: Mapping[str, Any]) -> list[str]:
    blockers = [str(value) for value in audit.get("audit_errors", [])]
    blockers.extend(runtime_mismatches(audit.get("runtime", {})))
    license_status = (
        audit.get("official_source", {})
        .get("license", {})
        .get("local_execution_status")
    )
    if license_status != LOCAL_EXECUTION_LICENSE_STATUS:
        blockers.append(LOCAL_EXECUTION_LICENSE_BLOCKER)
    return blockers


def execution_device_blockers(
    audit: Mapping[str, Any], requested_device: str
) -> list[str]:
    """Reject an accelerator request that the recorded build cannot execute."""

    if requested_device == "cpu":
        return []
    if requested_device != "cuda":
        return [f"unsupported execution device: {requested_device!r}"]
    runtime = audit.get("runtime", {})
    if not isinstance(runtime, Mapping):
        return ["CUDA requested but the runtime record is missing"]
    torch_build = runtime.get("torch_build", {})
    if not isinstance(torch_build, Mapping):
        return ["CUDA requested but the PyTorch build record is missing"]
    if torch_build.get("cuda_available") is not True:
        return ["CUDA requested but torch.cuda.is_available() is false"]
    if torch_build.get("cuda_execution_compatible") is not True:
        architectures = torch_build.get("compiled_architectures", [])
        devices = torch_build.get("devices", [])
        return [
            "CUDA requested but no visible GPU architecture is supported by this "
            f"PyTorch build (compiled={architectures!r}, devices={devices!r})"
        ]
    return []


def run_corrected_adaptation(args: argparse.Namespace) -> Path:
    project_root = Path(args.project_root).resolve()
    official_root = Path(args.official_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    audit = audit_official_and_project(project_root, official_root)
    blockers = [
        *formal_preflight(audit),
        *execution_device_blockers(audit, args.device),
    ]
    if blockers:
        blocker_path = run_dir / "preflight-blocker.json"
        run_dir.mkdir(parents=True, exist_ok=True)
        if not blocker_path.exists():
            atomic_write_json(
                blocker_path,
                {
                    "schema": "GATDep-corrected-preflight-blocker-v1",
                    "generated_at_utc": utc_now(),
                    "scientific_result": False,
                    "blockers": blockers,
                    "audit": audit,
                },
            )
        raise AuditError("formal run blocked: " + "; ".join(blockers))
    if (
        run_dir.exists()
        and any(run_dir.iterdir())
        and not (run_dir / "run-manifest.json").exists()
    ):
        raise AtomicWriteError(
            f"refusing to use non-empty/unbound run directory: {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    cohort = build_common_cohort(project_root, official_root)
    if (len(cohort.cell_ids), len(cohort.gene_ids)) != (1_088, 2_341):
        raise AuditError(
            "formal run requires the audited 1088-cell/2341-gene intersection"
        )
    split_contract = outer_split_contract(cohort, seed=args.seed, n_splits=5)
    code_paths = [project_root / relative for relative in RUN_SOURCE_PATHS]
    missing_code = [str(path) for path in code_paths if not path.is_file()]
    if missing_code:
        raise AuditError(
            "formal run requires the complete verifier source inventory: "
            + ", ".join(missing_code)
        )
    input_paths = [project_root / relative for relative in PROJECT_INPUTS]
    input_paths.extend(official_root / relative for relative in OFFICIAL_RUN_INPUTS)
    manifest_path = run_dir / "run-manifest.json"
    manifest = {
        "schema": RUN_SCHEMA,
        "created_at_utc": _immutable_created_at(manifest_path),
        "scientific_result": False,
        "method_label": "GATDep-corrected-adaptation",
        "official_commit": OFFICIAL_COMMIT,
        "cohort_contract": cohort.contract(),
        "cohort_contract_sha256": cohort.contract()["sha256"],
        "outer_split_contract": split_contract,
        "outer_split_contract_sha256": split_contract["sha256"],
        "seed": int(args.seed),
        "planned_folds": list(range(5)),
        "feature_method": FEATURE_METHOD,
        "endpoint": "raw_DepMap_gene_effect_common_2341_genes",
        "outer_test_policy": "exactly_once_after_inner_selection_and_outer_train_refit",
        "execution_device": args.device,
        "fixed_configuration": {
            "official_model_source": "GeneDependencyGAT.py",
            "in_channels": 6_561,
            "hidden_channels": 64,
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-2,
            "loss": "Huber(delta=1.0)",
            "scheduler": dict(SCHEDULER_CONFIGURATION),
            "inner_max_epochs": int(args.max_epochs),
            "patience": int(args.patience),
            "inner_val_fraction": 0.2,
            "batch_size": int(args.batch_size),
        },
        "runtime": current_runtime(),
        "source_files": [file_record(path) for path in code_paths],
        "input_files": [file_record(path) for path in input_paths],
        "license_evidence_status": audit["official_source"]["license"]["status"],
        "local_execution_license_status": audit["official_source"]["license"][
            "local_execution_status"
        ],
        "redistribution_license_status": audit["official_source"]["license"][
            "redistribution_status"
        ],
        "redistribution_blocker": REDISTRIBUTION_LICENSE_BLOCKER,
    }
    _write_or_match(manifest_path, manifest)
    data = load_corrected_data(project_root, official_root, cohort)
    model_class = load_official_model_class(official_root)
    folds = make_outer_splits(len(cohort.cell_ids), 5, args.seed)
    selected_folds = range(5) if args.fold_id == -1 else [args.fold_id]
    for fold_number in selected_folds:
        run_one_fold(
            fold_number=fold_number,
            outer_test=folds[fold_number],
            cohort=cohort,
            data=data,
            model_class=model_class,
            run_dir=run_dir,
            seed=args.seed,
            device=args.device,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )

    completed = [
        fold_number
        for fold_number in range(5)
        if _fold_paths(run_dir, fold_number)["result"].exists()
    ]
    if completed == list(range(5)):
        coverage = []
        for fold_number in completed:
            with np.load(
                _fold_paths(run_dir, fold_number)["prediction"], allow_pickle=False
            ) as bundle:
                coverage.extend(bundle["outer_test_indices_common"].tolist())
        if sorted(coverage) != list(range(len(cohort.cell_ids))):
            raise RuntimeError(
                "outer-test predictions do not cover each common cell once"
            )
        completion = {
            "schema": "GATDep-corrected-completion-v1",
            "completed_at_utc": utc_now(),
            "completed_folds": completed,
            "outer_test_cells_exactly_once": True,
            "eligible_for_comparison_after_independent_verification": True,
            "scientific_result": False,
        }
        if not (run_dir / "completion.json").exists():
            atomic_write_json(run_dir / "completion.json", completion)
    return run_dir


def run_cpu_smoke(output: Path) -> dict[str, Any]:
    rng = np.random.default_rng(7)
    expression = rng.normal(size=(12, 5)).astype(np.float32)
    membership = (rng.uniform(size=(5, 7)) > 0.55).astype(np.float32)
    train = np.arange(8, dtype=np.int64)
    test = np.arange(8, 12, dtype=np.int64)
    fitted = fit_pathway_transform(expression, train, membership)
    first = fitted.transform(expression[test])
    perturbed = expression.copy()
    perturbed[test] += 1000.0
    refit = fit_pathway_transform(perturbed, train, membership)
    if not np.array_equal(fitted.mean_, refit.mean_) or not np.array_equal(
        fitted.scale_, refit.scale_
    ):
        raise RuntimeError("held-out values changed train-fitted preprocessing")
    smoke = {
        "schema": "GATDep-corrected-cpu-smoke-v1",
        "generated_at_utc": utc_now(),
        "scientific_result": False,
        "n_train": int(train.size),
        "n_test": int(test.size),
        "transformed_shape": list(first.shape),
        "held_out_perturbation_changed_fit": False,
        "feature_provenance": fitted.provenance(),
    }
    atomic_write_json(output, smoke)
    return smoke


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "smoke", "run"), required=True)
    parser.add_argument(
        "--project-root", default=str(Path(__file__).resolve().parent.parent)
    )
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--output")
    parser.add_argument("--run-dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold-id", type=int, choices=(-1, 0, 1, 2, 3, 4), default=-1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args(argv)
    if args.mode in {"audit", "smoke"} and not args.output:
        parser.error(f"--output is required for --mode {args.mode}")
    if args.mode == "run" and not args.run_dir:
        parser.error("--run-dir is required for --mode run")
    if args.batch_size < 1 or args.max_epochs < 1 or args.patience < 1:
        parser.error("batch size, max epochs, and patience must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "audit":
        audit = audit_official_and_project(
            Path(args.project_root).resolve(), Path(args.official_root).resolve()
        )
        atomic_write_json(Path(args.output), audit)
        print(
            json.dumps({"audit_passed": audit["audit_passed"], "output": args.output})
        )
        return 0 if audit["audit_passed"] else 2
    if args.mode == "smoke":
        run_cpu_smoke(Path(args.output))
        print(json.dumps({"smoke_passed": True, "output": args.output}))
        return 0
    run_dir = run_corrected_adaptation(args)
    print(json.dumps({"run_dir": str(run_dir)}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuditError, AtomicWriteError) as exc:
        print(
            json.dumps(
                {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
