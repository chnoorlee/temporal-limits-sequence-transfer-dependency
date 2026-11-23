"""Hash-bound inference-time attention perturbation study for DepPLM.

This module is deliberately separate from the corrected-v2 core runner.  It
reuses that runner's P1 outer folds, inner split, fold-local preprocessing,
hyperparameter grid, training loop, and deterministic seed derivation without
modifying any frozen source file.

The study asks whether mutation tokens receiving high *mean model attention*
are more influential under inference-time masking than covariate-matched
random mutation tokens.  It also permutes attention weights and randomizes the
trained attention projections as sanity checks.  These are associational
faithfulness diagnostics of a fitted model, not causal biological effects.

Formal evidence requires all 3 seeds x 5 folds.  Pilot mode is only a CPU
structural smoke test and is always ineligible for scientific aggregation.
"""

from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import importlib.metadata as importlib_metadata
import io
import json
import math
import os
import pickle
import platform
import stat
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import numpy as np
    import scipy
    import sklearn
    import torch

    import run_corrected_experiments as corrected
    from corrected_protocol import (
        K_TOKENS,
        atomic_write_json,
        dependency_metrics,
        derive_seed,
        fit_gene_mean_baseline,
        make_inner_split,
        make_kfold_outer_splits,
        set_global_seed,
    )
    from models import DepPLM
except (ImportError, OSError, RuntimeError) as exc:
    if __name__ == "__main__":
        print(
            "attention-faithfulness startup failed while importing a required "
            f"dependency: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(2) from None
    raise


ROOT = Path(os.path.abspath(os.fspath(__file__))).parent.parent
OUTPUT_ROOT = ROOT / "results" / "attention_faithfulness_v1"
SCHEMA_VERSION = "DepPLM-attention-faithfulness-v1"
CANONICAL_SEEDS = (42, 43, 44)
CANONICAL_FOLDS = 5
MASK_FRACTIONS = (0.10, 0.25, 0.50)
MATCHED_RANDOM_DRAWS = 3
MATCH_NEAREST_CANDIDATES = 3
PANEL_SIZE = 64
TOP_PARTNERS = 10
DEFAULT_ATTENTION_BATCH_SIZE = 4
DEFAULT_GENE_CHUNK_SIZE = 1_024
FLOAT_TOLERANCE = 1e-7
REPLAY_RTOL = 1e-5
REPLAY_ATOL = 1e-6
MATCHING_FEATURE_COLUMNS = (
    "top_log1p_prevalence_mean",
    "control_log1p_prevalence_mean",
    "top_log1p_degree_mean",
    "control_log1p_degree_mean",
    "standardized_l1_imbalance",
    "top_control_overlap_fraction",
    "fallback_count",
)
MATCHING_OVERLAP_COLUMN = MATCHING_FEATURE_COLUMNS.index(
    "top_control_overlap_fraction"
)
MATCHING_FALLBACK_COLUMN = MATCHING_FEATURE_COLUMNS.index("fallback_count")

PROTOCOL_SCOPE = (
    "All masking and attention-weight permutation operations are inference-time "
    "perturbations of fitted DepPLM models. They test model faithfulness and "
    "must not be described as causal biological interventions or mechanisms."
)

INPUT_PATHS = (
    corrected.PROCESSED / "gene_effect.npy",
    ROOT / corrected.EMBEDDING_RELATIVE_PATH,
    ROOT / corrected.NORMALIZED_RELATIVE_PATH,
    ROOT / corrected.NAMES_RELATIVE_PATH,
    ROOT / corrected.MANIFEST_RELATIVE_PATH,
    corrected.PROCESSED / "gene_seqs.fa",
    corrected.PROCESSED / "ppi_graph.npz",
    corrected.PROCESSED / "genes.txt",
    corrected.PROCESSED / "cell_lines.txt",
    corrected.PROCESSED / "lineages.txt",
    corrected.DEPMAP / "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
    corrected.DEPMAP / "OmicsSomaticMutations.csv",
)
CODE_PATHS = (
    Path(os.path.abspath(os.fspath(__file__))),
    ROOT / "code" / "run_corrected_experiments.py",
    ROOT / "code" / "corrected_protocol.py",
    ROOT / "code" / "models.py",
    ROOT / "code" / "03_embed_proteins.py",
    ROOT / "code" / "embedding_provenance.py",
)


class VerificationError(RuntimeError):
    """Raised when a persisted study artifact fails closed verification."""


@dataclass(frozen=True)
class ArtifactSnapshot:
    """Immutable bytes that were checked against one persisted file record."""

    path: Path
    data: bytes


@dataclass(frozen=True)
class ValidatedTrainingAttempt:
    """A resume candidate whose JSON, preprocessing, and checkpoint are bound."""

    training: dict[str, Any]
    fold_data: corrected.FoldData
    checkpoint: ArtifactSnapshot


@dataclass(frozen=True)
class MaskPlan:
    top_mask: np.ndarray
    matched_masks: tuple[np.ndarray, ...]
    matching_features: tuple[np.ndarray, ...]
    random_seeds: tuple[int, ...]


@dataclass(frozen=True)
class InferenceEvidence:
    prediction: np.ndarray
    mean_attention: np.ndarray | None = None
    panel_attention: np.ndarray | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _fraction_key(fraction: float) -> str:
    return f"f{int(round(float(fraction) * 100)):03d}"


def _required_prediction_modes() -> tuple[str, ...]:
    modes = ["baseline"]
    for fraction in MASK_FRACTIONS:
        key = _fraction_key(fraction)
        modes.append(f"top_mask_{key}")
        modes.extend(
            f"matched_mask_{key}_d{draw:02d}" for draw in range(MATCHED_RANDOM_DRAWS)
        )
    modes.extend(("attention_permuted", "parameter_randomized"))
    return tuple(modes)


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


def _sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _distribution_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _installed_distribution_inventory() -> list[dict[str, str]]:
    """Return a deterministic inventory of installed Python distributions."""

    records: list[dict[str, str]] = []
    for distribution in importlib_metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        records.append({"name": str(name), "version": str(distribution.version)})
    records.sort(key=lambda item: (item["name"].casefold(), item["version"]))
    return records


def _nvidia_smi_probe() -> dict[str, Any]:
    """Capture nvidia-smi inventory without conflating failure with no GPUs."""

    def failed(error: str) -> dict[str, Any]:
        return {"status": "probe_failed", "error": error, "gpus": []}

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        return failed(f"{type(exc).__name__}: {exc}")
    if completed.returncode != 0:
        detail = completed.stderr.strip().replace("\r", " ").replace("\n", " ")
        return failed(detail or f"nvidia-smi exited {completed.returncode}")
    records: list[dict[str, Any]] = []
    try:
        rows = csv.reader(io.StringIO(completed.stdout))
        for row in rows:
            if len(row) != 6:
                return failed("nvidia-smi returned a malformed CSV row")
            (
                index_text,
                gpu_uuid,
                pci_bus_id,
                name,
                driver_version,
                memory_total_mib,
            ) = (value.strip() for value in row)
            records.append(
                {
                    "nvidia_smi_index": int(index_text),
                    "uuid": gpu_uuid,
                    "pci_bus_id": pci_bus_id,
                    "name": name,
                    "driver_version": driver_version,
                    "memory_total_mib": int(memory_total_mib),
                }
            )
    except (TypeError, ValueError, UnicodeError) as exc:
        return failed(f"{type(exc).__name__}: malformed nvidia-smi output")
    records.sort(key=lambda item: int(item["nvidia_smi_index"]))
    return {"status": "ok", "error": None, "gpus": records}


def _nvidia_smi_gpus() -> list[dict[str, Any]]:
    """Compatibility view of the physical inventory only."""

    return list(_nvidia_smi_probe()["gpus"])


def _optional_device_identifier(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            value = value.decode("ascii", errors="strict")
        except UnicodeError:
            return None
    text = str(value).strip()
    return text or None


def _resolve_cuda_physical_binding(
    *,
    logical_index: int,
    nvidia_smi_gpus: Sequence[Mapping[str, Any]],
    cuda_visible_devices: str | None,
    torch_uuid: str | None,
    torch_pci_bus_id: str | None,
) -> dict[str, Any]:
    tokens = (
        [item.strip() for item in cuda_visible_devices.split(",")]
        if cuda_visible_devices is not None
        else []
    )
    visibility_token = (
        tokens[logical_index]
        if logical_index < len(tokens) and tokens[logical_index]
        else None
    )

    def unique_match(field: str, identifier: str) -> Mapping[str, Any] | None:
        folded = identifier.casefold()
        matches = [
            item
            for item in nvidia_smi_gpus
            if str(item[field]).casefold() == folded
        ]
        return matches[0] if len(matches) == 1 else None

    matches: list[tuple[str, Mapping[str, Any]]] = []
    unresolved_statuses: list[str] = []
    if torch_uuid is not None:
        match = unique_match("uuid", torch_uuid)
        if match is None:
            unresolved_statuses.append("unresolved_torch_uuid")
        else:
            matches.append(("torch_uuid", match))
    if torch_pci_bus_id is not None:
        match = unique_match("pci_bus_id", torch_pci_bus_id)
        if match is None:
            unresolved_statuses.append("unresolved_torch_pci_bus_id")
        else:
            matches.append(("torch_pci_bus_id", match))
    if visibility_token is not None:
        if visibility_token.casefold().startswith("gpu-"):
            exact = unique_match("uuid", visibility_token)
            if exact is None:
                prefix_matches = [
                    item
                    for item in nvidia_smi_gpus
                    if str(item["uuid"]).casefold().startswith(
                        visibility_token.casefold()
                    )
                ]
                exact = prefix_matches[0] if len(prefix_matches) == 1 else None
            if exact is None:
                unresolved_statuses.append("unresolved_cuda_visible_devices_uuid")
            else:
                matches.append(("cuda_visible_devices_uuid", exact))
        elif visibility_token.isdecimal():
            # CUDA ordinals are not guaranteed to equal nvidia-smi indices.
            unresolved_statuses.append(
                "unresolved_cuda_visible_devices_numeric_ordinal"
            )
        else:
            unresolved_statuses.append("unresolved_cuda_visible_devices_token")
    elif cuda_visible_devices is not None:
        unresolved_statuses.append("unresolved_cuda_visible_devices_logical_index")

    match: Mapping[str, Any] | None = None
    status = "unresolved"
    if unresolved_statuses:
        status = unresolved_statuses[0]
    elif matches:
        identities = {
            (
                int(item["nvidia_smi_index"]),
                str(item["uuid"]).casefold(),
                str(item["pci_bus_id"]).casefold(),
            )
            for _, item in matches
        }
        if len(identities) != 1:
            status = "conflicting_cuda_identifiers"
        else:
            match = matches[0][1]
            sources = {source for source, _ in matches}
            if len(sources) > 1:
                status = "resolved_by_consistent_identifiers"
            elif "torch_uuid" in sources:
                status = "resolved_by_torch_uuid"
            elif "torch_pci_bus_id" in sources:
                status = "resolved_by_torch_pci_bus_id"
            else:
                status = "resolved_by_cuda_visible_devices_uuid"
    return {
        "cuda_visibility_token": visibility_token,
        "physical_binding_status": status,
        "physical_device": dict(match) if match is not None else None,
    }


def _environment_manifest(device: str) -> dict[str, Any]:
    packages = _installed_distribution_inventory()
    nvidia_probe = _nvidia_smi_probe()
    nvidia_smi_gpus = list(nvidia_probe["gpus"])
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    cuda_device: dict[str, Any] | None = None
    if device == "cuda":
        if nvidia_probe["status"] != "ok" or not nvidia_smi_gpus:
            raise VerificationError(
                "formal CUDA evidence requires a successful, non-empty nvidia-smi probe"
            )
        if not torch.cuda.is_available():
            raise VerificationError("CUDA environment capture requested but CUDA is unavailable")
        index = int(torch.cuda.current_device())
        properties = torch.cuda.get_device_properties(index)
        torch_uuid = _optional_device_identifier(getattr(properties, "uuid", None))
        torch_pci_bus_id = _optional_device_identifier(
            getattr(properties, "pci_bus_id", None)
        )
        binding = _resolve_cuda_physical_binding(
            logical_index=index,
            nvidia_smi_gpus=nvidia_smi_gpus,
            cuda_visible_devices=cuda_visible_devices,
            torch_uuid=torch_uuid,
            torch_pci_bus_id=torch_pci_bus_id,
        )
        if not str(binding["physical_binding_status"]).startswith("resolved_by_"):
            raise VerificationError(
                "formal CUDA evidence requires a unique, consistent physical GPU binding; "
                f"got {binding['physical_binding_status']}"
            )
        cuda_device = {
            "logical_index": index,
            "name": str(properties.name),
            "capability": [int(properties.major), int(properties.minor)],
            "total_memory_bytes": int(properties.total_memory),
            "torch_uuid": torch_uuid,
            "torch_pci_bus_id": torch_pci_bus_id,
            **binding,
        }
    return {
        "python": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": _distribution_version("pandas"),
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cudnn": torch.backends.cudnn.version(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_visible_devices": cuda_visible_devices,
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "cuda_device": cuda_device,
        "nvidia_smi_probe_status": nvidia_probe["status"],
        "nvidia_smi_probe_error": nvidia_probe["error"],
        "nvidia_smi_gpus": nvidia_smi_gpus,
        "nvidia_driver_versions": sorted(
            {str(item["driver_version"]) for item in nvidia_smi_gpus}
        ),
        "installed_distributions": packages,
        "installed_distributions_sha256": _json_sha256(packages),
    }


def _file_record(path: Path, *, relative_to: Path = ROOT) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        data = resolved.read_bytes()
    except OSError as exc:
        raise VerificationError(f"cannot read artifact {resolved}: {exc}") from exc
    try:
        display = resolved.relative_to(relative_to.resolve()).as_posix()
    except ValueError:
        display = str(resolved)
    return {
        "path": display,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _resolve_record(record: Mapping[str, Any], *, relative_to: Path) -> Path:
    path = Path(str(record["path"]))
    try:
        resolved_root = relative_to.resolve()
        resolved = (
            path.resolve()
            if path.is_absolute()
            else (resolved_root / path).resolve()
        )
    except (OSError, RuntimeError) as exc:
        raise VerificationError(
            f"cannot resolve recorded artifact path {path}: {exc}"
        ) from exc
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise VerificationError(
            f"recorded artifact resolves outside the selected output root: {resolved}"
        ) from exc
    return resolved


_FILE_RECORD_FIELDS = frozenset({"path", "bytes", "sha256"})


def _validate_file_record(record: Mapping[str, Any], *, label: str) -> None:
    if not isinstance(record, Mapping) or set(record) != _FILE_RECORD_FIELDS:
        raise VerificationError(f"{label} is not an exact file record")
    if not isinstance(record["path"], str) or not record["path"]:
        raise VerificationError(f"{label} has an invalid path")
    byte_count = record["bytes"]
    if (
        isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 0
    ):
        raise VerificationError(f"{label} has an invalid byte count")
    digest = record["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise VerificationError(f"{label} has an invalid SHA256 digest")


def _is_exact_file_record(value: Any) -> bool:
    try:
        _validate_file_record(value, label="embedded file record")
    except (VerificationError, TypeError):
        return False
    return True


def _snapshot_file_record(
    record: Mapping[str, Any], *, relative_to: Path, label: str = "file record"
) -> ArtifactSnapshot:
    _validate_file_record(record, label=label)
    path = _resolve_record(record, relative_to=relative_to)
    if not path.is_file():
        raise VerificationError(f"recorded artifact is missing: {path}")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise VerificationError(f"cannot read recorded artifact {path}: {exc}") from exc
    if (
        len(data) != record["bytes"]
        or hashlib.sha256(data).hexdigest() != record["sha256"]
    ):
        raise VerificationError(f"recorded artifact changed: {path}")
    return ArtifactSnapshot(path=path, data=data)


def _verify_file_record(record: Mapping[str, Any], *, relative_to: Path) -> Path:
    return _snapshot_file_record(record, relative_to=relative_to).path


def _atomic_npz(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    relative_to: Path,
) -> dict[str, Any]:
    _resolve_record({"path": str(path.resolve())}, relative_to=relative_to)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}.npz")
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise VerificationError(f"refusing to overwrite artifact: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return _file_record(path, relative_to=relative_to)


def _atomic_torch_save(
    path: Path,
    payload: Mapping[str, Any],
    *,
    relative_to: Path,
) -> dict[str, Any]:
    _resolve_record({"path": str(path.resolve())}, relative_to=relative_to)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise VerificationError(f"refusing to overwrite artifact: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return _file_record(path, relative_to=relative_to)


def _load_json_bytes(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read JSON artifact {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise VerificationError(f"JSON artifact must contain an object: {label}")
    return payload


def _load_json_snapshot(snapshot: ArtifactSnapshot) -> dict[str, Any]:
    return _load_json_bytes(snapshot.data, label=str(snapshot.path))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise VerificationError(f"cannot read JSON artifact {path}: {exc}") from exc
    return _load_json_bytes(data, label=str(path))


def _read_artifact_snapshot(path: Path, *, relative_to: Path) -> ArtifactSnapshot:
    resolved = _resolve_record({"path": str(path)}, relative_to=relative_to)
    try:
        data = resolved.read_bytes()
    except OSError as exc:
        raise VerificationError(f"cannot read artifact {resolved}: {exc}") from exc
    return ArtifactSnapshot(path=resolved, data=data)


def _snapshot_record(
    snapshot: ArtifactSnapshot, *, relative_to: Path
) -> dict[str, Any]:
    try:
        display = snapshot.path.relative_to(relative_to.resolve()).as_posix()
    except ValueError:
        display = str(snapshot.path)
    return {
        "path": display,
        "bytes": len(snapshot.data),
        "sha256": hashlib.sha256(snapshot.data).hexdigest(),
    }


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        atomic_write_json(path, dict(payload))
    except VerificationError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise VerificationError(
            f"cannot atomically write JSON artifact {path}: {exc}"
        ) from exc


def _default_detached_receipt_root(output_root: Path) -> Path:
    resolved = output_root.resolve()
    return resolved.parent / f"{resolved.name}-verification-receipts"


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _assert_no_link_or_reparse(path: Path) -> None:
    """Reject existing symlink/reparse components without following them."""

    absolute = _absolute_without_resolving(path)
    current = Path(absolute.anchor)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            break
        except OSError as exc:
            raise VerificationError(
                f"cannot inspect detached verification path {current}: {exc}"
            ) from exc
        file_attributes = int(getattr(metadata, "st_file_attributes", 0))
        if stat.S_ISLNK(metadata.st_mode) or file_attributes & reparse_flag:
            raise VerificationError(
                "detached verification path contains a symlink or reparse "
                f"point: {current}"
            )


def _resolve_detached_path(path: Path, *, label: str, strict: bool) -> Path:
    try:
        return path.resolve(strict=strict)
    except (OSError, RuntimeError) as exc:
        raise VerificationError(f"cannot resolve {label} {path}: {exc}") from exc


def _detached_directory_identity(path: Path) -> tuple[int, int]:
    try:
        metadata = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise VerificationError(
            f"cannot inspect detached verification directory {path}: {exc}"
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise VerificationError(
            f"detached verification receipt root is not a directory: {path}"
        )
    return int(metadata.st_dev), int(metadata.st_ino)


def _prepare_detached_receipt_root(
    *, output_root: Path, receipt_root: Path
) -> tuple[Path, Path, tuple[int, int]]:
    resolved_output = _resolve_detached_path(
        output_root, label="output_root", strict=True
    )
    if not resolved_output.is_dir():
        raise VerificationError(f"output_root is not a directory: {resolved_output}")

    lexical_receipt = _absolute_without_resolving(receipt_root)
    _assert_no_link_or_reparse(lexical_receipt)
    resolved_receipt = _resolve_detached_path(
        lexical_receipt, label="detached verification receipt root", strict=False
    )
    if _is_within(resolved_receipt, resolved_output):
        raise VerificationError(
            "detached verification receipt root must be outside output_root"
        )
    try:
        lexical_receipt.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise VerificationError(
            f"cannot create detached verification receipt root {lexical_receipt}: {exc}"
        ) from exc

    # Recheck after mkdir: a parent may have changed while the directory was made.
    _assert_no_link_or_reparse(lexical_receipt)
    resolved_receipt = _resolve_detached_path(
        lexical_receipt, label="detached verification receipt root", strict=True
    )
    if _is_within(resolved_receipt, resolved_output):
        raise VerificationError(
            "detached verification receipt root resolves inside output_root"
        )
    identity = _detached_directory_identity(lexical_receipt)
    return resolved_output, resolved_receipt, identity


def _guard_detached_artifact_path(
    path: Path,
    *,
    output_root: Path,
    receipt_root: Path,
    receipt_identity: tuple[int, int],
    written: bool,
) -> Path:
    """Revalidate a detached target immediately before or after a write."""

    _assert_no_link_or_reparse(receipt_root)
    if _detached_directory_identity(receipt_root) != receipt_identity:
        raise VerificationError(
            "detached verification receipt directory changed during write"
        )
    resolved_parent = _resolve_detached_path(
        path.parent, label="detached artifact parent", strict=True
    )
    resolved_receipt = _resolve_detached_path(
        receipt_root, label="detached verification receipt root", strict=True
    )
    if resolved_parent != resolved_receipt:
        raise VerificationError(
            "detached artifact parent changed during verification write"
        )
    if _is_within(resolved_parent, output_root):
        raise VerificationError(
            "detached artifact parent resolves inside output_root"
        )

    if written:
        _assert_no_link_or_reparse(path)
        resolved_path = _resolve_detached_path(
            path, label="detached artifact", strict=True
        )
        if not resolved_path.is_file():
            raise VerificationError(
                f"detached verification artifact is not a file: {resolved_path}"
            )
    else:
        if path.exists() or path.is_symlink():
            raise VerificationError(
                f"refusing to overwrite detached verification artifact: {path}"
            )
        resolved_path = _resolve_detached_path(
            path, label="detached artifact", strict=False
        )
    if resolved_path.parent != resolved_receipt:
        raise VerificationError(
            "detached artifact resolved outside its receipt directory"
        )
    if _is_within(resolved_path, output_root):
        raise VerificationError("detached artifact resolves inside output_root")
    return resolved_path


def _validated_summary_completion_roots(
    summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    roots = summary.get("accepted_attempt_completion_roots")
    evidence = summary.get("fold_replay_evidence")
    if not isinstance(roots, list) or not isinstance(evidence, list):
        raise VerificationError(
            "verification summary lacks accepted completion-root evidence"
        )
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(roots):
        _validate_file_record(
            record, label=f"accepted completion root[{index}]"
        )
        normalized.append(dict(record))
    evidence_roots: list[dict[str, Any]] = []
    for index, item in enumerate(evidence):
        if not isinstance(item, Mapping):
            raise VerificationError(f"fold replay evidence[{index}] is invalid")
        record = item.get("completion")
        _validate_file_record(
            record, label=f"fold replay evidence[{index}].completion"
        )
        evidence_roots.append(dict(record))
    if normalized != evidence_roots:
        raise VerificationError(
            "accepted completion roots disagree with fold replay evidence"
        )
    if summary.get("verification_passed") is True and not normalized:
        raise VerificationError(
            "passing verification summary has no accepted completion roots"
        )
    return normalized


def write_detached_verification_package(
    summary: Mapping[str, Any],
    *,
    output_root: Path,
    receipt_root: Path,
    verifier_entrypoint: Path | None = None,
) -> dict[str, Path]:
    """Write a checksum-bound verifier package without modifying output_root.

    Repeated path checks fail closed on observed symlink/reparse substitutions;
    they are not an absolute defence against a concurrent same-account attacker.
    """

    resolved_output, resolved_receipt, receipt_identity = (
        _prepare_detached_receipt_root(
            output_root=output_root, receipt_root=receipt_root
        )
    )
    package_id = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}"
        f"-p{os.getpid()}-{uuid.uuid4().hex[:8]}"
    )
    summary_path = resolved_receipt / f"verification-summary-{package_id}.json"
    completion_roots = _validated_summary_completion_roots(summary)
    _guard_detached_artifact_path(
        summary_path,
        output_root=resolved_output,
        receipt_root=resolved_receipt,
        receipt_identity=receipt_identity,
        written=False,
    )
    _write_json_once(summary_path, summary)
    _guard_detached_artifact_path(
        summary_path,
        output_root=resolved_output,
        receipt_root=resolved_receipt,
        receipt_identity=receipt_identity,
        written=True,
    )
    verifier_paths = [Path(__file__).resolve()]
    if verifier_entrypoint is not None:
        resolved_entrypoint = verifier_entrypoint.resolve()
        if resolved_entrypoint not in verifier_paths:
            verifier_paths.append(resolved_entrypoint)
    receipt = {
        "schema": f"{SCHEMA_VERSION}-detached-verification-receipt",
        "created_utc": _utc_now(),
        "source_output_root": str(resolved_output),
        "scope": summary.get("scope"),
        "verification_passed": summary.get("verification_passed"),
        "scientific_result": summary.get("scientific_result"),
        "summary": _file_record(summary_path, relative_to=resolved_receipt),
        "accepted_attempt_completion_roots": completion_roots,
        "verifier_code": [
            _file_record(path, relative_to=ROOT) for path in verifier_paths
        ],
        "assurance_scope": (
            "Detached SHA256 binding produced by a separate-capable verifier entry; "
            "this is not a signature, immutable archive, or independent verification "
            "implementation. Best-effort pre/post path checks reject observed "
            "symlink or reparse substitutions and output-root containment, but do "
            "not guarantee resistance to concurrent same-account path replacement."
        ),
    }
    receipt_path = resolved_receipt / f"verification-receipt-{package_id}.json"
    _guard_detached_artifact_path(
        receipt_path,
        output_root=resolved_output,
        receipt_root=resolved_receipt,
        receipt_identity=receipt_identity,
        written=False,
    )
    _write_json_once(receipt_path, receipt)
    _guard_detached_artifact_path(
        receipt_path,
        output_root=resolved_output,
        receipt_root=resolved_receipt,
        receipt_identity=receipt_identity,
        written=True,
    )
    return {"summary": summary_path, "receipt": receipt_path}


_DETACHED_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "created_utc",
        "source_output_root",
        "scope",
        "verification_passed",
        "scientific_result",
        "summary",
        "accepted_attempt_completion_roots",
        "verifier_code",
        "assurance_scope",
    }
)


def _verify_detached_completion_tree(
    completion_record: Mapping[str, Any], *, output_root: Path
) -> dict[str, int]:
    """Rehash one completion and every file record reachable from it.

    This is intentionally a checksum/schema replay.  The checkpoint is hashed
    as opaque bytes and is never deserialized with PyTorch here.
    """

    completion_snapshot = _snapshot_file_record(
        completion_record,
        relative_to=output_root,
        label="detached completion root",
    )
    completion = _load_json_snapshot(completion_snapshot)
    _require_exact_fields(completion, _COMPLETION_FIELDS, label="completion")
    if completion.get("schema") != f"{SCHEMA_VERSION}-completion":
        raise VerificationError("completion schema mismatch")
    pilot = completion.get("pilot")
    if not isinstance(pilot, bool):
        raise VerificationError("completion pilot must be boolean")
    seed = _require_int_field(completion, "seed", label="completion")
    fold = _require_int_field(completion, "fold", label="completion")
    _validate_completion_semantics(completion, pilot=pilot)

    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != _COMPLETION_ARTIFACT_KEYS:
        raise VerificationError("completion artifact key set mismatch")
    attempt = completion_snapshot.path.parent
    leaf_specs = {
        "training": (completion.get("training"), attempt / "training.json"),
        "run_manifest": (
            completion.get("run_manifest"),
            attempt / "run-manifest.json",
        ),
        "predictions": (
            artifacts.get("predictions"),
            attempt / "outer-test-predictions.npz",
        ),
        "attention_evidence": (
            artifacts.get("attention_evidence"),
            attempt / "attention-evidence.npz",
        ),
        "inference": (artifacts.get("inference"), attempt / "inference.json"),
        "metrics": (artifacts.get("metrics"), attempt / "metrics.json"),
    }
    leaves: dict[str, ArtifactSnapshot] = {}
    for label, (record, expected) in leaf_specs.items():
        leaves[label] = _snapshot_expected_file_record(
            record,
            expected=expected,
            relative_to=output_root,
            label=f"detached completion {label}",
        )

    manifest = _load_json_snapshot(leaves["run_manifest"])
    _require_exact_fields(manifest, _MANIFEST_FIELDS, label="run manifest")
    if manifest.get("schema") != f"{SCHEMA_VERSION}-run-manifest":
        raise VerificationError("run-manifest schema mismatch")
    manifest_seed = _require_int_field(manifest, "seed", label="run manifest")
    manifest_fold = _require_int_field(manifest, "fold", label="run manifest")
    manifest_pilot = manifest.get("pilot")
    arguments = manifest.get("arguments")
    if not isinstance(manifest_pilot, bool) or not isinstance(arguments, Mapping):
        raise VerificationError("run-manifest pilot/arguments fields are invalid")
    if set(arguments) != _MANIFEST_ARGUMENT_FIELDS:
        raise VerificationError("run-manifest arguments field set mismatch")
    device = arguments.get("device")
    if device not in {"cpu", "cuda"}:
        raise VerificationError("run-manifest device is invalid")
    _require_pilot_contract(
        pilot=manifest_pilot,
        seed=manifest_seed,
        fold=manifest_fold,
        device=str(device),
    )
    if (manifest_seed, manifest_fold, manifest_pilot) != (seed, fold, pilot):
        raise VerificationError("completion identity disagrees with run manifest")

    training = _load_json_snapshot(leaves["training"])
    _require_exact_fields(training, _TRAINING_FIELDS, label="training")
    if training.get("schema") != f"{SCHEMA_VERSION}-training":
        raise VerificationError("training schema mismatch")
    if (
        _require_int_field(training, "seed", label="training") != seed
        or _require_int_field(training, "fold", label="training") != fold
    ):
        raise VerificationError("training identity disagrees with completion")
    _snapshot_expected_file_record(
        training.get("checkpoint"),
        expected=attempt / "trained-model.pt",
        relative_to=output_root,
        label="detached training checkpoint",
    )

    inference = _load_json_snapshot(leaves["inference"])
    _require_exact_fields(inference, _INFERENCE_FIELDS, label="inference")
    if inference.get("schema") != f"{SCHEMA_VERSION}-inference":
        raise VerificationError("inference schema mismatch")
    for key in ("predictions", "attention_evidence"):
        _validate_file_record(inference.get(key), label=f"inference {key}")
        if dict(inference[key]) != dict(artifacts[key]):
            raise VerificationError(
                f"completion artifact {key} is not cross-bound to inference"
            )

    metrics = _load_json_snapshot(leaves["metrics"])
    _require_exact_fields(metrics, _METRICS_FIELDS, label="metrics")
    if metrics.get("schema") != f"{SCHEMA_VERSION}-metrics":
        raise VerificationError("metrics schema mismatch")

    nested_count = 1  # training checkpoint
    for field in ("code", "inputs"):
        records = manifest.get(field)
        if not isinstance(records, list) or not records:
            raise VerificationError(f"run manifest {field} must be a non-empty list")
        for index, record in enumerate(records):
            _snapshot_file_record(
                record,
                relative_to=ROOT,
                label=f"run manifest {field}[{index}]",
            )
            nested_count += 1
    return {
        "direct_leaf_records": len(leaves),
        "nested_leaf_records": nested_count,
    }


def verify_detached_verification_package(
    receipt_path: Path,
    *,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Read-only replay of the hash bindings rooted in a detached receipt."""

    receipt_path = _absolute_without_resolving(receipt_path)
    _assert_no_link_or_reparse(receipt_path)
    receipt_snapshot = _read_artifact_snapshot(
        receipt_path, relative_to=receipt_path.parent
    )
    receipt = _load_json_snapshot(receipt_snapshot)
    if set(receipt) != _DETACHED_RECEIPT_FIELDS:
        raise VerificationError("detached verification receipt field set mismatch")
    if receipt.get("schema") != f"{SCHEMA_VERSION}-detached-verification-receipt":
        raise VerificationError("detached verification receipt schema mismatch")

    recorded_output = receipt.get("source_output_root")
    if not isinstance(recorded_output, str) or not recorded_output:
        raise VerificationError("detached receipt source_output_root is invalid")
    if output_root is not None:
        resolved_output = _resolve_detached_path(
            output_root, label="selected output_root", strict=True
        )
    else:
        resolved_output = _resolve_detached_path(
            Path(recorded_output), label="receipt source_output_root", strict=True
        )
    if not resolved_output.is_dir():
        raise VerificationError(
            f"verification output_root is not a directory: {resolved_output}"
        )
    resolved_receipt_path = _resolve_detached_path(
        receipt_snapshot.path, label="detached verification receipt", strict=True
    )
    if _is_within(resolved_receipt_path, resolved_output):
        raise VerificationError(
            "detached verification receipt must be outside output_root"
        )

    summary_record = receipt.get("summary")
    _validate_file_record(summary_record, label="detached receipt summary")
    summary_recorded_path = Path(str(summary_record["path"]))
    summary_lexical_path = (
        summary_recorded_path
        if summary_recorded_path.is_absolute()
        else receipt_snapshot.path.parent / summary_recorded_path
    )
    _assert_no_link_or_reparse(summary_lexical_path)
    summary_snapshot = _snapshot_file_record(
        summary_record,
        relative_to=receipt_snapshot.path.parent,
        label="detached receipt summary",
    )
    resolved_summary_path = _resolve_detached_path(
        summary_snapshot.path, label="detached verification summary", strict=True
    )
    if _is_within(resolved_summary_path, resolved_output):
        raise VerificationError(
            "detached verification summary must be outside output_root"
        )
    summary = _load_json_snapshot(summary_snapshot)
    for field in ("scope", "verification_passed", "scientific_result"):
        if receipt.get(field) != summary.get(field):
            raise VerificationError(
                f"detached receipt {field} disagrees with bound summary"
            )

    summary_roots = _validated_summary_completion_roots(summary)
    receipt_roots = receipt.get("accepted_attempt_completion_roots")
    if not isinstance(receipt_roots, list):
        raise VerificationError("detached receipt completion roots must be a list")
    normalized_receipt_roots: list[dict[str, Any]] = []
    for index, record in enumerate(receipt_roots):
        _validate_file_record(
            record, label=f"detached receipt completion root[{index}]"
        )
        normalized_receipt_roots.append(dict(record))
    if normalized_receipt_roots != summary_roots:
        raise VerificationError(
            "detached receipt completion roots disagree with bound summary"
        )
    direct_leaf_count = 0
    nested_leaf_count = 0
    for record in normalized_receipt_roots:
        counts = _verify_detached_completion_tree(
            record, output_root=resolved_output
        )
        direct_leaf_count += counts["direct_leaf_records"]
        nested_leaf_count += counts["nested_leaf_records"]

    verifier_records = receipt.get("verifier_code")
    if not isinstance(verifier_records, list) or not verifier_records:
        raise VerificationError("detached receipt verifier_code is invalid")
    for index, record in enumerate(verifier_records):
        _snapshot_file_record(
            record,
            relative_to=ROOT,
            label=f"detached receipt verifier_code[{index}]",
        )
    return {
        "schema": f"{SCHEMA_VERSION}-detached-verification-replay",
        "receipt_verified": True,
        "receipt": _snapshot_record(
            receipt_snapshot, relative_to=receipt_snapshot.path.parent
        ),
        "source_output_root": recorded_output,
        "verification_output_root": str(resolved_output),
        "scope": summary.get("scope"),
        "verification_passed": summary.get("verification_passed"),
        "scientific_result": summary.get("scientific_result"),
        "n_completion_roots_verified": len(summary_roots),
        "n_direct_leaf_records_verified": direct_leaf_count,
        "n_nested_leaf_records_verified": nested_leaf_count,
        "n_verifier_code_files_verified": len(verifier_records),
        "assurance_scope": (
            "Read-only checksum replay of an unsigned detached receipt; this does "
            "not establish independent implementation or archive immutability."
        ),
    }


def _panel_indices(genes: Sequence[str], panel_size: int = PANEL_SIZE) -> np.ndarray:
    if panel_size < 1:
        raise ValueError("panel_size must be positive")
    ranked = sorted(
        range(len(genes)),
        key=lambda idx: (
            hashlib.sha256(
                f"DepPLM-attention-panel-v1\0{genes[idx]}".encode("utf-8")
            ).digest(),
            str(genes[idx]),
        ),
    )
    return np.asarray(ranked[: min(panel_size, len(ranked))], dtype=np.int64)


def _training_token_covariates(
    fold_data: corrected.FoldData,
    outer_train: np.ndarray,
    edge_index: np.ndarray,
    genes: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return fold-training prevalence counts and fixed STRING record degree."""
    gene_to_index = {str(gene): idx for idx, gene in enumerate(genes)}
    vocabulary_index = {
        str(gene): idx for idx, gene in enumerate(fold_data.mutation.vocabulary)
    }
    prevalence = np.zeros(len(genes), dtype=np.float32)
    train_matrix = fold_data.mutation.matrix[np.asarray(outer_train, dtype=np.int64)]
    counts = train_matrix.sum(axis=0, dtype=np.float64)
    for gene, column in vocabulary_index.items():
        gene_idx = gene_to_index.get(gene)
        if gene_idx is not None:
            prevalence[gene_idx] = float(counts[column])

    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim != 2:
        raise ValueError("edge_index must be two-dimensional")
    if edges.shape[0] != 2 and edges.shape[1] == 2:
        edges = edges.T
    if edges.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E) or (E, 2)")
    degree = np.bincount(edges.reshape(-1), minlength=len(genes)).astype(np.int32)
    return prevalence, degree


def _mask_count(n_valid: int, fraction: float) -> int:
    if n_valid < 0:
        raise ValueError("n_valid cannot be negative")
    if not 0.0 < float(fraction) <= 0.5:
        raise ValueError("mask fractions must be in (0, 0.5]")
    if n_valid == 0:
        return 0
    return max(1, int(math.floor(n_valid * float(fraction))))


def build_mask_plan(
    mean_attention: np.ndarray,
    token_indices: np.ndarray,
    token_validity: np.ndarray,
    prevalence_count: np.ndarray,
    string_degree: np.ndarray,
    *,
    fraction: float,
    random_seeds: Sequence[int],
    nearest_candidates: int = MATCH_NEAREST_CANDIDATES,
) -> MaskPlan:
    """Build fixed top-attention and frequency/degree-matched random masks.

    Each random mask is restricted to the cell's valid mutation tokens and has
    exactly the same cardinality as its top-attention mask.  Matching uses the
    two preregistered covariates ``log1p(training prevalence count)`` and
    ``log1p(STRING edge-record degree)``.  For every top token, a seeded draw is
    made among the nearest available non-top candidates.  A one-token cell has
    no alternative; the unavoidable overlap is recorded in feature column 6.

    Matching feature columns are: top prevalence mean, control prevalence mean,
    top degree mean, control degree mean, standardized L1 imbalance, overlap
    fraction, and fallback count.
    """
    attention = np.asarray(mean_attention, dtype=np.float64)
    tokens = np.asarray(token_indices, dtype=np.int64)
    valid = np.asarray(token_validity, dtype=bool)
    prevalence = np.asarray(prevalence_count, dtype=np.float64)
    degree = np.asarray(string_degree, dtype=np.float64)
    if attention.shape != tokens.shape or valid.shape != tokens.shape:
        raise ValueError("attention, token_indices, and token_validity must align")
    if tokens.ndim != 2:
        raise ValueError("token arrays must be two-dimensional")
    if valid.any():
        used = tokens[valid]
        if int(used.min()) < 0 or int(used.max()) >= prevalence.size:
            raise IndexError("valid token index is outside the covariate arrays")
    if prevalence.shape != degree.shape or prevalence.ndim != 1:
        raise ValueError("prevalence and degree must be aligned vectors")
    if nearest_candidates < 1:
        raise ValueError("nearest_candidates must be positive")
    if not random_seeds:
        raise ValueError("at least one matched-random seed is required")

    transformed = np.column_stack([np.log1p(prevalence), np.log1p(degree)]).astype(
        np.float64
    )
    centre = np.median(transformed, axis=0)
    scale = np.median(np.abs(transformed - centre), axis=0) * 1.4826
    scale[~np.isfinite(scale) | (scale <= 0.0)] = 1.0
    standardized = (transformed - centre) / scale

    top_mask = np.zeros_like(valid)
    ranked_top: list[np.ndarray] = []
    for row in range(tokens.shape[0]):
        positions = np.flatnonzero(valid[row])
        count = _mask_count(int(positions.size), fraction)
        if count == 0:
            ranked_top.append(np.empty(0, dtype=np.int64))
            continue
        order = np.lexsort((positions, -attention[row, positions]))
        chosen = positions[order[:count]]
        top_mask[row, chosen] = True
        ranked_top.append(chosen)

    matched_masks: list[np.ndarray] = []
    feature_arrays: list[np.ndarray] = []
    for seed in random_seeds:
        rng = np.random.default_rng(int(seed))
        matched = np.zeros_like(valid)
        features = np.full((tokens.shape[0], 7), np.nan, dtype=np.float32)
        for row, top_positions in enumerate(ranked_top):
            valid_positions = np.flatnonzero(valid[row])
            count = int(top_positions.size)
            if count == 0:
                features[row] = 0.0
                continue
            top_set = set(int(value) for value in top_positions)
            available = [
                int(value) for value in valid_positions if int(value) not in top_set
            ]
            fallback_count = 0
            selected: list[int] = []
            for top_position in top_positions:
                if not available:
                    fallback_count += 1
                    available = [
                        int(value)
                        for value in valid_positions
                        if int(value) not in selected
                    ]
                top_gene = int(tokens[row, int(top_position)])
                distances = np.asarray(
                    [
                        np.abs(
                            standardized[top_gene]
                            - standardized[int(tokens[row, candidate])]
                        ).sum()
                        for candidate in available
                    ],
                    dtype=np.float64,
                )
                stable = np.asarray(available, dtype=np.int64)
                ranked = np.lexsort((stable, distances))
                pool = ranked[: min(int(nearest_candidates), ranked.size)]
                selected_rank = int(pool[int(rng.integers(pool.size))])
                chosen = int(available[selected_rank])
                selected.append(chosen)
                available.pop(selected_rank)
            matched[row, np.asarray(selected, dtype=np.int64)] = True
            top_genes = tokens[row, top_positions]
            control_genes = tokens[row, np.asarray(selected, dtype=np.int64)]
            top_cov = transformed[top_genes].mean(axis=0)
            control_cov = transformed[control_genes].mean(axis=0)
            imbalance = np.abs(
                standardized[top_genes].mean(axis=0)
                - standardized[control_genes].mean(axis=0)
            ).sum()
            overlap = len(top_set.intersection(selected)) / count
            features[row] = np.asarray(
                [
                    top_cov[0],
                    control_cov[0],
                    top_cov[1],
                    control_cov[1],
                    imbalance,
                    overlap,
                    fallback_count,
                ],
                dtype=np.float32,
            )
        if not np.array_equal(matched.sum(axis=1), top_mask.sum(axis=1)):
            raise RuntimeError("matched masks do not preserve per-cell mask counts")
        if np.any(matched & ~valid):
            raise RuntimeError("matched masks include invalid token slots")
        matched_masks.append(matched)
        feature_arrays.append(features)
    return MaskPlan(
        top_mask=top_mask,
        matched_masks=tuple(matched_masks),
        matching_features=tuple(feature_arrays),
        random_seeds=tuple(int(value) for value in random_seeds),
    )


def build_attention_permutation(token_validity: np.ndarray, *, seed: int) -> np.ndarray:
    """Return a fixed per-cell permutation restricted to valid token slots."""
    validity = np.asarray(token_validity, dtype=bool)
    if validity.ndim != 2:
        raise ValueError("token_validity must be two-dimensional")
    rng = np.random.default_rng(int(seed))
    permutation = np.broadcast_to(
        np.arange(validity.shape[1], dtype=np.int64), validity.shape
    ).copy()
    for row in range(validity.shape[0]):
        positions = np.flatnonzero(validity[row])
        permutation[row, positions] = rng.permutation(positions)
    return permutation


def _matching_quality_diagnostic(
    features: np.ndarray, *, fraction_key: str, draw: int
) -> dict[str, Any]:
    values = np.asarray(features)
    if values.ndim != 2 or values.shape[1] != len(MATCHING_FEATURE_COLUMNS):
        raise VerificationError("matching feature matrix shape is invalid")
    fallback = np.asarray(values[:, MATCHING_FALLBACK_COLUMN], dtype=np.float64)
    overlap = np.asarray(values[:, MATCHING_OVERLAP_COLUMN], dtype=np.float64)
    if not np.isfinite(fallback).all() or not np.isfinite(overlap).all():
        raise VerificationError("matching quality fields contain non-finite values")
    if (
        np.any(fallback < 0.0)
        or np.any(fallback != np.floor(fallback))
        or np.any(overlap < 0.0)
        or np.any(overlap > 1.0)
    ):
        raise VerificationError("matching quality fields are outside their valid ranges")
    fallback_rows = int(np.count_nonzero(fallback > 0.0))
    overlap_rows = int(np.count_nonzero(overlap > 0.0))
    return {
        "fraction": fraction_key,
        "draw": int(draw),
        "fallback_rows": fallback_rows,
        "fallback_count_total": int(fallback.sum()),
        "overlap_rows": overlap_rows,
        "max_overlap_fraction": float(overlap.max()) if overlap.size else 0.0,
        "passed": fallback_rows == 0 and overlap_rows == 0,
    }


def _attention_details(
    model: DepPLM,
    z_query: torch.Tensor,
    token_indices: torch.Tensor,
    token_validity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct DepPLM attention exactly while exposing zero-filled scores."""
    batch_size, n_tokens = token_indices.shape
    query = model.wq(z_query)
    safe_tokens = token_indices.clamp(min=0)
    token_repr = model.proj(model.emb[safe_tokens.reshape(-1)]).reshape(
        batch_size, n_tokens, -1
    )
    token_repr = torch.where(token_validity.unsqueeze(-1), token_repr, model.pad_token)
    keys = model.wk(token_repr)
    hidden = query.shape[-1]
    scores = (query @ keys.reshape(batch_size * n_tokens, hidden).T).reshape(
        query.shape[0], batch_size, n_tokens
    ) / math.sqrt(hidden)
    scores = scores.permute(1, 0, 2)
    scores = scores.masked_fill(~token_validity.unsqueeze(1), float("-inf"))
    all_masked = (~token_validity.unsqueeze(1)).all(-1, keepdim=True)
    scores = torch.where(all_masked.expand_as(scores), torch.zeros_like(scores), scores)
    attention = torch.softmax(scores, dim=-1)
    scores_zero = torch.where(scores == float("-inf"), torch.zeros_like(scores), scores)
    return scores_zero, attention


def infer_with_attention(
    model: DepPLM,
    fold_data: corrected.FoldData,
    indices: np.ndarray,
    *,
    device: str,
    batch_size: int,
    gene_chunk_size: int,
    token_validity_override: np.ndarray | None = None,
    attention_permutation: np.ndarray | None = None,
    panel_indices: np.ndarray | None = None,
    collect_attention: bool = False,
) -> InferenceEvidence:
    """Predict without labels, optionally collecting compact attention evidence."""
    if not isinstance(model, DepPLM) or not model.use_crossattn:
        raise TypeError("attention faithfulness requires a cross-attentive DepPLM")
    selected = np.asarray(indices, dtype=np.int64)
    if selected.ndim != 1 or np.unique(selected).size != selected.size:
        raise ValueError("indices must be a unique one-dimensional vector")
    if batch_size < 1 or gene_chunk_size < 1:
        raise ValueError("inference batch and gene chunk sizes must be positive")
    original_validity = fold_data.mutation.token_mask[selected, :K_TOKENS]
    if token_validity_override is None:
        effective_validity = original_validity
    else:
        effective_validity = np.asarray(token_validity_override, dtype=bool)
        if effective_validity.shape != original_validity.shape:
            raise ValueError("token_validity_override has the wrong shape")
        if np.any(effective_validity & ~original_validity):
            raise ValueError("token_validity_override activates an invalid token")
    if attention_permutation is not None:
        permutation = np.asarray(attention_permutation, dtype=np.int64)
        if permutation.shape != original_validity.shape:
            raise ValueError("attention_permutation has the wrong shape")
        expected = np.arange(permutation.shape[1], dtype=np.int64)
        for row in range(permutation.shape[0]):
            if not np.array_equal(np.sort(permutation[row]), expected):
                raise ValueError("each attention permutation row must be bijective")
    else:
        permutation = None
    panel = (
        np.asarray(panel_indices, dtype=np.int64)
        if panel_indices is not None
        else np.empty(0, dtype=np.int64)
    )

    model = model.to(device)
    model.eval()
    predictions: list[np.ndarray] = []
    attention_means: list[np.ndarray] = []
    panel_values: list[np.ndarray] = []
    with torch.inference_mode():
        z_all = model.encode_genes()
        for batch_start in range(0, selected.size, int(batch_size)):
            batch_stop = min(batch_start + int(batch_size), selected.size)
            cell_idx = selected[batch_start:batch_stop]
            mutation = torch.from_numpy(fold_data.mutation.matrix[cell_idx]).to(device)
            expression = torch.from_numpy(fold_data.expression[cell_idx]).to(device)
            lineage = torch.from_numpy(fold_data.lineage[cell_idx]).to(device)
            tokens = torch.from_numpy(
                fold_data.mutation.tokens[cell_idx, :K_TOKENS]
            ).to(device)
            validity = torch.from_numpy(effective_validity[batch_start:batch_stop]).to(
                device
            )
            cell_repr = model.cell(mutation, expression, lineage)
            chunks: list[np.ndarray] = []
            mean_sum = torch.zeros(
                (cell_idx.size, K_TOKENS), dtype=torch.float64, device=device
            )
            perm_tensor = (
                torch.from_numpy(permutation[batch_start:batch_stop]).to(device)
                if permutation is not None
                else None
            )
            for gene_start in range(0, z_all.shape[0], int(gene_chunk_size)):
                gene_stop = min(gene_start + int(gene_chunk_size), z_all.shape[0])
                gene_idx = torch.arange(gene_start, gene_stop, device=device)
                z_query = z_all[gene_start:gene_stop]
                baseline = model.head(z_query, cell_repr, gene_idx)
                scores, attention = _attention_details(model, z_query, tokens, validity)
                if collect_attention:
                    mean_sum += attention.double().sum(dim=1)
                if perm_tensor is not None:
                    gather_idx = perm_tensor.unsqueeze(1).expand(
                        -1, attention.shape[1], -1
                    )
                    attention = torch.gather(attention, 2, gather_idx)
                energy = (attention * scores).sum(dim=-1)
                prediction = baseline + model.attn_scale * energy
                chunks.append(prediction.float().cpu().numpy())
            predictions.append(np.concatenate(chunks, axis=1))
            if collect_attention:
                attention_means.append(
                    (mean_sum / float(z_all.shape[0])).float().cpu().numpy()
                )
                if panel.size:
                    panel_tensor = torch.from_numpy(panel).to(device)
                    _, panel_attention = _attention_details(
                        model, z_all[panel_tensor], tokens, validity
                    )
                    panel_values.append(panel_attention.float().cpu().numpy())
    return InferenceEvidence(
        prediction=np.ascontiguousarray(
            np.concatenate(predictions, axis=0), dtype=np.float32
        ),
        mean_attention=(
            np.ascontiguousarray(np.concatenate(attention_means), dtype=np.float32)
            if collect_attention
            else None
        ),
        panel_attention=(
            np.ascontiguousarray(np.concatenate(panel_values), dtype=np.float32)
            if collect_attention and panel.size
            else None
        ),
    )


def _hash_attention_parameters(model: DepPLM) -> str:
    digest = hashlib.sha256()
    for name in ("wq.weight", "wq.bias", "wk.weight", "wk.bias"):
        module_name, parameter_name = name.split(".")
        parameter = getattr(getattr(model, module_name), parameter_name)
        array = parameter.detach().float().cpu().numpy()
        digest.update(name.encode("ascii"))
        digest.update(_hash_array(array).encode("ascii"))
    return digest.hexdigest()


def _randomize_attention_parameters(model: DepPLM, *, seed: int) -> None:
    randomized_wq = copy.deepcopy(model.wq).cpu()
    randomized_wk = copy.deepcopy(model.wk).cpu()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        randomized_wq.reset_parameters()
        randomized_wk.reset_parameters()
    model.wq.load_state_dict(randomized_wq.state_dict())
    model.wk.load_state_dict(randomized_wk.state_dict())


def parameter_randomized_inference(
    model: DepPLM,
    fold_data: corrected.FoldData,
    indices: np.ndarray,
    *,
    seed: int,
    device: str,
    batch_size: int,
    gene_chunk_size: int,
    panel_indices: np.ndarray,
) -> tuple[InferenceEvidence, dict[str, Any]]:
    """Randomize only Wq/Wk, infer, then byte-restore the trained parameters."""
    original = {
        "wq": copy.deepcopy(model.wq.state_dict()),
        "wk": copy.deepcopy(model.wk.state_dict()),
    }
    trained_hash = _hash_attention_parameters(model)
    try:
        _randomize_attention_parameters(model, seed=seed)
        randomized_hash = _hash_attention_parameters(model)
        evidence = infer_with_attention(
            model,
            fold_data,
            indices,
            device=device,
            batch_size=batch_size,
            gene_chunk_size=gene_chunk_size,
            panel_indices=panel_indices,
            collect_attention=True,
        )
    finally:
        model.wq.load_state_dict(original["wq"])
        model.wk.load_state_dict(original["wk"])
    restored_hash = _hash_attention_parameters(model)
    if restored_hash != trained_hash:
        raise RuntimeError("trained attention parameters were not restored exactly")
    if randomized_hash == trained_hash:
        raise RuntimeError("parameter randomization did not change Wq/Wk")
    return evidence, {
        "scope": "wq_and_wk_only",
        "initialization": "torch.nn.Linear.reset_parameters_on_cpu_then_copy",
        "seed": int(seed),
        "trained_sha256": trained_hash,
        "randomized_sha256": randomized_hash,
        "restored_sha256": restored_hash,
    }


def _top_partner_indices(
    panel_attention: np.ndarray,
    token_indices: np.ndarray,
    token_validity: np.ndarray,
    *,
    n_genes: int,
    top_n: int = TOP_PARTNERS,
) -> np.ndarray:
    panel_values = np.asarray(panel_attention, dtype=np.float64)
    tokens = np.asarray(token_indices, dtype=np.int64)
    valid = np.asarray(token_validity, dtype=bool)
    if panel_values.ndim != 3 or panel_values.shape[0] != tokens.shape[0]:
        raise ValueError("panel attention and token evidence do not align")
    if panel_values.shape[2] != tokens.shape[1] or valid.shape != tokens.shape:
        raise ValueError("panel attention and token slots do not align")
    totals = np.zeros((panel_values.shape[1], n_genes), dtype=np.float64)
    for row in range(tokens.shape[0]):
        positions = np.flatnonzero(valid[row])
        if positions.size:
            np.add.at(
                totals,
                (slice(None), tokens[row, positions]),
                panel_values[row][:, positions],
            )
    result = np.full((totals.shape[0], min(top_n, n_genes)), -1, dtype=np.int64)
    gene_indices = np.arange(n_genes, dtype=np.int64)
    for row in range(totals.shape[0]):
        positive = totals[row] > 0.0
        ranked = np.lexsort((gene_indices, -totals[row]))
        ranked = ranked[positive[ranked]]
        result[row, : min(result.shape[1], ranked.size)] = ranked[: result.shape[1]]
    return result


def _mode_metrics(
    predictions: Mapping[str, np.ndarray],
    target: np.ndarray,
    train_gene_mean: np.ndarray,
) -> dict[str, Any]:
    baseline = np.asarray(predictions["baseline"], dtype=np.float32)
    result: dict[str, Any] = {}
    for name, prediction in predictions.items():
        array = np.asarray(prediction, dtype=np.float32)
        metrics = dependency_metrics(array, target, train_gene_mean)
        metrics["mean_absolute_prediction_change_vs_baseline"] = float(
            np.mean(np.abs(array.astype(np.float64) - baseline.astype(np.float64)))
        )
        result[name] = metrics
    for fraction in MASK_FRACTIONS:
        key = _fraction_key(fraction)
        top = result[f"top_mask_{key}"]
        controls = [
            result[f"matched_mask_{key}_d{draw:02d}"]
            for draw in range(MATCHED_RANDOM_DRAWS)
        ]
        result[f"contrast_{key}"] = {
            "top_minus_matched_mean_delta_mse": float(
                (top["mse"] - result["baseline"]["mse"])
                - np.mean(
                    [item["mse"] - result["baseline"]["mse"] for item in controls]
                )
            ),
            "top_minus_matched_mean_delta_residual_r2": float(
                (
                    top["residual_r2_vs_train_gene_mean"]
                    - result["baseline"]["residual_r2_vs_train_gene_mean"]
                )
                - np.mean(
                    [
                        item["residual_r2_vs_train_gene_mean"]
                        - result["baseline"]["residual_r2_vs_train_gene_mean"]
                        for item in controls
                    ]
                )
            ),
            "n_matched_random_draws": MATCHED_RANDOM_DRAWS,
        }
    return result


def _arrays_close(left: Any, right: Any, *, path: str = "root") -> None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            raise VerificationError(f"metric keys differ at {path}")
        for key in left:
            _arrays_close(left[key], right[key], path=f"{path}.{key}")
        return
    if isinstance(left, (int, str, bool)) or left is None:
        if left != right:
            raise VerificationError(f"value differs at {path}: {left!r} != {right!r}")
        return
    left_value = float(left)
    right_value = float(right)
    if not np.isclose(left_value, right_value, rtol=1e-6, atol=FLOAT_TOLERANCE):
        raise VerificationError(
            f"numeric value differs at {path}: {left_value!r} != {right_value!r}"
        )


def _verify_numeric_replay(
    saved: np.ndarray,
    replayed: np.ndarray,
    *,
    label: str,
) -> dict[str, Any]:
    expected = np.asarray(saved)
    observed = np.asarray(replayed)
    if expected.shape != observed.shape:
        raise VerificationError(
            f"numeric replay shape mismatch for {label}: "
            f"{observed.shape} != {expected.shape}"
        )
    if expected.dtype != observed.dtype:
        raise VerificationError(
            f"numeric replay dtype mismatch for {label}: "
            f"{observed.dtype} != {expected.dtype}"
        )
    if not np.isfinite(expected).all() or not np.isfinite(observed).all():
        raise VerificationError(
            f"numeric replay contains non-finite values for {label}"
        )
    expected64 = expected.astype(np.float64, copy=False)
    observed64 = observed.astype(np.float64, copy=False)
    absolute_error = np.abs(observed64 - expected64)
    close = np.isclose(
        observed64,
        expected64,
        rtol=REPLAY_RTOL,
        atol=REPLAY_ATOL,
        equal_nan=False,
    )
    max_absolute_error = float(absolute_error.max()) if absolute_error.size else 0.0
    if not bool(np.all(close)):
        raise VerificationError(
            f"numeric replay mismatch for {label}: "
            f"{int(close.size - np.count_nonzero(close))}/{close.size} values differ, "
            f"max_absolute_error={max_absolute_error:.9g}, "
            f"rtol={REPLAY_RTOL}, atol={REPLAY_ATOL}"
        )
    scale = np.maximum(np.abs(expected64), REPLAY_ATOL)
    relative_error = absolute_error / scale
    return {
        "shape": [int(value) for value in expected.shape],
        "dtype": str(expected.dtype),
        "saved_sha256": _hash_array(expected),
        "replayed_sha256": _hash_array(observed),
        "bitwise_equal": bool(np.array_equal(expected, observed)),
        "max_absolute_error": max_absolute_error,
        "mean_absolute_error": (
            float(absolute_error.mean()) if absolute_error.size else 0.0
        ),
        "max_relative_error": (
            float(relative_error.max()) if relative_error.size else 0.0
        ),
        "rtol": REPLAY_RTOL,
        "atol": REPLAY_ATOL,
        "matched": True,
    }


def _prepare_replay_runtime(device: str) -> dict[str, Any]:
    """Enable and record the fail-closed deterministic replay contract."""
    if device not in {"cpu", "cuda"}:
        raise VerificationError(f"unsupported replay device: {device!r}")
    cuda_initialized_before_setup = bool(torch.cuda.is_initialized())
    if device == "cuda":
        cublas_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if cublas_config is None:
            if cuda_initialized_before_setup:
                raise VerificationError(
                    "CUDA was initialized before deterministic CUBLAS replay setup"
                )
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        elif cublas_config not in {":4096:8", ":16:8"}:
            raise VerificationError(
                "CUDA replay requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8"
            )

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if device == "cuda" and not torch.cuda.is_available():
        raise VerificationError("CUDA replay was requested but CUDA is unavailable")
    if not torch.are_deterministic_algorithms_enabled():
        raise VerificationError("strict deterministic algorithms are not enabled")
    if torch.is_deterministic_algorithms_warn_only_enabled():
        raise VerificationError("deterministic replay cannot use warn-only mode")
    if not torch.backends.cudnn.deterministic or torch.backends.cudnn.benchmark:
        raise VerificationError("deterministic cuDNN replay settings are not active")
    return {
        "device": device,
        "torch_version": torch.__version__,
        "cuda_build_version": torch.version.cuda,
        "cuda_initialized_before_setup": cuda_initialized_before_setup,
        "cublas_workspace_config": (
            os.environ.get("CUBLAS_WORKSPACE_CONFIG") if device == "cuda" else None
        ),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def _validate_replay_device(device: str) -> None:
    """Compatibility wrapper retained for callers that only need validation."""
    _prepare_replay_runtime(device)


def _replay_prediction_modes(
    model: DepPLM,
    fold_data: corrected.FoldData,
    outer_test: np.ndarray,
    *,
    saved_predictions: Mapping[str, np.ndarray],
    evidence: Mapping[str, np.ndarray],
    panel_indices: np.ndarray,
    attention_permutation: np.ndarray,
    randomization_seed: int,
    expected_randomization_record: Mapping[str, Any],
    device: str,
    batch_size: int,
    gene_chunk_size: int,
) -> dict[str, Any]:
    """Reconstruct and numerically replay every persisted prediction mode."""
    replay_runtime = _prepare_replay_runtime(device)
    if batch_size < 1 or gene_chunk_size < 1:
        raise VerificationError(
            "recorded replay batch and chunk sizes must be positive"
        )
    validity = np.asarray(evidence["token_validity"], dtype=bool)
    prediction_records: dict[str, dict[str, Any]] = {}
    attention_records: dict[str, dict[str, Any]] = {}

    baseline = infer_with_attention(
        model,
        fold_data,
        outer_test,
        device=device,
        batch_size=batch_size,
        gene_chunk_size=gene_chunk_size,
        panel_indices=panel_indices,
        collect_attention=True,
    )
    if baseline.mean_attention is None or baseline.panel_attention is None:
        raise VerificationError("baseline replay omitted required attention evidence")
    prediction_records["baseline"] = _verify_numeric_replay(
        saved_predictions["baseline"], baseline.prediction, label="prediction.baseline"
    )
    attention_records["mean_attention"] = _verify_numeric_replay(
        evidence["mean_attention"],
        baseline.mean_attention,
        label="attention.mean_attention",
    )
    attention_records["panel_attention"] = _verify_numeric_replay(
        evidence["panel_attention"],
        baseline.panel_attention,
        label="attention.panel_attention",
    )

    for fraction in MASK_FRACTIONS:
        key = _fraction_key(fraction)
        top_name = f"top_mask_{key}"
        top_mask = np.asarray(evidence[top_name], dtype=bool)
        replayed = infer_with_attention(
            model,
            fold_data,
            outer_test,
            device=device,
            batch_size=batch_size,
            gene_chunk_size=gene_chunk_size,
            token_validity_override=validity & ~top_mask,
        ).prediction
        prediction_records[top_name] = _verify_numeric_replay(
            saved_predictions[top_name], replayed, label=f"prediction.{top_name}"
        )
        for draw in range(MATCHED_RANDOM_DRAWS):
            matched_name = f"matched_mask_{key}_d{draw:02d}"
            matched_mask = np.asarray(evidence[matched_name], dtype=bool)
            replayed = infer_with_attention(
                model,
                fold_data,
                outer_test,
                device=device,
                batch_size=batch_size,
                gene_chunk_size=gene_chunk_size,
                token_validity_override=validity & ~matched_mask,
            ).prediction
            prediction_records[matched_name] = _verify_numeric_replay(
                saved_predictions[matched_name],
                replayed,
                label=f"prediction.{matched_name}",
            )

    permuted = infer_with_attention(
        model,
        fold_data,
        outer_test,
        device=device,
        batch_size=batch_size,
        gene_chunk_size=gene_chunk_size,
        attention_permutation=attention_permutation,
    ).prediction
    prediction_records["attention_permuted"] = _verify_numeric_replay(
        saved_predictions["attention_permuted"],
        permuted,
        label="prediction.attention_permuted",
    )

    randomized, randomization_record = parameter_randomized_inference(
        model,
        fold_data,
        outer_test,
        seed=randomization_seed,
        device=device,
        batch_size=batch_size,
        gene_chunk_size=gene_chunk_size,
        panel_indices=panel_indices,
    )
    _arrays_close(
        expected_randomization_record,
        randomization_record,
        path="replay.parameter_randomization",
    )
    if randomized.mean_attention is None or randomized.panel_attention is None:
        raise VerificationError(
            "parameter-randomized replay omitted attention evidence"
        )
    prediction_records["parameter_randomized"] = _verify_numeric_replay(
        saved_predictions["parameter_randomized"],
        randomized.prediction,
        label="prediction.parameter_randomized",
    )
    attention_records["randomized_mean_attention"] = _verify_numeric_replay(
        evidence["randomized_mean_attention"],
        randomized.mean_attention,
        label="attention.randomized_mean_attention",
    )
    attention_records["randomized_panel_attention"] = _verify_numeric_replay(
        evidence["randomized_panel_attention"],
        randomized.panel_attention,
        label="attention.randomized_panel_attention",
    )

    required_modes = set(_required_prediction_modes())
    if set(prediction_records) != required_modes:
        raise VerificationError("numeric replay did not cover all 15 prediction modes")
    return {
        "device": device,
        "deterministic_runtime": replay_runtime,
        "batch_size": int(batch_size),
        "gene_chunk_size": int(gene_chunk_size),
        "rtol": REPLAY_RTOL,
        "atol": REPLAY_ATOL,
        "required_prediction_modes": list(_required_prediction_modes()),
        "required_prediction_mode_count": len(required_modes),
        "replayed_prediction_mode_count": len(prediction_records),
        "all_prediction_modes_matched": True,
        "prediction_arrays": prediction_records,
        "attention_evidence_arrays": attention_records,
        "parameter_randomization": randomization_record,
    }


def _finite_metric(mode: Mapping[str, Any], name: str, *, path: str) -> float:
    try:
        value = float(mode[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise VerificationError(f"missing numeric metric {path}.{name}") from exc
    if not np.isfinite(value):
        raise VerificationError(f"non-finite metric {path}.{name}")
    return value


def _paired_fold_descriptives(modes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Create prespecified within-fold paired contrasts without IID inference."""
    baseline = modes.get("baseline")
    if not isinstance(baseline, Mapping):
        raise VerificationError("baseline metrics are missing from paired analysis")
    metric_names = (
        "mean_absolute_prediction_change_vs_baseline",
        "mse",
        "residual_r2_vs_train_gene_mean",
    )
    baseline_values = {
        name: _finite_metric(baseline, name, path="baseline") for name in metric_names
    }
    top_vs_matched: dict[str, dict[str, float]] = {}
    for fraction in MASK_FRACTIONS:
        key = _fraction_key(fraction)
        top_name = f"top_mask_{key}"
        top = modes.get(top_name)
        controls = [
            modes.get(f"matched_mask_{key}_d{draw:02d}")
            for draw in range(MATCHED_RANDOM_DRAWS)
        ]
        if not isinstance(top, Mapping) or not all(
            isinstance(control, Mapping) for control in controls
        ):
            raise VerificationError(f"paired mask metrics are incomplete for {key}")
        differences: dict[str, float] = {}
        for metric_name in metric_names:
            top_delta = (
                _finite_metric(top, metric_name, path=top_name)
                - baseline_values[metric_name]
            )
            control_deltas = [
                _finite_metric(
                    control,
                    metric_name,
                    path=f"matched_mask_{key}_d{draw:02d}",
                )
                - baseline_values[metric_name]
                for draw, control in enumerate(controls)
            ]
            differences[metric_name] = float(top_delta - np.mean(control_deltas))
        top_vs_matched[key] = differences

    perturbations: dict[str, dict[str, float]] = {}
    for label, mode_name in (
        ("attention_permutation_vs_baseline", "attention_permuted"),
        ("wq_wk_randomization_vs_baseline", "parameter_randomized"),
    ):
        mode = modes.get(mode_name)
        if not isinstance(mode, Mapping):
            raise VerificationError(f"{mode_name} metrics are missing")
        perturbations[label] = {
            metric_name: float(
                _finite_metric(mode, metric_name, path=mode_name)
                - baseline_values[metric_name]
            )
            for metric_name in metric_names
        }
    return {
        "top_attention_mask_vs_matched_random": top_vs_matched,
        **perturbations,
    }


def _descriptive_scalar_summary(
    values_by_seed_fold: Sequence[tuple[int, int, float]],
) -> dict[str, Any]:
    if not values_by_seed_fold:
        raise VerificationError("cannot summarize an empty paired contrast")
    values = np.asarray([item[2] for item in values_by_seed_fold], dtype=np.float64)
    if not np.isfinite(values).all():
        raise VerificationError("paired contrast contains non-finite values")
    per_seed: list[dict[str, Any]] = []
    for seed in sorted({item[0] for item in values_by_seed_fold}):
        seed_values = np.asarray(
            [item[2] for item in values_by_seed_fold if item[0] == seed],
            dtype=np.float64,
        )
        per_seed.append(
            {
                "seed": int(seed),
                "n_folds": int(seed_values.size),
                "mean": float(seed_values.mean()),
            }
        )
    seed_means = np.asarray([item["mean"] for item in per_seed], dtype=np.float64)
    return {
        "n_seed_folds": int(values.size),
        "mean_across_seed_folds": float(values.mean()),
        "median_across_seed_folds": float(np.median(values)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "n_positive": int(np.count_nonzero(values > 0.0)),
        "n_zero": int(np.count_nonzero(values == 0.0)),
        "n_negative": int(np.count_nonzero(values < 0.0)),
        "per_seed_means": per_seed,
        "mean_of_seed_means": float(seed_means.mean()),
    }


def _aggregate_paired_descriptives(
    verified: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate paired fold estimates descriptively, preserving seed dependence."""
    expected = {
        (seed, fold) for seed in CANONICAL_SEEDS for fold in range(CANONICAL_FOLDS)
    }
    identities = {(int(item["seed"]), int(item["fold"])) for item in verified}
    if identities != expected or len(verified) != len(expected):
        raise VerificationError(
            "paired descriptive aggregation requires all 15 unique seed/folds"
        )

    ordered = sorted(verified, key=lambda item: (int(item["seed"]), int(item["fold"])))
    raw_rows = [
        {
            "seed": int(item["seed"]),
            "fold": int(item["fold"]),
            "paired_values": item["paired_descriptive_values"],
        }
        for item in ordered
    ]

    metric_names = (
        "mean_absolute_prediction_change_vs_baseline",
        "mse",
        "residual_r2_vs_train_gene_mean",
    )
    top_summary: dict[str, dict[str, Any]] = {}
    for fraction in MASK_FRACTIONS:
        key = _fraction_key(fraction)
        top_summary[key] = {}
        for metric_name in metric_names:
            values = [
                (
                    int(item["seed"]),
                    int(item["fold"]),
                    float(
                        item["paired_descriptive_values"][
                            "top_attention_mask_vs_matched_random"
                        ][key][metric_name]
                    ),
                )
                for item in ordered
            ]
            top_summary[key][metric_name] = _descriptive_scalar_summary(values)

    perturbation_summary: dict[str, dict[str, Any]] = {}
    for comparison in (
        "attention_permutation_vs_baseline",
        "wq_wk_randomization_vs_baseline",
    ):
        perturbation_summary[comparison] = {}
        for metric_name in metric_names:
            values = [
                (
                    int(item["seed"]),
                    int(item["fold"]),
                    float(item["paired_descriptive_values"][comparison][metric_name]),
                )
                for item in ordered
            ]
            perturbation_summary[comparison][metric_name] = _descriptive_scalar_summary(
                values
            )

    return {
        "analysis_status": "complete_prespecified_descriptive_analysis",
        "n_seed_folds": len(ordered),
        "pairing_unit": "within outer-test seed-fold",
        "matched_control_handling": (
            "The three matched-random draws are averaged within each seed-fold "
            "before the top-attention minus matched-random contrast is formed."
        ),
        "dependence_handling": (
            "Folds within a seed and repeated seeds over the same cells are not "
            "treated as IID. No IID p-values or confidence intervals are reported."
        ),
        "direction_note": (
            "For MSE and absolute prediction change, positive contrasts indicate "
            "greater disruption. For residual R2, negative contrasts indicate "
            "greater disruption. Estimates diagnose fitted-model faithfulness only."
        ),
        "top_attention_mask_vs_matched_random": top_summary,
        **perturbation_summary,
        "raw_seed_fold_values": raw_rows,
        "scope_statement": PROTOCOL_SCOPE,
    }


def _fold_splits(
    base: corrected.BaseData, *, seed: int, pilot: bool
) -> list[tuple[np.ndarray, np.ndarray]]:
    return make_kfold_outer_splits(
        len(base.cell_ids), 2 if pilot else CANONICAL_FOLDS, int(seed)
    )


def _attempt_root(output_root: Path, *, pilot: bool, seed: int, fold: int) -> Path:
    scope = "pilot" if pilot else "canonical"
    return output_root / scope / f"seed-{seed}" / f"fold-{fold:02d}"


def _new_attempt_path(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    name = f"attempt-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}-p{os.getpid()}-{uuid.uuid4().hex[:8]}"
    path = root / name
    path.mkdir(exist_ok=False)
    return path


def _code_records() -> list[dict[str, Any]]:
    return [_file_record(path) for path in CODE_PATHS]


def _input_records() -> list[dict[str, Any]]:
    return [_file_record(path) for path in INPUT_PATHS]


def _same_records(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> bool:
    return [dict(item) for item in left] == [dict(item) for item in right]


def _parameter_manifest() -> dict[str, Any]:
    return {
        "mask_fractions": list(MASK_FRACTIONS),
        "matched_random_draws": MATCHED_RANDOM_DRAWS,
        "match_nearest_candidates": MATCH_NEAREST_CANDIDATES,
        "matching_covariates": [
            "log1p_outer_train_mutation_prevalence_count",
            "log1p_STRING_edge_record_degree",
        ],
        "mask_cardinality": "floor(fraction * valid_token_count), minimum one",
        "validity_constraint": "valid mutation-token slots only",
        "attention_ranking": "mean attention over every predicted dependency gene",
        "attention_permutation": "within-cell valid-slot weight permutation",
        "parameter_randomization": (
            "Wq and Wk reset_parameters on CPU, then copied to the model device"
        ),
        "panel_selection": "64 genes with lowest SHA256 rank, label independent",
        "panel_top_partners": TOP_PARTNERS,
        "formal_replay_gate": (
            "reconstruct each checkpoint and numerically replay all 15 prediction "
            "modes plus baseline/randomized attention evidence"
        ),
        "replay_runtime_contract": (
            "strict PyTorch deterministic algorithms (not warn-only), cuDNN "
            "deterministic with benchmark disabled, and a supported CUBLAS "
            "workspace configuration established before CUDA initialization"
        ),
        "matched_control_quality_gate": (
            "formal verification requires zero matched-control fallback selections "
            "and zero overlap with the corresponding top-attention mask in every "
            "cell and draw; pilot verification retains these values as diagnostics"
        ),
        "replay_tolerance": {"rtol": REPLAY_RTOL, "atol": REPLAY_ATOL},
        "paired_aggregation": (
            "within-seed-fold paired descriptive contrasts; matched controls are "
            "averaged within fold; no IID p-values or confidence intervals"
        ),
        "scope_statement": PROTOCOL_SCOPE,
    }


def _run_manifest(
    *,
    seed: int,
    fold: int,
    pilot: bool,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    code_records: Sequence[Mapping[str, Any]],
    input_records: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA_VERSION}-run-manifest",
        "created_utc": _utc_now(),
        "seed": int(seed),
        "fold": int(fold),
        "pilot": bool(pilot),
        "scientific_result": False,
        "protocol": "corrected-nested-P1",
        "condition": "DepPLM",
        "outer_train_indices_sha256": _hash_array(
            np.asarray(outer_train, dtype=np.int64)
        ),
        "outer_test_indices_sha256": _hash_array(
            np.asarray(outer_test, dtype=np.int64)
        ),
        "arguments": {
            "device": args.device,
            "attention_batch_size": int(args.attention_batch_size),
            "gene_chunk_size": int(args.gene_chunk_size),
            "pilot_cells": int(args.pilot_cells),
            "pilot_genes": int(args.pilot_genes),
        },
        "corrected_protocol_arguments": _training_args(pilot),
        "perturbation_protocol": _parameter_manifest(),
        "code": [dict(item) for item in code_records],
        "inputs": [dict(item) for item in input_records],
        "environment": _environment_manifest(str(args.device)),
        "run_status": "in_progress",
        "eligible_for_scientific_aggregation": False,
    }


def _training_args(pilot: bool) -> dict[str, int | float]:
    values = dict(corrected.CANONICAL_ARGUMENTS)
    if pilot:
        values.update(
            {
                "inner_max_epochs": 1,
                "patience": 1,
                "batch_size": min(int(values["batch_size"]), 32),
                "gene_subsample": 0,
                "top_mut": min(int(values["top_mut"]), 128),
                "n_expr_pc": min(int(values["n_expr_pc"]), 32),
            }
        )
    return values


_MANIFEST_ARGUMENT_FIELDS = frozenset(
    {
        "device",
        "attention_batch_size",
        "gene_chunk_size",
        "pilot_cells",
        "pilot_genes",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "created_utc",
        "seed",
        "fold",
        "pilot",
        "scientific_result",
        "protocol",
        "condition",
        "outer_train_indices_sha256",
        "outer_test_indices_sha256",
        "arguments",
        "corrected_protocol_arguments",
        "perturbation_protocol",
        "code",
        "inputs",
        "environment",
        "run_status",
        "eligible_for_scientific_aggregation",
    }
)
_MANIFEST_ENVIRONMENT_FIELDS = frozenset(
    {
        "python",
        "python_executable",
        "platform",
        "numpy",
        "pandas",
        "scipy",
        "scikit_learn",
        "torch",
        "torch_cuda",
        "torch_cudnn",
        "cublas_workspace_config",
        "cuda_visible_devices",
        "cuda_device_order",
        "cuda_device",
        "nvidia_smi_probe_status",
        "nvidia_smi_probe_error",
        "nvidia_smi_gpus",
        "nvidia_driver_versions",
        "installed_distributions",
        "installed_distributions_sha256",
    }
)
_MANIFEST_MATCH_FIELDS = (
    "schema",
    "seed",
    "fold",
    "pilot",
    "protocol",
    "condition",
    "outer_train_indices_sha256",
    "outer_test_indices_sha256",
    "arguments",
    "corrected_protocol_arguments",
    "perturbation_protocol",
    "environment",
)


def _validate_manifest_records(manifest: Mapping[str, Any], field: str) -> None:
    records = manifest.get(field)
    if not isinstance(records, list):
        raise VerificationError(f"run manifest {field} must be a list")
    for index, record in enumerate(records):
        _validate_file_record(record, label=f"run manifest {field}[{index}]")


def _validate_environment_manifest(environment: Any) -> None:
    if (
        not isinstance(environment, Mapping)
        or set(environment) != _MANIFEST_ENVIRONMENT_FIELDS
    ):
        raise VerificationError("run-manifest environment field set mismatch")
    for field in (
        "python",
        "python_executable",
        "platform",
        "numpy",
        "scipy",
        "scikit_learn",
        "torch",
    ):
        if not isinstance(environment[field], str) or not environment[field]:
            raise VerificationError(f"run-manifest environment {field} is invalid")
    for field in (
        "pandas",
        "torch_cuda",
        "cublas_workspace_config",
        "cuda_visible_devices",
        "cuda_device_order",
    ):
        if environment[field] is not None and not isinstance(environment[field], str):
            raise VerificationError(f"run-manifest environment {field} is invalid")
    cudnn = environment["torch_cudnn"]
    if cudnn is not None and (isinstance(cudnn, bool) or not isinstance(cudnn, int)):
        raise VerificationError("run-manifest environment torch_cudnn is invalid")
    probe_status = environment["nvidia_smi_probe_status"]
    probe_error = environment["nvidia_smi_probe_error"]
    if probe_status not in {"ok", "probe_failed"}:
        raise VerificationError("run-manifest nvidia-smi probe status is invalid")
    if probe_status == "ok" and probe_error is not None:
        raise VerificationError("successful nvidia-smi probe unexpectedly records an error")
    if probe_status == "probe_failed" and (
        not isinstance(probe_error, str) or not probe_error
    ):
        raise VerificationError("failed nvidia-smi probe lacks an error")
    drivers = environment["nvidia_driver_versions"]
    if not isinstance(drivers, list) or any(
        not isinstance(value, str) or not value for value in drivers
    ):
        raise VerificationError("run-manifest NVIDIA driver inventory is invalid")
    if drivers != sorted(set(drivers)):
        raise VerificationError("run-manifest NVIDIA driver inventory is not canonical")
    nvidia_smi_gpus = environment["nvidia_smi_gpus"]
    if not isinstance(nvidia_smi_gpus, list):
        raise VerificationError("run-manifest NVIDIA GPU inventory must be a list")
    previous_gpu_index = -1
    for item in nvidia_smi_gpus:
        if not isinstance(item, Mapping) or set(item) != {
            "nvidia_smi_index",
            "uuid",
            "pci_bus_id",
            "name",
            "driver_version",
            "memory_total_mib",
        }:
            raise VerificationError("run-manifest NVIDIA GPU inventory entry is invalid")
        if (
            isinstance(item["nvidia_smi_index"], bool)
            or not isinstance(item["nvidia_smi_index"], int)
            or item["nvidia_smi_index"] <= previous_gpu_index
            or not isinstance(item["uuid"], str)
            or not item["uuid"]
            or not isinstance(item["pci_bus_id"], str)
            or not item["pci_bus_id"]
            or not isinstance(item["name"], str)
            or not item["name"]
            or not isinstance(item["driver_version"], str)
            or not item["driver_version"]
            or isinstance(item["memory_total_mib"], bool)
            or not isinstance(item["memory_total_mib"], int)
            or item["memory_total_mib"] <= 0
        ):
            raise VerificationError("run-manifest NVIDIA GPU inventory value is invalid")
        previous_gpu_index = int(item["nvidia_smi_index"])
    inventory_drivers = sorted(
        {str(item["driver_version"]) for item in nvidia_smi_gpus}
    )
    if drivers != inventory_drivers:
        raise VerificationError("run-manifest NVIDIA driver/GPU inventories disagree")
    if probe_status == "probe_failed" and (nvidia_smi_gpus or drivers):
        raise VerificationError("failed nvidia-smi probe cannot claim an inventory")
    cuda_device = environment["cuda_device"]
    if cuda_device is not None:
        if not isinstance(cuda_device, Mapping) or set(cuda_device) != {
            "logical_index",
            "name",
            "capability",
            "total_memory_bytes",
            "torch_uuid",
            "torch_pci_bus_id",
            "cuda_visibility_token",
            "physical_binding_status",
            "physical_device",
        }:
            raise VerificationError("run-manifest CUDA device record is invalid")
        if (
            isinstance(cuda_device["logical_index"], bool)
            or not isinstance(cuda_device["logical_index"], int)
            or cuda_device["logical_index"] < 0
            or not isinstance(cuda_device["name"], str)
            or not cuda_device["name"]
            or not isinstance(cuda_device["capability"], list)
            or len(cuda_device["capability"]) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in cuda_device["capability"]
            )
            or isinstance(cuda_device["total_memory_bytes"], bool)
            or not isinstance(cuda_device["total_memory_bytes"], int)
            or cuda_device["total_memory_bytes"] <= 0
        ):
            raise VerificationError("run-manifest CUDA device values are invalid")
        for field in ("torch_uuid", "torch_pci_bus_id", "cuda_visibility_token"):
            value = cuda_device[field]
            if value is not None and (not isinstance(value, str) or not value):
                raise VerificationError(
                    f"run-manifest CUDA device {field} is invalid"
                )
        binding_status = cuda_device["physical_binding_status"]
        allowed_binding_statuses = {
            "unresolved",
            "unresolved_torch_uuid",
            "unresolved_torch_pci_bus_id",
            "unresolved_cuda_visible_devices_uuid",
            "unresolved_cuda_visible_devices_numeric_ordinal",
            "unresolved_cuda_visible_devices_token",
            "unresolved_cuda_visible_devices_logical_index",
            "conflicting_cuda_identifiers",
            "resolved_by_torch_uuid",
            "resolved_by_torch_pci_bus_id",
            "resolved_by_cuda_visible_devices_uuid",
            "resolved_by_consistent_identifiers",
        }
        if binding_status not in allowed_binding_statuses:
            raise VerificationError(
                "run-manifest CUDA device physical binding status is invalid"
            )
        physical_device = cuda_device["physical_device"]
        resolved_binding = binding_status.startswith("resolved_by_")
        if resolved_binding != (physical_device is not None):
            raise VerificationError(
                "run-manifest CUDA physical binding status/device disagree"
            )
        if physical_device is not None:
            if not isinstance(physical_device, Mapping):
                raise VerificationError(
                    "run-manifest selected CUDA physical device is invalid"
                )
            if dict(physical_device) not in [dict(item) for item in nvidia_smi_gpus]:
                raise VerificationError(
                    "run-manifest selected CUDA physical device is absent from "
                    "nvidia-smi inventory"
                )
            physical_uuid = str(physical_device["uuid"]).casefold()
            physical_pci = str(physical_device["pci_bus_id"]).casefold()
            if (
                binding_status == "resolved_by_torch_uuid"
                and (
                    cuda_device["torch_uuid"] is None
                    or str(cuda_device["torch_uuid"]).casefold() != physical_uuid
                )
            ):
                raise VerificationError(
                    "resolved CUDA device does not match the recorded torch UUID"
                )
            if (
                binding_status == "resolved_by_torch_pci_bus_id"
                and (
                    cuda_device["torch_pci_bus_id"] is None
                    or str(cuda_device["torch_pci_bus_id"]).casefold()
                    != physical_pci
                )
            ):
                raise VerificationError(
                    "resolved CUDA device does not match the recorded torch PCI bus ID"
                )
            if binding_status == "resolved_by_cuda_visible_devices_uuid":
                token = cuda_device["cuda_visibility_token"]
                if token is None or not physical_uuid.startswith(str(token).casefold()):
                    raise VerificationError(
                        "resolved CUDA device does not match the CUDA visibility UUID"
                    )
        if probe_status != "ok" or not nvidia_smi_gpus:
            raise VerificationError(
                "formal CUDA run manifest requires a successful, non-empty "
                "nvidia-smi probe"
            )
        recomputed_binding = _resolve_cuda_physical_binding(
            logical_index=int(cuda_device["logical_index"]),
            nvidia_smi_gpus=nvidia_smi_gpus,
            cuda_visible_devices=environment["cuda_visible_devices"],
            torch_uuid=cuda_device["torch_uuid"],
            torch_pci_bus_id=cuda_device["torch_pci_bus_id"],
        )
        recorded_binding = {
            "cuda_visibility_token": cuda_device["cuda_visibility_token"],
            "physical_binding_status": binding_status,
            "physical_device": (
                dict(physical_device) if physical_device is not None else None
            ),
        }
        if recomputed_binding != recorded_binding:
            raise VerificationError(
                "run-manifest CUDA physical binding disagrees with the recorded "
                "UUID, PCI bus ID, visibility mapping, or NVIDIA inventory"
            )
        if not str(binding_status).startswith("resolved_by_"):
            raise VerificationError(
                "formal CUDA run manifest requires a uniquely resolved, consistent "
                "physical GPU binding"
            )
    packages = environment["installed_distributions"]
    if not isinstance(packages, list):
        raise VerificationError("run-manifest package inventory must be a list")
    previous: tuple[str, str] | None = None
    for item in packages:
        if not isinstance(item, Mapping) or set(item) != {"name", "version"}:
            raise VerificationError("run-manifest package inventory entry is invalid")
        name = item["name"]
        version = item["version"]
        if not isinstance(name, str) or not name or not isinstance(version, str):
            raise VerificationError("run-manifest package inventory value is invalid")
        current = (name.casefold(), version)
        if previous is not None and current < previous:
            raise VerificationError("run-manifest package inventory is not canonical")
        previous = current
    digest = environment["installed_distributions_sha256"]
    if digest != _json_sha256(packages):
        raise VerificationError("run-manifest package inventory digest mismatch")


def _validate_manifest_semantics(manifest: Mapping[str, Any]) -> None:
    if set(manifest) != _MANIFEST_FIELDS:
        raise VerificationError("run-manifest field set mismatch")
    if manifest.get("schema") != f"{SCHEMA_VERSION}-run-manifest":
        raise VerificationError("run-manifest schema mismatch")
    if manifest.get("protocol") != "corrected-nested-P1":
        raise VerificationError("run-manifest protocol mismatch")
    if manifest.get("condition") != "DepPLM":
        raise VerificationError("run-manifest condition mismatch")
    pilot = manifest.get("pilot")
    if not isinstance(pilot, bool):
        raise VerificationError("run-manifest pilot flag must be boolean")
    for field in ("seed", "fold"):
        value = manifest.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise VerificationError(f"run-manifest {field} must be an integer")
    if manifest["fold"] < 0:
        raise VerificationError("run-manifest fold must be non-negative")
    arguments = manifest.get("arguments")
    if (
        not isinstance(arguments, Mapping)
        or set(arguments) != _MANIFEST_ARGUMENT_FIELDS
    ):
        raise VerificationError("run-manifest arguments field set mismatch")
    if arguments["device"] not in {"cpu", "cuda"}:
        raise VerificationError("run-manifest device is invalid")
    for field in _MANIFEST_ARGUMENT_FIELDS - {"device"}:
        value = arguments[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise VerificationError(
                f"run-manifest argument {field} must be a positive integer"
            )
    _require_pilot_contract(
        pilot=pilot,
        seed=int(manifest["seed"]),
        fold=int(manifest["fold"]),
        device=str(arguments["device"]),
    )
    if manifest.get("corrected_protocol_arguments") != _training_args(pilot):
        raise VerificationError("run-manifest corrected protocol arguments mismatch")
    if manifest.get("perturbation_protocol") != _parameter_manifest():
        raise VerificationError("run-manifest perturbation protocol mismatch")
    _validate_environment_manifest(manifest.get("environment"))
    environment = manifest["environment"]
    if arguments["device"] == "cuda" and environment["cuda_device"] is None:
        raise VerificationError("CUDA run manifest lacks the selected CUDA device")
    if arguments["device"] == "cpu" and environment["cuda_device"] is not None:
        raise VerificationError("CPU run manifest unexpectedly names a CUDA device")
    if manifest.get("scientific_result") is not False:
        raise VerificationError("run-manifest cannot claim a scientific result")
    if manifest.get("run_status") != "in_progress":
        raise VerificationError("run-manifest status mismatch")
    if manifest.get("eligible_for_scientific_aggregation") is not False:
        raise VerificationError("run-manifest cannot claim aggregation eligibility")
    _validate_manifest_records(manifest, "code")
    _validate_manifest_records(manifest, "inputs")


def _require_pilot_contract(
    *, pilot: bool, seed: int, fold: int, device: str
) -> None:
    """Enforce the fixed pilot identity below the command-line layer."""

    if not pilot:
        return
    if (seed, fold) != (42, 0):
        raise VerificationError("pilot evidence is fixed to seed 42, fold 0")
    if device != "cpu":
        raise VerificationError("pilot evidence is CPU-only")


def _validate_manifest_against_expected(
    manifest: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    _validate_manifest_semantics(manifest)
    _validate_manifest_semantics(expected)
    for field in _MANIFEST_MATCH_FIELDS:
        if manifest.get(field) != expected.get(field):
            raise VerificationError(
                f"run manifest {field} differs from the current request"
            )
    for field in ("code", "inputs"):
        if not _same_records(manifest[field], expected[field]):
            raise VerificationError(
                f"run manifest {field} bindings differ from the current request"
            )


def _grid(pilot: bool) -> list[dict[str, Any]]:
    if pilot:
        return [{"hidden": 32, "lr": 1e-3, "wd": 1e-5, "lamb_sel": 0.1}]
    return [dict(item) for item in corrected.FULL_GRID]


def _save_checkpoint(
    path: Path,
    model: DepPLM,
    *,
    output_root: Path,
) -> dict[str, Any]:
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    return _atomic_torch_save(
        path,
        {"schema": f"{SCHEMA_VERSION}-checkpoint", "state_dict": state},
        relative_to=output_root,
    )


def _load_checkpoint(
    source: Path | ArtifactSnapshot, model: DepPLM, device: str
) -> None:
    path = source.path if isinstance(source, ArtifactSnapshot) else source
    try:
        data = (
            source.data if isinstance(source, ArtifactSnapshot) else path.read_bytes()
        )
        try:
            payload = torch.load(
                io.BytesIO(data), map_location=device, weights_only=True
            )
        except TypeError as exc:
            raise VerificationError(
                "safe checkpoint loading requires torch.load(..., weights_only=True); "
                "the installed PyTorch does not support that interface"
            ) from exc
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != f"{SCHEMA_VERSION}-checkpoint"
        ):
            raise VerificationError("checkpoint schema mismatch")
        state = payload.get("state_dict")
        if not isinstance(state, Mapping):
            raise VerificationError("checkpoint does not contain a state_dict")
        model.load_state_dict(state, strict=True)
    except VerificationError:
        raise
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
        raise VerificationError(f"cannot load checkpoint {path}: {exc}") from exc


_TRAINING_FIELDS = frozenset(
    {
        "schema",
        "seed",
        "fold",
        "fold_seed",
        "inner_train_indices_sha256",
        "inner_validation_indices_sha256",
        "outer_train_indices_sha256",
        "outer_test_indices_sha256",
        "fit_indices_sha256",
        "best_config",
        "best_epoch",
        "selection_score",
        "tuning",
        "refit",
        "train_gene_mean_sha256",
        "inner_preprocessing",
        "preprocessing",
        "checkpoint",
        "outer_test_labels_accessed_during_training_or_prediction",
        "scientific_result",
    }
)
_TUNING_FIELDS = frozenset(
    {
        "config",
        "trained_epochs",
        "scheduler",
        "scheduler_horizon_epochs",
        "history",
        "selection_metric",
        "selection_score",
        "best_epoch",
        "best_validation_metrics",
    }
)
_REFIT_FIELDS = frozenset(
    {"trained_epochs", "scheduler", "scheduler_horizon_epochs", "history"}
)


def _require_int_field(
    payload: Mapping[str, Any],
    field: str,
    *,
    label: str,
    minimum: int | None = None,
) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise VerificationError(f"{label} {field} must be an integer")
    if minimum is not None and value < minimum:
        raise VerificationError(f"{label} {field} must be at least {minimum}")
    return value


def _require_finite_number(
    payload: Mapping[str, Any], field: str, *, label: str
) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VerificationError(f"{label} {field} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise VerificationError(f"{label} {field} must be finite")
    return numeric


def _validate_sha256(value: Any, *, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise VerificationError(f"{label} is not a lowercase SHA256 digest")


def _require_expected_record_path(
    record: Mapping[str, Any],
    *,
    expected: Path,
    relative_to: Path,
    label: str,
) -> None:
    _validate_file_record(record, label=label)
    observed = _resolve_record(record, relative_to=relative_to)
    try:
        expected_resolved = expected.resolve()
    except (OSError, RuntimeError) as exc:
        raise VerificationError(
            f"cannot resolve expected {label} path {expected}: {exc}"
        ) from exc
    if observed != expected_resolved:
        raise VerificationError(
            f"{label} path mismatch: expected {expected_resolved}, got {observed}"
        )


def _snapshot_expected_file_record(
    record: Mapping[str, Any],
    *,
    expected: Path,
    relative_to: Path,
    label: str,
) -> ArtifactSnapshot:
    _require_expected_record_path(
        record,
        expected=expected,
        relative_to=relative_to,
        label=label,
    )
    return _snapshot_file_record(record, relative_to=relative_to, label=label)


def _validate_training_history(
    record: Mapping[str, Any],
    *,
    label: str,
    expected_horizon: int,
    require_validation: bool,
) -> tuple[int, list[Mapping[str, Any]]]:
    trained_epochs = _require_int_field(
        record, "trained_epochs", label=label, minimum=1
    )
    if record.get("scheduler") != "CosineAnnealingLR":
        raise VerificationError(f"{label} scheduler mismatch")
    if (
        _require_int_field(record, "scheduler_horizon_epochs", label=label, minimum=1)
        != expected_horizon
    ):
        raise VerificationError(f"{label} scheduler horizon mismatch")
    history = record.get("history")
    if not isinstance(history, list) or len(history) != trained_epochs:
        raise VerificationError(f"{label} history length mismatch")
    for index, epoch_record in enumerate(history, start=1):
        if not isinstance(epoch_record, Mapping):
            raise VerificationError(f"{label} history[{index - 1}] must be a mapping")
        if _require_int_field(epoch_record, "epoch", label=label, minimum=1) != index:
            raise VerificationError(f"{label} history epoch sequence mismatch")
        _require_finite_number(epoch_record, "train_loss", label=label)
        if require_validation and not isinstance(
            epoch_record.get("validation"), Mapping
        ):
            raise VerificationError(
                f"{label} history[{index - 1}] lacks validation metrics"
            )
    return trained_epochs, history


def _validate_training_record(
    training: Mapping[str, Any],
    *,
    attempt: Path,
    output_root: Path,
    base: corrected.BaseData,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    seed: int,
    fold: int,
    pilot: bool,
) -> corrected.FoldData:
    """Validate every persisted resume-critical training field and recompute fits."""
    if not isinstance(training, Mapping) or set(training) != _TRAINING_FIELDS:
        raise VerificationError("training field set mismatch")
    if training.get("schema") != f"{SCHEMA_VERSION}-training":
        raise VerificationError("training schema mismatch")
    expected_fold_seed = derive_seed(seed, f"fold-{fold:02d}")
    if (
        _require_int_field(training, "seed", label="training") != seed
        or _require_int_field(training, "fold", label="training") != fold
        or _require_int_field(training, "fold_seed", label="training")
        != expected_fold_seed
    ):
        raise VerificationError("training identity disagrees with run manifest")
    if (
        training.get("outer_test_labels_accessed_during_training_or_prediction")
        is not False
    ):
        raise VerificationError("training outer-test label access assertion is invalid")
    if training.get("scientific_result") is not False:
        raise VerificationError("training scientific-result flag must be false")

    hash_fields = (
        "inner_train_indices_sha256",
        "inner_validation_indices_sha256",
        "outer_train_indices_sha256",
        "outer_test_indices_sha256",
        "fit_indices_sha256",
        "train_gene_mean_sha256",
    )
    for field in hash_fields:
        _validate_sha256(training.get(field), label=f"training {field}")
    if training["outer_train_indices_sha256"] != _hash_array(outer_train):
        raise VerificationError("training outer-train split binding mismatch")
    if training["outer_test_indices_sha256"] != _hash_array(outer_test):
        raise VerificationError("training outer-test split binding mismatch")
    if np.intersect1d(outer_train, outer_test).size:
        raise VerificationError("outer train and test splits overlap")

    parameters = _training_args(pilot)
    inner_train, inner_val = make_inner_split(
        outer_train,
        base.lineages,
        val_fraction=float(parameters["inner_val_fraction"]),
        seed=derive_seed(expected_fold_seed, "inner-split"),
    )
    if training["inner_train_indices_sha256"] != _hash_array(inner_train):
        raise VerificationError("inner-train split binding mismatch")
    if training["inner_validation_indices_sha256"] != _hash_array(inner_val):
        raise VerificationError("inner-validation split binding mismatch")
    if np.intersect1d(inner_train, inner_val).size:
        raise VerificationError("inner train and validation splits overlap")
    if np.intersect1d(np.concatenate([inner_train, inner_val]), outer_test).size:
        raise VerificationError("inner selection includes outer-test rows")
    if set(np.concatenate([inner_train, inner_val]).tolist()) != set(
        outer_train.tolist()
    ):
        raise VerificationError("inner split does not partition outer train")

    inner_data = corrected.fit_fold_data(
        base,
        inner_train,
        seed=derive_seed(expected_fold_seed, "inner-preprocess"),
        n_expr_pc=int(parameters["n_expr_pc"]),
        top_mut=int(parameters["top_mut"]),
    )
    if not isinstance(training.get("inner_preprocessing"), Mapping):
        raise VerificationError("training inner_preprocessing must be a mapping")
    if training["inner_preprocessing"] != corrected.preprocessing_manifest(
        inner_data, inner_val
    ):
        raise VerificationError("inner-train preprocessing manifest mismatch")

    expected_configs = [
        {**corrected.CONDITIONS["DepPLM"], **candidate} for candidate in _grid(pilot)
    ]
    tuning = training.get("tuning")
    if not isinstance(tuning, list) or len(tuning) != len(expected_configs):
        raise VerificationError("nested tuning grid length mismatch")
    selection_scores: list[float] = []
    for index, (entry, expected_config) in enumerate(
        zip(tuning, expected_configs, strict=True)
    ):
        label = f"training tuning[{index}]"
        if not isinstance(entry, Mapping) or set(entry) != _TUNING_FIELDS:
            raise VerificationError(f"{label} field set mismatch")
        if entry.get("config") != expected_config:
            raise VerificationError("nested tuning grid or order mismatch")
        trained_epochs, _ = _validate_training_history(
            entry,
            label=label,
            expected_horizon=int(parameters["inner_max_epochs"]),
            require_validation=True,
        )
        if entry.get("selection_metric") != "residual_r2_vs_train_gene_mean":
            raise VerificationError(f"{label} selection metric mismatch")
        score = _require_finite_number(entry, "selection_score", label=label)
        chosen_epoch = _require_int_field(entry, "best_epoch", label=label, minimum=1)
        if chosen_epoch > trained_epochs:
            raise VerificationError(f"{label} best epoch exceeds trained epochs")
        if not isinstance(entry.get("best_validation_metrics"), Mapping):
            raise VerificationError(
                f"{label} best validation metrics must be a mapping"
            )
        selection_scores.append(score)
    selected = tuning[int(np.argmax(np.asarray(selection_scores, dtype=np.float64)))]
    if training.get("best_config") != selected["config"]:
        raise VerificationError("saved best configuration is not the nested selection")
    best_epoch = _require_int_field(training, "best_epoch", label="training", minimum=1)
    if best_epoch != selected["best_epoch"]:
        raise VerificationError("saved refit epoch is not the nested selection")
    if (
        _require_finite_number(training, "selection_score", label="training")
        != selection_scores[int(np.argmax(selection_scores))]
    ):
        raise VerificationError("saved nested selection score mismatch")

    refit = training.get("refit")
    if not isinstance(refit, Mapping) or set(refit) != _REFIT_FIELDS:
        raise VerificationError("training refit field set mismatch")
    refit_epochs, _ = _validate_training_history(
        refit,
        label="training refit",
        expected_horizon=int(parameters["inner_max_epochs"]),
        require_validation=False,
    )
    if refit_epochs != best_epoch:
        raise VerificationError("training refit epoch count mismatch")

    refit_seed = derive_seed(expected_fold_seed, "refit")
    fold_data = corrected.fit_fold_data(
        base,
        outer_train,
        seed=derive_seed(refit_seed, "outer-preprocess"),
        n_expr_pc=int(parameters["n_expr_pc"]),
        top_mut=int(parameters["top_mut"]),
    )
    if training["fit_indices_sha256"] != _hash_array(fold_data.fit_indices):
        raise VerificationError("final preprocessing was not fit on outer train")
    if not isinstance(training.get("preprocessing"), Mapping):
        raise VerificationError("training preprocessing must be a mapping")
    if training["preprocessing"] != corrected.preprocessing_manifest(
        fold_data, outer_test
    ):
        raise VerificationError("outer-train preprocessing manifest mismatch")
    train_gene_mean = fit_gene_mean_baseline(base.gene_effect, outer_train)
    if training["train_gene_mean_sha256"] != _hash_array(train_gene_mean):
        raise VerificationError("training gene-mean reference hash mismatch")
    _require_expected_record_path(
        training["checkpoint"],
        expected=attempt / "trained-model.pt",
        relative_to=output_root,
        label="training checkpoint",
    )
    return fold_data


def _train_fold(
    *,
    base: corrected.BaseData,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    seed: int,
    fold: int,
    pilot: bool,
    device: str,
    attempt: Path,
    output_root: Path,
) -> tuple[DepPLM, corrected.FoldData, dict[str, Any]]:
    parameters = _training_args(pilot)
    fold_name = f"fold-{fold:02d}"
    fold_seed = derive_seed(seed, fold_name)
    inner_train, inner_val = make_inner_split(
        outer_train,
        base.lineages,
        val_fraction=float(parameters["inner_val_fraction"]),
        seed=derive_seed(fold_seed, "inner-split"),
    )
    inner_data = corrected.fit_fold_data(
        base,
        inner_train,
        seed=derive_seed(fold_seed, "inner-preprocess"),
        n_expr_pc=int(parameters["n_expr_pc"]),
        top_mut=int(parameters["top_mut"]),
    )
    tuning: list[dict[str, Any]] = []
    tune_seed = derive_seed(fold_seed, "tune")
    best_score = -np.inf
    best_config: dict[str, Any] | None = None
    best_epoch = 0
    for candidate in _grid(pilot):
        config = {**corrected.CONDITIONS["DepPLM"], **candidate}
        set_global_seed(tune_seed)
        model = corrected.build_model(config, base, inner_data)
        _, record = corrected._train_epochs(
            model,
            config,
            base,
            inner_data,
            inner_train,
            epochs=int(parameters["inner_max_epochs"]),
            batch_size=int(parameters["batch_size"]),
            gene_subsample=int(parameters["gene_subsample"]),
            device=device,
            seed=tune_seed,
            validation_idx=inner_val,
            patience=int(parameters["patience"]),
            scheduler_horizon=int(parameters["inner_max_epochs"]),
        )
        score = float(record["selection_score"])
        epoch = int(record["best_epoch"])
        tuning.append({"config": config, **record})
        if score > best_score:
            best_score = score
            best_config = dict(config)
            best_epoch = epoch
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    if best_config is None or best_epoch < 1:
        raise RuntimeError("inner selection did not choose a valid configuration")

    refit_seed = derive_seed(fold_seed, "refit")
    final_data = corrected.fit_fold_data(
        base,
        outer_train,
        seed=derive_seed(refit_seed, "outer-preprocess"),
        n_expr_pc=int(parameters["n_expr_pc"]),
        top_mut=int(parameters["top_mut"]),
    )
    set_global_seed(refit_seed)
    final_model = corrected.build_model(best_config, base, final_data)
    final_model, refit_record = corrected._train_epochs(
        final_model,
        best_config,
        base,
        final_data,
        outer_train,
        epochs=best_epoch,
        batch_size=int(parameters["batch_size"]),
        gene_subsample=int(parameters["gene_subsample"]),
        device=device,
        seed=refit_seed,
        validation_idx=None,
        patience=int(parameters["patience"]),
        scheduler_horizon=int(parameters["inner_max_epochs"]),
    )
    if not isinstance(final_model, DepPLM):
        raise TypeError("DepPLM training produced an unexpected model type")
    checkpoint = _save_checkpoint(
        attempt / "trained-model.pt", final_model, output_root=output_root
    )
    train_gene_mean = fit_gene_mean_baseline(base.gene_effect, outer_train)
    training_record = {
        "schema": f"{SCHEMA_VERSION}-training",
        "seed": int(seed),
        "fold": int(fold),
        "fold_seed": int(fold_seed),
        "inner_train_indices_sha256": _hash_array(inner_train),
        "inner_validation_indices_sha256": _hash_array(inner_val),
        "outer_train_indices_sha256": _hash_array(outer_train),
        "outer_test_indices_sha256": _hash_array(outer_test),
        "fit_indices_sha256": _hash_array(final_data.fit_indices),
        "best_config": best_config,
        "best_epoch": int(best_epoch),
        "selection_score": float(best_score),
        "tuning": tuning,
        "refit": refit_record,
        "train_gene_mean_sha256": _hash_array(train_gene_mean),
        "inner_preprocessing": corrected.preprocessing_manifest(inner_data, inner_val),
        "preprocessing": corrected.preprocessing_manifest(final_data, outer_test),
        "checkpoint": checkpoint,
        "outer_test_labels_accessed_during_training_or_prediction": False,
        "scientific_result": False,
    }
    _write_json_once(attempt / "training.json", training_record)
    return final_model, final_data, training_record


def _valid_training_attempt(
    attempt: Path,
    *,
    manifest_expected: Mapping[str, Any],
    output_root: Path,
    base: corrected.BaseData,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
) -> ValidatedTrainingAttempt | None:
    manifest_path = attempt / "run-manifest.json"
    training_path = attempt / "training.json"
    if not manifest_path.is_file() or not training_path.is_file():
        return None
    inference_artifacts = (
        "outer-test-predictions.npz",
        "attention-evidence.npz",
        "inference.json",
        "metrics.json",
        "completion.json",
    )
    if any((attempt / name).exists() for name in inference_artifacts):
        return None
    try:
        manifest = _load_json(manifest_path)
        training = _load_json(training_path)
        _validate_manifest_against_expected(manifest, manifest_expected)
        fold_data = _validate_training_record(
            training,
            attempt=attempt,
            output_root=output_root,
            base=base,
            outer_train=outer_train,
            outer_test=outer_test,
            seed=_require_int_field(manifest, "seed", label="run manifest"),
            fold=_require_int_field(manifest, "fold", label="run manifest"),
            pilot=manifest["pilot"],
        )
        checkpoint = _snapshot_file_record(
            training["checkpoint"],
            relative_to=output_root,
            label="training checkpoint",
        )
        return ValidatedTrainingAttempt(
            training=training,
            fold_data=fold_data,
            checkpoint=checkpoint,
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
        IndexError,
        RuntimeError,
        VerificationError,
    ):
        return None


def _load_or_train(
    *,
    base: corrected.BaseData,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    seed: int,
    fold: int,
    pilot: bool,
    device: str,
    root: Path,
    output_root: Path,
    manifest_expected: Mapping[str, Any],
) -> tuple[Path, DepPLM, corrected.FoldData, dict[str, Any]]:
    _prepare_replay_runtime(device)
    for attempt in sorted(root.glob("attempt-*"), reverse=True):
        validated = _valid_training_attempt(
            attempt,
            manifest_expected=manifest_expected,
            output_root=output_root,
            base=base,
            outer_train=outer_train,
            outer_test=outer_test,
        )
        if validated is None or (attempt / "completion.json").is_file():
            continue
        try:
            model = corrected.build_model(
                validated.training["best_config"], base, validated.fold_data
            )
            _load_checkpoint(validated.checkpoint, model, device)
        except (TypeError, ValueError, RuntimeError, VerificationError):
            continue
        return (
            attempt,
            model.to(device),
            validated.fold_data,
            validated.training,
        )

    attempt = _new_attempt_path(root)
    _write_json_once(attempt / "run-manifest.json", manifest_expected)
    model, fold_data, training = _train_fold(
        base=base,
        outer_train=outer_train,
        outer_test=outer_test,
        seed=seed,
        fold=fold,
        pilot=pilot,
        device=device,
        attempt=attempt,
        output_root=output_root,
    )
    return attempt, model, fold_data, training


def _predict_fold(
    *,
    attempt: Path,
    output_root: Path,
    model: DepPLM,
    fold_data: corrected.FoldData,
    base: corrected.BaseData,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    seed: int,
    fold: int,
    device: str,
    attention_batch_size: int,
    gene_chunk_size: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    fold_seed = derive_seed(seed, f"fold-{fold:02d}")
    panel = _panel_indices(base.genes)
    baseline = infer_with_attention(
        model,
        fold_data,
        outer_test,
        device=device,
        batch_size=attention_batch_size,
        gene_chunk_size=gene_chunk_size,
        panel_indices=panel,
        collect_attention=True,
    )
    if baseline.mean_attention is None or baseline.panel_attention is None:
        raise RuntimeError("baseline inference did not return attention evidence")
    tokens = np.asarray(
        fold_data.mutation.tokens[outer_test, :K_TOKENS], dtype=np.int64
    )
    validity = np.asarray(
        fold_data.mutation.token_mask[outer_test, :K_TOKENS], dtype=bool
    )
    prevalence, degree = _training_token_covariates(
        fold_data, outer_train, base.edge_index, base.genes
    )

    predictions: dict[str, np.ndarray] = {"baseline": baseline.prediction}
    evidence_arrays: dict[str, np.ndarray] = {
        "outer_train_indices": np.asarray(outer_train, dtype=np.int64),
        "outer_test_indices": np.asarray(outer_test, dtype=np.int64),
        "source_cell_indices": np.asarray(
            base.source_cell_indices[outer_test], dtype=np.int64
        ),
        "source_gene_indices": np.asarray(base.source_gene_indices, dtype=np.int64),
        "cell_ids": np.asarray([base.cell_ids[int(i)] for i in outer_test]),
        "gene_ids": np.asarray(base.genes),
        "token_indices": tokens,
        "token_validity": validity,
        "mean_attention": baseline.mean_attention,
        "panel_gene_indices": panel,
        "panel_attention": baseline.panel_attention,
        "train_prevalence_count_by_gene": prevalence,
        "string_degree_by_gene": degree,
    }
    mask_metadata: dict[str, Any] = {}
    for fraction in MASK_FRACTIONS:
        key = _fraction_key(fraction)
        random_seeds = tuple(
            derive_seed(fold_seed, f"matched-{key}-draw-{draw:02d}")
            for draw in range(MATCHED_RANDOM_DRAWS)
        )
        plan = build_mask_plan(
            baseline.mean_attention,
            tokens,
            validity,
            prevalence,
            degree,
            fraction=fraction,
            random_seeds=random_seeds,
        )
        evidence_arrays[f"top_mask_{key}"] = plan.top_mask
        top_validity = validity & ~plan.top_mask
        predictions[f"top_mask_{key}"] = infer_with_attention(
            model,
            fold_data,
            outer_test,
            device=device,
            batch_size=attention_batch_size,
            gene_chunk_size=gene_chunk_size,
            token_validity_override=top_validity,
        ).prediction
        for draw, (matched, features) in enumerate(
            zip(plan.matched_masks, plan.matching_features)
        ):
            evidence_arrays[f"matched_mask_{key}_d{draw:02d}"] = matched
            evidence_arrays[f"matching_features_{key}_d{draw:02d}"] = features
            predictions[f"matched_mask_{key}_d{draw:02d}"] = infer_with_attention(
                model,
                fold_data,
                outer_test,
                device=device,
                batch_size=attention_batch_size,
                gene_chunk_size=gene_chunk_size,
                token_validity_override=validity & ~matched,
            ).prediction
        mask_metadata[key] = {
            "fraction": float(fraction),
            "random_seeds": list(plan.random_seeds),
            "top_mask_sha256": _hash_array(plan.top_mask),
            "matching_feature_columns": list(MATCHING_FEATURE_COLUMNS),
        }

    permutation_seed = derive_seed(fold_seed, "attention-permutation")
    permutation = build_attention_permutation(validity, seed=permutation_seed)
    evidence_arrays["attention_permutation"] = permutation
    predictions["attention_permuted"] = infer_with_attention(
        model,
        fold_data,
        outer_test,
        device=device,
        batch_size=attention_batch_size,
        gene_chunk_size=gene_chunk_size,
        attention_permutation=permutation,
    ).prediction

    randomization_seed = derive_seed(fold_seed, "parameter-randomization")
    randomized, randomization_record = parameter_randomized_inference(
        model,
        fold_data,
        outer_test,
        seed=randomization_seed,
        device=device,
        batch_size=attention_batch_size,
        gene_chunk_size=gene_chunk_size,
        panel_indices=panel,
    )
    if randomized.mean_attention is None or randomized.panel_attention is None:
        raise RuntimeError("randomized inference did not return attention evidence")
    predictions["parameter_randomized"] = randomized.prediction
    evidence_arrays["randomized_mean_attention"] = randomized.mean_attention
    evidence_arrays["randomized_panel_attention"] = randomized.panel_attention
    evidence_arrays["panel_top_partner_indices"] = _top_partner_indices(
        baseline.panel_attention,
        tokens,
        validity,
        n_genes=len(base.genes),
    )
    evidence_arrays["randomized_panel_top_partner_indices"] = _top_partner_indices(
        randomized.panel_attention,
        tokens,
        validity,
        n_genes=len(base.genes),
    )

    train_gene_mean = fit_gene_mean_baseline(base.gene_effect, outer_train)
    prediction_arrays: dict[str, np.ndarray] = {
        **predictions,
        "train_gene_mean": train_gene_mean,
        "outer_test_indices": np.asarray(outer_test, dtype=np.int64),
        "source_cell_indices": np.asarray(
            base.source_cell_indices[outer_test], dtype=np.int64
        ),
        "source_gene_indices": np.asarray(base.source_gene_indices, dtype=np.int64),
        "cell_ids": np.asarray([base.cell_ids[int(i)] for i in outer_test]),
        "gene_ids": np.asarray(base.genes),
    }
    prediction_record = _atomic_npz(
        attempt / "outer-test-predictions.npz",
        prediction_arrays,
        relative_to=output_root,
    )
    evidence_record = _atomic_npz(
        attempt / "attention-evidence.npz",
        evidence_arrays,
        relative_to=output_root,
    )
    inference_record = {
        "schema": f"{SCHEMA_VERSION}-inference",
        "prediction_generation_calls": 1,
        "outer_test_prediction_bundle_generation_calls": 1,
        "one_prediction_row_per_outer_test_cell_per_mode": True,
        "outer_test_rows": int(outer_test.size),
        "outer_test_unique_rows": int(np.unique(outer_test).size),
        "outer_test_labels_accessed_during_prediction_generation": False,
        "prediction_modes": sorted(predictions),
        "mask_plans": mask_metadata,
        "attention_permutation": {
            "seed": int(permutation_seed),
            "sha256": _hash_array(permutation),
        },
        "parameter_randomization": randomization_record,
        "predictions": prediction_record,
        "attention_evidence": evidence_record,
        "scope_statement": PROTOCOL_SCOPE,
    }
    _write_json_once(attempt / "inference.json", inference_record)
    target = np.asarray(base.gene_effect[:, outer_test].T, dtype=np.float32)
    metrics = _mode_metrics(predictions, target, train_gene_mean)
    metrics_record = {
        "schema": f"{SCHEMA_VERSION}-metrics",
        "computed_only_after_predictions_and_evidence_were_atomically_persisted": True,
        "outer_test_target_sha256": _hash_array(target),
        "modes": metrics,
        "scope_statement": PROTOCOL_SCOPE,
        "scientific_result": False,
    }
    _write_json_once(attempt / "metrics.json", metrics_record)
    return (
        inference_record,
        metrics_record,
        {
            "predictions": prediction_record,
            "attention_evidence": evidence_record,
            "inference": _file_record(
                attempt / "inference.json", relative_to=output_root
            ),
            "metrics": _file_record(attempt / "metrics.json", relative_to=output_root),
        },
    )


def _completed_attempts(root: Path) -> list[Path]:
    return sorted(
        path.parent
        for path in root.glob("**/completion.json")
        if path.parent.name.startswith("attempt-")
    )


def run_fold(
    args: argparse.Namespace,
    *,
    seed: int,
    fold: int,
    base: corrected.BaseData,
    code_records: Sequence[Mapping[str, Any]],
    input_records: Sequence[Mapping[str, Any]],
) -> Path:
    pilot = bool(args.pilot)
    splits = _fold_splits(base, seed=seed, pilot=pilot)
    if fold < 0 or fold >= len(splits):
        raise IndexError(f"fold {fold} is outside 0..{len(splits) - 1}")
    outer_train, outer_test = splits[fold]
    root = _attempt_root(args.output_root, pilot=pilot, seed=seed, fold=fold)
    manifest = _run_manifest(
        seed=seed,
        fold=fold,
        pilot=pilot,
        outer_train=outer_train,
        outer_test=outer_test,
        code_records=code_records,
        input_records=input_records,
        args=args,
    )
    for completed in sorted(root.glob("attempt-*/completion.json")):
        try:
            completed_manifest = _load_json(completed.parent / "run-manifest.json")
            _validate_manifest_against_expected(completed_manifest, manifest)
            verify_attempt(
                completed.parent,
                output_root=args.output_root,
                base=base,
                replay_device=args.device,
            )
            print(
                f"[attention] verified existing fold, skipping: {completed.parent}",
                flush=True,
            )
            return completed.parent
        except VerificationError:
            continue

    attempt, model, fold_data, training = _load_or_train(
        base=base,
        outer_train=outer_train,
        outer_test=outer_test,
        seed=seed,
        fold=fold,
        pilot=pilot,
        device=args.device,
        root=root,
        output_root=args.output_root,
        manifest_expected=manifest,
    )
    started = time.time()
    inference, metrics, artifacts = _predict_fold(
        attempt=attempt,
        output_root=args.output_root,
        model=model,
        fold_data=fold_data,
        base=base,
        outer_train=outer_train,
        outer_test=outer_test,
        seed=seed,
        fold=fold,
        device=args.device,
        attention_batch_size=args.attention_batch_size,
        gene_chunk_size=args.gene_chunk_size,
    )
    completion = {
        "schema": f"{SCHEMA_VERSION}-completion",
        "completed_utc": _utc_now(),
        "seed": int(seed),
        "fold": int(fold),
        "pilot": pilot,
        "run_status": "pilot_complete" if pilot else "fold_complete",
        "protocol_complete": False,
        "eligible_for_scientific_aggregation": not pilot,
        "scientific_result": False,
        "scientific_result_reason": (
            "A single outer fold cannot establish the preregistered 3-seed x 5-fold result."
            if not pilot
            else "Pilot mode is a structural CPU smoke test only."
        ),
        "training": _file_record(
            attempt / "training.json", relative_to=args.output_root
        ),
        "run_manifest": _file_record(
            attempt / "run-manifest.json", relative_to=args.output_root
        ),
        "artifacts": artifacts,
        "prediction_generation_calls": inference["prediction_generation_calls"],
        "outer_test_prediction_bundle_generation_calls": inference[
            "outer_test_prediction_bundle_generation_calls"
        ],
        "one_prediction_row_per_outer_test_cell_per_mode": inference[
            "one_prediction_row_per_outer_test_cell_per_mode"
        ],
        "wall_seconds_after_training": float(time.time() - started),
        "scope_statement": PROTOCOL_SCOPE,
    }
    _write_json_once(attempt / "completion.json", completion)
    verify_attempt(
        attempt,
        output_root=args.output_root,
        base=base,
        replay_device=args.device,
    )
    return attempt


def _verify_records_current(manifest: Mapping[str, Any]) -> None:
    current_code = _code_records()
    current_inputs = _input_records()
    if not _same_records(manifest.get("code", []), current_code):
        raise VerificationError("code bindings differ from the current source files")
    if not _same_records(manifest.get("inputs", []), current_inputs):
        raise VerificationError("input bindings differ from the current data files")


def _load_npz_bytes(data: bytes, *, label: str) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    except (OSError, ValueError) as exc:
        raise VerificationError(f"cannot read NPZ artifact {label}: {exc}") from exc


def _load_npz_snapshot(snapshot: ArtifactSnapshot) -> dict[str, np.ndarray]:
    return _load_npz_bytes(snapshot.data, label=str(snapshot.path))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise VerificationError(f"cannot read NPZ artifact {path}: {exc}") from exc
    return _load_npz_bytes(data, label=str(path))


_COMPLETION_ARTIFACT_KEYS = frozenset(
    {"predictions", "attention_evidence", "inference", "metrics"}
)
_COMPLETION_FIELDS = frozenset(
    {
        "schema",
        "completed_utc",
        "seed",
        "fold",
        "pilot",
        "run_status",
        "protocol_complete",
        "eligible_for_scientific_aggregation",
        "scientific_result",
        "scientific_result_reason",
        "training",
        "run_manifest",
        "artifacts",
        "prediction_generation_calls",
        "outer_test_prediction_bundle_generation_calls",
        "one_prediction_row_per_outer_test_cell_per_mode",
        "wall_seconds_after_training",
        "scope_statement",
    }
)
_INFERENCE_FIELDS = frozenset(
    {
        "schema",
        "prediction_generation_calls",
        "outer_test_prediction_bundle_generation_calls",
        "one_prediction_row_per_outer_test_cell_per_mode",
        "outer_test_rows",
        "outer_test_unique_rows",
        "outer_test_labels_accessed_during_prediction_generation",
        "prediction_modes",
        "mask_plans",
        "attention_permutation",
        "parameter_randomization",
        "predictions",
        "attention_evidence",
        "scope_statement",
    }
)
_METRICS_FIELDS = frozenset(
    {
        "schema",
        "computed_only_after_predictions_and_evidence_were_atomically_persisted",
        "outer_test_target_sha256",
        "modes",
        "scope_statement",
        "scientific_result",
    }
)
_MASK_PLAN_FIELDS = frozenset(
    {"fraction", "random_seeds", "top_mask_sha256", "matching_feature_columns"}
)
_ATTENTION_PERMUTATION_FIELDS = frozenset({"seed", "sha256"})
_PARAMETER_RANDOMIZATION_FIELDS = frozenset(
    {
        "scope",
        "initialization",
        "seed",
        "trained_sha256",
        "randomized_sha256",
        "restored_sha256",
    }
)


def _expected_prediction_array_keys() -> frozenset[str]:
    return frozenset(
        {
            *_required_prediction_modes(),
            "train_gene_mean",
            "outer_test_indices",
            "source_cell_indices",
            "source_gene_indices",
            "cell_ids",
            "gene_ids",
        }
    )


def _expected_evidence_array_keys() -> frozenset[str]:
    keys = {
        "outer_train_indices",
        "outer_test_indices",
        "source_cell_indices",
        "source_gene_indices",
        "cell_ids",
        "gene_ids",
        "token_indices",
        "token_validity",
        "mean_attention",
        "panel_gene_indices",
        "panel_attention",
        "train_prevalence_count_by_gene",
        "string_degree_by_gene",
        "attention_permutation",
        "randomized_mean_attention",
        "randomized_panel_attention",
        "panel_top_partner_indices",
        "randomized_panel_top_partner_indices",
    }
    for fraction in MASK_FRACTIONS:
        key = _fraction_key(fraction)
        keys.add(f"top_mask_{key}")
        for draw in range(MATCHED_RANDOM_DRAWS):
            keys.add(f"matched_mask_{key}_d{draw:02d}")
            keys.add(f"matching_features_{key}_d{draw:02d}")
    return frozenset(keys)


def _require_exact_fields(
    payload: Any, expected: frozenset[str], *, label: str
) -> None:
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise VerificationError(f"{label} field set mismatch")


def _require_exact_array_keys(
    arrays: Mapping[str, np.ndarray], expected: frozenset[str], *, label: str
) -> None:
    if set(arrays) != expected:
        raise VerificationError(f"{label} array key set mismatch")


def _validate_completion_semantics(
    completion: Mapping[str, Any], *, pilot: bool
) -> None:
    _require_exact_fields(completion, _COMPLETION_FIELDS, label="completion")
    expected_status = "pilot_complete" if pilot else "fold_complete"
    if completion.get("run_status") != expected_status:
        raise VerificationError("completion status mismatch")
    if completion.get("protocol_complete") is not False:
        raise VerificationError("single-fold completion cannot claim protocol completion")
    expected_eligibility = not pilot
    if (
        completion.get("eligible_for_scientific_aggregation")
        is not expected_eligibility
    ):
        raise VerificationError("completion aggregation eligibility mismatch")
    if completion.get("scientific_result") is not False:
        raise VerificationError("single-fold completion cannot claim scientific result")
    if completion.get("scope_statement") != PROTOCOL_SCOPE:
        raise VerificationError("completion scope statement mismatch")
    wall_seconds = completion.get("wall_seconds_after_training")
    if (
        isinstance(wall_seconds, bool)
        or not isinstance(wall_seconds, (int, float))
        or not math.isfinite(float(wall_seconds))
        or float(wall_seconds) < 0.0
    ):
        raise VerificationError("completion wall time must be finite and non-negative")


def _validate_inference_semantics(inference: Mapping[str, Any]) -> None:
    _require_exact_fields(inference, _INFERENCE_FIELDS, label="inference")
    if inference.get("outer_test_labels_accessed_during_prediction_generation") is not False:
        raise VerificationError("outer-test labels were accessed during prediction generation")
    if inference.get("scope_statement") != PROTOCOL_SCOPE:
        raise VerificationError("inference scope statement mismatch")
    expected_modes = sorted(_required_prediction_modes())
    if inference.get("prediction_modes") != expected_modes:
        raise VerificationError("prediction mode list mismatch")
    mask_plans = inference.get("mask_plans")
    expected_plan_keys = {_fraction_key(fraction) for fraction in MASK_FRACTIONS}
    if not isinstance(mask_plans, Mapping) or set(mask_plans) != expected_plan_keys:
        raise VerificationError("mask-plan key set mismatch")
    for key, metadata in mask_plans.items():
        _require_exact_fields(
            metadata, _MASK_PLAN_FIELDS, label=f"mask-plan {key}"
        )
        if metadata.get("matching_feature_columns") != list(
            MATCHING_FEATURE_COLUMNS
        ):
            raise VerificationError(f"matching feature columns mismatch for {key}")
    _require_exact_fields(
        inference.get("attention_permutation"),
        _ATTENTION_PERMUTATION_FIELDS,
        label="attention permutation",
    )
    _require_exact_fields(
        inference.get("parameter_randomization"),
        _PARAMETER_RANDOMIZATION_FIELDS,
        label="parameter randomization",
    )


def _validate_metrics_semantics(metrics: Mapping[str, Any]) -> None:
    _require_exact_fields(metrics, _METRICS_FIELDS, label="metrics")
    if metrics.get("scientific_result") is not False:
        raise VerificationError("single-fold metrics cannot claim scientific result")
    if metrics.get("scope_statement") != PROTOCOL_SCOPE:
        raise VerificationError("metrics scope statement mismatch")


def verify_attempt(
    attempt: Path,
    *,
    output_root: Path = OUTPUT_ROOT,
    base: corrected.BaseData | None = None,
    replay_device: str = "cpu",
) -> dict[str, Any]:
    """Verify an attempt, normalizing malformed bundle failures."""
    try:
        return _verify_attempt(
            attempt,
            output_root=output_root,
            base=base,
            replay_device=replay_device,
        )
    except VerificationError:
        raise
    except (
        OSError,
        RuntimeError,
        ImportError,
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
        IndexError,
    ) as exc:
        raise VerificationError(
            f"malformed attempt bundle {attempt}: {type(exc).__name__}: {exc}"
        ) from exc


def _verify_attempt(
    attempt: Path,
    *,
    output_root: Path,
    base: corrected.BaseData | None,
    replay_device: str,
) -> dict[str, Any]:
    """Verify bindings and numerically replay all modes from the checkpoint."""
    _prepare_replay_runtime(replay_device)
    attempt = _resolve_record({"path": str(attempt)}, relative_to=output_root)
    completion_snapshot = _read_artifact_snapshot(
        attempt / "completion.json", relative_to=output_root
    )
    completion = _load_json_snapshot(completion_snapshot)
    completion_record = _snapshot_record(
        completion_snapshot, relative_to=output_root
    )
    if completion.get("schema") != f"{SCHEMA_VERSION}-completion":
        raise VerificationError("completion schema mismatch")
    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise VerificationError("completion artifacts must be a mapping")
    if set(artifacts) != _COMPLETION_ARTIFACT_KEYS:
        raise VerificationError(
            "completion artifact key set mismatch: expected "
            "attention_evidence, inference, metrics, predictions"
        )
    expected_records = {
        "completion training": (completion.get("training"), attempt / "training.json"),
        "completion run_manifest": (
            completion.get("run_manifest"),
            attempt / "run-manifest.json",
        ),
        "completion artifacts.predictions": (
            artifacts["predictions"],
            attempt / "outer-test-predictions.npz",
        ),
        "completion artifacts.attention_evidence": (
            artifacts["attention_evidence"],
            attempt / "attention-evidence.npz",
        ),
        "completion artifacts.inference": (
            artifacts["inference"],
            attempt / "inference.json",
        ),
        "completion artifacts.metrics": (
            artifacts["metrics"],
            attempt / "metrics.json",
        ),
    }
    for label, (record, expected_path) in expected_records.items():
        _require_expected_record_path(
            record,
            expected=expected_path,
            relative_to=output_root,
            label=label,
        )

    manifest_snapshot = _snapshot_expected_file_record(
        completion["run_manifest"],
        expected=attempt / "run-manifest.json",
        relative_to=output_root,
        label="completion run_manifest",
    )
    manifest = _load_json_snapshot(manifest_snapshot)
    _validate_manifest_semantics(manifest)
    _verify_records_current(manifest)

    seed = _require_int_field(manifest, "seed", label="run manifest")
    fold = _require_int_field(manifest, "fold", label="run manifest")
    pilot = manifest["pilot"]
    if pilot and replay_device != "cpu":
        raise VerificationError("pilot verification replay is CPU-only")
    completion_seed = _require_int_field(completion, "seed", label="completion")
    completion_fold = _require_int_field(completion, "fold", label="completion")
    completion_pilot = completion.get("pilot")
    if not isinstance(completion_pilot, bool):
        raise VerificationError("completion pilot must be boolean")
    if (
        completion_seed != seed
        or completion_fold != fold
        or completion_pilot is not pilot
    ):
        raise VerificationError("completion identity disagrees with run manifest")
    _validate_completion_semantics(completion, pilot=pilot)
    if (
        _require_int_field(
            completion, "prediction_generation_calls", label="completion", minimum=1
        )
        != 1
    ):
        raise VerificationError("completion does not record exact-once prediction")
    if (
        _require_int_field(
            completion,
            "outer_test_prediction_bundle_generation_calls",
            label="completion",
            minimum=1,
        )
        != 1
    ):
        raise VerificationError(
            "completion does not record exact-once prediction bundle"
        )
    if completion.get("one_prediction_row_per_outer_test_cell_per_mode") is not True:
        raise VerificationError("completion does not record exact-once prediction rows")
    if base is None:
        base = corrected.load_base_data(
            pilot=pilot,
            seed=seed,
            pilot_cells=int(manifest["arguments"]["pilot_cells"]),
            pilot_genes=int(manifest["arguments"]["pilot_genes"]),
        )
    splits = _fold_splits(base, seed=seed, pilot=pilot)
    if fold < 0 or fold >= len(splits):
        raise VerificationError(f"manifest fold {fold} is outside 0..{len(splits) - 1}")
    outer_train, outer_test = splits[fold]
    if manifest["outer_train_indices_sha256"] != _hash_array(outer_train):
        raise VerificationError("outer-train split binding mismatch")
    if manifest["outer_test_indices_sha256"] != _hash_array(outer_test):
        raise VerificationError("outer-test split binding mismatch")

    training_snapshot = _snapshot_expected_file_record(
        completion["training"],
        expected=attempt / "training.json",
        relative_to=output_root,
        label="completion training",
    )
    inference_snapshot = _snapshot_expected_file_record(
        artifacts["inference"],
        expected=attempt / "inference.json",
        relative_to=output_root,
        label="completion artifacts.inference",
    )
    metrics_snapshot = _snapshot_expected_file_record(
        artifacts["metrics"],
        expected=attempt / "metrics.json",
        relative_to=output_root,
        label="completion artifacts.metrics",
    )
    prediction_snapshot = _snapshot_expected_file_record(
        artifacts["predictions"],
        expected=attempt / "outer-test-predictions.npz",
        relative_to=output_root,
        label="completion artifacts.predictions",
    )
    evidence_snapshot = _snapshot_expected_file_record(
        artifacts["attention_evidence"],
        expected=attempt / "attention-evidence.npz",
        relative_to=output_root,
        label="completion artifacts.attention_evidence",
    )
    training = _load_json_snapshot(training_snapshot)
    inference = _load_json_snapshot(inference_snapshot)
    saved_metrics = _load_json_snapshot(metrics_snapshot)
    if inference.get("schema") != f"{SCHEMA_VERSION}-inference":
        raise VerificationError("inference schema mismatch")
    if saved_metrics.get("schema") != f"{SCHEMA_VERSION}-metrics":
        raise VerificationError("metrics schema mismatch")
    _validate_inference_semantics(inference)
    _validate_metrics_semantics(saved_metrics)
    if (
        _require_int_field(
            inference, "prediction_generation_calls", label="inference", minimum=1
        )
        != 1
    ):
        raise VerificationError("outer-test prediction generation was not exact-once")
    if (
        _require_int_field(
            inference,
            "outer_test_prediction_bundle_generation_calls",
            label="inference",
            minimum=1,
        )
        != 1
    ):
        raise VerificationError(
            "outer-test prediction bundle was not generated exact-once"
        )
    if inference.get("one_prediction_row_per_outer_test_cell_per_mode") is not True:
        raise VerificationError("outer-test row cardinality assertion is missing")
    if (
        completion["prediction_generation_calls"]
        != inference["prediction_generation_calls"]
        or completion["outer_test_prediction_bundle_generation_calls"]
        != inference["outer_test_prediction_bundle_generation_calls"]
    ):
        raise VerificationError("completion inference call counts are not cross-bound")
    for key in ("predictions", "attention_evidence"):
        _validate_file_record(inference.get(key), label=f"inference {key}")
        if dict(inference[key]) != dict(artifacts[key]):
            raise VerificationError(
                f"completion artifact {key} is not cross-bound to inference"
            )
    if (
        saved_metrics.get(
            "computed_only_after_predictions_and_evidence_were_atomically_persisted"
        )
        is not True
    ):
        raise VerificationError("metric persistence order assertion is missing")
    fold_data = _validate_training_record(
        training,
        attempt=attempt,
        output_root=output_root,
        base=base,
        outer_train=outer_train,
        outer_test=outer_test,
        seed=seed,
        fold=fold,
        pilot=pilot,
    )
    fold_seed = derive_seed(seed, f"fold-{fold:02d}")
    checkpoint_snapshot = _snapshot_expected_file_record(
        training["checkpoint"],
        expected=attempt / "trained-model.pt",
        relative_to=output_root,
        label="training checkpoint",
    )
    predictions = _load_npz_snapshot(prediction_snapshot)
    evidence = _load_npz_snapshot(evidence_snapshot)
    _require_exact_array_keys(
        predictions, _expected_prediction_array_keys(), label="prediction"
    )
    _require_exact_array_keys(
        evidence, _expected_evidence_array_keys(), label="attention evidence"
    )
    if not np.array_equal(predictions["outer_test_indices"], outer_test):
        raise VerificationError(
            "saved prediction rows do not equal the outer test split"
        )
    if np.unique(predictions["outer_test_indices"]).size != outer_test.size:
        raise VerificationError("outer-test predictions contain duplicate rows")
    if not np.array_equal(evidence["outer_test_indices"], outer_test):
        raise VerificationError(
            "attention evidence rows do not equal the outer test split"
        )
    if not np.array_equal(evidence["outer_train_indices"], outer_train):
        raise VerificationError("attention evidence outer-train identity mismatch")
    if not np.array_equal(
        predictions["cell_ids"], np.asarray([base.cell_ids[int(i)] for i in outer_test])
    ):
        raise VerificationError("prediction cell identity mismatch")
    if not np.array_equal(predictions["gene_ids"], np.asarray(base.genes)):
        raise VerificationError("prediction gene identity mismatch")
    if not np.array_equal(
        predictions["source_cell_indices"],
        np.asarray(base.source_cell_indices[outer_test], dtype=np.int64),
    ):
        raise VerificationError("prediction source-cell identity mismatch")
    if not np.array_equal(
        predictions["source_gene_indices"],
        np.asarray(base.source_gene_indices, dtype=np.int64),
    ):
        raise VerificationError("prediction source-gene identity mismatch")
    for key, expected in (
        ("source_cell_indices", base.source_cell_indices[outer_test]),
        ("source_gene_indices", base.source_gene_indices),
        ("cell_ids", [base.cell_ids[int(i)] for i in outer_test]),
        ("gene_ids", base.genes),
    ):
        if not np.array_equal(evidence[key], np.asarray(expected)):
            raise VerificationError(f"attention evidence {key} identity mismatch")
    if (
        _require_int_field(inference, "outer_test_rows", label="inference", minimum=0)
        != outer_test.size
    ):
        raise VerificationError("recorded outer-test row count mismatch")
    if (
        _require_int_field(
            inference, "outer_test_unique_rows", label="inference", minimum=0
        )
        != outer_test.size
    ):
        raise VerificationError("recorded outer-test unique-row count mismatch")
    expected_tokens = fold_data.mutation.tokens[outer_test, :K_TOKENS]
    expected_validity = fold_data.mutation.token_mask[outer_test, :K_TOKENS]
    if not np.array_equal(evidence["token_indices"], expected_tokens):
        raise VerificationError(
            "token identities do not reproduce from outer-train fit"
        )
    if not np.array_equal(evidence["token_validity"], expected_validity):
        raise VerificationError(
            "token validity does not reproduce from outer-train fit"
        )
    expected_panel = _panel_indices(base.genes)
    if not np.array_equal(evidence["panel_gene_indices"], expected_panel):
        raise VerificationError("label-independent attention panel does not reproduce")
    expected_attention_shape = (outer_test.size, K_TOKENS)
    if evidence["mean_attention"].shape != expected_attention_shape:
        raise VerificationError("mean-attention evidence shape mismatch")
    expected_panel_shape = (outer_test.size, expected_panel.size, K_TOKENS)
    for key in ("panel_attention", "randomized_panel_attention"):
        if evidence[key].shape != expected_panel_shape:
            raise VerificationError(f"{key} evidence shape mismatch")
        if not np.isfinite(evidence[key]).all():
            raise VerificationError(f"non-finite {key} evidence")
    if not np.isfinite(evidence["mean_attention"]).all():
        raise VerificationError("non-finite mean-attention evidence")
    if evidence["randomized_mean_attention"].shape != expected_attention_shape:
        raise VerificationError("randomized mean-attention evidence shape mismatch")
    if not np.isfinite(evidence["randomized_mean_attention"]).all():
        raise VerificationError("non-finite randomized mean-attention evidence")
    prevalence, degree = _training_token_covariates(
        fold_data, outer_train, base.edge_index, base.genes
    )
    if not np.array_equal(evidence["train_prevalence_count_by_gene"], prevalence):
        raise VerificationError("training prevalence covariate mismatch")
    if not np.array_equal(evidence["string_degree_by_gene"], degree):
        raise VerificationError("STRING degree covariate mismatch")
    matching_quality: list[dict[str, Any]] = []
    for fraction in MASK_FRACTIONS:
        key = _fraction_key(fraction)
        metadata = inference["mask_plans"][key]
        if not np.isclose(float(metadata["fraction"]), fraction, rtol=0.0, atol=0.0):
            raise VerificationError(f"mask fraction metadata mismatch for {key}")
        expected_seeds = tuple(
            derive_seed(fold_seed, f"matched-{key}-draw-{draw:02d}")
            for draw in range(MATCHED_RANDOM_DRAWS)
        )
        seeds = tuple(int(value) for value in metadata["random_seeds"])
        if seeds != expected_seeds:
            raise VerificationError(f"matched-random seeds do not reproduce for {key}")
        plan = build_mask_plan(
            evidence["mean_attention"],
            expected_tokens,
            expected_validity,
            prevalence,
            degree,
            fraction=fraction,
            random_seeds=seeds,
        )
        if not np.array_equal(evidence[f"top_mask_{key}"], plan.top_mask):
            raise VerificationError(f"top-attention mask does not reproduce for {key}")
        if metadata["top_mask_sha256"] != _hash_array(plan.top_mask):
            raise VerificationError(f"top-attention mask hash mismatch for {key}")
        for draw, (mask, features) in enumerate(
            zip(plan.matched_masks, plan.matching_features)
        ):
            matching_quality.append(
                _matching_quality_diagnostic(
                    features,
                    fraction_key=key,
                    draw=draw,
                )
            )
            if not np.array_equal(evidence[f"matched_mask_{key}_d{draw:02d}"], mask):
                raise VerificationError(
                    f"matched mask does not reproduce for {key} draw {draw}"
                )
            if not np.allclose(
                evidence[f"matching_features_{key}_d{draw:02d}"],
                features,
                equal_nan=True,
                rtol=0.0,
                atol=1e-6,
            ):
                raise VerificationError(
                    f"matching features do not reproduce for {key} draw {draw}"
                )
    matching_quality_passed = bool(matching_quality) and all(
        item["passed"] is True for item in matching_quality
    )
    if not pilot and not matching_quality_passed:
        raise VerificationError(
            "formal matched-control quality gate failed: fallback selections or "
            "top-mask overlap were detected"
        )
    permutation_seed = derive_seed(fold_seed, "attention-permutation")
    if int(inference["attention_permutation"]["seed"]) != permutation_seed:
        raise VerificationError("attention permutation seed mismatch")
    permutation = build_attention_permutation(expected_validity, seed=permutation_seed)
    if not np.array_equal(evidence["attention_permutation"], permutation):
        raise VerificationError("attention permutation does not reproduce")
    if _hash_array(permutation) != inference["attention_permutation"]["sha256"]:
        raise VerificationError("attention permutation hash mismatch")

    partner = _top_partner_indices(
        evidence["panel_attention"],
        expected_tokens,
        expected_validity,
        n_genes=len(base.genes),
    )
    if not np.array_equal(evidence["panel_top_partner_indices"], partner):
        raise VerificationError("panel top partners do not reproduce")
    randomized_partner = _top_partner_indices(
        evidence["randomized_panel_attention"],
        expected_tokens,
        expected_validity,
        n_genes=len(base.genes),
    )
    if not np.array_equal(
        evidence["randomized_panel_top_partner_indices"], randomized_partner
    ):
        raise VerificationError("randomized panel top partners do not reproduce")

    model = corrected.build_model(training["best_config"], base, fold_data)
    _load_checkpoint(checkpoint_snapshot, model, "cpu")
    randomization = inference["parameter_randomization"]
    expected_randomization_seed = derive_seed(fold_seed, "parameter-randomization")
    if int(randomization["seed"]) != expected_randomization_seed:
        raise VerificationError("parameter-randomization seed mismatch")
    if randomization.get("scope") != "wq_and_wk_only":
        raise VerificationError("parameter-randomization scope mismatch")
    if (
        randomization.get("initialization")
        != "torch.nn.Linear.reset_parameters_on_cpu_then_copy"
    ):
        raise VerificationError("parameter-randomization initialization mismatch")
    if _hash_attention_parameters(model) != randomization["trained_sha256"]:
        raise VerificationError("trained attention parameter hash mismatch")
    if randomization["restored_sha256"] != randomization["trained_sha256"]:
        raise VerificationError("attention parameters were not recorded as restored")

    mode_names = set(inference["prediction_modes"])
    required_modes = set(_required_prediction_modes())
    if mode_names != required_modes or not required_modes.issubset(predictions):
        raise VerificationError("prediction mode set is incomplete")
    expected_shape = (outer_test.size, len(base.genes))
    for name in required_modes:
        if predictions[name].shape != expected_shape:
            raise VerificationError(f"prediction shape mismatch for {name}")
        if not np.isfinite(predictions[name]).all():
            raise VerificationError(f"non-finite saved predictions for {name}")
    replay = _replay_prediction_modes(
        model,
        fold_data,
        outer_test,
        saved_predictions=predictions,
        evidence=evidence,
        panel_indices=expected_panel,
        attention_permutation=permutation,
        randomization_seed=expected_randomization_seed,
        expected_randomization_record=randomization,
        device=replay_device,
        batch_size=int(manifest["arguments"]["attention_batch_size"]),
        gene_chunk_size=int(manifest["arguments"]["gene_chunk_size"]),
    )
    if replay["replayed_prediction_mode_count"] != len(required_modes):
        raise VerificationError("checkpoint replay did not cover all prediction modes")
    if replay["all_prediction_modes_matched"] is not True:
        raise VerificationError("checkpoint replay did not match all prediction modes")
    train_gene_mean = fit_gene_mean_baseline(base.gene_effect, outer_train)
    if training["train_gene_mean_sha256"] != _hash_array(train_gene_mean):
        raise VerificationError("training gene-mean reference hash mismatch")
    if not np.allclose(
        predictions["train_gene_mean"],
        train_gene_mean,
        equal_nan=True,
        rtol=0.0,
        atol=0.0,
    ):
        raise VerificationError("outer-train gene-mean reference mismatch")
    target = np.asarray(base.gene_effect[:, outer_test].T, dtype=np.float32)
    recomputed = _mode_metrics(
        {name: predictions[name] for name in required_modes}, target, train_gene_mean
    )
    _arrays_close(saved_metrics["modes"], recomputed, path="metrics.modes")
    if saved_metrics["outer_test_target_sha256"] != _hash_array(target):
        raise VerificationError("outer-test target binding mismatch")
    paired_values = _paired_fold_descriptives(recomputed)
    return {
        "attempt": str(attempt),
        "completion": completion_record,
        "seed": seed,
        "fold": fold,
        "pilot": pilot,
        "verified": True,
        "replay_complete": True,
        "replay": replay,
        "matching_quality_passed": matching_quality_passed,
        "matching_quality_diagnostics": matching_quality,
        "paired_descriptive_values": paired_values,
        "panel_gene_indices": evidence["panel_gene_indices"],
        "panel_top_partner_indices": partner,
        "scope_statement": PROTOCOL_SCOPE,
    }


def _mean_set_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    values: list[float] = []
    for row in range(left.shape[0]):
        a = set(int(value) for value in left[row] if int(value) >= 0)
        b = set(int(value) for value in right[row] if int(value) >= 0)
        if a or b:
            values.append(len(a.intersection(b)) / len(a.union(b)))
    return float(np.mean(values)) if values else float("nan")


def _verify_campaign(
    output_root: Path,
    *,
    pilot: bool,
    write_summary: bool,
    replay_device: str = "cpu",
    detached_receipt_root: Path | None = None,
    verifier_entrypoint: Path | None = None,
) -> dict[str, Any]:
    if pilot and replay_device != "cpu":
        raise VerificationError("pilot verification replay is CPU-only")
    replay_runtime = _prepare_replay_runtime(replay_device)
    scope_root = output_root / ("pilot" if pilot else "canonical")
    attempts = _completed_attempts(scope_root)
    verified: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    seen: set[tuple[int, int]] = set()
    base_cache: dict[tuple[object, ...], corrected.BaseData] = {}
    for attempt in attempts:
        try:
            manifest = _load_json(attempt / "run-manifest.json")
            seed = int(manifest["seed"])
            fold = int(manifest["fold"])
            if bool(manifest["pilot"]) is not pilot:
                raise VerificationError("attempt scope disagrees with campaign scope")
            manifest_arguments = manifest.get("arguments")
            if not isinstance(manifest_arguments, Mapping):
                raise VerificationError("attempt manifest arguments are invalid")
            _require_pilot_contract(
                pilot=pilot,
                seed=seed,
                fold=fold,
                device=str(manifest_arguments.get("device")),
            )
            pilot_cells = int(manifest["arguments"]["pilot_cells"])
            pilot_genes = int(manifest["arguments"]["pilot_genes"])
            cache_key = (
                ("pilot", seed, pilot_cells, pilot_genes) if pilot else ("canonical",)
            )
            if cache_key not in base_cache:
                base_cache[cache_key] = corrected.load_base_data(
                    pilot=pilot,
                    seed=seed,
                    pilot_cells=pilot_cells,
                    pilot_genes=pilot_genes,
                )
            result = verify_attempt(
                attempt,
                output_root=output_root,
                base=base_cache[cache_key],
                replay_device=replay_device,
            )
        except (VerificationError, OSError, KeyError, TypeError, ValueError) as exc:
            rejected.append(
                {
                    "attempt": str(attempt),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue
        key = (seed, fold)
        if key in seen:
            rejected.append(
                {
                    "attempt": str(attempt),
                    "error_type": "VerificationError",
                    "error": (
                        "duplicate verifier-accepted attempt for seed/fold "
                        f"{key}"
                    ),
                }
            )
            continue
        seen.add(key)
        verified.append(result)
    expected = (
        {(42, 0)}
        if pilot
        else {
            (seed, fold) for seed in CANONICAL_SEEDS for fold in range(CANONICAL_FOLDS)
        }
    )
    replay_gates_passed = bool(verified) and all(
        item.get("verified") is True
        and item.get("replay_complete") is True
        and int(item.get("replay", {}).get("replayed_prediction_mode_count", 0))
        == len(_required_prediction_modes())
        and item.get("replay", {}).get("all_prediction_modes_matched") is True
        and (pilot or item.get("matching_quality_passed") is True)
        and _is_exact_file_record(item.get("completion"))
        for item in verified
    )
    verification_passed = seen == expected and replay_gates_passed and not rejected
    complete = verification_passed and not pilot
    paired_analysis = _aggregate_paired_descriptives(verified) if complete else None
    scientific_result = bool(
        complete
        and paired_analysis is not None
        and paired_analysis["n_seed_folds"] == len(CANONICAL_SEEDS) * CANONICAL_FOLDS
    )
    sorted_verified = sorted(
        verified, key=lambda value: (int(value["seed"]), int(value["fold"]))
    )
    completion_roots = [
        dict(item["completion"])
        for item in sorted_verified
        if _is_exact_file_record(item.get("completion"))
    ]

    fold_jaccards: list[float] = []
    cross_seed_jaccards: list[float] = []
    for index, left in enumerate(verified):
        for right in verified[index + 1 :]:
            if not np.array_equal(
                left["panel_gene_indices"], right["panel_gene_indices"]
            ):
                continue
            value = _mean_set_jaccard(
                left["panel_top_partner_indices"], right["panel_top_partner_indices"]
            )
            fold_jaccards.append(value)
            if left["seed"] != right["seed"]:
                cross_seed_jaccards.append(value)
    summary = {
        "schema": f"{SCHEMA_VERSION}-campaign-verification",
        "verified_utc": _utc_now(),
        "scope": "pilot" if pilot else "canonical",
        "verified_seed_folds": [list(value) for value in sorted(seen)],
        "expected_seed_folds": [list(value) for value in sorted(expected)],
        "n_verified_folds": len(verified),
        "rejected_attempts": rejected,
        "n_rejected_attempts": len(rejected),
        "verification_passed": verification_passed,
        "checkpoint_replay_gates_passed": replay_gates_passed,
        "rejected_attempt_gate_passed": not rejected,
        "formal_matching_quality_gate_passed": bool(
            not pilot
            and not rejected
            and seen == expected
            and verified
            and all(item.get("matching_quality_passed") is True for item in verified)
        ),
        "required_prediction_modes_per_fold": len(_required_prediction_modes()),
        "accepted_attempt_completion_roots": completion_roots,
        "replayed_prediction_modes_total": int(
            sum(
                int(item.get("replay", {}).get("replayed_prediction_mode_count", 0))
                for item in verified
            )
        ),
        "fold_replay_evidence": [
            {
                "seed": int(item["seed"]),
                "fold": int(item["fold"]),
                "attempt": str(item["attempt"]),
                "completion": (
                    dict(item["completion"])
                    if _is_exact_file_record(item.get("completion"))
                    else None
                ),
                "replay": item.get("replay"),
                "matching_quality_passed": item.get("matching_quality_passed"),
                "matching_quality_diagnostics": item.get(
                    "matching_quality_diagnostics"
                ),
            }
            for item in sorted_verified
        ],
        "replay_device": replay_device,
        "replay_runtime": replay_runtime,
        "protocol_complete": complete,
        "eligible_for_scientific_aggregation": complete,
        "scientific_result": scientific_result,
        "scientific_result_reason": (
            "All 3 seeds x 5 folds, artifact bindings, checkpoint reconstructions, "
            "15-mode numeric replays, matched-control quality gates, absence of "
            "rejected attempts, and prespecified paired descriptive summaries passed."
            if scientific_result
            else (
                "The pilot's single planned fold is hash-verified and numerically replayed, "
                "but pilot evidence is non-scientific."
                if pilot and verification_passed
                else "The planned seed/fold set or a required verification, replay, "
                "matched-control quality, or rejected-attempt gate is incomplete."
            )
        ),
        "scientific_result_interpretation": (
            "A true value means that a complete verifier-accepted descriptive result "
            "exists; it does not assert that the estimated effects support a particular "
            "direction, and it does not establish causal biology."
        ),
        "paired_descriptive_analysis": paired_analysis,
        "attention_partner_stability": {
            "unit": "mean per-panel-gene top-partner set Jaccard",
            "top_partners": TOP_PARTNERS,
            "all_comparable_fold_pairs_mean": (
                float(np.mean(fold_jaccards)) if fold_jaccards else None
            ),
            "all_comparable_fold_pairs_n": len(fold_jaccards),
            "cross_seed_fold_pairs_mean": (
                float(np.mean(cross_seed_jaccards)) if cross_seed_jaccards else None
            ),
            "cross_seed_fold_pairs_n": len(cross_seed_jaccards),
            "limitation": (
                "Fold-level partner sets average over different held-out cells; this is a "
                "descriptive stability check and may reflect cohort composition."
            ),
        },
        "scope_statement": PROTOCOL_SCOPE,
    }
    if write_summary:
        destination = scope_root / "verification-summaries"
        destination.mkdir(parents=True, exist_ok=True)
        path = (
            destination
            / f"verification-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}.json"
        )
        _write_json_once(path, summary)
        print(f"[attention] verification summary={path}", flush=True)
    if detached_receipt_root is not None:
        detached = write_detached_verification_package(
            summary,
            output_root=output_root,
            receipt_root=detached_receipt_root,
            verifier_entrypoint=verifier_entrypoint,
        )
        print(
            f"[attention] detached verification summary={detached['summary']}",
            flush=True,
        )
        print(
            f"[attention] detached verification receipt={detached['receipt']}",
            flush=True,
        )
    return summary


def verify_campaign(
    output_root: Path,
    *,
    pilot: bool,
    write_summary: bool,
    replay_device: str = "cpu",
    detached_receipt_root: Path | None = None,
    verifier_entrypoint: Path | None = None,
) -> dict[str, Any]:
    """Verify a campaign and normalize malformed evidence failures."""

    try:
        return _verify_campaign(
            output_root,
            pilot=pilot,
            write_summary=write_summary,
            replay_device=replay_device,
            detached_receipt_root=detached_receipt_root,
            verifier_entrypoint=verifier_entrypoint,
        )
    except VerificationError:
        raise
    except (
        OSError,
        RuntimeError,
        ImportError,
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
        IndexError,
    ) as exc:
        raise VerificationError(
            "malformed attention campaign evidence under "
            f"{output_root}: {type(exc).__name__}: {exc}"
        ) from exc


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan", help="print the frozen formal design")

    def add_execution_options(target: argparse.ArgumentParser) -> None:
        target.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
        target.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
        target.add_argument(
            "--attention-batch-size",
            type=_positive_int,
            default=DEFAULT_ATTENTION_BATCH_SIZE,
        )
        target.add_argument(
            "--gene-chunk-size", type=_positive_int, default=DEFAULT_GENE_CHUNK_SIZE
        )
        target.add_argument("--pilot", action="store_true")
        target.add_argument("--pilot-cells", type=_positive_int, default=96)
        target.add_argument("--pilot-genes", type=_positive_int, default=128)

    run_parser = subparsers.add_parser("run", help="run one seed, resumably")
    add_execution_options(run_parser)
    run_parser.add_argument("--seed", type=int, choices=CANONICAL_SEEDS, default=42)
    run_parser.add_argument("--fold-id", type=int, default=-1)

    campaign_parser = subparsers.add_parser(
        "campaign", help="run the fixed 3-seed x 5-fold formal campaign"
    )
    add_execution_options(campaign_parser)
    campaign_parser.add_argument(
        "--detached-receipt-root",
        type=Path,
        default=None,
        help="write a checksum-bound verification package outside output-root",
    )

    verify_parser = subparsers.add_parser(
        "verify", help="verify artifacts and replay every mode from each checkpoint"
    )
    verify_parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    verify_parser.add_argument(
        "--replay-device", choices=["cpu", "cuda"], default="cpu"
    )
    verify_parser.add_argument("--pilot", action="store_true")
    verify_parser.add_argument("--no-write-summary", action="store_true")
    verify_parser.add_argument(
        "--detached-receipt-root",
        type=Path,
        default=None,
        help="write a checksum-bound verification package outside output-root",
    )
    args = parser.parse_args(argv)
    if getattr(args, "pilot", False) and getattr(args, "device", "cpu") != "cpu":
        parser.error("pilot mode is CPU-only")
    if getattr(args, "pilot", False) and args.command == "campaign":
        parser.error("pilot mode uses 'run --pilot'; it is not a campaign")
    if (
        getattr(args, "pilot", False)
        and args.command == "verify"
        and args.replay_device != "cpu"
    ):
        parser.error("pilot verification is CPU-only")
    if args.command == "run" and args.pilot and args.seed != 42:
        parser.error("pilot mode is fixed to --seed 42")
    if args.command == "run" and args.fold_id < -1:
        parser.error("--fold-id must be -1 or non-negative")
    return args


def _run_seed(args: argparse.Namespace, seed: int) -> list[Path]:
    folds = [0] if args.pilot else list(range(CANONICAL_FOLDS))
    if args.fold_id >= 0:
        if args.fold_id not in folds:
            raise VerificationError(
                f"--fold-id {args.fold_id} is outside the planned folds"
            )
        folds = [args.fold_id]
    _require_pilot_contract(
        pilot=bool(args.pilot),
        seed=int(seed),
        fold=int(folds[0]),
        device=str(args.device),
    )
    _prepare_replay_runtime(args.device)
    base = corrected.load_base_data(
        pilot=bool(args.pilot),
        seed=seed,
        pilot_cells=int(args.pilot_cells),
        pilot_genes=int(args.pilot_genes),
    )
    code_records = _code_records()
    input_records = _input_records()
    completed: list[Path] = []
    for fold in folds:
        completed.append(
            run_fold(
                args,
                seed=seed,
                fold=fold,
                base=base,
                code_records=code_records,
                input_records=input_records,
            )
        )
    return completed


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "plan":
        print(
            json.dumps(
                {
                    "schema": f"{SCHEMA_VERSION}-plan",
                    "formal_seed_folds": [
                        [seed, fold]
                        for seed in CANONICAL_SEEDS
                        for fold in range(CANONICAL_FOLDS)
                    ],
                    "n_final_refits": len(CANONICAL_SEEDS) * CANONICAL_FOLDS,
                    "inner_grid_size": len(corrected.FULL_GRID),
                    "corrected_protocol_arguments": corrected.CANONICAL_ARGUMENTS,
                    "perturbation_protocol": _parameter_manifest(),
                    "scope_statement": PROTOCOL_SCOPE,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "verify":
        write_summary = not bool(args.no_write_summary)
        try:
            detached_receipt_root = args.detached_receipt_root
            if detached_receipt_root is None and write_summary and not args.pilot:
                detached_receipt_root = _default_detached_receipt_root(
                    args.output_root
                )
            summary = verify_campaign(
                args.output_root,
                pilot=bool(args.pilot),
                write_summary=write_summary,
                replay_device=args.replay_device,
                detached_receipt_root=detached_receipt_root,
            )
        except (
            VerificationError,
            OSError,
            RuntimeError,
            KeyError,
            TypeError,
            ValueError,
            AttributeError,
            ImportError,
        ) as exc:
            print(f"attention verification failed: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(summary, indent=2, sort_keys=True))
        if not summary["verification_passed"]:
            return 2
        if not args.pilot and not summary["scientific_result"]:
            return 2
        return 0
    if args.command == "run":
        try:
            completed = _run_seed(args, int(args.seed))
        except (
            VerificationError,
            OSError,
            RuntimeError,
            KeyError,
            TypeError,
            ValueError,
            AttributeError,
            ImportError,
        ) as exc:
            print(f"attention evidence run failed: {exc}", file=sys.stderr)
            return 2
        print(f"[attention] completed fold attempts={len(completed)}", flush=True)
        return 0
    if args.command == "campaign":
        try:
            for seed in CANONICAL_SEEDS:
                _run_seed(args, seed)
            detached_receipt_root = (
                args.detached_receipt_root
                if args.detached_receipt_root is not None
                else _default_detached_receipt_root(args.output_root)
            )
            summary = verify_campaign(
                args.output_root,
                pilot=False,
                write_summary=True,
                replay_device=args.device,
                detached_receipt_root=detached_receipt_root,
            )
        except (
            VerificationError,
            OSError,
            RuntimeError,
            KeyError,
            TypeError,
            ValueError,
            AttributeError,
            ImportError,
        ) as exc:
            print(f"attention campaign failed: {exc}", file=sys.stderr)
            return 2
        return 0 if summary["scientific_result"] else 2
    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
