"""Independent verifier for the pinned GATDep audit and corrected runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import types
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


OFFICIAL_COMMIT = "047ffdc2df8fafe58dd50d91293cd0e0cb7cb0a1"
AUDIT_SCHEMA = "GATDep-official-audit-v2"
RUN_SCHEMA = "GATDep-corrected-adaptation-run-v2"
PREDICTION_SCHEMA = "GATDep-corrected-outer-test-predictions-v1"
COMPARISON_VERIFICATION_SCHEMA = "DepPLM-GATDep-common-cohort-verification-v3"
COMPARISON_REPLAY_TOLERANCE = {"rtol": 1e-5, "atol": 1e-6}
COMPARISON_REPLAY_INPUT_POLICY = (
    "not_embedded_reconstructed_from_hash_bound_project_inputs"
)
COMPARISON_PREPROCESSING_CONTENT_POLICY = (
    "training_fitted_aggregate_statistics_and_feature_definitions_no_raw_cell_rows"
)
COMPARISON_ARTIFACT_REDISTRIBUTION_STATUS = "not_cleared_for_public_redistribution"
SUPPORTED_COMPARISON_CONDITIONS = {"DepPLM", "C-PLM-MLP"}
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
CRITICAL_BLOBS = {
    "GeneDependencyGAT.py": "4a9ba8ed73fcd00cf1c2fccad9753e3edb0f5c89",
    "geneExpression2GSVA.R": "c818d2a1312a4adff891f455f8d8e16906d86a77",
    "environment.yml": "765a410ca959bddddffe046db5f3546a5d62438e",
    "GATDep.ipynb": "c76458ac8993b52435fc5163059b54b9ea0b5524",
    "datasets/adj_gene_geneset.npy": "fb3ebcdc4bec600d58526b38f14900a0894e054d",
    "datasets/genes2ID_2k.json": "c926cac3f6b04d37e5f12d408ffcbcbc86f7bdf0",
    "datasets/genesets_select_6k.json": "1d153ad142d38ca1aa5a87b2e87e5143d4f70f75",
    "datasets/ggi_id.txt": "36d40f98ae7394f168ca8650b6dbe52ccd8fa0a4",
    "datasets/gsva_score_go_kegg_v3_format.npy": (
        "bb0962e1868804a29f91bd75de2847cdc1b4563f"
    ),
    "datasets/samples2ID.json": "2a9541f1a27390624062c12a31de66b6b829ae54",
}
REQUIRED_RUN_SOURCE_PATHS = (
    "code/gatdep_official_baseline.py",
    "code/verify_gatdep_official_baseline.py",
)
REQUIRED_PROJECT_INPUT_PATHS = (
    "data/processed/cell_lines.txt",
    "data/processed/genes.txt",
    "data/processed/lineages.txt",
    "data/processed/gene_effect.npy",
    "data/depmap/OmicsExpressionProteinCodingGenesTPMLogp1.csv",
)
REQUIRED_OFFICIAL_INPUT_PATHS = (
    "datasets/adj_gene_geneset.npy",
    "datasets/genes2ID_2k.json",
    "datasets/genesets_select_6k.json",
    "datasets/samples2ID.json",
    "datasets/ggi_id.txt",
    "GeneDependencyGAT.py",
)
REQUIRED_ARRAYS = {
    "prediction",
    "target_raw",
    "train_gene_mean",
    "outer_test_indices_common",
    "source_cell_indices_project",
    "source_gene_indices_project",
    "official_cell_indices",
    "official_gene_indices",
    "cell_ids",
    "gene_ids",
}


class VerificationError(ValueError):
    """A fail-closed evidence validation error."""


def write_json_exclusive(path: str | Path, value: Mapping[str, Any]) -> None:
    """Durably create a JSON report without replacing an existing artifact."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp-{uuid.uuid4().hex}")
    payload = (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise VerificationError(f"refusing to overwrite {target}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    """Hash checkout bytes exactly as Git stores a blob."""

    size = path.stat().st_size
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1_bytes(source: bytes) -> str:
    """Hash already-read source bytes exactly as Git stores a blob."""

    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(source)}\0".encode("ascii"))
    digest.update(source)
    return digest.hexdigest()


def read_pinned_text_source(path: Path, expected_blob: str) -> bytes:
    """Return canonical Git bytes, allowing only Git's CRLF checkout smudge."""

    source = path.read_bytes()
    if git_blob_sha1_bytes(source) == expected_blob:
        return source
    normalized = source.replace(b"\r\n", b"\n")
    if normalized != source and git_blob_sha1_bytes(normalized) == expected_blob:
        return normalized
    raise VerificationError(
        "loaded official model source bytes differ from the pinned blob"
    )


def _parse_torch_version(value: Any) -> tuple[str, str | None] | None:
    """Parse the exact PyTorch version forms accepted by the runtime contract."""

    if not isinstance(value, str):
        return None
    match = TORCH_VERSION_PATTERN.fullmatch(value)
    if match is None:
        return None
    return match.group("base"), match.group("tag")


def runtime_mismatches(runtime: Mapping[str, Any]) -> list[str]:
    """Recompute the runner's exact distribution-version execution gate."""

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


def verify_runtime_record(runtime: Any, *, record_name: str) -> list[str]:
    """Validate recorded build/platform details without asserting bit identity."""

    if not isinstance(runtime, Mapping):
        return [f"{record_name} runtime record is missing or invalid"]

    errors: list[str] = []
    packages = runtime.get("packages", {})
    if not isinstance(packages, Mapping):
        packages = {}
        errors.append(f"{record_name} runtime package record is invalid")

    for field in ("python_executable", "platform", "platform_system", "machine"):
        if not isinstance(runtime.get(field), str) or not runtime[field]:
            errors.append(f"{record_name} runtime lacks {field}")

    torch_build = runtime.get("torch_build", {})
    if not isinstance(torch_build, Mapping):
        torch_build = {}
        errors.append(f"{record_name} PyTorch build record is invalid")
    module_version = torch_build.get("module_version")
    if not isinstance(module_version, str) or not module_version:
        errors.append(f"{record_name} runtime does not record torch.__version__")
        module_base = None
        expected_build_tag = None
    else:
        module_parts = _parse_torch_version(module_version)
        if module_parts is None:
            errors.append(f"{record_name} torch.__version__ is malformed")
            module_base = None
            expected_build_tag = None
        else:
            module_base, expected_build_tag = module_parts
    if torch_build.get("base_version") != module_base:
        errors.append(f"{record_name} PyTorch module base version is inconsistent")
    if torch_build.get("build_tag") != expected_build_tag:
        errors.append(f"{record_name} PyTorch build tag is inconsistent")
    distribution_version = packages.get("torch")
    distribution_parts = _parse_torch_version(distribution_version)
    if isinstance(distribution_version, str) and distribution_parts is None:
        errors.append(f"{record_name} torch distribution version is malformed")
    distribution_base = (
        distribution_parts[0] if distribution_parts is not None else None
    )
    distribution_build_tag = (
        distribution_parts[1] if distribution_parts is not None else None
    )
    if distribution_base != module_base:
        errors.append(
            f"{record_name} PyTorch distribution and module base versions differ"
        )
    if (
        distribution_build_tag is not None
        and distribution_build_tag != expected_build_tag
    ):
        errors.append(
            f"{record_name} PyTorch distribution and module build tags differ"
        )

    cuda_runtime = torch_build.get("cuda_runtime")
    cuda_available = torch_build.get("cuda_available")
    device_count = torch_build.get("device_count")
    devices = torch_build.get("devices")
    compiled_architectures = torch_build.get("compiled_architectures")
    if (
        not isinstance(compiled_architectures, list)
        or not all(
            isinstance(value, str) and value.startswith("sm_") and value[3:].isdigit()
            for value in compiled_architectures
        )
        or compiled_architectures != sorted(set(compiled_architectures))
    ):
        errors.append(f"{record_name} runtime has invalid compiled GPU architectures")
        compiled_architectures = []
    if not isinstance(cuda_available, bool):
        errors.append(f"{record_name} runtime has invalid CUDA availability")
    if (
        not isinstance(device_count, int)
        or isinstance(device_count, bool)
        or device_count < 0
    ):
        errors.append(f"{record_name} runtime has invalid CUDA device count")
        device_count = -1
    if not isinstance(devices, list) or len(devices) != device_count:
        errors.append(f"{record_name} runtime has inconsistent GPU inventory")
        devices = []
    for index, device in enumerate(devices):
        invalid_device = (
            not isinstance(device, Mapping)
            or device.get("index") != index
            or not isinstance(device.get("name"), str)
            or not device.get("name")
            or not isinstance(device.get("capability"), list)
            or len(device.get("capability", [])) != 2
            or not all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in device.get("capability", [])
            )
        )
        if invalid_device:
            errors.append(f"{record_name} runtime has invalid GPU record {index}")
            continue
        capability = device["capability"]
        architecture = f"sm_{capability[0]}{capability[1]}"
        if device.get("architecture") != architecture:
            errors.append(
                f"{record_name} runtime GPU {index} architecture is inconsistent"
            )
        if device.get("kernel_supported_by_build") is not (
            architecture in compiled_architectures
        ):
            errors.append(
                f"{record_name} runtime GPU {index} kernel support is inconsistent"
            )
    if cuda_available is False and device_count != 0:
        errors.append(f"{record_name} runtime CUDA state contradicts device count")
    if cuda_available is True and device_count < 1:
        errors.append(f"{record_name} runtime CUDA state lacks a visible GPU")
    if cuda_available is True and not isinstance(cuda_runtime, str):
        errors.append(f"{record_name} runtime lacks the active CUDA runtime version")
    if expected_build_tag == "cpu" and (
        cuda_runtime is not None or cuda_available is not False or device_count != 0
    ):
        errors.append(f"{record_name} CPU build has inconsistent CUDA metadata")
    if expected_build_tag == "cpu" and compiled_architectures:
        errors.append(f"{record_name} CPU build unexpectedly records CUDA kernels")
    if (
        isinstance(expected_build_tag, str)
        and expected_build_tag.startswith("cu")
        and len(expected_build_tag) == 5
        and expected_build_tag[2:].isdigit()
    ):
        digits = expected_build_tag[2:]
        expected_cuda_runtime = f"{digits[:2]}.{digits[2:]}"
        if cuda_runtime != expected_cuda_runtime:
            errors.append(f"{record_name} PyTorch build tag and CUDA runtime differ")
    expected_cuda_execution_compatible = bool(
        cuda_available is True
        and any(
            isinstance(device, Mapping)
            and device.get("kernel_supported_by_build") is True
            for device in devices
        )
    )
    if torch_build.get("cuda_execution_compatible") is not (
        expected_cuda_execution_compatible
    ):
        errors.append(
            f"{record_name} CUDA execution compatibility flag is inconsistent"
        )

    mismatches = runtime_mismatches(runtime)
    fidelity = runtime.get("fidelity", {})
    if not isinstance(fidelity, Mapping):
        fidelity = {}
        errors.append(f"{record_name} runtime fidelity record is invalid")
    platform_compatible = (
        runtime.get("platform_system") == "Linux"
        and module_base == EXPECTED_RUNTIME["torch"]
        and cuda_runtime == UPSTREAM_PLATFORM_CONTRACT["cuda_runtime_major_minor"]
    )
    expected_classification = (
        "upstream_declared_platform_compatible_not_bit_exact"
        if platform_compatible
        else "platform_adapted_local_runtime_not_upstream_bit_exact"
    )
    if fidelity.get("distribution_versions_exact") is not (not mismatches):
        errors.append(
            f"{record_name} distribution-version fidelity flag is inconsistent"
        )
    if fidelity.get("torch_distribution_version") != packages.get("torch"):
        errors.append(
            f"{record_name} fidelity record has the wrong torch distribution version"
        )
    if fidelity.get("upstream_platform_contract") != UPSTREAM_PLATFORM_CONTRACT:
        errors.append(
            f"{record_name} fidelity record has the wrong upstream platform contract"
        )
    if fidelity.get("upstream_platform_compatible") is not platform_compatible:
        errors.append(
            f"{record_name} platform compatibility classification is inconsistent"
        )
    if fidelity.get("cuda_device_kernel_compatible") is not (
        expected_cuda_execution_compatible
    ):
        errors.append(
            f"{record_name} CUDA device/build compatibility record is inconsistent"
        )
    if fidelity.get("classification") != expected_classification:
        errors.append(f"{record_name} runtime fidelity classification is inconsistent")
    if fidelity.get("upstream_bit_exact_environment_claimed") is not False:
        errors.append(
            f"{record_name} improperly claims an upstream bit-exact environment"
        )
    return errors


def expected_run_file_inventories(
    project_root: Path, official_root: Path
) -> dict[str, tuple[Path, ...]]:
    """Return the only source and input paths accepted for a formal run."""

    return {
        "source_files": tuple(
            project_root / relative for relative in REQUIRED_RUN_SOURCE_PATHS
        ),
        "input_files": (
            *(project_root / relative for relative in REQUIRED_PROJECT_INPUT_PATHS),
            *(official_root / relative for relative in REQUIRED_OFFICIAL_INPUT_PATHS),
        ),
    }


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def verify_fixed_file_inventory(
    records: Any,
    expected_paths: Sequence[Path],
    *,
    group_name: str,
) -> list[str]:
    """Verify an exact, duplicate-free manifest inventory and every file hash."""

    errors: list[str] = []
    expected = {_path_key(path): Path(os.path.abspath(path)) for path in expected_paths}
    if len(expected) != len(expected_paths):
        raise VerificationError(
            f"internal duplicate in required {group_name} inventory"
        )
    if not isinstance(records, list):
        return [f"run manifest lacks {group_name}"]

    observed: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            errors.append(f"invalid {group_name} record")
            continue
        path = Path(record["path"])
        if not path.is_absolute():
            errors.append(f"{group_name} path is not absolute: {record['path']}")
            continue
        key = _path_key(path)
        observed.setdefault(key, []).append(record)

    for key, matching in observed.items():
        if len(matching) > 1:
            errors.append(f"duplicate {group_name} path: {matching[0]['path']}")
        if key not in expected:
            errors.append(f"unexpected {group_name} path: {matching[0]['path']}")

    for key, path in expected.items():
        matching = observed.get(key)
        if not matching:
            errors.append(f"missing required {group_name} path: {path}")
            continue
        record = matching[0]
        recorded_bytes = record.get("bytes")
        recorded_sha256 = record.get("sha256")
        if (
            not isinstance(recorded_bytes, int)
            or isinstance(recorded_bytes, bool)
            or not isinstance(recorded_sha256, str)
        ):
            errors.append(f"invalid {group_name} hash record: {path}")
            continue
        if (
            not path.is_file()
            or path.stat().st_size != recorded_bytes
            or sha256_file(path) != recorded_sha256
        ):
            errors.append(f"bound source/input changed: {path}")
    return errors


def hash_strings(values: Sequence[str]) -> str:
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
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"{path} does not contain a JSON object")
    return value


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]


def ordered_mapping(path: Path) -> tuple[list[str], dict[str, int]]:
    value = load_json(path)
    mapping = {str(key): int(index) for key, index in value.items()}
    if len(set(mapping.values())) != len(mapping):
        raise VerificationError(f"duplicate indices in {path}")
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise VerificationError(f"non-contiguous indices in {path}")
    return (
        [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])],
        mapping,
    )


def git(root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise VerificationError(f"git {' '.join(args)} failed: {exc}") from exc


def verify_pinned_model_source(official_root: Path) -> list[str]:
    """Verify the commit and canonical bytes used for model import."""

    errors: list[str] = []
    if git(official_root, "rev-parse", "HEAD") != OFFICIAL_COMMIT:
        errors.append("current official checkout is not at the pinned commit")
    if git(official_root, "status", "--porcelain", "--untracked-files=no"):
        errors.append("current official checkout has tracked modifications")

    relative = "GeneDependencyGAT.py"
    expected_blob = CRITICAL_BLOBS[relative]
    commit_blob = git(official_root, "rev-parse", f"{OFFICIAL_COMMIT}:{relative}")
    if commit_blob != expected_blob:
        errors.append("official model source commit blob differs from the pinned blob")

    source_path = official_root / relative
    if not source_path.is_file():
        errors.append("official model source is missing from the checkout")
    else:
        try:
            read_pinned_text_source(source_path, expected_blob)
        except VerificationError as exc:
            errors.append(str(exc))
    return errors


def load_verified_model_class(official_root: Path) -> type[Any]:
    """Execute only the pinned source bytes, bypassing Python bytecode caches."""

    source_errors = verify_pinned_model_source(official_root)
    if source_errors:
        raise VerificationError("; ".join(source_errors))

    source_path = official_root / "GeneDependencyGAT.py"
    source = read_pinned_text_source(
        source_path, CRITICAL_BLOBS["GeneDependencyGAT.py"]
    )

    module = types.ModuleType("gatdep_pinned_model_replay")
    module.__file__ = str(source_path)
    module.__package__ = ""
    code = compile(source, str(source_path), "exec", dont_inherit=True)
    exec(code, module.__dict__)
    model_class = getattr(module, "GeneDependencyGAT", None)
    if model_class is None:
        raise VerificationError("pinned source does not define GeneDependencyGAT")
    return model_class


def make_outer_splits(n_cells: int, n_splits: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(int(seed))
    return [
        np.sort(part).astype(np.int64)
        for part in np.array_split(rng.permutation(n_cells), n_splits)
    ]


def derive_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(int(seed)), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def make_inner_split(
    outer_train: np.ndarray,
    lineages: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    outer = np.asarray(outer_train, dtype=np.int64)
    categories = np.asarray(lineages, dtype=str)
    rng = np.random.default_rng(int(seed))
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    for category in sorted(set(categories[outer].tolist())):
        members = outer[categories[outer] == category].copy()
        rng.shuffle(members)
        if members.size < 2:
            train_parts.append(members)
            continue
        n_validation = min(members.size - 1, max(1, round(members.size * val_fraction)))
        validation_parts.append(members[:n_validation])
        train_parts.append(members[n_validation:])
    inner_train = np.sort(np.concatenate(train_parts)).astype(np.int64)
    if validation_parts:
        inner_validation = np.sort(np.concatenate(validation_parts)).astype(np.int64)
    elif inner_train.size >= 2:
        inner_validation = inner_train[-1:].copy()
        inner_train = inner_train[:-1]
    else:
        raise VerificationError("outer training split is too small")
    return inner_train, inner_validation


def expected_split_record(
    cohort: Mapping[str, Any],
    lineages: np.ndarray,
    outer_test: np.ndarray,
    *,
    experiment_seed: int,
    fold_number: int,
) -> dict[str, Any]:
    outer_train = np.asarray(
        sorted(set(range(len(cohort["cell_ids"]))) - set(outer_test.tolist())),
        dtype=np.int64,
    )
    fold_seed = derive_seed(experiment_seed, f"fold-{fold_number:02d}")
    inner_train, inner_validation = make_inner_split(
        outer_train,
        lineages,
        0.2,
        derive_seed(fold_seed, "inner-split"),
    )

    def describe(indices: np.ndarray) -> dict[str, Any]:
        return {
            "indices": indices.tolist(),
            "cell_ids": [cohort["cell_ids"][int(index)] for index in indices],
            "indices_sha256": hash_array(indices.astype("<i8", copy=False)),
        }

    record = {
        "schema": "GATDep-corrected-split-v1",
        "seed": fold_seed,
        "fold_number": fold_number,
        "outer_train": describe(outer_train),
        "inner_train": describe(inner_train),
        "inner_validation": describe(inner_validation),
        "outer_test": describe(outer_test),
        "cohort_contract_sha256": cohort["contract"]["sha256"],
        "fit_scope": {
            "expression_scaling": "inner_train_then_outer_train_refit",
            "pathway_aggregation": "fixed_annotation_transform",
            "target_scaling": "none_raw_gene_effect",
            "outer_test": "transform_and_evaluate_exactly_once_after_selection",
        },
    }
    return {**record, "split_sha256": canonical_hash(record)}


def observed_official_assets(official_root: Path) -> dict[str, Any]:
    datasets = official_root / "datasets"
    genes, _ = ordered_mapping(datasets / "genes2ID_2k.json")
    gene_sets, _ = ordered_mapping(datasets / "genesets_select_6k.json")
    cells, _ = ordered_mapping(datasets / "samples2ID.json")
    adjacency = np.load(datasets / "adj_gene_geneset.npy", mmap_mode="r")
    gsva = np.load(datasets / "gsva_score_go_kegg_v3_format.npy", mmap_mode="r")
    graph_path = datasets / "ggi_id.txt"
    with graph_path.open("r", encoding="utf-8") as handle:
        columns = handle.readline().strip().split()
    endpoints = np.loadtxt(
        graph_path,
        skiprows=1,
        usecols=(0, 1),
        dtype=np.int64,
        ndmin=2,
    )
    duplicate_count = int(endpoints.shape[0] - np.unique(endpoints, axis=0).shape[0])
    return {
        "genes": {"count": len(genes), "ordered_names_sha256": hash_strings(genes)},
        "gene_sets": {
            "count": len(gene_sets),
            "ordered_names_sha256": hash_strings(gene_sets),
        },
        "cells": {"count": len(cells), "ordered_names_sha256": hash_strings(cells)},
        "gene_gene_set_adjacency": {
            "shape": list(adjacency.shape),
            "dtype": str(adjacency.dtype),
            "nonzero": int(np.count_nonzero(adjacency)),
        },
        "precomputed_gsva": {"shape": list(gsva.shape), "dtype": str(gsva.dtype)},
        "gene_graph": {
            "rows": int(endpoints.shape[0]),
            "columns": columns,
            "minimum_endpoint": int(endpoints.min()),
            "maximum_endpoint": int(endpoints.max()),
            "self_edges": int((endpoints[:, 0] == endpoints[:, 1]).sum()),
            "duplicate_directed_edges": duplicate_count,
        },
    }


def observed_protocol(official_root: Path) -> dict[str, Any]:
    notebook = json.loads((official_root / "GATDep.ipynb").read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    r_source = (official_root / "geneExpression2GSVA.R").read_text(encoding="utf-8")
    return {
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
            "checkpoint_selected_by_validation_loss": (
                "if val_loss < best_val_loss" in source
            ),
        },
        "gsva_script_static_checks": {
            "gaussian_kernel": 'kcdf="Gaussian"' in r_source,
            "single_cohort_call": "gsva(as.matrix(gene_expression)" in r_source,
            "serialized_fit_state": False,
            "held_out_transform_api": False,
        },
    }


def load_expression_support(
    project_root: Path,
    official_root: Path,
    cohort: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    expression_path = (
        project_root / "data/depmap/OmicsExpressionProteinCodingGenesTPMLogp1.csv"
    )
    header = pd.read_csv(expression_path, nrows=0).columns.tolist()
    id_column = header[0]
    expression_columns: dict[str, str] = {}
    for column in header[1:]:
        expression_columns.setdefault(column.split(" (")[0], column)
    official_genes, official_gene_map = ordered_mapping(
        official_root / "datasets/genes2ID_2k.json"
    )
    support_names = [gene for gene in official_genes if gene in expression_columns]
    selected_columns = [expression_columns[gene] for gene in support_names]
    frame = pd.read_csv(
        expression_path,
        usecols=[id_column, *selected_columns],
        index_col=0,
        dtype={column: np.float32 for column in selected_columns},
    )
    missing_cells = [cell for cell in cohort["cell_ids"] if cell not in frame.index]
    if missing_cells:
        raise VerificationError(
            f"expression is missing {len(missing_cells)} common cohort cells"
        )
    expression = frame.reindex(cohort["cell_ids"]).to_numpy(dtype=np.float32, copy=True)
    adjacency = np.load(
        official_root / "datasets/adj_gene_geneset.npy", allow_pickle=False
    ).astype(np.float32, copy=False)
    support_indices = np.asarray(
        [official_gene_map[gene] for gene in support_names], dtype=np.int64
    )
    return expression, np.asarray(adjacency[support_indices], dtype=np.float32)


def expected_preprocessing_provenance(
    expression: np.ndarray,
    fit_indices: np.ndarray,
    support_membership: np.ndarray,
) -> dict[str, Any]:
    mean, scale, weights = fitted_pathway_components(
        expression, fit_indices, support_membership
    )
    return {
        "method": FEATURE_METHOD,
        "fit_scope": "provided_training_indices_only",
        "fit_indices_sha256": hash_array(fit_indices.astype("<i8", copy=False)),
        "mean_sha256": hash_array(mean),
        "scale_sha256": hash_array(scale),
        "pathway_weights_sha256": hash_array(weights),
        "n_expression_genes": int(mean.size),
        "n_pathways": int(weights.shape[1]),
        "material_deviation": (
            "This is a train-z-scored pathway mean over expression-available "
            "official GATDep genes. It is not GSVA and must not be labeled as "
            "the upstream feature generator."
        ),
    }


def fitted_pathway_components(
    expression: np.ndarray,
    fit_indices: np.ndarray,
    support_membership: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independently reproduce the runner's fold-local pathway transform."""

    values = np.asarray(expression, dtype=np.float64)
    membership = np.asarray(support_membership, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(values[fit_indices], axis=0)
        scale = np.nanstd(values[fit_indices], axis=0, ddof=0)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float64, copy=False)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0).astype(
        np.float64, copy=False
    )
    denominator = membership.sum(axis=0)
    weights = (membership / np.where(denominator > 0, denominator, 1.0)).astype(
        np.float64, copy=False
    )
    return mean, scale, weights


def compare_replayed_predictions(
    stored: np.ndarray,
    replayed: np.ndarray,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> dict[str, Any]:
    """Return auditable numeric agreement between stored and replayed predictions."""

    observed = np.asarray(stored, dtype=np.float32)
    reproduced = np.asarray(replayed, dtype=np.float32)
    same_shape = observed.shape == reproduced.shape
    finite = bool(np.isfinite(observed).all() and np.isfinite(reproduced).all())
    if same_shape and observed.size and finite:
        max_abs_error = float(np.max(np.abs(observed - reproduced)))
        matched = bool(np.allclose(observed, reproduced, rtol=rtol, atol=atol))
    else:
        max_abs_error = None
        matched = False
    return {
        "matched": matched,
        "same_shape": same_shape,
        "finite": finite,
        "stored_shape": list(observed.shape),
        "replayed_shape": list(reproduced.shape),
        "rtol": rtol,
        "atol": atol,
        "max_abs_error": max_abs_error,
        "stored_sha256": hash_array(observed),
        "replayed_sha256": hash_array(reproduced),
    }


def replay_checkpoint_predictions(
    *,
    checkpoint_path: Path,
    official_root: Path,
    cohort: Mapping[str, Any],
    expression: np.ndarray,
    support_membership: np.ndarray,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Reload the official model checkpoint and independently replay outer inference."""

    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    if device == "cuda" and not torch.cuda.is_available():
        raise VerificationError("CUDA replay requested but CUDA is unavailable")
    model_class = load_verified_model_class(official_root)

    adjacency = np.load(
        official_root / "datasets/adj_gene_geneset.npy", allow_pickle=False
    ).astype(np.float32, copy=False)
    node_membership = np.asarray(
        adjacency[np.asarray(cohort["official_gene_indices"], dtype=np.int64)],
        dtype=np.float32,
    )
    official_genes, _ = ordered_mapping(official_root / "datasets/genes2ID_2k.json")
    graph = pd.read_csv(official_root / "datasets/ggi_id.txt", sep=r"\s+")
    remap = np.full(len(official_genes), -1, dtype=np.int64)
    remap[np.asarray(cohort["official_gene_indices"], dtype=np.int64)] = np.arange(
        len(cohort["gene_ids"])
    )
    original_edges = graph[["g1_id", "g2_id"]].to_numpy(dtype=np.int64, copy=False)
    keep = (remap[original_edges[:, 0]] >= 0) & (remap[original_edges[:, 1]] >= 0)
    edge_index = np.stack(
        [remap[original_edges[keep, 0]], remap[original_edges[keep, 1]]], axis=0
    ).astype(np.int64, copy=False)
    edge_tensor = torch.from_numpy(edge_index).long()
    mean, scale, weights = fitted_pathway_components(
        expression, outer_train, support_membership
    )

    class ReplayDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return int(outer_test.size)

        def __getitem__(self, item: int) -> Any:
            index = int(outer_test[int(item)])
            values = np.asarray(expression[index : index + 1], dtype=np.float64)
            clean = np.where(np.isfinite(values), values, mean)
            pathway = (((clean - mean) / scale) @ weights).astype(
                np.float32, copy=False
            )[0]
            node_features = node_membership * pathway[None, :]
            return Data(
                x=torch.from_numpy(np.asarray(node_features, dtype=np.float32)),
                edge_index=edge_tensor,
            )

    model = model_class(in_channels=6_561, hidden_channels=64, out_channels=1)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    loader = DataLoader(
        ReplayDataset(), batch_size=int(batch_size), shuffle=False, num_workers=0
    )
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.batch)
            predictions.append(
                output.reshape(batch.num_graphs, -1).float().cpu().numpy()
            )
    if not predictions:
        raise VerificationError("checkpoint replay produced no predictions")
    return np.concatenate(predictions, axis=0).astype(np.float32, copy=False)


def common_cohort(project_root: Path, official_root: Path) -> dict[str, Any]:
    project_cells = read_lines(project_root / "data/processed/cell_lines.txt")
    project_genes = read_lines(project_root / "data/processed/genes.txt")
    project_lineages = np.asarray(
        read_lines(project_root / "data/processed/lineages.txt"), dtype=str
    )
    if project_lineages.size != len(project_cells):
        raise VerificationError("project lineages do not align with cell manifest")
    official_cells, official_cell_map = ordered_mapping(
        official_root / "datasets/samples2ID.json"
    )
    official_genes, official_gene_map = ordered_mapping(
        official_root / "datasets/genes2ID_2k.json"
    )
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
    official_cell_indices = np.asarray(
        [official_cell_map[cell] for cell in cell_ids], dtype=np.int64
    )
    official_gene_indices = np.asarray(
        [official_gene_map[gene] for gene in gene_ids], dtype=np.int64
    )
    base = {
        "ordering": "project-manifest-order-filtered-to-official-intersection",
        "n_cells": len(cell_ids),
        "n_genes": len(gene_ids),
        "cell_ids_sha256": hash_strings(cell_ids),
        "gene_ids_sha256": hash_strings(gene_ids),
        "project_cell_indices_sha256": hash_array(
            project_cell_indices.astype("<i8", copy=False)
        ),
        "project_gene_indices_sha256": hash_array(
            project_gene_indices.astype("<i8", copy=False)
        ),
        "official_cell_indices_sha256": hash_array(
            official_cell_indices.astype("<i8", copy=False)
        ),
        "official_gene_indices_sha256": hash_array(
            official_gene_indices.astype("<i8", copy=False)
        ),
    }
    return {
        "cell_ids": cell_ids,
        "gene_ids": gene_ids,
        "project_cell_indices": project_cell_indices,
        "project_gene_indices": project_gene_indices,
        "official_cell_indices": official_cell_indices,
        "official_gene_indices": official_gene_indices,
        "lineages": project_lineages[project_cell_indices],
        "contract": {**base, "sha256": canonical_hash(base)},
    }


def verify_audit_manifest(
    manifest_path: Path, project_root: Path, official_root: Path
) -> dict[str, Any]:
    errors: list[str] = []
    manifest = load_json(manifest_path)
    if manifest.get("schema") != AUDIT_SCHEMA:
        errors.append(f"unexpected audit schema: {manifest.get('schema')!r}")
    if manifest.get("scientific_result") is not False:
        errors.append("audit must be marked scientific_result=false")
    head = git(official_root, "rev-parse", "HEAD")
    if head != OFFICIAL_COMMIT:
        errors.append(f"official HEAD {head} != {OFFICIAL_COMMIT}")
    dirty = bool(git(official_root, "status", "--porcelain", "--untracked-files=no"))
    source = manifest.get("official_source")
    if not isinstance(source, dict) or source.get("commit") != OFFICIAL_COMMIT:
        errors.append("audit does not bind the fixed official commit")
        source = {}
    if source.get("observed_head") != head:
        errors.append("audit observed-head record does not match checkout")
    if source.get("tracked_worktree_dirty") is not dirty or dirty:
        errors.append("official tracked worktree is dirty or audit status is incorrect")
    blobs = source.get("blobs", {}) if isinstance(source, dict) else {}
    for relative, expected_blob in CRITICAL_BLOBS.items():
        try:
            observed = git(official_root, "rev-parse", f"{OFFICIAL_COMMIT}:{relative}")
        except VerificationError as exc:
            errors.append(str(exc))
            continue
        if observed != expected_blob:
            errors.append(f"critical official blob mismatch: {relative}")
        record = blobs.get(relative) if isinstance(blobs, dict) else None
        if not isinstance(record, dict) or record.get("git_blob") != expected_blob:
            errors.append(f"audit blob record is absent/incorrect: {relative}")
        checkout = record.get("checkout") if isinstance(record, dict) else None
        if isinstance(checkout, dict):
            path = official_root / relative
            if not path.is_file() or sha256_file(path) != checkout.get("sha256"):
                errors.append(f"checkout bytes changed after audit: {relative}")

    root_names = set(
        git(official_root, "ls-tree", "--name-only", OFFICIAL_COMMIT).splitlines()
    )
    has_license = any(
        name.upper().startswith(("LICENSE", "COPYING", "NOTICE")) for name in root_names
    )
    readme = git(official_root, "show", f"{OFFICIAL_COMMIT}:README.md")
    license_record = source.get("license", {}) if isinstance(source, dict) else {}
    if not isinstance(license_record, dict):
        license_record = {}
    status = license_record.get("status")
    if has_license or status != "ambiguous_missing_license_text":
        errors.append(
            "license evidence does not match fixed tree without a license file"
        )
    if license_record.get("repository_license_files") != []:
        errors.append("audit incorrectly records a repository license file")
    if license_record.get("readme_mit_badge") is not ("License-MIT" in readme):
        errors.append("README MIT badge evidence mismatch")
    if license_record.get("readme_mit_statement") is not (
        "licensed under the MIT License" in readme
    ):
        errors.append("README MIT statement evidence mismatch")
    if license_record.get("local_execution_status") != (LOCAL_EXECUTION_LICENSE_STATUS):
        errors.append("local execution license-evidence status is incorrect")
    if license_record.get("redistribution_status") != REDISTRIBUTION_LICENSE_STATUS:
        errors.append("redistribution license status is incorrect")

    try:
        assets = observed_official_assets(official_root)
    except (OSError, ValueError, VerificationError) as exc:
        errors.append(f"cannot independently inspect official assets: {exc}")
        assets = None
    if assets is not None and manifest.get("official_assets") != assets:
        errors.append("official asset inventory does not reproduce")

    protocol = observed_protocol(official_root)
    reported_protocol = manifest.get("official_protocol", {})
    for section, expected in protocol.items():
        if (
            not isinstance(reported_protocol, dict)
            or reported_protocol.get(section) != expected
        ):
            errors.append(f"official protocol evidence mismatch: {section}")

    cohort = common_cohort(project_root, official_root)
    observed_contract = manifest.get("project_intersection", {}).get("cohort_contract")
    if observed_contract != cohort["contract"]:
        errors.append("common cohort contract does not reproduce")
    if (len(cohort["cell_ids"]), len(cohort["gene_ids"])) != (1_088, 2_341):
        errors.append("current exact intersection is not 1088 cells x 2341 genes")
    project_cells = read_lines(project_root / "data/processed/cell_lines.txt")
    project_genes = read_lines(project_root / "data/processed/genes.txt")
    official_cells, _ = ordered_mapping(official_root / "datasets/samples2ID.json")
    official_genes, _ = ordered_mapping(official_root / "datasets/genes2ID_2k.json")
    expected_intersection = {
        "project_cells": len(project_cells),
        "project_genes": len(project_genes),
        "official_cells": len(official_cells),
        "official_genes": len(official_genes),
        "common_cells": len(cohort["cell_ids"]),
        "common_genes": len(cohort["gene_ids"]),
        "official_cells_missing_from_project": sorted(
            set(official_cells) - set(project_cells)
        ),
        "official_genes_missing_from_project": sorted(
            set(official_genes) - set(project_genes)
        ),
        "cohort_contract": cohort["contract"],
    }
    if manifest.get("project_intersection") != expected_intersection:
        errors.append("project/official intersection inventory does not reproduce")
    if manifest.get("expected_runtime_from_fixed_environment_yml") != EXPECTED_RUNTIME:
        errors.append("fixed upstream runtime declaration mismatch")
    runtime = manifest.get("runtime")
    errors.extend(verify_runtime_record(runtime, record_name="audit"))
    comparability = manifest.get("comparability", {})
    official_status = (
        comparability.get("official_reproduction", {}).get("status")
        if isinstance(comparability, dict)
        else None
    )
    corrected = (
        comparability.get("corrected_adaptation", {})
        if isinstance(comparability, dict)
        else {}
    )
    if official_status != "not_eligible_for_strict_corrected_comparison":
        errors.append("official reproduction eligibility is misstated")
    if corrected.get("label") != "GATDep-corrected-adaptation":
        errors.append("corrected adaptation label is missing or incorrect")
    if corrected.get("feature_method") != FEATURE_METHOD:
        errors.append("corrected adaptation feature method is missing or incorrect")
    blockers = corrected.get("current_preflight_blockers")
    expected_blockers = (
        runtime_mismatches(runtime) if isinstance(runtime, Mapping) else []
    )
    if license_record.get("local_execution_status") != (LOCAL_EXECUTION_LICENSE_STATUS):
        expected_blockers.append(LOCAL_EXECUTION_LICENSE_BLOCKER)
    if blockers != expected_blockers:
        errors.append("corrected adaptation preflight blockers do not reproduce")
    if isinstance(blockers, list) and REDISTRIBUTION_LICENSE_BLOCKER in blockers:
        errors.append(
            "redistribution status is incorrectly used as an execution blocker"
        )
    if corrected.get("local_execution_license_gate") != (
        LOCAL_EXECUTION_LICENSE_STATUS
    ):
        errors.append("corrected adaptation lacks the local execution gate")
    if corrected.get("redistribution_license_gate") != (REDISTRIBUTION_LICENSE_STATUS):
        errors.append("corrected adaptation lacks the redistribution gate")
    if corrected.get("redistribution_blocker") != REDISTRIBUTION_LICENSE_BLOCKER:
        errors.append("corrected adaptation redistribution blocker is incorrect")
    if corrected.get("redistribution_gate_overridable") is not False:
        errors.append("audit incorrectly permits overriding the redistribution gate")
    if manifest.get("audit_errors"):
        errors.append(f"audit recorded errors: {manifest.get('audit_errors')!r}")
    if manifest.get("audit_passed") is not True:
        errors.append("audit manifest is not marked passed")
    return {
        "schema": "GATDep-official-audit-verification-v1",
        "valid": not errors,
        "errors": errors,
        "manifest_sha256": sha256_file(manifest_path),
        "official_commit": head,
        "cohort_contract_sha256": cohort["contract"]["sha256"],
    }


def dependency_metrics(
    prediction: np.ndarray, target: np.ndarray, reference: np.ndarray
) -> dict[str, Any]:
    predicted = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    baseline = np.asarray(reference, dtype=np.float64)
    valid = np.isfinite(truth)
    if predicted.shape != truth.shape or predicted.ndim != 2:
        raise VerificationError("prediction and target shapes differ")
    if baseline.shape != (truth.shape[1],):
        raise VerificationError("train_gene_mean shape differs")
    if np.any(valid & ~np.isfinite(predicted)):
        raise VerificationError("non-finite prediction on evaluable target")
    error = np.where(valid, predicted - truth, 0.0)
    base_error = np.where(valid, baseline[None, :] - truth, 0.0)
    count = int(valid.sum())
    if count == 0:
        raise VerificationError("no finite targets")
    sse = float(np.square(error).sum())
    base_sse = float(np.square(base_error).sum())
    x = predicted.ravel()[valid.ravel()]
    y = truth.ravel()[valid.ravel()]
    pearson = None
    if x.size > 1 and float(x.std()) > 0 and float(y.std()) > 0:
        pearson = float(np.corrcoef(x, y)[0, 1])
    return {
        "n_cells": int(truth.shape[0]),
        "n_genes": int(truth.shape[1]),
        "n_evaluable_pairs": count,
        "mse": sse / count,
        "mae": float(np.abs(error).sum()) / count,
        "gene_mean_baseline_mse": base_sse / count,
        "residual_r2_vs_train_gene_mean": (
            1.0 - sse / base_sse if base_sse > 0 else None
        ),
        "flattened_pearson": pearson,
    }


def _equal_number(left: Any, right: Any, tolerance: float = 1e-9) -> bool:
    if left is None or right is None:
        return left is None and right is None
    try:
        return bool(
            np.isclose(float(left), float(right), rtol=tolerance, atol=tolerance)
        )
    except (TypeError, ValueError):
        return False


def expected_plateau_learning_rates(
    history: Sequence[Mapping[str, Any]],
) -> list[tuple[float, float]]:
    """Replay the pinned ReduceLROnPlateau defaults without importing PyTorch."""
    learning_rate = 1e-3
    best = np.inf
    bad_epochs = 0
    trace: list[tuple[float, float]] = []
    for item in history:
        validation = item.get("validation")
        if not isinstance(validation, Mapping):
            raise VerificationError("tuning history lacks validation metrics")
        try:
            mse = float(validation["mse"])
        except (KeyError, TypeError, ValueError) as exc:
            raise VerificationError(
                "tuning history has an invalid validation MSE"
            ) from exc
        if not np.isfinite(mse):
            raise VerificationError("tuning history has a non-finite validation MSE")
        used = learning_rate
        if mse < best * (1.0 - 1e-4):
            best = mse
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs > 3:
            candidate = learning_rate * 0.5
            if learning_rate - candidate > 1e-8:
                learning_rate = candidate
            bad_epochs = 0
        trace.append((used, learning_rate))
    return trace


def verify_prediction_bundle(
    path: Path,
    index: Mapping[str, Any],
    *,
    outer_test: np.ndarray,
    outer_train: np.ndarray,
    cohort: Mapping[str, Any],
    target_source: np.ndarray,
) -> tuple[list[str], dict[str, Any] | None]:
    errors: list[str] = []
    if index.get("schema") != PREDICTION_SCHEMA:
        errors.append("prediction index has wrong schema")
    if index.get("test_evaluation_calls") != 1:
        errors.append("prediction index does not record exactly one test evaluation")
    if not path.is_file():
        return [*errors, f"prediction file is missing: {path}"], None
    if index.get("sha256") != sha256_file(path):
        errors.append("prediction NPZ hash mismatch")
    try:
        with np.load(path, allow_pickle=False) as bundle:
            if set(bundle.files) != REQUIRED_ARRAYS:
                errors.append(
                    f"prediction arrays {sorted(bundle.files)!r} != required set"
                )
                return errors, None
            arrays = {name: bundle[name] for name in bundle.files}
    except (OSError, ValueError) as exc:
        return [*errors, f"cannot load prediction NPZ: {exc}"], None

    n_test = int(outer_test.size)
    n_genes = len(cohort["gene_ids"])
    expected_metadata = {
        name: {"shape": list(array.shape), "dtype": str(array.dtype)}
        for name, array in arrays.items()
    }
    if index.get("arrays") != expected_metadata:
        errors.append("prediction array metadata mismatch")
    if arrays["prediction"].shape != (n_test, n_genes):
        errors.append("prediction shape is inconsistent with split/cohort")
    if arrays["target_raw"].shape != (n_test, n_genes):
        errors.append("target shape is inconsistent with split/cohort")
    if arrays["train_gene_mean"].shape != (n_genes,):
        errors.append("train_gene_mean shape is inconsistent with cohort")
    identities = {
        "outer_test_indices_common": outer_test,
        "source_cell_indices_project": cohort["project_cell_indices"][outer_test],
        "source_gene_indices_project": cohort["project_gene_indices"],
        "official_cell_indices": cohort["official_cell_indices"][outer_test],
        "official_gene_indices": cohort["official_gene_indices"],
        "cell_ids": np.asarray(
            [cohort["cell_ids"][int(index)] for index in outer_test], dtype=str
        ),
        "gene_ids": np.asarray(cohort["gene_ids"], dtype=str),
    }
    for name, expected in identities.items():
        if not np.array_equal(arrays[name], expected):
            errors.append(f"identity array mismatch: {name}")

    expected_target = np.asarray(
        target_source[
            np.ix_(
                cohort["project_gene_indices"],
                cohort["project_cell_indices"][outer_test],
            )
        ].T,
        dtype=np.float32,
    )
    if not np.array_equal(arrays["target_raw"], expected_target, equal_nan=True):
        errors.append("bound target values differ from current gene_effect.npy")
    outer_train_project = cohort["project_cell_indices"][outer_train]
    train_target = np.asarray(
        target_source[np.ix_(cohort["project_gene_indices"], outer_train_project)].T,
        dtype=np.float32,
    )
    with np.errstate(invalid="ignore"):
        expected_reference = np.nanmean(train_target, axis=0).astype(np.float32)
    if not np.array_equal(
        arrays["train_gene_mean"], expected_reference, equal_nan=True
    ):
        errors.append("train_gene_mean was not fit on the bound outer-train cells")
    if errors:
        return errors, None
    return [], dependency_metrics(
        arrays["prediction"], arrays["target_raw"], arrays["train_gene_mean"]
    )


def comparison_contract_status(
    paths: Sequence[Path], cohort_hash: str, split_hash: str, seed: int
) -> tuple[list[str], list[str]]:
    def is_sha256(value: Any) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    def is_nonnegative_finite(value: Any) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and bool(np.isfinite(value))
            and value >= 0
        )

    errors: list[str] = []
    observed_conditions: set[str] = set()
    verified_conditions: set[str] = set()
    for path in paths:
        report_errors: list[str] = []
        try:
            value = load_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"comparison report cannot be read: {path}: {exc}")
            continue
        if value.get("schema") != COMPARISON_VERIFICATION_SCHEMA:
            report_errors.append(
                f"comparison is not an independent verification report: {path}"
            )
        if value.get("scientific_result") is not False:
            report_errors.append(
                f"comparison report self-declares a scientific result: {path}"
            )
        if value.get("valid") is not True or value.get("errors") != []:
            report_errors.append(
                f"comparison report is not valid and error-free: {path}"
            )
        condition = value.get("condition")
        if condition not in SUPPORTED_COMPARISON_CONDITIONS:
            report_errors.append(f"comparison condition is unsupported: {path}")
        elif condition in observed_conditions:
            report_errors.append(f"duplicate comparison condition {condition}: {path}")
        else:
            observed_conditions.add(condition)
        if value.get("seed") != int(seed):
            report_errors.append(f"comparison seed mismatch: {path}")
        if value.get("completed_folds") != list(range(5)):
            report_errors.append(f"comparison does not verify all five folds: {path}")
        if value.get("outer_test_exactly_once") is not True:
            report_errors.append(f"comparison outer-test coverage is not exact: {path}")
        if value.get("predictions_exact_once_and_hash_bound") is not True:
            report_errors.append(
                f"comparison predictions are not verified and hash-bound: {path}"
            )
        if value.get("checkpoint_replay_all_folds") is not True:
            report_errors.append(
                f"comparison does not verify checkpoint replay for all folds: {path}"
            )
        if value.get("replayed_folds") != list(range(5)):
            report_errors.append(
                f"comparison replayed-fold coverage is incomplete: {path}"
            )
        if value.get("elementwise_tolerance") != COMPARISON_REPLAY_TOLERANCE:
            report_errors.append(
                f"comparison replay tolerance differs from the fixed contract: {path}"
            )
        if value.get("replay_inputs_embedded") is not False:
            report_errors.append(
                f"comparison permits embedded cell-level replay inputs: {path}"
            )
        if value.get("replay_input_policy") != COMPARISON_REPLAY_INPUT_POLICY:
            report_errors.append(f"comparison replay-input policy differs: {path}")
        source_inventory_hash = value.get("source_input_inventory_sha256")
        if not is_sha256(source_inventory_hash):
            report_errors.append(
                f"comparison source-input inventory hash is invalid: {path}"
            )
        if (
            value.get("checkpoint_preprocessing_contains_raw_cell_level_values")
            is not False
        ):
            report_errors.append(
                f"comparison permits raw cell-level preprocessing values: {path}"
            )
        if (
            value.get("checkpoint_preprocessing_contains_source_derived_statistics")
            is not True
        ):
            report_errors.append(
                f"comparison omits source-derived preprocessing disclosure: {path}"
            )
        if (
            value.get("checkpoint_preprocessing_content_policy")
            != COMPARISON_PREPROCESSING_CONTENT_POLICY
        ):
            report_errors.append(
                f"comparison preprocessing content policy differs: {path}"
            )
        if (
            value.get("generated_artifact_redistribution_status")
            != COMPARISON_ARTIFACT_REDISTRIBUTION_STATUS
        ):
            report_errors.append(
                f"comparison generated-artifact redistribution status differs: {path}"
            )
        fold_replay = value.get("fold_replay")
        expected_fold_names = {f"fold-{fold:02d}" for fold in range(5)}
        if (
            not isinstance(fold_replay, Mapping)
            or set(fold_replay) != expected_fold_names
        ):
            report_errors.append(
                f"comparison fold replay records are incomplete: {path}"
            )
        else:
            for fold_name in sorted(expected_fold_names):
                replay = fold_replay[fold_name]
                if not isinstance(replay, Mapping):
                    report_errors.append(
                        f"comparison {fold_name} replay record is invalid: {path}"
                    )
                    continue
                if replay.get("fold") != fold_name:
                    report_errors.append(
                        f"comparison {fold_name} replay identity differs: {path}"
                    )
                if replay.get("allclose") is not True:
                    report_errors.append(
                        f"comparison {fold_name} checkpoint replay failed: {path}"
                    )
                if replay.get("elementwise_tolerance") != COMPARISON_REPLAY_TOLERANCE:
                    report_errors.append(
                        f"comparison {fold_name} replay tolerance differs: {path}"
                    )
                if replay.get("replay_inputs_embedded") is not False:
                    report_errors.append(
                        f"comparison {fold_name} embeds replay inputs: {path}"
                    )
                if replay.get("replay_input_policy") != COMPARISON_REPLAY_INPUT_POLICY:
                    report_errors.append(
                        f"comparison {fold_name} replay-input policy differs: {path}"
                    )
                if (
                    replay.get("preprocessing_reconstructed_from_current_sources")
                    is not True
                ):
                    report_errors.append(
                        f"comparison {fold_name} preprocessing was not reconstructed: {path}"
                    )
                if replay.get("source_input_inventory_sha256") != source_inventory_hash:
                    report_errors.append(
                        f"comparison {fold_name} source-input hash differs: {path}"
                    )
                for hash_field in (
                    "checkpoint_sha256",
                    "prediction_sha256",
                    "preprocessing_state_sha256",
                    "model_state_sha256",
                    "reconstructed_inputs_sha256",
                    "source_input_inventory_sha256",
                ):
                    if not is_sha256(replay.get(hash_field)):
                        report_errors.append(
                            f"comparison {fold_name} {hash_field} is invalid: {path}"
                        )
                for error_field in ("max_abs_error", "max_rel_error"):
                    if not is_nonnegative_finite(replay.get(error_field)):
                        report_errors.append(
                            f"comparison {fold_name} {error_field} is invalid: {path}"
                        )
                n_elements = replay.get("n_elements_compared")
                if (
                    not isinstance(n_elements, int)
                    or isinstance(n_elements, bool)
                    or n_elements <= 0
                ):
                    report_errors.append(
                        f"comparison {fold_name} replay element count is invalid: {path}"
                    )
        if not is_sha256(value.get("run_manifest_sha256")):
            report_errors.append(f"comparison run manifest hash is invalid: {path}")
        if value.get("cohort_contract_sha256") != cohort_hash:
            report_errors.append(f"comparison cohort mismatch: {path}")
        if value.get("outer_split_contract_sha256") != split_hash:
            report_errors.append(f"comparison outer-split mismatch: {path}")
        errors.extend(report_errors)
        if not report_errors and condition in SUPPORTED_COMPARISON_CONDITIONS:
            verified_conditions.add(str(condition))
    return errors, sorted(verified_conditions)


def verify_comparison_contracts(
    paths: Sequence[Path], cohort_hash: str, split_hash: str, seed: int
) -> list[str]:
    errors, _ = comparison_contract_status(paths, cohort_hash, split_hash, seed)
    return errors


def verify_run(
    run_dir: Path,
    project_root: Path,
    official_root: Path,
    comparison_manifests: Sequence[Path] = (),
    allow_partial: bool = False,
    replay_device: str = "cpu",
) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = run_dir / "run-manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema") != RUN_SCHEMA:
        errors.append(f"unexpected run schema: {manifest.get('schema')!r}")
    if manifest.get("method_label") != "GATDep-corrected-adaptation":
        errors.append("run is not labeled as the corrected adaptation")
    if manifest.get("scientific_result") is not False:
        errors.append("run manifest must not self-declare a scientific result")
    if manifest.get("official_commit") != OFFICIAL_COMMIT:
        errors.append("run does not bind the pinned official commit")
    errors.extend(verify_pinned_model_source(official_root))
    if manifest.get("feature_method") != FEATURE_METHOD:
        errors.append("run feature method is not the corrected fold-local transform")
    if manifest.get("endpoint") != "raw_DepMap_gene_effect_common_2341_genes":
        errors.append("run endpoint is not the audited common raw gene-effect endpoint")
    if manifest.get("outer_test_policy") != (
        "exactly_once_after_inner_selection_and_outer_train_refit"
    ):
        errors.append("run outer-test policy is incorrect")
    execution_device = manifest.get("execution_device")
    if execution_device not in {"cpu", "cuda"}:
        errors.append("run execution device is missing or invalid")
    if manifest.get("planned_folds") != list(range(5)):
        errors.append("run does not plan all five fixed outer folds")
    root_names = set(
        git(official_root, "ls-tree", "--name-only", OFFICIAL_COMMIT).splitlines()
    )
    has_license_text = any(
        name.upper().startswith(("LICENSE", "COPYING", "NOTICE")) for name in root_names
    )
    readme = git(official_root, "show", f"{OFFICIAL_COMMIT}:README.md")
    if has_license_text:
        errors.append("pinned official tree unexpectedly contains license text")
    if "License-MIT" not in readme or "licensed under the MIT License" not in readme:
        errors.append("pinned README no longer supports the local execution basis")
    if manifest.get("license_evidence_status") != "ambiguous_missing_license_text":
        errors.append("run license evidence status is incorrect")
    if manifest.get("local_execution_license_status") != (
        LOCAL_EXECUTION_LICENSE_STATUS
    ):
        errors.append("run lacks the audited local execution basis")
    if manifest.get("redistribution_license_status") != (REDISTRIBUTION_LICENSE_STATUS):
        errors.append("run fails to retain the redistribution blocker")
    if manifest.get("redistribution_blocker") != REDISTRIBUTION_LICENSE_BLOCKER:
        errors.append("run redistribution blocker text is incorrect")
    runtime = manifest.get("runtime", {})
    errors.extend(verify_runtime_record(runtime, record_name="run"))
    packages = runtime.get("packages", {}) if isinstance(runtime, dict) else {}
    if not isinstance(packages, dict):
        packages = {}
    if (
        not isinstance(runtime, dict)
        or runtime.get("python") != EXPECTED_RUNTIME["python"]
    ):
        errors.append("run Python version differs from the pinned upstream environment")
    for name, expected_version in EXPECTED_RUNTIME.items():
        observed_version = packages.get(name)
        if name == "torch":
            distribution_parts = _parse_torch_version(observed_version)
            compared_version = (
                distribution_parts[0] if distribution_parts is not None else None
            )
        else:
            compared_version = observed_version
        if name != "python" and compared_version != expected_version:
            errors.append(f"run package version mismatch: {name}")
    torch_build = runtime.get("torch_build", {}) if isinstance(runtime, dict) else {}
    if not isinstance(torch_build, dict):
        torch_build = {}
    module_version = torch_build.get("module_version")
    module_base = torch_build.get("base_version")
    module_parts = _parse_torch_version(module_version)
    if not isinstance(module_version, str) or module_base != EXPECTED_RUNTIME["torch"]:
        errors.append("run PyTorch module build does not retain the 2.7.0 base version")
    expected_build_tag = module_parts[1] if module_parts is not None else None
    if torch_build.get("build_tag") != expected_build_tag:
        errors.append("run PyTorch build tag is internally inconsistent")
    if (
        execution_device == "cuda"
        and torch_build.get("cuda_execution_compatible") is not True
    ):
        errors.append("run claims CUDA execution on an incompatible PyTorch/GPU build")
    fidelity = runtime.get("fidelity", {}) if isinstance(runtime, dict) else {}
    if not isinstance(fidelity, dict):
        fidelity = {}
    platform_compatible = (
        runtime.get("platform_system") == "Linux"
        and module_base == EXPECTED_RUNTIME["torch"]
        and torch_build.get("cuda_runtime")
        == UPSTREAM_PLATFORM_CONTRACT["cuda_runtime_major_minor"]
    )
    expected_classification = (
        "upstream_declared_platform_compatible_not_bit_exact"
        if platform_compatible
        else "platform_adapted_local_runtime_not_upstream_bit_exact"
    )
    if fidelity.get("distribution_versions_exact") is not True:
        errors.append("run does not record exact distribution base versions")
    recorded_distribution = fidelity.get("torch_distribution_version")
    recorded_distribution_parts = _parse_torch_version(recorded_distribution)
    recorded_distribution_base = (
        recorded_distribution_parts[0]
        if recorded_distribution_parts is not None
        else None
    )
    if recorded_distribution_base != EXPECTED_RUNTIME["torch"]:
        errors.append("run fidelity record has the wrong torch distribution version")
    if fidelity.get("upstream_platform_contract") != UPSTREAM_PLATFORM_CONTRACT:
        errors.append("run fidelity record has the wrong upstream platform contract")
    if fidelity.get("upstream_platform_compatible") is not platform_compatible:
        errors.append("run platform compatibility classification is inconsistent")
    if fidelity.get("classification") != expected_classification:
        errors.append("run runtime fidelity classification is inconsistent")
    if fidelity.get("upstream_bit_exact_environment_claimed") is not False:
        errors.append("run improperly claims an upstream bit-exact environment")
    configuration = manifest.get("fixed_configuration", {})
    expected_configuration = {
        "official_model_source": "GeneDependencyGAT.py",
        "in_channels": 6_561,
        "hidden_channels": 64,
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-2,
        "loss": "Huber(delta=1.0)",
        "scheduler": SCHEDULER_CONFIGURATION,
        "inner_val_fraction": 0.2,
    }
    if not isinstance(configuration, dict):
        errors.append("run lacks a fixed configuration")
        configuration = {}
    for name, expected_value in expected_configuration.items():
        if configuration.get(name) != expected_value:
            errors.append(f"fixed configuration mismatch: {name}")
    for name in ("inner_max_epochs", "patience", "batch_size"):
        value = configuration.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            errors.append(f"fixed configuration is invalid: {name}")

    inventories = expected_run_file_inventories(project_root, official_root)
    for group_name, expected_paths in inventories.items():
        errors.extend(
            verify_fixed_file_inventory(
                manifest.get(group_name),
                expected_paths,
                group_name=group_name,
            )
        )

    cohort = common_cohort(project_root, official_root)
    if cohort["contract"] != manifest.get("cohort_contract"):
        errors.append("run cohort contract does not reproduce")
    cohort_hash = cohort["contract"]["sha256"]
    if manifest.get("cohort_contract_sha256") != cohort_hash:
        errors.append("run cohort hash mismatch")
    seed = int(manifest.get("seed", -1))
    folds = make_outer_splits(len(cohort["cell_ids"]), 5, seed)
    split_payload = {
        "generator": "numpy.default_rng-permutation-array_split-v1",
        "seed": seed,
        "n_splits": 5,
        "cohort_contract_sha256": cohort_hash,
        "outer_test_indices": [fold.tolist() for fold in folds],
        "outer_test_cell_ids": [
            [cohort["cell_ids"][int(index)] for index in fold] for fold in folds
        ],
    }
    split_hash = canonical_hash(split_payload)
    if manifest.get("outer_split_contract_sha256") != split_hash:
        errors.append("run outer-split contract does not reproduce")
    if manifest.get("outer_split_contract") != {**split_payload, "sha256": split_hash}:
        errors.append("embedded outer-split contract does not reproduce")
    comparison_errors, comparison_conditions = comparison_contract_status(
        comparison_manifests, cohort_hash, split_hash, seed
    )
    errors.extend(comparison_errors)
    external_comparison_complete = (
        not comparison_errors
        and set(comparison_conditions) == SUPPORTED_COMPARISON_CONDITIONS
    )

    target_source = np.load(
        project_root / "data/processed/gene_effect.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    expression, support_membership = load_expression_support(
        project_root, official_root, cohort
    )
    completed: list[int] = []
    coverage: list[int] = []
    verified_metrics: dict[str, Any] = {}
    checkpoint_replay: dict[str, Any] = {}
    for fold_number, outer_test in enumerate(folds):
        prefix = run_dir / f"fold-{fold_number:02d}"
        paths = {
            "split": prefix.with_name(prefix.name + "-split.json"),
            "pretest": prefix.with_name(prefix.name + "-pretest.json"),
            "model": prefix.with_name(prefix.name + "-model.pt"),
            "intent": prefix.with_name(prefix.name + "-test-intent.json"),
            "prediction": prefix.with_name(prefix.name + "-predictions.npz"),
            "index": prefix.with_name(prefix.name + "-predictions.index.json"),
            "result": prefix.with_name(prefix.name + "-result.json"),
        }
        if not paths["result"].exists():
            if not allow_partial:
                errors.append(f"fold {fold_number} is incomplete")
            continue
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            errors.append(f"fold {fold_number} missing artifacts: {missing!r}")
            continue
        split = load_json(paths["split"])
        expected_split = expected_split_record(
            cohort,
            cohort["lineages"],
            outer_test,
            experiment_seed=seed,
            fold_number=fold_number,
        )
        if split != expected_split:
            errors.append(f"fold {fold_number} nested split does not reproduce")

        pretest = load_json(paths["pretest"])
        intent = load_json(paths["intent"])
        index = load_json(paths["index"])
        result = load_json(paths["result"])
        outer_train = np.asarray(
            expected_split["outer_train"]["indices"], dtype=np.int64
        )
        inner_train = np.asarray(
            expected_split["inner_train"]["indices"], dtype=np.int64
        )
        if pretest.get("outer_test_accessed") is not False:
            errors.append(f"fold {fold_number} pretest record is not sealed")
        if pretest.get("schema") != "GATDep-corrected-pretest-v1":
            errors.append(f"fold {fold_number} pretest schema mismatch")
        if pretest.get("fold_number") != fold_number:
            errors.append(f"fold {fold_number} pretest identity mismatch")
        if pretest.get("split_sha256") != expected_split["split_sha256"]:
            errors.append(f"fold {fold_number} pretest/split hash mismatch")
        fold_seed = derive_seed(seed, f"fold-{fold_number:02d}")
        expected_seeds = {
            "fold": fold_seed,
            "inner_split": derive_seed(fold_seed, "inner-split"),
            "tune": derive_seed(fold_seed, "tune"),
            "refit": derive_seed(fold_seed, "refit"),
        }
        if pretest.get("seeds") != expected_seeds:
            errors.append(f"fold {fold_number} deterministic seed record mismatch")
        preprocessing_scopes = {
            "inner_preprocessing": inner_train,
            "outer_train_preprocessing": outer_train,
        }
        for section, fit_indices in preprocessing_scopes.items():
            provenance = pretest.get(section, {})
            if not isinstance(provenance, dict):
                errors.append(f"fold {fold_number} lacks {section}")
                continue
            expected_provenance = expected_preprocessing_provenance(
                expression, fit_indices, support_membership
            )
            if provenance != expected_provenance:
                errors.append(f"fold {fold_number} {section} does not reproduce")

        selection = pretest.get("selection", {})
        tuning_history = pretest.get("tuning_history", [])
        refit_history = pretest.get("refit_history", [])
        if not isinstance(selection, dict) or not isinstance(tuning_history, list):
            errors.append(f"fold {fold_number} selection record is invalid")
            selection = {}
            tuning_history = []
        if selection.get("metric") != "residual_r2_vs_train_gene_mean":
            errors.append(f"fold {fold_number} selection metric mismatch")
        if selection.get("maximize") is not True:
            errors.append(f"fold {fold_number} selection direction mismatch")
        selection_configuration = selection.get("fixed_configuration", {})
        if not isinstance(selection_configuration, dict):
            selection_configuration = {}
        for name in ("optimizer", "learning_rate", "weight_decay", "loss"):
            if selection_configuration.get(name) != configuration.get(name):
                errors.append(f"fold {fold_number} selection config mismatch: {name}")
        if selection_configuration.get("inner_max_epochs") != configuration.get(
            "inner_max_epochs"
        ):
            errors.append(f"fold {fold_number} selection max-epoch mismatch")
        if selection_configuration.get("patience") != configuration.get("patience"):
            errors.append(f"fold {fold_number} selection patience mismatch")
        best_epoch = selection.get("best_epoch")
        if (
            not isinstance(best_epoch, int)
            or isinstance(best_epoch, bool)
            or best_epoch < 1
        ):
            errors.append(f"fold {fold_number} selected epoch is invalid")
        expected_epochs = list(range(1, len(tuning_history) + 1))
        if [
            item.get("epoch") for item in tuning_history if isinstance(item, dict)
        ] != expected_epochs:
            errors.append(f"fold {fold_number} tuning epochs are not contiguous")
        if len(tuning_history) > int(configuration.get("inner_max_epochs") or 0):
            errors.append(f"fold {fold_number} tuning exceeds fixed epoch budget")
        try:
            expected_learning_rates = expected_plateau_learning_rates(tuning_history)
        except VerificationError as exc:
            errors.append(f"fold {fold_number} scheduler evidence is invalid: {exc}")
            expected_learning_rates = []
        for epoch_index, (used, post) in enumerate(expected_learning_rates):
            item = tuning_history[epoch_index]
            if not _equal_number(item.get("learning_rate_used"), used):
                errors.append(
                    f"fold {fold_number} tuning learning rate mismatch at epoch "
                    f"{epoch_index + 1}"
                )
            if not _equal_number(item.get("post_scheduler_learning_rate"), post):
                errors.append(
                    f"fold {fold_number} post-scheduler learning rate mismatch at "
                    f"epoch {epoch_index + 1}"
                )
        reproduced_best_epoch: int | None = None
        reproduced_best_score = -np.inf
        for item in tuning_history:
            if not isinstance(item, dict) or not isinstance(
                item.get("validation"), dict
            ):
                continue
            score = item["validation"].get("residual_r2_vs_train_gene_mean")
            if score is None or not np.isfinite(float(score)):
                continue
            numeric_score = float(score)
            if numeric_score > reproduced_best_score + 1e-12:
                reproduced_best_score = numeric_score
                reproduced_best_epoch = int(item.get("epoch", -1))
        if reproduced_best_epoch is None:
            errors.append(f"fold {fold_number} has no finite inner-validation score")
        elif best_epoch != reproduced_best_epoch or not _equal_number(
            selection.get("score"), reproduced_best_score
        ):
            errors.append(f"fold {fold_number} selected epoch/score mismatch")
        selected_learning_rates = selection.get("refit_learning_rates")
        if isinstance(best_epoch, int) and not isinstance(best_epoch, bool):
            expected_refit_rates = [
                used for used, _ in expected_learning_rates[:best_epoch]
            ]
        else:
            expected_refit_rates = []
        if (
            not isinstance(selected_learning_rates, list)
            or len(selected_learning_rates) != len(expected_refit_rates)
            or any(
                not _equal_number(observed, expected)
                for observed, expected in zip(
                    selected_learning_rates or [], expected_refit_rates, strict=False
                )
            )
        ):
            errors.append(f"fold {fold_number} frozen refit learning rates mismatch")
        if not isinstance(refit_history, list) or len(refit_history) != best_epoch:
            errors.append(f"fold {fold_number} refit epoch count mismatch")
        elif [
            item.get("epoch") for item in refit_history if isinstance(item, dict)
        ] != list(range(1, len(refit_history) + 1)):
            errors.append(f"fold {fold_number} refit epochs are not contiguous")
        elif any(
            not isinstance(item, dict)
            or not _equal_number(item.get("learning_rate_used"), expected)
            for item, expected in zip(refit_history, expected_refit_rates, strict=False)
        ):
            errors.append(f"fold {fold_number} refit learning-rate replay mismatch")

        model_record = pretest.get("model_checkpoint", {})
        if not isinstance(model_record, dict):
            model_record = {}
        if (
            model_record.get("sha256") != sha256_file(paths["model"])
            or model_record.get("bytes") != paths["model"].stat().st_size
        ):
            errors.append(f"fold {fold_number} pretest/model record mismatch")
        if intent.get("schema") != "GATDep-corrected-test-intent-v1":
            errors.append(f"fold {fold_number} test-intent schema mismatch")
        if intent.get("fold_number") != fold_number:
            errors.append(f"fold {fold_number} test-intent identity mismatch")
        if intent.get("policy") != "one_intent_allows_one_outer_test_evaluation":
            errors.append(f"fold {fold_number} test-intent policy mismatch")
        if not isinstance(intent.get("nonce"), str) or not intent.get("nonce"):
            errors.append(f"fold {fold_number} test-intent nonce is missing")
        if intent.get("pretest_sha256") != sha256_file(paths["pretest"]):
            errors.append(f"fold {fold_number} test intent/pretest hash mismatch")
        if intent.get("model_checkpoint_sha256") != sha256_file(paths["model"]):
            errors.append(f"fold {fold_number} test intent/model hash mismatch")
        if index.get("test_intent_sha256") != sha256_file(paths["intent"]):
            errors.append(f"fold {fold_number} index/test-intent hash mismatch")
        if index.get("split_sha256") != expected_split["split_sha256"]:
            errors.append(f"fold {fold_number} prediction-index/split hash mismatch")
        if index.get("cohort_contract_sha256") != cohort_hash:
            errors.append(f"fold {fold_number} prediction-index/cohort hash mismatch")
        bundle_errors, metrics = verify_prediction_bundle(
            paths["prediction"],
            index,
            outer_test=outer_test,
            outer_train=outer_train,
            cohort=cohort,
            target_source=target_source,
        )
        errors.extend(f"fold {fold_number}: {error}" for error in bundle_errors)
        if metrics is not None:
            try:
                replayed = replay_checkpoint_predictions(
                    checkpoint_path=paths["model"],
                    official_root=official_root,
                    cohort=cohort,
                    expression=expression,
                    support_membership=support_membership,
                    outer_train=outer_train,
                    outer_test=outer_test,
                    device=replay_device,
                    batch_size=int(configuration.get("batch_size") or 0),
                )
                with np.load(paths["prediction"], allow_pickle=False) as bundle:
                    replay_record = compare_replayed_predictions(
                        bundle["prediction"], replayed
                    )
                replay_record["device"] = replay_device
                checkpoint_replay[f"fold-{fold_number:02d}"] = replay_record
                if not replay_record["matched"]:
                    errors.append(
                        f"fold {fold_number} stored predictions do not match "
                        "checkpoint replay"
                    )
            except (
                ImportError,
                OSError,
                RuntimeError,
                ValueError,
                VerificationError,
            ) as exc:
                checkpoint_replay[f"fold-{fold_number:02d}"] = {
                    "matched": False,
                    "device": replay_device,
                    "error": str(exc),
                }
                errors.append(f"fold {fold_number} checkpoint replay failed: {exc}")
        if result.get("schema") != "GATDep-corrected-fold-result-v1":
            errors.append(f"fold {fold_number} result schema mismatch")
        if result.get("fold_number") != fold_number:
            errors.append(f"fold {fold_number} result identity mismatch")
        if result.get("selection") != selection:
            errors.append(f"fold {fold_number} result/pretest selection mismatch")
        if result.get("split_sha256") != expected_split["split_sha256"]:
            errors.append(f"fold {fold_number} result/split hash mismatch")
        if result.get("cohort_contract_sha256") != cohort_hash:
            errors.append(f"fold {fold_number} result/cohort hash mismatch")
        result_records = {
            "prediction_index": paths["index"],
            "prediction_npz": paths["prediction"],
        }
        for record_name, artifact_path in result_records.items():
            record = result.get(record_name, {})
            if not isinstance(record, dict):
                record = {}
            if (
                record.get("sha256") != sha256_file(artifact_path)
                or record.get("bytes") != artifact_path.stat().st_size
            ):
                errors.append(
                    f"fold {fold_number} result/{record_name} hash record mismatch"
                )
        if result.get("test_evaluation_calls") != 1:
            errors.append(f"fold {fold_number} result lacks exact-once evidence")
        if result.get("scientific_result") is not False:
            errors.append(f"fold {fold_number} result overclaims scientific status")
        if metrics is not None:
            reported = result.get("test_metrics", {})
            for name, expected in metrics.items():
                if not _equal_number(reported.get(name), expected):
                    errors.append(f"fold {fold_number} metric mismatch: {name}")
            verified_metrics[f"fold-{fold_number:02d}"] = metrics
        completed.append(fold_number)
        coverage.extend(outer_test.tolist())

    if completed == list(range(5)):
        counts = Counter(coverage)
        if counts != Counter({index: 1 for index in range(len(cohort["cell_ids"]))}):
            errors.append(
                "outer-test predictions do not cover each common cell exactly once"
            )
        completion_path = run_dir / "completion.json"
        if not completion_path.is_file():
            errors.append("complete run lacks completion.json")
        else:
            completion = load_json(completion_path)
            if completion.get("schema") != "GATDep-corrected-completion-v1":
                errors.append("completion record schema mismatch")
            if completion.get("completed_folds") != list(range(5)):
                errors.append("completion record fold list mismatch")
            if completion.get("outer_test_cells_exactly_once") is not True:
                errors.append("completion record lacks exact-once coverage")
            if (
                completion.get("eligible_for_comparison_after_independent_verification")
                is not True
            ):
                errors.append("completion record lacks independent-verification gate")
            if completion.get("scientific_result") is not False:
                errors.append("completion record overclaims scientific status")
    elif not allow_partial:
        errors.append(f"completed folds {completed!r} != [0, 1, 2, 3, 4]")

    return {
        "schema": "GATDep-corrected-run-verification-v1",
        "valid": not errors,
        "errors": errors,
        "run_manifest_sha256": sha256_file(manifest_path),
        "cohort_contract_sha256": cohort_hash,
        "outer_split_contract_sha256": split_hash,
        "completed_folds": completed,
        "outer_test_exactly_once": completed == list(range(5)) and not errors,
        "verified_metrics": verified_metrics,
        "checkpoint_replay": checkpoint_replay,
        "comparison_conditions_verified": comparison_conditions,
        "external_comparison_contract_complete": external_comparison_complete,
        "scientific_result": False,
        "interpretation": (
            "A valid report establishes artifact/protocol integrity only. Scientific "
            "comparison additionally requires all named models to share both contract hashes."
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", default=str(Path(__file__).resolve().parent.parent)
    )
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--audit-manifest")
    parser.add_argument("--run-dir")
    parser.add_argument("--comparison-manifest", action="append", default=[])
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--replay-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if bool(args.audit_manifest) == bool(args.run_dir):
        parser.error("provide exactly one of --audit-manifest or --run-dir")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(args.project_root).resolve()
    official_root = Path(args.official_root).resolve()
    target = "audit_manifest" if args.audit_manifest else "corrected_run"
    try:
        if args.audit_manifest:
            report = verify_audit_manifest(
                Path(args.audit_manifest).resolve(), project_root, official_root
            )
        else:
            report = verify_run(
                Path(args.run_dir).resolve(),
                project_root,
                official_root,
                [Path(path).resolve() for path in args.comparison_manifest],
                allow_partial=bool(args.allow_partial),
                replay_device=args.replay_device,
            )
    except Exception as exc:
        # Preserve a structured fail-closed artifact even when an input cannot
        # be decoded far enough for the detailed verifier to collect errors.
        report = {
            "schema": "GATDep-independent-verification-failure-v1",
            "valid": False,
            "scientific_result": False,
            "verification_complete": False,
            "verification_target": target,
            "errors": [
                "verification aborted while parsing or reconstructing artifacts: "
                f"{type(exc).__name__}: {exc}"
            ],
        }
    rendered = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True)
    if args.output:
        write_json_exclusive(args.output, report)
    print(rendered)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as exc:
        print(json.dumps({"valid": False, "error": str(exc)}), file=sys.stderr)
        raise SystemExit(2) from exc
