"""Audit prediction error by protein length under chunked ESM embeddings.

The revised primary model represents every residue with overlapping ESM windows.
This analysis therefore tests whether long proteins still have different outer-test
errors after that mitigation.  It does not estimate a causal benefit of chunking
over the legacy first-1,022-residue policy because that would require a separately
trained, otherwise identical comparator.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

from aggregate_corrected_campaign import (
    RunValidationError as CampaignValidationError,
    validate_campaign as validate_authoritative_campaign,
)
from embedding_provenance import (
    EMBEDDING_RELATIVE_PATH,
    EmbeddingProvenanceError,
    MANIFEST_RELATIVE_PATH,
    validate_embedding_bundle,
)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "long_protein_analysis_v3"
DEFAULT_EMBEDDING_MANIFEST = PROCESSED / "esm2_t6_8M_residue_chunked.manifest.json"
DEFAULT_FASTA = PROCESSED / "gene_seqs.fa"
DEFAULT_GENE_EFFECT = PROCESSED / "gene_effect.npy"
DEFAULT_GENES = PROCESSED / "genes.txt"
DEFAULT_CELLS = PROCESSED / "cell_lines.txt"

CAMPAIGN_SCHEMA = "DepPLM-corrected-v2.2-campaign"
OUTPUT_SCHEMA = "DepPLM-long-protein-analysis-v3"
RUN_SCHEMA = "DepPLM-long-protein-analysis-run-v3"
COMPLETION_SCHEMA = "DepPLM-long-protein-analysis-completion-v3"
SELF_VERIFICATION_SCHEMA = "DepPLM-long-protein-analysis-verification-v3"
PREDICTION_SCHEMA = "DepPLM-outer-test-predictions-v1"
CONDITIONS = ("DepPLM", "C-PLM-MLP", "C-GCN")
SEEDS = (42, 43, 44)
FOLDS = tuple(f"fold-{index:02d}" for index in range(5))
LONG_THRESHOLD = 1022
EXPECTED_CAMPAIGN_RUNS = 99
EXPECTED_FOLD_ARTIFACTS_BY_PROTOCOL = {
    "p1": 15,
    "transfer-forward": 3,
    "transfer-reverse": 3,
}
EXPECTED_FOLD_ARTIFACTS = 693
EXPECTED_N_GENES = 17_768
EXPECTED_N_LONG = 2_152
EXPECTED_PREDICTION_ARRAYS = (
    "prediction",
    "train_gene_mean",
    "outer_test_indices",
    "source_cell_indices",
    "source_gene_indices",
    "cell_ids",
    "gene_ids",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
RUN_ROOT_JSON_FILES = (
    "completion.json",
    "summary.json",
    "run-manifest.json",
    "cohort-manifest.json",
)
ANALYSIS_RUN_FILES = frozenset(
    {
        "completion.json",
        "run-manifest.json",
        "summary.json",
        "per-gene-errors.npz",
    }
)
CANONICAL_ANALYSIS_INPUTS = {
    "embedding_array": EMBEDDING_RELATIVE_PATH,
    "embedding_manifest": MANIFEST_RELATIVE_PATH,
    "fasta": "data/processed/gene_seqs.fa",
    "genes": "data/processed/genes.txt",
    "cells": "data/processed/cell_lines.txt",
    "gene_effect": "data/processed/gene_effect.npy",
}
REGISTERED_PARAMETERS = {
    "conditions": list(CONDITIONS),
    "seeds": list(SEEDS),
    "long_threshold_residues": LONG_THRESHOLD,
    "inference_scope": "conditional_descriptive_only",
}
ANALYSIS_SOURCE_PATHS = {
    "analysis_code": "code/long_protein_analysis.py",
    "campaign_validator_code": "code/aggregate_corrected_campaign.py",
    "embedding_validator_code": "code/embedding_provenance.py",
}
ANALYSIS_ENVIRONMENT_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "torch",
)
ANALYSIS_MANIFEST_FIELDS = {
    "schema",
    "created_utc",
    "registered_parameters",
    "source",
    "artifacts",
}


class ValidationError(RuntimeError):
    """Raised when an input or output violates the registered contract."""


def _sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity(path: Path) -> dict[str, int | str]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValidationError(f"cannot read loaded source: {path}") from exc
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


_RUNTIME_SOURCE_PATHS = {
    "analysis_code": Path(__file__).resolve(),
    "campaign_validator_code": Path(
        validate_authoritative_campaign.__code__.co_filename
    ).resolve(),
    "embedding_validator_code": Path(
        validate_embedding_bundle.__code__.co_filename
    ).resolve(),
}
for _source_label, _source_path in _RUNTIME_SOURCE_PATHS.items():
    _expected_source_path = (ROOT / ANALYSIS_SOURCE_PATHS[_source_label]).resolve()
    if _source_path != _expected_source_path:
        raise ValidationError(
            f"loaded {_source_label} does not match the canonical workspace source"
        )
_RUNTIME_SOURCE_AT_IMPORT = {
    label: _source_identity(path) for label, path in _RUNTIME_SOURCE_PATHS.items()
}


def _verify_runtime_source_identities() -> None:
    for label, path in _RUNTIME_SOURCE_PATHS.items():
        if _source_identity(path) != _RUNTIME_SOURCE_AT_IMPORT[label]:
            raise ValidationError(
                f"{label} differs from the source loaded into this process"
            )


def _record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    rendered = (
        resolved.relative_to(relative_to.resolve(strict=True)).as_posix()
        if relative_to is not None
        else str(resolved)
    )
    return {
        "path": rendered,
        "bytes": int(stat.st_size),
        "sha256": _sha256_file(resolved),
    }


def _workspace_record(path: Path) -> dict[str, Any]:
    """Record an immutable source using a path portable within the workspace."""
    try:
        return _record(path, relative_to=ROOT)
    except ValueError as exc:
        raise ValidationError(f"source is outside the workspace: {path}") from exc


def _analysis_source_records() -> dict[str, dict[str, Any]]:
    return {
        label: _workspace_record(_workspace_child(path))
        for label, path in ANALYSIS_SOURCE_PATHS.items()
    }


def _verify_analysis_source_records(
    records: Mapping[str, Mapping[str, Any]],
) -> None:
    if set(records) != set(ANALYSIS_SOURCE_PATHS):
        raise ValidationError("analysis source snapshot inventory is invalid")
    for label, relative in ANALYSIS_SOURCE_PATHS.items():
        record = records[label]
        if record.get("path") != relative:
            raise ValidationError(f"analysis source snapshot path changed: {label}")
        _verify_record(
            _workspace_child(relative), record, label=f"analysis source {label}"
        )


def _verify_record(path: Path, record: Mapping[str, Any], *, label: str) -> None:
    if not path.is_file():
        raise ValidationError(f"{label} is missing: {path}")
    expected_bytes = record.get("bytes")
    expected_sha = record.get("sha256")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 0
    ):
        raise ValidationError(f"{label} has an invalid byte count")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValidationError(f"{label} has an invalid SHA-256")
    stat = path.stat()
    if int(stat.st_size) != expected_bytes or _sha256_file(path) != expected_sha:
        raise ValidationError(f"{label} no longer matches its bound bytes")


def _safe_child(parent: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValidationError("artifact path is missing or invalid")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValidationError(
            f"artifact path is not a safe relative path: {relative!r}"
        )
    root = parent.resolve(strict=True)
    unresolved = root / candidate
    cursor = root
    for part in candidate.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValidationError(f"artifact path traverses a symlink: {relative!r}")
    resolved = unresolved.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValidationError(f"artifact escapes its run directory: {relative!r}")
    return resolved


def _workspace_child(relative: Any) -> Path:
    """Resolve a normalized workspace-relative record without allowing escapes."""
    path = _safe_child(ROOT, relative)
    if path.is_symlink():
        raise ValidationError(f"workspace source must not be a symlink: {relative!r}")
    return path


class _DuplicateJSONKey(ValueError):
    """Internal signal used by the duplicate-key rejecting JSON decoder."""


class _NonFiniteJSONConstant(ValueError):
    """Internal signal used by the non-finite-value rejecting JSON decoder."""


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJSONKey(f"duplicate JSON object key: {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise _NonFiniteJSONConstant(f"non-finite JSON constant: {value}")


def _parse_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise _NonFiniteJSONConstant(f"non-finite JSON number: {value}")
    return parsed


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_parse_json_float,
        )
    except _DuplicateJSONKey as exc:
        raise ValidationError(f"{label} contains duplicate JSON keys: {path}") from exc
    except _NonFiniteJSONConstant as exc:
        raise ValidationError(
            f"{label} contains a non-finite JSON value: {path}"
        ) from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{label} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"{label} must be a JSON object")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValidationError(f"{label} must be a lowercase 64-character SHA-256")
    return value


def _require_canonical_path(path: Path, expected: Path, *, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
        canonical = expected.resolve(strict=True)
    except OSError as exc:
        raise ValidationError(f"{label} is missing") from exc
    if resolved != canonical:
        try:
            canonical_display = canonical.relative_to(
                ROOT.resolve(strict=True)
            ).as_posix()
        except (OSError, ValueError):
            canonical_display = str(canonical)
        raise ValidationError(
            f"{label} must be the canonical workspace artifact: {canonical_display}"
        )
    return canonical


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            dict(value), indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
        )
        + "\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise ValidationError(f"refusing to overwrite existing artifact: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    payload: dict[str, np.ndarray] = dict(arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **payload)  # type: ignore[arg-type]
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ValidationError(
                f"refusing to overwrite existing artifact: {path}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _read_lines(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if not values or any(not value for value in values):
        raise ValidationError(f"manifest contains empty or no records: {path}")
    if len(values) != len(set(values)):
        raise ValidationError(f"manifest contains duplicate records: {path}")
    return values


def _parse_fasta_lengths(path: Path) -> tuple[list[str], np.ndarray]:
    names: list[str] = []
    lengths: list[int] = []
    current_name: str | None = None
    current_length = 0
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ValidationError(f"cannot open FASTA: {path}") from exc
    with handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name is not None:
                    names.append(current_name)
                    lengths.append(current_length)
                header = line[1:].split(maxsplit=1)[0]
                current_name = header.split("|", maxsplit=1)[0]
                if not current_name:
                    raise ValidationError("FASTA contains an empty sequence name")
                current_length = 0
                continue
            if current_name is None:
                raise ValidationError("FASTA sequence occurs before the first header")
            if not line.isalpha():
                raise ValidationError(
                    f"FASTA sequence for {current_name} contains non-letter characters"
                )
            current_length += len(line)
    if current_name is not None:
        names.append(current_name)
        lengths.append(current_length)
    if (
        not names
        or len(names) != len(set(names))
        or any(length <= 0 for length in lengths)
    ):
        raise ValidationError("FASTA names must be unique and sequences nonempty")
    return names, np.asarray(lengths, dtype=np.int32)


def _validate_embedding_manifest(
    path: Path, *, fasta: Path, genes: Sequence[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    canonical_manifest = _require_canonical_path(
        path, DEFAULT_EMBEDDING_MANIFEST, label="embedding manifest"
    )
    canonical_fasta = _require_canonical_path(fasta, DEFAULT_FASTA, label="FASTA")
    try:
        provenance = validate_embedding_bundle(ROOT)
    except EmbeddingProvenanceError as exc:
        raise ValidationError(
            f"canonical embedding bundle failed provenance verification: {exc}"
        ) from exc
    manifest = _read_json(canonical_manifest, label="embedding manifest")
    if (
        manifest.get("schema") != "DepPLM-esm-embedding-v1"
        or manifest.get("status") != "complete"
    ):
        raise ValidationError("embedding manifest is not a complete canonical bundle")
    generation = manifest.get("generation")
    inputs = manifest.get("input")
    outputs = manifest.get("outputs")
    if not isinstance(generation, dict) or generation.get("strategy") != "chunked":
        raise ValidationError(
            "embedding manifest is not the registered chunked strategy"
        )
    required = {
        "maximum_residues_per_window": LONG_THRESHOLD,
        "truncation_enabled_in_tokenizer": False,
        "special_tokens_excluded": True,
    }
    for field, expected in required.items():
        if generation.get(field) != expected:
            raise ValidationError(
                f"embedding manifest has noncanonical {field}: {generation.get(field)!r}"
            )
    if not isinstance(inputs, dict) or not isinstance(inputs.get("fasta"), dict):
        raise ValidationError("embedding manifest FASTA provenance is missing")
    _verify_record(canonical_fasta, inputs["fasta"], label="embedding FASTA")
    fasta_names, lengths = _parse_fasta_lengths(canonical_fasta)
    if fasta_names != list(genes):
        raise ValidationError("embedding FASTA order does not match the gene manifest")
    statistics = inputs.get("statistics")
    if not isinstance(statistics, dict):
        raise ValidationError("embedding sequence statistics are missing")
    if statistics.get("n_sequences") != len(genes):
        raise ValidationError("embedding sequence count does not match genes")
    if statistics.get("n_sequences_longer_than_window") != int(
        np.sum(lengths > LONG_THRESHOLD)
    ):
        raise ValidationError("embedding long-protein count does not match the FASTA")
    if not isinstance(outputs, dict) or not isinstance(outputs.get("embeddings"), dict):
        raise ValidationError("embedding output array provenance is missing")
    array_record = outputs["embeddings"]
    array_path = Path(str(array_record.get("path", "")))
    if not array_path.is_absolute():
        array_path = ROOT / array_path
    _verify_record(array_path, array_record, label="chunked embedding array")
    if (
        array_record.get("dtype") != "float32"
        or array_record.get("shape") != [len(genes), 320]
        or array_record.get("numpy_allow_pickle") is not False
    ):
        raise ValidationError("chunked embedding array metadata is noncanonical")
    return manifest, provenance


def _expected_campaign_keys() -> set[str]:
    protocols = ("p1", "transfer-forward", "transfer-reverse")
    conditions = (
        "DepPLM",
        "C-DEEPDEP",
        "C-ONEHOT",
        "C-PLM-MLP",
        "C-GCN",
        "C-GCN-comut",
        "A1-ESM",
        "A2-GNN",
        "A3-comut",
        "A4-MT",
        "A5-XATTN",
    )
    return {
        f"{protocol}|{condition}|seed-{seed}"
        for protocol in protocols
        for condition in conditions
        for seed in SEEDS
    }


def _campaign_run_path_map(
    report: Mapping[str, Any], *, recorded_paths: Mapping[str, Any] | None = None
) -> dict[str, str]:
    runs = report.get("source_runs")
    if not isinstance(runs, list) or len(runs) != EXPECTED_CAMPAIGN_RUNS:
        raise ValidationError("campaign must bind exactly 99 source runs")
    keys = [run.get("key") for run in runs if isinstance(run, dict)]
    if len(keys) != EXPECTED_CAMPAIGN_RUNS or set(keys) != _expected_campaign_keys():
        raise ValidationError("campaign source-run matrix is noncanonical")
    if recorded_paths is not None and set(recorded_paths) != set(keys):
        raise ValidationError("portable campaign run inventory is incomplete")

    workspace = ROOT.resolve(strict=True)
    portable: dict[str, str] = {}
    resolved_paths: set[Path] = set()
    for run in runs:
        if not isinstance(run, dict):
            raise ValidationError("campaign source run must be an object")
        key = str(run["key"])
        if recorded_paths is None:
            raw = run.get("path")
            if not isinstance(raw, str) or not raw:
                raise ValidationError(f"campaign source run path is missing: {key}")
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = workspace / candidate
            try:
                resolved = candidate.resolve(strict=True)
            except OSError as exc:
                raise ValidationError(
                    f"campaign source run directory is missing: {key}"
                ) from exc
            try:
                relative = resolved.relative_to(workspace).as_posix()
            except ValueError as exc:
                raise ValidationError(
                    f"campaign source run is outside the workspace: {key}"
                ) from exc
        else:
            relative_value = recorded_paths[key]
            resolved = _workspace_child(relative_value)
            relative = resolved.relative_to(workspace).as_posix()
        if not resolved.is_dir() or resolved.is_symlink():
            raise ValidationError(
                f"campaign source run is not a regular directory: {key}"
            )
        if resolved in resolved_paths:
            raise ValidationError("campaign source run directory is reused")
        resolved_paths.add(resolved)
        portable[key] = relative
    return dict(sorted(portable.items()))


def _strong_validate_portable_campaign(
    report: Mapping[str, Any], *, run_paths: Mapping[str, str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    created_utc = report.get("created_utc")
    if not isinstance(created_utc, str) or not created_utc:
        raise ValidationError("bound campaign report creation time is missing")
    resolved_run_dirs = [_workspace_child(path) for path in run_paths.values()]
    parent_dirs = {path.parent for path in resolved_run_dirs}
    if len(parent_dirs) != 1:
        raise ValidationError(
            "campaign source runs must be direct children of one results root"
        )
    results_root = next(iter(parent_dirs))
    try:
        replayed = validate_authoritative_campaign(
            results_root, workspace_root=ROOT.resolve(strict=True)
        )
    except CampaignValidationError as exc:
        raise ValidationError(
            f"campaign evidence failed authoritative validation: {exc}"
        ) from exc
    if _without_creation_time(replayed) != _without_creation_time(report):
        issues = replayed.get("validation", {}).get("issues", [])
        raise ValidationError(
            "campaign evidence failed authoritative replay; the current 99-run "
            f"validation report differs from the bound report (issues={issues!r})"
        )
    if replayed.get("schema") != CAMPAIGN_SCHEMA:
        raise ValidationError("authoritative replay returned a noncanonical schema")
    if replayed.get("protocol_validation_passed") is not True:
        raise ValidationError("authoritative replay did not pass protocol validation")
    gates = replayed.get("scientific_readiness_gates")
    expected_gates = {
        "verified_embedding_generation_provenance",
        "outer_test_metrics_recomputed_from_bound_predictions",
    }
    if (
        not isinstance(gates, dict)
        or set(gates) != expected_gates
        or any(value is not True for value in gates.values())
    ):
        raise ValidationError(
            "authoritative replay did not pass scientific-readiness gates"
        )
    if replayed.get("scientific_result") is not True:
        raise ValidationError("authoritative replay is not a scientific result")
    validation = replayed.get("validation")
    if not isinstance(validation, dict):
        raise ValidationError("authoritative replay validation summary is missing")
    if (
        validation.get("validated_runs") != EXPECTED_CAMPAIGN_RUNS
        or validation.get("unique_matrix_cells") != EXPECTED_CAMPAIGN_RUNS
        or validation.get("missing_matrix_cells") != []
        or validation.get("duplicate_matrix_cells") != {}
        or validation.get("issues") != []
    ):
        raise ValidationError(
            "authoritative replay did not validate a clean 99-run matrix"
        )
    replayed_runs = replayed.get("source_runs")
    if (
        not isinstance(replayed_runs, list)
        or len(replayed_runs) != EXPECTED_CAMPAIGN_RUNS
    ):
        raise ValidationError(
            "authoritative replay did not validate exactly 99 source runs"
        )
    source_artifacts_revalidated = 0
    for run in replayed_runs:
        if not isinstance(run, dict) or not isinstance(run.get("fold_artifacts"), list):
            raise ValidationError(
                "authoritative replay returned invalid source-run evidence"
            )
        key = run.get("key")
        if not isinstance(key, str):
            raise ValidationError("authoritative replay returned an invalid run key")
        protocol = key.split("|", maxsplit=1)[0]
        expected_fold_artifacts = EXPECTED_FOLD_ARTIFACTS_BY_PROTOCOL.get(protocol)
        if (
            expected_fold_artifacts is None
            or len(run["fold_artifacts"]) != expected_fold_artifacts
        ):
            raise ValidationError(
                "authoritative replay returned a noncanonical fold-artifact count "
                f"for {key}"
            )
        source_artifacts_revalidated += len(RUN_ROOT_JSON_FILES) + len(
            run["fold_artifacts"]
        )
    expected_revalidated_artifacts = (
        EXPECTED_FOLD_ARTIFACTS + EXPECTED_CAMPAIGN_RUNS * len(RUN_ROOT_JSON_FILES)
    )
    if source_artifacts_revalidated != expected_revalidated_artifacts:
        raise ValidationError(
            "authoritative replay returned a noncanonical campaign artifact count"
        )
    record = {
        "results_root": results_root.relative_to(ROOT.resolve(strict=True)).as_posix(),
        "source_runs_revalidated": len(replayed_runs),
        "source_artifacts_revalidated": source_artifacts_revalidated,
    }
    return dict(report), record


def _validate_campaign_report(
    path: Path, *, recorded_run_paths: Mapping[str, Any] | None = None
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """Replay all 99 runs semantically and return a portable run-path binding."""
    report_path = path.resolve(strict=True)
    report = _read_json(report_path, label="campaign report")
    run_paths = _campaign_run_path_map(report, recorded_paths=recorded_run_paths)
    validated, strong_record = _strong_validate_portable_campaign(
        report, run_paths=run_paths
    )
    source_runs_revalidated = strong_record.get("source_runs_revalidated")
    source_artifacts_revalidated = strong_record.get("source_artifacts_revalidated")
    if source_runs_revalidated != EXPECTED_CAMPAIGN_RUNS:
        raise ValidationError(
            "authoritative validator did not replay all 99 campaign runs"
        )
    if (
        isinstance(source_artifacts_revalidated, bool)
        or not isinstance(source_artifacts_revalidated, int)
        or source_artifacts_revalidated
        != EXPECTED_FOLD_ARTIFACTS + EXPECTED_CAMPAIGN_RUNS * len(RUN_ROOT_JSON_FILES)
    ):
        raise ValidationError(
            "authoritative validator did not replay the exact campaign artifact set"
        )
    record = {
        **_workspace_record(report_path),
        **strong_record,
    }
    return validated, record, run_paths


def _analysis_runtime_environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ANALYSIS_ENVIRONMENT_PACKAGES
        },
    }


def _validate_analysis_runtime_environment(value: Any) -> None:
    if value != _analysis_runtime_environment():
        raise ValidationError(
            "analysis runtime environment does not match this verifier"
        )


def _validate_analysis_manifest_contract(manifest: Mapping[str, Any]) -> None:
    if set(manifest) != ANALYSIS_MANIFEST_FIELDS:
        raise ValidationError("run manifest inventory is invalid")
    if manifest.get("schema") != RUN_SCHEMA:
        raise ValidationError("run manifest schema is invalid")
    if not isinstance(manifest.get("created_utc"), str) or not manifest["created_utc"]:
        raise ValidationError("run manifest creation time is invalid")
    if manifest.get("registered_parameters") != REGISTERED_PARAMETERS:
        raise ValidationError("run manifest registered parameters are invalid")


def _run_manifest_inputs(
    run_manifest: Mapping[str, Any], *, key: str
) -> dict[str, Any]:
    records = run_manifest.get("inputs")
    if not isinstance(records, list):
        raise ValidationError(f"run manifest inputs are missing: {key}")
    indexed: dict[str, Any] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValidationError(f"run manifest input is invalid: {key}[{index}]")
        raw_path = record.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValidationError(f"run manifest input path is invalid: {key}[{index}]")
        normalized = raw_path.replace("\\", "/")
        if Path(normalized).is_absolute() or ".." in Path(normalized).parts:
            raise ValidationError(f"run manifest input path is unsafe: {key}[{index}]")
        if normalized in indexed:
            raise ValidationError(f"duplicate run manifest input: {key}:{normalized}")
        indexed[normalized] = record
    return indexed


def _validate_campaign_input_binding(
    report: Mapping[str, Any],
    *,
    run_paths: Mapping[str, str],
    embedding_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind every analysis input to each selected P1 run manifest."""
    aggregation_provenance = report.get("aggregation_provenance")
    if not isinstance(aggregation_provenance, Mapping):
        raise ValidationError("campaign aggregation provenance is missing")
    campaign_bundle = aggregation_provenance.get("embedding_bundle")
    if not isinstance(campaign_bundle, Mapping):
        raise ValidationError("campaign embedding provenance is missing")
    if dict(campaign_bundle) != dict(embedding_provenance):
        raise ValidationError(
            "campaign embedding provenance differs from the canonical validated bundle"
        )
    required = {
        label: _workspace_record(_workspace_child(relative))
        for label, relative in CANONICAL_ANALYSIS_INPUTS.items()
    }
    selected_runs: list[dict[str, Any]] = []
    for (condition, seed), run in sorted(_p1_run_map(report).items()):
        key = str(run["key"])
        run_path = _workspace_child(run_paths[key])
        manifest_path = run_path / "run-manifest.json"
        expected_manifest_sha = _require_sha256(
            run.get("run_manifest_sha256"), label=f"{key} run manifest SHA-256"
        )
        if _sha256_file(manifest_path) != expected_manifest_sha:
            raise ValidationError(f"selected run manifest changed: {key}")
        run_manifest = _read_json(manifest_path, label=f"{key}.run-manifest.json")
        if run_manifest.get("embedding_provenance") != dict(embedding_provenance):
            raise ValidationError(
                f"selected run embedding provenance is noncanonical: {key}"
            )
        inputs = _run_manifest_inputs(run_manifest, key=key)
        for label, relative in CANONICAL_ANALYSIS_INPUTS.items():
            record = inputs.get(relative)
            if not isinstance(record, dict):
                raise ValidationError(f"selected run does not bind {relative}: {key}")
            expected = required[label]
            if (
                record.get("bytes") != expected["bytes"]
                or record.get("sha256") != expected["sha256"]
            ):
                raise ValidationError(
                    f"selected run input differs from canonical {relative}: {key}"
                )
            _verify_record(
                _workspace_child(relative), record, label=f"{key} input {relative}"
            )
        selected_runs.append(
            {
                "key": key,
                "path": run_paths[key],
                "run_manifest_sha256": expected_manifest_sha,
            }
        )
    return {
        "schema": "DepPLM-long-protein-campaign-input-binding-v1",
        "embedding_provenance_sha256": _require_sha256(
            embedding_provenance.get("provenance_sha256"),
            label="embedding provenance SHA-256",
        ),
        "required_inputs": required,
        "selected_runs": selected_runs,
    }


def _p1_run_map(report: Mapping[str, Any]) -> dict[tuple[str, int], Mapping[str, Any]]:
    selected: dict[tuple[str, int], Mapping[str, Any]] = {}
    for run in report["source_runs"]:
        protocol, condition, seed_token = str(run["key"]).split("|")
        if protocol != "p1" or condition not in CONDITIONS:
            continue
        key = (condition, int(seed_token.removeprefix("seed-")))
        if key in selected:
            raise ValidationError(f"duplicate selected P1 run: {key}")
        selected[key] = run
    expected = {(condition, seed) for condition in CONDITIONS for seed in SEEDS}
    if set(selected) != expected:
        raise ValidationError("selected P1 run matrix is incomplete")
    return selected


def _load_p1_oof(
    run: Mapping[str, Any],
    *,
    run_path: Path,
    cells: Sequence[str],
    genes: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    run_path = run_path.resolve(strict=True)
    records: dict[str, Mapping[str, Any]] = {}
    for record in run.get("fold_artifacts", []):
        if not isinstance(record, dict):
            continue
        artifact_name = record.get("path")
        if not isinstance(artifact_name, str):
            continue
        if artifact_name in records:
            raise ValidationError(
                f"duplicate fold artifact path: {run.get('key')}.{artifact_name}"
            )
        records[artifact_name] = record
    expected_paths = {f"{fold}-predictions.npz" for fold in FOLDS}
    if {
        name for name in records if name.endswith("-predictions.npz")
    } != expected_paths:
        raise ValidationError(
            f"P1 prediction inventory is incomplete: {run.get('key')}"
        )
    predictions: np.ndarray = np.full(
        (len(cells), len(genes)), np.nan, dtype=np.float32
    )
    baselines: np.ndarray = np.full((len(cells), len(genes)), np.nan, dtype=np.float32)
    coverage: np.ndarray = np.zeros(len(cells), dtype=np.int8)
    expected_gene_ids = np.asarray(genes, dtype=str)
    for fold in FOLDS:
        relative = f"{fold}-predictions.npz"
        path = _safe_child(run_path, relative)
        _verify_record(path, records[relative], label=f"{run.get('key')}.{relative}")
        try:
            with np.load(path, allow_pickle=False) as archive:
                if tuple(archive.files) != EXPECTED_PREDICTION_ARRAYS:
                    raise ValidationError(f"noncanonical prediction arrays: {path}")
                arrays = {name: np.asarray(archive[name]) for name in archive.files}
        except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile) as exc:
            raise ValidationError(f"cannot load prediction artifact: {path}") from exc
        indices = arrays["outer_test_indices"]
        prediction = arrays["prediction"]
        train_gene_mean = arrays["train_gene_mean"]
        if indices.dtype != np.int64 or indices.ndim != 1:
            raise ValidationError(f"invalid outer-test indices: {path}")
        if (
            prediction.dtype != np.float32
            or prediction.shape != (len(indices), len(genes))
            or not np.isfinite(prediction).all()
        ):
            raise ValidationError(f"invalid predictions: {path}")
        if (
            train_gene_mean.dtype != np.float32
            or train_gene_mean.shape != (len(genes),)
            or not np.isfinite(train_gene_mean).all()
        ):
            raise ValidationError(f"invalid training gene mean: {path}")
        if (
            np.any(indices < 0)
            or np.any(indices >= len(cells))
            or len(indices) != len(np.unique(indices))
        ):
            raise ValidationError(f"outer-test indices are invalid: {path}")
        if not np.array_equal(arrays["gene_ids"], expected_gene_ids):
            raise ValidationError(f"prediction gene order is not canonical: {path}")
        expected_cells = np.asarray([cells[int(index)] for index in indices], dtype=str)
        if not np.array_equal(arrays["cell_ids"], expected_cells):
            raise ValidationError(
                f"prediction cell identities are not canonical: {path}"
            )
        if not np.array_equal(arrays["source_cell_indices"], indices):
            raise ValidationError(f"prediction source-cell indices are invalid: {path}")
        if not np.array_equal(
            arrays["source_gene_indices"], np.arange(len(genes), dtype=np.int64)
        ):
            raise ValidationError(f"prediction source-gene indices are invalid: {path}")
        predictions[indices] = prediction
        baselines[indices] = train_gene_mean[None, :]
        coverage[indices] += 1
    if not np.all(coverage == 1) or not np.isfinite(predictions).all():
        raise ValidationError(
            f"P1 folds do not cover cells exactly once: {run.get('key')}"
        )
    return predictions, baselines


def _per_gene_error(
    prediction: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValidationError("prediction and target matrices must share a 2-D shape")
    valid = np.isfinite(prediction) & np.isfinite(target)
    counts = valid.sum(axis=0).astype(np.int32)
    difference = np.where(valid, prediction - target, np.nan)
    with np.errstate(invalid="ignore"):
        mae = np.nanmean(np.abs(difference), axis=0)
        mse = np.nanmean(np.square(difference), axis=0)
    mae[counts == 0] = np.nan
    mse[counts == 0] = np.nan
    return mae.astype(np.float64), mse.astype(np.float64), counts


def _finite_mean(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if len(finite) else None


def _finite_sd(values: Sequence[float | None]) -> float | None:
    finite = np.asarray(
        [value for value in values if value is not None], dtype=np.float64
    )
    return float(np.std(finite, ddof=1)) if len(finite) > 1 else None


def _descriptive_difference(
    long_values: np.ndarray, short_values: np.ndarray
) -> dict[str, Any]:
    long_finite = np.asarray(long_values, dtype=np.float64)
    short_finite = np.asarray(short_values, dtype=np.float64)
    long_finite = long_finite[np.isfinite(long_finite)]
    short_finite = short_finite[np.isfinite(short_finite)]
    if len(long_finite) < 2 or len(short_finite) < 2:
        raise ValidationError("too few genes for descriptive length comparison")
    long_mean = float(long_finite.mean())
    short_mean = float(short_finite.mean())
    return {
        "estimate_long_minus_short": long_mean - short_mean,
        "mean_long": long_mean,
        "mean_short": short_mean,
        "n_long": int(len(long_finite)),
        "n_short": int(len(short_finite)),
        "inference": "conditional_descriptive_only_no_iid_confidence_interval",
    }


def _spearman(lengths: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(values) & (lengths > 0)
    if int(mask.sum()) < 3 or np.ptp(values[mask]) == 0:
        return {
            "rho": None,
            "n_genes": int(mask.sum()),
            "inference": "conditional_descriptive_only_no_iid_p_value",
        }
    result = stats.spearmanr(np.log1p(lengths[mask]), values[mask])
    rho = float(result.statistic) if math.isfinite(float(result.statistic)) else None
    return {
        "rho": rho,
        "n_genes": int(mask.sum()),
        "inference": "conditional_descriptive_only_no_iid_p_value",
    }


def _length_bin_masks(lengths: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "25-512": lengths <= 512,
        "513-1022": (lengths >= 513) & (lengths <= LONG_THRESHOLD),
        "1023-2044": (lengths > LONG_THRESHOLD) & (lengths <= 2044),
        ">2044": lengths > 2044,
    }


def _summarize_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    genes = np.asarray(arrays["gene_ids"])
    lengths = np.asarray(arrays["length_residues"], dtype=np.int32)
    conditions = [str(value) for value in np.asarray(arrays["condition_ids"])]
    seeds = [int(value) for value in np.asarray(arrays["seed_ids"])]
    mae = np.asarray(arrays["mae"], dtype=np.float64)
    mse = np.asarray(arrays["mse"], dtype=np.float64)
    baseline_mae = np.asarray(arrays["baseline_mae"], dtype=np.float64)
    baseline_mse = np.asarray(arrays["baseline_mse"], dtype=np.float64)
    counts = np.asarray(arrays["finite_target_counts"], dtype=np.int32)
    if (
        genes.ndim != 1
        or lengths.shape != genes.shape
        or np.any(lengths <= 0)
        or len(set(str(value) for value in genes)) != len(genes)
        or any(not str(value) for value in genes)
    ):
        raise ValidationError("gene identifiers or protein lengths are invalid")
    expected_shape = (len(conditions), len(seeds), len(genes))
    if mae.shape != expected_shape or mse.shape != expected_shape:
        raise ValidationError("per-gene model error arrays have invalid shape")
    if baseline_mae.shape != (len(seeds), len(genes)) or baseline_mse.shape != (
        len(seeds),
        len(genes),
    ):
        raise ValidationError("per-gene baseline error arrays have invalid shape")
    if counts.shape != (len(genes),) or np.any(counts <= 0):
        raise ValidationError("finite target counts are invalid")
    if not all(
        np.isfinite(value).all() for value in (mae, mse, baseline_mae, baseline_mse)
    ) or any(np.any(value < 0) for value in (mae, mse, baseline_mae, baseline_mse)):
        raise ValidationError("per-gene error arrays contain invalid values")
    if conditions != list(CONDITIONS) or seeds != list(SEEDS):
        raise ValidationError("condition or seed axes are noncanonical")
    long_mask = lengths > LONG_THRESHOLD
    short_mask = ~long_mask
    if int(long_mask.sum()) != EXPECTED_N_LONG or len(genes) != EXPECTED_N_GENES:
        raise ValidationError("registered long-protein cohort size changed")

    strata = {"short_le_1022": short_mask, "long_gt_1022": long_mask}
    per_seed: list[dict[str, Any]] = []
    for condition_index, condition in enumerate(conditions):
        for seed_index, seed in enumerate(seeds):
            for stratum, mask in strata.items():
                per_seed.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "stratum": stratum,
                        "n_genes": int(mask.sum()),
                        "mean_per_gene_mae": _finite_mean(
                            mae[condition_index, seed_index, mask]
                        ),
                        "mean_per_gene_mse": _finite_mean(
                            mse[condition_index, seed_index, mask]
                        ),
                        "mean_gene_mean_baseline_mae": _finite_mean(
                            baseline_mae[seed_index, mask]
                        ),
                        "mean_gene_mean_baseline_mse": _finite_mean(
                            baseline_mse[seed_index, mask]
                        ),
                    }
                )

    averaged_mae = np.mean(mae, axis=1)
    averaged_mse = np.mean(mse, axis=1)
    averaged_baseline_mae = np.mean(baseline_mae, axis=0)
    dep_index = conditions.index("DepPLM")
    gcn_index = conditions.index("C-GCN")
    plm_index = conditions.index("C-PLM-MLP")
    dep_excess = averaged_mae[dep_index] - averaged_baseline_mae
    dep_vs_gcn = averaged_mae[dep_index] - averaged_mae[gcn_index]
    dep_vs_plm = averaged_mae[dep_index] - averaged_mae[plm_index]
    primary = {
        "depplm_excess_mae_long_minus_short": _descriptive_difference(
            dep_excess[long_mask], dep_excess[short_mask]
        ),
        "depplm_minus_gcn_mae_long_minus_short": _descriptive_difference(
            dep_vs_gcn[long_mask], dep_vs_gcn[short_mask]
        ),
        "depplm_minus_plm_mlp_mae_long_minus_short": _descriptive_difference(
            dep_vs_plm[long_mask], dep_vs_plm[short_mask]
        ),
        "length_vs_depplm_excess_mae": _spearman(lengths, dep_excess),
        "interpretation_guardrail": (
            "These associations evaluate residual length dependence under the "
            "all-residue chunked representation. They do not compare chunking with "
            "legacy first-1022 truncation and must not be described as a causal "
            "chunking benefit."
        ),
        "inference_scope": (
            "Conditional descriptive estimates for these observed outer-test genes "
            "and three fitted seed reruns only. Gene-family, pathway, shared-cell, "
            "and model-fit dependence are not modelled; no population p-values or "
            "confidence intervals are reported."
        ),
    }

    bins: list[dict[str, Any]] = []
    for label, mask in _length_bin_masks(lengths).items():
        record: dict[str, Any] = {"length_bin": label, "n_genes": int(mask.sum())}
        for index, condition in enumerate(conditions):
            record[f"{condition}_mae_seed_mean"] = _finite_mean(
                averaged_mae[index, mask]
            )
            record[f"{condition}_mse_seed_mean"] = _finite_mean(
                averaged_mse[index, mask]
            )
        record["gene_mean_baseline_mae_seed_mean"] = _finite_mean(
            averaged_baseline_mae[mask]
        )
        bins.append(record)

    condition_strata: list[dict[str, Any]] = []
    for condition_index, condition in enumerate(conditions):
        for stratum, mask in strata.items():
            seed_values = [
                _finite_mean(mae[condition_index, seed_index, mask])
                for seed_index in range(len(seeds))
            ]
            condition_strata.append(
                {
                    "condition": condition,
                    "stratum": stratum,
                    "n_genes": int(mask.sum()),
                    "mae_mean_across_seed_summaries": _finite_mean(
                        np.asarray(seed_values, dtype=np.float64)
                    ),
                    "mae_sd_across_seed_summaries": _finite_sd(seed_values),
                }
            )

    return {
        "schema": OUTPUT_SCHEMA,
        "analysis_scope": (
            "Conditional descriptive outer-test P1 error stratification under the "
            "verified all-residue chunked ESM-2 representation"
        ),
        "n_genes": int(len(genes)),
        "long_threshold_residues": LONG_THRESHOLD,
        "n_long": int(long_mask.sum()),
        "n_short": int(short_mask.sum()),
        "fraction_long": float(long_mask.mean()),
        "minimum_length_residues": int(lengths.min()),
        "median_length_residues": float(np.median(lengths)),
        "maximum_length_residues": int(lengths.max()),
        "conditions": conditions,
        "seeds": seeds,
        "error_unit": (
            "gene-level error after reconstructing exactly-once held-out P1 "
            "predictions across all cells; the three registered seed reruns are "
            "averaged before descriptive gene-level comparisons"
        ),
        "per_seed_strata": per_seed,
        "condition_strata": condition_strata,
        "length_bins": bins,
        "primary_associations": primary,
    }


def _make_run_dir(output_root: Path) -> Path:
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    for nonce in range(1000):
        candidate = output_root / f"{timestamp}-long-protein-{os.getpid()}-{nonce:03d}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise ValidationError("could not create a unique output directory")


def _compute_per_gene_arrays(
    *,
    report: Mapping[str, Any],
    run_paths: Mapping[str, str],
    cells: Sequence[str],
    genes: Sequence[str],
    targets: np.ndarray,
    lengths: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    p1_runs = _p1_run_map(report)
    mae: np.ndarray = np.full(
        (len(CONDITIONS), len(SEEDS), len(genes)), np.nan, dtype=np.float64
    )
    mse = np.full_like(mae, np.nan)
    baseline_mae: np.ndarray = np.full(
        (len(SEEDS), len(genes)), np.nan, dtype=np.float64
    )
    baseline_mse = np.full_like(baseline_mae, np.nan)
    finite_counts: np.ndarray | None = None
    used_runs: list[dict[str, Any]] = []
    for condition_index, condition in enumerate(CONDITIONS):
        for seed_index, seed in enumerate(SEEDS):
            run = p1_runs[(condition, seed)]
            run_key = run.get("key")
            if not isinstance(run_key, str) or run_key not in run_paths:
                raise ValidationError("selected P1 run lacks a portable path binding")
            prediction, baseline = _load_p1_oof(
                run,
                run_path=_workspace_child(run_paths[run_key]),
                cells=cells,
                genes=genes,
            )
            model_mae, model_mse, counts = _per_gene_error(prediction, targets)
            mae[condition_index, seed_index] = model_mae
            mse[condition_index, seed_index] = model_mse
            if finite_counts is None:
                finite_counts = counts
            elif not np.array_equal(finite_counts, counts):
                raise ValidationError("finite target masks changed across matched runs")
            current_baseline_mae, current_baseline_mse, baseline_counts = (
                _per_gene_error(baseline, targets)
            )
            if not np.array_equal(counts, baseline_counts):
                raise ValidationError("baseline and model target masks differ")
            if condition_index == 0:
                baseline_mae[seed_index] = current_baseline_mae
                baseline_mse[seed_index] = current_baseline_mse
            elif not np.array_equal(
                baseline_mae[seed_index], current_baseline_mae, equal_nan=True
            ) or not np.array_equal(
                baseline_mse[seed_index], current_baseline_mse, equal_nan=True
            ):
                raise ValidationError(
                    f"outer split baseline differs across conditions for seed {seed}"
                )
            used_runs.append(
                {
                    "key": run["key"],
                    "completion_sha256": run.get("completion_sha256"),
                    "summary_sha256": run.get("summary_sha256"),
                    "run_manifest_sha256": run.get("run_manifest_sha256"),
                    "cohort_manifest_sha256": run.get("cohort_manifest_sha256"),
                }
            )
    if finite_counts is None:
        raise ValidationError("no matched P1 predictions were loaded")
    arrays = {
        "gene_ids": np.asarray(genes, dtype=str),
        "length_residues": np.asarray(lengths, dtype=np.int32),
        "condition_ids": np.asarray(CONDITIONS, dtype=str),
        "seed_ids": np.asarray(SEEDS, dtype=np.int64),
        "mae": mae,
        "mse": mse,
        "baseline_mae": baseline_mae,
        "baseline_mse": baseline_mse,
        "finite_target_counts": finite_counts,
    }
    return arrays, used_runs


def run_analysis(
    *,
    campaign_report: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    fasta: Path = DEFAULT_FASTA,
    embedding_manifest: Path = DEFAULT_EMBEDDING_MANIFEST,
    genes_path: Path = DEFAULT_GENES,
    cells_path: Path = DEFAULT_CELLS,
    gene_effect_path: Path = DEFAULT_GENE_EFFECT,
) -> Path:
    _verify_runtime_source_identities()
    analysis_source_records = _analysis_source_records()
    canonical_fasta = _require_canonical_path(fasta, DEFAULT_FASTA, label="FASTA")
    canonical_manifest = _require_canonical_path(
        embedding_manifest,
        DEFAULT_EMBEDDING_MANIFEST,
        label="embedding manifest",
    )
    canonical_genes = _require_canonical_path(
        genes_path, DEFAULT_GENES, label="gene manifest"
    )
    canonical_cells = _require_canonical_path(
        cells_path, DEFAULT_CELLS, label="cell manifest"
    )
    canonical_targets = _require_canonical_path(
        gene_effect_path, DEFAULT_GENE_EFFECT, label="gene-effect matrix"
    )
    report, campaign_record, campaign_run_paths = _validate_campaign_report(
        campaign_report.resolve()
    )
    genes = _read_lines(canonical_genes)
    cells = _read_lines(canonical_cells)
    fasta_names, lengths = _parse_fasta_lengths(canonical_fasta)
    if fasta_names != genes:
        raise ValidationError("FASTA gene order does not match genes.txt")
    embedding, embedding_provenance = _validate_embedding_manifest(
        canonical_manifest, fasta=canonical_fasta, genes=genes
    )
    campaign_input_binding = _validate_campaign_input_binding(
        report,
        run_paths=campaign_run_paths,
        embedding_provenance=embedding_provenance,
    )
    targets_source = np.load(canonical_targets, mmap_mode="r", allow_pickle=False)
    if targets_source.shape != (len(genes), len(cells)):
        raise ValidationError("gene-effect matrix does not match gene/cell manifests")
    targets = np.asarray(targets_source.T, dtype=np.float32)
    arrays, used_runs = _compute_per_gene_arrays(
        report=report,
        run_paths=campaign_run_paths,
        cells=cells,
        genes=genes,
        targets=targets,
        lengths=lengths,
    )
    summary = _summarize_arrays(arrays)
    _verify_runtime_source_identities()
    _verify_analysis_source_records(analysis_source_records)
    run_dir = _make_run_dir(output_root.resolve())
    arrays_path = run_dir / "per-gene-errors.npz"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "run-manifest.json"
    _atomic_npz(arrays_path, arrays)
    _atomic_json(summary_path, summary)
    manifest = {
        "schema": RUN_SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "registered_parameters": REGISTERED_PARAMETERS,
        "source": {
            "campaign_report": campaign_record,
            "campaign_run_paths": campaign_run_paths,
            "campaign_input_binding": campaign_input_binding,
            "embedding_manifest": _workspace_record(canonical_manifest),
            "embedding_array_sha256": embedding["outputs"]["embeddings"]["sha256"],
            "embedding_array": _workspace_record(
                _workspace_child(embedding["outputs"]["embeddings"]["path"])
            ),
            "fasta": _workspace_record(canonical_fasta),
            "genes": _workspace_record(canonical_genes),
            "cells": _workspace_record(canonical_cells),
            "gene_effect": _workspace_record(canonical_targets),
            **analysis_source_records,
            "analysis_environment": _analysis_runtime_environment(),
            "selected_runs": used_runs,
        },
        "artifacts": {
            "per_gene_errors": _record(arrays_path, relative_to=run_dir),
            "summary": _record(summary_path, relative_to=run_dir),
        },
    }
    _atomic_json(manifest_path, manifest)
    completion = {
        "schema": COMPLETION_SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_manifest": _record(manifest_path, relative_to=run_dir),
        "artifacts": manifest["artifacts"],
    }
    _atomic_json(run_dir / "completion.json", completion)
    completion_sha256 = _sha256_file(run_dir / "completion.json")
    verify_analysis(run_dir, trusted_completion_sha256=completion_sha256)
    return run_dir


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    expected = {
        "gene_ids",
        "length_residues",
        "condition_ids",
        "seed_ids",
        "mae",
        "mse",
        "baseline_mae",
        "baseline_mse",
        "finite_target_counts",
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != expected:
                raise ValidationError("per-gene NPZ inventory is noncanonical")
            return {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile) as exc:
        raise ValidationError("per-gene NPZ is unreadable") from exc


def _without_creation_time(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))
    normalized.pop("created_utc", None)
    return normalized


def _validate_analysis_run_inventory(root: Path) -> None:
    entries = list(root.iterdir())
    names = {entry.name for entry in entries}
    if names != ANALYSIS_RUN_FILES:
        missing = sorted(ANALYSIS_RUN_FILES - names)
        unexpected = sorted(names - ANALYSIS_RUN_FILES)
        raise ValidationError(
            f"analysis run inventory mismatch; missing={missing!r}, "
            f"unexpected={unexpected!r}"
        )
    if any(not entry.is_file() or entry.is_symlink() for entry in entries):
        raise ValidationError("analysis run inventory must contain regular files only")


def _verified_workspace_source(
    source: Mapping[str, Any], label: str, *, expected_path: str | None = None
) -> Path:
    record = source.get(label)
    if not isinstance(record, dict):
        raise ValidationError(f"source record is missing: {label}")
    if expected_path is not None and record.get("path") != expected_path:
        raise ValidationError(f"source record path is noncanonical: {label}")
    path = _workspace_child(record.get("path"))
    _verify_record(path, record, label=f"source {label}")
    return path


def verify_analysis(run_dir: Path, *, trusted_completion_sha256: str) -> dict[str, Any]:
    _verify_runtime_source_identities()
    analysis_source_records = _analysis_source_records()
    root = run_dir.resolve(strict=True)
    _validate_analysis_run_inventory(root)
    trusted_digest = _require_sha256(
        trusted_completion_sha256, label="trusted completion digest"
    )
    completion_path = root / "completion.json"
    actual_completion_sha256 = _sha256_file(completion_path)
    if actual_completion_sha256 != trusted_digest:
        raise ValidationError("completion marker does not match the trusted SHA-256")
    completion = _read_json(completion_path, label="completion marker")
    if completion.get("schema") != COMPLETION_SCHEMA:
        raise ValidationError("completion marker schema is invalid")
    if set(completion) != {"schema", "created_utc", "run_manifest", "artifacts"}:
        raise ValidationError("completion marker inventory is invalid")
    manifest_record = completion.get("run_manifest")
    if not isinstance(manifest_record, dict):
        raise ValidationError("completion marker lacks the run manifest record")
    manifest_path = _safe_child(root, manifest_record.get("path"))
    _verify_record(manifest_path, manifest_record, label="run manifest")
    manifest = _read_json(manifest_path, label="run manifest")
    _validate_analysis_manifest_contract(manifest)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "per_gene_errors",
        "summary",
    }:
        raise ValidationError("run artifact inventory is invalid")
    if completion.get("artifacts") != artifacts:
        raise ValidationError(
            "completion artifact records differ from the run manifest"
        )
    for label, record in artifacts.items():
        if not isinstance(record, dict):
            raise ValidationError(f"{label} record is invalid")
        path = _safe_child(root, record.get("path"))
        _verify_record(path, record, label=label)
    arrays = _load_arrays(_safe_child(root, artifacts["per_gene_errors"]["path"]))
    expected_summary = _summarize_arrays(arrays)
    saved_summary = _read_json(
        _safe_child(root, artifacts["summary"]["path"]), label="summary"
    )
    if _without_creation_time(saved_summary) != _without_creation_time(
        expected_summary
    ):
        raise ValidationError("summary does not reproduce from bound per-gene arrays")
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ValidationError("source provenance is missing")
    expected_source_fields = {
        "campaign_report",
        "campaign_run_paths",
        "campaign_input_binding",
        "embedding_manifest",
        "embedding_array_sha256",
        "embedding_array",
        "fasta",
        "genes",
        "cells",
        "gene_effect",
        "analysis_code",
        "campaign_validator_code",
        "embedding_validator_code",
        "analysis_environment",
        "selected_runs",
    }
    if set(source) != expected_source_fields:
        raise ValidationError("source provenance inventory is invalid")
    campaign_path = _verified_workspace_source(source, "campaign_report")
    campaign_run_paths = source.get("campaign_run_paths")
    if not isinstance(campaign_run_paths, dict):
        raise ValidationError("portable campaign run paths are missing")
    report, campaign_record, verified_run_paths = _validate_campaign_report(
        campaign_path, recorded_run_paths=campaign_run_paths
    )
    if source["campaign_report"] != campaign_record:
        raise ValidationError("campaign report re-hash metadata changed")
    if campaign_run_paths != verified_run_paths:
        raise ValidationError("portable campaign run paths changed")
    genes_path = _verified_workspace_source(
        source, "genes", expected_path=CANONICAL_ANALYSIS_INPUTS["genes"]
    )
    cells_path = _verified_workspace_source(
        source, "cells", expected_path=CANONICAL_ANALYSIS_INPUTS["cells"]
    )
    fasta_path = _verified_workspace_source(
        source, "fasta", expected_path=CANONICAL_ANALYSIS_INPUTS["fasta"]
    )
    gene_effect_path = _verified_workspace_source(
        source,
        "gene_effect",
        expected_path=CANONICAL_ANALYSIS_INPUTS["gene_effect"],
    )
    embedding_manifest_path = _verified_workspace_source(
        source,
        "embedding_manifest",
        expected_path=CANONICAL_ANALYSIS_INPUTS["embedding_manifest"],
    )
    _verified_workspace_source(
        source,
        "embedding_array",
        expected_path=CANONICAL_ANALYSIS_INPUTS["embedding_array"],
    )
    for label, expected_path in ANALYSIS_SOURCE_PATHS.items():
        _verified_workspace_source(source, label, expected_path=expected_path)
    _validate_analysis_runtime_environment(source.get("analysis_environment"))
    genes = _read_lines(genes_path)
    cells = _read_lines(cells_path)
    fasta_names, lengths = _parse_fasta_lengths(fasta_path)
    if fasta_names != genes:
        raise ValidationError("verified FASTA order does not match genes.txt")
    embedding, embedding_provenance = _validate_embedding_manifest(
        embedding_manifest_path, fasta=fasta_path, genes=genes
    )
    if source.get("embedding_array_sha256") != embedding["outputs"]["embeddings"].get(
        "sha256"
    ):
        raise ValidationError("recorded embedding SHA-256 disagrees with its manifest")
    expected_input_binding = _validate_campaign_input_binding(
        report,
        run_paths=verified_run_paths,
        embedding_provenance=embedding_provenance,
    )
    if source.get("campaign_input_binding") != expected_input_binding:
        raise ValidationError("campaign input binding does not reproduce")
    targets_source = np.load(gene_effect_path, mmap_mode="r", allow_pickle=False)
    if targets_source.shape != (len(genes), len(cells)):
        raise ValidationError("verified gene-effect matrix has an invalid shape")
    targets = np.asarray(targets_source.T, dtype=np.float32)
    recomputed_arrays, used_runs = _compute_per_gene_arrays(
        report=report,
        run_paths=verified_run_paths,
        cells=cells,
        genes=genes,
        targets=targets,
        lengths=lengths,
    )
    for name, expected in recomputed_arrays.items():
        actual = arrays[name]
        equal = (
            np.array_equal(actual, expected, equal_nan=True)
            if np.issubdtype(np.asarray(actual).dtype, np.inexact)
            else np.array_equal(actual, expected)
        )
        if not equal:
            raise ValidationError(
                f"bound per-gene array does not replay from outer predictions: {name}"
            )
    if source.get("selected_runs") != used_runs:
        raise ValidationError(
            "selected-run provenance does not match the campaign report"
        )
    _verify_runtime_source_identities()
    _verify_analysis_source_records(analysis_source_records)
    return {
        "schema": SELF_VERIFICATION_SCHEMA,
        "valid": True,
        "run_dir": str(root),
        "completion_sha256": actual_completion_sha256,
        "summary_sha256": artifacts["summary"]["sha256"],
        "per_gene_errors_sha256": artifacts["per_gene_errors"]["sha256"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--campaign-report", type=Path, required=True)
    run_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run_parser.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    run_parser.add_argument(
        "--embedding-manifest", type=Path, default=DEFAULT_EMBEDDING_MANIFEST
    )
    run_parser.add_argument("--genes", type=Path, default=DEFAULT_GENES)
    run_parser.add_argument("--cells", type=Path, default=DEFAULT_CELLS)
    run_parser.add_argument("--gene-effect", type=Path, default=DEFAULT_GENE_EFFECT)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("run_dir", type=Path)
    verify_parser.add_argument(
        "--trusted-completion-sha256",
        required=True,
        help="Externally stored SHA-256 of completion.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        run_dir = run_analysis(
            campaign_report=args.campaign_report,
            output_root=args.output_root,
            fasta=args.fasta,
            embedding_manifest=args.embedding_manifest,
            genes_path=args.genes,
            cells_path=args.cells,
            gene_effect_path=args.gene_effect,
        )
        print(run_dir)
        return 0
    verification = verify_analysis(
        args.run_dir,
        trusted_completion_sha256=args.trusted_completion_sha256,
    )
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
