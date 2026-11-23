#!/usr/bin/env python3
"""Freeze and execute the locked DepMap 26Q3 time-out validation on the server."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import traceback
import uuid
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from temporal_26q3_validation import (
    TemporalValidationError,
    align_temporal_cohort,
    audit_organoid_overlap,
    evaluate_shared_model_pipeline_correction,
    evaluate_temporal_cohort,
    hash_array,
    parse_gene_headers,
    select_new_models,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOCK = ROOT / "docs" / "revision_v4" / "family_transfer_lock.json"
DEFAULT_PROTOCOL = ROOT / "docs" / "revision_v4" / "FAMILY_TRANSFER_PROTOCOL.md"
DEFAULT_ORGANOID_MODEL_IDS = (
    ROOT / "docs" / "revision_v4" / "organoid_model_ids_predeclared.txt"
)
ORGANOID_SOURCE_CSV = (
    ROOT
    / "data"
    / "external"
    / "sanger_organoid_2026"
    / "features-v1"
    / "data_covariates_ALL-organoids.csv"
)
ORGANOID_SOURCE_SHA256 = (
    "e57603e0ae26780ef112cbdbf063818dd17837858353894ba29e75f6936e9f64"
)
ORGANOID_SOURCE_ID_COLUMN = "sample_ID"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "family_inductive_v2" / "temporal-26q3"
FREEZE_SCHEMA = "DepPLM-temporal-26q3-freeze-v1"
RESULT_SCHEMA = "DepPLM-temporal-26q3-result-v1"
EVIDENCE_SCHEMA = "DepPLM-temporal-26q3-evidence-v1"
COMPLETION_SCHEMA = "DepPLM-temporal-26q3-completion-v1"
CAMPAIGN_COMPLETION_SCHEMA = "DepPLM-family-inductive-campaign-completion-v2"
CONSENSUS_PREDICTIONS_SCHEMA = "DepPLM-family-transfer-consensus-predictions-v2"
IDENTIFIER = __import__("re").compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
HEX64 = __import__("re").compile(r"[0-9a-f]{64}")
HEX32 = __import__("re").compile(r"[0-9a-f]{32}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TemporalValidationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TemporalValidationError(f"cannot load {label} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TemporalValidationError(f"{label} JSON root must be an object")
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
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _json_safe(value),
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, *, relative_to: Path = ROOT) -> dict[str, Any]:
    if path.is_symlink():
        raise TemporalValidationError(f"required artifact cannot be a symlink: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise TemporalValidationError(f"required artifact is not a regular file: {path}")
    try:
        relative = resolved.relative_to(relative_to.resolve(strict=True)).as_posix()
    except ValueError as exc:
        raise TemporalValidationError(f"artifact escapes frozen root: {resolved}") from exc
    return {
        "path": relative,
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _verify_file_record(record: Mapping[str, Any], *, base: Path, label: str) -> Path:
    if set(record) != {"path", "bytes", "sha256"}:
        raise TemporalValidationError(f"{label} file record fields mismatch")
    digest = str(record["sha256"])
    if HEX64.fullmatch(digest) is None:
        raise TemporalValidationError(f"{label} has invalid SHA-256")
    unresolved = base / str(record["path"])
    if unresolved.is_symlink():
        raise TemporalValidationError(f"{label} cannot be a symlink")
    candidate = unresolved.resolve(strict=True)
    try:
        candidate.relative_to(base.resolve(strict=True))
    except ValueError as exc:
        raise TemporalValidationError(f"{label} path escapes frozen root") from exc
    if not candidate.is_file():
        raise TemporalValidationError(f"{label} is not a regular file")
    byte_count = record["bytes"]
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        raise TemporalValidationError(f"{label} has invalid byte count")
    if candidate.stat().st_size != byte_count or _sha256_file(candidate) != digest:
        raise TemporalValidationError(f"{label} bytes differ from frozen record")
    return candidate


def _load_lock(path: Path) -> dict[str, Any]:
    lock = _load_json(path, label="analysis lock")
    if lock.get("schema") != "DepPLM-family-transfer-analysis-lock-v2":
        raise TemporalValidationError("analysis lock schema mismatch")
    if lock.get("protocol_version") != "1.4":
        raise TemporalValidationError("analysis protocol version mismatch")
    if lock.get("implementation_namespace") != "family_inductive_v2":
        raise TemporalValidationError("analysis namespace mismatch")
    temporal = lock.get("temporal_26q3")
    if not isinstance(temporal, Mapping):
        raise TemporalValidationError("temporal contract is missing")
    if temporal.get("outcome_filename") != "CRISPRGeneEffect.csv":
        raise TemporalValidationError("temporal outcome filename differs from lock")
    if temporal.get("prediction_source") != "frozen_24q4_primary_consensus_without_refit_or_recalibration":
        raise TemporalValidationError("temporal prediction source differs from lock")
    if temporal.get("first_access_event_required_before_outcome_parse") is not True:
        raise TemporalValidationError("first-access ledger is not required by lock")
    if temporal.get("acquisition_ledger_event_required_before_outcome_bytes") is not True:
        raise TemporalValidationError("pre-acquisition ledger event is not required by lock")
    if temporal.get("access_ledger_path") != "results/family_inductive_v2/temporal-access-ledger.jsonl":
        raise TemporalValidationError("temporal access ledger path differs from v1.4")
    if temporal.get("input_snapshot_sha256_required") is not True:
        raise TemporalValidationError("input snapshot hash is not required by lock")
    if temporal.get("primary_method_prediction") != "FAMILY-DISTANCE-STACK":
        raise TemporalValidationError("temporal primary method differs from v1.4")
    if temporal.get("comparison_prediction") != "consensus_per_job_selected_baseline":
        raise TemporalValidationError("temporal comparison differs from v1.4")
    if temporal.get("reference_prediction") != "consensus_GLOBAL-TRAIN-MEAN":
        raise TemporalValidationError("temporal reference differs from v1.4")
    if temporal.get("sensitivity_metrics") != ["pearson", "spearman", "mse", "mae"]:
        raise TemporalValidationError("temporal sensitivity metrics differ from v1.4")
    if temporal.get("source_provenance_contract") != {
        "required": True,
        "receipt_path": "data/depmap/DepMap_26Q3_Public/source_receipt.json",
        "release_id": "DepMap_26Q3_Public",
        "outcome_filename": "CRISPRGeneEffect.csv",
        "required_fields": [
            "release_id", "official_release_url", "official_file_url",
            "archive_file_id", "retrieved_at_utc", "bytes", "sha256",
        ],
        "allowed_url_hosts": ["depmap.org", "figshare.com", "api.figshare.com"],
        "manual_official_metadata_verification_required": True,
    }:
        raise TemporalValidationError("26Q3 source provenance contract differs from v1.4")
    if temporal.get("shared_model_pipeline_correction_contract") != {
        "role": "secondary_cannot_replace_new_model_primary",
        "shared_models": "exact_24q4_26q3_model_id_intersection_all",
        "gene_matching": "same_exact_symbol_mapping_as_primary",
        "minimum_finite_paired_shared_models_per_gene": 30,
        "minimum_eligible_genes": 5000,
        "correction": "per_gene_median_of_26q3_minus_24q4_paired_shared_model_values",
        "evaluation_cohort": "same_new_models_as_primary",
        "predictions": "frozen_24q4_consensus_predictions_plus_per_gene_correction",
        "reference": "frozen_24q4_consensus_global_prediction_plus_same_correction",
        "failed_eligibility_action": "report_ineligible_no_primary_rescue",
    }:
        raise TemporalValidationError("shared-model correction contract differs from v1.4")
    cohort = temporal.get("cohort")
    if not isinstance(cohort, Mapping):
        raise TemporalValidationError("temporal cohort contract is missing")
    expected_cohort = {
        "minimum_new_models": 100,
        "gene_header_regex": r"^(?P<symbol>.+) \((?P<entrez>[0-9]+)\)$",
        "gene_matching": "exact_symbol_after_terminal_numeric_entrez_suffix_removal",
        "synonym_rescue": "forbidden",
        "duplicate_symbol_action": "fail_closed",
        "minimum_finite_model_fraction_per_gene": 0.8,
        "minimum_eligible_genes": 5000,
    }
    for key, expected in expected_cohort.items():
        if cohort.get(key) != expected:
            raise TemporalValidationError(f"temporal cohort {key} differs from v1.4")
    baseline = cohort.get("baseline_model_ids_source")
    if (
        not isinstance(baseline, Mapping)
        or set(baseline) != {"path", "sha256", "semantics"}
        or HEX64.fullmatch(str(baseline.get("sha256"))) is None
        or baseline.get("semantics") != "exact_24q4_crispr_gene_effect_row_order_and_model_ids"
        or baseline.get("path") != lock.get("locked_assets", {}).get("cell_manifest", {}).get("path")
        or baseline.get("sha256") != lock.get("locked_assets", {}).get("cell_manifest", {}).get("sha256")
    ):
        raise TemporalValidationError("24Q4 baseline ModelID source binding is invalid")
    overlap = temporal.get("organoid_overlap_contract")
    if overlap != {
        "exact_model_id_overlap_audit_required": True,
        "predeclared_organoid_model_ids": {
            "path": "docs/revision_v4/organoid_model_ids_predeclared.txt",
            "sha256": "39c5fa0e2eaa0d7157732803ab4bc1f0ad81bd101a66fc09ea6bb101d6ea9035",
            "count": 256,
            "format": "utf8_one_nonempty_unique_exact_id_per_line_lf_sorted_bytewise",
            "source_path": "data/external/sanger_organoid_2026/features-v1/data_covariates_ALL-organoids.csv",
            "source_column": "sample_ID",
            "source_sha256": ORGANOID_SOURCE_SHA256,
        },
        "identifier_namespace_mismatch_action": "fail_unresolved_never_infer_zero_overlap",
        "treat_temporal_and_organoid_evidence_as_independent_if_overlap_nonzero": False,
        "nonoverlap_sensitivity_required_when_minimum_models_available": 30,
        "nonoverlap_sensitivity_minimum_eligible_genes": 5000,
        "report_overlap_counts_and_model_ids": True,
    }:
        raise TemporalValidationError("organoid overlap contract differs from v1.4")
    if temporal.get("bootstrap") != {
        "unit": "sequence_component",
        "replicates": 5000,
        "confidence_level": 0.95,
        "seed": 20261029,
    }:
        raise TemporalValidationError("temporal bootstrap contract differs from v1.4")
    campaign = lock.get("campaign_aggregation")
    expected_consensus = {
        "filename": "consensus-predictions.npz",
        "schema": CONSENSUS_PREDICTIONS_SCHEMA,
        "exact_keys": [
            "schema",
            "gene_ids",
            "canonical_gene_indices",
            "component_ids",
            "family_distance_stack_gene_mean_prediction",
            "consensus_per_job_selected_baseline_gene_mean_prediction",
            "consensus_global_train_mean_prediction",
        ],
        "completion_binding": (
            "completion.json.artifacts.consensus_predictions_records_file_bytes_"
            "sha256_and_per_array_dtype_shape_sha256"
        ),
    }
    if (
        not isinstance(campaign, Mapping)
        or campaign.get("consensus_prediction_artifact") != expected_consensus
        or campaign.get("external_outcomes_forbidden_until_consensus_artifact_is_frozen")
        is not True
    ):
        raise TemporalValidationError("frozen consensus prediction contract differs from v1.4")
    return lock


def _host_record(lock: Mapping[str, Any]) -> dict[str, str]:
    expected = lock["designated_server"]
    hostname = socket.gethostname()
    machine_path = Path("/etc/machine-id")
    if not machine_path.is_file():
        raise TemporalValidationError("designated server lacks /etc/machine-id")
    machine_digest = hashlib.sha256(machine_path.read_bytes()).hexdigest()
    if hostname != expected["hostname"] or machine_digest != expected["machine_id_sha256"]:
        raise TemporalValidationError("execution host is not the locked designated server")
    observed_ipv4: set[str] = set()
    try:
        output = subprocess.run(
            ["ip", "-o", "-4", "addr", "show"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        import re

        observed_ipv4.update(re.findall(r"\binet\s+([0-9.]+)/", output))
    except (OSError, subprocess.CalledProcessError):
        pass
    try:
        observed_ipv4.update(
            item[4][0]
            for item in socket.getaddrinfo(hostname, None, socket.AF_INET)
        )
    except OSError:
        pass
    required_ipv4 = str(expected["ipv4"])
    if required_ipv4 not in observed_ipv4:
        raise TemporalValidationError(
            f"designated server IPv4 {required_ipv4!r} is not active"
        )
    return {
        "hostname": hostname,
        "machine_id_sha256": machine_digest,
        "ipv4": required_ipv4,
    }


def _consensus_contract(lock: Mapping[str, Any]) -> Mapping[str, Any]:
    return lock["campaign_aggregation"]["consensus_prediction_artifact"]


def _verify_consensus_record(
    record: Any, *, directory: Path, lock: Mapping[str, Any]
) -> Path:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "bytes",
        "sha256",
        "schema",
        "arrays",
    }:
        raise TemporalValidationError("consensus prediction artifact record fields mismatch")
    contract = _consensus_contract(lock)
    if record.get("path") != contract["filename"]:
        raise TemporalValidationError("consensus prediction filename differs from lock")
    if record.get("schema") != contract["schema"]:
        raise TemporalValidationError("consensus prediction record schema differs from lock")
    arrays = record.get("arrays")
    if not isinstance(arrays, Mapping) or set(arrays) != set(contract["exact_keys"]):
        raise TemporalValidationError("consensus prediction array record allowlist mismatch")
    return _verify_file_record(
        {key: record[key] for key in ("path", "bytes", "sha256")},
        base=directory,
        label="campaign consensus predictions",
    )


def _load_campaign(
    campaign_dir: Path, lock: Mapping[str, Any]
) -> tuple[dict[str, Any], Path, Mapping[str, Any]]:
    if campaign_dir.is_symlink():
        raise TemporalValidationError("campaign directory cannot be a symlink")
    directory = campaign_dir.resolve(strict=True)
    if not directory.is_dir():
        raise TemporalValidationError("campaign directory is not a regular directory")
    completion_path = directory / "completion.json"
    completion = _load_json(completion_path, label="campaign completion")
    if completion.get("schema") != CAMPAIGN_COMPLETION_SCHEMA or completion.get("status") != "complete":
        raise TemporalValidationError("campaign is not complete")
    if completion.get("external_outcomes_read") is not False:
        raise TemporalValidationError("campaign completion does not certify sealed external outcomes")
    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, Mapping) or "consensus_predictions" not in artifacts:
        raise TemporalValidationError("campaign completion lacks frozen consensus predictions")
    record = artifacts["consensus_predictions"]
    consensus = _verify_consensus_record(record, directory=directory, lock=lock)
    return completion, consensus, record


def _load_consensus(
    path: Path, *, artifact_record: Mapping[str, Any], lock: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    required = set(_consensus_contract(lock)["exact_keys"])
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise TemporalValidationError("consensus NPZ has duplicate ZIP members")
            if set(names) != {f"{name}.npy" for name in required}:
                raise TemporalValidationError("consensus NPZ member allowlist mismatch")
        with np.load(path, allow_pickle=False) as source:
            if set(source.files) != required:
                raise TemporalValidationError("consensus NPZ key allowlist mismatch")
            arrays = {name: np.asarray(source[name]).copy() for name in source.files}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise TemporalValidationError(f"cannot safely load consensus NPZ: {exc}") from exc
    if any(array.dtype.hasobject for array in arrays.values()):
        raise TemporalValidationError("object arrays are forbidden in consensus NPZ")
    records = artifact_record.get("arrays")
    if not isinstance(records, Mapping) or set(records) != required:
        raise TemporalValidationError("consensus per-array records differ from exact allowlist")
    for name, array in arrays.items():
        record = records[name]
        if not isinstance(record, Mapping) or set(record) != {"dtype", "shape", "sha256"}:
            raise TemporalValidationError(f"invalid consensus array record for {name!r}")
        if record.get("dtype") != str(array.dtype) or record.get("shape") != list(array.shape):
            raise TemporalValidationError(f"consensus dtype/shape mismatch for {name!r}")
        digest = str(record.get("sha256"))
        if HEX64.fullmatch(digest) is None or hash_array(array) != digest:
            raise TemporalValidationError(f"consensus array SHA-256 mismatch for {name!r}")
    schema = arrays["schema"]
    if (
        schema.ndim != 0
        or schema.dtype.kind not in {"U", "S"}
        or str(schema.item()) != CONSENSUS_PREDICTIONS_SCHEMA
    ):
        raise TemporalValidationError("consensus schema mismatch")
    gene_ids = arrays["gene_ids"]
    canonical = arrays["canonical_gene_indices"]
    components = arrays["component_ids"]
    if gene_ids.ndim != 1 or gene_ids.size == 0 or gene_ids.dtype.kind not in {"U", "S"}:
        raise TemporalValidationError("consensus gene IDs must be one-dimensional and nonempty")
    gene_strings = [str(value) for value in gene_ids.tolist()]
    if any(not value or value != value.strip() for value in gene_strings):
        raise TemporalValidationError("consensus gene IDs contain empty or normalized values")
    if len(set(gene_strings)) != len(gene_strings):
        raise TemporalValidationError("consensus gene IDs are not unique")
    n_genes = gene_ids.size
    if (
        canonical.ndim != 1
        or canonical.size != n_genes
        or canonical.dtype.kind not in "iu"
        or np.any(canonical < 0)
        or len(set(canonical.tolist())) != n_genes
    ):
        raise TemporalValidationError("consensus canonical indices are invalid")
    if (
        components.ndim != 1
        or components.size != n_genes
        or components.dtype.kind not in {"U", "S"}
    ):
        raise TemporalValidationError("consensus component IDs do not align to genes")
    component_strings = [str(value) for value in components.tolist()]
    if any(not value or value != value.strip() for value in component_strings):
        raise TemporalValidationError("consensus component IDs contain empty or normalized values")
    for key in (
        "family_distance_stack_gene_mean_prediction",
        "consensus_per_job_selected_baseline_gene_mean_prediction",
        "consensus_global_train_mean_prediction",
    ):
        vector = arrays[key]
        if (
            vector.ndim != 1
            or vector.size != n_genes
            or vector.dtype.kind not in "fiu"
            or not np.all(np.isfinite(vector))
        ):
            raise TemporalValidationError(f"consensus prediction vector is invalid: {key}")
    return arrays


def _hash_sorted_identifiers(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _load_model_id_manifest(
    path: Path, *, label: str, require_bytewise_sorted_lf: bool = False
) -> list[str]:
    if path.is_symlink():
        raise TemporalValidationError(f"{label} cannot be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise TemporalValidationError(f"{label} must be a regular file")
    try:
        raw = resolved.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise TemporalValidationError(f"cannot read {label}: {exc}") from exc
    if require_bytewise_sorted_lf and (b"\r" in raw or not raw.endswith(b"\n")):
        raise TemporalValidationError(f"{label} must use LF lines with a final newline")
    lines = text.splitlines()
    if not lines:
        raise TemporalValidationError(f"{label} is empty")
    if any(not value or value != value.strip() for value in lines):
        raise TemporalValidationError(
            f"{label} must contain one nonempty exact ModelID per line"
        )
    if len(set(lines)) != len(lines):
        raise TemporalValidationError(f"{label} contains duplicate ModelIDs")
    if require_bytewise_sorted_lf:
        expected = sorted(lines, key=lambda value: value.encode("utf-8"))
        if lines != expected:
            raise TemporalValidationError(f"{label} is not sorted bytewise")
    return lines


def _verify_organoid_reference(
    manifest_path: Path, lock: Mapping[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    contract = lock["temporal_26q3"]["organoid_overlap_contract"][
        "predeclared_organoid_model_ids"
    ]
    if manifest_path.resolve(strict=True) != DEFAULT_ORGANOID_MODEL_IDS.resolve():
        raise TemporalValidationError(
            "organoid ModelID manifest is not the canonical predeclared path"
        )
    model_ids = _load_model_id_manifest(
        manifest_path,
        label="predeclared organoid ModelID manifest",
        require_bytewise_sorted_lf=True,
    )
    if (
        _sha256_file(manifest_path) != contract["sha256"]
        or len(model_ids) != int(contract["count"])
    ):
        raise TemporalValidationError(
            "predeclared organoid ModelID manifest differs from analysis lock"
        )
    if ORGANOID_SOURCE_CSV.is_symlink():
        raise TemporalValidationError("organoid covariate source cannot be a symlink")
    source = ORGANOID_SOURCE_CSV.resolve(strict=True)
    if (
        not source.is_file()
        or contract["source_path"] != ORGANOID_SOURCE_CSV.relative_to(ROOT).as_posix()
        or contract["source_column"] != ORGANOID_SOURCE_ID_COLUMN
        or contract["source_sha256"] != ORGANOID_SOURCE_SHA256
        or _sha256_file(source) != ORGANOID_SOURCE_SHA256
    ):
        raise TemporalValidationError("organoid covariate source SHA-256 mismatch")
    try:
        source_series = pd.read_csv(
            source,
            usecols=[ORGANOID_SOURCE_ID_COLUMN],
            dtype={ORGANOID_SOURCE_ID_COLUMN: str},
        )[ORGANOID_SOURCE_ID_COLUMN]
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        raise TemporalValidationError(
            f"cannot read locked organoid sample_ID source: {exc}"
        ) from exc
    if source_series.isna().any():
        raise TemporalValidationError("organoid source contains missing sample_ID values")
    source_ids = [str(value) for value in source_series.tolist()]
    if (
        not source_ids
        or any(not value or value != value.strip() for value in source_ids)
        or len(set(source_ids)) != len(source_ids)
    ):
        raise TemporalValidationError(
            "organoid source sample_ID values must be nonempty unique exact strings"
        )
    if set(source_ids) != set(model_ids):
        raise TemporalValidationError(
            "predeclared organoid ModelIDs differ from locked source sample_ID set"
        )
    return model_ids, {
        "manifest_count": len(model_ids),
        "sorted_model_id_set_sha256": _hash_sorted_identifiers(model_ids),
        "set_hash_algorithm": "sha256(sorted_utf8_id_plus_nul)",
        "source_csv": _file_record(source),
        "source_id_column": ORGANOID_SOURCE_ID_COLUMN,
    }


def _acquisition_paths(lock: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    temporal = lock["temporal_26q3"]
    receipt = ROOT / temporal["source_provenance_contract"]["receipt_path"]
    outcome = receipt.parent / temporal["outcome_filename"]
    ledger = ROOT / temporal["access_ledger_path"]
    project_root = ROOT.resolve(strict=True)
    for path in (outcome, receipt, ledger):
        if any(parent.is_symlink() for parent in path.parents if parent != project_root):
            raise TemporalValidationError(f"acquisition artifact parent cannot be a symlink: {path}")
        try:
            path.parent.resolve().relative_to(project_root)
        except ValueError as exc:
            raise TemporalValidationError(f"acquisition artifact escapes the project: {path}") from exc
    return outcome, receipt, ledger


def _require_no_acquisition_artifacts(lock: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    paths = _acquisition_paths(lock)
    for path in paths:
        if path.exists() or path.is_symlink():
            raise TemporalValidationError(
                f"freeze/acquisition requires absent canonical 26Q3 artifact: {path}"
            )
    return paths


def freeze_temporal_analysis(
    *,
    freeze_id: str,
    campaign_dir: Path,
    freeze_manifest: Path,
    lock_path: Path,
    protocol_path: Path,
    organoid_model_ids_path: Path,
) -> Path:
    if IDENTIFIER.fullmatch(freeze_id) is None:
        raise TemporalValidationError("invalid freeze identifier")
    lock = _load_lock(lock_path)
    host = _host_record(lock)
    if lock_path.resolve(strict=True) != DEFAULT_LOCK.resolve(strict=True):
        raise TemporalValidationError("freeze must use the canonical analysis lock")
    if protocol_path.resolve(strict=True) != DEFAULT_PROTOCOL.resolve(strict=True):
        raise TemporalValidationError("freeze must use the canonical protocol")
    _require_no_acquisition_artifacts(lock)
    completion, consensus_path, consensus_record = _load_campaign(campaign_dir, lock)
    _load_consensus(consensus_path, artifact_record=consensus_record, lock=lock)
    _, organoid_reference = _verify_organoid_reference(organoid_model_ids_path, lock)
    if freeze_manifest.is_symlink():
        raise TemporalValidationError("temporal freeze manifest cannot be a symlink")
    manifest_path = freeze_manifest.resolve()
    try:
        manifest_path.relative_to(ROOT.resolve(strict=True))
    except ValueError as exc:
        raise TemporalValidationError("freeze manifest must reside inside the project") from exc
    if manifest_path in _acquisition_paths(lock):
        raise TemporalValidationError("freeze manifest cannot use a 26Q3 acquisition artifact path")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists() or manifest_path.is_symlink():
        raise TemporalValidationError("refusing to overwrite temporal freeze manifest")
    campaign_directory = campaign_dir.resolve(strict=True)
    payload = {
        "schema": FREEZE_SCHEMA,
        "status": "locked_before_26q3_access",
        "freeze_id": freeze_id,
        "created_at_utc": _utc_now(),
        "outcome_accessed": False,
        "designated_server": host,
        "frozen_files": {
            "analysis_lock": _file_record(lock_path),
            "protocol": _file_record(protocol_path),
            "validator_library": _file_record(Path(__file__).with_name("temporal_26q3_validation.py")),
            "validator_runner": _file_record(Path(__file__)),
            "campaign_completion": _file_record(campaign_directory / "completion.json"),
            "campaign_consensus": _file_record(consensus_path),
            "organoid_model_ids": _file_record(organoid_model_ids_path),
        },
        "campaign": {
            "campaign_id": completion.get("campaign_id"),
            "directory": campaign_directory.relative_to(ROOT.resolve(strict=True)).as_posix(),
        },
        "locked_temporal_contract": lock["temporal_26q3"],
        "locked_temporal_gates": lock["gates"]["temporal_26q3"],
        "organoid_model_id_reference": organoid_reference,
    }
    _require_no_acquisition_artifacts(lock)
    with manifest_path.open("xb") as handle:
        handle.write(_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())
    return manifest_path


def _verify_freeze(
    path: Path, lock: Mapping[str, Any]
) -> tuple[dict[str, Any], Path, Path, Mapping[str, Any]]:
    if path.is_symlink():
        raise TemporalValidationError("temporal freeze manifest cannot be a symlink")
    try:
        path.resolve(strict=True).relative_to(ROOT.resolve(strict=True))
    except ValueError as exc:
        raise TemporalValidationError("freeze manifest must reside inside the project") from exc
    freeze = _load_json(path, label="temporal freeze")
    if freeze.get("schema") != FREEZE_SCHEMA or freeze.get("status") != "locked_before_26q3_access":
        raise TemporalValidationError("temporal freeze state mismatch")
    if freeze.get("outcome_accessed") is not False:
        raise TemporalValidationError("temporal freeze was not created before outcome access")
    if freeze.get("locked_temporal_contract") != lock["temporal_26q3"]:
        raise TemporalValidationError("frozen temporal contract differs from current lock")
    if freeze.get("locked_temporal_gates") != lock["gates"]["temporal_26q3"]:
        raise TemporalValidationError("frozen temporal gates differ from current lock")
    records = freeze.get("frozen_files")
    if not isinstance(records, Mapping):
        raise TemporalValidationError("temporal freeze file records are missing")
    expected = {
        "analysis_lock",
        "protocol",
        "validator_library",
        "validator_runner",
        "campaign_completion",
        "campaign_consensus",
        "organoid_model_ids",
    }
    if set(records) != expected:
        raise TemporalValidationError("temporal freeze file allowlist mismatch")
    paths = {
        name: _verify_file_record(record, base=ROOT, label=f"frozen {name}")
        for name, record in records.items()
    }
    if paths["analysis_lock"] != DEFAULT_LOCK.resolve(strict=True):
        raise TemporalValidationError("freeze does not bind the canonical v1.4 lock")
    if paths["protocol"] != DEFAULT_PROTOCOL.resolve(strict=True):
        raise TemporalValidationError("freeze does not bind the canonical v1.4 protocol")
    if paths["validator_library"] != Path(__file__).with_name("temporal_26q3_validation.py").resolve(strict=True):
        raise TemporalValidationError("freeze validator-library identity mismatch")
    if paths["validator_runner"] != Path(__file__).resolve(strict=True):
        raise TemporalValidationError("freeze validator-runner identity mismatch")
    if paths["organoid_model_ids"] != DEFAULT_ORGANOID_MODEL_IDS.resolve(strict=True):
        raise TemporalValidationError("freeze does not bind the canonical organoid ModelID manifest")
    organoid_ids = _load_model_id_manifest(
        paths["organoid_model_ids"],
        label="frozen organoid ModelID manifest",
        require_bytewise_sorted_lf=True,
    )
    locked_organoid = lock["temporal_26q3"]["organoid_overlap_contract"][
        "predeclared_organoid_model_ids"
    ]
    organoid_file_record = records["organoid_model_ids"]
    if (
        organoid_file_record.get("path") != locked_organoid["path"]
        or organoid_file_record.get("sha256") != locked_organoid["sha256"]
        or len(organoid_ids) != int(locked_organoid["count"])
    ):
        raise TemporalValidationError("frozen organoid ModelID manifest differs from lock")
    organoid_reference = freeze.get("organoid_model_id_reference")
    if not isinstance(organoid_reference, Mapping):
        raise TemporalValidationError("freeze lacks organoid ModelID provenance")
    expected_source_path = ORGANOID_SOURCE_CSV.relative_to(ROOT).as_posix()
    source_record = organoid_reference.get("source_csv")
    if (
        organoid_reference.get("manifest_count") != len(organoid_ids)
        or organoid_reference.get("sorted_model_id_set_sha256")
        != _hash_sorted_identifiers(organoid_ids)
        or organoid_reference.get("set_hash_algorithm")
        != "sha256(sorted_utf8_id_plus_nul)"
        or organoid_reference.get("source_id_column") != ORGANOID_SOURCE_ID_COLUMN
        or not isinstance(source_record, Mapping)
        or set(source_record) != {"path", "bytes", "sha256"}
        or source_record.get("path") != expected_source_path
        or source_record.get("sha256") != ORGANOID_SOURCE_SHA256
        or isinstance(source_record.get("bytes"), bool)
        or not isinstance(source_record.get("bytes"), int)
        or int(source_record["bytes"]) <= 0
        or locked_organoid["source_path"] != expected_source_path
        or locked_organoid["source_column"] != ORGANOID_SOURCE_ID_COLUMN
        or locked_organoid["source_sha256"] != ORGANOID_SOURCE_SHA256
    ):
        raise TemporalValidationError("frozen organoid ModelID provenance mismatch")
    completion = _load_json(paths["campaign_completion"], label="frozen campaign completion")
    if (
        completion.get("schema") != CAMPAIGN_COMPLETION_SCHEMA
        or completion.get("status") != "complete"
        or completion.get("external_outcomes_read") is not False
    ):
        raise TemporalValidationError("frozen campaign completion is invalid")
    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, Mapping) or "consensus_predictions" not in artifacts:
        raise TemporalValidationError("frozen completion lacks consensus predictions")
    consensus_record = artifacts["consensus_predictions"]
    consensus_path = _verify_consensus_record(
        consensus_record,
        directory=paths["campaign_completion"].parent,
        lock=lock,
    )
    if consensus_path != paths["campaign_consensus"]:
        raise TemporalValidationError("freeze consensus path differs from campaign completion")
    return freeze, consensus_path, paths["organoid_model_ids"], consensus_record


def _append_ledger(path: Path, event: Mapping[str, Any]) -> None:
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise TemporalValidationError("access ledger cannot be a symlink")
    payload = json.dumps(_json_safe(event), sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    descriptor = os.open(
        path,
        os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "a", encoding="utf-8", closefd=True) as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    import fcntl

    if path.is_symlink():
        raise TemporalValidationError("access ledger cannot be a symlink")
    if not path.exists():
        return []
    if not path.is_file() or path.is_symlink():
        raise TemporalValidationError("access ledger is not a regular file")
    events: list[dict[str, Any]] = []
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "r", encoding="utf-8", closefd=True) as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        for line_number, line in enumerate(handle, start=1):
            try:
                event = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
            except (json.JSONDecodeError, TypeError) as exc:
                raise TemporalValidationError(
                    f"invalid access ledger line {line_number}: {exc}"
                ) from exc
            if not isinstance(event, dict):
                raise TemporalValidationError(
                    f"access ledger line {line_number} is not an object"
                )
            if event.get("schema") != "DepPLM-temporal-access-event-v1":
                raise TemporalValidationError(
                    f"access ledger line {line_number} has an invalid schema"
                )
            events.append(event)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return events


def _utc_attestation(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TemporalValidationError(f"{label} must be a nonempty UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TemporalValidationError(f"{label} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise TemporalValidationError(f"{label} must have an explicit UTC offset")
    if parsed > datetime.now(timezone.utc):
        raise TemporalValidationError(f"{label} cannot be in the future")
    return parsed


def _acquisition_binding(
    *, freeze: Mapping[str, Any], outcome: Path, receipt_path: Path
) -> dict[str, Any]:
    return {
        "freeze_id": freeze["freeze_id"],
        "freeze_manifest_sha256": _sha256_file(Path(freeze["__path"])),
        "outcome_path": outcome.relative_to(ROOT.resolve(strict=True)).as_posix(),
        "source_receipt_path": receipt_path.relative_to(ROOT.resolve(strict=True)).as_posix(),
    }


def _verify_acquisition_event(
    *, ledger: Path, freeze: Mapping[str, Any], outcome: Path, receipt_path: Path,
    events: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    records = _read_ledger(ledger) if events is None else events
    if not records:
        raise TemporalValidationError("pre-acquisition ledger event is missing")
    first = records[0]
    if (
        first.get("schema") != "DepPLM-temporal-access-event-v1"
        or first.get("event") != "acquisition_prepared_before_outcome_bytes"
        or set(first) != {
            "schema", "event", "recorded_at_utc", "acquisition_id",
            "freeze_id", "freeze_manifest_sha256", "outcome_path", "source_receipt_path",
        }
        or not isinstance(first.get("acquisition_id"), str)
        or HEX32.fullmatch(first["acquisition_id"]) is None
    ):
        raise TemporalValidationError("access ledger does not begin with a valid pre-acquisition event")
    for key, expected in _acquisition_binding(
        freeze=freeze, outcome=outcome, receipt_path=receipt_path
    ).items():
        if first.get(key) != expected:
            raise TemporalValidationError(f"pre-acquisition ledger binds another {key}")
    freeze_at = _utc_attestation(freeze["created_at_utc"], label="freeze time")
    acquisition_at = _utc_attestation(first["recorded_at_utc"], label="acquisition time")
    if not freeze_at < acquisition_at:
        raise TemporalValidationError("freeze must precede pre-acquisition ledger event")
    return dict(first)


def prepare_temporal_acquisition(*, freeze_manifest: Path, lock_path: Path) -> Path:
    lock = _load_lock(lock_path)
    if lock_path.resolve(strict=True) != DEFAULT_LOCK.resolve(strict=True):
        raise TemporalValidationError("acquisition must use the canonical analysis lock")
    _host_record(lock)
    freeze, _, _, _ = _verify_freeze(freeze_manifest, lock)
    freeze_path = freeze_manifest.resolve(strict=True)
    try:
        freeze_path.relative_to(ROOT.resolve(strict=True))
    except ValueError as exc:
        raise TemporalValidationError("freeze manifest must reside inside the project") from exc
    freeze = {**freeze, "__path": str(freeze_path)}
    outcome, receipt_path, ledger = _require_no_acquisition_artifacts(lock)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    freeze_at = _utc_attestation(freeze["created_at_utc"], label="freeze time")
    binding = _acquisition_binding(
        freeze=freeze, outcome=outcome, receipt_path=receipt_path
    )
    _require_no_acquisition_artifacts(lock)
    prepared_at = datetime.now(timezone.utc)
    if not freeze_at < prepared_at:
        raise TemporalValidationError("freeze must precede pre-acquisition ledger event")
    event = {
        "schema": "DepPLM-temporal-access-event-v1",
        "event": "acquisition_prepared_before_outcome_bytes",
        "recorded_at_utc": prepared_at.isoformat(),
        "acquisition_id": uuid.uuid4().hex,
        **binding,
    }
    descriptor = os.open(
        ledger, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600
    )
    with os.fdopen(descriptor, "w", encoding="utf-8", closefd=True) as handle:
        handle.write(json.dumps(event, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return ledger


def _official_source_url(value: Any, *, label: str, allowed_hosts: Sequence[str]) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TemporalValidationError(f"{label} is missing")
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise TemporalValidationError(f"{label} is malformed") from exc
    if (
        parsed.scheme != "https"
        or not host
        or not any(host == allowed or host.endswith("." + allowed) for allowed in allowed_hosts)
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
        or parsed.fragment
        or not parsed.path or parsed.path == "/"
    ):
        raise TemporalValidationError(f"{label} is not an allowed official HTTPS URL")


def _verify_source_receipt(
    *, outcome: Path, contract: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    receipt_path = ROOT / str(contract["receipt_path"])
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise TemporalValidationError("26Q3 source receipt must be a regular non-symlink file")
    if outcome != receipt_path.parent.resolve(strict=True) / str(contract["outcome_filename"]):
        raise TemporalValidationError("26Q3 outcome must be next to the locked source receipt")
    receipt_record = _file_record(receipt_path)
    if receipt_record["path"] != contract["receipt_path"]:
        raise TemporalValidationError("26Q3 source receipt resolves outside its locked path")
    receipt = _load_json(receipt_path, label="26Q3 source receipt")
    if _file_record(receipt_path) != receipt_record:
        raise TemporalValidationError("26Q3 source receipt changed during verification")
    if not set(contract["required_fields"]).issubset(receipt):
        raise TemporalValidationError("26Q3 source receipt lacks required provenance fields")
    if receipt["release_id"] != contract["release_id"]:
        raise TemporalValidationError("26Q3 source receipt release ID differs from lock")
    allowed_hosts = contract["allowed_url_hosts"]
    for field in ("official_release_url", "official_file_url"):
        _official_source_url(receipt[field], label=field, allowed_hosts=allowed_hosts)
    archive_file_id = receipt["archive_file_id"]
    if (
        isinstance(archive_file_id, bool)
        or not isinstance(archive_file_id, (str, int))
        or not str(archive_file_id).strip()
        or str(archive_file_id) != str(archive_file_id).strip()
    ):
        raise TemporalValidationError("26Q3 source receipt archive file ID is invalid")
    _utc_attestation(receipt["retrieved_at_utc"], label="26Q3 retrieval time")
    byte_count = receipt["bytes"]
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 1:
        raise TemporalValidationError("26Q3 source receipt byte count is invalid")
    if not isinstance(receipt["sha256"], str) or HEX64.fullmatch(receipt["sha256"]) is None:
        raise TemporalValidationError("26Q3 source receipt SHA-256 is invalid")
    manual = receipt.get("manual_official_metadata_verification")
    if (
        not isinstance(manual, Mapping)
        or set(manual) != {
            "verified", "verified_at_utc", "verifier", "official_release_url",
            "official_file_url", "archive_file_id",
        }
        or manual["verified"] is not True
        or not isinstance(manual["verifier"], str)
        or not manual["verifier"].strip()
        or manual["verifier"] != manual["verifier"].strip()
        or any(
            manual[field] != receipt[field]
            for field in ("official_release_url", "official_file_url", "archive_file_id")
        )
    ):
        raise TemporalValidationError("26Q3 source receipt lacks matching manual metadata verification")
    _utc_attestation(manual["verified_at_utc"], label="26Q3 metadata verification time")
    return receipt, receipt_record, receipt_path


def _record_access_start(
    *, ledger: Path, freeze: Mapping[str, Any], outcome: Path, attempt_id: str,
    source_receipt: Mapping[str, Any], source_receipt_record: Mapping[str, Any],
    source_receipt_path: Path,
) -> None:
    import fcntl

    stat = outcome.stat()
    binding = {
        **_acquisition_binding(
            freeze=freeze, outcome=outcome, receipt_path=source_receipt_path
        ),
        "source_receipt": dict(source_receipt_record),
        "source_outcome_sha256": source_receipt["sha256"],
        "source_outcome_bytes": source_receipt["bytes"],
    }
    if ledger.is_symlink():
        raise TemporalValidationError("access ledger cannot be a symlink")
    descriptor = os.open(
        ledger,
        os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
    )
    with os.fdopen(descriptor, "r+", encoding="utf-8", closefd=True) as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        existing: list[dict[str, Any]] = []
        handle.seek(0)
        for line_number, line in enumerate(handle, start=1):
            try:
                event = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
            except (json.JSONDecodeError, TypeError) as exc:
                raise TemporalValidationError(
                    f"invalid access ledger line {line_number}: {exc}"
                ) from exc
            if (
                not isinstance(event, dict)
                or event.get("schema") != "DepPLM-temporal-access-event-v1"
            ):
                raise TemporalValidationError(
                    f"access ledger line {line_number} has an invalid record"
                )
            existing.append(event)
        first = _verify_acquisition_event(
            ledger=ledger, freeze=freeze, outcome=outcome,
            receipt_path=source_receipt_path, events=existing,
        )
        acquisition_at = _utc_attestation(first["recorded_at_utc"], label="acquisition time")
        retrieved_at = _utc_attestation(source_receipt["retrieved_at_utc"], label="26Q3 retrieval time")
        manual_at = _utc_attestation(
            source_receipt["manual_official_metadata_verification"]["verified_at_utc"],
            label="26Q3 metadata verification time",
        )
        validated_at = datetime.now(timezone.utc)
        if not acquisition_at < retrieved_at <= manual_at <= validated_at:
            raise TemporalValidationError(
                "26Q3 chronology must be freeze < acquisition < retrieval <= manual verification <= validation"
            )
        for event in existing[1:]:
            if event.get("event") == "validation_attempt_started_before_outcome_open":
                if any(event.get(key) != expected for key, expected in binding.items()):
                    raise TemporalValidationError(
                        "existing validation attempt binds another source receipt or outcome hash"
                    )
                if event.get("acquisition_id") != first["acquisition_id"]:
                    raise TemporalValidationError("existing validation attempt binds another acquisition")
        event = {
            "schema": "DepPLM-temporal-access-event-v1",
            "event": "validation_attempt_started_before_outcome_open",
            "recorded_at_utc": validated_at.isoformat(),
            "attempt_id": attempt_id,
            "acquisition_id": first["acquisition_id"],
            **binding,
            "outcome_size_from_stat": int(stat.st_size),
            "outcome_mtime_ns_from_stat": int(stat.st_mtime_ns),
        }
        handle.seek(0, os.SEEK_END)
        handle.write(json.dumps(event, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    normalized = {key: np.ascontiguousarray(np.asarray(value)) for key, value in arrays.items()}
    if any(value.dtype.hasobject for value in normalized.values()):
        raise TemporalValidationError("object arrays are forbidden in temporal evidence")
    with path.open("xb") as handle:
        np.savez_compressed(handle, **normalized)
        handle.flush()
        os.fsync(handle.fileno())


def _npz_artifact_record(
    path: Path, *, arrays: Mapping[str, np.ndarray], relative_to: Path, schema: str
) -> dict[str, Any]:
    return {
        **_file_record(path, relative_to=relative_to),
        "schema": schema,
        "arrays": {
            name: {
                "dtype": str(np.asarray(value).dtype),
                "shape": list(np.asarray(value).shape),
                "sha256": hash_array(np.asarray(value)),
            }
            for name, value in sorted(arrays.items())
        },
    }


def _read_temporal_csv(
    path: Path, *, gene_header_pattern: str
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            raw_header = next(csv.reader(handle))
    except (OSError, UnicodeError, csv.Error, StopIteration) as exc:
        raise TemporalValidationError(f"cannot read 26Q3 CSV header: {exc}") from exc
    if len(raw_header) < 2 or raw_header[0] != "ModelID":
        raise TemporalValidationError(
            "26Q3 CSV first column must be exactly 'ModelID' and include gene columns"
        )
    if len(set(raw_header)) != len(raw_header):
        raise TemporalValidationError("26Q3 CSV contains duplicate raw column headers")
    symbols, entrez_ids, gene_positions = parse_gene_headers(
        raw_header[1:], gene_header_pattern
    )
    try:
        frame = pd.read_csv(
            path,
            index_col=0,
            dtype={"ModelID": str},
            encoding="utf-8-sig",
            low_memory=False,
        )
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
        raise TemporalValidationError(f"cannot parse 26Q3 outcome CSV: {exc}") from exc
    if frame.index.name != "ModelID" or frame.columns.tolist() != raw_header[1:]:
        raise TemporalValidationError(
            "parsed 26Q3 table does not preserve the exact raw header"
        )
    return frame, symbols, entrez_ids, gene_positions, raw_header


def run_temporal_validation(
    *,
    analysis_id: str,
    freeze_manifest: Path,
    outcome_path: Path,
    output_root: Path,
    lock_path: Path,
) -> Path:
    if IDENTIFIER.fullmatch(analysis_id) is None:
        raise TemporalValidationError("invalid analysis identifier")
    lock = _load_lock(lock_path)
    if lock_path.resolve(strict=True) != DEFAULT_LOCK.resolve(strict=True):
        raise TemporalValidationError("validation must use the canonical analysis lock")
    host = _host_record(lock)
    freeze, consensus_path, organoid_manifest, consensus_record = _verify_freeze(
        freeze_manifest, lock
    )
    freeze = {**freeze, "__path": str(freeze_manifest.resolve(strict=True))}
    consensus = _load_consensus(
        consensus_path, artifact_record=consensus_record, lock=lock
    )
    organoid_model_ids = _load_model_id_manifest(
        organoid_manifest,
        label="frozen organoid ModelID manifest",
        require_bytewise_sorted_lf=True,
    )
    temporal = lock["temporal_26q3"]
    cohort_contract = temporal["cohort"]
    overlap_contract = temporal["organoid_overlap_contract"]
    baseline_record = cohort_contract["baseline_model_ids_source"]
    baseline_unresolved = ROOT / str(baseline_record["path"])
    if baseline_unresolved.is_symlink():
        raise TemporalValidationError("24Q4 baseline ModelID manifest cannot be a symlink")
    baseline_path = baseline_unresolved.resolve(strict=True)
    if (
        not baseline_path.is_file()
        or _sha256_file(baseline_path) != baseline_record["sha256"]
    ):
        raise TemporalValidationError("24Q4 baseline ModelID manifest hash mismatch")
    baseline_models = _load_model_id_manifest(
        baseline_path, label="24Q4 baseline ModelID manifest"
    )
    gene_record = lock["locked_assets"]["gene_manifest"]
    matrix_record = lock["locked_assets"]["dependency_matrix"]
    if gene_record["path"] != "data/processed/genes.txt" or matrix_record["path"] != "data/processed/gene_effect.npy":
        raise TemporalValidationError("24Q4 gene-effect assets differ from v1.4")
    gene_path = ROOT / gene_record["path"]
    matrix_path = ROOT / matrix_record["path"]
    for path, record, label in (
        (gene_path, gene_record, "24Q4 gene manifest"),
        (matrix_path, matrix_record, "24Q4 gene-effect matrix"),
    ):
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != record["sha256"]:
            raise TemporalValidationError(f"{label} is missing or differs from lock")
    baseline_genes = _load_model_id_manifest(gene_path, label="24Q4 gene manifest")

    if outcome_path.is_symlink():
        raise TemporalValidationError("26Q3 outcome cannot be a symlink")
    outcome = outcome_path.resolve(strict=True)
    if outcome.name != lock["temporal_26q3"]["outcome_filename"]:
        raise TemporalValidationError("outcome filename differs from lock")
    if not outcome.is_file():
        raise TemporalValidationError("26Q3 outcome must be a regular non-symlink file")
    project_root = ROOT.resolve(strict=True)
    try:
        outcome.relative_to(project_root)
    except ValueError as exc:
        raise TemporalValidationError("26Q3 outcome must reside inside the project") from exc
    canonical_outcome, canonical_receipt, ledger = _acquisition_paths(lock)
    if outcome != canonical_outcome:
        raise TemporalValidationError("26Q3 outcome differs from the canonical acquisition path")
    acquisition_event = _verify_acquisition_event(
        ledger=ledger, freeze=freeze, outcome=outcome, receipt_path=canonical_receipt
    )
    source_receipt, source_receipt_record, source_receipt_path = _verify_source_receipt(
        outcome=outcome,
        contract=temporal["source_provenance_contract"],
    )

    if output_root.is_symlink():
        raise TemporalValidationError("temporal output root cannot be a symlink")
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_output_root = output_root.resolve(strict=True)
    try:
        resolved_output_root.relative_to(project_root)
    except ValueError as exc:
        raise TemporalValidationError("temporal output root must reside inside the project") from exc
    destination = resolved_output_root / analysis_id
    if destination.exists() or destination.is_symlink():
        raise TemporalValidationError("refusing to overwrite immutable temporal output")
    attempt_id = uuid.uuid4().hex
    _record_access_start(
        ledger=ledger, freeze=freeze, outcome=outcome, attempt_id=attempt_id,
        source_receipt=source_receipt, source_receipt_record=source_receipt_record,
        source_receipt_path=source_receipt_path,
    )
    try:
        outcome_sha256 = _sha256_file(outcome)
        if (
            outcome.stat().st_size != source_receipt["bytes"]
            or outcome_sha256 != source_receipt["sha256"]
        ):
            raise TemporalValidationError("26Q3 outcome bytes differ from source receipt")
        previous_snapshots = [
            event
            for event in _read_ledger(ledger)
            if event.get("event") == "input_snapshot_bound_before_csv_parse"
        ]
        if any(
            event.get("outcome_sha256") != outcome_sha256
            for event in previous_snapshots
        ):
            raise TemporalValidationError(
                "26Q3 outcome bytes differ from the first ledger-bound input snapshot"
            )
        _append_ledger(
            ledger,
            {
                "schema": "DepPLM-temporal-access-event-v1",
                "event": "input_snapshot_bound_before_csv_parse",
                "recorded_at_utc": _utc_now(),
                "attempt_id": attempt_id,
                "outcome_sha256": outcome_sha256,
                "outcome_bytes": outcome.stat().st_size,
            },
        )

        frame, symbols, entrez_ids, gene_positions, raw_header = _read_temporal_csv(
            outcome, gene_header_pattern=str(cohort_contract["gene_header_regex"])
        )
        new_models, model_positions = select_new_models(
            frame.index.tolist(),
            baseline_models,
            minimum_new_models=int(cohort_contract["minimum_new_models"]),
        )
        try:
            full_outcomes = (
                frame.iloc[:, gene_positions]
                .apply(pd.to_numeric, errors="raise")
                .to_numpy(dtype=np.float64, copy=True)
            )
            values = full_outcomes[model_positions, :]
        except (TypeError, ValueError) as exc:
            raise TemporalValidationError(
                f"26Q3 matched gene columns contain a nonnumeric value: {exc}"
            ) from exc
        if (
            _sha256_file(outcome) != outcome_sha256
            or _file_record(source_receipt_path) != source_receipt_record
        ):
            raise TemporalValidationError("26Q3 outcome bytes changed during parsing")
        _append_ledger(
            ledger,
            {
                "schema": "DepPLM-temporal-access-event-v1",
                "event": "input_snapshot_verified_after_csv_parse",
                "recorded_at_utc": _utc_now(),
                "attempt_id": attempt_id,
                "outcome_sha256": outcome_sha256,
            },
        )

        cohort = align_temporal_cohort(
            outcome_model_by_gene=values,
            outcome_gene_symbols=symbols,
            new_model_ids=new_models,
            consensus_gene_ids=consensus["gene_ids"],
            consensus_canonical_gene_indices=consensus["canonical_gene_indices"],
            consensus_component_ids=consensus["component_ids"],
            primary_prediction=consensus["family_distance_stack_gene_mean_prediction"],
            selected_baseline_prediction=consensus[
                "consensus_per_job_selected_baseline_gene_mean_prediction"
            ],
            global_prediction=consensus["consensus_global_train_mean_prediction"],
            minimum_finite_fraction=float(
                cohort_contract["minimum_finite_model_fraction_per_gene"]
            ),
            minimum_eligible_genes=int(cohort_contract["minimum_eligible_genes"]),
        )
        primary_metrics = evaluate_temporal_cohort(
            cohort,
            bootstrap_contract=temporal["bootstrap"],
            gate_contract=lock["gates"]["temporal_26q3"],
        )
        try:
            baseline_matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise TemporalValidationError(f"cannot read locked 24Q4 gene effects: {exc}") from exc
        correction_contract = temporal["shared_model_pipeline_correction_contract"]
        correction_metrics, correction_evidence = evaluate_shared_model_pipeline_correction(
            primary_cohort=cohort,
            outcome_model_ids=frame.index.tolist(),
            outcome_gene_symbols=symbols,
            outcome_model_by_gene=full_outcomes,
            baseline_model_ids=baseline_models,
            baseline_gene_symbols=baseline_genes,
            baseline_gene_by_model=baseline_matrix,
            minimum_finite_paired_shared_models_per_gene=int(
                correction_contract["minimum_finite_paired_shared_models_per_gene"]
            ),
            minimum_eligible_genes=int(correction_contract["minimum_eligible_genes"]),
            bootstrap_contract=temporal["bootstrap"],
        )
        if _sha256_file(matrix_path) != matrix_record["sha256"]:
            raise TemporalValidationError("24Q4 gene-effect matrix changed during correction")

        overlap_audit = audit_organoid_overlap(new_models, organoid_model_ids)
        overlap_report = {
            key: value
            for key, value in overlap_audit.items()
            if key != "nonoverlap_temporal_positions"
        }
        nonoverlap_cohort = None
        nonoverlap_metrics: dict[str, Any]
        minimum_nonoverlap_models = int(
            overlap_contract["nonoverlap_sensitivity_required_when_minimum_models_available"]
        )
        minimum_nonoverlap_genes = int(
            overlap_contract["nonoverlap_sensitivity_minimum_eligible_genes"]
        )
        overlap_resolved = overlap_audit["audit_status"] == "resolved_exact_model_id_audit"
        if not overlap_resolved:
            nonoverlap_metrics = {
                "status": "blocked_identifier_namespace_unresolved",
                "requirement_triggered": False,
                "n_nonoverlap_models_from_raw_exact_comparison": int(
                    overlap_audit["n_nonoverlap_temporal_models"]
                ),
                "reason": (
                    "exact overlap cannot be interpreted as zero because the frozen "
                    "identifier namespaces are not resolved"
                ),
            }
        elif int(overlap_audit["n_nonoverlap_temporal_models"]) < minimum_nonoverlap_models:
            nonoverlap_metrics = {
                "status": "not_required_insufficient_nonoverlap_models",
                "requirement_triggered": False,
                "n_nonoverlap_models": int(
                    overlap_audit["n_nonoverlap_temporal_models"]
                ),
                "minimum_required_models": minimum_nonoverlap_models,
            }
        else:
            nonoverlap_positions = np.asarray(
                overlap_audit["nonoverlap_temporal_positions"], dtype=np.int64
            )
            nonoverlap_cohort = align_temporal_cohort(
                outcome_model_by_gene=values[nonoverlap_positions, :],
                outcome_gene_symbols=symbols,
                new_model_ids=np.asarray(
                    overlap_audit["nonoverlap_temporal_model_ids"]
                ),
                consensus_gene_ids=consensus["gene_ids"],
                consensus_canonical_gene_indices=consensus["canonical_gene_indices"],
                consensus_component_ids=consensus["component_ids"],
                primary_prediction=consensus[
                    "family_distance_stack_gene_mean_prediction"
                ],
                selected_baseline_prediction=consensus[
                    "consensus_per_job_selected_baseline_gene_mean_prediction"
                ],
                global_prediction=consensus[
                    "consensus_global_train_mean_prediction"
                ],
                minimum_finite_fraction=float(
                    cohort_contract["minimum_finite_model_fraction_per_gene"]
                ),
                minimum_eligible_genes=0,
            )
            if nonoverlap_cohort.gene_ids.size < minimum_nonoverlap_genes:
                nonoverlap_metrics = {
                    "status": "not_required_insufficient_eligible_genes",
                    "requirement_triggered": False,
                    "n_nonoverlap_models": int(nonoverlap_cohort.new_model_ids.size),
                    "n_eligible_genes": int(nonoverlap_cohort.gene_ids.size),
                    "minimum_required_genes": minimum_nonoverlap_genes,
                }
            else:
                nonoverlap_metrics = {
                    "status": "complete",
                    "requirement_triggered": True,
                    "analysis": evaluate_temporal_cohort(
                        nonoverlap_cohort,
                        bootstrap_contract=temporal["bootstrap"],
                        gate_contract=lock["gates"]["temporal_26q3"],
                    ),
                }

        metrics = {
            "schema": RESULT_SCHEMA,
            "analysis_id": analysis_id,
            "release": temporal["release"],
            "primary_method": temporal["primary_method_prediction"],
            "comparison": temporal["comparison_prediction"],
            "reference": temporal["reference_prediction"],
            "prediction_source_schema": CONSENSUS_PREDICTIONS_SCHEMA,
            "primary_post_access_refit_or_recalibration": False,
            "secondary_shared_model_correction_uses_26q3_outcomes": True,
            "primary_temporal_cohort": primary_metrics,
            "shared_model_pipeline_correction_sensitivity": correction_metrics,
            "organoid_overlap_audit": overlap_report,
            "nonoverlap_sensitivity": nonoverlap_metrics,
            "interpretation_constraints": {
                "new_model_cohort_is_primary": True,
                "shared_model_correction_cannot_replace_primary": True,
                "source_receipt_is_recorded_provenance_not_authenticity_proof": True,
                "nonoverlap_sensitivity_cannot_replace_primary": True,
                "temporal_and_organoid_independence_claim_forbidden": bool(
                    overlap_audit["independence_claim_forbidden"]
                ),
                "failed_gates_retained_without_tuning": True,
            },
        }
        staging = resolved_output_root / f".{analysis_id}.partial-{uuid.uuid4().hex}"
        staging.mkdir(mode=0o700)
        try:
            (staging / "metrics.json").write_bytes(_json_bytes(metrics))
            evidence = {
                "schema": np.asarray(EVIDENCE_SCHEMA),
                "gene_ids": cohort.gene_ids,
                "canonical_gene_indices": cohort.canonical_gene_indices,
                "component_ids": cohort.component_ids,
                "new_model_ids": cohort.new_model_ids,
                "new_model_source_row_indices": model_positions,
                "observed_gene_mean": cohort.observed_gene_mean,
                "family_distance_stack_gene_mean_prediction": cohort.primary_prediction,
                "consensus_per_job_selected_baseline_gene_mean_prediction": cohort.selected_baseline_prediction,
                "consensus_global_train_mean_prediction": cohort.global_prediction,
                "finite_model_fraction": cohort.finite_model_fraction,
                "all_parsed_gene_symbols": symbols,
                "all_parsed_entrez_ids": entrez_ids,
                "all_parsed_gene_column_indices": gene_positions,
                "organoid_overlap_model_ids": np.asarray(
                    overlap_audit["overlap_model_ids"]
                ),
                "organoid_raw_exact_nonoverlap_temporal_model_ids": np.asarray(
                    overlap_audit["nonoverlap_temporal_model_ids"]
                ),
            }
            evidence.update(correction_evidence)
            if nonoverlap_cohort is not None:
                evidence.update(
                    {
                        "nonoverlap_gene_ids": nonoverlap_cohort.gene_ids,
                        "nonoverlap_canonical_gene_indices": (
                            nonoverlap_cohort.canonical_gene_indices
                        ),
                        "nonoverlap_component_ids": nonoverlap_cohort.component_ids,
                        "nonoverlap_new_model_ids": nonoverlap_cohort.new_model_ids,
                        "nonoverlap_observed_gene_mean": (
                            nonoverlap_cohort.observed_gene_mean
                        ),
                        "nonoverlap_family_distance_stack_gene_mean_prediction": (
                            nonoverlap_cohort.primary_prediction
                        ),
                        "nonoverlap_consensus_per_job_selected_baseline_gene_mean_prediction": (
                            nonoverlap_cohort.selected_baseline_prediction
                        ),
                        "nonoverlap_consensus_global_train_mean_prediction": (
                            nonoverlap_cohort.global_prediction
                        ),
                        "nonoverlap_finite_model_fraction": (
                            nonoverlap_cohort.finite_model_fraction
                        ),
                    }
                )
            _write_npz(staging / "temporal-evidence.npz", evidence)
            baseline_gene_file_record = _file_record(gene_path)
            baseline_matrix_file_record = _file_record(matrix_path)
            if (
                baseline_gene_file_record["sha256"] != gene_record["sha256"]
                or baseline_matrix_file_record["sha256"] != matrix_record["sha256"]
                or _file_record(source_receipt_path) != source_receipt_record
                or _verify_acquisition_event(
                    ledger=ledger, freeze=freeze, outcome=outcome,
                    receipt_path=source_receipt_path,
                ) != acquisition_event
            ):
                raise TemporalValidationError("temporal input provenance changed before output binding")
            inputs = {
                "schema": "DepPLM-temporal-26q3-inputs-v1",
                "analysis_id": analysis_id,
                "attempt_id": attempt_id,
                "designated_server": host,
                "freeze_manifest": _file_record(freeze_manifest),
                "pre_acquisition_ledger_event": {
                    "ledger_path": ledger.relative_to(project_root).as_posix(),
                    "record": acquisition_event,
                },
                "campaign_consensus_project_file": _file_record(consensus_path),
                "campaign_consensus_completion_record": consensus_record,
                "outcome": {
                    "path": outcome.relative_to(project_root).as_posix(),
                    "bytes": outcome.stat().st_size,
                    "sha256": outcome_sha256,
                },
                "source_receipt": source_receipt_record,
                "source_receipt_manual_verification": source_receipt[
                    "manual_official_metadata_verification"
                ],
                "baseline_model_ids": _file_record(baseline_path),
                "baseline_gene_manifest": baseline_gene_file_record,
                "baseline_gene_effect_matrix": baseline_matrix_file_record,
                "organoid_model_ids": _file_record(organoid_manifest),
                "array_hashes": {
                    "eligible_gene_ids": hash_array(cohort.gene_ids),
                    "eligible_canonical_gene_indices": hash_array(
                        cohort.canonical_gene_indices
                    ),
                    "new_model_ids": hash_array(cohort.new_model_ids),
                    "observed_gene_mean": hash_array(cohort.observed_gene_mean),
                    "all_parsed_gene_symbols": hash_array(symbols),
                    "all_parsed_entrez_ids": hash_array(entrez_ids),
                    "shared_model_ids": hash_array(correction_evidence["shared_model_ids"]),
                    "shared_correction_paired_counts": (
                        hash_array(correction_evidence["shared_correction_paired_counts_per_primary_gene"])
                        if "shared_correction_paired_counts_per_primary_gene" in correction_evidence
                        else None
                    ),
                    "shared_correction_values": (
                        hash_array(correction_evidence["shared_correction_values"])
                        if "shared_correction_values" in correction_evidence
                        else None
                    ),
                },
                "mapping": {
                    "gene_header_regex": cohort_contract["gene_header_regex"],
                    "gene_matching": cohort_contract["gene_matching"],
                    "synonym_rescue": cohort_contract["synonym_rescue"],
                    "n_parsed_gene_columns": int(symbols.size),
                    "n_unparsed_gene_columns": int(len(raw_header) - 1 - symbols.size),
                    "duplicate_symbol_action": cohort_contract[
                        "duplicate_symbol_action"
                    ],
                },
                "primary_no_refit_or_recalibration": True,
                "secondary_shared_model_correction_uses_26q3_outcomes": True,
            }
            (staging / "input-manifest.json").write_bytes(_json_bytes(inputs))
            artifacts = {
                "metrics": _file_record(staging / "metrics.json", relative_to=staging),
                "evidence": _npz_artifact_record(
                    staging / "temporal-evidence.npz",
                    arrays=evidence,
                    relative_to=staging,
                    schema=EVIDENCE_SCHEMA,
                ),
                "inputs": _file_record(
                    staging / "input-manifest.json", relative_to=staging
                ),
            }
            completion_status = (
                "complete"
                if overlap_resolved
                else "blocked_unresolved_organoid_overlap"
            )
            completion = {
                "schema": COMPLETION_SCHEMA,
                "status": completion_status,
                "analysis_id": analysis_id,
                "attempt_id": attempt_id,
                "completed_at_utc": _utc_now(),
                "all_failed_gates_retained": True,
                "primary_post_access_refit_or_recalibration": False,
                "secondary_shared_model_correction_uses_26q3_outcomes": True,
                "organoid_overlap_audit_resolved": overlap_resolved,
                "primary_temporal_gates_all_passed": bool(
                    primary_metrics["gates"]["all_passed"]
                ),
                "artifacts": artifacts,
            }
            (staging / "completion.json").write_bytes(_json_bytes(completion))
            for path in staging.iterdir():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
            os.rename(staging, destination)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise
        _append_ledger(
            ledger,
            {
                "schema": "DepPLM-temporal-access-event-v1",
                "event": (
                    "validation_completed"
                    if overlap_resolved
                    else "validation_completed_with_unresolved_overlap_blocker"
                ),
                "recorded_at_utc": _utc_now(),
                "attempt_id": attempt_id,
                "analysis_id": analysis_id,
                "outcome_sha256": outcome_sha256,
                "output_completion_sha256": _sha256_file(destination / "completion.json"),
                "primary_temporal_gates_all_passed": bool(
                    primary_metrics["gates"]["all_passed"]
                ),
                "organoid_overlap_audit_resolved": overlap_resolved,
            },
        )
        return destination
    except Exception as exc:
        _append_ledger(
            ledger,
            {
                "schema": "DepPLM-temporal-access-event-v1",
                "event": "validation_failed",
                "recorded_at_utc": _utc_now(),
                "attempt_id": attempt_id,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback_sha256": hashlib.sha256(
                    traceback.format_exc().encode("utf-8")
                ).hexdigest(),
            },
        )
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze", help="freeze code and 24Q4 consensus before 26Q3 access")
    freeze.add_argument("--freeze-id", required=True)
    freeze.add_argument("--campaign-dir", type=Path, required=True)
    freeze.add_argument("--freeze-manifest", type=Path, required=True)
    freeze.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    freeze.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    freeze.add_argument("--organoid-model-ids", type=Path, required=True)
    prepare = subparsers.add_parser(
        "prepare-acquisition", help="record the frozen acquisition before downloading 26Q3 outcomes"
    )
    prepare.add_argument("--freeze-manifest", type=Path, required=True)
    prepare.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    validate = subparsers.add_parser("validate", help="run the single locked 26Q3 validation")
    validate.add_argument("--analysis-id", required=True)
    validate.add_argument("--freeze-manifest", type=Path, required=True)
    validate.add_argument("--outcome", type=Path, required=True)
    validate.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    validate.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "freeze":
            path = freeze_temporal_analysis(
                freeze_id=args.freeze_id,
                campaign_dir=args.campaign_dir,
                freeze_manifest=args.freeze_manifest,
                lock_path=args.lock,
                protocol_path=args.protocol,
                organoid_model_ids_path=args.organoid_model_ids,
            )
        elif args.command == "prepare-acquisition":
            path = prepare_temporal_acquisition(
                freeze_manifest=args.freeze_manifest, lock_path=args.lock,
            )
        else:
            path = run_temporal_validation(
                analysis_id=args.analysis_id,
                freeze_manifest=args.freeze_manifest,
                outcome_path=args.outcome,
                output_root=args.output_root,
                lock_path=args.lock,
            )
        print(path)
        return 0
    except (OSError, ValueError, TemporalValidationError) as exc:
        print(f"[temporal-26q3] ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
