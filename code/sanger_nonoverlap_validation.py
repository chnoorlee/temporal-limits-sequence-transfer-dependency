"""Frozen, non-overlapping Sanger validation for the corrected DepPLM campaign.

The primary cohort excludes every Sanger model whose unique Broad/DepMap model
identifier occurs anywhere in the frozen Broad CRISPR label table.  A larger,
explicitly secondary sensitivity cohort excludes only the 1,103 cells used to
fit DepPLM.  Cohort identity, model configuration, and training duration are
frozen before numeric Project Score outcomes may be parsed.

Raw Cell Model Passports (CMP) RNA and mutation responses remain in a local
cache.  The result directory contains hashes and derived predictions, never the
raw Sanger files.  ``verify`` rebuilds the external feature matrices, reloads
the saved checkpoints, replays CPU inference, and recomputes all reported
descriptive metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
CODE_PATH = Path(__file__).resolve()
PROJECT_SCORE = (
    ROOT
    / "data"
    / "external"
    / "sanger_project_score"
    / "Project_Score2_fitness_scores_Sanger_v2_Broad_21Q2_20250624.zip"
)
MODEL_CSV = ROOT / "data" / "depmap" / "Model.csv"
BROAD_LABEL_CSV = ROOT / "data" / "depmap" / "CRISPRGeneEffect.csv"
TRAINING_CELLS = ROOT / "data" / "processed" / "cell_lines.txt"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "sanger_nonoverlap_validation_v1"

CMP_API_ROOT = "https://api.cellmodelpassports.sanger.ac.uk"
CMP_API_HOST = "api.cellmodelpassports.sanger.ac.uk"
CMP_RNA_HOST = "cog.sanger.ac.uk"
CMP_OPENAPI_URL = f"{CMP_API_ROOT}/static/swagger.json"
CMP_MODEL_FILES_URL = f"{CMP_API_ROOT}/model_files?page%5Bsize%5D=20000"
CMP_MUTATION_ENDPOINT = (
    f"{CMP_API_ROOT}/models/{{sidm}}/datasets/mutations"
    "?include=gene&page%5Bsize%5D=1000&page%5Bnumber%5D={page}"
)

EXPECTED_PROJECT_SCORE_SHA256 = (
    "ef62be1b953ab2d5b3d76b23853724d0339d8b729a0349152d208fa42e4b74b1"
)
EXPECTED_PROJECT_SCORE_BYTES = 198_873_260
CANONICAL_SEEDS = (42, 43, 44)
EXPECTED_TRAINING_CELLS = 1_103
PREDICTION_ATOL = 1e-6
PREDICTION_RTOL = 1e-6
ANALYSIS_ATOL = 1e-6
ANALYSIS_RTOL = 1e-6
MIN_CELLS_PER_COHORT = 20
MIN_MATCHED_GENES = 1_000
MIN_FINITE_GENES_PER_CELL = 1_000
MIN_FINITE_CELLS_PER_GENE = 10
MIN_GENES_WITH_FINITE_CELL_PROFILES = 1_000
MIN_DEFINED_GENE_CORRELATION_COUNT = 1_000
MIN_DEFINED_GENE_CORRELATION_FRACTION = 0.80

FREEZE_SCHEMA = "DepPLM-Sanger-nonoverlap-freeze-v2"
RAW_MANIFEST_SCHEMA = "DepPLM-Sanger-nonoverlap-raw-cache-v1"
SOURCE_BINDINGS_SCHEMA = "DepPLM-Sanger-nonoverlap-source-bindings-v1"
MODEL_LOCK_SCHEMA = "DepPLM-Sanger-nonoverlap-model-lock-v1"
ANALYSIS_SCHEMA = "DepPLM-Sanger-nonoverlap-analysis-v3"
COMPLETION_SCHEMA = "DepPLM-Sanger-nonoverlap-completion-v3"
PREPROCESSING_SCHEMA = "DepPLM-Sanger-nonoverlap-preprocessing-v1"
PREDICTION_SCHEMA = "DepPLM-Sanger-nonoverlap-predictions-v1"
VERIFICATION_SCHEMA = "DepPLM-Sanger-nonoverlap-verification-v1"
FEASIBILITY_SCHEMA = "DepPLM-Sanger-nonoverlap-feasibility-audit-v1"
PORTABLE_SOURCE_BINDINGS_SCHEMA = "DepPLM-Sanger-nonoverlap-portable-source-bindings-v1"
PORTABLE_PATH_SEMANTICS = (
    "repository-relative when available; otherwise role-scoped local-only logical paths"
)

CLAIM_BOUNDARY = (
    "external cell-model and assay-platform validation; not patient, clinical, "
    "prospective, in vivo, or wet-lab validation"
)
EVALUATION_UNIT = "cell_model"
COHORT_RELATION = "nonoverlapping_external_cell_models"
CLAIM_CLASS = "cell_model_and_assay_platform_concordance_only"
TRUST_SCOPE = "caller_supplied_digest_byte_binding_only"
REDISTRIBUTION_STATUS = "not_cleared_for_public_redistribution"

REPLAY_GATE_NAMES = (
    "exact_run_inventory",
    "caller_supplied_completion_digest_match",
    "completion_digest_bound_to_artifact_inventory",
    "completion_artifact_hashes",
    "freeze_rederived",
    "raw_cache_manifest",
    "source_bindings",
    "run_environment_schema",
    "esm_embedding_provenance",
    "source_snapshot_rehash",
    "project_score_archive",
    "model_lock",
    "feature_rederivation",
    "preprocessing_replay",
    "checkpoint_state_load",
    "per_seed_prediction_replay",
    "ensemble_prediction_replay",
    "analysis_recomputation",
    "analysis_artifact_comparison",
    "primary_cohort_minimum",
    "sensitivity_cohort_minimum",
    "matched_gene_minimum",
    "finite_gene_coverage",
    "gene_correlation_coverage",
)


class ValidationError(RuntimeError):
    """Raised when a frozen identity, source, or replay check fails."""


@dataclass(frozen=True)
class ExternalCell:
    depmap_model_id: str
    sanger_model_id: str
    sanger_model_name: str
    sanger_column: int
    lineage: str
    rna_file_id: str
    rna_url: str
    rna_bytes: int
    primary_strict: bool
    sensitivity_1103_nonoverlap: bool


def _reject_duplicate_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> Any:
    _require_regular_file(path, label="JSON input")
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_object_pairs,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid UTF-8 JSON: {path}") from exc


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_strings(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _validated_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValidationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_regular_file(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValidationError(f"{label} must be a non-symlink regular file: {path}")


def _file_record(path: Path, *, base: Path | None = ROOT) -> dict[str, Any]:
    path = path.resolve()
    _require_regular_file(path, label="source")
    display = path
    if base is not None:
        try:
            display = path.relative_to(base.resolve())
        except ValueError:
            pass
    stat = path.stat()
    return {
        "path": display.as_posix(),
        "bytes": int(stat.st_size),
        "sha256": _sha256_file(path),
    }


def _verify_file_record(path: Path, record: Mapping[str, Any], *, label: str) -> None:
    _require_regular_file(path, label=label)
    if record.get("bytes") != path.stat().st_size:
        raise ValidationError(f"{label} byte count changed: {path}")
    if record.get("sha256") != _sha256_file(path):
        raise ValidationError(f"{label} SHA-256 changed: {path}")


def _require_imported_module_path(module: Any, expected: Path) -> None:
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or not module_file:
        raise ValidationError("Project Score parser module has no source path")
    actual = Path(module_file).resolve()
    expected = expected.resolve()
    _require_regular_file(actual, label="Project Score parser module")
    if actual != expected:
        raise ValidationError(
            "Project Score parser module does not match the hash-bound source"
        )


def _path_from_record(record: Mapping[str, Any], *, label: str) -> Path:
    value = record.get("path")
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{label} record has no path")
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    _verify_file_record(path, record, label=label)
    return path


def _atomic_new_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ValidationError(f"refusing to overwrite existing output: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ValidationError(
                f"refusing to overwrite existing output: {path}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_new_json(path: Path, value: Any) -> None:
    _atomic_new_bytes(path, _canonical_json_bytes(value))


def _atomic_new_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ValidationError(f"refusing to overwrite existing output: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ValidationError(
                f"refusing to overwrite existing output: {path}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _validated_posix_relative_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValidationError(f"{label} must be a non-empty POSIX relative path")
    parts = value.split("/")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or any(part in {"", ".", ".."} for part in parts)
        or any(":" in part for part in parts)
        or parsed.as_posix() != value
        or any(ord(character) < 32 for character in value)
    ):
        raise ValidationError(f"unsafe {label}: {value!r}")
    return value


def _safe_child(root: Path, relative: str) -> Path:
    relative = _validated_posix_relative_path(relative, label="artifact path")
    resolved = (root.resolve() / Path(relative)).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValidationError(f"artifact escapes its root: {relative!r}") from exc
    return resolved


def _read_unique_lines(path: Path, *, label: str) -> list[str]:
    _require_regular_file(path, label=label)
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if not values or any(not value for value in values):
        raise ValidationError(f"{label} contains empty records")
    if len(values) != len(set(values)):
        raise ValidationError(f"{label} contains duplicate records")
    return values


def read_broad_label_ids(path: Path) -> list[str]:
    """Read every row identity in the frozen Broad CRISPR label source."""
    _require_regular_file(path, label="Broad CRISPR label source")
    values: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValidationError("Broad CRISPR label source is empty") from exc
        if not header or header[0] not in {"", "ModelID"}:
            raise ValidationError(
                "Broad CRISPR label source has an unexpected ID column"
            )
        for row_number, row in enumerate(reader, start=2):
            model_id = row[0].strip() if row else ""
            if not model_id.startswith("ACH-"):
                raise ValidationError(
                    f"invalid Broad model identifier on row {row_number}: {model_id!r}"
                )
            values.append(model_id)
    if len(values) != len(set(values)):
        raise ValidationError("Broad CRISPR label source contains duplicate row IDs")
    return values


def _read_model_mapping(
    path: Path,
) -> tuple[dict[str, tuple[str, str]], dict[str, Any]]:
    """Return globally one-to-one ACH-to-SIDM mappings and their lineages."""
    _require_regular_file(path, label="Model.csv")
    ach_to_sidm: defaultdict[str, set[str]] = defaultdict(set)
    sidm_to_ach: defaultdict[str, set[str]] = defaultdict(set)
    ach_lineages: defaultdict[str, set[str]] = defaultdict(set)
    rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"ModelID", "SangerModelID", "OncotreeLineage"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValidationError("Model.csv lacks required mapping or lineage fields")
        for row in reader:
            rows += 1
            ach = (row.get("ModelID") or "").strip()
            sidm = (row.get("SangerModelID") or "").strip()
            lineage = (row.get("OncotreeLineage") or "").strip()
            if not ach or not sidm:
                continue
            ach_to_sidm[ach].add(sidm)
            sidm_to_ach[sidm].add(ach)
            if lineage:
                ach_lineages[ach].add(lineage)

    exact: dict[str, tuple[str, str]] = {}
    ambiguous_ach = sorted(
        ach for ach, values in ach_to_sidm.items() if len(values) != 1
    )
    ambiguous_sidm = sorted(
        sidm for sidm, values in sidm_to_ach.items() if len(values) != 1
    )
    ambiguous_lineage: list[str] = []
    missing_lineage: list[str] = []
    for ach, sidms in ach_to_sidm.items():
        if len(sidms) != 1:
            continue
        sidm = next(iter(sidms))
        if len(sidm_to_ach[sidm]) != 1:
            continue
        lineages = ach_lineages.get(ach, set())
        if len(lineages) > 1:
            ambiguous_lineage.append(ach)
            continue
        if not lineages:
            missing_lineage.append(ach)
            continue
        exact[ach] = (sidm, next(iter(lineages)))
    return exact, {
        "model_rows": rows,
        "globally_one_to_one_with_lineage": len(exact),
        "ambiguous_ach_ids_excluded": ambiguous_ach,
        "ambiguous_sidm_ids_excluded": ambiguous_sidm,
        "ambiguous_lineage_ids_excluded": sorted(ambiguous_lineage),
        "missing_lineage_ids_excluded": sorted(missing_lineage),
    }


def _validate_openapi_snapshot(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    paths = payload.get("paths") if isinstance(payload, dict) else None
    required = {"/model_files", "/models/{model_id}/datasets/mutations"}
    if not isinstance(paths, dict) or not required.issubset(paths):
        raise ValidationError("CMP OpenAPI snapshot lacks required official endpoints")
    return {
        "url": CMP_OPENAPI_URL,
        "record": _file_record(path),
        "required_paths": sorted(required),
    }


def _validate_cmp_api_url(
    url: Any,
    *,
    expected_path: str,
    expected_query: str | None = None,
    label: str,
    allow_http_metadata: bool = False,
) -> str:
    """Validate an API metadata URL and return its HTTPS transport form."""
    if not isinstance(url, str):
        raise ValidationError(f"{label} is not a string")
    if (
        not url.isascii()
        or url != url.strip()
        or any(ord(character) <= 32 or ord(character) == 127 for character in url)
    ):
        raise ValidationError(f"{label} contains unsafe URL characters")
    try:
        parsed = urllib.parse.urlsplit(url)
    except ValueError as exc:
        raise ValidationError(f"{label} is malformed") from exc
    allowed_schemes = {"https", "http"} if allow_http_metadata else {"https"}
    if (
        parsed.scheme not in allowed_schemes
        or parsed.netloc != CMP_API_HOST
        or parsed.path != expected_path
        or (expected_query is not None and parsed.query != expected_query)
        or "#" in url
        or parsed.fragment
    ):
        raise ValidationError(f"{label} is not the expected official CMP API endpoint")
    return parsed._replace(scheme="https").geturl()


def _validate_https_transfer_url(
    url: Any,
    *,
    expected_host: str,
    expected_path: str,
    expected_query: str,
    label: str,
) -> str:
    """Require an exact HTTPS host/path/query for a transport URL."""
    if not isinstance(url, str):
        raise ValidationError(f"{label} is not a string")
    if (
        not url.isascii()
        or url != url.strip()
        or any(ord(character) <= 32 or ord(character) == 127 for character in url)
    ):
        raise ValidationError(f"{label} contains unsafe URL characters")
    try:
        parsed = urllib.parse.urlsplit(url)
    except ValueError as exc:
        raise ValidationError(f"{label} is malformed") from exc
    if (
        parsed.scheme != "https"
        or parsed.netloc != expected_host
        or parsed.path != expected_path
        or parsed.query != expected_query
        or "#" in url
        or parsed.fragment
    ):
        raise ValidationError(f"{label} is not the expected HTTPS endpoint")
    return url


def _is_sanger_model_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("SIDM")
        and bool(value[4:])
        and value[4:].isascii()
        and value[4:].isdigit()
    )


def _rna_download_path(sidm: str) -> str:
    if not _is_sanger_model_id(sidm):
        raise ValidationError("invalid Sanger model identity")
    return f"/cmp/download/{sidm}_rnaseq.csv.zip"


class _PinnedHTTPSRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject a redirect before urllib can leave the pinned HTTPS endpoint."""

    def __init__(
        self, *, expected_host: str, expected_path: str, expected_query: str
    ) -> None:
        super().__init__()
        self.expected_host = expected_host
        self.expected_path = expected_path
        self.expected_query = expected_query

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        absolute = urllib.parse.urljoin(req.full_url, newurl)
        _validate_https_transfer_url(
            absolute,
            expected_host=self.expected_host,
            expected_path=self.expected_path,
            expected_query=self.expected_query,
            label="HTTP redirect URL",
        )
        return super().redirect_request(req, fp, code, msg, headers, absolute)


def read_processed_rna_index(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Index exactly one official per-model processed RNA ZIP per SIDM."""
    payload = _load_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValidationError("CMP model-files snapshot is not JSON:API data")
    records = payload["data"]
    meta = payload.get("meta")
    count = meta.get("count") if isinstance(meta, dict) else None
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        or count != len(records)
    ):
        raise ValidationError("CMP model-files snapshot count does not match its data")
    links = payload.get("links")
    if not isinstance(links, dict):
        raise ValidationError("CMP model-files snapshot lacks pagination links")
    self_url = links.get("self")
    expected_query = urllib.parse.urlsplit(CMP_MODEL_FILES_URL).query
    canonical_self_url = _validate_cmp_api_url(
        self_url,
        expected_path="/model_files",
        expected_query=expected_query,
        label="CMP model-files snapshot self link",
        allow_http_metadata=True,
    )
    if "next" in links and links["next"] is not None:
        raise ValidationError("CMP model-files snapshot is paginated or incomplete")

    candidates: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()
    for item in records:
        if not isinstance(item, dict):
            raise ValidationError("CMP model-files data contains a non-object")
        raw_file_id = item.get("id")
        if isinstance(raw_file_id, bool) or not isinstance(raw_file_id, (int, str)):
            raise ValidationError(
                "CMP model-files contains an invalid or duplicate file ID"
            )
        if isinstance(raw_file_id, int):
            if raw_file_id <= 0:
                raise ValidationError(
                    "CMP model-files contains an invalid or duplicate file ID"
                )
            file_id = str(raw_file_id)
        else:
            file_id = raw_file_id
        if not file_id or file_id != file_id.strip() or file_id in seen_ids:
            raise ValidationError(
                "CMP model-files contains an invalid or duplicate file ID"
            )
        seen_ids.add(file_id)
        attributes = item.get("attributes")
        if not isinstance(attributes, dict):
            continue
        file_meta = attributes.get("meta")
        if not isinstance(file_meta, dict):
            continue
        if not (
            file_meta.get("data_type_abbr") == "RNA"
            and file_meta.get("level") == "processed"
            and file_meta.get("filetype") == "ZIP"
            and str(file_meta.get("used_for_annotation")) == "1"
        ):
            continue
        sidm_value = attributes.get("model_id")
        sidm = sidm_value if isinstance(sidm_value, str) else ""
        url_value = attributes.get("url")
        url = url_value if isinstance(url_value, str) else ""
        size = attributes.get("size")
        try:
            expected_rna_path = _rna_download_path(sidm)
            _validate_https_transfer_url(
                url,
                expected_host=CMP_RNA_HOST,
                expected_path=expected_rna_path,
                expected_query="",
                label=f"processed RNA URL for {sidm or file_id}",
            )
        except ValidationError:
            raise ValidationError(
                f"invalid processed RNA record for {sidm or file_id}"
            ) from None
        if (
            not _is_sanger_model_id(sidm)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
        ):
            raise ValidationError(f"invalid processed RNA record for {sidm or file_id}")
        candidates[sidm].append(
            {
                "file_id": file_id,
                "url": url,
                "bytes": size,
                "meta": file_meta,
            }
        )
    unique = {
        sidm: values[0] for sidm, values in candidates.items() if len(values) == 1
    }
    duplicates = sorted(sidm for sidm, values in candidates.items() if len(values) != 1)
    return unique, {
        "snapshot_records": len(records),
        "processed_rna_candidate_models": len(candidates),
        "unique_processed_rna_models": len(unique),
        "duplicate_processed_rna_models_excluded": duplicates,
        "source_url": canonical_self_url,
        "source_url_reported": self_url,
        "http_metadata_scheme_normalized": self_url != canonical_self_url,
    }


def _derive_cohort_definition(
    *,
    model_csv: Path,
    broad_label_ids: Sequence[str],
    training_ids: Sequence[str],
    sanger_columns: Sequence[Any],
    rna_index: Mapping[str, Mapping[str, Any]],
) -> tuple[list[ExternalCell], dict[str, Any]]:
    """Apply the shared cohort predicates without enforcing adequacy gates."""
    broad = set(broad_label_ids)
    training = set(training_ids)
    if len(broad) != len(broad_label_ids):
        raise ValidationError("Broad label row identities are duplicated")
    if len(training) != len(training_ids):
        raise ValidationError("training cell identities are duplicated")
    if not training.issubset(broad):
        missing = sorted(training - broad)
        raise ValidationError(
            "the 1,103 training cohort is not a subset of all Broad label rows: "
            f"{missing[:5]}"
        )
    exact, mapping_meta = _read_model_mapping(model_csv)
    by_sidm: dict[str, Any] = {}
    for column in sanger_columns:
        sidm = str(column.sanger_model_id)
        if sidm in by_sidm:
            raise ValidationError("Project Score Sanger columns are not unique")
        by_sidm[sidm] = column

    exact_with_project_score: list[str] = []
    exact_with_project_score_and_rna: list[str] = []
    cells: list[ExternalCell] = []
    for ach, (sidm, lineage) in exact.items():
        column = by_sidm.get(sidm)
        if column is None:
            continue
        exact_with_project_score.append(ach)
        rna = rna_index.get(sidm)
        if rna is None:
            continue
        exact_with_project_score_and_rna.append(ach)
        sensitivity = ach not in training
        primary = ach not in broad
        if not sensitivity:
            continue
        if primary and not sensitivity:
            raise ValidationError(
                "strict non-overlap is not nested in sensitivity cohort"
            )
        cells.append(
            ExternalCell(
                depmap_model_id=ach,
                sanger_model_id=sidm,
                sanger_model_name=str(column.model_name),
                sanger_column=int(column.csv_index),
                lineage=lineage,
                rna_file_id=str(rna["file_id"]),
                rna_url=str(rna["url"]),
                rna_bytes=int(rna["bytes"]),
                primary_strict=primary,
                sensitivity_1103_nonoverlap=True,
            )
        )
    cells.sort(key=lambda cell: (cell.sanger_model_id, cell.depmap_model_id))
    primary_cells = [cell for cell in cells if cell.primary_strict]
    if any(cell.depmap_model_id in broad for cell in primary_cells):
        raise ValidationError(
            "Broad-labelled cell leaked into the strict primary cohort"
        )
    return cells, {
        **mapping_meta,
        "all_broad_label_rows": len(broad),
        "depplm_training_cells": len(training),
        "project_score_unique_sanger_models": len(by_sidm),
        "exact_mappings_with_project_score_column": len(exact_with_project_score),
        "exact_mappings_with_project_score_and_unique_processed_rna": len(
            exact_with_project_score_and_rna
        ),
        "exact_mappings_missing_unique_processed_rna": (
            len(exact_with_project_score) - len(exact_with_project_score_and_rna)
        ),
        "training_overlap_cells_excluded": sum(
            ach in training for ach in exact_with_project_score_and_rna
        ),
        "sensitivity_1103_nonoverlap_cells": len(cells),
        "broad_labelled_cells_remaining_in_sensitivity": sum(
            cell.depmap_model_id in broad for cell in cells
        ),
        "primary_all_broad_nonoverlap_cells": len(primary_cells),
        "primary_rule": (
            "exclude every exact-mapped Sanger model whose ACH identifier appears in "
            "any row of the frozen Broad CRISPRGeneEffect.csv label source"
        ),
        "sensitivity_rule": (
            "exclude only the 1,103 DepPLM training ACH identifiers; secondary and "
            "less strict because additional Broad-labelled models can remain"
        ),
    }


def build_cohort_definition(
    *,
    model_csv: Path,
    broad_label_ids: Sequence[str],
    training_ids: Sequence[str],
    sanger_columns: Sequence[Any],
    rna_index: Mapping[str, Mapping[str, Any]],
) -> tuple[list[ExternalCell], dict[str, Any]]:
    """Create strict and sensitivity cohorts without reading numeric outcomes."""
    cells, audit = _derive_cohort_definition(
        model_csv=model_csv,
        broad_label_ids=broad_label_ids,
        training_ids=training_ids,
        sanger_columns=sanger_columns,
        rna_index=rna_index,
    )
    if not cells:
        raise ValidationError(
            "no exact 1,103-nonoverlap Sanger models have processed RNA"
        )
    primary_count = sum(cell.primary_strict for cell in cells)
    if primary_count < MIN_CELLS_PER_COHORT:
        raise ValidationError(
            "the all-Broad-label non-overlap primary cohort has fewer than "
            f"{MIN_CELLS_PER_COHORT} cells"
        )
    if len(cells) < MIN_CELLS_PER_COHORT:
        raise ValidationError(
            "the 1,103-nonoverlap sensitivity cohort has fewer than "
            f"{MIN_CELLS_PER_COHORT} cells"
        )
    return cells, audit


def select_final_config_from_fold_results(
    fold_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Prespecified full-data configuration rule using P1 inner results only."""
    if len(fold_results) != 5:
        raise ValidationError(
            "final configuration requires exactly five P1 fold results"
        )
    entries: list[dict[str, Any]] = []
    expected_folds = {f"fold-{index:02d}" for index in range(5)}
    for result in fold_results:
        fold = result.get("fold")
        config = result.get("best_config")
        epoch = result.get("best_epoch")
        if fold not in expected_folds or not isinstance(config, dict):
            raise ValidationError("P1 fold result has invalid fold or best_config")
        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or not 1 <= epoch <= 20
        ):
            raise ValidationError("P1 best_epoch is outside the canonical 1..20 range")
        score: float | None = None
        tuning = result.get("tuning")
        if isinstance(tuning, list):
            matches = [
                row
                for row in tuning
                if isinstance(row, dict) and row.get("config") == config
            ]
            exact = [row for row in matches if row.get("best_epoch") == epoch]
            if len(exact) == 1:
                candidate = exact[0].get("selection_score")
                if isinstance(candidate, (int, float)) and not isinstance(
                    candidate, bool
                ):
                    score = float(candidate)
        if score is None or not math.isfinite(score):
            raise ValidationError("cannot recover the selected inner-validation score")
        canonical = json.dumps(
            config, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        entries.append(
            {
                "fold": fold,
                "config": dict(config),
                "config_canonical": canonical,
                "best_epoch": epoch,
                "selection_score": score,
            }
        )
    if {entry["fold"] for entry in entries} != expected_folds:
        raise ValidationError("P1 fold results are duplicated or incomplete")

    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[entry["config_canonical"]].append(entry)
    ranked = sorted(
        grouped.items(),
        key=lambda pair: (
            -len(pair[1]),
            -statistics.fmean(row["selection_score"] for row in pair[1]),
            pair[0],
        ),
    )
    selected_key, selected_rows = ranked[0]
    selected_config = json.loads(selected_key)
    epochs = sorted(int(entry["best_epoch"]) for entry in selected_rows)
    return {
        "config": selected_config,
        "epochs": int(statistics.median_low(epochs)),
        "rule": (
            "modal best_config across five verified P1 folds; ties use higher mean "
            "inner selection score then canonical JSON; epochs use lower median of "
            "best_epoch values only from folds selecting the final configuration"
        ),
        "selected_config_fold_count": len(selected_rows),
        "fold_evidence": sorted(entries, key=lambda row: row["fold"]),
    }


def _select_campaign_training_specs(report: Mapping[str, Any]) -> dict[str, Any]:
    specs: dict[str, Any] = {}
    runs = {
        run.get("key"): run
        for run in report.get("source_runs", [])
        if isinstance(run, dict)
    }
    for seed in CANONICAL_SEEDS:
        key = f"p1|DepPLM|seed-{seed}"
        run = runs.get(key)
        if not isinstance(run, dict):
            raise ValidationError(f"verified campaign lacks {key}")
        raw_run_path = run.get("path")
        if not isinstance(raw_run_path, str) or not raw_run_path:
            raise ValidationError(f"verified campaign has no path for {key}")
        run_path = Path(raw_run_path)
        if not run_path.is_absolute():
            run_path = ROOT / run_path
        run_path = run_path.resolve()
        result_records = {
            artifact.get("path"): artifact
            for artifact in run.get("fold_artifacts", [])
            if isinstance(artifact, dict)
            and isinstance(artifact.get("path"), str)
            and artifact["path"].endswith("-result.json")
        }
        expected = {f"fold-{index:02d}-result.json" for index in range(5)}
        if set(result_records) != expected:
            raise ValidationError(f"{key} lacks exactly five P1 result artifacts")
        results = []
        result_sources = []
        for relative in sorted(expected):
            path = _safe_child(run_path, relative)
            _verify_file_record(
                path, result_records[relative], label=f"{key}:{relative}"
            )
            result = _load_json(path)
            if (
                not isinstance(result, dict)
                or result.get("protocol") != "p1"
                or result.get("condition") != "DepPLM"
                or result.get("seed") != seed
                or result.get("pilot") is not False
            ):
                raise ValidationError(f"invalid canonical P1 fold result: {path}")
            results.append(result)
            result_sources.append(_file_record(path, base=ROOT))
        specs[str(seed)] = {
            **select_final_config_from_fold_results(results),
            "result_sources": result_sources,
        }
    return specs


def _environment_manifest() -> dict[str, Any]:
    import pandas
    import scipy
    import sklearn
    import torch

    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pandas.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def _validate_environment_manifest(value: Any) -> dict[str, Any]:
    required = {
        "python",
        "python_implementation",
        "platform",
        "packages",
        "torch_cuda_version",
        "cudnn_version",
        "cuda_available",
    }
    package_names = {"numpy", "pandas", "scipy", "scikit_learn", "torch"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValidationError("run environment inventory fields are invalid")
    if any(
        not isinstance(value[key], str) or not value[key]
        for key in ("python", "python_implementation", "platform")
    ):
        raise ValidationError("run environment identity is invalid")
    packages = value["packages"]
    if (
        not isinstance(packages, dict)
        or set(packages) != package_names
        or any(
            not isinstance(version, str) or not version for version in packages.values()
        )
    ):
        raise ValidationError("run package-version inventory is invalid")
    if value["torch_cuda_version"] is not None and not isinstance(
        value["torch_cuda_version"], str
    ):
        raise ValidationError("run CUDA version is invalid")
    cudnn = value["cudnn_version"]
    if cudnn is not None and (isinstance(cudnn, bool) or not isinstance(cudnn, int)):
        raise ValidationError("run cuDNN version is invalid")
    if not isinstance(value["cuda_available"], bool):
        raise ValidationError("run CUDA availability flag is invalid")
    return value


def _import_existing_validation() -> Any:
    try:
        import sanger_independent_validation as module
    except ImportError:
        sys.path.insert(0, str(CODE_PATH.parent))
        import sanger_independent_validation as module
    return module


def create_feasibility_audit(
    *,
    output: Path,
    model_files_json: Path,
    project_score: Path = PROJECT_SCORE,
    model_csv: Path = MODEL_CSV,
    broad_labels: Path = BROAD_LABEL_CSV,
    training_cells: Path = TRAINING_CELLS,
) -> Path:
    """Write a diagnostic cohort audit without parsing numeric outcomes."""
    source_parser_path = CODE_PATH.with_name("sanger_independent_validation.py")
    existing = _import_existing_validation()
    _require_imported_module_path(existing, source_parser_path)
    input_paths = {
        "project_score_archive": project_score,
        "cmp_model_files_snapshot": model_files_json,
        "model_csv": model_csv,
        "broad_label_source": broad_labels,
        "training_cells": training_cells,
    }
    input_records = {role: _file_record(path) for role, path in input_paths.items()}
    source_paths = {
        "cohort_audit": CODE_PATH,
        "project_score_header_parser": source_parser_path,
    }
    source_records = {role: _file_record(path) for role, path in source_paths.items()}
    external_record = existing.validate_external_archive(project_score)
    if (
        external_record.get("sha256")
        != input_records["project_score_archive"]["sha256"]
        or external_record.get("bytes")
        != input_records["project_score_archive"]["bytes"]
        or external_record.get("sha256") != EXPECTED_PROJECT_SCORE_SHA256
        or external_record.get("bytes") != EXPECTED_PROJECT_SCORE_BYTES
    ):
        raise ValidationError(
            "Project Score archive is not the fixed Sanger v2 release"
        )
    columns, column_meta = existing.read_sanger_columns(project_score)
    rna_index, rna_meta = read_processed_rna_index(model_files_json)
    broad_ids = read_broad_label_ids(broad_labels)
    training_ids = _read_unique_lines(training_cells, label="DepPLM training cells")
    if len(training_ids) != EXPECTED_TRAINING_CELLS:
        raise ValidationError(
            f"training cohort must contain {EXPECTED_TRAINING_CELLS} cells"
        )
    cells, cohort_meta = _derive_cohort_definition(
        model_csv=model_csv,
        broad_label_ids=broad_ids,
        training_ids=training_ids,
        sanger_columns=columns,
        rna_index=rna_index,
    )
    primary_cells = [cell for cell in cells if cell.primary_strict]
    cohort_gates = {
        "primary_all_broad_nonoverlap_minimum": {
            "observed": len(primary_cells),
            "required": MIN_CELLS_PER_COHORT,
            "passed": len(primary_cells) >= MIN_CELLS_PER_COHORT,
        },
        "sensitivity_1103_nonoverlap_minimum": {
            "observed": len(cells),
            "required": MIN_CELLS_PER_COHORT,
            "passed": len(cells) >= MIN_CELLS_PER_COHORT,
        },
    }
    primary_per_gene_possible = len(primary_cells) >= MIN_FINITE_CELLS_PER_GENE
    sensitivity_per_gene_possible = len(cells) >= MIN_FINITE_CELLS_PER_GENE
    per_gene_gates = {
        "minimum_finite_external_cells_per_gene": {
            "required": MIN_FINITE_CELLS_PER_GENE,
            "primary": {
                "cohort_size_upper_bound": len(primary_cells),
                "structurally_possible": primary_per_gene_possible,
                "passed": False if not primary_per_gene_possible else None,
            },
            "sensitivity": {
                "cohort_size_upper_bound": len(cells),
                "structurally_possible": sensitivity_per_gene_possible,
                "passed": False if not sensitivity_per_gene_possible else None,
            },
            "null_means_not_assessed_without_numeric_outcomes": True,
        },
        "minimum_genes_with_finite_cell_profiles": {
            "required": MIN_GENES_WITH_FINITE_CELL_PROFILES,
            "assessed": False,
            "passed": None,
        },
        "minimum_defined_gene_correlation_count": {
            "required": MIN_DEFINED_GENE_CORRELATION_COUNT,
            "assessed": False,
            "passed": None,
        },
        "minimum_defined_gene_correlation_fraction": {
            "required": MIN_DEFINED_GENE_CORRELATION_FRACTION,
            "assessed": False,
            "passed": None,
        },
    }
    cohort_sizes_passed = all(gate["passed"] for gate in cohort_gates.values())
    if not cohort_sizes_passed or not primary_per_gene_possible:
        status = "cohort_feasibility_gates_failed"
    else:
        status = "cohort_size_gates_passed_outcome_coverage_unassessed"
    funnel_counts = {
        "model_csv_rows": cohort_meta["model_rows"],
        "globally_one_to_one_with_lineage": cohort_meta[
            "globally_one_to_one_with_lineage"
        ],
        "project_score_table_columns": column_meta["table_columns"],
        "project_score_sanger_qc_pass_columns": column_meta["sanger_qc_pass_columns"],
        "project_score_unique_sanger_models": cohort_meta[
            "project_score_unique_sanger_models"
        ],
        "cmp_model_file_snapshot_records": rna_meta["snapshot_records"],
        "cmp_processed_rna_candidate_models": rna_meta[
            "processed_rna_candidate_models"
        ],
        "cmp_unique_processed_rna_models": rna_meta["unique_processed_rna_models"],
        "exact_mappings_with_project_score_column": cohort_meta[
            "exact_mappings_with_project_score_column"
        ],
        "exact_mappings_with_project_score_and_unique_processed_rna": cohort_meta[
            "exact_mappings_with_project_score_and_unique_processed_rna"
        ],
        "exact_mappings_missing_unique_processed_rna": cohort_meta[
            "exact_mappings_missing_unique_processed_rna"
        ],
        "training_overlap_cells_excluded": cohort_meta[
            "training_overlap_cells_excluded"
        ],
        "sensitivity_1103_nonoverlap_cells": len(cells),
        "broad_labelled_cells_remaining_in_sensitivity": cohort_meta[
            "broad_labelled_cells_remaining_in_sensitivity"
        ],
        "primary_all_broad_nonoverlap_cells": len(primary_cells),
    }
    payload = {
        "schema": FEASIBILITY_SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scientific_result": False,
        "diagnostic_only": True,
        "numeric_sanger_outcomes_accessed": False,
        "status": status,
        "formal_freeze_created": False,
        "preregistration_status": "not_preregistered",
        "independent_biological_validation_established": False,
        "claim_boundary": (
            "cohort-feasibility audit only; not a preregistration, protocol freeze, "
            "performance result, or independent biological validation"
        ),
        "input_bindings": input_records,
        "source_code_bindings": source_records,
        "identity_hashes": {
            "all_broad_label_ids_sha256": _sha256_strings(broad_ids),
            "training_ids_sha256": _sha256_strings(training_ids),
        },
        "project_score_header_audit": column_meta,
        "cmp_model_files_audit": rna_meta,
        "cohort_derivation_audit": cohort_meta,
        "funnel_counts": funnel_counts,
        "cohort_gates": cohort_gates,
        "per_gene_feasibility_gates": per_gene_gates,
        "retained_cell_ids": {
            "primary_all_broad_nonoverlap": {
                "depmap_model_ids": sorted(
                    cell.depmap_model_id for cell in primary_cells
                ),
                "sanger_model_ids": sorted(
                    cell.sanger_model_id for cell in primary_cells
                ),
            },
            "sensitivity_1103_nonoverlap_less_strict": {
                "depmap_model_ids": sorted(cell.depmap_model_id for cell in cells),
                "sanger_model_ids": sorted(cell.sanger_model_id for cell in cells),
            },
        },
        "retained_cell_mappings": [
            {
                "depmap_model_id": cell.depmap_model_id,
                "sanger_model_id": cell.sanger_model_id,
                "lineage": cell.lineage,
                "primary_strict": cell.primary_strict,
                "sensitivity_1103_nonoverlap": cell.sensitivity_1103_nonoverlap,
            }
            for cell in cells
        ],
        "conclusion": (
            "No formal freeze or validation result was produced. Failed gates must "
            "remain failed; the less-strict sensitivity cohort is not the primary "
            "cohort."
        ),
    }
    for role, path in input_paths.items():
        _verify_file_record(path.resolve(), input_records[role], label=role)
    for role, path in source_paths.items():
        _verify_file_record(path.resolve(), source_records[role], label=role)
    _atomic_new_json(output, payload)
    return output


def create_freeze(
    *,
    output: Path,
    campaign_report: Path,
    openapi_json: Path,
    model_files_json: Path,
    project_score: Path = PROJECT_SCORE,
    model_csv: Path = MODEL_CSV,
    broad_labels: Path = BROAD_LABEL_CSV,
    training_cells: Path = TRAINING_CELLS,
) -> Path:
    """Freeze identities/configuration while reading no numeric Sanger outcomes."""
    existing = _import_existing_validation()
    external_record = existing.validate_external_archive(project_score)
    if (
        external_record.get("sha256") != EXPECTED_PROJECT_SCORE_SHA256
        or external_record.get("bytes") != EXPECTED_PROJECT_SCORE_BYTES
    ):
        raise ValidationError(
            "Project Score archive is not the fixed Sanger v2 release"
        )
    columns, column_meta = existing.read_sanger_columns(project_score)
    campaign, campaign_record = existing.validate_campaign_report(campaign_report)
    openapi_record = _validate_openapi_snapshot(openapi_json)
    rna_index, rna_meta = read_processed_rna_index(model_files_json)
    broad_ids = read_broad_label_ids(broad_labels)
    training_ids = _read_unique_lines(training_cells, label="DepPLM training cells")
    if len(training_ids) != EXPECTED_TRAINING_CELLS:
        raise ValidationError(
            f"training cohort must contain {EXPECTED_TRAINING_CELLS} cells"
        )
    cells, cohort_meta = build_cohort_definition(
        model_csv=model_csv,
        broad_label_ids=broad_ids,
        training_ids=training_ids,
        sanger_columns=columns,
        rna_index=rna_index,
    )
    training_specs = _select_campaign_training_specs(campaign)
    payload = {
        "schema": FREEZE_SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scientific_result": False,
        "numeric_sanger_outcomes_accessed": False,
        "reason": (
            "Cohort identities, feature sources, configurations, and epochs are "
            "frozen; this file contains no model performance result."
        ),
        "official_sources": {
            "cmp_openapi": openapi_record,
            "cmp_model_files": {
                "url": CMP_MODEL_FILES_URL,
                "record": _file_record(model_files_json),
                "metadata": rna_meta,
            },
            "project_score": external_record,
        },
        "inputs": {
            "campaign_report": campaign_record,
            "model_csv": _file_record(model_csv),
            "broad_label_source": _file_record(broad_labels),
            "training_cells": _file_record(training_cells),
        },
        "identity_hashes": {
            "all_broad_label_ids_sha256": _sha256_strings(broad_ids),
            "training_ids_sha256": _sha256_strings(training_ids),
        },
        "project_score_header_audit": column_meta,
        "cohort_audit": cohort_meta,
        "cells": [asdict(cell) for cell in cells],
        "training_specs": training_specs,
        "preprocessing_policy": {
            "training_rows": EXPECTED_TRAINING_CELLS,
            "expression": (
                "CMP rsem_tpm is converted to log2(TPM+1); training-only column "
                "means/scales and full-SVD PCA; missing external features are imputed "
                "with training means by FittedExpressionPCA.transform"
            ),
            "mutation": "training-only top-5000 vocabulary and K=64 token order",
            "lineage": "training-only one-hot categories; unseen categories are all-zero",
            "graph": "training-only mutation PMI edge weights on the fixed STRING graph",
            "n_expr_pc": 128,
            "top_mut": 5000,
            "batch_size": 64,
            "gene_subsample": 6000,
            "scheduler_horizon": 20,
        },
        "analysis_policy": {
            "primary": "all-Broad-label non-overlap",
            "sensitivity": "1,103-training-cell non-overlap (less strict)",
            "minimum_cells_per_cohort": MIN_CELLS_PER_COHORT,
            "minimum_matched_genes": MIN_MATCHED_GENES,
            "minimum_finite_genes_per_cell": MIN_FINITE_GENES_PER_CELL,
            "minimum_finite_cells_per_gene": MIN_FINITE_CELLS_PER_GENE,
            "minimum_genes_with_finite_cell_profiles": (
                MIN_GENES_WITH_FINITE_CELL_PROFILES
            ),
            "minimum_defined_gene_correlation_count": (
                MIN_DEFINED_GENE_CORRELATION_COUNT
            ),
            "minimum_defined_gene_correlation_fraction": (
                MIN_DEFINED_GENE_CORRELATION_FRACTION
            ),
            "baseline": "training-only per-gene mean dependency",
            "primary_metrics": [
                "residual R2 versus training-gene-mean baseline",
                "MAE improvement versus training-gene-mean baseline",
                "per-cell correlations across genes",
                "per-gene correlations across external cells",
            ],
            "secondary_metrics": [
                "pooled matrix Pearson r",
                "pooled matrix Spearman rho",
                "raw-scale MAE",
            ],
            "interpretation": "descriptive only",
            "prediction_replay_tolerance": {
                "rtol": PREDICTION_RTOL,
                "atol": PREDICTION_ATOL,
            },
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "analysis_source": _file_record(CODE_PATH),
    }
    _atomic_new_json(output, payload)
    return output


def validate_freeze(path: Path, *, rederive: bool = True) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict) or payload.get("schema") != FREEZE_SCHEMA:
        raise ValidationError("invalid Sanger non-overlap freeze schema")
    if payload.get("scientific_result") is not False:
        raise ValidationError("freeze must be explicitly non-scientific")
    if payload.get("numeric_sanger_outcomes_accessed") is not False:
        raise ValidationError("freeze claims numeric Sanger outcome access")
    if payload.get("claim_boundary") != CLAIM_BOUNDARY:
        raise ValidationError("freeze claim boundary is missing or changed")
    analysis_policy = payload.get("analysis_policy")
    expected_analysis_policy = {
        "primary": "all-Broad-label non-overlap",
        "sensitivity": "1,103-training-cell non-overlap (less strict)",
        "minimum_cells_per_cohort": MIN_CELLS_PER_COHORT,
        "minimum_matched_genes": MIN_MATCHED_GENES,
        "minimum_finite_genes_per_cell": MIN_FINITE_GENES_PER_CELL,
        "minimum_finite_cells_per_gene": MIN_FINITE_CELLS_PER_GENE,
        "minimum_genes_with_finite_cell_profiles": (
            MIN_GENES_WITH_FINITE_CELL_PROFILES
        ),
        "minimum_defined_gene_correlation_count": (MIN_DEFINED_GENE_CORRELATION_COUNT),
        "minimum_defined_gene_correlation_fraction": (
            MIN_DEFINED_GENE_CORRELATION_FRACTION
        ),
        "interpretation": "descriptive only",
        "prediction_replay_tolerance": {
            "rtol": PREDICTION_RTOL,
            "atol": PREDICTION_ATOL,
        },
    }
    if not isinstance(analysis_policy, dict) or any(
        analysis_policy.get(field) != expected
        for field, expected in expected_analysis_policy.items()
    ):
        raise ValidationError("freeze analysis policy or adequacy thresholds changed")
    source = payload.get("analysis_source")
    if not isinstance(source, dict):
        raise ValidationError("freeze lacks analysis source binding")
    _verify_file_record(CODE_PATH, source, label="freeze analysis source")
    cells = payload.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValidationError("freeze has no external cells")
    sidms = [cell.get("sanger_model_id") for cell in cells if isinstance(cell, dict)]
    achs = [cell.get("depmap_model_id") for cell in cells if isinstance(cell, dict)]
    if len(sidms) != len(cells) or len(set(sidms)) != len(cells):
        raise ValidationError("freeze SIDM identities are missing or duplicated")
    if len(achs) != len(cells) or len(set(achs)) != len(cells):
        raise ValidationError("freeze ACH identities are missing or duplicated")
    primary_count = sum(cell.get("primary_strict") is True for cell in cells)
    if primary_count < MIN_CELLS_PER_COHORT:
        raise ValidationError(
            f"freeze strict primary cohort has fewer than {MIN_CELLS_PER_COHORT} cells"
        )
    if len(cells) < MIN_CELLS_PER_COHORT:
        raise ValidationError(
            f"freeze sensitivity cohort has fewer than {MIN_CELLS_PER_COHORT} cells"
        )
    if any(cell.get("sensitivity_1103_nonoverlap") is not True for cell in cells):
        raise ValidationError("freeze contains a cell outside the sensitivity cohort")
    specs = payload.get("training_specs")
    if not isinstance(specs, dict) or set(specs) != {
        str(seed) for seed in CANONICAL_SEEDS
    }:
        raise ValidationError("freeze training specifications are incomplete")
    if rederive:
        official = payload.get("official_sources")
        inputs = payload.get("inputs")
        if not isinstance(official, dict) or not isinstance(inputs, dict):
            raise ValidationError("freeze source records are incomplete")
        openapi = official.get("cmp_openapi")
        model_files = official.get("cmp_model_files")
        project_score_record = official.get("project_score")
        if (
            not isinstance(openapi, dict)
            or not isinstance(openapi.get("record"), dict)
            or not isinstance(model_files, dict)
            or not isinstance(model_files.get("record"), dict)
            or not isinstance(project_score_record, dict)
        ):
            raise ValidationError("freeze official-source records are malformed")
        campaign_record = inputs.get("campaign_report")
        model_record = inputs.get("model_csv")
        broad_record = inputs.get("broad_label_source")
        training_record = inputs.get("training_cells")
        if not all(
            isinstance(record, dict)
            for record in (
                campaign_record,
                model_record,
                broad_record,
                training_record,
            )
        ):
            raise ValidationError("freeze input records are malformed")

        openapi_path = _path_from_record(
            openapi["record"], label="CMP OpenAPI snapshot"
        )
        model_files_path = _path_from_record(
            model_files["record"], label="CMP model-files snapshot"
        )
        project_score_path = _path_from_record(
            project_score_record, label="Project Score archive"
        )
        campaign_path = _path_from_record(campaign_record, label="campaign report")
        model_path = _path_from_record(model_record, label="Model.csv")
        broad_path = _path_from_record(broad_record, label="Broad label source")
        training_path = _path_from_record(training_record, label="training cells")

        existing = _import_existing_validation()
        expected_archive = existing.validate_external_archive(project_score_path)
        if (
            expected_archive.get("sha256") != EXPECTED_PROJECT_SCORE_SHA256
            or expected_archive.get("bytes") != EXPECTED_PROJECT_SCORE_BYTES
        ):
            raise ValidationError(
                "Project Score archive is not the fixed Sanger v2 release"
            )
        if expected_archive != project_score_record:
            raise ValidationError("Project Score source record was rewritten")
        columns, column_meta = existing.read_sanger_columns(project_score_path)
        campaign, expected_campaign_record = existing.validate_campaign_report(
            campaign_path
        )
        if expected_campaign_record != campaign_record:
            raise ValidationError("campaign report record was rewritten")
        expected_openapi = _validate_openapi_snapshot(openapi_path)
        if expected_openapi != openapi:
            raise ValidationError("CMP OpenAPI derivation differs from the freeze")
        rna_index, rna_meta = read_processed_rna_index(model_files_path)
        expected_model_files = {
            "url": CMP_MODEL_FILES_URL,
            "record": _file_record(model_files_path),
            "metadata": rna_meta,
        }
        if expected_model_files != model_files:
            raise ValidationError("CMP model-files derivation differs from the freeze")
        broad_ids = read_broad_label_ids(broad_path)
        training_ids = _read_unique_lines(training_path, label="DepPLM training cells")
        if len(training_ids) != EXPECTED_TRAINING_CELLS:
            raise ValidationError(
                f"training cohort must contain {EXPECTED_TRAINING_CELLS} cells"
            )
        expected_cells, cohort_meta = build_cohort_definition(
            model_csv=model_path,
            broad_label_ids=broad_ids,
            training_ids=training_ids,
            sanger_columns=columns,
            rna_index=rna_index,
        )
        if [asdict(cell) for cell in expected_cells] != cells:
            raise ValidationError("frozen external cohort cannot be rederived")
        if payload.get("cohort_audit") != cohort_meta:
            raise ValidationError("frozen cohort audit cannot be rederived")
        if payload.get("project_score_header_audit") != column_meta:
            raise ValidationError("Project Score header audit cannot be rederived")
        expected_hashes = {
            "all_broad_label_ids_sha256": _sha256_strings(broad_ids),
            "training_ids_sha256": _sha256_strings(training_ids),
        }
        if payload.get("identity_hashes") != expected_hashes:
            raise ValidationError("frozen identity hashes cannot be rederived")
        if specs != _select_campaign_training_specs(campaign):
            raise ValidationError("frozen training specifications cannot be rederived")
    return payload


def _http_get(
    url: str,
    *,
    expected_host: str,
    expected_path: str,
    expected_query: str,
    timeout: int = 90,
) -> bytes:
    _validate_https_transfer_url(
        url,
        expected_host=expected_host,
        expected_path=expected_path,
        expected_query=expected_query,
        label="HTTP request URL",
    )
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "DepPLM-Sanger-nonoverlap-validator/1.0"},
    )
    opener = urllib.request.build_opener(
        _PinnedHTTPSRedirectHandler(
            expected_host=expected_host,
            expected_path=expected_path,
            expected_query=expected_query,
        )
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            if getattr(response, "status", 200) != 200:
                raise ValidationError(f"HTTP status {response.status} for {url}")
            geturl = getattr(response, "geturl", None)
            if not callable(geturl):
                raise ValidationError("HTTP response does not expose its final URL")
            _validate_https_transfer_url(
                geturl(),
                expected_host=expected_host,
                expected_path=expected_path,
                expected_query=expected_query,
                label="HTTP final response URL",
            )
            return response.read()
    except (OSError, urllib.error.URLError) as exc:
        raise ValidationError(f"failed to download {url}") from exc


def _download_new(
    path: Path,
    url: str,
    *,
    expected_host: str,
    expected_path: str,
    expected_query: str,
    expected_bytes: int | None = None,
) -> None:
    content = _http_get(
        url,
        expected_host=expected_host,
        expected_path=expected_path,
        expected_query=expected_query,
    )
    if expected_bytes is not None and len(content) != expected_bytes:
        raise ValidationError(
            f"download byte count differs from CMP metadata for {url}: "
            f"{len(content)} != {expected_bytes}"
        )
    _atomic_new_bytes(path, content)


def _mutation_page_url(sidm: str, page: int) -> str:
    if (
        not _is_sanger_model_id(sidm)
        or isinstance(page, bool)
        or not isinstance(page, int)
        or page < 1
    ):
        raise ValidationError("invalid mutation endpoint identity or page")
    return CMP_MUTATION_ENDPOINT.format(sidm=sidm, page=page)


def _validate_mutation_api_url(
    url: Any,
    *,
    sidm: str,
    page: int,
    label: str,
    allow_http_metadata: bool,
) -> str:
    """Bind a mutation URL to one exact SIDM and logical pagination request."""
    expected_url = _mutation_page_url(sidm, page)
    expected_path = urllib.parse.urlsplit(expected_url).path
    canonical = _validate_cmp_api_url(
        url,
        expected_path=expected_path,
        label=label,
        allow_http_metadata=allow_http_metadata,
    )
    try:
        query = urllib.parse.parse_qsl(
            urllib.parse.urlsplit(canonical).query,
            keep_blank_values=True,
            strict_parsing=True,
            max_num_fields=8,
        )
    except ValueError as exc:
        raise ValidationError(f"{label} has an invalid mutation query") from exc
    required = Counter(
        {
            ("include", "gene"): 1,
            ("page[size]", "1000"): 1,
            ("page[number]", str(page)): 1,
        }
    )
    if Counter(query) != required:
        raise ValidationError(f"{label} has an unexpected mutation query")
    return canonical


def _validate_mutation_page(payload: Any, *, sidm: str, page: int) -> str | None:
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValidationError(f"mutation response is not JSON:API data for {sidm}")
    links = payload.get("links")
    if not isinstance(links, dict):
        raise ValidationError(f"mutation response lacks pagination links for {sidm}")
    self_link = links.get("self")
    _validate_mutation_api_url(
        self_link,
        sidm=sidm,
        page=page,
        label=f"mutation response self link for {sidm}",
        allow_http_metadata=True,
    )
    for item in payload["data"]:
        if not isinstance(item, dict):
            raise ValidationError("mutation data contains a non-object")
        attributes = item.get("attributes")
        relationships = item.get("relationships")
        model_values: set[str] = set()
        if isinstance(attributes, dict):
            for key in ("model_id", "model", "sanger_model_id"):
                value = attributes.get(key)
                if isinstance(value, str) and value.startswith("SIDM"):
                    model_values.add(value)
        if isinstance(relationships, dict):
            model_data = (relationships.get("model") or {}).get("data")
            if isinstance(model_data, dict) and isinstance(model_data.get("id"), str):
                model_values.add(model_data["id"])
        if model_values and model_values != {sidm}:
            raise ValidationError(
                f"mutation response mixes model identities for {sidm}"
            )
    next_url = links.get("next")
    if next_url in (None, ""):
        return None
    if not isinstance(next_url, str):
        raise ValidationError("mutation pagination next link is not a string")
    return _validate_mutation_api_url(
        next_url,
        sidm=sidm,
        page=page + 1,
        label=f"mutation pagination next link for {sidm}",
        allow_http_metadata=True,
    )


def download_frozen_raw(freeze_path: Path, cache_dir: Path) -> Path:
    """Download frozen feature inputs; never read the Project Score numeric matrix."""
    freeze = validate_freeze(freeze_path)
    cache_dir = cache_dir.resolve()
    if cache_dir.exists() and any(cache_dir.iterdir()):
        raise ValidationError("raw cache directory must be absent or empty")
    (cache_dir / "rna").mkdir(parents=True, exist_ok=True)
    (cache_dir / "mutations").mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for cell in freeze["cells"]:
        sidm = cell["sanger_model_id"]
        rna_relative = f"rna/{sidm}_rnaseq.csv.zip"
        rna_path = _safe_child(cache_dir, rna_relative)
        _download_new(
            rna_path,
            cell["rna_url"],
            expected_host=CMP_RNA_HOST,
            expected_path=_rna_download_path(sidm),
            expected_query="",
            expected_bytes=int(cell["rna_bytes"]),
        )
        _validate_rna_zip_structure(rna_path, sidm=sidm)
        records.append(
            {
                "kind": "processed_rna_zip",
                "sidm": sidm,
                "url": cell["rna_url"],
                **_file_record(rna_path, base=cache_dir),
            }
        )

        mutation_dir = cache_dir / "mutations" / sidm
        mutation_dir.mkdir(parents=True, exist_ok=False)
        url = _mutation_page_url(sidm, 1)
        page = 1
        seen_urls: set[str] = set()
        while url is not None:
            if url in seen_urls or page > 1000:
                raise ValidationError(f"mutation pagination cycle/overflow for {sidm}")
            url = _validate_mutation_api_url(
                url,
                sidm=sidm,
                page=page,
                label=f"mutation request URL for {sidm}",
                allow_http_metadata=False,
            )
            seen_urls.add(url)
            relative = f"mutations/{sidm}/page-{page:04d}.json"
            page_path = _safe_child(cache_dir, relative)
            content = _http_get(
                url,
                expected_host=CMP_API_HOST,
                expected_path=f"/models/{sidm}/datasets/mutations",
                expected_query=urllib.parse.urlsplit(url).query,
            )
            try:
                payload = json.loads(
                    content.decode("utf-8"),
                    object_pairs_hook=_reject_duplicate_object_pairs,
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValidationError(
                    f"invalid mutation JSON response for {sidm}"
                ) from exc
            next_url = _validate_mutation_page(payload, sidm=sidm, page=page)
            _atomic_new_bytes(page_path, content)
            records.append(
                {
                    "kind": "processed_mutations_json",
                    "sidm": sidm,
                    "page": page,
                    "url": url,
                    **_file_record(page_path, base=cache_dir),
                }
            )
            url = next_url
            page += 1
    manifest = {
        "schema": RAW_MANIFEST_SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "freeze_sha256": _sha256_file(freeze_path),
        "raw_data_redistribution_allowed": False,
        "records": records,
    }
    manifest_path = cache_dir / "raw-manifest.json"
    _atomic_new_json(manifest_path, manifest)
    return manifest_path


def _validate_rna_zip_structure(path: Path, *, sidm: str) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            members = [info for info in archive.infolist() if not info.is_dir()]
            if len(members) != 1 or not members[0].filename.lower().endswith(".csv"):
                raise ValidationError(
                    f"RNA ZIP must contain exactly one CSV for {sidm}"
                )
            info = members[0]
            member = Path(info.filename)
            if member.is_absolute() or ".." in member.parts or "\\" in info.filename:
                raise ValidationError(f"unsafe RNA ZIP member for {sidm}")
            if info.file_size <= 0 or info.file_size > 200_000_000:
                raise ValidationError(f"implausible RNA CSV size for {sidm}")
            if info.compress_size and info.file_size / info.compress_size > 250:
                raise ValidationError(f"RNA ZIP compression ratio is unsafe for {sidm}")
            return info.filename
    except zipfile.BadZipFile as exc:
        raise ValidationError(f"RNA input is not a readable ZIP for {sidm}") from exc


def _load_raw_manifest(
    freeze_path: Path, cache_dir: Path, *, redownload: bool = False
) -> tuple[dict[str, Any], dict[str, list[Mapping[str, Any]]]]:
    freeze = validate_freeze(freeze_path, rederive=False)
    cache_dir = cache_dir.resolve()
    manifest_path = cache_dir / "raw-manifest.json"
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema") != RAW_MANIFEST_SCHEMA:
        raise ValidationError("raw cache manifest schema is invalid")
    if manifest.get("freeze_sha256") != _sha256_file(freeze_path):
        raise ValidationError("raw cache is bound to another freeze")
    if manifest.get("raw_data_redistribution_allowed") is not False:
        raise ValidationError("raw cache manifest lacks the no-redistribution boundary")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValidationError("raw cache manifest contains no records")
    seen_paths: set[str] = set()
    seen_urls: set[str] = set()
    by_sidm: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    expected_files = {"raw-manifest.json"}
    for record in records:
        if not isinstance(record, dict):
            raise ValidationError("raw cache contains a non-object record")
        relative = record.get("path")
        url = record.get("url")
        sidm = record.get("sidm")
        if (
            not isinstance(relative, str)
            or relative in seen_paths
            or not isinstance(url, str)
            or url in seen_urls
            or not isinstance(sidm, str)
        ):
            raise ValidationError("raw cache has duplicate/invalid path, URL, or SIDM")
        seen_paths.add(relative)
        seen_urls.add(url)
        expected_files.add(relative)
        path = _safe_child(cache_dir, relative)
        _verify_file_record(path, record, label="raw cache artifact")
        if redownload:
            if record.get("kind") == "processed_rna_zip":
                expected_host = CMP_RNA_HOST
                expected_path = _rna_download_path(sidm)
                expected_query = ""
                _validate_https_transfer_url(
                    url,
                    expected_host=expected_host,
                    expected_path=expected_path,
                    expected_query=expected_query,
                    label=f"raw RNA manifest URL for {sidm}",
                )
            elif record.get("kind") == "processed_mutations_json":
                expected_host = CMP_API_HOST
                expected_path = f"/models/{sidm}/datasets/mutations"
                url = _validate_mutation_api_url(
                    url,
                    sidm=sidm,
                    page=record.get("page"),
                    label=f"raw mutation manifest URL for {sidm}",
                    allow_http_metadata=False,
                )
                expected_query = urllib.parse.urlsplit(url).query
            else:
                raise ValidationError(
                    f"raw cache contains an unknown record kind for {sidm}"
                )
            content = _http_get(
                url,
                expected_host=expected_host,
                expected_path=expected_path,
                expected_query=expected_query,
            )
            if len(content) != record.get("bytes") or hashlib.sha256(
                content
            ).hexdigest() != record.get("sha256"):
                raise ValidationError(
                    f"fresh download differs from frozen cache: {url}"
                )
        by_sidm[sidm].append(record)
    expected_cells = {cell["sanger_model_id"]: cell for cell in freeze["cells"]}
    if set(by_sidm) != set(expected_cells):
        raise ValidationError("raw cache SIDM inventory differs from the frozen cohort")
    for sidm, cell in expected_cells.items():
        sidm_records = by_sidm[sidm]
        rna_records = [
            record
            for record in sidm_records
            if record.get("kind") == "processed_rna_zip"
        ]
        mutation_records = sorted(
            (
                record
                for record in sidm_records
                if record.get("kind") == "processed_mutations_json"
            ),
            key=lambda record: int(record.get("page", -1)),
        )
        if len(rna_records) != 1 or len(mutation_records) < 1:
            raise ValidationError(f"raw cache is incomplete for {sidm}")
        if len(rna_records) + len(mutation_records) != len(sidm_records):
            raise ValidationError(
                f"raw cache contains an unknown record kind for {sidm}"
            )
        rna = rna_records[0]
        if (
            rna.get("path") != f"rna/{sidm}_rnaseq.csv.zip"
            or rna.get("url") != cell["rna_url"]
            or rna.get("bytes") != cell["rna_bytes"]
        ):
            raise ValidationError(
                f"raw RNA identity differs from the freeze for {sidm}"
            )
        _validate_rna_zip_structure(_safe_child(cache_dir, rna["path"]), sidm=sidm)
        expected_url: str | None = _mutation_page_url(sidm, 1)
        for page, record in enumerate(mutation_records, start=1):
            if (
                record.get("page") != page
                or record.get("path") != f"mutations/{sidm}/page-{page:04d}.json"
                or record.get("url") != expected_url
            ):
                raise ValidationError(
                    f"mutation pagination identity differs for {sidm} page {page}"
                )
            payload = _load_json(_safe_child(cache_dir, record["path"]))
            expected_url = _validate_mutation_page(payload, sidm=sidm, page=page)
        if expected_url is not None:
            raise ValidationError(f"mutation pagination is truncated for {sidm}")
    expected_dirs: set[str] = set()
    for relative in expected_files:
        for parent in Path(relative).parents:
            if parent == Path("."):
                break
            expected_dirs.add(parent.as_posix())
    actual_files: set[str] = set()
    actual_dirs: set[str] = set()
    for path in cache_dir.rglob("*"):
        if path.is_symlink():
            raise ValidationError(f"raw cache contains a symlink: {path}")
        if path.is_file():
            actual_files.add(path.relative_to(cache_dir).as_posix())
        elif path.is_dir():
            actual_dirs.add(path.relative_to(cache_dir).as_posix())
        else:
            raise ValidationError(f"raw cache contains a non-file entry: {path}")
    if actual_files != expected_files or actual_dirs != expected_dirs:
        raise ValidationError(
            "raw cache inventory contains missing or extra entries: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}, "
            f"missing_dirs={sorted(expected_dirs - actual_dirs)}, "
            f"extra_dirs={sorted(actual_dirs - expected_dirs)}"
        )
    return manifest, dict(by_sidm)


def parse_processed_rna(
    path: Path, *, sidm: str, expression_genes: Sequence[str]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map exact symbols and convert CMP TPM to the training log2(TPM+1) scale."""
    member = _validate_rna_zip_structure(path, sidm=sidm)
    wanted = set(expression_genes)
    candidates: defaultdict[str, list[float]] = defaultdict(list)
    data_sources: Counter[str] = Counter()
    rows = 0
    invalid_values = 0
    with zipfile.ZipFile(path) as archive, archive.open(member) as raw:
        reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
        required = {"model_id", "gene_symbol", "rsem_tpm", "data_source"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValidationError(f"processed RNA CSV lacks required fields for {sidm}")
        for row in reader:
            rows += 1
            if (row.get("model_id") or "").strip() != sidm:
                raise ValidationError(
                    f"processed RNA CSV mixes model identities for {sidm}"
                )
            source = (row.get("data_source") or "").strip() or "unspecified"
            data_sources[source] += 1
            symbol = (row.get("gene_symbol") or "").strip()
            if symbol not in wanted:
                continue
            token = (row.get("rsem_tpm") or "").strip()
            try:
                value = float(token)
            except ValueError:
                invalid_values += 1
                continue
            if not math.isfinite(value) or value < 0:
                invalid_values += 1
                continue
            candidates[symbol].append(value)
    index = {gene: position for position, gene in enumerate(expression_genes)}
    values = np.full(len(expression_genes), np.nan, dtype=np.float32)
    duplicates = []
    for symbol, observed in candidates.items():
        if len(observed) == 1:
            values[index[symbol]] = math.log2(observed[0] + 1.0)
        else:
            duplicates.append(symbol)
    finite = int(np.isfinite(values).sum())
    if finite < 1_000:
        raise ValidationError(f"fewer than 1,000 expression features map for {sidm}")
    return values, {
        "rows": rows,
        "mapped_unique_features": finite,
        "missing_features": int(values.size - finite),
        "duplicate_symbols_excluded": sorted(duplicates),
        "invalid_values_excluded": invalid_values,
        "data_source_row_counts": dict(sorted(data_sources.items())),
        "value_field": "rsem_tpm",
        "additional_transform": "log2(rsem_tpm + 1)",
    }


def parse_processed_mutations(
    page_paths: Sequence[Path], *, sidm: str
) -> tuple[set[str], dict[str, Any]]:
    """Derive a mutation-gene set from frozen official CMP JSON:API pages."""
    symbols: set[str] = set()
    mutation_ids: set[str] = set()
    records = 0
    unresolved = 0
    for page, path in enumerate(page_paths, start=1):
        payload = _load_json(path)
        _validate_mutation_page(payload, sidm=sidm, page=page)
        included_map: dict[str, str] = {}
        included = payload.get("included", [])
        if included is not None and not isinstance(included, list):
            raise ValidationError("mutation included section is not a list")
        for item in included or []:
            if not isinstance(item, dict) or item.get("type") != "gene":
                continue
            gene_id = str(item.get("id", ""))
            attributes = item.get("attributes")
            symbol = ""
            if isinstance(attributes, dict):
                symbol = str(
                    attributes.get("symbol") or attributes.get("gene_symbol") or ""
                ).strip()
            if gene_id and symbol:
                prior = included_map.get(gene_id)
                if prior is not None and prior != symbol:
                    raise ValidationError(
                        "CMP included gene ID maps to multiple symbols"
                    )
                included_map[gene_id] = symbol
        for item in payload["data"]:
            records += 1
            mutation_id = str(item.get("id", ""))
            if not mutation_id or mutation_id in mutation_ids:
                raise ValidationError(f"duplicate/empty mutation record ID for {sidm}")
            mutation_ids.add(mutation_id)
            attributes = item.get("attributes") if isinstance(item, dict) else None
            relationships = (
                item.get("relationships") if isinstance(item, dict) else None
            )
            symbol = ""
            if isinstance(attributes, dict):
                symbol = str(
                    attributes.get("gene_symbol") or attributes.get("symbol") or ""
                ).strip()
            if not symbol and isinstance(relationships, dict):
                gene_data = (relationships.get("gene") or {}).get("data")
                if isinstance(gene_data, dict):
                    symbol = included_map.get(str(gene_data.get("id", "")), "")
            if symbol:
                symbols.add(symbol)
            else:
                unresolved += 1
    return symbols, {
        "mutation_records": records,
        "unique_mutated_gene_symbols": len(symbols),
        "records_without_resolved_gene_symbol": unresolved,
        "pages": len(page_paths),
    }


def _raw_records_for_sidm(
    cache_dir: Path,
    records: Sequence[Mapping[str, Any]],
    sidm: str,
) -> tuple[Path, list[Path]]:
    rna = [record for record in records if record.get("kind") == "processed_rna_zip"]
    mutations = [
        record for record in records if record.get("kind") == "processed_mutations_json"
    ]
    if len(rna) != 1 or not mutations:
        raise ValidationError(f"raw cache is incomplete for {sidm}")
    pages = sorted(int(record.get("page", -1)) for record in mutations)
    if pages != list(range(1, len(pages) + 1)):
        raise ValidationError(f"mutation pages are incomplete or duplicated for {sidm}")
    mutation_by_page = {int(record["page"]): record for record in mutations}
    return (
        _safe_child(cache_dir, str(rna[0]["path"])),
        [_safe_child(cache_dir, str(mutation_by_page[page]["path"])) for page in pages],
    )


def _import_corrected_runner() -> Any:
    try:
        import run_corrected_experiments as corrected
    except ImportError:
        sys.path.insert(0, str(CODE_PATH.parent))
        import run_corrected_experiments as corrected
    return corrected


def _build_combined_base(
    freeze: Mapping[str, Any],
    cache_dir: Path,
    by_sidm: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[Any, Any, list[dict[str, Any]]]:
    corrected = _import_corrected_runner()

    base = corrected.load_base_data(pilot=False, seed=42, pilot_cells=0, pilot_genes=0)
    if len(base.cell_ids) != EXPECTED_TRAINING_CELLS:
        raise ValidationError("loaded DepPLM base is not the frozen 1,103-cell cohort")
    frozen_training = _read_unique_lines(TRAINING_CELLS, label="DepPLM training cells")
    if base.cell_ids != frozen_training:
        raise ValidationError("loaded DepPLM cell order differs from cell_lines.txt")

    expression_rows: list[np.ndarray] = []
    mutations: list[set[str]] = []
    audit: list[dict[str, Any]] = []
    external_cells = freeze["cells"]
    for cell in external_cells:
        sidm = cell["sanger_model_id"]
        records = by_sidm.get(sidm)
        if records is None:
            raise ValidationError(f"raw manifest lacks frozen cell {sidm}")
        rna_path, mutation_paths = _raw_records_for_sidm(cache_dir, records, sidm)
        expression, expression_meta = parse_processed_rna(
            rna_path,
            sidm=sidm,
            expression_genes=base.expression_gene_names,
        )
        mutated, mutation_meta = parse_processed_mutations(mutation_paths, sidm=sidm)
        expression_rows.append(expression)
        mutations.append(mutated)
        audit.append(
            {
                "depmap_model_id": cell["depmap_model_id"],
                "sanger_model_id": sidm,
                "expression": expression_meta,
                "mutations": mutation_meta,
            }
        )
    external_expression = np.stack(expression_rows).astype(np.float32, copy=False)
    n_external = len(external_cells)
    combined = corrected.BaseData(
        gene_effect=np.concatenate(
            [
                np.asarray(base.gene_effect, dtype=np.float32),
                np.full((len(base.genes), n_external), np.nan, dtype=np.float32),
            ],
            axis=1,
        ),
        expression_raw=np.concatenate(
            [base.expression_raw, external_expression], axis=0
        ),
        mutations_by_cell=[*base.mutations_by_cell, *mutations],
        embeddings=base.embeddings,
        edge_index=base.edge_index,
        genes=list(base.genes),
        cell_ids=[
            *base.cell_ids,
            *[cell["depmap_model_id"] for cell in external_cells],
        ],
        lineages=np.concatenate(
            [
                base.lineages,
                np.asarray([cell["lineage"] for cell in external_cells], dtype=str),
            ]
        ),
        source_cell_indices=np.concatenate(
            [base.source_cell_indices, np.full(n_external, -1, dtype=np.int64)]
        ),
        source_gene_indices=base.source_gene_indices,
        expression_gene_names=list(base.expression_gene_names),
    )
    return corrected, combined, audit


def _training_gene_mean(combined: Any) -> np.ndarray:
    gene_effect = np.asarray(combined.gene_effect, dtype=np.float64)
    if (
        gene_effect.ndim != 2
        or gene_effect.shape[0] != len(combined.genes)
        or gene_effect.shape[1] < EXPECTED_TRAINING_CELLS
    ):
        raise ValidationError("combined gene-effect matrix has an invalid shape")
    training = gene_effect[:, :EXPECTED_TRAINING_CELLS]
    finite_count = np.isfinite(training).sum(axis=1)
    if np.any(finite_count == 0):
        raise ValidationError(
            "at least one gene lacks a finite training dependency value"
        )
    means = np.nansum(training, axis=1, dtype=np.float64) / finite_count
    if means.shape != (len(combined.genes),) or not np.isfinite(means).all():
        raise ValidationError("training-only per-gene mean baseline is non-finite")
    return means


def _preprocessing_arrays(fold_data: Any) -> dict[str, np.ndarray]:
    return {
        "schema": np.asarray(PREPROCESSING_SCHEMA),
        "fit_indices": np.asarray(fold_data.fit_indices, dtype=np.int64),
        "expression_mean": np.asarray(fold_data.expression_fit.mean_, dtype=np.float64),
        "expression_scale": np.asarray(
            fold_data.expression_fit.scale_, dtype=np.float64
        ),
        "expression_pca_mean": np.asarray(
            fold_data.expression_fit.pca_mean_, dtype=np.float64
        ),
        "expression_components": np.asarray(
            fold_data.expression_fit.components_, dtype=np.float64
        ),
        "expression_explained_variance_ratio": np.asarray(
            fold_data.expression_fit.explained_variance_ratio_, dtype=np.float64
        ),
        "expression_n_output_components": np.asarray(
            fold_data.expression_fit.n_output_components, dtype=np.int64
        ),
        "mutation_vocabulary": np.asarray(fold_data.mutation.vocabulary, dtype=str),
        "mutation_token_order": np.asarray(fold_data.mutation.token_order, dtype=str),
        "lineage_categories": np.asarray(fold_data.lineage_fit.categories, dtype=str),
        "edge_index": np.asarray(fold_data.edge_index.cpu().numpy(), dtype=np.int64),
        "edge_weight": np.asarray(
            fold_data.edge_weight.cpu().numpy(), dtype=np.float32
        ),
        "selectivity_labels": np.asarray(
            fold_data.selectivity_labels, dtype=np.float32
        ),
    }


def _compare_preprocessing(path: Path, fold_data: Any) -> None:
    expected = _preprocessing_arrays(fold_data)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected):
            raise ValidationError("preprocessing state array inventory changed")
        for name, wanted in expected.items():
            observed = archive[name]
            if wanted.dtype.kind in "fc":
                if not np.allclose(
                    observed, wanted, rtol=1e-7, atol=1e-8, equal_nan=True
                ):
                    raise ValidationError(f"rederived preprocessing differs: {name}")
            elif not np.array_equal(observed, wanted):
                raise ValidationError(f"rederived preprocessing differs: {name}")


def _predict_rows(
    corrected: Any,
    model: Any,
    config: Mapping[str, Any],
    fold_data: Any,
    indices: np.ndarray,
    *,
    batch_size: int = 64,
) -> np.ndarray:
    import torch

    model = model.to("cpu")
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, indices.size, batch_size):
            rows = indices[start : start + batch_size]
            mutation = torch.from_numpy(fold_data.mutation.matrix[rows]).float()
            expression = torch.from_numpy(fold_data.expression[rows]).float()
            lineage = torch.from_numpy(fold_data.lineage[rows]).float()
            tokens = torch.from_numpy(fold_data.mutation.tokens[rows]).long()
            mask = torch.from_numpy(fold_data.mutation.token_mask[rows]).bool()
            prediction, _ = corrected._forward(
                model,
                config,
                mutation,
                expression,
                lineage,
                tokens,
                mask,
            )
            outputs.append(prediction.detach().cpu().float().numpy())
    prediction = np.concatenate(outputs, axis=0).astype(np.float32, copy=False)
    if not np.isfinite(prediction).all():
        raise ValidationError("external inference produced non-finite predictions")
    return prediction


def _prediction_arrays(
    prediction: np.ndarray,
    *,
    cells: Sequence[Mapping[str, Any]],
    genes: Sequence[str],
) -> dict[str, np.ndarray]:
    if prediction.shape != (len(cells), len(genes)):
        raise ValidationError("prediction dimensions do not match frozen identities")
    return {
        "schema": np.asarray(PREDICTION_SCHEMA),
        "prediction": np.asarray(prediction, dtype=np.float32),
        "depmap_model_ids": np.asarray(
            [cell["depmap_model_id"] for cell in cells], dtype=str
        ),
        "sanger_model_ids": np.asarray(
            [cell["sanger_model_id"] for cell in cells], dtype=str
        ),
        "gene_ids": np.asarray(genes, dtype=str),
        "primary_strict": np.asarray(
            [cell["primary_strict"] for cell in cells], dtype=bool
        ),
    }


def compare_replayed_predictions(
    path: Path, expected: Mapping[str, np.ndarray]
) -> None:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected):
            raise ValidationError("prediction array inventory changed")
        for name, wanted in expected.items():
            observed = archive[name]
            if name == "prediction":
                if not np.allclose(
                    observed,
                    wanted,
                    rtol=PREDICTION_RTOL,
                    atol=PREDICTION_ATOL,
                    equal_nan=False,
                ):
                    difference = float(np.max(np.abs(observed - wanted)))
                    raise ValidationError(
                        f"checkpoint replay differs from saved predictions: max_abs={difference}"
                    )
            elif not np.array_equal(observed, wanted):
                raise ValidationError(f"prediction identity array changed: {name}")


def _torch_save_new(path: Path, state_dict: Mapping[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ValidationError(f"refusing to overwrite checkpoint: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        cpu_state = {name: tensor.detach().cpu() for name, tensor in state_dict.items()}
        torch.save(cpu_state, temporary)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ValidationError(f"refusing to overwrite checkpoint: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _source_paths() -> list[Path]:
    return [
        CODE_PATH,
        ROOT / "code" / "sanger_independent_validation.py",
        ROOT / "code" / "run_corrected_experiments.py",
        ROOT / "code" / "corrected_protocol.py",
        ROOT / "code" / "embedding_provenance.py",
        ROOT / "code" / "03_embed_proteins.py",
        ROOT / "code" / "models.py",
        MODEL_CSV,
        BROAD_LABEL_CSV,
        TRAINING_CELLS,
        ROOT / "data" / "processed" / "genes.txt",
        ROOT / "data" / "processed" / "gene_effect.npy",
        ROOT / "data" / "processed" / "lineages.txt",
        ROOT / "data" / "processed" / "ppi_graph.npz",
        ROOT / "data" / "processed" / "gene_seqs.fa",
        ROOT / "data" / "processed" / "esm2_t6_8M_residue_chunked.npy",
        ROOT / "data" / "processed" / "esm2_t6_8M_residue_chunked.l2.npy",
        ROOT / "data" / "processed" / "esm2_t6_8M_residue_chunked.names.txt",
        ROOT / "data" / "processed" / "esm2_t6_8M_residue_chunked.manifest.json",
        ROOT / "data" / "depmap" / "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
        ROOT / "data" / "depmap" / "OmicsSomaticMutations.csv",
    ]


def _record_sources(paths: Sequence[Path]) -> list[dict[str, Any]]:
    return [_file_record(path) for path in paths]


def _verify_source_snapshot(records: Sequence[Mapping[str, Any]]) -> None:
    expected = {str(record.get("path")): record for record in records}
    if len(expected) != len(records):
        raise ValidationError("source snapshot contains duplicate paths")
    for path in _source_paths():
        relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
        record = expected.get(relative)
        if record is None:
            raise ValidationError(f"source snapshot lacks {relative}")
        _verify_file_record(path, record, label="bound source")


def _validated_embedding_provenance() -> dict[str, Any]:
    try:
        from embedding_provenance import validate_embedding_bundle
    except ImportError:
        sys.path.insert(0, str(CODE_PATH.parent))
        from embedding_provenance import validate_embedding_bundle
    try:
        return validate_embedding_bundle(ROOT)
    except Exception as exc:
        raise ValidationError(
            "canonical ESM embedding bundle did not validate"
        ) from exc


def _write_cohort_tsv(path: Path, cells: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "depmap_model_id",
        "sanger_model_id",
        "sanger_model_name",
        "lineage",
        "primary_strict",
        "sensitivity_1103_nonoverlap",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=fields, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    for cell in cells:
        writer.writerow({field: cell[field] for field in fields})
    _atomic_new_bytes(path, buffer.getvalue().encode("utf-8"))


def _optional_correlation(
    left: np.ndarray, right: np.ndarray, *, method: str
) -> float | None:
    valid = np.isfinite(left) & np.isfinite(right)
    x = np.asarray(left[valid], dtype=np.float64)
    y = np.asarray(right[valid], dtype=np.float64)
    if x.size < 3 or np.ptp(x) <= 0 or np.ptp(y) <= 0:
        return None
    if method == "pearson":
        value = stats.pearsonr(x, y).statistic
    elif method == "spearman":
        value = stats.spearmanr(x, y).statistic
    else:  # pragma: no cover - internal callers use only fixed methods
        raise ValueError(f"unknown correlation method: {method}")
    return float(value) if np.isfinite(value) else None


def _metric_summary(
    values: Sequence[float], *, eligible_count: int | None = None
) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    summary: dict[str, Any] = {
        "defined_count": len(finite),
        "mean": float(statistics.fmean(finite)) if finite else None,
        "median": float(statistics.median(finite)) if finite else None,
    }
    if eligible_count is not None:
        if eligible_count < len(finite) or eligible_count < 1:
            raise ValidationError("metric eligible count is inconsistent")
        summary.update(
            {
                "eligible_count": eligible_count,
                "undefined_count": eligible_count - len(finite),
                "defined_fraction": len(finite) / eligible_count,
            }
        )
    return summary


def _structured_metrics(
    prediction: np.ndarray,
    outcome: np.ndarray,
    training_gene_mean: np.ndarray,
) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.float64)
    outcome = np.asarray(outcome, dtype=np.float64)
    baseline = np.asarray(training_gene_mean, dtype=np.float64)
    if prediction.shape != outcome.shape or prediction.ndim != 2:
        raise ValidationError("prediction/outcome matrices do not share a 2-D shape")
    n_cells, n_genes = prediction.shape
    if n_cells < MIN_CELLS_PER_COHORT:
        raise ValidationError(
            f"analysis cohort has fewer than {MIN_CELLS_PER_COHORT} cells"
        )
    if n_genes < MIN_MATCHED_GENES:
        raise ValidationError(
            f"analysis has fewer than {MIN_MATCHED_GENES} matched genes"
        )
    if baseline.shape != (n_genes,) or not np.isfinite(baseline).all():
        raise ValidationError("training-gene-mean baseline is incomplete or misaligned")
    baseline_matrix = np.broadcast_to(baseline, prediction.shape)
    valid = (
        np.isfinite(prediction) & np.isfinite(outcome) & np.isfinite(baseline_matrix)
    )
    finite_per_cell = valid.sum(axis=1)
    if np.any(finite_per_cell < MIN_FINITE_GENES_PER_CELL):
        raise ValidationError(
            "at least one external cell has fewer than "
            f"{MIN_FINITE_GENES_PER_CELL} finite matched genes"
        )
    finite_per_gene = valid.sum(axis=0)
    eligible_genes = finite_per_gene >= MIN_FINITE_CELLS_PER_GENE
    if int(eligible_genes.sum()) < MIN_GENES_WITH_FINITE_CELL_PROFILES:
        raise ValidationError(
            "fewer than "
            f"{MIN_GENES_WITH_FINITE_CELL_PROFILES} genes have at least "
            f"{MIN_FINITE_CELLS_PER_GENE} finite external-cell observations"
        )

    x = prediction[valid]
    y = outcome[valid]
    b = baseline_matrix[valid]
    baseline_squared_error = float(np.sum(np.square(y - b), dtype=np.float64))
    if baseline_squared_error <= 0 or not math.isfinite(baseline_squared_error):
        raise ValidationError("training-gene-mean baseline has zero/non-finite SSE")
    model_squared_error = float(np.sum(np.square(y - x), dtype=np.float64))
    model_mae = float(np.mean(np.abs(x - y), dtype=np.float64))
    baseline_mae = float(np.mean(np.abs(b - y), dtype=np.float64))

    cell_pearson: list[float] = []
    cell_spearman: list[float] = []
    cell_residual_pearson: list[float] = []
    cell_residual_spearman: list[float] = []
    cell_mae: list[float] = []
    for row in range(n_cells):
        row_valid = valid[row]
        row_prediction = prediction[row, row_valid]
        row_outcome = outcome[row, row_valid]
        row_baseline = baseline_matrix[row, row_valid]
        pearson = _optional_correlation(row_prediction, row_outcome, method="pearson")
        spearman = _optional_correlation(row_prediction, row_outcome, method="spearman")
        residual_pearson = _optional_correlation(
            row_prediction - row_baseline,
            row_outcome - row_baseline,
            method="pearson",
        )
        residual_spearman = _optional_correlation(
            row_prediction - row_baseline,
            row_outcome - row_baseline,
            method="spearman",
        )
        if pearson is not None:
            cell_pearson.append(pearson)
        if spearman is not None:
            cell_spearman.append(spearman)
        if residual_pearson is not None:
            cell_residual_pearson.append(residual_pearson)
        if residual_spearman is not None:
            cell_residual_spearman.append(residual_spearman)
        cell_mae.append(float(np.mean(np.abs(row_prediction - row_outcome))))

    gene_pearson: list[float] = []
    gene_spearman: list[float] = []
    gene_mae: list[float] = []
    for column in np.flatnonzero(eligible_genes):
        column_valid = valid[:, column]
        column_prediction = prediction[column_valid, column]
        column_outcome = outcome[column_valid, column]
        pearson = _optional_correlation(
            column_prediction, column_outcome, method="pearson"
        )
        spearman = _optional_correlation(
            column_prediction, column_outcome, method="spearman"
        )
        if pearson is not None:
            gene_pearson.append(pearson)
        if spearman is not None:
            gene_spearman.append(spearman)
        gene_mae.append(float(np.mean(np.abs(column_prediction - column_outcome))))

    eligible_gene_count = int(eligible_genes.sum())
    gene_pearson_summary = _metric_summary(
        gene_pearson, eligible_count=eligible_gene_count
    )
    gene_spearman_summary = _metric_summary(
        gene_spearman, eligible_count=eligible_gene_count
    )
    gene_correlation_coverage_sufficient = (
        gene_pearson_summary["defined_count"] >= MIN_DEFINED_GENE_CORRELATION_COUNT
        and gene_spearman_summary["defined_count"] >= MIN_DEFINED_GENE_CORRELATION_COUNT
        and gene_pearson_summary["defined_fraction"]
        >= MIN_DEFINED_GENE_CORRELATION_FRACTION
        and gene_spearman_summary["defined_fraction"]
        >= MIN_DEFINED_GENE_CORRELATION_FRACTION
    )

    pooled_pearson = _optional_correlation(x, y, method="pearson")
    pooled_spearman = _optional_correlation(x, y, method="spearman")
    pooled_residual_pearson = _optional_correlation(x - b, y - b, method="pearson")
    return {
        "adequacy": {
            "paired_finite_entries": int(valid.sum()),
            "minimum_finite_genes_in_any_cell": int(finite_per_cell.min()),
            "genes_with_minimum_finite_cell_profile": int(eligible_genes.sum()),
            "minimum_cells_per_cohort_required": MIN_CELLS_PER_COHORT,
            "minimum_matched_genes_required": MIN_MATCHED_GENES,
            "minimum_finite_genes_per_cell_required": MIN_FINITE_GENES_PER_CELL,
            "minimum_finite_cells_per_gene_required": MIN_FINITE_CELLS_PER_GENE,
            "minimum_genes_with_finite_cell_profiles_required": (
                MIN_GENES_WITH_FINITE_CELL_PROFILES
            ),
            "minimum_defined_gene_correlation_fraction_required": (
                MIN_DEFINED_GENE_CORRELATION_FRACTION
            ),
            "minimum_defined_gene_correlation_count_required": (
                MIN_DEFINED_GENE_CORRELATION_COUNT
            ),
        },
        "baseline_comparison": {
            "baseline": "training_only_gene_mean",
            "raw_scale_mae": model_mae,
            "baseline_raw_scale_mae": baseline_mae,
            "delta_mae_vs_baseline": baseline_mae - model_mae,
            "residual_r2_vs_baseline": 1.0
            - model_squared_error / baseline_squared_error,
            "pooled_residual_pearson_r": pooled_residual_pearson,
        },
        "per_cell_across_genes": {
            "pearson_r": _metric_summary(cell_pearson),
            "spearman_rho": _metric_summary(cell_spearman),
            "residual_pearson_r": _metric_summary(cell_residual_pearson),
            "residual_spearman_rho": _metric_summary(cell_residual_spearman),
            "raw_scale_mae": _metric_summary(cell_mae),
        },
        "per_gene_across_cells": {
            "eligible_gene_count": eligible_gene_count,
            "pearson_r": gene_pearson_summary,
            "spearman_rho": gene_spearman_summary,
            "raw_scale_mae": _metric_summary(gene_mae),
            "minimum_defined_correlation_fraction_required": (
                MIN_DEFINED_GENE_CORRELATION_FRACTION
            ),
            "minimum_defined_correlation_count_required": (
                MIN_DEFINED_GENE_CORRELATION_COUNT
            ),
            "cell_context_signal_evaluable": gene_correlation_coverage_sufficient,
        },
        "secondary_pooled_matrix": {
            "pearson_r": pooled_pearson,
            "spearman_rho": pooled_spearman,
            "raw_scale_mae": model_mae,
        },
    }


def _load_project_score_outcomes(
    freeze: Mapping[str, Any], genes: Sequence[str], project_score: Path
) -> tuple[np.ndarray, list[Any], dict[str, Any]]:
    """The only function allowed to parse numeric Project Score outcomes."""
    existing = _import_existing_validation()
    matches = [
        existing.CellMatch(
            cohort_index=index,
            depmap_model_id=cell["depmap_model_id"],
            sanger_model_id=cell["sanger_model_id"],
            sanger_column=int(cell["sanger_column"]),
            sanger_model_name=cell["sanger_model_name"],
        )
        for index, cell in enumerate(freeze["cells"])
    ]
    return existing.load_sanger_matrix(project_score, genes, matches)


def _require_model_lock_before_outcomes(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict) or payload.get("schema") != MODEL_LOCK_SCHEMA:
        raise ValidationError(
            "numeric outcomes are sealed until a valid model lock exists"
        )
    if payload.get("numeric_sanger_outcomes_accessed") is not False:
        raise ValidationError(
            "model lock was not created before numeric outcome access"
        )
    return payload


def _verify_model_lock(
    run_dir: Path, freeze_path: Path, freeze: Mapping[str, Any]
) -> dict[str, Any]:
    lock = _require_model_lock_before_outcomes(run_dir / "model-lock.json")
    if lock.get("freeze_sha256") != _sha256_file(freeze_path):
        raise ValidationError("model lock is bound to another freeze")
    records = lock.get("artifacts")
    if not isinstance(records, list):
        raise ValidationError("model lock lacks artifact records")
    expected_paths = _expected_run_names() - {
        "model-lock.json",
        "analysis.json",
        "completion.json",
    }
    paths = [record.get("path") for record in records if isinstance(record, dict)]
    if (
        len(paths) != len(records)
        or len(paths) != len(set(paths))
        or set(paths) != expected_paths
    ):
        raise ValidationError(
            "model-lock artifact inventory is incomplete or duplicated"
        )
    for record in records:
        _verify_file_record(
            _safe_child(run_dir, record["path"]), record, label="model-lock artifact"
        )
    training = lock.get("training")
    if not isinstance(training, list):
        raise ValidationError("model lock lacks training records")
    by_seed = {
        record.get("seed"): record for record in training if isinstance(record, dict)
    }
    if len(by_seed) != len(training) or set(by_seed) != set(CANONICAL_SEEDS):
        raise ValidationError("model-lock training seeds are incomplete or duplicated")
    corrected = _import_corrected_runner()
    for seed in CANONICAL_SEEDS:
        record = by_seed[seed]
        spec = freeze["training_specs"][str(seed)]
        expected = {
            "preprocessing_seed": corrected.derive_seed(seed, "sanger-full-preprocess"),
            "training_seed": corrected.derive_seed(seed, "sanger-full-train"),
            "config": spec["config"],
            "epochs": spec["epochs"],
        }
        for field, value in expected.items():
            if record.get(field) != value:
                raise ValidationError(f"model-lock seed {seed} has changed {field}")
        prediction_digest = _validated_sha256(
            record.get("prediction_sha256"),
            label=f"model-lock seed {seed} prediction digest",
        )
        _verify_locked_prediction_digest(
            run_dir / f"predictions-seed-{seed}.npz",
            prediction_digest,
            seed=seed,
        )
    if not isinstance(lock.get("feature_audit"), dict):
        raise ValidationError("model lock lacks the frozen feature audit")
    return lock


def _verify_locked_prediction_digest(
    path: Path, prediction_digest: str, *, seed: int
) -> None:
    """Bind the model lock to the saved prediction array, not a later replay."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "prediction" not in archive.files:
                raise ValidationError(
                    f"saved predictions lack the prediction array for seed {seed}"
                )
            saved_prediction = np.asarray(archive["prediction"])
    except ValidationError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise ValidationError(
            f"saved predictions cannot be read for seed {seed}"
        ) from exc
    if prediction_digest != _sha256_array(saved_prediction):
        raise ValidationError(
            f"model-lock prediction digest differs from saved predictions for seed {seed}"
        )


def _analysis_from_predictions(
    *,
    freeze: Mapping[str, Any],
    predictions: Mapping[str, np.ndarray],
    training_gene_mean: np.ndarray,
    genes: Sequence[str],
    project_score: Path,
) -> dict[str, Any]:
    outcomes, gene_matches, gene_meta = _load_project_score_outcomes(
        freeze, genes, project_score
    )
    gene_indices = np.asarray(
        [match.cohort_index for match in gene_matches], dtype=np.int64
    )
    primary_rows = np.asarray(
        [index for index, cell in enumerate(freeze["cells"]) if cell["primary_strict"]],
        dtype=np.int64,
    )
    all_rows = np.arange(len(freeze["cells"]), dtype=np.int64)
    if primary_rows.size < MIN_CELLS_PER_COHORT:
        raise ValidationError(
            f"strict primary cohort has fewer than {MIN_CELLS_PER_COHORT} cells"
        )
    if all_rows.size < MIN_CELLS_PER_COHORT:
        raise ValidationError(
            f"sensitivity cohort has fewer than {MIN_CELLS_PER_COHORT} cells"
        )
    training_gene_mean = np.asarray(training_gene_mean, dtype=np.float64)
    if training_gene_mean.shape != (len(genes),):
        raise ValidationError("training-gene-mean baseline does not align to genes")
    aligned_baseline = training_gene_mean[gene_indices]
    result_rows = []
    for name, prediction in predictions.items():
        aligned = prediction[:, gene_indices]
        result_rows.append(
            {
                "prediction": name,
                "cohort": "primary_all_broad_nonoverlap",
                "n_cells": int(primary_rows.size),
                "n_genes": int(gene_indices.size),
                **_structured_metrics(
                    aligned[primary_rows],
                    outcomes[primary_rows],
                    aligned_baseline,
                ),
            }
        )
        result_rows.append(
            {
                "prediction": name,
                "cohort": "sensitivity_1103_nonoverlap_less_strict",
                "n_cells": int(all_rows.size),
                "n_genes": int(gene_indices.size),
                **_structured_metrics(
                    aligned[all_rows],
                    outcomes[all_rows],
                    aligned_baseline,
                ),
            }
        )
    return {
        "schema": ANALYSIS_SCHEMA,
        "scientific_result": True,
        "interpretation": "descriptive external cell-model validation",
        "primary_cohort": "primary_all_broad_nonoverlap",
        "sensitivity_cohort": "sensitivity_1103_nonoverlap_less_strict",
        "training_gene_mean_sha256": _sha256_array(training_gene_mean),
        "metrics": result_rows,
        "gene_mapping": gene_meta,
        "matched_genes": [
            {
                "cohort_index": int(match.cohort_index),
                "symbol": match.symbol,
                "sanger_gene_id": match.sanger_gene_id,
                "ensembl_gene_id": match.ensembl_gene_id,
            }
            for match in gene_matches
        ],
        "limitations": [
            "The strict primary cohort is finite and all metrics are descriptive.",
            "CMP RNA sources and processing pipelines can differ by model and from DepMap training RNA.",
            "No batch correction is fitted on external cells because that would adapt preprocessing to the evaluation cohort.",
            "Project Score and Broad gene-effect values may differ in scale and screening pipeline; raw-scale MAE is therefore descriptive.",
            "Pooled matrix correlations are secondary because between-gene effects can mask weak cell-context prediction.",
            "This is not patient, clinical, prospective, in vivo, or wet-lab validation.",
        ],
    }


def _analysis_gate_summary(
    analysis: Mapping[str, Any], freeze: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate prespecified adequacy gates without filtering metric direction."""
    if analysis.get("schema") != ANALYSIS_SCHEMA:
        raise ValidationError("analysis schema is invalid")
    if analysis.get("scientific_result") is not True:
        raise ValidationError("analysis is not marked as a scientific result")
    if analysis.get("interpretation") != "descriptive external cell-model validation":
        raise ValidationError("analysis interpretation changed")
    if (
        analysis.get("primary_cohort") != "primary_all_broad_nonoverlap"
        or analysis.get("sensitivity_cohort")
        != "sensitivity_1103_nonoverlap_less_strict"
    ):
        raise ValidationError("analysis cohort labels changed")

    cells = freeze.get("cells")
    if not isinstance(cells, list):
        raise ValidationError("freeze cells are unavailable for analysis gates")
    primary_cells = sum(
        isinstance(cell, dict) and cell.get("primary_strict") is True for cell in cells
    )
    sensitivity_cells = len(cells)
    if primary_cells < MIN_CELLS_PER_COHORT:
        raise ValidationError("primary cohort fails the frozen cell-count gate")
    if sensitivity_cells < MIN_CELLS_PER_COHORT:
        raise ValidationError("sensitivity cohort fails the frozen cell-count gate")

    matched = analysis.get("matched_genes")
    if not isinstance(matched, list) or len(matched) < MIN_MATCHED_GENES:
        raise ValidationError("analysis fails the frozen matched-gene gate")
    matched_identities: set[tuple[int, str, str, str]] = set()
    cohort_indices: set[int] = set()
    for index, record in enumerate(matched):
        if not isinstance(record, dict):
            raise ValidationError(f"matched gene {index} is not an object")
        cohort_index = record.get("cohort_index")
        strings = [
            record.get("symbol"),
            record.get("sanger_gene_id"),
            record.get("ensembl_gene_id"),
        ]
        if (
            isinstance(cohort_index, bool)
            or not isinstance(cohort_index, int)
            or cohort_index < 0
            or any(not isinstance(value, str) or not value for value in strings)
        ):
            raise ValidationError(f"matched gene {index} identity is invalid")
        identity = (cohort_index, *strings)
        if identity in matched_identities or cohort_index in cohort_indices:
            raise ValidationError("matched gene identities are duplicated")
        matched_identities.add(identity)
        cohort_indices.add(cohort_index)

    predictions = {f"seed-{seed}" for seed in CANONICAL_SEEDS} | {"ensemble_mean"}
    cohort_sizes = {
        "primary_all_broad_nonoverlap": primary_cells,
        "sensitivity_1103_nonoverlap_less_strict": sensitivity_cells,
    }
    expected_rows = {
        (prediction, cohort) for prediction in predictions for cohort in cohort_sizes
    }
    metrics = analysis.get("metrics")
    if not isinstance(metrics, list) or len(metrics) != len(expected_rows):
        raise ValidationError("analysis metric-row inventory is incomplete")

    seen_rows: set[tuple[str, str]] = set()
    observed_finite_genes: list[int] = []
    observed_finite_profiles: list[int] = []
    observed_defined_counts: list[int] = []
    observed_defined_fractions: list[float] = []
    required_adequacy = {
        "minimum_cells_per_cohort_required": MIN_CELLS_PER_COHORT,
        "minimum_matched_genes_required": MIN_MATCHED_GENES,
        "minimum_finite_genes_per_cell_required": MIN_FINITE_GENES_PER_CELL,
        "minimum_finite_cells_per_gene_required": MIN_FINITE_CELLS_PER_GENE,
        "minimum_genes_with_finite_cell_profiles_required": (
            MIN_GENES_WITH_FINITE_CELL_PROFILES
        ),
        "minimum_defined_gene_correlation_count_required": (
            MIN_DEFINED_GENE_CORRELATION_COUNT
        ),
        "minimum_defined_gene_correlation_fraction_required": (
            MIN_DEFINED_GENE_CORRELATION_FRACTION
        ),
    }
    for index, row in enumerate(metrics):
        if not isinstance(row, dict):
            raise ValidationError(f"analysis metric row {index} is not an object")
        identity = (row.get("prediction"), row.get("cohort"))
        if identity not in expected_rows or identity in seen_rows:
            raise ValidationError(
                "analysis metric identities are invalid or duplicated"
            )
        seen_rows.add(identity)
        expected_cells = cohort_sizes[identity[1]]
        if row.get("n_cells") != expected_cells:
            raise ValidationError("analysis metric cohort size differs from the freeze")
        if row.get("n_genes") != len(matched):
            raise ValidationError("analysis metric gene count differs from its mapping")

        adequacy = row.get("adequacy")
        if not isinstance(adequacy, dict) or any(
            adequacy.get(field) != expected
            for field, expected in required_adequacy.items()
        ):
            raise ValidationError("analysis metric adequacy policy changed")
        minimum_finite_genes = adequacy.get("minimum_finite_genes_in_any_cell")
        finite_profiles = adequacy.get("genes_with_minimum_finite_cell_profile")
        paired_entries = adequacy.get("paired_finite_entries")
        if (
            isinstance(minimum_finite_genes, bool)
            or not isinstance(minimum_finite_genes, int)
            or minimum_finite_genes < MIN_FINITE_GENES_PER_CELL
            or isinstance(finite_profiles, bool)
            or not isinstance(finite_profiles, int)
            or finite_profiles < MIN_GENES_WITH_FINITE_CELL_PROFILES
            or isinstance(paired_entries, bool)
            or not isinstance(paired_entries, int)
            or paired_entries < expected_cells * MIN_FINITE_GENES_PER_CELL
        ):
            raise ValidationError("analysis finite-coverage gate failed")
        observed_finite_genes.append(minimum_finite_genes)
        observed_finite_profiles.append(finite_profiles)

        per_gene = row.get("per_gene_across_cells")
        if not isinstance(per_gene, dict):
            raise ValidationError("analysis lacks per-gene cell-context metrics")
        if (
            per_gene.get("minimum_defined_correlation_count_required")
            != MIN_DEFINED_GENE_CORRELATION_COUNT
            or per_gene.get("minimum_defined_correlation_fraction_required")
            != MIN_DEFINED_GENE_CORRELATION_FRACTION
            or per_gene.get("cell_context_signal_evaluable") is not True
        ):
            raise ValidationError("analysis gene-correlation coverage gate failed")
        eligible_count = per_gene.get("eligible_gene_count")
        if (
            isinstance(eligible_count, bool)
            or not isinstance(eligible_count, int)
            or eligible_count < MIN_GENES_WITH_FINITE_CELL_PROFILES
            or eligible_count != finite_profiles
        ):
            raise ValidationError("analysis eligible-gene count is invalid")
        for field in ("pearson_r", "spearman_rho"):
            summary = per_gene.get(field)
            if not isinstance(summary, dict):
                raise ValidationError(f"analysis per-gene {field} summary is missing")
            defined_count = summary.get("defined_count")
            defined_fraction = summary.get("defined_fraction")
            if (
                isinstance(defined_count, bool)
                or not isinstance(defined_count, int)
                or defined_count < MIN_DEFINED_GENE_CORRELATION_COUNT
                or isinstance(defined_fraction, bool)
                or not isinstance(defined_fraction, (int, float))
                or not math.isfinite(float(defined_fraction))
                or float(defined_fraction) < MIN_DEFINED_GENE_CORRELATION_FRACTION
            ):
                raise ValidationError(
                    "analysis defined-correlation coverage gate failed"
                )
            observed_defined_counts.append(defined_count)
            observed_defined_fractions.append(float(defined_fraction))
    if seen_rows != expected_rows:
        raise ValidationError("analysis metric-row inventory is incomplete")

    return {
        "primary_cells": primary_cells,
        "sensitivity_cells": sensitivity_cells,
        "matched_genes": len(matched),
        "metric_rows_checked": len(metrics),
        "minimum_observed_finite_genes_per_cell": min(observed_finite_genes),
        "minimum_observed_genes_with_finite_cell_profiles": min(
            observed_finite_profiles
        ),
        "minimum_observed_defined_gene_correlation_count": min(observed_defined_counts),
        "minimum_observed_defined_gene_correlation_fraction": min(
            observed_defined_fractions
        ),
    }


def _artifact_records(run_dir: Path, names: Sequence[str]) -> list[dict[str, Any]]:
    if len(names) != len(set(names)):
        raise ValidationError("duplicate artifact name")
    return [_file_record(run_dir / name, base=run_dir) for name in sorted(names)]


def run_validation(
    *,
    freeze_path: Path,
    cache_dir: Path,
    output_root: Path,
    project_score: Path = PROJECT_SCORE,
    device: str = "cuda",
) -> Path:
    """Train frozen models first, lock predictions, then read Sanger outcomes."""
    import torch

    freeze = validate_freeze(freeze_path)
    if device == "cuda" and not torch.cuda.is_available():
        raise ValidationError("CUDA requested but unavailable")
    if device not in {"cpu", "cuda"}:
        raise ValidationError("device must be cpu or cuda")
    existing = _import_existing_validation()
    existing.validate_external_archive(project_score)
    raw_manifest, by_sidm = _load_raw_manifest(freeze_path, cache_dir)

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = output_root.resolve() / f"{timestamp}-{uuid.uuid4().hex[:12]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    cells = freeze["cells"]
    embedding_provenance = _validated_embedding_provenance()
    source_records = _record_sources(_source_paths())
    source_bindings = {
        "schema": SOURCE_BINDINGS_SCHEMA,
        "freeze": _file_record(freeze_path, base=None),
        "raw_manifest": _file_record(cache_dir / "raw-manifest.json", base=None),
        "project_score": _file_record(project_score, base=None),
        "sources": source_records,
        "embedding_provenance": embedding_provenance,
        "environment": _environment_manifest(),
        "device_used_for_training": device,
        "raw_data_packaged": False,
    }
    _atomic_new_json(run_dir / "source-bindings.json", source_bindings)
    _write_cohort_tsv(run_dir / "cohort.tsv", cells)

    corrected, combined, feature_audit = _build_combined_base(
        freeze, cache_dir, by_sidm
    )
    training_gene_mean = _training_gene_mean(combined)
    training_idx = np.arange(EXPECTED_TRAINING_CELLS, dtype=np.int64)
    external_idx = np.arange(
        EXPECTED_TRAINING_CELLS, len(combined.cell_ids), dtype=np.int64
    )
    predictions: dict[str, np.ndarray] = {}
    model_artifacts: list[str] = ["source-bindings.json", "cohort.tsv"]
    training_records = []
    for seed in CANONICAL_SEEDS:
        spec = freeze["training_specs"][str(seed)]
        preprocessing_seed = corrected.derive_seed(seed, "sanger-full-preprocess")
        training_seed = corrected.derive_seed(seed, "sanger-full-train")
        fold_data = corrected.fit_fold_data(
            combined,
            training_idx,
            seed=preprocessing_seed,
            n_expr_pc=128,
            top_mut=5000,
        )
        preprocessing_name = f"preprocessing-seed-{seed}.npz"
        _atomic_new_npz(run_dir / preprocessing_name, _preprocessing_arrays(fold_data))
        corrected.set_global_seed(training_seed)
        model = corrected.build_model(spec["config"], combined, fold_data)
        model, training = corrected._train_epochs(
            model,
            spec["config"],
            combined,
            fold_data,
            training_idx,
            epochs=int(spec["epochs"]),
            batch_size=64,
            gene_subsample=6000,
            device=device,
            seed=training_seed,
            validation_idx=None,
            patience=5,
            scheduler_horizon=20,
        )
        prediction = _predict_rows(
            corrected, model, spec["config"], fold_data, external_idx
        )
        checkpoint_name = f"checkpoint-seed-{seed}.pt"
        prediction_name = f"predictions-seed-{seed}.npz"
        _torch_save_new(run_dir / checkpoint_name, model.state_dict())
        _atomic_new_npz(
            run_dir / prediction_name,
            _prediction_arrays(prediction, cells=cells, genes=combined.genes),
        )
        predictions[f"seed-{seed}"] = prediction
        model_artifacts.extend([preprocessing_name, checkpoint_name, prediction_name])
        training_records.append(
            {
                "seed": seed,
                "preprocessing_seed": preprocessing_seed,
                "training_seed": training_seed,
                "config": spec["config"],
                "epochs": spec["epochs"],
                "training": training,
                "prediction_sha256": _sha256_array(prediction),
            }
        )
    ensemble = np.mean(
        np.stack(list(predictions.values())), axis=0, dtype=np.float64
    ).astype(np.float32)
    ensemble_name = "predictions-ensemble.npz"
    _atomic_new_npz(
        run_dir / ensemble_name,
        _prediction_arrays(ensemble, cells=cells, genes=combined.genes),
    )
    predictions["ensemble_mean"] = ensemble
    model_artifacts.append(ensemble_name)

    _verify_source_snapshot(source_records)
    if (
        _sha256_file(cache_dir / "raw-manifest.json")
        != source_bindings["raw_manifest"]["sha256"]
    ):
        raise ValidationError("raw cache manifest changed during model fitting")
    model_lock = {
        "schema": MODEL_LOCK_SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "numeric_sanger_outcomes_accessed": False,
        "freeze_sha256": _sha256_file(freeze_path),
        "feature_audit": feature_audit,
        "training": training_records,
        "artifacts": _artifact_records(run_dir, model_artifacts),
    }
    _atomic_new_json(run_dir / "model-lock.json", model_lock)

    # This guard is deliberately adjacent to the sole numeric-outcome parser.
    _require_model_lock_before_outcomes(run_dir / "model-lock.json")
    analysis = _analysis_from_predictions(
        freeze=freeze,
        predictions=predictions,
        training_gene_mean=training_gene_mean,
        genes=combined.genes,
        project_score=project_score,
    )
    _atomic_new_json(run_dir / "analysis.json", analysis)
    artifact_names = [*model_artifacts, "model-lock.json", "analysis.json"]
    completion = {
        "schema": COMPLETION_SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scientific_result": True,
        "verification_required": True,
        "artifacts": _artifact_records(run_dir, artifact_names),
    }
    _atomic_new_json(run_dir / "completion.json", completion)
    return run_dir


def _expected_run_names() -> set[str]:
    names = {
        "source-bindings.json",
        "cohort.tsv",
        "model-lock.json",
        "analysis.json",
        "completion.json",
        "predictions-ensemble.npz",
    }
    for seed in CANONICAL_SEEDS:
        names.update(
            {
                f"preprocessing-seed-{seed}.npz",
                f"checkpoint-seed-{seed}.pt",
                f"predictions-seed-{seed}.npz",
            }
        )
    return names


def _public_artifact_kinds() -> dict[str, str]:
    kinds = {
        "cohort.tsv": "derived_cohort",
        "model-lock.json": "model_lock",
        "analysis.json": "analysis",
        "completion.json": "completion",
        "predictions-ensemble.npz": "outer_prediction",
    }
    for seed in CANONICAL_SEEDS:
        kinds.update(
            {
                f"preprocessing-seed-{seed}.npz": "preprocessing_state",
                f"checkpoint-seed-{seed}.pt": "model_checkpoint",
                f"predictions-seed-{seed}.npz": "outer_prediction",
            }
        )
    return kinds


def _validate_portable_artifact_inventory(
    records: Any, *, run_dir: Path | None = None
) -> list[dict[str, Any]]:
    kinds = _public_artifact_kinds()
    if not isinstance(records, list) or len(records) != len(kinds):
        raise ValidationError("portable artifact inventory is incomplete")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {
            "path",
            "bytes",
            "sha256",
            "kind",
        }:
            raise ValidationError(
                f"portable artifact {index} must contain path/bytes/sha256/kind only"
            )
        relative = _validated_posix_relative_path(
            record.get("path"), label=f"portable artifact {index} path"
        )
        if relative not in kinds or relative in seen:
            raise ValidationError("portable artifact path is unknown or duplicated")
        size = record.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValidationError("portable artifact byte count is invalid")
        digest = _validated_sha256(
            record.get("sha256"), label=f"portable artifact {index} digest"
        )
        if record.get("kind") != kinds[relative]:
            raise ValidationError("portable artifact kind does not match its path")
        if run_dir is not None:
            _verify_file_record(
                _safe_child(run_dir, relative),
                {"bytes": size, "sha256": digest},
                label="portable verified artifact",
            )
        seen.add(relative)
        normalized.append(dict(record))
    if seen != set(kinds):
        raise ValidationError("portable artifact inventory paths are incomplete")
    return normalized


def _portable_artifact_inventory(run_dir: Path) -> list[dict[str, Any]]:
    records = []
    for relative, kind in sorted(_public_artifact_kinds().items()):
        record = _file_record(_safe_child(run_dir, relative), base=run_dir)
        records.append({**record, "kind": kind})
    return _validate_portable_artifact_inventory(records, run_dir=run_dir)


def validate_exact_run_inventory(run_dir: Path) -> None:
    run_dir = run_dir.resolve()
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ValidationError("run directory must be a non-symlink directory")
    actual: set[str] = set()
    for path in run_dir.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValidationError(f"run directory contains symlink/non-file: {path}")
        actual.add(path.name)
    expected = _expected_run_names()
    if actual != expected:
        raise ValidationError(
            f"run inventory mismatch: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _verify_completion(
    run_dir: Path, *, trusted_completion_sha256: str
) -> dict[str, Any]:
    trusted = _validated_sha256(
        trusted_completion_sha256, label="trusted completion digest"
    )
    if _sha256_file(run_dir / "completion.json") != trusted:
        raise ValidationError(
            "completion.json differs from the externally trusted archive digest"
        )
    completion = _load_json(run_dir / "completion.json")
    expected_fields = {
        "schema",
        "created_utc",
        "scientific_result",
        "verification_required",
        "artifacts",
    }
    if not isinstance(completion, dict) or set(completion) != expected_fields:
        raise ValidationError("completion fields are invalid")
    if completion.get("schema") != COMPLETION_SCHEMA:
        raise ValidationError("completion schema is invalid")
    created_utc = completion.get("created_utc")
    if not isinstance(created_utc, str) or not created_utc.endswith("Z"):
        raise ValidationError("completion timestamp is invalid")
    if completion.get("scientific_result") is not True:
        raise ValidationError("completion is not a scientific result")
    if completion.get("verification_required") is not True:
        raise ValidationError("completion must require independent verification")
    records = completion.get("artifacts")
    if not isinstance(records, list):
        raise ValidationError("completion lacks artifact records")
    expected = _expected_run_names() - {"completion.json"}
    if any(
        not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}
        for record in records
    ):
        raise ValidationError("completion artifact records are invalid")
    paths = [record["path"] for record in records]
    if (
        len(paths) != len(records)
        or len(set(paths)) != len(paths)
        or set(paths) != expected
    ):
        raise ValidationError("completion artifact paths are incomplete or duplicated")
    for record in records:
        path = _safe_child(run_dir, record["path"])
        _verify_file_record(path, record, label="completed artifact")
    return completion


def _compare_analysis_value(observed: Any, expected: Any, *, path: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping) or set(observed) != set(expected):
            raise ValidationError(f"recomputed analysis fields differ: {path}")
        for key, expected_value in expected.items():
            _compare_analysis_value(observed[key], expected_value, path=f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise ValidationError(f"recomputed analysis list differs: {path}")
        for index, (observed_value, expected_value) in enumerate(
            zip(observed, expected, strict=True)
        ):
            _compare_analysis_value(
                observed_value, expected_value, path=f"{path}[{index}]"
            )
        return
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if type(observed) is not type(expected) or observed != expected:
            raise ValidationError(f"recomputed analysis identity differs: {path}")
        return
    if isinstance(expected, int):
        if (
            isinstance(observed, bool)
            or not isinstance(observed, int)
            or observed != expected
        ):
            raise ValidationError(f"recomputed analysis integer differs: {path}")
        return
    if isinstance(expected, float):
        if isinstance(observed, bool) or not isinstance(observed, (int, float)):
            raise ValidationError(f"recomputed analysis metric type differs: {path}")
        if not math.isclose(
            float(observed),
            expected,
            rel_tol=ANALYSIS_RTOL,
            abs_tol=ANALYSIS_ATOL,
        ):
            raise ValidationError(f"recomputed analysis metric differs: {path}")
        return
    if type(observed) is not type(expected) or observed != expected:
        raise ValidationError(f"recomputed analysis value differs: {path}")


def _compare_analysis(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    if set(observed) != set(expected):
        raise ValidationError("analysis fields changed")
    observed_copy = json.loads(json.dumps(observed))
    expected_copy = json.loads(json.dumps(expected))
    observed_metrics = observed_copy.pop("metrics")
    expected_metrics = expected_copy.pop("metrics")
    if observed_copy != expected_copy:
        raise ValidationError("analysis identities or metadata changed")
    _compare_analysis_value(observed_metrics, expected_metrics, path="metrics")


def _portable_local_input_path(path: Path, *, role: str) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        relative = f"local-only/{role}/{resolved.name}"
    return _validated_posix_relative_path(relative, label=f"{role} binding path")


def _portable_source_bindings(
    *,
    bindings: Mapping[str, Any],
    freeze_path: Path,
    cache_dir: Path,
    project_score: Path,
) -> dict[str, Any]:
    local_specs = (
        (
            "protocol_freeze",
            freeze_path,
            bindings.get("freeze"),
            "frozen_protocol",
            "author_review_required",
        ),
        (
            "raw_cache_manifest",
            cache_dir / "raw-manifest.json",
            bindings.get("raw_manifest"),
            "local_raw_input_manifest",
            REDISTRIBUTION_STATUS,
        ),
        (
            "project_score_archive",
            project_score,
            bindings.get("project_score"),
            "local_raw_input_archive",
            REDISTRIBUTION_STATUS,
        ),
    )
    local_inputs: list[dict[str, Any]] = []
    for role, path, bound, kind, status in local_specs:
        if not isinstance(bound, Mapping):
            raise ValidationError(f"source binding for {role} is missing")
        _verify_file_record(path, bound, label=f"bound {role}")
        local_inputs.append(
            {
                "role": role,
                "path": _portable_local_input_path(path, role=role),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
                "kind": kind,
                "payload_packaged": False,
                "redistribution_status": status,
            }
        )

    source_records = bindings.get("sources")
    if not isinstance(source_records, list):
        raise ValidationError("bound source inventory is missing")
    by_path: dict[str, Mapping[str, Any]] = {}
    for record in source_records:
        if not isinstance(record, Mapping):
            raise ValidationError("bound source inventory contains a non-object")
        relative = _validated_posix_relative_path(
            record.get("path"), label="bound source path"
        )
        if relative in by_path:
            raise ValidationError("bound source inventory contains duplicate paths")
        by_path[relative] = record
    expected_paths = [
        path.resolve().relative_to(ROOT.resolve()).as_posix()
        for path in _source_paths()
    ]
    if set(by_path) != set(expected_paths):
        raise ValidationError("bound source inventory is incomplete or has extras")
    portable_sources: list[dict[str, Any]] = []
    for relative in sorted(expected_paths):
        record = by_path[relative]
        path = _safe_child(ROOT, relative)
        _verify_file_record(path, record, label="portable bound source")
        portable_sources.append(
            {
                "path": relative,
                "bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
                "kind": "bound_source",
            }
        )
    result = {
        "schema": PORTABLE_SOURCE_BINDINGS_SCHEMA,
        "path_semantics": PORTABLE_PATH_SEMANTICS,
        "contains_source_bytes": False,
        "raw_cmp_payloads_packaged": False,
        "project_score_archive_packaged": False,
        "local_inputs": local_inputs,
        "bound_sources": portable_sources,
    }
    _validate_portable_source_bindings(result)
    return result


def _validate_portable_source_bindings(value: Any) -> dict[str, Any]:
    required = {
        "schema",
        "path_semantics",
        "contains_source_bytes",
        "raw_cmp_payloads_packaged",
        "project_score_archive_packaged",
        "local_inputs",
        "bound_sources",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValidationError("portable source-binding fields are invalid")
    if value.get("schema") != PORTABLE_SOURCE_BINDINGS_SCHEMA:
        raise ValidationError("portable source-binding schema is invalid")
    if (
        value.get("path_semantics") != PORTABLE_PATH_SEMANTICS
        or value.get("contains_source_bytes") is not False
        or value.get("raw_cmp_payloads_packaged") is not False
        or value.get("project_score_archive_packaged") is not False
    ):
        raise ValidationError("portable source-binding boundary is invalid")
    local_inputs = value.get("local_inputs")
    if not isinstance(local_inputs, list) or len(local_inputs) != 3:
        raise ValidationError("portable local-input bindings are incomplete")
    expected_local = {
        "protocol_freeze": ("frozen_protocol", "author_review_required"),
        "raw_cache_manifest": ("local_raw_input_manifest", REDISTRIBUTION_STATUS),
        "project_score_archive": (
            "local_raw_input_archive",
            REDISTRIBUTION_STATUS,
        ),
    }
    seen_roles: set[str] = set()
    for index, record in enumerate(local_inputs):
        if not isinstance(record, dict) or set(record) != {
            "role",
            "path",
            "bytes",
            "sha256",
            "kind",
            "payload_packaged",
            "redistribution_status",
        }:
            raise ValidationError(f"portable local-input record {index} is invalid")
        role = record.get("role")
        if role not in expected_local or role in seen_roles:
            raise ValidationError("portable local-input role is unknown or duplicated")
        _validated_posix_relative_path(
            record.get("path"), label=f"portable {role} path"
        )
        size = record.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValidationError("portable local-input byte count is invalid")
        _validated_sha256(record.get("sha256"), label=f"portable {role} digest")
        expected_kind, expected_status = expected_local[role]
        if (
            record.get("kind") != expected_kind
            or record.get("payload_packaged") is not False
            or record.get("redistribution_status") != expected_status
        ):
            raise ValidationError("portable local-input boundary changed")
        seen_roles.add(role)
    if seen_roles != set(expected_local):
        raise ValidationError("portable local-input roles are incomplete")

    sources = value.get("bound_sources")
    if not isinstance(sources, list) or not sources:
        raise ValidationError("portable bound-source inventory is empty")
    seen_paths: set[str] = set()
    for index, record in enumerate(sources):
        if not isinstance(record, dict) or set(record) != {
            "path",
            "bytes",
            "sha256",
            "kind",
        }:
            raise ValidationError(f"portable bound-source record {index} is invalid")
        relative = _validated_posix_relative_path(
            record.get("path"), label=f"portable bound-source {index} path"
        )
        size = record.get("bytes")
        if (
            relative in seen_paths
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or record.get("kind") != "bound_source"
        ):
            raise ValidationError("portable bound-source record is invalid")
        _validated_sha256(
            record.get("sha256"), label=f"portable bound-source {index} digest"
        )
        seen_paths.add(relative)
    expected_paths = {
        path.resolve().relative_to(ROOT.resolve()).as_posix()
        for path in _source_paths()
    }
    if seen_paths != expected_paths:
        raise ValidationError(
            "portable bound-source inventory is incomplete or changed"
        )
    return value


def _verification_report(
    *,
    trusted_completion_sha256: str,
    redownload: bool,
    raw_records: int,
    run_environment: Mapping[str, Any],
    verification_environment: Mapping[str, Any],
    embedding_provenance: Mapping[str, Any],
    portable_source_bindings: Mapping[str, Any],
    artifacts: list[dict[str, Any]],
    gate_summary: Mapping[str, Any],
) -> dict[str, Any]:
    report = {
        "schema": VERIFICATION_SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verified": True,
        "scientific_result": True,
        "errors": [],
        "evaluation_unit": EVALUATION_UNIT,
        "cohort_relation": COHORT_RELATION,
        "claim_class": CLAIM_CLASS,
        "claim_boundary": CLAIM_BOUNDARY,
        "interpretation": "descriptive only",
        "verification_method": (
            "all bound local inputs rehashed; preprocessing, checkpoints, per-seed "
            "and ensemble predictions replayed; analysis recomputed; frozen adequacy "
            "gates checked without a performance-direction filter"
        ),
        "completion_schema": COMPLETION_SCHEMA,
        "trusted_completion_sha256": _validated_sha256(
            trusted_completion_sha256, label="trusted completion digest"
        ),
        "trust_scope": TRUST_SCOPE,
        "public_trust_anchor_verified": False,
        "public_versioned_archive_verified": False,
        "doi_verified": False,
        "verification_gates": {name: True for name in REPLAY_GATE_NAMES},
        "cohort_gates": {
            "primary_all_broad_nonoverlap": {
                "observed": gate_summary["primary_cells"],
                "minimum_required": MIN_CELLS_PER_COHORT,
                "passed": True,
            },
            "sensitivity_1103_nonoverlap_less_strict": {
                "observed": gate_summary["sensitivity_cells"],
                "minimum_required": MIN_CELLS_PER_COHORT,
                "passed": True,
            },
        },
        "matched_gene_gate": {
            "observed": gate_summary["matched_genes"],
            "minimum_required": MIN_MATCHED_GENES,
            "passed": True,
        },
        "finite_coverage_gates": {
            "minimum_finite_genes_per_cell": {
                "minimum_observed": gate_summary[
                    "minimum_observed_finite_genes_per_cell"
                ],
                "minimum_required": MIN_FINITE_GENES_PER_CELL,
                "passed": True,
            },
            "genes_with_minimum_finite_cell_profiles": {
                "minimum_observed": gate_summary[
                    "minimum_observed_genes_with_finite_cell_profiles"
                ],
                "minimum_required": MIN_GENES_WITH_FINITE_CELL_PROFILES,
                "minimum_cells_per_gene": MIN_FINITE_CELLS_PER_GENE,
                "passed": True,
            },
            "defined_gene_correlation_count": {
                "minimum_observed": gate_summary[
                    "minimum_observed_defined_gene_correlation_count"
                ],
                "minimum_required": MIN_DEFINED_GENE_CORRELATION_COUNT,
                "passed": True,
            },
            "defined_gene_correlation_fraction": {
                "minimum_observed": gate_summary[
                    "minimum_observed_defined_gene_correlation_fraction"
                ],
                "minimum_required": MIN_DEFINED_GENE_CORRELATION_FRACTION,
                "passed": True,
            },
            "metric_rows_checked": gate_summary["metric_rows_checked"],
            "passed": True,
        },
        "performance_direction_gate": {
            "positive_direction_required": False,
            "negative_null_and_partial_results_preserved": True,
        },
        "tolerances": {
            "prediction_replay": {
                "rtol": PREDICTION_RTOL,
                "atol": PREDICTION_ATOL,
            },
            "analysis_replay": {"rtol": ANALYSIS_RTOL, "atol": ANALYSIS_ATOL},
        },
        "remote_bytes_redownloaded": bool(redownload),
        "raw_records": raw_records,
        "run_environment": dict(run_environment),
        "verification_environment": dict(verification_environment),
        "environment_matches_run": run_environment == verification_environment,
        "embedding_provenance": dict(embedding_provenance),
        "portable_source_bindings": dict(portable_source_bindings),
        "artifact_path_base": "verified_run_directory",
        "artifacts": artifacts,
        "raw_cmp_payloads_packaged": False,
        "raw_cache_manifest_packaged": False,
        "project_score_archive_packaged": False,
        "redistribution_status": REDISTRIBUTION_STATUS,
    }
    _validate_verification_payload(report)
    return report


def _validate_verification_payload(
    value: Any, *, run_dir: Path | None = None
) -> dict[str, Any]:
    required = {
        "schema",
        "created_utc",
        "verified",
        "scientific_result",
        "errors",
        "evaluation_unit",
        "cohort_relation",
        "claim_class",
        "claim_boundary",
        "interpretation",
        "verification_method",
        "completion_schema",
        "trusted_completion_sha256",
        "trust_scope",
        "public_trust_anchor_verified",
        "public_versioned_archive_verified",
        "doi_verified",
        "verification_gates",
        "cohort_gates",
        "matched_gene_gate",
        "finite_coverage_gates",
        "performance_direction_gate",
        "tolerances",
        "remote_bytes_redownloaded",
        "raw_records",
        "run_environment",
        "verification_environment",
        "environment_matches_run",
        "embedding_provenance",
        "portable_source_bindings",
        "artifact_path_base",
        "artifacts",
        "raw_cmp_payloads_packaged",
        "raw_cache_manifest_packaged",
        "project_score_archive_packaged",
        "redistribution_status",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValidationError("verification artifact fields are invalid")
    if value.get("schema") != VERIFICATION_SCHEMA:
        raise ValidationError("verification artifact schema is invalid")
    if (
        value.get("verified") is not True
        or value.get("scientific_result") is not True
        or value.get("errors") != []
    ):
        raise ValidationError("verification artifact does not record a clean replay")
    if (
        value.get("evaluation_unit") != EVALUATION_UNIT
        or value.get("cohort_relation") != COHORT_RELATION
        or value.get("claim_class") != CLAIM_CLASS
        or value.get("claim_boundary") != CLAIM_BOUNDARY
        or value.get("interpretation") != "descriptive only"
    ):
        raise ValidationError("verification claim scope changed")
    if (
        not isinstance(value.get("created_utc"), str)
        or not value["created_utc"].endswith("Z")
        or not isinstance(value.get("verification_method"), str)
        or not value["verification_method"]
        or value.get("completion_schema") != COMPLETION_SCHEMA
    ):
        raise ValidationError("verification identity metadata is invalid")
    trusted_completion_sha256 = _validated_sha256(
        value.get("trusted_completion_sha256"),
        label="verification trusted completion digest",
    )
    if (
        value.get("trust_scope") != TRUST_SCOPE
        or value.get("public_trust_anchor_verified") is not False
        or value.get("public_versioned_archive_verified") is not False
        or value.get("doi_verified") is not False
    ):
        raise ValidationError("verification artifact overstates its trust scope")
    gates = value.get("verification_gates")
    if (
        not isinstance(gates, dict)
        or set(gates) != set(REPLAY_GATE_NAMES)
        or any(gates[name] is not True for name in REPLAY_GATE_NAMES)
    ):
        raise ValidationError("verification replay gates are incomplete")

    cohort_gates = value.get("cohort_gates")
    expected_cohorts = {
        "primary_all_broad_nonoverlap",
        "sensitivity_1103_nonoverlap_less_strict",
    }
    if not isinstance(cohort_gates, dict) or set(cohort_gates) != expected_cohorts:
        raise ValidationError("verification cohort gates are incomplete")
    for record in cohort_gates.values():
        if not isinstance(record, dict) or set(record) != {
            "observed",
            "minimum_required",
            "passed",
        }:
            raise ValidationError("verification cohort gate fields are invalid")
        observed = record.get("observed")
        if (
            isinstance(observed, bool)
            or not isinstance(observed, int)
            or observed < MIN_CELLS_PER_COHORT
            or record.get("minimum_required") != MIN_CELLS_PER_COHORT
            or record.get("passed") is not True
        ):
            raise ValidationError("verification cohort gate failed")
    matched_gate = value.get("matched_gene_gate")
    if not isinstance(matched_gate, dict) or set(matched_gate) != {
        "observed",
        "minimum_required",
        "passed",
    }:
        raise ValidationError("verification matched-gene gate fields are invalid")
    matched = matched_gate.get("observed")
    if (
        isinstance(matched, bool)
        or not isinstance(matched, int)
        or matched < MIN_MATCHED_GENES
        or matched_gate.get("minimum_required") != MIN_MATCHED_GENES
        or matched_gate.get("passed") is not True
    ):
        raise ValidationError("verification matched-gene gate failed")

    coverage = value.get("finite_coverage_gates")
    if not isinstance(coverage, dict) or set(coverage) != {
        "minimum_finite_genes_per_cell",
        "genes_with_minimum_finite_cell_profiles",
        "defined_gene_correlation_count",
        "defined_gene_correlation_fraction",
        "metric_rows_checked",
        "passed",
    }:
        raise ValidationError("verification finite-coverage gates are invalid")
    finite_specs = {
        "minimum_finite_genes_per_cell": MIN_FINITE_GENES_PER_CELL,
        "genes_with_minimum_finite_cell_profiles": (
            MIN_GENES_WITH_FINITE_CELL_PROFILES
        ),
        "defined_gene_correlation_count": MIN_DEFINED_GENE_CORRELATION_COUNT,
        "defined_gene_correlation_fraction": (MIN_DEFINED_GENE_CORRELATION_FRACTION),
    }
    for name, required_minimum in finite_specs.items():
        record = coverage.get(name)
        allowed = {"minimum_observed", "minimum_required", "passed"}
        if name == "genes_with_minimum_finite_cell_profiles":
            allowed.add("minimum_cells_per_gene")
        if not isinstance(record, dict) or set(record) != allowed:
            raise ValidationError(f"verification coverage gate {name} is invalid")
        observed = record.get("minimum_observed")
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
            or float(observed) < required_minimum
            or record.get("minimum_required") != required_minimum
            or record.get("passed") is not True
        ):
            raise ValidationError(f"verification coverage gate {name} failed")
        if (
            name == "genes_with_minimum_finite_cell_profiles"
            and record.get("minimum_cells_per_gene") != MIN_FINITE_CELLS_PER_GENE
        ):
            raise ValidationError("verification finite-cells-per-gene gate changed")
    if coverage.get("metric_rows_checked") != 8 or coverage.get("passed") is not True:
        raise ValidationError("verification did not check every analysis metric row")
    if value.get("performance_direction_gate") != {
        "positive_direction_required": False,
        "negative_null_and_partial_results_preserved": True,
    }:
        raise ValidationError("verification incorrectly gates performance direction")
    if value.get("tolerances") != {
        "prediction_replay": {"rtol": PREDICTION_RTOL, "atol": PREDICTION_ATOL},
        "analysis_replay": {"rtol": ANALYSIS_RTOL, "atol": ANALYSIS_ATOL},
    }:
        raise ValidationError("verification replay tolerances changed")
    if not isinstance(value.get("remote_bytes_redownloaded"), bool):
        raise ValidationError("verification redownload flag is invalid")
    raw_records = value.get("raw_records")
    if (
        isinstance(raw_records, bool)
        or not isinstance(raw_records, int)
        or raw_records < 1
    ):
        raise ValidationError("verification raw-record count is invalid")
    run_environment = _validate_environment_manifest(value.get("run_environment"))
    verification_environment = _validate_environment_manifest(
        value.get("verification_environment")
    )
    if value.get("environment_matches_run") is not (
        run_environment == verification_environment
    ):
        raise ValidationError("verification environment comparison is invalid")
    embedding = value.get("embedding_provenance")
    if (
        not isinstance(embedding, dict)
        or embedding.get("schema") != "DepPLM-verified-embedding-provenance-v1"
        or embedding.get("verification_status") != "verified"
    ):
        raise ValidationError("verification embedding provenance is invalid")
    _validate_portable_source_bindings(value.get("portable_source_bindings"))
    if value.get("artifact_path_base") != "verified_run_directory":
        raise ValidationError("verification artifact path base changed")
    artifacts = _validate_portable_artifact_inventory(
        value.get("artifacts"), run_dir=run_dir
    )
    completion_records = [
        record for record in artifacts if record["path"] == "completion.json"
    ]
    if (
        len(completion_records) != 1
        or completion_records[0]["sha256"] != trusted_completion_sha256
    ):
        raise ValidationError(
            "trusted completion digest differs from the portable artifact inventory"
        )
    if (
        value.get("raw_cmp_payloads_packaged") is not False
        or value.get("raw_cache_manifest_packaged") is not False
        or value.get("project_score_archive_packaged") is not False
        or value.get("redistribution_status") != REDISTRIBUTION_STATUS
    ):
        raise ValidationError(
            "verification artifact overstates redistribution clearance"
        )
    return value


def validate_verification_artifact(
    path: Path, *, run_dir: Path | None = None
) -> dict[str, Any]:
    """Validate a persisted report; optionally rehash every portable run artifact."""
    payload = _load_json(path)
    return _validate_verification_payload(payload, run_dir=run_dir)


def _write_verification_artifact(
    path: Path, payload: Mapping[str, Any], *, run_dir: Path
) -> Path:
    if path.is_symlink():
        raise ValidationError("verification output must not be a symlink")
    output = path.resolve()
    resolved_run = run_dir.resolve()
    try:
        output.relative_to(resolved_run)
    except ValueError:
        pass
    else:
        raise ValidationError(
            "verification output must stay outside the exact run directory"
        )
    _validate_verification_payload(dict(payload), run_dir=resolved_run)
    _atomic_new_json(output, payload)
    validate_verification_artifact(output, run_dir=resolved_run)
    return output


def verify_run(
    *,
    run_dir: Path,
    freeze_path: Path,
    cache_dir: Path,
    project_score: Path = PROJECT_SCORE,
    redownload: bool = False,
    trusted_completion_sha256: str,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Independently rebuild features, replay checkpoints, and recompute metrics."""
    import torch

    run_dir = run_dir.resolve()
    validate_exact_run_inventory(run_dir)
    _verify_completion(run_dir, trusted_completion_sha256=trusted_completion_sha256)
    freeze = validate_freeze(freeze_path)
    raw_manifest, by_sidm = _load_raw_manifest(
        freeze_path, cache_dir, redownload=redownload
    )
    bindings = _load_json(run_dir / "source-bindings.json")
    if (
        not isinstance(bindings, dict)
        or bindings.get("schema") != SOURCE_BINDINGS_SCHEMA
    ):
        raise ValidationError("source bindings schema is invalid")
    run_environment = _validate_environment_manifest(bindings.get("environment"))
    verification_environment = _environment_manifest()
    embedding_provenance = _validated_embedding_provenance()
    if bindings.get("embedding_provenance") != embedding_provenance:
        raise ValidationError("verified ESM embedding provenance differs from the run")
    _verify_source_snapshot(bindings.get("sources", []))
    _verify_file_record(freeze_path, bindings["freeze"], label="bound freeze")
    _verify_file_record(
        cache_dir / "raw-manifest.json",
        bindings["raw_manifest"],
        label="bound raw manifest",
    )
    _verify_file_record(
        project_score, bindings["project_score"], label="bound Project Score archive"
    )
    existing = _import_existing_validation()
    existing.validate_external_archive(project_score)
    model_lock = _verify_model_lock(run_dir, freeze_path, freeze)

    corrected, combined, feature_audit = _build_combined_base(
        freeze, cache_dir, by_sidm
    )
    training_gene_mean = _training_gene_mean(combined)
    if feature_audit != model_lock["feature_audit"]:
        raise ValidationError(
            "rederived external feature audit differs from model lock"
        )
    training_idx = np.arange(EXPECTED_TRAINING_CELLS, dtype=np.int64)
    external_idx = np.arange(
        EXPECTED_TRAINING_CELLS, len(combined.cell_ids), dtype=np.int64
    )
    predictions: dict[str, np.ndarray] = {}
    for seed in CANONICAL_SEEDS:
        spec = freeze["training_specs"][str(seed)]
        preprocessing_seed = corrected.derive_seed(seed, "sanger-full-preprocess")
        fold_data = corrected.fit_fold_data(
            combined,
            training_idx,
            seed=preprocessing_seed,
            n_expr_pc=128,
            top_mut=5000,
        )
        _compare_preprocessing(run_dir / f"preprocessing-seed-{seed}.npz", fold_data)
        model = corrected.build_model(spec["config"], combined, fold_data)
        try:
            state = torch.load(
                run_dir / f"checkpoint-seed-{seed}.pt",
                map_location="cpu",
                weights_only=True,
            )
        except Exception as exc:
            raise ValidationError(
                f"checkpoint cannot be loaded safely for seed {seed}"
            ) from exc
        model.load_state_dict(state, strict=True)
        prediction = _predict_rows(
            corrected, model, spec["config"], fold_data, external_idx
        )
        arrays = _prediction_arrays(
            prediction, cells=freeze["cells"], genes=combined.genes
        )
        compare_replayed_predictions(run_dir / f"predictions-seed-{seed}.npz", arrays)
        predictions[f"seed-{seed}"] = prediction
    ensemble = np.mean(
        np.stack(list(predictions.values())), axis=0, dtype=np.float64
    ).astype(np.float32)
    compare_replayed_predictions(
        run_dir / "predictions-ensemble.npz",
        _prediction_arrays(ensemble, cells=freeze["cells"], genes=combined.genes),
    )
    predictions["ensemble_mean"] = ensemble
    expected_analysis = _analysis_from_predictions(
        freeze=freeze,
        predictions=predictions,
        training_gene_mean=training_gene_mean,
        genes=combined.genes,
        project_score=project_score,
    )
    observed_analysis = _load_json(run_dir / "analysis.json")
    _compare_analysis(observed_analysis, expected_analysis)
    gate_summary = _analysis_gate_summary(expected_analysis, freeze)
    portable_bindings = _portable_source_bindings(
        bindings=bindings,
        freeze_path=freeze_path,
        cache_dir=cache_dir,
        project_score=project_score,
    )
    artifacts = _portable_artifact_inventory(run_dir)
    report = _verification_report(
        trusted_completion_sha256=trusted_completion_sha256,
        redownload=redownload,
        raw_records=len(raw_manifest["records"]),
        run_environment=run_environment,
        verification_environment=verification_environment,
        embedding_provenance=embedding_provenance,
        portable_source_bindings=portable_bindings,
        artifacts=artifacts,
        gate_summary=gate_summary,
    )
    if output_path is not None:
        _write_verification_artifact(output_path, report, run_dir=run_dir)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    feasibility = subparsers.add_parser(
        "feasibility",
        help="write a diagnostic cohort audit without numeric outcomes",
    )
    feasibility.add_argument("--output", type=Path, required=True)
    feasibility.add_argument("--model-files-json", type=Path, required=True)
    feasibility.add_argument("--project-score", type=Path, default=PROJECT_SCORE)
    feasibility.add_argument("--model-csv", type=Path, default=MODEL_CSV)
    feasibility.add_argument("--broad-labels", type=Path, default=BROAD_LABEL_CSV)
    feasibility.add_argument("--training-cells", type=Path, default=TRAINING_CELLS)

    freeze = subparsers.add_parser(
        "freeze", help="freeze cohort/config before outcomes"
    )
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--campaign-report", type=Path, required=True)
    freeze.add_argument("--openapi-json", type=Path, required=True)
    freeze.add_argument("--model-files-json", type=Path, required=True)
    freeze.add_argument("--project-score", type=Path, default=PROJECT_SCORE)

    download = subparsers.add_parser(
        "download", help="download frozen CMP feature inputs"
    )
    download.add_argument("--freeze", type=Path, required=True)
    download.add_argument("--cache-dir", type=Path, required=True)

    run = subparsers.add_parser("run", help="train, lock, then evaluate")
    run.add_argument("--freeze", type=Path, required=True)
    run.add_argument("--cache-dir", type=Path, required=True)
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run.add_argument("--project-score", type=Path, default=PROJECT_SCORE)
    run.add_argument("--device", choices=("cpu", "cuda"), default="cuda")

    verify = subparsers.add_parser(
        "verify", help="replay checkpoint inference and metrics"
    )
    verify.add_argument("--run-dir", type=Path, required=True)
    verify.add_argument("--freeze", type=Path, required=True)
    verify.add_argument("--cache-dir", type=Path, required=True)
    verify.add_argument("--project-score", type=Path, default=PROJECT_SCORE)
    verify.add_argument("--redownload", action="store_true")
    verify.add_argument(
        "--trusted-completion-sha256",
        required=True,
        help=(
            "caller-supplied SHA-256 stored outside the run directory; supplying it "
            "does not prove that a DOI or public trust anchor exists"
        ),
    )
    verify.add_argument(
        "--output",
        type=Path,
        help=(
            "write-once verification JSON outside the exact run directory; existing "
            "files are never overwritten"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "feasibility":
            path = create_feasibility_audit(
                output=args.output,
                model_files_json=args.model_files_json,
                project_score=args.project_score,
                model_csv=args.model_csv,
                broad_labels=args.broad_labels,
                training_cells=args.training_cells,
            )
            print(path.resolve())
        elif args.command == "freeze":
            path = create_freeze(
                output=args.output,
                campaign_report=args.campaign_report,
                openapi_json=args.openapi_json,
                model_files_json=args.model_files_json,
                project_score=args.project_score,
            )
            print(path)
        elif args.command == "download":
            print(download_frozen_raw(args.freeze, args.cache_dir))
        elif args.command == "run":
            print(
                run_validation(
                    freeze_path=args.freeze,
                    cache_dir=args.cache_dir,
                    output_root=args.output_root,
                    project_score=args.project_score,
                    device=args.device,
                )
            )
        elif args.command == "verify":
            report = verify_run(
                run_dir=args.run_dir,
                freeze_path=args.freeze,
                cache_dir=args.cache_dir,
                project_score=args.project_score,
                redownload=args.redownload,
                trusted_completion_sha256=args.trusted_completion_sha256,
                output_path=args.output,
            )
            if args.output is None:
                print(json.dumps(report, indent=2, sort_keys=True))
            else:
                print(args.output.resolve())
        else:  # pragma: no cover - argparse enforces the subcommand
            raise ValidationError(f"unknown command: {args.command}")
    except ValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
