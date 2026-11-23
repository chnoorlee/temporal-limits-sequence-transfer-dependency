"""Hash-bound Open Targets 26.09 clinical validation for DepPLM.

The module deliberately separates two operations:

``freeze``
    Validate the exact local Open Targets 26.09 Parquet bytes and derive a
    compact, versioned TSV snapshot.

``run`` / ``verify``
    Consume the frozen TSV snapshot and the completed, independently aggregated
    99-run corrected campaign.  These commands fail closed unless the snapshot
    is rederived from the fixed raw Parquet bytes, every campaign artifact still
    hashes to the campaign report, every P1 held-out prediction is present
    exactly once, and all covariates can be matched exactly.

No association score is used as a clinical endpoint.  The primary endpoint is
a target-drug row that reached phase 2 or later and whose complete, mapped
disease list is cancer scoped.  The row-level maximum is not interpreted as a
current-development status.  A WITHDRAWAL record is retained only as a
separately labelled sensitivity outcome under the same cancer-only rule.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.machinery
import io
import json
import math
import os
import platform
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

import numpy as np

import clinical_snapshot_builder_v4 as snapshot_builder_v4

_SRC_SHA256 = "3e1a1c2da848fd9daf2264d29dbf0b7f613d44ab693b8e710f76f577b0e54d29"

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_ROOT = ROOT / "data" / "external" / "opentargets_26_09" / "raw"
DEFAULT_SNAPSHOT_DIR = ROOT / "data" / "external" / "opentargets_26_09" / "frozen_v3"
DEFAULT_SNAPSHOT_BUILD_DIR = (
    ROOT / "data" / "external" / "opentargets_26_09" / "frozen_v4_final"
)
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "clinical_validation_v2"
DEFAULT_PPI = ROOT / "data" / "processed" / "ppi_graph.npz"
DEFAULT_FAMILY = (
    ROOT / "results" / "family_holdout_v1" / "identity_30" / "family_units.tsv"
)
DEFAULT_FAMILY_RELEASE_MANIFEST = (
    ROOT / "results" / "family_holdout_v1" / "release_manifest.json"
)
DEFAULT_FASTA = ROOT / "data" / "processed" / "gene_seqs.fa"
DEFAULT_CELLS = ROOT / "data" / "processed" / "cell_lines.txt"
DEFAULT_GENES = ROOT / "data" / "processed" / "genes.txt"

RELEASE = "26.09"
CANCER_ROOT = "MONDO_0045024"
PHASE2_PLUS_STAGES = (
    "PHASE_2",
    "PHASE_2_3",
    "PHASE_3",
    "PHASE_4",
    "PREAPPROVAL",
    "APPROVAL",
)
WITHDRAWAL_STAGE = "WITHDRAWAL"
STAGE_ORDER = {stage: index for index, stage in enumerate(PHASE2_PLUS_STAGES, start=1)}

CANONICAL_PROTOCOLS = ("p1", "transfer-forward", "transfer-reverse")
CANONICAL_CONDITIONS = (
    "A1-ESM",
    "A2-GNN",
    "A3-comut",
    "A4-MT",
    "A5-XATTN",
    "C-DEEPDEP",
    "C-GCN",
    "C-GCN-comut",
    "C-ONEHOT",
    "C-PLM-MLP",
    "DepPLM",
)
CANONICAL_SEEDS = (42, 43, 44)
EXPECTED_RUNS = 99
P1_FOLDS = tuple(f"fold-{index:02d}" for index in range(5))
PREDICTION_ARRAYS = (
    "prediction",
    "train_gene_mean",
    "outer_test_indices",
    "source_cell_indices",
    "source_gene_indices",
    "cell_ids",
    "gene_ids",
)
PPI_ARRAYS = ("edge_index", "edge_weight", "nodes")
COHORT_MANIFEST_FIELDS = {
    "cell_ids_sha256",
    "expression_gene_names_sha256",
    "genes_sha256",
    "n_cells",
    "n_expression_features_raw",
    "n_genes",
    "n_string_edges",
    "source_cell_indices",
    "source_gene_indices",
}
CAMPAIGN_SCHEMA = "DepPLM-corrected-v2.2-campaign"
CAMPAIGN_FIELDS = {
    "aggregates",
    "aggregation_provenance",
    "created_utc",
    "paired_descriptive_comparisons",
    "prespecified_matrix",
    "protocol_validation_passed",
    "schema",
    "scientific_readiness_gates",
    "scientific_result",
    "scientific_result_reason",
    "source_runs",
    "statistical_policy",
    "validation",
}
CAMPAIGN_READINESS_GATES = {
    "outer_test_metrics_recomputed_from_bound_predictions": True,
    "verified_embedding_generation_provenance": True,
}
CAMPAIGN_MATRIX = {
    "canonical_arguments": {
        "batch_size": 64,
        "folds": 5,
        "gene_subsample": 6000,
        "inner_max_epochs": 20,
        "inner_val_fraction": 0.2,
        "n_expr_pc": 128,
        "patience": 5,
        "top_mut": 5000,
    },
    "conditions": list(CANONICAL_CONDITIONS),
    "expected_runs": EXPECTED_RUNS,
    "p1_outer_folds_per_run": 5,
    "protocols": list(CANONICAL_PROTOCOLS),
    "seeds": list(CANONICAL_SEEDS),
    "transfer_outer_folds_per_run": 1,
    "tuning_grid": [
        {"hidden": hidden, "lamb_sel": 0.1, "lr": learning_rate, "wd": 1e-5}
        for hidden in (128, 256)
        for learning_rate in (0.001, 0.0003)
    ],
}
CAMPAIGN_VALIDATION_FIELDS = {
    "code_hash_conflicts",
    "cohort_hashes",
    "completion_bearing_directories",
    "duplicate_matrix_cells",
    "excluded_noncomplete_runs",
    "input_hash_conflicts",
    "issues",
    "missing_matrix_cells",
    "runtime_fingerprints",
    "unique_matrix_cells",
    "validated_runs",
}
CAMPAIGN_SOURCE_RUN_FIELDS = {
    "key",
    "path",
    "completion_sha256",
    "summary_sha256",
    "run_manifest_sha256",
    "cohort_manifest_sha256",
    "runtime_fingerprint_sha256",
    "fold_artifacts",
}
CAMPAIGN_PPI_INPUT_ROLE = "data/processed/ppi_graph.npz"
CAMPAIGN_INPUT_RECORD_FIELDS = {"path", "bytes", "mtime_ns", "sha256"}
CAMPAIGN_PPI_BINDING_FIELDS = {"role", "bytes", "sha256"}
CAMPAIGN_COMMON_INPUT_ROLES = (
    "data/processed/gene_effect.npy",
    "data/processed/esm2_t6_8M_residue_chunked.npy",
    "data/processed/esm2_t6_8M_residue_chunked.l2.npy",
    "data/processed/esm2_t6_8M_residue_chunked.names.txt",
    "data/processed/esm2_t6_8M_residue_chunked.manifest.json",
    "data/processed/gene_seqs.fa",
    CAMPAIGN_PPI_INPUT_ROLE,
    "data/processed/genes.txt",
    "data/processed/cell_lines.txt",
    "data/processed/lineages.txt",
    "data/depmap/OmicsExpressionProteinCodingGenesTPMLogp1.csv",
    "data/depmap/OmicsSomaticMutations.csv",
)
CAMPAIGN_TRANSFER_INPUT_ROLE = "data/processed/splits.json"
CAMPAIGN_CODE_ROLES = (
    "code/run_corrected_experiments.py",
    "code/corrected_protocol.py",
    "code/models.py",
    "code/03_embed_proteins.py",
    "code/embedding_provenance.py",
)
CAMPAIGN_AGGREGATOR_ROLE = "code/aggregate_corrected_campaign.py"
CAMPAIGN_EXECUTED_MODULE_ROLES = {
    "aggregate_corrected_campaign": CAMPAIGN_AGGREGATOR_ROLE,
    "corrected_protocol": "code/corrected_protocol.py",
    "embedding_provenance": "code/embedding_provenance.py",
    "run_corrected_experiments": "code/run_corrected_experiments.py",
    "models": "code/models.py",
}

PRIMARY_TOP_N = 500
SENSITIVITY_TOP_N = (200, 1_000)
MIN_PERMUTATIONS = 10_000
DEFAULT_PERMUTATION_SEED = 20261021

SNAPSHOT_SCHEMA = "DepPLM-OpenTargets-26.09-clinical-snapshot-v3"
FORMAL_SNAPSHOT_SCHEMA = snapshot_builder_v4.SNAPSHOT_SCHEMA
RUN_SCHEMA = "DepPLM-OpenTargets-clinical-validation-v3"
VERIFICATION_SCHEMA = "DepPLM-OpenTargets-clinical-verification-v3"
COMPLETION_SCHEMA = "DepPLM-OpenTargets-clinical-completion-v3"
CLAIM_BOUNDARY = (
    "Matched association with frozen historical Open Targets annotations is external "
    "annotation concordance, not a population effect, prospective clinical validation, "
    "causal evidence, or DepPLM-specific validation."
)
FORMAL_VERIFICATION_ARTIFACTS = (
    "run-manifest.json",
    "rankings.tsv",
    "tests.tsv",
    "results.json",
)
FORMAL_RUN_FILES = {
    *FORMAL_VERIFICATION_ARTIFACTS,
    "verification.json",
    "completion.json",
}
FORMAL_RUN_MANIFEST_FIELDS = {
    "schema",
    "created_utc",
    "inputs",
    "snapshot_dir",
    "analysis_spec",
    "runtime",
}
FORMAL_RUNTIME_FIELDS = {
    "python",
    "platform",
    "numpy",
    "pyarrow_required_for_snapshot_rederivation",
}
FORMAL_INPUT_FIELDS = {
    "campaign_report",
    "snapshot_manifest",
    "ppi",
    "family",
    "family_provenance",
    "fasta",
    "cells",
    "genes",
    "analysis_source",
}
FORMAL_FILE_INPUT_FIELDS = {
    "snapshot_manifest",
    "ppi",
    "family",
    "fasta",
    "cells",
    "genes",
    "analysis_source",
}
FORMAL_CAMPAIGN_RECORD_FIELDS = {
    "path",
    "bytes",
    "sha256",
    "source_runs_rehashed",
    "source_artifacts_rehashed",
}
EXPECTED_CAMPAIGN_SOURCE_ARTIFACTS = len(CANONICAL_CONDITIONS) * len(
    CANONICAL_SEEDS
) * (4 + 15) + 2 * len(CANONICAL_CONDITIONS) * len(CANONICAL_SEEDS) * (4 + 3)
FORMAL_FAMILY_PROVENANCE_FIELDS = {
    "verification_status",
    "schema",
    "tool",
    "identity_percent",
    "minimum_bidirectional_coverage",
    "component_definition",
    "family_release_manifest",
    "family_source_manifest",
    "family_split_summary",
    "family_homology_hits",
    "semantic_replay",
}
FORMAL_VERIFICATION_FIELDS = {
    "schema",
    "verified_utc",
    "scientific_result",
    "verification_method",
    "artifacts",
    "analysis_sha256",
}
FORMAL_COMPLETION_FIELDS = {
    "schema",
    "scientific_result",
    "verification",
    "results",
}
FORMAL_VERIFICATION_METHOD = (
    "all bound inputs rehashed; 99-run/P1 exact-once gates rerun; deterministic "
    "analysis artifacts regenerated and byte-compared"
)

FAMILY_RELEASE_SCHEMA = "depplm-family-holdout-v1"
FAMILY_UNITS_RELATIVE = "identity_30/family_units.tsv"
FAMILY_SUMMARY_RELATIVE = "identity_30/split_summary.json"
FAMILY_SOURCE_RELATIVE = "source_manifest.json"
FAMILY_HITS_RELATIVE = "homology_hits.tsv.gz"
FAMILY_IDENTITY_PERCENT = 30.0
FAMILY_MIN_BIDIRECTIONAL_COVERAGE = 0.8
FAMILY_COMPONENT_DEFINITION = "connected_components_of_qualifying_mmseqs_pairs"
FAMILY_RELEASE_MANIFEST_SHA256 = (
    "1067b40db70998e3ffcd414c2912dd44341fccca9b2f0d43c2ab85d4dcde3c64"
)
FAMILY_SOURCE_MANIFEST_SHA256 = (
    "0098b31c76ee1b0619dd71257ab893013e0d725eae6839c67b3db5657b577c51"
)
FAMILY_UNITS_SHA256 = "1f28929d88581fbfa5af9d39e2bde0db222ec0db9d84cbab5cb52b497c88b916"
FAMILY_SPLIT_SUMMARY_SHA256 = (
    "95ec5257ae4afc9c2613fbc76348b7300bab7eb37d47954963e4737c87c4b466"
)
FAMILY_HOMOLOGY_HITS_SHA256 = (
    "60b6a81866513ab7ccfbd73b29841eec478b19532e48dd419e0b8a9346694b8e"
)
FAMILY_MMSEQS2_RELEASE = "18-8cc5c"
FAMILY_MMSEQS2_COMMIT = "8cc5ce367b5638c4306c2d7cfc652dd099a4643f"
FAMILY_EXPECTED_GENE_COUNT = 17_768
FAMILY_EXPECTED_QUALIFYING_EDGE_COUNT = 83_009
FAMILY_EXPECTED_COMPONENT_COUNT = 11_577
FAMILY_UNITS_FIELDS = (
    "gene",
    "accession",
    "sequence_sha256",
    "component_id",
    "component_size",
)
FAMILY_HITS_FIELDS = (
    "gene_a",
    "gene_b",
    "pident_percent",
    "coverage_a",
    "coverage_b",
    "alignment_length",
    "evalue",
    "bits",
)
FAMILY_PROVENANCE_RECORD_FIELDS = (
    "family_release_manifest",
    "family_source_manifest",
    "family_split_summary",
    "family_homology_hits",
)

OUTCOME_FIELDS = (
    "approved_symbol",
    "target_id",
    "clinical_oncology_phase2_plus",
    "oncology_phase2_plus_drug_count",
    "oncology_phase2_plus_stages",
    "maximum_oncology_phase2_plus_stage",
    "clinical_oncology_withdrawal",
    "oncology_withdrawn_drug_count",
    "tractable_any",
    "tractability_true_record_count",
    "tractability_true_modalities",
)
DISEASE_FIELDS = ("disease_id", "name", "scope_relation")
AMBIGUOUS_FIELDS = ("approved_symbol", "target_ids", "reason")


class ValidationError(RuntimeError):
    """Raised when a provenance, identity, or analysis gate fails."""


def _source_binding_sha256(payload: bytes) -> str:
    """Hash exact source bytes after blanking only their embedded digest value."""
    marker = b"_SRC_" + b'SHA256 = "'
    start = payload.find(marker)
    if start < 0 or payload.find(marker, start + 1) >= 0:
        raise ValidationError("analysis source binding marker is noncanonical")
    value_start = start + len(marker)
    value_end = value_start + 64
    value = payload[value_start:value_end]
    if (
        len(value) != 64
        or payload[value_end : value_end + 1] != b'"'
        or any(byte not in b"0123456789abcdef" for byte in value)
    ):
        raise ValidationError("analysis source binding value is noncanonical")
    if value.decode("ascii") != _SRC_SHA256:
        raise ValidationError(
            "analysis source binding value differs from the executing module"
        )
    normalized = payload[:value_start] + (b"0" * 64) + payload[value_end:]
    return hashlib.sha256(normalized).hexdigest()


def _capture_module_source(path: Path) -> tuple[bytes, dict[str, Any]]:
    """Bind bytes cryptographically identified by the executing module payload."""
    lexical = Path(os.path.abspath(os.fspath(path)))
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lexical, flags)
    try:
        opened = os.fstat(descriptor)
        named = os.stat(lexical, follow_symlinks=False)
        if not stat.S_ISREG(opened.st_mode) or not os.path.samestat(opened, named):
            raise ValidationError("analysis source is not a stable regular file")
        digest = hashlib.sha256()
        payload = bytearray()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            payload.extend(chunk)
        after = os.fstat(descriptor)
        named_after = os.stat(lexical, follow_symlinks=False)
        if (
            not os.path.samestat(opened, after)
            or not os.path.samestat(opened, named_after)
            or after.st_size != len(payload)
        ):
            raise ValidationError("analysis source changed while being bound")
    finally:
        os.close(descriptor)
    bound_payload = bytes(payload)
    if _source_binding_sha256(bound_payload) != _SRC_SHA256:
        raise ValidationError(
            "analysis source payload does not match the executing module binding"
        )
    return bound_payload, {
        "bytes": len(payload),
        "sha256": digest.hexdigest(),
    }


_LOADED_SOURCE_PATH = Path(__file__).resolve()
_LOADED_SOURCE_PAYLOAD, _LOADED_SOURCE_FINGERPRINT = _capture_module_source(
    _LOADED_SOURCE_PATH
)


def _require_loaded_source_unchanged() -> None:
    """Reject a stale loaded module if its source name now binds other bytes."""
    try:
        _payload, observed = _capture_module_source(_LOADED_SOURCE_PATH)
    except ValidationError as exc:
        raise ValidationError(
            "analysis source changed after this module was loaded"
        ) from exc
    if observed != _LOADED_SOURCE_FINGERPRINT:
        raise ValidationError("analysis source changed after this module was loaded")


def _loaded_analysis_source_record() -> dict[str, Any]:
    return {"path": str(_LOADED_SOURCE_PATH), **_LOADED_SOURCE_FINGERPRINT}


@dataclass(frozen=True)
class RawFileSpec:
    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class FamilyUnit:
    gene: str
    accession: str
    sequence_sha256: str
    component_id: str
    component_size: int


@dataclass(frozen=True)
class _CapturedFile:
    """One exact-byte, verifier-owned capture of a source file."""

    source_path: Path
    path: Path
    bytes: int
    sha256: str

    def record(self, logical_path: str | Path | None = None) -> dict[str, Any]:
        label = self.source_path if logical_path is None else logical_path
        return {
            "path": str(label),
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


class _OwnedWindowsHandle:
    """Own a Win32 HANDLE from before a native out-parameter is populated."""

    __slots__ = ("_slot",)

    def __init__(self, raw_handle: int | None = None) -> None:
        from ctypes import wintypes

        self._slot = wintypes.HANDLE(raw_handle)

    @property
    def value(self) -> int:
        value = self._slot.value
        if value is None:
            raise RuntimeError("Windows handle is not owned")
        return int(value)

    @property
    def pointer(self) -> Any:
        import ctypes

        return ctypes.byref(self._slot)

    def release(self) -> int:
        value = self.value
        self._slot.value = None
        return value

    def close(self) -> int | None:
        value = self._slot.value
        if value is None:
            return None
        self._slot.value = None
        return _windows_close_handle(int(value))

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            pass


@dataclass
class _PinnedDirectory:
    """An opened directory identity held across publication or cleanup."""

    path: Path
    identity: tuple[int, int]
    windows_handle: _OwnedWindowsHandle | None = None
    posix_fd: int | None = None


@dataclass(frozen=True)
class _PendingFormalPublication:
    """An exclusively reserved formal-run directory awaiting its commit marker."""

    destination: Path
    staging: Path
    parent_identity: tuple[int, int]
    staging_identity: tuple[int, int]
    parent_pin: _PinnedDirectory
    staging_pin: _PinnedDirectory
    created_file_identities: dict[str, tuple[int, int]] = field(default_factory=dict)


@dataclass(frozen=True)
class _ReplayProcessOutcome:
    """Bounded output collected from one isolated campaign replay."""

    returncode: int
    stdout: bytes
    stderr: bytes
    result: bytes | None
    result_identity: tuple[int, int] | None = None


RAW_FILES = (
    RawFileSpec(
        "clinical_target/00000000.parquet",
        2_815_006,
        "33feae900b97a0935c3e30a5c745d3074a2ca46c5e50456b334482c0da113735",
    ),
    RawFileSpec(
        "disease/00000000.parquet",
        6_346_458,
        "a89b8b8b89bad168016e6ff8302d2bcc8d9b7f9cbb35c459ef847c36a4f17aed",
    ),
    RawFileSpec(
        "target/part-00000-ddfc1f8c-60a1-40fe-885a-2c1dd3f9a9c1-c000.zstd.parquet",
        7_165_956,
        "80186f1c9919d5d1351f009253480fad93b3bb3e2de506d8948ed2126ac48cab",
    ),
    RawFileSpec(
        "target/part-00001-ddfc1f8c-60a1-40fe-885a-2c1dd3f9a9c1-c000.zstd.parquet",
        7_196_171,
        "a777ec38d5fbf6e6475d85158141c0a01f2c32cadc7f8769256eaab6e034e702",
    ),
    RawFileSpec(
        "target/part-00002-ddfc1f8c-60a1-40fe-885a-2c1dd3f9a9c1-c000.zstd.parquet",
        7_270_990,
        "3621b2510eb3f9da9b533f40559b4f4de390fffc57935726a2b0eebcc40aa028",
    ),
    RawFileSpec(
        "target_tractability/part-00000-e2201205-916c-4b28-971d-6511ae9ea691-c000.zstd.parquet",
        55_708,
        "c81b9d1c61d3fa1e23e67a5b91c082e7d20829aba578ffe95adc9ea1ef2332fe",
    ),
    RawFileSpec(
        "target_tractability/part-00001-e2201205-916c-4b28-971d-6511ae9ea691-c000.zstd.parquet",
        72_189,
        "ebc49a49ae182dee29be43664dfc61c21897bf3d776d9df18f7e0e73bc0990ea",
    ),
)


def _sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _strict_json_bytes(payload: bytes, *, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValidationError(f"{label} contains duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ValidationError(f"{label} contains non-finite JSON constant: {value}")

    try:
        result = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except ValidationError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{label} is not valid UTF-8 JSON") from exc

    def reject_nonfinite(value: Any) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValidationError(f"{label} contains a non-finite number")
        if isinstance(value, dict):
            for nested in value.values():
                reject_nonfinite(nested)
        elif isinstance(value, list):
            for nested in value:
                reject_nonfinite(nested)

    reject_nonfinite(result)
    return result


def _record_for_bytes(payload: bytes, path: str | Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    label = (
        resolved.relative_to(relative_to.resolve()).as_posix()
        if relative_to is not None
        else str(resolved)
    )
    return {
        "path": label,
        "bytes": int(resolved.stat().st_size),
        "sha256": _sha256_file(resolved),
    }


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    primary_error: BaseException | None = None
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        cleanup_error: OSError | None = None
        for _attempt in range(_CLEANUP_ATTEMPTS):
            try:
                temporary.unlink(missing_ok=True)
            except OSError as exc:
                cleanup_error = exc
                if _attempt + 1 < _CLEANUP_ATTEMPTS:
                    time.sleep(0.01 * (_attempt + 1))
                continue
            cleanup_error = None
            break
        if cleanup_error is not None:
            message = f"could not remove atomic-write temporary file: {temporary}"
            if primary_error is None:
                raise ValidationError(message) from cleanup_error
            primary_error.add_note(f"atomic-write cleanup also failed: {message}")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_bytes(path, _canonical_json_bytes(payload))


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _formal_runtime_identity() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pyarrow_required_for_snapshot_rederivation": True,
    }


def _require_canonical_utc_timestamp(value: Any, *, label: str) -> None:
    if not isinstance(value, str):
        raise ValidationError(f"{label} is not a canonical UTC timestamp")
    try:
        parsed = time.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ValidationError(f"{label} is not a canonical UTC timestamp") from exc
    if time.strftime("%Y-%m-%dT%H:%M:%SZ", parsed) != value:
        raise ValidationError(f"{label} is not a canonical UTC timestamp")


def _safe_child(base: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValidationError("artifact path must be a non-empty string")
    candidate = (base / relative).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError as exc:
        raise ValidationError(f"artifact path escapes its root: {relative}") from exc
    return candidate


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _has_reparse_point(path: Path) -> bool:
    try:
        attributes = path.stat(follow_symlinks=False).st_file_attributes
    except (AttributeError, OSError):
        return False
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _reject_link_or_reparse_components(path: Path, *, label: str) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = current.stat(follow_symlinks=False)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValidationError(f"cannot inspect {label}: {current}") from exc
        if current.is_symlink() or bool(
            getattr(metadata, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        ):
            raise ValidationError(
                f"{label} contains a symlink or reparse point: {current}"
            )


def _open_regular_source_once(path: Path, *, label: str) -> Any:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValidationError(f"cannot open {label}: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        lexical = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or not os.path.samestat(opened, lexical)
            or bool(
                getattr(lexical, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            )
        ):
            raise ValidationError(f"{label} is not a stable regular file: {path}")
        return os.fdopen(descriptor, "rb")
    except Exception:
        os.close(descriptor)
        raise


_CLEANUP_ATTEMPTS = 3


def _directory_identity(path: Path, *, label: str) -> tuple[int, int]:
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ValidationError(f"cannot inspect {label}: {path}") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or path.is_symlink()
        or bool(
            getattr(metadata, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        )
    ):
        raise ValidationError(f"{label} is not a stable directory: {path}")
    return int(metadata.st_dev), int(metadata.st_ino)


def _require_directory_identity(
    path: Path, expected: tuple[int, int], *, label: str
) -> None:
    if _directory_identity(path, label=label) != expected:
        raise ValidationError(f"{label} identity changed: {path}")


def _pin_directory(path: Path, *, label: str) -> _PinnedDirectory:
    """Open a directory without following links and retain its identity."""
    expected = _directory_identity(path, label=label)
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        create_file = ctypes.WinDLL("kernel32", use_last_error=True).CreateFileW
        create_file.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        create_file.restype = wintypes.HANDLE
        handle = create_file(
            str(path),
            0x00010000 | 0x00000080 | 0x00000001,
            # DELETE | FILE_READ_ATTRIBUTES | FILE_LIST_DIRECTORY
            0x00000001 | 0x00000002,  # FILE_SHARE_READ | FILE_SHARE_WRITE
            None,
            3,  # OPEN_EXISTING
            0x02000000 | 0x00200000,  # BACKUP_SEMANTICS | OPEN_REPARSE_POINT
            None,
        )
        invalid = ctypes.c_void_p(-1).value
        if handle in (None, invalid):
            error = ctypes.get_last_error()
            raise ValidationError(f"cannot pin {label}: {path}") from OSError(
                error, os.strerror(error), str(path)
            )
        handle_owner = _OwnedWindowsHandle(int(handle))
        pin = _PinnedDirectory(
            path=path, identity=expected, windows_handle=handle_owner
        )
        try:
            _require_directory_identity(path, expected, label=label)
        except BaseException:
            handle_owner.close()
            raise
        return pin

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValidationError(f"cannot pin {label}: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (int(opened.st_dev), int(opened.st_ino)) != expected
        ):
            raise ValidationError(f"{label} identity changed while opening: {path}")
    except BaseException:
        os.close(descriptor)
        raise
    return _PinnedDirectory(path=path, identity=expected, posix_fd=descriptor)


def _close_pinned_directory(pin: _PinnedDirectory, *, label: str) -> None:
    if pin.windows_handle is not None:
        handle = pin.windows_handle
        pin.windows_handle = None
        error = handle.close()
        if error is not None:
            raise ValidationError(f"could not close {label} handle") from OSError(
                error, os.strerror(error), str(pin.path)
            )
    if pin.posix_fd is not None:
        descriptor = pin.posix_fd
        pin.posix_fd = None
        try:
            os.close(descriptor)
        except OSError as exc:
            raise ValidationError(f"could not close {label} descriptor") from exc


def _require_leaf_name(name: str, *, label: str) -> None:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValidationError(f"{label} name is noncanonical")


def _windows_open_relative_handle(
    directory: _PinnedDirectory,
    name: str,
    *,
    desired_access: int,
    share_access: int,
    create_disposition: int,
    create_options: int,
    label: str,
) -> _OwnedWindowsHandle:
    """Open a child relative to a retained Windows directory handle."""
    import ctypes
    from ctypes import wintypes

    _require_leaf_name(name, label=label)
    if directory.windows_handle is None:
        raise ValidationError(f"{label} directory handle is closed")

    class UnicodeString(ctypes.Structure):
        _fields_ = [
            ("Length", wintypes.USHORT),
            ("MaximumLength", wintypes.USHORT),
            ("Buffer", wintypes.LPWSTR),
        ]

    class ObjectAttributes(ctypes.Structure):
        _fields_ = [
            ("Length", wintypes.ULONG),
            ("RootDirectory", wintypes.HANDLE),
            ("ObjectName", ctypes.POINTER(UnicodeString)),
            ("Attributes", wintypes.ULONG),
            ("SecurityDescriptor", wintypes.LPVOID),
            ("SecurityQualityOfService", wintypes.LPVOID),
        ]

    class IoStatusBlock(ctypes.Structure):
        _fields_ = [("Status", ctypes.c_void_p), ("Information", ctypes.c_size_t)]

    name_buffer = ctypes.create_unicode_buffer(name)
    encoded_length = len(name.encode("utf-16-le"))
    object_name = UnicodeString(
        encoded_length,
        encoded_length + ctypes.sizeof(ctypes.c_wchar),
        ctypes.cast(name_buffer, wintypes.LPWSTR),
    )
    attributes = ObjectAttributes(
        ctypes.sizeof(ObjectAttributes),
        wintypes.HANDLE(directory.windows_handle.value),
        ctypes.pointer(object_name),
        0x00000040,  # OBJ_CASE_INSENSITIVE
        None,
        None,
    )
    io_status = IoStatusBlock()
    handle_owner = _OwnedWindowsHandle()
    nt_create_file = ctypes.WinDLL("ntdll", use_last_error=True).NtCreateFile
    nt_create_file.argtypes = [
        ctypes.POINTER(wintypes.HANDLE),
        wintypes.DWORD,
        ctypes.POINTER(ObjectAttributes),
        ctypes.POINTER(IoStatusBlock),
        ctypes.c_void_p,
        wintypes.ULONG,
        wintypes.ULONG,
        wintypes.ULONG,
        wintypes.ULONG,
        ctypes.c_void_p,
        wintypes.ULONG,
    ]
    nt_create_file.restype = ctypes.c_long
    status = nt_create_file(
        handle_owner.pointer,
        desired_access,
        ctypes.byref(attributes),
        ctypes.byref(io_status),
        None,
        0x00000080,  # FILE_ATTRIBUTE_NORMAL
        share_access,
        create_disposition,
        create_options,
        None,
        0,
    )
    if status < 0:
        rtl_error = ctypes.WinDLL("ntdll").RtlNtStatusToDosError
        rtl_error.argtypes = [ctypes.c_long]
        rtl_error.restype = wintypes.ULONG
        error = int(rtl_error(status))
        raise OSError(error, os.strerror(error), str(directory.path / name))
    try:
        handle_owner.value
    except RuntimeError:
        raise ValidationError(f"could not open {label}: {name}")
    return handle_owner


def _windows_list_pinned_directory_names(
    directory: _PinnedDirectory, *, label: str
) -> list[str]:
    """Enumerate one retained Windows directory handle without reopening its path."""
    import ctypes
    from ctypes import wintypes

    if directory.windows_handle is None:
        raise ValidationError(f"{label} directory handle is closed")

    class IoStatusBlock(ctypes.Structure):
        _fields_ = [("Status", ctypes.c_void_p), ("Information", ctypes.c_size_t)]

    query = ctypes.WinDLL("ntdll", use_last_error=True).NtQueryDirectoryFile
    query.argtypes = [
        wintypes.HANDLE,
        wintypes.HANDLE,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(IoStatusBlock),
        ctypes.c_void_p,
        wintypes.ULONG,
        ctypes.c_int,
        wintypes.BOOLEAN,
        ctypes.c_void_p,
        wintypes.BOOLEAN,
    ]
    query.restype = ctypes.c_long
    names: list[str] = []
    seen: set[str] = set()
    restart = True
    while True:
        buffer = ctypes.create_string_buffer(64 * 1024)
        io_status = IoStatusBlock()
        status = query(
            wintypes.HANDLE(directory.windows_handle.value),
            None,
            None,
            None,
            ctypes.byref(io_status),
            buffer,
            len(buffer),
            12,  # FileNamesInformation
            False,
            None,
            restart,
        )
        status_code = ctypes.c_ulong(status).value
        if status_code == 0x80000006:  # STATUS_NO_MORE_FILES
            break
        if status_code != 0:
            rtl_error = ctypes.WinDLL("ntdll").RtlNtStatusToDosError
            rtl_error.argtypes = [ctypes.c_long]
            rtl_error.restype = wintypes.ULONG
            error = int(rtl_error(status))
            raise ValidationError(
                f"could not enumerate {label}: {directory.path}"
            ) from OSError(error, os.strerror(error), str(directory.path))
        used = int(io_status.Information)
        if used <= 0 or used > len(buffer):
            raise ValidationError(f"{label} directory listing is invalid")
        offset = 0
        while True:
            if offset + 12 > used:
                raise ValidationError(f"{label} directory listing is invalid")
            next_offset = int.from_bytes(buffer.raw[offset : offset + 4], "little")
            name_bytes = int.from_bytes(buffer.raw[offset + 8 : offset + 12], "little")
            end = offset + 12 + name_bytes
            if name_bytes % 2 or end > used:
                raise ValidationError(f"{label} directory listing is invalid")
            try:
                name = buffer.raw[offset + 12 : end].decode("utf-16-le")
            except UnicodeDecodeError as exc:
                raise ValidationError(
                    f"{label} directory listing contains an invalid name"
                ) from exc
            if name not in {".", ".."}:
                _require_leaf_name(name, label=label)
                folded = name.casefold()
                if folded in seen:
                    raise ValidationError(
                        f"{label} directory listing contains a duplicate name"
                    )
                seen.add(folded)
                names.append(name)
            if next_offset == 0:
                break
            if next_offset < 12 or next_offset % 4 or offset + next_offset >= used:
                raise ValidationError(f"{label} directory listing is invalid")
            offset += next_offset
        restart = False
    return names


def _windows_relative_entry_metadata(
    directory: _PinnedDirectory, name: str, *, label: str
) -> os.stat_result:
    """Inspect a child through the retained parent handle without a path reopen."""
    try:
        handle_owner = _windows_open_relative_handle(
            directory,
            name,
            desired_access=0x00000080 | 0x00100000,
            share_access=0x00000001 | 0x00000002,
            create_disposition=1,
            create_options=0x00000020 | 0x00200000,
            label=label,
        )
    except OSError as exc:
        raise ValidationError(f"could not inspect {label}: {name}") from exc
    return _windows_fstat_owned_handle(handle_owner, label=label, display=name)


def _windows_close_handle(raw_handle: int) -> int | None:
    import ctypes
    from ctypes import wintypes

    close_handle = ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    if not close_handle(wintypes.HANDLE(raw_handle)):
        return int(ctypes.get_last_error())
    return None


def _windows_fstat_owned_handle(
    handle: int | _OwnedWindowsHandle, *, label: str, display: str | Path
) -> os.stat_result:
    """Consume one Win32 handle and close it even if CRT adoption or fstat fails."""
    import msvcrt

    if isinstance(handle, _OwnedWindowsHandle):
        handle_owner: _OwnedWindowsHandle | None = handle
        raw_handle = handle.value
    else:
        handle_owner = None
        raw_handle = handle
    descriptor: int | None = None
    try:
        try:
            descriptor = msvcrt.open_osfhandle(raw_handle, os.O_RDONLY)
            if handle_owner is not None:
                handle_owner.release()
        except BaseException:
            if handle_owner is None:
                _windows_close_handle(raw_handle)
            else:
                handle_owner.close()
            raise
        return os.fstat(descriptor)
    except OSError as exc:
        raise ValidationError(f"could not inspect {label}: {display}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _windows_duplicate_handle_metadata(
    raw_handle: int, *, label: str, display: str | Path
) -> os.stat_result:
    """fstat a duplicate while leaving the caller's retained handle open."""
    import ctypes
    from ctypes import wintypes

    duplicate = wintypes.HANDLE()
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.DuplicateHandle.argtypes = [
        wintypes.HANDLE,
        wintypes.HANDLE,
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HANDLE),
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    ]
    kernel32.DuplicateHandle.restype = wintypes.BOOL
    current_process = kernel32.GetCurrentProcess()
    if not kernel32.DuplicateHandle(
        current_process,
        wintypes.HANDLE(raw_handle),
        current_process,
        ctypes.byref(duplicate),
        0,
        False,
        0x00000002,  # DUPLICATE_SAME_ACCESS
    ):
        error = ctypes.get_last_error()
        raise ValidationError(f"could not inspect {label}: {display}") from OSError(
            error, os.strerror(error), str(display)
        )
    duplicate_value = duplicate.value
    if duplicate_value is None:
        raise ValidationError(f"could not inspect {label}: {display}")
    return _windows_fstat_owned_handle(
        int(duplicate_value), label=label, display=display
    )


def _create_pinned_directory(
    parent: _PinnedDirectory, name: str, path: Path, *, label: str
) -> _PinnedDirectory:
    """Create a child exclusively and retain the identity observed after opening.

    POSIX ``mkdirat`` does not return a directory descriptor.  The checks below
    therefore provide point-in-time name-to-handle binding, not immunity to a
    same-UID replacement between ``mkdirat`` and the first ``openat``.
    """
    _require_leaf_name(name, label=label)
    if os.name == "nt":
        try:
            handle_owner = _windows_open_relative_handle(
                parent,
                name,
                desired_access=(0x00010000 | 0x00000080 | 0x00100000 | 0x00000001),
                share_access=0x00000001 | 0x00000002,
                create_disposition=2,  # FILE_CREATE
                create_options=0x00000001 | 0x00000020 | 0x00200000,
                label=label,
            )
        except OSError as exc:
            if getattr(exc, "winerror", None) in (80, 183) or exc.errno in (
                17,
                80,
                183,
            ):
                raise FileExistsError(str(path)) from exc
            raise ValidationError(f"could not create {label}: {path}") from exc
        try:
            metadata = _windows_duplicate_handle_metadata(
                handle_owner.value, label=label, display=path
            )
        except BaseException:
            handle_owner.close()
            raise
        return _PinnedDirectory(
            path=path,
            identity=(int(metadata.st_dev), int(metadata.st_ino)),
            windows_handle=handle_owner,
        )

    if parent.posix_fd is None:
        raise ValidationError(f"{label} parent descriptor is closed")
    os.mkdir(name, 0o700, dir_fd=parent.posix_fd)
    created = os.stat(name, dir_fd=parent.posix_fd, follow_symlinks=False)
    if not stat.S_ISDIR(created.st_mode):
        raise ValidationError(f"{label} creation identity is invalid")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=parent.posix_fd)
    try:
        metadata = os.fstat(descriptor)
        current = os.stat(name, dir_fd=parent.posix_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or not stat.S_ISDIR(current.st_mode)
            or not os.path.samestat(created, metadata)
            or not os.path.samestat(metadata, current)
        ):
            raise ValidationError(f"{label} identity changed while opening")
        pin = _PinnedDirectory(
            path=path,
            identity=(int(metadata.st_dev), int(metadata.st_ino)),
            posix_fd=descriptor,
        )
        _require_pinned_directory_entry(
            parent, name, pin.identity, label=f"{label} final creation binding"
        )
        return pin
    except BaseException:
        os.close(descriptor)
        raise


def _open_pinned_child_directory(
    parent: _PinnedDirectory,
    name: str,
    path: Path,
    *,
    label: str,
    expected_identity: tuple[int, int] | None = None,
) -> _PinnedDirectory:
    """Open one existing child relative to ``parent`` and pin that identity."""
    _require_leaf_name(name, label=label)
    if os.name == "nt":
        try:
            handle_owner = _windows_open_relative_handle(
                parent,
                name,
                desired_access=(0x00010000 | 0x00000080 | 0x00100000 | 0x00000001),
                share_access=0x00000001 | 0x00000002,
                create_disposition=1,  # FILE_OPEN
                create_options=0x00000001 | 0x00000020 | 0x00200000,
                label=label,
            )
        except OSError as exc:
            raise ValidationError(f"could not open {label}: {path}") from exc
        try:
            metadata = _windows_duplicate_handle_metadata(
                handle_owner.value, label=label, display=path
            )
        except BaseException:
            handle_owner.close()
            raise
        identity = (int(metadata.st_dev), int(metadata.st_ino))
        if not stat.S_ISDIR(metadata.st_mode) or (
            expected_identity is not None and identity != expected_identity
        ):
            handle_owner.close()
            raise ValidationError(f"{label} identity changed: {path}")
        return _PinnedDirectory(
            path=path,
            identity=identity,
            windows_handle=handle_owner,
        )

    if parent.posix_fd is None:
        raise ValidationError(f"{label} parent descriptor is closed")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent.posix_fd)
    except OSError as exc:
        raise ValidationError(f"could not open {label}: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        identity = (int(metadata.st_dev), int(metadata.st_ino))
        if not stat.S_ISDIR(metadata.st_mode) or (
            expected_identity is not None and identity != expected_identity
        ):
            raise ValidationError(f"{label} identity changed: {path}")
    except BaseException:
        os.close(descriptor)
        raise
    return _PinnedDirectory(path=path, identity=identity, posix_fd=descriptor)


def _require_pinned_directory_entry(
    parent: _PinnedDirectory,
    name: str,
    expected_identity: tuple[int, int],
    *,
    label: str,
) -> None:
    if os.name == "nt":
        try:
            handle_owner = _windows_open_relative_handle(
                parent,
                name,
                desired_access=0x00000080 | 0x00100000,
                share_access=0x00000001 | 0x00000002 | 0x00000004,
                create_disposition=1,
                create_options=0x00000001 | 0x00000020 | 0x00200000,
                label=label,
            )
        except OSError as exc:
            raise ValidationError(f"could not inspect {label}: {name}") from exc
        try:
            metadata = _windows_duplicate_handle_metadata(
                handle_owner.value, label=label, display=name
            )
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or (int(metadata.st_dev), int(metadata.st_ino)) != expected_identity
            ):
                raise ValidationError(f"{label} identity changed: {name}")
        finally:
            handle_owner.close()
        return
    child = _open_pinned_child_directory(
        parent,
        name,
        parent.path / name,
        label=label,
        expected_identity=expected_identity,
    )
    _close_pinned_directory(child, label=label)


def _open_pinned_exclusive_file(
    directory: _PinnedDirectory, name: str, *, label: str
) -> Any:
    """Create one file under an open directory without following a leaf link."""
    _require_leaf_name(name, label=label)
    if os.name == "nt":
        import msvcrt

        try:
            handle_owner = _windows_open_relative_handle(
                directory,
                name,
                desired_access=(0x40000000 | 0x00010000 | 0x00000080 | 0x00100000),
                share_access=0x00000001,
                create_disposition=2,  # FILE_CREATE
                create_options=0x00000040 | 0x00000020 | 0x00200000,
                label=label,
            )
        except OSError as exc:
            raise ValidationError(f"could not create {label}: {name}") from exc
        try:
            descriptor = msvcrt.open_osfhandle(
                handle_owner.value, os.O_WRONLY | getattr(os, "O_BINARY", 0)
            )
            handle_owner.release()
        except BaseException:
            handle_owner.close()
            raise
        try:
            return os.fdopen(descriptor, "wb")
        except BaseException:
            os.close(descriptor)
            raise

    if directory.posix_fd is None:
        raise ValidationError(f"{label} directory descriptor is closed")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=directory.posix_fd)
    except OSError as exc:
        raise ValidationError(f"could not create {label}: {name}") from exc
    try:
        return os.fdopen(descriptor, "wb")
    except BaseException:
        os.close(descriptor)
        raise


def _write_pinned_bytes(
    directory: _PinnedDirectory,
    name: str,
    payload: bytes,
    identities: dict[str, tuple[int, int]],
    *,
    label: str,
) -> None:
    """Write final bytes once and remember the exact created file identity."""
    with _open_pinned_exclusive_file(directory, name, label=label) as handle:
        metadata = os.fstat(handle.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise ValidationError(f"{label} is not a regular file: {name}")
        identities[name] = (int(metadata.st_dev), int(metadata.st_ino))
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _require_pinned_file_identity(
    directory: _PinnedDirectory,
    name: str,
    expected_identity: tuple[int, int],
    *,
    label: str,
) -> None:
    descriptor, observed = _open_pinned_regular_file(
        directory,
        name,
        label=label,
    )
    try:
        if observed != expected_identity:
            raise ValidationError(f"{label} identity changed: {name}")
    finally:
        os.close(descriptor)


def _open_pinned_regular_file(
    directory: _PinnedDirectory,
    name: str,
    *,
    label: str,
) -> tuple[int, tuple[int, int]]:
    """Open one regular file relative to a retained directory identity."""
    _require_leaf_name(name, label=label)
    descriptor: int | None = None
    handle_owner: _OwnedWindowsHandle | None = None
    try:
        if os.name == "nt":
            import msvcrt

            handle_owner = _windows_open_relative_handle(
                directory,
                name,
                desired_access=0x80000000 | 0x00000080 | 0x00100000,
                share_access=0x00000001 | 0x00000002 | 0x00000004,
                create_disposition=1,  # FILE_OPEN
                create_options=0x00000040 | 0x00000020 | 0x00200000,
                label=label,
            )
            descriptor = msvcrt.open_osfhandle(handle_owner.value, os.O_RDONLY)
            handle_owner.release()
            handle_owner = None
        else:
            if directory.posix_fd is None:
                raise ValidationError(f"{label} directory descriptor is closed")
            descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory.posix_fd,
            )
        metadata = os.fstat(descriptor)
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None
        if handle_owner is not None:
            handle_owner.close()
            handle_owner = None
        raise ValidationError(f"cannot inspect {label}: {name}") from exc
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None
        if handle_owner is not None:
            handle_owner.close()
        raise
    if descriptor is None or not stat.S_ISREG(metadata.st_mode):
        if descriptor is not None:
            os.close(descriptor)
        raise ValidationError(f"{label} is not a regular file: {name}")
    return descriptor, (int(metadata.st_dev), int(metadata.st_ino))


def _pinned_regular_file_exists(
    directory: _PinnedDirectory, name: str, *, label: str
) -> bool:
    try:
        descriptor, _identity = _open_pinned_regular_file(directory, name, label=label)
    except ValidationError as exc:
        cause = exc.__cause__
        if isinstance(cause, OSError) and (
            cause.errno in (2,) or getattr(cause, "winerror", None) in (2, 3)
        ):
            return False
        raise
    os.close(descriptor)
    return True


def _rename_pinned_entry_no_replace(
    directory: _PinnedDirectory, source: str, target: str, *, label: str
) -> None:
    """Rename one POSIX entry relative to its pin without replacing a target."""
    import ctypes
    import errno

    if directory.posix_fd is None:
        raise ValidationError(f"{label} directory descriptor is closed")
    _require_leaf_name(source, label=label)
    _require_leaf_name(target, label=label)
    library = ctypes.CDLL(None, use_errno=True)
    operation: Any
    flags: int
    if sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        operation = library.renameat2
        flags = 1  # RENAME_NOREPLACE
    elif sys.platform == "darwin" and hasattr(library, "renameatx_np"):
        operation = library.renameatx_np
        flags = 0x00000004  # RENAME_EXCL
    else:
        raise ValidationError(
            f"{label} requires an atomic no-replace rename on this platform"
        )
    operation.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    operation.restype = ctypes.c_int
    result = operation(
        directory.posix_fd,
        os.fsencode(source),
        directory.posix_fd,
        os.fsencode(target),
        flags,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(error, os.strerror(error), target)
        raise OSError(error, os.strerror(error), source)


def _restore_quarantined_entry_after_mismatch(
    directory: _PinnedDirectory,
    quarantine: str,
    original: str,
    primary_error: BaseException,
    *,
    label: str,
) -> None:
    """Restore a post-rename substitution without replacing a concurrent name."""
    try:
        _rename_pinned_entry_no_replace(
            directory,
            quarantine,
            original,
            label=f"{label} restoration",
        )
    except BaseException as restoration_error:
        primary_error.add_note(f"{label} restoration also failed: {restoration_error}")
    raise primary_error


def _delete_pinned_file(
    directory: _PinnedDirectory,
    name: str,
    expected_identity: tuple[int, int],
    *,
    label: str,
) -> None:
    """Delete only the exact file created by this operation."""
    _require_leaf_name(name, label=label)
    if os.name == "nt":
        import ctypes
        import msvcrt
        from ctypes import wintypes

        try:
            handle_owner = _windows_open_relative_handle(
                directory,
                name,
                desired_access=0x00010000 | 0x00000080 | 0x00100000,
                share_access=0x00000001 | 0x00000002,
                create_disposition=1,  # FILE_OPEN
                create_options=0x00000040 | 0x00000020 | 0x00200000,
                label=label,
            )
        except OSError as exc:
            if getattr(exc, "winerror", None) in (2, 3) or exc.errno in (2,):
                return
            raise ValidationError(f"could not open {label}: {name}") from exc
        descriptor: int | None = None
        try:
            descriptor = msvcrt.open_osfhandle(handle_owner.value, os.O_RDONLY)
            handle_owner.release()
            metadata = os.fstat(descriptor)
            observed = (int(metadata.st_dev), int(metadata.st_ino))
            if not stat.S_ISREG(metadata.st_mode) or observed != expected_identity:
                raise ValidationError(f"refusing to remove substituted {label}: {name}")

            class FileDispositionInfo(ctypes.Structure):
                _fields_ = [("DeleteFile", wintypes.BOOLEAN)]

            raw_handle = msvcrt.get_osfhandle(descriptor)
            disposition = FileDispositionInfo(True)
            set_information = ctypes.WinDLL(
                "kernel32", use_last_error=True
            ).SetFileInformationByHandle
            set_information.argtypes = [
                wintypes.HANDLE,
                ctypes.c_int,
                wintypes.LPVOID,
                wintypes.DWORD,
            ]
            set_information.restype = wintypes.BOOL
            if not set_information(
                raw_handle,
                4,  # FileDispositionInfo
                ctypes.byref(disposition),
                ctypes.sizeof(disposition),
            ):
                error = ctypes.get_last_error()
                raise ValidationError(f"could not remove {label}: {name}") from OSError(
                    error, os.strerror(error), str(directory.path / name)
                )
        finally:
            if descriptor is not None:
                os.close(descriptor)
            else:
                handle_owner.close()
        return

    if directory.posix_fd is None:
        raise ValidationError(f"{label} directory descriptor is closed")
    quarantine = f".cleanup-{uuid.uuid4().hex}"
    try:
        _rename_pinned_entry_no_replace(
            directory,
            name,
            quarantine,
            label=f"{label} quarantine",
        )
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ValidationError(f"could not quarantine {label}: {name}") from exc
    try:
        metadata = os.stat(quarantine, dir_fd=directory.posix_fd, follow_symlinks=False)
        observed = (int(metadata.st_dev), int(metadata.st_ino))
        if not stat.S_ISREG(metadata.st_mode) or observed != expected_identity:
            raise ValidationError(f"refusing to remove substituted {label}: {name}")
    except BaseException as primary_error:
        _restore_quarantined_entry_after_mismatch(
            directory,
            quarantine,
            name,
            primary_error,
            label=label,
        )
    try:
        os.unlink(quarantine, dir_fd=directory.posix_fd)
    except OSError as exc:
        raise ValidationError(f"could not remove {label}: {name}") from exc


def _quarantine_pinned_directory(
    directory: _PinnedDirectory, parent: _PinnedDirectory, *, label: str
) -> str:
    """Detach an owned POSIX directory before traversal; Windows pins the name."""
    if os.name == "nt":
        return directory.path.name
    if directory.posix_fd is None or parent.posix_fd is None:
        raise ValidationError(f"{label} descriptor is closed")
    try:
        current = os.stat(
            directory.path.name,
            dir_fd=parent.posix_fd,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise ValidationError(f"cannot inspect {label} name") from exc
    if (
        not stat.S_ISDIR(current.st_mode)
        or (int(current.st_dev), int(current.st_ino)) != directory.identity
    ):
        raise ValidationError(f"refusing to quarantine substituted {label}")
    quarantine = f".{directory.path.name}.cleanup-{uuid.uuid4().hex}"
    try:
        _rename_pinned_entry_no_replace(
            parent,
            directory.path.name,
            quarantine,
            label=f"{label} quarantine",
        )
    except OSError as exc:
        raise ValidationError(
            f"could not quarantine {label}: {directory.path}"
        ) from exc
    try:
        metadata = os.stat(quarantine, dir_fd=parent.posix_fd, follow_symlinks=False)
        observed = (int(metadata.st_dev), int(metadata.st_ino))
        if not stat.S_ISDIR(metadata.st_mode) or observed != directory.identity:
            raise ValidationError(f"refusing to clean substituted {label}")
    except BaseException as primary_error:
        _restore_quarantined_entry_after_mismatch(
            parent,
            quarantine,
            directory.path.name,
            primary_error,
            label=label,
        )
    return quarantine


def _purge_pinned_tree(directory: _PinnedDirectory, *, label: str) -> None:
    """Remove a private tree without following links or reopening its root."""
    if os.name != "nt":
        if directory.posix_fd is None:
            raise ValidationError(f"{label} descriptor is closed")
        for entry in list(os.scandir(directory.posix_fd)):
            metadata = entry.stat(follow_symlinks=False)
            identity = (int(metadata.st_dev), int(metadata.st_ino))
            if stat.S_ISDIR(metadata.st_mode):
                child = _open_pinned_child_directory(
                    directory,
                    entry.name,
                    directory.path / entry.name,
                    label=label,
                    expected_identity=identity,
                )
                quarantined_name: str | None = None
                try:
                    quarantined_name = _quarantine_pinned_directory(
                        child, directory, label=label
                    )
                    _purge_pinned_tree(child, label=label)
                    _remove_pinned_empty_directory(
                        child,
                        directory,
                        label=label,
                        entry_name=quarantined_name,
                    )
                finally:
                    _close_pinned_directory(child, label=label)
            elif stat.S_ISREG(metadata.st_mode):
                _delete_pinned_file(directory, entry.name, identity, label=label)
            else:
                raise ValidationError(f"{label} contains a non-regular entry")
        return

    for child_name in _windows_list_pinned_directory_names(directory, label=label):
        metadata = _windows_relative_entry_metadata(directory, child_name, label=label)
        if bool(
            getattr(metadata, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        ):
            raise ValidationError(f"{label} contains a link or reparse point")
        identity = (int(metadata.st_dev), int(metadata.st_ino))
        if stat.S_ISDIR(metadata.st_mode):
            child = _open_pinned_child_directory(
                directory,
                child_name,
                directory.path / child_name,
                label=label,
                expected_identity=identity,
            )
            try:
                _purge_pinned_tree(child, label=label)
                _remove_pinned_empty_directory(child, directory, label=label)
            finally:
                _close_pinned_directory(child, label=label)
        elif stat.S_ISREG(metadata.st_mode):
            _delete_pinned_file(directory, child_name, identity, label=label)
        else:
            raise ValidationError(f"{label} contains a non-regular entry")


def _pinned_directory_names(directory: _PinnedDirectory, *, label: str) -> list[str]:
    if directory.posix_fd is not None:
        with os.scandir(directory.posix_fd) as entries:
            return [entry.name for entry in entries]
    return _windows_list_pinned_directory_names(directory, label=label)


def _require_pinned_inventory(
    directory: _PinnedDirectory,
    identities: Mapping[str, tuple[int, int]],
    expected_order: Sequence[str],
    *,
    label: str,
) -> None:
    """Bind the complete committed inventory through the retained directory pin."""
    if tuple(identities) != tuple(expected_order):
        raise ValidationError(f"{label} creation order is noncanonical")
    names = _pinned_directory_names(directory, label=label)
    if len(names) != len(set(names)) or set(names) != set(expected_order):
        raise ValidationError(f"{label} inventory is noncanonical")
    for name in expected_order:
        _require_pinned_file_identity(
            directory,
            name,
            identities[name],
            label=f"{label} {name}",
        )


def _remove_owned_capture_tree(
    path: Path, expected_identity: tuple[int, int], *, label: str
) -> None:
    if not os.path.lexists(path):
        return
    if os.name == "nt":
        directory = _pin_directory(path, label=label)
        windows_primary_error: BaseException | None = None
        try:
            if directory.identity != expected_identity:
                raise ValidationError(f"{label} identity changed: {path}")
            _purge_pinned_tree(directory, label=label)
            _remove_pinned_empty_directory(
                directory,
                directory,
                label=label,
            )
        except BaseException as exc:
            windows_primary_error = exc
            raise
        finally:
            try:
                _close_pinned_directory(directory, label=label)
            except Exception as cleanup_error:
                if windows_primary_error is None:
                    raise
                windows_primary_error.add_note(
                    f"{label} handle close also failed: {cleanup_error}"
                )
        return
    parent = _pin_directory(path.parent, label=f"{label} parent")
    directory = _pin_directory(path, label=label)
    primary_error: BaseException | None = None
    try:
        if directory.identity != expected_identity:
            raise ValidationError(f"{label} identity changed: {path}")
        quarantined_name = _quarantine_pinned_directory(directory, parent, label=label)
        _purge_pinned_tree(directory, label=label)
        _remove_pinned_empty_directory(
            directory, parent, label=label, entry_name=quarantined_name
        )
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        for pin, pin_label in ((directory, label), (parent, f"{label} parent")):
            try:
                _close_pinned_directory(pin, label=pin_label)
            except Exception as cleanup_error:
                if primary_error is None:
                    raise
                primary_error.add_note(
                    f"{pin_label} close also failed: {cleanup_error}"
                )


class _CaptureStore:
    """Disk-backed, hash-recorded captures used by one validation operation."""

    def __init__(self, *, prefix: str = "depplm-clinical-capture-") -> None:
        self.root = Path(tempfile.mkdtemp(prefix=prefix))
        self._root_identity = _directory_identity(
            self.root, label="private clinical capture directory"
        )
        self._root_pin = _pin_directory(
            self.root, label="private clinical capture directory"
        )
        if self._root_pin.identity != self._root_identity:
            _close_pinned_directory(
                self._root_pin, label="private clinical capture directory"
            )
            raise ValidationError(
                "private clinical capture directory changed while being pinned"
            )
        self._root_parent_pin = (
            None
            if os.name == "nt"
            else _pin_directory(
                self.root.parent,
                label="private clinical capture parent",
            )
        )
        self._directory_pins: dict[tuple[str, ...], _PinnedDirectory] = {
            (): self._root_pin
        }
        self._file_identities: dict[tuple[str, ...], tuple[int, int]] = {}
        self._process_temp_directories: list[
            tuple[Path, tuple[int, int], str]
        ] = []
        self._counter = 0
        self._closed = False

    def __enter__(self) -> _CaptureStore:
        return self

    def __exit__(self, _kind: Any, _value: Any, _traceback: Any) -> None:
        try:
            self._cleanup()
        except Exception as cleanup_error:
            if _value is None:
                raise
            _value.add_note(
                "private clinical capture cleanup also failed: " f"{cleanup_error}"
            )

    def _capture_parent(self, parts: tuple[str, ...]) -> _PinnedDirectory:
        current_key: tuple[str, ...] = ()
        current = self._root_pin
        for part in parts:
            _require_leaf_name(part, label="private capture directory")
            current_key = (*current_key, part)
            existing = self._directory_pins.get(current_key)
            if existing is not None:
                current = existing
                continue
            child = _create_pinned_directory(
                current,
                part,
                self.root.joinpath(*current_key),
                label="private capture directory",
            )
            self._directory_pins[current_key] = child
            current = child
        return current

    @staticmethod
    def _relative_parts(relative: Path, *, label: str) -> tuple[str, ...]:
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in ("", ".", "..") for part in relative.parts)
        ):
            raise ValidationError(f"{label} path is not canonical relative")
        parts = tuple(relative.parts)
        for part in parts:
            _require_leaf_name(part, label=label)
        return parts

    def create_private_directory(
        self, relative: Path, *, label: str
    ) -> _PinnedDirectory:
        """Create and retain an exclusive private directory hierarchy."""
        parts = self._relative_parts(relative, label=label)
        return self._capture_parent(parts)

    def create_process_temp_directory(self, *, label: str) -> Path:
        """Create an owned child-process temp directory without a retained pin.

        Windows runtimes may need to create and remove entries below TEMP.  A
        retained no-delete-share directory handle would prevent that normal
        operation, so this directory is identity-bound for later cleanup but is
        deliberately not kept open while the child process runs.
        """
        path = Path(
            tempfile.mkdtemp(
                prefix="depplm-clinical-replay-temp-",
                dir=self.root.parent,
            )
        )
        identity = _directory_identity(path, label=label)
        self._process_temp_directories.append((path, identity, label))
        return path

    def _pin_existing_parent(
        self, parts: tuple[str, ...], *, label: str
    ) -> _PinnedDirectory:
        current_key: tuple[str, ...] = ()
        current = self._root_pin
        for part in parts:
            current_key = (*current_key, part)
            existing = self._directory_pins.get(current_key)
            if existing is not None:
                current = existing
                continue
            child = _open_pinned_child_directory(
                current,
                part,
                self.root.joinpath(*current_key),
                label=label,
            )
            self._directory_pins[current_key] = child
            current = child
        return current

    def write_private_bytes(
        self, relative: Path, payload: bytes, *, label: str
    ) -> _CapturedFile:
        """Write one exclusively created private file via the retained parent handle."""
        parts = self._relative_parts(relative, label=label)
        parent = self._capture_parent(parts[:-1])
        identities: dict[str, tuple[int, int]] = {}
        _write_pinned_bytes(parent, parts[-1], payload, identities, label=label)
        identity = identities[parts[-1]]
        self._file_identities[parts] = identity
        return _CapturedFile(
            source_path=self.root.joinpath(*parts),
            path=self.root.joinpath(*parts),
            bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )

    def register_private_file(
        self,
        relative: Path,
        *,
        label: str,
        expected_identity: tuple[int, int] | None = None,
    ) -> tuple[int, int]:
        """Adopt a child-created regular file after relative-handle inspection."""
        parts = self._relative_parts(relative, label=label)
        parent = self._pin_existing_parent(parts[:-1], label=label)
        descriptor, identity = _open_pinned_regular_file(parent, parts[-1], label=label)
        try:
            if expected_identity is not None and identity != expected_identity:
                raise ValidationError(f"{label} identity changed: {parts[-1]}")
        finally:
            os.close(descriptor)
        existing = self._file_identities.get(parts)
        if existing is not None and existing != identity:
            raise ValidationError(f"{label} replaced a registered private file")
        self._file_identities[parts] = identity
        return identity

    def verify_private_file(
        self,
        relative: Path,
        *,
        label: str,
        expected_record: Mapping[str, Any],
        max_bytes: int | None = None,
    ) -> _CapturedFile:
        """Hash a registered private file through its retained parent handle."""
        parts = self._relative_parts(relative, label=label)
        parent = self._pin_existing_parent(parts[:-1], label=label)
        descriptor, identity = _open_pinned_regular_file(parent, parts[-1], label=label)
        registered = self._file_identities.get(parts)
        if registered is None or registered != identity:
            os.close(descriptor)
            raise ValidationError(f"{label} is not the registered private file")
        digest = hashlib.sha256()
        byte_count = 0
        try:
            opened = os.fstat(descriptor)
            while True:
                chunk = os.read(descriptor, 4 * 1024 * 1024)
                if not chunk:
                    break
                if max_bytes is not None and byte_count + len(chunk) > max_bytes:
                    raise ValidationError(f"{label} exceeds its capture byte limit")
                digest.update(chunk)
                byte_count += len(chunk)
            after = os.fstat(descriptor)
            if (
                (int(after.st_dev), int(after.st_ino)) != identity
                or after.st_size != byte_count
                or not os.path.samestat(opened, after)
            ):
                raise ValidationError(f"{label} changed while being verified")
        finally:
            os.close(descriptor)
        expected_bytes = expected_record.get("bytes")
        expected_sha256 = expected_record.get("sha256")
        if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
            raise ValidationError(f"{label}.bytes is invalid")
        if not _valid_sha256(expected_sha256):
            raise ValidationError(f"{label}.sha256 is invalid")
        if byte_count != expected_bytes:
            raise ValidationError(f"{label} byte count changed")
        observed_sha256 = digest.hexdigest()
        if observed_sha256 != expected_sha256:
            raise ValidationError(f"{label} SHA-256 changed")
        path = self.root.joinpath(*parts)
        return _CapturedFile(
            source_path=path,
            path=path,
            bytes=byte_count,
            sha256=observed_sha256,
        )

    def adopt_direct_regular_files(
        self,
        relative: Path,
        *,
        label: str,
        max_entries: int,
    ) -> dict[str, tuple[int, int]]:
        """Adopt a bounded flat child-output directory without following names."""
        parts = self._relative_parts(relative, label=label)
        directory = self._pin_existing_parent(parts, label=label)
        if directory.posix_fd is not None:
            with os.scandir(directory.posix_fd) as iterator:
                names = sorted(entry.name for entry in iterator)
        else:
            names = sorted(_windows_list_pinned_directory_names(directory, label=label))
        if len(names) > max_entries:
            raise ValidationError(f"{label} exceeds its file-count limit")
        adopted: dict[str, tuple[int, int]] = {}
        for name in names:
            _require_leaf_name(name, label=label)
            identity = self.register_private_file(Path(*parts, name), label=label)
            adopted[name] = identity
        return adopted

    @staticmethod
    def _pinned_directory_empty(directory: _PinnedDirectory) -> bool:
        return not _pinned_directory_names(
            directory, label="private clinical capture directory"
        )

    @staticmethod
    def _retry_cleanup(operation: Any, *, label: str) -> None:
        last_error: Exception | None = None
        for attempt in range(_CLEANUP_ATTEMPTS):
            try:
                operation()
            except Exception as exc:
                last_error = exc
                if attempt + 1 < _CLEANUP_ATTEMPTS:
                    time.sleep(0.01 * (attempt + 1))
                continue
            return
        raise ValidationError(f"could not complete {label}") from last_error

    def _cleanup(self) -> None:
        if self._closed:
            return
        primary_error: BaseException | None = None
        try:
            for path, identity, label in reversed(self._process_temp_directories):
                self._retry_cleanup(
                    lambda path=path, identity=identity, label=label: _remove_owned_capture_tree(
                        path,
                        identity,
                        label=label,
                    ),
                    label=f"{label} cleanup",
                )
            for parts, identity in sorted(
                self._file_identities.items(), key=lambda item: item[0], reverse=True
            ):
                parent = self._directory_pins[parts[:-1]]
                self._retry_cleanup(
                    lambda parent=parent, parts=parts, identity=identity: _delete_pinned_file(
                        parent,
                        parts[-1],
                        identity,
                        label="private clinical capture file",
                    ),
                    label="private clinical capture file cleanup",
                )
            for parts in sorted(
                (key for key in self._directory_pins if key),
                key=lambda key: (len(key), key),
                reverse=True,
            ):
                directory = self._directory_pins[parts]
                parent = self._directory_pins[parts[:-1]]
                if os.name != "nt" and not self._pinned_directory_empty(directory):
                    raise ValidationError(
                        "private clinical capture directory contains unowned entries"
                    )
                quarantined = _quarantine_pinned_directory(
                    directory, parent, label="private clinical capture directory"
                )
                self._retry_cleanup(
                    lambda directory=directory, parent=parent, quarantined=quarantined: _remove_pinned_empty_directory(
                        directory,
                        parent,
                        label="private clinical capture directory",
                        entry_name=quarantined,
                    ),
                    label="private clinical capture directory cleanup",
                )
            if os.name != "nt" and not self._pinned_directory_empty(self._root_pin):
                raise ValidationError(
                    "private clinical capture directory contains unowned entries"
                )
            if os.name == "nt":
                self._retry_cleanup(
                    lambda: _remove_pinned_empty_directory(
                        self._root_pin,
                        self._root_pin,
                        label="private clinical capture directory",
                    ),
                    label="private clinical capture root cleanup",
                )
            else:
                if self._root_parent_pin is None:
                    raise ValidationError("private clinical capture parent is closed")
                quarantined = _quarantine_pinned_directory(
                    self._root_pin,
                    self._root_parent_pin,
                    label="private clinical capture directory",
                )
                self._retry_cleanup(
                    lambda: _remove_pinned_empty_directory(
                        self._root_pin,
                        self._root_parent_pin,
                        label="private clinical capture directory",
                        entry_name=quarantined,
                    ),
                    label="private clinical capture root cleanup",
                )
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            self._closed = True
            close_errors: list[Exception] = []
            pins = sorted(
                self._directory_pins.items(),
                key=lambda item: (len(item[0]), item[0]),
                reverse=True,
            )
            if self._root_parent_pin is not None:
                pins.append((("<parent>",), self._root_parent_pin))
            for _key, pin in pins:
                try:
                    _close_pinned_directory(
                        pin, label="private clinical capture directory"
                    )
                except Exception as close_error:
                    close_errors.append(close_error)
                    if primary_error is None:
                        continue
                    primary_error.add_note(
                        "private clinical capture handle close also failed: "
                        f"{close_error}"
                    )
            if primary_error is None and close_errors:
                raise close_errors[0]

    def capture(
        self,
        path: Path,
        *,
        label: str,
        expected_record: Mapping[str, Any] | None = None,
        logical_path: str | Path | None = None,
        destination_relative: Path | None = None,
        chunk_size: int = 4 * 1024 * 1024,
        max_bytes: int | None = None,
        _payload_sink: bytearray | None = None,
    ) -> _CapturedFile:
        if (
            isinstance(chunk_size, bool)
            or not isinstance(chunk_size, int)
            or chunk_size <= 0
        ):
            raise ValidationError("capture chunk_size must be a positive integer")
        if isinstance(max_bytes, bool) or (
            max_bytes is not None and (not isinstance(max_bytes, int) or max_bytes < 0)
        ):
            raise ValidationError(
                "capture max_bytes must be a non-negative integer or null"
            )
        source = Path(os.path.abspath(os.fspath(path)))
        _reject_link_or_reparse_components(source, label=label)
        relative_parts: tuple[str, ...]
        if destination_relative is None:
            relative_parts = (f"capture-{self._counter:06d}",)
            self._counter += 1
        else:
            if destination_relative.is_absolute() or any(
                part in ("", ".", "..") for part in destination_relative.parts
            ):
                raise ValidationError("private capture path is noncanonical")
            relative_parts = tuple(destination_relative.parts)
        destination = self.root.joinpath(*relative_parts)
        digest = hashlib.sha256()
        byte_count = 0
        destination_parent = self._capture_parent(relative_parts[:-1])
        created_identity: tuple[int, int] | None = None
        try:
            with _open_regular_source_once(source, label=label) as source_handle:
                with _open_pinned_exclusive_file(
                    destination_parent,
                    destination.name,
                    label="private capture file",
                ) as destination_handle:
                    created_metadata = os.fstat(destination_handle.fileno())
                    created_identity = (
                        int(created_metadata.st_dev),
                        int(created_metadata.st_ino),
                    )
                    while True:
                        read_size = chunk_size
                        if max_bytes is not None:
                            remaining = max_bytes - byte_count
                            read_size = min(chunk_size, remaining + 1)
                        chunk = source_handle.read(read_size)
                        if not chunk:
                            break
                        if max_bytes is not None and len(chunk) > remaining:
                            raise ValidationError(
                                f"{label} exceeds capture byte budget of "
                                f"{max_bytes} bytes"
                            )
                        destination_handle.write(chunk)
                        digest.update(chunk)
                        byte_count += len(chunk)
                        if _payload_sink is not None:
                            _payload_sink.extend(chunk)
                    destination_handle.flush()
                    os.fsync(destination_handle.fileno())
        except BaseException as exc:
            try:
                if created_identity is not None:
                    _delete_pinned_file(
                        destination_parent,
                        destination.name,
                        created_identity,
                        label="partial private capture file",
                    )
            except Exception as cleanup_error:
                exc.add_note(
                    "private capture partial-file cleanup also failed: "
                    f"{cleanup_error}"
                )
            raise
        if created_identity is None:
            raise ValidationError("private capture file identity was not recorded")
        self._file_identities[relative_parts] = created_identity
        captured = _CapturedFile(
            source_path=source,
            path=destination,
            bytes=byte_count,
            sha256=digest.hexdigest(),
        )
        if expected_record is not None:
            expected_bytes = expected_record.get("bytes")
            expected_sha256 = expected_record.get("sha256")
            if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
                raise ValidationError(f"{label}.bytes is invalid")
            if not _valid_sha256(expected_sha256):
                raise ValidationError(f"{label}.sha256 is invalid")
            if captured.bytes != expected_bytes:
                raise ValidationError(f"{label} byte count changed: {source}")
            if captured.sha256 != expected_sha256:
                raise ValidationError(f"{label} SHA-256 changed: {source}")
        return captured

    def capture_bytes(
        self,
        path: Path,
        *,
        label: str,
        expected_record: Mapping[str, Any] | None = None,
        logical_path: str | Path | None = None,
        destination_relative: Path | None = None,
        max_bytes: int | None = None,
    ) -> tuple[bytes, _CapturedFile]:
        payload = bytearray()
        captured = self.capture(
            path,
            label=label,
            expected_record=expected_record,
            logical_path=logical_path,
            destination_relative=destination_relative,
            max_bytes=max_bytes,
            _payload_sink=payload,
        )
        return bytes(payload), captured


def _relocated_child(root: Path, relative: Path, *, label: str) -> Path:
    if relative.is_absolute() or any(
        part in ("", ".", "..") for part in relative.parts
    ):
        raise ValidationError(f"{label} path is not a canonical relative path")
    resolved_root = root.resolve()
    cursor = resolved_root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink() or _has_reparse_point(cursor):
            raise ValidationError(f"{label} path contains a symlink or reparse point")
    candidate = (resolved_root / relative).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValidationError(f"{label} path escapes the workspace root") from exc
    return candidate


def _resolve_bound_path(
    value: Any,
    *,
    label: str,
    workspace_root: Path | None = None,
    recorded_root: Path | None = None,
) -> Path:
    """Resolve a bound path, optionally remapping one explicit legacy root."""
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{label} path must be a non-empty string")
    current_root = (ROOT if workspace_root is None else workspace_root).resolve()
    raw_path = Path(value)
    if not raw_path.is_absolute():
        return _relocated_child(current_root, raw_path, label=label)
    if recorded_root is None:
        absolute = Path(os.path.abspath(os.fspath(raw_path)))
        _reject_link_or_reparse_components(absolute, label=label)
        return absolute
    old_root = recorded_root.absolute()
    if not old_root.is_absolute():
        raise ValidationError("recorded root must be absolute")
    try:
        relative = raw_path.relative_to(old_root)
    except ValueError as exc:
        raise ValidationError(f"{label} path is outside the recorded root") from exc
    return _relocated_child(current_root, relative, label=label)


def _resolve_campaign_run_path(
    value: Any,
    *,
    workspace_root: Path | None = None,
    recorded_root: Path | None = None,
) -> Path:
    """Resolve a campaign run path against the workspace, never process CWD."""
    return _resolve_bound_path(
        value,
        label="campaign source run",
        workspace_root=workspace_root,
        recorded_root=recorded_root,
    )


def _verify_file_record(path: Path, record: Mapping[str, Any], *, label: str) -> None:
    if not path.is_file():
        raise ValidationError(f"{label} is missing: {path}")
    expected_bytes = record.get("bytes")
    expected_sha = record.get("sha256")
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
        raise ValidationError(f"{label}.bytes is invalid")
    if not _valid_sha256(expected_sha):
        raise ValidationError(f"{label}.sha256 is invalid")
    if path.stat().st_size != expected_bytes:
        raise ValidationError(f"{label} byte count changed: {path}")
    if _sha256_file(path) != expected_sha:
        raise ValidationError(f"{label} SHA-256 changed: {path}")


def _unique_lines_from_bytes(payload: bytes, *, label: str) -> tuple[str, ...]:
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise ValidationError(f"identifier list is not valid UTF-8: {label}") from exc
    values = tuple(line.strip() for line in text.splitlines())
    if not values or any(not value for value in values):
        raise ValidationError(f"identifier list is empty or contains blanks: {label}")
    if len(values) != len(set(values)):
        raise ValidationError(f"identifier list contains duplicates: {label}")
    return values


def _read_unique_lines(path: Path) -> tuple[str, ...]:
    with _CaptureStore(prefix="depplm-identifiers-") as store:
        payload, _captured = store.capture_bytes(path, label="identifier list")
        return _unique_lines_from_bytes(payload, label=str(path))


def _iter_ids(value: Any) -> Iterable[str]:
    if value is None:
        return ()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return ()
    identifiers: list[str] = []
    for item in value:
        if isinstance(item, Mapping):
            item = item.get("id", item.get("diseaseId"))
        if item is not None and str(item).strip():
            identifiers.append(str(item).strip())
    return identifiers


def classify_stage(stage: Any) -> str:
    """Classify a row-level maximum without implying current activity."""
    normalized = str(stage).strip().upper() if stage is not None else ""
    if normalized in PHASE2_PLUS_STAGES:
        return "phase2_plus"
    if normalized == WITHDRAWAL_STAGE:
        return "withdrawal_sensitivity"
    return "excluded_stage"


def _complete_cancer_only_disease_ids(
    value: Any, cancer_ids: set[str]
) -> tuple[str, ...]:
    """Return disease ids only when every listed indication is mapped cancer."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not value:
        return ()
    identifiers: list[str] = []
    for item in value:
        if not isinstance(item, Mapping):
            return ()
        identifier = item.get("diseaseId", item.get("id"))
        if identifier is None or not str(identifier).strip():
            return ()
        identifiers.append(str(identifier).strip())
    if not identifiers or any(
        identifier not in cancer_ids for identifier in identifiers
    ):
        return ()
    return tuple(sorted(set(identifiers)))


def cancer_disease_scope(
    disease_rows: Sequence[Mapping[str, Any]], root: str = CANCER_ROOT
) -> tuple[set[str], list[dict[str, str]]]:
    """Return root plus descendants, using both root and child ancestor records."""
    root_rows = [row for row in disease_rows if str(row.get("id", "")) == root]
    if len(root_rows) != 1:
        raise ValidationError(
            f"expected exactly one disease row for cancer root {root}, found {len(root_rows)}"
        )
    root_row = root_rows[0]
    identifiers = {root, *set(_iter_ids(root_row.get("descendants")))}
    for row in disease_rows:
        identifier = str(row.get("id", "")).strip()
        if identifier and root in set(_iter_ids(row.get("ancestors"))):
            identifiers.add(identifier)
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in disease_rows:
        identifier = str(row.get("id", "")).strip()
        if identifier in by_id:
            raise ValidationError(f"duplicate disease identifier: {identifier}")
        if identifier:
            by_id[identifier] = row
    missing = identifiers - set(by_id)
    if missing:
        raise ValidationError(
            f"cancer descendants are absent from the disease table: {sorted(missing)[:5]}"
        )
    scope = []
    direct_descendants = set(_iter_ids(root_row.get("descendants")))
    for identifier in sorted(identifiers):
        row = by_id[identifier]
        relation = "root" if identifier == root else "descendant"
        if identifier != root and identifier not in direct_descendants:
            relation = "ancestor_relation"
        scope.append(
            {
                "disease_id": identifier,
                "name": str(row.get("name") or ""),
                "scope_relation": relation,
            }
        )
    return identifiers, scope


def build_snapshot_rows(
    target_rows: Sequence[Mapping[str, Any]],
    disease_rows: Sequence[Mapping[str, Any]],
    clinical_rows: Sequence[Mapping[str, Any]],
    tractability_rows: Sequence[Mapping[str, Any]],
) -> tuple[
    list[dict[str, Any]], list[dict[str, str]], list[dict[str, str]], dict[str, Any]
]:
    """Derive exact-symbol outcomes from already decoded Open Targets rows."""
    cancer_ids, disease_scope = cancer_disease_scope(disease_rows)

    symbol_to_ids: dict[str, set[str]] = defaultdict(set)
    id_to_symbols: dict[str, set[str]] = defaultdict(set)
    for row in target_rows:
        identifier = str(row.get("id") or "").strip()
        symbol = str(row.get("approvedSymbol") or "").strip()
        if identifier and symbol:
            symbol_to_ids[symbol].add(identifier)
            id_to_symbols[identifier].add(symbol)
    unambiguous = {
        symbol: next(iter(identifiers))
        for symbol, identifiers in symbol_to_ids.items()
        if len(identifiers) == 1 and len(id_to_symbols[next(iter(identifiers))]) == 1
    }
    ambiguous = []
    for symbol, identifiers in sorted(symbol_to_ids.items()):
        if len(identifiers) != 1 or any(
            len(id_to_symbols[identifier]) != 1 for identifier in identifiers
        ):
            ambiguous.append(
                {
                    "approved_symbol": symbol,
                    "target_ids": ";".join(sorted(identifiers)),
                    "reason": "symbol_or_target_not_one_to_one",
                }
            )

    phase2_plus_stages: dict[str, set[str]] = defaultdict(set)
    phase2_plus_drugs: dict[str, set[str]] = defaultdict(set)
    withdrawal_drugs: dict[str, set[str]] = defaultdict(set)
    rows_with_any_cancer_disease = 0
    cancer_only_rows = 0
    mixed_cancer_noncancer_rows = 0
    cancer_rows_with_unmapped_disease = 0
    for row in clinical_rows:
        target_id = str(row.get("targetId") or "").strip()
        if not target_id:
            continue
        indications = set(_iter_ids(row.get("diseases")))
        if not indications.intersection(cancer_ids):
            continue
        rows_with_any_cancer_disease += 1
        raw_diseases = row.get("diseases")
        if isinstance(raw_diseases, np.ndarray):
            raw_diseases = raw_diseases.tolist()
        if not isinstance(raw_diseases, (list, tuple)) or any(
            not isinstance(item, Mapping)
            or not str(item.get("diseaseId", item.get("id")) or "").strip()
            for item in raw_diseases
        ):
            cancer_rows_with_unmapped_disease += 1
            continue
        if any(identifier not in cancer_ids for identifier in indications):
            mixed_cancer_noncancer_rows += 1
            continue
        cancer_only_indications = _complete_cancer_only_disease_ids(
            raw_diseases, cancer_ids
        )
        if not cancer_only_indications:
            raise ValidationError(
                "internal cancer-only indication classification is inconsistent"
            )
        cancer_only_rows += 1
        stage = str(row.get("maxClinicalStage") or "").strip().upper()
        drug_id = str(row.get("drugId") or "").strip()
        classification = classify_stage(stage)
        if classification == "phase2_plus":
            phase2_plus_stages[target_id].add(stage)
            if drug_id:
                phase2_plus_drugs[target_id].add(drug_id)
        elif classification == "withdrawal_sensitivity" and drug_id:
            withdrawal_drugs[target_id].add(drug_id)

    true_modalities: dict[str, set[str]] = defaultdict(set)
    true_record_counts: Counter[str] = Counter()
    for row in tractability_rows:
        if row.get("value") is not True:
            continue
        target_id = str(row.get("targetId") or "").strip()
        modality = str(row.get("modality") or "").strip()
        if not target_id:
            continue
        true_record_counts[target_id] += 1
        if modality:
            true_modalities[target_id].add(modality)

    outcomes = []
    for symbol, target_id in sorted(unambiguous.items()):
        stages = phase2_plus_stages.get(target_id, set())
        max_stage = max(stages, key=STAGE_ORDER.__getitem__) if stages else ""
        modalities = true_modalities.get(target_id, set())
        outcomes.append(
            {
                "approved_symbol": symbol,
                "target_id": target_id,
                "clinical_oncology_phase2_plus": int(bool(stages)),
                "oncology_phase2_plus_drug_count": len(
                    phase2_plus_drugs.get(target_id, set())
                ),
                "oncology_phase2_plus_stages": ";".join(
                    sorted(stages, key=STAGE_ORDER.__getitem__)
                ),
                "maximum_oncology_phase2_plus_stage": max_stage,
                "clinical_oncology_withdrawal": int(
                    bool(withdrawal_drugs.get(target_id))
                ),
                "oncology_withdrawn_drug_count": len(
                    withdrawal_drugs.get(target_id, set())
                ),
                "tractable_any": int(bool(true_record_counts[target_id])),
                "tractability_true_record_count": int(true_record_counts[target_id]),
                "tractability_true_modalities": ";".join(sorted(modalities)),
            }
        )
    metadata = {
        "target_rows": len(target_rows),
        "disease_rows": len(disease_rows),
        "clinical_target_rows": len(clinical_rows),
        "tractability_rows": len(tractability_rows),
        "cancer_disease_count": len(cancer_ids),
        "rows_with_any_cancer_disease": rows_with_any_cancer_disease,
        "cancer_only_clinical_rows": cancer_only_rows,
        "mixed_cancer_noncancer_rows_excluded": mixed_cancer_noncancer_rows,
        "cancer_rows_with_unmapped_disease_excluded": (
            cancer_rows_with_unmapped_disease
        ),
        "symbols_observed": len(symbol_to_ids),
        "one_to_one_symbols": len(unambiguous),
        "ambiguous_symbols_excluded": len(ambiguous),
        "clinical_oncology_phase2_plus_targets": sum(
            cast(int, row["clinical_oncology_phase2_plus"]) for row in outcomes
        ),
        "clinical_oncology_withdrawal_targets": sum(
            cast(int, row["clinical_oncology_withdrawal"]) for row in outcomes
        ),
        "tractable_targets": sum(cast(int, row["tractable_any"]) for row in outcomes),
    }
    return outcomes, disease_scope, ambiguous, metadata


def _tsv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=list(fields), delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return stream.getvalue().encode("utf-8")


def _read_parquet_rows(
    paths: Sequence[Path], columns: Sequence[str]
) -> list[dict[str, Any]]:
    try:
        import pyarrow.dataset as ds
    except ImportError as exc:
        raise ValidationError(
            "snapshot derivation and verification require pyarrow"
        ) from exc
    try:
        table = ds.dataset([str(path) for path in paths], format="parquet").to_table(
            columns=list(columns)
        )
    except Exception as exc:
        raise ValidationError(f"cannot read fixed Parquet inputs: {exc}") from exc
    return table.to_pylist()


def validate_raw_release(
    raw_root: Path, specs: Sequence[RawFileSpec] = RAW_FILES
) -> list[dict[str, Any]]:
    records = []
    for spec in specs:
        path = _safe_child(raw_root.resolve(), spec.path)
        record = {"path": spec.path, "bytes": spec.bytes, "sha256": spec.sha256}
        _verify_file_record(path, record, label=f"Open Targets {RELEASE} raw file")
        records.append(record)
    observed = {
        path.relative_to(raw_root).as_posix()
        for path in raw_root.rglob("*.parquet")
        if path.is_file()
    }
    expected = {spec.path for spec in specs}
    if observed != expected:
        raise ValidationError(
            "Open Targets raw Parquet inventory differs from the fixed seven files: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    return records


def _derive_snapshot_from_raw(raw_root: Path, specs: Sequence[RawFileSpec]) -> tuple[
    list[dict[str, Any]],
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, Any],
]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for spec in specs:
        grouped[spec.path.split("/", 1)[0]].append(raw_root / spec.path)
    required_groups = {"target", "clinical_target", "disease", "target_tractability"}
    if set(grouped) != required_groups:
        raise ValidationError(
            f"raw specification groups must be {sorted(required_groups)}, found {sorted(grouped)}"
        )
    target_rows = _read_parquet_rows(grouped["target"], ("id", "approvedSymbol"))
    disease_rows = _read_parquet_rows(
        grouped["disease"], ("id", "name", "ancestors", "descendants")
    )
    clinical_rows = _read_parquet_rows(
        grouped["clinical_target"],
        ("targetId", "drugId", "diseases", "maxClinicalStage"),
    )
    tractability_rows = _read_parquet_rows(
        grouped["target_tractability"],
        ("targetId", "modality", "category", "value"),
    )
    return build_snapshot_rows(
        target_rows, disease_rows, clinical_rows, tractability_rows
    )


def _legacy_freeze_snapshot(
    raw_root: Path = DEFAULT_RAW_ROOT,
    output_dir: Path = DEFAULT_SNAPSHOT_DIR,
    *,
    specs: Sequence[RawFileSpec] = RAW_FILES,
) -> Path:
    """Create a versioned, hash-bound TSV snapshot from fixed Parquet bytes."""
    raw_root = raw_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ValidationError(f"refusing to overwrite snapshot directory: {output_dir}")
    raw_records = validate_raw_release(raw_root, specs)
    outcomes, disease_scope, ambiguous, counts = _derive_snapshot_from_raw(
        raw_root, specs
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(exist_ok=False)
    try:
        _atomic_bytes(
            staging / "target_outcomes.tsv", _tsv_bytes(outcomes, OUTCOME_FIELDS)
        )
        _atomic_bytes(
            staging / "disease_scope.tsv", _tsv_bytes(disease_scope, DISEASE_FIELDS)
        )
        _atomic_bytes(
            staging / "ambiguous_symbols.tsv", _tsv_bytes(ambiguous, AMBIGUOUS_FIELDS)
        )
        artifacts = [
            _file_record(staging / name, relative_to=staging)
            for name in (
                "target_outcomes.tsv",
                "disease_scope.tsv",
                "ambiguous_symbols.tsv",
            )
        ]
        manifest = {
            "schema": SNAPSHOT_SCHEMA,
            "release": RELEASE,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "scientific_result": False,
            "scientific_result_reason": (
                "External-data snapshot only; no DepPLM prediction was analysed."
            ),
            "cancer_scope": {
                "root": CANCER_ROOT,
                "rule": "root plus descendants and rows whose ancestors contain the root",
            },
            "clinical_endpoint": {
                "source_table": "clinical_target",
                "cancer_only_complete_disease_list_required": True,
                "unmapped_or_mixed_disease_rows_excluded": True,
                "phase2_plus_stages": list(PHASE2_PLUS_STAGES),
                "maxClinicalStage_interpretation": (
                    "historical row-level maximum; not current activity"
                ),
                "withdrawal_stage": WITHDRAWAL_STAGE,
                "withdrawal_role": "separate_sensitivity_only",
                "associatedTargets_used": False,
            },
            "tractability_endpoint": {
                "rule": "target has at least one target_tractability record with value exactly true",
                "false_or_null_records_count_as_positive": False,
            },
            "target_mapping": {
                "rule": "case-sensitive exact approvedSymbol with one symbol to one target id",
                "ambiguous_symbols_excluded": True,
            },
            "raw_files": raw_records,
            "artifacts": artifacts,
            "counts": counts,
            "generator": {
                **_file_record(Path(__file__)),
                "python": platform.python_version(),
                "pyarrow": __import__("pyarrow").__version__,
            },
        }
        _atomic_json(staging / "snapshot-manifest.json", manifest)
        os.replace(staging, output_dir)
    except Exception:
        for path in sorted(staging.glob("*"), reverse=True):
            path.unlink(missing_ok=True)
        staging.rmdir()
        raise
    return output_dir / "snapshot-manifest.json"


def _legacy_validate_snapshot(
    snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR,
    *,
    specs: Sequence[RawFileSpec] = RAW_FILES,
    raw_root: Path | None = None,
) -> dict[str, Any]:
    snapshot_dir = snapshot_dir.resolve()
    manifest_path = snapshot_dir / "snapshot-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError("snapshot manifest is not valid UTF-8 JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != SNAPSHOT_SCHEMA:
        raise ValidationError("snapshot manifest schema is invalid")
    if (
        manifest.get("release") != RELEASE
        or manifest.get("scientific_result") is not False
    ):
        raise ValidationError("snapshot release/result boundary is invalid")
    endpoint = manifest.get("clinical_endpoint")
    expected_endpoint = {
        "source_table": "clinical_target",
        "cancer_only_complete_disease_list_required": True,
        "unmapped_or_mixed_disease_rows_excluded": True,
        "phase2_plus_stages": list(PHASE2_PLUS_STAGES),
        "maxClinicalStage_interpretation": (
            "historical row-level maximum; not current activity"
        ),
        "withdrawal_stage": WITHDRAWAL_STAGE,
        "withdrawal_role": "separate_sensitivity_only",
        "associatedTargets_used": False,
    }
    if endpoint != expected_endpoint:
        raise ValidationError(
            "snapshot clinical endpoint differs from the fixed protocol"
        )
    raw_records = manifest.get("raw_files")
    expected_raw_records = [
        {"path": spec.path, "bytes": spec.bytes, "sha256": spec.sha256}
        for spec in specs
    ]
    if raw_records != expected_raw_records:
        raise ValidationError(
            "snapshot raw-file bindings differ from the fixed release"
        )
    bound_raw_root = (
        raw_root.resolve()
        if raw_root is not None
        else (snapshot_dir.parent / "raw").resolve()
    )
    validate_raw_release(bound_raw_root, specs)
    outcomes, disease_scope, ambiguous, counts = _derive_snapshot_from_raw(
        bound_raw_root, specs
    )
    if manifest.get("counts") != counts:
        raise ValidationError("snapshot counts differ from raw-data rederivation")
    expected_payloads = {
        "target_outcomes.tsv": _tsv_bytes(outcomes, OUTCOME_FIELDS),
        "disease_scope.tsv": _tsv_bytes(disease_scope, DISEASE_FIELDS),
        "ambiguous_symbols.tsv": _tsv_bytes(ambiguous, AMBIGUOUS_FIELDS),
    }
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 3:
        raise ValidationError("snapshot artifact inventory is invalid")
    expected_names = {
        "target_outcomes.tsv",
        "disease_scope.tsv",
        "ambiguous_symbols.tsv",
    }
    names = {record.get("path") for record in artifacts if isinstance(record, dict)}
    if names != expected_names:
        raise ValidationError("snapshot artifact names are incomplete or duplicated")
    for record in artifacts:
        path = _safe_child(snapshot_dir, record["path"])
        _verify_file_record(path, record, label="snapshot artifact")
        if path.read_bytes() != expected_payloads[record["path"]]:
            raise ValidationError(
                f"snapshot artifact differs from raw-data rederivation: {record['path']}"
            )
    generator = manifest.get("generator")
    if not isinstance(generator, dict):
        raise ValidationError("snapshot generator binding is missing")
    _verify_file_record(Path(__file__), generator, label="snapshot generator source")
    if not isinstance(generator.get("python"), str) or not isinstance(
        generator.get("pyarrow"), str
    ):
        raise ValidationError("snapshot generator environment is incomplete")
    return {
        "manifest": manifest,
        "manifest_record": _file_record(manifest_path),
        "snapshot_dir": str(snapshot_dir),
        "raw_files_rehashed": len(expected_raw_records),
        "artifacts_rederived": len(expected_payloads),
    }


def preflight_snapshot_freeze(
    raw_root: Path = DEFAULT_RAW_ROOT,
    output_dir: Path = DEFAULT_SNAPSHOT_BUILD_DIR,
    *,
    specs: Sequence[RawFileSpec] = RAW_FILES,
) -> dict[str, Any]:
    """Rehash raw inputs and report the exact future v4 freeze command."""
    try:
        return snapshot_builder_v4.preflight_freeze(
            raw_root,
            output_dir,
            specs=cast(Sequence[snapshot_builder_v4.RawFileSpec], specs),
        )
    except snapshot_builder_v4.SnapshotValidationError as exc:
        raise ValidationError(str(exc)) from exc


def freeze_snapshot(
    raw_root: Path = DEFAULT_RAW_ROOT,
    output_dir: Path = DEFAULT_SNAPSHOT_BUILD_DIR,
    *,
    specs: Sequence[RawFileSpec] = RAW_FILES,
) -> Path:
    """Create a v4 snapshot and return its point-in-time manifest path.

    Call :func:`validate_snapshot` again immediately before consuming the
    returned path; a plain ``Path`` does not retain the published identity.
    """
    try:
        return snapshot_builder_v4.freeze_snapshot(
            raw_root,
            output_dir,
            specs=cast(Sequence[snapshot_builder_v4.RawFileSpec], specs),
        )
    except snapshot_builder_v4.SnapshotValidationError as exc:
        raise ValidationError(str(exc)) from exc


def validate_snapshot(
    snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR,
    *,
    specs: Sequence[RawFileSpec] = RAW_FILES,
    raw_root: Path | None = None,
) -> dict[str, Any]:
    """Rederive the bytes currently visible for a registered snapshot.

    The result is a point-in-time verification, not a continuing pathname
    ownership guarantee against a same-UID replacement after this call.
    """
    try:
        validated = snapshot_builder_v4.validate_snapshot(
            snapshot_dir,
            specs=cast(Sequence[snapshot_builder_v4.RawFileSpec], specs),
            raw_root=raw_root,
        )
    except snapshot_builder_v4.SnapshotValidationError as exc:
        raise ValidationError(str(exc)) from exc
    manifest_record = validated.get("manifest_record")
    if not isinstance(manifest_record, Mapping):
        raise ValidationError("snapshot validation did not return a manifest record")
    return {
        **validated,
        "verification_scope": "point_in_time_hash_verification",
        "path_identity_retained": False,
        "marker_record": dict(manifest_record),
    }


def _runtime_fingerprint_sha256(run_manifest: Mapping[str, Any]) -> str:
    device = run_manifest.get("device")
    runtime = run_manifest.get("runtime")
    versions = run_manifest.get("versions")
    if device not in {"cpu", "cuda"} or not isinstance(runtime, dict):
        raise ValidationError("run manifest lacks a valid device/runtime fingerprint")
    if not isinstance(versions, dict):
        raise ValidationError("run manifest lacks package versions")
    version_names = ("numpy", "pandas", "scipy", "scikit_learn", "torch")
    if any(
        not isinstance(versions.get(name), str) or not versions[name]
        for name in version_names
    ):
        raise ValidationError("run manifest package fingerprint is incomplete")
    amp_enabled = runtime.get("amp_enabled")
    cuda_available = runtime.get("cuda_available")
    if not isinstance(amp_enabled, bool) or not isinstance(cuda_available, bool):
        raise ValidationError("run manifest runtime booleans are invalid")
    expected_amp = device == "cuda"
    if amp_enabled is not expected_amp or runtime.get("amp_precision") != (
        "bfloat16" if expected_amp else None
    ):
        raise ValidationError("run manifest AMP policy is inconsistent")
    capability = runtime.get("cuda_device_capability")
    if device == "cpu":
        if capability is not None or runtime.get("cuda_device_name") is not None:
            raise ValidationError("CPU run contains a CUDA device identity")
    elif (
        cuda_available is not True
        or not isinstance(runtime.get("torch_cuda_version"), str)
        or not runtime.get("torch_cuda_version")
        or isinstance(runtime.get("cudnn_version"), bool)
        or not isinstance(runtime.get("cudnn_version"), int)
        or not isinstance(runtime.get("cuda_device_name"), str)
        or not runtime.get("cuda_device_name")
        or not isinstance(capability, list)
        or len(capability) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in capability
        )
    ):
        raise ValidationError("CUDA run has incomplete accelerator identity")
    if any(
        not isinstance(run_manifest.get(name), str) or not run_manifest[name]
        for name in ("python", "platform")
    ):
        raise ValidationError("run manifest python/platform fingerprint is incomplete")
    fingerprint = {
        "device_type": device,
        "amp_enabled": amp_enabled,
        "amp_precision": runtime.get("amp_precision"),
        "cuda_available": cuda_available,
        "torch_cuda_version": runtime.get("torch_cuda_version"),
        "cudnn_version": runtime.get("cudnn_version"),
        "cuda_device_name": runtime.get("cuda_device_name"),
        "cuda_device_capability": capability,
        "python": run_manifest["python"],
        "platform": run_manifest["platform"],
        "versions": {name: versions[name] for name in version_names},
    }
    canonical = json.dumps(
        fingerprint, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _expected_campaign_keys() -> set[str]:
    return {
        f"{protocol}|{condition}|seed-{seed}"
        for protocol in CANONICAL_PROTOCOLS
        for condition in CANONICAL_CONDITIONS
        for seed in CANONICAL_SEEDS
    }


def _expected_campaign_artifacts(protocol: str) -> tuple[str, ...]:
    if protocol == "p1":
        return tuple(
            f"fold-{fold:02d}-{suffix}"
            for fold in range(5)
            for suffix in ("predictions.npz", "result.json", "split.json")
        )
    direction = {
        "transfer-forward": "complement-to-bkm",
        "transfer-reverse": "bkm-to-complement",
    }.get(protocol)
    if direction is None:
        raise ValidationError(f"noncanonical campaign protocol: {protocol}")
    return tuple(
        f"{direction}-{suffix}"
        for suffix in ("predictions.npz", "result.json", "split.json")
    )


def _campaign_results_root(run_paths: Sequence[Path]) -> Path:
    if not run_paths:
        raise ValidationError("campaign source run paths are empty")
    parents = {Path(os.path.abspath(os.fspath(path))).parent for path in run_paths}
    if len(parents) != 1:
        raise ValidationError("campaign source runs do not share one results root")
    root = next(iter(parents))
    _reject_link_or_reparse_components(root, label="campaign results root")
    return root


CAMPAIGN_REPLAY_PROBE_SCHEMA = "DepPLM-isolated-campaign-replay-probe-v1"
CAMPAIGN_REPLAY_REQUEST_SCHEMA = "DepPLM-isolated-campaign-replay-request-v4"
CAMPAIGN_REPLAY_RESULT_SCHEMA = "DepPLM-isolated-campaign-replay-result-v4"
CAMPAIGN_REPLAY_TIMEOUT_SECONDS = 60 * 60
CAMPAIGN_REPLAY_PROBE_TIMEOUT_SECONDS = 30
CAMPAIGN_REPLAY_PROBE_MAX_BYTES = 1024 * 1024
CAMPAIGN_REPLAY_RESULT_MAX_BYTES = 128 * 1024 * 1024
CAMPAIGN_REPLAY_STDOUT_MAX_BYTES = 64 * 1024
CAMPAIGN_REPLAY_STDERR_MAX_BYTES = 256 * 1024
CAMPAIGN_REPLAY_DEPENDENCY_MAX_BYTES = 2 * 1024 * 1024 * 1024
CAMPAIGN_REPLAY_DEPENDENCY_MAX_FILES = 10_000
CAMPAIGN_REPLAY_OUTPUT_MAX_BYTES = CAMPAIGN_REPLAY_RESULT_MAX_BYTES
CAMPAIGN_REPLAY_FLAGS = ("-I", "-S", "-B", "-c")
CAMPAIGN_REPLAY_PROBE_BOOTSTRAP = r"""
import json
import math
import os
import site
import sys
import sysconfig

SCHEMA = "DepPLM-isolated-campaign-replay-probe-v1"


def fail(message):
    raise RuntimeError(message)


def canonical_json_bytes(value):
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


def normalized_path(value):
    return os.path.normcase(os.path.realpath(os.path.abspath(value)))


def normalized_paths(values):
    result = []
    for value in values:
        if not isinstance(value, str) or not value:
            fail("isolated replay probe sys.path contains an invalid entry")
        normalized = normalized_path(value)
        if normalized in result:
            fail("isolated replay probe sys.path contains a duplicate entry")
        result.append(normalized)
    return result


def normalized_site_paths():
    candidates = []
    try:
        candidates.extend(site.getsitepackages())
    except AttributeError:
        pass
    for key in ("purelib", "platlib"):
        candidate = sysconfig.get_path(key)
        if candidate:
            candidates.append(candidate)
    executable = os.path.abspath(sys.executable)
    for root in (os.path.dirname(executable), os.path.dirname(os.path.dirname(executable))):
        if not os.path.isfile(os.path.join(root, "pyvenv.cfg")):
            continue
        candidates.extend(
            (
                os.path.join(root, "Lib", "site-packages"),
                os.path.join(
                    root,
                    "lib",
                    "python%d.%d" % sys.version_info[:2],
                    "site-packages",
                ),
            )
        )
    result = []
    for candidate in candidates:
        normalized = normalized_path(candidate)
        lowered = normalized.replace("\\", "/").lower()
        if (
            os.path.isdir(normalized)
            and ("/site-packages" in lowered or "/dist-packages" in lowered)
            and normalized not in result
        ):
            result.append(normalized)
    return result


def normalized_environment(values):
    return {
        (key.upper() if os.name == "nt" else key): value
        for key, value in sorted(values.items())
    }


if len(sys.argv) != 2:
    fail("isolated replay probe requires one output path")
output_path = os.path.abspath(sys.argv[1])
flags = {
    "dont_write_bytecode": bool(sys.flags.dont_write_bytecode),
    "ignore_environment": bool(sys.flags.ignore_environment),
    "isolated": bool(sys.flags.isolated),
    "no_site": bool(sys.flags.no_site),
    "no_user_site": bool(sys.flags.no_user_site),
    "safe_path": bool(sys.flags.safe_path),
}
if flags != {
    "dont_write_bytecode": True,
    "ignore_environment": True,
    "isolated": True,
    "no_site": True,
    "no_user_site": True,
    "safe_path": True,
}:
    fail("isolated replay probe flags are invalid")
result = {
    "schema": SCHEMA,
    "runtime": {
        "executable": normalized_path(sys.executable),
        "implementation": sys.implementation.name,
        "version": list(sys.version_info[:3]),
        "prefix": normalized_path(sys.prefix),
        "base_prefix": normalized_path(sys.base_prefix),
        "exec_prefix": normalized_path(sys.exec_prefix),
        "base_exec_prefix": normalized_path(sys.base_exec_prefix),
    },
    "flags": flags,
    "environment": normalized_environment(dict(os.environ)),
    "startup_sys_path": normalized_paths(sys.path),
    "site_paths": normalized_site_paths(),
}
payload = canonical_json_bytes(result)
descriptor = os.open(
    output_path,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
    0o600,
)
with os.fdopen(descriptor, "wb") as handle:
    handle.write(payload)
    handle.flush()
    os.fsync(handle.fileno())
"""
CAMPAIGN_REPLAY_BOOTSTRAP = r"""
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import math
import os
import site
import stat
import sys
import sysconfig
from pathlib import Path

REQUEST_SCHEMA = "DepPLM-isolated-campaign-replay-request-v4"
RESULT_SCHEMA = "DepPLM-isolated-campaign-replay-result-v4"
REQUEST_FIELDS = {
    "schema",
    "workspace_root",
    "results_root",
    "dependency_root",
    "interpreter",
    "command",
    "environment",
    "startup_sys_path",
    "effective_sys_path",
    "site_paths",
    "code_records",
    "common_input_paths",
    "transfer_input_path",
    "code_paths",
    "executed_modules",
    "limits",
}


def fail(message):
    raise RuntimeError(message)


def canonical_json_bytes(value):
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


def strict_json_bytes(payload, label):
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                fail(label + " contains a duplicate key")
            result[key] = value
        return result

    def reject_constant(value):
        fail(label + " contains a non-finite constant")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(label + " is invalid JSON") from exc

    def require_finite(item):
        if isinstance(item, float) and not math.isfinite(item):
            fail(label + " contains a non-finite number")
        if isinstance(item, dict):
            for nested in item.values():
                require_finite(nested)
        elif isinstance(item, list):
            for nested in item:
                require_finite(nested)

    require_finite(value)
    return value


def normalized_path(value):
    return os.path.normcase(os.path.realpath(os.path.abspath(value)))


def normalized_site_paths():
    candidates = []
    try:
        candidates.extend(site.getsitepackages())
    except AttributeError:
        pass
    for key in ("purelib", "platlib"):
        candidate = sysconfig.get_path(key)
        if candidate:
            candidates.append(candidate)
    executable = os.path.abspath(sys.executable)
    for root in (os.path.dirname(executable), os.path.dirname(os.path.dirname(executable))):
        if not os.path.isfile(os.path.join(root, "pyvenv.cfg")):
            continue
        candidates.extend(
            (
                os.path.join(root, "Lib", "site-packages"),
                os.path.join(
                    root,
                    "lib",
                    "python%d.%d" % sys.version_info[:2],
                    "site-packages",
                ),
            )
        )
    result = []
    for candidate in candidates:
        normalized = normalized_path(candidate)
        lowered = normalized.replace("\\", "/").lower()
        if (
            os.path.isdir(normalized)
            and ("/site-packages" in lowered or "/dist-packages" in lowered)
            and normalized not in result
        ):
            result.append(normalized)
    return result


def normalized_environment(values):
    return {
        (key.upper() if os.name == "nt" else key): value
        for key, value in sorted(values.items())
    }


def normalized_paths(values):
    result = []
    for value in values:
        if not isinstance(value, str) or not value:
            fail("isolated replay sys.path contains an invalid entry")
        normalized = normalized_path(value)
        if normalized in result:
            fail("isolated replay sys.path contains a duplicate entry")
        result.append(normalized)
    return result


def is_within(path, root):
    try:
        return os.path.commonpath((path, root)) == root
    except ValueError:
        return False


def canonical_role(value):
    if not isinstance(value, str) or not value or "\\" in value:
        fail("code role is noncanonical")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts) or ":" in parts[0]:
        fail("code role is noncanonical")
    return parts


def file_record(workspace_root, role, expected):
    if not isinstance(expected, dict) or set(expected) != {"bytes", "sha256"}:
        fail("code record is noncanonical")
    expected_bytes = expected["bytes"]
    expected_sha256 = expected["sha256"]
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 0
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        fail("code record is invalid")
    target = os.path.realpath(os.path.join(workspace_root, *canonical_role(role)))
    if os.path.commonpath((workspace_root, target)) != workspace_root:
        fail("code path escapes captured workspace")
    digest = hashlib.sha256()
    byte_count = 0
    with open(target, "rb") as handle:
        while True:
            chunk = handle.read(4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
    observed = {"bytes": byte_count, "sha256": digest.hexdigest()}
    if observed != expected:
        fail("captured code bytes do not match their parent record")
    return {"role": role, **observed}


def capture_project_source(workspace_root, role, expected):
    if not isinstance(expected, dict) or set(expected) != {"bytes", "sha256"}:
        fail("code record is noncanonical")
    expected_bytes = expected["bytes"]
    expected_sha256 = expected["sha256"]
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 0
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        fail("code record is invalid")
    target = normalized_path(os.path.join(workspace_root, *canonical_role(role)))
    if os.path.commonpath((workspace_root, target)) != workspace_root:
        fail("code path escapes captured workspace")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(target, flags)
    try:
        opened = os.fstat(descriptor)
        lexical = os.stat(target, follow_symlinks=False)
        if not stat.S_ISREG(opened.st_mode) or not os.path.samestat(opened, lexical):
            fail("captured project source is not a stable regular file")
        digest = hashlib.sha256()
        payload = bytearray()
        while True:
            chunk = os.read(descriptor, 4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            payload.extend(chunk)
        after = os.fstat(descriptor)
        lexical_after = os.stat(target, follow_symlinks=False)
        if (
            not os.path.samestat(opened, after)
            or not os.path.samestat(opened, lexical_after)
            or after.st_size != len(payload)
        ):
            fail("captured project source changed while capturing")
    finally:
        os.close(descriptor)
    observed = {"bytes": len(payload), "sha256": digest.hexdigest()}
    if observed != expected:
        fail("captured code bytes do not match their parent record")
    return {
        "role": role,
        "origin": target,
        **observed,
        "_payload": bytes(payload),
    }


def stable_origin_record(origin, module_names):
    normalized = normalized_path(origin)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(normalized, flags)
    try:
        opened = os.fstat(descriptor)
        lexical = os.stat(normalized, follow_symlinks=False)
        if not stat.S_ISREG(opened.st_mode) or not os.path.samestat(opened, lexical):
            fail("external dependency origin is not a stable regular file")
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            chunk = os.read(descriptor, 4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
        after = os.fstat(descriptor)
        if not os.path.samestat(opened, after) or after.st_size != byte_count:
            fail("external dependency origin changed while hashing")
    finally:
        os.close(descriptor)
    kind = "native-extension" if any(
        normalized.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES
    ) else "python-module"
    return {
        "origin": normalized,
        "kind": kind,
        "modules": sorted(module_names),
        "bytes": byte_count,
        "sha256": digest.hexdigest(),
    }


def dependency_fingerprint(record):
    return {
        "origin": record["origin"],
        "kind": record["kind"],
        "bytes": record["bytes"],
        "sha256": record["sha256"],
        "executed_origin": record["executed_origin"],
        "executed_bytes": record["executed_bytes"],
        "executed_sha256": record["executed_sha256"],
    }


def public_dependency_record(record):
    return {
        key: record[key]
        for key in (
            "origin",
            "executed_origin",
            "kind",
            "modules",
            "bytes",
            "sha256",
            "executed_bytes",
            "executed_sha256",
        )
    }


def capture_dependency_origin(origin, module_name, kind):
    global dependency_capture_bytes, dependency_capture_counter
    normalized = normalized_path(origin)
    if dependency_capture_counter >= limits["dependency_files"]:
        fail("external dependency capture exceeds its file-count limit")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    source_descriptor = os.open(normalized, flags)
    destination = os.path.join(
        dependency_root,
        "%06d-%s" % (dependency_capture_counter, os.path.basename(normalized)),
    )
    try:
        destination_descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
    except BaseException:
        os.close(source_descriptor)
        raise
    dependency_capture_counter += 1
    digest = hashlib.sha256()
    payload = bytearray() if kind == "python-source" else None
    byte_count = 0
    try:
        opened = os.fstat(source_descriptor)
        lexical = os.stat(normalized, follow_symlinks=False)
        if not stat.S_ISREG(opened.st_mode) or not os.path.samestat(opened, lexical):
            fail("external dependency origin is not a stable regular file")
        while True:
            chunk = os.read(source_descriptor, 4 * 1024 * 1024)
            if not chunk:
                break
            if dependency_capture_bytes + len(chunk) > limits["dependency_bytes"]:
                fail("external dependency capture exceeds its byte limit")
            view = memoryview(chunk)
            while view:
                written = os.write(destination_descriptor, view)
                if written <= 0:
                    fail("external dependency private capture write failed")
                view = view[written:]
            digest.update(chunk)
            if payload is not None:
                payload.extend(chunk)
            byte_count += len(chunk)
            dependency_capture_bytes += len(chunk)
        os.fsync(destination_descriptor)
        after = os.fstat(source_descriptor)
        lexical_after = os.stat(normalized, follow_symlinks=False)
        if (
            not os.path.samestat(opened, after)
            or not os.path.samestat(opened, lexical_after)
            or after.st_size != byte_count
        ):
            fail("external dependency origin changed while capturing")
    finally:
        os.close(source_descriptor)
        os.close(destination_descriptor)
    captured = os.stat(destination, follow_symlinks=False)
    if not stat.S_ISREG(captured.st_mode) or captured.st_size != byte_count:
        fail("external dependency private capture is invalid")
    sha256 = digest.hexdigest()
    return (
        {
            "origin": normalized,
            "executed_origin": normalized_path(destination),
            "kind": (
                "native-extension" if kind == "native-extension" else "python-module"
            ),
            "modules": [module_name],
            "bytes": byte_count,
            "sha256": sha256,
            "executed_bytes": byte_count,
            "executed_sha256": sha256,
            "_payload": bytes(payload) if payload is not None else None,
        },
        bytes(payload) if payload is not None else None,
    )


class CapturedSourceLoader:
    def __init__(self, record, module_name, is_package):
        self._record = record
        self._module_name = module_name
        self._is_package = is_package
        self.path = record["origin"]

    def create_module(self, spec):
        return None

    def get_filename(self, fullname):
        if fullname != self._module_name:
            fail("captured source loader module identity changed")
        return self._record["origin"]

    def is_package(self, fullname):
        if fullname != self._module_name:
            fail("captured source loader module identity changed")
        return self._is_package

    def get_source(self, fullname):
        self.get_filename(fullname)
        return importlib.util.decode_source(self._record["_payload"])

    def get_code(self, fullname):
        filename = self.get_filename(fullname)
        return compile(
            self._record["_payload"],
            filename,
            "exec",
            dont_inherit=True,
        )

    def exec_module(self, module):
        module.__cached__ = None
        exec(self.get_code(self._module_name), module.__dict__)


class CapturedExtensionLoader:
    def __init__(self, record, module_name):
        self._record = record
        self._loader = importlib.machinery.ExtensionFileLoader(
            module_name, record["executed_origin"]
        )

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)

    def __getattr__(self, name):
        return getattr(self._loader, name)


class CapturedProjectFinder:
    def __init__(self, code_root, records):
        self._code_root = code_root
        self._records = records

    def find_spec(self, fullname, path=None, target=None):
        record = self._records.get(fullname)
        if record is not None:
            loader = CapturedSourceLoader(record, fullname, False)
            spec = importlib.util.spec_from_loader(
                fullname,
                loader,
                origin=record["origin"],
            )
            if spec is None:
                fail("could not construct captured project module spec")
            spec.has_location = True
            spec.cached = None
            return spec
        candidate = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if candidate is not None:
            origin = getattr(candidate, "origin", None)
            if isinstance(origin, str) and is_within(normalized_path(origin), self._code_root):
                fail("campaign verifier attempted to load an unregistered project module")
            locations = getattr(candidate, "submodule_search_locations", None)
            if locations and any(
                is_within(normalized_path(item), self._code_root)
                for item in locations
            ):
                fail("campaign verifier attempted to load an unregistered project package")
        return None


class DependencyTrackingFinder:
    def __init__(self, site_roots, records):
        self._site_roots = tuple(site_roots)
        self._records = records

    def find_spec(self, fullname, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None or not isinstance(spec.origin, str):
            return spec
        origin = normalized_path(spec.origin)
        if any(is_within(origin, root) for root in self._site_roots):
            if isinstance(spec.loader, importlib.machinery.SourcelessFileLoader) or origin.endswith(
                tuple(importlib.machinery.BYTECODE_SUFFIXES)
            ):
                fail("sourceless bytecode dependencies are forbidden")
            is_extension = isinstance(
                spec.loader, importlib.machinery.ExtensionFileLoader
            ) or any(origin.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES)
            is_source = isinstance(
                spec.loader, importlib.machinery.SourceFileLoader
            ) or origin.endswith(tuple(importlib.machinery.SOURCE_SUFFIXES))
            if not is_extension and not is_source:
                fail("external dependency loader is not source or native-extension backed")
            existing = self._records.get(origin)
            if existing is None:
                existing, _payload = capture_dependency_origin(
                    origin,
                    fullname,
                    "native-extension" if is_extension else "python-source",
                )
                self._records[origin] = existing
            else:
                existing["modules"] = sorted(
                    set(existing["modules"]) | {fullname}
                )
            if is_extension:
                spec.origin = existing["executed_origin"]
                spec.loader = CapturedExtensionLoader(existing, fullname)
            else:
                spec.cached = None
                spec.loader = CapturedSourceLoader(
                    existing,
                    fullname,
                    spec.submodule_search_locations is not None,
                )
        return spec


def dependency_inventory(
    code_root,
    stdlib_roots,
    site_roots,
    expected_project_paths,
    tracked_records,
    dependency_capture_root,
):
    observed_origins = set()
    namespaces = {}
    for module_name, module in sorted(sys.modules.items()):
        if module is None:
            continue
        spec = getattr(module, "__spec__", None)
        origin = getattr(module, "__file__", None)
        if not isinstance(origin, str) and spec is not None:
            candidate = getattr(spec, "origin", None)
            if isinstance(candidate, str) and candidate not in ("built-in", "frozen"):
                origin = candidate
        if isinstance(origin, str):
            normalized = normalized_path(origin)
            if is_within(normalized, code_root):
                if normalized not in expected_project_paths:
                    fail("campaign verifier loaded an unregistered project module")
                continue
            tracked = [
                record
                for record in tracked_records.values()
                if module_name in record["modules"]
            ]
            if tracked:
                if len(tracked) != 1 or normalized not in {
                    tracked[0]["origin"],
                    tracked[0]["executed_origin"],
                }:
                    fail("campaign verifier dependency module origin changed")
                observed_origins.add(tracked[0]["origin"])
                continue
            if any(is_within(normalized, root) for root in site_roots) or is_within(
                normalized, dependency_capture_root
            ):
                fail("campaign verifier loaded an untracked external dependency")
            if any(is_within(normalized, root) for root in stdlib_roots):
                continue
            fail("campaign verifier loaded a module outside attested roots")
        locations = getattr(spec, "submodule_search_locations", None)
        if locations:
            normalized_locations = sorted({normalized_path(item) for item in locations})
            if all(
                any(is_within(item, root) for root in site_roots)
                for item in normalized_locations
            ):
                namespaces[module_name] = normalized_locations
            elif not all(
                is_within(item, code_root)
                or any(is_within(item, root) for root in stdlib_roots)
                for item in normalized_locations
            ):
                fail("campaign verifier loaded a namespace outside attested roots")
    return {
        "files": [
            public_dependency_record(tracked_records[origin])
            for origin in sorted(observed_origins)
        ],
        "namespaces": [
            {"module": module, "locations": locations}
            for module, locations in sorted(namespaces.items())
        ],
    }


if len(sys.argv) != 3:
    fail("isolated replay requires request and output paths")
request_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
if os.path.realpath(request_path.parent) != os.path.realpath(output_path.parent):
    fail("isolated replay IPC paths do not share their private directory")
request_payload = request_path.read_bytes()
request = strict_json_bytes(request_payload, "isolated replay request")
if canonical_json_bytes(request) != request_payload:
    fail("isolated replay request is not canonical JSON")
if not isinstance(request, dict) or set(request) != REQUEST_FIELDS:
    fail("isolated replay request fields are noncanonical")
if request["schema"] != REQUEST_SCHEMA:
    fail("isolated replay request schema is invalid")

limits = request["limits"]
if (
    not isinstance(limits, dict)
    or set(limits)
    != {
        "stdout_bytes",
        "stderr_bytes",
        "result_bytes",
        "dependency_bytes",
        "dependency_files",
    }
    or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in limits.values())
):
    fail("isolated replay output limits are invalid")

expected_environment = request["environment"]
if not isinstance(expected_environment, dict) or any(
    not isinstance(key, str) or not isinstance(value, str)
    for key, value in expected_environment.items()
):
    fail("isolated replay environment contract is invalid")
actual_environment = normalized_environment(dict(os.environ))
if actual_environment != expected_environment:
    fail("isolated replay environment changed")

workspace_root = os.path.realpath(os.path.abspath(request["workspace_root"]))
results_root = os.path.realpath(os.path.abspath(request["results_root"]))
dependency_root = normalized_path(request["dependency_root"])
if request["workspace_root"] != workspace_root:
    fail("captured workspace path is noncanonical")
if request["results_root"] != results_root:
    fail("captured results path is noncanonical")
if os.path.commonpath((workspace_root, results_root)) != workspace_root:
    fail("captured results path escapes captured workspace")
if os.path.commonpath((os.path.realpath(request_path.parent), dependency_root)) != os.path.realpath(request_path.parent):
    fail("dependency capture root escapes private replay IPC")
if request["dependency_root"] != dependency_root or not os.path.isdir(dependency_root):
    fail("dependency capture root is noncanonical")
if os.listdir(dependency_root):
    fail("dependency capture root is not empty")
dependency_capture_bytes = 0
dependency_capture_counter = 0

actual_interpreter = {
    "executable": os.path.normcase(os.path.realpath(sys.executable)),
    "implementation": sys.implementation.name,
    "version": list(sys.version_info[:3]),
    "prefix": normalized_path(sys.prefix),
    "base_prefix": normalized_path(sys.base_prefix),
    "exec_prefix": normalized_path(sys.exec_prefix),
    "base_exec_prefix": normalized_path(sys.base_exec_prefix),
}
if request["interpreter"] != actual_interpreter:
    fail("isolated replay interpreter identity changed")
command = request["command"]
if (
    not isinstance(command, dict)
    or set(command)
    != {
        "executable",
        "flags",
        "bootstrap_sha256",
        "probe_bootstrap_sha256",
    }
    or not isinstance(command["executable"], str)
    or not os.path.isabs(command["executable"])
    or normalized_path(command["executable"]) != actual_interpreter["executable"]
    or command["flags"] != ["-I", "-S", "-B", "-c"]
    or not isinstance(command["bootstrap_sha256"], str)
    or len(command["bootstrap_sha256"]) != 64
    or not isinstance(command["probe_bootstrap_sha256"], str)
    or len(command["probe_bootstrap_sha256"]) != 64
):
    fail("isolated replay command identity is invalid")
actual_flags = {
    "dont_write_bytecode": bool(sys.flags.dont_write_bytecode),
    "ignore_environment": bool(sys.flags.ignore_environment),
    "isolated": bool(sys.flags.isolated),
    "no_site": bool(sys.flags.no_site),
    "no_user_site": bool(sys.flags.no_user_site),
    "safe_path": bool(sys.flags.safe_path),
}
if actual_flags != {
    "dont_write_bytecode": True,
    "ignore_environment": True,
    "isolated": True,
    "no_site": True,
    "no_user_site": True,
    "safe_path": True,
}:
    fail("isolated replay interpreter flags are invalid")
site_paths = normalized_site_paths()
if request["site_paths"] != site_paths:
    fail("isolated replay site-package roots changed")
startup_sys_path = normalized_paths(sys.path)
if request["startup_sys_path"] != startup_sys_path:
    fail("isolated replay startup sys.path changed")

code_records = request["code_records"]
executed_modules = request["executed_modules"]
if not isinstance(code_records, dict) or not isinstance(executed_modules, dict):
    fail("isolated replay code inventory is invalid")
if set(code_records) != {
    "code/aggregate_corrected_campaign.py",
    "code/run_corrected_experiments.py",
    "code/corrected_protocol.py",
    "code/models.py",
    "code/03_embed_proteins.py",
    "code/embedding_provenance.py",
}:
    fail("isolated replay code inventory is incomplete")
if executed_modules != {
    "aggregate_corrected_campaign": "code/aggregate_corrected_campaign.py",
    "corrected_protocol": "code/corrected_protocol.py",
    "embedding_provenance": "code/embedding_provenance.py",
    "run_corrected_experiments": "code/run_corrected_experiments.py",
    "models": "code/models.py",
}:
    fail("isolated replay module inventory is noncanonical")
project_sources_by_role = {
    role: capture_project_source(workspace_root, role, expected)
    for role, expected in code_records.items()
}
project_sources_by_module = {
    module_name: project_sources_by_role[role]
    for module_name, role in executed_modules.items()
}

project_modules = set(executed_modules)
if any(name in sys.modules for name in project_modules):
    fail("isolated replay inherited a project module")
code_root = os.path.join(workspace_root, "code")
code_root = normalized_path(code_root)
effective_sys_path = []
for path in (code_root, *startup_sys_path, *site_paths):
    if path not in effective_sys_path:
        effective_sys_path.append(path)
if request["effective_sys_path"] != effective_sys_path:
    fail("isolated replay effective sys.path contract changed")
sys.path[:] = list(effective_sys_path)
stdlib_roots = sorted({
    normalized_path(sys.base_prefix),
    normalized_path(sysconfig.get_path("stdlib")),
    normalized_path(sysconfig.get_path("platstdlib")),
})
expected_project_paths = {
    normalized_path(os.path.join(workspace_root, *role.split("/")))
    for role in code_records
}
tracked_dependencies = {}
dependency_finder = DependencyTrackingFinder(site_paths, tracked_dependencies)
project_finder = CapturedProjectFinder(code_root, project_sources_by_module)
sys.meta_path.insert(0, dependency_finder)
sys.meta_path.insert(0, project_finder)
aggregator = importlib.import_module("aggregate_corrected_campaign")
post_import_sys_path = normalized_paths(sys.path)
if post_import_sys_path != effective_sys_path:
    fail("campaign verifier changed sys.path during import")

module_records = {}
for module_name, role in executed_modules.items():
    module = sys.modules.get(module_name)
    if module is None or not isinstance(getattr(module, "__file__", None), str):
        fail("campaign verifier did not load a required captured module")
    expected_path = os.path.realpath(os.path.join(workspace_root, *role.split("/")))
    if os.path.realpath(module.__file__) != expected_path:
        fail("campaign verifier loaded a project module outside the capture")
    source_record = project_sources_by_role[role]
    module_records[module_name] = {
        "role": role,
        "bytes": source_record["bytes"],
        "sha256": source_record["sha256"],
    }

if list(aggregator.COMMON_INPUT_PATHS) != request["common_input_paths"]:
    fail("captured aggregator common-input contract changed")
if aggregator.TRANSFER_INPUT_PATH != request["transfer_input_path"]:
    fail("captured aggregator transfer-input contract changed")
if list(aggregator.CODE_PATHS) != request["code_paths"]:
    fail("captured aggregator code contract changed")

dependencies_before = dependency_inventory(
    code_root,
    stdlib_roots,
    site_paths,
    expected_project_paths,
    tracked_dependencies,
    dependency_root,
)

report = aggregator.validate_campaign(
    Path(results_root), workspace_root=Path(workspace_root)
)
if not isinstance(report, dict):
    fail("captured campaign verifier returned a non-object")
post_replay_sys_path = normalized_paths(sys.path)
if post_replay_sys_path != effective_sys_path:
    fail("campaign verifier changed sys.path during replay")
dependencies_after = dependency_inventory(
    code_root,
    stdlib_roots,
    site_paths,
    expected_project_paths,
    tracked_dependencies,
    dependency_root,
)
before_origins = {record["origin"] for record in dependencies_before["files"]}
after_origins = {record["origin"] for record in dependencies_after["files"]}
if not before_origins.issubset(after_origins):
    fail("external dependency inventory shrank during replay")
attested_dependency_files = []
for origin, import_record in sorted(tracked_dependencies.items()):
    final_origin = stable_origin_record(origin, import_record["modules"])
    if (
        final_origin["bytes"] != import_record["bytes"]
        or final_origin["sha256"] != import_record["sha256"]
    ):
        fail("external dependency origin changed after capture")
    final_executed = stable_origin_record(
        import_record["executed_origin"], import_record["modules"]
    )
    if (
        final_executed["bytes"] != import_record["executed_bytes"]
        or final_executed["sha256"] != import_record["executed_sha256"]
    ):
        fail("external dependency private capture changed after import")
    attested_dependency_files.append(public_dependency_record(import_record))
dependencies_attested = {
    "files": attested_dependency_files,
    "namespaces": dependencies_after["namespaces"],
}
envelope = {
    "schema": RESULT_SCHEMA,
    "status": "ok",
    "execution": {
        "interpreter": actual_interpreter,
        "command": command,
        "flags": actual_flags,
        "environment": actual_environment,
        "startup_sys_path": startup_sys_path,
        "effective_sys_path": effective_sys_path,
        "post_import_sys_path": post_import_sys_path,
        "post_replay_sys_path": post_replay_sys_path,
        "site_paths": site_paths,
        "code_root": code_root,
        "dependency_scope": (
            "Python source modules and importable native-extension module files "
            "loaded from the configured site-package roots through the replay "
            "import hook; transitive shared libraries and data/resource files "
            "are outside this byte attestation"
        ),
    },
    "module_records": module_records,
    "dependency_records": dependencies_attested,
    "report": report,
}
payload = canonical_json_bytes(envelope)
if len(payload) > limits["result_bytes"]:
    fail("isolated replay result exceeds its byte limit")
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
descriptor = os.open(output_path, flags, 0o600)
with os.fdopen(descriptor, "wb") as handle:
    handle.write(payload)
    handle.flush()
    os.fsync(handle.fileno())
"""


def _campaign_replay_launcher() -> str:
    """Return the exact lexical interpreter path used for both child processes."""
    launcher = os.path.abspath(os.fspath(sys.executable))
    if not Path(launcher).is_file():
        raise ValidationError("campaign replay interpreter launcher is unavailable")
    return launcher


def _campaign_replay_expected_flags() -> dict[str, bool]:
    return {
        "dont_write_bytecode": True,
        "ignore_environment": True,
        "isolated": True,
        "no_site": True,
        "no_user_site": True,
        "safe_path": True,
    }


def _campaign_replay_inline_code(source: str, *, label: str) -> str:
    """Encode a bounded bootstrap below Windows' process command-line limit."""
    if not label or any(
        character not in "abcdefghijklmnopqrstuvwxyz-" for character in label
    ):
        raise ValidationError("campaign replay bootstrap label is invalid")
    compressed = zlib.compress(source.encode("utf-8"), level=9).hex()
    return (
        "import zlib;exec(compile(zlib.decompress(bytes.fromhex('"
        + compressed
        + "')), '<"
        + label
        + ">', 'exec', dont_inherit=True))"
    )


def _require_probe_paths(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValidationError(f"{label} is invalid")
    paths: list[str] = []
    for candidate in value:
        if (
            not isinstance(candidate, str)
            or not candidate
            or not Path(candidate).is_absolute()
            or candidate
            != os.path.normcase(os.path.realpath(os.path.abspath(candidate)))
            or candidate in paths
        ):
            raise ValidationError(f"{label} is noncanonical")
        paths.append(candidate)
    return paths


def _probe_campaign_replay_runtime(
    launcher: str,
    *,
    ipc_root: Path,
    environment: Mapping[str, str],
    result_directory: _PinnedDirectory | None = None,
    capture_store: _CaptureStore | None = None,
    result_relative: Path | None = None,
) -> dict[str, Any]:
    """Measure ``-I -S`` startup state in the exact child interpreter."""
    result_path = ipc_root / "probe-result.json"
    try:
        completed = _run_bounded_replay_process(
            [
                launcher,
                *CAMPAIGN_REPLAY_FLAGS,
                _campaign_replay_inline_code(
                    CAMPAIGN_REPLAY_PROBE_BOOTSTRAP,
                    label="campaign-replay-probe",
                ),
                str(result_path),
            ],
            cwd=ipc_root,
            environment=environment,
            result_path=result_path,
            result_directory=result_directory,
            stdout_limit=CAMPAIGN_REPLAY_STDOUT_MAX_BYTES,
            stderr_limit=CAMPAIGN_REPLAY_STDERR_MAX_BYTES,
            result_limit=CAMPAIGN_REPLAY_PROBE_MAX_BYTES,
            timeout_seconds=CAMPAIGN_REPLAY_PROBE_TIMEOUT_SECONDS,
        )
    except BaseException as exc:
        try:
            if (
                result_directory is not None
                and capture_store is not None
                and result_relative is not None
                and _pinned_regular_file_exists(
                    result_directory,
                    result_path.name,
                    label="campaign replay runtime probe result",
                )
            ):
                capture_store.register_private_file(
                    result_relative,
                    label="campaign replay runtime probe result",
                )
        except Exception as adoption_error:
            exc.add_note(
                "campaign replay probe-output adoption also failed: "
                f"{adoption_error}"
            )
        raise
    if completed.result_identity is not None and capture_store is not None:
        if result_relative is None:
            raise ValidationError("campaign replay probe private path is missing")
        capture_store.register_private_file(
            result_relative,
            label="campaign replay runtime probe result",
            expected_identity=completed.result_identity,
        )
    if completed.returncode != 0:
        raise ValidationError(
            "campaign replay runtime probe failed with exit code "
            f"{completed.returncode}"
        )
    if completed.stdout or completed.stderr:
        raise ValidationError("campaign replay runtime probe emitted process output")
    if completed.result is None:
        raise ValidationError("campaign replay runtime probe produced no result")
    if completed.result_identity is None:
        raise ValidationError("campaign replay runtime probe result is unbound")
    payload = completed.result
    result = _strict_json_bytes(payload, label="campaign replay runtime probe")
    if _canonical_json_bytes(result) != payload:
        raise ValidationError("campaign replay runtime probe is not canonical JSON")
    if not isinstance(result, dict) or set(result) != {
        "schema",
        "runtime",
        "flags",
        "environment",
        "startup_sys_path",
        "site_paths",
    }:
        raise ValidationError("campaign replay runtime probe fields are noncanonical")
    if result.get("schema") != CAMPAIGN_REPLAY_PROBE_SCHEMA:
        raise ValidationError("campaign replay runtime probe schema is invalid")
    if result.get("flags") != _campaign_replay_expected_flags():
        raise ValidationError("campaign replay runtime probe flags are invalid")
    if result.get("environment") != dict(environment):
        raise ValidationError("campaign replay runtime probe environment changed")
    runtime = result.get("runtime")
    if not isinstance(runtime, dict) or set(runtime) != {
        "executable",
        "implementation",
        "version",
        "prefix",
        "base_prefix",
        "exec_prefix",
        "base_exec_prefix",
    }:
        raise ValidationError("campaign replay runtime identity is noncanonical")
    version = runtime.get("version")
    if (
        runtime.get("executable")
        != os.path.normcase(os.path.realpath(os.path.abspath(launcher)))
        or not isinstance(runtime.get("implementation"), str)
        or not runtime["implementation"]
        or not isinstance(version, list)
        or len(version) != 3
        or any(isinstance(item, bool) or not isinstance(item, int) for item in version)
    ):
        raise ValidationError("campaign replay runtime identity is invalid")
    for key in ("prefix", "base_prefix", "exec_prefix", "base_exec_prefix"):
        value = runtime.get(key)
        if (
            not isinstance(value, str)
            or not Path(value).is_absolute()
            or value != os.path.normcase(os.path.realpath(os.path.abspath(value)))
        ):
            raise ValidationError("campaign replay runtime prefix is invalid")
    startup_sys_path = _require_probe_paths(
        result.get("startup_sys_path"), label="campaign replay startup sys.path"
    )
    site_paths = _require_probe_paths(
        result.get("site_paths"), label="campaign replay site-package roots"
    )
    if any(
        not Path(path).is_dir()
        or (
            "/site-packages" not in path.replace("\\", "/").lower()
            and "/dist-packages" not in path.replace("\\", "/").lower()
        )
        for path in site_paths
    ):
        raise ValidationError("campaign replay site-package roots are invalid")
    return {
        "interpreter": runtime,
        "startup_sys_path": startup_sys_path,
        "site_paths": site_paths,
        "result_identity": completed.result_identity,
    }


def _campaign_replay_environment(ipc_root: Path) -> dict[str, str]:
    """Construct the complete allowlisted environment used by replay."""
    environment: dict[str, str] = {}
    allowlisted = (
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "SYSTEMDRIVE",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "HOME",
    )
    for name in allowlisted:
        value = os.environ.get(name)
        if value is not None:
            key = name.upper() if os.name == "nt" else name
            environment[key] = value
    private_temp = str(ipc_root.resolve())
    for name in ("TEMP", "TMP", "TMPDIR"):
        environment[name] = private_temp
    environment.update(
        {
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    return dict(sorted(environment.items()))


def _create_windows_replay_job() -> int:
    """Create a Job Object whose assigned process tree dies when the job closes."""
    if os.name != "nt":
        raise ValidationError("Windows replay jobs are unavailable on this platform")
    import ctypes
    from ctypes import wintypes

    class JobObjectBasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class JobObjectExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JobObjectBasicLimitInformation),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    handle = kernel32.CreateJobObjectW(None, None)
    if not handle:
        error = ctypes.get_last_error()
        raise ValidationError(
            "could not create campaign replay Job Object"
        ) from OSError(error, os.strerror(error))
    handle_value = int(handle)
    information = JobObjectExtendedLimitInformation()
    information.BasicLimitInformation.LimitFlags = (
        0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    )
    if not kernel32.SetInformationJobObject(
        wintypes.HANDLE(handle_value),
        9,  # JobObjectExtendedLimitInformation
        ctypes.byref(information),
        ctypes.sizeof(information),
    ):
        error = ctypes.get_last_error()
        _windows_close_handle(handle_value)
        raise ValidationError(
            "could not configure campaign replay Job Object"
        ) from OSError(error, os.strerror(error))
    return handle_value


def _assign_windows_replay_job(
    job_handle: int, process: subprocess.Popen[bytes]
) -> None:
    import ctypes
    from ctypes import wintypes

    process_handle = getattr(process, "_handle", None)
    if process_handle is None:
        raise ValidationError("campaign replay process handle is unavailable")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    if not kernel32.AssignProcessToJobObject(
        wintypes.HANDLE(job_handle), wintypes.HANDLE(int(process_handle))
    ):
        error = ctypes.get_last_error()
        raise ValidationError(
            "could not assign campaign replay process to its Job Object"
        ) from OSError(error, os.strerror(error))


def _resume_windows_replay_process(process: subprocess.Popen[bytes]) -> None:
    """Resume a replay leader only after it is covered by its Job Object."""
    import ctypes
    from ctypes import wintypes

    process_handle = getattr(process, "_handle", None)
    if process_handle is None:
        raise ValidationError("campaign replay process handle is unavailable")
    ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
    ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
    ntdll.NtResumeProcess.restype = ctypes.c_long
    status = int(ntdll.NtResumeProcess(wintypes.HANDLE(int(process_handle))))
    if status != 0:
        raise ValidationError(
            f"could not resume campaign replay process (NTSTATUS 0x{status & 0xFFFFFFFF:08x})"
        )


def _terminate_replay_process(
    process: subprocess.Popen[bytes],
    *,
    process_group_id: int | None = None,
    windows_job_handle: int | None = None,
) -> None:
    """Best-effort termination of the replay leader and all inherited descendants."""
    if windows_job_handle is not None:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.TerminateJobObject(wintypes.HANDLE(windows_job_handle), 1)
        try:
            process.wait(timeout=5.0)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
                process.wait(timeout=5.0)
            except (OSError, subprocess.TimeoutExpired):
                pass
        return

    if process_group_id is not None and os.name != "nt":
        kill_process_group = getattr(os, "killpg", None)
        sigkill = getattr(signal, "SIGKILL", None)
        if not callable(kill_process_group) or sigkill is None:
            _terminate_replay_process(process)
            return
        try:
            kill_process_group(process_group_id, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
        deadline = time.monotonic() + 1.0
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        try:
            kill_process_group(process_group_id, sigkill)
        except (ProcessLookupError, OSError):
            pass
        try:
            process.wait(timeout=5.0)
        except (OSError, subprocess.TimeoutExpired):
            pass
        return

    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)
    except OSError:
        try:
            process.kill()
        except OSError:
            pass


def _run_bounded_replay_process(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    result_path: Path,
    result_directory: _PinnedDirectory | None = None,
    stdout_limit: int,
    stderr_limit: int,
    result_limit: int,
    timeout_seconds: float,
) -> _ReplayProcessOutcome:
    """Run replay while draining and bounding all three child output channels."""
    limits = {
        "stdout": stdout_limit,
        "stderr": stderr_limit,
        "result": result_limit,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in limits.values()
    ):
        raise ValidationError(
            "campaign replay byte limits must be non-negative integers"
        )
    if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
        raise ValidationError("campaign replay timeout must be positive")
    result_name = result_path.name
    _require_leaf_name(result_name, label="authoritative campaign replay result")
    owned_result_directory = result_directory is None
    pinned_result_directory = (
        _pin_directory(
            result_path.parent,
            label="authoritative campaign replay result directory",
        )
        if result_directory is None
        else result_directory
    )
    expected_parent = os.path.normcase(
        os.path.abspath(os.fspath(pinned_result_directory.path))
    )
    observed_parent = os.path.normcase(os.path.abspath(os.fspath(result_path.parent)))
    if observed_parent != expected_parent:
        if owned_result_directory:
            _close_pinned_directory(
                pinned_result_directory,
                label="authoritative campaign replay result directory",
            )
        raise ValidationError(
            "authoritative campaign replay result parent does not match its pin"
        )
    try:
        result_exists = _pinned_regular_file_exists(
            pinned_result_directory,
            result_name,
            label="authoritative campaign replay result",
        )
    except BaseException as exc:
        if owned_result_directory:
            try:
                _close_pinned_directory(
                    pinned_result_directory,
                    label="authoritative campaign replay result directory",
                )
            except Exception as cleanup_error:
                exc.add_note(
                    "replay result-directory cleanup also failed: " f"{cleanup_error}"
                )
        raise
    if result_exists:
        error = ValidationError(
            "authoritative campaign replay result path already exists"
        )
        if owned_result_directory:
            try:
                _close_pinned_directory(
                    pinned_result_directory,
                    label="authoritative campaign replay result directory",
                )
            except Exception as cleanup_error:
                error.add_note(
                    "replay result-directory cleanup also failed: " f"{cleanup_error}"
                )
        raise error

    windows_job_handle: int | None = None
    try:
        if os.name == "nt":
            windows_job_handle = _create_windows_replay_job()
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            bufsize=0,
            close_fds=True,
            start_new_session=os.name != "nt",
            creationflags=0x00000004 if os.name == "nt" else 0,
        )
        if windows_job_handle is not None:
            try:
                _assign_windows_replay_job(windows_job_handle, process)
                _resume_windows_replay_process(process)
            except BaseException:
                try:
                    _terminate_replay_process(
                        process, windows_job_handle=windows_job_handle
                    )
                    process.wait(timeout=5.0)
                except (OSError, subprocess.TimeoutExpired):
                    pass
                raise
    except BaseException as exc:
        if windows_job_handle is not None:
            _windows_close_handle(windows_job_handle)
        if owned_result_directory:
            try:
                _close_pinned_directory(
                    pinned_result_directory,
                    label="authoritative campaign replay result directory",
                )
            except Exception as cleanup_error:
                exc.add_note(
                    "replay result-directory cleanup also failed: " f"{cleanup_error}"
                )
        raise ValidationError(
            "authoritative campaign replay process could not be started"
        ) from exc
    process_group_id = process.pid if os.name != "nt" else None
    assert process.stdout is not None
    assert process.stderr is not None
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    result_buffer = bytearray()
    result_descriptor: int | None = None
    result_identity: tuple[int, int] | None = None
    failure: list[str] = []
    failure_lock = threading.Lock()
    termination_lock = threading.Lock()
    tree_terminated = False

    def terminate_tree() -> None:
        nonlocal tree_terminated
        with termination_lock:
            if tree_terminated:
                return
            _terminate_replay_process(
                process,
                process_group_id=process_group_id,
                windows_job_handle=windows_job_handle,
            )
            tree_terminated = True

    def fail_and_kill(reason: str) -> None:
        with failure_lock:
            if not failure:
                failure.append(reason)
        terminate_tree()

    def drain(name: str, stream: Any, limit: int) -> None:
        while True:
            try:
                chunk = stream.read(16 * 1024)
            except OSError:
                fail_and_kill(f"authoritative campaign replay {name} could not be read")
                return
            if not chunk:
                return
            remaining = max(0, limit + 1 - len(buffers[name]))
            buffers[name].extend(chunk[:remaining])
            if len(buffers[name]) > limit:
                fail_and_kill(
                    f"authoritative campaign replay {name} exceeded "
                    f"{limit}-byte quota"
                )
                return

    threads = [
        threading.Thread(
            target=drain,
            args=("stdout", process.stdout, stdout_limit),
            daemon=True,
        ),
        threading.Thread(
            target=drain,
            args=("stderr", process.stderr, stderr_limit),
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()

    def drain_result() -> None:
        nonlocal result_descriptor, result_identity
        if result_descriptor is None:
            try:
                descriptor, identity = _open_pinned_regular_file(
                    pinned_result_directory,
                    result_name,
                    label="authoritative campaign replay result",
                )
            except ValidationError as exc:
                cause = exc.__cause__
                if isinstance(cause, OSError) and (
                    cause.errno in (2,) or getattr(cause, "winerror", None) in (2, 3)
                ):
                    return
                raise
            result_descriptor = descriptor
            result_identity = identity
        while True:
            read_size = min(
                16 * 1024,
                max(1, result_limit + 1 - len(result_buffer)),
            )
            try:
                chunk = os.read(result_descriptor, read_size)
            except OSError as exc:
                raise ValidationError(
                    "authoritative campaign replay result could not be read"
                ) from exc
            if not chunk:
                return
            result_buffer.extend(chunk)
            if len(result_buffer) > result_limit:
                fail_and_kill(
                    "authoritative campaign replay result exceeded "
                    f"{result_limit}-byte quota"
                )
                return

    deadline = time.monotonic() + float(timeout_seconds)
    try:
        while process.poll() is None:
            if failure:
                terminate_tree()
                break
            try:
                drain_result()
            except ValidationError as exc:
                fail_and_kill(str(exc))
            if time.monotonic() >= deadline:
                fail_and_kill("authoritative campaign replay timed out")
            if failure:
                terminate_tree()
                break
            time.sleep(0.01)
        process.wait(timeout=5.0)
        terminate_tree()
        if not failure:
            try:
                drain_result()
            except ValidationError as exc:
                fail_and_kill(str(exc))
    except subprocess.TimeoutExpired:
        fail_and_kill("authoritative campaign replay timed out")
        terminate_tree()
    finally:
        terminate_tree()
        for thread in threads:
            thread.join(timeout=5.0)
            if thread.is_alive():
                fail_and_kill(
                    "authoritative campaign replay output drain did not terminate"
                )
        try:
            process.stdout.close()
            process.stderr.close()
        except OSError:
            if not failure:
                failure.append(
                    "authoritative campaign replay output pipe could not be closed"
                )
        if windows_job_handle is not None:
            _windows_close_handle(windows_job_handle)
    if not failure and result_identity is not None:
        try:
            if result_descriptor is None:
                raise OSError("campaign replay result descriptor is unavailable")
            metadata = os.fstat(result_descriptor)
        except OSError:
            failure.append(
                "authoritative campaign replay result could not be inspected"
            )
        else:
            observed_identity = (int(metadata.st_dev), int(metadata.st_ino))
            if observed_identity != result_identity or not stat.S_ISREG(
                metadata.st_mode
            ):
                failure.append("authoritative campaign replay result identity changed")
            elif metadata.st_size != len(result_buffer):
                failure.append(
                    "authoritative campaign replay result size changed while reading"
                )
    close_failure: Exception | None = None
    if result_descriptor is not None:
        try:
            os.close(result_descriptor)
        except OSError as exc:
            close_failure = exc
    if owned_result_directory:
        try:
            _close_pinned_directory(
                pinned_result_directory,
                label="authoritative campaign replay result directory",
            )
        except Exception as exc:
            if close_failure is None:
                close_failure = exc
    if failure:
        error = ValidationError(failure[0])
        if close_failure is not None:
            error.add_note(f"replay result cleanup also failed: {close_failure}")
        raise error
    if close_failure is not None:
        raise ValidationError(
            "authoritative campaign replay result could not be closed"
        ) from close_failure
    return _ReplayProcessOutcome(
        returncode=int(process.returncode),
        stdout=bytes(buffers["stdout"]),
        stderr=bytes(buffers["stderr"]),
        result=(bytes(result_buffer) if result_identity is not None else None),
        result_identity=result_identity,
    )


def _campaign_replay_code_records(
    workspace_root: Path, captures: Mapping[str, _CapturedFile]
) -> dict[str, dict[str, Any]]:
    roles = (CAMPAIGN_AGGREGATOR_ROLE, *CAMPAIGN_CODE_ROLES)
    records: dict[str, dict[str, Any]] = {}
    for role in roles:
        captured = captures.get(role)
        expected_path = workspace_root / Path(role)
        if captured is None or captured.path.resolve() != expected_path.resolve():
            raise ValidationError(
                f"captured campaign verifier dependency is missing: {role}"
            )
        records[role] = {"bytes": captured.bytes, "sha256": captured.sha256}
    return records


def _require_campaign_replay_dependencies(
    value: Any,
    *,
    site_paths: Sequence[str],
    dependency_root: Path,
    dependency_relative: Path,
    adopted_files: Mapping[str, tuple[int, int]],
    capture_store: _CaptureStore,
) -> None:
    if not isinstance(value, dict) or set(value) != {"files", "namespaces"}:
        raise ValidationError("campaign replay dependency inventory is noncanonical")
    file_records = value.get("files")
    namespace_records = value.get("namespaces")
    if not isinstance(file_records, list) or not isinstance(namespace_records, list):
        raise ValidationError("campaign replay dependency inventory is invalid")
    observed_origins: list[str] = []
    observed_executed_names: list[str] = []
    source_bytes = 0
    executed_bytes = 0
    for index, record in enumerate(file_records):
        if not isinstance(record, dict) or set(record) != {
            "origin",
            "executed_origin",
            "kind",
            "modules",
            "bytes",
            "sha256",
            "executed_bytes",
            "executed_sha256",
        }:
            raise ValidationError("campaign replay dependency record is noncanonical")
        origin = record.get("origin")
        executed_origin = record.get("executed_origin")
        modules = record.get("modules")
        if (
            not isinstance(origin, str)
            or not Path(origin).is_absolute()
            or origin != os.path.normcase(os.path.realpath(origin))
            or not isinstance(executed_origin, str)
            or not Path(executed_origin).is_absolute()
            or executed_origin != os.path.normcase(os.path.realpath(executed_origin))
            or record.get("kind") not in {"python-module", "native-extension"}
            or not isinstance(modules, list)
            or not modules
            or modules != sorted(set(modules))
            or not all(isinstance(module, str) and module for module in modules)
        ):
            raise ValidationError("campaign replay dependency record is invalid")
        if not any(
            _path_is_within(Path(origin), Path(site_root)) for site_root in site_paths
        ):
            raise ValidationError(
                "campaign replay dependency origin is outside site-package roots"
            )
        executed_path = Path(executed_origin)
        if (
            os.path.normcase(os.path.realpath(executed_path.parent))
            != os.path.normcase(os.path.realpath(dependency_root))
            or executed_path.name not in adopted_files
        ):
            raise ValidationError(
                "campaign replay executed dependency is outside its private inventory"
            )
        expected_kind = (
            "native-extension"
            if any(
                origin.endswith(suffix)
                for suffix in importlib.machinery.EXTENSION_SUFFIXES
            )
            else "python-module"
        )
        if record["kind"] != expected_kind:
            raise ValidationError("campaign replay dependency kind is invalid")
        if record.get("executed_bytes") != record.get("bytes") or record.get(
            "executed_sha256"
        ) != record.get("sha256"):
            raise ValidationError(
                "campaign replay executed dependency differs from captured source"
            )
        source_remaining = CAMPAIGN_REPLAY_DEPENDENCY_MAX_BYTES - source_bytes
        captured_source = capture_store.capture(
            Path(origin),
            label=f"campaign replay dependency {origin}",
            expected_record=record,
            destination_relative=(
                Path("campaign-replay-dependencies") / "source" / f"{index:06d}"
            ),
            max_bytes=source_remaining,
        )
        source_bytes += captured_source.bytes
        executed_remaining = CAMPAIGN_REPLAY_DEPENDENCY_MAX_BYTES - executed_bytes
        captured_executed = capture_store.verify_private_file(
            dependency_relative / executed_path.name,
            label=f"campaign replay executed dependency {executed_origin}",
            expected_record={
                "bytes": record["executed_bytes"],
                "sha256": record["executed_sha256"],
            },
            max_bytes=executed_remaining,
        )
        executed_bytes += captured_executed.bytes
        observed_origins.append(origin)
        observed_executed_names.append(executed_path.name)
    if observed_origins != sorted(set(observed_origins)):
        raise ValidationError("campaign replay dependency origins are noncanonical")
    if len(observed_executed_names) != len(set(observed_executed_names)) or set(
        observed_executed_names
    ) != set(adopted_files):
        raise ValidationError(
            "campaign replay executed dependency inventory is not exact"
        )
    observed_namespaces: list[str] = []
    for record in namespace_records:
        if not isinstance(record, dict) or set(record) != {"module", "locations"}:
            raise ValidationError(
                "campaign replay namespace dependency record is noncanonical"
            )
        module = record.get("module")
        locations = record.get("locations")
        if (
            not isinstance(module, str)
            or not module
            or not isinstance(locations, list)
            or not locations
            or locations != sorted(set(locations))
        ):
            raise ValidationError("campaign replay namespace record is invalid")
        for location in locations:
            if (
                not isinstance(location, str)
                or not Path(location).is_absolute()
                or location != os.path.normcase(os.path.realpath(location))
                or not any(
                    _path_is_within(Path(location), Path(site_root))
                    for site_root in site_paths
                )
            ):
                raise ValidationError(
                    "campaign replay namespace location is outside site-package roots"
                )
            _reject_link_or_reparse_components(
                Path(location), label="campaign replay namespace dependency"
            )
        observed_namespaces.append(module)
    if observed_namespaces != sorted(set(observed_namespaces)):
        raise ValidationError("campaign replay namespace modules are noncanonical")


def _authoritative_campaign_replay(
    results_root: Path,
    *,
    workspace_root: Path,
    workspace_captures: Mapping[str, _CapturedFile],
    capture_store: _CaptureStore,
) -> dict[str, Any]:
    """Replay captured verifier bytes in a fresh, attested interpreter process.

    The interpreter/import and inherited process-tree controls are integrity and
    resource-containment measures.  They are not an operating-system sandbox.
    """
    workspace_root = Path(os.path.abspath(os.fspath(workspace_root)))
    results_root = Path(os.path.abspath(os.fspath(results_root)))
    try:
        results_root.relative_to(workspace_root)
    except ValueError as exc:
        raise ValidationError(
            "captured campaign results root escapes the captured workspace"
        ) from exc
    code_records = _campaign_replay_code_records(workspace_root, workspace_captures)
    bootstrap_sha256 = hashlib.sha256(
        CAMPAIGN_REPLAY_BOOTSTRAP.encode("utf-8")
    ).hexdigest()
    probe_bootstrap_sha256 = hashlib.sha256(
        CAMPAIGN_REPLAY_PROBE_BOOTSTRAP.encode("utf-8")
    ).hexdigest()
    invocation_id = uuid.uuid4().hex
    ipc_relative = Path(f"campaign-replay-{invocation_id}")
    ipc_root = capture_store.root / ipc_relative
    ipc_pin = capture_store.create_private_directory(
        ipc_relative, label="authoritative campaign replay IPC directory"
    )
    request_path = ipc_root / "request.json"
    output_path = ipc_root / "result.json"
    dependency_relative = ipc_relative / "dependencies"
    dependency_root = capture_store.root / dependency_relative
    capture_store.create_private_directory(
        dependency_relative,
        label="authoritative campaign replay dependency directory",
    )
    process_temp_root = capture_store.create_process_temp_directory(
        label="authoritative campaign replay process temp directory"
    )
    environment = _campaign_replay_environment(process_temp_root)
    launcher = _campaign_replay_launcher()
    probe = _probe_campaign_replay_runtime(
        launcher,
        ipc_root=ipc_root,
        environment=environment,
        result_directory=ipc_pin,
        capture_store=capture_store,
        result_relative=ipc_relative / "probe-result.json",
    )
    probe_identity = probe.get("result_identity")
    if (
        not isinstance(probe_identity, tuple)
        or len(probe_identity) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in probe_identity
        )
    ):
        raise ValidationError(
            "campaign replay runtime probe result identity is invalid"
        )
    interpreter = probe["interpreter"]
    site_paths = probe["site_paths"]
    startup_sys_path = probe["startup_sys_path"]
    command_record = {
        "executable": launcher,
        "flags": list(CAMPAIGN_REPLAY_FLAGS),
        "bootstrap_sha256": bootstrap_sha256,
        "probe_bootstrap_sha256": probe_bootstrap_sha256,
    }
    code_root = os.path.normcase(os.path.realpath(str(workspace_root / "code")))
    effective_sys_path: list[str] = []
    for path in (code_root, *startup_sys_path, *site_paths):
        if path not in effective_sys_path:
            effective_sys_path.append(path)
    limits = {
        "stdout_bytes": CAMPAIGN_REPLAY_STDOUT_MAX_BYTES,
        "stderr_bytes": CAMPAIGN_REPLAY_STDERR_MAX_BYTES,
        "result_bytes": CAMPAIGN_REPLAY_RESULT_MAX_BYTES,
        "dependency_bytes": CAMPAIGN_REPLAY_DEPENDENCY_MAX_BYTES,
        "dependency_files": CAMPAIGN_REPLAY_DEPENDENCY_MAX_FILES,
    }
    request = {
        "schema": CAMPAIGN_REPLAY_REQUEST_SCHEMA,
        "workspace_root": str(workspace_root),
        "results_root": str(results_root),
        "dependency_root": os.path.normcase(os.path.realpath(dependency_root)),
        "interpreter": interpreter,
        "command": command_record,
        "environment": environment,
        "startup_sys_path": startup_sys_path,
        "effective_sys_path": effective_sys_path,
        "site_paths": site_paths,
        "code_records": code_records,
        "common_input_paths": list(CAMPAIGN_COMMON_INPUT_ROLES),
        "transfer_input_path": CAMPAIGN_TRANSFER_INPUT_ROLE,
        "code_paths": list(CAMPAIGN_CODE_ROLES),
        "executed_modules": CAMPAIGN_EXECUTED_MODULE_ROLES,
        "limits": limits,
    }
    capture_store.write_private_bytes(
        ipc_relative / "request.json",
        _canonical_json_bytes(request),
        label="authoritative campaign replay request",
    )
    command = [
        launcher,
        *CAMPAIGN_REPLAY_FLAGS,
        _campaign_replay_inline_code(
            CAMPAIGN_REPLAY_BOOTSTRAP,
            label="campaign-replay-bootstrap",
        ),
        str(request_path),
        str(output_path),
    ]
    process_error: BaseException | None = None
    completed: _ReplayProcessOutcome | None = None
    try:
        completed = _run_bounded_replay_process(
            command,
            cwd=workspace_root,
            environment=environment,
            result_path=output_path,
            result_directory=ipc_pin,
            stdout_limit=limits["stdout_bytes"],
            stderr_limit=limits["stderr_bytes"],
            result_limit=limits["result_bytes"],
            timeout_seconds=CAMPAIGN_REPLAY_TIMEOUT_SECONDS,
        )
    except BaseException as exc:
        process_error = exc
    try:
        if completed is not None and completed.result_identity is not None:
            capture_store.register_private_file(
                ipc_relative / "result.json",
                label="authoritative campaign replay result",
                expected_identity=completed.result_identity,
            )
        elif _pinned_regular_file_exists(
            ipc_pin,
            output_path.name,
            label="authoritative campaign replay result",
        ):
            capture_store.register_private_file(
                ipc_relative / "result.json",
                label="authoritative campaign replay result",
            )
        adopted_dependency_files = capture_store.adopt_direct_regular_files(
            dependency_relative,
            label="authoritative campaign replay dependency output",
            max_entries=CAMPAIGN_REPLAY_DEPENDENCY_MAX_FILES,
        )
    except BaseException as adoption_error:
        if process_error is not None:
            process_error.add_note(
                "campaign replay child-output adoption also failed: "
                f"{adoption_error}"
            )
        else:
            raise
    if process_error is not None:
        raise process_error
    if completed is None:
        raise ValidationError("authoritative campaign replay process did not complete")
    if completed.returncode != 0:
        raise ValidationError(
            "authoritative campaign replay process failed with exit code "
            f"{completed.returncode}"
        )
    if completed.stdout or completed.stderr:
        raise ValidationError(
            "authoritative campaign replay emitted unexpected process output"
        )
    if completed.result is None:
        raise ValidationError("authoritative campaign replay produced no result")
    output_payload = completed.result
    envelope = _strict_json_bytes(
        output_payload, label="authoritative campaign replay result"
    )
    if _canonical_json_bytes(envelope) != output_payload:
        raise ValidationError(
            "authoritative campaign replay result is not canonical JSON"
        )
    if not isinstance(envelope, dict) or set(envelope) != {
        "schema",
        "status",
        "execution",
        "module_records",
        "dependency_records",
        "report",
    }:
        raise ValidationError(
            "authoritative campaign replay result fields are noncanonical"
        )
    if (
        envelope.get("schema") != CAMPAIGN_REPLAY_RESULT_SCHEMA
        or envelope.get("status") != "ok"
    ):
        raise ValidationError("authoritative campaign replay result is unsuccessful")
    expected_execution = {
        "interpreter": interpreter,
        "command": command_record,
        "flags": _campaign_replay_expected_flags(),
        "environment": environment,
        "startup_sys_path": startup_sys_path,
        "effective_sys_path": effective_sys_path,
        "post_import_sys_path": effective_sys_path,
        "post_replay_sys_path": effective_sys_path,
        "site_paths": site_paths,
        "code_root": code_root,
        "dependency_scope": (
            "Python source modules and importable native-extension module files "
            "loaded from the configured site-package roots through the replay "
            "import hook; transitive shared libraries and data/resource files "
            "are outside this byte attestation"
        ),
    }
    if envelope.get("execution") != expected_execution:
        raise ValidationError(
            "authoritative campaign replay execution identity changed"
        )
    expected_module_records = {
        module_name: {
            "role": role,
            **code_records[role],
        }
        for module_name, role in CAMPAIGN_EXECUTED_MODULE_ROLES.items()
    }
    if envelope.get("module_records") != expected_module_records:
        raise ValidationError("authoritative campaign replay module records changed")
    _require_campaign_replay_dependencies(
        envelope.get("dependency_records"),
        site_paths=site_paths,
        dependency_root=dependency_root,
        dependency_relative=dependency_relative,
        adopted_files=adopted_dependency_files,
        capture_store=capture_store,
    )
    replay = envelope.get("report")
    if not isinstance(replay, dict):
        raise ValidationError("authoritative campaign replay did not return an object")
    return cast(dict[str, Any], replay)


def _canonical_campaign_workspace_path(
    value: Any,
    *,
    workspace_root: Path,
    recorded_root: Path | None,
    label: str,
) -> str:
    """Canonicalize a campaign path without dereferencing a stale old root."""
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{label} path must be a non-empty string")
    current_root = workspace_root.resolve()
    raw = Path(value)
    if not raw.is_absolute():
        resolved = _relocated_child(current_root, raw, label=label)
        return resolved.relative_to(current_root).as_posix()

    roots = [current_root]
    if recorded_root is not None:
        roots.append(recorded_root.absolute())
    for candidate_root in roots:
        try:
            relative = raw.relative_to(candidate_root)
        except ValueError:
            continue
        resolved = _relocated_child(current_root, relative, label=label)
        return resolved.relative_to(current_root).as_posix()
    raise ValidationError(f"{label} path is outside the current and recorded roots")


def _campaign_replay_payload(
    report: Mapping[str, Any],
    *,
    workspace_root: Path | None = None,
    recorded_root: Path | None = None,
) -> bytes:
    deterministic = json.loads(
        json.dumps(report, sort_keys=True, ensure_ascii=True, allow_nan=False)
    )
    deterministic.pop("created_utc", None)
    if workspace_root is None:
        return _canonical_json_bytes(deterministic)

    source_runs = deterministic.get("source_runs")
    if isinstance(source_runs, list):
        for index, run in enumerate(source_runs):
            if isinstance(run, dict) and "path" in run:
                run["path"] = _canonical_campaign_workspace_path(
                    run["path"],
                    workspace_root=workspace_root,
                    recorded_root=recorded_root,
                    label=f"campaign source_runs[{index}]",
                )
    validation = deterministic.get("validation")
    excluded = (
        validation.get("excluded_noncomplete_runs")
        if isinstance(validation, dict)
        else None
    )
    if isinstance(excluded, list):
        for index, issue in enumerate(excluded):
            if isinstance(issue, dict) and "path" in issue:
                issue["path"] = _canonical_campaign_workspace_path(
                    issue["path"],
                    workspace_root=workspace_root,
                    recorded_root=recorded_root,
                    label=f"campaign excluded_noncomplete_runs[{index}]",
                )
    return _canonical_json_bytes(deterministic)


def _require_campaign_replay_match(
    report: Mapping[str, Any],
    replay: Mapping[str, Any],
    *,
    workspace_root: Path | None = None,
    recorded_root: Path | None = None,
) -> None:
    replay_payload = _campaign_replay_payload(
        replay,
        workspace_root=workspace_root,
        recorded_root=recorded_root,
    )
    report_payload = _campaign_replay_payload(
        report,
        workspace_root=workspace_root,
        recorded_root=recorded_root,
    )
    if replay_payload != report_payload:
        raise ValidationError(
            "campaign report differs from authoritative semantic replay"
        )


def _canonical_campaign_input_role(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{label} path role must be a non-empty string")
    normalized = value.replace("\\", "/")
    parts = normalized.split("/")
    if (
        normalized.startswith("/")
        or ":" in parts[0]
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise ValidationError(f"{label} path role is not canonical")
    return "/".join(parts)


def _campaign_ppi_record_from_manifest(
    run_manifest: Mapping[str, Any], *, run_key: str
) -> dict[str, Any]:
    inputs = run_manifest.get("inputs")
    if not isinstance(inputs, list):
        raise ValidationError(f"campaign run inputs are missing for {run_key}")
    matches: list[Mapping[str, Any]] = []
    for index, record in enumerate(inputs):
        if not isinstance(record, Mapping):
            raise ValidationError(
                f"campaign input record is not an object for {run_key}: {index}"
            )
        role = _canonical_campaign_input_role(
            record.get("path"), label=f"{run_key}.inputs[{index}]"
        )
        if role == CAMPAIGN_PPI_INPUT_ROLE:
            matches.append(record)
    if len(matches) != 1:
        raise ValidationError(
            f"campaign run must bind exactly one canonical PPI input role: {run_key}"
        )
    record = matches[0]
    if set(record) != CAMPAIGN_INPUT_RECORD_FIELDS:
        raise ValidationError(f"campaign PPI input record is noncanonical: {run_key}")
    size = record.get("bytes")
    mtime_ns = record.get("mtime_ns")
    digest = record.get("sha256")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValidationError(f"campaign PPI byte count is invalid: {run_key}")
    if isinstance(mtime_ns, bool) or not isinstance(mtime_ns, int) or mtime_ns < 0:
        raise ValidationError(f"campaign PPI mtime_ns is invalid: {run_key}")
    if not _valid_sha256(digest):
        raise ValidationError(f"campaign PPI SHA-256 is invalid: {run_key}")
    return {
        "role": CAMPAIGN_PPI_INPUT_ROLE,
        "bytes": size,
        "sha256": digest,
    }


def _canonical_campaign_ppi_binding(
    report: Mapping[str, Any],
    *,
    workspace_root: Path | None = None,
    recorded_root: Path | None = None,
    capture_store: _CaptureStore | None = None,
) -> dict[str, Any]:
    """Extract one byte identity for the PPI role from all 99 run manifests."""
    source_runs = report.get("source_runs")
    if not isinstance(source_runs, list) or len(source_runs) != EXPECTED_RUNS:
        raise ValidationError(
            "campaign PPI binding requires exactly 99 source-run manifests"
        )
    keys = [run.get("key") for run in source_runs if isinstance(run, Mapping)]
    if len(keys) != EXPECTED_RUNS or set(keys) != _expected_campaign_keys():
        raise ValidationError("campaign PPI binding source-run keys are noncanonical")
    owned_store = capture_store is None
    store = (
        _CaptureStore(prefix="depplm-campaign-ppi-") if owned_store else capture_store
    )
    assert store is not None
    try:
        consensus: dict[str, Any] | None = None
        for run in source_runs:
            if not isinstance(run, Mapping):
                raise ValidationError("campaign PPI source run is not an object")
            run_key = run.get("key")
            if not isinstance(run_key, str):
                raise ValidationError("campaign PPI source-run key is invalid")
            run_path = _resolve_campaign_run_path(
                run.get("path"),
                workspace_root=workspace_root,
                recorded_root=recorded_root,
            )
            manifest_path = run_path / "run-manifest.json"
            expected_digest = run.get("run_manifest_sha256")
            if not _valid_sha256(expected_digest):
                raise ValidationError(
                    f"campaign PPI run manifest binding is invalid: {run_key}"
                )
            payload, captured = store.capture_bytes(
                manifest_path, label=f"campaign PPI run manifest {run_key}"
            )
            if captured.sha256 != expected_digest:
                raise ValidationError(f"campaign PPI run manifest changed: {run_key}")
            run_manifest = _strict_json_bytes(
                payload, label=f"campaign PPI run manifest {run_key}"
            )
            if not isinstance(run_manifest, Mapping):
                raise ValidationError(
                    f"campaign PPI run manifest is not an object: {run_key}"
                )
            candidate = _campaign_ppi_record_from_manifest(
                run_manifest, run_key=run_key
            )
            if consensus is None:
                consensus = candidate
            elif candidate != consensus:
                raise ValidationError(
                    "campaign run manifests do not agree on the canonical PPI "
                    "role, byte count, and SHA-256"
                )
    finally:
        if owned_store:
            store.__exit__(None, None, None)
    if consensus is None:
        raise ValidationError("campaign PPI binding is empty")
    return consensus


def _require_campaign_ppi_match(
    ppi_path: Path, canonical_binding: Mapping[str, Any]
) -> None:
    if set(canonical_binding) != CAMPAIGN_PPI_BINDING_FIELDS or (
        canonical_binding.get("role") != CAMPAIGN_PPI_INPUT_ROLE
    ):
        raise ValidationError("canonical campaign PPI binding is invalid")
    with _CaptureStore(prefix="depplm-ppi-match-") as store:
        captured = store.capture(ppi_path, label="supplied PPI file")
        _require_campaign_ppi_capture(captured, canonical_binding)


def _require_campaign_ppi_capture(
    captured: _CapturedFile, canonical_binding: Mapping[str, Any]
) -> None:
    if set(canonical_binding) != CAMPAIGN_PPI_BINDING_FIELDS or (
        canonical_binding.get("role") != CAMPAIGN_PPI_INPUT_ROLE
    ):
        raise ValidationError("canonical campaign PPI binding is invalid")
    if captured.bytes != canonical_binding.get(
        "bytes"
    ) or captured.sha256 != canonical_binding.get("sha256"):
        raise ValidationError(
            "supplied PPI bytes do not match the canonical PPI input bound by "
            "all validated campaign run manifests"
        )


CAMPAIGN_CAPTURE_MAX_BYTES = 16 * 1024 * 1024 * 1024
CAMPAIGN_WORKSPACE_CAPTURE_MAX_BYTES = 16 * 1024 * 1024 * 1024


def _registered_campaign_workspace_paths() -> tuple[str, ...]:
    """Return the authoritative verifier's complete workspace file inventory."""
    raw_paths = (
        *CAMPAIGN_COMMON_INPUT_ROLES,
        CAMPAIGN_TRANSFER_INPUT_ROLE,
        CAMPAIGN_AGGREGATOR_ROLE,
        *CAMPAIGN_CODE_ROLES,
    )
    paths: list[str] = []
    for index, raw_path in enumerate(raw_paths):
        if not isinstance(raw_path, str):
            raise ValidationError(
                "authoritative campaign workspace inventory contains a non-string "
                f"entry at index {index}"
            )
        normalized = _canonical_campaign_input_role(
            raw_path, label=f"campaign workspace inventory[{index}]"
        )
        if normalized != raw_path.replace("\\", "/"):
            raise ValidationError(
                "authoritative campaign workspace inventory is noncanonical"
            )
        paths.append(normalized)
    if len(paths) != len(set(paths)):
        raise ValidationError(
            "authoritative campaign workspace inventory contains duplicate paths"
        )
    return tuple(paths)


def _capture_campaign_workspace_inputs(
    source_root: Path,
    store: _CaptureStore,
    *,
    destination_relative: Path = Path("campaign-workspace"),
) -> tuple[Path, dict[str, _CapturedFile]]:
    """Capture every workspace file dereferenced by authoritative replay once."""
    source_root = Path(os.path.abspath(os.fspath(source_root)))
    _reject_link_or_reparse_components(source_root, label="campaign workspace root")
    if not source_root.is_dir():
        raise ValidationError(f"campaign workspace root is missing: {source_root}")
    mirror_root = store.root / destination_relative
    store.create_private_directory(
        destination_relative, label="campaign workspace mirror"
    )
    captures: dict[str, _CapturedFile] = {}
    total_bytes = 0
    for role in _registered_campaign_workspace_paths():
        source = _relocated_child(source_root, Path(role), label=f"campaign {role}")
        remaining = CAMPAIGN_WORKSPACE_CAPTURE_MAX_BYTES - total_bytes
        captured = store.capture(
            source,
            label=f"campaign workspace input {role}",
            logical_path=role,
            destination_relative=destination_relative / Path(role),
            max_bytes=remaining,
        )
        total_bytes += captured.bytes
        captures[role] = captured
    return mirror_root, captures


def _capture_campaign_results_tree(
    source_root: Path,
    store: _CaptureStore,
    *,
    destination_relative: Path = Path("campaign-results"),
) -> tuple[Path, dict[Path, _CapturedFile]]:
    """Capture one bounded, stable filesystem view for authoritative replay."""
    source_root = Path(os.path.abspath(os.fspath(source_root)))
    _reject_link_or_reparse_components(source_root, label="campaign results root")
    if not source_root.is_dir():
        raise ValidationError(f"campaign results root is missing: {source_root}")
    mirror_root = store.root / destination_relative
    store.create_private_directory(
        destination_relative, label="campaign results mirror"
    )
    captures: dict[Path, _CapturedFile] = {}
    total_bytes = 0
    for source in sorted(source_root.rglob("*"), key=lambda item: item.as_posix()):
        relative = source.relative_to(source_root)
        if source.is_symlink() or _has_reparse_point(source):
            raise ValidationError(
                f"campaign results tree contains a symlink or reparse point: {source}"
            )
        if source.is_dir():
            store.create_private_directory(
                destination_relative / relative,
                label="campaign results mirror directory",
            )
            continue
        if not source.is_file():
            raise ValidationError(f"campaign results tree entry is invalid: {source}")
        remaining = CAMPAIGN_CAPTURE_MAX_BYTES - total_bytes
        captured = store.capture(
            source,
            label="campaign source artifact",
            destination_relative=destination_relative / relative,
            max_bytes=remaining,
        )
        total_bytes += captured.bytes
        captures[Path(os.path.abspath(os.fspath(source)))] = captured
    return mirror_root, captures


def _strict_parse_campaign_json_tree(root: Path, *, label: str) -> None:
    """Reject ambiguous JSON before the legacy campaign verifier can parse it."""
    for path in sorted(root.rglob("*.json"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.is_symlink() or _has_reparse_point(path):
            raise ValidationError(f"{label} JSON entry is invalid: {path}")
        _strict_json_bytes(path.read_bytes(), label=f"{label} JSON {path.name}")


def _rebase_replay_paths(
    replay: Mapping[str, Any], *, captured_root: Path, source_root: Path
) -> dict[str, Any]:
    rebased = json.loads(
        json.dumps(replay, sort_keys=True, ensure_ascii=True, allow_nan=False)
    )

    def rebase(value: Any, *, label: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValidationError(f"{label} path is invalid")
        path = Path(value)
        if not path.is_absolute():
            return value
        try:
            relative = path.resolve().relative_to(captured_root.resolve())
        except ValueError:
            return value
        return str(source_root.resolve() / relative)

    runs = rebased.get("source_runs")
    if isinstance(runs, list):
        for index, run in enumerate(runs):
            if isinstance(run, dict) and "path" in run:
                run["path"] = rebase(run["path"], label=f"replay source_runs[{index}]")
    validation = rebased.get("validation")
    excluded = (
        validation.get("excluded_noncomplete_runs")
        if isinstance(validation, dict)
        else None
    )
    if isinstance(excluded, list):
        for index, issue in enumerate(excluded):
            if isinstance(issue, dict) and "path" in issue:
                issue["path"] = rebase(issue["path"], label=f"replay excluded[{index}]")
    return rebased


def validate_campaign_report(
    path: Path,
    *,
    workspace_root: Path | None = None,
    recorded_root: Path | None = None,
    _capture_store: _CaptureStore | None = None,
    _return_captured_paths: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require the complete scientific 99-run report and re-hash every source file."""
    if _capture_store is None:
        if _return_captured_paths:
            raise ValidationError(
                "captured campaign paths require a caller-owned capture store"
            )
        with _CaptureStore(prefix="depplm-campaign-validation-") as store:
            return validate_campaign_report(
                path,
                workspace_root=workspace_root,
                recorded_root=recorded_root,
                _capture_store=store,
                _return_captured_paths=_return_captured_paths,
            )
    path = Path(os.path.abspath(os.fspath(path)))
    report_payload, report_capture = _capture_store.capture_bytes(
        path, label="campaign report"
    )
    report = _strict_json_bytes(report_payload, label="campaign report")
    if not isinstance(report, dict) or report.get("schema") != CAMPAIGN_SCHEMA:
        raise ValidationError("campaign report schema is not corrected-v2.2")
    if set(report) != CAMPAIGN_FIELDS:
        raise ValidationError("campaign report fields are not canonical")
    if report.get("scientific_result") is not True:
        raise ValidationError("campaign scientific_result must be true")
    if report.get("protocol_validation_passed") is not True:
        raise ValidationError("campaign protocol validation did not pass")
    gates = report.get("scientific_readiness_gates")
    if gates != CAMPAIGN_READINESS_GATES:
        raise ValidationError("campaign scientific-readiness gates are noncanonical")
    matrix = report.get("prespecified_matrix")
    if matrix != CAMPAIGN_MATRIX:
        raise ValidationError("campaign prespecified matrix is noncanonical")
    validation = report.get("validation")
    if (
        not isinstance(validation, dict)
        or set(validation) != CAMPAIGN_VALIDATION_FIELDS
    ):
        raise ValidationError("campaign validation section is noncanonical")
    if validation.get("missing_matrix_cells") != [] or validation.get("issues") != []:
        raise ValidationError("campaign contains missing cells or validation issues")
    if validation.get("duplicate_matrix_cells") != {}:
        raise ValidationError("campaign contains duplicate matrix cells")
    if (
        validation.get("validated_runs") != EXPECTED_RUNS
        or validation.get("unique_matrix_cells") != EXPECTED_RUNS
    ):
        raise ValidationError("campaign did not validate exactly 99 unique runs")
    source_runs = report.get("source_runs")
    if not isinstance(source_runs, list) or len(source_runs) != EXPECTED_RUNS:
        raise ValidationError("campaign source_runs must contain exactly 99 records")
    keys = [run.get("key") for run in source_runs if isinstance(run, dict)]
    if len(keys) != EXPECTED_RUNS or set(keys) != _expected_campaign_keys():
        raise ValidationError(
            "campaign keys are incomplete, duplicated, or noncanonical"
        )
    if any(set(run) != CAMPAIGN_SOURCE_RUN_FIELDS for run in source_runs):
        raise ValidationError("campaign source-run fields are noncanonical")
    resolved_run_paths = [
        _resolve_campaign_run_path(
            run.get("path"),
            workspace_root=workspace_root,
            recorded_root=recorded_root,
        )
        for run in source_runs
    ]
    source_paths = [os.path.normcase(str(path)) for path in resolved_run_paths]
    if len(set(source_paths)) != EXPECTED_RUNS:
        raise ValidationError("campaign source run paths must be unique")
    current_root = (ROOT if workspace_root is None else workspace_root).resolve()
    results_root = _campaign_results_root(resolved_run_paths)
    try:
        results_relative = results_root.resolve().relative_to(current_root)
    except ValueError as exc:
        raise ValidationError(
            "campaign results root must be contained by the current workspace"
        ) from exc
    captured_workspace_root, _workspace_captures = _capture_campaign_workspace_inputs(
        current_root, _capture_store
    )
    captured_results_root, captured_files = _capture_campaign_results_tree(
        results_root,
        _capture_store,
        destination_relative=Path("campaign-workspace") / results_relative,
    )
    _strict_parse_campaign_json_tree(
        captured_workspace_root, label="captured campaign workspace"
    )
    analysis_report = json.loads(
        json.dumps(report, sort_keys=True, ensure_ascii=True, allow_nan=False)
    )
    analysis_run_paths: dict[str, Path] = {}
    for run, source_path in zip(analysis_report["source_runs"], resolved_run_paths):
        relative = source_path.resolve().relative_to(results_root.resolve())
        captured_path = captured_results_root / relative
        run["path"] = str(captured_path)
        analysis_run_paths[run["key"]] = captured_path

    artifact_count = 0
    for run in analysis_report["source_runs"]:
        if not isinstance(run, dict):
            raise ValidationError("campaign source run is not an object")
        run_path = analysis_run_paths[run["key"]]
        if not run_path.is_dir():
            raise ValidationError(
                f"campaign source run directory is missing: {run_path}"
            )
        root_hashes = {
            "completion.json": run.get("completion_sha256"),
            "summary.json": run.get("summary_sha256"),
            "run-manifest.json": run.get("run_manifest_sha256"),
            "cohort-manifest.json": run.get("cohort_manifest_sha256"),
        }
        for filename, digest in root_hashes.items():
            file_path = run_path / filename
            if not _valid_sha256(digest) or not file_path.is_file():
                raise ValidationError(f"invalid campaign source artifact: {file_path}")
            captured = captured_files.get(
                Path(
                    os.path.abspath(
                        os.fspath(
                            results_root
                            / run_path.relative_to(captured_results_root)
                            / filename
                        )
                    )
                )
            )
            if captured is None or captured.sha256 != digest:
                raise ValidationError(f"campaign source artifact changed: {file_path}")
            artifact_count += 1
        run_manifest = _read_json_mapping(
            run_path / "run-manifest.json", label="campaign run manifest"
        )
        protocol, condition, seed_token = run["key"].split("|")
        arguments = run_manifest.get("arguments")
        if not isinstance(arguments, dict) or (
            arguments.get("protocol") != protocol
            or arguments.get("condition") != condition
            or arguments.get("seed") != int(seed_token.removeprefix("seed-"))
        ):
            raise ValidationError(
                f"campaign key does not match run-manifest arguments: {run.get('key')}"
            )
        expected_fingerprint = run.get("runtime_fingerprint_sha256")
        if (
            not _valid_sha256(expected_fingerprint)
            or _runtime_fingerprint_sha256(run_manifest) != expected_fingerprint
        ):
            raise ValidationError(f"runtime fingerprint changed for {run.get('key')}")
        artifacts = run.get("fold_artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ValidationError(
                f"fold artifact inventory missing for {run.get('key')}"
            )
        expected_artifacts = _expected_campaign_artifacts(protocol)
        observed_artifacts = tuple(
            artifact.get("path") if isinstance(artifact, dict) else None
            for artifact in artifacts
        )
        if observed_artifacts != expected_artifacts:
            raise ValidationError(
                f"fold artifact inventory is noncanonical for {run.get('key')}"
            )
        expected_directory = {
            "completion.json",
            "summary.json",
            "run-manifest.json",
            "cohort-manifest.json",
            *expected_artifacts,
        }
        directory_entries = list(run_path.iterdir())
        observed_directory = {
            child.name for child in directory_entries if child.is_file()
        }
        if observed_directory != expected_directory or any(
            not child.is_file() or child.is_symlink() or _has_reparse_point(child)
            for child in directory_entries
        ):
            raise ValidationError(
                f"source run file inventory is noncanonical for {run.get('key')}"
            )
        seen: set[str] = set()
        for index, artifact in enumerate(artifacts):
            if not isinstance(artifact, dict):
                raise ValidationError("campaign fold artifact record is not an object")
            if set(artifact) != {"path", "bytes", "sha256"}:
                raise ValidationError("campaign fold artifact record is noncanonical")
            artifact_relative = artifact.get("path")
            if not isinstance(artifact_relative, str):
                raise ValidationError("campaign fold artifact path is invalid")
            if artifact_relative in seen:
                raise ValidationError(
                    "duplicate fold artifact for "
                    f"{run.get('key')}: {artifact_relative}"
                )
            seen.add(artifact_relative)
            _verify_file_record(
                _safe_child(run_path, artifact_relative),
                artifact,
                label=f"{run.get('key')}.fold_artifacts[{index}]",
            )
            artifact_count += 1
    replay = _authoritative_campaign_replay(
        captured_results_root,
        workspace_root=captured_workspace_root,
        workspace_captures=_workspace_captures,
        capture_store=_capture_store,
    )
    replay = _rebase_replay_paths(
        replay, captured_root=captured_results_root, source_root=results_root
    )
    _require_campaign_replay_match(
        report,
        replay,
        workspace_root=current_root,
        recorded_root=recorded_root,
    )
    _canonical_campaign_ppi_binding(
        analysis_report,
        workspace_root=current_root,
        recorded_root=None,
        capture_store=_capture_store,
    )
    returned_report = analysis_report if _return_captured_paths else report
    return returned_report, {
        **report_capture.record(),
        "source_runs_rehashed": EXPECTED_RUNS,
        "source_artifacts_rehashed": artifact_count,
    }


def _p1_run_map(report: Mapping[str, Any]) -> dict[tuple[str, int], Mapping[str, Any]]:
    runs: dict[tuple[str, int], Mapping[str, Any]] = {}
    for run in report["source_runs"]:
        protocol, condition, seed_token = run["key"].split("|")
        if protocol != "p1":
            continue
        key = (condition, int(seed_token.removeprefix("seed-")))
        if key in runs:
            raise ValidationError(f"duplicate P1 source run: {key}")
        runs[key] = run
    expected = {
        (condition, seed)
        for condition in CANONICAL_CONDITIONS
        for seed in CANONICAL_SEEDS
    }
    if set(runs) != expected:
        raise ValidationError("P1 condition/seed source runs are incomplete")
    return runs


@dataclass(frozen=True)
class P1OOFBundle:
    prediction: np.ndarray
    source_cell_indices: np.ndarray
    source_gene_indices: np.ndarray


@dataclass(frozen=True)
class P1PPIReplay:
    edge_index: np.ndarray
    n_source_genes: int


def _hash_strings(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _load_bound_p1_cohort_manifest(
    run: Mapping[str, Any],
    run_path: Path,
    cohort_cells: Sequence[str],
    cohort_genes: Sequence[str],
    *,
    capture_store: _CaptureStore | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    path = run_path / "cohort-manifest.json"
    expected_sha = run.get("cohort_manifest_sha256")
    if not _valid_sha256(expected_sha):
        raise ValidationError(f"P1 cohort manifest binding is invalid: {path}")
    owned_store = capture_store is None
    store = _CaptureStore(prefix="depplm-p1-cohort-") if owned_store else capture_store
    assert store is not None
    try:
        payload, captured = store.capture_bytes(path, label="P1 cohort manifest")
    finally:
        if owned_store:
            store.__exit__(None, None, None)
    if captured.sha256 != expected_sha:
        raise ValidationError(f"P1 cohort manifest changed: {path}")
    manifest = _strict_json_bytes(payload, label="P1 cohort manifest")
    if not isinstance(manifest, dict):
        raise ValidationError("P1 cohort manifest must contain a JSON object")
    if set(manifest) != COHORT_MANIFEST_FIELDS:
        raise ValidationError("P1 cohort manifest fields are noncanonical")
    dimensions = {
        "n_cells": len(cohort_cells),
        "n_genes": len(cohort_genes),
    }
    for key, expected in dimensions.items():
        value = manifest.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise ValidationError(
                f"P1 cohort manifest {key} does not match the bound cohort"
            )
    expression_features = manifest.get("n_expression_features_raw")
    if (
        isinstance(expression_features, bool)
        or not isinstance(expression_features, int)
        or expression_features <= 0
    ):
        raise ValidationError("P1 cohort manifest n_expression_features_raw is invalid")
    n_string_edges = manifest.get("n_string_edges")
    if (
        isinstance(n_string_edges, bool)
        or not isinstance(n_string_edges, int)
        or n_string_edges < 0
    ):
        raise ValidationError("P1 cohort manifest n_string_edges is invalid")
    expected_hashes = {
        "cell_ids_sha256": _hash_strings(cohort_cells),
        "genes_sha256": _hash_strings(cohort_genes),
    }
    for key, expected_hash in expected_hashes.items():
        if manifest.get(key) != expected_hash:
            raise ValidationError(
                f"P1 cohort manifest {key} does not match the bound identifiers"
            )
    if not _valid_sha256(manifest.get("expression_gene_names_sha256")):
        raise ValidationError(
            "P1 cohort manifest expression_gene_names_sha256 is invalid"
        )

    sources: dict[str, np.ndarray] = {}
    for key, size in (
        ("source_cell_indices", len(cohort_cells)),
        ("source_gene_indices", len(cohort_genes)),
    ):
        raw = manifest.get(key)
        if (
            not isinstance(raw, list)
            or len(raw) != size
            or any(
                isinstance(value, bool) or not isinstance(value, int) for value in raw
            )
        ):
            raise ValidationError(f"P1 cohort manifest {key} is invalid")
        values = np.asarray(raw, dtype=np.int64)
        if not np.array_equal(values, np.arange(size, dtype=np.int64)):
            raise ValidationError(
                f"P1 cohort manifest {key} is not the complete canonical source order"
            )
        sources[key] = values
    return (
        sources["source_cell_indices"],
        sources["source_gene_indices"],
        n_string_edges,
    )


def _validate_npz_member_inventory(
    path: Path, expected_arrays: Sequence[str], *, label: str
) -> None:
    expected = tuple(f"{name}.npy" for name in expected_arrays)
    try:
        with zipfile.ZipFile(path) as archive:
            observed = tuple(member.filename for member in archive.infolist())
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValidationError(f"{label} is not a valid NPZ archive") from exc
    if len(observed) != len(set(observed)):
        raise ValidationError(f"{label} contains duplicate ZIP members")
    if observed != expected:
        raise ValidationError(f"{label} member inventory/order is noncanonical")


def _load_p1_ppi_replay(
    ppi_path: Path,
    genes: Sequence[str],
    *,
    capture_store: _CaptureStore | None = None,
) -> P1PPIReplay:
    if capture_store is None:
        with _CaptureStore(prefix="depplm-p1-ppi-") as store:
            return _load_p1_ppi_replay(ppi_path, genes, capture_store=store)
    captured = capture_store.capture(ppi_path, label="PPI archive")
    _validate_npz_member_inventory(captured.path, PPI_ARRAYS, label="PPI archive")
    try:
        with np.load(captured.path, allow_pickle=False) as archive:
            if tuple(archive.files) != PPI_ARRAYS:
                raise ValidationError("PPI archive fields are noncanonical")
            edge_index = np.asarray(archive["edge_index"])
            edge_weight = np.asarray(archive["edge_weight"])
            nodes = np.asarray(archive["nodes"])
    except ValidationError:
        raise
    except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile) as exc:
        raise ValidationError("PPI archive cannot be replayed") from exc
    if (
        edge_index.dtype != np.dtype(np.int64)
        or edge_index.ndim != 2
        or edge_index.shape[1] != 2
    ):
        raise ValidationError("PPI edge_index must be an int64 [n,2] array")
    if (
        edge_weight.ndim != 1
        or edge_weight.shape[0] != edge_index.shape[0]
        or not np.issubdtype(edge_weight.dtype, np.floating)
        or not np.isfinite(edge_weight).all()
    ):
        raise ValidationError("PPI edge_weight is invalid")
    if nodes.ndim != 1 or not np.array_equal(nodes.astype(str), np.asarray(genes)):
        raise ValidationError("PPI nodes do not exactly match cohort genes.txt")
    if edge_index.size and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= len(genes)
    ):
        raise ValidationError("PPI edge_index is out of range")
    return P1PPIReplay(edge_index.copy(), len(genes))


def _run_registered_subgraph(
    edge_index: np.ndarray, selected_genes: np.ndarray, n_source_genes: int
) -> np.ndarray:
    try:
        from run_corrected_experiments import _subgraph

        replayed = _subgraph(edge_index, selected_genes, n_source_genes)
    except Exception as exc:
        raise ValidationError(
            "run_corrected_experiments._subgraph replay failed"
        ) from exc
    replayed = np.asarray(replayed)
    if replayed.dtype != np.dtype(np.int64) or (
        replayed.ndim != 2 or replayed.shape[1] != 2
    ):
        raise ValidationError("PPI subgraph replay returned a noncanonical edge array")
    return replayed


def _validate_p1_ppi_replay(
    run: Mapping[str, Any],
    run_path: Path,
    records: Mapping[str, Mapping[str, Any]],
    source_gene_indices: np.ndarray,
    cohort_n_edges: int,
    replay: P1PPIReplay,
    *,
    capture_store: _CaptureStore,
) -> None:
    replayed = _run_registered_subgraph(
        replay.edge_index, source_gene_indices, replay.n_source_genes
    )
    expected_n_edges = int(replayed.shape[0])
    if cohort_n_edges != expected_n_edges:
        raise ValidationError(
            "P1 cohort n_string_edges differs from PPI subgraph replay: "
            f"observed={cohort_n_edges}, expected={expected_n_edges}"
        )
    for fold in P1_FOLDS:
        relative = f"{fold}-result.json"
        artifact = records.get(relative)
        if not isinstance(artifact, Mapping):
            raise ValidationError(
                f"P1 graph evidence is missing for {run.get('key')}: {relative}"
            )
        path = _safe_child(run_path, relative)
        payload, _captured = capture_store.capture_bytes(
            path,
            label=f"{run.get('key')}.{relative}",
            expected_record=artifact,
        )
        result = _strict_json_bytes(payload, label="P1 fold result")
        if not isinstance(result, dict):
            raise ValidationError("P1 fold result must contain a JSON object")
        preprocessing = result.get("preprocessing")
        if not isinstance(preprocessing, Mapping):
            raise ValidationError(f"P1 preprocessing evidence is invalid in {path}")
        for stage in ("inner_train_fit", "outer_train_refit"):
            stage_record = preprocessing.get(stage)
            graph = (
                stage_record.get("graph") if isinstance(stage_record, Mapping) else None
            )
            observed = graph.get("n_edges") if isinstance(graph, Mapping) else None
            if (
                isinstance(observed, bool)
                or not isinstance(observed, int)
                or observed != expected_n_edges
            ):
                raise ValidationError(
                    f"P1 {stage}.graph.n_edges differs from PPI subgraph replay "
                    f"in {path}: observed={observed!r}, expected={expected_n_edges}"
                )


def _load_p1_oof_bundle(
    run: Mapping[str, Any],
    cohort_cells: Sequence[str],
    cohort_genes: Sequence[str],
    *,
    workspace_root: Path | None = None,
    recorded_root: Path | None = None,
    ppi_replay: P1PPIReplay | None = None,
    capture_store: _CaptureStore | None = None,
) -> P1OOFBundle:
    """Reconstruct one P1 matrix with exact-once held-out coverage."""
    if capture_store is None:
        with _CaptureStore(prefix="depplm-p1-oof-") as store:
            return _load_p1_oof_bundle(
                run,
                cohort_cells,
                cohort_genes,
                workspace_root=workspace_root,
                recorded_root=recorded_root,
                ppi_replay=ppi_replay,
                capture_store=store,
            )
    run_path = _resolve_campaign_run_path(
        run.get("path"),
        workspace_root=workspace_root,
        recorded_root=recorded_root,
    )
    (
        cohort_source_cells,
        cohort_source_genes,
        cohort_n_edges,
    ) = _load_bound_p1_cohort_manifest(
        run,
        run_path,
        cohort_cells,
        cohort_genes,
        capture_store=capture_store,
    )
    records = {
        artifact["path"]: artifact
        for artifact in run["fold_artifacts"]
        if isinstance(artifact, dict) and isinstance(artifact.get("path"), str)
    }
    expected_paths = {f"{fold}-predictions.npz" for fold in P1_FOLDS}
    prediction_paths = {path for path in records if path.endswith("-predictions.npz")}
    if prediction_paths != expected_paths:
        raise ValidationError(
            f"P1 prediction artifacts are incomplete for {run.get('key')}"
        )
    if ppi_replay is not None:
        _validate_p1_ppi_replay(
            run,
            run_path,
            records,
            cohort_source_genes,
            cohort_n_edges,
            ppi_replay,
            capture_store=capture_store,
        )
    prediction: np.ndarray = np.full(
        (len(cohort_cells), len(cohort_genes)), np.nan, dtype=np.float32
    )
    coverage: np.ndarray = np.zeros(len(cohort_cells), dtype=np.int8)
    for fold in P1_FOLDS:
        relative = f"{fold}-predictions.npz"
        artifact = records[relative]
        path = _safe_child(run_path, relative)
        captured = capture_store.capture(
            path,
            label=f"P1 prediction artifact {run.get('key')}.{relative}",
            expected_record=artifact,
        )
        _validate_npz_member_inventory(
            captured.path, PREDICTION_ARRAYS, label="P1 prediction archive"
        )
        try:
            with np.load(captured.path, allow_pickle=False) as archive:
                if tuple(archive.files) != PREDICTION_ARRAYS:
                    raise ValidationError(f"noncanonical arrays in {path}")
                arrays = {
                    name: np.asarray(archive[name]).copy() for name in archive.files
                }
        except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile) as exc:
            raise ValidationError(
                f"cannot load P1 prediction artifact: {path}"
            ) from exc
        indices = arrays["outer_test_indices"]
        values = arrays["prediction"]
        if indices.dtype != np.int64 or indices.ndim != 1:
            raise ValidationError(f"outer_test_indices is invalid in {path}")
        if len(indices) != len(np.unique(indices)):
            raise ValidationError(f"outer_test_indices contains duplicates in {path}")
        if np.any(indices < 0) or np.any(indices >= len(cohort_cells)):
            raise ValidationError(f"outer_test_indices is out of range in {path}")
        if values.dtype != np.float32 or values.shape != (
            len(indices),
            len(cohort_genes),
        ):
            raise ValidationError(f"prediction dtype/shape is invalid in {path}")
        if not np.isfinite(values).all():
            raise ValidationError(f"prediction contains non-finite values in {path}")
        train_mean = arrays["train_gene_mean"]
        if (
            train_mean.dtype != np.float32
            or train_mean.shape != (len(cohort_genes),)
            or not np.isfinite(train_mean).all()
        ):
            raise ValidationError(f"train_gene_mean is invalid in {path}")
        expected_cell_ids = np.asarray(
            [cohort_cells[int(index)] for index in indices], dtype=str
        )
        if not np.array_equal(arrays["cell_ids"], expected_cell_ids):
            raise ValidationError(f"cell_ids do not match held-out indices in {path}")
        if not np.array_equal(arrays["gene_ids"], np.asarray(cohort_genes, dtype=str)):
            raise ValidationError(f"gene_ids do not match the bound cohort in {path}")
        current_genes = arrays["source_gene_indices"]
        if current_genes.dtype != np.int64 or current_genes.shape != (
            len(cohort_genes),
        ):
            raise ValidationError(f"source_gene_indices is invalid in {path}")
        if not np.array_equal(current_genes, cohort_source_genes):
            raise ValidationError(
                f"source_gene_indices do not match cohort-manifest.json in {path}"
            )
        current_cells = arrays["source_cell_indices"]
        if current_cells.dtype != np.int64 or current_cells.shape != indices.shape:
            raise ValidationError(f"source_cell_indices is invalid in {path}")
        if not np.array_equal(current_cells, cohort_source_cells[indices]):
            raise ValidationError(
                f"source_cell_indices do not match cohort-manifest.json in {path}"
            )
        prediction[indices] = values
        coverage[indices] += 1
    if not np.all(coverage == 1):
        raise ValidationError(
            f"P1 folds do not cover every cohort cell exactly once for {run.get('key')}"
        )
    return P1OOFBundle(
        prediction, cohort_source_cells.copy(), cohort_source_genes.copy()
    )


def load_p1_oof(
    run: Mapping[str, Any],
    cohort_cells: Sequence[str],
    cohort_genes: Sequence[str],
    *,
    workspace_root: Path | None = None,
    recorded_root: Path | None = None,
    ppi_path: Path | None = None,
) -> np.ndarray:
    """Return the exact-once P1 held-out matrix after identity validation."""
    ppi_replay = (
        _load_p1_ppi_replay(ppi_path, cohort_genes) if ppi_path is not None else None
    )
    return _load_p1_oof_bundle(
        run,
        cohort_cells,
        cohort_genes,
        workspace_root=workspace_root,
        recorded_root=recorded_root,
        ppi_replay=ppi_replay,
    ).prediction


def robust_mad(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValidationError("prediction matrix must be non-empty and two-dimensional")
    if not np.isfinite(values).all():
        raise ValidationError("prediction matrix contains non-finite values")
    medians = np.median(values, axis=0)
    return 1.4826 * np.median(np.abs(values - medians), axis=0)


def directional_tenth_percentile(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValidationError("prediction matrix is invalid for directional scoring")
    return -np.quantile(values, 0.10, axis=0)


def _average_rank_desc(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValidationError("rank values must be a finite one-dimensional array")
    order = np.argsort(-values, kind="mergesort")
    ranks: np.ndarray = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _read_tsv_bytes(payload: bytes, *, label: str) -> list[dict[str, str]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise ValidationError(f"TSV is not valid UTF-8: {label}") from exc
    with io.StringIO(text, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValidationError(f"TSV header is missing: {label}")
        return [dict(row) for row in reader]


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with _CaptureStore(prefix="depplm-tsv-") as store:
        payload, _captured = store.capture_bytes(path, label="TSV")
        return _read_tsv_bytes(payload, label=str(path))


def _read_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    with _CaptureStore(prefix="depplm-json-") as store:
        raw, _captured = store.capture_bytes(path, label=label)
    payload = _strict_json_bytes(raw, label=label)
    if not isinstance(payload, dict):
        raise ValidationError(f"{label} must contain a JSON object")
    return payload


def _manifest_file_records(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    raw_records = manifest.get("files")
    if not isinstance(raw_records, list):
        raise ValidationError("family release file inventory is missing")
    records: dict[str, dict[str, Any]] = {}
    for raw_record in raw_records:
        if not isinstance(raw_record, dict):
            raise ValidationError("family release file record is invalid")
        relative = raw_record.get("path")
        if not isinstance(relative, str):
            raise ValidationError("family release file path is invalid")
        normalized = relative.replace("\\", "/")
        if normalized != relative or normalized in records:
            raise ValidationError(
                "family release file paths are noncanonical or duplicated"
            )
        _safe_child(root, normalized)
        records[normalized] = dict(raw_record)
    return records


def _require_manifest_record(
    root: Path,
    records: Mapping[str, Mapping[str, Any]],
    relative: str,
    *,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    record = records.get(relative)
    if not isinstance(record, Mapping):
        raise ValidationError(f"family release lacks {relative}")
    path = _safe_child(root, relative)
    _verify_file_record(path, record, label=label)
    return path, dict(record)


def _same_file_record(left: Any, right: Mapping[str, Any]) -> bool:
    return isinstance(left, Mapping) and all(
        left.get(field) == right.get(field) for field in ("path", "bytes", "sha256")
    )


def _valid_git_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _read_family_units(path: Path) -> list[FamilyUnit]:
    units: list[FamilyUnit] = []
    seen: set[str] = set()
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if tuple(reader.fieldnames or ()) != FAMILY_UNITS_FIELDS:
                raise ValidationError("family_units.tsv header is noncanonical")
            for line_number, row in enumerate(reader, start=2):
                if None in row or any(
                    row.get(field) is None for field in FAMILY_UNITS_FIELDS
                ):
                    raise ValidationError(
                        f"family_units.tsv row {line_number} has the wrong column count"
                    )
                gene = row["gene"]
                accession = row["accession"]
                sequence_sha256 = row["sequence_sha256"]
                component_id = row["component_id"]
                try:
                    component_size = int(row["component_size"])
                except ValueError as exc:
                    raise ValidationError(
                        f"family_units.tsv row {line_number} has invalid component_size"
                    ) from exc
                if not gene or gene in seen:
                    raise ValidationError(
                        f"family_units.tsv has a blank or duplicate gene at row {line_number}"
                    )
                if not accession or not _valid_sha256(sequence_sha256):
                    raise ValidationError(
                        f"family_units.tsv row {line_number} has invalid sequence identity"
                    )
                if (
                    not component_id.startswith("fhc-")
                    or len(component_id) != 24
                    or any(
                        character not in "0123456789abcdef"
                        for character in component_id[4:]
                    )
                    or component_size < 1
                ):
                    raise ValidationError(
                        f"family_units.tsv row {line_number} has invalid component metadata"
                    )
                seen.add(gene)
                units.append(
                    FamilyUnit(
                        gene,
                        accession,
                        sequence_sha256,
                        component_id,
                        component_size,
                    )
                )
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ValidationError("family_units.tsv is unreadable") from exc
    if not units:
        raise ValidationError("family_units.tsv contains no genes")
    return units


def _validate_family_fasta(
    fasta_path: Path,
    expected_genes: Sequence[str],
    units_by_gene: Mapping[str, FamilyUnit],
) -> int:
    record_index = 0
    header: str | None = None
    sequence_parts: list[str] = []

    def finish_record() -> None:
        nonlocal header, sequence_parts, record_index
        if header is None:
            return
        tokens = header.split()
        if not tokens or "|" not in tokens[0]:
            raise ValidationError("family FASTA header lacks gene|accession")
        gene, accession = tokens[0].split("|", 1)
        sequence = "".join(sequence_parts).upper()
        if (
            not gene
            or not accession
            or not sequence
            or any(character.isspace() for character in sequence)
        ):
            raise ValidationError(
                f"family FASTA record is invalid: {gene or '<blank>'}"
            )
        if record_index >= len(expected_genes) or gene != expected_genes[record_index]:
            raise ValidationError("family FASTA and genes.txt identifiers/order differ")
        unit = units_by_gene.get(gene)
        if unit is None or unit.accession != accession:
            raise ValidationError(f"family FASTA accession differs for {gene}")
        try:
            sequence_sha256 = hashlib.sha256(sequence.encode("ascii")).hexdigest()
        except UnicodeEncodeError as exc:
            raise ValidationError(
                f"family FASTA sequence is non-ASCII for {gene}"
            ) from exc
        if sequence_sha256 != unit.sequence_sha256:
            raise ValidationError(f"family FASTA sequence hash differs for {gene}")
        record_index += 1
        header = None
        sequence_parts = []

    try:
        with fasta_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    finish_record()
                    header = line[1:]
                elif header is None:
                    raise ValidationError(
                        f"family FASTA sequence precedes its header at line {line_number}"
                    )
                else:
                    sequence_parts.append(line)
        finish_record()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValidationError("family FASTA is unreadable") from exc
    if record_index != len(expected_genes):
        raise ValidationError("family FASTA and genes.txt record counts differ")
    return record_index


def replay_family_covariate_semantics(
    family_path: Path,
    hits_path: Path,
    *,
    genes_path: Path,
    fasta_path: Path,
    identity_percent: float,
    minimum_bidirectional_coverage: float,
    expected_gene_count: int,
    expected_qualifying_edge_count: int,
    expected_component_count: int,
) -> dict[str, Any]:
    """Rebuild MMseqs2 components from streamed canonical hits and bound inputs."""
    units = _read_family_units(family_path)
    expected_genes = _read_unique_lines(genes_path)
    observed_genes = tuple(unit.gene for unit in units)
    if observed_genes != expected_genes:
        raise ValidationError("family_units.tsv and genes.txt identifiers/order differ")
    if len(units) != expected_gene_count:
        raise ValidationError(
            f"family semantic gene count changed: {len(units)} != {expected_gene_count}"
        )
    units_by_gene = {unit.gene: unit for unit in units}
    fasta_records = _validate_family_fasta(fasta_path, expected_genes, units_by_gene)

    gene_to_index = {gene: index for index, gene in enumerate(expected_genes)}
    parent = list(range(len(expected_genes)))
    sizes = [1] * len(expected_genes)

    def find(index: int) -> int:
        root = index
        while root != parent[root]:
            root = parent[root]
        while index != root:
            next_index = parent[index]
            parent[index] = root
            index = next_index
        return root

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return
        if (sizes[root_left], expected_genes[root_left]) < (
            sizes[root_right],
            expected_genes[root_right],
        ):
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        sizes[root_left] += sizes[root_right]

    qualifying_edges = 0
    hit_rows = 0
    previous_pair: tuple[str, str] | None = None
    try:
        with gzip.open(hits_path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if tuple(reader.fieldnames or ()) != FAMILY_HITS_FIELDS:
                raise ValidationError("canonical MMseqs2 hits header is noncanonical")
            for line_number, row in enumerate(reader, start=2):
                hit_rows += 1
                if None in row or any(
                    row.get(field) is None for field in FAMILY_HITS_FIELDS
                ):
                    raise ValidationError(
                        f"canonical MMseqs2 hits row {line_number} has the wrong column count"
                    )
                gene_a = row["gene_a"]
                gene_b = row["gene_b"]
                pair = (gene_a, gene_b)
                if (
                    gene_a not in gene_to_index
                    or gene_b not in gene_to_index
                    or gene_a >= gene_b
                ):
                    raise ValidationError(
                        f"canonical MMseqs2 hits row {line_number} has invalid gene identities"
                    )
                if previous_pair is not None and pair <= previous_pair:
                    raise ValidationError(
                        "canonical MMseqs2 hits are duplicated or not strictly sorted"
                    )
                previous_pair = pair
                try:
                    pident = float(row["pident_percent"])
                    coverage_a = float(row["coverage_a"])
                    coverage_b = float(row["coverage_b"])
                    alignment_length = int(row["alignment_length"])
                    evalue = float(row["evalue"])
                    bits = float(row["bits"])
                except ValueError as exc:
                    raise ValidationError(
                        f"canonical MMseqs2 hits row {line_number} is nonnumeric"
                    ) from exc
                if (
                    not all(
                        math.isfinite(value)
                        for value in (pident, coverage_a, coverage_b, evalue, bits)
                    )
                    or not 0.0 <= pident <= 100.0
                    or not 0.0 <= coverage_a <= 1.0
                    or not 0.0 <= coverage_b <= 1.0
                    or alignment_length < 1
                    or evalue < 0.0
                ):
                    raise ValidationError(
                        f"canonical MMseqs2 hits row {line_number} has invalid evidence"
                    )
                if (
                    pident + 1e-9 >= identity_percent
                    and coverage_a + 1e-9 >= minimum_bidirectional_coverage
                    and coverage_b + 1e-9 >= minimum_bidirectional_coverage
                ):
                    qualifying_edges += 1
                    union(gene_to_index[gene_a], gene_to_index[gene_b])
    except (OSError, EOFError, UnicodeDecodeError, csv.Error) as exc:
        raise ValidationError("canonical MMseqs2 hits are unreadable") from exc

    if qualifying_edges != expected_qualifying_edge_count:
        raise ValidationError(
            "family semantic qualifying-edge count changed: "
            f"{qualifying_edges} != {expected_qualifying_edge_count}"
        )
    members_by_root: dict[int, list[str]] = defaultdict(list)
    for index, gene in enumerate(expected_genes):
        members_by_root[find(index)].append(gene)
    if len(members_by_root) != expected_component_count:
        raise ValidationError(
            "family semantic component count changed: "
            f"{len(members_by_root)} != {expected_component_count}"
        )
    for members in members_by_root.values():
        members.sort()
        component_id = (
            "fhc-" + hashlib.sha256("\0".join(members).encode("utf-8")).hexdigest()[:20]
        )
        component_size = len(members)
        for gene in members:
            unit = units_by_gene[gene]
            if (
                unit.component_id != component_id
                or unit.component_size != component_size
            ):
                raise ValidationError(
                    f"family component identity/size differs for {gene}"
                )
    return {
        "gene_count": len(units),
        "fasta_record_count": fasta_records,
        "canonical_hit_rows": hit_rows,
        "qualifying_edge_count": qualifying_edges,
        "component_count": len(members_by_root),
        "rule": {
            "identity_percent": identity_percent,
            "minimum_bidirectional_coverage": minimum_bidirectional_coverage,
            "component_definition": FAMILY_COMPONENT_DEFINITION,
        },
    }


def _capture_family_validation_inputs(
    store: _CaptureStore,
    family_path: Path,
    release_manifest_path: Path,
    *,
    genes_path: Path,
    fasta_path: Path,
) -> dict[str, _CapturedFile]:
    release_manifest_path = Path(os.path.abspath(os.fspath(release_manifest_path)))
    release_root = release_manifest_path.parent
    expected_family = release_root / FAMILY_UNITS_RELATIVE
    if os.path.normcase(os.path.abspath(family_path)) != os.path.normcase(
        os.path.abspath(expected_family)
    ):
        raise ValidationError(
            "family covariate path does not match the registered family release"
        )
    mirror = Path("family-release")
    captures = {
        "release": store.capture(
            release_manifest_path,
            label="family release manifest",
            destination_relative=mirror / release_manifest_path.name,
        ),
        "family": store.capture(
            expected_family,
            label="family covariate from release",
            destination_relative=mirror / FAMILY_UNITS_RELATIVE,
        ),
        "summary": store.capture(
            release_root / FAMILY_SUMMARY_RELATIVE,
            label="family split summary",
            destination_relative=mirror / FAMILY_SUMMARY_RELATIVE,
        ),
        "source": store.capture(
            release_root / FAMILY_SOURCE_RELATIVE,
            label="family source manifest",
            destination_relative=mirror / FAMILY_SOURCE_RELATIVE,
        ),
        "hits": store.capture(
            release_root / FAMILY_HITS_RELATIVE,
            label="family canonical MMseqs2 hits",
            destination_relative=mirror / FAMILY_HITS_RELATIVE,
        ),
        "genes": store.capture(
            genes_path,
            label="family source genes",
            destination_relative=Path("family-inputs/genes.txt"),
        ),
        "fasta": store.capture(
            fasta_path,
            label="family source FASTA",
            destination_relative=Path("family-inputs/proteins.fasta"),
        ),
    }
    return captures


def validate_family_covariate_provenance(
    family_path: Path,
    release_manifest_path: Path,
    *,
    genes_path: Path,
    fasta_path: Path,
    capture_store: _CaptureStore | None = None,
    captured_inputs: Mapping[str, _CapturedFile] | None = None,
) -> dict[str, Any]:
    """Bind the family-size covariate to its exact registered MMseqs2 release."""

    if captured_inputs is None:
        owned_store = capture_store is None
        store = (
            _CaptureStore(prefix="depplm-family-validation-")
            if owned_store
            else capture_store
        )
        assert store is not None
        try:
            captures = _capture_family_validation_inputs(
                store,
                family_path,
                release_manifest_path,
                genes_path=genes_path,
                fasta_path=fasta_path,
            )
            return validate_family_covariate_provenance(
                captures["family"].path,
                captures["release"].path,
                genes_path=captures["genes"].path,
                fasta_path=captures["fasta"].path,
                capture_store=store,
                captured_inputs=captures,
            )
        finally:
            if owned_store:
                store.__exit__(None, None, None)

    release_manifest_path = release_manifest_path.resolve()
    release_root = release_manifest_path.parent
    release = _read_json_mapping(release_manifest_path, label="family release manifest")
    if release.get("schema") != FAMILY_RELEASE_SCHEMA:
        raise ValidationError("family release schema is invalid")
    if release.get("status") != "complete":
        raise ValidationError("family release is not complete")
    if release.get("scientific_use") != "eligible_for_family_holdout_split_use":
        raise ValidationError("family release is not eligible for scientific use")
    if release.get("main_identity_percentage") != FAMILY_IDENTITY_PERCENT:
        raise ValidationError("family release main identity threshold changed")

    records = _manifest_file_records(release_root, release)
    selected_family_path, family_record = _require_manifest_record(
        release_root,
        records,
        FAMILY_UNITS_RELATIVE,
        label="family covariate from release",
    )
    if family_path.resolve() != selected_family_path:
        raise ValidationError(
            "family covariate path does not match the registered family release"
        )
    summary_path, summary_record = _require_manifest_record(
        release_root,
        records,
        FAMILY_SUMMARY_RELATIVE,
        label="family split summary",
    )
    source_path, _ = _require_manifest_record(
        release_root,
        records,
        FAMILY_SOURCE_RELATIVE,
        label="family source manifest",
    )
    hits_path, hits_record = _require_manifest_record(
        release_root,
        records,
        FAMILY_HITS_RELATIVE,
        label="family canonical MMseqs2 hits",
    )

    thresholds = release.get("thresholds")
    if not isinstance(thresholds, list):
        raise ValidationError("family release threshold inventory is missing")
    selected_thresholds = [
        item
        for item in thresholds
        if isinstance(item, Mapping)
        and item.get("identity_percent") == FAMILY_IDENTITY_PERCENT
    ]
    if len(selected_thresholds) != 1:
        raise ValidationError("family release must contain one 30% identity threshold")
    if not _same_file_record(selected_thresholds[0].get("summary"), summary_record):
        raise ValidationError("family release threshold summary binding changed")

    summary = _read_json_mapping(summary_path, label="family split summary")
    if summary.get("schema") != FAMILY_RELEASE_SCHEMA:
        raise ValidationError("family split summary schema is invalid")
    if summary.get("identity_percent") != FAMILY_IDENTITY_PERCENT:
        raise ValidationError("family split identity threshold changed")
    if (
        summary.get("minimum_bidirectional_coverage")
        != FAMILY_MIN_BIDIRECTIONAL_COVERAGE
    ):
        raise ValidationError("family split coverage threshold changed")
    if summary.get("component_definition") != FAMILY_COMPONENT_DEFINITION:
        raise ValidationError("family component definition changed")
    if not _same_file_record(summary.get("family_units"), family_record):
        raise ValidationError("family split summary does not bind the family covariate")
    validation = summary.get("validation")
    if not isinstance(validation, Mapping):
        raise ValidationError("family split validation block is missing")
    for key in (
        "cross_component_qualifying_edges",
        "cross_fold_qualifying_edges",
        "exact_duplicate_fold_violations",
    ):
        if validation.get(key) != 0:
            raise ValidationError(f"family split validation failed: {key}")
    expected_summary_counts = {
        "gene_count": FAMILY_EXPECTED_GENE_COUNT,
        "component_count": FAMILY_EXPECTED_COMPONENT_COUNT,
    }
    for key, expected in expected_summary_counts.items():
        if summary.get(key) != expected:
            raise ValidationError(f"family split summary count changed: {key}")
    if validation.get("qualifying_edges") != FAMILY_EXPECTED_QUALIFYING_EDGE_COUNT:
        raise ValidationError("family split qualifying-edge count changed")

    source = _read_json_mapping(source_path, label="family source manifest")
    if source.get("schema") != FAMILY_RELEASE_SCHEMA:
        raise ValidationError("family source manifest schema is invalid")
    tool = source.get("tool")
    if not isinstance(tool, Mapping) or tool.get("name") != "MMseqs2":
        raise ValidationError("family source tool is not MMseqs2")
    tool_release = tool.get("release")
    if tool_release != FAMILY_MMSEQS2_RELEASE:
        raise ValidationError("family source MMseqs2 release changed")
    reported_commit = tool.get("reported_commit")
    if not _valid_git_commit(reported_commit):
        raise ValidationError("family source MMseqs2 commit is invalid")
    if reported_commit != FAMILY_MMSEQS2_COMMIT:
        raise ValidationError("family source MMseqs2 commit changed")
    for key in ("binary", "release_archive"):
        record = tool.get(key)
        if not isinstance(record, Mapping):
            raise ValidationError(f"family source MMseqs2 {key} record is missing")
        record_path = record.get("path")
        record_bytes = record.get("bytes")
        if (
            not isinstance(record_path, str)
            or not record_path
            or isinstance(record_bytes, bool)
            or not isinstance(record_bytes, int)
            or record_bytes <= 0
            or not _valid_sha256(record.get("sha256"))
        ):
            raise ValidationError(f"family source MMseqs2 {key} record is invalid")

    search = source.get("search")
    if not isinstance(search, Mapping):
        raise ValidationError("family source MMseqs2 search metadata is missing")
    expected_search: dict[str, int | float] = {
        "main_identity_percentage": FAMILY_IDENTITY_PERCENT,
        "minimum_bidirectional_coverage": FAMILY_MIN_BIDIRECTIONAL_COVERAGE,
        "coverage_mode": 0,
        "alignment_mode": 3,
    }
    for key, expected_value in expected_search.items():
        if search.get(key) != expected_value:
            raise ValidationError(f"family source MMseqs2 parameter changed: {key}")
    identities = search.get("registered_identity_percentages")
    if not isinstance(identities, list) or FAMILY_IDENTITY_PERCENT not in identities:
        raise ValidationError(
            "family source lacks the registered 30% identity threshold"
        )
    if not _same_file_record(source.get("canonical_hits"), hits_record):
        raise ValidationError(
            "family source manifest does not bind canonical MMseqs2 hits"
        )

    source_inputs = source.get("inputs")
    if not isinstance(source_inputs, Mapping):
        raise ValidationError("family source input bindings are missing")
    for label, current_path in (("genes", genes_path), ("fasta", fasta_path)):
        record = source_inputs.get(label)
        if not isinstance(record, Mapping):
            raise ValidationError(f"family source {label} binding is missing")
        _verify_file_record(
            current_path.resolve(), record, label=f"family source {label}"
        )

    trusted_hashes = {
        FAMILY_UNITS_RELATIVE: FAMILY_UNITS_SHA256,
        FAMILY_SUMMARY_RELATIVE: FAMILY_SPLIT_SUMMARY_SHA256,
        FAMILY_SOURCE_RELATIVE: FAMILY_SOURCE_MANIFEST_SHA256,
        FAMILY_HITS_RELATIVE: FAMILY_HOMOLOGY_HITS_SHA256,
    }
    for relative, expected_sha256 in trusted_hashes.items():
        if records[relative].get("sha256") != expected_sha256:
            raise ValidationError(
                f"family release does not bind the canonical artifact: {relative}"
            )
    release_manifest_record = captured_inputs["release"].record()
    if release_manifest_record["sha256"] != FAMILY_RELEASE_MANIFEST_SHA256:
        raise ValidationError("family release manifest is not the canonical release")

    semantic_replay = replay_family_covariate_semantics(
        selected_family_path,
        hits_path,
        genes_path=genes_path,
        fasta_path=fasta_path,
        identity_percent=FAMILY_IDENTITY_PERCENT,
        minimum_bidirectional_coverage=FAMILY_MIN_BIDIRECTIONAL_COVERAGE,
        expected_gene_count=FAMILY_EXPECTED_GENE_COUNT,
        expected_qualifying_edge_count=FAMILY_EXPECTED_QUALIFYING_EDGE_COUNT,
        expected_component_count=FAMILY_EXPECTED_COMPONENT_COUNT,
    )

    return {
        "verification_status": "verified",
        "schema": FAMILY_RELEASE_SCHEMA,
        "tool": {
            "name": "MMseqs2",
            "release": tool_release,
            "reported_commit": reported_commit,
        },
        "identity_percent": FAMILY_IDENTITY_PERCENT,
        "minimum_bidirectional_coverage": FAMILY_MIN_BIDIRECTIONAL_COVERAGE,
        "component_definition": FAMILY_COMPONENT_DEFINITION,
        "family_release_manifest": release_manifest_record,
        "family_source_manifest": captured_inputs["source"].record(),
        "family_split_summary": captured_inputs["summary"].record(),
        "family_homology_hits": captured_inputs["hits"].record(),
        "semantic_replay": semantic_replay,
    }


def load_outcomes(
    snapshot_dir: Path,
    *,
    capture_store: _CaptureStore | None = None,
    expected_record: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    if capture_store is None:
        with _CaptureStore(prefix="depplm-outcomes-") as store:
            return load_outcomes(
                snapshot_dir,
                capture_store=store,
                expected_record=expected_record,
            )
    payload, _captured = capture_store.capture_bytes(
        snapshot_dir / "target_outcomes.tsv",
        label="snapshot target outcomes",
        expected_record=expected_record,
    )
    rows = _read_tsv_bytes(payload, label="snapshot target outcomes")
    outcomes: dict[str, dict[str, Any]] = {}
    required = {
        "approved_symbol",
        "target_id",
        "clinical_oncology_phase2_plus",
        "clinical_oncology_withdrawal",
        "tractable_any",
    }
    for row in rows:
        if not required.issubset(row):
            raise ValidationError("target_outcomes.tsv columns are incomplete")
        symbol = row["approved_symbol"]
        if not symbol or symbol in outcomes:
            raise ValidationError("snapshot outcome symbols are empty or duplicated")
        binary: dict[str, int] = {}
        for key in (
            "clinical_oncology_phase2_plus",
            "clinical_oncology_withdrawal",
            "tractable_any",
        ):
            if row[key] not in {"0", "1"}:
                raise ValidationError(f"snapshot endpoint {key} is not binary")
            binary[key] = int(row[key])
        outcomes[symbol] = {"target_id": row["target_id"], **binary}
    return outcomes


def load_covariates(
    genes: Sequence[str],
    ppi_path: Path,
    family_path: Path,
    fasta_path: Path,
    *,
    capture_store: _CaptureStore | None = None,
    captured_inputs: Mapping[str, _CapturedFile] | None = None,
) -> tuple[dict[str, dict[str, int]], dict[str, Any]]:
    if capture_store is None:
        with _CaptureStore(prefix="depplm-covariates-") as store:
            return load_covariates(
                genes,
                ppi_path,
                family_path,
                fasta_path,
                capture_store=store,
                captured_inputs=captured_inputs,
            )
    captures = dict(captured_inputs or {})
    for role, path in (
        ("ppi", ppi_path),
        ("family", family_path),
        ("fasta", fasta_path),
    ):
        if role not in captures:
            captures[role] = capture_store.capture(path, label=f"{role} covariate")
    ppi_capture = captures["ppi"]
    family_capture = captures["family"]
    fasta_capture = captures["fasta"]
    gene_set = set(genes)
    if len(gene_set) != len(genes):
        raise ValidationError("cohort genes are duplicated")
    try:
        _validate_npz_member_inventory(
            ppi_capture.path, PPI_ARRAYS, label="PPI covariate archive"
        )
        with np.load(ppi_capture.path, allow_pickle=False) as archive:
            if "nodes" not in archive.files or "edge_index" not in archive.files:
                raise ValidationError("PPI archive lacks nodes or edge_index")
            nodes = np.asarray(archive["nodes"]).astype(str)
            edges = np.asarray(archive["edge_index"])
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        raise ValidationError("PPI archive cannot be read") from exc
    if nodes.ndim != 1 or len(nodes) != len(set(nodes.tolist())):
        raise ValidationError("PPI nodes are not unique")
    if (
        edges.ndim != 2
        or edges.shape[1] != 2
        or not np.issubdtype(edges.dtype, np.integer)
    ):
        raise ValidationError("PPI edge_index must be an integer [n,2] array")
    if edges.size and (int(edges.min()) < 0 or int(edges.max()) >= len(nodes)):
        raise ValidationError("PPI edge_index is out of range")
    undirected = {
        (min(int(left), int(right)), max(int(left), int(right)))
        for left, right in edges.tolist()
        if int(left) != int(right)
    }
    degree: Counter[str] = Counter()
    for left, right in undirected:
        degree[nodes[left]] += 1
        degree[nodes[right]] += 1

    family_rows = _read_tsv_bytes(
        family_capture.path.read_bytes(), label="family covariate"
    )
    family_size: dict[str, int] = {}
    for row in family_rows:
        gene = row.get("gene", "")
        if not gene or gene in family_size:
            raise ValidationError("family covariate genes are empty or duplicated")
        try:
            size = int(row.get("component_size", ""))
        except ValueError as exc:
            raise ValidationError("family component_size is not integer") from exc
        if size < 1:
            raise ValidationError("family component_size must be positive")
        family_size[gene] = size

    lengths: dict[str, int] = {}
    current: str | None = None
    chunks: list[str] = []
    try:
        fasta_text = fasta_capture.path.read_bytes().decode("utf-8")
    except UnicodeError as exc:
        raise ValidationError("FASTA covariate is not valid UTF-8") from exc
    for raw_line in fasta_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current is not None:
                if current in lengths:
                    raise ValidationError(f"duplicate FASTA gene: {current}")
                lengths[current] = len("".join(chunks))
            current = line[1:].split("|", 1)[0].strip()
            if not current:
                raise ValidationError("FASTA contains an empty gene header")
            chunks = []
        else:
            if current is None:
                raise ValidationError("FASTA sequence appears before a header")
            chunks.append(line)
    if current is not None:
        if current in lengths:
            raise ValidationError(f"duplicate FASTA gene: {current}")
        lengths[current] = len("".join(chunks))
    missing = [
        gene
        for gene in genes
        if gene not in set(nodes.tolist())
        or gene not in family_size
        or gene not in lengths
    ]
    if missing:
        raise ValidationError(
            f"covariates are incomplete for {len(missing)} cohort genes; first={missing[:5]}"
        )
    covariates = {
        gene: {
            "ppi_degree": int(degree.get(gene, 0)),
            "family_size": int(family_size[gene]),
            "protein_length": int(lengths[gene]),
        }
        for gene in genes
    }
    return covariates, {
        "ppi": ppi_capture.record(),
        "family": family_capture.record(),
        "fasta": fasta_capture.record(),
        "ppi_unique_undirected_edges": len(undirected),
    }


def _quantile_bin(values: np.ndarray, bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValidationError("binning covariate must be finite and one-dimensional")
    if bins < 1:
        raise ValidationError("number of bins must be positive")
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1))[1:-1])
    return np.searchsorted(edges, values, side="right").astype(np.int16)


def make_strata(
    records: Sequence[Mapping[str, Any]], *, include_tractability: bool
) -> list[tuple[int, ...]]:
    if not records:
        raise ValidationError("cannot construct strata for an empty universe")
    degrees = np.asarray([int(row["ppi_degree"]) for row in records], dtype=np.int64)
    lengths = np.asarray(
        [int(row["protein_length"]) for row in records], dtype=np.int64
    )
    families = np.asarray([int(row["family_size"]) for row in records], dtype=np.int64)
    if np.any(degrees < 0) or np.any(lengths < 1) or np.any(families < 1):
        raise ValidationError("matching covariates contain impossible values")
    degree_bins: np.ndarray = np.zeros(len(records), dtype=np.int16)
    positive = degrees > 0
    if np.any(positive):
        degree_bins[positive] = 1 + _quantile_bin(degrees[positive], 4)
    length_bins = _quantile_bin(lengths, 5)
    family_bins = np.select(
        [families == 1, families <= 3, families <= 10], [0, 1, 2], default=3
    ).astype(np.int16)
    strata = []
    for index, row in enumerate(records):
        parts = [
            int(degree_bins[index]),
            int(family_bins[index]),
            int(length_bins[index]),
        ]
        if include_tractability:
            value = int(row["tractable_any"])
            if value not in {0, 1}:
                raise ValidationError("tractability matching covariate is not binary")
            parts.append(value)
        strata.append(tuple(parts))
    return strata


def matched_permutation_test(
    selected: Sequence[bool] | np.ndarray,
    outcome: Sequence[int] | np.ndarray,
    strata: Sequence[tuple[int, ...]],
    *,
    n_permutations: int = MIN_PERMUTATIONS,
    seed: int = DEFAULT_PERMUTATION_SEED,
) -> dict[str, Any]:
    """Fixed-stratum hypergeometric permutation with a +1-corrected empirical Monte Carlo P value."""
    if n_permutations < MIN_PERMUTATIONS:
        raise ValidationError(
            f"matched permutation requires at least {MIN_PERMUTATIONS}"
        )
    selected_array = np.asarray(selected, dtype=bool)
    outcome_array = np.asarray(outcome, dtype=np.int8)
    if (
        selected_array.ndim != 1
        or outcome_array.shape != selected_array.shape
        or len(strata) != len(selected_array)
        or not np.isin(outcome_array, [0, 1]).all()
    ):
        raise ValidationError("matched permutation arrays are incompatible")
    if not np.any(selected_array):
        raise ValidationError("matched permutation selection is empty")
    grouped: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, stratum in enumerate(strata):
        grouped[tuple(stratum)].append(index)
    parameters = []
    for stratum, indices in sorted(grouped.items()):
        index_array = np.asarray(indices, dtype=np.int64)
        n_selected = int(np.sum(selected_array[index_array]))
        if n_selected == 0:
            continue
        n_positive = int(np.sum(outcome_array[index_array]))
        parameters.append((stratum, len(indices), n_positive, n_selected))
    observed = int(np.sum(outcome_array[selected_array]))
    rng = np.random.default_rng(int(seed))
    null: np.ndarray = np.zeros(int(n_permutations), dtype=np.int32)
    for _, size, positives, draws in parameters:
        null += rng.hypergeometric(
            positives, size - positives, draws, size=n_permutations
        )
    p_value = (1 + int(np.sum(null >= observed))) / (int(n_permutations) + 1)
    expected = float(
        sum(draws * positives / size for _, size, positives, draws in parameters)
    )
    selected_n = int(np.sum(selected_array))
    outcome_positive_n = int(np.sum(outcome_array))
    unselected_n = len(selected_array) - selected_n
    unselected_positive_n = outcome_positive_n - observed
    effect_size = _mantel_haenszel_effect_size(selected_array, outcome_array, grouped)
    return {
        "selected_n": selected_n,
        "unselected_n": unselected_n,
        "outcome_positive_n": outcome_positive_n,
        "unselected_outcome_positive_n": unselected_positive_n,
        "observed_overlap": observed,
        "expected_overlap": expected,
        "fold_enrichment": (float(observed / expected) if expected > 0 else None),
        "selected_event_rate": float(observed / selected_n),
        "matched_expected_event_rate": float(expected / selected_n),
        "matched_excess_proportion": float((observed - expected) / selected_n),
        **effect_size,
        "p_value": float(p_value),
        "n_permutations": int(n_permutations),
        "n_selected_strata": len(parameters),
        "seed": int(seed),
        "null_rule": "sum of fixed-stratum hypergeometric draws; greater-or-equal tail",
        "p_value_rule": "(1 + null_ge_observed) / (n_permutations + 1)",
    }


def _mantel_haenszel_effect_size(
    selected: np.ndarray,
    outcome: np.ndarray,
    grouped: Mapping[tuple[int, ...], Sequence[int]],
) -> dict[str, Any]:
    """Return a common odds ratio and nominal 95% RBG confidence interval."""
    tables: list[tuple[float, float, float, float, float]] = []
    for indices in grouped.values():
        index_array = np.asarray(indices, dtype=np.int64)
        exposed = selected[index_array]
        events = outcome[index_array] == 1
        a = float(np.sum(exposed & events))
        b = float(np.sum(exposed & ~events))
        c = float(np.sum(~exposed & events))
        d = float(np.sum(~exposed & ~events))
        n = a + b + c + d
        if n == 0 or a + b == 0 or c + d == 0 or a + c == 0 or b + d == 0:
            continue
        tables.append((a, b, c, d, n))

    base = {
        "mh_or_ci_method": (
            "Mantel-Haenszel common odds ratio with nominal 95% "
            "Robins-Breslow-Greenland log-scale confidence interval"
        ),
        "mh_informative_strata_n": len(tables),
        "mh_informative_gene_n": int(sum(table[4] for table in tables)),
    }
    if not tables:
        return {
            **base,
            "mh_common_odds_ratio": None,
            "mh_common_odds_ratio_ci95_lower": None,
            "mh_common_odds_ratio_ci95_upper": None,
            "mh_or_status": "not_estimable_no_informative_strata",
        }

    numerator = sum(a * d / n for a, _, _, d, n in tables)
    denominator = sum(b * c / n for _, b, c, _, n in tables)
    if numerator == 0 or denominator == 0:
        boundary = (
            "not_estimable_zero_cross_products"
            if numerator == 0 and denominator == 0
            else (
                "not_estimable_zero_numerator"
                if numerator == 0
                else "not_estimable_zero_denominator"
            )
        )
        return {
            **base,
            "mh_common_odds_ratio": None,
            "mh_common_odds_ratio_ci95_lower": None,
            "mh_common_odds_ratio_ci95_upper": None,
            "mh_or_status": boundary,
        }

    variance = sum((a + d) * a * d / (n * n) for a, _, _, d, n in tables) / (
        numerator * numerator
    )
    variance += sum(
        ((a + d) * b * c / (n * n)) + ((1.0 - (a + d) / n) * a * d / n)
        for a, b, c, d, n in tables
    ) / (numerator * denominator)
    variance += sum((1.0 - (a + d) / n) * b * c / n for a, b, c, d, n in tables) / (
        denominator * denominator
    )
    variance /= 2.0
    odds_ratio = numerator / denominator
    if variance < 0 or not math.isfinite(variance) or not math.isfinite(odds_ratio):
        return {
            **base,
            "mh_common_odds_ratio": None,
            "mh_common_odds_ratio_ci95_lower": None,
            "mh_common_odds_ratio_ci95_upper": None,
            "mh_or_status": "not_estimable_numerical_failure",
        }
    standard_error = math.sqrt(variance)
    z_975 = 1.959963984540054
    try:
        lower = math.exp(math.log(odds_ratio) - z_975 * standard_error)
        upper = math.exp(math.log(odds_ratio) + z_975 * standard_error)
    except OverflowError:
        lower = math.nan
        upper = math.nan
    if not all(math.isfinite(value) for value in (lower, upper)):
        return {
            **base,
            "mh_common_odds_ratio": None,
            "mh_common_odds_ratio_ci95_lower": None,
            "mh_common_odds_ratio_ci95_upper": None,
            "mh_or_status": "not_estimable_numerical_failure",
        }
    return {
        **base,
        "mh_common_odds_ratio": float(odds_ratio),
        "mh_common_odds_ratio_ci95_lower": float(lower),
        "mh_common_odds_ratio_ci95_upper": float(upper),
        "mh_or_status": "estimated",
    }


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or np.any((values < 0) | (values > 1))
    ):
        raise ValidationError("BH p-values must be finite values in [0,1]")
    if len(values) == 0:
        return []
    order = np.argsort(values, kind="mergesort")
    adjusted: np.ndarray = np.empty(len(values), dtype=np.float64)
    running = 1.0
    for position in range(len(values) - 1, -1, -1):
        index = int(order[position])
        running = min(running, float(values[index]) * len(values) / (position + 1))
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def _derive_seed(master: int, *parts: Any) -> int:
    payload = "|".join([str(int(master)), *(str(part) for part in parts)])
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big"
    ) % (2**31 - 1)


def _analysis_spec(n_permutations: int, permutation_seed: int) -> dict[str, Any]:
    return {
        "prediction_protocol": "P1 five-fold held-out predictions only",
        "prediction_coverage": "every cohort cell exactly once per condition and seed",
        "conditions": list(CANONICAL_CONDITIONS),
        "seeds": list(CANONICAL_SEEDS),
        "ranking": {
            "primary": "gene-wise 1.4826 * median absolute deviation across cells",
            "directional_secondary": "negative 10th percentile across cells",
            "seed_aggregation": "arithmetic mean of within-seed descending average ranks",
        },
        "top_n": {
            "primary": PRIMARY_TOP_N,
            "sensitivity": list(SENSITIVITY_TOP_N),
        },
        "endpoints": {
            "primary": "clinical_oncology_phase2_plus",
            "secondary": "tractable_any",
            "withdrawal_sensitivity": "clinical_oncology_withdrawal",
        },
        "matching": {
            "base": [
                "PPI degree",
                "MMseqs2 sequence-similarity-component size",
                "protein length",
            ],
            "clinical_extra": "tractable_any",
            "tractability_extra": None,
            "degree_bins": "degree zero separate; positive-degree quartiles",
            "family_bins": ["1", "2-3", "4-10", ">10"],
            "length_bins": "quintiles",
            "null": "fixed-stratum hypergeometric draws",
        },
        "n_permutations": int(n_permutations),
        "permutation_master_seed": int(permutation_seed),
        "p_value": "+1-corrected empirical Monte Carlo greater-or-equal tail",
        "effect_size": {
            "estimand": (
                "common odds ratio for selection and the endpoint conditional on "
                "the exact prespecified matching strata"
            ),
            "point_estimator": "Mantel-Haenszel common odds ratio",
            "confidence_level": 0.95,
            "confidence_interval": (
                "nominal two-sided Robins-Breslow-Greenland log-scale interval"
            ),
            "informative_strata": (
                "exclude strata with a zero selection or endpoint margin; retain "
                "individual zero cells without a pseudocount"
            ),
            "boundary_rule": (
                "return null point and bounds with an explicit status when the pooled "
                "Mantel-Haenszel numerator or denominator is zero"
            ),
            "claim_boundary": (
                "conditional matched association with frozen historical Open Targets "
                "annotations; not a population effect, prospective clinical validation, "
                "causal evidence, or DepPLM-specific validation"
            ),
        },
        "multiple_testing": "Benjamini-Hochberg over every reported test in tests.tsv",
    }


def _build_test_definitions() -> list[dict[str, Any]]:
    definitions = [
        {
            "ranking_metric": "robust_mad",
            "top_n": PRIMARY_TOP_N,
            "endpoint": "clinical_oncology_phase2_plus",
            "role": "primary",
        },
        {
            "ranking_metric": "robust_mad",
            "top_n": PRIMARY_TOP_N,
            "endpoint": "tractable_any",
            "role": "secondary_tractability",
        },
        {
            "ranking_metric": "directional_negative_q10",
            "top_n": PRIMARY_TOP_N,
            "endpoint": "clinical_oncology_phase2_plus",
            "role": "directional_secondary",
        },
        {
            "ranking_metric": "robust_mad",
            "top_n": PRIMARY_TOP_N,
            "endpoint": "clinical_oncology_withdrawal",
            "role": "withdrawal_sensitivity",
        },
    ]
    for top_n in SENSITIVITY_TOP_N:
        definitions.append(
            {
                "ranking_metric": "robust_mad",
                "top_n": top_n,
                "endpoint": "clinical_oncology_phase2_plus",
                "role": f"top_{top_n}_sensitivity",
            }
        )
    return definitions


def compute_analysis(
    report: Mapping[str, Any],
    outcomes: Mapping[str, Mapping[str, Any]],
    covariates: Mapping[str, Mapping[str, int]],
    cells: Sequence[str],
    genes: Sequence[str],
    *,
    ppi_path: Path,
    n_permutations: int,
    permutation_seed: int,
    workspace_root: Path | None = None,
    recorded_root: Path | None = None,
    capture_store: _CaptureStore | None = None,
) -> dict[str, Any]:
    if capture_store is None:
        with _CaptureStore(prefix="depplm-analysis-") as store:
            return compute_analysis(
                report,
                outcomes,
                covariates,
                cells,
                genes,
                ppi_path=ppi_path,
                n_permutations=n_permutations,
                permutation_seed=permutation_seed,
                workspace_root=workspace_root,
                recorded_root=recorded_root,
                capture_store=store,
            )
    if n_permutations < MIN_PERMUTATIONS:
        raise ValidationError(
            f"formal run requires at least {MIN_PERMUTATIONS} permutations"
        )
    p1_runs = _p1_run_map(report)
    ppi_replay = _load_p1_ppi_replay(ppi_path, genes, capture_store=capture_store)
    per_condition_scores: dict[str, dict[str, np.ndarray]] = {}
    reference_source_cells: np.ndarray | None = None
    reference_source_genes: np.ndarray | None = None
    for condition in CANONICAL_CONDITIONS:
        mad_ranks = []
        directional_ranks = []
        mad_scores = []
        directional_scores = []
        for seed in CANONICAL_SEEDS:
            bundle = _load_p1_oof_bundle(
                p1_runs[(condition, seed)],
                cells,
                genes,
                workspace_root=workspace_root,
                recorded_root=recorded_root,
                ppi_replay=ppi_replay,
                capture_store=capture_store,
            )
            if reference_source_cells is None:
                reference_source_cells = bundle.source_cell_indices.copy()
                reference_source_genes = bundle.source_gene_indices.copy()
            else:
                assert reference_source_genes is not None
                if not np.array_equal(
                    reference_source_cells, bundle.source_cell_indices
                ) or not np.array_equal(
                    reference_source_genes, bundle.source_gene_indices
                ):
                    raise ValidationError(
                        "P1 source cell/gene identities changed across condition or seed"
                    )
            matrix = bundle.prediction
            mad = robust_mad(matrix)
            directional = directional_tenth_percentile(matrix)
            mad_scores.append(mad)
            directional_scores.append(directional)
            mad_ranks.append(_average_rank_desc(mad))
            directional_ranks.append(_average_rank_desc(directional))
        per_condition_scores[condition] = {
            "robust_mad_rank": np.mean(np.stack(mad_ranks), axis=0),
            "robust_mad_score": np.mean(np.stack(mad_scores), axis=0),
            "directional_negative_q10_rank": np.mean(
                np.stack(directional_ranks), axis=0
            ),
            "directional_negative_q10_score": np.mean(
                np.stack(directional_scores), axis=0
            ),
        }

    eligible_genes = [gene for gene in genes if gene in outcomes and gene in covariates]
    if len(eligible_genes) < max(SENSITIVITY_TOP_N):
        raise ValidationError(
            f"only {len(eligible_genes)} genes have complete outcome/covariate data; "
            f"at least {max(SENSITIVITY_TOP_N)} are required"
        )
    gene_index = {gene: index for index, gene in enumerate(genes)}
    universe = []
    for gene in eligible_genes:
        universe.append(
            {
                "gene": gene,
                **dict(outcomes[gene]),
                **dict(covariates[gene]),
            }
        )
    endpoint_arrays = {
        endpoint: np.asarray([int(row[endpoint]) for row in universe], dtype=np.int8)
        for endpoint in (
            "clinical_oncology_phase2_plus",
            "clinical_oncology_withdrawal",
            "tractable_any",
        )
    }
    clinical_strata = make_strata(universe, include_tractability=True)
    tractability_strata = make_strata(universe, include_tractability=False)

    rankings = []
    ranking_orders: dict[tuple[str, str], list[int]] = {}
    for condition in CANONICAL_CONDITIONS:
        scores = per_condition_scores[condition]
        for metric in ("robust_mad", "directional_negative_q10"):
            eligible_indices = np.asarray(
                [gene_index[gene] for gene in eligible_genes], dtype=np.int64
            )
            rank_values = scores[f"{metric}_rank"][eligible_indices]
            mean_scores = scores[f"{metric}_score"][eligible_indices]
            order = sorted(
                range(len(universe)),
                key=lambda index: (float(rank_values[index]), universe[index]["gene"]),
            )
            ranking_orders[(condition, metric)] = order
            for position, index in enumerate(order, start=1):
                row = universe[index]
                rankings.append(
                    {
                        "condition": condition,
                        "ranking_metric": metric,
                        "rank": position,
                        "gene": row["gene"],
                        "target_id": row["target_id"],
                        "mean_seed_rank": float(rank_values[index]),
                        "mean_seed_score": float(mean_scores[index]),
                        "clinical_oncology_phase2_plus": int(
                            row["clinical_oncology_phase2_plus"]
                        ),
                        "clinical_oncology_withdrawal": int(
                            row["clinical_oncology_withdrawal"]
                        ),
                        "tractable_any": int(row["tractable_any"]),
                        "ppi_degree": int(row["ppi_degree"]),
                        "family_size": int(row["family_size"]),
                        "protein_length": int(row["protein_length"]),
                    }
                )

    tests = []
    for condition in CANONICAL_CONDITIONS:
        for definition in _build_test_definitions():
            metric = definition["ranking_metric"]
            top_n = int(definition["top_n"])
            endpoint = definition["endpoint"]
            selected: np.ndarray = np.zeros(len(universe), dtype=bool)
            selected[ranking_orders[(condition, metric)][:top_n]] = True
            strata = (
                tractability_strata if endpoint == "tractable_any" else clinical_strata
            )
            test_seed = _derive_seed(
                permutation_seed, condition, metric, top_n, endpoint
            )
            result = matched_permutation_test(
                selected,
                endpoint_arrays[endpoint],
                strata,
                n_permutations=n_permutations,
                seed=test_seed,
            )
            tests.append(
                {
                    "condition": condition,
                    **definition,
                    "matching_includes_tractability": endpoint != "tractable_any",
                    **result,
                }
            )
    adjusted = benjamini_hochberg([row["p_value"] for row in tests])
    for row, q_value in zip(tests, adjusted):
        row["q_value_bh_global"] = float(q_value)
    return {
        "analysis_spec": _analysis_spec(n_permutations, permutation_seed),
        "universe": {
            "cohort_genes": len(genes),
            "exact_outcome_and_covariate_genes": len(universe),
            "excluded_genes": len(genes) - len(universe),
            "clinical_oncology_phase2_plus_targets": int(
                np.sum(endpoint_arrays["clinical_oncology_phase2_plus"])
            ),
            "clinical_oncology_withdrawal_targets": int(
                np.sum(endpoint_arrays["clinical_oncology_withdrawal"])
            ),
            "tractable_targets": int(np.sum(endpoint_arrays["tractable_any"])),
        },
        "rankings": rankings,
        "tests": tests,
    }


RANKING_FIELDS = (
    "condition",
    "ranking_metric",
    "rank",
    "gene",
    "target_id",
    "mean_seed_rank",
    "mean_seed_score",
    "clinical_oncology_phase2_plus",
    "clinical_oncology_withdrawal",
    "tractable_any",
    "ppi_degree",
    "family_size",
    "protein_length",
)
TEST_FIELDS = (
    "condition",
    "ranking_metric",
    "top_n",
    "endpoint",
    "role",
    "matching_includes_tractability",
    "selected_n",
    "unselected_n",
    "outcome_positive_n",
    "unselected_outcome_positive_n",
    "observed_overlap",
    "expected_overlap",
    "fold_enrichment",
    "selected_event_rate",
    "matched_expected_event_rate",
    "matched_excess_proportion",
    "mh_common_odds_ratio",
    "mh_common_odds_ratio_ci95_lower",
    "mh_common_odds_ratio_ci95_upper",
    "mh_or_ci_method",
    "mh_or_status",
    "mh_informative_strata_n",
    "mh_informative_gene_n",
    "p_value",
    "q_value_bh_global",
    "n_permutations",
    "n_selected_strata",
    "seed",
    "null_rule",
    "p_value_rule",
)


def _workspace_relative_path(path: Path, *, label: str) -> str:
    try:
        relative = path.resolve().relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValidationError(
            f"{label} must be inside the workspace for portable provenance"
        ) from exc
    return relative.as_posix()


def _workspace_relative_record(
    record: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    raw_path = record.get("path")
    path = _resolve_bound_path(raw_path, label=label, workspace_root=ROOT)
    portable = dict(record)
    portable["path"] = _workspace_relative_path(path, label=label)
    return portable


def _portable_family_provenance(
    provenance: Mapping[str, Any], *, label: str = "family provenance"
) -> dict[str, Any]:
    portable = dict(provenance)
    for key in FAMILY_PROVENANCE_RECORD_FIELDS:
        record = provenance.get(key)
        if not isinstance(record, Mapping):
            raise ValidationError(f"{label} {key} record is missing")
        portable[key] = _workspace_relative_record(record, label=f"{label} {key}")
    return portable


def _input_bindings(
    campaign_record: Mapping[str, Any],
    snapshot_record: Mapping[str, Any],
    covariate_records: Mapping[str, Any],
    cells_path: Path,
    genes_path: Path,
    *,
    cells_record: Mapping[str, Any] | None = None,
    genes_record: Mapping[str, Any] | None = None,
    analysis_source_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "campaign_report": _workspace_relative_record(
            campaign_record, label="campaign report"
        ),
        "snapshot_manifest": _workspace_relative_record(
            snapshot_record, label="snapshot manifest"
        ),
        "ppi": _workspace_relative_record(
            covariate_records["ppi"], label="PPI covariate"
        ),
        "family": _workspace_relative_record(
            covariate_records["family"], label="family covariate"
        ),
        "family_provenance": _portable_family_provenance(
            covariate_records["family_provenance"]
        ),
        "fasta": _workspace_relative_record(
            covariate_records["fasta"], label="FASTA covariate"
        ),
        "cells": _workspace_relative_record(
            _file_record(cells_path) if cells_record is None else cells_record,
            label="cohort cells",
        ),
        "genes": _workspace_relative_record(
            _file_record(genes_path) if genes_record is None else genes_record,
            label="cohort genes",
        ),
        "analysis_source": _workspace_relative_record(
            (
                _loaded_analysis_source_record()
                if analysis_source_record is None
                else analysis_source_record
            ),
            label="analysis source",
        ),
    }


def _write_analysis_artifacts(
    directory: Path,
    analysis: Mapping[str, Any],
    run_manifest_record: Mapping[str, Any],
    *,
    directory_identity: tuple[int, int] | None = None,
    publication: _PendingFormalPublication | None = None,
) -> dict[str, bytes]:
    payloads = {
        "rankings.tsv": _tsv_bytes(analysis["rankings"], RANKING_FIELDS),
        "tests.tsv": _tsv_bytes(analysis["tests"], TEST_FIELDS),
    }
    results = {
        "schema": RUN_SCHEMA,
        "scientific_result": True,
        "claim_boundary": CLAIM_BOUNDARY,
        "run_manifest": dict(run_manifest_record),
        "analysis_spec": analysis["analysis_spec"],
        "universe": analysis["universe"],
        "n_rankings": len(analysis["rankings"]),
        "n_tests": len(analysis["tests"]),
        "tests": analysis["tests"],
    }
    payloads["results.json"] = _canonical_json_bytes(results)
    for name, payload in payloads.items():
        if publication is not None:
            _write_pinned_bytes(
                publication.staging_pin,
                name,
                payload,
                publication.created_file_identities,
                label="formal-run artifact",
            )
        else:
            if directory_identity is not None:
                _require_directory_identity(
                    directory,
                    directory_identity,
                    label="private formal-run staging directory",
                )
            _atomic_bytes(directory / name, payload)
    return payloads


def _require_formal_snapshot(snapshot: Mapping[str, Any]) -> None:
    manifest = snapshot.get("manifest")
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema") != FORMAL_SNAPSHOT_SCHEMA
    ):
        raise ValidationError(
            "new formal analyses require a source-hash-bound builder-v4 snapshot"
        )


def _reserve_formal_destination(
    output_root: Path, identifier: str
) -> _PendingFormalPublication:
    output_root = Path(os.path.abspath(os.fspath(output_root)))
    output_root.mkdir(parents=True, exist_ok=True)
    _reject_link_or_reparse_components(output_root, label="formal output root")
    parent_pin = _pin_directory(output_root, label="formal output root")
    parent_identity = parent_pin.identity
    destination = output_root / identifier
    try:
        staging = destination
        staging_pin = _create_pinned_directory(
            parent_pin,
            identifier,
            staging,
            label="reserved formal-run destination",
        )
    except FileExistsError as exc:
        try:
            _close_pinned_directory(parent_pin, label="formal output root")
        except Exception as cleanup_error:
            exc.add_note(f"formal output-root handle cleanup failed: {cleanup_error}")
        raise ValidationError(
            f"refusing to overwrite formal run directory: {destination}"
        ) from exc
    except BaseException as exc:
        try:
            _close_pinned_directory(parent_pin, label="formal output root")
        except Exception as cleanup_error:
            exc.add_note(f"formal output-root handle cleanup failed: {cleanup_error}")
        raise
    return _PendingFormalPublication(
        destination=destination,
        staging=staging,
        parent_identity=parent_identity,
        staging_identity=staging_pin.identity,
        parent_pin=parent_pin,
        staging_pin=staging_pin,
    )


def _close_formal_publication_pins(
    publication: _PendingFormalPublication,
    *,
    primary_error: BaseException | None = None,
) -> None:
    close_errors: list[Exception] = []
    for pin, label in (
        (publication.staging_pin, "reserved formal-run destination"),
        (publication.parent_pin, "formal output root"),
    ):
        try:
            _close_pinned_directory(pin, label=label)
        except Exception as exc:
            close_errors.append(exc)
    if close_errors:
        for error in close_errors:
            if primary_error is not None:
                primary_error.add_note(
                    f"publication-handle cleanup also failed: {error}"
                )
        if primary_error is None:
            raise close_errors[0]


def _remove_pinned_empty_directory(
    directory: _PinnedDirectory,
    parent: _PinnedDirectory,
    *,
    label: str,
    entry_name: str | None = None,
) -> None:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        if directory.windows_handle is None:
            raise ValidationError(f"{label} handle is closed")

        class FileDispositionInfo(ctypes.Structure):
            _fields_ = [("DeleteFile", wintypes.BOOLEAN)]

        disposition = FileDispositionInfo(True)
        set_information = ctypes.WinDLL(
            "kernel32", use_last_error=True
        ).SetFileInformationByHandle
        set_information.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        set_information.restype = wintypes.BOOL
        if not set_information(
            directory.windows_handle.value,
            4,  # FileDispositionInfo
            ctypes.byref(disposition),
            ctypes.sizeof(disposition),
        ):
            error = ctypes.get_last_error()
            raise ValidationError(
                f"could not remove {label}: {directory.path}"
            ) from OSError(error, os.strerror(error), str(directory.path))
        # Windows completes deletion only after the final handle closes.  Close
        # this child now so a retained parent can subsequently become empty.
        _close_pinned_directory(directory, label=label)
        return

    if parent.posix_fd is None:
        raise ValidationError("formal output-root descriptor is closed")
    try:
        os.rmdir(
            directory.path.name if entry_name is None else entry_name,
            dir_fd=parent.posix_fd,
        )
    except OSError as exc:
        raise ValidationError(f"could not remove {label}: {directory.path}") from exc


def _cleanup_owned_formal_destination(
    publication: _PendingFormalPublication,
) -> None:
    primary_error: BaseException | None = None
    quarantined_name: str | None = None
    try:
        for name, identity in list(publication.created_file_identities.items()):
            _delete_pinned_file(
                publication.staging_pin,
                name,
                identity,
                label="owned formal-run artifact",
            )
        quarantined_name = _quarantine_pinned_directory(
            publication.staging_pin,
            publication.parent_pin,
            label="reserved formal-run destination",
        )
        _remove_pinned_empty_directory(
            publication.staging_pin,
            publication.parent_pin,
            label="reserved formal-run destination",
            entry_name=quarantined_name,
        )
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        _close_formal_publication_pins(publication, primary_error=primary_error)


def _publish_formal_directory(publication: _PendingFormalPublication) -> Path:
    """Commit the pinned inventory and return a point-in-time validated path.

    The final-named directory is visible while it is populated. Publication is
    commit-marker based, not an atomic directory rename, so consumers must
    reject a missing or invalid completion marker.

    A returned ``Path`` is not a retained capability.  Downstream consumers
    must independently verify the committed file hashes; this function does
    not claim protection from same-UID replacement after its final check.
    """
    _require_pinned_directory_entry(
        publication.parent_pin,
        publication.destination.name,
        publication.staging_identity,
        label="reserved formal-run destination",
    )
    marker_identity = publication.created_file_identities.get("completion.json")
    if marker_identity is None:
        raise ValidationError("formal run has no completion commit marker")
    expected_order = (
        "run-manifest.json",
        "rankings.tsv",
        "tests.tsv",
        "results.json",
        "verification.json",
        "completion.json",
    )
    _require_pinned_inventory(
        publication.staging_pin,
        publication.created_file_identities,
        expected_order,
        label="formal publication",
    )
    _close_formal_publication_pins(publication)
    _require_directory_identity(
        publication.destination,
        publication.staging_identity,
        label="published formal-run destination",
    )
    return publication.destination


def run_formal_analysis(
    campaign_report: Path,
    *,
    snapshot_dir: Path,
    ppi_path: Path = DEFAULT_PPI,
    family_path: Path = DEFAULT_FAMILY,
    family_release_manifest: Path = DEFAULT_FAMILY_RELEASE_MANIFEST,
    fasta_path: Path = DEFAULT_FASTA,
    cells_path: Path = DEFAULT_CELLS,
    genes_path: Path = DEFAULT_GENES,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    n_permutations: int = MIN_PERMUTATIONS,
    permutation_seed: int = DEFAULT_PERMUTATION_SEED,
) -> Path:
    """Run and publish the clinical analysis after every producer gate passes.

    The returned completion path has only point-in-time publication identity.
    A downstream release consumer must reopen the directory and call
    :func:`verify_formal_run` before using its contents.
    """
    _require_loaded_source_unchanged()
    with _CaptureStore(prefix="depplm-formal-inputs-") as capture_store:
        campaign, campaign_record = validate_campaign_report(
            campaign_report,
            _capture_store=capture_store,
            _return_captured_paths=True,
        )
        canonical_ppi = _canonical_campaign_ppi_binding(
            campaign, capture_store=capture_store
        )
        ppi_capture = capture_store.capture(ppi_path, label="supplied PPI file")
        _require_campaign_ppi_capture(ppi_capture, canonical_ppi)
        snapshot = validate_snapshot(snapshot_dir)
        _require_formal_snapshot(snapshot)
        cells_payload, cells_capture = capture_store.capture_bytes(
            cells_path, label="cohort cells"
        )
        genes_payload, genes_capture = capture_store.capture_bytes(
            genes_path, label="cohort genes"
        )
        cells = _unique_lines_from_bytes(cells_payload, label="cohort cells")
        genes = _unique_lines_from_bytes(genes_payload, label="cohort genes")
        outcome_records = snapshot.get("manifest", {}).get("artifacts", [])
        outcome_record = next(
            (
                record
                for record in outcome_records
                if isinstance(record, Mapping)
                and record.get("path") == "target_outcomes.tsv"
            ),
            None,
        )
        if outcome_record is None:
            raise ValidationError("snapshot lacks the target_outcomes.tsv binding")
        outcomes = load_outcomes(
            snapshot_dir,
            capture_store=capture_store,
            expected_record=outcome_record,
        )
        family_captures = _capture_family_validation_inputs(
            capture_store,
            family_path,
            family_release_manifest,
            genes_path=genes_capture.path,
            fasta_path=fasta_path,
        )
        family_provenance = validate_family_covariate_provenance(
            family_captures["family"].path,
            family_captures["release"].path,
            genes_path=family_captures["genes"].path,
            fasta_path=family_captures["fasta"].path,
            capture_store=capture_store,
            captured_inputs=family_captures,
        )
        covariate_inputs = {
            "ppi": ppi_capture,
            "family": family_captures["family"],
            "fasta": family_captures["fasta"],
        }
        covariates, covariate_records = load_covariates(
            genes,
            ppi_capture.path,
            family_captures["family"].path,
            family_captures["fasta"].path,
            capture_store=capture_store,
            captured_inputs=covariate_inputs,
        )
        covariate_records["family_provenance"] = family_provenance
        analysis = compute_analysis(
            campaign,
            outcomes,
            covariates,
            cells,
            genes,
            ppi_path=ppi_capture.path,
            n_permutations=n_permutations,
            permutation_seed=permutation_seed,
            capture_store=capture_store,
        )
        bindings = _input_bindings(
            campaign_record,
            snapshot["manifest_record"],
            covariate_records,
            cells_path,
            genes_path,
            cells_record=cells_capture.record(),
            genes_record=genes_capture.record(),
            analysis_source_record=_loaded_analysis_source_record(),
        )
    run_manifest = {
        "schema": f"{RUN_SCHEMA}-manifest",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": bindings,
        "snapshot_dir": _workspace_relative_path(
            snapshot_dir, label="snapshot directory"
        ),
        "analysis_spec": _analysis_spec(n_permutations, permutation_seed),
        "runtime": _formal_runtime_identity(),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    identifier = (
        time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        + f".{time.time_ns() % 1_000_000_000:09d}Z-{uuid.uuid4().hex[:8]}"
    )
    publication = _reserve_formal_destination(output_root, identifier)
    destination = publication.staging
    try:
        manifest_payload = _canonical_json_bytes(run_manifest)
        _write_pinned_bytes(
            publication.staging_pin,
            "run-manifest.json",
            manifest_payload,
            publication.created_file_identities,
            label="formal-run manifest",
        )
        manifest_record = _record_for_bytes(manifest_payload, "run-manifest.json")
        analysis_payloads = _write_analysis_artifacts(
            destination,
            analysis,
            manifest_record,
            directory_identity=publication.staging_identity,
            publication=publication,
        )
        artifact_payloads = {
            "run-manifest.json": manifest_payload,
            **analysis_payloads,
        }
        artifacts = [
            _record_for_bytes(artifact_payloads[name], name)
            for name in FORMAL_VERIFICATION_ARTIFACTS
        ]
        verification = {
            "schema": VERIFICATION_SCHEMA,
            "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "scientific_result": True,
            "verification_method": FORMAL_VERIFICATION_METHOD,
            "artifacts": artifacts,
            "analysis_sha256": hashlib.sha256(
                _canonical_json_bytes(
                    {
                        "analysis_spec": analysis["analysis_spec"],
                        "universe": analysis["universe"],
                        "rankings": analysis["rankings"],
                        "tests": analysis["tests"],
                    }
                )
            ).hexdigest(),
        }
        verification_payload = _canonical_json_bytes(verification)
        _write_pinned_bytes(
            publication.staging_pin,
            "verification.json",
            verification_payload,
            publication.created_file_identities,
            label="formal verification",
        )
        completion = {
            "schema": COMPLETION_SCHEMA,
            "scientific_result": True,
            "verification": _record_for_bytes(
                verification_payload, "verification.json"
            ),
            "results": _record_for_bytes(
                analysis_payloads["results.json"], "results.json"
            ),
        }
        _write_pinned_bytes(
            publication.staging_pin,
            "completion.json",
            _canonical_json_bytes(completion),
            publication.created_file_identities,
            label="formal completion marker",
        )
        verify_formal_run(destination)
        _publish_formal_directory(publication)
    except BaseException as exc:
        try:
            _cleanup_owned_formal_destination(publication)
        except Exception as cleanup_error:
            exc.add_note(f"formal staging cleanup also failed: {cleanup_error}")
        raise
    return publication.destination / "completion.json"


def _verify_bound_record(
    record: Mapping[str, Any],
    *,
    label: str,
    workspace_root: Path | None = None,
    recorded_root: Path | None = None,
) -> Path:
    path = _resolve_bound_path(
        record.get("path"),
        label=label,
        workspace_root=workspace_root,
        recorded_root=recorded_root,
    )
    _verify_file_record(path, record, label=label)
    return path


def _verification_roots(
    workspace_root: Path | None, recorded_root: Path | None
) -> tuple[Path, Path]:
    if (workspace_root is None) != (recorded_root is None):
        raise ValidationError(
            "workspace_root and recorded_root must be supplied together"
        )
    if workspace_root is None:
        current_root = ROOT.resolve()
        return current_root, current_root
    assert recorded_root is not None
    if not workspace_root.is_absolute() or not recorded_root.is_absolute():
        raise ValidationError("workspace_root and recorded_root must be absolute")
    resolved_workspace = workspace_root.resolve()
    if not resolved_workspace.is_dir():
        raise ValidationError(f"workspace_root is not a directory: {workspace_root}")
    return resolved_workspace, recorded_root.absolute()


def _materialize_family_provenance_binding(
    binding: Mapping[str, Any],
    *,
    workspace_root: Path,
    recorded_root: Path | None,
) -> dict[str, Any]:
    materialized = dict(binding)
    for key in FAMILY_PROVENANCE_RECORD_FIELDS:
        record = binding.get(key)
        if not isinstance(record, Mapping):
            raise ValidationError(f"formal run family provenance {key} is missing")
        if set(record) != {"path", "bytes", "sha256"}:
            raise ValidationError(
                f"formal run family provenance {key} record is noncanonical"
            )
        path = _verify_bound_record(
            record,
            label=f"family provenance {key}",
            workspace_root=workspace_root,
            recorded_root=recorded_root,
        )
        materialized[key] = {
            "path": str(path),
            "bytes": record["bytes"],
            "sha256": record["sha256"],
        }
    return materialized


def _require_formal_metadata_inventory(
    run_dir: Path,
    manifest: Mapping[str, Any],
    verification: Mapping[str, Any],
    completion: Mapping[str, Any],
) -> None:
    entries = list(run_dir.iterdir())
    if {entry.name for entry in entries} != FORMAL_RUN_FILES or any(
        not entry.is_file() or entry.is_symlink() or _has_reparse_point(entry)
        for entry in entries
    ):
        raise ValidationError("formal run file inventory is noncanonical")
    if set(manifest) != FORMAL_RUN_MANIFEST_FIELDS:
        raise ValidationError("formal run manifest fields are noncanonical")
    if manifest.get("schema") != f"{RUN_SCHEMA}-manifest":
        raise ValidationError("formal run manifest schema is invalid")
    _require_canonical_utc_timestamp(
        manifest.get("created_utc"), label="formal run created_utc"
    )
    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict) or set(runtime) != FORMAL_RUNTIME_FIELDS:
        raise ValidationError("formal run runtime identity is noncanonical")
    if runtime != _formal_runtime_identity():
        raise ValidationError("formal run runtime identity changed")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != FORMAL_INPUT_FIELDS:
        raise ValidationError("formal run input bindings are noncanonical")
    campaign_record = inputs.get("campaign_report")
    if (
        not isinstance(campaign_record, dict)
        or set(campaign_record) != FORMAL_CAMPAIGN_RECORD_FIELDS
        or campaign_record.get("source_runs_rehashed") != EXPECTED_RUNS
        or campaign_record.get("source_artifacts_rehashed")
        != EXPECTED_CAMPAIGN_SOURCE_ARTIFACTS
    ):
        raise ValidationError("formal campaign binding is noncanonical")
    for key in FORMAL_FILE_INPUT_FIELDS:
        record = inputs.get(key)
        if not isinstance(record, dict) or set(record) != {
            "path",
            "bytes",
            "sha256",
        }:
            raise ValidationError(f"formal input {key} binding is noncanonical")
    family_provenance = inputs.get("family_provenance")
    if (
        not isinstance(family_provenance, dict)
        or set(family_provenance) != FORMAL_FAMILY_PROVENANCE_FIELDS
    ):
        raise ValidationError("formal family provenance binding is noncanonical")
    for key in FAMILY_PROVENANCE_RECORD_FIELDS:
        record = family_provenance.get(key)
        if not isinstance(record, dict) or set(record) != {
            "path",
            "bytes",
            "sha256",
        }:
            raise ValidationError(
                f"formal family provenance {key} binding is noncanonical"
            )
    if set(verification) != FORMAL_VERIFICATION_FIELDS:
        raise ValidationError("formal verification fields are noncanonical")
    if verification.get("schema") != VERIFICATION_SCHEMA:
        raise ValidationError("formal verification schema is invalid")
    _require_canonical_utc_timestamp(
        verification.get("verified_utc"), label="formal verification verified_utc"
    )
    if verification.get("scientific_result") is not True:
        raise ValidationError("formal verification result marker is invalid")
    if verification.get("verification_method") != FORMAL_VERIFICATION_METHOD:
        raise ValidationError("formal verification method is noncanonical")
    if not _valid_sha256(verification.get("analysis_sha256")):
        raise ValidationError("formal verification analysis hash is invalid")
    records = verification.get("artifacts")
    if not isinstance(records, list) or len(records) != len(
        FORMAL_VERIFICATION_ARTIFACTS
    ):
        raise ValidationError("verification artifact inventory is invalid")
    paths: list[Any] = []
    by_path: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "path",
            "bytes",
            "sha256",
        }:
            raise ValidationError("verification artifact record is noncanonical")
        relative = record.get("path")
        paths.append(relative)
        if isinstance(relative, str):
            by_path[relative] = record
    if tuple(paths) != FORMAL_VERIFICATION_ARTIFACTS or len(by_path) != len(records):
        raise ValidationError("verification artifact paths are noncanonical")
    if set(completion) != FORMAL_COMPLETION_FIELDS:
        raise ValidationError("formal completion fields are noncanonical")
    if completion.get("schema") != COMPLETION_SCHEMA:
        raise ValidationError("formal completion schema is invalid")
    if completion.get("scientific_result") is not True:
        raise ValidationError("formal completion result marker is invalid")
    for key, expected_path in (
        ("verification", "verification.json"),
        ("results", "results.json"),
    ):
        record = completion.get(key)
        if not isinstance(record, dict) or set(record) != {
            "path",
            "bytes",
            "sha256",
        }:
            raise ValidationError(f"completion {key} binding is noncanonical")
        if record.get("path") != expected_path:
            raise ValidationError(f"completion {key} path is noncanonical")
    if completion["results"] != by_path["results.json"]:
        raise ValidationError(
            "completion results binding differs from the verification inventory"
        )


def _capture_bound_record(
    store: _CaptureStore,
    record: Mapping[str, Any],
    *,
    label: str,
    workspace_root: Path | None = None,
    recorded_root: Path | None = None,
) -> tuple[Path, _CapturedFile]:
    path = _resolve_bound_path(
        record.get("path"),
        label=label,
        workspace_root=workspace_root,
        recorded_root=recorded_root,
    )
    return path, store.capture(path, label=label, expected_record=record)


def _require_captured_record(
    captured: _CapturedFile,
    record: Mapping[str, Any],
    *,
    label: str,
    logical_path: str,
) -> None:
    if set(record) != {"path", "bytes", "sha256"}:
        raise ValidationError(f"{label} record is noncanonical")
    if dict(record) != captured.record(logical_path):
        raise ValidationError(f"{label} binding changed")


def verify_formal_run(
    run_dir: Path,
    *,
    workspace_root: Path | None = None,
    recorded_root: Path | None = None,
) -> dict[str, Any]:
    """Recompute and compare the exact artifact bytes visible during this call.

    Verification is point-in-time: the returned report is not a retained
    capability for ``run_dir`` and cannot prevent a later same-UID pathname
    replacement.
    """
    _require_loaded_source_unchanged()
    with _CaptureStore(prefix="depplm-formal-verification-") as capture_store:
        return _verify_formal_run_captured(
            run_dir,
            workspace_root=workspace_root,
            recorded_root=recorded_root,
            capture_store=capture_store,
        )


def _verify_formal_run_captured(
    run_dir: Path,
    *,
    workspace_root: Path | None,
    recorded_root: Path | None,
    capture_store: _CaptureStore,
) -> dict[str, Any]:
    current_root, legacy_root = _verification_roots(workspace_root, recorded_root)
    run_dir = Path(os.path.abspath(os.fspath(run_dir)))
    _reject_link_or_reparse_components(run_dir, label="formal run directory")
    formal_payload_captures = {
        name: capture_store.capture_bytes(
            run_dir / name, label=f"formal run file {name}"
        )
        for name in sorted(FORMAL_RUN_FILES)
    }
    formal_payloads = {
        name: payload for name, (payload, _captured) in formal_payload_captures.items()
    }
    formal_captures = {
        name: captured for name, (_payload, captured) in formal_payload_captures.items()
    }
    metadata: dict[str, Any] = {}
    for name in (
        "run-manifest.json",
        "results.json",
        "verification.json",
        "completion.json",
    ):
        parsed = _strict_json_bytes(formal_payloads[name], label=name)
        if not isinstance(parsed, dict):
            raise ValidationError(f"{name} must contain a JSON object")
        if formal_payloads[name] != _canonical_json_bytes(parsed):
            raise ValidationError(f"{name} bytes are not canonical JSON")
        metadata[name] = parsed
    manifest = metadata["run-manifest.json"]
    verification = metadata["verification.json"]
    completion = metadata["completion.json"]
    if not all(
        isinstance(payload, dict) for payload in (manifest, verification, completion)
    ):
        raise ValidationError("formal run metadata must contain JSON objects")
    _require_formal_metadata_inventory(run_dir, manifest, verification, completion)
    if manifest.get("schema") != f"{RUN_SCHEMA}-manifest":
        raise ValidationError("formal run manifest schema is invalid")
    if verification.get("schema") != VERIFICATION_SCHEMA:
        raise ValidationError("formal verification schema is invalid")
    if completion.get("schema") != COMPLETION_SCHEMA:
        raise ValidationError("formal completion schema is invalid")
    if (
        verification.get("scientific_result") is not True
        or completion.get("scientific_result") is not True
    ):
        raise ValidationError(
            "formal run is not marked as a verified scientific result"
        )
    bindings = manifest.get("inputs")
    if not isinstance(bindings, dict):
        raise ValidationError("formal run input bindings are missing")
    source_path, _source_capture = _capture_bound_record(
        capture_store,
        bindings["analysis_source"],
        label="analysis source",
        workspace_root=current_root,
        recorded_root=legacy_root,
    )
    if source_path != Path(__file__).resolve():
        raise ValidationError("formal run is bound to another analysis source")
    campaign_path, campaign_capture = _capture_bound_record(
        capture_store,
        bindings["campaign_report"],
        label="campaign report",
        workspace_root=current_root,
        recorded_root=legacy_root,
    )
    campaign, campaign_record = validate_campaign_report(
        campaign_capture.path,
        workspace_root=current_root,
        recorded_root=legacy_root,
        _capture_store=capture_store,
        _return_captured_paths=True,
    )
    if campaign_record["sha256"] != bindings["campaign_report"]["sha256"]:
        raise ValidationError("campaign report binding changed")
    canonical_ppi = _canonical_campaign_ppi_binding(
        campaign,
        workspace_root=current_root,
        recorded_root=legacy_root,
        capture_store=capture_store,
    )
    snapshot_manifest, snapshot_manifest_capture = _capture_bound_record(
        capture_store,
        bindings["snapshot_manifest"],
        label="snapshot manifest",
        workspace_root=current_root,
        recorded_root=legacy_root,
    )
    snapshot_dir_binding = _resolve_bound_path(
        manifest.get("snapshot_dir"),
        label="snapshot directory",
        workspace_root=current_root,
        recorded_root=legacy_root,
    )
    if snapshot_dir_binding != snapshot_manifest.parent:
        raise ValidationError(
            "formal snapshot directory differs from the bound snapshot manifest"
        )
    snapshot = validate_snapshot(snapshot_manifest.parent)
    _require_formal_snapshot(snapshot)
    if snapshot["manifest_record"]["sha256"] != bindings["snapshot_manifest"]["sha256"]:
        raise ValidationError("snapshot manifest binding changed")
    ppi_path, ppi_capture = _capture_bound_record(
        capture_store,
        bindings["ppi"],
        label="PPI covariate",
        workspace_root=current_root,
        recorded_root=legacy_root,
    )
    _require_campaign_ppi_capture(ppi_capture, canonical_ppi)
    family_path, family_capture = _capture_bound_record(
        capture_store,
        bindings["family"],
        label="family covariate",
        workspace_root=current_root,
        recorded_root=legacy_root,
    )
    fasta_path, fasta_capture = _capture_bound_record(
        capture_store,
        bindings["fasta"],
        label="FASTA covariate",
        workspace_root=current_root,
        recorded_root=legacy_root,
    )
    cells_path, cells_capture = _capture_bound_record(
        capture_store,
        bindings["cells"],
        label="cohort cells",
        workspace_root=current_root,
        recorded_root=legacy_root,
    )
    genes_path, genes_capture = _capture_bound_record(
        capture_store,
        bindings["genes"],
        label="cohort genes",
        workspace_root=current_root,
        recorded_root=legacy_root,
    )
    family_provenance_binding = bindings.get("family_provenance")
    if not isinstance(family_provenance_binding, dict):
        raise ValidationError("formal run family provenance binding is missing")
    family_bound_captures: dict[str, tuple[Path, _CapturedFile]] = {}
    for key in FAMILY_PROVENANCE_RECORD_FIELDS:
        record = family_provenance_binding.get(key)
        if not isinstance(record, Mapping):
            raise ValidationError(f"formal run family provenance {key} is missing")
        family_bound_captures[key] = _capture_bound_record(
            capture_store,
            record,
            label=f"family provenance {key}",
            workspace_root=current_root,
            recorded_root=legacy_root,
        )

    staged_family_root = Path("formal-family-release")
    staged_family_captures = {
        "release": capture_store.capture(
            family_bound_captures["family_release_manifest"][1].path,
            label="captured family release manifest",
            destination_relative=staged_family_root / "release_manifest.json",
        ),
        "family": capture_store.capture(
            family_capture.path,
            label="captured family covariate",
            destination_relative=staged_family_root / FAMILY_UNITS_RELATIVE,
        ),
        "summary": capture_store.capture(
            family_bound_captures["family_split_summary"][1].path,
            label="captured family split summary",
            destination_relative=staged_family_root / FAMILY_SUMMARY_RELATIVE,
        ),
        "source": capture_store.capture(
            family_bound_captures["family_source_manifest"][1].path,
            label="captured family source manifest",
            destination_relative=staged_family_root / FAMILY_SOURCE_RELATIVE,
        ),
        "hits": capture_store.capture(
            family_bound_captures["family_homology_hits"][1].path,
            label="captured family homology hits",
            destination_relative=staged_family_root / FAMILY_HITS_RELATIVE,
        ),
        "genes": genes_capture,
        "fasta": fasta_capture,
    }
    family_provenance = validate_family_covariate_provenance(
        staged_family_captures["family"].path,
        staged_family_captures["release"].path,
        genes_path=genes_capture.path,
        fasta_path=fasta_capture.path,
        capture_store=capture_store,
        captured_inputs=staged_family_captures,
    )
    materialized_family_provenance = dict(family_provenance_binding)
    for key, (logical_path, captured) in family_bound_captures.items():
        materialized_family_provenance[key] = captured.record(logical_path)
        family_provenance[key] = captured.record(logical_path)
    if family_provenance != materialized_family_provenance:
        raise ValidationError("formal run family provenance binding changed")
    cells = _unique_lines_from_bytes(
        cells_capture.path.read_bytes(), label="cohort cells"
    )
    genes = _unique_lines_from_bytes(
        genes_capture.path.read_bytes(), label="cohort genes"
    )
    outcome_records = snapshot.get("manifest", {}).get("artifacts", [])
    outcome_record = next(
        (
            record
            for record in outcome_records
            if isinstance(record, Mapping)
            and record.get("path") == "target_outcomes.tsv"
        ),
        None,
    )
    if outcome_record is None:
        raise ValidationError("snapshot lacks the target_outcomes.tsv binding")
    outcomes = load_outcomes(
        snapshot_manifest.parent,
        capture_store=capture_store,
        expected_record=outcome_record,
    )
    covariates, _ = load_covariates(
        genes,
        ppi_capture.path,
        family_capture.path,
        fasta_capture.path,
        capture_store=capture_store,
        captured_inputs={
            "ppi": ppi_capture,
            "family": family_capture,
            "fasta": fasta_capture,
        },
    )
    spec = manifest.get("analysis_spec")
    if not isinstance(spec, dict):
        raise ValidationError("formal analysis specification is missing")
    n_permutations = int(spec.get("n_permutations", 0))
    permutation_seed = int(spec.get("permutation_master_seed", -1))
    if spec != _analysis_spec(n_permutations, permutation_seed):
        raise ValidationError("formal analysis specification is noncanonical")
    analysis = compute_analysis(
        campaign,
        outcomes,
        covariates,
        cells,
        genes,
        ppi_path=ppi_capture.path,
        n_permutations=n_permutations,
        permutation_seed=permutation_seed,
        workspace_root=current_root,
        recorded_root=legacy_root,
        capture_store=capture_store,
    )
    manifest_record = formal_captures["run-manifest.json"].record("run-manifest.json")
    expected_rankings = _tsv_bytes(analysis["rankings"], RANKING_FIELDS)
    expected_tests = _tsv_bytes(analysis["tests"], TEST_FIELDS)
    results = {
        "schema": RUN_SCHEMA,
        "scientific_result": True,
        "claim_boundary": CLAIM_BOUNDARY,
        "run_manifest": manifest_record,
        "analysis_spec": analysis["analysis_spec"],
        "universe": analysis["universe"],
        "n_rankings": len(analysis["rankings"]),
        "n_tests": len(analysis["tests"]),
        "tests": analysis["tests"],
    }
    expected_results = _canonical_json_bytes(results)
    expected_payloads = {
        "rankings.tsv": expected_rankings,
        "tests.tsv": expected_tests,
        "results.json": expected_results,
    }
    for name, expected in expected_payloads.items():
        observed = formal_payloads[name]
        if observed != expected:
            raise ValidationError(f"formal artifact differs from recomputation: {name}")
    records = verification.get("artifacts")
    if not isinstance(records, list) or len(records) != 4:
        raise ValidationError("verification artifact inventory is invalid")
    for record in records:
        if not isinstance(record, dict):
            raise ValidationError("verification artifact record is invalid")
        relative = record.get("path")
        if not isinstance(relative, str) or relative not in formal_captures:
            raise ValidationError("verification artifact path is invalid")
        _require_captured_record(
            formal_captures[relative],
            record,
            label="verified artifact",
            logical_path=relative,
        )
    expected_analysis_sha = hashlib.sha256(
        _canonical_json_bytes(
            {
                "analysis_spec": analysis["analysis_spec"],
                "universe": analysis["universe"],
                "rankings": analysis["rankings"],
                "tests": analysis["tests"],
            }
        )
    ).hexdigest()
    if verification.get("analysis_sha256") != expected_analysis_sha:
        raise ValidationError("verification analysis hash differs from recomputation")
    for key in ("verification", "results"):
        record = completion.get(key)
        if not isinstance(record, dict):
            raise ValidationError(f"completion {key} binding is missing")
        relative = record.get("path")
        if not isinstance(relative, str) or relative not in formal_captures:
            raise ValidationError(f"completion {key} path is invalid")
        _require_captured_record(
            formal_captures[relative],
            record,
            label=f"completion {key}",
            logical_path=relative,
        )
    return {
        "schema": VERIFICATION_SCHEMA,
        "scientific_result": True,
        "run_dir": str(run_dir),
        "verification_scope": "point_in_time_hash_verification",
        "path_identity_retained": False,
        "marker_record": formal_captures["completion.json"].record(
            "completion.json"
        ),
        "analysis_sha256": expected_analysis_sha,
        "campaign_source_runs_rehashed": campaign_record["source_runs_rehashed"],
        "campaign_source_artifacts_rehashed": campaign_record[
            "source_artifacts_rehashed"
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser(
        "freeze", help="build a new source-hash-bound builder-v4 Open Targets snapshot"
    )
    freeze.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    freeze.add_argument("--output-dir", type=Path, default=DEFAULT_SNAPSHOT_BUILD_DIR)
    preflight = subparsers.add_parser(
        "preflight-freeze",
        help="rehash canonical raw inputs and print the exact v4 freeze command",
    )
    preflight.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    preflight.add_argument(
        "--output-dir", type=Path, default=DEFAULT_SNAPSHOT_BUILD_DIR
    )
    snapshot = subparsers.add_parser(
        "verify-snapshot", help="verify the frozen snapshot"
    )
    snapshot.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    run = subparsers.add_parser("run", help="run formal clinical enrichment")
    run.add_argument("--campaign-report", type=Path, required=True)
    run.add_argument(
        "--snapshot-dir",
        type=Path,
        required=True,
        help="explicit source-hash-bound builder-v4 snapshot directory",
    )
    run.add_argument("--ppi", type=Path, default=DEFAULT_PPI)
    run.add_argument("--family", type=Path, default=DEFAULT_FAMILY)
    run.add_argument(
        "--family-release-manifest",
        type=Path,
        default=DEFAULT_FAMILY_RELEASE_MANIFEST,
        help="release_manifest.json that hash-binds the MMseqs2 family covariate",
    )
    run.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    run.add_argument("--cells", type=Path, default=DEFAULT_CELLS)
    run.add_argument("--genes", type=Path, default=DEFAULT_GENES)
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run.add_argument("--permutations", type=int, default=MIN_PERMUTATIONS)
    run.add_argument("--permutation-seed", type=int, default=DEFAULT_PERMUTATION_SEED)
    verify = subparsers.add_parser("verify", help="recompute and verify a formal run")
    verify.add_argument("--run-dir", type=Path, required=True)
    verify.add_argument(
        "--recorded-root",
        type=Path,
        help=(
            "legacy absolute workspace root recorded in the run manifest; must be "
            "paired with --workspace-root"
        ),
    )
    verify.add_argument(
        "--workspace-root",
        type=Path,
        help=(
            "absolute root of the copied/restored workspace; must be paired with "
            "--recorded-root and never falls back to recorded absolute paths"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "freeze":
            path = freeze_snapshot(args.raw_root, args.output_dir)
            print(path)
        elif args.command == "preflight-freeze":
            print(
                json.dumps(
                    preflight_snapshot_freeze(args.raw_root, args.output_dir),
                    indent=2,
                    sort_keys=True,
                )
            )
        elif args.command == "verify-snapshot":
            payload = validate_snapshot(args.snapshot_dir)
            print(json.dumps(payload["manifest_record"], sort_keys=True))
        elif args.command == "run":
            path = run_formal_analysis(
                args.campaign_report,
                snapshot_dir=args.snapshot_dir,
                ppi_path=args.ppi,
                family_path=args.family,
                family_release_manifest=args.family_release_manifest,
                fasta_path=args.fasta,
                cells_path=args.cells,
                genes_path=args.genes,
                output_root=args.output_root,
                n_permutations=args.permutations,
                permutation_seed=args.permutation_seed,
            )
            print(path)
        elif args.command == "verify":
            print(
                json.dumps(
                    verify_formal_run(
                        args.run_dir,
                        workspace_root=args.workspace_root,
                        recorded_root=args.recorded_root,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            raise AssertionError(f"unsupported command: {args.command}")
    except (ValidationError, OSError, ValueError, KeyError) as exc:
        print(f"[clinical-validation-v3] REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
