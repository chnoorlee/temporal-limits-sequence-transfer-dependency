#!/usr/bin/env python3
"""Build and verify a deterministic, hash-bound DepPLM public release bundle.

This is a packaging gate, not a replacement for the experiment-specific
verifiers.  It accepts only an explicit, workspace-relative file allowlist and
requires completion evidence for every prespecified analysis.  The resulting
ZIP parts are uncompressed and deterministic, which keeps already-compressed
prediction arrays byte-identical and makes archive hashes reproducible.
"""

from __future__ import annotations

import argparse
import builtins
import ctypes
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import io
import json
import math
import os
import re
import signal
import site
import stat
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
import urllib.parse
import uuid
import weakref
import zipfile
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from types import MappingProxyType, ModuleType
from typing import (
    Any,
    BinaryIO,
    IO,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    NoReturn,
    Sequence,
)


def _capture_executing_builder_source() -> tuple[Path, bytes, tuple[int, ...]]:
    reported = Path(__file__)
    lexical = Path(os.path.abspath(os.fspath(reported)))
    try:
        before_path = lexical.stat(follow_symlinks=False)
        with lexical.open("rb") as handle:
            before_open = os.fstat(handle.fileno())
            payload = handle.read()
            after_open = os.fstat(handle.fileno())
        after_path = lexical.stat(follow_symlinks=False)
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("executing release builder source is unavailable") from exc
    fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_size",
        "st_mtime_ns",
        "st_nlink",
    )
    snapshots = (before_open, after_open, after_path)
    if (
        any(
            getattr(before_path, field) != getattr(snapshot, field)
            for snapshot in snapshots
            for field in fields
        )
        or len(payload) != before_path.st_size
    ):
        raise RuntimeError("executing release builder source changed during import")
    fingerprint = (
        before_path.st_size,
        before_path.st_dev,
        before_path.st_ino,
        before_path.st_mode,
        before_path.st_mtime_ns,
        before_path.st_nlink,
    )
    return resolved, payload, fingerprint


(
    _EXECUTING_BUILDER_PATH,
    _EXECUTING_BUILDER_PAYLOAD,
    _EXECUTING_BUILDER_IDENTITY,
) = _capture_executing_builder_source()
_EXECUTING_BUILDER_SHA256 = hashlib.sha256(_EXECUTING_BUILDER_PAYLOAD).hexdigest()
_BASELINE_BUILTIN_IMPORT = builtins.__import__
_BASELINE_IMPORT_MODULE = importlib.import_module
_BASELINE_PATH_HOOKS = tuple(sys.path_hooks)
_BASELINE_MODULES = MappingProxyType(dict(sys.modules))
_BASELINE_STDLIB_MODULES = MappingProxyType(
    {
        name: module
        for name, module in _BASELINE_MODULES.items()
        if name.partition(".")[0] in getattr(sys, "stdlib_module_names", ())
    }
)
_MODULE_NOT_CAPTURED = object()
_TRUSTED_DEPENDENCY_MODULES: dict[str, Any] = {}
_TRUSTED_DEPENDENCY_BINDINGS: dict[tuple[str, str], Any] = {}
_TRUSTED_DEPENDENCY_STAR_EXPORTS: dict[str, tuple[str, ...]] = {}
_ACTIVE_TRUSTED_MODULE_SESSION: Any | None = None

ROOT = Path(__file__).resolve().parent.parent
SPEC_SCHEMA = "DepPLM-public-release-spec-v3"
MANIFEST_SCHEMA = "DepPLM-public-release-manifest-v3"
PREFLIGHT_SCHEMA = "DepPLM-public-release-preflight-v1"
DEFAULT_PART_SIZE = 1_900_000_000
MIN_PART_SIZE = 65_536
ZIP_FIXED_TIME = (1980, 1, 1, 0, 0, 0)
COMMON_COHORT_REPLAY_SCHEMA = "DepPLM-GATDep-common-cohort-verification-v3"
COMMON_COHORT_REPLAY_TOLERANCE = {"rtol": 1e-5, "atol": 1e-6}
COMMON_COHORT_REPLAY_INPUT_POLICY = (
    "not_embedded_reconstructed_from_hash_bound_project_inputs"
)
COMMON_COHORT_PREPROCESSING_POLICY = (
    "training_fitted_aggregate_statistics_and_feature_definitions_no_raw_cell_rows"
)
COMMON_COHORT_REDISTRIBUTION_STATUS = "not_cleared_for_public_redistribution"
SANGER_PLATFORM_RELEASE_BINDING_SCHEMA = "DepPLM-Sanger-platform-release-binding-v1"
LONG_PROTEIN_RELEASE_BINDING_SCHEMA = "DepPLM-long-protein-analysis-release-binding-v2"
ORGANOID_RELEASE_BINDING_SCHEMA = "DepPLM-Sanger-organoid-release-binding-v2"
DETACHED_TRUST_RECEIPT_SCHEMA = "DepPLM-detached-release-trust-receipt-v2"
DETACHED_TRUST_PROVENANCE = {
    "source_kind": "local_file_outside_workspace",
    "signature_verified": False,
    "immutable_storage_verified": False,
}
RELEASE_BINDING_SCHEMAS = {
    "sanger_platform_concordance": SANGER_PLATFORM_RELEASE_BINDING_SCHEMA,
    "long_protein_analysis": LONG_PROTEIN_RELEASE_BINDING_SCHEMA,
    "independent_biological_validation": ORGANOID_RELEASE_BINDING_SCHEMA,
}
DETACHED_TRUST_RECEIPT_ROLES = frozenset(RELEASE_BINDING_SCHEMAS)
SANGER_PLATFORM_PUBLIC_RUN_FILES = (
    "completion.json",
    "analysis.json",
    "source-bindings.json",
    "matched-cells.tsv",
    "matched-genes.tsv",
)
LONG_PROTEIN_PUBLIC_RUN_FILES = (
    "completion.json",
    "run-manifest.json",
    "summary.json",
    "per-gene-errors.npz",
)
ORGANOID_PUBLIC_RUN_FILES = (
    "completion.json",
    "analysis.json",
    "cellwise-metrics.tsv",
    "per-gene-correlations.tsv",
)
ORGANOID_LOCAL_ONLY_RUN_FILES = (
    "predictions.npz",
    "training-records.json",
    "prediction-lock.json",
    "outcome-snapshot.RData",
    "source-bindings.json",
)
ORGANOID_COMPLETION_ARTIFACTS = frozenset(
    (*ORGANOID_PUBLIC_RUN_FILES[1:], *ORGANOID_LOCAL_ONLY_RUN_FILES)
)
CLINICAL_PUBLIC_RUN_FILES = (
    "completion.json",
    "verification.json",
    "run-manifest.json",
    "rankings.tsv",
    "tests.tsv",
    "results.json",
)
CLINICAL_VERIFICATION_ARTIFACTS = CLINICAL_PUBLIC_RUN_FILES[2:]

REQUIRED_EVIDENCE_ROLES = frozenset(
    {
        "core_campaign",
        "esm_embedding_provenance",
        "family_holdout",
        "k_token_pmi",
        "attention_faithfulness",
        "cnv_sensitivity",
        "external_architecture_baseline",
        "clinical_annotation_concordance",
        "sanger_platform_concordance",
        "long_protein_analysis",
        "independent_biological_validation",
    }
)
ALLOWED_CATEGORIES = frozenset(
    {
        "source",
        "test",
        "manuscript",
        "documentation",
        "evidence",
        "prediction",
        "provenance",
    }
)
ESM_PUBLIC_FILES = frozenset(
    {
        "data/processed/esm2_t6_8M_residue_chunked.npy",
        "data/processed/esm2_t6_8M_residue_chunked.l2.npy",
        "data/processed/esm2_t6_8M_residue_chunked.names.txt",
        "data/processed/esm2_t6_8M_residue_chunked.manifest.json",
    }
)
K_TOKEN_PMI_VARIANTS = (
    "k16-frequency",
    "k32-frequency",
    "k64-frequency",
    "k64-vep-impact",
    "k64-driver-lof",
    "k64-frequency-unit",
    "k64-frequency-valid-only",
)
K_TOKEN_PMI_SEEDS = (42, 43, 44)
K_TOKEN_PMI_FOLDS = tuple(range(5))
K_TOKEN_PMI_INDEPENDENT_GATES = frozenset(
    {
        "detached_receipt_report_binding",
        "primary_report_contract",
        "campaign_plan_and_completion_hashes",
        "primary_verifier_source_hash",
        "gene_effect_input_hash",
        "gene_and_cell_identity_hashes",
        "exact_105_identity_coverage",
        "all_105_prediction_hashes",
        "prediction_zip_npy_shape_and_resource_bounds",
        "all_105_sidecar_hashes_and_gzip_integrity",
        "sidecar_gzip_csv_schema_and_resource_bounds",
        "outer_test_exact_once",
        "outer_train_gene_means_recomputed",
        "four_paired_metrics_recomputed",
        "end_of_audit_stability",
    }
)
CLINICAL_PUBLICATION_GENERATOR_SOURCE = (
    "paper/nar_major_revision/figures/gen_fig_corrected_campaign.py"
)
CORRECTED_CAMPAIGN_FIGURE_PDF = (
    "paper/nar_major_revision/figures/fig_corrected_campaign.pdf"
)
CORRECTED_CAMPAIGN_FIGURE_PNG = (
    "paper/nar_major_revision/figures/fig_corrected_campaign.png"
)
CORRECTED_CAMPAIGN_TABLE = (
    "paper/nar_major_revision/sections/generated_corrected_campaign_table.tex"
)
CORRECTED_CAMPAIGN_PROVENANCE = (
    "paper/nar_major_revision/figures/fig_corrected_campaign.provenance.json"
)
CORRECTED_CAMPAIGN_PROVENANCE_SCHEMA = (
    "DepPLM-corrected-campaign-publication-provenance-v1"
)
CORRECTED_CAMPAIGN_OUTPUT_FRAME_LIMITS = MappingProxyType(
    {
        "campaign_table": 8 * 1024 * 1024,
        "figure_pdf": 16 * 1024 * 1024,
        "figure_png": 64 * 1024 * 1024,
        "clinical_table": 8 * 1024 * 1024,
    }
)
CORRECTED_CAMPAIGN_INPUT_FRAME_LIMITS = MappingProxyType(
    {
        "generator_source": 1024 * 1024,
        "campaign_report": 16 * 1024 * 1024,
        "mode": 64,
        "clinical_results": 16 * 1024 * 1024,
        "clinical_completion_sha256": 64,
        "clinical_results_sha256": 64,
    }
)
CORRECTED_CAMPAIGN_STDOUT_LIMIT = sum(
    CORRECTED_CAMPAIGN_OUTPUT_FRAME_LIMITS.values()
) + 8 * len(CORRECTED_CAMPAIGN_OUTPUT_FRAME_LIMITS)
CORRECTED_CAMPAIGN_STDERR_LIMIT = 1024 * 1024
BOUNDED_SUBPROCESS_CLEANUP_TIMEOUT = 2.0
CORRECTED_CAMPAIGN_CORE_OUTPUTS = MappingProxyType(
    {
        "figure_pdf": CORRECTED_CAMPAIGN_FIGURE_PDF,
        "figure_png": CORRECTED_CAMPAIGN_FIGURE_PNG,
        "campaign_table": CORRECTED_CAMPAIGN_TABLE,
    }
)
CORRECTED_CAMPAIGN_CLINICAL_TABLE = (
    "paper/nar_major_revision/sections/generated_clinical_table.tex"
)
REQUIRED_SOURCE_FILES = frozenset(
    {
        "code/03_embed_proteins.py",
        "code/aggregate_corrected_campaign.py",
        "code/attention_faithfulness.py",
        "code/build_public_release.py",
        "code/clinical_snapshot_builder_v4.py",
        "code/clinical_validation_v2.py",
        "code/cnv_sensitivity.py",
        "code/corrected_protocol.py",
        "code/embedding_provenance.py",
        "code/family_holdout.py",
        "code/gatdep_data.py",
        "code/gatdep_official_baseline.py",
        "code/k_token_pmi_protocol.py",
        "code/long_protein_analysis.py",
        "code/models.py",
        "code/organoid_independent_validation.py",
        "code/run_corrected_campaign.py",
        "code/run_corrected_experiments.py",
        "code/run_family_holdout_campaign.py",
        "code/run_family_holdout_experiment.py",
        "code/run_gatdep_common_depplm.py",
        "code/run_k_token_pmi_sensitivity.py",
        "code/sanger_independent_validation.py",
        "code/sanger_nonoverlap_validation.py",
        "code/verify_k_token_pmi_sensitivity.py",
        "code/verify_family_holdout_campaign.py",
        "code/verify_family_holdout_experiment.py",
        "code/verify_gatdep_common_depplm.py",
        "code/verify_gatdep_official_baseline.py",
        "code/verify_long_protein_analysis.py",
        CLINICAL_PUBLICATION_GENERATOR_SOURCE,
    }
)
TRUSTED_WORKSPACE_MODULE_PATHS = MappingProxyType(
    {
        Path(relative).stem: relative
        for relative in REQUIRED_SOURCE_FILES
        if (
            relative.startswith("code/")
            or relative == CLINICAL_PUBLICATION_GENERATOR_SOURCE
        )
        and Path(relative).stem.isidentifier()
        and Path(relative).stem != "build_public_release"
    }
)
TRUSTED_WORKSPACE_MODULE_NAMES = frozenset(TRUSTED_WORKSPACE_MODULE_PATHS)
PROJECT_CODE_PATH_SENTINELS = (
    "build_public_release.py",
    "corrected_protocol.py",
    "embedding_provenance.py",
)
REQUIRED_TEST_FILES = frozenset(
    {
        "tests/test_attention_faithfulness.py",
        "tests/test_build_public_release.py",
        "tests/test_campaign_aggregator.py",
        "tests/test_clinical_validation_v2.py",
        "tests/test_gen_fig_corrected_campaign.py",
        "tests/test_cnv_sensitivity.py",
        "tests/test_corrected_protocol.py",
        "tests/test_embed_proteins.py",
        "tests/test_embedding_provenance.py",
        "tests/test_family_holdout.py",
        "tests/test_gatdep_common_depplm.py",
        "tests/test_gatdep_official_baseline.py",
        "tests/test_k_token_pmi_protocol.py",
        "tests/test_verify_k_token_pmi_independent.py",
        "tests/test_long_protein_analysis.py",
        "tests/test_verify_long_protein_analysis.py",
        "tests/test_organoid_independent_validation.py",
        "tests/test_run_corrected_campaign.py",
        "tests/test_sanger_independent_validation.py",
        "tests/test_sanger_nonoverlap_validation.py",
    }
)
REQUIRED_MANUSCRIPT_FILES = frozenset(
    {
        "paper/nar_major_revision/main.tex",
        "paper/nar_major_revision/oup-abbrvnat.bst",
        "paper/nar_major_revision/oup-authoring-template.cls",
        "paper/nar_major_revision/oup-plain.bst",
        "paper/nar_major_revision/references.bib",
        "paper/nar_major_revision/sections/abstract.tex",
        "paper/nar_major_revision/sections/back_matter.tex",
        "paper/nar_major_revision/sections/data_availability.tex",
        "paper/nar_major_revision/sections/discussion.tex",
        "paper/nar_major_revision/sections/introduction.tex",
        "paper/nar_major_revision/sections/methods.tex",
        "paper/nar_major_revision/sections/results.tex",
        "paper/nar_major_revision/figures/README.md",
        CORRECTED_CAMPAIGN_FIGURE_PDF,
        CORRECTED_CAMPAIGN_FIGURE_PNG,
        CORRECTED_CAMPAIGN_PROVENANCE,
        "paper/nar_major_revision/sections/generated_clinical_table.tex",
        CORRECTED_CAMPAIGN_TABLE,
    }
)
ACTIVE_MANUSCRIPT_DEPENDENCIES = frozenset(
    {
        "paper/nar_major_revision/main.tex",
        "paper/nar_major_revision/oup-abbrvnat.bst",
        "paper/nar_major_revision/oup-authoring-template.cls",
        "paper/nar_major_revision/references.bib",
        "paper/nar_major_revision/figures/fig_corrected_campaign.pdf",
        "paper/nar_major_revision/sections/abstract.tex",
        "paper/nar_major_revision/sections/back_matter.tex",
        "paper/nar_major_revision/sections/data_availability.tex",
        "paper/nar_major_revision/sections/discussion.tex",
        "paper/nar_major_revision/sections/generated_corrected_campaign_table.tex",
        "paper/nar_major_revision/sections/introduction.tex",
        "paper/nar_major_revision/sections/methods.tex",
        "paper/nar_major_revision/sections/results.tex",
    }
)
REVIEWED_INACTIVE_MANUSCRIPT_FILES = (
    REQUIRED_MANUSCRIPT_FILES - ACTIVE_MANUSCRIPT_DEPENDENCIES
)
REQUIRED_DOCUMENTATION_FILES = frozenset(
    {
        "docs/revision_v2/PUBLIC_ARCHIVE_PROTOCOL.md",
        "docs/revision_v2/ORGANOID_INDEPENDENT_VALIDATION_PROTOCOL.md",
        "docs/revision_v2/SANGER_NONOVERLAP_PROTOCOL.md",
    }
)
REQUIRED_CONTENT_FILES = {
    "source": REQUIRED_SOURCE_FILES,
    "test": REQUIRED_TEST_FILES,
    "manuscript": REQUIRED_MANUSCRIPT_FILES,
    "documentation": REQUIRED_DOCUMENTATION_FILES,
}
REVIEWED_PUBLICATION_FILES = {
    "manuscript": REQUIRED_MANUSCRIPT_FILES,
    "documentation": REQUIRED_DOCUMENTATION_FILES,
}
REVIEWED_PDF_FILES = frozenset({CORRECTED_CAMPAIGN_FIGURE_PDF})
PLACEHOLDER_RE = re.compile(
    r"(?:\bTODO\b|\bTBD\b|\bUNKNOWN\b|\bPLACEHOLDER\b|\bANONYMOUS\b|\[.+?\])",
    re.IGNORECASE,
)
MANUSCRIPT_BLOCKER_RE = re.compile(
    r"(?:\bTODO\b|\bTBD\b|\bPLACEHOLDER\b|\bANONYMOUS\b|"
    r"\bSUBMISSION\s+BLOCKER\b|\bAUTHOR\s+CONFIRMATION\s+REQUIRED\b|"
    + re.escape(chr(92) + "PackageError")
    + r"\b)",
    re.IGNORECASE,
)
DOI_RE = re.compile(r"^10\.[0-9]{4,9}/[-._;()/:A-Z0-9]+$", re.IGNORECASE | re.ASCII)
ORCID_RE = re.compile(r"^[0-9]{4}-[0-9]{4}-[0-9]{4}-[0-9]{3}[0-9X]$", re.ASCII)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CLINICAL_RESULT_SCHEMA = "DepPLM-OpenTargets-clinical-validation-v3"
CLINICAL_VERIFICATION_SCHEMA = "DepPLM-OpenTargets-clinical-verification-v3"
CLINICAL_COMPLETION_SCHEMA = "DepPLM-OpenTargets-clinical-completion-v3"
CLINICAL_COMPLETION_FIELDS = frozenset(
    {"schema", "scientific_result", "verification", "results"}
)
CLINICAL_VERIFICATION_FIELDS = frozenset(
    {
        "schema",
        "verified_utc",
        "scientific_result",
        "verification_method",
        "artifacts",
        "analysis_sha256",
    }
)
CLINICAL_REPLAY_FIELDS = frozenset(
    {
        "schema",
        "scientific_result",
        "run_dir",
        "verification_scope",
        "path_identity_retained",
        "marker_record",
        "analysis_sha256",
        "campaign_source_runs_rehashed",
        "campaign_source_artifacts_rehashed",
    }
)
CLINICAL_REPLAY_VERIFICATION_SCOPE = "point_in_time_hash_verification"
CLINICAL_VERIFICATION_METHOD = (
    "all bound inputs rehashed; 99-run/P1 exact-once gates rerun; deterministic "
    "analysis artifacts regenerated and byte-compared"
)
CLINICAL_PUBLICATION_TABLE = (
    "paper/nar_major_revision/sections/generated_clinical_table.tex"
)
CLINICAL_PUBLICATION_COMPLETION_PREFIX = "% Clinical completion SHA-256: "
CLINICAL_PUBLICATION_RESULTS_PREFIX = "% Clinical results SHA-256: "
CLINICAL_EXPECTED_SOURCE_RUNS = 99
CLINICAL_EXPECTED_SOURCE_ARTIFACTS = 1_089
ORGANOID_COMPLETION_FIELDS = frozenset(
    {
        "schema",
        "created_utc",
        "scientific_result",
        "run_id",
        "freeze",
        "trusted_freeze_sha256",
        "trusted_prediction_lock_sha256",
        "outcome_snapshot_sha256",
        "analysis_canonical_sha256",
        "artifacts",
        "primary_results",
        "conclusion",
        "verification_requirement",
    }
)
ORGANOID_VERIFICATION_FIELDS = frozenset(
    {
        "schema",
        "created_utc",
        "scientific_result",
        "verification_passed",
        "verification_scope",
        "run_directory",
        "trusted_freeze_sha256",
        "trusted_prediction_lock_sha256",
        "trusted_completion_sha256",
        "completion",
        "analysis_canonical_sha256",
        "recomputed_primary_results",
        "recomputed_conclusion",
    }
)
ORGANOID_VERIFICATION_SCOPE = (
    "same-code deterministic replay and artifact-integrity verification; "
    "not independent scientific verification"
)
ORGANOID_VERIFICATION_REQUIREMENT = (
    "Supply the externally retained freeze, prediction-lock, and completion "
    "SHA-256 values to verify."
)
_POSIX_SEPARATOR = chr(47)
_PORTABLE_ENV = _POSIX_SEPARATOR + _POSIX_SEPARATOR.join(("usr", "bin", "env"))
_PORTABLE_BIN = _POSIX_SEPARATOR + _POSIX_SEPARATOR.join(("bin",))
PORTABLE_SHEBANGS = frozenset(
    {
        f"#!{_PORTABLE_ENV} python",
        f"#!{_PORTABLE_ENV} python3",
        f"#!{_PORTABLE_BIN}{_POSIX_SEPARATOR}bash",
        f"#!{_PORTABLE_BIN}{_POSIX_SEPARATOR}sh",
    }
)
FORBIDDEN_PARTS = frozenset(
    {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        "credentials",
        "cmp-cache",
        "cmp-raw-cache",
        "cmp-sanger-nonoverlap-cache",
        "cmp-snapshots",
        "cmp_cache",
        "cmp_raw_cache",
        "cmp_sanger_nonoverlap_cache",
        "cmp_snapshots",
        "data-cache",
        "data_cache",
        "original-data",
        "original_data",
        "private",
        "private-data",
        "private_data",
        "raw",
        "raw-data",
        "raw_data",
        "sanger_organoid_2026",
        "sanger-project-score",
        "sanger_project_score",
        "secrets",
        "source-data",
        "source_data",
        "third-party",
        "third-party-cache",
        "third-party-checkout",
        "third_party",
        "third_party_cache",
        "third_party_checkout",
        "thirdparty",
        "vendor-cache",
        "vendor_cache",
        "unredacted",
    }
)
FORBIDDEN_SUFFIXES = frozenset(
    {".env", ".jks", ".key", ".kdbx", ".p12", ".pem", ".pfx"}
)
FORBIDDEN_FILE_RE = re.compile(
    r"^(?:(?:access[-_]?key|api[-_]?key|auth[-_]?key|client[-_]?secret|"
    r"credential|credentials|password|passwords|private[-_]key|secret|secrets|"
    r"service[-_]account)(?:[._-].*)?|(?:token|tokens)\."
    r"(?:env|json|txt|yaml|yml)|(?:api|access|auth|refresh)[-_]token"
    r"(?:[._-].*)?)$",
    re.IGNORECASE,
)
FORBIDDEN_RAW_FILE_RE = re.compile(
    r"^(?:cmp[-_]?(?:expression|mutation|mutations|rna)|original[-_]data|"
    r"project[-_]?score(?:2)?|raw|raw[-_]data|source[-_]data|unredacted)"
    r"(?:[._-].*)?$",
    re.IGNORECASE,
)
FORBIDDEN_SENSITIVE_DATA_NAMES = frozenset(
    {
        "crisprgeneeffect.csv",
        "project_score2_fitness_scores_sanger_v2_broad_21q2_20250624.zip",
    }
)
FORBIDDEN_SSH_NAMES = frozenset(
    {
        "id_dsa",
        "id_dsa.pub",
        "id_ecdsa",
        "id_ecdsa.pub",
        "id_ed25519",
        "id_ed25519.pub",
        "id_rsa",
        "id_rsa.pub",
    }
)
PUBLIC_TEXT_CATEGORIES = frozenset(
    {
        "documentation",
        "evidence",
        "manuscript",
        "prediction",
        "provenance",
        "source",
        "test",
    }
)
PUBLIC_TEXT_SUFFIXES = frozenset(
    {
        ".bib",
        ".bst",
        ".cls",
        ".csv",
        ".json",
        ".md",
        ".py",
        ".r",
        ".sh",
        ".ps1",
        ".bat",
        ".cmd",
        ".tex",
        ".toml",
        ".tsv",
        ".txt",
        ".yaml",
        ".yml",
    }
)
_FILE_URI_PATTERN = re.compile(
    r"(?i)(?<![A-Za-z0-9])file:" + re.escape(_POSIX_SEPARATOR) + r"+[^\s\"'<>]*"
)
_REGEX_BACKSLASH = re.escape(chr(92))
_TEXT_UNC_SEGMENT_PATTERN = (
    r"[^\s" + _REGEX_BACKSLASH + re.escape(chr(34) + "'<>|") + r"]+"
)
_TEXT_UNC_PATTERN = re.compile(
    r"(?<!"
    + _REGEX_BACKSLASH
    + r")"
    + _REGEX_BACKSLASH
    + r"{2}(?!"
    + _REGEX_BACKSLASH
    + r")(?![?.]"
    + _REGEX_BACKSLASH
    + r")"
    + _TEXT_UNC_SEGMENT_PATTERN
    + _REGEX_BACKSLASH
    + _TEXT_UNC_SEGMENT_PATTERN
    + r"(?:"
    + _REGEX_BACKSLASH
    + _TEXT_UNC_SEGMENT_PATTERN
    + r")*"
)
_WINDOWS_EXTENDED_NAMESPACE_PATTERN = re.compile(
    r"(?i)(?<!"
    + _REGEX_BACKSLASH
    + r")(?:(?:"
    + _REGEX_BACKSLASH
    + r"{2}|"
    + _REGEX_BACKSLASH
    + r"{4})[?.](?:"
    + _REGEX_BACKSLASH
    + r"{1,2})|"
    + _REGEX_BACKSLASH
    + r"{1,2}[?]{2}"
    + _REGEX_BACKSLASH
    + r"{1,2}|"
    + _REGEX_BACKSLASH
    + r"{1,2}Device"
    + _REGEX_BACKSLASH
    + r"{1,2})"
)
_FORWARD_UNC_PATTERN = re.compile(
    r"(?<![A-Za-z0-9:+."
    + re.escape(_POSIX_SEPARATOR)
    + r"])"
    + re.escape(_POSIX_SEPARATOR)
    + r"{2}(?![?.]/)[^\s/\"'<>|]+"
    + re.escape(_POSIX_SEPARATOR)
    + r"[^\s/\"'<>|]+(?:"
    + re.escape(_POSIX_SEPARATOR)
    + r"[^\s\"'<>|]+)*"
)
_FORWARD_WINDOWS_EXTENDED_NAMESPACE_PATTERN = re.compile(
    r"(?i)(?<![A-Za-z0-9:+./])(?://[?.]/|/[?]{2}/)"
)
_URL_EMBEDDED_WINDOWS_DRIVE_PATTERN = re.compile(
    r"(?i)(?<![A-Za-z0-9+.-])[A-Za-z]"
    + re.escape(":")
    + r"["
    + _REGEX_BACKSLASH
    + re.escape(_POSIX_SEPARATOR)
    + r"]+[^\s"
    + _REGEX_BACKSLASH
    + re.escape(chr(34) + "'<>|")
    + r"]*"
)
_URL_EMBEDDED_WINDOWS_NAMESPACE_PATTERN = re.compile(
    r"(?i)(?:(?:"
    + _REGEX_BACKSLASH
    + r"{2}|"
    + _REGEX_BACKSLASH
    + r"{4})[?.](?:"
    + _REGEX_BACKSLASH
    + r"{1,2})|"
    + _REGEX_BACKSLASH
    + r"{1,2}[?]{2}"
    + _REGEX_BACKSLASH
    + r"{1,2}|"
    + _REGEX_BACKSLASH
    + r"{1,2}Device"
    + _REGEX_BACKSLASH
    + r"{1,2}|//[?.]/|/[?]{2}/)"
)
_MAX_AUTHORITY_URL_DECODE_PASSES = 8
_AUTHORITY_URI_PATTERN = re.compile(
    r"(?i)\b(?!file:)[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'<>{}]+"
)
_TEX_PDFMARK_NAME_PATTERN = re.compile(
    r"/(?:CropBox|MediaBox|TrimBox|BleedBox|PUT)(?=$|[\s\[])"
)
_WINDOWS_CMD_MKLINK_SWITCH_PATTERN = re.compile(r"(?<=[\"'])/(?:c|J)(?=[\"'])")
_POSIX_ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9:"
    + re.escape(_POSIX_SEPARATOR)
    + r"\\])"
    + re.escape(_POSIX_SEPARATOR)
    + r"(?!"
    + re.escape(_POSIX_SEPARATOR)
    + r")[^\s\"'<>]*"
)
_GENERAL_POSIX_ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9._*?~+:>}"
    + re.escape(_POSIX_SEPARATOR)
    + r"\\-])"
    + re.escape(_POSIX_SEPARATOR)
    + r"(?!"
    + re.escape(_POSIX_SEPARATOR)
    + r")(?=[\w._~+-])"
    + r"[^\s\"'<>|{}\\]*"
)
LOCAL_ABSOLUTE_PATH_PATTERNS = (
    re.compile(
        r"(?<![A-Za-z0-9])(?:[A-Za-z]"
        + re.escape(":")
        + r"["
        + _REGEX_BACKSLASH
        + re.escape(_POSIX_SEPARATOR)
        + r"]+)[^\s"
        + _REGEX_BACKSLASH
        + re.escape(chr(34) + "'<>|")
        + r"]*"
    ),
    _WINDOWS_EXTENDED_NAMESPACE_PATTERN,
    _FORWARD_WINDOWS_EXTENDED_NAMESPACE_PATTERN,
    re.compile(
        r"(?<!"
        + _REGEX_BACKSLASH
        + r")"
        + _REGEX_BACKSLASH
        + r"{2,}(?!"
        + _REGEX_BACKSLASH
        + r")[^\s"
        + _REGEX_BACKSLASH
        + re.escape(chr(34) + "'<>")
        + r"]*"
    ),
    _FILE_URI_PATTERN,
    _POSIX_ABSOLUTE_PATH_PATTERN,
    re.compile(
        r"(?<![A-Za-z0-9:"
        + re.escape(_POSIX_SEPARATOR)
        + r"\\])"
        + re.escape(_POSIX_SEPARATOR)
        + r"{2,}[^\s\"'<>]*"
    ),
)
TEXT_LOCAL_ABSOLUTE_PATH_PATTERNS = (
    LOCAL_ABSOLUTE_PATH_PATTERNS[0],
    _WINDOWS_EXTENDED_NAMESPACE_PATTERN,
    _FORWARD_WINDOWS_EXTENDED_NAMESPACE_PATTERN,
    _TEXT_UNC_PATTERN,
    _FORWARD_UNC_PATTERN,
    _FILE_URI_PATTERN,
    _GENERAL_POSIX_ABSOLUTE_PATH_PATTERN,
)
SOURCE_LOCAL_ABSOLUTE_PATH_PATTERNS = (
    LOCAL_ABSOLUTE_PATH_PATTERNS[0],
    _WINDOWS_EXTENDED_NAMESPACE_PATTERN,
    _FORWARD_WINDOWS_EXTENDED_NAMESPACE_PATTERN,
    _TEXT_UNC_PATTERN,
    _FORWARD_UNC_PATTERN,
    re.compile(
        r"(?:[rubfRUBF]{0,2}[\"'])(?:\\{2}|\\{4})(?!\\)"
        r"[A-Za-z0-9._$-]+(?:\\|\\{2})[A-Za-z0-9._$-]+"
    ),
    _FILE_URI_PATTERN,
    _GENERAL_POSIX_ABSOLUTE_PATH_PATTERN,
)

_URI_REGEX_SLASH = re.escape(chr(47))


def _root_relative_uri_regex(*parts: str) -> str:
    return _URI_REGEX_SLASH + _URI_REGEX_SLASH.join(parts)


_MODEL_FILES_URI_REGEX = _root_relative_uri_regex("model_files")
_MODEL_DATASETS_URI_REGEX = _root_relative_uri_regex(
    "models", r"\{model_id\}", "datasets", "mutations"
)
_SIDM_MODEL_DATASETS_URI_REGEX = _root_relative_uri_regex(
    "models", r"\{sidm\}", "datasets", "mutations"
)
_CMP_DOWNLOAD_URI_REGEX = _root_relative_uri_regex(
    "cmp", "download", r"\{sidm\}_rnaseq[.]csv[.]zip"
)
_CMP_TEST_DOWNLOAD_URI_REGEX = _root_relative_uri_regex(
    "cmp", "download", "SIDM00001_rnaseq[.]csv[.]zip"
)
_SIDM_TEST_DATASETS_URI_REGEX = _root_relative_uri_regex(
    "models", "SIDM00001", "datasets", "mutations"
)
_CMP_TEST_FORWARD_UNC_PREFIX_REGEX = (
    _URI_REGEX_SLASH * 2
    + _URI_REGEX_SLASH.join(
        ("api[.]cellmodelpassports[.]sanger[.]ac[.]uk", "models", "SIDM00001")
    )
    + _URI_REGEX_SLASH
)
_PROC_DESCRIPTOR_PATH_REGEX = _root_relative_uri_regex(
    "proc", "self", "fd", r"\{descriptor\}"
)
_DEV_DESCRIPTOR_PATH_REGEX = _root_relative_uri_regex("dev", "fd", r"\{descriptor\}")


_REGISTERED_ROOT_RELATIVE_URI_LINES: dict[str, tuple[re.Pattern[str], ...]] = {
    "code/sanger_nonoverlap_validation.py": (
        re.compile(
            r'^\s*required\s*=\s*\{"'
            + _MODEL_FILES_URI_REGEX
            + r'",\s*"'
            + _MODEL_DATASETS_URI_REGEX
            + r'"\}\s*$'
        ),
        re.compile(r'^\s*expected_path\s*=\s*"' + _MODEL_FILES_URI_REGEX + r'",\s*$'),
        re.compile(r'^\s*return\s+f"' + _CMP_DOWNLOAD_URI_REGEX + r'"\s*$'),
        re.compile(
            r'^\s*expected_path\s*=\s*f"' + _SIDM_MODEL_DATASETS_URI_REGEX + r'",\s*$'
        ),
        re.compile(
            r'^\s*expected_path\s*=\s*f"' + _SIDM_MODEL_DATASETS_URI_REGEX + r'"\s*$'
        ),
    ),
    "docs/revision_v2/SANGER_NONOVERLAP_PROTOCOL.md": (
        re.compile(
            r"^\s*- official CMP `"
            + _MODEL_FILES_URI_REGEX
            + r"` JSON:API snapshot;\s*$"
        ),
    ),
    "tests/test_sanger_nonoverlap_validation.py": (
        re.compile(r'^\s*path\s*=\s*"' + _CMP_TEST_DOWNLOAD_URI_REGEX + r'"\s*$'),
        re.compile(
            r'^\s*expected_path\s*=\s*"' + _SIDM_TEST_DATASETS_URI_REGEX + r'",\s*$'
        ),
        re.compile(r'^\s*path\s*=\s*"' + _SIDM_TEST_DATASETS_URI_REGEX + r'"\s*$'),
        re.compile(r'^\s*"' + _SIDM_TEST_DATASETS_URI_REGEX + r'"\s*$'),
    ),
}

_REGISTERED_PORTABLE_SYSTEM_PATH_LINES: dict[str, tuple[re.Pattern[str], ...]] = {
    "code/organoid_independent_validation.py": (
        re.compile(
            r'^\s*proc_descriptor\s*=\s*Path\(f"'
            + _PROC_DESCRIPTOR_PATH_REGEX
            + r'"\)\s*$'
        ),
        re.compile(
            r'^\s*dev_descriptor\s*=\s*Path\(f"'
            + _DEV_DESCRIPTOR_PATH_REGEX
            + r'"\)\s*$'
        ),
    ),
    "code/verify_long_protein_analysis.py": (
        re.compile(r'^\s*Path\(f"' + _PROC_DESCRIPTOR_PATH_REGEX + r'"\),\s*$'),
        re.compile(r'^\s*Path\(f"' + _DEV_DESCRIPTOR_PATH_REGEX + r'"\),\s*$'),
    ),
}

_REGISTERED_SOURCE_FIXTURE_PATH_LINES: dict[str, tuple[re.Pattern[str], ...]] = {
    "tests/test_sanger_nonoverlap_validation.py": (
        re.compile(r'^\s*"' + _CMP_TEST_FORWARD_UNC_PREFIX_REGEX + r'"\s*$'),
    ),
}

_REGISTERED_STRING_SUFFIX_LINES: dict[str, dict[int, re.Pattern[str]]] = {
    "tests/test_organoid_independent_validation.py": {
        1242: re.compile(
            r'^\s*assert first[.]result[(]timeout=5[)]\["run_directory"\]'
            r'[.]endswith[(]"' + _URI_REGEX_SLASH + r'first"[)]\s*$'
        ),
        1243: re.compile(
            r'^\s*assert second[.]result[(]timeout=5[)]\["run_directory"\]'
            r'[.]endswith[(]"' + _URI_REGEX_SLASH + r'second"[)]\s*$'
        ),
        1275: re.compile(
            r'^\s*assert completed\["run_directory"\][.]endswith[(]'
            r'"' + _URI_REGEX_SLASH + r'recovered"[)]\s*$'
        ),
        1428: re.compile(
            r'^\s*assert completed\["run_directory"\][.]endswith[(]'
            r'"' + _URI_REGEX_SLASH + r'outer"[)]\s*$'
        ),
    }
}

_REGISTERED_LOCAL_PATH_SOURCE_LINE_SHA256: dict[str, dict[int, str]] = {
    "code/clinical_snapshot_builder_v4.py": {
        2519: "b627421aa461b7387efe7317d78456fd0ab073a54882c84cf9d9039f16bfa995",
    },
    "code/clinical_validation_v2.py": {
        3336: "9e215f976dc59054fc0365d33191e06d61467a82e1d05975652989aee0174118",
        3520: "9e215f976dc59054fc0365d33191e06d61467a82e1d05975652989aee0174118",
        4462: "9a552803e18f2758e7e56eb0b276389c01038651344d61b5318765e769899d50",
        4463: "980936a59d927963836e0c6124610cb4033aa095ade0867779ccc4102b130b46",
    },
    "code/verify_long_protein_analysis.py": {
        8515: "569d72d13d98b55362cab62fc92f7f6c7d04d6b65293934dcdc24f80e323d9ed",
        8516: "96147f52762f5985862dd786e95e61066b0da4c37bb0ff95121329024b61d33f",
        8517: "9131b16e21de34bbe045c7cb8a8db9bb2e601bcbe519fb0cffe54c881429dd85",
    },
    "tests/test_clinical_validation_v2.py": {
        1996: "4d19464ad3d2e70cda5308065df89ecf681ca06a9638c1bcfd047f3afe9b5b78",
    },
}


class ReleaseValidationError(ValueError):
    """Raised when a release candidate violates the fail-closed contract."""


@dataclass(frozen=True)
class SelectedFile:
    relative: str
    category: str
    source: Path
    bytes: int
    sha256: str
    device: int
    inode: int
    mode: int
    modified_ns: int
    links: int
    payload: builtins.bytes


@dataclass(frozen=True)
class FileFingerprint:
    bytes: int
    sha256: str
    device: int
    inode: int
    mode: int
    modified_ns: int
    links: int


@dataclass(frozen=True)
class TrustedModuleSource:
    name: str
    path: Path
    payload: bytes
    fingerprint: FileFingerprint


@dataclass(frozen=True)
class GatDepOfficialFileBinding:
    relative: str
    payload: bytes
    fingerprint: FileFingerprint
    git_object_id: str


class _TrustedWorkspaceModuleSession(
    importlib.abc.MetaPathFinder, importlib.abc.Loader
):
    """Load reviewed first-party modules in a constrained import environment."""

    _IMPORT_LOCK = threading.RLock()

    def __init__(
        self,
        root: Path,
        *,
        expected_sources: Mapping[str, SelectedFile] | None = None,
    ) -> None:
        self.root = root.resolve(strict=True)
        self._expected_sources = dict(expected_sources or {})
        self._sources: dict[str, TrustedModuleSource] = {}
        self._loaded: dict[str, ModuleType] = {}
        self._saved_modules: dict[str, ModuleType] = {}
        self._saved_dependency_modules: dict[str, Any] = {}
        self._saved_sys_path: list[str] = []
        self._saved_meta_path: list[Any] = []
        self._saved_path_hooks: list[Any] = []
        self._saved_path_importer_cache: dict[Any, Any] = {}
        self._saved_builtin_import: Any = _BASELINE_BUILTIN_IMPORT
        self._saved_import_module: Any = _BASELINE_IMPORT_MODULE
        self._session_dependency_modules: dict[str, Any] = {}
        self._session_dependency_bindings: dict[tuple[str, str], Any] = {}
        self._session_dependency_star_exports: dict[str, tuple[str, ...]] = {}
        self._dependency_root_snapshots: dict[str, dict[str, Any]] = {}
        self._dependency_binding_snapshots: dict[tuple[str, str], Any] = {}
        self._validated_dependency_modules: dict[str, tuple[Any, tuple[Any, ...]]] = {}
        self._active_dependency_roots: set[str] = set()
        self._prepared_stdlib_roots: set[str] = set()
        self._stdlib_search_paths: tuple[Path, ...] = ()
        self._site_search_paths: tuple[Path, ...] = ()
        self._guard_depth = 0
        self._owner_thread_id: int | None = None
        self._active = False
        self._lock_held = False

    @staticmethod
    def _is_project_code_path(entry: str) -> bool:
        if not entry:
            return False
        try:
            candidate = Path(entry).resolve(strict=True)
        except (OSError, TypeError, ValueError):
            return False
        return candidate.is_dir() and any(
            (candidate / sentinel).is_file() for sentinel in PROJECT_CODE_PATH_SENTINELS
        )

    @staticmethod
    def _is_relative_to(path: Path, parent: Path) -> bool:
        try:
            path.relative_to(parent)
        except ValueError:
            return False
        return True

    @staticmethod
    def _resolved_import_entry(entry: str) -> Path | None:
        if not entry:
            return None
        try:
            return Path(entry).resolve(strict=False)
        except (OSError, TypeError, ValueError):
            return None

    def _trusted_search_path(
        self, entries: Sequence[str]
    ) -> tuple[list[str], tuple[Path, ...], tuple[Path, ...]]:
        try:
            cwd = Path.cwd().resolve(strict=True)
            home = Path.home().resolve(strict=True)
            prefixes = tuple(
                dict.fromkeys(
                    Path(value).resolve(strict=True)
                    for value in (
                        sys.prefix,
                        sys.base_prefix,
                        sys.exec_prefix,
                        sys.base_exec_prefix,
                    )
                    if value
                )
            )
            configured = sysconfig.get_paths()
            stdlib_roots = tuple(
                dict.fromkeys(
                    Path(configured[key]).resolve(strict=True)
                    for key in ("stdlib", "platstdlib")
                    if configured.get(key)
                )
            )
            site_values = [*site.getsitepackages()]
            user_site = site.getusersitepackages()
            if isinstance(user_site, str):
                site_values.append(user_site)
            else:
                site_values.extend(user_site)
            site_roots = tuple(
                dict.fromkeys(
                    resolved
                    for value in site_values
                    for resolved in (Path(value).resolve(strict=False),)
                    if any(
                        part.casefold() in {"site-packages", "dist-packages"}
                        for part in resolved.parts
                    )
                )
            )
        except (OSError, TypeError, ValueError) as exc:
            raise ReleaseValidationError(
                "trusted import roots cannot be resolved"
            ) from exc

        workspace_code = (self.root / "code").resolve(strict=False)
        stdlib_entries: list[tuple[str, Path]] = []
        site_entries: list[tuple[str, Path]] = []
        seen: set[str] = set()
        for entry in entries:
            candidate = self._resolved_import_entry(entry)
            if candidate is None:
                continue
            key = os.path.normcase(os.fspath(candidate))
            if key in seen:
                continue
            seen.add(key)
            if (
                candidate == cwd
                or candidate == self.root
                or candidate == workspace_code
                or self._is_project_code_path(entry)
            ):
                continue
            canonical_site = any(
                self._is_relative_to(candidate, root) for root in site_roots
            )
            if canonical_site:
                if any(
                    self._is_relative_to(candidate, prefix) for prefix in prefixes
                ) or self._is_relative_to(candidate, home):
                    site_entries.append((entry, candidate))
                continue
            canonical_archive = (
                candidate.suffix.casefold() == ".zip"
                and candidate.name.casefold().startswith("python")
                and any(self._is_relative_to(candidate, prefix) for prefix in prefixes)
            )
            canonical_stdlib = (
                candidate in prefixes
                or any(self._is_relative_to(candidate, root) for root in stdlib_roots)
                or (
                    candidate.name.casefold() == "dlls" and candidate.parent in prefixes
                )
                or canonical_archive
            )
            if canonical_stdlib:
                stdlib_entries.append((entry, candidate))

        if not stdlib_entries:
            _fail("trusted import session found no interpreter standard-library path")
        sanitized = [
            os.fspath(path) for _entry, path in (*stdlib_entries, *site_entries)
        ]
        return (
            sanitized,
            tuple(path for _entry, path in stdlib_entries),
            tuple(path for _entry, path in site_entries),
        )

    def _remove_dependency_root(self, root: str) -> None:
        prefix = f"{root}."
        for name in tuple(sys.modules):
            if name == root or name.startswith(prefix):
                sys.modules.pop(name, None)

    def _prepare_stdlib_root(self, root: str) -> None:
        if root in self._prepared_stdlib_roots:
            return
        if root == "sys":
            if sys.modules.get("sys") is not sys:
                _fail("trusted import session found a substituted sys module")
            return
        if root == "builtins":
            if sys.modules.get("builtins") is not builtins:
                _fail("trusted import session found a substituted builtins module")
            return
        if root == "importlib" and sys.modules.get("importlib") is not importlib:
            _fail("trusted import session found a substituted importlib module")
        prefix = f"{root}."
        existing = {
            name: module
            for name, module in tuple(sys.modules.items())
            if (name == root or name.startswith(prefix))
        }
        self._saved_dependency_modules.update(existing)
        for name, module in existing.items():
            if _BASELINE_STDLIB_MODULES.get(name) is module:
                if isinstance(module, ModuleType) and not self._module_origin_is_within(
                    module,
                    self._stdlib_search_paths,
                    reject_roots=self._site_search_paths,
                    allow_builtin_or_frozen=True,
                ):
                    _fail(
                        "baseline standard-library module has an untrusted "
                        f"origin: {name}"
                    )
                continue
            sys.modules.pop(name, None)
        self._prepared_stdlib_roots.add(root)

    def _module_origin_is_within(
        self,
        module: ModuleType,
        roots: Sequence[Path],
        *,
        reject_roots: Sequence[Path] = (),
        allow_builtin_or_frozen: bool = False,
    ) -> bool:
        spec = getattr(module, "__spec__", None)
        origin = getattr(spec, "origin", None)
        if origin in {"built-in", "frozen"}:
            return allow_builtin_or_frozen
        locations = getattr(spec, "submodule_search_locations", None)
        raw_candidates: list[Any] = []
        if isinstance(origin, (str, os.PathLike)):
            raw_candidates.append(origin)
        reported = getattr(module, "__file__", None)
        if isinstance(reported, (str, os.PathLike)):
            raw_candidates.append(reported)
        if locations is not None:
            raw_candidates.extend(locations)
        if not raw_candidates:
            return False
        for raw in raw_candidates:
            try:
                candidate = Path(raw).resolve(strict=False)
            except (OSError, TypeError, ValueError):
                return False
            if any(self._is_relative_to(candidate, denied) for denied in reject_roots):
                return False
            if not any(self._is_relative_to(candidate, root) for root in roots):
                return False
        return True

    @staticmethod
    def _dependency_closure(root: str) -> dict[str, Any]:
        prefix = f"{root}."
        return {
            name: module
            for name, module in tuple(sys.modules.items())
            if name == root or name.startswith(prefix)
        }

    @staticmethod
    def _requested_dependency_bindings(
        name: str,
        fromlist: Sequence[str] | None,
    ) -> list[tuple[str, str]]:
        parts = name.split(".")
        requested = [
            (".".join(parts[:index]), parts[index]) for index in range(1, len(parts))
        ]
        requested.extend((name, item) for item in fromlist or ())
        return requested

    def _snapshot_dependency_root(self, root: str) -> None:
        if root not in self._dependency_root_snapshots:
            self._dependency_root_snapshots[root] = self._dependency_closure(root)

    def _snapshot_dependency_bindings(
        self,
        name: str,
        fromlist: Sequence[str] | None,
    ) -> None:
        for package_name, item in self._requested_dependency_bindings(name, fromlist):
            if not isinstance(item, str) or not item:
                continue
            package = sys.modules.get(package_name)
            if not isinstance(package, ModuleType):
                continue
            if type(package) is not ModuleType:
                _fail(
                    "dependency import cannot bind a module subclass: "
                    f"{package_name}"
                )
            if item == "*":
                items = self._stable_star_export_names(package_name, package)
            else:
                items = (item,)
            for requested_item in items:
                binding_key = (package_name, requested_item)
                if binding_key in self._dependency_binding_snapshots:
                    continue
                self._dependency_binding_snapshots[binding_key] = vars(package).get(
                    requested_item,
                    _MODULE_NOT_CAPTURED,
                )

    @staticmethod
    def _module_reports_origin(module: ModuleType) -> bool:
        spec = getattr(module, "__spec__", None)
        if getattr(spec, "origin", None) is not None:
            return True
        if getattr(module, "__file__", None) is not None:
            return True
        return getattr(spec, "submodule_search_locations", None) is not None

    @staticmethod
    def _module_origin_state(module: ModuleType) -> tuple[Any, ...]:
        spec = getattr(module, "__spec__", None)
        locations = getattr(spec, "submodule_search_locations", None)
        try:
            location_state = tuple(os.fspath(item) for item in locations or ())
        except (TypeError, ValueError):
            location_state = (object(),)

        def reported_path(value: Any) -> Any:
            if not isinstance(value, (str, os.PathLike)):
                return value
            try:
                return os.fspath(value)
            except (TypeError, ValueError):
                return object()

        return (
            reported_path(getattr(spec, "origin", None)),
            reported_path(getattr(module, "__file__", None)),
            location_state,
        )

    def _dependency_object_is_registered(self, name: str, module: Any) -> bool:
        return any(
            registry.get(name, _MODULE_NOT_CAPTURED) is module
            for registry in (
                _BASELINE_MODULES,
                _TRUSTED_DEPENDENCY_MODULES,
                self._session_dependency_modules,
            )
        )

    @staticmethod
    def _stable_star_export_names(
        package_name: str,
        package: ModuleType,
    ) -> tuple[str, ...]:
        if type(package) is not ModuleType:
            _fail("dependency import cannot bind a module subclass: " f"{package_name}")
        namespace = vars(package)
        declared = namespace.get("__all__", _MODULE_NOT_CAPTURED)
        if declared is _MODULE_NOT_CAPTURED:
            if "__getattr__" in namespace:
                _fail(
                    "dependency star import cannot resolve dynamic exports: "
                    f"{package_name}"
                )
            if any(type(name) is not str for name in namespace):
                _fail(
                    "dependency star import has an unstable module dictionary: "
                    f"{package_name}"
                )
            names = tuple(
                sorted(name for name in namespace if not name.startswith("_"))
            )
        else:
            if type(declared) not in (list, tuple) or any(
                type(item) is not str or not item for item in declared
            ):
                _fail(
                    "dependency star import has an unstable __all__: " f"{package_name}"
                )
            names = tuple(declared)
        for item in names:
            if item not in namespace:
                _fail(
                    "dependency star import does not expose a stable package binding: "
                    f"{package_name}.{item}"
                )
        return names

    @classmethod
    def _validate_registered_dependency_state(cls, root: str) -> None:
        prefix = f"{root}."
        for name, registered in _TRUSTED_DEPENDENCY_MODULES.items():
            if not (name == root or name.startswith(prefix)):
                continue
            if sys.modules.get(name, _MODULE_NOT_CAPTURED) is not registered:
                _fail(f"registered dependency module changed: {name}")
        for (package_name, item), registered in _TRUSTED_DEPENDENCY_BINDINGS.items():
            if not (package_name == root or package_name.startswith(prefix)):
                continue
            package = sys.modules.get(package_name)
            if not isinstance(package, ModuleType):
                _fail(f"registered dependency package changed: {package_name}")
            if type(package) is not ModuleType:
                _fail(
                    "registered dependency package is a module subclass: "
                    f"{package_name}"
                )
            if vars(package).get(item, _MODULE_NOT_CAPTURED) is not registered:
                _fail(f"dependency package binding changed: {package_name}.{item}")
        for package_name, registered in _TRUSTED_DEPENDENCY_STAR_EXPORTS.items():
            if not (package_name == root or package_name.startswith(prefix)):
                continue
            package = sys.modules.get(package_name)
            if not isinstance(package, ModuleType):
                _fail(f"registered dependency package changed: {package_name}")
            if cls._stable_star_export_names(package_name, package) != registered:
                _fail(f"dependency star exports changed: {package_name}")

    def _validate_dependency_closure(
        self,
        root: str,
        *,
        allow_new: bool,
    ) -> None:
        self._validate_registered_dependency_state(root)
        closure = self._dependency_closure(root)
        module = closure.get(root, _MODULE_NOT_CAPTURED)
        if not isinstance(module, ModuleType):
            _fail(f"trusted import did not produce a module object: {root}")
        allowed = (*self._stdlib_search_paths, *self._site_search_paths)
        for name, loaded in closure.items():
            for registry in (_BASELINE_MODULES, _TRUSTED_DEPENDENCY_MODULES):
                expected = registry.get(name, _MODULE_NOT_CAPTURED)
                if expected is not _MODULE_NOT_CAPTURED and loaded is not expected:
                    _fail(f"registered dependency module changed: {name}")
            registered = self._dependency_object_is_registered(name, loaded)
            if not registered and not allow_new:
                _fail(f"unregistered preloaded dependency module: {name}")
            if isinstance(loaded, ModuleType):
                reports_origin = self._module_reports_origin(loaded)
                if name == root or reports_origin:
                    origin_state = self._module_origin_state(loaded)
                    cached = self._validated_dependency_modules.get(name)
                    if not (
                        cached is not None
                        and cached[0] is loaded
                        and cached[1] == origin_state
                    ):
                        if not self._module_origin_is_within(loaded, allowed):
                            _fail(
                                f"imported dependency has an untrusted origin: {name}"
                            )
                        self._validated_dependency_modules[name] = (
                            loaded,
                            origin_state,
                        )
            elif name == root:
                _fail(f"trusted import did not produce a module object: {root}")
            if not registered:
                self._session_dependency_modules[name] = loaded

    def _validate_dependency_bindings(
        self,
        name: str,
        fromlist: Sequence[str] | None,
    ) -> None:
        explicit_bindings = {
            (name, item)
            for item in fromlist or ()
            if isinstance(item, str) and item and item != "*"
        }
        requested_bindings: list[tuple[str, str]] = []
        for package_name, item in self._requested_dependency_bindings(name, fromlist):
            if not isinstance(item, str) or not item:
                continue
            if item == "*":
                package = sys.modules.get(package_name)
                if not isinstance(package, ModuleType):
                    _fail(
                        "dependency star import did not produce a module object: "
                        f"{package_name}"
                    )
                exports = self._stable_star_export_names(package_name, package)
                registered_exports = _TRUSTED_DEPENDENCY_STAR_EXPORTS.get(
                    package_name,
                    _MODULE_NOT_CAPTURED,
                )
                if (
                    registered_exports is not _MODULE_NOT_CAPTURED
                    and exports != registered_exports
                ):
                    _fail(f"dependency star exports changed: {package_name}")
                self._session_dependency_star_exports[package_name] = exports
                requested_bindings.extend((package_name, export) for export in exports)
                continue
            requested_bindings.append((package_name, item))
        seen: set[tuple[str, str]] = set()
        for package_name, item in requested_bindings:
            binding_key = (package_name, item)
            if binding_key in seen:
                continue
            seen.add(binding_key)
            package = sys.modules.get(package_name)
            if not isinstance(package, ModuleType):
                continue
            if type(package) is not ModuleType:
                _fail(
                    "dependency import cannot bind a module subclass: "
                    f"{package_name}"
                )
            child_name = f"{package_name}.{item}"
            child = sys.modules.get(child_name, _MODULE_NOT_CAPTURED)
            registered = next(
                (
                    candidate
                    for registry in (
                        self._session_dependency_modules,
                        _TRUSTED_DEPENDENCY_MODULES,
                        _BASELINE_MODULES,
                    )
                    for candidate in (registry.get(child_name, _MODULE_NOT_CAPTURED),)
                    if candidate is not _MODULE_NOT_CAPTURED
                ),
                _MODULE_NOT_CAPTURED,
            )
            attribute = vars(package).get(item, _MODULE_NOT_CAPTURED)
            registered_binding = _TRUSTED_DEPENDENCY_BINDINGS.get(
                binding_key,
                _MODULE_NOT_CAPTURED,
            )
            if (
                attribute is _MODULE_NOT_CAPTURED
                and binding_key in explicit_bindings
                and "__getattr__" in vars(package)
            ):
                _fail(
                    "dependency import does not expose a stable package binding: "
                    f"{child_name}"
                )
            if (
                registered_binding is not _MODULE_NOT_CAPTURED
                and attribute is not registered_binding
            ):
                _fail(f"dependency package binding changed: {child_name}")
            if child is _MODULE_NOT_CAPTURED:
                if registered is not _MODULE_NOT_CAPTURED:
                    _fail(f"registered dependency module changed: {child_name}")
            if (
                child is not _MODULE_NOT_CAPTURED
                and registered is not _MODULE_NOT_CAPTURED
                and child is not registered
            ):
                _fail(f"registered dependency module changed: {child_name}")
            if attribute is not _MODULE_NOT_CAPTURED:
                self._session_dependency_bindings[binding_key] = attribute

    def _validate_imported_root(
        self,
        root: str,
        *,
        allow_new_dependency: bool = False,
    ) -> None:
        if root in TRUSTED_WORKSPACE_MODULE_NAMES:
            return
        module = sys.modules.get(root)
        if not isinstance(module, ModuleType):
            _fail(f"trusted import did not produce a module object: {root}")
        if root in getattr(sys, "stdlib_module_names", ()):
            prefix = f"{root}."
            for name, loaded in tuple(sys.modules.items()):
                if not (name == root or name.startswith(prefix)):
                    continue
                if not isinstance(loaded, ModuleType):
                    if _BASELINE_STDLIB_MODULES.get(name) is loaded:
                        continue
                    _fail(
                        "standard-library module slot is not a trusted module: "
                        f"{name}"
                    )
                if not self._module_origin_is_within(
                    loaded,
                    self._stdlib_search_paths,
                    reject_roots=self._site_search_paths,
                    allow_builtin_or_frozen=True,
                ):
                    _fail(f"standard-library module has an untrusted origin: {name}")
            return
        self._validate_dependency_closure(
            root,
            allow_new=allow_new_dependency,
        )

    @staticmethod
    def _resolve_absolute_import_name(
        name: str,
        globals: Mapping[str, Any] | None,
        level: int,
    ) -> str:
        if level == 0:
            return name
        if not isinstance(globals, Mapping):
            _fail("relative import has no trusted package context")
        package = globals.get("__package__")
        spec = globals.get("__spec__")
        if package is None and spec is not None:
            package = getattr(spec, "parent", None)
        if package is None:
            package = globals.get("__name__")
            if isinstance(package, str) and "__path__" not in globals:
                package = package.rpartition(".")[0]
        if not isinstance(package, str) or not package:
            _fail("relative import has no trusted package context")
        try:
            return importlib.util.resolve_name(f"{'.' * level}{name}", package)
        except (ImportError, TypeError, ValueError) as exc:
            raise ReleaseValidationError(
                "relative import cannot be resolved inside the trusted session"
            ) from exc

    def _run_guarded_import(
        self,
        absolute_name: str,
        fromlist: Sequence[str] | None,
        importer: Any,
    ) -> Any:
        if not absolute_name:
            _fail("trusted import resolved to an empty module name")
        root = absolute_name.partition(".")[0]
        started_dependency_import = False
        if root in getattr(sys, "stdlib_module_names", ()):
            self._prepare_stdlib_root(root)
        elif root not in TRUSTED_WORKSPACE_MODULE_NAMES:
            self._snapshot_dependency_root(root)
            closure = self._dependency_closure(root)
            if closure:
                self._validate_imported_root(
                    root,
                    allow_new_dependency=root in self._active_dependency_roots,
                )
                self._snapshot_dependency_bindings(absolute_name, fromlist)
            if root not in self._active_dependency_roots:
                self._active_dependency_roots.add(root)
                started_dependency_import = True
        try:
            self._guard_depth += 1
            try:
                imported = importer()
            finally:
                self._guard_depth -= 1
            self._validate_imported_root(
                root,
                allow_new_dependency=root in self._active_dependency_roots,
            )
            if (
                root not in getattr(sys, "stdlib_module_names", ())
                and root not in TRUSTED_WORKSPACE_MODULE_NAMES
            ):
                self._validate_dependency_bindings(absolute_name, fromlist)
            return imported
        finally:
            if started_dependency_import:
                self._active_dependency_roots.discard(root)

    def _guarded_import(
        self,
        name: str,
        globals: Mapping[str, Any] | None = None,
        locals: Mapping[str, Any] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> Any:
        active_session = _ACTIVE_TRUSTED_MODULE_SESSION
        if active_session is not None and active_session is not self:
            return active_session._guarded_import(
                name,
                globals,
                locals,
                fromlist,
                level,
            )
        guarded = self._active and threading.get_ident() == self._owner_thread_id
        if not guarded:
            return _BASELINE_BUILTIN_IMPORT(name, globals, locals, fromlist, level)
        if not isinstance(name, str) or not isinstance(level, int) or level < 0:
            return _BASELINE_BUILTIN_IMPORT(name, globals, locals, fromlist, level)
        absolute_name = self._resolve_absolute_import_name(name, globals, level)
        return self._run_guarded_import(
            absolute_name,
            fromlist,
            lambda: _BASELINE_BUILTIN_IMPORT(
                name,
                globals,
                locals,
                fromlist,
                level,
            ),
        )

    def _guarded_import_module(
        self,
        name: str,
        package: str | None = None,
    ) -> Any:
        active_session = _ACTIVE_TRUSTED_MODULE_SESSION
        if active_session is not None and active_session is not self:
            return active_session._guarded_import_module(name, package)
        guarded = self._active and threading.get_ident() == self._owner_thread_id
        if not guarded:
            return _BASELINE_IMPORT_MODULE(name, package)
        try:
            absolute_name = importlib.util.resolve_name(name, package)
        except (ImportError, TypeError, ValueError) as exc:
            raise ReleaseValidationError(
                "dynamic import cannot be resolved inside the trusted session"
            ) from exc
        return self._run_guarded_import(
            absolute_name,
            (),
            lambda: _BASELINE_IMPORT_MODULE(name, package),
        )

    def __enter__(self) -> _TrustedWorkspaceModuleSession:
        global _ACTIVE_TRUSTED_MODULE_SESSION

        if self._active or self._lock_held:
            _fail("trusted first-party module session cannot be re-entered")
        state_captured = False
        try:
            try:
                self._lock_held = self._IMPORT_LOCK.acquire()
            except BaseException:
                # acquire() can be interrupted after the underlying RLock has
                # changed ownership but before its return value reaches Python.
                try:
                    self._IMPORT_LOCK.release()
                except RuntimeError:
                    pass
                self._lock_held = False
                raise
            saved_modules = {
                name: sys.modules[name]
                for name in TRUSTED_WORKSPACE_MODULE_NAMES
                if name in sys.modules
            }
            saved_sys_path = list(sys.path)
            saved_meta_path = list(sys.meta_path)
            saved_path_hooks = list(sys.path_hooks)
            saved_path_importer_cache = dict(sys.path_importer_cache)
            saved_builtin_import = builtins.__import__
            saved_import_module = importlib.import_module
            self._saved_modules = saved_modules
            self._saved_sys_path = saved_sys_path
            self._saved_meta_path = saved_meta_path
            self._saved_path_hooks = saved_path_hooks
            self._saved_path_importer_cache = saved_path_importer_cache
            self._saved_builtin_import = saved_builtin_import
            self._saved_import_module = saved_import_module
            state_captured = True
            self._sources.clear()
            self._loaded.clear()
            self._saved_dependency_modules.clear()
            self._session_dependency_modules.clear()
            self._session_dependency_bindings.clear()
            self._session_dependency_star_exports.clear()
            self._dependency_root_snapshots.clear()
            self._dependency_binding_snapshots.clear()
            self._validated_dependency_modules.clear()
            self._active_dependency_roots.clear()
            self._prepared_stdlib_roots.clear()
            self._guard_depth = 0
            self._owner_thread_id = None
            if saved_builtin_import is not _BASELINE_BUILTIN_IMPORT:
                _fail("trusted import session refuses an ambient builtins import hook")
            if saved_import_module is not _BASELINE_IMPORT_MODULE:
                _fail(
                    "trusted import session refuses an ambient importlib import_module hook"
                )
            self._owner_thread_id = threading.get_ident()
            for name in TRUSTED_WORKSPACE_MODULE_NAMES:
                sys.modules.pop(name, None)
            (
                sanitized_path,
                self._stdlib_search_paths,
                self._site_search_paths,
            ) = self._trusted_search_path(saved_sys_path)
            sys.path[:] = sanitized_path
            sys.meta_path[:] = [
                self,
                importlib.machinery.BuiltinImporter,
                importlib.machinery.FrozenImporter,
                importlib.machinery.PathFinder,
            ]
            sys.path_hooks[:] = _BASELINE_PATH_HOOKS
            sys.path_importer_cache.clear()
            builtins.__import__ = self._guarded_import
            importlib.import_module = self._guarded_import_module
            self._active = True
            _ACTIVE_TRUSTED_MODULE_SESSION = self
            return self
        except BaseException:
            try:
                if state_captured:
                    self._restore_import_state(rollback_dependencies=True)
            finally:
                self._release_import_lock()
            raise

    def _restore_import_state(self, *, rollback_dependencies: bool) -> None:
        global _ACTIVE_TRUSTED_MODULE_SESSION

        self._active = False
        if _ACTIVE_TRUSTED_MODULE_SESSION is self:
            _ACTIVE_TRUSTED_MODULE_SESSION = None
        builtins.__import__ = self._saved_builtin_import
        importlib.import_module = self._saved_import_module
        sys.meta_path[:] = self._saved_meta_path
        sys.path[:] = self._saved_sys_path
        sys.path_hooks[:] = self._saved_path_hooks
        sys.path_importer_cache.clear()
        sys.path_importer_cache.update(self._saved_path_importer_cache)
        for name in TRUSTED_WORKSPACE_MODULE_NAMES:
            sys.modules.pop(name, None)
        for root in self._prepared_stdlib_roots:
            self._remove_dependency_root(root)
        sys.modules.update(self._saved_dependency_modules)
        if rollback_dependencies:
            for root, snapshot in self._dependency_root_snapshots.items():
                self._remove_dependency_root(root)
                sys.modules.update(snapshot)
            for (
                package_name,
                item,
            ), previous in self._dependency_binding_snapshots.items():
                package = sys.modules.get(package_name)
                if not isinstance(package, ModuleType):
                    continue
                if type(package) is not ModuleType:
                    continue
                if previous is _MODULE_NOT_CAPTURED:
                    vars(package).pop(item, None)
                else:
                    vars(package)[item] = previous
        sys.modules.update(self._saved_modules)
        self._guard_depth = 0
        self._validated_dependency_modules.clear()
        self._session_dependency_bindings.clear()
        self._session_dependency_star_exports.clear()
        self._dependency_root_snapshots.clear()
        self._dependency_binding_snapshots.clear()
        self._active_dependency_roots.clear()
        self._owner_thread_id = None

    def _commit_dependency_modules(self) -> None:
        allowed = (*self._stdlib_search_paths, *self._site_search_paths)
        for name, module in self._session_dependency_modules.items():
            current = sys.modules.get(name, _MODULE_NOT_CAPTURED)
            if current is _MODULE_NOT_CAPTURED:
                continue
            if current is not module:
                _fail(f"dependency module changed before session exit: {name}")
            if (
                isinstance(module, ModuleType)
                and self._module_reports_origin(module)
                and not self._module_origin_is_within(module, allowed)
            ):
                _fail(f"dependency module origin changed before session exit: {name}")
        committed_bindings: dict[tuple[str, str], Any] = {}
        for binding_key in self._session_dependency_bindings:
            package_name, item = binding_key
            package = sys.modules.get(package_name)
            if not isinstance(package, ModuleType):
                _fail(f"dependency package changed before session exit: {package_name}")
            if type(package) is not ModuleType:
                _fail(
                    "dependency package became a module subclass before session exit: "
                    f"{package_name}"
                )
            current = vars(package).get(item, _MODULE_NOT_CAPTURED)
            registered = _TRUSTED_DEPENDENCY_BINDINGS.get(
                binding_key,
                _MODULE_NOT_CAPTURED,
            )
            if current is _MODULE_NOT_CAPTURED:
                if registered is not _MODULE_NOT_CAPTURED:
                    _fail(
                        "dependency package binding disappeared before session exit: "
                        f"{package_name}.{item}"
                    )
                continue
            if registered is not _MODULE_NOT_CAPTURED and current is not registered:
                _fail(f"dependency package binding changed: {package_name}.{item}")
            committed_bindings[binding_key] = current
        for package_name, exports in self._session_dependency_star_exports.items():
            package = sys.modules.get(package_name)
            if not isinstance(package, ModuleType):
                _fail(f"dependency package changed before session exit: {package_name}")
            if self._stable_star_export_names(package_name, package) != exports:
                _fail(f"dependency star exports changed: {package_name}")
        _TRUSTED_DEPENDENCY_MODULES.update(
            {
                name: module
                for name, module in self._session_dependency_modules.items()
                if sys.modules.get(name, _MODULE_NOT_CAPTURED) is module
            }
        )
        _TRUSTED_DEPENDENCY_BINDINGS.update(committed_bindings)
        _TRUSTED_DEPENDENCY_STAR_EXPORTS.update(self._session_dependency_star_exports)

    def _release_import_lock(self) -> None:
        if self._lock_held:
            self._lock_held = False
            self._IMPORT_LOCK.release()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> Literal[False]:
        validation_error: BaseException | None = None
        try:
            for name, module in self._loaded.items():
                self._validate_loaded_module(name, module)
            if exc_type is None:
                self._commit_dependency_modules()
        except BaseException as caught:
            validation_error = caught
        finally:
            try:
                self._restore_import_state(
                    rollback_dependencies=(
                        exc_type is not None or validation_error is not None
                    )
                )
            finally:
                self._release_import_lock()
        if validation_error is not None:
            raise validation_error
        return False

    def _source_for(self, name: str) -> TrustedModuleSource:
        if name not in TRUSTED_WORKSPACE_MODULE_NAMES:
            _fail(f"unregistered first-party verifier module: {name!r}")
        cached = self._sources.get(name)
        if cached is not None:
            return cached
        relative = TRUSTED_WORKSPACE_MODULE_PATHS[name]
        if relative not in REQUIRED_SOURCE_FILES:
            _fail(
                f"first-party verifier module is not in the reviewed inventory: {name}"
            )
        path = _safe_workspace_file(
            self.root,
            relative,
            field=f"trusted first-party module {name}",
        )
        expected_parent = (self.root / PurePosixPath(relative).parent).resolve(
            strict=True
        )
        if path.parent.resolve(strict=True) != expected_parent:
            _fail(f"trusted first-party module has an unexpected parent: {name}")
        payload, fingerprint = _read_stable_bytes(
            path,
            label=f"trusted first-party module source {name}",
        )
        expected = self._expected_sources.get(name)
        if self._expected_sources and expected is None:
            _fail(f"trusted first-party module was not selected: {name}")
        if expected is not None:
            expected_path = expected.source.resolve(strict=True)
            if expected.relative != relative or expected_path != path.resolve(
                strict=True
            ):
                _fail(f"trusted first-party module selection is invalid: {name}")
            _assert_fingerprint_matches(
                expected,
                fingerprint,
                label=f"trusted first-party module source {name}",
            )
            if payload != expected.payload:
                _fail(f"trusted first-party module bytes differ from selection: {name}")
            payload = expected.payload
        source = TrustedModuleSource(
            name=name,
            path=path.resolve(strict=True),
            payload=payload,
            fingerprint=fingerprint,
        )
        self._sources[name] = source
        return source

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: Any | None = None,
    ) -> Any | None:
        del path, target
        if fullname not in TRUSTED_WORKSPACE_MODULE_NAMES:
            return None
        source = self._source_for(fullname)
        spec = importlib.util.spec_from_loader(
            fullname,
            self,
            origin=str(source.path),
        )
        if spec is None:
            _fail(f"could not create a trusted module specification: {fullname}")
        spec.has_location = True
        return spec

    def exec_module(self, module: ModuleType) -> None:
        name = getattr(module, "__name__", None)
        if not isinstance(name, str):
            _fail("trusted first-party module has no valid name")
        source = self._source_for(name)
        module.__file__ = str(source.path)
        module.__package__ = ""
        compiled = compile(
            source.payload,
            str(source.path),
            "exec",
            dont_inherit=True,
        )
        exec(compiled, module.__dict__)
        if sys.modules.get(name) is not module:
            _fail(
                f"trusted first-party module replaced itself during execution: {name}"
            )
        setattr(
            module,
            "__trusted_workspace_source_sha256__",
            source.fingerprint.sha256,
        )
        self._loaded[name] = module
        self._validate_loaded_module(name, module, rehash_source=False)

    def _validate_loaded_module(
        self,
        name: str,
        module: ModuleType,
        *,
        rehash_source: bool = True,
    ) -> None:
        source = self._source_for(name)
        if self._loaded.get(name) is not module or sys.modules.get(name) is not module:
            _fail(f"trusted first-party module identity changed: {name}")
        reported = getattr(module, "__file__", None)
        if not isinstance(reported, (str, os.PathLike)):
            _fail(f"trusted first-party module has no valid origin: {name}")
        try:
            origin = Path(reported).resolve(strict=True)
        except (OSError, TypeError, ValueError) as exc:
            raise ReleaseValidationError(
                f"trusted first-party module origin cannot be resolved: {name}"
            ) from exc
        if origin != source.path:
            _fail(
                f"trusted first-party module origin differs from selected workspace: {name}"
            )
        if (
            getattr(module, "__trusted_workspace_source_sha256__", None)
            != source.fingerprint.sha256
        ):
            _fail(f"trusted first-party module source binding changed: {name}")
        if rehash_source:
            payload, fingerprint = _read_stable_bytes(
                source.path,
                label=f"trusted first-party module source {name}",
            )
            if payload != source.payload or fingerprint != source.fingerprint:
                _fail(
                    f"trusted first-party module source changed after execution: {name}"
                )
            expected = self._expected_sources.get(name)
            if expected is not None:
                _assert_fingerprint_matches(
                    expected,
                    fingerprint,
                    label=f"trusted first-party module source {name}",
                )
                if payload != expected.payload:
                    _fail(
                        f"trusted first-party module bytes differ from selection: {name}"
                    )

    def load(self, name: str) -> ModuleType:
        if not self._active:
            _fail("trusted first-party module session is not active")
        module = importlib.import_module(name)
        if self._loaded.get(name) is not module:
            _fail(f"ambient code was selected instead of trusted module bytes: {name}")
        self._validate_loaded_module(name, module)
        return module


DirectoryIdentity = tuple[int, int, int]
PathComponentIdentities = tuple[tuple[int, int, int], ...]


class _NativeDescriptorResult(ctypes.c_int):
    """A ctypes return value that owns a successful native descriptor."""

    def close_quietly(self) -> None:
        descriptor = int(self.value)
        if descriptor < 0:
            return
        # Retire the numeric value before the close boundary. An asynchronous
        # exception after a successful close must never leave a reusable fd
        # reachable by this independently finalizable result object.
        self.value = -1
        try:
            os.close(descriptor)
        except BaseException:
            pass

    def __del__(self) -> None:
        self.close_quietly()


def _close_native_windows_handle_quietly(raw_value: int) -> None:
    if os.name != "nt" or raw_value == ctypes.c_void_p(-1).value:
        return
    try:
        from ctypes import wintypes

        kernel32 = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL
        close_handle(raw_value)
    except BaseException:
        pass


class _NativeWindowsHandleResult(ctypes.c_void_p):
    """A ctypes return/output cell that owns a successful Win32 handle."""

    def close_quietly(self) -> None:
        raw_value = self.value
        if raw_value is None or raw_value == ctypes.c_void_p(-1).value:
            return
        # As above, make the result permanently non-owning before entering the
        # one native close attempt; an ambiguous interruption is never retried.
        self.value = None
        _close_native_windows_handle_quietly(int(raw_value))

    def __del__(self) -> None:
        self.close_quietly()


class _WindowsTransferredDescriptorResult(_NativeDescriptorResult):
    """A CRT descriptor produced while a Win32-handle transfer is pending."""

    _source_transfer_reference = None

    def _source_transfer(self) -> _WindowsDescriptorTransfer | None:
        reference = type(self)._source_transfer_reference
        return None if reference is None else reference()

    def __del__(self) -> None:
        transfer = self._source_transfer()
        if transfer is None:
            self.close_quietly()
            return
        transfer.abandon_result(self)


class _ResourceOwnershipToken:
    """Own one native resource across interruptible Python return boundaries."""

    __slots__ = ("_state",)

    def __init__(self, state: tuple[str, Any, Any]) -> None:
        self._state = state

    @classmethod
    def for_descriptor(
        cls, descriptor: int | _NativeDescriptorResult
    ) -> _ResourceOwnershipToken:
        cell = (
            descriptor
            if isinstance(descriptor, _NativeDescriptorResult)
            else _NativeDescriptorResult(descriptor)
        )
        return cls(("descriptor", cell, None))

    @classmethod
    def for_windows_handle(
        cls,
        raw_handle: Any,
        close_handle: Any,
    ) -> _ResourceOwnershipToken:
        return cls(("windows_handle", raw_handle, close_handle))

    @property
    def descriptor(self) -> int:
        kind, value, _metadata = self._state
        if kind != "descriptor":
            return -1
        return int(getattr(value, "value", value))

    @property
    def windows_handle(self) -> int | None:
        kind, value, _close_handle = self._state
        if kind != "windows_handle":
            return None
        raw_value = getattr(value, "value", value)
        return int(raw_value) if raw_value is not None else None

    def adopt_descriptor(self, descriptor: int | _NativeDescriptorResult) -> None:
        """Atomically replace a transferred native handle with its CRT descriptor."""
        kind, value, _metadata = self._state
        if kind == "windows_handle" and isinstance(value, _NativeWindowsHandleResult):
            # _open_osfhandle transferred this exact handle into the descriptor.
            # Disarm the old view before publishing the descriptor owner.
            value.value = None
        cell = (
            descriptor
            if isinstance(descriptor, _NativeDescriptorResult)
            else _NativeDescriptorResult(descriptor)
        )
        self._state = ("descriptor", cell, None)

    def bind_descriptor_identity(self, identity: DirectoryIdentity) -> None:
        kind, value, _metadata = self._state
        if kind == "descriptor":
            self._state = (kind, value, identity)

    def relinquish_descriptor(self) -> int:
        descriptor = self.descriptor
        if descriptor >= 0:
            kind, value, _metadata = self._state
            if kind == "descriptor" and isinstance(value, _NativeDescriptorResult):
                value.value = -1
            self._state = ("closed", None, None)
        return descriptor

    def close(self) -> None:
        kind, value, metadata = self._state
        if kind == "closed":
            return
        if kind == "windows_handle":
            raw_value = getattr(value, "value", value)
            if raw_value is None:
                self._state = ("closed", None, None)
                return
            # A Win32 HANDLE value has no generation identity.  Disarm the
            # independently finalizable result cell and publish retirement before
            # making the one permitted CloseHandle attempt; an asynchronous
            # exception at the native return boundary must never close a reused
            # numeric handle during later cleanup.
            if isinstance(value, _NativeWindowsHandleResult):
                value.value = None
            self._state = ("closed", None, None)
            result = metadata(raw_value)
            if not result:
                raise ctypes.WinError(ctypes.get_last_error())
            return

        descriptor = int(getattr(value, "value", value))
        # A descriptor number has no generation identity.  Once close starts, an
        # exception cannot distinguish an unclosed descriptor from a completed
        # close followed by immediate reuse, even when fstat reports the same file.
        # Publish retirement first and make exactly one close attempt.  This may
        # leak the original descriptor when close did not run, but it cannot close
        # a replacement owned by unrelated code.
        if isinstance(value, _NativeDescriptorResult):
            value.value = -1
        self._state = ("closed", None, None)
        os.close(descriptor)

    def close_quietly(self) -> None:
        for _ in range(2):
            if self._state[0] == "closed":
                return
            try:
                self.close()
            except BaseException:
                continue
            return

    def __del__(self) -> None:
        self.close_quietly()


class _OwnedDescriptor:
    """A descriptor view whose shared token closes after the last owner is lost."""

    __slots__ = ("_token",)

    def __init__(self, descriptor: int | _ResourceOwnershipToken) -> None:
        self._token = (
            descriptor
            if isinstance(descriptor, _ResourceOwnershipToken)
            else _ResourceOwnershipToken.for_descriptor(descriptor)
        )

    @property
    def descriptor(self) -> int:
        return self._token.descriptor

    def bind_identity(self, identity: DirectoryIdentity) -> None:
        self._token.bind_descriptor_identity(identity)

    def close(self) -> None:
        self._token.close()

    def close_quietly(self) -> None:
        self._token.close_quietly()

    def relinquish(self) -> int:
        return self._token.relinquish_descriptor()

    def __index__(self) -> int:
        return self.descriptor


class _OwnedWindowsHandle:
    """A native Windows handle backed by the same transferable ownership token."""

    __slots__ = ("_token",)

    def __init__(self, token: _ResourceOwnershipToken) -> None:
        self._token = token

    @property
    def value(self) -> int | None:
        return self._token.windows_handle


_WINDOWS_DESCRIPTOR_TRANSFER_STATE = threading.local()


def _windows_descriptor_transfer_stack() -> (
    list[weakref.ReferenceType[_WindowsDescriptorTransfer]]
):
    stack = getattr(_WINDOWS_DESCRIPTOR_TRANSFER_STATE, "stack", None)
    if stack is None:
        stack = []
        _WINDOWS_DESCRIPTOR_TRANSFER_STATE.stack = stack
    return stack


def _current_windows_descriptor_transfer() -> _WindowsDescriptorTransfer | None:
    stack = _windows_descriptor_transfer_stack()
    while stack:
        transfer = stack[-1]()
        if transfer is not None:
            return transfer
        stack.pop()
    return None


class _WindowsDescriptorTransfer:
    """Keep the source handle owned until the CRT result itself can take over."""

    __slots__ = ("_token", "_settled", "__weakref__")

    def __init__(self, raw_owner: _OwnedWindowsHandle) -> None:
        self._token = raw_owner._token
        self._settled = False
        _windows_descriptor_transfer_stack().append(weakref.ref(self))

    def _remove_from_stack(self) -> None:
        stack = _windows_descriptor_transfer_stack()
        stack[:] = [
            reference
            for reference in stack
            if (transfer := reference()) is not None and transfer is not self
        ]

    def result_type(self) -> type[_WindowsTransferredDescriptorResult]:
        """Return a ctypes result type bound only to this transfer guard."""

        return type(
            "_BoundWindowsTransferredDescriptorResult",
            (_WindowsTransferredDescriptorResult,),
            {
                "__module__": __name__,
                "_source_transfer_reference": weakref.ref(self),
            },
        )

    def _require_source(self, result: _WindowsTransferredDescriptorResult) -> None:
        if result._source_transfer() is not self:
            _fail("Windows descriptor result belongs to a different transfer")

    def adopt_result(
        self, result: _WindowsTransferredDescriptorResult
    ) -> _OwnedDescriptor:
        self._require_source(result)
        if self._settled:
            _fail("Windows handle conversion ownership was already settled")
        self._token.adopt_descriptor(result)
        self._settled = True
        self._remove_from_stack()
        return _OwnedDescriptor(self._token)

    def abandon_result(self, result: _WindowsTransferredDescriptorResult) -> None:
        self._require_source(result)
        if self._settled:
            result.close_quietly()
            return
        if result.value >= 0:
            self._token.adopt_descriptor(result)
        self._token.close_quietly()
        self._settled = True
        self._remove_from_stack()

    def abandon_without_result(self) -> None:
        if self._settled:
            return
        self._token.close_quietly()
        self._settled = True
        self._remove_from_stack()

    def __del__(self) -> None:
        try:
            self.abandon_without_result()
        except BaseException:
            pass


class BoundDirectoryHandle:
    """An open directory descriptor bound to one filesystem object."""

    __slots__ = ("path", "identity", "_descriptor_owner")

    def __init__(
        self,
        path: Path,
        identity: DirectoryIdentity,
        descriptor: int | _OwnedDescriptor,
    ) -> None:
        self.path = path
        self.identity = identity
        self._descriptor_owner = (
            descriptor
            if isinstance(descriptor, _OwnedDescriptor)
            else _OwnedDescriptor(descriptor)
        )
        self._descriptor_owner.bind_identity(identity)

    @property
    def descriptor(self) -> int:
        return self._descriptor_owner.descriptor

    @descriptor.setter
    def descriptor(self, value: int) -> None:
        if value != -1:
            raise ValueError("a bound directory descriptor can only be retired")
        self._descriptor_owner.relinquish()

    def close(self) -> None:
        self._descriptor_owner.close()


def _register_owned_directory_handle(
    handle: BoundDirectoryHandle,
    owned_handles: list[BoundDirectoryHandle] | None,
) -> BoundDirectoryHandle:
    if owned_handles is not None:
        owned_handles.append(handle)
    return handle


def _close_owned_directory_handles(
    handles: Sequence[BoundDirectoryHandle],
    *,
    active_exception: BaseException | None = None,
) -> None:
    """Close each owned handle, retrying interruptions without masking a primary."""
    close_exception: BaseException | None = None
    close_traceback = None
    seen: set[int] = set()
    for handle in reversed(handles):
        marker = id(handle)
        if marker in seen:
            continue
        seen.add(marker)
        for _ in range(2):
            if handle.descriptor < 0:
                break
            try:
                handle.close()
            except BaseException as exc:
                if close_exception is None:
                    close_exception = exc
                    close_traceback = exc.__traceback__
                continue
            break
    if active_exception is None and close_exception is not None:
        raise close_exception.with_traceback(close_traceback)


def _fail(message: str) -> NoReturn:
    raise ReleaseValidationError(message)


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    return _stream_stable_file(
        path,
        label=f"file {path}",
        chunk_size=chunk_size,
    ).sha256


def _lexical_absolute(path: Path) -> Path:
    """Return an absolute path without resolving links or reparse points."""
    return Path(os.path.abspath(os.fspath(path)))


def _is_link_or_reparse(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ReleaseValidationError(
            f"cannot inspect path component for links/reparse points: {path}"
        ) from exc
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(metadata.st_mode) or bool(
        getattr(metadata, "st_file_attributes", 0) & reparse_flag
    )


def _assert_no_link_or_reparse_components(path: Path, *, label: str) -> Path:
    """Reject symlinks and Windows junction/reparse points before resolution."""
    absolute = _lexical_absolute(path)
    anchor = Path(absolute.anchor)
    current = anchor
    parts = absolute.parts[1:] if absolute.anchor else absolute.parts
    for part in parts:
        current = current / part
        if _is_link_or_reparse(current):
            _fail(f"{label} traverses a symbolic link, junction, or reparse point")
    return absolute


def _path_component_identities(path: Path, *, label: str) -> PathComponentIdentities:
    """Bind every existing lexical component without following a link."""
    absolute = _assert_no_link_or_reparse_components(path, label=label)
    anchor = Path(absolute.anchor)
    current = anchor
    identities: list[tuple[int, int, int]] = []
    parts = absolute.parts[1:] if absolute.anchor else absolute.parts
    try:
        for part in parts:
            current = current / part
            metadata = current.stat(follow_symlinks=False)
            if _is_link_or_reparse(current):
                _fail(f"{label} traverses a symbolic link, junction, or reparse point")
            identities.append((metadata.st_dev, metadata.st_ino, metadata.st_mode))
    except ReleaseValidationError:
        raise
    except OSError as exc:
        raise ReleaseValidationError(
            f"{label} path components could not be bound"
        ) from exc
    return tuple(identities)


def _bind_directory(path: Path, *, label: str) -> tuple[Path, DirectoryIdentity]:
    absolute = _assert_no_link_or_reparse_components(path, label=label)
    try:
        metadata = absolute.stat(follow_symlinks=False)
    except OSError as exc:
        raise ReleaseValidationError(f"{label} is unavailable") from exc
    if _is_link_or_reparse(absolute) or not stat.S_ISDIR(metadata.st_mode):
        _fail(f"{label} is linked, reparse-backed, or not a directory")
    return absolute, (metadata.st_dev, metadata.st_ino, metadata.st_mode)


def _assert_directory_identity(
    path: Path, expected: DirectoryIdentity, *, label: str
) -> Path:
    absolute, observed = _bind_directory(path, label=label)
    if observed != expected:
        _fail(f"{label} changed after it was bound")
    return absolute


def _directory_identity_from_stat(metadata: os.stat_result) -> DirectoryIdentity:
    return metadata.st_dev, metadata.st_ino, metadata.st_mode


def _validate_child_name(name: str, *, label: str) -> str:
    if (
        not name
        or name in {".", ".."}
        or "\x00" in name
        or "/" in name
        or "\\" in name
        or Path(name).name != name
    ):
        _fail(f"{label} is not a single safe path component")
    return name


def _coerce_owned_descriptor(
    descriptor: int | _OwnedDescriptor,
) -> _OwnedDescriptor:
    if isinstance(descriptor, _OwnedDescriptor):
        return descriptor
    return _OwnedDescriptor(descriptor)


def _posix_open_owned_descriptor(
    path: str | Path,
    flags: int,
    mode: int = 0,
    *,
    dir_fd: int | None = None,
) -> _OwnedDescriptor:
    """Open with a native result object that owns the fd before Python resumes."""
    libc = ctypes.CDLL(None, use_errno=True)
    if dir_fd is None:
        open_function = libc.open
        open_function.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint]
        arguments: tuple[Any, ...] = (os.fsencode(path), flags, mode)
    else:
        open_function = libc.openat
        open_function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_uint,
        ]
        arguments = (dir_fd, os.fsencode(path), flags, mode)
    open_function.restype = _NativeDescriptorResult
    result = open_function(*arguments)
    if result.value < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), os.fspath(path))
    return _OwnedDescriptor(result)


def _windows_descriptor_from_handle(
    raw_owner: _OwnedWindowsHandle,
    flags: int,
) -> _OwnedDescriptor:
    """Transfer a Win32 handle while both native return values remain owned."""
    raw_value = raw_owner.value
    if raw_value is None:
        _fail("Windows handle conversion received an invalid handle")
    ucrt = ctypes.CDLL("ucrtbase", use_errno=True)
    open_osfhandle = ucrt._open_osfhandle
    open_osfhandle.argtypes = [ctypes.c_ssize_t, ctypes.c_int]
    transfer = _WindowsDescriptorTransfer(raw_owner)
    open_osfhandle.restype = transfer.result_type()
    result: _WindowsTransferredDescriptorResult | None = None
    try:
        result = open_osfhandle(raw_value, flags)
        if result.value < 0:
            error = ctypes.get_errno()
            transfer.abandon_result(result)
            raise OSError(error, os.strerror(error))
        return transfer.adopt_result(result)
    except BaseException:
        if not transfer._settled:
            if result is None:
                transfer.abandon_without_result()
            else:
                transfer.abandon_result(result)
        else:
            transfer._remove_from_stack()
        raise


def _windows_open_directory_descriptor(path: Path) -> _OwnedDescriptor:
    from ctypes import wintypes

    kernel32 = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = _NativeWindowsHandleResult
    desired_access = 0x0001 | 0x0080 | 0x00100000
    share_access = 0x00000001 | 0x00000002 | 0x00000004
    flags = 0x02000000 | 0x00200000
    invalid_handle = ctypes.c_void_p(-1).value
    raw_handle = create_file(
        str(path),
        desired_access,
        share_access,
        None,
        3,
        flags,
        None,
    )
    if raw_handle.value == invalid_handle:
        raw_handle.value = None
        raise ctypes.WinError(ctypes.get_last_error())
    raw_owner = _OwnedWindowsHandle(
        _ResourceOwnershipToken.for_windows_handle(raw_handle, close_handle)
    )
    return _windows_descriptor_from_handle(raw_owner, os.O_RDONLY)


def _windows_nt_create_relative(
    parent: BoundDirectoryHandle,
    name: str,
    *,
    directory: bool,
    create_new: bool = True,
    inspection_only: bool = False,
) -> _OwnedDescriptor:
    import msvcrt
    from ctypes import wintypes

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
        _fields_ = [("Status", wintypes.LPVOID), ("Information", ctypes.c_size_t)]

    ntdll = getattr(ctypes, "WinDLL")("ntdll")
    nt_create_file = ntdll.NtCreateFile
    nt_create_file.argtypes = [
        ctypes.POINTER(wintypes.HANDLE),
        wintypes.ULONG,
        ctypes.POINTER(ObjectAttributes),
        ctypes.POINTER(IoStatusBlock),
        wintypes.LPVOID,
        wintypes.ULONG,
        wintypes.ULONG,
        wintypes.ULONG,
        wintypes.ULONG,
        wintypes.LPVOID,
        wintypes.ULONG,
    ]
    nt_create_file.restype = wintypes.LONG
    rtl_status_to_error = ntdll.RtlNtStatusToDosError
    rtl_status_to_error.argtypes = [wintypes.LONG]
    rtl_status_to_error.restype = wintypes.ULONG

    name_buffer = ctypes.create_unicode_buffer(name)
    encoded_length = len(name.encode("utf-16-le"))
    unicode_name = UnicodeString(
        encoded_length,
        encoded_length + ctypes.sizeof(wintypes.WCHAR),
        ctypes.cast(name_buffer, wintypes.LPWSTR),
    )
    parent_handle = msvcrt.get_osfhandle(parent.descriptor)
    attributes = ObjectAttributes(
        ctypes.sizeof(ObjectAttributes),
        wintypes.HANDLE(parent_handle),
        ctypes.pointer(unicode_name),
        0x00000040,
        None,
        None,
    )
    io_status = IoStatusBlock()
    kernel32 = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    raw_handle = _NativeWindowsHandleResult()
    raw_owner = _OwnedWindowsHandle(
        _ResourceOwnershipToken.for_windows_handle(raw_handle, close_handle)
    )
    if directory:
        desired_access = 0x0080 | 0x00100000
        if not inspection_only:
            desired_access |= 0x0001 | 0x0002 | 0x0004 | 0x00010000
        file_attributes = 0x00000010
        create_options = 0x00000001 | 0x00000020 | 0x00200000
        open_flags = os.O_RDONLY
    else:
        desired_access = 0x80000000 | 0x40000000 | 0x0080 | 0x00100000
        file_attributes = 0x00000080
        create_options = 0x00000040 | 0x00000020 | 0x00200000
        open_flags = os.O_RDWR | os.O_BINARY
    status = nt_create_file(
        ctypes.byref(raw_handle),
        desired_access,
        ctypes.byref(attributes),
        ctypes.byref(io_status),
        None,
        file_attributes,
        0x00000001 | 0x00000002 | (0x00000004 if inspection_only else 0),
        2 if create_new else 1,
        create_options,
        None,
        0,
    )
    if status < 0:
        raise ctypes.WinError(int(rtl_status_to_error(status)))
    raw_value = raw_owner.value
    if raw_value is None:
        _fail("Windows relative creation returned an invalid handle")
    return _windows_descriptor_from_handle(raw_owner, open_flags)


def _open_bound_directory_handle(
    path: Path,
    expected: DirectoryIdentity,
    *,
    label: str,
    owned_handles: list[BoundDirectoryHandle] | None = None,
) -> BoundDirectoryHandle:
    descriptor_owner: _OwnedDescriptor | None = None
    handle: BoundDirectoryHandle | None = None
    try:
        if os.name == "nt":
            descriptor_owner = _coerce_owned_descriptor(
                _windows_open_directory_descriptor(path)
            )
        else:
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor_owner = _posix_open_owned_descriptor(path, flags)
        descriptor = descriptor_owner.descriptor
        opened = os.fstat(descriptor)
        opened_identity = _directory_identity_from_stat(opened)
        if not stat.S_ISDIR(opened.st_mode) or opened_identity != expected:
            _fail(f"{label} changed while its directory handle was opened")
        absolute = _assert_directory_identity(path, expected, label=label)
        handle = BoundDirectoryHandle(absolute, expected, descriptor_owner)
        return _register_owned_directory_handle(handle, owned_handles)
    except BaseException as primary:
        primary_traceback = primary.__traceback__
        registered = owned_handles is not None and any(
            candidate is handle for candidate in owned_handles
        )
        if descriptor_owner is not None and not registered:
            descriptor_owner.close_quietly()
        raise primary.with_traceback(primary_traceback)


def _open_bound_child_directory_handle(
    parent: BoundDirectoryHandle,
    name: str,
    expected: DirectoryIdentity,
    *,
    path: Path,
    label: str,
    owned_handles: list[BoundDirectoryHandle] | None = None,
) -> BoundDirectoryHandle:
    """Open an existing child relative to a bound parent and verify its identity."""
    child_name = _validate_child_name(name, label=label)
    _assert_bound_directory_handle(parent, label=f"{label} parent")
    descriptor_owner: _OwnedDescriptor | None = None
    handle: BoundDirectoryHandle | None = None
    try:
        if os.name == "nt":
            descriptor_owner = _coerce_owned_descriptor(
                _windows_nt_create_relative(
                    parent,
                    child_name,
                    directory=True,
                    create_new=False,
                )
            )
        else:
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor_owner = _posix_open_owned_descriptor(
                child_name,
                flags,
                dir_fd=parent.descriptor,
            )
        descriptor = descriptor_owner.descriptor
        observed = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(observed.st_mode)
            or _directory_identity_from_stat(observed) != expected
        ):
            _fail(f"{label} changed while its directory handle was opened")
        handle = BoundDirectoryHandle(path, expected, descriptor_owner)
        return _register_owned_directory_handle(handle, owned_handles)
    except BaseException as primary:
        primary_traceback = primary.__traceback__
        registered = owned_handles is not None and any(
            candidate is handle for candidate in owned_handles
        )
        if descriptor_owner is not None and not registered:
            descriptor_owner.close_quietly()
        raise primary.with_traceback(primary_traceback)


def _assert_bound_directory_handle(
    directory: BoundDirectoryHandle,
    *,
    label: str,
) -> None:
    try:
        observed = os.fstat(directory.descriptor)
    except OSError as exc:
        raise ReleaseValidationError(f"{label} handle is unavailable") from exc
    if (
        not stat.S_ISDIR(observed.st_mode)
        or _directory_identity_from_stat(observed) != directory.identity
    ):
        _fail(f"{label} handle changed after it was bound")


def _create_bound_child_directory(
    parent: BoundDirectoryHandle,
    name: str,
    *,
    label: str,
    owned_handles: list[BoundDirectoryHandle] | None = None,
) -> BoundDirectoryHandle:
    child_name = _validate_child_name(name, label=label)
    _assert_bound_directory_handle(parent, label=f"{label} parent")
    descriptor_owner: _OwnedDescriptor | None = None
    provisional_identity: DirectoryIdentity | None = None
    handle: BoundDirectoryHandle | None = None
    child_path = parent.path / child_name
    try:
        try:
            if os.name == "nt":
                descriptor_owner = _coerce_owned_descriptor(
                    _windows_nt_create_relative(
                        parent,
                        child_name,
                        directory=True,
                    )
                )
            else:
                os.mkdir(child_name, mode=0o700, dir_fd=parent.descriptor)
                flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                flags |= getattr(os, "O_NOFOLLOW", 0)
                descriptor_owner = _posix_open_owned_descriptor(
                    child_name,
                    flags,
                    dir_fd=parent.descriptor,
                )
        except OSError as exc:
            raise ReleaseValidationError(
                f"{label} could not be created safely"
            ) from exc
        descriptor = descriptor_owner.descriptor
        metadata = os.fstat(descriptor)
        provisional_identity = _directory_identity_from_stat(metadata)
        if not stat.S_ISDIR(metadata.st_mode):
            _fail(f"{label} is not a directory after creation")
        if os.name != "nt":
            os.fchmod(descriptor, 0o700)
            metadata = os.fstat(descriptor)
            provisional_identity = _directory_identity_from_stat(metadata)
        bound_path, bound_identity = _bind_directory(child_path, label=label)
        if bound_identity != provisional_identity:
            _fail(f"{label} changed during initial binding")
        handle = BoundDirectoryHandle(bound_path, bound_identity, descriptor_owner)
        return _register_owned_directory_handle(handle, owned_handles)
    except BaseException as primary:
        primary_traceback = primary.__traceback__
        registered = owned_handles is not None and any(
            candidate is handle for candidate in owned_handles
        )
        if registered:
            raise primary.with_traceback(primary_traceback)
        if provisional_identity is None and descriptor_owner is not None:
            try:
                recovered = os.fstat(descriptor_owner.descriptor)
                if stat.S_ISDIR(recovered.st_mode):
                    provisional_identity = _directory_identity_from_stat(recovered)
            except BaseException:
                pass
        if provisional_identity is not None:
            provisional_handle = (
                BoundDirectoryHandle(
                    child_path,
                    provisional_identity,
                    descriptor_owner,
                )
                if descriptor_owner is not None
                else None
            )
            try:
                _quarantine_bound_directory(
                    child_path,
                    provisional_identity,
                    label=label,
                    parent_handle=parent,
                    child_handle=provisional_handle,
                )
            except BaseException:
                pass
        if descriptor_owner is not None:
            descriptor_owner.close_quietly()
        raise primary.with_traceback(primary_traceback)


class _OwnedBufferedRandom(io.BufferedRandom):
    """Buffered file whose descriptor remains token-owned until close."""

    def __init__(self, descriptor_owner: _OwnedDescriptor) -> None:
        self._descriptor_owner = descriptor_owner
        raw = io.FileIO(descriptor_owner.descriptor, mode="r+", closefd=False)
        super().__init__(raw)

    @property
    def owned_descriptor(self) -> int:
        """Return the separately owned OS descriptor, even after buffering closes."""
        return self._descriptor_owner.descriptor

    def close(self) -> None:
        primary: BaseException | None = None
        primary_traceback = None
        try:
            super().close()
        except BaseException as exc:
            primary = exc
            primary_traceback = exc.__traceback__
        try:
            self._descriptor_owner.close()
        except BaseException:
            if primary is None:
                raise
        if primary is not None:
            raise primary.with_traceback(primary_traceback)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        if exc_value is None:
            self.close()
            return
        try:
            self.close()
        except BaseException:
            # The with-body exception is primary. Cleanup cannot replace it.
            pass


def _register_owned_output_descriptor(
    descriptor_owner: _OwnedDescriptor,
    owned_descriptors: list[_OwnedDescriptor] | None,
) -> _OwnedDescriptor:
    if owned_descriptors is not None:
        owned_descriptors.append(descriptor_owner)
    return descriptor_owner


def _close_owned_output_descriptors(
    descriptor_owners: Sequence[_OwnedDescriptor],
    *,
    active_exception: BaseException | None = None,
) -> None:
    """Retire transaction-owned output descriptors before ancestor quarantine."""
    close_exception: BaseException | None = None
    close_traceback = None
    seen: set[int] = set()
    for descriptor_owner in reversed(descriptor_owners):
        marker = id(descriptor_owner)
        if marker in seen:
            continue
        seen.add(marker)
        for _ in range(2):
            if descriptor_owner.descriptor < 0:
                break
            try:
                descriptor_owner.close()
            except BaseException as exc:
                if close_exception is None:
                    close_exception = exc
                    close_traceback = exc.__traceback__
                continue
            break
    if active_exception is None and close_exception is not None:
        raise close_exception.with_traceback(close_traceback)


def _open_new_file_at(
    parent: BoundDirectoryHandle,
    name: str,
    *,
    label: str,
    owned_descriptors: list[_OwnedDescriptor] | None = None,
) -> Any:
    child_name = _validate_child_name(name, label=label)
    _assert_bound_directory_handle(parent, label=f"{label} parent")
    try:
        if os.name == "nt":
            descriptor_owner = _coerce_owned_descriptor(
                _windows_nt_create_relative(
                    parent,
                    child_name,
                    directory=False,
                )
            )
        else:
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor_owner = _posix_open_owned_descriptor(
                child_name,
                flags,
                0o600,
                dir_fd=parent.descriptor,
            )
    except OSError as exc:
        raise ReleaseValidationError(f"{label} could not be created safely") from exc
    try:
        metadata = os.fstat(descriptor_owner.descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            _fail(f"{label} is not a singly linked regular file after creation")
        descriptor_owner.bind_identity(_directory_identity_from_stat(metadata))
        descriptor_owner = _register_owned_output_descriptor(
            descriptor_owner,
            owned_descriptors,
        )
        return _OwnedBufferedRandom(descriptor_owner)
    except BaseException as primary:
        primary_traceback = primary.__traceback__
        descriptor_owner.close_quietly()
        raise primary.with_traceback(primary_traceback)


def _fingerprint_open_file(handle: Any, *, label: str) -> FileFingerprint:
    try:
        handle.flush()
        os.fsync(handle.fileno())
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            _fail(f"{label} is not a singly linked regular file")
        handle.seek(0)
        digest = hashlib.sha256()
        size = 0
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(handle.fileno())
    except ReleaseValidationError:
        raise
    except OSError as exc:
        raise ReleaseValidationError(f"{label} could not be fingerprinted") from exc
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_size",
        "st_mtime_ns",
        "st_nlink",
    )
    if (
        any(getattr(before, field) != getattr(after, field) for field in stable_fields)
        or size != before.st_size
    ):
        _fail(f"{label} changed while it was fingerprinted")
    return FileFingerprint(
        bytes=size,
        sha256=digest.hexdigest(),
        device=before.st_dev,
        inode=before.st_ino,
        mode=before.st_mode,
        modified_ns=before.st_mtime_ns,
        links=before.st_nlink,
    )


def _write_new_bound_file(
    parent: BoundDirectoryHandle,
    name: str,
    payload: bytes,
    *,
    label: str,
    owned_descriptors: list[_OwnedDescriptor] | None = None,
) -> FileFingerprint:
    with _open_new_file_at(
        parent,
        name,
        label=label,
        owned_descriptors=owned_descriptors,
    ) as handle:
        handle.write(payload)
        return _fingerprint_open_file(handle, label=label)


def _windows_rename_bound_child(
    parent: BoundDirectoryHandle,
    child: BoundDirectoryHandle,
    destination_name: str,
) -> None:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    class IoStatusBlock(ctypes.Structure):
        _fields_ = [("Status", wintypes.LPVOID), ("Information", ctypes.c_size_t)]

    encoded = destination_name.encode("utf-16-le")
    buffer = ctypes.create_string_buffer(20 + len(encoded))
    ctypes.c_ubyte.from_buffer(buffer, 0).value = 0
    ctypes.c_void_p.from_buffer(buffer, 8).value = msvcrt.get_osfhandle(
        parent.descriptor
    )
    ctypes.c_ulong.from_buffer(buffer, 16).value = len(encoded)
    ctypes.memmove(ctypes.addressof(buffer) + 20, encoded, len(encoded))
    ntdll = getattr(ctypes, "WinDLL")("ntdll")
    nt_set_information = ntdll.NtSetInformationFile
    nt_set_information.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(IoStatusBlock),
        wintypes.LPVOID,
        wintypes.ULONG,
        wintypes.ULONG,
    ]
    nt_set_information.restype = wintypes.LONG
    rtl_status_to_error = ntdll.RtlNtStatusToDosError
    rtl_status_to_error.argtypes = [wintypes.LONG]
    rtl_status_to_error.restype = wintypes.ULONG
    io_status = IoStatusBlock()
    status = nt_set_information(
        wintypes.HANDLE(msvcrt.get_osfhandle(child.descriptor)),
        ctypes.byref(io_status),
        ctypes.byref(buffer),
        len(buffer),
        10,
    )
    if status < 0:
        raise ctypes.WinError(int(rtl_status_to_error(status)))


def _posix_rename_bound_child(
    source_parent: BoundDirectoryHandle,
    child_name: str,
    destination_parent: BoundDirectoryHandle,
    destination_name: str,
) -> None:
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    source = os.fsencode(child_name)
    destination = os.fsencode(destination_name)
    result = -1
    if sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            source_parent.descriptor,
            source,
            destination_parent.descriptor,
            destination,
            1,
        )
    elif sys.platform == "darwin" and hasattr(libc, "renameatx_np"):
        rename = libc.renameatx_np
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            source_parent.descriptor,
            source,
            destination_parent.descriptor,
            destination,
            0x00000004,
        )
    else:
        _fail("atomic handle-relative no-replace install is unavailable")
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), destination_name)


def _relative_directory_identity(
    parent: BoundDirectoryHandle,
    name: str,
    *,
    label: str,
) -> DirectoryIdentity | None:
    """Read one child identity relative to an already-bound parent handle."""
    child_name = _validate_child_name(name, label=label)
    _assert_bound_directory_handle(parent, label=f"{label} parent")
    if os.name == "nt":
        descriptor_owner: _OwnedDescriptor | None = None
        primary: BaseException | None = None
        try:
            try:
                descriptor_owner = _coerce_owned_descriptor(
                    _windows_nt_create_relative(
                        parent,
                        child_name,
                        directory=True,
                        create_new=False,
                        inspection_only=True,
                    )
                )
                metadata = os.fstat(descriptor_owner.descriptor)
            except FileNotFoundError:
                return None
            except OSError as exc:
                raise ReleaseValidationError(
                    f"{label} could not be inspected safely"
                ) from exc
        except BaseException as exc:
            primary = exc
            raise
        finally:
            if descriptor_owner is not None:
                try:
                    descriptor_owner.close()
                except BaseException:
                    if primary is None:
                        raise
    else:
        try:
            metadata = os.stat(
                child_name,
                dir_fd=parent.descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise ReleaseValidationError(
                f"{label} could not be inspected safely"
            ) from exc
    if not stat.S_ISDIR(metadata.st_mode):
        _fail(f"{label} is not a directory")
    return _directory_identity_from_stat(metadata)


def _rename_bound_child_no_replace(
    source_parent: BoundDirectoryHandle,
    child: BoundDirectoryHandle,
    source_name: str,
    destination_parent: BoundDirectoryHandle,
    destination_name: str,
    *,
    label: str,
) -> None:
    """Move a bound child between bound parents without replacing a destination."""
    source = _validate_child_name(source_name, label=f"{label} source name")
    destination = _validate_child_name(
        destination_name,
        label=f"{label} destination name",
    )
    _assert_bound_directory_handle(source_parent, label=f"{label} source parent")
    _assert_bound_directory_handle(
        destination_parent, label=f"{label} destination parent"
    )
    _assert_bound_directory_handle(child, label=label)
    if (
        _relative_directory_identity(
            source_parent,
            source,
            label=f"{label} source",
        )
        != child.identity
    ):
        _fail(f"{label} source name no longer identifies the bound directory")
    if (
        _relative_directory_identity(
            destination_parent,
            destination,
            label=f"{label} destination",
        )
        is not None
    ):
        _fail(f"{label} destination already exists")
    if os.name == "nt":
        _windows_rename_bound_child(destination_parent, child, destination)
    else:
        _posix_rename_bound_child(
            source_parent,
            source,
            destination_parent,
            destination,
        )
    child.path = destination_parent.path / destination
    _assert_bound_directory_handle(child, label=label)
    if (
        _relative_directory_identity(
            destination_parent,
            destination,
            label=f"{label} destination",
        )
        != child.identity
    ):
        _fail(f"{label} destination does not identify the bound directory")


def _install_bound_child(
    parent: BoundDirectoryHandle,
    child: BoundDirectoryHandle,
    destination_name: str,
    *,
    source_parent: BoundDirectoryHandle | None = None,
) -> None:
    destination = _validate_child_name(
        destination_name,
        label="release output name",
    )
    actual_source_parent = source_parent if source_parent is not None else parent
    _assert_bound_directory_handle(parent, label="release output parent")
    _assert_bound_directory_handle(
        actual_source_parent,
        label="release private staging parent",
    )
    _assert_bound_directory_handle(child, label="release staging directory")
    source_name = _validate_child_name(
        child.path.name,
        label="release staging name",
    )
    try:
        _rename_bound_child_no_replace(
            actual_source_parent,
            child,
            source_name,
            parent,
            destination,
            label="release staging directory",
        )
    except ReleaseValidationError:
        raise
    except OSError as exc:
        raise ReleaseValidationError(
            "release staging directory could not be installed atomically"
        ) from exc


def _remove_empty_bound_child(
    parent: BoundDirectoryHandle,
    child: BoundDirectoryHandle,
    child_name: str,
    *,
    label: str,
) -> None:
    """Remove an empty POSIX child relative to its still-bound parent."""
    if os.name == "nt":
        _fail(f"{label} empty-directory retirement is unavailable on Windows")
    name = _validate_child_name(child_name, label=f"{label} name")
    _assert_bound_directory_handle(parent, label=f"{label} parent")
    _assert_bound_directory_handle(child, label=label)
    if _relative_directory_identity(parent, name, label=label) != child.identity:
        _fail(f"{label} name no longer identifies the bound directory")
    try:
        os.rmdir(name, dir_fd=parent.descriptor)
    except OSError as exc:
        raise ReleaseValidationError(f"{label} could not be retired safely") from exc


def _quarantine_bound_directory(
    path: Path,
    expected: DirectoryIdentity | None,
    *,
    label: str,
    parent_handle: BoundDirectoryHandle | None = None,
    child_handle: BoundDirectoryHandle | None = None,
) -> bool:
    """Detach a directory relative to a bound parent without recursive deletion."""
    if expected is None:
        return False
    owned_handles: list[BoundDirectoryHandle] = []
    parent: BoundDirectoryHandle | None = None
    child: BoundDirectoryHandle | None = None
    primary: BaseException | None = None
    try:
        try:
            absolute = _lexical_absolute(path)
            source_name = _validate_child_name(absolute.name, label=f"{label} name")
            if parent_handle is None:
                absolute = _assert_directory_identity(path, expected, label=label)
                parent_path, parent_identity = _bind_directory(
                    absolute.parent,
                    label=f"{label} parent",
                )
                parent = _open_bound_directory_handle(
                    parent_path,
                    parent_identity,
                    label=f"{label} parent",
                    owned_handles=owned_handles,
                )
            else:
                parent = parent_handle
                _assert_bound_directory_handle(parent, label=f"{label} parent")
                if _lexical_absolute(parent.path) != absolute.parent:
                    _fail(f"{label} parent handle does not bind the path parent")
            if (
                _relative_directory_identity(
                    parent,
                    source_name,
                    label=label,
                )
                != expected
            ):
                return False
            if child_handle is None:
                child = _open_bound_child_directory_handle(
                    parent,
                    source_name,
                    expected,
                    path=absolute,
                    label=label,
                    owned_handles=owned_handles,
                )
            else:
                child = child_handle
                _assert_bound_directory_handle(child, label=label)
                if child.identity != expected:
                    _fail(f"{label} handle does not match the expected directory")

            quarantine_name = f".{source_name}.cleanup-{uuid.uuid4().hex}"
            pending_exception: BaseException | None = None
            try:
                _rename_bound_child_no_replace(
                    parent,
                    child,
                    source_name,
                    parent,
                    quarantine_name,
                    label=label,
                )
            except BaseException as exc:
                pending_exception = exc
                primary = exc
            try:
                _assert_bound_directory_handle(parent, label=f"{label} parent")
                _assert_bound_directory_handle(child, label=f"quarantined {label}")
                if (
                    _relative_directory_identity(
                        parent,
                        quarantine_name,
                        label=f"quarantined {label}",
                    )
                    != expected
                ):
                    if pending_exception is not None and not isinstance(
                        pending_exception, Exception
                    ):
                        raise pending_exception
                    return False
            except BaseException as exc:
                primary = exc
                if pending_exception is not None and not isinstance(
                    pending_exception, Exception
                ):
                    primary = pending_exception
                    raise pending_exception
                if not isinstance(exc, Exception):
                    raise
                return False
            # Retain the quarantine: recursively deleting by pathname would reopen a
            # replacement race after the final descriptor-relative identity check.
            if pending_exception is not None and not isinstance(
                pending_exception, Exception
            ):
                primary = pending_exception
                raise pending_exception
            return True
        except (OSError, ReleaseValidationError) as exc:
            primary = exc
            return False
    except BaseException as exc:
        primary = exc
        raise
    finally:
        _close_owned_directory_handles(
            owned_handles,
            active_exception=primary,
        )


def _capture_new_directory_identity(
    path: Path,
    *,
    label: str,
) -> tuple[Path, DirectoryIdentity]:
    """Capture a just-created directory before performing full path validation."""
    absolute = _lexical_absolute(path)
    try:
        metadata = absolute.stat(follow_symlinks=False)
    except OSError as exc:
        raise ReleaseValidationError(f"{label} is unavailable after creation") from exc
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or bool(getattr(metadata, "st_file_attributes", 0) & reparse_flag)
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        _fail(f"{label} is linked, reparse-backed, or not a directory")
    return absolute, (metadata.st_dev, metadata.st_ino, metadata.st_mode)


def _create_bound_directory(
    path: Path, *, label: str
) -> tuple[Path, DirectoryIdentity]:
    provisional_identity: DirectoryIdentity | None = None
    try:
        path.mkdir(exist_ok=False)
        path, provisional_identity = _capture_new_directory_identity(
            path,
            label=label,
        )
        bound_path, bound_identity = _bind_directory(path, label=label)
        if bound_identity != provisional_identity:
            _fail(f"{label} changed during initial binding")
        return bound_path, bound_identity
    except BaseException:
        if provisional_identity is not None:
            try:
                _quarantine_bound_directory(
                    path,
                    provisional_identity,
                    label=label,
                )
            except BaseException:
                pass
        raise


def _stream_stable_file(
    path: Path,
    *,
    label: str,
    target: Any | None = None,
    chunk_size: int = 4 * 1024 * 1024,
) -> FileFingerprint:
    """Hash a regular file while rejecting path or inode changes during the read."""
    try:
        component_identities = _path_component_identities(path, label=label)
        before = path.stat(follow_symlinks=False)
        if _is_link_or_reparse(path) or not stat.S_ISREG(before.st_mode):
            _fail(f"{label} is linked or not a regular file")
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as source:
            opened_before = os.fstat(source.fileno())
            stable_fields = (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_size",
                "st_mtime_ns",
                "st_nlink",
            )
            if any(
                getattr(before, field) != getattr(opened_before, field)
                for field in stable_fields
            ):
                _fail(f"{label} changed while it was opened")
            while chunk := source.read(chunk_size):
                if target is not None:
                    target.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            opened_after = os.fstat(source.fileno())
        after = path.stat(follow_symlinks=False)
        final_component_identities = _path_component_identities(path, label=label)
    except ReleaseValidationError:
        raise
    except OSError as exc:
        raise ReleaseValidationError(f"{label} could not be read stably") from exc
    if _is_link_or_reparse(path) or not stat.S_ISREG(after.st_mode):
        _fail(f"{label} became linked or non-regular while it was read")
    if final_component_identities != component_identities:
        _fail(f"{label} path components changed while it was being read")
    snapshots = (opened_before, opened_after, after)
    if (
        any(
            getattr(before, field) != getattr(snapshot, field)
            for snapshot in snapshots
            for field in stable_fields
        )
        or size != before.st_size
    ):
        _fail(f"{label} changed while it was being read")
    return FileFingerprint(
        bytes=size,
        sha256=digest.hexdigest(),
        device=before.st_dev,
        inode=before.st_ino,
        mode=before.st_mode,
        modified_ns=before.st_mtime_ns,
        links=before.st_nlink,
    )


def _fingerprint_file(path: Path, *, label: str) -> FileFingerprint:
    return _stream_stable_file(path, label=label)


def _read_stable_bytes(path: Path, *, label: str) -> tuple[bytes, FileFingerprint]:
    buffer = io.BytesIO()
    fingerprint = _stream_stable_file(path, label=label, target=buffer)
    return buffer.getvalue(), fingerprint


def _bind_executing_builder(root: Path) -> TrustedModuleSource:
    relative = "code/build_public_release.py"
    selected_path = _safe_workspace_file(
        root,
        relative,
        field="executing release builder",
    )
    try:
        reported = Path(__file__).resolve(strict=True)
    except (OSError, TypeError, ValueError) as exc:
        raise ReleaseValidationError(
            "executing release builder origin cannot be resolved"
        ) from exc
    expected_path = selected_path.resolve(strict=True)
    if reported != _EXECUTING_BUILDER_PATH or reported != expected_path:
        _fail(
            "executing release builder origin differs from selected workspace "
            "code/build_public_release.py"
        )
    payload, fingerprint = _read_stable_bytes(
        selected_path,
        label="executing release builder source",
    )
    imported_identity = (
        fingerprint.bytes,
        fingerprint.device,
        fingerprint.inode,
        fingerprint.mode,
        fingerprint.modified_ns,
        fingerprint.links,
    )
    if (
        imported_identity != _EXECUTING_BUILDER_IDENTITY
        or fingerprint.sha256 != _EXECUTING_BUILDER_SHA256
        or payload != _EXECUTING_BUILDER_PAYLOAD
    ):
        _fail("executing release builder source changed since module import")
    return TrustedModuleSource(
        name="build_public_release",
        path=expected_path,
        payload=payload,
        fingerprint=fingerprint,
    )


def _assert_executing_builder_selected(
    binding: TrustedModuleSource,
    selected: Sequence[SelectedFile],
) -> None:
    matches = [
        item for item in selected if item.relative == "code/build_public_release.py"
    ]
    if len(matches) != 1:
        _fail("selected release content does not bind exactly one release builder")
    item = matches[0]
    if item.source.resolve(strict=True) != binding.path:
        _fail("selected release builder path differs from the executing builder")
    _assert_fingerprint_matches(
        item,
        binding.fingerprint,
        label="selected release builder",
    )
    if item.payload != binding.payload:
        _fail("selected release builder bytes differ from the executing builder")


def _assert_fingerprint_matches(
    item: SelectedFile, observed: FileFingerprint, *, label: str
) -> None:
    expected = (
        item.bytes,
        item.sha256,
        item.device,
        item.inode,
        item.mode,
        item.modified_ns,
        item.links,
    )
    actual = (
        observed.bytes,
        observed.sha256,
        observed.device,
        observed.inode,
        observed.mode,
        observed.modified_ns,
        observed.links,
    )
    if actual != expected:
        _fail(f"{label} no longer matches its validated file binding")


def _git_bytes(
    root: Path,
    *args: str,
    input_bytes: bytes | None = None,
    label: str,
) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except (OSError, ValueError) as exc:
        raise ReleaseValidationError(f"{label} could not invoke Git") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ReleaseValidationError(
            f"{label} failed with Git exit {completed.returncode}: {detail}"
        )
    return completed.stdout


def _git_text(
    root: Path,
    *args: str,
    input_bytes: bytes | None = None,
    label: str,
) -> str:
    try:
        return (
            _git_bytes(
                root,
                *args,
                input_bytes=input_bytes,
                label=label,
            )
            .decode("utf-8")
            .strip()
        )
    except UnicodeDecodeError as exc:
        raise ReleaseValidationError(f"{label} returned non-UTF-8 Git output") from exc


def _gatdep_relative_path(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        _fail(f"{field} must be a non-empty POSIX relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        _fail(f"{field} is not a normalized relative path: {value!r}")
    if any("\n" in part or "\r" in part for part in pure.parts):
        _fail(f"{field} contains a line break: {value!r}")
    if pure.parts[0].casefold() == ".git":
        _fail(f"{field} may not enter Git metadata")
    return pure.as_posix()


def _safe_gatdep_file(root: Path, relative: str, *, label: str) -> Path:
    lexical_root = _assert_no_link_or_reparse_components(root, label="GATDep root")
    resolved_root = lexical_root.resolve(strict=True)
    candidate = lexical_root.joinpath(*PurePosixPath(relative).parts)
    try:
        _assert_no_link_or_reparse_components(candidate, label=label)
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
        metadata = candidate.stat(follow_symlinks=False)
    except (OSError, ValueError) as exc:
        raise ReleaseValidationError(
            f"{label} escapes the GATDep root or is missing: {relative}"
        ) from exc
    if _is_link_or_reparse(candidate) or not stat.S_ISREG(metadata.st_mode):
        _fail(f"{label} is linked or not a regular file: {relative}")
    return candidate


def _gatdep_git_state(root: Path) -> tuple[str, str]:
    head = _git_text(root, "rev-parse", "HEAD", label="GATDep HEAD")
    tree = _git_text(root, "rev-parse", "HEAD^{tree}", label="GATDep tree")
    if not re.fullmatch(r"[0-9a-f]{40,64}", head) or not re.fullmatch(
        r"[0-9a-f]{40,64}", tree
    ):
        _fail("GATDep HEAD or tree object identifier is invalid")
    dirty = _git_text(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
        label="GATDep tracked status",
    )
    if dirty:
        _fail("GATDep tracked worktree is not clean")
    return head, tree


def _bind_gatdep_files(
    root: Path,
    paths: Sequence[str],
    *,
    head: str,
) -> tuple[GatDepOfficialFileBinding, ...]:
    bindings: list[GatDepOfficialFileBinding] = []
    seen: set[str] = set()
    for index, raw_relative in enumerate(paths):
        relative = _gatdep_relative_path(
            raw_relative,
            field=f"GATDep snapshot path[{index}]",
        )
        folded = relative.casefold()
        if folded in seen:
            _fail(f"GATDep snapshot paths collide by case: {relative}")
        seen.add(folded)
        path = _safe_gatdep_file(
            root,
            relative,
            label=f"GATDep input {relative}",
        )
        payload, fingerprint = _read_stable_bytes(
            path,
            label=f"GATDep input {relative}",
        )
        object_id = _git_text(
            root,
            "rev-parse",
            f"{head}:{relative}",
            label=f"GATDep Git object {relative}",
        )
        if not re.fullmatch(r"[0-9a-f]{40,64}", object_id):
            _fail(f"GATDep Git object identifier is invalid: {relative}")
        object_type = _git_text(
            root,
            "cat-file",
            "-t",
            object_id,
            label=f"GATDep Git object type {relative}",
        )
        if object_type != "blob":
            _fail(f"GATDep Git object is not a file blob: {relative}")
        worktree_object = _git_text(
            root,
            "hash-object",
            "--no-filters",
            "--stdin",
            input_bytes=payload,
            label=f"GATDep worktree object {relative}",
        )
        if worktree_object != object_id:
            _fail(f"GATDep worktree file differs from HEAD: {relative}")
        bindings.append(
            GatDepOfficialFileBinding(
                relative=relative,
                payload=payload,
                fingerprint=fingerprint,
                git_object_id=object_id,
            )
        )
    return tuple(bindings)


def _assert_gatdep_source_unchanged(
    root: Path,
    *,
    identity: DirectoryIdentity,
    component_identities: PathComponentIdentities,
    head: str,
    tree: str,
    bindings: Sequence[GatDepOfficialFileBinding],
) -> None:
    _assert_directory_identity(root, identity, label="GATDep official root")
    if (
        _path_component_identities(root, label="GATDep official root")
        != component_identities
    ):
        _fail("GATDep official root path components changed after binding")
    observed_head, observed_tree = _gatdep_git_state(root)
    if observed_head != head or observed_tree != tree:
        _fail("GATDep official Git identity changed after binding")
    for binding in bindings:
        path = _safe_gatdep_file(
            root,
            binding.relative,
            label=f"GATDep input {binding.relative}",
        )
        payload, fingerprint = _read_stable_bytes(
            path,
            label=f"GATDep input {binding.relative}",
        )
        if payload != binding.payload or fingerprint != binding.fingerprint:
            _fail(f"GATDep input changed after binding: {binding.relative}")


def _copy_git_object(
    source_root: Path,
    snapshot_root: Path,
    object_id: str,
) -> None:
    object_type = _git_text(
        source_root,
        "cat-file",
        "-t",
        object_id,
        label=f"GATDep source object type {object_id}",
    )
    if object_type not in {"blob", "tree", "commit", "tag"}:
        _fail(f"unsupported GATDep Git object type: {object_type}")
    payload = _git_bytes(
        source_root,
        "cat-file",
        object_type,
        object_id,
        label=f"GATDep source object {object_id}",
    )
    copied = _git_text(
        snapshot_root,
        "hash-object",
        "-w",
        "-t",
        object_type,
        "--stdin",
        input_bytes=payload,
        label=f"GATDep snapshot object {object_id}",
    )
    if copied != object_id:
        _fail(f"GATDep Git object changed while copied: {object_id}")


def _gatdep_tree_object_ids(root: Path, head: str, root_tree: str) -> set[str]:
    payload = _git_bytes(
        root,
        "ls-tree",
        "-r",
        "-t",
        "-z",
        head,
        label="GATDep recursive tree inventory",
    )
    object_ids = {head, root_tree}
    for entry in payload.split(b"\x00"):
        if not entry:
            continue
        metadata, separator, _ = entry.partition(b"\t")
        fields = metadata.split()
        if not separator or len(fields) != 3:
            _fail("GATDep recursive tree inventory is malformed")
        if fields[1] == b"tree":
            try:
                object_id = fields[2].decode("ascii")
            except UnicodeDecodeError as exc:
                raise ReleaseValidationError(
                    "GATDep tree object identifier is not ASCII"
                ) from exc
            if not re.fullmatch(r"[0-9a-f]{40,64}", object_id):
                _fail("GATDep tree object identifier is invalid")
            object_ids.add(object_id)
    return object_ids


def _set_files_read_only(root: Path, paths: Sequence[str]) -> None:
    for relative in paths:
        path = root.joinpath(*PurePosixPath(relative).parts)
        mode = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
        path.chmod(mode & ~0o222)


def _make_tree_writable(root: Path) -> None:
    try:
        entries = sorted(
            root.rglob("*"), key=lambda path: len(path.parts), reverse=True
        )
    except OSError:
        return
    for path in entries:
        try:
            mode = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
            path.chmod(mode | stat.S_IWUSR)
        except OSError:
            continue
    try:
        mode = stat.S_IMODE(root.stat(follow_symlinks=False).st_mode)
        root.chmod(mode | stat.S_IWUSR)
    except OSError:
        pass


@contextmanager
def _stable_gatdep_official_snapshot(
    root: Path,
    required_paths: Sequence[str],
) -> Iterator[Path]:
    lexical_root = _assert_no_link_or_reparse_components(
        root,
        label="GATDep official root",
    )
    lexical_root, root_identity = _bind_directory(
        lexical_root,
        label="GATDep official root",
    )
    component_identities = _path_component_identities(
        lexical_root,
        label="GATDep official root",
    )
    head, tree = _gatdep_git_state(lexical_root)
    normalized_paths = tuple(
        sorted(
            _gatdep_relative_path(path, field="GATDep required path")
            for path in required_paths
        )
    )
    if not normalized_paths:
        _fail("GATDep stable snapshot has no required inputs")
    bindings = _bind_gatdep_files(lexical_root, normalized_paths, head=head)
    _assert_gatdep_source_unchanged(
        lexical_root,
        identity=root_identity,
        component_identities=component_identities,
        head=head,
        tree=tree,
        bindings=bindings,
    )

    snapshot_parent = Path(tempfile.mkdtemp(prefix="depplm-gatdep-snapshot-parent-"))
    snapshot_parent_handle: BoundDirectoryHandle | None = None
    snapshot_handle: BoundDirectoryHandle | None = None
    snapshot_parent_handles: list[BoundDirectoryHandle] = []
    snapshot_handles: list[BoundDirectoryHandle] = []
    try:
        snapshot_parent, provisional_parent_identity = _capture_new_directory_identity(
            snapshot_parent,
            label="GATDep stable snapshot parent",
        )
        snapshot_parent, snapshot_parent_identity = _bind_directory(
            snapshot_parent,
            label="GATDep stable snapshot parent",
        )
        if snapshot_parent_identity != provisional_parent_identity:
            _fail("GATDep stable snapshot parent changed during initial binding")
        snapshot_parent_handle = _open_bound_directory_handle(
            snapshot_parent,
            snapshot_parent_identity,
            label="GATDep stable snapshot parent",
            owned_handles=snapshot_parent_handles,
        )
        if (
            os.name != "nt"
            and stat.S_IMODE(os.fstat(snapshot_parent_handle.descriptor).st_mode)
            != 0o700
        ):
            _fail("GATDep stable snapshot parent is not private")
        snapshot_handle = _create_bound_child_directory(
            snapshot_parent_handle,
            "snapshot",
            label="GATDep stable snapshot",
            owned_handles=snapshot_handles,
        )
    except BaseException as primary:
        primary_traceback = primary.__traceback__
        cleanup_parent = snapshot_parent_handle or (
            snapshot_parent_handles[-1] if snapshot_parent_handles else None
        )
        cleanup_snapshot = snapshot_handle or (
            snapshot_handles[-1] if snapshot_handles else None
        )
        if cleanup_snapshot is not None and cleanup_parent is not None:
            try:
                _quarantine_bound_directory(
                    cleanup_snapshot.path,
                    cleanup_snapshot.identity,
                    label="GATDep stable snapshot",
                    parent_handle=(
                        cleanup_parent if cleanup_parent.descriptor >= 0 else None
                    ),
                    child_handle=(
                        cleanup_snapshot if cleanup_snapshot.descriptor >= 0 else None
                    ),
                )
            except BaseException:
                pass
        _close_owned_directory_handles(
            (*snapshot_handles, *snapshot_parent_handles),
            active_exception=primary,
        )
        raise primary.with_traceback(primary_traceback)
    if snapshot_parent_handle is None or snapshot_handle is None:
        _fail("GATDep stable snapshot handles were not initialized")
    snapshot = snapshot_handle.path
    snapshot_identity = snapshot_handle.identity
    snapshot_bindings: dict[str, FileFingerprint] = {}
    try:
        if os.name == "nt":
            # The Windows directory handle carries DELETE access so cleanup can be
            # object-bound. Keeping that child handle open prevents Git for Windows
            # from entering the worktree; the still-open parent handle is sufficient
            # to reopen this exact child relatively at quarantine time.
            snapshot_handle.close()
            snapshot_handle = None
        _git_text(snapshot, "init", "--quiet", label="GATDep snapshot initialization")
        _git_text(
            snapshot,
            "config",
            "core.autocrlf",
            "false",
            label="GATDep snapshot configuration",
        )
        _git_text(
            snapshot,
            "config",
            "core.filemode",
            "false",
            label="GATDep snapshot configuration",
        )
        object_ids = _gatdep_tree_object_ids(lexical_root, head, tree)
        object_ids.update(binding.git_object_id for binding in bindings)
        for object_id in sorted(object_ids):
            _copy_git_object(lexical_root, snapshot, object_id)
        _git_text(
            snapshot,
            "update-ref",
            "refs/heads/snapshot",
            head,
            label="GATDep snapshot ref",
        )
        _git_text(
            snapshot,
            "symbolic-ref",
            "HEAD",
            "refs/heads/snapshot",
            label="GATDep snapshot HEAD",
        )
        _git_text(
            snapshot,
            "config",
            "core.sparseCheckout",
            "true",
            label="GATDep sparse checkout configuration",
        )
        _git_text(
            snapshot,
            "config",
            "core.sparseCheckoutCone",
            "false",
            label="GATDep sparse checkout configuration",
        )
        sparse_file = snapshot / ".git" / "info" / "sparse-checkout"
        sparse_file.parent.mkdir(parents=True, exist_ok=True)
        sparse_file.write_text(
            "".join(f"/{binding.relative}\n" for binding in bindings),
            encoding="utf-8",
            newline="\n",
        )
        _git_text(
            snapshot,
            "read-tree",
            "-mu",
            "HEAD",
            label="GATDep sparse checkout",
        )
        for binding in bindings:
            snapshot_path = _safe_gatdep_file(
                snapshot,
                binding.relative,
                label=f"GATDep snapshot input {binding.relative}",
            )
            payload, _ = _read_stable_bytes(
                snapshot_path,
                label=f"GATDep snapshot input {binding.relative}",
            )
            if payload != binding.payload:
                _fail(f"GATDep snapshot bytes differ: {binding.relative}")
        _set_files_read_only(snapshot, normalized_paths)
        for binding in bindings:
            snapshot_bindings[binding.relative] = _fingerprint_file(
                snapshot.joinpath(*PurePosixPath(binding.relative).parts),
                label=f"GATDep read-only snapshot input {binding.relative}",
            )
        if _gatdep_git_state(snapshot) != (head, tree):
            _fail("GATDep stable snapshot Git identity differs")
        _assert_gatdep_source_unchanged(
            lexical_root,
            identity=root_identity,
            component_identities=component_identities,
            head=head,
            tree=tree,
            bindings=bindings,
        )
        body_exception: BaseException | None = None
        body_traceback = None
        try:
            yield snapshot
        except BaseException as exc:
            body_exception = exc
            body_traceback = exc.__traceback__
        validation_exception: BaseException | None = None
        try:
            _assert_directory_identity(
                snapshot,
                snapshot_identity,
                label="GATDep stable snapshot",
            )
            if _gatdep_git_state(snapshot) != (head, tree):
                _fail("GATDep stable snapshot Git identity changed during replay")
            for binding in bindings:
                observed = _fingerprint_file(
                    snapshot.joinpath(*PurePosixPath(binding.relative).parts),
                    label=f"GATDep read-only snapshot input {binding.relative}",
                )
                if observed != snapshot_bindings[binding.relative]:
                    _fail(
                        f"GATDep stable snapshot changed during replay: "
                        f"{binding.relative}"
                    )
            _assert_gatdep_source_unchanged(
                lexical_root,
                identity=root_identity,
                component_identities=component_identities,
                head=head,
                tree=tree,
                bindings=bindings,
            )
        except BaseException as exc:
            validation_exception = exc
        if body_exception is not None:
            raise body_exception.with_traceback(body_traceback)
        if validation_exception is not None:
            raise validation_exception
    finally:
        active_exception = sys.exc_info()[1]
        quarantine_exception: BaseException | None = None
        quarantined = False
        try:
            quarantined = _quarantine_bound_directory(
                snapshot,
                snapshot_identity,
                label="GATDep stable snapshot",
                parent_handle=snapshot_parent_handle,
                child_handle=(
                    snapshot_handle
                    if snapshot_handle is not None and snapshot_handle.descriptor >= 0
                    else None
                ),
            )
        except BaseException as exc:
            quarantine_exception = exc
        close_exception: BaseException | None = None
        try:
            _close_owned_directory_handles(
                (*snapshot_handles, *snapshot_parent_handles),
                active_exception=active_exception,
            )
        except BaseException as exc:
            close_exception = exc
        if active_exception is None:
            if quarantine_exception is not None:
                raise quarantine_exception
            if close_exception is not None:
                raise close_exception
            if not quarantined:
                _fail("GATDep stable snapshot could not be quarantined safely")


def _gatdep_snapshot_required_paths(
    common_module: ModuleType,
    official_module: ModuleType,
) -> tuple[str, ...]:
    sources: tuple[tuple[str, Any], ...] = (
        ("common verifier OFFICIAL_INPUT_PATHS", common_module.OFFICIAL_INPUT_PATHS),
        (
            "common verifier EXPECTED_BLOBS",
            common_module.gatdep.EXPECTED_BLOBS,
        ),
        (
            "official verifier REQUIRED_OFFICIAL_INPUT_PATHS",
            official_module.REQUIRED_OFFICIAL_INPUT_PATHS,
        ),
        ("official verifier CRITICAL_BLOBS", official_module.CRITICAL_BLOBS),
    )
    required: dict[str, str] = {}
    for label, value in sources:
        raw_paths: Any
        if isinstance(value, Mapping):
            raw_paths = value.keys()
        elif isinstance(value, (tuple, list)):
            raw_paths = value
        else:
            _fail(f"{label} is not a path inventory")
        for raw_path in raw_paths:
            if not isinstance(raw_path, str):
                _fail(f"{label} contains a non-string path")
            normalized = _gatdep_relative_path(raw_path, field=label)
            folded = normalized.casefold()
            previous = required.get(folded)
            if previous is not None and previous != normalized:
                _fail(
                    "GATDep verifier inputs collide under case folding: "
                    f"{previous!r} and {normalized!r}"
                )
            required[folded] = normalized
    return tuple(sorted(required.values()))


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"JSON contains a duplicate object key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> NoReturn:
    _fail(f"JSON contains a non-finite numeric constant: {value}")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        _fail(f"JSON contains a non-finite numeric value: {value}")
    return parsed


def _parse_json_bytes(
    payload: bytes,
    *,
    label: str,
    path: Path | None = None,
) -> dict[str, Any]:
    location = f": {path}" if path is not None else ""
    try:
        text = payload.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_pairs,
            parse_constant=_reject_nonfinite_json_constant,
            parse_float=_parse_finite_json_float,
        )
    except ReleaseValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseValidationError(
            f"{label} is unreadable or invalid strict JSON{location}"
        ) from exc
    if not isinstance(value, dict):
        _fail(f"{label} must contain a JSON object{location}")
    return value


def _load_json_with_fingerprint(
    path: Path,
    *,
    label: str,
    expected: SelectedFile | Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], FileFingerprint]:
    payload, fingerprint = _read_stable_bytes(path, label=label)
    if isinstance(expected, SelectedFile):
        _assert_fingerprint_matches(expected, fingerprint, label=label)
    elif expected is not None:
        sha = expected.get("sha256")
        size = expected.get("bytes")
        if (
            not isinstance(sha, str)
            or not SHA256_RE.fullmatch(sha)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or fingerprint.sha256 != sha
            or fingerprint.bytes != size
        ):
            _fail(f"{label} no longer matches its bound file")
    return _parse_json_bytes(payload, label=label, path=path), fingerprint


def _fail_public_local_path(
    *,
    relative: str,
    location: int | str,
    decoded_json: bool = False,
) -> NoReturn:
    if decoded_json:
        _fail(
            "public JSON contains a local absolute path "
            "(decoded local absolute path) "
            f"at {relative}:{location}; producer output must use a "
            "workspace-relative path"
        )
    _fail(
        "public text/JSON contains a local absolute path "
        f"at {relative}:{location}; producer output must use a "
        "workspace-relative path"
    )


def _mask_registered_root_relative_uri(
    line: str,
    *,
    relative: str,
) -> str:
    registrations = _REGISTERED_ROOT_RELATIVE_URI_LINES.get(relative, ())
    if not any(pattern.fullmatch(line) for pattern in registrations):
        return line
    return " " * len(line)


def _mask_registered_portable_system_path(
    line: str,
    *,
    relative: str,
) -> str:
    registrations = _REGISTERED_PORTABLE_SYSTEM_PATH_LINES.get(relative, ())
    if not any(pattern.fullmatch(line) for pattern in registrations):
        return line
    return " " * len(line)


def _mask_registered_source_fixture_path(
    line: str,
    *,
    relative: str,
) -> str:
    registrations = _REGISTERED_SOURCE_FIXTURE_PATH_LINES.get(relative, ())
    if not any(pattern.fullmatch(line) for pattern in registrations):
        return line
    return " " * len(line)


def _mask_registered_string_suffix_line(
    line: str,
    *,
    relative: str,
    line_number: int,
) -> str:
    registration = _REGISTERED_STRING_SUFFIX_LINES.get(relative, {}).get(line_number)
    if registration is None or registration.fullmatch(line) is None:
        return line
    return " " * len(line)


def _mask_registered_local_path_source_line(
    line: str,
    *,
    relative: str,
    line_number: int,
) -> str:
    expected = _REGISTERED_LOCAL_PATH_SOURCE_LINE_SHA256.get(relative, {}).get(
        line_number
    )
    if expected is None:
        return line
    observed = hashlib.sha256(line.encode("utf-8")).hexdigest()
    return " " * len(line) if observed == expected else line


def _bounded_percent_decoded_layers(
    text: str,
    *,
    relative: str,
    location: int | str,
    decoded_json: bool,
) -> tuple[str, ...]:
    layers = [text]
    for _ in range(_MAX_AUTHORITY_URL_DECODE_PASSES):
        decoded = urllib.parse.unquote(layers[-1])
        if decoded == layers[-1]:
            break
        layers.append(decoded)
    if urllib.parse.unquote(layers[-1]) != layers[-1]:
        _fail_public_local_path(
            relative=relative,
            location=location,
            decoded_json=decoded_json,
        )
    return tuple(layers)


def _validate_authority_uri_token(
    core: str,
    suffix: str,
    *,
    relative: str,
    location: int | str,
    decoded_json: bool,
) -> None:
    scheme = core.partition("://")[0]
    if len(scheme) == 1 and scheme.isascii() and scheme.isalpha():
        _fail_public_local_path(
            relative=relative,
            location=location,
            decoded_json=decoded_json,
        )
    for decoded in _bounded_percent_decoded_layers(
        core,
        relative=relative,
        location=location,
        decoded_json=decoded_json,
    ):
        after_scheme = decoded.partition("://")[2]
        query_start = min(
            (index for index in (decoded.find("?"), decoded.find("#")) if index >= 0),
            default=len(decoded),
        )
        query_or_fragment = decoded[query_start:]
        has_embedded_local_path = any(
            pattern.search(query_or_fragment)
            for pattern in (
                _FILE_URI_PATTERN,
                _WINDOWS_EXTENDED_NAMESPACE_PATTERN,
                _FORWARD_WINDOWS_EXTENDED_NAMESPACE_PATTERN,
                _FORWARD_UNC_PATTERN,
                _GENERAL_POSIX_ABSOLUTE_PATH_PATTERN,
                LOCAL_ABSOLUTE_PATH_PATTERNS[0],
                _TEXT_UNC_PATTERN,
            )
        )
        has_embedded_windows_drive = bool(
            _URL_EMBEDDED_WINDOWS_DRIVE_PATTERN.search(after_scheme)
        )
        has_embedded_windows_namespace = bool(
            _URL_EMBEDDED_WINDOWS_NAMESPACE_PATTERN.search(after_scheme)
        )
        has_embedded_unc = bool(
            _TEXT_UNC_PATTERN.search(after_scheme)
            or _FORWARD_UNC_PATTERN.search(after_scheme)
        )
        has_embedded_file_uri = bool(_FILE_URI_PATTERN.search(after_scheme))
        has_adjacent_local_path = bool(re.search(r"[,;)\]]/[A-Za-z0-9._~+-]", decoded))
        parsed = urllib.parse.urlsplit(decoded)
        has_encoded_posix_path = bool(
            re.search(r"/{2}(?=[A-Za-z0-9._~+-])", parsed.path)
        )
        has_local_home_template = (
            parsed.path.rstrip(_POSIX_SEPARATOR).casefold() == _POSIX_SEPARATOR + "home"
            and decoded.endswith(_POSIX_SEPARATOR)
            and suffix.startswith("{")
        )
        if (
            has_embedded_local_path
            or has_embedded_windows_drive
            or has_embedded_windows_namespace
            or has_embedded_unc
            or has_embedded_file_uri
            or has_adjacent_local_path
            or has_encoded_posix_path
            or has_local_home_template
        ):
            _fail_public_local_path(
                relative=relative,
                location=location,
                decoded_json=decoded_json,
            )


def _reject_unsafe_authority_uris(
    text: str,
    *,
    relative: str,
    location: int | str,
    decoded_json: bool,
) -> None:
    """Validate authority URLs found at every bounded whole-scalar decode layer."""
    for layer in _bounded_percent_decoded_layers(
        text,
        relative=relative,
        location=location,
        decoded_json=decoded_json,
    ):
        _mask_authority_uris_in_layer(
            layer,
            relative=relative,
            location=location,
            decoded_json=decoded_json,
        )


def _mask_authority_uris_in_layer(
    text: str,
    *,
    relative: str,
    location: int | str,
    decoded_json: bool,
) -> str:
    """Validate and mask authority URLs in exactly one scalar decode layer."""
    rendered = list(text)
    for match in _AUTHORITY_URI_PATTERN.finditer(text):
        token = match.group(0)
        core = token.rstrip(".,;)]")
        if core.endswith("}") and "{" not in core:
            core = core[:-1]
        _validate_authority_uri_token(
            core,
            text[match.end() :],
            relative=relative,
            location=location,
            decoded_json=decoded_json,
        )
        rendered[match.start() : match.end()] = " " * len(token)
    return "".join(rendered)


def _mask_safe_authority_uris(
    text: str,
    *,
    relative: str,
    location: int | str,
    decoded_json: bool = False,
) -> str:
    """Mask URLs only after rejecting local-path payloads hidden inside them."""
    _reject_unsafe_authority_uris(
        text,
        relative=relative,
        location=location,
        decoded_json=decoded_json,
    )
    return _mask_authority_uris_in_layer(
        text,
        relative=relative,
        location=location,
        decoded_json=decoded_json,
    )


def _scan_public_scalar_text_for_local_paths(
    value: str,
    *,
    relative: str,
    location: int | str,
    patterns: Sequence[re.Pattern[str]],
    decoded_json: bool = False,
    raw_line_number: int | None = None,
    suffix: str | None = None,
    category: str | None = None,
) -> None:
    """Check one text scalar without recursively invoking the file scanner."""
    layers = _bounded_percent_decoded_layers(
        value,
        relative=relative,
        location=location,
        decoded_json=decoded_json,
    )
    for layer_index, layer in enumerate(layers):
        scan_text = _mask_authority_uris_in_layer(
            layer,
            relative=relative,
            location=location,
            decoded_json=decoded_json,
        )
        if raw_line_number is not None and layer_index == 0:
            scan_text = _mask_registered_root_relative_uri(
                scan_text,
                relative=relative,
            )
            scan_text = _mask_registered_portable_system_path(
                scan_text,
                relative=relative,
            )
            scan_text = _mask_registered_source_fixture_path(
                scan_text,
                relative=relative,
            )
            scan_text = _mask_registered_string_suffix_line(
                scan_text,
                relative=relative,
                line_number=raw_line_number,
            )
            scan_text = _mask_registered_local_path_source_line(
                scan_text,
                relative=relative,
                line_number=raw_line_number,
            )
            if suffix in {".tex", ".cls", ".sty", ".bst"} and ("pdfmark" in scan_text):
                scan_text = _TEX_PDFMARK_NAME_PATTERN.sub(
                    lambda match: " " * len(match.group(0)),
                    scan_text,
                )
            if category in {"source", "test"} and (
                "cmd" in scan_text and "mklink" in scan_text
            ):
                scan_text = _WINDOWS_CMD_MKLINK_SWITCH_PATTERN.sub(
                    lambda match: " " * len(match.group(0)),
                    scan_text,
                )
        if any(pattern.search(scan_text) for pattern in patterns):
            _fail_public_local_path(
                relative=relative,
                location=location,
                decoded_json=(
                    decoded_json
                    or (
                        raw_line_number is not None
                        and suffix == ".json"
                        and layer_index > 0
                    )
                ),
            )


def _scan_public_text_for_local_paths(
    path: Path,
    *,
    relative: str,
    category: str,
    payload: bytes | None = None,
) -> None:
    if (
        category not in PUBLIC_TEXT_CATEGORIES
        or path.suffix.casefold() not in PUBLIC_TEXT_SUFFIXES
    ):
        return
    try:
        if payload is None:
            payload, _ = _read_stable_bytes(path, label=f"public text file {relative}")
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReleaseValidationError(
            f"public text file is not readable UTF-8: {relative}"
        ) from exc
    patterns = (
        SOURCE_LOCAL_ABSOLUTE_PATH_PATTERNS
        if category in {"source", "test"}
        else TEXT_LOCAL_ABSOLUTE_PATH_PATTERNS
    )
    lines = text.splitlines()
    suffix = path.suffix.casefold()
    for line_number, line in enumerate(lines, start=1):
        if line_number == 1 and line.startswith("#!"):
            if line not in PORTABLE_SHEBANGS:
                _fail(
                    "public source/test file has a non-portable shebang "
                    f"at {relative}:{line_number}"
                )
            continue
        _scan_public_scalar_text_for_local_paths(
            line,
            relative=relative,
            location=line_number,
            patterns=patterns,
            raw_line_number=line_number,
            suffix=suffix,
            category=category,
        )
    if suffix != ".json":
        return
    decoded = _parse_json_bytes(
        payload,
        label="public JSON file",
        path=path,
    )

    def inspect(value: Any, *, location: str) -> None:
        if isinstance(value, str):
            _scan_public_scalar_text_for_local_paths(
                value,
                relative=relative,
                location=location,
                patterns=LOCAL_ABSOLUTE_PATH_PATTERNS,
                decoded_json=True,
            )
            return
        if isinstance(value, Mapping):
            for key, nested in value.items():
                inspect(key, location=f"{location}.<key>")
                inspect(nested, location=f"{location}.{key}")
            return
        if isinstance(value, list):
            for index, nested in enumerate(value):
                inspect(nested, location=f"{location}[{index}]")

    inspect(decoded, location="$")


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    return _load_json_with_fingerprint(path, label=label)[0]


def _relative_path(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        _fail(f"{field} must be a non-empty POSIX relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        _fail(f"{field} is not a normalized relative path: {value!r}")
    if any(":" in part or part.startswith(".") for part in pure.parts):
        _fail(f"{field} contains a forbidden path component: {value!r}")
    if any(part.casefold() in FORBIDDEN_PARTS for part in pure.parts):
        _fail(f"{field} enters a forbidden directory: {value!r}")
    if pure.suffix.casefold() in FORBIDDEN_SUFFIXES:
        _fail(f"{field} has a forbidden credential-like suffix: {value!r}")
    filename = pure.name.casefold()
    if filename in FORBIDDEN_SSH_NAMES or FORBIDDEN_FILE_RE.fullmatch(filename):
        _fail(f"{field} has a forbidden credential-like filename: {value!r}")
    if (
        filename in FORBIDDEN_SENSITIVE_DATA_NAMES
        or FORBIDDEN_RAW_FILE_RE.fullmatch(filename)
        or (filename == "model.csv" and "depmap" in map(str.casefold, pure.parts))
    ):
        _fail(f"{field} has a forbidden raw/source-data filename: {value!r}")
    return pure.as_posix()


def _safe_workspace_file(root: Path, relative: str, *, field: str) -> Path:
    try:
        lexical_root = _assert_no_link_or_reparse_components(
            root, label="workspace root"
        )
        resolved_root = lexical_root.resolve(strict=True)
        candidate = resolved_root.joinpath(*PurePosixPath(relative).parts)
        _assert_no_link_or_reparse_components(candidate, label=field)
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise ReleaseValidationError(
            f"{field} escapes the workspace or is missing: {relative}"
        ) from exc
    if not candidate.is_file() or not stat.S_ISREG(
        candidate.stat(follow_symlinks=False).st_mode
    ):
        _fail(f"{field} is not a regular file: {relative}")
    return candidate


def _path_allowed_for_category(relative: str, category: str) -> bool:
    if category == "source":
        return (
            relative.startswith(("code/", "tools/"))
            or relative
            == "paper/nar_major_revision/figures/gen_fig_corrected_campaign.py"
        ) and relative.endswith(".py")
    if category == "test":
        return relative.startswith("tests/") and relative.endswith(".py")
    if category == "manuscript":
        return relative in REVIEWED_PUBLICATION_FILES["manuscript"]
    if category == "documentation":
        return relative in REVIEWED_PUBLICATION_FILES["documentation"]
    if category in {"evidence", "prediction"}:
        return relative.startswith("results/")
    if category == "provenance":
        return relative in ESM_PUBLIC_FILES
    return False


def _meaningful_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{field} must be a non-empty string")
    normalized = value.strip()
    if PLACEHOLDER_RE.search(normalized):
        _fail(f"{field} contains placeholder text")
    return normalized


def _substantive_text(value: Any, *, field: str, minimum: int = 3) -> str:
    normalized = _meaningful_text(value, field=field)
    if len(normalized) < minimum:
        _fail(f"{field} must be substantive (at least {minimum} characters)")
    return normalized


def _orcid_checksum_valid(orcid: str) -> bool:
    compact = orcid.replace("-", "")
    total = 0
    for character in compact[:15]:
        total = (total + int(character)) * 2
    remainder = (12 - total % 11) % 11
    expected = "X" if remainder == 10 else str(remainder)
    return compact[-1] == expected


def _validate_png_bytes(payload: bytes, *, relative: str) -> None:
    signature = b"\x89PNG\r\n\x1a\n"
    if not payload.startswith(signature):
        _fail(f"reviewed publication image is not a PNG: {relative}")
    offset = len(signature)
    chunk_types: list[bytes] = []
    while offset + 12 <= len(payload):
        length = int.from_bytes(payload[offset : offset + 4], "big")
        chunk_type = payload[offset + 4 : offset + 8]
        end = offset + 12 + length
        if end > len(payload):
            _fail(f"reviewed publication PNG is truncated: {relative}")
        chunk_data = payload[offset + 8 : offset + 8 + length]
        expected_crc = int.from_bytes(payload[offset + 8 + length : end], "big")
        if zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF != expected_crc:
            _fail(f"reviewed publication PNG has an invalid CRC: {relative}")
        chunk_types.append(chunk_type)
        offset = end
        if chunk_type == b"IEND":
            break
    if (
        not chunk_types
        or chunk_types[0] != b"IHDR"
        or b"IDAT" not in chunk_types
        or chunk_types[-1] != b"IEND"
        or offset != len(payload)
    ):
        _fail(f"reviewed publication PNG structure is invalid: {relative}")


def _validate_pdf_bytes(payload: bytes, *, relative: str) -> None:
    if relative not in REVIEWED_PDF_FILES:
        _fail(f"reviewed publication PDF is not in the exact PDF allowlist: {relative}")
    if not re.match(rb"%PDF-[12]\.[0-9]\r?\n", payload[:16]):
        _fail(f"reviewed publication PDF has an invalid header: {relative}")
    if b" obj" not in payload or b"startxref" not in payload:
        _fail(f"reviewed publication PDF structure is incomplete: {relative}")
    if re.search(rb"%%EOF\s*\Z", payload) is None:
        _fail(f"reviewed publication PDF has no terminal EOF marker: {relative}")


def _validate_publication_provenance_record(
    value: Any,
    *,
    field: str,
    expected_path: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"path", "bytes", "sha256"}:
        _fail(f"{field} must contain exactly path/bytes/sha256")
    relative = _relative_path(value.get("path"), field=f"{field}.path")
    size = value.get("bytes")
    digest = value.get("sha256")
    if type(size) is not int or size < 0:
        _fail(f"{field}.bytes must be a non-negative integer")
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        _fail(f"{field}.sha256 must be a lowercase SHA-256 digest")
    if expected_path is not None and relative != expected_path:
        _fail(f"{field}.path must be {expected_path}")
    return {"path": relative, "bytes": size, "sha256": digest}


def _require_core_campaign_publication_path(
    value: str | PurePosixPath, *, field: str
) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        len(path.parts) != 5
        or path.parts[:3] != ("results", "corrected_v2", "campaigns")
        or path.name != "campaign.json"
    ):
        _fail(
            f"{field} must be the canonical "
            "results/corrected_v2/campaigns/<id>/campaign.json path"
        )
    return path


def _require_clinical_publication_path(
    value: str | PurePosixPath, *, filename: str, field: str
) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        len(path.parts) != 4
        or path.parts[:2] != ("results", "clinical_validation_v2")
        or path.name != filename
    ):
        _fail(
            f"{field} must be the canonical "
            f"results/clinical_validation_v2/<run>/{filename} path"
        )
    return path


def _validate_corrected_campaign_provenance_document(
    payload: bytes,
) -> dict[str, Any]:
    document = _parse_json_bytes(
        payload,
        label="corrected-campaign publication provenance",
    )
    if set(document) != {
        "schema",
        "mode",
        "clinical_evidence_loaded",
        "generator",
        "inputs",
        "outputs",
    }:
        _fail("corrected-campaign publication provenance fields are not exact")
    if document.get("schema") != CORRECTED_CAMPAIGN_PROVENANCE_SCHEMA:
        _fail("corrected-campaign publication provenance schema is invalid")
    mode = document.get("mode")
    if mode not in {"core_only", "full"}:
        _fail("corrected-campaign publication provenance mode is invalid")
    expected_clinical = mode == "full"
    if document.get("clinical_evidence_loaded") is not expected_clinical:
        _fail("corrected-campaign clinical-evidence mode flag is inconsistent")

    generator = _validate_publication_provenance_record(
        document.get("generator"),
        field="corrected-campaign generator",
        expected_path=CLINICAL_PUBLICATION_GENERATOR_SOURCE,
    )
    inputs = document.get("inputs")
    expected_input_names = (
        {"campaign", "clinical_completion", "clinical_results"}
        if expected_clinical
        else {"campaign"}
    )
    if not isinstance(inputs, dict) or set(inputs) != expected_input_names:
        _fail("corrected-campaign publication input inventory is not exact")
    normalized_inputs = {
        name: _validate_publication_provenance_record(
            record,
            field=f"corrected-campaign input {name}",
        )
        for name, record in inputs.items()
    }
    _require_core_campaign_publication_path(
        normalized_inputs["campaign"]["path"],
        field="corrected-campaign publication input",
    )
    if expected_clinical:
        completion_path = _require_clinical_publication_path(
            normalized_inputs["clinical_completion"]["path"],
            filename="completion.json",
            field="corrected-campaign clinical completion input",
        )
        results_path = _require_clinical_publication_path(
            normalized_inputs["clinical_results"]["path"],
            filename="results.json",
            field="corrected-campaign clinical results input",
        )
        if results_path.parent != completion_path.parent:
            _fail("corrected-campaign clinical input paths are noncanonical")

    expected_outputs = dict(CORRECTED_CAMPAIGN_CORE_OUTPUTS)
    if expected_clinical:
        expected_outputs["clinical_table"] = CORRECTED_CAMPAIGN_CLINICAL_TABLE
    outputs = document.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(expected_outputs):
        _fail("corrected-campaign publication output inventory is not exact")
    normalized_outputs = {
        name: _validate_publication_provenance_record(
            outputs[name],
            field=f"corrected-campaign output {name}",
            expected_path=expected_path,
        )
        for name, expected_path in expected_outputs.items()
    }
    canonical = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if payload != canonical:
        _fail("corrected-campaign publication provenance JSON is not canonical")
    return {
        **document,
        "generator": generator,
        "inputs": normalized_inputs,
        "outputs": normalized_outputs,
    }


class _WindowsKillOnCloseJob:
    """Own a Windows Job Object that kills every member when closed."""

    _CREATE_SUSPENDED = 0x00000004
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

    def __init__(self) -> None:
        from ctypes import wintypes

        class IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class BasicLimitInformation(ctypes.Structure):
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

        class ExtendedLimitInformation(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimitInformation),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)
        self._close_handle = kernel32.CloseHandle
        self._close_handle.argtypes = [wintypes.HANDLE]
        self._close_handle.restype = wintypes.BOOL
        create_job = kernel32.CreateJobObjectW
        create_job.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        create_job.restype = wintypes.HANDLE
        set_information = kernel32.SetInformationJobObject
        set_information.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        set_information.restype = wintypes.BOOL
        self._assign_process = kernel32.AssignProcessToJobObject
        self._assign_process.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self._assign_process.restype = wintypes.BOOL

        ntdll = getattr(ctypes, "WinDLL")("ntdll")
        self._resume_process = ntdll.NtResumeProcess
        self._resume_process.argtypes = [wintypes.HANDLE]
        self._resume_process.restype = wintypes.LONG
        self._rtl_status_to_error = ntdll.RtlNtStatusToDosError
        self._rtl_status_to_error.argtypes = [wintypes.LONG]
        self._rtl_status_to_error.restype = wintypes.ULONG

        handle = create_job(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        self._handle: Any | None = handle
        information = ExtendedLimitInformation()
        information.BasicLimitInformation.LimitFlags = (
            self._JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        if not set_information(
            handle,
            self._JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            error = ctypes.get_last_error()
            self.terminate()
            raise ctypes.WinError(error)

    def attach_and_resume(self, process: subprocess.Popen[bytes]) -> None:
        from ctypes import wintypes

        if self._handle is None:
            raise RuntimeError("Windows subprocess job is already closed")
        raw_process_handle = getattr(process, "_handle", None)
        if raw_process_handle is None:
            raise RuntimeError("Windows subprocess handle is unavailable")
        process_handle = wintypes.HANDLE(int(raw_process_handle))
        if not self._assign_process(self._handle, process_handle):
            raise ctypes.WinError(ctypes.get_last_error())
        status = int(self._resume_process(process_handle))
        if status < 0:
            error = int(self._rtl_status_to_error(status))
            raise OSError(error, ctypes.FormatError(error))

    def terminate(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        if not self._close_handle(handle):
            raise ctypes.WinError(ctypes.get_last_error())


class _OwnedBoundedPopen(subprocess.Popen[bytes]):
    """Register a created child before a caller can be interrupted on return."""

    def __init__(
        self,
        *args: Any,
        owned_processes: list[subprocess.Popen[bytes]],
        **kwargs: Any,
    ) -> None:
        try:
            super().__init__(*args, **kwargs)
        finally:
            if bool(getattr(self, "_child_created", False)):
                owned_processes.append(self)


class _BoundedSubprocessCancelled(Exception):
    """Internal request for the subprocess owner to stop and clean up."""


class _BoundedSubprocessCleanupError(RuntimeError):
    """Cleanup could not be confirmed within the bounded deadline."""


def _settle_attempted_thread_start(
    thread: threading.Thread,
    deadline: float,
) -> bool | None:
    """Resolve an interrupted Thread.start call without guessing from ident alone."""

    limbo = getattr(threading, "_limbo", None)
    limbo_lock = getattr(threading, "_active_limbo_lock", None)
    while thread.ident is None:
        if limbo is None or limbo_lock is None:
            in_startup_limbo: bool | None = None
        else:
            with limbo_lock:
                in_startup_limbo = thread in limbo
        if in_startup_limbo is False:
            return False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        started_event = getattr(thread, "_started", None)
        wait_for_start = getattr(started_event, "wait", None)
        if callable(wait_for_start):
            wait_for_start(min(remaining, 0.02))
        else:
            time.sleep(min(remaining, 0.02))
    return True


def _cleanup_unmanaged_subprocess(
    process: subprocess.Popen[bytes],
) -> list[str]:
    """Bound cleanup after a tree guard fails before worker threads start."""

    cleanup_failures: list[str] = []
    if process.poll() is None:
        try:
            process.kill()
        except OSError as exc:
            cleanup_failures.append(f"direct-process termination failed: {exc}")
    if process.poll() is None:
        try:
            process.wait(timeout=BOUNDED_SUBPROCESS_CLEANUP_TIMEOUT)
        except (OSError, subprocess.TimeoutExpired) as exc:
            cleanup_failures.append(f"direct-process wait failed: {exc}")
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            try:
                stream.close()
            except OSError as exc:
                cleanup_failures.append(f"subprocess pipe close failed: {exc}")
    return cleanup_failures


def _kill_posix_process_group(pid: int) -> None:
    killpg = getattr(os, "killpg", None)
    sigkill = getattr(signal, "SIGKILL", 9)
    if not callable(killpg):
        raise RuntimeError("POSIX process-group termination is unavailable")
    killpg(pid, sigkill)


def _run_bounded_binary_subprocess_owned(
    command: Sequence[str],
    *,
    input_payload: bytes,
    cwd: Path,
    env: Mapping[str, str],
    timeout: float,
    stdout_limit: int,
    stderr_limit: int,
    cancel_event: threading.Event,
) -> subprocess.CompletedProcess[bytes]:
    """Own one native child lifecycle inside the dedicated owner thread."""

    if timeout <= 0 or stdout_limit < 0 or stderr_limit < 0:
        raise ValueError("bounded subprocess limits must be non-negative")

    def raise_if_cancelled() -> None:
        if cancel_event.is_set():
            raise _BoundedSubprocessCancelled(
                "bounded subprocess cancelled by its caller"
            )

    argv = list(command)
    started_at = time.monotonic()
    deadline = started_at + timeout
    windows_job: _WindowsKillOnCloseJob | None = None
    popen_options: dict[str, Any] = {}
    if os.name == "nt":
        windows_job = _WindowsKillOnCloseJob()
        popen_options["creationflags"] = windows_job._CREATE_SUSPENDED
    else:
        popen_options["start_new_session"] = True
    owned_processes: list[subprocess.Popen[bytes]] = []
    try:
        raise_if_cancelled()
        process: subprocess.Popen[bytes] = _OwnedBoundedPopen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=dict(env),
            owned_processes=owned_processes,
            **popen_options,
        )
        raise_if_cancelled()
    except BaseException as exc:
        spawn_cleanup_failures: list[str] = []
        if windows_job is not None:
            try:
                windows_job.terminate()
            except Exception as cleanup_exc:
                spawn_cleanup_failures.append(
                    f"process-tree guard cleanup failed: {cleanup_exc}"
                )
        else:
            for owned_process in owned_processes:
                try:
                    _kill_posix_process_group(owned_process.pid)
                except ProcessLookupError:
                    pass
                except Exception as cleanup_exc:
                    spawn_cleanup_failures.append(
                        f"process-tree termination failed: {cleanup_exc}"
                    )
        for owned_process in owned_processes:
            spawn_cleanup_failures.extend(_cleanup_unmanaged_subprocess(owned_process))
        if spawn_cleanup_failures:
            raise _BoundedSubprocessCleanupError(
                "isolated corrected-campaign subprocess setup cleanup failed: "
                + "; ".join(spawn_cleanup_failures)
            ) from exc
        raise
    if windows_job is not None:
        try:
            windows_job.attach_and_resume(process)
            raise_if_cancelled()
        except BaseException as exc:
            cleanup_failures: list[str] = []
            try:
                windows_job.terminate()
            except Exception as cleanup_exc:
                cleanup_failures.append(
                    f"process-tree termination failed: {cleanup_exc}"
                )
            cleanup_failures.extend(_cleanup_unmanaged_subprocess(process))
            if cleanup_failures:
                raise _BoundedSubprocessCleanupError(
                    "isolated corrected-campaign subprocess setup cleanup failed: "
                    + "; ".join(cleanup_failures)
                ) from exc
            raise
    stdin = process.stdin
    stdout = process.stdout
    stderr = process.stderr
    if stdin is None or stdout is None or stderr is None:
        primary_error = RuntimeError("bounded subprocess pipes are unavailable")
        cleanup_failures = []
        try:
            if windows_job is not None:
                windows_job.terminate()
            else:
                _kill_posix_process_group(process.pid)
        except (OSError, RuntimeError) as cleanup_exc:
            cleanup_failures.append(f"process-tree termination failed: {cleanup_exc}")
        cleanup_failures.extend(_cleanup_unmanaged_subprocess(process))
        if cleanup_failures:
            raise _BoundedSubprocessCleanupError(
                "isolated corrected-campaign subprocess setup cleanup failed: "
                + "; ".join(cleanup_failures)
            ) from primary_error
        raise primary_error

    failures: list[RuntimeError] = []
    worker_cleanup_failures: list[str] = []
    failure_lock = threading.Lock()
    failure_event = threading.Event()
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []

    def fail(message: str) -> None:
        should_kill = False
        with failure_lock:
            if not failures:
                failures.append(RuntimeError(message))
                should_kill = True
        if should_kill:
            failure_event.set()

    def fail_cleanup(message: str) -> None:
        with failure_lock:
            worker_cleanup_failures.append(message)
        failure_event.set()

    def read_pipe(
        stream: IO[bytes],
        chunks: list[bytes],
        *,
        limit: int,
        label: str,
    ) -> None:
        total = 0
        try:
            while True:
                read_size = min(64 * 1024, max(1, limit - total + 1))
                chunk = os.read(stream.fileno(), read_size)
                if not chunk:
                    break
                remaining = limit - total
                if remaining > 0:
                    chunks.append(chunk[:remaining])
                total += len(chunk)
                if total > limit:
                    fail(f"isolated corrected-campaign {label} exceeds byte limit")
                    break
        except (OSError, ValueError) as exc:
            fail(f"isolated corrected-campaign {label} read failed: {exc}")
        finally:
            try:
                stream.close()
            except (OSError, ValueError) as exc:
                fail_cleanup(
                    f"isolated corrected-campaign {label} pipe close failed: {exc}"
                )

    def write_stdin() -> None:
        try:
            stdin.write(input_payload)
            stdin.flush()
        except BrokenPipeError:
            pass
        except (OSError, ValueError) as exc:
            fail(f"isolated corrected-campaign input write failed: {exc}")
        finally:
            try:
                stdin.close()
            except (OSError, ValueError) as exc:
                fail_cleanup(
                    f"isolated corrected-campaign stdin pipe close failed: {exc}"
                )

    threads: list[threading.Thread] = []
    thread_streams: list[IO[bytes]] = []
    attempted_thread_bindings: list[tuple[threading.Thread, IO[bytes]]] = []
    lifecycle_error: BaseException | None = None
    timed_out = False
    try:
        raise_if_cancelled()
        threads.append(
            threading.Thread(
                target=read_pipe,
                args=(stdout, stdout_chunks),
                kwargs={"limit": stdout_limit, "label": "stdout"},
                daemon=True,
                name=f"depplm-bounded-{process.pid}-stdout",
            )
        )
        thread_streams.append(stdout)
        threads.append(
            threading.Thread(
                target=read_pipe,
                args=(stderr, stderr_chunks),
                kwargs={"limit": stderr_limit, "label": "stderr"},
                daemon=True,
                name=f"depplm-bounded-{process.pid}-stderr",
            )
        )
        thread_streams.append(stderr)
        threads.append(
            threading.Thread(
                target=write_stdin,
                daemon=True,
                name=f"depplm-bounded-{process.pid}-stdin",
            )
        )
        thread_streams.append(stdin)

        for thread, stream in zip(threads, thread_streams, strict=True):
            raise_if_cancelled()
            attempted_thread_bindings.append((thread, stream))
            thread.start()

        while True:
            raise_if_cancelled()
            if failure_event.is_set():
                break
            if process.poll() is not None:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            failure_event.wait(min(remaining, 0.02))
    except BaseException as exc:
        lifecycle_error = exc

    lifecycle_cleanup_failures: list[str] = []
    try:
        if windows_job is not None:
            windows_job.terminate()
        else:
            try:
                _kill_posix_process_group(process.pid)
            except ProcessLookupError:
                pass
    except Exception as exc:
        lifecycle_cleanup_failures.append(f"process-tree termination failed: {exc}")

    cleanup_deadline = time.monotonic() + BOUNDED_SUBPROCESS_CLEANUP_TIMEOUT
    if process.poll() is None:
        try:
            process.kill()
        except OSError as exc:
            lifecycle_cleanup_failures.append(
                f"direct-process termination failed: {exc}"
            )
    if process.poll() is None:
        remaining = cleanup_deadline - time.monotonic()
        if remaining > 0:
            try:
                process.wait(timeout=remaining)
            except (OSError, subprocess.TimeoutExpired) as exc:
                lifecycle_cleanup_failures.append(f"direct-process wait failed: {exc}")
        else:
            lifecycle_cleanup_failures.append("direct-process cleanup deadline expired")

    started_thread_bindings: list[tuple[threading.Thread, IO[bytes]]] = []
    unsettled_thread_bindings: list[tuple[threading.Thread, IO[bytes]]] = []
    for thread, stream in attempted_thread_bindings:
        start_status = _settle_attempted_thread_start(thread, cleanup_deadline)
        if start_status is True:
            started_thread_bindings.append((thread, stream))
        elif start_status is None:
            unsettled_thread_bindings.append((thread, stream))
            lifecycle_cleanup_failures.append(
                f"pipe worker start did not settle: {thread.name}"
            )
    started_threads = [thread for thread, _stream in started_thread_bindings]
    started_streams = [stream for _thread, stream in started_thread_bindings]
    unsettled_streams = [stream for _thread, stream in unsettled_thread_bindings]
    for stream in (stdout, stderr, stdin):
        if any(
            stream is owned_stream
            for owned_stream in (*started_streams, *unsettled_streams)
        ):
            continue
        try:
            stream.close()
        except OSError as exc:
            lifecycle_cleanup_failures.append(
                f"unowned subprocess pipe close failed: {exc}"
            )

    for thread in started_threads:
        remaining = cleanup_deadline - time.monotonic()
        if remaining <= 0:
            break
        thread.join(timeout=remaining)
    alive_threads = [thread.name for thread in started_threads if thread.is_alive()]
    if alive_threads:
        lifecycle_cleanup_failures.append(
            "pipe workers did not terminate: " + ", ".join(alive_threads)
        )
    with failure_lock:
        lifecycle_cleanup_failures.extend(worker_cleanup_failures)

    if lifecycle_cleanup_failures:
        cleanup_error = _BoundedSubprocessCleanupError(
            "isolated corrected-campaign subprocess cleanup failed: "
            + "; ".join(lifecycle_cleanup_failures)
        )
        cleanup_cause = lifecycle_error
        if cleanup_cause is None and timed_out:
            cleanup_cause = subprocess.TimeoutExpired(argv, timeout)
        if cleanup_cause is None and not alive_threads and failures:
            cleanup_cause = failures[0]
        if cleanup_cause is not None:
            raise cleanup_error from cleanup_cause
        raise cleanup_error

    stdout_payload = b"".join(tuple(stdout_chunks))
    stderr_payload = b"".join(tuple(stderr_chunks))
    if lifecycle_error is not None:
        raise lifecycle_error
    if failures:
        raise failures[0]
    if timed_out:
        raise subprocess.TimeoutExpired(
            argv,
            timeout,
            output=stdout_payload,
            stderr=stderr_payload,
        )
    returncode = process.returncode
    if returncode is None:
        raise RuntimeError("isolated corrected-campaign process did not terminate")
    return subprocess.CompletedProcess(
        argv,
        returncode,
        stdout=stdout_payload,
        stderr=stderr_payload,
    )


def _run_bounded_binary_subprocess(
    command: Sequence[str],
    *,
    input_payload: bytes,
    cwd: Path,
    env: Mapping[str, str],
    timeout: float,
    stdout_limit: int,
    stderr_limit: int,
) -> subprocess.CompletedProcess[bytes]:
    """Run trusted bound-source code with byte, time, and tree cleanup limits.

    Native process acquisition and cleanup run in a dedicated owner thread so a
    single ordinary caller-thread KeyboardInterrupt or SystemExit can request
    cancellation and wait for bounded cleanup.  This is not an operating-system
    sandbox and cannot guarantee cleanup after interpreter termination, repeated
    asynchronous exceptions, or an injected BaseException inside the owner.
    """

    if timeout <= 0 or stdout_limit < 0 or stderr_limit < 0:
        raise ValueError("bounded subprocess limits must be non-negative")

    cancel_event = threading.Event()
    done_event = threading.Event()
    results: list[subprocess.CompletedProcess[bytes]] = []
    errors: list[tuple[BaseException, Any]] = []

    def own_lifecycle() -> None:
        try:
            if cancel_event.is_set():
                raise _BoundedSubprocessCancelled(
                    "bounded subprocess cancelled before native acquisition"
                )
            results.append(
                _run_bounded_binary_subprocess_owned(
                    command,
                    input_payload=input_payload,
                    cwd=cwd,
                    env=env,
                    timeout=timeout,
                    stdout_limit=stdout_limit,
                    stderr_limit=stderr_limit,
                    cancel_event=cancel_event,
                )
            )
        except BaseException as exc:
            errors.append((exc, exc.__traceback__))
        finally:
            done_event.set()

    owner = threading.Thread(
        target=own_lifecycle,
        daemon=True,
        name="depplm-subprocess-owner",
    )
    attempted_start = False
    try:
        attempted_start = True
        owner.start()
        while not done_event.wait(0.02):
            pass
        owner.join()
    except BaseException as primary:
        primary_traceback = primary.__traceback__
        cancel_event.set()
        cleanup_deadline = time.monotonic() + BOUNDED_SUBPROCESS_CLEANUP_TIMEOUT
        start_status = (
            _settle_attempted_thread_start(owner, cleanup_deadline)
            if attempted_start
            else False
        )
        if start_status is None:
            raise _BoundedSubprocessCleanupError(
                "bounded subprocess owner start did not settle"
            ) from primary
        if start_status is True:
            remaining = cleanup_deadline - time.monotonic()
            if remaining <= 0 or not done_event.wait(remaining):
                raise _BoundedSubprocessCleanupError(
                    "bounded subprocess owner did not confirm cleanup"
                ) from primary
            remaining = cleanup_deadline - time.monotonic()
            if remaining > 0:
                owner.join(timeout=remaining)
            if owner.is_alive():
                raise _BoundedSubprocessCleanupError(
                    "bounded subprocess owner did not terminate after cleanup"
                ) from primary
            if errors and isinstance(errors[0][0], _BoundedSubprocessCleanupError):
                cleanup_error, cleanup_traceback = errors[0]
                raise cleanup_error.with_traceback(cleanup_traceback) from primary
        raise primary.with_traceback(primary_traceback)

    if errors:
        owner_error, owner_traceback = errors[0]
        raise owner_error.with_traceback(owner_traceback)
    if len(results) != 1:
        raise RuntimeError("bounded subprocess owner returned no result")
    return results[0]


_CORRECTED_CAMPAIGN_RENDER_PROGRAM = r"""
import json
import sys
import types


def read_payload():
    header = sys.stdin.buffer.read(8)
    if len(header) != 8:
        raise RuntimeError("isolated rendering input frame is truncated")
    size = int.from_bytes(header, "big")
    payload = sys.stdin.buffer.read(size)
    if len(payload) != size:
        raise RuntimeError("isolated rendering input payload is truncated")
    return payload


generator_payload = read_payload()
campaign_payload = read_payload()
mode = read_payload().decode("ascii")
clinical_results_payload = read_payload()
completion_sha256 = read_payload().decode("ascii")
results_sha256 = read_payload().decode("ascii")
if sys.stdin.buffer.read(1):
    raise RuntimeError("isolated rendering input has trailing bytes")

module_name = "_depplm_hash_bound_campaign_generator"
generator = types.ModuleType(module_name)
generator.__file__ = "paper/nar_major_revision/figures/gen_fig_corrected_campaign.py"
generator.__package__ = ""
sys.modules[module_name] = generator
exec(
    compile(generator_payload, generator.__file__, "exec"),
    generator.__dict__,
)

table_renderer = getattr(generator, "render_table", None)
figure_renderer = getattr(generator, "render_core_figure_payloads", None)
if (
    generator.CAMPAIGN_SCHEMA != "DepPLM-corrected-v2.2-campaign"
    or generator.PUBLICATION_PROVENANCE_SCHEMA
    != "DepPLM-corrected-campaign-publication-provenance-v1"
    or not callable(table_renderer)
    or not callable(figure_renderer)
):
    raise RuntimeError("corrected-campaign generator contract is invalid")

campaign = json.loads(campaign_payload)
campaign_table = table_renderer(campaign)
figures = figure_renderer(campaign)
if not isinstance(campaign_table, str):
    raise RuntimeError("corrected-campaign table renderer returned invalid output")
if (
    not isinstance(figures, dict)
    or set(figures) != {"figure_pdf", "figure_png"}
    or any(not isinstance(value, bytes) for value in figures.values())
):
    raise RuntimeError("corrected-campaign figure renderer returned invalid output")

clinical_table = b""
if mode == "full":
    clinical_renderer = getattr(generator, "render_clinical_table", None)
    if not callable(clinical_renderer) or not clinical_results_payload:
        raise RuntimeError("clinical table renderer contract is invalid")
    clinical_results = json.loads(clinical_results_payload)
    rendered_clinical_table = clinical_renderer(
        clinical_results,
        completion_sha256=completion_sha256,
        results_sha256=results_sha256,
    )
    if not isinstance(rendered_clinical_table, str):
        raise RuntimeError("clinical table renderer returned invalid output")
    clinical_table = rendered_clinical_table.encode("utf-8")
elif mode != "core_only" or clinical_results_payload:
    raise RuntimeError("corrected-campaign rendering mode is invalid")

output = sys.stdout.buffer
for payload in (
    campaign_table.encode("utf-8"),
    figures["figure_pdf"],
    figures["figure_png"],
    clinical_table,
):
    output.write(len(payload).to_bytes(8, "big"))
    output.write(payload)
output.flush()
"""


def _render_corrected_campaign_in_isolated_process(
    *,
    generator_payload: bytes,
    campaign_payload: bytes,
    mode: str,
    clinical_results_payload: bytes = b"",
    completion_sha256: str = "",
    results_sha256: str = "",
) -> dict[str, bytes]:
    request_frames = (
        ("generator_source", generator_payload),
        ("campaign_report", campaign_payload),
        ("mode", mode.encode("ascii")),
        ("clinical_results", clinical_results_payload),
        ("clinical_completion_sha256", completion_sha256.encode("ascii")),
        ("clinical_results_sha256", results_sha256.encode("ascii")),
    )
    for name, payload in request_frames:
        if len(payload) > CORRECTED_CAMPAIGN_INPUT_FRAME_LIMITS[name]:
            raise RuntimeError(f"corrected-campaign {name} input exceeds byte limit")
    request = b"".join(
        len(payload).to_bytes(8, "big") + payload for _, payload in request_frames
    )
    try:
        interpreter = sys.executable
        if not interpreter:
            raise OSError("current Python interpreter path is unavailable")
        Path(interpreter).resolve(strict=True)
        with tempfile.TemporaryDirectory(
            prefix="depplm-corrected-campaign-render-"
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            environment = os.environ.copy()
            for name in tuple(environment):
                normalized = name.upper()
                if normalized.startswith("PYTHON") or normalized.startswith("MPL"):
                    environment.pop(name, None)
            environment.update(
                {
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONPATH": "",
                    "MPLBACKEND": "Agg",
                    "MPLCONFIGDIR": os.fspath(temporary_root / "mplconfig"),
                }
            )
            completed = _run_bounded_binary_subprocess(
                [
                    interpreter,
                    "-I",
                    "-c",
                    _CORRECTED_CAMPAIGN_RENDER_PROGRAM,
                ],
                input_payload=request,
                cwd=temporary_root,
                env=environment,
                timeout=180,
                stdout_limit=CORRECTED_CAMPAIGN_STDOUT_LIMIT,
                stderr_limit=CORRECTED_CAMPAIGN_STDERR_LIMIT,
            )
    except (OSError, OverflowError, subprocess.TimeoutExpired, ValueError) as exc:
        raise RuntimeError("isolated corrected-campaign rendering failed") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        if len(detail) > 2_000:
            detail = detail[-2_000:]
        raise RuntimeError(
            "isolated corrected-campaign rendering failed"
            + (f": {detail}" if detail else "")
        )

    output_names = (
        "campaign_table",
        "figure_pdf",
        "figure_png",
        "clinical_table",
    )
    output_payload = completed.stdout
    offset = 0
    rendered: dict[str, bytes] = {}
    for name in output_names:
        if len(output_payload) - offset < 8:
            raise RuntimeError("isolated corrected-campaign output is truncated")
        size = int.from_bytes(output_payload[offset : offset + 8], "big")
        offset += 8
        if size > CORRECTED_CAMPAIGN_OUTPUT_FRAME_LIMITS[name]:
            raise RuntimeError(
                f"isolated corrected-campaign {name} frame exceeds byte limit"
            )
        end = offset + size
        if end > len(output_payload):
            raise RuntimeError("isolated corrected-campaign output is truncated")
        rendered[name] = output_payload[offset:end]
        offset = end
    if offset != len(output_payload):
        raise RuntimeError("isolated corrected-campaign output has trailing bytes")
    if mode == "full":
        if not rendered["clinical_table"]:
            raise RuntimeError("isolated clinical table output is empty")
    elif rendered.pop("clinical_table"):
        raise RuntimeError("core-only rendering returned a clinical table")
    return rendered


def _validate_corrected_campaign_provenance_bindings(
    root: Path,
    *,
    selected: Sequence[SelectedFile] | None = None,
) -> None:
    provenance_path = _safe_workspace_file(
        root,
        CORRECTED_CAMPAIGN_PROVENANCE,
        field="corrected-campaign publication provenance",
    )
    provenance_payload, provenance_fingerprint = _read_stable_bytes(
        provenance_path,
        label="corrected-campaign publication provenance",
    )
    document = _validate_corrected_campaign_provenance_document(provenance_payload)
    records: dict[str, tuple[dict[str, Any], str]] = {
        "generator": (document["generator"], "source"),
        **{
            f"input {name}": (record, "evidence")
            for name, record in document["inputs"].items()
        },
        **{
            f"output {name}": (record, "manuscript")
            for name, record in document["outputs"].items()
        },
    }
    selected_by_path = (
        {item.relative: item for item in selected} if selected is not None else None
    )
    bound_records: dict[str, tuple[Path, bytes, FileFingerprint]] = {}
    for label, (record, expected_category) in records.items():
        if selected_by_path is not None:
            selected_item = selected_by_path.get(record["path"])
            if selected_item is None:
                _fail(
                    f"corrected-campaign publication {label} is absent from the "
                    "release allowlist"
                )
            if selected_item.category != expected_category:
                _fail(
                    f"corrected-campaign publication {label} must be selected as "
                    f"{expected_category}"
                )
            if (
                selected_item.bytes != record["bytes"]
                or selected_item.sha256 != record["sha256"]
            ):
                _fail(
                    f"corrected-campaign publication {label} selection differs "
                    "from its provenance binding"
                )
        bound_path = _safe_workspace_file(
            root,
            record["path"],
            field=f"corrected-campaign publication {label}",
        )
        payload, fingerprint = _read_stable_bytes(
            bound_path,
            label=f"corrected-campaign publication {label}",
        )
        if (
            fingerprint.bytes != record["bytes"]
            or fingerprint.sha256 != record["sha256"]
        ):
            _fail(
                f"corrected-campaign publication {label} differs from its "
                "provenance binding"
            )
        bound_records[label] = (bound_path, payload, fingerprint)

    campaign_path, campaign_payload, _ = bound_records["input campaign"]
    campaign = _parse_json_bytes(
        campaign_payload,
        label="corrected-campaign publication campaign",
        path=campaign_path,
    )
    if (
        campaign.get("schema") != "DepPLM-corrected-v2.2-campaign"
        or campaign.get("protocol_validation_passed") is not True
        or campaign.get("scientific_result") is not True
    ):
        _fail("corrected-campaign publication campaign is ineligible")

    clinical_results_payload = b""
    completion_sha256 = ""
    results_sha256 = ""
    if document["mode"] == "full":
        completion_path, completion_payload, _ = bound_records[
            "input clinical_completion"
        ]
        results_path, clinical_results_payload, _ = bound_records[
            "input clinical_results"
        ]
        completion = _parse_json_bytes(
            completion_payload,
            label="corrected-campaign publication clinical completion",
            path=completion_path,
        )
        results = _parse_json_bytes(
            clinical_results_payload,
            label="corrected-campaign publication clinical results",
            path=results_path,
        )
        if (
            completion.get("schema") != CLINICAL_COMPLETION_SCHEMA
            or completion.get("scientific_result") is not True
            or results.get("schema") != CLINICAL_RESULT_SCHEMA
            or results.get("scientific_result") is not True
        ):
            _fail("corrected-campaign publication clinical inputs are ineligible")
        completion_sha256 = document["inputs"]["clinical_completion"]["sha256"]
        results_sha256 = document["inputs"]["clinical_results"]["sha256"]

    generator_path, generator_payload, generator_fingerprint = bound_records[
        "generator"
    ]
    generator_binding = SelectedFile(
        relative=document["generator"]["path"],
        category="source",
        source=generator_path,
        bytes=generator_fingerprint.bytes,
        sha256=generator_fingerprint.sha256,
        device=generator_fingerprint.device,
        inode=generator_fingerprint.inode,
        mode=generator_fingerprint.mode,
        modified_ns=generator_fingerprint.modified_ns,
        links=generator_fingerprint.links,
        payload=generator_payload,
    )
    try:
        with _TrustedWorkspaceModuleSession(
            root,
            expected_sources={"gen_fig_corrected_campaign": generator_binding},
        ) as trusted_modules:
            generator = trusted_modules.load("gen_fig_corrected_campaign")
            table_renderer = getattr(generator, "render_table", None)
            figure_renderer = getattr(generator, "render_core_figure_payloads", None)
            core_contract_differs = (
                generator.CAMPAIGN_SCHEMA != "DepPLM-corrected-v2.2-campaign"
                or generator.PUBLICATION_PROVENANCE_SCHEMA
                != CORRECTED_CAMPAIGN_PROVENANCE_SCHEMA
                or not callable(table_renderer)
                or not callable(figure_renderer)
            )
            clinical_renderer = getattr(generator, "render_clinical_table", None)
            clinical_contract_matches = document["mode"] != "full" or (
                generator.CLINICAL_RESULT_SCHEMA == CLINICAL_RESULT_SCHEMA
                and generator.CLINICAL_VERIFICATION_SCHEMA
                == CLINICAL_VERIFICATION_SCHEMA
                and generator.CLINICAL_COMPLETION_SCHEMA == CLINICAL_COMPLETION_SCHEMA
                and generator.CLINICAL_COMPLETION_FIELDS == CLINICAL_COMPLETION_FIELDS
                and generator.CLINICAL_VERIFICATION_FIELDS
                == CLINICAL_VERIFICATION_FIELDS
                and generator.CLINICAL_VERIFICATION_ARTIFACTS
                == CLINICAL_VERIFICATION_ARTIFACTS
                and generator.CLINICAL_VERIFICATION_METHOD
                == CLINICAL_VERIFICATION_METHOD
                and generator.CLINICAL_PUBLICATION_COMPLETION_PREFIX
                == CLINICAL_PUBLICATION_COMPLETION_PREFIX
                and generator.CLINICAL_PUBLICATION_RESULTS_PREFIX
                == CLINICAL_PUBLICATION_RESULTS_PREFIX
                and callable(clinical_renderer)
            )
            if core_contract_differs or not clinical_contract_matches:
                _fail(
                    "corrected-campaign publication generator contract differs "
                    "from release"
                )
        expected_outputs = _render_corrected_campaign_in_isolated_process(
            generator_payload=generator_payload,
            campaign_payload=campaign_payload,
            mode=document["mode"],
            clinical_results_payload=clinical_results_payload,
            completion_sha256=completion_sha256,
            results_sha256=results_sha256,
        )
    except ReleaseValidationError:
        raise
    except Exception as exc:
        raise ReleaseValidationError(
            "corrected-campaign publication outputs cannot be reproduced from "
            "the bound campaign"
        ) from exc
    for name, expected_payload in expected_outputs.items():
        if bound_records[f"output {name}"][1] != expected_payload:
            _fail(
                f"corrected-campaign publication {name} differs from the "
                "deterministic rendering of the bound campaign"
            )

    for label, (bound_path, payload, fingerprint) in bound_records.items():
        final_payload, final_fingerprint = _read_stable_bytes(
            bound_path,
            label=f"corrected-campaign publication {label} final binding",
        )
        if final_payload != payload or final_fingerprint != fingerprint:
            _fail(
                f"corrected-campaign publication {label} changed during "
                "provenance validation"
            )
    final_provenance_payload, final_provenance_fingerprint = _read_stable_bytes(
        provenance_path,
        label="corrected-campaign publication provenance final binding",
    )
    if (
        final_provenance_payload != provenance_payload
        or final_provenance_fingerprint != provenance_fingerprint
    ):
        _fail(
            "corrected-campaign publication provenance changed during "
            "provenance validation"
        )


def _decode_public_text(payload: bytes, *, relative: str) -> str:
    binary_magics = (
        b"RDX2",
        b"RDX3",
        b"RDA2",
        b"RDA3",
        b"\x1f\x8b",
        b"PK\x03\x04",
        b"\x93NUMPY",
        b"SQLite format 3\x00",
        b"\x7fELF",
        b"%PDF-",
        b"\x89PNG\r\n\x1a\n",
    )
    if any(payload.startswith(magic) for magic in binary_magics) or b"\x00" in payload:
        _fail(f"public text has binary/data-file content: {relative}")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReleaseValidationError(f"public text is not UTF-8: {relative}") from exc
    if any(ord(character) < 32 and character not in "\t\n\r" for character in text):
        _fail(f"public text has binary control bytes: {relative}")
    if re.search(
        r"-{5}BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-{5}",
        text,
        flags=re.ASCII,
    ):
        _fail(f"public text contains private-key material: {relative}")
    return text


def _validate_clinical_publication_binding(
    *,
    root: Path,
    publication: SelectedFile,
    results: Mapping[str, Any],
    completion_sha256: str,
    results_sha256: str,
    trusted_source_bindings: Mapping[str, SelectedFile] | None,
) -> None:
    if not SHA256_RE.fullmatch(completion_sha256) or not SHA256_RE.fullmatch(
        results_sha256
    ):
        _fail("clinical publication binding digests are invalid")
    if (
        publication.relative != CLINICAL_PUBLICATION_TABLE
        or publication.category != "manuscript"
    ):
        _fail("clinical publication table selection is invalid")
    _decode_public_text(
        publication.payload,
        relative=CLINICAL_PUBLICATION_TABLE,
    )
    try:
        with _TrustedWorkspaceModuleSession(
            root,
            expected_sources=trusted_source_bindings,
        ) as trusted_modules:
            generator = trusted_modules.load("gen_fig_corrected_campaign")
            contract_matches = (
                generator.CLINICAL_RESULT_SCHEMA == CLINICAL_RESULT_SCHEMA
                and generator.CLINICAL_VERIFICATION_SCHEMA
                == CLINICAL_VERIFICATION_SCHEMA
                and generator.CLINICAL_COMPLETION_SCHEMA == CLINICAL_COMPLETION_SCHEMA
                and generator.CLINICAL_COMPLETION_FIELDS == CLINICAL_COMPLETION_FIELDS
                and generator.CLINICAL_VERIFICATION_FIELDS
                == CLINICAL_VERIFICATION_FIELDS
                and generator.CLINICAL_VERIFICATION_ARTIFACTS
                == CLINICAL_VERIFICATION_ARTIFACTS
                and generator.CLINICAL_VERIFICATION_METHOD
                == CLINICAL_VERIFICATION_METHOD
                and generator.CLINICAL_PUBLICATION_COMPLETION_PREFIX
                == CLINICAL_PUBLICATION_COMPLETION_PREFIX
                and generator.CLINICAL_PUBLICATION_RESULTS_PREFIX
                == CLINICAL_PUBLICATION_RESULTS_PREFIX
            )
            renderer = getattr(generator, "render_clinical_table", None)
            if not contract_matches or not callable(renderer):
                _fail("clinical publication generator contract differs from release")
            expected_payload = renderer(
                results,
                completion_sha256=completion_sha256,
                results_sha256=results_sha256,
            ).encode("utf-8")
    except ReleaseValidationError:
        raise
    except Exception as exc:
        raise ReleaseValidationError(
            "clinical publication table cannot be reproduced from selected v3 results"
        ) from exc
    if publication.payload != expected_payload:
        _fail(
            "generated clinical table differs from the deterministic rendering of "
            "the selected clinical completion and results"
        )


def _validate_source_or_test_file(
    payload: bytes, *, relative: str, category: str
) -> str:
    if Path(relative).suffix != ".py":
        _fail(f"{category} files must use the reviewed Python text format: {relative}")
    text = _decode_public_text(payload, relative=relative)
    try:
        compile(text, relative, "exec", dont_inherit=True)
    except (SyntaxError, ValueError) as exc:
        raise ReleaseValidationError(
            f"{category} file is not syntactically valid UTF-8 Python: {relative}"
        ) from exc
    return text


def _reject_renamed_tabular_content(text: str, *, relative: str) -> None:
    nonempty = [line for line in text.splitlines() if line.strip()]
    for delimiter in (",", "\t"):
        if len(nonempty) >= 3:
            counts = [line.count(delimiter) for line in nonempty]
            if counts[0] > 0 and len(set(counts)) == 1:
                _fail(
                    "reviewed publication text appears to contain renamed tabular "
                    f"outcome data: {relative}"
                )


def _active_manuscript_placeholder_findings(
    lines: Iterable[str], *, relative: str
) -> list[dict[str, Any]]:
    """Return explicit submission blockers in an active author-authored TeX file."""
    if (
        relative not in ACTIVE_MANUSCRIPT_DEPENDENCIES
        or Path(relative).suffix.casefold() != ".tex"
    ):
        return []
    findings: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        for match in MANUSCRIPT_BLOCKER_RE.finditer(line):
            findings.append(
                {
                    "path": relative,
                    "line": line_number,
                    "marker": match.group(0),
                }
            )
    return findings


def _validate_active_manuscript_source(lines: Iterable[str], *, relative: str) -> None:
    findings = _active_manuscript_placeholder_findings(lines, relative=relative)
    if findings:
        locations = ", ".join(
            f"{item['path']}:{item['line']} ({item['marker']})" for item in findings
        )
        _fail(
            f"active manuscript contains unresolved submission placeholders: {locations}"
        )


def _validate_reviewed_publication_file(
    payload: bytes,
    *,
    relative: str,
    category: str,
    enforce_manuscript_placeholders: bool = True,
) -> None:
    if category in {"source", "test"}:
        text = _validate_source_or_test_file(
            payload,
            relative=relative,
            category=category,
        )
        _reject_renamed_tabular_content(text, relative=relative)
        return
    if category not in REVIEWED_PUBLICATION_FILES:
        return
    if relative not in REVIEWED_PUBLICATION_FILES[category]:
        _fail(f"{category} file is absent from the exact reviewed public inventory")
    if relative == CORRECTED_CAMPAIGN_PROVENANCE:
        _validate_corrected_campaign_provenance_document(payload)
        return
    suffix = Path(relative).suffix.casefold()
    if suffix == ".png":
        _validate_png_bytes(payload, relative=relative)
        return
    if suffix == ".pdf":
        _validate_pdf_bytes(payload, relative=relative)
        return
    if suffix not in {".tex", ".bib", ".bst", ".cls", ".md"}:
        _fail(f"reviewed publication file extension is not approved: {relative}")
    text = _decode_public_text(payload, relative=relative)
    if enforce_manuscript_placeholders:
        _validate_active_manuscript_source(text.splitlines(), relative=relative)
    _reject_renamed_tabular_content(text, relative=relative)


def _validate_release_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail("release must be an object")
    required = {
        "title",
        "version",
        "author_supplied_doi",
        "publication_date",
        "license",
        "creators",
        "funding",
        "conflict_of_interest",
        "ai_use_disclosure",
    }
    if set(value) != required:
        _fail(f"release fields must be exactly {sorted(required)}")
    raw_doi = value["author_supplied_doi"]
    if not isinstance(raw_doi, str) or any(
        not character.isprintable() for character in raw_doi
    ):
        _fail("release.author_supplied_doi contains control or non-printable text")
    normalized_fields = {
        field: _meaningful_text(value[field], field=f"release.{field}")
        for field in required - {"creators"}
    }
    doi = normalized_fields["author_supplied_doi"]
    if not DOI_RE.fullmatch(doi):
        _fail("release.author_supplied_doi must have valid DOI syntax")
    try:
        date.fromisoformat(normalized_fields["publication_date"])
    except ValueError as exc:
        raise ReleaseValidationError(
            "release.publication_date must be YYYY-MM-DD"
        ) from exc
    if not re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z._+-]*", normalized_fields["version"]):
        _fail("release.version contains unsupported characters")
    creators = value["creators"]
    if not isinstance(creators, list) or not creators:
        _fail("release.creators must be a non-empty list")
    normalized_creators: list[dict[str, Any]] = []
    for index, creator in enumerate(creators):
        if not isinstance(creator, dict):
            _fail(f"release.creators[{index}] must be an object")
        required_creator = {"name", "affiliation", "contributions"}
        if not required_creator.issubset(creator) or set(creator) - (
            required_creator | {"orcid"}
        ):
            _fail(f"release.creators[{index}] has incomplete or unexpected fields")
        name = _meaningful_text(
            creator["name"], field=f"release.creators[{index}].name"
        )
        affiliation = _meaningful_text(
            creator["affiliation"], field=f"release.creators[{index}].affiliation"
        )
        contributions = creator["contributions"]
        if not isinstance(contributions, list) or not contributions:
            _fail(f"release.creators[{index}].contributions must be non-empty")
        normalized = {
            "name": name,
            "affiliation": affiliation,
            "contributions": [
                _meaningful_text(item, field=f"release.creators[{index}].contributions")
                for item in contributions
            ],
        }
        if "orcid" in creator:
            orcid = _meaningful_text(
                creator["orcid"], field=f"release.creators[{index}].orcid"
            )
            if not ORCID_RE.fullmatch(orcid):
                _fail(f"release.creators[{index}].orcid is not in ORCID format")
            if not _orcid_checksum_valid(orcid):
                _fail(f"release.creators[{index}].orcid checksum is invalid")
            normalized["orcid"] = orcid
        normalized_creators.append(normalized)
    if len({item["name"].casefold() for item in normalized_creators}) != len(
        normalized_creators
    ):
        _fail("release.creators contains duplicate names")
    normalized_orcids = [
        item["orcid"].upper() for item in normalized_creators if "orcid" in item
    ]
    if len(set(normalized_orcids)) != len(normalized_orcids):
        _fail("release.creators contains duplicate normalized ORCIDs")
    return {**normalized_fields, "creators": normalized_creators}


def _select_files(
    root: Path,
    records: Any,
    *,
    enforce_manuscript_placeholders: bool = True,
) -> tuple[list[SelectedFile], dict[str, str]]:
    if not isinstance(records, list) or not records:
        _fail("files must be a non-empty explicit allowlist")
    selected: list[SelectedFile] = []
    categories: dict[str, str] = {}
    seen_casefold: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {"path", "category"}:
            _fail(f"files[{index}] must contain exactly path and category")
        relative = _relative_path(record["path"], field=f"files[{index}].path")
        category = record["category"]
        if not isinstance(category, str) or category not in ALLOWED_CATEGORIES:
            _fail(f"files[{index}].category is unsupported: {category!r}")
        if not _path_allowed_for_category(relative, category):
            _fail(f"files[{index}] path is outside the allowlisted category root")
        if PurePosixPath(relative).name.casefold() == "outcome-snapshot.rdata":
            _fail(
                "organoid outcome snapshots are local verifier inputs and may not "
                "enter the public allowlist"
            )
        folded = relative.casefold()
        if folded in seen_casefold:
            _fail(f"files contains a duplicate or case-colliding path: {relative}")
        seen_casefold.add(folded)
        source = _safe_workspace_file(root, relative, field=f"files[{index}].path")
        payload, fingerprint = _read_stable_bytes(
            source,
            label=f"files[{index}].path",
        )
        _validate_reviewed_publication_file(
            payload,
            relative=relative,
            category=category,
            enforce_manuscript_placeholders=enforce_manuscript_placeholders,
        )
        _scan_public_text_for_local_paths(
            source,
            relative=relative,
            category=category,
            payload=payload,
        )
        selected.append(
            SelectedFile(
                relative=relative,
                category=category,
                source=source,
                bytes=fingerprint.bytes,
                sha256=fingerprint.sha256,
                device=fingerprint.device,
                inode=fingerprint.inode,
                mode=fingerprint.mode,
                modified_ns=fingerprint.modified_ns,
                links=fingerprint.links,
                payload=payload,
            )
        )
        categories[relative] = category
    selected.sort(key=lambda item: item.relative)
    return selected, categories


def _assert_selected_files_unchanged(
    root: Path, selected: Sequence[SelectedFile]
) -> None:
    for item in selected:
        source = _safe_workspace_file(
            root,
            item.relative,
            field=f"validated file {item.relative}",
        )
        payload, observed = _read_stable_bytes(
            source,
            label=f"validated file {item.relative}",
        )
        _validate_reviewed_publication_file(
            payload,
            relative=item.relative,
            category=item.category,
        )
        _assert_fingerprint_matches(
            item,
            observed,
            label=f"validated file {item.relative}",
        )
        _scan_public_text_for_local_paths(
            source,
            relative=item.relative,
            category=item.category,
            payload=payload,
        )


def _snapshot_selected_files(
    root: Path,
    snapshot_root: Path,
    selected: Sequence[SelectedFile],
) -> list[SelectedFile]:
    """Revalidate and freeze selected files without a pathname-backed snapshot."""
    # Retain the argument so callers cannot silently choose a new storage path.
    # It is deliberately never created or traversed.
    del snapshot_root
    snapshotted: list[SelectedFile] = []
    for item in selected:
        source = _safe_workspace_file(
            root,
            item.relative,
            field=f"snapshot source {item.relative}",
        )
        payload, copied = _read_stable_bytes(
            source,
            label=f"snapshot source {item.relative}",
        )
        _assert_fingerprint_matches(
            item,
            copied,
            label=f"snapshot source {item.relative}",
        )
        if payload != item.payload:
            _fail(f"snapshot source bytes changed: {item.relative}")
        snapshotted.append(item)
    return snapshotted


def _record_matches_file(
    path: Path, record: Mapping[str, Any], *, label: str
) -> FileFingerprint:
    sha = record.get("sha256")
    size = record.get("bytes")
    if not isinstance(sha, str) or not SHA256_RE.fullmatch(sha):
        _fail(f"{label} has an invalid SHA-256")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        _fail(f"{label} has an invalid byte count")
    fingerprint = _fingerprint_file(path, label=label)
    if fingerprint.bytes != size or fingerprint.sha256 != sha:
        _fail(f"{label} no longer matches its bound file")
    return fingerprint


def _require_included(relative: str, included: set[str], *, label: str) -> None:
    normalized = _relative_path(relative.replace("\\", "/"), field=label)
    if normalized not in included:
        _fail(
            f"{label} is not included in the explicit release allowlist: {normalized}"
        )


def _authorize(
    authorized: dict[str, set[str]], relative: str, *categories: str
) -> None:
    authorized.setdefault(relative, set()).update(categories)


def _record_path(
    record: Mapping[str, Any],
    *,
    root: Path,
    base: Path,
    label: str,
) -> tuple[str, Path]:
    raw = record.get("path")
    if not isinstance(raw, str) or not raw:
        _fail(f"{label}.path must be a non-empty path")
    normalized = raw.replace("\\", "/")
    candidate_raw = Path(raw)
    if candidate_raw.is_absolute() or PurePosixPath(normalized).is_absolute():
        _fail(
            f"{label}.path must be workspace-relative; local absolute paths "
            "are forbidden in public evidence"
        )
    try:
        root_lexical = _assert_no_link_or_reparse_components(
            root, label="workspace root"
        )
        base_lexical = _assert_no_link_or_reparse_components(
            base, label=f"{label}.base"
        )
        candidate_lexical = base_lexical.joinpath(
            *PurePosixPath(_relative_path(normalized, field=f"{label}.path")).parts
        )
        _assert_no_link_or_reparse_components(candidate_lexical, label=f"{label}.path")
        candidate = candidate_lexical.resolve(strict=True)
        relative = candidate.relative_to(root_lexical.resolve(strict=True)).as_posix()
    except (OSError, ValueError) as exc:
        raise ReleaseValidationError(
            f"{label}.path escapes the workspace or is missing: {raw}"
        ) from exc
    safe = _safe_workspace_file(root, relative, field=f"{label}.path")
    _record_matches_file(safe, record, label=label)
    return relative, safe


def _strict_workspace_record(
    value: Any,
    *,
    root: Path,
    label: str,
) -> tuple[dict[str, Any], str, Path]:
    normalized_record = _normalize_workspace_record(value, label=label)
    relative = normalized_record["path"]
    observed_relative, path = _record_path(
        normalized_record,
        root=root,
        base=root,
        label=label,
    )
    if observed_relative != relative:
        _fail(f"{label}.path does not resolve to its canonical workspace path")
    return normalized_record, relative, path


def _normalize_workspace_record(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"path", "bytes", "sha256"}:
        _fail(f"{label} must contain exactly path, bytes, and sha256")
    raw_path = value.get("path")
    relative = _relative_path(raw_path, field=f"{label}.path")
    if raw_path != relative:
        _fail(f"{label}.path must be a canonical POSIX workspace-relative path")
    size = value.get("bytes")
    sha256 = value.get("sha256")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        _fail(f"{label}.bytes is invalid")
    if not isinstance(sha256, str) or not SHA256_RE.fullmatch(sha256):
        _fail(f"{label}.sha256 is invalid")
    return {"path": relative, "bytes": size, "sha256": sha256}


def _normalize_receipt_fingerprint(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"bytes", "sha256"}:
        _fail(f"{label} must contain exactly bytes and sha256")
    size = value.get("bytes")
    sha256 = value.get("sha256")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        _fail(f"{label}.bytes is invalid")
    if not isinstance(sha256, str) or not SHA256_RE.fullmatch(sha256):
        _fail(f"{label}.sha256 is invalid")
    return {"bytes": size, "sha256": sha256}


def _normalize_trust_receipt_payload(
    value: Any,
    *,
    role: str,
    root: Path,
) -> dict[str, Any]:
    expected = {"schema", "role", "provenance", "binding", "completion"}
    if role == "independent_biological_validation":
        expected.update({"protocol_lock", "prediction_lock"})
    if not isinstance(value, Mapping) or set(value) != expected:
        _fail(
            f"detached trust receipt for {role} must contain exactly {sorted(expected)}"
        )
    if value.get("schema") != DETACHED_TRUST_RECEIPT_SCHEMA:
        _fail(f"detached trust receipt schema is invalid for {role}")
    if value.get("role") != role:
        _fail(f"detached trust receipt role differs for {role}")
    if value.get("provenance") != DETACHED_TRUST_PROVENANCE:
        _fail(
            f"detached trust receipt provenance is invalid for {role}; "
            "local receipts are not treated as signed or immutable"
        )
    binding, _, _ = _strict_workspace_record(
        value.get("binding"), root=root, label=f"{role} receipt.binding"
    )
    completion, _, _ = _strict_workspace_record(
        value.get("completion"), root=root, label=f"{role} receipt.completion"
    )
    normalized: dict[str, Any] = {
        "schema": DETACHED_TRUST_RECEIPT_SCHEMA,
        "role": role,
        "provenance": dict(DETACHED_TRUST_PROVENANCE),
        "binding": binding,
        "completion": completion,
    }
    if role == "independent_biological_validation":
        protocol_lock, _, _ = _strict_workspace_record(
            value.get("protocol_lock"),
            root=root,
            label=f"{role} receipt.protocol_lock",
        )
        normalized["protocol_lock"] = protocol_lock
        prediction_lock, _, _ = _strict_workspace_record(
            value.get("prediction_lock"),
            root=root,
            label=f"{role} receipt.prediction_lock",
        )
        normalized["prediction_lock"] = prediction_lock
    return normalized


def _load_detached_trust_receipts(
    paths: Mapping[str, Path],
    *,
    root: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(paths, Mapping):
        _fail("detached trust receipt inputs must be a role-to-path mapping")
    roles = set(paths)
    if any(not isinstance(role, str) for role in roles):
        _fail("detached trust receipt roles must be strings")
    if roles != DETACHED_TRUST_RECEIPT_ROLES:
        _fail(
            "detached trust receipt inputs must cover exactly "
            f"{sorted(DETACHED_TRUST_RECEIPT_ROLES)}"
        )
    root_lexical = _assert_no_link_or_reparse_components(root, label="workspace root")
    root_resolved = root_lexical.resolve(strict=True)
    payloads: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    for role in sorted(DETACHED_TRUST_RECEIPT_ROLES):
        raw_path = paths[role]
        if not isinstance(raw_path, Path):
            _fail(f"detached trust receipt path for {role} must be a Path")
        receipt_path = _assert_no_link_or_reparse_components(
            raw_path, label=f"detached trust receipt for {role}"
        )
        try:
            receipt_path.relative_to(root_lexical)
        except ValueError:
            pass
        else:
            _fail(f"detached trust receipt for {role} must be outside the workspace")
        try:
            resolved = receipt_path.resolve(strict=True)
        except OSError as exc:
            raise ReleaseValidationError(
                f"detached trust receipt for {role} is missing"
            ) from exc
        try:
            resolved.relative_to(root_resolved)
        except ValueError:
            pass
        else:
            _fail(f"detached trust receipt for {role} resolves inside the workspace")
        payload, fingerprint = _load_json_with_fingerprint(
            receipt_path,
            label=f"detached trust receipt for {role}",
        )
        if fingerprint.links != 1:
            _fail(f"detached trust receipt for {role} must have exactly one hard link")
        normalized = _normalize_trust_receipt_payload(payload, role=role, root=root)
        payloads[role] = normalized
        summary = {
            **normalized,
            "receipt": {
                "bytes": fingerprint.bytes,
                "sha256": fingerprint.sha256,
            },
        }
        summaries.append(summary)
    return payloads, summaries


def _validate_embedded_trust_receipts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(DETACHED_TRUST_RECEIPT_ROLES):
        _fail("embedded detached trust receipt inventory is incomplete")
    normalized: list[dict[str, Any]] = []
    roles: set[str] = set()
    for index, record in enumerate(value):
        if not isinstance(record, Mapping):
            _fail(f"embedded trust_receipts[{index}] must be an object")
        role = record.get("role")
        if not isinstance(role, str) or role not in DETACHED_TRUST_RECEIPT_ROLES:
            _fail(f"embedded trust_receipts[{index}].role is unsupported")
        if role in roles:
            _fail(f"embedded detached trust receipt role is duplicated: {role}")
        expected = {
            "schema",
            "role",
            "provenance",
            "receipt",
            "binding",
            "completion",
        }
        if role == "independent_biological_validation":
            expected.update({"protocol_lock", "prediction_lock"})
        if set(record) != expected:
            _fail(
                f"embedded trust_receipts[{index}] must contain exactly "
                f"{sorted(expected)}"
            )
        if record.get("schema") != DETACHED_TRUST_RECEIPT_SCHEMA:
            _fail(f"embedded detached trust receipt schema is invalid for {role}")
        if record.get("provenance") != DETACHED_TRUST_PROVENANCE:
            _fail(f"embedded detached trust receipt provenance is invalid for {role}")
        item: dict[str, Any] = {
            "schema": DETACHED_TRUST_RECEIPT_SCHEMA,
            "role": role,
            "provenance": dict(DETACHED_TRUST_PROVENANCE),
            "binding": _normalize_workspace_record(
                record.get("binding"), label=f"embedded {role} receipt.binding"
            ),
            "completion": _normalize_workspace_record(
                record.get("completion"), label=f"embedded {role} receipt.completion"
            ),
            "receipt": _normalize_receipt_fingerprint(
                record.get("receipt"), label=f"embedded {role} receipt"
            ),
        }
        if role == "independent_biological_validation":
            item["protocol_lock"] = _normalize_workspace_record(
                record.get("protocol_lock"),
                label=f"embedded {role} receipt.protocol_lock",
            )
            item["prediction_lock"] = _normalize_workspace_record(
                record.get("prediction_lock"),
                label=f"embedded {role} receipt.prediction_lock",
            )
        normalized.append(item)
        roles.add(role)
    if roles != DETACHED_TRUST_RECEIPT_ROLES:
        _fail("embedded detached trust receipt roles are incomplete")
    if [item["role"] for item in normalized] != sorted(roles):
        _fail("embedded detached trust receipts are not in canonical role order")
    return normalized


def _binding_run_directory(
    payload: Mapping[str, Any], *, root: Path, role: str
) -> tuple[str, Path]:
    relative = _relative_path(
        payload.get("run_directory"), field=f"{role}.run_directory"
    )
    if not relative.startswith("results/"):
        _fail(f"{role}.run_directory must be below results/")
    lexical_root = _assert_no_link_or_reparse_components(root, label="workspace root")
    resolved_root = lexical_root.resolve(strict=True)
    candidate = resolved_root.joinpath(*PurePosixPath(relative).parts)
    try:
        _assert_no_link_or_reparse_components(candidate, label=f"{role}.run_directory")
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise ReleaseValidationError(
            f"{role}.run_directory escapes the workspace or is missing"
        ) from exc
    if not candidate.is_dir():
        _fail(f"{role}.run_directory is not a directory")
    return relative, candidate


def _validate_redistribution_review(value: Any, *, role: str) -> None:
    expected = {
        "status",
        "reviewer",
        "review_date",
        "license_basis",
        "raw_outcome_redistribution",
        "third_party_source_redistribution",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        _fail(f"{role}.redistribution_review must contain exactly {sorted(expected)}")
    if value.get("status") != "approved":
        _fail(f"{role}.redistribution_review is not approved")
    reviewer = value.get("reviewer")
    if not isinstance(reviewer, Mapping) or set(reviewer) != {
        "name",
        "affiliation",
    }:
        _fail(
            f"{role}.redistribution_review.reviewer must contain exactly name and affiliation"
        )
    _substantive_text(
        reviewer.get("name"),
        field=f"{role}.redistribution_review.reviewer.name",
    )
    _substantive_text(
        reviewer.get("affiliation"),
        field=f"{role}.redistribution_review.reviewer.affiliation",
    )
    license_basis = value.get("license_basis")
    if not isinstance(license_basis, Mapping) or set(license_basis) != {
        "decision",
        "reference",
    }:
        _fail(
            f"{role}.redistribution_review.license_basis must contain exactly decision and reference"
        )
    if license_basis.get("decision") != "approved_for_public_release":
        _fail(f"{role}.redistribution_review licence decision is not approved")
    _substantive_text(
        license_basis.get("reference"),
        field=f"{role}.redistribution_review.license_basis.reference",
        minimum=8,
    )
    review_date = value.get("review_date")
    if not isinstance(review_date, str):
        _fail(f"{role}.redistribution_review.review_date must be YYYY-MM-DD")
    try:
        date.fromisoformat(review_date)
    except ValueError as exc:
        raise ReleaseValidationError(
            f"{role}.redistribution_review.review_date must be YYYY-MM-DD"
        ) from exc
    if value.get("raw_outcome_redistribution") is not False:
        _fail(f"{role} may not approve raw outcome redistribution")
    if value.get("third_party_source_redistribution") is not False:
        _fail(f"{role} may not approve third-party source redistribution")


def _validate_public_artifacts(
    records: Any,
    *,
    expected_paths: Sequence[str],
    root: Path,
    role: str,
    included: set[str],
    authorized: dict[str, set[str]],
) -> dict[str, dict[str, Any]]:
    if not isinstance(records, list) or len(records) != len(expected_paths):
        _fail(
            f"{role}.public_artifacts must contain exactly {len(expected_paths)} records"
        )
    observed: dict[str, dict[str, Any]] = {}
    for index, (value, expected_path) in enumerate(
        zip(records, expected_paths, strict=True)
    ):
        record, relative, _ = _strict_workspace_record(
            value,
            root=root,
            label=f"{role}.public_artifacts[{index}]",
        )
        if relative != expected_path:
            _fail(
                f"{role}.public_artifacts is not the exact canonical public inventory"
            )
        if relative.casefold() in observed:
            _fail(f"{role}.public_artifacts repeats a path")
        _require_included(
            relative,
            included,
            label=f"{role}.public_artifacts[{index}]",
        )
        _authorize(authorized, relative, "evidence")
        observed[relative.casefold()] = record
    return observed


def _validate_local_only_artifacts(
    records: Any,
    *,
    expected_paths: Sequence[str],
    root: Path,
    role: str,
    included: set[str],
) -> dict[str, dict[str, Any]]:
    if not isinstance(records, list) or len(records) != len(expected_paths):
        _fail(
            f"{role}.local_only_artifacts must contain exactly {len(expected_paths)} records"
        )
    observed: dict[str, dict[str, Any]] = {}
    for index, (value, expected_path) in enumerate(
        zip(records, expected_paths, strict=True)
    ):
        record, relative, _ = _strict_workspace_record(
            value,
            root=root,
            label=f"{role}.local_only_artifacts[{index}]",
        )
        if relative != expected_path:
            _fail(
                f"{role}.local_only_artifacts is not the exact canonical local inventory"
            )
        if relative in included:
            _fail(
                f"{role} local-only artifact is present in the public allowlist: {relative}"
            )
        observed[relative.casefold()] = record
    return observed


def _compare_replayed_fields(
    role: str,
    persisted: Mapping[str, Any],
    replayed: Mapping[str, Any],
    fields: Sequence[str],
) -> None:
    for field in fields:
        if field not in persisted or field not in replayed:
            _fail(f"{role} replay comparison is missing field {field}")
        if persisted[field] != replayed[field]:
            _fail(f"{role} persisted evidence differs from replay for {field}")


def _without_volatile_fields(
    value: Mapping[str, Any], *, fields: frozenset[str]
) -> dict[str, Any]:
    normalized = json.loads(
        json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True)
    )
    for field in fields:
        normalized.pop(field, None)
    return normalized


def _require_long_protein_independent_report(
    report: Mapping[str, Any],
    *,
    trusted_completion_sha256: str,
    run_directory: str,
    completion_record: Mapping[str, Any],
) -> None:
    claimed_body_sha256 = report.get("verification_body_sha256")
    if not isinstance(claimed_body_sha256, str) or not SHA256_RE.fullmatch(
        claimed_body_sha256
    ):
        _fail("long-protein independent report body digest is invalid")
    body = dict(report)
    del body["verification_body_sha256"]
    observed_body_sha256 = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if observed_body_sha256 != claimed_body_sha256:
        _fail("long-protein independent report body digest does not verify")
    expected_contract = {
        "analysis_schema": "DepPLM-long-protein-analysis-v3",
        "run_manifest_schema": "DepPLM-long-protein-analysis-run-v3",
        "completion_schema": "DepPLM-long-protein-analysis-completion-v3",
        "current": True,
    }
    gates = report.get("verification_gates")
    verified_analysis = report.get("verified_analysis")
    completion = (
        verified_analysis.get("completion")
        if isinstance(verified_analysis, Mapping)
        else None
    )
    if (
        report.get("schema") != "DepPLM-long-protein-independent-verification-v2"
        or report.get("verification_passed") is not True
        or report.get("verification_status") != "passed"
        or report.get("release_ready") is not True
        or report.get("scientific_result") is not False
        or report.get("analysis_contract") != expected_contract
        or not isinstance(gates, Mapping)
        or gates.get("analysis_contract_current") is not True
        or not gates
        or any(value is not True for value in gates.values())
        or not isinstance(verified_analysis, Mapping)
        or set(verified_analysis)
        != {
            "run_directory",
            "trusted_completion_sha256",
            "completion",
            "run_manifest",
            "summary",
            "per_gene_errors",
        }
        or verified_analysis.get("run_directory") != run_directory
        or verified_analysis.get("trusted_completion_sha256")
        != trusted_completion_sha256
        or not isinstance(completion, Mapping)
        or dict(completion) != dict(completion_record)
        or completion.get("sha256") != trusted_completion_sha256
    ):
        _fail("long-protein independent report is not v3 release eligible")


def _require_common_checkpoint_replay(report: Mapping[str, Any]) -> None:
    """Reject legacy comparator reports that do not replay saved checkpoints."""
    if report.get("schema") != COMMON_COHORT_REPLAY_SCHEMA:
        _fail(
            "common-cohort verification must use the v3 source-reconstructed "
            "checkpoint/inference replay schema; legacy verification is insufficient"
        )
    if report.get("valid") is not True or report.get("errors") != []:
        _fail("common-cohort checkpoint replay report is not valid")
    if report.get("scientific_result") is not False:
        _fail("common-cohort comparison must remain a non-scientific result")
    if report.get("completed_folds") != list(range(5)):
        _fail("common-cohort report must contain exactly folds 0 through 4")
    if report.get("outer_test_exactly_once") is not True:
        _fail("common-cohort outer-test rows are not exact-once")
    if report.get("predictions_exact_once_and_hash_bound") is not True:
        _fail("common-cohort predictions are not exact-once and hash-bound")
    if report.get("checkpoint_replay_all_folds") is not True:
        _fail("common-cohort checkpoint replay did not pass for every fold")
    if report.get("replayed_folds") != list(range(5)):
        _fail("common-cohort checkpoint replay must cover exactly folds 0 through 4")
    if report.get("elementwise_tolerance") != COMMON_COHORT_REPLAY_TOLERANCE:
        _fail("common-cohort checkpoint replay tolerance differs")
    if report.get("replay_inputs_embedded") is not False:
        _fail("common-cohort checkpoints permit embedded cell-level replay inputs")
    if report.get("replay_input_policy") != COMMON_COHORT_REPLAY_INPUT_POLICY:
        _fail("common-cohort replay-input reconstruction policy differs")
    if (
        report.get("checkpoint_preprocessing_contains_raw_cell_level_values")
        is not False
    ):
        _fail("common-cohort preprocessing permits raw cell-level values")
    if (
        report.get("checkpoint_preprocessing_contains_source_derived_statistics")
        is not True
    ):
        _fail("common-cohort preprocessing source-derived disclosure is missing")
    if (
        report.get("checkpoint_preprocessing_content_policy")
        != COMMON_COHORT_PREPROCESSING_POLICY
    ):
        _fail("common-cohort preprocessing content policy differs")
    if (
        report.get("generated_artifact_redistribution_status")
        != COMMON_COHORT_REDISTRIBUTION_STATUS
    ):
        _fail("common-cohort generated-artifact redistribution status differs")
    source_inventory_sha = report.get("source_input_inventory_sha256")
    if not isinstance(source_inventory_sha, str) or not SHA256_RE.fullmatch(
        source_inventory_sha
    ):
        _fail("common-cohort source-input inventory hash is invalid")

    fold_replay = report.get("fold_replay")
    if not isinstance(fold_replay, Mapping) or list(fold_replay) != [
        f"fold-{fold:02d}" for fold in range(5)
    ]:
        _fail("common-cohort checkpoint replay inventory is not the exact five folds")
    required_hashes = {
        "checkpoint_sha256",
        "model_state_sha256",
        "prediction_sha256",
        "preprocessing_state_sha256",
        "reconstructed_inputs_sha256",
        "source_input_inventory_sha256",
    }
    for fold, (key, replay) in enumerate(fold_replay.items()):
        if key != f"fold-{fold:02d}" or not isinstance(replay, Mapping):
            _fail("common-cohort checkpoint replay fold identity is invalid")
        if replay.get("allclose") is not True:
            _fail(f"common-cohort checkpoint replay differs for {key}")
        if replay.get("replay_inputs_embedded") is not False:
            _fail(f"common-cohort checkpoint embeds replay inputs for {key}")
        if replay.get("replay_input_policy") != COMMON_COHORT_REPLAY_INPUT_POLICY:
            _fail(f"common-cohort replay-input policy differs for {key}")
        if replay.get("preprocessing_reconstructed_from_current_sources") is not True:
            _fail(f"common-cohort preprocessing was not reconstructed for {key}")
        if replay.get("elementwise_tolerance") != COMMON_COHORT_REPLAY_TOLERANCE:
            _fail(f"common-cohort checkpoint replay tolerance differs for {key}")
        for field in required_hashes:
            value = replay.get(field)
            if (
                not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{64}", value) is None
            ):
                _fail(f"common-cohort checkpoint replay has invalid {field} for {key}")
        if replay["source_input_inventory_sha256"] != source_inventory_sha:
            _fail(f"common-cohort source-input inventory differs for {key}")
        n_elements = replay.get("n_elements_compared")
        if (
            not isinstance(n_elements, int)
            or isinstance(n_elements, bool)
            or n_elements <= 0
        ):
            _fail(f"common-cohort checkpoint replay count is invalid for {key}")
        for field in ("max_abs_error", "max_rel_error"):
            value = replay.get(field)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                _fail(f"common-cohort checkpoint replay has invalid {field} for {key}")


def _require_public_common_replay_bundle(
    bindings: Sequence[tuple[str, Path, Path, dict[str, Any]]],
) -> NoReturn:
    """Keep non-redistributable common-cohort artifacts out of a public archive."""
    missing: list[str] = []
    for condition, run_dir, _, report in bindings:
        if (
            report.get("generated_artifact_redistribution_status")
            != COMMON_COHORT_REDISTRIBUTION_STATUS
        ):
            _fail(
                f"{condition} common-cohort report has an unregistered public "
                "redistribution status"
            )
        for fold in range(5):
            checkpoint = run_dir / f"fold-{fold:02d}-checkpoint.pt"
            if not checkpoint.is_file():
                missing.append(f"{condition}/fold-{fold:02d}")
    if missing:
        _fail(
            "public common-cohort replay material is incomplete; the private "
            "checkpoint inventory is missing: " + ", ".join(missing)
        )
    _fail(
        "public archive is blocked: current common-cohort v3 checkpoints exclude "
        "embedded cell-level replay inputs, but their model weights and "
        "training-fitted preprocessing statistics, together with generated "
        "predictions/results, are source-derived and explicitly not cleared for "
        "public redistribution. They must not be allowlisted until author and "
        "data-license review approves a registered public-safe replay contract."
    )


def _enforce_required_content(categories: Mapping[str, str]) -> None:
    for category, required in REQUIRED_CONTENT_FILES.items():
        missing = sorted(
            relative for relative in required if categories.get(relative) != category
        )
        if missing:
            _fail(
                f"release omits required {category} files or assigns the wrong "
                f"category: {missing}"
            )
        if category in {"source", "test"}:
            unexpected = sorted(
                relative
                for relative, assigned_category in categories.items()
                if assigned_category == category and relative not in required
            )
            if unexpected:
                _fail(
                    f"release contains unreviewed {category} files outside the exact "
                    f"inventory: {unexpected}"
                )


def _strip_tex_comments(text: str) -> str:
    stripped: list[str] = []
    for line in text.splitlines():
        comment_at: int | None = None
        for index, character in enumerate(line):
            if character != "%":
                continue
            preceding_backslashes = 0
            cursor = index - 1
            while cursor >= 0 and line[cursor] == "\\":
                preceding_backslashes += 1
                cursor -= 1
            if preceding_backslashes % 2 == 0:
                comment_at = index
                break
        stripped.append(line if comment_at is None else line[:comment_at])
    return "\n".join(stripped)


def _manuscript_dependency_closure(
    root: Path,
    categories: Mapping[str, str],
) -> set[str]:
    """Resolve repository-local dependencies reachable from the active main TeX."""
    manuscript_prefix = PurePosixPath("paper/nar_major_revision")
    main_relative = (manuscript_prefix / "main.tex").as_posix()
    main_path = _safe_workspace_file(root, main_relative, field="active manuscript")
    main_payload, _ = _read_stable_bytes(main_path, label="active manuscript")
    try:
        main_text = _strip_tex_comments(main_payload.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ReleaseValidationError("active manuscript is not UTF-8") from exc
    tex_control = re.escape(chr(92))
    graphic_match = re.search(
        rf"{tex_control}graphicspath\s*\{{\s*((?:\{{[^{{}}]+\}}\s*)+)\}}",
        main_text,
    )
    graphic_paths = (
        re.findall(r"\{([^{}]+)\}", graphic_match.group(1)) if graphic_match else []
    )
    command_re = re.compile(
        r"\\(?P<command>LoadClassWithOptions|RequirePackageWithOptions|"
        r"RequirePackage|documentclass|LoadClass|usepackage|includegraphics|"
        r"bibliographystyle|bibliography|addbibresource|subfile|include|input)"
        r"\s*(?:\[[^\]]*\]\s*)?\{(?P<target>[^{}]+)\}"
    )
    import_re = re.compile(
        r"\\(?P<command>subincludefrom|subinputfrom|includefrom|inputfrom|"
        r"subimport|import)\s*\*?\s*(?:\[[^\]]*\]\s*)?"
        r"\{(?P<directory>[^{}]+)\}\s*\{(?P<target>[^{}]+)\}"
    )
    dependencies: set[str] = set()
    visited: set[tuple[str, str]] = set()
    pending = [(main_relative, manuscript_prefix)]
    while pending:
        relative, active_import_base = pending.pop()
        state = (relative, active_import_base.as_posix())
        if state in visited:
            continue
        visited.add(state)
        if categories.get(relative) != "manuscript":
            _fail(f"active manuscript dependency is not allowlisted: {relative}")
        path = _safe_workspace_file(
            root, relative, field="active manuscript dependency"
        )
        payload, _ = _read_stable_bytes(path, label=f"manuscript dependency {relative}")
        dependencies.add(relative)
        source_suffix = Path(relative).suffix.casefold()
        if source_suffix not in {".tex", ".cls", ".sty"}:
            continue
        try:
            text = _strip_tex_comments(payload.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ReleaseValidationError(
                f"active manuscript dependency is not UTF-8: {relative}"
            ) from exc
        references: list[tuple[int, str, str | None, str]] = [
            (match.start(), match.group("command"), None, match.group("target"))
            for match in command_re.finditer(text)
        ]
        references.extend(
            (
                match.start(),
                match.group("command"),
                match.group("directory"),
                match.group("target"),
            )
            for match in import_re.finditer(text)
        )
        for _, command, raw_directory, raw_targets in sorted(references):
            targets = (
                raw_targets.split(",")
                if command
                in {
                    "bibliography",
                    "RequirePackage",
                    "RequirePackageWithOptions",
                    "usepackage",
                }
                else [raw_targets]
            )
            for raw_target in targets:
                target = raw_target.strip().replace("\\", "/")
                pure_target = PurePosixPath(target)
                if (
                    not target
                    or pure_target.is_absolute()
                    or any(part in {"", ".", ".."} for part in pure_target.parts)
                    or ":" in target
                ):
                    _fail(f"active manuscript has an unsafe dependency: {target!r}")
                suffixes: tuple[str, ...]
                prefixes: tuple[PurePosixPath, ...]
                optional_external = False
                child_import_base = active_import_base
                if command in {
                    "input",
                    "include",
                    "subfile",
                }:
                    suffixes = ("",) if pure_target.suffix else (".tex",)
                    prefixes = (active_import_base,)
                    optional_external = source_suffix in {".cls", ".sty"}
                elif command in {
                    "import",
                    "subimport",
                    "inputfrom",
                    "subinputfrom",
                    "includefrom",
                    "subincludefrom",
                }:
                    directory = (raw_directory or "").strip().replace("\\", "/")
                    pure_directory = PurePosixPath(directory.rstrip("/"))
                    if (
                        not directory
                        or pure_directory.is_absolute()
                        or any(part in {"", ".", ".."} for part in pure_directory.parts)
                        or ":" in directory
                    ):
                        _fail(
                            "active manuscript has an unsafe import directory: "
                            f"{directory!r}"
                        )
                    if command in {"subimport", "subinputfrom", "subincludefrom"}:
                        child_import_base = active_import_base / pure_directory
                    else:
                        child_import_base = manuscript_prefix / pure_directory
                    suffixes = ("",) if pure_target.suffix else (".tex",)
                    prefixes = (child_import_base,)
                    optional_external = source_suffix in {".cls", ".sty"}
                elif command == "includegraphics":
                    suffixes = ("",) if pure_target.suffix else (".pdf", ".png")
                    prefixes = tuple(
                        manuscript_prefix / PurePosixPath(item.rstrip("/"))
                        for item in graphic_paths
                    ) + (active_import_base,)
                elif command in {"bibliography", "addbibresource"}:
                    suffixes = ("",) if pure_target.suffix else (".bib",)
                    prefixes = (manuscript_prefix,)
                elif command == "bibliographystyle":
                    suffixes = ("",) if pure_target.suffix else (".bst",)
                    prefixes = (manuscript_prefix,)
                elif command in {
                    "RequirePackage",
                    "RequirePackageWithOptions",
                    "usepackage",
                }:
                    suffixes = ("",) if pure_target.suffix else (".sty",)
                    prefixes = (manuscript_prefix,)
                    optional_external = True
                else:
                    suffixes = ("",) if pure_target.suffix else (".cls",)
                    prefixes = (manuscript_prefix,)
                    optional_external = command in {
                        "LoadClass",
                        "LoadClassWithOptions",
                    }
                candidates = [
                    (prefix / f"{target}{suffix}").as_posix()
                    for prefix in prefixes
                    for suffix in suffixes
                ]
                resolved = next(
                    (
                        candidate
                        for candidate in candidates
                        if (root / Path(candidate)).is_file()
                    ),
                    None,
                )
                if resolved is None:
                    if optional_external:
                        continue
                    _fail(
                        "active manuscript dependency is missing: "
                        f"{target!r} from {relative}"
                    )
                if categories.get(resolved) != "manuscript":
                    _fail(
                        f"active manuscript dependency is not allowlisted: {resolved}"
                    )
                if Path(resolved).suffix.casefold() in {".tex", ".cls", ".sty"}:
                    pending.append((resolved, child_import_base))
                else:
                    dependencies.add(resolved)
    return dependencies


def _authorize_verified_file(
    path: Path,
    *,
    root: Path,
    included: set[str],
    authorized: dict[str, set[str]],
    category: str,
    label: str,
) -> str:
    try:
        relative = (
            path.resolve(strict=True).relative_to(root.resolve(strict=True)).as_posix()
        )
    except (OSError, ValueError) as exc:
        raise ReleaseValidationError(
            f"{label} escapes the workspace or is missing: {path}"
        ) from exc
    _safe_workspace_file(root, relative, field=label)
    _require_included(relative, included, label=label)
    _authorize(authorized, relative, category)
    return relative


def _authorize_attention_artifacts(
    output_root: Path,
    *,
    root: Path,
    included: set[str],
    authorized: dict[str, set[str]],
) -> None:
    completion_paths = sorted(
        (output_root / "canonical").glob("**/attempt-*/completion.json")
    )
    if len(completion_paths) != 15:
        _fail("attention verifier must resolve exactly 15 completed canonical attempts")
    expected = {(seed, fold) for seed in (42, 43, 44) for fold in range(5)}
    observed: set[tuple[int, int]] = set()
    artifact_names = (
        "completion.json",
        "run-manifest.json",
        "training.json",
        "inference.json",
        "metrics.json",
        "trained-model.pt",
        "outer-test-predictions.npz",
        "attention-evidence.npz",
    )
    for completion_path in completion_paths:
        attempt = completion_path.parent
        manifest = _load_json(
            attempt / "run-manifest.json", label="attention run manifest"
        )
        seed = manifest.get("seed")
        fold = manifest.get("fold")
        if type(seed) is not int or type(fold) is not int:
            _fail("attention attempt seed and fold must be integers")
        key = (seed, fold)
        if key not in expected or key in observed or manifest.get("pilot") is not False:
            _fail("attention attempt inventory is not the exact canonical 3x5 grid")
        observed.add(key)
        for name in artifact_names:
            _authorize_verified_file(
                attempt / name,
                root=root,
                included=included,
                authorized=authorized,
                category=(
                    "prediction" if name == "outer-test-predictions.npz" else "evidence"
                ),
                label=f"attention verified artifact {name}",
            )
    if observed != expected:
        _fail("attention attempt inventory is not the exact canonical 3x5 grid")


def _authorize_cnv_artifacts(
    run_dir: Path,
    *,
    root: Path,
    included: set[str],
    authorized: dict[str, set[str]],
) -> None:
    expected_root = (root / "results" / "cnv_sensitivity_v1").resolve(strict=True)
    try:
        run_dir.resolve(strict=True).relative_to(expected_root)
    except (OSError, ValueError) as exc:
        raise ReleaseValidationError(
            "CNV verification is outside results/cnv_sensitivity_v1"
        ) from exc
    for name in ("run-manifest.json", "completion.json", "results.json"):
        _authorize_verified_file(
            run_dir / name,
            root=root,
            included=included,
            authorized=authorized,
            category="evidence",
            label=f"CNV verified artifact {name}",
        )
    results = _load_json(run_dir / "results.json", label="CNV results")
    pairs = results.get("paired_results")
    if not isinstance(pairs, list) or len(pairs) != 15:
        _fail("CNV results must contain exactly 15 verified seed/fold pairs")
    seen_paths: set[str] = set()
    for pair_index, pair in enumerate(pairs):
        if not isinstance(pair, Mapping):
            _fail(f"CNV paired_results[{pair_index}] is invalid")
        records = (
            (pair.get("split_manifest"), "evidence", "split"),
            (
                (
                    pair.get("baseline", {}).get("outer_test_predictions")
                    if isinstance(pair.get("baseline"), Mapping)
                    else None
                ),
                "prediction",
                "baseline prediction",
            ),
            (
                (
                    pair.get("cnv", {}).get("outer_test_predictions")
                    if isinstance(pair.get("cnv"), Mapping)
                    else None
                ),
                "prediction",
                "CNV prediction",
            ),
        )
        for record, category, name in records:
            if not isinstance(record, Mapping):
                _fail(f"CNV paired_results[{pair_index}] lacks {name}")
            relative, _ = _record_path(
                record,
                root=root,
                base=run_dir,
                label=f"CNV paired_results[{pair_index}] {name}",
            )
            if relative.casefold() in seen_paths:
                _fail("CNV artifact inventory repeats a verifier input")
            seen_paths.add(relative.casefold())
            _require_included(relative, included, label=f"CNV {name}")
            _authorize(authorized, relative, category)


def _comparison_bindings(
    payload: Mapping[str, Any], *, root: Path
) -> list[tuple[str, Path, Path, dict[str, Any]]]:
    raw = payload.get("comparison_verifications")
    if not isinstance(raw, list) or len(raw) != 2:
        _fail("external baseline must bind exactly two common-cohort verifications")
    bindings: list[tuple[str, Path, Path, dict[str, Any]]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping) or set(item) != {
            "condition",
            "run_directory",
            "verification",
        }:
            _fail(f"external comparison_verifications[{index}] is invalid")
        condition = item.get("condition")
        run_relative = _relative_path(
            item.get("run_directory"),
            field=f"external comparison_verifications[{index}].run_directory",
        )
        run_dir = (root / Path(run_relative)).resolve(strict=True)
        expected_root = (root / "results").resolve(strict=True)
        try:
            run_dir.relative_to(expected_root)
        except ValueError as exc:
            raise ReleaseValidationError(
                "external common-cohort run escapes results/"
            ) from exc
        if run_dir.is_symlink() or not run_dir.is_dir():
            _fail("external common-cohort run is linked or not a directory")
        verification_record = item.get("verification")
        if not isinstance(verification_record, Mapping):
            _fail("external common-cohort verification record is missing")
        _, verification_path = _record_path(
            verification_record,
            root=root,
            base=run_dir,
            label=f"external comparison_verifications[{index}].verification",
        )
        report = _load_json(
            verification_path, label=f"external {condition} verification"
        )
        if report.get("condition") != condition:
            _fail("external comparison condition differs from its verification")
        bindings.append((str(condition), run_dir, verification_path, report))
    if [item[0] for item in bindings] != ["DepPLM", "C-PLM-MLP"]:
        _fail("external comparison verifications must be ordered DepPLM, C-PLM-MLP")
    return bindings


def _authorize_external_artifacts(
    gatdep_run_dir: Path,
    bindings: Sequence[tuple[str, Path, Path, dict[str, Any]]],
    *,
    root: Path,
    included: set[str],
    authorized: dict[str, set[str]],
) -> None:
    expected_root = (root / "results" / "gatdep_official_v1").resolve(strict=True)
    try:
        gatdep_run_dir.resolve(strict=True).relative_to(expected_root)
    except (OSError, ValueError) as exc:
        raise ReleaseValidationError(
            "GATDep verification is outside results/gatdep_official_v1"
        ) from exc
    _require_public_common_replay_bundle(bindings)
    gatdep_names = ["run-manifest.json", "completion.json"]
    for fold in range(5):
        prefix = f"fold-{fold:02d}"
        gatdep_names.extend(
            [
                f"{prefix}-split.json",
                f"{prefix}-pretest.json",
                f"{prefix}-model.pt",
                f"{prefix}-test-intent.json",
                f"{prefix}-predictions.npz",
                f"{prefix}-predictions.index.json",
                f"{prefix}-result.json",
            ]
        )
    for name in gatdep_names:
        _authorize_verified_file(
            gatdep_run_dir / name,
            root=root,
            included=included,
            authorized=authorized,
            category="prediction" if name.endswith("-predictions.npz") else "evidence",
            label=f"GATDep verified artifact {name}",
        )

    common_names = [
        "run-manifest.json",
        "cohort-manifest.json",
        "summary.json",
        "completion.json",
    ]
    for fold in range(5):
        prefix = f"fold-{fold:02d}"
        common_names.extend(
            [
                f"{prefix}-split.json",
                f"{prefix}-result.json",
                f"{prefix}-predictions.npz",
            ]
        )
    for condition, run_dir, verification_path, _ in bindings:
        _authorize_verified_file(
            verification_path,
            root=root,
            included=included,
            authorized=authorized,
            category="evidence",
            label=f"{condition} common-cohort verification",
        )
        for name in common_names:
            _authorize_verified_file(
                run_dir / name,
                root=root,
                included=included,
                authorized=authorized,
                category=(
                    "prediction" if name.endswith("-predictions.npz") else "evidence"
                ),
                label=f"{condition} common-cohort artifact {name}",
            )


def _replay_k_token_pmi_campaign(
    *,
    path: Path,
    root: Path,
    payload: Mapping[str, Any],
    trusted_modules: _TrustedWorkspaceModuleSession,
) -> None:
    module = trusted_modules.load("run_k_token_pmi_sensitivity")
    artifact_root = (root / "results" / "k_token_pmi_v1").resolve(strict=True)
    campaign_dir = path.parent.resolve(strict=True)
    try:
        campaign_dir.relative_to(artifact_root)
    except ValueError as exc:
        raise ReleaseValidationError(
            "k_token_pmi completion is outside results/k_token_pmi_v1"
        ) from exc
    expected_gates = {
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
    with tempfile.TemporaryDirectory(
        prefix="depplm-k-token-pmi-public-replay-"
    ) as raw_report_dir:
        report_path = Path(raw_report_dir) / "verification.json"
        receipt_path = Path(raw_report_dir) / "detached-receipt.json"
        audit_path = Path(raw_report_dir) / "independent-audit.json"
        try:
            replay_result = module.verify_campaign(
                campaign_dir,
                artifact_root=artifact_root,
                output_path=report_path,
            )
        except Exception as exc:
            raise ReleaseValidationError(
                "K/token/PMI full campaign replay raised an error: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(replay_result, tuple) or len(replay_result) != 2:
            _fail("K/token/PMI full campaign replay returned an invalid result")
        returned_path, report = replay_result
        if not isinstance(returned_path, (str, os.PathLike)):
            _fail("K/token/PMI full campaign replay returned an invalid report path")
        try:
            returned_resolved = Path(returned_path).resolve(strict=True)
            requested_resolved = report_path.resolve(strict=True)
        except OSError as exc:
            raise ReleaseValidationError(
                "K/token/PMI full campaign replay report is unavailable"
            ) from exc
        if returned_resolved != requested_resolved:
            _fail("K/token/PMI full campaign replay wrote an unexpected report path")
        persisted_report = _load_json(
            requested_resolved,
            label="K/token/PMI full campaign replay report",
        )
        if not isinstance(report, Mapping) or not module._strict_json_equal(
            persisted_report, report
        ):
            _fail("K/token/PMI replay return value differs from its persisted report")
        gates = report.get("verification_gates")
        if (
            not isinstance(gates, Mapping)
            or set(gates) != expected_gates
            or any(value is not True for value in gates.values())
        ):
            _fail("K/token/PMI full campaign replay did not pass every gate")
        expected_scalars = {
            "schema": module.VERIFICATION_SCHEMA,
            "campaign_directory": str(campaign_dir),
            "artifact_root": str(artifact_root),
            "protocol_complete": True,
            "scientific_result": True,
            "n_verified_variant_seed_jobs": 21,
            "n_verified_outer_fold_evaluations": 105,
            "n_verified_prediction_and_sidecar_artifacts": 210,
        }
        for field, expected in expected_scalars.items():
            if not module._strict_json_equal(report.get(field), expected):
                _fail(
                    "K/token/PMI full campaign replay returned a non-canonical "
                    f"{field}"
                )
        if report.get("errors") != []:
            _fail("K/token/PMI full campaign replay reported errors")
        completion_binding = report.get("campaign_completion")
        if not isinstance(completion_binding, Mapping):
            _fail("K/token/PMI replay report lacks its completion binding")
        _record_matches_file(
            path,
            completion_binding,
            label="K/token/PMI replayed campaign completion",
        )
        current_payload = _load_json(path, label="K/token/PMI completion")
        if not module._strict_json_equal(current_payload, payload):
            _fail("K/token/PMI selected completion changed around full replay")

        try:
            receipt_result = module._write_detached_verification_receipt(
                report_path=requested_resolved,
                report=report,
                campaign_dir=campaign_dir,
                artifact_root=artifact_root,
                receipt_path=receipt_path,
            )
        except Exception as exc:
            raise ReleaseValidationError(
                "K/token/PMI detached replay receipt creation raised an error: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(receipt_result, tuple) or len(receipt_result) != 3:
            _fail("K/token/PMI detached replay receipt returned an invalid result")
        returned_receipt_path, receipt, receipt_binding = receipt_result
        if not isinstance(returned_receipt_path, (str, os.PathLike)):
            _fail("K/token/PMI detached replay receipt returned an invalid path")
        try:
            returned_receipt_resolved = Path(returned_receipt_path).resolve(strict=True)
            requested_receipt_resolved = receipt_path.resolve(strict=True)
        except OSError as exc:
            raise ReleaseValidationError(
                "K/token/PMI detached replay receipt is unavailable"
            ) from exc
        if returned_receipt_resolved != requested_receipt_resolved:
            _fail("K/token/PMI replay wrote an unexpected detached receipt path")
        persisted_receipt = _load_json(
            requested_receipt_resolved,
            label="K/token/PMI detached replay receipt",
        )
        if not isinstance(receipt, Mapping) or not module._strict_json_equal(
            persisted_receipt, receipt
        ):
            _fail("K/token/PMI detached receipt return value differs from disk")
        if not isinstance(receipt_binding, Mapping) or set(receipt_binding) != {
            "path",
            "bytes",
            "mtime_ns",
            "sha256",
        }:
            _fail("K/token/PMI detached receipt lacks its file binding")
        receipt_binding_path = receipt_binding.get("path")
        if not isinstance(receipt_binding_path, str):
            _fail("K/token/PMI detached receipt binding has an invalid path")
        try:
            bound_receipt_resolved = Path(receipt_binding_path).resolve(strict=True)
        except OSError as exc:
            raise ReleaseValidationError(
                "K/token/PMI detached receipt binding path is unavailable"
            ) from exc
        if bound_receipt_resolved != requested_receipt_resolved:
            _fail("K/token/PMI detached receipt binding names an unexpected path")
        _record_matches_file(
            requested_receipt_resolved,
            receipt_binding,
            label="K/token/PMI detached replay receipt",
        )

        independent = trusted_modules.load("verify_k_token_pmi_sensitivity")
        try:
            audit_result = independent.verify(
                requested_receipt_resolved,
                requested_resolved,
                output_path=audit_path,
            )
        except Exception as exc:
            raise ReleaseValidationError(
                "K/token/PMI implementation-independent audit raised an error: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(audit_result, tuple) or len(audit_result) != 3:
            _fail("K/token/PMI independent audit returned an invalid result")
        returned_audit_path, audit, audit_binding = audit_result
        if not isinstance(returned_audit_path, (str, os.PathLike)):
            _fail("K/token/PMI independent audit returned an invalid report path")
        try:
            returned_audit_resolved = Path(returned_audit_path).resolve(strict=True)
            requested_audit_resolved = audit_path.resolve(strict=True)
        except OSError as exc:
            raise ReleaseValidationError(
                "K/token/PMI independent audit report is unavailable"
            ) from exc
        if returned_audit_resolved != requested_audit_resolved:
            _fail("K/token/PMI independent audit wrote an unexpected report path")
        persisted_audit = _load_json(
            requested_audit_resolved,
            label="K/token/PMI implementation-independent audit",
        )
        if not isinstance(audit, Mapping) or not module._strict_json_equal(
            persisted_audit, audit
        ):
            _fail("K/token/PMI independent audit return value differs from disk")
        if not isinstance(audit_binding, Mapping) or set(audit_binding) != {
            "path",
            "bytes",
            "mtime_ns",
            "sha256",
        }:
            _fail("K/token/PMI independent audit lacks its file binding")
        audit_binding_path = audit_binding.get("path")
        if not isinstance(audit_binding_path, str):
            _fail("K/token/PMI independent audit binding has an invalid path")
        try:
            bound_audit_resolved = Path(audit_binding_path).resolve(strict=True)
        except OSError as exc:
            raise ReleaseValidationError(
                "K/token/PMI independent audit binding path is unavailable"
            ) from exc
        if bound_audit_resolved != requested_audit_resolved:
            _fail("K/token/PMI independent audit binding names an unexpected path")
        _record_matches_file(
            requested_audit_resolved,
            audit_binding,
            label="K/token/PMI implementation-independent audit",
        )

        audit_gates = audit.get("verification_gates")
        if (
            not isinstance(audit_gates, Mapping)
            or set(audit_gates) != K_TOKEN_PMI_INDEPENDENT_GATES
            or any(value is not True for value in audit_gates.values())
        ):
            _fail("K/token/PMI independent audit did not pass every gate")
        expected_audit_scalars = {
            "schema": independent.AUDIT_SCHEMA,
            "audit_mode": "implementation_independent_read_only_core_replay",
            "artifact_root": str(artifact_root),
            "campaign_directory": str(campaign_dir),
            "audit_passed": True,
            "protocol_complete": True,
            "scientific_result": False,
            "n_verified_variant_seed_jobs": 21,
            "n_verified_outer_fold_evaluations": 105,
            "n_verified_prediction_and_sidecar_artifacts": 210,
        }
        for field, expected in expected_audit_scalars.items():
            if not module._strict_json_equal(audit.get(field), expected):
                _fail(
                    "K/token/PMI independent audit returned a non-canonical "
                    f"{field}"
                )
        if audit.get("errors") != []:
            _fail("K/token/PMI independent audit reported errors")
        if not isinstance(audit.get("limitations"), list) or not audit["limitations"]:
            _fail("K/token/PMI independent audit omitted its limitations")
        core_evidence = audit.get("outer_fold_core_evidence")
        if not isinstance(core_evidence, list) or len(core_evidence) != 105:
            _fail("K/token/PMI independent audit has incomplete core evidence")
        for field, expected_path in (
            ("detached_receipt", requested_receipt_resolved),
            ("primary_verification_report", requested_resolved),
        ):
            binding = audit.get(field)
            if not isinstance(binding, Mapping) or set(binding) != {
                "path",
                "bytes",
                "mtime_ns",
                "sha256",
            }:
                _fail(f"K/token/PMI independent audit lacks {field} binding")
            raw_binding_path = binding.get("path")
            if not isinstance(raw_binding_path, str):
                _fail(f"K/token/PMI independent audit {field} path is invalid")
            try:
                resolved_binding_path = Path(raw_binding_path).resolve(strict=True)
            except OSError as exc:
                raise ReleaseValidationError(
                    f"K/token/PMI independent audit {field} path is unavailable"
                ) from exc
            if resolved_binding_path != expected_path:
                _fail(f"K/token/PMI independent audit {field} path is unexpected")
            _record_matches_file(
                expected_path,
                binding,
                label=f"K/token/PMI independent audit {field}",
            )
        independent_source_path = Path(independent.SCRIPT_PATH).resolve(strict=True)
        expected_independent_source = (
            root / "code" / "verify_k_token_pmi_sensitivity.py"
        ).resolve(strict=True)
        if independent_source_path != expected_independent_source:
            _fail("K/token/PMI independent audit source path is unexpected")
        independent_source_binding = audit.get("independent_verifier_source")
        if not isinstance(independent_source_binding, Mapping) or set(
            independent_source_binding
        ) != {"path", "bytes", "mtime_ns", "sha256"}:
            _fail("K/token/PMI independent audit lacks its source binding")
        raw_source_binding_path = independent_source_binding.get("path")
        if not isinstance(raw_source_binding_path, str):
            _fail("K/token/PMI independent audit source binding path is invalid")
        try:
            resolved_source_binding_path = Path(raw_source_binding_path).resolve(
                strict=True
            )
        except OSError as exc:
            raise ReleaseValidationError(
                "K/token/PMI independent audit source binding path is unavailable"
            ) from exc
        if resolved_source_binding_path != independent_source_path:
            _fail("K/token/PMI independent audit source binding path is unexpected")
        _record_matches_file(
            independent_source_path,
            independent_source_binding,
            label="K/token/PMI independent verifier source",
        )

        final_report = _load_json(
            requested_resolved,
            label="K/token/PMI post-audit primary report",
        )
        final_receipt = _load_json(
            requested_receipt_resolved,
            label="K/token/PMI post-audit detached receipt",
        )
        final_payload = _load_json(path, label="K/token/PMI post-audit completion")
        if not module._strict_json_equal(final_report, report):
            _fail("K/token/PMI primary report changed during independent audit")
        if not module._strict_json_equal(final_receipt, receipt):
            _fail("K/token/PMI detached receipt changed during independent audit")
        if not module._strict_json_equal(final_payload, payload):
            _fail("K/token/PMI completion changed during independent audit")


def _require_registered_verification_with_trusted_modules(
    role: str,
    *,
    trusted_modules: _TrustedWorkspaceModuleSession,
    path: Path,
    root: Path,
    payload: Mapping[str, Any],
    gatdep_official_root: Path | None,
    included: set[str],
    authorized: dict[str, set[str]],
    trusted_completion_sha256: str | None = None,
    trusted_completion_bytes: int | None = None,
    trusted_freeze_sha256: str | None = None,
    trusted_prediction_lock_sha256: str | None = None,
) -> None:
    """Replay the role-specific verifier; persisted success flags never suffice."""
    try:
        if role == "core_campaign":
            module = trusted_modules.load("aggregate_corrected_campaign")
            results_root = (root / "results" / "corrected_v2").resolve(strict=True)
            try:
                relative = path.resolve(strict=True).relative_to(
                    root.resolve(strict=True)
                )
            except ValueError as exc:
                raise ReleaseValidationError(
                    "core campaign evidence is outside the workspace"
                ) from exc
            _require_core_campaign_publication_path(
                PurePosixPath(relative.as_posix()),
                field="core campaign evidence",
            )
            replayed = module.validate_campaign(results_root, workspace_root=root)
            volatile = frozenset({"created_utc"})
            if _without_volatile_fields(
                payload, fields=volatile
            ) != _without_volatile_fields(replayed, fields=volatile):
                _fail("core campaign differs from fresh 99-run validation replay")
            if (
                replayed.get("scientific_result") is not True
                or replayed.get("protocol_validation_passed") is not True
            ):
                _fail("core campaign fresh validation did not pass")
        elif role == "esm_embedding_provenance":
            module = trusted_modules.load("embedding_provenance")
            replayed = module.validate_embedding_bundle(root)
            manifest = replayed.get("manifest", {})
            if (
                not isinstance(manifest, Mapping)
                or manifest.get("sha256") != _sha256_file(path)
                or manifest.get("bytes") != path.stat().st_size
            ):
                _fail("ESM provenance replay is not bound to the selected manifest")
        elif role == "family_holdout":
            module = trusted_modules.load("verify_family_holdout_campaign")
            replayed = module.verify_campaign(path.parent)
            expected = {
                "status": "verified",
                "grid_cells_verified": 9,
                "folds_per_cell": 5,
                "child_campaigns_independently_verified": True,
                "predictions_exact_once_and_hash_bound": True,
                "repeat_aware_inference": "descriptive_only",
                "release_manifest_sha256": _sha256_file(path),
            }
            if any(replayed.get(field) != value for field, value in expected.items()):
                _fail("family-holdout independent 3x3 replay did not pass")
        elif role == "k_token_pmi":
            _replay_k_token_pmi_campaign(
                path=path,
                root=root,
                payload=payload,
                trusted_modules=trusted_modules,
            )
        elif role == "attention_faithfulness":
            module = trusted_modules.load("attention_faithfulness")
            output_root = (root / "results" / "attention_faithfulness_v1").resolve(
                strict=True
            )
            try:
                path.resolve(strict=True).relative_to(output_root)
            except ValueError as exc:
                raise ReleaseValidationError(
                    "attention evidence is outside results/attention_faithfulness_v1"
                ) from exc
            replayed = module.verify_campaign(
                output_root,
                pilot=False,
                write_summary=False,
            )
            _compare_replayed_fields(
                role,
                payload,
                replayed,
                (
                    "schema",
                    "scope",
                    "verified_seed_folds",
                    "expected_seed_folds",
                    "n_verified_folds",
                    "rejected_attempts",
                    "n_rejected_attempts",
                    "verification_passed",
                    "protocol_complete",
                    "eligible_for_scientific_aggregation",
                    "scientific_result",
                    "attention_partner_stability",
                ),
            )
            if replayed.get("rejected_attempts") != []:
                _fail("attention campaign contains rejected completed attempts")
            _authorize_attention_artifacts(
                output_root,
                root=root,
                included=included,
                authorized=authorized,
            )
        elif role == "cnv_sensitivity":
            module = trusted_modules.load("cnv_sensitivity")
            replayed = module.verify_cnv_run(path.parent)
            _compare_replayed_fields(
                role,
                payload,
                replayed,
                (
                    "schema",
                    "execution_complete",
                    "protocol_complete",
                    "artifacts_hash_verified",
                    "outer_test_metrics_recomputed",
                    "paired_targets_and_identities_identical",
                    "n_verified_seed_fold_pairs",
                    "input_files_hash_verified",
                    "embedding_provenance_verified",
                    "cnv_cohort_reloaded_and_verified",
                    "deterministic_splits_recomputed",
                    "scientific_result",
                    "result_sha256",
                    "summary",
                ),
            )
            _authorize_cnv_artifacts(
                path.parent,
                root=root,
                included=included,
                authorized=authorized,
            )
        elif role == "external_architecture_baseline":
            if gatdep_official_root is None:
                _fail(
                    "external_architecture_baseline requires an explicit "
                    "--gatdep-official-root for independent replay"
                )
            bindings = _comparison_bindings(payload, root=root)
            common_module = trusted_modules.load("verify_gatdep_common_depplm")
            module = trusted_modules.load("verify_gatdep_official_baseline")
            required_paths = _gatdep_snapshot_required_paths(common_module, module)
            with _stable_gatdep_official_snapshot(
                gatdep_official_root,
                required_paths,
            ) as official_root:
                comparison_paths: list[Path] = []
                comparator_means: dict[str, float] = {}
                comparator_seed: int | None = None
                for condition, run_dir, verification_path, persisted in bindings:
                    replayed_common = common_module.verify_run(
                        run_dir,
                        root,
                        official_root,
                    )
                    _require_common_checkpoint_replay(replayed_common)
                    if _without_volatile_fields(
                        persisted, fields=frozenset({"verified_at_utc"})
                    ) != _without_volatile_fields(
                        replayed_common, fields=frozenset({"verified_at_utc"})
                    ):
                        _fail(
                            f"{condition} common-cohort verification differs from replay"
                        )
                    if (
                        replayed_common.get("valid") is not True
                        or replayed_common.get("errors") != []
                        or replayed_common.get("predictions_exact_once_and_hash_bound")
                        is not True
                    ):
                        _fail(f"{condition} common-cohort verification did not pass")
                    seed = replayed_common.get("seed")
                    if not isinstance(seed, int) or isinstance(seed, bool):
                        _fail("external comparator seed is invalid")
                    if comparator_seed is None:
                        comparator_seed = seed
                    elif seed != comparator_seed:
                        _fail("external comparison runs use different seeds")
                    summary = _load_json(
                        run_dir / "summary.json",
                        label=f"{condition} common summary",
                    )
                    primary = summary.get("metrics", {}).get(
                        "residual_r2_vs_train_gene_mean", {}
                    )
                    mean = primary.get("mean") if isinstance(primary, Mapping) else None
                    if not isinstance(mean, (int, float)) or isinstance(mean, bool):
                        _fail(f"{condition} common-cohort primary metric is invalid")
                    comparator_means[condition] = float(mean)
                    comparison_paths.append(verification_path)
                replayed = module.verify_run(
                    path.parent,
                    root,
                    official_root,
                    comparison_manifests=comparison_paths,
                    allow_partial=False,
                    replay_device="cpu",
                )
                _compare_replayed_fields(
                    role,
                    payload,
                    replayed,
                    (
                        "schema",
                        "valid",
                        "errors",
                        "run_manifest_sha256",
                        "cohort_contract_sha256",
                        "outer_split_contract_sha256",
                        "completed_folds",
                        "outer_test_exactly_once",
                        "verified_metrics",
                        "checkpoint_replay",
                        "scientific_result",
                    ),
                )
                if replayed.get("valid") is not True or replayed.get("errors") != []:
                    _fail("GATDep common-cohort verification did not pass")
                gatdep_manifest = _load_json(
                    path.parent / "run-manifest.json", label="GATDep run manifest"
                )
                if gatdep_manifest.get("seed") != comparator_seed:
                    _fail("GATDep and common-cohort comparator seeds differ")
                metrics = replayed.get("verified_metrics")
                if not isinstance(metrics, Mapping) or list(metrics) != [
                    f"fold-{fold:02d}" for fold in range(5)
                ]:
                    _fail("GATDep verified metric inventory is incomplete")
                gatdep_values = [
                    metrics[f"fold-{fold:02d}"].get("residual_r2_vs_train_gene_mean")
                    for fold in range(5)
                ]
                if any(
                    not isinstance(value, (int, float)) or isinstance(value, bool)
                    for value in gatdep_values
                ):
                    _fail("GATDep verified primary fold metrics are invalid")
                gatdep_mean = float(sum(gatdep_values) / len(gatdep_values))
                expected_summary = {
                    "schema": "DepPLM-external-architecture-comparison-v1",
                    "primary_metric": "residual_r2_vs_train_gene_mean",
                    "cohort_contract_sha256": replayed["cohort_contract_sha256"],
                    "outer_split_contract_sha256": replayed[
                        "outer_split_contract_sha256"
                    ],
                    "seed": comparator_seed,
                    "models": [
                        {
                            "condition": "GATDep",
                            "mean": gatdep_mean,
                            "n_folds": 5,
                        },
                        {
                            "condition": "DepPLM",
                            "mean": comparator_means["DepPLM"],
                            "n_folds": 5,
                        },
                        {
                            "condition": "C-PLM-MLP",
                            "mean": comparator_means["C-PLM-MLP"],
                            "n_folds": 5,
                        },
                    ],
                    "mean_deltas_vs_gatdep": [
                        {
                            "condition": condition,
                            "delta": comparator_means[condition] - gatdep_mean,
                        }
                        for condition in ("DepPLM", "C-PLM-MLP")
                    ],
                    "interpretation": (
                        "Descriptive common-cohort comparison only; no inferential "
                        "significance claim is made."
                    ),
                }
                if payload.get("comparison_summary") != expected_summary:
                    _fail(
                        "external architecture comparison summary differs from replay"
                    )
            _authorize_external_artifacts(
                path.parent,
                bindings,
                root=root,
                included=included,
                authorized=authorized,
            )
        elif role == "clinical_annotation_concordance":
            if (
                not isinstance(trusted_completion_sha256, str)
                or not SHA256_RE.fullmatch(trusted_completion_sha256)
                or _sha256_file(path) != trusted_completion_sha256
            ):
                _fail("clinical completion differs from its selected digest")
            verification_record = payload.get("verification")
            if not isinstance(verification_record, Mapping):
                _fail("clinical completion verification record is invalid")
            _, verification_path = _record_path(
                verification_record,
                root=root,
                base=path.parent,
                label="clinical completion.verification",
            )
            stored_verification, _ = _load_json_with_fingerprint(
                verification_path,
                label="clinical verification",
                expected=verification_record,
            )
            if (
                set(stored_verification) != CLINICAL_VERIFICATION_FIELDS
                or stored_verification.get("schema") != CLINICAL_VERIFICATION_SCHEMA
                or stored_verification.get("scientific_result") is not True
                or stored_verification.get("verification_method")
                != CLINICAL_VERIFICATION_METHOD
                or not isinstance(stored_verification.get("analysis_sha256"), str)
                or not SHA256_RE.fullmatch(stored_verification["analysis_sha256"])
            ):
                _fail("clinical verification fields or semantics are invalid")
            module = trusted_modules.load("clinical_validation_v2")
            replayed = module.verify_formal_run(path.parent)
            marker_record = (
                replayed.get("marker_record")
                if isinstance(replayed, Mapping)
                else None
            )
            if (
                not isinstance(replayed, Mapping)
                or set(replayed) != CLINICAL_REPLAY_FIELDS
                or replayed.get("schema") != CLINICAL_VERIFICATION_SCHEMA
                or replayed.get("scientific_result") is not True
                or replayed.get("run_dir") != str(path.parent.resolve(strict=True))
                or replayed.get("verification_scope")
                != CLINICAL_REPLAY_VERIFICATION_SCOPE
                or replayed.get("path_identity_retained") is not False
                or not isinstance(marker_record, Mapping)
                or set(marker_record) != {"path", "bytes", "sha256"}
                or marker_record.get("path") != "completion.json"
                or type(marker_record.get("bytes")) is not int
                or type(trusted_completion_bytes) is not int
                or marker_record.get("bytes") != trusted_completion_bytes
                or marker_record.get("sha256") != trusted_completion_sha256
                or replayed.get("schema") != stored_verification.get("schema")
                or replayed.get("scientific_result")
                != stored_verification.get("scientific_result")
                or replayed.get("analysis_sha256")
                != stored_verification.get("analysis_sha256")
                or type(replayed.get("campaign_source_runs_rehashed")) is not int
                or replayed.get("campaign_source_runs_rehashed")
                != CLINICAL_EXPECTED_SOURCE_RUNS
                or type(replayed.get("campaign_source_artifacts_rehashed")) is not int
                or replayed.get("campaign_source_artifacts_rehashed")
                != CLINICAL_EXPECTED_SOURCE_ARTIFACTS
            ):
                _fail("clinical annotation independent replay did not pass")
            if _sha256_file(path) != trusted_completion_sha256:
                _fail("clinical completion changed during independent replay")
        elif role == "sanger_platform_concordance":
            module = trusted_modules.load("sanger_independent_validation")
            _, run_dir = _binding_run_directory(payload, root=root, role=role)
            trusted = trusted_completion_sha256
            completion_path = run_dir / "completion.json"
            if (
                not isinstance(trusted, str)
                or not SHA256_RE.fullmatch(trusted)
                or _sha256_file(completion_path) != trusted
            ):
                _fail("Sanger platform completion differs from its detached receipt")
            replayed = module.verify_run(
                run_dir,
                trusted_completion_sha256=trusted,
            )
            if (
                replayed.get("schema")
                != "DepPLM-Sanger-independent-platform-verification-v2"
                or replayed.get("verified") is not True
                or replayed.get("scientific_result") is not True
                or replayed.get("completion_sha256") != trusted
                or replayed.get("trusted_completion_sha256") != trusted
            ):
                _fail("Sanger platform independent replay did not pass")
            if _sha256_file(completion_path) != trusted:
                _fail("Sanger platform completion changed during independent replay")
        elif role == "long_protein_analysis":
            module = trusted_modules.load("verify_long_protein_analysis")
            run_relative, run_dir = _binding_run_directory(
                payload, root=root, role=role
            )
            trusted = trusted_completion_sha256
            completion_path = run_dir / "completion.json"
            if (
                not isinstance(trusted, str)
                or not SHA256_RE.fullmatch(trusted)
                or _sha256_file(completion_path) != trusted
            ):
                _fail("long-protein completion differs from its detached receipt")
            completion_record, _, _ = _strict_workspace_record(
                payload.get("completion"),
                root=root,
                label="long_protein_analysis.completion",
            )
            verification_record, _, verification_path = _strict_workspace_record(
                payload.get("verification"),
                root=root,
                label="long_protein_analysis.verification",
            )
            persisted, _ = _load_json_with_fingerprint(
                verification_path,
                label="long-protein independent verification",
                expected=verification_record,
            )
            _require_long_protein_independent_report(
                persisted,
                trusted_completion_sha256=trusted,
                run_directory=run_relative,
                completion_record=completion_record,
            )
            with tempfile.TemporaryDirectory(
                prefix="depplm-long-protein-independent-replay-"
            ) as temporary_directory:
                output = Path(temporary_directory) / "verification.json"
                replayed, output_record = module.verify_analysis(
                    run_dir,
                    trusted_completion_sha256=trusted,
                    output=output,
                )
                if not isinstance(replayed, Mapping) or not isinstance(
                    output_record, Mapping
                ):
                    _fail("long-protein independent verifier returned invalid output")
                expected_output_record = {
                    "path": str(output),
                    "bytes": output.stat().st_size,
                    "sha256": _sha256_file(output),
                }
                if dict(output_record) != expected_output_record:
                    _fail("long-protein independent replay output record is invalid")
                written_report, _ = _load_json_with_fingerprint(
                    output,
                    label="fresh long-protein independent verification",
                    expected=output_record,
                )
                if written_report != replayed:
                    _fail(
                        "long-protein independent verifier return differs from its output"
                    )
                _require_long_protein_independent_report(
                    replayed,
                    trusted_completion_sha256=trusted,
                    run_directory=run_relative,
                    completion_record=completion_record,
                )
                volatile = frozenset({"created_utc", "verification_body_sha256"})
                if _without_volatile_fields(
                    persisted, fields=volatile
                ) != _without_volatile_fields(replayed, fields=volatile):
                    _fail(
                        "long-protein independent verification differs from fresh replay"
                    )
            if _sha256_file(completion_path) != trusted:
                _fail("long-protein completion changed during independent replay")
        elif role == "independent_biological_validation":
            module = trusted_modules.load("organoid_independent_validation")
            run_relative, run_dir = _binding_run_directory(
                payload, root=root, role=role
            )
            trusted = trusted_completion_sha256
            trusted_freeze = trusted_freeze_sha256
            trusted_prediction = trusted_prediction_lock_sha256
            completion_path = run_dir / "completion.json"
            if (
                not isinstance(trusted, str)
                or not SHA256_RE.fullmatch(trusted)
                or not isinstance(trusted_freeze, str)
                or not SHA256_RE.fullmatch(trusted_freeze)
                or not isinstance(trusted_prediction, str)
                or not SHA256_RE.fullmatch(trusted_prediction)
                or _sha256_file(completion_path) != trusted
            ):
                _fail("organoid trusted digest set is invalid or changed")
            replayed = module.verify_run(
                run_dir=run_dir,
                trusted_freeze_sha256=trusted_freeze,
                trusted_prediction_lock_sha256=trusted_prediction,
                trusted_completion_sha256=trusted,
                root=root,
            )
            completion_record, _, _ = _strict_workspace_record(
                payload.get("completion"),
                root=root,
                label=f"{role}.completion",
            )
            completion_payload, _ = _load_json_with_fingerprint(
                completion_path,
                label="organoid completion",
                expected=completion_record,
            )
            if (
                set(completion_payload) != ORGANOID_COMPLETION_FIELDS
                or completion_payload.get("schema")
                != "DepPLM-Sanger-organoid-completion-v2"
                or completion_payload.get("scientific_result") is not True
                or completion_payload.get("trusted_freeze_sha256") != trusted_freeze
                or completion_payload.get("trusted_prediction_lock_sha256")
                != trusted_prediction
                or completion_payload.get("verification_requirement")
                != ORGANOID_VERIFICATION_REQUIREMENT
                or not isinstance(replayed, Mapping)
                or set(replayed) != ORGANOID_VERIFICATION_FIELDS
                or replayed.get("schema") != "DepPLM-Sanger-organoid-verification-v2"
                or replayed.get("verification_passed") is not True
                or replayed.get("scientific_result") is not False
                or replayed.get("verification_scope") != ORGANOID_VERIFICATION_SCOPE
                or replayed.get("trusted_freeze_sha256") != trusted_freeze
                or replayed.get("trusted_prediction_lock_sha256") != trusted_prediction
                or replayed.get("trusted_completion_sha256") != trusted
                or replayed.get("run_directory") != run_relative
                or replayed.get("completion") != completion_record
                or replayed.get("analysis_canonical_sha256")
                != completion_payload.get("analysis_canonical_sha256")
                or replayed.get("recomputed_primary_results")
                != completion_payload.get("primary_results")
                or replayed.get("recomputed_conclusion")
                != completion_payload.get("conclusion")
            ):
                _fail("organoid independent biological replay did not pass")
            verification_record, _, verification_path = _strict_workspace_record(
                payload.get("verification"),
                root=root,
                label=f"{role}.verification",
            )
            persisted, _ = _load_json_with_fingerprint(
                verification_path,
                label="organoid external verification report",
                expected=verification_record,
            )
            if set(persisted) != ORGANOID_VERIFICATION_FIELDS:
                _fail("organoid external verification fields are unexpected")
            volatile = frozenset({"created_utc"})
            if _without_volatile_fields(
                persisted, fields=volatile
            ) != _without_volatile_fields(replayed, fields=volatile):
                _fail("organoid external verification report differs from fresh replay")
            if _sha256_file(completion_path) != trusted:
                _fail("organoid completion changed during independent replay")
        else:
            _fail(f"no independent verifier is registered for {role}")
    except ReleaseValidationError:
        raise
    except Exception as exc:
        raise ReleaseValidationError(
            f"{role} independent verifier failed: {type(exc).__name__}: {exc}"
        ) from exc


def _require_registered_verification(
    role: str,
    *,
    path: Path,
    root: Path,
    payload: Mapping[str, Any],
    gatdep_official_root: Path | None,
    included: set[str],
    authorized: dict[str, set[str]],
    trusted_completion_sha256: str | None = None,
    trusted_completion_bytes: int | None = None,
    trusted_freeze_sha256: str | None = None,
    trusted_prediction_lock_sha256: str | None = None,
    trusted_source_bindings: Mapping[str, SelectedFile] | None = None,
) -> None:
    """Replay one role with all reviewed first-party imports rooted to ``root``."""
    with _TrustedWorkspaceModuleSession(
        root,
        expected_sources=trusted_source_bindings,
    ) as trusted_modules:
        _require_registered_verification_with_trusted_modules(
            role,
            trusted_modules=trusted_modules,
            path=path,
            root=root,
            payload=payload,
            gatdep_official_root=gatdep_official_root,
            included=included,
            authorized=authorized,
            trusted_completion_sha256=trusted_completion_sha256,
            trusted_completion_bytes=trusted_completion_bytes,
            trusted_freeze_sha256=trusted_freeze_sha256,
            trusted_prediction_lock_sha256=trusted_prediction_lock_sha256,
        )


def _contract_core(
    payload: Mapping[str, Any],
    path: Path,
    root: Path,
    included: set[str],
    authorized: dict[str, set[str]],
    trusted_source_bindings: Mapping[str, SelectedFile] | None = None,
) -> None:
    _require_registered_verification(
        "core_campaign",
        path=path,
        root=root,
        payload=payload,
        gatdep_official_root=None,
        included=included,
        authorized=authorized,
        trusted_source_bindings=trusted_source_bindings,
    )
    if payload.get("schema") != "DepPLM-corrected-v2.2-campaign":
        _fail("core campaign schema is invalid")
    if (
        payload.get("scientific_result") is not True
        or payload.get("protocol_validation_passed") is not True
    ):
        _fail("core campaign has not passed its scientific and protocol gates")
    gates = payload.get("scientific_readiness_gates")
    if (
        not isinstance(gates, dict)
        or not gates
        or not all(value is True for value in gates.values())
    ):
        _fail("core campaign scientific-readiness gates are incomplete")
    matrix = payload.get("prespecified_matrix", {})
    validation = payload.get("validation", {})
    if matrix.get("expected_runs") != 99 or validation.get("validated_runs") != 99:
        _fail("core campaign is not the complete 99-run matrix")
    if (
        validation.get("missing_matrix_cells") != []
        or validation.get("duplicate_matrix_cells") != {}
        or validation.get("issues") != []
    ):
        _fail("core campaign validation contains missing, duplicate, or invalid cells")
    runs = payload.get("source_runs")
    if not isinstance(runs, list) or len(runs) != 99:
        _fail("core campaign must bind exactly 99 source runs")
    keys: set[str] = set()
    prediction_count = 0
    resolved_root = root.resolve(strict=True)
    for index, run in enumerate(runs):
        if not isinstance(run, dict) or not isinstance(run.get("key"), str):
            _fail(f"core source_runs[{index}] is invalid")
        if run["key"] in keys:
            _fail(f"core source_runs contains duplicate key {run['key']!r}")
        keys.add(run["key"])
        try:
            run_path = run["path"]
        except KeyError as exc:
            raise ReleaseValidationError(
                f"core source_runs[{index}] path is missing"
            ) from exc
        run_relative_input = _relative_path(
            run_path, field=f"core source_runs[{index}].path"
        )
        try:
            run_dir = resolved_root.joinpath(
                *PurePosixPath(run_relative_input).parts
            ).resolve(strict=True)
            run_relative = run_dir.relative_to(resolved_root).as_posix()
        except (OSError, TypeError, ValueError) as exc:
            raise ReleaseValidationError(
                f"core source_runs[{index}] path is invalid"
            ) from exc
        metadata_hashes = {
            "completion.json": "completion_sha256",
            "summary.json": "summary_sha256",
            "run-manifest.json": "run_manifest_sha256",
            "cohort-manifest.json": "cohort_manifest_sha256",
        }
        for name, hash_field in metadata_hashes.items():
            workspace_relative = f"{run_relative}/{name}"
            _require_included(
                workspace_relative,
                included,
                label=f"core source_runs[{index}].{name}",
            )
            expected_hash = run.get(hash_field)
            if not isinstance(expected_hash, str) or not SHA256_RE.fullmatch(
                expected_hash
            ):
                _fail(f"core source_runs[{index}].{hash_field} is invalid")
            path = _safe_workspace_file(
                root, workspace_relative, field=f"core source_runs[{index}].{name}"
            )
            if _sha256_file(path) != expected_hash:
                _fail(
                    f"core source_runs[{index}].{name} differs from its campaign hash"
                )
            _authorize(authorized, workspace_relative, "evidence")
        artifacts = run.get("fold_artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            _fail(f"core source_runs[{index}] has no fold artifacts")
        seen: set[str] = set()
        for artifact_index, artifact in enumerate(artifacts):
            if not isinstance(artifact, dict):
                _fail(
                    f"core source_runs[{index}].fold_artifacts[{artifact_index}] is invalid"
                )
            relative = _relative_path(
                artifact.get("path"),
                field=f"core source_runs[{index}].fold_artifacts[{artifact_index}].path",
            )
            if relative in seen:
                _fail(f"core source_runs[{index}] repeats fold artifact {relative}")
            seen.add(relative)
            workspace_relative = f"{run_relative}/{relative}"
            _require_included(workspace_relative, included, label="core fold artifact")
            artifact_path = _safe_workspace_file(
                root, workspace_relative, field="core fold artifact"
            )
            _record_matches_file(artifact_path, artifact, label="core fold artifact")
            if relative.endswith("-predictions.npz"):
                prediction_count += 1
                _authorize(authorized, workspace_relative, "prediction")
            else:
                _authorize(authorized, workspace_relative, "evidence")
    if prediction_count != 231:
        _fail(
            f"core campaign must bind exactly 231 outer-test predictions, found {prediction_count}"
        )


def _contract_esm(
    payload: Mapping[str, Any],
    path: Path,
    root: Path,
    included: set[str],
    authorized: dict[str, set[str]],
    trusted_source_bindings: Mapping[str, SelectedFile] | None = None,
) -> None:
    if (
        payload.get("schema") != "DepPLM-esm-embedding-v1"
        or payload.get("status") != "complete"
    ):
        _fail("ESM embedding manifest is not complete")
    model = payload.get("model", {})
    revision = model.get("model_revision")
    if (
        model.get("repository") != "facebook/esm2_t6_8M_UR50D"
        or not isinstance(revision, str)
        or not re.fullmatch(r"[0-9a-f]{40}", revision)
    ):
        _fail("ESM model repository or immutable revision is missing")
    if model.get("tokenizer_revision") != revision:
        _fail("ESM model and tokenizer revisions differ")
    generation = payload.get("generation", {})
    expected_generation = {
        "strategy": "chunked",
        "output_shape": [17_768, 320],
        "output_dtype": "float32",
        "padding_tokens_excluded": True,
        "special_tokens_excluded": True,
        "truncation_enabled_in_tokenizer": False,
    }
    if any(generation.get(key) != value for key, value in expected_generation.items()):
        _fail("ESM chunking or output contract differs from the canonical manifest")
    environment = payload.get("environment", {})
    versions = environment.get("key_distribution_versions", {})
    installed = environment.get("installed_distributions", [])
    installed_names = {item.get("name") for item in installed if isinstance(item, dict)}
    if (
        not isinstance(environment.get("python"), str)
        or not all(
            isinstance(versions.get(name), str) and versions[name]
            for name in ("numpy", "torch", "transformers")
        )
        or "tokenizers" not in installed_names
    ):
        _fail("ESM environment inventory lacks exact Python/package versions")
    outputs = payload.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != {
        "embeddings",
        "normalized_embeddings",
        "names",
    }:
        _fail("ESM output inventory is incomplete")
    for name, record in outputs.items():
        if not isinstance(record, dict):
            _fail(f"ESM output {name} is invalid")
        relative = _relative_path(
            str(record.get("path", "")).replace("\\", "/"), field=f"ESM output {name}"
        )
        _require_included(relative, included, label=f"ESM output {name}")
        _record_matches_file(
            _safe_workspace_file(root, relative, field=f"ESM output {name}"),
            record,
            label=f"ESM output {name}",
        )
        _authorize(authorized, relative, "provenance")
    _require_registered_verification(
        "esm_embedding_provenance",
        path=path,
        root=root,
        payload=payload,
        gatdep_official_root=None,
        included=included,
        authorized=authorized,
        trusted_source_bindings=trusted_source_bindings,
    )


def _contract_family(
    payload: Mapping[str, Any],
    path: Path,
    included: set[str],
    root: Path,
    authorized: dict[str, set[str]],
    trusted_source_bindings: Mapping[str, SelectedFile] | None = None,
) -> None:
    if (
        payload.get("schema") != "DepPLM-family-holdout-multiseed-release-v1"
        or payload.get("status") != "complete"
    ):
        _fail("family-holdout release manifest is incomplete")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        _fail("family-holdout release inventory is missing")
    base = path.parent
    seen: set[str] = set()
    for index, record in enumerate(files):
        if not isinstance(record, dict):
            _fail(f"family-holdout files[{index}] is invalid")
        relative = _relative_path(
            record.get("path"), field=f"family-holdout files[{index}].path"
        )
        if relative in seen:
            _fail(f"family-holdout release repeats {relative}")
        seen.add(relative)
        artifact = _safe_workspace_file(base, relative, field="family-holdout artifact")
        _record_matches_file(artifact, record, label="family-holdout artifact")
        workspace_relative = artifact.resolve().relative_to(root.resolve()).as_posix()
        _require_included(workspace_relative, included, label="family-holdout artifact")
        category = (
            "prediction"
            if artifact.suffix.casefold() == ".npz"
            and "prediction" in artifact.name.casefold()
            else "evidence"
        )
        _authorize(authorized, workspace_relative, category)
    completion = _load_json(base / "completion.json", label="family-holdout completion")
    if (
        completion.get("schema") != "DepPLM-family-holdout-multiseed-campaign-v1"
        or completion.get("status") != "registered_multiseed_campaign_complete"
        or completion.get("n_completed_cells") != 9
        or completion.get("folds_per_cell") != 5
        or completion.get("all_child_verifiers_verified") is not True
    ):
        _fail("family-holdout 3x3 campaign completion contract is not satisfied")
    _require_registered_verification(
        "family_holdout",
        path=path,
        root=root,
        payload=payload,
        gatdep_official_root=None,
        included=included,
        authorized=authorized,
        trusted_source_bindings=trusted_source_bindings,
    )


def _validate_k_token_pmi_matrix(payload: Mapping[str, Any]) -> None:
    expected = [
        (variant, seed, f"fold-{fold:02d}")
        for variant in K_TOKEN_PMI_VARIANTS
        for seed in K_TOKEN_PMI_SEEDS
        for fold in K_TOKEN_PMI_FOLDS
    ]
    observed_by_field: dict[str, list[tuple[str, int, str]]] = {}
    for field in ("outer_test_predictions", "per_gene_prediction_sidecars"):
        records = payload.get(field)
        if not isinstance(records, list) or len(records) != len(expected):
            _fail(f"k_token_pmi.{field} must bind exactly 105 outer-fold artifacts")
        identities: list[tuple[str, int, str]] = []
        paths: set[str] = set()
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                _fail(f"k_token_pmi.{field}[{index}] is invalid")
            variant = record.get("variant")
            seed = record.get("seed")
            fold = record.get("fold")
            if (
                not isinstance(variant, str)
                or not isinstance(seed, int)
                or isinstance(seed, bool)
                or not isinstance(fold, str)
            ):
                _fail(f"k_token_pmi.{field}[{index}] has an invalid identity")
            identity = (variant, seed, fold)
            if identity not in expected:
                _fail(f"k_token_pmi.{field}[{index}] has an unregistered identity")
            identities.append(identity)
            raw_path = record.get("path")
            if not isinstance(raw_path, str):
                _fail(f"k_token_pmi.{field}[{index}].path is invalid")
            normalized = _relative_path(
                raw_path.replace("\\", "/"),
                field=f"k_token_pmi.{field}[{index}].path",
            )
            if normalized.casefold() in paths:
                _fail(f"k_token_pmi.{field} repeats an artifact path")
            paths.add(normalized.casefold())
        if identities != expected:
            _fail(
                f"k_token_pmi.{field} is not the exact canonical "
                "7-variant x 3-seed x 5-fold inventory"
            )
        observed_by_field[field] = identities
    if (
        observed_by_field["outer_test_predictions"]
        != observed_by_field["per_gene_prediction_sidecars"]
    ):
        _fail("K/token/PMI prediction and sidecar identities differ")


def _contract_release_binding(
    role: str,
    payload: Mapping[str, Any],
    *,
    trust_receipt: Mapping[str, Any],
    path: Path,
    root: Path,
    included: set[str],
    authorized: dict[str, set[str]],
    selected_fingerprint: FileFingerprint,
    trusted_source_bindings: Mapping[str, SelectedFile] | None = None,
) -> None:
    common_fields = {
        "schema",
        "run_directory",
        "completion",
        "public_artifacts",
        "redistribution_review",
        "claim_boundary",
    }
    expected_fields = set(common_fields)
    if role == "long_protein_analysis":
        expected_fields.add("verification")
    if role == "independent_biological_validation":
        expected_fields.update(
            {
                "protocol_lock",
                "local_only_artifacts",
                "verification",
            }
        )
    if set(payload) != expected_fields:
        _fail(
            f"{role} release binding fields must be exactly {sorted(expected_fields)}"
        )
    if payload.get("schema") != RELEASE_BINDING_SCHEMAS[role]:
        _fail(f"{role} release binding schema is invalid")
    _meaningful_text(payload.get("claim_boundary"), field=f"{role}.claim_boundary")
    _validate_redistribution_review(payload.get("redistribution_review"), role=role)

    normalized_receipt = _normalize_trust_receipt_payload(
        trust_receipt, role=role, root=root
    )
    _, binding_relative, binding_path = _strict_workspace_record(
        normalized_receipt["binding"],
        root=root,
        label=f"{role} receipt.binding",
    )
    observed_binding_relative = (
        path.resolve(strict=True).relative_to(root.resolve(strict=True)).as_posix()
    )
    if binding_relative != observed_binding_relative or binding_path != path:
        _fail(f"{role} detached receipt does not bind the selected release binding")
    if (
        normalized_receipt["binding"]["bytes"] != selected_fingerprint.bytes
        or normalized_receipt["binding"]["sha256"] != selected_fingerprint.sha256
    ):
        _fail(f"{role} detached receipt does not bind the parsed release bytes")

    run_relative, run_dir = _binding_run_directory(payload, root=root, role=role)
    completion_record, completion_relative, completion_path = _strict_workspace_record(
        payload.get("completion"),
        root=root,
        label=f"{role}.completion",
    )
    expected_completion = f"{run_relative}/completion.json"
    if completion_relative != expected_completion or completion_path.parent != run_dir:
        _fail(f"{role}.completion must bind run_directory/completion.json")
    if normalized_receipt["completion"] != completion_record:
        _fail(f"{role} completion differs from its detached trust receipt")
    trusted_completion = normalized_receipt["completion"]["sha256"]
    trusted_freeze: str | None = None
    trusted_prediction: str | None = None
    completion, _ = _load_json_with_fingerprint(
        completion_path,
        label=f"{role} completion",
        expected=completion_record,
    )

    if role == "sanger_platform_concordance":
        expected_paths = tuple(
            f"{run_relative}/{name}" for name in SANGER_PLATFORM_PUBLIC_RUN_FILES
        )
        public_records = _validate_public_artifacts(
            payload.get("public_artifacts"),
            expected_paths=expected_paths,
            root=root,
            role=role,
            included=included,
            authorized=authorized,
        )
        if (
            completion.get("schema")
            != "DepPLM-Sanger-independent-platform-completion-v2"
            or completion.get("scientific_result") is not True
        ):
            _fail("Sanger platform completion contract is invalid")
        artifacts = completion.get("artifacts")
        expected_names = SANGER_PLATFORM_PUBLIC_RUN_FILES[1:]
        if not isinstance(artifacts, list) or len(artifacts) != len(expected_names):
            _fail("Sanger platform completion artifact inventory is invalid")
        for index, (record, name) in enumerate(
            zip(artifacts, expected_names, strict=True)
        ):
            if not isinstance(record, Mapping) or set(record) != {
                "path",
                "bytes",
                "sha256",
            }:
                _fail(f"Sanger completion artifact[{index}] is invalid")
            relative, _ = _record_path(
                record,
                root=root,
                base=run_dir,
                label=f"Sanger completion artifact[{index}]",
            )
            expected_relative = f"{run_relative}/{name}"
            if record.get("path") != name or relative != expected_relative:
                _fail("Sanger completion artifact order or path is noncanonical")
            public = public_records[expected_relative.casefold()]
            if any(record.get(field) != public[field] for field in ("bytes", "sha256")):
                _fail("Sanger public artifact differs from its completion binding")
    elif role == "long_protein_analysis":
        if completion.get("schema") != "DepPLM-long-protein-analysis-completion-v3":
            _fail("long-protein completion contract is invalid")
        verification_record, verification_relative, verification_path = (
            _strict_workspace_record(
                payload.get("verification"),
                root=root,
                label=f"{role}.verification",
            )
        )
        if verification_path == run_dir or run_dir in verification_path.parents:
            _fail(
                "long-protein independent report must be outside and not below "
                "the immutable run directory"
            )
        expected_paths = tuple(
            f"{run_relative}/{name}" for name in LONG_PROTEIN_PUBLIC_RUN_FILES
        ) + (verification_relative,)
        public_records = _validate_public_artifacts(
            payload.get("public_artifacts"),
            expected_paths=expected_paths,
            root=root,
            role=role,
            included=included,
            authorized=authorized,
        )
        if public_records[verification_relative.casefold()] != verification_record:
            _fail("long-protein public verifier record differs from the binding")
        persisted_verification, _ = _load_json_with_fingerprint(
            verification_path,
            label="long-protein independent verification",
            expected=verification_record,
        )
        _require_long_protein_independent_report(
            persisted_verification,
            trusted_completion_sha256=trusted_completion,
            run_directory=run_relative,
            completion_record=completion_record,
        )
        nested_records = {
            "run-manifest.json": completion.get("run_manifest"),
        }
        artifacts = completion.get("artifacts")
        if not isinstance(artifacts, Mapping) or set(artifacts) != {
            "per_gene_errors",
            "summary",
        }:
            _fail("long-protein completion artifact inventory is invalid")
        nested_records.update(
            {
                "summary.json": artifacts.get("summary"),
                "per-gene-errors.npz": artifacts.get("per_gene_errors"),
            }
        )
        for name in LONG_PROTEIN_PUBLIC_RUN_FILES[1:]:
            record = nested_records[name]
            if not isinstance(record, Mapping) or set(record) != {
                "path",
                "bytes",
                "sha256",
            }:
                _fail(f"long-protein completion record is invalid: {name}")
            relative, _ = _record_path(
                record,
                root=root,
                base=run_dir,
                label=f"long-protein completion.{name}",
            )
            expected_relative = f"{run_relative}/{name}"
            if record.get("path") != name or relative != expected_relative:
                _fail(f"long-protein completion path is noncanonical: {name}")
            public = public_records[expected_relative.casefold()]
            if any(record.get(field) != public[field] for field in ("bytes", "sha256")):
                _fail(
                    "long-protein public artifact differs from its completion binding"
                )
        run_manifest, _ = _load_json_with_fingerprint(
            run_dir / "run-manifest.json",
            label="long-protein run manifest",
            expected=nested_records["run-manifest.json"],
        )
        summary, _ = _load_json_with_fingerprint(
            run_dir / "summary.json",
            label="long-protein summary",
            expected=nested_records["summary.json"],
        )
        if (
            run_manifest.get("schema") != "DepPLM-long-protein-analysis-run-v3"
            or summary.get("schema") != "DepPLM-long-protein-analysis-v3"
        ):
            _fail("long-protein release requires the complete v3 analysis contract")
    elif role == "independent_biological_validation":
        protocol_record, protocol_relative, _ = _strict_workspace_record(
            payload.get("protocol_lock"),
            root=root,
            label=f"{role}.protocol_lock",
        )
        if normalized_receipt.get("protocol_lock") != protocol_record:
            _fail("organoid protocol lock differs from its detached trust receipt")
        trusted_freeze = normalized_receipt["protocol_lock"]["sha256"]
        prediction_lock_record = normalized_receipt["prediction_lock"]
        trusted_prediction = prediction_lock_record["sha256"]
        verification_record, verification_relative, verification_path = (
            _strict_workspace_record(
                payload.get("verification"),
                root=root,
                label=f"{role}.verification",
            )
        )
        if verification_path == run_dir or run_dir in verification_path.parents:
            _fail(
                "organoid verifier report must be outside and not below the immutable run directory"
            )
        public_paths = tuple(
            f"{run_relative}/{name}" for name in ORGANOID_PUBLIC_RUN_FILES
        ) + (protocol_relative, verification_relative)
        public_records = _validate_public_artifacts(
            payload.get("public_artifacts"),
            expected_paths=public_paths,
            root=root,
            role=role,
            included=included,
            authorized=authorized,
        )
        if public_records[protocol_relative.casefold()] != protocol_record:
            _fail("organoid public protocol-lock record differs from the binding")
        if public_records[verification_relative.casefold()] != verification_record:
            _fail("organoid public verifier record differs from the binding")
        if (
            set(completion) != ORGANOID_COMPLETION_FIELDS
            or completion.get("schema") != "DepPLM-Sanger-organoid-completion-v2"
            or completion.get("scientific_result") is not True
            or completion.get("verification_requirement")
            != ORGANOID_VERIFICATION_REQUIREMENT
        ):
            _fail("organoid completion contract is invalid")
        if completion.get("freeze") != protocol_record:
            _fail("organoid completion does not bind the release protocol lock")
        if completion.get("trusted_freeze_sha256") != trusted_freeze:
            _fail("organoid completion trusted protocol digest differs")
        if completion.get("trusted_prediction_lock_sha256") != trusted_prediction:
            _fail("organoid completion trusted prediction-lock digest differs")
        artifact_records = completion.get("artifacts")
        if (
            not isinstance(artifact_records, Mapping)
            or set(artifact_records) != ORGANOID_COMPLETION_ARTIFACTS
        ):
            _fail("organoid completion artifact inventory is invalid")
        if artifact_records.get("prediction-lock.json") != prediction_lock_record:
            _fail(
                "organoid detached receipt does not bind the completion prediction lock"
            )
        local_paths = tuple(
            f"{run_relative}/{name}" for name in ORGANOID_LOCAL_ONLY_RUN_FILES
        )
        local_records = _validate_local_only_artifacts(
            payload.get("local_only_artifacts"),
            expected_paths=local_paths,
            root=root,
            role=role,
            included=included,
        )
        for name, nested_value in artifact_records.items():
            nested_record, nested_relative, _ = _strict_workspace_record(
                nested_value,
                root=root,
                label=f"organoid completion.artifacts.{name}",
            )
            expected_relative = f"{run_relative}/{name}"
            if nested_relative != expected_relative:
                _fail(f"organoid completion artifact escapes its run: {name}")
            lookup = (
                public_records
                if name in ORGANOID_PUBLIC_RUN_FILES[1:]
                else local_records
            )
            if lookup[expected_relative.casefold()] != nested_record:
                _fail(f"organoid release binding differs from completion: {name}")
    else:
        _fail(f"no release-binding contract is registered for {role}")

    if public_records[completion_relative.casefold()] != completion_record:
        _fail(f"{role} public completion record differs from its trust binding")
    if trusted_completion != completion_record["sha256"]:
        _fail(f"{role} trusted completion digest changed during validation")
    _require_registered_verification(
        role,
        path=path,
        root=root,
        payload=payload,
        gatdep_official_root=None,
        included=included,
        authorized=authorized,
        trusted_completion_sha256=trusted_completion,
        trusted_freeze_sha256=trusted_freeze,
        trusted_prediction_lock_sha256=trusted_prediction,
        trusted_source_bindings=trusted_source_bindings,
    )


def _contract_simple(
    role: str,
    payload: Mapping[str, Any],
    *,
    path: Path,
    root: Path,
    included: set[str],
    authorized: dict[str, set[str]],
    gatdep_official_root: Path | None = None,
    selected_sha256: str | None = None,
    selected_bytes: int | None = None,
    selected_files: Mapping[str, SelectedFile] | None = None,
    trusted_source_bindings: Mapping[str, SelectedFile] | None = None,
) -> None:
    checks: tuple[bool, ...]
    if role == "k_token_pmi":
        checks = (
            payload.get("schema") == "DepPLM-k-token-pmi-campaign-completion-v1",
            payload.get("protocol_complete") is True,
            payload.get("canonical_campaign_complete") is True,
            type(payload.get("n_completed_variant_seed_jobs")) is int
            and payload.get("n_completed_variant_seed_jobs") == 21,
            type(payload.get("n_completed_outer_fold_model_evaluations")) is int
            and payload.get("n_completed_outer_fold_model_evaluations") == 105,
            payload.get("eligible_for_scientific_aggregation") is True,
        )
    elif role == "attention_faithfulness":
        checks = (
            payload.get("schema")
            == "DepPLM-attention-faithfulness-v1-campaign-verification",
            payload.get("scope") == "canonical",
            payload.get("n_verified_folds") == 15,
            payload.get("verification_passed") is True,
            payload.get("protocol_complete") is True,
            payload.get("eligible_for_scientific_aggregation") is True,
        )
    elif role == "cnv_sensitivity":
        checks = (
            payload.get("schema") == "DepPLM-CNV-sensitivity-verification-v1",
            payload.get("scientific_result") is True,
            payload.get("protocol_complete") is True,
            payload.get("execution_complete") is True,
            payload.get("n_verified_seed_fold_pairs") == 15,
            payload.get("outer_test_metrics_recomputed") is True,
            payload.get("paired_targets_and_identities_identical") is True,
        )
    elif role == "external_architecture_baseline":
        checks = (
            payload.get("schema") == "GATDep-corrected-run-verification-v1",
            payload.get("valid") is True,
            payload.get("errors") == [],
            payload.get("completed_folds") == [0, 1, 2, 3, 4],
            payload.get("outer_test_exactly_once") is True,
            payload.get("scientific_result") is False,
        )
    elif role == "clinical_annotation_concordance":
        try:
            clinical_relative = path.resolve(strict=True).relative_to(
                root.resolve(strict=True)
            )
        except ValueError as exc:
            raise ReleaseValidationError(
                "clinical completion is outside the workspace"
            ) from exc
        _require_clinical_publication_path(
            PurePosixPath(clinical_relative.as_posix()),
            filename="completion.json",
            field="clinical selected evidence",
        )
        checks = (
            payload.get("schema") == CLINICAL_COMPLETION_SCHEMA,
            payload.get("scientific_result") is True,
            set(payload) == CLINICAL_COMPLETION_FIELDS,
        )
    else:
        _fail(f"no evidence contract is registered for {role}")
    if not all(checks):
        _fail(f"evidence contract is incomplete for role {role}")

    record_groups: list[tuple[Path, list[Mapping[str, Any]], str]] = []
    if role == "k_token_pmi":
        _validate_k_token_pmi_matrix(payload)
        for field, category in (
            ("outer_test_predictions", "prediction"),
            ("per_gene_prediction_sidecars", "evidence"),
        ):
            records = payload.get(field)
            if not isinstance(records, list) or len(records) != 105:
                _fail(f"{role}.{field} must bind exactly 105 outer-fold artifacts")
            record_groups.append(
                (root / "results" / "k_token_pmi_v1", records, category)
            )
    elif role == "clinical_annotation_concordance":
        completion_records = {
            "verification.json": payload.get("verification"),
            "results.json": payload.get("results"),
        }
        validated_records: list[Mapping[str, Any]] = []
        for name, record in completion_records.items():
            if not isinstance(record, Mapping) or set(record) != {
                "path",
                "bytes",
                "sha256",
            }:
                _fail(f"clinical completion record is invalid: {name}")
            if record.get("path") != name:
                _fail(f"clinical completion path is noncanonical: {name}")
            validated_records.append(record)
        record_groups.append((path.parent, validated_records, "evidence"))
    clinical_results: dict[str, Any] | None = None
    for base, records, category in record_groups:
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                _fail(f"{role} artifact[{index}] is invalid")
            relative, artifact = _record_path(
                record,
                root=root,
                base=base,
                label=f"{role} artifact[{index}]",
            )
            _require_included(relative, included, label=f"{role} artifact[{index}]")
            _authorize(authorized, relative, category)
            if (
                role == "clinical_annotation_concordance"
                and artifact.name == "verification.json"
            ):
                verification, _ = _load_json_with_fingerprint(
                    artifact,
                    label="clinical verification",
                    expected=record,
                )
                if (
                    set(verification) != CLINICAL_VERIFICATION_FIELDS
                    or verification.get("verification_method")
                    != CLINICAL_VERIFICATION_METHOD
                    or verification.get("schema") != CLINICAL_VERIFICATION_SCHEMA
                    or verification.get("scientific_result") is not True
                    or not isinstance(verification.get("analysis_sha256"), str)
                    or not SHA256_RE.fullmatch(verification["analysis_sha256"])
                ):
                    _fail("clinical verification contract is invalid")
                _meaningful_text(
                    verification.get("verified_utc"),
                    field="clinical verification.verified_utc",
                )
                nested = verification.get("artifacts")
                if not isinstance(nested, list) or len(nested) != len(
                    CLINICAL_VERIFICATION_ARTIFACTS
                ):
                    _fail("clinical verification artifact inventory is not exact")
                for nested_index, (nested_record, expected_name) in enumerate(
                    zip(nested, CLINICAL_VERIFICATION_ARTIFACTS, strict=True)
                ):
                    if not isinstance(nested_record, Mapping) or set(nested_record) != {
                        "path",
                        "bytes",
                        "sha256",
                    }:
                        _fail("clinical verification artifact record is invalid")
                    if nested_record.get("path") != expected_name:
                        _fail(
                            "clinical verification artifact order or path is noncanonical"
                        )
                    nested_relative, _ = _record_path(
                        nested_record,
                        root=root,
                        base=path.parent,
                        label=f"clinical verification artifact[{nested_index}]",
                    )
                    _require_included(
                        nested_relative,
                        included,
                        label=f"clinical verification artifact[{nested_index}]",
                    )
                    _authorize(authorized, nested_relative, "evidence")
                    if (
                        expected_name == "results.json"
                        and nested_record != payload.get("results")
                    ):
                        _fail(
                            "clinical results binding differs between completion and verification"
                        )
            elif (
                role == "clinical_annotation_concordance"
                and artifact.name == "results.json"
            ):
                if selected_files is None:
                    _fail("clinical selected-file bindings are required")
                selected_results = selected_files.get(relative)
                if (
                    selected_results is None
                    or selected_results.relative != relative
                    or selected_results.category != "evidence"
                    or selected_results.bytes != record.get("bytes")
                    or selected_results.sha256 != record.get("sha256")
                ):
                    _fail("clinical results selection differs from completion")
                clinical_results = _parse_json_bytes(
                    selected_results.payload,
                    label="selected clinical results",
                    path=selected_results.source,
                )
    if role == "clinical_annotation_concordance":
        if selected_files is None or clinical_results is None:
            _fail("clinical selected-file bindings are incomplete")
        if (
            clinical_results.get("schema") != CLINICAL_RESULT_SCHEMA
            or clinical_results.get("scientific_result") is not True
        ):
            _fail("clinical results contract is not eligible v3 evidence")
        publication = selected_files.get(CLINICAL_PUBLICATION_TABLE)
        if publication is None:
            _fail("clinical publication table was not selected")
        completion_digest = selected_sha256 or _sha256_file(path)
        results_digest = payload["results"]["sha256"]
        _validate_clinical_publication_binding(
            root=root,
            publication=publication,
            results=clinical_results,
            completion_sha256=completion_digest,
            results_sha256=results_digest,
            trusted_source_bindings=trusted_source_bindings,
        )
    _require_registered_verification(
        role,
        path=path,
        root=root,
        payload=payload,
        gatdep_official_root=gatdep_official_root,
        included=included,
        authorized=authorized,
        trusted_completion_sha256=(
            selected_sha256 if role == "clinical_annotation_concordance" else None
        ),
        trusted_completion_bytes=(
            selected_bytes if role == "clinical_annotation_concordance" else None
        ),
        trusted_source_bindings=trusted_source_bindings,
    )


def _trusted_source_bindings(
    selected: Sequence[SelectedFile],
) -> dict[str, SelectedFile]:
    selected_by_path = {item.relative: item for item in selected}
    bindings: dict[str, SelectedFile] = {}
    for name in sorted(TRUSTED_WORKSPACE_MODULE_NAMES):
        relative = TRUSTED_WORKSPACE_MODULE_PATHS[name]
        item = selected_by_path.get(relative)
        if item is None:
            _fail(f"reviewed verifier source is absent from selection: {relative}")
        bindings[name] = item
    return bindings


def _validate_evidence(
    records: Any,
    *,
    root: Path,
    categories: Mapping[str, str],
    selected: Sequence[SelectedFile],
    gatdep_official_root: Path | None,
    trust_receipts: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, set[str]]]:
    if not isinstance(records, list):
        _fail("evidence must be a list")
    roles: set[str] = set()
    normalized: list[dict[str, str]] = []
    included = set(categories)
    selected_by_path = {item.relative: item for item in selected}
    trusted_sources = _trusted_source_bindings(selected)
    authorized: dict[str, set[str]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {"role", "path"}:
            _fail(f"evidence[{index}] must contain exactly role and path")
        role = record["role"]
        if not isinstance(role, str) or role not in REQUIRED_EVIDENCE_ROLES:
            _fail(f"evidence[{index}] has unsupported role {role!r}")
        if role in roles:
            _fail(f"evidence contains duplicate role {role}")
        roles.add(role)
        relative = _relative_path(record["path"], field=f"evidence[{index}].path")
        if relative not in included or categories[relative] not in {
            "evidence",
            "provenance",
        }:
            _fail(
                f"evidence[{index}] is not an explicitly included evidence/provenance file"
            )
        path = _safe_workspace_file(root, relative, field=f"evidence[{index}].path")
        payload, fingerprint = _load_json_with_fingerprint(
            path,
            label=role,
            expected=selected_by_path[relative],
        )
        _authorize(
            authorized,
            relative,
            "provenance" if role == "esm_embedding_provenance" else "evidence",
        )
        if role == "core_campaign":
            _contract_core(
                payload,
                path,
                root,
                included,
                authorized,
                trusted_source_bindings=trusted_sources,
            )
        elif role == "esm_embedding_provenance":
            _contract_esm(
                payload,
                path,
                root,
                included,
                authorized,
                trusted_source_bindings=trusted_sources,
            )
        elif role == "family_holdout":
            _contract_family(
                payload,
                path,
                included,
                root,
                authorized,
                trusted_source_bindings=trusted_sources,
            )
        elif role in RELEASE_BINDING_SCHEMAS:
            _contract_release_binding(
                role,
                payload,
                trust_receipt=trust_receipts[role],
                path=path,
                root=root,
                included=included,
                authorized=authorized,
                selected_fingerprint=fingerprint,
                trusted_source_bindings=trusted_sources,
            )
        else:
            _contract_simple(
                role,
                payload,
                path=path,
                root=root,
                included=included,
                authorized=authorized,
                gatdep_official_root=gatdep_official_root,
                selected_sha256=fingerprint.sha256,
                selected_bytes=fingerprint.bytes,
                selected_files=selected_by_path,
                trusted_source_bindings=trusted_sources,
            )
        normalized.append(
            {"role": role, "path": relative, "sha256": fingerprint.sha256}
        )
    missing = REQUIRED_EVIDENCE_ROLES - roles
    if missing:
        _fail(f"required evidence roles are missing: {sorted(missing)}")
    return sorted(normalized, key=lambda item: item["role"]), authorized


def _preflight_metadata_report(value: Any) -> dict[str, Any]:
    required_fields = (
        "title",
        "version",
        "author_supplied_doi",
        "publication_date",
        "license",
        "creators",
        "funding",
        "conflict_of_interest",
        "ai_use_disclosure",
    )
    missing_or_placeholder: list[str] = []

    def contains_placeholder(candidate: Any) -> bool:
        if isinstance(candidate, str):
            return bool(PLACEHOLDER_RE.search(candidate))
        if isinstance(candidate, Mapping):
            return any(contains_placeholder(item) for item in candidate.values())
        if isinstance(candidate, list):
            return any(contains_placeholder(item) for item in candidate)
        return False

    if not isinstance(value, Mapping):
        missing_or_placeholder.extend(required_fields)
    else:
        for field in required_fields:
            candidate = value.get(field)
            if field == "creators":
                if not isinstance(candidate, list) or not candidate:
                    missing_or_placeholder.append(field)
                    continue
                if contains_placeholder(candidate):
                    missing_or_placeholder.append(field)
            elif (
                not isinstance(candidate, str)
                or not candidate.strip()
                or PLACEHOLDER_RE.search(candidate)
            ):
                missing_or_placeholder.append(field)

    syntax_valid = False
    syntax_error: str | None = None
    if not missing_or_placeholder:
        try:
            _validate_release_metadata(value)
        except ReleaseValidationError as exc:
            syntax_error = str(exc)
        else:
            syntax_valid = True

    local_state = {
        field: (
            "missing_or_placeholder"
            if field in missing_or_placeholder
            else "provided_not_author_verified"
        )
        for field in required_fields
    }
    actions = [
        {
            "code": "author_metadata_and_contributions",
            "status": "required_author_action",
            "machine_verifiable": False,
            "local_fields": {
                field: local_state[field] for field in ("title", "creators")
            },
        },
        {
            "code": "funding_conflict_and_ai_disclosures",
            "status": "required_author_action",
            "machine_verifiable": False,
            "local_fields": {
                field: local_state[field]
                for field in (
                    "funding",
                    "conflict_of_interest",
                    "ai_use_disclosure",
                )
            },
        },
        {
            "code": "archive_license_and_redistribution_approval",
            "status": "required_author_and_rightsholder_action",
            "machine_verifiable": False,
            "local_fields": {"license": local_state["license"]},
        },
        {
            "code": "doi_reservation_and_ownership",
            "status": "required_repository_and_author_action",
            "machine_verifiable": False,
            "local_fields": {"author_supplied_doi": local_state["author_supplied_doi"]},
        },
        {
            "code": "public_deposit_and_post_deposit_hash_verification",
            "status": "required_external_action",
            "machine_verifiable": False,
            "local_fields": {
                field: local_state[field] for field in ("version", "publication_date")
            },
        },
    ]
    return {
        "local_syntax_valid": syntax_valid,
        "local_syntax_error": syntax_error,
        "missing_or_placeholder_fields": sorted(missing_or_placeholder),
        "field_states": local_state,
        "human_action_items": actions,
    }


def _preflight_required_content(root: Path) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    checked = 0
    for category, paths in sorted(REQUIRED_CONTENT_FILES.items()):
        for relative in sorted(paths):
            try:
                source = _safe_workspace_file(
                    root,
                    relative,
                    field=f"preflight required {category} file",
                )
                payload, _ = _read_stable_bytes(
                    source,
                    label=f"preflight required {category} file {relative}",
                )
                _validate_reviewed_publication_file(
                    payload,
                    relative=relative,
                    category=category,
                    enforce_manuscript_placeholders=False,
                )
                _scan_public_text_for_local_paths(
                    source,
                    relative=relative,
                    category=category,
                    payload=payload,
                )
                checked += 1
            except (OSError, ReleaseValidationError) as exc:
                errors.append({"path": relative, "error": str(exc)})
    if not any(error["path"] == CORRECTED_CAMPAIGN_PROVENANCE for error in errors):
        try:
            _validate_corrected_campaign_provenance_bindings(root)
        except (OSError, ReleaseValidationError) as exc:
            errors.append({"path": CORRECTED_CAMPAIGN_PROVENANCE, "error": str(exc)})
    return {
        "passed": not errors,
        "checked_files": checked,
        "required_files": sum(len(paths) for paths in REQUIRED_CONTENT_FILES.values()),
        "errors": errors,
    }


def _preflight_manuscript(root: Path) -> dict[str, Any]:
    categories = {relative: "manuscript" for relative in REQUIRED_MANUSCRIPT_FILES}
    dependency_error: str | None = None
    dependencies: set[str] = set()
    try:
        dependencies = _manuscript_dependency_closure(root, categories)
    except (OSError, ReleaseValidationError) as exc:
        dependency_error = str(exc)
    missing_dependencies = sorted(ACTIVE_MANUSCRIPT_DEPENDENCIES - dependencies)
    unexpected_dependencies = sorted(dependencies - ACTIVE_MANUSCRIPT_DEPENDENCIES)
    dependency_passed = (
        dependency_error is None
        and not missing_dependencies
        and not unexpected_dependencies
    )

    placeholder_findings: list[dict[str, Any]] = []
    placeholder_errors: list[dict[str, str]] = []
    active_text: list[str] = []
    for relative in sorted(
        dependency
        for dependency in (dependencies or ACTIVE_MANUSCRIPT_DEPENDENCIES)
        if Path(dependency).suffix.casefold() == ".tex"
    ):
        try:
            source = _safe_workspace_file(
                root, relative, field="preflight active manuscript source"
            )
            payload, _ = _read_stable_bytes(
                source, label=f"preflight active manuscript source {relative}"
            )
            text = _decode_public_text(payload, relative=relative)
            active_text.append(_strip_tex_comments(text))
            placeholder_findings.extend(
                _active_manuscript_placeholder_findings(
                    text.splitlines(), relative=relative
                )
            )
        except (OSError, ReleaseValidationError) as exc:
            placeholder_errors.append({"path": relative, "error": str(exc)})

    normalized_active_text = re.sub(r"\s+", " ", "\n".join(active_text)).casefold()
    explicit_exclusion = any(
        marker in normalized_active_text
        for marker in (
            "clinical-annotation results are excluded pending verified regeneration",
            "no clinical-annotation result is reported pending a fresh run",
        )
    )
    table_active = CLINICAL_PUBLICATION_TABLE in dependencies
    fail_closed_table_present = False
    table_error: str | None = None
    try:
        table_path = _safe_workspace_file(
            root,
            CLINICAL_PUBLICATION_TABLE,
            field="preflight clinical publication table",
        )
        table_payload, _ = _read_stable_bytes(
            table_path, label="preflight clinical publication table"
        )
        table_text = _decode_public_text(
            table_payload, relative=CLINICAL_PUBLICATION_TABLE
        )
        fail_closed_table_present = (
            "\\PackageError" in table_text
            and "no current clinical-annotation result is eligible"
            in table_text.casefold()
        )
    except (OSError, ReleaseValidationError) as exc:
        table_error = str(exc)

    if table_active:
        clinical_status = (
            "invalid_active_fail_closed_placeholder"
            if fail_closed_table_present
            else "active_table_requires_current_verified_evidence"
        )
        clinical_passed = False
    else:
        clinical_status = (
            "excluded_pending_verified_regeneration"
            if explicit_exclusion
            else "ambiguous_exclusion_state"
        )
        clinical_passed = explicit_exclusion and table_error is None

    return {
        "dependency_graph": {
            "passed": dependency_passed,
            "observed": sorted(dependencies),
            "expected": sorted(ACTIVE_MANUSCRIPT_DEPENDENCIES),
            "missing": missing_dependencies,
            "unexpected": unexpected_dependencies,
            "error": dependency_error,
        },
        "submission_placeholders": {
            "passed": not placeholder_findings and not placeholder_errors,
            "findings": placeholder_findings,
            "errors": placeholder_errors,
        },
        "clinical_annotation": {
            "passed": clinical_passed,
            "status": clinical_status,
            "active_table_dependency": table_active,
            "explicit_exclusion_statement_present": explicit_exclusion,
            "inactive_fail_closed_table_present": (
                fail_closed_table_present and not table_active
            ),
            "current_result_release_eligible": False,
            "table_error": table_error,
        },
    }


def _preflight_spec_inventory(
    spec: Any, *, root: Path
) -> tuple[dict[str, Any], list[SelectedFile], dict[str, str]]:
    if not isinstance(spec, Mapping):
        _fail("preflight release spec must be an object")
    required = {"schema", "files", "evidence"}
    allowed = required | {"release", "trust_receipts"}
    if not required.issubset(spec) or set(spec) - allowed:
        _fail(
            "preflight release spec must contain schema/files/evidence and may "
            "contain only release/trust_receipts in addition"
        )
    if spec.get("schema") != SPEC_SCHEMA:
        _fail(f"preflight release spec must use {SPEC_SCHEMA}")
    selected, categories = _select_files(
        root,
        spec.get("files"),
        enforce_manuscript_placeholders=False,
    )
    _enforce_required_content(categories)
    dependencies = _manuscript_dependency_closure(root, categories)
    if dependencies != ACTIVE_MANUSCRIPT_DEPENDENCIES:
        _fail(
            "preflight active manuscript dependency graph differs from the exact "
            "reviewed inventory"
        )

    roles: set[str] = set()
    evidence: list[dict[str, str]] = []
    selected_by_path = {item.relative: item for item in selected}
    records = spec.get("evidence")
    if not isinstance(records, list):
        _fail("preflight evidence must be a list")
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or set(record) != {"role", "path"}:
            _fail(f"preflight evidence[{index}] must contain exactly role and path")
        role = record.get("role")
        if not isinstance(role, str) or role not in REQUIRED_EVIDENCE_ROLES:
            _fail(f"preflight evidence[{index}] has unsupported role {role!r}")
        if role in roles:
            _fail(f"preflight evidence contains duplicate role {role}")
        relative = _relative_path(
            record.get("path"), field=f"preflight evidence[{index}].path"
        )
        expected_category = (
            "provenance" if role == "esm_embedding_provenance" else "evidence"
        )
        selected_item = selected_by_path.get(relative)
        if selected_item is None or categories.get(relative) != expected_category:
            _fail(
                f"preflight evidence[{index}] is not selected in category "
                f"{expected_category}"
            )
        _parse_json_bytes(
            selected_item.payload,
            label=f"preflight evidence role {role}",
            path=selected_item.source,
        )
        roles.add(role)
        evidence.append(
            {
                "role": role,
                "path": relative,
                "sha256": selected_item.sha256,
            }
        )
    missing_roles = sorted(REQUIRED_EVIDENCE_ROLES - roles)
    if missing_roles:
        _fail(f"preflight required evidence roles are missing: {missing_roles}")
    forbidden_selected = sorted(
        item.relative
        for item in selected
        if (
            any(
                part.casefold() in FORBIDDEN_PARTS
                for part in PurePosixPath(item.relative).parts
            )
            or item.relative.startswith("data/external/")
            or PurePosixPath(item.relative).name.casefold() == "outcome-snapshot.rdata"
        )
    )
    if forbidden_selected:
        _fail(
            "preflight selected files include forbidden raw or third-party material: "
            f"{forbidden_selected}"
        )
    return (
        {
            "passed": True,
            "selected_files": len(selected),
            "evidence_roles": sorted(roles),
            "evidence": sorted(evidence, key=lambda item: item["role"]),
            "third_party_or_raw_selected_paths": forbidden_selected,
        },
        selected,
        categories,
    )


def preflight_release(
    *,
    root: Path = ROOT,
    spec_path: Path | None = None,
    trust_receipt_paths: Mapping[str, Path] | None = None,
    gatdep_official_root: Path | None = None,
) -> dict[str, Any]:
    """Audit local release readiness without producing or publishing a bundle."""
    root = _assert_no_link_or_reparse_components(root, label="workspace root").resolve(
        strict=True
    )
    required_content = _preflight_required_content(root)
    manuscript = _preflight_manuscript(root)
    raw_spec: Mapping[str, Any] | None = None
    spec_inventory: dict[str, Any] = {
        "passed": False,
        "status": "not_evaluated_without_spec",
        "selected_files": 0,
        "evidence_roles": [],
        "evidence": [],
        "third_party_or_raw_selected_paths": [],
    }
    selected: list[SelectedFile] = []
    categories: dict[str, str] = {}
    if spec_path is not None:
        try:
            resolved_spec = _assert_no_link_or_reparse_components(
                spec_path, label="preflight release spec"
            ).resolve(strict=True)
            raw_spec = _load_json(resolved_spec, label="preflight release spec")
            spec_inventory, selected, categories = _preflight_spec_inventory(
                raw_spec, root=root
            )
            spec_inventory["status"] = "inventory_and_hashes_validated"
        except (OSError, ReleaseValidationError) as exc:
            spec_inventory = {
                **spec_inventory,
                "status": "failed",
                "error": str(exc),
            }

    metadata = _preflight_metadata_report(
        raw_spec.get("release") if isinstance(raw_spec, Mapping) else None
    )
    evidence_contracts: dict[str, Any] = {
        "passed": False,
        "status": "not_run",
        "reason": (
            "full contract replay requires a valid spec, all detached trust "
            "receipts, and the pinned official GATDep checkout"
        ),
    }
    if (
        raw_spec is not None
        and spec_inventory.get("passed") is True
        and trust_receipt_paths is not None
        and gatdep_official_root is not None
    ):
        try:
            trust_receipts, receipt_summaries = _load_detached_trust_receipts(
                trust_receipt_paths, root=root
            )
            if (
                _validate_embedded_trust_receipts(raw_spec.get("trust_receipts"))
                != receipt_summaries
            ):
                _fail(
                    "preflight detached trust receipt summaries differ from the inputs"
                )
            evidence, authorized = _validate_evidence(
                raw_spec.get("evidence"),
                root=root,
                categories=categories,
                selected=selected,
                gatdep_official_root=gatdep_official_root,
                trust_receipts=trust_receipts,
            )
            for item in selected:
                if item.category in {"evidence", "prediction", "provenance"} and (
                    item.category not in authorized.get(item.relative, set())
                ):
                    _fail(
                        f"{item.relative} is not hash-bound by a registered evidence "
                        f"contract for category {item.category}"
                    )
            _assert_selected_files_unchanged(root, selected)
            evidence_contracts = {
                "passed": True,
                "status": "registered_contracts_replayed",
                "evidence": evidence,
            }
        except (OSError, ReleaseValidationError) as exc:
            evidence_contracts = {
                "passed": False,
                "status": "failed",
                "error": str(exc),
            }

    machine_checks_passed = all(
        (
            required_content["passed"],
            manuscript["dependency_graph"]["passed"],
            manuscript["submission_placeholders"]["passed"],
            manuscript["clinical_annotation"]["passed"],
            spec_inventory["passed"],
            evidence_contracts["passed"],
        )
    )
    return {
        "schema": PREFLIGHT_SCHEMA,
        "preflight_completed": True,
        "production_release_created": False,
        "workspace_root": str(root),
        "spec_provided": spec_path is not None,
        "checks": {
            "required_content": required_content,
            "manuscript": manuscript,
            "spec_inventory_and_third_party_exclusion": spec_inventory,
            "registered_evidence_contracts": evidence_contracts,
        },
        "machine_checks_passed": machine_checks_passed,
        "metadata": metadata,
        "ready_for_build": machine_checks_passed and metadata["local_syntax_valid"],
        "ready_for_publication": False,
        "doi_reservation_verified": False,
        "public_deposit_verified": False,
        "claim_boundary": (
            "Preflight is a local readiness audit. It does not create a release, "
            "verify author statements or redistribution rights, reserve a DOI, "
            "or establish a public deposit or scientific validity."
        ),
    }


def validate_spec(
    spec: Mapping[str, Any],
    *,
    root: Path,
    trust_receipt_paths: Mapping[str, Path],
    gatdep_official_root: Path | None = None,
) -> tuple[dict[str, Any], list[SelectedFile], list[dict[str, str]]]:
    if (
        set(spec) != {"schema", "release", "files", "evidence", "trust_receipts"}
        or spec.get("schema") != SPEC_SCHEMA
    ):
        _fail(
            f"release spec must use {SPEC_SCHEMA} and contain exactly "
            "schema/release/files/evidence/trust_receipts"
        )
    trust_receipts, receipt_summaries = _load_detached_trust_receipts(
        trust_receipt_paths,
        root=root,
    )
    if (
        _validate_embedded_trust_receipts(spec.get("trust_receipts"))
        != receipt_summaries
    ):
        _fail("release spec detached trust receipt summaries differ from the inputs")
    release = _validate_release_metadata(spec["release"])
    selected, categories = _select_files(root, spec["files"])
    _enforce_required_content(categories)
    _validate_corrected_campaign_provenance_bindings(root, selected=selected)
    manuscript_dependencies = _manuscript_dependency_closure(root, categories)
    if manuscript_dependencies != ACTIVE_MANUSCRIPT_DEPENDENCIES:
        missing = sorted(ACTIVE_MANUSCRIPT_DEPENDENCIES - manuscript_dependencies)
        unexpected = sorted(manuscript_dependencies - ACTIVE_MANUSCRIPT_DEPENDENCIES)
        _fail(
            "active manuscript dependency graph differs from its reviewed exact "
            f"inventory; missing={missing}, unexpected={unexpected}"
        )
    evidence, authorized = _validate_evidence(
        spec["evidence"],
        root=root,
        categories=categories,
        selected=selected,
        gatdep_official_root=gatdep_official_root,
        trust_receipts=trust_receipts,
    )
    for item in selected:
        if item.category in {
            "evidence",
            "prediction",
            "provenance",
        } and item.category not in authorized.get(item.relative, set()):
            _fail(
                f"{item.relative} is not hash-bound by a registered evidence contract "
                f"for category {item.category}"
            )
    _assert_selected_files_unchanged(root, selected)
    normalized = {
        "schema": SPEC_SCHEMA,
        "release": release,
        "files": [
            {"path": item.relative, "category": item.category} for item in selected
        ],
        "evidence": [{"role": item["role"], "path": item["path"]} for item in evidence],
        "trust_receipts": receipt_summaries,
    }
    return normalized, selected, evidence


def _embedded_spec_inventory(
    spec: Mapping[str, Any],
) -> tuple[dict[str, str], list[dict[str, str]]]:
    if (
        set(spec) != {"schema", "release", "files", "evidence", "trust_receipts"}
        or spec.get("schema") != SPEC_SCHEMA
    ):
        _fail("embedded release spec fields or schema are invalid")
    _validate_release_metadata(spec.get("release"))
    _validate_embedded_trust_receipts(spec.get("trust_receipts"))
    records = spec.get("files")
    if not isinstance(records, list) or not records:
        _fail("embedded release file allowlist is empty")
    categories: dict[str, str] = {}
    folded_paths: set[str] = set()
    observed_order: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {"path", "category"}:
            _fail(f"embedded files[{index}] is invalid")
        relative = _relative_path(record["path"], field=f"embedded files[{index}].path")
        category = record["category"]
        if (
            not isinstance(category, str)
            or category not in ALLOWED_CATEGORIES
            or not _path_allowed_for_category(relative, category)
        ):
            _fail(f"embedded files[{index}] violates its category route")
        if relative.casefold() in folded_paths:
            _fail(f"embedded file allowlist duplicates or case-collides: {relative}")
        folded_paths.add(relative.casefold())
        categories[relative] = category
        observed_order.append(relative)
    if observed_order != sorted(observed_order):
        _fail("embedded file allowlist is not in canonical path order")
    _enforce_required_content(categories)

    evidence_records = spec.get("evidence")
    if not isinstance(evidence_records, list):
        _fail("embedded evidence inventory is invalid")
    normalized_evidence: list[dict[str, str]] = []
    roles: set[str] = set()
    for index, record in enumerate(evidence_records):
        if not isinstance(record, dict) or set(record) != {"role", "path"}:
            _fail(f"embedded evidence[{index}] is invalid")
        role = record["role"]
        if not isinstance(role, str):
            _fail(f"embedded evidence[{index}].role must be a string")
        relative = _relative_path(
            record["path"], field=f"embedded evidence[{index}].path"
        )
        if role not in REQUIRED_EVIDENCE_ROLES or role in roles:
            _fail(f"embedded evidence role is unsupported or duplicated: {role!r}")
        expected_category = (
            "provenance" if role == "esm_embedding_provenance" else "evidence"
        )
        if categories.get(relative) != expected_category:
            _fail(f"embedded evidence path/category mismatch for role {role}")
        roles.add(role)
        normalized_evidence.append({"role": role, "path": relative})
    if roles != REQUIRED_EVIDENCE_ROLES:
        _fail("embedded evidence roles are incomplete")
    if normalized_evidence != sorted(
        normalized_evidence, key=lambda item: item["role"]
    ):
        _fail("embedded evidence inventory is not in canonical role order")
    return categories, normalized_evidence


def _zip_info(member: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(member, date_time=ZIP_FIXED_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.flag_bits = 0
    return info


def _partition(
    files: Sequence[SelectedFile], part_size: int
) -> list[list[SelectedFile]]:
    if part_size < MIN_PART_SIZE or part_size >= 2 * 1024 * 1024 * 1024:
        _fail("part size must be at least 65536 and strictly below 2 GiB")
    parts: list[list[SelectedFile]] = []
    current: list[SelectedFile] = []
    estimated = 4096
    for item in files:
        size = item.bytes
        member_overhead = 512 + 2 * len(item.relative.encode("utf-8"))
        if size + member_overhead + 4096 >= part_size:
            _fail(
                f"single file cannot fit below the configured archive limit: {item.relative}"
            )
        if current and estimated + size + member_overhead >= part_size:
            parts.append(current)
            current = []
            estimated = 4096
        current.append(item)
        estimated += size + member_overhead
    if current:
        parts.append(current)
    return parts


def _write_archive(
    target: Path | BinaryIO, files: Sequence[SelectedFile], *, prefix: str
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(target, os.PathLike):
        archive_context = zipfile.ZipFile(
            target,
            "x",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        )
    else:
        archive_context = zipfile.ZipFile(
            target,
            "w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        )
    with archive_context as archive:
        for item in files:
            member = f"{prefix}/{item.relative}"
            if (
                len(item.payload) != item.bytes
                or hashlib.sha256(item.payload).hexdigest() != item.sha256
            ):
                _fail(f"bound snapshot payload changed: {item.relative}")
            with archive.open(
                _zip_info(member), "w", force_zip64=True
            ) as member_stream:
                member_stream.write(item.payload)
            records.append(
                {
                    "path": item.relative,
                    "category": item.category,
                    "archive_member": member,
                    "bytes": item.bytes,
                    "sha256": item.sha256,
                }
            )
    return records


def _deposit_metadata(release: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "title": release["title"],
        "version": release["version"],
        "doi": release["author_supplied_doi"],
        "doi_claim_status": "author_supplied_not_independently_verified",
        "publication_date": release["publication_date"],
        "license": release["license"],
        "creators": release["creators"],
        "funding": release["funding"],
        "conflict_of_interest": release["conflict_of_interest"],
        "ai_use_disclosure": release["ai_use_disclosure"],
        "upload_type": "dataset",
    }


def _prepare_output_parent(
    output_dir: Path,
    *,
    owned_handles: list[BoundDirectoryHandle] | None = None,
) -> tuple[Path, Path, DirectoryIdentity, BoundDirectoryHandle]:
    """Create and bind a safe lexical output parent before staging is written."""
    output = _lexical_absolute(output_dir)
    try:
        output.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise ReleaseValidationError(
            f"cannot inspect release output path: {output}"
        ) from exc
    else:
        _fail(f"refusing to overwrite release directory: {output}")

    nearest = output.parent
    while True:
        try:
            nearest.lstat()
        except FileNotFoundError:
            parent = nearest.parent
            if parent == nearest:
                _fail("release output has no existing safe ancestor")
            nearest = parent
            continue
        except OSError as exc:
            raise ReleaseValidationError(
                f"cannot inspect release output parent: {nearest}"
            ) from exc
        break
    nearest, nearest_identity = _bind_directory(
        nearest,
        label="release output ancestor",
    )
    registered_handles = owned_handles if owned_handles is not None else []
    registration_start = len(registered_handles)
    current: BoundDirectoryHandle | None = None
    try:
        current = _open_bound_directory_handle(
            nearest,
            nearest_identity,
            label="release output ancestor",
            owned_handles=registered_handles,
        )
        for component in output.parent.relative_to(nearest).parts:
            child = _create_bound_child_directory(
                current,
                component,
                label="release output parent component",
                owned_handles=registered_handles,
            )
            previous = current
            current = child
            try:
                previous.close()
            except BaseException as primary:
                primary_traceback = primary.__traceback__
                raise primary.with_traceback(primary_traceback)
        parent = current.path
        identity = current.identity
        _assert_directory_identity(
            parent,
            identity,
            label="release output parent",
        )
        try:
            output.lstat()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise ReleaseValidationError(
                f"cannot inspect release output path: {output}"
            ) from exc
        else:
            _fail(f"refusing to overwrite release directory: {output}")
        return output, parent, identity, current
    except BaseException as primary:
        primary_traceback = primary.__traceback__
        newly_registered = registered_handles[registration_start:]
        created_component = newly_registered[1] if len(newly_registered) > 1 else None
        if created_component is not None:
            try:
                _quarantine_bound_directory(
                    created_component.path,
                    created_component.identity,
                    label="release output parent component",
                    parent_handle=(
                        newly_registered[0]
                        if newly_registered[0].descriptor >= 0
                        else None
                    ),
                    child_handle=(
                        created_component if created_component.descriptor >= 0 else None
                    ),
                )
            except BaseException:
                pass
        _close_owned_directory_handles(
            newly_registered,
            active_exception=primary,
        )
        raise primary.with_traceback(primary_traceback)


def build_release(
    *,
    spec_path: Path,
    output_dir: Path,
    trust_receipt_paths: Mapping[str, Path],
    root: Path = ROOT,
    part_size: int = DEFAULT_PART_SIZE,
    gatdep_official_root: Path | None = None,
) -> Path:
    root = _assert_no_link_or_reparse_components(root, label="workspace root").resolve(
        strict=True
    )
    executing_builder = _bind_executing_builder(root)
    spec_path = _assert_no_link_or_reparse_components(
        spec_path, label="release spec"
    ).resolve(strict=True)
    raw_spec = _load_json(spec_path, label="release spec")
    spec, selected, evidence = validate_spec(
        raw_spec,
        root=root,
        trust_receipt_paths=trust_receipt_paths,
        gatdep_official_root=gatdep_official_root,
    )
    _assert_executing_builder_selected(executing_builder, selected)
    output_parent_handles: list[BoundDirectoryHandle] = []
    try:
        (
            output_dir,
            output_parent,
            output_parent_identity,
            output_parent_handle,
        ) = _prepare_output_parent(
            output_dir,
            owned_handles=output_parent_handles,
        )
    except BaseException as primary:
        primary_traceback = primary.__traceback__
        _close_owned_directory_handles(
            output_parent_handles,
            active_exception=primary,
        )
        raise primary.with_traceback(primary_traceback)
    staging_name = f".{output_dir.name}.tmp-{uuid.uuid4().hex}"
    staging = output_parent / staging_name
    staging_source_parent = output_parent_handle
    staging_container_handles: list[BoundDirectoryHandle] = []
    staging_handles: list[BoundDirectoryHandle] = []
    archives_handles: list[BoundDirectoryHandle] = []
    output_descriptor_owners: list[_OwnedDescriptor] = []
    staging_container_handle: BoundDirectoryHandle | None = None
    staging_identity: DirectoryIdentity | None = None
    staging_handle: BoundDirectoryHandle | None = None
    archives_handle: BoundDirectoryHandle | None = None
    installed_root_identity: DirectoryIdentity | None = None
    try:
        if os.name == "nt":
            staging_handle = _create_bound_child_directory(
                output_parent_handle,
                staging_name,
                label="release staging directory",
                owned_handles=staging_handles,
            )
        else:
            staging_container_handle = _create_bound_child_directory(
                output_parent_handle,
                staging_name,
                label="release private staging parent",
                owned_handles=staging_container_handles,
            )
            staging_source_parent = staging_container_handle
            staging_handle = _create_bound_child_directory(
                staging_container_handle,
                "release-payload",
                label="release staging directory",
                owned_handles=staging_handles,
            )
        staging = staging_handle.path
        staging_identity = staging_handle.identity
        _assert_directory_identity(
            output_parent,
            output_parent_identity,
            label="release output parent",
        )
        snapshot_root = staging / ".bound-source-snapshot"
        snapshot_selected = _snapshot_selected_files(
            root,
            snapshot_root,
            selected,
        )
        parts = _partition(snapshot_selected, part_size)
        archives_dir = staging / "archives"
        archives_handle = _create_bound_child_directory(
            staging_handle,
            "archives",
            label="release archives directory",
            owned_handles=archives_handles,
        )
        release_prefix = f"DepPLM-{spec['release']['version']}"
        archive_records: list[dict[str, Any]] = []
        archive_fingerprints: dict[str, FileFingerprint] = {}
        file_records: list[dict[str, Any]] = []
        for index, files in enumerate(parts, start=1):
            archive_name = f"DepPLM-{spec['release']['version']}-part-{index:04d}.zip"
            archive_path = archives_dir / archive_name
            archive_relative = f"archives/{archive_name}"
            with _open_new_file_at(
                archives_handle,
                archive_name,
                label=f"archive {archive_relative}",
                owned_descriptors=output_descriptor_owners,
            ) as archive_handle:
                members = _write_archive(
                    archive_handle,
                    files,
                    prefix=release_prefix,
                )
                archive_fingerprint = _fingerprint_open_file(
                    archive_handle,
                    label=f"archive {archive_relative}",
                )
            if archive_fingerprint.bytes >= part_size:
                _fail(f"archive exceeds configured part limit: {archive_path.name}")
            archive_fingerprints[archive_relative] = archive_fingerprint
            archive_records.append(
                {
                    "path": archive_relative,
                    "bytes": archive_fingerprint.bytes,
                    "sha256": archive_fingerprint.sha256,
                    "member_count": len(members),
                }
            )
            for record in members:
                file_records.append({**record, "archive": archive_relative})
        _close_owned_output_descriptors(output_descriptor_owners)
        archives_handle.close()
        archives_handle = None
        spec_bytes = _canonical_json_bytes(spec)
        spec_fingerprint = _write_new_bound_file(
            staging_handle,
            "release-spec.json",
            spec_bytes,
            label="release spec",
            owned_descriptors=output_descriptor_owners,
        )
        deposit_bytes = _canonical_json_bytes(_deposit_metadata(spec["release"]))
        deposit_fingerprint = _write_new_bound_file(
            staging_handle,
            "deposit-metadata.json",
            deposit_bytes,
            label="deposit metadata",
            owned_descriptors=output_descriptor_owners,
        )
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "release": spec["release"],
            "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
            "part_size_limit_bytes": part_size,
            "archive_format": "deterministic ZIP_STORED with fixed member timestamps",
            "evidence": evidence,
            "files": sorted(file_records, key=lambda item: item["path"]),
            "archives": archive_records,
            "totals": {
                "files": len(file_records),
                "archives": len(archive_records),
                "unpacked_bytes": sum(item["bytes"] for item in file_records),
                "archive_bytes": sum(item["bytes"] for item in archive_records),
            },
            "claim_boundary": (
                "Archive completeness and byte integrity do not by themselves establish "
                "scientific validity; the experiment-specific verified evidence remains controlling."
            ),
        }
        manifest_fingerprint = _write_new_bound_file(
            staging_handle,
            "release-manifest.json",
            _canonical_json_bytes(manifest),
            label="release manifest",
            owned_descriptors=output_descriptor_owners,
        )
        release_fingerprints = {
            "release-spec.json": spec_fingerprint,
            "deposit-metadata.json": deposit_fingerprint,
            "release-manifest.json": manifest_fingerprint,
            **archive_fingerprints,
        }
        checksum_text = _expected_checksum_text(
            release_fingerprints,
            release_fingerprints,
        )
        _write_new_bound_file(
            staging_handle,
            "SHA256SUMS.txt",
            checksum_text.encode("ascii"),
            label="release checksums",
            owned_descriptors=output_descriptor_owners,
        )
        _close_owned_output_descriptors(output_descriptor_owners)
        _assert_bound_directory_handle(
            output_parent_handle,
            label="release output parent",
        )
        _assert_bound_directory_handle(
            staging_handle,
            label="release staging directory",
        )
        _assert_directory_identity(
            output_parent,
            output_parent_identity,
            label="release output parent",
        )
        _assert_directory_identity(
            staging,
            staging_identity,
            label="release staging directory",
        )
        built_inventory = _release_inventory(staging)
        verify_release(staging)
        verified_inventory = _release_inventory(staging)
        if verified_inventory != built_inventory:
            _fail("staging release inventory changed during verification")
        _assert_directory_identity(
            output_parent,
            output_parent_identity,
            label="release output parent",
        )
        _assert_directory_identity(
            staging,
            staging_identity,
            label="release staging directory",
        )
        try:
            output_dir.lstat()
        except FileNotFoundError:
            pass
        else:
            _fail(f"refusing to overwrite release directory: {output_dir}")
        installed_root_identity = staging_identity
        _install_bound_child(
            output_parent_handle,
            staging_handle,
            output_dir.name,
            source_parent=staging_source_parent,
        )
        staging = output_dir
        _assert_directory_identity(
            output_parent,
            output_parent_identity,
            label="release output parent",
        )
        _assert_directory_identity(
            output_dir,
            installed_root_identity,
            label="installed release directory",
        )
        if _release_inventory(output_dir) != verified_inventory:
            _fail("installed release inventory differs from the verified staging tree")
        if staging_container_handle is not None:
            _remove_empty_bound_child(
                output_parent_handle,
                staging_container_handle,
                staging_name,
                label="release private staging parent",
            )
            staging_container_handle.close()
            staging_container_handle = None
        if archives_handle is not None:
            archives_handle.close()
            archives_handle = None
        if staging_handle is not None:
            staging_handle.close()
            staging_handle = None
        if staging_container_handle is not None:
            staging_container_handle.close()
            staging_container_handle = None
        output_parent_handle.close()
        return output_dir / "release-manifest.json"
    except BaseException as primary:
        primary_traceback = primary.__traceback__
        if staging_handle is None and staging_handles:
            staging_handle = staging_handles[-1]
        if staging_identity is None and staging_handle is not None:
            staging = staging_handle.path
            staging_identity = staging_handle.identity
        if staging_container_handle is None and staging_container_handles:
            staging_container_handle = staging_container_handles[-1]
        if archives_handle is None and archives_handles:
            archives_handle = archives_handles[-1]
        staging_quarantined = False
        # Windows cannot rename an ancestor while a descendant handle that omits
        # FILE_SHARE_DELETE remains open.  Exhaust the state-reconciled output-file
        # and archive-directory close attempts before each identity-bound staging
        # detachment attempt.
        for _cleanup_attempt in range(2):
            try:
                _close_owned_output_descriptors(
                    output_descriptor_owners,
                    active_exception=primary,
                )
            except BaseException:
                # An asynchronous exception at a cleanup helper's own trace
                # boundary cannot replace the primary; the registry remains
                # available for the next reconciliation attempt.
                pass
            _close_owned_directory_handles(
                archives_handles,
                active_exception=primary,
            )
            if installed_root_identity is not None and staging_handle is not None:
                try:
                    staging_quarantined = _quarantine_bound_directory(
                        output_dir,
                        installed_root_identity,
                        label="installed release directory",
                        parent_handle=(
                            output_parent_handle
                            if output_parent_handle.descriptor >= 0
                            else None
                        ),
                        child_handle=(
                            staging_handle if staging_handle.descriptor >= 0 else None
                        ),
                    )
                except BaseException:
                    pass
            if (
                not staging_quarantined
                and staging_identity is not None
                and staging_handle is not None
            ):
                try:
                    staging_quarantined = _quarantine_bound_directory(
                        staging,
                        staging_identity,
                        label="release staging directory",
                        parent_handle=(
                            staging_source_parent
                            if staging_source_parent.descriptor >= 0
                            else None
                        ),
                        child_handle=(
                            staging_handle if staging_handle.descriptor >= 0 else None
                        ),
                    )
                except BaseException:
                    pass
            if staging_quarantined:
                break
        if staging_container_handle is not None:
            try:
                _quarantine_bound_directory(
                    output_parent / staging_name,
                    staging_container_handle.identity,
                    label="release private staging parent",
                    parent_handle=(
                        output_parent_handle
                        if output_parent_handle.descriptor >= 0
                        else None
                    ),
                    child_handle=(
                        staging_container_handle
                        if staging_container_handle.descriptor >= 0
                        else None
                    ),
                )
            except BaseException:
                pass
        _close_owned_directory_handles(
            (
                *staging_handles,
                *staging_container_handles,
                *output_parent_handles,
            ),
            active_exception=primary,
        )
        raise primary.with_traceback(primary_traceback)


def _expected_checksum_text(
    fingerprints: Mapping[str, FileFingerprint], paths: Iterable[str]
) -> str:
    return "".join(
        f"{fingerprints[relative].sha256}  {relative}\n" for relative in sorted(paths)
    )


def _release_inventory(release_dir: Path) -> dict[str, tuple[Any, ...]]:
    """Fingerprint every entry without following links or opening special files."""
    root = _assert_no_link_or_reparse_components(release_dir, label="release directory")
    try:
        root_metadata = root.stat(follow_symlinks=False)
    except OSError as exc:
        raise ReleaseValidationError("release directory is unavailable") from exc
    if _is_link_or_reparse(root) or not stat.S_ISDIR(root_metadata.st_mode):
        _fail("release path is linked, reparse-backed, or not a directory")
    inventory: dict[str, tuple[Any, ...]] = {}
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            raise ReleaseValidationError(
                f"release directory cannot be enumerated: {directory}"
            ) from exc
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ReleaseValidationError(
                    f"release entry cannot be inspected: {relative}"
                ) from exc
            if _is_link_or_reparse(path):
                _fail(
                    f"release entry is a link, junction, or reparse point: {relative}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                inventory[relative] = (
                    "directory",
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_mode,
                    metadata.st_nlink,
                )
                pending.append(path)
            elif stat.S_ISREG(metadata.st_mode):
                fingerprint = _fingerprint_file(
                    path,
                    label=f"release entry {relative}",
                )
                inventory[relative] = (
                    "file",
                    fingerprint.bytes,
                    fingerprint.sha256,
                    fingerprint.device,
                    fingerprint.inode,
                    fingerprint.mode,
                    fingerprint.modified_ns,
                    fingerprint.links,
                )
            else:
                _fail(f"release entry is not a regular file or directory: {relative}")
    return inventory


def verify_release(release_dir: Path) -> dict[str, Any]:
    release_dir = _assert_no_link_or_reparse_components(
        release_dir, label="release directory"
    ).resolve(strict=True)
    initial_inventory = _release_inventory(release_dir)
    expected_top = {
        "archives",
        "release-spec.json",
        "deposit-metadata.json",
        "release-manifest.json",
        "SHA256SUMS.txt",
    }
    observed_top = {PurePosixPath(relative).parts[0] for relative in initial_inventory}
    if observed_top != expected_top:
        _fail("release directory contains missing or extra top-level entries")
    if initial_inventory.get("archives", (None,))[0] != "directory" or any(
        initial_inventory.get(name, (None,))[0] != "file"
        for name in expected_top
        if name != "archives"
    ):
        _fail("release top-level entries contain links or invalid file types")

    manifest_bytes, manifest_fingerprint = _read_stable_bytes(
        release_dir / "release-manifest.json", label="release manifest"
    )
    spec_bytes, spec_fingerprint = _read_stable_bytes(
        release_dir / "release-spec.json", label="release spec"
    )
    deposit_bytes, deposit_fingerprint = _read_stable_bytes(
        release_dir / "deposit-metadata.json", label="deposit metadata"
    )
    manifest = _parse_json_bytes(
        manifest_bytes,
        label="release manifest",
        path=release_dir / "release-manifest.json",
    )
    spec = _parse_json_bytes(
        spec_bytes,
        label="release spec",
        path=release_dir / "release-spec.json",
    )
    deposit = _parse_json_bytes(
        deposit_bytes,
        label="deposit metadata",
        path=release_dir / "deposit-metadata.json",
    )
    if manifest_bytes != _canonical_json_bytes(manifest):
        _fail("release manifest JSON is not canonical")
    if spec_bytes != _canonical_json_bytes(spec):
        _fail("release spec JSON is not canonical")
    if deposit_bytes != _canonical_json_bytes(deposit):
        _fail("deposit metadata JSON is not canonical")
    if manifest.get("schema") != MANIFEST_SCHEMA or spec.get("schema") != SPEC_SCHEMA:
        _fail("release manifest or embedded spec schema is invalid")
    required_manifest_fields = {
        "schema",
        "release",
        "spec_sha256",
        "part_size_limit_bytes",
        "archive_format",
        "evidence",
        "files",
        "archives",
        "totals",
        "claim_boundary",
    }
    if set(manifest) != required_manifest_fields:
        _fail("release manifest fields are incomplete or unexpected")
    if spec_fingerprint.sha256 != manifest.get("spec_sha256"):
        _fail("embedded release spec hash differs")
    _validate_release_metadata(spec.get("release"))
    spec_categories, spec_evidence = _embedded_spec_inventory(spec)
    if manifest.get("release") != spec.get("release"):
        _fail("release metadata differs between spec and manifest")
    part_size_limit = manifest.get("part_size_limit_bytes")
    if (
        not isinstance(part_size_limit, int)
        or isinstance(part_size_limit, bool)
        or not MIN_PART_SIZE <= part_size_limit < 2 * 1024 * 1024 * 1024
    ):
        _fail("release part-size limit is invalid")
    if (
        manifest.get("archive_format")
        != "deterministic ZIP_STORED with fixed member timestamps"
    ):
        _fail("release archive format declaration is invalid")
    archive_records = manifest.get("archives")
    file_records = manifest.get("files")
    if (
        not isinstance(archive_records, list)
        or not archive_records
        or not isinstance(file_records, list)
        or not file_records
    ):
        _fail("release archive or file inventory is empty")
    archive_paths: set[str] = set()
    archive_paths_casefold: set[str] = set()
    archive_record_by_path: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(archive_records):
        if not isinstance(record, dict) or set(record) != {
            "path",
            "bytes",
            "sha256",
            "member_count",
        }:
            _fail(f"archives[{index}] record is invalid")
        relative = _relative_path(record["path"], field=f"archives[{index}].path")
        folded_relative = relative.casefold()
        if folded_relative in archive_paths_casefold:
            _fail(f"duplicate or case-colliding archive path: {relative}")
        if (
            PurePosixPath(relative).parent != PurePosixPath("archives")
            or PurePosixPath(relative).suffix != ".zip"
        ):
            _fail(f"archives[{index}].path is outside the flat ZIP inventory")
        size = record["bytes"]
        sha256 = record["sha256"]
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(sha256, str)
            or not SHA256_RE.fullmatch(sha256)
        ):
            _fail(f"archives[{index}] has invalid bytes or SHA-256")
        archive_paths.add(relative)
        archive_paths_casefold.add(folded_relative)
        archive_record_by_path[relative] = record
        member_count = record["member_count"]
        if (
            not isinstance(member_count, int)
            or isinstance(member_count, bool)
            or member_count <= 0
        ):
            _fail(f"archives[{index}].member_count is invalid")
        if size >= part_size_limit:
            _fail(f"archive is not below its declared part limit: {relative}")
    if [record["path"] for record in archive_records] != sorted(archive_paths):
        _fail("archive inventory is not in canonical path order")
    actual_archive_entries = {
        relative
        for relative in initial_inventory
        if relative != "archives" and relative.startswith("archives/")
    }
    if actual_archive_entries != archive_paths or any(
        initial_inventory[relative][0] != "file" for relative in actual_archive_entries
    ):
        _fail("archives directory contains missing, extra, nested, or linked entries")
    by_archive: dict[str, list[dict[str, Any]]] = {path: [] for path in archive_paths}
    seen_paths: set[str] = set()
    seen_members: set[str] = set()
    for index, record in enumerate(file_records):
        required = {"path", "category", "archive_member", "bytes", "sha256", "archive"}
        if not isinstance(record, dict) or set(record) != required:
            _fail(f"files[{index}] record is invalid")
        relative = _relative_path(record["path"], field=f"files[{index}].path")
        member = _relative_path(
            record["archive_member"], field=f"files[{index}].archive_member"
        )
        archive = _relative_path(record["archive"], field=f"files[{index}].archive")
        if relative in seen_paths or member in seen_members:
            _fail("release file inventory contains duplicate paths or members")
        category = record["category"]
        if (
            not isinstance(category, str)
            or category not in ALLOWED_CATEGORIES
            or not _path_allowed_for_category(relative, category)
            or archive not in by_archive
        ):
            _fail(f"files[{index}] has invalid category or archive binding")
        expected_member = f"DepPLM-{spec['release']['version']}/{relative}"
        if member != expected_member:
            _fail(f"files[{index}] has a noncanonical archive member path")
        if (
            not isinstance(record["bytes"], int)
            or isinstance(record["bytes"], bool)
            or record["bytes"] < 0
            or not isinstance(record["sha256"], str)
            or not SHA256_RE.fullmatch(record["sha256"])
        ):
            _fail(f"files[{index}] has invalid bytes or SHA-256")
        seen_paths.add(relative)
        seen_members.add(member)
        by_archive[archive].append(record)
    if [record["path"] for record in file_records] != sorted(seen_paths):
        _fail("release file inventory is not in canonical path order")
    manifest_categories = {
        record["path"]: record["category"] for record in file_records
    }
    if manifest_categories != spec_categories:
        _fail("release file inventory differs from the embedded explicit allowlist")

    manifest_evidence = manifest.get("evidence")
    if not isinstance(manifest_evidence, list):
        _fail("release evidence hash inventory is invalid")
    expected_evidence = []
    file_by_path = {record["path"]: record for record in file_records}
    for index, record in enumerate(manifest_evidence):
        if not isinstance(record, dict) or set(record) != {"role", "path", "sha256"}:
            _fail(f"release evidence[{index}] record is invalid")
        role = record["role"]
        raw_path = record["path"]
        if not isinstance(role, str) or role not in REQUIRED_EVIDENCE_ROLES:
            _fail(f"release evidence[{index}].role is invalid")
        if not isinstance(raw_path, str):
            _fail(f"release evidence[{index}].path must be a string")
        relative = _relative_path(raw_path, field=f"release evidence[{index}].path")
        if not isinstance(record["sha256"], str) or not SHA256_RE.fullmatch(
            record["sha256"]
        ):
            _fail(f"release evidence[{index}] SHA-256 is invalid")
        expected_evidence.append({"role": role, "path": relative})
        file_record = file_by_path.get(relative)
        if file_record is None or file_record["sha256"] != record["sha256"]:
            _fail(f"release evidence[{index}] is not bound to its archived bytes")
    if expected_evidence != spec_evidence:
        _fail("release evidence inventory differs from the embedded spec")
    archive_fingerprints: dict[str, FileFingerprint] = {}
    expected_active_tex = {
        relative
        for relative in ACTIVE_MANUSCRIPT_DEPENDENCIES
        if Path(relative).suffix.casefold() == ".tex"
    }
    verified_active_tex: set[str] = set()
    for archive_relative, expected in by_archive.items():
        archive_path = _safe_workspace_file(
            release_dir, archive_relative, field="release archive"
        )
        with tempfile.SpooledTemporaryFile(max_size=16 * 1024 * 1024) as snapshot:
            archive_fingerprint = _stream_stable_file(
                archive_path,
                label=f"release archive {archive_relative}",
                target=snapshot,
            )
            archive_record = archive_record_by_path[archive_relative]
            if (
                archive_fingerprint.bytes != archive_record["bytes"]
                or archive_fingerprint.sha256 != archive_record["sha256"]
            ):
                _fail(f"release archive differs from manifest: {archive_relative}")
            archive_fingerprints[archive_relative] = archive_fingerprint
            snapshot.seek(0)
            with zipfile.ZipFile(snapshot, "r") as zip_handle:
                infos = zip_handle.infolist()
                names = [info.filename for info in infos]
                expected_names = [item["archive_member"] for item in expected]
                if len(names) != len(set(names)) or names != expected_names:
                    _fail(f"archive member inventory differs: {archive_relative}")
                if len(infos) != archive_record["member_count"]:
                    _fail(f"archive member count differs: {archive_relative}")
                if zip_handle.comment != b"":
                    _fail(f"archive comment is noncanonical: {archive_relative}")
                record_by_member = {item["archive_member"]: item for item in expected}
                for info in infos:
                    try:
                        info.filename.encode("ascii")
                        expected_flag_bits = 0
                    except UnicodeEncodeError:
                        expected_flag_bits = 0x800
                    if (
                        info.is_dir()
                        or info.compress_type != zipfile.ZIP_STORED
                        or info.date_time != ZIP_FIXED_TIME
                        or info.create_system != 3
                        or info.external_attr != (stat.S_IFREG | 0o644) << 16
                        or info.flag_bits != expected_flag_bits
                        or info.extra != b""
                        or info.comment != b""
                    ):
                        _fail(
                            "archive member metadata is noncanonical: "
                            f"{info.filename}"
                        )
                    digest = hashlib.sha256()
                    size = 0
                    with zip_handle.open(info, "r") as handle:
                        while chunk := handle.read(4 * 1024 * 1024):
                            digest.update(chunk)
                            size += len(chunk)
                    record = record_by_member[info.filename]
                    if (
                        size != record["bytes"]
                        or digest.hexdigest() != record["sha256"]
                    ):
                        _fail(f"archive member differs from manifest: {info.filename}")
                    relative = record["path"]
                    if relative in expected_active_tex:
                        with zip_handle.open(info, "r") as handle:
                            payload = handle.read()
                        text = _decode_public_text(payload, relative=relative)
                        _validate_active_manuscript_source(
                            text.splitlines(), relative=relative
                        )
                        verified_active_tex.add(relative)
    if verified_active_tex != expected_active_tex:
        _fail(
            "release archive is missing active manuscript sources required for "
            "submission-placeholder validation"
        )
    totals = manifest.get("totals", {})
    expected_totals = {
        "files": len(file_records),
        "archives": len(archive_records),
        "unpacked_bytes": sum(item["bytes"] for item in file_records),
        "archive_bytes": sum(item["bytes"] for item in archive_records),
    }
    if totals != expected_totals:
        _fail("release totals differ from the exact inventories")
    checksum_members = {
        "release-spec.json",
        "deposit-metadata.json",
        "release-manifest.json",
        *archive_paths,
    }
    checksum_fingerprints = {
        "release-spec.json": spec_fingerprint,
        "deposit-metadata.json": deposit_fingerprint,
        "release-manifest.json": manifest_fingerprint,
        **archive_fingerprints,
    }
    expected_checksums = _expected_checksum_text(
        checksum_fingerprints, checksum_members
    )
    try:
        checksum_bytes, _ = _read_stable_bytes(
            release_dir / "SHA256SUMS.txt", label="SHA256SUMS.txt"
        )
        observed_checksums = checksum_bytes.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ReleaseValidationError("SHA256SUMS.txt is unreadable") from exc
    if observed_checksums != expected_checksums:
        _fail("SHA256SUMS.txt differs from the release contents")
    if deposit != _deposit_metadata(spec["release"]):
        _fail("deposit metadata differs from the frozen release metadata")
    final_inventory = _release_inventory(release_dir)
    if final_inventory != initial_inventory:
        changed = sorted(
            relative
            for relative in set(initial_inventory) | set(final_inventory)
            if initial_inventory.get(relative) != final_inventory.get(relative)
        )
        _fail(f"release inventory changed during verification: {changed}")
    return {
        "schema": "DepPLM-public-release-verification-v2",
        "verified": True,
        "verification_scope": "package_bytes_and_inventory_only",
        "package_integrity_verified": True,
        "doi": spec["release"]["author_supplied_doi"],
        "doi_reservation_verified": False,
        "public_deposit_verified": False,
        "version": spec["release"]["version"],
        "files": len(file_records),
        "archives": len(archive_records),
        "archive_bytes": expected_totals["archive_bytes"],
    }


def _parse_trust_receipt_arguments(values: Sequence[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for index, value in enumerate(values):
        if not isinstance(value, str) or "=" not in value:
            _fail(f"--trust-receipt entry {index} must have the form ROLE=PATH")
        role, raw_path = value.split("=", 1)
        if role not in DETACHED_TRUST_RECEIPT_ROLES:
            _fail(f"--trust-receipt has unsupported role: {role!r}")
        if role in parsed:
            _fail(f"--trust-receipt duplicates role: {role}")
        if not raw_path:
            _fail(f"--trust-receipt path is empty for role: {role}")
        parsed[role] = Path(raw_path)
    if set(parsed) != DETACHED_TRUST_RECEIPT_ROLES:
        missing = sorted(DETACHED_TRUST_RECEIPT_ROLES - set(parsed))
        _fail(f"--trust-receipt is missing required roles: {missing}")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser(
        "build", help="validate evidence and build immutable ZIP parts"
    )
    build.add_argument("--spec", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--workspace-root", type=Path, default=ROOT)
    build.add_argument("--part-size", type=int, default=DEFAULT_PART_SIZE)
    build.add_argument(
        "--trust-receipt",
        action="append",
        required=True,
        metavar="ROLE=PATH",
        help=(
            "detached local trust receipt outside the workspace; repeat once for "
            "each release-binding role"
        ),
    )
    build.add_argument(
        "--gatdep-official-root",
        type=Path,
        required=True,
        help=(
            "local checkout of the pinned official GATDep repository used only "
            "for build-time replay; it is never included in the release"
        ),
    )
    preflight = subparsers.add_parser(
        "preflight",
        help="audit release readiness without creating or publishing a bundle",
    )
    preflight.add_argument("--workspace-root", type=Path, default=ROOT)
    preflight.add_argument(
        "--spec",
        type=Path,
        help="optional draft release specification to audit",
    )
    preflight.add_argument(
        "--trust-receipt",
        action="append",
        default=[],
        metavar="ROLE=PATH",
        help=(
            "optional detached receipt; if supplied, all release-binding roles "
            "must be supplied for evidence-contract replay"
        ),
    )
    preflight.add_argument(
        "--gatdep-official-root",
        type=Path,
        help="optional pinned official GATDep checkout for full contract replay",
    )
    verify = subparsers.add_parser(
        "verify", help="rehash and inspect an existing release"
    )
    verify.add_argument("release_dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            build_trust_receipt_paths = _parse_trust_receipt_arguments(
                args.trust_receipt
            )
            path = build_release(
                spec_path=args.spec,
                output_dir=args.output_dir,
                trust_receipt_paths=build_trust_receipt_paths,
                root=args.workspace_root,
                part_size=args.part_size,
                gatdep_official_root=args.gatdep_official_root,
            )
            report = verify_release(path.parent)
        elif args.command == "preflight":
            preflight_trust_receipt_paths = (
                _parse_trust_receipt_arguments(args.trust_receipt)
                if args.trust_receipt
                else None
            )
            report = preflight_release(
                root=args.workspace_root,
                spec_path=args.spec,
                trust_receipt_paths=preflight_trust_receipt_paths,
                gatdep_official_root=args.gatdep_official_root,
            )
        else:
            report = verify_release(args.release_dir)
        print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
        return 0
    except (OSError, ReleaseValidationError, zipfile.BadZipFile) as exc:
        print(
            json.dumps({"verified": False, "error": str(exc)}, ensure_ascii=True),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
