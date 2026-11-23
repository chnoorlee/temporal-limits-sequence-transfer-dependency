"""Independent, fail-closed verification of the long-protein analysis.

This verifier deliberately does not import :mod:`long_protein_analysis`.  It
replays the scientific calculation from the hash-bound outer-test prediction
files and canonical metadata, then compares that replay with every published
analysis artifact.  A passing report is a verification record, not a new
scientific result.
"""

from __future__ import annotations

import argparse
import base64
import builtins
from contextlib import contextmanager
import ctypes as _owner_ctypes
import csv
import errno
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import io
import json
import math
import ntpath
import os
import platform
import re
import stat
import sys
import sysconfig
import tempfile
import uuid
import weakref
import zipfile
import zipimport
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, BinaryIO, Iterator, Mapping, Sequence

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd  # type: ignore[import-untyped]
    import torch
    from scipy import stats  # type: ignore[import-untyped]
    from sklearn.decomposition import PCA  # type: ignore[import-untyped]
else:
    # Bound only after the verifier source and trusted completion digest are read.
    np = None
    pd = None
    torch = None
    stats = None
    PCA = None

ROOT = Path(__file__).resolve().parent.parent

CAMPAIGN_SCHEMA = "DepPLM-corrected-v2.2-campaign"
RUN_SCHEMA = "DepPLM-corrected-v2.2"
RUN_COMPLETION_SCHEMA = "DepPLM-corrected-v2.2-completion"
ANALYSIS_SCHEMA = "DepPLM-long-protein-analysis-v3"
ANALYSIS_MANIFEST_SCHEMA = "DepPLM-long-protein-analysis-run-v3"
ANALYSIS_COMPLETION_SCHEMA = "DepPLM-long-protein-analysis-completion-v3"
LEGACY_ANALYSIS_SCHEMA = "DepPLM-long-protein-analysis-v2"
LEGACY_ANALYSIS_MANIFEST_SCHEMA = "DepPLM-long-protein-analysis-run-v2"
LEGACY_ANALYSIS_COMPLETION_SCHEMA = "DepPLM-long-protein-analysis-completion-v2"
VERIFICATION_SCHEMA = "DepPLM-long-protein-independent-verification-v2"
PREDICTION_SCHEMA = "DepPLM-outer-test-predictions-v1"
EMBEDDING_MANIFEST_SCHEMA = "DepPLM-esm-embedding-v1"
EMBEDDING_PROVENANCE_SCHEMA = "DepPLM-verified-embedding-provenance-v1"

CONDITIONS = ("DepPLM", "C-PLM-MLP", "C-GCN")
CAMPAIGN_CONDITIONS = (
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
CAMPAIGN_PROTOCOLS = ("p1", "transfer-forward", "transfer-reverse")
SEEDS = (42, 43, 44)
FOLDS = tuple(f"fold-{index:02d}" for index in range(5))
LONG_THRESHOLD = 1022
EXPECTED_N_GENES = 17_768
EXPECTED_N_LONG = 2_152
EXPECTED_N_CELLS = 1_103
EXPECTED_CAMPAIGN_RUNS = 99
EXPECTED_FOLD_ARTIFACTS_BY_PROTOCOL = {
    "p1": 15,
    "transfer-forward": 3,
    "transfer-reverse": 3,
}
EXPECTED_FOLD_ARTIFACTS = 693
RAW_PREPROCESSING_INPUTS = (
    "data/depmap/OmicsExpressionProteinCodingGenesTPMLogp1.csv",
    "data/depmap/OmicsSomaticMutations.csv",
)
RAW_INPUT_SNAPSHOT_PREFIX = "depplm-long-protein-raw-"
ANALYSIS_ENVIRONMENT_FIELDS = {
    "packages",
    "platform",
    "python",
    "python_implementation",
}
ANALYSIS_ENVIRONMENT_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "torch",
)
LEGACY_ANALYSIS_ENVIRONMENT_PACKAGES = ("numpy", "scipy")
ANALYSIS_CONTRACTS = {
    ANALYSIS_COMPLETION_SCHEMA: {
        "analysis_schema": ANALYSIS_SCHEMA,
        "run_manifest_schema": ANALYSIS_MANIFEST_SCHEMA,
        "completion_schema": ANALYSIS_COMPLETION_SCHEMA,
        "current": True,
    },
    LEGACY_ANALYSIS_COMPLETION_SCHEMA: {
        "analysis_schema": LEGACY_ANALYSIS_SCHEMA,
        "run_manifest_schema": LEGACY_ANALYSIS_MANIFEST_SCHEMA,
        "completion_schema": LEGACY_ANALYSIS_COMPLETION_SCHEMA,
        "current": False,
    },
}
SCIENTIFIC_DEPENDENCY_MODULES = (
    ("numpy", "numpy", "numpy/__init__.py"),
    ("scipy", "scipy", "scipy/__init__.py"),
    ("scipy.stats", "scipy", "scipy/stats/__init__.py"),
    ("pandas", "pandas", "pandas/__init__.py"),
    ("sklearn", "scikit-learn", "sklearn/__init__.py"),
    (
        "sklearn.decomposition",
        "scikit-learn",
        "sklearn/decomposition/__init__.py",
    ),
    ("torch", "torch", "torch/__init__.py"),
)
SCIENTIFIC_DEPENDENCY_ROOTS = tuple(
    dict.fromkeys(
        name.split(".", 1)[0]
        for name, _distribution, _path in SCIENTIFIC_DEPENDENCY_MODULES
    )
)
TRUSTED_STARTUP_PRELOADED_DEPENDENCIES = {
    "_distutils_hack": "setuptools",
}
# This is intentionally closed. Additions require an exact module name, a
# builtin/frozen origin, and a documented reason recorded in verifier evidence.
TRUSTED_NONFILE_DEPENDENCY_MODULES: Mapping[str, str] = {}
_TRUSTED_SCIENTIFIC_DEPENDENCY_EVIDENCE: dict[str, Any] | None = None
_TRUSTED_SCIENTIFIC_DEPENDENCY_SNAPSHOTS: dict[str, FileSnapshot] = {}
_TRUSTED_SCIENTIFIC_DEPENDENCY_MODULE_PATHS: dict[str, str] = {}
_TRUSTED_DISTRIBUTION_ARTIFACTS: dict[str, DistributionArtifact] = {}
_TRUSTED_DEPENDENCY_CLOSURE: dict[str, tuple[str, str]] = {}
_TRUSTED_DEPENDENCY_IMPORT_ROOTS: dict[str, tuple[str, ...]] = {}
_TRUSTED_NONFILE_DEPENDENCY_BINDINGS: dict[str, tuple[Any, dict[str, Any]]] = {}
_TRUSTED_STARTUP_DEPENDENCY_BINDINGS: dict[str, Any] = {}
_TRUSTED_IMPORT_RUNTIME: TrustedImportRuntime | None = None
_SCIENTIFIC_DEPENDENCIES_BOUND_AFTER_TRUST = False
_OWNED_QUARANTINE_ENTRIES: set[tuple[int, int, str, str, int, int, bool, str]] = set()
_OWNED_DESCRIPTOR_TOKENS: dict[int, object] = {}
_OWNED_DESCRIPTOR_REFS: dict[int, weakref.ReferenceType[_OwnedDescriptor]] = {}
EXPECTED_PREDICTION_ARRAYS = (
    "prediction",
    "train_gene_mean",
    "outer_test_indices",
    "source_cell_indices",
    "source_gene_indices",
    "cell_ids",
    "gene_ids",
)
EXPECTED_PER_GENE_ARRAYS = (
    "gene_ids",
    "length_residues",
    "condition_ids",
    "seed_ids",
    "mae",
    "mse",
    "baseline_mae",
    "baseline_mse",
    "finite_target_counts",
)
RUN_ROOT_JSON_FILES = (
    "completion.json",
    "summary.json",
    "run-manifest.json",
    "cohort-manifest.json",
)
ANALYSIS_RUN_FILES = frozenset(
    {"completion.json", "run-manifest.json", "summary.json", "per-gene-errors.npz"}
)
ANALYSIS_MANIFEST_FIELDS = {
    "schema",
    "created_utc",
    "registered_parameters",
    "source",
    "artifacts",
}
REGISTERED_PARAMETERS = {
    "conditions": list(CONDITIONS),
    "seeds": list(SEEDS),
    "long_threshold_residues": LONG_THRESHOLD,
    "inference_scope": "conditional_descriptive_only",
}
CANONICAL_ANALYSIS_INPUTS = {
    "embedding_array": "data/processed/esm2_t6_8M_residue_chunked.npy",
    "embedding_manifest": "data/processed/esm2_t6_8M_residue_chunked.manifest.json",
    "fasta": "data/processed/gene_seqs.fa",
    "genes": "data/processed/genes.txt",
    "cells": "data/processed/cell_lines.txt",
    "gene_effect": "data/processed/gene_effect.npy",
}
ANALYSIS_SOURCE_PATHS = {
    "analysis_code": "code/long_protein_analysis.py",
    "campaign_validator_code": "code/aggregate_corrected_campaign.py",
    "embedding_validator_code": "code/embedding_provenance.py",
}
MODEL_REPOSITORY = "facebook/esm2_t6_8M_UR50D"
MODEL_REVISION = "c731040fcd8d73dceaa04b0a8e6329b345b0f5df"
EXPECTED_SNAPSHOT_FILES = {
    "config.json": {
        "bytes": 775,
        "sha256": "facb9355e8252149f6f99dff5fd1e32e890c3c655362b261d0c81cd6b5839c85",
    },
    "model.safetensors": {
        "bytes": 31_384_292,
        "sha256": "24c5fa474c48f3b754b86efe752d5f189d2bcd88190fa2270fc92b2ef3034189",
    },
    "special_tokens_map.json": {
        "bytes": 125,
        "sha256": "3aedcd4211c0d43aec4e607ff60a63255f3174ead795e997350f09a5f8cd9ee1",
    },
    "tokenizer_config.json": {
        "bytes": 95,
        "sha256": "7e9161ecdb548ec45a41cbc6b24aa4476fdd418461f491c4207baa99419a29ad",
    },
    "vocab.txt": {
        "bytes": 93,
        "sha256": "0b82cc0a7c7cf9e567b1e5892d793285b9fbae822c964ca48696f7db44598e03",
    },
}
EXPECTED_EMBEDDING_OUTPUTS = {
    "embeddings": {
        "path": "data/processed/esm2_t6_8M_residue_chunked.npy",
        "bytes": 22_743_168,
        "sha256": "60a3c09611afd1141bc6898f91523cf8b75cdd6dbf90f474f88b823318dab631",
    },
    "normalized_embeddings": {
        "path": "data/processed/esm2_t6_8M_residue_chunked.l2.npy",
        "bytes": 22_743_168,
        "sha256": "be2e4421e2c0893771fe94fe239738a13f7fc387d6a35105acb64cafc96e6e8d",
    },
    "names": {
        "path": "data/processed/esm2_t6_8M_residue_chunked.names.txt",
        "bytes": 113_021,
        "sha256": "0ec8b13f99cafd082da85f6fe0e086076d26bbc99878395caad9cf5a43128398",
    },
}
EXPECTED_FASTA_RECORD = {
    "path": "data/processed/gene_seqs.fa",
    "bytes": 10_717_335,
    "sha256": "c2859d727d1fc8ac4fd7b2bb42b1970d196eb6592586770d0749985d8f863c9a",
}
EXPECTED_GENERATOR_RECORD = {
    "path": "code/03_embed_proteins.py",
    "bytes": 31_259,
    "sha256": "8a34c0470577135b67ebb0d4f38a833c78c08590463a6693a97582cf8e8f3282",
}
EXPECTED_SEQUENCE_NAMES_SHA256 = (
    "8ef4a2027b88c132ba4c020f56b8bfbf5907308f364d08a3f9fdc643fed11e45"
)
EXPECTED_SEQUENCE_CONTENT_SHA256 = (
    "4c5c676cf1b8f935bb6973e8d536eee43026c2ac05549bccabc3fbc1f39f626c"
)
EXPECTED_LONG_NAMES_SHA256 = (
    "f611e80b719b7a0bdcce4c28198909591a0f1d02ef3dbae950224f7aff6bc543"
)
EXPECTED_ENVIRONMENT_INVENTORY_SHA256 = (
    "a3629e085fc728905d65236334b8fb004147d451bfbe6daba8319e21730cb769"
)
EXPECTED_KEY_VERSIONS = {
    "huggingface_hub": "2.2.0",
    "numpy": "2.5.4",
    "safetensors": "0.8.0",
    "torch": "2.11.0+cu128",
    "transformers": "5.20.1",
}
EMBEDDING_MANIFEST_FIELDS = {
    "argv",
    "created_utc",
    "elapsed_seconds",
    "environment",
    "generation",
    "generator",
    "input",
    "model",
    "outputs",
    "schema",
    "status",
}
EMBEDDING_MODEL_FIELDS = {
    "configuration",
    "model_class",
    "model_revision",
    "repository",
    "snapshot_files",
    "snapshot_path",
    "tokenizer_class",
    "tokenizer_revision",
}
EMBEDDING_CONFIGURATION_FIELDS = {
    "_name_or_path",
    "add_cross_attention",
    "architectures",
    "attention_probs_dropout_prob",
    "bos_token_id",
    "chunk_size_feed_forward",
    "classifier_dropout",
    "dtype",
    "emb_layer_norm_before",
    "eos_token_id",
    "esmfold_config",
    "hidden_act",
    "hidden_dropout_prob",
    "hidden_size",
    "id2label",
    "initializer_range",
    "intermediate_size",
    "is_decoder",
    "is_encoder_decoder",
    "is_folding_model",
    "label2id",
    "layer_norm_eps",
    "mask_token_id",
    "max_position_embeddings",
    "model_type",
    "num_attention_heads",
    "num_hidden_layers",
    "output_attentions",
    "output_hidden_states",
    "pad_token_id",
    "position_embedding_type",
    "problem_type",
    "return_dict",
    "rope_theta",
    "tie_word_embeddings",
    "token_dropout",
    "transformers_version",
    "use_cache",
    "vocab_list",
    "vocab_size",
}
EMBEDDING_GENERATION_FIELDS = {
    "batch_size_chunks",
    "long_protein_behavior",
    "maximum_residues_per_window",
    "output_dtype",
    "output_shape",
    "overlap_residues",
    "padding_tokens_excluded",
    "pooling",
    "recommended_for_primary_analysis",
    "special_tokens_excluded",
    "strategy",
    "stride_residues",
    "truncation_enabled_in_tokenizer",
}
EMBEDDING_ENVIRONMENT_FIELDS = {
    "cuda_available",
    "cuda_device_capability",
    "cuda_device_name",
    "cudnn_benchmark",
    "cudnn_deterministic",
    "cudnn_version",
    "deterministic_algorithms_enabled",
    "deterministic_debug_mode",
    "device",
    "installed_distributions",
    "key_distribution_versions",
    "machine",
    "numerical_environment",
    "nvidia_smi",
    "platform",
    "processor",
    "python",
    "python_executable",
    "torch_build_config",
    "torch_cuda_version",
    "torch_version",
}
RUN_COMPLETION_FIELDS = {
    "artifacts",
    "cohort_manifest_sha256",
    "eligible_for_scientific_aggregation",
    "required_next_step",
    "run_manifest_sha256",
    "run_status",
    "schema",
    "scientific_result",
    "summary_file",
    "summary_sha256",
}
RUN_SUMMARY_FIELDS = {
    "artifacts",
    "canonical_campaign",
    "completed_fold_numbers",
    "condition",
    "eligible_for_scientific_aggregation",
    "fold_files",
    "metrics",
    "n_completed_folds",
    "outer_test_policy",
    "pilot",
    "planned_fold_numbers",
    "primary_metric",
    "protocol",
    "protocol_complete",
    "required_finite_metrics",
    "run_status",
    "schema",
    "scientific_result",
    "scientific_result_reason",
    "seed",
    "selection_policy",
    "source_stability",
    "wall_seconds",
}
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
RESULT_FIELDS = {
    "best_config",
    "best_epoch",
    "condition",
    "fold",
    "fold_number",
    "fold_seed",
    "optimization",
    "outer_test_predictions",
    "pilot",
    "preprocessing",
    "protocol",
    "result_scope",
    "scientific_result",
    "seed",
    "selection",
    "test_evaluation_calls",
    "test_metrics",
    "tuning",
    "wall_seconds",
}
SUMMARY_METRIC_FIELDS = {
    "cellwise_pearson_mean",
    "cellwise_spearman_mean",
    "fraction_genes_positive_residual_r2",
    "gene_mean_baseline_mae",
    "gene_mean_baseline_mse",
    "gene_wise_mse_mean",
    "gene_wise_pearson_mean",
    "gene_wise_residual_r2_mean",
    "mae",
    "mse",
    "residual_cellwise_pearson_mean",
    "residual_pearson_pooled",
    "residual_r2_vs_train_gene_mean",
}
TEST_METRIC_FIELDS = {
    "cellwise_pearson_mean",
    "cellwise_pearson_median",
    "cellwise_pearson_sd",
    "cellwise_spearman_mean",
    "cellwise_spearman_sd",
    "fraction_genes_positive_residual_r2",
    "gene_mean_baseline_mae",
    "gene_mean_baseline_mse",
    "gene_mean_reference_sha256",
    "gene_wise_mae_mean",
    "gene_wise_mae_median",
    "gene_wise_mae_sd",
    "gene_wise_mse_mean",
    "gene_wise_mse_median",
    "gene_wise_mse_sd",
    "gene_wise_pearson_mean",
    "gene_wise_pearson_median",
    "gene_wise_pearson_sd",
    "gene_wise_residual_r2_mean",
    "gene_wise_residual_r2_median",
    "gene_wise_residual_r2_sd",
    "mae",
    "mse",
    "n_cells",
    "n_finite_cell_correlations",
    "n_finite_cellwise_pearson",
    "n_finite_cellwise_spearman",
    "n_finite_gene_correlations",
    "n_finite_gene_wise_mae",
    "n_finite_gene_wise_mse",
    "n_finite_gene_wise_pearson",
    "n_finite_gene_wise_residual_r2",
    "n_finite_pairs",
    "n_finite_residual_cellwise_pearson",
    "n_reference_genes",
    "raw_cellwise_correlation_role",
    "reference_scope",
    "residual_cellwise_pearson_mean",
    "residual_cellwise_pearson_median",
    "residual_cellwise_pearson_sd",
    "residual_pearson_pooled",
    "residual_r2_vs_train_gene_mean",
}
VALIDATION_METRIC_FIELDS = TEST_METRIC_FIELDS.difference(
    {
        "cellwise_spearman_mean",
        "cellwise_spearman_sd",
        "n_finite_cellwise_spearman",
    }
)
NONNEGATIVE_METRIC_FIELDS = {
    "mse",
    "mae",
    "gene_mean_baseline_mse",
    "gene_mean_baseline_mae",
    "gene_wise_mae_mean",
    "gene_wise_mae_median",
    "gene_wise_mae_sd",
    "gene_wise_mse_mean",
    "gene_wise_mse_median",
    "gene_wise_mse_sd",
    "cellwise_pearson_sd",
    "cellwise_spearman_sd",
    "gene_wise_pearson_sd",
    "gene_wise_residual_r2_sd",
    "residual_cellwise_pearson_sd",
}
CORRELATION_METRIC_FIELDS = {
    "cellwise_pearson_mean",
    "cellwise_pearson_median",
    "cellwise_spearman_mean",
    "gene_wise_pearson_mean",
    "gene_wise_pearson_median",
    "residual_cellwise_pearson_mean",
    "residual_cellwise_pearson_median",
    "residual_pearson_pooled",
}
CORRELATION_SD_FIELDS = {
    "cellwise_pearson_sd",
    "cellwise_spearman_sd",
    "gene_wise_pearson_sd",
    "residual_cellwise_pearson_sd",
}
R2_METRIC_FIELDS = {
    "gene_wise_residual_r2_mean",
    "gene_wise_residual_r2_median",
    "residual_r2_vs_train_gene_mean",
}
FRACTION_METRIC_FIELDS = {"fraction_genes_positive_residual_r2"}
TUNING_RESULT_FIELDS = {
    "best_epoch",
    "best_validation_metrics",
    "config",
    "history",
    "scheduler",
    "scheduler_horizon_epochs",
    "seed",
    "selection_metric",
    "selection_score",
    "trained_epochs",
}
CONDITION_BASE_CONFIGS: dict[str, dict[str, Any]] = {
    "C-DEEPDEP": {"model": "deepdep"},
    "C-ONEHOT": {"model": "onehot"},
    "C-PLM-MLP": {"model": "plmmlp"},
    "C-GCN": {"model": "gnn", "comut": False},
    "C-GCN-comut": {"model": "gnn", "comut": True},
    "DepPLM": {"model": "depplm"},
    "A1-ESM": {"model": "depplm", "use_pretrained": False},
    "A2-GNN": {"model": "depplm", "use_gnn": False},
    "A3-comut": {"model": "depplm", "use_comut": False},
    "A4-MT": {"model": "depplm", "use_multitask": False},
    "A5-XATTN": {"model": "depplm", "use_crossattn": False},
}
EXPECTED_FIT_SCOPE = {
    "expression_pca": "training_cells_only",
    "mutation_vocabulary": "training_cells_only",
    "mutation_token_order": "training_cells_only",
    "comutation_pmi": "training_cells_only",
    "selectivity_label": "training_cells_only",
    "lineage_categories": "training_cells_only_unseen_is_zero",
}
EXPECTED_COMMON_INPUT_PATHS = {
    "data/processed/gene_effect.npy",
    "data/processed/esm2_t6_8M_residue_chunked.npy",
    "data/processed/esm2_t6_8M_residue_chunked.l2.npy",
    "data/processed/esm2_t6_8M_residue_chunked.names.txt",
    "data/processed/esm2_t6_8M_residue_chunked.manifest.json",
    "data/processed/gene_seqs.fa",
    "data/processed/ppi_graph.npz",
    "data/processed/genes.txt",
    "data/processed/cell_lines.txt",
    "data/processed/lineages.txt",
    "data/depmap/OmicsExpressionProteinCodingGenesTPMLogp1.csv",
    "data/depmap/OmicsSomaticMutations.csv",
}
EXPECTED_CODE_PATHS = {
    "code/run_corrected_experiments.py",
    "code/corrected_protocol.py",
    "code/models.py",
    "code/03_embed_proteins.py",
    "code/embedding_provenance.py",
}
EXPECTED_ARGUMENTS: dict[str, int | float] = {
    "batch_size": 64,
    "folds": 5,
    "gene_subsample": 6000,
    "inner_max_epochs": 20,
    "inner_val_fraction": 0.2,
    "n_expr_pc": 128,
    "patience": 5,
    "pilot_cells": 96,
    "pilot_genes": 128,
    "top_mut": 5000,
}
REQUIRED_FINITE_METRIC_ORDER = (
    "mse",
    "mae",
    "gene_mean_baseline_mse",
    "gene_mean_baseline_mae",
    "residual_r2_vs_train_gene_mean",
    "residual_cellwise_pearson_mean",
)
REQUIRED_FINITE_METRICS = set(REQUIRED_FINITE_METRIC_ORDER)
RUN_REQUIRED_NEXT_STEP = (
    "Validate and aggregate the prespecified conditions and seeds before using "
    "this run for manuscript conclusions."
)
OUTER_TEST_POLICY = "exactly_once_per_fold_after_fixed_epoch_refit"
SELECTION_POLICY = (
    "maximize inner-validation residual R2 against inner-training gene means; "
    "report outer-test metrics against outer-training gene means"
)
SOURCE_STABILITY = {"checked_after_training": True, "reasons": [], "unchanged": True}
TRANSFER_HOLDOUT_COUNTS = {"Bone": 35, "Kidney": 31, "Myeloid": 39}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
REPARSE_POINT = 0x400


class VerificationError(RuntimeError):
    """Raised when any evidence gate cannot be established."""


class _OwnedDescriptor(_owner_ctypes.c_int):
    """A native-call result that uniquely owns one CRT/POSIX descriptor.

    Descriptor-producing functions use this type as their ``ctypes`` result
    type.  Thus Python bytecode never receives a successful bare integer fd:
    the object on the CALL result stack is already destructible.  Registration
    tokens add generation protection after acquisition; the scalar value is
    the primary owner before and during registration.
    """

    _descriptor_number: int
    _ownership_token: object
    _registration_complete: bool

    @property
    def closed(self) -> bool:
        return int(self.value) < 0

    def fileno(self) -> int:
        number = int(self.value)
        if number < 0:
            raise ValueError("I/O operation on closed descriptor owner")
        return number

    def _retire(self) -> int:
        """Permanently revoke this object's authority before a native close."""

        number = int(self.value)
        self.value = -1
        return number

    def __index__(self) -> int:
        try:
            return self.fileno()
        except ValueError:
            raise

    def __int__(self) -> int:
        try:
            return self.fileno()
        except ValueError:
            return int(getattr(self, "_descriptor_number", self.value))

    def __hash__(self) -> int:
        return hash(int(self))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, int):
            return int(self) == other
        if isinstance(other, _OwnedDescriptor):
            return int(self) == int(other)
        return NotImplemented

    def close(self) -> None:
        _close_registered_descriptor_only(self)

    def __del__(self) -> None:
        try:
            _close_registered_descriptor_only(self)
        except BaseException:
            # Finalizers cannot report cleanup errors. Explicit owners use
            # _close_owned_descriptor, which preserves the active exception.
            pass


class _DescriptorCloseResult(_owner_ctypes.c_int):
    """A native close result bound to the exact descriptor generation."""

    _source_attempt: _DescriptorCloseAttempt | None = None

    def source_attempt(self) -> _DescriptorCloseAttempt | None:
        return type(self)._source_attempt

    def _finalize_once(self) -> None:
        attempt = self.source_attempt()
        if attempt is not None:
            attempt.settle_result(self)

    def __del__(self) -> None:
        try:
            self._finalize_once()
        except BaseException:
            # Retry a one-shot interruption without ever recovering authority
            # from the descriptor number alone.
            try:
                self._finalize_once()
            except BaseException:
                pass


class _DescriptorCloseAttempt:
    """Own one descriptor generation from retirement through native close."""

    __slots__ = (
        "_close_token",
        "_descriptor",
        "_generation_token",
        "_native_closed",
        "_native_failed",
        "_number",
        "_settled",
    )

    def __init__(
        self, descriptor: _OwnedDescriptor, number: int, generation_token: object | None
    ) -> None:
        self._descriptor = descriptor
        self._number = number
        self._generation_token = generation_token
        self._close_token = object()
        self._native_closed = False
        self._native_failed = False
        self._settled = False
        try:
            self._claim()
        except BaseException as primary:
            try:
                self.close()
            except BaseException as cleanup_error:
                primary.add_note(
                    f"descriptor close-attempt recovery failed: {cleanup_error!r}"
                )
            raise

    def __int__(self) -> int:
        return self._number

    def _claim(self) -> bool:
        if self._settled:
            return False
        current_token = _OWNED_DESCRIPTOR_TOKENS.get(self._number)
        if (
            self._generation_token is not None
            and current_token is not self._generation_token
        ):
            # A different owner has already registered this number. This
            # attempt has no authority over that newer generation.
            self._descriptor._retire()
            self._descriptor._registration_complete = False
            self._settled = True
            return False
        if self._generation_token is None and current_token is not None:
            self._descriptor._retire()
            self._descriptor._registration_complete = False
            self._settled = True
            return False
        active = getattr(self._descriptor, "_active_close_token", None)
        if active is not None and active is not self._close_token:
            raise VerificationError("descriptor close identity changed")
        self._descriptor._active_close_token = self._close_token
        active_number = int(self._descriptor.value)
        if active_number == self._number:
            # The independently owned attempt is already live, so retiring the
            # scalar here cannot create an ownerless pre-call window.
            self._descriptor._retire()
        elif active_number >= 0:
            raise VerificationError("descriptor close number changed")
        return True

    def result_type(self) -> type[_DescriptorCloseResult]:
        return type(
            "_BoundDescriptorCloseResult",
            (_DescriptorCloseResult,),
            {"__module__": __name__, "_source_attempt": self},
        )

    def _require_source(self, result: _DescriptorCloseResult) -> None:
        if result.source_attempt() is not self:
            raise VerificationError(
                "native descriptor close result belongs to a different attempt"
            )

    def _finish_generation(self) -> None:
        descriptor = self._descriptor
        if int(descriptor.value) == self._number:
            descriptor._retire()
        descriptor._registration_complete = False
        if (
            self._generation_token is not None
            and _OWNED_DESCRIPTOR_TOKENS.get(self._number) is self._generation_token
        ):
            del _OWNED_DESCRIPTOR_TOKENS[self._number]
        reference = _OWNED_DESCRIPTOR_REFS.get(self._number)
        if reference is not None and reference() is descriptor:
            del _OWNED_DESCRIPTOR_REFS[self._number]
        if getattr(descriptor, "_active_close_token", None) is self._close_token:
            descriptor._active_close_token = None
        self._settled = True

    def settle_result(self, result: _DescriptorCloseResult) -> None:
        self._require_source(result)
        if self._settled:
            return
        if int(result.value) == 0:
            # Record the native outcome before any registry mutation. A retry
            # now completes metadata retirement without a second numeric close.
            self._native_closed = True
            self._finish_generation()
            return
        error = _owner_ctypes.get_errno() or errno.EIO
        self._native_failed = True
        self._finish_generation()
        raise OSError(error, os.strerror(error))

    def close(self) -> None:
        if self._settled:
            return
        if self._native_closed or self._native_failed:
            self._finish_generation()
            return
        if not self._claim():
            return
        result: _DescriptorCloseResult | None = None
        try:
            result = _native_close_descriptor(self, self._number)
            self.settle_result(result)
        except BaseException as primary:
            if not self._settled:
                try:
                    if result is not None:
                        self.settle_result(result)
                    else:
                        retry_result = _native_close_descriptor(self, self._number)
                        self.settle_result(retry_result)
                except BaseException as cleanup_error:
                    primary.add_note(
                        f"descriptor close-attempt retry failed: {cleanup_error!r}"
                    )
            raise

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            try:
                self.close()
            except BaseException:
                pass


class _OwnedDescriptorStream(io.RawIOBase):
    """Binary stream facade whose lifetime is backed by an fd owner."""

    def __init__(self, owner: _OwnedDescriptor) -> None:
        super().__init__()
        self._owner: _OwnedDescriptor | None = owner

    @property
    def mode(self) -> str:
        return "rb"

    @property
    def name(self) -> int:
        return self.fileno()

    def fileno(self) -> int:
        owner = self._owner
        if owner is None:
            raise ValueError("I/O operation on closed file")
        return owner.fileno()

    def readable(self) -> bool:
        return not self.closed

    def writable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return not self.closed

    def readinto(self, buffer: Any) -> int:
        view = memoryview(buffer).cast("B")
        payload = os.read(self.fileno(), len(view))
        view[: len(payload)] = payload
        return len(payload)

    def readall(self) -> bytes:
        pieces: list[bytes] = []
        while True:
            payload = os.read(self.fileno(), 1024 * 1024)
            if not payload:
                return b"".join(pieces)
            pieces.append(payload)

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        return os.lseek(self.fileno(), offset, whence)

    def tell(self) -> int:
        return os.lseek(self.fileno(), 0, os.SEEK_CUR)

    def close(self) -> None:
        if self.closed:
            return
        owner = self._owner
        self._owner = None
        try:
            if owner is not None:
                _close_owned_descriptor(owner)
        finally:
            super().close()


if os.name == "nt":

    class _WindowsHandleAcquisitionGuard(_owner_ctypes.c_void_p):
        """A CreateFileW result that owns its HANDLE as soon as CALL returns."""

        @property
        def handle(self) -> int | None:
            return self.value

        @handle.setter
        def handle(self, value: int | None) -> None:
            self.value = value
            if value is None:
                self._active_transfer_token = None

        def __int__(self) -> int:
            value = self.value
            if value is None:
                raise ValueError("Windows HANDLE owner has been released")
            return int(value)

        def begin_transfer(self, token: object) -> int:
            """Reserve this HANDLE for one identity-bound CRT transfer."""

            value = self.value
            if value is None or value == _owner_ctypes.c_void_p(-1).value:
                raise VerificationError("cannot transfer an absent Windows handle")
            if getattr(self, "_active_transfer_token", None) is not None:
                raise VerificationError(
                    "Windows handle transfer is already in progress"
                )
            if getattr(self, "_active_close_token", None) is not None:
                raise VerificationError("Windows handle close is already in progress")
            self._active_transfer_token = token
            return int(value)

        def complete_transfer(self, token: object) -> None:
            active = getattr(self, "_active_transfer_token", None)
            if active is not token:
                if self.value is None and active is None:
                    return
                raise VerificationError("Windows handle transfer identity changed")
            # Retire the source authority before publishing or closing the CRT
            # descriptor.  A repeated cleanup by this exact transfer is safe.
            self.value = None
            self._active_transfer_token = None

        def restore_failed_transfer(self, token: object) -> None:
            """Return an untransferred HANDLE to its original close guard."""

            active = getattr(self, "_active_transfer_token", None)
            if active is token:
                self._active_transfer_token = None
                return
            if active is None:
                return
            raise VerificationError("Windows handle transfer identity changed")

        def close(self, *, preserve: BaseException | None = None) -> None:
            if getattr(self, "_active_transfer_token", None) is not None:
                # The transfer object owns cleanup until a source-bound native
                # result proves whether the HANDLE was adopted by the CRT.
                return
            if getattr(self, "_active_close_token", None) is not None:
                return
            value = self.value
            if value is None or value == _owner_ctypes.c_void_p(-1).value:
                self.value = None
                return
            attempt: _WindowsHandleCloseAttempt | None = None
            try:
                attempt = _WindowsHandleCloseAttempt(self)
                _close_windows_handle(attempt)
            except BaseException as close_error:
                if attempt is not None:
                    try:
                        attempt.close()
                    except BaseException as cleanup_error:
                        close_error.add_note(
                            f"Windows HANDLE close retry failed: {cleanup_error!r}"
                        )
                if preserve is None:
                    raise
                preserve.add_note(f"Windows HANDLE close failed: {close_error!r}")

        def __del__(self) -> None:
            try:
                self.close()
            except BaseException:
                pass

else:

    class _WindowsHandleAcquisitionGuard:
        """Type placeholder for Windows-only ownership helpers."""


class _WindowsTransferredDescriptorResult(_OwnedDescriptor):
    """A CRT descriptor result bound to the HANDLE transfer that produced it."""

    _source_transfer: _WindowsHandleDescriptorTransfer | None = None

    def source_transfer(self) -> _WindowsHandleDescriptorTransfer | None:
        return type(self)._source_transfer

    def _finalize_once(self) -> None:
        transfer = self.source_transfer()
        if transfer is None:
            _OwnedDescriptor.__del__(self)
            return
        transfer.abandon_result(self)

    def __del__(self) -> None:
        try:
            self._finalize_once()
        except BaseException:
            # A one-shot asynchronous failure can interrupt cleanup before
            # either owner is retired.  Retry through the same state machine
            # so its exact source/result relationship remains authoritative.
            try:
                self._finalize_once()
            except BaseException:
                pass


class _WindowsHandleDescriptorTransfer:
    """Settle one Win32-HANDLE-to-CRT-descriptor ownership transition."""

    __slots__ = ("_guard", "_token", "_settled")

    def __init__(self, guard: _WindowsHandleAcquisitionGuard) -> None:
        self._guard = guard
        self._token = object()
        self._settled = False

    def begin(self) -> None:
        """Activate this transfer after its owner is protected by a caller."""

        if self._settled:
            raise VerificationError("Windows handle transfer is already settled")
        self._guard.begin_transfer(self._token)

    def result_type(self) -> type[_WindowsTransferredDescriptorResult]:
        # The per-transfer result type holds a strong source reference.  The
        # transfer does not retain the type or result, so this creates no cycle
        # and prevents cleanup order from separating a result from its source.
        return type(
            "_BoundWindowsTransferredDescriptorResult",
            (_WindowsTransferredDescriptorResult,),
            {"__module__": __name__, "_source_transfer": self},
        )

    def _require_source(self, result: _WindowsTransferredDescriptorResult) -> None:
        if result.source_transfer() is not self:
            raise VerificationError(
                "Windows descriptor result belongs to a different transfer"
            )

    def adopt_result(
        self, result: _WindowsTransferredDescriptorResult
    ) -> _WindowsTransferredDescriptorResult:
        self._require_source(result)
        if self._settled:
            raise VerificationError("Windows handle transfer is already settled")
        if result.value < 0:
            raise VerificationError("cannot adopt a failed Windows descriptor result")
        self._guard.complete_transfer(self._token)
        self._settled = True
        return result

    def abandon_result(
        self,
        result: _WindowsTransferredDescriptorResult,
        *,
        preserve: BaseException | None = None,
    ) -> None:
        self._require_source(result)
        if self._settled:
            _close_owned_descriptor(result, preserve=preserve)
            return
        if result.value >= 0:
            self._guard.complete_transfer(self._token)
            _close_owned_descriptor(result, preserve=preserve)
            self._settled = True
            return
        result.value = -1
        self._guard.restore_failed_transfer(self._token)
        self._guard.close(preserve=preserve)
        self._settled = True

    def abandon_without_result(self, *, preserve: BaseException | None = None) -> None:
        if self._settled:
            return
        self._guard.restore_failed_transfer(self._token)
        self._guard.close(preserve=preserve)
        self._settled = True

    def __del__(self) -> None:
        try:
            self.abandon_without_result()
        except BaseException:
            try:
                self.abandon_without_result()
            except BaseException:
                pass


class _WindowsHandleCloseResult(_owner_ctypes.c_int):
    """A CloseHandle result bound to its exact independently owned attempt."""

    _source_attempt: _WindowsHandleCloseAttempt | None = None

    def source_attempt(self) -> _WindowsHandleCloseAttempt | None:
        return type(self)._source_attempt

    def _finalize_once(self) -> None:
        attempt = self.source_attempt()
        if attempt is not None:
            attempt.settle_result(self)

    def __del__(self) -> None:
        try:
            self._finalize_once()
        except BaseException:
            try:
                self._finalize_once()
            except BaseException:
                pass


class _WindowsHandleCloseAttempt:
    """Own one HANDLE generation from source retirement through CloseHandle."""

    __slots__ = (
        "_close_token",
        "_guard",
        "_handle",
        "_native_closed",
        "_native_failed",
        "_settled",
    )

    def __init__(self, guard: _WindowsHandleAcquisitionGuard) -> None:
        self._guard = guard
        self._handle = int(guard)
        self._close_token = object()
        self._native_closed = False
        self._native_failed = False
        self._settled = False
        try:
            self._claim()
        except BaseException as primary:
            try:
                self.close()
            except BaseException as cleanup_error:
                primary.add_note(
                    f"Windows HANDLE close-attempt recovery failed: {cleanup_error!r}"
                )
            raise

    def __int__(self) -> int:
        return self._handle

    def _claim(self) -> bool:
        if self._settled:
            return False
        if getattr(self._guard, "_active_transfer_token", None) is not None:
            raise VerificationError("Windows handle transfer is already in progress")
        active = getattr(self._guard, "_active_close_token", None)
        if active is not None and active is not self._close_token:
            raise VerificationError("Windows handle close identity changed")
        value = self._guard.value
        if value not in {None, self._handle}:
            self._settled = True
            return False
        self._guard._active_close_token = self._close_token
        # The attempt already carries the exact HANDLE identity, so source
        # retirement cannot create an ownerless boundary before CloseHandle.
        self._guard.value = None
        return True

    def result_type(self) -> type[_WindowsHandleCloseResult]:
        return type(
            "_BoundWindowsHandleCloseResult",
            (_WindowsHandleCloseResult,),
            {"__module__": __name__, "_source_attempt": self},
        )

    def _require_source(self, result: _WindowsHandleCloseResult) -> None:
        if result.source_attempt() is not self:
            raise VerificationError(
                "CloseHandle result belongs to a different close attempt"
            )

    def _finish_success(self) -> None:
        self._guard.value = None
        if getattr(self._guard, "_active_close_token", None) is self._close_token:
            self._guard._active_close_token = None
        self._settled = True

    def _restore_failure(self) -> None:
        if getattr(self._guard, "_active_close_token", None) is self._close_token:
            self._guard.value = self._handle
            self._guard._active_close_token = None
        self._settled = True

    def settle_result(self, result: _WindowsHandleCloseResult) -> None:
        self._require_source(result)
        if self._settled:
            return
        if int(result.value):
            self._native_closed = True
            self._finish_success()
            return
        error = _owner_ctypes.get_last_error() or errno.EIO
        self._native_failed = True
        self._restore_failure()
        raise OSError(error, _owner_ctypes.FormatError(error))

    def close(self) -> None:
        if self._settled:
            return
        if self._native_closed:
            self._finish_success()
            return
        if self._native_failed:
            self._restore_failure()
            return
        if not self._claim():
            return
        result: _WindowsHandleCloseResult | None = None
        try:
            result = _native_close_windows_handle(self)
            self.settle_result(result)
        except BaseException as primary:
            if not self._settled:
                try:
                    if result is not None:
                        self.settle_result(result)
                    else:
                        retry_result = _native_close_windows_handle(self)
                        self.settle_result(retry_result)
                except BaseException as cleanup_error:
                    primary.add_note(
                        f"Windows HANDLE close-attempt retry failed: {cleanup_error!r}"
                    )
            raise

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:
            try:
                self.close()
            except BaseException:
                pass


@dataclass(frozen=True)
class FileIdentity:
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class BoundReportDirectory:
    """A report parent held by identity for the full publication transaction."""

    requested_path: Path
    operation_path: Path
    descriptor: int
    identity: FileIdentity
    supports_dir_fd: bool

    def __del__(self) -> None:
        try:
            _close_registered_descriptor_only(self.descriptor)
        except BaseException:
            pass


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    identity: FileIdentity
    sha256: str
    bytes: int


@dataclass(frozen=True)
class DistributionArtifactOwner:
    distribution: str
    distribution_version: str
    relative_path: str


@dataclass(frozen=True)
class DistributionArtifact:
    distribution: str
    distribution_version: str
    relative_path: str
    path: Path
    bytes: int
    record_sha256_urlsafe_base64: str
    owners: tuple[DistributionArtifactOwner, ...] = ()


def _distribution_artifact_owners(
    artifact: DistributionArtifact,
) -> tuple[DistributionArtifactOwner, ...]:
    if artifact.owners:
        return artifact.owners
    return (
        DistributionArtifactOwner(
            distribution=artifact.distribution,
            distribution_version=artifact.distribution_version,
            relative_path=artifact.relative_path,
        ),
    )


def _distribution_artifact_owner_for(
    artifact: DistributionArtifact, distribution: str
) -> DistributionArtifactOwner | None:
    normalized = _normalized_distribution_name(distribution)
    matches = [
        owner
        for owner in _distribution_artifact_owners(artifact)
        if _normalized_distribution_name(owner.distribution) == normalized
    ]
    if len(matches) > 1:
        raise VerificationError(
            f"ambiguous distribution RECORD ownership: {artifact.path}"
        )
    return matches[0] if matches else None


def _distribution_artifact_has_dependency_owner(
    artifact: DistributionArtifact,
    dependency_closure: Mapping[str, tuple[str, str]],
) -> bool:
    return any(
        _normalized_distribution_name(owner.distribution) in dependency_closure
        for owner in _distribution_artifact_owners(artifact)
    )


def _distribution_artifact_owner_records(
    artifact: DistributionArtifact,
) -> list[dict[str, str]]:
    return [
        {
            "distribution": owner.distribution,
            "distribution_version": owner.distribution_version,
            "relative_path": owner.relative_path,
        }
        for owner in _distribution_artifact_owners(artifact)
    ]


@dataclass
class TrustedImportRuntime:
    original_path: list[str]
    original_meta_path: list[Any]
    original_path_hooks: list[Any]
    original_importer_cache: dict[str, Any]


@dataclass
class BoundDistributionArtifact:
    """A verified artifact held open until its loader has finished."""

    descriptor: int
    execution_path: Path
    snapshot: FileSnapshot

    def __del__(self) -> None:
        try:
            _close_registered_descriptor_only(self.descriptor)
        except BaseException:
            pass


@dataclass(frozen=True)
class BoundInputSnapshot:
    """A private input whose parser reads only the held file description."""

    path: Path
    descriptor: int
    identity: FileIdentity
    sha256: str
    bytes: int
    path_is_bound: bool

    def __del__(self) -> None:
        try:
            _close_registered_descriptor_only(self.descriptor)
        except BaseException:
            pass

    def open_binary(self) -> BinaryIO:
        duplicate: _OwnedDescriptor | None = None
        try:
            duplicate = _owned_os_dup(self.descriptor)
            os.lseek(duplicate, 0, os.SEEK_SET)
            return _OwnedDescriptorStream(duplicate)
        except BaseException as primary:
            if duplicate is not None:
                _close_owned_descriptor(duplicate, preserve=primary)
            raise

    def read_bytes(self) -> builtins.bytes:
        with self.open_binary() as handle:
            return handle.read()


@dataclass
class VerificationContext:
    snapshots: dict[str, FileSnapshot]

    def remember(self, snapshot: FileSnapshot) -> None:
        key = os.path.normcase(str(snapshot.path))
        previous = self.snapshots.get(key)
        if previous is not None and (
            previous.identity != snapshot.identity
            or previous.sha256 != snapshot.sha256
            or previous.bytes != snapshot.bytes
        ):
            raise VerificationError(
                f"file changed between verification reads: {snapshot.path}"
            )
        self.snapshots[key] = snapshot


@dataclass(frozen=True)
class FoldEvidence:
    key: str
    protocol: str
    condition: str
    seed: int
    fold: str
    fold_number: int
    fold_seed: int
    outer_train: np.ndarray
    outer_test: np.ndarray
    source_cell_indices: np.ndarray
    source_gene_indices: np.ndarray
    prediction_record: dict[str, Any]


@dataclass(frozen=True)
class CanonicalPreprocessingInputs:
    expression: np.ndarray
    expression_names: tuple[str, ...]
    mutations_by_cell: tuple[frozenset[str], ...]
    edge_index: np.ndarray
    genes: tuple[str, ...]
    lineages: np.ndarray
    gene_effect: np.ndarray


def _current_analysis_environment() -> dict[str, Any]:
    evidence = _revalidate_trusted_scientific_dependencies()
    recorded_versions = {
        record["distribution"]: record["distribution_version"]
        for record in evidence["modules"]
    }
    packages: dict[str, str] = {}
    for name in ANALYSIS_ENVIRONMENT_PACKAGES:
        version = recorded_versions.get(name)
        if not isinstance(version, str) or not version:
            raise VerificationError(
                f"required verifier dependency is not trust-bound: {name}"
            )
        packages[name] = version
    return {
        "packages": packages,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }


def _resolve_analysis_contract(completion_schema: Any) -> dict[str, Any]:
    if not isinstance(completion_schema, str):
        raise VerificationError("analysis completion schema is invalid")
    contract = ANALYSIS_CONTRACTS.get(completion_schema)
    if contract is None:
        raise VerificationError("unexpected analysis completion schema")
    return dict(contract)


def _validate_analysis_environment(
    value: Any, *, allow_legacy_inventory: bool = True
) -> dict[str, Any]:
    """Validate producer environment evidence and compare recorded versions."""
    if not isinstance(value, Mapping):
        raise VerificationError("analysis_environment must be an object")
    _require_exact_keys(
        value, ANALYSIS_ENVIRONMENT_FIELDS, label="analysis_environment"
    )
    for field in ("platform", "python", "python_implementation"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise VerificationError(f"analysis_environment.{field} is invalid")
    packages = value.get("packages")
    if not isinstance(packages, Mapping):
        raise VerificationError("analysis_environment.packages must be an object")
    observed_package_names = set(packages)
    complete_package_names = set(ANALYSIS_ENVIRONMENT_PACKAGES)
    legacy_package_names = set(LEGACY_ANALYSIS_ENVIRONMENT_PACKAGES)
    if observed_package_names == complete_package_names:
        inventory_status = "complete"
    elif allow_legacy_inventory and observed_package_names == legacy_package_names:
        inventory_status = "legacy_incomplete"
    else:
        raise VerificationError(
            "analysis_environment.packages inventory mismatch: expected the full "
            "dependency inventory or the registered legacy numpy/scipy inventory"
        )
    for name, version in packages.items():
        if not isinstance(name, str) or not isinstance(version, str) or not version:
            raise VerificationError(f"analysis_environment.packages.{name} is invalid")

    current = _current_analysis_environment()
    for field in ("platform", "python", "python_implementation"):
        if value[field] != current[field]:
            raise VerificationError(
                f"analysis_environment.{field} differs from the verifier runtime"
            )
    current_packages = current["packages"]
    for name, recorded_version in packages.items():
        if recorded_version != current_packages[name]:
            raise VerificationError(
                "analysis_environment package version differs from the verifier "
                f"runtime: {name}"
            )

    missing = sorted(complete_package_names - observed_package_names)
    return {
        "inventory_status": inventory_status,
        "recorded_environment": {
            "packages": dict(packages),
            "platform": value["platform"],
            "python": value["python"],
            "python_implementation": value["python_implementation"],
        },
        "current_verifier_environment": current,
        "recorded_versions_match_current_verifier": True,
        "required_package_inventory_complete": not missing,
        "complete_environment_match": not missing,
        "missing_recorded_packages": missing,
    }


def _analysis_environment_verdict(
    evidence: Mapping[str, Any], *, analysis_contract_current: bool = True
) -> dict[str, Any]:
    """Return the release verdict implied by producer-environment evidence."""
    complete = (
        evidence.get("required_package_inventory_complete") is True
        and evidence.get("complete_environment_match") is True
    )
    return {
        "release_ready": complete and analysis_contract_current,
        "verification_passed": complete,
        "verification_status": (
            "passed"
            if complete and analysis_contract_current
            else (
                "passed_diagnostic_legacy_contract_not_release_eligible"
                if complete
                else "blocked_incomplete_environment_inventory"
            )
        ),
        "exact_environment_gate": complete,
        "analysis_contract_current": analysis_contract_current,
    }


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise VerificationError(
            "value cannot be represented as canonical JSON"
        ) from exc
    return rendered.encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise VerificationError(f"non-finite JSON value is forbidden: {value}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except VerificationError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot parse {label} as strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"{label} must be a JSON object")
    _reject_nonfinite_json(value, label=label)
    return value


def _reject_nonfinite_json(value: Any, *, label: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise VerificationError(f"{label} contains a non-finite number")
        return
    if isinstance(value, list):
        for item in value:
            _reject_nonfinite_json(item, label=label)
        return
    if isinstance(value, dict):
        for item in value.values():
            _reject_nonfinite_json(item, label=label)
        return
    raise VerificationError(f"{label} contains an unsupported JSON value")


def _identity(info: os.stat_result) -> FileIdentity:
    # Windows path-stat and handle-stat expose different st_ctime semantics.
    ctime_ns = int(info.st_mtime_ns) if os.name == "nt" else int(info.st_ctime_ns)
    return FileIdentity(
        device=int(info.st_dev),
        inode=int(info.st_ino),
        mode=int(info.st_mode),
        size=int(info.st_size),
        mtime_ns=int(info.st_mtime_ns),
        ctime_ns=ctime_ns,
    )


def _is_link_like(info: os.stat_result) -> bool:
    attributes = int(getattr(info, "st_file_attributes", 0))
    return stat.S_ISLNK(info.st_mode) or bool(attributes & REPARSE_POINT)


def _check_path_components(path: Path, *, require_leaf: bool = True) -> None:
    absolute = Path(os.path.abspath(path))
    anchor = Path(absolute.anchor)
    current = anchor
    parts = absolute.parts[1:] if absolute.anchor else absolute.parts
    for index, part in enumerate(parts):
        current = current / part
        if index == len(parts) - 1 and not require_leaf and not current.exists():
            break
        try:
            info = current.lstat()
        except OSError as exc:
            raise VerificationError(
                f"path component is unavailable: {current}"
            ) from exc
        if _is_link_like(info):
            raise VerificationError(
                f"symlink or reparse-point path is forbidden: {current}"
            )
        if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise VerificationError(f"non-directory path component: {current}")


def _normalize_relative(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise VerificationError(f"{label} must be a non-empty relative path")
    rendered = value.replace("\\", "/")
    pure = PurePosixPath(rendered)
    if (
        rendered.startswith("/")
        or ntpath.isabs(value)
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or any(":" in part for part in pure.parts)
    ):
        raise VerificationError(f"unsafe relative path in {label}: {value!r}")
    normalized = pure.as_posix()
    if normalized != rendered:
        raise VerificationError(f"noncanonical relative path in {label}: {value!r}")
    return normalized


def _safe_child(root: Path, relative: Any, *, label: str) -> Path:
    normalized = _normalize_relative(relative, label=label)
    base = Path(os.path.abspath(root))
    _check_path_components(base)
    candidate = base.joinpath(*PurePosixPath(normalized).parts)
    _check_path_components(candidate)
    try:
        if os.path.commonpath((str(base), str(candidate))) != str(base):
            raise VerificationError(f"{label} escapes its declared root")
    except ValueError as exc:
        raise VerificationError(f"{label} escapes its declared root") from exc
    return candidate


def _workspace_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise VerificationError(f"{label} path is missing")
    candidate = Path(value)
    if candidate.is_absolute() or ntpath.isabs(value):
        _check_path_components(candidate)
        absolute = Path(os.path.abspath(candidate))
        root = Path(os.path.abspath(ROOT))
        try:
            relative = absolute.relative_to(root)
        except ValueError as exc:
            raise VerificationError(f"{label} is outside the workspace") from exc
        normalized = _normalize_relative(relative.as_posix(), label=label)
    else:
        normalized = _normalize_relative(value, label=label)
    return _safe_child(ROOT, normalized, label=label)


def _register_acquired_descriptor(owned: _OwnedDescriptor) -> _OwnedDescriptor:
    """Add reuse metadata to a descriptor that is already natively owned."""

    descriptor = owned.fileno()
    registry = _OWNED_DESCRIPTOR_TOKENS
    references = _OWNED_DESCRIPTOR_REFS
    token = object()
    try:
        # If the kernel returned a number already present in the registry, that
        # previous descriptor has necessarily been closed outside this owner.  Its
        # token must be retired before the number can be assigned a new generation.
        # The stale scalar owner's finalizer sees the token mismatch and retires
        # itself without closing a potentially reused descriptor number.
        if descriptor in registry:
            del registry[descriptor]
            references.pop(descriptor, None)
        owned._descriptor_number = descriptor
        owned._ownership_token = token
        owned._registration_complete = False
        registry[descriptor] = token
        references[descriptor] = weakref.ref(owned)
        owned._registration_complete = True
        return owned
    except BaseException as primary:
        if registry.get(descriptor) is token:
            registry.pop(descriptor, None)
        reference = references.get(descriptor)
        if reference is not None and reference() is owned:
            references.pop(descriptor, None)
        _close_registered_descriptor_only(owned, preserve=primary)
        raise


def _native_owned_descriptor_from_call(
    function: Any, arguments: tuple[Any, ...]
) -> _OwnedDescriptor:
    """Return a native fd producer's result directly as an owning scalar."""

    if not hasattr(function, "restype"):
        raise VerificationError("descriptor producer is not a native ctypes call")
    function.restype = _OwnedDescriptor
    owned = function(*arguments)
    if not isinstance(owned, _OwnedDescriptor):
        raise VerificationError("native descriptor producer returned no owner")
    return owned


def _native_runtime() -> Any:
    if os.name == "nt":
        return _owner_ctypes.CDLL("ucrtbase.dll", use_errno=True)
    return _owner_ctypes.CDLL(None, use_errno=True)


def _native_close_descriptor(
    attempt: _DescriptorCloseAttempt, descriptor: int
) -> _DescriptorCloseResult:
    """Return a source-bound native close result without a bare-result window."""

    runtime = _native_runtime()
    function = runtime._close if os.name == "nt" else runtime.close
    function.argtypes = (_owner_ctypes.c_int,)
    function.restype = attempt.result_type()
    _owner_ctypes.set_errno(0)
    return function(int(descriptor))


def _raise_native_descriptor_error(path: Any | None = None) -> None:
    error = _owner_ctypes.get_errno() or errno.EIO
    if path is None:
        raise OSError(error, os.strerror(error))
    raise OSError(error, os.strerror(error), os.fspath(path))


def _native_open_descriptor(
    path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    flags: int,
    mode: int,
    dir_fd: int | None,
) -> _OwnedDescriptor:
    """Acquire an fd with ownership established by the native CALL result."""

    runtime = _native_runtime()
    native_path = os.fspath(path)
    if os.name == "nt":
        if dir_fd is not None:
            raise OSError(errno.ENOTSUP, "dir_fd is unavailable on Windows")
        flags |= getattr(os, "O_NOINHERIT", 0)
        if isinstance(native_path, bytes):
            if b"\0" in native_path:
                raise ValueError("embedded null byte")
            function = runtime._open
            function.argtypes = (
                _owner_ctypes.c_char_p,
                _owner_ctypes.c_int,
                _owner_ctypes.c_int,
            )
            arguments: tuple[Any, ...] = (native_path, flags, mode)
        else:
            if "\0" in native_path:
                raise ValueError("embedded null character")
            function = runtime._wopen
            function.argtypes = (
                _owner_ctypes.c_wchar_p,
                _owner_ctypes.c_int,
                _owner_ctypes.c_int,
            )
            arguments = (native_path, flags, mode)
    else:
        encoded = os.fsencode(native_path)
        if b"\0" in encoded:
            raise ValueError("embedded null byte")
        flags |= getattr(os, "O_CLOEXEC", 0)
        if dir_fd is None:
            function = runtime.open
            function.argtypes = (
                _owner_ctypes.c_char_p,
                _owner_ctypes.c_int,
                _owner_ctypes.c_uint,
            )
            arguments = (encoded, flags, mode)
        else:
            try:
                function = runtime.openat
            except AttributeError as exc:
                raise OSError(errno.ENOTSUP, "native openat is unavailable") from exc
            function.argtypes = (
                _owner_ctypes.c_int,
                _owner_ctypes.c_char_p,
                _owner_ctypes.c_int,
                _owner_ctypes.c_uint,
            )
            arguments = (int(dir_fd), encoded, flags, mode)
    _owner_ctypes.set_errno(0)
    owned = _native_owned_descriptor_from_call(function, arguments)
    if owned.value < 0:
        owned.value = -1
        _raise_native_descriptor_error(path)
    return owned


def _native_duplicate_descriptor(descriptor: int) -> _OwnedDescriptor:
    runtime = _native_runtime()
    function = runtime._dup if os.name == "nt" else runtime.dup
    function.argtypes = (_owner_ctypes.c_int,)
    _owner_ctypes.set_errno(0)
    owned = _native_owned_descriptor_from_call(function, (int(descriptor),))
    if owned.value < 0:
        owned.value = -1
        _raise_native_descriptor_error()
    try:
        os.set_inheritable(owned, False)
    except BaseException as primary:
        _close_registered_descriptor_only(owned, preserve=primary)
        raise
    return owned


def _native_adopt_windows_handle(
    transfer: _WindowsHandleDescriptorTransfer, raw_handle: int, flags: int
) -> _WindowsTransferredDescriptorResult:
    if os.name != "nt":
        raise VerificationError("Windows HANDLE adoption is unavailable")
    runtime = _native_runtime()
    function = runtime._open_osfhandle
    function.argtypes = (_owner_ctypes.c_void_p, _owner_ctypes.c_int)
    result_type = transfer.result_type()
    function.restype = result_type
    _owner_ctypes.set_errno(0)
    # Return the native result directly: if RETURN_VALUE is interrupted, the
    # source-bound temporary settles the transfer while the frame unwinds.
    return function(raw_handle, flags)


def _owned_os_open(
    path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    flags: int,
    mode: int = 0o777,
    *,
    dir_fd: int | None = None,
) -> _OwnedDescriptor:
    """Open an fd into a native owner, then add reuse metadata."""

    owned: _OwnedDescriptor | None = None
    try:
        owned = _native_open_descriptor(path, flags, mode, dir_fd)
        return _register_acquired_descriptor(owned)
    except BaseException as primary:
        if owned is not None:
            _close_owned_descriptor(owned, preserve=primary)
        raise


def _owned_os_dup(descriptor: int) -> _OwnedDescriptor:
    """Duplicate an fd into a native owner, then add reuse metadata."""

    owned: _OwnedDescriptor | None = None
    try:
        owned = _native_duplicate_descriptor(descriptor)
        return _register_acquired_descriptor(owned)
    except BaseException as primary:
        if owned is not None:
            _close_owned_descriptor(owned, preserve=primary)
        raise


def _owned_windows_handle(
    create_file: Any, *arguments: Any
) -> _WindowsHandleAcquisitionGuard:
    """Make ctypes return a destructible HANDLE owner directly from CreateFileW."""

    if os.name != "nt" or not hasattr(create_file, "restype"):
        raise VerificationError("native Windows HANDLE ownership is unavailable")
    create_file.restype = _WindowsHandleAcquisitionGuard
    handle = create_file(*arguments)
    if not isinstance(handle, _WindowsHandleAcquisitionGuard):
        raise VerificationError("CreateFileW did not return a native HANDLE owner")
    return handle


def _adopt_windows_handle_as_descriptor(
    handle_guard: _WindowsHandleAcquisitionGuard, flags: int
) -> _OwnedDescriptor:
    """Atomically transfer a guarded Win32 HANDLE to a registered CRT fd."""
    if handle_guard.handle is None:
        raise VerificationError("cannot transfer an absent Windows handle")
    raw_handle = int(handle_guard)
    transfer: _WindowsHandleDescriptorTransfer | None = None
    native_result: _WindowsTransferredDescriptorResult | None = None
    try:
        transfer = _WindowsHandleDescriptorTransfer(handle_guard)
        transfer.begin()
        native_result = _native_adopt_windows_handle(transfer, raw_handle, flags)
        if native_result.value < 0:
            error = OSError(
                _owner_ctypes.get_errno() or errno.EBADF,
                os.strerror(_owner_ctypes.get_errno() or errno.EBADF),
            )
            transfer.abandon_result(native_result, preserve=error)
            raise error
        owned = transfer.adopt_result(native_result)
        return _register_acquired_descriptor(owned)
    except BaseException as primary:
        if transfer is None:
            handle_guard.close(preserve=primary)
        elif native_result is None:
            transfer.abandon_without_result(preserve=primary)
        else:
            transfer.abandon_result(native_result, preserve=primary)
        raise


def _open_stable(path: Path, *, label: str) -> tuple[_OwnedDescriptor, FileIdentity]:
    _check_path_components(path)
    try:
        before = path.lstat()
    except OSError as exc:
        raise VerificationError(f"cannot stat {label}: {path}") from exc
    if _is_link_like(before) or not stat.S_ISREG(before.st_mode):
        raise VerificationError(f"{label} is not a regular non-link file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: _OwnedDescriptor | None = None
    try:
        descriptor = _owned_os_open(path, flags)
        opened = os.fstat(descriptor)
    except BaseException as exc:
        if descriptor is not None:
            _close_owned_descriptor(descriptor, preserve=exc)
        if isinstance(exc, OSError):
            raise VerificationError(f"cannot open {label}: {path}") from exc
        raise
    before_identity = _identity(before)
    opened_identity = _identity(opened)
    if before_identity != opened_identity:
        error = VerificationError(f"{label} changed while being opened: {path}")
        _close_owned_descriptor(descriptor, preserve=error)
        raise error
    return descriptor, opened_identity


def _create_exclusive_bound_file(path: Path) -> _OwnedDescriptor:
    """Create a private file while denying replacement for its open lifetime."""
    if os.name != "nt":
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        return _owned_os_open(path, flags, 0o600)

    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    handle_guard: _WindowsHandleAcquisitionGuard | None = None
    try:
        handle_guard = _owned_windows_handle(
            create_file,
            str(path),
            0x80000000 | 0x40000000,  # GENERIC_READ | GENERIC_WRITE
            0x00000001,  # FILE_SHARE_READ; deny write/delete/rename sharing
            None,
            1,  # CREATE_NEW
            0x00000080 | 0x00200000,  # NORMAL | OPEN_REPARSE_POINT
            None,
        )
        invalid = ctypes.c_void_p(-1).value
        handle = handle_guard.handle
        if handle == invalid:
            handle_guard.handle = None
            error = ctypes.get_last_error()
            if error in {80, 183}:
                raise FileExistsError(error, ctypes.FormatError(error), str(path))
            raise OSError(error, ctypes.FormatError(error), str(path))
        return _adopt_windows_handle_as_descriptor(
            handle_guard, os.O_RDWR | getattr(os, "O_BINARY", 0)
        )
    except BaseException as primary:
        if handle_guard is not None:
            handle_guard.close(preserve=primary)
        raise


def _hash_open_descriptor(descriptor: int) -> tuple[str, int, FileIdentity]:
    duplicate: _OwnedDescriptor | None = None
    try:
        duplicate = _owned_os_dup(descriptor)
        digest = hashlib.sha256()
        total = 0
        os.lseek(duplicate, 0, os.SEEK_SET)
        while True:
            chunk = os.read(duplicate, 4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        identity = _identity(os.fstat(duplicate))
    except BaseException as primary:
        if duplicate is not None:
            _close_owned_descriptor(duplicate, preserve=primary)
        raise
    else:
        _close_owned_descriptor(duplicate)
    return digest.hexdigest(), total, identity


def _close_registered_descriptor_only(
    descriptor: int, *, preserve: BaseException | None = None
) -> bool:
    """Close *descriptor* only when its exact ownership token remains current."""
    if not isinstance(descriptor, _OwnedDescriptor):
        return False
    active_number = int(descriptor.value)
    if active_number < 0:
        return False
    try:
        number = descriptor._descriptor_number
    except AttributeError:
        number = active_number
    token = getattr(descriptor, "_ownership_token", None)
    complete = getattr(descriptor, "_registration_complete", False)
    current_token = _OWNED_DESCRIPTOR_TOKENS.get(number)
    if token is not None and (
        (complete and current_token is not token)
        or (current_token is not None and current_token is not token)
    ):
        # Another owner now has this number. Permanently retire this stale
        # scalar without issuing any numeric close against the replacement.
        descriptor._retire()
        descriptor._registration_complete = False
        return False
    primary = preserve if preserve is not None else sys.exc_info()[1]
    attempt: _DescriptorCloseAttempt | None = None
    try:
        attempt = _DescriptorCloseAttempt(descriptor, number, token)
        attempt.close()
    except BaseException as close_error:
        if attempt is not None:
            try:
                attempt.close()
            except BaseException as cleanup_error:
                close_error.add_note(
                    f"owned-descriptor close retry failed: {cleanup_error!r}"
                )
        if primary is None:
            raise close_error
        primary.add_note(f"owned-descriptor close failed: {close_error!r}")
    return True


def _close_owned_descriptor(
    descriptor: int, *, preserve: BaseException | None = None
) -> None:
    """Close an owned descriptor without replacing an exception in flight."""
    if isinstance(descriptor, _OwnedDescriptor):
        _close_registered_descriptor_only(descriptor, preserve=preserve)
        return
    primary = preserve if preserve is not None else sys.exc_info()[1]
    try:
        os.close(descriptor)
    except BaseException as close_error:
        # os.close may close the CRT descriptor and then raise. Its former
        # numeric HANDLE can already have been reused, so never pass that stale
        # value to CloseHandle as a fallback.
        if primary is None:
            raise
        primary.add_note(f"owned-descriptor close failed: {close_error!r}")


def _native_close_windows_handle(
    attempt: _WindowsHandleCloseAttempt,
) -> _WindowsHandleCloseResult:
    if os.name != "nt":
        raise VerificationError("CloseHandle is unavailable")
    from ctypes import wintypes

    close_handle = _owner_ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = attempt.result_type()
    _owner_ctypes.set_last_error(0)
    return close_handle(wintypes.HANDLE(int(attempt)))


def _close_windows_handle(
    handle: Any, *, preserve: BaseException | None = None
) -> None:
    """Close an owned Windows handle without replacing an active exception."""
    primary = preserve if preserve is not None else sys.exc_info()[1]
    try:
        if isinstance(handle, _WindowsHandleCloseAttempt):
            handle.close()
            return
        import ctypes
        from ctypes import wintypes

        close_handle = ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle
        close_handle.argtypes = (wintypes.HANDLE,)
        close_handle.restype = wintypes.BOOL
        if not close_handle(wintypes.HANDLE(int(handle))):
            error = ctypes.get_last_error()
            raise OSError(error, ctypes.FormatError(error))
    except BaseException:
        if primary is None:
            raise


def _verify_bound_input_unchanged(snapshot: BoundInputSnapshot, *, label: str) -> None:
    digest, total, identity = _hash_open_descriptor(snapshot.descriptor)
    if (
        identity != snapshot.identity
        or total != snapshot.bytes
        or digest != snapshot.sha256
    ):
        raise VerificationError(f"{label} descriptor content changed")
    if snapshot.path_is_bound:
        try:
            path_identity = _identity(snapshot.path.lstat())
        except OSError as exc:
            raise VerificationError(f"{label} private path disappeared") from exc
        if path_identity != snapshot.identity:
            raise VerificationError(f"{label} private path identity changed")


@contextmanager
def _open_bound_input(
    snapshot: BoundInputSnapshot, *, label: str
) -> Iterator[BinaryIO]:
    _verify_bound_input_unchanged(snapshot, label=label)
    try:
        with snapshot.open_binary() as handle:
            yield handle
    except BaseException as primary:
        try:
            _verify_bound_input_unchanged(snapshot, label=label)
        except BaseException as final_check_error:
            primary.add_note(
                f"bound-input final integrity check failed: {final_check_error!r}"
            )
        raise
    else:
        _verify_bound_input_unchanged(snapshot, label=label)


def _capture_file(
    path: Path,
    *,
    label: str,
    context: VerificationContext,
    keep_bytes: bool,
) -> tuple[bytes | None, FileSnapshot]:
    descriptor: _OwnedDescriptor | None = None
    try:
        descriptor, opened_identity = _open_stable(path, label=label)
        digest = hashlib.sha256()
        pieces: list[bytes] | None = [] if keep_bytes else None
        total = 0
        while True:
            chunk = os.read(descriptor, 4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
            if pieces is not None:
                pieces.append(chunk)
        closed_identity = _identity(os.fstat(descriptor))
    except BaseException as primary:
        if descriptor is not None:
            _close_owned_descriptor(descriptor, preserve=primary)
        raise
    else:
        _close_owned_descriptor(descriptor)
    try:
        after_identity = _identity(path.lstat())
    except OSError as exc:
        raise VerificationError(
            f"{label} disappeared during verification: {path}"
        ) from exc
    if (
        opened_identity != closed_identity
        or opened_identity != after_identity
        or total != opened_identity.size
    ):
        raise VerificationError(f"{label} changed during verification: {path}")
    snapshot = FileSnapshot(
        path=Path(os.path.abspath(path)),
        identity=opened_identity,
        sha256=digest.hexdigest(),
        bytes=total,
    )
    context.remember(snapshot)
    return (b"".join(pieces) if pieces is not None else None), snapshot


def _validate_record(record: Any, *, label: str) -> tuple[str, int, str]:
    if not isinstance(record, Mapping):
        raise VerificationError(f"{label} must be a file record")
    relative = _normalize_relative(record.get("path"), label=f"{label}.path")
    size = record.get("bytes")
    digest = record.get("sha256")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise VerificationError(f"{label}.bytes is invalid")
    if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
        raise VerificationError(f"{label}.sha256 is invalid")
    return relative, size, digest


def _verify_record(
    path: Path,
    record: Any,
    *,
    label: str,
    context: VerificationContext,
    keep_bytes: bool = False,
) -> tuple[bytes | None, FileSnapshot]:
    _relative, expected_size, expected_digest = _validate_record(record, label=label)
    payload, snapshot = _capture_file(
        path, label=label, context=context, keep_bytes=keep_bytes
    )
    if snapshot.bytes != expected_size or snapshot.sha256 != expected_digest:
        raise VerificationError(
            f"{label} does not match its declared byte/hash binding"
        )
    return payload, snapshot


def _create_private_snapshot_file(
    directory: BoundReportDirectory, name: str
) -> _OwnedDescriptor:
    parent = _report_directory_parent(directory, require_requested_identity=True)
    if not directory.supports_dir_fd:
        return _create_exclusive_bound_file(parent / name)
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    return _owned_os_open(name, flags, 0o600, dir_fd=directory.descriptor)


def _create_posix_anonymous_file(
    directory: BoundReportDirectory, *, label: str
) -> _OwnedDescriptor:
    """Create an unnamed regular file in a held POSIX directory."""
    if os.name == "nt" or not directory.supports_dir_fd:
        raise VerificationError(f"{label} requires POSIX descriptor-relative I/O")
    anonymous_flag = getattr(os, "O_TMPFILE", None)
    if not isinstance(anonymous_flag, int):
        raise VerificationError(f"{label} requires O_TMPFILE support")
    flags = os.O_RDWR | anonymous_flag | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = _owned_os_open(".", flags, 0o600, dir_fd=directory.descriptor)
    except OSError as exc:
        raise VerificationError(
            f"{label} cannot create an anonymous O_TMPFILE on this filesystem"
        ) from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 0:
            raise VerificationError(
                f"{label} did not create an unlinked anonymous regular file"
            )
    except BaseException as primary:
        _close_owned_descriptor(descriptor, preserve=primary)
        raise
    return descriptor


def _private_snapshot_entry_identity(
    directory: BoundReportDirectory, name: str
) -> FileIdentity | None:
    parent = _report_directory_parent(directory, require_requested_identity=False)
    directory_descriptor = directory.descriptor if directory.supports_dir_fd else None
    return _report_entry_stat(parent, directory_descriptor, name)


def _remove_identity_bound_private_snapshot(
    directory: BoundReportDirectory, name: str, identity: FileIdentity
) -> None:
    """Remove only the private file with *identity*, preserving any replacement."""
    _rollback_identity_bound_entry(directory, name, identity)
    remaining = _private_snapshot_entry_identity(directory, name)
    if remaining is None:
        return
    if _same_file_object(remaining, identity):
        raise VerificationError(f"owned private snapshot remains present: {name}")
    raise VerificationError(
        f"private snapshot path was replaced; replacement was preserved: {name}"
    )


def _remove_identity_bound_private_directory(
    parent_directory: BoundReportDirectory,
    name: str,
    identity: FileIdentity,
) -> None:
    """Atomically isolate and remove only the verifier-owned empty directory."""
    parent = _report_directory_parent(
        parent_directory, require_requested_identity=False
    )
    directory_descriptor = (
        parent_directory.descriptor if parent_directory.supports_dir_fd else None
    )
    resumed = _resume_owned_quarantines(
        parent,
        directory_descriptor,
        name,
        identity,
        suffix="snapshot-rollback",
        is_directory=True,
    )
    current = _report_entry_stat(parent, directory_descriptor, name)
    if current is None:
        if resumed:
            return
        raise VerificationError(
            f"private snapshot directory disappeared before cleanup: {parent / name}"
        )
    if not _same_directory_object(current, identity):
        raise VerificationError(
            f"private snapshot directory was replaced; replacement was preserved: "
            f"{parent / name}"
        )
    try:
        isolated_name, quarantine_token = _move_to_unique_quarantine(
            parent,
            directory_descriptor,
            name,
            identity,
            suffix="snapshot-rollback",
            label=f"isolate private snapshot directory {parent / name}",
            is_directory=True,
        )
    except FileNotFoundError as exc:
        raise VerificationError(
            f"private snapshot directory changed during cleanup: {parent / name}"
        ) from exc
    isolated_identity = _report_entry_stat(parent, directory_descriptor, isolated_name)
    if isolated_identity is None or not _same_directory_object(
        isolated_identity, identity
    ):
        raise VerificationError(
            "private snapshot directory identity changed during isolation; "
            f"the isolated entry was preserved: {parent / isolated_name}"
        )
    try:
        _retry_report_cleanup(
            lambda: _remove_quarantined_report_entry(
                parent,
                directory_descriptor,
                isolated_name,
                identity,
                is_directory=True,
            ),
            label=f"remove isolated private snapshot directory {parent / name}",
        )
    except BaseException as exc:
        if not isinstance(exc, OSError):
            raise
        raise VerificationError(
            "private snapshot directory was not empty or removable; "
            f"the isolated entry was preserved: {parent / isolated_name}"
        ) from exc
    else:
        _OWNED_QUARANTINE_ENTRIES.discard(quarantine_token)
    if _report_entry_stat(parent, directory_descriptor, isolated_name) is not None:
        raise VerificationError("isolated private snapshot directory remains present")
    if _report_entry_stat(parent, directory_descriptor, name) is not None:
        raise VerificationError(
            f"private snapshot directory path was replaced during cleanup: "
            f"{parent / name}"
        )


def _copy_verified_input_to_private_snapshot(
    source: Path,
    record: Any,
    destination: Path,
    *,
    label: str,
    context: VerificationContext,
    directory: BoundReportDirectory,
) -> BoundInputSnapshot:
    """Hash a source stream while copying exactly those bytes for later parsing."""
    _relative, expected_size, expected_digest = _validate_record(record, label=label)
    source_descriptor: _OwnedDescriptor | None = None
    destination_descriptor: _OwnedDescriptor | None = None
    destination_created = False
    destination_identity: FileIdentity | None = None
    digest = hashlib.sha256()
    total = 0
    try:
        try:
            source_descriptor, source_identity = _open_stable(source, label=label)
            if os.name == "nt":
                destination_descriptor = _create_private_snapshot_file(
                    directory, destination.name
                )
                destination_created = True
            else:
                destination_descriptor = _create_posix_anonymous_file(
                    directory, label=f"private snapshot for {label}"
                )
            destination_identity = _identity(os.fstat(destination_descriptor))
            while True:
                chunk = os.read(source_descriptor, 4 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                total += len(chunk)
                offset = 0
                while offset < len(chunk):
                    written = os.write(destination_descriptor, chunk[offset:])
                    if written <= 0:
                        raise OSError(
                            "short write while creating private input snapshot"
                        )
                    offset += written
            os.fsync(destination_descriptor)
            source_closed_identity = _identity(os.fstat(source_descriptor))
            destination_closed_identity = _identity(os.fstat(destination_descriptor))
        except BaseException as primary:
            if source_descriptor is not None:
                _close_owned_descriptor(source_descriptor, preserve=primary)
            raise
        else:
            assert source_descriptor is not None
            _close_owned_descriptor(source_descriptor)
            source_descriptor = None

        try:
            source_after_identity = _identity(source.lstat())
            if destination_created:
                destination_after_identity = _private_snapshot_entry_identity(
                    directory, destination.name
                )
            else:
                destination_after_identity = _identity(os.fstat(destination_descriptor))
        except OSError as exc:
            raise VerificationError(
                f"{label} or its private snapshot disappeared during copying"
            ) from exc
        if destination_after_identity is None:
            raise VerificationError(
                f"{label} private snapshot disappeared during copying"
            )
        if (
            source_identity != source_closed_identity
            or source_identity != source_after_identity
            or destination_closed_identity != destination_after_identity
            or total != source_identity.size
            or total != destination_after_identity.size
        ):
            raise VerificationError(f"{label} changed while its snapshot was copied")
        observed_digest = digest.hexdigest()
        if total != expected_size or observed_digest != expected_digest:
            raise VerificationError(
                f"{label} does not match its declared byte/hash binding"
            )
        source_snapshot = FileSnapshot(
            path=Path(os.path.abspath(source)),
            identity=source_identity,
            sha256=observed_digest,
            bytes=total,
        )
        context.remember(source_snapshot)
        path_is_bound = destination_created
        snapshot = BoundInputSnapshot(
            path=Path(os.path.abspath(destination)),
            descriptor=destination_descriptor,
            identity=destination_after_identity,
            sha256=observed_digest,
            bytes=total,
            path_is_bound=path_is_bound,
        )
        _verify_bound_input_unchanged(snapshot, label=f"private snapshot for {label}")
        destination_descriptor = None
        return snapshot
    except BaseException as primary:
        cleanup_errors: list[BaseException] = []
        if source_descriptor is not None:
            try:
                _close_owned_descriptor(source_descriptor, preserve=primary)
            except BaseException as exc:
                cleanup_errors.append(exc)
        if destination_descriptor is not None:
            try:
                _close_owned_descriptor(destination_descriptor, preserve=primary)
            except BaseException as exc:
                cleanup_errors.append(exc)
        if destination_created:
            if destination_identity is None:
                cleanup_errors.append(
                    VerificationError(
                        f"cannot safely identify failed private snapshot {destination}"
                    )
                )
            else:
                try:
                    _remove_identity_bound_private_snapshot(
                        directory, destination.name, destination_identity
                    )
                except BaseException as exc:
                    cleanup_errors.append(exc)
        for cleanup_error in cleanup_errors:
            primary.add_note(f"private snapshot cleanup failed: {cleanup_error!r}")
        raise


@contextmanager
def _verified_raw_input_copies(
    declared_workspace_records: Mapping[str, Mapping[str, Any]],
    *,
    context: VerificationContext,
) -> Iterator[dict[str, BoundInputSnapshot]]:
    """Yield verifier-private copies whose parsed bytes are the hashed bytes."""
    owns_temp_root = os.name == "nt"
    temp_root = (
        Path(tempfile.mkdtemp(prefix=RAW_INPUT_SNAPSHOT_PREFIX))
        if owns_temp_root
        else Path(os.path.abspath(tempfile.gettempdir()))
    )
    _check_path_components(temp_root)
    parent_directory: BoundReportDirectory | None = None
    snapshot_directory: BoundReportDirectory | None = None
    snapshots: dict[str, BoundInputSnapshot] = {}
    probe: _OwnedDescriptor | None = None
    primary: BaseException | None = None
    cleanup_errors: list[BaseException] = []
    try:
        if owns_temp_root:
            parent_directory = _open_report_directory(temp_root.parent)
        snapshot_directory = _open_report_directory(temp_root)
        if not owns_temp_root:
            try:
                probe = _create_posix_anonymous_file(
                    snapshot_directory, label="raw-input snapshot capability probe"
                )
            finally:
                if probe is not None:
                    closing_probe, probe = probe, None
                    _close_owned_descriptor(closing_probe, preserve=sys.exc_info()[1])
        for relative in RAW_PREPROCESSING_INPUTS:
            destination = temp_root / (
                f".{RAW_INPUT_SNAPSHOT_PREFIX}{uuid.uuid4().hex}-"
                f"{Path(relative).name}"
            )
            snapshots[relative] = _copy_verified_input_to_private_snapshot(
                _workspace_path(relative, label=f"raw preprocessing input {relative}"),
                declared_workspace_records[relative],
                destination,
                label=f"raw preprocessing input {relative}",
                context=context,
                directory=snapshot_directory,
            )
        yield snapshots
        for relative, snapshot in snapshots.items():
            _verify_bound_input_unchanged(
                snapshot, label=f"private raw input snapshot {relative}"
            )
    except BaseException as exc:
        primary = exc
        raise
    finally:
        if probe is not None:
            try:
                _close_owned_descriptor(probe, preserve=primary)
            except BaseException as exc:
                cleanup_errors.append(exc)
        for snapshot in snapshots.values():
            try:
                _close_owned_descriptor(snapshot.descriptor, preserve=primary)
            except BaseException as exc:
                cleanup_errors.append(exc)
            if snapshot.path_is_bound and snapshot_directory is not None:
                try:
                    _remove_identity_bound_private_snapshot(
                        snapshot_directory, snapshot.path.name, snapshot.identity
                    )
                except BaseException as exc:
                    cleanup_errors.append(exc)
        if snapshot_directory is not None:
            try:
                _close_owned_descriptor(snapshot_directory.descriptor, preserve=primary)
            except BaseException as exc:
                cleanup_errors.append(exc)
        if (
            owns_temp_root
            and parent_directory is not None
            and snapshot_directory is not None
        ):
            try:
                _remove_identity_bound_private_directory(
                    parent_directory, temp_root.name, snapshot_directory.identity
                )
            except BaseException as exc:
                cleanup_errors.append(exc)
        if parent_directory is not None:
            try:
                _close_owned_descriptor(parent_directory.descriptor, preserve=primary)
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            if primary is not None:
                for cleanup_error in cleanup_errors:
                    primary.add_note(
                        f"private snapshot cleanup failed: {cleanup_error!r}"
                    )
            else:
                cleanup_primary = cleanup_errors[0]
                for cleanup_error in cleanup_errors[1:]:
                    cleanup_primary.add_note(
                        f"additional private snapshot cleanup failure: "
                        f"{cleanup_error!r}"
                    )
                raise cleanup_primary


def _record_core(record: Any, *, label: str) -> dict[str, Any]:
    relative, size, digest = _validate_record(record, label=label)
    return {"path": relative, "bytes": size, "sha256": digest}


def _manifest_record_core(
    record: Any,
    *,
    fields: set[str],
    label: str,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise VerificationError(f"{label} must be a file record")
    _require_exact_keys(record, fields, label=label)
    mtime_ns = record.get("mtime_ns")
    if not isinstance(mtime_ns, int) or isinstance(mtime_ns, bool) or mtime_ns < 0:
        raise VerificationError(f"{label}.mtime_ns is invalid")
    core = _record_core(record, label=label)
    if expected is not None and core != dict(expected):
        raise VerificationError(f"{label} is not the registered production artifact")
    return core


def _snapshot_record(snapshot: FileSnapshot) -> dict[str, Any]:
    try:
        rendered = snapshot.path.relative_to(Path(os.path.abspath(ROOT))).as_posix()
    except ValueError:
        rendered = str(snapshot.path)
    return {"path": rendered, "bytes": snapshot.bytes, "sha256": snapshot.sha256}


def _verify_snapshot_unchanged(snapshot: FileSnapshot, *, label: str) -> None:
    descriptor: _OwnedDescriptor | None = None
    try:
        descriptor, opened_identity = _open_stable(snapshot.path, label=label)
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        closed_identity = _identity(os.fstat(descriptor))
    except BaseException as primary:
        if descriptor is not None:
            _close_owned_descriptor(descriptor, preserve=primary)
        raise
    else:
        _close_owned_descriptor(descriptor)
    try:
        after_identity = _identity(snapshot.path.lstat())
    except OSError as exc:
        raise VerificationError(f"{label} disappeared after it was checked") from exc
    if (
        opened_identity != snapshot.identity
        or opened_identity != closed_identity
        or opened_identity != after_identity
    ):
        raise VerificationError(f"{label} changed after it was checked")
    if total != snapshot.bytes or digest.hexdigest() != snapshot.sha256:
        raise VerificationError(f"{label} content changed after it was checked")


def _assert_snapshots_unchanged(context: VerificationContext) -> None:
    for snapshot in context.snapshots.values():
        _verify_snapshot_unchanged(snapshot, label=str(snapshot.path))


def _normalized_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def _trusted_install_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    seen: set[str] = set()
    configured = sysconfig.get_paths()
    for key in ("purelib", "platlib"):
        raw = configured.get(key)
        if not raw:
            continue
        root = Path(os.path.abspath(raw))
        _check_path_components(root)
        identity = os.path.normcase(str(root))
        if identity not in seen:
            roots.append(root)
            seen.add(identity)
    if not roots:
        raise VerificationError("trusted site-package roots are unavailable")
    return tuple(roots)


def _trusted_distribution(name: str, roots: Sequence[Path]) -> Any:
    expected = _normalized_distribution_name(name)
    matches: list[Any] = []
    for root in roots:
        for distribution in importlib.metadata.distributions(path=[str(root)]):
            recorded_name = distribution.metadata.get("Name")
            if (
                isinstance(recorded_name, str)
                and _normalized_distribution_name(recorded_name) == expected
            ):
                matches.append(distribution)
    if len(matches) != 1:
        raise VerificationError(
            f"trusted distribution inventory is ambiguous or missing: {name}"
        )
    distribution = matches[0]
    located_root = Path(os.path.abspath(distribution.locate_file("")))
    if all(
        os.path.normcase(str(located_root)) != os.path.normcase(str(root))
        for root in roots
    ):
        raise VerificationError(f"trusted distribution root is invalid: {name}")
    version = distribution.version
    if not isinstance(version, str) or not version:
        raise VerificationError(f"trusted distribution version is invalid: {name}")
    return distribution


def _trusted_distribution_inventory(roots: Sequence[Path]) -> dict[str, Any]:
    inventory: dict[str, Any] = {}
    for root in roots:
        for distribution in importlib.metadata.distributions(path=[str(root)]):
            name = distribution.metadata.get("Name")
            version = distribution.version
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(version, str)
                or not version
            ):
                raise VerificationError("trusted distribution metadata is incomplete")
            normalized = _normalized_distribution_name(name)
            previous = inventory.get(normalized)
            if previous is not None:
                raise VerificationError(
                    "trusted distribution inventory is ambiguous: " f"{name}"
                )
            inventory[normalized] = distribution
    if not inventory:
        raise VerificationError("trusted distribution inventory is empty")
    return inventory


def _distribution_dependency_closure(
    inventory: Mapping[str, Any], roots: Sequence[str]
) -> dict[str, tuple[str, str]]:
    """Return a conservative installed closure without importing packaging code."""
    pending = [_normalized_distribution_name(name) for name in roots]
    closure: dict[str, tuple[str, str]] = {}
    requirement_name = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
    while pending:
        normalized = pending.pop()
        if normalized in closure:
            continue
        distribution = inventory.get(normalized)
        if distribution is None:
            raise VerificationError(
                f"required trusted distribution is unavailable: {normalized}"
            )
        name = distribution.metadata.get("Name")
        version = distribution.version
        if not isinstance(name, str) or not isinstance(version, str):
            raise VerificationError("trusted distribution metadata is incomplete")
        closure[normalized] = (name, version)
        requirements = distribution.metadata.get_all("Requires-Dist") or []
        for requirement in requirements:
            if not isinstance(requirement, str):
                raise VerificationError(
                    f"trusted distribution requirement is invalid: {name}"
                )
            match = requirement_name.match(requirement)
            if match is None:
                raise VerificationError(
                    f"cannot parse trusted distribution requirement: {requirement!r}"
                )
            dependency = _normalized_distribution_name(match.group(1))
            # Requirements guarded by extras are deliberately over-approximated.
            if dependency in inventory and dependency not in closure:
                pending.append(dependency)
    return closure


def _dependency_closure_import_roots(
    inventory: Mapping[str, Any],
    closure: Mapping[str, tuple[str, str]],
) -> dict[str, tuple[str, ...]]:
    """Map import roots to their RECORD-bound dependency distributions."""
    owners: dict[str, set[str]] = defaultdict(set)
    for normalized_distribution in closure:
        distribution = inventory[normalized_distribution]
        candidates: set[str] = set()
        top_level = distribution.read_text("top_level.txt")
        if isinstance(top_level, str):
            candidates.update(line.strip() for line in top_level.splitlines())
        files = distribution.files or []
        for entry in files:
            relative = PurePosixPath(str(entry).replace("\\", "/"))
            if not relative.parts:
                continue
            first = relative.parts[0]
            if first.endswith((".dist-info", ".data")):
                continue
            if len(relative.parts) == 1:
                first = first.split(".", 1)[0]
            candidates.add(first)
        for candidate in candidates:
            if candidate.isidentifier():
                owners[candidate].add(normalized_distribution)
    return {root: tuple(sorted(values)) for root, values in sorted(owners.items())}


def _module_is_in_dependency_closure(
    module_name: str, import_roots: Mapping[str, Sequence[str]]
) -> bool:
    return module_name.split(".", 1)[0] in import_roots


def _validate_dependency_closure_module_origins(
    import_roots: Mapping[str, Sequence[str]],
    *,
    bind_new_modules: bool,
    modules_before: set[str] | None = None,
    allow_preexisting_for_internal_checks: bool = False,
) -> list[dict[str, Any]]:
    """Fail closed for closure modules that cannot be tied to a file artifact."""
    accepted: list[dict[str, Any]] = []
    for module_name, module in tuple(sys.modules.items()):
        if not _module_is_in_dependency_closure(module_name, import_roots):
            continue
        module_file = getattr(module, "__file__", None)
        specification = getattr(module, "__spec__", None)
        origin = getattr(specification, "origin", None)
        if isinstance(module_file, str) and isinstance(origin, str):
            continue
        existing = _TRUSTED_NONFILE_DEPENDENCY_BINDINGS.get(module_name)
        if existing is not None:
            bound_module, record = existing
            if module is not bound_module:
                raise VerificationError(
                    f"trust-bound non-file dependency changed object: {module_name}"
                )
            backing_path = record.get("backing_path")
            if isinstance(backing_path, str):
                snapshot = _TRUSTED_SCIENTIFIC_DEPENDENCY_SNAPSHOTS.get(
                    os.path.normcase(backing_path)
                )
                if snapshot is None:
                    raise VerificationError(
                        f"non-file dependency backing disappeared: {module_name}"
                    )
                _verify_snapshot_unchanged(
                    snapshot, label=f"non-file dependency backing {module_name}"
                )
            accepted.append(dict(record))
            continue
        reason = TRUSTED_NONFILE_DEPENDENCY_MODULES.get(module_name)
        if reason is not None and origin in {"built-in", "frozen"}:
            record = {
                "module": module_name,
                "module_type": type(module).__name__,
                "spec_origin": origin,
                "reason": reason,
            }
            _TRUSTED_NONFILE_DEPENDENCY_BINDINGS[module_name] = (module, record)
            accepted.append(record)
            continue
        if bind_new_modules and (
            allow_preexisting_for_internal_checks
            or modules_before is None
            or module_name not in modules_before
        ):
            if isinstance(module_file, str) and os.path.isabs(module_file):
                module_path = Path(os.path.abspath(module_file))
                artifact = _TRUSTED_DISTRIBUTION_ARTIFACTS.get(
                    os.path.normcase(str(module_path))
                )
                if (
                    artifact is not None
                    and _distribution_artifact_has_dependency_owner(
                        artifact, _TRUSTED_DEPENDENCY_CLOSURE
                    )
                ):
                    _payload, snapshot = _capture_distribution_artifact(
                        artifact,
                        label=f"preloaded dependency module {module_name}",
                        keep_bytes=False,
                    )
                    _remember_trusted_module_artifact(
                        module_name, artifact, snapshot
                    )
                    record = {
                        "module": module_name,
                        "module_type": type(module).__name__,
                        "spec_origin": origin,
                        "reason": "record_bound_preloaded_module_file",
                        "backing_path": str(snapshot.path),
                        "backing_sha256": snapshot.sha256,
                    }
                    _TRUSTED_NONFILE_DEPENDENCY_BINDINGS[module_name] = (
                        module,
                        record,
                    )
                    accepted.append(record)
                    continue
            locations = getattr(specification, "submodule_search_locations", None)
            if (
                module_file is None
                and origin is None
                and isinstance(
                    getattr(specification, "loader", None),
                    importlib.machinery.NamespaceLoader,
                )
                and locations is not None
            ):
                namespace_locations = [
                    Path(os.path.abspath(value)) for value in locations
                ]
                backing_artifacts = [
                    artifact
                    for artifact in _TRUSTED_DISTRIBUTION_ARTIFACTS.values()
                    if any(
                        _is_within(artifact.path, value)
                        for value in namespace_locations
                    )
                    and _distribution_artifact_has_dependency_owner(
                        artifact, _TRUSTED_DEPENDENCY_CLOSURE
                    )
                ]
                if namespace_locations and backing_artifacts:
                    for value in namespace_locations:
                        _check_path_components(value)
                        if not value.is_dir():
                            raise VerificationError(
                                f"namespace dependency location is invalid: {module_name}"
                            )
                    record = {
                        "module": module_name,
                        "module_type": type(module).__name__,
                        "spec_origin": None,
                        "reason": "record_bound_namespace_created_after_trust",
                        "namespace_locations": [
                            str(value) for value in namespace_locations
                        ],
                        "backing_distributions": sorted(
                            {
                                owner.distribution
                                for artifact in backing_artifacts
                                for owner in _distribution_artifact_owners(artifact)
                                if _normalized_distribution_name(
                                    owner.distribution
                                )
                                in _TRUSTED_DEPENDENCY_CLOSURE
                            }
                        ),
                    }
                    _TRUSTED_NONFILE_DEPENDENCY_BINDINGS[module_name] = (
                        module,
                        record,
                    )
                    accepted.append(record)
                    continue
            backing_module: str | None = None
            backing_path_key: str | None = None
            candidate = module_name
            while "." in candidate:
                candidate = candidate.rsplit(".", 1)[0]
                candidate_path = _TRUSTED_SCIENTIFIC_DEPENDENCY_MODULE_PATHS.get(
                    candidate
                )
                if candidate_path is not None:
                    backing_module = candidate
                    backing_path_key = candidate_path
                    break
            own_path_key = _TRUSTED_SCIENTIFIC_DEPENDENCY_MODULE_PATHS.get(module_name)
            if own_path_key is not None:
                backing_module = module_name
                backing_path_key = own_path_key
            if backing_module is not None and backing_path_key is not None:
                snapshot = _TRUSTED_SCIENTIFIC_DEPENDENCY_SNAPSHOTS.get(
                    backing_path_key
                )
                if snapshot is None:
                    raise VerificationError(
                        f"non-file dependency backing disappeared: {module_name}"
                    )
                if isinstance(module_file, str) and os.path.isabs(module_file):
                    module_path = Path(os.path.abspath(module_file))
                    same_canonical_path = os.path.normcase(
                        str(module_path)
                    ) == os.path.normcase(str(snapshot.path))
                    try:
                        same_file_object = _same_file_object(
                            snapshot.identity, _identity(module_path.stat())
                        )
                    except OSError:
                        same_file_object = False
                    if not same_canonical_path and not same_file_object:
                        raise VerificationError(
                            f"non-file dependency file differs from its backing: {module_name}"
                        )
                record = {
                    "module": module_name,
                    "module_type": type(module).__name__,
                    "spec_origin": origin,
                    "reason": "record_bound_dynamic_module_created_after_trust",
                    "backing_module": backing_module,
                    "backing_path": str(snapshot.path),
                    "backing_sha256": snapshot.sha256,
                }
                _TRUSTED_NONFILE_DEPENDENCY_BINDINGS[module_name] = (module, record)
                accepted.append(record)
                continue
        raise VerificationError(
            "dependency-closure module has no trust-bindable file origin: "
            f"{module_name} (module_type={type(module).__name__}, "
            f"module_file_type={type(module_file).__name__}, "
            f"spec_origin={origin!r}, reason=not in the closed builtin/frozen "
            "allowlist)"
        )
    return accepted


def _distribution_module_snapshot(
    distribution: Any,
    relative_path: str,
    *,
    label: str,
) -> tuple[FileSnapshot, str]:
    files = distribution.files
    if files is None:
        raise VerificationError(f"{label} distribution has no RECORD inventory")
    matches = [
        entry for entry in files if str(entry).replace("\\", "/") == relative_path
    ]
    if len(matches) != 1:
        raise VerificationError(f"{label} is absent from the distribution RECORD")
    entry = matches[0]
    record_hash = entry.hash
    record_size = entry.size
    if (
        record_hash is None
        or record_hash.mode != "sha256"
        or not isinstance(record_hash.value, str)
        or not record_hash.value
        or not isinstance(record_size, int)
        or isinstance(record_size, bool)
        or record_size < 0
    ):
        raise VerificationError(f"{label} lacks a SHA-256 RECORD binding")
    path = Path(os.path.abspath(distribution.locate_file(entry)))
    distribution_root = Path(os.path.abspath(distribution.locate_file("")))
    try:
        observed_relative = path.relative_to(distribution_root).as_posix()
    except ValueError as exc:
        raise VerificationError(f"{label} escapes its distribution root") from exc
    if observed_relative != relative_path:
        raise VerificationError(f"{label} has a noncanonical distribution path")
    context = VerificationContext(snapshots={})
    payload, snapshot = _capture_file(
        path,
        label=label,
        context=context,
        keep_bytes=False,
    )
    assert payload is None
    observed_record_hash = (
        base64.urlsafe_b64encode(bytes.fromhex(snapshot.sha256))
        .decode("ascii")
        .rstrip("=")
    )
    if snapshot.bytes != record_size or observed_record_hash != record_hash.value:
        raise VerificationError(f"{label} bytes differ from the distribution RECORD")
    return snapshot, record_hash.value


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _trusted_import_search_path(install_roots: Sequence[Path]) -> list[str]:
    configured = sysconfig.get_paths()
    standard_roots = [
        Path(os.path.abspath(value))
        for key in ("stdlib", "platstdlib")
        if (value := configured.get(key))
    ]
    allowed: list[str] = []
    seen: set[str] = set()
    for raw in sys.path:
        if not raw:
            continue
        candidate = Path(os.path.abspath(raw))
        is_install_root = any(
            os.path.normcase(str(candidate)) == os.path.normcase(str(root))
            for root in install_roots
        )
        is_standard = any(
            os.path.normcase(str(candidate)) == os.path.normcase(str(root))
            or _is_within(candidate, root)
            for root in standard_roots
        )
        is_runtime_archive = candidate.suffix.casefold() == ".zip" and _is_within(
            candidate, Path(os.path.abspath(sys.base_prefix))
        )
        is_runtime_native = candidate.name.casefold() in {
            "dlls",
            "lib-dynload",
        } and _is_within(candidate, Path(os.path.abspath(sys.base_prefix)))
        if not (
            is_install_root or is_standard or is_runtime_archive or is_runtime_native
        ):
            continue
        rendered = str(candidate)
        identity = os.path.normcase(rendered)
        if identity not in seen:
            allowed.append(rendered)
            seen.add(identity)
    for root in (*standard_roots, *install_roots):
        rendered = str(root)
        identity = os.path.normcase(rendered)
        if identity not in seen:
            allowed.append(rendered)
            seen.add(identity)
    return allowed


def _contains_dependency_shadow(directory: Path, module_name: str) -> bool:
    if not directory.is_dir():
        return False
    if (directory / module_name).exists():
        return True
    suffixes = {
        *importlib.machinery.SOURCE_SUFFIXES,
        *importlib.machinery.BYTECODE_SUFFIXES,
        *importlib.machinery.EXTENSION_SUFFIXES,
    }
    return any((directory / f"{module_name}{suffix}").exists() for suffix in suffixes)


def _reject_untrusted_dependency_shadows(install_roots: Sequence[Path]) -> None:
    candidates: list[Path] = [Path(os.path.abspath(Path.cwd()))]
    candidates.extend(
        Path(os.path.abspath(raw or Path.cwd())) for raw in sys.path if raw is not None
    )
    pythonpath = os.environ.get("PYTHONPATH")
    if pythonpath:
        candidates.extend(
            Path(os.path.abspath(raw or Path.cwd()))
            for raw in pythonpath.split(os.pathsep)
        )
    seen: set[str] = set()
    for directory in candidates:
        identity = os.path.normcase(str(directory))
        if identity in seen:
            continue
        seen.add(identity)
        if any(identity == os.path.normcase(str(root)) for root in install_roots):
            continue
        for module_name in SCIENTIFIC_DEPENDENCY_ROOTS:
            if _contains_dependency_shadow(directory, module_name):
                raise VerificationError(
                    "untrusted import-path shadow exists for scientific dependency: "
                    f"{directory / module_name}"
                )


def _module_origin(module: Any, *, label: str) -> Path:
    module_file = getattr(module, "__file__", None)
    specification = getattr(module, "__spec__", None)
    specification_origin = getattr(specification, "origin", None)
    if not isinstance(module_file, str) or not isinstance(specification_origin, str):
        raise VerificationError(f"{label} has no file-backed import origin")
    file_path = Path(os.path.abspath(module_file))
    origin_path = Path(os.path.abspath(specification_origin))
    if os.path.normcase(str(file_path)) != os.path.normcase(str(origin_path)):
        raise VerificationError(f"{label} import origin and module file differ")
    return file_path


def _expected_spec_origin(
    module_name: str,
    *,
    expected: FileSnapshot,
    search_path: Sequence[str],
) -> None:
    specification = importlib.machinery.PathFinder.find_spec(module_name, search_path)
    origin = None if specification is None else specification.origin
    if not isinstance(origin, str) or os.path.normcase(
        str(Path(os.path.abspath(origin)))
    ) != os.path.normcase(str(expected.path)):
        raise VerificationError(
            f"{module_name} would not import from its trusted distribution RECORD"
        )


def _distribution_artifact_inventory(
    roots: Sequence[Path],
) -> tuple[dict[str, DistributionArtifact], int]:
    artifacts: dict[str, DistributionArtifact] = {}
    shared_artifact_keys: set[str] = set()
    distribution_count = 0
    for root in roots:
        for distribution in importlib.metadata.distributions(path=[str(root)]):
            name = distribution.metadata.get("Name")
            version = distribution.version
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(version, str)
                or not version
            ):
                raise VerificationError("trusted distribution metadata is incomplete")
            distribution_count += 1
            files = distribution.files
            if files is None:
                continue
            distribution_root = Path(os.path.abspath(str(distribution.locate_file(""))))
            for entry in files:
                record_hash = entry.hash
                record_size = entry.size
                if (
                    record_hash is None
                    or record_hash.mode != "sha256"
                    or not isinstance(record_hash.value, str)
                    or not record_hash.value
                    or not isinstance(record_size, int)
                    or isinstance(record_size, bool)
                    or record_size < 0
                ):
                    continue
                path = Path(os.path.abspath(str(distribution.locate_file(entry))))
                if not any(_is_within(path, trusted_root) for trusted_root in roots):
                    continue
                try:
                    relative_path = path.relative_to(distribution_root).as_posix()
                except ValueError:
                    continue
                artifact = DistributionArtifact(
                    distribution=name,
                    distribution_version=version,
                    relative_path=relative_path,
                    path=path,
                    bytes=record_size,
                    record_sha256_urlsafe_base64=record_hash.value,
                    owners=(
                        DistributionArtifactOwner(
                            distribution=name,
                            distribution_version=version,
                            relative_path=relative_path,
                        ),
                    ),
                )
                key = os.path.normcase(str(path))
                previous = artifacts.get(key)
                if previous is not None:
                    if (
                        previous.relative_path != artifact.relative_path
                        or previous.bytes != artifact.bytes
                        or previous.record_sha256_urlsafe_base64
                        != artifact.record_sha256_urlsafe_base64
                    ):
                        raise VerificationError(
                            f"conflicting distribution RECORD ownership: {path}"
                        )
                    owners = tuple(
                        sorted(
                            {
                                *_distribution_artifact_owners(previous),
                                *_distribution_artifact_owners(artifact),
                            },
                            key=lambda owner: (
                                _normalized_distribution_name(owner.distribution),
                                owner.distribution.casefold(),
                                owner.distribution,
                                owner.distribution_version,
                                owner.relative_path,
                            ),
                        )
                    )
                    primary = owners[0]
                    artifact = DistributionArtifact(
                        distribution=primary.distribution,
                        distribution_version=primary.distribution_version,
                        relative_path=primary.relative_path,
                        path=previous.path,
                        bytes=previous.bytes,
                        record_sha256_urlsafe_base64=(
                            previous.record_sha256_urlsafe_base64
                        ),
                        owners=owners,
                    )
                    shared_artifact_keys.add(key)
                artifacts[key] = artifact
    if not artifacts or distribution_count <= 0:
        raise VerificationError("trusted distribution RECORD inventory is empty")
    for key in sorted(shared_artifact_keys):
        _capture_distribution_artifact(
            artifacts[key],
            label=f"shared distribution RECORD artifact {artifacts[key].path}",
            keep_bytes=False,
        )
    return artifacts, distribution_count


def _capture_distribution_artifact(
    artifact: DistributionArtifact,
    *,
    label: str,
    keep_bytes: bool,
) -> tuple[bytes | None, FileSnapshot]:
    context = VerificationContext(snapshots={})
    payload, snapshot = _capture_file(
        artifact.path,
        label=label,
        context=context,
        keep_bytes=keep_bytes,
    )
    observed_record_hash = (
        base64.urlsafe_b64encode(bytes.fromhex(snapshot.sha256))
        .decode("ascii")
        .rstrip("=")
    )
    if (
        snapshot.bytes != artifact.bytes
        or observed_record_hash != artifact.record_sha256_urlsafe_base64
    ):
        raise VerificationError(f"{label} bytes differ from the distribution RECORD")
    return payload, snapshot


def _open_stable_for_native_execution(
    path: Path, *, label: str
) -> tuple[_OwnedDescriptor, FileIdentity]:
    """Open a native artifact while denying replacement on Windows."""
    if os.name != "nt":
        return _open_stable(path, label=label)

    import ctypes
    from ctypes import wintypes

    _check_path_components(path)
    try:
        before = path.lstat()
    except OSError as exc:
        raise VerificationError(f"cannot stat {label}: {path}") from exc
    if _is_link_like(before) or not stat.S_ISREG(before.st_mode):
        raise VerificationError(f"{label} is not a regular non-link file: {path}")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    handle_guard: _WindowsHandleAcquisitionGuard | None = None
    descriptor: _OwnedDescriptor | None = None
    try:
        handle_guard = _owned_windows_handle(
            create_file,
            str(path),
            0x80000000,  # GENERIC_READ
            0x00000001,  # FILE_SHARE_READ; deny write/delete/rename sharing
            None,
            3,  # OPEN_EXISTING
            0x00000080 | 0x00200000,  # NORMAL | OPEN_REPARSE_POINT
            None,
        )
        invalid = ctypes.c_void_p(-1).value
        handle = handle_guard.handle
        if handle == invalid:
            handle_guard.handle = None
            error = ctypes.get_last_error()
            raise VerificationError(f"cannot open {label}: {path}") from OSError(
                error, ctypes.FormatError(error), str(path)
            )
        descriptor = _adopt_windows_handle_as_descriptor(
            handle_guard, os.O_RDONLY | getattr(os, "O_BINARY", 0)
        )
        opened_identity = _identity(os.fstat(descriptor))
    except BaseException as primary:
        if descriptor is not None:
            _close_owned_descriptor(descriptor, preserve=primary)
        elif handle_guard is not None:
            handle_guard.close(preserve=primary)
        raise
    if _identity(before) != opened_identity:
        identity_error = VerificationError(
            f"{label} changed while being opened: {path}"
        )
        _close_owned_descriptor(descriptor, preserve=identity_error)
        raise identity_error
    return descriptor, opened_identity


def _descriptor_execution_path(
    descriptor: int, identity: FileIdentity, *, label: str
) -> Path:
    if os.name == "nt":
        raise VerificationError("Windows native execution must use its locked path")
    descriptor_number = int(descriptor)
    for candidate in (
        Path(f"/proc/self/fd/{descriptor_number}"),
        Path(f"/dev/fd/{descriptor_number}"),
    ):
        try:
            observed = _identity(candidate.stat())
        except OSError:
            continue
        if _same_file_object(identity, observed):
            return candidate
    raise VerificationError(f"descriptor-bound native loading is unavailable: {label}")


def _open_bound_distribution_artifact(
    module_name: str, artifact: DistributionArtifact
) -> BoundDistributionArtifact:
    label = f"trusted module {module_name}"
    descriptor: _OwnedDescriptor | None = None
    try:
        descriptor, opened_identity = _open_stable_for_native_execution(
            artifact.path, label=label
        )
        digest, total, closed_identity = _hash_open_descriptor(descriptor)
        try:
            path_identity = _identity(artifact.path.lstat())
        except OSError as exc:
            raise VerificationError(f"{label} disappeared during validation") from exc
        observed_record_hash = (
            base64.urlsafe_b64encode(bytes.fromhex(digest)).decode("ascii").rstrip("=")
        )
        if (
            opened_identity != closed_identity
            or opened_identity != path_identity
            or total != artifact.bytes
            or observed_record_hash != artifact.record_sha256_urlsafe_base64
        ):
            raise VerificationError(
                f"{label} bytes differ from the distribution RECORD"
            )
        snapshot = FileSnapshot(
            path=Path(os.path.abspath(artifact.path)),
            identity=opened_identity,
            sha256=digest,
            bytes=total,
        )
        _remember_trusted_module_artifact(module_name, artifact, snapshot)
        execution_path = (
            artifact.path
            if os.name == "nt"
            else _descriptor_execution_path(descriptor, opened_identity, label=label)
        )
        return BoundDistributionArtifact(
            descriptor=descriptor,
            execution_path=execution_path,
            snapshot=snapshot,
        )
    except BaseException as primary:
        if descriptor is not None:
            _close_owned_descriptor(descriptor, preserve=primary)
        raise


def _remember_trusted_module_artifact(
    module_name: str,
    artifact: DistributionArtifact,
    snapshot: FileSnapshot,
) -> None:
    key = os.path.normcase(str(artifact.path))
    previous = _TRUSTED_SCIENTIFIC_DEPENDENCY_SNAPSHOTS.get(key)
    if previous is not None and (
        previous.identity != snapshot.identity
        or previous.sha256 != snapshot.sha256
        or previous.bytes != snapshot.bytes
    ):
        raise VerificationError(
            f"trusted dependency changed while importing: {artifact.path}"
        )
    _TRUSTED_SCIENTIFIC_DEPENDENCY_SNAPSHOTS[key] = snapshot
    _TRUSTED_SCIENTIFIC_DEPENDENCY_MODULE_PATHS[module_name] = key


class _RecordValidatingLoader:
    """Validate a distribution artifact before any of its code can execute."""

    def __init__(
        self,
        module_name: str,
        delegate: Any,
        artifact: DistributionArtifact,
    ) -> None:
        self.module_name = module_name
        self.delegate = delegate
        self.artifact = artifact
        self._bound_artifact: BoundDistributionArtifact | None = None
        self._bound_delegate: Any | None = None
        self._bound_specification: Any | None = None
        self._public_specification: Any | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def _capture(self, *, keep_bytes: bool) -> tuple[bytes | None, FileSnapshot]:
        return _capture_distribution_artifact(
            self.artifact,
            label=f"trusted module {self.module_name}",
            keep_bytes=keep_bytes,
        )

    def _close_bound_artifact(self, *, preserve: BaseException | None = None) -> None:
        bound = self._bound_artifact
        self._bound_artifact = None
        self._bound_delegate = None
        self._bound_specification = None
        if bound is not None:
            _close_owned_descriptor(bound.descriptor, preserve=preserve)

    def _prepare_non_source(self, specification: Any) -> None:
        if self._bound_artifact is not None:
            return
        bound: BoundDistributionArtifact | None = None
        try:
            bound = _open_bound_distribution_artifact(self.module_name, self.artifact)
            delegate: Any
            if isinstance(self.delegate, importlib.machinery.ExtensionFileLoader):
                delegate = importlib.machinery.ExtensionFileLoader(
                    self.module_name, str(bound.execution_path)
                )
            elif isinstance(self.delegate, importlib.machinery.SourcelessFileLoader):
                delegate = importlib.machinery.SourcelessFileLoader(
                    self.module_name, str(bound.execution_path)
                )
            else:
                clone = getattr(
                    self.delegate, "_record_validating_loader_for_path", None
                )
                if clone is None:
                    raise VerificationError(
                        f"trusted module loader cannot be safely rebound: "
                        f"{self.module_name}"
                    )
                delegate = clone(str(bound.execution_path))
            locations = getattr(specification, "submodule_search_locations", None)
            bound_specification = importlib.util.spec_from_file_location(
                self.module_name,
                str(bound.execution_path),
                loader=delegate,
                submodule_search_locations=(
                    None if locations is None else list(locations)
                ),
            )
            if bound_specification is None:
                raise VerificationError(
                    f"cannot create a descriptor-bound spec for {self.module_name}"
                )
            self._bound_artifact = bound
            self._bound_delegate = delegate
            self._bound_specification = bound_specification
            self._public_specification = specification
        except BaseException as primary:
            if bound is not None:
                _close_owned_descriptor(bound.descriptor, preserve=primary)
            raise

    def create_module(self, specification: Any) -> Any:
        create_module = getattr(self.delegate, "create_module", None)
        if isinstance(self.delegate, importlib.machinery.SourceFileLoader):
            return None if create_module is None else create_module(specification)
        self._prepare_non_source(specification)
        assert self._bound_delegate is not None
        assert self._bound_specification is not None
        create_module = getattr(self._bound_delegate, "create_module", None)
        try:
            return (
                None
                if create_module is None
                else create_module(self._bound_specification)
            )
        except BaseException:
            self._close_bound_artifact()
            raise

    def exec_module(self, module: Any) -> None:
        is_source = isinstance(self.delegate, importlib.machinery.SourceFileLoader)
        if is_source:
            payload, snapshot = self._capture(keep_bytes=True)
            assert payload is not None
            code = self.delegate.source_to_code(payload, str(self.artifact.path))
            exec(code, module.__dict__)
            _verify_snapshot_unchanged(
                snapshot, label=f"trusted module {self.module_name}"
            )
            _remember_trusted_module_artifact(
                self.module_name, self.artifact, snapshot
            )
            return

        if self._bound_artifact is None:
            self._prepare_non_source(getattr(module, "__spec__", None))
        assert self._bound_artifact is not None
        assert self._bound_delegate is not None
        assert self._bound_specification is not None
        exec_module = getattr(self._bound_delegate, "exec_module", None)
        if exec_module is None:
            error = VerificationError(
                f"trusted module loader cannot execute {self.module_name}"
            )
            self._close_bound_artifact(preserve=error)
            raise error
        bound = self._bound_artifact
        module.__spec__ = self._bound_specification
        module.__loader__ = self._bound_delegate
        module.__file__ = str(bound.execution_path)
        retain_bound_artifact = False
        try:
            exec_module(module)
            digest, total, identity = _hash_open_descriptor(bound.descriptor)
            if (
                not _same_file_object(identity, bound.snapshot.identity)
                or total != bound.snapshot.bytes
                or digest != bound.snapshot.sha256
            ):
                raise VerificationError(
                    f"trusted module {self.module_name} descriptor content changed"
                )
            # dlopen caches /proc/self/fd/N by path. Retaining native descriptors
            # prevents a later extension from reusing N and receiving the wrong handle.
            retain_bound_artifact = isinstance(
                self._bound_delegate, importlib.machinery.ExtensionFileLoader
            )
        finally:
            module.__spec__ = self._public_specification
            module.__loader__ = self
            module.__file__ = str(self.artifact.path)
            if not retain_bound_artifact:
                self._close_bound_artifact()


class _RecordValidatingFinder:
    def __init__(
        self,
        install_roots: Sequence[Path],
        artifacts: Mapping[str, DistributionArtifact],
    ) -> None:
        self.install_roots = tuple(install_roots)
        self.artifacts = artifacts

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: Any | None = None,
    ) -> Any:
        specification = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if specification is None or not isinstance(specification.origin, str):
            return specification
        origin = Path(os.path.abspath(specification.origin))
        if not any(_is_within(origin, root) for root in self.install_roots):
            return specification
        artifact = self.artifacts.get(os.path.normcase(str(origin)))
        if artifact is None:
            raise VerificationError(
                f"{fullname} has no SHA-256 distribution RECORD binding"
            )
        if specification.loader is None:
            raise VerificationError(f"{fullname} has no executable module loader")
        validating_loader: Any = _RecordValidatingLoader(
            fullname, specification.loader, artifact
        )
        specification.loader = validating_loader
        return specification


def _activate_trusted_import_runtime(
    install_roots: Sequence[Path],
    artifacts: Mapping[str, DistributionArtifact],
) -> None:
    global _TRUSTED_IMPORT_RUNTIME
    if _TRUSTED_IMPORT_RUNTIME is not None:
        return
    runtime = TrustedImportRuntime(
        original_path=list(sys.path),
        original_meta_path=list(sys.meta_path),
        original_path_hooks=list(sys.path_hooks),
        original_importer_cache=dict(sys.path_importer_cache),
    )
    import_path = _trusted_import_search_path(install_roots)
    standard_file_hook = importlib.machinery.FileFinder.path_hook(
        (
            importlib.machinery.ExtensionFileLoader,
            importlib.machinery.EXTENSION_SUFFIXES,
        ),
        (importlib.machinery.SourceFileLoader, importlib.machinery.SOURCE_SUFFIXES),
        (
            importlib.machinery.SourcelessFileLoader,
            importlib.machinery.BYTECODE_SUFFIXES,
        ),
    )
    finder = _RecordValidatingFinder(install_roots, artifacts)
    sys.path[:] = import_path
    sys.meta_path[:] = [
        importlib.machinery.BuiltinImporter,
        importlib.machinery.FrozenImporter,
        finder,
    ]
    sys.path_hooks[:] = [zipimport.zipimporter, standard_file_hook]
    sys.path_importer_cache.clear()
    _TRUSTED_IMPORT_RUNTIME = runtime


def _restore_trusted_import_runtime() -> None:
    global _TRUSTED_IMPORT_RUNTIME
    runtime = _TRUSTED_IMPORT_RUNTIME
    if runtime is None:
        return
    sys.path[:] = runtime.original_path
    sys.meta_path[:] = runtime.original_meta_path
    sys.path_hooks[:] = runtime.original_path_hooks
    sys.path_importer_cache.clear()
    sys.path_importer_cache.update(runtime.original_importer_cache)
    _TRUSTED_IMPORT_RUNTIME = None


def _loaded_site_package_modules(
    install_roots: Sequence[Path],
) -> Iterator[tuple[str, Any, Path]]:
    for module_name, module in tuple(sys.modules.items()):
        module_file = getattr(module, "__file__", None)
        specification = getattr(module, "__spec__", None)
        origin = getattr(specification, "origin", None)
        if not isinstance(module_file, str) or not isinstance(origin, str):
            continue
        file_path = Path(os.path.abspath(module_file))
        origin_path = Path(os.path.abspath(origin))
        file_is_installed = any(_is_within(file_path, root) for root in install_roots)
        origin_is_installed = any(
            _is_within(origin_path, root) for root in install_roots
        )
        if not file_is_installed and not origin_is_installed:
            continue
        if os.path.normcase(str(file_path)) != os.path.normcase(str(origin_path)):
            raise VerificationError(
                f"{module_name} import origin and module file differ"
            )
        yield module_name, module, file_path


def _bind_startup_preloaded_dependency(
    module_name: str,
    module: Any,
    artifacts: Mapping[str, DistributionArtifact],
    dependency_closure: Mapping[str, tuple[str, str]],
) -> dict[str, Any]:
    expected_distribution = TRUSTED_STARTUP_PRELOADED_DEPENDENCIES[module_name]
    specification = getattr(module, "__spec__", None)
    module_file = getattr(module, "__file__", None)
    origin = getattr(specification, "origin", None)
    if not isinstance(module_file, str) or not isinstance(origin, str):
        raise VerificationError(
            f"startup dependency {module_name} has no file-backed import origin"
        )
    path = Path(os.path.abspath(module_file))
    origin_path = Path(os.path.abspath(origin))
    if os.path.normcase(str(path)) != os.path.normcase(str(origin_path)):
        raise VerificationError(
            f"startup dependency {module_name} import origin and module file differ"
        )
    artifact = artifacts.get(os.path.normcase(str(path)))
    normalized_distribution = _normalized_distribution_name(expected_distribution)
    owner = (
        None
        if artifact is None
        else _distribution_artifact_owner_for(artifact, expected_distribution)
    )
    if (
        artifact is None
        or owner is None
        or normalized_distribution not in dependency_closure
    ):
        raise VerificationError(
            f"startup dependency {module_name} is not owned by its trusted distribution"
        )
    loader = getattr(module, "__loader__", None)
    if not isinstance(loader, importlib.machinery.SourceFileLoader):
        raise VerificationError(
            f"startup dependency {module_name} has an unexpected loader"
        )
    _payload, snapshot = _capture_distribution_artifact(
        artifact,
        label=f"startup dependency {module_name}",
        keep_bytes=False,
    )
    _remember_trusted_module_artifact(module_name, artifact, snapshot)
    _TRUSTED_STARTUP_DEPENDENCY_BINDINGS[module_name] = module
    return {
        "module": module_name,
        "distribution": owner.distribution,
        "distribution_version": owner.distribution_version,
        "relative_path": owner.relative_path,
        "distribution_owners": _distribution_artifact_owner_records(artifact),
        "path": str(snapshot.path),
        "bytes": snapshot.bytes,
        "sha256": snapshot.sha256,
        "record_sha256_urlsafe_base64": artifact.record_sha256_urlsafe_base64,
        "loader_type": type(loader).__name__,
        "reason": "interpreter_startup_hook",
    }


def _is_registered_native_alias(module_name: str, module: Any, path_key: str) -> bool:
    specification = getattr(module, "__spec__", None)
    canonical_name = getattr(specification, "name", None)
    if (
        not isinstance(canonical_name, str)
        or canonical_name == module_name
        or sys.modules.get(canonical_name) is not module
        or _TRUSTED_SCIENTIFIC_DEPENDENCY_MODULE_PATHS.get(canonical_name) != path_key
    ):
        return False
    snapshot = _TRUSTED_SCIENTIFIC_DEPENDENCY_SNAPSHOTS.get(path_key)
    if snapshot is None or not str(snapshot.path).endswith(
        tuple(importlib.machinery.EXTENSION_SUFFIXES)
    ):
        return False
    _verify_snapshot_unchanged(snapshot, label=f"trusted native alias {module_name}")
    _TRUSTED_SCIENTIFIC_DEPENDENCY_MODULE_PATHS[module_name] = path_key
    return True


def _bind_loaded_site_package_artifacts(
    install_roots: Sequence[Path],
    artifacts: Mapping[str, DistributionArtifact],
    *,
    require_loader_registration: bool,
) -> None:
    for module_name, module, path in _loaded_site_package_modules(install_roots):
        path_key = os.path.normcase(str(path))
        artifact = artifacts.get(path_key)
        if artifact is None:
            # Some environment bootstrap modules are not installed distributions.
            continue
        if (
            _TRUSTED_SCIENTIFIC_DEPENDENCY_MODULE_PATHS.get(module_name) == path_key
            and path_key in _TRUSTED_SCIENTIFIC_DEPENDENCY_SNAPSHOTS
        ):
            continue
        if require_loader_registration and _is_registered_native_alias(
            module_name, module, path_key
        ):
            continue
        if require_loader_registration:
            raise VerificationError(
                f"loaded module {module_name} bypassed the RECORD-validating loader"
            )
        _payload, snapshot = _capture_distribution_artifact(
            artifact,
            label=f"trusted module {module_name}",
            keep_bytes=False,
        )
        _remember_trusted_module_artifact(module_name, artifact, snapshot)


def _refresh_trusted_dependency_evidence() -> dict[str, Any]:
    evidence = _TRUSTED_SCIENTIFIC_DEPENDENCY_EVIDENCE
    if evidence is None:
        raise VerificationError("scientific dependencies have not been trust-bound")
    modules_by_path: dict[str, list[str]] = defaultdict(list)
    for module_name, path_key in _TRUSTED_SCIENTIFIC_DEPENDENCY_MODULE_PATHS.items():
        modules_by_path[path_key].append(module_name)
    records: list[dict[str, Any]] = []
    for path_key, module_names in sorted(modules_by_path.items()):
        artifact = _TRUSTED_DISTRIBUTION_ARTIFACTS.get(path_key)
        snapshot = _TRUSTED_SCIENTIFIC_DEPENDENCY_SNAPSHOTS.get(path_key)
        if artifact is None or snapshot is None:
            raise VerificationError("trusted dependency artifact evidence disappeared")
        aliases = sorted(module_names)
        records.append(
            {
                "module": aliases[0],
                "module_aliases": aliases,
                "distribution": artifact.distribution,
                "distribution_version": artifact.distribution_version,
                "relative_path": artifact.relative_path,
                "distribution_owners": _distribution_artifact_owner_records(artifact),
                "path": str(snapshot.path),
                "bytes": snapshot.bytes,
                "sha256": snapshot.sha256,
                "record_sha256_urlsafe_base64": (artifact.record_sha256_urlsafe_base64),
            }
        )
    evidence["modules"] = records
    evidence["loaded_artifact_count"] = len(records)
    return evidence


def _revalidate_trusted_scientific_dependencies() -> dict[str, Any]:
    evidence = _TRUSTED_SCIENTIFIC_DEPENDENCY_EVIDENCE
    if evidence is None:
        raise VerificationError("scientific dependencies have not been trust-bound")
    install_roots = tuple(Path(value) for value in evidence["trusted_install_roots"])
    evidence["non_file_modules"] = _validate_dependency_closure_module_origins(
        _TRUSTED_DEPENDENCY_IMPORT_ROOTS,
        bind_new_modules=False,
    )
    _bind_loaded_site_package_artifacts(
        install_roots,
        _TRUSTED_DISTRIBUTION_ARTIFACTS,
        require_loader_registration=_SCIENTIFIC_DEPENDENCIES_BOUND_AFTER_TRUST,
    )
    for path_key, snapshot in _TRUSTED_SCIENTIFIC_DEPENDENCY_SNAPSHOTS.items():
        if path_key not in _TRUSTED_DISTRIBUTION_ARTIFACTS:
            raise VerificationError("trusted dependency RECORD evidence disappeared")
        _verify_snapshot_unchanged(snapshot, label=f"trusted artifact {snapshot.path}")
    for module_name, path_key in _TRUSTED_SCIENTIFIC_DEPENDENCY_MODULE_PATHS.items():
        module = sys.modules.get(module_name)
        current_snapshot = _TRUSTED_SCIENTIFIC_DEPENDENCY_SNAPSHOTS.get(path_key)
        if current_snapshot is None:
            raise VerificationError(
                f"trust-bound scientific dependency disappeared: {module_name}"
            )
        if module is None:
            raise VerificationError(
                f"trust-bound scientific dependency disappeared: {module_name}"
            )
        startup_module = _TRUSTED_STARTUP_DEPENDENCY_BINDINGS.get(module_name)
        if startup_module is not None and module is not startup_module:
            raise VerificationError(
                f"trust-bound startup dependency changed object: {module_name}"
            )
        non_file_binding = _TRUSTED_NONFILE_DEPENDENCY_BINDINGS.get(module_name)
        if non_file_binding is not None:
            bound_module, record = non_file_binding
            if module is not bound_module:
                raise VerificationError(
                    f"trust-bound non-file dependency changed object: {module_name}"
                )
            backing_path = record.get("backing_path")
            if not isinstance(backing_path, str) or os.path.normcase(
                backing_path
            ) != os.path.normcase(str(current_snapshot.path)):
                raise VerificationError(
                    f"trust-bound non-file dependency backing changed: {module_name}"
                )
            continue
        module_file = getattr(module, "__file__", None)
        specification = getattr(module, "__spec__", None)
        specification_origin = getattr(specification, "origin", None)
        if not isinstance(module_file, str) or not isinstance(
            specification_origin, str
        ):
            raise VerificationError(f"{module_name} has no file-backed import origin")
        if os.path.normcase(
            str(Path(os.path.abspath(module_file)))
        ) != os.path.normcase(str(current_snapshot.path)) or os.path.normcase(
            str(Path(os.path.abspath(specification_origin)))
        ) != os.path.normcase(
            str(current_snapshot.path)
        ):
            raise VerificationError(
                f"trust-bound scientific dependency origin changed: {module_name}"
            )
    return _refresh_trusted_dependency_evidence()


def _load_trusted_scientific_dependencies(
    *, allow_preloaded_for_internal_checks: bool = False
) -> dict[str, Any]:
    """Load scientific code through RECORD validation before every execution."""
    global PCA, _SCIENTIFIC_DEPENDENCIES_BOUND_AFTER_TRUST
    global _TRUSTED_DEPENDENCY_CLOSURE
    global _TRUSTED_DEPENDENCY_IMPORT_ROOTS
    global _TRUSTED_NONFILE_DEPENDENCY_BINDINGS
    global _TRUSTED_STARTUP_DEPENDENCY_BINDINGS
    global _TRUSTED_DISTRIBUTION_ARTIFACTS
    global _TRUSTED_SCIENTIFIC_DEPENDENCY_EVIDENCE
    global _TRUSTED_SCIENTIFIC_DEPENDENCY_MODULE_PATHS
    global _TRUSTED_SCIENTIFIC_DEPENDENCY_SNAPSHOTS
    global np, pd, stats, torch

    if _TRUSTED_SCIENTIFIC_DEPENDENCY_EVIDENCE is not None:
        if (
            not allow_preloaded_for_internal_checks
            and not _SCIENTIFIC_DEPENDENCIES_BOUND_AFTER_TRUST
        ):
            raise VerificationError(
                "scientific dependencies were loaded before verifier trust binding"
            )
        if not allow_preloaded_for_internal_checks:
            roots = tuple(
                Path(value)
                for value in _TRUSTED_SCIENTIFIC_DEPENDENCY_EVIDENCE[
                    "trusted_install_roots"
                ]
            )
            _activate_trusted_import_runtime(roots, _TRUSTED_DISTRIBUTION_ARTIFACTS)
        return _revalidate_trusted_scientific_dependencies()

    preloaded = sorted(
        name
        for name in sys.modules
        if any(
            name == root or name.startswith(f"{root}.")
            for root in SCIENTIFIC_DEPENDENCY_ROOTS
        )
    )
    if preloaded and not allow_preloaded_for_internal_checks:
        raise VerificationError(
            "scientific dependencies were preloaded before verifier trust binding: "
            + ", ".join(preloaded[:8])
        )

    install_roots = _trusted_install_roots()
    _reject_untrusted_dependency_shadows(install_roots)
    distribution_inventory = _trusted_distribution_inventory(install_roots)
    direct_distribution_names = tuple(
        dict.fromkeys(
            distribution
            for _module, distribution, _path in SCIENTIFIC_DEPENDENCY_MODULES
        )
    )
    try:
        distributions = {
            name: distribution_inventory[_normalized_distribution_name(name)]
            for name in direct_distribution_names
        }
    except KeyError as exc:
        raise VerificationError(
            f"required trusted distribution is unavailable: {exc.args[0]}"
        ) from exc
    dependency_closure = _distribution_dependency_closure(
        distribution_inventory, direct_distribution_names
    )
    artifacts, distribution_count = _distribution_artifact_inventory(install_roots)
    dependency_import_roots = _dependency_closure_import_roots(
        distribution_inventory, dependency_closure
    )
    startup_preloaded_modules: list[tuple[str, Any]] = []
    if not allow_preloaded_for_internal_checks:
        preloaded_closure: list[str] = []
        for module_name, module in tuple(sys.modules.items()):
            if not _module_is_in_dependency_closure(
                module_name, dependency_import_roots
            ):
                continue
            if module_name in TRUSTED_STARTUP_PRELOADED_DEPENDENCIES:
                startup_preloaded_modules.append((module_name, module))
            else:
                preloaded_closure.append(module_name)
        startup_preloaded_modules.sort(key=lambda item: item[0])
        preloaded_closure.sort()
        if preloaded_closure:
            raise VerificationError(
                "dependency-closure modules were preloaded before verifier trust "
                "binding: " + ", ".join(preloaded_closure[:8])
            )
    direct_snapshots: dict[str, FileSnapshot] = {}
    for module_name, distribution_name, relative_path in SCIENTIFIC_DEPENDENCY_MODULES:
        snapshot, _record_hash = _distribution_module_snapshot(
            distributions[distribution_name],
            relative_path,
            label=f"scientific dependency {module_name}",
        )
        artifact = artifacts.get(os.path.normcase(str(snapshot.path)))
        owner = (
            None
            if artifact is None
            else _distribution_artifact_owner_for(artifact, distribution_name)
        )
        if (
            artifact is None
            or owner is None
            or owner.relative_path != relative_path
        ):
            raise VerificationError(
                f"scientific dependency inventory differs for {module_name}"
            )
        direct_snapshots[module_name] = snapshot

    import_path = _trusted_import_search_path(install_roots)
    modules_before = set(sys.modules)
    loaded: dict[str, Any] = {}
    _TRUSTED_DISTRIBUTION_ARTIFACTS = artifacts
    _TRUSTED_DEPENDENCY_CLOSURE = dependency_closure
    _TRUSTED_DEPENDENCY_IMPORT_ROOTS = dependency_import_roots
    _TRUSTED_NONFILE_DEPENDENCY_BINDINGS = {}
    _TRUSTED_STARTUP_DEPENDENCY_BINDINGS = {}
    _TRUSTED_SCIENTIFIC_DEPENDENCY_SNAPSHOTS = {}
    _TRUSTED_SCIENTIFIC_DEPENDENCY_MODULE_PATHS = {}
    startup_preloaded_evidence: list[dict[str, Any]] = []
    try:
        for module_name, module in startup_preloaded_modules:
            startup_preloaded_evidence.append(
                _bind_startup_preloaded_dependency(
                    module_name, module, artifacts, dependency_closure
                )
            )
        _activate_trusted_import_runtime(install_roots, artifacts)
        for (
            module_name,
            _distribution_name,
            _relative_path,
        ) in SCIENTIFIC_DEPENDENCY_MODULES:
            if "." in module_name:
                parent_name = module_name.rsplit(".", 1)[0]
                parent = loaded[parent_name]
                module_search_path = list(parent.__path__)
            else:
                module_search_path = import_path
            _expected_spec_origin(
                module_name,
                expected=direct_snapshots[module_name],
                search_path=module_search_path,
            )
            module = importlib.import_module(module_name)
            if os.path.normcase(
                str(_module_origin(module, label=module_name))
            ) != os.path.normcase(str(direct_snapshots[module_name].path)):
                raise VerificationError(
                    f"{module_name} imported from outside its trusted distribution"
                )
            _verify_snapshot_unchanged(
                direct_snapshots[module_name], label=f"trusted module {module_name}"
            )
            loaded[module_name] = module
        _bind_loaded_site_package_artifacts(
            install_roots,
            artifacts,
            require_loader_registration=not allow_preloaded_for_internal_checks,
        )
        non_file_modules = _validate_dependency_closure_module_origins(
            dependency_import_roots,
            bind_new_modules=True,
            modules_before=modules_before,
            allow_preexisting_for_internal_checks=allow_preloaded_for_internal_checks,
        )
    except BaseException:
        for name in tuple(sys.modules):
            if name not in modules_before:
                sys.modules.pop(name, None)
        _restore_trusted_import_runtime()
        _TRUSTED_DISTRIBUTION_ARTIFACTS = {}
        _TRUSTED_DEPENDENCY_CLOSURE = {}
        _TRUSTED_DEPENDENCY_IMPORT_ROOTS = {}
        _TRUSTED_NONFILE_DEPENDENCY_BINDINGS = {}
        _TRUSTED_STARTUP_DEPENDENCY_BINDINGS = {}
        _TRUSTED_SCIENTIFIC_DEPENDENCY_SNAPSHOTS = {}
        _TRUSTED_SCIENTIFIC_DEPENDENCY_MODULE_PATHS = {}
        raise
    finally:
        if allow_preloaded_for_internal_checks:
            _restore_trusted_import_runtime()

    np = loaded["numpy"]
    stats = loaded["scipy.stats"]
    pd = loaded["pandas"]
    PCA = loaded["sklearn.decomposition"].PCA
    torch = loaded["torch"]
    evidence = {
        "policy": "stdlib_bootstrap_then_record_prevalidated_loaded_artifacts",
        "preloaded_dependencies_accepted": bool(
            preloaded or startup_preloaded_evidence
        ),
        "startup_preloaded_modules": startup_preloaded_evidence,
        "non_file_modules": non_file_modules,
        "trusted_install_roots": [str(root) for root in install_roots],
        "record_inventory_distribution_count": distribution_count,
        "record_inventory_hashed_artifact_count": len(artifacts),
        "dependency_closure": [
            {
                "distribution": name,
                "distribution_version": version,
                "normalized_distribution": normalized,
            }
            for normalized, (name, version) in sorted(dependency_closure.items())
        ],
        "dependency_closure_distribution_count": len(dependency_closure),
        "loaded_artifact_count": 0,
        "modules": [],
    }
    _TRUSTED_SCIENTIFIC_DEPENDENCY_EVIDENCE = evidence
    _SCIENTIFIC_DEPENDENCIES_BOUND_AFTER_TRUST = not allow_preloaded_for_internal_checks
    return _revalidate_trusted_scientific_dependencies()


def _verification_checkpoint(_label: str) -> None:
    """Scheduling checkpoint used by callers; intentionally has no side effects."""


def _require_exact_keys(
    value: Mapping[str, Any], keys: set[str], *, label: str
) -> None:
    if set(value) != keys:
        raise VerificationError(
            f"{label} inventory mismatch: expected {sorted(keys)}, found {sorted(value)}"
        )


def _require_file_inventory(directory: Path, expected: set[str], *, label: str) -> None:
    _check_path_components(directory)
    try:
        entries = list(directory.iterdir())
    except OSError as exc:
        raise VerificationError(f"cannot enumerate {label}: {directory}") from exc
    for entry in entries:
        info = entry.lstat()
        if _is_link_like(info) or not stat.S_ISREG(info.st_mode):
            raise VerificationError(
                f"{label} contains a non-regular entry: {entry.name}"
            )
    observed = {entry.name for entry in entries}
    if observed != expected:
        raise VerificationError(
            f"{label} inventory mismatch: expected {sorted(expected)}, found {sorted(observed)}"
        )


def _load_bound_json(
    path: Path,
    record: Any | None,
    *,
    label: str,
    context: VerificationContext,
) -> tuple[dict[str, Any], FileSnapshot]:
    if record is None:
        payload, snapshot = _capture_file(
            path, label=label, context=context, keep_bytes=True
        )
    else:
        payload, snapshot = _verify_record(
            path, record, label=label, context=context, keep_bytes=True
        )
    assert payload is not None
    return _parse_json(payload, label=label), snapshot


def _verify_declared_workspace_records(
    records: Mapping[str, dict[str, Any]], *, context: VerificationContext
) -> None:
    for relative, record in sorted(records.items()):
        path = _workspace_path(relative, label=f"declared input {relative}")
        _verify_record(
            path,
            record,
            label=f"declared input {relative}",
            context=context,
        )


def _expected_campaign_keys() -> set[str]:
    return {
        f"{protocol}|{condition}|seed-{seed}"
        for protocol in CAMPAIGN_PROTOCOLS
        for condition in CAMPAIGN_CONDITIONS
        for seed in SEEDS
    }


def _parse_run_key(value: Any) -> tuple[str, str, int]:
    if not isinstance(value, str):
        raise VerificationError("campaign run key must be a string")
    pieces = value.split("|")
    if len(pieces) != 3 or not pieces[2].startswith("seed-"):
        raise VerificationError(f"malformed campaign run key: {value!r}")
    try:
        seed = int(pieces[2][5:])
    except ValueError as exc:
        raise VerificationError(f"malformed campaign run seed: {value!r}") from exc
    return pieces[0], pieces[1], seed


def _merge_declared_record(
    destination: dict[str, dict[str, Any]], record: Any, *, label: str
) -> None:
    core = _record_core(record, label=label)
    relative = core["path"]
    previous = destination.get(relative)
    if previous is not None and previous != core:
        raise VerificationError(
            f"inconsistent declarations for workspace input {relative}"
        )
    destination[relative] = core


def _verify_campaign(
    campaign_path: Path,
    campaign_record: Any,
    *,
    run_paths: Mapping[str, Any],
    context: VerificationContext,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, Path],
    int,
    dict[tuple[str, str], FoldEvidence],
]:
    campaign, _ = _load_bound_json(
        campaign_path,
        campaign_record,
        label="campaign report",
        context=context,
    )
    _require_exact_keys(campaign, CAMPAIGN_FIELDS, label="campaign report")
    if campaign.get("schema") != CAMPAIGN_SCHEMA:
        raise VerificationError("unexpected campaign schema")
    if (
        campaign.get("scientific_result") is not True
        or campaign.get("protocol_validation_passed") is not True
    ):
        raise VerificationError("campaign did not pass its scientific/protocol gates")
    readiness = campaign.get("scientific_readiness_gates")
    if (
        not isinstance(readiness, Mapping)
        or set(readiness)
        != {
            "outer_test_metrics_recomputed_from_bound_predictions",
            "verified_embedding_generation_provenance",
        }
        or not all(value is True for value in readiness.values())
    ):
        raise VerificationError("campaign scientific-readiness gates are not all true")
    matrix = campaign.get("prespecified_matrix")
    expected_matrix = {
        "canonical_arguments": {
            name: EXPECTED_ARGUMENTS[name]
            for name in (
                "batch_size",
                "folds",
                "gene_subsample",
                "inner_max_epochs",
                "inner_val_fraction",
                "n_expr_pc",
                "patience",
                "top_mut",
            )
        },
        "conditions": sorted(CAMPAIGN_CONDITIONS),
        "expected_runs": EXPECTED_CAMPAIGN_RUNS,
        "p1_outer_folds_per_run": 5,
        "protocols": list(CAMPAIGN_PROTOCOLS),
        "seeds": list(SEEDS),
        "transfer_outer_folds_per_run": 1,
        "tuning_grid": [
            {"hidden": hidden, "lamb_sel": 0.1, "lr": lr, "wd": 1e-5}
            for hidden in (128, 256)
            for lr in (1e-3, 3e-4)
        ],
    }
    if matrix != expected_matrix:
        raise VerificationError("campaign prespecified matrix is noncanonical")
    aggregation = campaign.get("aggregation_provenance")
    if not isinstance(aggregation, Mapping) or not isinstance(
        aggregation.get("embedding_bundle"), Mapping
    ):
        raise VerificationError("campaign embedding provenance is missing")
    expected_embedding_bundle = aggregation["embedding_bundle"]
    source_runs = campaign.get("source_runs")
    if not isinstance(source_runs, list) or len(source_runs) != EXPECTED_CAMPAIGN_RUNS:
        raise VerificationError("campaign source-run count is noncanonical")
    if not isinstance(run_paths, Mapping):
        raise VerificationError("analysis campaign_run_paths must be an object")
    if set(run_paths) != _expected_campaign_keys():
        raise VerificationError("analysis campaign_run_paths key set is noncanonical")

    observed_runs: dict[str, dict[str, Any]] = {}
    resolved_paths: dict[str, Path] = {}
    declared_workspace_records: dict[str, dict[str, Any]] = {}
    fold_count = 0
    for index, raw_run in enumerate(source_runs):
        label = f"campaign.source_runs[{index}]"
        if not isinstance(raw_run, dict):
            raise VerificationError(f"{label} must be an object")
        required = {
            "key",
            "path",
            "completion_sha256",
            "summary_sha256",
            "run_manifest_sha256",
            "cohort_manifest_sha256",
            "runtime_fingerprint_sha256",
            "fold_artifacts",
        }
        _require_exact_keys(raw_run, required, label=label)
        key = raw_run.get("key")
        protocol, condition, seed = _parse_run_key(key)
        if key in observed_runs:
            raise VerificationError(f"duplicate campaign run key: {key}")
        if key not in _expected_campaign_keys():
            raise VerificationError(f"unexpected campaign run key: {key}")
        if not all(
            isinstance(raw_run.get(name), str)
            and SHA256_PATTERN.fullmatch(raw_run[name]) is not None
            for name in (
                "completion_sha256",
                "summary_sha256",
                "run_manifest_sha256",
                "cohort_manifest_sha256",
                "runtime_fingerprint_sha256",
            )
        ):
            raise VerificationError(f"{label} has an invalid root digest")
        declared_path = run_paths.get(key)
        normalized_declared = _normalize_relative(
            declared_path, label=f"campaign_run_paths[{key}]"
        )
        report_path = raw_run.get("path")
        if not isinstance(report_path, str):
            raise VerificationError(f"{label}.path is invalid")
        report_resolved = _workspace_path(report_path, label=f"{label}.path")
        run_dir = _workspace_path(
            normalized_declared, label=f"campaign_run_paths[{key}]"
        )
        if report_resolved != run_dir:
            raise VerificationError(f"campaign path binding differs for {key}")
        artifacts = raw_run.get("fold_artifacts")
        expected_count = EXPECTED_FOLD_ARTIFACTS_BY_PROTOCOL[protocol]
        if not isinstance(artifacts, list) or len(artifacts) != expected_count:
            raise VerificationError(f"noncanonical fold-artifact count for {key}")
        artifact_records: dict[str, dict[str, Any]] = {}
        for artifact_index, artifact in enumerate(artifacts):
            artifact_label = f"{label}.fold_artifacts[{artifact_index}]"
            core = _record_core(artifact, label=artifact_label)
            name = core["path"]
            if "/" in name or name in artifact_records:
                raise VerificationError(
                    f"invalid or duplicate run artifact path for {key}"
                )
            artifact_records[name] = core
        if protocol == "p1":
            expected_names = {
                f"{fold}-{suffix}"
                for fold in FOLDS
                for suffix in ("predictions.npz", "result.json", "split.json")
            }
            if set(artifact_records) != expected_names:
                raise VerificationError(
                    f"P1 artifact inventory is noncanonical for {key}"
                )
        else:
            fold_name = (
                "complement-to-bkm"
                if protocol == "transfer-forward"
                else "bkm-to-complement"
            )
            expected_names = {
                f"{fold_name}-{suffix}"
                for suffix in ("predictions.npz", "result.json", "split.json")
            }
            if set(artifact_records) != expected_names:
                raise VerificationError(
                    f"transfer artifact inventory is noncanonical for {key}"
                )
        expected_inventory = set(RUN_ROOT_JSON_FILES) | set(artifact_records)
        _require_file_inventory(run_dir, expected_inventory, label=f"source run {key}")

        completion_payload, completion_snapshot = _capture_file(
            run_dir / "completion.json",
            label=f"{key}.completion.json",
            context=context,
            keep_bytes=True,
        )
        assert completion_payload is not None
        if completion_snapshot.sha256 != raw_run["completion_sha256"]:
            raise VerificationError(f"completion hash mismatch for {key}")
        completion = _parse_json(completion_payload, label=f"{key}.completion.json")
        _require_exact_keys(
            completion, RUN_COMPLETION_FIELDS, label=f"{key}.completion.json"
        )
        if (
            completion.get("schema") != RUN_COMPLETION_SCHEMA
            or completion.get("run_status") != "protocol_complete"
            or completion.get("eligible_for_scientific_aggregation") is not True
            or completion.get("scientific_result") is not False
            or completion.get("required_next_step") != RUN_REQUIRED_NEXT_STEP
        ):
            raise VerificationError(f"completion marker is not valid for {key}")
        if (
            completion.get("run_manifest_sha256") != raw_run["run_manifest_sha256"]
            or completion.get("summary_sha256") != raw_run["summary_sha256"]
            or completion.get("cohort_manifest_sha256")
            != raw_run["cohort_manifest_sha256"]
            or completion.get("summary_file") != "summary.json"
        ):
            raise VerificationError(f"completion root bindings differ for {key}")
        completion_artifacts = completion.get("artifacts")
        if not isinstance(completion_artifacts, list):
            raise VerificationError(f"completion artifact list is missing for {key}")
        completion_map: dict[str, dict[str, Any]] = {}
        for artifact_index, artifact in enumerate(completion_artifacts):
            core = _record_core(
                artifact, label=f"{key}.completion.artifacts[{artifact_index}]"
            )
            if core["path"] in completion_map:
                raise VerificationError(f"duplicate completion artifact for {key}")
            completion_map[core["path"]] = core
        if completion_map != artifact_records:
            raise VerificationError(
                f"completion/campaign artifact bindings differ for {key}"
            )

        root_hashes = {
            "summary.json": raw_run["summary_sha256"],
            "run-manifest.json": raw_run["run_manifest_sha256"],
            "cohort-manifest.json": raw_run["cohort_manifest_sha256"],
        }
        run_manifest: dict[str, Any] | None = None
        for name, expected_digest in root_hashes.items():
            payload, snapshot = _capture_file(
                run_dir / name,
                label=f"{key}.{name}",
                context=context,
                keep_bytes=name == "run-manifest.json",
            )
            if snapshot.sha256 != expected_digest:
                raise VerificationError(f"root artifact hash mismatch for {key}.{name}")
            if name == "run-manifest.json":
                assert payload is not None
                run_manifest = _parse_json(payload, label=f"{key}.{name}")
        assert run_manifest is not None
        _require_exact_keys(
            run_manifest,
            {
                "arguments",
                "argv",
                "code",
                "created_utc",
                "device",
                "eligible_for_scientific_aggregation",
                "embedding_provenance",
                "excluded_conditions",
                "git",
                "inputs",
                "output_policy",
                "pilot",
                "platform",
                "protocol_complete",
                "python",
                "run_status",
                "runtime",
                "schema",
                "scientific_result",
                "scientific_result_reason",
                "versions",
            },
            label=f"{key}.run-manifest.json",
        )
        if run_manifest.get("schema") != RUN_SCHEMA:
            raise VerificationError(f"unexpected run-manifest schema for {key}")
        arguments = run_manifest.get("arguments")
        if not isinstance(arguments, Mapping):
            raise VerificationError(f"run-manifest arguments differ from key {key}")
        expected_arguments = {
            **EXPECTED_ARGUMENTS,
            "condition": condition,
            "device": "cuda",
            "fold_id": -1,
            "pilot": False,
            "protocol": protocol,
            "seed": seed,
        }
        if dict(arguments) != expected_arguments or (
            arguments.get("protocol"),
            arguments.get("condition"),
            arguments.get("seed"),
        ) != (protocol, condition, seed):
            raise VerificationError(f"run-manifest arguments differ from key {key}")
        if (
            run_manifest.get("pilot") is not False
            or run_manifest.get("device") != "cuda"
            or run_manifest.get("scientific_result") is not False
            or run_manifest.get("embedding_provenance") != expected_embedding_bundle
        ):
            raise VerificationError(f"run-manifest provenance differs for {key}")
        for section in ("inputs", "code"):
            records = run_manifest.get(section)
            if not isinstance(records, list) or not records:
                raise VerificationError(f"{key}.{section} declarations are missing")
            for record_index, record in enumerate(records):
                _merge_declared_record(
                    declared_workspace_records,
                    record,
                    label=f"{key}.{section}[{record_index}]",
                )
            observed_paths = {
                _record_core(record, label=f"{key}.{section} path")["path"]
                for record in records
            }
            expected_paths = (
                set(EXPECTED_CODE_PATHS)
                if section == "code"
                else set(EXPECTED_COMMON_INPUT_PATHS)
                | ({"data/processed/splits.json"} if protocol != "p1" else set())
            )
            if observed_paths != expected_paths:
                raise VerificationError(
                    f"{key}.{section} path inventory is noncanonical"
                )
        for name, record in sorted(artifact_records.items()):
            _verify_record(
                _safe_child(run_dir, name, label=f"{key}.{name}"),
                record,
                label=f"{key}.{name}",
                context=context,
            )
        fold_count += len(artifact_records)
        observed_runs[key] = raw_run
        resolved_paths[key] = run_dir

    if set(observed_runs) != _expected_campaign_keys():
        raise VerificationError("campaign source-run key set is incomplete")
    if fold_count != EXPECTED_FOLD_ARTIFACTS:
        raise VerificationError("campaign fold-artifact total is noncanonical")
    _verify_declared_workspace_records(declared_workspace_records, context=context)
    fold_evidence = _verify_protocol_artifacts(
        observed_runs,
        resolved_paths,
        declared_workspace_records=declared_workspace_records,
        context=context,
    )
    return campaign, observed_runs, resolved_paths, fold_count, fold_evidence


def _derive_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(int(seed)), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _hash_strings(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _hash_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _as_json_indices(value: Any, *, label: str, n: int) -> np.ndarray:
    if not isinstance(value, list) or not value:
        raise VerificationError(f"{label} must be a non-empty index list")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        raise VerificationError(f"{label} contains a non-integer index")
    indices = np.asarray(value, dtype=np.int64)
    if (
        indices.ndim != 1
        or np.unique(indices).size != indices.size
        or int(indices.min()) < 0
        or int(indices.max()) >= n
    ):
        raise VerificationError(f"{label} contains invalid indices")
    return indices


def _validate_index_record(
    value: Any, *, label: str, cells: Sequence[str]
) -> np.ndarray:
    if not isinstance(value, Mapping):
        raise VerificationError(f"{label} must be an index record")
    _require_exact_keys(value, {"indices", "cell_ids", "sha256"}, label=label)
    indices = _as_json_indices(
        value.get("indices"), label=f"{label}.indices", n=len(cells)
    )
    expected_ids = [str(cells[int(index)]) for index in indices]
    if value.get("cell_ids") != expected_ids:
        raise VerificationError(f"{label} cell identities differ from canonical order")
    expected_hash = hashlib.sha256(
        indices.astype("<i8", copy=False).tobytes()
    ).hexdigest()
    if value.get("sha256") != expected_hash:
        raise VerificationError(f"{label} index hash does not recompute")
    return indices


def _make_p1_outer_splits(
    n_cells: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(int(seed))
    folds = np.array_split(rng.permutation(n_cells), len(FOLDS))
    return [
        (
            np.sort(
                np.concatenate([folds[j] for j in range(len(folds)) if j != i])
            ).astype(np.int64),
            np.sort(folds[i]).astype(np.int64),
        )
        for i in range(len(folds))
    ]


def _make_inner_split(
    outer_train: np.ndarray, lineages: Sequence[str], *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    lineage_array = np.asarray(lineages, dtype=str)
    rng = np.random.default_rng(int(seed))
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    for category in sorted(set(lineage_array[outer_train].tolist())):
        members = outer_train[lineage_array[outer_train] == category].copy()
        rng.shuffle(members)
        if members.size < 2:
            train_parts.append(members)
            continue
        n_validation = min(members.size - 1, max(1, int(round(members.size * 0.2))))
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


def _fit_gene_mean_baseline(
    gene_effect: np.ndarray, outer_train: np.ndarray
) -> np.ndarray:
    values = np.asarray(gene_effect, dtype=np.float32)
    train = values[:, outer_train]
    finite = np.isfinite(train)
    counts = finite.sum(axis=1)
    totals = np.where(finite, train, 0.0).sum(axis=1, dtype=np.float64)
    means = np.full(values.shape[0], np.nan, dtype=np.float64)
    np.divide(totals, counts, out=means, where=counts > 0)
    return means.astype(np.float32)


def _lineage_counts(lineages: Sequence[str], indices: np.ndarray) -> dict[str, int]:
    values = np.asarray(lineages, dtype=str)
    return dict(sorted(Counter(str(value) for value in values[indices]).items()))


def _artifact_map(run: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    artifacts = run.get("fold_artifacts")
    if not isinstance(artifacts, list):
        raise VerificationError(f"{run.get('key')} has no fold-artifact inventory")
    result: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(artifacts):
        core = _record_core(record, label=f"{run.get('key')}.artifact[{index}]")
        if core["path"] in result:
            raise VerificationError(f"duplicate fold artifact in {run.get('key')}")
        result[core["path"]] = core
    return result


def _load_run_json(
    run_dir: Path,
    name: str,
    record: Mapping[str, Any] | None,
    *,
    label: str,
    context: VerificationContext,
) -> dict[str, Any]:
    value, _ = _load_bound_json(run_dir / name, record, label=label, context=context)
    return value


def _load_run_root_json(
    run_dir: Path,
    name: str,
    expected_sha256: Any,
    *,
    label: str,
    context: VerificationContext,
) -> dict[str, Any]:
    if (
        not isinstance(expected_sha256, str)
        or SHA256_PATTERN.fullmatch(expected_sha256) is None
    ):
        raise VerificationError(f"{label} has an invalid trusted digest")
    payload, snapshot = _capture_file(
        run_dir / name, label=label, context=context, keep_bytes=True
    )
    if snapshot.sha256 != expected_sha256:
        raise VerificationError(f"{label} differs from its campaign root digest")
    assert payload is not None
    return _parse_json(payload, label=label)


def _validate_cohort_manifest(
    cohort: Mapping[str, Any],
    *,
    genes: Sequence[str],
    cells: Sequence[str],
    expression_names_sha256: str,
    n_string_edges: int,
) -> tuple[np.ndarray, np.ndarray]:
    _require_exact_keys(
        cohort,
        {
            "cell_ids_sha256",
            "expression_gene_names_sha256",
            "genes_sha256",
            "n_cells",
            "n_expression_features_raw",
            "n_genes",
            "n_string_edges",
            "source_cell_indices",
            "source_gene_indices",
        },
        label="cohort manifest",
    )
    expected_scalars = {
        "cell_ids_sha256": _hash_strings(cells),
        "expression_gene_names_sha256": expression_names_sha256,
        "genes_sha256": _hash_strings(genes),
        "n_cells": len(cells),
        "n_expression_features_raw": 17_572,
        "n_genes": len(genes),
        "n_string_edges": n_string_edges,
    }
    if any(cohort.get(name) != expected for name, expected in expected_scalars.items()):
        raise VerificationError("cohort manifest differs from canonical inputs")
    source_cells = _as_json_indices(
        cohort.get("source_cell_indices"),
        label="cohort source_cell_indices",
        n=len(cells),
    )
    source_genes = _as_json_indices(
        cohort.get("source_gene_indices"),
        label="cohort source_gene_indices",
        n=len(genes),
    )
    if not np.array_equal(source_cells, np.arange(len(cells), dtype=np.int64)):
        raise VerificationError("non-pilot cohort has a non-identity cell mapping")
    if not np.array_equal(source_genes, np.arange(len(genes), dtype=np.int64)):
        raise VerificationError("non-pilot cohort has a non-identity gene mapping")
    return source_cells, source_genes


def _validate_processed_splits(
    stored: Mapping[str, Any],
    *,
    cells: Sequence[str],
    lineages: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    _require_exact_keys(
        stored,
        {"P1_folds", "P2_train", "P2_test", "P3", "lineage_idx"},
        label="processed splits.json",
    )
    raw_folds = stored.get("P1_folds")
    if not isinstance(raw_folds, list) or len(raw_folds) != len(FOLDS):
        raise VerificationError("processed P1 fold inventory is noncanonical")
    observed_folds = [
        _as_json_indices(value, label=f"splits.P1_folds[{index}]", n=len(cells))
        for index, value in enumerate(raw_folds)
    ]
    expected_folds = np.array_split(
        np.random.default_rng(42).permutation(len(cells)), 5
    )
    if any(
        not np.array_equal(observed, expected.astype(np.int64, copy=False))
        for observed, expected in zip(observed_folds, expected_folds, strict=True)
    ):
        raise VerificationError(
            "processed P1 folds differ from the frozen preprocessing split"
        )
    if not np.array_equal(
        np.sort(np.concatenate(observed_folds)), np.arange(len(cells), dtype=np.int64)
    ):
        raise VerificationError("processed P1 folds do not partition every cell once")

    complement = _as_json_indices(
        stored.get("P2_train"), label="splits.P2_train", n=len(cells)
    )
    bkm = _as_json_indices(stored.get("P2_test"), label="splits.P2_test", n=len(cells))
    if (
        len(complement) != 998
        or len(bkm) != 105
        or set(complement).intersection(bkm)
        or set(complement).union(bkm) != set(range(len(cells)))
        or stored.get("P3") is not None
    ):
        raise VerificationError("processed transfer split topology is noncanonical")
    if _lineage_counts(lineages, bkm) != TRANSFER_HOLDOUT_COUNTS:
        raise VerificationError("BKM holdout lineage counts are noncanonical")
    if set(np.asarray(lineages, dtype=str)[complement]).intersection(
        TRANSFER_HOLDOUT_COUNTS
    ):
        raise VerificationError("BKM lineages leaked into transfer complement")

    category_ids: dict[str, int] = {}
    expected_lineage_indices: list[int] = []
    for lineage in lineages:
        category_ids.setdefault(str(lineage), len(category_ids))
        expected_lineage_indices.append(category_ids[str(lineage)])
    raw_lineage_indices = stored.get("lineage_idx")
    if (
        not isinstance(raw_lineage_indices, list)
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in raw_lineage_indices
        )
        or raw_lineage_indices != expected_lineage_indices
        or len(category_ids) != 29
    ):
        raise VerificationError("processed lineage_idx does not recompute")
    return complement, bkm


def _validate_split_manifest(
    split: Mapping[str, Any],
    *,
    protocol: str,
    fold: str,
    fold_seed: int,
    expected_outer_train: np.ndarray,
    expected_outer_test: np.ndarray,
    cells: Sequence[str],
    lineages: Sequence[str],
    source_cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _require_exact_keys(
        split,
        {
            "fit_scope",
            "fold",
            "inner_train",
            "inner_validation",
            "k_tokens",
            "lineage_counts",
            "outer_test",
            "outer_train",
            "protocol",
            "seed",
            "source_cell_indices",
        },
        label=f"{protocol}.{fold} split",
    )
    if (
        split.get("fit_scope") != EXPECTED_FIT_SCOPE
        or split.get("k_tokens") != 64
        or split.get("fold") != fold
        or split.get("protocol") != protocol
        or split.get("seed") != fold_seed
    ):
        raise VerificationError(f"{protocol}.{fold} split metadata is noncanonical")
    outer_train = _validate_index_record(
        split.get("outer_train"), label=f"{protocol}.{fold}.outer_train", cells=cells
    )
    outer_test = _validate_index_record(
        split.get("outer_test"), label=f"{protocol}.{fold}.outer_test", cells=cells
    )
    inner_train = _validate_index_record(
        split.get("inner_train"), label=f"{protocol}.{fold}.inner_train", cells=cells
    )
    inner_validation = _validate_index_record(
        split.get("inner_validation"),
        label=f"{protocol}.{fold}.inner_validation",
        cells=cells,
    )
    if not np.array_equal(outer_train, expected_outer_train) or not np.array_equal(
        outer_test, expected_outer_test
    ):
        raise VerificationError(f"{protocol}.{fold} outer partition is noncanonical")
    expected_inner = _make_inner_split(
        expected_outer_train,
        lineages,
        seed=_derive_seed(fold_seed, "inner-split"),
    )
    if not np.array_equal(inner_train, expected_inner[0]) or not np.array_equal(
        inner_validation, expected_inner[1]
    ):
        raise VerificationError(f"{protocol}.{fold} inner partition is noncanonical")
    source = split.get("source_cell_indices")
    if not isinstance(source, Mapping):
        raise VerificationError(f"{protocol}.{fold} source mapping is missing")
    _require_exact_keys(
        source,
        {"outer_train", "inner_train", "inner_validation", "outer_test"},
        label=f"{protocol}.{fold}.source_cell_indices",
    )
    partitions = {
        "outer_train": outer_train,
        "inner_train": inner_train,
        "inner_validation": inner_validation,
        "outer_test": outer_test,
    }
    for name, indices in partitions.items():
        if source.get(name) != source_cells[indices].tolist():
            raise VerificationError(
                f"{protocol}.{fold} {name} source mapping is noncanonical"
            )
    if split.get("lineage_counts") != {
        "outer_train": _lineage_counts(lineages, outer_train),
        "outer_test": _lineage_counts(lineages, outer_test),
    }:
        raise VerificationError(f"{protocol}.{fold} lineage counts do not recompute")
    return outer_train, outer_test, inner_train, inner_validation


def _validate_prediction_artifact(
    payload: bytes,
    *,
    label: str,
    prediction_record: Mapping[str, Any],
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    cells: Sequence[str],
    genes: Sequence[str],
    source_cells: np.ndarray,
    source_genes: np.ndarray,
    gene_effect: np.ndarray,
) -> dict[str, np.ndarray]:
    arrays = _load_npz(payload, label=label, names=EXPECTED_PREDICTION_ARRAYS)
    expected_shapes = {
        "prediction": (len(outer_test), len(genes)),
        "train_gene_mean": (len(genes),),
        "outer_test_indices": (len(outer_test),),
        "source_cell_indices": (len(outer_test),),
        "source_gene_indices": (len(genes),),
        "cell_ids": (len(outer_test),),
        "gene_ids": (len(genes),),
    }
    expected_dtypes = {
        "prediction": np.dtype("float32"),
        "train_gene_mean": np.dtype("float32"),
        "outer_test_indices": np.dtype("int64"),
        "source_cell_indices": np.dtype("int64"),
        "source_gene_indices": np.dtype("int64"),
    }
    for name, array in arrays.items():
        if array.shape != expected_shapes[name] or not array.flags.c_contiguous:
            raise VerificationError(f"{label}.{name} has an invalid shape/layout")
        if name in expected_dtypes and array.dtype != expected_dtypes[name]:
            raise VerificationError(f"{label}.{name} has an invalid dtype")
        if name in {"cell_ids", "gene_ids"} and array.dtype.kind != "U":
            raise VerificationError(f"{label}.{name} must be a Unicode array")
    if (
        not np.isfinite(arrays["prediction"]).all()
        or not np.isfinite(arrays["train_gene_mean"]).all()
    ):
        raise VerificationError(f"{label} contains non-finite predictions/baseline")
    expected_values: dict[str, np.ndarray] = {
        "outer_test_indices": outer_test,
        "source_cell_indices": source_cells[outer_test],
        "source_gene_indices": source_genes,
        "cell_ids": np.asarray([cells[int(index)] for index in outer_test], dtype=str),
        "gene_ids": np.asarray(genes, dtype=str),
        "train_gene_mean": _fit_gene_mean_baseline(gene_effect, outer_train),
    }
    for name, expected in expected_values.items():
        if not np.array_equal(arrays[name], expected):
            raise VerificationError(f"{label}.{name} differs from its bound partition")
    metadata = prediction_record.get("arrays")
    expected_metadata = {
        name: {"dtype": str(array.dtype), "shape": list(array.shape)}
        for name, array in arrays.items()
    }
    if metadata != expected_metadata:
        raise VerificationError(f"{label} array metadata does not match the archive")
    return arrays


def _expression_names_sha256(snapshot: BoundInputSnapshot, genes: Sequence[str]) -> str:
    try:
        with _open_bound_input(snapshot, label="expression CSV") as binary_handle:
            with io.TextIOWrapper(
                binary_handle, encoding="utf-8-sig", newline=""
            ) as text_handle:
                header = next(csv.reader(text_handle))
    except (OSError, UnicodeError, StopIteration, csv.Error) as exc:
        raise VerificationError("cannot parse the expression CSV header") from exc
    allowed = set(genes)
    selected: list[str] = []
    seen: set[str] = set()
    for column in header[1:]:
        symbol = column.split(" (")[0]
        if symbol in allowed and symbol not in seen:
            selected.append(symbol)
            seen.add(symbol)
    if len(selected) != 17_572:
        raise VerificationError("expression feature topology is noncanonical")
    return _hash_strings(selected)


def _load_canonical_preprocessing_inputs(
    *,
    expression_snapshot: BoundInputSnapshot,
    mutation_snapshot: BoundInputSnapshot,
    cells: Sequence[str],
    genes: Sequence[str],
    lineages: Sequence[str],
    gene_effect: np.ndarray,
    stored_edge_index: np.ndarray,
) -> CanonicalPreprocessingInputs:
    """Load the hash-bound raw inputs without importing the producer."""
    try:
        with _open_bound_input(
            expression_snapshot, label="expression CSV header"
        ) as handle:
            header = pd.read_csv(handle, nrows=0).columns.tolist()
    except (OSError, ValueError) as exc:
        raise VerificationError("cannot read the canonical expression header") from exc
    if len(header) < 2 or not isinstance(header[0], str):
        raise VerificationError("canonical expression CSV has an invalid header")
    allowed = set(genes)
    selected_columns: list[str] = []
    selected_names: list[str] = []
    seen: set[str] = set()
    for raw_column in header[1:]:
        if not isinstance(raw_column, str):
            raise VerificationError("expression CSV contains a non-string column")
        symbol = raw_column.split(" (")[0]
        if symbol in allowed and symbol not in seen:
            selected_columns.append(raw_column)
            selected_names.append(symbol)
            seen.add(symbol)
    if len(selected_names) != 17_572:
        raise VerificationError("expression feature topology is noncanonical")
    try:
        with _open_bound_input(
            expression_snapshot, label="expression CSV values"
        ) as handle:
            expression_frame = pd.read_csv(
                handle,
                usecols=[header[0], *selected_columns],
                index_col=0,
                dtype={column: np.float32 for column in selected_columns},
            )
        if not expression_frame.index.is_unique:
            raise VerificationError(
                "expression CSV contains duplicate cell identifiers"
            )
        missing_cells = [cell for cell in cells if cell not in expression_frame.index]
        if missing_cells:
            raise VerificationError(
                f"expression CSV is missing {len(missing_cells)} canonical cells"
            )
        expression = expression_frame.reindex(cells).to_numpy(
            dtype=np.float32, copy=True
        )
    except VerificationError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise VerificationError("cannot load canonical expression values") from exc
    if expression.shape != (len(cells), len(selected_names)):
        raise VerificationError("canonical expression matrix has an invalid shape")

    wanted = set(cells)
    mutations: defaultdict[str, set[str]] = defaultdict(set)
    try:
        with _open_bound_input(mutation_snapshot, label="mutation CSV") as handle:
            chunks = pd.read_csv(
                handle,
                usecols=["HugoSymbol", "ModelID"],
                dtype=str,
                chunksize=250_000,
            )
            for chunk in chunks:
                selected = chunk[chunk["ModelID"].isin(wanted)].dropna()
                for model_id, symbol in zip(
                    selected["ModelID"], selected["HugoSymbol"]
                ):
                    if symbol:
                        mutations[str(model_id)].add(str(symbol))
    except (OSError, ValueError, KeyError) as exc:
        raise VerificationError("cannot load canonical mutation calls") from exc

    edges = np.asarray(stored_edge_index, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise VerificationError("stored PPI edge_index must have shape (E, 2)")
    if edges.size and (int(edges.min()) < 0 or int(edges.max()) >= len(genes)):
        raise VerificationError("stored PPI edge_index refers to an unknown gene")
    undirected_edges = edges.copy()
    undirected_edges.sort(axis=1)
    undirected_edges = undirected_edges[
        undirected_edges[:, 0] != undirected_edges[:, 1]
    ]
    undirected_edges = np.ascontiguousarray(
        np.unique(undirected_edges, axis=0), dtype=np.int64
    )
    if undirected_edges.shape != (209_315, 2):
        raise VerificationError("canonical undirected PPI topology is noncanonical")

    lineage_array = np.asarray(lineages, dtype=str)
    if lineage_array.shape != (len(cells),):
        raise VerificationError("canonical lineage vector has an invalid shape")
    return CanonicalPreprocessingInputs(
        expression=np.ascontiguousarray(expression, dtype=np.float32),
        expression_names=tuple(selected_names),
        mutations_by_cell=tuple(frozenset(mutations[cell]) for cell in cells),
        edge_index=undirected_edges,
        genes=tuple(genes),
        lineages=lineage_array,
        gene_effect=np.asarray(gene_effect, dtype=np.float32),
    )


def _as_replay_indices(values: np.ndarray, *, n_cells: int, label: str) -> np.ndarray:
    indices = np.asarray(values, dtype=np.int64)
    if (
        indices.ndim != 1
        or indices.size == 0
        or np.unique(indices).size != indices.size
        or int(indices.min()) < 0
        or int(indices.max()) >= n_cells
    ):
        raise VerificationError(f"{label} contains invalid replay indices")
    return indices


def _recompute_preprocessing_manifest(
    inputs: CanonicalPreprocessingInputs,
    *,
    fit_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    fold_seed: int,
) -> dict[str, Any]:
    """Independently replay every value stored in a preprocessing manifest."""
    fit = _as_replay_indices(
        fit_indices, n_cells=inputs.expression.shape[0], label="fit_indices"
    )
    evaluation = _as_replay_indices(
        evaluation_indices,
        n_cells=inputs.expression.shape[0],
        label="evaluation_indices",
    )

    expression = np.asarray(inputs.expression, dtype=np.float64)
    train_expression = expression[fit]
    with np.errstate(invalid="ignore"):
        expression_mean = np.nanmean(train_expression, axis=0)
        expression_scale = np.nanstd(train_expression, axis=0, ddof=0)
    expression_mean = np.where(np.isfinite(expression_mean), expression_mean, 0.0)
    expression_scale = np.where(
        np.isfinite(expression_scale) & (expression_scale > 0.0),
        expression_scale,
        1.0,
    )
    clean_expression = np.where(
        np.isfinite(train_expression), train_expression, expression_mean
    )
    standardized_expression = (clean_expression - expression_mean) / expression_scale
    fitted_components = min(
        EXPECTED_ARGUMENTS["n_expr_pc"],
        standardized_expression.shape[0],
        standardized_expression.shape[1],
    )
    try:
        pca = PCA(
            n_components=fitted_components,
            svd_solver="full",
            random_state=_derive_seed(fold_seed, "pca"),
        ).fit(standardized_expression)
    except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
        raise VerificationError("cannot replay expression PCA") from exc
    components = pca.components_.astype(np.float64, copy=False)

    prevalence: Counter[str] = Counter()
    for cell_index in fit:
        prevalence.update(inputs.mutations_by_cell[int(cell_index)])
    ranked = sorted(prevalence, key=lambda gene: (-prevalence[gene], gene))
    top_mut = int(EXPECTED_ARGUMENTS["top_mut"])
    vocabulary = tuple(ranked[:top_mut])
    vocabulary_index = {gene: index for index, gene in enumerate(vocabulary)}
    mutation_matrix: np.ndarray = np.zeros(
        (len(inputs.mutations_by_cell), len(vocabulary)), dtype=np.float32
    )
    for cell_index, cell_mutations in enumerate(inputs.mutations_by_cell):
        columns = [
            vocabulary_index[gene]
            for gene in cell_mutations
            if gene in vocabulary_index
        ]
        if columns:
            mutation_matrix[cell_index, columns] = 1.0
    gene_set = set(inputs.genes)
    token_order = tuple(gene for gene in vocabulary if gene in gene_set)

    endpoint_a = np.fromiter(
        (
            vocabulary_index.get(inputs.genes[int(row)], -1)
            for row in inputs.edge_index[:, 0]
        ),
        dtype=np.int64,
        count=inputs.edge_index.shape[0],
    )
    endpoint_b = np.fromiter(
        (
            vocabulary_index.get(inputs.genes[int(row)], -1)
            for row in inputs.edge_index[:, 1]
        ),
        dtype=np.int64,
        count=inputs.edge_index.shape[0],
    )
    valid_edges = np.flatnonzero((endpoint_a >= 0) & (endpoint_b >= 0))
    edge_weights = np.ones(inputs.edge_index.shape[0], dtype=np.float32)
    train_mutation = mutation_matrix[fit]
    mutation_prevalence = train_mutation.mean(axis=0, dtype=np.float64)
    for start in range(0, valid_edges.size, 20_000):
        positions = valid_edges[start : start + 20_000]
        a = endpoint_a[positions]
        b = endpoint_b[positions]
        joint = np.mean(
            train_mutation[:, a] * train_mutation[:, b],
            axis=0,
            dtype=np.float64,
        )
        denominator = mutation_prevalence[a] * mutation_prevalence[b]
        pmi = np.log((joint + 1e-3) / (denominator + 1e-3))
        edge_weights[positions] = np.clip(pmi, 0.0, 3.0).astype(np.float32)

    categories = tuple(sorted(set(inputs.lineages[fit].tolist())))
    if not categories:
        raise VerificationError("preprocessing replay found no lineage categories")
    category_set = set(categories)
    unknown_count = sum(
        str(inputs.lineages[int(index)]) not in category_set for index in evaluation
    )

    train_effect = inputs.gene_effect[:, fit]
    median = np.nanmedian(train_effect, axis=1, keepdims=True)
    mad = 1.4826 * np.nanmedian(np.abs(train_effect - median), axis=1)
    finite_mad = np.isfinite(mad)
    if finite_mad.any():
        threshold = np.nanquantile(mad[finite_mad], 0.75)
        selectivity = (mad >= threshold).astype(np.float32)
    else:
        selectivity = np.zeros(inputs.gene_effect.shape[0], dtype=np.float32)
    selectivity[~finite_mad] = 0.0

    edge_weight_mean = float(torch.from_numpy(edge_weights.copy()).mean().item())
    return {
        "fit_indices_sha256": _hash_array(fit.astype("<i8", copy=False)),
        "n_fit_cells": int(fit.size),
        "expression": {
            "input_features": int(expression_mean.size),
            "output_components": EXPECTED_ARGUMENTS["n_expr_pc"],
            "fitted_components": int(components.shape[0]),
            "mean_sha256": _hash_array(expression_mean),
            "scale_sha256": _hash_array(expression_scale),
            "components_sha256": _hash_array(components),
        },
        "mutation": {
            "vocabulary": list(vocabulary),
            "vocabulary_sha256": _hash_strings(vocabulary),
            "token_order_sha256": _hash_strings(token_order),
            "n_vocabulary": len(vocabulary),
            "n_token_genes": len(token_order),
            "k_tokens": 64,
        },
        "graph": {
            "n_edges": int(inputs.edge_index.shape[0]),
            "edge_weight_sha256": _hash_array(edge_weights),
            "edge_weight_mean": edge_weight_mean,
            "n_non_neutral_weights": int((~np.isclose(edge_weights, 1.0)).sum()),
        },
        "lineage": {
            "categories": list(categories),
            "n_unknown_evaluation_cells": int(unknown_count),
            "unknown_encoding": "all_zero",
        },
        "selectivity_labels_sha256": _hash_array(selectivity),
    }


def _cached_preprocessing_manifest(
    cache: dict[tuple[str, str, int], dict[str, Any]],
    inputs: CanonicalPreprocessingInputs,
    *,
    fit_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    fold_seed: int,
) -> dict[str, Any]:
    fit_hash = _hash_array(np.asarray(fit_indices, dtype="<i8"))
    evaluation_hash = _hash_array(np.asarray(evaluation_indices, dtype="<i8"))
    pca_seed = _derive_seed(fold_seed, "pca")
    key = (fit_hash, evaluation_hash, pca_seed)
    cached = cache.get(key)
    if cached is None:
        cached = _recompute_preprocessing_manifest(
            inputs,
            fit_indices=fit_indices,
            evaluation_indices=evaluation_indices,
            fold_seed=fold_seed,
        )
        cache[key] = cached
    return cached


def _require_manifest_integer(
    value: Any, *, label: str, minimum: int = 0, maximum: int | None = None
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise VerificationError(f"{label} is not a valid integer")
    return value


def _require_manifest_digest(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise VerificationError(f"{label} is not a SHA-256 digest")
    return value


def _validate_preprocessing_manifest(
    observed: Any, *, expected: Mapping[str, Any], label: str
) -> None:
    if not isinstance(observed, Mapping):
        raise VerificationError(f"{label} must be an object")
    top_fields = {
        "expression",
        "fit_indices_sha256",
        "graph",
        "lineage",
        "mutation",
        "n_fit_cells",
        "selectivity_labels_sha256",
    }
    _require_exact_keys(observed, top_fields, label=label)
    _require_manifest_digest(
        observed.get("fit_indices_sha256"), label=f"{label}.fit_indices_sha256"
    )
    n_fit = _require_manifest_integer(
        observed.get("n_fit_cells"), label=f"{label}.n_fit_cells", minimum=1
    )
    _require_manifest_digest(
        observed.get("selectivity_labels_sha256"),
        label=f"{label}.selectivity_labels_sha256",
    )

    expression = observed.get("expression")
    if not isinstance(expression, Mapping):
        raise VerificationError(f"{label}.expression must be an object")
    _require_exact_keys(
        expression,
        {
            "input_features",
            "output_components",
            "fitted_components",
            "mean_sha256",
            "scale_sha256",
            "components_sha256",
        },
        label=f"{label}.expression",
    )
    input_features = _require_manifest_integer(
        expression.get("input_features"),
        label=f"{label}.expression.input_features",
        minimum=1,
    )
    output_components = _require_manifest_integer(
        expression.get("output_components"),
        label=f"{label}.expression.output_components",
        minimum=1,
    )
    fitted_components = _require_manifest_integer(
        expression.get("fitted_components"),
        label=f"{label}.expression.fitted_components",
        minimum=1,
        maximum=min(n_fit, input_features, output_components),
    )
    del fitted_components
    for name in ("mean_sha256", "scale_sha256", "components_sha256"):
        _require_manifest_digest(
            expression.get(name), label=f"{label}.expression.{name}"
        )

    mutation = observed.get("mutation")
    if not isinstance(mutation, Mapping):
        raise VerificationError(f"{label}.mutation must be an object")
    _require_exact_keys(
        mutation,
        {
            "vocabulary",
            "vocabulary_sha256",
            "token_order_sha256",
            "n_vocabulary",
            "n_token_genes",
            "k_tokens",
        },
        label=f"{label}.mutation",
    )
    vocabulary = mutation.get("vocabulary")
    if (
        not isinstance(vocabulary, list)
        or any(not isinstance(gene, str) or not gene for gene in vocabulary)
        or len(vocabulary) != len(set(vocabulary))
    ):
        raise VerificationError(f"{label}.mutation.vocabulary is invalid")
    n_vocabulary = _require_manifest_integer(
        mutation.get("n_vocabulary"),
        label=f"{label}.mutation.n_vocabulary",
        maximum=int(EXPECTED_ARGUMENTS["top_mut"]),
    )
    n_token_genes = _require_manifest_integer(
        mutation.get("n_token_genes"),
        label=f"{label}.mutation.n_token_genes",
        maximum=n_vocabulary,
    )
    if len(vocabulary) != n_vocabulary or mutation.get("k_tokens") != 64:
        raise VerificationError(f"{label}.mutation dimensions are invalid")
    del n_token_genes
    for name in ("vocabulary_sha256", "token_order_sha256"):
        _require_manifest_digest(mutation.get(name), label=f"{label}.mutation.{name}")

    graph = observed.get("graph")
    if not isinstance(graph, Mapping):
        raise VerificationError(f"{label}.graph must be an object")
    _require_exact_keys(
        graph,
        {
            "n_edges",
            "edge_weight_sha256",
            "edge_weight_mean",
            "n_non_neutral_weights",
        },
        label=f"{label}.graph",
    )
    n_edges = _require_manifest_integer(
        graph.get("n_edges"), label=f"{label}.graph.n_edges", minimum=1
    )
    _require_manifest_integer(
        graph.get("n_non_neutral_weights"),
        label=f"{label}.graph.n_non_neutral_weights",
        maximum=n_edges,
    )
    _require_manifest_digest(
        graph.get("edge_weight_sha256"), label=f"{label}.graph.edge_weight_sha256"
    )
    edge_mean = graph.get("edge_weight_mean")
    if (
        not isinstance(edge_mean, (int, float))
        or isinstance(edge_mean, bool)
        or not math.isfinite(float(edge_mean))
        or not 0.0 <= float(edge_mean) <= 3.0
    ):
        raise VerificationError(f"{label}.graph.edge_weight_mean is invalid")

    lineage = observed.get("lineage")
    if not isinstance(lineage, Mapping):
        raise VerificationError(f"{label}.lineage must be an object")
    _require_exact_keys(
        lineage,
        {"categories", "n_unknown_evaluation_cells", "unknown_encoding"},
        label=f"{label}.lineage",
    )
    categories = lineage.get("categories")
    if (
        not isinstance(categories, list)
        or not categories
        or any(not isinstance(value, str) or not value for value in categories)
        or categories != sorted(set(categories))
        or lineage.get("unknown_encoding") != "all_zero"
    ):
        raise VerificationError(f"{label}.lineage schema is invalid")
    _require_manifest_integer(
        lineage.get("n_unknown_evaluation_cells"),
        label=f"{label}.lineage.n_unknown_evaluation_cells",
    )

    if dict(observed) != dict(expected):
        raise VerificationError(f"{label} does not independently recompute")


def _expected_tuning_configs(condition: str) -> list[dict[str, Any]]:
    base = CONDITION_BASE_CONFIGS.get(condition)
    if base is None:
        raise VerificationError(f"unknown campaign condition: {condition}")
    return [
        {
            **base,
            "hidden": hidden,
            "lamb_sel": 0.1,
            "lr": learning_rate,
            "wd": 1e-5,
        }
        for hidden in (128, 256)
        for learning_rate in (1e-3, 3e-4)
    ]


def _correlation_by_axis(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    *,
    axis: int,
    min_count: int,
) -> np.ndarray:
    counts = valid.sum(axis=axis)
    x = np.where(valid, prediction, 0.0)
    y = np.where(valid, target, 0.0)
    sum_x = x.sum(axis=axis, dtype=np.float64)
    sum_y = y.sum(axis=axis, dtype=np.float64)
    sum_xx = np.square(x, dtype=np.float64).sum(axis=axis, dtype=np.float64)
    sum_yy = np.square(y, dtype=np.float64).sum(axis=axis, dtype=np.float64)
    sum_xy = np.multiply(x, y, dtype=np.float64).sum(axis=axis, dtype=np.float64)
    safe_counts = np.maximum(counts, 1)
    covariance = sum_xy - sum_x * sum_y / safe_counts
    variance_x = sum_xx - np.square(sum_x) / safe_counts
    variance_y = sum_yy - np.square(sum_y) / safe_counts
    denominator = np.sqrt(np.maximum(variance_x, 0.0) * np.maximum(variance_y, 0.0))
    result = np.full(np.asarray(counts).shape, np.nan, dtype=np.float64)
    eligible = (counts >= int(min_count)) & (denominator > 0.0)
    np.divide(covariance, denominator, out=result, where=eligible)
    return result


def _finite_metric_summary(values: np.ndarray, prefix: str) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_sd": float("nan"),
            f"n_finite_{prefix}": 0,
        }
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_median": float(np.median(finite)),
        f"{prefix}_sd": float(np.std(finite)),
        f"n_finite_{prefix}": int(finite.size),
    }


def _recompute_test_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    train_gene_mean: np.ndarray,
) -> dict[str, Any]:
    """Replay every persisted outer-test metric from bound numeric arrays."""
    predicted = np.asarray(prediction, dtype=np.float32)
    observed = np.asarray(target, dtype=np.float32)
    reference = np.asarray(train_gene_mean, dtype=np.float32)
    if predicted.ndim != 2 or observed.shape != predicted.shape:
        raise VerificationError("prediction and target metric arrays do not align")
    if reference.ndim != 1 or reference.size != predicted.shape[1]:
        raise VerificationError("metric reference has an invalid gene dimension")

    reference_2d = reference[None, :]
    valid = np.isfinite(observed) & np.isfinite(reference_2d)
    n_pairs = int(valid.sum())
    if n_pairs == 0:
        raise VerificationError("outer-test metrics have no evaluable pairs")
    invalid_predictions = valid & ~np.isfinite(predicted)
    if invalid_predictions.any():
        raise VerificationError("outer-test predictions are non-finite on valid pairs")

    error = np.where(valid, predicted - observed, 0.0)
    baseline_error = np.where(valid, reference_2d - observed, 0.0)
    squared_error = np.square(error, dtype=np.float64)
    baseline_squared_error = np.square(baseline_error, dtype=np.float64)
    absolute_error = np.abs(error).astype(np.float64, copy=False)
    baseline_absolute_error = np.abs(baseline_error).astype(np.float64, copy=False)
    sse = float(squared_error.sum(dtype=np.float64))
    baseline_sse = float(baseline_squared_error.sum(dtype=np.float64))

    target_residual = observed - reference_2d
    prediction_residual = predicted - reference_2d
    pooled_corr = _correlation_by_axis(
        prediction_residual.reshape(1, -1),
        target_residual.reshape(1, -1),
        valid.reshape(1, -1),
        axis=1,
        min_count=3,
    )[0]
    residual_cell_corr = _correlation_by_axis(
        prediction_residual,
        target_residual,
        valid,
        axis=1,
        min_count=3,
    )
    cell_corr = _correlation_by_axis(predicted, observed, valid, axis=1, min_count=3)
    gene_corr = _correlation_by_axis(predicted, observed, valid, axis=0, min_count=3)

    gene_counts = valid.sum(axis=0)
    gene_mse = np.full(predicted.shape[1], np.nan, dtype=np.float64)
    gene_mae = np.full(predicted.shape[1], np.nan, dtype=np.float64)
    np.divide(
        squared_error.sum(axis=0, dtype=np.float64),
        gene_counts,
        out=gene_mse,
        where=gene_counts > 0,
    )
    np.divide(
        absolute_error.sum(axis=0, dtype=np.float64),
        gene_counts,
        out=gene_mae,
        where=gene_counts > 0,
    )
    gene_baseline_sse = baseline_squared_error.sum(axis=0, dtype=np.float64)
    gene_model_sse = squared_error.sum(axis=0, dtype=np.float64)
    gene_r2 = np.full(predicted.shape[1], np.nan, dtype=np.float64)
    np.divide(
        gene_model_sse,
        gene_baseline_sse,
        out=gene_r2,
        where=gene_baseline_sse > 0.0,
    )
    gene_r2 = 1.0 - gene_r2

    metrics: dict[str, Any] = {
        "mse": float(sse / n_pairs),
        "mae": float(absolute_error.sum(dtype=np.float64) / n_pairs),
        "gene_mean_baseline_mse": float(baseline_sse / n_pairs),
        "gene_mean_baseline_mae": float(
            baseline_absolute_error.sum(dtype=np.float64) / n_pairs
        ),
        "residual_r2_vs_train_gene_mean": (
            float(1.0 - sse / baseline_sse) if baseline_sse > 0.0 else float("nan")
        ),
        "residual_pearson_pooled": float(pooled_corr),
        "n_finite_pairs": n_pairs,
        "n_reference_genes": int(np.isfinite(reference).sum()),
    }
    metrics.update(
        _finite_metric_summary(residual_cell_corr, "residual_cellwise_pearson")
    )
    metrics.update(_finite_metric_summary(cell_corr, "cellwise_pearson"))
    metrics.update(_finite_metric_summary(gene_corr, "gene_wise_pearson"))
    metrics.update(_finite_metric_summary(gene_mse, "gene_wise_mse"))
    metrics.update(_finite_metric_summary(gene_mae, "gene_wise_mae"))
    metrics.update(_finite_metric_summary(gene_r2, "gene_wise_residual_r2"))
    metrics["n_finite_cell_correlations"] = int(np.isfinite(cell_corr).sum())
    metrics["n_finite_gene_correlations"] = int(np.isfinite(gene_corr).sum())
    finite_gene_r2 = gene_r2[np.isfinite(gene_r2)]
    metrics["fraction_genes_positive_residual_r2"] = (
        float(np.mean(finite_gene_r2 > 0.0)) if finite_gene_r2.size else float("nan")
    )

    spearman: list[float] = []
    for row in range(predicted.shape[0]):
        mask = np.isfinite(predicted[row]) & np.isfinite(observed[row])
        if int(mask.sum()) < 10:
            spearman.append(float("nan"))
            continue
        ranked_prediction = stats.rankdata(predicted[row, mask])
        ranked_target = stats.rankdata(observed[row, mask])
        if np.std(ranked_prediction) <= 0.0 or np.std(ranked_target) <= 0.0:
            spearman.append(float("nan"))
        else:
            spearman.append(float(np.corrcoef(ranked_prediction, ranked_target)[0, 1]))
    spearman_array = np.asarray(spearman, dtype=np.float64)
    finite_spearman = spearman_array[np.isfinite(spearman_array)]
    metrics["cellwise_spearman_mean"] = (
        float(np.mean(finite_spearman)) if finite_spearman.size else float("nan")
    )
    metrics["cellwise_spearman_sd"] = (
        float(np.std(finite_spearman)) if finite_spearman.size else float("nan")
    )
    metrics["n_finite_cellwise_spearman"] = int(finite_spearman.size)
    metrics.update(
        {
            "n_cells": int(predicted.shape[0]),
            "reference_scope": "outer_train_gene_mean",
            "gene_mean_reference_sha256": _hash_array(reference),
            "raw_cellwise_correlation_role": "secondary_descriptive_only",
        }
    )
    if set(metrics) != TEST_METRIC_FIELDS:
        raise VerificationError("independent metric replay has an invalid inventory")
    return metrics


def _compare_recomputed_metrics(
    observed: Mapping[str, Any],
    recomputed: Mapping[str, Any],
    *,
    label: str,
) -> None:
    _require_exact_keys(observed, TEST_METRIC_FIELDS, label=label)
    for field in sorted(TEST_METRIC_FIELDS):
        actual = observed[field]
        expected = recomputed[field]
        if isinstance(expected, (str, int)) and not isinstance(expected, bool):
            if type(actual) is not type(expected) or actual != expected:
                raise VerificationError(f"{label}.{field} does not recompute")
            continue
        if (
            not isinstance(actual, (int, float))
            or isinstance(actual, bool)
            or not math.isfinite(float(actual))
        ):
            raise VerificationError(f"{label}.{field} does not recompute")
        _validate_metric_semantics(field, float(actual), label=f"{label}.{field}")
        if not math.isclose(
            float(actual),
            float(expected),
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise VerificationError(f"{label}.{field} does not recompute")


def _validate_metric_semantics(name: str, value: float, *, label: str) -> None:
    if name in NONNEGATIVE_METRIC_FIELDS and value < 0.0:
        raise VerificationError(f"{label} must be nonnegative")
    if name in CORRELATION_METRIC_FIELDS and not -1.0 <= value <= 1.0:
        raise VerificationError(f"{label} must be within [-1, 1]")
    if name in CORRELATION_SD_FIELDS and not 0.0 <= value <= 1.0:
        raise VerificationError(f"{label} must be within [0, 1]")
    if name in R2_METRIC_FIELDS and value > 1.0:
        raise VerificationError(f"{label} must not exceed 1")
    if name in FRACTION_METRIC_FIELDS and not 0.0 <= value <= 1.0:
        raise VerificationError(f"{label} must be within [0, 1]")


def _validate_metric_record(
    metrics: Any,
    *,
    fields: set[str],
    reference_scope: str,
    reference_sha256: str,
    n_cells: int,
    n_genes: int,
    label: str,
) -> None:
    if not isinstance(metrics, Mapping):
        raise VerificationError(f"{label} must be an object")
    _require_exact_keys(metrics, fields, label=label)
    if (
        metrics.get("reference_scope") != reference_scope
        or metrics.get("raw_cellwise_correlation_role") != "secondary_descriptive_only"
        or metrics.get("n_cells") != n_cells
        or metrics.get("n_reference_genes") != n_genes
    ):
        raise VerificationError(f"{label} scope metadata is invalid")
    digest = metrics.get("gene_mean_reference_sha256")
    if (
        not isinstance(digest, str)
        or SHA256_PATTERN.fullmatch(digest) is None
        or digest != reference_sha256
    ):
        raise VerificationError(f"{label} has an invalid reference digest")
    count_fields = {name for name in fields if name.startswith("n_")}
    for name in count_fields:
        value = metrics.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise VerificationError(f"{label}.{name} is invalid")
    required_positive_counts = {
        "n_cells",
        "n_reference_genes",
        *{name for name in count_fields if name.startswith("n_finite_")},
    }
    for name in fields & required_positive_counts:
        if metrics[name] <= 0:
            raise VerificationError(
                f"{label}.{name} must be positive for its finite aggregates"
            )
    if metrics["n_finite_pairs"] > n_cells * n_genes:
        raise VerificationError(f"{label}.n_finite_pairs exceeds the matrix size")
    cell_count_fields = {
        "n_finite_cell_correlations",
        "n_finite_cellwise_pearson",
        "n_finite_cellwise_spearman",
        "n_finite_residual_cellwise_pearson",
    }
    gene_count_fields = {
        "n_finite_gene_correlations",
        "n_finite_gene_wise_mae",
        "n_finite_gene_wise_mse",
        "n_finite_gene_wise_pearson",
        "n_finite_gene_wise_residual_r2",
        "n_reference_genes",
    }
    if any(metrics[name] > n_cells for name in fields & cell_count_fields) or any(
        metrics[name] > n_genes for name in fields & gene_count_fields
    ):
        raise VerificationError(f"{label} finite-count metadata exceeds its axis")
    exact_count_aliases = (
        ("n_finite_cell_correlations", "n_finite_cellwise_pearson"),
        ("n_finite_gene_correlations", "n_finite_gene_wise_pearson"),
        ("n_finite_gene_wise_mse", "n_finite_gene_wise_mae"),
    )
    for left, right in exact_count_aliases:
        if left in fields and right in fields and metrics[left] != metrics[right]:
            raise VerificationError(f"{label}.{left} must equal {label}.{right}")
    minimum_pairs_per_aggregate = (
        ("n_finite_cellwise_pearson", 3),
        ("n_finite_residual_cellwise_pearson", 3),
        ("n_finite_gene_wise_pearson", 3),
        ("n_finite_cellwise_spearman", 10),
    )
    for aggregate_count, minimum_pairs in minimum_pairs_per_aggregate:
        if (
            aggregate_count in fields
            and metrics["n_finite_pairs"] < minimum_pairs * metrics[aggregate_count]
        ):
            raise VerificationError(
                f"{label}.n_finite_pairs does not support " f"{label}.{aggregate_count}"
            )
    for aggregate_count in (
        "n_finite_gene_wise_mse",
        "n_finite_gene_wise_mae",
    ):
        if (
            aggregate_count in fields
            and metrics[aggregate_count] > metrics["n_finite_pairs"]
        ):
            raise VerificationError(
                f"{label}.{aggregate_count} exceeds finite-pair support"
            )
    if (
        "n_finite_gene_wise_pearson" in fields
        and "n_finite_gene_wise_mse" in fields
        and metrics["n_finite_gene_wise_pearson"] > metrics["n_finite_gene_wise_mse"]
    ):
        raise VerificationError(
            f"{label}.n_finite_gene_wise_pearson exceeds evaluable genes"
        )
    bounded_gene_counts = {
        "n_finite_gene_correlations",
        "n_finite_gene_wise_mae",
        "n_finite_gene_wise_mse",
        "n_finite_gene_wise_pearson",
        "n_finite_gene_wise_residual_r2",
    }
    if any(
        metrics[name] > metrics["n_reference_genes"]
        for name in fields & bounded_gene_counts
    ):
        raise VerificationError(
            f"{label} finite gene counts exceed the finite reference count"
        )
    if (
        "n_finite_gene_wise_residual_r2" in fields
        and "n_finite_gene_wise_mse" in fields
        and metrics["n_finite_gene_wise_residual_r2"]
        > metrics["n_finite_gene_wise_mse"]
    ):
        raise VerificationError(f"{label} residual-R2 support exceeds evaluable genes")
    for name in fields.difference(
        count_fields
        | {
            "gene_mean_reference_sha256",
            "raw_cellwise_correlation_role",
            "reference_scope",
        }
    ):
        value = metrics.get(name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise VerificationError(f"{label}.{name} is not finite numeric evidence")
        _validate_metric_semantics(name, float(value), label=f"{label}.{name}")


def _validate_result_selection_contract(
    result: Mapping[str, Any],
    *,
    condition: str,
    fold_seed: int,
    inner_validation: np.ndarray,
    inner_reference_sha256: str,
    n_genes: int,
    label: str,
) -> None:
    best_epoch = result.get("best_epoch")
    wall_seconds = result.get("wall_seconds")
    if (
        not isinstance(best_epoch, int)
        or isinstance(best_epoch, bool)
        or not 1 <= best_epoch <= EXPECTED_ARGUMENTS["inner_max_epochs"]
        or not isinstance(wall_seconds, (int, float))
        or isinstance(wall_seconds, bool)
        or not math.isfinite(float(wall_seconds))
        or float(wall_seconds) <= 0
    ):
        raise VerificationError(f"{label} epoch/timing metadata is invalid")

    expected_configs = _expected_tuning_configs(condition)
    if result.get("best_config") not in expected_configs:
        raise VerificationError(f"{label}.best_config is outside the frozen grid")
    optimization = result.get("optimization")
    if not isinstance(optimization, Mapping):
        raise VerificationError(f"{label}.optimization must be an object")
    _require_exact_keys(
        optimization,
        {
            "refit_epochs",
            "scheduler",
            "scheduler_horizon_epochs",
            "selection_epoch_budget",
        },
        label=f"{label}.optimization",
    )
    if dict(optimization) != {
        "refit_epochs": best_epoch,
        "scheduler": "CosineAnnealingLR",
        "scheduler_horizon_epochs": EXPECTED_ARGUMENTS["inner_max_epochs"],
        "selection_epoch_budget": EXPECTED_ARGUMENTS["inner_max_epochs"],
    }:
        raise VerificationError(f"{label}.optimization is noncanonical")

    selection = result.get("selection")
    if not isinstance(selection, Mapping):
        raise VerificationError(f"{label}.selection must be an object")
    _require_exact_keys(
        selection, {"maximize", "metric", "score"}, label=f"{label}.selection"
    )
    selection_score = selection.get("score")
    if (
        selection.get("maximize") is not True
        or selection.get("metric") != "residual_r2_vs_train_gene_mean"
        or not isinstance(selection_score, (int, float))
        or isinstance(selection_score, bool)
        or not math.isfinite(float(selection_score))
    ):
        raise VerificationError(f"{label}.selection is noncanonical")
    _validate_metric_semantics(
        "residual_r2_vs_train_gene_mean",
        float(selection_score),
        label=f"{label}.selection.score",
    )

    tuning = result.get("tuning")
    if not isinstance(tuning, list) or len(tuning) != len(expected_configs):
        raise VerificationError(f"{label}.tuning does not cover the frozen grid")
    tune_seed = _derive_seed(fold_seed, "tune")
    observed_configs: list[dict[str, Any]] = []
    recomputed_outcomes: list[tuple[float, int]] = []
    for tuning_index, record in enumerate(tuning):
        tuning_label = f"{label}.tuning[{tuning_index}]"
        if not isinstance(record, Mapping):
            raise VerificationError(f"{tuning_label} must be an object")
        _require_exact_keys(record, TUNING_RESULT_FIELDS, label=tuning_label)
        config = record.get("config")
        if not isinstance(config, dict):
            raise VerificationError(f"{tuning_label}.config must be an object")
        observed_configs.append(config)
        record_epoch = record.get("best_epoch")
        trained_epochs = record.get("trained_epochs")
        score = record.get("selection_score")
        if (
            record.get("seed") != tune_seed
            or record.get("scheduler") != "CosineAnnealingLR"
            or record.get("scheduler_horizon_epochs")
            != EXPECTED_ARGUMENTS["inner_max_epochs"]
            or record.get("selection_metric") != "residual_r2_vs_train_gene_mean"
            or not isinstance(record_epoch, int)
            or isinstance(record_epoch, bool)
            or not isinstance(trained_epochs, int)
            or isinstance(trained_epochs, bool)
            or not 1 <= record_epoch <= trained_epochs
            or trained_epochs > EXPECTED_ARGUMENTS["inner_max_epochs"]
            or not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(float(score))
        ):
            raise VerificationError(f"{tuning_label} metadata is invalid")
        _validate_metric_semantics(
            "residual_r2_vs_train_gene_mean",
            float(score),
            label=f"{tuning_label}.selection_score",
        )
        history = record.get("history")
        if not isinstance(history, list) or len(history) != trained_epochs:
            raise VerificationError(f"{tuning_label}.history has invalid length")
        history_best_score = -math.inf
        history_best_epoch = 0
        history_best_metrics: Mapping[str, Any] | None = None
        bad_epochs = 0
        for history_index, epoch_record in enumerate(history):
            history_label = f"{tuning_label}.history[{history_index}]"
            if not isinstance(epoch_record, Mapping):
                raise VerificationError(f"{history_label} must be an object")
            _require_exact_keys(
                epoch_record,
                {"epoch", "train_loss", "validation"},
                label=history_label,
            )
            train_loss = epoch_record.get("train_loss")
            if (
                epoch_record.get("epoch") != history_index + 1
                or not isinstance(train_loss, (int, float))
                or isinstance(train_loss, bool)
                or not math.isfinite(float(train_loss))
                or float(train_loss) < 0.0
            ):
                raise VerificationError(f"{history_label} metadata is invalid")
            validation_metrics = epoch_record.get("validation")
            _validate_metric_record(
                validation_metrics,
                fields=VALIDATION_METRIC_FIELDS,
                reference_scope="inner_train_gene_mean",
                reference_sha256=inner_reference_sha256,
                n_cells=len(inner_validation),
                n_genes=n_genes,
                label=f"{history_label}.validation",
            )
            assert isinstance(validation_metrics, Mapping)
            validation_score = float(
                validation_metrics["residual_r2_vs_train_gene_mean"]
            )
            if validation_score > history_best_score + 1e-8:
                history_best_score = validation_score
                history_best_epoch = history_index + 1
                history_best_metrics = validation_metrics
                bad_epochs = 0
            else:
                bad_epochs += 1
            if bad_epochs >= EXPECTED_ARGUMENTS["patience"] and (
                history_index + 1 != trained_epochs
            ):
                raise VerificationError(
                    f"{tuning_label}.history continues after early stopping"
                )
        if (
            trained_epochs < EXPECTED_ARGUMENTS["inner_max_epochs"]
            and bad_epochs < EXPECTED_ARGUMENTS["patience"]
        ):
            raise VerificationError(
                f"{tuning_label}.history ends before early stopping or budget"
            )
        best_metrics = record.get("best_validation_metrics")
        _validate_metric_record(
            best_metrics,
            fields=VALIDATION_METRIC_FIELDS,
            reference_scope="inner_train_gene_mean",
            reference_sha256=inner_reference_sha256,
            n_cells=len(inner_validation),
            n_genes=n_genes,
            label=f"{tuning_label}.best_validation_metrics",
        )
        if (
            history_best_metrics is None
            or record_epoch != history_best_epoch
            or best_metrics != history_best_metrics
            or not math.isclose(
                float(score),
                history_best_score,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise VerificationError(
                f"{tuning_label} best epoch/score does not replay from the "
                "hash-bound producer history"
            )
        recomputed_outcomes.append((history_best_score, history_best_epoch))
    if observed_configs != expected_configs:
        raise VerificationError(f"{label}.tuning grid/order is noncanonical")
    selected_index = 0
    selected_score = recomputed_outcomes[0][0]
    for candidate_index in range(1, len(recomputed_outcomes)):
        candidate_score = recomputed_outcomes[candidate_index][0]
        if candidate_score > selected_score:
            selected_index = candidate_index
            selected_score = candidate_score
    selected = tuning[selected_index]
    if (
        result.get("best_config") != selected["config"]
        or best_epoch != recomputed_outcomes[selected_index][1]
        or not math.isclose(
            float(selection_score),
            selected_score,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise VerificationError(
            f"{label} selected-model argmax does not replay from the hash-bound "
            "producer history"
        )


def _validate_result_contract(
    result: Mapping[str, Any],
    *,
    condition: str,
    fold_seed: int,
    inner_train: np.ndarray,
    inner_validation: np.ndarray,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    inner_reference_sha256: str,
    outer_reference_sha256: str,
    expected_preprocessing: Mapping[str, Mapping[str, Any]],
    n_genes: int,
    label: str,
) -> None:
    _validate_result_selection_contract(
        result,
        condition=condition,
        fold_seed=fold_seed,
        inner_validation=inner_validation,
        inner_reference_sha256=inner_reference_sha256,
        n_genes=n_genes,
        label=label,
    )

    preprocessing = result.get("preprocessing")
    if not isinstance(preprocessing, Mapping):
        raise VerificationError(f"{label}.preprocessing must be an object")
    _require_exact_keys(
        preprocessing,
        {"inner_train_fit", "outer_train_refit"},
        label=f"{label}.preprocessing",
    )
    _require_exact_keys(
        expected_preprocessing,
        {"inner_train_fit", "outer_train_refit"},
        label=f"{label}.expected_preprocessing",
    )
    for name in ("inner_train_fit", "outer_train_refit"):
        _validate_preprocessing_manifest(
            preprocessing.get(name),
            expected=expected_preprocessing[name],
            label=f"{label}.preprocessing.{name}",
        )

    _validate_metric_record(
        result.get("test_metrics"),
        fields=TEST_METRIC_FIELDS,
        reference_scope="outer_train_gene_mean",
        reference_sha256=outer_reference_sha256,
        n_cells=len(outer_test),
        n_genes=n_genes,
        label=f"{label}.test_metrics",
    )


def _verify_protocol_artifacts(
    runs: Mapping[str, Mapping[str, Any]],
    run_paths: Mapping[str, Path],
    *,
    declared_workspace_records: Mapping[str, Mapping[str, Any]],
    context: VerificationContext,
) -> dict[tuple[str, str], FoldEvidence]:
    required = {
        "data/processed/genes.txt",
        "data/processed/cell_lines.txt",
        "data/processed/lineages.txt",
        "data/processed/gene_effect.npy",
        "data/processed/ppi_graph.npz",
        "data/processed/splits.json",
        "data/depmap/OmicsExpressionProteinCodingGenesTPMLogp1.csv",
        "data/depmap/OmicsSomaticMutations.csv",
    }
    missing = required.difference(declared_workspace_records)
    if missing:
        raise VerificationError(f"campaign lacks canonical inputs: {sorted(missing)}")
    payloads: dict[str, bytes] = {}
    large_csv_inputs = set(RAW_PREPROCESSING_INPUTS)
    for relative in sorted(required - large_csv_inputs):
        payload, _ = _verify_record(
            _workspace_path(relative, label=relative),
            declared_workspace_records[relative],
            label=f"protocol input {relative}",
            context=context,
            keep_bytes=True,
        )
        assert payload is not None
        payloads[relative] = payload
    genes = _load_lines(payloads["data/processed/genes.txt"], label="genes.txt")
    cells = _load_lines(
        payloads["data/processed/cell_lines.txt"], label="cell_lines.txt"
    )
    lineages = _load_lines(
        payloads["data/processed/lineages.txt"],
        label="lineages.txt",
        require_unique=False,
    )
    if (
        len(genes) != EXPECTED_N_GENES
        or len(cells) != EXPECTED_N_CELLS
        or len(lineages) != EXPECTED_N_CELLS
    ):
        raise VerificationError("canonical cohort dimensions are invalid")
    gene_effect = _load_npy(
        payloads["data/processed/gene_effect.npy"], label="gene_effect.npy"
    )
    if gene_effect.dtype != np.float32 or gene_effect.shape != (
        EXPECTED_N_GENES,
        EXPECTED_N_CELLS,
    ):
        raise VerificationError("gene_effect.npy topology is noncanonical")
    ppi = _load_npz(
        payloads["data/processed/ppi_graph.npz"],
        label="ppi_graph.npz",
        names=("edge_index", "edge_weight", "nodes"),
    )
    if (
        ppi["edge_index"].dtype != np.int64
        or ppi["edge_index"].shape != (418_630, 2)
        or ppi["edge_weight"].dtype != np.float32
        or ppi["edge_weight"].shape != (418_630,)
        or not np.isfinite(ppi["edge_weight"]).all()
        or not np.array_equal(ppi["nodes"], np.asarray(genes, dtype=str))
    ):
        raise VerificationError("PPI topology is noncanonical")
    n_string_edges = 209_315
    stored_splits = _parse_json(
        payloads["data/processed/splits.json"], label="processed splits.json"
    )
    complement, bkm = _validate_processed_splits(
        stored_splits, cells=cells, lineages=lineages
    )
    with _verified_raw_input_copies(
        declared_workspace_records, context=context
    ) as raw_input_copies:
        expression_snapshot = raw_input_copies[RAW_PREPROCESSING_INPUTS[0]]
        mutation_snapshot = raw_input_copies[RAW_PREPROCESSING_INPUTS[1]]
        expression_hash = _expression_names_sha256(expression_snapshot, genes)
        preprocessing_inputs = _load_canonical_preprocessing_inputs(
            expression_snapshot=expression_snapshot,
            mutation_snapshot=mutation_snapshot,
            cells=cells,
            genes=genes,
            lineages=lineages,
            gene_effect=gene_effect,
            stored_edge_index=ppi["edge_index"],
        )

    p1_splits = {seed: _make_p1_outer_splits(len(cells), seed) for seed in SEEDS}
    evidence: dict[tuple[str, str], FoldEvidence] = {}
    split_fingerprints: dict[tuple[str, int, str], str] = {}
    preprocessing_cache: dict[tuple[str, str, int], dict[str, Any]] = {}
    reference_hash_cache: dict[str, str] = {}
    cohort_reference: Mapping[str, Any] | None = None
    for key in sorted(runs):
        run = runs[key]
        run_dir = run_paths[key]
        protocol, condition, seed = _parse_run_key(key)
        records = _artifact_map(run)
        summary = _load_run_root_json(
            run_dir,
            "summary.json",
            run["summary_sha256"],
            label=f"{key}.summary.json",
            context=context,
        )
        cohort = _load_run_root_json(
            run_dir,
            "cohort-manifest.json",
            run["cohort_manifest_sha256"],
            label=f"{key}.cohort-manifest.json",
            context=context,
        )
        if cohort_reference is None:
            cohort_reference = cohort
        elif cohort != cohort_reference:
            raise VerificationError("campaign run cohort manifests are inconsistent")
        source_cells, source_genes = _validate_cohort_manifest(
            cohort,
            genes=genes,
            cells=cells,
            expression_names_sha256=expression_hash,
            n_string_edges=n_string_edges,
        )
        expected_fold_names = (
            list(FOLDS)
            if protocol == "p1"
            else [
                (
                    "complement-to-bkm"
                    if protocol == "transfer-forward"
                    else "bkm-to-complement"
                )
            ]
        )
        expected_fold_numbers = list(range(len(expected_fold_names)))
        _require_exact_keys(summary, RUN_SUMMARY_FIELDS, label=f"{key}.summary.json")
        expected_campaign = {
            "arguments": {
                name: EXPECTED_ARGUMENTS[name]
                for name in (
                    "batch_size",
                    "folds",
                    "gene_subsample",
                    "inner_max_epochs",
                    "inner_val_fraction",
                    "n_expr_pc",
                    "patience",
                    "top_mut",
                )
            },
            "seeds": list(SEEDS),
            "tuning_grid": [
                {"hidden": hidden, "lamb_sel": 0.1, "lr": lr, "wd": 1e-5}
                for hidden in (128, 256)
                for lr in (1e-3, 3e-4)
            ],
        }
        if (
            summary.get("schema") != RUN_SCHEMA
            or summary.get("protocol") != protocol
            or summary.get("condition") != condition
            or summary.get("seed") != seed
            or summary.get("pilot") is not False
            or summary.get("run_status") != "protocol_complete"
            or summary.get("protocol_complete") is not True
            or summary.get("eligible_for_scientific_aggregation") is not True
            or summary.get("scientific_result") is not False
            or summary.get("n_completed_folds") != len(expected_fold_names)
            or summary.get("completed_fold_numbers") != expected_fold_numbers
            or summary.get("planned_fold_numbers") != expected_fold_numbers
            or summary.get("fold_files")
            != [f"{name}-result.json" for name in expected_fold_names]
            or summary.get("primary_metric") != "residual_r2_vs_train_gene_mean"
            or summary.get("required_finite_metrics")
            != list(REQUIRED_FINITE_METRIC_ORDER)
            or summary.get("canonical_campaign") != expected_campaign
            or summary.get("outer_test_policy") != OUTER_TEST_POLICY
            or summary.get("selection_policy") != SELECTION_POLICY
            or summary.get("source_stability") != SOURCE_STABILITY
        ):
            raise VerificationError(f"{key} summary protocol contract is invalid")
        summary_metrics = summary.get("metrics")
        if not isinstance(summary_metrics, Mapping):
            raise VerificationError(f"{key} summary metrics are invalid")
        _require_exact_keys(
            summary_metrics, SUMMARY_METRIC_FIELDS, label=f"{key}.summary.metrics"
        )
        for metric_name, metric_record in summary_metrics.items():
            if not isinstance(metric_record, Mapping):
                raise VerificationError(
                    f"{key}.summary.metrics.{metric_name} must be an object"
                )
            _require_exact_keys(
                metric_record,
                {"mean", "n", "sd"},
                label=f"{key}.summary.metrics.{metric_name}",
            )
            if (
                metric_record.get("n") != len(expected_fold_names)
                or not isinstance(metric_record.get("mean"), (int, float))
                or isinstance(metric_record.get("mean"), bool)
                or not math.isfinite(float(metric_record["mean"]))
                or not isinstance(metric_record.get("sd"), (int, float))
                or isinstance(metric_record.get("sd"), bool)
                or not math.isfinite(float(metric_record["sd"]))
            ):
                raise VerificationError(
                    f"{key}.summary.metrics.{metric_name} is invalid"
                )
            _validate_metric_semantics(
                metric_name,
                float(metric_record["mean"]),
                label=f"{key}.summary.metrics.{metric_name}.mean",
            )
            if float(metric_record["sd"]) < 0.0:
                raise VerificationError(
                    f"{key}.summary.metrics.{metric_name}.sd must be nonnegative"
                )
        summary_artifacts = summary.get("artifacts")
        if not isinstance(summary_artifacts, list):
            raise VerificationError(f"{key} summary artifact bindings differ")
        summary_artifact_map: dict[str, dict[str, Any]] = {}
        for artifact_index, item in enumerate(summary_artifacts):
            core = _record_core(
                item, label=f"{key}.summary.artifacts[{artifact_index}]"
            )
            if core["path"] in summary_artifact_map:
                raise VerificationError(f"{key} has a duplicate summary artifact")
            summary_artifact_map[core["path"]] = core
        if summary_artifact_map != records:
            raise VerificationError(f"{key} summary artifact bindings differ")

        fold_test_metrics: list[Mapping[str, Any]] = []
        for fold_number, fold in enumerate(expected_fold_names):
            fold_seed = _derive_seed(seed, fold)
            if protocol == "p1":
                expected_outer_train, expected_outer_test = p1_splits[seed][fold_number]
            elif protocol == "transfer-forward":
                expected_outer_train, expected_outer_test = complement, bkm
            else:
                expected_outer_train, expected_outer_test = bkm, complement
            split_name = f"{fold}-split.json"
            result_name = f"{fold}-result.json"
            prediction_name = f"{fold}-predictions.npz"
            split = _load_run_json(
                run_dir,
                split_name,
                records[split_name],
                label=f"{key}.{split_name}",
                context=context,
            )
            split_fingerprint = _canonical_sha256(split)
            fingerprint_key = (protocol, seed, fold)
            previous_fingerprint = split_fingerprints.get(fingerprint_key)
            if (
                previous_fingerprint is not None
                and previous_fingerprint != split_fingerprint
            ):
                raise VerificationError(
                    f"split differs across conditions for {protocol}, seed {seed}, {fold}"
                )
            split_fingerprints[fingerprint_key] = split_fingerprint
            outer_train, outer_test, inner_train, inner_validation = (
                _validate_split_manifest(
                    split,
                    protocol=protocol,
                    fold=fold,
                    fold_seed=fold_seed,
                    expected_outer_train=expected_outer_train,
                    expected_outer_test=expected_outer_test,
                    cells=cells,
                    lineages=lineages,
                    source_cells=source_cells,
                )
            )
            inner_fit_hash = _hash_array(inner_train.astype("<i8", copy=False))
            outer_fit_hash = _hash_array(outer_train.astype("<i8", copy=False))
            inner_reference_sha256 = reference_hash_cache.get(inner_fit_hash)
            if inner_reference_sha256 is None:
                inner_reference_sha256 = _hash_array(
                    _fit_gene_mean_baseline(gene_effect, inner_train)
                )
                reference_hash_cache[inner_fit_hash] = inner_reference_sha256
            outer_reference_sha256 = reference_hash_cache.get(outer_fit_hash)
            if outer_reference_sha256 is None:
                outer_reference_sha256 = _hash_array(
                    _fit_gene_mean_baseline(gene_effect, outer_train)
                )
                reference_hash_cache[outer_fit_hash] = outer_reference_sha256
            expected_preprocessing = {
                "inner_train_fit": _cached_preprocessing_manifest(
                    preprocessing_cache,
                    preprocessing_inputs,
                    fit_indices=inner_train,
                    evaluation_indices=inner_validation,
                    fold_seed=fold_seed,
                ),
                "outer_train_refit": _cached_preprocessing_manifest(
                    preprocessing_cache,
                    preprocessing_inputs,
                    fit_indices=outer_train,
                    evaluation_indices=outer_test,
                    fold_seed=fold_seed,
                ),
            }
            result = _load_run_json(
                run_dir,
                result_name,
                records[result_name],
                label=f"{key}.{result_name}",
                context=context,
            )
            _require_exact_keys(
                result,
                RESULT_FIELDS,
                label=f"{key}.{result_name}",
            )
            if (
                result.get("condition") != condition
                or result.get("protocol") != protocol
                or result.get("seed") != seed
                or result.get("fold") != fold
                or result.get("fold_number") != fold_number
                or result.get("fold_seed") != fold_seed
                or result.get("pilot") is not False
                or result.get("scientific_result") is not False
                or result.get("result_scope") != "single_outer_fold"
                or result.get("test_evaluation_calls") != 1
            ):
                raise VerificationError(f"{key}.{result_name} identity is invalid")
            prediction_record = result.get("outer_test_predictions")
            if not isinstance(prediction_record, Mapping):
                raise VerificationError(
                    f"{key}.{result_name} prediction record is missing"
                )
            _require_exact_keys(
                prediction_record,
                {"arrays", "bytes", "path", "schema", "sha256"},
                label=f"{key}.{result_name}.prediction",
            )
            if (
                prediction_record.get("schema") != PREDICTION_SCHEMA
                or _record_core(
                    prediction_record, label=f"{key}.{result_name}.prediction"
                )
                != records[prediction_name]
            ):
                raise VerificationError(
                    f"{key}.{result_name} prediction binding differs"
                )
            _validate_result_contract(
                result,
                condition=condition,
                fold_seed=fold_seed,
                inner_train=inner_train,
                inner_validation=inner_validation,
                outer_train=outer_train,
                outer_test=outer_test,
                inner_reference_sha256=inner_reference_sha256,
                outer_reference_sha256=outer_reference_sha256,
                expected_preprocessing=expected_preprocessing,
                n_genes=len(genes),
                label=f"{key}.{result_name}",
            )
            metrics = result.get("test_metrics")
            assert isinstance(metrics, Mapping)
            payload, _ = _verify_record(
                run_dir / prediction_name,
                records[prediction_name],
                label=f"{key}.{prediction_name}",
                context=context,
                keep_bytes=True,
            )
            assert payload is not None
            arrays = _validate_prediction_artifact(
                payload,
                label=f"{key}.{prediction_name}",
                prediction_record=prediction_record,
                outer_train=outer_train,
                outer_test=outer_test,
                cells=cells,
                genes=genes,
                source_cells=source_cells,
                source_genes=source_genes,
                gene_effect=gene_effect,
            )
            recomputed_metrics = _recompute_test_metrics(
                arrays["prediction"],
                gene_effect[:, outer_test].T,
                arrays["train_gene_mean"],
            )
            _compare_recomputed_metrics(
                metrics,
                recomputed_metrics,
                label=f"{key}.{result_name}.test_metrics",
            )
            fold_test_metrics.append(recomputed_metrics)
            evidence[(key, fold)] = FoldEvidence(
                key=key,
                protocol=protocol,
                condition=condition,
                seed=seed,
                fold=fold,
                fold_number=fold_number,
                fold_seed=fold_seed,
                outer_train=outer_train,
                outer_test=outer_test,
                source_cell_indices=source_cells,
                source_gene_indices=source_genes,
                prediction_record=dict(records[prediction_name]),
            )
        for metric_name in sorted(SUMMARY_METRIC_FIELDS):
            values = np.asarray(
                [float(metrics[metric_name]) for metrics in fold_test_metrics],
                dtype=np.float64,
            )
            expected_summary = {
                "mean": float(np.mean(values)),
                "sd": float(np.std(values)),
                "n": int(values.size),
            }
            observed_summary = summary_metrics[metric_name]
            if observed_summary["n"] != expected_summary["n"] or any(
                not math.isclose(
                    float(observed_summary[name]),
                    expected_summary[name],
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                for name in ("mean", "sd")
            ):
                raise VerificationError(
                    f"{key}.summary.metrics.{metric_name} does not recompute"
                )
    expected_fold_records = sum(
        len(FOLDS) if _parse_run_key(key)[0] == "p1" else 1 for key in runs
    )
    if len(evidence) != expected_fold_records:
        raise VerificationError("not every campaign fold has protocol evidence")
    return evidence


def _validate_embedding_bundle(
    manifest: Mapping[str, Any],
    campaign: Mapping[str, Any],
    *,
    manifest_record: Mapping[str, Any],
    analysis_embedding_record: Mapping[str, Any],
    genes: Sequence[str],
    context: VerificationContext,
) -> dict[str, Any]:
    _require_exact_keys(manifest, EMBEDDING_MANIFEST_FIELDS, label="embedding manifest")
    if (
        manifest.get("schema") != EMBEDDING_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
    ):
        raise VerificationError(
            "embedding manifest is not a completed canonical manifest"
        )
    if manifest.get("argv") != [
        "code/03_embed_proteins.py",
        "--strategy",
        "chunked",
        "--batch-size",
        "64",
        "--device",
        "cuda",
        "--local-files-only",
    ]:
        raise VerificationError("embedding generator arguments are noncanonical")
    if (
        not isinstance(manifest.get("created_utc"), str)
        or not manifest["created_utc"].endswith("Z")
        or not isinstance(manifest.get("elapsed_seconds"), (int, float))
        or isinstance(manifest.get("elapsed_seconds"), bool)
        or not math.isfinite(float(manifest["elapsed_seconds"]))
        or float(manifest["elapsed_seconds"]) <= 0
    ):
        raise VerificationError("embedding manifest timing fields are invalid")
    generator = _manifest_record_core(
        manifest.get("generator"),
        fields={"bytes", "mtime_ns", "path", "sha256"},
        label="embedding.generator",
        expected=EXPECTED_GENERATOR_RECORD,
    )
    input_block = manifest.get("input")
    outputs = manifest.get("outputs")
    model = manifest.get("model")
    generation = manifest.get("generation")
    environment = manifest.get("environment")
    if not all(
        isinstance(value, Mapping)
        for value in (input_block, outputs, model, generation, environment)
    ):
        raise VerificationError("embedding manifest is missing a provenance section")
    assert isinstance(input_block, Mapping)
    assert isinstance(outputs, Mapping)
    assert isinstance(model, Mapping)
    assert isinstance(generation, Mapping)
    assert isinstance(environment, Mapping)
    _require_exact_keys(
        input_block,
        {
            "fasta",
            "sequence_names_sha256",
            "sequence_order_and_content_sha256",
            "statistics",
        },
        label="embedding.input",
    )
    _require_exact_keys(
        outputs,
        {"embeddings", "normalized_embeddings", "names"},
        label="embedding.outputs",
    )
    fasta = _manifest_record_core(
        input_block.get("fasta"),
        fields={"bytes", "mtime_ns", "path", "sha256"},
        label="embedding.input.fasta",
        expected=EXPECTED_FASTA_RECORD,
    )
    output_fields = {
        "embeddings": {
            "bytes",
            "dtype",
            "mtime_ns",
            "numpy_allow_pickle",
            "path",
            "sha256",
            "shape",
        },
        "normalized_embeddings": {
            "bytes",
            "dtype",
            "mtime_ns",
            "normalization",
            "numpy_allow_pickle",
            "path",
            "sha256",
            "shape",
        },
        "names": {
            "bytes",
            "encoding",
            "line_ending",
            "mtime_ns",
            "path",
            "records",
            "sha256",
        },
    }
    output_records = {
        name: _manifest_record_core(
            outputs.get(name),
            fields=output_fields[name],
            label=f"embedding.outputs.{name}",
            expected=EXPECTED_EMBEDDING_OUTPUTS[name],
        )
        for name in ("embeddings", "normalized_embeddings", "names")
    }
    for name in ("embeddings", "normalized_embeddings"):
        record = outputs[name]
        if (
            record.get("dtype") != "float32"
            or record.get("shape") != [EXPECTED_N_GENES, 320]
            or record.get("numpy_allow_pickle") is not False
        ):
            raise VerificationError(f"embedding.outputs.{name} metadata is invalid")
    if outputs["normalized_embeddings"].get("normalization") != (
        "row-wise L2 with float32 epsilon floor"
    ):
        raise VerificationError("normalized embedding policy is noncanonical")
    if (
        outputs["names"].get("records") != EXPECTED_N_GENES
        or outputs["names"].get("encoding") != "UTF-8"
        or outputs["names"].get("line_ending") != "LF"
    ):
        raise VerificationError("embedding name-file metadata is noncanonical")

    _verify_record(
        _workspace_path(generator["path"], label="embedding.generator"),
        generator,
        label="embedding.generator",
        context=context,
    )
    fasta_payload, _ = _verify_record(
        _workspace_path(fasta["path"], label="embedding.input.fasta"),
        fasta,
        label="embedding.input.fasta",
        context=context,
        keep_bytes=True,
    )
    assert fasta_payload is not None
    output_payloads: dict[str, bytes] = {}
    for name, record in output_records.items():
        payload, _ = _verify_record(
            _workspace_path(record["path"], label=f"embedding.outputs.{name}"),
            record,
            label=f"embedding.outputs.{name}",
            context=context,
            keep_bytes=True,
        )
        assert payload is not None
        output_payloads[name] = payload

    _require_exact_keys(model, EMBEDDING_MODEL_FIELDS, label="embedding.model")
    if (
        model.get("repository") != MODEL_REPOSITORY
        or model.get("model_revision") != MODEL_REVISION
        or model.get("tokenizer_revision") != MODEL_REVISION
        or model.get("model_class") != "EsmModel"
        or model.get("tokenizer_class") != "EsmTokenizer"
    ):
        raise VerificationError("embedding model/tokenizer identity is not canonical")
    configuration = model.get("configuration")
    if not isinstance(configuration, Mapping):
        raise VerificationError("embedding model configuration is missing")
    _require_exact_keys(
        configuration,
        EMBEDDING_CONFIGURATION_FIELDS,
        label="embedding.model.configuration",
    )
    expected_configuration = {
        "architectures": ["EsmForMaskedLM"],
        "attention_probs_dropout_prob": 0.0,
        "dtype": "float32",
        "hidden_dropout_prob": 0.0,
        "hidden_size": 320,
        "intermediate_size": 1280,
        "mask_token_id": 32,
        "max_position_embeddings": 1026,
        "model_type": "esm",
        "num_attention_heads": 20,
        "num_hidden_layers": 6,
        "pad_token_id": 1,
        "position_embedding_type": "rotary",
        "token_dropout": True,
        "transformers_version": EXPECTED_KEY_VERSIONS["transformers"],
        "vocab_size": 33,
    }
    if any(
        configuration.get(name) != value
        for name, value in expected_configuration.items()
    ):
        raise VerificationError("embedding model configuration is noncanonical")

    expected_generation = {
        "batch_size_chunks": 64,
        "strategy": "chunked",
        "maximum_residues_per_window": 1022,
        "overlap_residues": 128,
        "stride_residues": 894,
        "special_tokens_excluded": True,
        "padding_tokens_excluded": True,
        "truncation_enabled_in_tokenizer": False,
        "output_dtype": "float32",
        "output_shape": [EXPECTED_N_GENES, 320],
        "recommended_for_primary_analysis": True,
        "pooling": (
            "inverse-coverage-weighted arithmetic mean of residue hidden states "
            "across deterministic overlapping windows"
        ),
        "long_protein_behavior": (
            "all residues represented; overlap duplicates receive inverse coverage "
            "weights so every original residue has total weight one"
        ),
    }
    _require_exact_keys(
        generation, EMBEDDING_GENERATION_FIELDS, label="embedding.generation"
    )
    if dict(generation) != expected_generation:
        raise VerificationError("embedding generation protocol is noncanonical")

    _require_exact_keys(
        environment, EMBEDDING_ENVIRONMENT_FIELDS, label="embedding.environment"
    )
    if (
        environment.get("cuda_available") is not True
        or environment.get("device") != "cuda"
        or environment.get("deterministic_algorithms_enabled") is not True
        or environment.get("deterministic_debug_mode") != 2
        or environment.get("cudnn_deterministic") is not True
        or environment.get("cudnn_benchmark") is not False
    ):
        raise VerificationError("embedding generation environment is not deterministic")
    installed = environment.get("installed_distributions")
    key_versions = environment.get("key_distribution_versions")
    if (
        not isinstance(installed, list)
        or len(installed) != 56
        or not isinstance(key_versions, Mapping)
        or dict(key_versions) != EXPECTED_KEY_VERSIONS
    ):
        raise VerificationError("embedding environment inventory is missing")
    inventory: dict[str, str] = {}
    inventory_records: list[dict[str, str]] = []
    for index, item in enumerate(installed):
        if not isinstance(item, Mapping):
            raise VerificationError(f"embedding environment item {index} is invalid")
        item_name, item_version = item.get("name"), item.get("version")
        if (
            not isinstance(item_name, str)
            or not item_name
            or not isinstance(item_version, str)
            or not item_version
        ):
            raise VerificationError(f"embedding environment item {index} is invalid")
        normalized = item_name.casefold()
        if normalized in inventory:
            raise VerificationError(
                "embedding environment has a duplicate distribution"
            )
        inventory[normalized] = item_version
        inventory_records.append({"name": item_name, "version": item_version})
    for package in ("huggingface_hub", "numpy", "safetensors", "torch", "transformers"):
        if not isinstance(key_versions.get(package), str) or inventory.get(
            package.casefold()
        ) != key_versions.get(package):
            raise VerificationError(f"embedding environment disagrees for {package}")
    inventory_sha256 = _canonical_sha256({"installed_distributions": inventory_records})
    if inventory_sha256 != EXPECTED_ENVIRONMENT_INVENTORY_SHA256:
        raise VerificationError("embedding environment inventory is noncanonical")
    string_environment_fields = (
        "cuda_device_name",
        "machine",
        "platform",
        "processor",
        "python",
        "python_executable",
        "torch_build_config",
        "torch_cuda_version",
        "torch_version",
    )
    if any(
        not isinstance(environment.get(name), str) or not environment[name]
        for name in string_environment_fields
    ):
        raise VerificationError("embedding environment string metadata is invalid")
    if (
        environment.get("torch_version") != EXPECTED_KEY_VERSIONS["torch"]
        or not isinstance(environment.get("cudnn_version"), int)
        or isinstance(environment.get("cudnn_version"), bool)
        or not isinstance(environment.get("cuda_device_capability"), list)
        or len(environment["cuda_device_capability"]) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in environment["cuda_device_capability"]
        )
    ):
        raise VerificationError("embedding CUDA/runtime metadata is invalid")
    numerical_environment = environment.get("numerical_environment")
    if (
        not isinstance(numerical_environment, Mapping)
        or set(numerical_environment)
        != {
            "CUBLAS_WORKSPACE_CONFIG",
            "CUDA_VISIBLE_DEVICES",
            "PYTORCH_CUDA_ALLOC_CONF",
        }
        or numerical_environment.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
    ):
        raise VerificationError("embedding numerical environment is noncanonical")
    nvidia_smi = environment.get("nvidia_smi")
    if (
        not isinstance(nvidia_smi, Mapping)
        or set(nvidia_smi) != {"available", "format", "query", "rows"}
        or nvidia_smi.get("available") is not True
        or not isinstance(nvidia_smi.get("rows"), list)
        or not nvidia_smi["rows"]
        or any(not isinstance(row, str) or not row for row in nvidia_smi["rows"])
    ):
        raise VerificationError("embedding NVIDIA environment record is invalid")

    snapshot_path_value = model.get("snapshot_path")
    if not isinstance(snapshot_path_value, str) or not snapshot_path_value:
        raise VerificationError("embedding model snapshot path is missing")
    snapshot_root = Path(snapshot_path_value)
    if not snapshot_root.is_absolute() and not ntpath.isabs(snapshot_path_value):
        raise VerificationError("embedding model snapshot path must be absolute")
    _check_path_components(snapshot_root)
    if snapshot_root.name != MODEL_REVISION:
        raise VerificationError(
            "embedding snapshot path does not end in the pinned revision"
        )
    configuration_snapshot = configuration.get("_name_or_path")
    if not isinstance(configuration_snapshot, str) or os.path.normcase(
        os.path.abspath(configuration_snapshot)
    ) != os.path.normcase(os.path.abspath(snapshot_root)):
        raise VerificationError(
            "loaded model configuration is not bound to snapshot_path"
        )
    snapshot_records = model.get("snapshot_files")
    if not isinstance(snapshot_records, list) or len(snapshot_records) != len(
        EXPECTED_SNAPSHOT_FILES
    ):
        raise VerificationError("embedding snapshot file inventory is missing")
    observed_snapshot_names: set[str] = set()
    normalized_snapshot_records: list[dict[str, Any]] = []
    snapshot_payloads: dict[str, bytes] = {}
    for index, raw_record in enumerate(snapshot_records):
        if not isinstance(raw_record, Mapping):
            raise VerificationError("embedding snapshot record is invalid")
        raw_name = _normalize_relative(
            raw_record.get("path"),
            label=f"embedding.model.snapshot_files[{index}].path",
        )
        expected_snapshot = EXPECTED_SNAPSHOT_FILES.get(raw_name)
        if expected_snapshot is None:
            raise VerificationError("embedding snapshot contains an unregistered file")
        core = _manifest_record_core(
            raw_record,
            fields={"bytes", "mtime_ns", "path", "sha256"},
            label=f"embedding.model.snapshot_files[{index}]",
            expected={"path": raw_name, **expected_snapshot},
        )
        if "/" in core["path"] or core["path"] in observed_snapshot_names:
            raise VerificationError(
                "embedding snapshot file path is invalid or duplicated"
            )
        observed_snapshot_names.add(core["path"])
        normalized_snapshot_records.append(core)
        payload, _ = _verify_record(
            _safe_child(
                snapshot_root,
                core["path"],
                label=f"embedding snapshot {core['path']}",
            ),
            core,
            label=f"embedding snapshot {core['path']}",
            context=context,
            keep_bytes=core["path"] == "config.json",
        )
        if payload is not None:
            snapshot_payloads[core["path"]] = payload
    if observed_snapshot_names != set(EXPECTED_SNAPSHOT_FILES):
        raise VerificationError("embedding snapshot inventory is noncanonical")
    _require_file_inventory(
        snapshot_root,
        set(EXPECTED_SNAPSHOT_FILES),
        label="embedding model snapshot",
    )
    snapshot_config = _parse_json(
        snapshot_payloads["config.json"], label="embedding snapshot config.json"
    )
    if any(
        snapshot_config.get(name) != expected
        for name, expected in {
            "hidden_size": 320,
            "intermediate_size": 1280,
            "max_position_embeddings": 1026,
            "model_type": "esm",
            "num_attention_heads": 20,
            "num_hidden_layers": 6,
            "vocab_size": 33,
        }.items()
    ):
        raise VerificationError("snapshot config does not describe the registered ESM")

    fasta_names, fasta_sequences = _load_fasta_records(fasta_payload)
    if fasta_names != list(genes):
        raise VerificationError("FASTA identifiers/order do not match genes.txt")
    sequence_digest = hashlib.sha256()
    for name, sequence in zip(fasta_names, fasta_sequences, strict=True):
        sequence_digest.update(name.encode("utf-8"))
        sequence_digest.update(b"\0")
        sequence_digest.update(sequence.encode("ascii"))
        sequence_digest.update(b"\0")
    lengths = np.asarray(
        [len(sequence) for sequence in fasta_sequences], dtype=np.int64
    )
    long_names = [
        name
        for name, length in zip(fasta_names, lengths, strict=True)
        if int(length) > LONG_THRESHOLD
    ]
    expected_statistics = {
        "fraction_sequences_longer_than_window": float(len(long_names) / len(lengths)),
        "long_sequence_names_sha256": _hash_strings(long_names),
        "maximum_length_residues": int(lengths.max()),
        "median_length_residues": float(np.median(lengths)),
        "minimum_length_residues": int(lengths.min()),
        "n_sequences": int(len(lengths)),
        "n_sequences_longer_than_window": len(long_names),
    }
    if (
        input_block.get("sequence_names_sha256") != _hash_strings(fasta_names)
        or input_block.get("sequence_names_sha256") != EXPECTED_SEQUENCE_NAMES_SHA256
        or input_block.get("sequence_order_and_content_sha256")
        != sequence_digest.hexdigest()
        or input_block.get("sequence_order_and_content_sha256")
        != EXPECTED_SEQUENCE_CONTENT_SHA256
        or input_block.get("statistics") != expected_statistics
        or expected_statistics["long_sequence_names_sha256"]
        != EXPECTED_LONG_NAMES_SHA256
        or expected_statistics["n_sequences"] != EXPECTED_N_GENES
        or expected_statistics["n_sequences_longer_than_window"] != EXPECTED_N_LONG
    ):
        raise VerificationError(
            "embedding FASTA provenance does not independently recompute"
        )

    names_payload = output_payloads["names"]
    if b"\r" in names_payload or not names_payload.endswith(b"\n"):
        raise VerificationError("embedding name file is not canonical LF text")
    embedded_names = _load_lines(names_payload, label="embedding names")
    if embedded_names != fasta_names:
        raise VerificationError("embedding names do not match FASTA order")
    raw_embeddings = _load_npy(
        output_payloads["embeddings"], label="raw ESM embeddings"
    )
    normalized_embeddings = _load_npy(
        output_payloads["normalized_embeddings"], label="normalized ESM embeddings"
    )
    expected_shape = (EXPECTED_N_GENES, 320)
    for label, array in (
        ("raw ESM embeddings", raw_embeddings),
        ("normalized ESM embeddings", normalized_embeddings),
    ):
        if (
            array.dtype != np.float32
            or array.shape != expected_shape
            or not array.flags.c_contiguous
            or not np.isfinite(array).all()
        ):
            raise VerificationError(f"{label} has invalid dtype/shape/layout/values")
    for start in range(0, len(raw_embeddings), 2048):
        raw_chunk = raw_embeddings[start : start + 2048]
        expected_normalized = raw_chunk / np.maximum(
            np.linalg.norm(raw_chunk, axis=1, keepdims=True),
            np.finfo(np.float32).eps,
        )
        expected_normalized = expected_normalized.astype(np.float32)
        if not np.allclose(
            normalized_embeddings[start : start + 2048],
            expected_normalized,
            rtol=2e-6,
            atol=2e-6,
        ):
            raise VerificationError("normalized ESM embeddings do not recompute")
    if (
        _record_core(analysis_embedding_record, label="analysis source embedding_array")
        != output_records["embeddings"]
    ):
        raise VerificationError(
            "analysis embedding array is not the manifest raw output"
        )

    aggregation = campaign.get("aggregation_provenance")
    if not isinstance(aggregation, Mapping):
        raise VerificationError("campaign aggregation provenance is missing")
    bundle = aggregation.get("embedding_bundle")
    if (
        not isinstance(bundle, dict)
        or bundle.get("schema") != EMBEDDING_PROVENANCE_SCHEMA
    ):
        raise VerificationError("campaign embedding provenance is missing")
    _require_exact_keys(
        bundle,
        {
            "environment",
            "generation",
            "generator",
            "input",
            "manifest",
            "model",
            "outputs",
            "provenance_sha256",
            "schema",
            "verification_status",
        },
        label="campaign embedding provenance",
    )
    if _record_core(
        bundle.get("manifest"), label="campaign embedding manifest"
    ) != _record_core(manifest_record, label="analysis embedding manifest"):
        raise VerificationError(
            "campaign embedding manifest binding differs from analysis source"
        )
    declared_provenance_hash = bundle.get("provenance_sha256")
    if (
        not isinstance(declared_provenance_hash, str)
        or SHA256_PATTERN.fullmatch(declared_provenance_hash) is None
    ):
        raise VerificationError("campaign embedding provenance hash is invalid")
    without_hash = {
        key: value for key, value in bundle.items() if key != "provenance_sha256"
    }
    if _canonical_sha256(without_hash) != declared_provenance_hash:
        raise VerificationError("campaign embedding provenance hash does not recompute")
    bundle_model = bundle.get("model")
    bundle_generation = bundle.get("generation")
    bundle_input = bundle.get("input")
    bundle_outputs = bundle.get("outputs")
    bundle_environment = bundle.get("environment")
    if not all(
        isinstance(value, Mapping)
        for value in (
            bundle_model,
            bundle_generation,
            bundle_input,
            bundle_outputs,
            bundle_environment,
        )
    ):
        raise VerificationError("campaign embedding provenance is incomplete")
    assert isinstance(bundle_model, Mapping)
    assert isinstance(bundle_generation, Mapping)
    assert isinstance(bundle_input, Mapping)
    assert isinstance(bundle_outputs, Mapping)
    assert isinstance(bundle_environment, Mapping)
    _require_exact_keys(
        bundle_model,
        {
            "hidden_size",
            "model_class",
            "model_revision",
            "num_hidden_layers",
            "repository",
            "snapshot_files",
            "tokenizer_class",
            "tokenizer_revision",
        },
        label="campaign embedding model",
    )
    _require_exact_keys(
        bundle_generation,
        EMBEDDING_GENERATION_FIELDS - {"batch_size_chunks"},
        label="campaign embedding generation",
    )
    _require_exact_keys(
        bundle_input,
        {
            "fasta",
            "sequence_names_sha256",
            "sequence_order_and_content_sha256",
            "statistics",
        },
        label="campaign embedding input",
    )
    _require_exact_keys(
        bundle_outputs,
        {"embeddings", "normalized_embeddings", "names"},
        label="campaign embedding outputs",
    )
    if (
        bundle.get("verification_status") != "verified"
        or _record_core(bundle.get("generator"), label="campaign embedding generator")
        != generator
        or _record_core(bundle_input.get("fasta"), label="campaign embedding FASTA")
        != fasta
        or bundle_input.get("sequence_names_sha256")
        != input_block.get("sequence_names_sha256")
        or bundle_input.get("sequence_order_and_content_sha256")
        != input_block.get("sequence_order_and_content_sha256")
        or bundle_input.get("statistics") != input_block.get("statistics")
    ):
        raise VerificationError(
            "campaign embedding input provenance differs from manifest"
        )
    for name, record in output_records.items():
        if (
            _record_core(
                bundle_outputs.get(name), label=f"campaign embedding output {name}"
            )
            != record
        ):
            raise VerificationError(
                f"campaign embedding output {name} differs from manifest"
            )
    expected_bundle_model = {
        "repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "model_class": "EsmModel",
        "tokenizer_class": "EsmTokenizer",
        "hidden_size": 320,
        "num_hidden_layers": 6,
        "snapshot_files": sorted(
            normalized_snapshot_records, key=lambda item: item["path"]
        ),
    }
    if dict(bundle_model) != expected_bundle_model:
        raise VerificationError(
            "campaign embedding model provenance differs from manifest"
        )
    generation_keys = (
        "strategy",
        "maximum_residues_per_window",
        "overlap_residues",
        "stride_residues",
        "pooling",
        "long_protein_behavior",
        "special_tokens_excluded",
        "padding_tokens_excluded",
        "truncation_enabled_in_tokenizer",
        "output_dtype",
        "output_shape",
        "recommended_for_primary_analysis",
    )
    if dict(bundle_generation) != {key: generation.get(key) for key in generation_keys}:
        raise VerificationError(
            "campaign embedding generation provenance differs from manifest"
        )
    compact_environment = {
        "python": environment.get("python"),
        "platform": environment.get("platform"),
        "machine": environment.get("machine"),
        "device": environment.get("device"),
        "cuda_available": environment.get("cuda_available"),
        "cuda_device_name": environment.get("cuda_device_name"),
        "torch_cuda_version": environment.get("torch_cuda_version"),
        "cudnn_version": environment.get("cudnn_version"),
        "key_distribution_versions": dict(key_versions),
        "installed_distribution_count": len(installed),
        "installed_distributions_sha256": _canonical_sha256(
            {"installed_distributions": inventory_records}
        ),
        "deterministic_algorithms_enabled": True,
        "deterministic_debug_mode": 2,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
    }
    if dict(bundle_environment) != compact_environment:
        raise VerificationError("campaign embedding environment differs from manifest")
    return {
        "provenance_sha256": declared_provenance_hash,
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "environment_inventory_sha256": compact_environment[
            "installed_distributions_sha256"
        ],
        "snapshot_files_verified": len(snapshot_records),
        "raw_embeddings_hash_verified": int(raw_embeddings.shape[0]),
        "normalized_embeddings_recomputed": int(normalized_embeddings.shape[0]),
        "fasta_records_recomputed": len(fasta_names),
        "all_residue_pooling_protocol_bound_to_registered_generator": True,
    }


def _load_lines(
    payload: bytes, *, label: str, require_unique: bool = True
) -> list[str]:
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise VerificationError(f"{label} is not UTF-8") from exc
    values = [line.strip() for line in text.splitlines()]
    if (
        not values
        or any(not value for value in values)
        or (require_unique and len(values) != len(set(values)))
    ):
        raise VerificationError(f"{label} contains empty or duplicate identifiers")
    return values


def _load_fasta_records(payload: bytes) -> tuple[list[str], list[str]]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise VerificationError("protein FASTA is not UTF-8") from exc
    names: list[str] = []
    sequences: list[str] = []
    current: str | None = None
    pieces: list[str] = []
    seen: set[str] = set()

    def finish() -> None:
        if current is None:
            return
        sequence = "".join(pieces)
        if not sequence:
            raise VerificationError(f"empty FASTA sequence for {current}")
        try:
            sequence.encode("ascii")
        except UnicodeError as exc:
            raise VerificationError(
                f"non-ASCII residue data for FASTA record {current}"
            ) from exc
        sequences.append(sequence)

    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            finish()
            candidate = line[1:].split("|", 1)[0]
            if not candidate or candidate in seen:
                raise VerificationError(f"invalid FASTA header at line {line_number}")
            seen.add(candidate)
            names.append(candidate)
            current = candidate
            pieces = []
        else:
            if current is None or any(character.isspace() for character in line):
                raise VerificationError(f"invalid FASTA sequence at line {line_number}")
            pieces.append(line)
    finish()
    if not names:
        raise VerificationError("protein FASTA contains no records")
    return names, sequences


def _load_fasta_lengths(payload: bytes, *, genes: Sequence[str]) -> np.ndarray:
    names, sequences = _load_fasta_records(payload)
    if names != list(genes):
        raise VerificationError("FASTA identifiers/order do not match genes.txt")
    return np.asarray([len(sequence) for sequence in sequences], dtype=np.int32)


def _load_npy(payload: bytes, *, label: str) -> np.ndarray:
    try:
        value = np.load(io.BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise VerificationError(f"cannot load {label}") from exc
    if not isinstance(value, np.ndarray):
        raise VerificationError(f"{label} is not a NumPy array")
    return np.asarray(value)


def _load_npz(
    payload: bytes, *, label: str, names: tuple[str, ...]
) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if tuple(archive.files) != names:
                raise VerificationError(f"{label} has a noncanonical array inventory")
            return {name: np.asarray(archive[name]) for name in archive.files}
    except VerificationError:
        raise
    except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile) as exc:
        raise VerificationError(f"cannot load {label}") from exc


def _load_p1_oof(
    run: Mapping[str, Any],
    *,
    run_path: Path,
    cells: Sequence[str],
    genes: Sequence[str],
    fold_evidence: Mapping[tuple[str, str], FoldEvidence],
    context: VerificationContext,
) -> tuple[np.ndarray, np.ndarray]:
    key = run.get("key")
    protocol, condition, seed = _parse_run_key(key)
    if not isinstance(key, str):  # Narrow the type after the fail-closed parser.
        raise VerificationError("campaign run key must be a string")
    if protocol != "p1":
        raise VerificationError(f"selected OOF run is not P1: {key}")
    records: dict[str, dict[str, Any]] = {}
    artifacts = run.get("fold_artifacts")
    if not isinstance(artifacts, list):
        raise VerificationError("P1 run lacks fold artifacts")
    for index, raw_record in enumerate(artifacts):
        core = _record_core(raw_record, label=f"{run.get('key')}.artifact[{index}]")
        if core["path"] in records:
            raise VerificationError(f"duplicate fold artifact in {run.get('key')}")
        records[core["path"]] = core
    expected_prediction_names = {f"{fold}-predictions.npz" for fold in FOLDS}
    if {
        name for name in records if name.endswith("-predictions.npz")
    } != expected_prediction_names:
        raise VerificationError(
            f"P1 prediction inventory is incomplete: {run.get('key')}"
        )
    predictions: np.ndarray = np.full(
        (len(cells), len(genes)), np.nan, dtype=np.float32
    )
    baselines = np.full_like(predictions, np.nan)
    coverage: np.ndarray = np.zeros(len(cells), dtype=np.int16)
    expected_gene_ids = np.asarray(genes, dtype=str)
    expected_gene_indices: np.ndarray = np.arange(len(genes), dtype=np.int64)
    for fold_number, fold in enumerate(FOLDS):
        name = f"{fold}-predictions.npz"
        proof = fold_evidence.get((key, fold))
        if proof is None:
            raise VerificationError(
                f"selected fold lacks protocol evidence: {key}.{fold}"
            )
        if (
            proof.key != key
            or proof.protocol != protocol
            or proof.condition != condition
            or proof.seed != seed
            or proof.fold != fold
            or proof.fold_number != fold_number
            or proof.fold_seed != _derive_seed(seed, fold)
            or proof.prediction_record != records[name]
        ):
            raise VerificationError(
                f"selected fold evidence identity differs: {key}.{fold}"
            )
        payload, _ = _verify_record(
            _safe_child(run_path, name, label=f"{run.get('key')}.{name}"),
            records[name],
            label=f"{run.get('key')}.{name}",
            context=context,
            keep_bytes=True,
        )
        assert payload is not None
        arrays = _load_npz(
            payload, label=f"{run.get('key')}.{name}", names=EXPECTED_PREDICTION_ARRAYS
        )
        indices = arrays["outer_test_indices"]
        prediction = arrays["prediction"]
        train_gene_mean = arrays["train_gene_mean"]
        if indices.dtype != np.int64 or indices.ndim != 1:
            raise VerificationError(
                f"invalid outer-test indices in {run.get('key')}.{name}"
            )
        if (
            prediction.dtype != np.float32
            or prediction.shape != (len(indices), len(genes))
            or not np.isfinite(prediction).all()
        ):
            raise VerificationError(f"invalid predictions in {run.get('key')}.{name}")
        if (
            train_gene_mean.dtype != np.float32
            or train_gene_mean.shape != (len(genes),)
            or not np.isfinite(train_gene_mean).all()
        ):
            raise VerificationError(
                f"invalid training gene mean in {run.get('key')}.{name}"
            )
        if (
            np.any(indices < 0)
            or np.any(indices >= len(cells))
            or len(indices) != len(np.unique(indices))
        ):
            raise VerificationError(f"out-of-range or duplicate test indices in {name}")
        if not np.array_equal(indices, proof.outer_test):
            raise VerificationError(
                f"selected prediction indices differ from split evidence in {name}"
            )
        if not np.array_equal(arrays["gene_ids"], expected_gene_ids):
            raise VerificationError(f"prediction gene order differs in {name}")
        expected_cells = np.asarray([cells[int(index)] for index in indices], dtype=str)
        if not np.array_equal(arrays["cell_ids"], expected_cells):
            raise VerificationError(f"prediction cell identities differ in {name}")
        if not np.array_equal(
            arrays["source_cell_indices"], proof.source_cell_indices[proof.outer_test]
        ):
            raise VerificationError(f"source-cell indices differ in {name}")
        if not np.array_equal(
            arrays["source_gene_indices"], proof.source_gene_indices
        ) or not np.array_equal(proof.source_gene_indices, expected_gene_indices):
            raise VerificationError(f"source-gene indices differ in {name}")
        predictions[indices] = prediction
        baselines[indices] = train_gene_mean[None, :]
        coverage[indices] += 1
    if not np.all(coverage == 1) or not np.isfinite(predictions).all():
        raise VerificationError(
            f"P1 folds do not cover cells exactly once: {run.get('key')}"
        )
    return predictions, baselines


def _per_gene_error(
    prediction: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if prediction.ndim != 2 or prediction.shape != target.shape:
        raise VerificationError("prediction and target matrices must share a 2-D shape")
    valid = np.isfinite(prediction) & np.isfinite(target)
    counts = valid.sum(axis=0).astype(np.int32)
    difference = np.where(valid, prediction - target, np.nan)
    with np.errstate(invalid="ignore"):
        mae = np.nanmean(np.abs(difference), axis=0)
        mse = np.nanmean(np.square(difference), axis=0)
    mae[counts == 0] = np.nan
    mse[counts == 0] = np.nan
    return mae.astype(np.float64), mse.astype(np.float64), counts


def _compute_arrays(
    source_runs: Mapping[str, Mapping[str, Any]],
    run_paths: Mapping[str, Path],
    *,
    cells: Sequence[str],
    genes: Sequence[str],
    targets: np.ndarray,
    lengths: np.ndarray,
    fold_evidence: Mapping[tuple[str, str], FoldEvidence],
    context: VerificationContext,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    shape = (len(CONDITIONS), len(SEEDS), len(genes))
    mae: np.ndarray = np.full(shape, np.nan, dtype=np.float64)
    mse: np.ndarray = np.full(shape, np.nan, dtype=np.float64)
    baseline_mae: np.ndarray = np.full(
        (len(SEEDS), len(genes)), np.nan, dtype=np.float64
    )
    baseline_mse = np.full_like(baseline_mae, np.nan)
    finite_counts: np.ndarray | None = None
    used_runs: list[dict[str, Any]] = []
    for condition_index, condition in enumerate(CONDITIONS):
        for seed_index, seed in enumerate(SEEDS):
            key = f"p1|{condition}|seed-{seed}"
            run = source_runs.get(key)
            run_path = run_paths.get(key)
            if run is None or run_path is None:
                raise VerificationError(f"selected P1 run is missing: {key}")
            prediction, baseline = _load_p1_oof(
                run,
                run_path=run_path,
                cells=cells,
                genes=genes,
                fold_evidence=fold_evidence,
                context=context,
            )
            model_mae, model_mse, counts = _per_gene_error(prediction, targets)
            current_baseline_mae, current_baseline_mse, baseline_counts = (
                _per_gene_error(baseline, targets)
            )
            if not np.array_equal(counts, baseline_counts):
                raise VerificationError(f"baseline target mask differs for {key}")
            if finite_counts is None:
                finite_counts = counts
            elif not np.array_equal(finite_counts, counts):
                raise VerificationError(
                    "finite target masks differ across matched runs"
                )
            mae[condition_index, seed_index] = model_mae
            mse[condition_index, seed_index] = model_mse
            if condition_index == 0:
                baseline_mae[seed_index] = current_baseline_mae
                baseline_mse[seed_index] = current_baseline_mse
            elif not np.array_equal(
                baseline_mae[seed_index], current_baseline_mae, equal_nan=True
            ) or not np.array_equal(
                baseline_mse[seed_index], current_baseline_mse, equal_nan=True
            ):
                raise VerificationError(f"outer split baseline differs for seed {seed}")
            used_runs.append(
                {
                    "key": key,
                    "completion_sha256": run["completion_sha256"],
                    "summary_sha256": run["summary_sha256"],
                    "run_manifest_sha256": run["run_manifest_sha256"],
                    "cohort_manifest_sha256": run["cohort_manifest_sha256"],
                }
            )
    assert finite_counts is not None
    return (
        {
            "gene_ids": np.asarray(genes, dtype=str),
            "length_residues": np.asarray(lengths, dtype=np.int32),
            "condition_ids": np.asarray(CONDITIONS, dtype=str),
            "seed_ids": np.asarray(SEEDS, dtype=np.int64),
            "mae": mae,
            "mse": mse,
            "baseline_mae": baseline_mae,
            "baseline_mse": baseline_mse,
            "finite_target_counts": finite_counts,
        },
        used_runs,
    )


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
        raise VerificationError("too few genes for descriptive length comparison")
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
    statistic = float(result.statistic)
    return {
        "rho": statistic if math.isfinite(statistic) else None,
        "n_genes": int(mask.sum()),
        "inference": "conditional_descriptive_only_no_iid_p_value",
    }


def _summarize_arrays(
    arrays: Mapping[str, np.ndarray], *, analysis_schema: str = ANALYSIS_SCHEMA
) -> dict[str, Any]:
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
        raise VerificationError("gene identifiers or protein lengths are invalid")
    expected_shape = (len(conditions), len(seeds), len(genes))
    if mae.shape != expected_shape or mse.shape != expected_shape:
        raise VerificationError("per-gene model error arrays have invalid shape")
    if baseline_mae.shape != (len(seeds), len(genes)) or baseline_mse.shape != (
        len(seeds),
        len(genes),
    ):
        raise VerificationError("per-gene baseline arrays have invalid shape")
    if counts.shape != (len(genes),) or np.any(counts <= 0):
        raise VerificationError("finite target counts are invalid")
    if not all(
        np.isfinite(value).all() for value in (mae, mse, baseline_mae, baseline_mse)
    ) or any(np.any(value < 0) for value in (mae, mse, baseline_mae, baseline_mse)):
        raise VerificationError("per-gene error arrays contain invalid values")
    if conditions != list(CONDITIONS) or seeds != list(SEEDS):
        raise VerificationError("condition or seed axes are noncanonical")
    long_mask = lengths > LONG_THRESHOLD
    short_mask = ~long_mask
    if len(genes) != EXPECTED_N_GENES or int(long_mask.sum()) != EXPECTED_N_LONG:
        raise VerificationError("registered long-protein cohort size changed")
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
    bin_masks = {
        "25-512": lengths <= 512,
        "513-1022": (lengths >= 513) & (lengths <= LONG_THRESHOLD),
        "1023-2044": (lengths > LONG_THRESHOLD) & (lengths <= 2044),
        ">2044": lengths > 2044,
    }
    bins: list[dict[str, Any]] = []
    for label, mask in bin_masks.items():
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
        "schema": analysis_schema,
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


def _compare_arrays(
    recomputed: Mapping[str, np.ndarray], saved: Mapping[str, np.ndarray]
) -> None:
    if set(recomputed) != set(saved):
        raise VerificationError("saved per-gene array inventory differs from replay")
    for name in EXPECTED_PER_GENE_ARRAYS:
        left = np.asarray(recomputed[name])
        right = np.asarray(saved[name])
        if left.dtype != right.dtype or left.shape != right.shape:
            raise VerificationError(f"saved per-gene array differs from replay: {name}")
        equal = (
            np.array_equal(left, right, equal_nan=True)
            if np.issubdtype(left.dtype, np.inexact)
            else np.array_equal(left, right)
        )
        if not equal:
            raise VerificationError(f"saved per-gene array differs from replay: {name}")


def _selected_run_bindings(
    source_runs: Mapping[str, Mapping[str, Any]], run_paths: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    input_binding: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            key = f"p1|{condition}|seed-{seed}"
            run = source_runs[key]
            input_binding.append(
                {
                    "key": key,
                    "path": _normalize_relative(
                        run_paths[key], label=f"run path {key}"
                    ),
                    "run_manifest_sha256": run["run_manifest_sha256"],
                }
            )
            selected.append(
                {
                    "key": key,
                    "completion_sha256": run["completion_sha256"],
                    "summary_sha256": run["summary_sha256"],
                    "run_manifest_sha256": run["run_manifest_sha256"],
                    "cohort_manifest_sha256": run["cohort_manifest_sha256"],
                }
            )
    return sorted(input_binding, key=lambda item: item["key"]), selected


def verify_analysis(
    run_directory: Path,
    *,
    trusted_completion_sha256: str,
    output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify an analysis while keeping the RECORD-validating importer active."""
    try:
        return _verify_analysis_with_bound_runtime(
            run_directory,
            trusted_completion_sha256=trusted_completion_sha256,
            output=output,
        )
    finally:
        _restore_trusted_import_runtime()


def _verify_analysis_with_bound_runtime(
    run_directory: Path,
    *,
    trusted_completion_sha256: str,
    output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify a completed analysis and atomically write a self-bound report."""
    if SHA256_PATTERN.fullmatch(trusted_completion_sha256) is None:
        raise VerificationError(
            "trusted completion SHA-256 must be 64 lowercase hex characters"
        )
    context = VerificationContext(snapshots={})
    verifier_payload, verifier_snapshot = _capture_file(
        Path(__file__),
        label="independent verifier source",
        context=context,
        keep_bytes=False,
    )
    assert verifier_payload is None
    run_directory = Path(os.path.abspath(run_directory))
    _check_path_components(run_directory)
    if not run_directory.is_dir():
        raise VerificationError("analysis run directory does not exist")
    _require_file_inventory(
        run_directory, set(ANALYSIS_RUN_FILES), label="analysis run"
    )
    output = Path(os.path.abspath(output))
    output_parent = output.parent
    _check_path_components(output_parent)
    if not output_parent.is_dir():
        raise VerificationError("verification output parent does not exist")
    try:
        output.relative_to(run_directory)
    except ValueError:
        pass
    else:
        raise VerificationError(
            "verification report must be outside the analyzed run directory"
        )
    if output.exists() or output.is_symlink():
        raise VerificationError("verification output already exists")

    completion_payload, completion_snapshot = _capture_file(
        run_directory / "completion.json",
        label="analysis completion",
        context=context,
        keep_bytes=True,
    )
    assert completion_payload is not None
    if completion_snapshot.sha256 != trusted_completion_sha256:
        raise VerificationError(
            "analysis completion does not match the trusted SHA-256"
        )
    dependency_evidence = _load_trusted_scientific_dependencies()
    completion = _parse_json(completion_payload, label="analysis completion")
    _require_exact_keys(
        completion,
        {"schema", "created_utc", "run_manifest", "artifacts"},
        label="analysis completion",
    )
    analysis_contract = _resolve_analysis_contract(completion.get("schema"))
    analysis_contract_current = analysis_contract["current"] is True
    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "summary",
        "per_gene_errors",
    }:
        raise VerificationError("analysis completion artifact inventory is invalid")
    run_manifest_record = completion.get("run_manifest")
    if (
        _record_core(run_manifest_record, label="analysis completion run manifest")[
            "path"
        ]
        != "run-manifest.json"
    ):
        raise VerificationError("analysis completion run-manifest path is invalid")
    expected_artifact_paths = {
        "summary": "summary.json",
        "per_gene_errors": "per-gene-errors.npz",
    }
    for name, expected_path in expected_artifact_paths.items():
        if (
            _record_core(artifacts[name], label=f"analysis completion {name}")["path"]
            != expected_path
        ):
            raise VerificationError(f"analysis completion {name} path is invalid")

    manifest, manifest_snapshot = _load_bound_json(
        run_directory / "run-manifest.json",
        run_manifest_record,
        label="analysis run manifest",
        context=context,
    )
    _require_exact_keys(
        manifest, ANALYSIS_MANIFEST_FIELDS, label="analysis run manifest"
    )
    if manifest.get("schema") != analysis_contract["run_manifest_schema"]:
        raise VerificationError("unexpected analysis run-manifest schema")
    if manifest.get("registered_parameters") != REGISTERED_PARAMETERS:
        raise VerificationError("analysis registered parameters differ from protocol")
    if manifest.get("artifacts") != artifacts:
        raise VerificationError("analysis manifest/completion artifact bindings differ")
    summary, summary_snapshot = _load_bound_json(
        run_directory / "summary.json",
        artifacts["summary"],
        label="analysis summary",
        context=context,
    )
    per_gene_payload, per_gene_snapshot = _verify_record(
        run_directory / "per-gene-errors.npz",
        artifacts["per_gene_errors"],
        label="analysis per-gene arrays",
        context=context,
        keep_bytes=True,
    )
    assert per_gene_payload is not None
    saved_arrays = _load_npz(
        per_gene_payload,
        label="analysis per-gene arrays",
        names=EXPECTED_PER_GENE_ARRAYS,
    )

    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise VerificationError("analysis source block is missing")
    expected_source_keys = {
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
    _require_exact_keys(source, expected_source_keys, label="analysis source")
    analysis_environment_evidence = _validate_analysis_environment(
        source.get("analysis_environment"),
        allow_legacy_inventory=not analysis_contract_current,
    )
    for name, expected_path in CANONICAL_ANALYSIS_INPUTS.items():
        core = _record_core(source.get(name), label=f"analysis source {name}")
        if core["path"] != expected_path:
            raise VerificationError(f"analysis source {name} path is noncanonical")
    for name, expected_path in ANALYSIS_SOURCE_PATHS.items():
        core = _record_core(source.get(name), label=f"analysis source {name}")
        if core["path"] != expected_path:
            raise VerificationError(f"analysis source {name} path is noncanonical")
    if source.get("embedding_array_sha256") != source["embedding_array"]["sha256"]:
        raise VerificationError("embedding array SHA binding is inconsistent")
    for name in (*CANONICAL_ANALYSIS_INPUTS, *ANALYSIS_SOURCE_PATHS):
        record = source[name]
        _verify_record(
            _workspace_path(record["path"], label=f"analysis source {name}"),
            record,
            label=f"analysis source {name}",
            context=context,
        )

    campaign_record = source.get("campaign_report")
    campaign_core = _record_core(campaign_record, label="analysis campaign report")
    campaign_path = _workspace_path(
        campaign_core["path"], label="analysis campaign report"
    )
    campaign_run_paths = source.get("campaign_run_paths")
    if not isinstance(campaign_run_paths, Mapping):
        raise VerificationError("analysis campaign_run_paths must be an object")
    campaign, source_runs, resolved_run_paths, fold_count, fold_evidence = (
        _verify_campaign(
            campaign_path,
            campaign_record,
            run_paths=campaign_run_paths,
            context=context,
        )
    )
    _verification_checkpoint("campaign_validated")

    input_binding = source.get("campaign_input_binding")
    if not isinstance(input_binding, Mapping) or set(input_binding) != {
        "schema",
        "embedding_provenance_sha256",
        "required_inputs",
        "selected_runs",
    }:
        raise VerificationError("analysis campaign input binding is malformed")
    if input_binding.get("schema") != "DepPLM-long-protein-campaign-input-binding-v1":
        raise VerificationError("unexpected campaign input-binding schema")
    required_inputs = input_binding.get("required_inputs")
    if not isinstance(required_inputs, Mapping) or set(required_inputs) != set(
        CANONICAL_ANALYSIS_INPUTS
    ):
        raise VerificationError("campaign required-input binding is incomplete")
    for name in CANONICAL_ANALYSIS_INPUTS:
        if _record_core(
            required_inputs[name], label=f"campaign required input {name}"
        ) != _record_core(source[name], label=f"analysis source {name}"):
            raise VerificationError(f"campaign required input differs for {name}")
    expected_input_runs, expected_selected_runs = _selected_run_bindings(
        source_runs, source["campaign_run_paths"]
    )
    selected_input_runs = input_binding.get("selected_runs")
    if (
        not isinstance(selected_input_runs, list)
        or sorted(
            selected_input_runs,
            key=lambda item: item.get("key", "") if isinstance(item, Mapping) else "",
        )
        != expected_input_runs
    ):
        raise VerificationError("campaign selected-run input bindings differ")
    if source.get("selected_runs") != expected_selected_runs:
        raise VerificationError("analysis selected-run completion bindings differ")

    bound_payloads: dict[str, bytes] = {}
    for name in ("genes", "cells", "fasta", "gene_effect"):
        payload, _ = _verify_record(
            _workspace_path(source[name]["path"], label=f"analysis {name}"),
            source[name],
            label=f"analysis {name}",
            context=context,
            keep_bytes=True,
        )
        assert payload is not None
        bound_payloads[name] = payload
    genes = _load_lines(bound_payloads["genes"], label="genes.txt")
    cells = _load_lines(bound_payloads["cells"], label="cell_lines.txt")
    if len(genes) != EXPECTED_N_GENES:
        raise VerificationError("genes.txt has a noncanonical record count")

    embedding_manifest_payload, _ = _verify_record(
        _workspace_path(
            source["embedding_manifest"]["path"], label="analysis embedding manifest"
        ),
        source["embedding_manifest"],
        label="analysis embedding manifest",
        context=context,
        keep_bytes=True,
    )
    assert embedding_manifest_payload is not None
    embedding_manifest = _parse_json(
        embedding_manifest_payload, label="embedding manifest"
    )
    embedding_evidence = _validate_embedding_bundle(
        embedding_manifest,
        campaign,
        manifest_record=source["embedding_manifest"],
        analysis_embedding_record=source["embedding_array"],
        genes=genes,
        context=context,
    )
    if (
        input_binding.get("embedding_provenance_sha256")
        != embedding_evidence["provenance_sha256"]
    ):
        raise VerificationError(
            "campaign input binding has a different embedding provenance"
        )

    lengths = _load_fasta_lengths(bound_payloads["fasta"], genes=genes)
    targets_gene_by_cell = _load_npy(
        bound_payloads["gene_effect"], label="gene_effect.npy"
    )
    if targets_gene_by_cell.dtype != np.float32 or targets_gene_by_cell.shape != (
        len(genes),
        len(cells),
    ):
        raise VerificationError("gene_effect.npy has an invalid dtype or shape")
    targets = np.asarray(targets_gene_by_cell.T, dtype=np.float32)
    recomputed_arrays, used_runs = _compute_arrays(
        source_runs,
        resolved_run_paths,
        cells=cells,
        genes=genes,
        targets=targets,
        lengths=lengths,
        fold_evidence=fold_evidence,
        context=context,
    )
    recomputed_summary = _summarize_arrays(
        recomputed_arrays, analysis_schema=analysis_contract["analysis_schema"]
    )
    _compare_arrays(recomputed_arrays, saved_arrays)
    if summary != recomputed_summary:
        raise VerificationError(
            "saved summary differs from independently recomputed summary"
        )
    if used_runs != expected_selected_runs:
        raise VerificationError(
            "replay used-run bindings differ from analysis manifest"
        )
    _verification_checkpoint("analysis_recomputed")

    try:
        run_rendered = run_directory.relative_to(Path(os.path.abspath(ROOT))).as_posix()
    except ValueError:
        run_rendered = str(run_directory)
    environment_complete = bool(
        analysis_environment_evidence["required_package_inventory_complete"]
    )
    environment_verdict = _analysis_environment_verdict(
        analysis_environment_evidence,
        analysis_contract_current=analysis_contract_current,
    )
    limitations = [
        "The replay verifies integrity and arithmetic for the registered completed artifacts; it does not establish causal benefit of chunking over truncation.",
        "The verifier hash-verifies the registered raw embeddings and independently recomputes the normalized embedding array, but it does not regenerate every raw embedding from FASTA/model inference.",
        "Inner-validation prediction arrays were not retained. Tuning verification is limited to the hash-bound producer history's schema, semantic ranges, internal consistency, early-stopping trace, and selected-configuration argmax replay; it does not independently recompute inner-validation metrics.",
        "The reported associations remain conditional descriptive estimates without IID confidence intervals or population p-values.",
        "A verifier cannot prove that upstream data-generation code faithfully represented phenomena not encoded in the bound artifacts.",
    ]
    if not environment_complete:
        missing_packages = ", ".join(
            analysis_environment_evidence["missing_recorded_packages"]
        )
        limitations.append(
            "The registered legacy analysis_environment records Python, platform, "
            "NumPy, and SciPy but omits producer versions for "
            f"{missing_packages}; current verifier versions are reported only for "
            "comparison and are not retroactive producer provenance."
        )
    if not analysis_contract_current:
        limitations.append(
            "This is a diagnostic replay of the registered v2 analysis contract. "
            "Legacy contracts are permanently ineligible for public release."
        )
    dependency_evidence = _revalidate_trusted_scientific_dependencies()
    body: dict[str, Any] = {
        "schema": VERIFICATION_SCHEMA,
        "created_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "verification_passed": environment_verdict["verification_passed"],
        "verification_status": environment_verdict["verification_status"],
        "release_ready": environment_verdict["release_ready"],
        "scientific_result": False,
        "scientific_result_reason": (
            "This file independently verifies a previously completed conditional "
            "descriptive analysis; it is not itself a new scientific result."
        ),
        "analysis_contract": analysis_contract,
        "verified_analysis": {
            "run_directory": run_rendered,
            "trusted_completion_sha256": trusted_completion_sha256,
            "completion": _snapshot_record(completion_snapshot),
            "run_manifest": _snapshot_record(manifest_snapshot),
            "summary": _snapshot_record(summary_snapshot),
            "per_gene_errors": _snapshot_record(per_gene_snapshot),
        },
        "verification_gates": {
            "analysis_contract_current": environment_verdict[
                "analysis_contract_current"
            ],
            "trusted_completion_hash": True,
            "trusted_scientific_dependency_origins_and_record_hashes": True,
            "analysis_inventory_and_hashes": True,
            "campaign_99_run_matrix": True,
            "campaign_completion_and_root_hashes": True,
            "campaign_693_fold_artifacts": True,
            "all_fold_split_result_prediction_bindings": True,
            "transfer_and_p1_topology_recomputed": True,
            "all_declared_run_inputs_and_code": True,
            "raw_csv_hash_and_preprocessing_parse_bound_to_same_private_snapshots": True,
            "analysis_environment_recorded_versions_match_current_runtime": (
                environment_verdict["exact_environment_gate"]
            ),
            "embedding_manifest_model_tokenizer_environment_and_snapshot": True,
            "embedding_raw_hash_verified_and_normalized_array_recomputed": True,
            "embedding_manifest_campaign_analysis_binding": True,
            "tuning_history_schema_ranges_consistency_and_argmax_replayed": True,
            "outer_test_predictions_exactly_once": True,
            "per_gene_arrays_recomputed": True,
            "summary_recomputed": True,
            "no_observed_mid_verification_mutation": True,
        },
        "verified_counts": {
            "campaign_runs": len(source_runs),
            "campaign_fold_artifacts": fold_count,
            "selected_p1_runs": len(used_runs),
            "selected_prediction_files": len(used_runs) * len(FOLDS),
            "genes": len(genes),
            "long_genes": int(np.sum(lengths > LONG_THRESHOLD)),
            "cells": len(cells),
        },
        "embedding_evidence": embedding_evidence,
        "analysis_environment_evidence": analysis_environment_evidence,
        "tuning_evidence": {
            "scope": "hash_bound_producer_history_only",
            "inner_validation_prediction_arrays_available": False,
            "inner_validation_metrics_independently_recomputed": False,
            "checks": [
                "exact_schema",
                "semantic_metric_ranges",
                "history_internal_consistency",
                "early_stopping_trace",
                "selected_configuration_argmax_replay",
            ],
        },
        "recomputed_summary": recomputed_summary,
        "recomputed_summary_sha256": _canonical_sha256(recomputed_summary),
        "verifier": {
            "source": _snapshot_record(verifier_snapshot),
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy": np.__version__,
            "scipy": sys.modules["scipy"].__version__,
            "trusted_scientific_dependencies": dependency_evidence,
        },
        "limitations": limitations,
    }
    report = dict(body)
    report["verification_body_sha256"] = _canonical_sha256(body)
    _revalidate_trusted_scientific_dependencies()
    _assert_snapshots_unchanged(context)
    output_record = _write_report_once(output, report)
    return report, output_record


def _same_file_object(left: FileIdentity, right: FileIdentity) -> bool:
    return (
        left.device == right.device
        and left.inode == right.inode
        and stat.S_ISREG(left.mode)
        and stat.S_ISREG(right.mode)
    )


def _posix_link_open_file(
    descriptor: int, destination_directory_descriptor: int, destination_name: str
) -> None:
    """Link the inode held by *descriptor*, never its replaceable leaf name."""
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    linkat = getattr(libc, "linkat", None)
    if linkat is None:
        raise VerificationError("descriptor-bound hard linking is unavailable")
    linkat.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    )
    linkat.restype = ctypes.c_int
    destination = os.fsencode(destination_name)
    if (
        linkat(
            descriptor,
            b"",
            destination_directory_descriptor,
            destination,
            0x1000,  # AT_EMPTY_PATH
        )
        == 0
    ):
        return
    last_error = ctypes.get_errno()
    if last_error == errno.EEXIST:
        raise FileExistsError(last_error, os.strerror(last_error), destination_name)
    if last_error not in {
        errno.EACCES,
        errno.EINVAL,
        errno.ENOENT,
        errno.ENOTSUP,
        errno.EOPNOTSUPP,
        errno.EPERM,
    }:
        raise OSError(last_error, os.strerror(last_error), destination_name)
    raise VerificationError(
        "the filesystem cannot publish an anonymous report with "
        "linkat(AT_EMPTY_PATH)"
    ) from OSError(last_error, os.strerror(last_error), destination_name)


def _same_directory_object(left: FileIdentity, right: FileIdentity) -> bool:
    return (
        left.device == right.device
        and left.inode == right.inode
        and stat.S_ISDIR(left.mode)
        and stat.S_ISDIR(right.mode)
    )


def _windows_directory_path_from_descriptor(descriptor: int) -> Path:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    get_final_path = ctypes.WinDLL(
        "kernel32", use_last_error=True
    ).GetFinalPathNameByHandleW
    get_final_path.argtypes = (
        wintypes.HANDLE,
        wintypes.LPWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
    )
    get_final_path.restype = wintypes.DWORD
    handle = wintypes.HANDLE(msvcrt.get_osfhandle(descriptor))
    required = get_final_path(handle, None, 0, 0)
    if required == 0:
        error = ctypes.get_last_error()
        raise VerificationError(
            "cannot resolve the bound report output directory"
        ) from OSError(error, ctypes.FormatError(error))
    buffer = ctypes.create_unicode_buffer(required + 1)
    written = get_final_path(handle, buffer, len(buffer), 0)
    if written == 0 or written >= len(buffer):
        error = ctypes.get_last_error()
        raise VerificationError(
            "cannot resolve the bound report output directory"
        ) from OSError(error, ctypes.FormatError(error))
    rendered = buffer.value
    if rendered.startswith("\\\\?\\UNC\\"):
        rendered = "\\\\" + rendered[8:]
    elif rendered.startswith("\\\\?\\"):
        rendered = rendered[4:]
    return Path(rendered)


def _open_report_directory(path: Path) -> BoundReportDirectory:
    requested = Path(os.path.abspath(path))
    _check_path_components(requested)
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = (
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        )
        create_file.restype = wintypes.HANDLE
        handle_guard: _WindowsHandleAcquisitionGuard | None = None
        windows_descriptor: _OwnedDescriptor | None = None
        try:
            handle_guard = _owned_windows_handle(
                create_file,
                str(requested),
                0x00000080,  # FILE_READ_ATTRIBUTES
                0x00000001 | 0x00000002,  # share read/write, deny delete/rename
                None,
                3,  # OPEN_EXISTING
                0x02000000 | 0x00200000,  # BACKUP_SEMANTICS | OPEN_REPARSE_POINT
                None,
            )
            invalid = ctypes.c_void_p(-1).value
            handle = handle_guard.handle
            if handle == invalid:
                handle_guard.handle = None
                error = ctypes.get_last_error()
                raise VerificationError(
                    "cannot bind the report output directory"
                ) from OSError(error, ctypes.FormatError(error), str(requested))
            windows_descriptor = _adopt_windows_handle_as_descriptor(
                handle_guard, os.O_RDONLY
            )
            identity = _identity(os.fstat(windows_descriptor))
            current = _identity(requested.lstat())
            if not _same_directory_object(identity, current):
                raise VerificationError("report output directory identity changed")
            operation_path = _windows_directory_path_from_descriptor(windows_descriptor)
        except BaseException as primary:
            if windows_descriptor is not None:
                _close_owned_descriptor(windows_descriptor, preserve=primary)
            elif handle_guard is not None:
                handle_guard.close(preserve=primary)
            raise
        assert windows_descriptor is not None
        return BoundReportDirectory(
            requested_path=requested,
            operation_path=operation_path,
            descriptor=windows_descriptor,
            identity=identity,
            supports_dir_fd=False,
        )

    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise VerificationError("secure output-directory binding is unavailable")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY") | getattr(os, "O_NOFOLLOW")
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor: _OwnedDescriptor | None = None
    last_error: OSError | None = None
    for _ in range(3):
        try:
            descriptor = _owned_os_open(requested, flags)
            opened = _identity(os.fstat(descriptor))
            current = _identity(requested.lstat())
            break
        except BaseException as exc:
            if descriptor is not None:
                _close_owned_descriptor(descriptor, preserve=exc)
                descriptor = None
            if not isinstance(exc, OSError):
                raise
            last_error = exc
    if descriptor is None:
        assert last_error is not None
        raise VerificationError(
            "cannot bind the report output directory"
        ) from last_error
    if not _same_directory_object(opened, current):
        identity_error = VerificationError("report output directory identity changed")
        _close_owned_descriptor(descriptor, preserve=identity_error)
        raise identity_error
    return BoundReportDirectory(
        requested_path=requested,
        operation_path=requested,
        descriptor=descriptor,
        identity=opened,
        supports_dir_fd=True,
    )


def _report_directory_parent(
    directory: BoundReportDirectory, *, require_requested_identity: bool
) -> Path:
    held = _identity(os.fstat(directory.descriptor))
    if not _same_directory_object(held, directory.identity):
        raise VerificationError("bound report output directory identity changed")
    if require_requested_identity:
        try:
            requested = _identity(directory.requested_path.lstat())
        except OSError as exc:
            raise VerificationError("report output directory path disappeared") from exc
        if not _same_directory_object(requested, directory.identity):
            raise VerificationError("report output directory path identity changed")
    if os.name == "nt":
        return _windows_directory_path_from_descriptor(directory.descriptor)
    return directory.operation_path


def _create_report_temporary(
    path: Path, directory: BoundReportDirectory
) -> tuple[_OwnedDescriptor, Path | None]:
    parent = _report_directory_parent(directory, require_requested_identity=True)
    if directory.supports_dir_fd:
        return (
            _create_posix_anonymous_file(
                directory, label="verification-report publication"
            ),
            None,
        )
    for _ in range(100):
        temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
        descriptor: _OwnedDescriptor | None = None
        try:
            descriptor = _create_exclusive_bound_file(parent / temporary.name)
            return descriptor, temporary
        except FileExistsError as exc:
            if descriptor is not None:
                _close_owned_descriptor(descriptor, preserve=exc)
                raise
            continue
        except OSError as exc:
            if descriptor is not None:
                _close_owned_descriptor(descriptor, preserve=exc)
                raise
            raise VerificationError(
                f"cannot create private verification-report temporary file: {path}"
            ) from exc
        except BaseException as primary:
            if descriptor is not None:
                _close_owned_descriptor(descriptor, preserve=primary)
            raise
    raise VerificationError(
        f"cannot allocate a unique verification-report temporary file: {path}"
    )


def _link_open_report(
    descriptor: int,
    temporary_path: Path | None,
    destination: Path,
    directory: BoundReportDirectory,
) -> None:
    parent = _report_directory_parent(directory, require_requested_identity=True)
    if directory.supports_dir_fd:
        _posix_link_open_file(descriptor, directory.descriptor, destination.name)
        os.fsync(directory.descriptor)
        return
    if temporary_path is None:
        raise VerificationError("Windows report publication temporary is absent")
    os.link(parent / temporary_path.name, parent / destination.name)


def _report_entry_stat(
    parent: Path, directory_descriptor: int | None, name: str
) -> FileIdentity | None:
    last_error: OSError | None = None
    for _ in range(3):
        try:
            info = (
                (parent / name).lstat()
                if directory_descriptor is None
                else os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
            )
            return _identity(info)
        except FileNotFoundError:
            return None
        except OSError as exc:
            last_error = exc
    assert last_error is not None
    raise VerificationError(
        f"cannot inspect report entry {parent / name}"
    ) from last_error


def _retry_report_cleanup(action: Any, *, label: str) -> Any:
    last_error: OSError | None = None
    for _ in range(3):
        try:
            return action()
        except FileNotFoundError:
            raise
        except FileExistsError:
            raise
        except OSError as exc:
            last_error = exc
    assert last_error is not None
    raise VerificationError(f"cannot {label} after three attempts") from last_error


def _move_report_entry_no_clobber(
    parent: Path,
    directory_descriptor: int | None,
    source: str,
    destination: str,
) -> None:
    """Atomically move an entry without replacing an existing destination."""
    if os.name == "nt":
        try:
            os.rename(parent / source, parent / destination)
        except OSError as exc:
            if exc.errno == errno.EEXIST or getattr(exc, "winerror", None) in {
                80,
                183,
            }:
                raise FileExistsError(
                    errno.EEXIST,
                    os.strerror(errno.EEXIST),
                    str(parent / destination),
                ) from exc
            raise
        return

    if directory_descriptor is None:
        raise VerificationError("secure no-clobber quarantine is unavailable")

    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise VerificationError("secure no-clobber quarantine is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    if (
        renameat2(
            directory_descriptor,
            os.fsencode(source),
            directory_descriptor,
            os.fsencode(destination),
            1,  # RENAME_NOREPLACE
        )
        == 0
    ):
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(error, os.strerror(error), destination)
    if error == errno.ENOENT:
        raise FileNotFoundError(error, os.strerror(error), source)
    if error in {errno.EINVAL, errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP}:
        raise VerificationError(
            "the filesystem does not support no-clobber quarantine moves"
        ) from OSError(error, os.strerror(error), destination)
    raise OSError(error, os.strerror(error), destination)


def _move_to_unique_quarantine(
    parent: Path,
    directory_descriptor: int | None,
    source: str,
    expected: FileIdentity,
    *,
    suffix: str,
    label: str,
    is_directory: bool,
) -> tuple[str, tuple[int, int, str, str, int, int, bool, str]]:
    """Register, then move *source* to a fresh no-clobber quarantine."""
    for _ in range(100):
        destination = f".{source}.{uuid.uuid4().hex}.{suffix}"
        pending_token = _owned_quarantine_token(
            parent,
            directory_descriptor,
            source,
            destination,
            expected,
            is_directory=is_directory,
            state="pending",
        )
        _OWNED_QUARANTINE_ENTRIES.add(pending_token)
        last_error: OSError | None = None
        destination_collision = False
        for attempt in range(3):
            try:
                _move_report_entry_no_clobber(
                    parent, directory_descriptor, source, destination
                )
            except OSError as move_error:
                last_error = move_error
                try:
                    reconciled = _reconcile_pending_quarantine(
                        parent,
                        directory_descriptor,
                        pending_token,
                        expected,
                        is_directory=is_directory,
                    )
                except BaseException as reconciliation_error:
                    move_error.add_note(
                        "quarantine move reconciliation failed: "
                        f"{reconciliation_error!r}"
                    )
                    raise move_error
                if reconciled is not None and reconciled[7] == "moved":
                    return destination, reconciled
                if reconciled == pending_token:
                    raise VerificationError(
                        "quarantine move outcome is ambiguous; all entries were "
                        "preserved"
                    ) from move_error
                if isinstance(move_error, FileExistsError):
                    destination_collision = True
                    break
                if attempt == 2:
                    raise VerificationError(
                        f"cannot {label} after three attempts"
                    ) from move_error
                # Reconciliation proved that the original source is still in
                # place. Re-register ownership before the next rename attempt.
                _OWNED_QUARANTINE_ENTRIES.add(pending_token)
            except BaseException as primary:
                try:
                    _reconcile_pending_quarantine(
                        parent,
                        directory_descriptor,
                        pending_token,
                        expected,
                        is_directory=is_directory,
                    )
                except BaseException as reconciliation_error:
                    primary.add_note(
                        "quarantine move reconciliation failed: "
                        f"{reconciliation_error!r}"
                    )
                raise
            else:
                moved_token = _quarantine_token_with_state(pending_token, "moved")
                _OWNED_QUARANTINE_ENTRIES.add(moved_token)
                _OWNED_QUARANTINE_ENTRIES.discard(pending_token)
                return destination, moved_token
        if destination_collision:
            continue
        assert last_error is not None
        raise VerificationError(f"cannot {label} after three attempts") from last_error
    raise VerificationError(f"cannot allocate a collision-free quarantine for {source}")


def _entry_matches_identity(
    actual: FileIdentity, expected: FileIdentity, *, is_directory: bool
) -> bool:
    if is_directory:
        return _same_directory_object(actual, expected)
    return _same_file_object(actual, expected)


def _quarantine_parent_identity(
    parent: Path, directory_descriptor: int | None
) -> FileIdentity:
    info = (
        parent.lstat()
        if directory_descriptor is None
        else os.fstat(directory_descriptor)
    )
    identity = _identity(info)
    if not stat.S_ISDIR(identity.mode):
        raise VerificationError("quarantine parent is no longer a directory")
    return identity


def _owned_quarantine_token(
    parent: Path,
    directory_descriptor: int | None,
    source: str,
    quarantine: str,
    expected: FileIdentity,
    *,
    is_directory: bool,
    state: str,
) -> tuple[int, int, str, str, int, int, bool, str]:
    if state not in {"pending", "moved"}:
        raise VerificationError("invalid quarantine ownership state")
    parent_identity = _quarantine_parent_identity(parent, directory_descriptor)
    return (
        parent_identity.device,
        parent_identity.inode,
        source,
        quarantine,
        expected.device,
        expected.inode,
        is_directory,
        state,
    )


def _quarantine_token_with_state(
    token: tuple[int, int, str, str, int, int, bool, str], state: str
) -> tuple[int, int, str, str, int, int, bool, str]:
    if state not in {"pending", "moved"}:
        raise VerificationError("invalid quarantine ownership state")
    return (
        token[0],
        token[1],
        token[2],
        token[3],
        token[4],
        token[5],
        token[6],
        state,
    )


def _reconcile_pending_quarantine(
    parent: Path,
    directory_descriptor: int | None,
    token: tuple[int, int, str, str, int, int, bool, str],
    expected: FileIdentity,
    *,
    is_directory: bool,
) -> tuple[int, int, str, str, int, int, bool, str] | None:
    """Advance or revoke a pending token only when the outcome is unambiguous."""
    if token[7] != "pending":
        return token
    source = _report_entry_stat(parent, directory_descriptor, token[2])
    quarantine = _report_entry_stat(parent, directory_descriptor, token[3])
    source_matches = source is not None and _entry_matches_identity(
        source, expected, is_directory=is_directory
    )
    quarantine_matches = quarantine is not None and _entry_matches_identity(
        quarantine, expected, is_directory=is_directory
    )
    if quarantine_matches and not source_matches:
        moved_token = _quarantine_token_with_state(token, "moved")
        _OWNED_QUARANTINE_ENTRIES.add(moved_token)
        _OWNED_QUARANTINE_ENTRIES.discard(token)
        return moved_token
    if source_matches and not quarantine_matches:
        _OWNED_QUARANTINE_ENTRIES.discard(token)
        return None
    if source is None and quarantine is None:
        # Absence at both names does not prove whether an interrupted rename
        # moved the owned inode and a competing actor removed it. Keep the
        # pending recovery token and fail closed.
        return token
    return token


def _matching_quarantine_names(
    parent: Path,
    directory_descriptor: int | None,
    source: str,
    *,
    suffix: str,
) -> set[str]:
    prefix = f".{source}."
    ending = f".{suffix}"
    try:
        names = os.listdir(
            parent if directory_descriptor is None else directory_descriptor
        )
    except OSError as exc:
        raise VerificationError(
            f"cannot enumerate quarantine entries below {parent}"
        ) from exc
    return {
        name
        for name in names
        if isinstance(name, str) and name.startswith(prefix) and name.endswith(ending)
    }


def _resume_owned_quarantines(
    parent: Path,
    directory_descriptor: int | None,
    source: str,
    expected: FileIdentity,
    *,
    suffix: str,
    is_directory: bool,
) -> bool:
    """Resume only quarantines registered by this verifier process."""
    parent_identity = _quarantine_parent_identity(parent, directory_descriptor)
    prefix = (
        parent_identity.device,
        parent_identity.inode,
        source,
    )
    suffix_key = (expected.device, expected.inode, is_directory)
    registered = {
        token
        for token in _OWNED_QUARANTINE_ENTRIES
        if token[:3] == prefix and token[4:7] == suffix_key
    }
    for token in sorted(registered):
        if token[7] != "pending":
            continue
        reconciled = _reconcile_pending_quarantine(
            parent,
            directory_descriptor,
            token,
            expected,
            is_directory=is_directory,
        )
        if reconciled == token:
            raise VerificationError(
                "pending quarantine move outcome is ambiguous; all entries were "
                "preserved"
            )

    registered = {
        token
        for token in _OWNED_QUARANTINE_ENTRIES
        if token[:3] == prefix and token[4:7] == suffix_key and token[7] == "moved"
    }
    resumed = bool(registered)
    for token in sorted(registered):
        quarantine = token[3]
        current = _report_entry_stat(parent, directory_descriptor, quarantine)
        if current is None:
            _OWNED_QUARANTINE_ENTRIES.discard(token)
            continue
        if not _entry_matches_identity(current, expected, is_directory=is_directory):
            raise VerificationError(
                "registered quarantine identity changed; competing entry was preserved"
            )
        _retry_report_cleanup(
            lambda quarantine=quarantine: _remove_quarantined_report_entry(
                parent,
                directory_descriptor,
                quarantine,
                expected,
                is_directory=is_directory,
            ),
            label=f"remove registered quarantine {parent / quarantine}",
        )
        _OWNED_QUARANTINE_ENTRIES.discard(token)

    registered_names = {
        token[3]
        for token in _OWNED_QUARANTINE_ENTRIES
        if token[:3] == prefix and token[4:7] == suffix_key
    }
    for quarantine in sorted(
        _matching_quarantine_names(
            parent,
            directory_descriptor,
            source,
            suffix=suffix,
        )
        - registered_names
    ):
        current = _report_entry_stat(parent, directory_descriptor, quarantine)
        if current is not None and _entry_matches_identity(
            current, expected, is_directory=is_directory
        ):
            raise VerificationError(
                "matching quarantine residue cannot be proven verifier-owned; "
                "it was preserved"
            )
    return resumed


def _remove_windows_quarantined_entry(
    path: Path, expected: FileIdentity, *, is_directory: bool
) -> None:
    """Delete a Windows entry through the same identity-checked open HANDLE."""
    import ctypes
    import msvcrt
    from ctypes import wintypes

    class FileDispositionInfo(ctypes.Structure):
        _fields_ = (("delete_file", ctypes.c_ubyte),)

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    flags = 0x00200000  # FILE_FLAG_OPEN_REPARSE_POINT
    if is_directory:
        flags |= 0x02000000  # FILE_FLAG_BACKUP_SEMANTICS
    handle_guard: _WindowsHandleAcquisitionGuard | None = None
    descriptor: _OwnedDescriptor | None = None
    try:
        handle_guard = _owned_windows_handle(
            create_file,
            str(path),
            0x00010000 | 0x00000080,  # DELETE | FILE_READ_ATTRIBUTES
            0x00000001 | 0x00000002,  # share read/write; deny delete/rename
            None,
            3,  # OPEN_EXISTING
            flags,
            None,
        )
        invalid = ctypes.c_void_p(-1).value
        handle = handle_guard.handle
        if handle == invalid:
            handle_guard.handle = None
            error = ctypes.get_last_error()
            if error in {2, 3}:
                raise FileNotFoundError(error, ctypes.FormatError(error), str(path))
            raise OSError(error, ctypes.FormatError(error), str(path))
        descriptor = _adopt_windows_handle_as_descriptor(
            handle_guard, os.O_RDONLY | getattr(os, "O_BINARY", 0)
        )
        opened = _identity(os.fstat(descriptor))
        if not _entry_matches_identity(opened, expected, is_directory=is_directory):
            raise VerificationError(
                "quarantined entry identity changed before handle-bound removal"
            )

        set_information = kernel32.SetFileInformationByHandle
        set_information.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        )
        set_information.restype = wintypes.BOOL
        disposition = FileDispositionInfo(1)
        if not set_information(
            wintypes.HANDLE(msvcrt.get_osfhandle(descriptor)),
            4,  # FileDispositionInfo
            ctypes.byref(disposition),
            ctypes.sizeof(disposition),
        ):
            error = ctypes.get_last_error()
            raise OSError(error, ctypes.FormatError(error), str(path))
    except BaseException as primary:
        if descriptor is not None:
            _close_owned_descriptor(descriptor, preserve=primary)
        elif handle_guard is not None:
            handle_guard.close(preserve=primary)
        raise
    else:
        assert descriptor is not None
        _close_owned_descriptor(descriptor)


def _remove_posix_quarantined_entry(
    parent: Path,
    directory_descriptor: int | None,
    name: str,
    expected: FileIdentity,
    *,
    is_directory: bool,
) -> None:
    """Refuse a POSIX name-based deletion that cannot be inode-conditional."""
    if directory_descriptor is None:
        raise VerificationError("secure descriptor-relative cleanup is unavailable")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    if is_directory:
        flags |= getattr(os, "O_DIRECTORY", 0)
    descriptor: int | None = None
    try:
        descriptor = _owned_os_open(name, flags, dir_fd=directory_descriptor)
        before = os.fstat(descriptor)
        opened = _identity(before)
        if not _entry_matches_identity(opened, expected, is_directory=is_directory):
            raise VerificationError(
                "quarantined entry identity changed before descriptor-bound removal"
            )
        named = _report_entry_stat(parent, directory_descriptor, name)
        if named is None or not _entry_matches_identity(
            named, opened, is_directory=is_directory
        ):
            raise VerificationError(
                "quarantined entry identity changed before descriptor-bound removal"
            )
        # POSIX unlinkat/rmdir take a directory descriptor plus a replaceable
        # leaf name; they do not accept an expected inode.  The name can be
        # exchanged after the checks above and immediately before the syscall,
        # so a path-based deletion could remove a competing entry.  Linux also
        # rejects unlinkat(fd, "", AT_EMPTY_PATH).  Preserve the quarantine
        # rather than claim an identity guarantee the kernel cannot provide.
        raise VerificationError(
            "atomic identity-bound POSIX quarantine removal is unavailable; "
            "the owned quarantined entry was preserved"
        )
    except BaseException as primary:
        if descriptor is not None:
            _close_owned_descriptor(descriptor, preserve=primary)
        raise
    else:
        assert descriptor is not None
        _close_owned_descriptor(descriptor)


def _remove_quarantined_report_entry(
    parent: Path,
    directory_descriptor: int | None,
    name: str,
    expected: FileIdentity,
    *,
    is_directory: bool,
) -> None:
    """Remove only the identity-bound entry at a private quarantine name."""
    if os.name == "nt":
        _remove_windows_quarantined_entry(
            parent / name, expected, is_directory=is_directory
        )
    else:
        _remove_posix_quarantined_entry(
            parent,
            directory_descriptor,
            name,
            expected,
            is_directory=is_directory,
        )
    remaining = _report_entry_stat(parent, directory_descriptor, name)
    if remaining is None:
        return
    if _entry_matches_identity(remaining, expected, is_directory=is_directory):
        raise VerificationError("owned quarantined entry remains present")
    raise VerificationError(
        "quarantine path was replaced after removal; replacement was preserved"
    )


def _restore_isolated_report_entry(
    parent: Path,
    directory_descriptor: int | None,
    isolated_name: str,
    destination_name: str,
    isolated_identity: FileIdentity,
) -> None:
    destination_identity = _report_entry_stat(
        parent, directory_descriptor, destination_name
    )
    if destination_identity is not None:
        raise VerificationError(
            "cannot restore an identity-mismatched report entry without "
            "overwriting a competing destination"
        )
    try:
        _retry_report_cleanup(
            lambda: _move_report_entry_no_clobber(
                parent,
                directory_descriptor,
                isolated_name,
                destination_name,
            ),
            label="restore an identity-mismatched report entry",
        )
    except FileExistsError as exc:
        raise VerificationError(
            "cannot restore an identity-mismatched report entry without "
            "overwriting a competing destination"
        ) from exc
    destination_identity = _report_entry_stat(
        parent, directory_descriptor, destination_name
    )
    if destination_identity is None or not _same_file_object(
        destination_identity, isolated_identity
    ):
        raise VerificationError("cannot safely restore an identity-mismatched entry")
    if _report_entry_stat(parent, directory_descriptor, isolated_name) is not None:
        raise VerificationError("restored report isolation entry remains present")


def _rollback_identity_bound_entry(
    directory: BoundReportDirectory, name: str, identity: FileIdentity
) -> None:
    """Remove only *identity* from the held directory, isolating before unlink."""
    parent = _report_directory_parent(directory, require_requested_identity=False)
    directory_descriptor = directory.descriptor if directory.supports_dir_fd else None
    display_path = directory.requested_path / name
    _resume_owned_quarantines(
        parent,
        directory_descriptor,
        name,
        identity,
        suffix="rollback",
        is_directory=False,
    )
    for _ in range(3):
        try:
            current = _retry_report_cleanup(
                lambda: _report_entry_stat(parent, directory_descriptor, name),
                label=f"inspect report rollback target {display_path}",
            )
        except FileNotFoundError:
            current = None
        if current is None or not _same_file_object(current, identity):
            return
        try:
            isolated_name, quarantine_token = _move_to_unique_quarantine(
                parent,
                directory_descriptor,
                name,
                identity,
                suffix="rollback",
                label=f"isolate report rollback target {display_path}",
                is_directory=False,
            )
        except FileNotFoundError:
            continue
        isolated_identity = _report_entry_stat(
            parent, directory_descriptor, isolated_name
        )
        if isolated_identity is None:
            raise VerificationError("isolated report entry disappeared")
        if not _same_file_object(isolated_identity, identity):
            _restore_isolated_report_entry(
                parent,
                directory_descriptor,
                isolated_name,
                name,
                isolated_identity,
            )
            raise VerificationError(
                "report rollback target identity changed during isolation"
            )
        try:
            _retry_report_cleanup(
                lambda: _remove_quarantined_report_entry(
                    parent,
                    directory_descriptor,
                    isolated_name,
                    identity,
                    is_directory=False,
                ),
                label=f"remove isolated report rollback target {display_path}",
            )
        except BaseException:
            raise
        else:
            _OWNED_QUARANTINE_ENTRIES.discard(quarantine_token)
        if _report_entry_stat(parent, directory_descriptor, isolated_name) is not None:
            raise VerificationError("isolated report rollback target remains present")
        remaining = _report_entry_stat(parent, directory_descriptor, name)
        if remaining is None or not _same_file_object(remaining, identity):
            return
    raise VerificationError(f"cannot complete report rollback for {display_path}")


def _rollback_identity_bound_path(path: Path, identity: FileIdentity) -> None:
    """Compatibility wrapper for cleanup outside an active report transaction."""
    directory: BoundReportDirectory | None = None
    try:
        directory = _open_report_directory(path.parent)
        _rollback_identity_bound_entry(directory, path.name, identity)
    finally:
        if directory is not None:
            _close_owned_descriptor(directory.descriptor, preserve=sys.exc_info()[1])


def _rollback_report_paths(
    directory: BoundReportDirectory,
    path: Path,
    temporary_path: Path | None,
    identity: FileIdentity,
) -> None:
    if directory.supports_dir_fd:
        # POSIX publication gives the anonymous inode its first name directly
        # from the still-open descriptor.  After that atomic link succeeds,
        # pathname inspection cannot authorize unlinking: a competitor may
        # replace the leaf between any identity check and a later unlink.  A
        # failed post-publication verification therefore preserves whichever
        # entry is present and reports the incomplete transaction.
        current = _report_entry_stat(
            _report_directory_parent(directory, require_requested_identity=False),
            directory.descriptor,
            path.name,
        )
        if current is None:
            return
        if _same_file_object(current, identity):
            raise VerificationError(
                "POSIX verification-report rollback preserved the published entry"
            )
        raise VerificationError(
            "POSIX verification-report rollback preserved a competing replacement"
        )
    errors: list[BaseException] = []
    for candidate in (path, temporary_path):
        if candidate is None:
            continue
        try:
            _rollback_identity_bound_entry(directory, candidate.name, identity)
        except BaseException as exc:
            errors.append(exc)
    if errors:
        raise VerificationError(
            "verification-report cleanup did not complete"
        ) from errors[0]


def _open_bound_report_entry(
    directory: BoundReportDirectory, name: str, *, label: str
) -> tuple[_OwnedDescriptor, FileIdentity]:
    parent = _report_directory_parent(directory, require_requested_identity=True)
    if not directory.supports_dir_fd:
        return _open_stable(parent / name, label=label)
    try:
        before = os.stat(name, dir_fd=directory.descriptor, follow_symlinks=False)
    except OSError as exc:
        raise VerificationError(f"cannot stat {label}: {parent / name}") from exc
    if _is_link_like(before) or not stat.S_ISREG(before.st_mode):
        raise VerificationError(
            f"{label} is not a regular non-link file: {parent / name}"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: _OwnedDescriptor | None = None
    try:
        descriptor = _owned_os_open(name, flags, dir_fd=directory.descriptor)
        opened = _identity(os.fstat(descriptor))
    except BaseException as exc:
        if descriptor is not None:
            _close_owned_descriptor(descriptor, preserve=exc)
        if isinstance(exc, OSError):
            raise VerificationError(f"cannot open {label}: {parent / name}") from exc
        raise
    assert descriptor is not None
    before_identity = _identity(before)
    if before_identity != opened:
        error = VerificationError(
            f"{label} changed while being opened: {parent / name}"
        )
        _close_owned_descriptor(descriptor, preserve=error)
        raise error
    return descriptor, opened


def _write_report_once(path: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    payload = (
        json.dumps(
            report,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
        + b"\n"
    )
    _check_path_components(path.parent)
    directory: BoundReportDirectory | None = None
    descriptor: _OwnedDescriptor | None = None
    temporary_path: Path | None = None
    descriptor_identity: FileIdentity | None = None
    published_descriptor: _OwnedDescriptor | None = None
    result: dict[str, Any] | None = None
    try:
        directory = _open_report_directory(path.parent)
        descriptor, temporary_path = _create_report_temporary(path, directory)
        descriptor_identity = _identity(os.fstat(descriptor))
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short write while creating verification report")
            offset += written
        os.fsync(descriptor)
        before_digest, before_size, descriptor_identity = _hash_open_descriptor(
            descriptor
        )
        if (
            before_size != len(payload)
            or before_digest != hashlib.sha256(payload).hexdigest()
        ):
            raise VerificationError(
                "verification-report temporary descriptor content changed"
            )
        try:
            _link_open_report(descriptor, temporary_path, path, directory)
        except FileExistsError as exc:
            raise VerificationError("verification output already exists") from exc
        except VerificationError:
            raise
        except OSError as exc:
            raise VerificationError(
                f"cannot atomically publish verification output: {path}"
            ) from exc
        after_digest, after_size, after_identity = _hash_open_descriptor(descriptor)
        if (
            not _same_file_object(descriptor_identity, after_identity)
            or after_size != before_size
            or after_digest != before_digest
        ):
            raise VerificationError(
                "verification-report descriptor changed during publication"
            )
        descriptor_identity = after_identity
        try:
            published_descriptor, published_identity = _open_bound_report_entry(
                directory, path.name, label="published verification output"
            )
            output_digest, output_size, output_closed_identity = _hash_open_descriptor(
                published_descriptor
            )
        except BaseException as primary:
            if published_descriptor is not None:
                _close_owned_descriptor(published_descriptor, preserve=primary)
                published_descriptor = None
            raise
        else:
            _close_owned_descriptor(published_descriptor)
            published_descriptor = None
        if (
            published_identity != output_closed_identity
            or not _same_file_object(descriptor_identity, published_identity)
            or output_size != after_size
            or output_digest != after_digest
        ):
            raise VerificationError(
                "published verification output is not bound to the report descriptor"
            )
        result = {
            "path": str(path),
            "bytes": output_size,
            "sha256": output_digest,
        }
    except BaseException as primary:
        cleanup_errors: list[BaseException] = []
        if published_descriptor is not None:
            try:
                _close_owned_descriptor(published_descriptor, preserve=primary)
            except BaseException as close_error:
                cleanup_errors.append(close_error)
            finally:
                published_descriptor = None
        if descriptor is not None and descriptor_identity is None:
            # A cleanup-time BaseException must not prevent descriptor closure.
            # Two bounded recovery reads allow one failing recovery attempt while
            # still obtaining an identity for path-safe rollback.
            for _ in range(2):
                try:
                    descriptor_identity = _identity(os.fstat(descriptor))
                except BaseException as recovery_error:
                    cleanup_errors.append(recovery_error)
                else:
                    break
        if descriptor is not None:
            try:
                _close_owned_descriptor(descriptor, preserve=primary)
            except BaseException as close_error:
                cleanup_errors.append(close_error)
            finally:
                descriptor = None
        if directory is not None and descriptor_identity is not None:
            try:
                _rollback_report_paths(
                    directory, path, temporary_path, descriptor_identity
                )
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        elif temporary_path is not None:
            cleanup_errors.append(
                VerificationError(
                    "verification-report temporary identity could not be recovered; "
                    "the unverified path was preserved"
                )
            )
        for recorded_cleanup_error in cleanup_errors:
            primary.add_note(
                f"verification-report cleanup failed: {recorded_cleanup_error!r}"
            )
        raise
    else:
        assert directory is not None
        assert descriptor_identity is not None
        assert result is not None
        try:
            if descriptor is not None:
                closing_descriptor = descriptor
                descriptor = None
                _close_owned_descriptor(closing_descriptor)
            if temporary_path is not None:
                _rollback_identity_bound_entry(
                    directory, temporary_path.name, descriptor_identity
                )
            _report_directory_parent(directory, require_requested_identity=True)
        except BaseException as cleanup_error:
            if descriptor is not None:
                closing_descriptor = descriptor
                descriptor = None
                _close_owned_descriptor(closing_descriptor, preserve=cleanup_error)
            try:
                _rollback_report_paths(
                    directory, path, temporary_path, descriptor_identity
                )
            except BaseException as rollback_error:
                cleanup_error.add_note(
                    "verification-report publication rollback failed: "
                    f"{rollback_error!r}"
                )
            raise
        return result
    finally:
        if directory is not None:
            _close_owned_descriptor(directory.descriptor, preserve=sys.exc_info()[1])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument(
        "--trusted-completion-sha256",
        required=True,
        help="Out-of-band SHA-256 of the analyzed run's completion.json",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New JSON path outside the analyzed run directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report, output_record = verify_analysis(
            args.run_directory,
            trusted_completion_sha256=args.trusted_completion_sha256,
            output=args.output,
        )
    except VerificationError as exc:
        print(f"VERIFICATION FAILED: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "verification_passed": report["verification_passed"],
                "release_ready": report["release_ready"],
                "verification_body_sha256": report["verification_body_sha256"],
                "output": output_record,
            },
            sort_keys=True,
        )
    )
    return 0 if report.get("release_ready") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
