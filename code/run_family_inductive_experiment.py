#!/usr/bin/env python3
"""Run the locked family-inductive mean/residual experiment on the server.

The outer-test outcomes are not indexed until every fitted model, prediction,
and validation-fitted descriptive risk scorer is fixed. Pilot runs exercise
the same path on a deterministic subset but are explicitly non-scientific.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import scipy
import sklearn
import torch
import torch.nn as nn
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from corrected_protocol import derive_seed, set_global_seed
from family_inductive import (
    DistanceRiskScorer,
    FamilyInductiveModel,
    center_residual_by_gene,
    context_residual_targets,
    gene_mean_metrics,
    gene_mean_targets,
    optimizer_steps_per_epoch,
)
from run_corrected_experiments import BaseData, FoldData, load_base_data
from run_family_holdout_experiment import (
    FamilyAssignment,
    derive_seeded_family_assignments,
    fit_family_preprocessing,
    make_family_split,
    preprocessing_record,
    validate_family_release,
)


ROOT = Path(__file__).resolve().parent.parent
LOCK_PATH = ROOT / "docs" / "revision_v4" / "family_transfer_lock.json"
OUTPUT_ROOT = ROOT / "results" / "family_inductive_v2"
SCHEMA = "DepPLM-family-inductive-run-v2"
PREDICTION_SCHEMA = "DepPLM-family-inductive-predictions-v2"
METRICS_SCHEMA = "DepPLM-family-inductive-metrics-v2"
COMPLETION_SCHEMA = "DepPLM-family-inductive-completion-v2"
FAILURE_SCHEMA = "DepPLM-family-inductive-failure-v2"
PILOT_SUBSET_SEED = 20261027
PILOT_CAMPAIGN_ROSTER_SHA256 = hashlib.sha256(
    b"engineering-pilot-no-formal-campaign-roster"
).hexdigest()

EXPECTED_DESIGNATED_SERVER: dict[str, str] = {
    "hostname": "user",
    "machine_id_sha256": "b2bb6e825d62892dc430d8999743a2c2eb6fc98178338fc1735fa6442a1368f8",
    "ipv4": "192.168.50.7",
}

EXPECTED_ROBUSTNESS_STRATA: dict[str, Any] = {
    "component_size_bins": [
        {"label": "size_1", "minimum": 1, "maximum": 1},
        {"label": "size_2", "minimum": 2, "maximum": 2},
        {"label": "size_3_to_5", "minimum": 3, "maximum": 5},
        {"label": "size_6_to_10", "minimum": 6, "maximum": 10},
        {"label": "size_11_plus", "minimum": 11, "maximum": None},
    ],
    "protein_length": {
        "scheme": "outer_test_quartiles",
        "quantiles": [0.0, 0.25, 0.5, 0.75, 1.0],
    },
}

REQUIRED_BASELINES = [
    "GLOBAL-TRAIN-MEAN",
    "SEQUENCE-KNN",
    "SEQUENCE-WEIGHTED-KNN",
    "PLM-RIDGE",
    "STATIC-CONVEX-STACK",
    "IDFREE-PLM-MEAN-MLP",
    "DECOMP-MEAN-ONLY",
]

PRIMARY_METHOD = "FAMILY-DISTANCE-STACK"
SECONDARY_CONTEXT_METHOD = "FAMILY-INDUCTIVE-DECOMP"
STACK_METHOD_ORDER = (
    "GLOBAL-TRAIN-MEAN",
    "SEQUENCE-WEIGHTED-KNN",
    "PLM-RIDGE",
)
STACK_BASE_PREDICTION_ARRAY_KEYS: dict[str, dict[str, str]] = {
    PRIMARY_METHOD: {
        "GLOBAL-TRAIN-MEAN": "primary_stack_base_global_prediction",
        "SEQUENCE-WEIGHTED-KNN": "primary_stack_base_weighted_knn_prediction",
        "PLM-RIDGE": "primary_stack_base_ridge_prediction",
    },
    "STATIC-CONVEX-STACK": {
        "GLOBAL-TRAIN-MEAN": "static_stack_base_global_prediction",
        "SEQUENCE-WEIGHTED-KNN": "static_stack_base_weighted_knn_prediction",
        "PLM-RIDGE": "static_stack_base_ridge_prediction",
    },
}
STACK_SOLVER_CONTRACT: dict[str, Any] = {
    "solver": "scipy.optimize.SLSQP",
    "max_iterations": 2000,
    "ftol": 1e-12,
    "constraint_tolerance": 1e-8,
    "initialization": "equal_endpoint_weights",
    "objective_weighting": "gene_weighted_component_cross_fitted_mse",
}
CROSSFIT_HASH_CONTRACT = (
    "sha256(utf8(decimal_training_seed)+NUL+utf8(component_id))_"
    "first_8_bytes_big_endian_mod_n_folds"
)
STACK_IMPLEMENTATION_CONTRACT: dict[str, Any] = {
    **STACK_SOLVER_CONTRACT,
    "failure_conditions": [
        "solver_unsuccessful",
        "nonfinite_objective_or_coefficient",
        "constraint_violation_above_tolerance",
    ],
    "component_fold_hash": CROSSFIT_HASH_CONTRACT,
    "distance_stack_weight_arrays": (
        "one_weight_per_query_gene_equal_to_a_m_plus_b_m_times_z"
    ),
    "selected_baseline_name_array": "length_one_fixed_width_unicode",
    "static_stack_grid": (
        "one_static_fit_per_knn_k_temperature_and_ridge_alpha_without_lambda_duplication"
    ),
    "audit": (
        "record_scipy_version_coefficients_constraint_residuals_and_solver_status"
    ),
}

EXPECTED_SENSITIVITY_CONTRACT: dict[str, Any] = {
    "outer_folds": 5,
    "split_seeds": [42],
    "training_seeds": [42],
    "role": "secondary_cannot_replace_primary_30_percent_analysis",
}

EXPECTED_SENSITIVITY_ASSETS: dict[str, Any] = {
    "20": {
        "path": "results/family_holdout_v1/identity_20/fold_manifest.tsv",
        "sha256": "bf165538bda47da5a5baac32564124381808611f397b833e2648d208ffa40bd0",
    },
    "40": {
        "path": "results/family_holdout_v1/identity_40/fold_manifest.tsv",
        "sha256": "8e564403a147927eae408253580f3b2da175517a0a6ebedacffe6549ce617822",
    },
    "50": {
        "path": "results/family_holdout_v1/identity_50/fold_manifest.tsv",
        "sha256": "548ea6620dac02f835054e8de72030c802ccff5f2855b06629104ab6f8b66ec9",
    },
}

EXPECTED_SELECTIVE_RISK_SCORE: dict[str, str] = {
    "type": "validation_fitted_nonnegative_distance_to_absolute_error_score",
    "ranking": "ascending_score",
    "coverage_guarantee": "none",
    "reporting_role": "descriptive_secondary_only",
}

# This is duplicated deliberately: an edited or absent lock contract must stop
# execution rather than silently changing the registered analysis.
EXPECTED_TRAINING_CONTRACT: dict[str, Any] = {
    "optimizer": "AdamW",
    "inner_max_epochs": 30,
    "patience": 5,
    "batch_size": 64,
    "gene_subsample": 6000,
    "top_mut": 5000,
    "n_expr_pc": 128,
    "neural_context_role": "secondary_only",
    "mean_only_ablation_architecture": (
        "matches_neural_context_sequence_encoder_and_mean_head"
    ),
    "mean_only_ablation_steps_per_epoch": "matches_neural_context_cell_batches",
    "baseline_selection_rule": (
        "per_job_max_inner_r2_then_min_mse_then_lexicographic_method_name"
    ),
    "outer_test_outcomes_forbidden_for_all_selection": True,
    "baseline_grids": {
        "uniform_knn_neighbors": [5, 20, 50],
        "ridge_alpha": [0.1, 1.0, 10.0, 100.0],
    },
    "mean_mlp_grid": {
        "hidden": [128, 256],
        "learning_rate": [0.001, 0.0003],
        "weight_decay": [1e-5],
    },
    "decomposition_grid": {
        "hidden": [128],
        "rank": [32, 64],
        "learning_rate": [0.001, 0.0003],
        "weight_decay": [1e-5],
        "gene_mean_loss_weight": [1.0],
        "context_residual_loss_weight": [1.0],
    },
    "pilot": {
        "cells": 64,
        "genes": 1000,
        "inner_max_epochs": 2,
        "patience": 1,
        "gene_subsample": 256,
        "bootstrap_replicates": 200,
        "scientific_use": "forbidden",
    },
}

EXPECTED_PRIMARY_METHOD_CONTRACT: dict[str, Any] = {
    "representation": "frozen_esm_only_no_gene_identifier",
    "base_methods": [
        "GLOBAL-TRAIN-MEAN",
        "SEQUENCE-WEIGHTED-KNN",
        "PLM-RIDGE",
    ],
    "formula": "sum_m((a_m+b_m*z)*p_m)",
    "distance": "one_minus_maximum_outer_training_cosine_similarity",
    "distance_scaling": "clip((d-q10)/(q90-q10),0,1)",
    "distance_quantiles": [0.1, 0.9],
    "degenerate_quantile_action": "fail_closed",
    "constraints": [
        "sum_a_equals_one",
        "sum_b_equals_zero",
        "endpoint_weights_nonnegative",
        "weighted_knn_slope_nonpositive",
        "global_slope_nonnegative",
    ],
    "objective": "component_cross_fitted_mse_plus_lambda_sum_squared_slopes",
    "inner_crossfit_folds": 3,
    "outer_refit_crossfit_folds": 4,
    "crossfit_assignment": "deterministic_component_hash_with_training_seed",
    "candidate_grid": {
        "knn_neighbors": [5, 20, 50],
        "knn_temperature": [0.03, 0.1, 0.3],
        "ridge_alpha": [0.1, 1.0, 10.0, 100.0],
        "gate_lambda": [0.0, 0.01, 0.1],
    },
    "implementation_contract": STACK_IMPLEMENTATION_CONTRACT,
    "selection_rule": (
        "max_inner_component_heldout_r2_then_min_mse_then_canonical_config"
    ),
    "required_ablation": "STATIC-CONVEX-STACK",
}

EXPECTED_DATA_ROLES: dict[str, str] = {
    "depmap_24q4": "development_only",
    "depmap_26q3": "single_access_same_platform_time_out_confirmation",
    "organoid_3d": "previously_accessed_orthogonal_assay_secondary_validation",
}

EXPECTED_EXTERNAL_3D_ANALYSIS: dict[str, Any] = {
    "prediction_source": "frozen_24q4_primary_consensus",
    "primary_method_prediction": "FAMILY-DISTANCE-STACK",
    "comparison_prediction": "consensus_per_job_selected_baseline",
    "outcome_scale": "organoid_corrected_crispr_log_fold_change",
    "outcome_access_status": "previously_downloaded_and_inspected",
    "relationship_to_26q3": (
        "potential_shared_organoid_models_and_related_screen_source_requires_exact_"
        "overlap_audit_and_forbids_independence_claim"
    ),
    "primary_metric": "arithmetic_mean_per_organoid_spearman",
    "comparison": "paired_per_organoid_method_minus_baseline_spearman",
    "forbidden_metrics": ["raw_mse", "raw_mae", "raw_r2"],
    "minimum_eligible_organoids": 100,
    "minimum_common_genes_per_organoid": 5000,
    "required_tissues": ["COLO", "OESO", "PANC"],
    "bootstrap": {
        "scheme": "two_way_sequence_component_and_tissue_stratified_organoid",
        "replicates": 5000,
        "confidence_level": 0.95,
        "seed": 20261028,
    },
}

EXPECTED_TEMPORAL_26Q3: dict[str, Any] = {
    "release": "DepMap_26Q3_Public",
    "outcome_filename": "CRISPRGeneEffect.csv",
    "outcome_project_state_at_lock": "not_downloaded_or_parsed",
    "access_scope": "designated_server_only_after_code_and_24q4_consensus_freeze",
    "access_ledger_path": "results/family_inductive_v2/temporal-access-ledger.jsonl",
    "first_access_event_required_before_outcome_parse": True,
    "acquisition_ledger_event_required_before_outcome_bytes": True,
    "input_snapshot_sha256_required": True,
    "cohort": {
        "primary_models": (
            "model_ids_present_in_26q3_and_absent_from_exact_24q4_crispr_gene_effect_source"
        ),
        "baseline_model_ids_source": {
            "path": "data/processed/cell_lines.txt",
            "sha256": "a70121fdaf136915e4e82e781fc25959477c220499a455f5c235f9574a2b7683",
            "semantics": "exact_24q4_crispr_gene_effect_row_order_and_model_ids",
        },
        "minimum_new_models": 100,
        "gene_header_regex": r"^(?P<symbol>.+) \((?P<entrez>[0-9]+)\)$",
        "gene_matching": "exact_symbol_after_terminal_numeric_entrez_suffix_removal",
        "synonym_rescue": "forbidden",
        "duplicate_symbol_action": "fail_closed",
        "minimum_finite_model_fraction_per_gene": 0.8,
        "minimum_eligible_genes": 5000,
    },
    "prediction_source": "frozen_24q4_primary_consensus_without_refit_or_recalibration",
    "primary_method_prediction": "FAMILY-DISTANCE-STACK",
    "comparison_prediction": "consensus_per_job_selected_baseline",
    "reference_prediction": "consensus_GLOBAL-TRAIN-MEAN",
    "primary_metric": "gene_mean_r2_vs_frozen_consensus_global_prediction",
    "primary_comparison": "delta_r2_vs_frozen_consensus_per_job_selected_baseline",
    "sensitivity_metrics": ["pearson", "spearman", "mse", "mae"],
    "bootstrap": {
        "unit": "sequence_component",
        "replicates": 5000,
        "confidence_level": 0.95,
        "seed": 20261029,
    },
    "source_provenance_contract": {
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
    },
    "shared_model_pipeline_correction_analysis": (
        "secondary_cannot_replace_new_model_primary"
    ),
    "shared_model_pipeline_correction_contract": {
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
    },
    "organoid_overlap_contract": {
        "exact_model_id_overlap_audit_required": True,
        "predeclared_organoid_model_ids": {
            "path": "docs/revision_v4/organoid_model_ids_predeclared.txt",
            "sha256": (
                "39c5fa0e2eaa0d7157732803ab4bc1f0ad81bd101a66fc09ea6bb101d6ea9035"
            ),
            "count": 256,
            "format": (
                "utf8_one_nonempty_unique_exact_id_per_line_lf_sorted_bytewise"
            ),
            "source_path": (
                "data/external/sanger_organoid_2026/features-v1/"
                "data_covariates_ALL-organoids.csv"
            ),
            "source_column": "sample_ID",
            "source_sha256": (
                "e57603e0ae26780ef112cbdbf063818dd17837858353894ba29e75f6936e9f64"
            ),
        },
        "identifier_namespace_mismatch_action": (
            "fail_unresolved_never_infer_zero_overlap"
        ),
        "treat_temporal_and_organoid_evidence_as_independent_if_overlap_nonzero": False,
        "nonoverlap_sensitivity_required_when_minimum_models_available": 30,
        "nonoverlap_sensitivity_minimum_eligible_genes": 5000,
        "report_overlap_counts_and_model_ids": True,
    },
    "post_access_changes_forbidden": [
        "model_refit",
        "gate_recalibration",
        "hyperparameter_change",
        "endpoint_change",
        "mapping_change",
        "subset_replacement",
        "threshold_change",
    ],
}

EXPECTED_CAMPAIGN_AGGREGATION: dict[str, Any] = {
    "required_primary_jobs": 45,
    "require_complete_cartesian_product": True,
    "job_grid": {
        "outer_folds": [0, 1, 2, 3, 4],
        "split_seeds": [42, 43, 44],
        "training_seeds": [42, 43, 44],
    },
    "per_seed_pair_oof_rule": "each_gene_exactly_once_across_five_outer_folds",
    "per_gene_prediction_count": 9,
    "prediction_aggregation": (
        "arithmetic_mean_across_three_split_seeds_by_three_training_seeds"
    ),
    "baseline_family_candidates": REQUIRED_BASELINES,
    "baseline_family_selection": (
        "independent_within_each_job_from_that_jobs_inner_validation_only"
    ),
    "cross_job_baseline_family_selection_forbidden": True,
    "selected_baseline_consensus": (
        "arithmetic_mean_of_each_jobs_explicit_selected_baseline_outer_prediction"
    ),
    "consensus_reference": {
        "global_train_mean_prediction_aggregation": (
            "arithmetic_mean_across_nine_out_of_fold_predictions"
        ),
        "gene_mean_r2_denominator": (
            "sum_squared_error_of_consensus_GLOBAL-TRAIN-MEAN_prediction"
        ),
    },
    "consensus_risk_and_distance": {
        "nearest_training_distance_aggregation": (
            "arithmetic_mean_across_nine_out_of_fold_nearest_outer_train_cosine_distances"
        ),
        "selective_risk_score_aggregation": (
            "arithmetic_mean_across_nine_out_of_fold_validation_fitted_risk_scores"
        ),
        "selective_order": (
            "ascending_consensus_risk_score_then_canonical_gene_index"
        ),
        "distance_quartile_definition": (
            "quartiles_of_consensus_nearest_training_distance_across_all_primary_genes"
        ),
        "distance_quartile_quantiles": [0.0, 0.25, 0.5, 0.75, 1.0],
    },
    "run_prediction_array_keys": {
        "gene_ids": "outer_test_gene_ids",
        "canonical_gene_indices": "outer_test_gene_indices",
        "source_gene_indices": "source_test_gene_indices",
        "component_ids": "outer_test_component_ids",
        "component_sizes": "outer_test_component_size_full_release",
        "protein_lengths": "outer_test_protein_length_residues",
        "cell_ids": "cell_ids",
        "source_cell_indices": "source_cell_indices",
        "observed_gene_effect": "observed_gene_effect",
        "observed_gene_mean": "observed_gene_mean",
        "observed_gene_centered_residual": "observed_gene_centered_residual",
        "nearest_training_distance": "nearest_outer_train_cosine_distance",
        "distance_gate_coordinate": "distance_gate_z",
        "distance_stack_weight_global": "distance_stack_weight_global",
        "distance_stack_weight_knn": "distance_stack_weight_knn",
        "distance_stack_weight_ridge": "distance_stack_weight_ridge",
        "selective_risk_score": "validation_fitted_distance_risk_score",
        "selected_baseline_name": "job_selected_baseline_name",
        "selected_baseline_prediction": (
            "job_selected_baseline_gene_mean_prediction"
        ),
        "method_gene_mean_predictions": {
            "GLOBAL-TRAIN-MEAN": "global_train_mean_prediction",
            "SEQUENCE-KNN": "sequence_knn_gene_mean_prediction",
            "SEQUENCE-WEIGHTED-KNN": "sequence_weighted_knn_gene_mean_prediction",
            "PLM-RIDGE": "plm_ridge_gene_mean_prediction",
            "STATIC-CONVEX-STACK": "static_convex_stack_gene_mean_prediction",
            "FAMILY-DISTANCE-STACK": "family_distance_stack_gene_mean_prediction",
            "IDFREE-PLM-MEAN-MLP": "idfree_plm_mean_mlp_prediction",
            "DECOMP-MEAN-ONLY": "decomp_mean_only_gene_mean_prediction",
            "FAMILY-INDUCTIVE-DECOMP": (
                "family_inductive_decomp_gene_mean_prediction"
            ),
        },
        "secondary_centered_residual_prediction": (
            "family_inductive_decomp_centered_residual_prediction"
        ),
        "secondary_total_prediction": "family_inductive_decomp_total_prediction",
    },
    "consensus_prediction_artifact": {
        "filename": "consensus-predictions.npz",
        "schema": "DepPLM-family-transfer-consensus-predictions-v2",
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
    },
    "campaign_roster_digest_required": True,
    "primary_metrics_and_gates_use_consensus_predictions": True,
    "single_run_gates_are_diagnostic_only": True,
    "missing_failed_or_duplicate_job_blocks_aggregation": True,
    "external_outcomes_forbidden_until_consensus_artifact_is_frozen": True,
}

EXPECTED_GATES: dict[str, Any] = {
    "development_signal": {
        "full_coverage_gene_mean_r2_min_exclusive": 0.0,
        "mse_improvement_vs_global_bootstrap_lower_min_exclusive": 0.0,
        "multi_gene_component_r2_min_exclusive": 0.0,
        "farthest_distance_quartile_r2_min_exclusive": 0.0,
    },
    "method": {
        "delta_r2_vs_per_job_selected_baseline_min": 0.01,
        "delta_r2_bootstrap_lower_min_exclusive": 0.0,
    },
    "temporal_26q3": {
        "gene_mean_r2_min_exclusive": 0.0,
        "delta_r2_vs_frozen_selected_baseline_min": 0.01,
        "delta_r2_bootstrap_lower_min_exclusive": 0.0,
    },
    "context": {
        "residual_r2_min_exclusive": 0.0,
        "residual_r2_bootstrap_lower_min_exclusive": 0.0,
        "mean_within_gene_pearson_min": 0.05,
    },
    "external_3d": {
        "mean_per_organoid_spearman_min": 0.2,
        "paired_spearman_delta_min_exclusive": 0.0,
        "paired_spearman_delta_two_way_bootstrap_lower_min_exclusive": 0.0,
    },
}


class FamilyInductiveRunError(RuntimeError):
    """Raised when a run cannot satisfy the locked analysis contract."""


class AtomicWriteError(RuntimeError):
    """Raised when an output would overwrite an existing artifact."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


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


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _json_safe(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _reject_duplicate_json_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise FamilyInductiveRunError(f"duplicate key in analysis lock: {key!r}")
        value[key] = item
    return value


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise AtomicWriteError(f"refusing to overwrite existing output: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_bytes(path, _json_bytes(value))


def _file_record(path: Path, *, relative_to: Path = ROOT) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    try:
        display = resolved.relative_to(relative_to.resolve(strict=True)).as_posix()
    except ValueError:
        display = str(resolved)
    stat = resolved.stat()
    return {
        "path": display,
        "bytes": int(stat.st_size),
        "sha256": _sha256_file(resolved),
    }


def _atomic_write_npz(
    path: Path, arrays: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    normalized = {
        name: np.ascontiguousarray(np.asarray(value))
        for name, value in arrays.items()
    }
    for name, value in normalized.items():
        if value.dtype == object:
            raise ValueError(f"object arrays are forbidden in predictions: {name}")
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, **normalized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise AtomicWriteError(f"refusing to overwrite existing output: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "schema": PREDICTION_SCHEMA,
        **_file_record(path, relative_to=path.parent),
        "arrays": {
            name: {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": _hash_array(value),
            }
            for name, value in sorted(normalized.items())
        },
    }


def _safe_repo_path(relative: str) -> Path:
    raw = Path(str(relative).replace("\\", "/"))
    if raw.is_absolute() or ".." in raw.parts:
        raise FamilyInductiveRunError(f"unsafe locked path: {relative!r}")
    target = (ROOT / raw).resolve(strict=True)
    try:
        target.relative_to(ROOT.resolve(strict=True))
    except ValueError as exc:
        raise FamilyInductiveRunError(f"locked path escapes repository: {relative!r}") from exc
    return target


def _protein_fasta_path(lock: Mapping[str, Any]) -> Path:
    protein = lock["locked_assets"]["protein_representation"]
    manifest_path = _safe_repo_path(str(protein["manifest_path"]))
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    try:
        fasta_record = manifest["input"]["fasta"]
        relative_path = str(fasta_record["path"])
        expected_hash = str(fasta_record["sha256"])
    except (KeyError, TypeError) as exc:
        raise FamilyInductiveRunError(
            "locked protein manifest lacks its source FASTA provenance"
        ) from exc
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise FamilyInductiveRunError("protein manifest contains an invalid FASTA SHA-256")
    fasta_path = _safe_repo_path(relative_path)
    observed_hash = _sha256_file(fasta_path)
    if observed_hash != expected_hash:
        raise FamilyInductiveRunError(
            "protein FASTA differs from the transitively locked embedding input: "
            f"expected {expected_hash}, observed {observed_hash}"
        )
    return fasta_path


def _load_protein_lengths(
    lock: Mapping[str, Any], genes: Sequence[str]
) -> np.ndarray:
    fasta_path = _protein_fasta_path(lock)
    lengths: dict[str, int] = {}
    current_gene: str | None = None
    current_length = 0

    def finish_record() -> None:
        nonlocal current_gene, current_length
        if current_gene is None:
            return
        if current_gene in lengths:
            raise FamilyInductiveRunError(
                f"duplicate protein FASTA gene identifier: {current_gene!r}"
            )
        if current_length <= 0:
            raise FamilyInductiveRunError(
                f"empty protein FASTA sequence: {current_gene!r}"
            )
        lengths[current_gene] = current_length

    with fasta_path.open("rt", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                finish_record()
                header = line[1:].strip()
                current_gene = header.split("|", 1)[0].strip()
                current_length = 0
                if not current_gene:
                    raise FamilyInductiveRunError(
                        f"empty FASTA identifier at line {line_number}"
                    )
            else:
                if current_gene is None:
                    raise FamilyInductiveRunError(
                        f"protein FASTA sequence precedes a header at line {line_number}"
                    )
                if re.search(r"\s", line):
                    raise FamilyInductiveRunError(
                        f"embedded whitespace in protein FASTA at line {line_number}"
                    )
                current_length += len(line)
    finish_record()

    missing = [str(gene) for gene in genes if str(gene) not in lengths]
    if missing:
        raise FamilyInductiveRunError(
            f"protein FASTA is missing {len(missing)} modeled genes; examples={missing[:10]!r}"
        )
    return np.asarray([lengths[str(gene)] for gene in genes], dtype=np.int64)


def _assert_designated_server(lock: Mapping[str, Any]) -> dict[str, Any]:
    registered = lock.get("designated_server")
    _require_exact(
        registered,
        EXPECTED_DESIGNATED_SERVER,
        "designated server identity",
    )
    system = platform.system().lower()
    release = platform.release()
    if system != "linux" or "microsoft" in release.lower():
        raise FamilyInductiveRunError(
            "execution is restricted to the designated Linux server, not Windows/WSL/other hosts"
        )

    observed: set[str] = set()
    command_outputs: dict[str, str | None] = {}
    for label, command in (
        ("hostname_all_ip_addresses", ["hostname", "-I"]),
        ("ip_global_ipv4", ["ip", "-o", "-4", "addr", "show", "scope", "global"]),
    ):
        try:
            result = subprocess.run(
                command,
                check=True,
                text=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError):
            command_outputs[label] = None
            continue
        command_outputs[label] = result.stdout.strip()
        observed.update(
            re.findall(r"(?<![0-9.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9.])", result.stdout)
        )

    try:
        observed.update(
            address[4][0]
            for address in socket.getaddrinfo(
                socket.gethostname(), None, family=socket.AF_INET
            )
        )
    except OSError:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("1.1.1.1", 53))
            observed.add(str(probe.getsockname()[0]))
    except OSError:
        pass

    required_ipv4 = str(registered["ipv4"])
    if required_ipv4 not in observed:
        raise FamilyInductiveRunError(
            "designated-server identity check failed: required interface address "
            f"{required_ipv4}, observed {sorted(observed)!r}"
        )
    machine_id_path = Path("/etc/machine-id")
    if not machine_id_path.is_file():
        raise FamilyInductiveRunError("designated-server /etc/machine-id is missing")
    machine_id_sha256 = hashlib.sha256(machine_id_path.read_bytes()).hexdigest()
    if machine_id_sha256 != str(registered["machine_id_sha256"]):
        raise FamilyInductiveRunError(
            "designated-server machine-id SHA-256 mismatch: "
            f"expected {registered['machine_id_sha256']}, observed {machine_id_sha256}"
        )
    hostname = platform.node()
    if hostname != str(registered["hostname"]):
        raise FamilyInductiveRunError(
            "designated-server hostname mismatch: "
            f"expected {registered['hostname']!r}, observed {hostname!r}"
        )
    return {
        "required_ipv4": required_ipv4,
        "verified_ipv4": required_ipv4,
        "observed_ipv4_addresses": sorted(observed),
        "hostname": hostname,
        "system": platform.system(),
        "release": release,
        "machine_id_sha256": machine_id_sha256,
        "discovery_commands": command_outputs,
    }


def _require_exact(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise FamilyInductiveRunError(
            f"{label} differs from the runner's registered contract; "
            f"expected {expected!r}, observed {actual!r}"
        )


def _require_keys(actual: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(actual)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise FamilyInductiveRunError(
            f"{label} keys differ from the registered contract; "
            f"missing={missing!r}, extra={extra!r}"
        )


def validate_analysis_lock(path: Path = LOCK_PATH) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_lock = path.read_bytes()
    lock = json.loads(
        raw_lock.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys
    )
    if not isinstance(lock, dict):
        raise FamilyInductiveRunError("analysis lock root must be a JSON object")
    _require_keys(
        lock,
        {
            "schema",
            "protocol_version",
            "locked_date",
            "status",
            "implementation_namespace",
            "execution_scope",
            "designated_server",
            "data_roles",
            "primary_split",
            "sensitivity_identity_percentages",
            "sensitivity_contract",
            "locked_assets",
            "primary_endpoint",
            "primary_metric",
            "primary_method",
            "primary_method_contract",
            "bootstrap",
            "coverage_levels",
            "selective_risk_score",
            "robustness_strata",
            "temporal_26q3",
            "external_3d_analysis",
            "training_contract",
            "campaign_aggregation",
            "gates",
            "primary_feature_access",
            "required_baselines",
            "context_claim_requires_context_gate",
            "selective_results_cannot_replace_full_coverage",
            "failed_gates_must_be_reported",
            "positive_result_is_not_guaranteed",
        },
        "analysis lock root",
    )
    _require_exact(lock.get("schema"), "DepPLM-family-transfer-analysis-lock-v2", "lock schema")
    _require_exact(lock.get("protocol_version"), "1.4", "protocol version")
    _require_exact(lock.get("locked_date"), "2026-10-27", "lock date")
    _require_exact(
        lock.get("status"),
        "post-discovery-development-and-timeout-confirmation-lock",
        "lock status",
    )
    _require_exact(
        lock.get("implementation_namespace"),
        "family_inductive_v2",
        "implementation namespace",
    )
    _require_exact(lock.get("execution_scope"), "remote-server-only", "execution scope")
    _require_exact(lock.get("data_roles"), EXPECTED_DATA_ROLES, "data roles")
    _require_exact(
        lock.get("designated_server"),
        EXPECTED_DESIGNATED_SERVER,
        "designated server identity",
    )
    _require_exact(
        lock.get("robustness_strata"),
        EXPECTED_ROBUSTNESS_STRATA,
        "robustness strata",
    )
    _require_exact(
        lock.get("primary_endpoint"),
        "held_out_component_gene_pan_context_mean_dependency",
        "primary endpoint",
    )
    _require_exact(
        lock.get("primary_metric"),
        "gene_mean_r2_vs_consensus_per_job_outer_training_global_prediction",
        "primary metric",
    )
    _require_exact(lock.get("primary_method"), PRIMARY_METHOD, "primary method")
    _require_exact(
        lock.get("primary_method_contract"),
        EXPECTED_PRIMARY_METHOD_CONTRACT,
        "primary method contract",
    )
    _require_exact(
        lock.get("sensitivity_identity_percentages"),
        [20.0, 40.0, 50.0],
        "identity sensitivity thresholds",
    )
    _require_exact(
        lock.get("sensitivity_contract"),
        EXPECTED_SENSITIVITY_CONTRACT,
        "identity sensitivity contract",
    )
    _require_exact(
        lock.get("selective_risk_score"),
        EXPECTED_SELECTIVE_RISK_SCORE,
        "selective risk-score contract",
    )
    _require_exact(lock.get("required_baselines"), REQUIRED_BASELINES, "required baselines")
    _require_exact(lock.get("training_contract"), EXPECTED_TRAINING_CONTRACT, "training contract")
    _require_exact(
        lock.get("campaign_aggregation"),
        EXPECTED_CAMPAIGN_AGGREGATION,
        "campaign aggregation contract",
    )
    _require_exact(
        lock.get("temporal_26q3"),
        EXPECTED_TEMPORAL_26Q3,
        "temporal 26Q3 analysis contract",
    )
    _require_exact(
        lock.get("external_3d_analysis"),
        EXPECTED_EXTERNAL_3D_ANALYSIS,
        "external 3D analysis contract",
    )
    _require_exact(lock.get("gates"), EXPECTED_GATES, "go/no-go gates")
    _require_exact(
        lock.get("primary_feature_access"),
        "gene_label_inductive_and_gene_feature_inductive_sequence_only_query",
        "primary feature access",
    )
    _require_exact(lock.get("context_claim_requires_context_gate"), True, "context claim gate")
    _require_exact(lock.get("selective_results_cannot_replace_full_coverage"), True, "coverage rule")
    _require_exact(lock.get("failed_gates_must_be_reported"), True, "failed-gate reporting")
    _require_exact(
        lock.get("positive_result_is_not_guaranteed"),
        True,
        "positive-result guarantee prohibition",
    )

    split = lock.get("primary_split")
    if not isinstance(split, Mapping):
        raise FamilyInductiveRunError("lock primary_split is missing")
    _require_keys(
        split,
        {
            "identity_percent",
            "minimum_bidirectional_coverage",
            "outer_folds",
            "split_seeds",
            "training_seeds",
        },
        "primary split",
    )
    _require_exact(float(split.get("identity_percent", -1.0)), 30.0, "identity threshold")
    _require_exact(
        float(split.get("minimum_bidirectional_coverage", -1.0)),
        0.8,
        "minimum bidirectional coverage",
    )
    _require_exact(int(split.get("outer_folds", -1)), 5, "outer fold count")
    _require_exact(split.get("split_seeds"), [42, 43, 44], "split seeds")
    _require_exact(split.get("training_seeds"), [42, 43, 44], "training seeds")

    bootstrap = lock.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise FamilyInductiveRunError("lock bootstrap contract is missing")
    _require_keys(
        bootstrap,
        {"unit", "replicates", "confidence_level", "seed"},
        "bootstrap contract",
    )
    _require_exact(bootstrap.get("unit"), "sequence_component", "bootstrap unit")
    _require_exact(int(bootstrap.get("replicates", -1)), 5000, "bootstrap replicates")
    _require_exact(float(bootstrap.get("confidence_level", -1.0)), 0.95, "bootstrap confidence")
    _require_exact(int(bootstrap.get("seed", -1)), 20261027, "bootstrap seed")
    _require_exact(lock.get("coverage_levels"), [1.0, 0.9, 0.75, 0.5], "coverage levels")

    assets = lock.get("locked_assets")
    if not isinstance(assets, Mapping):
        raise FamilyInductiveRunError("locked_assets is missing")
    _require_keys(
        assets,
        {
            "protein_representation",
            "dependency_matrix",
            "gene_manifest",
            "cell_manifest",
            "family_release",
        },
        "locked assets",
    )
    protein = assets.get("protein_representation")
    family = assets.get("family_release")
    if not isinstance(protein, Mapping) or not isinstance(family, Mapping):
        raise FamilyInductiveRunError("protein/family locked assets are missing")
    _require_keys(
        protein,
        {
            "repository",
            "model_revision",
            "pooling",
            "embedding_path",
            "embedding_sha256",
            "names_path",
            "names_sha256",
            "manifest_path",
            "manifest_sha256",
        },
        "protein representation asset",
    )
    _require_keys(
        family,
        {
            "release_manifest_path",
            "release_manifest_sha256",
            "primary_fold_manifest_path",
            "primary_fold_manifest_sha256",
            "sensitivity_fold_manifests",
        },
        "family release asset",
    )
    _require_exact(
        family.get("sensitivity_fold_manifests"),
        EXPECTED_SENSITIVITY_ASSETS,
        "sensitivity fold-manifest assets",
    )
    for name in ("gene_manifest", "cell_manifest"):
        block = assets.get(name)
        if not isinstance(block, Mapping):
            raise FamilyInductiveRunError(f"locked asset block missing: {name}")
        _require_keys(block, {"path", "sha256"}, f"{name} asset")
    dependency = assets.get("dependency_matrix")
    if not isinstance(dependency, Mapping):
        raise FamilyInductiveRunError("locked asset block missing: dependency_matrix")
    _require_keys(
        dependency,
        {"release", "path", "sha256"},
        "dependency_matrix asset",
    )
    _require_exact(
        dependency.get("release"),
        "DepMap_24Q4_Public",
        "dependency matrix release",
    )
    _require_exact(
        protein.get("repository"),
        "facebook/esm2_t6_8M_UR50D",
        "protein representation repository",
    )
    _require_exact(
        protein.get("model_revision"),
        "c731040fcd8d73dceaa04b0a8e6329b345b0f5df",
        "protein representation revision",
    )
    _require_exact(
        protein.get("pooling"),
        "inverse-coverage-weighted_residue_mean_across_deterministic_overlapping_windows",
        "protein representation pooling",
    )
    specifications = [
        (protein, "embedding_path", "embedding_sha256", "protein embedding"),
        (protein, "names_path", "names_sha256", "protein names"),
        (protein, "manifest_path", "manifest_sha256", "protein manifest"),
        (assets.get("dependency_matrix"), "path", "sha256", "dependency matrix"),
        (assets.get("gene_manifest"), "path", "sha256", "gene manifest"),
        (assets.get("cell_manifest"), "path", "sha256", "cell manifest"),
        (family, "release_manifest_path", "release_manifest_sha256", "family release manifest"),
        (family, "primary_fold_manifest_path", "primary_fold_manifest_sha256", "family fold manifest"),
    ]
    for identity_key, sensitivity_asset in EXPECTED_SENSITIVITY_ASSETS.items():
        specifications.append(
            (
                sensitivity_asset,
                "path",
                "sha256",
                f"family sensitivity fold manifest identity {identity_key}",
            )
        )
    verified_assets: dict[str, Any] = {}
    for container, path_key, hash_key, label in specifications:
        if not isinstance(container, Mapping):
            raise FamilyInductiveRunError(f"locked asset block missing: {label}")
        expected_hash = str(container.get(hash_key, ""))
        if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise FamilyInductiveRunError(f"invalid locked SHA-256 for {label}")
        target = _safe_repo_path(str(container.get(path_key, "")))
        observed_hash = _sha256_file(target)
        if observed_hash != expected_hash:
            raise FamilyInductiveRunError(
                f"locked SHA-256 mismatch for {label}: expected {expected_hash}, "
                f"observed {observed_hash}"
            )
        verified_assets[label] = _file_record(target)

    protein_fasta = _protein_fasta_path(lock)
    verified_assets["protein FASTA (transitively locked by embedding manifest)"] = (
        _file_record(protein_fasta)
    )

    release_manifest_path = _safe_repo_path(str(family["release_manifest_path"]))
    family_validation = validate_family_release(release_manifest_path.parent)
    if family_validation.get("verification_status") != "verified":
        raise FamilyInductiveRunError("complete family release verification did not pass")
    if path.read_bytes() != raw_lock:
        raise FamilyInductiveRunError("analysis lock changed during validation")
    return lock, {
        "lock": {
            "path": path.resolve(strict=True).relative_to(ROOT.resolve(strict=True)).as_posix(),
            "bytes": len(raw_lock),
            "sha256": hashlib.sha256(raw_lock).hexdigest(),
        },
        "locked_assets": verified_assets,
        "family_release_validation": family_validation,
    }


def _validate_registered_run(
    lock: Mapping[str, Any],
    *,
    identity_percent: float,
    outer_fold: int,
    split_seed: int,
    training_seed: int,
) -> None:
    identity = float(identity_percent)
    registered_identities = [
        float(lock["primary_split"]["identity_percent"]),
        *[float(value) for value in lock["sensitivity_identity_percentages"]],
    ]
    if identity not in registered_identities:
        raise FamilyInductiveRunError(
            "--identity-percent is not registered in the analysis lock"
        )
    contract = (
        lock["primary_split"]
        if identity == float(lock["primary_split"]["identity_percent"])
        else lock["sensitivity_contract"]
    )
    if outer_fold not in range(int(contract["outer_folds"])):
        raise FamilyInductiveRunError("--outer-fold is outside the registered range 0..4")
    if split_seed not in [int(value) for value in contract["split_seeds"]]:
        raise FamilyInductiveRunError("--split-seed is not registered in the analysis lock")
    if training_seed not in [int(value) for value in contract["training_seeds"]]:
        raise FamilyInductiveRunError("--training-seed is not registered in the analysis lock")


def _family_manifest_for_identity(
    lock: Mapping[str, Any], identity_percent: float
) -> Path:
    family = lock["locked_assets"]["family_release"]
    identity = float(identity_percent)
    if identity == float(lock["primary_split"]["identity_percent"]):
        return _safe_repo_path(str(family["primary_fold_manifest_path"]))
    identity_key = str(int(identity))
    sensitivity = family["sensitivity_fold_manifests"]
    if identity_key not in sensitivity:
        raise FamilyInductiveRunError(
            f"no locked family manifest for identity {identity_percent}"
        )
    return _safe_repo_path(str(sensitivity[identity_key]["path"]))


def _source_paths(lock: Mapping[str, Any]) -> tuple[list[Path], list[Path]]:
    code_paths = [
        Path(__file__),
        ROOT / "code" / "family_inductive.py",
        ROOT / "code" / "run_family_holdout_experiment.py",
        ROOT / "code" / "run_corrected_experiments.py",
        ROOT / "code" / "corrected_protocol.py",
        ROOT / "code" / "embedding_provenance.py",
        ROOT / "code" / "family_holdout.py",
        ROOT / "code" / "models.py",
        LOCK_PATH,
        ROOT / "docs" / "revision_v4" / "FAMILY_TRANSFER_PROTOCOL.md",
    ]
    assets = lock["locked_assets"]
    protein = assets["protein_representation"]
    family = assets["family_release"]
    input_paths = [
        _safe_repo_path(protein["embedding_path"]),
        _safe_repo_path(protein["names_path"]),
        _safe_repo_path(protein["manifest_path"]),
        _protein_fasta_path(lock),
        _safe_repo_path(assets["dependency_matrix"]["path"]),
        _safe_repo_path(assets["gene_manifest"]["path"]),
        _safe_repo_path(assets["cell_manifest"]["path"]),
        _safe_repo_path(family["release_manifest_path"]),
        _safe_repo_path(family["primary_fold_manifest_path"]),
        ROOT / "data" / "processed" / "lineages.txt",
        ROOT / "data" / "processed" / "ppi_graph.npz",
        ROOT / "data" / "depmap" / "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
        ROOT / "data" / "depmap" / "OmicsSomaticMutations.csv",
    ]
    input_paths.extend(
        _safe_repo_path(asset["path"])
        for asset in family["sensitivity_fold_manifests"].values()
    )
    for source in (*code_paths, *input_paths):
        source.resolve(strict=True)
    return code_paths, input_paths


def _records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    return [_file_record(path) for path in paths]


def _git_record() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    status = run("status", "--porcelain=v1")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
        "status_sha256": (
            None if status is None else hashlib.sha256(status.encode("utf-8")).hexdigest()
        ),
    }


def _environment_record(device: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_debug_mode": int(torch.get_deterministic_debug_mode()),
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "git": _git_record(),
    }
    if device.startswith("cuda"):
        index = torch.device(device).index
        if index is None:
            index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        record["cuda_device"] = {
            "index": int(index),
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "capability": [int(properties.major), int(properties.minor)],
        }
        device_uuid = getattr(properties, "uuid", None)
        if device_uuid is not None:
            record["cuda_device"]["uuid"] = str(device_uuid)
    return record


def _resolve_device(raw: str) -> str:
    if raw == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        parsed = torch.device(raw)
    except (RuntimeError, ValueError) as exc:
        raise FamilyInductiveRunError(f"invalid --device: {raw!r}") from exc
    if parsed.type not in {"cpu", "cuda"}:
        raise FamilyInductiveRunError("--device must be cpu, cuda, cuda:N, or auto")
    if parsed.type == "cuda":
        if not torch.cuda.is_available():
            raise FamilyInductiveRunError("CUDA was requested but torch reports no CUDA device")
        index = 0 if parsed.index is None else int(parsed.index)
        if index < 0 or index >= torch.cuda.device_count():
            raise FamilyInductiveRunError(f"CUDA device index is unavailable: {index}")
        return f"cuda:{index}"
    return "cpu"


def _empty_device_cache(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def _expand_grid(grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    names = sorted(grid)
    return [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*(grid[name] for name in names))
    ]


def _canonical_config(config: Mapping[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _selection_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = candidate["validation_metrics"]
    r2 = float(metrics["r2"])
    mse = float(metrics["mse"])
    if not math.isfinite(r2) or not math.isfinite(mse):
        raise FamilyInductiveRunError("inner validation selection metric is non-finite")
    return (
        -r2,
        mse,
        _canonical_config(candidate["config"]),
        int(candidate.get("epoch", 0)),
    )


def _job_baseline_selection_key(
    name: str, candidate: Mapping[str, Any]
) -> tuple[float, float, str]:
    metrics = candidate["validation_metrics"]
    r2 = float(metrics["r2"])
    mse = float(metrics["mse"])
    if not math.isfinite(r2) or not math.isfinite(mse):
        raise FamilyInductiveRunError(
            "job baseline inner-validation selection metric is non-finite"
        )
    return (-r2, mse, str(name))


def _global_gene_mean(y: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, float]:
    means = gene_mean_targets(y, indices)
    value = float(np.mean(means, dtype=np.float64))
    if not math.isfinite(value):
        raise FamilyInductiveRunError("training-gene global mean is non-finite")
    return means, value


def _unit_embeddings(values: np.ndarray, label: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or not np.all(np.isfinite(matrix)):
        raise FamilyInductiveRunError(f"{label} embeddings are empty or non-finite")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms <= 0.0):
        raise FamilyInductiveRunError(f"{label} embeddings contain zero-norm rows")
    return matrix / norms[:, None]


def _knn_predict(
    train_embeddings: np.ndarray,
    train_target: np.ndarray,
    query_embeddings: np.ndarray,
    *,
    neighbors: int,
    chunk_size: int = 1024,
) -> np.ndarray:
    training = _unit_embeddings(train_embeddings, "training")
    query = _unit_embeddings(query_embeddings, "query")
    target = np.asarray(train_target, dtype=np.float64)
    if target.shape != (training.shape[0],) or not np.all(np.isfinite(target)):
        raise FamilyInductiveRunError("KNN training target does not align or is non-finite")
    if neighbors <= 0 or neighbors > training.shape[0]:
        raise FamilyInductiveRunError("KNN neighbor count exceeds available training genes")
    output = np.empty(query.shape[0], dtype=np.float64)
    for start in range(0, query.shape[0], chunk_size):
        stop = min(start + chunk_size, query.shape[0])
        similarity = query[start:stop] @ training.T
        selected = np.argpartition(-similarity, neighbors - 1, axis=1)[:, :neighbors]
        output[start:stop] = np.mean(target[selected], axis=1)
    return output


def _sequence_prediction_grid(
    train_embeddings: np.ndarray,
    train_target: np.ndarray,
    query_embeddings: np.ndarray,
    *,
    uniform_neighbors: Sequence[int],
    weighted_neighbors: Sequence[int],
    temperatures: Sequence[float],
    ridge_alphas: Sequence[float],
    chunk_size: int = 512,
) -> dict[str, Any]:
    """Fit all registered sequence predictors once for one train/query partition."""

    training = _unit_embeddings(train_embeddings, "training")
    query = _unit_embeddings(query_embeddings, "query")
    target = np.asarray(train_target, dtype=np.float64)
    if target.shape != (training.shape[0],) or not np.all(np.isfinite(target)):
        raise FamilyInductiveRunError(
            "sequence-grid training target does not align or is non-finite"
        )
    uniform_k = tuple(sorted({int(value) for value in uniform_neighbors}))
    weighted_k = tuple(sorted({int(value) for value in weighted_neighbors}))
    tau_grid = tuple(sorted({float(value) for value in temperatures}))
    alpha_grid = tuple(sorted({float(value) for value in ridge_alphas}))
    all_k = uniform_k + weighted_k
    if not all_k or min(all_k) <= 0 or max(all_k) > training.shape[0]:
        raise FamilyInductiveRunError(
            "registered KNN grid exceeds the available training genes"
        )
    if not tau_grid or any(not math.isfinite(value) or value <= 0.0 for value in tau_grid):
        raise FamilyInductiveRunError("weighted-KNN temperatures must be positive and finite")
    if not alpha_grid or any(not math.isfinite(value) or value <= 0.0 for value in alpha_grid):
        raise FamilyInductiveRunError("ridge alphas must be positive and finite")

    n_query = query.shape[0]
    uniform = {value: np.empty(n_query, dtype=np.float64) for value in uniform_k}
    weighted = {
        (neighbors, temperature): np.empty(n_query, dtype=np.float64)
        for neighbors in weighted_k
        for temperature in tau_grid
    }
    distance = np.empty(n_query, dtype=np.float64)
    maximum_k = max(all_k)
    for start in range(0, n_query, chunk_size):
        stop = min(start + chunk_size, n_query)
        similarity = query[start:stop] @ training.T
        nearest = np.argpartition(-similarity, maximum_k - 1, axis=1)[:, :maximum_k]
        nearest_similarity = np.take_along_axis(similarity, nearest, axis=1)
        order = np.argsort(-nearest_similarity, axis=1, kind="mergesort")
        nearest = np.take_along_axis(nearest, order, axis=1)
        nearest_similarity = np.take_along_axis(nearest_similarity, order, axis=1)
        nearest_target = target[nearest]
        distance[start:stop] = np.clip(
            1.0 - np.max(similarity, axis=1),
            0.0,
            2.0,
        )
        for neighbors in uniform_k:
            uniform[neighbors][start:stop] = np.mean(
                nearest_target[:, :neighbors], axis=1, dtype=np.float64
            )
        for neighbors in weighted_k:
            local_similarity = nearest_similarity[:, :neighbors]
            local_target = nearest_target[:, :neighbors]
            shifted = local_similarity - np.max(local_similarity, axis=1, keepdims=True)
            for temperature in tau_grid:
                weight = np.exp(shifted / temperature)
                denominator = np.sum(weight, axis=1)
                if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0.0):
                    raise FamilyInductiveRunError(
                        "SEQUENCE-WEIGHTED-KNN produced invalid normalization"
                    )
                weighted[(neighbors, temperature)][start:stop] = np.sum(
                    weight * local_target, axis=1
                ) / denominator

    ridge = {
        alpha: _fit_ridge_predict(
            train_embeddings,
            target,
            query_embeddings,
            alpha=alpha,
        )
        for alpha in alpha_grid
    }
    global_value = float(np.mean(target, dtype=np.float64))
    result = {
        "global": np.full(n_query, global_value, dtype=np.float64),
        "uniform_knn": uniform,
        "weighted_knn": weighted,
        "ridge": ridge,
        "distance": distance,
    }
    for value in [distance, *uniform.values(), *weighted.values(), *ridge.values()]:
        if not np.all(np.isfinite(value)):
            raise FamilyInductiveRunError("sequence prediction grid contains non-finite values")
    return result


def _component_crossfit_fold_ids(
    component_ids: Sequence[str], *, training_seed: int, n_folds: int
) -> np.ndarray:
    if n_folds < 2:
        raise FamilyInductiveRunError("component cross-fitting requires at least two folds")
    values = np.asarray(component_ids, dtype=str)
    if values.ndim != 1 or values.size == 0 or any(not value for value in values):
        raise FamilyInductiveRunError("component IDs are empty or malformed")
    unique = sorted(set(values.tolist()))
    if len(unique) < n_folds:
        raise FamilyInductiveRunError(
            "fewer sequence components than registered cross-fit folds"
        )
    component_fold: dict[str, int] = {}
    prefix = str(int(training_seed)).encode("utf-8") + b"\x00"
    for component_id in unique:
        digest = hashlib.sha256(prefix + component_id.encode("utf-8")).digest()
        component_fold[component_id] = int.from_bytes(digest[:8], "big") % n_folds
    fold_ids = np.asarray([component_fold[value] for value in values], dtype=np.int64)
    observed = set(fold_ids.tolist())
    expected = set(range(n_folds))
    if observed != expected:
        raise FamilyInductiveRunError(
            "deterministic component hash produced an empty cross-fit fold"
        )
    return fold_ids


def _component_crossfit_prediction_grid(
    embeddings: np.ndarray,
    target: np.ndarray,
    component_ids: Sequence[str],
    *,
    training_seed: int,
    n_folds: int,
    uniform_neighbors: Sequence[int],
    weighted_neighbors: Sequence[int],
    temperatures: Sequence[float],
    ridge_alphas: Sequence[float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    matrix = np.asarray(embeddings)
    values = np.asarray(target, dtype=np.float64)
    if matrix.ndim != 2 or values.shape != (matrix.shape[0],):
        raise FamilyInductiveRunError("cross-fit embeddings and targets do not align")
    fold_ids = _component_crossfit_fold_ids(
        component_ids, training_seed=training_seed, n_folds=n_folds
    )
    uniform_k = tuple(sorted({int(value) for value in uniform_neighbors}))
    weighted_k = tuple(sorted({int(value) for value in weighted_neighbors}))
    tau_grid = tuple(sorted({float(value) for value in temperatures}))
    alpha_grid = tuple(sorted({float(value) for value in ridge_alphas}))
    output: dict[str, Any] = {
        "global": np.empty(values.size, dtype=np.float64),
        "uniform_knn": {
            neighbors: np.empty(values.size, dtype=np.float64)
            for neighbors in uniform_k
        },
        "weighted_knn": {
            (neighbors, temperature): np.empty(values.size, dtype=np.float64)
            for neighbors in weighted_k
            for temperature in tau_grid
        },
        "ridge": {
            alpha: np.empty(values.size, dtype=np.float64) for alpha in alpha_grid
        },
        "distance": np.empty(values.size, dtype=np.float64),
    }
    fold_audit: list[dict[str, Any]] = []
    for fold in range(n_folds):
        validation_mask = fold_ids == fold
        training_mask = ~validation_mask
        if not np.any(validation_mask) or not np.any(training_mask):
            raise FamilyInductiveRunError("component cross-fit partition is empty")
        grid = _sequence_prediction_grid(
            matrix[training_mask],
            values[training_mask],
            matrix[validation_mask],
            uniform_neighbors=uniform_k,
            weighted_neighbors=weighted_k,
            temperatures=tau_grid,
            ridge_alphas=alpha_grid,
        )
        output["global"][validation_mask] = grid["global"]
        output["distance"][validation_mask] = grid["distance"]
        for neighbors in uniform_k:
            output["uniform_knn"][neighbors][validation_mask] = grid[
                "uniform_knn"
            ][neighbors]
        for key in output["weighted_knn"]:
            output["weighted_knn"][key][validation_mask] = grid["weighted_knn"][key]
        for alpha in alpha_grid:
            output["ridge"][alpha][validation_mask] = grid["ridge"][alpha]
        fold_audit.append(
            {
                "fold": fold,
                "n_training_genes": int(np.sum(training_mask)),
                "n_validation_genes": int(np.sum(validation_mask)),
                "n_training_components": len(
                    set(np.asarray(component_ids, dtype=str)[training_mask].tolist())
                ),
                "n_validation_components": len(
                    set(np.asarray(component_ids, dtype=str)[validation_mask].tolist())
                ),
            }
        )
    for group in ("global", "distance"):
        if not np.all(np.isfinite(output[group])):
            raise FamilyInductiveRunError("cross-fit output contains non-finite values")
    audit = {
        "n_folds": int(n_folds),
        "training_seed": int(training_seed),
        "assignment_contract": CROSSFIT_HASH_CONTRACT,
        "fold_ids_sha256": _hash_array(fold_ids.astype("<i8")),
        "folds": fold_audit,
    }
    return output, audit


def _distance_gate_coordinate(
    distance: np.ndarray, *, q10: float, q90: float
) -> np.ndarray:
    values = np.asarray(distance, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise FamilyInductiveRunError("distance gate received non-finite distances")
    if not math.isfinite(q10) or not math.isfinite(q90) or q90 <= q10:
        raise FamilyInductiveRunError(
            "distance gate q10/q90 are degenerate; registered action is fail closed"
        )
    return np.clip((values - q10) / (q90 - q10), 0.0, 1.0)


def _fit_distance_stack(
    base_predictions: np.ndarray,
    target: np.ndarray,
    distance: np.ndarray,
    *,
    gate_lambda: float,
    static: bool,
) -> dict[str, Any]:
    predictions = np.asarray(base_predictions, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    distances = np.asarray(distance, dtype=np.float64)
    if predictions.ndim != 2 or predictions.shape[1] != len(STACK_METHOD_ORDER):
        raise FamilyInductiveRunError("distance stack requires exactly three base predictions")
    if truth.shape != (predictions.shape[0],) or distances.shape != truth.shape:
        raise FamilyInductiveRunError("distance-stack arrays do not align")
    if not np.all(np.isfinite(predictions)) or not np.all(np.isfinite(truth)):
        raise FamilyInductiveRunError("distance-stack fit contains non-finite values")
    q10, q90 = np.quantile(
        distances,
        EXPECTED_PRIMARY_METHOD_CONTRACT["distance_quantiles"],
    )
    z = _distance_gate_coordinate(distances, q10=float(q10), q90=float(q90))
    penalty = float(gate_lambda)
    if not math.isfinite(penalty) or penalty < 0.0:
        raise FamilyInductiveRunError("distance-stack lambda must be finite and nonnegative")
    if static and penalty != 0.0:
        raise FamilyInductiveRunError("static convex stack must not have a lambda dimension")

    if static:
        def objective(parameter: np.ndarray) -> float:
            error = predictions @ parameter - truth
            return float(np.mean(np.square(error), dtype=np.float64))

        result = minimize(
            objective,
            np.full(3, 1.0 / 3.0, dtype=np.float64),
            method="SLSQP",
            bounds=[(0.0, None)] * 3,
            constraints=[{"type": "eq", "fun": lambda value: np.sum(value) - 1.0}],
            options={
                "maxiter": int(STACK_SOLVER_CONTRACT["max_iterations"]),
                "ftol": float(STACK_SOLVER_CONTRACT["ftol"]),
                "disp": False,
            },
        )
        endpoint_start = np.asarray(result.x, dtype=np.float64)
        endpoint_end = endpoint_start.copy()
    else:
        def objective(parameter: np.ndarray) -> float:
            start = parameter[:3]
            end = parameter[3:]
            weight = (1.0 - z[:, None]) * start + z[:, None] * end
            error = np.sum(weight * predictions, axis=1) - truth
            slope = end - start
            return float(
                np.mean(np.square(error), dtype=np.float64)
                + penalty * np.sum(np.square(slope), dtype=np.float64)
            )

        result = minimize(
            objective,
            np.full(6, 1.0 / 3.0, dtype=np.float64),
            method="SLSQP",
            bounds=[(0.0, None)] * 6,
            constraints=[
                {"type": "eq", "fun": lambda value: np.sum(value[:3]) - 1.0},
                {"type": "eq", "fun": lambda value: np.sum(value[3:]) - 1.0},
                {"type": "ineq", "fun": lambda value: value[3] - value[0]},
                {"type": "ineq", "fun": lambda value: value[1] - value[4]},
            ],
            options={
                "maxiter": int(STACK_SOLVER_CONTRACT["max_iterations"]),
                "ftol": float(STACK_SOLVER_CONTRACT["ftol"]),
                "disp": False,
            },
        )
        endpoint_start = np.asarray(result.x[:3], dtype=np.float64)
        endpoint_end = np.asarray(result.x[3:], dtype=np.float64)

    tolerance = float(STACK_SOLVER_CONTRACT["constraint_tolerance"])
    slope = endpoint_end - endpoint_start
    constraint_residuals = {
        "abs_sum_a_minus_one": float(abs(np.sum(endpoint_start) - 1.0)),
        "abs_sum_b": float(abs(np.sum(slope))),
        "minimum_endpoint_weight": float(
            min(np.min(endpoint_start), np.min(endpoint_end))
        ),
        "global_slope": float(slope[0]),
        "weighted_knn_slope": float(slope[1]),
    }
    valid = (
        bool(result.success)
        and np.all(np.isfinite(endpoint_start))
        and np.all(np.isfinite(endpoint_end))
        and math.isfinite(float(result.fun))
        and constraint_residuals["abs_sum_a_minus_one"] <= tolerance
        and constraint_residuals["abs_sum_b"] <= tolerance
        and constraint_residuals["minimum_endpoint_weight"] >= -tolerance
        and (static or constraint_residuals["global_slope"] >= -tolerance)
        and (static or constraint_residuals["weighted_knn_slope"] <= tolerance)
    )
    if not valid:
        raise FamilyInductiveRunError(
            "distance-stack constrained optimization failed closed: "
            f"success={result.success!r}, message={result.message!s}, "
            f"constraints={constraint_residuals!r}"
        )
    return {
        "schema": "DepPLM-distance-stack-fit-v1",
        "static": bool(static),
        "method_order": list(STACK_METHOD_ORDER),
        "q10": float(q10),
        "q90": float(q90),
        "a": endpoint_start.tolist(),
        "b": slope.tolist(),
        "endpoint_weights_z0": endpoint_start.tolist(),
        "endpoint_weights_z1": endpoint_end.tolist(),
        "gate_lambda": penalty,
        "objective_value": float(result.fun),
        "solver": STACK_SOLVER_CONTRACT["solver"],
        "solver_scipy_version": scipy.__version__,
        "solver_success": bool(result.success),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_iterations": int(result.nit),
        "solver_contract": dict(STACK_SOLVER_CONTRACT),
        "constraint_residuals": constraint_residuals,
    }


def _predict_distance_stack(
    fit: Mapping[str, Any],
    base_predictions: np.ndarray,
    distance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions = np.asarray(base_predictions, dtype=np.float64)
    if predictions.ndim != 2 or predictions.shape[1] != len(STACK_METHOD_ORDER):
        raise FamilyInductiveRunError("distance-stack prediction bases are malformed")
    z = _distance_gate_coordinate(
        distance, q10=float(fit["q10"]), q90=float(fit["q90"])
    )
    a = np.asarray(fit["a"], dtype=np.float64)
    b = np.asarray(fit["b"], dtype=np.float64)
    weight = a[None, :] + z[:, None] * b[None, :]
    tolerance = float(STACK_SOLVER_CONTRACT["constraint_tolerance"])
    if np.any(weight < -tolerance) or not np.all(np.isfinite(weight)):
        raise FamilyInductiveRunError("distance stack produced invalid per-query weights")
    if np.max(np.abs(np.sum(weight, axis=1) - 1.0)) > tolerance:
        raise FamilyInductiveRunError("distance stack per-query weights do not sum to one")
    output = np.sum(weight * predictions, axis=1)
    if not np.all(np.isfinite(output)):
        raise FamilyInductiveRunError("distance stack produced non-finite predictions")
    return output, z, weight


def _fit_ridge_predict(
    train_embeddings: np.ndarray,
    train_target: np.ndarray,
    query_embeddings: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    train = scaler.fit_transform(np.asarray(train_embeddings, dtype=np.float64))
    query = scaler.transform(np.asarray(query_embeddings, dtype=np.float64))
    model = Ridge(alpha=float(alpha), solver="lsqr", tol=1e-8, max_iter=10_000)
    model.fit(train, np.asarray(train_target, dtype=np.float64))
    output = np.asarray(model.predict(query), dtype=np.float64)
    if not np.all(np.isfinite(output)):
        raise FamilyInductiveRunError("PLM-RIDGE produced non-finite predictions")
    return output


class SequenceMeanMLP(nn.Module):
    """Frozen sequence table plus shared MLP; contains no gene-ID parameter."""

    def __init__(self, embeddings: torch.Tensor, hidden: int) -> None:
        super().__init__()
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a genes-by-features matrix")
        self.emb = nn.Parameter(embeddings.detach().clone().float(), requires_grad=False)
        self.network = nn.Sequential(
            nn.Linear(int(embeddings.shape[1]), int(hidden)),
            nn.GELU(),
            nn.LayerNorm(int(hidden)),
            nn.Linear(int(hidden), 1),
        )

    def forward(self, gene_indices: torch.Tensor) -> torch.Tensor:
        return self.network(self.emb.index_select(0, gene_indices)).squeeze(-1)


def _copy_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _architecture_signature(model: nn.Module) -> str:
    structure = [
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
        }
        for name, parameter in model.named_parameters()
    ]
    return hashlib.sha256(_json_bytes(structure)).hexdigest()


def _assert_matched_optimizer_invariants(
    proposed: Mapping[str, Any], mean_only: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed unless a proposed/mean-only candidate used a matched schedule."""
    exact_pairs = {
        "model_architecture_sha256": "architecture",
        "optimizer_steps_per_epoch": "optimizer steps per epoch",
        "optimizer_steps_requested": "requested optimizer steps",
        "epochs_executed": "executed epochs",
        "optimizer_steps_executed": "executed optimizer steps",
        "gene_sampling_schedule_sha256": "per-step gene sampling",
        "cell_batch_schedule_sha256": "cell-batch schedule",
        "mean_path_trainable_parameter_count": "mean-path parameter count",
    }
    for key, label in exact_pairs.items():
        if proposed.get(key) != mean_only.get(key):
            raise FamilyInductiveRunError(
                f"DECOMP-MEAN-ONLY does not match proposed {label}: "
                f"{proposed.get(key)!r} != {mean_only.get(key)!r}"
            )
    for label, record in (("proposed", proposed), ("mean-only", mean_only)):
        expected_executed = int(record["epochs_executed"]) * int(
            record["optimizer_steps_per_epoch"]
        )
        if int(record["optimizer_steps_executed"]) != expected_executed:
            raise FamilyInductiveRunError(
                f"{label} optimizer-step accounting mismatch"
            )
        if int(record["optimizer_steps_executed"]) > int(
            record["optimizer_steps_requested"]
        ):
            raise FamilyInductiveRunError(
                f"{label} executed more optimizer steps than requested"
            )
    if int(mean_only["residual_path_trainable_parameter_count"]) != 0:
        raise FamilyInductiveRunError(
            "DECOMP-MEAN-ONLY exposes residual/cell parameters to optimization"
        )
    if int(proposed["residual_path_trainable_parameter_count"]) <= 0:
        raise FamilyInductiveRunError(
            "proposed decomposition has no active residual/cell parameters"
        )
    if int(mean_only["active_trainable_parameter_count"]) != int(
        mean_only["mean_path_trainable_parameter_count"]
    ):
        raise FamilyInductiveRunError(
            "DECOMP-MEAN-ONLY active parameters are not exactly the mean path"
        )
    if float(mean_only["context_residual_loss_weight_effective"]) != 0.0:
        raise FamilyInductiveRunError(
            "DECOMP-MEAN-ONLY residual loss weight is not fixed to zero"
        )
    return {
        "status": "matched",
        "architecture_sha256": str(proposed["model_architecture_sha256"]),
        "optimizer_steps_per_epoch": int(proposed["optimizer_steps_per_epoch"]),
        "optimizer_steps_requested_each": int(proposed["optimizer_steps_requested"]),
        "optimizer_steps_executed_each": int(proposed["optimizer_steps_executed"]),
        "gene_sampling_schedule_sha256": str(
            proposed["gene_sampling_schedule_sha256"]
        ),
        "cell_batch_schedule_sha256": str(proposed["cell_batch_schedule_sha256"]),
        "mean_path_trainable_parameter_count": int(
            proposed["mean_path_trainable_parameter_count"]
        ),
        "proposed_active_trainable_parameter_count": int(
            proposed["active_trainable_parameter_count"]
        ),
        "mean_only_active_trainable_parameter_count": int(
            mean_only["active_trainable_parameter_count"]
        ),
    }


def _predict_mean_mlp(
    model: SequenceMeanMLP,
    indices: np.ndarray,
    *,
    device: str,
    chunk_size: int = 4096,
) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, indices.size, chunk_size):
            selected = torch.from_numpy(indices[start : start + chunk_size]).long().to(device)
            chunks.append(model(selected).float().cpu().numpy())
    output = np.concatenate(chunks).astype(np.float64, copy=False)
    if output.shape != (indices.size,) or not np.all(np.isfinite(output)):
        raise FamilyInductiveRunError("IDFREE-PLM-MEAN-MLP produced invalid predictions")
    return output


def _train_mean_mlp(
    base: BaseData,
    train_indices: np.ndarray,
    validation_indices: np.ndarray | None,
    *,
    config: Mapping[str, Any],
    epochs: int,
    patience: int,
    gene_subsample: int,
    device: str,
    seed: int,
) -> tuple[SequenceMeanMLP, dict[str, Any]]:
    set_global_seed(seed)
    model = SequenceMeanMLP(
        torch.from_numpy(np.asarray(base.embeddings, dtype=np.float32)),
        hidden=int(config["hidden"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    train_means, train_global = _global_gene_mean(base.gene_effect, train_indices)
    mean_lookup = np.full(len(base.genes), np.nan, dtype=np.float64)
    mean_lookup[train_indices] = train_means
    validation_target = (
        None
        if validation_indices is None
        else gene_mean_targets(base.gene_effect, validation_indices)
    )
    rng = np.random.default_rng(derive_seed(seed, "mean-mlp-gene-sampling"))
    best_candidate: dict[str, Any] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    stale_epochs = 0

    for epoch in range(1, int(epochs) + 1):
        model.train()
        if 0 < gene_subsample < train_indices.size:
            selected_np = np.sort(
                rng.choice(train_indices, gene_subsample, replace=False)
            ).astype(np.int64)
        else:
            selected_np = train_indices
        selected = torch.from_numpy(selected_np).long().to(device)
        target = torch.from_numpy(mean_lookup[selected_np].astype(np.float32)).to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(selected)
        loss = torch.mean(torch.square(prediction - target))
        if not torch.isfinite(loss).item():
            raise FamilyInductiveRunError("mean MLP training loss became non-finite")
        loss.backward()
        optimizer.step()
        record: dict[str, Any] = {"epoch": epoch, "train_mean_mse": float(loss.detach().cpu())}

        if validation_indices is not None:
            if validation_target is None:
                raise AssertionError("validation target was not initialized")
            validation_prediction = _predict_mean_mlp(
                model, validation_indices, device=device
            )
            metrics = gene_mean_metrics(
                validation_prediction, validation_target, train_global
            )
            candidate = {
                "config": dict(config),
                "epoch": epoch,
                "validation_metrics": metrics,
            }
            record["validation"] = metrics
            if best_candidate is None or _selection_key(candidate) < _selection_key(best_candidate):
                best_candidate = candidate
                best_state = _copy_state(model)
                stale_epochs = 0
            else:
                stale_epochs += 1
        history.append(record)
        print(
            f"[family-inductive] mean-mlp epoch={epoch}/{epochs} "
            f"train_mse={record['train_mean_mse']:.6f}",
            flush=True,
        )
        if validation_indices is not None and stale_epochs >= int(patience):
            break

    if validation_indices is not None:
        if best_candidate is None or best_state is None:
            raise FamilyInductiveRunError("mean MLP did not produce a selectable epoch")
        model.load_state_dict(best_state)
        selected_epoch = int(best_candidate["epoch"])
        validation_metrics = best_candidate["validation_metrics"]
    else:
        selected_epoch = int(epochs)
        validation_metrics = None
    return model, {
        "config": dict(config),
        "seed": int(seed),
        "epochs_requested": int(epochs),
        "epochs_executed": len(history),
        "selected_epoch": selected_epoch,
        "validation_metrics": validation_metrics,
        "history": history,
        "gene_subsample": int(gene_subsample),
        "optimizer": "AdamW",
        "objective": "gene_mean_mse",
    }


def _cell_inputs(
    fold_data: FoldData, cell_indices: np.ndarray, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mutation = torch.from_numpy(
        np.asarray(fold_data.mutation.matrix[cell_indices], dtype=np.float32)
    ).to(device)
    expression = torch.from_numpy(
        np.asarray(fold_data.expression[cell_indices], dtype=np.float32)
    ).to(device)
    lineage = torch.from_numpy(
        np.asarray(fold_data.lineage[cell_indices], dtype=np.float32)
    ).to(device)
    return mutation, expression, lineage


def _predict_decomp_mean(
    model: FamilyInductiveModel,
    gene_indices: np.ndarray,
    *,
    device: str,
    chunk_size: int = 4096,
) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, gene_indices.size, chunk_size):
            selected = torch.from_numpy(gene_indices[start : start + chunk_size]).long().to(device)
            encoded = model.encode_genes(selected)
            chunks.append(model.mean_head(encoded).squeeze(-1).float().cpu().numpy())
    output = np.concatenate(chunks).astype(np.float64, copy=False)
    if output.shape != (gene_indices.size,) or not np.all(np.isfinite(output)):
        raise FamilyInductiveRunError("FAMILY-INDUCTIVE-DECOMP mean prediction is invalid")
    return output


def _predict_decomp(
    model: FamilyInductiveModel,
    fold_data: FoldData,
    gene_indices: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    selected = torch.from_numpy(gene_indices).long().to(device)
    all_cells = np.arange(fold_data.expression.shape[0], dtype=np.int64)
    residual_chunks: list[np.ndarray] = []
    reference_mean: np.ndarray | None = None
    with torch.inference_mode():
        for start in range(0, all_cells.size, int(batch_size)):
            cell_indices = all_cells[start : start + int(batch_size)]
            mean, raw_residual = model(
                *_cell_inputs(fold_data, cell_indices, device),
                gene_subset=selected,
            )
            observed_mean = mean.float().cpu().numpy().astype(np.float64, copy=False)
            if reference_mean is None:
                reference_mean = observed_mean.copy()
            elif not np.array_equal(reference_mean, observed_mean):
                raise FamilyInductiveRunError("gene mean changed across inference cell batches")
            residual_chunks.append(raw_residual.float().cpu().numpy())
    if reference_mean is None:
        raise FamilyInductiveRunError("no cells were available for decomposition inference")
    raw = np.concatenate(residual_chunks, axis=0).astype(np.float64, copy=False)
    expected = (all_cells.size, gene_indices.size)
    if raw.shape != expected or not np.all(np.isfinite(raw)):
        raise FamilyInductiveRunError(
            f"invalid raw residual prediction shape {raw.shape}; expected {expected}"
        )
    # This is intentionally the first and only centering of predicted residuals.
    centered = np.asarray(center_residual_by_gene(raw), dtype=np.float64)
    if not np.all(np.isfinite(centered)):
        raise FamilyInductiveRunError("centered residual prediction is non-finite")
    return reference_mean, centered


def _train_decomp(
    base: BaseData,
    fold_data: FoldData,
    train_indices: np.ndarray,
    validation_indices: np.ndarray | None,
    *,
    config: Mapping[str, Any],
    epochs: int,
    patience: int,
    batch_size: int,
    gene_subsample: int,
    device: str,
    seed: int,
    mean_only: bool = False,
    force_full_epochs: bool = False,
    requested_epochs: int | None = None,
) -> tuple[FamilyInductiveModel, dict[str, Any]]:
    budget_epochs = int(epochs if requested_epochs is None else requested_epochs)
    if int(epochs) <= 0 or budget_epochs < int(epochs):
        raise FamilyInductiveRunError(
            "decomposition epoch budget must be positive and no smaller than the execution cap"
        )
    set_global_seed(seed)
    model = FamilyInductiveModel(
        torch.from_numpy(np.asarray(base.embeddings, dtype=np.float32)),
        n_mut=int(fold_data.mutation.matrix.shape[1]),
        n_expr=int(fold_data.expression.shape[1]),
        n_lin=int(fold_data.lineage.shape[1]),
        hidden=int(config["hidden"]),
        rank=int(config["rank"]),
        freeze=True,
    ).to(device)
    if mean_only:
        model.configure_mean_only_training()
    parameter_audit = model.trainable_parameter_audit()
    if mean_only and int(parameter_audit["residual_path_trainable_parameter_count"]) != 0:
        raise FamilyInductiveRunError(
            "mean-only configuration left residual/cell parameters trainable"
        )
    if not mean_only and int(parameter_audit["residual_path_trainable_parameter_count"]) <= 0:
        raise FamilyInductiveRunError(
            "proposed decomposition has no active residual/cell parameters"
        )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    train_means, train_global = _global_gene_mean(base.gene_effect, train_indices)
    mean_lookup = np.full(len(base.genes), np.nan, dtype=np.float64)
    mean_lookup[train_indices] = train_means
    validation_target = (
        None
        if validation_indices is None
        else gene_mean_targets(base.gene_effect, validation_indices)
    )
    all_cells = np.arange(len(base.cell_ids), dtype=np.int64)
    rng = np.random.default_rng(derive_seed(seed, "decomp-batches"))
    best_candidate: dict[str, Any] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    stale_epochs = 0
    steps_each_epoch = optimizer_steps_per_epoch(len(base.cell_ids), int(batch_size))
    optimizer_steps_requested = budget_epochs * steps_each_epoch
    optimizer_steps_executed = 0
    gene_schedule_digest = hashlib.sha256()
    cell_schedule_digest = hashlib.sha256()

    for epoch in range(1, int(epochs) + 1):
        model.train()
        cell_order = rng.permutation(all_cells)
        weighted_loss = 0.0
        weighted_mean_loss = 0.0
        weighted_residual_loss = 0.0
        cells_seen = 0
        for start in range(0, cell_order.size, int(batch_size)):
            cell_indices = cell_order[start : start + int(batch_size)]
            if 0 < gene_subsample < train_indices.size:
                selected_np = np.sort(
                    rng.choice(train_indices, gene_subsample, replace=False)
                ).astype(np.int64)
            else:
                selected_np = train_indices
            schedule_prefix = np.asarray(
                [epoch, optimizer_steps_executed], dtype="<i8"
            ).tobytes()
            gene_schedule_digest.update(schedule_prefix)
            gene_schedule_digest.update(
                np.ascontiguousarray(selected_np, dtype="<i8").tobytes()
            )
            cell_schedule_digest.update(schedule_prefix)
            cell_schedule_digest.update(
                np.ascontiguousarray(cell_indices, dtype="<i8").tobytes()
            )
            selected = torch.from_numpy(selected_np).long().to(device)
            mean_np = mean_lookup[selected_np].astype(np.float32)
            mean_target = torch.from_numpy(mean_np).to(device)
            optimizer.zero_grad(set_to_none=True)
            if mean_only:
                encoded = model.encode_genes(selected)
                predicted_mean = model.mean_head(encoded).squeeze(-1)
                residual_loss = torch.zeros((), dtype=predicted_mean.dtype, device=device)
            else:
                absolute_np = np.asarray(
                    base.gene_effect[np.ix_(selected_np, cell_indices)].T,
                    dtype=np.float32,
                )
                residual_np = absolute_np - mean_np[None, :]
                residual_target = torch.from_numpy(residual_np).to(device)
                predicted_mean, raw_predicted_residual = model(
                    *_cell_inputs(fold_data, cell_indices, device),
                    gene_subset=selected,
                )
                finite = torch.isfinite(residual_target)
                residual_error = torch.where(
                    finite,
                    raw_predicted_residual - residual_target,
                    torch.zeros_like(raw_predicted_residual),
                )
                residual_loss = residual_error.square().sum() / finite.sum().clamp(min=1)
            mean_loss = torch.mean(torch.square(predicted_mean - mean_target))
            loss = (
                float(config["gene_mean_loss_weight"]) * mean_loss
                + (
                    0.0
                    if mean_only
                    else float(config["context_residual_loss_weight"])
                )
                * residual_loss
            )
            if not torch.isfinite(loss).item():
                raise FamilyInductiveRunError("decomposition training loss became non-finite")
            # The raw residual is never centered within a mini-batch.
            loss.backward()
            optimizer.step()
            optimizer_steps_executed += 1
            weight = int(cell_indices.size)
            weighted_loss += float(loss.detach().cpu()) * weight
            weighted_mean_loss += float(mean_loss.detach().cpu()) * weight
            weighted_residual_loss += float(residual_loss.detach().cpu()) * weight
            cells_seen += weight

        record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": weighted_loss / max(cells_seen, 1),
            "train_mean_mse": weighted_mean_loss / max(cells_seen, 1),
            "train_raw_residual_mse": weighted_residual_loss / max(cells_seen, 1),
        }
        if validation_indices is not None:
            if validation_target is None:
                raise AssertionError("validation target was not initialized")
            validation_prediction = _predict_decomp_mean(
                model, validation_indices, device=device
            )
            metrics = gene_mean_metrics(
                validation_prediction, validation_target, train_global
            )
            candidate = {
                "config": dict(config),
                "epoch": epoch,
                "validation_metrics": metrics,
            }
            record["validation"] = metrics
            if best_candidate is None or _selection_key(candidate) < _selection_key(best_candidate):
                best_candidate = candidate
                best_state = _copy_state(model)
                stale_epochs = 0
            else:
                stale_epochs += 1
        history.append(record)
        print(
            f"[family-inductive] {'mean-only' if mean_only else 'decomp'} "
            f"epoch={epoch}/{epochs} "
            f"loss={record['train_loss']:.6f} mean={record['train_mean_mse']:.6f} "
            f"raw_residual={record['train_raw_residual_mse']:.6f}",
            flush=True,
        )
        if (
            validation_indices is not None
            and not force_full_epochs
            and stale_epochs >= int(patience)
        ):
            break

    if validation_indices is not None:
        if best_candidate is None or best_state is None:
            raise FamilyInductiveRunError("decomposition model did not produce a selectable epoch")
        model.load_state_dict(best_state)
        selected_epoch = int(best_candidate["epoch"])
        validation_metrics = best_candidate["validation_metrics"]
    else:
        selected_epoch = int(epochs)
        validation_metrics = None
    expected_executed = len(history) * steps_each_epoch
    if optimizer_steps_executed != expected_executed:
        raise FamilyInductiveRunError(
            "decomposition optimizer-step accounting mismatch: "
            f"expected {expected_executed}, observed {optimizer_steps_executed}"
        )
    return model, {
        "config": dict(config),
        "training_mode": "DECOMP-MEAN-ONLY" if mean_only else "FAMILY-INDUCTIVE-DECOMP",
        "seed": int(seed),
        "epochs_requested": budget_epochs,
        "epochs_executed": len(history),
        "selected_epoch": selected_epoch,
        "validation_metrics": validation_metrics,
        "history": history,
        "batch_size": int(batch_size),
        "gene_subsample": int(gene_subsample),
        "optimizer": "AdamW",
        "objective": (
            "gene_mean_mse_with_residual_loss_fixed_zero"
            if mean_only
            else "separate_gene_mean_mse_plus_gene_centered_context_residual_mse"
        ),
        "context_residual_loss_weight_effective": (
            0.0 if mean_only else float(config["context_residual_loss_weight"])
        ),
        "model_architecture_sha256": _architecture_signature(model),
        **parameter_audit,
        "optimizer_steps_per_epoch": steps_each_epoch,
        "optimizer_steps_requested": optimizer_steps_requested,
        "optimizer_steps_executed": optimizer_steps_executed,
        "gene_sampling_schedule_sha256": gene_schedule_digest.hexdigest(),
        "cell_batch_schedule_sha256": cell_schedule_digest.hexdigest(),
        "early_stopping_enabled": bool(
            validation_indices is not None and not force_full_epochs
        ),
        "raw_residual_centered_during_training": False,
    }


def _finite_correlation(x: np.ndarray, y: np.ndarray, *, rank: bool) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 2:
        return float("nan")
    x = x[valid]
    y = y[valid]
    if rank:
        x = rankdata(x, method="average")
        y = rankdata(y, method="average")
    x = x - np.mean(x)
    y = y - np.mean(y)
    denominator = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    if denominator <= 0.0:
        return float("nan")
    return float(np.dot(x, y) / denominator)


def _residual_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    predicted = np.asarray(prediction, dtype=np.float64)
    observed = np.asarray(target, dtype=np.float64)
    if predicted.shape != observed.shape or predicted.ndim != 2:
        raise ValueError("residual prediction/target must be equal cells-by-genes matrices")
    valid = np.isfinite(observed)
    if np.any(valid & ~np.isfinite(predicted)) or not np.any(valid):
        raise FamilyInductiveRunError("residual evaluation contains invalid predictions/targets")
    error = np.where(valid, predicted - observed, 0.0)
    target_zero = np.where(valid, observed, 0.0)
    sse = float(np.square(error).sum(dtype=np.float64))
    zero_sse = float(np.square(target_zero).sum(dtype=np.float64))
    pearson = np.asarray(
        [_finite_correlation(predicted[:, j], observed[:, j], rank=False) for j in range(observed.shape[1])],
        dtype=np.float64,
    )
    spearman = np.asarray(
        [_finite_correlation(predicted[:, j], observed[:, j], rank=True) for j in range(observed.shape[1])],
        dtype=np.float64,
    )
    finite_pearson = pearson[np.isfinite(pearson)]
    finite_spearman = spearman[np.isfinite(spearman)]
    return {
        "mse": sse / int(valid.sum()),
        "r2_vs_zero": float(1.0 - sse / zero_sse) if zero_sse > 0.0 else float("nan"),
        "mean_within_gene_pearson": (
            float(np.mean(finite_pearson)) if finite_pearson.size else float("nan")
        ),
        "median_within_gene_pearson": (
            float(np.median(finite_pearson)) if finite_pearson.size else float("nan")
        ),
        "mean_within_gene_spearman": (
            float(np.mean(finite_spearman)) if finite_spearman.size else float("nan")
        ),
        "median_within_gene_spearman": (
            float(np.median(finite_spearman)) if finite_spearman.size else float("nan")
        ),
        "n_genes": int(observed.shape[1]),
        "n_finite_pearson": int(finite_pearson.size),
        "n_finite_spearman": int(finite_spearman.size),
        "n_observations": int(valid.sum()),
    }


def _subset_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    global_mean: float,
    mask: np.ndarray,
) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != target.shape:
        raise ValueError("stratum mask does not align with target")
    if not np.any(mask):
        return {"n": 0, "mse": None, "mae": None, "r2": None, "pearson": None, "spearman": None}
    return gene_mean_metrics(prediction[mask], target[mask], global_mean)


def _component_aggregates(
    values: np.ndarray, components: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = np.asarray(values, dtype=np.float64)
    component = np.asarray(components, dtype=str)
    if value.ndim != 1 or component.shape != value.shape or not np.all(np.isfinite(value)):
        raise ValueError("component aggregate inputs must be finite aligned vectors")
    unique, inverse = np.unique(component, return_inverse=True)
    sums = np.bincount(inverse, weights=value, minlength=unique.size).astype(np.float64)
    sizes = np.bincount(inverse, minlength=unique.size).astype(np.float64)
    return unique, sums, sizes


def _bootstrap_component_statistic(
    component_arrays: Sequence[np.ndarray],
    statistic: Callable[..., np.ndarray],
    *,
    point: float,
    n_bootstrap: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    count = int(component_arrays[0].size)
    if count <= 0 or any(array.shape != (count,) for array in component_arrays):
        raise ValueError("bootstrap component arrays must be non-empty and aligned")
    rng = np.random.default_rng(seed)
    replicates = np.empty(n_bootstrap, dtype=np.float64)
    chunk_size = min(128, n_bootstrap)
    for start in range(0, n_bootstrap, chunk_size):
        stop = min(start + chunk_size, n_bootstrap)
        sampled = rng.integers(0, count, size=(stop - start, count))
        totals = [np.sum(array[sampled], axis=1) for array in component_arrays]
        replicates[start:stop] = statistic(*totals)
    finite = replicates[np.isfinite(replicates)]
    if finite.size != n_bootstrap:
        raise FamilyInductiveRunError("component bootstrap generated undefined replicates")
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(finite, [alpha, 1.0 - alpha])
    return {
        "point": float(point),
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence": float(confidence),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "n_components": count,
        "unit": "sequence_component",
    }


def _bootstrap_mse_improvement(
    candidate_error: np.ndarray,
    reference_error: np.ndarray,
    components: Sequence[str],
    *,
    n_bootstrap: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    improvement = np.square(reference_error) - np.square(candidate_error)
    unique, component_sum, component_size = _component_aggregates(improvement, components)
    point = float(np.mean(improvement))
    result = _bootstrap_component_statistic(
        [component_sum, component_size],
        lambda numerator, denominator: numerator / denominator,
        point=point,
        n_bootstrap=n_bootstrap,
        seed=seed,
        confidence=confidence,
    )
    result.update(
        {
            "estimand": "reference_mse_minus_candidate_mse",
            "direction": "positive_favors_candidate",
            "n_genes": int(improvement.size),
            "component_ids_sha256": _hash_strings(unique.tolist()),
        }
    )
    return result


def _bootstrap_delta_r2(
    candidate_error: np.ndarray,
    reference_error: np.ndarray,
    global_error: np.ndarray,
    components: Sequence[str],
    *,
    n_bootstrap: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    candidate_sq = np.square(candidate_error)
    reference_sq = np.square(reference_error)
    global_sq = np.square(global_error)
    unique, candidate_sum, _ = _component_aggregates(candidate_sq, components)
    _, reference_sum, _ = _component_aggregates(reference_sq, components)
    _, global_sum, _ = _component_aggregates(global_sq, components)
    denominator = float(np.sum(global_sq))
    if denominator <= 0.0:
        raise FamilyInductiveRunError("outer-test global baseline SSE is zero")
    point = float((np.sum(reference_sq) - np.sum(candidate_sq)) / denominator)
    result = _bootstrap_component_statistic(
        [candidate_sum, reference_sum, global_sum],
        lambda candidate, reference, global_: (reference - candidate) / global_,
        point=point,
        n_bootstrap=n_bootstrap,
        seed=seed,
        confidence=confidence,
    )
    result.update(
        {
            "estimand": "candidate_r2_minus_reference_r2",
            "direction": "positive_favors_candidate",
            "n_genes": int(candidate_sq.size),
            "component_ids_sha256": _hash_strings(unique.tolist()),
        }
    )
    return result


def _bootstrap_residual_r2(
    prediction: np.ndarray,
    target: np.ndarray,
    components: Sequence[str],
    *,
    n_bootstrap: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    valid = np.isfinite(target)
    error = np.where(valid, prediction - target, 0.0)
    observed = np.where(valid, target, 0.0)
    prediction_sse = np.sum(np.square(error), axis=0, dtype=np.float64)
    zero_sse = np.sum(np.square(observed), axis=0, dtype=np.float64)
    unique, prediction_sum, _ = _component_aggregates(prediction_sse, components)
    _, zero_sum, _ = _component_aggregates(zero_sse, components)
    denominator = float(np.sum(zero_sse))
    if denominator <= 0.0:
        raise FamilyInductiveRunError("context residual zero-baseline SSE is zero")
    point = float(1.0 - np.sum(prediction_sse) / denominator)
    result = _bootstrap_component_statistic(
        [prediction_sum, zero_sum],
        lambda prediction_, zero_: 1.0 - prediction_ / zero_,
        point=point,
        n_bootstrap=n_bootstrap,
        seed=seed,
        confidence=confidence,
    )
    result.update(
        {
            "estimand": "context_residual_r2_vs_zero",
            "direction": "positive_favors_model",
            "n_genes": int(target.shape[1]),
            "component_ids_sha256": _hash_strings(unique.tolist()),
        }
    )
    return result


def _partition_audit(
    assignments: Sequence[FamilyAssignment], split: Any
) -> dict[str, Any]:
    partitions = {
        "inner_train": np.asarray(split.inner_train, dtype=np.int64),
        "inner_validation": np.asarray(split.inner_validation, dtype=np.int64),
        "outer_train": np.asarray(split.outer_train, dtype=np.int64),
        "outer_test": np.asarray(split.outer_test, dtype=np.int64),
    }
    for name, indices in partitions.items():
        if indices.size == 0:
            raise FamilyInductiveRunError(f"family partition is empty: {name}")
    component_sets = {
        name: {assignments[int(index)].component_id for index in indices}
        for name, indices in partitions.items()
    }
    for left, right in (
        ("inner_train", "inner_validation"),
        ("outer_train", "outer_test"),
    ):
        overlap = component_sets[left].intersection(component_sets[right])
        if overlap:
            raise FamilyInductiveRunError(
                f"sequence components overlap {left}/{right}: {sorted(overlap)[:5]}"
            )
    return {
        name: {
            "n_genes": int(indices.size),
            "n_components": len(component_sets[name]),
            "gene_indices_sha256": _hash_array(indices.astype("<i8")),
            "component_ids_sha256": _hash_strings(sorted(component_sets[name])),
        }
        for name, indices in partitions.items()
    }


def _stack_base_matrix(grid: Mapping[str, Any], config: Mapping[str, Any]) -> np.ndarray:
    key = (int(config["neighbors"]), float(config["temperature"]))
    values = np.column_stack(
        [
            np.asarray(grid["global"], dtype=np.float64),
            np.asarray(grid["weighted_knn"][key], dtype=np.float64),
            np.asarray(grid["ridge"][float(config["alpha"])], dtype=np.float64),
        ]
    )
    if values.ndim != 2 or values.shape[1] != len(STACK_METHOD_ORDER):
        raise FamilyInductiveRunError("stack base prediction matrix is malformed")
    return values


def _trace_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in candidate.items()
        if key not in {"validation_prediction", "validation_weight", "validation_z"}
    }


def _select_sequence_methods(
    base: BaseData,
    split: Any,
    assignments: Sequence[FamilyAssignment],
    contract: Mapping[str, Any],
    primary_contract: Mapping[str, Any],
    *,
    training_seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    inner_target, inner_global = _global_gene_mean(base.gene_effect, split.inner_train)
    validation_target = gene_mean_targets(base.gene_effect, split.inner_validation)
    traces: dict[str, Any] = {}

    grid = primary_contract["candidate_grid"]
    uniform_grid = contract["baseline_grids"]["uniform_knn_neighbors"]
    weighted_grid = grid["knn_neighbors"]
    temperature_grid = grid["knn_temperature"]
    ridge_grid = grid["ridge_alpha"]
    inner_components = [
        assignments[int(index)].component_id for index in split.inner_train
    ]
    crossfit, crossfit_audit = _component_crossfit_prediction_grid(
        base.embeddings[split.inner_train],
        inner_target,
        inner_components,
        training_seed=int(training_seed),
        n_folds=int(primary_contract["inner_crossfit_folds"]),
        uniform_neighbors=uniform_grid,
        weighted_neighbors=weighted_grid,
        temperatures=temperature_grid,
        ridge_alphas=ridge_grid,
    )
    validation_grid = _sequence_prediction_grid(
        base.embeddings[split.inner_train],
        inner_target,
        base.embeddings[split.inner_validation],
        uniform_neighbors=uniform_grid,
        weighted_neighbors=weighted_grid,
        temperatures=temperature_grid,
        ridge_alphas=ridge_grid,
    )

    knn_candidates: list[dict[str, Any]] = []
    for neighbors in uniform_grid:
        prediction = validation_grid["uniform_knn"][int(neighbors)]
        knn_candidates.append(
            {
                "config": {"neighbors": int(neighbors), "weights": "uniform", "metric": "cosine"},
                "epoch": 0,
                "validation_metrics": gene_mean_metrics(
                    prediction, validation_target, inner_global
                ),
                "validation_prediction": prediction,
            }
        )
    knn_selected = min(knn_candidates, key=_selection_key)
    traces["SEQUENCE-KNN"] = {
        "candidates": [_trace_candidate(value) for value in knn_candidates],
        "selected": _trace_candidate(knn_selected),
    }

    weighted_candidates: list[dict[str, Any]] = []
    for neighbors, temperature in itertools.product(weighted_grid, temperature_grid):
        prediction = validation_grid["weighted_knn"][(int(neighbors), float(temperature))]
        weighted_candidates.append(
            {
                "config": {
                    "neighbors": int(neighbors),
                    "temperature": float(temperature),
                    "weights": "softmax_cosine",
                    "formula": "exp((similarity-max_similarity)/temperature)",
                    "metric": "cosine",
                },
                "epoch": 0,
                "validation_metrics": gene_mean_metrics(
                    prediction, validation_target, inner_global
                ),
                "validation_prediction": prediction,
            }
        )
    weighted_selected = min(weighted_candidates, key=_selection_key)
    traces["SEQUENCE-WEIGHTED-KNN"] = {
        "candidates": [_trace_candidate(value) for value in weighted_candidates],
        "selected": _trace_candidate(weighted_selected),
    }

    ridge_candidates: list[dict[str, Any]] = []
    for alpha in ridge_grid:
        prediction = validation_grid["ridge"][float(alpha)]
        ridge_candidates.append(
            {
                "config": {
                    "alpha": float(alpha),
                    "standardization": "inner_train_only",
                    "solver": "lsqr",
                    "tol": 1e-8,
                    "max_iter": 10000,
                },
                "epoch": 0,
                "validation_metrics": gene_mean_metrics(
                    prediction, validation_target, inner_global
                ),
                "validation_prediction": prediction,
            }
        )
    ridge_selected = min(ridge_candidates, key=_selection_key)
    traces["PLM-RIDGE"] = {
        "candidates": [_trace_candidate(value) for value in ridge_candidates],
        "selected": _trace_candidate(ridge_selected),
    }

    global_prediction = validation_grid["global"]
    global_selected = {
        "config": {"definition": "mean_of_inner_training_gene_means"},
        "epoch": 0,
        "validation_metrics": gene_mean_metrics(
            global_prediction, validation_target, inner_global
        ),
        "validation_prediction": global_prediction,
    }
    traces["GLOBAL-TRAIN-MEAN"] = {
        "candidates": [_trace_candidate(global_selected)],
        "selected": _trace_candidate(global_selected),
    }

    static_candidates: list[dict[str, Any]] = []
    primary_candidates: list[dict[str, Any]] = []
    for neighbors, temperature, alpha in itertools.product(
        weighted_grid, temperature_grid, ridge_grid
    ):
        base_config = {
            "neighbors": int(neighbors),
            "temperature": float(temperature),
            "alpha": float(alpha),
        }
        crossfit_base = _stack_base_matrix(crossfit, base_config)
        validation_base = _stack_base_matrix(validation_grid, base_config)
        static_fit = _fit_distance_stack(
            crossfit_base,
            inner_target,
            crossfit["distance"],
            gate_lambda=0.0,
            static=True,
        )
        static_prediction, static_z, static_weight = _predict_distance_stack(
            static_fit, validation_base, validation_grid["distance"]
        )
        static_candidates.append(
            {
                "config": {**base_config, "static": True},
                "epoch": 0,
                "validation_metrics": gene_mean_metrics(
                    static_prediction, validation_target, inner_global
                ),
                "fit": static_fit,
                "validation_prediction": static_prediction,
                "validation_z": static_z,
                "validation_weight": static_weight,
            }
        )
        for gate_lambda in grid["gate_lambda"]:
            primary_fit = _fit_distance_stack(
                crossfit_base,
                inner_target,
                crossfit["distance"],
                gate_lambda=float(gate_lambda),
                static=False,
            )
            primary_prediction, primary_z, primary_weight = _predict_distance_stack(
                primary_fit, validation_base, validation_grid["distance"]
            )
            primary_candidates.append(
                {
                    "config": {
                        **base_config,
                        "gate_lambda": float(gate_lambda),
                        "static": False,
                    },
                    "epoch": 0,
                    "validation_metrics": gene_mean_metrics(
                        primary_prediction, validation_target, inner_global
                    ),
                    "fit": primary_fit,
                    "validation_prediction": primary_prediction,
                    "validation_z": primary_z,
                    "validation_weight": primary_weight,
                }
            )
    static_selected = min(static_candidates, key=_selection_key)
    primary_selected = min(primary_candidates, key=_selection_key)
    traces["STATIC-CONVEX-STACK"] = {
        "candidates": [_trace_candidate(value) for value in static_candidates],
        "selected": _trace_candidate(static_selected),
        "lambda_dimension": "absent_by_registered_design",
    }
    traces[PRIMARY_METHOD] = {
        "candidates": [_trace_candidate(value) for value in primary_candidates],
        "selected": _trace_candidate(primary_selected),
        "selection_rule": primary_contract["selection_rule"],
    }
    return {
        "GLOBAL-TRAIN-MEAN": global_selected,
        "SEQUENCE-KNN": knn_selected,
        "SEQUENCE-WEIGHTED-KNN": weighted_selected,
        "PLM-RIDGE": ridge_selected,
        "STATIC-CONVEX-STACK": static_selected,
        PRIMARY_METHOD: primary_selected,
    }, traces, {
        "inner_component_crossfit": crossfit_audit,
        "solver_contract": dict(STACK_SOLVER_CONTRACT),
        "crossfit_hash_contract": CROSSFIT_HASH_CONTRACT,
    }


def _refit_sequence_methods(
    base: BaseData,
    split: Any,
    assignments: Sequence[FamilyAssignment],
    selected: Mapping[str, Mapping[str, Any]],
    primary_contract: Mapping[str, Any],
    *,
    training_seed: int,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, Any],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, dict[str, np.ndarray]],
]:
    outer_target, _ = _global_gene_mean(base.gene_effect, split.outer_train)
    uniform_neighbors = [
        int(selected["SEQUENCE-KNN"]["config"]["neighbors"])
    ]
    weighted_neighbors = sorted(
        {
            int(selected[name]["config"]["neighbors"])
            for name in (
                "SEQUENCE-WEIGHTED-KNN",
                "STATIC-CONVEX-STACK",
                PRIMARY_METHOD,
            )
        }
    )
    temperatures = sorted(
        {
            float(selected[name]["config"]["temperature"])
            for name in (
                "SEQUENCE-WEIGHTED-KNN",
                "STATIC-CONVEX-STACK",
                PRIMARY_METHOD,
            )
        }
    )
    ridge_alphas = sorted(
        {
            float(selected[name]["config"]["alpha"])
            for name in ("PLM-RIDGE", "STATIC-CONVEX-STACK", PRIMARY_METHOD)
        }
    )
    outer_components = [
        assignments[int(index)].component_id for index in split.outer_train
    ]
    crossfit, crossfit_audit = _component_crossfit_prediction_grid(
        base.embeddings[split.outer_train],
        outer_target,
        outer_components,
        training_seed=int(training_seed),
        n_folds=int(primary_contract["outer_refit_crossfit_folds"]),
        uniform_neighbors=uniform_neighbors,
        weighted_neighbors=weighted_neighbors,
        temperatures=temperatures,
        ridge_alphas=ridge_alphas,
    )
    query_grid = _sequence_prediction_grid(
        base.embeddings[split.outer_train],
        outer_target,
        base.embeddings[split.outer_test],
        uniform_neighbors=uniform_neighbors,
        weighted_neighbors=weighted_neighbors,
        temperatures=temperatures,
        ridge_alphas=ridge_alphas,
    )
    primary_config = selected[PRIMARY_METHOD]["config"]
    primary_query_base = _stack_base_matrix(query_grid, primary_config)
    primary_fit = _fit_distance_stack(
        _stack_base_matrix(crossfit, primary_config),
        outer_target,
        crossfit["distance"],
        gate_lambda=float(primary_config["gate_lambda"]),
        static=False,
    )
    primary_prediction, primary_z, primary_weight = _predict_distance_stack(
        primary_fit,
        primary_query_base,
        query_grid["distance"],
    )
    static_config = selected["STATIC-CONVEX-STACK"]["config"]
    static_query_base = _stack_base_matrix(query_grid, static_config)
    static_fit = _fit_distance_stack(
        _stack_base_matrix(crossfit, static_config),
        outer_target,
        crossfit["distance"],
        gate_lambda=0.0,
        static=True,
    )
    static_prediction, _, _ = _predict_distance_stack(
        static_fit,
        static_query_base,
        query_grid["distance"],
    )
    uniform_config = selected["SEQUENCE-KNN"]["config"]
    weighted_config = selected["SEQUENCE-WEIGHTED-KNN"]["config"]
    ridge_config = selected["PLM-RIDGE"]["config"]
    predictions = {
        "GLOBAL-TRAIN-MEAN": np.asarray(query_grid["global"], dtype=np.float64),
        "SEQUENCE-KNN": np.asarray(
            query_grid["uniform_knn"][int(uniform_config["neighbors"])],
            dtype=np.float64,
        ),
        "SEQUENCE-WEIGHTED-KNN": np.asarray(
            query_grid["weighted_knn"][(
                int(weighted_config["neighbors"]),
                float(weighted_config["temperature"]),
            )],
            dtype=np.float64,
        ),
        "PLM-RIDGE": np.asarray(
            query_grid["ridge"][float(ridge_config["alpha"])], dtype=np.float64
        ),
        "STATIC-CONVEX-STACK": static_prediction,
        PRIMARY_METHOD: primary_prediction,
    }
    audit = {
        "outer_component_crossfit": crossfit_audit,
        "primary_config": dict(primary_config),
        "primary_fit": primary_fit,
        "static_ablation_config": dict(static_config),
        "static_ablation_fit": static_fit,
        "per_query_weight_semantics": "a_m_plus_b_m_times_distance_gate_z",
        "weight_column_order": list(STACK_METHOD_ORDER),
    }
    return (
        predictions,
        audit,
        np.asarray(query_grid["distance"], dtype=np.float64),
        primary_z,
        primary_weight,
        {
            PRIMARY_METHOD: {
                method: np.asarray(primary_query_base[:, column], dtype=np.float64)
                for column, method in enumerate(STACK_METHOD_ORDER)
            },
            "STATIC-CONVEX-STACK": {
                method: np.asarray(static_query_base[:, column], dtype=np.float64)
                for column, method in enumerate(STACK_METHOD_ORDER)
            },
        },
    )


def _run_experiment(
    args: argparse.Namespace,
    *,
    lock: dict[str, Any],
    lock_provenance: dict[str, Any],
    run_dir: Path,
    device: str,
    server_identity: Mapping[str, Any],
) -> Path:
    started_at = _utc_now()
    wall_start = time.monotonic()
    contract = lock["training_contract"]
    code_paths, input_paths = _source_paths(lock)
    code_records_start = _records(code_paths)
    input_records_start = _records(input_paths)
    if lock_provenance["lock"] not in code_records_start:
        raise FamilyInductiveRunError("analysis lock changed before run initialization")
    set_global_seed(int(args.training_seed))
    environment = _environment_record(device)

    base = load_base_data(
        pilot=bool(args.pilot),
        seed=PILOT_SUBSET_SEED,
        pilot_cells=int(contract["pilot"]["cells"]),
        pilot_genes=int(contract["pilot"]["genes"]),
    )
    protein_length = _load_protein_lengths(lock, base.genes)
    full_genes = [
        line.strip()
        for line in _safe_repo_path(
            lock["locked_assets"]["gene_manifest"]["path"]
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    family_manifest = _family_manifest_for_identity(lock, args.identity_percent)
    full_assignments = derive_seeded_family_assignments(
        family_manifest,
        full_genes,
        split_seed=int(args.split_seed),
    )
    assignment_by_gene = {assignment.gene: assignment for assignment in full_assignments}
    if len(assignment_by_gene) != len(full_assignments):
        raise FamilyInductiveRunError("full family assignment contains duplicate genes")
    try:
        assignments = [assignment_by_gene[gene] for gene in base.genes]
    except KeyError as exc:
        raise FamilyInductiveRunError(f"loaded gene lacks a family assignment: {exc}") from exc
    split = make_family_split(assignments, int(args.outer_fold))
    partition_audit = _partition_audit(assignments, split)

    preprocessing_seed = derive_seed(
        int(args.training_seed), "family-inductive-preprocessing"
    )
    fold_data = fit_family_preprocessing(
        base,
        seed=preprocessing_seed,
        n_expr_pc=int(contract["n_expr_pc"]),
        top_mut=int(contract["top_mut"]),
    )
    preprocessing = preprocessing_record(fold_data)
    inner_epochs = int(
        contract["pilot"]["inner_max_epochs"]
        if args.pilot
        else contract["inner_max_epochs"]
    )
    patience = int(
        contract["pilot"]["patience"] if args.pilot else contract["patience"]
    )
    gene_subsample = int(
        contract["pilot"]["gene_subsample"]
        if args.pilot
        else contract["gene_subsample"]
    )
    batch_size = int(contract["batch_size"])

    selected_sequence, selection_traces, inner_stack_audit = _select_sequence_methods(
        base,
        split,
        assignments,
        contract,
        lock["primary_method_contract"],
        training_seed=int(args.training_seed),
    )
    inner_validation_target = gene_mean_targets(
        base.gene_effect, split.inner_validation
    )

    mean_mlp_runs: list[dict[str, Any]] = []
    for grid_index, config in enumerate(_expand_grid(contract["mean_mlp_grid"])):
        seed = derive_seed(
            int(args.training_seed), "inner-mean-mlp", grid_index, _canonical_config(config)
        )
        model, training = _train_mean_mlp(
            base,
            split.inner_train,
            split.inner_validation,
            config=config,
            epochs=inner_epochs,
            patience=patience,
            gene_subsample=gene_subsample,
            device=device,
            seed=seed,
        )
        mean_mlp_runs.append(
            {
                "config": dict(config),
                "epoch": int(training["selected_epoch"]),
                "validation_metrics": training["validation_metrics"],
                "training": training,
                "grid_index": grid_index,
            }
        )
        del model
        _empty_device_cache(device)
    selected_mean_position = min(range(len(mean_mlp_runs)), key=lambda index: _selection_key(mean_mlp_runs[index]))
    selected_mean = mean_mlp_runs[selected_mean_position]
    selection_traces["IDFREE-PLM-MEAN-MLP"] = {
        "candidates": mean_mlp_runs,
        "selected": {key: value for key, value in selected_mean.items() if key != "training"},
    }

    decomp_runs: list[dict[str, Any]] = []
    mean_only_runs: list[dict[str, Any]] = []
    matched_candidate_audits: list[dict[str, Any]] = []
    for grid_index, config in enumerate(_expand_grid(contract["decomposition_grid"])):
        seed = derive_seed(
            int(args.training_seed), "inner-decomp", grid_index, _canonical_config(config)
        )
        model, training = _train_decomp(
            base,
            fold_data,
            split.inner_train,
            split.inner_validation,
            config=config,
            epochs=inner_epochs,
            patience=patience,
            batch_size=batch_size,
            gene_subsample=gene_subsample,
            device=device,
            seed=seed,
        )
        decomp_runs.append(
            {
                "config": dict(config),
                "epoch": int(training["selected_epoch"]),
                "validation_metrics": training["validation_metrics"],
                "training": training,
                "grid_index": grid_index,
            }
        )
        del model
        _empty_device_cache(device)
        mean_only_model, mean_only_training = _train_decomp(
            base,
            fold_data,
            split.inner_train,
            split.inner_validation,
            config=config,
            epochs=int(training["epochs_executed"]),
            patience=patience,
            batch_size=batch_size,
            gene_subsample=gene_subsample,
            device=device,
            seed=seed,
            mean_only=True,
            force_full_epochs=True,
            requested_epochs=inner_epochs,
        )
        matched_candidate_audits.append(
            {
                "grid_index": grid_index,
                "config": dict(config),
                **_assert_matched_optimizer_invariants(
                    training, mean_only_training
                ),
            }
        )
        mean_only_runs.append(
            {
                "config": dict(config),
                "epoch": int(training["selected_epoch"]),
                "validation_metrics": mean_only_training["history"][
                    int(training["selected_epoch"]) - 1
                ]["validation"],
                "training": mean_only_training,
                "grid_index": grid_index,
                "paired_proposed_grid_index": grid_index,
                "selection_basis": (
                    "matched_to_proposed_candidate_config_and_selected_epoch"
                ),
            }
        )
        del mean_only_model
        _empty_device_cache(device)
    selected_decomp_position = min(range(len(decomp_runs)), key=lambda index: _selection_key(decomp_runs[index]))
    selected_decomp = decomp_runs[selected_decomp_position]
    selection_traces["FAMILY-INDUCTIVE-DECOMP"] = {
        "candidates": decomp_runs,
        "selected": {key: value for key, value in selected_decomp.items() if key != "training"},
    }
    selected_mean_only_position = selected_decomp_position
    selected_mean_only = mean_only_runs[selected_mean_only_position]
    selection_traces["DECOMP-MEAN-ONLY"] = {
        "candidates": mean_only_runs,
        "selected": {
            key: value
            for key, value in selected_mean_only.items()
            if key != "training"
        },
        "matched_candidate_audits": matched_candidate_audits,
        "selection_basis": "matched_to_selected_proposed_config_and_epoch",
    }
    risk_scorer = DistanceRiskScorer()
    risk_scorer.fit(
        base.embeddings[split.inner_train],
        base.embeddings[split.inner_validation],
        np.abs(
            np.asarray(
                selected_sequence[PRIMARY_METHOD]["validation_prediction"],
                dtype=np.float64,
            )
            - inner_validation_target
        ),
    )

    baseline_validation_candidates = {
        name: selected_sequence[name]
        for name in (
            "GLOBAL-TRAIN-MEAN",
            "SEQUENCE-KNN",
            "SEQUENCE-WEIGHTED-KNN",
            "PLM-RIDGE",
            "STATIC-CONVEX-STACK",
        )
    }
    baseline_validation_candidates.update({
        "IDFREE-PLM-MEAN-MLP": selected_mean,
        "DECOMP-MEAN-ONLY": selected_mean_only,
    })
    job_selected_baseline = min(
        baseline_validation_candidates,
        key=lambda name: _job_baseline_selection_key(
            name, baseline_validation_candidates[name]
        ),
    )

    # Fresh outer refits use only outer-training outcomes and selected inner settings.
    _, outer_global = _global_gene_mean(base.gene_effect, split.outer_train)
    test_embeddings = base.embeddings[split.outer_test]
    (
        outer_predictions,
        outer_stack_audit,
        distance,
        distance_gate_z,
        distance_stack_weight,
        stack_base_predictions,
    ) = _refit_sequence_methods(
        base,
        split,
        assignments,
        selected_sequence,
        lock["primary_method_contract"],
        training_seed=int(args.training_seed),
    )
    final_mean_seed = derive_seed(int(args.training_seed), "outer-mean-mlp-refit")
    final_mean_model, final_mean_training = _train_mean_mlp(
        base,
        split.outer_train,
        None,
        config=selected_mean["config"],
        epochs=int(selected_mean["epoch"]),
        patience=patience,
        gene_subsample=gene_subsample,
        device=device,
        seed=final_mean_seed,
    )
    outer_predictions["IDFREE-PLM-MEAN-MLP"] = _predict_mean_mlp(
        final_mean_model, split.outer_test, device=device
    )
    del final_mean_model
    _empty_device_cache(device)

    final_decomp_seed = derive_seed(int(args.training_seed), "outer-decomp-refit")
    final_mean_only_model, final_mean_only_training = _train_decomp(
        base,
        fold_data,
        split.outer_train,
        None,
        config=selected_mean_only["config"],
        epochs=int(selected_mean_only["epoch"]),
        patience=patience,
        batch_size=batch_size,
        gene_subsample=gene_subsample,
        device=device,
        seed=final_decomp_seed,
        mean_only=True,
        force_full_epochs=True,
    )
    outer_predictions["DECOMP-MEAN-ONLY"] = _predict_decomp_mean(
        final_mean_only_model, split.outer_test, device=device
    )
    del final_mean_only_model
    _empty_device_cache(device)

    final_decomp_model, final_decomp_training = _train_decomp(
        base,
        fold_data,
        split.outer_train,
        None,
        config=selected_decomp["config"],
        epochs=int(selected_decomp["epoch"]),
        patience=patience,
        batch_size=batch_size,
        gene_subsample=gene_subsample,
        device=device,
        seed=final_decomp_seed,
    )
    final_matched_optimizer_audit = _assert_matched_optimizer_invariants(
        final_decomp_training, final_mean_only_training
    )
    proposed_mean, proposed_residual = _predict_decomp(
        final_decomp_model,
        fold_data,
        split.outer_test,
        batch_size=batch_size,
        device=device,
    )
    del final_decomp_model
    _empty_device_cache(device)
    outer_predictions["FAMILY-INDUCTIVE-DECOMP"] = proposed_mean
    primary_mean = np.asarray(outer_predictions[PRIMARY_METHOD], dtype=np.float64)
    selected_baseline_prediction = np.asarray(
        outer_predictions[job_selected_baseline], dtype=np.float64
    ).copy()
    risk_score = risk_scorer.score(
        base.embeddings[split.outer_train], test_embeddings
    )
    secondary_total = proposed_mean[None, :] + proposed_residual

    # Single outer-test outcome access begins here, after all predictions and the
    # validation-fitted descriptive risk scorer are fixed.
    outer_target = gene_mean_targets(base.gene_effect, split.outer_test)
    observed_gene_effect = np.asarray(
        base.gene_effect[split.outer_test].T, dtype=np.float64
    )
    observed_residual = np.asarray(
        context_residual_targets(base.gene_effect, split.outer_test).T,
        dtype=np.float64,
    )

    primary = {
        name: gene_mean_metrics(prediction, outer_target, outer_global)
        for name, prediction in outer_predictions.items()
    }
    secondary = _residual_metrics(proposed_residual, observed_residual)
    component_ids = np.asarray(
        [assignments[int(index)].component_id for index in split.outer_test],
        dtype=str,
    )
    component_sizes_full: dict[str, int] = {}
    for assignment in full_assignments:
        component_sizes_full[assignment.component_id] = (
            component_sizes_full.get(assignment.component_id, 0) + 1
        )
    component_size = np.asarray(
        [component_sizes_full[value] for value in component_ids], dtype=np.int64
    )
    test_protein_length = np.asarray(
        protein_length[split.outer_test], dtype=np.int64
    )
    singleton_mask = component_size == 1
    robustness: dict[str, Any] = {
        "component_multiplicity": {
            "singleton": _subset_metrics(
                primary_mean, outer_target, outer_global, singleton_mask
            ),
            "multi_gene": _subset_metrics(
                primary_mean, outer_target, outer_global, ~singleton_mask
            ),
        }
    }
    component_size_bins = lock["robustness_strata"]["component_size_bins"]
    robustness["component_size"] = {
        str(bin_spec["label"]): {
            "minimum_size": int(bin_spec["minimum"]),
            "maximum_size": (
                None if bin_spec["maximum"] is None else int(bin_spec["maximum"])
            ),
            "metrics": _subset_metrics(
                primary_mean,
                outer_target,
                outer_global,
                (component_size >= int(bin_spec["minimum"]))
                & (
                    True
                    if bin_spec["maximum"] is None
                    else component_size <= int(bin_spec["maximum"])
                ),
            ),
        }
        for bin_spec in component_size_bins
    }
    quartile_edges = np.quantile(distance, [0.0, 0.25, 0.5, 0.75, 1.0])
    quartile = np.searchsorted(quartile_edges[1:-1], distance, side="right")
    robustness["distance_quartiles"] = {
        f"Q{index + 1}": {
            "bounds": [float(quartile_edges[index]), float(quartile_edges[index + 1])],
            "metrics": _subset_metrics(
                primary_mean, outer_target, outer_global, quartile == index
            ),
        }
        for index in range(4)
    }
    protein_quantiles = lock["robustness_strata"]["protein_length"]["quantiles"]
    protein_length_edges = np.quantile(
        test_protein_length.astype(np.float64), protein_quantiles
    )
    protein_length_quartile = np.searchsorted(
        protein_length_edges[1:-1], test_protein_length, side="right"
    )
    robustness["protein_length_quartiles"] = {
        f"Q{index + 1}": {
            "bounds_residues": [
                float(protein_length_edges[index]),
                float(protein_length_edges[index + 1]),
            ],
            "metrics": _subset_metrics(
                primary_mean,
                outer_target,
                outer_global,
                protein_length_quartile == index,
            ),
        }
        for index in range(4)
    }

    selective: dict[str, Any] = {}
    risk_order = np.argsort(risk_score, kind="mergesort")
    for retained_fraction in lock["coverage_levels"]:
        retained = max(
            1, int(math.ceil(float(retained_fraction) * outer_target.size))
        )
        retained_indices = risk_order[:retained]
        mask = np.zeros(outer_target.size, dtype=bool)
        mask[retained_indices] = True
        selective[str(float(retained_fraction))] = {
            "requested_retained_fraction": float(retained_fraction),
            "realized_retained_fraction": float(retained / outer_target.size),
            "n": retained,
            "maximum_retained_distance_risk_score": float(
                np.max(risk_score[retained_indices])
            ),
            "metrics": _subset_metrics(
                primary_mean, outer_target, outer_global, mask
            ),
        }
    absolute_outer_error = np.abs(primary_mean - outer_target)
    risk_score_evaluation = {
        **lock["selective_risk_score"],
        "schema": "DepPLM-validation-fitted-distance-risk-score-v2",
        "fit_scope": "inner_component_heldout_validation_only",
        "fit_method": PRIMARY_METHOD,
        "mean_score": float(np.mean(risk_score)),
        "median_score": float(np.median(risk_score)),
        "outer_test_descriptive_pearson_with_absolute_error": _finite_correlation(
            risk_score, absolute_outer_error, rank=False
        ),
        "outer_test_descriptive_spearman_with_absolute_error": _finite_correlation(
            risk_score, absolute_outer_error, rank=True
        ),
    }

    bootstrap_replicates = int(
        contract["pilot"]["bootstrap_replicates"]
        if args.pilot
        else lock["bootstrap"]["replicates"]
    )
    bootstrap_seed = int(lock["bootstrap"]["seed"])
    confidence = float(lock["bootstrap"]["confidence_level"])
    primary_error = primary_mean - outer_target
    global_error = outer_predictions["GLOBAL-TRAIN-MEAN"] - outer_target
    selected_baseline_error = selected_baseline_prediction - outer_target
    bootstrap = {
        "primary_mse_improvement_vs_global": _bootstrap_mse_improvement(
            primary_error,
            global_error,
            component_ids,
            n_bootstrap=bootstrap_replicates,
            seed=derive_seed(bootstrap_seed, "mse-vs-global"),
            confidence=confidence,
        ),
        "primary_mse_improvement_vs_job_selected_baseline": _bootstrap_mse_improvement(
            primary_error,
            selected_baseline_error,
            component_ids,
            n_bootstrap=bootstrap_replicates,
            seed=derive_seed(bootstrap_seed, "mse-vs-strongest"),
            confidence=confidence,
        ),
        "primary_delta_r2_vs_job_selected_baseline": _bootstrap_delta_r2(
            primary_error,
            selected_baseline_error,
            global_error,
            component_ids,
            n_bootstrap=bootstrap_replicates,
            seed=derive_seed(bootstrap_seed, "delta-r2-vs-strongest"),
            confidence=confidence,
        ),
        "context_residual_r2": _bootstrap_residual_r2(
            proposed_residual,
            observed_residual,
            component_ids,
            n_bootstrap=bootstrap_replicates,
            seed=derive_seed(bootstrap_seed, "context-residual-r2"),
            confidence=confidence,
        ),
    }

    gates = lock["gates"]
    development_gate = gates["development_signal"]
    method_gate = gates["method"]
    context_gate = gates["context"]
    primary_r2 = float(primary[PRIMARY_METHOD]["r2"])
    selected_baseline_r2 = float(primary[job_selected_baseline]["r2"])
    multi_r2 = robustness["component_multiplicity"]["multi_gene"]["r2"]
    farthest_r2 = robustness["distance_quartiles"]["Q4"]["metrics"]["r2"]
    diagnostic_gates = {
        "scope": (
            "engineering_only_not_evaluable"
            if args.pilot
            else (
                "single_run_diagnostic_not_campaign_decision"
                if float(args.identity_percent) == 30.0
                else "secondary_identity_sensitivity_not_primary_gate"
            )
        ),
        "development_signal": None,
        "method": None,
        "context": None,
        "temporal_26q3": "not_evaluated_by_this_runner",
        "external_3d": "not_evaluated_by_this_runner",
    }
    if not args.pilot and float(args.identity_percent) == 30.0:
        diagnostic_gates["development_signal"] = bool(
            primary_r2 > float(development_gate["full_coverage_gene_mean_r2_min_exclusive"])
            and float(bootstrap["primary_mse_improvement_vs_global"]["ci_low"])
            > float(
                development_gate[
                    "mse_improvement_vs_global_bootstrap_lower_min_exclusive"
                ]
            )
            and multi_r2 is not None
            and float(multi_r2) > float(development_gate["multi_gene_component_r2_min_exclusive"])
            and farthest_r2 is not None
            and float(farthest_r2) > float(development_gate["farthest_distance_quartile_r2_min_exclusive"])
        )
        diagnostic_gates["method"] = bool(
            primary_r2 - selected_baseline_r2
            >= float(method_gate["delta_r2_vs_per_job_selected_baseline_min"])
            and float(
                bootstrap["primary_delta_r2_vs_job_selected_baseline"]["ci_low"]
            )
            > float(method_gate["delta_r2_bootstrap_lower_min_exclusive"])
        )
        diagnostic_gates["context"] = bool(
            float(secondary["r2_vs_zero"])
            > float(context_gate["residual_r2_min_exclusive"])
            and float(bootstrap["context_residual_r2"]["ci_low"])
            > float(context_gate["residual_r2_bootstrap_lower_min_exclusive"])
            and float(secondary["mean_within_gene_pearson"])
            >= float(context_gate["mean_within_gene_pearson_min"])
        )

    run_key = {
        "identity_percent": int(args.identity_percent),
        "outer_fold": int(args.outer_fold),
        "split_seed": int(args.split_seed),
        "training_seed": int(args.training_seed),
        "campaign_roster_sha256": str(args.campaign_roster_sha256),
    }
    baseline_inner_validation = {
        name: {
            "r2": float(
                baseline_validation_candidates[name]["validation_metrics"]["r2"]
            ),
            "mse": float(
                baseline_validation_candidates[name]["validation_metrics"]["mse"]
            ),
        }
        for name in REQUIRED_BASELINES
    }
    outer_test_gene_ids = np.asarray(
        [base.genes[int(index)] for index in split.outer_test], dtype=str
    )
    prediction_array_keys = lock["campaign_aggregation"]["run_prediction_array_keys"]
    method_prediction_keys = prediction_array_keys["method_gene_mean_predictions"]
    selected_name_dtype = f"<U{max(1, len(job_selected_baseline))}"
    prediction_arrays: dict[str, np.ndarray] = {
        prediction_array_keys["gene_ids"]: outer_test_gene_ids,
        prediction_array_keys["component_ids"]: component_ids,
        prediction_array_keys["component_sizes"]: component_size,
        prediction_array_keys["protein_lengths"]: test_protein_length,
        prediction_array_keys["canonical_gene_indices"]: np.asarray(
            split.outer_test, dtype=np.int64
        ),
        prediction_array_keys["source_gene_indices"]: np.asarray(
            base.source_gene_indices[split.outer_test], dtype=np.int64
        ),
        prediction_array_keys["cell_ids"]: np.asarray(base.cell_ids, dtype=str),
        prediction_array_keys["source_cell_indices"]: np.asarray(
            base.source_cell_indices, dtype=np.int64
        ),
        prediction_array_keys["observed_gene_effect"]: observed_gene_effect.astype(
            np.float32
        ),
        prediction_array_keys["observed_gene_mean"]: outer_target.astype(np.float64),
        prediction_array_keys[
            "observed_gene_centered_residual"
        ]: observed_residual.astype(np.float32),
        prediction_array_keys["nearest_training_distance"]: distance.astype(
            np.float64
        ),
        prediction_array_keys["distance_gate_coordinate"]: distance_gate_z.astype(
            np.float64
        ),
        prediction_array_keys["distance_stack_weight_global"]: (
            distance_stack_weight[:, 0].astype(np.float64)
        ),
        prediction_array_keys["distance_stack_weight_knn"]: (
            distance_stack_weight[:, 1].astype(np.float64)
        ),
        prediction_array_keys["distance_stack_weight_ridge"]: (
            distance_stack_weight[:, 2].astype(np.float64)
        ),
        prediction_array_keys["selective_risk_score"]: risk_score.astype(np.float64),
        prediction_array_keys["selected_baseline_name"]: np.asarray(
            [job_selected_baseline], dtype=selected_name_dtype
        ),
        prediction_array_keys[
            "selected_baseline_prediction"
        ]: selected_baseline_prediction.astype(np.float64),
        prediction_array_keys[
            "secondary_centered_residual_prediction"
        ]: proposed_residual.astype(np.float32),
        prediction_array_keys["secondary_total_prediction"]: secondary_total.astype(
            np.float32
        ),
    }
    for method_name, array_name in method_prediction_keys.items():
        prediction_arrays[array_name] = np.asarray(
            outer_predictions[method_name], dtype=np.float64
        )
    for stack_name, method_map in STACK_BASE_PREDICTION_ARRAY_KEYS.items():
        for method_name, array_name in method_map.items():
            prediction_arrays[array_name] = np.asarray(
                stack_base_predictions[stack_name][method_name], dtype=np.float64
            )
    declared_array_names = [
        value
        for key, value in prediction_array_keys.items()
        if key != "method_gene_mean_predictions"
    ] + list(method_prediction_keys.values()) + [
        array_name
        for method_map in STACK_BASE_PREDICTION_ARRAY_KEYS.values()
        for array_name in method_map.values()
    ]
    if len(declared_array_names) != len(set(declared_array_names)):
        raise FamilyInductiveRunError(
            "locked run prediction array names are not unique"
        )
    if set(prediction_arrays) != set(declared_array_names):
        missing = sorted(set(declared_array_names).difference(prediction_arrays))
        extra = sorted(set(prediction_arrays).difference(declared_array_names))
        raise FamilyInductiveRunError(
            f"prediction payload violates exact locked allowlist; missing={missing}, extra={extra}"
        )
    expected_matrix_shape = (len(base.cell_ids), int(split.outer_test.size))
    for array_name in (
        prediction_array_keys["observed_gene_effect"],
        prediction_array_keys["observed_gene_centered_residual"],
        prediction_array_keys["secondary_centered_residual_prediction"],
        prediction_array_keys["secondary_total_prediction"],
    ):
        if prediction_arrays[array_name].shape != expected_matrix_shape:
            raise FamilyInductiveRunError(
                f"{array_name} is not cells-by-outer-test-genes: "
                f"{prediction_arrays[array_name].shape!r} != {expected_matrix_shape!r}"
            )
    predictions_path = run_dir / "predictions.npz"
    predictions_record = _atomic_write_npz(
        predictions_path,
        prediction_arrays,
    )

    job_selected_baseline_evidence = {
        "name": job_selected_baseline,
        "selection_scope": "per_outer_job_inner_validation_only",
        "selection_rule": contract["baseline_selection_rule"],
        "selected_validation": baseline_inner_validation[job_selected_baseline],
        "name_sha256": _hash_strings([job_selected_baseline]),
        "outer_prediction_sha256": predictions_record["arrays"][
            prediction_array_keys["selected_baseline_prediction"]
        ]["sha256"],
    }
    campaign_evidence = {
        "run_key": run_key,
        "campaign_roster_sha256": str(args.campaign_roster_sha256),
        "campaign_aggregation_contract_sha256": hashlib.sha256(
            _json_bytes(lock["campaign_aggregation"])
        ).hexdigest(),
        "baseline_inner_validation": baseline_inner_validation,
        "baseline_inner_validation_sha256": hashlib.sha256(
            _json_bytes(baseline_inner_validation)
        ).hexdigest(),
        "job_selected_baseline": job_selected_baseline_evidence,
        "selection_traces_sha256": hashlib.sha256(
            _json_bytes(selection_traces)
        ).hexdigest(),
        "outer_test_gene_ids_sha256": predictions_record["arrays"][
            prediction_array_keys["gene_ids"]
        ]["sha256"],
        "outer_test_component_ids_sha256": predictions_record["arrays"][
            prediction_array_keys["component_ids"]
        ]["sha256"],
        "method_gene_mean_prediction_sha256": {
            method_name: predictions_record["arrays"][array_name]["sha256"]
            for method_name, array_name in method_prediction_keys.items()
        },
        "stack_base_prediction_sha256": {
            stack_name: {
                method_name: predictions_record["arrays"][array_name]["sha256"]
                for method_name, array_name in method_map.items()
            }
            for stack_name, method_map in STACK_BASE_PREDICTION_ARRAY_KEYS.items()
        },
        "nearest_outer_train_cosine_distance_sha256": predictions_record["arrays"][
            prediction_array_keys["nearest_training_distance"]
        ]["sha256"],
        "distance_gate_z_sha256": predictions_record["arrays"][
            prediction_array_keys["distance_gate_coordinate"]
        ]["sha256"],
        "distance_stack_weight_sha256": {
            STACK_METHOD_ORDER[0]: predictions_record["arrays"][
                prediction_array_keys["distance_stack_weight_global"]
            ]["sha256"],
            STACK_METHOD_ORDER[1]: predictions_record["arrays"][
                prediction_array_keys["distance_stack_weight_knn"]
            ]["sha256"],
            STACK_METHOD_ORDER[2]: predictions_record["arrays"][
                prediction_array_keys["distance_stack_weight_ridge"]
            ]["sha256"],
        },
        "validation_fitted_distance_risk_score_sha256": predictions_record["arrays"][
            prediction_array_keys["selective_risk_score"]
        ]["sha256"],
        "prediction_array_manifest_sha256": hashlib.sha256(
            _json_bytes(predictions_record["arrays"])
        ).hexdigest(),
        "aggregation_status": "single_job_input_only_not_aggregated",
    }

    metrics = {
        "schema": METRICS_SCHEMA,
        "run_id": args.run_id,
        "run_key": run_key,
        "analysis_scope": (
            "engineering_only_not_for_scientific_use"
            if args.pilot
            else "registered_full_data_outer_fold"
        ),
        "analysis_role": (
            "primary_30_percent"
            if float(args.identity_percent) == 30.0
            else "secondary_identity_sensitivity"
        ),
        "primary_method": PRIMARY_METHOD,
        "secondary_context_method": SECONDARY_CONTEXT_METHOD,
        "inner_validation_fold": int(split.inner_validation_fold),
        "primary": primary,
        "primary_reference": {
            "definition": "per_job_outer_training_global_gene_mean",
            "value": outer_global,
        },
        "job_selected_baseline": {
            **job_selected_baseline_evidence,
            "outer_test_metrics": primary[job_selected_baseline],
            "primary_minus_selected_baseline_outer_test_delta_r2": (
                primary_r2 - selected_baseline_r2
            ),
        },
        "secondary_context": secondary,
        "robustness": robustness,
        "selective_prediction": selective,
        "distance_risk_score": risk_score_evaluation,
        "campaign_evidence": campaign_evidence,
        "component_block_bootstrap": bootstrap,
        "diagnostic_gates": diagnostic_gates,
        "reporting_note": (
            "Pilot metrics are engineering diagnostics only. Formal single-run gates do "
            "not establish campaign, external-validation, novelty, or NAR readiness."
        ),
    }
    metrics_path = run_dir / "metrics.json"
    _atomic_write_json(metrics_path, metrics)

    code_records_end = _records(code_paths)
    input_records_end = _records(input_paths)
    if code_records_end != code_records_start:
        raise FamilyInductiveRunError("code or lock files changed during execution")
    if input_records_end != input_records_start:
        raise FamilyInductiveRunError("input files changed during execution")

    manifest = {
        "schema": SCHEMA,
        "status": "complete",
        "run_id": args.run_id,
        "run_key": run_key,
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "wall_seconds": float(time.monotonic() - wall_start),
        "scientific_use": not bool(args.pilot),
        "pilot": bool(args.pilot),
        "remote_execution_enforcement": {
            "required_scope": "remote-server-only",
            "registered_identity": lock["designated_server"],
            "identity_check": dict(server_identity),
        },
        "robustness_strata_contract": lock["robustness_strata"],
        "selective_risk_score_contract": lock["selective_risk_score"],
        "campaign_aggregation_contract": lock["campaign_aggregation"],
        "campaign_evidence": campaign_evidence,
        "arguments": {
            "identity_percent": int(args.identity_percent),
            "outer_fold": int(args.outer_fold),
            "split_seed": int(args.split_seed),
            "training_seed": int(args.training_seed),
            "campaign_roster_sha256": str(args.campaign_roster_sha256),
            "campaign_roster_digest_source": str(
                args.campaign_roster_digest_source
            ),
            "device": device,
            "run_id": args.run_id,
            "pilot": bool(args.pilot),
        },
        "lock_provenance": lock_provenance,
        "training_contract": contract,
        "external_3d_contract": lock["external_3d_analysis"],
        "effective_training": {
            "inner_max_epochs": inner_epochs,
            "patience": patience,
            "batch_size": batch_size,
            "gene_subsample": gene_subsample,
            "bootstrap_replicates": bootstrap_replicates,
            "pilot_subset_seed": PILOT_SUBSET_SEED if args.pilot else None,
        },
        "environment": environment,
        "code_files": code_records_start,
        "input_files": input_records_start,
        "data": {
            "n_genes": len(base.genes),
            "n_cells": len(base.cell_ids),
            "gene_ids_sha256": _hash_strings(base.genes),
            "cell_ids_sha256": _hash_strings(base.cell_ids),
            "source_gene_indices_sha256": _hash_array(base.source_gene_indices.astype("<i8")),
            "source_cell_indices_sha256": _hash_array(base.source_cell_indices.astype("<i8")),
        },
        "partition_audit": partition_audit,
        "family_manifest": _file_record(family_manifest),
        "preprocessing": preprocessing,
        "selection": {
            "primary_method": {
                "name": PRIMARY_METHOD,
                "selection_scope": "inner_component_heldout_validation_only",
                "selection_rule": lock["primary_method_contract"]["selection_rule"],
                "selected": selection_traces[PRIMARY_METHOD]["selected"],
            },
            "job_selected_baseline": job_selected_baseline_evidence,
            "baseline_inner_validation": baseline_inner_validation,
            "neural_selection_rule": (
                "max_inner_component_heldout_gene_mean_r2_then_min_mse_then_"
                "canonical_config"
            ),
            "traces": selection_traces,
            "traces_sha256": campaign_evidence["selection_traces_sha256"],
            "distance_stack_audit": {
                "implementation_contract": lock["primary_method_contract"][
                    "implementation_contract"
                ],
                "inner_crossfit": inner_stack_audit,
                "selected_primary_fit": selected_sequence[PRIMARY_METHOD]["fit"],
                "selected_static_ablation_fit": selected_sequence[
                    "STATIC-CONVEX-STACK"
                ]["fit"],
            },
            "selected_decomposition": {
                key: value for key, value in selected_decomp.items() if key != "training"
            },
            "selected_matched_mean_only_ablation": {
                key: value
                for key, value in selected_mean_only.items()
                if key != "training"
            },
        },
        "final_refits": {
            "distance_stack_audit": outer_stack_audit,
            "IDFREE-PLM-MEAN-MLP": final_mean_training,
            "DECOMP-MEAN-ONLY": final_mean_only_training,
            "FAMILY-INDUCTIVE-DECOMP": final_decomp_training,
            "matched_optimizer_audit": final_matched_optimizer_audit,
        },
        "distance_risk_scorer": risk_scorer.to_dict(),
        "leakage_controls": {
            "outer_test_outcomes_first_indexed_after_predictions_fixed": True,
            "hyperparameters_selected_on_inner_component_holdout_only": True,
            "distance_risk_score_fit_on_inner_validation_errors_only": True,
            "distance_risk_score_coverage_guarantee": "none",
            "test_outcomes_used_for_risk_fit_or_retention_selection": False,
            "learned_gene_id_or_gene_bias": False,
            "raw_residual_centered_in_mini_batches": False,
            "predicted_residual_centering_scope": "all_eligible_cells_per_gene_after_inference",
        },
        "outputs": {
            "predictions": predictions_record,
            "metrics": _file_record(metrics_path, relative_to=run_dir),
            "prediction_array_keys": prediction_array_keys,
            "stack_base_prediction_array_keys": STACK_BASE_PREDICTION_ARRAY_KEYS,
            "prediction_array_names_exact": declared_array_names,
            "prediction_array_manifest_sha256": campaign_evidence[
                "prediction_array_manifest_sha256"
            ],
        },
    }
    manifest_path = run_dir / "run-manifest.json"
    _atomic_write_json(manifest_path, manifest)
    completion = {
        "schema": COMPLETION_SCHEMA,
        "status": "complete",
        "run_id": args.run_id,
        "run_key": run_key,
        "campaign_roster_sha256": str(args.campaign_roster_sha256),
        "campaign_roster_digest_source": str(args.campaign_roster_digest_source),
        "completed_at_utc": _utc_now(),
        "scientific_use": not bool(args.pilot),
        "artifacts": {
            "predictions": predictions_record,
            "metrics": _file_record(metrics_path, relative_to=run_dir),
            "run_manifest": _file_record(manifest_path, relative_to=run_dir),
        },
        "integrity": {
            "code_and_lock_unchanged_during_run": True,
            "inputs_unchanged_during_run": True,
            "outer_test_evaluated_once": True,
            "prediction_array_exact_allowlist_validated": True,
            "prediction_array_manifest_sha256": campaign_evidence[
                "prediction_array_manifest_sha256"
            ],
            "campaign_evidence_sha256": hashlib.sha256(
                _json_bytes(campaign_evidence)
            ).hexdigest(),
            "job_selected_baseline_outer_prediction_sha256": (
                job_selected_baseline_evidence["outer_prediction_sha256"]
            ),
            "stack_base_prediction_sha256": campaign_evidence[
                "stack_base_prediction_sha256"
            ],
            "distance_stack_solver_fail_closed": True,
        },
    }
    _atomic_write_json(run_dir / "completion.json", completion)
    return run_dir


def _valid_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
        raise argparse.ArgumentTypeError(
            "run id must use 1-128 ASCII letters, digits, dot, underscore, or hyphen"
        )
    return value


def _valid_sha256(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise argparse.ArgumentTypeError(
            "campaign roster digest must be exactly 64 lowercase hexadecimal characters"
        )
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--identity-percent", type=int, choices=(20, 30, 40, 50), default=30
    )
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--campaign-roster-sha256", type=_valid_sha256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-id", type=_valid_run_id)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    args = parse_args(argv)
    lock, lock_provenance = validate_analysis_lock()
    server_identity = _assert_designated_server(lock)
    if args.campaign_roster_sha256 is None:
        if not args.pilot:
            raise FamilyInductiveRunError(
                "formal runs require --campaign-roster-sha256"
            )
        args.campaign_roster_sha256 = PILOT_CAMPAIGN_ROSTER_SHA256
        args.campaign_roster_digest_source = "pilot_engineering_marker"
    else:
        args.campaign_roster_digest_source = "command_line"
    _validate_registered_run(
        lock,
        identity_percent=int(args.identity_percent),
        outer_fold=int(args.outer_fold),
        split_seed=int(args.split_seed),
        training_seed=int(args.training_seed),
    )
    device = _resolve_device(str(args.device))
    if args.run_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        mode = "pilot" if args.pilot else "formal"
        args.run_id = _valid_run_id(
            f"{mode}-id{args.identity_percent}-of{args.outer_fold}-"
            f"ss{args.split_seed}-ts{args.training_seed}-{timestamp}"
        )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / args.run_id
    try:
        run_dir.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise AtomicWriteError(f"refusing to overwrite existing run directory: {run_dir}") from exc

    try:
        completed = _run_experiment(
            args,
            lock=lock,
            lock_provenance=lock_provenance,
            run_dir=run_dir,
            device=device,
            server_identity=server_identity,
        )
    except BaseException as exc:
        failure = {
            "schema": FAILURE_SCHEMA,
            "status": "failed",
            "run_id": args.run_id,
            "run_key": {
                "identity_percent": int(args.identity_percent),
                "outer_fold": int(args.outer_fold),
                "split_seed": int(args.split_seed),
                "training_seed": int(args.training_seed),
                "campaign_roster_sha256": str(args.campaign_roster_sha256),
            },
            "failed_at_utc": _utc_now(),
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "scientific_use": False,
        }
        try:
            _atomic_write_json(run_dir / "failure.json", failure)
        except BaseException:
            pass
        raise
    print(f"[family-inductive] complete: {completed}", flush=True)
    return completed


if __name__ == "__main__":
    main()
