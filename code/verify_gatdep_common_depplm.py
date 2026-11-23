"""Independently verify a common-cohort DepPLM comparison run."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

import gatdep_official_baseline as gatdep


VERIFICATION_SCHEMA = "DepPLM-GATDep-common-cohort-verification-v3"
RUN_SCHEMA = "DepPLM-GATDep-common-cohort-run-v3"
COHORT_SCHEMA = "DepPLM-GATDep-common-cohort-v1"
SUMMARY_SCHEMA = "DepPLM-GATDep-common-cohort-summary-v3"
COMPLETION_SCHEMA = "DepPLM-GATDep-common-cohort-completion-v3"
CHECKPOINT_SCHEMA = "DepPLM-outer-test-replay-checkpoint-v3"
REPLAY_TOLERANCE = {"rtol": 1e-5, "atol": 1e-6}
CHECKPOINT_POLICY = (
    "per_fold_final_model_and_preprocessing_only_with_"
    "source_reconstructed_outer_test_replay"
)
REPLAY_INPUT_POLICY = (
    "not_embedded_reconstructed_from_hash_bound_project_inputs"
)
GENERATED_ARTIFACT_REDISTRIBUTION_STATUS = (
    "not_cleared_for_public_redistribution"
)
PREPROCESSING_CONTENT_POLICY = (
    "training_fitted_aggregate_statistics_and_feature_definitions_no_raw_cell_rows"
)
SUPPORTED_CONDITIONS = ("DepPLM", "C-PLM-MLP")
FIXED_ARGUMENTS: dict[str, Any] = {
    "protocol": "p1",
    "folds": 5,
    "inner_val_fraction": 0.2,
    "inner_max_epochs": 20,
    "patience": 5,
    "batch_size": 64,
    "gene_subsample": 6_000,
    "top_mut": 5_000,
    "n_expr_pc": 128,
    "pilot": False,
}
EXPECTED_CONDITION_CONFIGURATIONS: dict[str, dict[str, Any]] = {
    "DepPLM": {"model": "depplm"},
    "C-PLM-MLP": {"model": "plmmlp"},
}
EXPECTED_TUNING_GRID = [
    {"hidden": 128, "lr": 1e-3, "wd": 1e-5, "lamb_sel": 0.1},
    {"hidden": 128, "lr": 3e-4, "wd": 1e-5, "lamb_sel": 0.1},
    {"hidden": 256, "lr": 1e-3, "wd": 1e-5, "lamb_sel": 0.1},
    {"hidden": 256, "lr": 3e-4, "wd": 1e-5, "lamb_sel": 0.1},
]
SOURCE_PATHS = (
    "code/run_gatdep_common_depplm.py",
    "code/verify_gatdep_common_depplm.py",
    "code/run_corrected_experiments.py",
    "code/corrected_protocol.py",
    "code/models.py",
    "code/03_embed_proteins.py",
    "code/embedding_provenance.py",
    "code/gatdep_official_baseline.py",
    "code/verify_gatdep_official_baseline.py",
)
PROJECT_INPUT_PATHS = (
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
)
OFFICIAL_INPUT_PATHS = (
    "datasets/genes2ID_2k.json",
    "datasets/samples2ID.json",
)
METRIC_NAMES = (
    "mse",
    "mae",
    "gene_mean_baseline_mse",
    "gene_mean_baseline_mae",
    "residual_r2_vs_train_gene_mean",
    "residual_pearson_pooled",
    "residual_cellwise_pearson_mean",
    "gene_wise_pearson_mean",
    "gene_wise_mse_mean",
    "gene_wise_residual_r2_mean",
    "fraction_genes_positive_residual_r2",
    "cellwise_pearson_mean",
    "cellwise_spearman_mean",
)
REQUIRED_PREDICTION_ARRAYS = {
    "prediction",
    "train_gene_mean",
    "outer_test_indices",
    "source_cell_indices",
    "source_gene_indices",
    "cell_ids",
    "gene_ids",
}


def scoped_records(root: Path, paths: Sequence[str]) -> list[dict[str, Any]]:
    return [gatdep.file_record(root / relative, relative_to=root) for relative in paths]


def _tensor_record(value: Any) -> dict[str, Any]:
    import torch

    tensor = value.detach().cpu()
    if tensor.layout != torch.strided:
        if tensor.layout != torch.sparse_coo:
            raise TypeError(f"unsupported checkpoint tensor layout: {tensor.layout}")
        tensor = tensor.coalesce()
        return {
            "layout": str(tensor.layout),
            "shape": list(tensor.shape),
            "indices": _tensor_record(tensor.indices()),
            "values": _tensor_record(tensor.values()),
        }
    array = np.ascontiguousarray(tensor.numpy())
    return {
        "layout": "torch.strided",
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": hash_array(array),
    }


def _checkpoint_inventory(value: Any) -> Any:
    import torch

    if type(value) is torch.Tensor:
        return _tensor_record(value)
    if type(value) is dict:
        if not all(isinstance(key, str) for key in value):
            raise TypeError("checkpoint mapping keys must all be strings")
        return {
            key: _checkpoint_inventory(item)
            for key, item in sorted(value.items())
        }
    if type(value) in (tuple, list):
        return [_checkpoint_inventory(item) for item in value]
    if type(value) is float and not np.isfinite(value):
        raise ValueError("checkpoint scalar values must be finite")
    if type(value) in (str, int, float, bool) or value is None:
        return value
    raise TypeError(f"unsupported checkpoint inventory value: {type(value)!r}")


def checkpoint_component_hash(value: Any) -> str:
    return gatdep.canonical_hash(_checkpoint_inventory(value))


def _as_unique_indices(
    indices: Sequence[int] | np.ndarray, n: int | None = None
) -> np.ndarray:
    """Validate indices without calling the implementation under test."""
    out = np.asarray(indices, dtype=np.int64)
    if out.ndim != 1 or out.size == 0:
        raise ValueError("indices must be a non-empty one-dimensional array")
    if np.unique(out).size != out.size:
        raise ValueError("indices must not contain duplicates")
    if n is not None and (int(out.min()) < 0 or int(out.max()) >= n):
        raise IndexError("indices are outside the available rows")
    return out


def derive_seed(seed: int, *parts: object) -> int:
    """Independently reproduce the protocol's stable 31-bit seed."""
    payload = "|".join([str(int(seed)), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def make_inner_split(
    outer_train_idx: Sequence[int] | np.ndarray,
    lineages: Sequence[str] | np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Independently reproduce the deterministic lineage-stratified split."""
    values = np.asarray(lineages, dtype=str)
    outer = _as_unique_indices(outer_train_idx, values.size)
    if not 0.0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be between zero and 0.5")
    rng = np.random.default_rng(int(seed))
    val_parts: list[np.ndarray] = []
    train_parts: list[np.ndarray] = []
    for category in sorted(set(values[outer].tolist())):
        members = outer[values[outer] == category].copy()
        rng.shuffle(members)
        if members.size < 2:
            train_parts.append(members)
            continue
        n_val = min(members.size - 1, max(1, int(round(members.size * val_fraction))))
        val_parts.append(members[:n_val])
        train_parts.append(members[n_val:])
    inner_train = np.sort(np.concatenate(train_parts))
    if val_parts:
        inner_val = np.sort(np.concatenate(val_parts))
    elif inner_train.size >= 2:
        inner_val = inner_train[-1:].copy()
        inner_train = inner_train[:-1]
    else:
        raise ValueError("outer training split is too small for inner validation")
    return inner_train.astype(np.int64), inner_val.astype(np.int64)


def make_kfold_outer_splits(
    n_cells: int, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Independently reproduce the fixed shuffled K-fold allocation."""
    if n_splits < 2 or n_splits > n_cells:
        raise ValueError("n_splits must be between 2 and n_cells")
    rng = np.random.default_rng(int(seed))
    folds = np.array_split(rng.permutation(n_cells), n_splits)
    return [
        (
            np.sort(
                np.concatenate([folds[j] for j in range(n_splits) if j != i])
            ).astype(np.int64),
            np.sort(folds[i]).astype(np.int64),
        )
        for i in range(n_splits)
    ]


def _index_record(indices: np.ndarray, cell_ids: Sequence[str]) -> dict[str, Any]:
    little_endian = indices.astype("<i8", copy=False)
    return {
        "indices": [int(value) for value in indices],
        "cell_ids": [str(cell_ids[int(value)]) for value in indices],
        "sha256": hashlib.sha256(little_endian.tobytes()).hexdigest(),
    }


def build_split_manifest(
    *,
    outer_train_idx: Sequence[int] | np.ndarray,
    inner_train_idx: Sequence[int] | np.ndarray,
    inner_val_idx: Sequence[int] | np.ndarray,
    outer_test_idx: Sequence[int] | np.ndarray,
    cell_ids: Sequence[str],
    seed: int,
) -> dict[str, Any]:
    """Validate split isolation and independently reconstruct its manifest."""
    n_cells = len(cell_ids)
    outer_train = _as_unique_indices(outer_train_idx, n_cells)
    inner_train = _as_unique_indices(inner_train_idx, n_cells)
    inner_val = _as_unique_indices(inner_val_idx, n_cells)
    outer_test = _as_unique_indices(outer_test_idx, n_cells)
    sets = [set(inner_train), set(inner_val), set(outer_test)]
    if any(sets[i].intersection(sets[j]) for i in range(3) for j in range(i + 1, 3)):
        raise ValueError(
            "inner train, inner validation and outer test must be disjoint"
        )
    outer_set = set(outer_train)
    if set(inner_train).union(inner_val) != outer_set:
        raise ValueError("inner train and validation must partition outer train")
    if outer_set.intersection(outer_test):
        raise ValueError("outer train and outer test must be disjoint")
    return {
        "seed": int(seed),
        "outer_train": _index_record(outer_train, cell_ids),
        "inner_train": _index_record(inner_train, cell_ids),
        "inner_validation": _index_record(inner_val, cell_ids),
        "outer_test": _index_record(outer_test, cell_ids),
        "fit_scope": {
            "expression_pca": "training_cells_only",
            "mutation_vocabulary": "training_cells_only",
            "mutation_token_order": "training_cells_only",
            "comutation_pmi": "training_cells_only",
            "selectivity_label": "training_cells_only",
            "lineage_categories": "training_cells_only_unseen_is_zero",
        },
        "k_tokens": 64,
    }


def fit_gene_mean_baseline(
    gene_effect: np.ndarray,
    train_idx: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Independently fit finite-only per-gene means on outer-train cells."""
    values = np.asarray(gene_effect, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("gene_effect must have shape (genes, cells)")
    train_idx = _as_unique_indices(train_idx, values.shape[1])
    train = values[:, train_idx]
    finite = np.isfinite(train)
    counts = finite.sum(axis=1)
    totals = np.where(finite, train, 0.0).sum(axis=1, dtype=np.float64)
    means = np.full(values.shape[0], np.nan, dtype=np.float64)
    np.divide(totals, counts, out=means, where=counts > 0)
    return means.astype(np.float32)


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


def _finite_summary(values: np.ndarray, prefix: str) -> dict[str, float | int]:
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


def dependency_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    train_gene_mean: np.ndarray,
) -> dict[str, float | int]:
    """Independently recompute every corrected dependency metric."""
    predicted = np.asarray(prediction, dtype=np.float32)
    observed = np.asarray(target, dtype=np.float32)
    reference = np.asarray(train_gene_mean, dtype=np.float32)
    if predicted.ndim != 2 or observed.shape != predicted.shape:
        raise ValueError(
            "prediction and target must have the same two-dimensional shape"
        )
    if reference.ndim != 1 or reference.size != predicted.shape[1]:
        raise ValueError("train_gene_mean must contain one value per gene")

    reference_2d = reference[None, :]
    valid = np.isfinite(observed) & np.isfinite(reference_2d)
    n_pairs = int(valid.sum())
    if n_pairs == 0:
        raise ValueError("no finite target/reference pairs")
    invalid_predictions = valid & ~np.isfinite(predicted)
    if invalid_predictions.any():
        raise ValueError(
            "non-finite predictions occurred for "
            f"{int(invalid_predictions.sum())} evaluable target/reference pairs"
        )

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
    residual_cellwise_corr = _correlation_by_axis(
        prediction_residual, target_residual, valid, axis=1, min_count=3
    )
    cellwise_corr = _correlation_by_axis(
        predicted, observed, valid, axis=1, min_count=3
    )
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

    metrics: dict[str, float | int] = {
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
    metrics.update(_finite_summary(residual_cellwise_corr, "residual_cellwise_pearson"))
    metrics.update(_finite_summary(cellwise_corr, "cellwise_pearson"))
    metrics.update(_finite_summary(gene_corr, "gene_wise_pearson"))
    metrics.update(_finite_summary(gene_mse, "gene_wise_mse"))
    metrics.update(_finite_summary(gene_mae, "gene_wise_mae"))
    metrics.update(_finite_summary(gene_r2, "gene_wise_residual_r2"))
    metrics["n_finite_cell_correlations"] = int(np.isfinite(cellwise_corr).sum())
    metrics["n_finite_gene_correlations"] = int(np.isfinite(gene_corr).sum())
    finite_gene_r2 = gene_r2[np.isfinite(gene_r2)]
    metrics["fraction_genes_positive_residual_r2"] = (
        float(np.mean(finite_gene_r2 > 0.0)) if finite_gene_r2.size else float("nan")
    )
    return metrics


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def hash_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def hash_strings(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def mappings_close(observed: Any, expected: Any, *, path: str = "value") -> list[str]:
    """Compare JSON-like metric structures with tight numeric tolerance."""
    observed = _json_value(observed)
    expected = _json_value(expected)
    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            return [f"{path} is not an object"]
        errors: list[str] = []
        if set(observed) != set(expected):
            errors.append(f"{path} keys differ")
        for key in sorted(set(observed).intersection(expected)):
            errors.extend(
                mappings_close(observed[key], expected[key], path=f"{path}.{key}")
            )
        return errors
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            return [f"{path} list differs"]
        errors = []
        for index, (left, right) in enumerate(zip(observed, expected)):
            errors.extend(mappings_close(left, right, path=f"{path}[{index}]"))
        return errors
    if isinstance(expected, float) and isinstance(observed, (int, float)):
        if not np.isclose(float(observed), expected, rtol=1e-10, atol=1e-12):
            return [f"{path} differs: {observed!r} != {expected!r}"]
        return []
    return [] if observed == expected else [f"{path} differs"]


def expected_split(
    *,
    fold_number: int,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    cohort: gatdep.CommonCohort,
    seed: int,
) -> dict[str, Any]:
    fold_name = f"fold-{fold_number:02d}"
    fold_seed = derive_seed(seed, fold_name)
    inner_train, inner_val = make_inner_split(
        outer_train,
        cohort.lineages,
        val_fraction=FIXED_ARGUMENTS["inner_val_fraction"],
        seed=derive_seed(fold_seed, "inner-split"),
    )
    split = build_split_manifest(
        outer_train_idx=outer_train,
        inner_train_idx=inner_train,
        inner_val_idx=inner_val,
        outer_test_idx=outer_test,
        cell_ids=cohort.cell_ids,
        seed=fold_seed,
    )
    split.update(
        {
            "fold": fold_name,
            "protocol": "p1",
            "source_cell_indices": {
                "outer_train": [
                    int(cohort.project_cell_indices[index]) for index in outer_train
                ],
                "inner_train": [
                    int(cohort.project_cell_indices[index]) for index in inner_train
                ],
                "inner_validation": [
                    int(cohort.project_cell_indices[index]) for index in inner_val
                ],
                "outer_test": [
                    int(cohort.project_cell_indices[index]) for index in outer_test
                ],
            },
            "lineage_counts": {
                "outer_train": dict(
                    sorted(
                        Counter(
                            str(value) for value in cohort.lineages[outer_train]
                        ).items()
                    )
                ),
                "outer_test": dict(
                    sorted(
                        Counter(
                            str(value) for value in cohort.lineages[outer_test]
                        ).items()
                    )
                ),
            },
        }
    )
    return split


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    if int(finite.sum()) < 3:
        return float("nan")
    x = np.asarray(left[finite], dtype=np.float64)
    y = np.asarray(right[finite], dtype=np.float64)
    x -= x.mean()
    y -= y.mean()
    denominator = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return float(np.dot(x, y) / denominator) if denominator > 0 else float("nan")


def expected_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    train_gene_mean: np.ndarray,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        **dependency_metrics(prediction, target, train_gene_mean),
        "n_cells": int(prediction.shape[0]),
        "reference_scope": "outer_train_gene_mean",
        "gene_mean_reference_sha256": hash_array(train_gene_mean),
        "raw_cellwise_correlation_role": "secondary_descriptive_only",
    }
    spearman = []
    for predicted, observed in zip(prediction, target):
        finite = np.isfinite(predicted) & np.isfinite(observed)
        if int(finite.sum()) < 10:
            spearman.append(np.nan)
        else:
            spearman.append(
                _safe_corr(rankdata(predicted[finite]), rankdata(observed[finite]))
            )
    values = np.asarray(spearman, dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    metrics.update(
        {
            "cellwise_spearman_mean": float(np.mean(finite_values))
            if finite_values.size
            else float("nan"),
            "cellwise_spearman_sd": float(np.std(finite_values))
            if finite_values.size
            else float("nan"),
            "n_finite_cellwise_spearman": int(finite_values.size),
        }
    )
    return metrics


def verify_prediction_bundle(
    path: Path,
    record: Mapping[str, Any],
    *,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    cohort: gatdep.CommonCohort,
    target_source: np.ndarray,
) -> tuple[list[str], dict[str, Any] | None]:
    errors: list[str] = []
    expected_file = gatdep.file_record(path, relative_to=path.parent)
    if record.get("schema") != "DepPLM-outer-test-predictions-v1":
        errors.append("prediction record schema differs")
    for field in ("path", "bytes", "sha256"):
        if record.get(field) != expected_file[field]:
            errors.append(f"prediction record {field} differs")
    try:
        with np.load(path, allow_pickle=False) as bundle:
            if set(bundle.files) != REQUIRED_PREDICTION_ARRAYS:
                errors.append("prediction array inventory differs")
                return errors, None
            arrays = {name: np.asarray(bundle[name]) for name in bundle.files}
    except (OSError, ValueError) as exc:
        errors.append(f"prediction bundle cannot be read safely: {exc}")
        return errors, None

    expected_shapes = {
        "prediction": [outer_test.size, 2_341],
        "train_gene_mean": [2_341],
        "outer_test_indices": [outer_test.size],
        "source_cell_indices": [outer_test.size],
        "source_gene_indices": [2_341],
        "cell_ids": [outer_test.size],
        "gene_ids": [2_341],
    }
    observed_record = {
        name: {"dtype": str(array.dtype), "shape": list(array.shape)}
        for name, array in arrays.items()
    }
    if record.get("arrays") != observed_record:
        errors.append("prediction array metadata differs from the NPZ")
    for name, shape in expected_shapes.items():
        if list(arrays[name].shape) != shape:
            errors.append(f"prediction array shape differs: {name}")
    if errors:
        return errors, None
    if arrays["prediction"].dtype != np.float32:
        errors.append("prediction dtype is not float32")
    if arrays["train_gene_mean"].dtype != np.float32:
        errors.append("train gene mean dtype is not float32")
    if arrays["outer_test_indices"].dtype != np.int64:
        errors.append("outer-test index dtype is not int64")
    if arrays["source_cell_indices"].dtype != np.int64:
        errors.append("source cell index dtype is not int64")
    if arrays["source_gene_indices"].dtype != np.int64:
        errors.append("source gene index dtype is not int64")
    if not np.array_equal(arrays["outer_test_indices"], outer_test):
        errors.append("outer-test indices differ from the fixed split")
    if not np.array_equal(
        arrays["source_cell_indices"], cohort.project_cell_indices[outer_test]
    ):
        errors.append("prediction source cell indices differ")
    if not np.array_equal(arrays["source_gene_indices"], cohort.project_gene_indices):
        errors.append("prediction source gene indices differ")
    if arrays["cell_ids"].tolist() != [cohort.cell_ids[index] for index in outer_test]:
        errors.append("prediction cell IDs differ")
    if arrays["gene_ids"].tolist() != list(cohort.gene_ids):
        errors.append("prediction gene IDs differ")
    if not np.isfinite(arrays["prediction"]).all():
        errors.append("prediction contains non-finite values")

    target = np.asarray(
        target_source[
            np.ix_(
                cohort.project_gene_indices,
                cohort.project_cell_indices[outer_test],
            )
        ].T,
        dtype=np.float32,
    )
    train_mean = fit_gene_mean_baseline(
        np.asarray(
            target_source[
                np.ix_(cohort.project_gene_indices, cohort.project_cell_indices)
            ],
            dtype=np.float32,
        ),
        outer_train,
    )
    if not np.array_equal(arrays["train_gene_mean"], train_mean, equal_nan=True):
        errors.append("train gene mean does not replay from current gene_effect.npy")
    if errors:
        return errors, None
    return [], expected_metrics(arrays["prediction"], target, train_mean)


def selectivity_labels_from_train(
    gene_effect: np.ndarray, train_idx: Sequence[int] | np.ndarray
) -> np.ndarray:
    """Independently reproduce the training-only selectivity labels."""
    values = np.asarray(gene_effect, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("gene_effect must have shape (genes, cells)")
    train = values[:, _as_unique_indices(train_idx, values.shape[1])]
    median = np.nanmedian(train, axis=1, keepdims=True)
    mad = 1.4826 * np.nanmedian(np.abs(train - median), axis=1)
    finite = np.isfinite(mad)
    labels = np.zeros(values.shape[0], dtype=np.float32)
    if finite.any():
        threshold = np.nanquantile(mad[finite], 0.75)
        labels[finite] = (mad[finite] >= threshold).astype(np.float32)
    return labels


@dataclass(frozen=True)
class CurrentCommonFeatures:
    """Current hash-bound source features used for independent replay."""

    gene_ids: tuple[str, ...]
    cell_ids: tuple[str, ...]
    expression_raw: np.ndarray
    expression_gene_names: tuple[str, ...]
    mutations_by_cell: tuple[frozenset[str], ...]
    lineages: np.ndarray
    edge_index: np.ndarray
    embeddings: np.ndarray


def _read_nonempty_lines(path: Path) -> tuple[str, ...]:
    return tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _independent_subgraph(
    edge_index: np.ndarray, selected_genes: np.ndarray, n_genes: int
) -> np.ndarray:
    edges = np.asarray(edge_index, dtype=np.int64)
    selected = _as_unique_indices(selected_genes, n_genes)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("stored graph edge_index must have shape (E, 2)")
    if edges.size and (int(edges.min()) < 0 or int(edges.max()) >= n_genes):
        raise IndexError("stored graph edge_index is outside the gene manifest")
    remap = np.full(n_genes, -1, dtype=np.int64)
    remap[selected] = np.arange(selected.size, dtype=np.int64)
    keep = (remap[edges[:, 0]] >= 0) & (remap[edges[:, 1]] >= 0)
    mapped = np.stack(
        [remap[edges[keep, 0]], remap[edges[keep, 1]]], axis=1
    )
    mapped.sort(axis=1)
    mapped = mapped[mapped[:, 0] != mapped[:, 1]]
    return np.unique(mapped, axis=0).T.copy()


def load_current_common_features(
    project_root: Path, cohort: gatdep.CommonCohort
) -> CurrentCommonFeatures:
    """Load replay inputs only from fixed, hash-bound project source paths."""
    import pandas as pd

    processed = project_root / "data/processed"
    depmap = project_root / "data/depmap"
    all_genes = _read_nonempty_lines(processed / "genes.txt")
    all_cells = _read_nonempty_lines(processed / "cell_lines.txt")
    all_lineages = np.asarray(
        _read_nonempty_lines(processed / "lineages.txt"), dtype=str
    )
    if len(all_lineages) != len(all_cells):
        raise ValueError("processed lineage and cell manifests do not align")
    cells = _as_unique_indices(cohort.project_cell_indices, len(all_cells))
    genes = _as_unique_indices(cohort.project_gene_indices, len(all_genes))
    if tuple(all_cells[int(index)] for index in cells) != tuple(cohort.cell_ids):
        raise ValueError("common cell IDs do not align with the processed manifest")
    if tuple(all_genes[int(index)] for index in genes) != tuple(cohort.gene_ids):
        raise ValueError("common gene IDs do not align with the processed manifest")
    lineages = np.asarray(all_lineages[cells], dtype=str)
    if not np.array_equal(lineages, np.asarray(cohort.lineages, dtype=str)):
        raise ValueError("common lineages do not align with the processed manifest")

    expression_path = depmap / "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
    header = pd.read_csv(expression_path, nrows=0).columns.tolist()
    if not header:
        raise ValueError("expression source has no columns")
    id_column = header[0]
    allowed = set(all_genes)
    selected_columns: list[str] = []
    selected_names: list[str] = []
    seen: set[str] = set()
    for column in header[1:]:
        symbol = str(column).split(" (")[0]
        if symbol in allowed and symbol not in seen:
            selected_columns.append(str(column))
            selected_names.append(symbol)
            seen.add(symbol)
    if not selected_columns:
        raise ValueError("no expression columns map to the processed gene set")
    frame = pd.read_csv(
        expression_path,
        usecols=[id_column, *selected_columns],
        index_col=0,
        dtype={column: np.float32 for column in selected_columns},
    )
    if not frame.index.is_unique:
        raise ValueError("expression source contains duplicate cell IDs")
    missing = [cell_id for cell_id in cohort.cell_ids if cell_id not in frame.index]
    if missing:
        raise ValueError(f"expression source is missing {len(missing)} common cells")
    expression = frame.reindex(cohort.cell_ids).to_numpy(
        dtype=np.float32, copy=True
    )

    wanted = set(cohort.cell_ids)
    by_cell: defaultdict[str, set[str]] = defaultdict(set)
    mutation_path = depmap / "OmicsSomaticMutations.csv"
    for chunk in pd.read_csv(
        mutation_path,
        usecols=["HugoSymbol", "ModelID"],
        dtype=str,
        chunksize=250_000,
    ):
        selected_chunk = chunk[chunk["ModelID"].isin(wanted)].dropna()
        for model_id, symbol in zip(
            selected_chunk["ModelID"], selected_chunk["HugoSymbol"]
        ):
            if symbol:
                by_cell[str(model_id)].add(str(symbol))
    mutations = tuple(
        frozenset(by_cell.get(cell_id, set())) for cell_id in cohort.cell_ids
    )

    with np.load(processed / "ppi_graph.npz", allow_pickle=False) as graph:
        if set(graph.files) != {"edge_index", "edge_weight", "nodes"}:
            raise ValueError("processed graph field inventory differs")
        stored_edges = np.asarray(graph["edge_index"], dtype=np.int64)
        stored_weights = np.asarray(graph["edge_weight"], dtype=np.float32)
        stored_nodes = tuple(np.asarray(graph["nodes"], dtype=str).tolist())
        if stored_weights.shape != (stored_edges.shape[0],):
            raise ValueError("processed graph weights do not align with its edges")
        if not np.isfinite(stored_weights).all():
            raise ValueError("processed graph weights contain non-finite values")
        if stored_nodes != all_genes:
            raise ValueError("processed graph nodes do not align with genes.txt")
        edge_index = _independent_subgraph(
            stored_edges,
            genes,
            len(all_genes),
        )

    embedding_names = _read_nonempty_lines(
        processed / "esm2_t6_8M_residue_chunked.names.txt"
    )
    if embedding_names != all_genes:
        raise ValueError("embedding names do not align with the gene manifest")
    embedding_source = np.load(
        processed / "esm2_t6_8M_residue_chunked.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    if embedding_source.ndim != 2 or embedding_source.shape[0] != len(all_genes):
        raise ValueError("embedding array does not align with the gene manifest")
    embeddings = np.asarray(embedding_source[genes], dtype=np.float32)
    return CurrentCommonFeatures(
        gene_ids=tuple(cohort.gene_ids),
        cell_ids=tuple(cohort.cell_ids),
        expression_raw=expression,
        expression_gene_names=tuple(selected_names),
        mutations_by_cell=mutations,
        lineages=lineages,
        edge_index=edge_index,
        embeddings=embeddings,
    )


def _expected_mutation_features(
    mutations_by_cell: Sequence[Sequence[str] | frozenset[str]],
    gene_ids: Sequence[str],
    train_idx: np.ndarray,
) -> dict[str, Any]:
    cells = [{str(gene) for gene in genes} for genes in mutations_by_cell]
    prevalence: Counter[str] = Counter()
    for index in _as_unique_indices(train_idx, len(cells)):
        prevalence.update(cells[int(index)])
    vocabulary = tuple(
        sorted(prevalence, key=lambda gene: (-prevalence[gene], gene))[
            : FIXED_ARGUMENTS["top_mut"]
        ]
    )
    vocabulary_index = {gene: index for index, gene in enumerate(vocabulary)}
    matrix = np.zeros((len(cells), len(vocabulary)), dtype=np.float32)
    for row, mutations in enumerate(cells):
        columns = [
            vocabulary_index[gene]
            for gene in mutations
            if gene in vocabulary_index
        ]
        if columns:
            matrix[row, columns] = 1.0
    gene_index = {str(gene): index for index, gene in enumerate(gene_ids)}
    token_order = tuple(gene for gene in vocabulary if gene in gene_index)
    token_rank = {gene: rank for rank, gene in enumerate(token_order)}
    k_tokens = 64
    tokens = np.zeros((len(cells), k_tokens), dtype=np.int64)
    token_mask = np.zeros((len(cells), k_tokens), dtype=bool)
    for row, mutations in enumerate(cells):
        ordered = sorted(
            (gene for gene in mutations if gene in token_rank),
            key=token_rank.__getitem__,
        )[:k_tokens]
        if ordered:
            indices = np.fromiter(
                (gene_index[gene] for gene in ordered), dtype=np.int64
            )
            tokens[row, : indices.size] = indices
            token_mask[row, : indices.size] = True
    return {
        "vocabulary": vocabulary,
        "token_order": token_order,
        "k_tokens": k_tokens,
        "matrix": matrix,
        "tokens": tokens,
        "token_mask": token_mask,
    }


def _expected_graph_weights(
    edge_index: np.ndarray,
    gene_ids: Sequence[str],
    mutation_matrix: np.ndarray,
    mutation_vocabulary: Sequence[str],
    train_idx: np.ndarray,
) -> np.ndarray:
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("common graph must have shape (2, E)")
    vocabulary_index = {
        str(gene): index for index, gene in enumerate(mutation_vocabulary)
    }
    endpoint_a = np.asarray(
        [vocabulary_index.get(str(gene_ids[row]), -1) for row in edges[0]],
        dtype=np.int64,
    )
    endpoint_b = np.asarray(
        [vocabulary_index.get(str(gene_ids[row]), -1) for row in edges[1]],
        dtype=np.int64,
    )
    valid = np.flatnonzero((endpoint_a >= 0) & (endpoint_b >= 0))
    weights = np.ones(edges.shape[1], dtype=np.float32)
    train = np.asarray(mutation_matrix, dtype=np.float32)[
        _as_unique_indices(train_idx, mutation_matrix.shape[0])
    ]
    prevalence = train.mean(axis=0, dtype=np.float64)
    for start in range(0, valid.size, 20_000):
        positions = valid[start : start + 20_000]
        a = endpoint_a[positions]
        b = endpoint_b[positions]
        joint = np.mean(train[:, a] * train[:, b], axis=0, dtype=np.float64)
        denominator = prevalence[a] * prevalence[b]
        pmi = np.log((joint + 1e-3) / (denominator + 1e-3))
        weights[positions] = np.clip(pmi, 0.0, 3.0).astype(np.float32)
    return weights


def expected_preprocessing_state(
    features: CurrentCommonFeatures,
    gene_effect: np.ndarray,
    outer_train: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Independently refit every fold-specific preprocessing transform."""
    import torch
    from sklearn.decomposition import PCA

    train_idx = _as_unique_indices(outer_train, len(features.cell_ids))
    expression = np.asarray(features.expression_raw, dtype=np.float64)
    train = expression[train_idx]
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(train, axis=0)
        scale = np.nanstd(train, axis=0, ddof=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    train_clean = np.where(np.isfinite(train), train, mean)
    standardized = (train_clean - mean) / scale
    fitted_components = min(
        FIXED_ARGUMENTS["n_expr_pc"], standardized.shape[0], standardized.shape[1]
    )
    pca = PCA(n_components=fitted_components, svd_solver="full", random_state=0)
    pca.fit(standardized)

    mutation = _expected_mutation_features(
        features.mutations_by_cell, features.gene_ids, train_idx
    )
    edge_weight = _expected_graph_weights(
        features.edge_index,
        features.gene_ids,
        mutation["matrix"],
        mutation["vocabulary"],
        train_idx,
    )
    categories = tuple(sorted(set(features.lineages[train_idx].tolist())))
    common_effect = np.asarray(gene_effect, dtype=np.float32)
    if common_effect.shape != (len(features.gene_ids), len(features.cell_ids)):
        raise ValueError("common gene-effect source has an unexpected shape")
    preprocessing = {
        "fit_indices": torch.from_numpy(train_idx.copy()),
        "expression": {
            "input_feature_names": tuple(features.expression_gene_names),
            "mean": torch.from_numpy(mean.astype(np.float64, copy=False)),
            "scale": torch.from_numpy(scale.astype(np.float64, copy=False)),
            "pca_mean": torch.from_numpy(
                pca.mean_.astype(np.float64, copy=False)
            ),
            "components": torch.from_numpy(
                pca.components_.astype(np.float64, copy=False)
            ),
            "explained_variance_ratio": torch.from_numpy(
                pca.explained_variance_ratio_.astype(np.float64, copy=False)
            ),
            "n_output_components": FIXED_ARGUMENTS["n_expr_pc"],
        },
        "mutation": {
            "vocabulary": mutation["vocabulary"],
            "token_order": mutation["token_order"],
            "k_tokens": mutation["k_tokens"],
        },
        "graph": {
            "edge_index": torch.from_numpy(features.edge_index.copy()).long(),
            "edge_weight": torch.from_numpy(edge_weight.copy()).float(),
        },
        "lineage": {"categories": categories},
        "selectivity_labels": torch.from_numpy(
            selectivity_labels_from_train(common_effect, train_idx)
        ),
    }
    return preprocessing, mutation


def _independent_normalized_adjacency(
    edge_index: Any,
    edge_weight: Any,
    n_genes: int,
) -> Any:
    """Reconstruct the exact sparse adjacency without importing model code."""
    import torch

    edges = edge_index.detach().cpu().long()
    weights = edge_weight.detach().cpu().float()
    if edges.ndim != 2 or int(edges.shape[0]) != 2:
        raise ValueError("edge_index must have shape (2, E)")
    if weights.ndim != 1 or int(weights.shape[0]) != int(edges.shape[1]):
        raise ValueError("edge weights do not align with edge_index")
    if n_genes < 1:
        raise ValueError("n_genes must be positive")
    if edges.numel() and (
        int(edges.min()) < 0 or int(edges.max()) >= int(n_genes)
    ):
        raise IndexError("edge_index contains a node outside the common cohort")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("edge weights must be finite and non-negative")

    low = torch.minimum(edges[0], edges[1])
    high = torch.maximum(edges[0], edges[1])
    upper_indices = torch.stack([low, high])
    upper_sum = torch.sparse_coo_tensor(
        upper_indices, weights, (n_genes, n_genes)
    ).coalesce()
    upper_count = torch.sparse_coo_tensor(
        upper_indices, torch.ones_like(weights), (n_genes, n_genes)
    ).coalesce()
    unique_indices = upper_sum.indices()
    unique_values = upper_sum.values() / upper_count.values()
    unique_row, unique_col = unique_indices
    off_diagonal = unique_row != unique_col
    row = torch.cat(
        [unique_row, unique_col[off_diagonal], torch.arange(n_genes)]
    )
    column = torch.cat(
        [unique_col, unique_row[off_diagonal], torch.arange(n_genes)]
    )
    values = torch.cat(
        [
            unique_values,
            unique_values[off_diagonal],
            torch.ones(n_genes, dtype=weights.dtype),
        ]
    )
    adjacency = torch.sparse_coo_tensor(
        torch.stack([row, column]), values, (n_genes, n_genes)
    ).coalesce()
    degree = torch.sparse.sum(adjacency, dim=1).to_dense().clamp(min=1e-8)
    inverse_sqrt = 1.0 / degree.sqrt()
    indices = adjacency.indices()
    normalized = (
        adjacency.values()
        * inverse_sqrt[indices[0]]
        * inverse_sqrt[indices[1]]
    )
    return torch.sparse_coo_tensor(
        indices, normalized, (n_genes, n_genes)
    ).coalesce()


def _validate_finite_tensors(value: Any, *, path: str) -> list[str]:
    import torch

    errors: list[str] = []
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        values = tensor.coalesce().values() if tensor.layout == torch.sparse_coo else tensor
        if values.is_floating_point() or values.is_complex():
            if not bool(torch.isfinite(values).all()):
                errors.append(f"{path} contains non-finite tensor values")
        return errors
    if type(value) is dict:
        for key, item in value.items():
            errors.extend(_validate_finite_tensors(item, path=f"{path}.{key}"))
    elif type(value) in (tuple, list):
        for index, item in enumerate(value):
            errors.extend(_validate_finite_tensors(item, path=f"{path}[{index}]"))
    return errors


def reconstruct_replay_inputs(
    preprocessing: Mapping[str, Any],
    mutation: Mapping[str, Any],
    features: CurrentCommonFeatures,
    outer_test: np.ndarray,
) -> dict[str, Any]:
    """Transform current local sources; no replay tensor comes from checkpoint."""
    import torch

    test_idx = _as_unique_indices(outer_test, len(features.cell_ids))
    expression_state = preprocessing["expression"]
    raw_expression = np.asarray(
        features.expression_raw[test_idx], dtype=np.float64
    )
    mean = expression_state["mean"].detach().cpu().numpy()
    scale = expression_state["scale"].detach().cpu().numpy()
    pca_mean = expression_state["pca_mean"].detach().cpu().numpy()
    components = expression_state["components"].detach().cpu().numpy()
    clean = np.where(np.isfinite(raw_expression), raw_expression, mean)
    standardized = (clean - mean) / scale
    expression = (standardized - pca_mean) @ components.T
    n_output = int(expression_state["n_output_components"])
    if expression.shape[1] < n_output:
        expression = np.pad(
            expression,
            ((0, 0), (0, n_output - expression.shape[1])),
            mode="constant",
        )

    categories = tuple(preprocessing["lineage"]["categories"])
    category_index = {category: index for index, category in enumerate(categories)}
    lineage = np.zeros((test_idx.size, len(categories)), dtype=np.float32)
    for row, value in enumerate(features.lineages[test_idx]):
        column = category_index.get(str(value))
        if column is not None:
            lineage[row, column] = 1.0
    return {
        "outer_test_indices": torch.from_numpy(test_idx.copy()),
        "mutation": torch.from_numpy(
            np.ascontiguousarray(mutation["matrix"][test_idx], dtype=np.float32)
        ),
        "expression": torch.from_numpy(
            np.ascontiguousarray(expression, dtype=np.float32)
        ),
        "lineage": torch.from_numpy(lineage),
        "tokens": torch.from_numpy(
            np.ascontiguousarray(mutation["tokens"][test_idx], dtype=np.int64)
        ),
        "token_mask": torch.from_numpy(
            np.ascontiguousarray(mutation["token_mask"][test_idx], dtype=bool)
        ),
    }


def derived_model_spec(
    payload: Mapping[str, Any],
    result: Mapping[str, Any],
    cohort: gatdep.CommonCohort,
    replay_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive model constructor arguments without trusting the saved spec."""
    state = payload["model_state_dict"]
    config = result["best_config"]
    embedding = state["emb"]
    if embedding.ndim != 2 or int(embedding.shape[0]) != len(cohort.gene_ids):
        raise ValueError("checkpoint embedding shape does not match the cohort")
    return {
        "model": str(config["model"]),
        "n_genes": len(cohort.gene_ids),
        "embedding_dim": int(embedding.shape[1]),
        "n_mutation_features": int(replay_inputs["mutation"].shape[1]),
        "n_expression_features": int(replay_inputs["expression"].shape[1]),
        "n_lineage_features": int(replay_inputs["lineage"].shape[1]),
        "hidden": int(config["hidden"]),
        "k_tokens": int(replay_inputs["tokens"].shape[1]),
        "use_pretrained": bool(config.get("use_pretrained", True)),
        "use_gnn": bool(config.get("use_gnn", True)),
        "use_comut": bool(config.get("use_comut", True)),
        "use_multitask": bool(config.get("use_multitask", True)),
        "use_crossattn": bool(config.get("use_crossattn", True)),
        "freeze_embeddings": True,
    }


def replay_checkpoint_predictions(
    payload: Mapping[str, Any],
    *,
    model_spec: Mapping[str, Any],
    replay_inputs: Mapping[str, Any],
) -> np.ndarray:
    """Replay on explicitly reconstructed inputs; never read inputs from payload."""
    import torch
    from models import DepPLM, PLMMLP

    spec = model_spec
    state = payload.get("model_state_dict")
    inputs = replay_inputs
    if type(spec) is not dict or type(state) is not dict:
        raise TypeError("checkpoint model specification/state is not a mapping")
    if type(inputs) is not dict:
        raise TypeError("independently reconstructed replay inputs are not a mapping")
    required_inputs = {
        "outer_test_indices",
        "mutation",
        "expression",
        "lineage",
        "tokens",
        "token_mask",
    }
    if set(inputs) != required_inputs:
        raise ValueError("independently reconstructed replay input fields differ")
    embedding = state["emb"].detach().cpu()
    model_name = str(spec["model"])
    common = {
        "n_mut": int(spec["n_mutation_features"]),
        "n_expr": int(spec["n_expression_features"]),
        "n_lin": int(spec["n_lineage_features"]),
    }
    if model_name == "plmmlp":
        model = PLMMLP(
            embedding,
            **common,
            gene_dim=int(spec["hidden"]),
            freeze=bool(spec["freeze_embeddings"]),
        )
    elif model_name == "depplm":
        model = DepPLM(
            embedding,
            state["adj"].detach().cpu(),
            state["ei"].detach().cpu(),
            state["ew"].detach().cpu(),
            **common,
            hidden=int(spec["hidden"]),
            k_tokens=int(spec["k_tokens"]),
            use_pretrained=bool(spec["use_pretrained"]),
            use_gnn=bool(spec["use_gnn"]),
            use_comut=bool(spec["use_comut"]),
            use_multitask=bool(spec["use_multitask"]),
            use_crossattn=bool(spec["use_crossattn"]),
            freeze=bool(spec["freeze_embeddings"]),
        )
    else:
        raise ValueError(f"unsupported checkpoint model: {model_name}")
    model.load_state_dict(dict(state), strict=True)
    model.to("cpu")
    model.eval()

    batch_size = int(payload["batch_size"])
    if batch_size < 1:
        raise ValueError("checkpoint batch size must be positive")
    n_rows = int(inputs["outer_test_indices"].numel())
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, n_rows, batch_size):
            stop = min(start + batch_size, n_rows)
            mutation = inputs["mutation"][start:stop].detach().cpu()
            expression = inputs["expression"][start:stop].detach().cpu()
            lineage = inputs["lineage"][start:stop].detach().cpu()
            if model_name == "plmmlp":
                prediction = model(mutation, expression, lineage)
            else:
                prediction, _, _ = model(
                    mutation,
                    expression,
                    lineage,
                    inputs["tokens"][start:stop].detach().cpu(),
                    inputs["token_mask"][start:stop].detach().cpu(),
                )
            chunks.append(prediction.float().cpu().numpy())
    if not chunks:
        raise ValueError("there are no independently reconstructed outer-test rows")
    return np.ascontiguousarray(np.concatenate(chunks, axis=0), dtype=np.float32)


def verify_checkpoint_replay(
    path: Path,
    record: Mapping[str, Any],
    *,
    prediction_path: Path,
    result: Mapping[str, Any],
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    cohort: gatdep.CommonCohort,
    current_features: CurrentCommonFeatures,
    common_gene_effect: np.ndarray,
    source_input_inventory_sha256: str,
) -> tuple[list[str], dict[str, Any]]:
    import torch

    errors: list[str] = []
    expected_file = gatdep.file_record(path, relative_to=path.parent)
    replay_report: dict[str, Any] = {
        "fold": str(result.get("fold", path.stem.replace("-checkpoint", ""))),
        "checkpoint_sha256": expected_file["sha256"],
        "prediction_sha256": gatdep.sha256_file(prediction_path),
        "preprocessing_state_sha256": None,
        "model_state_sha256": None,
        "reconstructed_inputs_sha256": None,
        "source_input_inventory_sha256": source_input_inventory_sha256,
        "replay_inputs_embedded": False,
        "replay_input_policy": REPLAY_INPUT_POLICY,
        "preprocessing_reconstructed_from_current_sources": False,
        "elementwise_tolerance": dict(REPLAY_TOLERANCE),
        "max_abs_error": None,
        "max_rel_error": None,
        "n_elements_compared": 0,
        "allclose": False,
    }
    expected_record_keys = {
        "schema",
        "path",
        "bytes",
        "sha256",
        "format",
        "weights_only_load_required",
        "final_config",
        "selected_epoch",
        "trained_epochs",
        "prediction_sha256",
        "model_state_sha256",
        "preprocessing_state_sha256",
        "contains_cell_level_replay_inputs",
        "preprocessing_contains_raw_cell_level_values",
        "preprocessing_contains_source_derived_statistics",
        "preprocessing_content_policy",
        "replay_input_policy",
        "source_input_inventory_sha256",
        "redistribution_status",
        "elementwise_tolerance",
    }
    if type(record) is not dict:
        errors.append("checkpoint record is not a plain mapping")
        return errors, replay_report
    elif set(record) != expected_record_keys:
        errors.append("checkpoint record field inventory differs")
    if record.get("schema") != CHECKPOINT_SCHEMA:
        errors.append("checkpoint record schema differs")
    for field in ("path", "bytes", "sha256"):
        if record.get(field) != expected_file[field]:
            errors.append(f"checkpoint record {field} differs")
    if record.get("format") != "torch_save_weights_only_compatible":
        errors.append("checkpoint format differs")
    if record.get("weights_only_load_required") is not True:
        errors.append("checkpoint does not require safe weights-only loading")
    if record.get("elementwise_tolerance") != REPLAY_TOLERANCE:
        errors.append("checkpoint record replay tolerance differs")
    if record.get("contains_cell_level_replay_inputs") is not False:
        errors.append("checkpoint record permits embedded cell-level replay inputs")
    if record.get("preprocessing_contains_raw_cell_level_values") is not False:
        errors.append("checkpoint record permits raw cell-level preprocessing values")
    if record.get("preprocessing_contains_source_derived_statistics") is not True:
        errors.append("checkpoint record omits the source-derived-statistics disclosure")
    if record.get("preprocessing_content_policy") != PREPROCESSING_CONTENT_POLICY:
        errors.append("checkpoint preprocessing content policy differs")
    if record.get("replay_input_policy") != REPLAY_INPUT_POLICY:
        errors.append("checkpoint replay-input policy differs")
    if record.get("source_input_inventory_sha256") != source_input_inventory_sha256:
        errors.append("checkpoint record source-input inventory hash differs")
    if record.get("redistribution_status") != GENERATED_ARTIFACT_REDISTRIBUTION_STATUS:
        errors.append("checkpoint redistribution status differs")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, TypeError, pickle.UnpicklingError, EOFError) as exc:
        errors.append(f"checkpoint cannot be loaded with weights_only=True: {exc}")
        return errors, replay_report
    if type(payload) is not dict:
        errors.append("checkpoint payload is not a plain mapping")
        return errors, replay_report
    try:
        _checkpoint_inventory(payload)
    except (RuntimeError, TypeError, ValueError) as exc:
        errors.append(f"checkpoint contains a disallowed serialized value: {exc}")
        return errors, replay_report
    expected_payload_keys = {
        "schema",
        "fold",
        "fold_number",
        "condition",
        "seed",
        "fold_seed",
        "final_config",
        "selected_epoch",
        "trained_epochs",
        "batch_size",
        "prediction_sha256",
        "elementwise_tolerance",
        "model_spec",
        "model_state_dict",
        "preprocessing_state",
        "contains_cell_level_replay_inputs",
        "preprocessing_contains_raw_cell_level_values",
        "preprocessing_contains_source_derived_statistics",
        "preprocessing_content_policy",
        "replay_input_policy",
        "source_input_inventory_sha256",
        "redistribution_status",
        "redistribution_risk",
        "model_state_sha256",
        "preprocessing_state_sha256",
    }
    if set(payload) != expected_payload_keys:
        errors.append("checkpoint payload field inventory differs")

    expected_identity = {
        "schema": CHECKPOINT_SCHEMA,
        "fold": result.get("fold"),
        "fold_number": result.get("fold_number"),
        "condition": result.get("condition"),
        "seed": result.get("seed"),
        "fold_seed": result.get("fold_seed"),
        "final_config": result.get("best_config"),
        "selected_epoch": result.get("best_epoch"),
        "trained_epochs": result.get("optimization", {}).get("refit_epochs"),
        "batch_size": FIXED_ARGUMENTS["batch_size"],
        "prediction_sha256": replay_report["prediction_sha256"],
        "elementwise_tolerance": REPLAY_TOLERANCE,
        "contains_cell_level_replay_inputs": False,
        "preprocessing_contains_raw_cell_level_values": False,
        "preprocessing_contains_source_derived_statistics": True,
        "preprocessing_content_policy": PREPROCESSING_CONTENT_POLICY,
        "replay_input_policy": REPLAY_INPUT_POLICY,
        "source_input_inventory_sha256": source_input_inventory_sha256,
        "redistribution_status": GENERATED_ARTIFACT_REDISTRIBUTION_STATUS,
        "redistribution_risk": (
            "model_weights_and_preprocessing_are_derived_from_DepMap_inputs"
        ),
    }
    for field, value in expected_identity.items():
        if payload.get(field) != value:
            errors.append(f"checkpoint payload field differs: {field}")
    for field in (
        "final_config",
        "selected_epoch",
        "trained_epochs",
        "prediction_sha256",
        "elementwise_tolerance",
        "contains_cell_level_replay_inputs",
        "preprocessing_contains_raw_cell_level_values",
        "preprocessing_contains_source_derived_statistics",
        "preprocessing_content_policy",
        "replay_input_policy",
        "source_input_inventory_sha256",
        "redistribution_status",
    ):
        if record.get(field) != payload.get(field):
            errors.append(f"checkpoint record/payload field differs: {field}")

    components: dict[str, Any] = {}
    for name in ("model_state_dict", "preprocessing_state"):
        value = payload.get(name)
        if type(value) is not dict:
            errors.append(f"checkpoint {name} is not a mapping")
        else:
            components[name] = value
    if len(components) != 2:
        return errors, replay_report
    computed_hashes = {
        "model_state_sha256": checkpoint_component_hash(
            components["model_state_dict"]
        ),
        "preprocessing_state_sha256": checkpoint_component_hash(
            components["preprocessing_state"]
        ),
    }
    replay_report.update(computed_hashes)
    for field, value in computed_hashes.items():
        if payload.get(field) != value:
            errors.append(f"checkpoint payload {field} differs")
        if record.get(field) != value:
            errors.append(f"checkpoint record {field} differs")

    model_state = components["model_state_dict"]
    preprocessing = components["preprocessing_state"]
    errors.extend(_validate_finite_tensors(model_state, path="model_state_dict"))
    errors.extend(
        _validate_finite_tensors(preprocessing, path="preprocessing_state")
    )
    try:
        expected_preprocessing_state_value, mutation = expected_preprocessing_state(
            current_features, common_gene_effect, outer_train
        )
        if _checkpoint_inventory(preprocessing) != _checkpoint_inventory(
            expected_preprocessing_state_value
        ):
            errors.append(
                "checkpoint preprocessing does not independently reconstruct "
                "from current source inputs"
            )
        replay_inputs = reconstruct_replay_inputs(
            expected_preprocessing_state_value,
            mutation,
            current_features,
            outer_test,
        )
        replay_report["reconstructed_inputs_sha256"] = checkpoint_component_hash(
            replay_inputs
        )
        replay_report["preprocessing_reconstructed_from_current_sources"] = True
    except (KeyError, RuntimeError, TypeError, ValueError, IndexError) as exc:
        errors.append(f"source-based preprocessing/input reconstruction failed: {exc}")
        return errors, replay_report

    fit_indices = preprocessing.get("fit_indices")
    if not isinstance(fit_indices, torch.Tensor) or not np.array_equal(
        fit_indices.cpu().numpy(), outer_train
    ):
        errors.append("checkpoint preprocessing fit indices differ")

    expected_preprocessing = result.get("preprocessing", {}).get(
        "outer_train_refit", {}
    )
    expression_state = expected_preprocessing_state_value.get("expression", {})
    mutation_state = expected_preprocessing_state_value.get("mutation", {})
    graph_state = expected_preprocessing_state_value.get("graph", {})
    lineage_state = expected_preprocessing_state_value.get("lineage", {})
    try:
        preprocessing_checks = {
            "expression mean": (
                hash_array(expression_state["mean"].cpu().numpy()),
                expected_preprocessing["expression"]["mean_sha256"],
            ),
            "expression scale": (
                hash_array(expression_state["scale"].cpu().numpy()),
                expected_preprocessing["expression"]["scale_sha256"],
            ),
            "expression components": (
                hash_array(expression_state["components"].cpu().numpy()),
                expected_preprocessing["expression"]["components_sha256"],
            ),
            "mutation vocabulary": (
                hash_strings(mutation_state["vocabulary"]),
                expected_preprocessing["mutation"]["vocabulary_sha256"],
            ),
            "mutation token order": (
                hash_strings(mutation_state["token_order"]),
                expected_preprocessing["mutation"]["token_order_sha256"],
            ),
            "graph edge weight": (
                hash_array(graph_state["edge_weight"].cpu().numpy()),
                expected_preprocessing["graph"]["edge_weight_sha256"],
            ),
            "selectivity labels": (
                hash_array(
                    expected_preprocessing_state_value[
                        "selectivity_labels"
                    ].cpu().numpy()
                ),
                expected_preprocessing["selectivity_labels_sha256"],
            ),
        }
        for name, (observed, expected) in preprocessing_checks.items():
            if observed != expected:
                errors.append(f"checkpoint {name} does not match result manifest")
        if list(lineage_state["categories"]) != expected_preprocessing["lineage"][
            "categories"
        ]:
            errors.append("checkpoint lineage categories do not match result manifest")
        if int(mutation_state["k_tokens"]) != expected_preprocessing["mutation"][
            "k_tokens"
        ]:
            errors.append("checkpoint mutation K does not match result manifest")
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        errors.append(f"checkpoint preprocessing state is incomplete: {exc}")

    try:
        input_tensors = {
            name: replay_inputs[name]
            for name in ("mutation", "expression", "lineage", "tokens", "token_mask")
        }
        if not all(isinstance(value, torch.Tensor) for value in input_tensors.values()):
            raise TypeError("a replay input is not a tensor")
        if any(int(value.shape[0]) != outer_test.size for value in input_tensors.values()):
            errors.append("reconstructed replay input row count differs")
        if input_tensors["tokens"].shape != input_tensors["token_mask"].shape:
            errors.append("reconstructed token and token-mask shapes differ")
        if int(input_tensors["tokens"].shape[1]) != 64:
            errors.append("reconstructed replay token width differs")
        if not np.array_equal(
            replay_inputs["outer_test_indices"].cpu().numpy(), outer_test
        ):
            errors.append("reconstructed outer-test indices differ")
        errors.extend(_validate_finite_tensors(replay_inputs, path="replay_inputs"))
    except (KeyError, TypeError, IndexError) as exc:
        errors.append(f"reconstructed replay inputs are incomplete: {exc}")

    try:
        expected_model_spec = derived_model_spec(
            payload, result, cohort, replay_inputs
        )
        if payload.get("model_spec") != expected_model_spec:
            errors.append("checkpoint model specification does not derive independently")
        embedding = model_state.get("emb")
        expected_embedding = torch.from_numpy(
            np.ascontiguousarray(current_features.embeddings, dtype=np.float32)
        )
        if not isinstance(embedding, torch.Tensor) or _tensor_record(
            embedding
        ) != _tensor_record(expected_embedding):
            errors.append("checkpoint embedding does not match current source embedding")
        if expected_model_spec["model"] == "depplm":
            edge_index = expected_preprocessing_state_value["graph"]["edge_index"]
            edge_weight = expected_preprocessing_state_value["graph"]["edge_weight"]
            expected_graph_state = {
                "ei": edge_index,
                "ew": edge_weight,
                "adj": _independent_normalized_adjacency(
                    edge_index, edge_weight, len(cohort.gene_ids)
                ),
                "adj_static": _independent_normalized_adjacency(
                    edge_index,
                    torch.ones(int(edge_index.shape[1]), dtype=torch.float32),
                    len(cohort.gene_ids),
                ),
            }
            for name, expected_tensor in expected_graph_state.items():
                observed_tensor = model_state.get(name)
                if not isinstance(observed_tensor, torch.Tensor) or _tensor_record(
                    observed_tensor
                ) != _tensor_record(expected_tensor):
                    errors.append(
                        f"checkpoint model graph buffer {name} does not reconstruct"
                    )
    except (KeyError, RuntimeError, TypeError, ValueError, IndexError) as exc:
        errors.append(f"checkpoint model state validation failed: {exc}")
        expected_model_spec = None

    if errors:
        return errors, replay_report
    try:
        with np.load(prediction_path, allow_pickle=False) as bundle:
            stored_prediction = np.asarray(bundle["prediction"], dtype=np.float32)
        replayed_prediction = replay_checkpoint_predictions(
            payload,
            model_spec=expected_model_spec,
            replay_inputs=replay_inputs,
        )
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        errors.append(f"checkpoint inference replay failed: {exc}")
        return errors, replay_report
    if replayed_prediction.shape != stored_prediction.shape:
        errors.append("checkpoint replay prediction shape differs")
        return errors, replay_report
    if stored_prediction.size == 0:
        errors.append("stored checkpoint prediction is empty")
        return errors, replay_report
    if not np.isfinite(stored_prediction).all():
        errors.append("stored checkpoint prediction contains non-finite values")
        return errors, replay_report
    if not np.isfinite(replayed_prediction).all():
        errors.append("replayed checkpoint prediction contains non-finite values")
        return errors, replay_report
    absolute = np.abs(replayed_prediction.astype(np.float64) - stored_prediction)
    relative = absolute / np.maximum(
        np.abs(stored_prediction.astype(np.float64)), REPLAY_TOLERANCE["atol"]
    )
    replay_report["max_abs_error"] = float(np.max(absolute))
    replay_report["max_rel_error"] = float(np.max(relative))
    replay_report["n_elements_compared"] = int(stored_prediction.size)
    replay_report["allclose"] = bool(
        np.allclose(
            replayed_prediction,
            stored_prediction,
            rtol=REPLAY_TOLERANCE["rtol"],
            atol=REPLAY_TOLERANCE["atol"],
            equal_nan=False,
        )
    )
    if not replay_report["allclose"]:
        errors.append(
            "checkpoint replay predictions differ beyond the prespecified "
            "elementwise tolerance"
        )
    return errors, replay_report


def verify_inventory(
    observed: Any,
    expected: Sequence[Mapping[str, Any]],
    *,
    name: str,
) -> list[str]:
    if not isinstance(observed, list):
        return [f"{name} is not a list"]
    return [] if observed == list(expected) else [f"{name} differs from current files"]


def verify_run(
    run_dir: Path,
    project_root: Path,
    official_root: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = run_dir / "run-manifest.json"
    try:
        manifest = load_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        manifest = {}
        errors.append(f"run manifest cannot be read: {exc}")

    condition = manifest.get("condition")
    seed_value = manifest.get("seed")
    seed = int(seed_value) if isinstance(seed_value, int) else -1
    if manifest.get("schema") != RUN_SCHEMA:
        errors.append("run schema differs")
    if manifest.get("scientific_result") is not False:
        errors.append("run manifest must not self-declare a scientific result")
    if condition not in SUPPORTED_CONDITIONS:
        errors.append("unsupported or missing comparison condition")
    if not isinstance(seed_value, int) or isinstance(seed_value, bool):
        errors.append("run seed is invalid")
    if manifest.get("planned_folds") != list(range(5)):
        errors.append("run does not plan all five folds")
    if manifest.get("fixed_arguments") != FIXED_ARGUMENTS:
        errors.append("fixed corrected-protocol arguments differ")
    if manifest.get("tuning_grid") != EXPECTED_TUNING_GRID:
        errors.append("fixed tuning grid differs")
    if (
        condition in SUPPORTED_CONDITIONS
        and manifest.get("condition_configuration")
        != EXPECTED_CONDITION_CONFIGURATIONS[condition]
    ):
        errors.append("condition configuration differs")
    if manifest.get("endpoint") != "raw_DepMap_gene_effect_common_2341_genes":
        errors.append("comparison endpoint differs")
    if manifest.get("outer_test_policy") != (
        "exactly_once_per_fold_after_fixed_epoch_refit"
    ):
        errors.append("outer-test policy differs")
    if manifest.get("checkpoint_policy") != CHECKPOINT_POLICY:
        errors.append("checkpoint policy differs")
    if manifest.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
        errors.append("checkpoint schema differs")
    if manifest.get("checkpoint_replay_tolerance") != REPLAY_TOLERANCE:
        errors.append("checkpoint replay tolerance differs")
    if manifest.get("checkpoint_contains_cell_level_replay_inputs") is not False:
        errors.append("run manifest permits cell-level replay inputs in checkpoints")
    if (
        manifest.get("checkpoint_preprocessing_contains_raw_cell_level_values")
        is not False
    ):
        errors.append("run manifest permits raw cell-level preprocessing values")
    if (
        manifest.get("checkpoint_preprocessing_contains_source_derived_statistics")
        is not True
    ):
        errors.append("run manifest omits the source-derived-statistics disclosure")
    if (
        manifest.get("checkpoint_preprocessing_content_policy")
        != PREPROCESSING_CONTENT_POLICY
    ):
        errors.append("run manifest preprocessing content policy differs")
    if manifest.get("replay_input_policy") != REPLAY_INPUT_POLICY:
        errors.append("run manifest replay-input policy differs")
    if (
        manifest.get("generated_artifact_redistribution_status")
        != GENERATED_ARTIFACT_REDISTRIBUTION_STATUS
    ):
        errors.append("generated-artifact redistribution status differs")
    if manifest.get("execution_device") not in {"cpu", "cuda"}:
        errors.append("execution device is missing or invalid")
    if manifest.get("official_commit") != gatdep.OFFICIAL_COMMIT:
        errors.append("run is not bound to the pinned GATDep commit")
    if manifest.get("official_code_executed") is not False:
        errors.append("common comparator unexpectedly claims official code execution")
    if manifest.get("third_party_redistribution_status") != (
        gatdep.REDISTRIBUTION_LICENSE_STATUS
    ):
        errors.append("third-party redistribution ambiguity is not retained")
    if manifest.get("third_party_materials_in_public_release") is not False:
        errors.append("run permits third-party materials in the public release")

    try:
        observed_head = gatdep._git(official_root, "rev-parse", "HEAD")
        if observed_head != gatdep.OFFICIAL_COMMIT:
            errors.append("current official checkout is not at the pinned commit")
        if gatdep._git(official_root, "status", "--porcelain", "--untracked-files=no"):
            errors.append("current official checkout has tracked modifications")
    except gatdep.AuditError as exc:
        errors.append(str(exc))

    expected_sources = scoped_records(project_root, SOURCE_PATHS)
    expected_project_inputs = scoped_records(project_root, PROJECT_INPUT_PATHS)
    expected_official_inputs = scoped_records(official_root, OFFICIAL_INPUT_PATHS)
    source_input_inventory_sha256 = gatdep.canonical_hash(expected_project_inputs)
    errors.extend(
        verify_inventory(
            manifest.get("source_files"), expected_sources, name="source_files"
        )
    )
    errors.extend(
        verify_inventory(
            manifest.get("project_input_files"),
            expected_project_inputs,
            name="project_input_files",
        )
    )
    errors.extend(
        verify_inventory(
            manifest.get("official_input_files"),
            expected_official_inputs,
            name="official_input_files",
        )
    )

    cohort = gatdep.build_common_cohort(project_root, official_root)
    contract = cohort.contract()
    if (len(cohort.cell_ids), len(cohort.gene_ids)) != (1_088, 2_341):
        errors.append("current common cohort is not 1088 x 2341")
    if manifest.get("cohort_contract") != contract:
        errors.append("cohort contract does not reproduce")
    if manifest.get("cohort_contract_sha256") != contract["sha256"]:
        errors.append("cohort contract hash differs")
    split_contract = gatdep.outer_split_contract(cohort, seed=seed, n_splits=5)
    if manifest.get("outer_split_contract") != split_contract:
        errors.append("outer split contract does not reproduce")
    if manifest.get("outer_split_contract_sha256") != split_contract["sha256"]:
        errors.append("outer split contract hash differs")

    try:
        cohort_record = load_json(run_dir / "cohort-manifest.json")
        if cohort_record.get("schema") != COHORT_SCHEMA:
            errors.append("cohort manifest schema differs")
        if cohort_record.get("cohort_contract") != contract:
            errors.append("cohort manifest contract differs")
        if cohort_record.get("source_cell_indices") != (
            cohort.project_cell_indices.tolist()
        ):
            errors.append("cohort source cell indices differ")
        if cohort_record.get("source_gene_indices") != (
            cohort.project_gene_indices.tolist()
        ):
            errors.append("cohort source gene indices differ")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"cohort manifest cannot be read: {exc}")

    target_source = np.load(
        project_root / "data/processed/gene_effect.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    current_features: CurrentCommonFeatures | None = None
    common_gene_effect: np.ndarray | None = None
    try:
        current_features = load_current_common_features(project_root, cohort)
        common_gene_effect = np.asarray(
            target_source[
                np.ix_(cohort.project_gene_indices, cohort.project_cell_indices)
            ],
            dtype=np.float32,
        )
    except (OSError, RuntimeError, TypeError, ValueError, IndexError) as exc:
        errors.append(f"current replay sources cannot be reconstructed: {exc}")
    folds = make_kfold_outer_splits(1_088, 5, seed)
    completed_folds: list[int] = []
    prediction_metrics: list[dict[str, Any]] = []
    coverage: list[int] = []
    fold_artifacts: list[dict[str, Any]] = []
    fold_replay: dict[str, dict[str, Any]] = {}
    replayed_folds: list[int] = []
    for fold_number, (outer_train, outer_test) in enumerate(folds):
        prefix = f"fold-{fold_number:02d}"
        split_path = run_dir / f"{prefix}-split.json"
        result_path = run_dir / f"{prefix}-result.json"
        prediction_path = run_dir / f"{prefix}-predictions.npz"
        checkpoint_path = run_dir / f"{prefix}-checkpoint.pt"
        missing = [
            path.name
            for path in (split_path, result_path, prediction_path, checkpoint_path)
            if not path.is_file()
        ]
        if missing:
            errors.append(f"fold {fold_number} artifacts missing: {', '.join(missing)}")
            continue
        completed_folds.append(fold_number)
        for path in (split_path, result_path, prediction_path, checkpoint_path):
            fold_artifacts.append(gatdep.file_record(path, relative_to=run_dir))
        split = load_json(split_path)
        expected = expected_split(
            fold_number=fold_number,
            outer_train=outer_train,
            outer_test=outer_test,
            cohort=cohort,
            seed=seed,
        )
        if split != _json_value(expected):
            errors.append(f"fold {fold_number} split does not reproduce")
        result = load_json(result_path)
        expected_identity = {
            "fold": prefix,
            "fold_number": fold_number,
            "condition": condition,
            "protocol": "p1",
            "seed": seed,
            "fold_seed": derive_seed(seed, prefix),
            "pilot": False,
            "scientific_result": False,
            "result_scope": "single_outer_fold",
            "test_evaluation_calls": 1,
            "checkpoint_replay_required": True,
        }
        for field, value in expected_identity.items():
            if result.get(field) != value:
                errors.append(f"fold {fold_number} result field differs: {field}")
        allowed_configs = [
            {**EXPECTED_CONDITION_CONFIGURATIONS.get(str(condition), {}), **candidate}
            for candidate in EXPECTED_TUNING_GRID
        ]
        if result.get("best_config") not in allowed_configs:
            errors.append(f"fold {fold_number} selected configuration is not prespecified")
        best_epoch = result.get("best_epoch")
        if (
            not isinstance(best_epoch, int)
            or isinstance(best_epoch, bool)
            or not 1 <= best_epoch <= FIXED_ARGUMENTS["inner_max_epochs"]
        ):
            errors.append(f"fold {fold_number} selected epoch is invalid")
        prediction_errors, metrics = verify_prediction_bundle(
            prediction_path,
            result.get("outer_test_predictions", {}),
            outer_train=outer_train,
            outer_test=outer_test,
            cohort=cohort,
            target_source=target_source,
        )
        errors.extend(f"fold {fold_number}: {message}" for message in prediction_errors)
        if metrics is not None:
            errors.extend(
                f"fold {fold_number}: {message}"
                for message in mappings_close(
                    result.get("test_metrics"), metrics, path="test_metrics"
                )
            )
            prediction_metrics.append(metrics)
            coverage.extend(outer_test.tolist())
        if current_features is None or common_gene_effect is None:
            checkpoint_errors = [
                "checkpoint replay skipped because current source reconstruction failed"
            ]
            replay_record = {
                "fold": prefix,
                "checkpoint_sha256": gatdep.sha256_file(checkpoint_path),
                "prediction_sha256": gatdep.sha256_file(prediction_path),
                "preprocessing_state_sha256": None,
                "model_state_sha256": None,
                "reconstructed_inputs_sha256": None,
                "source_input_inventory_sha256": source_input_inventory_sha256,
                "replay_inputs_embedded": False,
                "replay_input_policy": REPLAY_INPUT_POLICY,
                "preprocessing_reconstructed_from_current_sources": False,
                "elementwise_tolerance": dict(REPLAY_TOLERANCE),
                "max_abs_error": None,
                "max_rel_error": None,
                "n_elements_compared": 0,
                "allclose": False,
            }
        else:
            checkpoint_errors, replay_record = verify_checkpoint_replay(
                checkpoint_path,
                result.get("outer_test_replay_checkpoint", {}),
                prediction_path=prediction_path,
                result=result,
                outer_train=outer_train,
                outer_test=outer_test,
                cohort=cohort,
                current_features=current_features,
                common_gene_effect=common_gene_effect,
                source_input_inventory_sha256=source_input_inventory_sha256,
            )
        fold_replay[prefix] = replay_record
        errors.extend(f"fold {fold_number}: {message}" for message in checkpoint_errors)
        if not checkpoint_errors and replay_record.get("allclose") is True:
            replayed_folds.append(fold_number)
        preprocessing = result.get("preprocessing", {})
        inner_train = np.asarray(expected["inner_train"]["indices"], dtype=np.int64)
        expected_fit_hashes = {
            "inner_train_fit": hash_array(inner_train.astype("<i8", copy=False)),
            "outer_train_refit": hash_array(outer_train.astype("<i8", copy=False)),
        }
        for scope, fit_hash in expected_fit_hashes.items():
            value = (
                preprocessing.get(scope, {}) if isinstance(preprocessing, dict) else {}
            )
            if value.get("fit_indices_sha256") != fit_hash:
                errors.append(f"fold {fold_number} {scope} fit indices differ")
            mutation = value.get("mutation", {}) if isinstance(value, dict) else {}
            if mutation.get("k_tokens") != 64:
                errors.append(f"fold {fold_number} {scope} K token contract differs")

    exactly_once = sorted(coverage) == list(range(1_088)) and len(coverage) == 1_088
    if not exactly_once:
        errors.append(
            "verified predictions do not cover each outer-test cell exactly once"
        )

    summary_path = run_dir / "summary.json"
    completion_path = run_dir / "completion.json"
    try:
        summary = load_json(summary_path)
        if summary.get("schema") != SUMMARY_SCHEMA:
            errors.append("summary schema differs")
        expected_summary_fields = {
            "scientific_result": False,
            "condition": condition,
            "seed": seed,
            "completed_folds": list(range(5)),
            "outer_test_cells_exactly_once": True,
            "checkpoint_replay_required": True,
            "cohort_contract_sha256": contract["sha256"],
            "outer_split_contract_sha256": split_contract["sha256"],
            "primary_metric": "residual_r2_vs_train_gene_mean",
            "artifacts": fold_artifacts,
        }
        for field, value in expected_summary_fields.items():
            if summary.get(field) != value:
                errors.append(f"summary field differs: {field}")
        if len(prediction_metrics) == 5:
            expected_aggregates: dict[str, dict[str, float | int | None]] = {}
            for name in METRIC_NAMES:
                values = np.asarray(
                    [metrics.get(name, np.nan) for metrics in prediction_metrics],
                    dtype=np.float64,
                )
                finite = values[np.isfinite(values)]
                expected_aggregates[name] = {
                    "mean": float(np.mean(finite)) if finite.size else None,
                    "sd": float(np.std(finite)) if finite.size else None,
                    "n": int(finite.size),
                }
            errors.extend(
                mappings_close(
                    summary.get("metrics"), expected_aggregates, path="summary.metrics"
                )
            )
        completion = load_json(completion_path)
        expected_completion = {
            "schema": COMPLETION_SCHEMA,
            "scientific_result": False,
            "protocol_complete": True,
            "independent_verification_required": True,
            "checkpoint_replay_required": True,
            "run_manifest_sha256": gatdep.sha256_file(manifest_path),
            "cohort_manifest_sha256": gatdep.sha256_file(
                run_dir / "cohort-manifest.json"
            ),
            "summary_sha256": gatdep.sha256_file(summary_path),
            "artifacts": fold_artifacts,
        }
        for field, value in expected_completion.items():
            if completion.get(field) != value:
                errors.append(f"completion field differs: {field}")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"summary/completion cannot be verified: {exc}")

    valid = not errors
    replay_complete = replayed_folds == list(range(5)) and set(fold_replay) == {
        f"fold-{fold:02d}" for fold in range(5)
    }
    return {
        "schema": VERIFICATION_SCHEMA,
        "verified_at_utc": gatdep.utc_now(),
        "scientific_result": False,
        "valid": valid,
        "errors": errors,
        "condition": condition,
        "seed": seed if seed >= 0 else None,
        "completed_folds": completed_folds,
        "outer_test_exactly_once": exactly_once,
        "checkpoint_replay_all_folds": valid and replay_complete,
        "replayed_folds": replayed_folds,
        "elementwise_tolerance": dict(REPLAY_TOLERANCE),
        "replay_inputs_embedded": False,
        "replay_input_policy": REPLAY_INPUT_POLICY,
        "source_input_inventory_sha256": source_input_inventory_sha256,
        "checkpoint_preprocessing_contains_raw_cell_level_values": False,
        "checkpoint_preprocessing_contains_source_derived_statistics": True,
        "checkpoint_preprocessing_content_policy": PREPROCESSING_CONTENT_POLICY,
        "generated_artifact_redistribution_status": (
            GENERATED_ARTIFACT_REDISTRIBUTION_STATUS
        ),
        "fold_replay": fold_replay,
        "predictions_exact_once_and_hash_bound": (
            valid
            and completed_folds == list(range(5))
            and exactly_once
            and replay_complete
        ),
        "cohort_contract_sha256": contract["sha256"],
        "outer_split_contract_sha256": split_contract["sha256"],
        "run_manifest_sha256": gatdep.sha256_file(manifest_path)
        if manifest_path.is_file()
        else None,
        "verification_scope": (
            "Replays fixed splits, target binding, training-only gene means, prediction "
            "hashes, metrics, aggregate summary, and current source/input inventories; "
            "safely reloads every final checkpoint with weights_only=True, verifies its "
            "exact configuration/epoch/preprocessing state against a fresh fit from "
            "current hash-bound sources, reconstructs outer-test features without "
            "embedding cell-level replay rows in the checkpoint, and independently "
            "replays inference on CPU with a prespecified elementwise tolerance."
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--project-root", default=str(Path(__file__).resolve().parent.parent)
    )
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = verify_run(
            Path(args.run_dir).resolve(),
            Path(args.project_root).resolve(),
            Path(args.official_root).resolve(),
        )
    except Exception as exc:
        # The CLI is an audit boundary: malformed artifacts must still leave a
        # machine-readable negative result at the caller's requested path.
        report = {
            "schema": VERIFICATION_SCHEMA,
            "verified_at_utc": gatdep.utc_now(),
            "scientific_result": False,
            "valid": False,
            "verification_complete": False,
            "errors": [
                "verification aborted while parsing or reconstructing artifacts: "
                f"{type(exc).__name__}: {exc}"
            ],
        }
    gatdep.atomic_write_json(Path(args.output).resolve(), report)
    print(json.dumps({"valid": report["valid"], "output": args.output}))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
