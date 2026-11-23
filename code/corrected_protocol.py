"""Leakage-resistant primitives for the corrected DepPLM evaluation.

The legacy pipeline materializes cohort-wide features before it creates data
splits.  This module makes the fitting population explicit in every API.  It
contains no project-level side effects and can therefore be tested with small
sentinel datasets before launching an expensive experiment.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


K_TOKENS = 64


def _as_unique_indices(indices: Sequence[int] | np.ndarray, n: int | None = None) -> np.ndarray:
    out = np.asarray(indices, dtype=np.int64)
    if out.ndim != 1 or out.size == 0:
        raise ValueError("indices must be a non-empty one-dimensional array")
    if np.unique(out).size != out.size:
        raise ValueError("indices must not contain duplicates")
    if n is not None and (int(out.min()) < 0 or int(out.max()) >= n):
        raise IndexError("indices are outside the available rows")
    return out


def derive_seed(seed: int, *parts: object) -> int:
    """Derive a stable 31-bit seed without relying on Python hash randomization."""
    payload = "|".join([str(int(seed)), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch and require deterministic kernels.

    PyTorch raises instead of warning when an operation has no deterministic
    implementation.  A canonical run must therefore stop rather than silently
    weakening its reproducibility contract.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=False)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


@dataclass(frozen=True)
class FittedExpressionPCA:
    """Training-only expression standardizer and PCA projection."""

    mean_: np.ndarray
    scale_: np.ndarray
    pca_mean_: np.ndarray
    components_: np.ndarray
    explained_variance_ratio_: np.ndarray
    n_output_components: int

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.mean_.shape[0]:
            raise ValueError("expression matrix has an incompatible feature dimension")
        clean = np.where(np.isfinite(values), values, self.mean_)
        standardized = (clean - self.mean_) / self.scale_
        scores = (standardized - self.pca_mean_) @ self.components_.T
        if scores.shape[1] < self.n_output_components:
            scores = np.pad(
                scores,
                ((0, 0), (0, self.n_output_components - scores.shape[1])),
                mode="constant",
            )
        return scores.astype(np.float32, copy=False)


def fit_expression_pca(
    expression: np.ndarray,
    train_idx: Sequence[int] | np.ndarray,
    n_components: int = 128,
    seed: int = 42,
) -> FittedExpressionPCA:
    """Fit expression scaling and PCA using only ``train_idx`` rows."""
    from sklearn.decomposition import PCA

    expression = np.asarray(expression, dtype=np.float64)
    if expression.ndim != 2 or expression.shape[1] == 0:
        raise ValueError("expression must be a non-empty two-dimensional matrix")
    train_idx = _as_unique_indices(train_idx, expression.shape[0])
    if train_idx.size < 2:
        raise ValueError("at least two training cells are required for PCA")
    if n_components < 1:
        raise ValueError("n_components must be positive")

    train = expression[train_idx]
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(train, axis=0)
        scale = np.nanstd(train, axis=0, ddof=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    train_clean = np.where(np.isfinite(train), train, mean)
    standardized = (train_clean - mean) / scale
    fitted_components = min(int(n_components), standardized.shape[0], standardized.shape[1])
    pca = PCA(n_components=fitted_components, svd_solver="full", random_state=int(seed))
    pca.fit(standardized)
    return FittedExpressionPCA(
        mean_=mean.astype(np.float64, copy=False),
        scale_=scale.astype(np.float64, copy=False),
        pca_mean_=pca.mean_.astype(np.float64, copy=False),
        components_=pca.components_.astype(np.float64, copy=False),
        explained_variance_ratio_=pca.explained_variance_ratio_.astype(np.float64, copy=False),
        n_output_components=int(n_components),
    )


@dataclass(frozen=True)
class MutationFeatures:
    """Mutation representation whose vocabulary and order were fit on training cells."""

    vocabulary: tuple[str, ...]
    token_order: tuple[str, ...]
    matrix: np.ndarray
    tokens: np.ndarray
    token_mask: np.ndarray
    k_tokens: int


def _deduplicated_mutations(
    mutations_by_cell: Sequence[Iterable[str]],
) -> list[set[str]]:
    return [
        {str(gene) for gene in mutations if gene is not None and str(gene)}
        for mutations in mutations_by_cell
    ]


def fit_mutation_features(
    mutations_by_cell: Sequence[Iterable[str]],
    gene_names: Sequence[str],
    train_idx: Sequence[int] | np.ndarray,
    top_n: int = 5_000,
    k_tokens: int = K_TOKENS,
) -> MutationFeatures:
    """Fit mutation vocabulary/token ranking on training cells, then transform all cells.

    Counts are binary cell prevalence rather than variant-row counts, preventing a
    hypermutated locus with several calls in one cell from receiving extra weight.
    """
    if top_n < 1:
        raise ValueError("top_n must be positive")
    if k_tokens != K_TOKENS:
        raise ValueError(f"corrected protocol requires k_tokens={K_TOKENS}")
    cells = _deduplicated_mutations(mutations_by_cell)
    train_idx = _as_unique_indices(train_idx, len(cells))
    prevalence: Counter[str] = Counter()
    for cell_idx in train_idx:
        prevalence.update(cells[int(cell_idx)])
    ranked = sorted(prevalence, key=lambda gene: (-prevalence[gene], gene))
    vocabulary = tuple(ranked[: int(top_n)])
    vocab_index = {gene: idx for idx, gene in enumerate(vocabulary)}

    matrix = np.zeros((len(cells), len(vocabulary)), dtype=np.float32)
    for cell_idx, mutations in enumerate(cells):
        cols = [vocab_index[gene] for gene in mutations if gene in vocab_index]
        if cols:
            matrix[cell_idx, cols] = 1.0

    gene_index = {str(gene): idx for idx, gene in enumerate(gene_names)}
    token_order = tuple(gene for gene in vocabulary if gene in gene_index)
    token_rank = {gene: rank for rank, gene in enumerate(token_order)}
    tokens = np.zeros((len(cells), k_tokens), dtype=np.int64)
    token_mask = np.zeros((len(cells), k_tokens), dtype=bool)
    for cell_idx, mutations in enumerate(cells):
        ordered = sorted(
            (gene for gene in mutations if gene in token_rank),
            key=token_rank.__getitem__,
        )[:k_tokens]
        if ordered:
            rows = np.fromiter((gene_index[gene] for gene in ordered), dtype=np.int64)
            tokens[cell_idx, : rows.size] = rows
            token_mask[cell_idx, : rows.size] = True

    return MutationFeatures(
        vocabulary=vocabulary,
        token_order=token_order,
        matrix=matrix,
        tokens=tokens[:, :K_TOKENS],
        token_mask=token_mask[:, :K_TOKENS],
        k_tokens=K_TOKENS,
    )


def fit_comutation_edge_weights(
    edge_index: np.ndarray,
    gene_names: Sequence[str],
    mutation_matrix: np.ndarray,
    mutation_vocabulary: Sequence[str],
    train_idx: Sequence[int] | np.ndarray,
    smoothing: float = 1e-3,
    clip: tuple[float, float] = (0.0, 3.0),
    chunk_size: int = 20_000,
) -> np.ndarray:
    """Compute edge-wise PMI from training cells only.

    STRING edges whose endpoints are outside the training-fitted mutation
    vocabulary retain the neutral legacy weight of one.
    """
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim != 2:
        raise ValueError("edge_index must be two-dimensional")
    if edges.shape[0] != 2 and edges.shape[1] == 2:
        edges = edges.T
    if edges.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E) or (E, 2)")
    matrix = np.asarray(mutation_matrix, dtype=np.float32)
    train_idx = _as_unique_indices(train_idx, matrix.shape[0])
    if matrix.ndim != 2 or matrix.shape[1] != len(mutation_vocabulary):
        raise ValueError("mutation matrix and vocabulary do not align")
    if edges.size and (int(edges.min()) < 0 or int(edges.max()) >= len(gene_names)):
        raise IndexError("edge_index refers to an unknown gene row")

    vocab_index = {str(gene): idx for idx, gene in enumerate(mutation_vocabulary)}
    endpoint_a = np.array(
        [vocab_index.get(str(gene_names[row]), -1) for row in edges[0]],
        dtype=np.int64,
    )
    endpoint_b = np.array(
        [vocab_index.get(str(gene_names[row]), -1) for row in edges[1]],
        dtype=np.int64,
    )
    valid = np.flatnonzero((endpoint_a >= 0) & (endpoint_b >= 0))
    weights = np.ones(edges.shape[1], dtype=np.float32)
    train = matrix[train_idx]
    prevalence = train.mean(axis=0, dtype=np.float64)
    for start in range(0, valid.size, int(chunk_size)):
        positions = valid[start : start + int(chunk_size)]
        a = endpoint_a[positions]
        b = endpoint_b[positions]
        joint = np.mean(train[:, a] * train[:, b], axis=0, dtype=np.float64)
        denominator = prevalence[a] * prevalence[b]
        pmi = np.log((joint + smoothing) / (denominator + smoothing))
        weights[positions] = np.clip(pmi, clip[0], clip[1]).astype(np.float32)
    return weights


def selectivity_labels_from_train(
    gene_effect: np.ndarray,
    train_idx: Sequence[int] | np.ndarray,
    top_frac: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Create auxiliary selectivity labels from training-cell outcomes only."""
    y = np.asarray(gene_effect, dtype=np.float32)
    if y.ndim != 2:
        raise ValueError("gene_effect must have shape (genes, cells)")
    train_idx = _as_unique_indices(train_idx, y.shape[1])
    if not 0.0 < top_frac < 1.0:
        raise ValueError("top_frac must be between zero and one")
    train_y = y[:, train_idx]
    median = np.nanmedian(train_y, axis=1, keepdims=True)
    mad = 1.4826 * np.nanmedian(np.abs(train_y - median), axis=1)
    finite = np.isfinite(mad)
    if finite.any():
        threshold = np.nanquantile(mad[finite], 1.0 - top_frac)
        labels = (mad >= threshold).astype(np.float32)
    else:
        labels = np.zeros(y.shape[0], dtype=np.float32)
    labels[~finite] = 0.0
    return labels, mad.astype(np.float32, copy=False)


def fit_gene_mean_baseline(
    gene_effect: np.ndarray,
    train_idx: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Fit a per-gene mean using only the requested training cells.

    Genes with no finite training observations retain ``NaN``.  Downstream
    metrics exclude those genes rather than imputing them from evaluation data.
    """
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


def residualize_against_gene_mean(
    values: np.ndarray,
    train_gene_mean: np.ndarray,
) -> np.ndarray:
    """Subtract a training-only per-gene anchor from a cell-by-gene matrix.

    Missing anchors intentionally propagate to ``NaN`` so downstream training
    masks genes that have no finite observations in the fitting partition.
    """
    matrix = np.asarray(values, dtype=np.float32)
    reference = np.asarray(train_gene_mean, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("values must have shape (cells, genes)")
    if reference.ndim != 1 or reference.size != matrix.shape[1]:
        raise ValueError("train_gene_mean must contain one value per matrix column")
    return np.ascontiguousarray(matrix - reference[None, :], dtype=np.float32)


def restore_from_gene_mean_residual(
    residual: np.ndarray,
    train_gene_mean: np.ndarray,
) -> np.ndarray:
    """Restore absolute predictions from residuals and a training-only anchor."""
    matrix = np.asarray(residual, dtype=np.float32)
    reference = np.asarray(train_gene_mean, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("residual must have shape (cells, genes)")
    if reference.ndim != 1 or reference.size != matrix.shape[1]:
        raise ValueError("train_gene_mean must contain one value per matrix column")
    return np.ascontiguousarray(matrix + reference[None, :], dtype=np.float32)


def fit_validation_residual_scale(
    predicted_residual: np.ndarray,
    observed: np.ndarray,
    train_gene_mean: np.ndarray,
) -> float:
    """Fit a validation-only residual shrinkage factor constrained to [0, 1].

    The zero endpoint is the training-only gene-mean predictor. The one endpoint
    is the unshrunk residual model. The closed-form least-squares coefficient is
    clipped between them; it must be fit on inner validation data and frozen
    before any outer-test evaluation.
    """
    residual = np.asarray(predicted_residual, dtype=np.float32)
    target = np.asarray(observed, dtype=np.float32)
    reference = np.asarray(train_gene_mean, dtype=np.float32)
    if residual.ndim != 2 or target.shape != residual.shape:
        raise ValueError(
            "predicted_residual and observed must have the same two-dimensional shape"
        )
    if reference.ndim != 1 or reference.size != residual.shape[1]:
        raise ValueError("train_gene_mean must contain one value per matrix column")
    valid = np.isfinite(target) & np.isfinite(reference[None, :])
    if not valid.any():
        raise ValueError("no finite target/reference pairs for residual scaling")
    if (valid & ~np.isfinite(residual)).any():
        raise ValueError("predicted_residual is non-finite on evaluable pairs")
    target_residual = target - reference[None, :]
    predicted = residual[valid].astype(np.float64, copy=False)
    expected = target_residual[valid].astype(np.float64, copy=False)
    denominator = float(np.dot(predicted, predicted))
    if denominator <= 0.0:
        return 0.0
    coefficient = float(np.dot(predicted, expected) / denominator)
    return float(np.clip(coefficient, 0.0, 1.0))


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
    """Evaluate dependency predictions against a training-only gene baseline.

    Arrays use ``(evaluation_cells, genes)`` order. The primary residual skill
    is pooled out-of-sample R-squared relative to predicting each gene's mean in
    the fitting cells. Per-cell Pearson correlation is reported both on raw
    values and after subtracting that same training-only gene mean, under
    distinct names so gene-level location effects cannot be mistaken for
    residual predictive skill.
    """
    predicted = np.asarray(prediction, dtype=np.float32)
    observed = np.asarray(target, dtype=np.float32)
    reference = np.asarray(train_gene_mean, dtype=np.float32)
    if predicted.ndim != 2 or observed.shape != predicted.shape:
        raise ValueError("prediction and target must have the same two-dimensional shape")
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
    mse = sse / n_pairs
    baseline_mse = baseline_sse / n_pairs

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
        prediction_residual,
        target_residual,
        valid,
        axis=1,
        min_count=3,
    )
    cellwise_corr = _correlation_by_axis(
        predicted, observed, valid, axis=1, min_count=3
    )
    gene_corr = _correlation_by_axis(
        predicted, observed, valid, axis=0, min_count=3
    )

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
        "mse": float(mse),
        "mae": float(absolute_error.sum(dtype=np.float64) / n_pairs),
        "gene_mean_baseline_mse": float(baseline_mse),
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
        _finite_summary(residual_cellwise_corr, "residual_cellwise_pearson")
    )
    metrics.update(_finite_summary(cellwise_corr, "cellwise_pearson"))
    metrics.update(_finite_summary(gene_corr, "gene_wise_pearson"))
    metrics.update(_finite_summary(gene_mse, "gene_wise_mse"))
    metrics.update(_finite_summary(gene_mae, "gene_wise_mae"))
    metrics.update(_finite_summary(gene_r2, "gene_wise_residual_r2"))
    metrics["n_finite_cell_correlations"] = int(np.isfinite(cellwise_corr).sum())
    metrics["n_finite_gene_correlations"] = int(np.isfinite(gene_corr).sum())
    finite_gene_r2 = gene_r2[np.isfinite(gene_r2)]
    metrics["fraction_genes_positive_residual_r2"] = (
        float(np.mean(finite_gene_r2 > 0.0))
        if finite_gene_r2.size
        else float("nan")
    )
    return metrics


@dataclass(frozen=True)
class FittedLineageEncoder:
    """One-hot encoder with an all-zero representation for unseen lineages."""

    categories: tuple[str, ...]

    def transform(self, lineages: Sequence[str] | np.ndarray) -> np.ndarray:
        values = np.asarray(lineages, dtype=str)
        mapping = {category: idx for idx, category in enumerate(self.categories)}
        encoded = np.zeros((values.size, len(self.categories)), dtype=np.float32)
        for row, value in enumerate(values):
            col = mapping.get(str(value))
            if col is not None:
                encoded[row, col] = 1.0
        return encoded


def fit_lineage_one_hot(
    lineages: Sequence[str] | np.ndarray,
    train_idx: Sequence[int] | np.ndarray,
) -> FittedLineageEncoder:
    values = np.asarray(lineages, dtype=str)
    train_idx = _as_unique_indices(train_idx, values.size)
    categories = tuple(sorted(set(values[train_idx].tolist())))
    if not categories:
        raise ValueError("training cells contain no lineage categories")
    return FittedLineageEncoder(categories=categories)


def make_inner_split(
    outer_train_idx: Sequence[int] | np.ndarray,
    lineages: Sequence[str] | np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic lineage-stratified inner split with singleton groups in train."""
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
    if n_splits < 2 or n_splits > n_cells:
        raise ValueError("n_splits must be between 2 and n_cells")
    rng = np.random.default_rng(int(seed))
    folds = np.array_split(rng.permutation(n_cells), n_splits)
    return [
        (
            np.sort(np.concatenate([folds[j] for j in range(n_splits) if j != i])).astype(
                np.int64
            ),
            np.sort(folds[i]).astype(np.int64),
        )
        for i in range(n_splits)
    ]


def _index_record(indices: np.ndarray, cell_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "indices": [int(value) for value in indices],
        "cell_ids": [str(cell_ids[int(value)]) for value in indices],
        "sha256": hashlib.sha256(indices.astype("<i8", copy=False).tobytes()).hexdigest(),
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
    """Validate split isolation and create an auditable global-index manifest."""
    n_cells = len(cell_ids)
    outer_train = _as_unique_indices(outer_train_idx, n_cells)
    inner_train = _as_unique_indices(inner_train_idx, n_cells)
    inner_val = _as_unique_indices(inner_val_idx, n_cells)
    outer_test = _as_unique_indices(outer_test_idx, n_cells)
    sets = [set(inner_train), set(inner_val), set(outer_test)]
    if any(sets[i].intersection(sets[j]) for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("inner train, inner validation and outer test must be disjoint")
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
        "k_tokens": K_TOKENS,
    }


@dataclass
class NestedSelectionResult:
    best_config: Mapping[str, Any]
    best_epoch: int
    selection_score: float
    tuning_results: list[dict[str, Any]]
    test_metrics: Mapping[str, Any]
    final_model: Any


def run_nested_selection(
    *,
    configs: Sequence[Mapping[str, Any]],
    outer_train_idx: Sequence[int] | np.ndarray,
    inner_train_idx: Sequence[int] | np.ndarray,
    inner_val_idx: Sequence[int] | np.ndarray,
    outer_test_idx: Sequence[int] | np.ndarray,
    tune_fn: Callable[[Mapping[str, Any], np.ndarray, np.ndarray, int], Mapping[str, Any]],
    refit_fn: Callable[[Mapping[str, Any], np.ndarray, int, int], Any],
    test_fn: Callable[[Any, np.ndarray], Mapping[str, Any]],
    seed: int,
    maximize: bool = True,
) -> NestedSelectionResult:
    """Tune on inner validation, refit on outer train, evaluate outer test once."""
    outer_train = _as_unique_indices(outer_train_idx)
    inner_train = _as_unique_indices(inner_train_idx)
    inner_val = _as_unique_indices(inner_val_idx)
    outer_test = _as_unique_indices(outer_test_idx)
    if set(inner_train).intersection(inner_val) or set(outer_train).intersection(outer_test):
        raise ValueError("nested protocol splits must be disjoint")
    if set(inner_train).union(inner_val) != set(outer_train):
        raise ValueError("inner splits must partition outer_train_idx")
    if not configs:
        raise ValueError("at least one tuning configuration is required")

    tuning_results: list[dict[str, Any]] = []
    best_config: Mapping[str, Any] | None = None
    best_epoch: int | None = None
    best_score = -np.inf if maximize else np.inf
    tune_seed = derive_seed(seed, "tune")
    for config in configs:
        raw = dict(tune_fn(config, inner_train.copy(), inner_val.copy(), tune_seed))
        if "selection_score" not in raw or "best_epoch" not in raw:
            raise ValueError("tune_fn must return selection_score and best_epoch")
        selected_updates = raw.get("selected_config_updates", {})
        if not isinstance(selected_updates, Mapping):
            raise ValueError("selected_config_updates must be a mapping")
        overlap = set(config).intersection(selected_updates)
        if any(config[key] != selected_updates[key] for key in overlap):
            raise ValueError(
                "selected_config_updates cannot replace a prespecified config value"
            )
        score = float(raw["selection_score"])
        epoch = int(raw["best_epoch"])
        if not np.isfinite(score) or epoch < 1:
            raise ValueError("tune_fn returned an invalid score or epoch")
        record = {
            "config": dict(config),
            "selection_score": score,
            "best_epoch": epoch,
            "seed": tune_seed,
            **{key: value for key, value in raw.items() if key not in {"selection_score", "best_epoch"}},
        }
        tuning_results.append(record)
        is_better = score > best_score if maximize else score < best_score
        if is_better:
            best_score = score
            best_config = {**dict(config), **dict(selected_updates)}
            best_epoch = epoch

    assert best_config is not None and best_epoch is not None
    refit_seed = derive_seed(seed, "refit")
    final_model = refit_fn(best_config, outer_train.copy(), best_epoch, refit_seed)
    test_metrics = dict(test_fn(final_model, outer_test.copy()))
    return NestedSelectionResult(
        best_config=best_config,
        best_epoch=best_epoch,
        selection_score=float(best_score),
        tuning_results=tuning_results,
        test_metrics=test_metrics,
        final_model=final_model,
    )


class AtomicWriteError(RuntimeError):
    pass


def _safe_slug(value: object) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-.")
    return slug or "run"


def make_unique_run_dir(
    output_root: str | Path, *, protocol: str, condition: str, seed: int
) -> Path:
    """Create a never-reused directory below the caller's output root."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_id = "-".join(
        [
            stamp,
            _safe_slug(protocol),
            _safe_slug(condition),
            f"s{int(seed)}",
            f"p{os.getpid()}",
            uuid.uuid4().hex[:8],
        ]
    )
    path = root / run_id
    path.mkdir(exist_ok=False)
    return path


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_compatible(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_write_json(path: str | Path, payload: Any) -> None:
    """Write valid JSON through a same-directory temporary file, without overwrite."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp-{uuid.uuid4().hex}")
    try:
        serialized = json.dumps(
            _json_compatible(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise AtomicWriteError(
                f"refusing to overwrite existing output: {target}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)
