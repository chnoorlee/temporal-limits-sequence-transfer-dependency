from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, overload

import numpy as np
import torch
import torch.nn as nn

from models import CellEncoder


__all__ = [
    "DistanceRiskScorer",
    "FamilyInductiveModel",
    "center_residual_by_gene",
    "component_block_bootstrap_delta",
    "context_residual_targets",
    "gene_mean_metrics",
    "gene_mean_targets",
    "nearest_training_distance",
    "optimizer_steps_per_epoch",
]


class FamilyInductiveModel(nn.Module):
    """Sequence-only decomposition into gene mean and context residual.

    The model has no learned gene identifier or per-gene bias. The PLM table is
    a sequence-derived input table; with the default ``freeze=True`` it is not
    trainable. Residual centering is deliberately external to ``forward`` so a
    prediction cannot change when cells are evaluated in different batches.
    """

    def __init__(
        self,
        plm: torch.Tensor,
        n_mut: int,
        n_expr: int,
        n_lin: int,
        hidden: int = 128,
        rank: int = 64,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        if plm.ndim != 2 or plm.shape[0] == 0 or plm.shape[1] == 0:
            raise ValueError("plm must be a non-empty genes-by-features tensor")
        if hidden <= 0 or rank <= 0:
            raise ValueError("hidden and rank must be positive")
        if not freeze:
            raise ValueError(
                "FamilyInductiveModel requires frozen PLM embeddings for the "
                "sequence-only inductive analysis"
            )

        self.emb = nn.Parameter(
            plm.detach().clone().to(dtype=torch.float32),
            requires_grad=False,
        )
        self.sequence_encoder = nn.Sequential(
            nn.Linear(int(plm.shape[1]), hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.mean_head = nn.Linear(hidden, 1)
        self.cell_encoder = CellEncoder(n_mut, n_expr, n_lin, out_dim=hidden)
        self.gene_factor = nn.Linear(hidden, rank, bias=False)
        self.cell_factor = nn.Linear(hidden, rank, bias=False)
        self.rank = int(rank)

    def configure_mean_only_training(self) -> "FamilyInductiveModel":
        """Freeze every parameter outside the sequence encoder and mean head."""
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for module in (self.sequence_encoder, self.mean_head):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        return self

    def trainable_parameter_audit(self) -> dict[str, Any]:
        """Return exact active-parameter counts for matched-budget auditing."""
        active = {
            name: int(parameter.numel())
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        mean_path = {
            name: count
            for name, count in active.items()
            if name.startswith("sequence_encoder.") or name.startswith("mean_head.")
        }
        residual_path = {
            name: count
            for name, count in active.items()
            if name.startswith("cell_encoder.")
            or name.startswith("gene_factor.")
            or name.startswith("cell_factor.")
        }
        return {
            "active_trainable_parameter_count": int(sum(active.values())),
            "mean_path_trainable_parameter_count": int(sum(mean_path.values())),
            "residual_path_trainable_parameter_count": int(sum(residual_path.values())),
            "active_trainable_parameter_names": sorted(active),
        }

    def encode_genes(
        self, gene_subset: torch.Tensor | Sequence[int] | None = None
    ) -> torch.Tensor:
        embeddings = self.emb
        if gene_subset is not None:
            indices = torch.as_tensor(
                gene_subset, dtype=torch.long, device=self.emb.device
            )
            if indices.ndim != 1 or indices.numel() == 0:
                raise ValueError("gene_subset must be a non-empty one-dimensional index")
            embeddings = embeddings.index_select(0, indices)
        return self.sequence_encoder(embeddings)

    def forward(
        self,
        mut: torch.Tensor,
        expr: torch.Tensor,
        lin: torch.Tensor,
        gene_subset: torch.Tensor | Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gene_repr = self.encode_genes(gene_subset)
        gene_mean = self.mean_head(gene_repr).squeeze(-1)
        cell_repr = self.cell_encoder(mut, expr, lin)
        gene_factors = self.gene_factor(gene_repr)
        cell_factors = self.cell_factor(cell_repr)
        residual = (cell_factors @ gene_factors.t()) / math.sqrt(self.rank)
        return gene_mean, residual


@overload
def center_residual_by_gene(
    residual: torch.Tensor, valid_mask: torch.Tensor | None = None
) -> torch.Tensor: ...


@overload
def center_residual_by_gene(
    residual: np.ndarray, valid_mask: np.ndarray | None = None
) -> np.ndarray: ...


def center_residual_by_gene(
    residual: torch.Tensor | np.ndarray,
    valid_mask: torch.Tensor | np.ndarray | None = None,
) -> torch.Tensor | np.ndarray:
    """Center a cells-by-genes residual matrix over evaluable cells."""
    if isinstance(residual, torch.Tensor):
        if residual.ndim != 2:
            raise ValueError("residual must be a two-dimensional cells-by-genes tensor")
        if valid_mask is None:
            mask = torch.isfinite(residual)
        else:
            mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=residual.device)
            if mask.shape != residual.shape:
                raise ValueError("valid_mask must have the same shape as residual")
            if torch.any(mask & ~torch.isfinite(residual)).item():
                raise ValueError("valid residual entries must be finite")
        counts = mask.sum(dim=0)
        if torch.any(counts == 0).item():
            raise ValueError("every gene must have at least one valid residual")
        sums = torch.where(mask, residual, torch.zeros_like(residual)).sum(dim=0)
        means = sums / counts.to(dtype=residual.dtype)
        return torch.where(mask, residual - means.unsqueeze(0), residual)

    values = np.asarray(residual)
    if values.ndim != 2:
        raise ValueError("residual must be a two-dimensional cells-by-genes array")
    if valid_mask is None:
        mask_array = np.isfinite(values)
    else:
        mask_array = np.asarray(valid_mask, dtype=bool)
        if mask_array.shape != values.shape:
            raise ValueError("valid_mask must have the same shape as residual")
        if np.any(mask_array & ~np.isfinite(values)):
            raise ValueError("valid residual entries must be finite")
    counts_array = mask_array.sum(axis=0)
    if np.any(counts_array == 0):
        raise ValueError("every gene must have at least one valid residual")
    sums_array = np.where(mask_array, values, 0.0).sum(axis=0, dtype=np.float64)
    means_array = sums_array / counts_array
    centered = np.asarray(values, dtype=np.float64) - means_array[None, :]
    return np.where(mask_array, centered, values)


def _validated_gene_indices(gene_indices: Sequence[int], n_genes: int) -> np.ndarray:
    raw = np.asarray(gene_indices)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("gene_indices must be a non-empty one-dimensional sequence")
    if raw.dtype.kind not in "iu":
        raise TypeError("gene_indices must contain integers")
    indices = raw.astype(np.int64, copy=False)
    if np.any(indices < 0) or np.any(indices >= n_genes):
        raise IndexError("gene_indices contains an out-of-range index")
    return indices


def gene_mean_targets(
    y_gene_by_cell: np.ndarray, gene_indices: Sequence[int]
) -> np.ndarray:
    """Compute finite cell means for selected genes, rejecting empty targets."""
    outcomes = np.asarray(y_gene_by_cell, dtype=np.float64)
    if outcomes.ndim != 2:
        raise ValueError("y_gene_by_cell must be two-dimensional")
    indices = _validated_gene_indices(gene_indices, outcomes.shape[0])
    selected = outcomes[indices]
    valid = np.isfinite(selected)
    counts = valid.sum(axis=1)
    if np.any(counts == 0):
        missing = indices[counts == 0].tolist()
        raise ValueError(f"selected genes have no finite outcomes: {missing}")
    sums = np.where(valid, selected, 0.0).sum(axis=1, dtype=np.float64)
    return sums / counts


def context_residual_targets(
    y_gene_by_cell: np.ndarray, gene_indices: Sequence[int]
) -> np.ndarray:
    """Subtract each selected gene's finite pan-context mean."""
    outcomes = np.asarray(y_gene_by_cell, dtype=np.float64)
    if outcomes.ndim != 2:
        raise ValueError("y_gene_by_cell must be two-dimensional")
    indices = _validated_gene_indices(gene_indices, outcomes.shape[0])
    selected = outcomes[indices]
    means = gene_mean_targets(outcomes, indices)
    return selected - means[:, None]


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    centered_x = x - np.mean(x)
    centered_y = y - np.mean(y)
    denominator = math.sqrt(
        float(np.dot(centered_x, centered_x) * np.dot(centered_y, centered_y))
    )
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(centered_x, centered_y) / denominator)


def gene_mean_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    global_train_mean: float,
) -> dict[str, float | int]:
    """Evaluate held-out gene means against a train-gene global baseline."""
    predicted = np.asarray(prediction, dtype=np.float64)
    observed = np.asarray(target, dtype=np.float64)
    if predicted.ndim != 1 or observed.shape != predicted.shape:
        raise ValueError("prediction and target must be same-length vectors")
    baseline = float(global_train_mean)
    if not math.isfinite(baseline):
        raise ValueError("global_train_mean must be finite")
    valid = np.isfinite(observed)
    if not np.any(valid):
        raise ValueError("target has no finite values")
    if np.any(valid & ~np.isfinite(predicted)):
        raise ValueError("prediction is non-finite for an evaluable target")
    pred = predicted[valid]
    truth = observed[valid]
    error = pred - truth
    squared_error = float(np.dot(error, error))
    baseline_error = truth - baseline
    baseline_sse = float(np.dot(baseline_error, baseline_error))
    pearson = _pearson(pred, truth)
    spearman = _pearson(_average_ranks(pred), _average_ranks(truth))
    return {
        "mse": squared_error / truth.size,
        "mae": float(np.mean(np.abs(error))),
        "r2": (
            float(1.0 - squared_error / baseline_sse)
            if baseline_sse > 0.0
            else float("nan")
        ),
        "spearman": spearman,
        "pearson": pearson,
        "n": int(truth.size),
    }


def component_block_bootstrap_delta(
    candidate_losses: np.ndarray,
    reference_losses: np.ndarray,
    components: Sequence[Any],
    *,
    n_bootstrap: int = 2_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, float | int | str]:
    """Gene-weighted paired loss delta with component-block resampling."""
    candidate = np.asarray(candidate_losses, dtype=np.float64)
    reference = np.asarray(reference_losses, dtype=np.float64)
    component_array = np.asarray(components, dtype=object)
    if candidate.ndim != 1 or reference.shape != candidate.shape:
        raise ValueError("candidate_losses and reference_losses must be equal vectors")
    if component_array.ndim != 1 or component_array.size != candidate.size:
        raise ValueError("components must contain one identifier per gene")
    if candidate.size == 0 or not np.all(np.isfinite(candidate)):
        raise ValueError("candidate_losses must be non-empty and finite")
    if not np.all(np.isfinite(reference)):
        raise ValueError("reference_losses must be finite")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")

    grouped: dict[Any, list[int]] = {}
    for gene_index, raw_component in enumerate(component_array.tolist()):
        component = (
            raw_component.item()
            if isinstance(raw_component, np.generic)
            else raw_component
        )
        if component is None or (
            isinstance(component, (float, np.floating))
            and not math.isfinite(float(component))
        ):
            raise ValueError("component identifiers must be non-missing")
        try:
            grouped.setdefault(component, []).append(gene_index)
        except TypeError as exc:
            raise TypeError("component identifiers must be hashable") from exc

    paired_delta = candidate - reference
    component_sums = np.asarray(
        [np.sum(paired_delta[index]) for index in grouped.values()], dtype=np.float64
    )
    component_sizes = np.asarray(
        [len(index) for index in grouped.values()], dtype=np.int64
    )
    point_delta = float(np.mean(paired_delta))
    rng = np.random.default_rng(seed)
    bootstrap_delta = np.empty(int(n_bootstrap), dtype=np.float64)
    max_draws_per_chunk = 1_000_000
    chunk_rows = max(1, max_draws_per_chunk // component_sums.size)
    for start in range(0, int(n_bootstrap), chunk_rows):
        stop = min(start + chunk_rows, int(n_bootstrap))
        sampled = rng.integers(
            0,
            component_sums.size,
            size=(stop - start, component_sums.size),
        )
        bootstrap_delta[start:stop] = (
            np.sum(component_sums[sampled], axis=1)
            / np.sum(component_sizes[sampled], axis=1)
        )
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(bootstrap_delta, [alpha, 1.0 - alpha])
    return {
        "delta": point_delta,
        "delta_mean_loss": point_delta,
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence": float(confidence),
        "n_components": int(component_sums.size),
        "n_genes": int(candidate.size),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "direction": "candidate_minus_reference",
    }


def _embedding_matrix(name: str, values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0.0):
        raise ValueError(f"{name} contains a zero-norm embedding")
    return matrix / norms[:, None]


def nearest_training_distance(
    training_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    *,
    chunk_size: int = 1_024,
) -> np.ndarray:
    """Return each query's nearest cosine distance to a training embedding."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    training = _embedding_matrix("training_embeddings", training_embeddings)
    query = _embedding_matrix("query_embeddings", query_embeddings)
    if training.shape[1] != query.shape[1]:
        raise ValueError("training and query embeddings must share a feature dimension")
    distances = np.empty(query.shape[0], dtype=np.float64)
    for start in range(0, query.shape[0], chunk_size):
        stop = min(start + chunk_size, query.shape[0])
        maximum_similarity = np.max(query[start:stop] @ training.T, axis=1)
        distances[start:stop] = np.clip(1.0 - maximum_similarity, 0.0, 2.0)
    return distances


def optimizer_steps_per_epoch(n_cells: int, batch_size: int) -> int:
    """Return the fixed number of cell-batch optimizer steps in one epoch."""
    if n_cells <= 0:
        raise ValueError("n_cells must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return int(math.ceil(int(n_cells) / int(batch_size)))


def _fit_nonnegative_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones(x.size, dtype=np.float64), x])
    candidates: list[tuple[float, float]] = []
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    if np.all(coefficients >= 0.0):
        candidates.append((float(coefficients[0]), float(coefficients[1])))
    candidates.append((float(max(np.mean(y), 0.0)), 0.0))
    x_norm = float(np.dot(x, x))
    slope = max(float(np.dot(x, y) / x_norm), 0.0) if x_norm > 0.0 else 0.0
    candidates.append((0.0, slope))
    candidates.append((0.0, 0.0))
    return min(
        candidates,
        key=lambda pair: float(np.sum(np.square(y - (pair[0] + pair[1] * x)))),
    )


@dataclass
class DistanceRiskScorer:
    """Descriptive validation-fitted distance-to-absolute-error risk score.

    The score is used only to rank queries. It is not a prediction interval and
    carries no nominal or empirical coverage guarantee.
    """

    chunk_size: int = 1_024
    intercept_: float | None = None
    slope_: float | None = None
    n_validation_: int | None = None

    def fit(
        self,
        training_embeddings: np.ndarray,
        validation_embeddings: np.ndarray,
        validation_absolute_errors: np.ndarray,
    ) -> "DistanceRiskScorer":
        distances = nearest_training_distance(
            training_embeddings,
            validation_embeddings,
            chunk_size=self.chunk_size,
        )
        errors = np.asarray(validation_absolute_errors, dtype=np.float64)
        if errors.ndim != 1 or errors.size == 0 or errors.size != distances.size:
            raise ValueError("validation_absolute_errors must align with validation genes")
        if not np.all(np.isfinite(errors)) or np.any(errors < 0.0):
            raise ValueError("validation_absolute_errors must be finite and nonnegative")
        intercept, slope = _fit_nonnegative_line(distances, errors)
        self.intercept_ = intercept
        self.slope_ = slope
        self.n_validation_ = int(errors.size)
        return self

    def score(
        self,
        training_embeddings: np.ndarray,
        query_embeddings: np.ndarray,
    ) -> np.ndarray:
        if self.intercept_ is None or self.slope_ is None:
            raise RuntimeError("risk scorer must be fitted before scoring")
        distances = nearest_training_distance(
            training_embeddings,
            query_embeddings,
            chunk_size=self.chunk_size,
        )
        return np.maximum(self.intercept_ + self.slope_ * distances, 0.0)

    def to_dict(self) -> dict[str, float | int | str | None]:
        return {
            "schema": "distance_risk_scorer_v1",
            "type": "validation_fitted_nonnegative_distance_to_absolute_error_score",
            "ranking": "ascending_score",
            "coverage_guarantee": "none",
            "reporting_role": "descriptive_secondary_only",
            "distance_metric": "cosine",
            "fit_scope": "inner_component_heldout_validation_only",
            "regression_constraint": "nonnegative_intercept_and_slope",
            "chunk_size": int(self.chunk_size),
            "intercept": self.intercept_,
            "slope": self.slope_,
            "n_validation": self.n_validation_,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "DistanceRiskScorer":
        expected_keys = {
            "schema",
            "type",
            "ranking",
            "coverage_guarantee",
            "reporting_role",
            "distance_metric",
            "fit_scope",
            "regression_constraint",
            "chunk_size",
            "intercept",
            "slope",
            "n_validation",
        }
        if set(payload) != expected_keys:
            raise ValueError("risk scorer payload keys do not match the v1 schema")
        if payload.get("schema") != "distance_risk_scorer_v1":
            raise ValueError("unsupported risk scorer schema")
        if payload.get("type") != (
            "validation_fitted_nonnegative_distance_to_absolute_error_score"
        ):
            raise ValueError("unsupported risk scorer type")
        if payload.get("ranking") != "ascending_score":
            raise ValueError("unsupported risk score ranking")
        if payload.get("coverage_guarantee") != "none":
            raise ValueError("risk score must not claim a coverage guarantee")
        if payload.get("reporting_role") != "descriptive_secondary_only":
            raise ValueError("unsupported risk score reporting role")
        if payload.get("distance_metric") != "cosine":
            raise ValueError("unsupported distance metric")
        if payload.get("fit_scope") != "inner_component_heldout_validation_only":
            raise ValueError("unsupported risk score fit scope")
        if payload.get("regression_constraint") != (
            "nonnegative_intercept_and_slope"
        ):
            raise ValueError("unsupported risk score regression constraint")
        chunk_size = int(payload["chunk_size"])
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        scorer = cls(chunk_size=chunk_size)
        scorer.intercept_ = (
            None if payload.get("intercept") is None else float(payload["intercept"])
        )
        scorer.slope_ = (
            None if payload.get("slope") is None else float(payload["slope"])
        )
        scorer.n_validation_ = (
            None
            if payload.get("n_validation") is None
            else int(payload["n_validation"])
        )
        if scorer.intercept_ is not None and (
            not math.isfinite(scorer.intercept_) or scorer.intercept_ < 0.0
        ):
            raise ValueError("risk scorer intercept must be finite and nonnegative")
        if scorer.slope_ is not None and (
            not math.isfinite(scorer.slope_) or scorer.slope_ < 0.0
        ):
            raise ValueError("risk scorer slope must be finite and nonnegative")
        if scorer.n_validation_ is not None and scorer.n_validation_ <= 0:
            raise ValueError("risk scorer validation count must be positive")
        return scorer
