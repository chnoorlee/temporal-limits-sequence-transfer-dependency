"""Leakage-controlled utilities for K, mutation-token, and PMI sensitivities.

The main corrected runner intentionally freezes its primary definition at K=64,
frequency-ranked mutation tokens, and clipped PMI with a neutral value of one for
edges outside the training-fitted mutation vocabulary.  This module defines
one-factor-at-a-time sensitivity variants without changing that primary protocol.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


MAX_TOKEN_SLOTS = 64
ALLOWED_K = (16, 32, 64)
TOKEN_STRATEGIES = ("frequency", "vep-impact", "driver-lof")
PMI_MODES = ("neutral-one", "valid-only", "unit")


@dataclass(frozen=True)
class SensitivityVariant:
    """A prespecified one-factor-at-a-time sensitivity configuration."""

    name: str
    axis: str
    k_tokens: int
    token_strategy: str
    pmi_mode: str
    reference: bool = False


VARIANTS: tuple[SensitivityVariant, ...] = (
    SensitivityVariant("k16-frequency", "K", 16, "frequency", "neutral-one"),
    SensitivityVariant("k32-frequency", "K", 32, "frequency", "neutral-one"),
    SensitivityVariant(
        "k64-frequency", "reference", 64, "frequency", "neutral-one", True
    ),
    SensitivityVariant(
        "k64-vep-impact", "token", 64, "vep-impact", "neutral-one"
    ),
    SensitivityVariant(
        "k64-driver-lof", "token", 64, "driver-lof", "neutral-one"
    ),
    SensitivityVariant("k64-frequency-unit", "PMI", 64, "frequency", "unit"),
    SensitivityVariant(
        "k64-frequency-valid-only", "PMI", 64, "frequency", "valid-only"
    ),
)


@dataclass(frozen=True)
class TokenAnnotation:
    """Cell-specific aggregate of the actual mutation annotation columns."""

    impact_rank: int = 0
    hotspot: bool = False
    likely_lof: bool = False
    oncogene_high_impact: bool = False
    tumour_suppressor_high_impact: bool = False

    @property
    def driver(self) -> bool:
        return bool(
            self.hotspot
            or self.oncogene_high_impact
            or self.tumour_suppressor_high_impact
        )

    def merge(self, other: "TokenAnnotation") -> "TokenAnnotation":
        return TokenAnnotation(
            impact_rank=max(int(self.impact_rank), int(other.impact_rank)),
            hotspot=bool(self.hotspot or other.hotspot),
            likely_lof=bool(self.likely_lof or other.likely_lof),
            oncogene_high_impact=bool(
                self.oncogene_high_impact or other.oncogene_high_impact
            ),
            tumour_suppressor_high_impact=bool(
                self.tumour_suppressor_high_impact
                or other.tumour_suppressor_high_impact
            ),
        )


@dataclass(frozen=True)
class SensitivityMutationFeatures:
    vocabulary: tuple[str, ...]
    token_order: tuple[str, ...]
    matrix: np.ndarray
    tokens: np.ndarray
    token_mask: np.ndarray
    effective_k: int
    token_strategy: str
    token_counts_before_truncation: np.ndarray


@dataclass(frozen=True)
class PmiFit:
    weights: np.ndarray
    valid_mask: np.ndarray
    raw_pmi: np.ndarray
    statistics: Mapping[str, Any]


IMPACT_RANK = {
    "MODIFIER": 1,
    "LOW": 2,
    "MODERATE": 3,
    "HIGH": 4,
}


def variant_by_name(name: str) -> SensitivityVariant:
    matches = [variant for variant in VARIANTS if variant.name == str(name)]
    if len(matches) != 1:
        raise KeyError(f"unknown sensitivity variant: {name}")
    return matches[0]


def _as_unique_indices(values: Sequence[int] | np.ndarray, size: int) -> np.ndarray:
    indices = np.asarray(values, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("training indices must be a non-empty one-dimensional array")
    if int(indices.min()) < 0 or int(indices.max()) >= int(size):
        raise IndexError("training index is outside the cell cohort")
    if np.unique(indices).size != indices.size:
        raise ValueError("training indices must be unique")
    return indices


def _truthy_series(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").str.strip().str.lower()
    return normalized.isin({"1", "1.0", "true", "t", "yes", "y"})


def load_token_annotations(
    path: str | Path,
    cell_ids: Sequence[str],
    *,
    chunksize: int = 250_000,
) -> list[dict[str, TokenAnnotation]]:
    """Load and aggregate mutation impact fields for exactly ``cell_ids``.

    Multiple variant rows for the same cell/gene are reduced by maxima/OR.  The
    function reads only named columns and fails if the release does not provide
    the prespecified fields.
    """
    if chunksize < 1:
        raise ValueError("chunksize must be positive")
    ordered_cells = [str(value) for value in cell_ids]
    if len(set(ordered_cells)) != len(ordered_cells):
        raise ValueError("cell_ids must be unique")
    wanted = set(ordered_cells)
    required = [
        "ModelID",
        "HugoSymbol",
        "VepImpact",
        "Hotspot",
        "LikelyLoF",
        "TranscriptLikelyLof",
        "OncogeneHighImpact",
        "TumorSuppressorHighImpact",
    ]
    aggregate: dict[tuple[str, str], TokenAnnotation] = {}
    for chunk in pd.read_csv(
        Path(path),
        usecols=required,
        dtype=str,
        chunksize=int(chunksize),
        low_memory=False,
    ):
        frame = chunk.loc[chunk["ModelID"].isin(wanted)].copy()
        frame = frame.dropna(subset=["ModelID", "HugoSymbol"])
        if frame.empty:
            continue
        frame["impact_rank"] = (
            frame["VepImpact"].astype("string").str.upper().map(IMPACT_RANK).fillna(0)
        )
        frame["hotspot"] = _truthy_series(frame["Hotspot"])
        frame["likely_lof"] = _truthy_series(frame["LikelyLoF"]) | _truthy_series(
            frame["TranscriptLikelyLof"]
        )
        frame["oncogene"] = _truthy_series(frame["OncogeneHighImpact"])
        frame["tumour_suppressor"] = _truthy_series(
            frame["TumorSuppressorHighImpact"]
        )
        grouped = frame.groupby(["ModelID", "HugoSymbol"], sort=False, observed=True)[
            [
                "impact_rank",
                "hotspot",
                "likely_lof",
                "oncogene",
                "tumour_suppressor",
            ]
        ].max()
        for row in grouped.itertuples():
            key = (str(row.Index[0]), str(row.Index[1]))
            current = TokenAnnotation(
                impact_rank=int(row.impact_rank),
                hotspot=bool(row.hotspot),
                likely_lof=bool(row.likely_lof),
                oncogene_high_impact=bool(row.oncogene),
                tumour_suppressor_high_impact=bool(row.tumour_suppressor),
            )
            aggregate[key] = aggregate.get(key, TokenAnnotation()).merge(current)

    by_cell = {cell_id: {} for cell_id in ordered_cells}
    for (cell_id, gene), annotation in aggregate.items():
        by_cell[cell_id][gene] = annotation
    return [by_cell[cell_id] for cell_id in ordered_cells]


def fit_sensitivity_mutation_features(
    mutations_by_cell: Sequence[Iterable[str]],
    annotations_by_cell: Sequence[Mapping[str, TokenAnnotation]],
    gene_names: Sequence[str],
    train_idx: Sequence[int] | np.ndarray,
    *,
    top_n: int = 5_000,
    effective_k: int = MAX_TOKEN_SLOTS,
    token_strategy: str = "frequency",
) -> SensitivityMutationFeatures:
    """Fit the vocabulary on training cells and transform all cells.

    VEP/driver priorities are cell-specific observed inputs.  Population
    prevalence, vocabulary membership, and tie breaking are learned exclusively
    from ``train_idx``.
    """
    if top_n < 1:
        raise ValueError("top_n must be positive")
    if int(effective_k) not in ALLOWED_K:
        raise ValueError(f"effective_k must be one of {ALLOWED_K}")
    if token_strategy not in TOKEN_STRATEGIES:
        raise ValueError(f"unknown token strategy: {token_strategy}")
    if len(mutations_by_cell) != len(annotations_by_cell):
        raise ValueError("mutations and token annotations must align by cell")
    cells = [
        {str(gene) for gene in mutations if gene is not None and str(gene)}
        for mutations in mutations_by_cell
    ]
    train = _as_unique_indices(train_idx, len(cells))
    prevalence: Counter[str] = Counter()
    for cell_idx in train:
        prevalence.update(cells[int(cell_idx)])
    ranked = sorted(prevalence, key=lambda gene: (-prevalence[gene], gene))
    vocabulary = tuple(ranked[: int(top_n)])
    vocab_index = {gene: idx for idx, gene in enumerate(vocabulary)}

    matrix = np.zeros((len(cells), len(vocabulary)), dtype=np.float32)
    gene_index = {str(gene): idx for idx, gene in enumerate(gene_names)}
    token_order = tuple(gene for gene in vocabulary if gene in gene_index)
    prevalence_rank = {gene: rank for rank, gene in enumerate(token_order)}
    tokens = np.zeros((len(cells), MAX_TOKEN_SLOTS), dtype=np.int64)
    token_mask = np.zeros((len(cells), MAX_TOKEN_SLOTS), dtype=bool)
    counts = np.zeros(len(cells), dtype=np.int64)

    for cell_idx, mutations in enumerate(cells):
        columns = [vocab_index[gene] for gene in mutations if gene in vocab_index]
        if columns:
            matrix[cell_idx, columns] = 1.0
        candidates = [gene for gene in mutations if gene in prevalence_rank]
        missing = [
            gene for gene in candidates if gene not in annotations_by_cell[cell_idx]
        ]
        if missing:
            raise ValueError(
                "mutation annotations are missing for cell-local genes: "
                + ", ".join(sorted(missing)[:5])
            )

        def priority(gene: str) -> tuple[int, ...]:
            rank = prevalence_rank[gene]
            annotation = annotations_by_cell[cell_idx][gene]
            if token_strategy == "frequency":
                return (rank,)
            if token_strategy == "vep-impact":
                return (-int(annotation.impact_rank), rank)
            return (
                -int(annotation.driver),
                -int(annotation.likely_lof),
                -int(annotation.impact_rank),
                rank,
            )

        ordered = sorted(candidates, key=priority)
        counts[cell_idx] = len(ordered)
        selected = ordered[: int(effective_k)]
        if selected:
            rows = np.fromiter(
                (gene_index[gene] for gene in selected), dtype=np.int64
            )
            tokens[cell_idx, : rows.size] = rows
            token_mask[cell_idx, : rows.size] = True

    return SensitivityMutationFeatures(
        vocabulary=vocabulary,
        token_order=token_order,
        matrix=matrix,
        tokens=tokens,
        token_mask=token_mask,
        effective_k=int(effective_k),
        token_strategy=str(token_strategy),
        token_counts_before_truncation=counts,
    )


def _normalise_edges(edge_index: np.ndarray, n_genes: int) -> np.ndarray:
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim != 2:
        raise ValueError("edge_index must be two-dimensional")
    if edges.shape[0] != 2 and edges.shape[1] == 2:
        edges = edges.T
    if edges.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E) or (E, 2)")
    if edges.size and (int(edges.min()) < 0 or int(edges.max()) >= int(n_genes)):
        raise IndexError("edge_index refers to an unknown gene row")
    return edges


def fit_pmi_sensitivity(
    edge_index: np.ndarray,
    gene_names: Sequence[str],
    mutation_matrix: np.ndarray,
    mutation_vocabulary: Sequence[str],
    train_idx: Sequence[int] | np.ndarray,
    *,
    mode: str = "neutral-one",
    smoothing: float = 1e-3,
    clip: tuple[float, float] = (0.0, 3.0),
    chunk_size: int = 20_000,
) -> PmiFit:
    """Fit a prespecified PMI edge-weight variant on training cells only."""
    if mode not in PMI_MODES:
        raise ValueError(f"unknown PMI mode: {mode}")
    if not np.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("smoothing must be finite and positive")
    if (
        len(clip) != 2
        or not np.isfinite(clip).all()
        or float(clip[0]) < 0
        or float(clip[0]) >= float(clip[1])
    ):
        raise ValueError("clip must be finite, non-negative, and increasing")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    edges = _normalise_edges(edge_index, len(gene_names))
    matrix = np.asarray(mutation_matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != len(mutation_vocabulary):
        raise ValueError("mutation matrix and vocabulary do not align")
    train = _as_unique_indices(train_idx, matrix.shape[0])
    if not np.isfinite(matrix[train]).all():
        raise ValueError("training mutation matrix must be finite")
    if bool(((matrix[train] != 0.0) & (matrix[train] != 1.0)).any()):
        raise ValueError("training mutation matrix must be binary")

    vocab_index = {str(gene): idx for idx, gene in enumerate(mutation_vocabulary)}
    endpoint_a = np.array(
        [vocab_index.get(str(gene_names[row]), -1) for row in edges[0]],
        dtype=np.int64,
    )
    endpoint_b = np.array(
        [vocab_index.get(str(gene_names[row]), -1) for row in edges[1]],
        dtype=np.int64,
    )
    valid_mask = (endpoint_a >= 0) & (endpoint_b >= 0)
    valid_positions = np.flatnonzero(valid_mask)
    raw_pmi = np.full(edges.shape[1], np.nan, dtype=np.float64)
    train_matrix = matrix[train]
    prevalence = train_matrix.mean(axis=0, dtype=np.float64)
    for start in range(0, valid_positions.size, int(chunk_size)):
        positions = valid_positions[start : start + int(chunk_size)]
        a = endpoint_a[positions]
        b = endpoint_b[positions]
        joint = np.mean(
            train_matrix[:, a] * train_matrix[:, b], axis=0, dtype=np.float64
        )
        denominator = prevalence[a] * prevalence[b]
        raw_pmi[positions] = np.log(
            (joint + float(smoothing)) / (denominator + float(smoothing))
        )

    if mode == "unit":
        weights = np.ones(edges.shape[1], dtype=np.float32)
    else:
        unavailable = 1.0 if mode == "neutral-one" else 0.0
        weights = np.full(edges.shape[1], unavailable, dtype=np.float32)
        weights[valid_mask] = np.clip(
            raw_pmi[valid_mask], float(clip[0]), float(clip[1])
        ).astype(np.float32)
    valid_raw = raw_pmi[valid_mask]
    valid_weights = weights[valid_mask]
    raw_quantiles = (
        {
            str(q): float(np.quantile(valid_raw, q))
            for q in (0.0, 0.25, 0.5, 0.75, 1.0)
        }
        if valid_raw.size
        else {}
    )
    weight_quantiles = (
        {
            str(q): float(np.quantile(weights.astype(np.float64), q))
            for q in (0.0, 0.25, 0.5, 0.75, 1.0)
        }
        if weights.size
        else {}
    )
    nonunit = ~np.isclose(weights, 1.0, rtol=0.0, atol=1e-7)
    statistics: dict[str, Any] = {
        "mode": mode,
        "smoothing": float(smoothing),
        "clip": [float(clip[0]), float(clip[1])],
        "n_edge_records": int(edges.shape[1]),
        "n_valid_edge_records": int(valid_mask.sum()),
        "fraction_valid_edge_records": float(valid_mask.mean())
        if valid_mask.size
        else 0.0,
        "n_unavailable_edge_records": int((~valid_mask).sum()),
        "unavailable_edge_weight": (
            None if mode == "unit" else (1.0 if mode == "neutral-one" else 0.0)
        ),
        "n_valid_lower_clipped": int((valid_raw < float(clip[0])).sum()),
        "n_valid_upper_clipped": int((valid_raw > float(clip[1])).sum()),
        "n_valid_nonunit_after_clip": int(
            (~np.isclose(valid_weights, 1.0, rtol=0.0, atol=1e-7)).sum()
        ),
        "n_nonunit_edge_records": int(nonunit.sum()),
        "fraction_nonunit_edge_records": float(nonunit.mean())
        if nonunit.size
        else 0.0,
        "n_zero_weight_edge_records": int(
            np.isclose(weights, 0.0, rtol=0.0, atol=1e-7).sum()
        ),
        "raw_pmi_quantiles": raw_quantiles,
        "weight_quantiles": weight_quantiles,
        "weight_min": float(weights.min()) if weights.size else None,
        "weight_max": float(weights.max()) if weights.size else None,
        "weight_mean": float(weights.mean()) if weights.size else None,
    }
    return PmiFit(
        weights=np.ascontiguousarray(weights),
        valid_mask=np.ascontiguousarray(valid_mask),
        raw_pmi=np.ascontiguousarray(raw_pmi),
        statistics=statistics,
    )


def pmi_per_gene_summary(
    edge_index: np.ndarray,
    gene_names: Sequence[str],
    pmi: PmiFit,
) -> pd.DataFrame:
    """Summarize fitted PMI values for every model gene.

    Each undirected edge record contributes once to each distinct endpoint.  Thus
    a self-loop, if supplied, contributes once rather than twice.  Genes without
    an incident edge remain in the output with zero counts and missing numerical
    summaries.  The function only summarizes a fitted object; it does not use
    held-out labels or predictions.
    """
    edges = _normalise_edges(edge_index, len(gene_names))
    if pmi.weights.ndim != 1 or pmi.weights.size != edges.shape[1]:
        raise ValueError("PMI weights do not align with edge_index")
    if pmi.valid_mask.ndim != 1 or pmi.valid_mask.size != edges.shape[1]:
        raise ValueError("PMI validity mask does not align with edge_index")
    if pmi.raw_pmi.ndim != 1 or pmi.raw_pmi.size != edges.shape[1]:
        raise ValueError("raw PMI values do not align with edge_index")
    if not np.isfinite(pmi.weights).all():
        raise ValueError("PMI weights must be finite")
    if not np.array_equal(np.isfinite(pmi.raw_pmi), pmi.valid_mask):
        raise ValueError("finite raw PMI values must exactly match valid_mask")

    edge_ids = np.arange(edges.shape[1], dtype=np.int64)
    non_loop = edges[1] != edges[0]
    endpoint_gene = np.concatenate([edges[0], edges[1, non_loop]])
    endpoint_edge = np.concatenate([edge_ids, edge_ids[non_loop]])
    endpoint_valid = pmi.valid_mask[endpoint_edge]
    endpoint_raw = pmi.raw_pmi[endpoint_edge]
    endpoint_weight = pmi.weights[endpoint_edge].astype(np.float64, copy=False)
    endpoint_nonunit = ~np.isclose(
        endpoint_weight, 1.0, rtol=0.0, atol=1e-7
    )

    n_genes = len(gene_names)
    incident = np.bincount(endpoint_gene, minlength=n_genes).astype(np.int64)
    valid = np.bincount(
        endpoint_gene, weights=endpoint_valid.astype(np.int64), minlength=n_genes
    ).astype(np.int64)
    nonunit = np.bincount(
        endpoint_gene, weights=endpoint_nonunit.astype(np.int64), minlength=n_genes
    ).astype(np.int64)

    valid_frame = pd.DataFrame(
        {
            "gene_index": endpoint_gene[endpoint_valid],
            "raw_pmi": endpoint_raw[endpoint_valid],
        }
    )
    if valid_frame.empty:
        raw_summary = pd.DataFrame(
            columns=["raw_pmi_mean", "raw_pmi_median", "raw_pmi_q75", "raw_pmi_max"]
        )
    else:
        grouped_raw = valid_frame.groupby("gene_index", sort=True)["raw_pmi"]
        raw_summary = grouped_raw.agg(
            raw_pmi_mean="mean",
            raw_pmi_median="median",
            raw_pmi_max="max",
        )
        raw_summary["raw_pmi_q75"] = grouped_raw.quantile(0.75)

    weight_frame = pd.DataFrame(
        {"gene_index": endpoint_gene, "weight": endpoint_weight}
    )
    if weight_frame.empty:
        weight_summary = pd.DataFrame(
            columns=["weight_mean", "weight_median", "weight_max"]
        )
    else:
        weight_summary = weight_frame.groupby("gene_index", sort=True)["weight"].agg(
            weight_mean="mean",
            weight_median="median",
            weight_max="max",
        )

    summary = pd.DataFrame(
        {
            "gene_index": np.arange(n_genes, dtype=np.int64),
            "gene": [str(gene) for gene in gene_names],
            "n_incident_edge_records": incident,
            "n_valid_pmi_edge_records": valid,
            "n_nonunit_weight_edge_records": nonunit,
        }
    ).set_index("gene_index", drop=False)
    summary["fraction_valid_pmi_edges"] = np.divide(
        valid,
        incident,
        out=np.zeros(n_genes, dtype=np.float64),
        where=incident > 0,
    )
    summary["fraction_nonunit_weight_edges"] = np.divide(
        nonunit,
        incident,
        out=np.zeros(n_genes, dtype=np.float64),
        where=incident > 0,
    )
    summary = summary.join(raw_summary, how="left").join(weight_summary, how="left")
    columns = [
        "gene_index",
        "gene",
        "n_incident_edge_records",
        "n_valid_pmi_edge_records",
        "fraction_valid_pmi_edges",
        "n_nonunit_weight_edge_records",
        "fraction_nonunit_weight_edges",
        "raw_pmi_mean",
        "raw_pmi_median",
        "raw_pmi_q75",
        "raw_pmi_max",
        "weight_mean",
        "weight_median",
        "weight_max",
    ]
    return summary.loc[:, columns].reset_index(drop=True)


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def feature_manifest(
    features: SensitivityMutationFeatures,
    pmi: PmiFit,
) -> dict[str, Any]:
    counts = features.token_counts_before_truncation
    return {
        "schema": "DepPLM-k-token-pmi-preprocessing-v1",
        "effective_k": int(features.effective_k),
        "allocated_token_slots": int(features.tokens.shape[1]),
        "token_strategy": features.token_strategy,
        "vocabulary_size": len(features.vocabulary),
        "vocabulary_sha256": hashlib.sha256(
            b"\0".join(gene.encode("utf-8") for gene in features.vocabulary)
        ).hexdigest(),
        "matrix_sha256": sha256_array(features.matrix),
        "tokens_sha256": sha256_array(features.tokens),
        "token_mask_sha256": sha256_array(features.token_mask),
        "token_count_before_truncation": {
            "min": int(counts.min()) if counts.size else None,
            "median": float(np.median(counts)) if counts.size else None,
            "max": int(counts.max()) if counts.size else None,
            "n_truncated": int((counts > features.effective_k).sum()),
        },
        "pmi": {
            **dict(pmi.statistics),
            "weights_sha256": sha256_array(pmi.weights),
            "valid_mask_sha256": sha256_array(pmi.valid_mask),
            "raw_pmi_sha256": sha256_array(pmi.raw_pmi),
        },
    }
